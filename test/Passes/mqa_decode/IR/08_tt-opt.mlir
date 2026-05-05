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
    func.func @flash_decode__x8_y1y8__d0i1_d1i1_d2i0__f01__dim_x_level0_bc8_n_n_n__tile_b1__tile_n512__tile_s512(%arg0: memref<16x128x8192xf16>, %arg1: memref<16x8192x128xf16>, %arg2: memref<16x32x128xf16>, %arg3: memref<16x32x128xf16>) {
      %c65536 = arith.constant 65536 : index
      %c1048576 = arith.constant 1048576 : index
      %c4096 = arith.constant 4096 : index
      %c2 = arith.constant 2 : index
      %c7 = arith.constant 7 : index
      %c0 = arith.constant 0 : index
      %cst = arith.constant 0.000000e+00 : f16
      %cst_0 = arith.constant 1.000000e+00 : f16
      %cst_1 = arith.constant 0xFC00 : f16
      %c1 = arith.constant 1 : index
      %cst_2 = arith.constant 8.837890e-02 : f16
      %c512 = arith.constant 512 : index
      %c8 = arith.constant 8 : index
      scf.parallel (%arg4, %arg5) = (%c0, %c0) to (%c8, %c8) step (%c1, %c1) {
        scf.for %arg6 = %c0 to %c2 step %c1 {
          scf.for %arg7 = %c0 to %c2 step %c1 {
            %20 = arith.muli %arg6, %c8 overflow<nsw> : index
            %21 = arith.addi %arg4, %20 : index
            %22 = arith.muli %arg7, %c8 overflow<nsw> : index
            %23 = arith.addi %arg5, %22 : index
            %24 = loom.alloc [1, 32, 128] on @L1 : memref<1x32x128xf16>
            %25 = loom.semaphore_take %24 : memref<1x32x128xf16> -> memref<1x32x128xf16>
            %26 = arith.muli %21, %c4096 overflow<nsw> : index
            %reinterpret_cast = memref.reinterpret_cast %arg3 to offset: [%26], sizes: [1, 32, 128], strides: [4096, 128, 1] : memref<16x32x128xf16> to memref<1x32x128xf16, strided<[4096, 128, 1], offset: ?>>
            loom.copy %reinterpret_cast, %25 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [8, 1] region : (UL : [%c0, %arg4], LR : [%c7, %arg4]) : memref<1x32x128xf16, strided<[4096, 128, 1], offset: ?>> to memref<1x32x128xf16>
            %27 = arith.muli %23, %c512 : index
            %28 = loom.alloc [1, 32, 128] on @L1 : memref<1x32x128xf16>
            %29 = loom.semaphore_take %28 : memref<1x32x128xf16> -> memref<1x32x128xf16>
            %30 = loom.semaphore_take %28 : memref<1x32x128xf16> -> memref<1x32x128xf16>
            linalg.fill ins(%cst : f16) outs(%30 : memref<1x32x128xf16>)
            %31 = loom.alloc [1, 32, 1] on @L1 : memref<1x32x1xf16>
            %32 = loom.semaphore_take %31 : memref<1x32x1xf16> -> memref<1x32x1xf16>
            %33 = loom.semaphore_take %31 : memref<1x32x1xf16> -> memref<1x32x1xf16>
            %34 = loom.semaphore_take %31 : memref<1x32x1xf16> -> memref<1x32x1xf16>
            linalg.fill ins(%cst_0 : f16) outs(%34 : memref<1x32x1xf16>)
            %35 = loom.alloc [1, 32, 1] on @L1 : memref<1x32x1xf16>
            %36 = loom.semaphore_take %35 : memref<1x32x1xf16> -> memref<1x32x1xf16>
            linalg.fill ins(%cst_1 : f16) outs(%36 : memref<1x32x1xf16>)
            %37 = loom.alloc [1, 32, 128] on @L1 : memref<1x32x128xf16>
            %38 = loom.semaphore_take %37 : memref<1x32x128xf16> -> memref<1x32x128xf16>
            %39 = loom.alloc [1, 32, 1] on @L1 : memref<1x32x1xf16>
            %40 = loom.semaphore_take %39 : memref<1x32x1xf16> -> memref<1x32x1xf16>
            %41 = loom.alloc [1, 32, 1] on @L1 : memref<1x32x1xf16>
            %42 = loom.semaphore_take %41 : memref<1x32x1xf16> -> memref<1x32x1xf16>
            %43 = loom.alloc [1, 32, 1] on @L1 : memref<1x32x1xf16>
            %44 = loom.semaphore_take %43 : memref<1x32x1xf16> -> memref<1x32x1xf16>
            %45 = loom.alloc [1, 128, 512] on @L1 : memref<1x128x512xf16>
            %46 = loom.semaphore_take %45 : memref<1x128x512xf16> -> memref<1x128x512xf16>
            %47 = loom.alloc [1, 32, 512] on @L1 : memref<1x32x512xf16>
            %48 = loom.semaphore_take %47 : memref<1x32x512xf16> -> memref<1x32x512xf16>
            %49 = loom.alloc [1, 32, 32] on @L1 : memref<1x32x32xf16>
            %50 = loom.semaphore_take %49 : memref<1x32x32xf16> -> memref<1x32x32xf16>
            %51 = loom.semaphore_take %49 : memref<1x32x32xf16> -> memref<1x32x32xf16>
            %52 = loom.semaphore_take %49 : memref<1x32x32xf16> -> memref<1x32x32xf16>
            %53 = loom.alloc [1, 512, 128] on @L1 : memref<1x512x128xf16>
            %54 = loom.semaphore_take %53 : memref<1x512x128xf16> -> memref<1x512x128xf16>
            %55 = arith.muli %21, %c1048576 overflow<nsw> : index
            %56 = arith.addi %55, %27 : index
            %reinterpret_cast_3 = memref.reinterpret_cast %arg0 to offset: [%56], sizes: [1, 128, 512], strides: [1048576, 8192, 1] : memref<16x128x8192xf16> to memref<1x128x512xf16, strided<[1048576, 8192, 1], offset: ?>>
            loom.copy %reinterpret_cast_3, %46 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] region : (UL : [%arg5, %arg4], LR : [%arg5, %arg4]) : memref<1x128x512xf16, strided<[1048576, 8192, 1], offset: ?>> to memref<1x128x512xf16>
            loom.batch_matmul ins(%25, %46 : memref<1x32x128xf16>, memref<1x128x512xf16>) outs(%48 : memref<1x32x512xf16>)
            loom.semaphore_give %46 : memref<1x128x512xf16>
            loom.semaphore_give %25 : memref<1x32x128xf16>
            linalg.fill ins(%cst_1 : f16) outs(%40 : memref<1x32x1xf16>)
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, 0)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%48 : memref<1x32x512xf16>) outs(%40 : memref<1x32x1xf16>) {
            ^bb0(%in: f16, %out: f16):
              %79 = arith.maximumf %in, %out : f16
              linalg.yield %79 : f16
            }
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%36, %40 : memref<1x32x1xf16>, memref<1x32x1xf16>) outs(%40 : memref<1x32x1xf16>) {
            ^bb0(%in: f16, %in_5: f16, %out: f16):
              %79 = arith.mulf %in_5, %cst_2 : f16
              %80 = arith.cmpf ogt, %in, %79 : f16
              %81 = arith.select %80, %in, %79 : f16
              linalg.yield %81 : f16
            }
            %57 = loom.broadcast ins(%40 : memref<1x32x1xf16>) outs(%52 : memref<1x32x32xf16>) dim(2) -> memref<1x32x512xf16, strided<[?, ?, ?], offset: ?>>
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%48, %57 : memref<1x32x512xf16>, memref<1x32x512xf16, strided<[?, ?, ?], offset: ?>>) outs(%48 : memref<1x32x512xf16>) {
            ^bb0(%in: f16, %in_5: f16, %out: f16):
              %79 = arith.mulf %in, %cst_2 : f16
              %80 = arith.subf %79, %in_5 : f16
              %81 = math.exp %80 : f16
              linalg.yield %81 : f16
            }
            loom.semaphore_give %52 : memref<1x32x32xf16>
            linalg.fill ins(%cst : f16) outs(%42 : memref<1x32x1xf16>)
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, 0)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%48 : memref<1x32x512xf16>) outs(%42 : memref<1x32x1xf16>) {
            ^bb0(%in: f16, %out: f16):
              %79 = arith.addf %in, %out : f16
              linalg.yield %79 : f16
            }
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%36, %40 : memref<1x32x1xf16>, memref<1x32x1xf16>) outs(%44 : memref<1x32x1xf16>) {
            ^bb0(%in: f16, %in_5: f16, %out: f16):
              %79 = arith.subf %in, %in_5 : f16
              %80 = math.exp %79 : f16
              linalg.yield %80 : f16
            }
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%34, %44, %42 : memref<1x32x1xf16>, memref<1x32x1xf16>, memref<1x32x1xf16>) outs(%34 : memref<1x32x1xf16>) {
            ^bb0(%in: f16, %in_5: f16, %in_6: f16, %out: f16):
              %79 = arith.mulf %in, %in_5 : f16
              %80 = arith.addf %79, %in_6 : f16
              linalg.yield %80 : f16
            }
            loom.semaphore_give %42 : memref<1x32x1xf16>
            %58 = loom.broadcast ins(%44 : memref<1x32x1xf16>) outs(%51 : memref<1x32x32xf16>) dim(2) -> memref<1x32x128xf16, strided<[?, ?, ?], offset: ?>>
            loom.semaphore_give %44 : memref<1x32x1xf16>
            %59 = arith.muli %23, %c65536 : index
            %60 = arith.addi %55, %59 : index
            %reinterpret_cast_4 = memref.reinterpret_cast %arg1 to offset: [%60], sizes: [1, 512, 128], strides: [1048576, 128, 1] : memref<16x8192x128xf16> to memref<1x512x128xf16, strided<[1048576, 128, 1], offset: ?>>
            loom.copy %reinterpret_cast_4, %54 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] region : (UL : [%arg5, %arg4], LR : [%arg5, %arg4]) : memref<1x512x128xf16, strided<[1048576, 128, 1], offset: ?>> to memref<1x512x128xf16>
            loom.batch_matmul ins(%48, %54 : memref<1x32x512xf16>, memref<1x512x128xf16>) outs(%38 : memref<1x32x128xf16>)
            loom.semaphore_give %54 : memref<1x512x128xf16>
            loom.semaphore_give %48 : memref<1x32x512xf16>
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%38, %30, %58 : memref<1x32x128xf16>, memref<1x32x128xf16>, memref<1x32x128xf16, strided<[?, ?, ?], offset: ?>>) outs(%30 : memref<1x32x128xf16>) {
            ^bb0(%in: f16, %in_5: f16, %in_6: f16, %out: f16):
              %79 = arith.mulf %in_5, %in_6 : f16
              %80 = arith.addf %in, %79 : f16
              linalg.yield %80 : f16
            }
            loom.semaphore_give %51 : memref<1x32x32xf16>
            loom.semaphore_give %38 : memref<1x32x128xf16>
            linalg.copy ins(%40 : memref<1x32x1xf16>) outs(%36 : memref<1x32x1xf16>)
            loom.semaphore_give %40 : memref<1x32x1xf16>
            %61 = loom.alloc [1, 32, 1] on @L1 : memref<1x32x1xf16>
            %62 = loom.semaphore_take %61 : memref<1x32x1xf16> -> memref<1x32x1xf16>
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%34, %36 : memref<1x32x1xf16>, memref<1x32x1xf16>) outs(%62 : memref<1x32x1xf16>) {
            ^bb0(%in: f16, %in_5: f16, %out: f16):
              %79 = math.log %in : f16
              %80 = arith.addf %79, %in_5 : f16
              linalg.yield %80 : f16
            }
            loom.semaphore_give %36 : memref<1x32x1xf16>
            %63 = loom.broadcast ins(%34 : memref<1x32x1xf16>) outs(%50 : memref<1x32x32xf16>) dim(2) -> memref<1x32x128xf16, strided<[?, ?, ?], offset: ?>>
            loom.semaphore_give %34 : memref<1x32x1xf16>
            %64 = loom.alloc [1, 32, 128] on @L1 : memref<1x32x128xf16>
            %65 = loom.semaphore_take %64 : memref<1x32x128xf16> -> memref<1x32x128xf16>
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%30, %63 : memref<1x32x128xf16>, memref<1x32x128xf16, strided<[?, ?, ?], offset: ?>>) outs(%65 : memref<1x32x128xf16>) {
            ^bb0(%in: f16, %in_5: f16, %out: f16):
              %79 = arith.divf %in, %in_5 : f16
              linalg.yield %79 : f16
            }
            loom.semaphore_give %50 : memref<1x32x32xf16>
            loom.semaphore_give %30 : memref<1x32x128xf16>
            %66 = loom.alloc [16, 1, 32, 1] on @L1 : memref<16x1x32x1xf16>
            %67 = loom.semaphore_take %66 : memref<16x1x32x1xf16> -> memref<16x1x32x1xf16>
            loom.gather ins(%62 : memref<1x32x1xf16>) outs(%67 : memref<16x1x32x1xf16>) across(%arg5 : index) region : (UL : [%c0, %arg4], LR : [%c7, %arg4])
            loom.semaphore_give %62 : memref<1x32x1xf16>
            %68 = loom.alloc [16, 1, 32, 128] on @L1 : memref<16x1x32x128xf16>
            %69 = loom.semaphore_take %68 : memref<16x1x32x128xf16> -> memref<16x1x32x128xf16>
            loom.gather ins(%65 : memref<1x32x128xf16>) outs(%69 : memref<16x1x32x128xf16>) across(%arg5 : index) region : (UL : [%c0, %arg4], LR : [%c7, %arg4])
            loom.semaphore_give %65 : memref<1x32x128xf16>
            %70 = loom.alloc [1, 32, 128] on @L1 : memref<1x32x128xf16>
            %71 = loom.semaphore_take %70 : memref<1x32x128xf16> -> memref<1x32x128xf16>
            %72 = loom.alloc [16, 1, 32, 1] on @L1 : memref<16x1x32x1xf16>
            %73 = loom.semaphore_take %72 : memref<16x1x32x1xf16> -> memref<16x1x32x1xf16>
            %74 = loom.alloc [16, 1, 32, 128] on @L1 : memref<16x1x32x128xf16>
            %75 = loom.semaphore_take %74 : memref<16x1x32x128xf16> -> memref<16x1x32x128xf16>
            %76 = loom.alloc [16, 1, 32, 32] on @L1 : memref<16x1x32x32xf16>
            %77 = loom.semaphore_take %76 : memref<16x1x32x32xf16> -> memref<16x1x32x32xf16>
            %78 = arith.cmpi eq, %arg5, %c0 : index
            scf.if %78 {
              linalg.fill ins(%cst_1 : f16) outs(%33 : memref<1x32x1xf16>)
              linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1, d2, d3)>], iterator_types = ["reduction", "parallel", "parallel", "parallel"]} ins(%67 : memref<16x1x32x1xf16>) outs(%33 : memref<1x32x1xf16>) {
              ^bb0(%in: f16, %out: f16):
                %80 = arith.maximumf %in, %out : f16
                linalg.yield %80 : f16
              }
              linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%67, %33 : memref<16x1x32x1xf16>, memref<1x32x1xf16>) outs(%73 : memref<16x1x32x1xf16>) {
              ^bb0(%in: f16, %in_6: f16, %out: f16):
                %80 = arith.subf %in, %in_6 : f16
                %81 = math.exp %80 : f16
                linalg.yield %81 : f16
              }
              loom.semaphore_give %67 : memref<16x1x32x1xf16>
              loom.semaphore_give %33 : memref<1x32x1xf16>
              linalg.fill ins(%cst : f16) outs(%32 : memref<1x32x1xf16>)
              linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1, d2, d3)>], iterator_types = ["reduction", "parallel", "parallel", "parallel"]} ins(%73 : memref<16x1x32x1xf16>) outs(%32 : memref<1x32x1xf16>) {
              ^bb0(%in: f16, %out: f16):
                %80 = arith.addf %in, %out : f16
                linalg.yield %80 : f16
              }
              linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%73, %32 : memref<16x1x32x1xf16>, memref<1x32x1xf16>) outs(%73 : memref<16x1x32x1xf16>) {
              ^bb0(%in: f16, %in_6: f16, %out: f16):
                %80 = arith.divf %in, %in_6 : f16
                linalg.yield %80 : f16
              }
              loom.semaphore_give %32 : memref<1x32x1xf16>
              %79 = loom.broadcast ins(%73 : memref<16x1x32x1xf16>) outs(%77 : memref<16x1x32x32xf16>) dim(3) -> memref<16x1x32x128xf16, strided<[?, ?, ?, ?], offset: ?>>
              loom.semaphore_give %73 : memref<16x1x32x1xf16>
              linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%69, %79 : memref<16x1x32x128xf16>, memref<16x1x32x128xf16, strided<[?, ?, ?, ?], offset: ?>>) outs(%75 : memref<16x1x32x128xf16>) {
              ^bb0(%in: f16, %in_6: f16, %out: f16):
                %80 = arith.mulf %in, %in_6 : f16
                linalg.yield %80 : f16
              }
              loom.semaphore_give %77 : memref<16x1x32x32xf16>
              loom.semaphore_give %69 : memref<16x1x32x128xf16>
              linalg.fill ins(%cst : f16) outs(%71 : memref<1x32x128xf16>)
              linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1, d2, d3)>], iterator_types = ["reduction", "parallel", "parallel", "parallel"]} ins(%75 : memref<16x1x32x128xf16>) outs(%71 : memref<1x32x128xf16>) {
              ^bb0(%in: f16, %out: f16):
                %80 = arith.addf %in, %out : f16
                linalg.yield %80 : f16
              }
              loom.semaphore_give %75 : memref<16x1x32x128xf16>
              loom.semaphore_give %29 : memref<1x32x128xf16>
              %reinterpret_cast_5 = memref.reinterpret_cast %arg2 to offset: [%26], sizes: [1, 32, 128], strides: [4096, 128, 1] : memref<16x32x128xf16> to memref<1x32x128xf16, strided<[4096, 128, 1], offset: ?>>
              loom.copy %71, %reinterpret_cast_5 src_mem_space @mem_L1 dst_mem_space @mem_DRAM, broadcast : [1, 1] region : (UL : [%arg5, %arg4], LR : [%arg5, %arg4]) : memref<1x32x128xf16> to memref<1x32x128xf16, strided<[4096, 128, 1], offset: ?>>
              loom.semaphore_give %71 : memref<1x32x128xf16>
            }
          } {loom.block_sym = @tile_s, loom.iter_type = #loom.iter_type<temporal>}
        } {loom.block_sym = @tile_b, loom.iter_type = #loom.iter_type<temporal>}
        scf.reduce 
      }
      return
    }
  }
  module attributes {loom.pass_name = "Materialize", loom.tile_b = {is_reduction = false, upper_bound = 16 : index}, loom.tile_n = {is_reduction = false, upper_bound = 64 : index}, loom.tile_s = {is_reduction = false, upper_bound = 8192 : index}} {
    func.func @flash_decode__x8_y2y4__d0i1_d1i1_d2i0__f01__dim_x_level0_bc8_dim_y_level0_bc2_n_n_n__tile_b1__tile_n512__tile_s512(%arg0: memref<16x128x8192xf16>, %arg1: memref<16x8192x128xf16>, %arg2: memref<16x32x128xf16>, %arg3: memref<16x32x128xf16>) {
      %c65536 = arith.constant 65536 : index
      %c1048576 = arith.constant 1048576 : index
      %c4096 = arith.constant 4096 : index
      %c4 = arith.constant 4 : index
      %c7 = arith.constant 7 : index
      %c2 = arith.constant 2 : index
      %c0 = arith.constant 0 : index
      %cst = arith.constant 0.000000e+00 : f16
      %cst_0 = arith.constant 1.000000e+00 : f16
      %cst_1 = arith.constant 0xFC00 : f16
      %c1 = arith.constant 1 : index
      %cst_2 = arith.constant 8.837890e-02 : f16
      %c512 = arith.constant 512 : index
      %c8 = arith.constant 8 : index
      scf.parallel (%arg4, %arg5, %arg6) = (%c0, %c0, %c0) to (%c4, %c8, %c2) step (%c1, %c1, %c1) {
        scf.for %arg7 = %c0 to %c4 step %c1 {
          %20 = arith.muli %arg7, %c4 overflow<nsw> : index
          %21 = arith.addi %arg4, %20 : index
          %22 = arith.muli %arg5, %c8 overflow<nsw> : index
          %23 = arith.addi %22, %arg6 : index
          %24 = loom.alloc [1, 32, 128] on @L1 : memref<1x32x128xf16>
          %25 = loom.semaphore_take %24 : memref<1x32x128xf16> -> memref<1x32x128xf16>
          %26 = arith.muli %21, %c4096 overflow<nsw> : index
          %reinterpret_cast = memref.reinterpret_cast %arg3 to offset: [%26], sizes: [1, 32, 128], strides: [4096, 128, 1] : memref<16x32x128xf16> to memref<1x32x128xf16, strided<[4096, 128, 1], offset: ?>>
          %27 = arith.muli %arg4, %c2 : index
          %28 = arith.addi %27, %c1 : index
          loom.copy %reinterpret_cast, %25 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [8, 2] region : (UL : [%c0, %27], LR : [%c7, %28]) : memref<1x32x128xf16, strided<[4096, 128, 1], offset: ?>> to memref<1x32x128xf16>
          %29 = arith.muli %23, %c512 : index
          %30 = loom.alloc [1, 32, 128] on @L1 : memref<1x32x128xf16>
          %31 = loom.semaphore_take %30 : memref<1x32x128xf16> -> memref<1x32x128xf16>
          %32 = loom.semaphore_take %30 : memref<1x32x128xf16> -> memref<1x32x128xf16>
          linalg.fill ins(%cst : f16) outs(%32 : memref<1x32x128xf16>)
          %33 = loom.alloc [1, 32, 1] on @L1 : memref<1x32x1xf16>
          %34 = loom.semaphore_take %33 : memref<1x32x1xf16> -> memref<1x32x1xf16>
          %35 = loom.semaphore_take %33 : memref<1x32x1xf16> -> memref<1x32x1xf16>
          %36 = loom.semaphore_take %33 : memref<1x32x1xf16> -> memref<1x32x1xf16>
          linalg.fill ins(%cst_0 : f16) outs(%36 : memref<1x32x1xf16>)
          %37 = loom.alloc [1, 32, 1] on @L1 : memref<1x32x1xf16>
          %38 = loom.semaphore_take %37 : memref<1x32x1xf16> -> memref<1x32x1xf16>
          linalg.fill ins(%cst_1 : f16) outs(%38 : memref<1x32x1xf16>)
          %39 = loom.alloc [1, 32, 128] on @L1 : memref<1x32x128xf16>
          %40 = loom.semaphore_take %39 : memref<1x32x128xf16> -> memref<1x32x128xf16>
          %41 = loom.alloc [1, 32, 1] on @L1 : memref<1x32x1xf16>
          %42 = loom.semaphore_take %41 : memref<1x32x1xf16> -> memref<1x32x1xf16>
          %43 = loom.alloc [1, 32, 1] on @L1 : memref<1x32x1xf16>
          %44 = loom.semaphore_take %43 : memref<1x32x1xf16> -> memref<1x32x1xf16>
          %45 = loom.alloc [1, 32, 1] on @L1 : memref<1x32x1xf16>
          %46 = loom.semaphore_take %45 : memref<1x32x1xf16> -> memref<1x32x1xf16>
          %47 = loom.alloc [1, 128, 512] on @L1 : memref<1x128x512xf16>
          %48 = loom.semaphore_take %47 : memref<1x128x512xf16> -> memref<1x128x512xf16>
          %49 = loom.alloc [1, 32, 512] on @L1 : memref<1x32x512xf16>
          %50 = loom.semaphore_take %49 : memref<1x32x512xf16> -> memref<1x32x512xf16>
          %51 = loom.alloc [1, 32, 32] on @L1 : memref<1x32x32xf16>
          %52 = loom.semaphore_take %51 : memref<1x32x32xf16> -> memref<1x32x32xf16>
          %53 = loom.semaphore_take %51 : memref<1x32x32xf16> -> memref<1x32x32xf16>
          %54 = loom.semaphore_take %51 : memref<1x32x32xf16> -> memref<1x32x32xf16>
          %55 = loom.alloc [1, 512, 128] on @L1 : memref<1x512x128xf16>
          %56 = loom.semaphore_take %55 : memref<1x512x128xf16> -> memref<1x512x128xf16>
          %57 = arith.muli %21, %c1048576 overflow<nsw> : index
          %58 = arith.addi %57, %29 : index
          %reinterpret_cast_3 = memref.reinterpret_cast %arg0 to offset: [%58], sizes: [1, 128, 512], strides: [1048576, 8192, 1] : memref<16x128x8192xf16> to memref<1x128x512xf16, strided<[1048576, 8192, 1], offset: ?>>
          %59 = arith.addi %arg6, %27 : index
          loom.copy %reinterpret_cast_3, %48 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] region : (UL : [%arg5, %59], LR : [%arg5, %59]) : memref<1x128x512xf16, strided<[1048576, 8192, 1], offset: ?>> to memref<1x128x512xf16>
          loom.batch_matmul ins(%25, %48 : memref<1x32x128xf16>, memref<1x128x512xf16>) outs(%50 : memref<1x32x512xf16>)
          loom.semaphore_give %48 : memref<1x128x512xf16>
          loom.semaphore_give %25 : memref<1x32x128xf16>
          linalg.fill ins(%cst_1 : f16) outs(%42 : memref<1x32x1xf16>)
          linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, 0)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%50 : memref<1x32x512xf16>) outs(%42 : memref<1x32x1xf16>) {
          ^bb0(%in: f16, %out: f16):
            %82 = arith.maximumf %in, %out : f16
            linalg.yield %82 : f16
          }
          linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%38, %42 : memref<1x32x1xf16>, memref<1x32x1xf16>) outs(%42 : memref<1x32x1xf16>) {
          ^bb0(%in: f16, %in_5: f16, %out: f16):
            %82 = arith.mulf %in_5, %cst_2 : f16
            %83 = arith.cmpf ogt, %in, %82 : f16
            %84 = arith.select %83, %in, %82 : f16
            linalg.yield %84 : f16
          }
          %60 = loom.broadcast ins(%42 : memref<1x32x1xf16>) outs(%54 : memref<1x32x32xf16>) dim(2) -> memref<1x32x512xf16, strided<[?, ?, ?], offset: ?>>
          linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%50, %60 : memref<1x32x512xf16>, memref<1x32x512xf16, strided<[?, ?, ?], offset: ?>>) outs(%50 : memref<1x32x512xf16>) {
          ^bb0(%in: f16, %in_5: f16, %out: f16):
            %82 = arith.mulf %in, %cst_2 : f16
            %83 = arith.subf %82, %in_5 : f16
            %84 = math.exp %83 : f16
            linalg.yield %84 : f16
          }
          loom.semaphore_give %54 : memref<1x32x32xf16>
          linalg.fill ins(%cst : f16) outs(%44 : memref<1x32x1xf16>)
          linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, 0)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%50 : memref<1x32x512xf16>) outs(%44 : memref<1x32x1xf16>) {
          ^bb0(%in: f16, %out: f16):
            %82 = arith.addf %in, %out : f16
            linalg.yield %82 : f16
          }
          linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%38, %42 : memref<1x32x1xf16>, memref<1x32x1xf16>) outs(%46 : memref<1x32x1xf16>) {
          ^bb0(%in: f16, %in_5: f16, %out: f16):
            %82 = arith.subf %in, %in_5 : f16
            %83 = math.exp %82 : f16
            linalg.yield %83 : f16
          }
          linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%36, %46, %44 : memref<1x32x1xf16>, memref<1x32x1xf16>, memref<1x32x1xf16>) outs(%36 : memref<1x32x1xf16>) {
          ^bb0(%in: f16, %in_5: f16, %in_6: f16, %out: f16):
            %82 = arith.mulf %in, %in_5 : f16
            %83 = arith.addf %82, %in_6 : f16
            linalg.yield %83 : f16
          }
          loom.semaphore_give %44 : memref<1x32x1xf16>
          %61 = loom.broadcast ins(%46 : memref<1x32x1xf16>) outs(%53 : memref<1x32x32xf16>) dim(2) -> memref<1x32x128xf16, strided<[?, ?, ?], offset: ?>>
          loom.semaphore_give %46 : memref<1x32x1xf16>
          %62 = arith.muli %23, %c65536 : index
          %63 = arith.addi %57, %62 : index
          %reinterpret_cast_4 = memref.reinterpret_cast %arg1 to offset: [%63], sizes: [1, 512, 128], strides: [1048576, 128, 1] : memref<16x8192x128xf16> to memref<1x512x128xf16, strided<[1048576, 128, 1], offset: ?>>
          loom.copy %reinterpret_cast_4, %56 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] region : (UL : [%arg5, %59], LR : [%arg5, %59]) : memref<1x512x128xf16, strided<[1048576, 128, 1], offset: ?>> to memref<1x512x128xf16>
          loom.batch_matmul ins(%50, %56 : memref<1x32x512xf16>, memref<1x512x128xf16>) outs(%40 : memref<1x32x128xf16>)
          loom.semaphore_give %56 : memref<1x512x128xf16>
          loom.semaphore_give %50 : memref<1x32x512xf16>
          linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%40, %32, %61 : memref<1x32x128xf16>, memref<1x32x128xf16>, memref<1x32x128xf16, strided<[?, ?, ?], offset: ?>>) outs(%32 : memref<1x32x128xf16>) {
          ^bb0(%in: f16, %in_5: f16, %in_6: f16, %out: f16):
            %82 = arith.mulf %in_5, %in_6 : f16
            %83 = arith.addf %in, %82 : f16
            linalg.yield %83 : f16
          }
          loom.semaphore_give %53 : memref<1x32x32xf16>
          loom.semaphore_give %40 : memref<1x32x128xf16>
          linalg.copy ins(%42 : memref<1x32x1xf16>) outs(%38 : memref<1x32x1xf16>)
          loom.semaphore_give %42 : memref<1x32x1xf16>
          %64 = loom.alloc [1, 32, 1] on @L1 : memref<1x32x1xf16>
          %65 = loom.semaphore_take %64 : memref<1x32x1xf16> -> memref<1x32x1xf16>
          linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%36, %38 : memref<1x32x1xf16>, memref<1x32x1xf16>) outs(%65 : memref<1x32x1xf16>) {
          ^bb0(%in: f16, %in_5: f16, %out: f16):
            %82 = math.log %in : f16
            %83 = arith.addf %82, %in_5 : f16
            linalg.yield %83 : f16
          }
          loom.semaphore_give %38 : memref<1x32x1xf16>
          %66 = loom.broadcast ins(%36 : memref<1x32x1xf16>) outs(%52 : memref<1x32x32xf16>) dim(2) -> memref<1x32x128xf16, strided<[?, ?, ?], offset: ?>>
          loom.semaphore_give %36 : memref<1x32x1xf16>
          %67 = loom.alloc [1, 32, 128] on @L1 : memref<1x32x128xf16>
          %68 = loom.semaphore_take %67 : memref<1x32x128xf16> -> memref<1x32x128xf16>
          linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%32, %66 : memref<1x32x128xf16>, memref<1x32x128xf16, strided<[?, ?, ?], offset: ?>>) outs(%68 : memref<1x32x128xf16>) {
          ^bb0(%in: f16, %in_5: f16, %out: f16):
            %82 = arith.divf %in, %in_5 : f16
            linalg.yield %82 : f16
          }
          loom.semaphore_give %52 : memref<1x32x32xf16>
          loom.semaphore_give %32 : memref<1x32x128xf16>
          %69 = loom.alloc [16, 1, 32, 1] on @L1 : memref<16x1x32x1xf16>
          %70 = loom.semaphore_take %69 : memref<16x1x32x1xf16> -> memref<16x1x32x1xf16>
          loom.gather ins(%65 : memref<1x32x1xf16>) outs(%70 : memref<16x1x32x1xf16>) across(%arg5 : index) region : (UL : [%c0, %59], LR : [%c7, %59])
          loom.semaphore_give %65 : memref<1x32x1xf16>
          %71 = loom.alloc [16, 1, 32, 128] on @L1 : memref<16x1x32x128xf16>
          %72 = loom.semaphore_take %71 : memref<16x1x32x128xf16> -> memref<16x1x32x128xf16>
          loom.gather ins(%68 : memref<1x32x128xf16>) outs(%72 : memref<16x1x32x128xf16>) across(%arg5 : index) region : (UL : [%c0, %59], LR : [%c7, %59])
          loom.semaphore_give %68 : memref<1x32x128xf16>
          %73 = loom.alloc [1, 32, 128] on @L1 : memref<1x32x128xf16>
          %74 = loom.semaphore_take %73 : memref<1x32x128xf16> -> memref<1x32x128xf16>
          %75 = loom.alloc [16, 1, 32, 1] on @L1 : memref<16x1x32x1xf16>
          %76 = loom.semaphore_take %75 : memref<16x1x32x1xf16> -> memref<16x1x32x1xf16>
          %77 = loom.alloc [16, 1, 32, 128] on @L1 : memref<16x1x32x128xf16>
          %78 = loom.semaphore_take %77 : memref<16x1x32x128xf16> -> memref<16x1x32x128xf16>
          %79 = loom.alloc [16, 1, 32, 32] on @L1 : memref<16x1x32x32xf16>
          %80 = loom.semaphore_take %79 : memref<16x1x32x32xf16> -> memref<16x1x32x32xf16>
          %81 = arith.cmpi eq, %arg5, %c0 : index
          scf.if %81 {
            linalg.fill ins(%cst_1 : f16) outs(%35 : memref<1x32x1xf16>)
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1, d2, d3)>], iterator_types = ["reduction", "parallel", "parallel", "parallel"]} ins(%70 : memref<16x1x32x1xf16>) outs(%35 : memref<1x32x1xf16>) {
            ^bb0(%in: f16, %out: f16):
              %83 = arith.maximumf %in, %out : f16
              linalg.yield %83 : f16
            }
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%70, %35 : memref<16x1x32x1xf16>, memref<1x32x1xf16>) outs(%76 : memref<16x1x32x1xf16>) {
            ^bb0(%in: f16, %in_6: f16, %out: f16):
              %83 = arith.subf %in, %in_6 : f16
              %84 = math.exp %83 : f16
              linalg.yield %84 : f16
            }
            loom.semaphore_give %70 : memref<16x1x32x1xf16>
            loom.semaphore_give %35 : memref<1x32x1xf16>
            linalg.fill ins(%cst : f16) outs(%34 : memref<1x32x1xf16>)
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1, d2, d3)>], iterator_types = ["reduction", "parallel", "parallel", "parallel"]} ins(%76 : memref<16x1x32x1xf16>) outs(%34 : memref<1x32x1xf16>) {
            ^bb0(%in: f16, %out: f16):
              %83 = arith.addf %in, %out : f16
              linalg.yield %83 : f16
            }
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%76, %34 : memref<16x1x32x1xf16>, memref<1x32x1xf16>) outs(%76 : memref<16x1x32x1xf16>) {
            ^bb0(%in: f16, %in_6: f16, %out: f16):
              %83 = arith.divf %in, %in_6 : f16
              linalg.yield %83 : f16
            }
            loom.semaphore_give %34 : memref<1x32x1xf16>
            %82 = loom.broadcast ins(%76 : memref<16x1x32x1xf16>) outs(%80 : memref<16x1x32x32xf16>) dim(3) -> memref<16x1x32x128xf16, strided<[?, ?, ?, ?], offset: ?>>
            loom.semaphore_give %76 : memref<16x1x32x1xf16>
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%72, %82 : memref<16x1x32x128xf16>, memref<16x1x32x128xf16, strided<[?, ?, ?, ?], offset: ?>>) outs(%78 : memref<16x1x32x128xf16>) {
            ^bb0(%in: f16, %in_6: f16, %out: f16):
              %83 = arith.mulf %in, %in_6 : f16
              linalg.yield %83 : f16
            }
            loom.semaphore_give %80 : memref<16x1x32x32xf16>
            loom.semaphore_give %72 : memref<16x1x32x128xf16>
            linalg.fill ins(%cst : f16) outs(%74 : memref<1x32x128xf16>)
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1, d2, d3)>], iterator_types = ["reduction", "parallel", "parallel", "parallel"]} ins(%78 : memref<16x1x32x128xf16>) outs(%74 : memref<1x32x128xf16>) {
            ^bb0(%in: f16, %out: f16):
              %83 = arith.addf %in, %out : f16
              linalg.yield %83 : f16
            }
            loom.semaphore_give %78 : memref<16x1x32x128xf16>
            loom.semaphore_give %31 : memref<1x32x128xf16>
            %reinterpret_cast_5 = memref.reinterpret_cast %arg2 to offset: [%26], sizes: [1, 32, 128], strides: [4096, 128, 1] : memref<16x32x128xf16> to memref<1x32x128xf16, strided<[4096, 128, 1], offset: ?>>
            loom.copy %74, %reinterpret_cast_5 src_mem_space @mem_L1 dst_mem_space @mem_DRAM, broadcast : [1, 1] region : (UL : [%arg5, %59], LR : [%arg5, %59]) : memref<1x32x128xf16> to memref<1x32x128xf16, strided<[4096, 128, 1], offset: ?>>
            loom.semaphore_give %74 : memref<1x32x128xf16>
          }
        } {loom.block_sym = @tile_b, loom.iter_type = #loom.iter_type<temporal>}
        scf.reduce 
      } {loom.iter_types = [#loom.iter_type<spatial>, #loom.iter_type<spatial>, #loom.iter_type<spatial>], loom.logical_levels = [1, 0, 0], loom.physical_dims = [@dim_y, @dim_x, @dim_y]}
      return
    }
  }
  module attributes {loom.pass_name = "Materialize", loom.tile_b = {is_reduction = false, upper_bound = 16 : index}, loom.tile_n = {is_reduction = false, upper_bound = 64 : index}, loom.tile_s = {is_reduction = false, upper_bound = 8192 : index}} {
    func.func @flash_decode__x8_y4y2__d0i1_d1i1_d2i0__f01__dim_x_level0_bc8_dim_y_level0_bc4_n_n_n__tile_b1__tile_n512__tile_s512(%arg0: memref<16x128x8192xf16>, %arg1: memref<16x8192x128xf16>, %arg2: memref<16x32x128xf16>, %arg3: memref<16x32x128xf16>) {
      %c65536 = arith.constant 65536 : index
      %c1048576 = arith.constant 1048576 : index
      %c4096 = arith.constant 4096 : index
      %c8 = arith.constant 8 : index
      %c3 = arith.constant 3 : index
      %c7 = arith.constant 7 : index
      %c4 = arith.constant 4 : index
      %c0 = arith.constant 0 : index
      %cst = arith.constant 0.000000e+00 : f16
      %cst_0 = arith.constant 1.000000e+00 : f16
      %cst_1 = arith.constant 0xFC00 : f16
      %c1 = arith.constant 1 : index
      %cst_2 = arith.constant 8.837890e-02 : f16
      %c512 = arith.constant 512 : index
      %c2 = arith.constant 2 : index
      scf.parallel (%arg4, %arg5, %arg6) = (%c0, %c0, %c0) to (%c2, %c8, %c4) step (%c1, %c1, %c1) {
        scf.for %arg7 = %c0 to %c8 step %c1 {
          %20 = arith.muli %arg7, %c2 overflow<nsw> : index
          %21 = arith.addi %arg4, %20 : index
          %22 = arith.muli %arg5, %c8 overflow<nsw> : index
          %23 = arith.addi %22, %arg6 : index
          %24 = loom.alloc [1, 32, 128] on @L1 : memref<1x32x128xf16>
          %25 = loom.semaphore_take %24 : memref<1x32x128xf16> -> memref<1x32x128xf16>
          %26 = arith.muli %21, %c4096 overflow<nsw> : index
          %reinterpret_cast = memref.reinterpret_cast %arg3 to offset: [%26], sizes: [1, 32, 128], strides: [4096, 128, 1] : memref<16x32x128xf16> to memref<1x32x128xf16, strided<[4096, 128, 1], offset: ?>>
          %27 = arith.muli %arg4, %c4 : index
          %28 = arith.addi %27, %c3 : index
          loom.copy %reinterpret_cast, %25 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [8, 4] region : (UL : [%c0, %27], LR : [%c7, %28]) : memref<1x32x128xf16, strided<[4096, 128, 1], offset: ?>> to memref<1x32x128xf16>
          %29 = arith.muli %23, %c512 : index
          %30 = loom.alloc [1, 32, 128] on @L1 : memref<1x32x128xf16>
          %31 = loom.semaphore_take %30 : memref<1x32x128xf16> -> memref<1x32x128xf16>
          %32 = loom.semaphore_take %30 : memref<1x32x128xf16> -> memref<1x32x128xf16>
          linalg.fill ins(%cst : f16) outs(%32 : memref<1x32x128xf16>)
          %33 = loom.alloc [1, 32, 1] on @L1 : memref<1x32x1xf16>
          %34 = loom.semaphore_take %33 : memref<1x32x1xf16> -> memref<1x32x1xf16>
          %35 = loom.semaphore_take %33 : memref<1x32x1xf16> -> memref<1x32x1xf16>
          %36 = loom.semaphore_take %33 : memref<1x32x1xf16> -> memref<1x32x1xf16>
          linalg.fill ins(%cst_0 : f16) outs(%36 : memref<1x32x1xf16>)
          %37 = loom.alloc [1, 32, 1] on @L1 : memref<1x32x1xf16>
          %38 = loom.semaphore_take %37 : memref<1x32x1xf16> -> memref<1x32x1xf16>
          linalg.fill ins(%cst_1 : f16) outs(%38 : memref<1x32x1xf16>)
          %39 = loom.alloc [1, 32, 128] on @L1 : memref<1x32x128xf16>
          %40 = loom.semaphore_take %39 : memref<1x32x128xf16> -> memref<1x32x128xf16>
          %41 = loom.alloc [1, 32, 1] on @L1 : memref<1x32x1xf16>
          %42 = loom.semaphore_take %41 : memref<1x32x1xf16> -> memref<1x32x1xf16>
          %43 = loom.alloc [1, 32, 1] on @L1 : memref<1x32x1xf16>
          %44 = loom.semaphore_take %43 : memref<1x32x1xf16> -> memref<1x32x1xf16>
          %45 = loom.alloc [1, 32, 1] on @L1 : memref<1x32x1xf16>
          %46 = loom.semaphore_take %45 : memref<1x32x1xf16> -> memref<1x32x1xf16>
          %47 = loom.alloc [1, 128, 512] on @L1 : memref<1x128x512xf16>
          %48 = loom.semaphore_take %47 : memref<1x128x512xf16> -> memref<1x128x512xf16>
          %49 = loom.alloc [1, 32, 512] on @L1 : memref<1x32x512xf16>
          %50 = loom.semaphore_take %49 : memref<1x32x512xf16> -> memref<1x32x512xf16>
          %51 = loom.alloc [1, 32, 32] on @L1 : memref<1x32x32xf16>
          %52 = loom.semaphore_take %51 : memref<1x32x32xf16> -> memref<1x32x32xf16>
          %53 = loom.semaphore_take %51 : memref<1x32x32xf16> -> memref<1x32x32xf16>
          %54 = loom.semaphore_take %51 : memref<1x32x32xf16> -> memref<1x32x32xf16>
          %55 = loom.alloc [1, 512, 128] on @L1 : memref<1x512x128xf16>
          %56 = loom.semaphore_take %55 : memref<1x512x128xf16> -> memref<1x512x128xf16>
          %57 = arith.muli %21, %c1048576 overflow<nsw> : index
          %58 = arith.addi %57, %29 : index
          %reinterpret_cast_3 = memref.reinterpret_cast %arg0 to offset: [%58], sizes: [1, 128, 512], strides: [1048576, 8192, 1] : memref<16x128x8192xf16> to memref<1x128x512xf16, strided<[1048576, 8192, 1], offset: ?>>
          %59 = arith.addi %arg6, %27 : index
          loom.copy %reinterpret_cast_3, %48 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] region : (UL : [%arg5, %59], LR : [%arg5, %59]) : memref<1x128x512xf16, strided<[1048576, 8192, 1], offset: ?>> to memref<1x128x512xf16>
          loom.batch_matmul ins(%25, %48 : memref<1x32x128xf16>, memref<1x128x512xf16>) outs(%50 : memref<1x32x512xf16>)
          loom.semaphore_give %48 : memref<1x128x512xf16>
          loom.semaphore_give %25 : memref<1x32x128xf16>
          linalg.fill ins(%cst_1 : f16) outs(%42 : memref<1x32x1xf16>)
          linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, 0)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%50 : memref<1x32x512xf16>) outs(%42 : memref<1x32x1xf16>) {
          ^bb0(%in: f16, %out: f16):
            %82 = arith.maximumf %in, %out : f16
            linalg.yield %82 : f16
          }
          linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%38, %42 : memref<1x32x1xf16>, memref<1x32x1xf16>) outs(%42 : memref<1x32x1xf16>) {
          ^bb0(%in: f16, %in_5: f16, %out: f16):
            %82 = arith.mulf %in_5, %cst_2 : f16
            %83 = arith.cmpf ogt, %in, %82 : f16
            %84 = arith.select %83, %in, %82 : f16
            linalg.yield %84 : f16
          }
          %60 = loom.broadcast ins(%42 : memref<1x32x1xf16>) outs(%54 : memref<1x32x32xf16>) dim(2) -> memref<1x32x512xf16, strided<[?, ?, ?], offset: ?>>
          linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%50, %60 : memref<1x32x512xf16>, memref<1x32x512xf16, strided<[?, ?, ?], offset: ?>>) outs(%50 : memref<1x32x512xf16>) {
          ^bb0(%in: f16, %in_5: f16, %out: f16):
            %82 = arith.mulf %in, %cst_2 : f16
            %83 = arith.subf %82, %in_5 : f16
            %84 = math.exp %83 : f16
            linalg.yield %84 : f16
          }
          loom.semaphore_give %54 : memref<1x32x32xf16>
          linalg.fill ins(%cst : f16) outs(%44 : memref<1x32x1xf16>)
          linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, 0)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%50 : memref<1x32x512xf16>) outs(%44 : memref<1x32x1xf16>) {
          ^bb0(%in: f16, %out: f16):
            %82 = arith.addf %in, %out : f16
            linalg.yield %82 : f16
          }
          linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%38, %42 : memref<1x32x1xf16>, memref<1x32x1xf16>) outs(%46 : memref<1x32x1xf16>) {
          ^bb0(%in: f16, %in_5: f16, %out: f16):
            %82 = arith.subf %in, %in_5 : f16
            %83 = math.exp %82 : f16
            linalg.yield %83 : f16
          }
          linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%36, %46, %44 : memref<1x32x1xf16>, memref<1x32x1xf16>, memref<1x32x1xf16>) outs(%36 : memref<1x32x1xf16>) {
          ^bb0(%in: f16, %in_5: f16, %in_6: f16, %out: f16):
            %82 = arith.mulf %in, %in_5 : f16
            %83 = arith.addf %82, %in_6 : f16
            linalg.yield %83 : f16
          }
          loom.semaphore_give %44 : memref<1x32x1xf16>
          %61 = loom.broadcast ins(%46 : memref<1x32x1xf16>) outs(%53 : memref<1x32x32xf16>) dim(2) -> memref<1x32x128xf16, strided<[?, ?, ?], offset: ?>>
          loom.semaphore_give %46 : memref<1x32x1xf16>
          %62 = arith.muli %23, %c65536 : index
          %63 = arith.addi %57, %62 : index
          %reinterpret_cast_4 = memref.reinterpret_cast %arg1 to offset: [%63], sizes: [1, 512, 128], strides: [1048576, 128, 1] : memref<16x8192x128xf16> to memref<1x512x128xf16, strided<[1048576, 128, 1], offset: ?>>
          loom.copy %reinterpret_cast_4, %56 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] region : (UL : [%arg5, %59], LR : [%arg5, %59]) : memref<1x512x128xf16, strided<[1048576, 128, 1], offset: ?>> to memref<1x512x128xf16>
          loom.batch_matmul ins(%50, %56 : memref<1x32x512xf16>, memref<1x512x128xf16>) outs(%40 : memref<1x32x128xf16>)
          loom.semaphore_give %56 : memref<1x512x128xf16>
          loom.semaphore_give %50 : memref<1x32x512xf16>
          linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%40, %32, %61 : memref<1x32x128xf16>, memref<1x32x128xf16>, memref<1x32x128xf16, strided<[?, ?, ?], offset: ?>>) outs(%32 : memref<1x32x128xf16>) {
          ^bb0(%in: f16, %in_5: f16, %in_6: f16, %out: f16):
            %82 = arith.mulf %in_5, %in_6 : f16
            %83 = arith.addf %in, %82 : f16
            linalg.yield %83 : f16
          }
          loom.semaphore_give %53 : memref<1x32x32xf16>
          loom.semaphore_give %40 : memref<1x32x128xf16>
          linalg.copy ins(%42 : memref<1x32x1xf16>) outs(%38 : memref<1x32x1xf16>)
          loom.semaphore_give %42 : memref<1x32x1xf16>
          %64 = loom.alloc [1, 32, 1] on @L1 : memref<1x32x1xf16>
          %65 = loom.semaphore_take %64 : memref<1x32x1xf16> -> memref<1x32x1xf16>
          linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%36, %38 : memref<1x32x1xf16>, memref<1x32x1xf16>) outs(%65 : memref<1x32x1xf16>) {
          ^bb0(%in: f16, %in_5: f16, %out: f16):
            %82 = math.log %in : f16
            %83 = arith.addf %82, %in_5 : f16
            linalg.yield %83 : f16
          }
          loom.semaphore_give %38 : memref<1x32x1xf16>
          %66 = loom.broadcast ins(%36 : memref<1x32x1xf16>) outs(%52 : memref<1x32x32xf16>) dim(2) -> memref<1x32x128xf16, strided<[?, ?, ?], offset: ?>>
          loom.semaphore_give %36 : memref<1x32x1xf16>
          %67 = loom.alloc [1, 32, 128] on @L1 : memref<1x32x128xf16>
          %68 = loom.semaphore_take %67 : memref<1x32x128xf16> -> memref<1x32x128xf16>
          linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%32, %66 : memref<1x32x128xf16>, memref<1x32x128xf16, strided<[?, ?, ?], offset: ?>>) outs(%68 : memref<1x32x128xf16>) {
          ^bb0(%in: f16, %in_5: f16, %out: f16):
            %82 = arith.divf %in, %in_5 : f16
            linalg.yield %82 : f16
          }
          loom.semaphore_give %52 : memref<1x32x32xf16>
          loom.semaphore_give %32 : memref<1x32x128xf16>
          %69 = loom.alloc [16, 1, 32, 1] on @L1 : memref<16x1x32x1xf16>
          %70 = loom.semaphore_take %69 : memref<16x1x32x1xf16> -> memref<16x1x32x1xf16>
          loom.gather ins(%65 : memref<1x32x1xf16>) outs(%70 : memref<16x1x32x1xf16>) across(%arg5 : index) region : (UL : [%c0, %59], LR : [%c7, %59])
          loom.semaphore_give %65 : memref<1x32x1xf16>
          %71 = loom.alloc [16, 1, 32, 128] on @L1 : memref<16x1x32x128xf16>
          %72 = loom.semaphore_take %71 : memref<16x1x32x128xf16> -> memref<16x1x32x128xf16>
          loom.gather ins(%68 : memref<1x32x128xf16>) outs(%72 : memref<16x1x32x128xf16>) across(%arg5 : index) region : (UL : [%c0, %59], LR : [%c7, %59])
          loom.semaphore_give %68 : memref<1x32x128xf16>
          %73 = loom.alloc [1, 32, 128] on @L1 : memref<1x32x128xf16>
          %74 = loom.semaphore_take %73 : memref<1x32x128xf16> -> memref<1x32x128xf16>
          %75 = loom.alloc [16, 1, 32, 1] on @L1 : memref<16x1x32x1xf16>
          %76 = loom.semaphore_take %75 : memref<16x1x32x1xf16> -> memref<16x1x32x1xf16>
          %77 = loom.alloc [16, 1, 32, 128] on @L1 : memref<16x1x32x128xf16>
          %78 = loom.semaphore_take %77 : memref<16x1x32x128xf16> -> memref<16x1x32x128xf16>
          %79 = loom.alloc [16, 1, 32, 32] on @L1 : memref<16x1x32x32xf16>
          %80 = loom.semaphore_take %79 : memref<16x1x32x32xf16> -> memref<16x1x32x32xf16>
          %81 = arith.cmpi eq, %arg5, %c0 : index
          scf.if %81 {
            linalg.fill ins(%cst_1 : f16) outs(%35 : memref<1x32x1xf16>)
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1, d2, d3)>], iterator_types = ["reduction", "parallel", "parallel", "parallel"]} ins(%70 : memref<16x1x32x1xf16>) outs(%35 : memref<1x32x1xf16>) {
            ^bb0(%in: f16, %out: f16):
              %83 = arith.maximumf %in, %out : f16
              linalg.yield %83 : f16
            }
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%70, %35 : memref<16x1x32x1xf16>, memref<1x32x1xf16>) outs(%76 : memref<16x1x32x1xf16>) {
            ^bb0(%in: f16, %in_6: f16, %out: f16):
              %83 = arith.subf %in, %in_6 : f16
              %84 = math.exp %83 : f16
              linalg.yield %84 : f16
            }
            loom.semaphore_give %70 : memref<16x1x32x1xf16>
            loom.semaphore_give %35 : memref<1x32x1xf16>
            linalg.fill ins(%cst : f16) outs(%34 : memref<1x32x1xf16>)
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1, d2, d3)>], iterator_types = ["reduction", "parallel", "parallel", "parallel"]} ins(%76 : memref<16x1x32x1xf16>) outs(%34 : memref<1x32x1xf16>) {
            ^bb0(%in: f16, %out: f16):
              %83 = arith.addf %in, %out : f16
              linalg.yield %83 : f16
            }
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%76, %34 : memref<16x1x32x1xf16>, memref<1x32x1xf16>) outs(%76 : memref<16x1x32x1xf16>) {
            ^bb0(%in: f16, %in_6: f16, %out: f16):
              %83 = arith.divf %in, %in_6 : f16
              linalg.yield %83 : f16
            }
            loom.semaphore_give %34 : memref<1x32x1xf16>
            %82 = loom.broadcast ins(%76 : memref<16x1x32x1xf16>) outs(%80 : memref<16x1x32x32xf16>) dim(3) -> memref<16x1x32x128xf16, strided<[?, ?, ?, ?], offset: ?>>
            loom.semaphore_give %76 : memref<16x1x32x1xf16>
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%72, %82 : memref<16x1x32x128xf16>, memref<16x1x32x128xf16, strided<[?, ?, ?, ?], offset: ?>>) outs(%78 : memref<16x1x32x128xf16>) {
            ^bb0(%in: f16, %in_6: f16, %out: f16):
              %83 = arith.mulf %in, %in_6 : f16
              linalg.yield %83 : f16
            }
            loom.semaphore_give %80 : memref<16x1x32x32xf16>
            loom.semaphore_give %72 : memref<16x1x32x128xf16>
            linalg.fill ins(%cst : f16) outs(%74 : memref<1x32x128xf16>)
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1, d2, d3)>], iterator_types = ["reduction", "parallel", "parallel", "parallel"]} ins(%78 : memref<16x1x32x128xf16>) outs(%74 : memref<1x32x128xf16>) {
            ^bb0(%in: f16, %out: f16):
              %83 = arith.addf %in, %out : f16
              linalg.yield %83 : f16
            }
            loom.semaphore_give %78 : memref<16x1x32x128xf16>
            loom.semaphore_give %31 : memref<1x32x128xf16>
            %reinterpret_cast_5 = memref.reinterpret_cast %arg2 to offset: [%26], sizes: [1, 32, 128], strides: [4096, 128, 1] : memref<16x32x128xf16> to memref<1x32x128xf16, strided<[4096, 128, 1], offset: ?>>
            loom.copy %74, %reinterpret_cast_5 src_mem_space @mem_L1 dst_mem_space @mem_DRAM, broadcast : [1, 1] region : (UL : [%arg5, %59], LR : [%arg5, %59]) : memref<1x32x128xf16> to memref<1x32x128xf16, strided<[4096, 128, 1], offset: ?>>
            loom.semaphore_give %74 : memref<1x32x128xf16>
          }
        } {loom.block_sym = @tile_b, loom.iter_type = #loom.iter_type<temporal>}
        scf.reduce 
      } {loom.iter_types = [#loom.iter_type<spatial>, #loom.iter_type<spatial>, #loom.iter_type<spatial>], loom.logical_levels = [1, 0, 0], loom.physical_dims = [@dim_y, @dim_x, @dim_y]}
      return
    }
  }
  module attributes {loom.pass_name = "Materialize", loom.tile_b = {is_reduction = false, upper_bound = 16 : index}, loom.tile_n = {is_reduction = false, upper_bound = 64 : index}, loom.tile_s = {is_reduction = false, upper_bound = 8192 : index}} {
    func.func @flash_decode__x8_y8y1__d0i1_d1i1_d2i0__f01__dim_x_level0_bc8_dim_y_level1_bc8_n_n_n__tile_b1__tile_n512__tile_s512(%arg0: memref<16x128x8192xf16>, %arg1: memref<16x8192x128xf16>, %arg2: memref<16x32x128xf16>, %arg3: memref<16x32x128xf16>) {
      %c65536 = arith.constant 65536 : index
      %c1048576 = arith.constant 1048576 : index
      %c4096 = arith.constant 4096 : index
      %c7 = arith.constant 7 : index
      %c8 = arith.constant 8 : index
      %c0 = arith.constant 0 : index
      %cst = arith.constant 0.000000e+00 : f16
      %cst_0 = arith.constant 1.000000e+00 : f16
      %cst_1 = arith.constant 0xFC00 : f16
      %c1 = arith.constant 1 : index
      %cst_2 = arith.constant 8.837890e-02 : f16
      %c16 = arith.constant 16 : index
      %c512 = arith.constant 512 : index
      scf.parallel (%arg4, %arg5) = (%c0, %c0) to (%c8, %c8) step (%c1, %c1) {
        scf.for %arg6 = %c0 to %c16 step %c1 {
          %20 = arith.muli %arg4, %c8 overflow<nsw> : index
          %21 = arith.addi %20, %arg5 : index
          %22 = loom.alloc [1, 32, 128] on @L1 : memref<1x32x128xf16>
          %23 = loom.semaphore_take %22 : memref<1x32x128xf16> -> memref<1x32x128xf16>
          %24 = arith.muli %arg6, %c4096 overflow<nsw> : index
          %reinterpret_cast = memref.reinterpret_cast %arg3 to offset: [%24], sizes: [1, 32, 128], strides: [4096, 128, 1] : memref<16x32x128xf16> to memref<1x32x128xf16, strided<[4096, 128, 1], offset: ?>>
          loom.copy %reinterpret_cast, %23 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [8, 8] region : (UL : [%c0, %c0], LR : [%c7, %c7]) : memref<1x32x128xf16, strided<[4096, 128, 1], offset: ?>> to memref<1x32x128xf16>
          %25 = arith.muli %21, %c512 : index
          %26 = loom.alloc [1, 32, 128] on @L1 : memref<1x32x128xf16>
          %27 = loom.semaphore_take %26 : memref<1x32x128xf16> -> memref<1x32x128xf16>
          %28 = loom.semaphore_take %26 : memref<1x32x128xf16> -> memref<1x32x128xf16>
          linalg.fill ins(%cst : f16) outs(%28 : memref<1x32x128xf16>)
          %29 = loom.alloc [1, 32, 1] on @L1 : memref<1x32x1xf16>
          %30 = loom.semaphore_take %29 : memref<1x32x1xf16> -> memref<1x32x1xf16>
          %31 = loom.semaphore_take %29 : memref<1x32x1xf16> -> memref<1x32x1xf16>
          %32 = loom.semaphore_take %29 : memref<1x32x1xf16> -> memref<1x32x1xf16>
          linalg.fill ins(%cst_0 : f16) outs(%32 : memref<1x32x1xf16>)
          %33 = loom.alloc [1, 32, 1] on @L1 : memref<1x32x1xf16>
          %34 = loom.semaphore_take %33 : memref<1x32x1xf16> -> memref<1x32x1xf16>
          linalg.fill ins(%cst_1 : f16) outs(%34 : memref<1x32x1xf16>)
          %35 = loom.alloc [1, 32, 128] on @L1 : memref<1x32x128xf16>
          %36 = loom.semaphore_take %35 : memref<1x32x128xf16> -> memref<1x32x128xf16>
          %37 = loom.alloc [1, 32, 1] on @L1 : memref<1x32x1xf16>
          %38 = loom.semaphore_take %37 : memref<1x32x1xf16> -> memref<1x32x1xf16>
          %39 = loom.alloc [1, 32, 1] on @L1 : memref<1x32x1xf16>
          %40 = loom.semaphore_take %39 : memref<1x32x1xf16> -> memref<1x32x1xf16>
          %41 = loom.alloc [1, 32, 1] on @L1 : memref<1x32x1xf16>
          %42 = loom.semaphore_take %41 : memref<1x32x1xf16> -> memref<1x32x1xf16>
          %43 = loom.alloc [1, 128, 512] on @L1 : memref<1x128x512xf16>
          %44 = loom.semaphore_take %43 : memref<1x128x512xf16> -> memref<1x128x512xf16>
          %45 = loom.alloc [1, 32, 512] on @L1 : memref<1x32x512xf16>
          %46 = loom.semaphore_take %45 : memref<1x32x512xf16> -> memref<1x32x512xf16>
          %47 = loom.alloc [1, 32, 32] on @L1 : memref<1x32x32xf16>
          %48 = loom.semaphore_take %47 : memref<1x32x32xf16> -> memref<1x32x32xf16>
          %49 = loom.semaphore_take %47 : memref<1x32x32xf16> -> memref<1x32x32xf16>
          %50 = loom.semaphore_take %47 : memref<1x32x32xf16> -> memref<1x32x32xf16>
          %51 = loom.alloc [1, 512, 128] on @L1 : memref<1x512x128xf16>
          %52 = loom.semaphore_take %51 : memref<1x512x128xf16> -> memref<1x512x128xf16>
          %53 = arith.muli %arg6, %c1048576 overflow<nsw> : index
          %54 = arith.addi %53, %25 : index
          %reinterpret_cast_3 = memref.reinterpret_cast %arg0 to offset: [%54], sizes: [1, 128, 512], strides: [1048576, 8192, 1] : memref<16x128x8192xf16> to memref<1x128x512xf16, strided<[1048576, 8192, 1], offset: ?>>
          loom.copy %reinterpret_cast_3, %44 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] region : (UL : [%arg4, %arg5], LR : [%arg4, %arg5]) : memref<1x128x512xf16, strided<[1048576, 8192, 1], offset: ?>> to memref<1x128x512xf16>
          loom.batch_matmul ins(%23, %44 : memref<1x32x128xf16>, memref<1x128x512xf16>) outs(%46 : memref<1x32x512xf16>)
          loom.semaphore_give %44 : memref<1x128x512xf16>
          loom.semaphore_give %23 : memref<1x32x128xf16>
          linalg.fill ins(%cst_1 : f16) outs(%38 : memref<1x32x1xf16>)
          linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, 0)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%46 : memref<1x32x512xf16>) outs(%38 : memref<1x32x1xf16>) {
          ^bb0(%in: f16, %out: f16):
            %77 = arith.maximumf %in, %out : f16
            linalg.yield %77 : f16
          }
          linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%34, %38 : memref<1x32x1xf16>, memref<1x32x1xf16>) outs(%38 : memref<1x32x1xf16>) {
          ^bb0(%in: f16, %in_5: f16, %out: f16):
            %77 = arith.mulf %in_5, %cst_2 : f16
            %78 = arith.cmpf ogt, %in, %77 : f16
            %79 = arith.select %78, %in, %77 : f16
            linalg.yield %79 : f16
          }
          %55 = loom.broadcast ins(%38 : memref<1x32x1xf16>) outs(%50 : memref<1x32x32xf16>) dim(2) -> memref<1x32x512xf16, strided<[?, ?, ?], offset: ?>>
          linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%46, %55 : memref<1x32x512xf16>, memref<1x32x512xf16, strided<[?, ?, ?], offset: ?>>) outs(%46 : memref<1x32x512xf16>) {
          ^bb0(%in: f16, %in_5: f16, %out: f16):
            %77 = arith.mulf %in, %cst_2 : f16
            %78 = arith.subf %77, %in_5 : f16
            %79 = math.exp %78 : f16
            linalg.yield %79 : f16
          }
          loom.semaphore_give %50 : memref<1x32x32xf16>
          linalg.fill ins(%cst : f16) outs(%40 : memref<1x32x1xf16>)
          linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, 0)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%46 : memref<1x32x512xf16>) outs(%40 : memref<1x32x1xf16>) {
          ^bb0(%in: f16, %out: f16):
            %77 = arith.addf %in, %out : f16
            linalg.yield %77 : f16
          }
          linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%34, %38 : memref<1x32x1xf16>, memref<1x32x1xf16>) outs(%42 : memref<1x32x1xf16>) {
          ^bb0(%in: f16, %in_5: f16, %out: f16):
            %77 = arith.subf %in, %in_5 : f16
            %78 = math.exp %77 : f16
            linalg.yield %78 : f16
          }
          linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%32, %42, %40 : memref<1x32x1xf16>, memref<1x32x1xf16>, memref<1x32x1xf16>) outs(%32 : memref<1x32x1xf16>) {
          ^bb0(%in: f16, %in_5: f16, %in_6: f16, %out: f16):
            %77 = arith.mulf %in, %in_5 : f16
            %78 = arith.addf %77, %in_6 : f16
            linalg.yield %78 : f16
          }
          loom.semaphore_give %40 : memref<1x32x1xf16>
          %56 = loom.broadcast ins(%42 : memref<1x32x1xf16>) outs(%49 : memref<1x32x32xf16>) dim(2) -> memref<1x32x128xf16, strided<[?, ?, ?], offset: ?>>
          loom.semaphore_give %42 : memref<1x32x1xf16>
          %57 = arith.muli %21, %c65536 : index
          %58 = arith.addi %53, %57 : index
          %reinterpret_cast_4 = memref.reinterpret_cast %arg1 to offset: [%58], sizes: [1, 512, 128], strides: [1048576, 128, 1] : memref<16x8192x128xf16> to memref<1x512x128xf16, strided<[1048576, 128, 1], offset: ?>>
          loom.copy %reinterpret_cast_4, %52 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] region : (UL : [%arg4, %arg5], LR : [%arg4, %arg5]) : memref<1x512x128xf16, strided<[1048576, 128, 1], offset: ?>> to memref<1x512x128xf16>
          loom.batch_matmul ins(%46, %52 : memref<1x32x512xf16>, memref<1x512x128xf16>) outs(%36 : memref<1x32x128xf16>)
          loom.semaphore_give %52 : memref<1x512x128xf16>
          loom.semaphore_give %46 : memref<1x32x512xf16>
          linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%36, %28, %56 : memref<1x32x128xf16>, memref<1x32x128xf16>, memref<1x32x128xf16, strided<[?, ?, ?], offset: ?>>) outs(%28 : memref<1x32x128xf16>) {
          ^bb0(%in: f16, %in_5: f16, %in_6: f16, %out: f16):
            %77 = arith.mulf %in_5, %in_6 : f16
            %78 = arith.addf %in, %77 : f16
            linalg.yield %78 : f16
          }
          loom.semaphore_give %49 : memref<1x32x32xf16>
          loom.semaphore_give %36 : memref<1x32x128xf16>
          linalg.copy ins(%38 : memref<1x32x1xf16>) outs(%34 : memref<1x32x1xf16>)
          loom.semaphore_give %38 : memref<1x32x1xf16>
          %59 = loom.alloc [1, 32, 1] on @L1 : memref<1x32x1xf16>
          %60 = loom.semaphore_take %59 : memref<1x32x1xf16> -> memref<1x32x1xf16>
          linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%32, %34 : memref<1x32x1xf16>, memref<1x32x1xf16>) outs(%60 : memref<1x32x1xf16>) {
          ^bb0(%in: f16, %in_5: f16, %out: f16):
            %77 = math.log %in : f16
            %78 = arith.addf %77, %in_5 : f16
            linalg.yield %78 : f16
          }
          loom.semaphore_give %34 : memref<1x32x1xf16>
          %61 = loom.broadcast ins(%32 : memref<1x32x1xf16>) outs(%48 : memref<1x32x32xf16>) dim(2) -> memref<1x32x128xf16, strided<[?, ?, ?], offset: ?>>
          loom.semaphore_give %32 : memref<1x32x1xf16>
          %62 = loom.alloc [1, 32, 128] on @L1 : memref<1x32x128xf16>
          %63 = loom.semaphore_take %62 : memref<1x32x128xf16> -> memref<1x32x128xf16>
          linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%28, %61 : memref<1x32x128xf16>, memref<1x32x128xf16, strided<[?, ?, ?], offset: ?>>) outs(%63 : memref<1x32x128xf16>) {
          ^bb0(%in: f16, %in_5: f16, %out: f16):
            %77 = arith.divf %in, %in_5 : f16
            linalg.yield %77 : f16
          }
          loom.semaphore_give %48 : memref<1x32x32xf16>
          loom.semaphore_give %28 : memref<1x32x128xf16>
          %64 = loom.alloc [16, 1, 32, 1] on @L1 : memref<16x1x32x1xf16>
          %65 = loom.semaphore_take %64 : memref<16x1x32x1xf16> -> memref<16x1x32x1xf16>
          loom.gather ins(%60 : memref<1x32x1xf16>) outs(%65 : memref<16x1x32x1xf16>) across(%arg4 : index) region : (UL : [%c0, %arg5], LR : [%c7, %arg5])
          loom.semaphore_give %60 : memref<1x32x1xf16>
          %66 = loom.alloc [16, 1, 32, 128] on @L1 : memref<16x1x32x128xf16>
          %67 = loom.semaphore_take %66 : memref<16x1x32x128xf16> -> memref<16x1x32x128xf16>
          loom.gather ins(%63 : memref<1x32x128xf16>) outs(%67 : memref<16x1x32x128xf16>) across(%arg4 : index) region : (UL : [%c0, %arg5], LR : [%c7, %arg5])
          loom.semaphore_give %63 : memref<1x32x128xf16>
          %68 = loom.alloc [1, 32, 128] on @L1 : memref<1x32x128xf16>
          %69 = loom.semaphore_take %68 : memref<1x32x128xf16> -> memref<1x32x128xf16>
          %70 = loom.alloc [16, 1, 32, 1] on @L1 : memref<16x1x32x1xf16>
          %71 = loom.semaphore_take %70 : memref<16x1x32x1xf16> -> memref<16x1x32x1xf16>
          %72 = loom.alloc [16, 1, 32, 128] on @L1 : memref<16x1x32x128xf16>
          %73 = loom.semaphore_take %72 : memref<16x1x32x128xf16> -> memref<16x1x32x128xf16>
          %74 = loom.alloc [16, 1, 32, 32] on @L1 : memref<16x1x32x32xf16>
          %75 = loom.semaphore_take %74 : memref<16x1x32x32xf16> -> memref<16x1x32x32xf16>
          %76 = arith.cmpi eq, %arg4, %c0 : index
          scf.if %76 {
            linalg.fill ins(%cst_1 : f16) outs(%31 : memref<1x32x1xf16>)
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1, d2, d3)>], iterator_types = ["reduction", "parallel", "parallel", "parallel"]} ins(%65 : memref<16x1x32x1xf16>) outs(%31 : memref<1x32x1xf16>) {
            ^bb0(%in: f16, %out: f16):
              %78 = arith.maximumf %in, %out : f16
              linalg.yield %78 : f16
            }
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%65, %31 : memref<16x1x32x1xf16>, memref<1x32x1xf16>) outs(%71 : memref<16x1x32x1xf16>) {
            ^bb0(%in: f16, %in_6: f16, %out: f16):
              %78 = arith.subf %in, %in_6 : f16
              %79 = math.exp %78 : f16
              linalg.yield %79 : f16
            }
            loom.semaphore_give %65 : memref<16x1x32x1xf16>
            loom.semaphore_give %31 : memref<1x32x1xf16>
            linalg.fill ins(%cst : f16) outs(%30 : memref<1x32x1xf16>)
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1, d2, d3)>], iterator_types = ["reduction", "parallel", "parallel", "parallel"]} ins(%71 : memref<16x1x32x1xf16>) outs(%30 : memref<1x32x1xf16>) {
            ^bb0(%in: f16, %out: f16):
              %78 = arith.addf %in, %out : f16
              linalg.yield %78 : f16
            }
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%71, %30 : memref<16x1x32x1xf16>, memref<1x32x1xf16>) outs(%71 : memref<16x1x32x1xf16>) {
            ^bb0(%in: f16, %in_6: f16, %out: f16):
              %78 = arith.divf %in, %in_6 : f16
              linalg.yield %78 : f16
            }
            loom.semaphore_give %30 : memref<1x32x1xf16>
            %77 = loom.broadcast ins(%71 : memref<16x1x32x1xf16>) outs(%75 : memref<16x1x32x32xf16>) dim(3) -> memref<16x1x32x128xf16, strided<[?, ?, ?, ?], offset: ?>>
            loom.semaphore_give %71 : memref<16x1x32x1xf16>
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%67, %77 : memref<16x1x32x128xf16>, memref<16x1x32x128xf16, strided<[?, ?, ?, ?], offset: ?>>) outs(%73 : memref<16x1x32x128xf16>) {
            ^bb0(%in: f16, %in_6: f16, %out: f16):
              %78 = arith.mulf %in, %in_6 : f16
              linalg.yield %78 : f16
            }
            loom.semaphore_give %75 : memref<16x1x32x32xf16>
            loom.semaphore_give %67 : memref<16x1x32x128xf16>
            linalg.fill ins(%cst : f16) outs(%69 : memref<1x32x128xf16>)
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1, d2, d3)>], iterator_types = ["reduction", "parallel", "parallel", "parallel"]} ins(%73 : memref<16x1x32x128xf16>) outs(%69 : memref<1x32x128xf16>) {
            ^bb0(%in: f16, %out: f16):
              %78 = arith.addf %in, %out : f16
              linalg.yield %78 : f16
            }
            loom.semaphore_give %73 : memref<16x1x32x128xf16>
            loom.semaphore_give %27 : memref<1x32x128xf16>
            %reinterpret_cast_5 = memref.reinterpret_cast %arg2 to offset: [%24], sizes: [1, 32, 128], strides: [4096, 128, 1] : memref<16x32x128xf16> to memref<1x32x128xf16, strided<[4096, 128, 1], offset: ?>>
            loom.copy %69, %reinterpret_cast_5 src_mem_space @mem_L1 dst_mem_space @mem_DRAM, broadcast : [1, 1] region : (UL : [%arg4, %arg5], LR : [%arg4, %arg5]) : memref<1x32x128xf16> to memref<1x32x128xf16, strided<[4096, 128, 1], offset: ?>>
            loom.semaphore_give %69 : memref<1x32x128xf16>
          }
        } {loom.block_sym = @tile_b, loom.iter_type = #loom.iter_type<temporal>}
        scf.reduce 
      }
      return
    }
  }
  module attributes {loom.pass_name = "Materialize", loom.tile_b = {is_reduction = false, upper_bound = 16 : index}, loom.tile_n = {is_reduction = false, upper_bound = 64 : index}, loom.tile_s = {is_reduction = false, upper_bound = 8192 : index}} {
    func.func @flash_decode__x1x8_y8__d0i1_d1i1_d2i0__f01__dim_y_level0_bc8_n_n_n__tile_b1__tile_n512__tile_s512(%arg0: memref<16x128x8192xf16>, %arg1: memref<16x8192x128xf16>, %arg2: memref<16x32x128xf16>, %arg3: memref<16x32x128xf16>) {
      %c65536 = arith.constant 65536 : index
      %c1048576 = arith.constant 1048576 : index
      %c4096 = arith.constant 4096 : index
      %c2 = arith.constant 2 : index
      %c7 = arith.constant 7 : index
      %c0 = arith.constant 0 : index
      %cst = arith.constant 0.000000e+00 : f16
      %cst_0 = arith.constant 1.000000e+00 : f16
      %cst_1 = arith.constant 0xFC00 : f16
      %c1 = arith.constant 1 : index
      %cst_2 = arith.constant 8.837890e-02 : f16
      %c512 = arith.constant 512 : index
      %c8 = arith.constant 8 : index
      scf.parallel (%arg4, %arg5) = (%c0, %c0) to (%c8, %c8) step (%c1, %c1) {
        scf.for %arg6 = %c0 to %c2 step %c1 {
          scf.for %arg7 = %c0 to %c2 step %c1 {
            %20 = arith.muli %arg6, %c8 overflow<nsw> : index
            %21 = arith.addi %arg4, %20 : index
            %22 = arith.muli %arg7, %c8 overflow<nsw> : index
            %23 = arith.addi %arg5, %22 : index
            %24 = loom.alloc [1, 32, 128] on @L1 : memref<1x32x128xf16>
            %25 = loom.semaphore_take %24 : memref<1x32x128xf16> -> memref<1x32x128xf16>
            %26 = arith.muli %21, %c4096 overflow<nsw> : index
            %reinterpret_cast = memref.reinterpret_cast %arg3 to offset: [%26], sizes: [1, 32, 128], strides: [4096, 128, 1] : memref<16x32x128xf16> to memref<1x32x128xf16, strided<[4096, 128, 1], offset: ?>>
            loom.copy %reinterpret_cast, %25 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 8] region : (UL : [%arg4, %c0], LR : [%arg4, %c7]) : memref<1x32x128xf16, strided<[4096, 128, 1], offset: ?>> to memref<1x32x128xf16>
            %27 = arith.muli %23, %c512 : index
            %28 = loom.alloc [1, 32, 128] on @L1 : memref<1x32x128xf16>
            %29 = loom.semaphore_take %28 : memref<1x32x128xf16> -> memref<1x32x128xf16>
            %30 = loom.semaphore_take %28 : memref<1x32x128xf16> -> memref<1x32x128xf16>
            linalg.fill ins(%cst : f16) outs(%30 : memref<1x32x128xf16>)
            %31 = loom.alloc [1, 32, 1] on @L1 : memref<1x32x1xf16>
            %32 = loom.semaphore_take %31 : memref<1x32x1xf16> -> memref<1x32x1xf16>
            %33 = loom.semaphore_take %31 : memref<1x32x1xf16> -> memref<1x32x1xf16>
            %34 = loom.semaphore_take %31 : memref<1x32x1xf16> -> memref<1x32x1xf16>
            linalg.fill ins(%cst_0 : f16) outs(%34 : memref<1x32x1xf16>)
            %35 = loom.alloc [1, 32, 1] on @L1 : memref<1x32x1xf16>
            %36 = loom.semaphore_take %35 : memref<1x32x1xf16> -> memref<1x32x1xf16>
            linalg.fill ins(%cst_1 : f16) outs(%36 : memref<1x32x1xf16>)
            %37 = loom.alloc [1, 32, 128] on @L1 : memref<1x32x128xf16>
            %38 = loom.semaphore_take %37 : memref<1x32x128xf16> -> memref<1x32x128xf16>
            %39 = loom.alloc [1, 32, 1] on @L1 : memref<1x32x1xf16>
            %40 = loom.semaphore_take %39 : memref<1x32x1xf16> -> memref<1x32x1xf16>
            %41 = loom.alloc [1, 32, 1] on @L1 : memref<1x32x1xf16>
            %42 = loom.semaphore_take %41 : memref<1x32x1xf16> -> memref<1x32x1xf16>
            %43 = loom.alloc [1, 32, 1] on @L1 : memref<1x32x1xf16>
            %44 = loom.semaphore_take %43 : memref<1x32x1xf16> -> memref<1x32x1xf16>
            %45 = loom.alloc [1, 128, 512] on @L1 : memref<1x128x512xf16>
            %46 = loom.semaphore_take %45 : memref<1x128x512xf16> -> memref<1x128x512xf16>
            %47 = loom.alloc [1, 32, 512] on @L1 : memref<1x32x512xf16>
            %48 = loom.semaphore_take %47 : memref<1x32x512xf16> -> memref<1x32x512xf16>
            %49 = loom.alloc [1, 32, 32] on @L1 : memref<1x32x32xf16>
            %50 = loom.semaphore_take %49 : memref<1x32x32xf16> -> memref<1x32x32xf16>
            %51 = loom.semaphore_take %49 : memref<1x32x32xf16> -> memref<1x32x32xf16>
            %52 = loom.semaphore_take %49 : memref<1x32x32xf16> -> memref<1x32x32xf16>
            %53 = loom.alloc [1, 512, 128] on @L1 : memref<1x512x128xf16>
            %54 = loom.semaphore_take %53 : memref<1x512x128xf16> -> memref<1x512x128xf16>
            %55 = arith.muli %21, %c1048576 overflow<nsw> : index
            %56 = arith.addi %55, %27 : index
            %reinterpret_cast_3 = memref.reinterpret_cast %arg0 to offset: [%56], sizes: [1, 128, 512], strides: [1048576, 8192, 1] : memref<16x128x8192xf16> to memref<1x128x512xf16, strided<[1048576, 8192, 1], offset: ?>>
            loom.copy %reinterpret_cast_3, %46 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] region : (UL : [%arg4, %arg5], LR : [%arg4, %arg5]) : memref<1x128x512xf16, strided<[1048576, 8192, 1], offset: ?>> to memref<1x128x512xf16>
            loom.batch_matmul ins(%25, %46 : memref<1x32x128xf16>, memref<1x128x512xf16>) outs(%48 : memref<1x32x512xf16>)
            loom.semaphore_give %46 : memref<1x128x512xf16>
            loom.semaphore_give %25 : memref<1x32x128xf16>
            linalg.fill ins(%cst_1 : f16) outs(%40 : memref<1x32x1xf16>)
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, 0)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%48 : memref<1x32x512xf16>) outs(%40 : memref<1x32x1xf16>) {
            ^bb0(%in: f16, %out: f16):
              %79 = arith.maximumf %in, %out : f16
              linalg.yield %79 : f16
            }
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%36, %40 : memref<1x32x1xf16>, memref<1x32x1xf16>) outs(%40 : memref<1x32x1xf16>) {
            ^bb0(%in: f16, %in_6: f16, %out: f16):
              %79 = arith.mulf %in_6, %cst_2 : f16
              %80 = arith.cmpf ogt, %in, %79 : f16
              %81 = arith.select %80, %in, %79 : f16
              linalg.yield %81 : f16
            }
            %57 = loom.broadcast ins(%40 : memref<1x32x1xf16>) outs(%52 : memref<1x32x32xf16>) dim(2) -> memref<1x32x512xf16, strided<[?, ?, ?], offset: ?>>
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%48, %57 : memref<1x32x512xf16>, memref<1x32x512xf16, strided<[?, ?, ?], offset: ?>>) outs(%48 : memref<1x32x512xf16>) {
            ^bb0(%in: f16, %in_6: f16, %out: f16):
              %79 = arith.mulf %in, %cst_2 : f16
              %80 = arith.subf %79, %in_6 : f16
              %81 = math.exp %80 : f16
              linalg.yield %81 : f16
            }
            loom.semaphore_give %52 : memref<1x32x32xf16>
            linalg.fill ins(%cst : f16) outs(%42 : memref<1x32x1xf16>)
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, 0)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%48 : memref<1x32x512xf16>) outs(%42 : memref<1x32x1xf16>) {
            ^bb0(%in: f16, %out: f16):
              %79 = arith.addf %in, %out : f16
              linalg.yield %79 : f16
            }
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%36, %40 : memref<1x32x1xf16>, memref<1x32x1xf16>) outs(%44 : memref<1x32x1xf16>) {
            ^bb0(%in: f16, %in_6: f16, %out: f16):
              %79 = arith.subf %in, %in_6 : f16
              %80 = math.exp %79 : f16
              linalg.yield %80 : f16
            }
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%34, %44, %42 : memref<1x32x1xf16>, memref<1x32x1xf16>, memref<1x32x1xf16>) outs(%34 : memref<1x32x1xf16>) {
            ^bb0(%in: f16, %in_6: f16, %in_7: f16, %out: f16):
              %79 = arith.mulf %in, %in_6 : f16
              %80 = arith.addf %79, %in_7 : f16
              linalg.yield %80 : f16
            }
            loom.semaphore_give %42 : memref<1x32x1xf16>
            %58 = loom.broadcast ins(%44 : memref<1x32x1xf16>) outs(%51 : memref<1x32x32xf16>) dim(2) -> memref<1x32x128xf16, strided<[?, ?, ?], offset: ?>>
            loom.semaphore_give %44 : memref<1x32x1xf16>
            %59 = arith.muli %23, %c65536 : index
            %60 = arith.addi %55, %59 : index
            %reinterpret_cast_4 = memref.reinterpret_cast %arg1 to offset: [%60], sizes: [1, 512, 128], strides: [1048576, 128, 1] : memref<16x8192x128xf16> to memref<1x512x128xf16, strided<[1048576, 128, 1], offset: ?>>
            loom.copy %reinterpret_cast_4, %54 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] region : (UL : [%arg4, %arg5], LR : [%arg4, %arg5]) : memref<1x512x128xf16, strided<[1048576, 128, 1], offset: ?>> to memref<1x512x128xf16>
            loom.batch_matmul ins(%48, %54 : memref<1x32x512xf16>, memref<1x512x128xf16>) outs(%38 : memref<1x32x128xf16>)
            loom.semaphore_give %54 : memref<1x512x128xf16>
            loom.semaphore_give %48 : memref<1x32x512xf16>
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%38, %30, %58 : memref<1x32x128xf16>, memref<1x32x128xf16>, memref<1x32x128xf16, strided<[?, ?, ?], offset: ?>>) outs(%30 : memref<1x32x128xf16>) {
            ^bb0(%in: f16, %in_6: f16, %in_7: f16, %out: f16):
              %79 = arith.mulf %in_6, %in_7 : f16
              %80 = arith.addf %in, %79 : f16
              linalg.yield %80 : f16
            }
            loom.semaphore_give %51 : memref<1x32x32xf16>
            loom.semaphore_give %38 : memref<1x32x128xf16>
            linalg.copy ins(%40 : memref<1x32x1xf16>) outs(%36 : memref<1x32x1xf16>)
            loom.semaphore_give %40 : memref<1x32x1xf16>
            %61 = loom.alloc [1, 32, 1] on @L1 : memref<1x32x1xf16>
            %62 = loom.semaphore_take %61 : memref<1x32x1xf16> -> memref<1x32x1xf16>
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%34, %36 : memref<1x32x1xf16>, memref<1x32x1xf16>) outs(%62 : memref<1x32x1xf16>) {
            ^bb0(%in: f16, %in_6: f16, %out: f16):
              %79 = math.log %in : f16
              %80 = arith.addf %79, %in_6 : f16
              linalg.yield %80 : f16
            }
            loom.semaphore_give %36 : memref<1x32x1xf16>
            %63 = loom.broadcast ins(%34 : memref<1x32x1xf16>) outs(%50 : memref<1x32x32xf16>) dim(2) -> memref<1x32x128xf16, strided<[?, ?, ?], offset: ?>>
            loom.semaphore_give %34 : memref<1x32x1xf16>
            %64 = loom.alloc [1, 32, 128] on @L1 : memref<1x32x128xf16>
            %65 = loom.semaphore_take %64 : memref<1x32x128xf16> -> memref<1x32x128xf16>
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%30, %63 : memref<1x32x128xf16>, memref<1x32x128xf16, strided<[?, ?, ?], offset: ?>>) outs(%65 : memref<1x32x128xf16>) {
            ^bb0(%in: f16, %in_6: f16, %out: f16):
              %79 = arith.divf %in, %in_6 : f16
              linalg.yield %79 : f16
            }
            loom.semaphore_give %50 : memref<1x32x32xf16>
            loom.semaphore_give %30 : memref<1x32x128xf16>
            %66 = loom.alloc [16, 1, 32, 1] on @L1 : memref<16x1x32x1xf16>
            %67 = loom.semaphore_take %66 : memref<16x1x32x1xf16> -> memref<16x1x32x1xf16>
            loom.gather ins(%62 : memref<1x32x1xf16>) outs(%67 : memref<16x1x32x1xf16>) across(%c0 : index) region : (UL : [%arg4, %arg5], LR : [%arg4, %arg5])
            loom.semaphore_give %62 : memref<1x32x1xf16>
            %68 = loom.alloc [16, 1, 32, 128] on @L1 : memref<16x1x32x128xf16>
            %69 = loom.semaphore_take %68 : memref<16x1x32x128xf16> -> memref<16x1x32x128xf16>
            loom.gather ins(%65 : memref<1x32x128xf16>) outs(%69 : memref<16x1x32x128xf16>) across(%c0 : index) region : (UL : [%arg4, %arg5], LR : [%arg4, %arg5])
            loom.semaphore_give %65 : memref<1x32x128xf16>
            %70 = loom.alloc [1, 32, 128] on @L1 : memref<1x32x128xf16>
            %71 = loom.semaphore_take %70 : memref<1x32x128xf16> -> memref<1x32x128xf16>
            %72 = loom.alloc [16, 1, 32, 1] on @L1 : memref<16x1x32x1xf16>
            %73 = loom.semaphore_take %72 : memref<16x1x32x1xf16> -> memref<16x1x32x1xf16>
            %74 = loom.alloc [16, 1, 32, 128] on @L1 : memref<16x1x32x128xf16>
            %75 = loom.semaphore_take %74 : memref<16x1x32x128xf16> -> memref<16x1x32x128xf16>
            %76 = loom.alloc [16, 1, 32, 32] on @L1 : memref<16x1x32x32xf16>
            %77 = loom.semaphore_take %76 : memref<16x1x32x32xf16> -> memref<16x1x32x32xf16>
            linalg.fill ins(%cst_1 : f16) outs(%33 : memref<1x32x1xf16>)
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1, d2, d3)>], iterator_types = ["reduction", "parallel", "parallel", "parallel"]} ins(%67 : memref<16x1x32x1xf16>) outs(%33 : memref<1x32x1xf16>) {
            ^bb0(%in: f16, %out: f16):
              %79 = arith.maximumf %in, %out : f16
              linalg.yield %79 : f16
            }
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%67, %33 : memref<16x1x32x1xf16>, memref<1x32x1xf16>) outs(%73 : memref<16x1x32x1xf16>) {
            ^bb0(%in: f16, %in_6: f16, %out: f16):
              %79 = arith.subf %in, %in_6 : f16
              %80 = math.exp %79 : f16
              linalg.yield %80 : f16
            }
            loom.semaphore_give %67 : memref<16x1x32x1xf16>
            loom.semaphore_give %33 : memref<1x32x1xf16>
            linalg.fill ins(%cst : f16) outs(%32 : memref<1x32x1xf16>)
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1, d2, d3)>], iterator_types = ["reduction", "parallel", "parallel", "parallel"]} ins(%73 : memref<16x1x32x1xf16>) outs(%32 : memref<1x32x1xf16>) {
            ^bb0(%in: f16, %out: f16):
              %79 = arith.addf %in, %out : f16
              linalg.yield %79 : f16
            }
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%73, %32 : memref<16x1x32x1xf16>, memref<1x32x1xf16>) outs(%73 : memref<16x1x32x1xf16>) {
            ^bb0(%in: f16, %in_6: f16, %out: f16):
              %79 = arith.divf %in, %in_6 : f16
              linalg.yield %79 : f16
            }
            loom.semaphore_give %32 : memref<1x32x1xf16>
            %78 = loom.broadcast ins(%73 : memref<16x1x32x1xf16>) outs(%77 : memref<16x1x32x32xf16>) dim(3) -> memref<16x1x32x128xf16, strided<[?, ?, ?, ?], offset: ?>>
            loom.semaphore_give %73 : memref<16x1x32x1xf16>
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%69, %78 : memref<16x1x32x128xf16>, memref<16x1x32x128xf16, strided<[?, ?, ?, ?], offset: ?>>) outs(%75 : memref<16x1x32x128xf16>) {
            ^bb0(%in: f16, %in_6: f16, %out: f16):
              %79 = arith.mulf %in, %in_6 : f16
              linalg.yield %79 : f16
            }
            loom.semaphore_give %77 : memref<16x1x32x32xf16>
            loom.semaphore_give %69 : memref<16x1x32x128xf16>
            linalg.fill ins(%cst : f16) outs(%71 : memref<1x32x128xf16>)
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1, d2, d3)>], iterator_types = ["reduction", "parallel", "parallel", "parallel"]} ins(%75 : memref<16x1x32x128xf16>) outs(%71 : memref<1x32x128xf16>) {
            ^bb0(%in: f16, %out: f16):
              %79 = arith.addf %in, %out : f16
              linalg.yield %79 : f16
            }
            loom.semaphore_give %75 : memref<16x1x32x128xf16>
            loom.semaphore_give %29 : memref<1x32x128xf16>
            %reinterpret_cast_5 = memref.reinterpret_cast %arg2 to offset: [%26], sizes: [1, 32, 128], strides: [4096, 128, 1] : memref<16x32x128xf16> to memref<1x32x128xf16, strided<[4096, 128, 1], offset: ?>>
            loom.copy %71, %reinterpret_cast_5 src_mem_space @mem_L1 dst_mem_space @mem_DRAM, broadcast : [1, 1] region : (UL : [%arg4, %arg5], LR : [%arg4, %arg5]) : memref<1x32x128xf16> to memref<1x32x128xf16, strided<[4096, 128, 1], offset: ?>>
            loom.semaphore_give %71 : memref<1x32x128xf16>
          } {loom.block_sym = @tile_s, loom.iter_type = #loom.iter_type<temporal>}
        } {loom.block_sym = @tile_b, loom.iter_type = #loom.iter_type<temporal>}
        scf.reduce 
      }
      return
    }
  }
  module attributes {loom.pass_name = "Materialize", loom.tile_b = {is_reduction = false, upper_bound = 16 : index}, loom.tile_n = {is_reduction = false, upper_bound = 64 : index}, loom.tile_s = {is_reduction = false, upper_bound = 8192 : index}} {
    func.func @flash_decode__x2x4_y8__d0i1_d1i1_d2i0__f01__dim_x_level0_bc2_dim_y_level0_bc8_n_n_n__tile_b1__tile_n512__tile_s512(%arg0: memref<16x128x8192xf16>, %arg1: memref<16x8192x128xf16>, %arg2: memref<16x32x128xf16>, %arg3: memref<16x32x128xf16>) {
      %c65536 = arith.constant 65536 : index
      %c1048576 = arith.constant 1048576 : index
      %c4096 = arith.constant 4096 : index
      %c4 = arith.constant 4 : index
      %c7 = arith.constant 7 : index
      %c2 = arith.constant 2 : index
      %c0 = arith.constant 0 : index
      %cst = arith.constant 0.000000e+00 : f16
      %cst_0 = arith.constant 1.000000e+00 : f16
      %cst_1 = arith.constant 0xFC00 : f16
      %c1 = arith.constant 1 : index
      %cst_2 = arith.constant 8.837890e-02 : f16
      %c512 = arith.constant 512 : index
      %c8 = arith.constant 8 : index
      scf.parallel (%arg4, %arg5, %arg6) = (%c0, %c0, %c0) to (%c4, %c2, %c8) step (%c1, %c1, %c1) {
        scf.for %arg7 = %c0 to %c4 step %c1 {
          %20 = arith.muli %arg7, %c4 overflow<nsw> : index
          %21 = arith.addi %arg4, %20 : index
          %22 = arith.muli %arg5, %c2 overflow<nsw> : index
          %23 = arith.addi %22, %arg6 : index
          %24 = loom.alloc [1, 32, 128] on @L1 : memref<1x32x128xf16>
          %25 = loom.semaphore_take %24 : memref<1x32x128xf16> -> memref<1x32x128xf16>
          %26 = arith.muli %21, %c4096 overflow<nsw> : index
          %reinterpret_cast = memref.reinterpret_cast %arg3 to offset: [%26], sizes: [1, 32, 128], strides: [4096, 128, 1] : memref<16x32x128xf16> to memref<1x32x128xf16, strided<[4096, 128, 1], offset: ?>>
          %27 = arith.muli %arg4, %c2 : index
          %28 = arith.addi %27, %c1 : index
          loom.copy %reinterpret_cast, %25 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [2, 8] region : (UL : [%27, %c0], LR : [%28, %c7]) : memref<1x32x128xf16, strided<[4096, 128, 1], offset: ?>> to memref<1x32x128xf16>
          %29 = arith.muli %23, %c512 : index
          %30 = loom.alloc [1, 32, 128] on @L1 : memref<1x32x128xf16>
          %31 = loom.semaphore_take %30 : memref<1x32x128xf16> -> memref<1x32x128xf16>
          %32 = loom.semaphore_take %30 : memref<1x32x128xf16> -> memref<1x32x128xf16>
          linalg.fill ins(%cst : f16) outs(%32 : memref<1x32x128xf16>)
          %33 = loom.alloc [1, 32, 1] on @L1 : memref<1x32x1xf16>
          %34 = loom.semaphore_take %33 : memref<1x32x1xf16> -> memref<1x32x1xf16>
          %35 = loom.semaphore_take %33 : memref<1x32x1xf16> -> memref<1x32x1xf16>
          %36 = loom.semaphore_take %33 : memref<1x32x1xf16> -> memref<1x32x1xf16>
          linalg.fill ins(%cst_0 : f16) outs(%36 : memref<1x32x1xf16>)
          %37 = loom.alloc [1, 32, 1] on @L1 : memref<1x32x1xf16>
          %38 = loom.semaphore_take %37 : memref<1x32x1xf16> -> memref<1x32x1xf16>
          linalg.fill ins(%cst_1 : f16) outs(%38 : memref<1x32x1xf16>)
          %39 = loom.alloc [1, 32, 128] on @L1 : memref<1x32x128xf16>
          %40 = loom.semaphore_take %39 : memref<1x32x128xf16> -> memref<1x32x128xf16>
          %41 = loom.alloc [1, 32, 1] on @L1 : memref<1x32x1xf16>
          %42 = loom.semaphore_take %41 : memref<1x32x1xf16> -> memref<1x32x1xf16>
          %43 = loom.alloc [1, 32, 1] on @L1 : memref<1x32x1xf16>
          %44 = loom.semaphore_take %43 : memref<1x32x1xf16> -> memref<1x32x1xf16>
          %45 = loom.alloc [1, 32, 1] on @L1 : memref<1x32x1xf16>
          %46 = loom.semaphore_take %45 : memref<1x32x1xf16> -> memref<1x32x1xf16>
          %47 = loom.alloc [1, 128, 512] on @L1 : memref<1x128x512xf16>
          %48 = loom.semaphore_take %47 : memref<1x128x512xf16> -> memref<1x128x512xf16>
          %49 = loom.alloc [1, 32, 512] on @L1 : memref<1x32x512xf16>
          %50 = loom.semaphore_take %49 : memref<1x32x512xf16> -> memref<1x32x512xf16>
          %51 = loom.alloc [1, 32, 32] on @L1 : memref<1x32x32xf16>
          %52 = loom.semaphore_take %51 : memref<1x32x32xf16> -> memref<1x32x32xf16>
          %53 = loom.semaphore_take %51 : memref<1x32x32xf16> -> memref<1x32x32xf16>
          %54 = loom.semaphore_take %51 : memref<1x32x32xf16> -> memref<1x32x32xf16>
          %55 = loom.alloc [1, 512, 128] on @L1 : memref<1x512x128xf16>
          %56 = loom.semaphore_take %55 : memref<1x512x128xf16> -> memref<1x512x128xf16>
          %57 = arith.muli %21, %c1048576 overflow<nsw> : index
          %58 = arith.addi %57, %29 : index
          %reinterpret_cast_3 = memref.reinterpret_cast %arg0 to offset: [%58], sizes: [1, 128, 512], strides: [1048576, 8192, 1] : memref<16x128x8192xf16> to memref<1x128x512xf16, strided<[1048576, 8192, 1], offset: ?>>
          %59 = arith.addi %arg5, %27 : index
          loom.copy %reinterpret_cast_3, %48 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] region : (UL : [%59, %arg6], LR : [%59, %arg6]) : memref<1x128x512xf16, strided<[1048576, 8192, 1], offset: ?>> to memref<1x128x512xf16>
          loom.batch_matmul ins(%25, %48 : memref<1x32x128xf16>, memref<1x128x512xf16>) outs(%50 : memref<1x32x512xf16>)
          loom.semaphore_give %48 : memref<1x128x512xf16>
          loom.semaphore_give %25 : memref<1x32x128xf16>
          linalg.fill ins(%cst_1 : f16) outs(%42 : memref<1x32x1xf16>)
          linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, 0)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%50 : memref<1x32x512xf16>) outs(%42 : memref<1x32x1xf16>) {
          ^bb0(%in: f16, %out: f16):
            %82 = arith.maximumf %in, %out : f16
            linalg.yield %82 : f16
          }
          linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%38, %42 : memref<1x32x1xf16>, memref<1x32x1xf16>) outs(%42 : memref<1x32x1xf16>) {
          ^bb0(%in: f16, %in_5: f16, %out: f16):
            %82 = arith.mulf %in_5, %cst_2 : f16
            %83 = arith.cmpf ogt, %in, %82 : f16
            %84 = arith.select %83, %in, %82 : f16
            linalg.yield %84 : f16
          }
          %60 = loom.broadcast ins(%42 : memref<1x32x1xf16>) outs(%54 : memref<1x32x32xf16>) dim(2) -> memref<1x32x512xf16, strided<[?, ?, ?], offset: ?>>
          linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%50, %60 : memref<1x32x512xf16>, memref<1x32x512xf16, strided<[?, ?, ?], offset: ?>>) outs(%50 : memref<1x32x512xf16>) {
          ^bb0(%in: f16, %in_5: f16, %out: f16):
            %82 = arith.mulf %in, %cst_2 : f16
            %83 = arith.subf %82, %in_5 : f16
            %84 = math.exp %83 : f16
            linalg.yield %84 : f16
          }
          loom.semaphore_give %54 : memref<1x32x32xf16>
          linalg.fill ins(%cst : f16) outs(%44 : memref<1x32x1xf16>)
          linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, 0)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%50 : memref<1x32x512xf16>) outs(%44 : memref<1x32x1xf16>) {
          ^bb0(%in: f16, %out: f16):
            %82 = arith.addf %in, %out : f16
            linalg.yield %82 : f16
          }
          linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%38, %42 : memref<1x32x1xf16>, memref<1x32x1xf16>) outs(%46 : memref<1x32x1xf16>) {
          ^bb0(%in: f16, %in_5: f16, %out: f16):
            %82 = arith.subf %in, %in_5 : f16
            %83 = math.exp %82 : f16
            linalg.yield %83 : f16
          }
          linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%36, %46, %44 : memref<1x32x1xf16>, memref<1x32x1xf16>, memref<1x32x1xf16>) outs(%36 : memref<1x32x1xf16>) {
          ^bb0(%in: f16, %in_5: f16, %in_6: f16, %out: f16):
            %82 = arith.mulf %in, %in_5 : f16
            %83 = arith.addf %82, %in_6 : f16
            linalg.yield %83 : f16
          }
          loom.semaphore_give %44 : memref<1x32x1xf16>
          %61 = loom.broadcast ins(%46 : memref<1x32x1xf16>) outs(%53 : memref<1x32x32xf16>) dim(2) -> memref<1x32x128xf16, strided<[?, ?, ?], offset: ?>>
          loom.semaphore_give %46 : memref<1x32x1xf16>
          %62 = arith.muli %23, %c65536 : index
          %63 = arith.addi %57, %62 : index
          %reinterpret_cast_4 = memref.reinterpret_cast %arg1 to offset: [%63], sizes: [1, 512, 128], strides: [1048576, 128, 1] : memref<16x8192x128xf16> to memref<1x512x128xf16, strided<[1048576, 128, 1], offset: ?>>
          loom.copy %reinterpret_cast_4, %56 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] region : (UL : [%59, %arg6], LR : [%59, %arg6]) : memref<1x512x128xf16, strided<[1048576, 128, 1], offset: ?>> to memref<1x512x128xf16>
          loom.batch_matmul ins(%50, %56 : memref<1x32x512xf16>, memref<1x512x128xf16>) outs(%40 : memref<1x32x128xf16>)
          loom.semaphore_give %56 : memref<1x512x128xf16>
          loom.semaphore_give %50 : memref<1x32x512xf16>
          linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%40, %32, %61 : memref<1x32x128xf16>, memref<1x32x128xf16>, memref<1x32x128xf16, strided<[?, ?, ?], offset: ?>>) outs(%32 : memref<1x32x128xf16>) {
          ^bb0(%in: f16, %in_5: f16, %in_6: f16, %out: f16):
            %82 = arith.mulf %in_5, %in_6 : f16
            %83 = arith.addf %in, %82 : f16
            linalg.yield %83 : f16
          }
          loom.semaphore_give %53 : memref<1x32x32xf16>
          loom.semaphore_give %40 : memref<1x32x128xf16>
          linalg.copy ins(%42 : memref<1x32x1xf16>) outs(%38 : memref<1x32x1xf16>)
          loom.semaphore_give %42 : memref<1x32x1xf16>
          %64 = loom.alloc [1, 32, 1] on @L1 : memref<1x32x1xf16>
          %65 = loom.semaphore_take %64 : memref<1x32x1xf16> -> memref<1x32x1xf16>
          linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%36, %38 : memref<1x32x1xf16>, memref<1x32x1xf16>) outs(%65 : memref<1x32x1xf16>) {
          ^bb0(%in: f16, %in_5: f16, %out: f16):
            %82 = math.log %in : f16
            %83 = arith.addf %82, %in_5 : f16
            linalg.yield %83 : f16
          }
          loom.semaphore_give %38 : memref<1x32x1xf16>
          %66 = loom.broadcast ins(%36 : memref<1x32x1xf16>) outs(%52 : memref<1x32x32xf16>) dim(2) -> memref<1x32x128xf16, strided<[?, ?, ?], offset: ?>>
          loom.semaphore_give %36 : memref<1x32x1xf16>
          %67 = loom.alloc [1, 32, 128] on @L1 : memref<1x32x128xf16>
          %68 = loom.semaphore_take %67 : memref<1x32x128xf16> -> memref<1x32x128xf16>
          linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%32, %66 : memref<1x32x128xf16>, memref<1x32x128xf16, strided<[?, ?, ?], offset: ?>>) outs(%68 : memref<1x32x128xf16>) {
          ^bb0(%in: f16, %in_5: f16, %out: f16):
            %82 = arith.divf %in, %in_5 : f16
            linalg.yield %82 : f16
          }
          loom.semaphore_give %52 : memref<1x32x32xf16>
          loom.semaphore_give %32 : memref<1x32x128xf16>
          %69 = loom.alloc [16, 1, 32, 1] on @L1 : memref<16x1x32x1xf16>
          %70 = loom.semaphore_take %69 : memref<16x1x32x1xf16> -> memref<16x1x32x1xf16>
          loom.gather ins(%65 : memref<1x32x1xf16>) outs(%70 : memref<16x1x32x1xf16>) across(%arg5 : index) region : (UL : [%27, %arg6], LR : [%28, %arg6])
          loom.semaphore_give %65 : memref<1x32x1xf16>
          %71 = loom.alloc [16, 1, 32, 128] on @L1 : memref<16x1x32x128xf16>
          %72 = loom.semaphore_take %71 : memref<16x1x32x128xf16> -> memref<16x1x32x128xf16>
          loom.gather ins(%68 : memref<1x32x128xf16>) outs(%72 : memref<16x1x32x128xf16>) across(%arg5 : index) region : (UL : [%27, %arg6], LR : [%28, %arg6])
          loom.semaphore_give %68 : memref<1x32x128xf16>
          %73 = loom.alloc [1, 32, 128] on @L1 : memref<1x32x128xf16>
          %74 = loom.semaphore_take %73 : memref<1x32x128xf16> -> memref<1x32x128xf16>
          %75 = loom.alloc [16, 1, 32, 1] on @L1 : memref<16x1x32x1xf16>
          %76 = loom.semaphore_take %75 : memref<16x1x32x1xf16> -> memref<16x1x32x1xf16>
          %77 = loom.alloc [16, 1, 32, 128] on @L1 : memref<16x1x32x128xf16>
          %78 = loom.semaphore_take %77 : memref<16x1x32x128xf16> -> memref<16x1x32x128xf16>
          %79 = loom.alloc [16, 1, 32, 32] on @L1 : memref<16x1x32x32xf16>
          %80 = loom.semaphore_take %79 : memref<16x1x32x32xf16> -> memref<16x1x32x32xf16>
          %81 = arith.cmpi eq, %arg5, %c0 : index
          scf.if %81 {
            linalg.fill ins(%cst_1 : f16) outs(%35 : memref<1x32x1xf16>)
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1, d2, d3)>], iterator_types = ["reduction", "parallel", "parallel", "parallel"]} ins(%70 : memref<16x1x32x1xf16>) outs(%35 : memref<1x32x1xf16>) {
            ^bb0(%in: f16, %out: f16):
              %83 = arith.maximumf %in, %out : f16
              linalg.yield %83 : f16
            }
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%70, %35 : memref<16x1x32x1xf16>, memref<1x32x1xf16>) outs(%76 : memref<16x1x32x1xf16>) {
            ^bb0(%in: f16, %in_6: f16, %out: f16):
              %83 = arith.subf %in, %in_6 : f16
              %84 = math.exp %83 : f16
              linalg.yield %84 : f16
            }
            loom.semaphore_give %70 : memref<16x1x32x1xf16>
            loom.semaphore_give %35 : memref<1x32x1xf16>
            linalg.fill ins(%cst : f16) outs(%34 : memref<1x32x1xf16>)
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1, d2, d3)>], iterator_types = ["reduction", "parallel", "parallel", "parallel"]} ins(%76 : memref<16x1x32x1xf16>) outs(%34 : memref<1x32x1xf16>) {
            ^bb0(%in: f16, %out: f16):
              %83 = arith.addf %in, %out : f16
              linalg.yield %83 : f16
            }
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%76, %34 : memref<16x1x32x1xf16>, memref<1x32x1xf16>) outs(%76 : memref<16x1x32x1xf16>) {
            ^bb0(%in: f16, %in_6: f16, %out: f16):
              %83 = arith.divf %in, %in_6 : f16
              linalg.yield %83 : f16
            }
            loom.semaphore_give %34 : memref<1x32x1xf16>
            %82 = loom.broadcast ins(%76 : memref<16x1x32x1xf16>) outs(%80 : memref<16x1x32x32xf16>) dim(3) -> memref<16x1x32x128xf16, strided<[?, ?, ?, ?], offset: ?>>
            loom.semaphore_give %76 : memref<16x1x32x1xf16>
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%72, %82 : memref<16x1x32x128xf16>, memref<16x1x32x128xf16, strided<[?, ?, ?, ?], offset: ?>>) outs(%78 : memref<16x1x32x128xf16>) {
            ^bb0(%in: f16, %in_6: f16, %out: f16):
              %83 = arith.mulf %in, %in_6 : f16
              linalg.yield %83 : f16
            }
            loom.semaphore_give %80 : memref<16x1x32x32xf16>
            loom.semaphore_give %72 : memref<16x1x32x128xf16>
            linalg.fill ins(%cst : f16) outs(%74 : memref<1x32x128xf16>)
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1, d2, d3)>], iterator_types = ["reduction", "parallel", "parallel", "parallel"]} ins(%78 : memref<16x1x32x128xf16>) outs(%74 : memref<1x32x128xf16>) {
            ^bb0(%in: f16, %out: f16):
              %83 = arith.addf %in, %out : f16
              linalg.yield %83 : f16
            }
            loom.semaphore_give %78 : memref<16x1x32x128xf16>
            loom.semaphore_give %31 : memref<1x32x128xf16>
            %reinterpret_cast_5 = memref.reinterpret_cast %arg2 to offset: [%26], sizes: [1, 32, 128], strides: [4096, 128, 1] : memref<16x32x128xf16> to memref<1x32x128xf16, strided<[4096, 128, 1], offset: ?>>
            loom.copy %74, %reinterpret_cast_5 src_mem_space @mem_L1 dst_mem_space @mem_DRAM, broadcast : [1, 1] region : (UL : [%59, %arg6], LR : [%59, %arg6]) : memref<1x32x128xf16> to memref<1x32x128xf16, strided<[4096, 128, 1], offset: ?>>
            loom.semaphore_give %74 : memref<1x32x128xf16>
          }
        } {loom.block_sym = @tile_b, loom.iter_type = #loom.iter_type<temporal>}
        scf.reduce 
      } {loom.iter_types = [#loom.iter_type<spatial>, #loom.iter_type<spatial>, #loom.iter_type<spatial>], loom.logical_levels = [1, 0, 0], loom.physical_dims = [@dim_x, @dim_x, @dim_y]}
      return
    }
  }
  module attributes {loom.pass_name = "Materialize", loom.tile_b = {is_reduction = false, upper_bound = 16 : index}, loom.tile_n = {is_reduction = false, upper_bound = 64 : index}, loom.tile_s = {is_reduction = false, upper_bound = 8192 : index}} {
    func.func @flash_decode__x4x2_y8__d0i1_d1i1_d2i0__f01__dim_x_level0_bc4_dim_y_level0_bc8_n_n_n__tile_b1__tile_n512__tile_s512(%arg0: memref<16x128x8192xf16>, %arg1: memref<16x8192x128xf16>, %arg2: memref<16x32x128xf16>, %arg3: memref<16x32x128xf16>) {
      %c65536 = arith.constant 65536 : index
      %c1048576 = arith.constant 1048576 : index
      %c4096 = arith.constant 4096 : index
      %c8 = arith.constant 8 : index
      %c7 = arith.constant 7 : index
      %c3 = arith.constant 3 : index
      %c4 = arith.constant 4 : index
      %c0 = arith.constant 0 : index
      %cst = arith.constant 0.000000e+00 : f16
      %cst_0 = arith.constant 1.000000e+00 : f16
      %cst_1 = arith.constant 0xFC00 : f16
      %c1 = arith.constant 1 : index
      %cst_2 = arith.constant 8.837890e-02 : f16
      %c512 = arith.constant 512 : index
      %c2 = arith.constant 2 : index
      scf.parallel (%arg4, %arg5, %arg6) = (%c0, %c0, %c0) to (%c2, %c4, %c8) step (%c1, %c1, %c1) {
        scf.for %arg7 = %c0 to %c8 step %c1 {
          %20 = arith.muli %arg7, %c2 overflow<nsw> : index
          %21 = arith.addi %arg4, %20 : index
          %22 = arith.muli %arg5, %c4 overflow<nsw> : index
          %23 = arith.addi %22, %arg6 : index
          %24 = loom.alloc [1, 32, 128] on @L1 : memref<1x32x128xf16>
          %25 = loom.semaphore_take %24 : memref<1x32x128xf16> -> memref<1x32x128xf16>
          %26 = arith.muli %21, %c4096 overflow<nsw> : index
          %reinterpret_cast = memref.reinterpret_cast %arg3 to offset: [%26], sizes: [1, 32, 128], strides: [4096, 128, 1] : memref<16x32x128xf16> to memref<1x32x128xf16, strided<[4096, 128, 1], offset: ?>>
          %27 = arith.muli %arg4, %c4 : index
          %28 = arith.addi %27, %c3 : index
          loom.copy %reinterpret_cast, %25 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [4, 8] region : (UL : [%27, %c0], LR : [%28, %c7]) : memref<1x32x128xf16, strided<[4096, 128, 1], offset: ?>> to memref<1x32x128xf16>
          %29 = arith.muli %23, %c512 : index
          %30 = loom.alloc [1, 32, 128] on @L1 : memref<1x32x128xf16>
          %31 = loom.semaphore_take %30 : memref<1x32x128xf16> -> memref<1x32x128xf16>
          %32 = loom.semaphore_take %30 : memref<1x32x128xf16> -> memref<1x32x128xf16>
          linalg.fill ins(%cst : f16) outs(%32 : memref<1x32x128xf16>)
          %33 = loom.alloc [1, 32, 1] on @L1 : memref<1x32x1xf16>
          %34 = loom.semaphore_take %33 : memref<1x32x1xf16> -> memref<1x32x1xf16>
          %35 = loom.semaphore_take %33 : memref<1x32x1xf16> -> memref<1x32x1xf16>
          %36 = loom.semaphore_take %33 : memref<1x32x1xf16> -> memref<1x32x1xf16>
          linalg.fill ins(%cst_0 : f16) outs(%36 : memref<1x32x1xf16>)
          %37 = loom.alloc [1, 32, 1] on @L1 : memref<1x32x1xf16>
          %38 = loom.semaphore_take %37 : memref<1x32x1xf16> -> memref<1x32x1xf16>
          linalg.fill ins(%cst_1 : f16) outs(%38 : memref<1x32x1xf16>)
          %39 = loom.alloc [1, 32, 128] on @L1 : memref<1x32x128xf16>
          %40 = loom.semaphore_take %39 : memref<1x32x128xf16> -> memref<1x32x128xf16>
          %41 = loom.alloc [1, 32, 1] on @L1 : memref<1x32x1xf16>
          %42 = loom.semaphore_take %41 : memref<1x32x1xf16> -> memref<1x32x1xf16>
          %43 = loom.alloc [1, 32, 1] on @L1 : memref<1x32x1xf16>
          %44 = loom.semaphore_take %43 : memref<1x32x1xf16> -> memref<1x32x1xf16>
          %45 = loom.alloc [1, 32, 1] on @L1 : memref<1x32x1xf16>
          %46 = loom.semaphore_take %45 : memref<1x32x1xf16> -> memref<1x32x1xf16>
          %47 = loom.alloc [1, 128, 512] on @L1 : memref<1x128x512xf16>
          %48 = loom.semaphore_take %47 : memref<1x128x512xf16> -> memref<1x128x512xf16>
          %49 = loom.alloc [1, 32, 512] on @L1 : memref<1x32x512xf16>
          %50 = loom.semaphore_take %49 : memref<1x32x512xf16> -> memref<1x32x512xf16>
          %51 = loom.alloc [1, 32, 32] on @L1 : memref<1x32x32xf16>
          %52 = loom.semaphore_take %51 : memref<1x32x32xf16> -> memref<1x32x32xf16>
          %53 = loom.semaphore_take %51 : memref<1x32x32xf16> -> memref<1x32x32xf16>
          %54 = loom.semaphore_take %51 : memref<1x32x32xf16> -> memref<1x32x32xf16>
          %55 = loom.alloc [1, 512, 128] on @L1 : memref<1x512x128xf16>
          %56 = loom.semaphore_take %55 : memref<1x512x128xf16> -> memref<1x512x128xf16>
          %57 = arith.muli %21, %c1048576 overflow<nsw> : index
          %58 = arith.addi %57, %29 : index
          %reinterpret_cast_3 = memref.reinterpret_cast %arg0 to offset: [%58], sizes: [1, 128, 512], strides: [1048576, 8192, 1] : memref<16x128x8192xf16> to memref<1x128x512xf16, strided<[1048576, 8192, 1], offset: ?>>
          %59 = arith.addi %arg5, %27 : index
          loom.copy %reinterpret_cast_3, %48 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] region : (UL : [%59, %arg6], LR : [%59, %arg6]) : memref<1x128x512xf16, strided<[1048576, 8192, 1], offset: ?>> to memref<1x128x512xf16>
          loom.batch_matmul ins(%25, %48 : memref<1x32x128xf16>, memref<1x128x512xf16>) outs(%50 : memref<1x32x512xf16>)
          loom.semaphore_give %48 : memref<1x128x512xf16>
          loom.semaphore_give %25 : memref<1x32x128xf16>
          linalg.fill ins(%cst_1 : f16) outs(%42 : memref<1x32x1xf16>)
          linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, 0)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%50 : memref<1x32x512xf16>) outs(%42 : memref<1x32x1xf16>) {
          ^bb0(%in: f16, %out: f16):
            %82 = arith.maximumf %in, %out : f16
            linalg.yield %82 : f16
          }
          linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%38, %42 : memref<1x32x1xf16>, memref<1x32x1xf16>) outs(%42 : memref<1x32x1xf16>) {
          ^bb0(%in: f16, %in_5: f16, %out: f16):
            %82 = arith.mulf %in_5, %cst_2 : f16
            %83 = arith.cmpf ogt, %in, %82 : f16
            %84 = arith.select %83, %in, %82 : f16
            linalg.yield %84 : f16
          }
          %60 = loom.broadcast ins(%42 : memref<1x32x1xf16>) outs(%54 : memref<1x32x32xf16>) dim(2) -> memref<1x32x512xf16, strided<[?, ?, ?], offset: ?>>
          linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%50, %60 : memref<1x32x512xf16>, memref<1x32x512xf16, strided<[?, ?, ?], offset: ?>>) outs(%50 : memref<1x32x512xf16>) {
          ^bb0(%in: f16, %in_5: f16, %out: f16):
            %82 = arith.mulf %in, %cst_2 : f16
            %83 = arith.subf %82, %in_5 : f16
            %84 = math.exp %83 : f16
            linalg.yield %84 : f16
          }
          loom.semaphore_give %54 : memref<1x32x32xf16>
          linalg.fill ins(%cst : f16) outs(%44 : memref<1x32x1xf16>)
          linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, 0)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%50 : memref<1x32x512xf16>) outs(%44 : memref<1x32x1xf16>) {
          ^bb0(%in: f16, %out: f16):
            %82 = arith.addf %in, %out : f16
            linalg.yield %82 : f16
          }
          linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%38, %42 : memref<1x32x1xf16>, memref<1x32x1xf16>) outs(%46 : memref<1x32x1xf16>) {
          ^bb0(%in: f16, %in_5: f16, %out: f16):
            %82 = arith.subf %in, %in_5 : f16
            %83 = math.exp %82 : f16
            linalg.yield %83 : f16
          }
          linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%36, %46, %44 : memref<1x32x1xf16>, memref<1x32x1xf16>, memref<1x32x1xf16>) outs(%36 : memref<1x32x1xf16>) {
          ^bb0(%in: f16, %in_5: f16, %in_6: f16, %out: f16):
            %82 = arith.mulf %in, %in_5 : f16
            %83 = arith.addf %82, %in_6 : f16
            linalg.yield %83 : f16
          }
          loom.semaphore_give %44 : memref<1x32x1xf16>
          %61 = loom.broadcast ins(%46 : memref<1x32x1xf16>) outs(%53 : memref<1x32x32xf16>) dim(2) -> memref<1x32x128xf16, strided<[?, ?, ?], offset: ?>>
          loom.semaphore_give %46 : memref<1x32x1xf16>
          %62 = arith.muli %23, %c65536 : index
          %63 = arith.addi %57, %62 : index
          %reinterpret_cast_4 = memref.reinterpret_cast %arg1 to offset: [%63], sizes: [1, 512, 128], strides: [1048576, 128, 1] : memref<16x8192x128xf16> to memref<1x512x128xf16, strided<[1048576, 128, 1], offset: ?>>
          loom.copy %reinterpret_cast_4, %56 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] region : (UL : [%59, %arg6], LR : [%59, %arg6]) : memref<1x512x128xf16, strided<[1048576, 128, 1], offset: ?>> to memref<1x512x128xf16>
          loom.batch_matmul ins(%50, %56 : memref<1x32x512xf16>, memref<1x512x128xf16>) outs(%40 : memref<1x32x128xf16>)
          loom.semaphore_give %56 : memref<1x512x128xf16>
          loom.semaphore_give %50 : memref<1x32x512xf16>
          linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%40, %32, %61 : memref<1x32x128xf16>, memref<1x32x128xf16>, memref<1x32x128xf16, strided<[?, ?, ?], offset: ?>>) outs(%32 : memref<1x32x128xf16>) {
          ^bb0(%in: f16, %in_5: f16, %in_6: f16, %out: f16):
            %82 = arith.mulf %in_5, %in_6 : f16
            %83 = arith.addf %in, %82 : f16
            linalg.yield %83 : f16
          }
          loom.semaphore_give %53 : memref<1x32x32xf16>
          loom.semaphore_give %40 : memref<1x32x128xf16>
          linalg.copy ins(%42 : memref<1x32x1xf16>) outs(%38 : memref<1x32x1xf16>)
          loom.semaphore_give %42 : memref<1x32x1xf16>
          %64 = loom.alloc [1, 32, 1] on @L1 : memref<1x32x1xf16>
          %65 = loom.semaphore_take %64 : memref<1x32x1xf16> -> memref<1x32x1xf16>
          linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%36, %38 : memref<1x32x1xf16>, memref<1x32x1xf16>) outs(%65 : memref<1x32x1xf16>) {
          ^bb0(%in: f16, %in_5: f16, %out: f16):
            %82 = math.log %in : f16
            %83 = arith.addf %82, %in_5 : f16
            linalg.yield %83 : f16
          }
          loom.semaphore_give %38 : memref<1x32x1xf16>
          %66 = loom.broadcast ins(%36 : memref<1x32x1xf16>) outs(%52 : memref<1x32x32xf16>) dim(2) -> memref<1x32x128xf16, strided<[?, ?, ?], offset: ?>>
          loom.semaphore_give %36 : memref<1x32x1xf16>
          %67 = loom.alloc [1, 32, 128] on @L1 : memref<1x32x128xf16>
          %68 = loom.semaphore_take %67 : memref<1x32x128xf16> -> memref<1x32x128xf16>
          linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%32, %66 : memref<1x32x128xf16>, memref<1x32x128xf16, strided<[?, ?, ?], offset: ?>>) outs(%68 : memref<1x32x128xf16>) {
          ^bb0(%in: f16, %in_5: f16, %out: f16):
            %82 = arith.divf %in, %in_5 : f16
            linalg.yield %82 : f16
          }
          loom.semaphore_give %52 : memref<1x32x32xf16>
          loom.semaphore_give %32 : memref<1x32x128xf16>
          %69 = loom.alloc [16, 1, 32, 1] on @L1 : memref<16x1x32x1xf16>
          %70 = loom.semaphore_take %69 : memref<16x1x32x1xf16> -> memref<16x1x32x1xf16>
          loom.gather ins(%65 : memref<1x32x1xf16>) outs(%70 : memref<16x1x32x1xf16>) across(%arg5 : index) region : (UL : [%27, %arg6], LR : [%28, %arg6])
          loom.semaphore_give %65 : memref<1x32x1xf16>
          %71 = loom.alloc [16, 1, 32, 128] on @L1 : memref<16x1x32x128xf16>
          %72 = loom.semaphore_take %71 : memref<16x1x32x128xf16> -> memref<16x1x32x128xf16>
          loom.gather ins(%68 : memref<1x32x128xf16>) outs(%72 : memref<16x1x32x128xf16>) across(%arg5 : index) region : (UL : [%27, %arg6], LR : [%28, %arg6])
          loom.semaphore_give %68 : memref<1x32x128xf16>
          %73 = loom.alloc [1, 32, 128] on @L1 : memref<1x32x128xf16>
          %74 = loom.semaphore_take %73 : memref<1x32x128xf16> -> memref<1x32x128xf16>
          %75 = loom.alloc [16, 1, 32, 1] on @L1 : memref<16x1x32x1xf16>
          %76 = loom.semaphore_take %75 : memref<16x1x32x1xf16> -> memref<16x1x32x1xf16>
          %77 = loom.alloc [16, 1, 32, 128] on @L1 : memref<16x1x32x128xf16>
          %78 = loom.semaphore_take %77 : memref<16x1x32x128xf16> -> memref<16x1x32x128xf16>
          %79 = loom.alloc [16, 1, 32, 32] on @L1 : memref<16x1x32x32xf16>
          %80 = loom.semaphore_take %79 : memref<16x1x32x32xf16> -> memref<16x1x32x32xf16>
          %81 = arith.cmpi eq, %arg5, %c0 : index
          scf.if %81 {
            linalg.fill ins(%cst_1 : f16) outs(%35 : memref<1x32x1xf16>)
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1, d2, d3)>], iterator_types = ["reduction", "parallel", "parallel", "parallel"]} ins(%70 : memref<16x1x32x1xf16>) outs(%35 : memref<1x32x1xf16>) {
            ^bb0(%in: f16, %out: f16):
              %83 = arith.maximumf %in, %out : f16
              linalg.yield %83 : f16
            }
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%70, %35 : memref<16x1x32x1xf16>, memref<1x32x1xf16>) outs(%76 : memref<16x1x32x1xf16>) {
            ^bb0(%in: f16, %in_6: f16, %out: f16):
              %83 = arith.subf %in, %in_6 : f16
              %84 = math.exp %83 : f16
              linalg.yield %84 : f16
            }
            loom.semaphore_give %70 : memref<16x1x32x1xf16>
            loom.semaphore_give %35 : memref<1x32x1xf16>
            linalg.fill ins(%cst : f16) outs(%34 : memref<1x32x1xf16>)
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1, d2, d3)>], iterator_types = ["reduction", "parallel", "parallel", "parallel"]} ins(%76 : memref<16x1x32x1xf16>) outs(%34 : memref<1x32x1xf16>) {
            ^bb0(%in: f16, %out: f16):
              %83 = arith.addf %in, %out : f16
              linalg.yield %83 : f16
            }
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%76, %34 : memref<16x1x32x1xf16>, memref<1x32x1xf16>) outs(%76 : memref<16x1x32x1xf16>) {
            ^bb0(%in: f16, %in_6: f16, %out: f16):
              %83 = arith.divf %in, %in_6 : f16
              linalg.yield %83 : f16
            }
            loom.semaphore_give %34 : memref<1x32x1xf16>
            %82 = loom.broadcast ins(%76 : memref<16x1x32x1xf16>) outs(%80 : memref<16x1x32x32xf16>) dim(3) -> memref<16x1x32x128xf16, strided<[?, ?, ?, ?], offset: ?>>
            loom.semaphore_give %76 : memref<16x1x32x1xf16>
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%72, %82 : memref<16x1x32x128xf16>, memref<16x1x32x128xf16, strided<[?, ?, ?, ?], offset: ?>>) outs(%78 : memref<16x1x32x128xf16>) {
            ^bb0(%in: f16, %in_6: f16, %out: f16):
              %83 = arith.mulf %in, %in_6 : f16
              linalg.yield %83 : f16
            }
            loom.semaphore_give %80 : memref<16x1x32x32xf16>
            loom.semaphore_give %72 : memref<16x1x32x128xf16>
            linalg.fill ins(%cst : f16) outs(%74 : memref<1x32x128xf16>)
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1, d2, d3)>], iterator_types = ["reduction", "parallel", "parallel", "parallel"]} ins(%78 : memref<16x1x32x128xf16>) outs(%74 : memref<1x32x128xf16>) {
            ^bb0(%in: f16, %out: f16):
              %83 = arith.addf %in, %out : f16
              linalg.yield %83 : f16
            }
            loom.semaphore_give %78 : memref<16x1x32x128xf16>
            loom.semaphore_give %31 : memref<1x32x128xf16>
            %reinterpret_cast_5 = memref.reinterpret_cast %arg2 to offset: [%26], sizes: [1, 32, 128], strides: [4096, 128, 1] : memref<16x32x128xf16> to memref<1x32x128xf16, strided<[4096, 128, 1], offset: ?>>
            loom.copy %74, %reinterpret_cast_5 src_mem_space @mem_L1 dst_mem_space @mem_DRAM, broadcast : [1, 1] region : (UL : [%59, %arg6], LR : [%59, %arg6]) : memref<1x32x128xf16> to memref<1x32x128xf16, strided<[4096, 128, 1], offset: ?>>
            loom.semaphore_give %74 : memref<1x32x128xf16>
          }
        } {loom.block_sym = @tile_b, loom.iter_type = #loom.iter_type<temporal>}
        scf.reduce 
      } {loom.iter_types = [#loom.iter_type<spatial>, #loom.iter_type<spatial>, #loom.iter_type<spatial>], loom.logical_levels = [1, 0, 0], loom.physical_dims = [@dim_x, @dim_x, @dim_y]}
      return
    }
  }
  module attributes {loom.pass_name = "Materialize", loom.tile_b = {is_reduction = false, upper_bound = 16 : index}, loom.tile_n = {is_reduction = false, upper_bound = 64 : index}, loom.tile_s = {is_reduction = false, upper_bound = 8192 : index}} {
    func.func @flash_decode__x8x1_y8__d0i1_d1i1_d2i0__f01__dim_x_level1_bc8_dim_y_level0_bc8_n_n_n__tile_b1__tile_n512__tile_s512(%arg0: memref<16x128x8192xf16>, %arg1: memref<16x8192x128xf16>, %arg2: memref<16x32x128xf16>, %arg3: memref<16x32x128xf16>) {
      %c65536 = arith.constant 65536 : index
      %c1048576 = arith.constant 1048576 : index
      %c4096 = arith.constant 4096 : index
      %c7 = arith.constant 7 : index
      %c8 = arith.constant 8 : index
      %c0 = arith.constant 0 : index
      %cst = arith.constant 0.000000e+00 : f16
      %cst_0 = arith.constant 1.000000e+00 : f16
      %cst_1 = arith.constant 0xFC00 : f16
      %c1 = arith.constant 1 : index
      %cst_2 = arith.constant 8.837890e-02 : f16
      %c16 = arith.constant 16 : index
      %c512 = arith.constant 512 : index
      scf.parallel (%arg4, %arg5) = (%c0, %c0) to (%c8, %c8) step (%c1, %c1) {
        scf.for %arg6 = %c0 to %c16 step %c1 {
          %20 = arith.muli %arg4, %c8 overflow<nsw> : index
          %21 = arith.addi %20, %arg5 : index
          %22 = loom.alloc [1, 32, 128] on @L1 : memref<1x32x128xf16>
          %23 = loom.semaphore_take %22 : memref<1x32x128xf16> -> memref<1x32x128xf16>
          %24 = arith.muli %arg6, %c4096 overflow<nsw> : index
          %reinterpret_cast = memref.reinterpret_cast %arg3 to offset: [%24], sizes: [1, 32, 128], strides: [4096, 128, 1] : memref<16x32x128xf16> to memref<1x32x128xf16, strided<[4096, 128, 1], offset: ?>>
          loom.copy %reinterpret_cast, %23 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [8, 8] region : (UL : [%c0, %c0], LR : [%c7, %c7]) : memref<1x32x128xf16, strided<[4096, 128, 1], offset: ?>> to memref<1x32x128xf16>
          %25 = arith.muli %21, %c512 : index
          %26 = loom.alloc [1, 32, 128] on @L1 : memref<1x32x128xf16>
          %27 = loom.semaphore_take %26 : memref<1x32x128xf16> -> memref<1x32x128xf16>
          %28 = loom.semaphore_take %26 : memref<1x32x128xf16> -> memref<1x32x128xf16>
          linalg.fill ins(%cst : f16) outs(%28 : memref<1x32x128xf16>)
          %29 = loom.alloc [1, 32, 1] on @L1 : memref<1x32x1xf16>
          %30 = loom.semaphore_take %29 : memref<1x32x1xf16> -> memref<1x32x1xf16>
          %31 = loom.semaphore_take %29 : memref<1x32x1xf16> -> memref<1x32x1xf16>
          %32 = loom.semaphore_take %29 : memref<1x32x1xf16> -> memref<1x32x1xf16>
          linalg.fill ins(%cst_0 : f16) outs(%32 : memref<1x32x1xf16>)
          %33 = loom.alloc [1, 32, 1] on @L1 : memref<1x32x1xf16>
          %34 = loom.semaphore_take %33 : memref<1x32x1xf16> -> memref<1x32x1xf16>
          linalg.fill ins(%cst_1 : f16) outs(%34 : memref<1x32x1xf16>)
          %35 = loom.alloc [1, 32, 128] on @L1 : memref<1x32x128xf16>
          %36 = loom.semaphore_take %35 : memref<1x32x128xf16> -> memref<1x32x128xf16>
          %37 = loom.alloc [1, 32, 1] on @L1 : memref<1x32x1xf16>
          %38 = loom.semaphore_take %37 : memref<1x32x1xf16> -> memref<1x32x1xf16>
          %39 = loom.alloc [1, 32, 1] on @L1 : memref<1x32x1xf16>
          %40 = loom.semaphore_take %39 : memref<1x32x1xf16> -> memref<1x32x1xf16>
          %41 = loom.alloc [1, 32, 1] on @L1 : memref<1x32x1xf16>
          %42 = loom.semaphore_take %41 : memref<1x32x1xf16> -> memref<1x32x1xf16>
          %43 = loom.alloc [1, 128, 512] on @L1 : memref<1x128x512xf16>
          %44 = loom.semaphore_take %43 : memref<1x128x512xf16> -> memref<1x128x512xf16>
          %45 = loom.alloc [1, 32, 512] on @L1 : memref<1x32x512xf16>
          %46 = loom.semaphore_take %45 : memref<1x32x512xf16> -> memref<1x32x512xf16>
          %47 = loom.alloc [1, 32, 32] on @L1 : memref<1x32x32xf16>
          %48 = loom.semaphore_take %47 : memref<1x32x32xf16> -> memref<1x32x32xf16>
          %49 = loom.semaphore_take %47 : memref<1x32x32xf16> -> memref<1x32x32xf16>
          %50 = loom.semaphore_take %47 : memref<1x32x32xf16> -> memref<1x32x32xf16>
          %51 = loom.alloc [1, 512, 128] on @L1 : memref<1x512x128xf16>
          %52 = loom.semaphore_take %51 : memref<1x512x128xf16> -> memref<1x512x128xf16>
          %53 = arith.muli %arg6, %c1048576 overflow<nsw> : index
          %54 = arith.addi %53, %25 : index
          %reinterpret_cast_3 = memref.reinterpret_cast %arg0 to offset: [%54], sizes: [1, 128, 512], strides: [1048576, 8192, 1] : memref<16x128x8192xf16> to memref<1x128x512xf16, strided<[1048576, 8192, 1], offset: ?>>
          loom.copy %reinterpret_cast_3, %44 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] region : (UL : [%arg4, %arg5], LR : [%arg4, %arg5]) : memref<1x128x512xf16, strided<[1048576, 8192, 1], offset: ?>> to memref<1x128x512xf16>
          loom.batch_matmul ins(%23, %44 : memref<1x32x128xf16>, memref<1x128x512xf16>) outs(%46 : memref<1x32x512xf16>)
          loom.semaphore_give %44 : memref<1x128x512xf16>
          loom.semaphore_give %23 : memref<1x32x128xf16>
          linalg.fill ins(%cst_1 : f16) outs(%38 : memref<1x32x1xf16>)
          linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, 0)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%46 : memref<1x32x512xf16>) outs(%38 : memref<1x32x1xf16>) {
          ^bb0(%in: f16, %out: f16):
            %77 = arith.maximumf %in, %out : f16
            linalg.yield %77 : f16
          }
          linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%34, %38 : memref<1x32x1xf16>, memref<1x32x1xf16>) outs(%38 : memref<1x32x1xf16>) {
          ^bb0(%in: f16, %in_5: f16, %out: f16):
            %77 = arith.mulf %in_5, %cst_2 : f16
            %78 = arith.cmpf ogt, %in, %77 : f16
            %79 = arith.select %78, %in, %77 : f16
            linalg.yield %79 : f16
          }
          %55 = loom.broadcast ins(%38 : memref<1x32x1xf16>) outs(%50 : memref<1x32x32xf16>) dim(2) -> memref<1x32x512xf16, strided<[?, ?, ?], offset: ?>>
          linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%46, %55 : memref<1x32x512xf16>, memref<1x32x512xf16, strided<[?, ?, ?], offset: ?>>) outs(%46 : memref<1x32x512xf16>) {
          ^bb0(%in: f16, %in_5: f16, %out: f16):
            %77 = arith.mulf %in, %cst_2 : f16
            %78 = arith.subf %77, %in_5 : f16
            %79 = math.exp %78 : f16
            linalg.yield %79 : f16
          }
          loom.semaphore_give %50 : memref<1x32x32xf16>
          linalg.fill ins(%cst : f16) outs(%40 : memref<1x32x1xf16>)
          linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, 0)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%46 : memref<1x32x512xf16>) outs(%40 : memref<1x32x1xf16>) {
          ^bb0(%in: f16, %out: f16):
            %77 = arith.addf %in, %out : f16
            linalg.yield %77 : f16
          }
          linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%34, %38 : memref<1x32x1xf16>, memref<1x32x1xf16>) outs(%42 : memref<1x32x1xf16>) {
          ^bb0(%in: f16, %in_5: f16, %out: f16):
            %77 = arith.subf %in, %in_5 : f16
            %78 = math.exp %77 : f16
            linalg.yield %78 : f16
          }
          linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%32, %42, %40 : memref<1x32x1xf16>, memref<1x32x1xf16>, memref<1x32x1xf16>) outs(%32 : memref<1x32x1xf16>) {
          ^bb0(%in: f16, %in_5: f16, %in_6: f16, %out: f16):
            %77 = arith.mulf %in, %in_5 : f16
            %78 = arith.addf %77, %in_6 : f16
            linalg.yield %78 : f16
          }
          loom.semaphore_give %40 : memref<1x32x1xf16>
          %56 = loom.broadcast ins(%42 : memref<1x32x1xf16>) outs(%49 : memref<1x32x32xf16>) dim(2) -> memref<1x32x128xf16, strided<[?, ?, ?], offset: ?>>
          loom.semaphore_give %42 : memref<1x32x1xf16>
          %57 = arith.muli %21, %c65536 : index
          %58 = arith.addi %53, %57 : index
          %reinterpret_cast_4 = memref.reinterpret_cast %arg1 to offset: [%58], sizes: [1, 512, 128], strides: [1048576, 128, 1] : memref<16x8192x128xf16> to memref<1x512x128xf16, strided<[1048576, 128, 1], offset: ?>>
          loom.copy %reinterpret_cast_4, %52 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] region : (UL : [%arg4, %arg5], LR : [%arg4, %arg5]) : memref<1x512x128xf16, strided<[1048576, 128, 1], offset: ?>> to memref<1x512x128xf16>
          loom.batch_matmul ins(%46, %52 : memref<1x32x512xf16>, memref<1x512x128xf16>) outs(%36 : memref<1x32x128xf16>)
          loom.semaphore_give %52 : memref<1x512x128xf16>
          loom.semaphore_give %46 : memref<1x32x512xf16>
          linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%36, %28, %56 : memref<1x32x128xf16>, memref<1x32x128xf16>, memref<1x32x128xf16, strided<[?, ?, ?], offset: ?>>) outs(%28 : memref<1x32x128xf16>) {
          ^bb0(%in: f16, %in_5: f16, %in_6: f16, %out: f16):
            %77 = arith.mulf %in_5, %in_6 : f16
            %78 = arith.addf %in, %77 : f16
            linalg.yield %78 : f16
          }
          loom.semaphore_give %49 : memref<1x32x32xf16>
          loom.semaphore_give %36 : memref<1x32x128xf16>
          linalg.copy ins(%38 : memref<1x32x1xf16>) outs(%34 : memref<1x32x1xf16>)
          loom.semaphore_give %38 : memref<1x32x1xf16>
          %59 = loom.alloc [1, 32, 1] on @L1 : memref<1x32x1xf16>
          %60 = loom.semaphore_take %59 : memref<1x32x1xf16> -> memref<1x32x1xf16>
          linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%32, %34 : memref<1x32x1xf16>, memref<1x32x1xf16>) outs(%60 : memref<1x32x1xf16>) {
          ^bb0(%in: f16, %in_5: f16, %out: f16):
            %77 = math.log %in : f16
            %78 = arith.addf %77, %in_5 : f16
            linalg.yield %78 : f16
          }
          loom.semaphore_give %34 : memref<1x32x1xf16>
          %61 = loom.broadcast ins(%32 : memref<1x32x1xf16>) outs(%48 : memref<1x32x32xf16>) dim(2) -> memref<1x32x128xf16, strided<[?, ?, ?], offset: ?>>
          loom.semaphore_give %32 : memref<1x32x1xf16>
          %62 = loom.alloc [1, 32, 128] on @L1 : memref<1x32x128xf16>
          %63 = loom.semaphore_take %62 : memref<1x32x128xf16> -> memref<1x32x128xf16>
          linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%28, %61 : memref<1x32x128xf16>, memref<1x32x128xf16, strided<[?, ?, ?], offset: ?>>) outs(%63 : memref<1x32x128xf16>) {
          ^bb0(%in: f16, %in_5: f16, %out: f16):
            %77 = arith.divf %in, %in_5 : f16
            linalg.yield %77 : f16
          }
          loom.semaphore_give %48 : memref<1x32x32xf16>
          loom.semaphore_give %28 : memref<1x32x128xf16>
          %64 = loom.alloc [16, 1, 32, 1] on @L1 : memref<16x1x32x1xf16>
          %65 = loom.semaphore_take %64 : memref<16x1x32x1xf16> -> memref<16x1x32x1xf16>
          loom.gather ins(%60 : memref<1x32x1xf16>) outs(%65 : memref<16x1x32x1xf16>) across(%arg4 : index) region : (UL : [%c0, %arg5], LR : [%c7, %arg5])
          loom.semaphore_give %60 : memref<1x32x1xf16>
          %66 = loom.alloc [16, 1, 32, 128] on @L1 : memref<16x1x32x128xf16>
          %67 = loom.semaphore_take %66 : memref<16x1x32x128xf16> -> memref<16x1x32x128xf16>
          loom.gather ins(%63 : memref<1x32x128xf16>) outs(%67 : memref<16x1x32x128xf16>) across(%arg4 : index) region : (UL : [%c0, %arg5], LR : [%c7, %arg5])
          loom.semaphore_give %63 : memref<1x32x128xf16>
          %68 = loom.alloc [1, 32, 128] on @L1 : memref<1x32x128xf16>
          %69 = loom.semaphore_take %68 : memref<1x32x128xf16> -> memref<1x32x128xf16>
          %70 = loom.alloc [16, 1, 32, 1] on @L1 : memref<16x1x32x1xf16>
          %71 = loom.semaphore_take %70 : memref<16x1x32x1xf16> -> memref<16x1x32x1xf16>
          %72 = loom.alloc [16, 1, 32, 128] on @L1 : memref<16x1x32x128xf16>
          %73 = loom.semaphore_take %72 : memref<16x1x32x128xf16> -> memref<16x1x32x128xf16>
          %74 = loom.alloc [16, 1, 32, 32] on @L1 : memref<16x1x32x32xf16>
          %75 = loom.semaphore_take %74 : memref<16x1x32x32xf16> -> memref<16x1x32x32xf16>
          %76 = arith.cmpi eq, %arg4, %c0 : index
          scf.if %76 {
            linalg.fill ins(%cst_1 : f16) outs(%31 : memref<1x32x1xf16>)
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1, d2, d3)>], iterator_types = ["reduction", "parallel", "parallel", "parallel"]} ins(%65 : memref<16x1x32x1xf16>) outs(%31 : memref<1x32x1xf16>) {
            ^bb0(%in: f16, %out: f16):
              %78 = arith.maximumf %in, %out : f16
              linalg.yield %78 : f16
            }
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%65, %31 : memref<16x1x32x1xf16>, memref<1x32x1xf16>) outs(%71 : memref<16x1x32x1xf16>) {
            ^bb0(%in: f16, %in_6: f16, %out: f16):
              %78 = arith.subf %in, %in_6 : f16
              %79 = math.exp %78 : f16
              linalg.yield %79 : f16
            }
            loom.semaphore_give %65 : memref<16x1x32x1xf16>
            loom.semaphore_give %31 : memref<1x32x1xf16>
            linalg.fill ins(%cst : f16) outs(%30 : memref<1x32x1xf16>)
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1, d2, d3)>], iterator_types = ["reduction", "parallel", "parallel", "parallel"]} ins(%71 : memref<16x1x32x1xf16>) outs(%30 : memref<1x32x1xf16>) {
            ^bb0(%in: f16, %out: f16):
              %78 = arith.addf %in, %out : f16
              linalg.yield %78 : f16
            }
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%71, %30 : memref<16x1x32x1xf16>, memref<1x32x1xf16>) outs(%71 : memref<16x1x32x1xf16>) {
            ^bb0(%in: f16, %in_6: f16, %out: f16):
              %78 = arith.divf %in, %in_6 : f16
              linalg.yield %78 : f16
            }
            loom.semaphore_give %30 : memref<1x32x1xf16>
            %77 = loom.broadcast ins(%71 : memref<16x1x32x1xf16>) outs(%75 : memref<16x1x32x32xf16>) dim(3) -> memref<16x1x32x128xf16, strided<[?, ?, ?, ?], offset: ?>>
            loom.semaphore_give %71 : memref<16x1x32x1xf16>
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%67, %77 : memref<16x1x32x128xf16>, memref<16x1x32x128xf16, strided<[?, ?, ?, ?], offset: ?>>) outs(%73 : memref<16x1x32x128xf16>) {
            ^bb0(%in: f16, %in_6: f16, %out: f16):
              %78 = arith.mulf %in, %in_6 : f16
              linalg.yield %78 : f16
            }
            loom.semaphore_give %75 : memref<16x1x32x32xf16>
            loom.semaphore_give %67 : memref<16x1x32x128xf16>
            linalg.fill ins(%cst : f16) outs(%69 : memref<1x32x128xf16>)
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1, d2, d3)>], iterator_types = ["reduction", "parallel", "parallel", "parallel"]} ins(%73 : memref<16x1x32x128xf16>) outs(%69 : memref<1x32x128xf16>) {
            ^bb0(%in: f16, %out: f16):
              %78 = arith.addf %in, %out : f16
              linalg.yield %78 : f16
            }
            loom.semaphore_give %73 : memref<16x1x32x128xf16>
            loom.semaphore_give %27 : memref<1x32x128xf16>
            %reinterpret_cast_5 = memref.reinterpret_cast %arg2 to offset: [%24], sizes: [1, 32, 128], strides: [4096, 128, 1] : memref<16x32x128xf16> to memref<1x32x128xf16, strided<[4096, 128, 1], offset: ?>>
            loom.copy %69, %reinterpret_cast_5 src_mem_space @mem_L1 dst_mem_space @mem_DRAM, broadcast : [1, 1] region : (UL : [%arg4, %arg5], LR : [%arg4, %arg5]) : memref<1x32x128xf16> to memref<1x32x128xf16, strided<[4096, 128, 1], offset: ?>>
            loom.semaphore_give %69 : memref<1x32x128xf16>
          }
        } {loom.block_sym = @tile_b, loom.iter_type = #loom.iter_type<temporal>}
        scf.reduce 
      }
      return
    }
  }
}
