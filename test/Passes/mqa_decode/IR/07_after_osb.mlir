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
            linalg.fill ins(%cst : f16) outs(%29 : memref<1x32x128xf16>)
            %30 = loom.alloc [1, 32, 1] on @L1 : memref<1x32x1xf16>
            %31 = loom.semaphore_take %30 : memref<1x32x1xf16> -> memref<1x32x1xf16>
            %32 = loom.semaphore_take %30 : memref<1x32x1xf16> -> memref<1x32x1xf16>
            %33 = loom.semaphore_take %30 : memref<1x32x1xf16> -> memref<1x32x1xf16>
            linalg.fill ins(%cst_0 : f16) outs(%33 : memref<1x32x1xf16>)
            %34 = loom.alloc [1, 32, 1] on @L1 : memref<1x32x1xf16>
            %35 = loom.semaphore_take %34 : memref<1x32x1xf16> -> memref<1x32x1xf16>
            linalg.fill ins(%cst_1 : f16) outs(%35 : memref<1x32x1xf16>)
            %36 = loom.alloc [1, 32, 128] on @L1 : memref<1x32x128xf16>
            %37 = loom.semaphore_take %36 : memref<1x32x128xf16> -> memref<1x32x128xf16>
            %38 = loom.alloc [1, 32, 1] on @L1 : memref<1x32x1xf16>
            %39 = loom.semaphore_take %38 : memref<1x32x1xf16> -> memref<1x32x1xf16>
            %40 = loom.alloc [1, 32, 1] on @L1 : memref<1x32x1xf16>
            %41 = loom.semaphore_take %40 : memref<1x32x1xf16> -> memref<1x32x1xf16>
            %42 = loom.alloc [1, 32, 1] on @L1 : memref<1x32x1xf16>
            %43 = loom.semaphore_take %42 : memref<1x32x1xf16> -> memref<1x32x1xf16>
            %44 = loom.alloc [1, 128, 512] on @L1 : memref<1x128x512xf16>
            %45 = loom.semaphore_take %44 : memref<1x128x512xf16> -> memref<1x128x512xf16>
            %46 = loom.alloc [1, 32, 512] on @L1 : memref<1x32x512xf16>
            %47 = loom.semaphore_take %46 : memref<1x32x512xf16> -> memref<1x32x512xf16>
            %48 = loom.alloc [1, 32, 32] on @L1 : memref<1x32x32xf16>
            %49 = loom.semaphore_take %48 : memref<1x32x32xf16> -> memref<1x32x32xf16>
            %50 = loom.semaphore_take %48 : memref<1x32x32xf16> -> memref<1x32x32xf16>
            %51 = loom.semaphore_take %48 : memref<1x32x32xf16> -> memref<1x32x32xf16>
            %52 = loom.alloc [1, 512, 128] on @L1 : memref<1x512x128xf16>
            %53 = loom.semaphore_take %52 : memref<1x512x128xf16> -> memref<1x512x128xf16>
            %54 = arith.muli %21, %c1048576 overflow<nsw> : index
            %55 = arith.addi %54, %27 : index
            %reinterpret_cast_3 = memref.reinterpret_cast %arg0 to offset: [%55], sizes: [1, 128, 512], strides: [1048576, 8192, 1] : memref<16x128x8192xf16> to memref<1x128x512xf16, strided<[1048576, 8192, 1], offset: ?>>
            loom.copy %reinterpret_cast_3, %45 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] region : (UL : [%arg5, %arg4], LR : [%arg5, %arg4]) : memref<1x128x512xf16, strided<[1048576, 8192, 1], offset: ?>> to memref<1x128x512xf16>
            linalg.fill ins(%cst : f16) outs(%47 : memref<1x32x512xf16>)
            linalg.batch_matmul ins(%25, %45 : memref<1x32x128xf16>, memref<1x128x512xf16>) outs(%47 : memref<1x32x512xf16>)
            loom.semaphore_give %45 : memref<1x128x512xf16>
            loom.semaphore_give %25 : memref<1x32x128xf16>
            linalg.fill ins(%cst_1 : f16) outs(%39 : memref<1x32x1xf16>)
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, 0)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%47 : memref<1x32x512xf16>) outs(%39 : memref<1x32x1xf16>) {
            ^bb0(%in: f16, %out: f16):
              %78 = arith.maximumf %in, %out : f16
              linalg.yield %78 : f16
            }
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%35, %39 : memref<1x32x1xf16>, memref<1x32x1xf16>) outs(%39 : memref<1x32x1xf16>) {
            ^bb0(%in: f16, %in_5: f16, %out: f16):
              %78 = arith.mulf %in_5, %cst_2 : f16
              %79 = arith.cmpf ogt, %in, %78 : f16
              %80 = arith.select %79, %in, %78 : f16
              linalg.yield %80 : f16
            }
            %56 = loom.broadcast ins(%39 : memref<1x32x1xf16>) outs(%51 : memref<1x32x32xf16>) dim(2) -> memref<1x32x512xf16, strided<[?, ?, ?], offset: ?>>
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%47, %56 : memref<1x32x512xf16>, memref<1x32x512xf16, strided<[?, ?, ?], offset: ?>>) outs(%47 : memref<1x32x512xf16>) {
            ^bb0(%in: f16, %in_5: f16, %out: f16):
              %78 = arith.mulf %in, %cst_2 : f16
              %79 = arith.subf %78, %in_5 : f16
              %80 = math.exp %79 : f16
              linalg.yield %80 : f16
            }
            loom.semaphore_give %51 : memref<1x32x32xf16>
            linalg.fill ins(%cst : f16) outs(%41 : memref<1x32x1xf16>)
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, 0)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%47 : memref<1x32x512xf16>) outs(%41 : memref<1x32x1xf16>) {
            ^bb0(%in: f16, %out: f16):
              %78 = arith.addf %in, %out : f16
              linalg.yield %78 : f16
            }
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%35, %39 : memref<1x32x1xf16>, memref<1x32x1xf16>) outs(%43 : memref<1x32x1xf16>) {
            ^bb0(%in: f16, %in_5: f16, %out: f16):
              %78 = arith.subf %in, %in_5 : f16
              %79 = math.exp %78 : f16
              linalg.yield %79 : f16
            }
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%33, %43, %41 : memref<1x32x1xf16>, memref<1x32x1xf16>, memref<1x32x1xf16>) outs(%33 : memref<1x32x1xf16>) {
            ^bb0(%in: f16, %in_5: f16, %in_6: f16, %out: f16):
              %78 = arith.mulf %in, %in_5 : f16
              %79 = arith.addf %78, %in_6 : f16
              linalg.yield %79 : f16
            }
            loom.semaphore_give %41 : memref<1x32x1xf16>
            %57 = loom.broadcast ins(%43 : memref<1x32x1xf16>) outs(%50 : memref<1x32x32xf16>) dim(2) -> memref<1x32x128xf16, strided<[?, ?, ?], offset: ?>>
            loom.semaphore_give %43 : memref<1x32x1xf16>
            %58 = arith.muli %23, %c65536 : index
            %59 = arith.addi %54, %58 : index
            %reinterpret_cast_4 = memref.reinterpret_cast %arg1 to offset: [%59], sizes: [1, 512, 128], strides: [1048576, 128, 1] : memref<16x8192x128xf16> to memref<1x512x128xf16, strided<[1048576, 128, 1], offset: ?>>
            loom.copy %reinterpret_cast_4, %53 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] region : (UL : [%arg5, %arg4], LR : [%arg5, %arg4]) : memref<1x512x128xf16, strided<[1048576, 128, 1], offset: ?>> to memref<1x512x128xf16>
            linalg.fill ins(%cst : f16) outs(%37 : memref<1x32x128xf16>)
            linalg.batch_matmul ins(%47, %53 : memref<1x32x512xf16>, memref<1x512x128xf16>) outs(%37 : memref<1x32x128xf16>)
            loom.semaphore_give %53 : memref<1x512x128xf16>
            loom.semaphore_give %47 : memref<1x32x512xf16>
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%37, %29, %57 : memref<1x32x128xf16>, memref<1x32x128xf16>, memref<1x32x128xf16, strided<[?, ?, ?], offset: ?>>) outs(%29 : memref<1x32x128xf16>) {
            ^bb0(%in: f16, %in_5: f16, %in_6: f16, %out: f16):
              %78 = arith.mulf %in_5, %in_6 : f16
              %79 = arith.addf %in, %78 : f16
              linalg.yield %79 : f16
            }
            loom.semaphore_give %50 : memref<1x32x32xf16>
            loom.semaphore_give %37 : memref<1x32x128xf16>
            linalg.copy ins(%39 : memref<1x32x1xf16>) outs(%35 : memref<1x32x1xf16>)
            loom.semaphore_give %39 : memref<1x32x1xf16>
            %60 = loom.alloc [1, 32, 1] on @L1 : memref<1x32x1xf16>
            %61 = loom.semaphore_take %60 : memref<1x32x1xf16> -> memref<1x32x1xf16>
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%33, %35 : memref<1x32x1xf16>, memref<1x32x1xf16>) outs(%61 : memref<1x32x1xf16>) {
            ^bb0(%in: f16, %in_5: f16, %out: f16):
              %78 = math.log %in : f16
              %79 = arith.addf %78, %in_5 : f16
              linalg.yield %79 : f16
            }
            loom.semaphore_give %35 : memref<1x32x1xf16>
            %62 = loom.broadcast ins(%33 : memref<1x32x1xf16>) outs(%49 : memref<1x32x32xf16>) dim(2) -> memref<1x32x128xf16, strided<[?, ?, ?], offset: ?>>
            loom.semaphore_give %33 : memref<1x32x1xf16>
            %63 = loom.alloc [1, 32, 128] on @L1 : memref<1x32x128xf16>
            %64 = loom.semaphore_take %63 : memref<1x32x128xf16> -> memref<1x32x128xf16>
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%29, %62 : memref<1x32x128xf16>, memref<1x32x128xf16, strided<[?, ?, ?], offset: ?>>) outs(%64 : memref<1x32x128xf16>) {
            ^bb0(%in: f16, %in_5: f16, %out: f16):
              %78 = arith.divf %in, %in_5 : f16
              linalg.yield %78 : f16
            }
            loom.semaphore_give %49 : memref<1x32x32xf16>
            loom.semaphore_give %29 : memref<1x32x128xf16>
            %65 = loom.alloc [16, 1, 32, 1] on @L1 : memref<16x1x32x1xf16>
            %66 = loom.semaphore_take %65 : memref<16x1x32x1xf16> -> memref<16x1x32x1xf16>
            loom.gather ins(%61 : memref<1x32x1xf16>) outs(%66 : memref<16x1x32x1xf16>) across(%arg5 : index) region : (UL : [%c0, %arg4], LR : [%c7, %arg4])
            loom.semaphore_give %61 : memref<1x32x1xf16>
            %67 = loom.alloc [16, 1, 32, 128] on @L1 : memref<16x1x32x128xf16>
            %68 = loom.semaphore_take %67 : memref<16x1x32x128xf16> -> memref<16x1x32x128xf16>
            loom.gather ins(%64 : memref<1x32x128xf16>) outs(%68 : memref<16x1x32x128xf16>) across(%arg5 : index) region : (UL : [%c0, %arg4], LR : [%c7, %arg4])
            loom.semaphore_give %64 : memref<1x32x128xf16>
            %69 = loom.alloc [1, 32, 128] on @L1 : memref<1x32x128xf16>
            %70 = loom.semaphore_take %69 : memref<1x32x128xf16> -> memref<1x32x128xf16>
            %71 = loom.alloc [16, 1, 32, 1] on @L1 : memref<16x1x32x1xf16>
            %72 = loom.semaphore_take %71 : memref<16x1x32x1xf16> -> memref<16x1x32x1xf16>
            %73 = loom.alloc [16, 1, 32, 128] on @L1 : memref<16x1x32x128xf16>
            %74 = loom.semaphore_take %73 : memref<16x1x32x128xf16> -> memref<16x1x32x128xf16>
            %75 = loom.alloc [16, 1, 32, 32] on @L1 : memref<16x1x32x32xf16>
            %76 = loom.semaphore_take %75 : memref<16x1x32x32xf16> -> memref<16x1x32x32xf16>
            %77 = arith.cmpi eq, %arg5, %c0 : index
            scf.if %77 {
              linalg.fill ins(%cst_1 : f16) outs(%32 : memref<1x32x1xf16>)
              linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1, d2, d3)>], iterator_types = ["reduction", "parallel", "parallel", "parallel"]} ins(%66 : memref<16x1x32x1xf16>) outs(%32 : memref<1x32x1xf16>) {
              ^bb0(%in: f16, %out: f16):
                %79 = arith.maximumf %in, %out : f16
                linalg.yield %79 : f16
              }
              linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%66, %32 : memref<16x1x32x1xf16>, memref<1x32x1xf16>) outs(%72 : memref<16x1x32x1xf16>) {
              ^bb0(%in: f16, %in_6: f16, %out: f16):
                %79 = arith.subf %in, %in_6 : f16
                %80 = math.exp %79 : f16
                linalg.yield %80 : f16
              }
              loom.semaphore_give %66 : memref<16x1x32x1xf16>
              loom.semaphore_give %32 : memref<1x32x1xf16>
              linalg.fill ins(%cst : f16) outs(%31 : memref<1x32x1xf16>)
              linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1, d2, d3)>], iterator_types = ["reduction", "parallel", "parallel", "parallel"]} ins(%72 : memref<16x1x32x1xf16>) outs(%31 : memref<1x32x1xf16>) {
              ^bb0(%in: f16, %out: f16):
                %79 = arith.addf %in, %out : f16
                linalg.yield %79 : f16
              }
              linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%72, %31 : memref<16x1x32x1xf16>, memref<1x32x1xf16>) outs(%72 : memref<16x1x32x1xf16>) {
              ^bb0(%in: f16, %in_6: f16, %out: f16):
                %79 = arith.divf %in, %in_6 : f16
                linalg.yield %79 : f16
              }
              loom.semaphore_give %31 : memref<1x32x1xf16>
              %78 = loom.broadcast ins(%72 : memref<16x1x32x1xf16>) outs(%76 : memref<16x1x32x32xf16>) dim(3) -> memref<16x1x32x128xf16, strided<[?, ?, ?, ?], offset: ?>>
              loom.semaphore_give %72 : memref<16x1x32x1xf16>
              linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%68, %78 : memref<16x1x32x128xf16>, memref<16x1x32x128xf16, strided<[?, ?, ?, ?], offset: ?>>) outs(%74 : memref<16x1x32x128xf16>) {
              ^bb0(%in: f16, %in_6: f16, %out: f16):
                %79 = arith.mulf %in, %in_6 : f16
                linalg.yield %79 : f16
              }
              loom.semaphore_give %76 : memref<16x1x32x32xf16>
              loom.semaphore_give %68 : memref<16x1x32x128xf16>
              linalg.fill ins(%cst : f16) outs(%70 : memref<1x32x128xf16>)
              linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1, d2, d3)>], iterator_types = ["reduction", "parallel", "parallel", "parallel"]} ins(%74 : memref<16x1x32x128xf16>) outs(%70 : memref<1x32x128xf16>) {
              ^bb0(%in: f16, %out: f16):
                %79 = arith.addf %in, %out : f16
                linalg.yield %79 : f16
              }
              loom.semaphore_give %74 : memref<16x1x32x128xf16>
              %reinterpret_cast_5 = memref.reinterpret_cast %arg2 to offset: [%26], sizes: [1, 32, 128], strides: [4096, 128, 1] : memref<16x32x128xf16> to memref<1x32x128xf16, strided<[4096, 128, 1], offset: ?>>
              loom.copy %70, %reinterpret_cast_5 src_mem_space @mem_L1 dst_mem_space @mem_DRAM, broadcast : [1, 1] region : (UL : [%arg5, %arg4], LR : [%arg5, %arg4]) : memref<1x32x128xf16> to memref<1x32x128xf16, strided<[4096, 128, 1], offset: ?>>
              loom.semaphore_give %70 : memref<1x32x128xf16>
            }
          } {loom.block_sym = @tile_s, loom.iter_type = #loom.iter_type<temporal>}
        } {loom.block_sym = @tile_b, loom.iter_type = #loom.iter_type<temporal>}
        scf.reduce 
      } {loom.block_syms = [@tile_b, @tile_s], loom.iter_types = [#loom.iter_type<spatial>, #loom.iter_type<spatial>], loom.logical_levels = [1, 0], loom.physical_dims = [@dim_y, @dim_x]}
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
          linalg.fill ins(%cst : f16) outs(%31 : memref<1x32x128xf16>)
          %32 = loom.alloc [1, 32, 1] on @L1 : memref<1x32x1xf16>
          %33 = loom.semaphore_take %32 : memref<1x32x1xf16> -> memref<1x32x1xf16>
          %34 = loom.semaphore_take %32 : memref<1x32x1xf16> -> memref<1x32x1xf16>
          %35 = loom.semaphore_take %32 : memref<1x32x1xf16> -> memref<1x32x1xf16>
          linalg.fill ins(%cst_0 : f16) outs(%35 : memref<1x32x1xf16>)
          %36 = loom.alloc [1, 32, 1] on @L1 : memref<1x32x1xf16>
          %37 = loom.semaphore_take %36 : memref<1x32x1xf16> -> memref<1x32x1xf16>
          linalg.fill ins(%cst_1 : f16) outs(%37 : memref<1x32x1xf16>)
          %38 = loom.alloc [1, 32, 128] on @L1 : memref<1x32x128xf16>
          %39 = loom.semaphore_take %38 : memref<1x32x128xf16> -> memref<1x32x128xf16>
          %40 = loom.alloc [1, 32, 1] on @L1 : memref<1x32x1xf16>
          %41 = loom.semaphore_take %40 : memref<1x32x1xf16> -> memref<1x32x1xf16>
          %42 = loom.alloc [1, 32, 1] on @L1 : memref<1x32x1xf16>
          %43 = loom.semaphore_take %42 : memref<1x32x1xf16> -> memref<1x32x1xf16>
          %44 = loom.alloc [1, 32, 1] on @L1 : memref<1x32x1xf16>
          %45 = loom.semaphore_take %44 : memref<1x32x1xf16> -> memref<1x32x1xf16>
          %46 = loom.alloc [1, 128, 512] on @L1 : memref<1x128x512xf16>
          %47 = loom.semaphore_take %46 : memref<1x128x512xf16> -> memref<1x128x512xf16>
          %48 = loom.alloc [1, 32, 512] on @L1 : memref<1x32x512xf16>
          %49 = loom.semaphore_take %48 : memref<1x32x512xf16> -> memref<1x32x512xf16>
          %50 = loom.alloc [1, 32, 32] on @L1 : memref<1x32x32xf16>
          %51 = loom.semaphore_take %50 : memref<1x32x32xf16> -> memref<1x32x32xf16>
          %52 = loom.semaphore_take %50 : memref<1x32x32xf16> -> memref<1x32x32xf16>
          %53 = loom.semaphore_take %50 : memref<1x32x32xf16> -> memref<1x32x32xf16>
          %54 = loom.alloc [1, 512, 128] on @L1 : memref<1x512x128xf16>
          %55 = loom.semaphore_take %54 : memref<1x512x128xf16> -> memref<1x512x128xf16>
          %56 = arith.muli %21, %c1048576 overflow<nsw> : index
          %57 = arith.addi %56, %29 : index
          %reinterpret_cast_3 = memref.reinterpret_cast %arg0 to offset: [%57], sizes: [1, 128, 512], strides: [1048576, 8192, 1] : memref<16x128x8192xf16> to memref<1x128x512xf16, strided<[1048576, 8192, 1], offset: ?>>
          %58 = arith.addi %arg6, %27 : index
          loom.copy %reinterpret_cast_3, %47 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] region : (UL : [%arg5, %58], LR : [%arg5, %58]) : memref<1x128x512xf16, strided<[1048576, 8192, 1], offset: ?>> to memref<1x128x512xf16>
          linalg.fill ins(%cst : f16) outs(%49 : memref<1x32x512xf16>)
          linalg.batch_matmul ins(%25, %47 : memref<1x32x128xf16>, memref<1x128x512xf16>) outs(%49 : memref<1x32x512xf16>)
          loom.semaphore_give %47 : memref<1x128x512xf16>
          loom.semaphore_give %25 : memref<1x32x128xf16>
          linalg.fill ins(%cst_1 : f16) outs(%41 : memref<1x32x1xf16>)
          linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, 0)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%49 : memref<1x32x512xf16>) outs(%41 : memref<1x32x1xf16>) {
          ^bb0(%in: f16, %out: f16):
            %81 = arith.maximumf %in, %out : f16
            linalg.yield %81 : f16
          }
          linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%37, %41 : memref<1x32x1xf16>, memref<1x32x1xf16>) outs(%41 : memref<1x32x1xf16>) {
          ^bb0(%in: f16, %in_5: f16, %out: f16):
            %81 = arith.mulf %in_5, %cst_2 : f16
            %82 = arith.cmpf ogt, %in, %81 : f16
            %83 = arith.select %82, %in, %81 : f16
            linalg.yield %83 : f16
          }
          %59 = loom.broadcast ins(%41 : memref<1x32x1xf16>) outs(%53 : memref<1x32x32xf16>) dim(2) -> memref<1x32x512xf16, strided<[?, ?, ?], offset: ?>>
          linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%49, %59 : memref<1x32x512xf16>, memref<1x32x512xf16, strided<[?, ?, ?], offset: ?>>) outs(%49 : memref<1x32x512xf16>) {
          ^bb0(%in: f16, %in_5: f16, %out: f16):
            %81 = arith.mulf %in, %cst_2 : f16
            %82 = arith.subf %81, %in_5 : f16
            %83 = math.exp %82 : f16
            linalg.yield %83 : f16
          }
          loom.semaphore_give %53 : memref<1x32x32xf16>
          linalg.fill ins(%cst : f16) outs(%43 : memref<1x32x1xf16>)
          linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, 0)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%49 : memref<1x32x512xf16>) outs(%43 : memref<1x32x1xf16>) {
          ^bb0(%in: f16, %out: f16):
            %81 = arith.addf %in, %out : f16
            linalg.yield %81 : f16
          }
          linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%37, %41 : memref<1x32x1xf16>, memref<1x32x1xf16>) outs(%45 : memref<1x32x1xf16>) {
          ^bb0(%in: f16, %in_5: f16, %out: f16):
            %81 = arith.subf %in, %in_5 : f16
            %82 = math.exp %81 : f16
            linalg.yield %82 : f16
          }
          linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%35, %45, %43 : memref<1x32x1xf16>, memref<1x32x1xf16>, memref<1x32x1xf16>) outs(%35 : memref<1x32x1xf16>) {
          ^bb0(%in: f16, %in_5: f16, %in_6: f16, %out: f16):
            %81 = arith.mulf %in, %in_5 : f16
            %82 = arith.addf %81, %in_6 : f16
            linalg.yield %82 : f16
          }
          loom.semaphore_give %43 : memref<1x32x1xf16>
          %60 = loom.broadcast ins(%45 : memref<1x32x1xf16>) outs(%52 : memref<1x32x32xf16>) dim(2) -> memref<1x32x128xf16, strided<[?, ?, ?], offset: ?>>
          loom.semaphore_give %45 : memref<1x32x1xf16>
          %61 = arith.muli %23, %c65536 : index
          %62 = arith.addi %56, %61 : index
          %reinterpret_cast_4 = memref.reinterpret_cast %arg1 to offset: [%62], sizes: [1, 512, 128], strides: [1048576, 128, 1] : memref<16x8192x128xf16> to memref<1x512x128xf16, strided<[1048576, 128, 1], offset: ?>>
          loom.copy %reinterpret_cast_4, %55 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] region : (UL : [%arg5, %58], LR : [%arg5, %58]) : memref<1x512x128xf16, strided<[1048576, 128, 1], offset: ?>> to memref<1x512x128xf16>
          linalg.fill ins(%cst : f16) outs(%39 : memref<1x32x128xf16>)
          linalg.batch_matmul ins(%49, %55 : memref<1x32x512xf16>, memref<1x512x128xf16>) outs(%39 : memref<1x32x128xf16>)
          loom.semaphore_give %55 : memref<1x512x128xf16>
          loom.semaphore_give %49 : memref<1x32x512xf16>
          linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%39, %31, %60 : memref<1x32x128xf16>, memref<1x32x128xf16>, memref<1x32x128xf16, strided<[?, ?, ?], offset: ?>>) outs(%31 : memref<1x32x128xf16>) {
          ^bb0(%in: f16, %in_5: f16, %in_6: f16, %out: f16):
            %81 = arith.mulf %in_5, %in_6 : f16
            %82 = arith.addf %in, %81 : f16
            linalg.yield %82 : f16
          }
          loom.semaphore_give %52 : memref<1x32x32xf16>
          loom.semaphore_give %39 : memref<1x32x128xf16>
          linalg.copy ins(%41 : memref<1x32x1xf16>) outs(%37 : memref<1x32x1xf16>)
          loom.semaphore_give %41 : memref<1x32x1xf16>
          %63 = loom.alloc [1, 32, 1] on @L1 : memref<1x32x1xf16>
          %64 = loom.semaphore_take %63 : memref<1x32x1xf16> -> memref<1x32x1xf16>
          linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%35, %37 : memref<1x32x1xf16>, memref<1x32x1xf16>) outs(%64 : memref<1x32x1xf16>) {
          ^bb0(%in: f16, %in_5: f16, %out: f16):
            %81 = math.log %in : f16
            %82 = arith.addf %81, %in_5 : f16
            linalg.yield %82 : f16
          }
          loom.semaphore_give %37 : memref<1x32x1xf16>
          %65 = loom.broadcast ins(%35 : memref<1x32x1xf16>) outs(%51 : memref<1x32x32xf16>) dim(2) -> memref<1x32x128xf16, strided<[?, ?, ?], offset: ?>>
          loom.semaphore_give %35 : memref<1x32x1xf16>
          %66 = loom.alloc [1, 32, 128] on @L1 : memref<1x32x128xf16>
          %67 = loom.semaphore_take %66 : memref<1x32x128xf16> -> memref<1x32x128xf16>
          linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%31, %65 : memref<1x32x128xf16>, memref<1x32x128xf16, strided<[?, ?, ?], offset: ?>>) outs(%67 : memref<1x32x128xf16>) {
          ^bb0(%in: f16, %in_5: f16, %out: f16):
            %81 = arith.divf %in, %in_5 : f16
            linalg.yield %81 : f16
          }
          loom.semaphore_give %51 : memref<1x32x32xf16>
          loom.semaphore_give %31 : memref<1x32x128xf16>
          %68 = loom.alloc [16, 1, 32, 1] on @L1 : memref<16x1x32x1xf16>
          %69 = loom.semaphore_take %68 : memref<16x1x32x1xf16> -> memref<16x1x32x1xf16>
          loom.gather ins(%64 : memref<1x32x1xf16>) outs(%69 : memref<16x1x32x1xf16>) across(%arg5 : index) region : (UL : [%c0, %58], LR : [%c7, %58])
          loom.semaphore_give %64 : memref<1x32x1xf16>
          %70 = loom.alloc [16, 1, 32, 128] on @L1 : memref<16x1x32x128xf16>
          %71 = loom.semaphore_take %70 : memref<16x1x32x128xf16> -> memref<16x1x32x128xf16>
          loom.gather ins(%67 : memref<1x32x128xf16>) outs(%71 : memref<16x1x32x128xf16>) across(%arg5 : index) region : (UL : [%c0, %58], LR : [%c7, %58])
          loom.semaphore_give %67 : memref<1x32x128xf16>
          %72 = loom.alloc [1, 32, 128] on @L1 : memref<1x32x128xf16>
          %73 = loom.semaphore_take %72 : memref<1x32x128xf16> -> memref<1x32x128xf16>
          %74 = loom.alloc [16, 1, 32, 1] on @L1 : memref<16x1x32x1xf16>
          %75 = loom.semaphore_take %74 : memref<16x1x32x1xf16> -> memref<16x1x32x1xf16>
          %76 = loom.alloc [16, 1, 32, 128] on @L1 : memref<16x1x32x128xf16>
          %77 = loom.semaphore_take %76 : memref<16x1x32x128xf16> -> memref<16x1x32x128xf16>
          %78 = loom.alloc [16, 1, 32, 32] on @L1 : memref<16x1x32x32xf16>
          %79 = loom.semaphore_take %78 : memref<16x1x32x32xf16> -> memref<16x1x32x32xf16>
          %80 = arith.cmpi eq, %arg5, %c0 : index
          scf.if %80 {
            linalg.fill ins(%cst_1 : f16) outs(%34 : memref<1x32x1xf16>)
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1, d2, d3)>], iterator_types = ["reduction", "parallel", "parallel", "parallel"]} ins(%69 : memref<16x1x32x1xf16>) outs(%34 : memref<1x32x1xf16>) {
            ^bb0(%in: f16, %out: f16):
              %82 = arith.maximumf %in, %out : f16
              linalg.yield %82 : f16
            }
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%69, %34 : memref<16x1x32x1xf16>, memref<1x32x1xf16>) outs(%75 : memref<16x1x32x1xf16>) {
            ^bb0(%in: f16, %in_6: f16, %out: f16):
              %82 = arith.subf %in, %in_6 : f16
              %83 = math.exp %82 : f16
              linalg.yield %83 : f16
            }
            loom.semaphore_give %69 : memref<16x1x32x1xf16>
            loom.semaphore_give %34 : memref<1x32x1xf16>
            linalg.fill ins(%cst : f16) outs(%33 : memref<1x32x1xf16>)
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1, d2, d3)>], iterator_types = ["reduction", "parallel", "parallel", "parallel"]} ins(%75 : memref<16x1x32x1xf16>) outs(%33 : memref<1x32x1xf16>) {
            ^bb0(%in: f16, %out: f16):
              %82 = arith.addf %in, %out : f16
              linalg.yield %82 : f16
            }
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%75, %33 : memref<16x1x32x1xf16>, memref<1x32x1xf16>) outs(%75 : memref<16x1x32x1xf16>) {
            ^bb0(%in: f16, %in_6: f16, %out: f16):
              %82 = arith.divf %in, %in_6 : f16
              linalg.yield %82 : f16
            }
            loom.semaphore_give %33 : memref<1x32x1xf16>
            %81 = loom.broadcast ins(%75 : memref<16x1x32x1xf16>) outs(%79 : memref<16x1x32x32xf16>) dim(3) -> memref<16x1x32x128xf16, strided<[?, ?, ?, ?], offset: ?>>
            loom.semaphore_give %75 : memref<16x1x32x1xf16>
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%71, %81 : memref<16x1x32x128xf16>, memref<16x1x32x128xf16, strided<[?, ?, ?, ?], offset: ?>>) outs(%77 : memref<16x1x32x128xf16>) {
            ^bb0(%in: f16, %in_6: f16, %out: f16):
              %82 = arith.mulf %in, %in_6 : f16
              linalg.yield %82 : f16
            }
            loom.semaphore_give %79 : memref<16x1x32x32xf16>
            loom.semaphore_give %71 : memref<16x1x32x128xf16>
            linalg.fill ins(%cst : f16) outs(%73 : memref<1x32x128xf16>)
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1, d2, d3)>], iterator_types = ["reduction", "parallel", "parallel", "parallel"]} ins(%77 : memref<16x1x32x128xf16>) outs(%73 : memref<1x32x128xf16>) {
            ^bb0(%in: f16, %out: f16):
              %82 = arith.addf %in, %out : f16
              linalg.yield %82 : f16
            }
            loom.semaphore_give %77 : memref<16x1x32x128xf16>
            %reinterpret_cast_5 = memref.reinterpret_cast %arg2 to offset: [%26], sizes: [1, 32, 128], strides: [4096, 128, 1] : memref<16x32x128xf16> to memref<1x32x128xf16, strided<[4096, 128, 1], offset: ?>>
            loom.copy %73, %reinterpret_cast_5 src_mem_space @mem_L1 dst_mem_space @mem_DRAM, broadcast : [1, 1] region : (UL : [%arg5, %58], LR : [%arg5, %58]) : memref<1x32x128xf16> to memref<1x32x128xf16, strided<[4096, 128, 1], offset: ?>>
            loom.semaphore_give %73 : memref<1x32x128xf16>
          }
        } {loom.block_sym = @tile_b, loom.iter_type = #loom.iter_type<temporal>}
        scf.reduce 
      } {loom.block_syms = [@tile_b, @tile_s, @tile_s], loom.iter_types = [#loom.iter_type<spatial>, #loom.iter_type<spatial>, #loom.iter_type<spatial>], loom.logical_levels = [1, 0, 0], loom.physical_dims = [@dim_y, @dim_x, @dim_y]}
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
          linalg.fill ins(%cst : f16) outs(%31 : memref<1x32x128xf16>)
          %32 = loom.alloc [1, 32, 1] on @L1 : memref<1x32x1xf16>
          %33 = loom.semaphore_take %32 : memref<1x32x1xf16> -> memref<1x32x1xf16>
          %34 = loom.semaphore_take %32 : memref<1x32x1xf16> -> memref<1x32x1xf16>
          %35 = loom.semaphore_take %32 : memref<1x32x1xf16> -> memref<1x32x1xf16>
          linalg.fill ins(%cst_0 : f16) outs(%35 : memref<1x32x1xf16>)
          %36 = loom.alloc [1, 32, 1] on @L1 : memref<1x32x1xf16>
          %37 = loom.semaphore_take %36 : memref<1x32x1xf16> -> memref<1x32x1xf16>
          linalg.fill ins(%cst_1 : f16) outs(%37 : memref<1x32x1xf16>)
          %38 = loom.alloc [1, 32, 128] on @L1 : memref<1x32x128xf16>
          %39 = loom.semaphore_take %38 : memref<1x32x128xf16> -> memref<1x32x128xf16>
          %40 = loom.alloc [1, 32, 1] on @L1 : memref<1x32x1xf16>
          %41 = loom.semaphore_take %40 : memref<1x32x1xf16> -> memref<1x32x1xf16>
          %42 = loom.alloc [1, 32, 1] on @L1 : memref<1x32x1xf16>
          %43 = loom.semaphore_take %42 : memref<1x32x1xf16> -> memref<1x32x1xf16>
          %44 = loom.alloc [1, 32, 1] on @L1 : memref<1x32x1xf16>
          %45 = loom.semaphore_take %44 : memref<1x32x1xf16> -> memref<1x32x1xf16>
          %46 = loom.alloc [1, 128, 512] on @L1 : memref<1x128x512xf16>
          %47 = loom.semaphore_take %46 : memref<1x128x512xf16> -> memref<1x128x512xf16>
          %48 = loom.alloc [1, 32, 512] on @L1 : memref<1x32x512xf16>
          %49 = loom.semaphore_take %48 : memref<1x32x512xf16> -> memref<1x32x512xf16>
          %50 = loom.alloc [1, 32, 32] on @L1 : memref<1x32x32xf16>
          %51 = loom.semaphore_take %50 : memref<1x32x32xf16> -> memref<1x32x32xf16>
          %52 = loom.semaphore_take %50 : memref<1x32x32xf16> -> memref<1x32x32xf16>
          %53 = loom.semaphore_take %50 : memref<1x32x32xf16> -> memref<1x32x32xf16>
          %54 = loom.alloc [1, 512, 128] on @L1 : memref<1x512x128xf16>
          %55 = loom.semaphore_take %54 : memref<1x512x128xf16> -> memref<1x512x128xf16>
          %56 = arith.muli %21, %c1048576 overflow<nsw> : index
          %57 = arith.addi %56, %29 : index
          %reinterpret_cast_3 = memref.reinterpret_cast %arg0 to offset: [%57], sizes: [1, 128, 512], strides: [1048576, 8192, 1] : memref<16x128x8192xf16> to memref<1x128x512xf16, strided<[1048576, 8192, 1], offset: ?>>
          %58 = arith.addi %arg6, %27 : index
          loom.copy %reinterpret_cast_3, %47 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] region : (UL : [%arg5, %58], LR : [%arg5, %58]) : memref<1x128x512xf16, strided<[1048576, 8192, 1], offset: ?>> to memref<1x128x512xf16>
          linalg.fill ins(%cst : f16) outs(%49 : memref<1x32x512xf16>)
          linalg.batch_matmul ins(%25, %47 : memref<1x32x128xf16>, memref<1x128x512xf16>) outs(%49 : memref<1x32x512xf16>)
          loom.semaphore_give %47 : memref<1x128x512xf16>
          loom.semaphore_give %25 : memref<1x32x128xf16>
          linalg.fill ins(%cst_1 : f16) outs(%41 : memref<1x32x1xf16>)
          linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, 0)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%49 : memref<1x32x512xf16>) outs(%41 : memref<1x32x1xf16>) {
          ^bb0(%in: f16, %out: f16):
            %81 = arith.maximumf %in, %out : f16
            linalg.yield %81 : f16
          }
          linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%37, %41 : memref<1x32x1xf16>, memref<1x32x1xf16>) outs(%41 : memref<1x32x1xf16>) {
          ^bb0(%in: f16, %in_5: f16, %out: f16):
            %81 = arith.mulf %in_5, %cst_2 : f16
            %82 = arith.cmpf ogt, %in, %81 : f16
            %83 = arith.select %82, %in, %81 : f16
            linalg.yield %83 : f16
          }
          %59 = loom.broadcast ins(%41 : memref<1x32x1xf16>) outs(%53 : memref<1x32x32xf16>) dim(2) -> memref<1x32x512xf16, strided<[?, ?, ?], offset: ?>>
          linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%49, %59 : memref<1x32x512xf16>, memref<1x32x512xf16, strided<[?, ?, ?], offset: ?>>) outs(%49 : memref<1x32x512xf16>) {
          ^bb0(%in: f16, %in_5: f16, %out: f16):
            %81 = arith.mulf %in, %cst_2 : f16
            %82 = arith.subf %81, %in_5 : f16
            %83 = math.exp %82 : f16
            linalg.yield %83 : f16
          }
          loom.semaphore_give %53 : memref<1x32x32xf16>
          linalg.fill ins(%cst : f16) outs(%43 : memref<1x32x1xf16>)
          linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, 0)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%49 : memref<1x32x512xf16>) outs(%43 : memref<1x32x1xf16>) {
          ^bb0(%in: f16, %out: f16):
            %81 = arith.addf %in, %out : f16
            linalg.yield %81 : f16
          }
          linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%37, %41 : memref<1x32x1xf16>, memref<1x32x1xf16>) outs(%45 : memref<1x32x1xf16>) {
          ^bb0(%in: f16, %in_5: f16, %out: f16):
            %81 = arith.subf %in, %in_5 : f16
            %82 = math.exp %81 : f16
            linalg.yield %82 : f16
          }
          linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%35, %45, %43 : memref<1x32x1xf16>, memref<1x32x1xf16>, memref<1x32x1xf16>) outs(%35 : memref<1x32x1xf16>) {
          ^bb0(%in: f16, %in_5: f16, %in_6: f16, %out: f16):
            %81 = arith.mulf %in, %in_5 : f16
            %82 = arith.addf %81, %in_6 : f16
            linalg.yield %82 : f16
          }
          loom.semaphore_give %43 : memref<1x32x1xf16>
          %60 = loom.broadcast ins(%45 : memref<1x32x1xf16>) outs(%52 : memref<1x32x32xf16>) dim(2) -> memref<1x32x128xf16, strided<[?, ?, ?], offset: ?>>
          loom.semaphore_give %45 : memref<1x32x1xf16>
          %61 = arith.muli %23, %c65536 : index
          %62 = arith.addi %56, %61 : index
          %reinterpret_cast_4 = memref.reinterpret_cast %arg1 to offset: [%62], sizes: [1, 512, 128], strides: [1048576, 128, 1] : memref<16x8192x128xf16> to memref<1x512x128xf16, strided<[1048576, 128, 1], offset: ?>>
          loom.copy %reinterpret_cast_4, %55 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] region : (UL : [%arg5, %58], LR : [%arg5, %58]) : memref<1x512x128xf16, strided<[1048576, 128, 1], offset: ?>> to memref<1x512x128xf16>
          linalg.fill ins(%cst : f16) outs(%39 : memref<1x32x128xf16>)
          linalg.batch_matmul ins(%49, %55 : memref<1x32x512xf16>, memref<1x512x128xf16>) outs(%39 : memref<1x32x128xf16>)
          loom.semaphore_give %55 : memref<1x512x128xf16>
          loom.semaphore_give %49 : memref<1x32x512xf16>
          linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%39, %31, %60 : memref<1x32x128xf16>, memref<1x32x128xf16>, memref<1x32x128xf16, strided<[?, ?, ?], offset: ?>>) outs(%31 : memref<1x32x128xf16>) {
          ^bb0(%in: f16, %in_5: f16, %in_6: f16, %out: f16):
            %81 = arith.mulf %in_5, %in_6 : f16
            %82 = arith.addf %in, %81 : f16
            linalg.yield %82 : f16
          }
          loom.semaphore_give %52 : memref<1x32x32xf16>
          loom.semaphore_give %39 : memref<1x32x128xf16>
          linalg.copy ins(%41 : memref<1x32x1xf16>) outs(%37 : memref<1x32x1xf16>)
          loom.semaphore_give %41 : memref<1x32x1xf16>
          %63 = loom.alloc [1, 32, 1] on @L1 : memref<1x32x1xf16>
          %64 = loom.semaphore_take %63 : memref<1x32x1xf16> -> memref<1x32x1xf16>
          linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%35, %37 : memref<1x32x1xf16>, memref<1x32x1xf16>) outs(%64 : memref<1x32x1xf16>) {
          ^bb0(%in: f16, %in_5: f16, %out: f16):
            %81 = math.log %in : f16
            %82 = arith.addf %81, %in_5 : f16
            linalg.yield %82 : f16
          }
          loom.semaphore_give %37 : memref<1x32x1xf16>
          %65 = loom.broadcast ins(%35 : memref<1x32x1xf16>) outs(%51 : memref<1x32x32xf16>) dim(2) -> memref<1x32x128xf16, strided<[?, ?, ?], offset: ?>>
          loom.semaphore_give %35 : memref<1x32x1xf16>
          %66 = loom.alloc [1, 32, 128] on @L1 : memref<1x32x128xf16>
          %67 = loom.semaphore_take %66 : memref<1x32x128xf16> -> memref<1x32x128xf16>
          linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%31, %65 : memref<1x32x128xf16>, memref<1x32x128xf16, strided<[?, ?, ?], offset: ?>>) outs(%67 : memref<1x32x128xf16>) {
          ^bb0(%in: f16, %in_5: f16, %out: f16):
            %81 = arith.divf %in, %in_5 : f16
            linalg.yield %81 : f16
          }
          loom.semaphore_give %51 : memref<1x32x32xf16>
          loom.semaphore_give %31 : memref<1x32x128xf16>
          %68 = loom.alloc [16, 1, 32, 1] on @L1 : memref<16x1x32x1xf16>
          %69 = loom.semaphore_take %68 : memref<16x1x32x1xf16> -> memref<16x1x32x1xf16>
          loom.gather ins(%64 : memref<1x32x1xf16>) outs(%69 : memref<16x1x32x1xf16>) across(%arg5 : index) region : (UL : [%c0, %58], LR : [%c7, %58])
          loom.semaphore_give %64 : memref<1x32x1xf16>
          %70 = loom.alloc [16, 1, 32, 128] on @L1 : memref<16x1x32x128xf16>
          %71 = loom.semaphore_take %70 : memref<16x1x32x128xf16> -> memref<16x1x32x128xf16>
          loom.gather ins(%67 : memref<1x32x128xf16>) outs(%71 : memref<16x1x32x128xf16>) across(%arg5 : index) region : (UL : [%c0, %58], LR : [%c7, %58])
          loom.semaphore_give %67 : memref<1x32x128xf16>
          %72 = loom.alloc [1, 32, 128] on @L1 : memref<1x32x128xf16>
          %73 = loom.semaphore_take %72 : memref<1x32x128xf16> -> memref<1x32x128xf16>
          %74 = loom.alloc [16, 1, 32, 1] on @L1 : memref<16x1x32x1xf16>
          %75 = loom.semaphore_take %74 : memref<16x1x32x1xf16> -> memref<16x1x32x1xf16>
          %76 = loom.alloc [16, 1, 32, 128] on @L1 : memref<16x1x32x128xf16>
          %77 = loom.semaphore_take %76 : memref<16x1x32x128xf16> -> memref<16x1x32x128xf16>
          %78 = loom.alloc [16, 1, 32, 32] on @L1 : memref<16x1x32x32xf16>
          %79 = loom.semaphore_take %78 : memref<16x1x32x32xf16> -> memref<16x1x32x32xf16>
          %80 = arith.cmpi eq, %arg5, %c0 : index
          scf.if %80 {
            linalg.fill ins(%cst_1 : f16) outs(%34 : memref<1x32x1xf16>)
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1, d2, d3)>], iterator_types = ["reduction", "parallel", "parallel", "parallel"]} ins(%69 : memref<16x1x32x1xf16>) outs(%34 : memref<1x32x1xf16>) {
            ^bb0(%in: f16, %out: f16):
              %82 = arith.maximumf %in, %out : f16
              linalg.yield %82 : f16
            }
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%69, %34 : memref<16x1x32x1xf16>, memref<1x32x1xf16>) outs(%75 : memref<16x1x32x1xf16>) {
            ^bb0(%in: f16, %in_6: f16, %out: f16):
              %82 = arith.subf %in, %in_6 : f16
              %83 = math.exp %82 : f16
              linalg.yield %83 : f16
            }
            loom.semaphore_give %69 : memref<16x1x32x1xf16>
            loom.semaphore_give %34 : memref<1x32x1xf16>
            linalg.fill ins(%cst : f16) outs(%33 : memref<1x32x1xf16>)
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1, d2, d3)>], iterator_types = ["reduction", "parallel", "parallel", "parallel"]} ins(%75 : memref<16x1x32x1xf16>) outs(%33 : memref<1x32x1xf16>) {
            ^bb0(%in: f16, %out: f16):
              %82 = arith.addf %in, %out : f16
              linalg.yield %82 : f16
            }
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%75, %33 : memref<16x1x32x1xf16>, memref<1x32x1xf16>) outs(%75 : memref<16x1x32x1xf16>) {
            ^bb0(%in: f16, %in_6: f16, %out: f16):
              %82 = arith.divf %in, %in_6 : f16
              linalg.yield %82 : f16
            }
            loom.semaphore_give %33 : memref<1x32x1xf16>
            %81 = loom.broadcast ins(%75 : memref<16x1x32x1xf16>) outs(%79 : memref<16x1x32x32xf16>) dim(3) -> memref<16x1x32x128xf16, strided<[?, ?, ?, ?], offset: ?>>
            loom.semaphore_give %75 : memref<16x1x32x1xf16>
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%71, %81 : memref<16x1x32x128xf16>, memref<16x1x32x128xf16, strided<[?, ?, ?, ?], offset: ?>>) outs(%77 : memref<16x1x32x128xf16>) {
            ^bb0(%in: f16, %in_6: f16, %out: f16):
              %82 = arith.mulf %in, %in_6 : f16
              linalg.yield %82 : f16
            }
            loom.semaphore_give %79 : memref<16x1x32x32xf16>
            loom.semaphore_give %71 : memref<16x1x32x128xf16>
            linalg.fill ins(%cst : f16) outs(%73 : memref<1x32x128xf16>)
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1, d2, d3)>], iterator_types = ["reduction", "parallel", "parallel", "parallel"]} ins(%77 : memref<16x1x32x128xf16>) outs(%73 : memref<1x32x128xf16>) {
            ^bb0(%in: f16, %out: f16):
              %82 = arith.addf %in, %out : f16
              linalg.yield %82 : f16
            }
            loom.semaphore_give %77 : memref<16x1x32x128xf16>
            %reinterpret_cast_5 = memref.reinterpret_cast %arg2 to offset: [%26], sizes: [1, 32, 128], strides: [4096, 128, 1] : memref<16x32x128xf16> to memref<1x32x128xf16, strided<[4096, 128, 1], offset: ?>>
            loom.copy %73, %reinterpret_cast_5 src_mem_space @mem_L1 dst_mem_space @mem_DRAM, broadcast : [1, 1] region : (UL : [%arg5, %58], LR : [%arg5, %58]) : memref<1x32x128xf16> to memref<1x32x128xf16, strided<[4096, 128, 1], offset: ?>>
            loom.semaphore_give %73 : memref<1x32x128xf16>
          }
        } {loom.block_sym = @tile_b, loom.iter_type = #loom.iter_type<temporal>}
        scf.reduce 
      } {loom.block_syms = [@tile_b, @tile_s, @tile_s], loom.iter_types = [#loom.iter_type<spatial>, #loom.iter_type<spatial>, #loom.iter_type<spatial>], loom.logical_levels = [1, 0, 0], loom.physical_dims = [@dim_y, @dim_x, @dim_y]}
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
          linalg.fill ins(%cst : f16) outs(%27 : memref<1x32x128xf16>)
          %28 = loom.alloc [1, 32, 1] on @L1 : memref<1x32x1xf16>
          %29 = loom.semaphore_take %28 : memref<1x32x1xf16> -> memref<1x32x1xf16>
          %30 = loom.semaphore_take %28 : memref<1x32x1xf16> -> memref<1x32x1xf16>
          %31 = loom.semaphore_take %28 : memref<1x32x1xf16> -> memref<1x32x1xf16>
          linalg.fill ins(%cst_0 : f16) outs(%31 : memref<1x32x1xf16>)
          %32 = loom.alloc [1, 32, 1] on @L1 : memref<1x32x1xf16>
          %33 = loom.semaphore_take %32 : memref<1x32x1xf16> -> memref<1x32x1xf16>
          linalg.fill ins(%cst_1 : f16) outs(%33 : memref<1x32x1xf16>)
          %34 = loom.alloc [1, 32, 128] on @L1 : memref<1x32x128xf16>
          %35 = loom.semaphore_take %34 : memref<1x32x128xf16> -> memref<1x32x128xf16>
          %36 = loom.alloc [1, 32, 1] on @L1 : memref<1x32x1xf16>
          %37 = loom.semaphore_take %36 : memref<1x32x1xf16> -> memref<1x32x1xf16>
          %38 = loom.alloc [1, 32, 1] on @L1 : memref<1x32x1xf16>
          %39 = loom.semaphore_take %38 : memref<1x32x1xf16> -> memref<1x32x1xf16>
          %40 = loom.alloc [1, 32, 1] on @L1 : memref<1x32x1xf16>
          %41 = loom.semaphore_take %40 : memref<1x32x1xf16> -> memref<1x32x1xf16>
          %42 = loom.alloc [1, 128, 512] on @L1 : memref<1x128x512xf16>
          %43 = loom.semaphore_take %42 : memref<1x128x512xf16> -> memref<1x128x512xf16>
          %44 = loom.alloc [1, 32, 512] on @L1 : memref<1x32x512xf16>
          %45 = loom.semaphore_take %44 : memref<1x32x512xf16> -> memref<1x32x512xf16>
          %46 = loom.alloc [1, 32, 32] on @L1 : memref<1x32x32xf16>
          %47 = loom.semaphore_take %46 : memref<1x32x32xf16> -> memref<1x32x32xf16>
          %48 = loom.semaphore_take %46 : memref<1x32x32xf16> -> memref<1x32x32xf16>
          %49 = loom.semaphore_take %46 : memref<1x32x32xf16> -> memref<1x32x32xf16>
          %50 = loom.alloc [1, 512, 128] on @L1 : memref<1x512x128xf16>
          %51 = loom.semaphore_take %50 : memref<1x512x128xf16> -> memref<1x512x128xf16>
          %52 = arith.muli %arg6, %c1048576 overflow<nsw> : index
          %53 = arith.addi %52, %25 : index
          %reinterpret_cast_3 = memref.reinterpret_cast %arg0 to offset: [%53], sizes: [1, 128, 512], strides: [1048576, 8192, 1] : memref<16x128x8192xf16> to memref<1x128x512xf16, strided<[1048576, 8192, 1], offset: ?>>
          loom.copy %reinterpret_cast_3, %43 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] region : (UL : [%arg4, %arg5], LR : [%arg4, %arg5]) : memref<1x128x512xf16, strided<[1048576, 8192, 1], offset: ?>> to memref<1x128x512xf16>
          linalg.fill ins(%cst : f16) outs(%45 : memref<1x32x512xf16>)
          linalg.batch_matmul ins(%23, %43 : memref<1x32x128xf16>, memref<1x128x512xf16>) outs(%45 : memref<1x32x512xf16>)
          loom.semaphore_give %43 : memref<1x128x512xf16>
          loom.semaphore_give %23 : memref<1x32x128xf16>
          linalg.fill ins(%cst_1 : f16) outs(%37 : memref<1x32x1xf16>)
          linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, 0)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%45 : memref<1x32x512xf16>) outs(%37 : memref<1x32x1xf16>) {
          ^bb0(%in: f16, %out: f16):
            %76 = arith.maximumf %in, %out : f16
            linalg.yield %76 : f16
          }
          linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%33, %37 : memref<1x32x1xf16>, memref<1x32x1xf16>) outs(%37 : memref<1x32x1xf16>) {
          ^bb0(%in: f16, %in_5: f16, %out: f16):
            %76 = arith.mulf %in_5, %cst_2 : f16
            %77 = arith.cmpf ogt, %in, %76 : f16
            %78 = arith.select %77, %in, %76 : f16
            linalg.yield %78 : f16
          }
          %54 = loom.broadcast ins(%37 : memref<1x32x1xf16>) outs(%49 : memref<1x32x32xf16>) dim(2) -> memref<1x32x512xf16, strided<[?, ?, ?], offset: ?>>
          linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%45, %54 : memref<1x32x512xf16>, memref<1x32x512xf16, strided<[?, ?, ?], offset: ?>>) outs(%45 : memref<1x32x512xf16>) {
          ^bb0(%in: f16, %in_5: f16, %out: f16):
            %76 = arith.mulf %in, %cst_2 : f16
            %77 = arith.subf %76, %in_5 : f16
            %78 = math.exp %77 : f16
            linalg.yield %78 : f16
          }
          loom.semaphore_give %49 : memref<1x32x32xf16>
          linalg.fill ins(%cst : f16) outs(%39 : memref<1x32x1xf16>)
          linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, 0)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%45 : memref<1x32x512xf16>) outs(%39 : memref<1x32x1xf16>) {
          ^bb0(%in: f16, %out: f16):
            %76 = arith.addf %in, %out : f16
            linalg.yield %76 : f16
          }
          linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%33, %37 : memref<1x32x1xf16>, memref<1x32x1xf16>) outs(%41 : memref<1x32x1xf16>) {
          ^bb0(%in: f16, %in_5: f16, %out: f16):
            %76 = arith.subf %in, %in_5 : f16
            %77 = math.exp %76 : f16
            linalg.yield %77 : f16
          }
          linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%31, %41, %39 : memref<1x32x1xf16>, memref<1x32x1xf16>, memref<1x32x1xf16>) outs(%31 : memref<1x32x1xf16>) {
          ^bb0(%in: f16, %in_5: f16, %in_6: f16, %out: f16):
            %76 = arith.mulf %in, %in_5 : f16
            %77 = arith.addf %76, %in_6 : f16
            linalg.yield %77 : f16
          }
          loom.semaphore_give %39 : memref<1x32x1xf16>
          %55 = loom.broadcast ins(%41 : memref<1x32x1xf16>) outs(%48 : memref<1x32x32xf16>) dim(2) -> memref<1x32x128xf16, strided<[?, ?, ?], offset: ?>>
          loom.semaphore_give %41 : memref<1x32x1xf16>
          %56 = arith.muli %21, %c65536 : index
          %57 = arith.addi %52, %56 : index
          %reinterpret_cast_4 = memref.reinterpret_cast %arg1 to offset: [%57], sizes: [1, 512, 128], strides: [1048576, 128, 1] : memref<16x8192x128xf16> to memref<1x512x128xf16, strided<[1048576, 128, 1], offset: ?>>
          loom.copy %reinterpret_cast_4, %51 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] region : (UL : [%arg4, %arg5], LR : [%arg4, %arg5]) : memref<1x512x128xf16, strided<[1048576, 128, 1], offset: ?>> to memref<1x512x128xf16>
          linalg.fill ins(%cst : f16) outs(%35 : memref<1x32x128xf16>)
          linalg.batch_matmul ins(%45, %51 : memref<1x32x512xf16>, memref<1x512x128xf16>) outs(%35 : memref<1x32x128xf16>)
          loom.semaphore_give %51 : memref<1x512x128xf16>
          loom.semaphore_give %45 : memref<1x32x512xf16>
          linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%35, %27, %55 : memref<1x32x128xf16>, memref<1x32x128xf16>, memref<1x32x128xf16, strided<[?, ?, ?], offset: ?>>) outs(%27 : memref<1x32x128xf16>) {
          ^bb0(%in: f16, %in_5: f16, %in_6: f16, %out: f16):
            %76 = arith.mulf %in_5, %in_6 : f16
            %77 = arith.addf %in, %76 : f16
            linalg.yield %77 : f16
          }
          loom.semaphore_give %48 : memref<1x32x32xf16>
          loom.semaphore_give %35 : memref<1x32x128xf16>
          linalg.copy ins(%37 : memref<1x32x1xf16>) outs(%33 : memref<1x32x1xf16>)
          loom.semaphore_give %37 : memref<1x32x1xf16>
          %58 = loom.alloc [1, 32, 1] on @L1 : memref<1x32x1xf16>
          %59 = loom.semaphore_take %58 : memref<1x32x1xf16> -> memref<1x32x1xf16>
          linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%31, %33 : memref<1x32x1xf16>, memref<1x32x1xf16>) outs(%59 : memref<1x32x1xf16>) {
          ^bb0(%in: f16, %in_5: f16, %out: f16):
            %76 = math.log %in : f16
            %77 = arith.addf %76, %in_5 : f16
            linalg.yield %77 : f16
          }
          loom.semaphore_give %33 : memref<1x32x1xf16>
          %60 = loom.broadcast ins(%31 : memref<1x32x1xf16>) outs(%47 : memref<1x32x32xf16>) dim(2) -> memref<1x32x128xf16, strided<[?, ?, ?], offset: ?>>
          loom.semaphore_give %31 : memref<1x32x1xf16>
          %61 = loom.alloc [1, 32, 128] on @L1 : memref<1x32x128xf16>
          %62 = loom.semaphore_take %61 : memref<1x32x128xf16> -> memref<1x32x128xf16>
          linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%27, %60 : memref<1x32x128xf16>, memref<1x32x128xf16, strided<[?, ?, ?], offset: ?>>) outs(%62 : memref<1x32x128xf16>) {
          ^bb0(%in: f16, %in_5: f16, %out: f16):
            %76 = arith.divf %in, %in_5 : f16
            linalg.yield %76 : f16
          }
          loom.semaphore_give %47 : memref<1x32x32xf16>
          loom.semaphore_give %27 : memref<1x32x128xf16>
          %63 = loom.alloc [16, 1, 32, 1] on @L1 : memref<16x1x32x1xf16>
          %64 = loom.semaphore_take %63 : memref<16x1x32x1xf16> -> memref<16x1x32x1xf16>
          loom.gather ins(%59 : memref<1x32x1xf16>) outs(%64 : memref<16x1x32x1xf16>) across(%arg4 : index) region : (UL : [%c0, %arg5], LR : [%c7, %arg5])
          loom.semaphore_give %59 : memref<1x32x1xf16>
          %65 = loom.alloc [16, 1, 32, 128] on @L1 : memref<16x1x32x128xf16>
          %66 = loom.semaphore_take %65 : memref<16x1x32x128xf16> -> memref<16x1x32x128xf16>
          loom.gather ins(%62 : memref<1x32x128xf16>) outs(%66 : memref<16x1x32x128xf16>) across(%arg4 : index) region : (UL : [%c0, %arg5], LR : [%c7, %arg5])
          loom.semaphore_give %62 : memref<1x32x128xf16>
          %67 = loom.alloc [1, 32, 128] on @L1 : memref<1x32x128xf16>
          %68 = loom.semaphore_take %67 : memref<1x32x128xf16> -> memref<1x32x128xf16>
          %69 = loom.alloc [16, 1, 32, 1] on @L1 : memref<16x1x32x1xf16>
          %70 = loom.semaphore_take %69 : memref<16x1x32x1xf16> -> memref<16x1x32x1xf16>
          %71 = loom.alloc [16, 1, 32, 128] on @L1 : memref<16x1x32x128xf16>
          %72 = loom.semaphore_take %71 : memref<16x1x32x128xf16> -> memref<16x1x32x128xf16>
          %73 = loom.alloc [16, 1, 32, 32] on @L1 : memref<16x1x32x32xf16>
          %74 = loom.semaphore_take %73 : memref<16x1x32x32xf16> -> memref<16x1x32x32xf16>
          %75 = arith.cmpi eq, %arg4, %c0 : index
          scf.if %75 {
            linalg.fill ins(%cst_1 : f16) outs(%30 : memref<1x32x1xf16>)
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1, d2, d3)>], iterator_types = ["reduction", "parallel", "parallel", "parallel"]} ins(%64 : memref<16x1x32x1xf16>) outs(%30 : memref<1x32x1xf16>) {
            ^bb0(%in: f16, %out: f16):
              %77 = arith.maximumf %in, %out : f16
              linalg.yield %77 : f16
            }
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%64, %30 : memref<16x1x32x1xf16>, memref<1x32x1xf16>) outs(%70 : memref<16x1x32x1xf16>) {
            ^bb0(%in: f16, %in_6: f16, %out: f16):
              %77 = arith.subf %in, %in_6 : f16
              %78 = math.exp %77 : f16
              linalg.yield %78 : f16
            }
            loom.semaphore_give %64 : memref<16x1x32x1xf16>
            loom.semaphore_give %30 : memref<1x32x1xf16>
            linalg.fill ins(%cst : f16) outs(%29 : memref<1x32x1xf16>)
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1, d2, d3)>], iterator_types = ["reduction", "parallel", "parallel", "parallel"]} ins(%70 : memref<16x1x32x1xf16>) outs(%29 : memref<1x32x1xf16>) {
            ^bb0(%in: f16, %out: f16):
              %77 = arith.addf %in, %out : f16
              linalg.yield %77 : f16
            }
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%70, %29 : memref<16x1x32x1xf16>, memref<1x32x1xf16>) outs(%70 : memref<16x1x32x1xf16>) {
            ^bb0(%in: f16, %in_6: f16, %out: f16):
              %77 = arith.divf %in, %in_6 : f16
              linalg.yield %77 : f16
            }
            loom.semaphore_give %29 : memref<1x32x1xf16>
            %76 = loom.broadcast ins(%70 : memref<16x1x32x1xf16>) outs(%74 : memref<16x1x32x32xf16>) dim(3) -> memref<16x1x32x128xf16, strided<[?, ?, ?, ?], offset: ?>>
            loom.semaphore_give %70 : memref<16x1x32x1xf16>
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%66, %76 : memref<16x1x32x128xf16>, memref<16x1x32x128xf16, strided<[?, ?, ?, ?], offset: ?>>) outs(%72 : memref<16x1x32x128xf16>) {
            ^bb0(%in: f16, %in_6: f16, %out: f16):
              %77 = arith.mulf %in, %in_6 : f16
              linalg.yield %77 : f16
            }
            loom.semaphore_give %74 : memref<16x1x32x32xf16>
            loom.semaphore_give %66 : memref<16x1x32x128xf16>
            linalg.fill ins(%cst : f16) outs(%68 : memref<1x32x128xf16>)
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1, d2, d3)>], iterator_types = ["reduction", "parallel", "parallel", "parallel"]} ins(%72 : memref<16x1x32x128xf16>) outs(%68 : memref<1x32x128xf16>) {
            ^bb0(%in: f16, %out: f16):
              %77 = arith.addf %in, %out : f16
              linalg.yield %77 : f16
            }
            loom.semaphore_give %72 : memref<16x1x32x128xf16>
            %reinterpret_cast_5 = memref.reinterpret_cast %arg2 to offset: [%24], sizes: [1, 32, 128], strides: [4096, 128, 1] : memref<16x32x128xf16> to memref<1x32x128xf16, strided<[4096, 128, 1], offset: ?>>
            loom.copy %68, %reinterpret_cast_5 src_mem_space @mem_L1 dst_mem_space @mem_DRAM, broadcast : [1, 1] region : (UL : [%arg4, %arg5], LR : [%arg4, %arg5]) : memref<1x32x128xf16> to memref<1x32x128xf16, strided<[4096, 128, 1], offset: ?>>
            loom.semaphore_give %68 : memref<1x32x128xf16>
          }
        } {loom.block_sym = @tile_b, loom.iter_type = #loom.iter_type<temporal>}
        scf.reduce 
      } {loom.block_syms = [@tile_s, @tile_s], loom.iter_types = [#loom.iter_type<spatial>, #loom.iter_type<spatial>], loom.logical_levels = [0, 0], loom.physical_dims = [@dim_x, @dim_y]}
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
            linalg.fill ins(%cst : f16) outs(%29 : memref<1x32x128xf16>)
            %30 = loom.alloc [1, 32, 1] on @L1 : memref<1x32x1xf16>
            %31 = loom.semaphore_take %30 : memref<1x32x1xf16> -> memref<1x32x1xf16>
            %32 = loom.semaphore_take %30 : memref<1x32x1xf16> -> memref<1x32x1xf16>
            %33 = loom.semaphore_take %30 : memref<1x32x1xf16> -> memref<1x32x1xf16>
            linalg.fill ins(%cst_0 : f16) outs(%33 : memref<1x32x1xf16>)
            %34 = loom.alloc [1, 32, 1] on @L1 : memref<1x32x1xf16>
            %35 = loom.semaphore_take %34 : memref<1x32x1xf16> -> memref<1x32x1xf16>
            linalg.fill ins(%cst_1 : f16) outs(%35 : memref<1x32x1xf16>)
            %36 = loom.alloc [1, 32, 128] on @L1 : memref<1x32x128xf16>
            %37 = loom.semaphore_take %36 : memref<1x32x128xf16> -> memref<1x32x128xf16>
            %38 = loom.alloc [1, 32, 1] on @L1 : memref<1x32x1xf16>
            %39 = loom.semaphore_take %38 : memref<1x32x1xf16> -> memref<1x32x1xf16>
            %40 = loom.alloc [1, 32, 1] on @L1 : memref<1x32x1xf16>
            %41 = loom.semaphore_take %40 : memref<1x32x1xf16> -> memref<1x32x1xf16>
            %42 = loom.alloc [1, 32, 1] on @L1 : memref<1x32x1xf16>
            %43 = loom.semaphore_take %42 : memref<1x32x1xf16> -> memref<1x32x1xf16>
            %44 = loom.alloc [1, 128, 512] on @L1 : memref<1x128x512xf16>
            %45 = loom.semaphore_take %44 : memref<1x128x512xf16> -> memref<1x128x512xf16>
            %46 = loom.alloc [1, 32, 512] on @L1 : memref<1x32x512xf16>
            %47 = loom.semaphore_take %46 : memref<1x32x512xf16> -> memref<1x32x512xf16>
            %48 = loom.alloc [1, 32, 32] on @L1 : memref<1x32x32xf16>
            %49 = loom.semaphore_take %48 : memref<1x32x32xf16> -> memref<1x32x32xf16>
            %50 = loom.semaphore_take %48 : memref<1x32x32xf16> -> memref<1x32x32xf16>
            %51 = loom.semaphore_take %48 : memref<1x32x32xf16> -> memref<1x32x32xf16>
            %52 = loom.alloc [1, 512, 128] on @L1 : memref<1x512x128xf16>
            %53 = loom.semaphore_take %52 : memref<1x512x128xf16> -> memref<1x512x128xf16>
            %54 = arith.muli %21, %c1048576 overflow<nsw> : index
            %55 = arith.addi %54, %27 : index
            %reinterpret_cast_3 = memref.reinterpret_cast %arg0 to offset: [%55], sizes: [1, 128, 512], strides: [1048576, 8192, 1] : memref<16x128x8192xf16> to memref<1x128x512xf16, strided<[1048576, 8192, 1], offset: ?>>
            loom.copy %reinterpret_cast_3, %45 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] region : (UL : [%arg4, %arg5], LR : [%arg4, %arg5]) : memref<1x128x512xf16, strided<[1048576, 8192, 1], offset: ?>> to memref<1x128x512xf16>
            linalg.fill ins(%cst : f16) outs(%47 : memref<1x32x512xf16>)
            linalg.batch_matmul ins(%25, %45 : memref<1x32x128xf16>, memref<1x128x512xf16>) outs(%47 : memref<1x32x512xf16>)
            loom.semaphore_give %45 : memref<1x128x512xf16>
            loom.semaphore_give %25 : memref<1x32x128xf16>
            linalg.fill ins(%cst_1 : f16) outs(%39 : memref<1x32x1xf16>)
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, 0)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%47 : memref<1x32x512xf16>) outs(%39 : memref<1x32x1xf16>) {
            ^bb0(%in: f16, %out: f16):
              %78 = arith.maximumf %in, %out : f16
              linalg.yield %78 : f16
            }
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%35, %39 : memref<1x32x1xf16>, memref<1x32x1xf16>) outs(%39 : memref<1x32x1xf16>) {
            ^bb0(%in: f16, %in_6: f16, %out: f16):
              %78 = arith.mulf %in_6, %cst_2 : f16
              %79 = arith.cmpf ogt, %in, %78 : f16
              %80 = arith.select %79, %in, %78 : f16
              linalg.yield %80 : f16
            }
            %56 = loom.broadcast ins(%39 : memref<1x32x1xf16>) outs(%51 : memref<1x32x32xf16>) dim(2) -> memref<1x32x512xf16, strided<[?, ?, ?], offset: ?>>
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%47, %56 : memref<1x32x512xf16>, memref<1x32x512xf16, strided<[?, ?, ?], offset: ?>>) outs(%47 : memref<1x32x512xf16>) {
            ^bb0(%in: f16, %in_6: f16, %out: f16):
              %78 = arith.mulf %in, %cst_2 : f16
              %79 = arith.subf %78, %in_6 : f16
              %80 = math.exp %79 : f16
              linalg.yield %80 : f16
            }
            loom.semaphore_give %51 : memref<1x32x32xf16>
            linalg.fill ins(%cst : f16) outs(%41 : memref<1x32x1xf16>)
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, 0)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%47 : memref<1x32x512xf16>) outs(%41 : memref<1x32x1xf16>) {
            ^bb0(%in: f16, %out: f16):
              %78 = arith.addf %in, %out : f16
              linalg.yield %78 : f16
            }
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%35, %39 : memref<1x32x1xf16>, memref<1x32x1xf16>) outs(%43 : memref<1x32x1xf16>) {
            ^bb0(%in: f16, %in_6: f16, %out: f16):
              %78 = arith.subf %in, %in_6 : f16
              %79 = math.exp %78 : f16
              linalg.yield %79 : f16
            }
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%33, %43, %41 : memref<1x32x1xf16>, memref<1x32x1xf16>, memref<1x32x1xf16>) outs(%33 : memref<1x32x1xf16>) {
            ^bb0(%in: f16, %in_6: f16, %in_7: f16, %out: f16):
              %78 = arith.mulf %in, %in_6 : f16
              %79 = arith.addf %78, %in_7 : f16
              linalg.yield %79 : f16
            }
            loom.semaphore_give %41 : memref<1x32x1xf16>
            %57 = loom.broadcast ins(%43 : memref<1x32x1xf16>) outs(%50 : memref<1x32x32xf16>) dim(2) -> memref<1x32x128xf16, strided<[?, ?, ?], offset: ?>>
            loom.semaphore_give %43 : memref<1x32x1xf16>
            %58 = arith.muli %23, %c65536 : index
            %59 = arith.addi %54, %58 : index
            %reinterpret_cast_4 = memref.reinterpret_cast %arg1 to offset: [%59], sizes: [1, 512, 128], strides: [1048576, 128, 1] : memref<16x8192x128xf16> to memref<1x512x128xf16, strided<[1048576, 128, 1], offset: ?>>
            loom.copy %reinterpret_cast_4, %53 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] region : (UL : [%arg4, %arg5], LR : [%arg4, %arg5]) : memref<1x512x128xf16, strided<[1048576, 128, 1], offset: ?>> to memref<1x512x128xf16>
            linalg.fill ins(%cst : f16) outs(%37 : memref<1x32x128xf16>)
            linalg.batch_matmul ins(%47, %53 : memref<1x32x512xf16>, memref<1x512x128xf16>) outs(%37 : memref<1x32x128xf16>)
            loom.semaphore_give %53 : memref<1x512x128xf16>
            loom.semaphore_give %47 : memref<1x32x512xf16>
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%37, %29, %57 : memref<1x32x128xf16>, memref<1x32x128xf16>, memref<1x32x128xf16, strided<[?, ?, ?], offset: ?>>) outs(%29 : memref<1x32x128xf16>) {
            ^bb0(%in: f16, %in_6: f16, %in_7: f16, %out: f16):
              %78 = arith.mulf %in_6, %in_7 : f16
              %79 = arith.addf %in, %78 : f16
              linalg.yield %79 : f16
            }
            loom.semaphore_give %50 : memref<1x32x32xf16>
            loom.semaphore_give %37 : memref<1x32x128xf16>
            linalg.copy ins(%39 : memref<1x32x1xf16>) outs(%35 : memref<1x32x1xf16>)
            loom.semaphore_give %39 : memref<1x32x1xf16>
            %60 = loom.alloc [1, 32, 1] on @L1 : memref<1x32x1xf16>
            %61 = loom.semaphore_take %60 : memref<1x32x1xf16> -> memref<1x32x1xf16>
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%33, %35 : memref<1x32x1xf16>, memref<1x32x1xf16>) outs(%61 : memref<1x32x1xf16>) {
            ^bb0(%in: f16, %in_6: f16, %out: f16):
              %78 = math.log %in : f16
              %79 = arith.addf %78, %in_6 : f16
              linalg.yield %79 : f16
            }
            loom.semaphore_give %35 : memref<1x32x1xf16>
            %62 = loom.broadcast ins(%33 : memref<1x32x1xf16>) outs(%49 : memref<1x32x32xf16>) dim(2) -> memref<1x32x128xf16, strided<[?, ?, ?], offset: ?>>
            loom.semaphore_give %33 : memref<1x32x1xf16>
            %63 = loom.alloc [1, 32, 128] on @L1 : memref<1x32x128xf16>
            %64 = loom.semaphore_take %63 : memref<1x32x128xf16> -> memref<1x32x128xf16>
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%29, %62 : memref<1x32x128xf16>, memref<1x32x128xf16, strided<[?, ?, ?], offset: ?>>) outs(%64 : memref<1x32x128xf16>) {
            ^bb0(%in: f16, %in_6: f16, %out: f16):
              %78 = arith.divf %in, %in_6 : f16
              linalg.yield %78 : f16
            }
            loom.semaphore_give %49 : memref<1x32x32xf16>
            loom.semaphore_give %29 : memref<1x32x128xf16>
            %65 = loom.alloc [16, 1, 32, 1] on @L1 : memref<16x1x32x1xf16>
            %66 = loom.semaphore_take %65 : memref<16x1x32x1xf16> -> memref<16x1x32x1xf16>
            loom.gather ins(%61 : memref<1x32x1xf16>) outs(%66 : memref<16x1x32x1xf16>) across(%c0 : index) region : (UL : [%arg4, %arg5], LR : [%arg4, %arg5])
            loom.semaphore_give %61 : memref<1x32x1xf16>
            %67 = loom.alloc [16, 1, 32, 128] on @L1 : memref<16x1x32x128xf16>
            %68 = loom.semaphore_take %67 : memref<16x1x32x128xf16> -> memref<16x1x32x128xf16>
            loom.gather ins(%64 : memref<1x32x128xf16>) outs(%68 : memref<16x1x32x128xf16>) across(%c0 : index) region : (UL : [%arg4, %arg5], LR : [%arg4, %arg5])
            loom.semaphore_give %64 : memref<1x32x128xf16>
            %69 = loom.alloc [1, 32, 128] on @L1 : memref<1x32x128xf16>
            %70 = loom.semaphore_take %69 : memref<1x32x128xf16> -> memref<1x32x128xf16>
            %71 = loom.alloc [16, 1, 32, 1] on @L1 : memref<16x1x32x1xf16>
            %72 = loom.semaphore_take %71 : memref<16x1x32x1xf16> -> memref<16x1x32x1xf16>
            %73 = loom.alloc [16, 1, 32, 128] on @L1 : memref<16x1x32x128xf16>
            %74 = loom.semaphore_take %73 : memref<16x1x32x128xf16> -> memref<16x1x32x128xf16>
            %75 = loom.alloc [16, 1, 32, 32] on @L1 : memref<16x1x32x32xf16>
            %76 = loom.semaphore_take %75 : memref<16x1x32x32xf16> -> memref<16x1x32x32xf16>
            linalg.fill ins(%cst_1 : f16) outs(%32 : memref<1x32x1xf16>)
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1, d2, d3)>], iterator_types = ["reduction", "parallel", "parallel", "parallel"]} ins(%66 : memref<16x1x32x1xf16>) outs(%32 : memref<1x32x1xf16>) {
            ^bb0(%in: f16, %out: f16):
              %78 = arith.maximumf %in, %out : f16
              linalg.yield %78 : f16
            }
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%66, %32 : memref<16x1x32x1xf16>, memref<1x32x1xf16>) outs(%72 : memref<16x1x32x1xf16>) {
            ^bb0(%in: f16, %in_6: f16, %out: f16):
              %78 = arith.subf %in, %in_6 : f16
              %79 = math.exp %78 : f16
              linalg.yield %79 : f16
            }
            loom.semaphore_give %66 : memref<16x1x32x1xf16>
            loom.semaphore_give %32 : memref<1x32x1xf16>
            linalg.fill ins(%cst : f16) outs(%31 : memref<1x32x1xf16>)
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1, d2, d3)>], iterator_types = ["reduction", "parallel", "parallel", "parallel"]} ins(%72 : memref<16x1x32x1xf16>) outs(%31 : memref<1x32x1xf16>) {
            ^bb0(%in: f16, %out: f16):
              %78 = arith.addf %in, %out : f16
              linalg.yield %78 : f16
            }
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%72, %31 : memref<16x1x32x1xf16>, memref<1x32x1xf16>) outs(%72 : memref<16x1x32x1xf16>) {
            ^bb0(%in: f16, %in_6: f16, %out: f16):
              %78 = arith.divf %in, %in_6 : f16
              linalg.yield %78 : f16
            }
            loom.semaphore_give %31 : memref<1x32x1xf16>
            %77 = loom.broadcast ins(%72 : memref<16x1x32x1xf16>) outs(%76 : memref<16x1x32x32xf16>) dim(3) -> memref<16x1x32x128xf16, strided<[?, ?, ?, ?], offset: ?>>
            loom.semaphore_give %72 : memref<16x1x32x1xf16>
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%68, %77 : memref<16x1x32x128xf16>, memref<16x1x32x128xf16, strided<[?, ?, ?, ?], offset: ?>>) outs(%74 : memref<16x1x32x128xf16>) {
            ^bb0(%in: f16, %in_6: f16, %out: f16):
              %78 = arith.mulf %in, %in_6 : f16
              linalg.yield %78 : f16
            }
            loom.semaphore_give %76 : memref<16x1x32x32xf16>
            loom.semaphore_give %68 : memref<16x1x32x128xf16>
            linalg.fill ins(%cst : f16) outs(%70 : memref<1x32x128xf16>)
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1, d2, d3)>], iterator_types = ["reduction", "parallel", "parallel", "parallel"]} ins(%74 : memref<16x1x32x128xf16>) outs(%70 : memref<1x32x128xf16>) {
            ^bb0(%in: f16, %out: f16):
              %78 = arith.addf %in, %out : f16
              linalg.yield %78 : f16
            }
            loom.semaphore_give %74 : memref<16x1x32x128xf16>
            %reinterpret_cast_5 = memref.reinterpret_cast %arg2 to offset: [%26], sizes: [1, 32, 128], strides: [4096, 128, 1] : memref<16x32x128xf16> to memref<1x32x128xf16, strided<[4096, 128, 1], offset: ?>>
            loom.copy %70, %reinterpret_cast_5 src_mem_space @mem_L1 dst_mem_space @mem_DRAM, broadcast : [1, 1] region : (UL : [%arg4, %arg5], LR : [%arg4, %arg5]) : memref<1x32x128xf16> to memref<1x32x128xf16, strided<[4096, 128, 1], offset: ?>>
            loom.semaphore_give %70 : memref<1x32x128xf16>
          } {loom.block_sym = @tile_s, loom.iter_type = #loom.iter_type<temporal>}
        } {loom.block_sym = @tile_b, loom.iter_type = #loom.iter_type<temporal>}
        scf.reduce 
      } {loom.block_syms = [@tile_b, @tile_s], loom.iter_types = [#loom.iter_type<spatial>, #loom.iter_type<spatial>], loom.logical_levels = [1, 0], loom.physical_dims = [@dim_x, @dim_y]}
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
          linalg.fill ins(%cst : f16) outs(%31 : memref<1x32x128xf16>)
          %32 = loom.alloc [1, 32, 1] on @L1 : memref<1x32x1xf16>
          %33 = loom.semaphore_take %32 : memref<1x32x1xf16> -> memref<1x32x1xf16>
          %34 = loom.semaphore_take %32 : memref<1x32x1xf16> -> memref<1x32x1xf16>
          %35 = loom.semaphore_take %32 : memref<1x32x1xf16> -> memref<1x32x1xf16>
          linalg.fill ins(%cst_0 : f16) outs(%35 : memref<1x32x1xf16>)
          %36 = loom.alloc [1, 32, 1] on @L1 : memref<1x32x1xf16>
          %37 = loom.semaphore_take %36 : memref<1x32x1xf16> -> memref<1x32x1xf16>
          linalg.fill ins(%cst_1 : f16) outs(%37 : memref<1x32x1xf16>)
          %38 = loom.alloc [1, 32, 128] on @L1 : memref<1x32x128xf16>
          %39 = loom.semaphore_take %38 : memref<1x32x128xf16> -> memref<1x32x128xf16>
          %40 = loom.alloc [1, 32, 1] on @L1 : memref<1x32x1xf16>
          %41 = loom.semaphore_take %40 : memref<1x32x1xf16> -> memref<1x32x1xf16>
          %42 = loom.alloc [1, 32, 1] on @L1 : memref<1x32x1xf16>
          %43 = loom.semaphore_take %42 : memref<1x32x1xf16> -> memref<1x32x1xf16>
          %44 = loom.alloc [1, 32, 1] on @L1 : memref<1x32x1xf16>
          %45 = loom.semaphore_take %44 : memref<1x32x1xf16> -> memref<1x32x1xf16>
          %46 = loom.alloc [1, 128, 512] on @L1 : memref<1x128x512xf16>
          %47 = loom.semaphore_take %46 : memref<1x128x512xf16> -> memref<1x128x512xf16>
          %48 = loom.alloc [1, 32, 512] on @L1 : memref<1x32x512xf16>
          %49 = loom.semaphore_take %48 : memref<1x32x512xf16> -> memref<1x32x512xf16>
          %50 = loom.alloc [1, 32, 32] on @L1 : memref<1x32x32xf16>
          %51 = loom.semaphore_take %50 : memref<1x32x32xf16> -> memref<1x32x32xf16>
          %52 = loom.semaphore_take %50 : memref<1x32x32xf16> -> memref<1x32x32xf16>
          %53 = loom.semaphore_take %50 : memref<1x32x32xf16> -> memref<1x32x32xf16>
          %54 = loom.alloc [1, 512, 128] on @L1 : memref<1x512x128xf16>
          %55 = loom.semaphore_take %54 : memref<1x512x128xf16> -> memref<1x512x128xf16>
          %56 = arith.muli %21, %c1048576 overflow<nsw> : index
          %57 = arith.addi %56, %29 : index
          %reinterpret_cast_3 = memref.reinterpret_cast %arg0 to offset: [%57], sizes: [1, 128, 512], strides: [1048576, 8192, 1] : memref<16x128x8192xf16> to memref<1x128x512xf16, strided<[1048576, 8192, 1], offset: ?>>
          %58 = arith.addi %arg5, %27 : index
          loom.copy %reinterpret_cast_3, %47 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] region : (UL : [%58, %arg6], LR : [%58, %arg6]) : memref<1x128x512xf16, strided<[1048576, 8192, 1], offset: ?>> to memref<1x128x512xf16>
          linalg.fill ins(%cst : f16) outs(%49 : memref<1x32x512xf16>)
          linalg.batch_matmul ins(%25, %47 : memref<1x32x128xf16>, memref<1x128x512xf16>) outs(%49 : memref<1x32x512xf16>)
          loom.semaphore_give %47 : memref<1x128x512xf16>
          loom.semaphore_give %25 : memref<1x32x128xf16>
          linalg.fill ins(%cst_1 : f16) outs(%41 : memref<1x32x1xf16>)
          linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, 0)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%49 : memref<1x32x512xf16>) outs(%41 : memref<1x32x1xf16>) {
          ^bb0(%in: f16, %out: f16):
            %81 = arith.maximumf %in, %out : f16
            linalg.yield %81 : f16
          }
          linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%37, %41 : memref<1x32x1xf16>, memref<1x32x1xf16>) outs(%41 : memref<1x32x1xf16>) {
          ^bb0(%in: f16, %in_5: f16, %out: f16):
            %81 = arith.mulf %in_5, %cst_2 : f16
            %82 = arith.cmpf ogt, %in, %81 : f16
            %83 = arith.select %82, %in, %81 : f16
            linalg.yield %83 : f16
          }
          %59 = loom.broadcast ins(%41 : memref<1x32x1xf16>) outs(%53 : memref<1x32x32xf16>) dim(2) -> memref<1x32x512xf16, strided<[?, ?, ?], offset: ?>>
          linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%49, %59 : memref<1x32x512xf16>, memref<1x32x512xf16, strided<[?, ?, ?], offset: ?>>) outs(%49 : memref<1x32x512xf16>) {
          ^bb0(%in: f16, %in_5: f16, %out: f16):
            %81 = arith.mulf %in, %cst_2 : f16
            %82 = arith.subf %81, %in_5 : f16
            %83 = math.exp %82 : f16
            linalg.yield %83 : f16
          }
          loom.semaphore_give %53 : memref<1x32x32xf16>
          linalg.fill ins(%cst : f16) outs(%43 : memref<1x32x1xf16>)
          linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, 0)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%49 : memref<1x32x512xf16>) outs(%43 : memref<1x32x1xf16>) {
          ^bb0(%in: f16, %out: f16):
            %81 = arith.addf %in, %out : f16
            linalg.yield %81 : f16
          }
          linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%37, %41 : memref<1x32x1xf16>, memref<1x32x1xf16>) outs(%45 : memref<1x32x1xf16>) {
          ^bb0(%in: f16, %in_5: f16, %out: f16):
            %81 = arith.subf %in, %in_5 : f16
            %82 = math.exp %81 : f16
            linalg.yield %82 : f16
          }
          linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%35, %45, %43 : memref<1x32x1xf16>, memref<1x32x1xf16>, memref<1x32x1xf16>) outs(%35 : memref<1x32x1xf16>) {
          ^bb0(%in: f16, %in_5: f16, %in_6: f16, %out: f16):
            %81 = arith.mulf %in, %in_5 : f16
            %82 = arith.addf %81, %in_6 : f16
            linalg.yield %82 : f16
          }
          loom.semaphore_give %43 : memref<1x32x1xf16>
          %60 = loom.broadcast ins(%45 : memref<1x32x1xf16>) outs(%52 : memref<1x32x32xf16>) dim(2) -> memref<1x32x128xf16, strided<[?, ?, ?], offset: ?>>
          loom.semaphore_give %45 : memref<1x32x1xf16>
          %61 = arith.muli %23, %c65536 : index
          %62 = arith.addi %56, %61 : index
          %reinterpret_cast_4 = memref.reinterpret_cast %arg1 to offset: [%62], sizes: [1, 512, 128], strides: [1048576, 128, 1] : memref<16x8192x128xf16> to memref<1x512x128xf16, strided<[1048576, 128, 1], offset: ?>>
          loom.copy %reinterpret_cast_4, %55 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] region : (UL : [%58, %arg6], LR : [%58, %arg6]) : memref<1x512x128xf16, strided<[1048576, 128, 1], offset: ?>> to memref<1x512x128xf16>
          linalg.fill ins(%cst : f16) outs(%39 : memref<1x32x128xf16>)
          linalg.batch_matmul ins(%49, %55 : memref<1x32x512xf16>, memref<1x512x128xf16>) outs(%39 : memref<1x32x128xf16>)
          loom.semaphore_give %55 : memref<1x512x128xf16>
          loom.semaphore_give %49 : memref<1x32x512xf16>
          linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%39, %31, %60 : memref<1x32x128xf16>, memref<1x32x128xf16>, memref<1x32x128xf16, strided<[?, ?, ?], offset: ?>>) outs(%31 : memref<1x32x128xf16>) {
          ^bb0(%in: f16, %in_5: f16, %in_6: f16, %out: f16):
            %81 = arith.mulf %in_5, %in_6 : f16
            %82 = arith.addf %in, %81 : f16
            linalg.yield %82 : f16
          }
          loom.semaphore_give %52 : memref<1x32x32xf16>
          loom.semaphore_give %39 : memref<1x32x128xf16>
          linalg.copy ins(%41 : memref<1x32x1xf16>) outs(%37 : memref<1x32x1xf16>)
          loom.semaphore_give %41 : memref<1x32x1xf16>
          %63 = loom.alloc [1, 32, 1] on @L1 : memref<1x32x1xf16>
          %64 = loom.semaphore_take %63 : memref<1x32x1xf16> -> memref<1x32x1xf16>
          linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%35, %37 : memref<1x32x1xf16>, memref<1x32x1xf16>) outs(%64 : memref<1x32x1xf16>) {
          ^bb0(%in: f16, %in_5: f16, %out: f16):
            %81 = math.log %in : f16
            %82 = arith.addf %81, %in_5 : f16
            linalg.yield %82 : f16
          }
          loom.semaphore_give %37 : memref<1x32x1xf16>
          %65 = loom.broadcast ins(%35 : memref<1x32x1xf16>) outs(%51 : memref<1x32x32xf16>) dim(2) -> memref<1x32x128xf16, strided<[?, ?, ?], offset: ?>>
          loom.semaphore_give %35 : memref<1x32x1xf16>
          %66 = loom.alloc [1, 32, 128] on @L1 : memref<1x32x128xf16>
          %67 = loom.semaphore_take %66 : memref<1x32x128xf16> -> memref<1x32x128xf16>
          linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%31, %65 : memref<1x32x128xf16>, memref<1x32x128xf16, strided<[?, ?, ?], offset: ?>>) outs(%67 : memref<1x32x128xf16>) {
          ^bb0(%in: f16, %in_5: f16, %out: f16):
            %81 = arith.divf %in, %in_5 : f16
            linalg.yield %81 : f16
          }
          loom.semaphore_give %51 : memref<1x32x32xf16>
          loom.semaphore_give %31 : memref<1x32x128xf16>
          %68 = loom.alloc [16, 1, 32, 1] on @L1 : memref<16x1x32x1xf16>
          %69 = loom.semaphore_take %68 : memref<16x1x32x1xf16> -> memref<16x1x32x1xf16>
          loom.gather ins(%64 : memref<1x32x1xf16>) outs(%69 : memref<16x1x32x1xf16>) across(%arg5 : index) region : (UL : [%27, %arg6], LR : [%28, %arg6])
          loom.semaphore_give %64 : memref<1x32x1xf16>
          %70 = loom.alloc [16, 1, 32, 128] on @L1 : memref<16x1x32x128xf16>
          %71 = loom.semaphore_take %70 : memref<16x1x32x128xf16> -> memref<16x1x32x128xf16>
          loom.gather ins(%67 : memref<1x32x128xf16>) outs(%71 : memref<16x1x32x128xf16>) across(%arg5 : index) region : (UL : [%27, %arg6], LR : [%28, %arg6])
          loom.semaphore_give %67 : memref<1x32x128xf16>
          %72 = loom.alloc [1, 32, 128] on @L1 : memref<1x32x128xf16>
          %73 = loom.semaphore_take %72 : memref<1x32x128xf16> -> memref<1x32x128xf16>
          %74 = loom.alloc [16, 1, 32, 1] on @L1 : memref<16x1x32x1xf16>
          %75 = loom.semaphore_take %74 : memref<16x1x32x1xf16> -> memref<16x1x32x1xf16>
          %76 = loom.alloc [16, 1, 32, 128] on @L1 : memref<16x1x32x128xf16>
          %77 = loom.semaphore_take %76 : memref<16x1x32x128xf16> -> memref<16x1x32x128xf16>
          %78 = loom.alloc [16, 1, 32, 32] on @L1 : memref<16x1x32x32xf16>
          %79 = loom.semaphore_take %78 : memref<16x1x32x32xf16> -> memref<16x1x32x32xf16>
          %80 = arith.cmpi eq, %arg5, %c0 : index
          scf.if %80 {
            linalg.fill ins(%cst_1 : f16) outs(%34 : memref<1x32x1xf16>)
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1, d2, d3)>], iterator_types = ["reduction", "parallel", "parallel", "parallel"]} ins(%69 : memref<16x1x32x1xf16>) outs(%34 : memref<1x32x1xf16>) {
            ^bb0(%in: f16, %out: f16):
              %82 = arith.maximumf %in, %out : f16
              linalg.yield %82 : f16
            }
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%69, %34 : memref<16x1x32x1xf16>, memref<1x32x1xf16>) outs(%75 : memref<16x1x32x1xf16>) {
            ^bb0(%in: f16, %in_6: f16, %out: f16):
              %82 = arith.subf %in, %in_6 : f16
              %83 = math.exp %82 : f16
              linalg.yield %83 : f16
            }
            loom.semaphore_give %69 : memref<16x1x32x1xf16>
            loom.semaphore_give %34 : memref<1x32x1xf16>
            linalg.fill ins(%cst : f16) outs(%33 : memref<1x32x1xf16>)
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1, d2, d3)>], iterator_types = ["reduction", "parallel", "parallel", "parallel"]} ins(%75 : memref<16x1x32x1xf16>) outs(%33 : memref<1x32x1xf16>) {
            ^bb0(%in: f16, %out: f16):
              %82 = arith.addf %in, %out : f16
              linalg.yield %82 : f16
            }
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%75, %33 : memref<16x1x32x1xf16>, memref<1x32x1xf16>) outs(%75 : memref<16x1x32x1xf16>) {
            ^bb0(%in: f16, %in_6: f16, %out: f16):
              %82 = arith.divf %in, %in_6 : f16
              linalg.yield %82 : f16
            }
            loom.semaphore_give %33 : memref<1x32x1xf16>
            %81 = loom.broadcast ins(%75 : memref<16x1x32x1xf16>) outs(%79 : memref<16x1x32x32xf16>) dim(3) -> memref<16x1x32x128xf16, strided<[?, ?, ?, ?], offset: ?>>
            loom.semaphore_give %75 : memref<16x1x32x1xf16>
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%71, %81 : memref<16x1x32x128xf16>, memref<16x1x32x128xf16, strided<[?, ?, ?, ?], offset: ?>>) outs(%77 : memref<16x1x32x128xf16>) {
            ^bb0(%in: f16, %in_6: f16, %out: f16):
              %82 = arith.mulf %in, %in_6 : f16
              linalg.yield %82 : f16
            }
            loom.semaphore_give %79 : memref<16x1x32x32xf16>
            loom.semaphore_give %71 : memref<16x1x32x128xf16>
            linalg.fill ins(%cst : f16) outs(%73 : memref<1x32x128xf16>)
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1, d2, d3)>], iterator_types = ["reduction", "parallel", "parallel", "parallel"]} ins(%77 : memref<16x1x32x128xf16>) outs(%73 : memref<1x32x128xf16>) {
            ^bb0(%in: f16, %out: f16):
              %82 = arith.addf %in, %out : f16
              linalg.yield %82 : f16
            }
            loom.semaphore_give %77 : memref<16x1x32x128xf16>
            %reinterpret_cast_5 = memref.reinterpret_cast %arg2 to offset: [%26], sizes: [1, 32, 128], strides: [4096, 128, 1] : memref<16x32x128xf16> to memref<1x32x128xf16, strided<[4096, 128, 1], offset: ?>>
            loom.copy %73, %reinterpret_cast_5 src_mem_space @mem_L1 dst_mem_space @mem_DRAM, broadcast : [1, 1] region : (UL : [%58, %arg6], LR : [%58, %arg6]) : memref<1x32x128xf16> to memref<1x32x128xf16, strided<[4096, 128, 1], offset: ?>>
            loom.semaphore_give %73 : memref<1x32x128xf16>
          }
        } {loom.block_sym = @tile_b, loom.iter_type = #loom.iter_type<temporal>}
        scf.reduce 
      } {loom.block_syms = [@tile_b, @tile_s, @tile_s], loom.iter_types = [#loom.iter_type<spatial>, #loom.iter_type<spatial>, #loom.iter_type<spatial>], loom.logical_levels = [1, 0, 0], loom.physical_dims = [@dim_x, @dim_x, @dim_y]}
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
          linalg.fill ins(%cst : f16) outs(%31 : memref<1x32x128xf16>)
          %32 = loom.alloc [1, 32, 1] on @L1 : memref<1x32x1xf16>
          %33 = loom.semaphore_take %32 : memref<1x32x1xf16> -> memref<1x32x1xf16>
          %34 = loom.semaphore_take %32 : memref<1x32x1xf16> -> memref<1x32x1xf16>
          %35 = loom.semaphore_take %32 : memref<1x32x1xf16> -> memref<1x32x1xf16>
          linalg.fill ins(%cst_0 : f16) outs(%35 : memref<1x32x1xf16>)
          %36 = loom.alloc [1, 32, 1] on @L1 : memref<1x32x1xf16>
          %37 = loom.semaphore_take %36 : memref<1x32x1xf16> -> memref<1x32x1xf16>
          linalg.fill ins(%cst_1 : f16) outs(%37 : memref<1x32x1xf16>)
          %38 = loom.alloc [1, 32, 128] on @L1 : memref<1x32x128xf16>
          %39 = loom.semaphore_take %38 : memref<1x32x128xf16> -> memref<1x32x128xf16>
          %40 = loom.alloc [1, 32, 1] on @L1 : memref<1x32x1xf16>
          %41 = loom.semaphore_take %40 : memref<1x32x1xf16> -> memref<1x32x1xf16>
          %42 = loom.alloc [1, 32, 1] on @L1 : memref<1x32x1xf16>
          %43 = loom.semaphore_take %42 : memref<1x32x1xf16> -> memref<1x32x1xf16>
          %44 = loom.alloc [1, 32, 1] on @L1 : memref<1x32x1xf16>
          %45 = loom.semaphore_take %44 : memref<1x32x1xf16> -> memref<1x32x1xf16>
          %46 = loom.alloc [1, 128, 512] on @L1 : memref<1x128x512xf16>
          %47 = loom.semaphore_take %46 : memref<1x128x512xf16> -> memref<1x128x512xf16>
          %48 = loom.alloc [1, 32, 512] on @L1 : memref<1x32x512xf16>
          %49 = loom.semaphore_take %48 : memref<1x32x512xf16> -> memref<1x32x512xf16>
          %50 = loom.alloc [1, 32, 32] on @L1 : memref<1x32x32xf16>
          %51 = loom.semaphore_take %50 : memref<1x32x32xf16> -> memref<1x32x32xf16>
          %52 = loom.semaphore_take %50 : memref<1x32x32xf16> -> memref<1x32x32xf16>
          %53 = loom.semaphore_take %50 : memref<1x32x32xf16> -> memref<1x32x32xf16>
          %54 = loom.alloc [1, 512, 128] on @L1 : memref<1x512x128xf16>
          %55 = loom.semaphore_take %54 : memref<1x512x128xf16> -> memref<1x512x128xf16>
          %56 = arith.muli %21, %c1048576 overflow<nsw> : index
          %57 = arith.addi %56, %29 : index
          %reinterpret_cast_3 = memref.reinterpret_cast %arg0 to offset: [%57], sizes: [1, 128, 512], strides: [1048576, 8192, 1] : memref<16x128x8192xf16> to memref<1x128x512xf16, strided<[1048576, 8192, 1], offset: ?>>
          %58 = arith.addi %arg5, %27 : index
          loom.copy %reinterpret_cast_3, %47 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] region : (UL : [%58, %arg6], LR : [%58, %arg6]) : memref<1x128x512xf16, strided<[1048576, 8192, 1], offset: ?>> to memref<1x128x512xf16>
          linalg.fill ins(%cst : f16) outs(%49 : memref<1x32x512xf16>)
          linalg.batch_matmul ins(%25, %47 : memref<1x32x128xf16>, memref<1x128x512xf16>) outs(%49 : memref<1x32x512xf16>)
          loom.semaphore_give %47 : memref<1x128x512xf16>
          loom.semaphore_give %25 : memref<1x32x128xf16>
          linalg.fill ins(%cst_1 : f16) outs(%41 : memref<1x32x1xf16>)
          linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, 0)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%49 : memref<1x32x512xf16>) outs(%41 : memref<1x32x1xf16>) {
          ^bb0(%in: f16, %out: f16):
            %81 = arith.maximumf %in, %out : f16
            linalg.yield %81 : f16
          }
          linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%37, %41 : memref<1x32x1xf16>, memref<1x32x1xf16>) outs(%41 : memref<1x32x1xf16>) {
          ^bb0(%in: f16, %in_5: f16, %out: f16):
            %81 = arith.mulf %in_5, %cst_2 : f16
            %82 = arith.cmpf ogt, %in, %81 : f16
            %83 = arith.select %82, %in, %81 : f16
            linalg.yield %83 : f16
          }
          %59 = loom.broadcast ins(%41 : memref<1x32x1xf16>) outs(%53 : memref<1x32x32xf16>) dim(2) -> memref<1x32x512xf16, strided<[?, ?, ?], offset: ?>>
          linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%49, %59 : memref<1x32x512xf16>, memref<1x32x512xf16, strided<[?, ?, ?], offset: ?>>) outs(%49 : memref<1x32x512xf16>) {
          ^bb0(%in: f16, %in_5: f16, %out: f16):
            %81 = arith.mulf %in, %cst_2 : f16
            %82 = arith.subf %81, %in_5 : f16
            %83 = math.exp %82 : f16
            linalg.yield %83 : f16
          }
          loom.semaphore_give %53 : memref<1x32x32xf16>
          linalg.fill ins(%cst : f16) outs(%43 : memref<1x32x1xf16>)
          linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, 0)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%49 : memref<1x32x512xf16>) outs(%43 : memref<1x32x1xf16>) {
          ^bb0(%in: f16, %out: f16):
            %81 = arith.addf %in, %out : f16
            linalg.yield %81 : f16
          }
          linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%37, %41 : memref<1x32x1xf16>, memref<1x32x1xf16>) outs(%45 : memref<1x32x1xf16>) {
          ^bb0(%in: f16, %in_5: f16, %out: f16):
            %81 = arith.subf %in, %in_5 : f16
            %82 = math.exp %81 : f16
            linalg.yield %82 : f16
          }
          linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%35, %45, %43 : memref<1x32x1xf16>, memref<1x32x1xf16>, memref<1x32x1xf16>) outs(%35 : memref<1x32x1xf16>) {
          ^bb0(%in: f16, %in_5: f16, %in_6: f16, %out: f16):
            %81 = arith.mulf %in, %in_5 : f16
            %82 = arith.addf %81, %in_6 : f16
            linalg.yield %82 : f16
          }
          loom.semaphore_give %43 : memref<1x32x1xf16>
          %60 = loom.broadcast ins(%45 : memref<1x32x1xf16>) outs(%52 : memref<1x32x32xf16>) dim(2) -> memref<1x32x128xf16, strided<[?, ?, ?], offset: ?>>
          loom.semaphore_give %45 : memref<1x32x1xf16>
          %61 = arith.muli %23, %c65536 : index
          %62 = arith.addi %56, %61 : index
          %reinterpret_cast_4 = memref.reinterpret_cast %arg1 to offset: [%62], sizes: [1, 512, 128], strides: [1048576, 128, 1] : memref<16x8192x128xf16> to memref<1x512x128xf16, strided<[1048576, 128, 1], offset: ?>>
          loom.copy %reinterpret_cast_4, %55 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] region : (UL : [%58, %arg6], LR : [%58, %arg6]) : memref<1x512x128xf16, strided<[1048576, 128, 1], offset: ?>> to memref<1x512x128xf16>
          linalg.fill ins(%cst : f16) outs(%39 : memref<1x32x128xf16>)
          linalg.batch_matmul ins(%49, %55 : memref<1x32x512xf16>, memref<1x512x128xf16>) outs(%39 : memref<1x32x128xf16>)
          loom.semaphore_give %55 : memref<1x512x128xf16>
          loom.semaphore_give %49 : memref<1x32x512xf16>
          linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%39, %31, %60 : memref<1x32x128xf16>, memref<1x32x128xf16>, memref<1x32x128xf16, strided<[?, ?, ?], offset: ?>>) outs(%31 : memref<1x32x128xf16>) {
          ^bb0(%in: f16, %in_5: f16, %in_6: f16, %out: f16):
            %81 = arith.mulf %in_5, %in_6 : f16
            %82 = arith.addf %in, %81 : f16
            linalg.yield %82 : f16
          }
          loom.semaphore_give %52 : memref<1x32x32xf16>
          loom.semaphore_give %39 : memref<1x32x128xf16>
          linalg.copy ins(%41 : memref<1x32x1xf16>) outs(%37 : memref<1x32x1xf16>)
          loom.semaphore_give %41 : memref<1x32x1xf16>
          %63 = loom.alloc [1, 32, 1] on @L1 : memref<1x32x1xf16>
          %64 = loom.semaphore_take %63 : memref<1x32x1xf16> -> memref<1x32x1xf16>
          linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%35, %37 : memref<1x32x1xf16>, memref<1x32x1xf16>) outs(%64 : memref<1x32x1xf16>) {
          ^bb0(%in: f16, %in_5: f16, %out: f16):
            %81 = math.log %in : f16
            %82 = arith.addf %81, %in_5 : f16
            linalg.yield %82 : f16
          }
          loom.semaphore_give %37 : memref<1x32x1xf16>
          %65 = loom.broadcast ins(%35 : memref<1x32x1xf16>) outs(%51 : memref<1x32x32xf16>) dim(2) -> memref<1x32x128xf16, strided<[?, ?, ?], offset: ?>>
          loom.semaphore_give %35 : memref<1x32x1xf16>
          %66 = loom.alloc [1, 32, 128] on @L1 : memref<1x32x128xf16>
          %67 = loom.semaphore_take %66 : memref<1x32x128xf16> -> memref<1x32x128xf16>
          linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%31, %65 : memref<1x32x128xf16>, memref<1x32x128xf16, strided<[?, ?, ?], offset: ?>>) outs(%67 : memref<1x32x128xf16>) {
          ^bb0(%in: f16, %in_5: f16, %out: f16):
            %81 = arith.divf %in, %in_5 : f16
            linalg.yield %81 : f16
          }
          loom.semaphore_give %51 : memref<1x32x32xf16>
          loom.semaphore_give %31 : memref<1x32x128xf16>
          %68 = loom.alloc [16, 1, 32, 1] on @L1 : memref<16x1x32x1xf16>
          %69 = loom.semaphore_take %68 : memref<16x1x32x1xf16> -> memref<16x1x32x1xf16>
          loom.gather ins(%64 : memref<1x32x1xf16>) outs(%69 : memref<16x1x32x1xf16>) across(%arg5 : index) region : (UL : [%27, %arg6], LR : [%28, %arg6])
          loom.semaphore_give %64 : memref<1x32x1xf16>
          %70 = loom.alloc [16, 1, 32, 128] on @L1 : memref<16x1x32x128xf16>
          %71 = loom.semaphore_take %70 : memref<16x1x32x128xf16> -> memref<16x1x32x128xf16>
          loom.gather ins(%67 : memref<1x32x128xf16>) outs(%71 : memref<16x1x32x128xf16>) across(%arg5 : index) region : (UL : [%27, %arg6], LR : [%28, %arg6])
          loom.semaphore_give %67 : memref<1x32x128xf16>
          %72 = loom.alloc [1, 32, 128] on @L1 : memref<1x32x128xf16>
          %73 = loom.semaphore_take %72 : memref<1x32x128xf16> -> memref<1x32x128xf16>
          %74 = loom.alloc [16, 1, 32, 1] on @L1 : memref<16x1x32x1xf16>
          %75 = loom.semaphore_take %74 : memref<16x1x32x1xf16> -> memref<16x1x32x1xf16>
          %76 = loom.alloc [16, 1, 32, 128] on @L1 : memref<16x1x32x128xf16>
          %77 = loom.semaphore_take %76 : memref<16x1x32x128xf16> -> memref<16x1x32x128xf16>
          %78 = loom.alloc [16, 1, 32, 32] on @L1 : memref<16x1x32x32xf16>
          %79 = loom.semaphore_take %78 : memref<16x1x32x32xf16> -> memref<16x1x32x32xf16>
          %80 = arith.cmpi eq, %arg5, %c0 : index
          scf.if %80 {
            linalg.fill ins(%cst_1 : f16) outs(%34 : memref<1x32x1xf16>)
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1, d2, d3)>], iterator_types = ["reduction", "parallel", "parallel", "parallel"]} ins(%69 : memref<16x1x32x1xf16>) outs(%34 : memref<1x32x1xf16>) {
            ^bb0(%in: f16, %out: f16):
              %82 = arith.maximumf %in, %out : f16
              linalg.yield %82 : f16
            }
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%69, %34 : memref<16x1x32x1xf16>, memref<1x32x1xf16>) outs(%75 : memref<16x1x32x1xf16>) {
            ^bb0(%in: f16, %in_6: f16, %out: f16):
              %82 = arith.subf %in, %in_6 : f16
              %83 = math.exp %82 : f16
              linalg.yield %83 : f16
            }
            loom.semaphore_give %69 : memref<16x1x32x1xf16>
            loom.semaphore_give %34 : memref<1x32x1xf16>
            linalg.fill ins(%cst : f16) outs(%33 : memref<1x32x1xf16>)
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1, d2, d3)>], iterator_types = ["reduction", "parallel", "parallel", "parallel"]} ins(%75 : memref<16x1x32x1xf16>) outs(%33 : memref<1x32x1xf16>) {
            ^bb0(%in: f16, %out: f16):
              %82 = arith.addf %in, %out : f16
              linalg.yield %82 : f16
            }
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%75, %33 : memref<16x1x32x1xf16>, memref<1x32x1xf16>) outs(%75 : memref<16x1x32x1xf16>) {
            ^bb0(%in: f16, %in_6: f16, %out: f16):
              %82 = arith.divf %in, %in_6 : f16
              linalg.yield %82 : f16
            }
            loom.semaphore_give %33 : memref<1x32x1xf16>
            %81 = loom.broadcast ins(%75 : memref<16x1x32x1xf16>) outs(%79 : memref<16x1x32x32xf16>) dim(3) -> memref<16x1x32x128xf16, strided<[?, ?, ?, ?], offset: ?>>
            loom.semaphore_give %75 : memref<16x1x32x1xf16>
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%71, %81 : memref<16x1x32x128xf16>, memref<16x1x32x128xf16, strided<[?, ?, ?, ?], offset: ?>>) outs(%77 : memref<16x1x32x128xf16>) {
            ^bb0(%in: f16, %in_6: f16, %out: f16):
              %82 = arith.mulf %in, %in_6 : f16
              linalg.yield %82 : f16
            }
            loom.semaphore_give %79 : memref<16x1x32x32xf16>
            loom.semaphore_give %71 : memref<16x1x32x128xf16>
            linalg.fill ins(%cst : f16) outs(%73 : memref<1x32x128xf16>)
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1, d2, d3)>], iterator_types = ["reduction", "parallel", "parallel", "parallel"]} ins(%77 : memref<16x1x32x128xf16>) outs(%73 : memref<1x32x128xf16>) {
            ^bb0(%in: f16, %out: f16):
              %82 = arith.addf %in, %out : f16
              linalg.yield %82 : f16
            }
            loom.semaphore_give %77 : memref<16x1x32x128xf16>
            %reinterpret_cast_5 = memref.reinterpret_cast %arg2 to offset: [%26], sizes: [1, 32, 128], strides: [4096, 128, 1] : memref<16x32x128xf16> to memref<1x32x128xf16, strided<[4096, 128, 1], offset: ?>>
            loom.copy %73, %reinterpret_cast_5 src_mem_space @mem_L1 dst_mem_space @mem_DRAM, broadcast : [1, 1] region : (UL : [%58, %arg6], LR : [%58, %arg6]) : memref<1x32x128xf16> to memref<1x32x128xf16, strided<[4096, 128, 1], offset: ?>>
            loom.semaphore_give %73 : memref<1x32x128xf16>
          }
        } {loom.block_sym = @tile_b, loom.iter_type = #loom.iter_type<temporal>}
        scf.reduce 
      } {loom.block_syms = [@tile_b, @tile_s, @tile_s], loom.iter_types = [#loom.iter_type<spatial>, #loom.iter_type<spatial>, #loom.iter_type<spatial>], loom.logical_levels = [1, 0, 0], loom.physical_dims = [@dim_x, @dim_x, @dim_y]}
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
          linalg.fill ins(%cst : f16) outs(%27 : memref<1x32x128xf16>)
          %28 = loom.alloc [1, 32, 1] on @L1 : memref<1x32x1xf16>
          %29 = loom.semaphore_take %28 : memref<1x32x1xf16> -> memref<1x32x1xf16>
          %30 = loom.semaphore_take %28 : memref<1x32x1xf16> -> memref<1x32x1xf16>
          %31 = loom.semaphore_take %28 : memref<1x32x1xf16> -> memref<1x32x1xf16>
          linalg.fill ins(%cst_0 : f16) outs(%31 : memref<1x32x1xf16>)
          %32 = loom.alloc [1, 32, 1] on @L1 : memref<1x32x1xf16>
          %33 = loom.semaphore_take %32 : memref<1x32x1xf16> -> memref<1x32x1xf16>
          linalg.fill ins(%cst_1 : f16) outs(%33 : memref<1x32x1xf16>)
          %34 = loom.alloc [1, 32, 128] on @L1 : memref<1x32x128xf16>
          %35 = loom.semaphore_take %34 : memref<1x32x128xf16> -> memref<1x32x128xf16>
          %36 = loom.alloc [1, 32, 1] on @L1 : memref<1x32x1xf16>
          %37 = loom.semaphore_take %36 : memref<1x32x1xf16> -> memref<1x32x1xf16>
          %38 = loom.alloc [1, 32, 1] on @L1 : memref<1x32x1xf16>
          %39 = loom.semaphore_take %38 : memref<1x32x1xf16> -> memref<1x32x1xf16>
          %40 = loom.alloc [1, 32, 1] on @L1 : memref<1x32x1xf16>
          %41 = loom.semaphore_take %40 : memref<1x32x1xf16> -> memref<1x32x1xf16>
          %42 = loom.alloc [1, 128, 512] on @L1 : memref<1x128x512xf16>
          %43 = loom.semaphore_take %42 : memref<1x128x512xf16> -> memref<1x128x512xf16>
          %44 = loom.alloc [1, 32, 512] on @L1 : memref<1x32x512xf16>
          %45 = loom.semaphore_take %44 : memref<1x32x512xf16> -> memref<1x32x512xf16>
          %46 = loom.alloc [1, 32, 32] on @L1 : memref<1x32x32xf16>
          %47 = loom.semaphore_take %46 : memref<1x32x32xf16> -> memref<1x32x32xf16>
          %48 = loom.semaphore_take %46 : memref<1x32x32xf16> -> memref<1x32x32xf16>
          %49 = loom.semaphore_take %46 : memref<1x32x32xf16> -> memref<1x32x32xf16>
          %50 = loom.alloc [1, 512, 128] on @L1 : memref<1x512x128xf16>
          %51 = loom.semaphore_take %50 : memref<1x512x128xf16> -> memref<1x512x128xf16>
          %52 = arith.muli %arg6, %c1048576 overflow<nsw> : index
          %53 = arith.addi %52, %25 : index
          %reinterpret_cast_3 = memref.reinterpret_cast %arg0 to offset: [%53], sizes: [1, 128, 512], strides: [1048576, 8192, 1] : memref<16x128x8192xf16> to memref<1x128x512xf16, strided<[1048576, 8192, 1], offset: ?>>
          loom.copy %reinterpret_cast_3, %43 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] region : (UL : [%arg4, %arg5], LR : [%arg4, %arg5]) : memref<1x128x512xf16, strided<[1048576, 8192, 1], offset: ?>> to memref<1x128x512xf16>
          linalg.fill ins(%cst : f16) outs(%45 : memref<1x32x512xf16>)
          linalg.batch_matmul ins(%23, %43 : memref<1x32x128xf16>, memref<1x128x512xf16>) outs(%45 : memref<1x32x512xf16>)
          loom.semaphore_give %43 : memref<1x128x512xf16>
          loom.semaphore_give %23 : memref<1x32x128xf16>
          linalg.fill ins(%cst_1 : f16) outs(%37 : memref<1x32x1xf16>)
          linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, 0)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%45 : memref<1x32x512xf16>) outs(%37 : memref<1x32x1xf16>) {
          ^bb0(%in: f16, %out: f16):
            %76 = arith.maximumf %in, %out : f16
            linalg.yield %76 : f16
          }
          linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%33, %37 : memref<1x32x1xf16>, memref<1x32x1xf16>) outs(%37 : memref<1x32x1xf16>) {
          ^bb0(%in: f16, %in_5: f16, %out: f16):
            %76 = arith.mulf %in_5, %cst_2 : f16
            %77 = arith.cmpf ogt, %in, %76 : f16
            %78 = arith.select %77, %in, %76 : f16
            linalg.yield %78 : f16
          }
          %54 = loom.broadcast ins(%37 : memref<1x32x1xf16>) outs(%49 : memref<1x32x32xf16>) dim(2) -> memref<1x32x512xf16, strided<[?, ?, ?], offset: ?>>
          linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%45, %54 : memref<1x32x512xf16>, memref<1x32x512xf16, strided<[?, ?, ?], offset: ?>>) outs(%45 : memref<1x32x512xf16>) {
          ^bb0(%in: f16, %in_5: f16, %out: f16):
            %76 = arith.mulf %in, %cst_2 : f16
            %77 = arith.subf %76, %in_5 : f16
            %78 = math.exp %77 : f16
            linalg.yield %78 : f16
          }
          loom.semaphore_give %49 : memref<1x32x32xf16>
          linalg.fill ins(%cst : f16) outs(%39 : memref<1x32x1xf16>)
          linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, 0)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%45 : memref<1x32x512xf16>) outs(%39 : memref<1x32x1xf16>) {
          ^bb0(%in: f16, %out: f16):
            %76 = arith.addf %in, %out : f16
            linalg.yield %76 : f16
          }
          linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%33, %37 : memref<1x32x1xf16>, memref<1x32x1xf16>) outs(%41 : memref<1x32x1xf16>) {
          ^bb0(%in: f16, %in_5: f16, %out: f16):
            %76 = arith.subf %in, %in_5 : f16
            %77 = math.exp %76 : f16
            linalg.yield %77 : f16
          }
          linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%31, %41, %39 : memref<1x32x1xf16>, memref<1x32x1xf16>, memref<1x32x1xf16>) outs(%31 : memref<1x32x1xf16>) {
          ^bb0(%in: f16, %in_5: f16, %in_6: f16, %out: f16):
            %76 = arith.mulf %in, %in_5 : f16
            %77 = arith.addf %76, %in_6 : f16
            linalg.yield %77 : f16
          }
          loom.semaphore_give %39 : memref<1x32x1xf16>
          %55 = loom.broadcast ins(%41 : memref<1x32x1xf16>) outs(%48 : memref<1x32x32xf16>) dim(2) -> memref<1x32x128xf16, strided<[?, ?, ?], offset: ?>>
          loom.semaphore_give %41 : memref<1x32x1xf16>
          %56 = arith.muli %21, %c65536 : index
          %57 = arith.addi %52, %56 : index
          %reinterpret_cast_4 = memref.reinterpret_cast %arg1 to offset: [%57], sizes: [1, 512, 128], strides: [1048576, 128, 1] : memref<16x8192x128xf16> to memref<1x512x128xf16, strided<[1048576, 128, 1], offset: ?>>
          loom.copy %reinterpret_cast_4, %51 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] region : (UL : [%arg4, %arg5], LR : [%arg4, %arg5]) : memref<1x512x128xf16, strided<[1048576, 128, 1], offset: ?>> to memref<1x512x128xf16>
          linalg.fill ins(%cst : f16) outs(%35 : memref<1x32x128xf16>)
          linalg.batch_matmul ins(%45, %51 : memref<1x32x512xf16>, memref<1x512x128xf16>) outs(%35 : memref<1x32x128xf16>)
          loom.semaphore_give %51 : memref<1x512x128xf16>
          loom.semaphore_give %45 : memref<1x32x512xf16>
          linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%35, %27, %55 : memref<1x32x128xf16>, memref<1x32x128xf16>, memref<1x32x128xf16, strided<[?, ?, ?], offset: ?>>) outs(%27 : memref<1x32x128xf16>) {
          ^bb0(%in: f16, %in_5: f16, %in_6: f16, %out: f16):
            %76 = arith.mulf %in_5, %in_6 : f16
            %77 = arith.addf %in, %76 : f16
            linalg.yield %77 : f16
          }
          loom.semaphore_give %48 : memref<1x32x32xf16>
          loom.semaphore_give %35 : memref<1x32x128xf16>
          linalg.copy ins(%37 : memref<1x32x1xf16>) outs(%33 : memref<1x32x1xf16>)
          loom.semaphore_give %37 : memref<1x32x1xf16>
          %58 = loom.alloc [1, 32, 1] on @L1 : memref<1x32x1xf16>
          %59 = loom.semaphore_take %58 : memref<1x32x1xf16> -> memref<1x32x1xf16>
          linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%31, %33 : memref<1x32x1xf16>, memref<1x32x1xf16>) outs(%59 : memref<1x32x1xf16>) {
          ^bb0(%in: f16, %in_5: f16, %out: f16):
            %76 = math.log %in : f16
            %77 = arith.addf %76, %in_5 : f16
            linalg.yield %77 : f16
          }
          loom.semaphore_give %33 : memref<1x32x1xf16>
          %60 = loom.broadcast ins(%31 : memref<1x32x1xf16>) outs(%47 : memref<1x32x32xf16>) dim(2) -> memref<1x32x128xf16, strided<[?, ?, ?], offset: ?>>
          loom.semaphore_give %31 : memref<1x32x1xf16>
          %61 = loom.alloc [1, 32, 128] on @L1 : memref<1x32x128xf16>
          %62 = loom.semaphore_take %61 : memref<1x32x128xf16> -> memref<1x32x128xf16>
          linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%27, %60 : memref<1x32x128xf16>, memref<1x32x128xf16, strided<[?, ?, ?], offset: ?>>) outs(%62 : memref<1x32x128xf16>) {
          ^bb0(%in: f16, %in_5: f16, %out: f16):
            %76 = arith.divf %in, %in_5 : f16
            linalg.yield %76 : f16
          }
          loom.semaphore_give %47 : memref<1x32x32xf16>
          loom.semaphore_give %27 : memref<1x32x128xf16>
          %63 = loom.alloc [16, 1, 32, 1] on @L1 : memref<16x1x32x1xf16>
          %64 = loom.semaphore_take %63 : memref<16x1x32x1xf16> -> memref<16x1x32x1xf16>
          loom.gather ins(%59 : memref<1x32x1xf16>) outs(%64 : memref<16x1x32x1xf16>) across(%arg4 : index) region : (UL : [%c0, %arg5], LR : [%c7, %arg5])
          loom.semaphore_give %59 : memref<1x32x1xf16>
          %65 = loom.alloc [16, 1, 32, 128] on @L1 : memref<16x1x32x128xf16>
          %66 = loom.semaphore_take %65 : memref<16x1x32x128xf16> -> memref<16x1x32x128xf16>
          loom.gather ins(%62 : memref<1x32x128xf16>) outs(%66 : memref<16x1x32x128xf16>) across(%arg4 : index) region : (UL : [%c0, %arg5], LR : [%c7, %arg5])
          loom.semaphore_give %62 : memref<1x32x128xf16>
          %67 = loom.alloc [1, 32, 128] on @L1 : memref<1x32x128xf16>
          %68 = loom.semaphore_take %67 : memref<1x32x128xf16> -> memref<1x32x128xf16>
          %69 = loom.alloc [16, 1, 32, 1] on @L1 : memref<16x1x32x1xf16>
          %70 = loom.semaphore_take %69 : memref<16x1x32x1xf16> -> memref<16x1x32x1xf16>
          %71 = loom.alloc [16, 1, 32, 128] on @L1 : memref<16x1x32x128xf16>
          %72 = loom.semaphore_take %71 : memref<16x1x32x128xf16> -> memref<16x1x32x128xf16>
          %73 = loom.alloc [16, 1, 32, 32] on @L1 : memref<16x1x32x32xf16>
          %74 = loom.semaphore_take %73 : memref<16x1x32x32xf16> -> memref<16x1x32x32xf16>
          %75 = arith.cmpi eq, %arg4, %c0 : index
          scf.if %75 {
            linalg.fill ins(%cst_1 : f16) outs(%30 : memref<1x32x1xf16>)
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1, d2, d3)>], iterator_types = ["reduction", "parallel", "parallel", "parallel"]} ins(%64 : memref<16x1x32x1xf16>) outs(%30 : memref<1x32x1xf16>) {
            ^bb0(%in: f16, %out: f16):
              %77 = arith.maximumf %in, %out : f16
              linalg.yield %77 : f16
            }
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%64, %30 : memref<16x1x32x1xf16>, memref<1x32x1xf16>) outs(%70 : memref<16x1x32x1xf16>) {
            ^bb0(%in: f16, %in_6: f16, %out: f16):
              %77 = arith.subf %in, %in_6 : f16
              %78 = math.exp %77 : f16
              linalg.yield %78 : f16
            }
            loom.semaphore_give %64 : memref<16x1x32x1xf16>
            loom.semaphore_give %30 : memref<1x32x1xf16>
            linalg.fill ins(%cst : f16) outs(%29 : memref<1x32x1xf16>)
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1, d2, d3)>], iterator_types = ["reduction", "parallel", "parallel", "parallel"]} ins(%70 : memref<16x1x32x1xf16>) outs(%29 : memref<1x32x1xf16>) {
            ^bb0(%in: f16, %out: f16):
              %77 = arith.addf %in, %out : f16
              linalg.yield %77 : f16
            }
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%70, %29 : memref<16x1x32x1xf16>, memref<1x32x1xf16>) outs(%70 : memref<16x1x32x1xf16>) {
            ^bb0(%in: f16, %in_6: f16, %out: f16):
              %77 = arith.divf %in, %in_6 : f16
              linalg.yield %77 : f16
            }
            loom.semaphore_give %29 : memref<1x32x1xf16>
            %76 = loom.broadcast ins(%70 : memref<16x1x32x1xf16>) outs(%74 : memref<16x1x32x32xf16>) dim(3) -> memref<16x1x32x128xf16, strided<[?, ?, ?, ?], offset: ?>>
            loom.semaphore_give %70 : memref<16x1x32x1xf16>
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%66, %76 : memref<16x1x32x128xf16>, memref<16x1x32x128xf16, strided<[?, ?, ?, ?], offset: ?>>) outs(%72 : memref<16x1x32x128xf16>) {
            ^bb0(%in: f16, %in_6: f16, %out: f16):
              %77 = arith.mulf %in, %in_6 : f16
              linalg.yield %77 : f16
            }
            loom.semaphore_give %74 : memref<16x1x32x32xf16>
            loom.semaphore_give %66 : memref<16x1x32x128xf16>
            linalg.fill ins(%cst : f16) outs(%68 : memref<1x32x128xf16>)
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1, d2, d3)>], iterator_types = ["reduction", "parallel", "parallel", "parallel"]} ins(%72 : memref<16x1x32x128xf16>) outs(%68 : memref<1x32x128xf16>) {
            ^bb0(%in: f16, %out: f16):
              %77 = arith.addf %in, %out : f16
              linalg.yield %77 : f16
            }
            loom.semaphore_give %72 : memref<16x1x32x128xf16>
            %reinterpret_cast_5 = memref.reinterpret_cast %arg2 to offset: [%24], sizes: [1, 32, 128], strides: [4096, 128, 1] : memref<16x32x128xf16> to memref<1x32x128xf16, strided<[4096, 128, 1], offset: ?>>
            loom.copy %68, %reinterpret_cast_5 src_mem_space @mem_L1 dst_mem_space @mem_DRAM, broadcast : [1, 1] region : (UL : [%arg4, %arg5], LR : [%arg4, %arg5]) : memref<1x32x128xf16> to memref<1x32x128xf16, strided<[4096, 128, 1], offset: ?>>
            loom.semaphore_give %68 : memref<1x32x128xf16>
          }
        } {loom.block_sym = @tile_b, loom.iter_type = #loom.iter_type<temporal>}
        scf.reduce 
      } {loom.block_syms = [@tile_s, @tile_s], loom.iter_types = [#loom.iter_type<spatial>, #loom.iter_type<spatial>], loom.logical_levels = [0, 0], loom.physical_dims = [@dim_x, @dim_y]}
      return
    }
  }
}
