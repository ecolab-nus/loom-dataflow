module attributes {loom.tile_b = {is_reduction = false, upper_bound = 16 : index}, loom.tile_n = {is_reduction = false, upper_bound = 64 : index}, loom.tile_s = {is_reduction = false, upper_bound = 8192 : index}} {
  %0 = adl.memory.bank "mem_DRAM_bank", {bsize = 8192 : i64, nblk = 196608 : i64}
  %1 = adl.spatial_dim "dim_dram_channel", 8
  %2 = adl.memory.array "mem_DRAM", [%1] of %0
  %3 = adl.memory.bank "mem_bank", {bsize = 16 : i64, nblk = 5856 : i64}
  %4 = adl.spatial_dim "dim_nbank", 16
  %5 = adl.memory.array "mem_L1", [%4] of %3
  %6 = adl.resource.exclusive "res_matrix_lane"
  %7 = adl.resource.exclusive "res_vector_lane"
  %8 = adl.processor.compute @proc_matrix_lane, [(%5, %5)], with [%6]
  %9 = adl.processor.compute @proc_vector_lane, [(%5, %5)], with [%7]
  %10 = adl.arch.compose "arch_core", arch[%8, %9], mem[%5]
  %11 = adl.spatial_dim "dim_x", 8
  %12 = adl.spatial_dim "dim_y", 8
  %13 = adl.arch.scale "arch_mesh", [%11, %12] of %10
  %14 = adl.memory.array "mem_array_L1", [%11, %12] of %5
  %15 = adl.processor.dmover @proc_dram_l1_noc0, [(%2, %14)]
  %16 = adl.processor.dmover @proc_dram_l1_noc1, [(%14, %2), (%14, %14)]
  %17 = adl.arch.compose "arch_system", arch[%13, %15, %16], mem[%2]
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
            %18 = arith.muli %arg6, %c8 overflow<nsw> : index
            %19 = arith.addi %arg4, %18 : index
            %20 = arith.muli %arg7, %c8 overflow<nsw> : index
            %21 = arith.addi %arg5, %20 : index
            %22 = loom.alloc [1, 32, 128] on @L1 : memref<1x32x128xf16>
            %23 = loom.semaphore_take %22 : memref<1x32x128xf16> -> memref<1x32x128xf16>
            %24 = arith.muli %19, %c4096 overflow<nsw> : index
            %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%24], sizes: [1, 32, 128], strides: [4096, 128, 1] : memref<16x32x128xf16> to memref<1x32x128xf16, strided<[4096, 128, 1], offset: ?>>
            loom.copy %reinterpret_cast, %23 src_mem_space @mem_DRAM dst_mem_space @mem_L1, area : [8, 1] region : (UL : [%c0, %arg4], LR : [%c7, %arg4]) : memref<1x32x128xf16, strided<[4096, 128, 1], offset: ?>> to memref<1x32x128xf16>
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
            %36 = loom.semaphore_take %34 : memref<1x32x128xf16> -> memref<1x32x128xf16>
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
            %49 = loom.alloc [1, 32, 512] on @L1 : memref<1x32x512xf16>
            %50 = loom.semaphore_take %49 : memref<1x32x512xf16> -> memref<1x32x512xf16>
            %51 = loom.alloc [1, 512, 128] on @L1 : memref<1x512x128xf16>
            %52 = loom.semaphore_take %51 : memref<1x512x128xf16> -> memref<1x512x128xf16>
            %53 = arith.muli %19, %c1048576 overflow<nsw> : index
            %54 = arith.addi %53, %25 : index
            %reinterpret_cast_3 = memref.reinterpret_cast %arg0 to offset: [%54], sizes: [1, 128, 512], strides: [1048576, 8192, 1] : memref<16x128x8192xf16> to memref<1x128x512xf16, strided<[1048576, 8192, 1], offset: ?>>
            loom.copy %reinterpret_cast_3, %46 src_mem_space @mem_DRAM dst_mem_space @mem_L1, area : [1, 1] region : (UL : [%arg5, %arg4], LR : [%arg5, %arg4]) : memref<1x128x512xf16, strided<[1048576, 8192, 1], offset: ?>> to memref<1x128x512xf16>
            loom.batch_matmul ins(%23, %46 : memref<1x32x128xf16>, memref<1x128x512xf16>) outs(%48 : memref<1x32x512xf16>)
            loom.semaphore_give %46 : memref<1x128x512xf16>
            linalg.fill ins(%cst_1 : f16) outs(%40 : memref<1x32x1xf16>)
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, 0)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%48 : memref<1x32x512xf16>) outs(%40 : memref<1x32x1xf16>) {
            ^bb0(%in: f16, %out: f16):
              %75 = arith.maximumf %in, %out : f16
              linalg.yield %75 : f16
            }
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%33, %40 : memref<1x32x1xf16>, memref<1x32x1xf16>) outs(%40 : memref<1x32x1xf16>) {
            ^bb0(%in: f16, %in_5: f16, %out: f16):
              %75 = arith.mulf %in_5, %cst_2 : f16
              %76 = arith.cmpf ogt, %in, %75 : f16
              %77 = arith.select %76, %in, %75 : f16
              linalg.yield %77 : f16
            }
            %55 = loom.broadcast ins(%40 : memref<1x32x1xf16>) outs(%50 : memref<1x32x512xf16>) dim(2) -> memref<1x32x512xf16, strided<[?, ?, ?], offset: ?>>
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%48, %55 : memref<1x32x512xf16>, memref<1x32x512xf16, strided<[?, ?, ?], offset: ?>>) outs(%48 : memref<1x32x512xf16>) {
            ^bb0(%in: f16, %in_5: f16, %out: f16):
              %75 = arith.mulf %in, %cst_2 : f16
              %76 = arith.subf %75, %in_5 : f16
              %77 = math.exp %76 : f16
              linalg.yield %77 : f16
            }
            loom.semaphore_give %50 : memref<1x32x512xf16>
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, 0)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%48 : memref<1x32x512xf16>) outs(%42 : memref<1x32x1xf16>) {
            ^bb0(%in: f16, %out: f16):
              %75 = arith.addf %in, %out : f16
              linalg.yield %75 : f16
            }
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%33, %40 : memref<1x32x1xf16>, memref<1x32x1xf16>) outs(%44 : memref<1x32x1xf16>) {
            ^bb0(%in: f16, %in_5: f16, %out: f16):
              %75 = arith.subf %in, %in_5 : f16
              %76 = math.exp %75 : f16
              linalg.yield %76 : f16
            }
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%31, %44, %42 : memref<1x32x1xf16>, memref<1x32x1xf16>, memref<1x32x1xf16>) outs(%31 : memref<1x32x1xf16>) {
            ^bb0(%in: f16, %in_5: f16, %in_6: f16, %out: f16):
              %75 = arith.mulf %in, %in_5 : f16
              %76 = arith.addf %75, %in_6 : f16
              linalg.yield %76 : f16
            }
            loom.semaphore_give %42 : memref<1x32x1xf16>
            %56 = loom.broadcast ins(%44 : memref<1x32x1xf16>) outs(%36 : memref<1x32x128xf16>) dim(2) -> memref<1x32x128xf16, strided<[?, ?, ?], offset: ?>>
            loom.semaphore_give %44 : memref<1x32x1xf16>
            %57 = arith.muli %21, %c65536 : index
            %58 = arith.addi %53, %57 : index
            %reinterpret_cast_4 = memref.reinterpret_cast %arg1 to offset: [%58], sizes: [1, 512, 128], strides: [1048576, 128, 1] : memref<16x8192x128xf16> to memref<1x512x128xf16, strided<[1048576, 128, 1], offset: ?>>
            loom.copy %reinterpret_cast_4, %52 src_mem_space @mem_DRAM dst_mem_space @mem_L1, area : [1, 1] region : (UL : [%arg5, %arg4], LR : [%arg5, %arg4]) : memref<1x512x128xf16, strided<[1048576, 128, 1], offset: ?>> to memref<1x512x128xf16>
            loom.batch_matmul ins(%48, %52 : memref<1x32x512xf16>, memref<1x512x128xf16>) outs(%38 : memref<1x32x128xf16>)
            loom.semaphore_give %52 : memref<1x512x128xf16>
            loom.semaphore_give %48 : memref<1x32x512xf16>
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%38, %27, %56 : memref<1x32x128xf16>, memref<1x32x128xf16>, memref<1x32x128xf16, strided<[?, ?, ?], offset: ?>>) outs(%27 : memref<1x32x128xf16>) {
            ^bb0(%in: f16, %in_5: f16, %in_6: f16, %out: f16):
              %75 = arith.mulf %in_5, %in_6 : f16
              %76 = arith.addf %in, %75 : f16
              linalg.yield %76 : f16
            }
            loom.semaphore_give %38 : memref<1x32x128xf16>
            loom.semaphore_give %36 : memref<1x32x128xf16>
            linalg.copy ins(%40 : memref<1x32x1xf16>) outs(%33 : memref<1x32x1xf16>)
            loom.semaphore_give %40 : memref<1x32x1xf16>
            loom.semaphore_give %23 : memref<1x32x128xf16>
            %59 = loom.alloc [1, 32, 1] on @L1 : memref<1x32x1xf16>
            %60 = loom.semaphore_take %59 : memref<1x32x1xf16> -> memref<1x32x1xf16>
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%31, %33 : memref<1x32x1xf16>, memref<1x32x1xf16>) outs(%60 : memref<1x32x1xf16>) {
            ^bb0(%in: f16, %in_5: f16, %out: f16):
              %75 = math.log %in : f16
              %76 = arith.addf %75, %in_5 : f16
              linalg.yield %76 : f16
            }
            loom.semaphore_give %33 : memref<1x32x1xf16>
            %61 = loom.broadcast ins(%31 : memref<1x32x1xf16>) outs(%35 : memref<1x32x128xf16>) dim(2) -> memref<1x32x128xf16, strided<[?, ?, ?], offset: ?>>
            loom.semaphore_give %31 : memref<1x32x1xf16>
            %62 = loom.alloc [1, 32, 128] on @L1 : memref<1x32x128xf16>
            %63 = loom.semaphore_take %62 : memref<1x32x128xf16> -> memref<1x32x128xf16>
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%27, %61 : memref<1x32x128xf16>, memref<1x32x128xf16, strided<[?, ?, ?], offset: ?>>) outs(%63 : memref<1x32x128xf16>) {
            ^bb0(%in: f16, %in_5: f16, %out: f16):
              %75 = arith.divf %in, %in_5 : f16
              linalg.yield %75 : f16
            }
            loom.semaphore_give %35 : memref<1x32x128xf16>
            loom.semaphore_give %27 : memref<1x32x128xf16>
            %64 = loom.alloc [16, 1, 32, 1] on @L1 : memref<16x1x32x1xf16>
            %65 = loom.semaphore_take %64 : memref<16x1x32x1xf16> -> memref<16x1x32x1xf16>
            loom.gather %60, %65 src_mem_space @mem_array_L1 dst_mem_space @mem_array_L1 across(%21 : index), area : [8, 1] region : (UL : [%c0, %arg4], LR : [%c7, %arg4]) : memref<1x32x1xf16> to memref<16x1x32x1xf16>
            loom.semaphore_give %60 : memref<1x32x1xf16>
            %66 = loom.alloc [16, 1, 32, 128] on @L1 : memref<16x1x32x128xf16>
            %67 = loom.semaphore_take %66 : memref<16x1x32x128xf16> -> memref<16x1x32x128xf16>
            loom.gather %63, %67 src_mem_space @mem_array_L1 dst_mem_space @mem_array_L1 across(%21 : index), area : [8, 1] region : (UL : [%c0, %arg4], LR : [%c7, %arg4]) : memref<1x32x128xf16> to memref<16x1x32x128xf16>
            loom.semaphore_give %63 : memref<1x32x128xf16>
            %68 = arith.cmpi eq, %21, %c0 : index
            %69 = loom.alloc [1, 32, 128] on @L1 : memref<1x32x128xf16>
            %70 = loom.semaphore_take %69 : memref<1x32x128xf16> -> memref<1x32x128xf16>
            %71 = loom.alloc [16, 1, 32, 1] on @L1 : memref<16x1x32x1xf16>
            %72 = loom.semaphore_take %71 : memref<16x1x32x1xf16> -> memref<16x1x32x1xf16>
            %73 = loom.alloc [16, 1, 32, 128] on @L1 : memref<16x1x32x128xf16>
            %74 = loom.semaphore_take %73 : memref<16x1x32x128xf16> -> memref<16x1x32x128xf16>
            scf.if %68 {
              linalg.fill ins(%cst_1 : f16) outs(%30 : memref<1x32x1xf16>)
              linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1, d2, d3)>], iterator_types = ["reduction", "parallel", "parallel", "parallel"]} ins(%65 : memref<16x1x32x1xf16>) outs(%30 : memref<1x32x1xf16>) {
              ^bb0(%in: f16, %out: f16):
                %76 = arith.maximumf %in, %out : f16
                linalg.yield %76 : f16
              }
              linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%65, %30 : memref<16x1x32x1xf16>, memref<1x32x1xf16>) outs(%72 : memref<16x1x32x1xf16>) {
              ^bb0(%in: f16, %in_6: f16, %out: f16):
                %76 = arith.subf %in, %in_6 : f16
                %77 = math.exp %76 : f16
                linalg.yield %77 : f16
              }
              loom.semaphore_give %65 : memref<16x1x32x1xf16>
              loom.semaphore_give %30 : memref<1x32x1xf16>
              linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1, d2, d3)>], iterator_types = ["reduction", "parallel", "parallel", "parallel"]} ins(%72 : memref<16x1x32x1xf16>) outs(%29 : memref<1x32x1xf16>) {
              ^bb0(%in: f16, %out: f16):
                %76 = arith.addf %in, %out : f16
                linalg.yield %76 : f16
              }
              linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%72, %29 : memref<16x1x32x1xf16>, memref<1x32x1xf16>) outs(%72 : memref<16x1x32x1xf16>) {
              ^bb0(%in: f16, %in_6: f16, %out: f16):
                %76 = arith.divf %in, %in_6 : f16
                linalg.yield %76 : f16
              }
              loom.semaphore_give %29 : memref<1x32x1xf16>
              %alloc = memref.alloc() {alignment = 64 : i64} : memref<16x1x32x128xf16>
              %75 = loom.broadcast ins(%72 : memref<16x1x32x1xf16>) outs(%alloc : memref<16x1x32x128xf16>) dim(3) -> memref<16x1x32x128xf16, strided<[?, ?, ?, ?], offset: ?>>
              loom.semaphore_give %72 : memref<16x1x32x1xf16>
              linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%67, %75 : memref<16x1x32x128xf16>, memref<16x1x32x128xf16, strided<[?, ?, ?, ?], offset: ?>>) outs(%74 : memref<16x1x32x128xf16>) {
              ^bb0(%in: f16, %in_6: f16, %out: f16):
                %76 = arith.mulf %in, %in_6 : f16
                linalg.yield %76 : f16
              }
              loom.semaphore_give %67 : memref<16x1x32x128xf16>
              linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1, d2, d3)>], iterator_types = ["reduction", "parallel", "parallel", "parallel"]} ins(%74 : memref<16x1x32x128xf16>) outs(%70 : memref<1x32x128xf16>) {
              ^bb0(%in: f16, %out: f16):
                %76 = arith.addf %in, %out : f16
                linalg.yield %76 : f16
              }
              loom.semaphore_give %74 : memref<16x1x32x128xf16>
              %reinterpret_cast_5 = memref.reinterpret_cast %arg3 to offset: [%24], sizes: [1, 32, 128], strides: [4096, 128, 1] : memref<16x32x128xf16> to memref<1x32x128xf16, strided<[4096, 128, 1], offset: ?>>
              loom.copy %70, %reinterpret_cast_5 src_mem_space @mem_L1 dst_mem_space @mem_DRAM, area : [1, 1] region : (UL : [%arg5, %arg4], LR : [%arg5, %arg4]) : memref<1x32x128xf16> to memref<1x32x128xf16, strided<[4096, 128, 1], offset: ?>>
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
          %18 = arith.muli %arg7, %c4 overflow<nsw> : index
          %19 = arith.addi %arg4, %18 : index
          %20 = arith.muli %arg5, %c8 overflow<nsw> : index
          %21 = arith.addi %20, %arg6 : index
          %22 = loom.alloc [1, 32, 128] on @L1 : memref<1x32x128xf16>
          %23 = loom.semaphore_take %22 : memref<1x32x128xf16> -> memref<1x32x128xf16>
          %24 = arith.muli %19, %c4096 overflow<nsw> : index
          %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%24], sizes: [1, 32, 128], strides: [4096, 128, 1] : memref<16x32x128xf16> to memref<1x32x128xf16, strided<[4096, 128, 1], offset: ?>>
          %25 = arith.muli %arg4, %c2 : index
          %26 = arith.addi %25, %c1 : index
          loom.copy %reinterpret_cast, %23 src_mem_space @mem_DRAM dst_mem_space @mem_L1, area : [8, 2] region : (UL : [%c0, %25], LR : [%c7, %26]) : memref<1x32x128xf16, strided<[4096, 128, 1], offset: ?>> to memref<1x32x128xf16>
          %27 = arith.muli %21, %c512 : index
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
          %38 = loom.semaphore_take %36 : memref<1x32x128xf16> -> memref<1x32x128xf16>
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
          %51 = loom.alloc [1, 32, 512] on @L1 : memref<1x32x512xf16>
          %52 = loom.semaphore_take %51 : memref<1x32x512xf16> -> memref<1x32x512xf16>
          %53 = loom.alloc [1, 512, 128] on @L1 : memref<1x512x128xf16>
          %54 = loom.semaphore_take %53 : memref<1x512x128xf16> -> memref<1x512x128xf16>
          %55 = arith.muli %19, %c1048576 overflow<nsw> : index
          %56 = arith.addi %55, %27 : index
          %reinterpret_cast_3 = memref.reinterpret_cast %arg0 to offset: [%56], sizes: [1, 128, 512], strides: [1048576, 8192, 1] : memref<16x128x8192xf16> to memref<1x128x512xf16, strided<[1048576, 8192, 1], offset: ?>>
          %57 = arith.addi %arg6, %25 : index
          loom.copy %reinterpret_cast_3, %48 src_mem_space @mem_DRAM dst_mem_space @mem_L1, area : [1, 1] region : (UL : [%arg5, %57], LR : [%arg5, %57]) : memref<1x128x512xf16, strided<[1048576, 8192, 1], offset: ?>> to memref<1x128x512xf16>
          loom.batch_matmul ins(%23, %48 : memref<1x32x128xf16>, memref<1x128x512xf16>) outs(%50 : memref<1x32x512xf16>)
          loom.semaphore_give %48 : memref<1x128x512xf16>
          linalg.fill ins(%cst_1 : f16) outs(%42 : memref<1x32x1xf16>)
          linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, 0)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%50 : memref<1x32x512xf16>) outs(%42 : memref<1x32x1xf16>) {
          ^bb0(%in: f16, %out: f16):
            %78 = arith.maximumf %in, %out : f16
            linalg.yield %78 : f16
          }
          linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%35, %42 : memref<1x32x1xf16>, memref<1x32x1xf16>) outs(%42 : memref<1x32x1xf16>) {
          ^bb0(%in: f16, %in_5: f16, %out: f16):
            %78 = arith.mulf %in_5, %cst_2 : f16
            %79 = arith.cmpf ogt, %in, %78 : f16
            %80 = arith.select %79, %in, %78 : f16
            linalg.yield %80 : f16
          }
          %58 = loom.broadcast ins(%42 : memref<1x32x1xf16>) outs(%52 : memref<1x32x512xf16>) dim(2) -> memref<1x32x512xf16, strided<[?, ?, ?], offset: ?>>
          linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%50, %58 : memref<1x32x512xf16>, memref<1x32x512xf16, strided<[?, ?, ?], offset: ?>>) outs(%50 : memref<1x32x512xf16>) {
          ^bb0(%in: f16, %in_5: f16, %out: f16):
            %78 = arith.mulf %in, %cst_2 : f16
            %79 = arith.subf %78, %in_5 : f16
            %80 = math.exp %79 : f16
            linalg.yield %80 : f16
          }
          loom.semaphore_give %52 : memref<1x32x512xf16>
          linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, 0)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%50 : memref<1x32x512xf16>) outs(%44 : memref<1x32x1xf16>) {
          ^bb0(%in: f16, %out: f16):
            %78 = arith.addf %in, %out : f16
            linalg.yield %78 : f16
          }
          linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%35, %42 : memref<1x32x1xf16>, memref<1x32x1xf16>) outs(%46 : memref<1x32x1xf16>) {
          ^bb0(%in: f16, %in_5: f16, %out: f16):
            %78 = arith.subf %in, %in_5 : f16
            %79 = math.exp %78 : f16
            linalg.yield %79 : f16
          }
          linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%33, %46, %44 : memref<1x32x1xf16>, memref<1x32x1xf16>, memref<1x32x1xf16>) outs(%33 : memref<1x32x1xf16>) {
          ^bb0(%in: f16, %in_5: f16, %in_6: f16, %out: f16):
            %78 = arith.mulf %in, %in_5 : f16
            %79 = arith.addf %78, %in_6 : f16
            linalg.yield %79 : f16
          }
          loom.semaphore_give %44 : memref<1x32x1xf16>
          %59 = loom.broadcast ins(%46 : memref<1x32x1xf16>) outs(%38 : memref<1x32x128xf16>) dim(2) -> memref<1x32x128xf16, strided<[?, ?, ?], offset: ?>>
          loom.semaphore_give %46 : memref<1x32x1xf16>
          %60 = arith.muli %21, %c65536 : index
          %61 = arith.addi %55, %60 : index
          %reinterpret_cast_4 = memref.reinterpret_cast %arg1 to offset: [%61], sizes: [1, 512, 128], strides: [1048576, 128, 1] : memref<16x8192x128xf16> to memref<1x512x128xf16, strided<[1048576, 128, 1], offset: ?>>
          loom.copy %reinterpret_cast_4, %54 src_mem_space @mem_DRAM dst_mem_space @mem_L1, area : [1, 1] region : (UL : [%arg5, %57], LR : [%arg5, %57]) : memref<1x512x128xf16, strided<[1048576, 128, 1], offset: ?>> to memref<1x512x128xf16>
          loom.batch_matmul ins(%50, %54 : memref<1x32x512xf16>, memref<1x512x128xf16>) outs(%40 : memref<1x32x128xf16>)
          loom.semaphore_give %54 : memref<1x512x128xf16>
          loom.semaphore_give %50 : memref<1x32x512xf16>
          linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%40, %29, %59 : memref<1x32x128xf16>, memref<1x32x128xf16>, memref<1x32x128xf16, strided<[?, ?, ?], offset: ?>>) outs(%29 : memref<1x32x128xf16>) {
          ^bb0(%in: f16, %in_5: f16, %in_6: f16, %out: f16):
            %78 = arith.mulf %in_5, %in_6 : f16
            %79 = arith.addf %in, %78 : f16
            linalg.yield %79 : f16
          }
          loom.semaphore_give %40 : memref<1x32x128xf16>
          loom.semaphore_give %38 : memref<1x32x128xf16>
          linalg.copy ins(%42 : memref<1x32x1xf16>) outs(%35 : memref<1x32x1xf16>)
          loom.semaphore_give %42 : memref<1x32x1xf16>
          loom.semaphore_give %23 : memref<1x32x128xf16>
          %62 = loom.alloc [1, 32, 1] on @L1 : memref<1x32x1xf16>
          %63 = loom.semaphore_take %62 : memref<1x32x1xf16> -> memref<1x32x1xf16>
          linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%33, %35 : memref<1x32x1xf16>, memref<1x32x1xf16>) outs(%63 : memref<1x32x1xf16>) {
          ^bb0(%in: f16, %in_5: f16, %out: f16):
            %78 = math.log %in : f16
            %79 = arith.addf %78, %in_5 : f16
            linalg.yield %79 : f16
          }
          loom.semaphore_give %35 : memref<1x32x1xf16>
          %64 = loom.broadcast ins(%33 : memref<1x32x1xf16>) outs(%37 : memref<1x32x128xf16>) dim(2) -> memref<1x32x128xf16, strided<[?, ?, ?], offset: ?>>
          loom.semaphore_give %33 : memref<1x32x1xf16>
          %65 = loom.alloc [1, 32, 128] on @L1 : memref<1x32x128xf16>
          %66 = loom.semaphore_take %65 : memref<1x32x128xf16> -> memref<1x32x128xf16>
          linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%29, %64 : memref<1x32x128xf16>, memref<1x32x128xf16, strided<[?, ?, ?], offset: ?>>) outs(%66 : memref<1x32x128xf16>) {
          ^bb0(%in: f16, %in_5: f16, %out: f16):
            %78 = arith.divf %in, %in_5 : f16
            linalg.yield %78 : f16
          }
          loom.semaphore_give %37 : memref<1x32x128xf16>
          loom.semaphore_give %29 : memref<1x32x128xf16>
          %67 = loom.alloc [16, 1, 32, 1] on @L1 : memref<16x1x32x1xf16>
          %68 = loom.semaphore_take %67 : memref<16x1x32x1xf16> -> memref<16x1x32x1xf16>
          loom.gather %63, %68 src_mem_space @mem_array_L1 dst_mem_space @mem_array_L1 across(%21 : index), area : [8, 2] region : (UL : [%c0, %25], LR : [%c7, %26]) : memref<1x32x1xf16> to memref<16x1x32x1xf16>
          loom.semaphore_give %63 : memref<1x32x1xf16>
          %69 = loom.alloc [16, 1, 32, 128] on @L1 : memref<16x1x32x128xf16>
          %70 = loom.semaphore_take %69 : memref<16x1x32x128xf16> -> memref<16x1x32x128xf16>
          loom.gather %66, %70 src_mem_space @mem_array_L1 dst_mem_space @mem_array_L1 across(%21 : index), area : [8, 2] region : (UL : [%c0, %25], LR : [%c7, %26]) : memref<1x32x128xf16> to memref<16x1x32x128xf16>
          loom.semaphore_give %66 : memref<1x32x128xf16>
          %71 = arith.cmpi eq, %21, %c0 : index
          %72 = loom.alloc [1, 32, 128] on @L1 : memref<1x32x128xf16>
          %73 = loom.semaphore_take %72 : memref<1x32x128xf16> -> memref<1x32x128xf16>
          %74 = loom.alloc [16, 1, 32, 1] on @L1 : memref<16x1x32x1xf16>
          %75 = loom.semaphore_take %74 : memref<16x1x32x1xf16> -> memref<16x1x32x1xf16>
          %76 = loom.alloc [16, 1, 32, 128] on @L1 : memref<16x1x32x128xf16>
          %77 = loom.semaphore_take %76 : memref<16x1x32x128xf16> -> memref<16x1x32x128xf16>
          scf.if %71 {
            linalg.fill ins(%cst_1 : f16) outs(%32 : memref<1x32x1xf16>)
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1, d2, d3)>], iterator_types = ["reduction", "parallel", "parallel", "parallel"]} ins(%68 : memref<16x1x32x1xf16>) outs(%32 : memref<1x32x1xf16>) {
            ^bb0(%in: f16, %out: f16):
              %79 = arith.maximumf %in, %out : f16
              linalg.yield %79 : f16
            }
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%68, %32 : memref<16x1x32x1xf16>, memref<1x32x1xf16>) outs(%75 : memref<16x1x32x1xf16>) {
            ^bb0(%in: f16, %in_6: f16, %out: f16):
              %79 = arith.subf %in, %in_6 : f16
              %80 = math.exp %79 : f16
              linalg.yield %80 : f16
            }
            loom.semaphore_give %68 : memref<16x1x32x1xf16>
            loom.semaphore_give %32 : memref<1x32x1xf16>
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1, d2, d3)>], iterator_types = ["reduction", "parallel", "parallel", "parallel"]} ins(%75 : memref<16x1x32x1xf16>) outs(%31 : memref<1x32x1xf16>) {
            ^bb0(%in: f16, %out: f16):
              %79 = arith.addf %in, %out : f16
              linalg.yield %79 : f16
            }
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%75, %31 : memref<16x1x32x1xf16>, memref<1x32x1xf16>) outs(%75 : memref<16x1x32x1xf16>) {
            ^bb0(%in: f16, %in_6: f16, %out: f16):
              %79 = arith.divf %in, %in_6 : f16
              linalg.yield %79 : f16
            }
            loom.semaphore_give %31 : memref<1x32x1xf16>
            %alloc = memref.alloc() {alignment = 64 : i64} : memref<16x1x32x128xf16>
            %78 = loom.broadcast ins(%75 : memref<16x1x32x1xf16>) outs(%alloc : memref<16x1x32x128xf16>) dim(3) -> memref<16x1x32x128xf16, strided<[?, ?, ?, ?], offset: ?>>
            loom.semaphore_give %75 : memref<16x1x32x1xf16>
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%70, %78 : memref<16x1x32x128xf16>, memref<16x1x32x128xf16, strided<[?, ?, ?, ?], offset: ?>>) outs(%77 : memref<16x1x32x128xf16>) {
            ^bb0(%in: f16, %in_6: f16, %out: f16):
              %79 = arith.mulf %in, %in_6 : f16
              linalg.yield %79 : f16
            }
            loom.semaphore_give %70 : memref<16x1x32x128xf16>
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1, d2, d3)>], iterator_types = ["reduction", "parallel", "parallel", "parallel"]} ins(%77 : memref<16x1x32x128xf16>) outs(%73 : memref<1x32x128xf16>) {
            ^bb0(%in: f16, %out: f16):
              %79 = arith.addf %in, %out : f16
              linalg.yield %79 : f16
            }
            loom.semaphore_give %77 : memref<16x1x32x128xf16>
            %reinterpret_cast_5 = memref.reinterpret_cast %arg3 to offset: [%24], sizes: [1, 32, 128], strides: [4096, 128, 1] : memref<16x32x128xf16> to memref<1x32x128xf16, strided<[4096, 128, 1], offset: ?>>
            loom.copy %73, %reinterpret_cast_5 src_mem_space @mem_L1 dst_mem_space @mem_DRAM, area : [1, 1] region : (UL : [%arg5, %57], LR : [%arg5, %57]) : memref<1x32x128xf16> to memref<1x32x128xf16, strided<[4096, 128, 1], offset: ?>>
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
          %18 = arith.muli %arg7, %c2 overflow<nsw> : index
          %19 = arith.addi %arg4, %18 : index
          %20 = arith.muli %arg5, %c8 overflow<nsw> : index
          %21 = arith.addi %20, %arg6 : index
          %22 = loom.alloc [1, 32, 128] on @L1 : memref<1x32x128xf16>
          %23 = loom.semaphore_take %22 : memref<1x32x128xf16> -> memref<1x32x128xf16>
          %24 = arith.muli %19, %c4096 overflow<nsw> : index
          %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%24], sizes: [1, 32, 128], strides: [4096, 128, 1] : memref<16x32x128xf16> to memref<1x32x128xf16, strided<[4096, 128, 1], offset: ?>>
          %25 = arith.muli %arg4, %c4 : index
          %26 = arith.addi %25, %c3 : index
          loom.copy %reinterpret_cast, %23 src_mem_space @mem_DRAM dst_mem_space @mem_L1, area : [8, 4] region : (UL : [%c0, %25], LR : [%c7, %26]) : memref<1x32x128xf16, strided<[4096, 128, 1], offset: ?>> to memref<1x32x128xf16>
          %27 = arith.muli %21, %c512 : index
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
          %38 = loom.semaphore_take %36 : memref<1x32x128xf16> -> memref<1x32x128xf16>
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
          %51 = loom.alloc [1, 32, 512] on @L1 : memref<1x32x512xf16>
          %52 = loom.semaphore_take %51 : memref<1x32x512xf16> -> memref<1x32x512xf16>
          %53 = loom.alloc [1, 512, 128] on @L1 : memref<1x512x128xf16>
          %54 = loom.semaphore_take %53 : memref<1x512x128xf16> -> memref<1x512x128xf16>
          %55 = arith.muli %19, %c1048576 overflow<nsw> : index
          %56 = arith.addi %55, %27 : index
          %reinterpret_cast_3 = memref.reinterpret_cast %arg0 to offset: [%56], sizes: [1, 128, 512], strides: [1048576, 8192, 1] : memref<16x128x8192xf16> to memref<1x128x512xf16, strided<[1048576, 8192, 1], offset: ?>>
          %57 = arith.addi %arg6, %25 : index
          loom.copy %reinterpret_cast_3, %48 src_mem_space @mem_DRAM dst_mem_space @mem_L1, area : [1, 1] region : (UL : [%arg5, %57], LR : [%arg5, %57]) : memref<1x128x512xf16, strided<[1048576, 8192, 1], offset: ?>> to memref<1x128x512xf16>
          loom.batch_matmul ins(%23, %48 : memref<1x32x128xf16>, memref<1x128x512xf16>) outs(%50 : memref<1x32x512xf16>)
          loom.semaphore_give %48 : memref<1x128x512xf16>
          linalg.fill ins(%cst_1 : f16) outs(%42 : memref<1x32x1xf16>)
          linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, 0)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%50 : memref<1x32x512xf16>) outs(%42 : memref<1x32x1xf16>) {
          ^bb0(%in: f16, %out: f16):
            %78 = arith.maximumf %in, %out : f16
            linalg.yield %78 : f16
          }
          linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%35, %42 : memref<1x32x1xf16>, memref<1x32x1xf16>) outs(%42 : memref<1x32x1xf16>) {
          ^bb0(%in: f16, %in_5: f16, %out: f16):
            %78 = arith.mulf %in_5, %cst_2 : f16
            %79 = arith.cmpf ogt, %in, %78 : f16
            %80 = arith.select %79, %in, %78 : f16
            linalg.yield %80 : f16
          }
          %58 = loom.broadcast ins(%42 : memref<1x32x1xf16>) outs(%52 : memref<1x32x512xf16>) dim(2) -> memref<1x32x512xf16, strided<[?, ?, ?], offset: ?>>
          linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%50, %58 : memref<1x32x512xf16>, memref<1x32x512xf16, strided<[?, ?, ?], offset: ?>>) outs(%50 : memref<1x32x512xf16>) {
          ^bb0(%in: f16, %in_5: f16, %out: f16):
            %78 = arith.mulf %in, %cst_2 : f16
            %79 = arith.subf %78, %in_5 : f16
            %80 = math.exp %79 : f16
            linalg.yield %80 : f16
          }
          loom.semaphore_give %52 : memref<1x32x512xf16>
          linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, 0)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%50 : memref<1x32x512xf16>) outs(%44 : memref<1x32x1xf16>) {
          ^bb0(%in: f16, %out: f16):
            %78 = arith.addf %in, %out : f16
            linalg.yield %78 : f16
          }
          linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%35, %42 : memref<1x32x1xf16>, memref<1x32x1xf16>) outs(%46 : memref<1x32x1xf16>) {
          ^bb0(%in: f16, %in_5: f16, %out: f16):
            %78 = arith.subf %in, %in_5 : f16
            %79 = math.exp %78 : f16
            linalg.yield %79 : f16
          }
          linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%33, %46, %44 : memref<1x32x1xf16>, memref<1x32x1xf16>, memref<1x32x1xf16>) outs(%33 : memref<1x32x1xf16>) {
          ^bb0(%in: f16, %in_5: f16, %in_6: f16, %out: f16):
            %78 = arith.mulf %in, %in_5 : f16
            %79 = arith.addf %78, %in_6 : f16
            linalg.yield %79 : f16
          }
          loom.semaphore_give %44 : memref<1x32x1xf16>
          %59 = loom.broadcast ins(%46 : memref<1x32x1xf16>) outs(%38 : memref<1x32x128xf16>) dim(2) -> memref<1x32x128xf16, strided<[?, ?, ?], offset: ?>>
          loom.semaphore_give %46 : memref<1x32x1xf16>
          %60 = arith.muli %21, %c65536 : index
          %61 = arith.addi %55, %60 : index
          %reinterpret_cast_4 = memref.reinterpret_cast %arg1 to offset: [%61], sizes: [1, 512, 128], strides: [1048576, 128, 1] : memref<16x8192x128xf16> to memref<1x512x128xf16, strided<[1048576, 128, 1], offset: ?>>
          loom.copy %reinterpret_cast_4, %54 src_mem_space @mem_DRAM dst_mem_space @mem_L1, area : [1, 1] region : (UL : [%arg5, %57], LR : [%arg5, %57]) : memref<1x512x128xf16, strided<[1048576, 128, 1], offset: ?>> to memref<1x512x128xf16>
          loom.batch_matmul ins(%50, %54 : memref<1x32x512xf16>, memref<1x512x128xf16>) outs(%40 : memref<1x32x128xf16>)
          loom.semaphore_give %54 : memref<1x512x128xf16>
          loom.semaphore_give %50 : memref<1x32x512xf16>
          linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%40, %29, %59 : memref<1x32x128xf16>, memref<1x32x128xf16>, memref<1x32x128xf16, strided<[?, ?, ?], offset: ?>>) outs(%29 : memref<1x32x128xf16>) {
          ^bb0(%in: f16, %in_5: f16, %in_6: f16, %out: f16):
            %78 = arith.mulf %in_5, %in_6 : f16
            %79 = arith.addf %in, %78 : f16
            linalg.yield %79 : f16
          }
          loom.semaphore_give %40 : memref<1x32x128xf16>
          loom.semaphore_give %38 : memref<1x32x128xf16>
          linalg.copy ins(%42 : memref<1x32x1xf16>) outs(%35 : memref<1x32x1xf16>)
          loom.semaphore_give %42 : memref<1x32x1xf16>
          loom.semaphore_give %23 : memref<1x32x128xf16>
          %62 = loom.alloc [1, 32, 1] on @L1 : memref<1x32x1xf16>
          %63 = loom.semaphore_take %62 : memref<1x32x1xf16> -> memref<1x32x1xf16>
          linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%33, %35 : memref<1x32x1xf16>, memref<1x32x1xf16>) outs(%63 : memref<1x32x1xf16>) {
          ^bb0(%in: f16, %in_5: f16, %out: f16):
            %78 = math.log %in : f16
            %79 = arith.addf %78, %in_5 : f16
            linalg.yield %79 : f16
          }
          loom.semaphore_give %35 : memref<1x32x1xf16>
          %64 = loom.broadcast ins(%33 : memref<1x32x1xf16>) outs(%37 : memref<1x32x128xf16>) dim(2) -> memref<1x32x128xf16, strided<[?, ?, ?], offset: ?>>
          loom.semaphore_give %33 : memref<1x32x1xf16>
          %65 = loom.alloc [1, 32, 128] on @L1 : memref<1x32x128xf16>
          %66 = loom.semaphore_take %65 : memref<1x32x128xf16> -> memref<1x32x128xf16>
          linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%29, %64 : memref<1x32x128xf16>, memref<1x32x128xf16, strided<[?, ?, ?], offset: ?>>) outs(%66 : memref<1x32x128xf16>) {
          ^bb0(%in: f16, %in_5: f16, %out: f16):
            %78 = arith.divf %in, %in_5 : f16
            linalg.yield %78 : f16
          }
          loom.semaphore_give %37 : memref<1x32x128xf16>
          loom.semaphore_give %29 : memref<1x32x128xf16>
          %67 = loom.alloc [16, 1, 32, 1] on @L1 : memref<16x1x32x1xf16>
          %68 = loom.semaphore_take %67 : memref<16x1x32x1xf16> -> memref<16x1x32x1xf16>
          loom.gather %63, %68 src_mem_space @mem_array_L1 dst_mem_space @mem_array_L1 across(%21 : index), area : [8, 4] region : (UL : [%c0, %25], LR : [%c7, %26]) : memref<1x32x1xf16> to memref<16x1x32x1xf16>
          loom.semaphore_give %63 : memref<1x32x1xf16>
          %69 = loom.alloc [16, 1, 32, 128] on @L1 : memref<16x1x32x128xf16>
          %70 = loom.semaphore_take %69 : memref<16x1x32x128xf16> -> memref<16x1x32x128xf16>
          loom.gather %66, %70 src_mem_space @mem_array_L1 dst_mem_space @mem_array_L1 across(%21 : index), area : [8, 4] region : (UL : [%c0, %25], LR : [%c7, %26]) : memref<1x32x128xf16> to memref<16x1x32x128xf16>
          loom.semaphore_give %66 : memref<1x32x128xf16>
          %71 = arith.cmpi eq, %21, %c0 : index
          %72 = loom.alloc [1, 32, 128] on @L1 : memref<1x32x128xf16>
          %73 = loom.semaphore_take %72 : memref<1x32x128xf16> -> memref<1x32x128xf16>
          %74 = loom.alloc [16, 1, 32, 1] on @L1 : memref<16x1x32x1xf16>
          %75 = loom.semaphore_take %74 : memref<16x1x32x1xf16> -> memref<16x1x32x1xf16>
          %76 = loom.alloc [16, 1, 32, 128] on @L1 : memref<16x1x32x128xf16>
          %77 = loom.semaphore_take %76 : memref<16x1x32x128xf16> -> memref<16x1x32x128xf16>
          scf.if %71 {
            linalg.fill ins(%cst_1 : f16) outs(%32 : memref<1x32x1xf16>)
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1, d2, d3)>], iterator_types = ["reduction", "parallel", "parallel", "parallel"]} ins(%68 : memref<16x1x32x1xf16>) outs(%32 : memref<1x32x1xf16>) {
            ^bb0(%in: f16, %out: f16):
              %79 = arith.maximumf %in, %out : f16
              linalg.yield %79 : f16
            }
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%68, %32 : memref<16x1x32x1xf16>, memref<1x32x1xf16>) outs(%75 : memref<16x1x32x1xf16>) {
            ^bb0(%in: f16, %in_6: f16, %out: f16):
              %79 = arith.subf %in, %in_6 : f16
              %80 = math.exp %79 : f16
              linalg.yield %80 : f16
            }
            loom.semaphore_give %68 : memref<16x1x32x1xf16>
            loom.semaphore_give %32 : memref<1x32x1xf16>
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1, d2, d3)>], iterator_types = ["reduction", "parallel", "parallel", "parallel"]} ins(%75 : memref<16x1x32x1xf16>) outs(%31 : memref<1x32x1xf16>) {
            ^bb0(%in: f16, %out: f16):
              %79 = arith.addf %in, %out : f16
              linalg.yield %79 : f16
            }
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%75, %31 : memref<16x1x32x1xf16>, memref<1x32x1xf16>) outs(%75 : memref<16x1x32x1xf16>) {
            ^bb0(%in: f16, %in_6: f16, %out: f16):
              %79 = arith.divf %in, %in_6 : f16
              linalg.yield %79 : f16
            }
            loom.semaphore_give %31 : memref<1x32x1xf16>
            %alloc = memref.alloc() {alignment = 64 : i64} : memref<16x1x32x128xf16>
            %78 = loom.broadcast ins(%75 : memref<16x1x32x1xf16>) outs(%alloc : memref<16x1x32x128xf16>) dim(3) -> memref<16x1x32x128xf16, strided<[?, ?, ?, ?], offset: ?>>
            loom.semaphore_give %75 : memref<16x1x32x1xf16>
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%70, %78 : memref<16x1x32x128xf16>, memref<16x1x32x128xf16, strided<[?, ?, ?, ?], offset: ?>>) outs(%77 : memref<16x1x32x128xf16>) {
            ^bb0(%in: f16, %in_6: f16, %out: f16):
              %79 = arith.mulf %in, %in_6 : f16
              linalg.yield %79 : f16
            }
            loom.semaphore_give %70 : memref<16x1x32x128xf16>
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1, d2, d3)>], iterator_types = ["reduction", "parallel", "parallel", "parallel"]} ins(%77 : memref<16x1x32x128xf16>) outs(%73 : memref<1x32x128xf16>) {
            ^bb0(%in: f16, %out: f16):
              %79 = arith.addf %in, %out : f16
              linalg.yield %79 : f16
            }
            loom.semaphore_give %77 : memref<16x1x32x128xf16>
            %reinterpret_cast_5 = memref.reinterpret_cast %arg3 to offset: [%24], sizes: [1, 32, 128], strides: [4096, 128, 1] : memref<16x32x128xf16> to memref<1x32x128xf16, strided<[4096, 128, 1], offset: ?>>
            loom.copy %73, %reinterpret_cast_5 src_mem_space @mem_L1 dst_mem_space @mem_DRAM, area : [1, 1] region : (UL : [%arg5, %57], LR : [%arg5, %57]) : memref<1x32x128xf16> to memref<1x32x128xf16, strided<[4096, 128, 1], offset: ?>>
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
          %18 = arith.muli %arg4, %c8 overflow<nsw> : index
          %19 = arith.addi %18, %arg5 : index
          %20 = loom.alloc [1, 32, 128] on @L1 : memref<1x32x128xf16>
          %21 = loom.semaphore_take %20 : memref<1x32x128xf16> -> memref<1x32x128xf16>
          %22 = arith.muli %arg6, %c4096 overflow<nsw> : index
          %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%22], sizes: [1, 32, 128], strides: [4096, 128, 1] : memref<16x32x128xf16> to memref<1x32x128xf16, strided<[4096, 128, 1], offset: ?>>
          loom.copy %reinterpret_cast, %21 src_mem_space @mem_DRAM dst_mem_space @mem_L1, area : [8, 8] region : (UL : [%c0, %c0], LR : [%c7, %c7]) : memref<1x32x128xf16, strided<[4096, 128, 1], offset: ?>> to memref<1x32x128xf16>
          %23 = arith.muli %19, %c512 : index
          %24 = loom.alloc [1, 32, 128] on @L1 : memref<1x32x128xf16>
          %25 = loom.semaphore_take %24 : memref<1x32x128xf16> -> memref<1x32x128xf16>
          linalg.fill ins(%cst : f16) outs(%25 : memref<1x32x128xf16>)
          %26 = loom.alloc [1, 32, 1] on @L1 : memref<1x32x1xf16>
          %27 = loom.semaphore_take %26 : memref<1x32x1xf16> -> memref<1x32x1xf16>
          %28 = loom.semaphore_take %26 : memref<1x32x1xf16> -> memref<1x32x1xf16>
          %29 = loom.semaphore_take %26 : memref<1x32x1xf16> -> memref<1x32x1xf16>
          linalg.fill ins(%cst_0 : f16) outs(%29 : memref<1x32x1xf16>)
          %30 = loom.alloc [1, 32, 1] on @L1 : memref<1x32x1xf16>
          %31 = loom.semaphore_take %30 : memref<1x32x1xf16> -> memref<1x32x1xf16>
          linalg.fill ins(%cst_1 : f16) outs(%31 : memref<1x32x1xf16>)
          %32 = loom.alloc [1, 32, 128] on @L1 : memref<1x32x128xf16>
          %33 = loom.semaphore_take %32 : memref<1x32x128xf16> -> memref<1x32x128xf16>
          %34 = loom.semaphore_take %32 : memref<1x32x128xf16> -> memref<1x32x128xf16>
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
          %47 = loom.alloc [1, 32, 512] on @L1 : memref<1x32x512xf16>
          %48 = loom.semaphore_take %47 : memref<1x32x512xf16> -> memref<1x32x512xf16>
          %49 = loom.alloc [1, 512, 128] on @L1 : memref<1x512x128xf16>
          %50 = loom.semaphore_take %49 : memref<1x512x128xf16> -> memref<1x512x128xf16>
          %51 = arith.muli %arg6, %c1048576 overflow<nsw> : index
          %52 = arith.addi %51, %23 : index
          %reinterpret_cast_3 = memref.reinterpret_cast %arg0 to offset: [%52], sizes: [1, 128, 512], strides: [1048576, 8192, 1] : memref<16x128x8192xf16> to memref<1x128x512xf16, strided<[1048576, 8192, 1], offset: ?>>
          loom.copy %reinterpret_cast_3, %44 src_mem_space @mem_DRAM dst_mem_space @mem_L1, area : [1, 1] region : (UL : [%arg4, %arg5], LR : [%arg4, %arg5]) : memref<1x128x512xf16, strided<[1048576, 8192, 1], offset: ?>> to memref<1x128x512xf16>
          loom.batch_matmul ins(%21, %44 : memref<1x32x128xf16>, memref<1x128x512xf16>) outs(%46 : memref<1x32x512xf16>)
          loom.semaphore_give %44 : memref<1x128x512xf16>
          linalg.fill ins(%cst_1 : f16) outs(%38 : memref<1x32x1xf16>)
          linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, 0)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%46 : memref<1x32x512xf16>) outs(%38 : memref<1x32x1xf16>) {
          ^bb0(%in: f16, %out: f16):
            %73 = arith.maximumf %in, %out : f16
            linalg.yield %73 : f16
          }
          linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%31, %38 : memref<1x32x1xf16>, memref<1x32x1xf16>) outs(%38 : memref<1x32x1xf16>) {
          ^bb0(%in: f16, %in_5: f16, %out: f16):
            %73 = arith.mulf %in_5, %cst_2 : f16
            %74 = arith.cmpf ogt, %in, %73 : f16
            %75 = arith.select %74, %in, %73 : f16
            linalg.yield %75 : f16
          }
          %53 = loom.broadcast ins(%38 : memref<1x32x1xf16>) outs(%48 : memref<1x32x512xf16>) dim(2) -> memref<1x32x512xf16, strided<[?, ?, ?], offset: ?>>
          linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%46, %53 : memref<1x32x512xf16>, memref<1x32x512xf16, strided<[?, ?, ?], offset: ?>>) outs(%46 : memref<1x32x512xf16>) {
          ^bb0(%in: f16, %in_5: f16, %out: f16):
            %73 = arith.mulf %in, %cst_2 : f16
            %74 = arith.subf %73, %in_5 : f16
            %75 = math.exp %74 : f16
            linalg.yield %75 : f16
          }
          loom.semaphore_give %48 : memref<1x32x512xf16>
          linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, 0)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%46 : memref<1x32x512xf16>) outs(%40 : memref<1x32x1xf16>) {
          ^bb0(%in: f16, %out: f16):
            %73 = arith.addf %in, %out : f16
            linalg.yield %73 : f16
          }
          linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%31, %38 : memref<1x32x1xf16>, memref<1x32x1xf16>) outs(%42 : memref<1x32x1xf16>) {
          ^bb0(%in: f16, %in_5: f16, %out: f16):
            %73 = arith.subf %in, %in_5 : f16
            %74 = math.exp %73 : f16
            linalg.yield %74 : f16
          }
          linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%29, %42, %40 : memref<1x32x1xf16>, memref<1x32x1xf16>, memref<1x32x1xf16>) outs(%29 : memref<1x32x1xf16>) {
          ^bb0(%in: f16, %in_5: f16, %in_6: f16, %out: f16):
            %73 = arith.mulf %in, %in_5 : f16
            %74 = arith.addf %73, %in_6 : f16
            linalg.yield %74 : f16
          }
          loom.semaphore_give %40 : memref<1x32x1xf16>
          %54 = loom.broadcast ins(%42 : memref<1x32x1xf16>) outs(%34 : memref<1x32x128xf16>) dim(2) -> memref<1x32x128xf16, strided<[?, ?, ?], offset: ?>>
          loom.semaphore_give %42 : memref<1x32x1xf16>
          %55 = arith.muli %19, %c65536 : index
          %56 = arith.addi %51, %55 : index
          %reinterpret_cast_4 = memref.reinterpret_cast %arg1 to offset: [%56], sizes: [1, 512, 128], strides: [1048576, 128, 1] : memref<16x8192x128xf16> to memref<1x512x128xf16, strided<[1048576, 128, 1], offset: ?>>
          loom.copy %reinterpret_cast_4, %50 src_mem_space @mem_DRAM dst_mem_space @mem_L1, area : [1, 1] region : (UL : [%arg4, %arg5], LR : [%arg4, %arg5]) : memref<1x512x128xf16, strided<[1048576, 128, 1], offset: ?>> to memref<1x512x128xf16>
          loom.batch_matmul ins(%46, %50 : memref<1x32x512xf16>, memref<1x512x128xf16>) outs(%36 : memref<1x32x128xf16>)
          loom.semaphore_give %50 : memref<1x512x128xf16>
          loom.semaphore_give %46 : memref<1x32x512xf16>
          linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%36, %25, %54 : memref<1x32x128xf16>, memref<1x32x128xf16>, memref<1x32x128xf16, strided<[?, ?, ?], offset: ?>>) outs(%25 : memref<1x32x128xf16>) {
          ^bb0(%in: f16, %in_5: f16, %in_6: f16, %out: f16):
            %73 = arith.mulf %in_5, %in_6 : f16
            %74 = arith.addf %in, %73 : f16
            linalg.yield %74 : f16
          }
          loom.semaphore_give %36 : memref<1x32x128xf16>
          loom.semaphore_give %34 : memref<1x32x128xf16>
          linalg.copy ins(%38 : memref<1x32x1xf16>) outs(%31 : memref<1x32x1xf16>)
          loom.semaphore_give %38 : memref<1x32x1xf16>
          loom.semaphore_give %21 : memref<1x32x128xf16>
          %57 = loom.alloc [1, 32, 1] on @L1 : memref<1x32x1xf16>
          %58 = loom.semaphore_take %57 : memref<1x32x1xf16> -> memref<1x32x1xf16>
          linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%29, %31 : memref<1x32x1xf16>, memref<1x32x1xf16>) outs(%58 : memref<1x32x1xf16>) {
          ^bb0(%in: f16, %in_5: f16, %out: f16):
            %73 = math.log %in : f16
            %74 = arith.addf %73, %in_5 : f16
            linalg.yield %74 : f16
          }
          loom.semaphore_give %31 : memref<1x32x1xf16>
          %59 = loom.broadcast ins(%29 : memref<1x32x1xf16>) outs(%33 : memref<1x32x128xf16>) dim(2) -> memref<1x32x128xf16, strided<[?, ?, ?], offset: ?>>
          loom.semaphore_give %29 : memref<1x32x1xf16>
          %60 = loom.alloc [1, 32, 128] on @L1 : memref<1x32x128xf16>
          %61 = loom.semaphore_take %60 : memref<1x32x128xf16> -> memref<1x32x128xf16>
          linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%25, %59 : memref<1x32x128xf16>, memref<1x32x128xf16, strided<[?, ?, ?], offset: ?>>) outs(%61 : memref<1x32x128xf16>) {
          ^bb0(%in: f16, %in_5: f16, %out: f16):
            %73 = arith.divf %in, %in_5 : f16
            linalg.yield %73 : f16
          }
          loom.semaphore_give %33 : memref<1x32x128xf16>
          loom.semaphore_give %25 : memref<1x32x128xf16>
          %62 = loom.alloc [16, 1, 32, 1] on @L1 : memref<16x1x32x1xf16>
          %63 = loom.semaphore_take %62 : memref<16x1x32x1xf16> -> memref<16x1x32x1xf16>
          loom.gather %58, %63 src_mem_space @mem_array_L1 dst_mem_space @mem_array_L1 across(%19 : index), area : [8, 8] region : (UL : [%c0, %c0], LR : [%c7, %c7]) : memref<1x32x1xf16> to memref<16x1x32x1xf16>
          loom.semaphore_give %58 : memref<1x32x1xf16>
          %64 = loom.alloc [16, 1, 32, 128] on @L1 : memref<16x1x32x128xf16>
          %65 = loom.semaphore_take %64 : memref<16x1x32x128xf16> -> memref<16x1x32x128xf16>
          loom.gather %61, %65 src_mem_space @mem_array_L1 dst_mem_space @mem_array_L1 across(%19 : index), area : [8, 8] region : (UL : [%c0, %c0], LR : [%c7, %c7]) : memref<1x32x128xf16> to memref<16x1x32x128xf16>
          loom.semaphore_give %61 : memref<1x32x128xf16>
          %66 = arith.cmpi eq, %19, %c0 : index
          %67 = loom.alloc [1, 32, 128] on @L1 : memref<1x32x128xf16>
          %68 = loom.semaphore_take %67 : memref<1x32x128xf16> -> memref<1x32x128xf16>
          %69 = loom.alloc [16, 1, 32, 1] on @L1 : memref<16x1x32x1xf16>
          %70 = loom.semaphore_take %69 : memref<16x1x32x1xf16> -> memref<16x1x32x1xf16>
          %71 = loom.alloc [16, 1, 32, 128] on @L1 : memref<16x1x32x128xf16>
          %72 = loom.semaphore_take %71 : memref<16x1x32x128xf16> -> memref<16x1x32x128xf16>
          scf.if %66 {
            linalg.fill ins(%cst_1 : f16) outs(%28 : memref<1x32x1xf16>)
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1, d2, d3)>], iterator_types = ["reduction", "parallel", "parallel", "parallel"]} ins(%63 : memref<16x1x32x1xf16>) outs(%28 : memref<1x32x1xf16>) {
            ^bb0(%in: f16, %out: f16):
              %74 = arith.maximumf %in, %out : f16
              linalg.yield %74 : f16
            }
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%63, %28 : memref<16x1x32x1xf16>, memref<1x32x1xf16>) outs(%70 : memref<16x1x32x1xf16>) {
            ^bb0(%in: f16, %in_6: f16, %out: f16):
              %74 = arith.subf %in, %in_6 : f16
              %75 = math.exp %74 : f16
              linalg.yield %75 : f16
            }
            loom.semaphore_give %63 : memref<16x1x32x1xf16>
            loom.semaphore_give %28 : memref<1x32x1xf16>
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1, d2, d3)>], iterator_types = ["reduction", "parallel", "parallel", "parallel"]} ins(%70 : memref<16x1x32x1xf16>) outs(%27 : memref<1x32x1xf16>) {
            ^bb0(%in: f16, %out: f16):
              %74 = arith.addf %in, %out : f16
              linalg.yield %74 : f16
            }
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%70, %27 : memref<16x1x32x1xf16>, memref<1x32x1xf16>) outs(%70 : memref<16x1x32x1xf16>) {
            ^bb0(%in: f16, %in_6: f16, %out: f16):
              %74 = arith.divf %in, %in_6 : f16
              linalg.yield %74 : f16
            }
            loom.semaphore_give %27 : memref<1x32x1xf16>
            %alloc = memref.alloc() {alignment = 64 : i64} : memref<16x1x32x128xf16>
            %73 = loom.broadcast ins(%70 : memref<16x1x32x1xf16>) outs(%alloc : memref<16x1x32x128xf16>) dim(3) -> memref<16x1x32x128xf16, strided<[?, ?, ?, ?], offset: ?>>
            loom.semaphore_give %70 : memref<16x1x32x1xf16>
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%65, %73 : memref<16x1x32x128xf16>, memref<16x1x32x128xf16, strided<[?, ?, ?, ?], offset: ?>>) outs(%72 : memref<16x1x32x128xf16>) {
            ^bb0(%in: f16, %in_6: f16, %out: f16):
              %74 = arith.mulf %in, %in_6 : f16
              linalg.yield %74 : f16
            }
            loom.semaphore_give %65 : memref<16x1x32x128xf16>
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1, d2, d3)>], iterator_types = ["reduction", "parallel", "parallel", "parallel"]} ins(%72 : memref<16x1x32x128xf16>) outs(%68 : memref<1x32x128xf16>) {
            ^bb0(%in: f16, %out: f16):
              %74 = arith.addf %in, %out : f16
              linalg.yield %74 : f16
            }
            loom.semaphore_give %72 : memref<16x1x32x128xf16>
            %reinterpret_cast_5 = memref.reinterpret_cast %arg3 to offset: [%22], sizes: [1, 32, 128], strides: [4096, 128, 1] : memref<16x32x128xf16> to memref<1x32x128xf16, strided<[4096, 128, 1], offset: ?>>
            loom.copy %68, %reinterpret_cast_5 src_mem_space @mem_L1 dst_mem_space @mem_DRAM, area : [1, 1] region : (UL : [%arg4, %arg5], LR : [%arg4, %arg5]) : memref<1x32x128xf16> to memref<1x32x128xf16, strided<[4096, 128, 1], offset: ?>>
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
            %18 = arith.muli %arg6, %c8 overflow<nsw> : index
            %19 = arith.addi %arg4, %18 : index
            %20 = arith.muli %arg7, %c8 overflow<nsw> : index
            %21 = arith.addi %arg5, %20 : index
            %22 = loom.alloc [1, 32, 128] on @L1 : memref<1x32x128xf16>
            %23 = loom.semaphore_take %22 : memref<1x32x128xf16> -> memref<1x32x128xf16>
            %24 = arith.muli %19, %c4096 overflow<nsw> : index
            %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%24], sizes: [1, 32, 128], strides: [4096, 128, 1] : memref<16x32x128xf16> to memref<1x32x128xf16, strided<[4096, 128, 1], offset: ?>>
            loom.copy %reinterpret_cast, %23 src_mem_space @mem_DRAM dst_mem_space @mem_L1, area : [1, 8] region : (UL : [%arg4, %c0], LR : [%arg4, %c7]) : memref<1x32x128xf16, strided<[4096, 128, 1], offset: ?>> to memref<1x32x128xf16>
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
            %36 = loom.semaphore_take %34 : memref<1x32x128xf16> -> memref<1x32x128xf16>
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
            %49 = loom.alloc [1, 32, 512] on @L1 : memref<1x32x512xf16>
            %50 = loom.semaphore_take %49 : memref<1x32x512xf16> -> memref<1x32x512xf16>
            %51 = loom.alloc [1, 512, 128] on @L1 : memref<1x512x128xf16>
            %52 = loom.semaphore_take %51 : memref<1x512x128xf16> -> memref<1x512x128xf16>
            %53 = arith.muli %19, %c1048576 overflow<nsw> : index
            %54 = arith.addi %53, %25 : index
            %reinterpret_cast_3 = memref.reinterpret_cast %arg0 to offset: [%54], sizes: [1, 128, 512], strides: [1048576, 8192, 1] : memref<16x128x8192xf16> to memref<1x128x512xf16, strided<[1048576, 8192, 1], offset: ?>>
            loom.copy %reinterpret_cast_3, %46 src_mem_space @mem_DRAM dst_mem_space @mem_L1, area : [1, 1] region : (UL : [%arg4, %arg5], LR : [%arg4, %arg5]) : memref<1x128x512xf16, strided<[1048576, 8192, 1], offset: ?>> to memref<1x128x512xf16>
            loom.batch_matmul ins(%23, %46 : memref<1x32x128xf16>, memref<1x128x512xf16>) outs(%48 : memref<1x32x512xf16>)
            loom.semaphore_give %46 : memref<1x128x512xf16>
            linalg.fill ins(%cst_1 : f16) outs(%40 : memref<1x32x1xf16>)
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, 0)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%48 : memref<1x32x512xf16>) outs(%40 : memref<1x32x1xf16>) {
            ^bb0(%in: f16, %out: f16):
              %75 = arith.maximumf %in, %out : f16
              linalg.yield %75 : f16
            }
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%33, %40 : memref<1x32x1xf16>, memref<1x32x1xf16>) outs(%40 : memref<1x32x1xf16>) {
            ^bb0(%in: f16, %in_5: f16, %out: f16):
              %75 = arith.mulf %in_5, %cst_2 : f16
              %76 = arith.cmpf ogt, %in, %75 : f16
              %77 = arith.select %76, %in, %75 : f16
              linalg.yield %77 : f16
            }
            %55 = loom.broadcast ins(%40 : memref<1x32x1xf16>) outs(%50 : memref<1x32x512xf16>) dim(2) -> memref<1x32x512xf16, strided<[?, ?, ?], offset: ?>>
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%48, %55 : memref<1x32x512xf16>, memref<1x32x512xf16, strided<[?, ?, ?], offset: ?>>) outs(%48 : memref<1x32x512xf16>) {
            ^bb0(%in: f16, %in_5: f16, %out: f16):
              %75 = arith.mulf %in, %cst_2 : f16
              %76 = arith.subf %75, %in_5 : f16
              %77 = math.exp %76 : f16
              linalg.yield %77 : f16
            }
            loom.semaphore_give %50 : memref<1x32x512xf16>
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, 0)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%48 : memref<1x32x512xf16>) outs(%42 : memref<1x32x1xf16>) {
            ^bb0(%in: f16, %out: f16):
              %75 = arith.addf %in, %out : f16
              linalg.yield %75 : f16
            }
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%33, %40 : memref<1x32x1xf16>, memref<1x32x1xf16>) outs(%44 : memref<1x32x1xf16>) {
            ^bb0(%in: f16, %in_5: f16, %out: f16):
              %75 = arith.subf %in, %in_5 : f16
              %76 = math.exp %75 : f16
              linalg.yield %76 : f16
            }
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%31, %44, %42 : memref<1x32x1xf16>, memref<1x32x1xf16>, memref<1x32x1xf16>) outs(%31 : memref<1x32x1xf16>) {
            ^bb0(%in: f16, %in_5: f16, %in_6: f16, %out: f16):
              %75 = arith.mulf %in, %in_5 : f16
              %76 = arith.addf %75, %in_6 : f16
              linalg.yield %76 : f16
            }
            loom.semaphore_give %42 : memref<1x32x1xf16>
            %56 = loom.broadcast ins(%44 : memref<1x32x1xf16>) outs(%36 : memref<1x32x128xf16>) dim(2) -> memref<1x32x128xf16, strided<[?, ?, ?], offset: ?>>
            loom.semaphore_give %44 : memref<1x32x1xf16>
            %57 = arith.muli %21, %c65536 : index
            %58 = arith.addi %53, %57 : index
            %reinterpret_cast_4 = memref.reinterpret_cast %arg1 to offset: [%58], sizes: [1, 512, 128], strides: [1048576, 128, 1] : memref<16x8192x128xf16> to memref<1x512x128xf16, strided<[1048576, 128, 1], offset: ?>>
            loom.copy %reinterpret_cast_4, %52 src_mem_space @mem_DRAM dst_mem_space @mem_L1, area : [1, 1] region : (UL : [%arg4, %arg5], LR : [%arg4, %arg5]) : memref<1x512x128xf16, strided<[1048576, 128, 1], offset: ?>> to memref<1x512x128xf16>
            loom.batch_matmul ins(%48, %52 : memref<1x32x512xf16>, memref<1x512x128xf16>) outs(%38 : memref<1x32x128xf16>)
            loom.semaphore_give %52 : memref<1x512x128xf16>
            loom.semaphore_give %48 : memref<1x32x512xf16>
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%38, %27, %56 : memref<1x32x128xf16>, memref<1x32x128xf16>, memref<1x32x128xf16, strided<[?, ?, ?], offset: ?>>) outs(%27 : memref<1x32x128xf16>) {
            ^bb0(%in: f16, %in_5: f16, %in_6: f16, %out: f16):
              %75 = arith.mulf %in_5, %in_6 : f16
              %76 = arith.addf %in, %75 : f16
              linalg.yield %76 : f16
            }
            loom.semaphore_give %38 : memref<1x32x128xf16>
            loom.semaphore_give %36 : memref<1x32x128xf16>
            linalg.copy ins(%40 : memref<1x32x1xf16>) outs(%33 : memref<1x32x1xf16>)
            loom.semaphore_give %40 : memref<1x32x1xf16>
            loom.semaphore_give %23 : memref<1x32x128xf16>
            %59 = loom.alloc [1, 32, 1] on @L1 : memref<1x32x1xf16>
            %60 = loom.semaphore_take %59 : memref<1x32x1xf16> -> memref<1x32x1xf16>
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%31, %33 : memref<1x32x1xf16>, memref<1x32x1xf16>) outs(%60 : memref<1x32x1xf16>) {
            ^bb0(%in: f16, %in_5: f16, %out: f16):
              %75 = math.log %in : f16
              %76 = arith.addf %75, %in_5 : f16
              linalg.yield %76 : f16
            }
            loom.semaphore_give %33 : memref<1x32x1xf16>
            %61 = loom.broadcast ins(%31 : memref<1x32x1xf16>) outs(%35 : memref<1x32x128xf16>) dim(2) -> memref<1x32x128xf16, strided<[?, ?, ?], offset: ?>>
            loom.semaphore_give %31 : memref<1x32x1xf16>
            %62 = loom.alloc [1, 32, 128] on @L1 : memref<1x32x128xf16>
            %63 = loom.semaphore_take %62 : memref<1x32x128xf16> -> memref<1x32x128xf16>
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%27, %61 : memref<1x32x128xf16>, memref<1x32x128xf16, strided<[?, ?, ?], offset: ?>>) outs(%63 : memref<1x32x128xf16>) {
            ^bb0(%in: f16, %in_5: f16, %out: f16):
              %75 = arith.divf %in, %in_5 : f16
              linalg.yield %75 : f16
            }
            loom.semaphore_give %35 : memref<1x32x128xf16>
            loom.semaphore_give %27 : memref<1x32x128xf16>
            %64 = loom.alloc [16, 1, 32, 1] on @L1 : memref<16x1x32x1xf16>
            %65 = loom.semaphore_take %64 : memref<16x1x32x1xf16> -> memref<16x1x32x1xf16>
            loom.gather %60, %65 src_mem_space @mem_array_L1 dst_mem_space @mem_array_L1 across(%21 : index), area : [1, 8] region : (UL : [%arg4, %c0], LR : [%arg4, %c7]) : memref<1x32x1xf16> to memref<16x1x32x1xf16>
            loom.semaphore_give %60 : memref<1x32x1xf16>
            %66 = loom.alloc [16, 1, 32, 128] on @L1 : memref<16x1x32x128xf16>
            %67 = loom.semaphore_take %66 : memref<16x1x32x128xf16> -> memref<16x1x32x128xf16>
            loom.gather %63, %67 src_mem_space @mem_array_L1 dst_mem_space @mem_array_L1 across(%21 : index), area : [1, 8] region : (UL : [%arg4, %c0], LR : [%arg4, %c7]) : memref<1x32x128xf16> to memref<16x1x32x128xf16>
            loom.semaphore_give %63 : memref<1x32x128xf16>
            %68 = arith.cmpi eq, %21, %c0 : index
            %69 = loom.alloc [1, 32, 128] on @L1 : memref<1x32x128xf16>
            %70 = loom.semaphore_take %69 : memref<1x32x128xf16> -> memref<1x32x128xf16>
            %71 = loom.alloc [16, 1, 32, 1] on @L1 : memref<16x1x32x1xf16>
            %72 = loom.semaphore_take %71 : memref<16x1x32x1xf16> -> memref<16x1x32x1xf16>
            %73 = loom.alloc [16, 1, 32, 128] on @L1 : memref<16x1x32x128xf16>
            %74 = loom.semaphore_take %73 : memref<16x1x32x128xf16> -> memref<16x1x32x128xf16>
            scf.if %68 {
              linalg.fill ins(%cst_1 : f16) outs(%30 : memref<1x32x1xf16>)
              linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1, d2, d3)>], iterator_types = ["reduction", "parallel", "parallel", "parallel"]} ins(%65 : memref<16x1x32x1xf16>) outs(%30 : memref<1x32x1xf16>) {
              ^bb0(%in: f16, %out: f16):
                %76 = arith.maximumf %in, %out : f16
                linalg.yield %76 : f16
              }
              linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%65, %30 : memref<16x1x32x1xf16>, memref<1x32x1xf16>) outs(%72 : memref<16x1x32x1xf16>) {
              ^bb0(%in: f16, %in_6: f16, %out: f16):
                %76 = arith.subf %in, %in_6 : f16
                %77 = math.exp %76 : f16
                linalg.yield %77 : f16
              }
              loom.semaphore_give %65 : memref<16x1x32x1xf16>
              loom.semaphore_give %30 : memref<1x32x1xf16>
              linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1, d2, d3)>], iterator_types = ["reduction", "parallel", "parallel", "parallel"]} ins(%72 : memref<16x1x32x1xf16>) outs(%29 : memref<1x32x1xf16>) {
              ^bb0(%in: f16, %out: f16):
                %76 = arith.addf %in, %out : f16
                linalg.yield %76 : f16
              }
              linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%72, %29 : memref<16x1x32x1xf16>, memref<1x32x1xf16>) outs(%72 : memref<16x1x32x1xf16>) {
              ^bb0(%in: f16, %in_6: f16, %out: f16):
                %76 = arith.divf %in, %in_6 : f16
                linalg.yield %76 : f16
              }
              loom.semaphore_give %29 : memref<1x32x1xf16>
              %alloc = memref.alloc() {alignment = 64 : i64} : memref<16x1x32x128xf16>
              %75 = loom.broadcast ins(%72 : memref<16x1x32x1xf16>) outs(%alloc : memref<16x1x32x128xf16>) dim(3) -> memref<16x1x32x128xf16, strided<[?, ?, ?, ?], offset: ?>>
              loom.semaphore_give %72 : memref<16x1x32x1xf16>
              linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%67, %75 : memref<16x1x32x128xf16>, memref<16x1x32x128xf16, strided<[?, ?, ?, ?], offset: ?>>) outs(%74 : memref<16x1x32x128xf16>) {
              ^bb0(%in: f16, %in_6: f16, %out: f16):
                %76 = arith.mulf %in, %in_6 : f16
                linalg.yield %76 : f16
              }
              loom.semaphore_give %67 : memref<16x1x32x128xf16>
              linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1, d2, d3)>], iterator_types = ["reduction", "parallel", "parallel", "parallel"]} ins(%74 : memref<16x1x32x128xf16>) outs(%70 : memref<1x32x128xf16>) {
              ^bb0(%in: f16, %out: f16):
                %76 = arith.addf %in, %out : f16
                linalg.yield %76 : f16
              }
              loom.semaphore_give %74 : memref<16x1x32x128xf16>
              %reinterpret_cast_5 = memref.reinterpret_cast %arg3 to offset: [%24], sizes: [1, 32, 128], strides: [4096, 128, 1] : memref<16x32x128xf16> to memref<1x32x128xf16, strided<[4096, 128, 1], offset: ?>>
              loom.copy %70, %reinterpret_cast_5 src_mem_space @mem_L1 dst_mem_space @mem_DRAM, area : [1, 1] region : (UL : [%arg4, %arg5], LR : [%arg4, %arg5]) : memref<1x32x128xf16> to memref<1x32x128xf16, strided<[4096, 128, 1], offset: ?>>
              loom.semaphore_give %70 : memref<1x32x128xf16>
            }
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
          %18 = arith.muli %arg7, %c4 overflow<nsw> : index
          %19 = arith.addi %arg4, %18 : index
          %20 = arith.muli %arg5, %c2 overflow<nsw> : index
          %21 = arith.addi %20, %arg6 : index
          %22 = loom.alloc [1, 32, 128] on @L1 : memref<1x32x128xf16>
          %23 = loom.semaphore_take %22 : memref<1x32x128xf16> -> memref<1x32x128xf16>
          %24 = arith.muli %19, %c4096 overflow<nsw> : index
          %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%24], sizes: [1, 32, 128], strides: [4096, 128, 1] : memref<16x32x128xf16> to memref<1x32x128xf16, strided<[4096, 128, 1], offset: ?>>
          %25 = arith.muli %arg4, %c2 : index
          %26 = arith.addi %25, %c1 : index
          loom.copy %reinterpret_cast, %23 src_mem_space @mem_DRAM dst_mem_space @mem_L1, area : [2, 8] region : (UL : [%25, %c0], LR : [%26, %c7]) : memref<1x32x128xf16, strided<[4096, 128, 1], offset: ?>> to memref<1x32x128xf16>
          %27 = arith.muli %21, %c512 : index
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
          %38 = loom.semaphore_take %36 : memref<1x32x128xf16> -> memref<1x32x128xf16>
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
          %51 = loom.alloc [1, 32, 512] on @L1 : memref<1x32x512xf16>
          %52 = loom.semaphore_take %51 : memref<1x32x512xf16> -> memref<1x32x512xf16>
          %53 = loom.alloc [1, 512, 128] on @L1 : memref<1x512x128xf16>
          %54 = loom.semaphore_take %53 : memref<1x512x128xf16> -> memref<1x512x128xf16>
          %55 = arith.muli %19, %c1048576 overflow<nsw> : index
          %56 = arith.addi %55, %27 : index
          %reinterpret_cast_3 = memref.reinterpret_cast %arg0 to offset: [%56], sizes: [1, 128, 512], strides: [1048576, 8192, 1] : memref<16x128x8192xf16> to memref<1x128x512xf16, strided<[1048576, 8192, 1], offset: ?>>
          %57 = arith.addi %arg5, %25 : index
          loom.copy %reinterpret_cast_3, %48 src_mem_space @mem_DRAM dst_mem_space @mem_L1, area : [1, 1] region : (UL : [%57, %arg6], LR : [%57, %arg6]) : memref<1x128x512xf16, strided<[1048576, 8192, 1], offset: ?>> to memref<1x128x512xf16>
          loom.batch_matmul ins(%23, %48 : memref<1x32x128xf16>, memref<1x128x512xf16>) outs(%50 : memref<1x32x512xf16>)
          loom.semaphore_give %48 : memref<1x128x512xf16>
          linalg.fill ins(%cst_1 : f16) outs(%42 : memref<1x32x1xf16>)
          linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, 0)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%50 : memref<1x32x512xf16>) outs(%42 : memref<1x32x1xf16>) {
          ^bb0(%in: f16, %out: f16):
            %78 = arith.maximumf %in, %out : f16
            linalg.yield %78 : f16
          }
          linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%35, %42 : memref<1x32x1xf16>, memref<1x32x1xf16>) outs(%42 : memref<1x32x1xf16>) {
          ^bb0(%in: f16, %in_5: f16, %out: f16):
            %78 = arith.mulf %in_5, %cst_2 : f16
            %79 = arith.cmpf ogt, %in, %78 : f16
            %80 = arith.select %79, %in, %78 : f16
            linalg.yield %80 : f16
          }
          %58 = loom.broadcast ins(%42 : memref<1x32x1xf16>) outs(%52 : memref<1x32x512xf16>) dim(2) -> memref<1x32x512xf16, strided<[?, ?, ?], offset: ?>>
          linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%50, %58 : memref<1x32x512xf16>, memref<1x32x512xf16, strided<[?, ?, ?], offset: ?>>) outs(%50 : memref<1x32x512xf16>) {
          ^bb0(%in: f16, %in_5: f16, %out: f16):
            %78 = arith.mulf %in, %cst_2 : f16
            %79 = arith.subf %78, %in_5 : f16
            %80 = math.exp %79 : f16
            linalg.yield %80 : f16
          }
          loom.semaphore_give %52 : memref<1x32x512xf16>
          linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, 0)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%50 : memref<1x32x512xf16>) outs(%44 : memref<1x32x1xf16>) {
          ^bb0(%in: f16, %out: f16):
            %78 = arith.addf %in, %out : f16
            linalg.yield %78 : f16
          }
          linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%35, %42 : memref<1x32x1xf16>, memref<1x32x1xf16>) outs(%46 : memref<1x32x1xf16>) {
          ^bb0(%in: f16, %in_5: f16, %out: f16):
            %78 = arith.subf %in, %in_5 : f16
            %79 = math.exp %78 : f16
            linalg.yield %79 : f16
          }
          linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%33, %46, %44 : memref<1x32x1xf16>, memref<1x32x1xf16>, memref<1x32x1xf16>) outs(%33 : memref<1x32x1xf16>) {
          ^bb0(%in: f16, %in_5: f16, %in_6: f16, %out: f16):
            %78 = arith.mulf %in, %in_5 : f16
            %79 = arith.addf %78, %in_6 : f16
            linalg.yield %79 : f16
          }
          loom.semaphore_give %44 : memref<1x32x1xf16>
          %59 = loom.broadcast ins(%46 : memref<1x32x1xf16>) outs(%38 : memref<1x32x128xf16>) dim(2) -> memref<1x32x128xf16, strided<[?, ?, ?], offset: ?>>
          loom.semaphore_give %46 : memref<1x32x1xf16>
          %60 = arith.muli %21, %c65536 : index
          %61 = arith.addi %55, %60 : index
          %reinterpret_cast_4 = memref.reinterpret_cast %arg1 to offset: [%61], sizes: [1, 512, 128], strides: [1048576, 128, 1] : memref<16x8192x128xf16> to memref<1x512x128xf16, strided<[1048576, 128, 1], offset: ?>>
          loom.copy %reinterpret_cast_4, %54 src_mem_space @mem_DRAM dst_mem_space @mem_L1, area : [1, 1] region : (UL : [%57, %arg6], LR : [%57, %arg6]) : memref<1x512x128xf16, strided<[1048576, 128, 1], offset: ?>> to memref<1x512x128xf16>
          loom.batch_matmul ins(%50, %54 : memref<1x32x512xf16>, memref<1x512x128xf16>) outs(%40 : memref<1x32x128xf16>)
          loom.semaphore_give %54 : memref<1x512x128xf16>
          loom.semaphore_give %50 : memref<1x32x512xf16>
          linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%40, %29, %59 : memref<1x32x128xf16>, memref<1x32x128xf16>, memref<1x32x128xf16, strided<[?, ?, ?], offset: ?>>) outs(%29 : memref<1x32x128xf16>) {
          ^bb0(%in: f16, %in_5: f16, %in_6: f16, %out: f16):
            %78 = arith.mulf %in_5, %in_6 : f16
            %79 = arith.addf %in, %78 : f16
            linalg.yield %79 : f16
          }
          loom.semaphore_give %40 : memref<1x32x128xf16>
          loom.semaphore_give %38 : memref<1x32x128xf16>
          linalg.copy ins(%42 : memref<1x32x1xf16>) outs(%35 : memref<1x32x1xf16>)
          loom.semaphore_give %42 : memref<1x32x1xf16>
          loom.semaphore_give %23 : memref<1x32x128xf16>
          %62 = loom.alloc [1, 32, 1] on @L1 : memref<1x32x1xf16>
          %63 = loom.semaphore_take %62 : memref<1x32x1xf16> -> memref<1x32x1xf16>
          linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%33, %35 : memref<1x32x1xf16>, memref<1x32x1xf16>) outs(%63 : memref<1x32x1xf16>) {
          ^bb0(%in: f16, %in_5: f16, %out: f16):
            %78 = math.log %in : f16
            %79 = arith.addf %78, %in_5 : f16
            linalg.yield %79 : f16
          }
          loom.semaphore_give %35 : memref<1x32x1xf16>
          %64 = loom.broadcast ins(%33 : memref<1x32x1xf16>) outs(%37 : memref<1x32x128xf16>) dim(2) -> memref<1x32x128xf16, strided<[?, ?, ?], offset: ?>>
          loom.semaphore_give %33 : memref<1x32x1xf16>
          %65 = loom.alloc [1, 32, 128] on @L1 : memref<1x32x128xf16>
          %66 = loom.semaphore_take %65 : memref<1x32x128xf16> -> memref<1x32x128xf16>
          linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%29, %64 : memref<1x32x128xf16>, memref<1x32x128xf16, strided<[?, ?, ?], offset: ?>>) outs(%66 : memref<1x32x128xf16>) {
          ^bb0(%in: f16, %in_5: f16, %out: f16):
            %78 = arith.divf %in, %in_5 : f16
            linalg.yield %78 : f16
          }
          loom.semaphore_give %37 : memref<1x32x128xf16>
          loom.semaphore_give %29 : memref<1x32x128xf16>
          %67 = loom.alloc [16, 1, 32, 1] on @L1 : memref<16x1x32x1xf16>
          %68 = loom.semaphore_take %67 : memref<16x1x32x1xf16> -> memref<16x1x32x1xf16>
          loom.gather %63, %68 src_mem_space @mem_array_L1 dst_mem_space @mem_array_L1 across(%21 : index), area : [2, 8] region : (UL : [%25, %c0], LR : [%26, %c7]) : memref<1x32x1xf16> to memref<16x1x32x1xf16>
          loom.semaphore_give %63 : memref<1x32x1xf16>
          %69 = loom.alloc [16, 1, 32, 128] on @L1 : memref<16x1x32x128xf16>
          %70 = loom.semaphore_take %69 : memref<16x1x32x128xf16> -> memref<16x1x32x128xf16>
          loom.gather %66, %70 src_mem_space @mem_array_L1 dst_mem_space @mem_array_L1 across(%21 : index), area : [2, 8] region : (UL : [%25, %c0], LR : [%26, %c7]) : memref<1x32x128xf16> to memref<16x1x32x128xf16>
          loom.semaphore_give %66 : memref<1x32x128xf16>
          %71 = arith.cmpi eq, %21, %c0 : index
          %72 = loom.alloc [1, 32, 128] on @L1 : memref<1x32x128xf16>
          %73 = loom.semaphore_take %72 : memref<1x32x128xf16> -> memref<1x32x128xf16>
          %74 = loom.alloc [16, 1, 32, 1] on @L1 : memref<16x1x32x1xf16>
          %75 = loom.semaphore_take %74 : memref<16x1x32x1xf16> -> memref<16x1x32x1xf16>
          %76 = loom.alloc [16, 1, 32, 128] on @L1 : memref<16x1x32x128xf16>
          %77 = loom.semaphore_take %76 : memref<16x1x32x128xf16> -> memref<16x1x32x128xf16>
          scf.if %71 {
            linalg.fill ins(%cst_1 : f16) outs(%32 : memref<1x32x1xf16>)
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1, d2, d3)>], iterator_types = ["reduction", "parallel", "parallel", "parallel"]} ins(%68 : memref<16x1x32x1xf16>) outs(%32 : memref<1x32x1xf16>) {
            ^bb0(%in: f16, %out: f16):
              %79 = arith.maximumf %in, %out : f16
              linalg.yield %79 : f16
            }
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%68, %32 : memref<16x1x32x1xf16>, memref<1x32x1xf16>) outs(%75 : memref<16x1x32x1xf16>) {
            ^bb0(%in: f16, %in_6: f16, %out: f16):
              %79 = arith.subf %in, %in_6 : f16
              %80 = math.exp %79 : f16
              linalg.yield %80 : f16
            }
            loom.semaphore_give %68 : memref<16x1x32x1xf16>
            loom.semaphore_give %32 : memref<1x32x1xf16>
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1, d2, d3)>], iterator_types = ["reduction", "parallel", "parallel", "parallel"]} ins(%75 : memref<16x1x32x1xf16>) outs(%31 : memref<1x32x1xf16>) {
            ^bb0(%in: f16, %out: f16):
              %79 = arith.addf %in, %out : f16
              linalg.yield %79 : f16
            }
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%75, %31 : memref<16x1x32x1xf16>, memref<1x32x1xf16>) outs(%75 : memref<16x1x32x1xf16>) {
            ^bb0(%in: f16, %in_6: f16, %out: f16):
              %79 = arith.divf %in, %in_6 : f16
              linalg.yield %79 : f16
            }
            loom.semaphore_give %31 : memref<1x32x1xf16>
            %alloc = memref.alloc() {alignment = 64 : i64} : memref<16x1x32x128xf16>
            %78 = loom.broadcast ins(%75 : memref<16x1x32x1xf16>) outs(%alloc : memref<16x1x32x128xf16>) dim(3) -> memref<16x1x32x128xf16, strided<[?, ?, ?, ?], offset: ?>>
            loom.semaphore_give %75 : memref<16x1x32x1xf16>
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%70, %78 : memref<16x1x32x128xf16>, memref<16x1x32x128xf16, strided<[?, ?, ?, ?], offset: ?>>) outs(%77 : memref<16x1x32x128xf16>) {
            ^bb0(%in: f16, %in_6: f16, %out: f16):
              %79 = arith.mulf %in, %in_6 : f16
              linalg.yield %79 : f16
            }
            loom.semaphore_give %70 : memref<16x1x32x128xf16>
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1, d2, d3)>], iterator_types = ["reduction", "parallel", "parallel", "parallel"]} ins(%77 : memref<16x1x32x128xf16>) outs(%73 : memref<1x32x128xf16>) {
            ^bb0(%in: f16, %out: f16):
              %79 = arith.addf %in, %out : f16
              linalg.yield %79 : f16
            }
            loom.semaphore_give %77 : memref<16x1x32x128xf16>
            %reinterpret_cast_5 = memref.reinterpret_cast %arg3 to offset: [%24], sizes: [1, 32, 128], strides: [4096, 128, 1] : memref<16x32x128xf16> to memref<1x32x128xf16, strided<[4096, 128, 1], offset: ?>>
            loom.copy %73, %reinterpret_cast_5 src_mem_space @mem_L1 dst_mem_space @mem_DRAM, area : [1, 1] region : (UL : [%57, %arg6], LR : [%57, %arg6]) : memref<1x32x128xf16> to memref<1x32x128xf16, strided<[4096, 128, 1], offset: ?>>
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
          %18 = arith.muli %arg7, %c2 overflow<nsw> : index
          %19 = arith.addi %arg4, %18 : index
          %20 = arith.muli %arg5, %c4 overflow<nsw> : index
          %21 = arith.addi %20, %arg6 : index
          %22 = loom.alloc [1, 32, 128] on @L1 : memref<1x32x128xf16>
          %23 = loom.semaphore_take %22 : memref<1x32x128xf16> -> memref<1x32x128xf16>
          %24 = arith.muli %19, %c4096 overflow<nsw> : index
          %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%24], sizes: [1, 32, 128], strides: [4096, 128, 1] : memref<16x32x128xf16> to memref<1x32x128xf16, strided<[4096, 128, 1], offset: ?>>
          %25 = arith.muli %arg4, %c4 : index
          %26 = arith.addi %25, %c3 : index
          loom.copy %reinterpret_cast, %23 src_mem_space @mem_DRAM dst_mem_space @mem_L1, area : [4, 8] region : (UL : [%25, %c0], LR : [%26, %c7]) : memref<1x32x128xf16, strided<[4096, 128, 1], offset: ?>> to memref<1x32x128xf16>
          %27 = arith.muli %21, %c512 : index
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
          %38 = loom.semaphore_take %36 : memref<1x32x128xf16> -> memref<1x32x128xf16>
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
          %51 = loom.alloc [1, 32, 512] on @L1 : memref<1x32x512xf16>
          %52 = loom.semaphore_take %51 : memref<1x32x512xf16> -> memref<1x32x512xf16>
          %53 = loom.alloc [1, 512, 128] on @L1 : memref<1x512x128xf16>
          %54 = loom.semaphore_take %53 : memref<1x512x128xf16> -> memref<1x512x128xf16>
          %55 = arith.muli %19, %c1048576 overflow<nsw> : index
          %56 = arith.addi %55, %27 : index
          %reinterpret_cast_3 = memref.reinterpret_cast %arg0 to offset: [%56], sizes: [1, 128, 512], strides: [1048576, 8192, 1] : memref<16x128x8192xf16> to memref<1x128x512xf16, strided<[1048576, 8192, 1], offset: ?>>
          %57 = arith.addi %arg5, %25 : index
          loom.copy %reinterpret_cast_3, %48 src_mem_space @mem_DRAM dst_mem_space @mem_L1, area : [1, 1] region : (UL : [%57, %arg6], LR : [%57, %arg6]) : memref<1x128x512xf16, strided<[1048576, 8192, 1], offset: ?>> to memref<1x128x512xf16>
          loom.batch_matmul ins(%23, %48 : memref<1x32x128xf16>, memref<1x128x512xf16>) outs(%50 : memref<1x32x512xf16>)
          loom.semaphore_give %48 : memref<1x128x512xf16>
          linalg.fill ins(%cst_1 : f16) outs(%42 : memref<1x32x1xf16>)
          linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, 0)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%50 : memref<1x32x512xf16>) outs(%42 : memref<1x32x1xf16>) {
          ^bb0(%in: f16, %out: f16):
            %78 = arith.maximumf %in, %out : f16
            linalg.yield %78 : f16
          }
          linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%35, %42 : memref<1x32x1xf16>, memref<1x32x1xf16>) outs(%42 : memref<1x32x1xf16>) {
          ^bb0(%in: f16, %in_5: f16, %out: f16):
            %78 = arith.mulf %in_5, %cst_2 : f16
            %79 = arith.cmpf ogt, %in, %78 : f16
            %80 = arith.select %79, %in, %78 : f16
            linalg.yield %80 : f16
          }
          %58 = loom.broadcast ins(%42 : memref<1x32x1xf16>) outs(%52 : memref<1x32x512xf16>) dim(2) -> memref<1x32x512xf16, strided<[?, ?, ?], offset: ?>>
          linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%50, %58 : memref<1x32x512xf16>, memref<1x32x512xf16, strided<[?, ?, ?], offset: ?>>) outs(%50 : memref<1x32x512xf16>) {
          ^bb0(%in: f16, %in_5: f16, %out: f16):
            %78 = arith.mulf %in, %cst_2 : f16
            %79 = arith.subf %78, %in_5 : f16
            %80 = math.exp %79 : f16
            linalg.yield %80 : f16
          }
          loom.semaphore_give %52 : memref<1x32x512xf16>
          linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, 0)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%50 : memref<1x32x512xf16>) outs(%44 : memref<1x32x1xf16>) {
          ^bb0(%in: f16, %out: f16):
            %78 = arith.addf %in, %out : f16
            linalg.yield %78 : f16
          }
          linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%35, %42 : memref<1x32x1xf16>, memref<1x32x1xf16>) outs(%46 : memref<1x32x1xf16>) {
          ^bb0(%in: f16, %in_5: f16, %out: f16):
            %78 = arith.subf %in, %in_5 : f16
            %79 = math.exp %78 : f16
            linalg.yield %79 : f16
          }
          linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%33, %46, %44 : memref<1x32x1xf16>, memref<1x32x1xf16>, memref<1x32x1xf16>) outs(%33 : memref<1x32x1xf16>) {
          ^bb0(%in: f16, %in_5: f16, %in_6: f16, %out: f16):
            %78 = arith.mulf %in, %in_5 : f16
            %79 = arith.addf %78, %in_6 : f16
            linalg.yield %79 : f16
          }
          loom.semaphore_give %44 : memref<1x32x1xf16>
          %59 = loom.broadcast ins(%46 : memref<1x32x1xf16>) outs(%38 : memref<1x32x128xf16>) dim(2) -> memref<1x32x128xf16, strided<[?, ?, ?], offset: ?>>
          loom.semaphore_give %46 : memref<1x32x1xf16>
          %60 = arith.muli %21, %c65536 : index
          %61 = arith.addi %55, %60 : index
          %reinterpret_cast_4 = memref.reinterpret_cast %arg1 to offset: [%61], sizes: [1, 512, 128], strides: [1048576, 128, 1] : memref<16x8192x128xf16> to memref<1x512x128xf16, strided<[1048576, 128, 1], offset: ?>>
          loom.copy %reinterpret_cast_4, %54 src_mem_space @mem_DRAM dst_mem_space @mem_L1, area : [1, 1] region : (UL : [%57, %arg6], LR : [%57, %arg6]) : memref<1x512x128xf16, strided<[1048576, 128, 1], offset: ?>> to memref<1x512x128xf16>
          loom.batch_matmul ins(%50, %54 : memref<1x32x512xf16>, memref<1x512x128xf16>) outs(%40 : memref<1x32x128xf16>)
          loom.semaphore_give %54 : memref<1x512x128xf16>
          loom.semaphore_give %50 : memref<1x32x512xf16>
          linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%40, %29, %59 : memref<1x32x128xf16>, memref<1x32x128xf16>, memref<1x32x128xf16, strided<[?, ?, ?], offset: ?>>) outs(%29 : memref<1x32x128xf16>) {
          ^bb0(%in: f16, %in_5: f16, %in_6: f16, %out: f16):
            %78 = arith.mulf %in_5, %in_6 : f16
            %79 = arith.addf %in, %78 : f16
            linalg.yield %79 : f16
          }
          loom.semaphore_give %40 : memref<1x32x128xf16>
          loom.semaphore_give %38 : memref<1x32x128xf16>
          linalg.copy ins(%42 : memref<1x32x1xf16>) outs(%35 : memref<1x32x1xf16>)
          loom.semaphore_give %42 : memref<1x32x1xf16>
          loom.semaphore_give %23 : memref<1x32x128xf16>
          %62 = loom.alloc [1, 32, 1] on @L1 : memref<1x32x1xf16>
          %63 = loom.semaphore_take %62 : memref<1x32x1xf16> -> memref<1x32x1xf16>
          linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%33, %35 : memref<1x32x1xf16>, memref<1x32x1xf16>) outs(%63 : memref<1x32x1xf16>) {
          ^bb0(%in: f16, %in_5: f16, %out: f16):
            %78 = math.log %in : f16
            %79 = arith.addf %78, %in_5 : f16
            linalg.yield %79 : f16
          }
          loom.semaphore_give %35 : memref<1x32x1xf16>
          %64 = loom.broadcast ins(%33 : memref<1x32x1xf16>) outs(%37 : memref<1x32x128xf16>) dim(2) -> memref<1x32x128xf16, strided<[?, ?, ?], offset: ?>>
          loom.semaphore_give %33 : memref<1x32x1xf16>
          %65 = loom.alloc [1, 32, 128] on @L1 : memref<1x32x128xf16>
          %66 = loom.semaphore_take %65 : memref<1x32x128xf16> -> memref<1x32x128xf16>
          linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%29, %64 : memref<1x32x128xf16>, memref<1x32x128xf16, strided<[?, ?, ?], offset: ?>>) outs(%66 : memref<1x32x128xf16>) {
          ^bb0(%in: f16, %in_5: f16, %out: f16):
            %78 = arith.divf %in, %in_5 : f16
            linalg.yield %78 : f16
          }
          loom.semaphore_give %37 : memref<1x32x128xf16>
          loom.semaphore_give %29 : memref<1x32x128xf16>
          %67 = loom.alloc [16, 1, 32, 1] on @L1 : memref<16x1x32x1xf16>
          %68 = loom.semaphore_take %67 : memref<16x1x32x1xf16> -> memref<16x1x32x1xf16>
          loom.gather %63, %68 src_mem_space @mem_array_L1 dst_mem_space @mem_array_L1 across(%21 : index), area : [4, 8] region : (UL : [%25, %c0], LR : [%26, %c7]) : memref<1x32x1xf16> to memref<16x1x32x1xf16>
          loom.semaphore_give %63 : memref<1x32x1xf16>
          %69 = loom.alloc [16, 1, 32, 128] on @L1 : memref<16x1x32x128xf16>
          %70 = loom.semaphore_take %69 : memref<16x1x32x128xf16> -> memref<16x1x32x128xf16>
          loom.gather %66, %70 src_mem_space @mem_array_L1 dst_mem_space @mem_array_L1 across(%21 : index), area : [4, 8] region : (UL : [%25, %c0], LR : [%26, %c7]) : memref<1x32x128xf16> to memref<16x1x32x128xf16>
          loom.semaphore_give %66 : memref<1x32x128xf16>
          %71 = arith.cmpi eq, %21, %c0 : index
          %72 = loom.alloc [1, 32, 128] on @L1 : memref<1x32x128xf16>
          %73 = loom.semaphore_take %72 : memref<1x32x128xf16> -> memref<1x32x128xf16>
          %74 = loom.alloc [16, 1, 32, 1] on @L1 : memref<16x1x32x1xf16>
          %75 = loom.semaphore_take %74 : memref<16x1x32x1xf16> -> memref<16x1x32x1xf16>
          %76 = loom.alloc [16, 1, 32, 128] on @L1 : memref<16x1x32x128xf16>
          %77 = loom.semaphore_take %76 : memref<16x1x32x128xf16> -> memref<16x1x32x128xf16>
          scf.if %71 {
            linalg.fill ins(%cst_1 : f16) outs(%32 : memref<1x32x1xf16>)
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1, d2, d3)>], iterator_types = ["reduction", "parallel", "parallel", "parallel"]} ins(%68 : memref<16x1x32x1xf16>) outs(%32 : memref<1x32x1xf16>) {
            ^bb0(%in: f16, %out: f16):
              %79 = arith.maximumf %in, %out : f16
              linalg.yield %79 : f16
            }
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%68, %32 : memref<16x1x32x1xf16>, memref<1x32x1xf16>) outs(%75 : memref<16x1x32x1xf16>) {
            ^bb0(%in: f16, %in_6: f16, %out: f16):
              %79 = arith.subf %in, %in_6 : f16
              %80 = math.exp %79 : f16
              linalg.yield %80 : f16
            }
            loom.semaphore_give %68 : memref<16x1x32x1xf16>
            loom.semaphore_give %32 : memref<1x32x1xf16>
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1, d2, d3)>], iterator_types = ["reduction", "parallel", "parallel", "parallel"]} ins(%75 : memref<16x1x32x1xf16>) outs(%31 : memref<1x32x1xf16>) {
            ^bb0(%in: f16, %out: f16):
              %79 = arith.addf %in, %out : f16
              linalg.yield %79 : f16
            }
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%75, %31 : memref<16x1x32x1xf16>, memref<1x32x1xf16>) outs(%75 : memref<16x1x32x1xf16>) {
            ^bb0(%in: f16, %in_6: f16, %out: f16):
              %79 = arith.divf %in, %in_6 : f16
              linalg.yield %79 : f16
            }
            loom.semaphore_give %31 : memref<1x32x1xf16>
            %alloc = memref.alloc() {alignment = 64 : i64} : memref<16x1x32x128xf16>
            %78 = loom.broadcast ins(%75 : memref<16x1x32x1xf16>) outs(%alloc : memref<16x1x32x128xf16>) dim(3) -> memref<16x1x32x128xf16, strided<[?, ?, ?, ?], offset: ?>>
            loom.semaphore_give %75 : memref<16x1x32x1xf16>
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%70, %78 : memref<16x1x32x128xf16>, memref<16x1x32x128xf16, strided<[?, ?, ?, ?], offset: ?>>) outs(%77 : memref<16x1x32x128xf16>) {
            ^bb0(%in: f16, %in_6: f16, %out: f16):
              %79 = arith.mulf %in, %in_6 : f16
              linalg.yield %79 : f16
            }
            loom.semaphore_give %70 : memref<16x1x32x128xf16>
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1, d2, d3)>], iterator_types = ["reduction", "parallel", "parallel", "parallel"]} ins(%77 : memref<16x1x32x128xf16>) outs(%73 : memref<1x32x128xf16>) {
            ^bb0(%in: f16, %out: f16):
              %79 = arith.addf %in, %out : f16
              linalg.yield %79 : f16
            }
            loom.semaphore_give %77 : memref<16x1x32x128xf16>
            %reinterpret_cast_5 = memref.reinterpret_cast %arg3 to offset: [%24], sizes: [1, 32, 128], strides: [4096, 128, 1] : memref<16x32x128xf16> to memref<1x32x128xf16, strided<[4096, 128, 1], offset: ?>>
            loom.copy %73, %reinterpret_cast_5 src_mem_space @mem_L1 dst_mem_space @mem_DRAM, area : [1, 1] region : (UL : [%57, %arg6], LR : [%57, %arg6]) : memref<1x32x128xf16> to memref<1x32x128xf16, strided<[4096, 128, 1], offset: ?>>
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
          %18 = arith.muli %arg4, %c8 overflow<nsw> : index
          %19 = arith.addi %18, %arg5 : index
          %20 = loom.alloc [1, 32, 128] on @L1 : memref<1x32x128xf16>
          %21 = loom.semaphore_take %20 : memref<1x32x128xf16> -> memref<1x32x128xf16>
          %22 = arith.muli %arg6, %c4096 overflow<nsw> : index
          %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%22], sizes: [1, 32, 128], strides: [4096, 128, 1] : memref<16x32x128xf16> to memref<1x32x128xf16, strided<[4096, 128, 1], offset: ?>>
          loom.copy %reinterpret_cast, %21 src_mem_space @mem_DRAM dst_mem_space @mem_L1, area : [8, 8] region : (UL : [%c0, %c0], LR : [%c7, %c7]) : memref<1x32x128xf16, strided<[4096, 128, 1], offset: ?>> to memref<1x32x128xf16>
          %23 = arith.muli %19, %c512 : index
          %24 = loom.alloc [1, 32, 128] on @L1 : memref<1x32x128xf16>
          %25 = loom.semaphore_take %24 : memref<1x32x128xf16> -> memref<1x32x128xf16>
          linalg.fill ins(%cst : f16) outs(%25 : memref<1x32x128xf16>)
          %26 = loom.alloc [1, 32, 1] on @L1 : memref<1x32x1xf16>
          %27 = loom.semaphore_take %26 : memref<1x32x1xf16> -> memref<1x32x1xf16>
          %28 = loom.semaphore_take %26 : memref<1x32x1xf16> -> memref<1x32x1xf16>
          %29 = loom.semaphore_take %26 : memref<1x32x1xf16> -> memref<1x32x1xf16>
          linalg.fill ins(%cst_0 : f16) outs(%29 : memref<1x32x1xf16>)
          %30 = loom.alloc [1, 32, 1] on @L1 : memref<1x32x1xf16>
          %31 = loom.semaphore_take %30 : memref<1x32x1xf16> -> memref<1x32x1xf16>
          linalg.fill ins(%cst_1 : f16) outs(%31 : memref<1x32x1xf16>)
          %32 = loom.alloc [1, 32, 128] on @L1 : memref<1x32x128xf16>
          %33 = loom.semaphore_take %32 : memref<1x32x128xf16> -> memref<1x32x128xf16>
          %34 = loom.semaphore_take %32 : memref<1x32x128xf16> -> memref<1x32x128xf16>
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
          %47 = loom.alloc [1, 32, 512] on @L1 : memref<1x32x512xf16>
          %48 = loom.semaphore_take %47 : memref<1x32x512xf16> -> memref<1x32x512xf16>
          %49 = loom.alloc [1, 512, 128] on @L1 : memref<1x512x128xf16>
          %50 = loom.semaphore_take %49 : memref<1x512x128xf16> -> memref<1x512x128xf16>
          %51 = arith.muli %arg6, %c1048576 overflow<nsw> : index
          %52 = arith.addi %51, %23 : index
          %reinterpret_cast_3 = memref.reinterpret_cast %arg0 to offset: [%52], sizes: [1, 128, 512], strides: [1048576, 8192, 1] : memref<16x128x8192xf16> to memref<1x128x512xf16, strided<[1048576, 8192, 1], offset: ?>>
          loom.copy %reinterpret_cast_3, %44 src_mem_space @mem_DRAM dst_mem_space @mem_L1, area : [1, 1] region : (UL : [%arg4, %arg5], LR : [%arg4, %arg5]) : memref<1x128x512xf16, strided<[1048576, 8192, 1], offset: ?>> to memref<1x128x512xf16>
          loom.batch_matmul ins(%21, %44 : memref<1x32x128xf16>, memref<1x128x512xf16>) outs(%46 : memref<1x32x512xf16>)
          loom.semaphore_give %44 : memref<1x128x512xf16>
          linalg.fill ins(%cst_1 : f16) outs(%38 : memref<1x32x1xf16>)
          linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, 0)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%46 : memref<1x32x512xf16>) outs(%38 : memref<1x32x1xf16>) {
          ^bb0(%in: f16, %out: f16):
            %73 = arith.maximumf %in, %out : f16
            linalg.yield %73 : f16
          }
          linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%31, %38 : memref<1x32x1xf16>, memref<1x32x1xf16>) outs(%38 : memref<1x32x1xf16>) {
          ^bb0(%in: f16, %in_5: f16, %out: f16):
            %73 = arith.mulf %in_5, %cst_2 : f16
            %74 = arith.cmpf ogt, %in, %73 : f16
            %75 = arith.select %74, %in, %73 : f16
            linalg.yield %75 : f16
          }
          %53 = loom.broadcast ins(%38 : memref<1x32x1xf16>) outs(%48 : memref<1x32x512xf16>) dim(2) -> memref<1x32x512xf16, strided<[?, ?, ?], offset: ?>>
          linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%46, %53 : memref<1x32x512xf16>, memref<1x32x512xf16, strided<[?, ?, ?], offset: ?>>) outs(%46 : memref<1x32x512xf16>) {
          ^bb0(%in: f16, %in_5: f16, %out: f16):
            %73 = arith.mulf %in, %cst_2 : f16
            %74 = arith.subf %73, %in_5 : f16
            %75 = math.exp %74 : f16
            linalg.yield %75 : f16
          }
          loom.semaphore_give %48 : memref<1x32x512xf16>
          linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, 0)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%46 : memref<1x32x512xf16>) outs(%40 : memref<1x32x1xf16>) {
          ^bb0(%in: f16, %out: f16):
            %73 = arith.addf %in, %out : f16
            linalg.yield %73 : f16
          }
          linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%31, %38 : memref<1x32x1xf16>, memref<1x32x1xf16>) outs(%42 : memref<1x32x1xf16>) {
          ^bb0(%in: f16, %in_5: f16, %out: f16):
            %73 = arith.subf %in, %in_5 : f16
            %74 = math.exp %73 : f16
            linalg.yield %74 : f16
          }
          linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%29, %42, %40 : memref<1x32x1xf16>, memref<1x32x1xf16>, memref<1x32x1xf16>) outs(%29 : memref<1x32x1xf16>) {
          ^bb0(%in: f16, %in_5: f16, %in_6: f16, %out: f16):
            %73 = arith.mulf %in, %in_5 : f16
            %74 = arith.addf %73, %in_6 : f16
            linalg.yield %74 : f16
          }
          loom.semaphore_give %40 : memref<1x32x1xf16>
          %54 = loom.broadcast ins(%42 : memref<1x32x1xf16>) outs(%34 : memref<1x32x128xf16>) dim(2) -> memref<1x32x128xf16, strided<[?, ?, ?], offset: ?>>
          loom.semaphore_give %42 : memref<1x32x1xf16>
          %55 = arith.muli %19, %c65536 : index
          %56 = arith.addi %51, %55 : index
          %reinterpret_cast_4 = memref.reinterpret_cast %arg1 to offset: [%56], sizes: [1, 512, 128], strides: [1048576, 128, 1] : memref<16x8192x128xf16> to memref<1x512x128xf16, strided<[1048576, 128, 1], offset: ?>>
          loom.copy %reinterpret_cast_4, %50 src_mem_space @mem_DRAM dst_mem_space @mem_L1, area : [1, 1] region : (UL : [%arg4, %arg5], LR : [%arg4, %arg5]) : memref<1x512x128xf16, strided<[1048576, 128, 1], offset: ?>> to memref<1x512x128xf16>
          loom.batch_matmul ins(%46, %50 : memref<1x32x512xf16>, memref<1x512x128xf16>) outs(%36 : memref<1x32x128xf16>)
          loom.semaphore_give %50 : memref<1x512x128xf16>
          loom.semaphore_give %46 : memref<1x32x512xf16>
          linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%36, %25, %54 : memref<1x32x128xf16>, memref<1x32x128xf16>, memref<1x32x128xf16, strided<[?, ?, ?], offset: ?>>) outs(%25 : memref<1x32x128xf16>) {
          ^bb0(%in: f16, %in_5: f16, %in_6: f16, %out: f16):
            %73 = arith.mulf %in_5, %in_6 : f16
            %74 = arith.addf %in, %73 : f16
            linalg.yield %74 : f16
          }
          loom.semaphore_give %36 : memref<1x32x128xf16>
          loom.semaphore_give %34 : memref<1x32x128xf16>
          linalg.copy ins(%38 : memref<1x32x1xf16>) outs(%31 : memref<1x32x1xf16>)
          loom.semaphore_give %38 : memref<1x32x1xf16>
          loom.semaphore_give %21 : memref<1x32x128xf16>
          %57 = loom.alloc [1, 32, 1] on @L1 : memref<1x32x1xf16>
          %58 = loom.semaphore_take %57 : memref<1x32x1xf16> -> memref<1x32x1xf16>
          linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%29, %31 : memref<1x32x1xf16>, memref<1x32x1xf16>) outs(%58 : memref<1x32x1xf16>) {
          ^bb0(%in: f16, %in_5: f16, %out: f16):
            %73 = math.log %in : f16
            %74 = arith.addf %73, %in_5 : f16
            linalg.yield %74 : f16
          }
          loom.semaphore_give %31 : memref<1x32x1xf16>
          %59 = loom.broadcast ins(%29 : memref<1x32x1xf16>) outs(%33 : memref<1x32x128xf16>) dim(2) -> memref<1x32x128xf16, strided<[?, ?, ?], offset: ?>>
          loom.semaphore_give %29 : memref<1x32x1xf16>
          %60 = loom.alloc [1, 32, 128] on @L1 : memref<1x32x128xf16>
          %61 = loom.semaphore_take %60 : memref<1x32x128xf16> -> memref<1x32x128xf16>
          linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%25, %59 : memref<1x32x128xf16>, memref<1x32x128xf16, strided<[?, ?, ?], offset: ?>>) outs(%61 : memref<1x32x128xf16>) {
          ^bb0(%in: f16, %in_5: f16, %out: f16):
            %73 = arith.divf %in, %in_5 : f16
            linalg.yield %73 : f16
          }
          loom.semaphore_give %33 : memref<1x32x128xf16>
          loom.semaphore_give %25 : memref<1x32x128xf16>
          %62 = loom.alloc [16, 1, 32, 1] on @L1 : memref<16x1x32x1xf16>
          %63 = loom.semaphore_take %62 : memref<16x1x32x1xf16> -> memref<16x1x32x1xf16>
          loom.gather %58, %63 src_mem_space @mem_array_L1 dst_mem_space @mem_array_L1 across(%19 : index), area : [8, 8] region : (UL : [%c0, %c0], LR : [%c7, %c7]) : memref<1x32x1xf16> to memref<16x1x32x1xf16>
          loom.semaphore_give %58 : memref<1x32x1xf16>
          %64 = loom.alloc [16, 1, 32, 128] on @L1 : memref<16x1x32x128xf16>
          %65 = loom.semaphore_take %64 : memref<16x1x32x128xf16> -> memref<16x1x32x128xf16>
          loom.gather %61, %65 src_mem_space @mem_array_L1 dst_mem_space @mem_array_L1 across(%19 : index), area : [8, 8] region : (UL : [%c0, %c0], LR : [%c7, %c7]) : memref<1x32x128xf16> to memref<16x1x32x128xf16>
          loom.semaphore_give %61 : memref<1x32x128xf16>
          %66 = arith.cmpi eq, %19, %c0 : index
          %67 = loom.alloc [1, 32, 128] on @L1 : memref<1x32x128xf16>
          %68 = loom.semaphore_take %67 : memref<1x32x128xf16> -> memref<1x32x128xf16>
          %69 = loom.alloc [16, 1, 32, 1] on @L1 : memref<16x1x32x1xf16>
          %70 = loom.semaphore_take %69 : memref<16x1x32x1xf16> -> memref<16x1x32x1xf16>
          %71 = loom.alloc [16, 1, 32, 128] on @L1 : memref<16x1x32x128xf16>
          %72 = loom.semaphore_take %71 : memref<16x1x32x128xf16> -> memref<16x1x32x128xf16>
          scf.if %66 {
            linalg.fill ins(%cst_1 : f16) outs(%28 : memref<1x32x1xf16>)
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1, d2, d3)>], iterator_types = ["reduction", "parallel", "parallel", "parallel"]} ins(%63 : memref<16x1x32x1xf16>) outs(%28 : memref<1x32x1xf16>) {
            ^bb0(%in: f16, %out: f16):
              %74 = arith.maximumf %in, %out : f16
              linalg.yield %74 : f16
            }
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%63, %28 : memref<16x1x32x1xf16>, memref<1x32x1xf16>) outs(%70 : memref<16x1x32x1xf16>) {
            ^bb0(%in: f16, %in_6: f16, %out: f16):
              %74 = arith.subf %in, %in_6 : f16
              %75 = math.exp %74 : f16
              linalg.yield %75 : f16
            }
            loom.semaphore_give %63 : memref<16x1x32x1xf16>
            loom.semaphore_give %28 : memref<1x32x1xf16>
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1, d2, d3)>], iterator_types = ["reduction", "parallel", "parallel", "parallel"]} ins(%70 : memref<16x1x32x1xf16>) outs(%27 : memref<1x32x1xf16>) {
            ^bb0(%in: f16, %out: f16):
              %74 = arith.addf %in, %out : f16
              linalg.yield %74 : f16
            }
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%70, %27 : memref<16x1x32x1xf16>, memref<1x32x1xf16>) outs(%70 : memref<16x1x32x1xf16>) {
            ^bb0(%in: f16, %in_6: f16, %out: f16):
              %74 = arith.divf %in, %in_6 : f16
              linalg.yield %74 : f16
            }
            loom.semaphore_give %27 : memref<1x32x1xf16>
            %alloc = memref.alloc() {alignment = 64 : i64} : memref<16x1x32x128xf16>
            %73 = loom.broadcast ins(%70 : memref<16x1x32x1xf16>) outs(%alloc : memref<16x1x32x128xf16>) dim(3) -> memref<16x1x32x128xf16, strided<[?, ?, ?, ?], offset: ?>>
            loom.semaphore_give %70 : memref<16x1x32x1xf16>
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%65, %73 : memref<16x1x32x128xf16>, memref<16x1x32x128xf16, strided<[?, ?, ?, ?], offset: ?>>) outs(%72 : memref<16x1x32x128xf16>) {
            ^bb0(%in: f16, %in_6: f16, %out: f16):
              %74 = arith.mulf %in, %in_6 : f16
              linalg.yield %74 : f16
            }
            loom.semaphore_give %65 : memref<16x1x32x128xf16>
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1, d2, d3)>], iterator_types = ["reduction", "parallel", "parallel", "parallel"]} ins(%72 : memref<16x1x32x128xf16>) outs(%68 : memref<1x32x128xf16>) {
            ^bb0(%in: f16, %out: f16):
              %74 = arith.addf %in, %out : f16
              linalg.yield %74 : f16
            }
            loom.semaphore_give %72 : memref<16x1x32x128xf16>
            %reinterpret_cast_5 = memref.reinterpret_cast %arg3 to offset: [%22], sizes: [1, 32, 128], strides: [4096, 128, 1] : memref<16x32x128xf16> to memref<1x32x128xf16, strided<[4096, 128, 1], offset: ?>>
            loom.copy %68, %reinterpret_cast_5 src_mem_space @mem_L1 dst_mem_space @mem_DRAM, area : [1, 1] region : (UL : [%arg4, %arg5], LR : [%arg4, %arg5]) : memref<1x32x128xf16> to memref<1x32x128xf16, strided<[4096, 128, 1], offset: ?>>
            loom.semaphore_give %68 : memref<1x32x128xf16>
          }
        } {loom.block_sym = @tile_b, loom.iter_type = #loom.iter_type<temporal>}
        scf.reduce 
      } {loom.block_syms = [@tile_s, @tile_s], loom.iter_types = [#loom.iter_type<spatial>, #loom.iter_type<spatial>], loom.logical_levels = [0, 0], loom.physical_dims = [@dim_x, @dim_y]}
      return
    }
  }
}
