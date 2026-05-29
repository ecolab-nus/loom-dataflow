module attributes {loom.tile_b = {is_reduction = false, upper_bound = 32 : index}, loom.tile_m = {is_reduction = false, upper_bound = 4096 : index}, loom.tile_n = {is_reduction = false, upper_bound = 4096 : index}} {
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
  module attributes {loom.pass_name = "Materialize", loom.tile_b = {is_reduction = false, upper_bound = 32 : index}, loom.tile_m = {is_reduction = false, upper_bound = 4096 : index}, loom.tile_n = {is_reduction = false, upper_bound = 4096 : index}} {
    func.func @attention__x8_y1y8__d0i1_d1i1_d2i0__f01__n_dim_x_level0_bc8_dim_x_level0_bc8_n__tile_b64__tile_m64__tile_n512(%arg0: memref<32x128x4096xf16>, %arg1: memref<32x4096x128xf16>, %arg2: memref<32x4096x128xf16>, %arg3: memref<32x4096x128xf16>) {
      %c65536 = arith.constant 65536 : index
      %c8192 = arith.constant 8192 : index
      %c33554432 = arith.constant 33554432 : index
      %c8 = arith.constant 8 : index
      %c7 = arith.constant 7 : index
      %c0 = arith.constant 0 : index
      %cst = arith.constant 0.000000e+00 : f16
      %cst_0 = arith.constant 1.000000e+00 : f16
      %cst_1 = arith.constant 0xFC00 : f16
      %c1 = arith.constant 1 : index
      %cst_2 = arith.constant 8.837890e-02 : f16
      %c512 = arith.constant 512 : index
      scf.parallel (%arg4, %arg5) = (%c0, %c0) to (%c8, %c8) step (%c1, %c1) {
        scf.for %arg6 = %c0 to %c8 step %c1 {
          %18 = arith.muli %arg6, %c8 overflow<nsw> : index
          %19 = arith.addi %arg5, %18 : index
          %20 = loom.alloc [64, 64, 128] on @L1 : memref<64x64x128xf16>
          %21 = loom.semaphore_take %20 : memref<64x64x128xf16> -> memref<64x64x128xf16>
          %22 = arith.muli %arg4, %c33554432 : index
          %23 = arith.muli %19, %c8192 : index
          %24 = arith.addi %22, %23 : index
          %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%24], sizes: [64, 64, 128], strides: [524288, 128, 1] : memref<32x4096x128xf16> to memref<64x64x128xf16, strided<[524288, 128, 1], offset: ?>>
          loom.copy %reinterpret_cast, %21 src_mem_space @mem_DRAM dst_mem_space @mem_L1, area : [1, 1] region : (UL : [%arg5, %arg4], LR : [%arg5, %arg4]) : memref<64x64x128xf16, strided<[524288, 128, 1], offset: ?>> to memref<64x64x128xf16>
          %25 = loom.alloc [64, 64, 128] on @L1 : memref<64x64x128xf16>
          %26 = loom.semaphore_take %25 : memref<64x64x128xf16> -> memref<64x64x128xf16>
          linalg.fill ins(%cst : f16) outs(%26 : memref<64x64x128xf16>)
          %27 = loom.alloc [64, 64, 1] on @L1 : memref<64x64x1xf16>
          %28 = loom.semaphore_take %27 : memref<64x64x1xf16> -> memref<64x64x1xf16>
          linalg.fill ins(%cst_0 : f16) outs(%28 : memref<64x64x1xf16>)
          %29 = loom.alloc [64, 64, 1] on @L1 : memref<64x64x1xf16>
          %30 = loom.semaphore_take %29 : memref<64x64x1xf16> -> memref<64x64x1xf16>
          linalg.fill ins(%cst_1 : f16) outs(%30 : memref<64x64x1xf16>)
          %31 = loom.alloc [64, 64, 128] on @L1 : memref<64x64x128xf16>
          %32 = loom.semaphore_take %31 : memref<64x64x128xf16> -> memref<64x64x128xf16>
          %33 = loom.semaphore_take %31 : memref<64x64x128xf16> -> memref<64x64x128xf16>
          %34 = loom.alloc [64, 64, 1] on @L1 : memref<64x64x1xf16>
          %35 = loom.semaphore_take %34 : memref<64x64x1xf16> -> memref<64x64x1xf16>
          %36 = loom.alloc [64, 64, 1] on @L1 : memref<64x64x1xf16>
          %37 = loom.semaphore_take %36 : memref<64x64x1xf16> -> memref<64x64x1xf16>
          %38 = loom.alloc [64, 64, 1] on @L1 : memref<64x64x1xf16>
          %39 = loom.semaphore_take %38 : memref<64x64x1xf16> -> memref<64x64x1xf16>
          %40 = loom.alloc [64, 128, 512] on @L1 : memref<64x128x512xf16>
          %41 = loom.semaphore_take %40 : memref<64x128x512xf16> -> memref<64x128x512xf16>
          %42 = loom.alloc [64, 64, 512] on @L1 : memref<64x64x512xf16>
          %43 = loom.semaphore_take %42 : memref<64x64x512xf16> -> memref<64x64x512xf16>
          %44 = loom.alloc [64, 64, 512] on @L1 : memref<64x64x512xf16>
          %45 = loom.semaphore_take %44 : memref<64x64x512xf16> -> memref<64x64x512xf16>
          %46 = loom.alloc [64, 512, 128] on @L1 : memref<64x512x128xf16>
          %47 = loom.semaphore_take %46 : memref<64x512x128xf16> -> memref<64x512x128xf16>
          scf.for %arg7 = %c0 to %c8 step %c1 {
            %51 = arith.muli %arg7, %c512 : index
            %52 = arith.addi %22, %51 : index
            %reinterpret_cast_4 = memref.reinterpret_cast %arg0 to offset: [%52], sizes: [64, 128, 512], strides: [524288, 4096, 1] : memref<32x128x4096xf16> to memref<64x128x512xf16, strided<[524288, 4096, 1], offset: ?>>
            loom.copy %reinterpret_cast_4, %41 src_mem_space @mem_DRAM dst_mem_space @mem_L1, area : [8, 1] region : (UL : [%c0, %arg4], LR : [%c7, %arg4]) : memref<64x128x512xf16, strided<[524288, 4096, 1], offset: ?>> to memref<64x128x512xf16>
            loom.batch_matmul ins(%21, %41 : memref<64x64x128xf16>, memref<64x128x512xf16>) outs(%43 : memref<64x64x512xf16>)
            loom.semaphore_give %41 : memref<64x128x512xf16>
            linalg.fill ins(%cst_1 : f16) outs(%35 : memref<64x64x1xf16>)
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, 0)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%43 : memref<64x64x512xf16>) outs(%35 : memref<64x64x1xf16>) {
            ^bb0(%in: f16, %out: f16):
              %56 = arith.maximumf %in, %out : f16
              linalg.yield %56 : f16
            }
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%30, %35 : memref<64x64x1xf16>, memref<64x64x1xf16>) outs(%35 : memref<64x64x1xf16>) {
            ^bb0(%in: f16, %in_6: f16, %out: f16):
              %56 = arith.mulf %in_6, %cst_2 : f16
              %57 = arith.cmpf ogt, %in, %56 : f16
              %58 = arith.select %57, %in, %56 : f16
              linalg.yield %58 : f16
            }
            %53 = loom.broadcast ins(%35 : memref<64x64x1xf16>) outs(%45 : memref<64x64x512xf16>) dim(2) -> memref<64x64x512xf16, strided<[?, ?, ?], offset: ?>>
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%43 : memref<64x64x512xf16>) outs(%43 : memref<64x64x512xf16>) {
            ^bb0(%in: f16, %out: f16):
              %56 = arith.mulf %in, %cst_2 : f16
              linalg.yield %56 : f16
            }
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%43, %53 : memref<64x64x512xf16>, memref<64x64x512xf16, strided<[?, ?, ?], offset: ?>>) outs(%43 : memref<64x64x512xf16>) {
            ^bb0(%in: f16, %in_6: f16, %out: f16):
              %56 = arith.subf %in, %in_6 : f16
              %57 = math.exp %56 : f16
              linalg.yield %57 : f16
            }
            loom.semaphore_give %45 : memref<64x64x512xf16>
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, 0)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%43 : memref<64x64x512xf16>) outs(%37 : memref<64x64x1xf16>) {
            ^bb0(%in: f16, %out: f16):
              %56 = arith.addf %in, %out : f16
              linalg.yield %56 : f16
            }
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%30, %35 : memref<64x64x1xf16>, memref<64x64x1xf16>) outs(%39 : memref<64x64x1xf16>) {
            ^bb0(%in: f16, %in_6: f16, %out: f16):
              %56 = arith.subf %in, %in_6 : f16
              %57 = math.exp %56 : f16
              linalg.yield %57 : f16
            }
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%28, %39 : memref<64x64x1xf16>, memref<64x64x1xf16>) outs(%28 : memref<64x64x1xf16>) {
            ^bb0(%in: f16, %in_6: f16, %out: f16):
              %56 = arith.mulf %in, %in_6 : f16
              linalg.yield %56 : f16
            }
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%28, %37 : memref<64x64x1xf16>, memref<64x64x1xf16>) outs(%28 : memref<64x64x1xf16>) {
            ^bb0(%in: f16, %in_6: f16, %out: f16):
              %56 = arith.addf %in, %in_6 : f16
              linalg.yield %56 : f16
            }
            loom.semaphore_give %37 : memref<64x64x1xf16>
            %54 = arith.muli %arg7, %c65536 : index
            %55 = arith.addi %22, %54 : index
            %reinterpret_cast_5 = memref.reinterpret_cast %arg1 to offset: [%55], sizes: [64, 512, 128], strides: [524288, 128, 1] : memref<32x4096x128xf16> to memref<64x512x128xf16, strided<[524288, 128, 1], offset: ?>>
            loom.copy %reinterpret_cast_5, %47 src_mem_space @mem_DRAM dst_mem_space @mem_L1, area : [8, 1] region : (UL : [%c0, %arg4], LR : [%c7, %arg4]) : memref<64x512x128xf16, strided<[524288, 128, 1], offset: ?>> to memref<64x512x128xf16>
            loom.batch_matmul ins(%43, %47 : memref<64x64x512xf16>, memref<64x512x128xf16>) outs(%33 : memref<64x64x128xf16>)
            loom.semaphore_give %47 : memref<64x512x128xf16>
            loom.semaphore_give %43 : memref<64x64x512xf16>
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, 0)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%26, %39 : memref<64x64x128xf16>, memref<64x64x1xf16>) outs(%26 : memref<64x64x128xf16>) {
            ^bb0(%in: f16, %in_6: f16, %out: f16):
              %56 = arith.mulf %in, %in_6 : f16
              linalg.yield %56 : f16
            }
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%26, %33 : memref<64x64x128xf16>, memref<64x64x128xf16>) outs(%26 : memref<64x64x128xf16>) {
            ^bb0(%in: f16, %in_6: f16, %out: f16):
              %56 = arith.addf %in_6, %in : f16
              linalg.yield %56 : f16
            }
            loom.semaphore_give %39 : memref<64x64x1xf16>
            loom.semaphore_give %33 : memref<64x64x128xf16>
            loom.copy %35, %30 src_mem_space @mem_L1 dst_mem_space @mem_L1, area : [1, 1] {reclaim = true} : memref<64x64x1xf16> to memref<64x64x1xf16>
            loom.semaphore_give %35 : memref<64x64x1xf16>
          }
          loom.semaphore_give %30 : memref<64x64x1xf16>
          loom.semaphore_give %21 : memref<64x64x128xf16>
          %48 = loom.broadcast ins(%28 : memref<64x64x1xf16>) outs(%32 : memref<64x64x128xf16>) dim(2) -> memref<64x64x128xf16, strided<[?, ?, ?], offset: ?>>
          loom.semaphore_give %28 : memref<64x64x1xf16>
          %49 = loom.alloc [64, 64, 128] on @L1 : memref<64x64x128xf16>
          %50 = loom.semaphore_take %49 : memref<64x64x128xf16> -> memref<64x64x128xf16>
          linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%26, %48 : memref<64x64x128xf16>, memref<64x64x128xf16, strided<[?, ?, ?], offset: ?>>) outs(%50 : memref<64x64x128xf16>) {
          ^bb0(%in: f16, %in_4: f16, %out: f16):
            %51 = arith.divf %in, %in_4 : f16
            linalg.yield %51 : f16
          }
          loom.semaphore_give %32 : memref<64x64x128xf16>
          loom.semaphore_give %26 : memref<64x64x128xf16>
          %reinterpret_cast_3 = memref.reinterpret_cast %arg3 to offset: [%24], sizes: [64, 64, 128], strides: [524288, 128, 1] : memref<32x4096x128xf16> to memref<64x64x128xf16, strided<[524288, 128, 1], offset: ?>>
          loom.copy %50, %reinterpret_cast_3 src_mem_space @mem_L1 dst_mem_space @mem_DRAM, area : [1, 1] region : (UL : [%arg5, %arg4], LR : [%arg5, %arg4]) : memref<64x64x128xf16> to memref<64x64x128xf16, strided<[524288, 128, 1], offset: ?>>
          loom.semaphore_give %50 : memref<64x64x128xf16>
        } {loom.block_sym = @tile_m, loom.iter_type = #loom.iter_type<temporal>}
        scf.reduce 
      } {loom.block_syms = [@tile_b, @tile_m], loom.iter_types = [#loom.iter_type<spatial>, #loom.iter_type<spatial>], loom.logical_levels = [1, 0], loom.physical_dims = [@dim_y, @dim_x]}
      return
    }
  }
  module attributes {loom.pass_name = "Materialize", loom.tile_b = {is_reduction = false, upper_bound = 32 : index}, loom.tile_m = {is_reduction = false, upper_bound = 4096 : index}, loom.tile_n = {is_reduction = false, upper_bound = 4096 : index}} {
    func.func @attention__x8_y2y4__d0i1_d1i1_d2i0__f01__n_dim_x_level0_bc8_dim_y_level0_bc2_dim_x_level0_bc8_dim_y_level0_bc2_n__tile_b64__tile_m64__tile_n512(%arg0: memref<32x128x4096xf16>, %arg1: memref<32x4096x128xf16>, %arg2: memref<32x4096x128xf16>, %arg3: memref<32x4096x128xf16>) {
      %c65536 = arith.constant 65536 : index
      %c8192 = arith.constant 8192 : index
      %c33554432 = arith.constant 33554432 : index
      %c16 = arith.constant 16 : index
      %c8 = arith.constant 8 : index
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
      scf.parallel (%arg4, %arg5, %arg6) = (%c0, %c0, %c0) to (%c4, %c8, %c2) step (%c1, %c1, %c1) {
        scf.for %arg7 = %c0 to %c4 step %c1 {
          %18 = arith.muli %arg5, %c8 overflow<nsw> : index
          %19 = arith.addi %18, %arg6 : index
          %20 = arith.muli %arg7, %c16 overflow<nsw> : index
          %21 = arith.addi %19, %20 : index
          %22 = loom.alloc [64, 64, 128] on @L1 : memref<64x64x128xf16>
          %23 = loom.semaphore_take %22 : memref<64x64x128xf16> -> memref<64x64x128xf16>
          %24 = arith.muli %arg4, %c33554432 : index
          %25 = arith.muli %21, %c8192 : index
          %26 = arith.addi %24, %25 : index
          %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%26], sizes: [64, 64, 128], strides: [524288, 128, 1] : memref<32x4096x128xf16> to memref<64x64x128xf16, strided<[524288, 128, 1], offset: ?>>
          %27 = arith.muli %arg4, %c2 : index
          %28 = arith.addi %arg6, %27 : index
          loom.copy %reinterpret_cast, %23 src_mem_space @mem_DRAM dst_mem_space @mem_L1, area : [1, 1] region : (UL : [%arg5, %28], LR : [%arg5, %28]) : memref<64x64x128xf16, strided<[524288, 128, 1], offset: ?>> to memref<64x64x128xf16>
          %29 = loom.alloc [64, 64, 128] on @L1 : memref<64x64x128xf16>
          %30 = loom.semaphore_take %29 : memref<64x64x128xf16> -> memref<64x64x128xf16>
          linalg.fill ins(%cst : f16) outs(%30 : memref<64x64x128xf16>)
          %31 = loom.alloc [64, 64, 1] on @L1 : memref<64x64x1xf16>
          %32 = loom.semaphore_take %31 : memref<64x64x1xf16> -> memref<64x64x1xf16>
          linalg.fill ins(%cst_0 : f16) outs(%32 : memref<64x64x1xf16>)
          %33 = loom.alloc [64, 64, 1] on @L1 : memref<64x64x1xf16>
          %34 = loom.semaphore_take %33 : memref<64x64x1xf16> -> memref<64x64x1xf16>
          linalg.fill ins(%cst_1 : f16) outs(%34 : memref<64x64x1xf16>)
          %35 = loom.alloc [64, 64, 128] on @L1 : memref<64x64x128xf16>
          %36 = loom.semaphore_take %35 : memref<64x64x128xf16> -> memref<64x64x128xf16>
          %37 = loom.semaphore_take %35 : memref<64x64x128xf16> -> memref<64x64x128xf16>
          %38 = loom.alloc [64, 64, 1] on @L1 : memref<64x64x1xf16>
          %39 = loom.semaphore_take %38 : memref<64x64x1xf16> -> memref<64x64x1xf16>
          %40 = loom.alloc [64, 64, 1] on @L1 : memref<64x64x1xf16>
          %41 = loom.semaphore_take %40 : memref<64x64x1xf16> -> memref<64x64x1xf16>
          %42 = loom.alloc [64, 64, 1] on @L1 : memref<64x64x1xf16>
          %43 = loom.semaphore_take %42 : memref<64x64x1xf16> -> memref<64x64x1xf16>
          %44 = loom.alloc [64, 128, 512] on @L1 : memref<64x128x512xf16>
          %45 = loom.semaphore_take %44 : memref<64x128x512xf16> -> memref<64x128x512xf16>
          %46 = loom.alloc [64, 64, 512] on @L1 : memref<64x64x512xf16>
          %47 = loom.semaphore_take %46 : memref<64x64x512xf16> -> memref<64x64x512xf16>
          %48 = loom.alloc [64, 64, 512] on @L1 : memref<64x64x512xf16>
          %49 = loom.semaphore_take %48 : memref<64x64x512xf16> -> memref<64x64x512xf16>
          %50 = loom.alloc [64, 512, 128] on @L1 : memref<64x512x128xf16>
          %51 = loom.semaphore_take %50 : memref<64x512x128xf16> -> memref<64x512x128xf16>
          scf.for %arg8 = %c0 to %c8 step %c1 {
            %55 = arith.muli %arg8, %c512 : index
            %56 = arith.addi %24, %55 : index
            %reinterpret_cast_4 = memref.reinterpret_cast %arg0 to offset: [%56], sizes: [64, 128, 512], strides: [524288, 4096, 1] : memref<32x128x4096xf16> to memref<64x128x512xf16, strided<[524288, 4096, 1], offset: ?>>
            %57 = arith.addi %27, %c1 : index
            loom.copy %reinterpret_cast_4, %45 src_mem_space @mem_DRAM dst_mem_space @mem_L1, area : [8, 2] region : (UL : [%c0, %27], LR : [%c7, %57]) : memref<64x128x512xf16, strided<[524288, 4096, 1], offset: ?>> to memref<64x128x512xf16>
            loom.batch_matmul ins(%23, %45 : memref<64x64x128xf16>, memref<64x128x512xf16>) outs(%47 : memref<64x64x512xf16>)
            loom.semaphore_give %45 : memref<64x128x512xf16>
            linalg.fill ins(%cst_1 : f16) outs(%39 : memref<64x64x1xf16>)
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, 0)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%47 : memref<64x64x512xf16>) outs(%39 : memref<64x64x1xf16>) {
            ^bb0(%in: f16, %out: f16):
              %61 = arith.maximumf %in, %out : f16
              linalg.yield %61 : f16
            }
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%34, %39 : memref<64x64x1xf16>, memref<64x64x1xf16>) outs(%39 : memref<64x64x1xf16>) {
            ^bb0(%in: f16, %in_6: f16, %out: f16):
              %61 = arith.mulf %in_6, %cst_2 : f16
              %62 = arith.cmpf ogt, %in, %61 : f16
              %63 = arith.select %62, %in, %61 : f16
              linalg.yield %63 : f16
            }
            %58 = loom.broadcast ins(%39 : memref<64x64x1xf16>) outs(%49 : memref<64x64x512xf16>) dim(2) -> memref<64x64x512xf16, strided<[?, ?, ?], offset: ?>>
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%47 : memref<64x64x512xf16>) outs(%47 : memref<64x64x512xf16>) {
            ^bb0(%in: f16, %out: f16):
              %61 = arith.mulf %in, %cst_2 : f16
              linalg.yield %61 : f16
            }
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%47, %58 : memref<64x64x512xf16>, memref<64x64x512xf16, strided<[?, ?, ?], offset: ?>>) outs(%47 : memref<64x64x512xf16>) {
            ^bb0(%in: f16, %in_6: f16, %out: f16):
              %61 = arith.subf %in, %in_6 : f16
              %62 = math.exp %61 : f16
              linalg.yield %62 : f16
            }
            loom.semaphore_give %49 : memref<64x64x512xf16>
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, 0)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%47 : memref<64x64x512xf16>) outs(%41 : memref<64x64x1xf16>) {
            ^bb0(%in: f16, %out: f16):
              %61 = arith.addf %in, %out : f16
              linalg.yield %61 : f16
            }
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%34, %39 : memref<64x64x1xf16>, memref<64x64x1xf16>) outs(%43 : memref<64x64x1xf16>) {
            ^bb0(%in: f16, %in_6: f16, %out: f16):
              %61 = arith.subf %in, %in_6 : f16
              %62 = math.exp %61 : f16
              linalg.yield %62 : f16
            }
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%32, %43 : memref<64x64x1xf16>, memref<64x64x1xf16>) outs(%32 : memref<64x64x1xf16>) {
            ^bb0(%in: f16, %in_6: f16, %out: f16):
              %61 = arith.mulf %in, %in_6 : f16
              linalg.yield %61 : f16
            }
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%32, %41 : memref<64x64x1xf16>, memref<64x64x1xf16>) outs(%32 : memref<64x64x1xf16>) {
            ^bb0(%in: f16, %in_6: f16, %out: f16):
              %61 = arith.addf %in, %in_6 : f16
              linalg.yield %61 : f16
            }
            loom.semaphore_give %41 : memref<64x64x1xf16>
            %59 = arith.muli %arg8, %c65536 : index
            %60 = arith.addi %24, %59 : index
            %reinterpret_cast_5 = memref.reinterpret_cast %arg1 to offset: [%60], sizes: [64, 512, 128], strides: [524288, 128, 1] : memref<32x4096x128xf16> to memref<64x512x128xf16, strided<[524288, 128, 1], offset: ?>>
            loom.copy %reinterpret_cast_5, %51 src_mem_space @mem_DRAM dst_mem_space @mem_L1, area : [8, 2] region : (UL : [%c0, %27], LR : [%c7, %57]) : memref<64x512x128xf16, strided<[524288, 128, 1], offset: ?>> to memref<64x512x128xf16>
            loom.batch_matmul ins(%47, %51 : memref<64x64x512xf16>, memref<64x512x128xf16>) outs(%37 : memref<64x64x128xf16>)
            loom.semaphore_give %51 : memref<64x512x128xf16>
            loom.semaphore_give %47 : memref<64x64x512xf16>
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, 0)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%30, %43 : memref<64x64x128xf16>, memref<64x64x1xf16>) outs(%30 : memref<64x64x128xf16>) {
            ^bb0(%in: f16, %in_6: f16, %out: f16):
              %61 = arith.mulf %in, %in_6 : f16
              linalg.yield %61 : f16
            }
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%30, %37 : memref<64x64x128xf16>, memref<64x64x128xf16>) outs(%30 : memref<64x64x128xf16>) {
            ^bb0(%in: f16, %in_6: f16, %out: f16):
              %61 = arith.addf %in_6, %in : f16
              linalg.yield %61 : f16
            }
            loom.semaphore_give %43 : memref<64x64x1xf16>
            loom.semaphore_give %37 : memref<64x64x128xf16>
            loom.copy %39, %34 src_mem_space @mem_L1 dst_mem_space @mem_L1, area : [1, 1] {reclaim = true} : memref<64x64x1xf16> to memref<64x64x1xf16>
            loom.semaphore_give %39 : memref<64x64x1xf16>
          }
          loom.semaphore_give %34 : memref<64x64x1xf16>
          loom.semaphore_give %23 : memref<64x64x128xf16>
          %52 = loom.broadcast ins(%32 : memref<64x64x1xf16>) outs(%36 : memref<64x64x128xf16>) dim(2) -> memref<64x64x128xf16, strided<[?, ?, ?], offset: ?>>
          loom.semaphore_give %32 : memref<64x64x1xf16>
          %53 = loom.alloc [64, 64, 128] on @L1 : memref<64x64x128xf16>
          %54 = loom.semaphore_take %53 : memref<64x64x128xf16> -> memref<64x64x128xf16>
          linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%30, %52 : memref<64x64x128xf16>, memref<64x64x128xf16, strided<[?, ?, ?], offset: ?>>) outs(%54 : memref<64x64x128xf16>) {
          ^bb0(%in: f16, %in_4: f16, %out: f16):
            %55 = arith.divf %in, %in_4 : f16
            linalg.yield %55 : f16
          }
          loom.semaphore_give %36 : memref<64x64x128xf16>
          loom.semaphore_give %30 : memref<64x64x128xf16>
          %reinterpret_cast_3 = memref.reinterpret_cast %arg3 to offset: [%26], sizes: [64, 64, 128], strides: [524288, 128, 1] : memref<32x4096x128xf16> to memref<64x64x128xf16, strided<[524288, 128, 1], offset: ?>>
          loom.copy %54, %reinterpret_cast_3 src_mem_space @mem_L1 dst_mem_space @mem_DRAM, area : [1, 1] region : (UL : [%arg5, %28], LR : [%arg5, %28]) : memref<64x64x128xf16> to memref<64x64x128xf16, strided<[524288, 128, 1], offset: ?>>
          loom.semaphore_give %54 : memref<64x64x128xf16>
        } {loom.block_sym = @tile_m, loom.iter_type = #loom.iter_type<temporal>}
        scf.reduce 
      } {loom.block_syms = [@tile_b, @tile_m, @tile_m], loom.iter_types = [#loom.iter_type<spatial>, #loom.iter_type<spatial>, #loom.iter_type<spatial>], loom.logical_levels = [1, 0, 0], loom.physical_dims = [@dim_y, @dim_x, @dim_y]}
      return
    }
  }
  module attributes {loom.pass_name = "Materialize", loom.tile_b = {is_reduction = false, upper_bound = 32 : index}, loom.tile_m = {is_reduction = false, upper_bound = 4096 : index}, loom.tile_n = {is_reduction = false, upper_bound = 4096 : index}} {
    func.func @attention__x8_y4y2__d0i1_d1i1_d2i0__f01__n_dim_x_level0_bc8_dim_y_level0_bc4_dim_x_level0_bc8_dim_y_level0_bc4_n__tile_b64__tile_m64__tile_n512(%arg0: memref<32x128x4096xf16>, %arg1: memref<32x4096x128xf16>, %arg2: memref<32x4096x128xf16>, %arg3: memref<32x4096x128xf16>) {
      %c65536 = arith.constant 65536 : index
      %c8192 = arith.constant 8192 : index
      %c33554432 = arith.constant 33554432 : index
      %c32 = arith.constant 32 : index
      %c8 = arith.constant 8 : index
      %c2 = arith.constant 2 : index
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
      scf.parallel (%arg4, %arg5, %arg6) = (%c0, %c0, %c0) to (%c2, %c8, %c4) step (%c1, %c1, %c1) {
        scf.for %arg7 = %c0 to %c2 step %c1 {
          %18 = arith.muli %arg5, %c8 overflow<nsw> : index
          %19 = arith.addi %18, %arg6 : index
          %20 = arith.muli %arg7, %c32 overflow<nsw> : index
          %21 = arith.addi %19, %20 : index
          %22 = loom.alloc [64, 64, 128] on @L1 : memref<64x64x128xf16>
          %23 = loom.semaphore_take %22 : memref<64x64x128xf16> -> memref<64x64x128xf16>
          %24 = arith.muli %arg4, %c33554432 : index
          %25 = arith.muli %21, %c8192 : index
          %26 = arith.addi %24, %25 : index
          %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%26], sizes: [64, 64, 128], strides: [524288, 128, 1] : memref<32x4096x128xf16> to memref<64x64x128xf16, strided<[524288, 128, 1], offset: ?>>
          %27 = arith.muli %arg4, %c4 : index
          %28 = arith.addi %arg6, %27 : index
          loom.copy %reinterpret_cast, %23 src_mem_space @mem_DRAM dst_mem_space @mem_L1, area : [1, 1] region : (UL : [%arg5, %28], LR : [%arg5, %28]) : memref<64x64x128xf16, strided<[524288, 128, 1], offset: ?>> to memref<64x64x128xf16>
          %29 = loom.alloc [64, 64, 128] on @L1 : memref<64x64x128xf16>
          %30 = loom.semaphore_take %29 : memref<64x64x128xf16> -> memref<64x64x128xf16>
          linalg.fill ins(%cst : f16) outs(%30 : memref<64x64x128xf16>)
          %31 = loom.alloc [64, 64, 1] on @L1 : memref<64x64x1xf16>
          %32 = loom.semaphore_take %31 : memref<64x64x1xf16> -> memref<64x64x1xf16>
          linalg.fill ins(%cst_0 : f16) outs(%32 : memref<64x64x1xf16>)
          %33 = loom.alloc [64, 64, 1] on @L1 : memref<64x64x1xf16>
          %34 = loom.semaphore_take %33 : memref<64x64x1xf16> -> memref<64x64x1xf16>
          linalg.fill ins(%cst_1 : f16) outs(%34 : memref<64x64x1xf16>)
          %35 = loom.alloc [64, 64, 128] on @L1 : memref<64x64x128xf16>
          %36 = loom.semaphore_take %35 : memref<64x64x128xf16> -> memref<64x64x128xf16>
          %37 = loom.semaphore_take %35 : memref<64x64x128xf16> -> memref<64x64x128xf16>
          %38 = loom.alloc [64, 64, 1] on @L1 : memref<64x64x1xf16>
          %39 = loom.semaphore_take %38 : memref<64x64x1xf16> -> memref<64x64x1xf16>
          %40 = loom.alloc [64, 64, 1] on @L1 : memref<64x64x1xf16>
          %41 = loom.semaphore_take %40 : memref<64x64x1xf16> -> memref<64x64x1xf16>
          %42 = loom.alloc [64, 64, 1] on @L1 : memref<64x64x1xf16>
          %43 = loom.semaphore_take %42 : memref<64x64x1xf16> -> memref<64x64x1xf16>
          %44 = loom.alloc [64, 128, 512] on @L1 : memref<64x128x512xf16>
          %45 = loom.semaphore_take %44 : memref<64x128x512xf16> -> memref<64x128x512xf16>
          %46 = loom.alloc [64, 64, 512] on @L1 : memref<64x64x512xf16>
          %47 = loom.semaphore_take %46 : memref<64x64x512xf16> -> memref<64x64x512xf16>
          %48 = loom.alloc [64, 64, 512] on @L1 : memref<64x64x512xf16>
          %49 = loom.semaphore_take %48 : memref<64x64x512xf16> -> memref<64x64x512xf16>
          %50 = loom.alloc [64, 512, 128] on @L1 : memref<64x512x128xf16>
          %51 = loom.semaphore_take %50 : memref<64x512x128xf16> -> memref<64x512x128xf16>
          scf.for %arg8 = %c0 to %c8 step %c1 {
            %55 = arith.muli %arg8, %c512 : index
            %56 = arith.addi %24, %55 : index
            %reinterpret_cast_4 = memref.reinterpret_cast %arg0 to offset: [%56], sizes: [64, 128, 512], strides: [524288, 4096, 1] : memref<32x128x4096xf16> to memref<64x128x512xf16, strided<[524288, 4096, 1], offset: ?>>
            %57 = arith.addi %27, %c3 : index
            loom.copy %reinterpret_cast_4, %45 src_mem_space @mem_DRAM dst_mem_space @mem_L1, area : [8, 4] region : (UL : [%c0, %27], LR : [%c7, %57]) : memref<64x128x512xf16, strided<[524288, 4096, 1], offset: ?>> to memref<64x128x512xf16>
            loom.batch_matmul ins(%23, %45 : memref<64x64x128xf16>, memref<64x128x512xf16>) outs(%47 : memref<64x64x512xf16>)
            loom.semaphore_give %45 : memref<64x128x512xf16>
            linalg.fill ins(%cst_1 : f16) outs(%39 : memref<64x64x1xf16>)
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, 0)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%47 : memref<64x64x512xf16>) outs(%39 : memref<64x64x1xf16>) {
            ^bb0(%in: f16, %out: f16):
              %61 = arith.maximumf %in, %out : f16
              linalg.yield %61 : f16
            }
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%34, %39 : memref<64x64x1xf16>, memref<64x64x1xf16>) outs(%39 : memref<64x64x1xf16>) {
            ^bb0(%in: f16, %in_6: f16, %out: f16):
              %61 = arith.mulf %in_6, %cst_2 : f16
              %62 = arith.cmpf ogt, %in, %61 : f16
              %63 = arith.select %62, %in, %61 : f16
              linalg.yield %63 : f16
            }
            %58 = loom.broadcast ins(%39 : memref<64x64x1xf16>) outs(%49 : memref<64x64x512xf16>) dim(2) -> memref<64x64x512xf16, strided<[?, ?, ?], offset: ?>>
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%47 : memref<64x64x512xf16>) outs(%47 : memref<64x64x512xf16>) {
            ^bb0(%in: f16, %out: f16):
              %61 = arith.mulf %in, %cst_2 : f16
              linalg.yield %61 : f16
            }
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%47, %58 : memref<64x64x512xf16>, memref<64x64x512xf16, strided<[?, ?, ?], offset: ?>>) outs(%47 : memref<64x64x512xf16>) {
            ^bb0(%in: f16, %in_6: f16, %out: f16):
              %61 = arith.subf %in, %in_6 : f16
              %62 = math.exp %61 : f16
              linalg.yield %62 : f16
            }
            loom.semaphore_give %49 : memref<64x64x512xf16>
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, 0)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%47 : memref<64x64x512xf16>) outs(%41 : memref<64x64x1xf16>) {
            ^bb0(%in: f16, %out: f16):
              %61 = arith.addf %in, %out : f16
              linalg.yield %61 : f16
            }
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%34, %39 : memref<64x64x1xf16>, memref<64x64x1xf16>) outs(%43 : memref<64x64x1xf16>) {
            ^bb0(%in: f16, %in_6: f16, %out: f16):
              %61 = arith.subf %in, %in_6 : f16
              %62 = math.exp %61 : f16
              linalg.yield %62 : f16
            }
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%32, %43 : memref<64x64x1xf16>, memref<64x64x1xf16>) outs(%32 : memref<64x64x1xf16>) {
            ^bb0(%in: f16, %in_6: f16, %out: f16):
              %61 = arith.mulf %in, %in_6 : f16
              linalg.yield %61 : f16
            }
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%32, %41 : memref<64x64x1xf16>, memref<64x64x1xf16>) outs(%32 : memref<64x64x1xf16>) {
            ^bb0(%in: f16, %in_6: f16, %out: f16):
              %61 = arith.addf %in, %in_6 : f16
              linalg.yield %61 : f16
            }
            loom.semaphore_give %41 : memref<64x64x1xf16>
            %59 = arith.muli %arg8, %c65536 : index
            %60 = arith.addi %24, %59 : index
            %reinterpret_cast_5 = memref.reinterpret_cast %arg1 to offset: [%60], sizes: [64, 512, 128], strides: [524288, 128, 1] : memref<32x4096x128xf16> to memref<64x512x128xf16, strided<[524288, 128, 1], offset: ?>>
            loom.copy %reinterpret_cast_5, %51 src_mem_space @mem_DRAM dst_mem_space @mem_L1, area : [8, 4] region : (UL : [%c0, %27], LR : [%c7, %57]) : memref<64x512x128xf16, strided<[524288, 128, 1], offset: ?>> to memref<64x512x128xf16>
            loom.batch_matmul ins(%47, %51 : memref<64x64x512xf16>, memref<64x512x128xf16>) outs(%37 : memref<64x64x128xf16>)
            loom.semaphore_give %51 : memref<64x512x128xf16>
            loom.semaphore_give %47 : memref<64x64x512xf16>
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, 0)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%30, %43 : memref<64x64x128xf16>, memref<64x64x1xf16>) outs(%30 : memref<64x64x128xf16>) {
            ^bb0(%in: f16, %in_6: f16, %out: f16):
              %61 = arith.mulf %in, %in_6 : f16
              linalg.yield %61 : f16
            }
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%30, %37 : memref<64x64x128xf16>, memref<64x64x128xf16>) outs(%30 : memref<64x64x128xf16>) {
            ^bb0(%in: f16, %in_6: f16, %out: f16):
              %61 = arith.addf %in_6, %in : f16
              linalg.yield %61 : f16
            }
            loom.semaphore_give %43 : memref<64x64x1xf16>
            loom.semaphore_give %37 : memref<64x64x128xf16>
            loom.copy %39, %34 src_mem_space @mem_L1 dst_mem_space @mem_L1, area : [1, 1] {reclaim = true} : memref<64x64x1xf16> to memref<64x64x1xf16>
            loom.semaphore_give %39 : memref<64x64x1xf16>
          }
          loom.semaphore_give %34 : memref<64x64x1xf16>
          loom.semaphore_give %23 : memref<64x64x128xf16>
          %52 = loom.broadcast ins(%32 : memref<64x64x1xf16>) outs(%36 : memref<64x64x128xf16>) dim(2) -> memref<64x64x128xf16, strided<[?, ?, ?], offset: ?>>
          loom.semaphore_give %32 : memref<64x64x1xf16>
          %53 = loom.alloc [64, 64, 128] on @L1 : memref<64x64x128xf16>
          %54 = loom.semaphore_take %53 : memref<64x64x128xf16> -> memref<64x64x128xf16>
          linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%30, %52 : memref<64x64x128xf16>, memref<64x64x128xf16, strided<[?, ?, ?], offset: ?>>) outs(%54 : memref<64x64x128xf16>) {
          ^bb0(%in: f16, %in_4: f16, %out: f16):
            %55 = arith.divf %in, %in_4 : f16
            linalg.yield %55 : f16
          }
          loom.semaphore_give %36 : memref<64x64x128xf16>
          loom.semaphore_give %30 : memref<64x64x128xf16>
          %reinterpret_cast_3 = memref.reinterpret_cast %arg3 to offset: [%26], sizes: [64, 64, 128], strides: [524288, 128, 1] : memref<32x4096x128xf16> to memref<64x64x128xf16, strided<[524288, 128, 1], offset: ?>>
          loom.copy %54, %reinterpret_cast_3 src_mem_space @mem_L1 dst_mem_space @mem_DRAM, area : [1, 1] region : (UL : [%arg5, %28], LR : [%arg5, %28]) : memref<64x64x128xf16> to memref<64x64x128xf16, strided<[524288, 128, 1], offset: ?>>
          loom.semaphore_give %54 : memref<64x64x128xf16>
        } {loom.block_sym = @tile_m, loom.iter_type = #loom.iter_type<temporal>}
        scf.reduce 
      } {loom.block_syms = [@tile_b, @tile_m, @tile_m], loom.iter_types = [#loom.iter_type<spatial>, #loom.iter_type<spatial>, #loom.iter_type<spatial>], loom.logical_levels = [1, 0, 0], loom.physical_dims = [@dim_y, @dim_x, @dim_y]}
      return
    }
  }
  module attributes {loom.pass_name = "Materialize", loom.tile_b = {is_reduction = false, upper_bound = 32 : index}, loom.tile_m = {is_reduction = false, upper_bound = 4096 : index}, loom.tile_n = {is_reduction = false, upper_bound = 4096 : index}} {
    func.func @attention__x8_y8y1__d0i1_d1i1_d2i0__f01__n_dim_x_level0_bc8_dim_y_level1_bc8_dim_x_level0_bc8_dim_y_level1_bc8_n__tile_b64__tile_m64__tile_n512(%arg0: memref<32x128x4096xf16>, %arg1: memref<32x4096x128xf16>, %arg2: memref<32x4096x128xf16>, %arg3: memref<32x4096x128xf16>) {
      %c65536 = arith.constant 65536 : index
      %c8192 = arith.constant 8192 : index
      %c7 = arith.constant 7 : index
      %c8 = arith.constant 8 : index
      %c0 = arith.constant 0 : index
      %cst = arith.constant 0.000000e+00 : f16
      %cst_0 = arith.constant 1.000000e+00 : f16
      %cst_1 = arith.constant 0xFC00 : f16
      %c1 = arith.constant 1 : index
      %cst_2 = arith.constant 8.837890e-02 : f16
      %c512 = arith.constant 512 : index
      scf.parallel (%arg4, %arg5) = (%c0, %c0) to (%c8, %c8) step (%c1, %c1) {
        %18 = arith.muli %arg4, %c8 overflow<nsw> : index
        %19 = arith.addi %18, %arg5 : index
        %20 = loom.alloc [64, 64, 128] on @L1 : memref<64x64x128xf16>
        %21 = loom.semaphore_take %20 : memref<64x64x128xf16> -> memref<64x64x128xf16>
        %22 = arith.muli %19, %c8192 : index
        %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%22], sizes: [64, 64, 128], strides: [524288, 128, 1] : memref<32x4096x128xf16> to memref<64x64x128xf16, strided<[524288, 128, 1], offset: ?>>
        loom.copy %reinterpret_cast, %21 src_mem_space @mem_DRAM dst_mem_space @mem_L1, area : [1, 1] region : (UL : [%arg4, %arg5], LR : [%arg4, %arg5]) : memref<64x64x128xf16, strided<[524288, 128, 1], offset: ?>> to memref<64x64x128xf16>
        %23 = loom.alloc [64, 64, 128] on @L1 : memref<64x64x128xf16>
        %24 = loom.semaphore_take %23 : memref<64x64x128xf16> -> memref<64x64x128xf16>
        linalg.fill ins(%cst : f16) outs(%24 : memref<64x64x128xf16>)
        %25 = loom.alloc [64, 64, 1] on @L1 : memref<64x64x1xf16>
        %26 = loom.semaphore_take %25 : memref<64x64x1xf16> -> memref<64x64x1xf16>
        linalg.fill ins(%cst_0 : f16) outs(%26 : memref<64x64x1xf16>)
        %27 = loom.alloc [64, 64, 1] on @L1 : memref<64x64x1xf16>
        %28 = loom.semaphore_take %27 : memref<64x64x1xf16> -> memref<64x64x1xf16>
        linalg.fill ins(%cst_1 : f16) outs(%28 : memref<64x64x1xf16>)
        %29 = loom.alloc [64, 64, 128] on @L1 : memref<64x64x128xf16>
        %30 = loom.semaphore_take %29 : memref<64x64x128xf16> -> memref<64x64x128xf16>
        %31 = loom.semaphore_take %29 : memref<64x64x128xf16> -> memref<64x64x128xf16>
        %32 = loom.alloc [64, 64, 1] on @L1 : memref<64x64x1xf16>
        %33 = loom.semaphore_take %32 : memref<64x64x1xf16> -> memref<64x64x1xf16>
        %34 = loom.alloc [64, 64, 1] on @L1 : memref<64x64x1xf16>
        %35 = loom.semaphore_take %34 : memref<64x64x1xf16> -> memref<64x64x1xf16>
        %36 = loom.alloc [64, 64, 1] on @L1 : memref<64x64x1xf16>
        %37 = loom.semaphore_take %36 : memref<64x64x1xf16> -> memref<64x64x1xf16>
        %38 = loom.alloc [64, 128, 512] on @L1 : memref<64x128x512xf16>
        %39 = loom.semaphore_take %38 : memref<64x128x512xf16> -> memref<64x128x512xf16>
        %40 = loom.alloc [64, 64, 512] on @L1 : memref<64x64x512xf16>
        %41 = loom.semaphore_take %40 : memref<64x64x512xf16> -> memref<64x64x512xf16>
        %42 = loom.alloc [64, 64, 512] on @L1 : memref<64x64x512xf16>
        %43 = loom.semaphore_take %42 : memref<64x64x512xf16> -> memref<64x64x512xf16>
        %44 = loom.alloc [64, 512, 128] on @L1 : memref<64x512x128xf16>
        %45 = loom.semaphore_take %44 : memref<64x512x128xf16> -> memref<64x512x128xf16>
        scf.for %arg6 = %c0 to %c8 step %c1 {
          %49 = arith.muli %arg6, %c512 : index
          %reinterpret_cast_4 = memref.reinterpret_cast %arg0 to offset: [%49], sizes: [64, 128, 512], strides: [524288, 4096, 1] : memref<32x128x4096xf16> to memref<64x128x512xf16, strided<[524288, 4096, 1], offset: ?>>
          loom.copy %reinterpret_cast_4, %39 src_mem_space @mem_DRAM dst_mem_space @mem_L1, area : [8, 8] region : (UL : [%c0, %c0], LR : [%c7, %c7]) : memref<64x128x512xf16, strided<[524288, 4096, 1], offset: ?>> to memref<64x128x512xf16>
          loom.batch_matmul ins(%21, %39 : memref<64x64x128xf16>, memref<64x128x512xf16>) outs(%41 : memref<64x64x512xf16>)
          loom.semaphore_give %39 : memref<64x128x512xf16>
          linalg.fill ins(%cst_1 : f16) outs(%33 : memref<64x64x1xf16>)
          linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, 0)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%41 : memref<64x64x512xf16>) outs(%33 : memref<64x64x1xf16>) {
          ^bb0(%in: f16, %out: f16):
            %52 = arith.maximumf %in, %out : f16
            linalg.yield %52 : f16
          }
          linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%28, %33 : memref<64x64x1xf16>, memref<64x64x1xf16>) outs(%33 : memref<64x64x1xf16>) {
          ^bb0(%in: f16, %in_6: f16, %out: f16):
            %52 = arith.mulf %in_6, %cst_2 : f16
            %53 = arith.cmpf ogt, %in, %52 : f16
            %54 = arith.select %53, %in, %52 : f16
            linalg.yield %54 : f16
          }
          %50 = loom.broadcast ins(%33 : memref<64x64x1xf16>) outs(%43 : memref<64x64x512xf16>) dim(2) -> memref<64x64x512xf16, strided<[?, ?, ?], offset: ?>>
          linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%41 : memref<64x64x512xf16>) outs(%41 : memref<64x64x512xf16>) {
          ^bb0(%in: f16, %out: f16):
            %52 = arith.mulf %in, %cst_2 : f16
            linalg.yield %52 : f16
          }
          linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%41, %50 : memref<64x64x512xf16>, memref<64x64x512xf16, strided<[?, ?, ?], offset: ?>>) outs(%41 : memref<64x64x512xf16>) {
          ^bb0(%in: f16, %in_6: f16, %out: f16):
            %52 = arith.subf %in, %in_6 : f16
            %53 = math.exp %52 : f16
            linalg.yield %53 : f16
          }
          loom.semaphore_give %43 : memref<64x64x512xf16>
          linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, 0)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%41 : memref<64x64x512xf16>) outs(%35 : memref<64x64x1xf16>) {
          ^bb0(%in: f16, %out: f16):
            %52 = arith.addf %in, %out : f16
            linalg.yield %52 : f16
          }
          linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%28, %33 : memref<64x64x1xf16>, memref<64x64x1xf16>) outs(%37 : memref<64x64x1xf16>) {
          ^bb0(%in: f16, %in_6: f16, %out: f16):
            %52 = arith.subf %in, %in_6 : f16
            %53 = math.exp %52 : f16
            linalg.yield %53 : f16
          }
          linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%26, %37 : memref<64x64x1xf16>, memref<64x64x1xf16>) outs(%26 : memref<64x64x1xf16>) {
          ^bb0(%in: f16, %in_6: f16, %out: f16):
            %52 = arith.mulf %in, %in_6 : f16
            linalg.yield %52 : f16
          }
          linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%26, %35 : memref<64x64x1xf16>, memref<64x64x1xf16>) outs(%26 : memref<64x64x1xf16>) {
          ^bb0(%in: f16, %in_6: f16, %out: f16):
            %52 = arith.addf %in, %in_6 : f16
            linalg.yield %52 : f16
          }
          loom.semaphore_give %35 : memref<64x64x1xf16>
          %51 = arith.muli %arg6, %c65536 : index
          %reinterpret_cast_5 = memref.reinterpret_cast %arg1 to offset: [%51], sizes: [64, 512, 128], strides: [524288, 128, 1] : memref<32x4096x128xf16> to memref<64x512x128xf16, strided<[524288, 128, 1], offset: ?>>
          loom.copy %reinterpret_cast_5, %45 src_mem_space @mem_DRAM dst_mem_space @mem_L1, area : [8, 8] region : (UL : [%c0, %c0], LR : [%c7, %c7]) : memref<64x512x128xf16, strided<[524288, 128, 1], offset: ?>> to memref<64x512x128xf16>
          loom.batch_matmul ins(%41, %45 : memref<64x64x512xf16>, memref<64x512x128xf16>) outs(%31 : memref<64x64x128xf16>)
          loom.semaphore_give %45 : memref<64x512x128xf16>
          loom.semaphore_give %41 : memref<64x64x512xf16>
          linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, 0)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%24, %37 : memref<64x64x128xf16>, memref<64x64x1xf16>) outs(%24 : memref<64x64x128xf16>) {
          ^bb0(%in: f16, %in_6: f16, %out: f16):
            %52 = arith.mulf %in, %in_6 : f16
            linalg.yield %52 : f16
          }
          linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%24, %31 : memref<64x64x128xf16>, memref<64x64x128xf16>) outs(%24 : memref<64x64x128xf16>) {
          ^bb0(%in: f16, %in_6: f16, %out: f16):
            %52 = arith.addf %in_6, %in : f16
            linalg.yield %52 : f16
          }
          loom.semaphore_give %37 : memref<64x64x1xf16>
          loom.semaphore_give %31 : memref<64x64x128xf16>
          loom.copy %33, %28 src_mem_space @mem_L1 dst_mem_space @mem_L1, area : [1, 1] {reclaim = true} : memref<64x64x1xf16> to memref<64x64x1xf16>
          loom.semaphore_give %33 : memref<64x64x1xf16>
        }
        loom.semaphore_give %28 : memref<64x64x1xf16>
        loom.semaphore_give %21 : memref<64x64x128xf16>
        %46 = loom.broadcast ins(%26 : memref<64x64x1xf16>) outs(%30 : memref<64x64x128xf16>) dim(2) -> memref<64x64x128xf16, strided<[?, ?, ?], offset: ?>>
        loom.semaphore_give %26 : memref<64x64x1xf16>
        %47 = loom.alloc [64, 64, 128] on @L1 : memref<64x64x128xf16>
        %48 = loom.semaphore_take %47 : memref<64x64x128xf16> -> memref<64x64x128xf16>
        linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%24, %46 : memref<64x64x128xf16>, memref<64x64x128xf16, strided<[?, ?, ?], offset: ?>>) outs(%48 : memref<64x64x128xf16>) {
        ^bb0(%in: f16, %in_4: f16, %out: f16):
          %49 = arith.divf %in, %in_4 : f16
          linalg.yield %49 : f16
        }
        loom.semaphore_give %30 : memref<64x64x128xf16>
        loom.semaphore_give %24 : memref<64x64x128xf16>
        %reinterpret_cast_3 = memref.reinterpret_cast %arg3 to offset: [%22], sizes: [64, 64, 128], strides: [524288, 128, 1] : memref<32x4096x128xf16> to memref<64x64x128xf16, strided<[524288, 128, 1], offset: ?>>
        loom.copy %48, %reinterpret_cast_3 src_mem_space @mem_L1 dst_mem_space @mem_DRAM, area : [1, 1] region : (UL : [%arg4, %arg5], LR : [%arg4, %arg5]) : memref<64x64x128xf16> to memref<64x64x128xf16, strided<[524288, 128, 1], offset: ?>>
        loom.semaphore_give %48 : memref<64x64x128xf16>
        scf.reduce 
      } {loom.block_syms = [@tile_m, @tile_m], loom.iter_types = [#loom.iter_type<spatial>, #loom.iter_type<spatial>], loom.logical_levels = [0, 0], loom.physical_dims = [@dim_x, @dim_y]}
      return
    }
  }
  module attributes {loom.pass_name = "Materialize", loom.tile_b = {is_reduction = false, upper_bound = 32 : index}, loom.tile_m = {is_reduction = false, upper_bound = 4096 : index}, loom.tile_n = {is_reduction = false, upper_bound = 4096 : index}} {
    func.func @attention__x1x8_y8__d0i1_d1i1_d2i0__f01__n_dim_y_level0_bc8_dim_y_level0_bc8_n__tile_b64__tile_m64__tile_n512(%arg0: memref<32x128x4096xf16>, %arg1: memref<32x4096x128xf16>, %arg2: memref<32x4096x128xf16>, %arg3: memref<32x4096x128xf16>) {
      %c65536 = arith.constant 65536 : index
      %c8192 = arith.constant 8192 : index
      %c33554432 = arith.constant 33554432 : index
      %c8 = arith.constant 8 : index
      %c7 = arith.constant 7 : index
      %c0 = arith.constant 0 : index
      %cst = arith.constant 0.000000e+00 : f16
      %cst_0 = arith.constant 1.000000e+00 : f16
      %cst_1 = arith.constant 0xFC00 : f16
      %c1 = arith.constant 1 : index
      %cst_2 = arith.constant 8.837890e-02 : f16
      %c512 = arith.constant 512 : index
      scf.parallel (%arg4, %arg5) = (%c0, %c0) to (%c8, %c8) step (%c1, %c1) {
        scf.for %arg6 = %c0 to %c8 step %c1 {
          %18 = arith.muli %arg6, %c8 overflow<nsw> : index
          %19 = arith.addi %arg5, %18 : index
          %20 = loom.alloc [64, 64, 128] on @L1 : memref<64x64x128xf16>
          %21 = loom.semaphore_take %20 : memref<64x64x128xf16> -> memref<64x64x128xf16>
          %22 = arith.muli %arg4, %c33554432 : index
          %23 = arith.muli %19, %c8192 : index
          %24 = arith.addi %22, %23 : index
          %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%24], sizes: [64, 64, 128], strides: [524288, 128, 1] : memref<32x4096x128xf16> to memref<64x64x128xf16, strided<[524288, 128, 1], offset: ?>>
          loom.copy %reinterpret_cast, %21 src_mem_space @mem_DRAM dst_mem_space @mem_L1, area : [1, 1] region : (UL : [%arg4, %arg5], LR : [%arg4, %arg5]) : memref<64x64x128xf16, strided<[524288, 128, 1], offset: ?>> to memref<64x64x128xf16>
          %25 = loom.alloc [64, 64, 128] on @L1 : memref<64x64x128xf16>
          %26 = loom.semaphore_take %25 : memref<64x64x128xf16> -> memref<64x64x128xf16>
          linalg.fill ins(%cst : f16) outs(%26 : memref<64x64x128xf16>)
          %27 = loom.alloc [64, 64, 1] on @L1 : memref<64x64x1xf16>
          %28 = loom.semaphore_take %27 : memref<64x64x1xf16> -> memref<64x64x1xf16>
          linalg.fill ins(%cst_0 : f16) outs(%28 : memref<64x64x1xf16>)
          %29 = loom.alloc [64, 64, 1] on @L1 : memref<64x64x1xf16>
          %30 = loom.semaphore_take %29 : memref<64x64x1xf16> -> memref<64x64x1xf16>
          linalg.fill ins(%cst_1 : f16) outs(%30 : memref<64x64x1xf16>)
          %31 = loom.alloc [64, 64, 128] on @L1 : memref<64x64x128xf16>
          %32 = loom.semaphore_take %31 : memref<64x64x128xf16> -> memref<64x64x128xf16>
          %33 = loom.semaphore_take %31 : memref<64x64x128xf16> -> memref<64x64x128xf16>
          %34 = loom.alloc [64, 64, 1] on @L1 : memref<64x64x1xf16>
          %35 = loom.semaphore_take %34 : memref<64x64x1xf16> -> memref<64x64x1xf16>
          %36 = loom.alloc [64, 64, 1] on @L1 : memref<64x64x1xf16>
          %37 = loom.semaphore_take %36 : memref<64x64x1xf16> -> memref<64x64x1xf16>
          %38 = loom.alloc [64, 64, 1] on @L1 : memref<64x64x1xf16>
          %39 = loom.semaphore_take %38 : memref<64x64x1xf16> -> memref<64x64x1xf16>
          %40 = loom.alloc [64, 128, 512] on @L1 : memref<64x128x512xf16>
          %41 = loom.semaphore_take %40 : memref<64x128x512xf16> -> memref<64x128x512xf16>
          %42 = loom.alloc [64, 64, 512] on @L1 : memref<64x64x512xf16>
          %43 = loom.semaphore_take %42 : memref<64x64x512xf16> -> memref<64x64x512xf16>
          %44 = loom.alloc [64, 64, 512] on @L1 : memref<64x64x512xf16>
          %45 = loom.semaphore_take %44 : memref<64x64x512xf16> -> memref<64x64x512xf16>
          %46 = loom.alloc [64, 512, 128] on @L1 : memref<64x512x128xf16>
          %47 = loom.semaphore_take %46 : memref<64x512x128xf16> -> memref<64x512x128xf16>
          scf.for %arg7 = %c0 to %c8 step %c1 {
            %51 = arith.muli %arg7, %c512 : index
            %52 = arith.addi %22, %51 : index
            %reinterpret_cast_4 = memref.reinterpret_cast %arg0 to offset: [%52], sizes: [64, 128, 512], strides: [524288, 4096, 1] : memref<32x128x4096xf16> to memref<64x128x512xf16, strided<[524288, 4096, 1], offset: ?>>
            loom.copy %reinterpret_cast_4, %41 src_mem_space @mem_DRAM dst_mem_space @mem_L1, area : [1, 8] region : (UL : [%arg4, %c0], LR : [%arg4, %c7]) : memref<64x128x512xf16, strided<[524288, 4096, 1], offset: ?>> to memref<64x128x512xf16>
            loom.batch_matmul ins(%21, %41 : memref<64x64x128xf16>, memref<64x128x512xf16>) outs(%43 : memref<64x64x512xf16>)
            loom.semaphore_give %41 : memref<64x128x512xf16>
            linalg.fill ins(%cst_1 : f16) outs(%35 : memref<64x64x1xf16>)
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, 0)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%43 : memref<64x64x512xf16>) outs(%35 : memref<64x64x1xf16>) {
            ^bb0(%in: f16, %out: f16):
              %56 = arith.maximumf %in, %out : f16
              linalg.yield %56 : f16
            }
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%30, %35 : memref<64x64x1xf16>, memref<64x64x1xf16>) outs(%35 : memref<64x64x1xf16>) {
            ^bb0(%in: f16, %in_6: f16, %out: f16):
              %56 = arith.mulf %in_6, %cst_2 : f16
              %57 = arith.cmpf ogt, %in, %56 : f16
              %58 = arith.select %57, %in, %56 : f16
              linalg.yield %58 : f16
            }
            %53 = loom.broadcast ins(%35 : memref<64x64x1xf16>) outs(%45 : memref<64x64x512xf16>) dim(2) -> memref<64x64x512xf16, strided<[?, ?, ?], offset: ?>>
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%43 : memref<64x64x512xf16>) outs(%43 : memref<64x64x512xf16>) {
            ^bb0(%in: f16, %out: f16):
              %56 = arith.mulf %in, %cst_2 : f16
              linalg.yield %56 : f16
            }
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%43, %53 : memref<64x64x512xf16>, memref<64x64x512xf16, strided<[?, ?, ?], offset: ?>>) outs(%43 : memref<64x64x512xf16>) {
            ^bb0(%in: f16, %in_6: f16, %out: f16):
              %56 = arith.subf %in, %in_6 : f16
              %57 = math.exp %56 : f16
              linalg.yield %57 : f16
            }
            loom.semaphore_give %45 : memref<64x64x512xf16>
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, 0)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%43 : memref<64x64x512xf16>) outs(%37 : memref<64x64x1xf16>) {
            ^bb0(%in: f16, %out: f16):
              %56 = arith.addf %in, %out : f16
              linalg.yield %56 : f16
            }
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%30, %35 : memref<64x64x1xf16>, memref<64x64x1xf16>) outs(%39 : memref<64x64x1xf16>) {
            ^bb0(%in: f16, %in_6: f16, %out: f16):
              %56 = arith.subf %in, %in_6 : f16
              %57 = math.exp %56 : f16
              linalg.yield %57 : f16
            }
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%28, %39 : memref<64x64x1xf16>, memref<64x64x1xf16>) outs(%28 : memref<64x64x1xf16>) {
            ^bb0(%in: f16, %in_6: f16, %out: f16):
              %56 = arith.mulf %in, %in_6 : f16
              linalg.yield %56 : f16
            }
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%28, %37 : memref<64x64x1xf16>, memref<64x64x1xf16>) outs(%28 : memref<64x64x1xf16>) {
            ^bb0(%in: f16, %in_6: f16, %out: f16):
              %56 = arith.addf %in, %in_6 : f16
              linalg.yield %56 : f16
            }
            loom.semaphore_give %37 : memref<64x64x1xf16>
            %54 = arith.muli %arg7, %c65536 : index
            %55 = arith.addi %22, %54 : index
            %reinterpret_cast_5 = memref.reinterpret_cast %arg1 to offset: [%55], sizes: [64, 512, 128], strides: [524288, 128, 1] : memref<32x4096x128xf16> to memref<64x512x128xf16, strided<[524288, 128, 1], offset: ?>>
            loom.copy %reinterpret_cast_5, %47 src_mem_space @mem_DRAM dst_mem_space @mem_L1, area : [1, 8] region : (UL : [%arg4, %c0], LR : [%arg4, %c7]) : memref<64x512x128xf16, strided<[524288, 128, 1], offset: ?>> to memref<64x512x128xf16>
            loom.batch_matmul ins(%43, %47 : memref<64x64x512xf16>, memref<64x512x128xf16>) outs(%33 : memref<64x64x128xf16>)
            loom.semaphore_give %47 : memref<64x512x128xf16>
            loom.semaphore_give %43 : memref<64x64x512xf16>
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, 0)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%26, %39 : memref<64x64x128xf16>, memref<64x64x1xf16>) outs(%26 : memref<64x64x128xf16>) {
            ^bb0(%in: f16, %in_6: f16, %out: f16):
              %56 = arith.mulf %in, %in_6 : f16
              linalg.yield %56 : f16
            }
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%26, %33 : memref<64x64x128xf16>, memref<64x64x128xf16>) outs(%26 : memref<64x64x128xf16>) {
            ^bb0(%in: f16, %in_6: f16, %out: f16):
              %56 = arith.addf %in_6, %in : f16
              linalg.yield %56 : f16
            }
            loom.semaphore_give %39 : memref<64x64x1xf16>
            loom.semaphore_give %33 : memref<64x64x128xf16>
            loom.copy %35, %30 src_mem_space @mem_L1 dst_mem_space @mem_L1, area : [1, 1] {reclaim = true} : memref<64x64x1xf16> to memref<64x64x1xf16>
            loom.semaphore_give %35 : memref<64x64x1xf16>
          }
          loom.semaphore_give %30 : memref<64x64x1xf16>
          loom.semaphore_give %21 : memref<64x64x128xf16>
          %48 = loom.broadcast ins(%28 : memref<64x64x1xf16>) outs(%32 : memref<64x64x128xf16>) dim(2) -> memref<64x64x128xf16, strided<[?, ?, ?], offset: ?>>
          loom.semaphore_give %28 : memref<64x64x1xf16>
          %49 = loom.alloc [64, 64, 128] on @L1 : memref<64x64x128xf16>
          %50 = loom.semaphore_take %49 : memref<64x64x128xf16> -> memref<64x64x128xf16>
          linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%26, %48 : memref<64x64x128xf16>, memref<64x64x128xf16, strided<[?, ?, ?], offset: ?>>) outs(%50 : memref<64x64x128xf16>) {
          ^bb0(%in: f16, %in_4: f16, %out: f16):
            %51 = arith.divf %in, %in_4 : f16
            linalg.yield %51 : f16
          }
          loom.semaphore_give %32 : memref<64x64x128xf16>
          loom.semaphore_give %26 : memref<64x64x128xf16>
          %reinterpret_cast_3 = memref.reinterpret_cast %arg3 to offset: [%24], sizes: [64, 64, 128], strides: [524288, 128, 1] : memref<32x4096x128xf16> to memref<64x64x128xf16, strided<[524288, 128, 1], offset: ?>>
          loom.copy %50, %reinterpret_cast_3 src_mem_space @mem_L1 dst_mem_space @mem_DRAM, area : [1, 1] region : (UL : [%arg4, %arg5], LR : [%arg4, %arg5]) : memref<64x64x128xf16> to memref<64x64x128xf16, strided<[524288, 128, 1], offset: ?>>
          loom.semaphore_give %50 : memref<64x64x128xf16>
        } {loom.block_sym = @tile_m, loom.iter_type = #loom.iter_type<temporal>}
        scf.reduce 
      } {loom.block_syms = [@tile_b, @tile_m], loom.iter_types = [#loom.iter_type<spatial>, #loom.iter_type<spatial>], loom.logical_levels = [1, 0], loom.physical_dims = [@dim_x, @dim_y]}
      return
    }
  }
  module attributes {loom.pass_name = "Materialize", loom.tile_b = {is_reduction = false, upper_bound = 32 : index}, loom.tile_m = {is_reduction = false, upper_bound = 4096 : index}, loom.tile_n = {is_reduction = false, upper_bound = 4096 : index}} {
    func.func @attention__x2x4_y8__d0i1_d1i1_d2i0__f01__n_dim_x_level0_bc2_dim_y_level0_bc8_dim_x_level0_bc2_dim_y_level0_bc8_n__tile_b64__tile_m64__tile_n512(%arg0: memref<32x128x4096xf16>, %arg1: memref<32x4096x128xf16>, %arg2: memref<32x4096x128xf16>, %arg3: memref<32x4096x128xf16>) {
      %c65536 = arith.constant 65536 : index
      %c8192 = arith.constant 8192 : index
      %c33554432 = arith.constant 33554432 : index
      %c16 = arith.constant 16 : index
      %c8 = arith.constant 8 : index
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
      scf.parallel (%arg4, %arg5, %arg6) = (%c0, %c0, %c0) to (%c4, %c2, %c8) step (%c1, %c1, %c1) {
        scf.for %arg7 = %c0 to %c4 step %c1 {
          %18 = arith.muli %arg5, %c2 overflow<nsw> : index
          %19 = arith.addi %18, %arg6 : index
          %20 = arith.muli %arg7, %c16 overflow<nsw> : index
          %21 = arith.addi %19, %20 : index
          %22 = loom.alloc [64, 64, 128] on @L1 : memref<64x64x128xf16>
          %23 = loom.semaphore_take %22 : memref<64x64x128xf16> -> memref<64x64x128xf16>
          %24 = arith.muli %arg4, %c33554432 : index
          %25 = arith.muli %21, %c8192 : index
          %26 = arith.addi %24, %25 : index
          %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%26], sizes: [64, 64, 128], strides: [524288, 128, 1] : memref<32x4096x128xf16> to memref<64x64x128xf16, strided<[524288, 128, 1], offset: ?>>
          %27 = arith.muli %arg4, %c2 : index
          %28 = arith.addi %arg5, %27 : index
          loom.copy %reinterpret_cast, %23 src_mem_space @mem_DRAM dst_mem_space @mem_L1, area : [1, 1] region : (UL : [%28, %arg6], LR : [%28, %arg6]) : memref<64x64x128xf16, strided<[524288, 128, 1], offset: ?>> to memref<64x64x128xf16>
          %29 = loom.alloc [64, 64, 128] on @L1 : memref<64x64x128xf16>
          %30 = loom.semaphore_take %29 : memref<64x64x128xf16> -> memref<64x64x128xf16>
          linalg.fill ins(%cst : f16) outs(%30 : memref<64x64x128xf16>)
          %31 = loom.alloc [64, 64, 1] on @L1 : memref<64x64x1xf16>
          %32 = loom.semaphore_take %31 : memref<64x64x1xf16> -> memref<64x64x1xf16>
          linalg.fill ins(%cst_0 : f16) outs(%32 : memref<64x64x1xf16>)
          %33 = loom.alloc [64, 64, 1] on @L1 : memref<64x64x1xf16>
          %34 = loom.semaphore_take %33 : memref<64x64x1xf16> -> memref<64x64x1xf16>
          linalg.fill ins(%cst_1 : f16) outs(%34 : memref<64x64x1xf16>)
          %35 = loom.alloc [64, 64, 128] on @L1 : memref<64x64x128xf16>
          %36 = loom.semaphore_take %35 : memref<64x64x128xf16> -> memref<64x64x128xf16>
          %37 = loom.semaphore_take %35 : memref<64x64x128xf16> -> memref<64x64x128xf16>
          %38 = loom.alloc [64, 64, 1] on @L1 : memref<64x64x1xf16>
          %39 = loom.semaphore_take %38 : memref<64x64x1xf16> -> memref<64x64x1xf16>
          %40 = loom.alloc [64, 64, 1] on @L1 : memref<64x64x1xf16>
          %41 = loom.semaphore_take %40 : memref<64x64x1xf16> -> memref<64x64x1xf16>
          %42 = loom.alloc [64, 64, 1] on @L1 : memref<64x64x1xf16>
          %43 = loom.semaphore_take %42 : memref<64x64x1xf16> -> memref<64x64x1xf16>
          %44 = loom.alloc [64, 128, 512] on @L1 : memref<64x128x512xf16>
          %45 = loom.semaphore_take %44 : memref<64x128x512xf16> -> memref<64x128x512xf16>
          %46 = loom.alloc [64, 64, 512] on @L1 : memref<64x64x512xf16>
          %47 = loom.semaphore_take %46 : memref<64x64x512xf16> -> memref<64x64x512xf16>
          %48 = loom.alloc [64, 64, 512] on @L1 : memref<64x64x512xf16>
          %49 = loom.semaphore_take %48 : memref<64x64x512xf16> -> memref<64x64x512xf16>
          %50 = loom.alloc [64, 512, 128] on @L1 : memref<64x512x128xf16>
          %51 = loom.semaphore_take %50 : memref<64x512x128xf16> -> memref<64x512x128xf16>
          scf.for %arg8 = %c0 to %c8 step %c1 {
            %55 = arith.muli %arg8, %c512 : index
            %56 = arith.addi %24, %55 : index
            %reinterpret_cast_4 = memref.reinterpret_cast %arg0 to offset: [%56], sizes: [64, 128, 512], strides: [524288, 4096, 1] : memref<32x128x4096xf16> to memref<64x128x512xf16, strided<[524288, 4096, 1], offset: ?>>
            %57 = arith.addi %27, %c1 : index
            loom.copy %reinterpret_cast_4, %45 src_mem_space @mem_DRAM dst_mem_space @mem_L1, area : [2, 8] region : (UL : [%27, %c0], LR : [%57, %c7]) : memref<64x128x512xf16, strided<[524288, 4096, 1], offset: ?>> to memref<64x128x512xf16>
            loom.batch_matmul ins(%23, %45 : memref<64x64x128xf16>, memref<64x128x512xf16>) outs(%47 : memref<64x64x512xf16>)
            loom.semaphore_give %45 : memref<64x128x512xf16>
            linalg.fill ins(%cst_1 : f16) outs(%39 : memref<64x64x1xf16>)
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, 0)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%47 : memref<64x64x512xf16>) outs(%39 : memref<64x64x1xf16>) {
            ^bb0(%in: f16, %out: f16):
              %61 = arith.maximumf %in, %out : f16
              linalg.yield %61 : f16
            }
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%34, %39 : memref<64x64x1xf16>, memref<64x64x1xf16>) outs(%39 : memref<64x64x1xf16>) {
            ^bb0(%in: f16, %in_6: f16, %out: f16):
              %61 = arith.mulf %in_6, %cst_2 : f16
              %62 = arith.cmpf ogt, %in, %61 : f16
              %63 = arith.select %62, %in, %61 : f16
              linalg.yield %63 : f16
            }
            %58 = loom.broadcast ins(%39 : memref<64x64x1xf16>) outs(%49 : memref<64x64x512xf16>) dim(2) -> memref<64x64x512xf16, strided<[?, ?, ?], offset: ?>>
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%47 : memref<64x64x512xf16>) outs(%47 : memref<64x64x512xf16>) {
            ^bb0(%in: f16, %out: f16):
              %61 = arith.mulf %in, %cst_2 : f16
              linalg.yield %61 : f16
            }
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%47, %58 : memref<64x64x512xf16>, memref<64x64x512xf16, strided<[?, ?, ?], offset: ?>>) outs(%47 : memref<64x64x512xf16>) {
            ^bb0(%in: f16, %in_6: f16, %out: f16):
              %61 = arith.subf %in, %in_6 : f16
              %62 = math.exp %61 : f16
              linalg.yield %62 : f16
            }
            loom.semaphore_give %49 : memref<64x64x512xf16>
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, 0)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%47 : memref<64x64x512xf16>) outs(%41 : memref<64x64x1xf16>) {
            ^bb0(%in: f16, %out: f16):
              %61 = arith.addf %in, %out : f16
              linalg.yield %61 : f16
            }
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%34, %39 : memref<64x64x1xf16>, memref<64x64x1xf16>) outs(%43 : memref<64x64x1xf16>) {
            ^bb0(%in: f16, %in_6: f16, %out: f16):
              %61 = arith.subf %in, %in_6 : f16
              %62 = math.exp %61 : f16
              linalg.yield %62 : f16
            }
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%32, %43 : memref<64x64x1xf16>, memref<64x64x1xf16>) outs(%32 : memref<64x64x1xf16>) {
            ^bb0(%in: f16, %in_6: f16, %out: f16):
              %61 = arith.mulf %in, %in_6 : f16
              linalg.yield %61 : f16
            }
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%32, %41 : memref<64x64x1xf16>, memref<64x64x1xf16>) outs(%32 : memref<64x64x1xf16>) {
            ^bb0(%in: f16, %in_6: f16, %out: f16):
              %61 = arith.addf %in, %in_6 : f16
              linalg.yield %61 : f16
            }
            loom.semaphore_give %41 : memref<64x64x1xf16>
            %59 = arith.muli %arg8, %c65536 : index
            %60 = arith.addi %24, %59 : index
            %reinterpret_cast_5 = memref.reinterpret_cast %arg1 to offset: [%60], sizes: [64, 512, 128], strides: [524288, 128, 1] : memref<32x4096x128xf16> to memref<64x512x128xf16, strided<[524288, 128, 1], offset: ?>>
            loom.copy %reinterpret_cast_5, %51 src_mem_space @mem_DRAM dst_mem_space @mem_L1, area : [2, 8] region : (UL : [%27, %c0], LR : [%57, %c7]) : memref<64x512x128xf16, strided<[524288, 128, 1], offset: ?>> to memref<64x512x128xf16>
            loom.batch_matmul ins(%47, %51 : memref<64x64x512xf16>, memref<64x512x128xf16>) outs(%37 : memref<64x64x128xf16>)
            loom.semaphore_give %51 : memref<64x512x128xf16>
            loom.semaphore_give %47 : memref<64x64x512xf16>
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, 0)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%30, %43 : memref<64x64x128xf16>, memref<64x64x1xf16>) outs(%30 : memref<64x64x128xf16>) {
            ^bb0(%in: f16, %in_6: f16, %out: f16):
              %61 = arith.mulf %in, %in_6 : f16
              linalg.yield %61 : f16
            }
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%30, %37 : memref<64x64x128xf16>, memref<64x64x128xf16>) outs(%30 : memref<64x64x128xf16>) {
            ^bb0(%in: f16, %in_6: f16, %out: f16):
              %61 = arith.addf %in_6, %in : f16
              linalg.yield %61 : f16
            }
            loom.semaphore_give %43 : memref<64x64x1xf16>
            loom.semaphore_give %37 : memref<64x64x128xf16>
            loom.copy %39, %34 src_mem_space @mem_L1 dst_mem_space @mem_L1, area : [1, 1] {reclaim = true} : memref<64x64x1xf16> to memref<64x64x1xf16>
            loom.semaphore_give %39 : memref<64x64x1xf16>
          }
          loom.semaphore_give %34 : memref<64x64x1xf16>
          loom.semaphore_give %23 : memref<64x64x128xf16>
          %52 = loom.broadcast ins(%32 : memref<64x64x1xf16>) outs(%36 : memref<64x64x128xf16>) dim(2) -> memref<64x64x128xf16, strided<[?, ?, ?], offset: ?>>
          loom.semaphore_give %32 : memref<64x64x1xf16>
          %53 = loom.alloc [64, 64, 128] on @L1 : memref<64x64x128xf16>
          %54 = loom.semaphore_take %53 : memref<64x64x128xf16> -> memref<64x64x128xf16>
          linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%30, %52 : memref<64x64x128xf16>, memref<64x64x128xf16, strided<[?, ?, ?], offset: ?>>) outs(%54 : memref<64x64x128xf16>) {
          ^bb0(%in: f16, %in_4: f16, %out: f16):
            %55 = arith.divf %in, %in_4 : f16
            linalg.yield %55 : f16
          }
          loom.semaphore_give %36 : memref<64x64x128xf16>
          loom.semaphore_give %30 : memref<64x64x128xf16>
          %reinterpret_cast_3 = memref.reinterpret_cast %arg3 to offset: [%26], sizes: [64, 64, 128], strides: [524288, 128, 1] : memref<32x4096x128xf16> to memref<64x64x128xf16, strided<[524288, 128, 1], offset: ?>>
          loom.copy %54, %reinterpret_cast_3 src_mem_space @mem_L1 dst_mem_space @mem_DRAM, area : [1, 1] region : (UL : [%28, %arg6], LR : [%28, %arg6]) : memref<64x64x128xf16> to memref<64x64x128xf16, strided<[524288, 128, 1], offset: ?>>
          loom.semaphore_give %54 : memref<64x64x128xf16>
        } {loom.block_sym = @tile_m, loom.iter_type = #loom.iter_type<temporal>}
        scf.reduce 
      } {loom.block_syms = [@tile_b, @tile_m, @tile_m], loom.iter_types = [#loom.iter_type<spatial>, #loom.iter_type<spatial>, #loom.iter_type<spatial>], loom.logical_levels = [1, 0, 0], loom.physical_dims = [@dim_x, @dim_x, @dim_y]}
      return
    }
  }
  module attributes {loom.pass_name = "Materialize", loom.tile_b = {is_reduction = false, upper_bound = 32 : index}, loom.tile_m = {is_reduction = false, upper_bound = 4096 : index}, loom.tile_n = {is_reduction = false, upper_bound = 4096 : index}} {
    func.func @attention__x4x2_y8__d0i1_d1i1_d2i0__f01__n_dim_x_level0_bc4_dim_y_level0_bc8_dim_x_level0_bc4_dim_y_level0_bc8_n__tile_b64__tile_m64__tile_n512(%arg0: memref<32x128x4096xf16>, %arg1: memref<32x4096x128xf16>, %arg2: memref<32x4096x128xf16>, %arg3: memref<32x4096x128xf16>) {
      %c65536 = arith.constant 65536 : index
      %c8192 = arith.constant 8192 : index
      %c33554432 = arith.constant 33554432 : index
      %c32 = arith.constant 32 : index
      %c8 = arith.constant 8 : index
      %c2 = arith.constant 2 : index
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
      scf.parallel (%arg4, %arg5, %arg6) = (%c0, %c0, %c0) to (%c2, %c4, %c8) step (%c1, %c1, %c1) {
        scf.for %arg7 = %c0 to %c2 step %c1 {
          %18 = arith.muli %arg5, %c4 overflow<nsw> : index
          %19 = arith.addi %18, %arg6 : index
          %20 = arith.muli %arg7, %c32 overflow<nsw> : index
          %21 = arith.addi %19, %20 : index
          %22 = loom.alloc [64, 64, 128] on @L1 : memref<64x64x128xf16>
          %23 = loom.semaphore_take %22 : memref<64x64x128xf16> -> memref<64x64x128xf16>
          %24 = arith.muli %arg4, %c33554432 : index
          %25 = arith.muli %21, %c8192 : index
          %26 = arith.addi %24, %25 : index
          %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%26], sizes: [64, 64, 128], strides: [524288, 128, 1] : memref<32x4096x128xf16> to memref<64x64x128xf16, strided<[524288, 128, 1], offset: ?>>
          %27 = arith.muli %arg4, %c4 : index
          %28 = arith.addi %arg5, %27 : index
          loom.copy %reinterpret_cast, %23 src_mem_space @mem_DRAM dst_mem_space @mem_L1, area : [1, 1] region : (UL : [%28, %arg6], LR : [%28, %arg6]) : memref<64x64x128xf16, strided<[524288, 128, 1], offset: ?>> to memref<64x64x128xf16>
          %29 = loom.alloc [64, 64, 128] on @L1 : memref<64x64x128xf16>
          %30 = loom.semaphore_take %29 : memref<64x64x128xf16> -> memref<64x64x128xf16>
          linalg.fill ins(%cst : f16) outs(%30 : memref<64x64x128xf16>)
          %31 = loom.alloc [64, 64, 1] on @L1 : memref<64x64x1xf16>
          %32 = loom.semaphore_take %31 : memref<64x64x1xf16> -> memref<64x64x1xf16>
          linalg.fill ins(%cst_0 : f16) outs(%32 : memref<64x64x1xf16>)
          %33 = loom.alloc [64, 64, 1] on @L1 : memref<64x64x1xf16>
          %34 = loom.semaphore_take %33 : memref<64x64x1xf16> -> memref<64x64x1xf16>
          linalg.fill ins(%cst_1 : f16) outs(%34 : memref<64x64x1xf16>)
          %35 = loom.alloc [64, 64, 128] on @L1 : memref<64x64x128xf16>
          %36 = loom.semaphore_take %35 : memref<64x64x128xf16> -> memref<64x64x128xf16>
          %37 = loom.semaphore_take %35 : memref<64x64x128xf16> -> memref<64x64x128xf16>
          %38 = loom.alloc [64, 64, 1] on @L1 : memref<64x64x1xf16>
          %39 = loom.semaphore_take %38 : memref<64x64x1xf16> -> memref<64x64x1xf16>
          %40 = loom.alloc [64, 64, 1] on @L1 : memref<64x64x1xf16>
          %41 = loom.semaphore_take %40 : memref<64x64x1xf16> -> memref<64x64x1xf16>
          %42 = loom.alloc [64, 64, 1] on @L1 : memref<64x64x1xf16>
          %43 = loom.semaphore_take %42 : memref<64x64x1xf16> -> memref<64x64x1xf16>
          %44 = loom.alloc [64, 128, 512] on @L1 : memref<64x128x512xf16>
          %45 = loom.semaphore_take %44 : memref<64x128x512xf16> -> memref<64x128x512xf16>
          %46 = loom.alloc [64, 64, 512] on @L1 : memref<64x64x512xf16>
          %47 = loom.semaphore_take %46 : memref<64x64x512xf16> -> memref<64x64x512xf16>
          %48 = loom.alloc [64, 64, 512] on @L1 : memref<64x64x512xf16>
          %49 = loom.semaphore_take %48 : memref<64x64x512xf16> -> memref<64x64x512xf16>
          %50 = loom.alloc [64, 512, 128] on @L1 : memref<64x512x128xf16>
          %51 = loom.semaphore_take %50 : memref<64x512x128xf16> -> memref<64x512x128xf16>
          scf.for %arg8 = %c0 to %c8 step %c1 {
            %55 = arith.muli %arg8, %c512 : index
            %56 = arith.addi %24, %55 : index
            %reinterpret_cast_4 = memref.reinterpret_cast %arg0 to offset: [%56], sizes: [64, 128, 512], strides: [524288, 4096, 1] : memref<32x128x4096xf16> to memref<64x128x512xf16, strided<[524288, 4096, 1], offset: ?>>
            %57 = arith.addi %27, %c3 : index
            loom.copy %reinterpret_cast_4, %45 src_mem_space @mem_DRAM dst_mem_space @mem_L1, area : [4, 8] region : (UL : [%27, %c0], LR : [%57, %c7]) : memref<64x128x512xf16, strided<[524288, 4096, 1], offset: ?>> to memref<64x128x512xf16>
            loom.batch_matmul ins(%23, %45 : memref<64x64x128xf16>, memref<64x128x512xf16>) outs(%47 : memref<64x64x512xf16>)
            loom.semaphore_give %45 : memref<64x128x512xf16>
            linalg.fill ins(%cst_1 : f16) outs(%39 : memref<64x64x1xf16>)
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, 0)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%47 : memref<64x64x512xf16>) outs(%39 : memref<64x64x1xf16>) {
            ^bb0(%in: f16, %out: f16):
              %61 = arith.maximumf %in, %out : f16
              linalg.yield %61 : f16
            }
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%34, %39 : memref<64x64x1xf16>, memref<64x64x1xf16>) outs(%39 : memref<64x64x1xf16>) {
            ^bb0(%in: f16, %in_6: f16, %out: f16):
              %61 = arith.mulf %in_6, %cst_2 : f16
              %62 = arith.cmpf ogt, %in, %61 : f16
              %63 = arith.select %62, %in, %61 : f16
              linalg.yield %63 : f16
            }
            %58 = loom.broadcast ins(%39 : memref<64x64x1xf16>) outs(%49 : memref<64x64x512xf16>) dim(2) -> memref<64x64x512xf16, strided<[?, ?, ?], offset: ?>>
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%47 : memref<64x64x512xf16>) outs(%47 : memref<64x64x512xf16>) {
            ^bb0(%in: f16, %out: f16):
              %61 = arith.mulf %in, %cst_2 : f16
              linalg.yield %61 : f16
            }
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%47, %58 : memref<64x64x512xf16>, memref<64x64x512xf16, strided<[?, ?, ?], offset: ?>>) outs(%47 : memref<64x64x512xf16>) {
            ^bb0(%in: f16, %in_6: f16, %out: f16):
              %61 = arith.subf %in, %in_6 : f16
              %62 = math.exp %61 : f16
              linalg.yield %62 : f16
            }
            loom.semaphore_give %49 : memref<64x64x512xf16>
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, 0)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%47 : memref<64x64x512xf16>) outs(%41 : memref<64x64x1xf16>) {
            ^bb0(%in: f16, %out: f16):
              %61 = arith.addf %in, %out : f16
              linalg.yield %61 : f16
            }
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%34, %39 : memref<64x64x1xf16>, memref<64x64x1xf16>) outs(%43 : memref<64x64x1xf16>) {
            ^bb0(%in: f16, %in_6: f16, %out: f16):
              %61 = arith.subf %in, %in_6 : f16
              %62 = math.exp %61 : f16
              linalg.yield %62 : f16
            }
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%32, %43 : memref<64x64x1xf16>, memref<64x64x1xf16>) outs(%32 : memref<64x64x1xf16>) {
            ^bb0(%in: f16, %in_6: f16, %out: f16):
              %61 = arith.mulf %in, %in_6 : f16
              linalg.yield %61 : f16
            }
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%32, %41 : memref<64x64x1xf16>, memref<64x64x1xf16>) outs(%32 : memref<64x64x1xf16>) {
            ^bb0(%in: f16, %in_6: f16, %out: f16):
              %61 = arith.addf %in, %in_6 : f16
              linalg.yield %61 : f16
            }
            loom.semaphore_give %41 : memref<64x64x1xf16>
            %59 = arith.muli %arg8, %c65536 : index
            %60 = arith.addi %24, %59 : index
            %reinterpret_cast_5 = memref.reinterpret_cast %arg1 to offset: [%60], sizes: [64, 512, 128], strides: [524288, 128, 1] : memref<32x4096x128xf16> to memref<64x512x128xf16, strided<[524288, 128, 1], offset: ?>>
            loom.copy %reinterpret_cast_5, %51 src_mem_space @mem_DRAM dst_mem_space @mem_L1, area : [4, 8] region : (UL : [%27, %c0], LR : [%57, %c7]) : memref<64x512x128xf16, strided<[524288, 128, 1], offset: ?>> to memref<64x512x128xf16>
            loom.batch_matmul ins(%47, %51 : memref<64x64x512xf16>, memref<64x512x128xf16>) outs(%37 : memref<64x64x128xf16>)
            loom.semaphore_give %51 : memref<64x512x128xf16>
            loom.semaphore_give %47 : memref<64x64x512xf16>
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, 0)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%30, %43 : memref<64x64x128xf16>, memref<64x64x1xf16>) outs(%30 : memref<64x64x128xf16>) {
            ^bb0(%in: f16, %in_6: f16, %out: f16):
              %61 = arith.mulf %in, %in_6 : f16
              linalg.yield %61 : f16
            }
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%30, %37 : memref<64x64x128xf16>, memref<64x64x128xf16>) outs(%30 : memref<64x64x128xf16>) {
            ^bb0(%in: f16, %in_6: f16, %out: f16):
              %61 = arith.addf %in_6, %in : f16
              linalg.yield %61 : f16
            }
            loom.semaphore_give %43 : memref<64x64x1xf16>
            loom.semaphore_give %37 : memref<64x64x128xf16>
            loom.copy %39, %34 src_mem_space @mem_L1 dst_mem_space @mem_L1, area : [1, 1] {reclaim = true} : memref<64x64x1xf16> to memref<64x64x1xf16>
            loom.semaphore_give %39 : memref<64x64x1xf16>
          }
          loom.semaphore_give %34 : memref<64x64x1xf16>
          loom.semaphore_give %23 : memref<64x64x128xf16>
          %52 = loom.broadcast ins(%32 : memref<64x64x1xf16>) outs(%36 : memref<64x64x128xf16>) dim(2) -> memref<64x64x128xf16, strided<[?, ?, ?], offset: ?>>
          loom.semaphore_give %32 : memref<64x64x1xf16>
          %53 = loom.alloc [64, 64, 128] on @L1 : memref<64x64x128xf16>
          %54 = loom.semaphore_take %53 : memref<64x64x128xf16> -> memref<64x64x128xf16>
          linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%30, %52 : memref<64x64x128xf16>, memref<64x64x128xf16, strided<[?, ?, ?], offset: ?>>) outs(%54 : memref<64x64x128xf16>) {
          ^bb0(%in: f16, %in_4: f16, %out: f16):
            %55 = arith.divf %in, %in_4 : f16
            linalg.yield %55 : f16
          }
          loom.semaphore_give %36 : memref<64x64x128xf16>
          loom.semaphore_give %30 : memref<64x64x128xf16>
          %reinterpret_cast_3 = memref.reinterpret_cast %arg3 to offset: [%26], sizes: [64, 64, 128], strides: [524288, 128, 1] : memref<32x4096x128xf16> to memref<64x64x128xf16, strided<[524288, 128, 1], offset: ?>>
          loom.copy %54, %reinterpret_cast_3 src_mem_space @mem_L1 dst_mem_space @mem_DRAM, area : [1, 1] region : (UL : [%28, %arg6], LR : [%28, %arg6]) : memref<64x64x128xf16> to memref<64x64x128xf16, strided<[524288, 128, 1], offset: ?>>
          loom.semaphore_give %54 : memref<64x64x128xf16>
        } {loom.block_sym = @tile_m, loom.iter_type = #loom.iter_type<temporal>}
        scf.reduce 
      } {loom.block_syms = [@tile_b, @tile_m, @tile_m], loom.iter_types = [#loom.iter_type<spatial>, #loom.iter_type<spatial>, #loom.iter_type<spatial>], loom.logical_levels = [1, 0, 0], loom.physical_dims = [@dim_x, @dim_x, @dim_y]}
      return
    }
  }
  module attributes {loom.pass_name = "Materialize", loom.tile_b = {is_reduction = false, upper_bound = 32 : index}, loom.tile_m = {is_reduction = false, upper_bound = 4096 : index}, loom.tile_n = {is_reduction = false, upper_bound = 4096 : index}} {
    func.func @attention__x8x1_y8__d0i1_d1i1_d2i0__f01__n_dim_x_level1_bc8_dim_y_level0_bc8_dim_x_level1_bc8_dim_y_level0_bc8_n__tile_b64__tile_m64__tile_n512(%arg0: memref<32x128x4096xf16>, %arg1: memref<32x4096x128xf16>, %arg2: memref<32x4096x128xf16>, %arg3: memref<32x4096x128xf16>) {
      %c65536 = arith.constant 65536 : index
      %c8192 = arith.constant 8192 : index
      %c7 = arith.constant 7 : index
      %c8 = arith.constant 8 : index
      %c0 = arith.constant 0 : index
      %cst = arith.constant 0.000000e+00 : f16
      %cst_0 = arith.constant 1.000000e+00 : f16
      %cst_1 = arith.constant 0xFC00 : f16
      %c1 = arith.constant 1 : index
      %cst_2 = arith.constant 8.837890e-02 : f16
      %c512 = arith.constant 512 : index
      scf.parallel (%arg4, %arg5) = (%c0, %c0) to (%c8, %c8) step (%c1, %c1) {
        %18 = arith.muli %arg4, %c8 overflow<nsw> : index
        %19 = arith.addi %18, %arg5 : index
        %20 = loom.alloc [64, 64, 128] on @L1 : memref<64x64x128xf16>
        %21 = loom.semaphore_take %20 : memref<64x64x128xf16> -> memref<64x64x128xf16>
        %22 = arith.muli %19, %c8192 : index
        %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%22], sizes: [64, 64, 128], strides: [524288, 128, 1] : memref<32x4096x128xf16> to memref<64x64x128xf16, strided<[524288, 128, 1], offset: ?>>
        loom.copy %reinterpret_cast, %21 src_mem_space @mem_DRAM dst_mem_space @mem_L1, area : [1, 1] region : (UL : [%arg4, %arg5], LR : [%arg4, %arg5]) : memref<64x64x128xf16, strided<[524288, 128, 1], offset: ?>> to memref<64x64x128xf16>
        %23 = loom.alloc [64, 64, 128] on @L1 : memref<64x64x128xf16>
        %24 = loom.semaphore_take %23 : memref<64x64x128xf16> -> memref<64x64x128xf16>
        linalg.fill ins(%cst : f16) outs(%24 : memref<64x64x128xf16>)
        %25 = loom.alloc [64, 64, 1] on @L1 : memref<64x64x1xf16>
        %26 = loom.semaphore_take %25 : memref<64x64x1xf16> -> memref<64x64x1xf16>
        linalg.fill ins(%cst_0 : f16) outs(%26 : memref<64x64x1xf16>)
        %27 = loom.alloc [64, 64, 1] on @L1 : memref<64x64x1xf16>
        %28 = loom.semaphore_take %27 : memref<64x64x1xf16> -> memref<64x64x1xf16>
        linalg.fill ins(%cst_1 : f16) outs(%28 : memref<64x64x1xf16>)
        %29 = loom.alloc [64, 64, 128] on @L1 : memref<64x64x128xf16>
        %30 = loom.semaphore_take %29 : memref<64x64x128xf16> -> memref<64x64x128xf16>
        %31 = loom.semaphore_take %29 : memref<64x64x128xf16> -> memref<64x64x128xf16>
        %32 = loom.alloc [64, 64, 1] on @L1 : memref<64x64x1xf16>
        %33 = loom.semaphore_take %32 : memref<64x64x1xf16> -> memref<64x64x1xf16>
        %34 = loom.alloc [64, 64, 1] on @L1 : memref<64x64x1xf16>
        %35 = loom.semaphore_take %34 : memref<64x64x1xf16> -> memref<64x64x1xf16>
        %36 = loom.alloc [64, 64, 1] on @L1 : memref<64x64x1xf16>
        %37 = loom.semaphore_take %36 : memref<64x64x1xf16> -> memref<64x64x1xf16>
        %38 = loom.alloc [64, 128, 512] on @L1 : memref<64x128x512xf16>
        %39 = loom.semaphore_take %38 : memref<64x128x512xf16> -> memref<64x128x512xf16>
        %40 = loom.alloc [64, 64, 512] on @L1 : memref<64x64x512xf16>
        %41 = loom.semaphore_take %40 : memref<64x64x512xf16> -> memref<64x64x512xf16>
        %42 = loom.alloc [64, 64, 512] on @L1 : memref<64x64x512xf16>
        %43 = loom.semaphore_take %42 : memref<64x64x512xf16> -> memref<64x64x512xf16>
        %44 = loom.alloc [64, 512, 128] on @L1 : memref<64x512x128xf16>
        %45 = loom.semaphore_take %44 : memref<64x512x128xf16> -> memref<64x512x128xf16>
        scf.for %arg6 = %c0 to %c8 step %c1 {
          %49 = arith.muli %arg6, %c512 : index
          %reinterpret_cast_4 = memref.reinterpret_cast %arg0 to offset: [%49], sizes: [64, 128, 512], strides: [524288, 4096, 1] : memref<32x128x4096xf16> to memref<64x128x512xf16, strided<[524288, 4096, 1], offset: ?>>
          loom.copy %reinterpret_cast_4, %39 src_mem_space @mem_DRAM dst_mem_space @mem_L1, area : [8, 8] region : (UL : [%c0, %c0], LR : [%c7, %c7]) : memref<64x128x512xf16, strided<[524288, 4096, 1], offset: ?>> to memref<64x128x512xf16>
          loom.batch_matmul ins(%21, %39 : memref<64x64x128xf16>, memref<64x128x512xf16>) outs(%41 : memref<64x64x512xf16>)
          loom.semaphore_give %39 : memref<64x128x512xf16>
          linalg.fill ins(%cst_1 : f16) outs(%33 : memref<64x64x1xf16>)
          linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, 0)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%41 : memref<64x64x512xf16>) outs(%33 : memref<64x64x1xf16>) {
          ^bb0(%in: f16, %out: f16):
            %52 = arith.maximumf %in, %out : f16
            linalg.yield %52 : f16
          }
          linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%28, %33 : memref<64x64x1xf16>, memref<64x64x1xf16>) outs(%33 : memref<64x64x1xf16>) {
          ^bb0(%in: f16, %in_6: f16, %out: f16):
            %52 = arith.mulf %in_6, %cst_2 : f16
            %53 = arith.cmpf ogt, %in, %52 : f16
            %54 = arith.select %53, %in, %52 : f16
            linalg.yield %54 : f16
          }
          %50 = loom.broadcast ins(%33 : memref<64x64x1xf16>) outs(%43 : memref<64x64x512xf16>) dim(2) -> memref<64x64x512xf16, strided<[?, ?, ?], offset: ?>>
          linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%41 : memref<64x64x512xf16>) outs(%41 : memref<64x64x512xf16>) {
          ^bb0(%in: f16, %out: f16):
            %52 = arith.mulf %in, %cst_2 : f16
            linalg.yield %52 : f16
          }
          linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%41, %50 : memref<64x64x512xf16>, memref<64x64x512xf16, strided<[?, ?, ?], offset: ?>>) outs(%41 : memref<64x64x512xf16>) {
          ^bb0(%in: f16, %in_6: f16, %out: f16):
            %52 = arith.subf %in, %in_6 : f16
            %53 = math.exp %52 : f16
            linalg.yield %53 : f16
          }
          loom.semaphore_give %43 : memref<64x64x512xf16>
          linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, 0)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%41 : memref<64x64x512xf16>) outs(%35 : memref<64x64x1xf16>) {
          ^bb0(%in: f16, %out: f16):
            %52 = arith.addf %in, %out : f16
            linalg.yield %52 : f16
          }
          linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%28, %33 : memref<64x64x1xf16>, memref<64x64x1xf16>) outs(%37 : memref<64x64x1xf16>) {
          ^bb0(%in: f16, %in_6: f16, %out: f16):
            %52 = arith.subf %in, %in_6 : f16
            %53 = math.exp %52 : f16
            linalg.yield %53 : f16
          }
          linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%26, %37 : memref<64x64x1xf16>, memref<64x64x1xf16>) outs(%26 : memref<64x64x1xf16>) {
          ^bb0(%in: f16, %in_6: f16, %out: f16):
            %52 = arith.mulf %in, %in_6 : f16
            linalg.yield %52 : f16
          }
          linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%26, %35 : memref<64x64x1xf16>, memref<64x64x1xf16>) outs(%26 : memref<64x64x1xf16>) {
          ^bb0(%in: f16, %in_6: f16, %out: f16):
            %52 = arith.addf %in, %in_6 : f16
            linalg.yield %52 : f16
          }
          loom.semaphore_give %35 : memref<64x64x1xf16>
          %51 = arith.muli %arg6, %c65536 : index
          %reinterpret_cast_5 = memref.reinterpret_cast %arg1 to offset: [%51], sizes: [64, 512, 128], strides: [524288, 128, 1] : memref<32x4096x128xf16> to memref<64x512x128xf16, strided<[524288, 128, 1], offset: ?>>
          loom.copy %reinterpret_cast_5, %45 src_mem_space @mem_DRAM dst_mem_space @mem_L1, area : [8, 8] region : (UL : [%c0, %c0], LR : [%c7, %c7]) : memref<64x512x128xf16, strided<[524288, 128, 1], offset: ?>> to memref<64x512x128xf16>
          loom.batch_matmul ins(%41, %45 : memref<64x64x512xf16>, memref<64x512x128xf16>) outs(%31 : memref<64x64x128xf16>)
          loom.semaphore_give %45 : memref<64x512x128xf16>
          loom.semaphore_give %41 : memref<64x64x512xf16>
          linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, 0)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%24, %37 : memref<64x64x128xf16>, memref<64x64x1xf16>) outs(%24 : memref<64x64x128xf16>) {
          ^bb0(%in: f16, %in_6: f16, %out: f16):
            %52 = arith.mulf %in, %in_6 : f16
            linalg.yield %52 : f16
          }
          linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%24, %31 : memref<64x64x128xf16>, memref<64x64x128xf16>) outs(%24 : memref<64x64x128xf16>) {
          ^bb0(%in: f16, %in_6: f16, %out: f16):
            %52 = arith.addf %in_6, %in : f16
            linalg.yield %52 : f16
          }
          loom.semaphore_give %37 : memref<64x64x1xf16>
          loom.semaphore_give %31 : memref<64x64x128xf16>
          loom.copy %33, %28 src_mem_space @mem_L1 dst_mem_space @mem_L1, area : [1, 1] {reclaim = true} : memref<64x64x1xf16> to memref<64x64x1xf16>
          loom.semaphore_give %33 : memref<64x64x1xf16>
        }
        loom.semaphore_give %28 : memref<64x64x1xf16>
        loom.semaphore_give %21 : memref<64x64x128xf16>
        %46 = loom.broadcast ins(%26 : memref<64x64x1xf16>) outs(%30 : memref<64x64x128xf16>) dim(2) -> memref<64x64x128xf16, strided<[?, ?, ?], offset: ?>>
        loom.semaphore_give %26 : memref<64x64x1xf16>
        %47 = loom.alloc [64, 64, 128] on @L1 : memref<64x64x128xf16>
        %48 = loom.semaphore_take %47 : memref<64x64x128xf16> -> memref<64x64x128xf16>
        linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%24, %46 : memref<64x64x128xf16>, memref<64x64x128xf16, strided<[?, ?, ?], offset: ?>>) outs(%48 : memref<64x64x128xf16>) {
        ^bb0(%in: f16, %in_4: f16, %out: f16):
          %49 = arith.divf %in, %in_4 : f16
          linalg.yield %49 : f16
        }
        loom.semaphore_give %30 : memref<64x64x128xf16>
        loom.semaphore_give %24 : memref<64x64x128xf16>
        %reinterpret_cast_3 = memref.reinterpret_cast %arg3 to offset: [%22], sizes: [64, 64, 128], strides: [524288, 128, 1] : memref<32x4096x128xf16> to memref<64x64x128xf16, strided<[524288, 128, 1], offset: ?>>
        loom.copy %48, %reinterpret_cast_3 src_mem_space @mem_L1 dst_mem_space @mem_DRAM, area : [1, 1] region : (UL : [%arg4, %arg5], LR : [%arg4, %arg5]) : memref<64x64x128xf16> to memref<64x64x128xf16, strided<[524288, 128, 1], offset: ?>>
        loom.semaphore_give %48 : memref<64x64x128xf16>
        scf.reduce 
      } {loom.block_syms = [@tile_m, @tile_m], loom.iter_types = [#loom.iter_type<spatial>, #loom.iter_type<spatial>], loom.logical_levels = [0, 0], loom.physical_dims = [@dim_x, @dim_y]}
      return
    }
  }
}
