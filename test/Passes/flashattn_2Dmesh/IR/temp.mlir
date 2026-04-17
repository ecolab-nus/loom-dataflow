module {
  %0 = df.mat "FPU" {shape = [32, 32, 32], throughput = 128}
  %1 = df.vec "SFPU" {shape = [32]}
  %2 = df.spatial_dim "x", 8
  %3 = df.spatial_dim "y", 8
  %4 = df.core "core" {scaleout=(%2, %3) , scalein=(%0, %1, [8, 1])}
  %5 = df.memory "L1" {scaleout=(%2, %3) , size = 1499136, bandwidth = 15}
  %6 = df.mux %4 : !df.compute, %5 : !df.memory  {map = affine_map<(d0, d1) -> (d0, d1)>}
  %7 = df.interconnects "horizontal_links" %5 : !df.memory, %5 : !df.memory  {bandwidth = 128 : i64, map = affine_map<(d0, d1) -> ((d0 + 1) mod 8, d1)>, spatial_dims = [@y]} : !df.interconnect
  %8 = df.interconnects "vertical_links" %5 : !df.memory, %5 : !df.memory  {bandwidth = 128 : i64, map = affine_map<(d0, d1) -> (d0, (d1 + 1) mod 8)>, spatial_dims = [@x]} : !df.interconnect
  %9 = df.spatial_dim "d", 4
  %10 = df.memory "DRAM" {scaleout=(%9) , size = 34359738368, bandwidth = 288}
  %11 = df.interconnects "NoC" %5 : !df.memory, %10 : !df.memory  {map = affine_map<(d0, d1) -> (d0 ceildiv 4 + (d1 ceildiv 4) * 2)>} : !df.interconnect
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
          %33 = loom.semaphore_take %29 : memref<1x32xf16> -> memref<1x32xf16>
          linalg.fill ins(%cst_1 : f16) outs(%33 : memref<1x32xf16>)
          %34 = loom.alloc [1, 32] on @L1 : memref<1x32xf16>
          %35 = loom.semaphore_take %34 : memref<1x32xf16> -> memref<1x32xf16>
          linalg.fill ins(%cst_2 : f16) outs(%35 : memref<1x32xf16>)
          %36 = loom.alloc [1, 32, 128] on @L1 : memref<1x32x128xf16>
          %37 = loom.semaphore_take %36 : memref<1x32x128xf16> -> memref<1x32x128xf16>
          %38 = loom.alloc [1, 32] on @L1 : memref<1x32xf16>
          %39 = loom.semaphore_take %38 : memref<1x32xf16> -> memref<1x32xf16>
          %40 = loom.alloc [1, 32] on @L1 : memref<1x32xf16>
          %41 = loom.semaphore_take %40 : memref<1x32xf16> -> memref<1x32xf16>
          %42 = loom.alloc [1, 32] on @L1 : memref<1x32xf16>
          %43 = loom.semaphore_take %42 : memref<1x32xf16> -> memref<1x32xf16>
          %44 = loom.alloc [1, 128, 64] on @L1 : memref<1x128x64xf16>
          %45 = loom.semaphore_take %44 : memref<1x128x64xf16> -> memref<1x128x64xf16>
          %46 = loom.alloc [1, 32, 64] on @L1 : memref<1x32x64xf16>
          %47 = loom.semaphore_take %46 : memref<1x32x64xf16> -> memref<1x32x64xf16>
          %48 = loom.alloc [1, 64, 128] on @L1 : memref<1x64x128xf16>
          %49 = loom.semaphore_take %48 : memref<1x64x128xf16> -> memref<1x64x128xf16>
          scf.for %arg7 = %c0 to %c16 step %c1 {
            %55 = arith.muli %arg7, %c64 : index
            %56 = arith.addi %26, %55 : index
            %57 = arith.muli %21, %c1048576 overflow<nsw> : index
            %58 = arith.addi %57, %56 : index
            %reinterpret_cast_4 = memref.reinterpret_cast %arg0 to offset: [%58], sizes: [1, 128, 64], strides: [1048576, 8192, 1] : memref<16x128x8192xf16> to memref<1x128x64xf16, strided<[1048576, 8192, 1], offset: ?>>
            loom.copy %reinterpret_cast_4, %45 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] region : (UL : [%arg5, %arg4], LR : [%arg5, %arg4]) : memref<1x128x64xf16, strided<[1048576, 8192, 1], offset: ?>> to memref<1x128x64xf16>
            loom.batch_matmul ins(%24, %45 : memref<1x32x128xf16>, memref<1x128x64xf16>) outs(%47 : memref<1x32x64xf16>)
            loom.semaphore_give %45 : memref<1x128x64xf16>
            linalg.fill ins(%cst_2 : f16) outs(%39 : memref<1x32xf16>)
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%47 : memref<1x32x64xf16>) outs(%39 : memref<1x32xf16>) {
            ^bb0(%in: f16, %out: f16):
              %61 = arith.maximumf %in, %out : f16
              linalg.yield %61 : f16
            }
            linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%35, %39 : memref<1x32xf16>, memref<1x32xf16>) outs(%39 : memref<1x32xf16>) {
            ^bb0(%in: f16, %in_6: f16, %out: f16):
              %61 = arith.mulf %in_6, %cst_3 : f16
              %62 = arith.cmpf ogt, %in, %61 : f16
              %63 = arith.select %62, %in, %61 : f16
              linalg.yield %63 : f16
            }
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%47, %39 : memref<1x32x64xf16>, memref<1x32xf16>) outs(%47 : memref<1x32x64xf16>) {
            ^bb0(%in: f16, %in_6: f16, %out: f16):
              %61 = arith.mulf %in, %cst_3 : f16
              %62 = arith.subf %61, %in_6 : f16
              %63 = math.powf %cst, %62 : f16
              linalg.yield %63 : f16
            }
            linalg.fill ins(%cst_0 : f16) outs(%41 : memref<1x32xf16>)
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%47 : memref<1x32x64xf16>) outs(%41 : memref<1x32xf16>) {
            ^bb0(%in: f16, %out: f16):
              %61 = arith.addf %in, %out : f16
              linalg.yield %61 : f16
            }
            linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%35, %39 : memref<1x32xf16>, memref<1x32xf16>) outs(%43 : memref<1x32xf16>) {
            ^bb0(%in: f16, %in_6: f16, %out: f16):
              %61 = arith.subf %in, %in_6 : f16
              %62 = math.powf %cst, %61 : f16
              linalg.yield %62 : f16
            }
            linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%33, %43, %41 : memref<1x32xf16>, memref<1x32xf16>, memref<1x32xf16>) outs(%33 : memref<1x32xf16>) {
            ^bb0(%in: f16, %in_6: f16, %in_7: f16, %out: f16):
              %61 = arith.mulf %in, %in_6 : f16
              %62 = arith.addf %61, %in_7 : f16
              linalg.yield %62 : f16
            }
            loom.semaphore_give %41 : memref<1x32xf16>
            %59 = arith.muli %56, %c128 overflow<nsw> : index
            %60 = arith.addi %57, %59 : index
            %reinterpret_cast_5 = memref.reinterpret_cast %arg1 to offset: [%60], sizes: [1, 64, 128], strides: [1048576, 128, 1] : memref<16x8192x128xf16> to memref<1x64x128xf16, strided<[1048576, 128, 1], offset: ?>>
            loom.copy %reinterpret_cast_5, %49 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] region : (UL : [%arg5, %arg4], LR : [%arg5, %arg4]) : memref<1x64x128xf16, strided<[1048576, 128, 1], offset: ?>> to memref<1x64x128xf16>
            loom.batch_matmul ins(%47, %49 : memref<1x32x64xf16>, memref<1x64x128xf16>) outs(%37 : memref<1x32x128xf16>)
            loom.semaphore_give %49 : memref<1x64x128xf16>
            loom.semaphore_give %47 : memref<1x32x64xf16>
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%37, %28, %43 : memref<1x32x128xf16>, memref<1x32x128xf16>, memref<1x32xf16>) outs(%28 : memref<1x32x128xf16>) {
            ^bb0(%in: f16, %in_6: f16, %in_7: f16, %out: f16):
              %61 = arith.mulf %in_6, %in_7 : f16
              %62 = arith.addf %in, %61 : f16
              linalg.yield %62 : f16
            }
            loom.semaphore_give %43 : memref<1x32xf16>
            loom.semaphore_give %37 : memref<1x32x128xf16>
            linalg.copy ins(%39 : memref<1x32xf16>) outs(%35 : memref<1x32xf16>)
            loom.semaphore_give %39 : memref<1x32xf16>
          } {loom.iter_type = #loom.iter_type<sequential>}
          loom.semaphore_give %24 : memref<1x32x128xf16>
          linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%33, %35 : memref<1x32xf16>, memref<1x32xf16>) outs(%32 : memref<1x32xf16>) {
          ^bb0(%in: f16, %in_4: f16, %out: f16):
            %55 = math.log2 %in : f16
            %56 = arith.addf %55, %in_4 : f16
            linalg.yield %56 : f16
          }
          loom.semaphore_give %33 : memref<1x32xf16>
          loom.semaphore_give %35 : memref<1x32xf16>
          %50 = loom.alloc [8, 1, 32] on @L1 : memref<8x1x32xf16>
          %51 = loom.semaphore_take %50 : memref<8x1x32xf16> -> memref<8x1x32xf16>
          loom.gather ins(%32 : memref<1x32xf16>) outs(%51 : memref<8x1x32xf16>) across(%arg5 : index) region : (UL : [%c0, %arg4], LR : [%c7, %arg4])
          loom.semaphore_give %32 : memref<1x32xf16>
          %52 = loom.alloc [8, 1, 32, 128] on @L1 : memref<8x1x32x128xf16>
          %53 = loom.semaphore_take %52 : memref<8x1x32x128xf16> -> memref<8x1x32x128xf16>
          loom.gather ins(%28 : memref<1x32x128xf16>) outs(%53 : memref<8x1x32x128xf16>) across(%arg5 : index) region : (UL : [%c0, %arg4], LR : [%c7, %arg4])
          loom.semaphore_give %28 : memref<1x32x128xf16>
          %54 = arith.cmpi eq, %arg5, %c0 : index
          scf.if %54 {
            linalg.fill ins(%cst_2 : f16) outs(%31 : memref<1x32xf16>)
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d1, d2)>], iterator_types = ["reduction", "parallel", "parallel"]} ins(%51 : memref<8x1x32xf16>) outs(%31 : memref<1x32xf16>) {
            ^bb0(%in: f16, %out: f16):
              %55 = arith.maximumf %in, %out : f16
              linalg.yield %55 : f16
            }
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%51, %31 : memref<8x1x32xf16>, memref<1x32xf16>) outs(%51 : memref<8x1x32xf16>) {
            ^bb0(%in: f16, %in_5: f16, %out: f16):
              %55 = arith.subf %in, %in_5 : f16
              %56 = math.powf %cst, %55 : f16
              linalg.yield %56 : f16
            }
            loom.semaphore_give %31 : memref<1x32xf16>
            linalg.fill ins(%cst_0 : f16) outs(%30 : memref<1x32xf16>)
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d1, d2)>], iterator_types = ["reduction", "parallel", "parallel"]} ins(%51 : memref<8x1x32xf16>) outs(%30 : memref<1x32xf16>) {
            ^bb0(%in: f16, %out: f16):
              %55 = arith.addf %in, %out : f16
              linalg.yield %55 : f16
            }
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%51, %30 : memref<8x1x32xf16>, memref<1x32xf16>) outs(%51 : memref<8x1x32xf16>) {
            ^bb0(%in: f16, %in_5: f16, %out: f16):
              %55 = arith.divf %in, %in_5 : f16
              linalg.yield %55 : f16
            }
            loom.semaphore_give %30 : memref<1x32xf16>
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%53, %51 : memref<8x1x32x128xf16>, memref<8x1x32xf16>) outs(%53 : memref<8x1x32x128xf16>) {
            ^bb0(%in: f16, %in_5: f16, %out: f16):
              %55 = arith.mulf %in, %in_5 : f16
              linalg.yield %55 : f16
            }
            loom.semaphore_give %51 : memref<8x1x32xf16>
            linalg.fill ins(%cst_0 : f16) outs(%23 : memref<1x32x128xf16>)
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1, d2, d3)>], iterator_types = ["reduction", "parallel", "parallel", "parallel"]} ins(%53 : memref<8x1x32x128xf16>) outs(%23 : memref<1x32x128xf16>) {
            ^bb0(%in: f16, %out: f16):
              %55 = arith.addf %in, %out : f16
              linalg.yield %55 : f16
            }
            loom.semaphore_give %53 : memref<8x1x32x128xf16>
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