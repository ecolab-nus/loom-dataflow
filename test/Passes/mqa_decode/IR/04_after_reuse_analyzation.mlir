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
  module attributes {loom.tile_b = {is_reduction = false, upper_bound = 16 : index}, loom.tile_n = {is_reduction = false, upper_bound = 64 : index}, loom.tile_s = {is_reduction = false, upper_bound = 8192 : index}} {
    func.func @flash_decode__x8_y8__d0i1_d1i0__f01(%arg0: memref<16x128x8192xf16>, %arg1: memref<16x8192x128xf16>, %arg2: memref<16x32x128xf16>, %arg3: memref<16x32x128xf16>) {
      %c7 = arith.constant 7 : index
      %c8 = arith.constant 8 : index
      %cst = arith.constant 2.000000e+00 : f16
      %c1 = arith.constant 1 : index
      %c0 = arith.constant 0 : index
      %cst_0 = arith.constant 0.000000e+00 : f16
      %cst_1 = arith.constant 1.000000e+00 : f16
      %cst_2 = arith.constant 0xFC00 : f16
      %cst_3 = arith.constant 1.275630e-01 : f16
      %c8192 = arith.constant 8192 : index
      %c16 = arith.constant 16 : index
      %20 = loom.sym @tile_b {upper_bound = 16 : index} : index
      %21 = loom.sym @tile_s {upper_bound = 8192 : index} : index
      %22 = loom.sym @tile_n {upper_bound = 64 : index} : index
      %23 = arith.ceildivui %c16, %20 : index
      %24 = arith.ceildivui %c8192, %21 : index
      affine.parallel (%arg4) = (0) to (8) {
        affine.parallel (%arg5) = (0) to (8) {
          %25 = arith.ceildivui %23, %c8 : index
          scf.for %arg6 = %c0 to %25 step %c1 {
            %26 = arith.ceildivui %24, %c8 : index
            scf.for %arg7 = %c0 to %26 step %c1 {
              %27 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 8)>(%arg4, %arg6)
              %28 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 8)>(%arg5, %arg7)
              %29 = arith.muli %27, %20 : index
              %30 = loom.alloc [%20, 32, 128] on @L1 : memref<?x32x128xf16>
              %31 = loom.semaphore_take %30 : memref<?x32x128xf16> -> memref<?x32x128xf16>
              %32 = loom.init_tensor %31[%20, 32, 128] : memref<?x32x128xf16> -> tensor<?x32x128xf16>
              %33 = loom.semaphore_take %30 : memref<?x32x128xf16> -> memref<?x32x128xf16>
              %34 = loom.subview %arg3[%29, 0, 0] [%20, 32, 128] [1, 1, 1], reuse : [seq = false, spat = true, temp = true] : memref<16x32x128xf16> to memref<?x32x128xf16, strided<[4096, 128, 1], offset: ?>>
              loom.copy %34, %33 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<?x32x128xf16, strided<[4096, 128, 1], offset: ?>> to memref<?x32x128xf16>
              %35 = loom.bufferize_to_tensor %33[%20, 32, 128] : memref<?x32x128xf16> -> tensor<?x32x128xf16>
              %36 = arith.muli %28, %21 : index
              %37 = arith.addi %36, %21 : index
              %38 = arith.ceildivui %21, %22 : index
              %39 = loom.alloc [%20, 32, 128] on @L1 : memref<?x32x128xf16>
              %40 = loom.semaphore_take %39 : memref<?x32x128xf16> -> memref<?x32x128xf16>
              %41 = loom.init_tensor %40[%20, 32, 128] : memref<?x32x128xf16> -> tensor<?x32x128xf16>
              %42 = linalg.fill ins(%cst_0 : f16) outs(%41 : tensor<?x32x128xf16>) -> tensor<?x32x128xf16>
              %43 = loom.alloc [%20, 32] on @L1 : memref<?x32xf16>
              %44 = loom.semaphore_take %43 : memref<?x32xf16> -> memref<?x32xf16>
              %45 = loom.init_tensor %44[%20, 32] : memref<?x32xf16> -> tensor<?x32xf16>
              %46 = loom.semaphore_take %43 : memref<?x32xf16> -> memref<?x32xf16>
              %47 = loom.init_tensor %46[%20, 32] : memref<?x32xf16> -> tensor<?x32xf16>
              %48 = loom.semaphore_take %43 : memref<?x32xf16> -> memref<?x32xf16>
              %49 = loom.init_tensor %48[%20, 32] : memref<?x32xf16> -> tensor<?x32xf16>
              %50 = loom.semaphore_take %43 : memref<?x32xf16> -> memref<?x32xf16>
              %51 = loom.init_tensor %50[%20, 32] : memref<?x32xf16> -> tensor<?x32xf16>
              %52 = linalg.fill ins(%cst_1 : f16) outs(%51 : tensor<?x32xf16>) -> tensor<?x32xf16>
              %53 = loom.alloc [%20, 32] on @L1 : memref<?x32xf16>
              %54 = loom.semaphore_take %53 : memref<?x32xf16> -> memref<?x32xf16>
              %55 = loom.init_tensor %54[%20, 32] : memref<?x32xf16> -> tensor<?x32xf16>
              %56 = linalg.fill ins(%cst_2 : f16) outs(%55 : tensor<?x32xf16>) -> tensor<?x32xf16>
              %57 = loom.alloc [%20, 32, 128] on @L1 : memref<?x32x128xf16>
              %58 = loom.semaphore_take %57 : memref<?x32x128xf16> -> memref<?x32x128xf16>
              %59 = loom.init_tensor %58[%20, 32, 128] : memref<?x32x128xf16> -> tensor<?x32x128xf16>
              %60 = loom.alloc [%20, 32] on @L1 : memref<?x32xf16>
              %61 = loom.semaphore_take %60 : memref<?x32xf16> -> memref<?x32xf16>
              %62 = loom.init_tensor %61[%20, 32] : memref<?x32xf16> -> tensor<?x32xf16>
              %63 = loom.alloc [%20, 32] on @L1 : memref<?x32xf16>
              %64 = loom.semaphore_take %63 : memref<?x32xf16> -> memref<?x32xf16>
              %65 = loom.init_tensor %64[%20, 32] : memref<?x32xf16> -> tensor<?x32xf16>
              %66 = loom.alloc [%20, 32] on @L1 : memref<?x32xf16>
              %67 = loom.semaphore_take %66 : memref<?x32xf16> -> memref<?x32xf16>
              %68 = loom.init_tensor %67[%20, 32] : memref<?x32xf16> -> tensor<?x32xf16>
              %69 = loom.alloc [%20, 32, %22] on @L1 : memref<?x32x?xf16>
              %70 = loom.semaphore_take %69 : memref<?x32x?xf16> -> memref<?x32x?xf16>
              %71 = loom.init_tensor %70[%20, 32, %22] : memref<?x32x?xf16> -> tensor<?x32x?xf16>
              %72:3 = scf.for %arg8 = %c0 to %38 step %c1 iter_args(%arg9 = %56, %arg10 = %52, %arg11 = %42) -> (tensor<?x32xf16>, tensor<?x32xf16>, tensor<?x32x128xf16>) {
                %81 = arith.muli %arg8, %22 : index
                %82 = arith.addi %36, %81 : index
                %83 = arith.addi %82, %22 : index
                %84 = arith.cmpi ult, %83, %37 : index
                %85 = arith.select %84, %83, %37 : index
                %86 = arith.subi %85, %82 : index
                %87 = loom.alloc [%20, 128, %86] on @L1 : memref<?x128x?xf16>
                %88 = loom.semaphore_take %87 : memref<?x128x?xf16> -> memref<?x128x?xf16>
                %89 = loom.subview %arg0[%29, 0, %82] [%20, 128, %86] [1, 1, 1], reuse : [seq = false, spat = false, temp = false] : memref<16x128x8192xf16> to memref<?x128x?xf16, strided<[1048576, 8192, 1], offset: ?>>
                loom.copy %89, %88 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<?x128x?xf16, strided<[1048576, 8192, 1], offset: ?>> to memref<?x128x?xf16>
                %90 = loom.bufferize_to_tensor %88[%20, 128, %86] : memref<?x128x?xf16> -> tensor<?x128x?xf16>
                %91 = linalg.fill ins(%cst_0 : f16) outs(%71 : tensor<?x32x?xf16>) -> tensor<?x32x?xf16>
                %92 = linalg.batch_matmul ins(%35, %90 : tensor<?x32x128xf16>, tensor<?x128x?xf16>) outs(%91 : tensor<?x32x?xf16>) -> tensor<?x32x?xf16>
                loom.semaphore_give %88 : memref<?x128x?xf16>
                %93 = linalg.fill ins(%cst_2 : f16) outs(%62 : tensor<?x32xf16>) -> tensor<?x32xf16>
                %94 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%92 : tensor<?x32x?xf16>) outs(%93 : tensor<?x32xf16>) {
                ^bb0(%in: f16, %out: f16):
                  %109 = arith.maximumf %in, %out : f16
                  linalg.yield %109 : f16
                } -> tensor<?x32xf16>
                %95 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%arg9, %94 : tensor<?x32xf16>, tensor<?x32xf16>) outs(%62 : tensor<?x32xf16>) {
                ^bb0(%in: f16, %in_4: f16, %out: f16):
                  %109 = arith.mulf %in_4, %cst_3 : f16
                  %110 = arith.cmpf ogt, %in, %109 : f16
                  %111 = arith.select %110, %in, %109 : f16
                  linalg.yield %111 : f16
                } -> tensor<?x32xf16>
                %96 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%92, %95 : tensor<?x32x?xf16>, tensor<?x32xf16>) outs(%71 : tensor<?x32x?xf16>) {
                ^bb0(%in: f16, %in_4: f16, %out: f16):
                  %109 = arith.mulf %in, %cst_3 : f16
                  %110 = arith.subf %109, %in_4 : f16
                  %111 = math.powf %cst, %110 : f16
                  linalg.yield %111 : f16
                } -> tensor<?x32x?xf16>
                %97 = linalg.fill ins(%cst_0 : f16) outs(%65 : tensor<?x32xf16>) -> tensor<?x32xf16>
                %98 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%96 : tensor<?x32x?xf16>) outs(%97 : tensor<?x32xf16>) {
                ^bb0(%in: f16, %out: f16):
                  %109 = arith.addf %in, %out : f16
                  linalg.yield %109 : f16
                } -> tensor<?x32xf16>
                %99 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%arg9, %95 : tensor<?x32xf16>, tensor<?x32xf16>) outs(%68 : tensor<?x32xf16>) {
                ^bb0(%in: f16, %in_4: f16, %out: f16):
                  %109 = arith.subf %in, %in_4 : f16
                  %110 = math.powf %cst, %109 : f16
                  linalg.yield %110 : f16
                } -> tensor<?x32xf16>
                %100 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%arg10, %99, %98 : tensor<?x32xf16>, tensor<?x32xf16>, tensor<?x32xf16>) outs(%arg10 : tensor<?x32xf16>) {
                ^bb0(%in: f16, %in_4: f16, %in_5: f16, %out: f16):
                  %109 = arith.mulf %in, %in_4 : f16
                  %110 = arith.addf %109, %in_5 : f16
                  linalg.yield %110 : f16
                } -> tensor<?x32xf16>
                loom.semaphore_give %64 : memref<?x32xf16>
                %101 = loom.alloc [%20, %86, 128] on @L1 : memref<?x?x128xf16>
                %102 = loom.semaphore_take %101 : memref<?x?x128xf16> -> memref<?x?x128xf16>
                %103 = loom.subview %arg1[%29, %82, 0] [%20, %86, 128] [1, 1, 1], reuse : [seq = false, spat = false, temp = false] : memref<16x8192x128xf16> to memref<?x?x128xf16, strided<[1048576, 128, 1], offset: ?>>
                loom.copy %103, %102 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<?x?x128xf16, strided<[1048576, 128, 1], offset: ?>> to memref<?x?x128xf16>
                %104 = loom.bufferize_to_tensor %102[%20, %86, 128] : memref<?x?x128xf16> -> tensor<?x?x128xf16>
                %105 = linalg.fill ins(%cst_0 : f16) outs(%59 : tensor<?x32x128xf16>) -> tensor<?x32x128xf16>
                %106 = linalg.batch_matmul ins(%96, %104 : tensor<?x32x?xf16>, tensor<?x?x128xf16>) outs(%105 : tensor<?x32x128xf16>) -> tensor<?x32x128xf16>
                loom.semaphore_give %102 : memref<?x?x128xf16>
                loom.semaphore_give %70 : memref<?x32x?xf16>
                %107 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%106, %arg11, %99 : tensor<?x32x128xf16>, tensor<?x32x128xf16>, tensor<?x32xf16>) outs(%arg11 : tensor<?x32x128xf16>) {
                ^bb0(%in: f16, %in_4: f16, %in_5: f16, %out: f16):
                  %109 = arith.mulf %in_4, %in_5 : f16
                  %110 = arith.addf %in, %109 : f16
                  linalg.yield %110 : f16
                } -> tensor<?x32x128xf16>
                loom.semaphore_give %67 : memref<?x32xf16>
                loom.semaphore_give %58 : memref<?x32x128xf16>
                %108 = linalg.copy ins(%95 : tensor<?x32xf16>) outs(%arg9 : tensor<?x32xf16>) -> tensor<?x32xf16>
                loom.semaphore_give %61 : memref<?x32xf16>
                scf.yield %108, %100, %107 : tensor<?x32xf16>, tensor<?x32xf16>, tensor<?x32x128xf16>
              } {loom.iter_type = #loom.iter_type<sequential>}
              loom.semaphore_give %33 : memref<?x32x128xf16>
              %73 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%72#1, %72#0 : tensor<?x32xf16>, tensor<?x32xf16>) outs(%49 : tensor<?x32xf16>) {
              ^bb0(%in: f16, %in_4: f16, %out: f16):
                %81 = math.log2 %in : f16
                %82 = arith.addf %81, %in_4 : f16
                linalg.yield %82 : f16
              } -> tensor<?x32xf16>
              loom.semaphore_give %50 : memref<?x32xf16>
              loom.semaphore_give %54 : memref<?x32xf16>
              %74 = loom.alloc [%24, %20, 32] on @L1 : memref<?x?x32xf16>
              %75 = loom.semaphore_take %74 : memref<?x?x32xf16> -> memref<?x?x32xf16>
              %76 = loom.init_tensor %75[%24, %20, 32] : memref<?x?x32xf16> -> tensor<?x?x32xf16>
              %77 = loom.alloc [%24, %20, 32, 128] on @L1 : memref<?x?x32x128xf16>
              %78 = loom.semaphore_take %77 : memref<?x?x32x128xf16> -> memref<?x?x32x128xf16>
              %79 = loom.init_tensor %78[%24, %20, 32, 128] : memref<?x?x32x128xf16> -> tensor<?x?x32x128xf16>
              %80 = arith.cmpi eq, %arg5, %c0 : index
              scf.if %80 {
                %81 = loom.gather ins(%73 : tensor<?x32xf16>) outs(%76 : tensor<?x?x32xf16>) across(%arg5 : index) region : (UL : [%c0, %arg4], LR : [%c7, %arg4]) -> tensor<?x?x32xf16>
                loom.semaphore_give %48 : memref<?x32xf16>
                %82 = linalg.fill ins(%cst_2 : f16) outs(%47 : tensor<?x32xf16>) -> tensor<?x32xf16>
                %83 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d1, d2)>], iterator_types = ["reduction", "parallel", "parallel"]} ins(%81 : tensor<?x?x32xf16>) outs(%82 : tensor<?x32xf16>) {
                ^bb0(%in: f16, %out: f16):
                  %93 = arith.maximumf %in, %out : f16
                  linalg.yield %93 : f16
                } -> tensor<?x32xf16>
                %84 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%81, %83 : tensor<?x?x32xf16>, tensor<?x32xf16>) outs(%76 : tensor<?x?x32xf16>) {
                ^bb0(%in: f16, %in_4: f16, %out: f16):
                  %93 = arith.subf %in, %in_4 : f16
                  %94 = math.powf %cst, %93 : f16
                  linalg.yield %94 : f16
                } -> tensor<?x?x32xf16>
                loom.semaphore_give %46 : memref<?x32xf16>
                %85 = linalg.fill ins(%cst_0 : f16) outs(%45 : tensor<?x32xf16>) -> tensor<?x32xf16>
                %86 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d1, d2)>], iterator_types = ["reduction", "parallel", "parallel"]} ins(%84 : tensor<?x?x32xf16>) outs(%85 : tensor<?x32xf16>) {
                ^bb0(%in: f16, %out: f16):
                  %93 = arith.addf %in, %out : f16
                  linalg.yield %93 : f16
                } -> tensor<?x32xf16>
                %87 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%84, %86 : tensor<?x?x32xf16>, tensor<?x32xf16>) outs(%76 : tensor<?x?x32xf16>) {
                ^bb0(%in: f16, %in_4: f16, %out: f16):
                  %93 = arith.divf %in, %in_4 : f16
                  linalg.yield %93 : f16
                } -> tensor<?x?x32xf16>
                loom.semaphore_give %44 : memref<?x32xf16>
                %88 = loom.gather ins(%72#2 : tensor<?x32x128xf16>) outs(%79 : tensor<?x?x32x128xf16>) across(%arg5 : index) region : (UL : [%c0, %arg4], LR : [%c7, %arg4]) -> tensor<?x?x32x128xf16>
                %89 = linalg.fill ins(%cst_0 : f16) outs(%32 : tensor<?x32x128xf16>) -> tensor<?x32x128xf16>
                %90 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>, affine_map<(d0, d1, d2, d3) -> (d1, d2, d3)>], iterator_types = ["reduction", "parallel", "parallel", "parallel"]} ins(%88, %87 : tensor<?x?x32x128xf16>, tensor<?x?x32xf16>) outs(%89 : tensor<?x32x128xf16>) {
                ^bb0(%in: f16, %in_4: f16, %out: f16):
                  %93 = arith.mulf %in, %in_4 : f16
                  %94 = arith.addf %93, %out : f16
                  linalg.yield %94 : f16
                } -> tensor<?x32x128xf16>
                loom.semaphore_give %78 : memref<?x?x32x128xf16>
                loom.semaphore_give %75 : memref<?x?x32xf16>
                %91 = loom.subview %arg2[%29, 0, 0] [%20, 32, 128] [1, 1, 1], reuse : [seq = false, spat = true, temp = true] : memref<16x32x128xf16> to memref<?x32x128xf16, strided<[4096, 128, 1], offset: ?>>
                %92 = loom.bufferize_to_memref %90 : tensor<?x32x128xf16> -> memref<?x32x128xf16>
                loom.copy %92, %91 src_mem_space @mem_L1 dst_mem_space @mem_DRAM, broadcast : [1, 1] : memref<?x32x128xf16> to memref<?x32x128xf16, strided<[4096, 128, 1], offset: ?>>
                loom.semaphore_give %31 : memref<?x32x128xf16>
              }
              loom.semaphore_give %40 : memref<?x32x128xf16>
            } {loom.block_sym = @tile_s, loom.iter_type = #loom.iter_type<temporal>}
          } {loom.block_sym = @tile_b, loom.iter_type = #loom.iter_type<temporal>}
        } {loom.block_sym = @tile_s, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 0 : i64, loom.physical_dim = @dim_x}
      } {loom.block_sym = @tile_b, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 0 : i64, loom.physical_dim = @dim_y}
      return
    }
  }
}
