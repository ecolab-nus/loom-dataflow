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
      %c0 = arith.constant 0 : index
      %cst = arith.constant 0.000000e+00 : f16
      %cst_0 = arith.constant 1.000000e+00 : f16
      %cst_1 = arith.constant 0xFC00 : f16
      %c1 = arith.constant 1 : index
      %cst_2 = arith.constant 8.837890e-02 : f16
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
              %34 = loom.init_tensor %33[%20, 32, 128] : memref<?x32x128xf16> -> tensor<?x32x128xf16>
              %35 = loom.semaphore_take %30 : memref<?x32x128xf16> -> memref<?x32x128xf16>
              %36 = loom.subview %arg3[%29, 0, 0] [%20, 32, 128] [1, 1, 1], reuse : [seq = false, spat = false, temp = false] : memref<16x32x128xf16> to memref<?x32x128xf16, strided<[4096, 128, 1], offset: ?>>
              loom.copy %36, %35 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<?x32x128xf16, strided<[4096, 128, 1], offset: ?>> to memref<?x32x128xf16>
              %37 = loom.bufferize_to_tensor %35[%20, 32, 128] : memref<?x32x128xf16> -> tensor<?x32x128xf16>
              %38 = arith.muli %28, %21 : index
              %39 = arith.ceildivui %21, %22 : index
              %40 = loom.alloc [%20, 32, 128] on @L1 : memref<?x32x128xf16>
              %41 = loom.semaphore_take %40 : memref<?x32x128xf16> -> memref<?x32x128xf16>
              %42 = loom.init_tensor %41[%20, 32, 128] : memref<?x32x128xf16> -> tensor<?x32x128xf16>
              %43 = loom.semaphore_take %40 : memref<?x32x128xf16> -> memref<?x32x128xf16>
              %44 = loom.init_tensor %43[%20, 32, 128] : memref<?x32x128xf16> -> tensor<?x32x128xf16>
              %45 = loom.semaphore_take %40 : memref<?x32x128xf16> -> memref<?x32x128xf16>
              %46 = loom.init_tensor %45[%20, 32, 128] : memref<?x32x128xf16> -> tensor<?x32x128xf16>
              %47 = linalg.fill ins(%cst : f16) outs(%46 : tensor<?x32x128xf16>) -> tensor<?x32x128xf16>
              %48 = loom.alloc [%20, 32, 1] on @L1 : memref<?x32x1xf16>
              %49 = loom.semaphore_take %48 : memref<?x32x1xf16> -> memref<?x32x1xf16>
              %50 = loom.init_tensor %49[%20, 32, 1] : memref<?x32x1xf16> -> tensor<?x32x1xf16>
              %51 = loom.semaphore_take %48 : memref<?x32x1xf16> -> memref<?x32x1xf16>
              %52 = loom.init_tensor %51[%20, 32, 1] : memref<?x32x1xf16> -> tensor<?x32x1xf16>
              %53 = loom.semaphore_take %48 : memref<?x32x1xf16> -> memref<?x32x1xf16>
              %54 = loom.init_tensor %53[%20, 32, 1] : memref<?x32x1xf16> -> tensor<?x32x1xf16>
              %55 = loom.semaphore_take %48 : memref<?x32x1xf16> -> memref<?x32x1xf16>
              %56 = loom.init_tensor %55[%20, 32, 1] : memref<?x32x1xf16> -> tensor<?x32x1xf16>
              %57 = linalg.fill ins(%cst_0 : f16) outs(%56 : tensor<?x32x1xf16>) -> tensor<?x32x1xf16>
              %58 = loom.alloc [%20, 32, 1] on @L1 : memref<?x32x1xf16>
              %59 = loom.semaphore_take %58 : memref<?x32x1xf16> -> memref<?x32x1xf16>
              %60 = loom.init_tensor %59[%20, 32, 1] : memref<?x32x1xf16> -> tensor<?x32x1xf16>
              %61 = loom.semaphore_take %58 : memref<?x32x1xf16> -> memref<?x32x1xf16>
              %62 = loom.init_tensor %61[%20, 32, 1] : memref<?x32x1xf16> -> tensor<?x32x1xf16>
              %63 = linalg.fill ins(%cst_1 : f16) outs(%62 : tensor<?x32x1xf16>) -> tensor<?x32x1xf16>
              %64 = loom.alloc [%20, 32, 128] on @L1 : memref<?x32x128xf16>
              %65 = loom.semaphore_take %64 : memref<?x32x128xf16> -> memref<?x32x128xf16>
              %66 = loom.init_tensor %65[%20, 32, 128] : memref<?x32x128xf16> -> tensor<?x32x128xf16>
              %67 = loom.alloc [%20, 32, 1] on @L1 : memref<?x32x1xf16>
              %68 = loom.semaphore_take %67 : memref<?x32x1xf16> -> memref<?x32x1xf16>
              %69 = loom.init_tensor %68[%20, 32, 1] : memref<?x32x1xf16> -> tensor<?x32x1xf16>
              %70 = loom.alloc [%20, 32, 1] on @L1 : memref<?x32x1xf16>
              %71 = loom.semaphore_take %70 : memref<?x32x1xf16> -> memref<?x32x1xf16>
              %72 = loom.init_tensor %71[%20, 32, 1] : memref<?x32x1xf16> -> tensor<?x32x1xf16>
              %73 = loom.alloc [%20, 32, 1] on @L1 : memref<?x32x1xf16>
              %74 = loom.semaphore_take %73 : memref<?x32x1xf16> -> memref<?x32x1xf16>
              %75 = loom.init_tensor %74[%20, 32, 1] : memref<?x32x1xf16> -> tensor<?x32x1xf16>
              %76 = loom.alloc [%20, 128, %22] on @L1 : memref<?x128x?xf16>
              %77 = loom.semaphore_take %76 : memref<?x128x?xf16> -> memref<?x128x?xf16>
              %78 = loom.alloc [%20, 32, %22] on @L1 : memref<?x32x?xf16>
              %79 = loom.semaphore_take %78 : memref<?x32x?xf16> -> memref<?x32x?xf16>
              %80 = loom.init_tensor %79[%20, 32, %22] : memref<?x32x?xf16> -> tensor<?x32x?xf16>
              %81 = loom.alloc [%20, 32, 32] on @L1 : memref<?x32x32xf16>
              %82 = loom.semaphore_take %81 : memref<?x32x32xf16> -> memref<?x32x32xf16>
              %83 = loom.init_tensor %82[%20, 32, 32] : memref<?x32x32xf16> -> tensor<?x32x32xf16>
              %84 = loom.semaphore_take %81 : memref<?x32x32xf16> -> memref<?x32x32xf16>
              %85 = loom.init_tensor %84[%20, 32, 32] : memref<?x32x32xf16> -> tensor<?x32x32xf16>
              %86 = loom.semaphore_take %81 : memref<?x32x32xf16> -> memref<?x32x32xf16>
              %87 = loom.init_tensor %86[%20, 32, 32] : memref<?x32x32xf16> -> tensor<?x32x32xf16>
              %88 = loom.alloc [%20, %22, 128] on @L1 : memref<?x?x128xf16>
              %89 = loom.semaphore_take %88 : memref<?x?x128xf16> -> memref<?x?x128xf16>
              %90:3 = scf.for %arg8 = %c0 to %39 step %c1 iter_args(%arg9 = %63, %arg10 = %57, %arg11 = %47) -> (tensor<?x32x1xf16>, tensor<?x32x1xf16>, tensor<?x32x128xf16>) {
                %114 = arith.muli %arg8, %22 : index
                %115 = arith.addi %38, %114 : index
                %116 = loom.subview %arg0[%29, 0, %115] [%20, 128, %22] [1, 1, 1], reuse : [seq = false, spat = false, temp = false] : memref<16x128x8192xf16> to memref<?x128x?xf16, strided<[1048576, 8192, 1], offset: ?>>
                loom.copy %116, %77 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<?x128x?xf16, strided<[1048576, 8192, 1], offset: ?>> to memref<?x128x?xf16>
                %117 = loom.bufferize_to_tensor %77[%20, 128, %22] : memref<?x128x?xf16> -> tensor<?x128x?xf16>
                %118 = linalg.fill ins(%cst : f16) outs(%80 : tensor<?x32x?xf16>) -> tensor<?x32x?xf16>
                %119 = linalg.batch_matmul ins(%37, %117 : tensor<?x32x128xf16>, tensor<?x128x?xf16>) outs(%118 : tensor<?x32x?xf16>) -> tensor<?x32x?xf16>
                loom.semaphore_give %77 : memref<?x128x?xf16>
                %120 = linalg.fill ins(%cst_1 : f16) outs(%69 : tensor<?x32x1xf16>) -> tensor<?x32x1xf16>
                %121 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, 0)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%119 : tensor<?x32x?xf16>) outs(%120 : tensor<?x32x1xf16>) {
                ^bb0(%in: f16, %out: f16):
                  %136 = arith.maximumf %in, %out : f16
                  linalg.yield %136 : f16
                } -> tensor<?x32x1xf16>
                %122 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%arg9, %121 : tensor<?x32x1xf16>, tensor<?x32x1xf16>) outs(%69 : tensor<?x32x1xf16>) {
                ^bb0(%in: f16, %in_3: f16, %out: f16):
                  %136 = arith.mulf %in_3, %cst_2 : f16
                  %137 = arith.cmpf ogt, %in, %136 : f16
                  %138 = arith.select %137, %in, %136 : f16
                  linalg.yield %138 : f16
                } -> tensor<?x32x1xf16>
                %123 = loom.broadcast ins(%122 : tensor<?x32x1xf16>) outs(%87 : tensor<?x32x32xf16>) dim(2) -> tensor<?x32x?xf16>
                %124 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%119, %123 : tensor<?x32x?xf16>, tensor<?x32x?xf16>) outs(%80 : tensor<?x32x?xf16>) {
                ^bb0(%in: f16, %in_3: f16, %out: f16):
                  %136 = arith.mulf %in, %cst_2 : f16
                  %137 = arith.subf %136, %in_3 : f16
                  %138 = math.exp %137 : f16
                  linalg.yield %138 : f16
                } -> tensor<?x32x?xf16>
                loom.semaphore_give %86 : memref<?x32x32xf16>
                %125 = linalg.fill ins(%cst : f16) outs(%72 : tensor<?x32x1xf16>) -> tensor<?x32x1xf16>
                %126 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, 0)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%124 : tensor<?x32x?xf16>) outs(%125 : tensor<?x32x1xf16>) {
                ^bb0(%in: f16, %out: f16):
                  %136 = arith.addf %in, %out : f16
                  linalg.yield %136 : f16
                } -> tensor<?x32x1xf16>
                %127 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%arg9, %122 : tensor<?x32x1xf16>, tensor<?x32x1xf16>) outs(%75 : tensor<?x32x1xf16>) {
                ^bb0(%in: f16, %in_3: f16, %out: f16):
                  %136 = arith.subf %in, %in_3 : f16
                  %137 = math.exp %136 : f16
                  linalg.yield %137 : f16
                } -> tensor<?x32x1xf16>
                %128 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%arg10, %127, %126 : tensor<?x32x1xf16>, tensor<?x32x1xf16>, tensor<?x32x1xf16>) outs(%arg10 : tensor<?x32x1xf16>) {
                ^bb0(%in: f16, %in_3: f16, %in_4: f16, %out: f16):
                  %136 = arith.mulf %in, %in_3 : f16
                  %137 = arith.addf %136, %in_4 : f16
                  linalg.yield %137 : f16
                } -> tensor<?x32x1xf16>
                loom.semaphore_give %71 : memref<?x32x1xf16>
                %129 = loom.broadcast ins(%127 : tensor<?x32x1xf16>) outs(%85 : tensor<?x32x32xf16>) dim(2) -> tensor<?x32x128xf16>
                loom.semaphore_give %74 : memref<?x32x1xf16>
                %130 = loom.subview %arg1[%29, %115, 0] [%20, %22, 128] [1, 1, 1], reuse : [seq = false, spat = false, temp = false] : memref<16x8192x128xf16> to memref<?x?x128xf16, strided<[1048576, 128, 1], offset: ?>>
                loom.copy %130, %89 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<?x?x128xf16, strided<[1048576, 128, 1], offset: ?>> to memref<?x?x128xf16>
                %131 = loom.bufferize_to_tensor %89[%20, %22, 128] : memref<?x?x128xf16> -> tensor<?x?x128xf16>
                %132 = linalg.fill ins(%cst : f16) outs(%66 : tensor<?x32x128xf16>) -> tensor<?x32x128xf16>
                %133 = linalg.batch_matmul ins(%124, %131 : tensor<?x32x?xf16>, tensor<?x?x128xf16>) outs(%132 : tensor<?x32x128xf16>) -> tensor<?x32x128xf16>
                loom.semaphore_give %89 : memref<?x?x128xf16>
                loom.semaphore_give %79 : memref<?x32x?xf16>
                %134 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%133, %arg11, %129 : tensor<?x32x128xf16>, tensor<?x32x128xf16>, tensor<?x32x128xf16>) outs(%arg11 : tensor<?x32x128xf16>) {
                ^bb0(%in: f16, %in_3: f16, %in_4: f16, %out: f16):
                  %136 = arith.mulf %in_3, %in_4 : f16
                  %137 = arith.addf %in, %136 : f16
                  linalg.yield %137 : f16
                } -> tensor<?x32x128xf16>
                loom.semaphore_give %84 : memref<?x32x32xf16>
                loom.semaphore_give %65 : memref<?x32x128xf16>
                %135 = linalg.copy ins(%122 : tensor<?x32x1xf16>) outs(%arg9 : tensor<?x32x1xf16>) -> tensor<?x32x1xf16>
                loom.semaphore_give %68 : memref<?x32x1xf16>
                scf.yield %135, %128, %134 : tensor<?x32x1xf16>, tensor<?x32x1xf16>, tensor<?x32x128xf16>
              } {loom.iter_type = #loom.iter_type<sequential>}
              loom.semaphore_give %35 : memref<?x32x128xf16>
              %91 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%90#1, %90#0 : tensor<?x32x1xf16>, tensor<?x32x1xf16>) outs(%60 : tensor<?x32x1xf16>) {
              ^bb0(%in: f16, %in_3: f16, %out: f16):
                %114 = math.log %in : f16
                %115 = arith.addf %114, %in_3 : f16
                linalg.yield %115 : f16
              } -> tensor<?x32x1xf16>
              loom.semaphore_give %61 : memref<?x32x1xf16>
              %92 = loom.broadcast ins(%90#1 : tensor<?x32x1xf16>) outs(%83 : tensor<?x32x32xf16>) dim(2) -> tensor<?x32x128xf16>
              loom.semaphore_give %55 : memref<?x32x1xf16>
              %93 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%90#2, %92 : tensor<?x32x128xf16>, tensor<?x32x128xf16>) outs(%34 : tensor<?x32x128xf16>) {
              ^bb0(%in: f16, %in_3: f16, %out: f16):
                %114 = arith.divf %in, %in_3 : f16
                linalg.yield %114 : f16
              } -> tensor<?x32x128xf16>
              loom.semaphore_give %82 : memref<?x32x32xf16>
              loom.semaphore_give %45 : memref<?x32x128xf16>
              %94 = loom.sync ins(%91 : tensor<?x32x1xf16>) outs(%54 : tensor<?x32x1xf16>) -> tensor<?x32x1xf16>
              loom.semaphore_give %59 : memref<?x32x1xf16>
              %95 = loom.alloc [%24, %20, 32, 1] on @L1 : memref<?x?x32x1xf16>
              %96 = loom.semaphore_take %95 : memref<?x?x32x1xf16> -> memref<?x?x32x1xf16>
              %97 = loom.init_tensor %96[%24, %20, 32, 1] : memref<?x?x32x1xf16> -> tensor<?x?x32x1xf16>
              %98 = loom.gather ins(%94 : tensor<?x32x1xf16>) outs(%97 : tensor<?x?x32x1xf16>) across(%arg5 : index) region : (UL : [%c0, %arg4], LR : [%c7, %arg4]) -> tensor<?x?x32x1xf16>
              loom.semaphore_give %53 : memref<?x32x1xf16>
              %99 = loom.sync ins(%93 : tensor<?x32x128xf16>) outs(%44 : tensor<?x32x128xf16>) -> tensor<?x32x128xf16>
              loom.semaphore_give %33 : memref<?x32x128xf16>
              %100 = loom.alloc [%24, %20, 32, 128] on @L1 : memref<?x?x32x128xf16>
              %101 = loom.semaphore_take %100 : memref<?x?x32x128xf16> -> memref<?x?x32x128xf16>
              %102 = loom.init_tensor %101[%24, %20, 32, 128] : memref<?x?x32x128xf16> -> tensor<?x?x32x128xf16>
              %103 = loom.gather ins(%99 : tensor<?x32x128xf16>) outs(%102 : tensor<?x?x32x128xf16>) across(%arg5 : index) region : (UL : [%c0, %arg4], LR : [%c7, %arg4]) -> tensor<?x?x32x128xf16>
              loom.semaphore_give %43 : memref<?x32x128xf16>
              %104 = loom.alloc [%24, %20, 32, 1] on @L1 : memref<?x?x32x1xf16>
              %105 = loom.semaphore_take %104 : memref<?x?x32x1xf16> -> memref<?x?x32x1xf16>
              %106 = loom.init_tensor %105[%24, %20, 32, 1] : memref<?x?x32x1xf16> -> tensor<?x?x32x1xf16>
              %107 = loom.alloc [%24, %20, 32, 128] on @L1 : memref<?x?x32x128xf16>
              %108 = loom.semaphore_take %107 : memref<?x?x32x128xf16> -> memref<?x?x32x128xf16>
              %109 = loom.init_tensor %108[%24, %20, 32, 128] : memref<?x?x32x128xf16> -> tensor<?x?x32x128xf16>
              %110 = loom.alloc [%24, %20, 32, 32] on @L1 : memref<?x?x32x32xf16>
              %111 = loom.semaphore_take %110 : memref<?x?x32x32xf16> -> memref<?x?x32x32xf16>
              %112 = loom.init_tensor %111[%24, %20, 32, 32] : memref<?x?x32x32xf16> -> tensor<?x?x32x32xf16>
              %113 = arith.cmpi eq, %arg5, %c0 : index
              scf.if %113 {
                %114 = linalg.fill ins(%cst_1 : f16) outs(%52 : tensor<?x32x1xf16>) -> tensor<?x32x1xf16>
                %115 = loom.sync ins(%98 : tensor<?x?x32x1xf16>) outs(%106 : tensor<?x?x32x1xf16>) -> tensor<?x?x32x1xf16>
                loom.semaphore_give %96 : memref<?x?x32x1xf16>
                %116 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1, d2, d3)>], iterator_types = ["reduction", "parallel", "parallel", "parallel"]} ins(%115 : tensor<?x?x32x1xf16>) outs(%114 : tensor<?x32x1xf16>) {
                ^bb0(%in: f16, %out: f16):
                  %129 = arith.maximumf %in, %out : f16
                  linalg.yield %129 : f16
                } -> tensor<?x32x1xf16>
                %117 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%115, %116 : tensor<?x?x32x1xf16>, tensor<?x32x1xf16>) outs(%106 : tensor<?x?x32x1xf16>) {
                ^bb0(%in: f16, %in_3: f16, %out: f16):
                  %129 = arith.subf %in, %in_3 : f16
                  %130 = math.exp %129 : f16
                  linalg.yield %130 : f16
                } -> tensor<?x?x32x1xf16>
                loom.semaphore_give %51 : memref<?x32x1xf16>
                %118 = linalg.fill ins(%cst : f16) outs(%50 : tensor<?x32x1xf16>) -> tensor<?x32x1xf16>
                %119 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1, d2, d3)>], iterator_types = ["reduction", "parallel", "parallel", "parallel"]} ins(%117 : tensor<?x?x32x1xf16>) outs(%118 : tensor<?x32x1xf16>) {
                ^bb0(%in: f16, %out: f16):
                  %129 = arith.addf %in, %out : f16
                  linalg.yield %129 : f16
                } -> tensor<?x32x1xf16>
                %120 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%117, %119 : tensor<?x?x32x1xf16>, tensor<?x32x1xf16>) outs(%106 : tensor<?x?x32x1xf16>) {
                ^bb0(%in: f16, %in_3: f16, %out: f16):
                  %129 = arith.divf %in, %in_3 : f16
                  linalg.yield %129 : f16
                } -> tensor<?x?x32x1xf16>
                loom.semaphore_give %49 : memref<?x32x1xf16>
                %121 = loom.broadcast ins(%120 : tensor<?x?x32x1xf16>) outs(%112 : tensor<?x?x32x32xf16>) dim(3) -> tensor<?x?x32x128xf16>
                loom.semaphore_give %105 : memref<?x?x32x1xf16>
                %122 = loom.sync ins(%103 : tensor<?x?x32x128xf16>) outs(%109 : tensor<?x?x32x128xf16>) -> tensor<?x?x32x128xf16>
                loom.semaphore_give %101 : memref<?x?x32x128xf16>
                %123 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%122, %121 : tensor<?x?x32x128xf16>, tensor<?x?x32x128xf16>) outs(%109 : tensor<?x?x32x128xf16>) {
                ^bb0(%in: f16, %in_3: f16, %out: f16):
                  %129 = arith.mulf %in, %in_3 : f16
                  linalg.yield %129 : f16
                } -> tensor<?x?x32x128xf16>
                loom.semaphore_give %111 : memref<?x?x32x32xf16>
                %124 = linalg.fill ins(%cst : f16) outs(%32 : tensor<?x32x128xf16>) -> tensor<?x32x128xf16>
                %125 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1, d2, d3)>], iterator_types = ["reduction", "parallel", "parallel", "parallel"]} ins(%123 : tensor<?x?x32x128xf16>) outs(%124 : tensor<?x32x128xf16>) {
                ^bb0(%in: f16, %out: f16):
                  %129 = arith.addf %in, %out : f16
                  linalg.yield %129 : f16
                } -> tensor<?x32x128xf16>
                loom.semaphore_give %108 : memref<?x?x32x128xf16>
                %126 = loom.sync ins(%125 : tensor<?x32x128xf16>) outs(%42 : tensor<?x32x128xf16>) -> tensor<?x32x128xf16>
                loom.semaphore_give %31 : memref<?x32x128xf16>
                %127 = loom.subview %arg2[%29, 0, 0] [%20, 32, 128] [1, 1, 1], reuse : [seq = false, spat = false, temp = false] : memref<16x32x128xf16> to memref<?x32x128xf16, strided<[4096, 128, 1], offset: ?>>
                %128 = loom.bufferize_to_memref %126 : tensor<?x32x128xf16> -> memref<?x32x128xf16>
                loom.copy %128, %127 src_mem_space @mem_L1 dst_mem_space @mem_DRAM, broadcast : [1, 1] : memref<?x32x128xf16> to memref<?x32x128xf16, strided<[4096, 128, 1], offset: ?>>
                loom.semaphore_give %41 : memref<?x32x128xf16>
              }
            } {loom.block_sym = @tile_s, loom.iter_type = #loom.iter_type<temporal>}
          } {loom.block_sym = @tile_b, loom.iter_type = #loom.iter_type<temporal>}
        } {loom.block_sym = @tile_s, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 0 : i64, loom.physical_dim = @dim_x}
      } {loom.block_sym = @tile_b, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 0 : i64, loom.physical_dim = @dim_y}
      return
    }
  }
}
