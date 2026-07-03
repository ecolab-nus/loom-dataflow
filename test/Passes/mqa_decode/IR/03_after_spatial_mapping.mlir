module attributes {loom.tile_b = {is_reduction = false, upper_bound = 16 : index}, loom.tile_n = {is_reduction = false, upper_bound = 8192 : index}, loom.tile_s = {is_reduction = false, upper_bound = 8192 : index}} {
  %0 = adl.memory.bank "mem_DRAM_bank", {bsize = 8192 : i64, nblk = 196608 : i64}
  %1 = adl.spatial_dim "dim_dram_channel", 8
  %2 = adl.memory.array "mem_DRAM", [%1] of %0
  %3 = adl.memory.bank "mem_bank", {bsize = 16 : i64, nblk = 5464 : i64}
  %4 = adl.spatial_dim "dim_nbank", 16
  %5 = adl.memory.array "mem_L1", [%4] of %3
  %6 = adl.resource.exclusive "res_matrix_lane"
  %7 = adl.resource.exclusive "res_vector_lane"
  %8 = adl.processor.compute @proc_matrix_lane, [(%5, %5)], with [%6]
  %9 = adl.processor.compute @proc_vector_lane, [(%5, %5)], with [%7]
  %10 = adl.arch.compose "arch_core", arch[%8, %9], mem[%5]
  %11 = adl.spatial_dim "dim_x", 8
  %12 = adl.spatial_dim "dim_y", 8
  %13 = adl.memory.array "mem_array_L1", [%11, %12] of %5
  %14 = adl.arch.scale "arch_mesh", [%11, %12] of %10, mem_region %13
  %15 = adl.processor.dmover @proc_dram_l1_noc0, [(%2, %13)]
  %16 = adl.processor.dmover @proc_dram_l1_noc1, [(%13, %2), (%13, %13)]
  %17 = adl.arch.compose "arch_system", arch[%14, %15, %16], mem[%2]
  func.func @flash_decode(%arg0: memref<16x128x8192xf16>, %arg1: memref<16x8192x128xf16>, %arg2: memref<16x32x128xf16>, %arg3: memref<16x32x128xf16>) {
    %c0 = arith.constant 0 : index
    %cst = arith.constant 0.000000e+00 : f16
    %cst_0 = arith.constant 1.000000e+00 : f16
    %cst_1 = arith.constant 0xFC00 : f16
    %c1 = arith.constant 1 : index
    %cst_2 = arith.constant 8.837890e-02 : f16
    %c16 = arith.constant 16 : index
    %c8192 = arith.constant 8192 : index
    %18 = loom.sym @tile_b {upper_bound = 16 : index} : index
    %19 = loom.sym @tile_s {upper_bound = 8192 : index} : index
    %20 = loom.sym @tile_n {upper_bound = 8192 : index} : index
    %21 = arith.ceildivui %c16, %18 : index
    %22 = arith.ceildivui %c8192, %19 : index
    affine.parallel (%arg4, %arg5) = (0, 0) to (symbol(%21), symbol(%22)) {
      %23 = arith.muli %arg4, %18 : index
      %24 = loom.alloc [%18, 32, 128] on @L1 : memref<?x32x128xf16>
      %25 = loom.semaphore_take %24 : memref<?x32x128xf16> -> memref<?x32x128xf16>
      %26 = loom.subview %arg2[%23, 0, 0] [%18, 32, 128] [1, 1, 1], reuse : [seq = false, spat = false, temp = false] : memref<16x32x128xf16> to memref<?x32x128xf16, strided<[4096, 128, 1], offset: ?>>
      loom.copy %26, %25 src_mem_space @mem_DRAM dst_mem_space @mem_L1, area : [1, 1] : memref<?x32x128xf16, strided<[4096, 128, 1], offset: ?>> to memref<?x32x128xf16>
      %27 = loom.bufferize_to_tensor %25[%18, 32, 128] : memref<?x32x128xf16> -> tensor<?x32x128xf16>
      %28 = arith.muli %arg5, %19 : index
      %29 = arith.ceildivui %19, %20 : index
      %30 = loom.alloc [%18, 32, 128] on @L1 : memref<?x32x128xf16>
      %31 = loom.semaphore_take %30 : memref<?x32x128xf16> -> memref<?x32x128xf16>
      %32 = loom.init_tensor %31[%18, 32, 128] : memref<?x32x128xf16> -> tensor<?x32x128xf16>
      %33 = linalg.fill ins(%cst : f16) outs(%32 : tensor<?x32x128xf16>) -> tensor<?x32x128xf16>
      %34 = loom.alloc [%18, 32, 1] on @L1 : memref<?x32x1xf16>
      %35 = loom.semaphore_take %34 : memref<?x32x1xf16> -> memref<?x32x1xf16>
      %36 = loom.init_tensor %35[%18, 32, 1] : memref<?x32x1xf16> -> tensor<?x32x1xf16>
      %37 = loom.semaphore_take %34 : memref<?x32x1xf16> -> memref<?x32x1xf16>
      %38 = loom.init_tensor %37[%18, 32, 1] : memref<?x32x1xf16> -> tensor<?x32x1xf16>
      %39 = loom.semaphore_take %34 : memref<?x32x1xf16> -> memref<?x32x1xf16>
      %40 = loom.init_tensor %39[%18, 32, 1] : memref<?x32x1xf16> -> tensor<?x32x1xf16>
      %41 = linalg.fill ins(%cst_0 : f16) outs(%40 : tensor<?x32x1xf16>) -> tensor<?x32x1xf16>
      %42 = loom.alloc [%18, 32, 1] on @L1 : memref<?x32x1xf16>
      %43 = loom.semaphore_take %42 : memref<?x32x1xf16> -> memref<?x32x1xf16>
      %44 = loom.init_tensor %43[%18, 32, 1] : memref<?x32x1xf16> -> tensor<?x32x1xf16>
      %45 = linalg.fill ins(%cst_1 : f16) outs(%44 : tensor<?x32x1xf16>) -> tensor<?x32x1xf16>
      %46 = loom.alloc [%18, 32, 128] on @L1 : memref<?x32x128xf16>
      %47 = loom.semaphore_take %46 : memref<?x32x128xf16> -> memref<?x32x128xf16>
      %48 = loom.init_tensor %47[%18, 32, 128] : memref<?x32x128xf16> -> tensor<?x32x128xf16>
      %49 = loom.semaphore_take %46 : memref<?x32x128xf16> -> memref<?x32x128xf16>
      %50 = loom.init_tensor %49[%18, 32, 128] : memref<?x32x128xf16> -> tensor<?x32x128xf16>
      %51 = loom.alloc [%18, 32, 128] on @L1 : memref<?x32x128xf16>
      %52 = loom.semaphore_take %51 : memref<?x32x128xf16> -> memref<?x32x128xf16>
      %53 = loom.init_tensor %52[%18, 32, 128] : memref<?x32x128xf16> -> tensor<?x32x128xf16>
      %54 = loom.alloc [%18, 32, 1] on @L1 : memref<?x32x1xf16>
      %55 = loom.semaphore_take %54 : memref<?x32x1xf16> -> memref<?x32x1xf16>
      %56 = loom.init_tensor %55[%18, 32, 1] : memref<?x32x1xf16> -> tensor<?x32x1xf16>
      %57 = loom.alloc [%18, 32, 1] on @L1 : memref<?x32x1xf16>
      %58 = loom.semaphore_take %57 : memref<?x32x1xf16> -> memref<?x32x1xf16>
      %59 = loom.init_tensor %58[%18, 32, 1] : memref<?x32x1xf16> -> tensor<?x32x1xf16>
      %60 = loom.alloc [%18, 32, 1] on @L1 : memref<?x32x1xf16>
      %61 = loom.semaphore_take %60 : memref<?x32x1xf16> -> memref<?x32x1xf16>
      %62 = loom.init_tensor %61[%18, 32, 1] : memref<?x32x1xf16> -> tensor<?x32x1xf16>
      %63 = loom.alloc [%18, 128, %20] on @L1 : memref<?x128x?xf16>
      %64 = loom.semaphore_take %63 : memref<?x128x?xf16> -> memref<?x128x?xf16>
      %65 = loom.alloc [%18, 32, %20] on @L1 : memref<?x32x?xf16>
      %66 = loom.semaphore_take %65 : memref<?x32x?xf16> -> memref<?x32x?xf16>
      %67 = loom.init_tensor %66[%18, 32, %20] : memref<?x32x?xf16> -> tensor<?x32x?xf16>
      %68 = loom.alloc [%18, 32, %20] on @L1 : memref<?x32x?xf16>
      %69 = loom.semaphore_take %68 : memref<?x32x?xf16> -> memref<?x32x?xf16>
      %70 = loom.init_tensor %69[%18, 32, %20] : memref<?x32x?xf16> -> tensor<?x32x?xf16>
      %71 = loom.alloc [%18, %20, 128] on @L1 : memref<?x?x128xf16>
      %72 = loom.semaphore_take %71 : memref<?x?x128xf16> -> memref<?x?x128xf16>
      %73:3 = scf.for %arg6 = %c0 to %29 step %c1 iter_args(%arg7 = %45, %arg8 = %41, %arg9 = %33) -> (tensor<?x32x1xf16>, tensor<?x32x1xf16>, tensor<?x32x128xf16>) {
        %101 = arith.muli %arg6, %20 : index
        %102 = arith.addi %28, %101 : index
        %103 = loom.subview %arg0[%23, 0, %102] [%18, 128, %20] [1, 1, 1], reuse : [seq = false, spat = false, temp = false] : memref<16x128x8192xf16> to memref<?x128x?xf16, strided<[1048576, 8192, 1], offset: ?>>
        loom.copy %103, %64 src_mem_space @mem_DRAM dst_mem_space @mem_L1, area : [1, 1] : memref<?x128x?xf16, strided<[1048576, 8192, 1], offset: ?>> to memref<?x128x?xf16>
        %104 = loom.bufferize_to_tensor %64[%18, 128, %20] : memref<?x128x?xf16> -> tensor<?x128x?xf16>
        %105 = linalg.fill ins(%cst : f16) outs(%67 : tensor<?x32x?xf16>) -> tensor<?x32x?xf16>
        %106 = linalg.batch_matmul ins(%27, %104 : tensor<?x32x128xf16>, tensor<?x128x?xf16>) outs(%105 : tensor<?x32x?xf16>) -> tensor<?x32x?xf16>
        loom.semaphore_give %64 : memref<?x128x?xf16>
        %107 = linalg.fill ins(%cst_1 : f16) outs(%56 : tensor<?x32x1xf16>) -> tensor<?x32x1xf16>
        %108 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, 0)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%106 : tensor<?x32x?xf16>) outs(%107 : tensor<?x32x1xf16>) {
        ^bb0(%in: f16, %out: f16):
          %123 = arith.maximumf %in, %out : f16
          linalg.yield %123 : f16
        } -> tensor<?x32x1xf16>
        %109 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%arg7, %108 : tensor<?x32x1xf16>, tensor<?x32x1xf16>) outs(%56 : tensor<?x32x1xf16>) {
        ^bb0(%in: f16, %in_3: f16, %out: f16):
          %123 = arith.mulf %in_3, %cst_2 : f16
          %124 = arith.cmpf ogt, %in, %123 : f16
          %125 = arith.select %124, %in, %123 : f16
          linalg.yield %125 : f16
        } -> tensor<?x32x1xf16>
        %110 = loom.broadcast ins(%109 : tensor<?x32x1xf16>) outs(%70 : tensor<?x32x?xf16>) dim(2) -> tensor<?x32x?xf16>
        %111 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%106, %110 : tensor<?x32x?xf16>, tensor<?x32x?xf16>) outs(%67 : tensor<?x32x?xf16>) {
        ^bb0(%in: f16, %in_3: f16, %out: f16):
          %123 = arith.mulf %in, %cst_2 : f16
          %124 = arith.subf %123, %in_3 : f16
          %125 = math.exp %124 : f16
          linalg.yield %125 : f16
        } -> tensor<?x32x?xf16>
        loom.semaphore_give %69 : memref<?x32x?xf16>
        %112 = linalg.fill ins(%cst : f16) outs(%59 : tensor<?x32x1xf16>) -> tensor<?x32x1xf16>
        %113 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, 0)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%111 : tensor<?x32x?xf16>) outs(%112 : tensor<?x32x1xf16>) {
        ^bb0(%in: f16, %out: f16):
          %123 = arith.addf %in, %out : f16
          linalg.yield %123 : f16
        } -> tensor<?x32x1xf16>
        %114 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%arg7, %109 : tensor<?x32x1xf16>, tensor<?x32x1xf16>) outs(%62 : tensor<?x32x1xf16>) {
        ^bb0(%in: f16, %in_3: f16, %out: f16):
          %123 = arith.subf %in, %in_3 : f16
          %124 = math.exp %123 : f16
          linalg.yield %124 : f16
        } -> tensor<?x32x1xf16>
        %115 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%arg8, %114, %113 : tensor<?x32x1xf16>, tensor<?x32x1xf16>, tensor<?x32x1xf16>) outs(%arg8 : tensor<?x32x1xf16>) {
        ^bb0(%in: f16, %in_3: f16, %in_4: f16, %out: f16):
          %123 = arith.mulf %in, %in_3 : f16
          %124 = arith.addf %123, %in_4 : f16
          linalg.yield %124 : f16
        } -> tensor<?x32x1xf16>
        loom.semaphore_give %58 : memref<?x32x1xf16>
        %116 = loom.subview %arg1[%23, %102, 0] [%18, %20, 128] [1, 1, 1], reuse : [seq = false, spat = false, temp = false] : memref<16x8192x128xf16> to memref<?x?x128xf16, strided<[1048576, 128, 1], offset: ?>>
        loom.copy %116, %72 src_mem_space @mem_DRAM dst_mem_space @mem_L1, area : [1, 1] : memref<?x?x128xf16, strided<[1048576, 128, 1], offset: ?>> to memref<?x?x128xf16>
        %117 = loom.bufferize_to_tensor %72[%18, %20, 128] : memref<?x?x128xf16> -> tensor<?x?x128xf16>
        %118 = linalg.fill ins(%cst : f16) outs(%50 : tensor<?x32x128xf16>) -> tensor<?x32x128xf16>
        %119 = linalg.batch_matmul ins(%111, %117 : tensor<?x32x?xf16>, tensor<?x?x128xf16>) outs(%118 : tensor<?x32x128xf16>) -> tensor<?x32x128xf16>
        loom.semaphore_give %72 : memref<?x?x128xf16>
        loom.semaphore_give %66 : memref<?x32x?xf16>
        %120 = loom.broadcast ins(%114 : tensor<?x32x1xf16>) outs(%53 : tensor<?x32x128xf16>) dim(2) -> tensor<?x32x128xf16>
        loom.semaphore_give %61 : memref<?x32x1xf16>
        %121 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%119, %arg9, %120 : tensor<?x32x128xf16>, tensor<?x32x128xf16>, tensor<?x32x128xf16>) outs(%arg9 : tensor<?x32x128xf16>) {
        ^bb0(%in: f16, %in_3: f16, %in_4: f16, %out: f16):
          %123 = arith.mulf %in_3, %in_4 : f16
          %124 = arith.addf %in, %123 : f16
          linalg.yield %124 : f16
        } -> tensor<?x32x128xf16>
        loom.semaphore_give %52 : memref<?x32x128xf16>
        loom.semaphore_give %49 : memref<?x32x128xf16>
        %122 = linalg.copy ins(%109 : tensor<?x32x1xf16>) outs(%arg7 : tensor<?x32x1xf16>) -> tensor<?x32x1xf16>
        loom.semaphore_give %55 : memref<?x32x1xf16>
        scf.yield %122, %115, %121 : tensor<?x32x1xf16>, tensor<?x32x1xf16>, tensor<?x32x128xf16>
      }
      loom.semaphore_give %25 : memref<?x32x128xf16>
      %74 = loom.alloc [%18, 32, 1] on @L1 : memref<?x32x1xf16>
      %75 = loom.semaphore_take %74 : memref<?x32x1xf16> -> memref<?x32x1xf16>
      %76 = loom.init_tensor %75[%18, 32, 1] : memref<?x32x1xf16> -> tensor<?x32x1xf16>
      %77 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%73#1, %73#0 : tensor<?x32x1xf16>, tensor<?x32x1xf16>) outs(%76 : tensor<?x32x1xf16>) {
      ^bb0(%in: f16, %in_3: f16, %out: f16):
        %101 = math.log %in : f16
        %102 = arith.addf %101, %in_3 : f16
        linalg.yield %102 : f16
      } -> tensor<?x32x1xf16>
      loom.semaphore_give %43 : memref<?x32x1xf16>
      %78 = loom.broadcast ins(%73#1 : tensor<?x32x1xf16>) outs(%48 : tensor<?x32x128xf16>) dim(2) -> tensor<?x32x128xf16>
      loom.semaphore_give %39 : memref<?x32x1xf16>
      %79 = loom.alloc [%18, 32, 128] on @L1 : memref<?x32x128xf16>
      %80 = loom.semaphore_take %79 : memref<?x32x128xf16> -> memref<?x32x128xf16>
      %81 = loom.init_tensor %80[%18, 32, 128] : memref<?x32x128xf16> -> tensor<?x32x128xf16>
      %82 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%73#2, %78 : tensor<?x32x128xf16>, tensor<?x32x128xf16>) outs(%81 : tensor<?x32x128xf16>) {
      ^bb0(%in: f16, %in_3: f16, %out: f16):
        %101 = arith.divf %in, %in_3 : f16
        linalg.yield %101 : f16
      } -> tensor<?x32x128xf16>
      loom.semaphore_give %47 : memref<?x32x128xf16>
      loom.semaphore_give %31 : memref<?x32x128xf16>
      %83 = loom.bufferize_to_memref %77 : tensor<?x32x1xf16> -> memref<?x32x1xf16>
      %84 = loom.alloc [%22, %18, 32, 1] on @L1 : memref<?x?x32x1xf16>
      %85 = loom.semaphore_take %84 : memref<?x?x32x1xf16> -> memref<?x?x32x1xf16>
      loom.gather %83, %85 src_mem_space @mem_array_L1 dst_mem_space @mem_array_L1 across(%arg5 : index), area : [1, 1] : memref<?x32x1xf16> to memref<?x?x32x1xf16>
      loom.semaphore_give %75 : memref<?x32x1xf16>
      %86 = loom.bufferize_to_tensor %85[%22, %18, 32, 1] : memref<?x?x32x1xf16> -> tensor<?x?x32x1xf16>
      %87 = loom.bufferize_to_memref %82 : tensor<?x32x128xf16> -> memref<?x32x128xf16>
      %88 = loom.alloc [%22, %18, 32, 128] on @L1 : memref<?x?x32x128xf16>
      %89 = loom.semaphore_take %88 : memref<?x?x32x128xf16> -> memref<?x?x32x128xf16>
      loom.gather %87, %89 src_mem_space @mem_array_L1 dst_mem_space @mem_array_L1 across(%arg5 : index), area : [1, 1] : memref<?x32x128xf16> to memref<?x?x32x128xf16>
      loom.semaphore_give %80 : memref<?x32x128xf16>
      %90 = loom.bufferize_to_tensor %89[%22, %18, 32, 128] : memref<?x?x32x128xf16> -> tensor<?x?x32x128xf16>
      %91 = arith.cmpi eq, %arg5, %c0 : index
      %92 = loom.alloc [%18, 32, 128] on @L1 : memref<?x32x128xf16>
      %93 = loom.semaphore_take %92 : memref<?x32x128xf16> -> memref<?x32x128xf16>
      %94 = loom.init_tensor %93[%18, 32, 128] : memref<?x32x128xf16> -> tensor<?x32x128xf16>
      %95 = loom.alloc [%22, %18, 32, 1] on @L1 : memref<?x?x32x1xf16>
      %96 = loom.semaphore_take %95 : memref<?x?x32x1xf16> -> memref<?x?x32x1xf16>
      %97 = loom.init_tensor %96[%22, %18, 32, 1] : memref<?x?x32x1xf16> -> tensor<?x?x32x1xf16>
      %98 = loom.alloc [%22, %18, 32, 128] on @L1 : memref<?x?x32x128xf16>
      %99 = loom.semaphore_take %98 : memref<?x?x32x128xf16> -> memref<?x?x32x128xf16>
      %100 = loom.init_tensor %99[%22, %18, 32, 128] : memref<?x?x32x128xf16> -> tensor<?x?x32x128xf16>
      scf.if %91 {
        %101 = linalg.fill ins(%cst_1 : f16) outs(%38 : tensor<?x32x1xf16>) -> tensor<?x32x1xf16>
        %102 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1, d2, d3)>], iterator_types = ["reduction", "parallel", "parallel", "parallel"]} ins(%86 : tensor<?x?x32x1xf16>) outs(%101 : tensor<?x32x1xf16>) {
        ^bb0(%in: f16, %out: f16):
          %113 = arith.maximumf %in, %out : f16
          linalg.yield %113 : f16
        } -> tensor<?x32x1xf16>
        %103 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%86, %102 : tensor<?x?x32x1xf16>, tensor<?x32x1xf16>) outs(%97 : tensor<?x?x32x1xf16>) {
        ^bb0(%in: f16, %in_3: f16, %out: f16):
          %113 = arith.subf %in, %in_3 : f16
          %114 = math.exp %113 : f16
          linalg.yield %114 : f16
        } -> tensor<?x?x32x1xf16>
        loom.semaphore_give %85 : memref<?x?x32x1xf16>
        loom.semaphore_give %37 : memref<?x32x1xf16>
        %104 = linalg.fill ins(%cst : f16) outs(%36 : tensor<?x32x1xf16>) -> tensor<?x32x1xf16>
        %105 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1, d2, d3)>], iterator_types = ["reduction", "parallel", "parallel", "parallel"]} ins(%103 : tensor<?x?x32x1xf16>) outs(%104 : tensor<?x32x1xf16>) {
        ^bb0(%in: f16, %out: f16):
          %113 = arith.addf %in, %out : f16
          linalg.yield %113 : f16
        } -> tensor<?x32x1xf16>
        %106 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%103, %105 : tensor<?x?x32x1xf16>, tensor<?x32x1xf16>) outs(%97 : tensor<?x?x32x1xf16>) {
        ^bb0(%in: f16, %in_3: f16, %out: f16):
          %113 = arith.divf %in, %in_3 : f16
          linalg.yield %113 : f16
        } -> tensor<?x?x32x1xf16>
        loom.semaphore_give %35 : memref<?x32x1xf16>
        %107 = loom.broadcast ins(%106 : tensor<?x?x32x1xf16>) outs(%100 : tensor<?x?x32x128xf16>) dim(3) -> tensor<?x?x32x128xf16>
        loom.semaphore_give %96 : memref<?x?x32x1xf16>
        %108 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%90, %107 : tensor<?x?x32x128xf16>, tensor<?x?x32x128xf16>) outs(%100 : tensor<?x?x32x128xf16>) {
        ^bb0(%in: f16, %in_3: f16, %out: f16):
          %113 = arith.mulf %in, %in_3 : f16
          linalg.yield %113 : f16
        } -> tensor<?x?x32x128xf16>
        loom.semaphore_give %89 : memref<?x?x32x128xf16>
        %109 = linalg.fill ins(%cst : f16) outs(%94 : tensor<?x32x128xf16>) -> tensor<?x32x128xf16>
        %110 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1, d2, d3)>], iterator_types = ["reduction", "parallel", "parallel", "parallel"]} ins(%108 : tensor<?x?x32x128xf16>) outs(%109 : tensor<?x32x128xf16>) {
        ^bb0(%in: f16, %out: f16):
          %113 = arith.addf %in, %out : f16
          linalg.yield %113 : f16
        } -> tensor<?x32x128xf16>
        loom.semaphore_give %99 : memref<?x?x32x128xf16>
        %111 = loom.subview %arg3[%23, 0, 0] [%18, 32, 128] [1, 1, 1], reuse : [seq = false, spat = false, temp = false] : memref<16x32x128xf16> to memref<?x32x128xf16, strided<[4096, 128, 1], offset: ?>>
        %112 = loom.bufferize_to_memref %110 : tensor<?x32x128xf16> -> memref<?x32x128xf16>
        loom.copy %112, %111 src_mem_space @mem_L1 dst_mem_space @mem_DRAM, area : [1, 1] : memref<?x32x128xf16> to memref<?x32x128xf16, strided<[4096, 128, 1], offset: ?>>
        loom.semaphore_give %93 : memref<?x32x128xf16>
      }
    }
    return
  }
}
