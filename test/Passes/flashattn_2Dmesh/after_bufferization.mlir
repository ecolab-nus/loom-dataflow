module {
  %0 = df.spatial_dim "x", 8
  %1 = df.spatial_dim "y", 8
  %2 = df.compute "cores", %0, %1 {map = affine_map<(d0, d1) -> (d0, d1)>}
  %3 = df.memory "L1", %0, %1 {bandwidth = 64 : i64, map = affine_map<(d0, d1) -> (d0, d1)>, size = 32768 : i64}
  %4 = df.mux %2 : !df.compute, %3 : !df.memory, %0, %1 {map = affine_map<(d0, d1) -> (d0, d1)>}
  %5 = df.interconnects "horizontal_links" %3 : !df.memory, %3 : !df.memory, %0, %1 {map = affine_map<(d0, d1) -> (d0 + 1, d1)>} : !df.interconnect
  %6 = df.interconnects "vertical_links" %3 : !df.memory, %3 : !df.memory, %0, %1 {map = affine_map<(d0, d1) -> (d0, d1 + 1)>} : !df.interconnect
  func.func @flashattn_fwd__d0i0_d1i0__c0mem_c1mem_c2mem(%arg0: memref<*xf16> {tt.divisibility = 16 : i32}, %arg1: memref<*xf16> {tt.divisibility = 16 : i32}, %arg2: memref<*xf16> {tt.divisibility = 16 : i32}, %arg3: memref<*xf16> {tt.divisibility = 16 : i32}, %arg4: index, %arg5: index, %arg6: index) {
    affine.for %arg7 = 0 to affine_map<(d0) -> ((d0 ceildiv 8) ceildiv 8)>(%arg4) {
      affine.parallel (%arg8) = (0) to (8) {
        affine.parallel (%arg9) = (0) to (8) {
          %cst = arith.constant 1.000000e+00 : f32
          %cst_0 = arith.constant 0.176776692 : f32
          %cst_1 = arith.constant 0.000000e+00 : f32
          %cst_2 = arith.constant 0xFF800000 : f32
          %alloc = memref.alloc() {alignment = 64 : i64} : memref<32xf32>
          %alloc_3 = memref.alloc() {alignment = 64 : i64} : memref<32xf32>
          linalg.fill ins(%cst : f32) outs(%alloc_3 : memref<32xf32>)
          %alloc_4 = memref.alloc() {alignment = 64 : i64} : memref<32x32xf32>
          %alloc_5 = memref.alloc() {alignment = 64 : i64} : memref<32x32xf32>
          linalg.fill ins(%cst_0 : f32) outs(%alloc_5 : memref<32x32xf32>)
          %alloc_6 = memref.alloc() {alignment = 64 : i64} : memref<32x32xf32>
          linalg.fill ins(%cst_1 : f32) outs(%alloc_6 : memref<32x32xf32>)
          %alloc_7 = memref.alloc() {alignment = 64 : i64} : memref<32xf32>
          linalg.fill ins(%cst_2 : f32) outs(%alloc_7 : memref<32xf32>)
          %7 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 1024 + d1 * 65536 + d2 * 8192)>(%arg9, %arg7, %arg8)
          %reinterpret_cast = memref.reinterpret_cast %arg0 to offset: [%7], sizes: [32, 32], strides: [32, 1] {loom.reuse = {spatial = [{depth = 1 : i64, iterator = "%arg8", mapped_to = "y", reuse_type = "no_reuse", volume = 0 : i64}, {depth = 2 : i64, iterator = "%arg9", mapped_to = "x", reuse_type = "no_reuse", volume = 0 : i64}], temporal = [{depth = 0 : i64, iterator = "%arg7", reuse_type = "no_reuse", volume = 0 : i64}]}} : memref<*xf16> to memref<32x32xf16, strided<[32, 1], offset: ?>>
          %alloc_8 = memref.alloc() {loom.alloc = {local = true, memory_name = "L1", size = 2048 : i64}} : memref<32x32xf16>
          memref.copy %reinterpret_cast, %alloc_8 {loom.copy.choice = {kind = "mem", memory_name = "L1"}} : memref<32x32xf16, strided<[32, 1], offset: ?>> to memref<32x32xf16>
          %c0 = arith.constant 0 : index
          %c512 = arith.constant 512 : index
          %c32 = arith.constant 32 : index
          %alloc_9 = memref.alloc() {alignment = 64 : i64} : memref<32xf32>
          memref.copy %alloc_7, %alloc_9 : memref<32xf32> to memref<32xf32>
          %alloc_10 = memref.alloc() {alignment = 64 : i64} : memref<32x32xf32>
          memref.copy %alloc_6, %alloc_10 : memref<32x32xf32> to memref<32x32xf32>
          %8:5 = scf.for %arg10 = %c0 to %c512 step %c32 iter_args(%arg11 = %c0, %arg12 = %c0, %arg13 = %alloc_9, %arg14 = %alloc_3, %arg15 = %alloc_10) -> (index, index, memref<32xf32>, memref<32xf32>, memref<32x32xf32>) {
            %10 = affine.apply affine_map<()[s0] -> (s0)>()[%arg12]
            %reinterpret_cast_13 = memref.reinterpret_cast %arg1 to offset: [%10], sizes: [32, 32], strides: [512, 1] {loom.reuse = {sequential = [{depth = 3 : i64, iterator = "%arg10", reuse_type = "total_reuse", volume = 2048 : i64}], spatial = [{depth = 1 : i64, iterator = "%arg8", mapped_to = "y", reuse_type = "total_reuse", volume = 2048 : i64}, {depth = 2 : i64, iterator = "%arg9", mapped_to = "x", reuse_type = "total_reuse", volume = 2048 : i64}], temporal = [{depth = 0 : i64, iterator = "%arg7", reuse_type = "total_reuse", volume = 2048 : i64}]}} : memref<*xf16> to memref<32x32xf16, strided<[512, 1], offset: ?>>
            %alloc_14 = memref.alloc() {loom.alloc = {local = true, memory_name = "L1", size = 2048 : i64}} : memref<32x32xf16>
            memref.copy %reinterpret_cast_13, %alloc_14 {loom.copy.choice = {kind = "mem", memory_name = "L1"}} : memref<32x32xf16, strided<[512, 1], offset: ?>> to memref<32x32xf16>
            %alloc_15 = memref.alloc() {alignment = 64 : i64} : memref<32x32xf32>
            memref.copy %alloc_6, %alloc_15 : memref<32x32xf32> to memref<32x32xf32>
            linalg.matmul ins(%alloc_8, %alloc_14 : memref<32x32xf16>, memref<32x32xf16>) outs(%alloc_15 : memref<32x32xf32>)
            linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%alloc_15, %alloc_5 : memref<32x32xf32>, memref<32x32xf32>) outs(%alloc_15 : memref<32x32xf32>) {
            ^bb0(%in: f32, %in_24: f32, %out: f32):
              %14 = arith.mulf %in, %in_24 : f32
              linalg.yield %14 : f32
            }
            linalg.transpose ins(%alloc_15 : memref<32x32xf32>) outs(%alloc_4 : memref<32x32xf32>) permutation = [1, 0] 
            %alloc_16 = memref.alloc() {alignment = 64 : i64} : memref<32xf32>
            memref.copy %alloc_7, %alloc_16 : memref<32xf32> to memref<32xf32>
            linalg.reduce ins(%alloc_4 : memref<32x32xf32>) outs(%alloc_16 : memref<32xf32>) dimensions = [0] 
              (%in: f32, %init: f32) {
                %14 = arith.maxnumf %in, %init : f32
                linalg.yield %14 : f32
              }
            %alloc_17 = memref.alloc() {alignment = 64 : i64} : memref<32xf32>
            linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%arg13, %alloc_16 : memref<32xf32>, memref<32xf32>) outs(%alloc_17 : memref<32xf32>) {
            ^bb0(%in: f32, %in_24: f32, %out: f32):
              %14 = arith.maxnumf %in, %in_24 : f32
              linalg.yield %14 : f32
            }
            %expand_shape_18 = memref.expand_shape %alloc_17 [[0, 1]] output_shape [32, 1] : memref<32xf32> into memref<32x1xf32>
            linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, 0)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%expand_shape_18 : memref<32x1xf32>) outs(%alloc_4 : memref<32x32xf32>) attrs =  {broadcastDims = array<i64: 1>} {
            ^bb0(%in: f32, %out: f32):
              linalg.yield %in : f32
            }
            linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%alloc_15, %alloc_4 : memref<32x32xf32>, memref<32x32xf32>) outs(%alloc_15 : memref<32x32xf32>) {
            ^bb0(%in: f32, %in_24: f32, %out: f32):
              %14 = arith.subf %in, %in_24 : f32
              linalg.yield %14 : f32
            }
            linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%alloc_15 : memref<32x32xf32>) outs(%alloc_15 : memref<32x32xf32>) {
            ^bb0(%in: f32, %out: f32):
              %14 = math.exp %in : f32
              linalg.yield %14 : f32
            }
            linalg.transpose ins(%alloc_15 : memref<32x32xf32>) outs(%alloc_4 : memref<32x32xf32>) permutation = [1, 0] 
            linalg.fill ins(%cst_1 : f32) outs(%alloc : memref<32xf32>)
            linalg.reduce ins(%alloc_4 : memref<32x32xf32>) outs(%alloc : memref<32xf32>) dimensions = [0] 
              (%in: f32, %init: f32) {
                %14 = arith.addf %in, %init : f32
                linalg.yield %14 : f32
              }
            linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%arg13, %alloc_17 : memref<32xf32>, memref<32xf32>) outs(%arg13 : memref<32xf32>) {
            ^bb0(%in: f32, %in_24: f32, %out: f32):
              %14 = arith.subf %in, %in_24 : f32
              linalg.yield %14 : f32
            }
            linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%arg13 : memref<32xf32>) outs(%arg13 : memref<32xf32>) {
            ^bb0(%in: f32, %out: f32):
              %14 = math.exp %in : f32
              linalg.yield %14 : f32
            }
            linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%arg14, %arg13 : memref<32xf32>, memref<32xf32>) outs(%arg14 : memref<32xf32>) {
            ^bb0(%in: f32, %in_24: f32, %out: f32):
              %14 = arith.mulf %in, %in_24 : f32
              linalg.yield %14 : f32
            }
            linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%arg14, %alloc : memref<32xf32>, memref<32xf32>) outs(%arg14 : memref<32xf32>) {
            ^bb0(%in: f32, %in_24: f32, %out: f32):
              %14 = arith.addf %in, %in_24 : f32
              linalg.yield %14 : f32
            }
            %11 = affine.apply affine_map<()[s0] -> (s0 * 32)>()[%arg11]
            %reinterpret_cast_19 = memref.reinterpret_cast %arg2 to offset: [%11], sizes: [32, 32], strides: [32, 1] {loom.reuse = {sequential = [{depth = 3 : i64, iterator = "%arg10", reuse_type = "total_reuse", volume = 2048 : i64}], spatial = [{depth = 1 : i64, iterator = "%arg8", mapped_to = "y", reuse_type = "total_reuse", volume = 2048 : i64}, {depth = 2 : i64, iterator = "%arg9", mapped_to = "x", reuse_type = "total_reuse", volume = 2048 : i64}], temporal = [{depth = 0 : i64, iterator = "%arg7", reuse_type = "total_reuse", volume = 2048 : i64}]}} : memref<*xf16> to memref<32x32xf16, strided<[32, 1], offset: ?>>
            %alloc_20 = memref.alloc() {loom.alloc = {local = true, memory_name = "L1", size = 2048 : i64}} : memref<32x32xf16>
            memref.copy %reinterpret_cast_19, %alloc_20 {loom.copy.choice = {kind = "mem", memory_name = "L1"}} : memref<32x32xf16, strided<[32, 1], offset: ?>> to memref<32x32xf16>
            %alloc_21 = memref.alloc() {alignment = 64 : i64} : memref<32x32xf16>
            linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%alloc_15 : memref<32x32xf32>) outs(%alloc_21 : memref<32x32xf16>) {
            ^bb0(%in: f32, %out: f16):
              %14 = arith.truncf %in : f32 to f16
              linalg.yield %14 : f16
            }
            %expand_shape_22 = memref.expand_shape %arg13 [[0, 1]] output_shape [32, 1] : memref<32xf32> into memref<32x1xf32>
            linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, 0)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%expand_shape_22 : memref<32x1xf32>) outs(%alloc_4 : memref<32x32xf32>) attrs =  {broadcastDims = array<i64: 1>} {
            ^bb0(%in: f32, %out: f32):
              linalg.yield %in : f32
            }
            linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%arg15, %alloc_4 : memref<32x32xf32>, memref<32x32xf32>) outs(%arg15 : memref<32x32xf32>) {
            ^bb0(%in: f32, %in_24: f32, %out: f32):
              %14 = arith.mulf %in, %in_24 : f32
              linalg.yield %14 : f32
            }
            %alloc_23 = memref.alloc() {alignment = 64 : i64} : memref<32x32xf32>
            memref.copy %alloc_6, %alloc_23 : memref<32x32xf32> to memref<32x32xf32>
            linalg.matmul ins(%alloc_21, %alloc_20 : memref<32x32xf16>, memref<32x32xf16>) outs(%alloc_23 : memref<32x32xf32>)
            linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%arg15, %alloc_23 : memref<32x32xf32>, memref<32x32xf32>) outs(%arg15 : memref<32x32xf32>) {
            ^bb0(%in: f32, %in_24: f32, %out: f32):
              %14 = arith.addf %in, %in_24 : f32
              linalg.yield %14 : f32
            }
            %12 = arith.addi %arg11, %c32 : index
            %13 = arith.addi %arg12, %c32 : index
            scf.yield %12, %13, %alloc_17, %arg14, %arg15 : index, index, memref<32xf32>, memref<32xf32>, memref<32x32xf32>
          }
          %expand_shape = memref.expand_shape %8#3 [[0, 1]] output_shape [32, 1] : memref<32xf32> into memref<32x1xf32>
          linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, 0)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%expand_shape : memref<32x1xf32>) outs(%alloc_4 : memref<32x32xf32>) attrs =  {broadcastDims = array<i64: 1>} {
          ^bb0(%in: f32, %out: f32):
            linalg.yield %in : f32
          }
          linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%8#4, %alloc_4 : memref<32x32xf32>, memref<32x32xf32>) outs(%8#4 : memref<32x32xf32>) {
          ^bb0(%in: f32, %in_13: f32, %out: f32):
            %10 = arith.divf %in, %in_13 : f32
            linalg.yield %10 : f32
          }
          %alloc_11 = memref.alloc() {alignment = 64 : i64} : memref<32x32xf16>
          linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%8#4 : memref<32x32xf32>) outs(%alloc_11 : memref<32x32xf16>) {
          ^bb0(%in: f32, %out: f16):
            %10 = arith.truncf %in : f32 to f16
            linalg.yield %10 : f16
          }
          %9 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 1024 + d1 * 65536 + d2 * 8192)>(%arg9, %arg7, %arg8)
          %reinterpret_cast_12 = memref.reinterpret_cast %arg3 to offset: [%9], sizes: [32, 32], strides: [32, 1] {loom.reuse = {spatial = [{depth = 1 : i64, iterator = "%arg8", mapped_to = "y", reuse_type = "no_reuse", volume = 0 : i64}, {depth = 2 : i64, iterator = "%arg9", mapped_to = "x", reuse_type = "no_reuse", volume = 0 : i64}], temporal = [{depth = 0 : i64, iterator = "%arg7", reuse_type = "no_reuse", volume = 0 : i64}]}} : memref<*xf16> to memref<32x32xf16, strided<[32, 1], offset: ?>>
          memref.copy %alloc_11, %reinterpret_cast_12 : memref<32x32xf16> to memref<32x32xf16, strided<[32, 1], offset: ?>>
        } {loom.mapped_to = "x"}
      } {loom.mapped_to = "y"}
    }
    return
  }
  func.func @flashattn_fwd__d0i0_d1i0__c0mem_c1bx_c2mem(%arg0: memref<*xf16> {tt.divisibility = 16 : i32}, %arg1: memref<*xf16> {tt.divisibility = 16 : i32}, %arg2: memref<*xf16> {tt.divisibility = 16 : i32}, %arg3: memref<*xf16> {tt.divisibility = 16 : i32}, %arg4: index, %arg5: index, %arg6: index) {
    affine.for %arg7 = 0 to affine_map<(d0) -> ((d0 ceildiv 8) ceildiv 8)>(%arg4) {
      affine.parallel (%arg8) = (0) to (8) {
        affine.parallel (%arg9) = (0) to (8) {
          %cst = arith.constant 1.000000e+00 : f32
          %cst_0 = arith.constant 0.176776692 : f32
          %cst_1 = arith.constant 0.000000e+00 : f32
          %cst_2 = arith.constant 0xFF800000 : f32
          %alloc = memref.alloc() {alignment = 64 : i64} : memref<32xf32>
          %alloc_3 = memref.alloc() {alignment = 64 : i64} : memref<32xf32>
          linalg.fill ins(%cst : f32) outs(%alloc_3 : memref<32xf32>)
          %alloc_4 = memref.alloc() {alignment = 64 : i64} : memref<32x32xf32>
          %alloc_5 = memref.alloc() {alignment = 64 : i64} : memref<32x32xf32>
          linalg.fill ins(%cst_0 : f32) outs(%alloc_5 : memref<32x32xf32>)
          %alloc_6 = memref.alloc() {alignment = 64 : i64} : memref<32x32xf32>
          linalg.fill ins(%cst_1 : f32) outs(%alloc_6 : memref<32x32xf32>)
          %alloc_7 = memref.alloc() {alignment = 64 : i64} : memref<32xf32>
          linalg.fill ins(%cst_2 : f32) outs(%alloc_7 : memref<32xf32>)
          %7 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 1024 + d1 * 65536 + d2 * 8192)>(%arg9, %arg7, %arg8)
          %reinterpret_cast = memref.reinterpret_cast %arg0 to offset: [%7], sizes: [32, 32], strides: [32, 1] {loom.reuse = {spatial = [{depth = 1 : i64, iterator = "%arg8", mapped_to = "y", reuse_type = "no_reuse", volume = 0 : i64}, {depth = 2 : i64, iterator = "%arg9", mapped_to = "x", reuse_type = "no_reuse", volume = 0 : i64}], temporal = [{depth = 0 : i64, iterator = "%arg7", reuse_type = "no_reuse", volume = 0 : i64}]}} : memref<*xf16> to memref<32x32xf16, strided<[32, 1], offset: ?>>
          %alloc_8 = memref.alloc() {loom.alloc = {local = true, memory_name = "L1", size = 2048 : i64}} : memref<32x32xf16>
          memref.copy %reinterpret_cast, %alloc_8 {loom.copy.choice = {kind = "mem", memory_name = "L1"}} : memref<32x32xf16, strided<[32, 1], offset: ?>> to memref<32x32xf16>
          %c0 = arith.constant 0 : index
          %c512 = arith.constant 512 : index
          %c32 = arith.constant 32 : index
          %alloc_9 = memref.alloc() {alignment = 64 : i64} : memref<32xf32>
          memref.copy %alloc_7, %alloc_9 : memref<32xf32> to memref<32xf32>
          %alloc_10 = memref.alloc() {alignment = 64 : i64} : memref<32x32xf32>
          memref.copy %alloc_6, %alloc_10 : memref<32x32xf32> to memref<32x32xf32>
          %8:5 = scf.for %arg10 = %c0 to %c512 step %c32 iter_args(%arg11 = %c0, %arg12 = %c0, %arg13 = %alloc_9, %arg14 = %alloc_3, %arg15 = %alloc_10) -> (index, index, memref<32xf32>, memref<32xf32>, memref<32x32xf32>) {
            %10 = affine.apply affine_map<()[s0] -> (s0)>()[%arg12]
            %reinterpret_cast_13 = memref.reinterpret_cast %arg1 to offset: [%10], sizes: [32, 32], strides: [512, 1] {loom.reuse = {sequential = [{depth = 3 : i64, iterator = "%arg10", reuse_type = "total_reuse", volume = 2048 : i64}], spatial = [{depth = 1 : i64, iterator = "%arg8", mapped_to = "y", reuse_type = "total_reuse", volume = 2048 : i64}, {depth = 2 : i64, iterator = "%arg9", mapped_to = "x", reuse_type = "total_reuse", volume = 2048 : i64}], temporal = [{depth = 0 : i64, iterator = "%arg7", reuse_type = "total_reuse", volume = 2048 : i64}]}} : memref<*xf16> to memref<32x32xf16, strided<[512, 1], offset: ?>>
            %alloc_14 = memref.alloc() {loom.alloc = {local = true, memory_name = "L1", size = 2048 : i64}} : memref<32x32xf16>
            memref.copy %reinterpret_cast_13, %alloc_14 {loom.copy.choice = {dim = "x", interconnect_name = "horizontal_links", kind = "broadcast"}} : memref<32x32xf16, strided<[512, 1], offset: ?>> to memref<32x32xf16>
            %alloc_15 = memref.alloc() {alignment = 64 : i64} : memref<32x32xf32>
            memref.copy %alloc_6, %alloc_15 : memref<32x32xf32> to memref<32x32xf32>
            linalg.matmul ins(%alloc_8, %alloc_14 : memref<32x32xf16>, memref<32x32xf16>) outs(%alloc_15 : memref<32x32xf32>)
            linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%alloc_15, %alloc_5 : memref<32x32xf32>, memref<32x32xf32>) outs(%alloc_15 : memref<32x32xf32>) {
            ^bb0(%in: f32, %in_24: f32, %out: f32):
              %14 = arith.mulf %in, %in_24 : f32
              linalg.yield %14 : f32
            }
            linalg.transpose ins(%alloc_15 : memref<32x32xf32>) outs(%alloc_4 : memref<32x32xf32>) permutation = [1, 0] 
            %alloc_16 = memref.alloc() {alignment = 64 : i64} : memref<32xf32>
            memref.copy %alloc_7, %alloc_16 : memref<32xf32> to memref<32xf32>
            linalg.reduce ins(%alloc_4 : memref<32x32xf32>) outs(%alloc_16 : memref<32xf32>) dimensions = [0] 
              (%in: f32, %init: f32) {
                %14 = arith.maxnumf %in, %init : f32
                linalg.yield %14 : f32
              }
            %alloc_17 = memref.alloc() {alignment = 64 : i64} : memref<32xf32>
            linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%arg13, %alloc_16 : memref<32xf32>, memref<32xf32>) outs(%alloc_17 : memref<32xf32>) {
            ^bb0(%in: f32, %in_24: f32, %out: f32):
              %14 = arith.maxnumf %in, %in_24 : f32
              linalg.yield %14 : f32
            }
            %expand_shape_18 = memref.expand_shape %alloc_17 [[0, 1]] output_shape [32, 1] : memref<32xf32> into memref<32x1xf32>
            linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, 0)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%expand_shape_18 : memref<32x1xf32>) outs(%alloc_4 : memref<32x32xf32>) attrs =  {broadcastDims = array<i64: 1>} {
            ^bb0(%in: f32, %out: f32):
              linalg.yield %in : f32
            }
            linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%alloc_15, %alloc_4 : memref<32x32xf32>, memref<32x32xf32>) outs(%alloc_15 : memref<32x32xf32>) {
            ^bb0(%in: f32, %in_24: f32, %out: f32):
              %14 = arith.subf %in, %in_24 : f32
              linalg.yield %14 : f32
            }
            linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%alloc_15 : memref<32x32xf32>) outs(%alloc_15 : memref<32x32xf32>) {
            ^bb0(%in: f32, %out: f32):
              %14 = math.exp %in : f32
              linalg.yield %14 : f32
            }
            linalg.transpose ins(%alloc_15 : memref<32x32xf32>) outs(%alloc_4 : memref<32x32xf32>) permutation = [1, 0] 
            linalg.fill ins(%cst_1 : f32) outs(%alloc : memref<32xf32>)
            linalg.reduce ins(%alloc_4 : memref<32x32xf32>) outs(%alloc : memref<32xf32>) dimensions = [0] 
              (%in: f32, %init: f32) {
                %14 = arith.addf %in, %init : f32
                linalg.yield %14 : f32
              }
            linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%arg13, %alloc_17 : memref<32xf32>, memref<32xf32>) outs(%arg13 : memref<32xf32>) {
            ^bb0(%in: f32, %in_24: f32, %out: f32):
              %14 = arith.subf %in, %in_24 : f32
              linalg.yield %14 : f32
            }
            linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%arg13 : memref<32xf32>) outs(%arg13 : memref<32xf32>) {
            ^bb0(%in: f32, %out: f32):
              %14 = math.exp %in : f32
              linalg.yield %14 : f32
            }
            linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%arg14, %arg13 : memref<32xf32>, memref<32xf32>) outs(%arg14 : memref<32xf32>) {
            ^bb0(%in: f32, %in_24: f32, %out: f32):
              %14 = arith.mulf %in, %in_24 : f32
              linalg.yield %14 : f32
            }
            linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%arg14, %alloc : memref<32xf32>, memref<32xf32>) outs(%arg14 : memref<32xf32>) {
            ^bb0(%in: f32, %in_24: f32, %out: f32):
              %14 = arith.addf %in, %in_24 : f32
              linalg.yield %14 : f32
            }
            %11 = affine.apply affine_map<()[s0] -> (s0 * 32)>()[%arg11]
            %reinterpret_cast_19 = memref.reinterpret_cast %arg2 to offset: [%11], sizes: [32, 32], strides: [32, 1] {loom.reuse = {sequential = [{depth = 3 : i64, iterator = "%arg10", reuse_type = "total_reuse", volume = 2048 : i64}], spatial = [{depth = 1 : i64, iterator = "%arg8", mapped_to = "y", reuse_type = "total_reuse", volume = 2048 : i64}, {depth = 2 : i64, iterator = "%arg9", mapped_to = "x", reuse_type = "total_reuse", volume = 2048 : i64}], temporal = [{depth = 0 : i64, iterator = "%arg7", reuse_type = "total_reuse", volume = 2048 : i64}]}} : memref<*xf16> to memref<32x32xf16, strided<[32, 1], offset: ?>>
            %alloc_20 = memref.alloc() {loom.alloc = {local = true, memory_name = "L1", size = 2048 : i64}} : memref<32x32xf16>
            memref.copy %reinterpret_cast_19, %alloc_20 {loom.copy.choice = {kind = "mem", memory_name = "L1"}} : memref<32x32xf16, strided<[32, 1], offset: ?>> to memref<32x32xf16>
            %alloc_21 = memref.alloc() {alignment = 64 : i64} : memref<32x32xf16>
            linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%alloc_15 : memref<32x32xf32>) outs(%alloc_21 : memref<32x32xf16>) {
            ^bb0(%in: f32, %out: f16):
              %14 = arith.truncf %in : f32 to f16
              linalg.yield %14 : f16
            }
            %expand_shape_22 = memref.expand_shape %arg13 [[0, 1]] output_shape [32, 1] : memref<32xf32> into memref<32x1xf32>
            linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, 0)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%expand_shape_22 : memref<32x1xf32>) outs(%alloc_4 : memref<32x32xf32>) attrs =  {broadcastDims = array<i64: 1>} {
            ^bb0(%in: f32, %out: f32):
              linalg.yield %in : f32
            }
            linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%arg15, %alloc_4 : memref<32x32xf32>, memref<32x32xf32>) outs(%arg15 : memref<32x32xf32>) {
            ^bb0(%in: f32, %in_24: f32, %out: f32):
              %14 = arith.mulf %in, %in_24 : f32
              linalg.yield %14 : f32
            }
            %alloc_23 = memref.alloc() {alignment = 64 : i64} : memref<32x32xf32>
            memref.copy %alloc_6, %alloc_23 : memref<32x32xf32> to memref<32x32xf32>
            linalg.matmul ins(%alloc_21, %alloc_20 : memref<32x32xf16>, memref<32x32xf16>) outs(%alloc_23 : memref<32x32xf32>)
            linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%arg15, %alloc_23 : memref<32x32xf32>, memref<32x32xf32>) outs(%arg15 : memref<32x32xf32>) {
            ^bb0(%in: f32, %in_24: f32, %out: f32):
              %14 = arith.addf %in, %in_24 : f32
              linalg.yield %14 : f32
            }
            %12 = arith.addi %arg11, %c32 : index
            %13 = arith.addi %arg12, %c32 : index
            scf.yield %12, %13, %alloc_17, %arg14, %arg15 : index, index, memref<32xf32>, memref<32xf32>, memref<32x32xf32>
          }
          %expand_shape = memref.expand_shape %8#3 [[0, 1]] output_shape [32, 1] : memref<32xf32> into memref<32x1xf32>
          linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, 0)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%expand_shape : memref<32x1xf32>) outs(%alloc_4 : memref<32x32xf32>) attrs =  {broadcastDims = array<i64: 1>} {
          ^bb0(%in: f32, %out: f32):
            linalg.yield %in : f32
          }
          linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%8#4, %alloc_4 : memref<32x32xf32>, memref<32x32xf32>) outs(%8#4 : memref<32x32xf32>) {
          ^bb0(%in: f32, %in_13: f32, %out: f32):
            %10 = arith.divf %in, %in_13 : f32
            linalg.yield %10 : f32
          }
          %alloc_11 = memref.alloc() {alignment = 64 : i64} : memref<32x32xf16>
          linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%8#4 : memref<32x32xf32>) outs(%alloc_11 : memref<32x32xf16>) {
          ^bb0(%in: f32, %out: f16):
            %10 = arith.truncf %in : f32 to f16
            linalg.yield %10 : f16
          }
          %9 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 1024 + d1 * 65536 + d2 * 8192)>(%arg9, %arg7, %arg8)
          %reinterpret_cast_12 = memref.reinterpret_cast %arg3 to offset: [%9], sizes: [32, 32], strides: [32, 1] {loom.reuse = {spatial = [{depth = 1 : i64, iterator = "%arg8", mapped_to = "y", reuse_type = "no_reuse", volume = 0 : i64}, {depth = 2 : i64, iterator = "%arg9", mapped_to = "x", reuse_type = "no_reuse", volume = 0 : i64}], temporal = [{depth = 0 : i64, iterator = "%arg7", reuse_type = "no_reuse", volume = 0 : i64}]}} : memref<*xf16> to memref<32x32xf16, strided<[32, 1], offset: ?>>
          memref.copy %alloc_11, %reinterpret_cast_12 : memref<32x32xf16> to memref<32x32xf16, strided<[32, 1], offset: ?>>
        } {loom.mapped_to = "x"}
      } {loom.mapped_to = "y"}
    }
    return
  }
  func.func @flashattn_fwd__d0i0_d1i0__c0mem_c1by_c2mem(%arg0: memref<*xf16> {tt.divisibility = 16 : i32}, %arg1: memref<*xf16> {tt.divisibility = 16 : i32}, %arg2: memref<*xf16> {tt.divisibility = 16 : i32}, %arg3: memref<*xf16> {tt.divisibility = 16 : i32}, %arg4: index, %arg5: index, %arg6: index) {
    affine.for %arg7 = 0 to affine_map<(d0) -> ((d0 ceildiv 8) ceildiv 8)>(%arg4) {
      affine.parallel (%arg8) = (0) to (8) {
        affine.parallel (%arg9) = (0) to (8) {
          %cst = arith.constant 1.000000e+00 : f32
          %cst_0 = arith.constant 0.176776692 : f32
          %cst_1 = arith.constant 0.000000e+00 : f32
          %cst_2 = arith.constant 0xFF800000 : f32
          %alloc = memref.alloc() {alignment = 64 : i64} : memref<32xf32>
          %alloc_3 = memref.alloc() {alignment = 64 : i64} : memref<32xf32>
          linalg.fill ins(%cst : f32) outs(%alloc_3 : memref<32xf32>)
          %alloc_4 = memref.alloc() {alignment = 64 : i64} : memref<32x32xf32>
          %alloc_5 = memref.alloc() {alignment = 64 : i64} : memref<32x32xf32>
          linalg.fill ins(%cst_0 : f32) outs(%alloc_5 : memref<32x32xf32>)
          %alloc_6 = memref.alloc() {alignment = 64 : i64} : memref<32x32xf32>
          linalg.fill ins(%cst_1 : f32) outs(%alloc_6 : memref<32x32xf32>)
          %alloc_7 = memref.alloc() {alignment = 64 : i64} : memref<32xf32>
          linalg.fill ins(%cst_2 : f32) outs(%alloc_7 : memref<32xf32>)
          %7 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 1024 + d1 * 65536 + d2 * 8192)>(%arg9, %arg7, %arg8)
          %reinterpret_cast = memref.reinterpret_cast %arg0 to offset: [%7], sizes: [32, 32], strides: [32, 1] {loom.reuse = {spatial = [{depth = 1 : i64, iterator = "%arg8", mapped_to = "y", reuse_type = "no_reuse", volume = 0 : i64}, {depth = 2 : i64, iterator = "%arg9", mapped_to = "x", reuse_type = "no_reuse", volume = 0 : i64}], temporal = [{depth = 0 : i64, iterator = "%arg7", reuse_type = "no_reuse", volume = 0 : i64}]}} : memref<*xf16> to memref<32x32xf16, strided<[32, 1], offset: ?>>
          %alloc_8 = memref.alloc() {loom.alloc = {local = true, memory_name = "L1", size = 2048 : i64}} : memref<32x32xf16>
          memref.copy %reinterpret_cast, %alloc_8 {loom.copy.choice = {kind = "mem", memory_name = "L1"}} : memref<32x32xf16, strided<[32, 1], offset: ?>> to memref<32x32xf16>
          %c0 = arith.constant 0 : index
          %c512 = arith.constant 512 : index
          %c32 = arith.constant 32 : index
          %alloc_9 = memref.alloc() {alignment = 64 : i64} : memref<32xf32>
          memref.copy %alloc_7, %alloc_9 : memref<32xf32> to memref<32xf32>
          %alloc_10 = memref.alloc() {alignment = 64 : i64} : memref<32x32xf32>
          memref.copy %alloc_6, %alloc_10 : memref<32x32xf32> to memref<32x32xf32>
          %8:5 = scf.for %arg10 = %c0 to %c512 step %c32 iter_args(%arg11 = %c0, %arg12 = %c0, %arg13 = %alloc_9, %arg14 = %alloc_3, %arg15 = %alloc_10) -> (index, index, memref<32xf32>, memref<32xf32>, memref<32x32xf32>) {
            %10 = affine.apply affine_map<()[s0] -> (s0)>()[%arg12]
            %reinterpret_cast_13 = memref.reinterpret_cast %arg1 to offset: [%10], sizes: [32, 32], strides: [512, 1] {loom.reuse = {sequential = [{depth = 3 : i64, iterator = "%arg10", reuse_type = "total_reuse", volume = 2048 : i64}], spatial = [{depth = 1 : i64, iterator = "%arg8", mapped_to = "y", reuse_type = "total_reuse", volume = 2048 : i64}, {depth = 2 : i64, iterator = "%arg9", mapped_to = "x", reuse_type = "total_reuse", volume = 2048 : i64}], temporal = [{depth = 0 : i64, iterator = "%arg7", reuse_type = "total_reuse", volume = 2048 : i64}]}} : memref<*xf16> to memref<32x32xf16, strided<[512, 1], offset: ?>>
            %alloc_14 = memref.alloc() {loom.alloc = {local = true, memory_name = "L1", size = 2048 : i64}} : memref<32x32xf16>
            memref.copy %reinterpret_cast_13, %alloc_14 {loom.copy.choice = {dim = "y", interconnect_name = "vertical_links", kind = "broadcast"}} : memref<32x32xf16, strided<[512, 1], offset: ?>> to memref<32x32xf16>
            %alloc_15 = memref.alloc() {alignment = 64 : i64} : memref<32x32xf32>
            memref.copy %alloc_6, %alloc_15 : memref<32x32xf32> to memref<32x32xf32>
            linalg.matmul ins(%alloc_8, %alloc_14 : memref<32x32xf16>, memref<32x32xf16>) outs(%alloc_15 : memref<32x32xf32>)
            linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%alloc_15, %alloc_5 : memref<32x32xf32>, memref<32x32xf32>) outs(%alloc_15 : memref<32x32xf32>) {
            ^bb0(%in: f32, %in_24: f32, %out: f32):
              %14 = arith.mulf %in, %in_24 : f32
              linalg.yield %14 : f32
            }
            linalg.transpose ins(%alloc_15 : memref<32x32xf32>) outs(%alloc_4 : memref<32x32xf32>) permutation = [1, 0] 
            %alloc_16 = memref.alloc() {alignment = 64 : i64} : memref<32xf32>
            memref.copy %alloc_7, %alloc_16 : memref<32xf32> to memref<32xf32>
            linalg.reduce ins(%alloc_4 : memref<32x32xf32>) outs(%alloc_16 : memref<32xf32>) dimensions = [0] 
              (%in: f32, %init: f32) {
                %14 = arith.maxnumf %in, %init : f32
                linalg.yield %14 : f32
              }
            %alloc_17 = memref.alloc() {alignment = 64 : i64} : memref<32xf32>
            linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%arg13, %alloc_16 : memref<32xf32>, memref<32xf32>) outs(%alloc_17 : memref<32xf32>) {
            ^bb0(%in: f32, %in_24: f32, %out: f32):
              %14 = arith.maxnumf %in, %in_24 : f32
              linalg.yield %14 : f32
            }
            %expand_shape_18 = memref.expand_shape %alloc_17 [[0, 1]] output_shape [32, 1] : memref<32xf32> into memref<32x1xf32>
            linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, 0)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%expand_shape_18 : memref<32x1xf32>) outs(%alloc_4 : memref<32x32xf32>) attrs =  {broadcastDims = array<i64: 1>} {
            ^bb0(%in: f32, %out: f32):
              linalg.yield %in : f32
            }
            linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%alloc_15, %alloc_4 : memref<32x32xf32>, memref<32x32xf32>) outs(%alloc_15 : memref<32x32xf32>) {
            ^bb0(%in: f32, %in_24: f32, %out: f32):
              %14 = arith.subf %in, %in_24 : f32
              linalg.yield %14 : f32
            }
            linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%alloc_15 : memref<32x32xf32>) outs(%alloc_15 : memref<32x32xf32>) {
            ^bb0(%in: f32, %out: f32):
              %14 = math.exp %in : f32
              linalg.yield %14 : f32
            }
            linalg.transpose ins(%alloc_15 : memref<32x32xf32>) outs(%alloc_4 : memref<32x32xf32>) permutation = [1, 0] 
            linalg.fill ins(%cst_1 : f32) outs(%alloc : memref<32xf32>)
            linalg.reduce ins(%alloc_4 : memref<32x32xf32>) outs(%alloc : memref<32xf32>) dimensions = [0] 
              (%in: f32, %init: f32) {
                %14 = arith.addf %in, %init : f32
                linalg.yield %14 : f32
              }
            linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%arg13, %alloc_17 : memref<32xf32>, memref<32xf32>) outs(%arg13 : memref<32xf32>) {
            ^bb0(%in: f32, %in_24: f32, %out: f32):
              %14 = arith.subf %in, %in_24 : f32
              linalg.yield %14 : f32
            }
            linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%arg13 : memref<32xf32>) outs(%arg13 : memref<32xf32>) {
            ^bb0(%in: f32, %out: f32):
              %14 = math.exp %in : f32
              linalg.yield %14 : f32
            }
            linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%arg14, %arg13 : memref<32xf32>, memref<32xf32>) outs(%arg14 : memref<32xf32>) {
            ^bb0(%in: f32, %in_24: f32, %out: f32):
              %14 = arith.mulf %in, %in_24 : f32
              linalg.yield %14 : f32
            }
            linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%arg14, %alloc : memref<32xf32>, memref<32xf32>) outs(%arg14 : memref<32xf32>) {
            ^bb0(%in: f32, %in_24: f32, %out: f32):
              %14 = arith.addf %in, %in_24 : f32
              linalg.yield %14 : f32
            }
            %11 = affine.apply affine_map<()[s0] -> (s0 * 32)>()[%arg11]
            %reinterpret_cast_19 = memref.reinterpret_cast %arg2 to offset: [%11], sizes: [32, 32], strides: [32, 1] {loom.reuse = {sequential = [{depth = 3 : i64, iterator = "%arg10", reuse_type = "total_reuse", volume = 2048 : i64}], spatial = [{depth = 1 : i64, iterator = "%arg8", mapped_to = "y", reuse_type = "total_reuse", volume = 2048 : i64}, {depth = 2 : i64, iterator = "%arg9", mapped_to = "x", reuse_type = "total_reuse", volume = 2048 : i64}], temporal = [{depth = 0 : i64, iterator = "%arg7", reuse_type = "total_reuse", volume = 2048 : i64}]}} : memref<*xf16> to memref<32x32xf16, strided<[32, 1], offset: ?>>
            %alloc_20 = memref.alloc() {loom.alloc = {local = true, memory_name = "L1", size = 2048 : i64}} : memref<32x32xf16>
            memref.copy %reinterpret_cast_19, %alloc_20 {loom.copy.choice = {kind = "mem", memory_name = "L1"}} : memref<32x32xf16, strided<[32, 1], offset: ?>> to memref<32x32xf16>
            %alloc_21 = memref.alloc() {alignment = 64 : i64} : memref<32x32xf16>
            linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%alloc_15 : memref<32x32xf32>) outs(%alloc_21 : memref<32x32xf16>) {
            ^bb0(%in: f32, %out: f16):
              %14 = arith.truncf %in : f32 to f16
              linalg.yield %14 : f16
            }
            %expand_shape_22 = memref.expand_shape %arg13 [[0, 1]] output_shape [32, 1] : memref<32xf32> into memref<32x1xf32>
            linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, 0)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%expand_shape_22 : memref<32x1xf32>) outs(%alloc_4 : memref<32x32xf32>) attrs =  {broadcastDims = array<i64: 1>} {
            ^bb0(%in: f32, %out: f32):
              linalg.yield %in : f32
            }
            linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%arg15, %alloc_4 : memref<32x32xf32>, memref<32x32xf32>) outs(%arg15 : memref<32x32xf32>) {
            ^bb0(%in: f32, %in_24: f32, %out: f32):
              %14 = arith.mulf %in, %in_24 : f32
              linalg.yield %14 : f32
            }
            %alloc_23 = memref.alloc() {alignment = 64 : i64} : memref<32x32xf32>
            memref.copy %alloc_6, %alloc_23 : memref<32x32xf32> to memref<32x32xf32>
            linalg.matmul ins(%alloc_21, %alloc_20 : memref<32x32xf16>, memref<32x32xf16>) outs(%alloc_23 : memref<32x32xf32>)
            linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%arg15, %alloc_23 : memref<32x32xf32>, memref<32x32xf32>) outs(%arg15 : memref<32x32xf32>) {
            ^bb0(%in: f32, %in_24: f32, %out: f32):
              %14 = arith.addf %in, %in_24 : f32
              linalg.yield %14 : f32
            }
            %12 = arith.addi %arg11, %c32 : index
            %13 = arith.addi %arg12, %c32 : index
            scf.yield %12, %13, %alloc_17, %arg14, %arg15 : index, index, memref<32xf32>, memref<32xf32>, memref<32x32xf32>
          }
          %expand_shape = memref.expand_shape %8#3 [[0, 1]] output_shape [32, 1] : memref<32xf32> into memref<32x1xf32>
          linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, 0)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%expand_shape : memref<32x1xf32>) outs(%alloc_4 : memref<32x32xf32>) attrs =  {broadcastDims = array<i64: 1>} {
          ^bb0(%in: f32, %out: f32):
            linalg.yield %in : f32
          }
          linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%8#4, %alloc_4 : memref<32x32xf32>, memref<32x32xf32>) outs(%8#4 : memref<32x32xf32>) {
          ^bb0(%in: f32, %in_13: f32, %out: f32):
            %10 = arith.divf %in, %in_13 : f32
            linalg.yield %10 : f32
          }
          %alloc_11 = memref.alloc() {alignment = 64 : i64} : memref<32x32xf16>
          linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%8#4 : memref<32x32xf32>) outs(%alloc_11 : memref<32x32xf16>) {
          ^bb0(%in: f32, %out: f16):
            %10 = arith.truncf %in : f32 to f16
            linalg.yield %10 : f16
          }
          %9 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 1024 + d1 * 65536 + d2 * 8192)>(%arg9, %arg7, %arg8)
          %reinterpret_cast_12 = memref.reinterpret_cast %arg3 to offset: [%9], sizes: [32, 32], strides: [32, 1] {loom.reuse = {spatial = [{depth = 1 : i64, iterator = "%arg8", mapped_to = "y", reuse_type = "no_reuse", volume = 0 : i64}, {depth = 2 : i64, iterator = "%arg9", mapped_to = "x", reuse_type = "no_reuse", volume = 0 : i64}], temporal = [{depth = 0 : i64, iterator = "%arg7", reuse_type = "no_reuse", volume = 0 : i64}]}} : memref<*xf16> to memref<32x32xf16, strided<[32, 1], offset: ?>>
          memref.copy %alloc_11, %reinterpret_cast_12 : memref<32x32xf16> to memref<32x32xf16, strided<[32, 1], offset: ?>>
        } {loom.mapped_to = "x"}
      } {loom.mapped_to = "y"}
    }
    return
  }
  func.func @flashattn_fwd__d0i0_d1i0__c0mem_c1mem_c2bx(%arg0: memref<*xf16> {tt.divisibility = 16 : i32}, %arg1: memref<*xf16> {tt.divisibility = 16 : i32}, %arg2: memref<*xf16> {tt.divisibility = 16 : i32}, %arg3: memref<*xf16> {tt.divisibility = 16 : i32}, %arg4: index, %arg5: index, %arg6: index) {
    affine.for %arg7 = 0 to affine_map<(d0) -> ((d0 ceildiv 8) ceildiv 8)>(%arg4) {
      affine.parallel (%arg8) = (0) to (8) {
        affine.parallel (%arg9) = (0) to (8) {
          %cst = arith.constant 1.000000e+00 : f32
          %cst_0 = arith.constant 0.176776692 : f32
          %cst_1 = arith.constant 0.000000e+00 : f32
          %cst_2 = arith.constant 0xFF800000 : f32
          %alloc = memref.alloc() {alignment = 64 : i64} : memref<32xf32>
          %alloc_3 = memref.alloc() {alignment = 64 : i64} : memref<32xf32>
          linalg.fill ins(%cst : f32) outs(%alloc_3 : memref<32xf32>)
          %alloc_4 = memref.alloc() {alignment = 64 : i64} : memref<32x32xf32>
          %alloc_5 = memref.alloc() {alignment = 64 : i64} : memref<32x32xf32>
          linalg.fill ins(%cst_0 : f32) outs(%alloc_5 : memref<32x32xf32>)
          %alloc_6 = memref.alloc() {alignment = 64 : i64} : memref<32x32xf32>
          linalg.fill ins(%cst_1 : f32) outs(%alloc_6 : memref<32x32xf32>)
          %alloc_7 = memref.alloc() {alignment = 64 : i64} : memref<32xf32>
          linalg.fill ins(%cst_2 : f32) outs(%alloc_7 : memref<32xf32>)
          %7 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 1024 + d1 * 65536 + d2 * 8192)>(%arg9, %arg7, %arg8)
          %reinterpret_cast = memref.reinterpret_cast %arg0 to offset: [%7], sizes: [32, 32], strides: [32, 1] {loom.reuse = {spatial = [{depth = 1 : i64, iterator = "%arg8", mapped_to = "y", reuse_type = "no_reuse", volume = 0 : i64}, {depth = 2 : i64, iterator = "%arg9", mapped_to = "x", reuse_type = "no_reuse", volume = 0 : i64}], temporal = [{depth = 0 : i64, iterator = "%arg7", reuse_type = "no_reuse", volume = 0 : i64}]}} : memref<*xf16> to memref<32x32xf16, strided<[32, 1], offset: ?>>
          %alloc_8 = memref.alloc() {loom.alloc = {local = true, memory_name = "L1", size = 2048 : i64}} : memref<32x32xf16>
          memref.copy %reinterpret_cast, %alloc_8 {loom.copy.choice = {kind = "mem", memory_name = "L1"}} : memref<32x32xf16, strided<[32, 1], offset: ?>> to memref<32x32xf16>
          %c0 = arith.constant 0 : index
          %c512 = arith.constant 512 : index
          %c32 = arith.constant 32 : index
          %alloc_9 = memref.alloc() {alignment = 64 : i64} : memref<32xf32>
          memref.copy %alloc_7, %alloc_9 : memref<32xf32> to memref<32xf32>
          %alloc_10 = memref.alloc() {alignment = 64 : i64} : memref<32x32xf32>
          memref.copy %alloc_6, %alloc_10 : memref<32x32xf32> to memref<32x32xf32>
          %8:5 = scf.for %arg10 = %c0 to %c512 step %c32 iter_args(%arg11 = %c0, %arg12 = %c0, %arg13 = %alloc_9, %arg14 = %alloc_3, %arg15 = %alloc_10) -> (index, index, memref<32xf32>, memref<32xf32>, memref<32x32xf32>) {
            %10 = affine.apply affine_map<()[s0] -> (s0)>()[%arg12]
            %reinterpret_cast_13 = memref.reinterpret_cast %arg1 to offset: [%10], sizes: [32, 32], strides: [512, 1] {loom.reuse = {sequential = [{depth = 3 : i64, iterator = "%arg10", reuse_type = "total_reuse", volume = 2048 : i64}], spatial = [{depth = 1 : i64, iterator = "%arg8", mapped_to = "y", reuse_type = "total_reuse", volume = 2048 : i64}, {depth = 2 : i64, iterator = "%arg9", mapped_to = "x", reuse_type = "total_reuse", volume = 2048 : i64}], temporal = [{depth = 0 : i64, iterator = "%arg7", reuse_type = "total_reuse", volume = 2048 : i64}]}} : memref<*xf16> to memref<32x32xf16, strided<[512, 1], offset: ?>>
            %alloc_14 = memref.alloc() {loom.alloc = {local = true, memory_name = "L1", size = 2048 : i64}} : memref<32x32xf16>
            memref.copy %reinterpret_cast_13, %alloc_14 {loom.copy.choice = {kind = "mem", memory_name = "L1"}} : memref<32x32xf16, strided<[512, 1], offset: ?>> to memref<32x32xf16>
            %alloc_15 = memref.alloc() {alignment = 64 : i64} : memref<32x32xf32>
            memref.copy %alloc_6, %alloc_15 : memref<32x32xf32> to memref<32x32xf32>
            linalg.matmul ins(%alloc_8, %alloc_14 : memref<32x32xf16>, memref<32x32xf16>) outs(%alloc_15 : memref<32x32xf32>)
            linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%alloc_15, %alloc_5 : memref<32x32xf32>, memref<32x32xf32>) outs(%alloc_15 : memref<32x32xf32>) {
            ^bb0(%in: f32, %in_24: f32, %out: f32):
              %14 = arith.mulf %in, %in_24 : f32
              linalg.yield %14 : f32
            }
            linalg.transpose ins(%alloc_15 : memref<32x32xf32>) outs(%alloc_4 : memref<32x32xf32>) permutation = [1, 0] 
            %alloc_16 = memref.alloc() {alignment = 64 : i64} : memref<32xf32>
            memref.copy %alloc_7, %alloc_16 : memref<32xf32> to memref<32xf32>
            linalg.reduce ins(%alloc_4 : memref<32x32xf32>) outs(%alloc_16 : memref<32xf32>) dimensions = [0] 
              (%in: f32, %init: f32) {
                %14 = arith.maxnumf %in, %init : f32
                linalg.yield %14 : f32
              }
            %alloc_17 = memref.alloc() {alignment = 64 : i64} : memref<32xf32>
            linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%arg13, %alloc_16 : memref<32xf32>, memref<32xf32>) outs(%alloc_17 : memref<32xf32>) {
            ^bb0(%in: f32, %in_24: f32, %out: f32):
              %14 = arith.maxnumf %in, %in_24 : f32
              linalg.yield %14 : f32
            }
            %expand_shape_18 = memref.expand_shape %alloc_17 [[0, 1]] output_shape [32, 1] : memref<32xf32> into memref<32x1xf32>
            linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, 0)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%expand_shape_18 : memref<32x1xf32>) outs(%alloc_4 : memref<32x32xf32>) attrs =  {broadcastDims = array<i64: 1>} {
            ^bb0(%in: f32, %out: f32):
              linalg.yield %in : f32
            }
            linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%alloc_15, %alloc_4 : memref<32x32xf32>, memref<32x32xf32>) outs(%alloc_15 : memref<32x32xf32>) {
            ^bb0(%in: f32, %in_24: f32, %out: f32):
              %14 = arith.subf %in, %in_24 : f32
              linalg.yield %14 : f32
            }
            linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%alloc_15 : memref<32x32xf32>) outs(%alloc_15 : memref<32x32xf32>) {
            ^bb0(%in: f32, %out: f32):
              %14 = math.exp %in : f32
              linalg.yield %14 : f32
            }
            linalg.transpose ins(%alloc_15 : memref<32x32xf32>) outs(%alloc_4 : memref<32x32xf32>) permutation = [1, 0] 
            linalg.fill ins(%cst_1 : f32) outs(%alloc : memref<32xf32>)
            linalg.reduce ins(%alloc_4 : memref<32x32xf32>) outs(%alloc : memref<32xf32>) dimensions = [0] 
              (%in: f32, %init: f32) {
                %14 = arith.addf %in, %init : f32
                linalg.yield %14 : f32
              }
            linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%arg13, %alloc_17 : memref<32xf32>, memref<32xf32>) outs(%arg13 : memref<32xf32>) {
            ^bb0(%in: f32, %in_24: f32, %out: f32):
              %14 = arith.subf %in, %in_24 : f32
              linalg.yield %14 : f32
            }
            linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%arg13 : memref<32xf32>) outs(%arg13 : memref<32xf32>) {
            ^bb0(%in: f32, %out: f32):
              %14 = math.exp %in : f32
              linalg.yield %14 : f32
            }
            linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%arg14, %arg13 : memref<32xf32>, memref<32xf32>) outs(%arg14 : memref<32xf32>) {
            ^bb0(%in: f32, %in_24: f32, %out: f32):
              %14 = arith.mulf %in, %in_24 : f32
              linalg.yield %14 : f32
            }
            linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%arg14, %alloc : memref<32xf32>, memref<32xf32>) outs(%arg14 : memref<32xf32>) {
            ^bb0(%in: f32, %in_24: f32, %out: f32):
              %14 = arith.addf %in, %in_24 : f32
              linalg.yield %14 : f32
            }
            %11 = affine.apply affine_map<()[s0] -> (s0 * 32)>()[%arg11]
            %reinterpret_cast_19 = memref.reinterpret_cast %arg2 to offset: [%11], sizes: [32, 32], strides: [32, 1] {loom.reuse = {sequential = [{depth = 3 : i64, iterator = "%arg10", reuse_type = "total_reuse", volume = 2048 : i64}], spatial = [{depth = 1 : i64, iterator = "%arg8", mapped_to = "y", reuse_type = "total_reuse", volume = 2048 : i64}, {depth = 2 : i64, iterator = "%arg9", mapped_to = "x", reuse_type = "total_reuse", volume = 2048 : i64}], temporal = [{depth = 0 : i64, iterator = "%arg7", reuse_type = "total_reuse", volume = 2048 : i64}]}} : memref<*xf16> to memref<32x32xf16, strided<[32, 1], offset: ?>>
            %alloc_20 = memref.alloc() {loom.alloc = {local = true, memory_name = "L1", size = 2048 : i64}} : memref<32x32xf16>
            memref.copy %reinterpret_cast_19, %alloc_20 {loom.copy.choice = {dim = "x", interconnect_name = "horizontal_links", kind = "broadcast"}} : memref<32x32xf16, strided<[32, 1], offset: ?>> to memref<32x32xf16>
            %alloc_21 = memref.alloc() {alignment = 64 : i64} : memref<32x32xf16>
            linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%alloc_15 : memref<32x32xf32>) outs(%alloc_21 : memref<32x32xf16>) {
            ^bb0(%in: f32, %out: f16):
              %14 = arith.truncf %in : f32 to f16
              linalg.yield %14 : f16
            }
            %expand_shape_22 = memref.expand_shape %arg13 [[0, 1]] output_shape [32, 1] : memref<32xf32> into memref<32x1xf32>
            linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, 0)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%expand_shape_22 : memref<32x1xf32>) outs(%alloc_4 : memref<32x32xf32>) attrs =  {broadcastDims = array<i64: 1>} {
            ^bb0(%in: f32, %out: f32):
              linalg.yield %in : f32
            }
            linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%arg15, %alloc_4 : memref<32x32xf32>, memref<32x32xf32>) outs(%arg15 : memref<32x32xf32>) {
            ^bb0(%in: f32, %in_24: f32, %out: f32):
              %14 = arith.mulf %in, %in_24 : f32
              linalg.yield %14 : f32
            }
            %alloc_23 = memref.alloc() {alignment = 64 : i64} : memref<32x32xf32>
            memref.copy %alloc_6, %alloc_23 : memref<32x32xf32> to memref<32x32xf32>
            linalg.matmul ins(%alloc_21, %alloc_20 : memref<32x32xf16>, memref<32x32xf16>) outs(%alloc_23 : memref<32x32xf32>)
            linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%arg15, %alloc_23 : memref<32x32xf32>, memref<32x32xf32>) outs(%arg15 : memref<32x32xf32>) {
            ^bb0(%in: f32, %in_24: f32, %out: f32):
              %14 = arith.addf %in, %in_24 : f32
              linalg.yield %14 : f32
            }
            %12 = arith.addi %arg11, %c32 : index
            %13 = arith.addi %arg12, %c32 : index
            scf.yield %12, %13, %alloc_17, %arg14, %arg15 : index, index, memref<32xf32>, memref<32xf32>, memref<32x32xf32>
          }
          %expand_shape = memref.expand_shape %8#3 [[0, 1]] output_shape [32, 1] : memref<32xf32> into memref<32x1xf32>
          linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, 0)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%expand_shape : memref<32x1xf32>) outs(%alloc_4 : memref<32x32xf32>) attrs =  {broadcastDims = array<i64: 1>} {
          ^bb0(%in: f32, %out: f32):
            linalg.yield %in : f32
          }
          linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%8#4, %alloc_4 : memref<32x32xf32>, memref<32x32xf32>) outs(%8#4 : memref<32x32xf32>) {
          ^bb0(%in: f32, %in_13: f32, %out: f32):
            %10 = arith.divf %in, %in_13 : f32
            linalg.yield %10 : f32
          }
          %alloc_11 = memref.alloc() {alignment = 64 : i64} : memref<32x32xf16>
          linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%8#4 : memref<32x32xf32>) outs(%alloc_11 : memref<32x32xf16>) {
          ^bb0(%in: f32, %out: f16):
            %10 = arith.truncf %in : f32 to f16
            linalg.yield %10 : f16
          }
          %9 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 1024 + d1 * 65536 + d2 * 8192)>(%arg9, %arg7, %arg8)
          %reinterpret_cast_12 = memref.reinterpret_cast %arg3 to offset: [%9], sizes: [32, 32], strides: [32, 1] {loom.reuse = {spatial = [{depth = 1 : i64, iterator = "%arg8", mapped_to = "y", reuse_type = "no_reuse", volume = 0 : i64}, {depth = 2 : i64, iterator = "%arg9", mapped_to = "x", reuse_type = "no_reuse", volume = 0 : i64}], temporal = [{depth = 0 : i64, iterator = "%arg7", reuse_type = "no_reuse", volume = 0 : i64}]}} : memref<*xf16> to memref<32x32xf16, strided<[32, 1], offset: ?>>
          memref.copy %alloc_11, %reinterpret_cast_12 : memref<32x32xf16> to memref<32x32xf16, strided<[32, 1], offset: ?>>
        } {loom.mapped_to = "x"}
      } {loom.mapped_to = "y"}
    }
    return
  }
  func.func @flashattn_fwd__d0i0_d1i0__c0mem_c1bx_c2bx(%arg0: memref<*xf16> {tt.divisibility = 16 : i32}, %arg1: memref<*xf16> {tt.divisibility = 16 : i32}, %arg2: memref<*xf16> {tt.divisibility = 16 : i32}, %arg3: memref<*xf16> {tt.divisibility = 16 : i32}, %arg4: index, %arg5: index, %arg6: index) {
    affine.for %arg7 = 0 to affine_map<(d0) -> ((d0 ceildiv 8) ceildiv 8)>(%arg4) {
      affine.parallel (%arg8) = (0) to (8) {
        affine.parallel (%arg9) = (0) to (8) {
          %cst = arith.constant 1.000000e+00 : f32
          %cst_0 = arith.constant 0.176776692 : f32
          %cst_1 = arith.constant 0.000000e+00 : f32
          %cst_2 = arith.constant 0xFF800000 : f32
          %alloc = memref.alloc() {alignment = 64 : i64} : memref<32xf32>
          %alloc_3 = memref.alloc() {alignment = 64 : i64} : memref<32xf32>
          linalg.fill ins(%cst : f32) outs(%alloc_3 : memref<32xf32>)
          %alloc_4 = memref.alloc() {alignment = 64 : i64} : memref<32x32xf32>
          %alloc_5 = memref.alloc() {alignment = 64 : i64} : memref<32x32xf32>
          linalg.fill ins(%cst_0 : f32) outs(%alloc_5 : memref<32x32xf32>)
          %alloc_6 = memref.alloc() {alignment = 64 : i64} : memref<32x32xf32>
          linalg.fill ins(%cst_1 : f32) outs(%alloc_6 : memref<32x32xf32>)
          %alloc_7 = memref.alloc() {alignment = 64 : i64} : memref<32xf32>
          linalg.fill ins(%cst_2 : f32) outs(%alloc_7 : memref<32xf32>)
          %7 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 1024 + d1 * 65536 + d2 * 8192)>(%arg9, %arg7, %arg8)
          %reinterpret_cast = memref.reinterpret_cast %arg0 to offset: [%7], sizes: [32, 32], strides: [32, 1] {loom.reuse = {spatial = [{depth = 1 : i64, iterator = "%arg8", mapped_to = "y", reuse_type = "no_reuse", volume = 0 : i64}, {depth = 2 : i64, iterator = "%arg9", mapped_to = "x", reuse_type = "no_reuse", volume = 0 : i64}], temporal = [{depth = 0 : i64, iterator = "%arg7", reuse_type = "no_reuse", volume = 0 : i64}]}} : memref<*xf16> to memref<32x32xf16, strided<[32, 1], offset: ?>>
          %alloc_8 = memref.alloc() {loom.alloc = {local = true, memory_name = "L1", size = 2048 : i64}} : memref<32x32xf16>
          memref.copy %reinterpret_cast, %alloc_8 {loom.copy.choice = {kind = "mem", memory_name = "L1"}} : memref<32x32xf16, strided<[32, 1], offset: ?>> to memref<32x32xf16>
          %c0 = arith.constant 0 : index
          %c512 = arith.constant 512 : index
          %c32 = arith.constant 32 : index
          %alloc_9 = memref.alloc() {alignment = 64 : i64} : memref<32xf32>
          memref.copy %alloc_7, %alloc_9 : memref<32xf32> to memref<32xf32>
          %alloc_10 = memref.alloc() {alignment = 64 : i64} : memref<32x32xf32>
          memref.copy %alloc_6, %alloc_10 : memref<32x32xf32> to memref<32x32xf32>
          %8:5 = scf.for %arg10 = %c0 to %c512 step %c32 iter_args(%arg11 = %c0, %arg12 = %c0, %arg13 = %alloc_9, %arg14 = %alloc_3, %arg15 = %alloc_10) -> (index, index, memref<32xf32>, memref<32xf32>, memref<32x32xf32>) {
            %10 = affine.apply affine_map<()[s0] -> (s0)>()[%arg12]
            %reinterpret_cast_13 = memref.reinterpret_cast %arg1 to offset: [%10], sizes: [32, 32], strides: [512, 1] {loom.reuse = {sequential = [{depth = 3 : i64, iterator = "%arg10", reuse_type = "total_reuse", volume = 2048 : i64}], spatial = [{depth = 1 : i64, iterator = "%arg8", mapped_to = "y", reuse_type = "total_reuse", volume = 2048 : i64}, {depth = 2 : i64, iterator = "%arg9", mapped_to = "x", reuse_type = "total_reuse", volume = 2048 : i64}], temporal = [{depth = 0 : i64, iterator = "%arg7", reuse_type = "total_reuse", volume = 2048 : i64}]}} : memref<*xf16> to memref<32x32xf16, strided<[512, 1], offset: ?>>
            %alloc_14 = memref.alloc() {loom.alloc = {local = true, memory_name = "L1", size = 2048 : i64}} : memref<32x32xf16>
            memref.copy %reinterpret_cast_13, %alloc_14 {loom.copy.choice = {dim = "x", interconnect_name = "horizontal_links", kind = "broadcast"}} : memref<32x32xf16, strided<[512, 1], offset: ?>> to memref<32x32xf16>
            %alloc_15 = memref.alloc() {alignment = 64 : i64} : memref<32x32xf32>
            memref.copy %alloc_6, %alloc_15 : memref<32x32xf32> to memref<32x32xf32>
            linalg.matmul ins(%alloc_8, %alloc_14 : memref<32x32xf16>, memref<32x32xf16>) outs(%alloc_15 : memref<32x32xf32>)
            linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%alloc_15, %alloc_5 : memref<32x32xf32>, memref<32x32xf32>) outs(%alloc_15 : memref<32x32xf32>) {
            ^bb0(%in: f32, %in_24: f32, %out: f32):
              %14 = arith.mulf %in, %in_24 : f32
              linalg.yield %14 : f32
            }
            linalg.transpose ins(%alloc_15 : memref<32x32xf32>) outs(%alloc_4 : memref<32x32xf32>) permutation = [1, 0] 
            %alloc_16 = memref.alloc() {alignment = 64 : i64} : memref<32xf32>
            memref.copy %alloc_7, %alloc_16 : memref<32xf32> to memref<32xf32>
            linalg.reduce ins(%alloc_4 : memref<32x32xf32>) outs(%alloc_16 : memref<32xf32>) dimensions = [0] 
              (%in: f32, %init: f32) {
                %14 = arith.maxnumf %in, %init : f32
                linalg.yield %14 : f32
              }
            %alloc_17 = memref.alloc() {alignment = 64 : i64} : memref<32xf32>
            linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%arg13, %alloc_16 : memref<32xf32>, memref<32xf32>) outs(%alloc_17 : memref<32xf32>) {
            ^bb0(%in: f32, %in_24: f32, %out: f32):
              %14 = arith.maxnumf %in, %in_24 : f32
              linalg.yield %14 : f32
            }
            %expand_shape_18 = memref.expand_shape %alloc_17 [[0, 1]] output_shape [32, 1] : memref<32xf32> into memref<32x1xf32>
            linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, 0)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%expand_shape_18 : memref<32x1xf32>) outs(%alloc_4 : memref<32x32xf32>) attrs =  {broadcastDims = array<i64: 1>} {
            ^bb0(%in: f32, %out: f32):
              linalg.yield %in : f32
            }
            linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%alloc_15, %alloc_4 : memref<32x32xf32>, memref<32x32xf32>) outs(%alloc_15 : memref<32x32xf32>) {
            ^bb0(%in: f32, %in_24: f32, %out: f32):
              %14 = arith.subf %in, %in_24 : f32
              linalg.yield %14 : f32
            }
            linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%alloc_15 : memref<32x32xf32>) outs(%alloc_15 : memref<32x32xf32>) {
            ^bb0(%in: f32, %out: f32):
              %14 = math.exp %in : f32
              linalg.yield %14 : f32
            }
            linalg.transpose ins(%alloc_15 : memref<32x32xf32>) outs(%alloc_4 : memref<32x32xf32>) permutation = [1, 0] 
            linalg.fill ins(%cst_1 : f32) outs(%alloc : memref<32xf32>)
            linalg.reduce ins(%alloc_4 : memref<32x32xf32>) outs(%alloc : memref<32xf32>) dimensions = [0] 
              (%in: f32, %init: f32) {
                %14 = arith.addf %in, %init : f32
                linalg.yield %14 : f32
              }
            linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%arg13, %alloc_17 : memref<32xf32>, memref<32xf32>) outs(%arg13 : memref<32xf32>) {
            ^bb0(%in: f32, %in_24: f32, %out: f32):
              %14 = arith.subf %in, %in_24 : f32
              linalg.yield %14 : f32
            }
            linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%arg13 : memref<32xf32>) outs(%arg13 : memref<32xf32>) {
            ^bb0(%in: f32, %out: f32):
              %14 = math.exp %in : f32
              linalg.yield %14 : f32
            }
            linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%arg14, %arg13 : memref<32xf32>, memref<32xf32>) outs(%arg14 : memref<32xf32>) {
            ^bb0(%in: f32, %in_24: f32, %out: f32):
              %14 = arith.mulf %in, %in_24 : f32
              linalg.yield %14 : f32
            }
            linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%arg14, %alloc : memref<32xf32>, memref<32xf32>) outs(%arg14 : memref<32xf32>) {
            ^bb0(%in: f32, %in_24: f32, %out: f32):
              %14 = arith.addf %in, %in_24 : f32
              linalg.yield %14 : f32
            }
            %11 = affine.apply affine_map<()[s0] -> (s0 * 32)>()[%arg11]
            %reinterpret_cast_19 = memref.reinterpret_cast %arg2 to offset: [%11], sizes: [32, 32], strides: [32, 1] {loom.reuse = {sequential = [{depth = 3 : i64, iterator = "%arg10", reuse_type = "total_reuse", volume = 2048 : i64}], spatial = [{depth = 1 : i64, iterator = "%arg8", mapped_to = "y", reuse_type = "total_reuse", volume = 2048 : i64}, {depth = 2 : i64, iterator = "%arg9", mapped_to = "x", reuse_type = "total_reuse", volume = 2048 : i64}], temporal = [{depth = 0 : i64, iterator = "%arg7", reuse_type = "total_reuse", volume = 2048 : i64}]}} : memref<*xf16> to memref<32x32xf16, strided<[32, 1], offset: ?>>
            %alloc_20 = memref.alloc() {loom.alloc = {local = true, memory_name = "L1", size = 2048 : i64}} : memref<32x32xf16>
            memref.copy %reinterpret_cast_19, %alloc_20 {loom.copy.choice = {dim = "x", interconnect_name = "horizontal_links", kind = "broadcast"}} : memref<32x32xf16, strided<[32, 1], offset: ?>> to memref<32x32xf16>
            %alloc_21 = memref.alloc() {alignment = 64 : i64} : memref<32x32xf16>
            linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%alloc_15 : memref<32x32xf32>) outs(%alloc_21 : memref<32x32xf16>) {
            ^bb0(%in: f32, %out: f16):
              %14 = arith.truncf %in : f32 to f16
              linalg.yield %14 : f16
            }
            %expand_shape_22 = memref.expand_shape %arg13 [[0, 1]] output_shape [32, 1] : memref<32xf32> into memref<32x1xf32>
            linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, 0)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%expand_shape_22 : memref<32x1xf32>) outs(%alloc_4 : memref<32x32xf32>) attrs =  {broadcastDims = array<i64: 1>} {
            ^bb0(%in: f32, %out: f32):
              linalg.yield %in : f32
            }
            linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%arg15, %alloc_4 : memref<32x32xf32>, memref<32x32xf32>) outs(%arg15 : memref<32x32xf32>) {
            ^bb0(%in: f32, %in_24: f32, %out: f32):
              %14 = arith.mulf %in, %in_24 : f32
              linalg.yield %14 : f32
            }
            %alloc_23 = memref.alloc() {alignment = 64 : i64} : memref<32x32xf32>
            memref.copy %alloc_6, %alloc_23 : memref<32x32xf32> to memref<32x32xf32>
            linalg.matmul ins(%alloc_21, %alloc_20 : memref<32x32xf16>, memref<32x32xf16>) outs(%alloc_23 : memref<32x32xf32>)
            linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%arg15, %alloc_23 : memref<32x32xf32>, memref<32x32xf32>) outs(%arg15 : memref<32x32xf32>) {
            ^bb0(%in: f32, %in_24: f32, %out: f32):
              %14 = arith.addf %in, %in_24 : f32
              linalg.yield %14 : f32
            }
            %12 = arith.addi %arg11, %c32 : index
            %13 = arith.addi %arg12, %c32 : index
            scf.yield %12, %13, %alloc_17, %arg14, %arg15 : index, index, memref<32xf32>, memref<32xf32>, memref<32x32xf32>
          }
          %expand_shape = memref.expand_shape %8#3 [[0, 1]] output_shape [32, 1] : memref<32xf32> into memref<32x1xf32>
          linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, 0)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%expand_shape : memref<32x1xf32>) outs(%alloc_4 : memref<32x32xf32>) attrs =  {broadcastDims = array<i64: 1>} {
          ^bb0(%in: f32, %out: f32):
            linalg.yield %in : f32
          }
          linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%8#4, %alloc_4 : memref<32x32xf32>, memref<32x32xf32>) outs(%8#4 : memref<32x32xf32>) {
          ^bb0(%in: f32, %in_13: f32, %out: f32):
            %10 = arith.divf %in, %in_13 : f32
            linalg.yield %10 : f32
          }
          %alloc_11 = memref.alloc() {alignment = 64 : i64} : memref<32x32xf16>
          linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%8#4 : memref<32x32xf32>) outs(%alloc_11 : memref<32x32xf16>) {
          ^bb0(%in: f32, %out: f16):
            %10 = arith.truncf %in : f32 to f16
            linalg.yield %10 : f16
          }
          %9 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 1024 + d1 * 65536 + d2 * 8192)>(%arg9, %arg7, %arg8)
          %reinterpret_cast_12 = memref.reinterpret_cast %arg3 to offset: [%9], sizes: [32, 32], strides: [32, 1] {loom.reuse = {spatial = [{depth = 1 : i64, iterator = "%arg8", mapped_to = "y", reuse_type = "no_reuse", volume = 0 : i64}, {depth = 2 : i64, iterator = "%arg9", mapped_to = "x", reuse_type = "no_reuse", volume = 0 : i64}], temporal = [{depth = 0 : i64, iterator = "%arg7", reuse_type = "no_reuse", volume = 0 : i64}]}} : memref<*xf16> to memref<32x32xf16, strided<[32, 1], offset: ?>>
          memref.copy %alloc_11, %reinterpret_cast_12 : memref<32x32xf16> to memref<32x32xf16, strided<[32, 1], offset: ?>>
        } {loom.mapped_to = "x"}
      } {loom.mapped_to = "y"}
    }
    return
  }
  func.func @flashattn_fwd__d0i0_d1i0__c0mem_c1by_c2bx(%arg0: memref<*xf16> {tt.divisibility = 16 : i32}, %arg1: memref<*xf16> {tt.divisibility = 16 : i32}, %arg2: memref<*xf16> {tt.divisibility = 16 : i32}, %arg3: memref<*xf16> {tt.divisibility = 16 : i32}, %arg4: index, %arg5: index, %arg6: index) {
    affine.for %arg7 = 0 to affine_map<(d0) -> ((d0 ceildiv 8) ceildiv 8)>(%arg4) {
      affine.parallel (%arg8) = (0) to (8) {
        affine.parallel (%arg9) = (0) to (8) {
          %cst = arith.constant 1.000000e+00 : f32
          %cst_0 = arith.constant 0.176776692 : f32
          %cst_1 = arith.constant 0.000000e+00 : f32
          %cst_2 = arith.constant 0xFF800000 : f32
          %alloc = memref.alloc() {alignment = 64 : i64} : memref<32xf32>
          %alloc_3 = memref.alloc() {alignment = 64 : i64} : memref<32xf32>
          linalg.fill ins(%cst : f32) outs(%alloc_3 : memref<32xf32>)
          %alloc_4 = memref.alloc() {alignment = 64 : i64} : memref<32x32xf32>
          %alloc_5 = memref.alloc() {alignment = 64 : i64} : memref<32x32xf32>
          linalg.fill ins(%cst_0 : f32) outs(%alloc_5 : memref<32x32xf32>)
          %alloc_6 = memref.alloc() {alignment = 64 : i64} : memref<32x32xf32>
          linalg.fill ins(%cst_1 : f32) outs(%alloc_6 : memref<32x32xf32>)
          %alloc_7 = memref.alloc() {alignment = 64 : i64} : memref<32xf32>
          linalg.fill ins(%cst_2 : f32) outs(%alloc_7 : memref<32xf32>)
          %7 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 1024 + d1 * 65536 + d2 * 8192)>(%arg9, %arg7, %arg8)
          %reinterpret_cast = memref.reinterpret_cast %arg0 to offset: [%7], sizes: [32, 32], strides: [32, 1] {loom.reuse = {spatial = [{depth = 1 : i64, iterator = "%arg8", mapped_to = "y", reuse_type = "no_reuse", volume = 0 : i64}, {depth = 2 : i64, iterator = "%arg9", mapped_to = "x", reuse_type = "no_reuse", volume = 0 : i64}], temporal = [{depth = 0 : i64, iterator = "%arg7", reuse_type = "no_reuse", volume = 0 : i64}]}} : memref<*xf16> to memref<32x32xf16, strided<[32, 1], offset: ?>>
          %alloc_8 = memref.alloc() {loom.alloc = {local = true, memory_name = "L1", size = 2048 : i64}} : memref<32x32xf16>
          memref.copy %reinterpret_cast, %alloc_8 {loom.copy.choice = {kind = "mem", memory_name = "L1"}} : memref<32x32xf16, strided<[32, 1], offset: ?>> to memref<32x32xf16>
          %c0 = arith.constant 0 : index
          %c512 = arith.constant 512 : index
          %c32 = arith.constant 32 : index
          %alloc_9 = memref.alloc() {alignment = 64 : i64} : memref<32xf32>
          memref.copy %alloc_7, %alloc_9 : memref<32xf32> to memref<32xf32>
          %alloc_10 = memref.alloc() {alignment = 64 : i64} : memref<32x32xf32>
          memref.copy %alloc_6, %alloc_10 : memref<32x32xf32> to memref<32x32xf32>
          %8:5 = scf.for %arg10 = %c0 to %c512 step %c32 iter_args(%arg11 = %c0, %arg12 = %c0, %arg13 = %alloc_9, %arg14 = %alloc_3, %arg15 = %alloc_10) -> (index, index, memref<32xf32>, memref<32xf32>, memref<32x32xf32>) {
            %10 = affine.apply affine_map<()[s0] -> (s0)>()[%arg12]
            %reinterpret_cast_13 = memref.reinterpret_cast %arg1 to offset: [%10], sizes: [32, 32], strides: [512, 1] {loom.reuse = {sequential = [{depth = 3 : i64, iterator = "%arg10", reuse_type = "total_reuse", volume = 2048 : i64}], spatial = [{depth = 1 : i64, iterator = "%arg8", mapped_to = "y", reuse_type = "total_reuse", volume = 2048 : i64}, {depth = 2 : i64, iterator = "%arg9", mapped_to = "x", reuse_type = "total_reuse", volume = 2048 : i64}], temporal = [{depth = 0 : i64, iterator = "%arg7", reuse_type = "total_reuse", volume = 2048 : i64}]}} : memref<*xf16> to memref<32x32xf16, strided<[512, 1], offset: ?>>
            %alloc_14 = memref.alloc() {loom.alloc = {local = true, memory_name = "L1", size = 2048 : i64}} : memref<32x32xf16>
            memref.copy %reinterpret_cast_13, %alloc_14 {loom.copy.choice = {dim = "y", interconnect_name = "vertical_links", kind = "broadcast"}} : memref<32x32xf16, strided<[512, 1], offset: ?>> to memref<32x32xf16>
            %alloc_15 = memref.alloc() {alignment = 64 : i64} : memref<32x32xf32>
            memref.copy %alloc_6, %alloc_15 : memref<32x32xf32> to memref<32x32xf32>
            linalg.matmul ins(%alloc_8, %alloc_14 : memref<32x32xf16>, memref<32x32xf16>) outs(%alloc_15 : memref<32x32xf32>)
            linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%alloc_15, %alloc_5 : memref<32x32xf32>, memref<32x32xf32>) outs(%alloc_15 : memref<32x32xf32>) {
            ^bb0(%in: f32, %in_24: f32, %out: f32):
              %14 = arith.mulf %in, %in_24 : f32
              linalg.yield %14 : f32
            }
            linalg.transpose ins(%alloc_15 : memref<32x32xf32>) outs(%alloc_4 : memref<32x32xf32>) permutation = [1, 0] 
            %alloc_16 = memref.alloc() {alignment = 64 : i64} : memref<32xf32>
            memref.copy %alloc_7, %alloc_16 : memref<32xf32> to memref<32xf32>
            linalg.reduce ins(%alloc_4 : memref<32x32xf32>) outs(%alloc_16 : memref<32xf32>) dimensions = [0] 
              (%in: f32, %init: f32) {
                %14 = arith.maxnumf %in, %init : f32
                linalg.yield %14 : f32
              }
            %alloc_17 = memref.alloc() {alignment = 64 : i64} : memref<32xf32>
            linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%arg13, %alloc_16 : memref<32xf32>, memref<32xf32>) outs(%alloc_17 : memref<32xf32>) {
            ^bb0(%in: f32, %in_24: f32, %out: f32):
              %14 = arith.maxnumf %in, %in_24 : f32
              linalg.yield %14 : f32
            }
            %expand_shape_18 = memref.expand_shape %alloc_17 [[0, 1]] output_shape [32, 1] : memref<32xf32> into memref<32x1xf32>
            linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, 0)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%expand_shape_18 : memref<32x1xf32>) outs(%alloc_4 : memref<32x32xf32>) attrs =  {broadcastDims = array<i64: 1>} {
            ^bb0(%in: f32, %out: f32):
              linalg.yield %in : f32
            }
            linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%alloc_15, %alloc_4 : memref<32x32xf32>, memref<32x32xf32>) outs(%alloc_15 : memref<32x32xf32>) {
            ^bb0(%in: f32, %in_24: f32, %out: f32):
              %14 = arith.subf %in, %in_24 : f32
              linalg.yield %14 : f32
            }
            linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%alloc_15 : memref<32x32xf32>) outs(%alloc_15 : memref<32x32xf32>) {
            ^bb0(%in: f32, %out: f32):
              %14 = math.exp %in : f32
              linalg.yield %14 : f32
            }
            linalg.transpose ins(%alloc_15 : memref<32x32xf32>) outs(%alloc_4 : memref<32x32xf32>) permutation = [1, 0] 
            linalg.fill ins(%cst_1 : f32) outs(%alloc : memref<32xf32>)
            linalg.reduce ins(%alloc_4 : memref<32x32xf32>) outs(%alloc : memref<32xf32>) dimensions = [0] 
              (%in: f32, %init: f32) {
                %14 = arith.addf %in, %init : f32
                linalg.yield %14 : f32
              }
            linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%arg13, %alloc_17 : memref<32xf32>, memref<32xf32>) outs(%arg13 : memref<32xf32>) {
            ^bb0(%in: f32, %in_24: f32, %out: f32):
              %14 = arith.subf %in, %in_24 : f32
              linalg.yield %14 : f32
            }
            linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%arg13 : memref<32xf32>) outs(%arg13 : memref<32xf32>) {
            ^bb0(%in: f32, %out: f32):
              %14 = math.exp %in : f32
              linalg.yield %14 : f32
            }
            linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%arg14, %arg13 : memref<32xf32>, memref<32xf32>) outs(%arg14 : memref<32xf32>) {
            ^bb0(%in: f32, %in_24: f32, %out: f32):
              %14 = arith.mulf %in, %in_24 : f32
              linalg.yield %14 : f32
            }
            linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%arg14, %alloc : memref<32xf32>, memref<32xf32>) outs(%arg14 : memref<32xf32>) {
            ^bb0(%in: f32, %in_24: f32, %out: f32):
              %14 = arith.addf %in, %in_24 : f32
              linalg.yield %14 : f32
            }
            %11 = affine.apply affine_map<()[s0] -> (s0 * 32)>()[%arg11]
            %reinterpret_cast_19 = memref.reinterpret_cast %arg2 to offset: [%11], sizes: [32, 32], strides: [32, 1] {loom.reuse = {sequential = [{depth = 3 : i64, iterator = "%arg10", reuse_type = "total_reuse", volume = 2048 : i64}], spatial = [{depth = 1 : i64, iterator = "%arg8", mapped_to = "y", reuse_type = "total_reuse", volume = 2048 : i64}, {depth = 2 : i64, iterator = "%arg9", mapped_to = "x", reuse_type = "total_reuse", volume = 2048 : i64}], temporal = [{depth = 0 : i64, iterator = "%arg7", reuse_type = "total_reuse", volume = 2048 : i64}]}} : memref<*xf16> to memref<32x32xf16, strided<[32, 1], offset: ?>>
            %alloc_20 = memref.alloc() {loom.alloc = {local = true, memory_name = "L1", size = 2048 : i64}} : memref<32x32xf16>
            memref.copy %reinterpret_cast_19, %alloc_20 {loom.copy.choice = {dim = "x", interconnect_name = "horizontal_links", kind = "broadcast"}} : memref<32x32xf16, strided<[32, 1], offset: ?>> to memref<32x32xf16>
            %alloc_21 = memref.alloc() {alignment = 64 : i64} : memref<32x32xf16>
            linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%alloc_15 : memref<32x32xf32>) outs(%alloc_21 : memref<32x32xf16>) {
            ^bb0(%in: f32, %out: f16):
              %14 = arith.truncf %in : f32 to f16
              linalg.yield %14 : f16
            }
            %expand_shape_22 = memref.expand_shape %arg13 [[0, 1]] output_shape [32, 1] : memref<32xf32> into memref<32x1xf32>
            linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, 0)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%expand_shape_22 : memref<32x1xf32>) outs(%alloc_4 : memref<32x32xf32>) attrs =  {broadcastDims = array<i64: 1>} {
            ^bb0(%in: f32, %out: f32):
              linalg.yield %in : f32
            }
            linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%arg15, %alloc_4 : memref<32x32xf32>, memref<32x32xf32>) outs(%arg15 : memref<32x32xf32>) {
            ^bb0(%in: f32, %in_24: f32, %out: f32):
              %14 = arith.mulf %in, %in_24 : f32
              linalg.yield %14 : f32
            }
            %alloc_23 = memref.alloc() {alignment = 64 : i64} : memref<32x32xf32>
            memref.copy %alloc_6, %alloc_23 : memref<32x32xf32> to memref<32x32xf32>
            linalg.matmul ins(%alloc_21, %alloc_20 : memref<32x32xf16>, memref<32x32xf16>) outs(%alloc_23 : memref<32x32xf32>)
            linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%arg15, %alloc_23 : memref<32x32xf32>, memref<32x32xf32>) outs(%arg15 : memref<32x32xf32>) {
            ^bb0(%in: f32, %in_24: f32, %out: f32):
              %14 = arith.addf %in, %in_24 : f32
              linalg.yield %14 : f32
            }
            %12 = arith.addi %arg11, %c32 : index
            %13 = arith.addi %arg12, %c32 : index
            scf.yield %12, %13, %alloc_17, %arg14, %arg15 : index, index, memref<32xf32>, memref<32xf32>, memref<32x32xf32>
          }
          %expand_shape = memref.expand_shape %8#3 [[0, 1]] output_shape [32, 1] : memref<32xf32> into memref<32x1xf32>
          linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, 0)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%expand_shape : memref<32x1xf32>) outs(%alloc_4 : memref<32x32xf32>) attrs =  {broadcastDims = array<i64: 1>} {
          ^bb0(%in: f32, %out: f32):
            linalg.yield %in : f32
          }
          linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%8#4, %alloc_4 : memref<32x32xf32>, memref<32x32xf32>) outs(%8#4 : memref<32x32xf32>) {
          ^bb0(%in: f32, %in_13: f32, %out: f32):
            %10 = arith.divf %in, %in_13 : f32
            linalg.yield %10 : f32
          }
          %alloc_11 = memref.alloc() {alignment = 64 : i64} : memref<32x32xf16>
          linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%8#4 : memref<32x32xf32>) outs(%alloc_11 : memref<32x32xf16>) {
          ^bb0(%in: f32, %out: f16):
            %10 = arith.truncf %in : f32 to f16
            linalg.yield %10 : f16
          }
          %9 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 1024 + d1 * 65536 + d2 * 8192)>(%arg9, %arg7, %arg8)
          %reinterpret_cast_12 = memref.reinterpret_cast %arg3 to offset: [%9], sizes: [32, 32], strides: [32, 1] {loom.reuse = {spatial = [{depth = 1 : i64, iterator = "%arg8", mapped_to = "y", reuse_type = "no_reuse", volume = 0 : i64}, {depth = 2 : i64, iterator = "%arg9", mapped_to = "x", reuse_type = "no_reuse", volume = 0 : i64}], temporal = [{depth = 0 : i64, iterator = "%arg7", reuse_type = "no_reuse", volume = 0 : i64}]}} : memref<*xf16> to memref<32x32xf16, strided<[32, 1], offset: ?>>
          memref.copy %alloc_11, %reinterpret_cast_12 : memref<32x32xf16> to memref<32x32xf16, strided<[32, 1], offset: ?>>
        } {loom.mapped_to = "x"}
      } {loom.mapped_to = "y"}
    }
    return
  }
  func.func @flashattn_fwd__d0i0_d1i0__c0mem_c1mem_c2by(%arg0: memref<*xf16> {tt.divisibility = 16 : i32}, %arg1: memref<*xf16> {tt.divisibility = 16 : i32}, %arg2: memref<*xf16> {tt.divisibility = 16 : i32}, %arg3: memref<*xf16> {tt.divisibility = 16 : i32}, %arg4: index, %arg5: index, %arg6: index) {
    affine.for %arg7 = 0 to affine_map<(d0) -> ((d0 ceildiv 8) ceildiv 8)>(%arg4) {
      affine.parallel (%arg8) = (0) to (8) {
        affine.parallel (%arg9) = (0) to (8) {
          %cst = arith.constant 1.000000e+00 : f32
          %cst_0 = arith.constant 0.176776692 : f32
          %cst_1 = arith.constant 0.000000e+00 : f32
          %cst_2 = arith.constant 0xFF800000 : f32
          %alloc = memref.alloc() {alignment = 64 : i64} : memref<32xf32>
          %alloc_3 = memref.alloc() {alignment = 64 : i64} : memref<32xf32>
          linalg.fill ins(%cst : f32) outs(%alloc_3 : memref<32xf32>)
          %alloc_4 = memref.alloc() {alignment = 64 : i64} : memref<32x32xf32>
          %alloc_5 = memref.alloc() {alignment = 64 : i64} : memref<32x32xf32>
          linalg.fill ins(%cst_0 : f32) outs(%alloc_5 : memref<32x32xf32>)
          %alloc_6 = memref.alloc() {alignment = 64 : i64} : memref<32x32xf32>
          linalg.fill ins(%cst_1 : f32) outs(%alloc_6 : memref<32x32xf32>)
          %alloc_7 = memref.alloc() {alignment = 64 : i64} : memref<32xf32>
          linalg.fill ins(%cst_2 : f32) outs(%alloc_7 : memref<32xf32>)
          %7 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 1024 + d1 * 65536 + d2 * 8192)>(%arg9, %arg7, %arg8)
          %reinterpret_cast = memref.reinterpret_cast %arg0 to offset: [%7], sizes: [32, 32], strides: [32, 1] {loom.reuse = {spatial = [{depth = 1 : i64, iterator = "%arg8", mapped_to = "y", reuse_type = "no_reuse", volume = 0 : i64}, {depth = 2 : i64, iterator = "%arg9", mapped_to = "x", reuse_type = "no_reuse", volume = 0 : i64}], temporal = [{depth = 0 : i64, iterator = "%arg7", reuse_type = "no_reuse", volume = 0 : i64}]}} : memref<*xf16> to memref<32x32xf16, strided<[32, 1], offset: ?>>
          %alloc_8 = memref.alloc() {loom.alloc = {local = true, memory_name = "L1", size = 2048 : i64}} : memref<32x32xf16>
          memref.copy %reinterpret_cast, %alloc_8 {loom.copy.choice = {kind = "mem", memory_name = "L1"}} : memref<32x32xf16, strided<[32, 1], offset: ?>> to memref<32x32xf16>
          %c0 = arith.constant 0 : index
          %c512 = arith.constant 512 : index
          %c32 = arith.constant 32 : index
          %alloc_9 = memref.alloc() {alignment = 64 : i64} : memref<32xf32>
          memref.copy %alloc_7, %alloc_9 : memref<32xf32> to memref<32xf32>
          %alloc_10 = memref.alloc() {alignment = 64 : i64} : memref<32x32xf32>
          memref.copy %alloc_6, %alloc_10 : memref<32x32xf32> to memref<32x32xf32>
          %8:5 = scf.for %arg10 = %c0 to %c512 step %c32 iter_args(%arg11 = %c0, %arg12 = %c0, %arg13 = %alloc_9, %arg14 = %alloc_3, %arg15 = %alloc_10) -> (index, index, memref<32xf32>, memref<32xf32>, memref<32x32xf32>) {
            %10 = affine.apply affine_map<()[s0] -> (s0)>()[%arg12]
            %reinterpret_cast_13 = memref.reinterpret_cast %arg1 to offset: [%10], sizes: [32, 32], strides: [512, 1] {loom.reuse = {sequential = [{depth = 3 : i64, iterator = "%arg10", reuse_type = "total_reuse", volume = 2048 : i64}], spatial = [{depth = 1 : i64, iterator = "%arg8", mapped_to = "y", reuse_type = "total_reuse", volume = 2048 : i64}, {depth = 2 : i64, iterator = "%arg9", mapped_to = "x", reuse_type = "total_reuse", volume = 2048 : i64}], temporal = [{depth = 0 : i64, iterator = "%arg7", reuse_type = "total_reuse", volume = 2048 : i64}]}} : memref<*xf16> to memref<32x32xf16, strided<[512, 1], offset: ?>>
            %alloc_14 = memref.alloc() {loom.alloc = {local = true, memory_name = "L1", size = 2048 : i64}} : memref<32x32xf16>
            memref.copy %reinterpret_cast_13, %alloc_14 {loom.copy.choice = {kind = "mem", memory_name = "L1"}} : memref<32x32xf16, strided<[512, 1], offset: ?>> to memref<32x32xf16>
            %alloc_15 = memref.alloc() {alignment = 64 : i64} : memref<32x32xf32>
            memref.copy %alloc_6, %alloc_15 : memref<32x32xf32> to memref<32x32xf32>
            linalg.matmul ins(%alloc_8, %alloc_14 : memref<32x32xf16>, memref<32x32xf16>) outs(%alloc_15 : memref<32x32xf32>)
            linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%alloc_15, %alloc_5 : memref<32x32xf32>, memref<32x32xf32>) outs(%alloc_15 : memref<32x32xf32>) {
            ^bb0(%in: f32, %in_24: f32, %out: f32):
              %14 = arith.mulf %in, %in_24 : f32
              linalg.yield %14 : f32
            }
            linalg.transpose ins(%alloc_15 : memref<32x32xf32>) outs(%alloc_4 : memref<32x32xf32>) permutation = [1, 0] 
            %alloc_16 = memref.alloc() {alignment = 64 : i64} : memref<32xf32>
            memref.copy %alloc_7, %alloc_16 : memref<32xf32> to memref<32xf32>
            linalg.reduce ins(%alloc_4 : memref<32x32xf32>) outs(%alloc_16 : memref<32xf32>) dimensions = [0] 
              (%in: f32, %init: f32) {
                %14 = arith.maxnumf %in, %init : f32
                linalg.yield %14 : f32
              }
            %alloc_17 = memref.alloc() {alignment = 64 : i64} : memref<32xf32>
            linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%arg13, %alloc_16 : memref<32xf32>, memref<32xf32>) outs(%alloc_17 : memref<32xf32>) {
            ^bb0(%in: f32, %in_24: f32, %out: f32):
              %14 = arith.maxnumf %in, %in_24 : f32
              linalg.yield %14 : f32
            }
            %expand_shape_18 = memref.expand_shape %alloc_17 [[0, 1]] output_shape [32, 1] : memref<32xf32> into memref<32x1xf32>
            linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, 0)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%expand_shape_18 : memref<32x1xf32>) outs(%alloc_4 : memref<32x32xf32>) attrs =  {broadcastDims = array<i64: 1>} {
            ^bb0(%in: f32, %out: f32):
              linalg.yield %in : f32
            }
            linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%alloc_15, %alloc_4 : memref<32x32xf32>, memref<32x32xf32>) outs(%alloc_15 : memref<32x32xf32>) {
            ^bb0(%in: f32, %in_24: f32, %out: f32):
              %14 = arith.subf %in, %in_24 : f32
              linalg.yield %14 : f32
            }
            linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%alloc_15 : memref<32x32xf32>) outs(%alloc_15 : memref<32x32xf32>) {
            ^bb0(%in: f32, %out: f32):
              %14 = math.exp %in : f32
              linalg.yield %14 : f32
            }
            linalg.transpose ins(%alloc_15 : memref<32x32xf32>) outs(%alloc_4 : memref<32x32xf32>) permutation = [1, 0] 
            linalg.fill ins(%cst_1 : f32) outs(%alloc : memref<32xf32>)
            linalg.reduce ins(%alloc_4 : memref<32x32xf32>) outs(%alloc : memref<32xf32>) dimensions = [0] 
              (%in: f32, %init: f32) {
                %14 = arith.addf %in, %init : f32
                linalg.yield %14 : f32
              }
            linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%arg13, %alloc_17 : memref<32xf32>, memref<32xf32>) outs(%arg13 : memref<32xf32>) {
            ^bb0(%in: f32, %in_24: f32, %out: f32):
              %14 = arith.subf %in, %in_24 : f32
              linalg.yield %14 : f32
            }
            linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%arg13 : memref<32xf32>) outs(%arg13 : memref<32xf32>) {
            ^bb0(%in: f32, %out: f32):
              %14 = math.exp %in : f32
              linalg.yield %14 : f32
            }
            linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%arg14, %arg13 : memref<32xf32>, memref<32xf32>) outs(%arg14 : memref<32xf32>) {
            ^bb0(%in: f32, %in_24: f32, %out: f32):
              %14 = arith.mulf %in, %in_24 : f32
              linalg.yield %14 : f32
            }
            linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%arg14, %alloc : memref<32xf32>, memref<32xf32>) outs(%arg14 : memref<32xf32>) {
            ^bb0(%in: f32, %in_24: f32, %out: f32):
              %14 = arith.addf %in, %in_24 : f32
              linalg.yield %14 : f32
            }
            %11 = affine.apply affine_map<()[s0] -> (s0 * 32)>()[%arg11]
            %reinterpret_cast_19 = memref.reinterpret_cast %arg2 to offset: [%11], sizes: [32, 32], strides: [32, 1] {loom.reuse = {sequential = [{depth = 3 : i64, iterator = "%arg10", reuse_type = "total_reuse", volume = 2048 : i64}], spatial = [{depth = 1 : i64, iterator = "%arg8", mapped_to = "y", reuse_type = "total_reuse", volume = 2048 : i64}, {depth = 2 : i64, iterator = "%arg9", mapped_to = "x", reuse_type = "total_reuse", volume = 2048 : i64}], temporal = [{depth = 0 : i64, iterator = "%arg7", reuse_type = "total_reuse", volume = 2048 : i64}]}} : memref<*xf16> to memref<32x32xf16, strided<[32, 1], offset: ?>>
            %alloc_20 = memref.alloc() {loom.alloc = {local = true, memory_name = "L1", size = 2048 : i64}} : memref<32x32xf16>
            memref.copy %reinterpret_cast_19, %alloc_20 {loom.copy.choice = {dim = "y", interconnect_name = "vertical_links", kind = "broadcast"}} : memref<32x32xf16, strided<[32, 1], offset: ?>> to memref<32x32xf16>
            %alloc_21 = memref.alloc() {alignment = 64 : i64} : memref<32x32xf16>
            linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%alloc_15 : memref<32x32xf32>) outs(%alloc_21 : memref<32x32xf16>) {
            ^bb0(%in: f32, %out: f16):
              %14 = arith.truncf %in : f32 to f16
              linalg.yield %14 : f16
            }
            %expand_shape_22 = memref.expand_shape %arg13 [[0, 1]] output_shape [32, 1] : memref<32xf32> into memref<32x1xf32>
            linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, 0)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%expand_shape_22 : memref<32x1xf32>) outs(%alloc_4 : memref<32x32xf32>) attrs =  {broadcastDims = array<i64: 1>} {
            ^bb0(%in: f32, %out: f32):
              linalg.yield %in : f32
            }
            linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%arg15, %alloc_4 : memref<32x32xf32>, memref<32x32xf32>) outs(%arg15 : memref<32x32xf32>) {
            ^bb0(%in: f32, %in_24: f32, %out: f32):
              %14 = arith.mulf %in, %in_24 : f32
              linalg.yield %14 : f32
            }
            %alloc_23 = memref.alloc() {alignment = 64 : i64} : memref<32x32xf32>
            memref.copy %alloc_6, %alloc_23 : memref<32x32xf32> to memref<32x32xf32>
            linalg.matmul ins(%alloc_21, %alloc_20 : memref<32x32xf16>, memref<32x32xf16>) outs(%alloc_23 : memref<32x32xf32>)
            linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%arg15, %alloc_23 : memref<32x32xf32>, memref<32x32xf32>) outs(%arg15 : memref<32x32xf32>) {
            ^bb0(%in: f32, %in_24: f32, %out: f32):
              %14 = arith.addf %in, %in_24 : f32
              linalg.yield %14 : f32
            }
            %12 = arith.addi %arg11, %c32 : index
            %13 = arith.addi %arg12, %c32 : index
            scf.yield %12, %13, %alloc_17, %arg14, %arg15 : index, index, memref<32xf32>, memref<32xf32>, memref<32x32xf32>
          }
          %expand_shape = memref.expand_shape %8#3 [[0, 1]] output_shape [32, 1] : memref<32xf32> into memref<32x1xf32>
          linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, 0)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%expand_shape : memref<32x1xf32>) outs(%alloc_4 : memref<32x32xf32>) attrs =  {broadcastDims = array<i64: 1>} {
          ^bb0(%in: f32, %out: f32):
            linalg.yield %in : f32
          }
          linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%8#4, %alloc_4 : memref<32x32xf32>, memref<32x32xf32>) outs(%8#4 : memref<32x32xf32>) {
          ^bb0(%in: f32, %in_13: f32, %out: f32):
            %10 = arith.divf %in, %in_13 : f32
            linalg.yield %10 : f32
          }
          %alloc_11 = memref.alloc() {alignment = 64 : i64} : memref<32x32xf16>
          linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%8#4 : memref<32x32xf32>) outs(%alloc_11 : memref<32x32xf16>) {
          ^bb0(%in: f32, %out: f16):
            %10 = arith.truncf %in : f32 to f16
            linalg.yield %10 : f16
          }
          %9 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 1024 + d1 * 65536 + d2 * 8192)>(%arg9, %arg7, %arg8)
          %reinterpret_cast_12 = memref.reinterpret_cast %arg3 to offset: [%9], sizes: [32, 32], strides: [32, 1] {loom.reuse = {spatial = [{depth = 1 : i64, iterator = "%arg8", mapped_to = "y", reuse_type = "no_reuse", volume = 0 : i64}, {depth = 2 : i64, iterator = "%arg9", mapped_to = "x", reuse_type = "no_reuse", volume = 0 : i64}], temporal = [{depth = 0 : i64, iterator = "%arg7", reuse_type = "no_reuse", volume = 0 : i64}]}} : memref<*xf16> to memref<32x32xf16, strided<[32, 1], offset: ?>>
          memref.copy %alloc_11, %reinterpret_cast_12 : memref<32x32xf16> to memref<32x32xf16, strided<[32, 1], offset: ?>>
        } {loom.mapped_to = "x"}
      } {loom.mapped_to = "y"}
    }
    return
  }
  func.func @flashattn_fwd__d0i0_d1i0__c0mem_c1bx_c2by(%arg0: memref<*xf16> {tt.divisibility = 16 : i32}, %arg1: memref<*xf16> {tt.divisibility = 16 : i32}, %arg2: memref<*xf16> {tt.divisibility = 16 : i32}, %arg3: memref<*xf16> {tt.divisibility = 16 : i32}, %arg4: index, %arg5: index, %arg6: index) {
    affine.for %arg7 = 0 to affine_map<(d0) -> ((d0 ceildiv 8) ceildiv 8)>(%arg4) {
      affine.parallel (%arg8) = (0) to (8) {
        affine.parallel (%arg9) = (0) to (8) {
          %cst = arith.constant 1.000000e+00 : f32
          %cst_0 = arith.constant 0.176776692 : f32
          %cst_1 = arith.constant 0.000000e+00 : f32
          %cst_2 = arith.constant 0xFF800000 : f32
          %alloc = memref.alloc() {alignment = 64 : i64} : memref<32xf32>
          %alloc_3 = memref.alloc() {alignment = 64 : i64} : memref<32xf32>
          linalg.fill ins(%cst : f32) outs(%alloc_3 : memref<32xf32>)
          %alloc_4 = memref.alloc() {alignment = 64 : i64} : memref<32x32xf32>
          %alloc_5 = memref.alloc() {alignment = 64 : i64} : memref<32x32xf32>
          linalg.fill ins(%cst_0 : f32) outs(%alloc_5 : memref<32x32xf32>)
          %alloc_6 = memref.alloc() {alignment = 64 : i64} : memref<32x32xf32>
          linalg.fill ins(%cst_1 : f32) outs(%alloc_6 : memref<32x32xf32>)
          %alloc_7 = memref.alloc() {alignment = 64 : i64} : memref<32xf32>
          linalg.fill ins(%cst_2 : f32) outs(%alloc_7 : memref<32xf32>)
          %7 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 1024 + d1 * 65536 + d2 * 8192)>(%arg9, %arg7, %arg8)
          %reinterpret_cast = memref.reinterpret_cast %arg0 to offset: [%7], sizes: [32, 32], strides: [32, 1] {loom.reuse = {spatial = [{depth = 1 : i64, iterator = "%arg8", mapped_to = "y", reuse_type = "no_reuse", volume = 0 : i64}, {depth = 2 : i64, iterator = "%arg9", mapped_to = "x", reuse_type = "no_reuse", volume = 0 : i64}], temporal = [{depth = 0 : i64, iterator = "%arg7", reuse_type = "no_reuse", volume = 0 : i64}]}} : memref<*xf16> to memref<32x32xf16, strided<[32, 1], offset: ?>>
          %alloc_8 = memref.alloc() {loom.alloc = {local = true, memory_name = "L1", size = 2048 : i64}} : memref<32x32xf16>
          memref.copy %reinterpret_cast, %alloc_8 {loom.copy.choice = {kind = "mem", memory_name = "L1"}} : memref<32x32xf16, strided<[32, 1], offset: ?>> to memref<32x32xf16>
          %c0 = arith.constant 0 : index
          %c512 = arith.constant 512 : index
          %c32 = arith.constant 32 : index
          %alloc_9 = memref.alloc() {alignment = 64 : i64} : memref<32xf32>
          memref.copy %alloc_7, %alloc_9 : memref<32xf32> to memref<32xf32>
          %alloc_10 = memref.alloc() {alignment = 64 : i64} : memref<32x32xf32>
          memref.copy %alloc_6, %alloc_10 : memref<32x32xf32> to memref<32x32xf32>
          %8:5 = scf.for %arg10 = %c0 to %c512 step %c32 iter_args(%arg11 = %c0, %arg12 = %c0, %arg13 = %alloc_9, %arg14 = %alloc_3, %arg15 = %alloc_10) -> (index, index, memref<32xf32>, memref<32xf32>, memref<32x32xf32>) {
            %10 = affine.apply affine_map<()[s0] -> (s0)>()[%arg12]
            %reinterpret_cast_13 = memref.reinterpret_cast %arg1 to offset: [%10], sizes: [32, 32], strides: [512, 1] {loom.reuse = {sequential = [{depth = 3 : i64, iterator = "%arg10", reuse_type = "total_reuse", volume = 2048 : i64}], spatial = [{depth = 1 : i64, iterator = "%arg8", mapped_to = "y", reuse_type = "total_reuse", volume = 2048 : i64}, {depth = 2 : i64, iterator = "%arg9", mapped_to = "x", reuse_type = "total_reuse", volume = 2048 : i64}], temporal = [{depth = 0 : i64, iterator = "%arg7", reuse_type = "total_reuse", volume = 2048 : i64}]}} : memref<*xf16> to memref<32x32xf16, strided<[512, 1], offset: ?>>
            %alloc_14 = memref.alloc() {loom.alloc = {local = true, memory_name = "L1", size = 2048 : i64}} : memref<32x32xf16>
            memref.copy %reinterpret_cast_13, %alloc_14 {loom.copy.choice = {dim = "x", interconnect_name = "horizontal_links", kind = "broadcast"}} : memref<32x32xf16, strided<[512, 1], offset: ?>> to memref<32x32xf16>
            %alloc_15 = memref.alloc() {alignment = 64 : i64} : memref<32x32xf32>
            memref.copy %alloc_6, %alloc_15 : memref<32x32xf32> to memref<32x32xf32>
            linalg.matmul ins(%alloc_8, %alloc_14 : memref<32x32xf16>, memref<32x32xf16>) outs(%alloc_15 : memref<32x32xf32>)
            linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%alloc_15, %alloc_5 : memref<32x32xf32>, memref<32x32xf32>) outs(%alloc_15 : memref<32x32xf32>) {
            ^bb0(%in: f32, %in_24: f32, %out: f32):
              %14 = arith.mulf %in, %in_24 : f32
              linalg.yield %14 : f32
            }
            linalg.transpose ins(%alloc_15 : memref<32x32xf32>) outs(%alloc_4 : memref<32x32xf32>) permutation = [1, 0] 
            %alloc_16 = memref.alloc() {alignment = 64 : i64} : memref<32xf32>
            memref.copy %alloc_7, %alloc_16 : memref<32xf32> to memref<32xf32>
            linalg.reduce ins(%alloc_4 : memref<32x32xf32>) outs(%alloc_16 : memref<32xf32>) dimensions = [0] 
              (%in: f32, %init: f32) {
                %14 = arith.maxnumf %in, %init : f32
                linalg.yield %14 : f32
              }
            %alloc_17 = memref.alloc() {alignment = 64 : i64} : memref<32xf32>
            linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%arg13, %alloc_16 : memref<32xf32>, memref<32xf32>) outs(%alloc_17 : memref<32xf32>) {
            ^bb0(%in: f32, %in_24: f32, %out: f32):
              %14 = arith.maxnumf %in, %in_24 : f32
              linalg.yield %14 : f32
            }
            %expand_shape_18 = memref.expand_shape %alloc_17 [[0, 1]] output_shape [32, 1] : memref<32xf32> into memref<32x1xf32>
            linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, 0)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%expand_shape_18 : memref<32x1xf32>) outs(%alloc_4 : memref<32x32xf32>) attrs =  {broadcastDims = array<i64: 1>} {
            ^bb0(%in: f32, %out: f32):
              linalg.yield %in : f32
            }
            linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%alloc_15, %alloc_4 : memref<32x32xf32>, memref<32x32xf32>) outs(%alloc_15 : memref<32x32xf32>) {
            ^bb0(%in: f32, %in_24: f32, %out: f32):
              %14 = arith.subf %in, %in_24 : f32
              linalg.yield %14 : f32
            }
            linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%alloc_15 : memref<32x32xf32>) outs(%alloc_15 : memref<32x32xf32>) {
            ^bb0(%in: f32, %out: f32):
              %14 = math.exp %in : f32
              linalg.yield %14 : f32
            }
            linalg.transpose ins(%alloc_15 : memref<32x32xf32>) outs(%alloc_4 : memref<32x32xf32>) permutation = [1, 0] 
            linalg.fill ins(%cst_1 : f32) outs(%alloc : memref<32xf32>)
            linalg.reduce ins(%alloc_4 : memref<32x32xf32>) outs(%alloc : memref<32xf32>) dimensions = [0] 
              (%in: f32, %init: f32) {
                %14 = arith.addf %in, %init : f32
                linalg.yield %14 : f32
              }
            linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%arg13, %alloc_17 : memref<32xf32>, memref<32xf32>) outs(%arg13 : memref<32xf32>) {
            ^bb0(%in: f32, %in_24: f32, %out: f32):
              %14 = arith.subf %in, %in_24 : f32
              linalg.yield %14 : f32
            }
            linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%arg13 : memref<32xf32>) outs(%arg13 : memref<32xf32>) {
            ^bb0(%in: f32, %out: f32):
              %14 = math.exp %in : f32
              linalg.yield %14 : f32
            }
            linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%arg14, %arg13 : memref<32xf32>, memref<32xf32>) outs(%arg14 : memref<32xf32>) {
            ^bb0(%in: f32, %in_24: f32, %out: f32):
              %14 = arith.mulf %in, %in_24 : f32
              linalg.yield %14 : f32
            }
            linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%arg14, %alloc : memref<32xf32>, memref<32xf32>) outs(%arg14 : memref<32xf32>) {
            ^bb0(%in: f32, %in_24: f32, %out: f32):
              %14 = arith.addf %in, %in_24 : f32
              linalg.yield %14 : f32
            }
            %11 = affine.apply affine_map<()[s0] -> (s0 * 32)>()[%arg11]
            %reinterpret_cast_19 = memref.reinterpret_cast %arg2 to offset: [%11], sizes: [32, 32], strides: [32, 1] {loom.reuse = {sequential = [{depth = 3 : i64, iterator = "%arg10", reuse_type = "total_reuse", volume = 2048 : i64}], spatial = [{depth = 1 : i64, iterator = "%arg8", mapped_to = "y", reuse_type = "total_reuse", volume = 2048 : i64}, {depth = 2 : i64, iterator = "%arg9", mapped_to = "x", reuse_type = "total_reuse", volume = 2048 : i64}], temporal = [{depth = 0 : i64, iterator = "%arg7", reuse_type = "total_reuse", volume = 2048 : i64}]}} : memref<*xf16> to memref<32x32xf16, strided<[32, 1], offset: ?>>
            %alloc_20 = memref.alloc() {loom.alloc = {local = true, memory_name = "L1", size = 2048 : i64}} : memref<32x32xf16>
            memref.copy %reinterpret_cast_19, %alloc_20 {loom.copy.choice = {dim = "y", interconnect_name = "vertical_links", kind = "broadcast"}} : memref<32x32xf16, strided<[32, 1], offset: ?>> to memref<32x32xf16>
            %alloc_21 = memref.alloc() {alignment = 64 : i64} : memref<32x32xf16>
            linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%alloc_15 : memref<32x32xf32>) outs(%alloc_21 : memref<32x32xf16>) {
            ^bb0(%in: f32, %out: f16):
              %14 = arith.truncf %in : f32 to f16
              linalg.yield %14 : f16
            }
            %expand_shape_22 = memref.expand_shape %arg13 [[0, 1]] output_shape [32, 1] : memref<32xf32> into memref<32x1xf32>
            linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, 0)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%expand_shape_22 : memref<32x1xf32>) outs(%alloc_4 : memref<32x32xf32>) attrs =  {broadcastDims = array<i64: 1>} {
            ^bb0(%in: f32, %out: f32):
              linalg.yield %in : f32
            }
            linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%arg15, %alloc_4 : memref<32x32xf32>, memref<32x32xf32>) outs(%arg15 : memref<32x32xf32>) {
            ^bb0(%in: f32, %in_24: f32, %out: f32):
              %14 = arith.mulf %in, %in_24 : f32
              linalg.yield %14 : f32
            }
            %alloc_23 = memref.alloc() {alignment = 64 : i64} : memref<32x32xf32>
            memref.copy %alloc_6, %alloc_23 : memref<32x32xf32> to memref<32x32xf32>
            linalg.matmul ins(%alloc_21, %alloc_20 : memref<32x32xf16>, memref<32x32xf16>) outs(%alloc_23 : memref<32x32xf32>)
            linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%arg15, %alloc_23 : memref<32x32xf32>, memref<32x32xf32>) outs(%arg15 : memref<32x32xf32>) {
            ^bb0(%in: f32, %in_24: f32, %out: f32):
              %14 = arith.addf %in, %in_24 : f32
              linalg.yield %14 : f32
            }
            %12 = arith.addi %arg11, %c32 : index
            %13 = arith.addi %arg12, %c32 : index
            scf.yield %12, %13, %alloc_17, %arg14, %arg15 : index, index, memref<32xf32>, memref<32xf32>, memref<32x32xf32>
          }
          %expand_shape = memref.expand_shape %8#3 [[0, 1]] output_shape [32, 1] : memref<32xf32> into memref<32x1xf32>
          linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, 0)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%expand_shape : memref<32x1xf32>) outs(%alloc_4 : memref<32x32xf32>) attrs =  {broadcastDims = array<i64: 1>} {
          ^bb0(%in: f32, %out: f32):
            linalg.yield %in : f32
          }
          linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%8#4, %alloc_4 : memref<32x32xf32>, memref<32x32xf32>) outs(%8#4 : memref<32x32xf32>) {
          ^bb0(%in: f32, %in_13: f32, %out: f32):
            %10 = arith.divf %in, %in_13 : f32
            linalg.yield %10 : f32
          }
          %alloc_11 = memref.alloc() {alignment = 64 : i64} : memref<32x32xf16>
          linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%8#4 : memref<32x32xf32>) outs(%alloc_11 : memref<32x32xf16>) {
          ^bb0(%in: f32, %out: f16):
            %10 = arith.truncf %in : f32 to f16
            linalg.yield %10 : f16
          }
          %9 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 1024 + d1 * 65536 + d2 * 8192)>(%arg9, %arg7, %arg8)
          %reinterpret_cast_12 = memref.reinterpret_cast %arg3 to offset: [%9], sizes: [32, 32], strides: [32, 1] {loom.reuse = {spatial = [{depth = 1 : i64, iterator = "%arg8", mapped_to = "y", reuse_type = "no_reuse", volume = 0 : i64}, {depth = 2 : i64, iterator = "%arg9", mapped_to = "x", reuse_type = "no_reuse", volume = 0 : i64}], temporal = [{depth = 0 : i64, iterator = "%arg7", reuse_type = "no_reuse", volume = 0 : i64}]}} : memref<*xf16> to memref<32x32xf16, strided<[32, 1], offset: ?>>
          memref.copy %alloc_11, %reinterpret_cast_12 : memref<32x32xf16> to memref<32x32xf16, strided<[32, 1], offset: ?>>
        } {loom.mapped_to = "x"}
      } {loom.mapped_to = "y"}
    }
    return
  }
  func.func @flashattn_fwd__d0i0_d1i0__c0mem_c1by_c2by(%arg0: memref<*xf16> {tt.divisibility = 16 : i32}, %arg1: memref<*xf16> {tt.divisibility = 16 : i32}, %arg2: memref<*xf16> {tt.divisibility = 16 : i32}, %arg3: memref<*xf16> {tt.divisibility = 16 : i32}, %arg4: index, %arg5: index, %arg6: index) {
    affine.for %arg7 = 0 to affine_map<(d0) -> ((d0 ceildiv 8) ceildiv 8)>(%arg4) {
      affine.parallel (%arg8) = (0) to (8) {
        affine.parallel (%arg9) = (0) to (8) {
          %cst = arith.constant 1.000000e+00 : f32
          %cst_0 = arith.constant 0.176776692 : f32
          %cst_1 = arith.constant 0.000000e+00 : f32
          %cst_2 = arith.constant 0xFF800000 : f32
          %alloc = memref.alloc() {alignment = 64 : i64} : memref<32xf32>
          %alloc_3 = memref.alloc() {alignment = 64 : i64} : memref<32xf32>
          linalg.fill ins(%cst : f32) outs(%alloc_3 : memref<32xf32>)
          %alloc_4 = memref.alloc() {alignment = 64 : i64} : memref<32x32xf32>
          %alloc_5 = memref.alloc() {alignment = 64 : i64} : memref<32x32xf32>
          linalg.fill ins(%cst_0 : f32) outs(%alloc_5 : memref<32x32xf32>)
          %alloc_6 = memref.alloc() {alignment = 64 : i64} : memref<32x32xf32>
          linalg.fill ins(%cst_1 : f32) outs(%alloc_6 : memref<32x32xf32>)
          %alloc_7 = memref.alloc() {alignment = 64 : i64} : memref<32xf32>
          linalg.fill ins(%cst_2 : f32) outs(%alloc_7 : memref<32xf32>)
          %7 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 1024 + d1 * 65536 + d2 * 8192)>(%arg9, %arg7, %arg8)
          %reinterpret_cast = memref.reinterpret_cast %arg0 to offset: [%7], sizes: [32, 32], strides: [32, 1] {loom.reuse = {spatial = [{depth = 1 : i64, iterator = "%arg8", mapped_to = "y", reuse_type = "no_reuse", volume = 0 : i64}, {depth = 2 : i64, iterator = "%arg9", mapped_to = "x", reuse_type = "no_reuse", volume = 0 : i64}], temporal = [{depth = 0 : i64, iterator = "%arg7", reuse_type = "no_reuse", volume = 0 : i64}]}} : memref<*xf16> to memref<32x32xf16, strided<[32, 1], offset: ?>>
          %alloc_8 = memref.alloc() {loom.alloc = {local = true, memory_name = "L1", size = 2048 : i64}} : memref<32x32xf16>
          memref.copy %reinterpret_cast, %alloc_8 {loom.copy.choice = {kind = "mem", memory_name = "L1"}} : memref<32x32xf16, strided<[32, 1], offset: ?>> to memref<32x32xf16>
          %c0 = arith.constant 0 : index
          %c512 = arith.constant 512 : index
          %c32 = arith.constant 32 : index
          %alloc_9 = memref.alloc() {alignment = 64 : i64} : memref<32xf32>
          memref.copy %alloc_7, %alloc_9 : memref<32xf32> to memref<32xf32>
          %alloc_10 = memref.alloc() {alignment = 64 : i64} : memref<32x32xf32>
          memref.copy %alloc_6, %alloc_10 : memref<32x32xf32> to memref<32x32xf32>
          %8:5 = scf.for %arg10 = %c0 to %c512 step %c32 iter_args(%arg11 = %c0, %arg12 = %c0, %arg13 = %alloc_9, %arg14 = %alloc_3, %arg15 = %alloc_10) -> (index, index, memref<32xf32>, memref<32xf32>, memref<32x32xf32>) {
            %10 = affine.apply affine_map<()[s0] -> (s0)>()[%arg12]
            %reinterpret_cast_13 = memref.reinterpret_cast %arg1 to offset: [%10], sizes: [32, 32], strides: [512, 1] {loom.reuse = {sequential = [{depth = 3 : i64, iterator = "%arg10", reuse_type = "total_reuse", volume = 2048 : i64}], spatial = [{depth = 1 : i64, iterator = "%arg8", mapped_to = "y", reuse_type = "total_reuse", volume = 2048 : i64}, {depth = 2 : i64, iterator = "%arg9", mapped_to = "x", reuse_type = "total_reuse", volume = 2048 : i64}], temporal = [{depth = 0 : i64, iterator = "%arg7", reuse_type = "total_reuse", volume = 2048 : i64}]}} : memref<*xf16> to memref<32x32xf16, strided<[512, 1], offset: ?>>
            %alloc_14 = memref.alloc() {loom.alloc = {local = true, memory_name = "L1", size = 2048 : i64}} : memref<32x32xf16>
            memref.copy %reinterpret_cast_13, %alloc_14 {loom.copy.choice = {dim = "y", interconnect_name = "vertical_links", kind = "broadcast"}} : memref<32x32xf16, strided<[512, 1], offset: ?>> to memref<32x32xf16>
            %alloc_15 = memref.alloc() {alignment = 64 : i64} : memref<32x32xf32>
            memref.copy %alloc_6, %alloc_15 : memref<32x32xf32> to memref<32x32xf32>
            linalg.matmul ins(%alloc_8, %alloc_14 : memref<32x32xf16>, memref<32x32xf16>) outs(%alloc_15 : memref<32x32xf32>)
            linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%alloc_15, %alloc_5 : memref<32x32xf32>, memref<32x32xf32>) outs(%alloc_15 : memref<32x32xf32>) {
            ^bb0(%in: f32, %in_24: f32, %out: f32):
              %14 = arith.mulf %in, %in_24 : f32
              linalg.yield %14 : f32
            }
            linalg.transpose ins(%alloc_15 : memref<32x32xf32>) outs(%alloc_4 : memref<32x32xf32>) permutation = [1, 0] 
            %alloc_16 = memref.alloc() {alignment = 64 : i64} : memref<32xf32>
            memref.copy %alloc_7, %alloc_16 : memref<32xf32> to memref<32xf32>
            linalg.reduce ins(%alloc_4 : memref<32x32xf32>) outs(%alloc_16 : memref<32xf32>) dimensions = [0] 
              (%in: f32, %init: f32) {
                %14 = arith.maxnumf %in, %init : f32
                linalg.yield %14 : f32
              }
            %alloc_17 = memref.alloc() {alignment = 64 : i64} : memref<32xf32>
            linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%arg13, %alloc_16 : memref<32xf32>, memref<32xf32>) outs(%alloc_17 : memref<32xf32>) {
            ^bb0(%in: f32, %in_24: f32, %out: f32):
              %14 = arith.maxnumf %in, %in_24 : f32
              linalg.yield %14 : f32
            }
            %expand_shape_18 = memref.expand_shape %alloc_17 [[0, 1]] output_shape [32, 1] : memref<32xf32> into memref<32x1xf32>
            linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, 0)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%expand_shape_18 : memref<32x1xf32>) outs(%alloc_4 : memref<32x32xf32>) attrs =  {broadcastDims = array<i64: 1>} {
            ^bb0(%in: f32, %out: f32):
              linalg.yield %in : f32
            }
            linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%alloc_15, %alloc_4 : memref<32x32xf32>, memref<32x32xf32>) outs(%alloc_15 : memref<32x32xf32>) {
            ^bb0(%in: f32, %in_24: f32, %out: f32):
              %14 = arith.subf %in, %in_24 : f32
              linalg.yield %14 : f32
            }
            linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%alloc_15 : memref<32x32xf32>) outs(%alloc_15 : memref<32x32xf32>) {
            ^bb0(%in: f32, %out: f32):
              %14 = math.exp %in : f32
              linalg.yield %14 : f32
            }
            linalg.transpose ins(%alloc_15 : memref<32x32xf32>) outs(%alloc_4 : memref<32x32xf32>) permutation = [1, 0] 
            linalg.fill ins(%cst_1 : f32) outs(%alloc : memref<32xf32>)
            linalg.reduce ins(%alloc_4 : memref<32x32xf32>) outs(%alloc : memref<32xf32>) dimensions = [0] 
              (%in: f32, %init: f32) {
                %14 = arith.addf %in, %init : f32
                linalg.yield %14 : f32
              }
            linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%arg13, %alloc_17 : memref<32xf32>, memref<32xf32>) outs(%arg13 : memref<32xf32>) {
            ^bb0(%in: f32, %in_24: f32, %out: f32):
              %14 = arith.subf %in, %in_24 : f32
              linalg.yield %14 : f32
            }
            linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%arg13 : memref<32xf32>) outs(%arg13 : memref<32xf32>) {
            ^bb0(%in: f32, %out: f32):
              %14 = math.exp %in : f32
              linalg.yield %14 : f32
            }
            linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%arg14, %arg13 : memref<32xf32>, memref<32xf32>) outs(%arg14 : memref<32xf32>) {
            ^bb0(%in: f32, %in_24: f32, %out: f32):
              %14 = arith.mulf %in, %in_24 : f32
              linalg.yield %14 : f32
            }
            linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%arg14, %alloc : memref<32xf32>, memref<32xf32>) outs(%arg14 : memref<32xf32>) {
            ^bb0(%in: f32, %in_24: f32, %out: f32):
              %14 = arith.addf %in, %in_24 : f32
              linalg.yield %14 : f32
            }
            %11 = affine.apply affine_map<()[s0] -> (s0 * 32)>()[%arg11]
            %reinterpret_cast_19 = memref.reinterpret_cast %arg2 to offset: [%11], sizes: [32, 32], strides: [32, 1] {loom.reuse = {sequential = [{depth = 3 : i64, iterator = "%arg10", reuse_type = "total_reuse", volume = 2048 : i64}], spatial = [{depth = 1 : i64, iterator = "%arg8", mapped_to = "y", reuse_type = "total_reuse", volume = 2048 : i64}, {depth = 2 : i64, iterator = "%arg9", mapped_to = "x", reuse_type = "total_reuse", volume = 2048 : i64}], temporal = [{depth = 0 : i64, iterator = "%arg7", reuse_type = "total_reuse", volume = 2048 : i64}]}} : memref<*xf16> to memref<32x32xf16, strided<[32, 1], offset: ?>>
            %alloc_20 = memref.alloc() {loom.alloc = {local = true, memory_name = "L1", size = 2048 : i64}} : memref<32x32xf16>
            memref.copy %reinterpret_cast_19, %alloc_20 {loom.copy.choice = {dim = "y", interconnect_name = "vertical_links", kind = "broadcast"}} : memref<32x32xf16, strided<[32, 1], offset: ?>> to memref<32x32xf16>
            %alloc_21 = memref.alloc() {alignment = 64 : i64} : memref<32x32xf16>
            linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%alloc_15 : memref<32x32xf32>) outs(%alloc_21 : memref<32x32xf16>) {
            ^bb0(%in: f32, %out: f16):
              %14 = arith.truncf %in : f32 to f16
              linalg.yield %14 : f16
            }
            %expand_shape_22 = memref.expand_shape %arg13 [[0, 1]] output_shape [32, 1] : memref<32xf32> into memref<32x1xf32>
            linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, 0)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%expand_shape_22 : memref<32x1xf32>) outs(%alloc_4 : memref<32x32xf32>) attrs =  {broadcastDims = array<i64: 1>} {
            ^bb0(%in: f32, %out: f32):
              linalg.yield %in : f32
            }
            linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%arg15, %alloc_4 : memref<32x32xf32>, memref<32x32xf32>) outs(%arg15 : memref<32x32xf32>) {
            ^bb0(%in: f32, %in_24: f32, %out: f32):
              %14 = arith.mulf %in, %in_24 : f32
              linalg.yield %14 : f32
            }
            %alloc_23 = memref.alloc() {alignment = 64 : i64} : memref<32x32xf32>
            memref.copy %alloc_6, %alloc_23 : memref<32x32xf32> to memref<32x32xf32>
            linalg.matmul ins(%alloc_21, %alloc_20 : memref<32x32xf16>, memref<32x32xf16>) outs(%alloc_23 : memref<32x32xf32>)
            linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%arg15, %alloc_23 : memref<32x32xf32>, memref<32x32xf32>) outs(%arg15 : memref<32x32xf32>) {
            ^bb0(%in: f32, %in_24: f32, %out: f32):
              %14 = arith.addf %in, %in_24 : f32
              linalg.yield %14 : f32
            }
            %12 = arith.addi %arg11, %c32 : index
            %13 = arith.addi %arg12, %c32 : index
            scf.yield %12, %13, %alloc_17, %arg14, %arg15 : index, index, memref<32xf32>, memref<32xf32>, memref<32x32xf32>
          }
          %expand_shape = memref.expand_shape %8#3 [[0, 1]] output_shape [32, 1] : memref<32xf32> into memref<32x1xf32>
          linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, 0)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%expand_shape : memref<32x1xf32>) outs(%alloc_4 : memref<32x32xf32>) attrs =  {broadcastDims = array<i64: 1>} {
          ^bb0(%in: f32, %out: f32):
            linalg.yield %in : f32
          }
          linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%8#4, %alloc_4 : memref<32x32xf32>, memref<32x32xf32>) outs(%8#4 : memref<32x32xf32>) {
          ^bb0(%in: f32, %in_13: f32, %out: f32):
            %10 = arith.divf %in, %in_13 : f32
            linalg.yield %10 : f32
          }
          %alloc_11 = memref.alloc() {alignment = 64 : i64} : memref<32x32xf16>
          linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%8#4 : memref<32x32xf32>) outs(%alloc_11 : memref<32x32xf16>) {
          ^bb0(%in: f32, %out: f16):
            %10 = arith.truncf %in : f32 to f16
            linalg.yield %10 : f16
          }
          %9 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 1024 + d1 * 65536 + d2 * 8192)>(%arg9, %arg7, %arg8)
          %reinterpret_cast_12 = memref.reinterpret_cast %arg3 to offset: [%9], sizes: [32, 32], strides: [32, 1] {loom.reuse = {spatial = [{depth = 1 : i64, iterator = "%arg8", mapped_to = "y", reuse_type = "no_reuse", volume = 0 : i64}, {depth = 2 : i64, iterator = "%arg9", mapped_to = "x", reuse_type = "no_reuse", volume = 0 : i64}], temporal = [{depth = 0 : i64, iterator = "%arg7", reuse_type = "no_reuse", volume = 0 : i64}]}} : memref<*xf16> to memref<32x32xf16, strided<[32, 1], offset: ?>>
          memref.copy %alloc_11, %reinterpret_cast_12 : memref<32x32xf16> to memref<32x32xf16, strided<[32, 1], offset: ?>>
        } {loom.mapped_to = "x"}
      } {loom.mapped_to = "y"}
    }
    return
  }
  func.func @flashattn_fwd__d1i0_d0i0__c0mem_c1mem_c2mem(%arg0: memref<*xf16> {tt.divisibility = 16 : i32}, %arg1: memref<*xf16> {tt.divisibility = 16 : i32}, %arg2: memref<*xf16> {tt.divisibility = 16 : i32}, %arg3: memref<*xf16> {tt.divisibility = 16 : i32}, %arg4: index, %arg5: index, %arg6: index) {
    affine.for %arg7 = 0 to affine_map<(d0) -> ((d0 ceildiv 8) ceildiv 8)>(%arg4) {
      affine.parallel (%arg8) = (0) to (8) {
        affine.parallel (%arg9) = (0) to (8) {
          %cst = arith.constant 1.000000e+00 : f32
          %cst_0 = arith.constant 0.176776692 : f32
          %cst_1 = arith.constant 0.000000e+00 : f32
          %cst_2 = arith.constant 0xFF800000 : f32
          %alloc = memref.alloc() {alignment = 64 : i64} : memref<32xf32>
          %alloc_3 = memref.alloc() {alignment = 64 : i64} : memref<32xf32>
          linalg.fill ins(%cst : f32) outs(%alloc_3 : memref<32xf32>)
          %alloc_4 = memref.alloc() {alignment = 64 : i64} : memref<32x32xf32>
          %alloc_5 = memref.alloc() {alignment = 64 : i64} : memref<32x32xf32>
          linalg.fill ins(%cst_0 : f32) outs(%alloc_5 : memref<32x32xf32>)
          %alloc_6 = memref.alloc() {alignment = 64 : i64} : memref<32x32xf32>
          linalg.fill ins(%cst_1 : f32) outs(%alloc_6 : memref<32x32xf32>)
          %alloc_7 = memref.alloc() {alignment = 64 : i64} : memref<32xf32>
          linalg.fill ins(%cst_2 : f32) outs(%alloc_7 : memref<32xf32>)
          %7 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 1024 + d1 * 65536 + d2 * 8192)>(%arg9, %arg7, %arg8)
          %reinterpret_cast = memref.reinterpret_cast %arg0 to offset: [%7], sizes: [32, 32], strides: [32, 1] {loom.reuse = {spatial = [{depth = 1 : i64, iterator = "%arg8", mapped_to = "x", reuse_type = "no_reuse", volume = 0 : i64}, {depth = 2 : i64, iterator = "%arg9", mapped_to = "y", reuse_type = "no_reuse", volume = 0 : i64}], temporal = [{depth = 0 : i64, iterator = "%arg7", reuse_type = "no_reuse", volume = 0 : i64}]}} : memref<*xf16> to memref<32x32xf16, strided<[32, 1], offset: ?>>
          %alloc_8 = memref.alloc() {loom.alloc = {local = true, memory_name = "L1", size = 2048 : i64}} : memref<32x32xf16>
          memref.copy %reinterpret_cast, %alloc_8 {loom.copy.choice = {kind = "mem", memory_name = "L1"}} : memref<32x32xf16, strided<[32, 1], offset: ?>> to memref<32x32xf16>
          %c0 = arith.constant 0 : index
          %c512 = arith.constant 512 : index
          %c32 = arith.constant 32 : index
          %alloc_9 = memref.alloc() {alignment = 64 : i64} : memref<32xf32>
          memref.copy %alloc_7, %alloc_9 : memref<32xf32> to memref<32xf32>
          %alloc_10 = memref.alloc() {alignment = 64 : i64} : memref<32x32xf32>
          memref.copy %alloc_6, %alloc_10 : memref<32x32xf32> to memref<32x32xf32>
          %8:5 = scf.for %arg10 = %c0 to %c512 step %c32 iter_args(%arg11 = %c0, %arg12 = %c0, %arg13 = %alloc_9, %arg14 = %alloc_3, %arg15 = %alloc_10) -> (index, index, memref<32xf32>, memref<32xf32>, memref<32x32xf32>) {
            %10 = affine.apply affine_map<()[s0] -> (s0)>()[%arg12]
            %reinterpret_cast_13 = memref.reinterpret_cast %arg1 to offset: [%10], sizes: [32, 32], strides: [512, 1] {loom.reuse = {sequential = [{depth = 3 : i64, iterator = "%arg10", reuse_type = "total_reuse", volume = 2048 : i64}], spatial = [{depth = 1 : i64, iterator = "%arg8", mapped_to = "x", reuse_type = "total_reuse", volume = 2048 : i64}, {depth = 2 : i64, iterator = "%arg9", mapped_to = "y", reuse_type = "total_reuse", volume = 2048 : i64}], temporal = [{depth = 0 : i64, iterator = "%arg7", reuse_type = "total_reuse", volume = 2048 : i64}]}} : memref<*xf16> to memref<32x32xf16, strided<[512, 1], offset: ?>>
            %alloc_14 = memref.alloc() {loom.alloc = {local = true, memory_name = "L1", size = 2048 : i64}} : memref<32x32xf16>
            memref.copy %reinterpret_cast_13, %alloc_14 {loom.copy.choice = {kind = "mem", memory_name = "L1"}} : memref<32x32xf16, strided<[512, 1], offset: ?>> to memref<32x32xf16>
            %alloc_15 = memref.alloc() {alignment = 64 : i64} : memref<32x32xf32>
            memref.copy %alloc_6, %alloc_15 : memref<32x32xf32> to memref<32x32xf32>
            linalg.matmul ins(%alloc_8, %alloc_14 : memref<32x32xf16>, memref<32x32xf16>) outs(%alloc_15 : memref<32x32xf32>)
            linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%alloc_15, %alloc_5 : memref<32x32xf32>, memref<32x32xf32>) outs(%alloc_15 : memref<32x32xf32>) {
            ^bb0(%in: f32, %in_24: f32, %out: f32):
              %14 = arith.mulf %in, %in_24 : f32
              linalg.yield %14 : f32
            }
            linalg.transpose ins(%alloc_15 : memref<32x32xf32>) outs(%alloc_4 : memref<32x32xf32>) permutation = [1, 0] 
            %alloc_16 = memref.alloc() {alignment = 64 : i64} : memref<32xf32>
            memref.copy %alloc_7, %alloc_16 : memref<32xf32> to memref<32xf32>
            linalg.reduce ins(%alloc_4 : memref<32x32xf32>) outs(%alloc_16 : memref<32xf32>) dimensions = [0] 
              (%in: f32, %init: f32) {
                %14 = arith.maxnumf %in, %init : f32
                linalg.yield %14 : f32
              }
            %alloc_17 = memref.alloc() {alignment = 64 : i64} : memref<32xf32>
            linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%arg13, %alloc_16 : memref<32xf32>, memref<32xf32>) outs(%alloc_17 : memref<32xf32>) {
            ^bb0(%in: f32, %in_24: f32, %out: f32):
              %14 = arith.maxnumf %in, %in_24 : f32
              linalg.yield %14 : f32
            }
            %expand_shape_18 = memref.expand_shape %alloc_17 [[0, 1]] output_shape [32, 1] : memref<32xf32> into memref<32x1xf32>
            linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, 0)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%expand_shape_18 : memref<32x1xf32>) outs(%alloc_4 : memref<32x32xf32>) attrs =  {broadcastDims = array<i64: 1>} {
            ^bb0(%in: f32, %out: f32):
              linalg.yield %in : f32
            }
            linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%alloc_15, %alloc_4 : memref<32x32xf32>, memref<32x32xf32>) outs(%alloc_15 : memref<32x32xf32>) {
            ^bb0(%in: f32, %in_24: f32, %out: f32):
              %14 = arith.subf %in, %in_24 : f32
              linalg.yield %14 : f32
            }
            linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%alloc_15 : memref<32x32xf32>) outs(%alloc_15 : memref<32x32xf32>) {
            ^bb0(%in: f32, %out: f32):
              %14 = math.exp %in : f32
              linalg.yield %14 : f32
            }
            linalg.transpose ins(%alloc_15 : memref<32x32xf32>) outs(%alloc_4 : memref<32x32xf32>) permutation = [1, 0] 
            linalg.fill ins(%cst_1 : f32) outs(%alloc : memref<32xf32>)
            linalg.reduce ins(%alloc_4 : memref<32x32xf32>) outs(%alloc : memref<32xf32>) dimensions = [0] 
              (%in: f32, %init: f32) {
                %14 = arith.addf %in, %init : f32
                linalg.yield %14 : f32
              }
            linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%arg13, %alloc_17 : memref<32xf32>, memref<32xf32>) outs(%arg13 : memref<32xf32>) {
            ^bb0(%in: f32, %in_24: f32, %out: f32):
              %14 = arith.subf %in, %in_24 : f32
              linalg.yield %14 : f32
            }
            linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%arg13 : memref<32xf32>) outs(%arg13 : memref<32xf32>) {
            ^bb0(%in: f32, %out: f32):
              %14 = math.exp %in : f32
              linalg.yield %14 : f32
            }
            linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%arg14, %arg13 : memref<32xf32>, memref<32xf32>) outs(%arg14 : memref<32xf32>) {
            ^bb0(%in: f32, %in_24: f32, %out: f32):
              %14 = arith.mulf %in, %in_24 : f32
              linalg.yield %14 : f32
            }
            linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%arg14, %alloc : memref<32xf32>, memref<32xf32>) outs(%arg14 : memref<32xf32>) {
            ^bb0(%in: f32, %in_24: f32, %out: f32):
              %14 = arith.addf %in, %in_24 : f32
              linalg.yield %14 : f32
            }
            %11 = affine.apply affine_map<()[s0] -> (s0 * 32)>()[%arg11]
            %reinterpret_cast_19 = memref.reinterpret_cast %arg2 to offset: [%11], sizes: [32, 32], strides: [32, 1] {loom.reuse = {sequential = [{depth = 3 : i64, iterator = "%arg10", reuse_type = "total_reuse", volume = 2048 : i64}], spatial = [{depth = 1 : i64, iterator = "%arg8", mapped_to = "x", reuse_type = "total_reuse", volume = 2048 : i64}, {depth = 2 : i64, iterator = "%arg9", mapped_to = "y", reuse_type = "total_reuse", volume = 2048 : i64}], temporal = [{depth = 0 : i64, iterator = "%arg7", reuse_type = "total_reuse", volume = 2048 : i64}]}} : memref<*xf16> to memref<32x32xf16, strided<[32, 1], offset: ?>>
            %alloc_20 = memref.alloc() {loom.alloc = {local = true, memory_name = "L1", size = 2048 : i64}} : memref<32x32xf16>
            memref.copy %reinterpret_cast_19, %alloc_20 {loom.copy.choice = {kind = "mem", memory_name = "L1"}} : memref<32x32xf16, strided<[32, 1], offset: ?>> to memref<32x32xf16>
            %alloc_21 = memref.alloc() {alignment = 64 : i64} : memref<32x32xf16>
            linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%alloc_15 : memref<32x32xf32>) outs(%alloc_21 : memref<32x32xf16>) {
            ^bb0(%in: f32, %out: f16):
              %14 = arith.truncf %in : f32 to f16
              linalg.yield %14 : f16
            }
            %expand_shape_22 = memref.expand_shape %arg13 [[0, 1]] output_shape [32, 1] : memref<32xf32> into memref<32x1xf32>
            linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, 0)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%expand_shape_22 : memref<32x1xf32>) outs(%alloc_4 : memref<32x32xf32>) attrs =  {broadcastDims = array<i64: 1>} {
            ^bb0(%in: f32, %out: f32):
              linalg.yield %in : f32
            }
            linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%arg15, %alloc_4 : memref<32x32xf32>, memref<32x32xf32>) outs(%arg15 : memref<32x32xf32>) {
            ^bb0(%in: f32, %in_24: f32, %out: f32):
              %14 = arith.mulf %in, %in_24 : f32
              linalg.yield %14 : f32
            }
            %alloc_23 = memref.alloc() {alignment = 64 : i64} : memref<32x32xf32>
            memref.copy %alloc_6, %alloc_23 : memref<32x32xf32> to memref<32x32xf32>
            linalg.matmul ins(%alloc_21, %alloc_20 : memref<32x32xf16>, memref<32x32xf16>) outs(%alloc_23 : memref<32x32xf32>)
            linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%arg15, %alloc_23 : memref<32x32xf32>, memref<32x32xf32>) outs(%arg15 : memref<32x32xf32>) {
            ^bb0(%in: f32, %in_24: f32, %out: f32):
              %14 = arith.addf %in, %in_24 : f32
              linalg.yield %14 : f32
            }
            %12 = arith.addi %arg11, %c32 : index
            %13 = arith.addi %arg12, %c32 : index
            scf.yield %12, %13, %alloc_17, %arg14, %arg15 : index, index, memref<32xf32>, memref<32xf32>, memref<32x32xf32>
          }
          %expand_shape = memref.expand_shape %8#3 [[0, 1]] output_shape [32, 1] : memref<32xf32> into memref<32x1xf32>
          linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, 0)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%expand_shape : memref<32x1xf32>) outs(%alloc_4 : memref<32x32xf32>) attrs =  {broadcastDims = array<i64: 1>} {
          ^bb0(%in: f32, %out: f32):
            linalg.yield %in : f32
          }
          linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%8#4, %alloc_4 : memref<32x32xf32>, memref<32x32xf32>) outs(%8#4 : memref<32x32xf32>) {
          ^bb0(%in: f32, %in_13: f32, %out: f32):
            %10 = arith.divf %in, %in_13 : f32
            linalg.yield %10 : f32
          }
          %alloc_11 = memref.alloc() {alignment = 64 : i64} : memref<32x32xf16>
          linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%8#4 : memref<32x32xf32>) outs(%alloc_11 : memref<32x32xf16>) {
          ^bb0(%in: f32, %out: f16):
            %10 = arith.truncf %in : f32 to f16
            linalg.yield %10 : f16
          }
          %9 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 1024 + d1 * 65536 + d2 * 8192)>(%arg9, %arg7, %arg8)
          %reinterpret_cast_12 = memref.reinterpret_cast %arg3 to offset: [%9], sizes: [32, 32], strides: [32, 1] {loom.reuse = {spatial = [{depth = 1 : i64, iterator = "%arg8", mapped_to = "x", reuse_type = "no_reuse", volume = 0 : i64}, {depth = 2 : i64, iterator = "%arg9", mapped_to = "y", reuse_type = "no_reuse", volume = 0 : i64}], temporal = [{depth = 0 : i64, iterator = "%arg7", reuse_type = "no_reuse", volume = 0 : i64}]}} : memref<*xf16> to memref<32x32xf16, strided<[32, 1], offset: ?>>
          memref.copy %alloc_11, %reinterpret_cast_12 : memref<32x32xf16> to memref<32x32xf16, strided<[32, 1], offset: ?>>
        } {loom.mapped_to = "y"}
      } {loom.mapped_to = "x"}
    }
    return
  }
  func.func @flashattn_fwd__d1i0_d0i0__c0mem_c1bx_c2mem(%arg0: memref<*xf16> {tt.divisibility = 16 : i32}, %arg1: memref<*xf16> {tt.divisibility = 16 : i32}, %arg2: memref<*xf16> {tt.divisibility = 16 : i32}, %arg3: memref<*xf16> {tt.divisibility = 16 : i32}, %arg4: index, %arg5: index, %arg6: index) {
    affine.for %arg7 = 0 to affine_map<(d0) -> ((d0 ceildiv 8) ceildiv 8)>(%arg4) {
      affine.parallel (%arg8) = (0) to (8) {
        affine.parallel (%arg9) = (0) to (8) {
          %cst = arith.constant 1.000000e+00 : f32
          %cst_0 = arith.constant 0.176776692 : f32
          %cst_1 = arith.constant 0.000000e+00 : f32
          %cst_2 = arith.constant 0xFF800000 : f32
          %alloc = memref.alloc() {alignment = 64 : i64} : memref<32xf32>
          %alloc_3 = memref.alloc() {alignment = 64 : i64} : memref<32xf32>
          linalg.fill ins(%cst : f32) outs(%alloc_3 : memref<32xf32>)
          %alloc_4 = memref.alloc() {alignment = 64 : i64} : memref<32x32xf32>
          %alloc_5 = memref.alloc() {alignment = 64 : i64} : memref<32x32xf32>
          linalg.fill ins(%cst_0 : f32) outs(%alloc_5 : memref<32x32xf32>)
          %alloc_6 = memref.alloc() {alignment = 64 : i64} : memref<32x32xf32>
          linalg.fill ins(%cst_1 : f32) outs(%alloc_6 : memref<32x32xf32>)
          %alloc_7 = memref.alloc() {alignment = 64 : i64} : memref<32xf32>
          linalg.fill ins(%cst_2 : f32) outs(%alloc_7 : memref<32xf32>)
          %7 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 1024 + d1 * 65536 + d2 * 8192)>(%arg9, %arg7, %arg8)
          %reinterpret_cast = memref.reinterpret_cast %arg0 to offset: [%7], sizes: [32, 32], strides: [32, 1] {loom.reuse = {spatial = [{depth = 1 : i64, iterator = "%arg8", mapped_to = "x", reuse_type = "no_reuse", volume = 0 : i64}, {depth = 2 : i64, iterator = "%arg9", mapped_to = "y", reuse_type = "no_reuse", volume = 0 : i64}], temporal = [{depth = 0 : i64, iterator = "%arg7", reuse_type = "no_reuse", volume = 0 : i64}]}} : memref<*xf16> to memref<32x32xf16, strided<[32, 1], offset: ?>>
          %alloc_8 = memref.alloc() {loom.alloc = {local = true, memory_name = "L1", size = 2048 : i64}} : memref<32x32xf16>
          memref.copy %reinterpret_cast, %alloc_8 {loom.copy.choice = {kind = "mem", memory_name = "L1"}} : memref<32x32xf16, strided<[32, 1], offset: ?>> to memref<32x32xf16>
          %c0 = arith.constant 0 : index
          %c512 = arith.constant 512 : index
          %c32 = arith.constant 32 : index
          %alloc_9 = memref.alloc() {alignment = 64 : i64} : memref<32xf32>
          memref.copy %alloc_7, %alloc_9 : memref<32xf32> to memref<32xf32>
          %alloc_10 = memref.alloc() {alignment = 64 : i64} : memref<32x32xf32>
          memref.copy %alloc_6, %alloc_10 : memref<32x32xf32> to memref<32x32xf32>
          %8:5 = scf.for %arg10 = %c0 to %c512 step %c32 iter_args(%arg11 = %c0, %arg12 = %c0, %arg13 = %alloc_9, %arg14 = %alloc_3, %arg15 = %alloc_10) -> (index, index, memref<32xf32>, memref<32xf32>, memref<32x32xf32>) {
            %10 = affine.apply affine_map<()[s0] -> (s0)>()[%arg12]
            %reinterpret_cast_13 = memref.reinterpret_cast %arg1 to offset: [%10], sizes: [32, 32], strides: [512, 1] {loom.reuse = {sequential = [{depth = 3 : i64, iterator = "%arg10", reuse_type = "total_reuse", volume = 2048 : i64}], spatial = [{depth = 1 : i64, iterator = "%arg8", mapped_to = "x", reuse_type = "total_reuse", volume = 2048 : i64}, {depth = 2 : i64, iterator = "%arg9", mapped_to = "y", reuse_type = "total_reuse", volume = 2048 : i64}], temporal = [{depth = 0 : i64, iterator = "%arg7", reuse_type = "total_reuse", volume = 2048 : i64}]}} : memref<*xf16> to memref<32x32xf16, strided<[512, 1], offset: ?>>
            %alloc_14 = memref.alloc() {loom.alloc = {local = true, memory_name = "L1", size = 2048 : i64}} : memref<32x32xf16>
            memref.copy %reinterpret_cast_13, %alloc_14 {loom.copy.choice = {dim = "x", interconnect_name = "horizontal_links", kind = "broadcast"}} : memref<32x32xf16, strided<[512, 1], offset: ?>> to memref<32x32xf16>
            %alloc_15 = memref.alloc() {alignment = 64 : i64} : memref<32x32xf32>
            memref.copy %alloc_6, %alloc_15 : memref<32x32xf32> to memref<32x32xf32>
            linalg.matmul ins(%alloc_8, %alloc_14 : memref<32x32xf16>, memref<32x32xf16>) outs(%alloc_15 : memref<32x32xf32>)
            linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%alloc_15, %alloc_5 : memref<32x32xf32>, memref<32x32xf32>) outs(%alloc_15 : memref<32x32xf32>) {
            ^bb0(%in: f32, %in_24: f32, %out: f32):
              %14 = arith.mulf %in, %in_24 : f32
              linalg.yield %14 : f32
            }
            linalg.transpose ins(%alloc_15 : memref<32x32xf32>) outs(%alloc_4 : memref<32x32xf32>) permutation = [1, 0] 
            %alloc_16 = memref.alloc() {alignment = 64 : i64} : memref<32xf32>
            memref.copy %alloc_7, %alloc_16 : memref<32xf32> to memref<32xf32>
            linalg.reduce ins(%alloc_4 : memref<32x32xf32>) outs(%alloc_16 : memref<32xf32>) dimensions = [0] 
              (%in: f32, %init: f32) {
                %14 = arith.maxnumf %in, %init : f32
                linalg.yield %14 : f32
              }
            %alloc_17 = memref.alloc() {alignment = 64 : i64} : memref<32xf32>
            linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%arg13, %alloc_16 : memref<32xf32>, memref<32xf32>) outs(%alloc_17 : memref<32xf32>) {
            ^bb0(%in: f32, %in_24: f32, %out: f32):
              %14 = arith.maxnumf %in, %in_24 : f32
              linalg.yield %14 : f32
            }
            %expand_shape_18 = memref.expand_shape %alloc_17 [[0, 1]] output_shape [32, 1] : memref<32xf32> into memref<32x1xf32>
            linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, 0)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%expand_shape_18 : memref<32x1xf32>) outs(%alloc_4 : memref<32x32xf32>) attrs =  {broadcastDims = array<i64: 1>} {
            ^bb0(%in: f32, %out: f32):
              linalg.yield %in : f32
            }
            linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%alloc_15, %alloc_4 : memref<32x32xf32>, memref<32x32xf32>) outs(%alloc_15 : memref<32x32xf32>) {
            ^bb0(%in: f32, %in_24: f32, %out: f32):
              %14 = arith.subf %in, %in_24 : f32
              linalg.yield %14 : f32
            }
            linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%alloc_15 : memref<32x32xf32>) outs(%alloc_15 : memref<32x32xf32>) {
            ^bb0(%in: f32, %out: f32):
              %14 = math.exp %in : f32
              linalg.yield %14 : f32
            }
            linalg.transpose ins(%alloc_15 : memref<32x32xf32>) outs(%alloc_4 : memref<32x32xf32>) permutation = [1, 0] 
            linalg.fill ins(%cst_1 : f32) outs(%alloc : memref<32xf32>)
            linalg.reduce ins(%alloc_4 : memref<32x32xf32>) outs(%alloc : memref<32xf32>) dimensions = [0] 
              (%in: f32, %init: f32) {
                %14 = arith.addf %in, %init : f32
                linalg.yield %14 : f32
              }
            linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%arg13, %alloc_17 : memref<32xf32>, memref<32xf32>) outs(%arg13 : memref<32xf32>) {
            ^bb0(%in: f32, %in_24: f32, %out: f32):
              %14 = arith.subf %in, %in_24 : f32
              linalg.yield %14 : f32
            }
            linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%arg13 : memref<32xf32>) outs(%arg13 : memref<32xf32>) {
            ^bb0(%in: f32, %out: f32):
              %14 = math.exp %in : f32
              linalg.yield %14 : f32
            }
            linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%arg14, %arg13 : memref<32xf32>, memref<32xf32>) outs(%arg14 : memref<32xf32>) {
            ^bb0(%in: f32, %in_24: f32, %out: f32):
              %14 = arith.mulf %in, %in_24 : f32
              linalg.yield %14 : f32
            }
            linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%arg14, %alloc : memref<32xf32>, memref<32xf32>) outs(%arg14 : memref<32xf32>) {
            ^bb0(%in: f32, %in_24: f32, %out: f32):
              %14 = arith.addf %in, %in_24 : f32
              linalg.yield %14 : f32
            }
            %11 = affine.apply affine_map<()[s0] -> (s0 * 32)>()[%arg11]
            %reinterpret_cast_19 = memref.reinterpret_cast %arg2 to offset: [%11], sizes: [32, 32], strides: [32, 1] {loom.reuse = {sequential = [{depth = 3 : i64, iterator = "%arg10", reuse_type = "total_reuse", volume = 2048 : i64}], spatial = [{depth = 1 : i64, iterator = "%arg8", mapped_to = "x", reuse_type = "total_reuse", volume = 2048 : i64}, {depth = 2 : i64, iterator = "%arg9", mapped_to = "y", reuse_type = "total_reuse", volume = 2048 : i64}], temporal = [{depth = 0 : i64, iterator = "%arg7", reuse_type = "total_reuse", volume = 2048 : i64}]}} : memref<*xf16> to memref<32x32xf16, strided<[32, 1], offset: ?>>
            %alloc_20 = memref.alloc() {loom.alloc = {local = true, memory_name = "L1", size = 2048 : i64}} : memref<32x32xf16>
            memref.copy %reinterpret_cast_19, %alloc_20 {loom.copy.choice = {kind = "mem", memory_name = "L1"}} : memref<32x32xf16, strided<[32, 1], offset: ?>> to memref<32x32xf16>
            %alloc_21 = memref.alloc() {alignment = 64 : i64} : memref<32x32xf16>
            linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%alloc_15 : memref<32x32xf32>) outs(%alloc_21 : memref<32x32xf16>) {
            ^bb0(%in: f32, %out: f16):
              %14 = arith.truncf %in : f32 to f16
              linalg.yield %14 : f16
            }
            %expand_shape_22 = memref.expand_shape %arg13 [[0, 1]] output_shape [32, 1] : memref<32xf32> into memref<32x1xf32>
            linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, 0)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%expand_shape_22 : memref<32x1xf32>) outs(%alloc_4 : memref<32x32xf32>) attrs =  {broadcastDims = array<i64: 1>} {
            ^bb0(%in: f32, %out: f32):
              linalg.yield %in : f32
            }
            linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%arg15, %alloc_4 : memref<32x32xf32>, memref<32x32xf32>) outs(%arg15 : memref<32x32xf32>) {
            ^bb0(%in: f32, %in_24: f32, %out: f32):
              %14 = arith.mulf %in, %in_24 : f32
              linalg.yield %14 : f32
            }
            %alloc_23 = memref.alloc() {alignment = 64 : i64} : memref<32x32xf32>
            memref.copy %alloc_6, %alloc_23 : memref<32x32xf32> to memref<32x32xf32>
            linalg.matmul ins(%alloc_21, %alloc_20 : memref<32x32xf16>, memref<32x32xf16>) outs(%alloc_23 : memref<32x32xf32>)
            linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%arg15, %alloc_23 : memref<32x32xf32>, memref<32x32xf32>) outs(%arg15 : memref<32x32xf32>) {
            ^bb0(%in: f32, %in_24: f32, %out: f32):
              %14 = arith.addf %in, %in_24 : f32
              linalg.yield %14 : f32
            }
            %12 = arith.addi %arg11, %c32 : index
            %13 = arith.addi %arg12, %c32 : index
            scf.yield %12, %13, %alloc_17, %arg14, %arg15 : index, index, memref<32xf32>, memref<32xf32>, memref<32x32xf32>
          }
          %expand_shape = memref.expand_shape %8#3 [[0, 1]] output_shape [32, 1] : memref<32xf32> into memref<32x1xf32>
          linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, 0)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%expand_shape : memref<32x1xf32>) outs(%alloc_4 : memref<32x32xf32>) attrs =  {broadcastDims = array<i64: 1>} {
          ^bb0(%in: f32, %out: f32):
            linalg.yield %in : f32
          }
          linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%8#4, %alloc_4 : memref<32x32xf32>, memref<32x32xf32>) outs(%8#4 : memref<32x32xf32>) {
          ^bb0(%in: f32, %in_13: f32, %out: f32):
            %10 = arith.divf %in, %in_13 : f32
            linalg.yield %10 : f32
          }
          %alloc_11 = memref.alloc() {alignment = 64 : i64} : memref<32x32xf16>
          linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%8#4 : memref<32x32xf32>) outs(%alloc_11 : memref<32x32xf16>) {
          ^bb0(%in: f32, %out: f16):
            %10 = arith.truncf %in : f32 to f16
            linalg.yield %10 : f16
          }
          %9 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 1024 + d1 * 65536 + d2 * 8192)>(%arg9, %arg7, %arg8)
          %reinterpret_cast_12 = memref.reinterpret_cast %arg3 to offset: [%9], sizes: [32, 32], strides: [32, 1] {loom.reuse = {spatial = [{depth = 1 : i64, iterator = "%arg8", mapped_to = "x", reuse_type = "no_reuse", volume = 0 : i64}, {depth = 2 : i64, iterator = "%arg9", mapped_to = "y", reuse_type = "no_reuse", volume = 0 : i64}], temporal = [{depth = 0 : i64, iterator = "%arg7", reuse_type = "no_reuse", volume = 0 : i64}]}} : memref<*xf16> to memref<32x32xf16, strided<[32, 1], offset: ?>>
          memref.copy %alloc_11, %reinterpret_cast_12 : memref<32x32xf16> to memref<32x32xf16, strided<[32, 1], offset: ?>>
        } {loom.mapped_to = "y"}
      } {loom.mapped_to = "x"}
    }
    return
  }
  func.func @flashattn_fwd__d1i0_d0i0__c0mem_c1by_c2mem(%arg0: memref<*xf16> {tt.divisibility = 16 : i32}, %arg1: memref<*xf16> {tt.divisibility = 16 : i32}, %arg2: memref<*xf16> {tt.divisibility = 16 : i32}, %arg3: memref<*xf16> {tt.divisibility = 16 : i32}, %arg4: index, %arg5: index, %arg6: index) {
    affine.for %arg7 = 0 to affine_map<(d0) -> ((d0 ceildiv 8) ceildiv 8)>(%arg4) {
      affine.parallel (%arg8) = (0) to (8) {
        affine.parallel (%arg9) = (0) to (8) {
          %cst = arith.constant 1.000000e+00 : f32
          %cst_0 = arith.constant 0.176776692 : f32
          %cst_1 = arith.constant 0.000000e+00 : f32
          %cst_2 = arith.constant 0xFF800000 : f32
          %alloc = memref.alloc() {alignment = 64 : i64} : memref<32xf32>
          %alloc_3 = memref.alloc() {alignment = 64 : i64} : memref<32xf32>
          linalg.fill ins(%cst : f32) outs(%alloc_3 : memref<32xf32>)
          %alloc_4 = memref.alloc() {alignment = 64 : i64} : memref<32x32xf32>
          %alloc_5 = memref.alloc() {alignment = 64 : i64} : memref<32x32xf32>
          linalg.fill ins(%cst_0 : f32) outs(%alloc_5 : memref<32x32xf32>)
          %alloc_6 = memref.alloc() {alignment = 64 : i64} : memref<32x32xf32>
          linalg.fill ins(%cst_1 : f32) outs(%alloc_6 : memref<32x32xf32>)
          %alloc_7 = memref.alloc() {alignment = 64 : i64} : memref<32xf32>
          linalg.fill ins(%cst_2 : f32) outs(%alloc_7 : memref<32xf32>)
          %7 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 1024 + d1 * 65536 + d2 * 8192)>(%arg9, %arg7, %arg8)
          %reinterpret_cast = memref.reinterpret_cast %arg0 to offset: [%7], sizes: [32, 32], strides: [32, 1] {loom.reuse = {spatial = [{depth = 1 : i64, iterator = "%arg8", mapped_to = "x", reuse_type = "no_reuse", volume = 0 : i64}, {depth = 2 : i64, iterator = "%arg9", mapped_to = "y", reuse_type = "no_reuse", volume = 0 : i64}], temporal = [{depth = 0 : i64, iterator = "%arg7", reuse_type = "no_reuse", volume = 0 : i64}]}} : memref<*xf16> to memref<32x32xf16, strided<[32, 1], offset: ?>>
          %alloc_8 = memref.alloc() {loom.alloc = {local = true, memory_name = "L1", size = 2048 : i64}} : memref<32x32xf16>
          memref.copy %reinterpret_cast, %alloc_8 {loom.copy.choice = {kind = "mem", memory_name = "L1"}} : memref<32x32xf16, strided<[32, 1], offset: ?>> to memref<32x32xf16>
          %c0 = arith.constant 0 : index
          %c512 = arith.constant 512 : index
          %c32 = arith.constant 32 : index
          %alloc_9 = memref.alloc() {alignment = 64 : i64} : memref<32xf32>
          memref.copy %alloc_7, %alloc_9 : memref<32xf32> to memref<32xf32>
          %alloc_10 = memref.alloc() {alignment = 64 : i64} : memref<32x32xf32>
          memref.copy %alloc_6, %alloc_10 : memref<32x32xf32> to memref<32x32xf32>
          %8:5 = scf.for %arg10 = %c0 to %c512 step %c32 iter_args(%arg11 = %c0, %arg12 = %c0, %arg13 = %alloc_9, %arg14 = %alloc_3, %arg15 = %alloc_10) -> (index, index, memref<32xf32>, memref<32xf32>, memref<32x32xf32>) {
            %10 = affine.apply affine_map<()[s0] -> (s0)>()[%arg12]
            %reinterpret_cast_13 = memref.reinterpret_cast %arg1 to offset: [%10], sizes: [32, 32], strides: [512, 1] {loom.reuse = {sequential = [{depth = 3 : i64, iterator = "%arg10", reuse_type = "total_reuse", volume = 2048 : i64}], spatial = [{depth = 1 : i64, iterator = "%arg8", mapped_to = "x", reuse_type = "total_reuse", volume = 2048 : i64}, {depth = 2 : i64, iterator = "%arg9", mapped_to = "y", reuse_type = "total_reuse", volume = 2048 : i64}], temporal = [{depth = 0 : i64, iterator = "%arg7", reuse_type = "total_reuse", volume = 2048 : i64}]}} : memref<*xf16> to memref<32x32xf16, strided<[512, 1], offset: ?>>
            %alloc_14 = memref.alloc() {loom.alloc = {local = true, memory_name = "L1", size = 2048 : i64}} : memref<32x32xf16>
            memref.copy %reinterpret_cast_13, %alloc_14 {loom.copy.choice = {dim = "y", interconnect_name = "vertical_links", kind = "broadcast"}} : memref<32x32xf16, strided<[512, 1], offset: ?>> to memref<32x32xf16>
            %alloc_15 = memref.alloc() {alignment = 64 : i64} : memref<32x32xf32>
            memref.copy %alloc_6, %alloc_15 : memref<32x32xf32> to memref<32x32xf32>
            linalg.matmul ins(%alloc_8, %alloc_14 : memref<32x32xf16>, memref<32x32xf16>) outs(%alloc_15 : memref<32x32xf32>)
            linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%alloc_15, %alloc_5 : memref<32x32xf32>, memref<32x32xf32>) outs(%alloc_15 : memref<32x32xf32>) {
            ^bb0(%in: f32, %in_24: f32, %out: f32):
              %14 = arith.mulf %in, %in_24 : f32
              linalg.yield %14 : f32
            }
            linalg.transpose ins(%alloc_15 : memref<32x32xf32>) outs(%alloc_4 : memref<32x32xf32>) permutation = [1, 0] 
            %alloc_16 = memref.alloc() {alignment = 64 : i64} : memref<32xf32>
            memref.copy %alloc_7, %alloc_16 : memref<32xf32> to memref<32xf32>
            linalg.reduce ins(%alloc_4 : memref<32x32xf32>) outs(%alloc_16 : memref<32xf32>) dimensions = [0] 
              (%in: f32, %init: f32) {
                %14 = arith.maxnumf %in, %init : f32
                linalg.yield %14 : f32
              }
            %alloc_17 = memref.alloc() {alignment = 64 : i64} : memref<32xf32>
            linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%arg13, %alloc_16 : memref<32xf32>, memref<32xf32>) outs(%alloc_17 : memref<32xf32>) {
            ^bb0(%in: f32, %in_24: f32, %out: f32):
              %14 = arith.maxnumf %in, %in_24 : f32
              linalg.yield %14 : f32
            }
            %expand_shape_18 = memref.expand_shape %alloc_17 [[0, 1]] output_shape [32, 1] : memref<32xf32> into memref<32x1xf32>
            linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, 0)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%expand_shape_18 : memref<32x1xf32>) outs(%alloc_4 : memref<32x32xf32>) attrs =  {broadcastDims = array<i64: 1>} {
            ^bb0(%in: f32, %out: f32):
              linalg.yield %in : f32
            }
            linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%alloc_15, %alloc_4 : memref<32x32xf32>, memref<32x32xf32>) outs(%alloc_15 : memref<32x32xf32>) {
            ^bb0(%in: f32, %in_24: f32, %out: f32):
              %14 = arith.subf %in, %in_24 : f32
              linalg.yield %14 : f32
            }
            linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%alloc_15 : memref<32x32xf32>) outs(%alloc_15 : memref<32x32xf32>) {
            ^bb0(%in: f32, %out: f32):
              %14 = math.exp %in : f32
              linalg.yield %14 : f32
            }
            linalg.transpose ins(%alloc_15 : memref<32x32xf32>) outs(%alloc_4 : memref<32x32xf32>) permutation = [1, 0] 
            linalg.fill ins(%cst_1 : f32) outs(%alloc : memref<32xf32>)
            linalg.reduce ins(%alloc_4 : memref<32x32xf32>) outs(%alloc : memref<32xf32>) dimensions = [0] 
              (%in: f32, %init: f32) {
                %14 = arith.addf %in, %init : f32
                linalg.yield %14 : f32
              }
            linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%arg13, %alloc_17 : memref<32xf32>, memref<32xf32>) outs(%arg13 : memref<32xf32>) {
            ^bb0(%in: f32, %in_24: f32, %out: f32):
              %14 = arith.subf %in, %in_24 : f32
              linalg.yield %14 : f32
            }
            linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%arg13 : memref<32xf32>) outs(%arg13 : memref<32xf32>) {
            ^bb0(%in: f32, %out: f32):
              %14 = math.exp %in : f32
              linalg.yield %14 : f32
            }
            linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%arg14, %arg13 : memref<32xf32>, memref<32xf32>) outs(%arg14 : memref<32xf32>) {
            ^bb0(%in: f32, %in_24: f32, %out: f32):
              %14 = arith.mulf %in, %in_24 : f32
              linalg.yield %14 : f32
            }
            linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%arg14, %alloc : memref<32xf32>, memref<32xf32>) outs(%arg14 : memref<32xf32>) {
            ^bb0(%in: f32, %in_24: f32, %out: f32):
              %14 = arith.addf %in, %in_24 : f32
              linalg.yield %14 : f32
            }
            %11 = affine.apply affine_map<()[s0] -> (s0 * 32)>()[%arg11]
            %reinterpret_cast_19 = memref.reinterpret_cast %arg2 to offset: [%11], sizes: [32, 32], strides: [32, 1] {loom.reuse = {sequential = [{depth = 3 : i64, iterator = "%arg10", reuse_type = "total_reuse", volume = 2048 : i64}], spatial = [{depth = 1 : i64, iterator = "%arg8", mapped_to = "x", reuse_type = "total_reuse", volume = 2048 : i64}, {depth = 2 : i64, iterator = "%arg9", mapped_to = "y", reuse_type = "total_reuse", volume = 2048 : i64}], temporal = [{depth = 0 : i64, iterator = "%arg7", reuse_type = "total_reuse", volume = 2048 : i64}]}} : memref<*xf16> to memref<32x32xf16, strided<[32, 1], offset: ?>>
            %alloc_20 = memref.alloc() {loom.alloc = {local = true, memory_name = "L1", size = 2048 : i64}} : memref<32x32xf16>
            memref.copy %reinterpret_cast_19, %alloc_20 {loom.copy.choice = {kind = "mem", memory_name = "L1"}} : memref<32x32xf16, strided<[32, 1], offset: ?>> to memref<32x32xf16>
            %alloc_21 = memref.alloc() {alignment = 64 : i64} : memref<32x32xf16>
            linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%alloc_15 : memref<32x32xf32>) outs(%alloc_21 : memref<32x32xf16>) {
            ^bb0(%in: f32, %out: f16):
              %14 = arith.truncf %in : f32 to f16
              linalg.yield %14 : f16
            }
            %expand_shape_22 = memref.expand_shape %arg13 [[0, 1]] output_shape [32, 1] : memref<32xf32> into memref<32x1xf32>
            linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, 0)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%expand_shape_22 : memref<32x1xf32>) outs(%alloc_4 : memref<32x32xf32>) attrs =  {broadcastDims = array<i64: 1>} {
            ^bb0(%in: f32, %out: f32):
              linalg.yield %in : f32
            }
            linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%arg15, %alloc_4 : memref<32x32xf32>, memref<32x32xf32>) outs(%arg15 : memref<32x32xf32>) {
            ^bb0(%in: f32, %in_24: f32, %out: f32):
              %14 = arith.mulf %in, %in_24 : f32
              linalg.yield %14 : f32
            }
            %alloc_23 = memref.alloc() {alignment = 64 : i64} : memref<32x32xf32>
            memref.copy %alloc_6, %alloc_23 : memref<32x32xf32> to memref<32x32xf32>
            linalg.matmul ins(%alloc_21, %alloc_20 : memref<32x32xf16>, memref<32x32xf16>) outs(%alloc_23 : memref<32x32xf32>)
            linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%arg15, %alloc_23 : memref<32x32xf32>, memref<32x32xf32>) outs(%arg15 : memref<32x32xf32>) {
            ^bb0(%in: f32, %in_24: f32, %out: f32):
              %14 = arith.addf %in, %in_24 : f32
              linalg.yield %14 : f32
            }
            %12 = arith.addi %arg11, %c32 : index
            %13 = arith.addi %arg12, %c32 : index
            scf.yield %12, %13, %alloc_17, %arg14, %arg15 : index, index, memref<32xf32>, memref<32xf32>, memref<32x32xf32>
          }
          %expand_shape = memref.expand_shape %8#3 [[0, 1]] output_shape [32, 1] : memref<32xf32> into memref<32x1xf32>
          linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, 0)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%expand_shape : memref<32x1xf32>) outs(%alloc_4 : memref<32x32xf32>) attrs =  {broadcastDims = array<i64: 1>} {
          ^bb0(%in: f32, %out: f32):
            linalg.yield %in : f32
          }
          linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%8#4, %alloc_4 : memref<32x32xf32>, memref<32x32xf32>) outs(%8#4 : memref<32x32xf32>) {
          ^bb0(%in: f32, %in_13: f32, %out: f32):
            %10 = arith.divf %in, %in_13 : f32
            linalg.yield %10 : f32
          }
          %alloc_11 = memref.alloc() {alignment = 64 : i64} : memref<32x32xf16>
          linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%8#4 : memref<32x32xf32>) outs(%alloc_11 : memref<32x32xf16>) {
          ^bb0(%in: f32, %out: f16):
            %10 = arith.truncf %in : f32 to f16
            linalg.yield %10 : f16
          }
          %9 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 1024 + d1 * 65536 + d2 * 8192)>(%arg9, %arg7, %arg8)
          %reinterpret_cast_12 = memref.reinterpret_cast %arg3 to offset: [%9], sizes: [32, 32], strides: [32, 1] {loom.reuse = {spatial = [{depth = 1 : i64, iterator = "%arg8", mapped_to = "x", reuse_type = "no_reuse", volume = 0 : i64}, {depth = 2 : i64, iterator = "%arg9", mapped_to = "y", reuse_type = "no_reuse", volume = 0 : i64}], temporal = [{depth = 0 : i64, iterator = "%arg7", reuse_type = "no_reuse", volume = 0 : i64}]}} : memref<*xf16> to memref<32x32xf16, strided<[32, 1], offset: ?>>
          memref.copy %alloc_11, %reinterpret_cast_12 : memref<32x32xf16> to memref<32x32xf16, strided<[32, 1], offset: ?>>
        } {loom.mapped_to = "y"}
      } {loom.mapped_to = "x"}
    }
    return
  }
  func.func @flashattn_fwd__d1i0_d0i0__c0mem_c1mem_c2bx(%arg0: memref<*xf16> {tt.divisibility = 16 : i32}, %arg1: memref<*xf16> {tt.divisibility = 16 : i32}, %arg2: memref<*xf16> {tt.divisibility = 16 : i32}, %arg3: memref<*xf16> {tt.divisibility = 16 : i32}, %arg4: index, %arg5: index, %arg6: index) {
    affine.for %arg7 = 0 to affine_map<(d0) -> ((d0 ceildiv 8) ceildiv 8)>(%arg4) {
      affine.parallel (%arg8) = (0) to (8) {
        affine.parallel (%arg9) = (0) to (8) {
          %cst = arith.constant 1.000000e+00 : f32
          %cst_0 = arith.constant 0.176776692 : f32
          %cst_1 = arith.constant 0.000000e+00 : f32
          %cst_2 = arith.constant 0xFF800000 : f32
          %alloc = memref.alloc() {alignment = 64 : i64} : memref<32xf32>
          %alloc_3 = memref.alloc() {alignment = 64 : i64} : memref<32xf32>
          linalg.fill ins(%cst : f32) outs(%alloc_3 : memref<32xf32>)
          %alloc_4 = memref.alloc() {alignment = 64 : i64} : memref<32x32xf32>
          %alloc_5 = memref.alloc() {alignment = 64 : i64} : memref<32x32xf32>
          linalg.fill ins(%cst_0 : f32) outs(%alloc_5 : memref<32x32xf32>)
          %alloc_6 = memref.alloc() {alignment = 64 : i64} : memref<32x32xf32>
          linalg.fill ins(%cst_1 : f32) outs(%alloc_6 : memref<32x32xf32>)
          %alloc_7 = memref.alloc() {alignment = 64 : i64} : memref<32xf32>
          linalg.fill ins(%cst_2 : f32) outs(%alloc_7 : memref<32xf32>)
          %7 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 1024 + d1 * 65536 + d2 * 8192)>(%arg9, %arg7, %arg8)
          %reinterpret_cast = memref.reinterpret_cast %arg0 to offset: [%7], sizes: [32, 32], strides: [32, 1] {loom.reuse = {spatial = [{depth = 1 : i64, iterator = "%arg8", mapped_to = "x", reuse_type = "no_reuse", volume = 0 : i64}, {depth = 2 : i64, iterator = "%arg9", mapped_to = "y", reuse_type = "no_reuse", volume = 0 : i64}], temporal = [{depth = 0 : i64, iterator = "%arg7", reuse_type = "no_reuse", volume = 0 : i64}]}} : memref<*xf16> to memref<32x32xf16, strided<[32, 1], offset: ?>>
          %alloc_8 = memref.alloc() {loom.alloc = {local = true, memory_name = "L1", size = 2048 : i64}} : memref<32x32xf16>
          memref.copy %reinterpret_cast, %alloc_8 {loom.copy.choice = {kind = "mem", memory_name = "L1"}} : memref<32x32xf16, strided<[32, 1], offset: ?>> to memref<32x32xf16>
          %c0 = arith.constant 0 : index
          %c512 = arith.constant 512 : index
          %c32 = arith.constant 32 : index
          %alloc_9 = memref.alloc() {alignment = 64 : i64} : memref<32xf32>
          memref.copy %alloc_7, %alloc_9 : memref<32xf32> to memref<32xf32>
          %alloc_10 = memref.alloc() {alignment = 64 : i64} : memref<32x32xf32>
          memref.copy %alloc_6, %alloc_10 : memref<32x32xf32> to memref<32x32xf32>
          %8:5 = scf.for %arg10 = %c0 to %c512 step %c32 iter_args(%arg11 = %c0, %arg12 = %c0, %arg13 = %alloc_9, %arg14 = %alloc_3, %arg15 = %alloc_10) -> (index, index, memref<32xf32>, memref<32xf32>, memref<32x32xf32>) {
            %10 = affine.apply affine_map<()[s0] -> (s0)>()[%arg12]
            %reinterpret_cast_13 = memref.reinterpret_cast %arg1 to offset: [%10], sizes: [32, 32], strides: [512, 1] {loom.reuse = {sequential = [{depth = 3 : i64, iterator = "%arg10", reuse_type = "total_reuse", volume = 2048 : i64}], spatial = [{depth = 1 : i64, iterator = "%arg8", mapped_to = "x", reuse_type = "total_reuse", volume = 2048 : i64}, {depth = 2 : i64, iterator = "%arg9", mapped_to = "y", reuse_type = "total_reuse", volume = 2048 : i64}], temporal = [{depth = 0 : i64, iterator = "%arg7", reuse_type = "total_reuse", volume = 2048 : i64}]}} : memref<*xf16> to memref<32x32xf16, strided<[512, 1], offset: ?>>
            %alloc_14 = memref.alloc() {loom.alloc = {local = true, memory_name = "L1", size = 2048 : i64}} : memref<32x32xf16>
            memref.copy %reinterpret_cast_13, %alloc_14 {loom.copy.choice = {kind = "mem", memory_name = "L1"}} : memref<32x32xf16, strided<[512, 1], offset: ?>> to memref<32x32xf16>
            %alloc_15 = memref.alloc() {alignment = 64 : i64} : memref<32x32xf32>
            memref.copy %alloc_6, %alloc_15 : memref<32x32xf32> to memref<32x32xf32>
            linalg.matmul ins(%alloc_8, %alloc_14 : memref<32x32xf16>, memref<32x32xf16>) outs(%alloc_15 : memref<32x32xf32>)
            linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%alloc_15, %alloc_5 : memref<32x32xf32>, memref<32x32xf32>) outs(%alloc_15 : memref<32x32xf32>) {
            ^bb0(%in: f32, %in_24: f32, %out: f32):
              %14 = arith.mulf %in, %in_24 : f32
              linalg.yield %14 : f32
            }
            linalg.transpose ins(%alloc_15 : memref<32x32xf32>) outs(%alloc_4 : memref<32x32xf32>) permutation = [1, 0] 
            %alloc_16 = memref.alloc() {alignment = 64 : i64} : memref<32xf32>
            memref.copy %alloc_7, %alloc_16 : memref<32xf32> to memref<32xf32>
            linalg.reduce ins(%alloc_4 : memref<32x32xf32>) outs(%alloc_16 : memref<32xf32>) dimensions = [0] 
              (%in: f32, %init: f32) {
                %14 = arith.maxnumf %in, %init : f32
                linalg.yield %14 : f32
              }
            %alloc_17 = memref.alloc() {alignment = 64 : i64} : memref<32xf32>
            linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%arg13, %alloc_16 : memref<32xf32>, memref<32xf32>) outs(%alloc_17 : memref<32xf32>) {
            ^bb0(%in: f32, %in_24: f32, %out: f32):
              %14 = arith.maxnumf %in, %in_24 : f32
              linalg.yield %14 : f32
            }
            %expand_shape_18 = memref.expand_shape %alloc_17 [[0, 1]] output_shape [32, 1] : memref<32xf32> into memref<32x1xf32>
            linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, 0)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%expand_shape_18 : memref<32x1xf32>) outs(%alloc_4 : memref<32x32xf32>) attrs =  {broadcastDims = array<i64: 1>} {
            ^bb0(%in: f32, %out: f32):
              linalg.yield %in : f32
            }
            linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%alloc_15, %alloc_4 : memref<32x32xf32>, memref<32x32xf32>) outs(%alloc_15 : memref<32x32xf32>) {
            ^bb0(%in: f32, %in_24: f32, %out: f32):
              %14 = arith.subf %in, %in_24 : f32
              linalg.yield %14 : f32
            }
            linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%alloc_15 : memref<32x32xf32>) outs(%alloc_15 : memref<32x32xf32>) {
            ^bb0(%in: f32, %out: f32):
              %14 = math.exp %in : f32
              linalg.yield %14 : f32
            }
            linalg.transpose ins(%alloc_15 : memref<32x32xf32>) outs(%alloc_4 : memref<32x32xf32>) permutation = [1, 0] 
            linalg.fill ins(%cst_1 : f32) outs(%alloc : memref<32xf32>)
            linalg.reduce ins(%alloc_4 : memref<32x32xf32>) outs(%alloc : memref<32xf32>) dimensions = [0] 
              (%in: f32, %init: f32) {
                %14 = arith.addf %in, %init : f32
                linalg.yield %14 : f32
              }
            linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%arg13, %alloc_17 : memref<32xf32>, memref<32xf32>) outs(%arg13 : memref<32xf32>) {
            ^bb0(%in: f32, %in_24: f32, %out: f32):
              %14 = arith.subf %in, %in_24 : f32
              linalg.yield %14 : f32
            }
            linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%arg13 : memref<32xf32>) outs(%arg13 : memref<32xf32>) {
            ^bb0(%in: f32, %out: f32):
              %14 = math.exp %in : f32
              linalg.yield %14 : f32
            }
            linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%arg14, %arg13 : memref<32xf32>, memref<32xf32>) outs(%arg14 : memref<32xf32>) {
            ^bb0(%in: f32, %in_24: f32, %out: f32):
              %14 = arith.mulf %in, %in_24 : f32
              linalg.yield %14 : f32
            }
            linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%arg14, %alloc : memref<32xf32>, memref<32xf32>) outs(%arg14 : memref<32xf32>) {
            ^bb0(%in: f32, %in_24: f32, %out: f32):
              %14 = arith.addf %in, %in_24 : f32
              linalg.yield %14 : f32
            }
            %11 = affine.apply affine_map<()[s0] -> (s0 * 32)>()[%arg11]
            %reinterpret_cast_19 = memref.reinterpret_cast %arg2 to offset: [%11], sizes: [32, 32], strides: [32, 1] {loom.reuse = {sequential = [{depth = 3 : i64, iterator = "%arg10", reuse_type = "total_reuse", volume = 2048 : i64}], spatial = [{depth = 1 : i64, iterator = "%arg8", mapped_to = "x", reuse_type = "total_reuse", volume = 2048 : i64}, {depth = 2 : i64, iterator = "%arg9", mapped_to = "y", reuse_type = "total_reuse", volume = 2048 : i64}], temporal = [{depth = 0 : i64, iterator = "%arg7", reuse_type = "total_reuse", volume = 2048 : i64}]}} : memref<*xf16> to memref<32x32xf16, strided<[32, 1], offset: ?>>
            %alloc_20 = memref.alloc() {loom.alloc = {local = true, memory_name = "L1", size = 2048 : i64}} : memref<32x32xf16>
            memref.copy %reinterpret_cast_19, %alloc_20 {loom.copy.choice = {dim = "x", interconnect_name = "horizontal_links", kind = "broadcast"}} : memref<32x32xf16, strided<[32, 1], offset: ?>> to memref<32x32xf16>
            %alloc_21 = memref.alloc() {alignment = 64 : i64} : memref<32x32xf16>
            linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%alloc_15 : memref<32x32xf32>) outs(%alloc_21 : memref<32x32xf16>) {
            ^bb0(%in: f32, %out: f16):
              %14 = arith.truncf %in : f32 to f16
              linalg.yield %14 : f16
            }
            %expand_shape_22 = memref.expand_shape %arg13 [[0, 1]] output_shape [32, 1] : memref<32xf32> into memref<32x1xf32>
            linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, 0)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%expand_shape_22 : memref<32x1xf32>) outs(%alloc_4 : memref<32x32xf32>) attrs =  {broadcastDims = array<i64: 1>} {
            ^bb0(%in: f32, %out: f32):
              linalg.yield %in : f32
            }
            linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%arg15, %alloc_4 : memref<32x32xf32>, memref<32x32xf32>) outs(%arg15 : memref<32x32xf32>) {
            ^bb0(%in: f32, %in_24: f32, %out: f32):
              %14 = arith.mulf %in, %in_24 : f32
              linalg.yield %14 : f32
            }
            %alloc_23 = memref.alloc() {alignment = 64 : i64} : memref<32x32xf32>
            memref.copy %alloc_6, %alloc_23 : memref<32x32xf32> to memref<32x32xf32>
            linalg.matmul ins(%alloc_21, %alloc_20 : memref<32x32xf16>, memref<32x32xf16>) outs(%alloc_23 : memref<32x32xf32>)
            linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%arg15, %alloc_23 : memref<32x32xf32>, memref<32x32xf32>) outs(%arg15 : memref<32x32xf32>) {
            ^bb0(%in: f32, %in_24: f32, %out: f32):
              %14 = arith.addf %in, %in_24 : f32
              linalg.yield %14 : f32
            }
            %12 = arith.addi %arg11, %c32 : index
            %13 = arith.addi %arg12, %c32 : index
            scf.yield %12, %13, %alloc_17, %arg14, %arg15 : index, index, memref<32xf32>, memref<32xf32>, memref<32x32xf32>
          }
          %expand_shape = memref.expand_shape %8#3 [[0, 1]] output_shape [32, 1] : memref<32xf32> into memref<32x1xf32>
          linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, 0)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%expand_shape : memref<32x1xf32>) outs(%alloc_4 : memref<32x32xf32>) attrs =  {broadcastDims = array<i64: 1>} {
          ^bb0(%in: f32, %out: f32):
            linalg.yield %in : f32
          }
          linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%8#4, %alloc_4 : memref<32x32xf32>, memref<32x32xf32>) outs(%8#4 : memref<32x32xf32>) {
          ^bb0(%in: f32, %in_13: f32, %out: f32):
            %10 = arith.divf %in, %in_13 : f32
            linalg.yield %10 : f32
          }
          %alloc_11 = memref.alloc() {alignment = 64 : i64} : memref<32x32xf16>
          linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%8#4 : memref<32x32xf32>) outs(%alloc_11 : memref<32x32xf16>) {
          ^bb0(%in: f32, %out: f16):
            %10 = arith.truncf %in : f32 to f16
            linalg.yield %10 : f16
          }
          %9 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 1024 + d1 * 65536 + d2 * 8192)>(%arg9, %arg7, %arg8)
          %reinterpret_cast_12 = memref.reinterpret_cast %arg3 to offset: [%9], sizes: [32, 32], strides: [32, 1] {loom.reuse = {spatial = [{depth = 1 : i64, iterator = "%arg8", mapped_to = "x", reuse_type = "no_reuse", volume = 0 : i64}, {depth = 2 : i64, iterator = "%arg9", mapped_to = "y", reuse_type = "no_reuse", volume = 0 : i64}], temporal = [{depth = 0 : i64, iterator = "%arg7", reuse_type = "no_reuse", volume = 0 : i64}]}} : memref<*xf16> to memref<32x32xf16, strided<[32, 1], offset: ?>>
          memref.copy %alloc_11, %reinterpret_cast_12 : memref<32x32xf16> to memref<32x32xf16, strided<[32, 1], offset: ?>>
        } {loom.mapped_to = "y"}
      } {loom.mapped_to = "x"}
    }
    return
  }
  func.func @flashattn_fwd__d1i0_d0i0__c0mem_c1bx_c2bx(%arg0: memref<*xf16> {tt.divisibility = 16 : i32}, %arg1: memref<*xf16> {tt.divisibility = 16 : i32}, %arg2: memref<*xf16> {tt.divisibility = 16 : i32}, %arg3: memref<*xf16> {tt.divisibility = 16 : i32}, %arg4: index, %arg5: index, %arg6: index) {
    affine.for %arg7 = 0 to affine_map<(d0) -> ((d0 ceildiv 8) ceildiv 8)>(%arg4) {
      affine.parallel (%arg8) = (0) to (8) {
        affine.parallel (%arg9) = (0) to (8) {
          %cst = arith.constant 1.000000e+00 : f32
          %cst_0 = arith.constant 0.176776692 : f32
          %cst_1 = arith.constant 0.000000e+00 : f32
          %cst_2 = arith.constant 0xFF800000 : f32
          %alloc = memref.alloc() {alignment = 64 : i64} : memref<32xf32>
          %alloc_3 = memref.alloc() {alignment = 64 : i64} : memref<32xf32>
          linalg.fill ins(%cst : f32) outs(%alloc_3 : memref<32xf32>)
          %alloc_4 = memref.alloc() {alignment = 64 : i64} : memref<32x32xf32>
          %alloc_5 = memref.alloc() {alignment = 64 : i64} : memref<32x32xf32>
          linalg.fill ins(%cst_0 : f32) outs(%alloc_5 : memref<32x32xf32>)
          %alloc_6 = memref.alloc() {alignment = 64 : i64} : memref<32x32xf32>
          linalg.fill ins(%cst_1 : f32) outs(%alloc_6 : memref<32x32xf32>)
          %alloc_7 = memref.alloc() {alignment = 64 : i64} : memref<32xf32>
          linalg.fill ins(%cst_2 : f32) outs(%alloc_7 : memref<32xf32>)
          %7 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 1024 + d1 * 65536 + d2 * 8192)>(%arg9, %arg7, %arg8)
          %reinterpret_cast = memref.reinterpret_cast %arg0 to offset: [%7], sizes: [32, 32], strides: [32, 1] {loom.reuse = {spatial = [{depth = 1 : i64, iterator = "%arg8", mapped_to = "x", reuse_type = "no_reuse", volume = 0 : i64}, {depth = 2 : i64, iterator = "%arg9", mapped_to = "y", reuse_type = "no_reuse", volume = 0 : i64}], temporal = [{depth = 0 : i64, iterator = "%arg7", reuse_type = "no_reuse", volume = 0 : i64}]}} : memref<*xf16> to memref<32x32xf16, strided<[32, 1], offset: ?>>
          %alloc_8 = memref.alloc() {loom.alloc = {local = true, memory_name = "L1", size = 2048 : i64}} : memref<32x32xf16>
          memref.copy %reinterpret_cast, %alloc_8 {loom.copy.choice = {kind = "mem", memory_name = "L1"}} : memref<32x32xf16, strided<[32, 1], offset: ?>> to memref<32x32xf16>
          %c0 = arith.constant 0 : index
          %c512 = arith.constant 512 : index
          %c32 = arith.constant 32 : index
          %alloc_9 = memref.alloc() {alignment = 64 : i64} : memref<32xf32>
          memref.copy %alloc_7, %alloc_9 : memref<32xf32> to memref<32xf32>
          %alloc_10 = memref.alloc() {alignment = 64 : i64} : memref<32x32xf32>
          memref.copy %alloc_6, %alloc_10 : memref<32x32xf32> to memref<32x32xf32>
          %8:5 = scf.for %arg10 = %c0 to %c512 step %c32 iter_args(%arg11 = %c0, %arg12 = %c0, %arg13 = %alloc_9, %arg14 = %alloc_3, %arg15 = %alloc_10) -> (index, index, memref<32xf32>, memref<32xf32>, memref<32x32xf32>) {
            %10 = affine.apply affine_map<()[s0] -> (s0)>()[%arg12]
            %reinterpret_cast_13 = memref.reinterpret_cast %arg1 to offset: [%10], sizes: [32, 32], strides: [512, 1] {loom.reuse = {sequential = [{depth = 3 : i64, iterator = "%arg10", reuse_type = "total_reuse", volume = 2048 : i64}], spatial = [{depth = 1 : i64, iterator = "%arg8", mapped_to = "x", reuse_type = "total_reuse", volume = 2048 : i64}, {depth = 2 : i64, iterator = "%arg9", mapped_to = "y", reuse_type = "total_reuse", volume = 2048 : i64}], temporal = [{depth = 0 : i64, iterator = "%arg7", reuse_type = "total_reuse", volume = 2048 : i64}]}} : memref<*xf16> to memref<32x32xf16, strided<[512, 1], offset: ?>>
            %alloc_14 = memref.alloc() {loom.alloc = {local = true, memory_name = "L1", size = 2048 : i64}} : memref<32x32xf16>
            memref.copy %reinterpret_cast_13, %alloc_14 {loom.copy.choice = {dim = "x", interconnect_name = "horizontal_links", kind = "broadcast"}} : memref<32x32xf16, strided<[512, 1], offset: ?>> to memref<32x32xf16>
            %alloc_15 = memref.alloc() {alignment = 64 : i64} : memref<32x32xf32>
            memref.copy %alloc_6, %alloc_15 : memref<32x32xf32> to memref<32x32xf32>
            linalg.matmul ins(%alloc_8, %alloc_14 : memref<32x32xf16>, memref<32x32xf16>) outs(%alloc_15 : memref<32x32xf32>)
            linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%alloc_15, %alloc_5 : memref<32x32xf32>, memref<32x32xf32>) outs(%alloc_15 : memref<32x32xf32>) {
            ^bb0(%in: f32, %in_24: f32, %out: f32):
              %14 = arith.mulf %in, %in_24 : f32
              linalg.yield %14 : f32
            }
            linalg.transpose ins(%alloc_15 : memref<32x32xf32>) outs(%alloc_4 : memref<32x32xf32>) permutation = [1, 0] 
            %alloc_16 = memref.alloc() {alignment = 64 : i64} : memref<32xf32>
            memref.copy %alloc_7, %alloc_16 : memref<32xf32> to memref<32xf32>
            linalg.reduce ins(%alloc_4 : memref<32x32xf32>) outs(%alloc_16 : memref<32xf32>) dimensions = [0] 
              (%in: f32, %init: f32) {
                %14 = arith.maxnumf %in, %init : f32
                linalg.yield %14 : f32
              }
            %alloc_17 = memref.alloc() {alignment = 64 : i64} : memref<32xf32>
            linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%arg13, %alloc_16 : memref<32xf32>, memref<32xf32>) outs(%alloc_17 : memref<32xf32>) {
            ^bb0(%in: f32, %in_24: f32, %out: f32):
              %14 = arith.maxnumf %in, %in_24 : f32
              linalg.yield %14 : f32
            }
            %expand_shape_18 = memref.expand_shape %alloc_17 [[0, 1]] output_shape [32, 1] : memref<32xf32> into memref<32x1xf32>
            linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, 0)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%expand_shape_18 : memref<32x1xf32>) outs(%alloc_4 : memref<32x32xf32>) attrs =  {broadcastDims = array<i64: 1>} {
            ^bb0(%in: f32, %out: f32):
              linalg.yield %in : f32
            }
            linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%alloc_15, %alloc_4 : memref<32x32xf32>, memref<32x32xf32>) outs(%alloc_15 : memref<32x32xf32>) {
            ^bb0(%in: f32, %in_24: f32, %out: f32):
              %14 = arith.subf %in, %in_24 : f32
              linalg.yield %14 : f32
            }
            linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%alloc_15 : memref<32x32xf32>) outs(%alloc_15 : memref<32x32xf32>) {
            ^bb0(%in: f32, %out: f32):
              %14 = math.exp %in : f32
              linalg.yield %14 : f32
            }
            linalg.transpose ins(%alloc_15 : memref<32x32xf32>) outs(%alloc_4 : memref<32x32xf32>) permutation = [1, 0] 
            linalg.fill ins(%cst_1 : f32) outs(%alloc : memref<32xf32>)
            linalg.reduce ins(%alloc_4 : memref<32x32xf32>) outs(%alloc : memref<32xf32>) dimensions = [0] 
              (%in: f32, %init: f32) {
                %14 = arith.addf %in, %init : f32
                linalg.yield %14 : f32
              }
            linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%arg13, %alloc_17 : memref<32xf32>, memref<32xf32>) outs(%arg13 : memref<32xf32>) {
            ^bb0(%in: f32, %in_24: f32, %out: f32):
              %14 = arith.subf %in, %in_24 : f32
              linalg.yield %14 : f32
            }
            linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%arg13 : memref<32xf32>) outs(%arg13 : memref<32xf32>) {
            ^bb0(%in: f32, %out: f32):
              %14 = math.exp %in : f32
              linalg.yield %14 : f32
            }
            linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%arg14, %arg13 : memref<32xf32>, memref<32xf32>) outs(%arg14 : memref<32xf32>) {
            ^bb0(%in: f32, %in_24: f32, %out: f32):
              %14 = arith.mulf %in, %in_24 : f32
              linalg.yield %14 : f32
            }
            linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%arg14, %alloc : memref<32xf32>, memref<32xf32>) outs(%arg14 : memref<32xf32>) {
            ^bb0(%in: f32, %in_24: f32, %out: f32):
              %14 = arith.addf %in, %in_24 : f32
              linalg.yield %14 : f32
            }
            %11 = affine.apply affine_map<()[s0] -> (s0 * 32)>()[%arg11]
            %reinterpret_cast_19 = memref.reinterpret_cast %arg2 to offset: [%11], sizes: [32, 32], strides: [32, 1] {loom.reuse = {sequential = [{depth = 3 : i64, iterator = "%arg10", reuse_type = "total_reuse", volume = 2048 : i64}], spatial = [{depth = 1 : i64, iterator = "%arg8", mapped_to = "x", reuse_type = "total_reuse", volume = 2048 : i64}, {depth = 2 : i64, iterator = "%arg9", mapped_to = "y", reuse_type = "total_reuse", volume = 2048 : i64}], temporal = [{depth = 0 : i64, iterator = "%arg7", reuse_type = "total_reuse", volume = 2048 : i64}]}} : memref<*xf16> to memref<32x32xf16, strided<[32, 1], offset: ?>>
            %alloc_20 = memref.alloc() {loom.alloc = {local = true, memory_name = "L1", size = 2048 : i64}} : memref<32x32xf16>
            memref.copy %reinterpret_cast_19, %alloc_20 {loom.copy.choice = {dim = "x", interconnect_name = "horizontal_links", kind = "broadcast"}} : memref<32x32xf16, strided<[32, 1], offset: ?>> to memref<32x32xf16>
            %alloc_21 = memref.alloc() {alignment = 64 : i64} : memref<32x32xf16>
            linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%alloc_15 : memref<32x32xf32>) outs(%alloc_21 : memref<32x32xf16>) {
            ^bb0(%in: f32, %out: f16):
              %14 = arith.truncf %in : f32 to f16
              linalg.yield %14 : f16
            }
            %expand_shape_22 = memref.expand_shape %arg13 [[0, 1]] output_shape [32, 1] : memref<32xf32> into memref<32x1xf32>
            linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, 0)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%expand_shape_22 : memref<32x1xf32>) outs(%alloc_4 : memref<32x32xf32>) attrs =  {broadcastDims = array<i64: 1>} {
            ^bb0(%in: f32, %out: f32):
              linalg.yield %in : f32
            }
            linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%arg15, %alloc_4 : memref<32x32xf32>, memref<32x32xf32>) outs(%arg15 : memref<32x32xf32>) {
            ^bb0(%in: f32, %in_24: f32, %out: f32):
              %14 = arith.mulf %in, %in_24 : f32
              linalg.yield %14 : f32
            }
            %alloc_23 = memref.alloc() {alignment = 64 : i64} : memref<32x32xf32>
            memref.copy %alloc_6, %alloc_23 : memref<32x32xf32> to memref<32x32xf32>
            linalg.matmul ins(%alloc_21, %alloc_20 : memref<32x32xf16>, memref<32x32xf16>) outs(%alloc_23 : memref<32x32xf32>)
            linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%arg15, %alloc_23 : memref<32x32xf32>, memref<32x32xf32>) outs(%arg15 : memref<32x32xf32>) {
            ^bb0(%in: f32, %in_24: f32, %out: f32):
              %14 = arith.addf %in, %in_24 : f32
              linalg.yield %14 : f32
            }
            %12 = arith.addi %arg11, %c32 : index
            %13 = arith.addi %arg12, %c32 : index
            scf.yield %12, %13, %alloc_17, %arg14, %arg15 : index, index, memref<32xf32>, memref<32xf32>, memref<32x32xf32>
          }
          %expand_shape = memref.expand_shape %8#3 [[0, 1]] output_shape [32, 1] : memref<32xf32> into memref<32x1xf32>
          linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, 0)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%expand_shape : memref<32x1xf32>) outs(%alloc_4 : memref<32x32xf32>) attrs =  {broadcastDims = array<i64: 1>} {
          ^bb0(%in: f32, %out: f32):
            linalg.yield %in : f32
          }
          linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%8#4, %alloc_4 : memref<32x32xf32>, memref<32x32xf32>) outs(%8#4 : memref<32x32xf32>) {
          ^bb0(%in: f32, %in_13: f32, %out: f32):
            %10 = arith.divf %in, %in_13 : f32
            linalg.yield %10 : f32
          }
          %alloc_11 = memref.alloc() {alignment = 64 : i64} : memref<32x32xf16>
          linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%8#4 : memref<32x32xf32>) outs(%alloc_11 : memref<32x32xf16>) {
          ^bb0(%in: f32, %out: f16):
            %10 = arith.truncf %in : f32 to f16
            linalg.yield %10 : f16
          }
          %9 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 1024 + d1 * 65536 + d2 * 8192)>(%arg9, %arg7, %arg8)
          %reinterpret_cast_12 = memref.reinterpret_cast %arg3 to offset: [%9], sizes: [32, 32], strides: [32, 1] {loom.reuse = {spatial = [{depth = 1 : i64, iterator = "%arg8", mapped_to = "x", reuse_type = "no_reuse", volume = 0 : i64}, {depth = 2 : i64, iterator = "%arg9", mapped_to = "y", reuse_type = "no_reuse", volume = 0 : i64}], temporal = [{depth = 0 : i64, iterator = "%arg7", reuse_type = "no_reuse", volume = 0 : i64}]}} : memref<*xf16> to memref<32x32xf16, strided<[32, 1], offset: ?>>
          memref.copy %alloc_11, %reinterpret_cast_12 : memref<32x32xf16> to memref<32x32xf16, strided<[32, 1], offset: ?>>
        } {loom.mapped_to = "y"}
      } {loom.mapped_to = "x"}
    }
    return
  }
  func.func @flashattn_fwd__d1i0_d0i0__c0mem_c1by_c2bx(%arg0: memref<*xf16> {tt.divisibility = 16 : i32}, %arg1: memref<*xf16> {tt.divisibility = 16 : i32}, %arg2: memref<*xf16> {tt.divisibility = 16 : i32}, %arg3: memref<*xf16> {tt.divisibility = 16 : i32}, %arg4: index, %arg5: index, %arg6: index) {
    affine.for %arg7 = 0 to affine_map<(d0) -> ((d0 ceildiv 8) ceildiv 8)>(%arg4) {
      affine.parallel (%arg8) = (0) to (8) {
        affine.parallel (%arg9) = (0) to (8) {
          %cst = arith.constant 1.000000e+00 : f32
          %cst_0 = arith.constant 0.176776692 : f32
          %cst_1 = arith.constant 0.000000e+00 : f32
          %cst_2 = arith.constant 0xFF800000 : f32
          %alloc = memref.alloc() {alignment = 64 : i64} : memref<32xf32>
          %alloc_3 = memref.alloc() {alignment = 64 : i64} : memref<32xf32>
          linalg.fill ins(%cst : f32) outs(%alloc_3 : memref<32xf32>)
          %alloc_4 = memref.alloc() {alignment = 64 : i64} : memref<32x32xf32>
          %alloc_5 = memref.alloc() {alignment = 64 : i64} : memref<32x32xf32>
          linalg.fill ins(%cst_0 : f32) outs(%alloc_5 : memref<32x32xf32>)
          %alloc_6 = memref.alloc() {alignment = 64 : i64} : memref<32x32xf32>
          linalg.fill ins(%cst_1 : f32) outs(%alloc_6 : memref<32x32xf32>)
          %alloc_7 = memref.alloc() {alignment = 64 : i64} : memref<32xf32>
          linalg.fill ins(%cst_2 : f32) outs(%alloc_7 : memref<32xf32>)
          %7 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 1024 + d1 * 65536 + d2 * 8192)>(%arg9, %arg7, %arg8)
          %reinterpret_cast = memref.reinterpret_cast %arg0 to offset: [%7], sizes: [32, 32], strides: [32, 1] {loom.reuse = {spatial = [{depth = 1 : i64, iterator = "%arg8", mapped_to = "x", reuse_type = "no_reuse", volume = 0 : i64}, {depth = 2 : i64, iterator = "%arg9", mapped_to = "y", reuse_type = "no_reuse", volume = 0 : i64}], temporal = [{depth = 0 : i64, iterator = "%arg7", reuse_type = "no_reuse", volume = 0 : i64}]}} : memref<*xf16> to memref<32x32xf16, strided<[32, 1], offset: ?>>
          %alloc_8 = memref.alloc() {loom.alloc = {local = true, memory_name = "L1", size = 2048 : i64}} : memref<32x32xf16>
          memref.copy %reinterpret_cast, %alloc_8 {loom.copy.choice = {kind = "mem", memory_name = "L1"}} : memref<32x32xf16, strided<[32, 1], offset: ?>> to memref<32x32xf16>
          %c0 = arith.constant 0 : index
          %c512 = arith.constant 512 : index
          %c32 = arith.constant 32 : index
          %alloc_9 = memref.alloc() {alignment = 64 : i64} : memref<32xf32>
          memref.copy %alloc_7, %alloc_9 : memref<32xf32> to memref<32xf32>
          %alloc_10 = memref.alloc() {alignment = 64 : i64} : memref<32x32xf32>
          memref.copy %alloc_6, %alloc_10 : memref<32x32xf32> to memref<32x32xf32>
          %8:5 = scf.for %arg10 = %c0 to %c512 step %c32 iter_args(%arg11 = %c0, %arg12 = %c0, %arg13 = %alloc_9, %arg14 = %alloc_3, %arg15 = %alloc_10) -> (index, index, memref<32xf32>, memref<32xf32>, memref<32x32xf32>) {
            %10 = affine.apply affine_map<()[s0] -> (s0)>()[%arg12]
            %reinterpret_cast_13 = memref.reinterpret_cast %arg1 to offset: [%10], sizes: [32, 32], strides: [512, 1] {loom.reuse = {sequential = [{depth = 3 : i64, iterator = "%arg10", reuse_type = "total_reuse", volume = 2048 : i64}], spatial = [{depth = 1 : i64, iterator = "%arg8", mapped_to = "x", reuse_type = "total_reuse", volume = 2048 : i64}, {depth = 2 : i64, iterator = "%arg9", mapped_to = "y", reuse_type = "total_reuse", volume = 2048 : i64}], temporal = [{depth = 0 : i64, iterator = "%arg7", reuse_type = "total_reuse", volume = 2048 : i64}]}} : memref<*xf16> to memref<32x32xf16, strided<[512, 1], offset: ?>>
            %alloc_14 = memref.alloc() {loom.alloc = {local = true, memory_name = "L1", size = 2048 : i64}} : memref<32x32xf16>
            memref.copy %reinterpret_cast_13, %alloc_14 {loom.copy.choice = {dim = "y", interconnect_name = "vertical_links", kind = "broadcast"}} : memref<32x32xf16, strided<[512, 1], offset: ?>> to memref<32x32xf16>
            %alloc_15 = memref.alloc() {alignment = 64 : i64} : memref<32x32xf32>
            memref.copy %alloc_6, %alloc_15 : memref<32x32xf32> to memref<32x32xf32>
            linalg.matmul ins(%alloc_8, %alloc_14 : memref<32x32xf16>, memref<32x32xf16>) outs(%alloc_15 : memref<32x32xf32>)
            linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%alloc_15, %alloc_5 : memref<32x32xf32>, memref<32x32xf32>) outs(%alloc_15 : memref<32x32xf32>) {
            ^bb0(%in: f32, %in_24: f32, %out: f32):
              %14 = arith.mulf %in, %in_24 : f32
              linalg.yield %14 : f32
            }
            linalg.transpose ins(%alloc_15 : memref<32x32xf32>) outs(%alloc_4 : memref<32x32xf32>) permutation = [1, 0] 
            %alloc_16 = memref.alloc() {alignment = 64 : i64} : memref<32xf32>
            memref.copy %alloc_7, %alloc_16 : memref<32xf32> to memref<32xf32>
            linalg.reduce ins(%alloc_4 : memref<32x32xf32>) outs(%alloc_16 : memref<32xf32>) dimensions = [0] 
              (%in: f32, %init: f32) {
                %14 = arith.maxnumf %in, %init : f32
                linalg.yield %14 : f32
              }
            %alloc_17 = memref.alloc() {alignment = 64 : i64} : memref<32xf32>
            linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%arg13, %alloc_16 : memref<32xf32>, memref<32xf32>) outs(%alloc_17 : memref<32xf32>) {
            ^bb0(%in: f32, %in_24: f32, %out: f32):
              %14 = arith.maxnumf %in, %in_24 : f32
              linalg.yield %14 : f32
            }
            %expand_shape_18 = memref.expand_shape %alloc_17 [[0, 1]] output_shape [32, 1] : memref<32xf32> into memref<32x1xf32>
            linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, 0)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%expand_shape_18 : memref<32x1xf32>) outs(%alloc_4 : memref<32x32xf32>) attrs =  {broadcastDims = array<i64: 1>} {
            ^bb0(%in: f32, %out: f32):
              linalg.yield %in : f32
            }
            linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%alloc_15, %alloc_4 : memref<32x32xf32>, memref<32x32xf32>) outs(%alloc_15 : memref<32x32xf32>) {
            ^bb0(%in: f32, %in_24: f32, %out: f32):
              %14 = arith.subf %in, %in_24 : f32
              linalg.yield %14 : f32
            }
            linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%alloc_15 : memref<32x32xf32>) outs(%alloc_15 : memref<32x32xf32>) {
            ^bb0(%in: f32, %out: f32):
              %14 = math.exp %in : f32
              linalg.yield %14 : f32
            }
            linalg.transpose ins(%alloc_15 : memref<32x32xf32>) outs(%alloc_4 : memref<32x32xf32>) permutation = [1, 0] 
            linalg.fill ins(%cst_1 : f32) outs(%alloc : memref<32xf32>)
            linalg.reduce ins(%alloc_4 : memref<32x32xf32>) outs(%alloc : memref<32xf32>) dimensions = [0] 
              (%in: f32, %init: f32) {
                %14 = arith.addf %in, %init : f32
                linalg.yield %14 : f32
              }
            linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%arg13, %alloc_17 : memref<32xf32>, memref<32xf32>) outs(%arg13 : memref<32xf32>) {
            ^bb0(%in: f32, %in_24: f32, %out: f32):
              %14 = arith.subf %in, %in_24 : f32
              linalg.yield %14 : f32
            }
            linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%arg13 : memref<32xf32>) outs(%arg13 : memref<32xf32>) {
            ^bb0(%in: f32, %out: f32):
              %14 = math.exp %in : f32
              linalg.yield %14 : f32
            }
            linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%arg14, %arg13 : memref<32xf32>, memref<32xf32>) outs(%arg14 : memref<32xf32>) {
            ^bb0(%in: f32, %in_24: f32, %out: f32):
              %14 = arith.mulf %in, %in_24 : f32
              linalg.yield %14 : f32
            }
            linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%arg14, %alloc : memref<32xf32>, memref<32xf32>) outs(%arg14 : memref<32xf32>) {
            ^bb0(%in: f32, %in_24: f32, %out: f32):
              %14 = arith.addf %in, %in_24 : f32
              linalg.yield %14 : f32
            }
            %11 = affine.apply affine_map<()[s0] -> (s0 * 32)>()[%arg11]
            %reinterpret_cast_19 = memref.reinterpret_cast %arg2 to offset: [%11], sizes: [32, 32], strides: [32, 1] {loom.reuse = {sequential = [{depth = 3 : i64, iterator = "%arg10", reuse_type = "total_reuse", volume = 2048 : i64}], spatial = [{depth = 1 : i64, iterator = "%arg8", mapped_to = "x", reuse_type = "total_reuse", volume = 2048 : i64}, {depth = 2 : i64, iterator = "%arg9", mapped_to = "y", reuse_type = "total_reuse", volume = 2048 : i64}], temporal = [{depth = 0 : i64, iterator = "%arg7", reuse_type = "total_reuse", volume = 2048 : i64}]}} : memref<*xf16> to memref<32x32xf16, strided<[32, 1], offset: ?>>
            %alloc_20 = memref.alloc() {loom.alloc = {local = true, memory_name = "L1", size = 2048 : i64}} : memref<32x32xf16>
            memref.copy %reinterpret_cast_19, %alloc_20 {loom.copy.choice = {dim = "x", interconnect_name = "horizontal_links", kind = "broadcast"}} : memref<32x32xf16, strided<[32, 1], offset: ?>> to memref<32x32xf16>
            %alloc_21 = memref.alloc() {alignment = 64 : i64} : memref<32x32xf16>
            linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%alloc_15 : memref<32x32xf32>) outs(%alloc_21 : memref<32x32xf16>) {
            ^bb0(%in: f32, %out: f16):
              %14 = arith.truncf %in : f32 to f16
              linalg.yield %14 : f16
            }
            %expand_shape_22 = memref.expand_shape %arg13 [[0, 1]] output_shape [32, 1] : memref<32xf32> into memref<32x1xf32>
            linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, 0)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%expand_shape_22 : memref<32x1xf32>) outs(%alloc_4 : memref<32x32xf32>) attrs =  {broadcastDims = array<i64: 1>} {
            ^bb0(%in: f32, %out: f32):
              linalg.yield %in : f32
            }
            linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%arg15, %alloc_4 : memref<32x32xf32>, memref<32x32xf32>) outs(%arg15 : memref<32x32xf32>) {
            ^bb0(%in: f32, %in_24: f32, %out: f32):
              %14 = arith.mulf %in, %in_24 : f32
              linalg.yield %14 : f32
            }
            %alloc_23 = memref.alloc() {alignment = 64 : i64} : memref<32x32xf32>
            memref.copy %alloc_6, %alloc_23 : memref<32x32xf32> to memref<32x32xf32>
            linalg.matmul ins(%alloc_21, %alloc_20 : memref<32x32xf16>, memref<32x32xf16>) outs(%alloc_23 : memref<32x32xf32>)
            linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%arg15, %alloc_23 : memref<32x32xf32>, memref<32x32xf32>) outs(%arg15 : memref<32x32xf32>) {
            ^bb0(%in: f32, %in_24: f32, %out: f32):
              %14 = arith.addf %in, %in_24 : f32
              linalg.yield %14 : f32
            }
            %12 = arith.addi %arg11, %c32 : index
            %13 = arith.addi %arg12, %c32 : index
            scf.yield %12, %13, %alloc_17, %arg14, %arg15 : index, index, memref<32xf32>, memref<32xf32>, memref<32x32xf32>
          }
          %expand_shape = memref.expand_shape %8#3 [[0, 1]] output_shape [32, 1] : memref<32xf32> into memref<32x1xf32>
          linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, 0)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%expand_shape : memref<32x1xf32>) outs(%alloc_4 : memref<32x32xf32>) attrs =  {broadcastDims = array<i64: 1>} {
          ^bb0(%in: f32, %out: f32):
            linalg.yield %in : f32
          }
          linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%8#4, %alloc_4 : memref<32x32xf32>, memref<32x32xf32>) outs(%8#4 : memref<32x32xf32>) {
          ^bb0(%in: f32, %in_13: f32, %out: f32):
            %10 = arith.divf %in, %in_13 : f32
            linalg.yield %10 : f32
          }
          %alloc_11 = memref.alloc() {alignment = 64 : i64} : memref<32x32xf16>
          linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%8#4 : memref<32x32xf32>) outs(%alloc_11 : memref<32x32xf16>) {
          ^bb0(%in: f32, %out: f16):
            %10 = arith.truncf %in : f32 to f16
            linalg.yield %10 : f16
          }
          %9 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 1024 + d1 * 65536 + d2 * 8192)>(%arg9, %arg7, %arg8)
          %reinterpret_cast_12 = memref.reinterpret_cast %arg3 to offset: [%9], sizes: [32, 32], strides: [32, 1] {loom.reuse = {spatial = [{depth = 1 : i64, iterator = "%arg8", mapped_to = "x", reuse_type = "no_reuse", volume = 0 : i64}, {depth = 2 : i64, iterator = "%arg9", mapped_to = "y", reuse_type = "no_reuse", volume = 0 : i64}], temporal = [{depth = 0 : i64, iterator = "%arg7", reuse_type = "no_reuse", volume = 0 : i64}]}} : memref<*xf16> to memref<32x32xf16, strided<[32, 1], offset: ?>>
          memref.copy %alloc_11, %reinterpret_cast_12 : memref<32x32xf16> to memref<32x32xf16, strided<[32, 1], offset: ?>>
        } {loom.mapped_to = "y"}
      } {loom.mapped_to = "x"}
    }
    return
  }
  func.func @flashattn_fwd__d1i0_d0i0__c0mem_c1mem_c2by(%arg0: memref<*xf16> {tt.divisibility = 16 : i32}, %arg1: memref<*xf16> {tt.divisibility = 16 : i32}, %arg2: memref<*xf16> {tt.divisibility = 16 : i32}, %arg3: memref<*xf16> {tt.divisibility = 16 : i32}, %arg4: index, %arg5: index, %arg6: index) {
    affine.for %arg7 = 0 to affine_map<(d0) -> ((d0 ceildiv 8) ceildiv 8)>(%arg4) {
      affine.parallel (%arg8) = (0) to (8) {
        affine.parallel (%arg9) = (0) to (8) {
          %cst = arith.constant 1.000000e+00 : f32
          %cst_0 = arith.constant 0.176776692 : f32
          %cst_1 = arith.constant 0.000000e+00 : f32
          %cst_2 = arith.constant 0xFF800000 : f32
          %alloc = memref.alloc() {alignment = 64 : i64} : memref<32xf32>
          %alloc_3 = memref.alloc() {alignment = 64 : i64} : memref<32xf32>
          linalg.fill ins(%cst : f32) outs(%alloc_3 : memref<32xf32>)
          %alloc_4 = memref.alloc() {alignment = 64 : i64} : memref<32x32xf32>
          %alloc_5 = memref.alloc() {alignment = 64 : i64} : memref<32x32xf32>
          linalg.fill ins(%cst_0 : f32) outs(%alloc_5 : memref<32x32xf32>)
          %alloc_6 = memref.alloc() {alignment = 64 : i64} : memref<32x32xf32>
          linalg.fill ins(%cst_1 : f32) outs(%alloc_6 : memref<32x32xf32>)
          %alloc_7 = memref.alloc() {alignment = 64 : i64} : memref<32xf32>
          linalg.fill ins(%cst_2 : f32) outs(%alloc_7 : memref<32xf32>)
          %7 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 1024 + d1 * 65536 + d2 * 8192)>(%arg9, %arg7, %arg8)
          %reinterpret_cast = memref.reinterpret_cast %arg0 to offset: [%7], sizes: [32, 32], strides: [32, 1] {loom.reuse = {spatial = [{depth = 1 : i64, iterator = "%arg8", mapped_to = "x", reuse_type = "no_reuse", volume = 0 : i64}, {depth = 2 : i64, iterator = "%arg9", mapped_to = "y", reuse_type = "no_reuse", volume = 0 : i64}], temporal = [{depth = 0 : i64, iterator = "%arg7", reuse_type = "no_reuse", volume = 0 : i64}]}} : memref<*xf16> to memref<32x32xf16, strided<[32, 1], offset: ?>>
          %alloc_8 = memref.alloc() {loom.alloc = {local = true, memory_name = "L1", size = 2048 : i64}} : memref<32x32xf16>
          memref.copy %reinterpret_cast, %alloc_8 {loom.copy.choice = {kind = "mem", memory_name = "L1"}} : memref<32x32xf16, strided<[32, 1], offset: ?>> to memref<32x32xf16>
          %c0 = arith.constant 0 : index
          %c512 = arith.constant 512 : index
          %c32 = arith.constant 32 : index
          %alloc_9 = memref.alloc() {alignment = 64 : i64} : memref<32xf32>
          memref.copy %alloc_7, %alloc_9 : memref<32xf32> to memref<32xf32>
          %alloc_10 = memref.alloc() {alignment = 64 : i64} : memref<32x32xf32>
          memref.copy %alloc_6, %alloc_10 : memref<32x32xf32> to memref<32x32xf32>
          %8:5 = scf.for %arg10 = %c0 to %c512 step %c32 iter_args(%arg11 = %c0, %arg12 = %c0, %arg13 = %alloc_9, %arg14 = %alloc_3, %arg15 = %alloc_10) -> (index, index, memref<32xf32>, memref<32xf32>, memref<32x32xf32>) {
            %10 = affine.apply affine_map<()[s0] -> (s0)>()[%arg12]
            %reinterpret_cast_13 = memref.reinterpret_cast %arg1 to offset: [%10], sizes: [32, 32], strides: [512, 1] {loom.reuse = {sequential = [{depth = 3 : i64, iterator = "%arg10", reuse_type = "total_reuse", volume = 2048 : i64}], spatial = [{depth = 1 : i64, iterator = "%arg8", mapped_to = "x", reuse_type = "total_reuse", volume = 2048 : i64}, {depth = 2 : i64, iterator = "%arg9", mapped_to = "y", reuse_type = "total_reuse", volume = 2048 : i64}], temporal = [{depth = 0 : i64, iterator = "%arg7", reuse_type = "total_reuse", volume = 2048 : i64}]}} : memref<*xf16> to memref<32x32xf16, strided<[512, 1], offset: ?>>
            %alloc_14 = memref.alloc() {loom.alloc = {local = true, memory_name = "L1", size = 2048 : i64}} : memref<32x32xf16>
            memref.copy %reinterpret_cast_13, %alloc_14 {loom.copy.choice = {kind = "mem", memory_name = "L1"}} : memref<32x32xf16, strided<[512, 1], offset: ?>> to memref<32x32xf16>
            %alloc_15 = memref.alloc() {alignment = 64 : i64} : memref<32x32xf32>
            memref.copy %alloc_6, %alloc_15 : memref<32x32xf32> to memref<32x32xf32>
            linalg.matmul ins(%alloc_8, %alloc_14 : memref<32x32xf16>, memref<32x32xf16>) outs(%alloc_15 : memref<32x32xf32>)
            linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%alloc_15, %alloc_5 : memref<32x32xf32>, memref<32x32xf32>) outs(%alloc_15 : memref<32x32xf32>) {
            ^bb0(%in: f32, %in_24: f32, %out: f32):
              %14 = arith.mulf %in, %in_24 : f32
              linalg.yield %14 : f32
            }
            linalg.transpose ins(%alloc_15 : memref<32x32xf32>) outs(%alloc_4 : memref<32x32xf32>) permutation = [1, 0] 
            %alloc_16 = memref.alloc() {alignment = 64 : i64} : memref<32xf32>
            memref.copy %alloc_7, %alloc_16 : memref<32xf32> to memref<32xf32>
            linalg.reduce ins(%alloc_4 : memref<32x32xf32>) outs(%alloc_16 : memref<32xf32>) dimensions = [0] 
              (%in: f32, %init: f32) {
                %14 = arith.maxnumf %in, %init : f32
                linalg.yield %14 : f32
              }
            %alloc_17 = memref.alloc() {alignment = 64 : i64} : memref<32xf32>
            linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%arg13, %alloc_16 : memref<32xf32>, memref<32xf32>) outs(%alloc_17 : memref<32xf32>) {
            ^bb0(%in: f32, %in_24: f32, %out: f32):
              %14 = arith.maxnumf %in, %in_24 : f32
              linalg.yield %14 : f32
            }
            %expand_shape_18 = memref.expand_shape %alloc_17 [[0, 1]] output_shape [32, 1] : memref<32xf32> into memref<32x1xf32>
            linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, 0)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%expand_shape_18 : memref<32x1xf32>) outs(%alloc_4 : memref<32x32xf32>) attrs =  {broadcastDims = array<i64: 1>} {
            ^bb0(%in: f32, %out: f32):
              linalg.yield %in : f32
            }
            linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%alloc_15, %alloc_4 : memref<32x32xf32>, memref<32x32xf32>) outs(%alloc_15 : memref<32x32xf32>) {
            ^bb0(%in: f32, %in_24: f32, %out: f32):
              %14 = arith.subf %in, %in_24 : f32
              linalg.yield %14 : f32
            }
            linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%alloc_15 : memref<32x32xf32>) outs(%alloc_15 : memref<32x32xf32>) {
            ^bb0(%in: f32, %out: f32):
              %14 = math.exp %in : f32
              linalg.yield %14 : f32
            }
            linalg.transpose ins(%alloc_15 : memref<32x32xf32>) outs(%alloc_4 : memref<32x32xf32>) permutation = [1, 0] 
            linalg.fill ins(%cst_1 : f32) outs(%alloc : memref<32xf32>)
            linalg.reduce ins(%alloc_4 : memref<32x32xf32>) outs(%alloc : memref<32xf32>) dimensions = [0] 
              (%in: f32, %init: f32) {
                %14 = arith.addf %in, %init : f32
                linalg.yield %14 : f32
              }
            linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%arg13, %alloc_17 : memref<32xf32>, memref<32xf32>) outs(%arg13 : memref<32xf32>) {
            ^bb0(%in: f32, %in_24: f32, %out: f32):
              %14 = arith.subf %in, %in_24 : f32
              linalg.yield %14 : f32
            }
            linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%arg13 : memref<32xf32>) outs(%arg13 : memref<32xf32>) {
            ^bb0(%in: f32, %out: f32):
              %14 = math.exp %in : f32
              linalg.yield %14 : f32
            }
            linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%arg14, %arg13 : memref<32xf32>, memref<32xf32>) outs(%arg14 : memref<32xf32>) {
            ^bb0(%in: f32, %in_24: f32, %out: f32):
              %14 = arith.mulf %in, %in_24 : f32
              linalg.yield %14 : f32
            }
            linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%arg14, %alloc : memref<32xf32>, memref<32xf32>) outs(%arg14 : memref<32xf32>) {
            ^bb0(%in: f32, %in_24: f32, %out: f32):
              %14 = arith.addf %in, %in_24 : f32
              linalg.yield %14 : f32
            }
            %11 = affine.apply affine_map<()[s0] -> (s0 * 32)>()[%arg11]
            %reinterpret_cast_19 = memref.reinterpret_cast %arg2 to offset: [%11], sizes: [32, 32], strides: [32, 1] {loom.reuse = {sequential = [{depth = 3 : i64, iterator = "%arg10", reuse_type = "total_reuse", volume = 2048 : i64}], spatial = [{depth = 1 : i64, iterator = "%arg8", mapped_to = "x", reuse_type = "total_reuse", volume = 2048 : i64}, {depth = 2 : i64, iterator = "%arg9", mapped_to = "y", reuse_type = "total_reuse", volume = 2048 : i64}], temporal = [{depth = 0 : i64, iterator = "%arg7", reuse_type = "total_reuse", volume = 2048 : i64}]}} : memref<*xf16> to memref<32x32xf16, strided<[32, 1], offset: ?>>
            %alloc_20 = memref.alloc() {loom.alloc = {local = true, memory_name = "L1", size = 2048 : i64}} : memref<32x32xf16>
            memref.copy %reinterpret_cast_19, %alloc_20 {loom.copy.choice = {dim = "y", interconnect_name = "vertical_links", kind = "broadcast"}} : memref<32x32xf16, strided<[32, 1], offset: ?>> to memref<32x32xf16>
            %alloc_21 = memref.alloc() {alignment = 64 : i64} : memref<32x32xf16>
            linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%alloc_15 : memref<32x32xf32>) outs(%alloc_21 : memref<32x32xf16>) {
            ^bb0(%in: f32, %out: f16):
              %14 = arith.truncf %in : f32 to f16
              linalg.yield %14 : f16
            }
            %expand_shape_22 = memref.expand_shape %arg13 [[0, 1]] output_shape [32, 1] : memref<32xf32> into memref<32x1xf32>
            linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, 0)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%expand_shape_22 : memref<32x1xf32>) outs(%alloc_4 : memref<32x32xf32>) attrs =  {broadcastDims = array<i64: 1>} {
            ^bb0(%in: f32, %out: f32):
              linalg.yield %in : f32
            }
            linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%arg15, %alloc_4 : memref<32x32xf32>, memref<32x32xf32>) outs(%arg15 : memref<32x32xf32>) {
            ^bb0(%in: f32, %in_24: f32, %out: f32):
              %14 = arith.mulf %in, %in_24 : f32
              linalg.yield %14 : f32
            }
            %alloc_23 = memref.alloc() {alignment = 64 : i64} : memref<32x32xf32>
            memref.copy %alloc_6, %alloc_23 : memref<32x32xf32> to memref<32x32xf32>
            linalg.matmul ins(%alloc_21, %alloc_20 : memref<32x32xf16>, memref<32x32xf16>) outs(%alloc_23 : memref<32x32xf32>)
            linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%arg15, %alloc_23 : memref<32x32xf32>, memref<32x32xf32>) outs(%arg15 : memref<32x32xf32>) {
            ^bb0(%in: f32, %in_24: f32, %out: f32):
              %14 = arith.addf %in, %in_24 : f32
              linalg.yield %14 : f32
            }
            %12 = arith.addi %arg11, %c32 : index
            %13 = arith.addi %arg12, %c32 : index
            scf.yield %12, %13, %alloc_17, %arg14, %arg15 : index, index, memref<32xf32>, memref<32xf32>, memref<32x32xf32>
          }
          %expand_shape = memref.expand_shape %8#3 [[0, 1]] output_shape [32, 1] : memref<32xf32> into memref<32x1xf32>
          linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, 0)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%expand_shape : memref<32x1xf32>) outs(%alloc_4 : memref<32x32xf32>) attrs =  {broadcastDims = array<i64: 1>} {
          ^bb0(%in: f32, %out: f32):
            linalg.yield %in : f32
          }
          linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%8#4, %alloc_4 : memref<32x32xf32>, memref<32x32xf32>) outs(%8#4 : memref<32x32xf32>) {
          ^bb0(%in: f32, %in_13: f32, %out: f32):
            %10 = arith.divf %in, %in_13 : f32
            linalg.yield %10 : f32
          }
          %alloc_11 = memref.alloc() {alignment = 64 : i64} : memref<32x32xf16>
          linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%8#4 : memref<32x32xf32>) outs(%alloc_11 : memref<32x32xf16>) {
          ^bb0(%in: f32, %out: f16):
            %10 = arith.truncf %in : f32 to f16
            linalg.yield %10 : f16
          }
          %9 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 1024 + d1 * 65536 + d2 * 8192)>(%arg9, %arg7, %arg8)
          %reinterpret_cast_12 = memref.reinterpret_cast %arg3 to offset: [%9], sizes: [32, 32], strides: [32, 1] {loom.reuse = {spatial = [{depth = 1 : i64, iterator = "%arg8", mapped_to = "x", reuse_type = "no_reuse", volume = 0 : i64}, {depth = 2 : i64, iterator = "%arg9", mapped_to = "y", reuse_type = "no_reuse", volume = 0 : i64}], temporal = [{depth = 0 : i64, iterator = "%arg7", reuse_type = "no_reuse", volume = 0 : i64}]}} : memref<*xf16> to memref<32x32xf16, strided<[32, 1], offset: ?>>
          memref.copy %alloc_11, %reinterpret_cast_12 : memref<32x32xf16> to memref<32x32xf16, strided<[32, 1], offset: ?>>
        } {loom.mapped_to = "y"}
      } {loom.mapped_to = "x"}
    }
    return
  }
  func.func @flashattn_fwd__d1i0_d0i0__c0mem_c1bx_c2by(%arg0: memref<*xf16> {tt.divisibility = 16 : i32}, %arg1: memref<*xf16> {tt.divisibility = 16 : i32}, %arg2: memref<*xf16> {tt.divisibility = 16 : i32}, %arg3: memref<*xf16> {tt.divisibility = 16 : i32}, %arg4: index, %arg5: index, %arg6: index) {
    affine.for %arg7 = 0 to affine_map<(d0) -> ((d0 ceildiv 8) ceildiv 8)>(%arg4) {
      affine.parallel (%arg8) = (0) to (8) {
        affine.parallel (%arg9) = (0) to (8) {
          %cst = arith.constant 1.000000e+00 : f32
          %cst_0 = arith.constant 0.176776692 : f32
          %cst_1 = arith.constant 0.000000e+00 : f32
          %cst_2 = arith.constant 0xFF800000 : f32
          %alloc = memref.alloc() {alignment = 64 : i64} : memref<32xf32>
          %alloc_3 = memref.alloc() {alignment = 64 : i64} : memref<32xf32>
          linalg.fill ins(%cst : f32) outs(%alloc_3 : memref<32xf32>)
          %alloc_4 = memref.alloc() {alignment = 64 : i64} : memref<32x32xf32>
          %alloc_5 = memref.alloc() {alignment = 64 : i64} : memref<32x32xf32>
          linalg.fill ins(%cst_0 : f32) outs(%alloc_5 : memref<32x32xf32>)
          %alloc_6 = memref.alloc() {alignment = 64 : i64} : memref<32x32xf32>
          linalg.fill ins(%cst_1 : f32) outs(%alloc_6 : memref<32x32xf32>)
          %alloc_7 = memref.alloc() {alignment = 64 : i64} : memref<32xf32>
          linalg.fill ins(%cst_2 : f32) outs(%alloc_7 : memref<32xf32>)
          %7 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 1024 + d1 * 65536 + d2 * 8192)>(%arg9, %arg7, %arg8)
          %reinterpret_cast = memref.reinterpret_cast %arg0 to offset: [%7], sizes: [32, 32], strides: [32, 1] {loom.reuse = {spatial = [{depth = 1 : i64, iterator = "%arg8", mapped_to = "x", reuse_type = "no_reuse", volume = 0 : i64}, {depth = 2 : i64, iterator = "%arg9", mapped_to = "y", reuse_type = "no_reuse", volume = 0 : i64}], temporal = [{depth = 0 : i64, iterator = "%arg7", reuse_type = "no_reuse", volume = 0 : i64}]}} : memref<*xf16> to memref<32x32xf16, strided<[32, 1], offset: ?>>
          %alloc_8 = memref.alloc() {loom.alloc = {local = true, memory_name = "L1", size = 2048 : i64}} : memref<32x32xf16>
          memref.copy %reinterpret_cast, %alloc_8 {loom.copy.choice = {kind = "mem", memory_name = "L1"}} : memref<32x32xf16, strided<[32, 1], offset: ?>> to memref<32x32xf16>
          %c0 = arith.constant 0 : index
          %c512 = arith.constant 512 : index
          %c32 = arith.constant 32 : index
          %alloc_9 = memref.alloc() {alignment = 64 : i64} : memref<32xf32>
          memref.copy %alloc_7, %alloc_9 : memref<32xf32> to memref<32xf32>
          %alloc_10 = memref.alloc() {alignment = 64 : i64} : memref<32x32xf32>
          memref.copy %alloc_6, %alloc_10 : memref<32x32xf32> to memref<32x32xf32>
          %8:5 = scf.for %arg10 = %c0 to %c512 step %c32 iter_args(%arg11 = %c0, %arg12 = %c0, %arg13 = %alloc_9, %arg14 = %alloc_3, %arg15 = %alloc_10) -> (index, index, memref<32xf32>, memref<32xf32>, memref<32x32xf32>) {
            %10 = affine.apply affine_map<()[s0] -> (s0)>()[%arg12]
            %reinterpret_cast_13 = memref.reinterpret_cast %arg1 to offset: [%10], sizes: [32, 32], strides: [512, 1] {loom.reuse = {sequential = [{depth = 3 : i64, iterator = "%arg10", reuse_type = "total_reuse", volume = 2048 : i64}], spatial = [{depth = 1 : i64, iterator = "%arg8", mapped_to = "x", reuse_type = "total_reuse", volume = 2048 : i64}, {depth = 2 : i64, iterator = "%arg9", mapped_to = "y", reuse_type = "total_reuse", volume = 2048 : i64}], temporal = [{depth = 0 : i64, iterator = "%arg7", reuse_type = "total_reuse", volume = 2048 : i64}]}} : memref<*xf16> to memref<32x32xf16, strided<[512, 1], offset: ?>>
            %alloc_14 = memref.alloc() {loom.alloc = {local = true, memory_name = "L1", size = 2048 : i64}} : memref<32x32xf16>
            memref.copy %reinterpret_cast_13, %alloc_14 {loom.copy.choice = {dim = "x", interconnect_name = "horizontal_links", kind = "broadcast"}} : memref<32x32xf16, strided<[512, 1], offset: ?>> to memref<32x32xf16>
            %alloc_15 = memref.alloc() {alignment = 64 : i64} : memref<32x32xf32>
            memref.copy %alloc_6, %alloc_15 : memref<32x32xf32> to memref<32x32xf32>
            linalg.matmul ins(%alloc_8, %alloc_14 : memref<32x32xf16>, memref<32x32xf16>) outs(%alloc_15 : memref<32x32xf32>)
            linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%alloc_15, %alloc_5 : memref<32x32xf32>, memref<32x32xf32>) outs(%alloc_15 : memref<32x32xf32>) {
            ^bb0(%in: f32, %in_24: f32, %out: f32):
              %14 = arith.mulf %in, %in_24 : f32
              linalg.yield %14 : f32
            }
            linalg.transpose ins(%alloc_15 : memref<32x32xf32>) outs(%alloc_4 : memref<32x32xf32>) permutation = [1, 0] 
            %alloc_16 = memref.alloc() {alignment = 64 : i64} : memref<32xf32>
            memref.copy %alloc_7, %alloc_16 : memref<32xf32> to memref<32xf32>
            linalg.reduce ins(%alloc_4 : memref<32x32xf32>) outs(%alloc_16 : memref<32xf32>) dimensions = [0] 
              (%in: f32, %init: f32) {
                %14 = arith.maxnumf %in, %init : f32
                linalg.yield %14 : f32
              }
            %alloc_17 = memref.alloc() {alignment = 64 : i64} : memref<32xf32>
            linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%arg13, %alloc_16 : memref<32xf32>, memref<32xf32>) outs(%alloc_17 : memref<32xf32>) {
            ^bb0(%in: f32, %in_24: f32, %out: f32):
              %14 = arith.maxnumf %in, %in_24 : f32
              linalg.yield %14 : f32
            }
            %expand_shape_18 = memref.expand_shape %alloc_17 [[0, 1]] output_shape [32, 1] : memref<32xf32> into memref<32x1xf32>
            linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, 0)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%expand_shape_18 : memref<32x1xf32>) outs(%alloc_4 : memref<32x32xf32>) attrs =  {broadcastDims = array<i64: 1>} {
            ^bb0(%in: f32, %out: f32):
              linalg.yield %in : f32
            }
            linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%alloc_15, %alloc_4 : memref<32x32xf32>, memref<32x32xf32>) outs(%alloc_15 : memref<32x32xf32>) {
            ^bb0(%in: f32, %in_24: f32, %out: f32):
              %14 = arith.subf %in, %in_24 : f32
              linalg.yield %14 : f32
            }
            linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%alloc_15 : memref<32x32xf32>) outs(%alloc_15 : memref<32x32xf32>) {
            ^bb0(%in: f32, %out: f32):
              %14 = math.exp %in : f32
              linalg.yield %14 : f32
            }
            linalg.transpose ins(%alloc_15 : memref<32x32xf32>) outs(%alloc_4 : memref<32x32xf32>) permutation = [1, 0] 
            linalg.fill ins(%cst_1 : f32) outs(%alloc : memref<32xf32>)
            linalg.reduce ins(%alloc_4 : memref<32x32xf32>) outs(%alloc : memref<32xf32>) dimensions = [0] 
              (%in: f32, %init: f32) {
                %14 = arith.addf %in, %init : f32
                linalg.yield %14 : f32
              }
            linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%arg13, %alloc_17 : memref<32xf32>, memref<32xf32>) outs(%arg13 : memref<32xf32>) {
            ^bb0(%in: f32, %in_24: f32, %out: f32):
              %14 = arith.subf %in, %in_24 : f32
              linalg.yield %14 : f32
            }
            linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%arg13 : memref<32xf32>) outs(%arg13 : memref<32xf32>) {
            ^bb0(%in: f32, %out: f32):
              %14 = math.exp %in : f32
              linalg.yield %14 : f32
            }
            linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%arg14, %arg13 : memref<32xf32>, memref<32xf32>) outs(%arg14 : memref<32xf32>) {
            ^bb0(%in: f32, %in_24: f32, %out: f32):
              %14 = arith.mulf %in, %in_24 : f32
              linalg.yield %14 : f32
            }
            linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%arg14, %alloc : memref<32xf32>, memref<32xf32>) outs(%arg14 : memref<32xf32>) {
            ^bb0(%in: f32, %in_24: f32, %out: f32):
              %14 = arith.addf %in, %in_24 : f32
              linalg.yield %14 : f32
            }
            %11 = affine.apply affine_map<()[s0] -> (s0 * 32)>()[%arg11]
            %reinterpret_cast_19 = memref.reinterpret_cast %arg2 to offset: [%11], sizes: [32, 32], strides: [32, 1] {loom.reuse = {sequential = [{depth = 3 : i64, iterator = "%arg10", reuse_type = "total_reuse", volume = 2048 : i64}], spatial = [{depth = 1 : i64, iterator = "%arg8", mapped_to = "x", reuse_type = "total_reuse", volume = 2048 : i64}, {depth = 2 : i64, iterator = "%arg9", mapped_to = "y", reuse_type = "total_reuse", volume = 2048 : i64}], temporal = [{depth = 0 : i64, iterator = "%arg7", reuse_type = "total_reuse", volume = 2048 : i64}]}} : memref<*xf16> to memref<32x32xf16, strided<[32, 1], offset: ?>>
            %alloc_20 = memref.alloc() {loom.alloc = {local = true, memory_name = "L1", size = 2048 : i64}} : memref<32x32xf16>
            memref.copy %reinterpret_cast_19, %alloc_20 {loom.copy.choice = {dim = "y", interconnect_name = "vertical_links", kind = "broadcast"}} : memref<32x32xf16, strided<[32, 1], offset: ?>> to memref<32x32xf16>
            %alloc_21 = memref.alloc() {alignment = 64 : i64} : memref<32x32xf16>
            linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%alloc_15 : memref<32x32xf32>) outs(%alloc_21 : memref<32x32xf16>) {
            ^bb0(%in: f32, %out: f16):
              %14 = arith.truncf %in : f32 to f16
              linalg.yield %14 : f16
            }
            %expand_shape_22 = memref.expand_shape %arg13 [[0, 1]] output_shape [32, 1] : memref<32xf32> into memref<32x1xf32>
            linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, 0)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%expand_shape_22 : memref<32x1xf32>) outs(%alloc_4 : memref<32x32xf32>) attrs =  {broadcastDims = array<i64: 1>} {
            ^bb0(%in: f32, %out: f32):
              linalg.yield %in : f32
            }
            linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%arg15, %alloc_4 : memref<32x32xf32>, memref<32x32xf32>) outs(%arg15 : memref<32x32xf32>) {
            ^bb0(%in: f32, %in_24: f32, %out: f32):
              %14 = arith.mulf %in, %in_24 : f32
              linalg.yield %14 : f32
            }
            %alloc_23 = memref.alloc() {alignment = 64 : i64} : memref<32x32xf32>
            memref.copy %alloc_6, %alloc_23 : memref<32x32xf32> to memref<32x32xf32>
            linalg.matmul ins(%alloc_21, %alloc_20 : memref<32x32xf16>, memref<32x32xf16>) outs(%alloc_23 : memref<32x32xf32>)
            linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%arg15, %alloc_23 : memref<32x32xf32>, memref<32x32xf32>) outs(%arg15 : memref<32x32xf32>) {
            ^bb0(%in: f32, %in_24: f32, %out: f32):
              %14 = arith.addf %in, %in_24 : f32
              linalg.yield %14 : f32
            }
            %12 = arith.addi %arg11, %c32 : index
            %13 = arith.addi %arg12, %c32 : index
            scf.yield %12, %13, %alloc_17, %arg14, %arg15 : index, index, memref<32xf32>, memref<32xf32>, memref<32x32xf32>
          }
          %expand_shape = memref.expand_shape %8#3 [[0, 1]] output_shape [32, 1] : memref<32xf32> into memref<32x1xf32>
          linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, 0)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%expand_shape : memref<32x1xf32>) outs(%alloc_4 : memref<32x32xf32>) attrs =  {broadcastDims = array<i64: 1>} {
          ^bb0(%in: f32, %out: f32):
            linalg.yield %in : f32
          }
          linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%8#4, %alloc_4 : memref<32x32xf32>, memref<32x32xf32>) outs(%8#4 : memref<32x32xf32>) {
          ^bb0(%in: f32, %in_13: f32, %out: f32):
            %10 = arith.divf %in, %in_13 : f32
            linalg.yield %10 : f32
          }
          %alloc_11 = memref.alloc() {alignment = 64 : i64} : memref<32x32xf16>
          linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%8#4 : memref<32x32xf32>) outs(%alloc_11 : memref<32x32xf16>) {
          ^bb0(%in: f32, %out: f16):
            %10 = arith.truncf %in : f32 to f16
            linalg.yield %10 : f16
          }
          %9 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 1024 + d1 * 65536 + d2 * 8192)>(%arg9, %arg7, %arg8)
          %reinterpret_cast_12 = memref.reinterpret_cast %arg3 to offset: [%9], sizes: [32, 32], strides: [32, 1] {loom.reuse = {spatial = [{depth = 1 : i64, iterator = "%arg8", mapped_to = "x", reuse_type = "no_reuse", volume = 0 : i64}, {depth = 2 : i64, iterator = "%arg9", mapped_to = "y", reuse_type = "no_reuse", volume = 0 : i64}], temporal = [{depth = 0 : i64, iterator = "%arg7", reuse_type = "no_reuse", volume = 0 : i64}]}} : memref<*xf16> to memref<32x32xf16, strided<[32, 1], offset: ?>>
          memref.copy %alloc_11, %reinterpret_cast_12 : memref<32x32xf16> to memref<32x32xf16, strided<[32, 1], offset: ?>>
        } {loom.mapped_to = "y"}
      } {loom.mapped_to = "x"}
    }
    return
  }
  func.func @flashattn_fwd__d1i0_d0i0__c0mem_c1by_c2by(%arg0: memref<*xf16> {tt.divisibility = 16 : i32}, %arg1: memref<*xf16> {tt.divisibility = 16 : i32}, %arg2: memref<*xf16> {tt.divisibility = 16 : i32}, %arg3: memref<*xf16> {tt.divisibility = 16 : i32}, %arg4: index, %arg5: index, %arg6: index) {
    affine.for %arg7 = 0 to affine_map<(d0) -> ((d0 ceildiv 8) ceildiv 8)>(%arg4) {
      affine.parallel (%arg8) = (0) to (8) {
        affine.parallel (%arg9) = (0) to (8) {
          %cst = arith.constant 1.000000e+00 : f32
          %cst_0 = arith.constant 0.176776692 : f32
          %cst_1 = arith.constant 0.000000e+00 : f32
          %cst_2 = arith.constant 0xFF800000 : f32
          %alloc = memref.alloc() {alignment = 64 : i64} : memref<32xf32>
          %alloc_3 = memref.alloc() {alignment = 64 : i64} : memref<32xf32>
          linalg.fill ins(%cst : f32) outs(%alloc_3 : memref<32xf32>)
          %alloc_4 = memref.alloc() {alignment = 64 : i64} : memref<32x32xf32>
          %alloc_5 = memref.alloc() {alignment = 64 : i64} : memref<32x32xf32>
          linalg.fill ins(%cst_0 : f32) outs(%alloc_5 : memref<32x32xf32>)
          %alloc_6 = memref.alloc() {alignment = 64 : i64} : memref<32x32xf32>
          linalg.fill ins(%cst_1 : f32) outs(%alloc_6 : memref<32x32xf32>)
          %alloc_7 = memref.alloc() {alignment = 64 : i64} : memref<32xf32>
          linalg.fill ins(%cst_2 : f32) outs(%alloc_7 : memref<32xf32>)
          %7 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 1024 + d1 * 65536 + d2 * 8192)>(%arg9, %arg7, %arg8)
          %reinterpret_cast = memref.reinterpret_cast %arg0 to offset: [%7], sizes: [32, 32], strides: [32, 1] {loom.reuse = {spatial = [{depth = 1 : i64, iterator = "%arg8", mapped_to = "x", reuse_type = "no_reuse", volume = 0 : i64}, {depth = 2 : i64, iterator = "%arg9", mapped_to = "y", reuse_type = "no_reuse", volume = 0 : i64}], temporal = [{depth = 0 : i64, iterator = "%arg7", reuse_type = "no_reuse", volume = 0 : i64}]}} : memref<*xf16> to memref<32x32xf16, strided<[32, 1], offset: ?>>
          %alloc_8 = memref.alloc() {loom.alloc = {local = true, memory_name = "L1", size = 2048 : i64}} : memref<32x32xf16>
          memref.copy %reinterpret_cast, %alloc_8 {loom.copy.choice = {kind = "mem", memory_name = "L1"}} : memref<32x32xf16, strided<[32, 1], offset: ?>> to memref<32x32xf16>
          %c0 = arith.constant 0 : index
          %c512 = arith.constant 512 : index
          %c32 = arith.constant 32 : index
          %alloc_9 = memref.alloc() {alignment = 64 : i64} : memref<32xf32>
          memref.copy %alloc_7, %alloc_9 : memref<32xf32> to memref<32xf32>
          %alloc_10 = memref.alloc() {alignment = 64 : i64} : memref<32x32xf32>
          memref.copy %alloc_6, %alloc_10 : memref<32x32xf32> to memref<32x32xf32>
          %8:5 = scf.for %arg10 = %c0 to %c512 step %c32 iter_args(%arg11 = %c0, %arg12 = %c0, %arg13 = %alloc_9, %arg14 = %alloc_3, %arg15 = %alloc_10) -> (index, index, memref<32xf32>, memref<32xf32>, memref<32x32xf32>) {
            %10 = affine.apply affine_map<()[s0] -> (s0)>()[%arg12]
            %reinterpret_cast_13 = memref.reinterpret_cast %arg1 to offset: [%10], sizes: [32, 32], strides: [512, 1] {loom.reuse = {sequential = [{depth = 3 : i64, iterator = "%arg10", reuse_type = "total_reuse", volume = 2048 : i64}], spatial = [{depth = 1 : i64, iterator = "%arg8", mapped_to = "x", reuse_type = "total_reuse", volume = 2048 : i64}, {depth = 2 : i64, iterator = "%arg9", mapped_to = "y", reuse_type = "total_reuse", volume = 2048 : i64}], temporal = [{depth = 0 : i64, iterator = "%arg7", reuse_type = "total_reuse", volume = 2048 : i64}]}} : memref<*xf16> to memref<32x32xf16, strided<[512, 1], offset: ?>>
            %alloc_14 = memref.alloc() {loom.alloc = {local = true, memory_name = "L1", size = 2048 : i64}} : memref<32x32xf16>
            memref.copy %reinterpret_cast_13, %alloc_14 {loom.copy.choice = {dim = "y", interconnect_name = "vertical_links", kind = "broadcast"}} : memref<32x32xf16, strided<[512, 1], offset: ?>> to memref<32x32xf16>
            %alloc_15 = memref.alloc() {alignment = 64 : i64} : memref<32x32xf32>
            memref.copy %alloc_6, %alloc_15 : memref<32x32xf32> to memref<32x32xf32>
            linalg.matmul ins(%alloc_8, %alloc_14 : memref<32x32xf16>, memref<32x32xf16>) outs(%alloc_15 : memref<32x32xf32>)
            linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%alloc_15, %alloc_5 : memref<32x32xf32>, memref<32x32xf32>) outs(%alloc_15 : memref<32x32xf32>) {
            ^bb0(%in: f32, %in_24: f32, %out: f32):
              %14 = arith.mulf %in, %in_24 : f32
              linalg.yield %14 : f32
            }
            linalg.transpose ins(%alloc_15 : memref<32x32xf32>) outs(%alloc_4 : memref<32x32xf32>) permutation = [1, 0] 
            %alloc_16 = memref.alloc() {alignment = 64 : i64} : memref<32xf32>
            memref.copy %alloc_7, %alloc_16 : memref<32xf32> to memref<32xf32>
            linalg.reduce ins(%alloc_4 : memref<32x32xf32>) outs(%alloc_16 : memref<32xf32>) dimensions = [0] 
              (%in: f32, %init: f32) {
                %14 = arith.maxnumf %in, %init : f32
                linalg.yield %14 : f32
              }
            %alloc_17 = memref.alloc() {alignment = 64 : i64} : memref<32xf32>
            linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%arg13, %alloc_16 : memref<32xf32>, memref<32xf32>) outs(%alloc_17 : memref<32xf32>) {
            ^bb0(%in: f32, %in_24: f32, %out: f32):
              %14 = arith.maxnumf %in, %in_24 : f32
              linalg.yield %14 : f32
            }
            %expand_shape_18 = memref.expand_shape %alloc_17 [[0, 1]] output_shape [32, 1] : memref<32xf32> into memref<32x1xf32>
            linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, 0)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%expand_shape_18 : memref<32x1xf32>) outs(%alloc_4 : memref<32x32xf32>) attrs =  {broadcastDims = array<i64: 1>} {
            ^bb0(%in: f32, %out: f32):
              linalg.yield %in : f32
            }
            linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%alloc_15, %alloc_4 : memref<32x32xf32>, memref<32x32xf32>) outs(%alloc_15 : memref<32x32xf32>) {
            ^bb0(%in: f32, %in_24: f32, %out: f32):
              %14 = arith.subf %in, %in_24 : f32
              linalg.yield %14 : f32
            }
            linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%alloc_15 : memref<32x32xf32>) outs(%alloc_15 : memref<32x32xf32>) {
            ^bb0(%in: f32, %out: f32):
              %14 = math.exp %in : f32
              linalg.yield %14 : f32
            }
            linalg.transpose ins(%alloc_15 : memref<32x32xf32>) outs(%alloc_4 : memref<32x32xf32>) permutation = [1, 0] 
            linalg.fill ins(%cst_1 : f32) outs(%alloc : memref<32xf32>)
            linalg.reduce ins(%alloc_4 : memref<32x32xf32>) outs(%alloc : memref<32xf32>) dimensions = [0] 
              (%in: f32, %init: f32) {
                %14 = arith.addf %in, %init : f32
                linalg.yield %14 : f32
              }
            linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%arg13, %alloc_17 : memref<32xf32>, memref<32xf32>) outs(%arg13 : memref<32xf32>) {
            ^bb0(%in: f32, %in_24: f32, %out: f32):
              %14 = arith.subf %in, %in_24 : f32
              linalg.yield %14 : f32
            }
            linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%arg13 : memref<32xf32>) outs(%arg13 : memref<32xf32>) {
            ^bb0(%in: f32, %out: f32):
              %14 = math.exp %in : f32
              linalg.yield %14 : f32
            }
            linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%arg14, %arg13 : memref<32xf32>, memref<32xf32>) outs(%arg14 : memref<32xf32>) {
            ^bb0(%in: f32, %in_24: f32, %out: f32):
              %14 = arith.mulf %in, %in_24 : f32
              linalg.yield %14 : f32
            }
            linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%arg14, %alloc : memref<32xf32>, memref<32xf32>) outs(%arg14 : memref<32xf32>) {
            ^bb0(%in: f32, %in_24: f32, %out: f32):
              %14 = arith.addf %in, %in_24 : f32
              linalg.yield %14 : f32
            }
            %11 = affine.apply affine_map<()[s0] -> (s0 * 32)>()[%arg11]
            %reinterpret_cast_19 = memref.reinterpret_cast %arg2 to offset: [%11], sizes: [32, 32], strides: [32, 1] {loom.reuse = {sequential = [{depth = 3 : i64, iterator = "%arg10", reuse_type = "total_reuse", volume = 2048 : i64}], spatial = [{depth = 1 : i64, iterator = "%arg8", mapped_to = "x", reuse_type = "total_reuse", volume = 2048 : i64}, {depth = 2 : i64, iterator = "%arg9", mapped_to = "y", reuse_type = "total_reuse", volume = 2048 : i64}], temporal = [{depth = 0 : i64, iterator = "%arg7", reuse_type = "total_reuse", volume = 2048 : i64}]}} : memref<*xf16> to memref<32x32xf16, strided<[32, 1], offset: ?>>
            %alloc_20 = memref.alloc() {loom.alloc = {local = true, memory_name = "L1", size = 2048 : i64}} : memref<32x32xf16>
            memref.copy %reinterpret_cast_19, %alloc_20 {loom.copy.choice = {dim = "y", interconnect_name = "vertical_links", kind = "broadcast"}} : memref<32x32xf16, strided<[32, 1], offset: ?>> to memref<32x32xf16>
            %alloc_21 = memref.alloc() {alignment = 64 : i64} : memref<32x32xf16>
            linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%alloc_15 : memref<32x32xf32>) outs(%alloc_21 : memref<32x32xf16>) {
            ^bb0(%in: f32, %out: f16):
              %14 = arith.truncf %in : f32 to f16
              linalg.yield %14 : f16
            }
            %expand_shape_22 = memref.expand_shape %arg13 [[0, 1]] output_shape [32, 1] : memref<32xf32> into memref<32x1xf32>
            linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, 0)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%expand_shape_22 : memref<32x1xf32>) outs(%alloc_4 : memref<32x32xf32>) attrs =  {broadcastDims = array<i64: 1>} {
            ^bb0(%in: f32, %out: f32):
              linalg.yield %in : f32
            }
            linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%arg15, %alloc_4 : memref<32x32xf32>, memref<32x32xf32>) outs(%arg15 : memref<32x32xf32>) {
            ^bb0(%in: f32, %in_24: f32, %out: f32):
              %14 = arith.mulf %in, %in_24 : f32
              linalg.yield %14 : f32
            }
            %alloc_23 = memref.alloc() {alignment = 64 : i64} : memref<32x32xf32>
            memref.copy %alloc_6, %alloc_23 : memref<32x32xf32> to memref<32x32xf32>
            linalg.matmul ins(%alloc_21, %alloc_20 : memref<32x32xf16>, memref<32x32xf16>) outs(%alloc_23 : memref<32x32xf32>)
            linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%arg15, %alloc_23 : memref<32x32xf32>, memref<32x32xf32>) outs(%arg15 : memref<32x32xf32>) {
            ^bb0(%in: f32, %in_24: f32, %out: f32):
              %14 = arith.addf %in, %in_24 : f32
              linalg.yield %14 : f32
            }
            %12 = arith.addi %arg11, %c32 : index
            %13 = arith.addi %arg12, %c32 : index
            scf.yield %12, %13, %alloc_17, %arg14, %arg15 : index, index, memref<32xf32>, memref<32xf32>, memref<32x32xf32>
          }
          %expand_shape = memref.expand_shape %8#3 [[0, 1]] output_shape [32, 1] : memref<32xf32> into memref<32x1xf32>
          linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, 0)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%expand_shape : memref<32x1xf32>) outs(%alloc_4 : memref<32x32xf32>) attrs =  {broadcastDims = array<i64: 1>} {
          ^bb0(%in: f32, %out: f32):
            linalg.yield %in : f32
          }
          linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%8#4, %alloc_4 : memref<32x32xf32>, memref<32x32xf32>) outs(%8#4 : memref<32x32xf32>) {
          ^bb0(%in: f32, %in_13: f32, %out: f32):
            %10 = arith.divf %in, %in_13 : f32
            linalg.yield %10 : f32
          }
          %alloc_11 = memref.alloc() {alignment = 64 : i64} : memref<32x32xf16>
          linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%8#4 : memref<32x32xf32>) outs(%alloc_11 : memref<32x32xf16>) {
          ^bb0(%in: f32, %out: f16):
            %10 = arith.truncf %in : f32 to f16
            linalg.yield %10 : f16
          }
          %9 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 1024 + d1 * 65536 + d2 * 8192)>(%arg9, %arg7, %arg8)
          %reinterpret_cast_12 = memref.reinterpret_cast %arg3 to offset: [%9], sizes: [32, 32], strides: [32, 1] {loom.reuse = {spatial = [{depth = 1 : i64, iterator = "%arg8", mapped_to = "x", reuse_type = "no_reuse", volume = 0 : i64}, {depth = 2 : i64, iterator = "%arg9", mapped_to = "y", reuse_type = "no_reuse", volume = 0 : i64}], temporal = [{depth = 0 : i64, iterator = "%arg7", reuse_type = "no_reuse", volume = 0 : i64}]}} : memref<*xf16> to memref<32x32xf16, strided<[32, 1], offset: ?>>
          memref.copy %alloc_11, %reinterpret_cast_12 : memref<32x32xf16> to memref<32x32xf16, strided<[32, 1], offset: ?>>
        } {loom.mapped_to = "y"}
      } {loom.mapped_to = "x"}
    }
    return
  }
}
