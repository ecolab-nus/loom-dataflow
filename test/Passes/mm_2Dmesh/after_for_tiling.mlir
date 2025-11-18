module {
  %0 = df.spatial_dim "x", 8
  %1 = df.spatial_dim "y", 8
  %2 = df.compute "cores", %0, %1 {map = affine_map<(d0, d1) -> (d0, d1)>}
  %3 = df.memory "L1", %0, %1 {bandwidth = 64 : i64, map = affine_map<(d0, d1) -> (d0, d1)>, size = 32768 : i64}
  %4 = df.mux %2 : !df.compute, %3 : !df.memory, %0, %1 {map = affine_map<(d0, d1) -> (d0, d1)>}
  %5 = df.interconnects "horizontal_links" %3 : !df.memory, %3 : !df.memory, %0, %1 {map = affine_map<(d0, d1) -> (d0 + 1, d1)>} : !df.interconnect
  %6 = df.interconnects "vertical_links" %3 : !df.memory, %3 : !df.memory, %0, %1 {map = affine_map<(d0, d1) -> (d0, d1 + 1)>} : !df.interconnect
  func.func @matmul_kernel__d0i0_d1i0__f01__c0mem_c1mem(%arg0: memref<*xf32> {tt.divisibility = 16 : i32}, %arg1: memref<*xf32> {tt.divisibility = 16 : i32}, %arg2: memref<*xf32> {tt.divisibility = 16 : i32}, %arg3: index, %arg4: index, %arg5: index) {
    affine.for %arg6 = 0 to affine_map<(d0, d1) -> ((d0 ceildiv 8) ceildiv 8)>(%arg3, %arg4) {
      affine.for %arg7 = 0 to affine_map<(d0, d1) -> (d1)>(%arg3, %arg4) {
        affine.parallel (%arg8) = (0) to (8) {
          affine.parallel (%arg9) = (0) to (8) {
            %cst = arith.constant 0.000000e+00 : f32
            %alloc = memref.alloc() {alignment = 64 : i64} : memref<64x64xf32>
            linalg.fill ins(%cst : f32) outs(%alloc : memref<64x64xf32>)
            %c0 = arith.constant 0 : index
            %c8 = arith.constant 8 : index
            %c1 = arith.constant 1 : index
            affine.for %arg10 = 0 to affine_map<(d0) -> (d0 ceildiv 64)>(%arg3) {
              affine.for %arg11 = 0 to affine_map<(d0) -> (d0 ceildiv 64)>(%arg4) {
                %alloc_0 = memref.alloc() {alignment = 64 : i64} : memref<64x64xf32>
                memref.copy %alloc, %alloc_0 : memref<64x64xf32> to memref<64x64xf32>
                %7 = scf.for %arg12 = %c0 to %c8 step %c1 iter_args(%arg13 = %alloc_0) -> (memref<64x64xf32>) {
                  %9 = affine.apply affine_map<(d0, d1, d2, d3, d4) -> (d0 * 64 + d1 * 2097152 + d2 * 32768 + d3 * 2097152 + d4 * 262144)>(%arg12, %arg10, %arg9, %arg6, %arg8)
                  %reinterpret_cast_1 = memref.reinterpret_cast %arg0 to offset: [%9], sizes: [64, 64], strides: [512, 1] {tmd.reuse = {sequential = [{depth = 6 : i64, iterator = "%arg12", reuse_type = "no_reuse", volume = 0 : i64}], spatial = [{depth = 2 : i64, iterator = "%arg8", mapped_to = "y", reuse_type = "no_reuse", volume = 0 : i64}, {depth = 3 : i64, iterator = "%arg9", mapped_to = "x", reuse_type = "no_reuse", volume = 0 : i64}], temporal = [{depth = 0 : i64, iterator = "%arg6", reuse_type = "no_reuse", volume = 0 : i64}, {depth = 1 : i64, iterator = "%arg7", reuse_type = "total_reuse", volume = 16384 : i64}, {depth = 4 : i64, iterator = "%arg10", reuse_type = "no_reuse", volume = 0 : i64}, {depth = 5 : i64, iterator = "%arg11", reuse_type = "total_reuse", volume = 16384 : i64}]}} : memref<*xf32> to memref<64x64xf32, strided<[512, 1], offset: ?>>
                  %alloc_2 = memref.alloc() {tmd.alloc = {local = true, memory_name = "L1", size = 16384 : i64}} : memref<64x64xf32>
                  memref.copy %reinterpret_cast_1, %alloc_2 {tmd.copy.choice = {kind = "mem", memory_name = "L1"}} : memref<64x64xf32, strided<[512, 1], offset: ?>> to memref<64x64xf32>
                  %10 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 32768 + d1 * 4096 + d2 * 64)>(%arg12, %arg11, %arg7)
                  %reinterpret_cast_3 = memref.reinterpret_cast %arg1 to offset: [%10], sizes: [64, 64], strides: [512, 1] {tmd.reuse = {sequential = [{depth = 6 : i64, iterator = "%arg12", reuse_type = "no_reuse", volume = 0 : i64}], spatial = [{depth = 2 : i64, iterator = "%arg8", mapped_to = "y", reuse_type = "total_reuse", volume = 16384 : i64}, {depth = 3 : i64, iterator = "%arg9", mapped_to = "x", reuse_type = "total_reuse", volume = 16384 : i64}], temporal = [{depth = 0 : i64, iterator = "%arg6", reuse_type = "total_reuse", volume = 16384 : i64}, {depth = 1 : i64, iterator = "%arg7", reuse_type = "no_reuse", volume = 0 : i64}, {depth = 4 : i64, iterator = "%arg10", reuse_type = "total_reuse", volume = 16384 : i64}, {depth = 5 : i64, iterator = "%arg11", reuse_type = "no_reuse", volume = 0 : i64}]}} : memref<*xf32> to memref<64x64xf32, strided<[512, 1], offset: ?>>
                  %alloc_4 = memref.alloc() {tmd.alloc = {local = true, memory_name = "L1", size = 16384 : i64}} : memref<64x64xf32>
                  memref.copy %reinterpret_cast_3, %alloc_4 {tmd.copy.choice = {kind = "mem", memory_name = "L1"}} : memref<64x64xf32, strided<[512, 1], offset: ?>> to memref<64x64xf32>
                  %alloc_5 = memref.alloc() {alignment = 64 : i64} : memref<64x64xf32>
                  memref.copy %alloc, %alloc_5 : memref<64x64xf32> to memref<64x64xf32>
                  %c32 = arith.constant 32 : index
                  %alloc_6 = memref.alloc() {tmd.alloc = {local = true, memory_name = "L1", size = 16384 : i64}} : memref<64x64xf32>
                  %alloc_7 = memref.alloc() {tmd.alloc = {local = true, memory_name = "L1", size = 16384 : i64}} : memref<64x64xf32>
                  memref.copy %alloc_2, %alloc_6 : memref<64x64xf32> to memref<64x64xf32>
                  memref.copy %alloc_4, %alloc_7 : memref<64x64xf32> to memref<64x64xf32>
                  affine.for %arg14 = 0 to 2 {
                    affine.for %arg15 = 0 to 2 {
                      %11 = affine.apply affine_map<(d0) -> (d0 * 32)>(%arg14)
                      %12 = affine.apply affine_map<(d0) -> (d0 * 32)>(%arg15)
                      %alloc_8 = memref.alloc() : memref<32x32xf32>
                      %subview = memref.subview %arg13[%11, %12] [%c32, %c32] [%c1, %c1] : memref<64x64xf32> to memref<?x?xf32, strided<[?, ?], offset: ?>>
                      memref.copy %subview, %alloc_8 : memref<?x?xf32, strided<[?, ?], offset: ?>> to memref<32x32xf32>
                      affine.for %arg16 = 0 to 2 {
                        %13 = affine.apply affine_map<(d0) -> (d0 * 32)>(%arg16)
                        %subview_9 = memref.subview %alloc_6[%11, %13] [%c32, %c32] [%c1, %c1] : memref<64x64xf32> to memref<?x?xf32, strided<[?, ?], offset: ?>>
                        %subview_10 = memref.subview %alloc_7[%13, %12] [%c32, %c32] [%c1, %c1] : memref<64x64xf32> to memref<?x?xf32, strided<[?, ?], offset: ?>>
                        linalg.matmul ins(%subview_9, %subview_10 : memref<?x?xf32, strided<[?, ?], offset: ?>>, memref<?x?xf32, strided<[?, ?], offset: ?>>) outs(%alloc_8 : memref<32x32xf32>)
                      }
                      memref.copy %alloc_8, %subview : memref<32x32xf32> to memref<?x?xf32, strided<[?, ?], offset: ?>>
                    }
                  }
                  linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%arg13, %alloc_5 : memref<64x64xf32>, memref<64x64xf32>) outs(%arg13 : memref<64x64xf32>) {
                  ^bb0(%in: f32, %in_8: f32, %out: f32):
                    %11 = arith.addf %in, %in_8 : f32
                    linalg.yield %11 : f32
                  }
                  scf.yield %arg13 : memref<64x64xf32>
                }
                %8 = affine.apply affine_map<(d0, d1, d2, d3, d4, d5) -> (d0 * 4096 + d1 * 64 + d2 * 2097152 + d3 * 32768 + d4 * 2097152 + d5 * 262144)>(%arg11, %arg7, %arg10, %arg9, %arg6, %arg8)
                %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%8], sizes: [64, 64], strides: [512, 1] {tmd.reuse = {spatial = [{depth = 2 : i64, iterator = "%arg8", mapped_to = "y", reuse_type = "no_reuse", volume = 0 : i64}, {depth = 3 : i64, iterator = "%arg9", mapped_to = "x", reuse_type = "no_reuse", volume = 0 : i64}], temporal = [{depth = 0 : i64, iterator = "%arg6", reuse_type = "no_reuse", volume = 0 : i64}, {depth = 1 : i64, iterator = "%arg7", reuse_type = "no_reuse", volume = 0 : i64}, {depth = 4 : i64, iterator = "%arg10", reuse_type = "no_reuse", volume = 0 : i64}, {depth = 5 : i64, iterator = "%arg11", reuse_type = "no_reuse", volume = 0 : i64}]}} : memref<*xf32> to memref<64x64xf32, strided<[512, 1], offset: ?>>
                memref.copy %7, %reinterpret_cast : memref<64x64xf32> to memref<64x64xf32, strided<[512, 1], offset: ?>>
              }
            }
          } {tmd.mapped_to = "x"}
        } {tmd.mapped_to = "y"}
      }
    }
    return
  }
  func.func @matmul_kernel__d0i0_d1i0__f01__c0mem_c1bx(%arg0: memref<*xf32> {tt.divisibility = 16 : i32}, %arg1: memref<*xf32> {tt.divisibility = 16 : i32}, %arg2: memref<*xf32> {tt.divisibility = 16 : i32}, %arg3: index, %arg4: index, %arg5: index) {
    affine.for %arg6 = 0 to affine_map<(d0, d1) -> ((d0 ceildiv 8) ceildiv 8)>(%arg3, %arg4) {
      affine.for %arg7 = 0 to affine_map<(d0, d1) -> (d1)>(%arg3, %arg4) {
        affine.parallel (%arg8) = (0) to (8) {
          affine.parallel (%arg9) = (0) to (8) {
            %cst = arith.constant 0.000000e+00 : f32
            %alloc = memref.alloc() {alignment = 64 : i64} : memref<64x64xf32>
            linalg.fill ins(%cst : f32) outs(%alloc : memref<64x64xf32>)
            %c0 = arith.constant 0 : index
            %c8 = arith.constant 8 : index
            %c1 = arith.constant 1 : index
            affine.for %arg10 = 0 to affine_map<(d0) -> (d0 ceildiv 64)>(%arg3) {
              affine.for %arg11 = 0 to affine_map<(d0) -> (d0 ceildiv 64)>(%arg4) {
                %alloc_0 = memref.alloc() {alignment = 64 : i64} : memref<64x64xf32>
                memref.copy %alloc, %alloc_0 : memref<64x64xf32> to memref<64x64xf32>
                %7 = scf.for %arg12 = %c0 to %c8 step %c1 iter_args(%arg13 = %alloc_0) -> (memref<64x64xf32>) {
                  %9 = affine.apply affine_map<(d0, d1, d2, d3, d4) -> (d0 * 64 + d1 * 2097152 + d2 * 32768 + d3 * 2097152 + d4 * 262144)>(%arg12, %arg10, %arg9, %arg6, %arg8)
                  %reinterpret_cast_1 = memref.reinterpret_cast %arg0 to offset: [%9], sizes: [64, 64], strides: [512, 1] {tmd.reuse = {sequential = [{depth = 6 : i64, iterator = "%arg12", reuse_type = "no_reuse", volume = 0 : i64}], spatial = [{depth = 2 : i64, iterator = "%arg8", mapped_to = "y", reuse_type = "no_reuse", volume = 0 : i64}, {depth = 3 : i64, iterator = "%arg9", mapped_to = "x", reuse_type = "no_reuse", volume = 0 : i64}], temporal = [{depth = 0 : i64, iterator = "%arg6", reuse_type = "no_reuse", volume = 0 : i64}, {depth = 1 : i64, iterator = "%arg7", reuse_type = "total_reuse", volume = 16384 : i64}, {depth = 4 : i64, iterator = "%arg10", reuse_type = "no_reuse", volume = 0 : i64}, {depth = 5 : i64, iterator = "%arg11", reuse_type = "total_reuse", volume = 16384 : i64}]}} : memref<*xf32> to memref<64x64xf32, strided<[512, 1], offset: ?>>
                  %alloc_2 = memref.alloc() {tmd.alloc = {local = true, memory_name = "L1", size = 16384 : i64}} : memref<64x64xf32>
                  memref.copy %reinterpret_cast_1, %alloc_2 {tmd.copy.choice = {kind = "mem", memory_name = "L1"}} : memref<64x64xf32, strided<[512, 1], offset: ?>> to memref<64x64xf32>
                  %10 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 32768 + d1 * 4096 + d2 * 64)>(%arg12, %arg11, %arg7)
                  %reinterpret_cast_3 = memref.reinterpret_cast %arg1 to offset: [%10], sizes: [64, 64], strides: [512, 1] {tmd.reuse = {sequential = [{depth = 6 : i64, iterator = "%arg12", reuse_type = "no_reuse", volume = 0 : i64}], spatial = [{depth = 2 : i64, iterator = "%arg8", mapped_to = "y", reuse_type = "total_reuse", volume = 16384 : i64}, {depth = 3 : i64, iterator = "%arg9", mapped_to = "x", reuse_type = "total_reuse", volume = 16384 : i64}], temporal = [{depth = 0 : i64, iterator = "%arg6", reuse_type = "total_reuse", volume = 16384 : i64}, {depth = 1 : i64, iterator = "%arg7", reuse_type = "no_reuse", volume = 0 : i64}, {depth = 4 : i64, iterator = "%arg10", reuse_type = "total_reuse", volume = 16384 : i64}, {depth = 5 : i64, iterator = "%arg11", reuse_type = "no_reuse", volume = 0 : i64}]}} : memref<*xf32> to memref<64x64xf32, strided<[512, 1], offset: ?>>
                  %alloc_4 = memref.alloc() {tmd.alloc = {local = true, memory_name = "L1", size = 16384 : i64}} : memref<64x64xf32>
                  memref.copy %reinterpret_cast_3, %alloc_4 {tmd.copy.choice = {dim = "x", interconnect_name = "horizontal_links", kind = "broadcast"}} : memref<64x64xf32, strided<[512, 1], offset: ?>> to memref<64x64xf32>
                  %alloc_5 = memref.alloc() {alignment = 64 : i64} : memref<64x64xf32>
                  memref.copy %alloc, %alloc_5 : memref<64x64xf32> to memref<64x64xf32>
                  %c32 = arith.constant 32 : index
                  %alloc_6 = memref.alloc() {tmd.alloc = {local = true, memory_name = "L1", size = 16384 : i64}} : memref<64x64xf32>
                  %alloc_7 = memref.alloc() {tmd.alloc = {local = true, memory_name = "L1", size = 16384 : i64}} : memref<64x64xf32>
                  memref.copy %alloc_2, %alloc_6 : memref<64x64xf32> to memref<64x64xf32>
                  memref.copy %alloc_4, %alloc_7 : memref<64x64xf32> to memref<64x64xf32>
                  affine.for %arg14 = 0 to 2 {
                    affine.for %arg15 = 0 to 2 {
                      %11 = affine.apply affine_map<(d0) -> (d0 * 32)>(%arg14)
                      %12 = affine.apply affine_map<(d0) -> (d0 * 32)>(%arg15)
                      %alloc_8 = memref.alloc() : memref<32x32xf32>
                      %subview = memref.subview %arg13[%11, %12] [%c32, %c32] [%c1, %c1] : memref<64x64xf32> to memref<?x?xf32, strided<[?, ?], offset: ?>>
                      memref.copy %subview, %alloc_8 : memref<?x?xf32, strided<[?, ?], offset: ?>> to memref<32x32xf32>
                      affine.for %arg16 = 0 to 2 {
                        %13 = affine.apply affine_map<(d0) -> (d0 * 32)>(%arg16)
                        %subview_9 = memref.subview %alloc_6[%11, %13] [%c32, %c32] [%c1, %c1] : memref<64x64xf32> to memref<?x?xf32, strided<[?, ?], offset: ?>>
                        %subview_10 = memref.subview %alloc_7[%13, %12] [%c32, %c32] [%c1, %c1] : memref<64x64xf32> to memref<?x?xf32, strided<[?, ?], offset: ?>>
                        linalg.matmul ins(%subview_9, %subview_10 : memref<?x?xf32, strided<[?, ?], offset: ?>>, memref<?x?xf32, strided<[?, ?], offset: ?>>) outs(%alloc_8 : memref<32x32xf32>)
                      }
                      memref.copy %alloc_8, %subview : memref<32x32xf32> to memref<?x?xf32, strided<[?, ?], offset: ?>>
                    }
                  }
                  linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%arg13, %alloc_5 : memref<64x64xf32>, memref<64x64xf32>) outs(%arg13 : memref<64x64xf32>) {
                  ^bb0(%in: f32, %in_8: f32, %out: f32):
                    %11 = arith.addf %in, %in_8 : f32
                    linalg.yield %11 : f32
                  }
                  scf.yield %arg13 : memref<64x64xf32>
                }
                %8 = affine.apply affine_map<(d0, d1, d2, d3, d4, d5) -> (d0 * 4096 + d1 * 64 + d2 * 2097152 + d3 * 32768 + d4 * 2097152 + d5 * 262144)>(%arg11, %arg7, %arg10, %arg9, %arg6, %arg8)
                %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%8], sizes: [64, 64], strides: [512, 1] {tmd.reuse = {spatial = [{depth = 2 : i64, iterator = "%arg8", mapped_to = "y", reuse_type = "no_reuse", volume = 0 : i64}, {depth = 3 : i64, iterator = "%arg9", mapped_to = "x", reuse_type = "no_reuse", volume = 0 : i64}], temporal = [{depth = 0 : i64, iterator = "%arg6", reuse_type = "no_reuse", volume = 0 : i64}, {depth = 1 : i64, iterator = "%arg7", reuse_type = "no_reuse", volume = 0 : i64}, {depth = 4 : i64, iterator = "%arg10", reuse_type = "no_reuse", volume = 0 : i64}, {depth = 5 : i64, iterator = "%arg11", reuse_type = "no_reuse", volume = 0 : i64}]}} : memref<*xf32> to memref<64x64xf32, strided<[512, 1], offset: ?>>
                memref.copy %7, %reinterpret_cast : memref<64x64xf32> to memref<64x64xf32, strided<[512, 1], offset: ?>>
              }
            }
          } {tmd.mapped_to = "x"}
        } {tmd.mapped_to = "y"}
      }
    }
    return
  }
  func.func @matmul_kernel__d0i0_d1i0__f01__c0mem_c1by(%arg0: memref<*xf32> {tt.divisibility = 16 : i32}, %arg1: memref<*xf32> {tt.divisibility = 16 : i32}, %arg2: memref<*xf32> {tt.divisibility = 16 : i32}, %arg3: index, %arg4: index, %arg5: index) {
    affine.for %arg6 = 0 to affine_map<(d0, d1) -> ((d0 ceildiv 8) ceildiv 8)>(%arg3, %arg4) {
      affine.for %arg7 = 0 to affine_map<(d0, d1) -> (d1)>(%arg3, %arg4) {
        affine.parallel (%arg8) = (0) to (8) {
          affine.parallel (%arg9) = (0) to (8) {
            %cst = arith.constant 0.000000e+00 : f32
            %alloc = memref.alloc() {alignment = 64 : i64} : memref<64x64xf32>
            linalg.fill ins(%cst : f32) outs(%alloc : memref<64x64xf32>)
            %c0 = arith.constant 0 : index
            %c8 = arith.constant 8 : index
            %c1 = arith.constant 1 : index
            affine.for %arg10 = 0 to affine_map<(d0) -> (d0 ceildiv 64)>(%arg3) {
              affine.for %arg11 = 0 to affine_map<(d0) -> (d0 ceildiv 64)>(%arg4) {
                %alloc_0 = memref.alloc() {alignment = 64 : i64} : memref<64x64xf32>
                memref.copy %alloc, %alloc_0 : memref<64x64xf32> to memref<64x64xf32>
                %7 = scf.for %arg12 = %c0 to %c8 step %c1 iter_args(%arg13 = %alloc_0) -> (memref<64x64xf32>) {
                  %9 = affine.apply affine_map<(d0, d1, d2, d3, d4) -> (d0 * 64 + d1 * 2097152 + d2 * 32768 + d3 * 2097152 + d4 * 262144)>(%arg12, %arg10, %arg9, %arg6, %arg8)
                  %reinterpret_cast_1 = memref.reinterpret_cast %arg0 to offset: [%9], sizes: [64, 64], strides: [512, 1] {tmd.reuse = {sequential = [{depth = 6 : i64, iterator = "%arg12", reuse_type = "no_reuse", volume = 0 : i64}], spatial = [{depth = 2 : i64, iterator = "%arg8", mapped_to = "y", reuse_type = "no_reuse", volume = 0 : i64}, {depth = 3 : i64, iterator = "%arg9", mapped_to = "x", reuse_type = "no_reuse", volume = 0 : i64}], temporal = [{depth = 0 : i64, iterator = "%arg6", reuse_type = "no_reuse", volume = 0 : i64}, {depth = 1 : i64, iterator = "%arg7", reuse_type = "total_reuse", volume = 16384 : i64}, {depth = 4 : i64, iterator = "%arg10", reuse_type = "no_reuse", volume = 0 : i64}, {depth = 5 : i64, iterator = "%arg11", reuse_type = "total_reuse", volume = 16384 : i64}]}} : memref<*xf32> to memref<64x64xf32, strided<[512, 1], offset: ?>>
                  %alloc_2 = memref.alloc() {tmd.alloc = {local = true, memory_name = "L1", size = 16384 : i64}} : memref<64x64xf32>
                  memref.copy %reinterpret_cast_1, %alloc_2 {tmd.copy.choice = {kind = "mem", memory_name = "L1"}} : memref<64x64xf32, strided<[512, 1], offset: ?>> to memref<64x64xf32>
                  %10 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 32768 + d1 * 4096 + d2 * 64)>(%arg12, %arg11, %arg7)
                  %reinterpret_cast_3 = memref.reinterpret_cast %arg1 to offset: [%10], sizes: [64, 64], strides: [512, 1] {tmd.reuse = {sequential = [{depth = 6 : i64, iterator = "%arg12", reuse_type = "no_reuse", volume = 0 : i64}], spatial = [{depth = 2 : i64, iterator = "%arg8", mapped_to = "y", reuse_type = "total_reuse", volume = 16384 : i64}, {depth = 3 : i64, iterator = "%arg9", mapped_to = "x", reuse_type = "total_reuse", volume = 16384 : i64}], temporal = [{depth = 0 : i64, iterator = "%arg6", reuse_type = "total_reuse", volume = 16384 : i64}, {depth = 1 : i64, iterator = "%arg7", reuse_type = "no_reuse", volume = 0 : i64}, {depth = 4 : i64, iterator = "%arg10", reuse_type = "total_reuse", volume = 16384 : i64}, {depth = 5 : i64, iterator = "%arg11", reuse_type = "no_reuse", volume = 0 : i64}]}} : memref<*xf32> to memref<64x64xf32, strided<[512, 1], offset: ?>>
                  %alloc_4 = memref.alloc() {tmd.alloc = {local = true, memory_name = "L1", size = 16384 : i64}} : memref<64x64xf32>
                  memref.copy %reinterpret_cast_3, %alloc_4 {tmd.copy.choice = {dim = "y", interconnect_name = "vertical_links", kind = "broadcast"}} : memref<64x64xf32, strided<[512, 1], offset: ?>> to memref<64x64xf32>
                  %alloc_5 = memref.alloc() {alignment = 64 : i64} : memref<64x64xf32>
                  memref.copy %alloc, %alloc_5 : memref<64x64xf32> to memref<64x64xf32>
                  %c32 = arith.constant 32 : index
                  %alloc_6 = memref.alloc() {tmd.alloc = {local = true, memory_name = "L1", size = 16384 : i64}} : memref<64x64xf32>
                  %alloc_7 = memref.alloc() {tmd.alloc = {local = true, memory_name = "L1", size = 16384 : i64}} : memref<64x64xf32>
                  memref.copy %alloc_2, %alloc_6 : memref<64x64xf32> to memref<64x64xf32>
                  memref.copy %alloc_4, %alloc_7 : memref<64x64xf32> to memref<64x64xf32>
                  affine.for %arg14 = 0 to 2 {
                    affine.for %arg15 = 0 to 2 {
                      %11 = affine.apply affine_map<(d0) -> (d0 * 32)>(%arg14)
                      %12 = affine.apply affine_map<(d0) -> (d0 * 32)>(%arg15)
                      %alloc_8 = memref.alloc() : memref<32x32xf32>
                      %subview = memref.subview %arg13[%11, %12] [%c32, %c32] [%c1, %c1] : memref<64x64xf32> to memref<?x?xf32, strided<[?, ?], offset: ?>>
                      memref.copy %subview, %alloc_8 : memref<?x?xf32, strided<[?, ?], offset: ?>> to memref<32x32xf32>
                      affine.for %arg16 = 0 to 2 {
                        %13 = affine.apply affine_map<(d0) -> (d0 * 32)>(%arg16)
                        %subview_9 = memref.subview %alloc_6[%11, %13] [%c32, %c32] [%c1, %c1] : memref<64x64xf32> to memref<?x?xf32, strided<[?, ?], offset: ?>>
                        %subview_10 = memref.subview %alloc_7[%13, %12] [%c32, %c32] [%c1, %c1] : memref<64x64xf32> to memref<?x?xf32, strided<[?, ?], offset: ?>>
                        linalg.matmul ins(%subview_9, %subview_10 : memref<?x?xf32, strided<[?, ?], offset: ?>>, memref<?x?xf32, strided<[?, ?], offset: ?>>) outs(%alloc_8 : memref<32x32xf32>)
                      }
                      memref.copy %alloc_8, %subview : memref<32x32xf32> to memref<?x?xf32, strided<[?, ?], offset: ?>>
                    }
                  }
                  linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%arg13, %alloc_5 : memref<64x64xf32>, memref<64x64xf32>) outs(%arg13 : memref<64x64xf32>) {
                  ^bb0(%in: f32, %in_8: f32, %out: f32):
                    %11 = arith.addf %in, %in_8 : f32
                    linalg.yield %11 : f32
                  }
                  scf.yield %arg13 : memref<64x64xf32>
                }
                %8 = affine.apply affine_map<(d0, d1, d2, d3, d4, d5) -> (d0 * 4096 + d1 * 64 + d2 * 2097152 + d3 * 32768 + d4 * 2097152 + d5 * 262144)>(%arg11, %arg7, %arg10, %arg9, %arg6, %arg8)
                %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%8], sizes: [64, 64], strides: [512, 1] {tmd.reuse = {spatial = [{depth = 2 : i64, iterator = "%arg8", mapped_to = "y", reuse_type = "no_reuse", volume = 0 : i64}, {depth = 3 : i64, iterator = "%arg9", mapped_to = "x", reuse_type = "no_reuse", volume = 0 : i64}], temporal = [{depth = 0 : i64, iterator = "%arg6", reuse_type = "no_reuse", volume = 0 : i64}, {depth = 1 : i64, iterator = "%arg7", reuse_type = "no_reuse", volume = 0 : i64}, {depth = 4 : i64, iterator = "%arg10", reuse_type = "no_reuse", volume = 0 : i64}, {depth = 5 : i64, iterator = "%arg11", reuse_type = "no_reuse", volume = 0 : i64}]}} : memref<*xf32> to memref<64x64xf32, strided<[512, 1], offset: ?>>
                memref.copy %7, %reinterpret_cast : memref<64x64xf32> to memref<64x64xf32, strided<[512, 1], offset: ?>>
              }
            }
          } {tmd.mapped_to = "x"}
        } {tmd.mapped_to = "y"}
      }
    }
    return
  }
  func.func @matmul_kernel__d0i0_d1i0__f10__c0mem_c1mem(%arg0: memref<*xf32> {tt.divisibility = 16 : i32}, %arg1: memref<*xf32> {tt.divisibility = 16 : i32}, %arg2: memref<*xf32> {tt.divisibility = 16 : i32}, %arg3: index, %arg4: index, %arg5: index) {
    affine.for %arg6 = 0 to affine_map<(d0, d1) -> (d1)>(%arg3, %arg4) {
      affine.for %arg7 = 0 to affine_map<(d0, d1) -> ((d0 ceildiv 8) ceildiv 8)>(%arg3, %arg4) {
        affine.parallel (%arg8) = (0) to (8) {
          affine.parallel (%arg9) = (0) to (8) {
            %cst = arith.constant 0.000000e+00 : f32
            %alloc = memref.alloc() {alignment = 64 : i64} : memref<64x64xf32>
            linalg.fill ins(%cst : f32) outs(%alloc : memref<64x64xf32>)
            %c0 = arith.constant 0 : index
            %c8 = arith.constant 8 : index
            %c1 = arith.constant 1 : index
            affine.for %arg10 = 0 to affine_map<(d0) -> (d0 ceildiv 64)>(%arg3) {
              affine.for %arg11 = 0 to affine_map<(d0) -> (d0 ceildiv 64)>(%arg4) {
                %alloc_0 = memref.alloc() {alignment = 64 : i64} : memref<64x64xf32>
                memref.copy %alloc, %alloc_0 : memref<64x64xf32> to memref<64x64xf32>
                %7 = scf.for %arg12 = %c0 to %c8 step %c1 iter_args(%arg13 = %alloc_0) -> (memref<64x64xf32>) {
                  %9 = affine.apply affine_map<(d0, d1, d2, d3, d4) -> (d0 * 64 + d1 * 2097152 + d2 * 32768 + d3 * 2097152 + d4 * 262144)>(%arg12, %arg10, %arg9, %arg7, %arg8)
                  %reinterpret_cast_1 = memref.reinterpret_cast %arg0 to offset: [%9], sizes: [64, 64], strides: [512, 1] {tmd.reuse = {sequential = [{depth = 6 : i64, iterator = "%arg12", reuse_type = "no_reuse", volume = 0 : i64}], spatial = [{depth = 2 : i64, iterator = "%arg8", mapped_to = "y", reuse_type = "no_reuse", volume = 0 : i64}, {depth = 3 : i64, iterator = "%arg9", mapped_to = "x", reuse_type = "no_reuse", volume = 0 : i64}], temporal = [{depth = 0 : i64, iterator = "%arg6", reuse_type = "total_reuse", volume = 16384 : i64}, {depth = 1 : i64, iterator = "%arg7", reuse_type = "no_reuse", volume = 0 : i64}, {depth = 4 : i64, iterator = "%arg10", reuse_type = "no_reuse", volume = 0 : i64}, {depth = 5 : i64, iterator = "%arg11", reuse_type = "total_reuse", volume = 16384 : i64}]}} : memref<*xf32> to memref<64x64xf32, strided<[512, 1], offset: ?>>
                  %alloc_2 = memref.alloc() {tmd.alloc = {local = true, memory_name = "L1", size = 16384 : i64}} : memref<64x64xf32>
                  memref.copy %reinterpret_cast_1, %alloc_2 {tmd.copy.choice = {kind = "mem", memory_name = "L1"}} : memref<64x64xf32, strided<[512, 1], offset: ?>> to memref<64x64xf32>
                  %10 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 32768 + d1 * 4096 + d2 * 64)>(%arg12, %arg11, %arg6)
                  %reinterpret_cast_3 = memref.reinterpret_cast %arg1 to offset: [%10], sizes: [64, 64], strides: [512, 1] {tmd.reuse = {sequential = [{depth = 6 : i64, iterator = "%arg12", reuse_type = "no_reuse", volume = 0 : i64}], spatial = [{depth = 2 : i64, iterator = "%arg8", mapped_to = "y", reuse_type = "total_reuse", volume = 16384 : i64}, {depth = 3 : i64, iterator = "%arg9", mapped_to = "x", reuse_type = "total_reuse", volume = 16384 : i64}], temporal = [{depth = 0 : i64, iterator = "%arg6", reuse_type = "no_reuse", volume = 0 : i64}, {depth = 1 : i64, iterator = "%arg7", reuse_type = "total_reuse", volume = 16384 : i64}, {depth = 4 : i64, iterator = "%arg10", reuse_type = "total_reuse", volume = 16384 : i64}, {depth = 5 : i64, iterator = "%arg11", reuse_type = "no_reuse", volume = 0 : i64}]}} : memref<*xf32> to memref<64x64xf32, strided<[512, 1], offset: ?>>
                  %alloc_4 = memref.alloc() {tmd.alloc = {local = true, memory_name = "L1", size = 16384 : i64}} : memref<64x64xf32>
                  memref.copy %reinterpret_cast_3, %alloc_4 {tmd.copy.choice = {kind = "mem", memory_name = "L1"}} : memref<64x64xf32, strided<[512, 1], offset: ?>> to memref<64x64xf32>
                  %alloc_5 = memref.alloc() {alignment = 64 : i64} : memref<64x64xf32>
                  memref.copy %alloc, %alloc_5 : memref<64x64xf32> to memref<64x64xf32>
                  %c32 = arith.constant 32 : index
                  %alloc_6 = memref.alloc() {tmd.alloc = {local = true, memory_name = "L1", size = 16384 : i64}} : memref<64x64xf32>
                  %alloc_7 = memref.alloc() {tmd.alloc = {local = true, memory_name = "L1", size = 16384 : i64}} : memref<64x64xf32>
                  memref.copy %alloc_2, %alloc_6 : memref<64x64xf32> to memref<64x64xf32>
                  memref.copy %alloc_4, %alloc_7 : memref<64x64xf32> to memref<64x64xf32>
                  affine.for %arg14 = 0 to 2 {
                    affine.for %arg15 = 0 to 2 {
                      %11 = affine.apply affine_map<(d0) -> (d0 * 32)>(%arg14)
                      %12 = affine.apply affine_map<(d0) -> (d0 * 32)>(%arg15)
                      %alloc_8 = memref.alloc() : memref<32x32xf32>
                      %subview = memref.subview %arg13[%11, %12] [%c32, %c32] [%c1, %c1] : memref<64x64xf32> to memref<?x?xf32, strided<[?, ?], offset: ?>>
                      memref.copy %subview, %alloc_8 : memref<?x?xf32, strided<[?, ?], offset: ?>> to memref<32x32xf32>
                      affine.for %arg16 = 0 to 2 {
                        %13 = affine.apply affine_map<(d0) -> (d0 * 32)>(%arg16)
                        %subview_9 = memref.subview %alloc_6[%11, %13] [%c32, %c32] [%c1, %c1] : memref<64x64xf32> to memref<?x?xf32, strided<[?, ?], offset: ?>>
                        %subview_10 = memref.subview %alloc_7[%13, %12] [%c32, %c32] [%c1, %c1] : memref<64x64xf32> to memref<?x?xf32, strided<[?, ?], offset: ?>>
                        linalg.matmul ins(%subview_9, %subview_10 : memref<?x?xf32, strided<[?, ?], offset: ?>>, memref<?x?xf32, strided<[?, ?], offset: ?>>) outs(%alloc_8 : memref<32x32xf32>)
                      }
                      memref.copy %alloc_8, %subview : memref<32x32xf32> to memref<?x?xf32, strided<[?, ?], offset: ?>>
                    }
                  }
                  linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%arg13, %alloc_5 : memref<64x64xf32>, memref<64x64xf32>) outs(%arg13 : memref<64x64xf32>) {
                  ^bb0(%in: f32, %in_8: f32, %out: f32):
                    %11 = arith.addf %in, %in_8 : f32
                    linalg.yield %11 : f32
                  }
                  scf.yield %arg13 : memref<64x64xf32>
                }
                %8 = affine.apply affine_map<(d0, d1, d2, d3, d4, d5) -> (d0 * 4096 + d1 * 64 + d2 * 2097152 + d3 * 32768 + d4 * 2097152 + d5 * 262144)>(%arg11, %arg6, %arg10, %arg9, %arg7, %arg8)
                %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%8], sizes: [64, 64], strides: [512, 1] {tmd.reuse = {spatial = [{depth = 2 : i64, iterator = "%arg8", mapped_to = "y", reuse_type = "no_reuse", volume = 0 : i64}, {depth = 3 : i64, iterator = "%arg9", mapped_to = "x", reuse_type = "no_reuse", volume = 0 : i64}], temporal = [{depth = 0 : i64, iterator = "%arg6", reuse_type = "no_reuse", volume = 0 : i64}, {depth = 1 : i64, iterator = "%arg7", reuse_type = "no_reuse", volume = 0 : i64}, {depth = 4 : i64, iterator = "%arg10", reuse_type = "no_reuse", volume = 0 : i64}, {depth = 5 : i64, iterator = "%arg11", reuse_type = "no_reuse", volume = 0 : i64}]}} : memref<*xf32> to memref<64x64xf32, strided<[512, 1], offset: ?>>
                memref.copy %7, %reinterpret_cast : memref<64x64xf32> to memref<64x64xf32, strided<[512, 1], offset: ?>>
              }
            }
          } {tmd.mapped_to = "x"}
        } {tmd.mapped_to = "y"}
      }
    }
    return
  }
  func.func @matmul_kernel__d0i0_d1i0__f10__c0mem_c1bx(%arg0: memref<*xf32> {tt.divisibility = 16 : i32}, %arg1: memref<*xf32> {tt.divisibility = 16 : i32}, %arg2: memref<*xf32> {tt.divisibility = 16 : i32}, %arg3: index, %arg4: index, %arg5: index) {
    affine.for %arg6 = 0 to affine_map<(d0, d1) -> (d1)>(%arg3, %arg4) {
      affine.for %arg7 = 0 to affine_map<(d0, d1) -> ((d0 ceildiv 8) ceildiv 8)>(%arg3, %arg4) {
        affine.parallel (%arg8) = (0) to (8) {
          affine.parallel (%arg9) = (0) to (8) {
            %cst = arith.constant 0.000000e+00 : f32
            %alloc = memref.alloc() {alignment = 64 : i64} : memref<64x64xf32>
            linalg.fill ins(%cst : f32) outs(%alloc : memref<64x64xf32>)
            %c0 = arith.constant 0 : index
            %c8 = arith.constant 8 : index
            %c1 = arith.constant 1 : index
            affine.for %arg10 = 0 to affine_map<(d0) -> (d0 ceildiv 64)>(%arg3) {
              affine.for %arg11 = 0 to affine_map<(d0) -> (d0 ceildiv 64)>(%arg4) {
                %alloc_0 = memref.alloc() {alignment = 64 : i64} : memref<64x64xf32>
                memref.copy %alloc, %alloc_0 : memref<64x64xf32> to memref<64x64xf32>
                %7 = scf.for %arg12 = %c0 to %c8 step %c1 iter_args(%arg13 = %alloc_0) -> (memref<64x64xf32>) {
                  %9 = affine.apply affine_map<(d0, d1, d2, d3, d4) -> (d0 * 64 + d1 * 2097152 + d2 * 32768 + d3 * 2097152 + d4 * 262144)>(%arg12, %arg10, %arg9, %arg7, %arg8)
                  %reinterpret_cast_1 = memref.reinterpret_cast %arg0 to offset: [%9], sizes: [64, 64], strides: [512, 1] {tmd.reuse = {sequential = [{depth = 6 : i64, iterator = "%arg12", reuse_type = "no_reuse", volume = 0 : i64}], spatial = [{depth = 2 : i64, iterator = "%arg8", mapped_to = "y", reuse_type = "no_reuse", volume = 0 : i64}, {depth = 3 : i64, iterator = "%arg9", mapped_to = "x", reuse_type = "no_reuse", volume = 0 : i64}], temporal = [{depth = 0 : i64, iterator = "%arg6", reuse_type = "total_reuse", volume = 16384 : i64}, {depth = 1 : i64, iterator = "%arg7", reuse_type = "no_reuse", volume = 0 : i64}, {depth = 4 : i64, iterator = "%arg10", reuse_type = "no_reuse", volume = 0 : i64}, {depth = 5 : i64, iterator = "%arg11", reuse_type = "total_reuse", volume = 16384 : i64}]}} : memref<*xf32> to memref<64x64xf32, strided<[512, 1], offset: ?>>
                  %alloc_2 = memref.alloc() {tmd.alloc = {local = true, memory_name = "L1", size = 16384 : i64}} : memref<64x64xf32>
                  memref.copy %reinterpret_cast_1, %alloc_2 {tmd.copy.choice = {kind = "mem", memory_name = "L1"}} : memref<64x64xf32, strided<[512, 1], offset: ?>> to memref<64x64xf32>
                  %10 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 32768 + d1 * 4096 + d2 * 64)>(%arg12, %arg11, %arg6)
                  %reinterpret_cast_3 = memref.reinterpret_cast %arg1 to offset: [%10], sizes: [64, 64], strides: [512, 1] {tmd.reuse = {sequential = [{depth = 6 : i64, iterator = "%arg12", reuse_type = "no_reuse", volume = 0 : i64}], spatial = [{depth = 2 : i64, iterator = "%arg8", mapped_to = "y", reuse_type = "total_reuse", volume = 16384 : i64}, {depth = 3 : i64, iterator = "%arg9", mapped_to = "x", reuse_type = "total_reuse", volume = 16384 : i64}], temporal = [{depth = 0 : i64, iterator = "%arg6", reuse_type = "no_reuse", volume = 0 : i64}, {depth = 1 : i64, iterator = "%arg7", reuse_type = "total_reuse", volume = 16384 : i64}, {depth = 4 : i64, iterator = "%arg10", reuse_type = "total_reuse", volume = 16384 : i64}, {depth = 5 : i64, iterator = "%arg11", reuse_type = "no_reuse", volume = 0 : i64}]}} : memref<*xf32> to memref<64x64xf32, strided<[512, 1], offset: ?>>
                  %alloc_4 = memref.alloc() {tmd.alloc = {local = true, memory_name = "L1", size = 16384 : i64}} : memref<64x64xf32>
                  memref.copy %reinterpret_cast_3, %alloc_4 {tmd.copy.choice = {dim = "x", interconnect_name = "horizontal_links", kind = "broadcast"}} : memref<64x64xf32, strided<[512, 1], offset: ?>> to memref<64x64xf32>
                  %alloc_5 = memref.alloc() {alignment = 64 : i64} : memref<64x64xf32>
                  memref.copy %alloc, %alloc_5 : memref<64x64xf32> to memref<64x64xf32>
                  %c32 = arith.constant 32 : index
                  %alloc_6 = memref.alloc() {tmd.alloc = {local = true, memory_name = "L1", size = 16384 : i64}} : memref<64x64xf32>
                  %alloc_7 = memref.alloc() {tmd.alloc = {local = true, memory_name = "L1", size = 16384 : i64}} : memref<64x64xf32>
                  memref.copy %alloc_2, %alloc_6 : memref<64x64xf32> to memref<64x64xf32>
                  memref.copy %alloc_4, %alloc_7 : memref<64x64xf32> to memref<64x64xf32>
                  affine.for %arg14 = 0 to 2 {
                    affine.for %arg15 = 0 to 2 {
                      %11 = affine.apply affine_map<(d0) -> (d0 * 32)>(%arg14)
                      %12 = affine.apply affine_map<(d0) -> (d0 * 32)>(%arg15)
                      %alloc_8 = memref.alloc() : memref<32x32xf32>
                      %subview = memref.subview %arg13[%11, %12] [%c32, %c32] [%c1, %c1] : memref<64x64xf32> to memref<?x?xf32, strided<[?, ?], offset: ?>>
                      memref.copy %subview, %alloc_8 : memref<?x?xf32, strided<[?, ?], offset: ?>> to memref<32x32xf32>
                      affine.for %arg16 = 0 to 2 {
                        %13 = affine.apply affine_map<(d0) -> (d0 * 32)>(%arg16)
                        %subview_9 = memref.subview %alloc_6[%11, %13] [%c32, %c32] [%c1, %c1] : memref<64x64xf32> to memref<?x?xf32, strided<[?, ?], offset: ?>>
                        %subview_10 = memref.subview %alloc_7[%13, %12] [%c32, %c32] [%c1, %c1] : memref<64x64xf32> to memref<?x?xf32, strided<[?, ?], offset: ?>>
                        linalg.matmul ins(%subview_9, %subview_10 : memref<?x?xf32, strided<[?, ?], offset: ?>>, memref<?x?xf32, strided<[?, ?], offset: ?>>) outs(%alloc_8 : memref<32x32xf32>)
                      }
                      memref.copy %alloc_8, %subview : memref<32x32xf32> to memref<?x?xf32, strided<[?, ?], offset: ?>>
                    }
                  }
                  linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%arg13, %alloc_5 : memref<64x64xf32>, memref<64x64xf32>) outs(%arg13 : memref<64x64xf32>) {
                  ^bb0(%in: f32, %in_8: f32, %out: f32):
                    %11 = arith.addf %in, %in_8 : f32
                    linalg.yield %11 : f32
                  }
                  scf.yield %arg13 : memref<64x64xf32>
                }
                %8 = affine.apply affine_map<(d0, d1, d2, d3, d4, d5) -> (d0 * 4096 + d1 * 64 + d2 * 2097152 + d3 * 32768 + d4 * 2097152 + d5 * 262144)>(%arg11, %arg6, %arg10, %arg9, %arg7, %arg8)
                %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%8], sizes: [64, 64], strides: [512, 1] {tmd.reuse = {spatial = [{depth = 2 : i64, iterator = "%arg8", mapped_to = "y", reuse_type = "no_reuse", volume = 0 : i64}, {depth = 3 : i64, iterator = "%arg9", mapped_to = "x", reuse_type = "no_reuse", volume = 0 : i64}], temporal = [{depth = 0 : i64, iterator = "%arg6", reuse_type = "no_reuse", volume = 0 : i64}, {depth = 1 : i64, iterator = "%arg7", reuse_type = "no_reuse", volume = 0 : i64}, {depth = 4 : i64, iterator = "%arg10", reuse_type = "no_reuse", volume = 0 : i64}, {depth = 5 : i64, iterator = "%arg11", reuse_type = "no_reuse", volume = 0 : i64}]}} : memref<*xf32> to memref<64x64xf32, strided<[512, 1], offset: ?>>
                memref.copy %7, %reinterpret_cast : memref<64x64xf32> to memref<64x64xf32, strided<[512, 1], offset: ?>>
              }
            }
          } {tmd.mapped_to = "x"}
        } {tmd.mapped_to = "y"}
      }
    }
    return
  }
  func.func @matmul_kernel__d0i0_d1i0__f10__c0mem_c1by(%arg0: memref<*xf32> {tt.divisibility = 16 : i32}, %arg1: memref<*xf32> {tt.divisibility = 16 : i32}, %arg2: memref<*xf32> {tt.divisibility = 16 : i32}, %arg3: index, %arg4: index, %arg5: index) {
    affine.for %arg6 = 0 to affine_map<(d0, d1) -> (d1)>(%arg3, %arg4) {
      affine.for %arg7 = 0 to affine_map<(d0, d1) -> ((d0 ceildiv 8) ceildiv 8)>(%arg3, %arg4) {
        affine.parallel (%arg8) = (0) to (8) {
          affine.parallel (%arg9) = (0) to (8) {
            %cst = arith.constant 0.000000e+00 : f32
            %alloc = memref.alloc() {alignment = 64 : i64} : memref<64x64xf32>
            linalg.fill ins(%cst : f32) outs(%alloc : memref<64x64xf32>)
            %c0 = arith.constant 0 : index
            %c8 = arith.constant 8 : index
            %c1 = arith.constant 1 : index
            affine.for %arg10 = 0 to affine_map<(d0) -> (d0 ceildiv 64)>(%arg3) {
              affine.for %arg11 = 0 to affine_map<(d0) -> (d0 ceildiv 64)>(%arg4) {
                %alloc_0 = memref.alloc() {alignment = 64 : i64} : memref<64x64xf32>
                memref.copy %alloc, %alloc_0 : memref<64x64xf32> to memref<64x64xf32>
                %7 = scf.for %arg12 = %c0 to %c8 step %c1 iter_args(%arg13 = %alloc_0) -> (memref<64x64xf32>) {
                  %9 = affine.apply affine_map<(d0, d1, d2, d3, d4) -> (d0 * 64 + d1 * 2097152 + d2 * 32768 + d3 * 2097152 + d4 * 262144)>(%arg12, %arg10, %arg9, %arg7, %arg8)
                  %reinterpret_cast_1 = memref.reinterpret_cast %arg0 to offset: [%9], sizes: [64, 64], strides: [512, 1] {tmd.reuse = {sequential = [{depth = 6 : i64, iterator = "%arg12", reuse_type = "no_reuse", volume = 0 : i64}], spatial = [{depth = 2 : i64, iterator = "%arg8", mapped_to = "y", reuse_type = "no_reuse", volume = 0 : i64}, {depth = 3 : i64, iterator = "%arg9", mapped_to = "x", reuse_type = "no_reuse", volume = 0 : i64}], temporal = [{depth = 0 : i64, iterator = "%arg6", reuse_type = "total_reuse", volume = 16384 : i64}, {depth = 1 : i64, iterator = "%arg7", reuse_type = "no_reuse", volume = 0 : i64}, {depth = 4 : i64, iterator = "%arg10", reuse_type = "no_reuse", volume = 0 : i64}, {depth = 5 : i64, iterator = "%arg11", reuse_type = "total_reuse", volume = 16384 : i64}]}} : memref<*xf32> to memref<64x64xf32, strided<[512, 1], offset: ?>>
                  %alloc_2 = memref.alloc() {tmd.alloc = {local = true, memory_name = "L1", size = 16384 : i64}} : memref<64x64xf32>
                  memref.copy %reinterpret_cast_1, %alloc_2 {tmd.copy.choice = {kind = "mem", memory_name = "L1"}} : memref<64x64xf32, strided<[512, 1], offset: ?>> to memref<64x64xf32>
                  %10 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 32768 + d1 * 4096 + d2 * 64)>(%arg12, %arg11, %arg6)
                  %reinterpret_cast_3 = memref.reinterpret_cast %arg1 to offset: [%10], sizes: [64, 64], strides: [512, 1] {tmd.reuse = {sequential = [{depth = 6 : i64, iterator = "%arg12", reuse_type = "no_reuse", volume = 0 : i64}], spatial = [{depth = 2 : i64, iterator = "%arg8", mapped_to = "y", reuse_type = "total_reuse", volume = 16384 : i64}, {depth = 3 : i64, iterator = "%arg9", mapped_to = "x", reuse_type = "total_reuse", volume = 16384 : i64}], temporal = [{depth = 0 : i64, iterator = "%arg6", reuse_type = "no_reuse", volume = 0 : i64}, {depth = 1 : i64, iterator = "%arg7", reuse_type = "total_reuse", volume = 16384 : i64}, {depth = 4 : i64, iterator = "%arg10", reuse_type = "total_reuse", volume = 16384 : i64}, {depth = 5 : i64, iterator = "%arg11", reuse_type = "no_reuse", volume = 0 : i64}]}} : memref<*xf32> to memref<64x64xf32, strided<[512, 1], offset: ?>>
                  %alloc_4 = memref.alloc() {tmd.alloc = {local = true, memory_name = "L1", size = 16384 : i64}} : memref<64x64xf32>
                  memref.copy %reinterpret_cast_3, %alloc_4 {tmd.copy.choice = {dim = "y", interconnect_name = "vertical_links", kind = "broadcast"}} : memref<64x64xf32, strided<[512, 1], offset: ?>> to memref<64x64xf32>
                  %alloc_5 = memref.alloc() {alignment = 64 : i64} : memref<64x64xf32>
                  memref.copy %alloc, %alloc_5 : memref<64x64xf32> to memref<64x64xf32>
                  %c32 = arith.constant 32 : index
                  %alloc_6 = memref.alloc() {tmd.alloc = {local = true, memory_name = "L1", size = 16384 : i64}} : memref<64x64xf32>
                  %alloc_7 = memref.alloc() {tmd.alloc = {local = true, memory_name = "L1", size = 16384 : i64}} : memref<64x64xf32>
                  memref.copy %alloc_2, %alloc_6 : memref<64x64xf32> to memref<64x64xf32>
                  memref.copy %alloc_4, %alloc_7 : memref<64x64xf32> to memref<64x64xf32>
                  affine.for %arg14 = 0 to 2 {
                    affine.for %arg15 = 0 to 2 {
                      %11 = affine.apply affine_map<(d0) -> (d0 * 32)>(%arg14)
                      %12 = affine.apply affine_map<(d0) -> (d0 * 32)>(%arg15)
                      %alloc_8 = memref.alloc() : memref<32x32xf32>
                      %subview = memref.subview %arg13[%11, %12] [%c32, %c32] [%c1, %c1] : memref<64x64xf32> to memref<?x?xf32, strided<[?, ?], offset: ?>>
                      memref.copy %subview, %alloc_8 : memref<?x?xf32, strided<[?, ?], offset: ?>> to memref<32x32xf32>
                      affine.for %arg16 = 0 to 2 {
                        %13 = affine.apply affine_map<(d0) -> (d0 * 32)>(%arg16)
                        %subview_9 = memref.subview %alloc_6[%11, %13] [%c32, %c32] [%c1, %c1] : memref<64x64xf32> to memref<?x?xf32, strided<[?, ?], offset: ?>>
                        %subview_10 = memref.subview %alloc_7[%13, %12] [%c32, %c32] [%c1, %c1] : memref<64x64xf32> to memref<?x?xf32, strided<[?, ?], offset: ?>>
                        linalg.matmul ins(%subview_9, %subview_10 : memref<?x?xf32, strided<[?, ?], offset: ?>>, memref<?x?xf32, strided<[?, ?], offset: ?>>) outs(%alloc_8 : memref<32x32xf32>)
                      }
                      memref.copy %alloc_8, %subview : memref<32x32xf32> to memref<?x?xf32, strided<[?, ?], offset: ?>>
                    }
                  }
                  linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%arg13, %alloc_5 : memref<64x64xf32>, memref<64x64xf32>) outs(%arg13 : memref<64x64xf32>) {
                  ^bb0(%in: f32, %in_8: f32, %out: f32):
                    %11 = arith.addf %in, %in_8 : f32
                    linalg.yield %11 : f32
                  }
                  scf.yield %arg13 : memref<64x64xf32>
                }
                %8 = affine.apply affine_map<(d0, d1, d2, d3, d4, d5) -> (d0 * 4096 + d1 * 64 + d2 * 2097152 + d3 * 32768 + d4 * 2097152 + d5 * 262144)>(%arg11, %arg6, %arg10, %arg9, %arg7, %arg8)
                %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%8], sizes: [64, 64], strides: [512, 1] {tmd.reuse = {spatial = [{depth = 2 : i64, iterator = "%arg8", mapped_to = "y", reuse_type = "no_reuse", volume = 0 : i64}, {depth = 3 : i64, iterator = "%arg9", mapped_to = "x", reuse_type = "no_reuse", volume = 0 : i64}], temporal = [{depth = 0 : i64, iterator = "%arg6", reuse_type = "no_reuse", volume = 0 : i64}, {depth = 1 : i64, iterator = "%arg7", reuse_type = "no_reuse", volume = 0 : i64}, {depth = 4 : i64, iterator = "%arg10", reuse_type = "no_reuse", volume = 0 : i64}, {depth = 5 : i64, iterator = "%arg11", reuse_type = "no_reuse", volume = 0 : i64}]}} : memref<*xf32> to memref<64x64xf32, strided<[512, 1], offset: ?>>
                memref.copy %7, %reinterpret_cast : memref<64x64xf32> to memref<64x64xf32, strided<[512, 1], offset: ?>>
              }
            }
          } {tmd.mapped_to = "x"}
        } {tmd.mapped_to = "y"}
      }
    }
    return
  }
  func.func @matmul_kernel__d0i0_d1i1__f01__c0mem_c1mem(%arg0: memref<*xf32> {tt.divisibility = 16 : i32}, %arg1: memref<*xf32> {tt.divisibility = 16 : i32}, %arg2: memref<*xf32> {tt.divisibility = 16 : i32}, %arg3: index, %arg4: index, %arg5: index) {
    affine.for %arg6 = 0 to affine_map<(d0, d1) -> (d0 ceildiv 8)>(%arg3, %arg4) {
      affine.for %arg7 = 0 to affine_map<(d0, d1) -> (d1 ceildiv 8)>(%arg3, %arg4) {
        affine.parallel (%arg8) = (0) to (8) {
          affine.parallel (%arg9) = (0) to (8) {
            %cst = arith.constant 0.000000e+00 : f32
            %alloc = memref.alloc() {alignment = 64 : i64} : memref<64x64xf32>
            linalg.fill ins(%cst : f32) outs(%alloc : memref<64x64xf32>)
            %c0 = arith.constant 0 : index
            %c8 = arith.constant 8 : index
            %c1 = arith.constant 1 : index
            affine.for %arg10 = 0 to affine_map<(d0) -> (d0 ceildiv 64)>(%arg3) {
              affine.for %arg11 = 0 to affine_map<(d0) -> (d0 ceildiv 64)>(%arg4) {
                %alloc_0 = memref.alloc() {alignment = 64 : i64} : memref<64x64xf32>
                memref.copy %alloc, %alloc_0 : memref<64x64xf32> to memref<64x64xf32>
                %7 = scf.for %arg12 = %c0 to %c8 step %c1 iter_args(%arg13 = %alloc_0) -> (memref<64x64xf32>) {
                  %9 = affine.apply affine_map<(d0, d1, d2, d3) -> (d0 * 64 + d1 * 2097152 + d2 * 262144 + d3 * 32768)>(%arg12, %arg10, %arg6, %arg9)
                  %reinterpret_cast_1 = memref.reinterpret_cast %arg0 to offset: [%9], sizes: [64, 64], strides: [512, 1] {tmd.reuse = {sequential = [{depth = 6 : i64, iterator = "%arg12", reuse_type = "no_reuse", volume = 0 : i64}], spatial = [{depth = 2 : i64, iterator = "%arg8", mapped_to = "y", reuse_type = "total_reuse", volume = 16384 : i64}, {depth = 3 : i64, iterator = "%arg9", mapped_to = "x", reuse_type = "no_reuse", volume = 0 : i64}], temporal = [{depth = 0 : i64, iterator = "%arg6", reuse_type = "no_reuse", volume = 0 : i64}, {depth = 1 : i64, iterator = "%arg7", reuse_type = "total_reuse", volume = 16384 : i64}, {depth = 4 : i64, iterator = "%arg10", reuse_type = "no_reuse", volume = 0 : i64}, {depth = 5 : i64, iterator = "%arg11", reuse_type = "total_reuse", volume = 16384 : i64}]}} : memref<*xf32> to memref<64x64xf32, strided<[512, 1], offset: ?>>
                  %alloc_2 = memref.alloc() {tmd.alloc = {local = true, memory_name = "L1", size = 16384 : i64}} : memref<64x64xf32>
                  memref.copy %reinterpret_cast_1, %alloc_2 {tmd.copy.choice = {kind = "mem", memory_name = "L1"}} : memref<64x64xf32, strided<[512, 1], offset: ?>> to memref<64x64xf32>
                  %10 = affine.apply affine_map<(d0, d1, d2, d3) -> (d0 * 32768 + d1 * 4096 + d2 * 512 + d3 * 64)>(%arg12, %arg11, %arg7, %arg8)
                  %reinterpret_cast_3 = memref.reinterpret_cast %arg1 to offset: [%10], sizes: [64, 64], strides: [512, 1] {tmd.reuse = {sequential = [{depth = 6 : i64, iterator = "%arg12", reuse_type = "no_reuse", volume = 0 : i64}], spatial = [{depth = 2 : i64, iterator = "%arg8", mapped_to = "y", reuse_type = "no_reuse", volume = 0 : i64}, {depth = 3 : i64, iterator = "%arg9", mapped_to = "x", reuse_type = "total_reuse", volume = 16384 : i64}], temporal = [{depth = 0 : i64, iterator = "%arg6", reuse_type = "total_reuse", volume = 16384 : i64}, {depth = 1 : i64, iterator = "%arg7", reuse_type = "no_reuse", volume = 0 : i64}, {depth = 4 : i64, iterator = "%arg10", reuse_type = "total_reuse", volume = 16384 : i64}, {depth = 5 : i64, iterator = "%arg11", reuse_type = "no_reuse", volume = 0 : i64}]}} : memref<*xf32> to memref<64x64xf32, strided<[512, 1], offset: ?>>
                  %alloc_4 = memref.alloc() {tmd.alloc = {local = true, memory_name = "L1", size = 16384 : i64}} : memref<64x64xf32>
                  memref.copy %reinterpret_cast_3, %alloc_4 {tmd.copy.choice = {kind = "mem", memory_name = "L1"}} : memref<64x64xf32, strided<[512, 1], offset: ?>> to memref<64x64xf32>
                  %alloc_5 = memref.alloc() {alignment = 64 : i64} : memref<64x64xf32>
                  memref.copy %alloc, %alloc_5 : memref<64x64xf32> to memref<64x64xf32>
                  %c32 = arith.constant 32 : index
                  %alloc_6 = memref.alloc() {tmd.alloc = {local = true, memory_name = "L1", size = 16384 : i64}} : memref<64x64xf32>
                  %alloc_7 = memref.alloc() {tmd.alloc = {local = true, memory_name = "L1", size = 16384 : i64}} : memref<64x64xf32>
                  memref.copy %alloc_2, %alloc_6 : memref<64x64xf32> to memref<64x64xf32>
                  memref.copy %alloc_4, %alloc_7 : memref<64x64xf32> to memref<64x64xf32>
                  affine.for %arg14 = 0 to 2 {
                    affine.for %arg15 = 0 to 2 {
                      %11 = affine.apply affine_map<(d0) -> (d0 * 32)>(%arg14)
                      %12 = affine.apply affine_map<(d0) -> (d0 * 32)>(%arg15)
                      %alloc_8 = memref.alloc() : memref<32x32xf32>
                      %subview = memref.subview %arg13[%11, %12] [%c32, %c32] [%c1, %c1] : memref<64x64xf32> to memref<?x?xf32, strided<[?, ?], offset: ?>>
                      memref.copy %subview, %alloc_8 : memref<?x?xf32, strided<[?, ?], offset: ?>> to memref<32x32xf32>
                      affine.for %arg16 = 0 to 2 {
                        %13 = affine.apply affine_map<(d0) -> (d0 * 32)>(%arg16)
                        %subview_9 = memref.subview %alloc_6[%11, %13] [%c32, %c32] [%c1, %c1] : memref<64x64xf32> to memref<?x?xf32, strided<[?, ?], offset: ?>>
                        %subview_10 = memref.subview %alloc_7[%13, %12] [%c32, %c32] [%c1, %c1] : memref<64x64xf32> to memref<?x?xf32, strided<[?, ?], offset: ?>>
                        linalg.matmul ins(%subview_9, %subview_10 : memref<?x?xf32, strided<[?, ?], offset: ?>>, memref<?x?xf32, strided<[?, ?], offset: ?>>) outs(%alloc_8 : memref<32x32xf32>)
                      }
                      memref.copy %alloc_8, %subview : memref<32x32xf32> to memref<?x?xf32, strided<[?, ?], offset: ?>>
                    }
                  }
                  linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%arg13, %alloc_5 : memref<64x64xf32>, memref<64x64xf32>) outs(%arg13 : memref<64x64xf32>) {
                  ^bb0(%in: f32, %in_8: f32, %out: f32):
                    %11 = arith.addf %in, %in_8 : f32
                    linalg.yield %11 : f32
                  }
                  scf.yield %arg13 : memref<64x64xf32>
                }
                %8 = affine.apply affine_map<(d0, d1, d2, d3, d4, d5) -> (d0 * 4096 + d1 * 512 + d2 * 64 + d3 * 2097152 + d4 * 262144 + d5 * 32768)>(%arg11, %arg7, %arg8, %arg10, %arg6, %arg9)
                %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%8], sizes: [64, 64], strides: [512, 1] {tmd.reuse = {spatial = [{depth = 2 : i64, iterator = "%arg8", mapped_to = "y", reuse_type = "no_reuse", volume = 0 : i64}, {depth = 3 : i64, iterator = "%arg9", mapped_to = "x", reuse_type = "no_reuse", volume = 0 : i64}], temporal = [{depth = 0 : i64, iterator = "%arg6", reuse_type = "no_reuse", volume = 0 : i64}, {depth = 1 : i64, iterator = "%arg7", reuse_type = "no_reuse", volume = 0 : i64}, {depth = 4 : i64, iterator = "%arg10", reuse_type = "no_reuse", volume = 0 : i64}, {depth = 5 : i64, iterator = "%arg11", reuse_type = "no_reuse", volume = 0 : i64}]}} : memref<*xf32> to memref<64x64xf32, strided<[512, 1], offset: ?>>
                memref.copy %7, %reinterpret_cast : memref<64x64xf32> to memref<64x64xf32, strided<[512, 1], offset: ?>>
              }
            }
          } {tmd.mapped_to = "x"}
        } {tmd.mapped_to = "y"}
      }
    }
    return
  }
  func.func @matmul_kernel__d0i0_d1i1__f01__c0by_c1mem(%arg0: memref<*xf32> {tt.divisibility = 16 : i32}, %arg1: memref<*xf32> {tt.divisibility = 16 : i32}, %arg2: memref<*xf32> {tt.divisibility = 16 : i32}, %arg3: index, %arg4: index, %arg5: index) {
    affine.for %arg6 = 0 to affine_map<(d0, d1) -> (d0 ceildiv 8)>(%arg3, %arg4) {
      affine.for %arg7 = 0 to affine_map<(d0, d1) -> (d1 ceildiv 8)>(%arg3, %arg4) {
        affine.parallel (%arg8) = (0) to (8) {
          affine.parallel (%arg9) = (0) to (8) {
            %cst = arith.constant 0.000000e+00 : f32
            %alloc = memref.alloc() {alignment = 64 : i64} : memref<64x64xf32>
            linalg.fill ins(%cst : f32) outs(%alloc : memref<64x64xf32>)
            %c0 = arith.constant 0 : index
            %c8 = arith.constant 8 : index
            %c1 = arith.constant 1 : index
            affine.for %arg10 = 0 to affine_map<(d0) -> (d0 ceildiv 64)>(%arg3) {
              affine.for %arg11 = 0 to affine_map<(d0) -> (d0 ceildiv 64)>(%arg4) {
                %alloc_0 = memref.alloc() {alignment = 64 : i64} : memref<64x64xf32>
                memref.copy %alloc, %alloc_0 : memref<64x64xf32> to memref<64x64xf32>
                %7 = scf.for %arg12 = %c0 to %c8 step %c1 iter_args(%arg13 = %alloc_0) -> (memref<64x64xf32>) {
                  %9 = affine.apply affine_map<(d0, d1, d2, d3) -> (d0 * 64 + d1 * 2097152 + d2 * 262144 + d3 * 32768)>(%arg12, %arg10, %arg6, %arg9)
                  %reinterpret_cast_1 = memref.reinterpret_cast %arg0 to offset: [%9], sizes: [64, 64], strides: [512, 1] {tmd.reuse = {sequential = [{depth = 6 : i64, iterator = "%arg12", reuse_type = "no_reuse", volume = 0 : i64}], spatial = [{depth = 2 : i64, iterator = "%arg8", mapped_to = "y", reuse_type = "total_reuse", volume = 16384 : i64}, {depth = 3 : i64, iterator = "%arg9", mapped_to = "x", reuse_type = "no_reuse", volume = 0 : i64}], temporal = [{depth = 0 : i64, iterator = "%arg6", reuse_type = "no_reuse", volume = 0 : i64}, {depth = 1 : i64, iterator = "%arg7", reuse_type = "total_reuse", volume = 16384 : i64}, {depth = 4 : i64, iterator = "%arg10", reuse_type = "no_reuse", volume = 0 : i64}, {depth = 5 : i64, iterator = "%arg11", reuse_type = "total_reuse", volume = 16384 : i64}]}} : memref<*xf32> to memref<64x64xf32, strided<[512, 1], offset: ?>>
                  %alloc_2 = memref.alloc() {tmd.alloc = {local = true, memory_name = "L1", size = 16384 : i64}} : memref<64x64xf32>
                  memref.copy %reinterpret_cast_1, %alloc_2 {tmd.copy.choice = {dim = "y", interconnect_name = "vertical_links", kind = "broadcast"}} : memref<64x64xf32, strided<[512, 1], offset: ?>> to memref<64x64xf32>
                  %10 = affine.apply affine_map<(d0, d1, d2, d3) -> (d0 * 32768 + d1 * 4096 + d2 * 512 + d3 * 64)>(%arg12, %arg11, %arg7, %arg8)
                  %reinterpret_cast_3 = memref.reinterpret_cast %arg1 to offset: [%10], sizes: [64, 64], strides: [512, 1] {tmd.reuse = {sequential = [{depth = 6 : i64, iterator = "%arg12", reuse_type = "no_reuse", volume = 0 : i64}], spatial = [{depth = 2 : i64, iterator = "%arg8", mapped_to = "y", reuse_type = "no_reuse", volume = 0 : i64}, {depth = 3 : i64, iterator = "%arg9", mapped_to = "x", reuse_type = "total_reuse", volume = 16384 : i64}], temporal = [{depth = 0 : i64, iterator = "%arg6", reuse_type = "total_reuse", volume = 16384 : i64}, {depth = 1 : i64, iterator = "%arg7", reuse_type = "no_reuse", volume = 0 : i64}, {depth = 4 : i64, iterator = "%arg10", reuse_type = "total_reuse", volume = 16384 : i64}, {depth = 5 : i64, iterator = "%arg11", reuse_type = "no_reuse", volume = 0 : i64}]}} : memref<*xf32> to memref<64x64xf32, strided<[512, 1], offset: ?>>
                  %alloc_4 = memref.alloc() {tmd.alloc = {local = true, memory_name = "L1", size = 16384 : i64}} : memref<64x64xf32>
                  memref.copy %reinterpret_cast_3, %alloc_4 {tmd.copy.choice = {kind = "mem", memory_name = "L1"}} : memref<64x64xf32, strided<[512, 1], offset: ?>> to memref<64x64xf32>
                  %alloc_5 = memref.alloc() {alignment = 64 : i64} : memref<64x64xf32>
                  memref.copy %alloc, %alloc_5 : memref<64x64xf32> to memref<64x64xf32>
                  %c32 = arith.constant 32 : index
                  %alloc_6 = memref.alloc() {tmd.alloc = {local = true, memory_name = "L1", size = 16384 : i64}} : memref<64x64xf32>
                  %alloc_7 = memref.alloc() {tmd.alloc = {local = true, memory_name = "L1", size = 16384 : i64}} : memref<64x64xf32>
                  memref.copy %alloc_2, %alloc_6 : memref<64x64xf32> to memref<64x64xf32>
                  memref.copy %alloc_4, %alloc_7 : memref<64x64xf32> to memref<64x64xf32>
                  affine.for %arg14 = 0 to 2 {
                    affine.for %arg15 = 0 to 2 {
                      %11 = affine.apply affine_map<(d0) -> (d0 * 32)>(%arg14)
                      %12 = affine.apply affine_map<(d0) -> (d0 * 32)>(%arg15)
                      %alloc_8 = memref.alloc() : memref<32x32xf32>
                      %subview = memref.subview %arg13[%11, %12] [%c32, %c32] [%c1, %c1] : memref<64x64xf32> to memref<?x?xf32, strided<[?, ?], offset: ?>>
                      memref.copy %subview, %alloc_8 : memref<?x?xf32, strided<[?, ?], offset: ?>> to memref<32x32xf32>
                      affine.for %arg16 = 0 to 2 {
                        %13 = affine.apply affine_map<(d0) -> (d0 * 32)>(%arg16)
                        %subview_9 = memref.subview %alloc_6[%11, %13] [%c32, %c32] [%c1, %c1] : memref<64x64xf32> to memref<?x?xf32, strided<[?, ?], offset: ?>>
                        %subview_10 = memref.subview %alloc_7[%13, %12] [%c32, %c32] [%c1, %c1] : memref<64x64xf32> to memref<?x?xf32, strided<[?, ?], offset: ?>>
                        linalg.matmul ins(%subview_9, %subview_10 : memref<?x?xf32, strided<[?, ?], offset: ?>>, memref<?x?xf32, strided<[?, ?], offset: ?>>) outs(%alloc_8 : memref<32x32xf32>)
                      }
                      memref.copy %alloc_8, %subview : memref<32x32xf32> to memref<?x?xf32, strided<[?, ?], offset: ?>>
                    }
                  }
                  linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%arg13, %alloc_5 : memref<64x64xf32>, memref<64x64xf32>) outs(%arg13 : memref<64x64xf32>) {
                  ^bb0(%in: f32, %in_8: f32, %out: f32):
                    %11 = arith.addf %in, %in_8 : f32
                    linalg.yield %11 : f32
                  }
                  scf.yield %arg13 : memref<64x64xf32>
                }
                %8 = affine.apply affine_map<(d0, d1, d2, d3, d4, d5) -> (d0 * 4096 + d1 * 512 + d2 * 64 + d3 * 2097152 + d4 * 262144 + d5 * 32768)>(%arg11, %arg7, %arg8, %arg10, %arg6, %arg9)
                %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%8], sizes: [64, 64], strides: [512, 1] {tmd.reuse = {spatial = [{depth = 2 : i64, iterator = "%arg8", mapped_to = "y", reuse_type = "no_reuse", volume = 0 : i64}, {depth = 3 : i64, iterator = "%arg9", mapped_to = "x", reuse_type = "no_reuse", volume = 0 : i64}], temporal = [{depth = 0 : i64, iterator = "%arg6", reuse_type = "no_reuse", volume = 0 : i64}, {depth = 1 : i64, iterator = "%arg7", reuse_type = "no_reuse", volume = 0 : i64}, {depth = 4 : i64, iterator = "%arg10", reuse_type = "no_reuse", volume = 0 : i64}, {depth = 5 : i64, iterator = "%arg11", reuse_type = "no_reuse", volume = 0 : i64}]}} : memref<*xf32> to memref<64x64xf32, strided<[512, 1], offset: ?>>
                memref.copy %7, %reinterpret_cast : memref<64x64xf32> to memref<64x64xf32, strided<[512, 1], offset: ?>>
              }
            }
          } {tmd.mapped_to = "x"}
        } {tmd.mapped_to = "y"}
      }
    }
    return
  }
  func.func @matmul_kernel__d0i0_d1i1__f01__c0mem_c1bx(%arg0: memref<*xf32> {tt.divisibility = 16 : i32}, %arg1: memref<*xf32> {tt.divisibility = 16 : i32}, %arg2: memref<*xf32> {tt.divisibility = 16 : i32}, %arg3: index, %arg4: index, %arg5: index) {
    affine.for %arg6 = 0 to affine_map<(d0, d1) -> (d0 ceildiv 8)>(%arg3, %arg4) {
      affine.for %arg7 = 0 to affine_map<(d0, d1) -> (d1 ceildiv 8)>(%arg3, %arg4) {
        affine.parallel (%arg8) = (0) to (8) {
          affine.parallel (%arg9) = (0) to (8) {
            %cst = arith.constant 0.000000e+00 : f32
            %alloc = memref.alloc() {alignment = 64 : i64} : memref<64x64xf32>
            linalg.fill ins(%cst : f32) outs(%alloc : memref<64x64xf32>)
            %c0 = arith.constant 0 : index
            %c8 = arith.constant 8 : index
            %c1 = arith.constant 1 : index
            affine.for %arg10 = 0 to affine_map<(d0) -> (d0 ceildiv 64)>(%arg3) {
              affine.for %arg11 = 0 to affine_map<(d0) -> (d0 ceildiv 64)>(%arg4) {
                %alloc_0 = memref.alloc() {alignment = 64 : i64} : memref<64x64xf32>
                memref.copy %alloc, %alloc_0 : memref<64x64xf32> to memref<64x64xf32>
                %7 = scf.for %arg12 = %c0 to %c8 step %c1 iter_args(%arg13 = %alloc_0) -> (memref<64x64xf32>) {
                  %9 = affine.apply affine_map<(d0, d1, d2, d3) -> (d0 * 64 + d1 * 2097152 + d2 * 262144 + d3 * 32768)>(%arg12, %arg10, %arg6, %arg9)
                  %reinterpret_cast_1 = memref.reinterpret_cast %arg0 to offset: [%9], sizes: [64, 64], strides: [512, 1] {tmd.reuse = {sequential = [{depth = 6 : i64, iterator = "%arg12", reuse_type = "no_reuse", volume = 0 : i64}], spatial = [{depth = 2 : i64, iterator = "%arg8", mapped_to = "y", reuse_type = "total_reuse", volume = 16384 : i64}, {depth = 3 : i64, iterator = "%arg9", mapped_to = "x", reuse_type = "no_reuse", volume = 0 : i64}], temporal = [{depth = 0 : i64, iterator = "%arg6", reuse_type = "no_reuse", volume = 0 : i64}, {depth = 1 : i64, iterator = "%arg7", reuse_type = "total_reuse", volume = 16384 : i64}, {depth = 4 : i64, iterator = "%arg10", reuse_type = "no_reuse", volume = 0 : i64}, {depth = 5 : i64, iterator = "%arg11", reuse_type = "total_reuse", volume = 16384 : i64}]}} : memref<*xf32> to memref<64x64xf32, strided<[512, 1], offset: ?>>
                  %alloc_2 = memref.alloc() {tmd.alloc = {local = true, memory_name = "L1", size = 16384 : i64}} : memref<64x64xf32>
                  memref.copy %reinterpret_cast_1, %alloc_2 {tmd.copy.choice = {kind = "mem", memory_name = "L1"}} : memref<64x64xf32, strided<[512, 1], offset: ?>> to memref<64x64xf32>
                  %10 = affine.apply affine_map<(d0, d1, d2, d3) -> (d0 * 32768 + d1 * 4096 + d2 * 512 + d3 * 64)>(%arg12, %arg11, %arg7, %arg8)
                  %reinterpret_cast_3 = memref.reinterpret_cast %arg1 to offset: [%10], sizes: [64, 64], strides: [512, 1] {tmd.reuse = {sequential = [{depth = 6 : i64, iterator = "%arg12", reuse_type = "no_reuse", volume = 0 : i64}], spatial = [{depth = 2 : i64, iterator = "%arg8", mapped_to = "y", reuse_type = "no_reuse", volume = 0 : i64}, {depth = 3 : i64, iterator = "%arg9", mapped_to = "x", reuse_type = "total_reuse", volume = 16384 : i64}], temporal = [{depth = 0 : i64, iterator = "%arg6", reuse_type = "total_reuse", volume = 16384 : i64}, {depth = 1 : i64, iterator = "%arg7", reuse_type = "no_reuse", volume = 0 : i64}, {depth = 4 : i64, iterator = "%arg10", reuse_type = "total_reuse", volume = 16384 : i64}, {depth = 5 : i64, iterator = "%arg11", reuse_type = "no_reuse", volume = 0 : i64}]}} : memref<*xf32> to memref<64x64xf32, strided<[512, 1], offset: ?>>
                  %alloc_4 = memref.alloc() {tmd.alloc = {local = true, memory_name = "L1", size = 16384 : i64}} : memref<64x64xf32>
                  memref.copy %reinterpret_cast_3, %alloc_4 {tmd.copy.choice = {dim = "x", interconnect_name = "horizontal_links", kind = "broadcast"}} : memref<64x64xf32, strided<[512, 1], offset: ?>> to memref<64x64xf32>
                  %alloc_5 = memref.alloc() {alignment = 64 : i64} : memref<64x64xf32>
                  memref.copy %alloc, %alloc_5 : memref<64x64xf32> to memref<64x64xf32>
                  %c32 = arith.constant 32 : index
                  %alloc_6 = memref.alloc() {tmd.alloc = {local = true, memory_name = "L1", size = 16384 : i64}} : memref<64x64xf32>
                  %alloc_7 = memref.alloc() {tmd.alloc = {local = true, memory_name = "L1", size = 16384 : i64}} : memref<64x64xf32>
                  memref.copy %alloc_2, %alloc_6 : memref<64x64xf32> to memref<64x64xf32>
                  memref.copy %alloc_4, %alloc_7 : memref<64x64xf32> to memref<64x64xf32>
                  affine.for %arg14 = 0 to 2 {
                    affine.for %arg15 = 0 to 2 {
                      %11 = affine.apply affine_map<(d0) -> (d0 * 32)>(%arg14)
                      %12 = affine.apply affine_map<(d0) -> (d0 * 32)>(%arg15)
                      %alloc_8 = memref.alloc() : memref<32x32xf32>
                      %subview = memref.subview %arg13[%11, %12] [%c32, %c32] [%c1, %c1] : memref<64x64xf32> to memref<?x?xf32, strided<[?, ?], offset: ?>>
                      memref.copy %subview, %alloc_8 : memref<?x?xf32, strided<[?, ?], offset: ?>> to memref<32x32xf32>
                      affine.for %arg16 = 0 to 2 {
                        %13 = affine.apply affine_map<(d0) -> (d0 * 32)>(%arg16)
                        %subview_9 = memref.subview %alloc_6[%11, %13] [%c32, %c32] [%c1, %c1] : memref<64x64xf32> to memref<?x?xf32, strided<[?, ?], offset: ?>>
                        %subview_10 = memref.subview %alloc_7[%13, %12] [%c32, %c32] [%c1, %c1] : memref<64x64xf32> to memref<?x?xf32, strided<[?, ?], offset: ?>>
                        linalg.matmul ins(%subview_9, %subview_10 : memref<?x?xf32, strided<[?, ?], offset: ?>>, memref<?x?xf32, strided<[?, ?], offset: ?>>) outs(%alloc_8 : memref<32x32xf32>)
                      }
                      memref.copy %alloc_8, %subview : memref<32x32xf32> to memref<?x?xf32, strided<[?, ?], offset: ?>>
                    }
                  }
                  linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%arg13, %alloc_5 : memref<64x64xf32>, memref<64x64xf32>) outs(%arg13 : memref<64x64xf32>) {
                  ^bb0(%in: f32, %in_8: f32, %out: f32):
                    %11 = arith.addf %in, %in_8 : f32
                    linalg.yield %11 : f32
                  }
                  scf.yield %arg13 : memref<64x64xf32>
                }
                %8 = affine.apply affine_map<(d0, d1, d2, d3, d4, d5) -> (d0 * 4096 + d1 * 512 + d2 * 64 + d3 * 2097152 + d4 * 262144 + d5 * 32768)>(%arg11, %arg7, %arg8, %arg10, %arg6, %arg9)
                %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%8], sizes: [64, 64], strides: [512, 1] {tmd.reuse = {spatial = [{depth = 2 : i64, iterator = "%arg8", mapped_to = "y", reuse_type = "no_reuse", volume = 0 : i64}, {depth = 3 : i64, iterator = "%arg9", mapped_to = "x", reuse_type = "no_reuse", volume = 0 : i64}], temporal = [{depth = 0 : i64, iterator = "%arg6", reuse_type = "no_reuse", volume = 0 : i64}, {depth = 1 : i64, iterator = "%arg7", reuse_type = "no_reuse", volume = 0 : i64}, {depth = 4 : i64, iterator = "%arg10", reuse_type = "no_reuse", volume = 0 : i64}, {depth = 5 : i64, iterator = "%arg11", reuse_type = "no_reuse", volume = 0 : i64}]}} : memref<*xf32> to memref<64x64xf32, strided<[512, 1], offset: ?>>
                memref.copy %7, %reinterpret_cast : memref<64x64xf32> to memref<64x64xf32, strided<[512, 1], offset: ?>>
              }
            }
          } {tmd.mapped_to = "x"}
        } {tmd.mapped_to = "y"}
      }
    }
    return
  }
  func.func @matmul_kernel__d0i0_d1i1__f01__c0by_c1bx(%arg0: memref<*xf32> {tt.divisibility = 16 : i32}, %arg1: memref<*xf32> {tt.divisibility = 16 : i32}, %arg2: memref<*xf32> {tt.divisibility = 16 : i32}, %arg3: index, %arg4: index, %arg5: index) {
    affine.for %arg6 = 0 to affine_map<(d0, d1) -> (d0 ceildiv 8)>(%arg3, %arg4) {
      affine.for %arg7 = 0 to affine_map<(d0, d1) -> (d1 ceildiv 8)>(%arg3, %arg4) {
        affine.parallel (%arg8) = (0) to (8) {
          affine.parallel (%arg9) = (0) to (8) {
            %cst = arith.constant 0.000000e+00 : f32
            %alloc = memref.alloc() {alignment = 64 : i64} : memref<64x64xf32>
            linalg.fill ins(%cst : f32) outs(%alloc : memref<64x64xf32>)
            %c0 = arith.constant 0 : index
            %c8 = arith.constant 8 : index
            %c1 = arith.constant 1 : index
            affine.for %arg10 = 0 to affine_map<(d0) -> (d0 ceildiv 64)>(%arg3) {
              affine.for %arg11 = 0 to affine_map<(d0) -> (d0 ceildiv 64)>(%arg4) {
                %alloc_0 = memref.alloc() {alignment = 64 : i64} : memref<64x64xf32>
                memref.copy %alloc, %alloc_0 : memref<64x64xf32> to memref<64x64xf32>
                %7 = scf.for %arg12 = %c0 to %c8 step %c1 iter_args(%arg13 = %alloc_0) -> (memref<64x64xf32>) {
                  %9 = affine.apply affine_map<(d0, d1, d2, d3) -> (d0 * 64 + d1 * 2097152 + d2 * 262144 + d3 * 32768)>(%arg12, %arg10, %arg6, %arg9)
                  %reinterpret_cast_1 = memref.reinterpret_cast %arg0 to offset: [%9], sizes: [64, 64], strides: [512, 1] {tmd.reuse = {sequential = [{depth = 6 : i64, iterator = "%arg12", reuse_type = "no_reuse", volume = 0 : i64}], spatial = [{depth = 2 : i64, iterator = "%arg8", mapped_to = "y", reuse_type = "total_reuse", volume = 16384 : i64}, {depth = 3 : i64, iterator = "%arg9", mapped_to = "x", reuse_type = "no_reuse", volume = 0 : i64}], temporal = [{depth = 0 : i64, iterator = "%arg6", reuse_type = "no_reuse", volume = 0 : i64}, {depth = 1 : i64, iterator = "%arg7", reuse_type = "total_reuse", volume = 16384 : i64}, {depth = 4 : i64, iterator = "%arg10", reuse_type = "no_reuse", volume = 0 : i64}, {depth = 5 : i64, iterator = "%arg11", reuse_type = "total_reuse", volume = 16384 : i64}]}} : memref<*xf32> to memref<64x64xf32, strided<[512, 1], offset: ?>>
                  %alloc_2 = memref.alloc() {tmd.alloc = {local = true, memory_name = "L1", size = 16384 : i64}} : memref<64x64xf32>
                  memref.copy %reinterpret_cast_1, %alloc_2 {tmd.copy.choice = {dim = "y", interconnect_name = "vertical_links", kind = "broadcast"}} : memref<64x64xf32, strided<[512, 1], offset: ?>> to memref<64x64xf32>
                  %10 = affine.apply affine_map<(d0, d1, d2, d3) -> (d0 * 32768 + d1 * 4096 + d2 * 512 + d3 * 64)>(%arg12, %arg11, %arg7, %arg8)
                  %reinterpret_cast_3 = memref.reinterpret_cast %arg1 to offset: [%10], sizes: [64, 64], strides: [512, 1] {tmd.reuse = {sequential = [{depth = 6 : i64, iterator = "%arg12", reuse_type = "no_reuse", volume = 0 : i64}], spatial = [{depth = 2 : i64, iterator = "%arg8", mapped_to = "y", reuse_type = "no_reuse", volume = 0 : i64}, {depth = 3 : i64, iterator = "%arg9", mapped_to = "x", reuse_type = "total_reuse", volume = 16384 : i64}], temporal = [{depth = 0 : i64, iterator = "%arg6", reuse_type = "total_reuse", volume = 16384 : i64}, {depth = 1 : i64, iterator = "%arg7", reuse_type = "no_reuse", volume = 0 : i64}, {depth = 4 : i64, iterator = "%arg10", reuse_type = "total_reuse", volume = 16384 : i64}, {depth = 5 : i64, iterator = "%arg11", reuse_type = "no_reuse", volume = 0 : i64}]}} : memref<*xf32> to memref<64x64xf32, strided<[512, 1], offset: ?>>
                  %alloc_4 = memref.alloc() {tmd.alloc = {local = true, memory_name = "L1", size = 16384 : i64}} : memref<64x64xf32>
                  memref.copy %reinterpret_cast_3, %alloc_4 {tmd.copy.choice = {dim = "x", interconnect_name = "horizontal_links", kind = "broadcast"}} : memref<64x64xf32, strided<[512, 1], offset: ?>> to memref<64x64xf32>
                  %alloc_5 = memref.alloc() {alignment = 64 : i64} : memref<64x64xf32>
                  memref.copy %alloc, %alloc_5 : memref<64x64xf32> to memref<64x64xf32>
                  %c32 = arith.constant 32 : index
                  %alloc_6 = memref.alloc() {tmd.alloc = {local = true, memory_name = "L1", size = 16384 : i64}} : memref<64x64xf32>
                  %alloc_7 = memref.alloc() {tmd.alloc = {local = true, memory_name = "L1", size = 16384 : i64}} : memref<64x64xf32>
                  memref.copy %alloc_2, %alloc_6 : memref<64x64xf32> to memref<64x64xf32>
                  memref.copy %alloc_4, %alloc_7 : memref<64x64xf32> to memref<64x64xf32>
                  affine.for %arg14 = 0 to 2 {
                    affine.for %arg15 = 0 to 2 {
                      %11 = affine.apply affine_map<(d0) -> (d0 * 32)>(%arg14)
                      %12 = affine.apply affine_map<(d0) -> (d0 * 32)>(%arg15)
                      %alloc_8 = memref.alloc() : memref<32x32xf32>
                      %subview = memref.subview %arg13[%11, %12] [%c32, %c32] [%c1, %c1] : memref<64x64xf32> to memref<?x?xf32, strided<[?, ?], offset: ?>>
                      memref.copy %subview, %alloc_8 : memref<?x?xf32, strided<[?, ?], offset: ?>> to memref<32x32xf32>
                      affine.for %arg16 = 0 to 2 {
                        %13 = affine.apply affine_map<(d0) -> (d0 * 32)>(%arg16)
                        %subview_9 = memref.subview %alloc_6[%11, %13] [%c32, %c32] [%c1, %c1] : memref<64x64xf32> to memref<?x?xf32, strided<[?, ?], offset: ?>>
                        %subview_10 = memref.subview %alloc_7[%13, %12] [%c32, %c32] [%c1, %c1] : memref<64x64xf32> to memref<?x?xf32, strided<[?, ?], offset: ?>>
                        linalg.matmul ins(%subview_9, %subview_10 : memref<?x?xf32, strided<[?, ?], offset: ?>>, memref<?x?xf32, strided<[?, ?], offset: ?>>) outs(%alloc_8 : memref<32x32xf32>)
                      }
                      memref.copy %alloc_8, %subview : memref<32x32xf32> to memref<?x?xf32, strided<[?, ?], offset: ?>>
                    }
                  }
                  linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%arg13, %alloc_5 : memref<64x64xf32>, memref<64x64xf32>) outs(%arg13 : memref<64x64xf32>) {
                  ^bb0(%in: f32, %in_8: f32, %out: f32):
                    %11 = arith.addf %in, %in_8 : f32
                    linalg.yield %11 : f32
                  }
                  scf.yield %arg13 : memref<64x64xf32>
                }
                %8 = affine.apply affine_map<(d0, d1, d2, d3, d4, d5) -> (d0 * 4096 + d1 * 512 + d2 * 64 + d3 * 2097152 + d4 * 262144 + d5 * 32768)>(%arg11, %arg7, %arg8, %arg10, %arg6, %arg9)
                %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%8], sizes: [64, 64], strides: [512, 1] {tmd.reuse = {spatial = [{depth = 2 : i64, iterator = "%arg8", mapped_to = "y", reuse_type = "no_reuse", volume = 0 : i64}, {depth = 3 : i64, iterator = "%arg9", mapped_to = "x", reuse_type = "no_reuse", volume = 0 : i64}], temporal = [{depth = 0 : i64, iterator = "%arg6", reuse_type = "no_reuse", volume = 0 : i64}, {depth = 1 : i64, iterator = "%arg7", reuse_type = "no_reuse", volume = 0 : i64}, {depth = 4 : i64, iterator = "%arg10", reuse_type = "no_reuse", volume = 0 : i64}, {depth = 5 : i64, iterator = "%arg11", reuse_type = "no_reuse", volume = 0 : i64}]}} : memref<*xf32> to memref<64x64xf32, strided<[512, 1], offset: ?>>
                memref.copy %7, %reinterpret_cast : memref<64x64xf32> to memref<64x64xf32, strided<[512, 1], offset: ?>>
              }
            }
          } {tmd.mapped_to = "x"}
        } {tmd.mapped_to = "y"}
      }
    }
    return
  }
  func.func @matmul_kernel__d0i0_d1i1__f10__c0mem_c1mem(%arg0: memref<*xf32> {tt.divisibility = 16 : i32}, %arg1: memref<*xf32> {tt.divisibility = 16 : i32}, %arg2: memref<*xf32> {tt.divisibility = 16 : i32}, %arg3: index, %arg4: index, %arg5: index) {
    affine.for %arg6 = 0 to affine_map<(d0, d1) -> (d1 ceildiv 8)>(%arg3, %arg4) {
      affine.for %arg7 = 0 to affine_map<(d0, d1) -> (d0 ceildiv 8)>(%arg3, %arg4) {
        affine.parallel (%arg8) = (0) to (8) {
          affine.parallel (%arg9) = (0) to (8) {
            %cst = arith.constant 0.000000e+00 : f32
            %alloc = memref.alloc() {alignment = 64 : i64} : memref<64x64xf32>
            linalg.fill ins(%cst : f32) outs(%alloc : memref<64x64xf32>)
            %c0 = arith.constant 0 : index
            %c8 = arith.constant 8 : index
            %c1 = arith.constant 1 : index
            affine.for %arg10 = 0 to affine_map<(d0) -> (d0 ceildiv 64)>(%arg3) {
              affine.for %arg11 = 0 to affine_map<(d0) -> (d0 ceildiv 64)>(%arg4) {
                %alloc_0 = memref.alloc() {alignment = 64 : i64} : memref<64x64xf32>
                memref.copy %alloc, %alloc_0 : memref<64x64xf32> to memref<64x64xf32>
                %7 = scf.for %arg12 = %c0 to %c8 step %c1 iter_args(%arg13 = %alloc_0) -> (memref<64x64xf32>) {
                  %9 = affine.apply affine_map<(d0, d1, d2, d3) -> (d0 * 64 + d1 * 2097152 + d2 * 262144 + d3 * 32768)>(%arg12, %arg10, %arg7, %arg9)
                  %reinterpret_cast_1 = memref.reinterpret_cast %arg0 to offset: [%9], sizes: [64, 64], strides: [512, 1] {tmd.reuse = {sequential = [{depth = 6 : i64, iterator = "%arg12", reuse_type = "no_reuse", volume = 0 : i64}], spatial = [{depth = 2 : i64, iterator = "%arg8", mapped_to = "y", reuse_type = "total_reuse", volume = 16384 : i64}, {depth = 3 : i64, iterator = "%arg9", mapped_to = "x", reuse_type = "no_reuse", volume = 0 : i64}], temporal = [{depth = 0 : i64, iterator = "%arg6", reuse_type = "total_reuse", volume = 16384 : i64}, {depth = 1 : i64, iterator = "%arg7", reuse_type = "no_reuse", volume = 0 : i64}, {depth = 4 : i64, iterator = "%arg10", reuse_type = "no_reuse", volume = 0 : i64}, {depth = 5 : i64, iterator = "%arg11", reuse_type = "total_reuse", volume = 16384 : i64}]}} : memref<*xf32> to memref<64x64xf32, strided<[512, 1], offset: ?>>
                  %alloc_2 = memref.alloc() {tmd.alloc = {local = true, memory_name = "L1", size = 16384 : i64}} : memref<64x64xf32>
                  memref.copy %reinterpret_cast_1, %alloc_2 {tmd.copy.choice = {kind = "mem", memory_name = "L1"}} : memref<64x64xf32, strided<[512, 1], offset: ?>> to memref<64x64xf32>
                  %10 = affine.apply affine_map<(d0, d1, d2, d3) -> (d0 * 32768 + d1 * 4096 + d2 * 512 + d3 * 64)>(%arg12, %arg11, %arg6, %arg8)
                  %reinterpret_cast_3 = memref.reinterpret_cast %arg1 to offset: [%10], sizes: [64, 64], strides: [512, 1] {tmd.reuse = {sequential = [{depth = 6 : i64, iterator = "%arg12", reuse_type = "no_reuse", volume = 0 : i64}], spatial = [{depth = 2 : i64, iterator = "%arg8", mapped_to = "y", reuse_type = "no_reuse", volume = 0 : i64}, {depth = 3 : i64, iterator = "%arg9", mapped_to = "x", reuse_type = "total_reuse", volume = 16384 : i64}], temporal = [{depth = 0 : i64, iterator = "%arg6", reuse_type = "no_reuse", volume = 0 : i64}, {depth = 1 : i64, iterator = "%arg7", reuse_type = "total_reuse", volume = 16384 : i64}, {depth = 4 : i64, iterator = "%arg10", reuse_type = "total_reuse", volume = 16384 : i64}, {depth = 5 : i64, iterator = "%arg11", reuse_type = "no_reuse", volume = 0 : i64}]}} : memref<*xf32> to memref<64x64xf32, strided<[512, 1], offset: ?>>
                  %alloc_4 = memref.alloc() {tmd.alloc = {local = true, memory_name = "L1", size = 16384 : i64}} : memref<64x64xf32>
                  memref.copy %reinterpret_cast_3, %alloc_4 {tmd.copy.choice = {kind = "mem", memory_name = "L1"}} : memref<64x64xf32, strided<[512, 1], offset: ?>> to memref<64x64xf32>
                  %alloc_5 = memref.alloc() {alignment = 64 : i64} : memref<64x64xf32>
                  memref.copy %alloc, %alloc_5 : memref<64x64xf32> to memref<64x64xf32>
                  %c32 = arith.constant 32 : index
                  %alloc_6 = memref.alloc() {tmd.alloc = {local = true, memory_name = "L1", size = 16384 : i64}} : memref<64x64xf32>
                  %alloc_7 = memref.alloc() {tmd.alloc = {local = true, memory_name = "L1", size = 16384 : i64}} : memref<64x64xf32>
                  memref.copy %alloc_2, %alloc_6 : memref<64x64xf32> to memref<64x64xf32>
                  memref.copy %alloc_4, %alloc_7 : memref<64x64xf32> to memref<64x64xf32>
                  affine.for %arg14 = 0 to 2 {
                    affine.for %arg15 = 0 to 2 {
                      %11 = affine.apply affine_map<(d0) -> (d0 * 32)>(%arg14)
                      %12 = affine.apply affine_map<(d0) -> (d0 * 32)>(%arg15)
                      %alloc_8 = memref.alloc() : memref<32x32xf32>
                      %subview = memref.subview %arg13[%11, %12] [%c32, %c32] [%c1, %c1] : memref<64x64xf32> to memref<?x?xf32, strided<[?, ?], offset: ?>>
                      memref.copy %subview, %alloc_8 : memref<?x?xf32, strided<[?, ?], offset: ?>> to memref<32x32xf32>
                      affine.for %arg16 = 0 to 2 {
                        %13 = affine.apply affine_map<(d0) -> (d0 * 32)>(%arg16)
                        %subview_9 = memref.subview %alloc_6[%11, %13] [%c32, %c32] [%c1, %c1] : memref<64x64xf32> to memref<?x?xf32, strided<[?, ?], offset: ?>>
                        %subview_10 = memref.subview %alloc_7[%13, %12] [%c32, %c32] [%c1, %c1] : memref<64x64xf32> to memref<?x?xf32, strided<[?, ?], offset: ?>>
                        linalg.matmul ins(%subview_9, %subview_10 : memref<?x?xf32, strided<[?, ?], offset: ?>>, memref<?x?xf32, strided<[?, ?], offset: ?>>) outs(%alloc_8 : memref<32x32xf32>)
                      }
                      memref.copy %alloc_8, %subview : memref<32x32xf32> to memref<?x?xf32, strided<[?, ?], offset: ?>>
                    }
                  }
                  linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%arg13, %alloc_5 : memref<64x64xf32>, memref<64x64xf32>) outs(%arg13 : memref<64x64xf32>) {
                  ^bb0(%in: f32, %in_8: f32, %out: f32):
                    %11 = arith.addf %in, %in_8 : f32
                    linalg.yield %11 : f32
                  }
                  scf.yield %arg13 : memref<64x64xf32>
                }
                %8 = affine.apply affine_map<(d0, d1, d2, d3, d4, d5) -> (d0 * 4096 + d1 * 512 + d2 * 64 + d3 * 2097152 + d4 * 262144 + d5 * 32768)>(%arg11, %arg6, %arg8, %arg10, %arg7, %arg9)
                %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%8], sizes: [64, 64], strides: [512, 1] {tmd.reuse = {spatial = [{depth = 2 : i64, iterator = "%arg8", mapped_to = "y", reuse_type = "no_reuse", volume = 0 : i64}, {depth = 3 : i64, iterator = "%arg9", mapped_to = "x", reuse_type = "no_reuse", volume = 0 : i64}], temporal = [{depth = 0 : i64, iterator = "%arg6", reuse_type = "no_reuse", volume = 0 : i64}, {depth = 1 : i64, iterator = "%arg7", reuse_type = "no_reuse", volume = 0 : i64}, {depth = 4 : i64, iterator = "%arg10", reuse_type = "no_reuse", volume = 0 : i64}, {depth = 5 : i64, iterator = "%arg11", reuse_type = "no_reuse", volume = 0 : i64}]}} : memref<*xf32> to memref<64x64xf32, strided<[512, 1], offset: ?>>
                memref.copy %7, %reinterpret_cast : memref<64x64xf32> to memref<64x64xf32, strided<[512, 1], offset: ?>>
              }
            }
          } {tmd.mapped_to = "x"}
        } {tmd.mapped_to = "y"}
      }
    }
    return
  }
  func.func @matmul_kernel__d0i0_d1i1__f10__c0by_c1mem(%arg0: memref<*xf32> {tt.divisibility = 16 : i32}, %arg1: memref<*xf32> {tt.divisibility = 16 : i32}, %arg2: memref<*xf32> {tt.divisibility = 16 : i32}, %arg3: index, %arg4: index, %arg5: index) {
    affine.for %arg6 = 0 to affine_map<(d0, d1) -> (d1 ceildiv 8)>(%arg3, %arg4) {
      affine.for %arg7 = 0 to affine_map<(d0, d1) -> (d0 ceildiv 8)>(%arg3, %arg4) {
        affine.parallel (%arg8) = (0) to (8) {
          affine.parallel (%arg9) = (0) to (8) {
            %cst = arith.constant 0.000000e+00 : f32
            %alloc = memref.alloc() {alignment = 64 : i64} : memref<64x64xf32>
            linalg.fill ins(%cst : f32) outs(%alloc : memref<64x64xf32>)
            %c0 = arith.constant 0 : index
            %c8 = arith.constant 8 : index
            %c1 = arith.constant 1 : index
            affine.for %arg10 = 0 to affine_map<(d0) -> (d0 ceildiv 64)>(%arg3) {
              affine.for %arg11 = 0 to affine_map<(d0) -> (d0 ceildiv 64)>(%arg4) {
                %alloc_0 = memref.alloc() {alignment = 64 : i64} : memref<64x64xf32>
                memref.copy %alloc, %alloc_0 : memref<64x64xf32> to memref<64x64xf32>
                %7 = scf.for %arg12 = %c0 to %c8 step %c1 iter_args(%arg13 = %alloc_0) -> (memref<64x64xf32>) {
                  %9 = affine.apply affine_map<(d0, d1, d2, d3) -> (d0 * 64 + d1 * 2097152 + d2 * 262144 + d3 * 32768)>(%arg12, %arg10, %arg7, %arg9)
                  %reinterpret_cast_1 = memref.reinterpret_cast %arg0 to offset: [%9], sizes: [64, 64], strides: [512, 1] {tmd.reuse = {sequential = [{depth = 6 : i64, iterator = "%arg12", reuse_type = "no_reuse", volume = 0 : i64}], spatial = [{depth = 2 : i64, iterator = "%arg8", mapped_to = "y", reuse_type = "total_reuse", volume = 16384 : i64}, {depth = 3 : i64, iterator = "%arg9", mapped_to = "x", reuse_type = "no_reuse", volume = 0 : i64}], temporal = [{depth = 0 : i64, iterator = "%arg6", reuse_type = "total_reuse", volume = 16384 : i64}, {depth = 1 : i64, iterator = "%arg7", reuse_type = "no_reuse", volume = 0 : i64}, {depth = 4 : i64, iterator = "%arg10", reuse_type = "no_reuse", volume = 0 : i64}, {depth = 5 : i64, iterator = "%arg11", reuse_type = "total_reuse", volume = 16384 : i64}]}} : memref<*xf32> to memref<64x64xf32, strided<[512, 1], offset: ?>>
                  %alloc_2 = memref.alloc() {tmd.alloc = {local = true, memory_name = "L1", size = 16384 : i64}} : memref<64x64xf32>
                  memref.copy %reinterpret_cast_1, %alloc_2 {tmd.copy.choice = {dim = "y", interconnect_name = "vertical_links", kind = "broadcast"}} : memref<64x64xf32, strided<[512, 1], offset: ?>> to memref<64x64xf32>
                  %10 = affine.apply affine_map<(d0, d1, d2, d3) -> (d0 * 32768 + d1 * 4096 + d2 * 512 + d3 * 64)>(%arg12, %arg11, %arg6, %arg8)
                  %reinterpret_cast_3 = memref.reinterpret_cast %arg1 to offset: [%10], sizes: [64, 64], strides: [512, 1] {tmd.reuse = {sequential = [{depth = 6 : i64, iterator = "%arg12", reuse_type = "no_reuse", volume = 0 : i64}], spatial = [{depth = 2 : i64, iterator = "%arg8", mapped_to = "y", reuse_type = "no_reuse", volume = 0 : i64}, {depth = 3 : i64, iterator = "%arg9", mapped_to = "x", reuse_type = "total_reuse", volume = 16384 : i64}], temporal = [{depth = 0 : i64, iterator = "%arg6", reuse_type = "no_reuse", volume = 0 : i64}, {depth = 1 : i64, iterator = "%arg7", reuse_type = "total_reuse", volume = 16384 : i64}, {depth = 4 : i64, iterator = "%arg10", reuse_type = "total_reuse", volume = 16384 : i64}, {depth = 5 : i64, iterator = "%arg11", reuse_type = "no_reuse", volume = 0 : i64}]}} : memref<*xf32> to memref<64x64xf32, strided<[512, 1], offset: ?>>
                  %alloc_4 = memref.alloc() {tmd.alloc = {local = true, memory_name = "L1", size = 16384 : i64}} : memref<64x64xf32>
                  memref.copy %reinterpret_cast_3, %alloc_4 {tmd.copy.choice = {kind = "mem", memory_name = "L1"}} : memref<64x64xf32, strided<[512, 1], offset: ?>> to memref<64x64xf32>
                  %alloc_5 = memref.alloc() {alignment = 64 : i64} : memref<64x64xf32>
                  memref.copy %alloc, %alloc_5 : memref<64x64xf32> to memref<64x64xf32>
                  %c32 = arith.constant 32 : index
                  %alloc_6 = memref.alloc() {tmd.alloc = {local = true, memory_name = "L1", size = 16384 : i64}} : memref<64x64xf32>
                  %alloc_7 = memref.alloc() {tmd.alloc = {local = true, memory_name = "L1", size = 16384 : i64}} : memref<64x64xf32>
                  memref.copy %alloc_2, %alloc_6 : memref<64x64xf32> to memref<64x64xf32>
                  memref.copy %alloc_4, %alloc_7 : memref<64x64xf32> to memref<64x64xf32>
                  affine.for %arg14 = 0 to 2 {
                    affine.for %arg15 = 0 to 2 {
                      %11 = affine.apply affine_map<(d0) -> (d0 * 32)>(%arg14)
                      %12 = affine.apply affine_map<(d0) -> (d0 * 32)>(%arg15)
                      %alloc_8 = memref.alloc() : memref<32x32xf32>
                      %subview = memref.subview %arg13[%11, %12] [%c32, %c32] [%c1, %c1] : memref<64x64xf32> to memref<?x?xf32, strided<[?, ?], offset: ?>>
                      memref.copy %subview, %alloc_8 : memref<?x?xf32, strided<[?, ?], offset: ?>> to memref<32x32xf32>
                      affine.for %arg16 = 0 to 2 {
                        %13 = affine.apply affine_map<(d0) -> (d0 * 32)>(%arg16)
                        %subview_9 = memref.subview %alloc_6[%11, %13] [%c32, %c32] [%c1, %c1] : memref<64x64xf32> to memref<?x?xf32, strided<[?, ?], offset: ?>>
                        %subview_10 = memref.subview %alloc_7[%13, %12] [%c32, %c32] [%c1, %c1] : memref<64x64xf32> to memref<?x?xf32, strided<[?, ?], offset: ?>>
                        linalg.matmul ins(%subview_9, %subview_10 : memref<?x?xf32, strided<[?, ?], offset: ?>>, memref<?x?xf32, strided<[?, ?], offset: ?>>) outs(%alloc_8 : memref<32x32xf32>)
                      }
                      memref.copy %alloc_8, %subview : memref<32x32xf32> to memref<?x?xf32, strided<[?, ?], offset: ?>>
                    }
                  }
                  linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%arg13, %alloc_5 : memref<64x64xf32>, memref<64x64xf32>) outs(%arg13 : memref<64x64xf32>) {
                  ^bb0(%in: f32, %in_8: f32, %out: f32):
                    %11 = arith.addf %in, %in_8 : f32
                    linalg.yield %11 : f32
                  }
                  scf.yield %arg13 : memref<64x64xf32>
                }
                %8 = affine.apply affine_map<(d0, d1, d2, d3, d4, d5) -> (d0 * 4096 + d1 * 512 + d2 * 64 + d3 * 2097152 + d4 * 262144 + d5 * 32768)>(%arg11, %arg6, %arg8, %arg10, %arg7, %arg9)
                %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%8], sizes: [64, 64], strides: [512, 1] {tmd.reuse = {spatial = [{depth = 2 : i64, iterator = "%arg8", mapped_to = "y", reuse_type = "no_reuse", volume = 0 : i64}, {depth = 3 : i64, iterator = "%arg9", mapped_to = "x", reuse_type = "no_reuse", volume = 0 : i64}], temporal = [{depth = 0 : i64, iterator = "%arg6", reuse_type = "no_reuse", volume = 0 : i64}, {depth = 1 : i64, iterator = "%arg7", reuse_type = "no_reuse", volume = 0 : i64}, {depth = 4 : i64, iterator = "%arg10", reuse_type = "no_reuse", volume = 0 : i64}, {depth = 5 : i64, iterator = "%arg11", reuse_type = "no_reuse", volume = 0 : i64}]}} : memref<*xf32> to memref<64x64xf32, strided<[512, 1], offset: ?>>
                memref.copy %7, %reinterpret_cast : memref<64x64xf32> to memref<64x64xf32, strided<[512, 1], offset: ?>>
              }
            }
          } {tmd.mapped_to = "x"}
        } {tmd.mapped_to = "y"}
      }
    }
    return
  }
  func.func @matmul_kernel__d0i0_d1i1__f10__c0mem_c1bx(%arg0: memref<*xf32> {tt.divisibility = 16 : i32}, %arg1: memref<*xf32> {tt.divisibility = 16 : i32}, %arg2: memref<*xf32> {tt.divisibility = 16 : i32}, %arg3: index, %arg4: index, %arg5: index) {
    affine.for %arg6 = 0 to affine_map<(d0, d1) -> (d1 ceildiv 8)>(%arg3, %arg4) {
      affine.for %arg7 = 0 to affine_map<(d0, d1) -> (d0 ceildiv 8)>(%arg3, %arg4) {
        affine.parallel (%arg8) = (0) to (8) {
          affine.parallel (%arg9) = (0) to (8) {
            %cst = arith.constant 0.000000e+00 : f32
            %alloc = memref.alloc() {alignment = 64 : i64} : memref<64x64xf32>
            linalg.fill ins(%cst : f32) outs(%alloc : memref<64x64xf32>)
            %c0 = arith.constant 0 : index
            %c8 = arith.constant 8 : index
            %c1 = arith.constant 1 : index
            affine.for %arg10 = 0 to affine_map<(d0) -> (d0 ceildiv 64)>(%arg3) {
              affine.for %arg11 = 0 to affine_map<(d0) -> (d0 ceildiv 64)>(%arg4) {
                %alloc_0 = memref.alloc() {alignment = 64 : i64} : memref<64x64xf32>
                memref.copy %alloc, %alloc_0 : memref<64x64xf32> to memref<64x64xf32>
                %7 = scf.for %arg12 = %c0 to %c8 step %c1 iter_args(%arg13 = %alloc_0) -> (memref<64x64xf32>) {
                  %9 = affine.apply affine_map<(d0, d1, d2, d3) -> (d0 * 64 + d1 * 2097152 + d2 * 262144 + d3 * 32768)>(%arg12, %arg10, %arg7, %arg9)
                  %reinterpret_cast_1 = memref.reinterpret_cast %arg0 to offset: [%9], sizes: [64, 64], strides: [512, 1] {tmd.reuse = {sequential = [{depth = 6 : i64, iterator = "%arg12", reuse_type = "no_reuse", volume = 0 : i64}], spatial = [{depth = 2 : i64, iterator = "%arg8", mapped_to = "y", reuse_type = "total_reuse", volume = 16384 : i64}, {depth = 3 : i64, iterator = "%arg9", mapped_to = "x", reuse_type = "no_reuse", volume = 0 : i64}], temporal = [{depth = 0 : i64, iterator = "%arg6", reuse_type = "total_reuse", volume = 16384 : i64}, {depth = 1 : i64, iterator = "%arg7", reuse_type = "no_reuse", volume = 0 : i64}, {depth = 4 : i64, iterator = "%arg10", reuse_type = "no_reuse", volume = 0 : i64}, {depth = 5 : i64, iterator = "%arg11", reuse_type = "total_reuse", volume = 16384 : i64}]}} : memref<*xf32> to memref<64x64xf32, strided<[512, 1], offset: ?>>
                  %alloc_2 = memref.alloc() {tmd.alloc = {local = true, memory_name = "L1", size = 16384 : i64}} : memref<64x64xf32>
                  memref.copy %reinterpret_cast_1, %alloc_2 {tmd.copy.choice = {kind = "mem", memory_name = "L1"}} : memref<64x64xf32, strided<[512, 1], offset: ?>> to memref<64x64xf32>
                  %10 = affine.apply affine_map<(d0, d1, d2, d3) -> (d0 * 32768 + d1 * 4096 + d2 * 512 + d3 * 64)>(%arg12, %arg11, %arg6, %arg8)
                  %reinterpret_cast_3 = memref.reinterpret_cast %arg1 to offset: [%10], sizes: [64, 64], strides: [512, 1] {tmd.reuse = {sequential = [{depth = 6 : i64, iterator = "%arg12", reuse_type = "no_reuse", volume = 0 : i64}], spatial = [{depth = 2 : i64, iterator = "%arg8", mapped_to = "y", reuse_type = "no_reuse", volume = 0 : i64}, {depth = 3 : i64, iterator = "%arg9", mapped_to = "x", reuse_type = "total_reuse", volume = 16384 : i64}], temporal = [{depth = 0 : i64, iterator = "%arg6", reuse_type = "no_reuse", volume = 0 : i64}, {depth = 1 : i64, iterator = "%arg7", reuse_type = "total_reuse", volume = 16384 : i64}, {depth = 4 : i64, iterator = "%arg10", reuse_type = "total_reuse", volume = 16384 : i64}, {depth = 5 : i64, iterator = "%arg11", reuse_type = "no_reuse", volume = 0 : i64}]}} : memref<*xf32> to memref<64x64xf32, strided<[512, 1], offset: ?>>
                  %alloc_4 = memref.alloc() {tmd.alloc = {local = true, memory_name = "L1", size = 16384 : i64}} : memref<64x64xf32>
                  memref.copy %reinterpret_cast_3, %alloc_4 {tmd.copy.choice = {dim = "x", interconnect_name = "horizontal_links", kind = "broadcast"}} : memref<64x64xf32, strided<[512, 1], offset: ?>> to memref<64x64xf32>
                  %alloc_5 = memref.alloc() {alignment = 64 : i64} : memref<64x64xf32>
                  memref.copy %alloc, %alloc_5 : memref<64x64xf32> to memref<64x64xf32>
                  %c32 = arith.constant 32 : index
                  %alloc_6 = memref.alloc() {tmd.alloc = {local = true, memory_name = "L1", size = 16384 : i64}} : memref<64x64xf32>
                  %alloc_7 = memref.alloc() {tmd.alloc = {local = true, memory_name = "L1", size = 16384 : i64}} : memref<64x64xf32>
                  memref.copy %alloc_2, %alloc_6 : memref<64x64xf32> to memref<64x64xf32>
                  memref.copy %alloc_4, %alloc_7 : memref<64x64xf32> to memref<64x64xf32>
                  affine.for %arg14 = 0 to 2 {
                    affine.for %arg15 = 0 to 2 {
                      %11 = affine.apply affine_map<(d0) -> (d0 * 32)>(%arg14)
                      %12 = affine.apply affine_map<(d0) -> (d0 * 32)>(%arg15)
                      %alloc_8 = memref.alloc() : memref<32x32xf32>
                      %subview = memref.subview %arg13[%11, %12] [%c32, %c32] [%c1, %c1] : memref<64x64xf32> to memref<?x?xf32, strided<[?, ?], offset: ?>>
                      memref.copy %subview, %alloc_8 : memref<?x?xf32, strided<[?, ?], offset: ?>> to memref<32x32xf32>
                      affine.for %arg16 = 0 to 2 {
                        %13 = affine.apply affine_map<(d0) -> (d0 * 32)>(%arg16)
                        %subview_9 = memref.subview %alloc_6[%11, %13] [%c32, %c32] [%c1, %c1] : memref<64x64xf32> to memref<?x?xf32, strided<[?, ?], offset: ?>>
                        %subview_10 = memref.subview %alloc_7[%13, %12] [%c32, %c32] [%c1, %c1] : memref<64x64xf32> to memref<?x?xf32, strided<[?, ?], offset: ?>>
                        linalg.matmul ins(%subview_9, %subview_10 : memref<?x?xf32, strided<[?, ?], offset: ?>>, memref<?x?xf32, strided<[?, ?], offset: ?>>) outs(%alloc_8 : memref<32x32xf32>)
                      }
                      memref.copy %alloc_8, %subview : memref<32x32xf32> to memref<?x?xf32, strided<[?, ?], offset: ?>>
                    }
                  }
                  linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%arg13, %alloc_5 : memref<64x64xf32>, memref<64x64xf32>) outs(%arg13 : memref<64x64xf32>) {
                  ^bb0(%in: f32, %in_8: f32, %out: f32):
                    %11 = arith.addf %in, %in_8 : f32
                    linalg.yield %11 : f32
                  }
                  scf.yield %arg13 : memref<64x64xf32>
                }
                %8 = affine.apply affine_map<(d0, d1, d2, d3, d4, d5) -> (d0 * 4096 + d1 * 512 + d2 * 64 + d3 * 2097152 + d4 * 262144 + d5 * 32768)>(%arg11, %arg6, %arg8, %arg10, %arg7, %arg9)
                %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%8], sizes: [64, 64], strides: [512, 1] {tmd.reuse = {spatial = [{depth = 2 : i64, iterator = "%arg8", mapped_to = "y", reuse_type = "no_reuse", volume = 0 : i64}, {depth = 3 : i64, iterator = "%arg9", mapped_to = "x", reuse_type = "no_reuse", volume = 0 : i64}], temporal = [{depth = 0 : i64, iterator = "%arg6", reuse_type = "no_reuse", volume = 0 : i64}, {depth = 1 : i64, iterator = "%arg7", reuse_type = "no_reuse", volume = 0 : i64}, {depth = 4 : i64, iterator = "%arg10", reuse_type = "no_reuse", volume = 0 : i64}, {depth = 5 : i64, iterator = "%arg11", reuse_type = "no_reuse", volume = 0 : i64}]}} : memref<*xf32> to memref<64x64xf32, strided<[512, 1], offset: ?>>
                memref.copy %7, %reinterpret_cast : memref<64x64xf32> to memref<64x64xf32, strided<[512, 1], offset: ?>>
              }
            }
          } {tmd.mapped_to = "x"}
        } {tmd.mapped_to = "y"}
      }
    }
    return
  }
  func.func @matmul_kernel__d0i0_d1i1__f10__c0by_c1bx(%arg0: memref<*xf32> {tt.divisibility = 16 : i32}, %arg1: memref<*xf32> {tt.divisibility = 16 : i32}, %arg2: memref<*xf32> {tt.divisibility = 16 : i32}, %arg3: index, %arg4: index, %arg5: index) {
    affine.for %arg6 = 0 to affine_map<(d0, d1) -> (d1 ceildiv 8)>(%arg3, %arg4) {
      affine.for %arg7 = 0 to affine_map<(d0, d1) -> (d0 ceildiv 8)>(%arg3, %arg4) {
        affine.parallel (%arg8) = (0) to (8) {
          affine.parallel (%arg9) = (0) to (8) {
            %cst = arith.constant 0.000000e+00 : f32
            %alloc = memref.alloc() {alignment = 64 : i64} : memref<64x64xf32>
            linalg.fill ins(%cst : f32) outs(%alloc : memref<64x64xf32>)
            %c0 = arith.constant 0 : index
            %c8 = arith.constant 8 : index
            %c1 = arith.constant 1 : index
            affine.for %arg10 = 0 to affine_map<(d0) -> (d0 ceildiv 64)>(%arg3) {
              affine.for %arg11 = 0 to affine_map<(d0) -> (d0 ceildiv 64)>(%arg4) {
                %alloc_0 = memref.alloc() {alignment = 64 : i64} : memref<64x64xf32>
                memref.copy %alloc, %alloc_0 : memref<64x64xf32> to memref<64x64xf32>
                %7 = scf.for %arg12 = %c0 to %c8 step %c1 iter_args(%arg13 = %alloc_0) -> (memref<64x64xf32>) {
                  %9 = affine.apply affine_map<(d0, d1, d2, d3) -> (d0 * 64 + d1 * 2097152 + d2 * 262144 + d3 * 32768)>(%arg12, %arg10, %arg7, %arg9)
                  %reinterpret_cast_1 = memref.reinterpret_cast %arg0 to offset: [%9], sizes: [64, 64], strides: [512, 1] {tmd.reuse = {sequential = [{depth = 6 : i64, iterator = "%arg12", reuse_type = "no_reuse", volume = 0 : i64}], spatial = [{depth = 2 : i64, iterator = "%arg8", mapped_to = "y", reuse_type = "total_reuse", volume = 16384 : i64}, {depth = 3 : i64, iterator = "%arg9", mapped_to = "x", reuse_type = "no_reuse", volume = 0 : i64}], temporal = [{depth = 0 : i64, iterator = "%arg6", reuse_type = "total_reuse", volume = 16384 : i64}, {depth = 1 : i64, iterator = "%arg7", reuse_type = "no_reuse", volume = 0 : i64}, {depth = 4 : i64, iterator = "%arg10", reuse_type = "no_reuse", volume = 0 : i64}, {depth = 5 : i64, iterator = "%arg11", reuse_type = "total_reuse", volume = 16384 : i64}]}} : memref<*xf32> to memref<64x64xf32, strided<[512, 1], offset: ?>>
                  %alloc_2 = memref.alloc() {tmd.alloc = {local = true, memory_name = "L1", size = 16384 : i64}} : memref<64x64xf32>
                  memref.copy %reinterpret_cast_1, %alloc_2 {tmd.copy.choice = {dim = "y", interconnect_name = "vertical_links", kind = "broadcast"}} : memref<64x64xf32, strided<[512, 1], offset: ?>> to memref<64x64xf32>
                  %10 = affine.apply affine_map<(d0, d1, d2, d3) -> (d0 * 32768 + d1 * 4096 + d2 * 512 + d3 * 64)>(%arg12, %arg11, %arg6, %arg8)
                  %reinterpret_cast_3 = memref.reinterpret_cast %arg1 to offset: [%10], sizes: [64, 64], strides: [512, 1] {tmd.reuse = {sequential = [{depth = 6 : i64, iterator = "%arg12", reuse_type = "no_reuse", volume = 0 : i64}], spatial = [{depth = 2 : i64, iterator = "%arg8", mapped_to = "y", reuse_type = "no_reuse", volume = 0 : i64}, {depth = 3 : i64, iterator = "%arg9", mapped_to = "x", reuse_type = "total_reuse", volume = 16384 : i64}], temporal = [{depth = 0 : i64, iterator = "%arg6", reuse_type = "no_reuse", volume = 0 : i64}, {depth = 1 : i64, iterator = "%arg7", reuse_type = "total_reuse", volume = 16384 : i64}, {depth = 4 : i64, iterator = "%arg10", reuse_type = "total_reuse", volume = 16384 : i64}, {depth = 5 : i64, iterator = "%arg11", reuse_type = "no_reuse", volume = 0 : i64}]}} : memref<*xf32> to memref<64x64xf32, strided<[512, 1], offset: ?>>
                  %alloc_4 = memref.alloc() {tmd.alloc = {local = true, memory_name = "L1", size = 16384 : i64}} : memref<64x64xf32>
                  memref.copy %reinterpret_cast_3, %alloc_4 {tmd.copy.choice = {dim = "x", interconnect_name = "horizontal_links", kind = "broadcast"}} : memref<64x64xf32, strided<[512, 1], offset: ?>> to memref<64x64xf32>
                  %alloc_5 = memref.alloc() {alignment = 64 : i64} : memref<64x64xf32>
                  memref.copy %alloc, %alloc_5 : memref<64x64xf32> to memref<64x64xf32>
                  %c32 = arith.constant 32 : index
                  %alloc_6 = memref.alloc() {tmd.alloc = {local = true, memory_name = "L1", size = 16384 : i64}} : memref<64x64xf32>
                  %alloc_7 = memref.alloc() {tmd.alloc = {local = true, memory_name = "L1", size = 16384 : i64}} : memref<64x64xf32>
                  memref.copy %alloc_2, %alloc_6 : memref<64x64xf32> to memref<64x64xf32>
                  memref.copy %alloc_4, %alloc_7 : memref<64x64xf32> to memref<64x64xf32>
                  affine.for %arg14 = 0 to 2 {
                    affine.for %arg15 = 0 to 2 {
                      %11 = affine.apply affine_map<(d0) -> (d0 * 32)>(%arg14)
                      %12 = affine.apply affine_map<(d0) -> (d0 * 32)>(%arg15)
                      %alloc_8 = memref.alloc() : memref<32x32xf32>
                      %subview = memref.subview %arg13[%11, %12] [%c32, %c32] [%c1, %c1] : memref<64x64xf32> to memref<?x?xf32, strided<[?, ?], offset: ?>>
                      memref.copy %subview, %alloc_8 : memref<?x?xf32, strided<[?, ?], offset: ?>> to memref<32x32xf32>
                      affine.for %arg16 = 0 to 2 {
                        %13 = affine.apply affine_map<(d0) -> (d0 * 32)>(%arg16)
                        %subview_9 = memref.subview %alloc_6[%11, %13] [%c32, %c32] [%c1, %c1] : memref<64x64xf32> to memref<?x?xf32, strided<[?, ?], offset: ?>>
                        %subview_10 = memref.subview %alloc_7[%13, %12] [%c32, %c32] [%c1, %c1] : memref<64x64xf32> to memref<?x?xf32, strided<[?, ?], offset: ?>>
                        linalg.matmul ins(%subview_9, %subview_10 : memref<?x?xf32, strided<[?, ?], offset: ?>>, memref<?x?xf32, strided<[?, ?], offset: ?>>) outs(%alloc_8 : memref<32x32xf32>)
                      }
                      memref.copy %alloc_8, %subview : memref<32x32xf32> to memref<?x?xf32, strided<[?, ?], offset: ?>>
                    }
                  }
                  linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%arg13, %alloc_5 : memref<64x64xf32>, memref<64x64xf32>) outs(%arg13 : memref<64x64xf32>) {
                  ^bb0(%in: f32, %in_8: f32, %out: f32):
                    %11 = arith.addf %in, %in_8 : f32
                    linalg.yield %11 : f32
                  }
                  scf.yield %arg13 : memref<64x64xf32>
                }
                %8 = affine.apply affine_map<(d0, d1, d2, d3, d4, d5) -> (d0 * 4096 + d1 * 512 + d2 * 64 + d3 * 2097152 + d4 * 262144 + d5 * 32768)>(%arg11, %arg6, %arg8, %arg10, %arg7, %arg9)
                %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%8], sizes: [64, 64], strides: [512, 1] {tmd.reuse = {spatial = [{depth = 2 : i64, iterator = "%arg8", mapped_to = "y", reuse_type = "no_reuse", volume = 0 : i64}, {depth = 3 : i64, iterator = "%arg9", mapped_to = "x", reuse_type = "no_reuse", volume = 0 : i64}], temporal = [{depth = 0 : i64, iterator = "%arg6", reuse_type = "no_reuse", volume = 0 : i64}, {depth = 1 : i64, iterator = "%arg7", reuse_type = "no_reuse", volume = 0 : i64}, {depth = 4 : i64, iterator = "%arg10", reuse_type = "no_reuse", volume = 0 : i64}, {depth = 5 : i64, iterator = "%arg11", reuse_type = "no_reuse", volume = 0 : i64}]}} : memref<*xf32> to memref<64x64xf32, strided<[512, 1], offset: ?>>
                memref.copy %7, %reinterpret_cast : memref<64x64xf32> to memref<64x64xf32, strided<[512, 1], offset: ?>>
              }
            }
          } {tmd.mapped_to = "x"}
        } {tmd.mapped_to = "y"}
      }
    }
    return
  }
  func.func @matmul_kernel__d1i0_d0i1__f01__c0mem_c1mem(%arg0: memref<*xf32> {tt.divisibility = 16 : i32}, %arg1: memref<*xf32> {tt.divisibility = 16 : i32}, %arg2: memref<*xf32> {tt.divisibility = 16 : i32}, %arg3: index, %arg4: index, %arg5: index) {
    affine.for %arg6 = 0 to affine_map<(d0, d1) -> (d0 ceildiv 8)>(%arg3, %arg4) {
      affine.for %arg7 = 0 to affine_map<(d0, d1) -> (d1 ceildiv 8)>(%arg3, %arg4) {
        affine.parallel (%arg8) = (0) to (8) {
          affine.parallel (%arg9) = (0) to (8) {
            %cst = arith.constant 0.000000e+00 : f32
            %alloc = memref.alloc() {alignment = 64 : i64} : memref<64x64xf32>
            linalg.fill ins(%cst : f32) outs(%alloc : memref<64x64xf32>)
            %c0 = arith.constant 0 : index
            %c8 = arith.constant 8 : index
            %c1 = arith.constant 1 : index
            affine.for %arg10 = 0 to affine_map<(d0) -> (d0 ceildiv 64)>(%arg3) {
              affine.for %arg11 = 0 to affine_map<(d0) -> (d0 ceildiv 64)>(%arg4) {
                %alloc_0 = memref.alloc() {alignment = 64 : i64} : memref<64x64xf32>
                memref.copy %alloc, %alloc_0 : memref<64x64xf32> to memref<64x64xf32>
                %7 = scf.for %arg12 = %c0 to %c8 step %c1 iter_args(%arg13 = %alloc_0) -> (memref<64x64xf32>) {
                  %9 = affine.apply affine_map<(d0, d1, d2, d3) -> (d0 * 64 + d1 * 2097152 + d2 * 262144 + d3 * 32768)>(%arg12, %arg10, %arg6, %arg9)
                  %reinterpret_cast_1 = memref.reinterpret_cast %arg0 to offset: [%9], sizes: [64, 64], strides: [512, 1] {tmd.reuse = {sequential = [{depth = 6 : i64, iterator = "%arg12", reuse_type = "no_reuse", volume = 0 : i64}], spatial = [{depth = 2 : i64, iterator = "%arg8", mapped_to = "x", reuse_type = "total_reuse", volume = 16384 : i64}, {depth = 3 : i64, iterator = "%arg9", mapped_to = "y", reuse_type = "no_reuse", volume = 0 : i64}], temporal = [{depth = 0 : i64, iterator = "%arg6", reuse_type = "no_reuse", volume = 0 : i64}, {depth = 1 : i64, iterator = "%arg7", reuse_type = "total_reuse", volume = 16384 : i64}, {depth = 4 : i64, iterator = "%arg10", reuse_type = "no_reuse", volume = 0 : i64}, {depth = 5 : i64, iterator = "%arg11", reuse_type = "total_reuse", volume = 16384 : i64}]}} : memref<*xf32> to memref<64x64xf32, strided<[512, 1], offset: ?>>
                  %alloc_2 = memref.alloc() {tmd.alloc = {local = true, memory_name = "L1", size = 16384 : i64}} : memref<64x64xf32>
                  memref.copy %reinterpret_cast_1, %alloc_2 {tmd.copy.choice = {kind = "mem", memory_name = "L1"}} : memref<64x64xf32, strided<[512, 1], offset: ?>> to memref<64x64xf32>
                  %10 = affine.apply affine_map<(d0, d1, d2, d3) -> (d0 * 32768 + d1 * 4096 + d2 * 512 + d3 * 64)>(%arg12, %arg11, %arg7, %arg8)
                  %reinterpret_cast_3 = memref.reinterpret_cast %arg1 to offset: [%10], sizes: [64, 64], strides: [512, 1] {tmd.reuse = {sequential = [{depth = 6 : i64, iterator = "%arg12", reuse_type = "no_reuse", volume = 0 : i64}], spatial = [{depth = 2 : i64, iterator = "%arg8", mapped_to = "x", reuse_type = "no_reuse", volume = 0 : i64}, {depth = 3 : i64, iterator = "%arg9", mapped_to = "y", reuse_type = "total_reuse", volume = 16384 : i64}], temporal = [{depth = 0 : i64, iterator = "%arg6", reuse_type = "total_reuse", volume = 16384 : i64}, {depth = 1 : i64, iterator = "%arg7", reuse_type = "no_reuse", volume = 0 : i64}, {depth = 4 : i64, iterator = "%arg10", reuse_type = "total_reuse", volume = 16384 : i64}, {depth = 5 : i64, iterator = "%arg11", reuse_type = "no_reuse", volume = 0 : i64}]}} : memref<*xf32> to memref<64x64xf32, strided<[512, 1], offset: ?>>
                  %alloc_4 = memref.alloc() {tmd.alloc = {local = true, memory_name = "L1", size = 16384 : i64}} : memref<64x64xf32>
                  memref.copy %reinterpret_cast_3, %alloc_4 {tmd.copy.choice = {kind = "mem", memory_name = "L1"}} : memref<64x64xf32, strided<[512, 1], offset: ?>> to memref<64x64xf32>
                  %alloc_5 = memref.alloc() {alignment = 64 : i64} : memref<64x64xf32>
                  memref.copy %alloc, %alloc_5 : memref<64x64xf32> to memref<64x64xf32>
                  %c32 = arith.constant 32 : index
                  %alloc_6 = memref.alloc() {tmd.alloc = {local = true, memory_name = "L1", size = 16384 : i64}} : memref<64x64xf32>
                  %alloc_7 = memref.alloc() {tmd.alloc = {local = true, memory_name = "L1", size = 16384 : i64}} : memref<64x64xf32>
                  memref.copy %alloc_2, %alloc_6 : memref<64x64xf32> to memref<64x64xf32>
                  memref.copy %alloc_4, %alloc_7 : memref<64x64xf32> to memref<64x64xf32>
                  affine.for %arg14 = 0 to 2 {
                    affine.for %arg15 = 0 to 2 {
                      %11 = affine.apply affine_map<(d0) -> (d0 * 32)>(%arg14)
                      %12 = affine.apply affine_map<(d0) -> (d0 * 32)>(%arg15)
                      %alloc_8 = memref.alloc() : memref<32x32xf32>
                      %subview = memref.subview %arg13[%11, %12] [%c32, %c32] [%c1, %c1] : memref<64x64xf32> to memref<?x?xf32, strided<[?, ?], offset: ?>>
                      memref.copy %subview, %alloc_8 : memref<?x?xf32, strided<[?, ?], offset: ?>> to memref<32x32xf32>
                      affine.for %arg16 = 0 to 2 {
                        %13 = affine.apply affine_map<(d0) -> (d0 * 32)>(%arg16)
                        %subview_9 = memref.subview %alloc_6[%11, %13] [%c32, %c32] [%c1, %c1] : memref<64x64xf32> to memref<?x?xf32, strided<[?, ?], offset: ?>>
                        %subview_10 = memref.subview %alloc_7[%13, %12] [%c32, %c32] [%c1, %c1] : memref<64x64xf32> to memref<?x?xf32, strided<[?, ?], offset: ?>>
                        linalg.matmul ins(%subview_9, %subview_10 : memref<?x?xf32, strided<[?, ?], offset: ?>>, memref<?x?xf32, strided<[?, ?], offset: ?>>) outs(%alloc_8 : memref<32x32xf32>)
                      }
                      memref.copy %alloc_8, %subview : memref<32x32xf32> to memref<?x?xf32, strided<[?, ?], offset: ?>>
                    }
                  }
                  linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%arg13, %alloc_5 : memref<64x64xf32>, memref<64x64xf32>) outs(%arg13 : memref<64x64xf32>) {
                  ^bb0(%in: f32, %in_8: f32, %out: f32):
                    %11 = arith.addf %in, %in_8 : f32
                    linalg.yield %11 : f32
                  }
                  scf.yield %arg13 : memref<64x64xf32>
                }
                %8 = affine.apply affine_map<(d0, d1, d2, d3, d4, d5) -> (d0 * 4096 + d1 * 512 + d2 * 64 + d3 * 2097152 + d4 * 262144 + d5 * 32768)>(%arg11, %arg7, %arg8, %arg10, %arg6, %arg9)
                %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%8], sizes: [64, 64], strides: [512, 1] {tmd.reuse = {spatial = [{depth = 2 : i64, iterator = "%arg8", mapped_to = "x", reuse_type = "no_reuse", volume = 0 : i64}, {depth = 3 : i64, iterator = "%arg9", mapped_to = "y", reuse_type = "no_reuse", volume = 0 : i64}], temporal = [{depth = 0 : i64, iterator = "%arg6", reuse_type = "no_reuse", volume = 0 : i64}, {depth = 1 : i64, iterator = "%arg7", reuse_type = "no_reuse", volume = 0 : i64}, {depth = 4 : i64, iterator = "%arg10", reuse_type = "no_reuse", volume = 0 : i64}, {depth = 5 : i64, iterator = "%arg11", reuse_type = "no_reuse", volume = 0 : i64}]}} : memref<*xf32> to memref<64x64xf32, strided<[512, 1], offset: ?>>
                memref.copy %7, %reinterpret_cast : memref<64x64xf32> to memref<64x64xf32, strided<[512, 1], offset: ?>>
              }
            }
          } {tmd.mapped_to = "y"}
        } {tmd.mapped_to = "x"}
      }
    }
    return
  }
  func.func @matmul_kernel__d1i0_d0i1__f01__c0bx_c1mem(%arg0: memref<*xf32> {tt.divisibility = 16 : i32}, %arg1: memref<*xf32> {tt.divisibility = 16 : i32}, %arg2: memref<*xf32> {tt.divisibility = 16 : i32}, %arg3: index, %arg4: index, %arg5: index) {
    affine.for %arg6 = 0 to affine_map<(d0, d1) -> (d0 ceildiv 8)>(%arg3, %arg4) {
      affine.for %arg7 = 0 to affine_map<(d0, d1) -> (d1 ceildiv 8)>(%arg3, %arg4) {
        affine.parallel (%arg8) = (0) to (8) {
          affine.parallel (%arg9) = (0) to (8) {
            %cst = arith.constant 0.000000e+00 : f32
            %alloc = memref.alloc() {alignment = 64 : i64} : memref<64x64xf32>
            linalg.fill ins(%cst : f32) outs(%alloc : memref<64x64xf32>)
            %c0 = arith.constant 0 : index
            %c8 = arith.constant 8 : index
            %c1 = arith.constant 1 : index
            affine.for %arg10 = 0 to affine_map<(d0) -> (d0 ceildiv 64)>(%arg3) {
              affine.for %arg11 = 0 to affine_map<(d0) -> (d0 ceildiv 64)>(%arg4) {
                %alloc_0 = memref.alloc() {alignment = 64 : i64} : memref<64x64xf32>
                memref.copy %alloc, %alloc_0 : memref<64x64xf32> to memref<64x64xf32>
                %7 = scf.for %arg12 = %c0 to %c8 step %c1 iter_args(%arg13 = %alloc_0) -> (memref<64x64xf32>) {
                  %9 = affine.apply affine_map<(d0, d1, d2, d3) -> (d0 * 64 + d1 * 2097152 + d2 * 262144 + d3 * 32768)>(%arg12, %arg10, %arg6, %arg9)
                  %reinterpret_cast_1 = memref.reinterpret_cast %arg0 to offset: [%9], sizes: [64, 64], strides: [512, 1] {tmd.reuse = {sequential = [{depth = 6 : i64, iterator = "%arg12", reuse_type = "no_reuse", volume = 0 : i64}], spatial = [{depth = 2 : i64, iterator = "%arg8", mapped_to = "x", reuse_type = "total_reuse", volume = 16384 : i64}, {depth = 3 : i64, iterator = "%arg9", mapped_to = "y", reuse_type = "no_reuse", volume = 0 : i64}], temporal = [{depth = 0 : i64, iterator = "%arg6", reuse_type = "no_reuse", volume = 0 : i64}, {depth = 1 : i64, iterator = "%arg7", reuse_type = "total_reuse", volume = 16384 : i64}, {depth = 4 : i64, iterator = "%arg10", reuse_type = "no_reuse", volume = 0 : i64}, {depth = 5 : i64, iterator = "%arg11", reuse_type = "total_reuse", volume = 16384 : i64}]}} : memref<*xf32> to memref<64x64xf32, strided<[512, 1], offset: ?>>
                  %alloc_2 = memref.alloc() {tmd.alloc = {local = true, memory_name = "L1", size = 16384 : i64}} : memref<64x64xf32>
                  memref.copy %reinterpret_cast_1, %alloc_2 {tmd.copy.choice = {dim = "x", interconnect_name = "horizontal_links", kind = "broadcast"}} : memref<64x64xf32, strided<[512, 1], offset: ?>> to memref<64x64xf32>
                  %10 = affine.apply affine_map<(d0, d1, d2, d3) -> (d0 * 32768 + d1 * 4096 + d2 * 512 + d3 * 64)>(%arg12, %arg11, %arg7, %arg8)
                  %reinterpret_cast_3 = memref.reinterpret_cast %arg1 to offset: [%10], sizes: [64, 64], strides: [512, 1] {tmd.reuse = {sequential = [{depth = 6 : i64, iterator = "%arg12", reuse_type = "no_reuse", volume = 0 : i64}], spatial = [{depth = 2 : i64, iterator = "%arg8", mapped_to = "x", reuse_type = "no_reuse", volume = 0 : i64}, {depth = 3 : i64, iterator = "%arg9", mapped_to = "y", reuse_type = "total_reuse", volume = 16384 : i64}], temporal = [{depth = 0 : i64, iterator = "%arg6", reuse_type = "total_reuse", volume = 16384 : i64}, {depth = 1 : i64, iterator = "%arg7", reuse_type = "no_reuse", volume = 0 : i64}, {depth = 4 : i64, iterator = "%arg10", reuse_type = "total_reuse", volume = 16384 : i64}, {depth = 5 : i64, iterator = "%arg11", reuse_type = "no_reuse", volume = 0 : i64}]}} : memref<*xf32> to memref<64x64xf32, strided<[512, 1], offset: ?>>
                  %alloc_4 = memref.alloc() {tmd.alloc = {local = true, memory_name = "L1", size = 16384 : i64}} : memref<64x64xf32>
                  memref.copy %reinterpret_cast_3, %alloc_4 {tmd.copy.choice = {kind = "mem", memory_name = "L1"}} : memref<64x64xf32, strided<[512, 1], offset: ?>> to memref<64x64xf32>
                  %alloc_5 = memref.alloc() {alignment = 64 : i64} : memref<64x64xf32>
                  memref.copy %alloc, %alloc_5 : memref<64x64xf32> to memref<64x64xf32>
                  %c32 = arith.constant 32 : index
                  %alloc_6 = memref.alloc() {tmd.alloc = {local = true, memory_name = "L1", size = 16384 : i64}} : memref<64x64xf32>
                  %alloc_7 = memref.alloc() {tmd.alloc = {local = true, memory_name = "L1", size = 16384 : i64}} : memref<64x64xf32>
                  memref.copy %alloc_2, %alloc_6 : memref<64x64xf32> to memref<64x64xf32>
                  memref.copy %alloc_4, %alloc_7 : memref<64x64xf32> to memref<64x64xf32>
                  affine.for %arg14 = 0 to 2 {
                    affine.for %arg15 = 0 to 2 {
                      %11 = affine.apply affine_map<(d0) -> (d0 * 32)>(%arg14)
                      %12 = affine.apply affine_map<(d0) -> (d0 * 32)>(%arg15)
                      %alloc_8 = memref.alloc() : memref<32x32xf32>
                      %subview = memref.subview %arg13[%11, %12] [%c32, %c32] [%c1, %c1] : memref<64x64xf32> to memref<?x?xf32, strided<[?, ?], offset: ?>>
                      memref.copy %subview, %alloc_8 : memref<?x?xf32, strided<[?, ?], offset: ?>> to memref<32x32xf32>
                      affine.for %arg16 = 0 to 2 {
                        %13 = affine.apply affine_map<(d0) -> (d0 * 32)>(%arg16)
                        %subview_9 = memref.subview %alloc_6[%11, %13] [%c32, %c32] [%c1, %c1] : memref<64x64xf32> to memref<?x?xf32, strided<[?, ?], offset: ?>>
                        %subview_10 = memref.subview %alloc_7[%13, %12] [%c32, %c32] [%c1, %c1] : memref<64x64xf32> to memref<?x?xf32, strided<[?, ?], offset: ?>>
                        linalg.matmul ins(%subview_9, %subview_10 : memref<?x?xf32, strided<[?, ?], offset: ?>>, memref<?x?xf32, strided<[?, ?], offset: ?>>) outs(%alloc_8 : memref<32x32xf32>)
                      }
                      memref.copy %alloc_8, %subview : memref<32x32xf32> to memref<?x?xf32, strided<[?, ?], offset: ?>>
                    }
                  }
                  linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%arg13, %alloc_5 : memref<64x64xf32>, memref<64x64xf32>) outs(%arg13 : memref<64x64xf32>) {
                  ^bb0(%in: f32, %in_8: f32, %out: f32):
                    %11 = arith.addf %in, %in_8 : f32
                    linalg.yield %11 : f32
                  }
                  scf.yield %arg13 : memref<64x64xf32>
                }
                %8 = affine.apply affine_map<(d0, d1, d2, d3, d4, d5) -> (d0 * 4096 + d1 * 512 + d2 * 64 + d3 * 2097152 + d4 * 262144 + d5 * 32768)>(%arg11, %arg7, %arg8, %arg10, %arg6, %arg9)
                %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%8], sizes: [64, 64], strides: [512, 1] {tmd.reuse = {spatial = [{depth = 2 : i64, iterator = "%arg8", mapped_to = "x", reuse_type = "no_reuse", volume = 0 : i64}, {depth = 3 : i64, iterator = "%arg9", mapped_to = "y", reuse_type = "no_reuse", volume = 0 : i64}], temporal = [{depth = 0 : i64, iterator = "%arg6", reuse_type = "no_reuse", volume = 0 : i64}, {depth = 1 : i64, iterator = "%arg7", reuse_type = "no_reuse", volume = 0 : i64}, {depth = 4 : i64, iterator = "%arg10", reuse_type = "no_reuse", volume = 0 : i64}, {depth = 5 : i64, iterator = "%arg11", reuse_type = "no_reuse", volume = 0 : i64}]}} : memref<*xf32> to memref<64x64xf32, strided<[512, 1], offset: ?>>
                memref.copy %7, %reinterpret_cast : memref<64x64xf32> to memref<64x64xf32, strided<[512, 1], offset: ?>>
              }
            }
          } {tmd.mapped_to = "y"}
        } {tmd.mapped_to = "x"}
      }
    }
    return
  }
  func.func @matmul_kernel__d1i0_d0i1__f01__c0mem_c1by(%arg0: memref<*xf32> {tt.divisibility = 16 : i32}, %arg1: memref<*xf32> {tt.divisibility = 16 : i32}, %arg2: memref<*xf32> {tt.divisibility = 16 : i32}, %arg3: index, %arg4: index, %arg5: index) {
    affine.for %arg6 = 0 to affine_map<(d0, d1) -> (d0 ceildiv 8)>(%arg3, %arg4) {
      affine.for %arg7 = 0 to affine_map<(d0, d1) -> (d1 ceildiv 8)>(%arg3, %arg4) {
        affine.parallel (%arg8) = (0) to (8) {
          affine.parallel (%arg9) = (0) to (8) {
            %cst = arith.constant 0.000000e+00 : f32
            %alloc = memref.alloc() {alignment = 64 : i64} : memref<64x64xf32>
            linalg.fill ins(%cst : f32) outs(%alloc : memref<64x64xf32>)
            %c0 = arith.constant 0 : index
            %c8 = arith.constant 8 : index
            %c1 = arith.constant 1 : index
            affine.for %arg10 = 0 to affine_map<(d0) -> (d0 ceildiv 64)>(%arg3) {
              affine.for %arg11 = 0 to affine_map<(d0) -> (d0 ceildiv 64)>(%arg4) {
                %alloc_0 = memref.alloc() {alignment = 64 : i64} : memref<64x64xf32>
                memref.copy %alloc, %alloc_0 : memref<64x64xf32> to memref<64x64xf32>
                %7 = scf.for %arg12 = %c0 to %c8 step %c1 iter_args(%arg13 = %alloc_0) -> (memref<64x64xf32>) {
                  %9 = affine.apply affine_map<(d0, d1, d2, d3) -> (d0 * 64 + d1 * 2097152 + d2 * 262144 + d3 * 32768)>(%arg12, %arg10, %arg6, %arg9)
                  %reinterpret_cast_1 = memref.reinterpret_cast %arg0 to offset: [%9], sizes: [64, 64], strides: [512, 1] {tmd.reuse = {sequential = [{depth = 6 : i64, iterator = "%arg12", reuse_type = "no_reuse", volume = 0 : i64}], spatial = [{depth = 2 : i64, iterator = "%arg8", mapped_to = "x", reuse_type = "total_reuse", volume = 16384 : i64}, {depth = 3 : i64, iterator = "%arg9", mapped_to = "y", reuse_type = "no_reuse", volume = 0 : i64}], temporal = [{depth = 0 : i64, iterator = "%arg6", reuse_type = "no_reuse", volume = 0 : i64}, {depth = 1 : i64, iterator = "%arg7", reuse_type = "total_reuse", volume = 16384 : i64}, {depth = 4 : i64, iterator = "%arg10", reuse_type = "no_reuse", volume = 0 : i64}, {depth = 5 : i64, iterator = "%arg11", reuse_type = "total_reuse", volume = 16384 : i64}]}} : memref<*xf32> to memref<64x64xf32, strided<[512, 1], offset: ?>>
                  %alloc_2 = memref.alloc() {tmd.alloc = {local = true, memory_name = "L1", size = 16384 : i64}} : memref<64x64xf32>
                  memref.copy %reinterpret_cast_1, %alloc_2 {tmd.copy.choice = {kind = "mem", memory_name = "L1"}} : memref<64x64xf32, strided<[512, 1], offset: ?>> to memref<64x64xf32>
                  %10 = affine.apply affine_map<(d0, d1, d2, d3) -> (d0 * 32768 + d1 * 4096 + d2 * 512 + d3 * 64)>(%arg12, %arg11, %arg7, %arg8)
                  %reinterpret_cast_3 = memref.reinterpret_cast %arg1 to offset: [%10], sizes: [64, 64], strides: [512, 1] {tmd.reuse = {sequential = [{depth = 6 : i64, iterator = "%arg12", reuse_type = "no_reuse", volume = 0 : i64}], spatial = [{depth = 2 : i64, iterator = "%arg8", mapped_to = "x", reuse_type = "no_reuse", volume = 0 : i64}, {depth = 3 : i64, iterator = "%arg9", mapped_to = "y", reuse_type = "total_reuse", volume = 16384 : i64}], temporal = [{depth = 0 : i64, iterator = "%arg6", reuse_type = "total_reuse", volume = 16384 : i64}, {depth = 1 : i64, iterator = "%arg7", reuse_type = "no_reuse", volume = 0 : i64}, {depth = 4 : i64, iterator = "%arg10", reuse_type = "total_reuse", volume = 16384 : i64}, {depth = 5 : i64, iterator = "%arg11", reuse_type = "no_reuse", volume = 0 : i64}]}} : memref<*xf32> to memref<64x64xf32, strided<[512, 1], offset: ?>>
                  %alloc_4 = memref.alloc() {tmd.alloc = {local = true, memory_name = "L1", size = 16384 : i64}} : memref<64x64xf32>
                  memref.copy %reinterpret_cast_3, %alloc_4 {tmd.copy.choice = {dim = "y", interconnect_name = "vertical_links", kind = "broadcast"}} : memref<64x64xf32, strided<[512, 1], offset: ?>> to memref<64x64xf32>
                  %alloc_5 = memref.alloc() {alignment = 64 : i64} : memref<64x64xf32>
                  memref.copy %alloc, %alloc_5 : memref<64x64xf32> to memref<64x64xf32>
                  %c32 = arith.constant 32 : index
                  %alloc_6 = memref.alloc() {tmd.alloc = {local = true, memory_name = "L1", size = 16384 : i64}} : memref<64x64xf32>
                  %alloc_7 = memref.alloc() {tmd.alloc = {local = true, memory_name = "L1", size = 16384 : i64}} : memref<64x64xf32>
                  memref.copy %alloc_2, %alloc_6 : memref<64x64xf32> to memref<64x64xf32>
                  memref.copy %alloc_4, %alloc_7 : memref<64x64xf32> to memref<64x64xf32>
                  affine.for %arg14 = 0 to 2 {
                    affine.for %arg15 = 0 to 2 {
                      %11 = affine.apply affine_map<(d0) -> (d0 * 32)>(%arg14)
                      %12 = affine.apply affine_map<(d0) -> (d0 * 32)>(%arg15)
                      %alloc_8 = memref.alloc() : memref<32x32xf32>
                      %subview = memref.subview %arg13[%11, %12] [%c32, %c32] [%c1, %c1] : memref<64x64xf32> to memref<?x?xf32, strided<[?, ?], offset: ?>>
                      memref.copy %subview, %alloc_8 : memref<?x?xf32, strided<[?, ?], offset: ?>> to memref<32x32xf32>
                      affine.for %arg16 = 0 to 2 {
                        %13 = affine.apply affine_map<(d0) -> (d0 * 32)>(%arg16)
                        %subview_9 = memref.subview %alloc_6[%11, %13] [%c32, %c32] [%c1, %c1] : memref<64x64xf32> to memref<?x?xf32, strided<[?, ?], offset: ?>>
                        %subview_10 = memref.subview %alloc_7[%13, %12] [%c32, %c32] [%c1, %c1] : memref<64x64xf32> to memref<?x?xf32, strided<[?, ?], offset: ?>>
                        linalg.matmul ins(%subview_9, %subview_10 : memref<?x?xf32, strided<[?, ?], offset: ?>>, memref<?x?xf32, strided<[?, ?], offset: ?>>) outs(%alloc_8 : memref<32x32xf32>)
                      }
                      memref.copy %alloc_8, %subview : memref<32x32xf32> to memref<?x?xf32, strided<[?, ?], offset: ?>>
                    }
                  }
                  linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%arg13, %alloc_5 : memref<64x64xf32>, memref<64x64xf32>) outs(%arg13 : memref<64x64xf32>) {
                  ^bb0(%in: f32, %in_8: f32, %out: f32):
                    %11 = arith.addf %in, %in_8 : f32
                    linalg.yield %11 : f32
                  }
                  scf.yield %arg13 : memref<64x64xf32>
                }
                %8 = affine.apply affine_map<(d0, d1, d2, d3, d4, d5) -> (d0 * 4096 + d1 * 512 + d2 * 64 + d3 * 2097152 + d4 * 262144 + d5 * 32768)>(%arg11, %arg7, %arg8, %arg10, %arg6, %arg9)
                %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%8], sizes: [64, 64], strides: [512, 1] {tmd.reuse = {spatial = [{depth = 2 : i64, iterator = "%arg8", mapped_to = "x", reuse_type = "no_reuse", volume = 0 : i64}, {depth = 3 : i64, iterator = "%arg9", mapped_to = "y", reuse_type = "no_reuse", volume = 0 : i64}], temporal = [{depth = 0 : i64, iterator = "%arg6", reuse_type = "no_reuse", volume = 0 : i64}, {depth = 1 : i64, iterator = "%arg7", reuse_type = "no_reuse", volume = 0 : i64}, {depth = 4 : i64, iterator = "%arg10", reuse_type = "no_reuse", volume = 0 : i64}, {depth = 5 : i64, iterator = "%arg11", reuse_type = "no_reuse", volume = 0 : i64}]}} : memref<*xf32> to memref<64x64xf32, strided<[512, 1], offset: ?>>
                memref.copy %7, %reinterpret_cast : memref<64x64xf32> to memref<64x64xf32, strided<[512, 1], offset: ?>>
              }
            }
          } {tmd.mapped_to = "y"}
        } {tmd.mapped_to = "x"}
      }
    }
    return
  }
  func.func @matmul_kernel__d1i0_d0i1__f01__c0bx_c1by(%arg0: memref<*xf32> {tt.divisibility = 16 : i32}, %arg1: memref<*xf32> {tt.divisibility = 16 : i32}, %arg2: memref<*xf32> {tt.divisibility = 16 : i32}, %arg3: index, %arg4: index, %arg5: index) {
    affine.for %arg6 = 0 to affine_map<(d0, d1) -> (d0 ceildiv 8)>(%arg3, %arg4) {
      affine.for %arg7 = 0 to affine_map<(d0, d1) -> (d1 ceildiv 8)>(%arg3, %arg4) {
        affine.parallel (%arg8) = (0) to (8) {
          affine.parallel (%arg9) = (0) to (8) {
            %cst = arith.constant 0.000000e+00 : f32
            %alloc = memref.alloc() {alignment = 64 : i64} : memref<64x64xf32>
            linalg.fill ins(%cst : f32) outs(%alloc : memref<64x64xf32>)
            %c0 = arith.constant 0 : index
            %c8 = arith.constant 8 : index
            %c1 = arith.constant 1 : index
            affine.for %arg10 = 0 to affine_map<(d0) -> (d0 ceildiv 64)>(%arg3) {
              affine.for %arg11 = 0 to affine_map<(d0) -> (d0 ceildiv 64)>(%arg4) {
                %alloc_0 = memref.alloc() {alignment = 64 : i64} : memref<64x64xf32>
                memref.copy %alloc, %alloc_0 : memref<64x64xf32> to memref<64x64xf32>
                %7 = scf.for %arg12 = %c0 to %c8 step %c1 iter_args(%arg13 = %alloc_0) -> (memref<64x64xf32>) {
                  %9 = affine.apply affine_map<(d0, d1, d2, d3) -> (d0 * 64 + d1 * 2097152 + d2 * 262144 + d3 * 32768)>(%arg12, %arg10, %arg6, %arg9)
                  %reinterpret_cast_1 = memref.reinterpret_cast %arg0 to offset: [%9], sizes: [64, 64], strides: [512, 1] {tmd.reuse = {sequential = [{depth = 6 : i64, iterator = "%arg12", reuse_type = "no_reuse", volume = 0 : i64}], spatial = [{depth = 2 : i64, iterator = "%arg8", mapped_to = "x", reuse_type = "total_reuse", volume = 16384 : i64}, {depth = 3 : i64, iterator = "%arg9", mapped_to = "y", reuse_type = "no_reuse", volume = 0 : i64}], temporal = [{depth = 0 : i64, iterator = "%arg6", reuse_type = "no_reuse", volume = 0 : i64}, {depth = 1 : i64, iterator = "%arg7", reuse_type = "total_reuse", volume = 16384 : i64}, {depth = 4 : i64, iterator = "%arg10", reuse_type = "no_reuse", volume = 0 : i64}, {depth = 5 : i64, iterator = "%arg11", reuse_type = "total_reuse", volume = 16384 : i64}]}} : memref<*xf32> to memref<64x64xf32, strided<[512, 1], offset: ?>>
                  %alloc_2 = memref.alloc() {tmd.alloc = {local = true, memory_name = "L1", size = 16384 : i64}} : memref<64x64xf32>
                  memref.copy %reinterpret_cast_1, %alloc_2 {tmd.copy.choice = {dim = "x", interconnect_name = "horizontal_links", kind = "broadcast"}} : memref<64x64xf32, strided<[512, 1], offset: ?>> to memref<64x64xf32>
                  %10 = affine.apply affine_map<(d0, d1, d2, d3) -> (d0 * 32768 + d1 * 4096 + d2 * 512 + d3 * 64)>(%arg12, %arg11, %arg7, %arg8)
                  %reinterpret_cast_3 = memref.reinterpret_cast %arg1 to offset: [%10], sizes: [64, 64], strides: [512, 1] {tmd.reuse = {sequential = [{depth = 6 : i64, iterator = "%arg12", reuse_type = "no_reuse", volume = 0 : i64}], spatial = [{depth = 2 : i64, iterator = "%arg8", mapped_to = "x", reuse_type = "no_reuse", volume = 0 : i64}, {depth = 3 : i64, iterator = "%arg9", mapped_to = "y", reuse_type = "total_reuse", volume = 16384 : i64}], temporal = [{depth = 0 : i64, iterator = "%arg6", reuse_type = "total_reuse", volume = 16384 : i64}, {depth = 1 : i64, iterator = "%arg7", reuse_type = "no_reuse", volume = 0 : i64}, {depth = 4 : i64, iterator = "%arg10", reuse_type = "total_reuse", volume = 16384 : i64}, {depth = 5 : i64, iterator = "%arg11", reuse_type = "no_reuse", volume = 0 : i64}]}} : memref<*xf32> to memref<64x64xf32, strided<[512, 1], offset: ?>>
                  %alloc_4 = memref.alloc() {tmd.alloc = {local = true, memory_name = "L1", size = 16384 : i64}} : memref<64x64xf32>
                  memref.copy %reinterpret_cast_3, %alloc_4 {tmd.copy.choice = {dim = "y", interconnect_name = "vertical_links", kind = "broadcast"}} : memref<64x64xf32, strided<[512, 1], offset: ?>> to memref<64x64xf32>
                  %alloc_5 = memref.alloc() {alignment = 64 : i64} : memref<64x64xf32>
                  memref.copy %alloc, %alloc_5 : memref<64x64xf32> to memref<64x64xf32>
                  %c32 = arith.constant 32 : index
                  %alloc_6 = memref.alloc() {tmd.alloc = {local = true, memory_name = "L1", size = 16384 : i64}} : memref<64x64xf32>
                  %alloc_7 = memref.alloc() {tmd.alloc = {local = true, memory_name = "L1", size = 16384 : i64}} : memref<64x64xf32>
                  memref.copy %alloc_2, %alloc_6 : memref<64x64xf32> to memref<64x64xf32>
                  memref.copy %alloc_4, %alloc_7 : memref<64x64xf32> to memref<64x64xf32>
                  affine.for %arg14 = 0 to 2 {
                    affine.for %arg15 = 0 to 2 {
                      %11 = affine.apply affine_map<(d0) -> (d0 * 32)>(%arg14)
                      %12 = affine.apply affine_map<(d0) -> (d0 * 32)>(%arg15)
                      %alloc_8 = memref.alloc() : memref<32x32xf32>
                      %subview = memref.subview %arg13[%11, %12] [%c32, %c32] [%c1, %c1] : memref<64x64xf32> to memref<?x?xf32, strided<[?, ?], offset: ?>>
                      memref.copy %subview, %alloc_8 : memref<?x?xf32, strided<[?, ?], offset: ?>> to memref<32x32xf32>
                      affine.for %arg16 = 0 to 2 {
                        %13 = affine.apply affine_map<(d0) -> (d0 * 32)>(%arg16)
                        %subview_9 = memref.subview %alloc_6[%11, %13] [%c32, %c32] [%c1, %c1] : memref<64x64xf32> to memref<?x?xf32, strided<[?, ?], offset: ?>>
                        %subview_10 = memref.subview %alloc_7[%13, %12] [%c32, %c32] [%c1, %c1] : memref<64x64xf32> to memref<?x?xf32, strided<[?, ?], offset: ?>>
                        linalg.matmul ins(%subview_9, %subview_10 : memref<?x?xf32, strided<[?, ?], offset: ?>>, memref<?x?xf32, strided<[?, ?], offset: ?>>) outs(%alloc_8 : memref<32x32xf32>)
                      }
                      memref.copy %alloc_8, %subview : memref<32x32xf32> to memref<?x?xf32, strided<[?, ?], offset: ?>>
                    }
                  }
                  linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%arg13, %alloc_5 : memref<64x64xf32>, memref<64x64xf32>) outs(%arg13 : memref<64x64xf32>) {
                  ^bb0(%in: f32, %in_8: f32, %out: f32):
                    %11 = arith.addf %in, %in_8 : f32
                    linalg.yield %11 : f32
                  }
                  scf.yield %arg13 : memref<64x64xf32>
                }
                %8 = affine.apply affine_map<(d0, d1, d2, d3, d4, d5) -> (d0 * 4096 + d1 * 512 + d2 * 64 + d3 * 2097152 + d4 * 262144 + d5 * 32768)>(%arg11, %arg7, %arg8, %arg10, %arg6, %arg9)
                %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%8], sizes: [64, 64], strides: [512, 1] {tmd.reuse = {spatial = [{depth = 2 : i64, iterator = "%arg8", mapped_to = "x", reuse_type = "no_reuse", volume = 0 : i64}, {depth = 3 : i64, iterator = "%arg9", mapped_to = "y", reuse_type = "no_reuse", volume = 0 : i64}], temporal = [{depth = 0 : i64, iterator = "%arg6", reuse_type = "no_reuse", volume = 0 : i64}, {depth = 1 : i64, iterator = "%arg7", reuse_type = "no_reuse", volume = 0 : i64}, {depth = 4 : i64, iterator = "%arg10", reuse_type = "no_reuse", volume = 0 : i64}, {depth = 5 : i64, iterator = "%arg11", reuse_type = "no_reuse", volume = 0 : i64}]}} : memref<*xf32> to memref<64x64xf32, strided<[512, 1], offset: ?>>
                memref.copy %7, %reinterpret_cast : memref<64x64xf32> to memref<64x64xf32, strided<[512, 1], offset: ?>>
              }
            }
          } {tmd.mapped_to = "y"}
        } {tmd.mapped_to = "x"}
      }
    }
    return
  }
  func.func @matmul_kernel__d1i0_d0i1__f10__c0mem_c1mem(%arg0: memref<*xf32> {tt.divisibility = 16 : i32}, %arg1: memref<*xf32> {tt.divisibility = 16 : i32}, %arg2: memref<*xf32> {tt.divisibility = 16 : i32}, %arg3: index, %arg4: index, %arg5: index) {
    affine.for %arg6 = 0 to affine_map<(d0, d1) -> (d1 ceildiv 8)>(%arg3, %arg4) {
      affine.for %arg7 = 0 to affine_map<(d0, d1) -> (d0 ceildiv 8)>(%arg3, %arg4) {
        affine.parallel (%arg8) = (0) to (8) {
          affine.parallel (%arg9) = (0) to (8) {
            %cst = arith.constant 0.000000e+00 : f32
            %alloc = memref.alloc() {alignment = 64 : i64} : memref<64x64xf32>
            linalg.fill ins(%cst : f32) outs(%alloc : memref<64x64xf32>)
            %c0 = arith.constant 0 : index
            %c8 = arith.constant 8 : index
            %c1 = arith.constant 1 : index
            affine.for %arg10 = 0 to affine_map<(d0) -> (d0 ceildiv 64)>(%arg3) {
              affine.for %arg11 = 0 to affine_map<(d0) -> (d0 ceildiv 64)>(%arg4) {
                %alloc_0 = memref.alloc() {alignment = 64 : i64} : memref<64x64xf32>
                memref.copy %alloc, %alloc_0 : memref<64x64xf32> to memref<64x64xf32>
                %7 = scf.for %arg12 = %c0 to %c8 step %c1 iter_args(%arg13 = %alloc_0) -> (memref<64x64xf32>) {
                  %9 = affine.apply affine_map<(d0, d1, d2, d3) -> (d0 * 64 + d1 * 2097152 + d2 * 262144 + d3 * 32768)>(%arg12, %arg10, %arg7, %arg9)
                  %reinterpret_cast_1 = memref.reinterpret_cast %arg0 to offset: [%9], sizes: [64, 64], strides: [512, 1] {tmd.reuse = {sequential = [{depth = 6 : i64, iterator = "%arg12", reuse_type = "no_reuse", volume = 0 : i64}], spatial = [{depth = 2 : i64, iterator = "%arg8", mapped_to = "x", reuse_type = "total_reuse", volume = 16384 : i64}, {depth = 3 : i64, iterator = "%arg9", mapped_to = "y", reuse_type = "no_reuse", volume = 0 : i64}], temporal = [{depth = 0 : i64, iterator = "%arg6", reuse_type = "total_reuse", volume = 16384 : i64}, {depth = 1 : i64, iterator = "%arg7", reuse_type = "no_reuse", volume = 0 : i64}, {depth = 4 : i64, iterator = "%arg10", reuse_type = "no_reuse", volume = 0 : i64}, {depth = 5 : i64, iterator = "%arg11", reuse_type = "total_reuse", volume = 16384 : i64}]}} : memref<*xf32> to memref<64x64xf32, strided<[512, 1], offset: ?>>
                  %alloc_2 = memref.alloc() {tmd.alloc = {local = true, memory_name = "L1", size = 16384 : i64}} : memref<64x64xf32>
                  memref.copy %reinterpret_cast_1, %alloc_2 {tmd.copy.choice = {kind = "mem", memory_name = "L1"}} : memref<64x64xf32, strided<[512, 1], offset: ?>> to memref<64x64xf32>
                  %10 = affine.apply affine_map<(d0, d1, d2, d3) -> (d0 * 32768 + d1 * 4096 + d2 * 512 + d3 * 64)>(%arg12, %arg11, %arg6, %arg8)
                  %reinterpret_cast_3 = memref.reinterpret_cast %arg1 to offset: [%10], sizes: [64, 64], strides: [512, 1] {tmd.reuse = {sequential = [{depth = 6 : i64, iterator = "%arg12", reuse_type = "no_reuse", volume = 0 : i64}], spatial = [{depth = 2 : i64, iterator = "%arg8", mapped_to = "x", reuse_type = "no_reuse", volume = 0 : i64}, {depth = 3 : i64, iterator = "%arg9", mapped_to = "y", reuse_type = "total_reuse", volume = 16384 : i64}], temporal = [{depth = 0 : i64, iterator = "%arg6", reuse_type = "no_reuse", volume = 0 : i64}, {depth = 1 : i64, iterator = "%arg7", reuse_type = "total_reuse", volume = 16384 : i64}, {depth = 4 : i64, iterator = "%arg10", reuse_type = "total_reuse", volume = 16384 : i64}, {depth = 5 : i64, iterator = "%arg11", reuse_type = "no_reuse", volume = 0 : i64}]}} : memref<*xf32> to memref<64x64xf32, strided<[512, 1], offset: ?>>
                  %alloc_4 = memref.alloc() {tmd.alloc = {local = true, memory_name = "L1", size = 16384 : i64}} : memref<64x64xf32>
                  memref.copy %reinterpret_cast_3, %alloc_4 {tmd.copy.choice = {kind = "mem", memory_name = "L1"}} : memref<64x64xf32, strided<[512, 1], offset: ?>> to memref<64x64xf32>
                  %alloc_5 = memref.alloc() {alignment = 64 : i64} : memref<64x64xf32>
                  memref.copy %alloc, %alloc_5 : memref<64x64xf32> to memref<64x64xf32>
                  %c32 = arith.constant 32 : index
                  %alloc_6 = memref.alloc() {tmd.alloc = {local = true, memory_name = "L1", size = 16384 : i64}} : memref<64x64xf32>
                  %alloc_7 = memref.alloc() {tmd.alloc = {local = true, memory_name = "L1", size = 16384 : i64}} : memref<64x64xf32>
                  memref.copy %alloc_2, %alloc_6 : memref<64x64xf32> to memref<64x64xf32>
                  memref.copy %alloc_4, %alloc_7 : memref<64x64xf32> to memref<64x64xf32>
                  affine.for %arg14 = 0 to 2 {
                    affine.for %arg15 = 0 to 2 {
                      %11 = affine.apply affine_map<(d0) -> (d0 * 32)>(%arg14)
                      %12 = affine.apply affine_map<(d0) -> (d0 * 32)>(%arg15)
                      %alloc_8 = memref.alloc() : memref<32x32xf32>
                      %subview = memref.subview %arg13[%11, %12] [%c32, %c32] [%c1, %c1] : memref<64x64xf32> to memref<?x?xf32, strided<[?, ?], offset: ?>>
                      memref.copy %subview, %alloc_8 : memref<?x?xf32, strided<[?, ?], offset: ?>> to memref<32x32xf32>
                      affine.for %arg16 = 0 to 2 {
                        %13 = affine.apply affine_map<(d0) -> (d0 * 32)>(%arg16)
                        %subview_9 = memref.subview %alloc_6[%11, %13] [%c32, %c32] [%c1, %c1] : memref<64x64xf32> to memref<?x?xf32, strided<[?, ?], offset: ?>>
                        %subview_10 = memref.subview %alloc_7[%13, %12] [%c32, %c32] [%c1, %c1] : memref<64x64xf32> to memref<?x?xf32, strided<[?, ?], offset: ?>>
                        linalg.matmul ins(%subview_9, %subview_10 : memref<?x?xf32, strided<[?, ?], offset: ?>>, memref<?x?xf32, strided<[?, ?], offset: ?>>) outs(%alloc_8 : memref<32x32xf32>)
                      }
                      memref.copy %alloc_8, %subview : memref<32x32xf32> to memref<?x?xf32, strided<[?, ?], offset: ?>>
                    }
                  }
                  linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%arg13, %alloc_5 : memref<64x64xf32>, memref<64x64xf32>) outs(%arg13 : memref<64x64xf32>) {
                  ^bb0(%in: f32, %in_8: f32, %out: f32):
                    %11 = arith.addf %in, %in_8 : f32
                    linalg.yield %11 : f32
                  }
                  scf.yield %arg13 : memref<64x64xf32>
                }
                %8 = affine.apply affine_map<(d0, d1, d2, d3, d4, d5) -> (d0 * 4096 + d1 * 512 + d2 * 64 + d3 * 2097152 + d4 * 262144 + d5 * 32768)>(%arg11, %arg6, %arg8, %arg10, %arg7, %arg9)
                %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%8], sizes: [64, 64], strides: [512, 1] {tmd.reuse = {spatial = [{depth = 2 : i64, iterator = "%arg8", mapped_to = "x", reuse_type = "no_reuse", volume = 0 : i64}, {depth = 3 : i64, iterator = "%arg9", mapped_to = "y", reuse_type = "no_reuse", volume = 0 : i64}], temporal = [{depth = 0 : i64, iterator = "%arg6", reuse_type = "no_reuse", volume = 0 : i64}, {depth = 1 : i64, iterator = "%arg7", reuse_type = "no_reuse", volume = 0 : i64}, {depth = 4 : i64, iterator = "%arg10", reuse_type = "no_reuse", volume = 0 : i64}, {depth = 5 : i64, iterator = "%arg11", reuse_type = "no_reuse", volume = 0 : i64}]}} : memref<*xf32> to memref<64x64xf32, strided<[512, 1], offset: ?>>
                memref.copy %7, %reinterpret_cast : memref<64x64xf32> to memref<64x64xf32, strided<[512, 1], offset: ?>>
              }
            }
          } {tmd.mapped_to = "y"}
        } {tmd.mapped_to = "x"}
      }
    }
    return
  }
  func.func @matmul_kernel__d1i0_d0i1__f10__c0bx_c1mem(%arg0: memref<*xf32> {tt.divisibility = 16 : i32}, %arg1: memref<*xf32> {tt.divisibility = 16 : i32}, %arg2: memref<*xf32> {tt.divisibility = 16 : i32}, %arg3: index, %arg4: index, %arg5: index) {
    affine.for %arg6 = 0 to affine_map<(d0, d1) -> (d1 ceildiv 8)>(%arg3, %arg4) {
      affine.for %arg7 = 0 to affine_map<(d0, d1) -> (d0 ceildiv 8)>(%arg3, %arg4) {
        affine.parallel (%arg8) = (0) to (8) {
          affine.parallel (%arg9) = (0) to (8) {
            %cst = arith.constant 0.000000e+00 : f32
            %alloc = memref.alloc() {alignment = 64 : i64} : memref<64x64xf32>
            linalg.fill ins(%cst : f32) outs(%alloc : memref<64x64xf32>)
            %c0 = arith.constant 0 : index
            %c8 = arith.constant 8 : index
            %c1 = arith.constant 1 : index
            affine.for %arg10 = 0 to affine_map<(d0) -> (d0 ceildiv 64)>(%arg3) {
              affine.for %arg11 = 0 to affine_map<(d0) -> (d0 ceildiv 64)>(%arg4) {
                %alloc_0 = memref.alloc() {alignment = 64 : i64} : memref<64x64xf32>
                memref.copy %alloc, %alloc_0 : memref<64x64xf32> to memref<64x64xf32>
                %7 = scf.for %arg12 = %c0 to %c8 step %c1 iter_args(%arg13 = %alloc_0) -> (memref<64x64xf32>) {
                  %9 = affine.apply affine_map<(d0, d1, d2, d3) -> (d0 * 64 + d1 * 2097152 + d2 * 262144 + d3 * 32768)>(%arg12, %arg10, %arg7, %arg9)
                  %reinterpret_cast_1 = memref.reinterpret_cast %arg0 to offset: [%9], sizes: [64, 64], strides: [512, 1] {tmd.reuse = {sequential = [{depth = 6 : i64, iterator = "%arg12", reuse_type = "no_reuse", volume = 0 : i64}], spatial = [{depth = 2 : i64, iterator = "%arg8", mapped_to = "x", reuse_type = "total_reuse", volume = 16384 : i64}, {depth = 3 : i64, iterator = "%arg9", mapped_to = "y", reuse_type = "no_reuse", volume = 0 : i64}], temporal = [{depth = 0 : i64, iterator = "%arg6", reuse_type = "total_reuse", volume = 16384 : i64}, {depth = 1 : i64, iterator = "%arg7", reuse_type = "no_reuse", volume = 0 : i64}, {depth = 4 : i64, iterator = "%arg10", reuse_type = "no_reuse", volume = 0 : i64}, {depth = 5 : i64, iterator = "%arg11", reuse_type = "total_reuse", volume = 16384 : i64}]}} : memref<*xf32> to memref<64x64xf32, strided<[512, 1], offset: ?>>
                  %alloc_2 = memref.alloc() {tmd.alloc = {local = true, memory_name = "L1", size = 16384 : i64}} : memref<64x64xf32>
                  memref.copy %reinterpret_cast_1, %alloc_2 {tmd.copy.choice = {dim = "x", interconnect_name = "horizontal_links", kind = "broadcast"}} : memref<64x64xf32, strided<[512, 1], offset: ?>> to memref<64x64xf32>
                  %10 = affine.apply affine_map<(d0, d1, d2, d3) -> (d0 * 32768 + d1 * 4096 + d2 * 512 + d3 * 64)>(%arg12, %arg11, %arg6, %arg8)
                  %reinterpret_cast_3 = memref.reinterpret_cast %arg1 to offset: [%10], sizes: [64, 64], strides: [512, 1] {tmd.reuse = {sequential = [{depth = 6 : i64, iterator = "%arg12", reuse_type = "no_reuse", volume = 0 : i64}], spatial = [{depth = 2 : i64, iterator = "%arg8", mapped_to = "x", reuse_type = "no_reuse", volume = 0 : i64}, {depth = 3 : i64, iterator = "%arg9", mapped_to = "y", reuse_type = "total_reuse", volume = 16384 : i64}], temporal = [{depth = 0 : i64, iterator = "%arg6", reuse_type = "no_reuse", volume = 0 : i64}, {depth = 1 : i64, iterator = "%arg7", reuse_type = "total_reuse", volume = 16384 : i64}, {depth = 4 : i64, iterator = "%arg10", reuse_type = "total_reuse", volume = 16384 : i64}, {depth = 5 : i64, iterator = "%arg11", reuse_type = "no_reuse", volume = 0 : i64}]}} : memref<*xf32> to memref<64x64xf32, strided<[512, 1], offset: ?>>
                  %alloc_4 = memref.alloc() {tmd.alloc = {local = true, memory_name = "L1", size = 16384 : i64}} : memref<64x64xf32>
                  memref.copy %reinterpret_cast_3, %alloc_4 {tmd.copy.choice = {kind = "mem", memory_name = "L1"}} : memref<64x64xf32, strided<[512, 1], offset: ?>> to memref<64x64xf32>
                  %alloc_5 = memref.alloc() {alignment = 64 : i64} : memref<64x64xf32>
                  memref.copy %alloc, %alloc_5 : memref<64x64xf32> to memref<64x64xf32>
                  %c32 = arith.constant 32 : index
                  %alloc_6 = memref.alloc() {tmd.alloc = {local = true, memory_name = "L1", size = 16384 : i64}} : memref<64x64xf32>
                  %alloc_7 = memref.alloc() {tmd.alloc = {local = true, memory_name = "L1", size = 16384 : i64}} : memref<64x64xf32>
                  memref.copy %alloc_2, %alloc_6 : memref<64x64xf32> to memref<64x64xf32>
                  memref.copy %alloc_4, %alloc_7 : memref<64x64xf32> to memref<64x64xf32>
                  affine.for %arg14 = 0 to 2 {
                    affine.for %arg15 = 0 to 2 {
                      %11 = affine.apply affine_map<(d0) -> (d0 * 32)>(%arg14)
                      %12 = affine.apply affine_map<(d0) -> (d0 * 32)>(%arg15)
                      %alloc_8 = memref.alloc() : memref<32x32xf32>
                      %subview = memref.subview %arg13[%11, %12] [%c32, %c32] [%c1, %c1] : memref<64x64xf32> to memref<?x?xf32, strided<[?, ?], offset: ?>>
                      memref.copy %subview, %alloc_8 : memref<?x?xf32, strided<[?, ?], offset: ?>> to memref<32x32xf32>
                      affine.for %arg16 = 0 to 2 {
                        %13 = affine.apply affine_map<(d0) -> (d0 * 32)>(%arg16)
                        %subview_9 = memref.subview %alloc_6[%11, %13] [%c32, %c32] [%c1, %c1] : memref<64x64xf32> to memref<?x?xf32, strided<[?, ?], offset: ?>>
                        %subview_10 = memref.subview %alloc_7[%13, %12] [%c32, %c32] [%c1, %c1] : memref<64x64xf32> to memref<?x?xf32, strided<[?, ?], offset: ?>>
                        linalg.matmul ins(%subview_9, %subview_10 : memref<?x?xf32, strided<[?, ?], offset: ?>>, memref<?x?xf32, strided<[?, ?], offset: ?>>) outs(%alloc_8 : memref<32x32xf32>)
                      }
                      memref.copy %alloc_8, %subview : memref<32x32xf32> to memref<?x?xf32, strided<[?, ?], offset: ?>>
                    }
                  }
                  linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%arg13, %alloc_5 : memref<64x64xf32>, memref<64x64xf32>) outs(%arg13 : memref<64x64xf32>) {
                  ^bb0(%in: f32, %in_8: f32, %out: f32):
                    %11 = arith.addf %in, %in_8 : f32
                    linalg.yield %11 : f32
                  }
                  scf.yield %arg13 : memref<64x64xf32>
                }
                %8 = affine.apply affine_map<(d0, d1, d2, d3, d4, d5) -> (d0 * 4096 + d1 * 512 + d2 * 64 + d3 * 2097152 + d4 * 262144 + d5 * 32768)>(%arg11, %arg6, %arg8, %arg10, %arg7, %arg9)
                %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%8], sizes: [64, 64], strides: [512, 1] {tmd.reuse = {spatial = [{depth = 2 : i64, iterator = "%arg8", mapped_to = "x", reuse_type = "no_reuse", volume = 0 : i64}, {depth = 3 : i64, iterator = "%arg9", mapped_to = "y", reuse_type = "no_reuse", volume = 0 : i64}], temporal = [{depth = 0 : i64, iterator = "%arg6", reuse_type = "no_reuse", volume = 0 : i64}, {depth = 1 : i64, iterator = "%arg7", reuse_type = "no_reuse", volume = 0 : i64}, {depth = 4 : i64, iterator = "%arg10", reuse_type = "no_reuse", volume = 0 : i64}, {depth = 5 : i64, iterator = "%arg11", reuse_type = "no_reuse", volume = 0 : i64}]}} : memref<*xf32> to memref<64x64xf32, strided<[512, 1], offset: ?>>
                memref.copy %7, %reinterpret_cast : memref<64x64xf32> to memref<64x64xf32, strided<[512, 1], offset: ?>>
              }
            }
          } {tmd.mapped_to = "y"}
        } {tmd.mapped_to = "x"}
      }
    }
    return
  }
  func.func @matmul_kernel__d1i0_d0i1__f10__c0mem_c1by(%arg0: memref<*xf32> {tt.divisibility = 16 : i32}, %arg1: memref<*xf32> {tt.divisibility = 16 : i32}, %arg2: memref<*xf32> {tt.divisibility = 16 : i32}, %arg3: index, %arg4: index, %arg5: index) {
    affine.for %arg6 = 0 to affine_map<(d0, d1) -> (d1 ceildiv 8)>(%arg3, %arg4) {
      affine.for %arg7 = 0 to affine_map<(d0, d1) -> (d0 ceildiv 8)>(%arg3, %arg4) {
        affine.parallel (%arg8) = (0) to (8) {
          affine.parallel (%arg9) = (0) to (8) {
            %cst = arith.constant 0.000000e+00 : f32
            %alloc = memref.alloc() {alignment = 64 : i64} : memref<64x64xf32>
            linalg.fill ins(%cst : f32) outs(%alloc : memref<64x64xf32>)
            %c0 = arith.constant 0 : index
            %c8 = arith.constant 8 : index
            %c1 = arith.constant 1 : index
            affine.for %arg10 = 0 to affine_map<(d0) -> (d0 ceildiv 64)>(%arg3) {
              affine.for %arg11 = 0 to affine_map<(d0) -> (d0 ceildiv 64)>(%arg4) {
                %alloc_0 = memref.alloc() {alignment = 64 : i64} : memref<64x64xf32>
                memref.copy %alloc, %alloc_0 : memref<64x64xf32> to memref<64x64xf32>
                %7 = scf.for %arg12 = %c0 to %c8 step %c1 iter_args(%arg13 = %alloc_0) -> (memref<64x64xf32>) {
                  %9 = affine.apply affine_map<(d0, d1, d2, d3) -> (d0 * 64 + d1 * 2097152 + d2 * 262144 + d3 * 32768)>(%arg12, %arg10, %arg7, %arg9)
                  %reinterpret_cast_1 = memref.reinterpret_cast %arg0 to offset: [%9], sizes: [64, 64], strides: [512, 1] {tmd.reuse = {sequential = [{depth = 6 : i64, iterator = "%arg12", reuse_type = "no_reuse", volume = 0 : i64}], spatial = [{depth = 2 : i64, iterator = "%arg8", mapped_to = "x", reuse_type = "total_reuse", volume = 16384 : i64}, {depth = 3 : i64, iterator = "%arg9", mapped_to = "y", reuse_type = "no_reuse", volume = 0 : i64}], temporal = [{depth = 0 : i64, iterator = "%arg6", reuse_type = "total_reuse", volume = 16384 : i64}, {depth = 1 : i64, iterator = "%arg7", reuse_type = "no_reuse", volume = 0 : i64}, {depth = 4 : i64, iterator = "%arg10", reuse_type = "no_reuse", volume = 0 : i64}, {depth = 5 : i64, iterator = "%arg11", reuse_type = "total_reuse", volume = 16384 : i64}]}} : memref<*xf32> to memref<64x64xf32, strided<[512, 1], offset: ?>>
                  %alloc_2 = memref.alloc() {tmd.alloc = {local = true, memory_name = "L1", size = 16384 : i64}} : memref<64x64xf32>
                  memref.copy %reinterpret_cast_1, %alloc_2 {tmd.copy.choice = {kind = "mem", memory_name = "L1"}} : memref<64x64xf32, strided<[512, 1], offset: ?>> to memref<64x64xf32>
                  %10 = affine.apply affine_map<(d0, d1, d2, d3) -> (d0 * 32768 + d1 * 4096 + d2 * 512 + d3 * 64)>(%arg12, %arg11, %arg6, %arg8)
                  %reinterpret_cast_3 = memref.reinterpret_cast %arg1 to offset: [%10], sizes: [64, 64], strides: [512, 1] {tmd.reuse = {sequential = [{depth = 6 : i64, iterator = "%arg12", reuse_type = "no_reuse", volume = 0 : i64}], spatial = [{depth = 2 : i64, iterator = "%arg8", mapped_to = "x", reuse_type = "no_reuse", volume = 0 : i64}, {depth = 3 : i64, iterator = "%arg9", mapped_to = "y", reuse_type = "total_reuse", volume = 16384 : i64}], temporal = [{depth = 0 : i64, iterator = "%arg6", reuse_type = "no_reuse", volume = 0 : i64}, {depth = 1 : i64, iterator = "%arg7", reuse_type = "total_reuse", volume = 16384 : i64}, {depth = 4 : i64, iterator = "%arg10", reuse_type = "total_reuse", volume = 16384 : i64}, {depth = 5 : i64, iterator = "%arg11", reuse_type = "no_reuse", volume = 0 : i64}]}} : memref<*xf32> to memref<64x64xf32, strided<[512, 1], offset: ?>>
                  %alloc_4 = memref.alloc() {tmd.alloc = {local = true, memory_name = "L1", size = 16384 : i64}} : memref<64x64xf32>
                  memref.copy %reinterpret_cast_3, %alloc_4 {tmd.copy.choice = {dim = "y", interconnect_name = "vertical_links", kind = "broadcast"}} : memref<64x64xf32, strided<[512, 1], offset: ?>> to memref<64x64xf32>
                  %alloc_5 = memref.alloc() {alignment = 64 : i64} : memref<64x64xf32>
                  memref.copy %alloc, %alloc_5 : memref<64x64xf32> to memref<64x64xf32>
                  %c32 = arith.constant 32 : index
                  %alloc_6 = memref.alloc() {tmd.alloc = {local = true, memory_name = "L1", size = 16384 : i64}} : memref<64x64xf32>
                  %alloc_7 = memref.alloc() {tmd.alloc = {local = true, memory_name = "L1", size = 16384 : i64}} : memref<64x64xf32>
                  memref.copy %alloc_2, %alloc_6 : memref<64x64xf32> to memref<64x64xf32>
                  memref.copy %alloc_4, %alloc_7 : memref<64x64xf32> to memref<64x64xf32>
                  affine.for %arg14 = 0 to 2 {
                    affine.for %arg15 = 0 to 2 {
                      %11 = affine.apply affine_map<(d0) -> (d0 * 32)>(%arg14)
                      %12 = affine.apply affine_map<(d0) -> (d0 * 32)>(%arg15)
                      %alloc_8 = memref.alloc() : memref<32x32xf32>
                      %subview = memref.subview %arg13[%11, %12] [%c32, %c32] [%c1, %c1] : memref<64x64xf32> to memref<?x?xf32, strided<[?, ?], offset: ?>>
                      memref.copy %subview, %alloc_8 : memref<?x?xf32, strided<[?, ?], offset: ?>> to memref<32x32xf32>
                      affine.for %arg16 = 0 to 2 {
                        %13 = affine.apply affine_map<(d0) -> (d0 * 32)>(%arg16)
                        %subview_9 = memref.subview %alloc_6[%11, %13] [%c32, %c32] [%c1, %c1] : memref<64x64xf32> to memref<?x?xf32, strided<[?, ?], offset: ?>>
                        %subview_10 = memref.subview %alloc_7[%13, %12] [%c32, %c32] [%c1, %c1] : memref<64x64xf32> to memref<?x?xf32, strided<[?, ?], offset: ?>>
                        linalg.matmul ins(%subview_9, %subview_10 : memref<?x?xf32, strided<[?, ?], offset: ?>>, memref<?x?xf32, strided<[?, ?], offset: ?>>) outs(%alloc_8 : memref<32x32xf32>)
                      }
                      memref.copy %alloc_8, %subview : memref<32x32xf32> to memref<?x?xf32, strided<[?, ?], offset: ?>>
                    }
                  }
                  linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%arg13, %alloc_5 : memref<64x64xf32>, memref<64x64xf32>) outs(%arg13 : memref<64x64xf32>) {
                  ^bb0(%in: f32, %in_8: f32, %out: f32):
                    %11 = arith.addf %in, %in_8 : f32
                    linalg.yield %11 : f32
                  }
                  scf.yield %arg13 : memref<64x64xf32>
                }
                %8 = affine.apply affine_map<(d0, d1, d2, d3, d4, d5) -> (d0 * 4096 + d1 * 512 + d2 * 64 + d3 * 2097152 + d4 * 262144 + d5 * 32768)>(%arg11, %arg6, %arg8, %arg10, %arg7, %arg9)
                %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%8], sizes: [64, 64], strides: [512, 1] {tmd.reuse = {spatial = [{depth = 2 : i64, iterator = "%arg8", mapped_to = "x", reuse_type = "no_reuse", volume = 0 : i64}, {depth = 3 : i64, iterator = "%arg9", mapped_to = "y", reuse_type = "no_reuse", volume = 0 : i64}], temporal = [{depth = 0 : i64, iterator = "%arg6", reuse_type = "no_reuse", volume = 0 : i64}, {depth = 1 : i64, iterator = "%arg7", reuse_type = "no_reuse", volume = 0 : i64}, {depth = 4 : i64, iterator = "%arg10", reuse_type = "no_reuse", volume = 0 : i64}, {depth = 5 : i64, iterator = "%arg11", reuse_type = "no_reuse", volume = 0 : i64}]}} : memref<*xf32> to memref<64x64xf32, strided<[512, 1], offset: ?>>
                memref.copy %7, %reinterpret_cast : memref<64x64xf32> to memref<64x64xf32, strided<[512, 1], offset: ?>>
              }
            }
          } {tmd.mapped_to = "y"}
        } {tmd.mapped_to = "x"}
      }
    }
    return
  }
  func.func @matmul_kernel__d1i0_d0i1__f10__c0bx_c1by(%arg0: memref<*xf32> {tt.divisibility = 16 : i32}, %arg1: memref<*xf32> {tt.divisibility = 16 : i32}, %arg2: memref<*xf32> {tt.divisibility = 16 : i32}, %arg3: index, %arg4: index, %arg5: index) {
    affine.for %arg6 = 0 to affine_map<(d0, d1) -> (d1 ceildiv 8)>(%arg3, %arg4) {
      affine.for %arg7 = 0 to affine_map<(d0, d1) -> (d0 ceildiv 8)>(%arg3, %arg4) {
        affine.parallel (%arg8) = (0) to (8) {
          affine.parallel (%arg9) = (0) to (8) {
            %cst = arith.constant 0.000000e+00 : f32
            %alloc = memref.alloc() {alignment = 64 : i64} : memref<64x64xf32>
            linalg.fill ins(%cst : f32) outs(%alloc : memref<64x64xf32>)
            %c0 = arith.constant 0 : index
            %c8 = arith.constant 8 : index
            %c1 = arith.constant 1 : index
            affine.for %arg10 = 0 to affine_map<(d0) -> (d0 ceildiv 64)>(%arg3) {
              affine.for %arg11 = 0 to affine_map<(d0) -> (d0 ceildiv 64)>(%arg4) {
                %alloc_0 = memref.alloc() {alignment = 64 : i64} : memref<64x64xf32>
                memref.copy %alloc, %alloc_0 : memref<64x64xf32> to memref<64x64xf32>
                %7 = scf.for %arg12 = %c0 to %c8 step %c1 iter_args(%arg13 = %alloc_0) -> (memref<64x64xf32>) {
                  %9 = affine.apply affine_map<(d0, d1, d2, d3) -> (d0 * 64 + d1 * 2097152 + d2 * 262144 + d3 * 32768)>(%arg12, %arg10, %arg7, %arg9)
                  %reinterpret_cast_1 = memref.reinterpret_cast %arg0 to offset: [%9], sizes: [64, 64], strides: [512, 1] {tmd.reuse = {sequential = [{depth = 6 : i64, iterator = "%arg12", reuse_type = "no_reuse", volume = 0 : i64}], spatial = [{depth = 2 : i64, iterator = "%arg8", mapped_to = "x", reuse_type = "total_reuse", volume = 16384 : i64}, {depth = 3 : i64, iterator = "%arg9", mapped_to = "y", reuse_type = "no_reuse", volume = 0 : i64}], temporal = [{depth = 0 : i64, iterator = "%arg6", reuse_type = "total_reuse", volume = 16384 : i64}, {depth = 1 : i64, iterator = "%arg7", reuse_type = "no_reuse", volume = 0 : i64}, {depth = 4 : i64, iterator = "%arg10", reuse_type = "no_reuse", volume = 0 : i64}, {depth = 5 : i64, iterator = "%arg11", reuse_type = "total_reuse", volume = 16384 : i64}]}} : memref<*xf32> to memref<64x64xf32, strided<[512, 1], offset: ?>>
                  %alloc_2 = memref.alloc() {tmd.alloc = {local = true, memory_name = "L1", size = 16384 : i64}} : memref<64x64xf32>
                  memref.copy %reinterpret_cast_1, %alloc_2 {tmd.copy.choice = {dim = "x", interconnect_name = "horizontal_links", kind = "broadcast"}} : memref<64x64xf32, strided<[512, 1], offset: ?>> to memref<64x64xf32>
                  %10 = affine.apply affine_map<(d0, d1, d2, d3) -> (d0 * 32768 + d1 * 4096 + d2 * 512 + d3 * 64)>(%arg12, %arg11, %arg6, %arg8)
                  %reinterpret_cast_3 = memref.reinterpret_cast %arg1 to offset: [%10], sizes: [64, 64], strides: [512, 1] {tmd.reuse = {sequential = [{depth = 6 : i64, iterator = "%arg12", reuse_type = "no_reuse", volume = 0 : i64}], spatial = [{depth = 2 : i64, iterator = "%arg8", mapped_to = "x", reuse_type = "no_reuse", volume = 0 : i64}, {depth = 3 : i64, iterator = "%arg9", mapped_to = "y", reuse_type = "total_reuse", volume = 16384 : i64}], temporal = [{depth = 0 : i64, iterator = "%arg6", reuse_type = "no_reuse", volume = 0 : i64}, {depth = 1 : i64, iterator = "%arg7", reuse_type = "total_reuse", volume = 16384 : i64}, {depth = 4 : i64, iterator = "%arg10", reuse_type = "total_reuse", volume = 16384 : i64}, {depth = 5 : i64, iterator = "%arg11", reuse_type = "no_reuse", volume = 0 : i64}]}} : memref<*xf32> to memref<64x64xf32, strided<[512, 1], offset: ?>>
                  %alloc_4 = memref.alloc() {tmd.alloc = {local = true, memory_name = "L1", size = 16384 : i64}} : memref<64x64xf32>
                  memref.copy %reinterpret_cast_3, %alloc_4 {tmd.copy.choice = {dim = "y", interconnect_name = "vertical_links", kind = "broadcast"}} : memref<64x64xf32, strided<[512, 1], offset: ?>> to memref<64x64xf32>
                  %alloc_5 = memref.alloc() {alignment = 64 : i64} : memref<64x64xf32>
                  memref.copy %alloc, %alloc_5 : memref<64x64xf32> to memref<64x64xf32>
                  %c32 = arith.constant 32 : index
                  %alloc_6 = memref.alloc() {tmd.alloc = {local = true, memory_name = "L1", size = 16384 : i64}} : memref<64x64xf32>
                  %alloc_7 = memref.alloc() {tmd.alloc = {local = true, memory_name = "L1", size = 16384 : i64}} : memref<64x64xf32>
                  memref.copy %alloc_2, %alloc_6 : memref<64x64xf32> to memref<64x64xf32>
                  memref.copy %alloc_4, %alloc_7 : memref<64x64xf32> to memref<64x64xf32>
                  affine.for %arg14 = 0 to 2 {
                    affine.for %arg15 = 0 to 2 {
                      %11 = affine.apply affine_map<(d0) -> (d0 * 32)>(%arg14)
                      %12 = affine.apply affine_map<(d0) -> (d0 * 32)>(%arg15)
                      %alloc_8 = memref.alloc() : memref<32x32xf32>
                      %subview = memref.subview %arg13[%11, %12] [%c32, %c32] [%c1, %c1] : memref<64x64xf32> to memref<?x?xf32, strided<[?, ?], offset: ?>>
                      memref.copy %subview, %alloc_8 : memref<?x?xf32, strided<[?, ?], offset: ?>> to memref<32x32xf32>
                      affine.for %arg16 = 0 to 2 {
                        %13 = affine.apply affine_map<(d0) -> (d0 * 32)>(%arg16)
                        %subview_9 = memref.subview %alloc_6[%11, %13] [%c32, %c32] [%c1, %c1] : memref<64x64xf32> to memref<?x?xf32, strided<[?, ?], offset: ?>>
                        %subview_10 = memref.subview %alloc_7[%13, %12] [%c32, %c32] [%c1, %c1] : memref<64x64xf32> to memref<?x?xf32, strided<[?, ?], offset: ?>>
                        linalg.matmul ins(%subview_9, %subview_10 : memref<?x?xf32, strided<[?, ?], offset: ?>>, memref<?x?xf32, strided<[?, ?], offset: ?>>) outs(%alloc_8 : memref<32x32xf32>)
                      }
                      memref.copy %alloc_8, %subview : memref<32x32xf32> to memref<?x?xf32, strided<[?, ?], offset: ?>>
                    }
                  }
                  linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%arg13, %alloc_5 : memref<64x64xf32>, memref<64x64xf32>) outs(%arg13 : memref<64x64xf32>) {
                  ^bb0(%in: f32, %in_8: f32, %out: f32):
                    %11 = arith.addf %in, %in_8 : f32
                    linalg.yield %11 : f32
                  }
                  scf.yield %arg13 : memref<64x64xf32>
                }
                %8 = affine.apply affine_map<(d0, d1, d2, d3, d4, d5) -> (d0 * 4096 + d1 * 512 + d2 * 64 + d3 * 2097152 + d4 * 262144 + d5 * 32768)>(%arg11, %arg6, %arg8, %arg10, %arg7, %arg9)
                %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%8], sizes: [64, 64], strides: [512, 1] {tmd.reuse = {spatial = [{depth = 2 : i64, iterator = "%arg8", mapped_to = "x", reuse_type = "no_reuse", volume = 0 : i64}, {depth = 3 : i64, iterator = "%arg9", mapped_to = "y", reuse_type = "no_reuse", volume = 0 : i64}], temporal = [{depth = 0 : i64, iterator = "%arg6", reuse_type = "no_reuse", volume = 0 : i64}, {depth = 1 : i64, iterator = "%arg7", reuse_type = "no_reuse", volume = 0 : i64}, {depth = 4 : i64, iterator = "%arg10", reuse_type = "no_reuse", volume = 0 : i64}, {depth = 5 : i64, iterator = "%arg11", reuse_type = "no_reuse", volume = 0 : i64}]}} : memref<*xf32> to memref<64x64xf32, strided<[512, 1], offset: ?>>
                memref.copy %7, %reinterpret_cast : memref<64x64xf32> to memref<64x64xf32, strided<[512, 1], offset: ?>>
              }
            }
          } {tmd.mapped_to = "y"}
        } {tmd.mapped_to = "x"}
      }
    }
    return
  }
  func.func @matmul_kernel__d0i1_d1i1__f01__c0mem_c1mem(%arg0: memref<*xf32> {tt.divisibility = 16 : i32}, %arg1: memref<*xf32> {tt.divisibility = 16 : i32}, %arg2: memref<*xf32> {tt.divisibility = 16 : i32}, %arg3: index, %arg4: index, %arg5: index) {
    affine.for %arg6 = 0 to affine_map<(d0, d1) -> (d0)>(%arg3, %arg4) {
      affine.for %arg7 = 0 to affine_map<(d0, d1) -> ((d1 ceildiv 8) ceildiv 8)>(%arg3, %arg4) {
        affine.parallel (%arg8) = (0) to (8) {
          affine.parallel (%arg9) = (0) to (8) {
            %cst = arith.constant 0.000000e+00 : f32
            %alloc = memref.alloc() {alignment = 64 : i64} : memref<64x64xf32>
            linalg.fill ins(%cst : f32) outs(%alloc : memref<64x64xf32>)
            %c0 = arith.constant 0 : index
            %c8 = arith.constant 8 : index
            %c1 = arith.constant 1 : index
            affine.for %arg10 = 0 to affine_map<(d0) -> (d0 ceildiv 64)>(%arg3) {
              affine.for %arg11 = 0 to affine_map<(d0) -> (d0 ceildiv 64)>(%arg4) {
                %alloc_0 = memref.alloc() {alignment = 64 : i64} : memref<64x64xf32>
                memref.copy %alloc, %alloc_0 : memref<64x64xf32> to memref<64x64xf32>
                %7 = scf.for %arg12 = %c0 to %c8 step %c1 iter_args(%arg13 = %alloc_0) -> (memref<64x64xf32>) {
                  %9 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 64 + d1 * 2097152 + d2 * 32768)>(%arg12, %arg10, %arg6)
                  %reinterpret_cast_1 = memref.reinterpret_cast %arg0 to offset: [%9], sizes: [64, 64], strides: [512, 1] {tmd.reuse = {sequential = [{depth = 6 : i64, iterator = "%arg12", reuse_type = "no_reuse", volume = 0 : i64}], spatial = [{depth = 2 : i64, iterator = "%arg8", mapped_to = "y", reuse_type = "total_reuse", volume = 16384 : i64}, {depth = 3 : i64, iterator = "%arg9", mapped_to = "x", reuse_type = "total_reuse", volume = 16384 : i64}], temporal = [{depth = 0 : i64, iterator = "%arg6", reuse_type = "no_reuse", volume = 0 : i64}, {depth = 1 : i64, iterator = "%arg7", reuse_type = "total_reuse", volume = 16384 : i64}, {depth = 4 : i64, iterator = "%arg10", reuse_type = "no_reuse", volume = 0 : i64}, {depth = 5 : i64, iterator = "%arg11", reuse_type = "total_reuse", volume = 16384 : i64}]}} : memref<*xf32> to memref<64x64xf32, strided<[512, 1], offset: ?>>
                  %alloc_2 = memref.alloc() {tmd.alloc = {local = true, memory_name = "L1", size = 16384 : i64}} : memref<64x64xf32>
                  memref.copy %reinterpret_cast_1, %alloc_2 {tmd.copy.choice = {kind = "mem", memory_name = "L1"}} : memref<64x64xf32, strided<[512, 1], offset: ?>> to memref<64x64xf32>
                  %10 = affine.apply affine_map<(d0, d1, d2, d3, d4) -> (d0 * 32768 + d1 * 4096 + d2 * 64 + d3 * 4096 + d4 * 512)>(%arg12, %arg11, %arg9, %arg7, %arg8)
                  %reinterpret_cast_3 = memref.reinterpret_cast %arg1 to offset: [%10], sizes: [64, 64], strides: [512, 1] {tmd.reuse = {sequential = [{depth = 6 : i64, iterator = "%arg12", reuse_type = "no_reuse", volume = 0 : i64}], spatial = [{depth = 2 : i64, iterator = "%arg8", mapped_to = "y", reuse_type = "no_reuse", volume = 0 : i64}, {depth = 3 : i64, iterator = "%arg9", mapped_to = "x", reuse_type = "no_reuse", volume = 0 : i64}], temporal = [{depth = 0 : i64, iterator = "%arg6", reuse_type = "total_reuse", volume = 16384 : i64}, {depth = 1 : i64, iterator = "%arg7", reuse_type = "no_reuse", volume = 0 : i64}, {depth = 4 : i64, iterator = "%arg10", reuse_type = "total_reuse", volume = 16384 : i64}, {depth = 5 : i64, iterator = "%arg11", reuse_type = "no_reuse", volume = 0 : i64}]}} : memref<*xf32> to memref<64x64xf32, strided<[512, 1], offset: ?>>
                  %alloc_4 = memref.alloc() {tmd.alloc = {local = true, memory_name = "L1", size = 16384 : i64}} : memref<64x64xf32>
                  memref.copy %reinterpret_cast_3, %alloc_4 {tmd.copy.choice = {kind = "mem", memory_name = "L1"}} : memref<64x64xf32, strided<[512, 1], offset: ?>> to memref<64x64xf32>
                  %alloc_5 = memref.alloc() {alignment = 64 : i64} : memref<64x64xf32>
                  memref.copy %alloc, %alloc_5 : memref<64x64xf32> to memref<64x64xf32>
                  %c32 = arith.constant 32 : index
                  %alloc_6 = memref.alloc() {tmd.alloc = {local = true, memory_name = "L1", size = 16384 : i64}} : memref<64x64xf32>
                  %alloc_7 = memref.alloc() {tmd.alloc = {local = true, memory_name = "L1", size = 16384 : i64}} : memref<64x64xf32>
                  memref.copy %alloc_2, %alloc_6 : memref<64x64xf32> to memref<64x64xf32>
                  memref.copy %alloc_4, %alloc_7 : memref<64x64xf32> to memref<64x64xf32>
                  affine.for %arg14 = 0 to 2 {
                    affine.for %arg15 = 0 to 2 {
                      %11 = affine.apply affine_map<(d0) -> (d0 * 32)>(%arg14)
                      %12 = affine.apply affine_map<(d0) -> (d0 * 32)>(%arg15)
                      %alloc_8 = memref.alloc() : memref<32x32xf32>
                      %subview = memref.subview %arg13[%11, %12] [%c32, %c32] [%c1, %c1] : memref<64x64xf32> to memref<?x?xf32, strided<[?, ?], offset: ?>>
                      memref.copy %subview, %alloc_8 : memref<?x?xf32, strided<[?, ?], offset: ?>> to memref<32x32xf32>
                      affine.for %arg16 = 0 to 2 {
                        %13 = affine.apply affine_map<(d0) -> (d0 * 32)>(%arg16)
                        %subview_9 = memref.subview %alloc_6[%11, %13] [%c32, %c32] [%c1, %c1] : memref<64x64xf32> to memref<?x?xf32, strided<[?, ?], offset: ?>>
                        %subview_10 = memref.subview %alloc_7[%13, %12] [%c32, %c32] [%c1, %c1] : memref<64x64xf32> to memref<?x?xf32, strided<[?, ?], offset: ?>>
                        linalg.matmul ins(%subview_9, %subview_10 : memref<?x?xf32, strided<[?, ?], offset: ?>>, memref<?x?xf32, strided<[?, ?], offset: ?>>) outs(%alloc_8 : memref<32x32xf32>)
                      }
                      memref.copy %alloc_8, %subview : memref<32x32xf32> to memref<?x?xf32, strided<[?, ?], offset: ?>>
                    }
                  }
                  linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%arg13, %alloc_5 : memref<64x64xf32>, memref<64x64xf32>) outs(%arg13 : memref<64x64xf32>) {
                  ^bb0(%in: f32, %in_8: f32, %out: f32):
                    %11 = arith.addf %in, %in_8 : f32
                    linalg.yield %11 : f32
                  }
                  scf.yield %arg13 : memref<64x64xf32>
                }
                %8 = affine.apply affine_map<(d0, d1, d2, d3, d4, d5) -> (d0 * 4096 + d1 * 64 + d2 * 4096 + d3 * 512 + d4 * 2097152 + d5 * 32768)>(%arg11, %arg9, %arg7, %arg8, %arg10, %arg6)
                %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%8], sizes: [64, 64], strides: [512, 1] {tmd.reuse = {spatial = [{depth = 2 : i64, iterator = "%arg8", mapped_to = "y", reuse_type = "no_reuse", volume = 0 : i64}, {depth = 3 : i64, iterator = "%arg9", mapped_to = "x", reuse_type = "no_reuse", volume = 0 : i64}], temporal = [{depth = 0 : i64, iterator = "%arg6", reuse_type = "no_reuse", volume = 0 : i64}, {depth = 1 : i64, iterator = "%arg7", reuse_type = "no_reuse", volume = 0 : i64}, {depth = 4 : i64, iterator = "%arg10", reuse_type = "no_reuse", volume = 0 : i64}, {depth = 5 : i64, iterator = "%arg11", reuse_type = "no_reuse", volume = 0 : i64}]}} : memref<*xf32> to memref<64x64xf32, strided<[512, 1], offset: ?>>
                memref.copy %7, %reinterpret_cast : memref<64x64xf32> to memref<64x64xf32, strided<[512, 1], offset: ?>>
              }
            }
          } {tmd.mapped_to = "x"}
        } {tmd.mapped_to = "y"}
      }
    }
    return
  }
  func.func @matmul_kernel__d0i1_d1i1__f01__c0bx_c1mem(%arg0: memref<*xf32> {tt.divisibility = 16 : i32}, %arg1: memref<*xf32> {tt.divisibility = 16 : i32}, %arg2: memref<*xf32> {tt.divisibility = 16 : i32}, %arg3: index, %arg4: index, %arg5: index) {
    affine.for %arg6 = 0 to affine_map<(d0, d1) -> (d0)>(%arg3, %arg4) {
      affine.for %arg7 = 0 to affine_map<(d0, d1) -> ((d1 ceildiv 8) ceildiv 8)>(%arg3, %arg4) {
        affine.parallel (%arg8) = (0) to (8) {
          affine.parallel (%arg9) = (0) to (8) {
            %cst = arith.constant 0.000000e+00 : f32
            %alloc = memref.alloc() {alignment = 64 : i64} : memref<64x64xf32>
            linalg.fill ins(%cst : f32) outs(%alloc : memref<64x64xf32>)
            %c0 = arith.constant 0 : index
            %c8 = arith.constant 8 : index
            %c1 = arith.constant 1 : index
            affine.for %arg10 = 0 to affine_map<(d0) -> (d0 ceildiv 64)>(%arg3) {
              affine.for %arg11 = 0 to affine_map<(d0) -> (d0 ceildiv 64)>(%arg4) {
                %alloc_0 = memref.alloc() {alignment = 64 : i64} : memref<64x64xf32>
                memref.copy %alloc, %alloc_0 : memref<64x64xf32> to memref<64x64xf32>
                %7 = scf.for %arg12 = %c0 to %c8 step %c1 iter_args(%arg13 = %alloc_0) -> (memref<64x64xf32>) {
                  %9 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 64 + d1 * 2097152 + d2 * 32768)>(%arg12, %arg10, %arg6)
                  %reinterpret_cast_1 = memref.reinterpret_cast %arg0 to offset: [%9], sizes: [64, 64], strides: [512, 1] {tmd.reuse = {sequential = [{depth = 6 : i64, iterator = "%arg12", reuse_type = "no_reuse", volume = 0 : i64}], spatial = [{depth = 2 : i64, iterator = "%arg8", mapped_to = "y", reuse_type = "total_reuse", volume = 16384 : i64}, {depth = 3 : i64, iterator = "%arg9", mapped_to = "x", reuse_type = "total_reuse", volume = 16384 : i64}], temporal = [{depth = 0 : i64, iterator = "%arg6", reuse_type = "no_reuse", volume = 0 : i64}, {depth = 1 : i64, iterator = "%arg7", reuse_type = "total_reuse", volume = 16384 : i64}, {depth = 4 : i64, iterator = "%arg10", reuse_type = "no_reuse", volume = 0 : i64}, {depth = 5 : i64, iterator = "%arg11", reuse_type = "total_reuse", volume = 16384 : i64}]}} : memref<*xf32> to memref<64x64xf32, strided<[512, 1], offset: ?>>
                  %alloc_2 = memref.alloc() {tmd.alloc = {local = true, memory_name = "L1", size = 16384 : i64}} : memref<64x64xf32>
                  memref.copy %reinterpret_cast_1, %alloc_2 {tmd.copy.choice = {dim = "x", interconnect_name = "horizontal_links", kind = "broadcast"}} : memref<64x64xf32, strided<[512, 1], offset: ?>> to memref<64x64xf32>
                  %10 = affine.apply affine_map<(d0, d1, d2, d3, d4) -> (d0 * 32768 + d1 * 4096 + d2 * 64 + d3 * 4096 + d4 * 512)>(%arg12, %arg11, %arg9, %arg7, %arg8)
                  %reinterpret_cast_3 = memref.reinterpret_cast %arg1 to offset: [%10], sizes: [64, 64], strides: [512, 1] {tmd.reuse = {sequential = [{depth = 6 : i64, iterator = "%arg12", reuse_type = "no_reuse", volume = 0 : i64}], spatial = [{depth = 2 : i64, iterator = "%arg8", mapped_to = "y", reuse_type = "no_reuse", volume = 0 : i64}, {depth = 3 : i64, iterator = "%arg9", mapped_to = "x", reuse_type = "no_reuse", volume = 0 : i64}], temporal = [{depth = 0 : i64, iterator = "%arg6", reuse_type = "total_reuse", volume = 16384 : i64}, {depth = 1 : i64, iterator = "%arg7", reuse_type = "no_reuse", volume = 0 : i64}, {depth = 4 : i64, iterator = "%arg10", reuse_type = "total_reuse", volume = 16384 : i64}, {depth = 5 : i64, iterator = "%arg11", reuse_type = "no_reuse", volume = 0 : i64}]}} : memref<*xf32> to memref<64x64xf32, strided<[512, 1], offset: ?>>
                  %alloc_4 = memref.alloc() {tmd.alloc = {local = true, memory_name = "L1", size = 16384 : i64}} : memref<64x64xf32>
                  memref.copy %reinterpret_cast_3, %alloc_4 {tmd.copy.choice = {kind = "mem", memory_name = "L1"}} : memref<64x64xf32, strided<[512, 1], offset: ?>> to memref<64x64xf32>
                  %alloc_5 = memref.alloc() {alignment = 64 : i64} : memref<64x64xf32>
                  memref.copy %alloc, %alloc_5 : memref<64x64xf32> to memref<64x64xf32>
                  %c32 = arith.constant 32 : index
                  %alloc_6 = memref.alloc() {tmd.alloc = {local = true, memory_name = "L1", size = 16384 : i64}} : memref<64x64xf32>
                  %alloc_7 = memref.alloc() {tmd.alloc = {local = true, memory_name = "L1", size = 16384 : i64}} : memref<64x64xf32>
                  memref.copy %alloc_2, %alloc_6 : memref<64x64xf32> to memref<64x64xf32>
                  memref.copy %alloc_4, %alloc_7 : memref<64x64xf32> to memref<64x64xf32>
                  affine.for %arg14 = 0 to 2 {
                    affine.for %arg15 = 0 to 2 {
                      %11 = affine.apply affine_map<(d0) -> (d0 * 32)>(%arg14)
                      %12 = affine.apply affine_map<(d0) -> (d0 * 32)>(%arg15)
                      %alloc_8 = memref.alloc() : memref<32x32xf32>
                      %subview = memref.subview %arg13[%11, %12] [%c32, %c32] [%c1, %c1] : memref<64x64xf32> to memref<?x?xf32, strided<[?, ?], offset: ?>>
                      memref.copy %subview, %alloc_8 : memref<?x?xf32, strided<[?, ?], offset: ?>> to memref<32x32xf32>
                      affine.for %arg16 = 0 to 2 {
                        %13 = affine.apply affine_map<(d0) -> (d0 * 32)>(%arg16)
                        %subview_9 = memref.subview %alloc_6[%11, %13] [%c32, %c32] [%c1, %c1] : memref<64x64xf32> to memref<?x?xf32, strided<[?, ?], offset: ?>>
                        %subview_10 = memref.subview %alloc_7[%13, %12] [%c32, %c32] [%c1, %c1] : memref<64x64xf32> to memref<?x?xf32, strided<[?, ?], offset: ?>>
                        linalg.matmul ins(%subview_9, %subview_10 : memref<?x?xf32, strided<[?, ?], offset: ?>>, memref<?x?xf32, strided<[?, ?], offset: ?>>) outs(%alloc_8 : memref<32x32xf32>)
                      }
                      memref.copy %alloc_8, %subview : memref<32x32xf32> to memref<?x?xf32, strided<[?, ?], offset: ?>>
                    }
                  }
                  linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%arg13, %alloc_5 : memref<64x64xf32>, memref<64x64xf32>) outs(%arg13 : memref<64x64xf32>) {
                  ^bb0(%in: f32, %in_8: f32, %out: f32):
                    %11 = arith.addf %in, %in_8 : f32
                    linalg.yield %11 : f32
                  }
                  scf.yield %arg13 : memref<64x64xf32>
                }
                %8 = affine.apply affine_map<(d0, d1, d2, d3, d4, d5) -> (d0 * 4096 + d1 * 64 + d2 * 4096 + d3 * 512 + d4 * 2097152 + d5 * 32768)>(%arg11, %arg9, %arg7, %arg8, %arg10, %arg6)
                %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%8], sizes: [64, 64], strides: [512, 1] {tmd.reuse = {spatial = [{depth = 2 : i64, iterator = "%arg8", mapped_to = "y", reuse_type = "no_reuse", volume = 0 : i64}, {depth = 3 : i64, iterator = "%arg9", mapped_to = "x", reuse_type = "no_reuse", volume = 0 : i64}], temporal = [{depth = 0 : i64, iterator = "%arg6", reuse_type = "no_reuse", volume = 0 : i64}, {depth = 1 : i64, iterator = "%arg7", reuse_type = "no_reuse", volume = 0 : i64}, {depth = 4 : i64, iterator = "%arg10", reuse_type = "no_reuse", volume = 0 : i64}, {depth = 5 : i64, iterator = "%arg11", reuse_type = "no_reuse", volume = 0 : i64}]}} : memref<*xf32> to memref<64x64xf32, strided<[512, 1], offset: ?>>
                memref.copy %7, %reinterpret_cast : memref<64x64xf32> to memref<64x64xf32, strided<[512, 1], offset: ?>>
              }
            }
          } {tmd.mapped_to = "x"}
        } {tmd.mapped_to = "y"}
      }
    }
    return
  }
  func.func @matmul_kernel__d0i1_d1i1__f01__c0by_c1mem(%arg0: memref<*xf32> {tt.divisibility = 16 : i32}, %arg1: memref<*xf32> {tt.divisibility = 16 : i32}, %arg2: memref<*xf32> {tt.divisibility = 16 : i32}, %arg3: index, %arg4: index, %arg5: index) {
    affine.for %arg6 = 0 to affine_map<(d0, d1) -> (d0)>(%arg3, %arg4) {
      affine.for %arg7 = 0 to affine_map<(d0, d1) -> ((d1 ceildiv 8) ceildiv 8)>(%arg3, %arg4) {
        affine.parallel (%arg8) = (0) to (8) {
          affine.parallel (%arg9) = (0) to (8) {
            %cst = arith.constant 0.000000e+00 : f32
            %alloc = memref.alloc() {alignment = 64 : i64} : memref<64x64xf32>
            linalg.fill ins(%cst : f32) outs(%alloc : memref<64x64xf32>)
            %c0 = arith.constant 0 : index
            %c8 = arith.constant 8 : index
            %c1 = arith.constant 1 : index
            affine.for %arg10 = 0 to affine_map<(d0) -> (d0 ceildiv 64)>(%arg3) {
              affine.for %arg11 = 0 to affine_map<(d0) -> (d0 ceildiv 64)>(%arg4) {
                %alloc_0 = memref.alloc() {alignment = 64 : i64} : memref<64x64xf32>
                memref.copy %alloc, %alloc_0 : memref<64x64xf32> to memref<64x64xf32>
                %7 = scf.for %arg12 = %c0 to %c8 step %c1 iter_args(%arg13 = %alloc_0) -> (memref<64x64xf32>) {
                  %9 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 64 + d1 * 2097152 + d2 * 32768)>(%arg12, %arg10, %arg6)
                  %reinterpret_cast_1 = memref.reinterpret_cast %arg0 to offset: [%9], sizes: [64, 64], strides: [512, 1] {tmd.reuse = {sequential = [{depth = 6 : i64, iterator = "%arg12", reuse_type = "no_reuse", volume = 0 : i64}], spatial = [{depth = 2 : i64, iterator = "%arg8", mapped_to = "y", reuse_type = "total_reuse", volume = 16384 : i64}, {depth = 3 : i64, iterator = "%arg9", mapped_to = "x", reuse_type = "total_reuse", volume = 16384 : i64}], temporal = [{depth = 0 : i64, iterator = "%arg6", reuse_type = "no_reuse", volume = 0 : i64}, {depth = 1 : i64, iterator = "%arg7", reuse_type = "total_reuse", volume = 16384 : i64}, {depth = 4 : i64, iterator = "%arg10", reuse_type = "no_reuse", volume = 0 : i64}, {depth = 5 : i64, iterator = "%arg11", reuse_type = "total_reuse", volume = 16384 : i64}]}} : memref<*xf32> to memref<64x64xf32, strided<[512, 1], offset: ?>>
                  %alloc_2 = memref.alloc() {tmd.alloc = {local = true, memory_name = "L1", size = 16384 : i64}} : memref<64x64xf32>
                  memref.copy %reinterpret_cast_1, %alloc_2 {tmd.copy.choice = {dim = "y", interconnect_name = "vertical_links", kind = "broadcast"}} : memref<64x64xf32, strided<[512, 1], offset: ?>> to memref<64x64xf32>
                  %10 = affine.apply affine_map<(d0, d1, d2, d3, d4) -> (d0 * 32768 + d1 * 4096 + d2 * 64 + d3 * 4096 + d4 * 512)>(%arg12, %arg11, %arg9, %arg7, %arg8)
                  %reinterpret_cast_3 = memref.reinterpret_cast %arg1 to offset: [%10], sizes: [64, 64], strides: [512, 1] {tmd.reuse = {sequential = [{depth = 6 : i64, iterator = "%arg12", reuse_type = "no_reuse", volume = 0 : i64}], spatial = [{depth = 2 : i64, iterator = "%arg8", mapped_to = "y", reuse_type = "no_reuse", volume = 0 : i64}, {depth = 3 : i64, iterator = "%arg9", mapped_to = "x", reuse_type = "no_reuse", volume = 0 : i64}], temporal = [{depth = 0 : i64, iterator = "%arg6", reuse_type = "total_reuse", volume = 16384 : i64}, {depth = 1 : i64, iterator = "%arg7", reuse_type = "no_reuse", volume = 0 : i64}, {depth = 4 : i64, iterator = "%arg10", reuse_type = "total_reuse", volume = 16384 : i64}, {depth = 5 : i64, iterator = "%arg11", reuse_type = "no_reuse", volume = 0 : i64}]}} : memref<*xf32> to memref<64x64xf32, strided<[512, 1], offset: ?>>
                  %alloc_4 = memref.alloc() {tmd.alloc = {local = true, memory_name = "L1", size = 16384 : i64}} : memref<64x64xf32>
                  memref.copy %reinterpret_cast_3, %alloc_4 {tmd.copy.choice = {kind = "mem", memory_name = "L1"}} : memref<64x64xf32, strided<[512, 1], offset: ?>> to memref<64x64xf32>
                  %alloc_5 = memref.alloc() {alignment = 64 : i64} : memref<64x64xf32>
                  memref.copy %alloc, %alloc_5 : memref<64x64xf32> to memref<64x64xf32>
                  %c32 = arith.constant 32 : index
                  %alloc_6 = memref.alloc() {tmd.alloc = {local = true, memory_name = "L1", size = 16384 : i64}} : memref<64x64xf32>
                  %alloc_7 = memref.alloc() {tmd.alloc = {local = true, memory_name = "L1", size = 16384 : i64}} : memref<64x64xf32>
                  memref.copy %alloc_2, %alloc_6 : memref<64x64xf32> to memref<64x64xf32>
                  memref.copy %alloc_4, %alloc_7 : memref<64x64xf32> to memref<64x64xf32>
                  affine.for %arg14 = 0 to 2 {
                    affine.for %arg15 = 0 to 2 {
                      %11 = affine.apply affine_map<(d0) -> (d0 * 32)>(%arg14)
                      %12 = affine.apply affine_map<(d0) -> (d0 * 32)>(%arg15)
                      %alloc_8 = memref.alloc() : memref<32x32xf32>
                      %subview = memref.subview %arg13[%11, %12] [%c32, %c32] [%c1, %c1] : memref<64x64xf32> to memref<?x?xf32, strided<[?, ?], offset: ?>>
                      memref.copy %subview, %alloc_8 : memref<?x?xf32, strided<[?, ?], offset: ?>> to memref<32x32xf32>
                      affine.for %arg16 = 0 to 2 {
                        %13 = affine.apply affine_map<(d0) -> (d0 * 32)>(%arg16)
                        %subview_9 = memref.subview %alloc_6[%11, %13] [%c32, %c32] [%c1, %c1] : memref<64x64xf32> to memref<?x?xf32, strided<[?, ?], offset: ?>>
                        %subview_10 = memref.subview %alloc_7[%13, %12] [%c32, %c32] [%c1, %c1] : memref<64x64xf32> to memref<?x?xf32, strided<[?, ?], offset: ?>>
                        linalg.matmul ins(%subview_9, %subview_10 : memref<?x?xf32, strided<[?, ?], offset: ?>>, memref<?x?xf32, strided<[?, ?], offset: ?>>) outs(%alloc_8 : memref<32x32xf32>)
                      }
                      memref.copy %alloc_8, %subview : memref<32x32xf32> to memref<?x?xf32, strided<[?, ?], offset: ?>>
                    }
                  }
                  linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%arg13, %alloc_5 : memref<64x64xf32>, memref<64x64xf32>) outs(%arg13 : memref<64x64xf32>) {
                  ^bb0(%in: f32, %in_8: f32, %out: f32):
                    %11 = arith.addf %in, %in_8 : f32
                    linalg.yield %11 : f32
                  }
                  scf.yield %arg13 : memref<64x64xf32>
                }
                %8 = affine.apply affine_map<(d0, d1, d2, d3, d4, d5) -> (d0 * 4096 + d1 * 64 + d2 * 4096 + d3 * 512 + d4 * 2097152 + d5 * 32768)>(%arg11, %arg9, %arg7, %arg8, %arg10, %arg6)
                %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%8], sizes: [64, 64], strides: [512, 1] {tmd.reuse = {spatial = [{depth = 2 : i64, iterator = "%arg8", mapped_to = "y", reuse_type = "no_reuse", volume = 0 : i64}, {depth = 3 : i64, iterator = "%arg9", mapped_to = "x", reuse_type = "no_reuse", volume = 0 : i64}], temporal = [{depth = 0 : i64, iterator = "%arg6", reuse_type = "no_reuse", volume = 0 : i64}, {depth = 1 : i64, iterator = "%arg7", reuse_type = "no_reuse", volume = 0 : i64}, {depth = 4 : i64, iterator = "%arg10", reuse_type = "no_reuse", volume = 0 : i64}, {depth = 5 : i64, iterator = "%arg11", reuse_type = "no_reuse", volume = 0 : i64}]}} : memref<*xf32> to memref<64x64xf32, strided<[512, 1], offset: ?>>
                memref.copy %7, %reinterpret_cast : memref<64x64xf32> to memref<64x64xf32, strided<[512, 1], offset: ?>>
              }
            }
          } {tmd.mapped_to = "x"}
        } {tmd.mapped_to = "y"}
      }
    }
    return
  }
  func.func @matmul_kernel__d0i1_d1i1__f10__c0mem_c1mem(%arg0: memref<*xf32> {tt.divisibility = 16 : i32}, %arg1: memref<*xf32> {tt.divisibility = 16 : i32}, %arg2: memref<*xf32> {tt.divisibility = 16 : i32}, %arg3: index, %arg4: index, %arg5: index) {
    affine.for %arg6 = 0 to affine_map<(d0, d1) -> ((d1 ceildiv 8) ceildiv 8)>(%arg3, %arg4) {
      affine.for %arg7 = 0 to affine_map<(d0, d1) -> (d0)>(%arg3, %arg4) {
        affine.parallel (%arg8) = (0) to (8) {
          affine.parallel (%arg9) = (0) to (8) {
            %cst = arith.constant 0.000000e+00 : f32
            %alloc = memref.alloc() {alignment = 64 : i64} : memref<64x64xf32>
            linalg.fill ins(%cst : f32) outs(%alloc : memref<64x64xf32>)
            %c0 = arith.constant 0 : index
            %c8 = arith.constant 8 : index
            %c1 = arith.constant 1 : index
            affine.for %arg10 = 0 to affine_map<(d0) -> (d0 ceildiv 64)>(%arg3) {
              affine.for %arg11 = 0 to affine_map<(d0) -> (d0 ceildiv 64)>(%arg4) {
                %alloc_0 = memref.alloc() {alignment = 64 : i64} : memref<64x64xf32>
                memref.copy %alloc, %alloc_0 : memref<64x64xf32> to memref<64x64xf32>
                %7 = scf.for %arg12 = %c0 to %c8 step %c1 iter_args(%arg13 = %alloc_0) -> (memref<64x64xf32>) {
                  %9 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 64 + d1 * 2097152 + d2 * 32768)>(%arg12, %arg10, %arg7)
                  %reinterpret_cast_1 = memref.reinterpret_cast %arg0 to offset: [%9], sizes: [64, 64], strides: [512, 1] {tmd.reuse = {sequential = [{depth = 6 : i64, iterator = "%arg12", reuse_type = "no_reuse", volume = 0 : i64}], spatial = [{depth = 2 : i64, iterator = "%arg8", mapped_to = "y", reuse_type = "total_reuse", volume = 16384 : i64}, {depth = 3 : i64, iterator = "%arg9", mapped_to = "x", reuse_type = "total_reuse", volume = 16384 : i64}], temporal = [{depth = 0 : i64, iterator = "%arg6", reuse_type = "total_reuse", volume = 16384 : i64}, {depth = 1 : i64, iterator = "%arg7", reuse_type = "no_reuse", volume = 0 : i64}, {depth = 4 : i64, iterator = "%arg10", reuse_type = "no_reuse", volume = 0 : i64}, {depth = 5 : i64, iterator = "%arg11", reuse_type = "total_reuse", volume = 16384 : i64}]}} : memref<*xf32> to memref<64x64xf32, strided<[512, 1], offset: ?>>
                  %alloc_2 = memref.alloc() {tmd.alloc = {local = true, memory_name = "L1", size = 16384 : i64}} : memref<64x64xf32>
                  memref.copy %reinterpret_cast_1, %alloc_2 {tmd.copy.choice = {kind = "mem", memory_name = "L1"}} : memref<64x64xf32, strided<[512, 1], offset: ?>> to memref<64x64xf32>
                  %10 = affine.apply affine_map<(d0, d1, d2, d3, d4) -> (d0 * 32768 + d1 * 4096 + d2 * 64 + d3 * 4096 + d4 * 512)>(%arg12, %arg11, %arg9, %arg6, %arg8)
                  %reinterpret_cast_3 = memref.reinterpret_cast %arg1 to offset: [%10], sizes: [64, 64], strides: [512, 1] {tmd.reuse = {sequential = [{depth = 6 : i64, iterator = "%arg12", reuse_type = "no_reuse", volume = 0 : i64}], spatial = [{depth = 2 : i64, iterator = "%arg8", mapped_to = "y", reuse_type = "no_reuse", volume = 0 : i64}, {depth = 3 : i64, iterator = "%arg9", mapped_to = "x", reuse_type = "no_reuse", volume = 0 : i64}], temporal = [{depth = 0 : i64, iterator = "%arg6", reuse_type = "no_reuse", volume = 0 : i64}, {depth = 1 : i64, iterator = "%arg7", reuse_type = "total_reuse", volume = 16384 : i64}, {depth = 4 : i64, iterator = "%arg10", reuse_type = "total_reuse", volume = 16384 : i64}, {depth = 5 : i64, iterator = "%arg11", reuse_type = "no_reuse", volume = 0 : i64}]}} : memref<*xf32> to memref<64x64xf32, strided<[512, 1], offset: ?>>
                  %alloc_4 = memref.alloc() {tmd.alloc = {local = true, memory_name = "L1", size = 16384 : i64}} : memref<64x64xf32>
                  memref.copy %reinterpret_cast_3, %alloc_4 {tmd.copy.choice = {kind = "mem", memory_name = "L1"}} : memref<64x64xf32, strided<[512, 1], offset: ?>> to memref<64x64xf32>
                  %alloc_5 = memref.alloc() {alignment = 64 : i64} : memref<64x64xf32>
                  memref.copy %alloc, %alloc_5 : memref<64x64xf32> to memref<64x64xf32>
                  %c32 = arith.constant 32 : index
                  %alloc_6 = memref.alloc() {tmd.alloc = {local = true, memory_name = "L1", size = 16384 : i64}} : memref<64x64xf32>
                  %alloc_7 = memref.alloc() {tmd.alloc = {local = true, memory_name = "L1", size = 16384 : i64}} : memref<64x64xf32>
                  memref.copy %alloc_2, %alloc_6 : memref<64x64xf32> to memref<64x64xf32>
                  memref.copy %alloc_4, %alloc_7 : memref<64x64xf32> to memref<64x64xf32>
                  affine.for %arg14 = 0 to 2 {
                    affine.for %arg15 = 0 to 2 {
                      %11 = affine.apply affine_map<(d0) -> (d0 * 32)>(%arg14)
                      %12 = affine.apply affine_map<(d0) -> (d0 * 32)>(%arg15)
                      %alloc_8 = memref.alloc() : memref<32x32xf32>
                      %subview = memref.subview %arg13[%11, %12] [%c32, %c32] [%c1, %c1] : memref<64x64xf32> to memref<?x?xf32, strided<[?, ?], offset: ?>>
                      memref.copy %subview, %alloc_8 : memref<?x?xf32, strided<[?, ?], offset: ?>> to memref<32x32xf32>
                      affine.for %arg16 = 0 to 2 {
                        %13 = affine.apply affine_map<(d0) -> (d0 * 32)>(%arg16)
                        %subview_9 = memref.subview %alloc_6[%11, %13] [%c32, %c32] [%c1, %c1] : memref<64x64xf32> to memref<?x?xf32, strided<[?, ?], offset: ?>>
                        %subview_10 = memref.subview %alloc_7[%13, %12] [%c32, %c32] [%c1, %c1] : memref<64x64xf32> to memref<?x?xf32, strided<[?, ?], offset: ?>>
                        linalg.matmul ins(%subview_9, %subview_10 : memref<?x?xf32, strided<[?, ?], offset: ?>>, memref<?x?xf32, strided<[?, ?], offset: ?>>) outs(%alloc_8 : memref<32x32xf32>)
                      }
                      memref.copy %alloc_8, %subview : memref<32x32xf32> to memref<?x?xf32, strided<[?, ?], offset: ?>>
                    }
                  }
                  linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%arg13, %alloc_5 : memref<64x64xf32>, memref<64x64xf32>) outs(%arg13 : memref<64x64xf32>) {
                  ^bb0(%in: f32, %in_8: f32, %out: f32):
                    %11 = arith.addf %in, %in_8 : f32
                    linalg.yield %11 : f32
                  }
                  scf.yield %arg13 : memref<64x64xf32>
                }
                %8 = affine.apply affine_map<(d0, d1, d2, d3, d4, d5) -> (d0 * 4096 + d1 * 64 + d2 * 4096 + d3 * 512 + d4 * 2097152 + d5 * 32768)>(%arg11, %arg9, %arg6, %arg8, %arg10, %arg7)
                %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%8], sizes: [64, 64], strides: [512, 1] {tmd.reuse = {spatial = [{depth = 2 : i64, iterator = "%arg8", mapped_to = "y", reuse_type = "no_reuse", volume = 0 : i64}, {depth = 3 : i64, iterator = "%arg9", mapped_to = "x", reuse_type = "no_reuse", volume = 0 : i64}], temporal = [{depth = 0 : i64, iterator = "%arg6", reuse_type = "no_reuse", volume = 0 : i64}, {depth = 1 : i64, iterator = "%arg7", reuse_type = "no_reuse", volume = 0 : i64}, {depth = 4 : i64, iterator = "%arg10", reuse_type = "no_reuse", volume = 0 : i64}, {depth = 5 : i64, iterator = "%arg11", reuse_type = "no_reuse", volume = 0 : i64}]}} : memref<*xf32> to memref<64x64xf32, strided<[512, 1], offset: ?>>
                memref.copy %7, %reinterpret_cast : memref<64x64xf32> to memref<64x64xf32, strided<[512, 1], offset: ?>>
              }
            }
          } {tmd.mapped_to = "x"}
        } {tmd.mapped_to = "y"}
      }
    }
    return
  }
  func.func @matmul_kernel__d0i1_d1i1__f10__c0bx_c1mem(%arg0: memref<*xf32> {tt.divisibility = 16 : i32}, %arg1: memref<*xf32> {tt.divisibility = 16 : i32}, %arg2: memref<*xf32> {tt.divisibility = 16 : i32}, %arg3: index, %arg4: index, %arg5: index) {
    affine.for %arg6 = 0 to affine_map<(d0, d1) -> ((d1 ceildiv 8) ceildiv 8)>(%arg3, %arg4) {
      affine.for %arg7 = 0 to affine_map<(d0, d1) -> (d0)>(%arg3, %arg4) {
        affine.parallel (%arg8) = (0) to (8) {
          affine.parallel (%arg9) = (0) to (8) {
            %cst = arith.constant 0.000000e+00 : f32
            %alloc = memref.alloc() {alignment = 64 : i64} : memref<64x64xf32>
            linalg.fill ins(%cst : f32) outs(%alloc : memref<64x64xf32>)
            %c0 = arith.constant 0 : index
            %c8 = arith.constant 8 : index
            %c1 = arith.constant 1 : index
            affine.for %arg10 = 0 to affine_map<(d0) -> (d0 ceildiv 64)>(%arg3) {
              affine.for %arg11 = 0 to affine_map<(d0) -> (d0 ceildiv 64)>(%arg4) {
                %alloc_0 = memref.alloc() {alignment = 64 : i64} : memref<64x64xf32>
                memref.copy %alloc, %alloc_0 : memref<64x64xf32> to memref<64x64xf32>
                %7 = scf.for %arg12 = %c0 to %c8 step %c1 iter_args(%arg13 = %alloc_0) -> (memref<64x64xf32>) {
                  %9 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 64 + d1 * 2097152 + d2 * 32768)>(%arg12, %arg10, %arg7)
                  %reinterpret_cast_1 = memref.reinterpret_cast %arg0 to offset: [%9], sizes: [64, 64], strides: [512, 1] {tmd.reuse = {sequential = [{depth = 6 : i64, iterator = "%arg12", reuse_type = "no_reuse", volume = 0 : i64}], spatial = [{depth = 2 : i64, iterator = "%arg8", mapped_to = "y", reuse_type = "total_reuse", volume = 16384 : i64}, {depth = 3 : i64, iterator = "%arg9", mapped_to = "x", reuse_type = "total_reuse", volume = 16384 : i64}], temporal = [{depth = 0 : i64, iterator = "%arg6", reuse_type = "total_reuse", volume = 16384 : i64}, {depth = 1 : i64, iterator = "%arg7", reuse_type = "no_reuse", volume = 0 : i64}, {depth = 4 : i64, iterator = "%arg10", reuse_type = "no_reuse", volume = 0 : i64}, {depth = 5 : i64, iterator = "%arg11", reuse_type = "total_reuse", volume = 16384 : i64}]}} : memref<*xf32> to memref<64x64xf32, strided<[512, 1], offset: ?>>
                  %alloc_2 = memref.alloc() {tmd.alloc = {local = true, memory_name = "L1", size = 16384 : i64}} : memref<64x64xf32>
                  memref.copy %reinterpret_cast_1, %alloc_2 {tmd.copy.choice = {dim = "x", interconnect_name = "horizontal_links", kind = "broadcast"}} : memref<64x64xf32, strided<[512, 1], offset: ?>> to memref<64x64xf32>
                  %10 = affine.apply affine_map<(d0, d1, d2, d3, d4) -> (d0 * 32768 + d1 * 4096 + d2 * 64 + d3 * 4096 + d4 * 512)>(%arg12, %arg11, %arg9, %arg6, %arg8)
                  %reinterpret_cast_3 = memref.reinterpret_cast %arg1 to offset: [%10], sizes: [64, 64], strides: [512, 1] {tmd.reuse = {sequential = [{depth = 6 : i64, iterator = "%arg12", reuse_type = "no_reuse", volume = 0 : i64}], spatial = [{depth = 2 : i64, iterator = "%arg8", mapped_to = "y", reuse_type = "no_reuse", volume = 0 : i64}, {depth = 3 : i64, iterator = "%arg9", mapped_to = "x", reuse_type = "no_reuse", volume = 0 : i64}], temporal = [{depth = 0 : i64, iterator = "%arg6", reuse_type = "no_reuse", volume = 0 : i64}, {depth = 1 : i64, iterator = "%arg7", reuse_type = "total_reuse", volume = 16384 : i64}, {depth = 4 : i64, iterator = "%arg10", reuse_type = "total_reuse", volume = 16384 : i64}, {depth = 5 : i64, iterator = "%arg11", reuse_type = "no_reuse", volume = 0 : i64}]}} : memref<*xf32> to memref<64x64xf32, strided<[512, 1], offset: ?>>
                  %alloc_4 = memref.alloc() {tmd.alloc = {local = true, memory_name = "L1", size = 16384 : i64}} : memref<64x64xf32>
                  memref.copy %reinterpret_cast_3, %alloc_4 {tmd.copy.choice = {kind = "mem", memory_name = "L1"}} : memref<64x64xf32, strided<[512, 1], offset: ?>> to memref<64x64xf32>
                  %alloc_5 = memref.alloc() {alignment = 64 : i64} : memref<64x64xf32>
                  memref.copy %alloc, %alloc_5 : memref<64x64xf32> to memref<64x64xf32>
                  %c32 = arith.constant 32 : index
                  %alloc_6 = memref.alloc() {tmd.alloc = {local = true, memory_name = "L1", size = 16384 : i64}} : memref<64x64xf32>
                  %alloc_7 = memref.alloc() {tmd.alloc = {local = true, memory_name = "L1", size = 16384 : i64}} : memref<64x64xf32>
                  memref.copy %alloc_2, %alloc_6 : memref<64x64xf32> to memref<64x64xf32>
                  memref.copy %alloc_4, %alloc_7 : memref<64x64xf32> to memref<64x64xf32>
                  affine.for %arg14 = 0 to 2 {
                    affine.for %arg15 = 0 to 2 {
                      %11 = affine.apply affine_map<(d0) -> (d0 * 32)>(%arg14)
                      %12 = affine.apply affine_map<(d0) -> (d0 * 32)>(%arg15)
                      %alloc_8 = memref.alloc() : memref<32x32xf32>
                      %subview = memref.subview %arg13[%11, %12] [%c32, %c32] [%c1, %c1] : memref<64x64xf32> to memref<?x?xf32, strided<[?, ?], offset: ?>>
                      memref.copy %subview, %alloc_8 : memref<?x?xf32, strided<[?, ?], offset: ?>> to memref<32x32xf32>
                      affine.for %arg16 = 0 to 2 {
                        %13 = affine.apply affine_map<(d0) -> (d0 * 32)>(%arg16)
                        %subview_9 = memref.subview %alloc_6[%11, %13] [%c32, %c32] [%c1, %c1] : memref<64x64xf32> to memref<?x?xf32, strided<[?, ?], offset: ?>>
                        %subview_10 = memref.subview %alloc_7[%13, %12] [%c32, %c32] [%c1, %c1] : memref<64x64xf32> to memref<?x?xf32, strided<[?, ?], offset: ?>>
                        linalg.matmul ins(%subview_9, %subview_10 : memref<?x?xf32, strided<[?, ?], offset: ?>>, memref<?x?xf32, strided<[?, ?], offset: ?>>) outs(%alloc_8 : memref<32x32xf32>)
                      }
                      memref.copy %alloc_8, %subview : memref<32x32xf32> to memref<?x?xf32, strided<[?, ?], offset: ?>>
                    }
                  }
                  linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%arg13, %alloc_5 : memref<64x64xf32>, memref<64x64xf32>) outs(%arg13 : memref<64x64xf32>) {
                  ^bb0(%in: f32, %in_8: f32, %out: f32):
                    %11 = arith.addf %in, %in_8 : f32
                    linalg.yield %11 : f32
                  }
                  scf.yield %arg13 : memref<64x64xf32>
                }
                %8 = affine.apply affine_map<(d0, d1, d2, d3, d4, d5) -> (d0 * 4096 + d1 * 64 + d2 * 4096 + d3 * 512 + d4 * 2097152 + d5 * 32768)>(%arg11, %arg9, %arg6, %arg8, %arg10, %arg7)
                %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%8], sizes: [64, 64], strides: [512, 1] {tmd.reuse = {spatial = [{depth = 2 : i64, iterator = "%arg8", mapped_to = "y", reuse_type = "no_reuse", volume = 0 : i64}, {depth = 3 : i64, iterator = "%arg9", mapped_to = "x", reuse_type = "no_reuse", volume = 0 : i64}], temporal = [{depth = 0 : i64, iterator = "%arg6", reuse_type = "no_reuse", volume = 0 : i64}, {depth = 1 : i64, iterator = "%arg7", reuse_type = "no_reuse", volume = 0 : i64}, {depth = 4 : i64, iterator = "%arg10", reuse_type = "no_reuse", volume = 0 : i64}, {depth = 5 : i64, iterator = "%arg11", reuse_type = "no_reuse", volume = 0 : i64}]}} : memref<*xf32> to memref<64x64xf32, strided<[512, 1], offset: ?>>
                memref.copy %7, %reinterpret_cast : memref<64x64xf32> to memref<64x64xf32, strided<[512, 1], offset: ?>>
              }
            }
          } {tmd.mapped_to = "x"}
        } {tmd.mapped_to = "y"}
      }
    }
    return
  }
  func.func @matmul_kernel__d0i1_d1i1__f10__c0by_c1mem(%arg0: memref<*xf32> {tt.divisibility = 16 : i32}, %arg1: memref<*xf32> {tt.divisibility = 16 : i32}, %arg2: memref<*xf32> {tt.divisibility = 16 : i32}, %arg3: index, %arg4: index, %arg5: index) {
    affine.for %arg6 = 0 to affine_map<(d0, d1) -> ((d1 ceildiv 8) ceildiv 8)>(%arg3, %arg4) {
      affine.for %arg7 = 0 to affine_map<(d0, d1) -> (d0)>(%arg3, %arg4) {
        affine.parallel (%arg8) = (0) to (8) {
          affine.parallel (%arg9) = (0) to (8) {
            %cst = arith.constant 0.000000e+00 : f32
            %alloc = memref.alloc() {alignment = 64 : i64} : memref<64x64xf32>
            linalg.fill ins(%cst : f32) outs(%alloc : memref<64x64xf32>)
            %c0 = arith.constant 0 : index
            %c8 = arith.constant 8 : index
            %c1 = arith.constant 1 : index
            affine.for %arg10 = 0 to affine_map<(d0) -> (d0 ceildiv 64)>(%arg3) {
              affine.for %arg11 = 0 to affine_map<(d0) -> (d0 ceildiv 64)>(%arg4) {
                %alloc_0 = memref.alloc() {alignment = 64 : i64} : memref<64x64xf32>
                memref.copy %alloc, %alloc_0 : memref<64x64xf32> to memref<64x64xf32>
                %7 = scf.for %arg12 = %c0 to %c8 step %c1 iter_args(%arg13 = %alloc_0) -> (memref<64x64xf32>) {
                  %9 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 64 + d1 * 2097152 + d2 * 32768)>(%arg12, %arg10, %arg7)
                  %reinterpret_cast_1 = memref.reinterpret_cast %arg0 to offset: [%9], sizes: [64, 64], strides: [512, 1] {tmd.reuse = {sequential = [{depth = 6 : i64, iterator = "%arg12", reuse_type = "no_reuse", volume = 0 : i64}], spatial = [{depth = 2 : i64, iterator = "%arg8", mapped_to = "y", reuse_type = "total_reuse", volume = 16384 : i64}, {depth = 3 : i64, iterator = "%arg9", mapped_to = "x", reuse_type = "total_reuse", volume = 16384 : i64}], temporal = [{depth = 0 : i64, iterator = "%arg6", reuse_type = "total_reuse", volume = 16384 : i64}, {depth = 1 : i64, iterator = "%arg7", reuse_type = "no_reuse", volume = 0 : i64}, {depth = 4 : i64, iterator = "%arg10", reuse_type = "no_reuse", volume = 0 : i64}, {depth = 5 : i64, iterator = "%arg11", reuse_type = "total_reuse", volume = 16384 : i64}]}} : memref<*xf32> to memref<64x64xf32, strided<[512, 1], offset: ?>>
                  %alloc_2 = memref.alloc() {tmd.alloc = {local = true, memory_name = "L1", size = 16384 : i64}} : memref<64x64xf32>
                  memref.copy %reinterpret_cast_1, %alloc_2 {tmd.copy.choice = {dim = "y", interconnect_name = "vertical_links", kind = "broadcast"}} : memref<64x64xf32, strided<[512, 1], offset: ?>> to memref<64x64xf32>
                  %10 = affine.apply affine_map<(d0, d1, d2, d3, d4) -> (d0 * 32768 + d1 * 4096 + d2 * 64 + d3 * 4096 + d4 * 512)>(%arg12, %arg11, %arg9, %arg6, %arg8)
                  %reinterpret_cast_3 = memref.reinterpret_cast %arg1 to offset: [%10], sizes: [64, 64], strides: [512, 1] {tmd.reuse = {sequential = [{depth = 6 : i64, iterator = "%arg12", reuse_type = "no_reuse", volume = 0 : i64}], spatial = [{depth = 2 : i64, iterator = "%arg8", mapped_to = "y", reuse_type = "no_reuse", volume = 0 : i64}, {depth = 3 : i64, iterator = "%arg9", mapped_to = "x", reuse_type = "no_reuse", volume = 0 : i64}], temporal = [{depth = 0 : i64, iterator = "%arg6", reuse_type = "no_reuse", volume = 0 : i64}, {depth = 1 : i64, iterator = "%arg7", reuse_type = "total_reuse", volume = 16384 : i64}, {depth = 4 : i64, iterator = "%arg10", reuse_type = "total_reuse", volume = 16384 : i64}, {depth = 5 : i64, iterator = "%arg11", reuse_type = "no_reuse", volume = 0 : i64}]}} : memref<*xf32> to memref<64x64xf32, strided<[512, 1], offset: ?>>
                  %alloc_4 = memref.alloc() {tmd.alloc = {local = true, memory_name = "L1", size = 16384 : i64}} : memref<64x64xf32>
                  memref.copy %reinterpret_cast_3, %alloc_4 {tmd.copy.choice = {kind = "mem", memory_name = "L1"}} : memref<64x64xf32, strided<[512, 1], offset: ?>> to memref<64x64xf32>
                  %alloc_5 = memref.alloc() {alignment = 64 : i64} : memref<64x64xf32>
                  memref.copy %alloc, %alloc_5 : memref<64x64xf32> to memref<64x64xf32>
                  %c32 = arith.constant 32 : index
                  %alloc_6 = memref.alloc() {tmd.alloc = {local = true, memory_name = "L1", size = 16384 : i64}} : memref<64x64xf32>
                  %alloc_7 = memref.alloc() {tmd.alloc = {local = true, memory_name = "L1", size = 16384 : i64}} : memref<64x64xf32>
                  memref.copy %alloc_2, %alloc_6 : memref<64x64xf32> to memref<64x64xf32>
                  memref.copy %alloc_4, %alloc_7 : memref<64x64xf32> to memref<64x64xf32>
                  affine.for %arg14 = 0 to 2 {
                    affine.for %arg15 = 0 to 2 {
                      %11 = affine.apply affine_map<(d0) -> (d0 * 32)>(%arg14)
                      %12 = affine.apply affine_map<(d0) -> (d0 * 32)>(%arg15)
                      %alloc_8 = memref.alloc() : memref<32x32xf32>
                      %subview = memref.subview %arg13[%11, %12] [%c32, %c32] [%c1, %c1] : memref<64x64xf32> to memref<?x?xf32, strided<[?, ?], offset: ?>>
                      memref.copy %subview, %alloc_8 : memref<?x?xf32, strided<[?, ?], offset: ?>> to memref<32x32xf32>
                      affine.for %arg16 = 0 to 2 {
                        %13 = affine.apply affine_map<(d0) -> (d0 * 32)>(%arg16)
                        %subview_9 = memref.subview %alloc_6[%11, %13] [%c32, %c32] [%c1, %c1] : memref<64x64xf32> to memref<?x?xf32, strided<[?, ?], offset: ?>>
                        %subview_10 = memref.subview %alloc_7[%13, %12] [%c32, %c32] [%c1, %c1] : memref<64x64xf32> to memref<?x?xf32, strided<[?, ?], offset: ?>>
                        linalg.matmul ins(%subview_9, %subview_10 : memref<?x?xf32, strided<[?, ?], offset: ?>>, memref<?x?xf32, strided<[?, ?], offset: ?>>) outs(%alloc_8 : memref<32x32xf32>)
                      }
                      memref.copy %alloc_8, %subview : memref<32x32xf32> to memref<?x?xf32, strided<[?, ?], offset: ?>>
                    }
                  }
                  linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%arg13, %alloc_5 : memref<64x64xf32>, memref<64x64xf32>) outs(%arg13 : memref<64x64xf32>) {
                  ^bb0(%in: f32, %in_8: f32, %out: f32):
                    %11 = arith.addf %in, %in_8 : f32
                    linalg.yield %11 : f32
                  }
                  scf.yield %arg13 : memref<64x64xf32>
                }
                %8 = affine.apply affine_map<(d0, d1, d2, d3, d4, d5) -> (d0 * 4096 + d1 * 64 + d2 * 4096 + d3 * 512 + d4 * 2097152 + d5 * 32768)>(%arg11, %arg9, %arg6, %arg8, %arg10, %arg7)
                %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%8], sizes: [64, 64], strides: [512, 1] {tmd.reuse = {spatial = [{depth = 2 : i64, iterator = "%arg8", mapped_to = "y", reuse_type = "no_reuse", volume = 0 : i64}, {depth = 3 : i64, iterator = "%arg9", mapped_to = "x", reuse_type = "no_reuse", volume = 0 : i64}], temporal = [{depth = 0 : i64, iterator = "%arg6", reuse_type = "no_reuse", volume = 0 : i64}, {depth = 1 : i64, iterator = "%arg7", reuse_type = "no_reuse", volume = 0 : i64}, {depth = 4 : i64, iterator = "%arg10", reuse_type = "no_reuse", volume = 0 : i64}, {depth = 5 : i64, iterator = "%arg11", reuse_type = "no_reuse", volume = 0 : i64}]}} : memref<*xf32> to memref<64x64xf32, strided<[512, 1], offset: ?>>
                memref.copy %7, %reinterpret_cast : memref<64x64xf32> to memref<64x64xf32, strided<[512, 1], offset: ?>>
              }
            }
          } {tmd.mapped_to = "x"}
        } {tmd.mapped_to = "y"}
      }
    }
    return
  }
}
