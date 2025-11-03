module {
  %0 = df.spatial_dim "x", 8
  %1 = df.spatial_dim "y", 8
  %2 = df.compute "cores", %0, %1 {map = affine_map<(d0, d1) -> (d0, d1)>}
  %3 = df.memory "L1", %0, %1 {bandwidth = 64 : i64, map = affine_map<(d0, d1) -> (d0, d1)>, size = 32768 : i64}
  %4 = df.mux %2 : !df.compute, %3 : !df.memory, %0, %1 {map = affine_map<(d0, d1) -> (d0, d1)>}
  %5 = df.interconnects "horizontal_links" %3 : !df.memory, %3 : !df.memory, %0, %1 {map = affine_map<(d0, d1) -> (d0 + 1, d1)>} : !df.interconnect
  %6 = df.interconnects "vertical_links" %3 : !df.memory, %3 : !df.memory, %0, %1 {map = affine_map<(d0, d1) -> (d0, d1 + 1)>} : !df.interconnect
  func.func @matmul_kernel__d0i0_d1i0__c0mem_c1mem(%arg0: memref<*xf32> {tt.divisibility = 16 : i32}, %arg1: memref<*xf32> {tt.divisibility = 16 : i32}, %arg2: memref<*xf32> {tt.divisibility = 16 : i32}, %arg3: index, %arg4: index, %arg5: index) {
    affine.for %arg6 = 0 to affine_map<(d0, d1) -> ((d0 ceildiv 8) ceildiv 8)>(%arg3, %arg4) {
      affine.for %arg7 = 0 to affine_map<(d0, d1) -> (d1)>(%arg3, %arg4) {
        affine.parallel (%arg8) = (0) to (8) {
          affine.parallel (%arg9) = (0) to (8) {
            %cst = arith.constant 0.000000e+00 : f32
            %alloc = memref.alloc() {alignment = 64 : i64} : memref<32x32xf32>
            linalg.fill ins(%cst : f32) outs(%alloc : memref<32x32xf32>)
            %c0 = arith.constant 0 : index
            %c1 = arith.constant 1 : index
            %alloc_0 = memref.alloc() {alignment = 64 : i64} : memref<32x32xf32>
            memref.copy %alloc, %alloc_0 : memref<32x32xf32> to memref<32x32xf32>
            %c4 = arith.constant 4 : index
            %7 = scf.for %arg10 = %c0 to %c4 step %c1 iter_args(%arg11 = %alloc_0) -> (memref<32x32xf32>) {
              %9 = scf.for %arg12 = %c0 to %c4 step %c1 iter_args(%arg13 = %arg11) -> (memref<32x32xf32>) {
                %10 = affine.apply affine_map<(d0, d1, d2, d3, d4) -> (d0 * 16384 + d1 * 1048576 + d2 * 131072 + d3 * 128 + d4 * 32)>(%arg9, %arg6, %arg8, %arg10, %arg12)
                %reinterpret_cast_1 = memref.reinterpret_cast %arg0 to offset: [%10], sizes: [32, 32], strides: [512, 1] {tmd.reuse = {sequential = [{depth = 4 : i64, iterator = "%arg10", reuse_type = "no_reuse", volume = 0 : i64}], spatial = [{depth = 2 : i64, iterator = "%arg8", mapped_to = "y", reuse_type = "no_reuse", volume = 0 : i64}, {depth = 3 : i64, iterator = "%arg9", mapped_to = "x", reuse_type = "no_reuse", volume = 0 : i64}], temporal = [{depth = 0 : i64, iterator = "%arg6", reuse_type = "no_reuse", volume = 0 : i64}, {depth = 1 : i64, iterator = "%arg7", reuse_type = "total_reuse", volume = 4096 : i64}]}} : memref<*xf32> to memref<32x32xf32, strided<[512, 1], offset: ?>>
                %alloc_2 = memref.alloc() {tmd.alloc = {local = true, memory_name = "L1", size = 4096 : i64}} : memref<32x32xf32>
                memref.copy %reinterpret_cast_1, %alloc_2 {tmd.copy.choice = {kind = "mem", memory_name = "L1"}} : memref<32x32xf32, strided<[512, 1], offset: ?>> to memref<32x32xf32>
                %11 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 32 + d1 * 65536 + d2 * 16384)>(%arg7, %arg10, %arg12)
                %reinterpret_cast_3 = memref.reinterpret_cast %arg1 to offset: [%11], sizes: [32, 32], strides: [512, 1] {tmd.reuse = {sequential = [{depth = 4 : i64, iterator = "%arg10", reuse_type = "no_reuse", volume = 0 : i64}], spatial = [{depth = 2 : i64, iterator = "%arg8", mapped_to = "y", reuse_type = "total_reuse", volume = 4096 : i64}, {depth = 3 : i64, iterator = "%arg9", mapped_to = "x", reuse_type = "total_reuse", volume = 4096 : i64}], temporal = [{depth = 0 : i64, iterator = "%arg6", reuse_type = "total_reuse", volume = 4096 : i64}, {depth = 1 : i64, iterator = "%arg7", reuse_type = "no_reuse", volume = 0 : i64}]}} : memref<*xf32> to memref<32x32xf32, strided<[512, 1], offset: ?>>
                %alloc_4 = memref.alloc() {tmd.alloc = {local = true, memory_name = "L1", size = 4096 : i64}} : memref<32x32xf32>
                memref.copy %reinterpret_cast_3, %alloc_4 {tmd.copy.choice = {kind = "mem", memory_name = "L1"}} : memref<32x32xf32, strided<[512, 1], offset: ?>> to memref<32x32xf32>
                %alloc_5 = memref.alloc() {alignment = 64 : i64} : memref<32x32xf32>
                memref.copy %alloc, %alloc_5 : memref<32x32xf32> to memref<32x32xf32>
                linalg.matmul ins(%alloc_2, %alloc_4 : memref<32x32xf32>, memref<32x32xf32>) outs(%alloc_5 : memref<32x32xf32>)
                linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%arg13, %alloc_5 : memref<32x32xf32>, memref<32x32xf32>) outs(%arg13 : memref<32x32xf32>) {
                ^bb0(%in: f32, %in_6: f32, %out: f32):
                  %12 = arith.addf %in, %in_6 : f32
                  linalg.yield %12 : f32
                }
                scf.yield %arg13 : memref<32x32xf32>
              }
              scf.yield %9 : memref<32x32xf32>
            }
            %8 = affine.apply affine_map<(d0, d1, d2, d3) -> (d0 * 32 + d1 * 16384 + d2 * 1048576 + d3 * 131072)>(%arg7, %arg9, %arg6, %arg8)
            %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%8], sizes: [32, 32], strides: [512, 1] {tmd.reuse = {spatial = [{depth = 2 : i64, iterator = "%arg8", mapped_to = "y", reuse_type = "no_reuse", volume = 0 : i64}, {depth = 3 : i64, iterator = "%arg9", mapped_to = "x", reuse_type = "no_reuse", volume = 0 : i64}], temporal = [{depth = 0 : i64, iterator = "%arg6", reuse_type = "no_reuse", volume = 0 : i64}, {depth = 1 : i64, iterator = "%arg7", reuse_type = "no_reuse", volume = 0 : i64}]}} : memref<*xf32> to memref<32x32xf32, strided<[512, 1], offset: ?>>
            memref.copy %7, %reinterpret_cast : memref<32x32xf32> to memref<32x32xf32, strided<[512, 1], offset: ?>>
          } {tmd.mapped_to = "x"}
        } {tmd.mapped_to = "y"}
      }
    }
    return
  }
  func.func @matmul_kernel__d0i0_d1i0__c0mem_c1bx(%arg0: memref<*xf32> {tt.divisibility = 16 : i32}, %arg1: memref<*xf32> {tt.divisibility = 16 : i32}, %arg2: memref<*xf32> {tt.divisibility = 16 : i32}, %arg3: index, %arg4: index, %arg5: index) {
    affine.for %arg6 = 0 to affine_map<(d0, d1) -> ((d0 ceildiv 8) ceildiv 8)>(%arg3, %arg4) {
      affine.for %arg7 = 0 to affine_map<(d0, d1) -> (d1)>(%arg3, %arg4) {
        affine.parallel (%arg8) = (0) to (8) {
          affine.parallel (%arg9) = (0) to (8) {
            %cst = arith.constant 0.000000e+00 : f32
            %alloc = memref.alloc() {alignment = 64 : i64} : memref<32x32xf32>
            linalg.fill ins(%cst : f32) outs(%alloc : memref<32x32xf32>)
            %c0 = arith.constant 0 : index
            %c1 = arith.constant 1 : index
            %alloc_0 = memref.alloc() {alignment = 64 : i64} : memref<32x32xf32>
            memref.copy %alloc, %alloc_0 : memref<32x32xf32> to memref<32x32xf32>
            %c4 = arith.constant 4 : index
            %7 = scf.for %arg10 = %c0 to %c4 step %c1 iter_args(%arg11 = %alloc_0) -> (memref<32x32xf32>) {
              %9 = scf.for %arg12 = %c0 to %c4 step %c1 iter_args(%arg13 = %arg11) -> (memref<32x32xf32>) {
                %10 = affine.apply affine_map<(d0, d1, d2, d3, d4) -> (d0 * 16384 + d1 * 1048576 + d2 * 131072 + d3 * 128 + d4 * 32)>(%arg9, %arg6, %arg8, %arg10, %arg12)
                %reinterpret_cast_1 = memref.reinterpret_cast %arg0 to offset: [%10], sizes: [32, 32], strides: [512, 1] {tmd.reuse = {sequential = [{depth = 4 : i64, iterator = "%arg10", reuse_type = "no_reuse", volume = 0 : i64}], spatial = [{depth = 2 : i64, iterator = "%arg8", mapped_to = "y", reuse_type = "no_reuse", volume = 0 : i64}, {depth = 3 : i64, iterator = "%arg9", mapped_to = "x", reuse_type = "no_reuse", volume = 0 : i64}], temporal = [{depth = 0 : i64, iterator = "%arg6", reuse_type = "no_reuse", volume = 0 : i64}, {depth = 1 : i64, iterator = "%arg7", reuse_type = "total_reuse", volume = 4096 : i64}]}} : memref<*xf32> to memref<32x32xf32, strided<[512, 1], offset: ?>>
                %alloc_2 = memref.alloc() {tmd.alloc = {local = true, memory_name = "L1", size = 4096 : i64}} : memref<32x32xf32>
                memref.copy %reinterpret_cast_1, %alloc_2 {tmd.copy.choice = {kind = "mem", memory_name = "L1"}} : memref<32x32xf32, strided<[512, 1], offset: ?>> to memref<32x32xf32>
                %11 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 32 + d1 * 65536 + d2 * 16384)>(%arg7, %arg10, %arg12)
                %reinterpret_cast_3 = memref.reinterpret_cast %arg1 to offset: [%11], sizes: [32, 32], strides: [512, 1] {tmd.reuse = {sequential = [{depth = 4 : i64, iterator = "%arg10", reuse_type = "no_reuse", volume = 0 : i64}], spatial = [{depth = 2 : i64, iterator = "%arg8", mapped_to = "y", reuse_type = "total_reuse", volume = 4096 : i64}, {depth = 3 : i64, iterator = "%arg9", mapped_to = "x", reuse_type = "total_reuse", volume = 4096 : i64}], temporal = [{depth = 0 : i64, iterator = "%arg6", reuse_type = "total_reuse", volume = 4096 : i64}, {depth = 1 : i64, iterator = "%arg7", reuse_type = "no_reuse", volume = 0 : i64}]}} : memref<*xf32> to memref<32x32xf32, strided<[512, 1], offset: ?>>
                %alloc_4 = memref.alloc() {tmd.alloc = {local = true, memory_name = "L1", size = 4096 : i64}} : memref<32x32xf32>
                memref.copy %reinterpret_cast_3, %alloc_4 {tmd.copy.choice = {dim = "x", interconnect_name = "horizontal_links", kind = "broadcast"}} : memref<32x32xf32, strided<[512, 1], offset: ?>> to memref<32x32xf32>
                %alloc_5 = memref.alloc() {alignment = 64 : i64} : memref<32x32xf32>
                memref.copy %alloc, %alloc_5 : memref<32x32xf32> to memref<32x32xf32>
                linalg.matmul ins(%alloc_2, %alloc_4 : memref<32x32xf32>, memref<32x32xf32>) outs(%alloc_5 : memref<32x32xf32>)
                linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%arg13, %alloc_5 : memref<32x32xf32>, memref<32x32xf32>) outs(%arg13 : memref<32x32xf32>) {
                ^bb0(%in: f32, %in_6: f32, %out: f32):
                  %12 = arith.addf %in, %in_6 : f32
                  linalg.yield %12 : f32
                }
                scf.yield %arg13 : memref<32x32xf32>
              }
              scf.yield %9 : memref<32x32xf32>
            }
            %8 = affine.apply affine_map<(d0, d1, d2, d3) -> (d0 * 32 + d1 * 16384 + d2 * 1048576 + d3 * 131072)>(%arg7, %arg9, %arg6, %arg8)
            %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%8], sizes: [32, 32], strides: [512, 1] {tmd.reuse = {spatial = [{depth = 2 : i64, iterator = "%arg8", mapped_to = "y", reuse_type = "no_reuse", volume = 0 : i64}, {depth = 3 : i64, iterator = "%arg9", mapped_to = "x", reuse_type = "no_reuse", volume = 0 : i64}], temporal = [{depth = 0 : i64, iterator = "%arg6", reuse_type = "no_reuse", volume = 0 : i64}, {depth = 1 : i64, iterator = "%arg7", reuse_type = "no_reuse", volume = 0 : i64}]}} : memref<*xf32> to memref<32x32xf32, strided<[512, 1], offset: ?>>
            memref.copy %7, %reinterpret_cast : memref<32x32xf32> to memref<32x32xf32, strided<[512, 1], offset: ?>>
          } {tmd.mapped_to = "x"}
        } {tmd.mapped_to = "y"}
      }
    }
    return
  }
  func.func @matmul_kernel__d0i0_d1i0__c0mem_c1by(%arg0: memref<*xf32> {tt.divisibility = 16 : i32}, %arg1: memref<*xf32> {tt.divisibility = 16 : i32}, %arg2: memref<*xf32> {tt.divisibility = 16 : i32}, %arg3: index, %arg4: index, %arg5: index) {
    affine.for %arg6 = 0 to affine_map<(d0, d1) -> ((d0 ceildiv 8) ceildiv 8)>(%arg3, %arg4) {
      affine.for %arg7 = 0 to affine_map<(d0, d1) -> (d1)>(%arg3, %arg4) {
        affine.parallel (%arg8) = (0) to (8) {
          affine.parallel (%arg9) = (0) to (8) {
            %cst = arith.constant 0.000000e+00 : f32
            %alloc = memref.alloc() {alignment = 64 : i64} : memref<32x32xf32>
            linalg.fill ins(%cst : f32) outs(%alloc : memref<32x32xf32>)
            %c0 = arith.constant 0 : index
            %c1 = arith.constant 1 : index
            %alloc_0 = memref.alloc() {alignment = 64 : i64} : memref<32x32xf32>
            memref.copy %alloc, %alloc_0 : memref<32x32xf32> to memref<32x32xf32>
            %c4 = arith.constant 4 : index
            %7 = scf.for %arg10 = %c0 to %c4 step %c1 iter_args(%arg11 = %alloc_0) -> (memref<32x32xf32>) {
              %9 = scf.for %arg12 = %c0 to %c4 step %c1 iter_args(%arg13 = %arg11) -> (memref<32x32xf32>) {
                %10 = affine.apply affine_map<(d0, d1, d2, d3, d4) -> (d0 * 16384 + d1 * 1048576 + d2 * 131072 + d3 * 128 + d4 * 32)>(%arg9, %arg6, %arg8, %arg10, %arg12)
                %reinterpret_cast_1 = memref.reinterpret_cast %arg0 to offset: [%10], sizes: [32, 32], strides: [512, 1] {tmd.reuse = {sequential = [{depth = 4 : i64, iterator = "%arg10", reuse_type = "no_reuse", volume = 0 : i64}], spatial = [{depth = 2 : i64, iterator = "%arg8", mapped_to = "y", reuse_type = "no_reuse", volume = 0 : i64}, {depth = 3 : i64, iterator = "%arg9", mapped_to = "x", reuse_type = "no_reuse", volume = 0 : i64}], temporal = [{depth = 0 : i64, iterator = "%arg6", reuse_type = "no_reuse", volume = 0 : i64}, {depth = 1 : i64, iterator = "%arg7", reuse_type = "total_reuse", volume = 4096 : i64}]}} : memref<*xf32> to memref<32x32xf32, strided<[512, 1], offset: ?>>
                %alloc_2 = memref.alloc() {tmd.alloc = {local = true, memory_name = "L1", size = 4096 : i64}} : memref<32x32xf32>
                memref.copy %reinterpret_cast_1, %alloc_2 {tmd.copy.choice = {kind = "mem", memory_name = "L1"}} : memref<32x32xf32, strided<[512, 1], offset: ?>> to memref<32x32xf32>
                %11 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 32 + d1 * 65536 + d2 * 16384)>(%arg7, %arg10, %arg12)
                %reinterpret_cast_3 = memref.reinterpret_cast %arg1 to offset: [%11], sizes: [32, 32], strides: [512, 1] {tmd.reuse = {sequential = [{depth = 4 : i64, iterator = "%arg10", reuse_type = "no_reuse", volume = 0 : i64}], spatial = [{depth = 2 : i64, iterator = "%arg8", mapped_to = "y", reuse_type = "total_reuse", volume = 4096 : i64}, {depth = 3 : i64, iterator = "%arg9", mapped_to = "x", reuse_type = "total_reuse", volume = 4096 : i64}], temporal = [{depth = 0 : i64, iterator = "%arg6", reuse_type = "total_reuse", volume = 4096 : i64}, {depth = 1 : i64, iterator = "%arg7", reuse_type = "no_reuse", volume = 0 : i64}]}} : memref<*xf32> to memref<32x32xf32, strided<[512, 1], offset: ?>>
                %alloc_4 = memref.alloc() {tmd.alloc = {local = true, memory_name = "L1", size = 4096 : i64}} : memref<32x32xf32>
                memref.copy %reinterpret_cast_3, %alloc_4 {tmd.copy.choice = {dim = "y", interconnect_name = "vertical_links", kind = "broadcast"}} : memref<32x32xf32, strided<[512, 1], offset: ?>> to memref<32x32xf32>
                %alloc_5 = memref.alloc() {alignment = 64 : i64} : memref<32x32xf32>
                memref.copy %alloc, %alloc_5 : memref<32x32xf32> to memref<32x32xf32>
                linalg.matmul ins(%alloc_2, %alloc_4 : memref<32x32xf32>, memref<32x32xf32>) outs(%alloc_5 : memref<32x32xf32>)
                linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%arg13, %alloc_5 : memref<32x32xf32>, memref<32x32xf32>) outs(%arg13 : memref<32x32xf32>) {
                ^bb0(%in: f32, %in_6: f32, %out: f32):
                  %12 = arith.addf %in, %in_6 : f32
                  linalg.yield %12 : f32
                }
                scf.yield %arg13 : memref<32x32xf32>
              }
              scf.yield %9 : memref<32x32xf32>
            }
            %8 = affine.apply affine_map<(d0, d1, d2, d3) -> (d0 * 32 + d1 * 16384 + d2 * 1048576 + d3 * 131072)>(%arg7, %arg9, %arg6, %arg8)
            %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%8], sizes: [32, 32], strides: [512, 1] {tmd.reuse = {spatial = [{depth = 2 : i64, iterator = "%arg8", mapped_to = "y", reuse_type = "no_reuse", volume = 0 : i64}, {depth = 3 : i64, iterator = "%arg9", mapped_to = "x", reuse_type = "no_reuse", volume = 0 : i64}], temporal = [{depth = 0 : i64, iterator = "%arg6", reuse_type = "no_reuse", volume = 0 : i64}, {depth = 1 : i64, iterator = "%arg7", reuse_type = "no_reuse", volume = 0 : i64}]}} : memref<*xf32> to memref<32x32xf32, strided<[512, 1], offset: ?>>
            memref.copy %7, %reinterpret_cast : memref<32x32xf32> to memref<32x32xf32, strided<[512, 1], offset: ?>>
          } {tmd.mapped_to = "x"}
        } {tmd.mapped_to = "y"}
      }
    }
    return
  }
  func.func @matmul_kernel__d1i0_d0i0__c0mem_c1mem(%arg0: memref<*xf32> {tt.divisibility = 16 : i32}, %arg1: memref<*xf32> {tt.divisibility = 16 : i32}, %arg2: memref<*xf32> {tt.divisibility = 16 : i32}, %arg3: index, %arg4: index, %arg5: index) {
    affine.for %arg6 = 0 to affine_map<(d0, d1) -> ((d0 ceildiv 8) ceildiv 8)>(%arg3, %arg4) {
      affine.for %arg7 = 0 to affine_map<(d0, d1) -> (d1)>(%arg3, %arg4) {
        affine.parallel (%arg8) = (0) to (8) {
          affine.parallel (%arg9) = (0) to (8) {
            %cst = arith.constant 0.000000e+00 : f32
            %alloc = memref.alloc() {alignment = 64 : i64} : memref<32x32xf32>
            linalg.fill ins(%cst : f32) outs(%alloc : memref<32x32xf32>)
            %c0 = arith.constant 0 : index
            %c1 = arith.constant 1 : index
            %alloc_0 = memref.alloc() {alignment = 64 : i64} : memref<32x32xf32>
            memref.copy %alloc, %alloc_0 : memref<32x32xf32> to memref<32x32xf32>
            %c4 = arith.constant 4 : index
            %7 = scf.for %arg10 = %c0 to %c4 step %c1 iter_args(%arg11 = %alloc_0) -> (memref<32x32xf32>) {
              %9 = scf.for %arg12 = %c0 to %c4 step %c1 iter_args(%arg13 = %arg11) -> (memref<32x32xf32>) {
                %10 = affine.apply affine_map<(d0, d1, d2, d3, d4) -> (d0 * 16384 + d1 * 1048576 + d2 * 131072 + d3 * 128 + d4 * 32)>(%arg9, %arg6, %arg8, %arg10, %arg12)
                %reinterpret_cast_1 = memref.reinterpret_cast %arg0 to offset: [%10], sizes: [32, 32], strides: [512, 1] {tmd.reuse = {sequential = [{depth = 4 : i64, iterator = "%arg10", reuse_type = "no_reuse", volume = 0 : i64}], spatial = [{depth = 2 : i64, iterator = "%arg8", mapped_to = "x", reuse_type = "no_reuse", volume = 0 : i64}, {depth = 3 : i64, iterator = "%arg9", mapped_to = "y", reuse_type = "no_reuse", volume = 0 : i64}], temporal = [{depth = 0 : i64, iterator = "%arg6", reuse_type = "no_reuse", volume = 0 : i64}, {depth = 1 : i64, iterator = "%arg7", reuse_type = "total_reuse", volume = 4096 : i64}]}} : memref<*xf32> to memref<32x32xf32, strided<[512, 1], offset: ?>>
                %alloc_2 = memref.alloc() {tmd.alloc = {local = true, memory_name = "L1", size = 4096 : i64}} : memref<32x32xf32>
                memref.copy %reinterpret_cast_1, %alloc_2 {tmd.copy.choice = {kind = "mem", memory_name = "L1"}} : memref<32x32xf32, strided<[512, 1], offset: ?>> to memref<32x32xf32>
                %11 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 32 + d1 * 65536 + d2 * 16384)>(%arg7, %arg10, %arg12)
                %reinterpret_cast_3 = memref.reinterpret_cast %arg1 to offset: [%11], sizes: [32, 32], strides: [512, 1] {tmd.reuse = {sequential = [{depth = 4 : i64, iterator = "%arg10", reuse_type = "no_reuse", volume = 0 : i64}], spatial = [{depth = 2 : i64, iterator = "%arg8", mapped_to = "x", reuse_type = "total_reuse", volume = 4096 : i64}, {depth = 3 : i64, iterator = "%arg9", mapped_to = "y", reuse_type = "total_reuse", volume = 4096 : i64}], temporal = [{depth = 0 : i64, iterator = "%arg6", reuse_type = "total_reuse", volume = 4096 : i64}, {depth = 1 : i64, iterator = "%arg7", reuse_type = "no_reuse", volume = 0 : i64}]}} : memref<*xf32> to memref<32x32xf32, strided<[512, 1], offset: ?>>
                %alloc_4 = memref.alloc() {tmd.alloc = {local = true, memory_name = "L1", size = 4096 : i64}} : memref<32x32xf32>
                memref.copy %reinterpret_cast_3, %alloc_4 {tmd.copy.choice = {kind = "mem", memory_name = "L1"}} : memref<32x32xf32, strided<[512, 1], offset: ?>> to memref<32x32xf32>
                %alloc_5 = memref.alloc() {alignment = 64 : i64} : memref<32x32xf32>
                memref.copy %alloc, %alloc_5 : memref<32x32xf32> to memref<32x32xf32>
                linalg.matmul ins(%alloc_2, %alloc_4 : memref<32x32xf32>, memref<32x32xf32>) outs(%alloc_5 : memref<32x32xf32>)
                linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%arg13, %alloc_5 : memref<32x32xf32>, memref<32x32xf32>) outs(%arg13 : memref<32x32xf32>) {
                ^bb0(%in: f32, %in_6: f32, %out: f32):
                  %12 = arith.addf %in, %in_6 : f32
                  linalg.yield %12 : f32
                }
                scf.yield %arg13 : memref<32x32xf32>
              }
              scf.yield %9 : memref<32x32xf32>
            }
            %8 = affine.apply affine_map<(d0, d1, d2, d3) -> (d0 * 32 + d1 * 16384 + d2 * 1048576 + d3 * 131072)>(%arg7, %arg9, %arg6, %arg8)
            %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%8], sizes: [32, 32], strides: [512, 1] {tmd.reuse = {spatial = [{depth = 2 : i64, iterator = "%arg8", mapped_to = "x", reuse_type = "no_reuse", volume = 0 : i64}, {depth = 3 : i64, iterator = "%arg9", mapped_to = "y", reuse_type = "no_reuse", volume = 0 : i64}], temporal = [{depth = 0 : i64, iterator = "%arg6", reuse_type = "no_reuse", volume = 0 : i64}, {depth = 1 : i64, iterator = "%arg7", reuse_type = "no_reuse", volume = 0 : i64}]}} : memref<*xf32> to memref<32x32xf32, strided<[512, 1], offset: ?>>
            memref.copy %7, %reinterpret_cast : memref<32x32xf32> to memref<32x32xf32, strided<[512, 1], offset: ?>>
          } {tmd.mapped_to = "y"}
        } {tmd.mapped_to = "x"}
      }
    }
    return
  }
  func.func @matmul_kernel__d1i0_d0i0__c0mem_c1bx(%arg0: memref<*xf32> {tt.divisibility = 16 : i32}, %arg1: memref<*xf32> {tt.divisibility = 16 : i32}, %arg2: memref<*xf32> {tt.divisibility = 16 : i32}, %arg3: index, %arg4: index, %arg5: index) {
    affine.for %arg6 = 0 to affine_map<(d0, d1) -> ((d0 ceildiv 8) ceildiv 8)>(%arg3, %arg4) {
      affine.for %arg7 = 0 to affine_map<(d0, d1) -> (d1)>(%arg3, %arg4) {
        affine.parallel (%arg8) = (0) to (8) {
          affine.parallel (%arg9) = (0) to (8) {
            %cst = arith.constant 0.000000e+00 : f32
            %alloc = memref.alloc() {alignment = 64 : i64} : memref<32x32xf32>
            linalg.fill ins(%cst : f32) outs(%alloc : memref<32x32xf32>)
            %c0 = arith.constant 0 : index
            %c1 = arith.constant 1 : index
            %alloc_0 = memref.alloc() {alignment = 64 : i64} : memref<32x32xf32>
            memref.copy %alloc, %alloc_0 : memref<32x32xf32> to memref<32x32xf32>
            %c4 = arith.constant 4 : index
            %7 = scf.for %arg10 = %c0 to %c4 step %c1 iter_args(%arg11 = %alloc_0) -> (memref<32x32xf32>) {
              %9 = scf.for %arg12 = %c0 to %c4 step %c1 iter_args(%arg13 = %arg11) -> (memref<32x32xf32>) {
                %10 = affine.apply affine_map<(d0, d1, d2, d3, d4) -> (d0 * 16384 + d1 * 1048576 + d2 * 131072 + d3 * 128 + d4 * 32)>(%arg9, %arg6, %arg8, %arg10, %arg12)
                %reinterpret_cast_1 = memref.reinterpret_cast %arg0 to offset: [%10], sizes: [32, 32], strides: [512, 1] {tmd.reuse = {sequential = [{depth = 4 : i64, iterator = "%arg10", reuse_type = "no_reuse", volume = 0 : i64}], spatial = [{depth = 2 : i64, iterator = "%arg8", mapped_to = "x", reuse_type = "no_reuse", volume = 0 : i64}, {depth = 3 : i64, iterator = "%arg9", mapped_to = "y", reuse_type = "no_reuse", volume = 0 : i64}], temporal = [{depth = 0 : i64, iterator = "%arg6", reuse_type = "no_reuse", volume = 0 : i64}, {depth = 1 : i64, iterator = "%arg7", reuse_type = "total_reuse", volume = 4096 : i64}]}} : memref<*xf32> to memref<32x32xf32, strided<[512, 1], offset: ?>>
                %alloc_2 = memref.alloc() {tmd.alloc = {local = true, memory_name = "L1", size = 4096 : i64}} : memref<32x32xf32>
                memref.copy %reinterpret_cast_1, %alloc_2 {tmd.copy.choice = {kind = "mem", memory_name = "L1"}} : memref<32x32xf32, strided<[512, 1], offset: ?>> to memref<32x32xf32>
                %11 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 32 + d1 * 65536 + d2 * 16384)>(%arg7, %arg10, %arg12)
                %reinterpret_cast_3 = memref.reinterpret_cast %arg1 to offset: [%11], sizes: [32, 32], strides: [512, 1] {tmd.reuse = {sequential = [{depth = 4 : i64, iterator = "%arg10", reuse_type = "no_reuse", volume = 0 : i64}], spatial = [{depth = 2 : i64, iterator = "%arg8", mapped_to = "x", reuse_type = "total_reuse", volume = 4096 : i64}, {depth = 3 : i64, iterator = "%arg9", mapped_to = "y", reuse_type = "total_reuse", volume = 4096 : i64}], temporal = [{depth = 0 : i64, iterator = "%arg6", reuse_type = "total_reuse", volume = 4096 : i64}, {depth = 1 : i64, iterator = "%arg7", reuse_type = "no_reuse", volume = 0 : i64}]}} : memref<*xf32> to memref<32x32xf32, strided<[512, 1], offset: ?>>
                %alloc_4 = memref.alloc() {tmd.alloc = {local = true, memory_name = "L1", size = 4096 : i64}} : memref<32x32xf32>
                memref.copy %reinterpret_cast_3, %alloc_4 {tmd.copy.choice = {dim = "x", interconnect_name = "horizontal_links", kind = "broadcast"}} : memref<32x32xf32, strided<[512, 1], offset: ?>> to memref<32x32xf32>
                %alloc_5 = memref.alloc() {alignment = 64 : i64} : memref<32x32xf32>
                memref.copy %alloc, %alloc_5 : memref<32x32xf32> to memref<32x32xf32>
                linalg.matmul ins(%alloc_2, %alloc_4 : memref<32x32xf32>, memref<32x32xf32>) outs(%alloc_5 : memref<32x32xf32>)
                linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%arg13, %alloc_5 : memref<32x32xf32>, memref<32x32xf32>) outs(%arg13 : memref<32x32xf32>) {
                ^bb0(%in: f32, %in_6: f32, %out: f32):
                  %12 = arith.addf %in, %in_6 : f32
                  linalg.yield %12 : f32
                }
                scf.yield %arg13 : memref<32x32xf32>
              }
              scf.yield %9 : memref<32x32xf32>
            }
            %8 = affine.apply affine_map<(d0, d1, d2, d3) -> (d0 * 32 + d1 * 16384 + d2 * 1048576 + d3 * 131072)>(%arg7, %arg9, %arg6, %arg8)
            %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%8], sizes: [32, 32], strides: [512, 1] {tmd.reuse = {spatial = [{depth = 2 : i64, iterator = "%arg8", mapped_to = "x", reuse_type = "no_reuse", volume = 0 : i64}, {depth = 3 : i64, iterator = "%arg9", mapped_to = "y", reuse_type = "no_reuse", volume = 0 : i64}], temporal = [{depth = 0 : i64, iterator = "%arg6", reuse_type = "no_reuse", volume = 0 : i64}, {depth = 1 : i64, iterator = "%arg7", reuse_type = "no_reuse", volume = 0 : i64}]}} : memref<*xf32> to memref<32x32xf32, strided<[512, 1], offset: ?>>
            memref.copy %7, %reinterpret_cast : memref<32x32xf32> to memref<32x32xf32, strided<[512, 1], offset: ?>>
          } {tmd.mapped_to = "y"}
        } {tmd.mapped_to = "x"}
      }
    }
    return
  }
  func.func @matmul_kernel__d1i0_d0i0__c0mem_c1by(%arg0: memref<*xf32> {tt.divisibility = 16 : i32}, %arg1: memref<*xf32> {tt.divisibility = 16 : i32}, %arg2: memref<*xf32> {tt.divisibility = 16 : i32}, %arg3: index, %arg4: index, %arg5: index) {
    affine.for %arg6 = 0 to affine_map<(d0, d1) -> ((d0 ceildiv 8) ceildiv 8)>(%arg3, %arg4) {
      affine.for %arg7 = 0 to affine_map<(d0, d1) -> (d1)>(%arg3, %arg4) {
        affine.parallel (%arg8) = (0) to (8) {
          affine.parallel (%arg9) = (0) to (8) {
            %cst = arith.constant 0.000000e+00 : f32
            %alloc = memref.alloc() {alignment = 64 : i64} : memref<32x32xf32>
            linalg.fill ins(%cst : f32) outs(%alloc : memref<32x32xf32>)
            %c0 = arith.constant 0 : index
            %c1 = arith.constant 1 : index
            %alloc_0 = memref.alloc() {alignment = 64 : i64} : memref<32x32xf32>
            memref.copy %alloc, %alloc_0 : memref<32x32xf32> to memref<32x32xf32>
            %c4 = arith.constant 4 : index
            %7 = scf.for %arg10 = %c0 to %c4 step %c1 iter_args(%arg11 = %alloc_0) -> (memref<32x32xf32>) {
              %9 = scf.for %arg12 = %c0 to %c4 step %c1 iter_args(%arg13 = %arg11) -> (memref<32x32xf32>) {
                %10 = affine.apply affine_map<(d0, d1, d2, d3, d4) -> (d0 * 16384 + d1 * 1048576 + d2 * 131072 + d3 * 128 + d4 * 32)>(%arg9, %arg6, %arg8, %arg10, %arg12)
                %reinterpret_cast_1 = memref.reinterpret_cast %arg0 to offset: [%10], sizes: [32, 32], strides: [512, 1] {tmd.reuse = {sequential = [{depth = 4 : i64, iterator = "%arg10", reuse_type = "no_reuse", volume = 0 : i64}], spatial = [{depth = 2 : i64, iterator = "%arg8", mapped_to = "x", reuse_type = "no_reuse", volume = 0 : i64}, {depth = 3 : i64, iterator = "%arg9", mapped_to = "y", reuse_type = "no_reuse", volume = 0 : i64}], temporal = [{depth = 0 : i64, iterator = "%arg6", reuse_type = "no_reuse", volume = 0 : i64}, {depth = 1 : i64, iterator = "%arg7", reuse_type = "total_reuse", volume = 4096 : i64}]}} : memref<*xf32> to memref<32x32xf32, strided<[512, 1], offset: ?>>
                %alloc_2 = memref.alloc() {tmd.alloc = {local = true, memory_name = "L1", size = 4096 : i64}} : memref<32x32xf32>
                memref.copy %reinterpret_cast_1, %alloc_2 {tmd.copy.choice = {kind = "mem", memory_name = "L1"}} : memref<32x32xf32, strided<[512, 1], offset: ?>> to memref<32x32xf32>
                %11 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 32 + d1 * 65536 + d2 * 16384)>(%arg7, %arg10, %arg12)
                %reinterpret_cast_3 = memref.reinterpret_cast %arg1 to offset: [%11], sizes: [32, 32], strides: [512, 1] {tmd.reuse = {sequential = [{depth = 4 : i64, iterator = "%arg10", reuse_type = "no_reuse", volume = 0 : i64}], spatial = [{depth = 2 : i64, iterator = "%arg8", mapped_to = "x", reuse_type = "total_reuse", volume = 4096 : i64}, {depth = 3 : i64, iterator = "%arg9", mapped_to = "y", reuse_type = "total_reuse", volume = 4096 : i64}], temporal = [{depth = 0 : i64, iterator = "%arg6", reuse_type = "total_reuse", volume = 4096 : i64}, {depth = 1 : i64, iterator = "%arg7", reuse_type = "no_reuse", volume = 0 : i64}]}} : memref<*xf32> to memref<32x32xf32, strided<[512, 1], offset: ?>>
                %alloc_4 = memref.alloc() {tmd.alloc = {local = true, memory_name = "L1", size = 4096 : i64}} : memref<32x32xf32>
                memref.copy %reinterpret_cast_3, %alloc_4 {tmd.copy.choice = {dim = "y", interconnect_name = "vertical_links", kind = "broadcast"}} : memref<32x32xf32, strided<[512, 1], offset: ?>> to memref<32x32xf32>
                %alloc_5 = memref.alloc() {alignment = 64 : i64} : memref<32x32xf32>
                memref.copy %alloc, %alloc_5 : memref<32x32xf32> to memref<32x32xf32>
                linalg.matmul ins(%alloc_2, %alloc_4 : memref<32x32xf32>, memref<32x32xf32>) outs(%alloc_5 : memref<32x32xf32>)
                linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%arg13, %alloc_5 : memref<32x32xf32>, memref<32x32xf32>) outs(%arg13 : memref<32x32xf32>) {
                ^bb0(%in: f32, %in_6: f32, %out: f32):
                  %12 = arith.addf %in, %in_6 : f32
                  linalg.yield %12 : f32
                }
                scf.yield %arg13 : memref<32x32xf32>
              }
              scf.yield %9 : memref<32x32xf32>
            }
            %8 = affine.apply affine_map<(d0, d1, d2, d3) -> (d0 * 32 + d1 * 16384 + d2 * 1048576 + d3 * 131072)>(%arg7, %arg9, %arg6, %arg8)
            %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%8], sizes: [32, 32], strides: [512, 1] {tmd.reuse = {spatial = [{depth = 2 : i64, iterator = "%arg8", mapped_to = "x", reuse_type = "no_reuse", volume = 0 : i64}, {depth = 3 : i64, iterator = "%arg9", mapped_to = "y", reuse_type = "no_reuse", volume = 0 : i64}], temporal = [{depth = 0 : i64, iterator = "%arg6", reuse_type = "no_reuse", volume = 0 : i64}, {depth = 1 : i64, iterator = "%arg7", reuse_type = "no_reuse", volume = 0 : i64}]}} : memref<*xf32> to memref<32x32xf32, strided<[512, 1], offset: ?>>
            memref.copy %7, %reinterpret_cast : memref<32x32xf32> to memref<32x32xf32, strided<[512, 1], offset: ?>>
          } {tmd.mapped_to = "y"}
        } {tmd.mapped_to = "x"}
      }
    }
    return
  }
  func.func @matmul_kernel__d0i0_d1i1__c0mem_c1mem(%arg0: memref<*xf32> {tt.divisibility = 16 : i32}, %arg1: memref<*xf32> {tt.divisibility = 16 : i32}, %arg2: memref<*xf32> {tt.divisibility = 16 : i32}, %arg3: index, %arg4: index, %arg5: index) {
    affine.for %arg6 = 0 to affine_map<(d0, d1) -> (d0 ceildiv 8)>(%arg3, %arg4) {
      affine.for %arg7 = 0 to affine_map<(d0, d1) -> (d1 ceildiv 8)>(%arg3, %arg4) {
        affine.parallel (%arg8) = (0) to (8) {
          affine.parallel (%arg9) = (0) to (8) {
            %cst = arith.constant 0.000000e+00 : f32
            %alloc = memref.alloc() {alignment = 64 : i64} : memref<32x32xf32>
            linalg.fill ins(%cst : f32) outs(%alloc : memref<32x32xf32>)
            %c0 = arith.constant 0 : index
            %c1 = arith.constant 1 : index
            %alloc_0 = memref.alloc() {alignment = 64 : i64} : memref<32x32xf32>
            memref.copy %alloc, %alloc_0 : memref<32x32xf32> to memref<32x32xf32>
            %c4 = arith.constant 4 : index
            %7 = scf.for %arg10 = %c0 to %c4 step %c1 iter_args(%arg11 = %alloc_0) -> (memref<32x32xf32>) {
              %9 = scf.for %arg12 = %c0 to %c4 step %c1 iter_args(%arg13 = %arg11) -> (memref<32x32xf32>) {
                %10 = affine.apply affine_map<(d0, d1, d2, d3) -> (d0 * 131072 + d1 * 16384 + d2 * 128 + d3 * 32)>(%arg6, %arg9, %arg10, %arg12)
                %reinterpret_cast_1 = memref.reinterpret_cast %arg0 to offset: [%10], sizes: [32, 32], strides: [512, 1] {tmd.reuse = {sequential = [{depth = 4 : i64, iterator = "%arg10", reuse_type = "no_reuse", volume = 0 : i64}], spatial = [{depth = 2 : i64, iterator = "%arg8", mapped_to = "y", reuse_type = "total_reuse", volume = 4096 : i64}, {depth = 3 : i64, iterator = "%arg9", mapped_to = "x", reuse_type = "no_reuse", volume = 0 : i64}], temporal = [{depth = 0 : i64, iterator = "%arg6", reuse_type = "no_reuse", volume = 0 : i64}, {depth = 1 : i64, iterator = "%arg7", reuse_type = "total_reuse", volume = 4096 : i64}]}} : memref<*xf32> to memref<32x32xf32, strided<[512, 1], offset: ?>>
                %alloc_2 = memref.alloc() {tmd.alloc = {local = true, memory_name = "L1", size = 4096 : i64}} : memref<32x32xf32>
                memref.copy %reinterpret_cast_1, %alloc_2 {tmd.copy.choice = {kind = "mem", memory_name = "L1"}} : memref<32x32xf32, strided<[512, 1], offset: ?>> to memref<32x32xf32>
                %11 = affine.apply affine_map<(d0, d1, d2, d3) -> (d0 * 256 + d1 * 32 + d2 * 65536 + d3 * 16384)>(%arg7, %arg8, %arg10, %arg12)
                %reinterpret_cast_3 = memref.reinterpret_cast %arg1 to offset: [%11], sizes: [32, 32], strides: [512, 1] {tmd.reuse = {sequential = [{depth = 4 : i64, iterator = "%arg10", reuse_type = "no_reuse", volume = 0 : i64}], spatial = [{depth = 2 : i64, iterator = "%arg8", mapped_to = "y", reuse_type = "no_reuse", volume = 0 : i64}, {depth = 3 : i64, iterator = "%arg9", mapped_to = "x", reuse_type = "total_reuse", volume = 4096 : i64}], temporal = [{depth = 0 : i64, iterator = "%arg6", reuse_type = "total_reuse", volume = 4096 : i64}, {depth = 1 : i64, iterator = "%arg7", reuse_type = "no_reuse", volume = 0 : i64}]}} : memref<*xf32> to memref<32x32xf32, strided<[512, 1], offset: ?>>
                %alloc_4 = memref.alloc() {tmd.alloc = {local = true, memory_name = "L1", size = 4096 : i64}} : memref<32x32xf32>
                memref.copy %reinterpret_cast_3, %alloc_4 {tmd.copy.choice = {kind = "mem", memory_name = "L1"}} : memref<32x32xf32, strided<[512, 1], offset: ?>> to memref<32x32xf32>
                %alloc_5 = memref.alloc() {alignment = 64 : i64} : memref<32x32xf32>
                memref.copy %alloc, %alloc_5 : memref<32x32xf32> to memref<32x32xf32>
                linalg.matmul ins(%alloc_2, %alloc_4 : memref<32x32xf32>, memref<32x32xf32>) outs(%alloc_5 : memref<32x32xf32>)
                linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%arg13, %alloc_5 : memref<32x32xf32>, memref<32x32xf32>) outs(%arg13 : memref<32x32xf32>) {
                ^bb0(%in: f32, %in_6: f32, %out: f32):
                  %12 = arith.addf %in, %in_6 : f32
                  linalg.yield %12 : f32
                }
                scf.yield %arg13 : memref<32x32xf32>
              }
              scf.yield %9 : memref<32x32xf32>
            }
            %8 = affine.apply affine_map<(d0, d1, d2, d3) -> (d0 * 256 + d1 * 32 + d2 * 131072 + d3 * 16384)>(%arg7, %arg8, %arg6, %arg9)
            %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%8], sizes: [32, 32], strides: [512, 1] {tmd.reuse = {spatial = [{depth = 2 : i64, iterator = "%arg8", mapped_to = "y", reuse_type = "no_reuse", volume = 0 : i64}, {depth = 3 : i64, iterator = "%arg9", mapped_to = "x", reuse_type = "no_reuse", volume = 0 : i64}], temporal = [{depth = 0 : i64, iterator = "%arg6", reuse_type = "no_reuse", volume = 0 : i64}, {depth = 1 : i64, iterator = "%arg7", reuse_type = "no_reuse", volume = 0 : i64}]}} : memref<*xf32> to memref<32x32xf32, strided<[512, 1], offset: ?>>
            memref.copy %7, %reinterpret_cast : memref<32x32xf32> to memref<32x32xf32, strided<[512, 1], offset: ?>>
          } {tmd.mapped_to = "x"}
        } {tmd.mapped_to = "y"}
      }
    }
    return
  }
  func.func @matmul_kernel__d0i0_d1i1__c0by_c1mem(%arg0: memref<*xf32> {tt.divisibility = 16 : i32}, %arg1: memref<*xf32> {tt.divisibility = 16 : i32}, %arg2: memref<*xf32> {tt.divisibility = 16 : i32}, %arg3: index, %arg4: index, %arg5: index) {
    affine.for %arg6 = 0 to affine_map<(d0, d1) -> (d0 ceildiv 8)>(%arg3, %arg4) {
      affine.for %arg7 = 0 to affine_map<(d0, d1) -> (d1 ceildiv 8)>(%arg3, %arg4) {
        affine.parallel (%arg8) = (0) to (8) {
          affine.parallel (%arg9) = (0) to (8) {
            %cst = arith.constant 0.000000e+00 : f32
            %alloc = memref.alloc() {alignment = 64 : i64} : memref<32x32xf32>
            linalg.fill ins(%cst : f32) outs(%alloc : memref<32x32xf32>)
            %c0 = arith.constant 0 : index
            %c1 = arith.constant 1 : index
            %alloc_0 = memref.alloc() {alignment = 64 : i64} : memref<32x32xf32>
            memref.copy %alloc, %alloc_0 : memref<32x32xf32> to memref<32x32xf32>
            %c4 = arith.constant 4 : index
            %7 = scf.for %arg10 = %c0 to %c4 step %c1 iter_args(%arg11 = %alloc_0) -> (memref<32x32xf32>) {
              %9 = scf.for %arg12 = %c0 to %c4 step %c1 iter_args(%arg13 = %arg11) -> (memref<32x32xf32>) {
                %10 = affine.apply affine_map<(d0, d1, d2, d3) -> (d0 * 131072 + d1 * 16384 + d2 * 128 + d3 * 32)>(%arg6, %arg9, %arg10, %arg12)
                %reinterpret_cast_1 = memref.reinterpret_cast %arg0 to offset: [%10], sizes: [32, 32], strides: [512, 1] {tmd.reuse = {sequential = [{depth = 4 : i64, iterator = "%arg10", reuse_type = "no_reuse", volume = 0 : i64}], spatial = [{depth = 2 : i64, iterator = "%arg8", mapped_to = "y", reuse_type = "total_reuse", volume = 4096 : i64}, {depth = 3 : i64, iterator = "%arg9", mapped_to = "x", reuse_type = "no_reuse", volume = 0 : i64}], temporal = [{depth = 0 : i64, iterator = "%arg6", reuse_type = "no_reuse", volume = 0 : i64}, {depth = 1 : i64, iterator = "%arg7", reuse_type = "total_reuse", volume = 4096 : i64}]}} : memref<*xf32> to memref<32x32xf32, strided<[512, 1], offset: ?>>
                %alloc_2 = memref.alloc() {tmd.alloc = {local = true, memory_name = "L1", size = 4096 : i64}} : memref<32x32xf32>
                memref.copy %reinterpret_cast_1, %alloc_2 {tmd.copy.choice = {dim = "y", interconnect_name = "vertical_links", kind = "broadcast"}} : memref<32x32xf32, strided<[512, 1], offset: ?>> to memref<32x32xf32>
                %11 = affine.apply affine_map<(d0, d1, d2, d3) -> (d0 * 256 + d1 * 32 + d2 * 65536 + d3 * 16384)>(%arg7, %arg8, %arg10, %arg12)
                %reinterpret_cast_3 = memref.reinterpret_cast %arg1 to offset: [%11], sizes: [32, 32], strides: [512, 1] {tmd.reuse = {sequential = [{depth = 4 : i64, iterator = "%arg10", reuse_type = "no_reuse", volume = 0 : i64}], spatial = [{depth = 2 : i64, iterator = "%arg8", mapped_to = "y", reuse_type = "no_reuse", volume = 0 : i64}, {depth = 3 : i64, iterator = "%arg9", mapped_to = "x", reuse_type = "total_reuse", volume = 4096 : i64}], temporal = [{depth = 0 : i64, iterator = "%arg6", reuse_type = "total_reuse", volume = 4096 : i64}, {depth = 1 : i64, iterator = "%arg7", reuse_type = "no_reuse", volume = 0 : i64}]}} : memref<*xf32> to memref<32x32xf32, strided<[512, 1], offset: ?>>
                %alloc_4 = memref.alloc() {tmd.alloc = {local = true, memory_name = "L1", size = 4096 : i64}} : memref<32x32xf32>
                memref.copy %reinterpret_cast_3, %alloc_4 {tmd.copy.choice = {kind = "mem", memory_name = "L1"}} : memref<32x32xf32, strided<[512, 1], offset: ?>> to memref<32x32xf32>
                %alloc_5 = memref.alloc() {alignment = 64 : i64} : memref<32x32xf32>
                memref.copy %alloc, %alloc_5 : memref<32x32xf32> to memref<32x32xf32>
                linalg.matmul ins(%alloc_2, %alloc_4 : memref<32x32xf32>, memref<32x32xf32>) outs(%alloc_5 : memref<32x32xf32>)
                linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%arg13, %alloc_5 : memref<32x32xf32>, memref<32x32xf32>) outs(%arg13 : memref<32x32xf32>) {
                ^bb0(%in: f32, %in_6: f32, %out: f32):
                  %12 = arith.addf %in, %in_6 : f32
                  linalg.yield %12 : f32
                }
                scf.yield %arg13 : memref<32x32xf32>
              }
              scf.yield %9 : memref<32x32xf32>
            }
            %8 = affine.apply affine_map<(d0, d1, d2, d3) -> (d0 * 256 + d1 * 32 + d2 * 131072 + d3 * 16384)>(%arg7, %arg8, %arg6, %arg9)
            %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%8], sizes: [32, 32], strides: [512, 1] {tmd.reuse = {spatial = [{depth = 2 : i64, iterator = "%arg8", mapped_to = "y", reuse_type = "no_reuse", volume = 0 : i64}, {depth = 3 : i64, iterator = "%arg9", mapped_to = "x", reuse_type = "no_reuse", volume = 0 : i64}], temporal = [{depth = 0 : i64, iterator = "%arg6", reuse_type = "no_reuse", volume = 0 : i64}, {depth = 1 : i64, iterator = "%arg7", reuse_type = "no_reuse", volume = 0 : i64}]}} : memref<*xf32> to memref<32x32xf32, strided<[512, 1], offset: ?>>
            memref.copy %7, %reinterpret_cast : memref<32x32xf32> to memref<32x32xf32, strided<[512, 1], offset: ?>>
          } {tmd.mapped_to = "x"}
        } {tmd.mapped_to = "y"}
      }
    }
    return
  }
  func.func @matmul_kernel__d0i0_d1i1__c0mem_c1bx(%arg0: memref<*xf32> {tt.divisibility = 16 : i32}, %arg1: memref<*xf32> {tt.divisibility = 16 : i32}, %arg2: memref<*xf32> {tt.divisibility = 16 : i32}, %arg3: index, %arg4: index, %arg5: index) {
    affine.for %arg6 = 0 to affine_map<(d0, d1) -> (d0 ceildiv 8)>(%arg3, %arg4) {
      affine.for %arg7 = 0 to affine_map<(d0, d1) -> (d1 ceildiv 8)>(%arg3, %arg4) {
        affine.parallel (%arg8) = (0) to (8) {
          affine.parallel (%arg9) = (0) to (8) {
            %cst = arith.constant 0.000000e+00 : f32
            %alloc = memref.alloc() {alignment = 64 : i64} : memref<32x32xf32>
            linalg.fill ins(%cst : f32) outs(%alloc : memref<32x32xf32>)
            %c0 = arith.constant 0 : index
            %c1 = arith.constant 1 : index
            %alloc_0 = memref.alloc() {alignment = 64 : i64} : memref<32x32xf32>
            memref.copy %alloc, %alloc_0 : memref<32x32xf32> to memref<32x32xf32>
            %c4 = arith.constant 4 : index
            %7 = scf.for %arg10 = %c0 to %c4 step %c1 iter_args(%arg11 = %alloc_0) -> (memref<32x32xf32>) {
              %9 = scf.for %arg12 = %c0 to %c4 step %c1 iter_args(%arg13 = %arg11) -> (memref<32x32xf32>) {
                %10 = affine.apply affine_map<(d0, d1, d2, d3) -> (d0 * 131072 + d1 * 16384 + d2 * 128 + d3 * 32)>(%arg6, %arg9, %arg10, %arg12)
                %reinterpret_cast_1 = memref.reinterpret_cast %arg0 to offset: [%10], sizes: [32, 32], strides: [512, 1] {tmd.reuse = {sequential = [{depth = 4 : i64, iterator = "%arg10", reuse_type = "no_reuse", volume = 0 : i64}], spatial = [{depth = 2 : i64, iterator = "%arg8", mapped_to = "y", reuse_type = "total_reuse", volume = 4096 : i64}, {depth = 3 : i64, iterator = "%arg9", mapped_to = "x", reuse_type = "no_reuse", volume = 0 : i64}], temporal = [{depth = 0 : i64, iterator = "%arg6", reuse_type = "no_reuse", volume = 0 : i64}, {depth = 1 : i64, iterator = "%arg7", reuse_type = "total_reuse", volume = 4096 : i64}]}} : memref<*xf32> to memref<32x32xf32, strided<[512, 1], offset: ?>>
                %alloc_2 = memref.alloc() {tmd.alloc = {local = true, memory_name = "L1", size = 4096 : i64}} : memref<32x32xf32>
                memref.copy %reinterpret_cast_1, %alloc_2 {tmd.copy.choice = {kind = "mem", memory_name = "L1"}} : memref<32x32xf32, strided<[512, 1], offset: ?>> to memref<32x32xf32>
                %11 = affine.apply affine_map<(d0, d1, d2, d3) -> (d0 * 256 + d1 * 32 + d2 * 65536 + d3 * 16384)>(%arg7, %arg8, %arg10, %arg12)
                %reinterpret_cast_3 = memref.reinterpret_cast %arg1 to offset: [%11], sizes: [32, 32], strides: [512, 1] {tmd.reuse = {sequential = [{depth = 4 : i64, iterator = "%arg10", reuse_type = "no_reuse", volume = 0 : i64}], spatial = [{depth = 2 : i64, iterator = "%arg8", mapped_to = "y", reuse_type = "no_reuse", volume = 0 : i64}, {depth = 3 : i64, iterator = "%arg9", mapped_to = "x", reuse_type = "total_reuse", volume = 4096 : i64}], temporal = [{depth = 0 : i64, iterator = "%arg6", reuse_type = "total_reuse", volume = 4096 : i64}, {depth = 1 : i64, iterator = "%arg7", reuse_type = "no_reuse", volume = 0 : i64}]}} : memref<*xf32> to memref<32x32xf32, strided<[512, 1], offset: ?>>
                %alloc_4 = memref.alloc() {tmd.alloc = {local = true, memory_name = "L1", size = 4096 : i64}} : memref<32x32xf32>
                memref.copy %reinterpret_cast_3, %alloc_4 {tmd.copy.choice = {dim = "x", interconnect_name = "horizontal_links", kind = "broadcast"}} : memref<32x32xf32, strided<[512, 1], offset: ?>> to memref<32x32xf32>
                %alloc_5 = memref.alloc() {alignment = 64 : i64} : memref<32x32xf32>
                memref.copy %alloc, %alloc_5 : memref<32x32xf32> to memref<32x32xf32>
                linalg.matmul ins(%alloc_2, %alloc_4 : memref<32x32xf32>, memref<32x32xf32>) outs(%alloc_5 : memref<32x32xf32>)
                linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%arg13, %alloc_5 : memref<32x32xf32>, memref<32x32xf32>) outs(%arg13 : memref<32x32xf32>) {
                ^bb0(%in: f32, %in_6: f32, %out: f32):
                  %12 = arith.addf %in, %in_6 : f32
                  linalg.yield %12 : f32
                }
                scf.yield %arg13 : memref<32x32xf32>
              }
              scf.yield %9 : memref<32x32xf32>
            }
            %8 = affine.apply affine_map<(d0, d1, d2, d3) -> (d0 * 256 + d1 * 32 + d2 * 131072 + d3 * 16384)>(%arg7, %arg8, %arg6, %arg9)
            %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%8], sizes: [32, 32], strides: [512, 1] {tmd.reuse = {spatial = [{depth = 2 : i64, iterator = "%arg8", mapped_to = "y", reuse_type = "no_reuse", volume = 0 : i64}, {depth = 3 : i64, iterator = "%arg9", mapped_to = "x", reuse_type = "no_reuse", volume = 0 : i64}], temporal = [{depth = 0 : i64, iterator = "%arg6", reuse_type = "no_reuse", volume = 0 : i64}, {depth = 1 : i64, iterator = "%arg7", reuse_type = "no_reuse", volume = 0 : i64}]}} : memref<*xf32> to memref<32x32xf32, strided<[512, 1], offset: ?>>
            memref.copy %7, %reinterpret_cast : memref<32x32xf32> to memref<32x32xf32, strided<[512, 1], offset: ?>>
          } {tmd.mapped_to = "x"}
        } {tmd.mapped_to = "y"}
      }
    }
    return
  }
  func.func @matmul_kernel__d0i0_d1i1__c0by_c1bx(%arg0: memref<*xf32> {tt.divisibility = 16 : i32}, %arg1: memref<*xf32> {tt.divisibility = 16 : i32}, %arg2: memref<*xf32> {tt.divisibility = 16 : i32}, %arg3: index, %arg4: index, %arg5: index) {
    affine.for %arg6 = 0 to affine_map<(d0, d1) -> (d0 ceildiv 8)>(%arg3, %arg4) {
      affine.for %arg7 = 0 to affine_map<(d0, d1) -> (d1 ceildiv 8)>(%arg3, %arg4) {
        affine.parallel (%arg8) = (0) to (8) {
          affine.parallel (%arg9) = (0) to (8) {
            %cst = arith.constant 0.000000e+00 : f32
            %alloc = memref.alloc() {alignment = 64 : i64} : memref<32x32xf32>
            linalg.fill ins(%cst : f32) outs(%alloc : memref<32x32xf32>)
            %c0 = arith.constant 0 : index
            %c1 = arith.constant 1 : index
            %alloc_0 = memref.alloc() {alignment = 64 : i64} : memref<32x32xf32>
            memref.copy %alloc, %alloc_0 : memref<32x32xf32> to memref<32x32xf32>
            %c4 = arith.constant 4 : index
            %7 = scf.for %arg10 = %c0 to %c4 step %c1 iter_args(%arg11 = %alloc_0) -> (memref<32x32xf32>) {
              %9 = scf.for %arg12 = %c0 to %c4 step %c1 iter_args(%arg13 = %arg11) -> (memref<32x32xf32>) {
                %10 = affine.apply affine_map<(d0, d1, d2, d3) -> (d0 * 131072 + d1 * 16384 + d2 * 128 + d3 * 32)>(%arg6, %arg9, %arg10, %arg12)
                %reinterpret_cast_1 = memref.reinterpret_cast %arg0 to offset: [%10], sizes: [32, 32], strides: [512, 1] {tmd.reuse = {sequential = [{depth = 4 : i64, iterator = "%arg10", reuse_type = "no_reuse", volume = 0 : i64}], spatial = [{depth = 2 : i64, iterator = "%arg8", mapped_to = "y", reuse_type = "total_reuse", volume = 4096 : i64}, {depth = 3 : i64, iterator = "%arg9", mapped_to = "x", reuse_type = "no_reuse", volume = 0 : i64}], temporal = [{depth = 0 : i64, iterator = "%arg6", reuse_type = "no_reuse", volume = 0 : i64}, {depth = 1 : i64, iterator = "%arg7", reuse_type = "total_reuse", volume = 4096 : i64}]}} : memref<*xf32> to memref<32x32xf32, strided<[512, 1], offset: ?>>
                %alloc_2 = memref.alloc() {tmd.alloc = {local = true, memory_name = "L1", size = 4096 : i64}} : memref<32x32xf32>
                memref.copy %reinterpret_cast_1, %alloc_2 {tmd.copy.choice = {dim = "y", interconnect_name = "vertical_links", kind = "broadcast"}} : memref<32x32xf32, strided<[512, 1], offset: ?>> to memref<32x32xf32>
                %11 = affine.apply affine_map<(d0, d1, d2, d3) -> (d0 * 256 + d1 * 32 + d2 * 65536 + d3 * 16384)>(%arg7, %arg8, %arg10, %arg12)
                %reinterpret_cast_3 = memref.reinterpret_cast %arg1 to offset: [%11], sizes: [32, 32], strides: [512, 1] {tmd.reuse = {sequential = [{depth = 4 : i64, iterator = "%arg10", reuse_type = "no_reuse", volume = 0 : i64}], spatial = [{depth = 2 : i64, iterator = "%arg8", mapped_to = "y", reuse_type = "no_reuse", volume = 0 : i64}, {depth = 3 : i64, iterator = "%arg9", mapped_to = "x", reuse_type = "total_reuse", volume = 4096 : i64}], temporal = [{depth = 0 : i64, iterator = "%arg6", reuse_type = "total_reuse", volume = 4096 : i64}, {depth = 1 : i64, iterator = "%arg7", reuse_type = "no_reuse", volume = 0 : i64}]}} : memref<*xf32> to memref<32x32xf32, strided<[512, 1], offset: ?>>
                %alloc_4 = memref.alloc() {tmd.alloc = {local = true, memory_name = "L1", size = 4096 : i64}} : memref<32x32xf32>
                memref.copy %reinterpret_cast_3, %alloc_4 {tmd.copy.choice = {dim = "x", interconnect_name = "horizontal_links", kind = "broadcast"}} : memref<32x32xf32, strided<[512, 1], offset: ?>> to memref<32x32xf32>
                %alloc_5 = memref.alloc() {alignment = 64 : i64} : memref<32x32xf32>
                memref.copy %alloc, %alloc_5 : memref<32x32xf32> to memref<32x32xf32>
                linalg.matmul ins(%alloc_2, %alloc_4 : memref<32x32xf32>, memref<32x32xf32>) outs(%alloc_5 : memref<32x32xf32>)
                linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%arg13, %alloc_5 : memref<32x32xf32>, memref<32x32xf32>) outs(%arg13 : memref<32x32xf32>) {
                ^bb0(%in: f32, %in_6: f32, %out: f32):
                  %12 = arith.addf %in, %in_6 : f32
                  linalg.yield %12 : f32
                }
                scf.yield %arg13 : memref<32x32xf32>
              }
              scf.yield %9 : memref<32x32xf32>
            }
            %8 = affine.apply affine_map<(d0, d1, d2, d3) -> (d0 * 256 + d1 * 32 + d2 * 131072 + d3 * 16384)>(%arg7, %arg8, %arg6, %arg9)
            %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%8], sizes: [32, 32], strides: [512, 1] {tmd.reuse = {spatial = [{depth = 2 : i64, iterator = "%arg8", mapped_to = "y", reuse_type = "no_reuse", volume = 0 : i64}, {depth = 3 : i64, iterator = "%arg9", mapped_to = "x", reuse_type = "no_reuse", volume = 0 : i64}], temporal = [{depth = 0 : i64, iterator = "%arg6", reuse_type = "no_reuse", volume = 0 : i64}, {depth = 1 : i64, iterator = "%arg7", reuse_type = "no_reuse", volume = 0 : i64}]}} : memref<*xf32> to memref<32x32xf32, strided<[512, 1], offset: ?>>
            memref.copy %7, %reinterpret_cast : memref<32x32xf32> to memref<32x32xf32, strided<[512, 1], offset: ?>>
          } {tmd.mapped_to = "x"}
        } {tmd.mapped_to = "y"}
      }
    }
    return
  }
  func.func @matmul_kernel__d1i0_d0i1__c0mem_c1mem(%arg0: memref<*xf32> {tt.divisibility = 16 : i32}, %arg1: memref<*xf32> {tt.divisibility = 16 : i32}, %arg2: memref<*xf32> {tt.divisibility = 16 : i32}, %arg3: index, %arg4: index, %arg5: index) {
    affine.for %arg6 = 0 to affine_map<(d0, d1) -> (d0 ceildiv 8)>(%arg3, %arg4) {
      affine.for %arg7 = 0 to affine_map<(d0, d1) -> (d1 ceildiv 8)>(%arg3, %arg4) {
        affine.parallel (%arg8) = (0) to (8) {
          affine.parallel (%arg9) = (0) to (8) {
            %cst = arith.constant 0.000000e+00 : f32
            %alloc = memref.alloc() {alignment = 64 : i64} : memref<32x32xf32>
            linalg.fill ins(%cst : f32) outs(%alloc : memref<32x32xf32>)
            %c0 = arith.constant 0 : index
            %c1 = arith.constant 1 : index
            %alloc_0 = memref.alloc() {alignment = 64 : i64} : memref<32x32xf32>
            memref.copy %alloc, %alloc_0 : memref<32x32xf32> to memref<32x32xf32>
            %c4 = arith.constant 4 : index
            %7 = scf.for %arg10 = %c0 to %c4 step %c1 iter_args(%arg11 = %alloc_0) -> (memref<32x32xf32>) {
              %9 = scf.for %arg12 = %c0 to %c4 step %c1 iter_args(%arg13 = %arg11) -> (memref<32x32xf32>) {
                %10 = affine.apply affine_map<(d0, d1, d2, d3) -> (d0 * 131072 + d1 * 16384 + d2 * 128 + d3 * 32)>(%arg6, %arg9, %arg10, %arg12)
                %reinterpret_cast_1 = memref.reinterpret_cast %arg0 to offset: [%10], sizes: [32, 32], strides: [512, 1] {tmd.reuse = {sequential = [{depth = 4 : i64, iterator = "%arg10", reuse_type = "no_reuse", volume = 0 : i64}], spatial = [{depth = 2 : i64, iterator = "%arg8", mapped_to = "x", reuse_type = "total_reuse", volume = 4096 : i64}, {depth = 3 : i64, iterator = "%arg9", mapped_to = "y", reuse_type = "no_reuse", volume = 0 : i64}], temporal = [{depth = 0 : i64, iterator = "%arg6", reuse_type = "no_reuse", volume = 0 : i64}, {depth = 1 : i64, iterator = "%arg7", reuse_type = "total_reuse", volume = 4096 : i64}]}} : memref<*xf32> to memref<32x32xf32, strided<[512, 1], offset: ?>>
                %alloc_2 = memref.alloc() {tmd.alloc = {local = true, memory_name = "L1", size = 4096 : i64}} : memref<32x32xf32>
                memref.copy %reinterpret_cast_1, %alloc_2 {tmd.copy.choice = {kind = "mem", memory_name = "L1"}} : memref<32x32xf32, strided<[512, 1], offset: ?>> to memref<32x32xf32>
                %11 = affine.apply affine_map<(d0, d1, d2, d3) -> (d0 * 256 + d1 * 32 + d2 * 65536 + d3 * 16384)>(%arg7, %arg8, %arg10, %arg12)
                %reinterpret_cast_3 = memref.reinterpret_cast %arg1 to offset: [%11], sizes: [32, 32], strides: [512, 1] {tmd.reuse = {sequential = [{depth = 4 : i64, iterator = "%arg10", reuse_type = "no_reuse", volume = 0 : i64}], spatial = [{depth = 2 : i64, iterator = "%arg8", mapped_to = "x", reuse_type = "no_reuse", volume = 0 : i64}, {depth = 3 : i64, iterator = "%arg9", mapped_to = "y", reuse_type = "total_reuse", volume = 4096 : i64}], temporal = [{depth = 0 : i64, iterator = "%arg6", reuse_type = "total_reuse", volume = 4096 : i64}, {depth = 1 : i64, iterator = "%arg7", reuse_type = "no_reuse", volume = 0 : i64}]}} : memref<*xf32> to memref<32x32xf32, strided<[512, 1], offset: ?>>
                %alloc_4 = memref.alloc() {tmd.alloc = {local = true, memory_name = "L1", size = 4096 : i64}} : memref<32x32xf32>
                memref.copy %reinterpret_cast_3, %alloc_4 {tmd.copy.choice = {kind = "mem", memory_name = "L1"}} : memref<32x32xf32, strided<[512, 1], offset: ?>> to memref<32x32xf32>
                %alloc_5 = memref.alloc() {alignment = 64 : i64} : memref<32x32xf32>
                memref.copy %alloc, %alloc_5 : memref<32x32xf32> to memref<32x32xf32>
                linalg.matmul ins(%alloc_2, %alloc_4 : memref<32x32xf32>, memref<32x32xf32>) outs(%alloc_5 : memref<32x32xf32>)
                linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%arg13, %alloc_5 : memref<32x32xf32>, memref<32x32xf32>) outs(%arg13 : memref<32x32xf32>) {
                ^bb0(%in: f32, %in_6: f32, %out: f32):
                  %12 = arith.addf %in, %in_6 : f32
                  linalg.yield %12 : f32
                }
                scf.yield %arg13 : memref<32x32xf32>
              }
              scf.yield %9 : memref<32x32xf32>
            }
            %8 = affine.apply affine_map<(d0, d1, d2, d3) -> (d0 * 256 + d1 * 32 + d2 * 131072 + d3 * 16384)>(%arg7, %arg8, %arg6, %arg9)
            %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%8], sizes: [32, 32], strides: [512, 1] {tmd.reuse = {spatial = [{depth = 2 : i64, iterator = "%arg8", mapped_to = "x", reuse_type = "no_reuse", volume = 0 : i64}, {depth = 3 : i64, iterator = "%arg9", mapped_to = "y", reuse_type = "no_reuse", volume = 0 : i64}], temporal = [{depth = 0 : i64, iterator = "%arg6", reuse_type = "no_reuse", volume = 0 : i64}, {depth = 1 : i64, iterator = "%arg7", reuse_type = "no_reuse", volume = 0 : i64}]}} : memref<*xf32> to memref<32x32xf32, strided<[512, 1], offset: ?>>
            memref.copy %7, %reinterpret_cast : memref<32x32xf32> to memref<32x32xf32, strided<[512, 1], offset: ?>>
          } {tmd.mapped_to = "y"}
        } {tmd.mapped_to = "x"}
      }
    }
    return
  }
  func.func @matmul_kernel__d1i0_d0i1__c0bx_c1mem(%arg0: memref<*xf32> {tt.divisibility = 16 : i32}, %arg1: memref<*xf32> {tt.divisibility = 16 : i32}, %arg2: memref<*xf32> {tt.divisibility = 16 : i32}, %arg3: index, %arg4: index, %arg5: index) {
    affine.for %arg6 = 0 to affine_map<(d0, d1) -> (d0 ceildiv 8)>(%arg3, %arg4) {
      affine.for %arg7 = 0 to affine_map<(d0, d1) -> (d1 ceildiv 8)>(%arg3, %arg4) {
        affine.parallel (%arg8) = (0) to (8) {
          affine.parallel (%arg9) = (0) to (8) {
            %cst = arith.constant 0.000000e+00 : f32
            %alloc = memref.alloc() {alignment = 64 : i64} : memref<32x32xf32>
            linalg.fill ins(%cst : f32) outs(%alloc : memref<32x32xf32>)
            %c0 = arith.constant 0 : index
            %c1 = arith.constant 1 : index
            %alloc_0 = memref.alloc() {alignment = 64 : i64} : memref<32x32xf32>
            memref.copy %alloc, %alloc_0 : memref<32x32xf32> to memref<32x32xf32>
            %c4 = arith.constant 4 : index
            %7 = scf.for %arg10 = %c0 to %c4 step %c1 iter_args(%arg11 = %alloc_0) -> (memref<32x32xf32>) {
              %9 = scf.for %arg12 = %c0 to %c4 step %c1 iter_args(%arg13 = %arg11) -> (memref<32x32xf32>) {
                %10 = affine.apply affine_map<(d0, d1, d2, d3) -> (d0 * 131072 + d1 * 16384 + d2 * 128 + d3 * 32)>(%arg6, %arg9, %arg10, %arg12)
                %reinterpret_cast_1 = memref.reinterpret_cast %arg0 to offset: [%10], sizes: [32, 32], strides: [512, 1] {tmd.reuse = {sequential = [{depth = 4 : i64, iterator = "%arg10", reuse_type = "no_reuse", volume = 0 : i64}], spatial = [{depth = 2 : i64, iterator = "%arg8", mapped_to = "x", reuse_type = "total_reuse", volume = 4096 : i64}, {depth = 3 : i64, iterator = "%arg9", mapped_to = "y", reuse_type = "no_reuse", volume = 0 : i64}], temporal = [{depth = 0 : i64, iterator = "%arg6", reuse_type = "no_reuse", volume = 0 : i64}, {depth = 1 : i64, iterator = "%arg7", reuse_type = "total_reuse", volume = 4096 : i64}]}} : memref<*xf32> to memref<32x32xf32, strided<[512, 1], offset: ?>>
                %alloc_2 = memref.alloc() {tmd.alloc = {local = true, memory_name = "L1", size = 4096 : i64}} : memref<32x32xf32>
                memref.copy %reinterpret_cast_1, %alloc_2 {tmd.copy.choice = {dim = "x", interconnect_name = "horizontal_links", kind = "broadcast"}} : memref<32x32xf32, strided<[512, 1], offset: ?>> to memref<32x32xf32>
                %11 = affine.apply affine_map<(d0, d1, d2, d3) -> (d0 * 256 + d1 * 32 + d2 * 65536 + d3 * 16384)>(%arg7, %arg8, %arg10, %arg12)
                %reinterpret_cast_3 = memref.reinterpret_cast %arg1 to offset: [%11], sizes: [32, 32], strides: [512, 1] {tmd.reuse = {sequential = [{depth = 4 : i64, iterator = "%arg10", reuse_type = "no_reuse", volume = 0 : i64}], spatial = [{depth = 2 : i64, iterator = "%arg8", mapped_to = "x", reuse_type = "no_reuse", volume = 0 : i64}, {depth = 3 : i64, iterator = "%arg9", mapped_to = "y", reuse_type = "total_reuse", volume = 4096 : i64}], temporal = [{depth = 0 : i64, iterator = "%arg6", reuse_type = "total_reuse", volume = 4096 : i64}, {depth = 1 : i64, iterator = "%arg7", reuse_type = "no_reuse", volume = 0 : i64}]}} : memref<*xf32> to memref<32x32xf32, strided<[512, 1], offset: ?>>
                %alloc_4 = memref.alloc() {tmd.alloc = {local = true, memory_name = "L1", size = 4096 : i64}} : memref<32x32xf32>
                memref.copy %reinterpret_cast_3, %alloc_4 {tmd.copy.choice = {kind = "mem", memory_name = "L1"}} : memref<32x32xf32, strided<[512, 1], offset: ?>> to memref<32x32xf32>
                %alloc_5 = memref.alloc() {alignment = 64 : i64} : memref<32x32xf32>
                memref.copy %alloc, %alloc_5 : memref<32x32xf32> to memref<32x32xf32>
                linalg.matmul ins(%alloc_2, %alloc_4 : memref<32x32xf32>, memref<32x32xf32>) outs(%alloc_5 : memref<32x32xf32>)
                linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%arg13, %alloc_5 : memref<32x32xf32>, memref<32x32xf32>) outs(%arg13 : memref<32x32xf32>) {
                ^bb0(%in: f32, %in_6: f32, %out: f32):
                  %12 = arith.addf %in, %in_6 : f32
                  linalg.yield %12 : f32
                }
                scf.yield %arg13 : memref<32x32xf32>
              }
              scf.yield %9 : memref<32x32xf32>
            }
            %8 = affine.apply affine_map<(d0, d1, d2, d3) -> (d0 * 256 + d1 * 32 + d2 * 131072 + d3 * 16384)>(%arg7, %arg8, %arg6, %arg9)
            %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%8], sizes: [32, 32], strides: [512, 1] {tmd.reuse = {spatial = [{depth = 2 : i64, iterator = "%arg8", mapped_to = "x", reuse_type = "no_reuse", volume = 0 : i64}, {depth = 3 : i64, iterator = "%arg9", mapped_to = "y", reuse_type = "no_reuse", volume = 0 : i64}], temporal = [{depth = 0 : i64, iterator = "%arg6", reuse_type = "no_reuse", volume = 0 : i64}, {depth = 1 : i64, iterator = "%arg7", reuse_type = "no_reuse", volume = 0 : i64}]}} : memref<*xf32> to memref<32x32xf32, strided<[512, 1], offset: ?>>
            memref.copy %7, %reinterpret_cast : memref<32x32xf32> to memref<32x32xf32, strided<[512, 1], offset: ?>>
          } {tmd.mapped_to = "y"}
        } {tmd.mapped_to = "x"}
      }
    }
    return
  }
  func.func @matmul_kernel__d1i0_d0i1__c0mem_c1by(%arg0: memref<*xf32> {tt.divisibility = 16 : i32}, %arg1: memref<*xf32> {tt.divisibility = 16 : i32}, %arg2: memref<*xf32> {tt.divisibility = 16 : i32}, %arg3: index, %arg4: index, %arg5: index) {
    affine.for %arg6 = 0 to affine_map<(d0, d1) -> (d0 ceildiv 8)>(%arg3, %arg4) {
      affine.for %arg7 = 0 to affine_map<(d0, d1) -> (d1 ceildiv 8)>(%arg3, %arg4) {
        affine.parallel (%arg8) = (0) to (8) {
          affine.parallel (%arg9) = (0) to (8) {
            %cst = arith.constant 0.000000e+00 : f32
            %alloc = memref.alloc() {alignment = 64 : i64} : memref<32x32xf32>
            linalg.fill ins(%cst : f32) outs(%alloc : memref<32x32xf32>)
            %c0 = arith.constant 0 : index
            %c1 = arith.constant 1 : index
            %alloc_0 = memref.alloc() {alignment = 64 : i64} : memref<32x32xf32>
            memref.copy %alloc, %alloc_0 : memref<32x32xf32> to memref<32x32xf32>
            %c4 = arith.constant 4 : index
            %7 = scf.for %arg10 = %c0 to %c4 step %c1 iter_args(%arg11 = %alloc_0) -> (memref<32x32xf32>) {
              %9 = scf.for %arg12 = %c0 to %c4 step %c1 iter_args(%arg13 = %arg11) -> (memref<32x32xf32>) {
                %10 = affine.apply affine_map<(d0, d1, d2, d3) -> (d0 * 131072 + d1 * 16384 + d2 * 128 + d3 * 32)>(%arg6, %arg9, %arg10, %arg12)
                %reinterpret_cast_1 = memref.reinterpret_cast %arg0 to offset: [%10], sizes: [32, 32], strides: [512, 1] {tmd.reuse = {sequential = [{depth = 4 : i64, iterator = "%arg10", reuse_type = "no_reuse", volume = 0 : i64}], spatial = [{depth = 2 : i64, iterator = "%arg8", mapped_to = "x", reuse_type = "total_reuse", volume = 4096 : i64}, {depth = 3 : i64, iterator = "%arg9", mapped_to = "y", reuse_type = "no_reuse", volume = 0 : i64}], temporal = [{depth = 0 : i64, iterator = "%arg6", reuse_type = "no_reuse", volume = 0 : i64}, {depth = 1 : i64, iterator = "%arg7", reuse_type = "total_reuse", volume = 4096 : i64}]}} : memref<*xf32> to memref<32x32xf32, strided<[512, 1], offset: ?>>
                %alloc_2 = memref.alloc() {tmd.alloc = {local = true, memory_name = "L1", size = 4096 : i64}} : memref<32x32xf32>
                memref.copy %reinterpret_cast_1, %alloc_2 {tmd.copy.choice = {kind = "mem", memory_name = "L1"}} : memref<32x32xf32, strided<[512, 1], offset: ?>> to memref<32x32xf32>
                %11 = affine.apply affine_map<(d0, d1, d2, d3) -> (d0 * 256 + d1 * 32 + d2 * 65536 + d3 * 16384)>(%arg7, %arg8, %arg10, %arg12)
                %reinterpret_cast_3 = memref.reinterpret_cast %arg1 to offset: [%11], sizes: [32, 32], strides: [512, 1] {tmd.reuse = {sequential = [{depth = 4 : i64, iterator = "%arg10", reuse_type = "no_reuse", volume = 0 : i64}], spatial = [{depth = 2 : i64, iterator = "%arg8", mapped_to = "x", reuse_type = "no_reuse", volume = 0 : i64}, {depth = 3 : i64, iterator = "%arg9", mapped_to = "y", reuse_type = "total_reuse", volume = 4096 : i64}], temporal = [{depth = 0 : i64, iterator = "%arg6", reuse_type = "total_reuse", volume = 4096 : i64}, {depth = 1 : i64, iterator = "%arg7", reuse_type = "no_reuse", volume = 0 : i64}]}} : memref<*xf32> to memref<32x32xf32, strided<[512, 1], offset: ?>>
                %alloc_4 = memref.alloc() {tmd.alloc = {local = true, memory_name = "L1", size = 4096 : i64}} : memref<32x32xf32>
                memref.copy %reinterpret_cast_3, %alloc_4 {tmd.copy.choice = {dim = "y", interconnect_name = "vertical_links", kind = "broadcast"}} : memref<32x32xf32, strided<[512, 1], offset: ?>> to memref<32x32xf32>
                %alloc_5 = memref.alloc() {alignment = 64 : i64} : memref<32x32xf32>
                memref.copy %alloc, %alloc_5 : memref<32x32xf32> to memref<32x32xf32>
                linalg.matmul ins(%alloc_2, %alloc_4 : memref<32x32xf32>, memref<32x32xf32>) outs(%alloc_5 : memref<32x32xf32>)
                linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%arg13, %alloc_5 : memref<32x32xf32>, memref<32x32xf32>) outs(%arg13 : memref<32x32xf32>) {
                ^bb0(%in: f32, %in_6: f32, %out: f32):
                  %12 = arith.addf %in, %in_6 : f32
                  linalg.yield %12 : f32
                }
                scf.yield %arg13 : memref<32x32xf32>
              }
              scf.yield %9 : memref<32x32xf32>
            }
            %8 = affine.apply affine_map<(d0, d1, d2, d3) -> (d0 * 256 + d1 * 32 + d2 * 131072 + d3 * 16384)>(%arg7, %arg8, %arg6, %arg9)
            %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%8], sizes: [32, 32], strides: [512, 1] {tmd.reuse = {spatial = [{depth = 2 : i64, iterator = "%arg8", mapped_to = "x", reuse_type = "no_reuse", volume = 0 : i64}, {depth = 3 : i64, iterator = "%arg9", mapped_to = "y", reuse_type = "no_reuse", volume = 0 : i64}], temporal = [{depth = 0 : i64, iterator = "%arg6", reuse_type = "no_reuse", volume = 0 : i64}, {depth = 1 : i64, iterator = "%arg7", reuse_type = "no_reuse", volume = 0 : i64}]}} : memref<*xf32> to memref<32x32xf32, strided<[512, 1], offset: ?>>
            memref.copy %7, %reinterpret_cast : memref<32x32xf32> to memref<32x32xf32, strided<[512, 1], offset: ?>>
          } {tmd.mapped_to = "y"}
        } {tmd.mapped_to = "x"}
      }
    }
    return
  }
  func.func @matmul_kernel__d1i0_d0i1__c0bx_c1by(%arg0: memref<*xf32> {tt.divisibility = 16 : i32}, %arg1: memref<*xf32> {tt.divisibility = 16 : i32}, %arg2: memref<*xf32> {tt.divisibility = 16 : i32}, %arg3: index, %arg4: index, %arg5: index) {
    affine.for %arg6 = 0 to affine_map<(d0, d1) -> (d0 ceildiv 8)>(%arg3, %arg4) {
      affine.for %arg7 = 0 to affine_map<(d0, d1) -> (d1 ceildiv 8)>(%arg3, %arg4) {
        affine.parallel (%arg8) = (0) to (8) {
          affine.parallel (%arg9) = (0) to (8) {
            %cst = arith.constant 0.000000e+00 : f32
            %alloc = memref.alloc() {alignment = 64 : i64} : memref<32x32xf32>
            linalg.fill ins(%cst : f32) outs(%alloc : memref<32x32xf32>)
            %c0 = arith.constant 0 : index
            %c1 = arith.constant 1 : index
            %alloc_0 = memref.alloc() {alignment = 64 : i64} : memref<32x32xf32>
            memref.copy %alloc, %alloc_0 : memref<32x32xf32> to memref<32x32xf32>
            %c4 = arith.constant 4 : index
            %7 = scf.for %arg10 = %c0 to %c4 step %c1 iter_args(%arg11 = %alloc_0) -> (memref<32x32xf32>) {
              %9 = scf.for %arg12 = %c0 to %c4 step %c1 iter_args(%arg13 = %arg11) -> (memref<32x32xf32>) {
                %10 = affine.apply affine_map<(d0, d1, d2, d3) -> (d0 * 131072 + d1 * 16384 + d2 * 128 + d3 * 32)>(%arg6, %arg9, %arg10, %arg12)
                %reinterpret_cast_1 = memref.reinterpret_cast %arg0 to offset: [%10], sizes: [32, 32], strides: [512, 1] {tmd.reuse = {sequential = [{depth = 4 : i64, iterator = "%arg10", reuse_type = "no_reuse", volume = 0 : i64}], spatial = [{depth = 2 : i64, iterator = "%arg8", mapped_to = "x", reuse_type = "total_reuse", volume = 4096 : i64}, {depth = 3 : i64, iterator = "%arg9", mapped_to = "y", reuse_type = "no_reuse", volume = 0 : i64}], temporal = [{depth = 0 : i64, iterator = "%arg6", reuse_type = "no_reuse", volume = 0 : i64}, {depth = 1 : i64, iterator = "%arg7", reuse_type = "total_reuse", volume = 4096 : i64}]}} : memref<*xf32> to memref<32x32xf32, strided<[512, 1], offset: ?>>
                %alloc_2 = memref.alloc() {tmd.alloc = {local = true, memory_name = "L1", size = 4096 : i64}} : memref<32x32xf32>
                memref.copy %reinterpret_cast_1, %alloc_2 {tmd.copy.choice = {dim = "x", interconnect_name = "horizontal_links", kind = "broadcast"}} : memref<32x32xf32, strided<[512, 1], offset: ?>> to memref<32x32xf32>
                %11 = affine.apply affine_map<(d0, d1, d2, d3) -> (d0 * 256 + d1 * 32 + d2 * 65536 + d3 * 16384)>(%arg7, %arg8, %arg10, %arg12)
                %reinterpret_cast_3 = memref.reinterpret_cast %arg1 to offset: [%11], sizes: [32, 32], strides: [512, 1] {tmd.reuse = {sequential = [{depth = 4 : i64, iterator = "%arg10", reuse_type = "no_reuse", volume = 0 : i64}], spatial = [{depth = 2 : i64, iterator = "%arg8", mapped_to = "x", reuse_type = "no_reuse", volume = 0 : i64}, {depth = 3 : i64, iterator = "%arg9", mapped_to = "y", reuse_type = "total_reuse", volume = 4096 : i64}], temporal = [{depth = 0 : i64, iterator = "%arg6", reuse_type = "total_reuse", volume = 4096 : i64}, {depth = 1 : i64, iterator = "%arg7", reuse_type = "no_reuse", volume = 0 : i64}]}} : memref<*xf32> to memref<32x32xf32, strided<[512, 1], offset: ?>>
                %alloc_4 = memref.alloc() {tmd.alloc = {local = true, memory_name = "L1", size = 4096 : i64}} : memref<32x32xf32>
                memref.copy %reinterpret_cast_3, %alloc_4 {tmd.copy.choice = {dim = "y", interconnect_name = "vertical_links", kind = "broadcast"}} : memref<32x32xf32, strided<[512, 1], offset: ?>> to memref<32x32xf32>
                %alloc_5 = memref.alloc() {alignment = 64 : i64} : memref<32x32xf32>
                memref.copy %alloc, %alloc_5 : memref<32x32xf32> to memref<32x32xf32>
                linalg.matmul ins(%alloc_2, %alloc_4 : memref<32x32xf32>, memref<32x32xf32>) outs(%alloc_5 : memref<32x32xf32>)
                linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%arg13, %alloc_5 : memref<32x32xf32>, memref<32x32xf32>) outs(%arg13 : memref<32x32xf32>) {
                ^bb0(%in: f32, %in_6: f32, %out: f32):
                  %12 = arith.addf %in, %in_6 : f32
                  linalg.yield %12 : f32
                }
                scf.yield %arg13 : memref<32x32xf32>
              }
              scf.yield %9 : memref<32x32xf32>
            }
            %8 = affine.apply affine_map<(d0, d1, d2, d3) -> (d0 * 256 + d1 * 32 + d2 * 131072 + d3 * 16384)>(%arg7, %arg8, %arg6, %arg9)
            %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%8], sizes: [32, 32], strides: [512, 1] {tmd.reuse = {spatial = [{depth = 2 : i64, iterator = "%arg8", mapped_to = "x", reuse_type = "no_reuse", volume = 0 : i64}, {depth = 3 : i64, iterator = "%arg9", mapped_to = "y", reuse_type = "no_reuse", volume = 0 : i64}], temporal = [{depth = 0 : i64, iterator = "%arg6", reuse_type = "no_reuse", volume = 0 : i64}, {depth = 1 : i64, iterator = "%arg7", reuse_type = "no_reuse", volume = 0 : i64}]}} : memref<*xf32> to memref<32x32xf32, strided<[512, 1], offset: ?>>
            memref.copy %7, %reinterpret_cast : memref<32x32xf32> to memref<32x32xf32, strided<[512, 1], offset: ?>>
          } {tmd.mapped_to = "y"}
        } {tmd.mapped_to = "x"}
      }
    }
    return
  }
  func.func @matmul_kernel__d0i1_d1i1__c0mem_c1mem(%arg0: memref<*xf32> {tt.divisibility = 16 : i32}, %arg1: memref<*xf32> {tt.divisibility = 16 : i32}, %arg2: memref<*xf32> {tt.divisibility = 16 : i32}, %arg3: index, %arg4: index, %arg5: index) {
    affine.for %arg6 = 0 to affine_map<(d0, d1) -> (d0)>(%arg3, %arg4) {
      affine.for %arg7 = 0 to affine_map<(d0, d1) -> ((d1 ceildiv 8) ceildiv 8)>(%arg3, %arg4) {
        affine.parallel (%arg8) = (0) to (8) {
          affine.parallel (%arg9) = (0) to (8) {
            %cst = arith.constant 0.000000e+00 : f32
            %alloc = memref.alloc() {alignment = 64 : i64} : memref<32x32xf32>
            linalg.fill ins(%cst : f32) outs(%alloc : memref<32x32xf32>)
            %c0 = arith.constant 0 : index
            %c1 = arith.constant 1 : index
            %alloc_0 = memref.alloc() {alignment = 64 : i64} : memref<32x32xf32>
            memref.copy %alloc, %alloc_0 : memref<32x32xf32> to memref<32x32xf32>
            %c4 = arith.constant 4 : index
            %7 = scf.for %arg10 = %c0 to %c4 step %c1 iter_args(%arg11 = %alloc_0) -> (memref<32x32xf32>) {
              %9 = scf.for %arg12 = %c0 to %c4 step %c1 iter_args(%arg13 = %arg11) -> (memref<32x32xf32>) {
                %10 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 16384 + d1 * 128 + d2 * 32)>(%arg6, %arg10, %arg12)
                %reinterpret_cast_1 = memref.reinterpret_cast %arg0 to offset: [%10], sizes: [32, 32], strides: [512, 1] {tmd.reuse = {sequential = [{depth = 4 : i64, iterator = "%arg10", reuse_type = "no_reuse", volume = 0 : i64}], spatial = [{depth = 2 : i64, iterator = "%arg8", mapped_to = "y", reuse_type = "total_reuse", volume = 4096 : i64}, {depth = 3 : i64, iterator = "%arg9", mapped_to = "x", reuse_type = "total_reuse", volume = 4096 : i64}], temporal = [{depth = 0 : i64, iterator = "%arg6", reuse_type = "no_reuse", volume = 0 : i64}, {depth = 1 : i64, iterator = "%arg7", reuse_type = "total_reuse", volume = 4096 : i64}]}} : memref<*xf32> to memref<32x32xf32, strided<[512, 1], offset: ?>>
                %alloc_2 = memref.alloc() {tmd.alloc = {local = true, memory_name = "L1", size = 4096 : i64}} : memref<32x32xf32>
                memref.copy %reinterpret_cast_1, %alloc_2 {tmd.copy.choice = {kind = "mem", memory_name = "L1"}} : memref<32x32xf32, strided<[512, 1], offset: ?>> to memref<32x32xf32>
                %11 = affine.apply affine_map<(d0, d1, d2, d3, d4) -> (d0 * 32 + d1 * 2048 + d2 * 256 + d3 * 65536 + d4 * 16384)>(%arg9, %arg7, %arg8, %arg10, %arg12)
                %reinterpret_cast_3 = memref.reinterpret_cast %arg1 to offset: [%11], sizes: [32, 32], strides: [512, 1] {tmd.reuse = {sequential = [{depth = 4 : i64, iterator = "%arg10", reuse_type = "no_reuse", volume = 0 : i64}], spatial = [{depth = 2 : i64, iterator = "%arg8", mapped_to = "y", reuse_type = "no_reuse", volume = 0 : i64}, {depth = 3 : i64, iterator = "%arg9", mapped_to = "x", reuse_type = "no_reuse", volume = 0 : i64}], temporal = [{depth = 0 : i64, iterator = "%arg6", reuse_type = "total_reuse", volume = 4096 : i64}, {depth = 1 : i64, iterator = "%arg7", reuse_type = "no_reuse", volume = 0 : i64}]}} : memref<*xf32> to memref<32x32xf32, strided<[512, 1], offset: ?>>
                %alloc_4 = memref.alloc() {tmd.alloc = {local = true, memory_name = "L1", size = 4096 : i64}} : memref<32x32xf32>
                memref.copy %reinterpret_cast_3, %alloc_4 {tmd.copy.choice = {kind = "mem", memory_name = "L1"}} : memref<32x32xf32, strided<[512, 1], offset: ?>> to memref<32x32xf32>
                %alloc_5 = memref.alloc() {alignment = 64 : i64} : memref<32x32xf32>
                memref.copy %alloc, %alloc_5 : memref<32x32xf32> to memref<32x32xf32>
                linalg.matmul ins(%alloc_2, %alloc_4 : memref<32x32xf32>, memref<32x32xf32>) outs(%alloc_5 : memref<32x32xf32>)
                linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%arg13, %alloc_5 : memref<32x32xf32>, memref<32x32xf32>) outs(%arg13 : memref<32x32xf32>) {
                ^bb0(%in: f32, %in_6: f32, %out: f32):
                  %12 = arith.addf %in, %in_6 : f32
                  linalg.yield %12 : f32
                }
                scf.yield %arg13 : memref<32x32xf32>
              }
              scf.yield %9 : memref<32x32xf32>
            }
            %8 = affine.apply affine_map<(d0, d1, d2, d3) -> (d0 * 16384 + d1 * 32 + d2 * 2048 + d3 * 256)>(%arg6, %arg9, %arg7, %arg8)
            %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%8], sizes: [32, 32], strides: [512, 1] {tmd.reuse = {spatial = [{depth = 2 : i64, iterator = "%arg8", mapped_to = "y", reuse_type = "no_reuse", volume = 0 : i64}, {depth = 3 : i64, iterator = "%arg9", mapped_to = "x", reuse_type = "no_reuse", volume = 0 : i64}], temporal = [{depth = 0 : i64, iterator = "%arg6", reuse_type = "no_reuse", volume = 0 : i64}, {depth = 1 : i64, iterator = "%arg7", reuse_type = "no_reuse", volume = 0 : i64}]}} : memref<*xf32> to memref<32x32xf32, strided<[512, 1], offset: ?>>
            memref.copy %7, %reinterpret_cast : memref<32x32xf32> to memref<32x32xf32, strided<[512, 1], offset: ?>>
          } {tmd.mapped_to = "x"}
        } {tmd.mapped_to = "y"}
      }
    }
    return
  }
  func.func @matmul_kernel__d0i1_d1i1__c0bx_c1mem(%arg0: memref<*xf32> {tt.divisibility = 16 : i32}, %arg1: memref<*xf32> {tt.divisibility = 16 : i32}, %arg2: memref<*xf32> {tt.divisibility = 16 : i32}, %arg3: index, %arg4: index, %arg5: index) {
    affine.for %arg6 = 0 to affine_map<(d0, d1) -> (d0)>(%arg3, %arg4) {
      affine.for %arg7 = 0 to affine_map<(d0, d1) -> ((d1 ceildiv 8) ceildiv 8)>(%arg3, %arg4) {
        affine.parallel (%arg8) = (0) to (8) {
          affine.parallel (%arg9) = (0) to (8) {
            %cst = arith.constant 0.000000e+00 : f32
            %alloc = memref.alloc() {alignment = 64 : i64} : memref<32x32xf32>
            linalg.fill ins(%cst : f32) outs(%alloc : memref<32x32xf32>)
            %c0 = arith.constant 0 : index
            %c1 = arith.constant 1 : index
            %alloc_0 = memref.alloc() {alignment = 64 : i64} : memref<32x32xf32>
            memref.copy %alloc, %alloc_0 : memref<32x32xf32> to memref<32x32xf32>
            %c4 = arith.constant 4 : index
            %7 = scf.for %arg10 = %c0 to %c4 step %c1 iter_args(%arg11 = %alloc_0) -> (memref<32x32xf32>) {
              %9 = scf.for %arg12 = %c0 to %c4 step %c1 iter_args(%arg13 = %arg11) -> (memref<32x32xf32>) {
                %10 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 16384 + d1 * 128 + d2 * 32)>(%arg6, %arg10, %arg12)
                %reinterpret_cast_1 = memref.reinterpret_cast %arg0 to offset: [%10], sizes: [32, 32], strides: [512, 1] {tmd.reuse = {sequential = [{depth = 4 : i64, iterator = "%arg10", reuse_type = "no_reuse", volume = 0 : i64}], spatial = [{depth = 2 : i64, iterator = "%arg8", mapped_to = "y", reuse_type = "total_reuse", volume = 4096 : i64}, {depth = 3 : i64, iterator = "%arg9", mapped_to = "x", reuse_type = "total_reuse", volume = 4096 : i64}], temporal = [{depth = 0 : i64, iterator = "%arg6", reuse_type = "no_reuse", volume = 0 : i64}, {depth = 1 : i64, iterator = "%arg7", reuse_type = "total_reuse", volume = 4096 : i64}]}} : memref<*xf32> to memref<32x32xf32, strided<[512, 1], offset: ?>>
                %alloc_2 = memref.alloc() {tmd.alloc = {local = true, memory_name = "L1", size = 4096 : i64}} : memref<32x32xf32>
                memref.copy %reinterpret_cast_1, %alloc_2 {tmd.copy.choice = {dim = "x", interconnect_name = "horizontal_links", kind = "broadcast"}} : memref<32x32xf32, strided<[512, 1], offset: ?>> to memref<32x32xf32>
                %11 = affine.apply affine_map<(d0, d1, d2, d3, d4) -> (d0 * 32 + d1 * 2048 + d2 * 256 + d3 * 65536 + d4 * 16384)>(%arg9, %arg7, %arg8, %arg10, %arg12)
                %reinterpret_cast_3 = memref.reinterpret_cast %arg1 to offset: [%11], sizes: [32, 32], strides: [512, 1] {tmd.reuse = {sequential = [{depth = 4 : i64, iterator = "%arg10", reuse_type = "no_reuse", volume = 0 : i64}], spatial = [{depth = 2 : i64, iterator = "%arg8", mapped_to = "y", reuse_type = "no_reuse", volume = 0 : i64}, {depth = 3 : i64, iterator = "%arg9", mapped_to = "x", reuse_type = "no_reuse", volume = 0 : i64}], temporal = [{depth = 0 : i64, iterator = "%arg6", reuse_type = "total_reuse", volume = 4096 : i64}, {depth = 1 : i64, iterator = "%arg7", reuse_type = "no_reuse", volume = 0 : i64}]}} : memref<*xf32> to memref<32x32xf32, strided<[512, 1], offset: ?>>
                %alloc_4 = memref.alloc() {tmd.alloc = {local = true, memory_name = "L1", size = 4096 : i64}} : memref<32x32xf32>
                memref.copy %reinterpret_cast_3, %alloc_4 {tmd.copy.choice = {kind = "mem", memory_name = "L1"}} : memref<32x32xf32, strided<[512, 1], offset: ?>> to memref<32x32xf32>
                %alloc_5 = memref.alloc() {alignment = 64 : i64} : memref<32x32xf32>
                memref.copy %alloc, %alloc_5 : memref<32x32xf32> to memref<32x32xf32>
                linalg.matmul ins(%alloc_2, %alloc_4 : memref<32x32xf32>, memref<32x32xf32>) outs(%alloc_5 : memref<32x32xf32>)
                linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%arg13, %alloc_5 : memref<32x32xf32>, memref<32x32xf32>) outs(%arg13 : memref<32x32xf32>) {
                ^bb0(%in: f32, %in_6: f32, %out: f32):
                  %12 = arith.addf %in, %in_6 : f32
                  linalg.yield %12 : f32
                }
                scf.yield %arg13 : memref<32x32xf32>
              }
              scf.yield %9 : memref<32x32xf32>
            }
            %8 = affine.apply affine_map<(d0, d1, d2, d3) -> (d0 * 16384 + d1 * 32 + d2 * 2048 + d3 * 256)>(%arg6, %arg9, %arg7, %arg8)
            %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%8], sizes: [32, 32], strides: [512, 1] {tmd.reuse = {spatial = [{depth = 2 : i64, iterator = "%arg8", mapped_to = "y", reuse_type = "no_reuse", volume = 0 : i64}, {depth = 3 : i64, iterator = "%arg9", mapped_to = "x", reuse_type = "no_reuse", volume = 0 : i64}], temporal = [{depth = 0 : i64, iterator = "%arg6", reuse_type = "no_reuse", volume = 0 : i64}, {depth = 1 : i64, iterator = "%arg7", reuse_type = "no_reuse", volume = 0 : i64}]}} : memref<*xf32> to memref<32x32xf32, strided<[512, 1], offset: ?>>
            memref.copy %7, %reinterpret_cast : memref<32x32xf32> to memref<32x32xf32, strided<[512, 1], offset: ?>>
          } {tmd.mapped_to = "x"}
        } {tmd.mapped_to = "y"}
      }
    }
    return
  }
  func.func @matmul_kernel__d0i1_d1i1__c0by_c1mem(%arg0: memref<*xf32> {tt.divisibility = 16 : i32}, %arg1: memref<*xf32> {tt.divisibility = 16 : i32}, %arg2: memref<*xf32> {tt.divisibility = 16 : i32}, %arg3: index, %arg4: index, %arg5: index) {
    affine.for %arg6 = 0 to affine_map<(d0, d1) -> (d0)>(%arg3, %arg4) {
      affine.for %arg7 = 0 to affine_map<(d0, d1) -> ((d1 ceildiv 8) ceildiv 8)>(%arg3, %arg4) {
        affine.parallel (%arg8) = (0) to (8) {
          affine.parallel (%arg9) = (0) to (8) {
            %cst = arith.constant 0.000000e+00 : f32
            %alloc = memref.alloc() {alignment = 64 : i64} : memref<32x32xf32>
            linalg.fill ins(%cst : f32) outs(%alloc : memref<32x32xf32>)
            %c0 = arith.constant 0 : index
            %c1 = arith.constant 1 : index
            %alloc_0 = memref.alloc() {alignment = 64 : i64} : memref<32x32xf32>
            memref.copy %alloc, %alloc_0 : memref<32x32xf32> to memref<32x32xf32>
            %c4 = arith.constant 4 : index
            %7 = scf.for %arg10 = %c0 to %c4 step %c1 iter_args(%arg11 = %alloc_0) -> (memref<32x32xf32>) {
              %9 = scf.for %arg12 = %c0 to %c4 step %c1 iter_args(%arg13 = %arg11) -> (memref<32x32xf32>) {
                %10 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 16384 + d1 * 128 + d2 * 32)>(%arg6, %arg10, %arg12)
                %reinterpret_cast_1 = memref.reinterpret_cast %arg0 to offset: [%10], sizes: [32, 32], strides: [512, 1] {tmd.reuse = {sequential = [{depth = 4 : i64, iterator = "%arg10", reuse_type = "no_reuse", volume = 0 : i64}], spatial = [{depth = 2 : i64, iterator = "%arg8", mapped_to = "y", reuse_type = "total_reuse", volume = 4096 : i64}, {depth = 3 : i64, iterator = "%arg9", mapped_to = "x", reuse_type = "total_reuse", volume = 4096 : i64}], temporal = [{depth = 0 : i64, iterator = "%arg6", reuse_type = "no_reuse", volume = 0 : i64}, {depth = 1 : i64, iterator = "%arg7", reuse_type = "total_reuse", volume = 4096 : i64}]}} : memref<*xf32> to memref<32x32xf32, strided<[512, 1], offset: ?>>
                %alloc_2 = memref.alloc() {tmd.alloc = {local = true, memory_name = "L1", size = 4096 : i64}} : memref<32x32xf32>
                memref.copy %reinterpret_cast_1, %alloc_2 {tmd.copy.choice = {dim = "y", interconnect_name = "vertical_links", kind = "broadcast"}} : memref<32x32xf32, strided<[512, 1], offset: ?>> to memref<32x32xf32>
                %11 = affine.apply affine_map<(d0, d1, d2, d3, d4) -> (d0 * 32 + d1 * 2048 + d2 * 256 + d3 * 65536 + d4 * 16384)>(%arg9, %arg7, %arg8, %arg10, %arg12)
                %reinterpret_cast_3 = memref.reinterpret_cast %arg1 to offset: [%11], sizes: [32, 32], strides: [512, 1] {tmd.reuse = {sequential = [{depth = 4 : i64, iterator = "%arg10", reuse_type = "no_reuse", volume = 0 : i64}], spatial = [{depth = 2 : i64, iterator = "%arg8", mapped_to = "y", reuse_type = "no_reuse", volume = 0 : i64}, {depth = 3 : i64, iterator = "%arg9", mapped_to = "x", reuse_type = "no_reuse", volume = 0 : i64}], temporal = [{depth = 0 : i64, iterator = "%arg6", reuse_type = "total_reuse", volume = 4096 : i64}, {depth = 1 : i64, iterator = "%arg7", reuse_type = "no_reuse", volume = 0 : i64}]}} : memref<*xf32> to memref<32x32xf32, strided<[512, 1], offset: ?>>
                %alloc_4 = memref.alloc() {tmd.alloc = {local = true, memory_name = "L1", size = 4096 : i64}} : memref<32x32xf32>
                memref.copy %reinterpret_cast_3, %alloc_4 {tmd.copy.choice = {kind = "mem", memory_name = "L1"}} : memref<32x32xf32, strided<[512, 1], offset: ?>> to memref<32x32xf32>
                %alloc_5 = memref.alloc() {alignment = 64 : i64} : memref<32x32xf32>
                memref.copy %alloc, %alloc_5 : memref<32x32xf32> to memref<32x32xf32>
                linalg.matmul ins(%alloc_2, %alloc_4 : memref<32x32xf32>, memref<32x32xf32>) outs(%alloc_5 : memref<32x32xf32>)
                linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%arg13, %alloc_5 : memref<32x32xf32>, memref<32x32xf32>) outs(%arg13 : memref<32x32xf32>) {
                ^bb0(%in: f32, %in_6: f32, %out: f32):
                  %12 = arith.addf %in, %in_6 : f32
                  linalg.yield %12 : f32
                }
                scf.yield %arg13 : memref<32x32xf32>
              }
              scf.yield %9 : memref<32x32xf32>
            }
            %8 = affine.apply affine_map<(d0, d1, d2, d3) -> (d0 * 16384 + d1 * 32 + d2 * 2048 + d3 * 256)>(%arg6, %arg9, %arg7, %arg8)
            %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%8], sizes: [32, 32], strides: [512, 1] {tmd.reuse = {spatial = [{depth = 2 : i64, iterator = "%arg8", mapped_to = "y", reuse_type = "no_reuse", volume = 0 : i64}, {depth = 3 : i64, iterator = "%arg9", mapped_to = "x", reuse_type = "no_reuse", volume = 0 : i64}], temporal = [{depth = 0 : i64, iterator = "%arg6", reuse_type = "no_reuse", volume = 0 : i64}, {depth = 1 : i64, iterator = "%arg7", reuse_type = "no_reuse", volume = 0 : i64}]}} : memref<*xf32> to memref<32x32xf32, strided<[512, 1], offset: ?>>
            memref.copy %7, %reinterpret_cast : memref<32x32xf32> to memref<32x32xf32, strided<[512, 1], offset: ?>>
          } {tmd.mapped_to = "x"}
        } {tmd.mapped_to = "y"}
      }
    }
    return
  }
  func.func @matmul_kernel__d1i1_d0i1__c0mem_c1mem(%arg0: memref<*xf32> {tt.divisibility = 16 : i32}, %arg1: memref<*xf32> {tt.divisibility = 16 : i32}, %arg2: memref<*xf32> {tt.divisibility = 16 : i32}, %arg3: index, %arg4: index, %arg5: index) {
    affine.for %arg6 = 0 to affine_map<(d0, d1) -> (d0)>(%arg3, %arg4) {
      affine.for %arg7 = 0 to affine_map<(d0, d1) -> ((d1 ceildiv 8) ceildiv 8)>(%arg3, %arg4) {
        affine.parallel (%arg8) = (0) to (8) {
          affine.parallel (%arg9) = (0) to (8) {
            %cst = arith.constant 0.000000e+00 : f32
            %alloc = memref.alloc() {alignment = 64 : i64} : memref<32x32xf32>
            linalg.fill ins(%cst : f32) outs(%alloc : memref<32x32xf32>)
            %c0 = arith.constant 0 : index
            %c1 = arith.constant 1 : index
            %alloc_0 = memref.alloc() {alignment = 64 : i64} : memref<32x32xf32>
            memref.copy %alloc, %alloc_0 : memref<32x32xf32> to memref<32x32xf32>
            %c4 = arith.constant 4 : index
            %7 = scf.for %arg10 = %c0 to %c4 step %c1 iter_args(%arg11 = %alloc_0) -> (memref<32x32xf32>) {
              %9 = scf.for %arg12 = %c0 to %c4 step %c1 iter_args(%arg13 = %arg11) -> (memref<32x32xf32>) {
                %10 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 16384 + d1 * 128 + d2 * 32)>(%arg6, %arg10, %arg12)
                %reinterpret_cast_1 = memref.reinterpret_cast %arg0 to offset: [%10], sizes: [32, 32], strides: [512, 1] {tmd.reuse = {sequential = [{depth = 4 : i64, iterator = "%arg10", reuse_type = "no_reuse", volume = 0 : i64}], spatial = [{depth = 2 : i64, iterator = "%arg8", mapped_to = "x", reuse_type = "total_reuse", volume = 4096 : i64}, {depth = 3 : i64, iterator = "%arg9", mapped_to = "y", reuse_type = "total_reuse", volume = 4096 : i64}], temporal = [{depth = 0 : i64, iterator = "%arg6", reuse_type = "no_reuse", volume = 0 : i64}, {depth = 1 : i64, iterator = "%arg7", reuse_type = "total_reuse", volume = 4096 : i64}]}} : memref<*xf32> to memref<32x32xf32, strided<[512, 1], offset: ?>>
                %alloc_2 = memref.alloc() {tmd.alloc = {local = true, memory_name = "L1", size = 4096 : i64}} : memref<32x32xf32>
                memref.copy %reinterpret_cast_1, %alloc_2 {tmd.copy.choice = {kind = "mem", memory_name = "L1"}} : memref<32x32xf32, strided<[512, 1], offset: ?>> to memref<32x32xf32>
                %11 = affine.apply affine_map<(d0, d1, d2, d3, d4) -> (d0 * 32 + d1 * 2048 + d2 * 256 + d3 * 65536 + d4 * 16384)>(%arg9, %arg7, %arg8, %arg10, %arg12)
                %reinterpret_cast_3 = memref.reinterpret_cast %arg1 to offset: [%11], sizes: [32, 32], strides: [512, 1] {tmd.reuse = {sequential = [{depth = 4 : i64, iterator = "%arg10", reuse_type = "no_reuse", volume = 0 : i64}], spatial = [{depth = 2 : i64, iterator = "%arg8", mapped_to = "x", reuse_type = "no_reuse", volume = 0 : i64}, {depth = 3 : i64, iterator = "%arg9", mapped_to = "y", reuse_type = "no_reuse", volume = 0 : i64}], temporal = [{depth = 0 : i64, iterator = "%arg6", reuse_type = "total_reuse", volume = 4096 : i64}, {depth = 1 : i64, iterator = "%arg7", reuse_type = "no_reuse", volume = 0 : i64}]}} : memref<*xf32> to memref<32x32xf32, strided<[512, 1], offset: ?>>
                %alloc_4 = memref.alloc() {tmd.alloc = {local = true, memory_name = "L1", size = 4096 : i64}} : memref<32x32xf32>
                memref.copy %reinterpret_cast_3, %alloc_4 {tmd.copy.choice = {kind = "mem", memory_name = "L1"}} : memref<32x32xf32, strided<[512, 1], offset: ?>> to memref<32x32xf32>
                %alloc_5 = memref.alloc() {alignment = 64 : i64} : memref<32x32xf32>
                memref.copy %alloc, %alloc_5 : memref<32x32xf32> to memref<32x32xf32>
                linalg.matmul ins(%alloc_2, %alloc_4 : memref<32x32xf32>, memref<32x32xf32>) outs(%alloc_5 : memref<32x32xf32>)
                linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%arg13, %alloc_5 : memref<32x32xf32>, memref<32x32xf32>) outs(%arg13 : memref<32x32xf32>) {
                ^bb0(%in: f32, %in_6: f32, %out: f32):
                  %12 = arith.addf %in, %in_6 : f32
                  linalg.yield %12 : f32
                }
                scf.yield %arg13 : memref<32x32xf32>
              }
              scf.yield %9 : memref<32x32xf32>
            }
            %8 = affine.apply affine_map<(d0, d1, d2, d3) -> (d0 * 16384 + d1 * 32 + d2 * 2048 + d3 * 256)>(%arg6, %arg9, %arg7, %arg8)
            %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%8], sizes: [32, 32], strides: [512, 1] {tmd.reuse = {spatial = [{depth = 2 : i64, iterator = "%arg8", mapped_to = "x", reuse_type = "no_reuse", volume = 0 : i64}, {depth = 3 : i64, iterator = "%arg9", mapped_to = "y", reuse_type = "no_reuse", volume = 0 : i64}], temporal = [{depth = 0 : i64, iterator = "%arg6", reuse_type = "no_reuse", volume = 0 : i64}, {depth = 1 : i64, iterator = "%arg7", reuse_type = "no_reuse", volume = 0 : i64}]}} : memref<*xf32> to memref<32x32xf32, strided<[512, 1], offset: ?>>
            memref.copy %7, %reinterpret_cast : memref<32x32xf32> to memref<32x32xf32, strided<[512, 1], offset: ?>>
          } {tmd.mapped_to = "y"}
        } {tmd.mapped_to = "x"}
      }
    }
    return
  }
  func.func @matmul_kernel__d1i1_d0i1__c0bx_c1mem(%arg0: memref<*xf32> {tt.divisibility = 16 : i32}, %arg1: memref<*xf32> {tt.divisibility = 16 : i32}, %arg2: memref<*xf32> {tt.divisibility = 16 : i32}, %arg3: index, %arg4: index, %arg5: index) {
    affine.for %arg6 = 0 to affine_map<(d0, d1) -> (d0)>(%arg3, %arg4) {
      affine.for %arg7 = 0 to affine_map<(d0, d1) -> ((d1 ceildiv 8) ceildiv 8)>(%arg3, %arg4) {
        affine.parallel (%arg8) = (0) to (8) {
          affine.parallel (%arg9) = (0) to (8) {
            %cst = arith.constant 0.000000e+00 : f32
            %alloc = memref.alloc() {alignment = 64 : i64} : memref<32x32xf32>
            linalg.fill ins(%cst : f32) outs(%alloc : memref<32x32xf32>)
            %c0 = arith.constant 0 : index
            %c1 = arith.constant 1 : index
            %alloc_0 = memref.alloc() {alignment = 64 : i64} : memref<32x32xf32>
            memref.copy %alloc, %alloc_0 : memref<32x32xf32> to memref<32x32xf32>
            %c4 = arith.constant 4 : index
            %7 = scf.for %arg10 = %c0 to %c4 step %c1 iter_args(%arg11 = %alloc_0) -> (memref<32x32xf32>) {
              %9 = scf.for %arg12 = %c0 to %c4 step %c1 iter_args(%arg13 = %arg11) -> (memref<32x32xf32>) {
                %10 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 16384 + d1 * 128 + d2 * 32)>(%arg6, %arg10, %arg12)
                %reinterpret_cast_1 = memref.reinterpret_cast %arg0 to offset: [%10], sizes: [32, 32], strides: [512, 1] {tmd.reuse = {sequential = [{depth = 4 : i64, iterator = "%arg10", reuse_type = "no_reuse", volume = 0 : i64}], spatial = [{depth = 2 : i64, iterator = "%arg8", mapped_to = "x", reuse_type = "total_reuse", volume = 4096 : i64}, {depth = 3 : i64, iterator = "%arg9", mapped_to = "y", reuse_type = "total_reuse", volume = 4096 : i64}], temporal = [{depth = 0 : i64, iterator = "%arg6", reuse_type = "no_reuse", volume = 0 : i64}, {depth = 1 : i64, iterator = "%arg7", reuse_type = "total_reuse", volume = 4096 : i64}]}} : memref<*xf32> to memref<32x32xf32, strided<[512, 1], offset: ?>>
                %alloc_2 = memref.alloc() {tmd.alloc = {local = true, memory_name = "L1", size = 4096 : i64}} : memref<32x32xf32>
                memref.copy %reinterpret_cast_1, %alloc_2 {tmd.copy.choice = {dim = "x", interconnect_name = "horizontal_links", kind = "broadcast"}} : memref<32x32xf32, strided<[512, 1], offset: ?>> to memref<32x32xf32>
                %11 = affine.apply affine_map<(d0, d1, d2, d3, d4) -> (d0 * 32 + d1 * 2048 + d2 * 256 + d3 * 65536 + d4 * 16384)>(%arg9, %arg7, %arg8, %arg10, %arg12)
                %reinterpret_cast_3 = memref.reinterpret_cast %arg1 to offset: [%11], sizes: [32, 32], strides: [512, 1] {tmd.reuse = {sequential = [{depth = 4 : i64, iterator = "%arg10", reuse_type = "no_reuse", volume = 0 : i64}], spatial = [{depth = 2 : i64, iterator = "%arg8", mapped_to = "x", reuse_type = "no_reuse", volume = 0 : i64}, {depth = 3 : i64, iterator = "%arg9", mapped_to = "y", reuse_type = "no_reuse", volume = 0 : i64}], temporal = [{depth = 0 : i64, iterator = "%arg6", reuse_type = "total_reuse", volume = 4096 : i64}, {depth = 1 : i64, iterator = "%arg7", reuse_type = "no_reuse", volume = 0 : i64}]}} : memref<*xf32> to memref<32x32xf32, strided<[512, 1], offset: ?>>
                %alloc_4 = memref.alloc() {tmd.alloc = {local = true, memory_name = "L1", size = 4096 : i64}} : memref<32x32xf32>
                memref.copy %reinterpret_cast_3, %alloc_4 {tmd.copy.choice = {kind = "mem", memory_name = "L1"}} : memref<32x32xf32, strided<[512, 1], offset: ?>> to memref<32x32xf32>
                %alloc_5 = memref.alloc() {alignment = 64 : i64} : memref<32x32xf32>
                memref.copy %alloc, %alloc_5 : memref<32x32xf32> to memref<32x32xf32>
                linalg.matmul ins(%alloc_2, %alloc_4 : memref<32x32xf32>, memref<32x32xf32>) outs(%alloc_5 : memref<32x32xf32>)
                linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%arg13, %alloc_5 : memref<32x32xf32>, memref<32x32xf32>) outs(%arg13 : memref<32x32xf32>) {
                ^bb0(%in: f32, %in_6: f32, %out: f32):
                  %12 = arith.addf %in, %in_6 : f32
                  linalg.yield %12 : f32
                }
                scf.yield %arg13 : memref<32x32xf32>
              }
              scf.yield %9 : memref<32x32xf32>
            }
            %8 = affine.apply affine_map<(d0, d1, d2, d3) -> (d0 * 16384 + d1 * 32 + d2 * 2048 + d3 * 256)>(%arg6, %arg9, %arg7, %arg8)
            %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%8], sizes: [32, 32], strides: [512, 1] {tmd.reuse = {spatial = [{depth = 2 : i64, iterator = "%arg8", mapped_to = "x", reuse_type = "no_reuse", volume = 0 : i64}, {depth = 3 : i64, iterator = "%arg9", mapped_to = "y", reuse_type = "no_reuse", volume = 0 : i64}], temporal = [{depth = 0 : i64, iterator = "%arg6", reuse_type = "no_reuse", volume = 0 : i64}, {depth = 1 : i64, iterator = "%arg7", reuse_type = "no_reuse", volume = 0 : i64}]}} : memref<*xf32> to memref<32x32xf32, strided<[512, 1], offset: ?>>
            memref.copy %7, %reinterpret_cast : memref<32x32xf32> to memref<32x32xf32, strided<[512, 1], offset: ?>>
          } {tmd.mapped_to = "y"}
        } {tmd.mapped_to = "x"}
      }
    }
    return
  }
  func.func @matmul_kernel__d1i1_d0i1__c0by_c1mem(%arg0: memref<*xf32> {tt.divisibility = 16 : i32}, %arg1: memref<*xf32> {tt.divisibility = 16 : i32}, %arg2: memref<*xf32> {tt.divisibility = 16 : i32}, %arg3: index, %arg4: index, %arg5: index) {
    affine.for %arg6 = 0 to affine_map<(d0, d1) -> (d0)>(%arg3, %arg4) {
      affine.for %arg7 = 0 to affine_map<(d0, d1) -> ((d1 ceildiv 8) ceildiv 8)>(%arg3, %arg4) {
        affine.parallel (%arg8) = (0) to (8) {
          affine.parallel (%arg9) = (0) to (8) {
            %cst = arith.constant 0.000000e+00 : f32
            %alloc = memref.alloc() {alignment = 64 : i64} : memref<32x32xf32>
            linalg.fill ins(%cst : f32) outs(%alloc : memref<32x32xf32>)
            %c0 = arith.constant 0 : index
            %c1 = arith.constant 1 : index
            %alloc_0 = memref.alloc() {alignment = 64 : i64} : memref<32x32xf32>
            memref.copy %alloc, %alloc_0 : memref<32x32xf32> to memref<32x32xf32>
            %c4 = arith.constant 4 : index
            %7 = scf.for %arg10 = %c0 to %c4 step %c1 iter_args(%arg11 = %alloc_0) -> (memref<32x32xf32>) {
              %9 = scf.for %arg12 = %c0 to %c4 step %c1 iter_args(%arg13 = %arg11) -> (memref<32x32xf32>) {
                %10 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 16384 + d1 * 128 + d2 * 32)>(%arg6, %arg10, %arg12)
                %reinterpret_cast_1 = memref.reinterpret_cast %arg0 to offset: [%10], sizes: [32, 32], strides: [512, 1] {tmd.reuse = {sequential = [{depth = 4 : i64, iterator = "%arg10", reuse_type = "no_reuse", volume = 0 : i64}], spatial = [{depth = 2 : i64, iterator = "%arg8", mapped_to = "x", reuse_type = "total_reuse", volume = 4096 : i64}, {depth = 3 : i64, iterator = "%arg9", mapped_to = "y", reuse_type = "total_reuse", volume = 4096 : i64}], temporal = [{depth = 0 : i64, iterator = "%arg6", reuse_type = "no_reuse", volume = 0 : i64}, {depth = 1 : i64, iterator = "%arg7", reuse_type = "total_reuse", volume = 4096 : i64}]}} : memref<*xf32> to memref<32x32xf32, strided<[512, 1], offset: ?>>
                %alloc_2 = memref.alloc() {tmd.alloc = {local = true, memory_name = "L1", size = 4096 : i64}} : memref<32x32xf32>
                memref.copy %reinterpret_cast_1, %alloc_2 {tmd.copy.choice = {dim = "y", interconnect_name = "vertical_links", kind = "broadcast"}} : memref<32x32xf32, strided<[512, 1], offset: ?>> to memref<32x32xf32>
                %11 = affine.apply affine_map<(d0, d1, d2, d3, d4) -> (d0 * 32 + d1 * 2048 + d2 * 256 + d3 * 65536 + d4 * 16384)>(%arg9, %arg7, %arg8, %arg10, %arg12)
                %reinterpret_cast_3 = memref.reinterpret_cast %arg1 to offset: [%11], sizes: [32, 32], strides: [512, 1] {tmd.reuse = {sequential = [{depth = 4 : i64, iterator = "%arg10", reuse_type = "no_reuse", volume = 0 : i64}], spatial = [{depth = 2 : i64, iterator = "%arg8", mapped_to = "x", reuse_type = "no_reuse", volume = 0 : i64}, {depth = 3 : i64, iterator = "%arg9", mapped_to = "y", reuse_type = "no_reuse", volume = 0 : i64}], temporal = [{depth = 0 : i64, iterator = "%arg6", reuse_type = "total_reuse", volume = 4096 : i64}, {depth = 1 : i64, iterator = "%arg7", reuse_type = "no_reuse", volume = 0 : i64}]}} : memref<*xf32> to memref<32x32xf32, strided<[512, 1], offset: ?>>
                %alloc_4 = memref.alloc() {tmd.alloc = {local = true, memory_name = "L1", size = 4096 : i64}} : memref<32x32xf32>
                memref.copy %reinterpret_cast_3, %alloc_4 {tmd.copy.choice = {kind = "mem", memory_name = "L1"}} : memref<32x32xf32, strided<[512, 1], offset: ?>> to memref<32x32xf32>
                %alloc_5 = memref.alloc() {alignment = 64 : i64} : memref<32x32xf32>
                memref.copy %alloc, %alloc_5 : memref<32x32xf32> to memref<32x32xf32>
                linalg.matmul ins(%alloc_2, %alloc_4 : memref<32x32xf32>, memref<32x32xf32>) outs(%alloc_5 : memref<32x32xf32>)
                linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%arg13, %alloc_5 : memref<32x32xf32>, memref<32x32xf32>) outs(%arg13 : memref<32x32xf32>) {
                ^bb0(%in: f32, %in_6: f32, %out: f32):
                  %12 = arith.addf %in, %in_6 : f32
                  linalg.yield %12 : f32
                }
                scf.yield %arg13 : memref<32x32xf32>
              }
              scf.yield %9 : memref<32x32xf32>
            }
            %8 = affine.apply affine_map<(d0, d1, d2, d3) -> (d0 * 16384 + d1 * 32 + d2 * 2048 + d3 * 256)>(%arg6, %arg9, %arg7, %arg8)
            %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%8], sizes: [32, 32], strides: [512, 1] {tmd.reuse = {spatial = [{depth = 2 : i64, iterator = "%arg8", mapped_to = "x", reuse_type = "no_reuse", volume = 0 : i64}, {depth = 3 : i64, iterator = "%arg9", mapped_to = "y", reuse_type = "no_reuse", volume = 0 : i64}], temporal = [{depth = 0 : i64, iterator = "%arg6", reuse_type = "no_reuse", volume = 0 : i64}, {depth = 1 : i64, iterator = "%arg7", reuse_type = "no_reuse", volume = 0 : i64}]}} : memref<*xf32> to memref<32x32xf32, strided<[512, 1], offset: ?>>
            memref.copy %7, %reinterpret_cast : memref<32x32xf32> to memref<32x32xf32, strided<[512, 1], offset: ?>>
          } {tmd.mapped_to = "y"}
        } {tmd.mapped_to = "x"}
      }
    }
    return
  }
}
