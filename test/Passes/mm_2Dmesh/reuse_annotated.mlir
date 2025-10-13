#map = affine_map<(d0, d1) -> (d0, d1)>
#map1 = affine_map<(d0, d1) -> (d0 + 1, d1)>
#map2 = affine_map<(d0, d1) -> (d0, d1 + 1)>
#map3 = affine_map<(d0, d1) -> ((d0 ceildiv 8) ceildiv 8)>
#map4 = affine_map<(d0, d1) -> (d1)>
#map5 = affine_map<()[s0] -> ((s0 + 31) floordiv 32)>
#map6 = affine_map<(d0, d1, d2, d3) -> (d0 * 32 + d1 * 16384 + d2 * 1048576 + d3 * 131072)>
#map7 = affine_map<(d0, d1) -> (d1 * 16384 + d0 * 32)>
#map8 = affine_map<(d0, d1) -> (d0 ceildiv 8)>
#map9 = affine_map<(d0, d1) -> (d1 ceildiv 8)>
#map10 = affine_map<(d0, d1, d2) -> (d0 * 32 + d1 * 131072 + d2 * 16384)>
#map11 = affine_map<(d0, d1, d2) -> (d0 * 16384 + d1 * 256 + d2 * 32)>
#map12 = affine_map<(d0, d1, d2, d3) -> (d0 * 256 + d1 * 32 + d2 * 131072 + d3 * 16384)>
#map13 = affine_map<(d0, d1) -> (d0)>
#map14 = affine_map<(d0, d1) -> ((d1 ceildiv 8) ceildiv 8)>
#map15 = affine_map<(d0, d1, d2, d3) -> (d0 * 16384 + d1 * 32 + d2 * 2048 + d3 * 256)>
module {
  %0 = df.spatial_dim "x", 8
  %1 = df.spatial_dim "y", 8
  %2 = df.compute "cores", %0, %1 {map = #map}
  %3 = df.memory "L1", %0, %1 {map = #map}
  %4 = df.mux %2 : !df.compute, %3 : !df.memory, %0, %1 {map = #map}
  %5 = df.interconnects "horizontal_links" %3 : !df.memory, %3 : !df.memory, %0, %1 {map = #map1} : !df.interconnect
  %6 = df.interconnects "vertical_links" %3 : !df.memory, %3 : !df.memory, %0, %1 {map = #map2} : !df.interconnect
  func.func @matmul_kernel__d0i0_d1i0_f0_f1(%arg0: memref<*xf32> {tt.divisibility = 16 : i32}, %arg1: memref<*xf32> {tt.divisibility = 16 : i32}, %arg2: memref<*xf32> {tt.divisibility = 16 : i32}, %arg3: index {tt.divisibility = 16 : i32}, %arg4: index {tt.divisibility = 16 : i32}, %arg5: index {tt.divisibility = 16 : i32}, %arg6: index, %arg7: index, %arg8: index) {
    affine.for %arg9 = 0 to #map3(%arg6, %arg7) {
      affine.for %arg10 = 0 to #map4(%arg6, %arg7) {
        affine.parallel (%arg11) = (0) to (8) {
          affine.parallel (%arg12) = (0) to (8) {
            %cst = arith.constant 0.000000e+00 : f32
            %alloc = memref.alloc() {alignment = 64 : i64} : memref<32x32xf32>
            linalg.fill ins(%cst : f32) outs(%alloc : memref<32x32xf32>)
            %7 = affine.apply #map5()[%arg5]
            %alloc_0 = memref.alloc() {alignment = 64 : i64} : memref<32x32xf32>
            memref.copy %alloc, %alloc_0 : memref<32x32xf32> to memref<32x32xf32>
            %c0 = arith.constant 0 : index
            %c1 = arith.constant 1 : index
            %8 = scf.for %arg13 = %c0 to %7 step %c1 iter_args(%arg14 = %alloc_0) -> (memref<32x32xf32>) {
              %10 = affine.apply #map6(%arg13, %arg12, %arg9, %arg11)
              %reinterpret_cast_1 = memref.reinterpret_cast %arg0 to offset: [%10], sizes: [32, 32], strides: [512, 1] {tmd.reuse = {sequential = [{depth = 4 : i64, iterator = "%arg13", reuse_type = "no_reuse", volume = 0 : i64}], spatial = [{depth = 2 : i64, iterator = "%arg11", mapped_to = "y", reuse_type = "no_reuse", volume = 0 : i64}, {depth = 3 : i64, iterator = "%arg12", mapped_to = "x", reuse_type = "no_reuse", volume = 0 : i64}], temporal = [{depth = 0 : i64, iterator = "%arg9", reuse_type = "no_reuse", volume = 0 : i64}, {depth = 1 : i64, iterator = "%arg10", reuse_type = "total_reuse", volume = 4096 : i64}]}} : memref<*xf32> to memref<32x32xf32, strided<[512, 1], offset: ?>>
              %alloc_2 = memref.alloc() : memref<32x32xf32>
              memref.copy %reinterpret_cast_1, %alloc_2 : memref<32x32xf32, strided<[512, 1], offset: ?>> to memref<32x32xf32>
              %11 = affine.apply #map7(%arg10, %arg13)
              %reinterpret_cast_3 = memref.reinterpret_cast %arg1 to offset: [%11], sizes: [32, 32], strides: [512, 1] {tmd.reuse = {sequential = [{depth = 4 : i64, iterator = "%arg13", reuse_type = "no_reuse", volume = 0 : i64}], spatial = [{depth = 2 : i64, iterator = "%arg11", mapped_to = "y", reuse_type = "total_reuse", volume = 4096 : i64}, {depth = 3 : i64, iterator = "%arg12", mapped_to = "x", reuse_type = "total_reuse", volume = 4096 : i64}], temporal = [{depth = 0 : i64, iterator = "%arg9", reuse_type = "total_reuse", volume = 4096 : i64}, {depth = 1 : i64, iterator = "%arg10", reuse_type = "no_reuse", volume = 0 : i64}]}} : memref<*xf32> to memref<32x32xf32, strided<[512, 1], offset: ?>>
              %alloc_4 = memref.alloc() : memref<32x32xf32>
              memref.copy %reinterpret_cast_3, %alloc_4 : memref<32x32xf32, strided<[512, 1], offset: ?>> to memref<32x32xf32>
              %alloc_5 = memref.alloc() {alignment = 64 : i64} : memref<32x32xf32>
              memref.copy %alloc, %alloc_5 : memref<32x32xf32> to memref<32x32xf32>
              linalg.matmul ins(%alloc_2, %alloc_4 : memref<32x32xf32>, memref<32x32xf32>) outs(%alloc_5 : memref<32x32xf32>)
              linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel", "parallel"]} ins(%arg14, %alloc_5 : memref<32x32xf32>, memref<32x32xf32>) outs(%arg14 : memref<32x32xf32>) {
              ^bb0(%in: f32, %in_6: f32, %out: f32):
                %12 = arith.addf %in, %in_6 : f32
                linalg.yield %12 : f32
              }
              scf.yield %arg14 : memref<32x32xf32>
            }
            %9 = affine.apply #map6(%arg10, %arg12, %arg9, %arg11)
            %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%9], sizes: [32, 32], strides: [512, 1] {tmd.reuse = {spatial = [{depth = 2 : i64, iterator = "%arg11", mapped_to = "y", reuse_type = "no_reuse", volume = 0 : i64}, {depth = 3 : i64, iterator = "%arg12", mapped_to = "x", reuse_type = "no_reuse", volume = 0 : i64}], temporal = [{depth = 0 : i64, iterator = "%arg9", reuse_type = "no_reuse", volume = 0 : i64}, {depth = 1 : i64, iterator = "%arg10", reuse_type = "no_reuse", volume = 0 : i64}]}} : memref<*xf32> to memref<32x32xf32, strided<[512, 1], offset: ?>>
            memref.copy %8, %reinterpret_cast : memref<32x32xf32> to memref<32x32xf32, strided<[512, 1], offset: ?>>
          } {tmd.mapped_to = "x"}
        } {tmd.mapped_to = "y"}
      }
    }
    return
  }
  func.func @matmul_kernel__d0i0_d1i0_f1_f0(%arg0: memref<*xf32> {tt.divisibility = 16 : i32}, %arg1: memref<*xf32> {tt.divisibility = 16 : i32}, %arg2: memref<*xf32> {tt.divisibility = 16 : i32}, %arg3: index {tt.divisibility = 16 : i32}, %arg4: index {tt.divisibility = 16 : i32}, %arg5: index {tt.divisibility = 16 : i32}, %arg6: index, %arg7: index, %arg8: index) {
    affine.for %arg9 = 0 to #map4(%arg6, %arg7) {
      affine.for %arg10 = 0 to #map3(%arg6, %arg7) {
        affine.parallel (%arg11) = (0) to (8) {
          affine.parallel (%arg12) = (0) to (8) {
            %cst = arith.constant 0.000000e+00 : f32
            %alloc = memref.alloc() {alignment = 64 : i64} : memref<32x32xf32>
            linalg.fill ins(%cst : f32) outs(%alloc : memref<32x32xf32>)
            %7 = affine.apply #map5()[%arg5]
            %alloc_0 = memref.alloc() {alignment = 64 : i64} : memref<32x32xf32>
            memref.copy %alloc, %alloc_0 : memref<32x32xf32> to memref<32x32xf32>
            %c0 = arith.constant 0 : index
            %c1 = arith.constant 1 : index
            %8 = scf.for %arg13 = %c0 to %7 step %c1 iter_args(%arg14 = %alloc_0) -> (memref<32x32xf32>) {
              %10 = affine.apply #map6(%arg13, %arg12, %arg10, %arg11)
              %reinterpret_cast_1 = memref.reinterpret_cast %arg0 to offset: [%10], sizes: [32, 32], strides: [512, 1] {tmd.reuse = {sequential = [{depth = 4 : i64, iterator = "%arg13", reuse_type = "no_reuse", volume = 0 : i64}], spatial = [{depth = 2 : i64, iterator = "%arg11", mapped_to = "y", reuse_type = "no_reuse", volume = 0 : i64}, {depth = 3 : i64, iterator = "%arg12", mapped_to = "x", reuse_type = "no_reuse", volume = 0 : i64}], temporal = [{depth = 0 : i64, iterator = "%arg9", reuse_type = "total_reuse", volume = 4096 : i64}, {depth = 1 : i64, iterator = "%arg10", reuse_type = "no_reuse", volume = 0 : i64}]}} : memref<*xf32> to memref<32x32xf32, strided<[512, 1], offset: ?>>
              %alloc_2 = memref.alloc() : memref<32x32xf32>
              memref.copy %reinterpret_cast_1, %alloc_2 : memref<32x32xf32, strided<[512, 1], offset: ?>> to memref<32x32xf32>
              %11 = affine.apply #map7(%arg9, %arg13)
              %reinterpret_cast_3 = memref.reinterpret_cast %arg1 to offset: [%11], sizes: [32, 32], strides: [512, 1] {tmd.reuse = {sequential = [{depth = 4 : i64, iterator = "%arg13", reuse_type = "no_reuse", volume = 0 : i64}], spatial = [{depth = 2 : i64, iterator = "%arg11", mapped_to = "y", reuse_type = "total_reuse", volume = 4096 : i64}, {depth = 3 : i64, iterator = "%arg12", mapped_to = "x", reuse_type = "total_reuse", volume = 4096 : i64}], temporal = [{depth = 0 : i64, iterator = "%arg9", reuse_type = "no_reuse", volume = 0 : i64}, {depth = 1 : i64, iterator = "%arg10", reuse_type = "total_reuse", volume = 4096 : i64}]}} : memref<*xf32> to memref<32x32xf32, strided<[512, 1], offset: ?>>
              %alloc_4 = memref.alloc() : memref<32x32xf32>
              memref.copy %reinterpret_cast_3, %alloc_4 : memref<32x32xf32, strided<[512, 1], offset: ?>> to memref<32x32xf32>
              %alloc_5 = memref.alloc() {alignment = 64 : i64} : memref<32x32xf32>
              memref.copy %alloc, %alloc_5 : memref<32x32xf32> to memref<32x32xf32>
              linalg.matmul ins(%alloc_2, %alloc_4 : memref<32x32xf32>, memref<32x32xf32>) outs(%alloc_5 : memref<32x32xf32>)
              linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel", "parallel"]} ins(%arg14, %alloc_5 : memref<32x32xf32>, memref<32x32xf32>) outs(%arg14 : memref<32x32xf32>) {
              ^bb0(%in: f32, %in_6: f32, %out: f32):
                %12 = arith.addf %in, %in_6 : f32
                linalg.yield %12 : f32
              }
              scf.yield %arg14 : memref<32x32xf32>
            }
            %9 = affine.apply #map6(%arg9, %arg12, %arg10, %arg11)
            %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%9], sizes: [32, 32], strides: [512, 1] {tmd.reuse = {spatial = [{depth = 2 : i64, iterator = "%arg11", mapped_to = "y", reuse_type = "no_reuse", volume = 0 : i64}, {depth = 3 : i64, iterator = "%arg12", mapped_to = "x", reuse_type = "no_reuse", volume = 0 : i64}], temporal = [{depth = 0 : i64, iterator = "%arg9", reuse_type = "no_reuse", volume = 0 : i64}, {depth = 1 : i64, iterator = "%arg10", reuse_type = "no_reuse", volume = 0 : i64}]}} : memref<*xf32> to memref<32x32xf32, strided<[512, 1], offset: ?>>
            memref.copy %8, %reinterpret_cast : memref<32x32xf32> to memref<32x32xf32, strided<[512, 1], offset: ?>>
          } {tmd.mapped_to = "x"}
        } {tmd.mapped_to = "y"}
      }
    }
    return
  }
  func.func @matmul_kernel__d1i0_d0i0_f0_f1(%arg0: memref<*xf32> {tt.divisibility = 16 : i32}, %arg1: memref<*xf32> {tt.divisibility = 16 : i32}, %arg2: memref<*xf32> {tt.divisibility = 16 : i32}, %arg3: index {tt.divisibility = 16 : i32}, %arg4: index {tt.divisibility = 16 : i32}, %arg5: index {tt.divisibility = 16 : i32}, %arg6: index, %arg7: index, %arg8: index) {
    affine.for %arg9 = 0 to #map3(%arg6, %arg7) {
      affine.for %arg10 = 0 to #map4(%arg6, %arg7) {
        affine.parallel (%arg11) = (0) to (8) {
          affine.parallel (%arg12) = (0) to (8) {
            %cst = arith.constant 0.000000e+00 : f32
            %alloc = memref.alloc() {alignment = 64 : i64} : memref<32x32xf32>
            linalg.fill ins(%cst : f32) outs(%alloc : memref<32x32xf32>)
            %7 = affine.apply #map5()[%arg5]
            %alloc_0 = memref.alloc() {alignment = 64 : i64} : memref<32x32xf32>
            memref.copy %alloc, %alloc_0 : memref<32x32xf32> to memref<32x32xf32>
            %c0 = arith.constant 0 : index
            %c1 = arith.constant 1 : index
            %8 = scf.for %arg13 = %c0 to %7 step %c1 iter_args(%arg14 = %alloc_0) -> (memref<32x32xf32>) {
              %10 = affine.apply #map6(%arg13, %arg12, %arg9, %arg11)
              %reinterpret_cast_1 = memref.reinterpret_cast %arg0 to offset: [%10], sizes: [32, 32], strides: [512, 1] {tmd.reuse = {sequential = [{depth = 4 : i64, iterator = "%arg13", reuse_type = "no_reuse", volume = 0 : i64}], spatial = [{depth = 2 : i64, iterator = "%arg11", mapped_to = "x", reuse_type = "no_reuse", volume = 0 : i64}, {depth = 3 : i64, iterator = "%arg12", mapped_to = "y", reuse_type = "no_reuse", volume = 0 : i64}], temporal = [{depth = 0 : i64, iterator = "%arg9", reuse_type = "no_reuse", volume = 0 : i64}, {depth = 1 : i64, iterator = "%arg10", reuse_type = "total_reuse", volume = 4096 : i64}]}} : memref<*xf32> to memref<32x32xf32, strided<[512, 1], offset: ?>>
              %alloc_2 = memref.alloc() : memref<32x32xf32>
              memref.copy %reinterpret_cast_1, %alloc_2 : memref<32x32xf32, strided<[512, 1], offset: ?>> to memref<32x32xf32>
              %11 = affine.apply #map7(%arg10, %arg13)
              %reinterpret_cast_3 = memref.reinterpret_cast %arg1 to offset: [%11], sizes: [32, 32], strides: [512, 1] {tmd.reuse = {sequential = [{depth = 4 : i64, iterator = "%arg13", reuse_type = "no_reuse", volume = 0 : i64}], spatial = [{depth = 2 : i64, iterator = "%arg11", mapped_to = "x", reuse_type = "total_reuse", volume = 4096 : i64}, {depth = 3 : i64, iterator = "%arg12", mapped_to = "y", reuse_type = "total_reuse", volume = 4096 : i64}], temporal = [{depth = 0 : i64, iterator = "%arg9", reuse_type = "total_reuse", volume = 4096 : i64}, {depth = 1 : i64, iterator = "%arg10", reuse_type = "no_reuse", volume = 0 : i64}]}} : memref<*xf32> to memref<32x32xf32, strided<[512, 1], offset: ?>>
              %alloc_4 = memref.alloc() : memref<32x32xf32>
              memref.copy %reinterpret_cast_3, %alloc_4 : memref<32x32xf32, strided<[512, 1], offset: ?>> to memref<32x32xf32>
              %alloc_5 = memref.alloc() {alignment = 64 : i64} : memref<32x32xf32>
              memref.copy %alloc, %alloc_5 : memref<32x32xf32> to memref<32x32xf32>
              linalg.matmul ins(%alloc_2, %alloc_4 : memref<32x32xf32>, memref<32x32xf32>) outs(%alloc_5 : memref<32x32xf32>)
              linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel", "parallel"]} ins(%arg14, %alloc_5 : memref<32x32xf32>, memref<32x32xf32>) outs(%arg14 : memref<32x32xf32>) {
              ^bb0(%in: f32, %in_6: f32, %out: f32):
                %12 = arith.addf %in, %in_6 : f32
                linalg.yield %12 : f32
              }
              scf.yield %arg14 : memref<32x32xf32>
            }
            %9 = affine.apply #map6(%arg10, %arg12, %arg9, %arg11)
            %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%9], sizes: [32, 32], strides: [512, 1] {tmd.reuse = {spatial = [{depth = 2 : i64, iterator = "%arg11", mapped_to = "x", reuse_type = "no_reuse", volume = 0 : i64}, {depth = 3 : i64, iterator = "%arg12", mapped_to = "y", reuse_type = "no_reuse", volume = 0 : i64}], temporal = [{depth = 0 : i64, iterator = "%arg9", reuse_type = "no_reuse", volume = 0 : i64}, {depth = 1 : i64, iterator = "%arg10", reuse_type = "no_reuse", volume = 0 : i64}]}} : memref<*xf32> to memref<32x32xf32, strided<[512, 1], offset: ?>>
            memref.copy %8, %reinterpret_cast : memref<32x32xf32> to memref<32x32xf32, strided<[512, 1], offset: ?>>
          } {tmd.mapped_to = "y"}
        } {tmd.mapped_to = "x"}
      }
    }
    return
  }
  func.func @matmul_kernel__d1i0_d0i0_f1_f0(%arg0: memref<*xf32> {tt.divisibility = 16 : i32}, %arg1: memref<*xf32> {tt.divisibility = 16 : i32}, %arg2: memref<*xf32> {tt.divisibility = 16 : i32}, %arg3: index {tt.divisibility = 16 : i32}, %arg4: index {tt.divisibility = 16 : i32}, %arg5: index {tt.divisibility = 16 : i32}, %arg6: index, %arg7: index, %arg8: index) {
    affine.for %arg9 = 0 to #map4(%arg6, %arg7) {
      affine.for %arg10 = 0 to #map3(%arg6, %arg7) {
        affine.parallel (%arg11) = (0) to (8) {
          affine.parallel (%arg12) = (0) to (8) {
            %cst = arith.constant 0.000000e+00 : f32
            %alloc = memref.alloc() {alignment = 64 : i64} : memref<32x32xf32>
            linalg.fill ins(%cst : f32) outs(%alloc : memref<32x32xf32>)
            %7 = affine.apply #map5()[%arg5]
            %alloc_0 = memref.alloc() {alignment = 64 : i64} : memref<32x32xf32>
            memref.copy %alloc, %alloc_0 : memref<32x32xf32> to memref<32x32xf32>
            %c0 = arith.constant 0 : index
            %c1 = arith.constant 1 : index
            %8 = scf.for %arg13 = %c0 to %7 step %c1 iter_args(%arg14 = %alloc_0) -> (memref<32x32xf32>) {
              %10 = affine.apply #map6(%arg13, %arg12, %arg10, %arg11)
              %reinterpret_cast_1 = memref.reinterpret_cast %arg0 to offset: [%10], sizes: [32, 32], strides: [512, 1] {tmd.reuse = {sequential = [{depth = 4 : i64, iterator = "%arg13", reuse_type = "no_reuse", volume = 0 : i64}], spatial = [{depth = 2 : i64, iterator = "%arg11", mapped_to = "x", reuse_type = "no_reuse", volume = 0 : i64}, {depth = 3 : i64, iterator = "%arg12", mapped_to = "y", reuse_type = "no_reuse", volume = 0 : i64}], temporal = [{depth = 0 : i64, iterator = "%arg9", reuse_type = "total_reuse", volume = 4096 : i64}, {depth = 1 : i64, iterator = "%arg10", reuse_type = "no_reuse", volume = 0 : i64}]}} : memref<*xf32> to memref<32x32xf32, strided<[512, 1], offset: ?>>
              %alloc_2 = memref.alloc() : memref<32x32xf32>
              memref.copy %reinterpret_cast_1, %alloc_2 : memref<32x32xf32, strided<[512, 1], offset: ?>> to memref<32x32xf32>
              %11 = affine.apply #map7(%arg9, %arg13)
              %reinterpret_cast_3 = memref.reinterpret_cast %arg1 to offset: [%11], sizes: [32, 32], strides: [512, 1] {tmd.reuse = {sequential = [{depth = 4 : i64, iterator = "%arg13", reuse_type = "no_reuse", volume = 0 : i64}], spatial = [{depth = 2 : i64, iterator = "%arg11", mapped_to = "x", reuse_type = "total_reuse", volume = 4096 : i64}, {depth = 3 : i64, iterator = "%arg12", mapped_to = "y", reuse_type = "total_reuse", volume = 4096 : i64}], temporal = [{depth = 0 : i64, iterator = "%arg9", reuse_type = "no_reuse", volume = 0 : i64}, {depth = 1 : i64, iterator = "%arg10", reuse_type = "total_reuse", volume = 4096 : i64}]}} : memref<*xf32> to memref<32x32xf32, strided<[512, 1], offset: ?>>
              %alloc_4 = memref.alloc() : memref<32x32xf32>
              memref.copy %reinterpret_cast_3, %alloc_4 : memref<32x32xf32, strided<[512, 1], offset: ?>> to memref<32x32xf32>
              %alloc_5 = memref.alloc() {alignment = 64 : i64} : memref<32x32xf32>
              memref.copy %alloc, %alloc_5 : memref<32x32xf32> to memref<32x32xf32>
              linalg.matmul ins(%alloc_2, %alloc_4 : memref<32x32xf32>, memref<32x32xf32>) outs(%alloc_5 : memref<32x32xf32>)
              linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel", "parallel"]} ins(%arg14, %alloc_5 : memref<32x32xf32>, memref<32x32xf32>) outs(%arg14 : memref<32x32xf32>) {
              ^bb0(%in: f32, %in_6: f32, %out: f32):
                %12 = arith.addf %in, %in_6 : f32
                linalg.yield %12 : f32
              }
              scf.yield %arg14 : memref<32x32xf32>
            }
            %9 = affine.apply #map6(%arg9, %arg12, %arg10, %arg11)
            %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%9], sizes: [32, 32], strides: [512, 1] {tmd.reuse = {spatial = [{depth = 2 : i64, iterator = "%arg11", mapped_to = "x", reuse_type = "no_reuse", volume = 0 : i64}, {depth = 3 : i64, iterator = "%arg12", mapped_to = "y", reuse_type = "no_reuse", volume = 0 : i64}], temporal = [{depth = 0 : i64, iterator = "%arg9", reuse_type = "no_reuse", volume = 0 : i64}, {depth = 1 : i64, iterator = "%arg10", reuse_type = "no_reuse", volume = 0 : i64}]}} : memref<*xf32> to memref<32x32xf32, strided<[512, 1], offset: ?>>
            memref.copy %8, %reinterpret_cast : memref<32x32xf32> to memref<32x32xf32, strided<[512, 1], offset: ?>>
          } {tmd.mapped_to = "y"}
        } {tmd.mapped_to = "x"}
      }
    }
    return
  }
  func.func @matmul_kernel__d0i0_d1i1_f0_f1(%arg0: memref<*xf32> {tt.divisibility = 16 : i32}, %arg1: memref<*xf32> {tt.divisibility = 16 : i32}, %arg2: memref<*xf32> {tt.divisibility = 16 : i32}, %arg3: index {tt.divisibility = 16 : i32}, %arg4: index {tt.divisibility = 16 : i32}, %arg5: index {tt.divisibility = 16 : i32}, %arg6: index, %arg7: index, %arg8: index) {
    affine.for %arg9 = 0 to #map8(%arg6, %arg7) {
      affine.for %arg10 = 0 to #map9(%arg6, %arg7) {
        affine.parallel (%arg11) = (0) to (8) {
          affine.parallel (%arg12) = (0) to (8) {
            %cst = arith.constant 0.000000e+00 : f32
            %alloc = memref.alloc() {alignment = 64 : i64} : memref<32x32xf32>
            linalg.fill ins(%cst : f32) outs(%alloc : memref<32x32xf32>)
            %7 = affine.apply #map5()[%arg5]
            %alloc_0 = memref.alloc() {alignment = 64 : i64} : memref<32x32xf32>
            memref.copy %alloc, %alloc_0 : memref<32x32xf32> to memref<32x32xf32>
            %c0 = arith.constant 0 : index
            %c1 = arith.constant 1 : index
            %8 = scf.for %arg13 = %c0 to %7 step %c1 iter_args(%arg14 = %alloc_0) -> (memref<32x32xf32>) {
              %10 = affine.apply #map10(%arg13, %arg9, %arg12)
              %reinterpret_cast_1 = memref.reinterpret_cast %arg0 to offset: [%10], sizes: [32, 32], strides: [512, 1] {tmd.reuse = {sequential = [{depth = 4 : i64, iterator = "%arg13", reuse_type = "no_reuse", volume = 0 : i64}], spatial = [{depth = 2 : i64, iterator = "%arg11", mapped_to = "y", reuse_type = "total_reuse", volume = 4096 : i64}, {depth = 3 : i64, iterator = "%arg12", mapped_to = "x", reuse_type = "no_reuse", volume = 0 : i64}], temporal = [{depth = 0 : i64, iterator = "%arg9", reuse_type = "no_reuse", volume = 0 : i64}, {depth = 1 : i64, iterator = "%arg10", reuse_type = "total_reuse", volume = 4096 : i64}]}} : memref<*xf32> to memref<32x32xf32, strided<[512, 1], offset: ?>>
              %alloc_2 = memref.alloc() : memref<32x32xf32>
              memref.copy %reinterpret_cast_1, %alloc_2 : memref<32x32xf32, strided<[512, 1], offset: ?>> to memref<32x32xf32>
              %11 = affine.apply #map11(%arg13, %arg10, %arg11)
              %reinterpret_cast_3 = memref.reinterpret_cast %arg1 to offset: [%11], sizes: [32, 32], strides: [512, 1] {tmd.reuse = {sequential = [{depth = 4 : i64, iterator = "%arg13", reuse_type = "no_reuse", volume = 0 : i64}], spatial = [{depth = 2 : i64, iterator = "%arg11", mapped_to = "y", reuse_type = "no_reuse", volume = 0 : i64}, {depth = 3 : i64, iterator = "%arg12", mapped_to = "x", reuse_type = "total_reuse", volume = 4096 : i64}], temporal = [{depth = 0 : i64, iterator = "%arg9", reuse_type = "total_reuse", volume = 4096 : i64}, {depth = 1 : i64, iterator = "%arg10", reuse_type = "no_reuse", volume = 0 : i64}]}} : memref<*xf32> to memref<32x32xf32, strided<[512, 1], offset: ?>>
              %alloc_4 = memref.alloc() : memref<32x32xf32>
              memref.copy %reinterpret_cast_3, %alloc_4 : memref<32x32xf32, strided<[512, 1], offset: ?>> to memref<32x32xf32>
              %alloc_5 = memref.alloc() {alignment = 64 : i64} : memref<32x32xf32>
              memref.copy %alloc, %alloc_5 : memref<32x32xf32> to memref<32x32xf32>
              linalg.matmul ins(%alloc_2, %alloc_4 : memref<32x32xf32>, memref<32x32xf32>) outs(%alloc_5 : memref<32x32xf32>)
              linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel", "parallel"]} ins(%arg14, %alloc_5 : memref<32x32xf32>, memref<32x32xf32>) outs(%arg14 : memref<32x32xf32>) {
              ^bb0(%in: f32, %in_6: f32, %out: f32):
                %12 = arith.addf %in, %in_6 : f32
                linalg.yield %12 : f32
              }
              scf.yield %arg14 : memref<32x32xf32>
            }
            %9 = affine.apply #map12(%arg10, %arg11, %arg9, %arg12)
            %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%9], sizes: [32, 32], strides: [512, 1] {tmd.reuse = {spatial = [{depth = 2 : i64, iterator = "%arg11", mapped_to = "y", reuse_type = "no_reuse", volume = 0 : i64}, {depth = 3 : i64, iterator = "%arg12", mapped_to = "x", reuse_type = "no_reuse", volume = 0 : i64}], temporal = [{depth = 0 : i64, iterator = "%arg9", reuse_type = "no_reuse", volume = 0 : i64}, {depth = 1 : i64, iterator = "%arg10", reuse_type = "no_reuse", volume = 0 : i64}]}} : memref<*xf32> to memref<32x32xf32, strided<[512, 1], offset: ?>>
            memref.copy %8, %reinterpret_cast : memref<32x32xf32> to memref<32x32xf32, strided<[512, 1], offset: ?>>
          } {tmd.mapped_to = "x"}
        } {tmd.mapped_to = "y"}
      }
    }
    return
  }
  func.func @matmul_kernel__d0i0_d1i1_f1_f0(%arg0: memref<*xf32> {tt.divisibility = 16 : i32}, %arg1: memref<*xf32> {tt.divisibility = 16 : i32}, %arg2: memref<*xf32> {tt.divisibility = 16 : i32}, %arg3: index {tt.divisibility = 16 : i32}, %arg4: index {tt.divisibility = 16 : i32}, %arg5: index {tt.divisibility = 16 : i32}, %arg6: index, %arg7: index, %arg8: index) {
    affine.for %arg9 = 0 to #map9(%arg6, %arg7) {
      affine.for %arg10 = 0 to #map8(%arg6, %arg7) {
        affine.parallel (%arg11) = (0) to (8) {
          affine.parallel (%arg12) = (0) to (8) {
            %cst = arith.constant 0.000000e+00 : f32
            %alloc = memref.alloc() {alignment = 64 : i64} : memref<32x32xf32>
            linalg.fill ins(%cst : f32) outs(%alloc : memref<32x32xf32>)
            %7 = affine.apply #map5()[%arg5]
            %alloc_0 = memref.alloc() {alignment = 64 : i64} : memref<32x32xf32>
            memref.copy %alloc, %alloc_0 : memref<32x32xf32> to memref<32x32xf32>
            %c0 = arith.constant 0 : index
            %c1 = arith.constant 1 : index
            %8 = scf.for %arg13 = %c0 to %7 step %c1 iter_args(%arg14 = %alloc_0) -> (memref<32x32xf32>) {
              %10 = affine.apply #map10(%arg13, %arg10, %arg12)
              %reinterpret_cast_1 = memref.reinterpret_cast %arg0 to offset: [%10], sizes: [32, 32], strides: [512, 1] {tmd.reuse = {sequential = [{depth = 4 : i64, iterator = "%arg13", reuse_type = "no_reuse", volume = 0 : i64}], spatial = [{depth = 2 : i64, iterator = "%arg11", mapped_to = "y", reuse_type = "total_reuse", volume = 4096 : i64}, {depth = 3 : i64, iterator = "%arg12", mapped_to = "x", reuse_type = "no_reuse", volume = 0 : i64}], temporal = [{depth = 0 : i64, iterator = "%arg9", reuse_type = "total_reuse", volume = 4096 : i64}, {depth = 1 : i64, iterator = "%arg10", reuse_type = "no_reuse", volume = 0 : i64}]}} : memref<*xf32> to memref<32x32xf32, strided<[512, 1], offset: ?>>
              %alloc_2 = memref.alloc() : memref<32x32xf32>
              memref.copy %reinterpret_cast_1, %alloc_2 : memref<32x32xf32, strided<[512, 1], offset: ?>> to memref<32x32xf32>
              %11 = affine.apply #map11(%arg13, %arg9, %arg11)
              %reinterpret_cast_3 = memref.reinterpret_cast %arg1 to offset: [%11], sizes: [32, 32], strides: [512, 1] {tmd.reuse = {sequential = [{depth = 4 : i64, iterator = "%arg13", reuse_type = "no_reuse", volume = 0 : i64}], spatial = [{depth = 2 : i64, iterator = "%arg11", mapped_to = "y", reuse_type = "no_reuse", volume = 0 : i64}, {depth = 3 : i64, iterator = "%arg12", mapped_to = "x", reuse_type = "total_reuse", volume = 4096 : i64}], temporal = [{depth = 0 : i64, iterator = "%arg9", reuse_type = "no_reuse", volume = 0 : i64}, {depth = 1 : i64, iterator = "%arg10", reuse_type = "total_reuse", volume = 4096 : i64}]}} : memref<*xf32> to memref<32x32xf32, strided<[512, 1], offset: ?>>
              %alloc_4 = memref.alloc() : memref<32x32xf32>
              memref.copy %reinterpret_cast_3, %alloc_4 : memref<32x32xf32, strided<[512, 1], offset: ?>> to memref<32x32xf32>
              %alloc_5 = memref.alloc() {alignment = 64 : i64} : memref<32x32xf32>
              memref.copy %alloc, %alloc_5 : memref<32x32xf32> to memref<32x32xf32>
              linalg.matmul ins(%alloc_2, %alloc_4 : memref<32x32xf32>, memref<32x32xf32>) outs(%alloc_5 : memref<32x32xf32>)
              linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel", "parallel"]} ins(%arg14, %alloc_5 : memref<32x32xf32>, memref<32x32xf32>) outs(%arg14 : memref<32x32xf32>) {
              ^bb0(%in: f32, %in_6: f32, %out: f32):
                %12 = arith.addf %in, %in_6 : f32
                linalg.yield %12 : f32
              }
              scf.yield %arg14 : memref<32x32xf32>
            }
            %9 = affine.apply #map12(%arg9, %arg11, %arg10, %arg12)
            %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%9], sizes: [32, 32], strides: [512, 1] {tmd.reuse = {spatial = [{depth = 2 : i64, iterator = "%arg11", mapped_to = "y", reuse_type = "no_reuse", volume = 0 : i64}, {depth = 3 : i64, iterator = "%arg12", mapped_to = "x", reuse_type = "no_reuse", volume = 0 : i64}], temporal = [{depth = 0 : i64, iterator = "%arg9", reuse_type = "no_reuse", volume = 0 : i64}, {depth = 1 : i64, iterator = "%arg10", reuse_type = "no_reuse", volume = 0 : i64}]}} : memref<*xf32> to memref<32x32xf32, strided<[512, 1], offset: ?>>
            memref.copy %8, %reinterpret_cast : memref<32x32xf32> to memref<32x32xf32, strided<[512, 1], offset: ?>>
          } {tmd.mapped_to = "x"}
        } {tmd.mapped_to = "y"}
      }
    }
    return
  }
  func.func @matmul_kernel__d1i0_d0i1_f0_f1(%arg0: memref<*xf32> {tt.divisibility = 16 : i32}, %arg1: memref<*xf32> {tt.divisibility = 16 : i32}, %arg2: memref<*xf32> {tt.divisibility = 16 : i32}, %arg3: index {tt.divisibility = 16 : i32}, %arg4: index {tt.divisibility = 16 : i32}, %arg5: index {tt.divisibility = 16 : i32}, %arg6: index, %arg7: index, %arg8: index) {
    affine.for %arg9 = 0 to #map8(%arg6, %arg7) {
      affine.for %arg10 = 0 to #map9(%arg6, %arg7) {
        affine.parallel (%arg11) = (0) to (8) {
          affine.parallel (%arg12) = (0) to (8) {
            %cst = arith.constant 0.000000e+00 : f32
            %alloc = memref.alloc() {alignment = 64 : i64} : memref<32x32xf32>
            linalg.fill ins(%cst : f32) outs(%alloc : memref<32x32xf32>)
            %7 = affine.apply #map5()[%arg5]
            %alloc_0 = memref.alloc() {alignment = 64 : i64} : memref<32x32xf32>
            memref.copy %alloc, %alloc_0 : memref<32x32xf32> to memref<32x32xf32>
            %c0 = arith.constant 0 : index
            %c1 = arith.constant 1 : index
            %8 = scf.for %arg13 = %c0 to %7 step %c1 iter_args(%arg14 = %alloc_0) -> (memref<32x32xf32>) {
              %10 = affine.apply #map10(%arg13, %arg9, %arg12)
              %reinterpret_cast_1 = memref.reinterpret_cast %arg0 to offset: [%10], sizes: [32, 32], strides: [512, 1] {tmd.reuse = {sequential = [{depth = 4 : i64, iterator = "%arg13", reuse_type = "no_reuse", volume = 0 : i64}], spatial = [{depth = 2 : i64, iterator = "%arg11", mapped_to = "x", reuse_type = "total_reuse", volume = 4096 : i64}, {depth = 3 : i64, iterator = "%arg12", mapped_to = "y", reuse_type = "no_reuse", volume = 0 : i64}], temporal = [{depth = 0 : i64, iterator = "%arg9", reuse_type = "no_reuse", volume = 0 : i64}, {depth = 1 : i64, iterator = "%arg10", reuse_type = "total_reuse", volume = 4096 : i64}]}} : memref<*xf32> to memref<32x32xf32, strided<[512, 1], offset: ?>>
              %alloc_2 = memref.alloc() : memref<32x32xf32>
              memref.copy %reinterpret_cast_1, %alloc_2 : memref<32x32xf32, strided<[512, 1], offset: ?>> to memref<32x32xf32>
              %11 = affine.apply #map11(%arg13, %arg10, %arg11)
              %reinterpret_cast_3 = memref.reinterpret_cast %arg1 to offset: [%11], sizes: [32, 32], strides: [512, 1] {tmd.reuse = {sequential = [{depth = 4 : i64, iterator = "%arg13", reuse_type = "no_reuse", volume = 0 : i64}], spatial = [{depth = 2 : i64, iterator = "%arg11", mapped_to = "x", reuse_type = "no_reuse", volume = 0 : i64}, {depth = 3 : i64, iterator = "%arg12", mapped_to = "y", reuse_type = "total_reuse", volume = 4096 : i64}], temporal = [{depth = 0 : i64, iterator = "%arg9", reuse_type = "total_reuse", volume = 4096 : i64}, {depth = 1 : i64, iterator = "%arg10", reuse_type = "no_reuse", volume = 0 : i64}]}} : memref<*xf32> to memref<32x32xf32, strided<[512, 1], offset: ?>>
              %alloc_4 = memref.alloc() : memref<32x32xf32>
              memref.copy %reinterpret_cast_3, %alloc_4 : memref<32x32xf32, strided<[512, 1], offset: ?>> to memref<32x32xf32>
              %alloc_5 = memref.alloc() {alignment = 64 : i64} : memref<32x32xf32>
              memref.copy %alloc, %alloc_5 : memref<32x32xf32> to memref<32x32xf32>
              linalg.matmul ins(%alloc_2, %alloc_4 : memref<32x32xf32>, memref<32x32xf32>) outs(%alloc_5 : memref<32x32xf32>)
              linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel", "parallel"]} ins(%arg14, %alloc_5 : memref<32x32xf32>, memref<32x32xf32>) outs(%arg14 : memref<32x32xf32>) {
              ^bb0(%in: f32, %in_6: f32, %out: f32):
                %12 = arith.addf %in, %in_6 : f32
                linalg.yield %12 : f32
              }
              scf.yield %arg14 : memref<32x32xf32>
            }
            %9 = affine.apply #map12(%arg10, %arg11, %arg9, %arg12)
            %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%9], sizes: [32, 32], strides: [512, 1] {tmd.reuse = {spatial = [{depth = 2 : i64, iterator = "%arg11", mapped_to = "x", reuse_type = "no_reuse", volume = 0 : i64}, {depth = 3 : i64, iterator = "%arg12", mapped_to = "y", reuse_type = "no_reuse", volume = 0 : i64}], temporal = [{depth = 0 : i64, iterator = "%arg9", reuse_type = "no_reuse", volume = 0 : i64}, {depth = 1 : i64, iterator = "%arg10", reuse_type = "no_reuse", volume = 0 : i64}]}} : memref<*xf32> to memref<32x32xf32, strided<[512, 1], offset: ?>>
            memref.copy %8, %reinterpret_cast : memref<32x32xf32> to memref<32x32xf32, strided<[512, 1], offset: ?>>
          } {tmd.mapped_to = "y"}
        } {tmd.mapped_to = "x"}
      }
    }
    return
  }
  func.func @matmul_kernel__d1i0_d0i1_f1_f0(%arg0: memref<*xf32> {tt.divisibility = 16 : i32}, %arg1: memref<*xf32> {tt.divisibility = 16 : i32}, %arg2: memref<*xf32> {tt.divisibility = 16 : i32}, %arg3: index {tt.divisibility = 16 : i32}, %arg4: index {tt.divisibility = 16 : i32}, %arg5: index {tt.divisibility = 16 : i32}, %arg6: index, %arg7: index, %arg8: index) {
    affine.for %arg9 = 0 to #map9(%arg6, %arg7) {
      affine.for %arg10 = 0 to #map8(%arg6, %arg7) {
        affine.parallel (%arg11) = (0) to (8) {
          affine.parallel (%arg12) = (0) to (8) {
            %cst = arith.constant 0.000000e+00 : f32
            %alloc = memref.alloc() {alignment = 64 : i64} : memref<32x32xf32>
            linalg.fill ins(%cst : f32) outs(%alloc : memref<32x32xf32>)
            %7 = affine.apply #map5()[%arg5]
            %alloc_0 = memref.alloc() {alignment = 64 : i64} : memref<32x32xf32>
            memref.copy %alloc, %alloc_0 : memref<32x32xf32> to memref<32x32xf32>
            %c0 = arith.constant 0 : index
            %c1 = arith.constant 1 : index
            %8 = scf.for %arg13 = %c0 to %7 step %c1 iter_args(%arg14 = %alloc_0) -> (memref<32x32xf32>) {
              %10 = affine.apply #map10(%arg13, %arg10, %arg12)
              %reinterpret_cast_1 = memref.reinterpret_cast %arg0 to offset: [%10], sizes: [32, 32], strides: [512, 1] {tmd.reuse = {sequential = [{depth = 4 : i64, iterator = "%arg13", reuse_type = "no_reuse", volume = 0 : i64}], spatial = [{depth = 2 : i64, iterator = "%arg11", mapped_to = "x", reuse_type = "total_reuse", volume = 4096 : i64}, {depth = 3 : i64, iterator = "%arg12", mapped_to = "y", reuse_type = "no_reuse", volume = 0 : i64}], temporal = [{depth = 0 : i64, iterator = "%arg9", reuse_type = "total_reuse", volume = 4096 : i64}, {depth = 1 : i64, iterator = "%arg10", reuse_type = "no_reuse", volume = 0 : i64}]}} : memref<*xf32> to memref<32x32xf32, strided<[512, 1], offset: ?>>
              %alloc_2 = memref.alloc() : memref<32x32xf32>
              memref.copy %reinterpret_cast_1, %alloc_2 : memref<32x32xf32, strided<[512, 1], offset: ?>> to memref<32x32xf32>
              %11 = affine.apply #map11(%arg13, %arg9, %arg11)
              %reinterpret_cast_3 = memref.reinterpret_cast %arg1 to offset: [%11], sizes: [32, 32], strides: [512, 1] {tmd.reuse = {sequential = [{depth = 4 : i64, iterator = "%arg13", reuse_type = "no_reuse", volume = 0 : i64}], spatial = [{depth = 2 : i64, iterator = "%arg11", mapped_to = "x", reuse_type = "no_reuse", volume = 0 : i64}, {depth = 3 : i64, iterator = "%arg12", mapped_to = "y", reuse_type = "total_reuse", volume = 4096 : i64}], temporal = [{depth = 0 : i64, iterator = "%arg9", reuse_type = "no_reuse", volume = 0 : i64}, {depth = 1 : i64, iterator = "%arg10", reuse_type = "total_reuse", volume = 4096 : i64}]}} : memref<*xf32> to memref<32x32xf32, strided<[512, 1], offset: ?>>
              %alloc_4 = memref.alloc() : memref<32x32xf32>
              memref.copy %reinterpret_cast_3, %alloc_4 : memref<32x32xf32, strided<[512, 1], offset: ?>> to memref<32x32xf32>
              %alloc_5 = memref.alloc() {alignment = 64 : i64} : memref<32x32xf32>
              memref.copy %alloc, %alloc_5 : memref<32x32xf32> to memref<32x32xf32>
              linalg.matmul ins(%alloc_2, %alloc_4 : memref<32x32xf32>, memref<32x32xf32>) outs(%alloc_5 : memref<32x32xf32>)
              linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel", "parallel"]} ins(%arg14, %alloc_5 : memref<32x32xf32>, memref<32x32xf32>) outs(%arg14 : memref<32x32xf32>) {
              ^bb0(%in: f32, %in_6: f32, %out: f32):
                %12 = arith.addf %in, %in_6 : f32
                linalg.yield %12 : f32
              }
              scf.yield %arg14 : memref<32x32xf32>
            }
            %9 = affine.apply #map12(%arg9, %arg11, %arg10, %arg12)
            %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%9], sizes: [32, 32], strides: [512, 1] {tmd.reuse = {spatial = [{depth = 2 : i64, iterator = "%arg11", mapped_to = "x", reuse_type = "no_reuse", volume = 0 : i64}, {depth = 3 : i64, iterator = "%arg12", mapped_to = "y", reuse_type = "no_reuse", volume = 0 : i64}], temporal = [{depth = 0 : i64, iterator = "%arg9", reuse_type = "no_reuse", volume = 0 : i64}, {depth = 1 : i64, iterator = "%arg10", reuse_type = "no_reuse", volume = 0 : i64}]}} : memref<*xf32> to memref<32x32xf32, strided<[512, 1], offset: ?>>
            memref.copy %8, %reinterpret_cast : memref<32x32xf32> to memref<32x32xf32, strided<[512, 1], offset: ?>>
          } {tmd.mapped_to = "y"}
        } {tmd.mapped_to = "x"}
      }
    }
    return
  }
  func.func @matmul_kernel__d0i1_d1i1_f0_f1(%arg0: memref<*xf32> {tt.divisibility = 16 : i32}, %arg1: memref<*xf32> {tt.divisibility = 16 : i32}, %arg2: memref<*xf32> {tt.divisibility = 16 : i32}, %arg3: index {tt.divisibility = 16 : i32}, %arg4: index {tt.divisibility = 16 : i32}, %arg5: index {tt.divisibility = 16 : i32}, %arg6: index, %arg7: index, %arg8: index) {
    affine.for %arg9 = 0 to #map13(%arg6, %arg7) {
      affine.for %arg10 = 0 to #map14(%arg6, %arg7) {
        affine.parallel (%arg11) = (0) to (8) {
          affine.parallel (%arg12) = (0) to (8) {
            %cst = arith.constant 0.000000e+00 : f32
            %alloc = memref.alloc() {alignment = 64 : i64} : memref<32x32xf32>
            linalg.fill ins(%cst : f32) outs(%alloc : memref<32x32xf32>)
            %7 = affine.apply #map5()[%arg5]
            %alloc_0 = memref.alloc() {alignment = 64 : i64} : memref<32x32xf32>
            memref.copy %alloc, %alloc_0 : memref<32x32xf32> to memref<32x32xf32>
            %c0 = arith.constant 0 : index
            %c1 = arith.constant 1 : index
            %8 = scf.for %arg13 = %c0 to %7 step %c1 iter_args(%arg14 = %alloc_0) -> (memref<32x32xf32>) {
              %10 = affine.apply #map7(%arg13, %arg9)
              %reinterpret_cast_1 = memref.reinterpret_cast %arg0 to offset: [%10], sizes: [32, 32], strides: [512, 1] {tmd.reuse = {sequential = [{depth = 4 : i64, iterator = "%arg13", reuse_type = "no_reuse", volume = 0 : i64}], spatial = [{depth = 2 : i64, iterator = "%arg11", mapped_to = "y", reuse_type = "total_reuse", volume = 4096 : i64}, {depth = 3 : i64, iterator = "%arg12", mapped_to = "x", reuse_type = "total_reuse", volume = 4096 : i64}], temporal = [{depth = 0 : i64, iterator = "%arg9", reuse_type = "no_reuse", volume = 0 : i64}, {depth = 1 : i64, iterator = "%arg10", reuse_type = "total_reuse", volume = 4096 : i64}]}} : memref<*xf32> to memref<32x32xf32, strided<[512, 1], offset: ?>>
              %alloc_2 = memref.alloc() : memref<32x32xf32>
              memref.copy %reinterpret_cast_1, %alloc_2 : memref<32x32xf32, strided<[512, 1], offset: ?>> to memref<32x32xf32>
              %11 = affine.apply #map15(%arg13, %arg12, %arg10, %arg11)
              %reinterpret_cast_3 = memref.reinterpret_cast %arg1 to offset: [%11], sizes: [32, 32], strides: [512, 1] {tmd.reuse = {sequential = [{depth = 4 : i64, iterator = "%arg13", reuse_type = "no_reuse", volume = 0 : i64}], spatial = [{depth = 2 : i64, iterator = "%arg11", mapped_to = "y", reuse_type = "no_reuse", volume = 0 : i64}, {depth = 3 : i64, iterator = "%arg12", mapped_to = "x", reuse_type = "no_reuse", volume = 0 : i64}], temporal = [{depth = 0 : i64, iterator = "%arg9", reuse_type = "total_reuse", volume = 4096 : i64}, {depth = 1 : i64, iterator = "%arg10", reuse_type = "no_reuse", volume = 0 : i64}]}} : memref<*xf32> to memref<32x32xf32, strided<[512, 1], offset: ?>>
              %alloc_4 = memref.alloc() : memref<32x32xf32>
              memref.copy %reinterpret_cast_3, %alloc_4 : memref<32x32xf32, strided<[512, 1], offset: ?>> to memref<32x32xf32>
              %alloc_5 = memref.alloc() {alignment = 64 : i64} : memref<32x32xf32>
              memref.copy %alloc, %alloc_5 : memref<32x32xf32> to memref<32x32xf32>
              linalg.matmul ins(%alloc_2, %alloc_4 : memref<32x32xf32>, memref<32x32xf32>) outs(%alloc_5 : memref<32x32xf32>)
              linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel", "parallel"]} ins(%arg14, %alloc_5 : memref<32x32xf32>, memref<32x32xf32>) outs(%arg14 : memref<32x32xf32>) {
              ^bb0(%in: f32, %in_6: f32, %out: f32):
                %12 = arith.addf %in, %in_6 : f32
                linalg.yield %12 : f32
              }
              scf.yield %arg14 : memref<32x32xf32>
            }
            %9 = affine.apply #map15(%arg9, %arg12, %arg10, %arg11)
            %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%9], sizes: [32, 32], strides: [512, 1] {tmd.reuse = {spatial = [{depth = 2 : i64, iterator = "%arg11", mapped_to = "y", reuse_type = "no_reuse", volume = 0 : i64}, {depth = 3 : i64, iterator = "%arg12", mapped_to = "x", reuse_type = "no_reuse", volume = 0 : i64}], temporal = [{depth = 0 : i64, iterator = "%arg9", reuse_type = "no_reuse", volume = 0 : i64}, {depth = 1 : i64, iterator = "%arg10", reuse_type = "no_reuse", volume = 0 : i64}]}} : memref<*xf32> to memref<32x32xf32, strided<[512, 1], offset: ?>>
            memref.copy %8, %reinterpret_cast : memref<32x32xf32> to memref<32x32xf32, strided<[512, 1], offset: ?>>
          } {tmd.mapped_to = "x"}
        } {tmd.mapped_to = "y"}
      }
    }
    return
  }
  func.func @matmul_kernel__d0i1_d1i1_f1_f0(%arg0: memref<*xf32> {tt.divisibility = 16 : i32}, %arg1: memref<*xf32> {tt.divisibility = 16 : i32}, %arg2: memref<*xf32> {tt.divisibility = 16 : i32}, %arg3: index {tt.divisibility = 16 : i32}, %arg4: index {tt.divisibility = 16 : i32}, %arg5: index {tt.divisibility = 16 : i32}, %arg6: index, %arg7: index, %arg8: index) {
    affine.for %arg9 = 0 to #map14(%arg6, %arg7) {
      affine.for %arg10 = 0 to #map13(%arg6, %arg7) {
        affine.parallel (%arg11) = (0) to (8) {
          affine.parallel (%arg12) = (0) to (8) {
            %cst = arith.constant 0.000000e+00 : f32
            %alloc = memref.alloc() {alignment = 64 : i64} : memref<32x32xf32>
            linalg.fill ins(%cst : f32) outs(%alloc : memref<32x32xf32>)
            %7 = affine.apply #map5()[%arg5]
            %alloc_0 = memref.alloc() {alignment = 64 : i64} : memref<32x32xf32>
            memref.copy %alloc, %alloc_0 : memref<32x32xf32> to memref<32x32xf32>
            %c0 = arith.constant 0 : index
            %c1 = arith.constant 1 : index
            %8 = scf.for %arg13 = %c0 to %7 step %c1 iter_args(%arg14 = %alloc_0) -> (memref<32x32xf32>) {
              %10 = affine.apply #map7(%arg13, %arg10)
              %reinterpret_cast_1 = memref.reinterpret_cast %arg0 to offset: [%10], sizes: [32, 32], strides: [512, 1] {tmd.reuse = {sequential = [{depth = 4 : i64, iterator = "%arg13", reuse_type = "no_reuse", volume = 0 : i64}], spatial = [{depth = 2 : i64, iterator = "%arg11", mapped_to = "y", reuse_type = "total_reuse", volume = 4096 : i64}, {depth = 3 : i64, iterator = "%arg12", mapped_to = "x", reuse_type = "total_reuse", volume = 4096 : i64}], temporal = [{depth = 0 : i64, iterator = "%arg9", reuse_type = "total_reuse", volume = 4096 : i64}, {depth = 1 : i64, iterator = "%arg10", reuse_type = "no_reuse", volume = 0 : i64}]}} : memref<*xf32> to memref<32x32xf32, strided<[512, 1], offset: ?>>
              %alloc_2 = memref.alloc() : memref<32x32xf32>
              memref.copy %reinterpret_cast_1, %alloc_2 : memref<32x32xf32, strided<[512, 1], offset: ?>> to memref<32x32xf32>
              %11 = affine.apply #map15(%arg13, %arg12, %arg9, %arg11)
              %reinterpret_cast_3 = memref.reinterpret_cast %arg1 to offset: [%11], sizes: [32, 32], strides: [512, 1] {tmd.reuse = {sequential = [{depth = 4 : i64, iterator = "%arg13", reuse_type = "no_reuse", volume = 0 : i64}], spatial = [{depth = 2 : i64, iterator = "%arg11", mapped_to = "y", reuse_type = "no_reuse", volume = 0 : i64}, {depth = 3 : i64, iterator = "%arg12", mapped_to = "x", reuse_type = "no_reuse", volume = 0 : i64}], temporal = [{depth = 0 : i64, iterator = "%arg9", reuse_type = "no_reuse", volume = 0 : i64}, {depth = 1 : i64, iterator = "%arg10", reuse_type = "total_reuse", volume = 4096 : i64}]}} : memref<*xf32> to memref<32x32xf32, strided<[512, 1], offset: ?>>
              %alloc_4 = memref.alloc() : memref<32x32xf32>
              memref.copy %reinterpret_cast_3, %alloc_4 : memref<32x32xf32, strided<[512, 1], offset: ?>> to memref<32x32xf32>
              %alloc_5 = memref.alloc() {alignment = 64 : i64} : memref<32x32xf32>
              memref.copy %alloc, %alloc_5 : memref<32x32xf32> to memref<32x32xf32>
              linalg.matmul ins(%alloc_2, %alloc_4 : memref<32x32xf32>, memref<32x32xf32>) outs(%alloc_5 : memref<32x32xf32>)
              linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel", "parallel"]} ins(%arg14, %alloc_5 : memref<32x32xf32>, memref<32x32xf32>) outs(%arg14 : memref<32x32xf32>) {
              ^bb0(%in: f32, %in_6: f32, %out: f32):
                %12 = arith.addf %in, %in_6 : f32
                linalg.yield %12 : f32
              }
              scf.yield %arg14 : memref<32x32xf32>
            }
            %9 = affine.apply #map15(%arg10, %arg12, %arg9, %arg11)
            %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%9], sizes: [32, 32], strides: [512, 1] {tmd.reuse = {spatial = [{depth = 2 : i64, iterator = "%arg11", mapped_to = "y", reuse_type = "no_reuse", volume = 0 : i64}, {depth = 3 : i64, iterator = "%arg12", mapped_to = "x", reuse_type = "no_reuse", volume = 0 : i64}], temporal = [{depth = 0 : i64, iterator = "%arg9", reuse_type = "no_reuse", volume = 0 : i64}, {depth = 1 : i64, iterator = "%arg10", reuse_type = "no_reuse", volume = 0 : i64}]}} : memref<*xf32> to memref<32x32xf32, strided<[512, 1], offset: ?>>
            memref.copy %8, %reinterpret_cast : memref<32x32xf32> to memref<32x32xf32, strided<[512, 1], offset: ?>>
          } {tmd.mapped_to = "x"}
        } {tmd.mapped_to = "y"}
      }
    }
    return
  }
  func.func @matmul_kernel__d1i1_d0i1_f0_f1(%arg0: memref<*xf32> {tt.divisibility = 16 : i32}, %arg1: memref<*xf32> {tt.divisibility = 16 : i32}, %arg2: memref<*xf32> {tt.divisibility = 16 : i32}, %arg3: index {tt.divisibility = 16 : i32}, %arg4: index {tt.divisibility = 16 : i32}, %arg5: index {tt.divisibility = 16 : i32}, %arg6: index, %arg7: index, %arg8: index) {
    affine.for %arg9 = 0 to #map13(%arg6, %arg7) {
      affine.for %arg10 = 0 to #map14(%arg6, %arg7) {
        affine.parallel (%arg11) = (0) to (8) {
          affine.parallel (%arg12) = (0) to (8) {
            %cst = arith.constant 0.000000e+00 : f32
            %alloc = memref.alloc() {alignment = 64 : i64} : memref<32x32xf32>
            linalg.fill ins(%cst : f32) outs(%alloc : memref<32x32xf32>)
            %7 = affine.apply #map5()[%arg5]
            %alloc_0 = memref.alloc() {alignment = 64 : i64} : memref<32x32xf32>
            memref.copy %alloc, %alloc_0 : memref<32x32xf32> to memref<32x32xf32>
            %c0 = arith.constant 0 : index
            %c1 = arith.constant 1 : index
            %8 = scf.for %arg13 = %c0 to %7 step %c1 iter_args(%arg14 = %alloc_0) -> (memref<32x32xf32>) {
              %10 = affine.apply #map7(%arg13, %arg9)
              %reinterpret_cast_1 = memref.reinterpret_cast %arg0 to offset: [%10], sizes: [32, 32], strides: [512, 1] {tmd.reuse = {sequential = [{depth = 4 : i64, iterator = "%arg13", reuse_type = "no_reuse", volume = 0 : i64}], spatial = [{depth = 2 : i64, iterator = "%arg11", mapped_to = "x", reuse_type = "total_reuse", volume = 4096 : i64}, {depth = 3 : i64, iterator = "%arg12", mapped_to = "y", reuse_type = "total_reuse", volume = 4096 : i64}], temporal = [{depth = 0 : i64, iterator = "%arg9", reuse_type = "no_reuse", volume = 0 : i64}, {depth = 1 : i64, iterator = "%arg10", reuse_type = "total_reuse", volume = 4096 : i64}]}} : memref<*xf32> to memref<32x32xf32, strided<[512, 1], offset: ?>>
              %alloc_2 = memref.alloc() : memref<32x32xf32>
              memref.copy %reinterpret_cast_1, %alloc_2 : memref<32x32xf32, strided<[512, 1], offset: ?>> to memref<32x32xf32>
              %11 = affine.apply #map15(%arg13, %arg12, %arg10, %arg11)
              %reinterpret_cast_3 = memref.reinterpret_cast %arg1 to offset: [%11], sizes: [32, 32], strides: [512, 1] {tmd.reuse = {sequential = [{depth = 4 : i64, iterator = "%arg13", reuse_type = "no_reuse", volume = 0 : i64}], spatial = [{depth = 2 : i64, iterator = "%arg11", mapped_to = "x", reuse_type = "no_reuse", volume = 0 : i64}, {depth = 3 : i64, iterator = "%arg12", mapped_to = "y", reuse_type = "no_reuse", volume = 0 : i64}], temporal = [{depth = 0 : i64, iterator = "%arg9", reuse_type = "total_reuse", volume = 4096 : i64}, {depth = 1 : i64, iterator = "%arg10", reuse_type = "no_reuse", volume = 0 : i64}]}} : memref<*xf32> to memref<32x32xf32, strided<[512, 1], offset: ?>>
              %alloc_4 = memref.alloc() : memref<32x32xf32>
              memref.copy %reinterpret_cast_3, %alloc_4 : memref<32x32xf32, strided<[512, 1], offset: ?>> to memref<32x32xf32>
              %alloc_5 = memref.alloc() {alignment = 64 : i64} : memref<32x32xf32>
              memref.copy %alloc, %alloc_5 : memref<32x32xf32> to memref<32x32xf32>
              linalg.matmul ins(%alloc_2, %alloc_4 : memref<32x32xf32>, memref<32x32xf32>) outs(%alloc_5 : memref<32x32xf32>)
              linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel", "parallel"]} ins(%arg14, %alloc_5 : memref<32x32xf32>, memref<32x32xf32>) outs(%arg14 : memref<32x32xf32>) {
              ^bb0(%in: f32, %in_6: f32, %out: f32):
                %12 = arith.addf %in, %in_6 : f32
                linalg.yield %12 : f32
              }
              scf.yield %arg14 : memref<32x32xf32>
            }
            %9 = affine.apply #map15(%arg9, %arg12, %arg10, %arg11)
            %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%9], sizes: [32, 32], strides: [512, 1] {tmd.reuse = {spatial = [{depth = 2 : i64, iterator = "%arg11", mapped_to = "x", reuse_type = "no_reuse", volume = 0 : i64}, {depth = 3 : i64, iterator = "%arg12", mapped_to = "y", reuse_type = "no_reuse", volume = 0 : i64}], temporal = [{depth = 0 : i64, iterator = "%arg9", reuse_type = "no_reuse", volume = 0 : i64}, {depth = 1 : i64, iterator = "%arg10", reuse_type = "no_reuse", volume = 0 : i64}]}} : memref<*xf32> to memref<32x32xf32, strided<[512, 1], offset: ?>>
            memref.copy %8, %reinterpret_cast : memref<32x32xf32> to memref<32x32xf32, strided<[512, 1], offset: ?>>
          } {tmd.mapped_to = "y"}
        } {tmd.mapped_to = "x"}
      }
    }
    return
  }
  func.func @matmul_kernel__d1i1_d0i1_f1_f0(%arg0: memref<*xf32> {tt.divisibility = 16 : i32}, %arg1: memref<*xf32> {tt.divisibility = 16 : i32}, %arg2: memref<*xf32> {tt.divisibility = 16 : i32}, %arg3: index {tt.divisibility = 16 : i32}, %arg4: index {tt.divisibility = 16 : i32}, %arg5: index {tt.divisibility = 16 : i32}, %arg6: index, %arg7: index, %arg8: index) {
    affine.for %arg9 = 0 to #map14(%arg6, %arg7) {
      affine.for %arg10 = 0 to #map13(%arg6, %arg7) {
        affine.parallel (%arg11) = (0) to (8) {
          affine.parallel (%arg12) = (0) to (8) {
            %cst = arith.constant 0.000000e+00 : f32
            %alloc = memref.alloc() {alignment = 64 : i64} : memref<32x32xf32>
            linalg.fill ins(%cst : f32) outs(%alloc : memref<32x32xf32>)
            %7 = affine.apply #map5()[%arg5]
            %alloc_0 = memref.alloc() {alignment = 64 : i64} : memref<32x32xf32>
            memref.copy %alloc, %alloc_0 : memref<32x32xf32> to memref<32x32xf32>
            %c0 = arith.constant 0 : index
            %c1 = arith.constant 1 : index
            %8 = scf.for %arg13 = %c0 to %7 step %c1 iter_args(%arg14 = %alloc_0) -> (memref<32x32xf32>) {
              %10 = affine.apply #map7(%arg13, %arg10)
              %reinterpret_cast_1 = memref.reinterpret_cast %arg0 to offset: [%10], sizes: [32, 32], strides: [512, 1] {tmd.reuse = {sequential = [{depth = 4 : i64, iterator = "%arg13", reuse_type = "no_reuse", volume = 0 : i64}], spatial = [{depth = 2 : i64, iterator = "%arg11", mapped_to = "x", reuse_type = "total_reuse", volume = 4096 : i64}, {depth = 3 : i64, iterator = "%arg12", mapped_to = "y", reuse_type = "total_reuse", volume = 4096 : i64}], temporal = [{depth = 0 : i64, iterator = "%arg9", reuse_type = "total_reuse", volume = 4096 : i64}, {depth = 1 : i64, iterator = "%arg10", reuse_type = "no_reuse", volume = 0 : i64}]}} : memref<*xf32> to memref<32x32xf32, strided<[512, 1], offset: ?>>
              %alloc_2 = memref.alloc() : memref<32x32xf32>
              memref.copy %reinterpret_cast_1, %alloc_2 : memref<32x32xf32, strided<[512, 1], offset: ?>> to memref<32x32xf32>
              %11 = affine.apply #map15(%arg13, %arg12, %arg9, %arg11)
              %reinterpret_cast_3 = memref.reinterpret_cast %arg1 to offset: [%11], sizes: [32, 32], strides: [512, 1] {tmd.reuse = {sequential = [{depth = 4 : i64, iterator = "%arg13", reuse_type = "no_reuse", volume = 0 : i64}], spatial = [{depth = 2 : i64, iterator = "%arg11", mapped_to = "x", reuse_type = "no_reuse", volume = 0 : i64}, {depth = 3 : i64, iterator = "%arg12", mapped_to = "y", reuse_type = "no_reuse", volume = 0 : i64}], temporal = [{depth = 0 : i64, iterator = "%arg9", reuse_type = "no_reuse", volume = 0 : i64}, {depth = 1 : i64, iterator = "%arg10", reuse_type = "total_reuse", volume = 4096 : i64}]}} : memref<*xf32> to memref<32x32xf32, strided<[512, 1], offset: ?>>
              %alloc_4 = memref.alloc() : memref<32x32xf32>
              memref.copy %reinterpret_cast_3, %alloc_4 : memref<32x32xf32, strided<[512, 1], offset: ?>> to memref<32x32xf32>
              %alloc_5 = memref.alloc() {alignment = 64 : i64} : memref<32x32xf32>
              memref.copy %alloc, %alloc_5 : memref<32x32xf32> to memref<32x32xf32>
              linalg.matmul ins(%alloc_2, %alloc_4 : memref<32x32xf32>, memref<32x32xf32>) outs(%alloc_5 : memref<32x32xf32>)
              linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel", "parallel"]} ins(%arg14, %alloc_5 : memref<32x32xf32>, memref<32x32xf32>) outs(%arg14 : memref<32x32xf32>) {
              ^bb0(%in: f32, %in_6: f32, %out: f32):
                %12 = arith.addf %in, %in_6 : f32
                linalg.yield %12 : f32
              }
              scf.yield %arg14 : memref<32x32xf32>
            }
            %9 = affine.apply #map15(%arg10, %arg12, %arg9, %arg11)
            %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%9], sizes: [32, 32], strides: [512, 1] {tmd.reuse = {spatial = [{depth = 2 : i64, iterator = "%arg11", mapped_to = "x", reuse_type = "no_reuse", volume = 0 : i64}, {depth = 3 : i64, iterator = "%arg12", mapped_to = "y", reuse_type = "no_reuse", volume = 0 : i64}], temporal = [{depth = 0 : i64, iterator = "%arg9", reuse_type = "no_reuse", volume = 0 : i64}, {depth = 1 : i64, iterator = "%arg10", reuse_type = "no_reuse", volume = 0 : i64}]}} : memref<*xf32> to memref<32x32xf32, strided<[512, 1], offset: ?>>
            memref.copy %8, %reinterpret_cast : memref<32x32xf32> to memref<32x32xf32, strided<[512, 1], offset: ?>>
          } {tmd.mapped_to = "y"}
        } {tmd.mapped_to = "x"}
      }
    }
    return
  }
}

