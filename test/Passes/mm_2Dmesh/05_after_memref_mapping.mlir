module {
  %0 = df.mat "FPU" {shape = [32, 32, 32]}
  %1 = df.vec "SFPU" {shape = [32]}
  %2 = df.spatial_dim "x", 8
  %3 = df.spatial_dim "y", 8
  %4 = df.core "core" {scaleout=(%2, %3) , scalein=(%0, %1, [8, 1])}
  %5 = df.memory "L1" {scaleout=(%2, %3) , size = 32768, bandwidth = 64}
  %6 = df.mux %4 : !df.compute, %5 : !df.memory  {map = affine_map<(d0, d1) -> (d0, d1)>}
  %7 = df.interconnects "horizontal_links" %5 : !df.memory, %5 : !df.memory  {bandwidth = 128 : i64, map = affine_map<(d0, d1) -> ((d0 + 1) mod 8, d1)>} : !df.interconnect
  %8 = df.interconnects "vertical_links" %5 : !df.memory, %5 : !df.memory  {bandwidth = 128 : i64, map = affine_map<(d0, d1) -> (d0, (d1 + 1) mod 8)>} : !df.interconnect
  %9 = df.spatial_dim "d", 4
  %10 = df.memory "DRAM" {scaleout=(%9) , size = 34359738368, bandwidth = 512}
  %11 = df.interconnects %5 : !df.memory, %10 : !df.memory  {map = affine_map<(d0, d1) -> (d0 ceildiv 4 + (d1 ceildiv 4) * 2)>} : !df.interconnect
  func.func @matmul_kernel__d0i0_d1i0__f01__c0mem_c1mem(%arg0: memref<*xf32> {tt.divisibility = 16 : i32}, %arg1: memref<*xf32> {tt.divisibility = 16 : i32}, %arg2: memref<*xf32> {tt.divisibility = 16 : i32}, %arg3: index, %arg4: index, %arg5: index) {
    affine.parallel (%arg6) = (0) to (8) {
      affine.parallel (%arg7) = (0) to (8) {
        affine.for %arg8 = 0 to affine_map<()[s0, s1] -> ((s0 ceildiv 8) ceildiv 8)>()[%arg3, %arg4] {
          affine.for %arg9 = 0 to affine_map<()[s0, s1] -> (s1)>()[%arg3, %arg4] {
            %cst = arith.constant 0.000000e+00 : f32
            %12 = tensor.empty() : tensor<64x64xf32>
            %13 = linalg.fill ins(%cst : f32) outs(%12 : tensor<64x64xf32>) -> tensor<64x64xf32>
            %c0 = arith.constant 0 : index
            %c8 = arith.constant 8 : index
            %c1 = arith.constant 1 : index
            %14 = scf.for %arg10 = %c0 to %c8 step %c1 iter_args(%arg11 = %13) -> (tensor<64x64xf32>) {
              %16 = affine.apply affine_map<(d0, d1, d2, d3) -> (d0 * 64 + d1 * 262144 + d2 * 262144 + d3 * 32768)>(%arg10, %arg6, %arg7, %arg8)
              %reinterpret_cast_0 = memref.reinterpret_cast %arg0 to offset: [%16], sizes: [64, 64], strides: [512, 1] {tmd.reuse = {sequential = [{depth = 4 : i64, iterator = "%arg10", reuse_type = "no_reuse", volume = 0 : i64}], spatial = [{depth = 0 : i64, iterator = "%arg6", mapped_to = "x", reuse_type = "no_reuse", volume = 0 : i64}, {depth = 1 : i64, iterator = "%arg7", mapped_to = "y", reuse_type = "no_reuse", volume = 0 : i64}], temporal = [{depth = 2 : i64, iterator = "%arg8", reuse_type = "no_reuse", volume = 0 : i64}, {depth = 3 : i64, iterator = "%arg9", reuse_type = "total_reuse", volume = 16384 : i64}]}} : memref<*xf32> to memref<64x64xf32, strided<[512, 1], offset: ?>>
              %alloc = memref.alloc() {tmd.alloc = {local = true, memory_name = "L1", size = 16384 : i64}} : memref<64x64xf32>
              memref.copy %reinterpret_cast_0, %alloc {tmd.copy.choice = {kind = "mem", memory_name = "L1"}} : memref<64x64xf32, strided<[512, 1], offset: ?>> to memref<64x64xf32>
              %17 = bufferization.to_tensor %alloc restrict writable : memref<64x64xf32> to tensor<64x64xf32>
              %18 = affine.apply affine_map<(d0, d1) -> (d1 * 32768 + d0 * 64)>(%arg9, %arg10)
              %reinterpret_cast_1 = memref.reinterpret_cast %arg1 to offset: [%18], sizes: [64, 64], strides: [512, 1] {tmd.reuse = {sequential = [{depth = 4 : i64, iterator = "%arg10", reuse_type = "no_reuse", volume = 0 : i64}], spatial = [{depth = 0 : i64, iterator = "%arg6", mapped_to = "x", reuse_type = "total_reuse", volume = 16384 : i64}, {depth = 1 : i64, iterator = "%arg7", mapped_to = "y", reuse_type = "total_reuse", volume = 16384 : i64}], temporal = [{depth = 2 : i64, iterator = "%arg8", reuse_type = "total_reuse", volume = 16384 : i64}, {depth = 3 : i64, iterator = "%arg9", reuse_type = "no_reuse", volume = 0 : i64}]}} : memref<*xf32> to memref<64x64xf32, strided<[512, 1], offset: ?>>
              %alloc_2 = memref.alloc() {tmd.alloc = {local = true, memory_name = "L1", size = 16384 : i64}} : memref<64x64xf32>
              memref.copy %reinterpret_cast_1, %alloc_2 {tmd.copy.choice = {kind = "mem", memory_name = "L1"}} : memref<64x64xf32, strided<[512, 1], offset: ?>> to memref<64x64xf32>
              %19 = bufferization.to_tensor %alloc_2 restrict writable : memref<64x64xf32> to tensor<64x64xf32>
              %20 = linalg.matmul ins(%17, %19 : tensor<64x64xf32>, tensor<64x64xf32>) outs(%13 : tensor<64x64xf32>) -> tensor<64x64xf32>
              %21 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%arg11, %20 : tensor<64x64xf32>, tensor<64x64xf32>) outs(%arg11 : tensor<64x64xf32>) {
              ^bb0(%in: f32, %in_3: f32, %out: f32):
                %22 = arith.addf %in, %in_3 : f32
                linalg.yield %22 : f32
              } -> tensor<64x64xf32>
              scf.yield %21 : tensor<64x64xf32>
            }
            %15 = affine.apply affine_map<(d0, d1, d2, d3) -> (d0 * 64 + d1 * 262144 + d2 * 262144 + d3 * 32768)>(%arg9, %arg6, %arg7, %arg8)
            %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%15], sizes: [64, 64], strides: [512, 1] {tmd.reuse = {spatial = [{depth = 0 : i64, iterator = "%arg6", mapped_to = "x", reuse_type = "no_reuse", volume = 0 : i64}, {depth = 1 : i64, iterator = "%arg7", mapped_to = "y", reuse_type = "no_reuse", volume = 0 : i64}], temporal = [{depth = 2 : i64, iterator = "%arg8", reuse_type = "no_reuse", volume = 0 : i64}, {depth = 3 : i64, iterator = "%arg9", reuse_type = "no_reuse", volume = 0 : i64}]}} : memref<*xf32> to memref<64x64xf32, strided<[512, 1], offset: ?>>
            bufferization.materialize_in_destination %14 in writable %reinterpret_cast : (tensor<64x64xf32>, memref<64x64xf32, strided<[512, 1], offset: ?>>) -> ()
          }
        }
      } {tmd.mapped_to = "y"}
    } {tmd.mapped_to = "x"}
    return
  }
  func.func @matmul_kernel__d0i0_d1i0__f01__c0mem_c1bx(%arg0: memref<*xf32> {tt.divisibility = 16 : i32}, %arg1: memref<*xf32> {tt.divisibility = 16 : i32}, %arg2: memref<*xf32> {tt.divisibility = 16 : i32}, %arg3: index, %arg4: index, %arg5: index) {
    affine.parallel (%arg6) = (0) to (8) {
      affine.parallel (%arg7) = (0) to (8) {
        affine.for %arg8 = 0 to affine_map<()[s0, s1] -> ((s0 ceildiv 8) ceildiv 8)>()[%arg3, %arg4] {
          affine.for %arg9 = 0 to affine_map<()[s0, s1] -> (s1)>()[%arg3, %arg4] {
            %cst = arith.constant 0.000000e+00 : f32
            %12 = tensor.empty() : tensor<64x64xf32>
            %13 = linalg.fill ins(%cst : f32) outs(%12 : tensor<64x64xf32>) -> tensor<64x64xf32>
            %c0 = arith.constant 0 : index
            %c8 = arith.constant 8 : index
            %c1 = arith.constant 1 : index
            %14 = scf.for %arg10 = %c0 to %c8 step %c1 iter_args(%arg11 = %13) -> (tensor<64x64xf32>) {
              %16 = affine.apply affine_map<(d0, d1, d2, d3) -> (d0 * 64 + d1 * 262144 + d2 * 262144 + d3 * 32768)>(%arg10, %arg6, %arg7, %arg8)
              %reinterpret_cast_0 = memref.reinterpret_cast %arg0 to offset: [%16], sizes: [64, 64], strides: [512, 1] {tmd.reuse = {sequential = [{depth = 4 : i64, iterator = "%arg10", reuse_type = "no_reuse", volume = 0 : i64}], spatial = [{depth = 0 : i64, iterator = "%arg6", mapped_to = "x", reuse_type = "no_reuse", volume = 0 : i64}, {depth = 1 : i64, iterator = "%arg7", mapped_to = "y", reuse_type = "no_reuse", volume = 0 : i64}], temporal = [{depth = 2 : i64, iterator = "%arg8", reuse_type = "no_reuse", volume = 0 : i64}, {depth = 3 : i64, iterator = "%arg9", reuse_type = "total_reuse", volume = 16384 : i64}]}} : memref<*xf32> to memref<64x64xf32, strided<[512, 1], offset: ?>>
              %alloc = memref.alloc() {tmd.alloc = {local = true, memory_name = "L1", size = 16384 : i64}} : memref<64x64xf32>
              memref.copy %reinterpret_cast_0, %alloc {tmd.copy.choice = {kind = "mem", memory_name = "L1"}} : memref<64x64xf32, strided<[512, 1], offset: ?>> to memref<64x64xf32>
              %17 = bufferization.to_tensor %alloc restrict writable : memref<64x64xf32> to tensor<64x64xf32>
              %18 = affine.apply affine_map<(d0, d1) -> (d1 * 32768 + d0 * 64)>(%arg9, %arg10)
              %reinterpret_cast_1 = memref.reinterpret_cast %arg1 to offset: [%18], sizes: [64, 64], strides: [512, 1] {tmd.reuse = {sequential = [{depth = 4 : i64, iterator = "%arg10", reuse_type = "no_reuse", volume = 0 : i64}], spatial = [{depth = 0 : i64, iterator = "%arg6", mapped_to = "x", reuse_type = "total_reuse", volume = 16384 : i64}, {depth = 1 : i64, iterator = "%arg7", mapped_to = "y", reuse_type = "total_reuse", volume = 16384 : i64}], temporal = [{depth = 2 : i64, iterator = "%arg8", reuse_type = "total_reuse", volume = 16384 : i64}, {depth = 3 : i64, iterator = "%arg9", reuse_type = "no_reuse", volume = 0 : i64}]}} : memref<*xf32> to memref<64x64xf32, strided<[512, 1], offset: ?>>
              %alloc_2 = memref.alloc() {tmd.alloc = {local = true, memory_name = "L1", size = 16384 : i64}} : memref<64x64xf32>
              memref.copy %reinterpret_cast_1, %alloc_2 {tmd.copy.choice = {dim = "x", interconnect_name = "horizontal_links", kind = "broadcast"}} : memref<64x64xf32, strided<[512, 1], offset: ?>> to memref<64x64xf32>
              %19 = bufferization.to_tensor %alloc_2 restrict writable : memref<64x64xf32> to tensor<64x64xf32>
              %20 = linalg.matmul ins(%17, %19 : tensor<64x64xf32>, tensor<64x64xf32>) outs(%13 : tensor<64x64xf32>) -> tensor<64x64xf32>
              %21 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%arg11, %20 : tensor<64x64xf32>, tensor<64x64xf32>) outs(%arg11 : tensor<64x64xf32>) {
              ^bb0(%in: f32, %in_3: f32, %out: f32):
                %22 = arith.addf %in, %in_3 : f32
                linalg.yield %22 : f32
              } -> tensor<64x64xf32>
              scf.yield %21 : tensor<64x64xf32>
            }
            %15 = affine.apply affine_map<(d0, d1, d2, d3) -> (d0 * 64 + d1 * 262144 + d2 * 262144 + d3 * 32768)>(%arg9, %arg6, %arg7, %arg8)
            %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%15], sizes: [64, 64], strides: [512, 1] {tmd.reuse = {spatial = [{depth = 0 : i64, iterator = "%arg6", mapped_to = "x", reuse_type = "no_reuse", volume = 0 : i64}, {depth = 1 : i64, iterator = "%arg7", mapped_to = "y", reuse_type = "no_reuse", volume = 0 : i64}], temporal = [{depth = 2 : i64, iterator = "%arg8", reuse_type = "no_reuse", volume = 0 : i64}, {depth = 3 : i64, iterator = "%arg9", reuse_type = "no_reuse", volume = 0 : i64}]}} : memref<*xf32> to memref<64x64xf32, strided<[512, 1], offset: ?>>
            bufferization.materialize_in_destination %14 in writable %reinterpret_cast : (tensor<64x64xf32>, memref<64x64xf32, strided<[512, 1], offset: ?>>) -> ()
          }
        }
      } {tmd.mapped_to = "y"}
    } {tmd.mapped_to = "x"}
    return
  }
  func.func @matmul_kernel__d0i0_d1i0__f01__c0mem_c1by(%arg0: memref<*xf32> {tt.divisibility = 16 : i32}, %arg1: memref<*xf32> {tt.divisibility = 16 : i32}, %arg2: memref<*xf32> {tt.divisibility = 16 : i32}, %arg3: index, %arg4: index, %arg5: index) {
    affine.parallel (%arg6) = (0) to (8) {
      affine.parallel (%arg7) = (0) to (8) {
        affine.for %arg8 = 0 to affine_map<()[s0, s1] -> ((s0 ceildiv 8) ceildiv 8)>()[%arg3, %arg4] {
          affine.for %arg9 = 0 to affine_map<()[s0, s1] -> (s1)>()[%arg3, %arg4] {
            %cst = arith.constant 0.000000e+00 : f32
            %12 = tensor.empty() : tensor<64x64xf32>
            %13 = linalg.fill ins(%cst : f32) outs(%12 : tensor<64x64xf32>) -> tensor<64x64xf32>
            %c0 = arith.constant 0 : index
            %c8 = arith.constant 8 : index
            %c1 = arith.constant 1 : index
            %14 = scf.for %arg10 = %c0 to %c8 step %c1 iter_args(%arg11 = %13) -> (tensor<64x64xf32>) {
              %16 = affine.apply affine_map<(d0, d1, d2, d3) -> (d0 * 64 + d1 * 262144 + d2 * 262144 + d3 * 32768)>(%arg10, %arg6, %arg7, %arg8)
              %reinterpret_cast_0 = memref.reinterpret_cast %arg0 to offset: [%16], sizes: [64, 64], strides: [512, 1] {tmd.reuse = {sequential = [{depth = 4 : i64, iterator = "%arg10", reuse_type = "no_reuse", volume = 0 : i64}], spatial = [{depth = 0 : i64, iterator = "%arg6", mapped_to = "x", reuse_type = "no_reuse", volume = 0 : i64}, {depth = 1 : i64, iterator = "%arg7", mapped_to = "y", reuse_type = "no_reuse", volume = 0 : i64}], temporal = [{depth = 2 : i64, iterator = "%arg8", reuse_type = "no_reuse", volume = 0 : i64}, {depth = 3 : i64, iterator = "%arg9", reuse_type = "total_reuse", volume = 16384 : i64}]}} : memref<*xf32> to memref<64x64xf32, strided<[512, 1], offset: ?>>
              %alloc = memref.alloc() {tmd.alloc = {local = true, memory_name = "L1", size = 16384 : i64}} : memref<64x64xf32>
              memref.copy %reinterpret_cast_0, %alloc {tmd.copy.choice = {kind = "mem", memory_name = "L1"}} : memref<64x64xf32, strided<[512, 1], offset: ?>> to memref<64x64xf32>
              %17 = bufferization.to_tensor %alloc restrict writable : memref<64x64xf32> to tensor<64x64xf32>
              %18 = affine.apply affine_map<(d0, d1) -> (d1 * 32768 + d0 * 64)>(%arg9, %arg10)
              %reinterpret_cast_1 = memref.reinterpret_cast %arg1 to offset: [%18], sizes: [64, 64], strides: [512, 1] {tmd.reuse = {sequential = [{depth = 4 : i64, iterator = "%arg10", reuse_type = "no_reuse", volume = 0 : i64}], spatial = [{depth = 0 : i64, iterator = "%arg6", mapped_to = "x", reuse_type = "total_reuse", volume = 16384 : i64}, {depth = 1 : i64, iterator = "%arg7", mapped_to = "y", reuse_type = "total_reuse", volume = 16384 : i64}], temporal = [{depth = 2 : i64, iterator = "%arg8", reuse_type = "total_reuse", volume = 16384 : i64}, {depth = 3 : i64, iterator = "%arg9", reuse_type = "no_reuse", volume = 0 : i64}]}} : memref<*xf32> to memref<64x64xf32, strided<[512, 1], offset: ?>>
              %alloc_2 = memref.alloc() {tmd.alloc = {local = true, memory_name = "L1", size = 16384 : i64}} : memref<64x64xf32>
              memref.copy %reinterpret_cast_1, %alloc_2 {tmd.copy.choice = {dim = "y", interconnect_name = "vertical_links", kind = "broadcast"}} : memref<64x64xf32, strided<[512, 1], offset: ?>> to memref<64x64xf32>
              %19 = bufferization.to_tensor %alloc_2 restrict writable : memref<64x64xf32> to tensor<64x64xf32>
              %20 = linalg.matmul ins(%17, %19 : tensor<64x64xf32>, tensor<64x64xf32>) outs(%13 : tensor<64x64xf32>) -> tensor<64x64xf32>
              %21 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%arg11, %20 : tensor<64x64xf32>, tensor<64x64xf32>) outs(%arg11 : tensor<64x64xf32>) {
              ^bb0(%in: f32, %in_3: f32, %out: f32):
                %22 = arith.addf %in, %in_3 : f32
                linalg.yield %22 : f32
              } -> tensor<64x64xf32>
              scf.yield %21 : tensor<64x64xf32>
            }
            %15 = affine.apply affine_map<(d0, d1, d2, d3) -> (d0 * 64 + d1 * 262144 + d2 * 262144 + d3 * 32768)>(%arg9, %arg6, %arg7, %arg8)
            %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%15], sizes: [64, 64], strides: [512, 1] {tmd.reuse = {spatial = [{depth = 0 : i64, iterator = "%arg6", mapped_to = "x", reuse_type = "no_reuse", volume = 0 : i64}, {depth = 1 : i64, iterator = "%arg7", mapped_to = "y", reuse_type = "no_reuse", volume = 0 : i64}], temporal = [{depth = 2 : i64, iterator = "%arg8", reuse_type = "no_reuse", volume = 0 : i64}, {depth = 3 : i64, iterator = "%arg9", reuse_type = "no_reuse", volume = 0 : i64}]}} : memref<*xf32> to memref<64x64xf32, strided<[512, 1], offset: ?>>
            bufferization.materialize_in_destination %14 in writable %reinterpret_cast : (tensor<64x64xf32>, memref<64x64xf32, strided<[512, 1], offset: ?>>) -> ()
          }
        }
      } {tmd.mapped_to = "y"}
    } {tmd.mapped_to = "x"}
    return
  }
  func.func @matmul_kernel__d0i0_d1i0__f01__c0mem_c1bd(%arg0: memref<*xf32> {tt.divisibility = 16 : i32}, %arg1: memref<*xf32> {tt.divisibility = 16 : i32}, %arg2: memref<*xf32> {tt.divisibility = 16 : i32}, %arg3: index, %arg4: index, %arg5: index) {
    affine.parallel (%arg6) = (0) to (8) {
      affine.parallel (%arg7) = (0) to (8) {
        affine.for %arg8 = 0 to affine_map<()[s0, s1] -> ((s0 ceildiv 8) ceildiv 8)>()[%arg3, %arg4] {
          affine.for %arg9 = 0 to affine_map<()[s0, s1] -> (s1)>()[%arg3, %arg4] {
            %cst = arith.constant 0.000000e+00 : f32
            %12 = tensor.empty() : tensor<64x64xf32>
            %13 = linalg.fill ins(%cst : f32) outs(%12 : tensor<64x64xf32>) -> tensor<64x64xf32>
            %c0 = arith.constant 0 : index
            %c8 = arith.constant 8 : index
            %c1 = arith.constant 1 : index
            %14 = scf.for %arg10 = %c0 to %c8 step %c1 iter_args(%arg11 = %13) -> (tensor<64x64xf32>) {
              %16 = affine.apply affine_map<(d0, d1, d2, d3) -> (d0 * 64 + d1 * 262144 + d2 * 262144 + d3 * 32768)>(%arg10, %arg6, %arg7, %arg8)
              %reinterpret_cast_0 = memref.reinterpret_cast %arg0 to offset: [%16], sizes: [64, 64], strides: [512, 1] {tmd.reuse = {sequential = [{depth = 4 : i64, iterator = "%arg10", reuse_type = "no_reuse", volume = 0 : i64}], spatial = [{depth = 0 : i64, iterator = "%arg6", mapped_to = "x", reuse_type = "no_reuse", volume = 0 : i64}, {depth = 1 : i64, iterator = "%arg7", mapped_to = "y", reuse_type = "no_reuse", volume = 0 : i64}], temporal = [{depth = 2 : i64, iterator = "%arg8", reuse_type = "no_reuse", volume = 0 : i64}, {depth = 3 : i64, iterator = "%arg9", reuse_type = "total_reuse", volume = 16384 : i64}]}} : memref<*xf32> to memref<64x64xf32, strided<[512, 1], offset: ?>>
              %alloc = memref.alloc() {tmd.alloc = {local = true, memory_name = "L1", size = 16384 : i64}} : memref<64x64xf32>
              memref.copy %reinterpret_cast_0, %alloc {tmd.copy.choice = {kind = "mem", memory_name = "L1"}} : memref<64x64xf32, strided<[512, 1], offset: ?>> to memref<64x64xf32>
              %17 = bufferization.to_tensor %alloc restrict writable : memref<64x64xf32> to tensor<64x64xf32>
              %18 = affine.apply affine_map<(d0, d1) -> (d1 * 32768 + d0 * 64)>(%arg9, %arg10)
              %reinterpret_cast_1 = memref.reinterpret_cast %arg1 to offset: [%18], sizes: [64, 64], strides: [512, 1] {tmd.reuse = {sequential = [{depth = 4 : i64, iterator = "%arg10", reuse_type = "no_reuse", volume = 0 : i64}], spatial = [{depth = 0 : i64, iterator = "%arg6", mapped_to = "x", reuse_type = "total_reuse", volume = 16384 : i64}, {depth = 1 : i64, iterator = "%arg7", mapped_to = "y", reuse_type = "total_reuse", volume = 16384 : i64}], temporal = [{depth = 2 : i64, iterator = "%arg8", reuse_type = "total_reuse", volume = 16384 : i64}, {depth = 3 : i64, iterator = "%arg9", reuse_type = "no_reuse", volume = 0 : i64}]}} : memref<*xf32> to memref<64x64xf32, strided<[512, 1], offset: ?>>
              %alloc_2 = memref.alloc() {tmd.alloc = {local = true, memory_name = "L1", size = 16384 : i64}} : memref<64x64xf32>
              memref.copy %reinterpret_cast_1, %alloc_2 {tmd.copy.choice = {dim = "xy", interconnect_name = "all_links", kind = "broadcast"}} : memref<64x64xf32, strided<[512, 1], offset: ?>> to memref<64x64xf32>
              %19 = bufferization.to_tensor %alloc_2 restrict writable : memref<64x64xf32> to tensor<64x64xf32>
              %20 = linalg.matmul ins(%17, %19 : tensor<64x64xf32>, tensor<64x64xf32>) outs(%13 : tensor<64x64xf32>) -> tensor<64x64xf32>
              %21 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%arg11, %20 : tensor<64x64xf32>, tensor<64x64xf32>) outs(%arg11 : tensor<64x64xf32>) {
              ^bb0(%in: f32, %in_3: f32, %out: f32):
                %22 = arith.addf %in, %in_3 : f32
                linalg.yield %22 : f32
              } -> tensor<64x64xf32>
              scf.yield %21 : tensor<64x64xf32>
            }
            %15 = affine.apply affine_map<(d0, d1, d2, d3) -> (d0 * 64 + d1 * 262144 + d2 * 262144 + d3 * 32768)>(%arg9, %arg6, %arg7, %arg8)
            %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%15], sizes: [64, 64], strides: [512, 1] {tmd.reuse = {spatial = [{depth = 0 : i64, iterator = "%arg6", mapped_to = "x", reuse_type = "no_reuse", volume = 0 : i64}, {depth = 1 : i64, iterator = "%arg7", mapped_to = "y", reuse_type = "no_reuse", volume = 0 : i64}], temporal = [{depth = 2 : i64, iterator = "%arg8", reuse_type = "no_reuse", volume = 0 : i64}, {depth = 3 : i64, iterator = "%arg9", reuse_type = "no_reuse", volume = 0 : i64}]}} : memref<*xf32> to memref<64x64xf32, strided<[512, 1], offset: ?>>
            bufferization.materialize_in_destination %14 in writable %reinterpret_cast : (tensor<64x64xf32>, memref<64x64xf32, strided<[512, 1], offset: ?>>) -> ()
          }
        }
      } {tmd.mapped_to = "y"}
    } {tmd.mapped_to = "x"}
    return
  }
  func.func @matmul_kernel__d0i0_d1i0__f10__c0mem_c1mem(%arg0: memref<*xf32> {tt.divisibility = 16 : i32}, %arg1: memref<*xf32> {tt.divisibility = 16 : i32}, %arg2: memref<*xf32> {tt.divisibility = 16 : i32}, %arg3: index, %arg4: index, %arg5: index) {
    affine.parallel (%arg6) = (0) to (8) {
      affine.parallel (%arg7) = (0) to (8) {
        affine.for %arg8 = 0 to affine_map<()[s0, s1] -> (s1)>()[%arg3, %arg4] {
          affine.for %arg9 = 0 to affine_map<()[s0, s1] -> ((s0 ceildiv 8) ceildiv 8)>()[%arg3, %arg4] {
            %cst = arith.constant 0.000000e+00 : f32
            %12 = tensor.empty() : tensor<64x64xf32>
            %13 = linalg.fill ins(%cst : f32) outs(%12 : tensor<64x64xf32>) -> tensor<64x64xf32>
            %c0 = arith.constant 0 : index
            %c8 = arith.constant 8 : index
            %c1 = arith.constant 1 : index
            %14 = scf.for %arg10 = %c0 to %c8 step %c1 iter_args(%arg11 = %13) -> (tensor<64x64xf32>) {
              %16 = affine.apply affine_map<(d0, d1, d2, d3) -> (d0 * 64 + d1 * 262144 + d2 * 262144 + d3 * 32768)>(%arg10, %arg6, %arg7, %arg9)
              %reinterpret_cast_0 = memref.reinterpret_cast %arg0 to offset: [%16], sizes: [64, 64], strides: [512, 1] {tmd.reuse = {sequential = [{depth = 4 : i64, iterator = "%arg10", reuse_type = "no_reuse", volume = 0 : i64}], spatial = [{depth = 0 : i64, iterator = "%arg6", mapped_to = "x", reuse_type = "no_reuse", volume = 0 : i64}, {depth = 1 : i64, iterator = "%arg7", mapped_to = "y", reuse_type = "no_reuse", volume = 0 : i64}], temporal = [{depth = 2 : i64, iterator = "%arg8", reuse_type = "total_reuse", volume = 16384 : i64}, {depth = 3 : i64, iterator = "%arg9", reuse_type = "no_reuse", volume = 0 : i64}]}} : memref<*xf32> to memref<64x64xf32, strided<[512, 1], offset: ?>>
              %alloc = memref.alloc() {tmd.alloc = {local = true, memory_name = "L1", size = 16384 : i64}} : memref<64x64xf32>
              memref.copy %reinterpret_cast_0, %alloc {tmd.copy.choice = {kind = "mem", memory_name = "L1"}} : memref<64x64xf32, strided<[512, 1], offset: ?>> to memref<64x64xf32>
              %17 = bufferization.to_tensor %alloc restrict writable : memref<64x64xf32> to tensor<64x64xf32>
              %18 = affine.apply affine_map<(d0, d1) -> (d1 * 32768 + d0 * 64)>(%arg8, %arg10)
              %reinterpret_cast_1 = memref.reinterpret_cast %arg1 to offset: [%18], sizes: [64, 64], strides: [512, 1] {tmd.reuse = {sequential = [{depth = 4 : i64, iterator = "%arg10", reuse_type = "no_reuse", volume = 0 : i64}], spatial = [{depth = 0 : i64, iterator = "%arg6", mapped_to = "x", reuse_type = "total_reuse", volume = 16384 : i64}, {depth = 1 : i64, iterator = "%arg7", mapped_to = "y", reuse_type = "total_reuse", volume = 16384 : i64}], temporal = [{depth = 2 : i64, iterator = "%arg8", reuse_type = "no_reuse", volume = 0 : i64}, {depth = 3 : i64, iterator = "%arg9", reuse_type = "total_reuse", volume = 16384 : i64}]}} : memref<*xf32> to memref<64x64xf32, strided<[512, 1], offset: ?>>
              %alloc_2 = memref.alloc() {tmd.alloc = {local = true, memory_name = "L1", size = 16384 : i64}} : memref<64x64xf32>
              memref.copy %reinterpret_cast_1, %alloc_2 {tmd.copy.choice = {kind = "mem", memory_name = "L1"}} : memref<64x64xf32, strided<[512, 1], offset: ?>> to memref<64x64xf32>
              %19 = bufferization.to_tensor %alloc_2 restrict writable : memref<64x64xf32> to tensor<64x64xf32>
              %20 = linalg.matmul ins(%17, %19 : tensor<64x64xf32>, tensor<64x64xf32>) outs(%13 : tensor<64x64xf32>) -> tensor<64x64xf32>
              %21 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%arg11, %20 : tensor<64x64xf32>, tensor<64x64xf32>) outs(%arg11 : tensor<64x64xf32>) {
              ^bb0(%in: f32, %in_3: f32, %out: f32):
                %22 = arith.addf %in, %in_3 : f32
                linalg.yield %22 : f32
              } -> tensor<64x64xf32>
              scf.yield %21 : tensor<64x64xf32>
            }
            %15 = affine.apply affine_map<(d0, d1, d2, d3) -> (d0 * 64 + d1 * 262144 + d2 * 262144 + d3 * 32768)>(%arg8, %arg6, %arg7, %arg9)
            %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%15], sizes: [64, 64], strides: [512, 1] {tmd.reuse = {spatial = [{depth = 0 : i64, iterator = "%arg6", mapped_to = "x", reuse_type = "no_reuse", volume = 0 : i64}, {depth = 1 : i64, iterator = "%arg7", mapped_to = "y", reuse_type = "no_reuse", volume = 0 : i64}], temporal = [{depth = 2 : i64, iterator = "%arg8", reuse_type = "no_reuse", volume = 0 : i64}, {depth = 3 : i64, iterator = "%arg9", reuse_type = "no_reuse", volume = 0 : i64}]}} : memref<*xf32> to memref<64x64xf32, strided<[512, 1], offset: ?>>
            bufferization.materialize_in_destination %14 in writable %reinterpret_cast : (tensor<64x64xf32>, memref<64x64xf32, strided<[512, 1], offset: ?>>) -> ()
          }
        }
      } {tmd.mapped_to = "y"}
    } {tmd.mapped_to = "x"}
    return
  }
  func.func @matmul_kernel__d0i0_d1i0__f10__c0mem_c1bx(%arg0: memref<*xf32> {tt.divisibility = 16 : i32}, %arg1: memref<*xf32> {tt.divisibility = 16 : i32}, %arg2: memref<*xf32> {tt.divisibility = 16 : i32}, %arg3: index, %arg4: index, %arg5: index) {
    affine.parallel (%arg6) = (0) to (8) {
      affine.parallel (%arg7) = (0) to (8) {
        affine.for %arg8 = 0 to affine_map<()[s0, s1] -> (s1)>()[%arg3, %arg4] {
          affine.for %arg9 = 0 to affine_map<()[s0, s1] -> ((s0 ceildiv 8) ceildiv 8)>()[%arg3, %arg4] {
            %cst = arith.constant 0.000000e+00 : f32
            %12 = tensor.empty() : tensor<64x64xf32>
            %13 = linalg.fill ins(%cst : f32) outs(%12 : tensor<64x64xf32>) -> tensor<64x64xf32>
            %c0 = arith.constant 0 : index
            %c8 = arith.constant 8 : index
            %c1 = arith.constant 1 : index
            %14 = scf.for %arg10 = %c0 to %c8 step %c1 iter_args(%arg11 = %13) -> (tensor<64x64xf32>) {
              %16 = affine.apply affine_map<(d0, d1, d2, d3) -> (d0 * 64 + d1 * 262144 + d2 * 262144 + d3 * 32768)>(%arg10, %arg6, %arg7, %arg9)
              %reinterpret_cast_0 = memref.reinterpret_cast %arg0 to offset: [%16], sizes: [64, 64], strides: [512, 1] {tmd.reuse = {sequential = [{depth = 4 : i64, iterator = "%arg10", reuse_type = "no_reuse", volume = 0 : i64}], spatial = [{depth = 0 : i64, iterator = "%arg6", mapped_to = "x", reuse_type = "no_reuse", volume = 0 : i64}, {depth = 1 : i64, iterator = "%arg7", mapped_to = "y", reuse_type = "no_reuse", volume = 0 : i64}], temporal = [{depth = 2 : i64, iterator = "%arg8", reuse_type = "total_reuse", volume = 16384 : i64}, {depth = 3 : i64, iterator = "%arg9", reuse_type = "no_reuse", volume = 0 : i64}]}} : memref<*xf32> to memref<64x64xf32, strided<[512, 1], offset: ?>>
              %alloc = memref.alloc() {tmd.alloc = {local = true, memory_name = "L1", size = 16384 : i64}} : memref<64x64xf32>
              memref.copy %reinterpret_cast_0, %alloc {tmd.copy.choice = {kind = "mem", memory_name = "L1"}} : memref<64x64xf32, strided<[512, 1], offset: ?>> to memref<64x64xf32>
              %17 = bufferization.to_tensor %alloc restrict writable : memref<64x64xf32> to tensor<64x64xf32>
              %18 = affine.apply affine_map<(d0, d1) -> (d1 * 32768 + d0 * 64)>(%arg8, %arg10)
              %reinterpret_cast_1 = memref.reinterpret_cast %arg1 to offset: [%18], sizes: [64, 64], strides: [512, 1] {tmd.reuse = {sequential = [{depth = 4 : i64, iterator = "%arg10", reuse_type = "no_reuse", volume = 0 : i64}], spatial = [{depth = 0 : i64, iterator = "%arg6", mapped_to = "x", reuse_type = "total_reuse", volume = 16384 : i64}, {depth = 1 : i64, iterator = "%arg7", mapped_to = "y", reuse_type = "total_reuse", volume = 16384 : i64}], temporal = [{depth = 2 : i64, iterator = "%arg8", reuse_type = "no_reuse", volume = 0 : i64}, {depth = 3 : i64, iterator = "%arg9", reuse_type = "total_reuse", volume = 16384 : i64}]}} : memref<*xf32> to memref<64x64xf32, strided<[512, 1], offset: ?>>
              %alloc_2 = memref.alloc() {tmd.alloc = {local = true, memory_name = "L1", size = 16384 : i64}} : memref<64x64xf32>
              memref.copy %reinterpret_cast_1, %alloc_2 {tmd.copy.choice = {dim = "x", interconnect_name = "horizontal_links", kind = "broadcast"}} : memref<64x64xf32, strided<[512, 1], offset: ?>> to memref<64x64xf32>
              %19 = bufferization.to_tensor %alloc_2 restrict writable : memref<64x64xf32> to tensor<64x64xf32>
              %20 = linalg.matmul ins(%17, %19 : tensor<64x64xf32>, tensor<64x64xf32>) outs(%13 : tensor<64x64xf32>) -> tensor<64x64xf32>
              %21 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%arg11, %20 : tensor<64x64xf32>, tensor<64x64xf32>) outs(%arg11 : tensor<64x64xf32>) {
              ^bb0(%in: f32, %in_3: f32, %out: f32):
                %22 = arith.addf %in, %in_3 : f32
                linalg.yield %22 : f32
              } -> tensor<64x64xf32>
              scf.yield %21 : tensor<64x64xf32>
            }
            %15 = affine.apply affine_map<(d0, d1, d2, d3) -> (d0 * 64 + d1 * 262144 + d2 * 262144 + d3 * 32768)>(%arg8, %arg6, %arg7, %arg9)
            %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%15], sizes: [64, 64], strides: [512, 1] {tmd.reuse = {spatial = [{depth = 0 : i64, iterator = "%arg6", mapped_to = "x", reuse_type = "no_reuse", volume = 0 : i64}, {depth = 1 : i64, iterator = "%arg7", mapped_to = "y", reuse_type = "no_reuse", volume = 0 : i64}], temporal = [{depth = 2 : i64, iterator = "%arg8", reuse_type = "no_reuse", volume = 0 : i64}, {depth = 3 : i64, iterator = "%arg9", reuse_type = "no_reuse", volume = 0 : i64}]}} : memref<*xf32> to memref<64x64xf32, strided<[512, 1], offset: ?>>
            bufferization.materialize_in_destination %14 in writable %reinterpret_cast : (tensor<64x64xf32>, memref<64x64xf32, strided<[512, 1], offset: ?>>) -> ()
          }
        }
      } {tmd.mapped_to = "y"}
    } {tmd.mapped_to = "x"}
    return
  }
  func.func @matmul_kernel__d0i0_d1i0__f10__c0mem_c1by(%arg0: memref<*xf32> {tt.divisibility = 16 : i32}, %arg1: memref<*xf32> {tt.divisibility = 16 : i32}, %arg2: memref<*xf32> {tt.divisibility = 16 : i32}, %arg3: index, %arg4: index, %arg5: index) {
    affine.parallel (%arg6) = (0) to (8) {
      affine.parallel (%arg7) = (0) to (8) {
        affine.for %arg8 = 0 to affine_map<()[s0, s1] -> (s1)>()[%arg3, %arg4] {
          affine.for %arg9 = 0 to affine_map<()[s0, s1] -> ((s0 ceildiv 8) ceildiv 8)>()[%arg3, %arg4] {
            %cst = arith.constant 0.000000e+00 : f32
            %12 = tensor.empty() : tensor<64x64xf32>
            %13 = linalg.fill ins(%cst : f32) outs(%12 : tensor<64x64xf32>) -> tensor<64x64xf32>
            %c0 = arith.constant 0 : index
            %c8 = arith.constant 8 : index
            %c1 = arith.constant 1 : index
            %14 = scf.for %arg10 = %c0 to %c8 step %c1 iter_args(%arg11 = %13) -> (tensor<64x64xf32>) {
              %16 = affine.apply affine_map<(d0, d1, d2, d3) -> (d0 * 64 + d1 * 262144 + d2 * 262144 + d3 * 32768)>(%arg10, %arg6, %arg7, %arg9)
              %reinterpret_cast_0 = memref.reinterpret_cast %arg0 to offset: [%16], sizes: [64, 64], strides: [512, 1] {tmd.reuse = {sequential = [{depth = 4 : i64, iterator = "%arg10", reuse_type = "no_reuse", volume = 0 : i64}], spatial = [{depth = 0 : i64, iterator = "%arg6", mapped_to = "x", reuse_type = "no_reuse", volume = 0 : i64}, {depth = 1 : i64, iterator = "%arg7", mapped_to = "y", reuse_type = "no_reuse", volume = 0 : i64}], temporal = [{depth = 2 : i64, iterator = "%arg8", reuse_type = "total_reuse", volume = 16384 : i64}, {depth = 3 : i64, iterator = "%arg9", reuse_type = "no_reuse", volume = 0 : i64}]}} : memref<*xf32> to memref<64x64xf32, strided<[512, 1], offset: ?>>
              %alloc = memref.alloc() {tmd.alloc = {local = true, memory_name = "L1", size = 16384 : i64}} : memref<64x64xf32>
              memref.copy %reinterpret_cast_0, %alloc {tmd.copy.choice = {kind = "mem", memory_name = "L1"}} : memref<64x64xf32, strided<[512, 1], offset: ?>> to memref<64x64xf32>
              %17 = bufferization.to_tensor %alloc restrict writable : memref<64x64xf32> to tensor<64x64xf32>
              %18 = affine.apply affine_map<(d0, d1) -> (d1 * 32768 + d0 * 64)>(%arg8, %arg10)
              %reinterpret_cast_1 = memref.reinterpret_cast %arg1 to offset: [%18], sizes: [64, 64], strides: [512, 1] {tmd.reuse = {sequential = [{depth = 4 : i64, iterator = "%arg10", reuse_type = "no_reuse", volume = 0 : i64}], spatial = [{depth = 0 : i64, iterator = "%arg6", mapped_to = "x", reuse_type = "total_reuse", volume = 16384 : i64}, {depth = 1 : i64, iterator = "%arg7", mapped_to = "y", reuse_type = "total_reuse", volume = 16384 : i64}], temporal = [{depth = 2 : i64, iterator = "%arg8", reuse_type = "no_reuse", volume = 0 : i64}, {depth = 3 : i64, iterator = "%arg9", reuse_type = "total_reuse", volume = 16384 : i64}]}} : memref<*xf32> to memref<64x64xf32, strided<[512, 1], offset: ?>>
              %alloc_2 = memref.alloc() {tmd.alloc = {local = true, memory_name = "L1", size = 16384 : i64}} : memref<64x64xf32>
              memref.copy %reinterpret_cast_1, %alloc_2 {tmd.copy.choice = {dim = "y", interconnect_name = "vertical_links", kind = "broadcast"}} : memref<64x64xf32, strided<[512, 1], offset: ?>> to memref<64x64xf32>
              %19 = bufferization.to_tensor %alloc_2 restrict writable : memref<64x64xf32> to tensor<64x64xf32>
              %20 = linalg.matmul ins(%17, %19 : tensor<64x64xf32>, tensor<64x64xf32>) outs(%13 : tensor<64x64xf32>) -> tensor<64x64xf32>
              %21 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%arg11, %20 : tensor<64x64xf32>, tensor<64x64xf32>) outs(%arg11 : tensor<64x64xf32>) {
              ^bb0(%in: f32, %in_3: f32, %out: f32):
                %22 = arith.addf %in, %in_3 : f32
                linalg.yield %22 : f32
              } -> tensor<64x64xf32>
              scf.yield %21 : tensor<64x64xf32>
            }
            %15 = affine.apply affine_map<(d0, d1, d2, d3) -> (d0 * 64 + d1 * 262144 + d2 * 262144 + d3 * 32768)>(%arg8, %arg6, %arg7, %arg9)
            %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%15], sizes: [64, 64], strides: [512, 1] {tmd.reuse = {spatial = [{depth = 0 : i64, iterator = "%arg6", mapped_to = "x", reuse_type = "no_reuse", volume = 0 : i64}, {depth = 1 : i64, iterator = "%arg7", mapped_to = "y", reuse_type = "no_reuse", volume = 0 : i64}], temporal = [{depth = 2 : i64, iterator = "%arg8", reuse_type = "no_reuse", volume = 0 : i64}, {depth = 3 : i64, iterator = "%arg9", reuse_type = "no_reuse", volume = 0 : i64}]}} : memref<*xf32> to memref<64x64xf32, strided<[512, 1], offset: ?>>
            bufferization.materialize_in_destination %14 in writable %reinterpret_cast : (tensor<64x64xf32>, memref<64x64xf32, strided<[512, 1], offset: ?>>) -> ()
          }
        }
      } {tmd.mapped_to = "y"}
    } {tmd.mapped_to = "x"}
    return
  }
  func.func @matmul_kernel__d0i0_d1i0__f10__c0mem_c1bd(%arg0: memref<*xf32> {tt.divisibility = 16 : i32}, %arg1: memref<*xf32> {tt.divisibility = 16 : i32}, %arg2: memref<*xf32> {tt.divisibility = 16 : i32}, %arg3: index, %arg4: index, %arg5: index) {
    affine.parallel (%arg6) = (0) to (8) {
      affine.parallel (%arg7) = (0) to (8) {
        affine.for %arg8 = 0 to affine_map<()[s0, s1] -> (s1)>()[%arg3, %arg4] {
          affine.for %arg9 = 0 to affine_map<()[s0, s1] -> ((s0 ceildiv 8) ceildiv 8)>()[%arg3, %arg4] {
            %cst = arith.constant 0.000000e+00 : f32
            %12 = tensor.empty() : tensor<64x64xf32>
            %13 = linalg.fill ins(%cst : f32) outs(%12 : tensor<64x64xf32>) -> tensor<64x64xf32>
            %c0 = arith.constant 0 : index
            %c8 = arith.constant 8 : index
            %c1 = arith.constant 1 : index
            %14 = scf.for %arg10 = %c0 to %c8 step %c1 iter_args(%arg11 = %13) -> (tensor<64x64xf32>) {
              %16 = affine.apply affine_map<(d0, d1, d2, d3) -> (d0 * 64 + d1 * 262144 + d2 * 262144 + d3 * 32768)>(%arg10, %arg6, %arg7, %arg9)
              %reinterpret_cast_0 = memref.reinterpret_cast %arg0 to offset: [%16], sizes: [64, 64], strides: [512, 1] {tmd.reuse = {sequential = [{depth = 4 : i64, iterator = "%arg10", reuse_type = "no_reuse", volume = 0 : i64}], spatial = [{depth = 0 : i64, iterator = "%arg6", mapped_to = "x", reuse_type = "no_reuse", volume = 0 : i64}, {depth = 1 : i64, iterator = "%arg7", mapped_to = "y", reuse_type = "no_reuse", volume = 0 : i64}], temporal = [{depth = 2 : i64, iterator = "%arg8", reuse_type = "total_reuse", volume = 16384 : i64}, {depth = 3 : i64, iterator = "%arg9", reuse_type = "no_reuse", volume = 0 : i64}]}} : memref<*xf32> to memref<64x64xf32, strided<[512, 1], offset: ?>>
              %alloc = memref.alloc() {tmd.alloc = {local = true, memory_name = "L1", size = 16384 : i64}} : memref<64x64xf32>
              memref.copy %reinterpret_cast_0, %alloc {tmd.copy.choice = {kind = "mem", memory_name = "L1"}} : memref<64x64xf32, strided<[512, 1], offset: ?>> to memref<64x64xf32>
              %17 = bufferization.to_tensor %alloc restrict writable : memref<64x64xf32> to tensor<64x64xf32>
              %18 = affine.apply affine_map<(d0, d1) -> (d1 * 32768 + d0 * 64)>(%arg8, %arg10)
              %reinterpret_cast_1 = memref.reinterpret_cast %arg1 to offset: [%18], sizes: [64, 64], strides: [512, 1] {tmd.reuse = {sequential = [{depth = 4 : i64, iterator = "%arg10", reuse_type = "no_reuse", volume = 0 : i64}], spatial = [{depth = 0 : i64, iterator = "%arg6", mapped_to = "x", reuse_type = "total_reuse", volume = 16384 : i64}, {depth = 1 : i64, iterator = "%arg7", mapped_to = "y", reuse_type = "total_reuse", volume = 16384 : i64}], temporal = [{depth = 2 : i64, iterator = "%arg8", reuse_type = "no_reuse", volume = 0 : i64}, {depth = 3 : i64, iterator = "%arg9", reuse_type = "total_reuse", volume = 16384 : i64}]}} : memref<*xf32> to memref<64x64xf32, strided<[512, 1], offset: ?>>
              %alloc_2 = memref.alloc() {tmd.alloc = {local = true, memory_name = "L1", size = 16384 : i64}} : memref<64x64xf32>
              memref.copy %reinterpret_cast_1, %alloc_2 {tmd.copy.choice = {dim = "xy", interconnect_name = "all_links", kind = "broadcast"}} : memref<64x64xf32, strided<[512, 1], offset: ?>> to memref<64x64xf32>
              %19 = bufferization.to_tensor %alloc_2 restrict writable : memref<64x64xf32> to tensor<64x64xf32>
              %20 = linalg.matmul ins(%17, %19 : tensor<64x64xf32>, tensor<64x64xf32>) outs(%13 : tensor<64x64xf32>) -> tensor<64x64xf32>
              %21 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%arg11, %20 : tensor<64x64xf32>, tensor<64x64xf32>) outs(%arg11 : tensor<64x64xf32>) {
              ^bb0(%in: f32, %in_3: f32, %out: f32):
                %22 = arith.addf %in, %in_3 : f32
                linalg.yield %22 : f32
              } -> tensor<64x64xf32>
              scf.yield %21 : tensor<64x64xf32>
            }
            %15 = affine.apply affine_map<(d0, d1, d2, d3) -> (d0 * 64 + d1 * 262144 + d2 * 262144 + d3 * 32768)>(%arg8, %arg6, %arg7, %arg9)
            %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%15], sizes: [64, 64], strides: [512, 1] {tmd.reuse = {spatial = [{depth = 0 : i64, iterator = "%arg6", mapped_to = "x", reuse_type = "no_reuse", volume = 0 : i64}, {depth = 1 : i64, iterator = "%arg7", mapped_to = "y", reuse_type = "no_reuse", volume = 0 : i64}], temporal = [{depth = 2 : i64, iterator = "%arg8", reuse_type = "no_reuse", volume = 0 : i64}, {depth = 3 : i64, iterator = "%arg9", reuse_type = "no_reuse", volume = 0 : i64}]}} : memref<*xf32> to memref<64x64xf32, strided<[512, 1], offset: ?>>
            bufferization.materialize_in_destination %14 in writable %reinterpret_cast : (tensor<64x64xf32>, memref<64x64xf32, strided<[512, 1], offset: ?>>) -> ()
          }
        }
      } {tmd.mapped_to = "y"}
    } {tmd.mapped_to = "x"}
    return
  }
  func.func @matmul_kernel__d0i0_d1i1__f01__c0mem_c1mem(%arg0: memref<*xf32> {tt.divisibility = 16 : i32}, %arg1: memref<*xf32> {tt.divisibility = 16 : i32}, %arg2: memref<*xf32> {tt.divisibility = 16 : i32}, %arg3: index, %arg4: index, %arg5: index) {
    affine.parallel (%arg6) = (0) to (8) {
      affine.parallel (%arg7) = (0) to (8) {
        affine.for %arg8 = 0 to affine_map<()[s0, s1] -> (s0 ceildiv 8)>()[%arg3, %arg4] {
          affine.for %arg9 = 0 to affine_map<()[s0, s1] -> (s1 ceildiv 8)>()[%arg3, %arg4] {
            %cst = arith.constant 0.000000e+00 : f32
            %12 = tensor.empty() : tensor<64x64xf32>
            %13 = linalg.fill ins(%cst : f32) outs(%12 : tensor<64x64xf32>) -> tensor<64x64xf32>
            %c0 = arith.constant 0 : index
            %c8 = arith.constant 8 : index
            %c1 = arith.constant 1 : index
            %14 = scf.for %arg10 = %c0 to %c8 step %c1 iter_args(%arg11 = %13) -> (tensor<64x64xf32>) {
              %16 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 64 + d1 * 262144 + d2 * 32768)>(%arg10, %arg6, %arg8)
              %reinterpret_cast_0 = memref.reinterpret_cast %arg0 to offset: [%16], sizes: [64, 64], strides: [512, 1] {tmd.reuse = {sequential = [{depth = 4 : i64, iterator = "%arg10", reuse_type = "no_reuse", volume = 0 : i64}], spatial = [{depth = 0 : i64, iterator = "%arg6", mapped_to = "x", reuse_type = "no_reuse", volume = 0 : i64}, {depth = 1 : i64, iterator = "%arg7", mapped_to = "y", reuse_type = "total_reuse", volume = 16384 : i64}], temporal = [{depth = 2 : i64, iterator = "%arg8", reuse_type = "no_reuse", volume = 0 : i64}, {depth = 3 : i64, iterator = "%arg9", reuse_type = "total_reuse", volume = 16384 : i64}]}} : memref<*xf32> to memref<64x64xf32, strided<[512, 1], offset: ?>>
              %alloc = memref.alloc() {tmd.alloc = {local = true, memory_name = "L1", size = 16384 : i64}} : memref<64x64xf32>
              memref.copy %reinterpret_cast_0, %alloc {tmd.copy.choice = {kind = "mem", memory_name = "L1"}} : memref<64x64xf32, strided<[512, 1], offset: ?>> to memref<64x64xf32>
              %17 = bufferization.to_tensor %alloc restrict writable : memref<64x64xf32> to tensor<64x64xf32>
              %18 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 32768 + d1 * 512 + d2 * 64)>(%arg10, %arg7, %arg9)
              %reinterpret_cast_1 = memref.reinterpret_cast %arg1 to offset: [%18], sizes: [64, 64], strides: [512, 1] {tmd.reuse = {sequential = [{depth = 4 : i64, iterator = "%arg10", reuse_type = "no_reuse", volume = 0 : i64}], spatial = [{depth = 0 : i64, iterator = "%arg6", mapped_to = "x", reuse_type = "total_reuse", volume = 16384 : i64}, {depth = 1 : i64, iterator = "%arg7", mapped_to = "y", reuse_type = "no_reuse", volume = 0 : i64}], temporal = [{depth = 2 : i64, iterator = "%arg8", reuse_type = "total_reuse", volume = 16384 : i64}, {depth = 3 : i64, iterator = "%arg9", reuse_type = "no_reuse", volume = 0 : i64}]}} : memref<*xf32> to memref<64x64xf32, strided<[512, 1], offset: ?>>
              %alloc_2 = memref.alloc() {tmd.alloc = {local = true, memory_name = "L1", size = 16384 : i64}} : memref<64x64xf32>
              memref.copy %reinterpret_cast_1, %alloc_2 {tmd.copy.choice = {kind = "mem", memory_name = "L1"}} : memref<64x64xf32, strided<[512, 1], offset: ?>> to memref<64x64xf32>
              %19 = bufferization.to_tensor %alloc_2 restrict writable : memref<64x64xf32> to tensor<64x64xf32>
              %20 = linalg.matmul ins(%17, %19 : tensor<64x64xf32>, tensor<64x64xf32>) outs(%13 : tensor<64x64xf32>) -> tensor<64x64xf32>
              %21 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%arg11, %20 : tensor<64x64xf32>, tensor<64x64xf32>) outs(%arg11 : tensor<64x64xf32>) {
              ^bb0(%in: f32, %in_3: f32, %out: f32):
                %22 = arith.addf %in, %in_3 : f32
                linalg.yield %22 : f32
              } -> tensor<64x64xf32>
              scf.yield %21 : tensor<64x64xf32>
            }
            %15 = affine.apply affine_map<(d0, d1, d2, d3) -> (d0 * 512 + d1 * 64 + d2 * 262144 + d3 * 32768)>(%arg7, %arg9, %arg6, %arg8)
            %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%15], sizes: [64, 64], strides: [512, 1] {tmd.reuse = {spatial = [{depth = 0 : i64, iterator = "%arg6", mapped_to = "x", reuse_type = "no_reuse", volume = 0 : i64}, {depth = 1 : i64, iterator = "%arg7", mapped_to = "y", reuse_type = "no_reuse", volume = 0 : i64}], temporal = [{depth = 2 : i64, iterator = "%arg8", reuse_type = "no_reuse", volume = 0 : i64}, {depth = 3 : i64, iterator = "%arg9", reuse_type = "no_reuse", volume = 0 : i64}]}} : memref<*xf32> to memref<64x64xf32, strided<[512, 1], offset: ?>>
            bufferization.materialize_in_destination %14 in writable %reinterpret_cast : (tensor<64x64xf32>, memref<64x64xf32, strided<[512, 1], offset: ?>>) -> ()
          }
        }
      } {tmd.mapped_to = "y"}
    } {tmd.mapped_to = "x"}
    return
  }
  func.func @matmul_kernel__d0i0_d1i1__f01__c0by_c1mem(%arg0: memref<*xf32> {tt.divisibility = 16 : i32}, %arg1: memref<*xf32> {tt.divisibility = 16 : i32}, %arg2: memref<*xf32> {tt.divisibility = 16 : i32}, %arg3: index, %arg4: index, %arg5: index) {
    affine.parallel (%arg6) = (0) to (8) {
      affine.parallel (%arg7) = (0) to (8) {
        affine.for %arg8 = 0 to affine_map<()[s0, s1] -> (s0 ceildiv 8)>()[%arg3, %arg4] {
          affine.for %arg9 = 0 to affine_map<()[s0, s1] -> (s1 ceildiv 8)>()[%arg3, %arg4] {
            %cst = arith.constant 0.000000e+00 : f32
            %12 = tensor.empty() : tensor<64x64xf32>
            %13 = linalg.fill ins(%cst : f32) outs(%12 : tensor<64x64xf32>) -> tensor<64x64xf32>
            %c0 = arith.constant 0 : index
            %c8 = arith.constant 8 : index
            %c1 = arith.constant 1 : index
            %14 = scf.for %arg10 = %c0 to %c8 step %c1 iter_args(%arg11 = %13) -> (tensor<64x64xf32>) {
              %16 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 64 + d1 * 262144 + d2 * 32768)>(%arg10, %arg6, %arg8)
              %reinterpret_cast_0 = memref.reinterpret_cast %arg0 to offset: [%16], sizes: [64, 64], strides: [512, 1] {tmd.reuse = {sequential = [{depth = 4 : i64, iterator = "%arg10", reuse_type = "no_reuse", volume = 0 : i64}], spatial = [{depth = 0 : i64, iterator = "%arg6", mapped_to = "x", reuse_type = "no_reuse", volume = 0 : i64}, {depth = 1 : i64, iterator = "%arg7", mapped_to = "y", reuse_type = "total_reuse", volume = 16384 : i64}], temporal = [{depth = 2 : i64, iterator = "%arg8", reuse_type = "no_reuse", volume = 0 : i64}, {depth = 3 : i64, iterator = "%arg9", reuse_type = "total_reuse", volume = 16384 : i64}]}} : memref<*xf32> to memref<64x64xf32, strided<[512, 1], offset: ?>>
              %alloc = memref.alloc() {tmd.alloc = {local = true, memory_name = "L1", size = 16384 : i64}} : memref<64x64xf32>
              memref.copy %reinterpret_cast_0, %alloc {tmd.copy.choice = {dim = "y", interconnect_name = "vertical_links", kind = "broadcast"}} : memref<64x64xf32, strided<[512, 1], offset: ?>> to memref<64x64xf32>
              %17 = bufferization.to_tensor %alloc restrict writable : memref<64x64xf32> to tensor<64x64xf32>
              %18 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 32768 + d1 * 512 + d2 * 64)>(%arg10, %arg7, %arg9)
              %reinterpret_cast_1 = memref.reinterpret_cast %arg1 to offset: [%18], sizes: [64, 64], strides: [512, 1] {tmd.reuse = {sequential = [{depth = 4 : i64, iterator = "%arg10", reuse_type = "no_reuse", volume = 0 : i64}], spatial = [{depth = 0 : i64, iterator = "%arg6", mapped_to = "x", reuse_type = "total_reuse", volume = 16384 : i64}, {depth = 1 : i64, iterator = "%arg7", mapped_to = "y", reuse_type = "no_reuse", volume = 0 : i64}], temporal = [{depth = 2 : i64, iterator = "%arg8", reuse_type = "total_reuse", volume = 16384 : i64}, {depth = 3 : i64, iterator = "%arg9", reuse_type = "no_reuse", volume = 0 : i64}]}} : memref<*xf32> to memref<64x64xf32, strided<[512, 1], offset: ?>>
              %alloc_2 = memref.alloc() {tmd.alloc = {local = true, memory_name = "L1", size = 16384 : i64}} : memref<64x64xf32>
              memref.copy %reinterpret_cast_1, %alloc_2 {tmd.copy.choice = {kind = "mem", memory_name = "L1"}} : memref<64x64xf32, strided<[512, 1], offset: ?>> to memref<64x64xf32>
              %19 = bufferization.to_tensor %alloc_2 restrict writable : memref<64x64xf32> to tensor<64x64xf32>
              %20 = linalg.matmul ins(%17, %19 : tensor<64x64xf32>, tensor<64x64xf32>) outs(%13 : tensor<64x64xf32>) -> tensor<64x64xf32>
              %21 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%arg11, %20 : tensor<64x64xf32>, tensor<64x64xf32>) outs(%arg11 : tensor<64x64xf32>) {
              ^bb0(%in: f32, %in_3: f32, %out: f32):
                %22 = arith.addf %in, %in_3 : f32
                linalg.yield %22 : f32
              } -> tensor<64x64xf32>
              scf.yield %21 : tensor<64x64xf32>
            }
            %15 = affine.apply affine_map<(d0, d1, d2, d3) -> (d0 * 512 + d1 * 64 + d2 * 262144 + d3 * 32768)>(%arg7, %arg9, %arg6, %arg8)
            %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%15], sizes: [64, 64], strides: [512, 1] {tmd.reuse = {spatial = [{depth = 0 : i64, iterator = "%arg6", mapped_to = "x", reuse_type = "no_reuse", volume = 0 : i64}, {depth = 1 : i64, iterator = "%arg7", mapped_to = "y", reuse_type = "no_reuse", volume = 0 : i64}], temporal = [{depth = 2 : i64, iterator = "%arg8", reuse_type = "no_reuse", volume = 0 : i64}, {depth = 3 : i64, iterator = "%arg9", reuse_type = "no_reuse", volume = 0 : i64}]}} : memref<*xf32> to memref<64x64xf32, strided<[512, 1], offset: ?>>
            bufferization.materialize_in_destination %14 in writable %reinterpret_cast : (tensor<64x64xf32>, memref<64x64xf32, strided<[512, 1], offset: ?>>) -> ()
          }
        }
      } {tmd.mapped_to = "y"}
    } {tmd.mapped_to = "x"}
    return
  }
  func.func @matmul_kernel__d0i0_d1i1__f01__c0mem_c1bx(%arg0: memref<*xf32> {tt.divisibility = 16 : i32}, %arg1: memref<*xf32> {tt.divisibility = 16 : i32}, %arg2: memref<*xf32> {tt.divisibility = 16 : i32}, %arg3: index, %arg4: index, %arg5: index) {
    affine.parallel (%arg6) = (0) to (8) {
      affine.parallel (%arg7) = (0) to (8) {
        affine.for %arg8 = 0 to affine_map<()[s0, s1] -> (s0 ceildiv 8)>()[%arg3, %arg4] {
          affine.for %arg9 = 0 to affine_map<()[s0, s1] -> (s1 ceildiv 8)>()[%arg3, %arg4] {
            %cst = arith.constant 0.000000e+00 : f32
            %12 = tensor.empty() : tensor<64x64xf32>
            %13 = linalg.fill ins(%cst : f32) outs(%12 : tensor<64x64xf32>) -> tensor<64x64xf32>
            %c0 = arith.constant 0 : index
            %c8 = arith.constant 8 : index
            %c1 = arith.constant 1 : index
            %14 = scf.for %arg10 = %c0 to %c8 step %c1 iter_args(%arg11 = %13) -> (tensor<64x64xf32>) {
              %16 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 64 + d1 * 262144 + d2 * 32768)>(%arg10, %arg6, %arg8)
              %reinterpret_cast_0 = memref.reinterpret_cast %arg0 to offset: [%16], sizes: [64, 64], strides: [512, 1] {tmd.reuse = {sequential = [{depth = 4 : i64, iterator = "%arg10", reuse_type = "no_reuse", volume = 0 : i64}], spatial = [{depth = 0 : i64, iterator = "%arg6", mapped_to = "x", reuse_type = "no_reuse", volume = 0 : i64}, {depth = 1 : i64, iterator = "%arg7", mapped_to = "y", reuse_type = "total_reuse", volume = 16384 : i64}], temporal = [{depth = 2 : i64, iterator = "%arg8", reuse_type = "no_reuse", volume = 0 : i64}, {depth = 3 : i64, iterator = "%arg9", reuse_type = "total_reuse", volume = 16384 : i64}]}} : memref<*xf32> to memref<64x64xf32, strided<[512, 1], offset: ?>>
              %alloc = memref.alloc() {tmd.alloc = {local = true, memory_name = "L1", size = 16384 : i64}} : memref<64x64xf32>
              memref.copy %reinterpret_cast_0, %alloc {tmd.copy.choice = {kind = "mem", memory_name = "L1"}} : memref<64x64xf32, strided<[512, 1], offset: ?>> to memref<64x64xf32>
              %17 = bufferization.to_tensor %alloc restrict writable : memref<64x64xf32> to tensor<64x64xf32>
              %18 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 32768 + d1 * 512 + d2 * 64)>(%arg10, %arg7, %arg9)
              %reinterpret_cast_1 = memref.reinterpret_cast %arg1 to offset: [%18], sizes: [64, 64], strides: [512, 1] {tmd.reuse = {sequential = [{depth = 4 : i64, iterator = "%arg10", reuse_type = "no_reuse", volume = 0 : i64}], spatial = [{depth = 0 : i64, iterator = "%arg6", mapped_to = "x", reuse_type = "total_reuse", volume = 16384 : i64}, {depth = 1 : i64, iterator = "%arg7", mapped_to = "y", reuse_type = "no_reuse", volume = 0 : i64}], temporal = [{depth = 2 : i64, iterator = "%arg8", reuse_type = "total_reuse", volume = 16384 : i64}, {depth = 3 : i64, iterator = "%arg9", reuse_type = "no_reuse", volume = 0 : i64}]}} : memref<*xf32> to memref<64x64xf32, strided<[512, 1], offset: ?>>
              %alloc_2 = memref.alloc() {tmd.alloc = {local = true, memory_name = "L1", size = 16384 : i64}} : memref<64x64xf32>
              memref.copy %reinterpret_cast_1, %alloc_2 {tmd.copy.choice = {dim = "x", interconnect_name = "horizontal_links", kind = "broadcast"}} : memref<64x64xf32, strided<[512, 1], offset: ?>> to memref<64x64xf32>
              %19 = bufferization.to_tensor %alloc_2 restrict writable : memref<64x64xf32> to tensor<64x64xf32>
              %20 = linalg.matmul ins(%17, %19 : tensor<64x64xf32>, tensor<64x64xf32>) outs(%13 : tensor<64x64xf32>) -> tensor<64x64xf32>
              %21 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%arg11, %20 : tensor<64x64xf32>, tensor<64x64xf32>) outs(%arg11 : tensor<64x64xf32>) {
              ^bb0(%in: f32, %in_3: f32, %out: f32):
                %22 = arith.addf %in, %in_3 : f32
                linalg.yield %22 : f32
              } -> tensor<64x64xf32>
              scf.yield %21 : tensor<64x64xf32>
            }
            %15 = affine.apply affine_map<(d0, d1, d2, d3) -> (d0 * 512 + d1 * 64 + d2 * 262144 + d3 * 32768)>(%arg7, %arg9, %arg6, %arg8)
            %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%15], sizes: [64, 64], strides: [512, 1] {tmd.reuse = {spatial = [{depth = 0 : i64, iterator = "%arg6", mapped_to = "x", reuse_type = "no_reuse", volume = 0 : i64}, {depth = 1 : i64, iterator = "%arg7", mapped_to = "y", reuse_type = "no_reuse", volume = 0 : i64}], temporal = [{depth = 2 : i64, iterator = "%arg8", reuse_type = "no_reuse", volume = 0 : i64}, {depth = 3 : i64, iterator = "%arg9", reuse_type = "no_reuse", volume = 0 : i64}]}} : memref<*xf32> to memref<64x64xf32, strided<[512, 1], offset: ?>>
            bufferization.materialize_in_destination %14 in writable %reinterpret_cast : (tensor<64x64xf32>, memref<64x64xf32, strided<[512, 1], offset: ?>>) -> ()
          }
        }
      } {tmd.mapped_to = "y"}
    } {tmd.mapped_to = "x"}
    return
  }
  func.func @matmul_kernel__d0i0_d1i1__f01__c0by_c1bx(%arg0: memref<*xf32> {tt.divisibility = 16 : i32}, %arg1: memref<*xf32> {tt.divisibility = 16 : i32}, %arg2: memref<*xf32> {tt.divisibility = 16 : i32}, %arg3: index, %arg4: index, %arg5: index) {
    affine.parallel (%arg6) = (0) to (8) {
      affine.parallel (%arg7) = (0) to (8) {
        affine.for %arg8 = 0 to affine_map<()[s0, s1] -> (s0 ceildiv 8)>()[%arg3, %arg4] {
          affine.for %arg9 = 0 to affine_map<()[s0, s1] -> (s1 ceildiv 8)>()[%arg3, %arg4] {
            %cst = arith.constant 0.000000e+00 : f32
            %12 = tensor.empty() : tensor<64x64xf32>
            %13 = linalg.fill ins(%cst : f32) outs(%12 : tensor<64x64xf32>) -> tensor<64x64xf32>
            %c0 = arith.constant 0 : index
            %c8 = arith.constant 8 : index
            %c1 = arith.constant 1 : index
            %14 = scf.for %arg10 = %c0 to %c8 step %c1 iter_args(%arg11 = %13) -> (tensor<64x64xf32>) {
              %16 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 64 + d1 * 262144 + d2 * 32768)>(%arg10, %arg6, %arg8)
              %reinterpret_cast_0 = memref.reinterpret_cast %arg0 to offset: [%16], sizes: [64, 64], strides: [512, 1] {tmd.reuse = {sequential = [{depth = 4 : i64, iterator = "%arg10", reuse_type = "no_reuse", volume = 0 : i64}], spatial = [{depth = 0 : i64, iterator = "%arg6", mapped_to = "x", reuse_type = "no_reuse", volume = 0 : i64}, {depth = 1 : i64, iterator = "%arg7", mapped_to = "y", reuse_type = "total_reuse", volume = 16384 : i64}], temporal = [{depth = 2 : i64, iterator = "%arg8", reuse_type = "no_reuse", volume = 0 : i64}, {depth = 3 : i64, iterator = "%arg9", reuse_type = "total_reuse", volume = 16384 : i64}]}} : memref<*xf32> to memref<64x64xf32, strided<[512, 1], offset: ?>>
              %alloc = memref.alloc() {tmd.alloc = {local = true, memory_name = "L1", size = 16384 : i64}} : memref<64x64xf32>
              memref.copy %reinterpret_cast_0, %alloc {tmd.copy.choice = {dim = "y", interconnect_name = "vertical_links", kind = "broadcast"}} : memref<64x64xf32, strided<[512, 1], offset: ?>> to memref<64x64xf32>
              %17 = bufferization.to_tensor %alloc restrict writable : memref<64x64xf32> to tensor<64x64xf32>
              %18 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 32768 + d1 * 512 + d2 * 64)>(%arg10, %arg7, %arg9)
              %reinterpret_cast_1 = memref.reinterpret_cast %arg1 to offset: [%18], sizes: [64, 64], strides: [512, 1] {tmd.reuse = {sequential = [{depth = 4 : i64, iterator = "%arg10", reuse_type = "no_reuse", volume = 0 : i64}], spatial = [{depth = 0 : i64, iterator = "%arg6", mapped_to = "x", reuse_type = "total_reuse", volume = 16384 : i64}, {depth = 1 : i64, iterator = "%arg7", mapped_to = "y", reuse_type = "no_reuse", volume = 0 : i64}], temporal = [{depth = 2 : i64, iterator = "%arg8", reuse_type = "total_reuse", volume = 16384 : i64}, {depth = 3 : i64, iterator = "%arg9", reuse_type = "no_reuse", volume = 0 : i64}]}} : memref<*xf32> to memref<64x64xf32, strided<[512, 1], offset: ?>>
              %alloc_2 = memref.alloc() {tmd.alloc = {local = true, memory_name = "L1", size = 16384 : i64}} : memref<64x64xf32>
              memref.copy %reinterpret_cast_1, %alloc_2 {tmd.copy.choice = {dim = "x", interconnect_name = "horizontal_links", kind = "broadcast"}} : memref<64x64xf32, strided<[512, 1], offset: ?>> to memref<64x64xf32>
              %19 = bufferization.to_tensor %alloc_2 restrict writable : memref<64x64xf32> to tensor<64x64xf32>
              %20 = linalg.matmul ins(%17, %19 : tensor<64x64xf32>, tensor<64x64xf32>) outs(%13 : tensor<64x64xf32>) -> tensor<64x64xf32>
              %21 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%arg11, %20 : tensor<64x64xf32>, tensor<64x64xf32>) outs(%arg11 : tensor<64x64xf32>) {
              ^bb0(%in: f32, %in_3: f32, %out: f32):
                %22 = arith.addf %in, %in_3 : f32
                linalg.yield %22 : f32
              } -> tensor<64x64xf32>
              scf.yield %21 : tensor<64x64xf32>
            }
            %15 = affine.apply affine_map<(d0, d1, d2, d3) -> (d0 * 512 + d1 * 64 + d2 * 262144 + d3 * 32768)>(%arg7, %arg9, %arg6, %arg8)
            %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%15], sizes: [64, 64], strides: [512, 1] {tmd.reuse = {spatial = [{depth = 0 : i64, iterator = "%arg6", mapped_to = "x", reuse_type = "no_reuse", volume = 0 : i64}, {depth = 1 : i64, iterator = "%arg7", mapped_to = "y", reuse_type = "no_reuse", volume = 0 : i64}], temporal = [{depth = 2 : i64, iterator = "%arg8", reuse_type = "no_reuse", volume = 0 : i64}, {depth = 3 : i64, iterator = "%arg9", reuse_type = "no_reuse", volume = 0 : i64}]}} : memref<*xf32> to memref<64x64xf32, strided<[512, 1], offset: ?>>
            bufferization.materialize_in_destination %14 in writable %reinterpret_cast : (tensor<64x64xf32>, memref<64x64xf32, strided<[512, 1], offset: ?>>) -> ()
          }
        }
      } {tmd.mapped_to = "y"}
    } {tmd.mapped_to = "x"}
    return
  }
  func.func @matmul_kernel__d0i0_d1i1__f10__c0mem_c1mem(%arg0: memref<*xf32> {tt.divisibility = 16 : i32}, %arg1: memref<*xf32> {tt.divisibility = 16 : i32}, %arg2: memref<*xf32> {tt.divisibility = 16 : i32}, %arg3: index, %arg4: index, %arg5: index) {
    affine.parallel (%arg6) = (0) to (8) {
      affine.parallel (%arg7) = (0) to (8) {
        affine.for %arg8 = 0 to affine_map<()[s0, s1] -> (s1 ceildiv 8)>()[%arg3, %arg4] {
          affine.for %arg9 = 0 to affine_map<()[s0, s1] -> (s0 ceildiv 8)>()[%arg3, %arg4] {
            %cst = arith.constant 0.000000e+00 : f32
            %12 = tensor.empty() : tensor<64x64xf32>
            %13 = linalg.fill ins(%cst : f32) outs(%12 : tensor<64x64xf32>) -> tensor<64x64xf32>
            %c0 = arith.constant 0 : index
            %c8 = arith.constant 8 : index
            %c1 = arith.constant 1 : index
            %14 = scf.for %arg10 = %c0 to %c8 step %c1 iter_args(%arg11 = %13) -> (tensor<64x64xf32>) {
              %16 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 64 + d1 * 262144 + d2 * 32768)>(%arg10, %arg6, %arg9)
              %reinterpret_cast_0 = memref.reinterpret_cast %arg0 to offset: [%16], sizes: [64, 64], strides: [512, 1] {tmd.reuse = {sequential = [{depth = 4 : i64, iterator = "%arg10", reuse_type = "no_reuse", volume = 0 : i64}], spatial = [{depth = 0 : i64, iterator = "%arg6", mapped_to = "x", reuse_type = "no_reuse", volume = 0 : i64}, {depth = 1 : i64, iterator = "%arg7", mapped_to = "y", reuse_type = "total_reuse", volume = 16384 : i64}], temporal = [{depth = 2 : i64, iterator = "%arg8", reuse_type = "total_reuse", volume = 16384 : i64}, {depth = 3 : i64, iterator = "%arg9", reuse_type = "no_reuse", volume = 0 : i64}]}} : memref<*xf32> to memref<64x64xf32, strided<[512, 1], offset: ?>>
              %alloc = memref.alloc() {tmd.alloc = {local = true, memory_name = "L1", size = 16384 : i64}} : memref<64x64xf32>
              memref.copy %reinterpret_cast_0, %alloc {tmd.copy.choice = {kind = "mem", memory_name = "L1"}} : memref<64x64xf32, strided<[512, 1], offset: ?>> to memref<64x64xf32>
              %17 = bufferization.to_tensor %alloc restrict writable : memref<64x64xf32> to tensor<64x64xf32>
              %18 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 32768 + d1 * 512 + d2 * 64)>(%arg10, %arg7, %arg8)
              %reinterpret_cast_1 = memref.reinterpret_cast %arg1 to offset: [%18], sizes: [64, 64], strides: [512, 1] {tmd.reuse = {sequential = [{depth = 4 : i64, iterator = "%arg10", reuse_type = "no_reuse", volume = 0 : i64}], spatial = [{depth = 0 : i64, iterator = "%arg6", mapped_to = "x", reuse_type = "total_reuse", volume = 16384 : i64}, {depth = 1 : i64, iterator = "%arg7", mapped_to = "y", reuse_type = "no_reuse", volume = 0 : i64}], temporal = [{depth = 2 : i64, iterator = "%arg8", reuse_type = "no_reuse", volume = 0 : i64}, {depth = 3 : i64, iterator = "%arg9", reuse_type = "total_reuse", volume = 16384 : i64}]}} : memref<*xf32> to memref<64x64xf32, strided<[512, 1], offset: ?>>
              %alloc_2 = memref.alloc() {tmd.alloc = {local = true, memory_name = "L1", size = 16384 : i64}} : memref<64x64xf32>
              memref.copy %reinterpret_cast_1, %alloc_2 {tmd.copy.choice = {kind = "mem", memory_name = "L1"}} : memref<64x64xf32, strided<[512, 1], offset: ?>> to memref<64x64xf32>
              %19 = bufferization.to_tensor %alloc_2 restrict writable : memref<64x64xf32> to tensor<64x64xf32>
              %20 = linalg.matmul ins(%17, %19 : tensor<64x64xf32>, tensor<64x64xf32>) outs(%13 : tensor<64x64xf32>) -> tensor<64x64xf32>
              %21 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%arg11, %20 : tensor<64x64xf32>, tensor<64x64xf32>) outs(%arg11 : tensor<64x64xf32>) {
              ^bb0(%in: f32, %in_3: f32, %out: f32):
                %22 = arith.addf %in, %in_3 : f32
                linalg.yield %22 : f32
              } -> tensor<64x64xf32>
              scf.yield %21 : tensor<64x64xf32>
            }
            %15 = affine.apply affine_map<(d0, d1, d2, d3) -> (d0 * 512 + d1 * 64 + d2 * 262144 + d3 * 32768)>(%arg7, %arg8, %arg6, %arg9)
            %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%15], sizes: [64, 64], strides: [512, 1] {tmd.reuse = {spatial = [{depth = 0 : i64, iterator = "%arg6", mapped_to = "x", reuse_type = "no_reuse", volume = 0 : i64}, {depth = 1 : i64, iterator = "%arg7", mapped_to = "y", reuse_type = "no_reuse", volume = 0 : i64}], temporal = [{depth = 2 : i64, iterator = "%arg8", reuse_type = "no_reuse", volume = 0 : i64}, {depth = 3 : i64, iterator = "%arg9", reuse_type = "no_reuse", volume = 0 : i64}]}} : memref<*xf32> to memref<64x64xf32, strided<[512, 1], offset: ?>>
            bufferization.materialize_in_destination %14 in writable %reinterpret_cast : (tensor<64x64xf32>, memref<64x64xf32, strided<[512, 1], offset: ?>>) -> ()
          }
        }
      } {tmd.mapped_to = "y"}
    } {tmd.mapped_to = "x"}
    return
  }
  func.func @matmul_kernel__d0i0_d1i1__f10__c0by_c1mem(%arg0: memref<*xf32> {tt.divisibility = 16 : i32}, %arg1: memref<*xf32> {tt.divisibility = 16 : i32}, %arg2: memref<*xf32> {tt.divisibility = 16 : i32}, %arg3: index, %arg4: index, %arg5: index) {
    affine.parallel (%arg6) = (0) to (8) {
      affine.parallel (%arg7) = (0) to (8) {
        affine.for %arg8 = 0 to affine_map<()[s0, s1] -> (s1 ceildiv 8)>()[%arg3, %arg4] {
          affine.for %arg9 = 0 to affine_map<()[s0, s1] -> (s0 ceildiv 8)>()[%arg3, %arg4] {
            %cst = arith.constant 0.000000e+00 : f32
            %12 = tensor.empty() : tensor<64x64xf32>
            %13 = linalg.fill ins(%cst : f32) outs(%12 : tensor<64x64xf32>) -> tensor<64x64xf32>
            %c0 = arith.constant 0 : index
            %c8 = arith.constant 8 : index
            %c1 = arith.constant 1 : index
            %14 = scf.for %arg10 = %c0 to %c8 step %c1 iter_args(%arg11 = %13) -> (tensor<64x64xf32>) {
              %16 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 64 + d1 * 262144 + d2 * 32768)>(%arg10, %arg6, %arg9)
              %reinterpret_cast_0 = memref.reinterpret_cast %arg0 to offset: [%16], sizes: [64, 64], strides: [512, 1] {tmd.reuse = {sequential = [{depth = 4 : i64, iterator = "%arg10", reuse_type = "no_reuse", volume = 0 : i64}], spatial = [{depth = 0 : i64, iterator = "%arg6", mapped_to = "x", reuse_type = "no_reuse", volume = 0 : i64}, {depth = 1 : i64, iterator = "%arg7", mapped_to = "y", reuse_type = "total_reuse", volume = 16384 : i64}], temporal = [{depth = 2 : i64, iterator = "%arg8", reuse_type = "total_reuse", volume = 16384 : i64}, {depth = 3 : i64, iterator = "%arg9", reuse_type = "no_reuse", volume = 0 : i64}]}} : memref<*xf32> to memref<64x64xf32, strided<[512, 1], offset: ?>>
              %alloc = memref.alloc() {tmd.alloc = {local = true, memory_name = "L1", size = 16384 : i64}} : memref<64x64xf32>
              memref.copy %reinterpret_cast_0, %alloc {tmd.copy.choice = {dim = "y", interconnect_name = "vertical_links", kind = "broadcast"}} : memref<64x64xf32, strided<[512, 1], offset: ?>> to memref<64x64xf32>
              %17 = bufferization.to_tensor %alloc restrict writable : memref<64x64xf32> to tensor<64x64xf32>
              %18 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 32768 + d1 * 512 + d2 * 64)>(%arg10, %arg7, %arg8)
              %reinterpret_cast_1 = memref.reinterpret_cast %arg1 to offset: [%18], sizes: [64, 64], strides: [512, 1] {tmd.reuse = {sequential = [{depth = 4 : i64, iterator = "%arg10", reuse_type = "no_reuse", volume = 0 : i64}], spatial = [{depth = 0 : i64, iterator = "%arg6", mapped_to = "x", reuse_type = "total_reuse", volume = 16384 : i64}, {depth = 1 : i64, iterator = "%arg7", mapped_to = "y", reuse_type = "no_reuse", volume = 0 : i64}], temporal = [{depth = 2 : i64, iterator = "%arg8", reuse_type = "no_reuse", volume = 0 : i64}, {depth = 3 : i64, iterator = "%arg9", reuse_type = "total_reuse", volume = 16384 : i64}]}} : memref<*xf32> to memref<64x64xf32, strided<[512, 1], offset: ?>>
              %alloc_2 = memref.alloc() {tmd.alloc = {local = true, memory_name = "L1", size = 16384 : i64}} : memref<64x64xf32>
              memref.copy %reinterpret_cast_1, %alloc_2 {tmd.copy.choice = {kind = "mem", memory_name = "L1"}} : memref<64x64xf32, strided<[512, 1], offset: ?>> to memref<64x64xf32>
              %19 = bufferization.to_tensor %alloc_2 restrict writable : memref<64x64xf32> to tensor<64x64xf32>
              %20 = linalg.matmul ins(%17, %19 : tensor<64x64xf32>, tensor<64x64xf32>) outs(%13 : tensor<64x64xf32>) -> tensor<64x64xf32>
              %21 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%arg11, %20 : tensor<64x64xf32>, tensor<64x64xf32>) outs(%arg11 : tensor<64x64xf32>) {
              ^bb0(%in: f32, %in_3: f32, %out: f32):
                %22 = arith.addf %in, %in_3 : f32
                linalg.yield %22 : f32
              } -> tensor<64x64xf32>
              scf.yield %21 : tensor<64x64xf32>
            }
            %15 = affine.apply affine_map<(d0, d1, d2, d3) -> (d0 * 512 + d1 * 64 + d2 * 262144 + d3 * 32768)>(%arg7, %arg8, %arg6, %arg9)
            %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%15], sizes: [64, 64], strides: [512, 1] {tmd.reuse = {spatial = [{depth = 0 : i64, iterator = "%arg6", mapped_to = "x", reuse_type = "no_reuse", volume = 0 : i64}, {depth = 1 : i64, iterator = "%arg7", mapped_to = "y", reuse_type = "no_reuse", volume = 0 : i64}], temporal = [{depth = 2 : i64, iterator = "%arg8", reuse_type = "no_reuse", volume = 0 : i64}, {depth = 3 : i64, iterator = "%arg9", reuse_type = "no_reuse", volume = 0 : i64}]}} : memref<*xf32> to memref<64x64xf32, strided<[512, 1], offset: ?>>
            bufferization.materialize_in_destination %14 in writable %reinterpret_cast : (tensor<64x64xf32>, memref<64x64xf32, strided<[512, 1], offset: ?>>) -> ()
          }
        }
      } {tmd.mapped_to = "y"}
    } {tmd.mapped_to = "x"}
    return
  }
  func.func @matmul_kernel__d0i0_d1i1__f10__c0mem_c1bx(%arg0: memref<*xf32> {tt.divisibility = 16 : i32}, %arg1: memref<*xf32> {tt.divisibility = 16 : i32}, %arg2: memref<*xf32> {tt.divisibility = 16 : i32}, %arg3: index, %arg4: index, %arg5: index) {
    affine.parallel (%arg6) = (0) to (8) {
      affine.parallel (%arg7) = (0) to (8) {
        affine.for %arg8 = 0 to affine_map<()[s0, s1] -> (s1 ceildiv 8)>()[%arg3, %arg4] {
          affine.for %arg9 = 0 to affine_map<()[s0, s1] -> (s0 ceildiv 8)>()[%arg3, %arg4] {
            %cst = arith.constant 0.000000e+00 : f32
            %12 = tensor.empty() : tensor<64x64xf32>
            %13 = linalg.fill ins(%cst : f32) outs(%12 : tensor<64x64xf32>) -> tensor<64x64xf32>
            %c0 = arith.constant 0 : index
            %c8 = arith.constant 8 : index
            %c1 = arith.constant 1 : index
            %14 = scf.for %arg10 = %c0 to %c8 step %c1 iter_args(%arg11 = %13) -> (tensor<64x64xf32>) {
              %16 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 64 + d1 * 262144 + d2 * 32768)>(%arg10, %arg6, %arg9)
              %reinterpret_cast_0 = memref.reinterpret_cast %arg0 to offset: [%16], sizes: [64, 64], strides: [512, 1] {tmd.reuse = {sequential = [{depth = 4 : i64, iterator = "%arg10", reuse_type = "no_reuse", volume = 0 : i64}], spatial = [{depth = 0 : i64, iterator = "%arg6", mapped_to = "x", reuse_type = "no_reuse", volume = 0 : i64}, {depth = 1 : i64, iterator = "%arg7", mapped_to = "y", reuse_type = "total_reuse", volume = 16384 : i64}], temporal = [{depth = 2 : i64, iterator = "%arg8", reuse_type = "total_reuse", volume = 16384 : i64}, {depth = 3 : i64, iterator = "%arg9", reuse_type = "no_reuse", volume = 0 : i64}]}} : memref<*xf32> to memref<64x64xf32, strided<[512, 1], offset: ?>>
              %alloc = memref.alloc() {tmd.alloc = {local = true, memory_name = "L1", size = 16384 : i64}} : memref<64x64xf32>
              memref.copy %reinterpret_cast_0, %alloc {tmd.copy.choice = {kind = "mem", memory_name = "L1"}} : memref<64x64xf32, strided<[512, 1], offset: ?>> to memref<64x64xf32>
              %17 = bufferization.to_tensor %alloc restrict writable : memref<64x64xf32> to tensor<64x64xf32>
              %18 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 32768 + d1 * 512 + d2 * 64)>(%arg10, %arg7, %arg8)
              %reinterpret_cast_1 = memref.reinterpret_cast %arg1 to offset: [%18], sizes: [64, 64], strides: [512, 1] {tmd.reuse = {sequential = [{depth = 4 : i64, iterator = "%arg10", reuse_type = "no_reuse", volume = 0 : i64}], spatial = [{depth = 0 : i64, iterator = "%arg6", mapped_to = "x", reuse_type = "total_reuse", volume = 16384 : i64}, {depth = 1 : i64, iterator = "%arg7", mapped_to = "y", reuse_type = "no_reuse", volume = 0 : i64}], temporal = [{depth = 2 : i64, iterator = "%arg8", reuse_type = "no_reuse", volume = 0 : i64}, {depth = 3 : i64, iterator = "%arg9", reuse_type = "total_reuse", volume = 16384 : i64}]}} : memref<*xf32> to memref<64x64xf32, strided<[512, 1], offset: ?>>
              %alloc_2 = memref.alloc() {tmd.alloc = {local = true, memory_name = "L1", size = 16384 : i64}} : memref<64x64xf32>
              memref.copy %reinterpret_cast_1, %alloc_2 {tmd.copy.choice = {dim = "x", interconnect_name = "horizontal_links", kind = "broadcast"}} : memref<64x64xf32, strided<[512, 1], offset: ?>> to memref<64x64xf32>
              %19 = bufferization.to_tensor %alloc_2 restrict writable : memref<64x64xf32> to tensor<64x64xf32>
              %20 = linalg.matmul ins(%17, %19 : tensor<64x64xf32>, tensor<64x64xf32>) outs(%13 : tensor<64x64xf32>) -> tensor<64x64xf32>
              %21 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%arg11, %20 : tensor<64x64xf32>, tensor<64x64xf32>) outs(%arg11 : tensor<64x64xf32>) {
              ^bb0(%in: f32, %in_3: f32, %out: f32):
                %22 = arith.addf %in, %in_3 : f32
                linalg.yield %22 : f32
              } -> tensor<64x64xf32>
              scf.yield %21 : tensor<64x64xf32>
            }
            %15 = affine.apply affine_map<(d0, d1, d2, d3) -> (d0 * 512 + d1 * 64 + d2 * 262144 + d3 * 32768)>(%arg7, %arg8, %arg6, %arg9)
            %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%15], sizes: [64, 64], strides: [512, 1] {tmd.reuse = {spatial = [{depth = 0 : i64, iterator = "%arg6", mapped_to = "x", reuse_type = "no_reuse", volume = 0 : i64}, {depth = 1 : i64, iterator = "%arg7", mapped_to = "y", reuse_type = "no_reuse", volume = 0 : i64}], temporal = [{depth = 2 : i64, iterator = "%arg8", reuse_type = "no_reuse", volume = 0 : i64}, {depth = 3 : i64, iterator = "%arg9", reuse_type = "no_reuse", volume = 0 : i64}]}} : memref<*xf32> to memref<64x64xf32, strided<[512, 1], offset: ?>>
            bufferization.materialize_in_destination %14 in writable %reinterpret_cast : (tensor<64x64xf32>, memref<64x64xf32, strided<[512, 1], offset: ?>>) -> ()
          }
        }
      } {tmd.mapped_to = "y"}
    } {tmd.mapped_to = "x"}
    return
  }
  func.func @matmul_kernel__d0i0_d1i1__f10__c0by_c1bx(%arg0: memref<*xf32> {tt.divisibility = 16 : i32}, %arg1: memref<*xf32> {tt.divisibility = 16 : i32}, %arg2: memref<*xf32> {tt.divisibility = 16 : i32}, %arg3: index, %arg4: index, %arg5: index) {
    affine.parallel (%arg6) = (0) to (8) {
      affine.parallel (%arg7) = (0) to (8) {
        affine.for %arg8 = 0 to affine_map<()[s0, s1] -> (s1 ceildiv 8)>()[%arg3, %arg4] {
          affine.for %arg9 = 0 to affine_map<()[s0, s1] -> (s0 ceildiv 8)>()[%arg3, %arg4] {
            %cst = arith.constant 0.000000e+00 : f32
            %12 = tensor.empty() : tensor<64x64xf32>
            %13 = linalg.fill ins(%cst : f32) outs(%12 : tensor<64x64xf32>) -> tensor<64x64xf32>
            %c0 = arith.constant 0 : index
            %c8 = arith.constant 8 : index
            %c1 = arith.constant 1 : index
            %14 = scf.for %arg10 = %c0 to %c8 step %c1 iter_args(%arg11 = %13) -> (tensor<64x64xf32>) {
              %16 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 64 + d1 * 262144 + d2 * 32768)>(%arg10, %arg6, %arg9)
              %reinterpret_cast_0 = memref.reinterpret_cast %arg0 to offset: [%16], sizes: [64, 64], strides: [512, 1] {tmd.reuse = {sequential = [{depth = 4 : i64, iterator = "%arg10", reuse_type = "no_reuse", volume = 0 : i64}], spatial = [{depth = 0 : i64, iterator = "%arg6", mapped_to = "x", reuse_type = "no_reuse", volume = 0 : i64}, {depth = 1 : i64, iterator = "%arg7", mapped_to = "y", reuse_type = "total_reuse", volume = 16384 : i64}], temporal = [{depth = 2 : i64, iterator = "%arg8", reuse_type = "total_reuse", volume = 16384 : i64}, {depth = 3 : i64, iterator = "%arg9", reuse_type = "no_reuse", volume = 0 : i64}]}} : memref<*xf32> to memref<64x64xf32, strided<[512, 1], offset: ?>>
              %alloc = memref.alloc() {tmd.alloc = {local = true, memory_name = "L1", size = 16384 : i64}} : memref<64x64xf32>
              memref.copy %reinterpret_cast_0, %alloc {tmd.copy.choice = {dim = "y", interconnect_name = "vertical_links", kind = "broadcast"}} : memref<64x64xf32, strided<[512, 1], offset: ?>> to memref<64x64xf32>
              %17 = bufferization.to_tensor %alloc restrict writable : memref<64x64xf32> to tensor<64x64xf32>
              %18 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 32768 + d1 * 512 + d2 * 64)>(%arg10, %arg7, %arg8)
              %reinterpret_cast_1 = memref.reinterpret_cast %arg1 to offset: [%18], sizes: [64, 64], strides: [512, 1] {tmd.reuse = {sequential = [{depth = 4 : i64, iterator = "%arg10", reuse_type = "no_reuse", volume = 0 : i64}], spatial = [{depth = 0 : i64, iterator = "%arg6", mapped_to = "x", reuse_type = "total_reuse", volume = 16384 : i64}, {depth = 1 : i64, iterator = "%arg7", mapped_to = "y", reuse_type = "no_reuse", volume = 0 : i64}], temporal = [{depth = 2 : i64, iterator = "%arg8", reuse_type = "no_reuse", volume = 0 : i64}, {depth = 3 : i64, iterator = "%arg9", reuse_type = "total_reuse", volume = 16384 : i64}]}} : memref<*xf32> to memref<64x64xf32, strided<[512, 1], offset: ?>>
              %alloc_2 = memref.alloc() {tmd.alloc = {local = true, memory_name = "L1", size = 16384 : i64}} : memref<64x64xf32>
              memref.copy %reinterpret_cast_1, %alloc_2 {tmd.copy.choice = {dim = "x", interconnect_name = "horizontal_links", kind = "broadcast"}} : memref<64x64xf32, strided<[512, 1], offset: ?>> to memref<64x64xf32>
              %19 = bufferization.to_tensor %alloc_2 restrict writable : memref<64x64xf32> to tensor<64x64xf32>
              %20 = linalg.matmul ins(%17, %19 : tensor<64x64xf32>, tensor<64x64xf32>) outs(%13 : tensor<64x64xf32>) -> tensor<64x64xf32>
              %21 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%arg11, %20 : tensor<64x64xf32>, tensor<64x64xf32>) outs(%arg11 : tensor<64x64xf32>) {
              ^bb0(%in: f32, %in_3: f32, %out: f32):
                %22 = arith.addf %in, %in_3 : f32
                linalg.yield %22 : f32
              } -> tensor<64x64xf32>
              scf.yield %21 : tensor<64x64xf32>
            }
            %15 = affine.apply affine_map<(d0, d1, d2, d3) -> (d0 * 512 + d1 * 64 + d2 * 262144 + d3 * 32768)>(%arg7, %arg8, %arg6, %arg9)
            %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%15], sizes: [64, 64], strides: [512, 1] {tmd.reuse = {spatial = [{depth = 0 : i64, iterator = "%arg6", mapped_to = "x", reuse_type = "no_reuse", volume = 0 : i64}, {depth = 1 : i64, iterator = "%arg7", mapped_to = "y", reuse_type = "no_reuse", volume = 0 : i64}], temporal = [{depth = 2 : i64, iterator = "%arg8", reuse_type = "no_reuse", volume = 0 : i64}, {depth = 3 : i64, iterator = "%arg9", reuse_type = "no_reuse", volume = 0 : i64}]}} : memref<*xf32> to memref<64x64xf32, strided<[512, 1], offset: ?>>
            bufferization.materialize_in_destination %14 in writable %reinterpret_cast : (tensor<64x64xf32>, memref<64x64xf32, strided<[512, 1], offset: ?>>) -> ()
          }
        }
      } {tmd.mapped_to = "y"}
    } {tmd.mapped_to = "x"}
    return
  }
  func.func @matmul_kernel__d1i0_d0i1__f01__c0mem_c1mem(%arg0: memref<*xf32> {tt.divisibility = 16 : i32}, %arg1: memref<*xf32> {tt.divisibility = 16 : i32}, %arg2: memref<*xf32> {tt.divisibility = 16 : i32}, %arg3: index, %arg4: index, %arg5: index) {
    affine.parallel (%arg6) = (0) to (8) {
      affine.parallel (%arg7) = (0) to (8) {
        affine.for %arg8 = 0 to affine_map<()[s0, s1] -> (s0 ceildiv 8)>()[%arg3, %arg4] {
          affine.for %arg9 = 0 to affine_map<()[s0, s1] -> (s1 ceildiv 8)>()[%arg3, %arg4] {
            %cst = arith.constant 0.000000e+00 : f32
            %12 = tensor.empty() : tensor<64x64xf32>
            %13 = linalg.fill ins(%cst : f32) outs(%12 : tensor<64x64xf32>) -> tensor<64x64xf32>
            %c0 = arith.constant 0 : index
            %c8 = arith.constant 8 : index
            %c1 = arith.constant 1 : index
            %14 = scf.for %arg10 = %c0 to %c8 step %c1 iter_args(%arg11 = %13) -> (tensor<64x64xf32>) {
              %16 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 64 + d1 * 262144 + d2 * 32768)>(%arg10, %arg6, %arg8)
              %reinterpret_cast_0 = memref.reinterpret_cast %arg0 to offset: [%16], sizes: [64, 64], strides: [512, 1] {tmd.reuse = {sequential = [{depth = 4 : i64, iterator = "%arg10", reuse_type = "no_reuse", volume = 0 : i64}], spatial = [{depth = 0 : i64, iterator = "%arg6", mapped_to = "y", reuse_type = "no_reuse", volume = 0 : i64}, {depth = 1 : i64, iterator = "%arg7", mapped_to = "x", reuse_type = "total_reuse", volume = 16384 : i64}], temporal = [{depth = 2 : i64, iterator = "%arg8", reuse_type = "no_reuse", volume = 0 : i64}, {depth = 3 : i64, iterator = "%arg9", reuse_type = "total_reuse", volume = 16384 : i64}]}} : memref<*xf32> to memref<64x64xf32, strided<[512, 1], offset: ?>>
              %alloc = memref.alloc() {tmd.alloc = {local = true, memory_name = "L1", size = 16384 : i64}} : memref<64x64xf32>
              memref.copy %reinterpret_cast_0, %alloc {tmd.copy.choice = {kind = "mem", memory_name = "L1"}} : memref<64x64xf32, strided<[512, 1], offset: ?>> to memref<64x64xf32>
              %17 = bufferization.to_tensor %alloc restrict writable : memref<64x64xf32> to tensor<64x64xf32>
              %18 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 32768 + d1 * 512 + d2 * 64)>(%arg10, %arg7, %arg9)
              %reinterpret_cast_1 = memref.reinterpret_cast %arg1 to offset: [%18], sizes: [64, 64], strides: [512, 1] {tmd.reuse = {sequential = [{depth = 4 : i64, iterator = "%arg10", reuse_type = "no_reuse", volume = 0 : i64}], spatial = [{depth = 0 : i64, iterator = "%arg6", mapped_to = "y", reuse_type = "total_reuse", volume = 16384 : i64}, {depth = 1 : i64, iterator = "%arg7", mapped_to = "x", reuse_type = "no_reuse", volume = 0 : i64}], temporal = [{depth = 2 : i64, iterator = "%arg8", reuse_type = "total_reuse", volume = 16384 : i64}, {depth = 3 : i64, iterator = "%arg9", reuse_type = "no_reuse", volume = 0 : i64}]}} : memref<*xf32> to memref<64x64xf32, strided<[512, 1], offset: ?>>
              %alloc_2 = memref.alloc() {tmd.alloc = {local = true, memory_name = "L1", size = 16384 : i64}} : memref<64x64xf32>
              memref.copy %reinterpret_cast_1, %alloc_2 {tmd.copy.choice = {kind = "mem", memory_name = "L1"}} : memref<64x64xf32, strided<[512, 1], offset: ?>> to memref<64x64xf32>
              %19 = bufferization.to_tensor %alloc_2 restrict writable : memref<64x64xf32> to tensor<64x64xf32>
              %20 = linalg.matmul ins(%17, %19 : tensor<64x64xf32>, tensor<64x64xf32>) outs(%13 : tensor<64x64xf32>) -> tensor<64x64xf32>
              %21 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%arg11, %20 : tensor<64x64xf32>, tensor<64x64xf32>) outs(%arg11 : tensor<64x64xf32>) {
              ^bb0(%in: f32, %in_3: f32, %out: f32):
                %22 = arith.addf %in, %in_3 : f32
                linalg.yield %22 : f32
              } -> tensor<64x64xf32>
              scf.yield %21 : tensor<64x64xf32>
            }
            %15 = affine.apply affine_map<(d0, d1, d2, d3) -> (d0 * 512 + d1 * 64 + d2 * 262144 + d3 * 32768)>(%arg7, %arg9, %arg6, %arg8)
            %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%15], sizes: [64, 64], strides: [512, 1] {tmd.reuse = {spatial = [{depth = 0 : i64, iterator = "%arg6", mapped_to = "y", reuse_type = "no_reuse", volume = 0 : i64}, {depth = 1 : i64, iterator = "%arg7", mapped_to = "x", reuse_type = "no_reuse", volume = 0 : i64}], temporal = [{depth = 2 : i64, iterator = "%arg8", reuse_type = "no_reuse", volume = 0 : i64}, {depth = 3 : i64, iterator = "%arg9", reuse_type = "no_reuse", volume = 0 : i64}]}} : memref<*xf32> to memref<64x64xf32, strided<[512, 1], offset: ?>>
            bufferization.materialize_in_destination %14 in writable %reinterpret_cast : (tensor<64x64xf32>, memref<64x64xf32, strided<[512, 1], offset: ?>>) -> ()
          }
        }
      } {tmd.mapped_to = "x"}
    } {tmd.mapped_to = "y"}
    return
  }
  func.func @matmul_kernel__d1i0_d0i1__f01__c0bx_c1mem(%arg0: memref<*xf32> {tt.divisibility = 16 : i32}, %arg1: memref<*xf32> {tt.divisibility = 16 : i32}, %arg2: memref<*xf32> {tt.divisibility = 16 : i32}, %arg3: index, %arg4: index, %arg5: index) {
    affine.parallel (%arg6) = (0) to (8) {
      affine.parallel (%arg7) = (0) to (8) {
        affine.for %arg8 = 0 to affine_map<()[s0, s1] -> (s0 ceildiv 8)>()[%arg3, %arg4] {
          affine.for %arg9 = 0 to affine_map<()[s0, s1] -> (s1 ceildiv 8)>()[%arg3, %arg4] {
            %cst = arith.constant 0.000000e+00 : f32
            %12 = tensor.empty() : tensor<64x64xf32>
            %13 = linalg.fill ins(%cst : f32) outs(%12 : tensor<64x64xf32>) -> tensor<64x64xf32>
            %c0 = arith.constant 0 : index
            %c8 = arith.constant 8 : index
            %c1 = arith.constant 1 : index
            %14 = scf.for %arg10 = %c0 to %c8 step %c1 iter_args(%arg11 = %13) -> (tensor<64x64xf32>) {
              %16 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 64 + d1 * 262144 + d2 * 32768)>(%arg10, %arg6, %arg8)
              %reinterpret_cast_0 = memref.reinterpret_cast %arg0 to offset: [%16], sizes: [64, 64], strides: [512, 1] {tmd.reuse = {sequential = [{depth = 4 : i64, iterator = "%arg10", reuse_type = "no_reuse", volume = 0 : i64}], spatial = [{depth = 0 : i64, iterator = "%arg6", mapped_to = "y", reuse_type = "no_reuse", volume = 0 : i64}, {depth = 1 : i64, iterator = "%arg7", mapped_to = "x", reuse_type = "total_reuse", volume = 16384 : i64}], temporal = [{depth = 2 : i64, iterator = "%arg8", reuse_type = "no_reuse", volume = 0 : i64}, {depth = 3 : i64, iterator = "%arg9", reuse_type = "total_reuse", volume = 16384 : i64}]}} : memref<*xf32> to memref<64x64xf32, strided<[512, 1], offset: ?>>
              %alloc = memref.alloc() {tmd.alloc = {local = true, memory_name = "L1", size = 16384 : i64}} : memref<64x64xf32>
              memref.copy %reinterpret_cast_0, %alloc {tmd.copy.choice = {dim = "x", interconnect_name = "horizontal_links", kind = "broadcast"}} : memref<64x64xf32, strided<[512, 1], offset: ?>> to memref<64x64xf32>
              %17 = bufferization.to_tensor %alloc restrict writable : memref<64x64xf32> to tensor<64x64xf32>
              %18 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 32768 + d1 * 512 + d2 * 64)>(%arg10, %arg7, %arg9)
              %reinterpret_cast_1 = memref.reinterpret_cast %arg1 to offset: [%18], sizes: [64, 64], strides: [512, 1] {tmd.reuse = {sequential = [{depth = 4 : i64, iterator = "%arg10", reuse_type = "no_reuse", volume = 0 : i64}], spatial = [{depth = 0 : i64, iterator = "%arg6", mapped_to = "y", reuse_type = "total_reuse", volume = 16384 : i64}, {depth = 1 : i64, iterator = "%arg7", mapped_to = "x", reuse_type = "no_reuse", volume = 0 : i64}], temporal = [{depth = 2 : i64, iterator = "%arg8", reuse_type = "total_reuse", volume = 16384 : i64}, {depth = 3 : i64, iterator = "%arg9", reuse_type = "no_reuse", volume = 0 : i64}]}} : memref<*xf32> to memref<64x64xf32, strided<[512, 1], offset: ?>>
              %alloc_2 = memref.alloc() {tmd.alloc = {local = true, memory_name = "L1", size = 16384 : i64}} : memref<64x64xf32>
              memref.copy %reinterpret_cast_1, %alloc_2 {tmd.copy.choice = {kind = "mem", memory_name = "L1"}} : memref<64x64xf32, strided<[512, 1], offset: ?>> to memref<64x64xf32>
              %19 = bufferization.to_tensor %alloc_2 restrict writable : memref<64x64xf32> to tensor<64x64xf32>
              %20 = linalg.matmul ins(%17, %19 : tensor<64x64xf32>, tensor<64x64xf32>) outs(%13 : tensor<64x64xf32>) -> tensor<64x64xf32>
              %21 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%arg11, %20 : tensor<64x64xf32>, tensor<64x64xf32>) outs(%arg11 : tensor<64x64xf32>) {
              ^bb0(%in: f32, %in_3: f32, %out: f32):
                %22 = arith.addf %in, %in_3 : f32
                linalg.yield %22 : f32
              } -> tensor<64x64xf32>
              scf.yield %21 : tensor<64x64xf32>
            }
            %15 = affine.apply affine_map<(d0, d1, d2, d3) -> (d0 * 512 + d1 * 64 + d2 * 262144 + d3 * 32768)>(%arg7, %arg9, %arg6, %arg8)
            %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%15], sizes: [64, 64], strides: [512, 1] {tmd.reuse = {spatial = [{depth = 0 : i64, iterator = "%arg6", mapped_to = "y", reuse_type = "no_reuse", volume = 0 : i64}, {depth = 1 : i64, iterator = "%arg7", mapped_to = "x", reuse_type = "no_reuse", volume = 0 : i64}], temporal = [{depth = 2 : i64, iterator = "%arg8", reuse_type = "no_reuse", volume = 0 : i64}, {depth = 3 : i64, iterator = "%arg9", reuse_type = "no_reuse", volume = 0 : i64}]}} : memref<*xf32> to memref<64x64xf32, strided<[512, 1], offset: ?>>
            bufferization.materialize_in_destination %14 in writable %reinterpret_cast : (tensor<64x64xf32>, memref<64x64xf32, strided<[512, 1], offset: ?>>) -> ()
          }
        }
      } {tmd.mapped_to = "x"}
    } {tmd.mapped_to = "y"}
    return
  }
  func.func @matmul_kernel__d1i0_d0i1__f01__c0mem_c1by(%arg0: memref<*xf32> {tt.divisibility = 16 : i32}, %arg1: memref<*xf32> {tt.divisibility = 16 : i32}, %arg2: memref<*xf32> {tt.divisibility = 16 : i32}, %arg3: index, %arg4: index, %arg5: index) {
    affine.parallel (%arg6) = (0) to (8) {
      affine.parallel (%arg7) = (0) to (8) {
        affine.for %arg8 = 0 to affine_map<()[s0, s1] -> (s0 ceildiv 8)>()[%arg3, %arg4] {
          affine.for %arg9 = 0 to affine_map<()[s0, s1] -> (s1 ceildiv 8)>()[%arg3, %arg4] {
            %cst = arith.constant 0.000000e+00 : f32
            %12 = tensor.empty() : tensor<64x64xf32>
            %13 = linalg.fill ins(%cst : f32) outs(%12 : tensor<64x64xf32>) -> tensor<64x64xf32>
            %c0 = arith.constant 0 : index
            %c8 = arith.constant 8 : index
            %c1 = arith.constant 1 : index
            %14 = scf.for %arg10 = %c0 to %c8 step %c1 iter_args(%arg11 = %13) -> (tensor<64x64xf32>) {
              %16 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 64 + d1 * 262144 + d2 * 32768)>(%arg10, %arg6, %arg8)
              %reinterpret_cast_0 = memref.reinterpret_cast %arg0 to offset: [%16], sizes: [64, 64], strides: [512, 1] {tmd.reuse = {sequential = [{depth = 4 : i64, iterator = "%arg10", reuse_type = "no_reuse", volume = 0 : i64}], spatial = [{depth = 0 : i64, iterator = "%arg6", mapped_to = "y", reuse_type = "no_reuse", volume = 0 : i64}, {depth = 1 : i64, iterator = "%arg7", mapped_to = "x", reuse_type = "total_reuse", volume = 16384 : i64}], temporal = [{depth = 2 : i64, iterator = "%arg8", reuse_type = "no_reuse", volume = 0 : i64}, {depth = 3 : i64, iterator = "%arg9", reuse_type = "total_reuse", volume = 16384 : i64}]}} : memref<*xf32> to memref<64x64xf32, strided<[512, 1], offset: ?>>
              %alloc = memref.alloc() {tmd.alloc = {local = true, memory_name = "L1", size = 16384 : i64}} : memref<64x64xf32>
              memref.copy %reinterpret_cast_0, %alloc {tmd.copy.choice = {kind = "mem", memory_name = "L1"}} : memref<64x64xf32, strided<[512, 1], offset: ?>> to memref<64x64xf32>
              %17 = bufferization.to_tensor %alloc restrict writable : memref<64x64xf32> to tensor<64x64xf32>
              %18 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 32768 + d1 * 512 + d2 * 64)>(%arg10, %arg7, %arg9)
              %reinterpret_cast_1 = memref.reinterpret_cast %arg1 to offset: [%18], sizes: [64, 64], strides: [512, 1] {tmd.reuse = {sequential = [{depth = 4 : i64, iterator = "%arg10", reuse_type = "no_reuse", volume = 0 : i64}], spatial = [{depth = 0 : i64, iterator = "%arg6", mapped_to = "y", reuse_type = "total_reuse", volume = 16384 : i64}, {depth = 1 : i64, iterator = "%arg7", mapped_to = "x", reuse_type = "no_reuse", volume = 0 : i64}], temporal = [{depth = 2 : i64, iterator = "%arg8", reuse_type = "total_reuse", volume = 16384 : i64}, {depth = 3 : i64, iterator = "%arg9", reuse_type = "no_reuse", volume = 0 : i64}]}} : memref<*xf32> to memref<64x64xf32, strided<[512, 1], offset: ?>>
              %alloc_2 = memref.alloc() {tmd.alloc = {local = true, memory_name = "L1", size = 16384 : i64}} : memref<64x64xf32>
              memref.copy %reinterpret_cast_1, %alloc_2 {tmd.copy.choice = {dim = "y", interconnect_name = "vertical_links", kind = "broadcast"}} : memref<64x64xf32, strided<[512, 1], offset: ?>> to memref<64x64xf32>
              %19 = bufferization.to_tensor %alloc_2 restrict writable : memref<64x64xf32> to tensor<64x64xf32>
              %20 = linalg.matmul ins(%17, %19 : tensor<64x64xf32>, tensor<64x64xf32>) outs(%13 : tensor<64x64xf32>) -> tensor<64x64xf32>
              %21 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%arg11, %20 : tensor<64x64xf32>, tensor<64x64xf32>) outs(%arg11 : tensor<64x64xf32>) {
              ^bb0(%in: f32, %in_3: f32, %out: f32):
                %22 = arith.addf %in, %in_3 : f32
                linalg.yield %22 : f32
              } -> tensor<64x64xf32>
              scf.yield %21 : tensor<64x64xf32>
            }
            %15 = affine.apply affine_map<(d0, d1, d2, d3) -> (d0 * 512 + d1 * 64 + d2 * 262144 + d3 * 32768)>(%arg7, %arg9, %arg6, %arg8)
            %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%15], sizes: [64, 64], strides: [512, 1] {tmd.reuse = {spatial = [{depth = 0 : i64, iterator = "%arg6", mapped_to = "y", reuse_type = "no_reuse", volume = 0 : i64}, {depth = 1 : i64, iterator = "%arg7", mapped_to = "x", reuse_type = "no_reuse", volume = 0 : i64}], temporal = [{depth = 2 : i64, iterator = "%arg8", reuse_type = "no_reuse", volume = 0 : i64}, {depth = 3 : i64, iterator = "%arg9", reuse_type = "no_reuse", volume = 0 : i64}]}} : memref<*xf32> to memref<64x64xf32, strided<[512, 1], offset: ?>>
            bufferization.materialize_in_destination %14 in writable %reinterpret_cast : (tensor<64x64xf32>, memref<64x64xf32, strided<[512, 1], offset: ?>>) -> ()
          }
        }
      } {tmd.mapped_to = "x"}
    } {tmd.mapped_to = "y"}
    return
  }
  func.func @matmul_kernel__d1i0_d0i1__f01__c0bx_c1by(%arg0: memref<*xf32> {tt.divisibility = 16 : i32}, %arg1: memref<*xf32> {tt.divisibility = 16 : i32}, %arg2: memref<*xf32> {tt.divisibility = 16 : i32}, %arg3: index, %arg4: index, %arg5: index) {
    affine.parallel (%arg6) = (0) to (8) {
      affine.parallel (%arg7) = (0) to (8) {
        affine.for %arg8 = 0 to affine_map<()[s0, s1] -> (s0 ceildiv 8)>()[%arg3, %arg4] {
          affine.for %arg9 = 0 to affine_map<()[s0, s1] -> (s1 ceildiv 8)>()[%arg3, %arg4] {
            %cst = arith.constant 0.000000e+00 : f32
            %12 = tensor.empty() : tensor<64x64xf32>
            %13 = linalg.fill ins(%cst : f32) outs(%12 : tensor<64x64xf32>) -> tensor<64x64xf32>
            %c0 = arith.constant 0 : index
            %c8 = arith.constant 8 : index
            %c1 = arith.constant 1 : index
            %14 = scf.for %arg10 = %c0 to %c8 step %c1 iter_args(%arg11 = %13) -> (tensor<64x64xf32>) {
              %16 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 64 + d1 * 262144 + d2 * 32768)>(%arg10, %arg6, %arg8)
              %reinterpret_cast_0 = memref.reinterpret_cast %arg0 to offset: [%16], sizes: [64, 64], strides: [512, 1] {tmd.reuse = {sequential = [{depth = 4 : i64, iterator = "%arg10", reuse_type = "no_reuse", volume = 0 : i64}], spatial = [{depth = 0 : i64, iterator = "%arg6", mapped_to = "y", reuse_type = "no_reuse", volume = 0 : i64}, {depth = 1 : i64, iterator = "%arg7", mapped_to = "x", reuse_type = "total_reuse", volume = 16384 : i64}], temporal = [{depth = 2 : i64, iterator = "%arg8", reuse_type = "no_reuse", volume = 0 : i64}, {depth = 3 : i64, iterator = "%arg9", reuse_type = "total_reuse", volume = 16384 : i64}]}} : memref<*xf32> to memref<64x64xf32, strided<[512, 1], offset: ?>>
              %alloc = memref.alloc() {tmd.alloc = {local = true, memory_name = "L1", size = 16384 : i64}} : memref<64x64xf32>
              memref.copy %reinterpret_cast_0, %alloc {tmd.copy.choice = {dim = "x", interconnect_name = "horizontal_links", kind = "broadcast"}} : memref<64x64xf32, strided<[512, 1], offset: ?>> to memref<64x64xf32>
              %17 = bufferization.to_tensor %alloc restrict writable : memref<64x64xf32> to tensor<64x64xf32>
              %18 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 32768 + d1 * 512 + d2 * 64)>(%arg10, %arg7, %arg9)
              %reinterpret_cast_1 = memref.reinterpret_cast %arg1 to offset: [%18], sizes: [64, 64], strides: [512, 1] {tmd.reuse = {sequential = [{depth = 4 : i64, iterator = "%arg10", reuse_type = "no_reuse", volume = 0 : i64}], spatial = [{depth = 0 : i64, iterator = "%arg6", mapped_to = "y", reuse_type = "total_reuse", volume = 16384 : i64}, {depth = 1 : i64, iterator = "%arg7", mapped_to = "x", reuse_type = "no_reuse", volume = 0 : i64}], temporal = [{depth = 2 : i64, iterator = "%arg8", reuse_type = "total_reuse", volume = 16384 : i64}, {depth = 3 : i64, iterator = "%arg9", reuse_type = "no_reuse", volume = 0 : i64}]}} : memref<*xf32> to memref<64x64xf32, strided<[512, 1], offset: ?>>
              %alloc_2 = memref.alloc() {tmd.alloc = {local = true, memory_name = "L1", size = 16384 : i64}} : memref<64x64xf32>
              memref.copy %reinterpret_cast_1, %alloc_2 {tmd.copy.choice = {dim = "y", interconnect_name = "vertical_links", kind = "broadcast"}} : memref<64x64xf32, strided<[512, 1], offset: ?>> to memref<64x64xf32>
              %19 = bufferization.to_tensor %alloc_2 restrict writable : memref<64x64xf32> to tensor<64x64xf32>
              %20 = linalg.matmul ins(%17, %19 : tensor<64x64xf32>, tensor<64x64xf32>) outs(%13 : tensor<64x64xf32>) -> tensor<64x64xf32>
              %21 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%arg11, %20 : tensor<64x64xf32>, tensor<64x64xf32>) outs(%arg11 : tensor<64x64xf32>) {
              ^bb0(%in: f32, %in_3: f32, %out: f32):
                %22 = arith.addf %in, %in_3 : f32
                linalg.yield %22 : f32
              } -> tensor<64x64xf32>
              scf.yield %21 : tensor<64x64xf32>
            }
            %15 = affine.apply affine_map<(d0, d1, d2, d3) -> (d0 * 512 + d1 * 64 + d2 * 262144 + d3 * 32768)>(%arg7, %arg9, %arg6, %arg8)
            %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%15], sizes: [64, 64], strides: [512, 1] {tmd.reuse = {spatial = [{depth = 0 : i64, iterator = "%arg6", mapped_to = "y", reuse_type = "no_reuse", volume = 0 : i64}, {depth = 1 : i64, iterator = "%arg7", mapped_to = "x", reuse_type = "no_reuse", volume = 0 : i64}], temporal = [{depth = 2 : i64, iterator = "%arg8", reuse_type = "no_reuse", volume = 0 : i64}, {depth = 3 : i64, iterator = "%arg9", reuse_type = "no_reuse", volume = 0 : i64}]}} : memref<*xf32> to memref<64x64xf32, strided<[512, 1], offset: ?>>
            bufferization.materialize_in_destination %14 in writable %reinterpret_cast : (tensor<64x64xf32>, memref<64x64xf32, strided<[512, 1], offset: ?>>) -> ()
          }
        }
      } {tmd.mapped_to = "x"}
    } {tmd.mapped_to = "y"}
    return
  }
  func.func @matmul_kernel__d1i0_d0i1__f10__c0mem_c1mem(%arg0: memref<*xf32> {tt.divisibility = 16 : i32}, %arg1: memref<*xf32> {tt.divisibility = 16 : i32}, %arg2: memref<*xf32> {tt.divisibility = 16 : i32}, %arg3: index, %arg4: index, %arg5: index) {
    affine.parallel (%arg6) = (0) to (8) {
      affine.parallel (%arg7) = (0) to (8) {
        affine.for %arg8 = 0 to affine_map<()[s0, s1] -> (s1 ceildiv 8)>()[%arg3, %arg4] {
          affine.for %arg9 = 0 to affine_map<()[s0, s1] -> (s0 ceildiv 8)>()[%arg3, %arg4] {
            %cst = arith.constant 0.000000e+00 : f32
            %12 = tensor.empty() : tensor<64x64xf32>
            %13 = linalg.fill ins(%cst : f32) outs(%12 : tensor<64x64xf32>) -> tensor<64x64xf32>
            %c0 = arith.constant 0 : index
            %c8 = arith.constant 8 : index
            %c1 = arith.constant 1 : index
            %14 = scf.for %arg10 = %c0 to %c8 step %c1 iter_args(%arg11 = %13) -> (tensor<64x64xf32>) {
              %16 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 64 + d1 * 262144 + d2 * 32768)>(%arg10, %arg6, %arg9)
              %reinterpret_cast_0 = memref.reinterpret_cast %arg0 to offset: [%16], sizes: [64, 64], strides: [512, 1] {tmd.reuse = {sequential = [{depth = 4 : i64, iterator = "%arg10", reuse_type = "no_reuse", volume = 0 : i64}], spatial = [{depth = 0 : i64, iterator = "%arg6", mapped_to = "y", reuse_type = "no_reuse", volume = 0 : i64}, {depth = 1 : i64, iterator = "%arg7", mapped_to = "x", reuse_type = "total_reuse", volume = 16384 : i64}], temporal = [{depth = 2 : i64, iterator = "%arg8", reuse_type = "total_reuse", volume = 16384 : i64}, {depth = 3 : i64, iterator = "%arg9", reuse_type = "no_reuse", volume = 0 : i64}]}} : memref<*xf32> to memref<64x64xf32, strided<[512, 1], offset: ?>>
              %alloc = memref.alloc() {tmd.alloc = {local = true, memory_name = "L1", size = 16384 : i64}} : memref<64x64xf32>
              memref.copy %reinterpret_cast_0, %alloc {tmd.copy.choice = {kind = "mem", memory_name = "L1"}} : memref<64x64xf32, strided<[512, 1], offset: ?>> to memref<64x64xf32>
              %17 = bufferization.to_tensor %alloc restrict writable : memref<64x64xf32> to tensor<64x64xf32>
              %18 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 32768 + d1 * 512 + d2 * 64)>(%arg10, %arg7, %arg8)
              %reinterpret_cast_1 = memref.reinterpret_cast %arg1 to offset: [%18], sizes: [64, 64], strides: [512, 1] {tmd.reuse = {sequential = [{depth = 4 : i64, iterator = "%arg10", reuse_type = "no_reuse", volume = 0 : i64}], spatial = [{depth = 0 : i64, iterator = "%arg6", mapped_to = "y", reuse_type = "total_reuse", volume = 16384 : i64}, {depth = 1 : i64, iterator = "%arg7", mapped_to = "x", reuse_type = "no_reuse", volume = 0 : i64}], temporal = [{depth = 2 : i64, iterator = "%arg8", reuse_type = "no_reuse", volume = 0 : i64}, {depth = 3 : i64, iterator = "%arg9", reuse_type = "total_reuse", volume = 16384 : i64}]}} : memref<*xf32> to memref<64x64xf32, strided<[512, 1], offset: ?>>
              %alloc_2 = memref.alloc() {tmd.alloc = {local = true, memory_name = "L1", size = 16384 : i64}} : memref<64x64xf32>
              memref.copy %reinterpret_cast_1, %alloc_2 {tmd.copy.choice = {kind = "mem", memory_name = "L1"}} : memref<64x64xf32, strided<[512, 1], offset: ?>> to memref<64x64xf32>
              %19 = bufferization.to_tensor %alloc_2 restrict writable : memref<64x64xf32> to tensor<64x64xf32>
              %20 = linalg.matmul ins(%17, %19 : tensor<64x64xf32>, tensor<64x64xf32>) outs(%13 : tensor<64x64xf32>) -> tensor<64x64xf32>
              %21 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%arg11, %20 : tensor<64x64xf32>, tensor<64x64xf32>) outs(%arg11 : tensor<64x64xf32>) {
              ^bb0(%in: f32, %in_3: f32, %out: f32):
                %22 = arith.addf %in, %in_3 : f32
                linalg.yield %22 : f32
              } -> tensor<64x64xf32>
              scf.yield %21 : tensor<64x64xf32>
            }
            %15 = affine.apply affine_map<(d0, d1, d2, d3) -> (d0 * 512 + d1 * 64 + d2 * 262144 + d3 * 32768)>(%arg7, %arg8, %arg6, %arg9)
            %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%15], sizes: [64, 64], strides: [512, 1] {tmd.reuse = {spatial = [{depth = 0 : i64, iterator = "%arg6", mapped_to = "y", reuse_type = "no_reuse", volume = 0 : i64}, {depth = 1 : i64, iterator = "%arg7", mapped_to = "x", reuse_type = "no_reuse", volume = 0 : i64}], temporal = [{depth = 2 : i64, iterator = "%arg8", reuse_type = "no_reuse", volume = 0 : i64}, {depth = 3 : i64, iterator = "%arg9", reuse_type = "no_reuse", volume = 0 : i64}]}} : memref<*xf32> to memref<64x64xf32, strided<[512, 1], offset: ?>>
            bufferization.materialize_in_destination %14 in writable %reinterpret_cast : (tensor<64x64xf32>, memref<64x64xf32, strided<[512, 1], offset: ?>>) -> ()
          }
        }
      } {tmd.mapped_to = "x"}
    } {tmd.mapped_to = "y"}
    return
  }
  func.func @matmul_kernel__d1i0_d0i1__f10__c0bx_c1mem(%arg0: memref<*xf32> {tt.divisibility = 16 : i32}, %arg1: memref<*xf32> {tt.divisibility = 16 : i32}, %arg2: memref<*xf32> {tt.divisibility = 16 : i32}, %arg3: index, %arg4: index, %arg5: index) {
    affine.parallel (%arg6) = (0) to (8) {
      affine.parallel (%arg7) = (0) to (8) {
        affine.for %arg8 = 0 to affine_map<()[s0, s1] -> (s1 ceildiv 8)>()[%arg3, %arg4] {
          affine.for %arg9 = 0 to affine_map<()[s0, s1] -> (s0 ceildiv 8)>()[%arg3, %arg4] {
            %cst = arith.constant 0.000000e+00 : f32
            %12 = tensor.empty() : tensor<64x64xf32>
            %13 = linalg.fill ins(%cst : f32) outs(%12 : tensor<64x64xf32>) -> tensor<64x64xf32>
            %c0 = arith.constant 0 : index
            %c8 = arith.constant 8 : index
            %c1 = arith.constant 1 : index
            %14 = scf.for %arg10 = %c0 to %c8 step %c1 iter_args(%arg11 = %13) -> (tensor<64x64xf32>) {
              %16 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 64 + d1 * 262144 + d2 * 32768)>(%arg10, %arg6, %arg9)
              %reinterpret_cast_0 = memref.reinterpret_cast %arg0 to offset: [%16], sizes: [64, 64], strides: [512, 1] {tmd.reuse = {sequential = [{depth = 4 : i64, iterator = "%arg10", reuse_type = "no_reuse", volume = 0 : i64}], spatial = [{depth = 0 : i64, iterator = "%arg6", mapped_to = "y", reuse_type = "no_reuse", volume = 0 : i64}, {depth = 1 : i64, iterator = "%arg7", mapped_to = "x", reuse_type = "total_reuse", volume = 16384 : i64}], temporal = [{depth = 2 : i64, iterator = "%arg8", reuse_type = "total_reuse", volume = 16384 : i64}, {depth = 3 : i64, iterator = "%arg9", reuse_type = "no_reuse", volume = 0 : i64}]}} : memref<*xf32> to memref<64x64xf32, strided<[512, 1], offset: ?>>
              %alloc = memref.alloc() {tmd.alloc = {local = true, memory_name = "L1", size = 16384 : i64}} : memref<64x64xf32>
              memref.copy %reinterpret_cast_0, %alloc {tmd.copy.choice = {dim = "x", interconnect_name = "horizontal_links", kind = "broadcast"}} : memref<64x64xf32, strided<[512, 1], offset: ?>> to memref<64x64xf32>
              %17 = bufferization.to_tensor %alloc restrict writable : memref<64x64xf32> to tensor<64x64xf32>
              %18 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 32768 + d1 * 512 + d2 * 64)>(%arg10, %arg7, %arg8)
              %reinterpret_cast_1 = memref.reinterpret_cast %arg1 to offset: [%18], sizes: [64, 64], strides: [512, 1] {tmd.reuse = {sequential = [{depth = 4 : i64, iterator = "%arg10", reuse_type = "no_reuse", volume = 0 : i64}], spatial = [{depth = 0 : i64, iterator = "%arg6", mapped_to = "y", reuse_type = "total_reuse", volume = 16384 : i64}, {depth = 1 : i64, iterator = "%arg7", mapped_to = "x", reuse_type = "no_reuse", volume = 0 : i64}], temporal = [{depth = 2 : i64, iterator = "%arg8", reuse_type = "no_reuse", volume = 0 : i64}, {depth = 3 : i64, iterator = "%arg9", reuse_type = "total_reuse", volume = 16384 : i64}]}} : memref<*xf32> to memref<64x64xf32, strided<[512, 1], offset: ?>>
              %alloc_2 = memref.alloc() {tmd.alloc = {local = true, memory_name = "L1", size = 16384 : i64}} : memref<64x64xf32>
              memref.copy %reinterpret_cast_1, %alloc_2 {tmd.copy.choice = {kind = "mem", memory_name = "L1"}} : memref<64x64xf32, strided<[512, 1], offset: ?>> to memref<64x64xf32>
              %19 = bufferization.to_tensor %alloc_2 restrict writable : memref<64x64xf32> to tensor<64x64xf32>
              %20 = linalg.matmul ins(%17, %19 : tensor<64x64xf32>, tensor<64x64xf32>) outs(%13 : tensor<64x64xf32>) -> tensor<64x64xf32>
              %21 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%arg11, %20 : tensor<64x64xf32>, tensor<64x64xf32>) outs(%arg11 : tensor<64x64xf32>) {
              ^bb0(%in: f32, %in_3: f32, %out: f32):
                %22 = arith.addf %in, %in_3 : f32
                linalg.yield %22 : f32
              } -> tensor<64x64xf32>
              scf.yield %21 : tensor<64x64xf32>
            }
            %15 = affine.apply affine_map<(d0, d1, d2, d3) -> (d0 * 512 + d1 * 64 + d2 * 262144 + d3 * 32768)>(%arg7, %arg8, %arg6, %arg9)
            %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%15], sizes: [64, 64], strides: [512, 1] {tmd.reuse = {spatial = [{depth = 0 : i64, iterator = "%arg6", mapped_to = "y", reuse_type = "no_reuse", volume = 0 : i64}, {depth = 1 : i64, iterator = "%arg7", mapped_to = "x", reuse_type = "no_reuse", volume = 0 : i64}], temporal = [{depth = 2 : i64, iterator = "%arg8", reuse_type = "no_reuse", volume = 0 : i64}, {depth = 3 : i64, iterator = "%arg9", reuse_type = "no_reuse", volume = 0 : i64}]}} : memref<*xf32> to memref<64x64xf32, strided<[512, 1], offset: ?>>
            bufferization.materialize_in_destination %14 in writable %reinterpret_cast : (tensor<64x64xf32>, memref<64x64xf32, strided<[512, 1], offset: ?>>) -> ()
          }
        }
      } {tmd.mapped_to = "x"}
    } {tmd.mapped_to = "y"}
    return
  }
  func.func @matmul_kernel__d1i0_d0i1__f10__c0mem_c1by(%arg0: memref<*xf32> {tt.divisibility = 16 : i32}, %arg1: memref<*xf32> {tt.divisibility = 16 : i32}, %arg2: memref<*xf32> {tt.divisibility = 16 : i32}, %arg3: index, %arg4: index, %arg5: index) {
    affine.parallel (%arg6) = (0) to (8) {
      affine.parallel (%arg7) = (0) to (8) {
        affine.for %arg8 = 0 to affine_map<()[s0, s1] -> (s1 ceildiv 8)>()[%arg3, %arg4] {
          affine.for %arg9 = 0 to affine_map<()[s0, s1] -> (s0 ceildiv 8)>()[%arg3, %arg4] {
            %cst = arith.constant 0.000000e+00 : f32
            %12 = tensor.empty() : tensor<64x64xf32>
            %13 = linalg.fill ins(%cst : f32) outs(%12 : tensor<64x64xf32>) -> tensor<64x64xf32>
            %c0 = arith.constant 0 : index
            %c8 = arith.constant 8 : index
            %c1 = arith.constant 1 : index
            %14 = scf.for %arg10 = %c0 to %c8 step %c1 iter_args(%arg11 = %13) -> (tensor<64x64xf32>) {
              %16 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 64 + d1 * 262144 + d2 * 32768)>(%arg10, %arg6, %arg9)
              %reinterpret_cast_0 = memref.reinterpret_cast %arg0 to offset: [%16], sizes: [64, 64], strides: [512, 1] {tmd.reuse = {sequential = [{depth = 4 : i64, iterator = "%arg10", reuse_type = "no_reuse", volume = 0 : i64}], spatial = [{depth = 0 : i64, iterator = "%arg6", mapped_to = "y", reuse_type = "no_reuse", volume = 0 : i64}, {depth = 1 : i64, iterator = "%arg7", mapped_to = "x", reuse_type = "total_reuse", volume = 16384 : i64}], temporal = [{depth = 2 : i64, iterator = "%arg8", reuse_type = "total_reuse", volume = 16384 : i64}, {depth = 3 : i64, iterator = "%arg9", reuse_type = "no_reuse", volume = 0 : i64}]}} : memref<*xf32> to memref<64x64xf32, strided<[512, 1], offset: ?>>
              %alloc = memref.alloc() {tmd.alloc = {local = true, memory_name = "L1", size = 16384 : i64}} : memref<64x64xf32>
              memref.copy %reinterpret_cast_0, %alloc {tmd.copy.choice = {kind = "mem", memory_name = "L1"}} : memref<64x64xf32, strided<[512, 1], offset: ?>> to memref<64x64xf32>
              %17 = bufferization.to_tensor %alloc restrict writable : memref<64x64xf32> to tensor<64x64xf32>
              %18 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 32768 + d1 * 512 + d2 * 64)>(%arg10, %arg7, %arg8)
              %reinterpret_cast_1 = memref.reinterpret_cast %arg1 to offset: [%18], sizes: [64, 64], strides: [512, 1] {tmd.reuse = {sequential = [{depth = 4 : i64, iterator = "%arg10", reuse_type = "no_reuse", volume = 0 : i64}], spatial = [{depth = 0 : i64, iterator = "%arg6", mapped_to = "y", reuse_type = "total_reuse", volume = 16384 : i64}, {depth = 1 : i64, iterator = "%arg7", mapped_to = "x", reuse_type = "no_reuse", volume = 0 : i64}], temporal = [{depth = 2 : i64, iterator = "%arg8", reuse_type = "no_reuse", volume = 0 : i64}, {depth = 3 : i64, iterator = "%arg9", reuse_type = "total_reuse", volume = 16384 : i64}]}} : memref<*xf32> to memref<64x64xf32, strided<[512, 1], offset: ?>>
              %alloc_2 = memref.alloc() {tmd.alloc = {local = true, memory_name = "L1", size = 16384 : i64}} : memref<64x64xf32>
              memref.copy %reinterpret_cast_1, %alloc_2 {tmd.copy.choice = {dim = "y", interconnect_name = "vertical_links", kind = "broadcast"}} : memref<64x64xf32, strided<[512, 1], offset: ?>> to memref<64x64xf32>
              %19 = bufferization.to_tensor %alloc_2 restrict writable : memref<64x64xf32> to tensor<64x64xf32>
              %20 = linalg.matmul ins(%17, %19 : tensor<64x64xf32>, tensor<64x64xf32>) outs(%13 : tensor<64x64xf32>) -> tensor<64x64xf32>
              %21 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%arg11, %20 : tensor<64x64xf32>, tensor<64x64xf32>) outs(%arg11 : tensor<64x64xf32>) {
              ^bb0(%in: f32, %in_3: f32, %out: f32):
                %22 = arith.addf %in, %in_3 : f32
                linalg.yield %22 : f32
              } -> tensor<64x64xf32>
              scf.yield %21 : tensor<64x64xf32>
            }
            %15 = affine.apply affine_map<(d0, d1, d2, d3) -> (d0 * 512 + d1 * 64 + d2 * 262144 + d3 * 32768)>(%arg7, %arg8, %arg6, %arg9)
            %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%15], sizes: [64, 64], strides: [512, 1] {tmd.reuse = {spatial = [{depth = 0 : i64, iterator = "%arg6", mapped_to = "y", reuse_type = "no_reuse", volume = 0 : i64}, {depth = 1 : i64, iterator = "%arg7", mapped_to = "x", reuse_type = "no_reuse", volume = 0 : i64}], temporal = [{depth = 2 : i64, iterator = "%arg8", reuse_type = "no_reuse", volume = 0 : i64}, {depth = 3 : i64, iterator = "%arg9", reuse_type = "no_reuse", volume = 0 : i64}]}} : memref<*xf32> to memref<64x64xf32, strided<[512, 1], offset: ?>>
            bufferization.materialize_in_destination %14 in writable %reinterpret_cast : (tensor<64x64xf32>, memref<64x64xf32, strided<[512, 1], offset: ?>>) -> ()
          }
        }
      } {tmd.mapped_to = "x"}
    } {tmd.mapped_to = "y"}
    return
  }
  func.func @matmul_kernel__d1i0_d0i1__f10__c0bx_c1by(%arg0: memref<*xf32> {tt.divisibility = 16 : i32}, %arg1: memref<*xf32> {tt.divisibility = 16 : i32}, %arg2: memref<*xf32> {tt.divisibility = 16 : i32}, %arg3: index, %arg4: index, %arg5: index) {
    affine.parallel (%arg6) = (0) to (8) {
      affine.parallel (%arg7) = (0) to (8) {
        affine.for %arg8 = 0 to affine_map<()[s0, s1] -> (s1 ceildiv 8)>()[%arg3, %arg4] {
          affine.for %arg9 = 0 to affine_map<()[s0, s1] -> (s0 ceildiv 8)>()[%arg3, %arg4] {
            %cst = arith.constant 0.000000e+00 : f32
            %12 = tensor.empty() : tensor<64x64xf32>
            %13 = linalg.fill ins(%cst : f32) outs(%12 : tensor<64x64xf32>) -> tensor<64x64xf32>
            %c0 = arith.constant 0 : index
            %c8 = arith.constant 8 : index
            %c1 = arith.constant 1 : index
            %14 = scf.for %arg10 = %c0 to %c8 step %c1 iter_args(%arg11 = %13) -> (tensor<64x64xf32>) {
              %16 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 64 + d1 * 262144 + d2 * 32768)>(%arg10, %arg6, %arg9)
              %reinterpret_cast_0 = memref.reinterpret_cast %arg0 to offset: [%16], sizes: [64, 64], strides: [512, 1] {tmd.reuse = {sequential = [{depth = 4 : i64, iterator = "%arg10", reuse_type = "no_reuse", volume = 0 : i64}], spatial = [{depth = 0 : i64, iterator = "%arg6", mapped_to = "y", reuse_type = "no_reuse", volume = 0 : i64}, {depth = 1 : i64, iterator = "%arg7", mapped_to = "x", reuse_type = "total_reuse", volume = 16384 : i64}], temporal = [{depth = 2 : i64, iterator = "%arg8", reuse_type = "total_reuse", volume = 16384 : i64}, {depth = 3 : i64, iterator = "%arg9", reuse_type = "no_reuse", volume = 0 : i64}]}} : memref<*xf32> to memref<64x64xf32, strided<[512, 1], offset: ?>>
              %alloc = memref.alloc() {tmd.alloc = {local = true, memory_name = "L1", size = 16384 : i64}} : memref<64x64xf32>
              memref.copy %reinterpret_cast_0, %alloc {tmd.copy.choice = {dim = "x", interconnect_name = "horizontal_links", kind = "broadcast"}} : memref<64x64xf32, strided<[512, 1], offset: ?>> to memref<64x64xf32>
              %17 = bufferization.to_tensor %alloc restrict writable : memref<64x64xf32> to tensor<64x64xf32>
              %18 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 32768 + d1 * 512 + d2 * 64)>(%arg10, %arg7, %arg8)
              %reinterpret_cast_1 = memref.reinterpret_cast %arg1 to offset: [%18], sizes: [64, 64], strides: [512, 1] {tmd.reuse = {sequential = [{depth = 4 : i64, iterator = "%arg10", reuse_type = "no_reuse", volume = 0 : i64}], spatial = [{depth = 0 : i64, iterator = "%arg6", mapped_to = "y", reuse_type = "total_reuse", volume = 16384 : i64}, {depth = 1 : i64, iterator = "%arg7", mapped_to = "x", reuse_type = "no_reuse", volume = 0 : i64}], temporal = [{depth = 2 : i64, iterator = "%arg8", reuse_type = "no_reuse", volume = 0 : i64}, {depth = 3 : i64, iterator = "%arg9", reuse_type = "total_reuse", volume = 16384 : i64}]}} : memref<*xf32> to memref<64x64xf32, strided<[512, 1], offset: ?>>
              %alloc_2 = memref.alloc() {tmd.alloc = {local = true, memory_name = "L1", size = 16384 : i64}} : memref<64x64xf32>
              memref.copy %reinterpret_cast_1, %alloc_2 {tmd.copy.choice = {dim = "y", interconnect_name = "vertical_links", kind = "broadcast"}} : memref<64x64xf32, strided<[512, 1], offset: ?>> to memref<64x64xf32>
              %19 = bufferization.to_tensor %alloc_2 restrict writable : memref<64x64xf32> to tensor<64x64xf32>
              %20 = linalg.matmul ins(%17, %19 : tensor<64x64xf32>, tensor<64x64xf32>) outs(%13 : tensor<64x64xf32>) -> tensor<64x64xf32>
              %21 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%arg11, %20 : tensor<64x64xf32>, tensor<64x64xf32>) outs(%arg11 : tensor<64x64xf32>) {
              ^bb0(%in: f32, %in_3: f32, %out: f32):
                %22 = arith.addf %in, %in_3 : f32
                linalg.yield %22 : f32
              } -> tensor<64x64xf32>
              scf.yield %21 : tensor<64x64xf32>
            }
            %15 = affine.apply affine_map<(d0, d1, d2, d3) -> (d0 * 512 + d1 * 64 + d2 * 262144 + d3 * 32768)>(%arg7, %arg8, %arg6, %arg9)
            %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%15], sizes: [64, 64], strides: [512, 1] {tmd.reuse = {spatial = [{depth = 0 : i64, iterator = "%arg6", mapped_to = "y", reuse_type = "no_reuse", volume = 0 : i64}, {depth = 1 : i64, iterator = "%arg7", mapped_to = "x", reuse_type = "no_reuse", volume = 0 : i64}], temporal = [{depth = 2 : i64, iterator = "%arg8", reuse_type = "no_reuse", volume = 0 : i64}, {depth = 3 : i64, iterator = "%arg9", reuse_type = "no_reuse", volume = 0 : i64}]}} : memref<*xf32> to memref<64x64xf32, strided<[512, 1], offset: ?>>
            bufferization.materialize_in_destination %14 in writable %reinterpret_cast : (tensor<64x64xf32>, memref<64x64xf32, strided<[512, 1], offset: ?>>) -> ()
          }
        }
      } {tmd.mapped_to = "x"}
    } {tmd.mapped_to = "y"}
    return
  }
  func.func @matmul_kernel__d0i1_d1i1__f01__c0mem_c1mem(%arg0: memref<*xf32> {tt.divisibility = 16 : i32}, %arg1: memref<*xf32> {tt.divisibility = 16 : i32}, %arg2: memref<*xf32> {tt.divisibility = 16 : i32}, %arg3: index, %arg4: index, %arg5: index) {
    affine.parallel (%arg6) = (0) to (8) {
      affine.parallel (%arg7) = (0) to (8) {
        affine.for %arg8 = 0 to affine_map<()[s0, s1] -> (s0)>()[%arg3, %arg4] {
          affine.for %arg9 = 0 to affine_map<()[s0, s1] -> ((s1 ceildiv 8) ceildiv 8)>()[%arg3, %arg4] {
            %cst = arith.constant 0.000000e+00 : f32
            %12 = tensor.empty() : tensor<64x64xf32>
            %13 = linalg.fill ins(%cst : f32) outs(%12 : tensor<64x64xf32>) -> tensor<64x64xf32>
            %c0 = arith.constant 0 : index
            %c8 = arith.constant 8 : index
            %c1 = arith.constant 1 : index
            %14 = scf.for %arg10 = %c0 to %c8 step %c1 iter_args(%arg11 = %13) -> (tensor<64x64xf32>) {
              %16 = affine.apply affine_map<(d0, d1) -> (d1 * 32768 + d0 * 64)>(%arg10, %arg8)
              %reinterpret_cast_0 = memref.reinterpret_cast %arg0 to offset: [%16], sizes: [64, 64], strides: [512, 1] {tmd.reuse = {sequential = [{depth = 4 : i64, iterator = "%arg10", reuse_type = "no_reuse", volume = 0 : i64}], spatial = [{depth = 0 : i64, iterator = "%arg6", mapped_to = "x", reuse_type = "total_reuse", volume = 16384 : i64}, {depth = 1 : i64, iterator = "%arg7", mapped_to = "y", reuse_type = "total_reuse", volume = 16384 : i64}], temporal = [{depth = 2 : i64, iterator = "%arg8", reuse_type = "no_reuse", volume = 0 : i64}, {depth = 3 : i64, iterator = "%arg9", reuse_type = "total_reuse", volume = 16384 : i64}]}} : memref<*xf32> to memref<64x64xf32, strided<[512, 1], offset: ?>>
              %alloc = memref.alloc() {tmd.alloc = {local = true, memory_name = "L1", size = 16384 : i64}} : memref<64x64xf32>
              memref.copy %reinterpret_cast_0, %alloc {tmd.copy.choice = {kind = "mem", memory_name = "L1"}} : memref<64x64xf32, strided<[512, 1], offset: ?>> to memref<64x64xf32>
              %17 = bufferization.to_tensor %alloc restrict writable : memref<64x64xf32> to tensor<64x64xf32>
              %18 = affine.apply affine_map<(d0, d1, d2, d3) -> (d0 * 32768 + d1 * 512 + d2 * 512 + d3 * 64)>(%arg10, %arg6, %arg7, %arg9)
              %reinterpret_cast_1 = memref.reinterpret_cast %arg1 to offset: [%18], sizes: [64, 64], strides: [512, 1] {tmd.reuse = {sequential = [{depth = 4 : i64, iterator = "%arg10", reuse_type = "no_reuse", volume = 0 : i64}], spatial = [{depth = 0 : i64, iterator = "%arg6", mapped_to = "x", reuse_type = "no_reuse", volume = 0 : i64}, {depth = 1 : i64, iterator = "%arg7", mapped_to = "y", reuse_type = "no_reuse", volume = 0 : i64}], temporal = [{depth = 2 : i64, iterator = "%arg8", reuse_type = "total_reuse", volume = 16384 : i64}, {depth = 3 : i64, iterator = "%arg9", reuse_type = "no_reuse", volume = 0 : i64}]}} : memref<*xf32> to memref<64x64xf32, strided<[512, 1], offset: ?>>
              %alloc_2 = memref.alloc() {tmd.alloc = {local = true, memory_name = "L1", size = 16384 : i64}} : memref<64x64xf32>
              memref.copy %reinterpret_cast_1, %alloc_2 {tmd.copy.choice = {kind = "mem", memory_name = "L1"}} : memref<64x64xf32, strided<[512, 1], offset: ?>> to memref<64x64xf32>
              %19 = bufferization.to_tensor %alloc_2 restrict writable : memref<64x64xf32> to tensor<64x64xf32>
              %20 = linalg.matmul ins(%17, %19 : tensor<64x64xf32>, tensor<64x64xf32>) outs(%13 : tensor<64x64xf32>) -> tensor<64x64xf32>
              %21 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%arg11, %20 : tensor<64x64xf32>, tensor<64x64xf32>) outs(%arg11 : tensor<64x64xf32>) {
              ^bb0(%in: f32, %in_3: f32, %out: f32):
                %22 = arith.addf %in, %in_3 : f32
                linalg.yield %22 : f32
              } -> tensor<64x64xf32>
              scf.yield %21 : tensor<64x64xf32>
            }
            %15 = affine.apply affine_map<(d0, d1, d2, d3) -> (d0 * 32768 + d1 * 512 + d2 * 512 + d3 * 64)>(%arg8, %arg6, %arg7, %arg9)
            %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%15], sizes: [64, 64], strides: [512, 1] {tmd.reuse = {spatial = [{depth = 0 : i64, iterator = "%arg6", mapped_to = "x", reuse_type = "no_reuse", volume = 0 : i64}, {depth = 1 : i64, iterator = "%arg7", mapped_to = "y", reuse_type = "no_reuse", volume = 0 : i64}], temporal = [{depth = 2 : i64, iterator = "%arg8", reuse_type = "no_reuse", volume = 0 : i64}, {depth = 3 : i64, iterator = "%arg9", reuse_type = "no_reuse", volume = 0 : i64}]}} : memref<*xf32> to memref<64x64xf32, strided<[512, 1], offset: ?>>
            bufferization.materialize_in_destination %14 in writable %reinterpret_cast : (tensor<64x64xf32>, memref<64x64xf32, strided<[512, 1], offset: ?>>) -> ()
          }
        }
      } {tmd.mapped_to = "y"}
    } {tmd.mapped_to = "x"}
    return
  }
  func.func @matmul_kernel__d0i1_d1i1__f01__c0bx_c1mem(%arg0: memref<*xf32> {tt.divisibility = 16 : i32}, %arg1: memref<*xf32> {tt.divisibility = 16 : i32}, %arg2: memref<*xf32> {tt.divisibility = 16 : i32}, %arg3: index, %arg4: index, %arg5: index) {
    affine.parallel (%arg6) = (0) to (8) {
      affine.parallel (%arg7) = (0) to (8) {
        affine.for %arg8 = 0 to affine_map<()[s0, s1] -> (s0)>()[%arg3, %arg4] {
          affine.for %arg9 = 0 to affine_map<()[s0, s1] -> ((s1 ceildiv 8) ceildiv 8)>()[%arg3, %arg4] {
            %cst = arith.constant 0.000000e+00 : f32
            %12 = tensor.empty() : tensor<64x64xf32>
            %13 = linalg.fill ins(%cst : f32) outs(%12 : tensor<64x64xf32>) -> tensor<64x64xf32>
            %c0 = arith.constant 0 : index
            %c8 = arith.constant 8 : index
            %c1 = arith.constant 1 : index
            %14 = scf.for %arg10 = %c0 to %c8 step %c1 iter_args(%arg11 = %13) -> (tensor<64x64xf32>) {
              %16 = affine.apply affine_map<(d0, d1) -> (d1 * 32768 + d0 * 64)>(%arg10, %arg8)
              %reinterpret_cast_0 = memref.reinterpret_cast %arg0 to offset: [%16], sizes: [64, 64], strides: [512, 1] {tmd.reuse = {sequential = [{depth = 4 : i64, iterator = "%arg10", reuse_type = "no_reuse", volume = 0 : i64}], spatial = [{depth = 0 : i64, iterator = "%arg6", mapped_to = "x", reuse_type = "total_reuse", volume = 16384 : i64}, {depth = 1 : i64, iterator = "%arg7", mapped_to = "y", reuse_type = "total_reuse", volume = 16384 : i64}], temporal = [{depth = 2 : i64, iterator = "%arg8", reuse_type = "no_reuse", volume = 0 : i64}, {depth = 3 : i64, iterator = "%arg9", reuse_type = "total_reuse", volume = 16384 : i64}]}} : memref<*xf32> to memref<64x64xf32, strided<[512, 1], offset: ?>>
              %alloc = memref.alloc() {tmd.alloc = {local = true, memory_name = "L1", size = 16384 : i64}} : memref<64x64xf32>
              memref.copy %reinterpret_cast_0, %alloc {tmd.copy.choice = {dim = "x", interconnect_name = "horizontal_links", kind = "broadcast"}} : memref<64x64xf32, strided<[512, 1], offset: ?>> to memref<64x64xf32>
              %17 = bufferization.to_tensor %alloc restrict writable : memref<64x64xf32> to tensor<64x64xf32>
              %18 = affine.apply affine_map<(d0, d1, d2, d3) -> (d0 * 32768 + d1 * 512 + d2 * 512 + d3 * 64)>(%arg10, %arg6, %arg7, %arg9)
              %reinterpret_cast_1 = memref.reinterpret_cast %arg1 to offset: [%18], sizes: [64, 64], strides: [512, 1] {tmd.reuse = {sequential = [{depth = 4 : i64, iterator = "%arg10", reuse_type = "no_reuse", volume = 0 : i64}], spatial = [{depth = 0 : i64, iterator = "%arg6", mapped_to = "x", reuse_type = "no_reuse", volume = 0 : i64}, {depth = 1 : i64, iterator = "%arg7", mapped_to = "y", reuse_type = "no_reuse", volume = 0 : i64}], temporal = [{depth = 2 : i64, iterator = "%arg8", reuse_type = "total_reuse", volume = 16384 : i64}, {depth = 3 : i64, iterator = "%arg9", reuse_type = "no_reuse", volume = 0 : i64}]}} : memref<*xf32> to memref<64x64xf32, strided<[512, 1], offset: ?>>
              %alloc_2 = memref.alloc() {tmd.alloc = {local = true, memory_name = "L1", size = 16384 : i64}} : memref<64x64xf32>
              memref.copy %reinterpret_cast_1, %alloc_2 {tmd.copy.choice = {kind = "mem", memory_name = "L1"}} : memref<64x64xf32, strided<[512, 1], offset: ?>> to memref<64x64xf32>
              %19 = bufferization.to_tensor %alloc_2 restrict writable : memref<64x64xf32> to tensor<64x64xf32>
              %20 = linalg.matmul ins(%17, %19 : tensor<64x64xf32>, tensor<64x64xf32>) outs(%13 : tensor<64x64xf32>) -> tensor<64x64xf32>
              %21 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%arg11, %20 : tensor<64x64xf32>, tensor<64x64xf32>) outs(%arg11 : tensor<64x64xf32>) {
              ^bb0(%in: f32, %in_3: f32, %out: f32):
                %22 = arith.addf %in, %in_3 : f32
                linalg.yield %22 : f32
              } -> tensor<64x64xf32>
              scf.yield %21 : tensor<64x64xf32>
            }
            %15 = affine.apply affine_map<(d0, d1, d2, d3) -> (d0 * 32768 + d1 * 512 + d2 * 512 + d3 * 64)>(%arg8, %arg6, %arg7, %arg9)
            %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%15], sizes: [64, 64], strides: [512, 1] {tmd.reuse = {spatial = [{depth = 0 : i64, iterator = "%arg6", mapped_to = "x", reuse_type = "no_reuse", volume = 0 : i64}, {depth = 1 : i64, iterator = "%arg7", mapped_to = "y", reuse_type = "no_reuse", volume = 0 : i64}], temporal = [{depth = 2 : i64, iterator = "%arg8", reuse_type = "no_reuse", volume = 0 : i64}, {depth = 3 : i64, iterator = "%arg9", reuse_type = "no_reuse", volume = 0 : i64}]}} : memref<*xf32> to memref<64x64xf32, strided<[512, 1], offset: ?>>
            bufferization.materialize_in_destination %14 in writable %reinterpret_cast : (tensor<64x64xf32>, memref<64x64xf32, strided<[512, 1], offset: ?>>) -> ()
          }
        }
      } {tmd.mapped_to = "y"}
    } {tmd.mapped_to = "x"}
    return
  }
  func.func @matmul_kernel__d0i1_d1i1__f01__c0by_c1mem(%arg0: memref<*xf32> {tt.divisibility = 16 : i32}, %arg1: memref<*xf32> {tt.divisibility = 16 : i32}, %arg2: memref<*xf32> {tt.divisibility = 16 : i32}, %arg3: index, %arg4: index, %arg5: index) {
    affine.parallel (%arg6) = (0) to (8) {
      affine.parallel (%arg7) = (0) to (8) {
        affine.for %arg8 = 0 to affine_map<()[s0, s1] -> (s0)>()[%arg3, %arg4] {
          affine.for %arg9 = 0 to affine_map<()[s0, s1] -> ((s1 ceildiv 8) ceildiv 8)>()[%arg3, %arg4] {
            %cst = arith.constant 0.000000e+00 : f32
            %12 = tensor.empty() : tensor<64x64xf32>
            %13 = linalg.fill ins(%cst : f32) outs(%12 : tensor<64x64xf32>) -> tensor<64x64xf32>
            %c0 = arith.constant 0 : index
            %c8 = arith.constant 8 : index
            %c1 = arith.constant 1 : index
            %14 = scf.for %arg10 = %c0 to %c8 step %c1 iter_args(%arg11 = %13) -> (tensor<64x64xf32>) {
              %16 = affine.apply affine_map<(d0, d1) -> (d1 * 32768 + d0 * 64)>(%arg10, %arg8)
              %reinterpret_cast_0 = memref.reinterpret_cast %arg0 to offset: [%16], sizes: [64, 64], strides: [512, 1] {tmd.reuse = {sequential = [{depth = 4 : i64, iterator = "%arg10", reuse_type = "no_reuse", volume = 0 : i64}], spatial = [{depth = 0 : i64, iterator = "%arg6", mapped_to = "x", reuse_type = "total_reuse", volume = 16384 : i64}, {depth = 1 : i64, iterator = "%arg7", mapped_to = "y", reuse_type = "total_reuse", volume = 16384 : i64}], temporal = [{depth = 2 : i64, iterator = "%arg8", reuse_type = "no_reuse", volume = 0 : i64}, {depth = 3 : i64, iterator = "%arg9", reuse_type = "total_reuse", volume = 16384 : i64}]}} : memref<*xf32> to memref<64x64xf32, strided<[512, 1], offset: ?>>
              %alloc = memref.alloc() {tmd.alloc = {local = true, memory_name = "L1", size = 16384 : i64}} : memref<64x64xf32>
              memref.copy %reinterpret_cast_0, %alloc {tmd.copy.choice = {dim = "y", interconnect_name = "vertical_links", kind = "broadcast"}} : memref<64x64xf32, strided<[512, 1], offset: ?>> to memref<64x64xf32>
              %17 = bufferization.to_tensor %alloc restrict writable : memref<64x64xf32> to tensor<64x64xf32>
              %18 = affine.apply affine_map<(d0, d1, d2, d3) -> (d0 * 32768 + d1 * 512 + d2 * 512 + d3 * 64)>(%arg10, %arg6, %arg7, %arg9)
              %reinterpret_cast_1 = memref.reinterpret_cast %arg1 to offset: [%18], sizes: [64, 64], strides: [512, 1] {tmd.reuse = {sequential = [{depth = 4 : i64, iterator = "%arg10", reuse_type = "no_reuse", volume = 0 : i64}], spatial = [{depth = 0 : i64, iterator = "%arg6", mapped_to = "x", reuse_type = "no_reuse", volume = 0 : i64}, {depth = 1 : i64, iterator = "%arg7", mapped_to = "y", reuse_type = "no_reuse", volume = 0 : i64}], temporal = [{depth = 2 : i64, iterator = "%arg8", reuse_type = "total_reuse", volume = 16384 : i64}, {depth = 3 : i64, iterator = "%arg9", reuse_type = "no_reuse", volume = 0 : i64}]}} : memref<*xf32> to memref<64x64xf32, strided<[512, 1], offset: ?>>
              %alloc_2 = memref.alloc() {tmd.alloc = {local = true, memory_name = "L1", size = 16384 : i64}} : memref<64x64xf32>
              memref.copy %reinterpret_cast_1, %alloc_2 {tmd.copy.choice = {kind = "mem", memory_name = "L1"}} : memref<64x64xf32, strided<[512, 1], offset: ?>> to memref<64x64xf32>
              %19 = bufferization.to_tensor %alloc_2 restrict writable : memref<64x64xf32> to tensor<64x64xf32>
              %20 = linalg.matmul ins(%17, %19 : tensor<64x64xf32>, tensor<64x64xf32>) outs(%13 : tensor<64x64xf32>) -> tensor<64x64xf32>
              %21 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%arg11, %20 : tensor<64x64xf32>, tensor<64x64xf32>) outs(%arg11 : tensor<64x64xf32>) {
              ^bb0(%in: f32, %in_3: f32, %out: f32):
                %22 = arith.addf %in, %in_3 : f32
                linalg.yield %22 : f32
              } -> tensor<64x64xf32>
              scf.yield %21 : tensor<64x64xf32>
            }
            %15 = affine.apply affine_map<(d0, d1, d2, d3) -> (d0 * 32768 + d1 * 512 + d2 * 512 + d3 * 64)>(%arg8, %arg6, %arg7, %arg9)
            %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%15], sizes: [64, 64], strides: [512, 1] {tmd.reuse = {spatial = [{depth = 0 : i64, iterator = "%arg6", mapped_to = "x", reuse_type = "no_reuse", volume = 0 : i64}, {depth = 1 : i64, iterator = "%arg7", mapped_to = "y", reuse_type = "no_reuse", volume = 0 : i64}], temporal = [{depth = 2 : i64, iterator = "%arg8", reuse_type = "no_reuse", volume = 0 : i64}, {depth = 3 : i64, iterator = "%arg9", reuse_type = "no_reuse", volume = 0 : i64}]}} : memref<*xf32> to memref<64x64xf32, strided<[512, 1], offset: ?>>
            bufferization.materialize_in_destination %14 in writable %reinterpret_cast : (tensor<64x64xf32>, memref<64x64xf32, strided<[512, 1], offset: ?>>) -> ()
          }
        }
      } {tmd.mapped_to = "y"}
    } {tmd.mapped_to = "x"}
    return
  }
  func.func @matmul_kernel__d0i1_d1i1__f01__c0bd_c1mem(%arg0: memref<*xf32> {tt.divisibility = 16 : i32}, %arg1: memref<*xf32> {tt.divisibility = 16 : i32}, %arg2: memref<*xf32> {tt.divisibility = 16 : i32}, %arg3: index, %arg4: index, %arg5: index) {
    affine.parallel (%arg6) = (0) to (8) {
      affine.parallel (%arg7) = (0) to (8) {
        affine.for %arg8 = 0 to affine_map<()[s0, s1] -> (s0)>()[%arg3, %arg4] {
          affine.for %arg9 = 0 to affine_map<()[s0, s1] -> ((s1 ceildiv 8) ceildiv 8)>()[%arg3, %arg4] {
            %cst = arith.constant 0.000000e+00 : f32
            %12 = tensor.empty() : tensor<64x64xf32>
            %13 = linalg.fill ins(%cst : f32) outs(%12 : tensor<64x64xf32>) -> tensor<64x64xf32>
            %c0 = arith.constant 0 : index
            %c8 = arith.constant 8 : index
            %c1 = arith.constant 1 : index
            %14 = scf.for %arg10 = %c0 to %c8 step %c1 iter_args(%arg11 = %13) -> (tensor<64x64xf32>) {
              %16 = affine.apply affine_map<(d0, d1) -> (d1 * 32768 + d0 * 64)>(%arg10, %arg8)
              %reinterpret_cast_0 = memref.reinterpret_cast %arg0 to offset: [%16], sizes: [64, 64], strides: [512, 1] {tmd.reuse = {sequential = [{depth = 4 : i64, iterator = "%arg10", reuse_type = "no_reuse", volume = 0 : i64}], spatial = [{depth = 0 : i64, iterator = "%arg6", mapped_to = "x", reuse_type = "total_reuse", volume = 16384 : i64}, {depth = 1 : i64, iterator = "%arg7", mapped_to = "y", reuse_type = "total_reuse", volume = 16384 : i64}], temporal = [{depth = 2 : i64, iterator = "%arg8", reuse_type = "no_reuse", volume = 0 : i64}, {depth = 3 : i64, iterator = "%arg9", reuse_type = "total_reuse", volume = 16384 : i64}]}} : memref<*xf32> to memref<64x64xf32, strided<[512, 1], offset: ?>>
              %alloc = memref.alloc() {tmd.alloc = {local = true, memory_name = "L1", size = 16384 : i64}} : memref<64x64xf32>
              memref.copy %reinterpret_cast_0, %alloc {tmd.copy.choice = {dim = "xy", interconnect_name = "all_links", kind = "broadcast"}} : memref<64x64xf32, strided<[512, 1], offset: ?>> to memref<64x64xf32>
              %17 = bufferization.to_tensor %alloc restrict writable : memref<64x64xf32> to tensor<64x64xf32>
              %18 = affine.apply affine_map<(d0, d1, d2, d3) -> (d0 * 32768 + d1 * 512 + d2 * 512 + d3 * 64)>(%arg10, %arg6, %arg7, %arg9)
              %reinterpret_cast_1 = memref.reinterpret_cast %arg1 to offset: [%18], sizes: [64, 64], strides: [512, 1] {tmd.reuse = {sequential = [{depth = 4 : i64, iterator = "%arg10", reuse_type = "no_reuse", volume = 0 : i64}], spatial = [{depth = 0 : i64, iterator = "%arg6", mapped_to = "x", reuse_type = "no_reuse", volume = 0 : i64}, {depth = 1 : i64, iterator = "%arg7", mapped_to = "y", reuse_type = "no_reuse", volume = 0 : i64}], temporal = [{depth = 2 : i64, iterator = "%arg8", reuse_type = "total_reuse", volume = 16384 : i64}, {depth = 3 : i64, iterator = "%arg9", reuse_type = "no_reuse", volume = 0 : i64}]}} : memref<*xf32> to memref<64x64xf32, strided<[512, 1], offset: ?>>
              %alloc_2 = memref.alloc() {tmd.alloc = {local = true, memory_name = "L1", size = 16384 : i64}} : memref<64x64xf32>
              memref.copy %reinterpret_cast_1, %alloc_2 {tmd.copy.choice = {kind = "mem", memory_name = "L1"}} : memref<64x64xf32, strided<[512, 1], offset: ?>> to memref<64x64xf32>
              %19 = bufferization.to_tensor %alloc_2 restrict writable : memref<64x64xf32> to tensor<64x64xf32>
              %20 = linalg.matmul ins(%17, %19 : tensor<64x64xf32>, tensor<64x64xf32>) outs(%13 : tensor<64x64xf32>) -> tensor<64x64xf32>
              %21 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%arg11, %20 : tensor<64x64xf32>, tensor<64x64xf32>) outs(%arg11 : tensor<64x64xf32>) {
              ^bb0(%in: f32, %in_3: f32, %out: f32):
                %22 = arith.addf %in, %in_3 : f32
                linalg.yield %22 : f32
              } -> tensor<64x64xf32>
              scf.yield %21 : tensor<64x64xf32>
            }
            %15 = affine.apply affine_map<(d0, d1, d2, d3) -> (d0 * 32768 + d1 * 512 + d2 * 512 + d3 * 64)>(%arg8, %arg6, %arg7, %arg9)
            %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%15], sizes: [64, 64], strides: [512, 1] {tmd.reuse = {spatial = [{depth = 0 : i64, iterator = "%arg6", mapped_to = "x", reuse_type = "no_reuse", volume = 0 : i64}, {depth = 1 : i64, iterator = "%arg7", mapped_to = "y", reuse_type = "no_reuse", volume = 0 : i64}], temporal = [{depth = 2 : i64, iterator = "%arg8", reuse_type = "no_reuse", volume = 0 : i64}, {depth = 3 : i64, iterator = "%arg9", reuse_type = "no_reuse", volume = 0 : i64}]}} : memref<*xf32> to memref<64x64xf32, strided<[512, 1], offset: ?>>
            bufferization.materialize_in_destination %14 in writable %reinterpret_cast : (tensor<64x64xf32>, memref<64x64xf32, strided<[512, 1], offset: ?>>) -> ()
          }
        }
      } {tmd.mapped_to = "y"}
    } {tmd.mapped_to = "x"}
    return
  }
  func.func @matmul_kernel__d0i1_d1i1__f10__c0mem_c1mem(%arg0: memref<*xf32> {tt.divisibility = 16 : i32}, %arg1: memref<*xf32> {tt.divisibility = 16 : i32}, %arg2: memref<*xf32> {tt.divisibility = 16 : i32}, %arg3: index, %arg4: index, %arg5: index) {
    affine.parallel (%arg6) = (0) to (8) {
      affine.parallel (%arg7) = (0) to (8) {
        affine.for %arg8 = 0 to affine_map<()[s0, s1] -> ((s1 ceildiv 8) ceildiv 8)>()[%arg3, %arg4] {
          affine.for %arg9 = 0 to affine_map<()[s0, s1] -> (s0)>()[%arg3, %arg4] {
            %cst = arith.constant 0.000000e+00 : f32
            %12 = tensor.empty() : tensor<64x64xf32>
            %13 = linalg.fill ins(%cst : f32) outs(%12 : tensor<64x64xf32>) -> tensor<64x64xf32>
            %c0 = arith.constant 0 : index
            %c8 = arith.constant 8 : index
            %c1 = arith.constant 1 : index
            %14 = scf.for %arg10 = %c0 to %c8 step %c1 iter_args(%arg11 = %13) -> (tensor<64x64xf32>) {
              %16 = affine.apply affine_map<(d0, d1) -> (d1 * 32768 + d0 * 64)>(%arg10, %arg9)
              %reinterpret_cast_0 = memref.reinterpret_cast %arg0 to offset: [%16], sizes: [64, 64], strides: [512, 1] {tmd.reuse = {sequential = [{depth = 4 : i64, iterator = "%arg10", reuse_type = "no_reuse", volume = 0 : i64}], spatial = [{depth = 0 : i64, iterator = "%arg6", mapped_to = "x", reuse_type = "total_reuse", volume = 16384 : i64}, {depth = 1 : i64, iterator = "%arg7", mapped_to = "y", reuse_type = "total_reuse", volume = 16384 : i64}], temporal = [{depth = 2 : i64, iterator = "%arg8", reuse_type = "total_reuse", volume = 16384 : i64}, {depth = 3 : i64, iterator = "%arg9", reuse_type = "no_reuse", volume = 0 : i64}]}} : memref<*xf32> to memref<64x64xf32, strided<[512, 1], offset: ?>>
              %alloc = memref.alloc() {tmd.alloc = {local = true, memory_name = "L1", size = 16384 : i64}} : memref<64x64xf32>
              memref.copy %reinterpret_cast_0, %alloc {tmd.copy.choice = {kind = "mem", memory_name = "L1"}} : memref<64x64xf32, strided<[512, 1], offset: ?>> to memref<64x64xf32>
              %17 = bufferization.to_tensor %alloc restrict writable : memref<64x64xf32> to tensor<64x64xf32>
              %18 = affine.apply affine_map<(d0, d1, d2, d3) -> (d0 * 32768 + d1 * 512 + d2 * 512 + d3 * 64)>(%arg10, %arg6, %arg7, %arg8)
              %reinterpret_cast_1 = memref.reinterpret_cast %arg1 to offset: [%18], sizes: [64, 64], strides: [512, 1] {tmd.reuse = {sequential = [{depth = 4 : i64, iterator = "%arg10", reuse_type = "no_reuse", volume = 0 : i64}], spatial = [{depth = 0 : i64, iterator = "%arg6", mapped_to = "x", reuse_type = "no_reuse", volume = 0 : i64}, {depth = 1 : i64, iterator = "%arg7", mapped_to = "y", reuse_type = "no_reuse", volume = 0 : i64}], temporal = [{depth = 2 : i64, iterator = "%arg8", reuse_type = "no_reuse", volume = 0 : i64}, {depth = 3 : i64, iterator = "%arg9", reuse_type = "total_reuse", volume = 16384 : i64}]}} : memref<*xf32> to memref<64x64xf32, strided<[512, 1], offset: ?>>
              %alloc_2 = memref.alloc() {tmd.alloc = {local = true, memory_name = "L1", size = 16384 : i64}} : memref<64x64xf32>
              memref.copy %reinterpret_cast_1, %alloc_2 {tmd.copy.choice = {kind = "mem", memory_name = "L1"}} : memref<64x64xf32, strided<[512, 1], offset: ?>> to memref<64x64xf32>
              %19 = bufferization.to_tensor %alloc_2 restrict writable : memref<64x64xf32> to tensor<64x64xf32>
              %20 = linalg.matmul ins(%17, %19 : tensor<64x64xf32>, tensor<64x64xf32>) outs(%13 : tensor<64x64xf32>) -> tensor<64x64xf32>
              %21 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%arg11, %20 : tensor<64x64xf32>, tensor<64x64xf32>) outs(%arg11 : tensor<64x64xf32>) {
              ^bb0(%in: f32, %in_3: f32, %out: f32):
                %22 = arith.addf %in, %in_3 : f32
                linalg.yield %22 : f32
              } -> tensor<64x64xf32>
              scf.yield %21 : tensor<64x64xf32>
            }
            %15 = affine.apply affine_map<(d0, d1, d2, d3) -> (d0 * 32768 + d1 * 512 + d2 * 512 + d3 * 64)>(%arg9, %arg6, %arg7, %arg8)
            %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%15], sizes: [64, 64], strides: [512, 1] {tmd.reuse = {spatial = [{depth = 0 : i64, iterator = "%arg6", mapped_to = "x", reuse_type = "no_reuse", volume = 0 : i64}, {depth = 1 : i64, iterator = "%arg7", mapped_to = "y", reuse_type = "no_reuse", volume = 0 : i64}], temporal = [{depth = 2 : i64, iterator = "%arg8", reuse_type = "no_reuse", volume = 0 : i64}, {depth = 3 : i64, iterator = "%arg9", reuse_type = "no_reuse", volume = 0 : i64}]}} : memref<*xf32> to memref<64x64xf32, strided<[512, 1], offset: ?>>
            bufferization.materialize_in_destination %14 in writable %reinterpret_cast : (tensor<64x64xf32>, memref<64x64xf32, strided<[512, 1], offset: ?>>) -> ()
          }
        }
      } {tmd.mapped_to = "y"}
    } {tmd.mapped_to = "x"}
    return
  }
  func.func @matmul_kernel__d0i1_d1i1__f10__c0bx_c1mem(%arg0: memref<*xf32> {tt.divisibility = 16 : i32}, %arg1: memref<*xf32> {tt.divisibility = 16 : i32}, %arg2: memref<*xf32> {tt.divisibility = 16 : i32}, %arg3: index, %arg4: index, %arg5: index) {
    affine.parallel (%arg6) = (0) to (8) {
      affine.parallel (%arg7) = (0) to (8) {
        affine.for %arg8 = 0 to affine_map<()[s0, s1] -> ((s1 ceildiv 8) ceildiv 8)>()[%arg3, %arg4] {
          affine.for %arg9 = 0 to affine_map<()[s0, s1] -> (s0)>()[%arg3, %arg4] {
            %cst = arith.constant 0.000000e+00 : f32
            %12 = tensor.empty() : tensor<64x64xf32>
            %13 = linalg.fill ins(%cst : f32) outs(%12 : tensor<64x64xf32>) -> tensor<64x64xf32>
            %c0 = arith.constant 0 : index
            %c8 = arith.constant 8 : index
            %c1 = arith.constant 1 : index
            %14 = scf.for %arg10 = %c0 to %c8 step %c1 iter_args(%arg11 = %13) -> (tensor<64x64xf32>) {
              %16 = affine.apply affine_map<(d0, d1) -> (d1 * 32768 + d0 * 64)>(%arg10, %arg9)
              %reinterpret_cast_0 = memref.reinterpret_cast %arg0 to offset: [%16], sizes: [64, 64], strides: [512, 1] {tmd.reuse = {sequential = [{depth = 4 : i64, iterator = "%arg10", reuse_type = "no_reuse", volume = 0 : i64}], spatial = [{depth = 0 : i64, iterator = "%arg6", mapped_to = "x", reuse_type = "total_reuse", volume = 16384 : i64}, {depth = 1 : i64, iterator = "%arg7", mapped_to = "y", reuse_type = "total_reuse", volume = 16384 : i64}], temporal = [{depth = 2 : i64, iterator = "%arg8", reuse_type = "total_reuse", volume = 16384 : i64}, {depth = 3 : i64, iterator = "%arg9", reuse_type = "no_reuse", volume = 0 : i64}]}} : memref<*xf32> to memref<64x64xf32, strided<[512, 1], offset: ?>>
              %alloc = memref.alloc() {tmd.alloc = {local = true, memory_name = "L1", size = 16384 : i64}} : memref<64x64xf32>
              memref.copy %reinterpret_cast_0, %alloc {tmd.copy.choice = {dim = "x", interconnect_name = "horizontal_links", kind = "broadcast"}} : memref<64x64xf32, strided<[512, 1], offset: ?>> to memref<64x64xf32>
              %17 = bufferization.to_tensor %alloc restrict writable : memref<64x64xf32> to tensor<64x64xf32>
              %18 = affine.apply affine_map<(d0, d1, d2, d3) -> (d0 * 32768 + d1 * 512 + d2 * 512 + d3 * 64)>(%arg10, %arg6, %arg7, %arg8)
              %reinterpret_cast_1 = memref.reinterpret_cast %arg1 to offset: [%18], sizes: [64, 64], strides: [512, 1] {tmd.reuse = {sequential = [{depth = 4 : i64, iterator = "%arg10", reuse_type = "no_reuse", volume = 0 : i64}], spatial = [{depth = 0 : i64, iterator = "%arg6", mapped_to = "x", reuse_type = "no_reuse", volume = 0 : i64}, {depth = 1 : i64, iterator = "%arg7", mapped_to = "y", reuse_type = "no_reuse", volume = 0 : i64}], temporal = [{depth = 2 : i64, iterator = "%arg8", reuse_type = "no_reuse", volume = 0 : i64}, {depth = 3 : i64, iterator = "%arg9", reuse_type = "total_reuse", volume = 16384 : i64}]}} : memref<*xf32> to memref<64x64xf32, strided<[512, 1], offset: ?>>
              %alloc_2 = memref.alloc() {tmd.alloc = {local = true, memory_name = "L1", size = 16384 : i64}} : memref<64x64xf32>
              memref.copy %reinterpret_cast_1, %alloc_2 {tmd.copy.choice = {kind = "mem", memory_name = "L1"}} : memref<64x64xf32, strided<[512, 1], offset: ?>> to memref<64x64xf32>
              %19 = bufferization.to_tensor %alloc_2 restrict writable : memref<64x64xf32> to tensor<64x64xf32>
              %20 = linalg.matmul ins(%17, %19 : tensor<64x64xf32>, tensor<64x64xf32>) outs(%13 : tensor<64x64xf32>) -> tensor<64x64xf32>
              %21 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%arg11, %20 : tensor<64x64xf32>, tensor<64x64xf32>) outs(%arg11 : tensor<64x64xf32>) {
              ^bb0(%in: f32, %in_3: f32, %out: f32):
                %22 = arith.addf %in, %in_3 : f32
                linalg.yield %22 : f32
              } -> tensor<64x64xf32>
              scf.yield %21 : tensor<64x64xf32>
            }
            %15 = affine.apply affine_map<(d0, d1, d2, d3) -> (d0 * 32768 + d1 * 512 + d2 * 512 + d3 * 64)>(%arg9, %arg6, %arg7, %arg8)
            %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%15], sizes: [64, 64], strides: [512, 1] {tmd.reuse = {spatial = [{depth = 0 : i64, iterator = "%arg6", mapped_to = "x", reuse_type = "no_reuse", volume = 0 : i64}, {depth = 1 : i64, iterator = "%arg7", mapped_to = "y", reuse_type = "no_reuse", volume = 0 : i64}], temporal = [{depth = 2 : i64, iterator = "%arg8", reuse_type = "no_reuse", volume = 0 : i64}, {depth = 3 : i64, iterator = "%arg9", reuse_type = "no_reuse", volume = 0 : i64}]}} : memref<*xf32> to memref<64x64xf32, strided<[512, 1], offset: ?>>
            bufferization.materialize_in_destination %14 in writable %reinterpret_cast : (tensor<64x64xf32>, memref<64x64xf32, strided<[512, 1], offset: ?>>) -> ()
          }
        }
      } {tmd.mapped_to = "y"}
    } {tmd.mapped_to = "x"}
    return
  }
  func.func @matmul_kernel__d0i1_d1i1__f10__c0by_c1mem(%arg0: memref<*xf32> {tt.divisibility = 16 : i32}, %arg1: memref<*xf32> {tt.divisibility = 16 : i32}, %arg2: memref<*xf32> {tt.divisibility = 16 : i32}, %arg3: index, %arg4: index, %arg5: index) {
    affine.parallel (%arg6) = (0) to (8) {
      affine.parallel (%arg7) = (0) to (8) {
        affine.for %arg8 = 0 to affine_map<()[s0, s1] -> ((s1 ceildiv 8) ceildiv 8)>()[%arg3, %arg4] {
          affine.for %arg9 = 0 to affine_map<()[s0, s1] -> (s0)>()[%arg3, %arg4] {
            %cst = arith.constant 0.000000e+00 : f32
            %12 = tensor.empty() : tensor<64x64xf32>
            %13 = linalg.fill ins(%cst : f32) outs(%12 : tensor<64x64xf32>) -> tensor<64x64xf32>
            %c0 = arith.constant 0 : index
            %c8 = arith.constant 8 : index
            %c1 = arith.constant 1 : index
            %14 = scf.for %arg10 = %c0 to %c8 step %c1 iter_args(%arg11 = %13) -> (tensor<64x64xf32>) {
              %16 = affine.apply affine_map<(d0, d1) -> (d1 * 32768 + d0 * 64)>(%arg10, %arg9)
              %reinterpret_cast_0 = memref.reinterpret_cast %arg0 to offset: [%16], sizes: [64, 64], strides: [512, 1] {tmd.reuse = {sequential = [{depth = 4 : i64, iterator = "%arg10", reuse_type = "no_reuse", volume = 0 : i64}], spatial = [{depth = 0 : i64, iterator = "%arg6", mapped_to = "x", reuse_type = "total_reuse", volume = 16384 : i64}, {depth = 1 : i64, iterator = "%arg7", mapped_to = "y", reuse_type = "total_reuse", volume = 16384 : i64}], temporal = [{depth = 2 : i64, iterator = "%arg8", reuse_type = "total_reuse", volume = 16384 : i64}, {depth = 3 : i64, iterator = "%arg9", reuse_type = "no_reuse", volume = 0 : i64}]}} : memref<*xf32> to memref<64x64xf32, strided<[512, 1], offset: ?>>
              %alloc = memref.alloc() {tmd.alloc = {local = true, memory_name = "L1", size = 16384 : i64}} : memref<64x64xf32>
              memref.copy %reinterpret_cast_0, %alloc {tmd.copy.choice = {dim = "y", interconnect_name = "vertical_links", kind = "broadcast"}} : memref<64x64xf32, strided<[512, 1], offset: ?>> to memref<64x64xf32>
              %17 = bufferization.to_tensor %alloc restrict writable : memref<64x64xf32> to tensor<64x64xf32>
              %18 = affine.apply affine_map<(d0, d1, d2, d3) -> (d0 * 32768 + d1 * 512 + d2 * 512 + d3 * 64)>(%arg10, %arg6, %arg7, %arg8)
              %reinterpret_cast_1 = memref.reinterpret_cast %arg1 to offset: [%18], sizes: [64, 64], strides: [512, 1] {tmd.reuse = {sequential = [{depth = 4 : i64, iterator = "%arg10", reuse_type = "no_reuse", volume = 0 : i64}], spatial = [{depth = 0 : i64, iterator = "%arg6", mapped_to = "x", reuse_type = "no_reuse", volume = 0 : i64}, {depth = 1 : i64, iterator = "%arg7", mapped_to = "y", reuse_type = "no_reuse", volume = 0 : i64}], temporal = [{depth = 2 : i64, iterator = "%arg8", reuse_type = "no_reuse", volume = 0 : i64}, {depth = 3 : i64, iterator = "%arg9", reuse_type = "total_reuse", volume = 16384 : i64}]}} : memref<*xf32> to memref<64x64xf32, strided<[512, 1], offset: ?>>
              %alloc_2 = memref.alloc() {tmd.alloc = {local = true, memory_name = "L1", size = 16384 : i64}} : memref<64x64xf32>
              memref.copy %reinterpret_cast_1, %alloc_2 {tmd.copy.choice = {kind = "mem", memory_name = "L1"}} : memref<64x64xf32, strided<[512, 1], offset: ?>> to memref<64x64xf32>
              %19 = bufferization.to_tensor %alloc_2 restrict writable : memref<64x64xf32> to tensor<64x64xf32>
              %20 = linalg.matmul ins(%17, %19 : tensor<64x64xf32>, tensor<64x64xf32>) outs(%13 : tensor<64x64xf32>) -> tensor<64x64xf32>
              %21 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%arg11, %20 : tensor<64x64xf32>, tensor<64x64xf32>) outs(%arg11 : tensor<64x64xf32>) {
              ^bb0(%in: f32, %in_3: f32, %out: f32):
                %22 = arith.addf %in, %in_3 : f32
                linalg.yield %22 : f32
              } -> tensor<64x64xf32>
              scf.yield %21 : tensor<64x64xf32>
            }
            %15 = affine.apply affine_map<(d0, d1, d2, d3) -> (d0 * 32768 + d1 * 512 + d2 * 512 + d3 * 64)>(%arg9, %arg6, %arg7, %arg8)
            %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%15], sizes: [64, 64], strides: [512, 1] {tmd.reuse = {spatial = [{depth = 0 : i64, iterator = "%arg6", mapped_to = "x", reuse_type = "no_reuse", volume = 0 : i64}, {depth = 1 : i64, iterator = "%arg7", mapped_to = "y", reuse_type = "no_reuse", volume = 0 : i64}], temporal = [{depth = 2 : i64, iterator = "%arg8", reuse_type = "no_reuse", volume = 0 : i64}, {depth = 3 : i64, iterator = "%arg9", reuse_type = "no_reuse", volume = 0 : i64}]}} : memref<*xf32> to memref<64x64xf32, strided<[512, 1], offset: ?>>
            bufferization.materialize_in_destination %14 in writable %reinterpret_cast : (tensor<64x64xf32>, memref<64x64xf32, strided<[512, 1], offset: ?>>) -> ()
          }
        }
      } {tmd.mapped_to = "y"}
    } {tmd.mapped_to = "x"}
    return
  }
  func.func @matmul_kernel__d0i1_d1i1__f10__c0bd_c1mem(%arg0: memref<*xf32> {tt.divisibility = 16 : i32}, %arg1: memref<*xf32> {tt.divisibility = 16 : i32}, %arg2: memref<*xf32> {tt.divisibility = 16 : i32}, %arg3: index, %arg4: index, %arg5: index) {
    affine.parallel (%arg6) = (0) to (8) {
      affine.parallel (%arg7) = (0) to (8) {
        affine.for %arg8 = 0 to affine_map<()[s0, s1] -> ((s1 ceildiv 8) ceildiv 8)>()[%arg3, %arg4] {
          affine.for %arg9 = 0 to affine_map<()[s0, s1] -> (s0)>()[%arg3, %arg4] {
            %cst = arith.constant 0.000000e+00 : f32
            %12 = tensor.empty() : tensor<64x64xf32>
            %13 = linalg.fill ins(%cst : f32) outs(%12 : tensor<64x64xf32>) -> tensor<64x64xf32>
            %c0 = arith.constant 0 : index
            %c8 = arith.constant 8 : index
            %c1 = arith.constant 1 : index
            %14 = scf.for %arg10 = %c0 to %c8 step %c1 iter_args(%arg11 = %13) -> (tensor<64x64xf32>) {
              %16 = affine.apply affine_map<(d0, d1) -> (d1 * 32768 + d0 * 64)>(%arg10, %arg9)
              %reinterpret_cast_0 = memref.reinterpret_cast %arg0 to offset: [%16], sizes: [64, 64], strides: [512, 1] {tmd.reuse = {sequential = [{depth = 4 : i64, iterator = "%arg10", reuse_type = "no_reuse", volume = 0 : i64}], spatial = [{depth = 0 : i64, iterator = "%arg6", mapped_to = "x", reuse_type = "total_reuse", volume = 16384 : i64}, {depth = 1 : i64, iterator = "%arg7", mapped_to = "y", reuse_type = "total_reuse", volume = 16384 : i64}], temporal = [{depth = 2 : i64, iterator = "%arg8", reuse_type = "total_reuse", volume = 16384 : i64}, {depth = 3 : i64, iterator = "%arg9", reuse_type = "no_reuse", volume = 0 : i64}]}} : memref<*xf32> to memref<64x64xf32, strided<[512, 1], offset: ?>>
              %alloc = memref.alloc() {tmd.alloc = {local = true, memory_name = "L1", size = 16384 : i64}} : memref<64x64xf32>
              memref.copy %reinterpret_cast_0, %alloc {tmd.copy.choice = {dim = "xy", interconnect_name = "all_links", kind = "broadcast"}} : memref<64x64xf32, strided<[512, 1], offset: ?>> to memref<64x64xf32>
              %17 = bufferization.to_tensor %alloc restrict writable : memref<64x64xf32> to tensor<64x64xf32>
              %18 = affine.apply affine_map<(d0, d1, d2, d3) -> (d0 * 32768 + d1 * 512 + d2 * 512 + d3 * 64)>(%arg10, %arg6, %arg7, %arg8)
              %reinterpret_cast_1 = memref.reinterpret_cast %arg1 to offset: [%18], sizes: [64, 64], strides: [512, 1] {tmd.reuse = {sequential = [{depth = 4 : i64, iterator = "%arg10", reuse_type = "no_reuse", volume = 0 : i64}], spatial = [{depth = 0 : i64, iterator = "%arg6", mapped_to = "x", reuse_type = "no_reuse", volume = 0 : i64}, {depth = 1 : i64, iterator = "%arg7", mapped_to = "y", reuse_type = "no_reuse", volume = 0 : i64}], temporal = [{depth = 2 : i64, iterator = "%arg8", reuse_type = "no_reuse", volume = 0 : i64}, {depth = 3 : i64, iterator = "%arg9", reuse_type = "total_reuse", volume = 16384 : i64}]}} : memref<*xf32> to memref<64x64xf32, strided<[512, 1], offset: ?>>
              %alloc_2 = memref.alloc() {tmd.alloc = {local = true, memory_name = "L1", size = 16384 : i64}} : memref<64x64xf32>
              memref.copy %reinterpret_cast_1, %alloc_2 {tmd.copy.choice = {kind = "mem", memory_name = "L1"}} : memref<64x64xf32, strided<[512, 1], offset: ?>> to memref<64x64xf32>
              %19 = bufferization.to_tensor %alloc_2 restrict writable : memref<64x64xf32> to tensor<64x64xf32>
              %20 = linalg.matmul ins(%17, %19 : tensor<64x64xf32>, tensor<64x64xf32>) outs(%13 : tensor<64x64xf32>) -> tensor<64x64xf32>
              %21 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%arg11, %20 : tensor<64x64xf32>, tensor<64x64xf32>) outs(%arg11 : tensor<64x64xf32>) {
              ^bb0(%in: f32, %in_3: f32, %out: f32):
                %22 = arith.addf %in, %in_3 : f32
                linalg.yield %22 : f32
              } -> tensor<64x64xf32>
              scf.yield %21 : tensor<64x64xf32>
            }
            %15 = affine.apply affine_map<(d0, d1, d2, d3) -> (d0 * 32768 + d1 * 512 + d2 * 512 + d3 * 64)>(%arg9, %arg6, %arg7, %arg8)
            %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%15], sizes: [64, 64], strides: [512, 1] {tmd.reuse = {spatial = [{depth = 0 : i64, iterator = "%arg6", mapped_to = "x", reuse_type = "no_reuse", volume = 0 : i64}, {depth = 1 : i64, iterator = "%arg7", mapped_to = "y", reuse_type = "no_reuse", volume = 0 : i64}], temporal = [{depth = 2 : i64, iterator = "%arg8", reuse_type = "no_reuse", volume = 0 : i64}, {depth = 3 : i64, iterator = "%arg9", reuse_type = "no_reuse", volume = 0 : i64}]}} : memref<*xf32> to memref<64x64xf32, strided<[512, 1], offset: ?>>
            bufferization.materialize_in_destination %14 in writable %reinterpret_cast : (tensor<64x64xf32>, memref<64x64xf32, strided<[512, 1], offset: ?>>) -> ()
          }
        }
      } {tmd.mapped_to = "y"}
    } {tmd.mapped_to = "x"}
    return
  }
}
