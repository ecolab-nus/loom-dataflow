module {
  %0 = df.mat "PT" {shape = [128, 128, 128], throughput = 2}
  %1 = df.vec "SFP" {shape = [32], throughput = 2}
  %2 = df.spatial_dim "x", 32
  %3 = df.spatial_dim "y", 2
  %4 = df.core "cores" {scaleout=(%2, %3) , scalein=(%0, %1, [1, 1])}
  %5 = df.memory "L1" {scaleout=(%2) , size = 2097152, bandwidth = 128}
  %6 = df.mux %4 : !df.compute, %5 : !df.memory  {map = affine_map<(d0, d1) -> (d0)>}
  %7 = df.interconnects "small_rings" %4 : !df.compute, %4 : !df.compute  {bandwidth = 32 : i64, map = affine_map<(d0, d1) -> ((d0 + 1) mod 8, d1)>} : !df.interconnect
  %8 = df.interconnects "big_ring" %5 : !df.memory, %5 : !df.memory  {bandwidth = 258 : i64, map = affine_map<(d0) -> ((d0 + 1) mod 8)>} : !df.interconnect
  %9 = df.spatial_dim "d", 2
  %10 = df.memory "DRAM" {scaleout=(%9) , size = 34359738368, bandwidth = 512}
  %11 = df.interconnects %10 : !df.memory, %5 : !df.memory  {map = affine_map<(d0) -> (d0 * 31)>} : !df.interconnect
  func.func @matmul_kernel__d0i0_d1i0__f01__c0mem_c1mem(%arg0: memref<*xf32> {tt.divisibility = 16 : i32}, %arg1: memref<*xf32> {tt.divisibility = 16 : i32}, %arg2: memref<*xf32> {tt.divisibility = 16 : i32}, %arg3: index, %arg4: index, %arg5: index) {
    affine.parallel (%arg6) = (0) to (32) {
      affine.parallel (%arg7) = (0) to (2) {
        affine.for %arg8 = 0 to affine_map<()[s0, s1] -> ((s0 ceildiv 32) ceildiv 2)>()[%arg3, %arg4] {
          affine.for %arg9 = 0 to affine_map<()[s0, s1] -> (s1)>()[%arg3, %arg4] {
            %cst = arith.constant 0.000000e+00 : f32
            %12 = tensor.empty() : tensor<64x64xf32>
            %13 = linalg.fill ins(%cst : f32) outs(%12 : tensor<64x64xf32>) -> tensor<64x64xf32>
            %c0 = arith.constant 0 : index
            %c8 = arith.constant 8 : index
            %c1 = arith.constant 1 : index
            %14 = scf.for %arg10 = %c0 to %c8 step %c1 iter_args(%arg11 = %13) -> (tensor<64x64xf32>) {
              %16 = affine.apply affine_map<(d0, d1, d2, d3) -> (d0 * 64 + d1 * 1048576 + d2 * 65536 + d3 * 32768)>(%arg10, %arg6, %arg7, %arg8)
              %reinterpret_cast_0 = memref.reinterpret_cast %arg0 to offset: [%16], sizes: [64, 64], strides: [512, 1] {loom.reuse = {sequential = [{depth = 4 : i64, iterator = "%arg10", reuse_type = "no_reuse", volume = 0 : i64}], spatial = [{depth = 0 : i64, iterator = "%arg6", mapped_to = "x", reuse_type = "no_reuse", volume = 0 : i64}, {depth = 1 : i64, iterator = "%arg7", mapped_to = "y", reuse_type = "no_reuse", volume = 0 : i64}], temporal = [{depth = 2 : i64, iterator = "%arg8", reuse_type = "no_reuse", volume = 0 : i64}, {depth = 3 : i64, iterator = "%arg9", reuse_type = "total_reuse", volume = 16384 : i64}]}} : memref<*xf32> to memref<64x64xf32, strided<[512, 1], offset: ?>>
              %alloc = memref.alloc() {loom.alloc = {local = true, memory_name = "L1", size = 16384 : i64}} : memref<64x64xf32>
              memref.copy %reinterpret_cast_0, %alloc {loom.copy.choice = {kind = "mem", memory_name = "L1"}} : memref<64x64xf32, strided<[512, 1], offset: ?>> to memref<64x64xf32>
              %17 = bufferization.to_tensor %alloc restrict writable : memref<64x64xf32> to tensor<64x64xf32>
              %18 = affine.apply affine_map<(d0, d1) -> (d1 * 32768 + d0 * 64)>(%arg9, %arg10)
              %reinterpret_cast_1 = memref.reinterpret_cast %arg1 to offset: [%18], sizes: [64, 64], strides: [512, 1] {loom.reuse = {sequential = [{depth = 4 : i64, iterator = "%arg10", reuse_type = "no_reuse", volume = 0 : i64}], spatial = [{depth = 0 : i64, iterator = "%arg6", mapped_to = "x", reuse_type = "total_reuse", volume = 16384 : i64}, {depth = 1 : i64, iterator = "%arg7", mapped_to = "y", reuse_type = "total_reuse", volume = 16384 : i64}], temporal = [{depth = 2 : i64, iterator = "%arg8", reuse_type = "total_reuse", volume = 16384 : i64}, {depth = 3 : i64, iterator = "%arg9", reuse_type = "no_reuse", volume = 0 : i64}]}} : memref<*xf32> to memref<64x64xf32, strided<[512, 1], offset: ?>>
              %alloc_2 = memref.alloc() {loom.alloc = {local = true, memory_name = "L1", size = 16384 : i64}} : memref<64x64xf32>
              memref.copy %reinterpret_cast_1, %alloc_2 {loom.copy.choice = {kind = "mem", memory_name = "L1"}} : memref<64x64xf32, strided<[512, 1], offset: ?>> to memref<64x64xf32>
              %19 = bufferization.to_tensor %alloc_2 restrict writable : memref<64x64xf32> to tensor<64x64xf32>
              %20 = linalg.matmul ins(%17, %19 : tensor<64x64xf32>, tensor<64x64xf32>) outs(%13 : tensor<64x64xf32>) -> tensor<64x64xf32>
              %21 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%arg11, %20 : tensor<64x64xf32>, tensor<64x64xf32>) outs(%arg11 : tensor<64x64xf32>) {
              ^bb0(%in: f32, %in_3: f32, %out: f32):
                %22 = arith.addf %in, %in_3 : f32
                linalg.yield %22 : f32
              } -> tensor<64x64xf32>
              scf.yield %21 : tensor<64x64xf32>
            }
            %15 = affine.apply affine_map<(d0, d1, d2, d3) -> (d0 * 64 + d1 * 1048576 + d2 * 65536 + d3 * 32768)>(%arg9, %arg6, %arg7, %arg8)
            %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%15], sizes: [64, 64], strides: [512, 1] {loom.reuse = {spatial = [{depth = 0 : i64, iterator = "%arg6", mapped_to = "x", reuse_type = "no_reuse", volume = 0 : i64}, {depth = 1 : i64, iterator = "%arg7", mapped_to = "y", reuse_type = "no_reuse", volume = 0 : i64}], temporal = [{depth = 2 : i64, iterator = "%arg8", reuse_type = "no_reuse", volume = 0 : i64}, {depth = 3 : i64, iterator = "%arg9", reuse_type = "no_reuse", volume = 0 : i64}]}} : memref<*xf32> to memref<64x64xf32, strided<[512, 1], offset: ?>>
            bufferization.materialize_in_destination %14 in writable %reinterpret_cast : (tensor<64x64xf32>, memref<64x64xf32, strided<[512, 1], offset: ?>>) -> ()
          }
        }
      } {loom.mapped_to = "y"}
    } {loom.mapped_to = "x"}
    return
  }
  func.func @matmul_kernel__d0i0_d1i0__f01__c0mem_c1bx(%arg0: memref<*xf32> {tt.divisibility = 16 : i32}, %arg1: memref<*xf32> {tt.divisibility = 16 : i32}, %arg2: memref<*xf32> {tt.divisibility = 16 : i32}, %arg3: index, %arg4: index, %arg5: index) {
    affine.parallel (%arg6) = (0) to (32) {
      affine.parallel (%arg7) = (0) to (2) {
        affine.for %arg8 = 0 to affine_map<()[s0, s1] -> ((s0 ceildiv 32) ceildiv 2)>()[%arg3, %arg4] {
          affine.for %arg9 = 0 to affine_map<()[s0, s1] -> (s1)>()[%arg3, %arg4] {
            %cst = arith.constant 0.000000e+00 : f32
            %12 = tensor.empty() : tensor<64x64xf32>
            %13 = linalg.fill ins(%cst : f32) outs(%12 : tensor<64x64xf32>) -> tensor<64x64xf32>
            %c0 = arith.constant 0 : index
            %c8 = arith.constant 8 : index
            %c1 = arith.constant 1 : index
            %14 = scf.for %arg10 = %c0 to %c8 step %c1 iter_args(%arg11 = %13) -> (tensor<64x64xf32>) {
              %16 = affine.apply affine_map<(d0, d1, d2, d3) -> (d0 * 64 + d1 * 1048576 + d2 * 65536 + d3 * 32768)>(%arg10, %arg6, %arg7, %arg8)
              %reinterpret_cast_0 = memref.reinterpret_cast %arg0 to offset: [%16], sizes: [64, 64], strides: [512, 1] {loom.reuse = {sequential = [{depth = 4 : i64, iterator = "%arg10", reuse_type = "no_reuse", volume = 0 : i64}], spatial = [{depth = 0 : i64, iterator = "%arg6", mapped_to = "x", reuse_type = "no_reuse", volume = 0 : i64}, {depth = 1 : i64, iterator = "%arg7", mapped_to = "y", reuse_type = "no_reuse", volume = 0 : i64}], temporal = [{depth = 2 : i64, iterator = "%arg8", reuse_type = "no_reuse", volume = 0 : i64}, {depth = 3 : i64, iterator = "%arg9", reuse_type = "total_reuse", volume = 16384 : i64}]}} : memref<*xf32> to memref<64x64xf32, strided<[512, 1], offset: ?>>
              %alloc = memref.alloc() {loom.alloc = {local = true, memory_name = "L1", size = 16384 : i64}} : memref<64x64xf32>
              memref.copy %reinterpret_cast_0, %alloc {loom.copy.choice = {kind = "mem", memory_name = "L1"}} : memref<64x64xf32, strided<[512, 1], offset: ?>> to memref<64x64xf32>
              %17 = bufferization.to_tensor %alloc restrict writable : memref<64x64xf32> to tensor<64x64xf32>
              %18 = affine.apply affine_map<(d0, d1) -> (d1 * 32768 + d0 * 64)>(%arg9, %arg10)
              %reinterpret_cast_1 = memref.reinterpret_cast %arg1 to offset: [%18], sizes: [64, 64], strides: [512, 1] {loom.reuse = {sequential = [{depth = 4 : i64, iterator = "%arg10", reuse_type = "no_reuse", volume = 0 : i64}], spatial = [{depth = 0 : i64, iterator = "%arg6", mapped_to = "x", reuse_type = "total_reuse", volume = 16384 : i64}, {depth = 1 : i64, iterator = "%arg7", mapped_to = "y", reuse_type = "total_reuse", volume = 16384 : i64}], temporal = [{depth = 2 : i64, iterator = "%arg8", reuse_type = "total_reuse", volume = 16384 : i64}, {depth = 3 : i64, iterator = "%arg9", reuse_type = "no_reuse", volume = 0 : i64}]}} : memref<*xf32> to memref<64x64xf32, strided<[512, 1], offset: ?>>
              %alloc_2 = memref.alloc() {loom.alloc = {local = true, memory_name = "L1", size = 16384 : i64}} : memref<64x64xf32>
              memref.copy %reinterpret_cast_1, %alloc_2 {loom.copy.choice = {dim = "x", interconnect_name = "big_ring", kind = "broadcast"}} : memref<64x64xf32, strided<[512, 1], offset: ?>> to memref<64x64xf32>
              %19 = bufferization.to_tensor %alloc_2 restrict writable : memref<64x64xf32> to tensor<64x64xf32>
              %20 = linalg.matmul ins(%17, %19 : tensor<64x64xf32>, tensor<64x64xf32>) outs(%13 : tensor<64x64xf32>) -> tensor<64x64xf32>
              %21 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%arg11, %20 : tensor<64x64xf32>, tensor<64x64xf32>) outs(%arg11 : tensor<64x64xf32>) {
              ^bb0(%in: f32, %in_3: f32, %out: f32):
                %22 = arith.addf %in, %in_3 : f32
                linalg.yield %22 : f32
              } -> tensor<64x64xf32>
              scf.yield %21 : tensor<64x64xf32>
            }
            %15 = affine.apply affine_map<(d0, d1, d2, d3) -> (d0 * 64 + d1 * 1048576 + d2 * 65536 + d3 * 32768)>(%arg9, %arg6, %arg7, %arg8)
            %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%15], sizes: [64, 64], strides: [512, 1] {loom.reuse = {spatial = [{depth = 0 : i64, iterator = "%arg6", mapped_to = "x", reuse_type = "no_reuse", volume = 0 : i64}, {depth = 1 : i64, iterator = "%arg7", mapped_to = "y", reuse_type = "no_reuse", volume = 0 : i64}], temporal = [{depth = 2 : i64, iterator = "%arg8", reuse_type = "no_reuse", volume = 0 : i64}, {depth = 3 : i64, iterator = "%arg9", reuse_type = "no_reuse", volume = 0 : i64}]}} : memref<*xf32> to memref<64x64xf32, strided<[512, 1], offset: ?>>
            bufferization.materialize_in_destination %14 in writable %reinterpret_cast : (tensor<64x64xf32>, memref<64x64xf32, strided<[512, 1], offset: ?>>) -> ()
          }
        }
      } {loom.mapped_to = "y"}
    } {loom.mapped_to = "x"}
    return
  }
  func.func @matmul_kernel__d0i0_d1i0__f01__hoist_block_0__c0mem_c1mem(%arg0: memref<*xf32> {tt.divisibility = 16 : i32}, %arg1: memref<*xf32> {tt.divisibility = 16 : i32}, %arg2: memref<*xf32> {tt.divisibility = 16 : i32}, %arg3: index, %arg4: index, %arg5: index) {
    affine.parallel (%arg6) = (0) to (32) {
      affine.parallel (%arg7) = (0) to (2) {
        affine.for %arg8 = 0 to affine_map<()[s0, s1] -> ((s0 ceildiv 32) ceildiv 2)>()[%arg3, %arg4] {
          %12 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 1048576 + d1 * 65536 + d2 * 32768)>(%arg6, %arg7, %arg8)
          %reinterpret_cast = memref.reinterpret_cast %arg0 to offset: [%12], sizes: [8, 64, 64], strides: [64, 512, 1] {loom.reuse = {spatial = [{depth = 0 : i64, iterator = "%arg6", mapped_to = "x", reuse_type = "no_reuse", volume = 0 : i64}, {depth = 1 : i64, iterator = "%arg7", mapped_to = "y", reuse_type = "no_reuse", volume = 0 : i64}], temporal = [{depth = 2 : i64, iterator = "%arg8", reuse_type = "no_reuse", volume = 0 : i64}]}} : memref<*xf32> to memref<8x64x64xf32, strided<[64, 512, 1], offset: ?>>
          %alloc = memref.alloc() {loom.alloc = {local = true, memory_name = "L1", size = 131072 : i64}} : memref<8x64x64xf32>
          memref.copy %reinterpret_cast, %alloc {loom.copy.choice = {kind = "mem", memory_name = "L1"}} : memref<8x64x64xf32, strided<[64, 512, 1], offset: ?>> to memref<8x64x64xf32>
          affine.for %arg9 = 0 to affine_map<()[s0, s1] -> (s1)>()[%arg3, %arg4] {
            %cst = arith.constant 0.000000e+00 : f32
            %13 = tensor.empty() : tensor<64x64xf32>
            %14 = linalg.fill ins(%cst : f32) outs(%13 : tensor<64x64xf32>) -> tensor<64x64xf32>
            %c0 = arith.constant 0 : index
            %c8 = arith.constant 8 : index
            %c1 = arith.constant 1 : index
            %15 = scf.for %arg10 = %c0 to %c8 step %c1 iter_args(%arg11 = %14) -> (tensor<64x64xf32>) {
              %subview = memref.subview %alloc[%arg10, 0, 0] [1, 64, 64] [1, 1, 1] : memref<8x64x64xf32> to memref<64x64xf32, strided<[64, 1], offset: ?>>
              %17 = bufferization.to_tensor %subview restrict writable : memref<64x64xf32, strided<[64, 1], offset: ?>> to tensor<64x64xf32>
              %18 = affine.apply affine_map<(d0, d1) -> (d1 * 32768 + d0 * 64)>(%arg9, %arg10)
              %reinterpret_cast_1 = memref.reinterpret_cast %arg1 to offset: [%18], sizes: [64, 64], strides: [512, 1] {loom.reuse = {sequential = [{depth = 4 : i64, iterator = "%arg10", reuse_type = "no_reuse", volume = 0 : i64}], spatial = [{depth = 0 : i64, iterator = "%arg6", mapped_to = "x", reuse_type = "total_reuse", volume = 16384 : i64}, {depth = 1 : i64, iterator = "%arg7", mapped_to = "y", reuse_type = "total_reuse", volume = 16384 : i64}], temporal = [{depth = 2 : i64, iterator = "%arg8", reuse_type = "total_reuse", volume = 16384 : i64}, {depth = 3 : i64, iterator = "%arg9", reuse_type = "no_reuse", volume = 0 : i64}]}} : memref<*xf32> to memref<64x64xf32, strided<[512, 1], offset: ?>>
              %alloc_2 = memref.alloc() {loom.alloc = {local = true, memory_name = "L1", size = 16384 : i64}} : memref<64x64xf32>
              memref.copy %reinterpret_cast_1, %alloc_2 {loom.copy.choice = {kind = "mem", memory_name = "L1"}} : memref<64x64xf32, strided<[512, 1], offset: ?>> to memref<64x64xf32>
              %19 = bufferization.to_tensor %alloc_2 restrict writable : memref<64x64xf32> to tensor<64x64xf32>
              %20 = linalg.matmul ins(%17, %19 : tensor<64x64xf32>, tensor<64x64xf32>) outs(%14 : tensor<64x64xf32>) -> tensor<64x64xf32>
              %21 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%arg11, %20 : tensor<64x64xf32>, tensor<64x64xf32>) outs(%arg11 : tensor<64x64xf32>) {
              ^bb0(%in: f32, %in_3: f32, %out: f32):
                %22 = arith.addf %in, %in_3 : f32
                linalg.yield %22 : f32
              } -> tensor<64x64xf32>
              scf.yield %21 : tensor<64x64xf32>
            }
            %16 = affine.apply affine_map<(d0, d1, d2, d3) -> (d0 * 64 + d1 * 1048576 + d2 * 65536 + d3 * 32768)>(%arg9, %arg6, %arg7, %arg8)
            %reinterpret_cast_0 = memref.reinterpret_cast %arg2 to offset: [%16], sizes: [64, 64], strides: [512, 1] {loom.reuse = {spatial = [{depth = 0 : i64, iterator = "%arg6", mapped_to = "x", reuse_type = "no_reuse", volume = 0 : i64}, {depth = 1 : i64, iterator = "%arg7", mapped_to = "y", reuse_type = "no_reuse", volume = 0 : i64}], temporal = [{depth = 2 : i64, iterator = "%arg8", reuse_type = "no_reuse", volume = 0 : i64}, {depth = 3 : i64, iterator = "%arg9", reuse_type = "no_reuse", volume = 0 : i64}]}} : memref<*xf32> to memref<64x64xf32, strided<[512, 1], offset: ?>>
            bufferization.materialize_in_destination %15 in writable %reinterpret_cast_0 : (tensor<64x64xf32>, memref<64x64xf32, strided<[512, 1], offset: ?>>) -> ()
          }
        }
      } {loom.mapped_to = "y"}
    } {loom.mapped_to = "x"}
    return
  }
  func.func @matmul_kernel__d0i0_d1i0__f01__hoist_block_0__c0mem_c1bx(%arg0: memref<*xf32> {tt.divisibility = 16 : i32}, %arg1: memref<*xf32> {tt.divisibility = 16 : i32}, %arg2: memref<*xf32> {tt.divisibility = 16 : i32}, %arg3: index, %arg4: index, %arg5: index) {
    affine.parallel (%arg6) = (0) to (32) {
      affine.parallel (%arg7) = (0) to (2) {
        affine.for %arg8 = 0 to affine_map<()[s0, s1] -> ((s0 ceildiv 32) ceildiv 2)>()[%arg3, %arg4] {
          %12 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 1048576 + d1 * 65536 + d2 * 32768)>(%arg6, %arg7, %arg8)
          %reinterpret_cast = memref.reinterpret_cast %arg0 to offset: [%12], sizes: [8, 64, 64], strides: [64, 512, 1] {loom.reuse = {spatial = [{depth = 0 : i64, iterator = "%arg6", mapped_to = "x", reuse_type = "no_reuse", volume = 0 : i64}, {depth = 1 : i64, iterator = "%arg7", mapped_to = "y", reuse_type = "no_reuse", volume = 0 : i64}], temporal = [{depth = 2 : i64, iterator = "%arg8", reuse_type = "no_reuse", volume = 0 : i64}]}} : memref<*xf32> to memref<8x64x64xf32, strided<[64, 512, 1], offset: ?>>
          %alloc = memref.alloc() {loom.alloc = {local = true, memory_name = "L1", size = 131072 : i64}} : memref<8x64x64xf32>
          memref.copy %reinterpret_cast, %alloc {loom.copy.choice = {kind = "mem", memory_name = "L1"}} : memref<8x64x64xf32, strided<[64, 512, 1], offset: ?>> to memref<8x64x64xf32>
          affine.for %arg9 = 0 to affine_map<()[s0, s1] -> (s1)>()[%arg3, %arg4] {
            %cst = arith.constant 0.000000e+00 : f32
            %13 = tensor.empty() : tensor<64x64xf32>
            %14 = linalg.fill ins(%cst : f32) outs(%13 : tensor<64x64xf32>) -> tensor<64x64xf32>
            %c0 = arith.constant 0 : index
            %c8 = arith.constant 8 : index
            %c1 = arith.constant 1 : index
            %15 = scf.for %arg10 = %c0 to %c8 step %c1 iter_args(%arg11 = %14) -> (tensor<64x64xf32>) {
              %subview = memref.subview %alloc[%arg10, 0, 0] [1, 64, 64] [1, 1, 1] : memref<8x64x64xf32> to memref<64x64xf32, strided<[64, 1], offset: ?>>
              %17 = bufferization.to_tensor %subview restrict writable : memref<64x64xf32, strided<[64, 1], offset: ?>> to tensor<64x64xf32>
              %18 = affine.apply affine_map<(d0, d1) -> (d1 * 32768 + d0 * 64)>(%arg9, %arg10)
              %reinterpret_cast_1 = memref.reinterpret_cast %arg1 to offset: [%18], sizes: [64, 64], strides: [512, 1] {loom.reuse = {sequential = [{depth = 4 : i64, iterator = "%arg10", reuse_type = "no_reuse", volume = 0 : i64}], spatial = [{depth = 0 : i64, iterator = "%arg6", mapped_to = "x", reuse_type = "total_reuse", volume = 16384 : i64}, {depth = 1 : i64, iterator = "%arg7", mapped_to = "y", reuse_type = "total_reuse", volume = 16384 : i64}], temporal = [{depth = 2 : i64, iterator = "%arg8", reuse_type = "total_reuse", volume = 16384 : i64}, {depth = 3 : i64, iterator = "%arg9", reuse_type = "no_reuse", volume = 0 : i64}]}} : memref<*xf32> to memref<64x64xf32, strided<[512, 1], offset: ?>>
              %alloc_2 = memref.alloc() {loom.alloc = {local = true, memory_name = "L1", size = 16384 : i64}} : memref<64x64xf32>
              memref.copy %reinterpret_cast_1, %alloc_2 {loom.copy.choice = {dim = "x", interconnect_name = "big_ring", kind = "broadcast"}} : memref<64x64xf32, strided<[512, 1], offset: ?>> to memref<64x64xf32>
              %19 = bufferization.to_tensor %alloc_2 restrict writable : memref<64x64xf32> to tensor<64x64xf32>
              %20 = linalg.matmul ins(%17, %19 : tensor<64x64xf32>, tensor<64x64xf32>) outs(%14 : tensor<64x64xf32>) -> tensor<64x64xf32>
              %21 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%arg11, %20 : tensor<64x64xf32>, tensor<64x64xf32>) outs(%arg11 : tensor<64x64xf32>) {
              ^bb0(%in: f32, %in_3: f32, %out: f32):
                %22 = arith.addf %in, %in_3 : f32
                linalg.yield %22 : f32
              } -> tensor<64x64xf32>
              scf.yield %21 : tensor<64x64xf32>
            }
            %16 = affine.apply affine_map<(d0, d1, d2, d3) -> (d0 * 64 + d1 * 1048576 + d2 * 65536 + d3 * 32768)>(%arg9, %arg6, %arg7, %arg8)
            %reinterpret_cast_0 = memref.reinterpret_cast %arg2 to offset: [%16], sizes: [64, 64], strides: [512, 1] {loom.reuse = {spatial = [{depth = 0 : i64, iterator = "%arg6", mapped_to = "x", reuse_type = "no_reuse", volume = 0 : i64}, {depth = 1 : i64, iterator = "%arg7", mapped_to = "y", reuse_type = "no_reuse", volume = 0 : i64}], temporal = [{depth = 2 : i64, iterator = "%arg8", reuse_type = "no_reuse", volume = 0 : i64}, {depth = 3 : i64, iterator = "%arg9", reuse_type = "no_reuse", volume = 0 : i64}]}} : memref<*xf32> to memref<64x64xf32, strided<[512, 1], offset: ?>>
            bufferization.materialize_in_destination %15 in writable %reinterpret_cast_0 : (tensor<64x64xf32>, memref<64x64xf32, strided<[512, 1], offset: ?>>) -> ()
          }
        }
      } {loom.mapped_to = "y"}
    } {loom.mapped_to = "x"}
    return
  }
  func.func @matmul_kernel__d0i0_d1i0__f10__c0mem_c1mem(%arg0: memref<*xf32> {tt.divisibility = 16 : i32}, %arg1: memref<*xf32> {tt.divisibility = 16 : i32}, %arg2: memref<*xf32> {tt.divisibility = 16 : i32}, %arg3: index, %arg4: index, %arg5: index) {
    affine.parallel (%arg6) = (0) to (32) {
      affine.parallel (%arg7) = (0) to (2) {
        affine.for %arg8 = 0 to affine_map<()[s0, s1] -> (s1)>()[%arg3, %arg4] {
          affine.for %arg9 = 0 to affine_map<()[s0, s1] -> ((s0 ceildiv 32) ceildiv 2)>()[%arg3, %arg4] {
            %cst = arith.constant 0.000000e+00 : f32
            %12 = tensor.empty() : tensor<64x64xf32>
            %13 = linalg.fill ins(%cst : f32) outs(%12 : tensor<64x64xf32>) -> tensor<64x64xf32>
            %c0 = arith.constant 0 : index
            %c8 = arith.constant 8 : index
            %c1 = arith.constant 1 : index
            %14 = scf.for %arg10 = %c0 to %c8 step %c1 iter_args(%arg11 = %13) -> (tensor<64x64xf32>) {
              %16 = affine.apply affine_map<(d0, d1, d2, d3) -> (d0 * 64 + d1 * 1048576 + d2 * 65536 + d3 * 32768)>(%arg10, %arg6, %arg7, %arg9)
              %reinterpret_cast_0 = memref.reinterpret_cast %arg0 to offset: [%16], sizes: [64, 64], strides: [512, 1] {loom.reuse = {sequential = [{depth = 4 : i64, iterator = "%arg10", reuse_type = "no_reuse", volume = 0 : i64}], spatial = [{depth = 0 : i64, iterator = "%arg6", mapped_to = "x", reuse_type = "no_reuse", volume = 0 : i64}, {depth = 1 : i64, iterator = "%arg7", mapped_to = "y", reuse_type = "no_reuse", volume = 0 : i64}], temporal = [{depth = 2 : i64, iterator = "%arg8", reuse_type = "total_reuse", volume = 16384 : i64}, {depth = 3 : i64, iterator = "%arg9", reuse_type = "no_reuse", volume = 0 : i64}]}} : memref<*xf32> to memref<64x64xf32, strided<[512, 1], offset: ?>>
              %alloc = memref.alloc() {loom.alloc = {local = true, memory_name = "L1", size = 16384 : i64}} : memref<64x64xf32>
              memref.copy %reinterpret_cast_0, %alloc {loom.copy.choice = {kind = "mem", memory_name = "L1"}} : memref<64x64xf32, strided<[512, 1], offset: ?>> to memref<64x64xf32>
              %17 = bufferization.to_tensor %alloc restrict writable : memref<64x64xf32> to tensor<64x64xf32>
              %18 = affine.apply affine_map<(d0, d1) -> (d1 * 32768 + d0 * 64)>(%arg8, %arg10)
              %reinterpret_cast_1 = memref.reinterpret_cast %arg1 to offset: [%18], sizes: [64, 64], strides: [512, 1] {loom.reuse = {sequential = [{depth = 4 : i64, iterator = "%arg10", reuse_type = "no_reuse", volume = 0 : i64}], spatial = [{depth = 0 : i64, iterator = "%arg6", mapped_to = "x", reuse_type = "total_reuse", volume = 16384 : i64}, {depth = 1 : i64, iterator = "%arg7", mapped_to = "y", reuse_type = "total_reuse", volume = 16384 : i64}], temporal = [{depth = 2 : i64, iterator = "%arg8", reuse_type = "no_reuse", volume = 0 : i64}, {depth = 3 : i64, iterator = "%arg9", reuse_type = "total_reuse", volume = 16384 : i64}]}} : memref<*xf32> to memref<64x64xf32, strided<[512, 1], offset: ?>>
              %alloc_2 = memref.alloc() {loom.alloc = {local = true, memory_name = "L1", size = 16384 : i64}} : memref<64x64xf32>
              memref.copy %reinterpret_cast_1, %alloc_2 {loom.copy.choice = {kind = "mem", memory_name = "L1"}} : memref<64x64xf32, strided<[512, 1], offset: ?>> to memref<64x64xf32>
              %19 = bufferization.to_tensor %alloc_2 restrict writable : memref<64x64xf32> to tensor<64x64xf32>
              %20 = linalg.matmul ins(%17, %19 : tensor<64x64xf32>, tensor<64x64xf32>) outs(%13 : tensor<64x64xf32>) -> tensor<64x64xf32>
              %21 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%arg11, %20 : tensor<64x64xf32>, tensor<64x64xf32>) outs(%arg11 : tensor<64x64xf32>) {
              ^bb0(%in: f32, %in_3: f32, %out: f32):
                %22 = arith.addf %in, %in_3 : f32
                linalg.yield %22 : f32
              } -> tensor<64x64xf32>
              scf.yield %21 : tensor<64x64xf32>
            }
            %15 = affine.apply affine_map<(d0, d1, d2, d3) -> (d0 * 64 + d1 * 1048576 + d2 * 65536 + d3 * 32768)>(%arg8, %arg6, %arg7, %arg9)
            %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%15], sizes: [64, 64], strides: [512, 1] {loom.reuse = {spatial = [{depth = 0 : i64, iterator = "%arg6", mapped_to = "x", reuse_type = "no_reuse", volume = 0 : i64}, {depth = 1 : i64, iterator = "%arg7", mapped_to = "y", reuse_type = "no_reuse", volume = 0 : i64}], temporal = [{depth = 2 : i64, iterator = "%arg8", reuse_type = "no_reuse", volume = 0 : i64}, {depth = 3 : i64, iterator = "%arg9", reuse_type = "no_reuse", volume = 0 : i64}]}} : memref<*xf32> to memref<64x64xf32, strided<[512, 1], offset: ?>>
            bufferization.materialize_in_destination %14 in writable %reinterpret_cast : (tensor<64x64xf32>, memref<64x64xf32, strided<[512, 1], offset: ?>>) -> ()
          }
        }
      } {loom.mapped_to = "y"}
    } {loom.mapped_to = "x"}
    return
  }
  func.func @matmul_kernel__d0i0_d1i0__f10__c0mem_c1bx(%arg0: memref<*xf32> {tt.divisibility = 16 : i32}, %arg1: memref<*xf32> {tt.divisibility = 16 : i32}, %arg2: memref<*xf32> {tt.divisibility = 16 : i32}, %arg3: index, %arg4: index, %arg5: index) {
    affine.parallel (%arg6) = (0) to (32) {
      affine.parallel (%arg7) = (0) to (2) {
        affine.for %arg8 = 0 to affine_map<()[s0, s1] -> (s1)>()[%arg3, %arg4] {
          affine.for %arg9 = 0 to affine_map<()[s0, s1] -> ((s0 ceildiv 32) ceildiv 2)>()[%arg3, %arg4] {
            %cst = arith.constant 0.000000e+00 : f32
            %12 = tensor.empty() : tensor<64x64xf32>
            %13 = linalg.fill ins(%cst : f32) outs(%12 : tensor<64x64xf32>) -> tensor<64x64xf32>
            %c0 = arith.constant 0 : index
            %c8 = arith.constant 8 : index
            %c1 = arith.constant 1 : index
            %14 = scf.for %arg10 = %c0 to %c8 step %c1 iter_args(%arg11 = %13) -> (tensor<64x64xf32>) {
              %16 = affine.apply affine_map<(d0, d1, d2, d3) -> (d0 * 64 + d1 * 1048576 + d2 * 65536 + d3 * 32768)>(%arg10, %arg6, %arg7, %arg9)
              %reinterpret_cast_0 = memref.reinterpret_cast %arg0 to offset: [%16], sizes: [64, 64], strides: [512, 1] {loom.reuse = {sequential = [{depth = 4 : i64, iterator = "%arg10", reuse_type = "no_reuse", volume = 0 : i64}], spatial = [{depth = 0 : i64, iterator = "%arg6", mapped_to = "x", reuse_type = "no_reuse", volume = 0 : i64}, {depth = 1 : i64, iterator = "%arg7", mapped_to = "y", reuse_type = "no_reuse", volume = 0 : i64}], temporal = [{depth = 2 : i64, iterator = "%arg8", reuse_type = "total_reuse", volume = 16384 : i64}, {depth = 3 : i64, iterator = "%arg9", reuse_type = "no_reuse", volume = 0 : i64}]}} : memref<*xf32> to memref<64x64xf32, strided<[512, 1], offset: ?>>
              %alloc = memref.alloc() {loom.alloc = {local = true, memory_name = "L1", size = 16384 : i64}} : memref<64x64xf32>
              memref.copy %reinterpret_cast_0, %alloc {loom.copy.choice = {kind = "mem", memory_name = "L1"}} : memref<64x64xf32, strided<[512, 1], offset: ?>> to memref<64x64xf32>
              %17 = bufferization.to_tensor %alloc restrict writable : memref<64x64xf32> to tensor<64x64xf32>
              %18 = affine.apply affine_map<(d0, d1) -> (d1 * 32768 + d0 * 64)>(%arg8, %arg10)
              %reinterpret_cast_1 = memref.reinterpret_cast %arg1 to offset: [%18], sizes: [64, 64], strides: [512, 1] {loom.reuse = {sequential = [{depth = 4 : i64, iterator = "%arg10", reuse_type = "no_reuse", volume = 0 : i64}], spatial = [{depth = 0 : i64, iterator = "%arg6", mapped_to = "x", reuse_type = "total_reuse", volume = 16384 : i64}, {depth = 1 : i64, iterator = "%arg7", mapped_to = "y", reuse_type = "total_reuse", volume = 16384 : i64}], temporal = [{depth = 2 : i64, iterator = "%arg8", reuse_type = "no_reuse", volume = 0 : i64}, {depth = 3 : i64, iterator = "%arg9", reuse_type = "total_reuse", volume = 16384 : i64}]}} : memref<*xf32> to memref<64x64xf32, strided<[512, 1], offset: ?>>
              %alloc_2 = memref.alloc() {loom.alloc = {local = true, memory_name = "L1", size = 16384 : i64}} : memref<64x64xf32>
              memref.copy %reinterpret_cast_1, %alloc_2 {loom.copy.choice = {dim = "x", interconnect_name = "big_ring", kind = "broadcast"}} : memref<64x64xf32, strided<[512, 1], offset: ?>> to memref<64x64xf32>
              %19 = bufferization.to_tensor %alloc_2 restrict writable : memref<64x64xf32> to tensor<64x64xf32>
              %20 = linalg.matmul ins(%17, %19 : tensor<64x64xf32>, tensor<64x64xf32>) outs(%13 : tensor<64x64xf32>) -> tensor<64x64xf32>
              %21 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%arg11, %20 : tensor<64x64xf32>, tensor<64x64xf32>) outs(%arg11 : tensor<64x64xf32>) {
              ^bb0(%in: f32, %in_3: f32, %out: f32):
                %22 = arith.addf %in, %in_3 : f32
                linalg.yield %22 : f32
              } -> tensor<64x64xf32>
              scf.yield %21 : tensor<64x64xf32>
            }
            %15 = affine.apply affine_map<(d0, d1, d2, d3) -> (d0 * 64 + d1 * 1048576 + d2 * 65536 + d3 * 32768)>(%arg8, %arg6, %arg7, %arg9)
            %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%15], sizes: [64, 64], strides: [512, 1] {loom.reuse = {spatial = [{depth = 0 : i64, iterator = "%arg6", mapped_to = "x", reuse_type = "no_reuse", volume = 0 : i64}, {depth = 1 : i64, iterator = "%arg7", mapped_to = "y", reuse_type = "no_reuse", volume = 0 : i64}], temporal = [{depth = 2 : i64, iterator = "%arg8", reuse_type = "no_reuse", volume = 0 : i64}, {depth = 3 : i64, iterator = "%arg9", reuse_type = "no_reuse", volume = 0 : i64}]}} : memref<*xf32> to memref<64x64xf32, strided<[512, 1], offset: ?>>
            bufferization.materialize_in_destination %14 in writable %reinterpret_cast : (tensor<64x64xf32>, memref<64x64xf32, strided<[512, 1], offset: ?>>) -> ()
          }
        }
      } {loom.mapped_to = "y"}
    } {loom.mapped_to = "x"}
    return
  }
  func.func @matmul_kernel__d0i0_d1i0__f10__hoist_block_1__c0mem_c1mem(%arg0: memref<*xf32> {tt.divisibility = 16 : i32}, %arg1: memref<*xf32> {tt.divisibility = 16 : i32}, %arg2: memref<*xf32> {tt.divisibility = 16 : i32}, %arg3: index, %arg4: index, %arg5: index) {
    affine.parallel (%arg6) = (0) to (32) {
      affine.parallel (%arg7) = (0) to (2) {
        affine.for %arg8 = 0 to affine_map<()[s0, s1] -> (s1)>()[%arg3, %arg4] {
          %12 = affine.apply affine_map<(d0) -> (d0 * 64)>(%arg8)
          %reinterpret_cast = memref.reinterpret_cast %arg1 to offset: [%12], sizes: [8, 64, 64], strides: [32768, 512, 1] {loom.reuse = {spatial = [{depth = 0 : i64, iterator = "%arg6", mapped_to = "x", reuse_type = "total_reuse", volume = 131072 : i64}, {depth = 1 : i64, iterator = "%arg7", mapped_to = "y", reuse_type = "total_reuse", volume = 131072 : i64}], temporal = [{depth = 2 : i64, iterator = "%arg8", reuse_type = "no_reuse", volume = 0 : i64}]}} : memref<*xf32> to memref<8x64x64xf32, strided<[32768, 512, 1], offset: ?>>
          %alloc = memref.alloc() {loom.alloc = {local = true, memory_name = "L1", size = 131072 : i64}} : memref<8x64x64xf32>
          memref.copy %reinterpret_cast, %alloc {loom.copy.choice = {kind = "mem", memory_name = "L1"}} : memref<8x64x64xf32, strided<[32768, 512, 1], offset: ?>> to memref<8x64x64xf32>
          affine.for %arg9 = 0 to affine_map<()[s0, s1] -> ((s0 ceildiv 32) ceildiv 2)>()[%arg3, %arg4] {
            %cst = arith.constant 0.000000e+00 : f32
            %13 = tensor.empty() : tensor<64x64xf32>
            %14 = linalg.fill ins(%cst : f32) outs(%13 : tensor<64x64xf32>) -> tensor<64x64xf32>
            %c0 = arith.constant 0 : index
            %c8 = arith.constant 8 : index
            %c1 = arith.constant 1 : index
            %15 = scf.for %arg10 = %c0 to %c8 step %c1 iter_args(%arg11 = %14) -> (tensor<64x64xf32>) {
              %17 = affine.apply affine_map<(d0, d1, d2, d3) -> (d0 * 64 + d1 * 1048576 + d2 * 65536 + d3 * 32768)>(%arg10, %arg6, %arg7, %arg9)
              %reinterpret_cast_1 = memref.reinterpret_cast %arg0 to offset: [%17], sizes: [64, 64], strides: [512, 1] {loom.reuse = {sequential = [{depth = 4 : i64, iterator = "%arg10", reuse_type = "no_reuse", volume = 0 : i64}], spatial = [{depth = 0 : i64, iterator = "%arg6", mapped_to = "x", reuse_type = "no_reuse", volume = 0 : i64}, {depth = 1 : i64, iterator = "%arg7", mapped_to = "y", reuse_type = "no_reuse", volume = 0 : i64}], temporal = [{depth = 2 : i64, iterator = "%arg8", reuse_type = "total_reuse", volume = 16384 : i64}, {depth = 3 : i64, iterator = "%arg9", reuse_type = "no_reuse", volume = 0 : i64}]}} : memref<*xf32> to memref<64x64xf32, strided<[512, 1], offset: ?>>
              %alloc_2 = memref.alloc() {loom.alloc = {local = true, memory_name = "L1", size = 16384 : i64}} : memref<64x64xf32>
              memref.copy %reinterpret_cast_1, %alloc_2 {loom.copy.choice = {kind = "mem", memory_name = "L1"}} : memref<64x64xf32, strided<[512, 1], offset: ?>> to memref<64x64xf32>
              %18 = bufferization.to_tensor %alloc_2 restrict writable : memref<64x64xf32> to tensor<64x64xf32>
              %subview = memref.subview %alloc[%arg10, 0, 0] [1, 64, 64] [1, 1, 1] : memref<8x64x64xf32> to memref<64x64xf32, strided<[64, 1], offset: ?>>
              %19 = bufferization.to_tensor %subview restrict writable : memref<64x64xf32, strided<[64, 1], offset: ?>> to tensor<64x64xf32>
              %20 = linalg.matmul ins(%18, %19 : tensor<64x64xf32>, tensor<64x64xf32>) outs(%14 : tensor<64x64xf32>) -> tensor<64x64xf32>
              %21 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%arg11, %20 : tensor<64x64xf32>, tensor<64x64xf32>) outs(%arg11 : tensor<64x64xf32>) {
              ^bb0(%in: f32, %in_3: f32, %out: f32):
                %22 = arith.addf %in, %in_3 : f32
                linalg.yield %22 : f32
              } -> tensor<64x64xf32>
              scf.yield %21 : tensor<64x64xf32>
            }
            %16 = affine.apply affine_map<(d0, d1, d2, d3) -> (d0 * 64 + d1 * 1048576 + d2 * 65536 + d3 * 32768)>(%arg8, %arg6, %arg7, %arg9)
            %reinterpret_cast_0 = memref.reinterpret_cast %arg2 to offset: [%16], sizes: [64, 64], strides: [512, 1] {loom.reuse = {spatial = [{depth = 0 : i64, iterator = "%arg6", mapped_to = "x", reuse_type = "no_reuse", volume = 0 : i64}, {depth = 1 : i64, iterator = "%arg7", mapped_to = "y", reuse_type = "no_reuse", volume = 0 : i64}], temporal = [{depth = 2 : i64, iterator = "%arg8", reuse_type = "no_reuse", volume = 0 : i64}, {depth = 3 : i64, iterator = "%arg9", reuse_type = "no_reuse", volume = 0 : i64}]}} : memref<*xf32> to memref<64x64xf32, strided<[512, 1], offset: ?>>
            bufferization.materialize_in_destination %15 in writable %reinterpret_cast_0 : (tensor<64x64xf32>, memref<64x64xf32, strided<[512, 1], offset: ?>>) -> ()
          }
        }
      } {loom.mapped_to = "y"}
    } {loom.mapped_to = "x"}
    return
  }
  func.func @matmul_kernel__d0i0_d1i0__f10__hoist_block_1__c0bx_c1mem(%arg0: memref<*xf32> {tt.divisibility = 16 : i32}, %arg1: memref<*xf32> {tt.divisibility = 16 : i32}, %arg2: memref<*xf32> {tt.divisibility = 16 : i32}, %arg3: index, %arg4: index, %arg5: index) {
    affine.parallel (%arg6) = (0) to (32) {
      affine.parallel (%arg7) = (0) to (2) {
        affine.for %arg8 = 0 to affine_map<()[s0, s1] -> (s1)>()[%arg3, %arg4] {
          %12 = affine.apply affine_map<(d0) -> (d0 * 64)>(%arg8)
          %reinterpret_cast = memref.reinterpret_cast %arg1 to offset: [%12], sizes: [8, 64, 64], strides: [32768, 512, 1] {loom.reuse = {spatial = [{depth = 0 : i64, iterator = "%arg6", mapped_to = "x", reuse_type = "total_reuse", volume = 131072 : i64}, {depth = 1 : i64, iterator = "%arg7", mapped_to = "y", reuse_type = "total_reuse", volume = 131072 : i64}], temporal = [{depth = 2 : i64, iterator = "%arg8", reuse_type = "no_reuse", volume = 0 : i64}]}} : memref<*xf32> to memref<8x64x64xf32, strided<[32768, 512, 1], offset: ?>>
          %alloc = memref.alloc() {loom.alloc = {local = true, memory_name = "L1", size = 131072 : i64}} : memref<8x64x64xf32>
          memref.copy %reinterpret_cast, %alloc {loom.copy.choice = {dim = "x", interconnect_name = "big_ring", kind = "broadcast"}} : memref<8x64x64xf32, strided<[32768, 512, 1], offset: ?>> to memref<8x64x64xf32>
          affine.for %arg9 = 0 to affine_map<()[s0, s1] -> ((s0 ceildiv 32) ceildiv 2)>()[%arg3, %arg4] {
            %cst = arith.constant 0.000000e+00 : f32
            %13 = tensor.empty() : tensor<64x64xf32>
            %14 = linalg.fill ins(%cst : f32) outs(%13 : tensor<64x64xf32>) -> tensor<64x64xf32>
            %c0 = arith.constant 0 : index
            %c8 = arith.constant 8 : index
            %c1 = arith.constant 1 : index
            %15 = scf.for %arg10 = %c0 to %c8 step %c1 iter_args(%arg11 = %14) -> (tensor<64x64xf32>) {
              %17 = affine.apply affine_map<(d0, d1, d2, d3) -> (d0 * 64 + d1 * 1048576 + d2 * 65536 + d3 * 32768)>(%arg10, %arg6, %arg7, %arg9)
              %reinterpret_cast_1 = memref.reinterpret_cast %arg0 to offset: [%17], sizes: [64, 64], strides: [512, 1] {loom.reuse = {sequential = [{depth = 4 : i64, iterator = "%arg10", reuse_type = "no_reuse", volume = 0 : i64}], spatial = [{depth = 0 : i64, iterator = "%arg6", mapped_to = "x", reuse_type = "no_reuse", volume = 0 : i64}, {depth = 1 : i64, iterator = "%arg7", mapped_to = "y", reuse_type = "no_reuse", volume = 0 : i64}], temporal = [{depth = 2 : i64, iterator = "%arg8", reuse_type = "total_reuse", volume = 16384 : i64}, {depth = 3 : i64, iterator = "%arg9", reuse_type = "no_reuse", volume = 0 : i64}]}} : memref<*xf32> to memref<64x64xf32, strided<[512, 1], offset: ?>>
              %alloc_2 = memref.alloc() {loom.alloc = {local = true, memory_name = "L1", size = 16384 : i64}} : memref<64x64xf32>
              memref.copy %reinterpret_cast_1, %alloc_2 {loom.copy.choice = {kind = "mem", memory_name = "L1"}} : memref<64x64xf32, strided<[512, 1], offset: ?>> to memref<64x64xf32>
              %18 = bufferization.to_tensor %alloc_2 restrict writable : memref<64x64xf32> to tensor<64x64xf32>
              %subview = memref.subview %alloc[%arg10, 0, 0] [1, 64, 64] [1, 1, 1] : memref<8x64x64xf32> to memref<64x64xf32, strided<[64, 1], offset: ?>>
              %19 = bufferization.to_tensor %subview restrict writable : memref<64x64xf32, strided<[64, 1], offset: ?>> to tensor<64x64xf32>
              %20 = linalg.matmul ins(%18, %19 : tensor<64x64xf32>, tensor<64x64xf32>) outs(%14 : tensor<64x64xf32>) -> tensor<64x64xf32>
              %21 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%arg11, %20 : tensor<64x64xf32>, tensor<64x64xf32>) outs(%arg11 : tensor<64x64xf32>) {
              ^bb0(%in: f32, %in_3: f32, %out: f32):
                %22 = arith.addf %in, %in_3 : f32
                linalg.yield %22 : f32
              } -> tensor<64x64xf32>
              scf.yield %21 : tensor<64x64xf32>
            }
            %16 = affine.apply affine_map<(d0, d1, d2, d3) -> (d0 * 64 + d1 * 1048576 + d2 * 65536 + d3 * 32768)>(%arg8, %arg6, %arg7, %arg9)
            %reinterpret_cast_0 = memref.reinterpret_cast %arg2 to offset: [%16], sizes: [64, 64], strides: [512, 1] {loom.reuse = {spatial = [{depth = 0 : i64, iterator = "%arg6", mapped_to = "x", reuse_type = "no_reuse", volume = 0 : i64}, {depth = 1 : i64, iterator = "%arg7", mapped_to = "y", reuse_type = "no_reuse", volume = 0 : i64}], temporal = [{depth = 2 : i64, iterator = "%arg8", reuse_type = "no_reuse", volume = 0 : i64}, {depth = 3 : i64, iterator = "%arg9", reuse_type = "no_reuse", volume = 0 : i64}]}} : memref<*xf32> to memref<64x64xf32, strided<[512, 1], offset: ?>>
            bufferization.materialize_in_destination %15 in writable %reinterpret_cast_0 : (tensor<64x64xf32>, memref<64x64xf32, strided<[512, 1], offset: ?>>) -> ()
          }
        }
      } {loom.mapped_to = "y"}
    } {loom.mapped_to = "x"}
    return
  }
  func.func @matmul_kernel__d1i0_d0i0__f01__c0mem_c1mem(%arg0: memref<*xf32> {tt.divisibility = 16 : i32}, %arg1: memref<*xf32> {tt.divisibility = 16 : i32}, %arg2: memref<*xf32> {tt.divisibility = 16 : i32}, %arg3: index, %arg4: index, %arg5: index) {
    affine.parallel (%arg6) = (0) to (2) {
      affine.parallel (%arg7) = (0) to (32) {
        affine.for %arg8 = 0 to affine_map<()[s0, s1] -> ((s0 ceildiv 2) ceildiv 32)>()[%arg3, %arg4] {
          affine.for %arg9 = 0 to affine_map<()[s0, s1] -> (s1)>()[%arg3, %arg4] {
            %cst = arith.constant 0.000000e+00 : f32
            %12 = tensor.empty() : tensor<64x64xf32>
            %13 = linalg.fill ins(%cst : f32) outs(%12 : tensor<64x64xf32>) -> tensor<64x64xf32>
            %c0 = arith.constant 0 : index
            %c8 = arith.constant 8 : index
            %c1 = arith.constant 1 : index
            %14 = scf.for %arg10 = %c0 to %c8 step %c1 iter_args(%arg11 = %13) -> (tensor<64x64xf32>) {
              %16 = affine.apply affine_map<(d0, d1, d2, d3) -> (d0 * 64 + d1 * 65536 + d2 * 1048576 + d3 * 32768)>(%arg10, %arg6, %arg7, %arg8)
              %reinterpret_cast_0 = memref.reinterpret_cast %arg0 to offset: [%16], sizes: [64, 64], strides: [512, 1] {loom.reuse = {sequential = [{depth = 4 : i64, iterator = "%arg10", reuse_type = "no_reuse", volume = 0 : i64}], spatial = [{depth = 0 : i64, iterator = "%arg6", mapped_to = "y", reuse_type = "no_reuse", volume = 0 : i64}, {depth = 1 : i64, iterator = "%arg7", mapped_to = "x", reuse_type = "no_reuse", volume = 0 : i64}], temporal = [{depth = 2 : i64, iterator = "%arg8", reuse_type = "no_reuse", volume = 0 : i64}, {depth = 3 : i64, iterator = "%arg9", reuse_type = "total_reuse", volume = 16384 : i64}]}} : memref<*xf32> to memref<64x64xf32, strided<[512, 1], offset: ?>>
              %alloc = memref.alloc() {loom.alloc = {local = true, memory_name = "L1", size = 16384 : i64}} : memref<64x64xf32>
              memref.copy %reinterpret_cast_0, %alloc {loom.copy.choice = {kind = "mem", memory_name = "L1"}} : memref<64x64xf32, strided<[512, 1], offset: ?>> to memref<64x64xf32>
              %17 = bufferization.to_tensor %alloc restrict writable : memref<64x64xf32> to tensor<64x64xf32>
              %18 = affine.apply affine_map<(d0, d1) -> (d1 * 32768 + d0 * 64)>(%arg9, %arg10)
              %reinterpret_cast_1 = memref.reinterpret_cast %arg1 to offset: [%18], sizes: [64, 64], strides: [512, 1] {loom.reuse = {sequential = [{depth = 4 : i64, iterator = "%arg10", reuse_type = "no_reuse", volume = 0 : i64}], spatial = [{depth = 0 : i64, iterator = "%arg6", mapped_to = "y", reuse_type = "total_reuse", volume = 16384 : i64}, {depth = 1 : i64, iterator = "%arg7", mapped_to = "x", reuse_type = "total_reuse", volume = 16384 : i64}], temporal = [{depth = 2 : i64, iterator = "%arg8", reuse_type = "total_reuse", volume = 16384 : i64}, {depth = 3 : i64, iterator = "%arg9", reuse_type = "no_reuse", volume = 0 : i64}]}} : memref<*xf32> to memref<64x64xf32, strided<[512, 1], offset: ?>>
              %alloc_2 = memref.alloc() {loom.alloc = {local = true, memory_name = "L1", size = 16384 : i64}} : memref<64x64xf32>
              memref.copy %reinterpret_cast_1, %alloc_2 {loom.copy.choice = {kind = "mem", memory_name = "L1"}} : memref<64x64xf32, strided<[512, 1], offset: ?>> to memref<64x64xf32>
              %19 = bufferization.to_tensor %alloc_2 restrict writable : memref<64x64xf32> to tensor<64x64xf32>
              %20 = linalg.matmul ins(%17, %19 : tensor<64x64xf32>, tensor<64x64xf32>) outs(%13 : tensor<64x64xf32>) -> tensor<64x64xf32>
              %21 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%arg11, %20 : tensor<64x64xf32>, tensor<64x64xf32>) outs(%arg11 : tensor<64x64xf32>) {
              ^bb0(%in: f32, %in_3: f32, %out: f32):
                %22 = arith.addf %in, %in_3 : f32
                linalg.yield %22 : f32
              } -> tensor<64x64xf32>
              scf.yield %21 : tensor<64x64xf32>
            }
            %15 = affine.apply affine_map<(d0, d1, d2, d3) -> (d0 * 64 + d1 * 65536 + d2 * 1048576 + d3 * 32768)>(%arg9, %arg6, %arg7, %arg8)
            %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%15], sizes: [64, 64], strides: [512, 1] {loom.reuse = {spatial = [{depth = 0 : i64, iterator = "%arg6", mapped_to = "y", reuse_type = "no_reuse", volume = 0 : i64}, {depth = 1 : i64, iterator = "%arg7", mapped_to = "x", reuse_type = "no_reuse", volume = 0 : i64}], temporal = [{depth = 2 : i64, iterator = "%arg8", reuse_type = "no_reuse", volume = 0 : i64}, {depth = 3 : i64, iterator = "%arg9", reuse_type = "no_reuse", volume = 0 : i64}]}} : memref<*xf32> to memref<64x64xf32, strided<[512, 1], offset: ?>>
            bufferization.materialize_in_destination %14 in writable %reinterpret_cast : (tensor<64x64xf32>, memref<64x64xf32, strided<[512, 1], offset: ?>>) -> ()
          }
        }
      } {loom.mapped_to = "x"}
    } {loom.mapped_to = "y"}
    return
  }
  func.func @matmul_kernel__d1i0_d0i0__f01__c0mem_c1bx(%arg0: memref<*xf32> {tt.divisibility = 16 : i32}, %arg1: memref<*xf32> {tt.divisibility = 16 : i32}, %arg2: memref<*xf32> {tt.divisibility = 16 : i32}, %arg3: index, %arg4: index, %arg5: index) {
    affine.parallel (%arg6) = (0) to (2) {
      affine.parallel (%arg7) = (0) to (32) {
        affine.for %arg8 = 0 to affine_map<()[s0, s1] -> ((s0 ceildiv 2) ceildiv 32)>()[%arg3, %arg4] {
          affine.for %arg9 = 0 to affine_map<()[s0, s1] -> (s1)>()[%arg3, %arg4] {
            %cst = arith.constant 0.000000e+00 : f32
            %12 = tensor.empty() : tensor<64x64xf32>
            %13 = linalg.fill ins(%cst : f32) outs(%12 : tensor<64x64xf32>) -> tensor<64x64xf32>
            %c0 = arith.constant 0 : index
            %c8 = arith.constant 8 : index
            %c1 = arith.constant 1 : index
            %14 = scf.for %arg10 = %c0 to %c8 step %c1 iter_args(%arg11 = %13) -> (tensor<64x64xf32>) {
              %16 = affine.apply affine_map<(d0, d1, d2, d3) -> (d0 * 64 + d1 * 65536 + d2 * 1048576 + d3 * 32768)>(%arg10, %arg6, %arg7, %arg8)
              %reinterpret_cast_0 = memref.reinterpret_cast %arg0 to offset: [%16], sizes: [64, 64], strides: [512, 1] {loom.reuse = {sequential = [{depth = 4 : i64, iterator = "%arg10", reuse_type = "no_reuse", volume = 0 : i64}], spatial = [{depth = 0 : i64, iterator = "%arg6", mapped_to = "y", reuse_type = "no_reuse", volume = 0 : i64}, {depth = 1 : i64, iterator = "%arg7", mapped_to = "x", reuse_type = "no_reuse", volume = 0 : i64}], temporal = [{depth = 2 : i64, iterator = "%arg8", reuse_type = "no_reuse", volume = 0 : i64}, {depth = 3 : i64, iterator = "%arg9", reuse_type = "total_reuse", volume = 16384 : i64}]}} : memref<*xf32> to memref<64x64xf32, strided<[512, 1], offset: ?>>
              %alloc = memref.alloc() {loom.alloc = {local = true, memory_name = "L1", size = 16384 : i64}} : memref<64x64xf32>
              memref.copy %reinterpret_cast_0, %alloc {loom.copy.choice = {kind = "mem", memory_name = "L1"}} : memref<64x64xf32, strided<[512, 1], offset: ?>> to memref<64x64xf32>
              %17 = bufferization.to_tensor %alloc restrict writable : memref<64x64xf32> to tensor<64x64xf32>
              %18 = affine.apply affine_map<(d0, d1) -> (d1 * 32768 + d0 * 64)>(%arg9, %arg10)
              %reinterpret_cast_1 = memref.reinterpret_cast %arg1 to offset: [%18], sizes: [64, 64], strides: [512, 1] {loom.reuse = {sequential = [{depth = 4 : i64, iterator = "%arg10", reuse_type = "no_reuse", volume = 0 : i64}], spatial = [{depth = 0 : i64, iterator = "%arg6", mapped_to = "y", reuse_type = "total_reuse", volume = 16384 : i64}, {depth = 1 : i64, iterator = "%arg7", mapped_to = "x", reuse_type = "total_reuse", volume = 16384 : i64}], temporal = [{depth = 2 : i64, iterator = "%arg8", reuse_type = "total_reuse", volume = 16384 : i64}, {depth = 3 : i64, iterator = "%arg9", reuse_type = "no_reuse", volume = 0 : i64}]}} : memref<*xf32> to memref<64x64xf32, strided<[512, 1], offset: ?>>
              %alloc_2 = memref.alloc() {loom.alloc = {local = true, memory_name = "L1", size = 16384 : i64}} : memref<64x64xf32>
              memref.copy %reinterpret_cast_1, %alloc_2 {loom.copy.choice = {dim = "x", interconnect_name = "big_ring", kind = "broadcast"}} : memref<64x64xf32, strided<[512, 1], offset: ?>> to memref<64x64xf32>
              %19 = bufferization.to_tensor %alloc_2 restrict writable : memref<64x64xf32> to tensor<64x64xf32>
              %20 = linalg.matmul ins(%17, %19 : tensor<64x64xf32>, tensor<64x64xf32>) outs(%13 : tensor<64x64xf32>) -> tensor<64x64xf32>
              %21 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%arg11, %20 : tensor<64x64xf32>, tensor<64x64xf32>) outs(%arg11 : tensor<64x64xf32>) {
              ^bb0(%in: f32, %in_3: f32, %out: f32):
                %22 = arith.addf %in, %in_3 : f32
                linalg.yield %22 : f32
              } -> tensor<64x64xf32>
              scf.yield %21 : tensor<64x64xf32>
            }
            %15 = affine.apply affine_map<(d0, d1, d2, d3) -> (d0 * 64 + d1 * 65536 + d2 * 1048576 + d3 * 32768)>(%arg9, %arg6, %arg7, %arg8)
            %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%15], sizes: [64, 64], strides: [512, 1] {loom.reuse = {spatial = [{depth = 0 : i64, iterator = "%arg6", mapped_to = "y", reuse_type = "no_reuse", volume = 0 : i64}, {depth = 1 : i64, iterator = "%arg7", mapped_to = "x", reuse_type = "no_reuse", volume = 0 : i64}], temporal = [{depth = 2 : i64, iterator = "%arg8", reuse_type = "no_reuse", volume = 0 : i64}, {depth = 3 : i64, iterator = "%arg9", reuse_type = "no_reuse", volume = 0 : i64}]}} : memref<*xf32> to memref<64x64xf32, strided<[512, 1], offset: ?>>
            bufferization.materialize_in_destination %14 in writable %reinterpret_cast : (tensor<64x64xf32>, memref<64x64xf32, strided<[512, 1], offset: ?>>) -> ()
          }
        }
      } {loom.mapped_to = "x"}
    } {loom.mapped_to = "y"}
    return
  }
  func.func @matmul_kernel__d1i0_d0i0__f01__hoist_block_0__c0mem_c1mem(%arg0: memref<*xf32> {tt.divisibility = 16 : i32}, %arg1: memref<*xf32> {tt.divisibility = 16 : i32}, %arg2: memref<*xf32> {tt.divisibility = 16 : i32}, %arg3: index, %arg4: index, %arg5: index) {
    affine.parallel (%arg6) = (0) to (2) {
      affine.parallel (%arg7) = (0) to (32) {
        affine.for %arg8 = 0 to affine_map<()[s0, s1] -> ((s0 ceildiv 2) ceildiv 32)>()[%arg3, %arg4] {
          %12 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 65536 + d1 * 1048576 + d2 * 32768)>(%arg6, %arg7, %arg8)
          %reinterpret_cast = memref.reinterpret_cast %arg0 to offset: [%12], sizes: [8, 64, 64], strides: [64, 512, 1] {loom.reuse = {spatial = [{depth = 0 : i64, iterator = "%arg6", mapped_to = "y", reuse_type = "no_reuse", volume = 0 : i64}, {depth = 1 : i64, iterator = "%arg7", mapped_to = "x", reuse_type = "no_reuse", volume = 0 : i64}], temporal = [{depth = 2 : i64, iterator = "%arg8", reuse_type = "no_reuse", volume = 0 : i64}]}} : memref<*xf32> to memref<8x64x64xf32, strided<[64, 512, 1], offset: ?>>
          %alloc = memref.alloc() {loom.alloc = {local = true, memory_name = "L1", size = 131072 : i64}} : memref<8x64x64xf32>
          memref.copy %reinterpret_cast, %alloc {loom.copy.choice = {kind = "mem", memory_name = "L1"}} : memref<8x64x64xf32, strided<[64, 512, 1], offset: ?>> to memref<8x64x64xf32>
          affine.for %arg9 = 0 to affine_map<()[s0, s1] -> (s1)>()[%arg3, %arg4] {
            %cst = arith.constant 0.000000e+00 : f32
            %13 = tensor.empty() : tensor<64x64xf32>
            %14 = linalg.fill ins(%cst : f32) outs(%13 : tensor<64x64xf32>) -> tensor<64x64xf32>
            %c0 = arith.constant 0 : index
            %c8 = arith.constant 8 : index
            %c1 = arith.constant 1 : index
            %15 = scf.for %arg10 = %c0 to %c8 step %c1 iter_args(%arg11 = %14) -> (tensor<64x64xf32>) {
              %subview = memref.subview %alloc[%arg10, 0, 0] [1, 64, 64] [1, 1, 1] : memref<8x64x64xf32> to memref<64x64xf32, strided<[64, 1], offset: ?>>
              %17 = bufferization.to_tensor %subview restrict writable : memref<64x64xf32, strided<[64, 1], offset: ?>> to tensor<64x64xf32>
              %18 = affine.apply affine_map<(d0, d1) -> (d1 * 32768 + d0 * 64)>(%arg9, %arg10)
              %reinterpret_cast_1 = memref.reinterpret_cast %arg1 to offset: [%18], sizes: [64, 64], strides: [512, 1] {loom.reuse = {sequential = [{depth = 4 : i64, iterator = "%arg10", reuse_type = "no_reuse", volume = 0 : i64}], spatial = [{depth = 0 : i64, iterator = "%arg6", mapped_to = "y", reuse_type = "total_reuse", volume = 16384 : i64}, {depth = 1 : i64, iterator = "%arg7", mapped_to = "x", reuse_type = "total_reuse", volume = 16384 : i64}], temporal = [{depth = 2 : i64, iterator = "%arg8", reuse_type = "total_reuse", volume = 16384 : i64}, {depth = 3 : i64, iterator = "%arg9", reuse_type = "no_reuse", volume = 0 : i64}]}} : memref<*xf32> to memref<64x64xf32, strided<[512, 1], offset: ?>>
              %alloc_2 = memref.alloc() {loom.alloc = {local = true, memory_name = "L1", size = 16384 : i64}} : memref<64x64xf32>
              memref.copy %reinterpret_cast_1, %alloc_2 {loom.copy.choice = {kind = "mem", memory_name = "L1"}} : memref<64x64xf32, strided<[512, 1], offset: ?>> to memref<64x64xf32>
              %19 = bufferization.to_tensor %alloc_2 restrict writable : memref<64x64xf32> to tensor<64x64xf32>
              %20 = linalg.matmul ins(%17, %19 : tensor<64x64xf32>, tensor<64x64xf32>) outs(%14 : tensor<64x64xf32>) -> tensor<64x64xf32>
              %21 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%arg11, %20 : tensor<64x64xf32>, tensor<64x64xf32>) outs(%arg11 : tensor<64x64xf32>) {
              ^bb0(%in: f32, %in_3: f32, %out: f32):
                %22 = arith.addf %in, %in_3 : f32
                linalg.yield %22 : f32
              } -> tensor<64x64xf32>
              scf.yield %21 : tensor<64x64xf32>
            }
            %16 = affine.apply affine_map<(d0, d1, d2, d3) -> (d0 * 64 + d1 * 65536 + d2 * 1048576 + d3 * 32768)>(%arg9, %arg6, %arg7, %arg8)
            %reinterpret_cast_0 = memref.reinterpret_cast %arg2 to offset: [%16], sizes: [64, 64], strides: [512, 1] {loom.reuse = {spatial = [{depth = 0 : i64, iterator = "%arg6", mapped_to = "y", reuse_type = "no_reuse", volume = 0 : i64}, {depth = 1 : i64, iterator = "%arg7", mapped_to = "x", reuse_type = "no_reuse", volume = 0 : i64}], temporal = [{depth = 2 : i64, iterator = "%arg8", reuse_type = "no_reuse", volume = 0 : i64}, {depth = 3 : i64, iterator = "%arg9", reuse_type = "no_reuse", volume = 0 : i64}]}} : memref<*xf32> to memref<64x64xf32, strided<[512, 1], offset: ?>>
            bufferization.materialize_in_destination %15 in writable %reinterpret_cast_0 : (tensor<64x64xf32>, memref<64x64xf32, strided<[512, 1], offset: ?>>) -> ()
          }
        }
      } {loom.mapped_to = "x"}
    } {loom.mapped_to = "y"}
    return
  }
  func.func @matmul_kernel__d1i0_d0i0__f01__hoist_block_0__c0mem_c1bx(%arg0: memref<*xf32> {tt.divisibility = 16 : i32}, %arg1: memref<*xf32> {tt.divisibility = 16 : i32}, %arg2: memref<*xf32> {tt.divisibility = 16 : i32}, %arg3: index, %arg4: index, %arg5: index) {
    affine.parallel (%arg6) = (0) to (2) {
      affine.parallel (%arg7) = (0) to (32) {
        affine.for %arg8 = 0 to affine_map<()[s0, s1] -> ((s0 ceildiv 2) ceildiv 32)>()[%arg3, %arg4] {
          %12 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 65536 + d1 * 1048576 + d2 * 32768)>(%arg6, %arg7, %arg8)
          %reinterpret_cast = memref.reinterpret_cast %arg0 to offset: [%12], sizes: [8, 64, 64], strides: [64, 512, 1] {loom.reuse = {spatial = [{depth = 0 : i64, iterator = "%arg6", mapped_to = "y", reuse_type = "no_reuse", volume = 0 : i64}, {depth = 1 : i64, iterator = "%arg7", mapped_to = "x", reuse_type = "no_reuse", volume = 0 : i64}], temporal = [{depth = 2 : i64, iterator = "%arg8", reuse_type = "no_reuse", volume = 0 : i64}]}} : memref<*xf32> to memref<8x64x64xf32, strided<[64, 512, 1], offset: ?>>
          %alloc = memref.alloc() {loom.alloc = {local = true, memory_name = "L1", size = 131072 : i64}} : memref<8x64x64xf32>
          memref.copy %reinterpret_cast, %alloc {loom.copy.choice = {kind = "mem", memory_name = "L1"}} : memref<8x64x64xf32, strided<[64, 512, 1], offset: ?>> to memref<8x64x64xf32>
          affine.for %arg9 = 0 to affine_map<()[s0, s1] -> (s1)>()[%arg3, %arg4] {
            %cst = arith.constant 0.000000e+00 : f32
            %13 = tensor.empty() : tensor<64x64xf32>
            %14 = linalg.fill ins(%cst : f32) outs(%13 : tensor<64x64xf32>) -> tensor<64x64xf32>
            %c0 = arith.constant 0 : index
            %c8 = arith.constant 8 : index
            %c1 = arith.constant 1 : index
            %15 = scf.for %arg10 = %c0 to %c8 step %c1 iter_args(%arg11 = %14) -> (tensor<64x64xf32>) {
              %subview = memref.subview %alloc[%arg10, 0, 0] [1, 64, 64] [1, 1, 1] : memref<8x64x64xf32> to memref<64x64xf32, strided<[64, 1], offset: ?>>
              %17 = bufferization.to_tensor %subview restrict writable : memref<64x64xf32, strided<[64, 1], offset: ?>> to tensor<64x64xf32>
              %18 = affine.apply affine_map<(d0, d1) -> (d1 * 32768 + d0 * 64)>(%arg9, %arg10)
              %reinterpret_cast_1 = memref.reinterpret_cast %arg1 to offset: [%18], sizes: [64, 64], strides: [512, 1] {loom.reuse = {sequential = [{depth = 4 : i64, iterator = "%arg10", reuse_type = "no_reuse", volume = 0 : i64}], spatial = [{depth = 0 : i64, iterator = "%arg6", mapped_to = "y", reuse_type = "total_reuse", volume = 16384 : i64}, {depth = 1 : i64, iterator = "%arg7", mapped_to = "x", reuse_type = "total_reuse", volume = 16384 : i64}], temporal = [{depth = 2 : i64, iterator = "%arg8", reuse_type = "total_reuse", volume = 16384 : i64}, {depth = 3 : i64, iterator = "%arg9", reuse_type = "no_reuse", volume = 0 : i64}]}} : memref<*xf32> to memref<64x64xf32, strided<[512, 1], offset: ?>>
              %alloc_2 = memref.alloc() {loom.alloc = {local = true, memory_name = "L1", size = 16384 : i64}} : memref<64x64xf32>
              memref.copy %reinterpret_cast_1, %alloc_2 {loom.copy.choice = {dim = "x", interconnect_name = "big_ring", kind = "broadcast"}} : memref<64x64xf32, strided<[512, 1], offset: ?>> to memref<64x64xf32>
              %19 = bufferization.to_tensor %alloc_2 restrict writable : memref<64x64xf32> to tensor<64x64xf32>
              %20 = linalg.matmul ins(%17, %19 : tensor<64x64xf32>, tensor<64x64xf32>) outs(%14 : tensor<64x64xf32>) -> tensor<64x64xf32>
              %21 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%arg11, %20 : tensor<64x64xf32>, tensor<64x64xf32>) outs(%arg11 : tensor<64x64xf32>) {
              ^bb0(%in: f32, %in_3: f32, %out: f32):
                %22 = arith.addf %in, %in_3 : f32
                linalg.yield %22 : f32
              } -> tensor<64x64xf32>
              scf.yield %21 : tensor<64x64xf32>
            }
            %16 = affine.apply affine_map<(d0, d1, d2, d3) -> (d0 * 64 + d1 * 65536 + d2 * 1048576 + d3 * 32768)>(%arg9, %arg6, %arg7, %arg8)
            %reinterpret_cast_0 = memref.reinterpret_cast %arg2 to offset: [%16], sizes: [64, 64], strides: [512, 1] {loom.reuse = {spatial = [{depth = 0 : i64, iterator = "%arg6", mapped_to = "y", reuse_type = "no_reuse", volume = 0 : i64}, {depth = 1 : i64, iterator = "%arg7", mapped_to = "x", reuse_type = "no_reuse", volume = 0 : i64}], temporal = [{depth = 2 : i64, iterator = "%arg8", reuse_type = "no_reuse", volume = 0 : i64}, {depth = 3 : i64, iterator = "%arg9", reuse_type = "no_reuse", volume = 0 : i64}]}} : memref<*xf32> to memref<64x64xf32, strided<[512, 1], offset: ?>>
            bufferization.materialize_in_destination %15 in writable %reinterpret_cast_0 : (tensor<64x64xf32>, memref<64x64xf32, strided<[512, 1], offset: ?>>) -> ()
          }
        }
      } {loom.mapped_to = "x"}
    } {loom.mapped_to = "y"}
    return
  }
  func.func @matmul_kernel__d1i0_d0i0__f10__c0mem_c1mem(%arg0: memref<*xf32> {tt.divisibility = 16 : i32}, %arg1: memref<*xf32> {tt.divisibility = 16 : i32}, %arg2: memref<*xf32> {tt.divisibility = 16 : i32}, %arg3: index, %arg4: index, %arg5: index) {
    affine.parallel (%arg6) = (0) to (2) {
      affine.parallel (%arg7) = (0) to (32) {
        affine.for %arg8 = 0 to affine_map<()[s0, s1] -> (s1)>()[%arg3, %arg4] {
          affine.for %arg9 = 0 to affine_map<()[s0, s1] -> ((s0 ceildiv 2) ceildiv 32)>()[%arg3, %arg4] {
            %cst = arith.constant 0.000000e+00 : f32
            %12 = tensor.empty() : tensor<64x64xf32>
            %13 = linalg.fill ins(%cst : f32) outs(%12 : tensor<64x64xf32>) -> tensor<64x64xf32>
            %c0 = arith.constant 0 : index
            %c8 = arith.constant 8 : index
            %c1 = arith.constant 1 : index
            %14 = scf.for %arg10 = %c0 to %c8 step %c1 iter_args(%arg11 = %13) -> (tensor<64x64xf32>) {
              %16 = affine.apply affine_map<(d0, d1, d2, d3) -> (d0 * 64 + d1 * 65536 + d2 * 1048576 + d3 * 32768)>(%arg10, %arg6, %arg7, %arg9)
              %reinterpret_cast_0 = memref.reinterpret_cast %arg0 to offset: [%16], sizes: [64, 64], strides: [512, 1] {loom.reuse = {sequential = [{depth = 4 : i64, iterator = "%arg10", reuse_type = "no_reuse", volume = 0 : i64}], spatial = [{depth = 0 : i64, iterator = "%arg6", mapped_to = "y", reuse_type = "no_reuse", volume = 0 : i64}, {depth = 1 : i64, iterator = "%arg7", mapped_to = "x", reuse_type = "no_reuse", volume = 0 : i64}], temporal = [{depth = 2 : i64, iterator = "%arg8", reuse_type = "total_reuse", volume = 16384 : i64}, {depth = 3 : i64, iterator = "%arg9", reuse_type = "no_reuse", volume = 0 : i64}]}} : memref<*xf32> to memref<64x64xf32, strided<[512, 1], offset: ?>>
              %alloc = memref.alloc() {loom.alloc = {local = true, memory_name = "L1", size = 16384 : i64}} : memref<64x64xf32>
              memref.copy %reinterpret_cast_0, %alloc {loom.copy.choice = {kind = "mem", memory_name = "L1"}} : memref<64x64xf32, strided<[512, 1], offset: ?>> to memref<64x64xf32>
              %17 = bufferization.to_tensor %alloc restrict writable : memref<64x64xf32> to tensor<64x64xf32>
              %18 = affine.apply affine_map<(d0, d1) -> (d1 * 32768 + d0 * 64)>(%arg8, %arg10)
              %reinterpret_cast_1 = memref.reinterpret_cast %arg1 to offset: [%18], sizes: [64, 64], strides: [512, 1] {loom.reuse = {sequential = [{depth = 4 : i64, iterator = "%arg10", reuse_type = "no_reuse", volume = 0 : i64}], spatial = [{depth = 0 : i64, iterator = "%arg6", mapped_to = "y", reuse_type = "total_reuse", volume = 16384 : i64}, {depth = 1 : i64, iterator = "%arg7", mapped_to = "x", reuse_type = "total_reuse", volume = 16384 : i64}], temporal = [{depth = 2 : i64, iterator = "%arg8", reuse_type = "no_reuse", volume = 0 : i64}, {depth = 3 : i64, iterator = "%arg9", reuse_type = "total_reuse", volume = 16384 : i64}]}} : memref<*xf32> to memref<64x64xf32, strided<[512, 1], offset: ?>>
              %alloc_2 = memref.alloc() {loom.alloc = {local = true, memory_name = "L1", size = 16384 : i64}} : memref<64x64xf32>
              memref.copy %reinterpret_cast_1, %alloc_2 {loom.copy.choice = {kind = "mem", memory_name = "L1"}} : memref<64x64xf32, strided<[512, 1], offset: ?>> to memref<64x64xf32>
              %19 = bufferization.to_tensor %alloc_2 restrict writable : memref<64x64xf32> to tensor<64x64xf32>
              %20 = linalg.matmul ins(%17, %19 : tensor<64x64xf32>, tensor<64x64xf32>) outs(%13 : tensor<64x64xf32>) -> tensor<64x64xf32>
              %21 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%arg11, %20 : tensor<64x64xf32>, tensor<64x64xf32>) outs(%arg11 : tensor<64x64xf32>) {
              ^bb0(%in: f32, %in_3: f32, %out: f32):
                %22 = arith.addf %in, %in_3 : f32
                linalg.yield %22 : f32
              } -> tensor<64x64xf32>
              scf.yield %21 : tensor<64x64xf32>
            }
            %15 = affine.apply affine_map<(d0, d1, d2, d3) -> (d0 * 64 + d1 * 65536 + d2 * 1048576 + d3 * 32768)>(%arg8, %arg6, %arg7, %arg9)
            %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%15], sizes: [64, 64], strides: [512, 1] {loom.reuse = {spatial = [{depth = 0 : i64, iterator = "%arg6", mapped_to = "y", reuse_type = "no_reuse", volume = 0 : i64}, {depth = 1 : i64, iterator = "%arg7", mapped_to = "x", reuse_type = "no_reuse", volume = 0 : i64}], temporal = [{depth = 2 : i64, iterator = "%arg8", reuse_type = "no_reuse", volume = 0 : i64}, {depth = 3 : i64, iterator = "%arg9", reuse_type = "no_reuse", volume = 0 : i64}]}} : memref<*xf32> to memref<64x64xf32, strided<[512, 1], offset: ?>>
            bufferization.materialize_in_destination %14 in writable %reinterpret_cast : (tensor<64x64xf32>, memref<64x64xf32, strided<[512, 1], offset: ?>>) -> ()
          }
        }
      } {loom.mapped_to = "x"}
    } {loom.mapped_to = "y"}
    return
  }
  func.func @matmul_kernel__d1i0_d0i0__f10__c0mem_c1bx(%arg0: memref<*xf32> {tt.divisibility = 16 : i32}, %arg1: memref<*xf32> {tt.divisibility = 16 : i32}, %arg2: memref<*xf32> {tt.divisibility = 16 : i32}, %arg3: index, %arg4: index, %arg5: index) {
    affine.parallel (%arg6) = (0) to (2) {
      affine.parallel (%arg7) = (0) to (32) {
        affine.for %arg8 = 0 to affine_map<()[s0, s1] -> (s1)>()[%arg3, %arg4] {
          affine.for %arg9 = 0 to affine_map<()[s0, s1] -> ((s0 ceildiv 2) ceildiv 32)>()[%arg3, %arg4] {
            %cst = arith.constant 0.000000e+00 : f32
            %12 = tensor.empty() : tensor<64x64xf32>
            %13 = linalg.fill ins(%cst : f32) outs(%12 : tensor<64x64xf32>) -> tensor<64x64xf32>
            %c0 = arith.constant 0 : index
            %c8 = arith.constant 8 : index
            %c1 = arith.constant 1 : index
            %14 = scf.for %arg10 = %c0 to %c8 step %c1 iter_args(%arg11 = %13) -> (tensor<64x64xf32>) {
              %16 = affine.apply affine_map<(d0, d1, d2, d3) -> (d0 * 64 + d1 * 65536 + d2 * 1048576 + d3 * 32768)>(%arg10, %arg6, %arg7, %arg9)
              %reinterpret_cast_0 = memref.reinterpret_cast %arg0 to offset: [%16], sizes: [64, 64], strides: [512, 1] {loom.reuse = {sequential = [{depth = 4 : i64, iterator = "%arg10", reuse_type = "no_reuse", volume = 0 : i64}], spatial = [{depth = 0 : i64, iterator = "%arg6", mapped_to = "y", reuse_type = "no_reuse", volume = 0 : i64}, {depth = 1 : i64, iterator = "%arg7", mapped_to = "x", reuse_type = "no_reuse", volume = 0 : i64}], temporal = [{depth = 2 : i64, iterator = "%arg8", reuse_type = "total_reuse", volume = 16384 : i64}, {depth = 3 : i64, iterator = "%arg9", reuse_type = "no_reuse", volume = 0 : i64}]}} : memref<*xf32> to memref<64x64xf32, strided<[512, 1], offset: ?>>
              %alloc = memref.alloc() {loom.alloc = {local = true, memory_name = "L1", size = 16384 : i64}} : memref<64x64xf32>
              memref.copy %reinterpret_cast_0, %alloc {loom.copy.choice = {kind = "mem", memory_name = "L1"}} : memref<64x64xf32, strided<[512, 1], offset: ?>> to memref<64x64xf32>
              %17 = bufferization.to_tensor %alloc restrict writable : memref<64x64xf32> to tensor<64x64xf32>
              %18 = affine.apply affine_map<(d0, d1) -> (d1 * 32768 + d0 * 64)>(%arg8, %arg10)
              %reinterpret_cast_1 = memref.reinterpret_cast %arg1 to offset: [%18], sizes: [64, 64], strides: [512, 1] {loom.reuse = {sequential = [{depth = 4 : i64, iterator = "%arg10", reuse_type = "no_reuse", volume = 0 : i64}], spatial = [{depth = 0 : i64, iterator = "%arg6", mapped_to = "y", reuse_type = "total_reuse", volume = 16384 : i64}, {depth = 1 : i64, iterator = "%arg7", mapped_to = "x", reuse_type = "total_reuse", volume = 16384 : i64}], temporal = [{depth = 2 : i64, iterator = "%arg8", reuse_type = "no_reuse", volume = 0 : i64}, {depth = 3 : i64, iterator = "%arg9", reuse_type = "total_reuse", volume = 16384 : i64}]}} : memref<*xf32> to memref<64x64xf32, strided<[512, 1], offset: ?>>
              %alloc_2 = memref.alloc() {loom.alloc = {local = true, memory_name = "L1", size = 16384 : i64}} : memref<64x64xf32>
              memref.copy %reinterpret_cast_1, %alloc_2 {loom.copy.choice = {dim = "x", interconnect_name = "big_ring", kind = "broadcast"}} : memref<64x64xf32, strided<[512, 1], offset: ?>> to memref<64x64xf32>
              %19 = bufferization.to_tensor %alloc_2 restrict writable : memref<64x64xf32> to tensor<64x64xf32>
              %20 = linalg.matmul ins(%17, %19 : tensor<64x64xf32>, tensor<64x64xf32>) outs(%13 : tensor<64x64xf32>) -> tensor<64x64xf32>
              %21 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%arg11, %20 : tensor<64x64xf32>, tensor<64x64xf32>) outs(%arg11 : tensor<64x64xf32>) {
              ^bb0(%in: f32, %in_3: f32, %out: f32):
                %22 = arith.addf %in, %in_3 : f32
                linalg.yield %22 : f32
              } -> tensor<64x64xf32>
              scf.yield %21 : tensor<64x64xf32>
            }
            %15 = affine.apply affine_map<(d0, d1, d2, d3) -> (d0 * 64 + d1 * 65536 + d2 * 1048576 + d3 * 32768)>(%arg8, %arg6, %arg7, %arg9)
            %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%15], sizes: [64, 64], strides: [512, 1] {loom.reuse = {spatial = [{depth = 0 : i64, iterator = "%arg6", mapped_to = "y", reuse_type = "no_reuse", volume = 0 : i64}, {depth = 1 : i64, iterator = "%arg7", mapped_to = "x", reuse_type = "no_reuse", volume = 0 : i64}], temporal = [{depth = 2 : i64, iterator = "%arg8", reuse_type = "no_reuse", volume = 0 : i64}, {depth = 3 : i64, iterator = "%arg9", reuse_type = "no_reuse", volume = 0 : i64}]}} : memref<*xf32> to memref<64x64xf32, strided<[512, 1], offset: ?>>
            bufferization.materialize_in_destination %14 in writable %reinterpret_cast : (tensor<64x64xf32>, memref<64x64xf32, strided<[512, 1], offset: ?>>) -> ()
          }
        }
      } {loom.mapped_to = "x"}
    } {loom.mapped_to = "y"}
    return
  }
  func.func @matmul_kernel__d1i0_d0i0__f10__hoist_block_1__c0mem_c1mem(%arg0: memref<*xf32> {tt.divisibility = 16 : i32}, %arg1: memref<*xf32> {tt.divisibility = 16 : i32}, %arg2: memref<*xf32> {tt.divisibility = 16 : i32}, %arg3: index, %arg4: index, %arg5: index) {
    affine.parallel (%arg6) = (0) to (2) {
      affine.parallel (%arg7) = (0) to (32) {
        affine.for %arg8 = 0 to affine_map<()[s0, s1] -> (s1)>()[%arg3, %arg4] {
          %12 = affine.apply affine_map<(d0) -> (d0 * 64)>(%arg8)
          %reinterpret_cast = memref.reinterpret_cast %arg1 to offset: [%12], sizes: [8, 64, 64], strides: [32768, 512, 1] {loom.reuse = {spatial = [{depth = 0 : i64, iterator = "%arg6", mapped_to = "y", reuse_type = "total_reuse", volume = 131072 : i64}, {depth = 1 : i64, iterator = "%arg7", mapped_to = "x", reuse_type = "total_reuse", volume = 131072 : i64}], temporal = [{depth = 2 : i64, iterator = "%arg8", reuse_type = "no_reuse", volume = 0 : i64}]}} : memref<*xf32> to memref<8x64x64xf32, strided<[32768, 512, 1], offset: ?>>
          %alloc = memref.alloc() {loom.alloc = {local = true, memory_name = "L1", size = 131072 : i64}} : memref<8x64x64xf32>
          memref.copy %reinterpret_cast, %alloc {loom.copy.choice = {kind = "mem", memory_name = "L1"}} : memref<8x64x64xf32, strided<[32768, 512, 1], offset: ?>> to memref<8x64x64xf32>
          affine.for %arg9 = 0 to affine_map<()[s0, s1] -> ((s0 ceildiv 2) ceildiv 32)>()[%arg3, %arg4] {
            %cst = arith.constant 0.000000e+00 : f32
            %13 = tensor.empty() : tensor<64x64xf32>
            %14 = linalg.fill ins(%cst : f32) outs(%13 : tensor<64x64xf32>) -> tensor<64x64xf32>
            %c0 = arith.constant 0 : index
            %c8 = arith.constant 8 : index
            %c1 = arith.constant 1 : index
            %15 = scf.for %arg10 = %c0 to %c8 step %c1 iter_args(%arg11 = %14) -> (tensor<64x64xf32>) {
              %17 = affine.apply affine_map<(d0, d1, d2, d3) -> (d0 * 64 + d1 * 65536 + d2 * 1048576 + d3 * 32768)>(%arg10, %arg6, %arg7, %arg9)
              %reinterpret_cast_1 = memref.reinterpret_cast %arg0 to offset: [%17], sizes: [64, 64], strides: [512, 1] {loom.reuse = {sequential = [{depth = 4 : i64, iterator = "%arg10", reuse_type = "no_reuse", volume = 0 : i64}], spatial = [{depth = 0 : i64, iterator = "%arg6", mapped_to = "y", reuse_type = "no_reuse", volume = 0 : i64}, {depth = 1 : i64, iterator = "%arg7", mapped_to = "x", reuse_type = "no_reuse", volume = 0 : i64}], temporal = [{depth = 2 : i64, iterator = "%arg8", reuse_type = "total_reuse", volume = 16384 : i64}, {depth = 3 : i64, iterator = "%arg9", reuse_type = "no_reuse", volume = 0 : i64}]}} : memref<*xf32> to memref<64x64xf32, strided<[512, 1], offset: ?>>
              %alloc_2 = memref.alloc() {loom.alloc = {local = true, memory_name = "L1", size = 16384 : i64}} : memref<64x64xf32>
              memref.copy %reinterpret_cast_1, %alloc_2 {loom.copy.choice = {kind = "mem", memory_name = "L1"}} : memref<64x64xf32, strided<[512, 1], offset: ?>> to memref<64x64xf32>
              %18 = bufferization.to_tensor %alloc_2 restrict writable : memref<64x64xf32> to tensor<64x64xf32>
              %subview = memref.subview %alloc[%arg10, 0, 0] [1, 64, 64] [1, 1, 1] : memref<8x64x64xf32> to memref<64x64xf32, strided<[64, 1], offset: ?>>
              %19 = bufferization.to_tensor %subview restrict writable : memref<64x64xf32, strided<[64, 1], offset: ?>> to tensor<64x64xf32>
              %20 = linalg.matmul ins(%18, %19 : tensor<64x64xf32>, tensor<64x64xf32>) outs(%14 : tensor<64x64xf32>) -> tensor<64x64xf32>
              %21 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%arg11, %20 : tensor<64x64xf32>, tensor<64x64xf32>) outs(%arg11 : tensor<64x64xf32>) {
              ^bb0(%in: f32, %in_3: f32, %out: f32):
                %22 = arith.addf %in, %in_3 : f32
                linalg.yield %22 : f32
              } -> tensor<64x64xf32>
              scf.yield %21 : tensor<64x64xf32>
            }
            %16 = affine.apply affine_map<(d0, d1, d2, d3) -> (d0 * 64 + d1 * 65536 + d2 * 1048576 + d3 * 32768)>(%arg8, %arg6, %arg7, %arg9)
            %reinterpret_cast_0 = memref.reinterpret_cast %arg2 to offset: [%16], sizes: [64, 64], strides: [512, 1] {loom.reuse = {spatial = [{depth = 0 : i64, iterator = "%arg6", mapped_to = "y", reuse_type = "no_reuse", volume = 0 : i64}, {depth = 1 : i64, iterator = "%arg7", mapped_to = "x", reuse_type = "no_reuse", volume = 0 : i64}], temporal = [{depth = 2 : i64, iterator = "%arg8", reuse_type = "no_reuse", volume = 0 : i64}, {depth = 3 : i64, iterator = "%arg9", reuse_type = "no_reuse", volume = 0 : i64}]}} : memref<*xf32> to memref<64x64xf32, strided<[512, 1], offset: ?>>
            bufferization.materialize_in_destination %15 in writable %reinterpret_cast_0 : (tensor<64x64xf32>, memref<64x64xf32, strided<[512, 1], offset: ?>>) -> ()
          }
        }
      } {loom.mapped_to = "x"}
    } {loom.mapped_to = "y"}
    return
  }
  func.func @matmul_kernel__d1i0_d0i0__f10__hoist_block_1__c0bx_c1mem(%arg0: memref<*xf32> {tt.divisibility = 16 : i32}, %arg1: memref<*xf32> {tt.divisibility = 16 : i32}, %arg2: memref<*xf32> {tt.divisibility = 16 : i32}, %arg3: index, %arg4: index, %arg5: index) {
    affine.parallel (%arg6) = (0) to (2) {
      affine.parallel (%arg7) = (0) to (32) {
        affine.for %arg8 = 0 to affine_map<()[s0, s1] -> (s1)>()[%arg3, %arg4] {
          %12 = affine.apply affine_map<(d0) -> (d0 * 64)>(%arg8)
          %reinterpret_cast = memref.reinterpret_cast %arg1 to offset: [%12], sizes: [8, 64, 64], strides: [32768, 512, 1] {loom.reuse = {spatial = [{depth = 0 : i64, iterator = "%arg6", mapped_to = "y", reuse_type = "total_reuse", volume = 131072 : i64}, {depth = 1 : i64, iterator = "%arg7", mapped_to = "x", reuse_type = "total_reuse", volume = 131072 : i64}], temporal = [{depth = 2 : i64, iterator = "%arg8", reuse_type = "no_reuse", volume = 0 : i64}]}} : memref<*xf32> to memref<8x64x64xf32, strided<[32768, 512, 1], offset: ?>>
          %alloc = memref.alloc() {loom.alloc = {local = true, memory_name = "L1", size = 131072 : i64}} : memref<8x64x64xf32>
          memref.copy %reinterpret_cast, %alloc {loom.copy.choice = {dim = "x", interconnect_name = "big_ring", kind = "broadcast"}} : memref<8x64x64xf32, strided<[32768, 512, 1], offset: ?>> to memref<8x64x64xf32>
          affine.for %arg9 = 0 to affine_map<()[s0, s1] -> ((s0 ceildiv 2) ceildiv 32)>()[%arg3, %arg4] {
            %cst = arith.constant 0.000000e+00 : f32
            %13 = tensor.empty() : tensor<64x64xf32>
            %14 = linalg.fill ins(%cst : f32) outs(%13 : tensor<64x64xf32>) -> tensor<64x64xf32>
            %c0 = arith.constant 0 : index
            %c8 = arith.constant 8 : index
            %c1 = arith.constant 1 : index
            %15 = scf.for %arg10 = %c0 to %c8 step %c1 iter_args(%arg11 = %14) -> (tensor<64x64xf32>) {
              %17 = affine.apply affine_map<(d0, d1, d2, d3) -> (d0 * 64 + d1 * 65536 + d2 * 1048576 + d3 * 32768)>(%arg10, %arg6, %arg7, %arg9)
              %reinterpret_cast_1 = memref.reinterpret_cast %arg0 to offset: [%17], sizes: [64, 64], strides: [512, 1] {loom.reuse = {sequential = [{depth = 4 : i64, iterator = "%arg10", reuse_type = "no_reuse", volume = 0 : i64}], spatial = [{depth = 0 : i64, iterator = "%arg6", mapped_to = "y", reuse_type = "no_reuse", volume = 0 : i64}, {depth = 1 : i64, iterator = "%arg7", mapped_to = "x", reuse_type = "no_reuse", volume = 0 : i64}], temporal = [{depth = 2 : i64, iterator = "%arg8", reuse_type = "total_reuse", volume = 16384 : i64}, {depth = 3 : i64, iterator = "%arg9", reuse_type = "no_reuse", volume = 0 : i64}]}} : memref<*xf32> to memref<64x64xf32, strided<[512, 1], offset: ?>>
              %alloc_2 = memref.alloc() {loom.alloc = {local = true, memory_name = "L1", size = 16384 : i64}} : memref<64x64xf32>
              memref.copy %reinterpret_cast_1, %alloc_2 {loom.copy.choice = {kind = "mem", memory_name = "L1"}} : memref<64x64xf32, strided<[512, 1], offset: ?>> to memref<64x64xf32>
              %18 = bufferization.to_tensor %alloc_2 restrict writable : memref<64x64xf32> to tensor<64x64xf32>
              %subview = memref.subview %alloc[%arg10, 0, 0] [1, 64, 64] [1, 1, 1] : memref<8x64x64xf32> to memref<64x64xf32, strided<[64, 1], offset: ?>>
              %19 = bufferization.to_tensor %subview restrict writable : memref<64x64xf32, strided<[64, 1], offset: ?>> to tensor<64x64xf32>
              %20 = linalg.matmul ins(%18, %19 : tensor<64x64xf32>, tensor<64x64xf32>) outs(%14 : tensor<64x64xf32>) -> tensor<64x64xf32>
              %21 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%arg11, %20 : tensor<64x64xf32>, tensor<64x64xf32>) outs(%arg11 : tensor<64x64xf32>) {
              ^bb0(%in: f32, %in_3: f32, %out: f32):
                %22 = arith.addf %in, %in_3 : f32
                linalg.yield %22 : f32
              } -> tensor<64x64xf32>
              scf.yield %21 : tensor<64x64xf32>
            }
            %16 = affine.apply affine_map<(d0, d1, d2, d3) -> (d0 * 64 + d1 * 65536 + d2 * 1048576 + d3 * 32768)>(%arg8, %arg6, %arg7, %arg9)
            %reinterpret_cast_0 = memref.reinterpret_cast %arg2 to offset: [%16], sizes: [64, 64], strides: [512, 1] {loom.reuse = {spatial = [{depth = 0 : i64, iterator = "%arg6", mapped_to = "y", reuse_type = "no_reuse", volume = 0 : i64}, {depth = 1 : i64, iterator = "%arg7", mapped_to = "x", reuse_type = "no_reuse", volume = 0 : i64}], temporal = [{depth = 2 : i64, iterator = "%arg8", reuse_type = "no_reuse", volume = 0 : i64}, {depth = 3 : i64, iterator = "%arg9", reuse_type = "no_reuse", volume = 0 : i64}]}} : memref<*xf32> to memref<64x64xf32, strided<[512, 1], offset: ?>>
            bufferization.materialize_in_destination %15 in writable %reinterpret_cast_0 : (tensor<64x64xf32>, memref<64x64xf32, strided<[512, 1], offset: ?>>) -> ()
          }
        }
      } {loom.mapped_to = "x"}
    } {loom.mapped_to = "y"}
    return
  }
  func.func @matmul_kernel__d0i0_d1i1__f01__c0mem_c1mem(%arg0: memref<*xf32> {tt.divisibility = 16 : i32}, %arg1: memref<*xf32> {tt.divisibility = 16 : i32}, %arg2: memref<*xf32> {tt.divisibility = 16 : i32}, %arg3: index, %arg4: index, %arg5: index) {
    affine.parallel (%arg6) = (0) to (32) {
      affine.parallel (%arg7) = (0) to (2) {
        affine.for %arg8 = 0 to affine_map<()[s0, s1] -> (s0 ceildiv 32)>()[%arg3, %arg4] {
          affine.for %arg9 = 0 to affine_map<()[s0, s1] -> (s1 ceildiv 2)>()[%arg3, %arg4] {
            %cst = arith.constant 0.000000e+00 : f32
            %12 = tensor.empty() : tensor<64x64xf32>
            %13 = linalg.fill ins(%cst : f32) outs(%12 : tensor<64x64xf32>) -> tensor<64x64xf32>
            %c0 = arith.constant 0 : index
            %c8 = arith.constant 8 : index
            %c1 = arith.constant 1 : index
            %14 = scf.for %arg10 = %c0 to %c8 step %c1 iter_args(%arg11 = %13) -> (tensor<64x64xf32>) {
              %16 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 64 + d1 * 1048576 + d2 * 32768)>(%arg10, %arg6, %arg8)
              %reinterpret_cast_0 = memref.reinterpret_cast %arg0 to offset: [%16], sizes: [64, 64], strides: [512, 1] {loom.reuse = {sequential = [{depth = 4 : i64, iterator = "%arg10", reuse_type = "no_reuse", volume = 0 : i64}], spatial = [{depth = 0 : i64, iterator = "%arg6", mapped_to = "x", reuse_type = "no_reuse", volume = 0 : i64}, {depth = 1 : i64, iterator = "%arg7", mapped_to = "y", reuse_type = "total_reuse", volume = 16384 : i64}], temporal = [{depth = 2 : i64, iterator = "%arg8", reuse_type = "no_reuse", volume = 0 : i64}, {depth = 3 : i64, iterator = "%arg9", reuse_type = "total_reuse", volume = 16384 : i64}]}} : memref<*xf32> to memref<64x64xf32, strided<[512, 1], offset: ?>>
              %alloc = memref.alloc() {loom.alloc = {local = true, memory_name = "L1", size = 16384 : i64}} : memref<64x64xf32>
              memref.copy %reinterpret_cast_0, %alloc {loom.copy.choice = {kind = "mem", memory_name = "L1"}} : memref<64x64xf32, strided<[512, 1], offset: ?>> to memref<64x64xf32>
              %17 = bufferization.to_tensor %alloc restrict writable : memref<64x64xf32> to tensor<64x64xf32>
              %18 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 32768 + d1 * 128 + d2 * 64)>(%arg10, %arg7, %arg9)
              %reinterpret_cast_1 = memref.reinterpret_cast %arg1 to offset: [%18], sizes: [64, 64], strides: [512, 1] {loom.reuse = {sequential = [{depth = 4 : i64, iterator = "%arg10", reuse_type = "no_reuse", volume = 0 : i64}], spatial = [{depth = 0 : i64, iterator = "%arg6", mapped_to = "x", reuse_type = "total_reuse", volume = 16384 : i64}, {depth = 1 : i64, iterator = "%arg7", mapped_to = "y", reuse_type = "no_reuse", volume = 0 : i64}], temporal = [{depth = 2 : i64, iterator = "%arg8", reuse_type = "total_reuse", volume = 16384 : i64}, {depth = 3 : i64, iterator = "%arg9", reuse_type = "no_reuse", volume = 0 : i64}]}} : memref<*xf32> to memref<64x64xf32, strided<[512, 1], offset: ?>>
              %alloc_2 = memref.alloc() {loom.alloc = {local = true, memory_name = "L1", size = 16384 : i64}} : memref<64x64xf32>
              memref.copy %reinterpret_cast_1, %alloc_2 {loom.copy.choice = {kind = "mem", memory_name = "L1"}} : memref<64x64xf32, strided<[512, 1], offset: ?>> to memref<64x64xf32>
              %19 = bufferization.to_tensor %alloc_2 restrict writable : memref<64x64xf32> to tensor<64x64xf32>
              %20 = linalg.matmul ins(%17, %19 : tensor<64x64xf32>, tensor<64x64xf32>) outs(%13 : tensor<64x64xf32>) -> tensor<64x64xf32>
              %21 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%arg11, %20 : tensor<64x64xf32>, tensor<64x64xf32>) outs(%arg11 : tensor<64x64xf32>) {
              ^bb0(%in: f32, %in_3: f32, %out: f32):
                %22 = arith.addf %in, %in_3 : f32
                linalg.yield %22 : f32
              } -> tensor<64x64xf32>
              scf.yield %21 : tensor<64x64xf32>
            }
            %15 = affine.apply affine_map<(d0, d1, d2, d3) -> (d0 * 128 + d1 * 64 + d2 * 1048576 + d3 * 32768)>(%arg7, %arg9, %arg6, %arg8)
            %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%15], sizes: [64, 64], strides: [512, 1] {loom.reuse = {spatial = [{depth = 0 : i64, iterator = "%arg6", mapped_to = "x", reuse_type = "no_reuse", volume = 0 : i64}, {depth = 1 : i64, iterator = "%arg7", mapped_to = "y", reuse_type = "no_reuse", volume = 0 : i64}], temporal = [{depth = 2 : i64, iterator = "%arg8", reuse_type = "no_reuse", volume = 0 : i64}, {depth = 3 : i64, iterator = "%arg9", reuse_type = "no_reuse", volume = 0 : i64}]}} : memref<*xf32> to memref<64x64xf32, strided<[512, 1], offset: ?>>
            bufferization.materialize_in_destination %14 in writable %reinterpret_cast : (tensor<64x64xf32>, memref<64x64xf32, strided<[512, 1], offset: ?>>) -> ()
          }
        }
      } {loom.mapped_to = "y"}
    } {loom.mapped_to = "x"}
    return
  }
  func.func @matmul_kernel__d0i0_d1i1__f01__c0mem_c1bx(%arg0: memref<*xf32> {tt.divisibility = 16 : i32}, %arg1: memref<*xf32> {tt.divisibility = 16 : i32}, %arg2: memref<*xf32> {tt.divisibility = 16 : i32}, %arg3: index, %arg4: index, %arg5: index) {
    affine.parallel (%arg6) = (0) to (32) {
      affine.parallel (%arg7) = (0) to (2) {
        affine.for %arg8 = 0 to affine_map<()[s0, s1] -> (s0 ceildiv 32)>()[%arg3, %arg4] {
          affine.for %arg9 = 0 to affine_map<()[s0, s1] -> (s1 ceildiv 2)>()[%arg3, %arg4] {
            %cst = arith.constant 0.000000e+00 : f32
            %12 = tensor.empty() : tensor<64x64xf32>
            %13 = linalg.fill ins(%cst : f32) outs(%12 : tensor<64x64xf32>) -> tensor<64x64xf32>
            %c0 = arith.constant 0 : index
            %c8 = arith.constant 8 : index
            %c1 = arith.constant 1 : index
            %14 = scf.for %arg10 = %c0 to %c8 step %c1 iter_args(%arg11 = %13) -> (tensor<64x64xf32>) {
              %16 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 64 + d1 * 1048576 + d2 * 32768)>(%arg10, %arg6, %arg8)
              %reinterpret_cast_0 = memref.reinterpret_cast %arg0 to offset: [%16], sizes: [64, 64], strides: [512, 1] {loom.reuse = {sequential = [{depth = 4 : i64, iterator = "%arg10", reuse_type = "no_reuse", volume = 0 : i64}], spatial = [{depth = 0 : i64, iterator = "%arg6", mapped_to = "x", reuse_type = "no_reuse", volume = 0 : i64}, {depth = 1 : i64, iterator = "%arg7", mapped_to = "y", reuse_type = "total_reuse", volume = 16384 : i64}], temporal = [{depth = 2 : i64, iterator = "%arg8", reuse_type = "no_reuse", volume = 0 : i64}, {depth = 3 : i64, iterator = "%arg9", reuse_type = "total_reuse", volume = 16384 : i64}]}} : memref<*xf32> to memref<64x64xf32, strided<[512, 1], offset: ?>>
              %alloc = memref.alloc() {loom.alloc = {local = true, memory_name = "L1", size = 16384 : i64}} : memref<64x64xf32>
              memref.copy %reinterpret_cast_0, %alloc {loom.copy.choice = {kind = "mem", memory_name = "L1"}} : memref<64x64xf32, strided<[512, 1], offset: ?>> to memref<64x64xf32>
              %17 = bufferization.to_tensor %alloc restrict writable : memref<64x64xf32> to tensor<64x64xf32>
              %18 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 32768 + d1 * 128 + d2 * 64)>(%arg10, %arg7, %arg9)
              %reinterpret_cast_1 = memref.reinterpret_cast %arg1 to offset: [%18], sizes: [64, 64], strides: [512, 1] {loom.reuse = {sequential = [{depth = 4 : i64, iterator = "%arg10", reuse_type = "no_reuse", volume = 0 : i64}], spatial = [{depth = 0 : i64, iterator = "%arg6", mapped_to = "x", reuse_type = "total_reuse", volume = 16384 : i64}, {depth = 1 : i64, iterator = "%arg7", mapped_to = "y", reuse_type = "no_reuse", volume = 0 : i64}], temporal = [{depth = 2 : i64, iterator = "%arg8", reuse_type = "total_reuse", volume = 16384 : i64}, {depth = 3 : i64, iterator = "%arg9", reuse_type = "no_reuse", volume = 0 : i64}]}} : memref<*xf32> to memref<64x64xf32, strided<[512, 1], offset: ?>>
              %alloc_2 = memref.alloc() {loom.alloc = {local = true, memory_name = "L1", size = 16384 : i64}} : memref<64x64xf32>
              memref.copy %reinterpret_cast_1, %alloc_2 {loom.copy.choice = {dim = "x", interconnect_name = "big_ring", kind = "broadcast"}} : memref<64x64xf32, strided<[512, 1], offset: ?>> to memref<64x64xf32>
              %19 = bufferization.to_tensor %alloc_2 restrict writable : memref<64x64xf32> to tensor<64x64xf32>
              %20 = linalg.matmul ins(%17, %19 : tensor<64x64xf32>, tensor<64x64xf32>) outs(%13 : tensor<64x64xf32>) -> tensor<64x64xf32>
              %21 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%arg11, %20 : tensor<64x64xf32>, tensor<64x64xf32>) outs(%arg11 : tensor<64x64xf32>) {
              ^bb0(%in: f32, %in_3: f32, %out: f32):
                %22 = arith.addf %in, %in_3 : f32
                linalg.yield %22 : f32
              } -> tensor<64x64xf32>
              scf.yield %21 : tensor<64x64xf32>
            }
            %15 = affine.apply affine_map<(d0, d1, d2, d3) -> (d0 * 128 + d1 * 64 + d2 * 1048576 + d3 * 32768)>(%arg7, %arg9, %arg6, %arg8)
            %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%15], sizes: [64, 64], strides: [512, 1] {loom.reuse = {spatial = [{depth = 0 : i64, iterator = "%arg6", mapped_to = "x", reuse_type = "no_reuse", volume = 0 : i64}, {depth = 1 : i64, iterator = "%arg7", mapped_to = "y", reuse_type = "no_reuse", volume = 0 : i64}], temporal = [{depth = 2 : i64, iterator = "%arg8", reuse_type = "no_reuse", volume = 0 : i64}, {depth = 3 : i64, iterator = "%arg9", reuse_type = "no_reuse", volume = 0 : i64}]}} : memref<*xf32> to memref<64x64xf32, strided<[512, 1], offset: ?>>
            bufferization.materialize_in_destination %14 in writable %reinterpret_cast : (tensor<64x64xf32>, memref<64x64xf32, strided<[512, 1], offset: ?>>) -> ()
          }
        }
      } {loom.mapped_to = "y"}
    } {loom.mapped_to = "x"}
    return
  }
  func.func @matmul_kernel__d0i0_d1i1__f01__hoist_block_0__c0mem_c1mem(%arg0: memref<*xf32> {tt.divisibility = 16 : i32}, %arg1: memref<*xf32> {tt.divisibility = 16 : i32}, %arg2: memref<*xf32> {tt.divisibility = 16 : i32}, %arg3: index, %arg4: index, %arg5: index) {
    affine.parallel (%arg6) = (0) to (32) {
      affine.parallel (%arg7) = (0) to (2) {
        affine.for %arg8 = 0 to affine_map<()[s0, s1] -> (s0 ceildiv 32)>()[%arg3, %arg4] {
          %12 = affine.apply affine_map<(d0, d1) -> (d0 * 1048576 + d1 * 32768)>(%arg6, %arg8)
          %reinterpret_cast = memref.reinterpret_cast %arg0 to offset: [%12], sizes: [8, 64, 64], strides: [64, 512, 1] {loom.reuse = {spatial = [{depth = 0 : i64, iterator = "%arg6", mapped_to = "x", reuse_type = "no_reuse", volume = 0 : i64}, {depth = 1 : i64, iterator = "%arg7", mapped_to = "y", reuse_type = "total_reuse", volume = 131072 : i64}], temporal = [{depth = 2 : i64, iterator = "%arg8", reuse_type = "no_reuse", volume = 0 : i64}]}} : memref<*xf32> to memref<8x64x64xf32, strided<[64, 512, 1], offset: ?>>
          %alloc = memref.alloc() {loom.alloc = {local = true, memory_name = "L1", size = 131072 : i64}} : memref<8x64x64xf32>
          memref.copy %reinterpret_cast, %alloc {loom.copy.choice = {kind = "mem", memory_name = "L1"}} : memref<8x64x64xf32, strided<[64, 512, 1], offset: ?>> to memref<8x64x64xf32>
          affine.for %arg9 = 0 to affine_map<()[s0, s1] -> (s1 ceildiv 2)>()[%arg3, %arg4] {
            %cst = arith.constant 0.000000e+00 : f32
            %13 = tensor.empty() : tensor<64x64xf32>
            %14 = linalg.fill ins(%cst : f32) outs(%13 : tensor<64x64xf32>) -> tensor<64x64xf32>
            %c0 = arith.constant 0 : index
            %c8 = arith.constant 8 : index
            %c1 = arith.constant 1 : index
            %15 = scf.for %arg10 = %c0 to %c8 step %c1 iter_args(%arg11 = %14) -> (tensor<64x64xf32>) {
              %subview = memref.subview %alloc[%arg10, 0, 0] [1, 64, 64] [1, 1, 1] : memref<8x64x64xf32> to memref<64x64xf32, strided<[64, 1], offset: ?>>
              %17 = bufferization.to_tensor %subview restrict writable : memref<64x64xf32, strided<[64, 1], offset: ?>> to tensor<64x64xf32>
              %18 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 32768 + d1 * 128 + d2 * 64)>(%arg10, %arg7, %arg9)
              %reinterpret_cast_1 = memref.reinterpret_cast %arg1 to offset: [%18], sizes: [64, 64], strides: [512, 1] {loom.reuse = {sequential = [{depth = 4 : i64, iterator = "%arg10", reuse_type = "no_reuse", volume = 0 : i64}], spatial = [{depth = 0 : i64, iterator = "%arg6", mapped_to = "x", reuse_type = "total_reuse", volume = 16384 : i64}, {depth = 1 : i64, iterator = "%arg7", mapped_to = "y", reuse_type = "no_reuse", volume = 0 : i64}], temporal = [{depth = 2 : i64, iterator = "%arg8", reuse_type = "total_reuse", volume = 16384 : i64}, {depth = 3 : i64, iterator = "%arg9", reuse_type = "no_reuse", volume = 0 : i64}]}} : memref<*xf32> to memref<64x64xf32, strided<[512, 1], offset: ?>>
              %alloc_2 = memref.alloc() {loom.alloc = {local = true, memory_name = "L1", size = 16384 : i64}} : memref<64x64xf32>
              memref.copy %reinterpret_cast_1, %alloc_2 {loom.copy.choice = {kind = "mem", memory_name = "L1"}} : memref<64x64xf32, strided<[512, 1], offset: ?>> to memref<64x64xf32>
              %19 = bufferization.to_tensor %alloc_2 restrict writable : memref<64x64xf32> to tensor<64x64xf32>
              %20 = linalg.matmul ins(%17, %19 : tensor<64x64xf32>, tensor<64x64xf32>) outs(%14 : tensor<64x64xf32>) -> tensor<64x64xf32>
              %21 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%arg11, %20 : tensor<64x64xf32>, tensor<64x64xf32>) outs(%arg11 : tensor<64x64xf32>) {
              ^bb0(%in: f32, %in_3: f32, %out: f32):
                %22 = arith.addf %in, %in_3 : f32
                linalg.yield %22 : f32
              } -> tensor<64x64xf32>
              scf.yield %21 : tensor<64x64xf32>
            }
            %16 = affine.apply affine_map<(d0, d1, d2, d3) -> (d0 * 128 + d1 * 64 + d2 * 1048576 + d3 * 32768)>(%arg7, %arg9, %arg6, %arg8)
            %reinterpret_cast_0 = memref.reinterpret_cast %arg2 to offset: [%16], sizes: [64, 64], strides: [512, 1] {loom.reuse = {spatial = [{depth = 0 : i64, iterator = "%arg6", mapped_to = "x", reuse_type = "no_reuse", volume = 0 : i64}, {depth = 1 : i64, iterator = "%arg7", mapped_to = "y", reuse_type = "no_reuse", volume = 0 : i64}], temporal = [{depth = 2 : i64, iterator = "%arg8", reuse_type = "no_reuse", volume = 0 : i64}, {depth = 3 : i64, iterator = "%arg9", reuse_type = "no_reuse", volume = 0 : i64}]}} : memref<*xf32> to memref<64x64xf32, strided<[512, 1], offset: ?>>
            bufferization.materialize_in_destination %15 in writable %reinterpret_cast_0 : (tensor<64x64xf32>, memref<64x64xf32, strided<[512, 1], offset: ?>>) -> ()
          }
        }
      } {loom.mapped_to = "y"}
    } {loom.mapped_to = "x"}
    return
  }
  func.func @matmul_kernel__d0i0_d1i1__f01__hoist_block_0__c0mem_c1bx(%arg0: memref<*xf32> {tt.divisibility = 16 : i32}, %arg1: memref<*xf32> {tt.divisibility = 16 : i32}, %arg2: memref<*xf32> {tt.divisibility = 16 : i32}, %arg3: index, %arg4: index, %arg5: index) {
    affine.parallel (%arg6) = (0) to (32) {
      affine.parallel (%arg7) = (0) to (2) {
        affine.for %arg8 = 0 to affine_map<()[s0, s1] -> (s0 ceildiv 32)>()[%arg3, %arg4] {
          %12 = affine.apply affine_map<(d0, d1) -> (d0 * 1048576 + d1 * 32768)>(%arg6, %arg8)
          %reinterpret_cast = memref.reinterpret_cast %arg0 to offset: [%12], sizes: [8, 64, 64], strides: [64, 512, 1] {loom.reuse = {spatial = [{depth = 0 : i64, iterator = "%arg6", mapped_to = "x", reuse_type = "no_reuse", volume = 0 : i64}, {depth = 1 : i64, iterator = "%arg7", mapped_to = "y", reuse_type = "total_reuse", volume = 131072 : i64}], temporal = [{depth = 2 : i64, iterator = "%arg8", reuse_type = "no_reuse", volume = 0 : i64}]}} : memref<*xf32> to memref<8x64x64xf32, strided<[64, 512, 1], offset: ?>>
          %alloc = memref.alloc() {loom.alloc = {local = true, memory_name = "L1", size = 131072 : i64}} : memref<8x64x64xf32>
          memref.copy %reinterpret_cast, %alloc {loom.copy.choice = {kind = "mem", memory_name = "L1"}} : memref<8x64x64xf32, strided<[64, 512, 1], offset: ?>> to memref<8x64x64xf32>
          affine.for %arg9 = 0 to affine_map<()[s0, s1] -> (s1 ceildiv 2)>()[%arg3, %arg4] {
            %cst = arith.constant 0.000000e+00 : f32
            %13 = tensor.empty() : tensor<64x64xf32>
            %14 = linalg.fill ins(%cst : f32) outs(%13 : tensor<64x64xf32>) -> tensor<64x64xf32>
            %c0 = arith.constant 0 : index
            %c8 = arith.constant 8 : index
            %c1 = arith.constant 1 : index
            %15 = scf.for %arg10 = %c0 to %c8 step %c1 iter_args(%arg11 = %14) -> (tensor<64x64xf32>) {
              %subview = memref.subview %alloc[%arg10, 0, 0] [1, 64, 64] [1, 1, 1] : memref<8x64x64xf32> to memref<64x64xf32, strided<[64, 1], offset: ?>>
              %17 = bufferization.to_tensor %subview restrict writable : memref<64x64xf32, strided<[64, 1], offset: ?>> to tensor<64x64xf32>
              %18 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 32768 + d1 * 128 + d2 * 64)>(%arg10, %arg7, %arg9)
              %reinterpret_cast_1 = memref.reinterpret_cast %arg1 to offset: [%18], sizes: [64, 64], strides: [512, 1] {loom.reuse = {sequential = [{depth = 4 : i64, iterator = "%arg10", reuse_type = "no_reuse", volume = 0 : i64}], spatial = [{depth = 0 : i64, iterator = "%arg6", mapped_to = "x", reuse_type = "total_reuse", volume = 16384 : i64}, {depth = 1 : i64, iterator = "%arg7", mapped_to = "y", reuse_type = "no_reuse", volume = 0 : i64}], temporal = [{depth = 2 : i64, iterator = "%arg8", reuse_type = "total_reuse", volume = 16384 : i64}, {depth = 3 : i64, iterator = "%arg9", reuse_type = "no_reuse", volume = 0 : i64}]}} : memref<*xf32> to memref<64x64xf32, strided<[512, 1], offset: ?>>
              %alloc_2 = memref.alloc() {loom.alloc = {local = true, memory_name = "L1", size = 16384 : i64}} : memref<64x64xf32>
              memref.copy %reinterpret_cast_1, %alloc_2 {loom.copy.choice = {dim = "x", interconnect_name = "big_ring", kind = "broadcast"}} : memref<64x64xf32, strided<[512, 1], offset: ?>> to memref<64x64xf32>
              %19 = bufferization.to_tensor %alloc_2 restrict writable : memref<64x64xf32> to tensor<64x64xf32>
              %20 = linalg.matmul ins(%17, %19 : tensor<64x64xf32>, tensor<64x64xf32>) outs(%14 : tensor<64x64xf32>) -> tensor<64x64xf32>
              %21 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%arg11, %20 : tensor<64x64xf32>, tensor<64x64xf32>) outs(%arg11 : tensor<64x64xf32>) {
              ^bb0(%in: f32, %in_3: f32, %out: f32):
                %22 = arith.addf %in, %in_3 : f32
                linalg.yield %22 : f32
              } -> tensor<64x64xf32>
              scf.yield %21 : tensor<64x64xf32>
            }
            %16 = affine.apply affine_map<(d0, d1, d2, d3) -> (d0 * 128 + d1 * 64 + d2 * 1048576 + d3 * 32768)>(%arg7, %arg9, %arg6, %arg8)
            %reinterpret_cast_0 = memref.reinterpret_cast %arg2 to offset: [%16], sizes: [64, 64], strides: [512, 1] {loom.reuse = {spatial = [{depth = 0 : i64, iterator = "%arg6", mapped_to = "x", reuse_type = "no_reuse", volume = 0 : i64}, {depth = 1 : i64, iterator = "%arg7", mapped_to = "y", reuse_type = "no_reuse", volume = 0 : i64}], temporal = [{depth = 2 : i64, iterator = "%arg8", reuse_type = "no_reuse", volume = 0 : i64}, {depth = 3 : i64, iterator = "%arg9", reuse_type = "no_reuse", volume = 0 : i64}]}} : memref<*xf32> to memref<64x64xf32, strided<[512, 1], offset: ?>>
            bufferization.materialize_in_destination %15 in writable %reinterpret_cast_0 : (tensor<64x64xf32>, memref<64x64xf32, strided<[512, 1], offset: ?>>) -> ()
          }
        }
      } {loom.mapped_to = "y"}
    } {loom.mapped_to = "x"}
    return
  }
  func.func @matmul_kernel__d0i0_d1i1__f10__c0mem_c1mem(%arg0: memref<*xf32> {tt.divisibility = 16 : i32}, %arg1: memref<*xf32> {tt.divisibility = 16 : i32}, %arg2: memref<*xf32> {tt.divisibility = 16 : i32}, %arg3: index, %arg4: index, %arg5: index) {
    affine.parallel (%arg6) = (0) to (32) {
      affine.parallel (%arg7) = (0) to (2) {
        affine.for %arg8 = 0 to affine_map<()[s0, s1] -> (s1 ceildiv 2)>()[%arg3, %arg4] {
          affine.for %arg9 = 0 to affine_map<()[s0, s1] -> (s0 ceildiv 32)>()[%arg3, %arg4] {
            %cst = arith.constant 0.000000e+00 : f32
            %12 = tensor.empty() : tensor<64x64xf32>
            %13 = linalg.fill ins(%cst : f32) outs(%12 : tensor<64x64xf32>) -> tensor<64x64xf32>
            %c0 = arith.constant 0 : index
            %c8 = arith.constant 8 : index
            %c1 = arith.constant 1 : index
            %14 = scf.for %arg10 = %c0 to %c8 step %c1 iter_args(%arg11 = %13) -> (tensor<64x64xf32>) {
              %16 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 64 + d1 * 1048576 + d2 * 32768)>(%arg10, %arg6, %arg9)
              %reinterpret_cast_0 = memref.reinterpret_cast %arg0 to offset: [%16], sizes: [64, 64], strides: [512, 1] {loom.reuse = {sequential = [{depth = 4 : i64, iterator = "%arg10", reuse_type = "no_reuse", volume = 0 : i64}], spatial = [{depth = 0 : i64, iterator = "%arg6", mapped_to = "x", reuse_type = "no_reuse", volume = 0 : i64}, {depth = 1 : i64, iterator = "%arg7", mapped_to = "y", reuse_type = "total_reuse", volume = 16384 : i64}], temporal = [{depth = 2 : i64, iterator = "%arg8", reuse_type = "total_reuse", volume = 16384 : i64}, {depth = 3 : i64, iterator = "%arg9", reuse_type = "no_reuse", volume = 0 : i64}]}} : memref<*xf32> to memref<64x64xf32, strided<[512, 1], offset: ?>>
              %alloc = memref.alloc() {loom.alloc = {local = true, memory_name = "L1", size = 16384 : i64}} : memref<64x64xf32>
              memref.copy %reinterpret_cast_0, %alloc {loom.copy.choice = {kind = "mem", memory_name = "L1"}} : memref<64x64xf32, strided<[512, 1], offset: ?>> to memref<64x64xf32>
              %17 = bufferization.to_tensor %alloc restrict writable : memref<64x64xf32> to tensor<64x64xf32>
              %18 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 32768 + d1 * 128 + d2 * 64)>(%arg10, %arg7, %arg8)
              %reinterpret_cast_1 = memref.reinterpret_cast %arg1 to offset: [%18], sizes: [64, 64], strides: [512, 1] {loom.reuse = {sequential = [{depth = 4 : i64, iterator = "%arg10", reuse_type = "no_reuse", volume = 0 : i64}], spatial = [{depth = 0 : i64, iterator = "%arg6", mapped_to = "x", reuse_type = "total_reuse", volume = 16384 : i64}, {depth = 1 : i64, iterator = "%arg7", mapped_to = "y", reuse_type = "no_reuse", volume = 0 : i64}], temporal = [{depth = 2 : i64, iterator = "%arg8", reuse_type = "no_reuse", volume = 0 : i64}, {depth = 3 : i64, iterator = "%arg9", reuse_type = "total_reuse", volume = 16384 : i64}]}} : memref<*xf32> to memref<64x64xf32, strided<[512, 1], offset: ?>>
              %alloc_2 = memref.alloc() {loom.alloc = {local = true, memory_name = "L1", size = 16384 : i64}} : memref<64x64xf32>
              memref.copy %reinterpret_cast_1, %alloc_2 {loom.copy.choice = {kind = "mem", memory_name = "L1"}} : memref<64x64xf32, strided<[512, 1], offset: ?>> to memref<64x64xf32>
              %19 = bufferization.to_tensor %alloc_2 restrict writable : memref<64x64xf32> to tensor<64x64xf32>
              %20 = linalg.matmul ins(%17, %19 : tensor<64x64xf32>, tensor<64x64xf32>) outs(%13 : tensor<64x64xf32>) -> tensor<64x64xf32>
              %21 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%arg11, %20 : tensor<64x64xf32>, tensor<64x64xf32>) outs(%arg11 : tensor<64x64xf32>) {
              ^bb0(%in: f32, %in_3: f32, %out: f32):
                %22 = arith.addf %in, %in_3 : f32
                linalg.yield %22 : f32
              } -> tensor<64x64xf32>
              scf.yield %21 : tensor<64x64xf32>
            }
            %15 = affine.apply affine_map<(d0, d1, d2, d3) -> (d0 * 128 + d1 * 64 + d2 * 1048576 + d3 * 32768)>(%arg7, %arg8, %arg6, %arg9)
            %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%15], sizes: [64, 64], strides: [512, 1] {loom.reuse = {spatial = [{depth = 0 : i64, iterator = "%arg6", mapped_to = "x", reuse_type = "no_reuse", volume = 0 : i64}, {depth = 1 : i64, iterator = "%arg7", mapped_to = "y", reuse_type = "no_reuse", volume = 0 : i64}], temporal = [{depth = 2 : i64, iterator = "%arg8", reuse_type = "no_reuse", volume = 0 : i64}, {depth = 3 : i64, iterator = "%arg9", reuse_type = "no_reuse", volume = 0 : i64}]}} : memref<*xf32> to memref<64x64xf32, strided<[512, 1], offset: ?>>
            bufferization.materialize_in_destination %14 in writable %reinterpret_cast : (tensor<64x64xf32>, memref<64x64xf32, strided<[512, 1], offset: ?>>) -> ()
          }
        }
      } {loom.mapped_to = "y"}
    } {loom.mapped_to = "x"}
    return
  }
  func.func @matmul_kernel__d0i0_d1i1__f10__c0mem_c1bx(%arg0: memref<*xf32> {tt.divisibility = 16 : i32}, %arg1: memref<*xf32> {tt.divisibility = 16 : i32}, %arg2: memref<*xf32> {tt.divisibility = 16 : i32}, %arg3: index, %arg4: index, %arg5: index) {
    affine.parallel (%arg6) = (0) to (32) {
      affine.parallel (%arg7) = (0) to (2) {
        affine.for %arg8 = 0 to affine_map<()[s0, s1] -> (s1 ceildiv 2)>()[%arg3, %arg4] {
          affine.for %arg9 = 0 to affine_map<()[s0, s1] -> (s0 ceildiv 32)>()[%arg3, %arg4] {
            %cst = arith.constant 0.000000e+00 : f32
            %12 = tensor.empty() : tensor<64x64xf32>
            %13 = linalg.fill ins(%cst : f32) outs(%12 : tensor<64x64xf32>) -> tensor<64x64xf32>
            %c0 = arith.constant 0 : index
            %c8 = arith.constant 8 : index
            %c1 = arith.constant 1 : index
            %14 = scf.for %arg10 = %c0 to %c8 step %c1 iter_args(%arg11 = %13) -> (tensor<64x64xf32>) {
              %16 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 64 + d1 * 1048576 + d2 * 32768)>(%arg10, %arg6, %arg9)
              %reinterpret_cast_0 = memref.reinterpret_cast %arg0 to offset: [%16], sizes: [64, 64], strides: [512, 1] {loom.reuse = {sequential = [{depth = 4 : i64, iterator = "%arg10", reuse_type = "no_reuse", volume = 0 : i64}], spatial = [{depth = 0 : i64, iterator = "%arg6", mapped_to = "x", reuse_type = "no_reuse", volume = 0 : i64}, {depth = 1 : i64, iterator = "%arg7", mapped_to = "y", reuse_type = "total_reuse", volume = 16384 : i64}], temporal = [{depth = 2 : i64, iterator = "%arg8", reuse_type = "total_reuse", volume = 16384 : i64}, {depth = 3 : i64, iterator = "%arg9", reuse_type = "no_reuse", volume = 0 : i64}]}} : memref<*xf32> to memref<64x64xf32, strided<[512, 1], offset: ?>>
              %alloc = memref.alloc() {loom.alloc = {local = true, memory_name = "L1", size = 16384 : i64}} : memref<64x64xf32>
              memref.copy %reinterpret_cast_0, %alloc {loom.copy.choice = {kind = "mem", memory_name = "L1"}} : memref<64x64xf32, strided<[512, 1], offset: ?>> to memref<64x64xf32>
              %17 = bufferization.to_tensor %alloc restrict writable : memref<64x64xf32> to tensor<64x64xf32>
              %18 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 32768 + d1 * 128 + d2 * 64)>(%arg10, %arg7, %arg8)
              %reinterpret_cast_1 = memref.reinterpret_cast %arg1 to offset: [%18], sizes: [64, 64], strides: [512, 1] {loom.reuse = {sequential = [{depth = 4 : i64, iterator = "%arg10", reuse_type = "no_reuse", volume = 0 : i64}], spatial = [{depth = 0 : i64, iterator = "%arg6", mapped_to = "x", reuse_type = "total_reuse", volume = 16384 : i64}, {depth = 1 : i64, iterator = "%arg7", mapped_to = "y", reuse_type = "no_reuse", volume = 0 : i64}], temporal = [{depth = 2 : i64, iterator = "%arg8", reuse_type = "no_reuse", volume = 0 : i64}, {depth = 3 : i64, iterator = "%arg9", reuse_type = "total_reuse", volume = 16384 : i64}]}} : memref<*xf32> to memref<64x64xf32, strided<[512, 1], offset: ?>>
              %alloc_2 = memref.alloc() {loom.alloc = {local = true, memory_name = "L1", size = 16384 : i64}} : memref<64x64xf32>
              memref.copy %reinterpret_cast_1, %alloc_2 {loom.copy.choice = {dim = "x", interconnect_name = "big_ring", kind = "broadcast"}} : memref<64x64xf32, strided<[512, 1], offset: ?>> to memref<64x64xf32>
              %19 = bufferization.to_tensor %alloc_2 restrict writable : memref<64x64xf32> to tensor<64x64xf32>
              %20 = linalg.matmul ins(%17, %19 : tensor<64x64xf32>, tensor<64x64xf32>) outs(%13 : tensor<64x64xf32>) -> tensor<64x64xf32>
              %21 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%arg11, %20 : tensor<64x64xf32>, tensor<64x64xf32>) outs(%arg11 : tensor<64x64xf32>) {
              ^bb0(%in: f32, %in_3: f32, %out: f32):
                %22 = arith.addf %in, %in_3 : f32
                linalg.yield %22 : f32
              } -> tensor<64x64xf32>
              scf.yield %21 : tensor<64x64xf32>
            }
            %15 = affine.apply affine_map<(d0, d1, d2, d3) -> (d0 * 128 + d1 * 64 + d2 * 1048576 + d3 * 32768)>(%arg7, %arg8, %arg6, %arg9)
            %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%15], sizes: [64, 64], strides: [512, 1] {loom.reuse = {spatial = [{depth = 0 : i64, iterator = "%arg6", mapped_to = "x", reuse_type = "no_reuse", volume = 0 : i64}, {depth = 1 : i64, iterator = "%arg7", mapped_to = "y", reuse_type = "no_reuse", volume = 0 : i64}], temporal = [{depth = 2 : i64, iterator = "%arg8", reuse_type = "no_reuse", volume = 0 : i64}, {depth = 3 : i64, iterator = "%arg9", reuse_type = "no_reuse", volume = 0 : i64}]}} : memref<*xf32> to memref<64x64xf32, strided<[512, 1], offset: ?>>
            bufferization.materialize_in_destination %14 in writable %reinterpret_cast : (tensor<64x64xf32>, memref<64x64xf32, strided<[512, 1], offset: ?>>) -> ()
          }
        }
      } {loom.mapped_to = "y"}
    } {loom.mapped_to = "x"}
    return
  }
  func.func @matmul_kernel__d0i0_d1i1__f10__hoist_block_1__c0mem_c1mem(%arg0: memref<*xf32> {tt.divisibility = 16 : i32}, %arg1: memref<*xf32> {tt.divisibility = 16 : i32}, %arg2: memref<*xf32> {tt.divisibility = 16 : i32}, %arg3: index, %arg4: index, %arg5: index) {
    affine.parallel (%arg6) = (0) to (32) {
      affine.parallel (%arg7) = (0) to (2) {
        affine.for %arg8 = 0 to affine_map<()[s0, s1] -> (s1 ceildiv 2)>()[%arg3, %arg4] {
          %12 = affine.apply affine_map<(d0, d1) -> (d0 * 128 + d1 * 64)>(%arg7, %arg8)
          %reinterpret_cast = memref.reinterpret_cast %arg1 to offset: [%12], sizes: [8, 64, 64], strides: [32768, 512, 1] {loom.reuse = {spatial = [{depth = 0 : i64, iterator = "%arg6", mapped_to = "x", reuse_type = "total_reuse", volume = 131072 : i64}, {depth = 1 : i64, iterator = "%arg7", mapped_to = "y", reuse_type = "no_reuse", volume = 0 : i64}], temporal = [{depth = 2 : i64, iterator = "%arg8", reuse_type = "no_reuse", volume = 0 : i64}]}} : memref<*xf32> to memref<8x64x64xf32, strided<[32768, 512, 1], offset: ?>>
          %alloc = memref.alloc() {loom.alloc = {local = true, memory_name = "L1", size = 131072 : i64}} : memref<8x64x64xf32>
          memref.copy %reinterpret_cast, %alloc {loom.copy.choice = {kind = "mem", memory_name = "L1"}} : memref<8x64x64xf32, strided<[32768, 512, 1], offset: ?>> to memref<8x64x64xf32>
          affine.for %arg9 = 0 to affine_map<()[s0, s1] -> (s0 ceildiv 32)>()[%arg3, %arg4] {
            %cst = arith.constant 0.000000e+00 : f32
            %13 = tensor.empty() : tensor<64x64xf32>
            %14 = linalg.fill ins(%cst : f32) outs(%13 : tensor<64x64xf32>) -> tensor<64x64xf32>
            %c0 = arith.constant 0 : index
            %c8 = arith.constant 8 : index
            %c1 = arith.constant 1 : index
            %15 = scf.for %arg10 = %c0 to %c8 step %c1 iter_args(%arg11 = %14) -> (tensor<64x64xf32>) {
              %17 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 64 + d1 * 1048576 + d2 * 32768)>(%arg10, %arg6, %arg9)
              %reinterpret_cast_1 = memref.reinterpret_cast %arg0 to offset: [%17], sizes: [64, 64], strides: [512, 1] {loom.reuse = {sequential = [{depth = 4 : i64, iterator = "%arg10", reuse_type = "no_reuse", volume = 0 : i64}], spatial = [{depth = 0 : i64, iterator = "%arg6", mapped_to = "x", reuse_type = "no_reuse", volume = 0 : i64}, {depth = 1 : i64, iterator = "%arg7", mapped_to = "y", reuse_type = "total_reuse", volume = 16384 : i64}], temporal = [{depth = 2 : i64, iterator = "%arg8", reuse_type = "total_reuse", volume = 16384 : i64}, {depth = 3 : i64, iterator = "%arg9", reuse_type = "no_reuse", volume = 0 : i64}]}} : memref<*xf32> to memref<64x64xf32, strided<[512, 1], offset: ?>>
              %alloc_2 = memref.alloc() {loom.alloc = {local = true, memory_name = "L1", size = 16384 : i64}} : memref<64x64xf32>
              memref.copy %reinterpret_cast_1, %alloc_2 {loom.copy.choice = {kind = "mem", memory_name = "L1"}} : memref<64x64xf32, strided<[512, 1], offset: ?>> to memref<64x64xf32>
              %18 = bufferization.to_tensor %alloc_2 restrict writable : memref<64x64xf32> to tensor<64x64xf32>
              %subview = memref.subview %alloc[%arg10, 0, 0] [1, 64, 64] [1, 1, 1] : memref<8x64x64xf32> to memref<64x64xf32, strided<[64, 1], offset: ?>>
              %19 = bufferization.to_tensor %subview restrict writable : memref<64x64xf32, strided<[64, 1], offset: ?>> to tensor<64x64xf32>
              %20 = linalg.matmul ins(%18, %19 : tensor<64x64xf32>, tensor<64x64xf32>) outs(%14 : tensor<64x64xf32>) -> tensor<64x64xf32>
              %21 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%arg11, %20 : tensor<64x64xf32>, tensor<64x64xf32>) outs(%arg11 : tensor<64x64xf32>) {
              ^bb0(%in: f32, %in_3: f32, %out: f32):
                %22 = arith.addf %in, %in_3 : f32
                linalg.yield %22 : f32
              } -> tensor<64x64xf32>
              scf.yield %21 : tensor<64x64xf32>
            }
            %16 = affine.apply affine_map<(d0, d1, d2, d3) -> (d0 * 128 + d1 * 64 + d2 * 1048576 + d3 * 32768)>(%arg7, %arg8, %arg6, %arg9)
            %reinterpret_cast_0 = memref.reinterpret_cast %arg2 to offset: [%16], sizes: [64, 64], strides: [512, 1] {loom.reuse = {spatial = [{depth = 0 : i64, iterator = "%arg6", mapped_to = "x", reuse_type = "no_reuse", volume = 0 : i64}, {depth = 1 : i64, iterator = "%arg7", mapped_to = "y", reuse_type = "no_reuse", volume = 0 : i64}], temporal = [{depth = 2 : i64, iterator = "%arg8", reuse_type = "no_reuse", volume = 0 : i64}, {depth = 3 : i64, iterator = "%arg9", reuse_type = "no_reuse", volume = 0 : i64}]}} : memref<*xf32> to memref<64x64xf32, strided<[512, 1], offset: ?>>
            bufferization.materialize_in_destination %15 in writable %reinterpret_cast_0 : (tensor<64x64xf32>, memref<64x64xf32, strided<[512, 1], offset: ?>>) -> ()
          }
        }
      } {loom.mapped_to = "y"}
    } {loom.mapped_to = "x"}
    return
  }
  func.func @matmul_kernel__d0i0_d1i1__f10__hoist_block_1__c0bx_c1mem(%arg0: memref<*xf32> {tt.divisibility = 16 : i32}, %arg1: memref<*xf32> {tt.divisibility = 16 : i32}, %arg2: memref<*xf32> {tt.divisibility = 16 : i32}, %arg3: index, %arg4: index, %arg5: index) {
    affine.parallel (%arg6) = (0) to (32) {
      affine.parallel (%arg7) = (0) to (2) {
        affine.for %arg8 = 0 to affine_map<()[s0, s1] -> (s1 ceildiv 2)>()[%arg3, %arg4] {
          %12 = affine.apply affine_map<(d0, d1) -> (d0 * 128 + d1 * 64)>(%arg7, %arg8)
          %reinterpret_cast = memref.reinterpret_cast %arg1 to offset: [%12], sizes: [8, 64, 64], strides: [32768, 512, 1] {loom.reuse = {spatial = [{depth = 0 : i64, iterator = "%arg6", mapped_to = "x", reuse_type = "total_reuse", volume = 131072 : i64}, {depth = 1 : i64, iterator = "%arg7", mapped_to = "y", reuse_type = "no_reuse", volume = 0 : i64}], temporal = [{depth = 2 : i64, iterator = "%arg8", reuse_type = "no_reuse", volume = 0 : i64}]}} : memref<*xf32> to memref<8x64x64xf32, strided<[32768, 512, 1], offset: ?>>
          %alloc = memref.alloc() {loom.alloc = {local = true, memory_name = "L1", size = 131072 : i64}} : memref<8x64x64xf32>
          memref.copy %reinterpret_cast, %alloc {loom.copy.choice = {dim = "x", interconnect_name = "big_ring", kind = "broadcast"}} : memref<8x64x64xf32, strided<[32768, 512, 1], offset: ?>> to memref<8x64x64xf32>
          affine.for %arg9 = 0 to affine_map<()[s0, s1] -> (s0 ceildiv 32)>()[%arg3, %arg4] {
            %cst = arith.constant 0.000000e+00 : f32
            %13 = tensor.empty() : tensor<64x64xf32>
            %14 = linalg.fill ins(%cst : f32) outs(%13 : tensor<64x64xf32>) -> tensor<64x64xf32>
            %c0 = arith.constant 0 : index
            %c8 = arith.constant 8 : index
            %c1 = arith.constant 1 : index
            %15 = scf.for %arg10 = %c0 to %c8 step %c1 iter_args(%arg11 = %14) -> (tensor<64x64xf32>) {
              %17 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 64 + d1 * 1048576 + d2 * 32768)>(%arg10, %arg6, %arg9)
              %reinterpret_cast_1 = memref.reinterpret_cast %arg0 to offset: [%17], sizes: [64, 64], strides: [512, 1] {loom.reuse = {sequential = [{depth = 4 : i64, iterator = "%arg10", reuse_type = "no_reuse", volume = 0 : i64}], spatial = [{depth = 0 : i64, iterator = "%arg6", mapped_to = "x", reuse_type = "no_reuse", volume = 0 : i64}, {depth = 1 : i64, iterator = "%arg7", mapped_to = "y", reuse_type = "total_reuse", volume = 16384 : i64}], temporal = [{depth = 2 : i64, iterator = "%arg8", reuse_type = "total_reuse", volume = 16384 : i64}, {depth = 3 : i64, iterator = "%arg9", reuse_type = "no_reuse", volume = 0 : i64}]}} : memref<*xf32> to memref<64x64xf32, strided<[512, 1], offset: ?>>
              %alloc_2 = memref.alloc() {loom.alloc = {local = true, memory_name = "L1", size = 16384 : i64}} : memref<64x64xf32>
              memref.copy %reinterpret_cast_1, %alloc_2 {loom.copy.choice = {kind = "mem", memory_name = "L1"}} : memref<64x64xf32, strided<[512, 1], offset: ?>> to memref<64x64xf32>
              %18 = bufferization.to_tensor %alloc_2 restrict writable : memref<64x64xf32> to tensor<64x64xf32>
              %subview = memref.subview %alloc[%arg10, 0, 0] [1, 64, 64] [1, 1, 1] : memref<8x64x64xf32> to memref<64x64xf32, strided<[64, 1], offset: ?>>
              %19 = bufferization.to_tensor %subview restrict writable : memref<64x64xf32, strided<[64, 1], offset: ?>> to tensor<64x64xf32>
              %20 = linalg.matmul ins(%18, %19 : tensor<64x64xf32>, tensor<64x64xf32>) outs(%14 : tensor<64x64xf32>) -> tensor<64x64xf32>
              %21 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%arg11, %20 : tensor<64x64xf32>, tensor<64x64xf32>) outs(%arg11 : tensor<64x64xf32>) {
              ^bb0(%in: f32, %in_3: f32, %out: f32):
                %22 = arith.addf %in, %in_3 : f32
                linalg.yield %22 : f32
              } -> tensor<64x64xf32>
              scf.yield %21 : tensor<64x64xf32>
            }
            %16 = affine.apply affine_map<(d0, d1, d2, d3) -> (d0 * 128 + d1 * 64 + d2 * 1048576 + d3 * 32768)>(%arg7, %arg8, %arg6, %arg9)
            %reinterpret_cast_0 = memref.reinterpret_cast %arg2 to offset: [%16], sizes: [64, 64], strides: [512, 1] {loom.reuse = {spatial = [{depth = 0 : i64, iterator = "%arg6", mapped_to = "x", reuse_type = "no_reuse", volume = 0 : i64}, {depth = 1 : i64, iterator = "%arg7", mapped_to = "y", reuse_type = "no_reuse", volume = 0 : i64}], temporal = [{depth = 2 : i64, iterator = "%arg8", reuse_type = "no_reuse", volume = 0 : i64}, {depth = 3 : i64, iterator = "%arg9", reuse_type = "no_reuse", volume = 0 : i64}]}} : memref<*xf32> to memref<64x64xf32, strided<[512, 1], offset: ?>>
            bufferization.materialize_in_destination %15 in writable %reinterpret_cast_0 : (tensor<64x64xf32>, memref<64x64xf32, strided<[512, 1], offset: ?>>) -> ()
          }
        }
      } {loom.mapped_to = "y"}
    } {loom.mapped_to = "x"}
    return
  }
  func.func @matmul_kernel__d1i0_d0i1__f01__c0mem_c1mem(%arg0: memref<*xf32> {tt.divisibility = 16 : i32}, %arg1: memref<*xf32> {tt.divisibility = 16 : i32}, %arg2: memref<*xf32> {tt.divisibility = 16 : i32}, %arg3: index, %arg4: index, %arg5: index) {
    affine.parallel (%arg6) = (0) to (2) {
      affine.parallel (%arg7) = (0) to (32) {
        affine.for %arg8 = 0 to affine_map<()[s0, s1] -> (s0 ceildiv 2)>()[%arg3, %arg4] {
          affine.for %arg9 = 0 to affine_map<()[s0, s1] -> (s1 ceildiv 32)>()[%arg3, %arg4] {
            %cst = arith.constant 0.000000e+00 : f32
            %12 = tensor.empty() : tensor<64x64xf32>
            %13 = linalg.fill ins(%cst : f32) outs(%12 : tensor<64x64xf32>) -> tensor<64x64xf32>
            %c0 = arith.constant 0 : index
            %c8 = arith.constant 8 : index
            %c1 = arith.constant 1 : index
            %14 = scf.for %arg10 = %c0 to %c8 step %c1 iter_args(%arg11 = %13) -> (tensor<64x64xf32>) {
              %16 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 64 + d1 * 65536 + d2 * 32768)>(%arg10, %arg6, %arg8)
              %reinterpret_cast_0 = memref.reinterpret_cast %arg0 to offset: [%16], sizes: [64, 64], strides: [512, 1] {loom.reuse = {sequential = [{depth = 4 : i64, iterator = "%arg10", reuse_type = "no_reuse", volume = 0 : i64}], spatial = [{depth = 0 : i64, iterator = "%arg6", mapped_to = "y", reuse_type = "no_reuse", volume = 0 : i64}, {depth = 1 : i64, iterator = "%arg7", mapped_to = "x", reuse_type = "total_reuse", volume = 16384 : i64}], temporal = [{depth = 2 : i64, iterator = "%arg8", reuse_type = "no_reuse", volume = 0 : i64}, {depth = 3 : i64, iterator = "%arg9", reuse_type = "total_reuse", volume = 16384 : i64}]}} : memref<*xf32> to memref<64x64xf32, strided<[512, 1], offset: ?>>
              %alloc = memref.alloc() {loom.alloc = {local = true, memory_name = "L1", size = 16384 : i64}} : memref<64x64xf32>
              memref.copy %reinterpret_cast_0, %alloc {loom.copy.choice = {kind = "mem", memory_name = "L1"}} : memref<64x64xf32, strided<[512, 1], offset: ?>> to memref<64x64xf32>
              %17 = bufferization.to_tensor %alloc restrict writable : memref<64x64xf32> to tensor<64x64xf32>
              %18 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 32768 + d1 * 2048 + d2 * 64)>(%arg10, %arg7, %arg9)
              %reinterpret_cast_1 = memref.reinterpret_cast %arg1 to offset: [%18], sizes: [64, 64], strides: [512, 1] {loom.reuse = {sequential = [{depth = 4 : i64, iterator = "%arg10", reuse_type = "no_reuse", volume = 0 : i64}], spatial = [{depth = 0 : i64, iterator = "%arg6", mapped_to = "y", reuse_type = "total_reuse", volume = 16384 : i64}, {depth = 1 : i64, iterator = "%arg7", mapped_to = "x", reuse_type = "no_reuse", volume = 0 : i64}], temporal = [{depth = 2 : i64, iterator = "%arg8", reuse_type = "total_reuse", volume = 16384 : i64}, {depth = 3 : i64, iterator = "%arg9", reuse_type = "no_reuse", volume = 0 : i64}]}} : memref<*xf32> to memref<64x64xf32, strided<[512, 1], offset: ?>>
              %alloc_2 = memref.alloc() {loom.alloc = {local = true, memory_name = "L1", size = 16384 : i64}} : memref<64x64xf32>
              memref.copy %reinterpret_cast_1, %alloc_2 {loom.copy.choice = {kind = "mem", memory_name = "L1"}} : memref<64x64xf32, strided<[512, 1], offset: ?>> to memref<64x64xf32>
              %19 = bufferization.to_tensor %alloc_2 restrict writable : memref<64x64xf32> to tensor<64x64xf32>
              %20 = linalg.matmul ins(%17, %19 : tensor<64x64xf32>, tensor<64x64xf32>) outs(%13 : tensor<64x64xf32>) -> tensor<64x64xf32>
              %21 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%arg11, %20 : tensor<64x64xf32>, tensor<64x64xf32>) outs(%arg11 : tensor<64x64xf32>) {
              ^bb0(%in: f32, %in_3: f32, %out: f32):
                %22 = arith.addf %in, %in_3 : f32
                linalg.yield %22 : f32
              } -> tensor<64x64xf32>
              scf.yield %21 : tensor<64x64xf32>
            }
            %15 = affine.apply affine_map<(d0, d1, d2, d3) -> (d0 * 2048 + d1 * 64 + d2 * 65536 + d3 * 32768)>(%arg7, %arg9, %arg6, %arg8)
            %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%15], sizes: [64, 64], strides: [512, 1] {loom.reuse = {spatial = [{depth = 0 : i64, iterator = "%arg6", mapped_to = "y", reuse_type = "no_reuse", volume = 0 : i64}, {depth = 1 : i64, iterator = "%arg7", mapped_to = "x", reuse_type = "no_reuse", volume = 0 : i64}], temporal = [{depth = 2 : i64, iterator = "%arg8", reuse_type = "no_reuse", volume = 0 : i64}, {depth = 3 : i64, iterator = "%arg9", reuse_type = "no_reuse", volume = 0 : i64}]}} : memref<*xf32> to memref<64x64xf32, strided<[512, 1], offset: ?>>
            bufferization.materialize_in_destination %14 in writable %reinterpret_cast : (tensor<64x64xf32>, memref<64x64xf32, strided<[512, 1], offset: ?>>) -> ()
          }
        }
      } {loom.mapped_to = "x"}
    } {loom.mapped_to = "y"}
    return
  }
  func.func @matmul_kernel__d1i0_d0i1__f01__c0bx_c1mem(%arg0: memref<*xf32> {tt.divisibility = 16 : i32}, %arg1: memref<*xf32> {tt.divisibility = 16 : i32}, %arg2: memref<*xf32> {tt.divisibility = 16 : i32}, %arg3: index, %arg4: index, %arg5: index) {
    affine.parallel (%arg6) = (0) to (2) {
      affine.parallel (%arg7) = (0) to (32) {
        affine.for %arg8 = 0 to affine_map<()[s0, s1] -> (s0 ceildiv 2)>()[%arg3, %arg4] {
          affine.for %arg9 = 0 to affine_map<()[s0, s1] -> (s1 ceildiv 32)>()[%arg3, %arg4] {
            %cst = arith.constant 0.000000e+00 : f32
            %12 = tensor.empty() : tensor<64x64xf32>
            %13 = linalg.fill ins(%cst : f32) outs(%12 : tensor<64x64xf32>) -> tensor<64x64xf32>
            %c0 = arith.constant 0 : index
            %c8 = arith.constant 8 : index
            %c1 = arith.constant 1 : index
            %14 = scf.for %arg10 = %c0 to %c8 step %c1 iter_args(%arg11 = %13) -> (tensor<64x64xf32>) {
              %16 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 64 + d1 * 65536 + d2 * 32768)>(%arg10, %arg6, %arg8)
              %reinterpret_cast_0 = memref.reinterpret_cast %arg0 to offset: [%16], sizes: [64, 64], strides: [512, 1] {loom.reuse = {sequential = [{depth = 4 : i64, iterator = "%arg10", reuse_type = "no_reuse", volume = 0 : i64}], spatial = [{depth = 0 : i64, iterator = "%arg6", mapped_to = "y", reuse_type = "no_reuse", volume = 0 : i64}, {depth = 1 : i64, iterator = "%arg7", mapped_to = "x", reuse_type = "total_reuse", volume = 16384 : i64}], temporal = [{depth = 2 : i64, iterator = "%arg8", reuse_type = "no_reuse", volume = 0 : i64}, {depth = 3 : i64, iterator = "%arg9", reuse_type = "total_reuse", volume = 16384 : i64}]}} : memref<*xf32> to memref<64x64xf32, strided<[512, 1], offset: ?>>
              %alloc = memref.alloc() {loom.alloc = {local = true, memory_name = "L1", size = 16384 : i64}} : memref<64x64xf32>
              memref.copy %reinterpret_cast_0, %alloc {loom.copy.choice = {dim = "x", interconnect_name = "big_ring", kind = "broadcast"}} : memref<64x64xf32, strided<[512, 1], offset: ?>> to memref<64x64xf32>
              %17 = bufferization.to_tensor %alloc restrict writable : memref<64x64xf32> to tensor<64x64xf32>
              %18 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 32768 + d1 * 2048 + d2 * 64)>(%arg10, %arg7, %arg9)
              %reinterpret_cast_1 = memref.reinterpret_cast %arg1 to offset: [%18], sizes: [64, 64], strides: [512, 1] {loom.reuse = {sequential = [{depth = 4 : i64, iterator = "%arg10", reuse_type = "no_reuse", volume = 0 : i64}], spatial = [{depth = 0 : i64, iterator = "%arg6", mapped_to = "y", reuse_type = "total_reuse", volume = 16384 : i64}, {depth = 1 : i64, iterator = "%arg7", mapped_to = "x", reuse_type = "no_reuse", volume = 0 : i64}], temporal = [{depth = 2 : i64, iterator = "%arg8", reuse_type = "total_reuse", volume = 16384 : i64}, {depth = 3 : i64, iterator = "%arg9", reuse_type = "no_reuse", volume = 0 : i64}]}} : memref<*xf32> to memref<64x64xf32, strided<[512, 1], offset: ?>>
              %alloc_2 = memref.alloc() {loom.alloc = {local = true, memory_name = "L1", size = 16384 : i64}} : memref<64x64xf32>
              memref.copy %reinterpret_cast_1, %alloc_2 {loom.copy.choice = {kind = "mem", memory_name = "L1"}} : memref<64x64xf32, strided<[512, 1], offset: ?>> to memref<64x64xf32>
              %19 = bufferization.to_tensor %alloc_2 restrict writable : memref<64x64xf32> to tensor<64x64xf32>
              %20 = linalg.matmul ins(%17, %19 : tensor<64x64xf32>, tensor<64x64xf32>) outs(%13 : tensor<64x64xf32>) -> tensor<64x64xf32>
              %21 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%arg11, %20 : tensor<64x64xf32>, tensor<64x64xf32>) outs(%arg11 : tensor<64x64xf32>) {
              ^bb0(%in: f32, %in_3: f32, %out: f32):
                %22 = arith.addf %in, %in_3 : f32
                linalg.yield %22 : f32
              } -> tensor<64x64xf32>
              scf.yield %21 : tensor<64x64xf32>
            }
            %15 = affine.apply affine_map<(d0, d1, d2, d3) -> (d0 * 2048 + d1 * 64 + d2 * 65536 + d3 * 32768)>(%arg7, %arg9, %arg6, %arg8)
            %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%15], sizes: [64, 64], strides: [512, 1] {loom.reuse = {spatial = [{depth = 0 : i64, iterator = "%arg6", mapped_to = "y", reuse_type = "no_reuse", volume = 0 : i64}, {depth = 1 : i64, iterator = "%arg7", mapped_to = "x", reuse_type = "no_reuse", volume = 0 : i64}], temporal = [{depth = 2 : i64, iterator = "%arg8", reuse_type = "no_reuse", volume = 0 : i64}, {depth = 3 : i64, iterator = "%arg9", reuse_type = "no_reuse", volume = 0 : i64}]}} : memref<*xf32> to memref<64x64xf32, strided<[512, 1], offset: ?>>
            bufferization.materialize_in_destination %14 in writable %reinterpret_cast : (tensor<64x64xf32>, memref<64x64xf32, strided<[512, 1], offset: ?>>) -> ()
          }
        }
      } {loom.mapped_to = "x"}
    } {loom.mapped_to = "y"}
    return
  }
  func.func @matmul_kernel__d1i0_d0i1__f01__hoist_block_0__c0mem_c1mem(%arg0: memref<*xf32> {tt.divisibility = 16 : i32}, %arg1: memref<*xf32> {tt.divisibility = 16 : i32}, %arg2: memref<*xf32> {tt.divisibility = 16 : i32}, %arg3: index, %arg4: index, %arg5: index) {
    affine.parallel (%arg6) = (0) to (2) {
      affine.parallel (%arg7) = (0) to (32) {
        affine.for %arg8 = 0 to affine_map<()[s0, s1] -> (s0 ceildiv 2)>()[%arg3, %arg4] {
          %12 = affine.apply affine_map<(d0, d1) -> (d0 * 65536 + d1 * 32768)>(%arg6, %arg8)
          %reinterpret_cast = memref.reinterpret_cast %arg0 to offset: [%12], sizes: [8, 64, 64], strides: [64, 512, 1] {loom.reuse = {spatial = [{depth = 0 : i64, iterator = "%arg6", mapped_to = "y", reuse_type = "no_reuse", volume = 0 : i64}, {depth = 1 : i64, iterator = "%arg7", mapped_to = "x", reuse_type = "total_reuse", volume = 131072 : i64}], temporal = [{depth = 2 : i64, iterator = "%arg8", reuse_type = "no_reuse", volume = 0 : i64}]}} : memref<*xf32> to memref<8x64x64xf32, strided<[64, 512, 1], offset: ?>>
          %alloc = memref.alloc() {loom.alloc = {local = true, memory_name = "L1", size = 131072 : i64}} : memref<8x64x64xf32>
          memref.copy %reinterpret_cast, %alloc {loom.copy.choice = {kind = "mem", memory_name = "L1"}} : memref<8x64x64xf32, strided<[64, 512, 1], offset: ?>> to memref<8x64x64xf32>
          affine.for %arg9 = 0 to affine_map<()[s0, s1] -> (s1 ceildiv 32)>()[%arg3, %arg4] {
            %cst = arith.constant 0.000000e+00 : f32
            %13 = tensor.empty() : tensor<64x64xf32>
            %14 = linalg.fill ins(%cst : f32) outs(%13 : tensor<64x64xf32>) -> tensor<64x64xf32>
            %c0 = arith.constant 0 : index
            %c8 = arith.constant 8 : index
            %c1 = arith.constant 1 : index
            %15 = scf.for %arg10 = %c0 to %c8 step %c1 iter_args(%arg11 = %14) -> (tensor<64x64xf32>) {
              %subview = memref.subview %alloc[%arg10, 0, 0] [1, 64, 64] [1, 1, 1] : memref<8x64x64xf32> to memref<64x64xf32, strided<[64, 1], offset: ?>>
              %17 = bufferization.to_tensor %subview restrict writable : memref<64x64xf32, strided<[64, 1], offset: ?>> to tensor<64x64xf32>
              %18 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 32768 + d1 * 2048 + d2 * 64)>(%arg10, %arg7, %arg9)
              %reinterpret_cast_1 = memref.reinterpret_cast %arg1 to offset: [%18], sizes: [64, 64], strides: [512, 1] {loom.reuse = {sequential = [{depth = 4 : i64, iterator = "%arg10", reuse_type = "no_reuse", volume = 0 : i64}], spatial = [{depth = 0 : i64, iterator = "%arg6", mapped_to = "y", reuse_type = "total_reuse", volume = 16384 : i64}, {depth = 1 : i64, iterator = "%arg7", mapped_to = "x", reuse_type = "no_reuse", volume = 0 : i64}], temporal = [{depth = 2 : i64, iterator = "%arg8", reuse_type = "total_reuse", volume = 16384 : i64}, {depth = 3 : i64, iterator = "%arg9", reuse_type = "no_reuse", volume = 0 : i64}]}} : memref<*xf32> to memref<64x64xf32, strided<[512, 1], offset: ?>>
              %alloc_2 = memref.alloc() {loom.alloc = {local = true, memory_name = "L1", size = 16384 : i64}} : memref<64x64xf32>
              memref.copy %reinterpret_cast_1, %alloc_2 {loom.copy.choice = {kind = "mem", memory_name = "L1"}} : memref<64x64xf32, strided<[512, 1], offset: ?>> to memref<64x64xf32>
              %19 = bufferization.to_tensor %alloc_2 restrict writable : memref<64x64xf32> to tensor<64x64xf32>
              %20 = linalg.matmul ins(%17, %19 : tensor<64x64xf32>, tensor<64x64xf32>) outs(%14 : tensor<64x64xf32>) -> tensor<64x64xf32>
              %21 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%arg11, %20 : tensor<64x64xf32>, tensor<64x64xf32>) outs(%arg11 : tensor<64x64xf32>) {
              ^bb0(%in: f32, %in_3: f32, %out: f32):
                %22 = arith.addf %in, %in_3 : f32
                linalg.yield %22 : f32
              } -> tensor<64x64xf32>
              scf.yield %21 : tensor<64x64xf32>
            }
            %16 = affine.apply affine_map<(d0, d1, d2, d3) -> (d0 * 2048 + d1 * 64 + d2 * 65536 + d3 * 32768)>(%arg7, %arg9, %arg6, %arg8)
            %reinterpret_cast_0 = memref.reinterpret_cast %arg2 to offset: [%16], sizes: [64, 64], strides: [512, 1] {loom.reuse = {spatial = [{depth = 0 : i64, iterator = "%arg6", mapped_to = "y", reuse_type = "no_reuse", volume = 0 : i64}, {depth = 1 : i64, iterator = "%arg7", mapped_to = "x", reuse_type = "no_reuse", volume = 0 : i64}], temporal = [{depth = 2 : i64, iterator = "%arg8", reuse_type = "no_reuse", volume = 0 : i64}, {depth = 3 : i64, iterator = "%arg9", reuse_type = "no_reuse", volume = 0 : i64}]}} : memref<*xf32> to memref<64x64xf32, strided<[512, 1], offset: ?>>
            bufferization.materialize_in_destination %15 in writable %reinterpret_cast_0 : (tensor<64x64xf32>, memref<64x64xf32, strided<[512, 1], offset: ?>>) -> ()
          }
        }
      } {loom.mapped_to = "x"}
    } {loom.mapped_to = "y"}
    return
  }
  func.func @matmul_kernel__d1i0_d0i1__f01__hoist_block_0__c0bx_c1mem(%arg0: memref<*xf32> {tt.divisibility = 16 : i32}, %arg1: memref<*xf32> {tt.divisibility = 16 : i32}, %arg2: memref<*xf32> {tt.divisibility = 16 : i32}, %arg3: index, %arg4: index, %arg5: index) {
    affine.parallel (%arg6) = (0) to (2) {
      affine.parallel (%arg7) = (0) to (32) {
        affine.for %arg8 = 0 to affine_map<()[s0, s1] -> (s0 ceildiv 2)>()[%arg3, %arg4] {
          %12 = affine.apply affine_map<(d0, d1) -> (d0 * 65536 + d1 * 32768)>(%arg6, %arg8)
          %reinterpret_cast = memref.reinterpret_cast %arg0 to offset: [%12], sizes: [8, 64, 64], strides: [64, 512, 1] {loom.reuse = {spatial = [{depth = 0 : i64, iterator = "%arg6", mapped_to = "y", reuse_type = "no_reuse", volume = 0 : i64}, {depth = 1 : i64, iterator = "%arg7", mapped_to = "x", reuse_type = "total_reuse", volume = 131072 : i64}], temporal = [{depth = 2 : i64, iterator = "%arg8", reuse_type = "no_reuse", volume = 0 : i64}]}} : memref<*xf32> to memref<8x64x64xf32, strided<[64, 512, 1], offset: ?>>
          %alloc = memref.alloc() {loom.alloc = {local = true, memory_name = "L1", size = 131072 : i64}} : memref<8x64x64xf32>
          memref.copy %reinterpret_cast, %alloc {loom.copy.choice = {dim = "x", interconnect_name = "big_ring", kind = "broadcast"}} : memref<8x64x64xf32, strided<[64, 512, 1], offset: ?>> to memref<8x64x64xf32>
          affine.for %arg9 = 0 to affine_map<()[s0, s1] -> (s1 ceildiv 32)>()[%arg3, %arg4] {
            %cst = arith.constant 0.000000e+00 : f32
            %13 = tensor.empty() : tensor<64x64xf32>
            %14 = linalg.fill ins(%cst : f32) outs(%13 : tensor<64x64xf32>) -> tensor<64x64xf32>
            %c0 = arith.constant 0 : index
            %c8 = arith.constant 8 : index
            %c1 = arith.constant 1 : index
            %15 = scf.for %arg10 = %c0 to %c8 step %c1 iter_args(%arg11 = %14) -> (tensor<64x64xf32>) {
              %subview = memref.subview %alloc[%arg10, 0, 0] [1, 64, 64] [1, 1, 1] : memref<8x64x64xf32> to memref<64x64xf32, strided<[64, 1], offset: ?>>
              %17 = bufferization.to_tensor %subview restrict writable : memref<64x64xf32, strided<[64, 1], offset: ?>> to tensor<64x64xf32>
              %18 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 32768 + d1 * 2048 + d2 * 64)>(%arg10, %arg7, %arg9)
              %reinterpret_cast_1 = memref.reinterpret_cast %arg1 to offset: [%18], sizes: [64, 64], strides: [512, 1] {loom.reuse = {sequential = [{depth = 4 : i64, iterator = "%arg10", reuse_type = "no_reuse", volume = 0 : i64}], spatial = [{depth = 0 : i64, iterator = "%arg6", mapped_to = "y", reuse_type = "total_reuse", volume = 16384 : i64}, {depth = 1 : i64, iterator = "%arg7", mapped_to = "x", reuse_type = "no_reuse", volume = 0 : i64}], temporal = [{depth = 2 : i64, iterator = "%arg8", reuse_type = "total_reuse", volume = 16384 : i64}, {depth = 3 : i64, iterator = "%arg9", reuse_type = "no_reuse", volume = 0 : i64}]}} : memref<*xf32> to memref<64x64xf32, strided<[512, 1], offset: ?>>
              %alloc_2 = memref.alloc() {loom.alloc = {local = true, memory_name = "L1", size = 16384 : i64}} : memref<64x64xf32>
              memref.copy %reinterpret_cast_1, %alloc_2 {loom.copy.choice = {kind = "mem", memory_name = "L1"}} : memref<64x64xf32, strided<[512, 1], offset: ?>> to memref<64x64xf32>
              %19 = bufferization.to_tensor %alloc_2 restrict writable : memref<64x64xf32> to tensor<64x64xf32>
              %20 = linalg.matmul ins(%17, %19 : tensor<64x64xf32>, tensor<64x64xf32>) outs(%14 : tensor<64x64xf32>) -> tensor<64x64xf32>
              %21 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%arg11, %20 : tensor<64x64xf32>, tensor<64x64xf32>) outs(%arg11 : tensor<64x64xf32>) {
              ^bb0(%in: f32, %in_3: f32, %out: f32):
                %22 = arith.addf %in, %in_3 : f32
                linalg.yield %22 : f32
              } -> tensor<64x64xf32>
              scf.yield %21 : tensor<64x64xf32>
            }
            %16 = affine.apply affine_map<(d0, d1, d2, d3) -> (d0 * 2048 + d1 * 64 + d2 * 65536 + d3 * 32768)>(%arg7, %arg9, %arg6, %arg8)
            %reinterpret_cast_0 = memref.reinterpret_cast %arg2 to offset: [%16], sizes: [64, 64], strides: [512, 1] {loom.reuse = {spatial = [{depth = 0 : i64, iterator = "%arg6", mapped_to = "y", reuse_type = "no_reuse", volume = 0 : i64}, {depth = 1 : i64, iterator = "%arg7", mapped_to = "x", reuse_type = "no_reuse", volume = 0 : i64}], temporal = [{depth = 2 : i64, iterator = "%arg8", reuse_type = "no_reuse", volume = 0 : i64}, {depth = 3 : i64, iterator = "%arg9", reuse_type = "no_reuse", volume = 0 : i64}]}} : memref<*xf32> to memref<64x64xf32, strided<[512, 1], offset: ?>>
            bufferization.materialize_in_destination %15 in writable %reinterpret_cast_0 : (tensor<64x64xf32>, memref<64x64xf32, strided<[512, 1], offset: ?>>) -> ()
          }
        }
      } {loom.mapped_to = "x"}
    } {loom.mapped_to = "y"}
    return
  }
  func.func @matmul_kernel__d1i0_d0i1__f10__c0mem_c1mem(%arg0: memref<*xf32> {tt.divisibility = 16 : i32}, %arg1: memref<*xf32> {tt.divisibility = 16 : i32}, %arg2: memref<*xf32> {tt.divisibility = 16 : i32}, %arg3: index, %arg4: index, %arg5: index) {
    affine.parallel (%arg6) = (0) to (2) {
      affine.parallel (%arg7) = (0) to (32) {
        affine.for %arg8 = 0 to affine_map<()[s0, s1] -> (s1 ceildiv 32)>()[%arg3, %arg4] {
          affine.for %arg9 = 0 to affine_map<()[s0, s1] -> (s0 ceildiv 2)>()[%arg3, %arg4] {
            %cst = arith.constant 0.000000e+00 : f32
            %12 = tensor.empty() : tensor<64x64xf32>
            %13 = linalg.fill ins(%cst : f32) outs(%12 : tensor<64x64xf32>) -> tensor<64x64xf32>
            %c0 = arith.constant 0 : index
            %c8 = arith.constant 8 : index
            %c1 = arith.constant 1 : index
            %14 = scf.for %arg10 = %c0 to %c8 step %c1 iter_args(%arg11 = %13) -> (tensor<64x64xf32>) {
              %16 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 64 + d1 * 65536 + d2 * 32768)>(%arg10, %arg6, %arg9)
              %reinterpret_cast_0 = memref.reinterpret_cast %arg0 to offset: [%16], sizes: [64, 64], strides: [512, 1] {loom.reuse = {sequential = [{depth = 4 : i64, iterator = "%arg10", reuse_type = "no_reuse", volume = 0 : i64}], spatial = [{depth = 0 : i64, iterator = "%arg6", mapped_to = "y", reuse_type = "no_reuse", volume = 0 : i64}, {depth = 1 : i64, iterator = "%arg7", mapped_to = "x", reuse_type = "total_reuse", volume = 16384 : i64}], temporal = [{depth = 2 : i64, iterator = "%arg8", reuse_type = "total_reuse", volume = 16384 : i64}, {depth = 3 : i64, iterator = "%arg9", reuse_type = "no_reuse", volume = 0 : i64}]}} : memref<*xf32> to memref<64x64xf32, strided<[512, 1], offset: ?>>
              %alloc = memref.alloc() {loom.alloc = {local = true, memory_name = "L1", size = 16384 : i64}} : memref<64x64xf32>
              memref.copy %reinterpret_cast_0, %alloc {loom.copy.choice = {kind = "mem", memory_name = "L1"}} : memref<64x64xf32, strided<[512, 1], offset: ?>> to memref<64x64xf32>
              %17 = bufferization.to_tensor %alloc restrict writable : memref<64x64xf32> to tensor<64x64xf32>
              %18 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 32768 + d1 * 2048 + d2 * 64)>(%arg10, %arg7, %arg8)
              %reinterpret_cast_1 = memref.reinterpret_cast %arg1 to offset: [%18], sizes: [64, 64], strides: [512, 1] {loom.reuse = {sequential = [{depth = 4 : i64, iterator = "%arg10", reuse_type = "no_reuse", volume = 0 : i64}], spatial = [{depth = 0 : i64, iterator = "%arg6", mapped_to = "y", reuse_type = "total_reuse", volume = 16384 : i64}, {depth = 1 : i64, iterator = "%arg7", mapped_to = "x", reuse_type = "no_reuse", volume = 0 : i64}], temporal = [{depth = 2 : i64, iterator = "%arg8", reuse_type = "no_reuse", volume = 0 : i64}, {depth = 3 : i64, iterator = "%arg9", reuse_type = "total_reuse", volume = 16384 : i64}]}} : memref<*xf32> to memref<64x64xf32, strided<[512, 1], offset: ?>>
              %alloc_2 = memref.alloc() {loom.alloc = {local = true, memory_name = "L1", size = 16384 : i64}} : memref<64x64xf32>
              memref.copy %reinterpret_cast_1, %alloc_2 {loom.copy.choice = {kind = "mem", memory_name = "L1"}} : memref<64x64xf32, strided<[512, 1], offset: ?>> to memref<64x64xf32>
              %19 = bufferization.to_tensor %alloc_2 restrict writable : memref<64x64xf32> to tensor<64x64xf32>
              %20 = linalg.matmul ins(%17, %19 : tensor<64x64xf32>, tensor<64x64xf32>) outs(%13 : tensor<64x64xf32>) -> tensor<64x64xf32>
              %21 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%arg11, %20 : tensor<64x64xf32>, tensor<64x64xf32>) outs(%arg11 : tensor<64x64xf32>) {
              ^bb0(%in: f32, %in_3: f32, %out: f32):
                %22 = arith.addf %in, %in_3 : f32
                linalg.yield %22 : f32
              } -> tensor<64x64xf32>
              scf.yield %21 : tensor<64x64xf32>
            }
            %15 = affine.apply affine_map<(d0, d1, d2, d3) -> (d0 * 2048 + d1 * 64 + d2 * 65536 + d3 * 32768)>(%arg7, %arg8, %arg6, %arg9)
            %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%15], sizes: [64, 64], strides: [512, 1] {loom.reuse = {spatial = [{depth = 0 : i64, iterator = "%arg6", mapped_to = "y", reuse_type = "no_reuse", volume = 0 : i64}, {depth = 1 : i64, iterator = "%arg7", mapped_to = "x", reuse_type = "no_reuse", volume = 0 : i64}], temporal = [{depth = 2 : i64, iterator = "%arg8", reuse_type = "no_reuse", volume = 0 : i64}, {depth = 3 : i64, iterator = "%arg9", reuse_type = "no_reuse", volume = 0 : i64}]}} : memref<*xf32> to memref<64x64xf32, strided<[512, 1], offset: ?>>
            bufferization.materialize_in_destination %14 in writable %reinterpret_cast : (tensor<64x64xf32>, memref<64x64xf32, strided<[512, 1], offset: ?>>) -> ()
          }
        }
      } {loom.mapped_to = "x"}
    } {loom.mapped_to = "y"}
    return
  }
  func.func @matmul_kernel__d1i0_d0i1__f10__c0bx_c1mem(%arg0: memref<*xf32> {tt.divisibility = 16 : i32}, %arg1: memref<*xf32> {tt.divisibility = 16 : i32}, %arg2: memref<*xf32> {tt.divisibility = 16 : i32}, %arg3: index, %arg4: index, %arg5: index) {
    affine.parallel (%arg6) = (0) to (2) {
      affine.parallel (%arg7) = (0) to (32) {
        affine.for %arg8 = 0 to affine_map<()[s0, s1] -> (s1 ceildiv 32)>()[%arg3, %arg4] {
          affine.for %arg9 = 0 to affine_map<()[s0, s1] -> (s0 ceildiv 2)>()[%arg3, %arg4] {
            %cst = arith.constant 0.000000e+00 : f32
            %12 = tensor.empty() : tensor<64x64xf32>
            %13 = linalg.fill ins(%cst : f32) outs(%12 : tensor<64x64xf32>) -> tensor<64x64xf32>
            %c0 = arith.constant 0 : index
            %c8 = arith.constant 8 : index
            %c1 = arith.constant 1 : index
            %14 = scf.for %arg10 = %c0 to %c8 step %c1 iter_args(%arg11 = %13) -> (tensor<64x64xf32>) {
              %16 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 64 + d1 * 65536 + d2 * 32768)>(%arg10, %arg6, %arg9)
              %reinterpret_cast_0 = memref.reinterpret_cast %arg0 to offset: [%16], sizes: [64, 64], strides: [512, 1] {loom.reuse = {sequential = [{depth = 4 : i64, iterator = "%arg10", reuse_type = "no_reuse", volume = 0 : i64}], spatial = [{depth = 0 : i64, iterator = "%arg6", mapped_to = "y", reuse_type = "no_reuse", volume = 0 : i64}, {depth = 1 : i64, iterator = "%arg7", mapped_to = "x", reuse_type = "total_reuse", volume = 16384 : i64}], temporal = [{depth = 2 : i64, iterator = "%arg8", reuse_type = "total_reuse", volume = 16384 : i64}, {depth = 3 : i64, iterator = "%arg9", reuse_type = "no_reuse", volume = 0 : i64}]}} : memref<*xf32> to memref<64x64xf32, strided<[512, 1], offset: ?>>
              %alloc = memref.alloc() {loom.alloc = {local = true, memory_name = "L1", size = 16384 : i64}} : memref<64x64xf32>
              memref.copy %reinterpret_cast_0, %alloc {loom.copy.choice = {dim = "x", interconnect_name = "big_ring", kind = "broadcast"}} : memref<64x64xf32, strided<[512, 1], offset: ?>> to memref<64x64xf32>
              %17 = bufferization.to_tensor %alloc restrict writable : memref<64x64xf32> to tensor<64x64xf32>
              %18 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 32768 + d1 * 2048 + d2 * 64)>(%arg10, %arg7, %arg8)
              %reinterpret_cast_1 = memref.reinterpret_cast %arg1 to offset: [%18], sizes: [64, 64], strides: [512, 1] {loom.reuse = {sequential = [{depth = 4 : i64, iterator = "%arg10", reuse_type = "no_reuse", volume = 0 : i64}], spatial = [{depth = 0 : i64, iterator = "%arg6", mapped_to = "y", reuse_type = "total_reuse", volume = 16384 : i64}, {depth = 1 : i64, iterator = "%arg7", mapped_to = "x", reuse_type = "no_reuse", volume = 0 : i64}], temporal = [{depth = 2 : i64, iterator = "%arg8", reuse_type = "no_reuse", volume = 0 : i64}, {depth = 3 : i64, iterator = "%arg9", reuse_type = "total_reuse", volume = 16384 : i64}]}} : memref<*xf32> to memref<64x64xf32, strided<[512, 1], offset: ?>>
              %alloc_2 = memref.alloc() {loom.alloc = {local = true, memory_name = "L1", size = 16384 : i64}} : memref<64x64xf32>
              memref.copy %reinterpret_cast_1, %alloc_2 {loom.copy.choice = {kind = "mem", memory_name = "L1"}} : memref<64x64xf32, strided<[512, 1], offset: ?>> to memref<64x64xf32>
              %19 = bufferization.to_tensor %alloc_2 restrict writable : memref<64x64xf32> to tensor<64x64xf32>
              %20 = linalg.matmul ins(%17, %19 : tensor<64x64xf32>, tensor<64x64xf32>) outs(%13 : tensor<64x64xf32>) -> tensor<64x64xf32>
              %21 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%arg11, %20 : tensor<64x64xf32>, tensor<64x64xf32>) outs(%arg11 : tensor<64x64xf32>) {
              ^bb0(%in: f32, %in_3: f32, %out: f32):
                %22 = arith.addf %in, %in_3 : f32
                linalg.yield %22 : f32
              } -> tensor<64x64xf32>
              scf.yield %21 : tensor<64x64xf32>
            }
            %15 = affine.apply affine_map<(d0, d1, d2, d3) -> (d0 * 2048 + d1 * 64 + d2 * 65536 + d3 * 32768)>(%arg7, %arg8, %arg6, %arg9)
            %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%15], sizes: [64, 64], strides: [512, 1] {loom.reuse = {spatial = [{depth = 0 : i64, iterator = "%arg6", mapped_to = "y", reuse_type = "no_reuse", volume = 0 : i64}, {depth = 1 : i64, iterator = "%arg7", mapped_to = "x", reuse_type = "no_reuse", volume = 0 : i64}], temporal = [{depth = 2 : i64, iterator = "%arg8", reuse_type = "no_reuse", volume = 0 : i64}, {depth = 3 : i64, iterator = "%arg9", reuse_type = "no_reuse", volume = 0 : i64}]}} : memref<*xf32> to memref<64x64xf32, strided<[512, 1], offset: ?>>
            bufferization.materialize_in_destination %14 in writable %reinterpret_cast : (tensor<64x64xf32>, memref<64x64xf32, strided<[512, 1], offset: ?>>) -> ()
          }
        }
      } {loom.mapped_to = "x"}
    } {loom.mapped_to = "y"}
    return
  }
  func.func @matmul_kernel__d1i0_d0i1__f10__hoist_block_1__c0mem_c1mem(%arg0: memref<*xf32> {tt.divisibility = 16 : i32}, %arg1: memref<*xf32> {tt.divisibility = 16 : i32}, %arg2: memref<*xf32> {tt.divisibility = 16 : i32}, %arg3: index, %arg4: index, %arg5: index) {
    affine.parallel (%arg6) = (0) to (2) {
      affine.parallel (%arg7) = (0) to (32) {
        affine.for %arg8 = 0 to affine_map<()[s0, s1] -> (s1 ceildiv 32)>()[%arg3, %arg4] {
          %12 = affine.apply affine_map<(d0, d1) -> (d0 * 2048 + d1 * 64)>(%arg7, %arg8)
          %reinterpret_cast = memref.reinterpret_cast %arg1 to offset: [%12], sizes: [8, 64, 64], strides: [32768, 512, 1] {loom.reuse = {spatial = [{depth = 0 : i64, iterator = "%arg6", mapped_to = "y", reuse_type = "total_reuse", volume = 131072 : i64}, {depth = 1 : i64, iterator = "%arg7", mapped_to = "x", reuse_type = "no_reuse", volume = 0 : i64}], temporal = [{depth = 2 : i64, iterator = "%arg8", reuse_type = "no_reuse", volume = 0 : i64}]}} : memref<*xf32> to memref<8x64x64xf32, strided<[32768, 512, 1], offset: ?>>
          %alloc = memref.alloc() {loom.alloc = {local = true, memory_name = "L1", size = 131072 : i64}} : memref<8x64x64xf32>
          memref.copy %reinterpret_cast, %alloc {loom.copy.choice = {kind = "mem", memory_name = "L1"}} : memref<8x64x64xf32, strided<[32768, 512, 1], offset: ?>> to memref<8x64x64xf32>
          affine.for %arg9 = 0 to affine_map<()[s0, s1] -> (s0 ceildiv 2)>()[%arg3, %arg4] {
            %cst = arith.constant 0.000000e+00 : f32
            %13 = tensor.empty() : tensor<64x64xf32>
            %14 = linalg.fill ins(%cst : f32) outs(%13 : tensor<64x64xf32>) -> tensor<64x64xf32>
            %c0 = arith.constant 0 : index
            %c8 = arith.constant 8 : index
            %c1 = arith.constant 1 : index
            %15 = scf.for %arg10 = %c0 to %c8 step %c1 iter_args(%arg11 = %14) -> (tensor<64x64xf32>) {
              %17 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 64 + d1 * 65536 + d2 * 32768)>(%arg10, %arg6, %arg9)
              %reinterpret_cast_1 = memref.reinterpret_cast %arg0 to offset: [%17], sizes: [64, 64], strides: [512, 1] {loom.reuse = {sequential = [{depth = 4 : i64, iterator = "%arg10", reuse_type = "no_reuse", volume = 0 : i64}], spatial = [{depth = 0 : i64, iterator = "%arg6", mapped_to = "y", reuse_type = "no_reuse", volume = 0 : i64}, {depth = 1 : i64, iterator = "%arg7", mapped_to = "x", reuse_type = "total_reuse", volume = 16384 : i64}], temporal = [{depth = 2 : i64, iterator = "%arg8", reuse_type = "total_reuse", volume = 16384 : i64}, {depth = 3 : i64, iterator = "%arg9", reuse_type = "no_reuse", volume = 0 : i64}]}} : memref<*xf32> to memref<64x64xf32, strided<[512, 1], offset: ?>>
              %alloc_2 = memref.alloc() {loom.alloc = {local = true, memory_name = "L1", size = 16384 : i64}} : memref<64x64xf32>
              memref.copy %reinterpret_cast_1, %alloc_2 {loom.copy.choice = {kind = "mem", memory_name = "L1"}} : memref<64x64xf32, strided<[512, 1], offset: ?>> to memref<64x64xf32>
              %18 = bufferization.to_tensor %alloc_2 restrict writable : memref<64x64xf32> to tensor<64x64xf32>
              %subview = memref.subview %alloc[%arg10, 0, 0] [1, 64, 64] [1, 1, 1] : memref<8x64x64xf32> to memref<64x64xf32, strided<[64, 1], offset: ?>>
              %19 = bufferization.to_tensor %subview restrict writable : memref<64x64xf32, strided<[64, 1], offset: ?>> to tensor<64x64xf32>
              %20 = linalg.matmul ins(%18, %19 : tensor<64x64xf32>, tensor<64x64xf32>) outs(%14 : tensor<64x64xf32>) -> tensor<64x64xf32>
              %21 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%arg11, %20 : tensor<64x64xf32>, tensor<64x64xf32>) outs(%arg11 : tensor<64x64xf32>) {
              ^bb0(%in: f32, %in_3: f32, %out: f32):
                %22 = arith.addf %in, %in_3 : f32
                linalg.yield %22 : f32
              } -> tensor<64x64xf32>
              scf.yield %21 : tensor<64x64xf32>
            }
            %16 = affine.apply affine_map<(d0, d1, d2, d3) -> (d0 * 2048 + d1 * 64 + d2 * 65536 + d3 * 32768)>(%arg7, %arg8, %arg6, %arg9)
            %reinterpret_cast_0 = memref.reinterpret_cast %arg2 to offset: [%16], sizes: [64, 64], strides: [512, 1] {loom.reuse = {spatial = [{depth = 0 : i64, iterator = "%arg6", mapped_to = "y", reuse_type = "no_reuse", volume = 0 : i64}, {depth = 1 : i64, iterator = "%arg7", mapped_to = "x", reuse_type = "no_reuse", volume = 0 : i64}], temporal = [{depth = 2 : i64, iterator = "%arg8", reuse_type = "no_reuse", volume = 0 : i64}, {depth = 3 : i64, iterator = "%arg9", reuse_type = "no_reuse", volume = 0 : i64}]}} : memref<*xf32> to memref<64x64xf32, strided<[512, 1], offset: ?>>
            bufferization.materialize_in_destination %15 in writable %reinterpret_cast_0 : (tensor<64x64xf32>, memref<64x64xf32, strided<[512, 1], offset: ?>>) -> ()
          }
        }
      } {loom.mapped_to = "x"}
    } {loom.mapped_to = "y"}
    return
  }
  func.func @matmul_kernel__d1i0_d0i1__f10__hoist_block_1__c0mem_c1bx(%arg0: memref<*xf32> {tt.divisibility = 16 : i32}, %arg1: memref<*xf32> {tt.divisibility = 16 : i32}, %arg2: memref<*xf32> {tt.divisibility = 16 : i32}, %arg3: index, %arg4: index, %arg5: index) {
    affine.parallel (%arg6) = (0) to (2) {
      affine.parallel (%arg7) = (0) to (32) {
        affine.for %arg8 = 0 to affine_map<()[s0, s1] -> (s1 ceildiv 32)>()[%arg3, %arg4] {
          %12 = affine.apply affine_map<(d0, d1) -> (d0 * 2048 + d1 * 64)>(%arg7, %arg8)
          %reinterpret_cast = memref.reinterpret_cast %arg1 to offset: [%12], sizes: [8, 64, 64], strides: [32768, 512, 1] {loom.reuse = {spatial = [{depth = 0 : i64, iterator = "%arg6", mapped_to = "y", reuse_type = "total_reuse", volume = 131072 : i64}, {depth = 1 : i64, iterator = "%arg7", mapped_to = "x", reuse_type = "no_reuse", volume = 0 : i64}], temporal = [{depth = 2 : i64, iterator = "%arg8", reuse_type = "no_reuse", volume = 0 : i64}]}} : memref<*xf32> to memref<8x64x64xf32, strided<[32768, 512, 1], offset: ?>>
          %alloc = memref.alloc() {loom.alloc = {local = true, memory_name = "L1", size = 131072 : i64}} : memref<8x64x64xf32>
          memref.copy %reinterpret_cast, %alloc {loom.copy.choice = {kind = "mem", memory_name = "L1"}} : memref<8x64x64xf32, strided<[32768, 512, 1], offset: ?>> to memref<8x64x64xf32>
          affine.for %arg9 = 0 to affine_map<()[s0, s1] -> (s0 ceildiv 2)>()[%arg3, %arg4] {
            %cst = arith.constant 0.000000e+00 : f32
            %13 = tensor.empty() : tensor<64x64xf32>
            %14 = linalg.fill ins(%cst : f32) outs(%13 : tensor<64x64xf32>) -> tensor<64x64xf32>
            %c0 = arith.constant 0 : index
            %c8 = arith.constant 8 : index
            %c1 = arith.constant 1 : index
            %15 = scf.for %arg10 = %c0 to %c8 step %c1 iter_args(%arg11 = %14) -> (tensor<64x64xf32>) {
              %17 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 64 + d1 * 65536 + d2 * 32768)>(%arg10, %arg6, %arg9)
              %reinterpret_cast_1 = memref.reinterpret_cast %arg0 to offset: [%17], sizes: [64, 64], strides: [512, 1] {loom.reuse = {sequential = [{depth = 4 : i64, iterator = "%arg10", reuse_type = "no_reuse", volume = 0 : i64}], spatial = [{depth = 0 : i64, iterator = "%arg6", mapped_to = "y", reuse_type = "no_reuse", volume = 0 : i64}, {depth = 1 : i64, iterator = "%arg7", mapped_to = "x", reuse_type = "total_reuse", volume = 16384 : i64}], temporal = [{depth = 2 : i64, iterator = "%arg8", reuse_type = "total_reuse", volume = 16384 : i64}, {depth = 3 : i64, iterator = "%arg9", reuse_type = "no_reuse", volume = 0 : i64}]}} : memref<*xf32> to memref<64x64xf32, strided<[512, 1], offset: ?>>
              %alloc_2 = memref.alloc() {loom.alloc = {local = true, memory_name = "L1", size = 16384 : i64}} : memref<64x64xf32>
              memref.copy %reinterpret_cast_1, %alloc_2 {loom.copy.choice = {dim = "x", interconnect_name = "big_ring", kind = "broadcast"}} : memref<64x64xf32, strided<[512, 1], offset: ?>> to memref<64x64xf32>
              %18 = bufferization.to_tensor %alloc_2 restrict writable : memref<64x64xf32> to tensor<64x64xf32>
              %subview = memref.subview %alloc[%arg10, 0, 0] [1, 64, 64] [1, 1, 1] : memref<8x64x64xf32> to memref<64x64xf32, strided<[64, 1], offset: ?>>
              %19 = bufferization.to_tensor %subview restrict writable : memref<64x64xf32, strided<[64, 1], offset: ?>> to tensor<64x64xf32>
              %20 = linalg.matmul ins(%18, %19 : tensor<64x64xf32>, tensor<64x64xf32>) outs(%14 : tensor<64x64xf32>) -> tensor<64x64xf32>
              %21 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%arg11, %20 : tensor<64x64xf32>, tensor<64x64xf32>) outs(%arg11 : tensor<64x64xf32>) {
              ^bb0(%in: f32, %in_3: f32, %out: f32):
                %22 = arith.addf %in, %in_3 : f32
                linalg.yield %22 : f32
              } -> tensor<64x64xf32>
              scf.yield %21 : tensor<64x64xf32>
            }
            %16 = affine.apply affine_map<(d0, d1, d2, d3) -> (d0 * 2048 + d1 * 64 + d2 * 65536 + d3 * 32768)>(%arg7, %arg8, %arg6, %arg9)
            %reinterpret_cast_0 = memref.reinterpret_cast %arg2 to offset: [%16], sizes: [64, 64], strides: [512, 1] {loom.reuse = {spatial = [{depth = 0 : i64, iterator = "%arg6", mapped_to = "y", reuse_type = "no_reuse", volume = 0 : i64}, {depth = 1 : i64, iterator = "%arg7", mapped_to = "x", reuse_type = "no_reuse", volume = 0 : i64}], temporal = [{depth = 2 : i64, iterator = "%arg8", reuse_type = "no_reuse", volume = 0 : i64}, {depth = 3 : i64, iterator = "%arg9", reuse_type = "no_reuse", volume = 0 : i64}]}} : memref<*xf32> to memref<64x64xf32, strided<[512, 1], offset: ?>>
            bufferization.materialize_in_destination %15 in writable %reinterpret_cast_0 : (tensor<64x64xf32>, memref<64x64xf32, strided<[512, 1], offset: ?>>) -> ()
          }
        }
      } {loom.mapped_to = "x"}
    } {loom.mapped_to = "y"}
    return
  }
  func.func @matmul_kernel__d0i1_d1i1__f01__c0mem_c1mem(%arg0: memref<*xf32> {tt.divisibility = 16 : i32}, %arg1: memref<*xf32> {tt.divisibility = 16 : i32}, %arg2: memref<*xf32> {tt.divisibility = 16 : i32}, %arg3: index, %arg4: index, %arg5: index) {
    affine.parallel (%arg6) = (0) to (32) {
      affine.parallel (%arg7) = (0) to (2) {
        affine.for %arg8 = 0 to affine_map<()[s0, s1] -> (s0)>()[%arg3, %arg4] {
          affine.for %arg9 = 0 to affine_map<()[s0, s1] -> ((s1 ceildiv 32) ceildiv 2)>()[%arg3, %arg4] {
            %cst = arith.constant 0.000000e+00 : f32
            %12 = tensor.empty() : tensor<64x64xf32>
            %13 = linalg.fill ins(%cst : f32) outs(%12 : tensor<64x64xf32>) -> tensor<64x64xf32>
            %c0 = arith.constant 0 : index
            %c8 = arith.constant 8 : index
            %c1 = arith.constant 1 : index
            %14 = scf.for %arg10 = %c0 to %c8 step %c1 iter_args(%arg11 = %13) -> (tensor<64x64xf32>) {
              %16 = affine.apply affine_map<(d0, d1) -> (d1 * 32768 + d0 * 64)>(%arg10, %arg8)
              %reinterpret_cast_0 = memref.reinterpret_cast %arg0 to offset: [%16], sizes: [64, 64], strides: [512, 1] {loom.reuse = {sequential = [{depth = 4 : i64, iterator = "%arg10", reuse_type = "no_reuse", volume = 0 : i64}], spatial = [{depth = 0 : i64, iterator = "%arg6", mapped_to = "x", reuse_type = "total_reuse", volume = 16384 : i64}, {depth = 1 : i64, iterator = "%arg7", mapped_to = "y", reuse_type = "total_reuse", volume = 16384 : i64}], temporal = [{depth = 2 : i64, iterator = "%arg8", reuse_type = "no_reuse", volume = 0 : i64}, {depth = 3 : i64, iterator = "%arg9", reuse_type = "total_reuse", volume = 16384 : i64}]}} : memref<*xf32> to memref<64x64xf32, strided<[512, 1], offset: ?>>
              %alloc = memref.alloc() {loom.alloc = {local = true, memory_name = "L1", size = 16384 : i64}} : memref<64x64xf32>
              memref.copy %reinterpret_cast_0, %alloc {loom.copy.choice = {kind = "mem", memory_name = "L1"}} : memref<64x64xf32, strided<[512, 1], offset: ?>> to memref<64x64xf32>
              %17 = bufferization.to_tensor %alloc restrict writable : memref<64x64xf32> to tensor<64x64xf32>
              %18 = affine.apply affine_map<(d0, d1, d2, d3) -> (d0 * 32768 + d1 * 2048 + d2 * 128 + d3 * 64)>(%arg10, %arg6, %arg7, %arg9)
              %reinterpret_cast_1 = memref.reinterpret_cast %arg1 to offset: [%18], sizes: [64, 64], strides: [512, 1] {loom.reuse = {sequential = [{depth = 4 : i64, iterator = "%arg10", reuse_type = "no_reuse", volume = 0 : i64}], spatial = [{depth = 0 : i64, iterator = "%arg6", mapped_to = "x", reuse_type = "no_reuse", volume = 0 : i64}, {depth = 1 : i64, iterator = "%arg7", mapped_to = "y", reuse_type = "no_reuse", volume = 0 : i64}], temporal = [{depth = 2 : i64, iterator = "%arg8", reuse_type = "total_reuse", volume = 16384 : i64}, {depth = 3 : i64, iterator = "%arg9", reuse_type = "no_reuse", volume = 0 : i64}]}} : memref<*xf32> to memref<64x64xf32, strided<[512, 1], offset: ?>>
              %alloc_2 = memref.alloc() {loom.alloc = {local = true, memory_name = "L1", size = 16384 : i64}} : memref<64x64xf32>
              memref.copy %reinterpret_cast_1, %alloc_2 {loom.copy.choice = {kind = "mem", memory_name = "L1"}} : memref<64x64xf32, strided<[512, 1], offset: ?>> to memref<64x64xf32>
              %19 = bufferization.to_tensor %alloc_2 restrict writable : memref<64x64xf32> to tensor<64x64xf32>
              %20 = linalg.matmul ins(%17, %19 : tensor<64x64xf32>, tensor<64x64xf32>) outs(%13 : tensor<64x64xf32>) -> tensor<64x64xf32>
              %21 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%arg11, %20 : tensor<64x64xf32>, tensor<64x64xf32>) outs(%arg11 : tensor<64x64xf32>) {
              ^bb0(%in: f32, %in_3: f32, %out: f32):
                %22 = arith.addf %in, %in_3 : f32
                linalg.yield %22 : f32
              } -> tensor<64x64xf32>
              scf.yield %21 : tensor<64x64xf32>
            }
            %15 = affine.apply affine_map<(d0, d1, d2, d3) -> (d0 * 32768 + d1 * 2048 + d2 * 128 + d3 * 64)>(%arg8, %arg6, %arg7, %arg9)
            %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%15], sizes: [64, 64], strides: [512, 1] {loom.reuse = {spatial = [{depth = 0 : i64, iterator = "%arg6", mapped_to = "x", reuse_type = "no_reuse", volume = 0 : i64}, {depth = 1 : i64, iterator = "%arg7", mapped_to = "y", reuse_type = "no_reuse", volume = 0 : i64}], temporal = [{depth = 2 : i64, iterator = "%arg8", reuse_type = "no_reuse", volume = 0 : i64}, {depth = 3 : i64, iterator = "%arg9", reuse_type = "no_reuse", volume = 0 : i64}]}} : memref<*xf32> to memref<64x64xf32, strided<[512, 1], offset: ?>>
            bufferization.materialize_in_destination %14 in writable %reinterpret_cast : (tensor<64x64xf32>, memref<64x64xf32, strided<[512, 1], offset: ?>>) -> ()
          }
        }
      } {loom.mapped_to = "y"}
    } {loom.mapped_to = "x"}
    return
  }
  func.func @matmul_kernel__d0i1_d1i1__f01__c0bx_c1mem(%arg0: memref<*xf32> {tt.divisibility = 16 : i32}, %arg1: memref<*xf32> {tt.divisibility = 16 : i32}, %arg2: memref<*xf32> {tt.divisibility = 16 : i32}, %arg3: index, %arg4: index, %arg5: index) {
    affine.parallel (%arg6) = (0) to (32) {
      affine.parallel (%arg7) = (0) to (2) {
        affine.for %arg8 = 0 to affine_map<()[s0, s1] -> (s0)>()[%arg3, %arg4] {
          affine.for %arg9 = 0 to affine_map<()[s0, s1] -> ((s1 ceildiv 32) ceildiv 2)>()[%arg3, %arg4] {
            %cst = arith.constant 0.000000e+00 : f32
            %12 = tensor.empty() : tensor<64x64xf32>
            %13 = linalg.fill ins(%cst : f32) outs(%12 : tensor<64x64xf32>) -> tensor<64x64xf32>
            %c0 = arith.constant 0 : index
            %c8 = arith.constant 8 : index
            %c1 = arith.constant 1 : index
            %14 = scf.for %arg10 = %c0 to %c8 step %c1 iter_args(%arg11 = %13) -> (tensor<64x64xf32>) {
              %16 = affine.apply affine_map<(d0, d1) -> (d1 * 32768 + d0 * 64)>(%arg10, %arg8)
              %reinterpret_cast_0 = memref.reinterpret_cast %arg0 to offset: [%16], sizes: [64, 64], strides: [512, 1] {loom.reuse = {sequential = [{depth = 4 : i64, iterator = "%arg10", reuse_type = "no_reuse", volume = 0 : i64}], spatial = [{depth = 0 : i64, iterator = "%arg6", mapped_to = "x", reuse_type = "total_reuse", volume = 16384 : i64}, {depth = 1 : i64, iterator = "%arg7", mapped_to = "y", reuse_type = "total_reuse", volume = 16384 : i64}], temporal = [{depth = 2 : i64, iterator = "%arg8", reuse_type = "no_reuse", volume = 0 : i64}, {depth = 3 : i64, iterator = "%arg9", reuse_type = "total_reuse", volume = 16384 : i64}]}} : memref<*xf32> to memref<64x64xf32, strided<[512, 1], offset: ?>>
              %alloc = memref.alloc() {loom.alloc = {local = true, memory_name = "L1", size = 16384 : i64}} : memref<64x64xf32>
              memref.copy %reinterpret_cast_0, %alloc {loom.copy.choice = {dim = "x", interconnect_name = "big_ring", kind = "broadcast"}} : memref<64x64xf32, strided<[512, 1], offset: ?>> to memref<64x64xf32>
              %17 = bufferization.to_tensor %alloc restrict writable : memref<64x64xf32> to tensor<64x64xf32>
              %18 = affine.apply affine_map<(d0, d1, d2, d3) -> (d0 * 32768 + d1 * 2048 + d2 * 128 + d3 * 64)>(%arg10, %arg6, %arg7, %arg9)
              %reinterpret_cast_1 = memref.reinterpret_cast %arg1 to offset: [%18], sizes: [64, 64], strides: [512, 1] {loom.reuse = {sequential = [{depth = 4 : i64, iterator = "%arg10", reuse_type = "no_reuse", volume = 0 : i64}], spatial = [{depth = 0 : i64, iterator = "%arg6", mapped_to = "x", reuse_type = "no_reuse", volume = 0 : i64}, {depth = 1 : i64, iterator = "%arg7", mapped_to = "y", reuse_type = "no_reuse", volume = 0 : i64}], temporal = [{depth = 2 : i64, iterator = "%arg8", reuse_type = "total_reuse", volume = 16384 : i64}, {depth = 3 : i64, iterator = "%arg9", reuse_type = "no_reuse", volume = 0 : i64}]}} : memref<*xf32> to memref<64x64xf32, strided<[512, 1], offset: ?>>
              %alloc_2 = memref.alloc() {loom.alloc = {local = true, memory_name = "L1", size = 16384 : i64}} : memref<64x64xf32>
              memref.copy %reinterpret_cast_1, %alloc_2 {loom.copy.choice = {kind = "mem", memory_name = "L1"}} : memref<64x64xf32, strided<[512, 1], offset: ?>> to memref<64x64xf32>
              %19 = bufferization.to_tensor %alloc_2 restrict writable : memref<64x64xf32> to tensor<64x64xf32>
              %20 = linalg.matmul ins(%17, %19 : tensor<64x64xf32>, tensor<64x64xf32>) outs(%13 : tensor<64x64xf32>) -> tensor<64x64xf32>
              %21 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%arg11, %20 : tensor<64x64xf32>, tensor<64x64xf32>) outs(%arg11 : tensor<64x64xf32>) {
              ^bb0(%in: f32, %in_3: f32, %out: f32):
                %22 = arith.addf %in, %in_3 : f32
                linalg.yield %22 : f32
              } -> tensor<64x64xf32>
              scf.yield %21 : tensor<64x64xf32>
            }
            %15 = affine.apply affine_map<(d0, d1, d2, d3) -> (d0 * 32768 + d1 * 2048 + d2 * 128 + d3 * 64)>(%arg8, %arg6, %arg7, %arg9)
            %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%15], sizes: [64, 64], strides: [512, 1] {loom.reuse = {spatial = [{depth = 0 : i64, iterator = "%arg6", mapped_to = "x", reuse_type = "no_reuse", volume = 0 : i64}, {depth = 1 : i64, iterator = "%arg7", mapped_to = "y", reuse_type = "no_reuse", volume = 0 : i64}], temporal = [{depth = 2 : i64, iterator = "%arg8", reuse_type = "no_reuse", volume = 0 : i64}, {depth = 3 : i64, iterator = "%arg9", reuse_type = "no_reuse", volume = 0 : i64}]}} : memref<*xf32> to memref<64x64xf32, strided<[512, 1], offset: ?>>
            bufferization.materialize_in_destination %14 in writable %reinterpret_cast : (tensor<64x64xf32>, memref<64x64xf32, strided<[512, 1], offset: ?>>) -> ()
          }
        }
      } {loom.mapped_to = "y"}
    } {loom.mapped_to = "x"}
    return
  }
  func.func @matmul_kernel__d0i1_d1i1__f01__hoist_block_0__c0mem_c1mem(%arg0: memref<*xf32> {tt.divisibility = 16 : i32}, %arg1: memref<*xf32> {tt.divisibility = 16 : i32}, %arg2: memref<*xf32> {tt.divisibility = 16 : i32}, %arg3: index, %arg4: index, %arg5: index) {
    affine.parallel (%arg6) = (0) to (32) {
      affine.parallel (%arg7) = (0) to (2) {
        affine.for %arg8 = 0 to affine_map<()[s0, s1] -> (s0)>()[%arg3, %arg4] {
          %12 = affine.apply affine_map<(d0) -> (d0 * 32768)>(%arg8)
          %reinterpret_cast = memref.reinterpret_cast %arg0 to offset: [%12], sizes: [8, 64, 64], strides: [64, 512, 1] {loom.reuse = {spatial = [{depth = 0 : i64, iterator = "%arg6", mapped_to = "x", reuse_type = "total_reuse", volume = 131072 : i64}, {depth = 1 : i64, iterator = "%arg7", mapped_to = "y", reuse_type = "total_reuse", volume = 131072 : i64}], temporal = [{depth = 2 : i64, iterator = "%arg8", reuse_type = "no_reuse", volume = 0 : i64}]}} : memref<*xf32> to memref<8x64x64xf32, strided<[64, 512, 1], offset: ?>>
          %alloc = memref.alloc() {loom.alloc = {local = true, memory_name = "L1", size = 131072 : i64}} : memref<8x64x64xf32>
          memref.copy %reinterpret_cast, %alloc {loom.copy.choice = {kind = "mem", memory_name = "L1"}} : memref<8x64x64xf32, strided<[64, 512, 1], offset: ?>> to memref<8x64x64xf32>
          affine.for %arg9 = 0 to affine_map<()[s0, s1] -> ((s1 ceildiv 32) ceildiv 2)>()[%arg3, %arg4] {
            %cst = arith.constant 0.000000e+00 : f32
            %13 = tensor.empty() : tensor<64x64xf32>
            %14 = linalg.fill ins(%cst : f32) outs(%13 : tensor<64x64xf32>) -> tensor<64x64xf32>
            %c0 = arith.constant 0 : index
            %c8 = arith.constant 8 : index
            %c1 = arith.constant 1 : index
            %15 = scf.for %arg10 = %c0 to %c8 step %c1 iter_args(%arg11 = %14) -> (tensor<64x64xf32>) {
              %subview = memref.subview %alloc[%arg10, 0, 0] [1, 64, 64] [1, 1, 1] : memref<8x64x64xf32> to memref<64x64xf32, strided<[64, 1], offset: ?>>
              %17 = bufferization.to_tensor %subview restrict writable : memref<64x64xf32, strided<[64, 1], offset: ?>> to tensor<64x64xf32>
              %18 = affine.apply affine_map<(d0, d1, d2, d3) -> (d0 * 32768 + d1 * 2048 + d2 * 128 + d3 * 64)>(%arg10, %arg6, %arg7, %arg9)
              %reinterpret_cast_1 = memref.reinterpret_cast %arg1 to offset: [%18], sizes: [64, 64], strides: [512, 1] {loom.reuse = {sequential = [{depth = 4 : i64, iterator = "%arg10", reuse_type = "no_reuse", volume = 0 : i64}], spatial = [{depth = 0 : i64, iterator = "%arg6", mapped_to = "x", reuse_type = "no_reuse", volume = 0 : i64}, {depth = 1 : i64, iterator = "%arg7", mapped_to = "y", reuse_type = "no_reuse", volume = 0 : i64}], temporal = [{depth = 2 : i64, iterator = "%arg8", reuse_type = "total_reuse", volume = 16384 : i64}, {depth = 3 : i64, iterator = "%arg9", reuse_type = "no_reuse", volume = 0 : i64}]}} : memref<*xf32> to memref<64x64xf32, strided<[512, 1], offset: ?>>
              %alloc_2 = memref.alloc() {loom.alloc = {local = true, memory_name = "L1", size = 16384 : i64}} : memref<64x64xf32>
              memref.copy %reinterpret_cast_1, %alloc_2 {loom.copy.choice = {kind = "mem", memory_name = "L1"}} : memref<64x64xf32, strided<[512, 1], offset: ?>> to memref<64x64xf32>
              %19 = bufferization.to_tensor %alloc_2 restrict writable : memref<64x64xf32> to tensor<64x64xf32>
              %20 = linalg.matmul ins(%17, %19 : tensor<64x64xf32>, tensor<64x64xf32>) outs(%14 : tensor<64x64xf32>) -> tensor<64x64xf32>
              %21 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%arg11, %20 : tensor<64x64xf32>, tensor<64x64xf32>) outs(%arg11 : tensor<64x64xf32>) {
              ^bb0(%in: f32, %in_3: f32, %out: f32):
                %22 = arith.addf %in, %in_3 : f32
                linalg.yield %22 : f32
              } -> tensor<64x64xf32>
              scf.yield %21 : tensor<64x64xf32>
            }
            %16 = affine.apply affine_map<(d0, d1, d2, d3) -> (d0 * 32768 + d1 * 2048 + d2 * 128 + d3 * 64)>(%arg8, %arg6, %arg7, %arg9)
            %reinterpret_cast_0 = memref.reinterpret_cast %arg2 to offset: [%16], sizes: [64, 64], strides: [512, 1] {loom.reuse = {spatial = [{depth = 0 : i64, iterator = "%arg6", mapped_to = "x", reuse_type = "no_reuse", volume = 0 : i64}, {depth = 1 : i64, iterator = "%arg7", mapped_to = "y", reuse_type = "no_reuse", volume = 0 : i64}], temporal = [{depth = 2 : i64, iterator = "%arg8", reuse_type = "no_reuse", volume = 0 : i64}, {depth = 3 : i64, iterator = "%arg9", reuse_type = "no_reuse", volume = 0 : i64}]}} : memref<*xf32> to memref<64x64xf32, strided<[512, 1], offset: ?>>
            bufferization.materialize_in_destination %15 in writable %reinterpret_cast_0 : (tensor<64x64xf32>, memref<64x64xf32, strided<[512, 1], offset: ?>>) -> ()
          }
        }
      } {loom.mapped_to = "y"}
    } {loom.mapped_to = "x"}
    return
  }
  func.func @matmul_kernel__d0i1_d1i1__f01__hoist_block_0__c0bx_c1mem(%arg0: memref<*xf32> {tt.divisibility = 16 : i32}, %arg1: memref<*xf32> {tt.divisibility = 16 : i32}, %arg2: memref<*xf32> {tt.divisibility = 16 : i32}, %arg3: index, %arg4: index, %arg5: index) {
    affine.parallel (%arg6) = (0) to (32) {
      affine.parallel (%arg7) = (0) to (2) {
        affine.for %arg8 = 0 to affine_map<()[s0, s1] -> (s0)>()[%arg3, %arg4] {
          %12 = affine.apply affine_map<(d0) -> (d0 * 32768)>(%arg8)
          %reinterpret_cast = memref.reinterpret_cast %arg0 to offset: [%12], sizes: [8, 64, 64], strides: [64, 512, 1] {loom.reuse = {spatial = [{depth = 0 : i64, iterator = "%arg6", mapped_to = "x", reuse_type = "total_reuse", volume = 131072 : i64}, {depth = 1 : i64, iterator = "%arg7", mapped_to = "y", reuse_type = "total_reuse", volume = 131072 : i64}], temporal = [{depth = 2 : i64, iterator = "%arg8", reuse_type = "no_reuse", volume = 0 : i64}]}} : memref<*xf32> to memref<8x64x64xf32, strided<[64, 512, 1], offset: ?>>
          %alloc = memref.alloc() {loom.alloc = {local = true, memory_name = "L1", size = 131072 : i64}} : memref<8x64x64xf32>
          memref.copy %reinterpret_cast, %alloc {loom.copy.choice = {dim = "x", interconnect_name = "big_ring", kind = "broadcast"}} : memref<8x64x64xf32, strided<[64, 512, 1], offset: ?>> to memref<8x64x64xf32>
          affine.for %arg9 = 0 to affine_map<()[s0, s1] -> ((s1 ceildiv 32) ceildiv 2)>()[%arg3, %arg4] {
            %cst = arith.constant 0.000000e+00 : f32
            %13 = tensor.empty() : tensor<64x64xf32>
            %14 = linalg.fill ins(%cst : f32) outs(%13 : tensor<64x64xf32>) -> tensor<64x64xf32>
            %c0 = arith.constant 0 : index
            %c8 = arith.constant 8 : index
            %c1 = arith.constant 1 : index
            %15 = scf.for %arg10 = %c0 to %c8 step %c1 iter_args(%arg11 = %14) -> (tensor<64x64xf32>) {
              %subview = memref.subview %alloc[%arg10, 0, 0] [1, 64, 64] [1, 1, 1] : memref<8x64x64xf32> to memref<64x64xf32, strided<[64, 1], offset: ?>>
              %17 = bufferization.to_tensor %subview restrict writable : memref<64x64xf32, strided<[64, 1], offset: ?>> to tensor<64x64xf32>
              %18 = affine.apply affine_map<(d0, d1, d2, d3) -> (d0 * 32768 + d1 * 2048 + d2 * 128 + d3 * 64)>(%arg10, %arg6, %arg7, %arg9)
              %reinterpret_cast_1 = memref.reinterpret_cast %arg1 to offset: [%18], sizes: [64, 64], strides: [512, 1] {loom.reuse = {sequential = [{depth = 4 : i64, iterator = "%arg10", reuse_type = "no_reuse", volume = 0 : i64}], spatial = [{depth = 0 : i64, iterator = "%arg6", mapped_to = "x", reuse_type = "no_reuse", volume = 0 : i64}, {depth = 1 : i64, iterator = "%arg7", mapped_to = "y", reuse_type = "no_reuse", volume = 0 : i64}], temporal = [{depth = 2 : i64, iterator = "%arg8", reuse_type = "total_reuse", volume = 16384 : i64}, {depth = 3 : i64, iterator = "%arg9", reuse_type = "no_reuse", volume = 0 : i64}]}} : memref<*xf32> to memref<64x64xf32, strided<[512, 1], offset: ?>>
              %alloc_2 = memref.alloc() {loom.alloc = {local = true, memory_name = "L1", size = 16384 : i64}} : memref<64x64xf32>
              memref.copy %reinterpret_cast_1, %alloc_2 {loom.copy.choice = {kind = "mem", memory_name = "L1"}} : memref<64x64xf32, strided<[512, 1], offset: ?>> to memref<64x64xf32>
              %19 = bufferization.to_tensor %alloc_2 restrict writable : memref<64x64xf32> to tensor<64x64xf32>
              %20 = linalg.matmul ins(%17, %19 : tensor<64x64xf32>, tensor<64x64xf32>) outs(%14 : tensor<64x64xf32>) -> tensor<64x64xf32>
              %21 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%arg11, %20 : tensor<64x64xf32>, tensor<64x64xf32>) outs(%arg11 : tensor<64x64xf32>) {
              ^bb0(%in: f32, %in_3: f32, %out: f32):
                %22 = arith.addf %in, %in_3 : f32
                linalg.yield %22 : f32
              } -> tensor<64x64xf32>
              scf.yield %21 : tensor<64x64xf32>
            }
            %16 = affine.apply affine_map<(d0, d1, d2, d3) -> (d0 * 32768 + d1 * 2048 + d2 * 128 + d3 * 64)>(%arg8, %arg6, %arg7, %arg9)
            %reinterpret_cast_0 = memref.reinterpret_cast %arg2 to offset: [%16], sizes: [64, 64], strides: [512, 1] {loom.reuse = {spatial = [{depth = 0 : i64, iterator = "%arg6", mapped_to = "x", reuse_type = "no_reuse", volume = 0 : i64}, {depth = 1 : i64, iterator = "%arg7", mapped_to = "y", reuse_type = "no_reuse", volume = 0 : i64}], temporal = [{depth = 2 : i64, iterator = "%arg8", reuse_type = "no_reuse", volume = 0 : i64}, {depth = 3 : i64, iterator = "%arg9", reuse_type = "no_reuse", volume = 0 : i64}]}} : memref<*xf32> to memref<64x64xf32, strided<[512, 1], offset: ?>>
            bufferization.materialize_in_destination %15 in writable %reinterpret_cast_0 : (tensor<64x64xf32>, memref<64x64xf32, strided<[512, 1], offset: ?>>) -> ()
          }
        }
      } {loom.mapped_to = "y"}
    } {loom.mapped_to = "x"}
    return
  }
  func.func @matmul_kernel__d0i1_d1i1__f10__c0mem_c1mem(%arg0: memref<*xf32> {tt.divisibility = 16 : i32}, %arg1: memref<*xf32> {tt.divisibility = 16 : i32}, %arg2: memref<*xf32> {tt.divisibility = 16 : i32}, %arg3: index, %arg4: index, %arg5: index) {
    affine.parallel (%arg6) = (0) to (32) {
      affine.parallel (%arg7) = (0) to (2) {
        affine.for %arg8 = 0 to affine_map<()[s0, s1] -> ((s1 ceildiv 32) ceildiv 2)>()[%arg3, %arg4] {
          affine.for %arg9 = 0 to affine_map<()[s0, s1] -> (s0)>()[%arg3, %arg4] {
            %cst = arith.constant 0.000000e+00 : f32
            %12 = tensor.empty() : tensor<64x64xf32>
            %13 = linalg.fill ins(%cst : f32) outs(%12 : tensor<64x64xf32>) -> tensor<64x64xf32>
            %c0 = arith.constant 0 : index
            %c8 = arith.constant 8 : index
            %c1 = arith.constant 1 : index
            %14 = scf.for %arg10 = %c0 to %c8 step %c1 iter_args(%arg11 = %13) -> (tensor<64x64xf32>) {
              %16 = affine.apply affine_map<(d0, d1) -> (d1 * 32768 + d0 * 64)>(%arg10, %arg9)
              %reinterpret_cast_0 = memref.reinterpret_cast %arg0 to offset: [%16], sizes: [64, 64], strides: [512, 1] {loom.reuse = {sequential = [{depth = 4 : i64, iterator = "%arg10", reuse_type = "no_reuse", volume = 0 : i64}], spatial = [{depth = 0 : i64, iterator = "%arg6", mapped_to = "x", reuse_type = "total_reuse", volume = 16384 : i64}, {depth = 1 : i64, iterator = "%arg7", mapped_to = "y", reuse_type = "total_reuse", volume = 16384 : i64}], temporal = [{depth = 2 : i64, iterator = "%arg8", reuse_type = "total_reuse", volume = 16384 : i64}, {depth = 3 : i64, iterator = "%arg9", reuse_type = "no_reuse", volume = 0 : i64}]}} : memref<*xf32> to memref<64x64xf32, strided<[512, 1], offset: ?>>
              %alloc = memref.alloc() {loom.alloc = {local = true, memory_name = "L1", size = 16384 : i64}} : memref<64x64xf32>
              memref.copy %reinterpret_cast_0, %alloc {loom.copy.choice = {kind = "mem", memory_name = "L1"}} : memref<64x64xf32, strided<[512, 1], offset: ?>> to memref<64x64xf32>
              %17 = bufferization.to_tensor %alloc restrict writable : memref<64x64xf32> to tensor<64x64xf32>
              %18 = affine.apply affine_map<(d0, d1, d2, d3) -> (d0 * 32768 + d1 * 2048 + d2 * 128 + d3 * 64)>(%arg10, %arg6, %arg7, %arg8)
              %reinterpret_cast_1 = memref.reinterpret_cast %arg1 to offset: [%18], sizes: [64, 64], strides: [512, 1] {loom.reuse = {sequential = [{depth = 4 : i64, iterator = "%arg10", reuse_type = "no_reuse", volume = 0 : i64}], spatial = [{depth = 0 : i64, iterator = "%arg6", mapped_to = "x", reuse_type = "no_reuse", volume = 0 : i64}, {depth = 1 : i64, iterator = "%arg7", mapped_to = "y", reuse_type = "no_reuse", volume = 0 : i64}], temporal = [{depth = 2 : i64, iterator = "%arg8", reuse_type = "no_reuse", volume = 0 : i64}, {depth = 3 : i64, iterator = "%arg9", reuse_type = "total_reuse", volume = 16384 : i64}]}} : memref<*xf32> to memref<64x64xf32, strided<[512, 1], offset: ?>>
              %alloc_2 = memref.alloc() {loom.alloc = {local = true, memory_name = "L1", size = 16384 : i64}} : memref<64x64xf32>
              memref.copy %reinterpret_cast_1, %alloc_2 {loom.copy.choice = {kind = "mem", memory_name = "L1"}} : memref<64x64xf32, strided<[512, 1], offset: ?>> to memref<64x64xf32>
              %19 = bufferization.to_tensor %alloc_2 restrict writable : memref<64x64xf32> to tensor<64x64xf32>
              %20 = linalg.matmul ins(%17, %19 : tensor<64x64xf32>, tensor<64x64xf32>) outs(%13 : tensor<64x64xf32>) -> tensor<64x64xf32>
              %21 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%arg11, %20 : tensor<64x64xf32>, tensor<64x64xf32>) outs(%arg11 : tensor<64x64xf32>) {
              ^bb0(%in: f32, %in_3: f32, %out: f32):
                %22 = arith.addf %in, %in_3 : f32
                linalg.yield %22 : f32
              } -> tensor<64x64xf32>
              scf.yield %21 : tensor<64x64xf32>
            }
            %15 = affine.apply affine_map<(d0, d1, d2, d3) -> (d0 * 32768 + d1 * 2048 + d2 * 128 + d3 * 64)>(%arg9, %arg6, %arg7, %arg8)
            %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%15], sizes: [64, 64], strides: [512, 1] {loom.reuse = {spatial = [{depth = 0 : i64, iterator = "%arg6", mapped_to = "x", reuse_type = "no_reuse", volume = 0 : i64}, {depth = 1 : i64, iterator = "%arg7", mapped_to = "y", reuse_type = "no_reuse", volume = 0 : i64}], temporal = [{depth = 2 : i64, iterator = "%arg8", reuse_type = "no_reuse", volume = 0 : i64}, {depth = 3 : i64, iterator = "%arg9", reuse_type = "no_reuse", volume = 0 : i64}]}} : memref<*xf32> to memref<64x64xf32, strided<[512, 1], offset: ?>>
            bufferization.materialize_in_destination %14 in writable %reinterpret_cast : (tensor<64x64xf32>, memref<64x64xf32, strided<[512, 1], offset: ?>>) -> ()
          }
        }
      } {loom.mapped_to = "y"}
    } {loom.mapped_to = "x"}
    return
  }
  func.func @matmul_kernel__d0i1_d1i1__f10__c0bx_c1mem(%arg0: memref<*xf32> {tt.divisibility = 16 : i32}, %arg1: memref<*xf32> {tt.divisibility = 16 : i32}, %arg2: memref<*xf32> {tt.divisibility = 16 : i32}, %arg3: index, %arg4: index, %arg5: index) {
    affine.parallel (%arg6) = (0) to (32) {
      affine.parallel (%arg7) = (0) to (2) {
        affine.for %arg8 = 0 to affine_map<()[s0, s1] -> ((s1 ceildiv 32) ceildiv 2)>()[%arg3, %arg4] {
          affine.for %arg9 = 0 to affine_map<()[s0, s1] -> (s0)>()[%arg3, %arg4] {
            %cst = arith.constant 0.000000e+00 : f32
            %12 = tensor.empty() : tensor<64x64xf32>
            %13 = linalg.fill ins(%cst : f32) outs(%12 : tensor<64x64xf32>) -> tensor<64x64xf32>
            %c0 = arith.constant 0 : index
            %c8 = arith.constant 8 : index
            %c1 = arith.constant 1 : index
            %14 = scf.for %arg10 = %c0 to %c8 step %c1 iter_args(%arg11 = %13) -> (tensor<64x64xf32>) {
              %16 = affine.apply affine_map<(d0, d1) -> (d1 * 32768 + d0 * 64)>(%arg10, %arg9)
              %reinterpret_cast_0 = memref.reinterpret_cast %arg0 to offset: [%16], sizes: [64, 64], strides: [512, 1] {loom.reuse = {sequential = [{depth = 4 : i64, iterator = "%arg10", reuse_type = "no_reuse", volume = 0 : i64}], spatial = [{depth = 0 : i64, iterator = "%arg6", mapped_to = "x", reuse_type = "total_reuse", volume = 16384 : i64}, {depth = 1 : i64, iterator = "%arg7", mapped_to = "y", reuse_type = "total_reuse", volume = 16384 : i64}], temporal = [{depth = 2 : i64, iterator = "%arg8", reuse_type = "total_reuse", volume = 16384 : i64}, {depth = 3 : i64, iterator = "%arg9", reuse_type = "no_reuse", volume = 0 : i64}]}} : memref<*xf32> to memref<64x64xf32, strided<[512, 1], offset: ?>>
              %alloc = memref.alloc() {loom.alloc = {local = true, memory_name = "L1", size = 16384 : i64}} : memref<64x64xf32>
              memref.copy %reinterpret_cast_0, %alloc {loom.copy.choice = {dim = "x", interconnect_name = "big_ring", kind = "broadcast"}} : memref<64x64xf32, strided<[512, 1], offset: ?>> to memref<64x64xf32>
              %17 = bufferization.to_tensor %alloc restrict writable : memref<64x64xf32> to tensor<64x64xf32>
              %18 = affine.apply affine_map<(d0, d1, d2, d3) -> (d0 * 32768 + d1 * 2048 + d2 * 128 + d3 * 64)>(%arg10, %arg6, %arg7, %arg8)
              %reinterpret_cast_1 = memref.reinterpret_cast %arg1 to offset: [%18], sizes: [64, 64], strides: [512, 1] {loom.reuse = {sequential = [{depth = 4 : i64, iterator = "%arg10", reuse_type = "no_reuse", volume = 0 : i64}], spatial = [{depth = 0 : i64, iterator = "%arg6", mapped_to = "x", reuse_type = "no_reuse", volume = 0 : i64}, {depth = 1 : i64, iterator = "%arg7", mapped_to = "y", reuse_type = "no_reuse", volume = 0 : i64}], temporal = [{depth = 2 : i64, iterator = "%arg8", reuse_type = "no_reuse", volume = 0 : i64}, {depth = 3 : i64, iterator = "%arg9", reuse_type = "total_reuse", volume = 16384 : i64}]}} : memref<*xf32> to memref<64x64xf32, strided<[512, 1], offset: ?>>
              %alloc_2 = memref.alloc() {loom.alloc = {local = true, memory_name = "L1", size = 16384 : i64}} : memref<64x64xf32>
              memref.copy %reinterpret_cast_1, %alloc_2 {loom.copy.choice = {kind = "mem", memory_name = "L1"}} : memref<64x64xf32, strided<[512, 1], offset: ?>> to memref<64x64xf32>
              %19 = bufferization.to_tensor %alloc_2 restrict writable : memref<64x64xf32> to tensor<64x64xf32>
              %20 = linalg.matmul ins(%17, %19 : tensor<64x64xf32>, tensor<64x64xf32>) outs(%13 : tensor<64x64xf32>) -> tensor<64x64xf32>
              %21 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%arg11, %20 : tensor<64x64xf32>, tensor<64x64xf32>) outs(%arg11 : tensor<64x64xf32>) {
              ^bb0(%in: f32, %in_3: f32, %out: f32):
                %22 = arith.addf %in, %in_3 : f32
                linalg.yield %22 : f32
              } -> tensor<64x64xf32>
              scf.yield %21 : tensor<64x64xf32>
            }
            %15 = affine.apply affine_map<(d0, d1, d2, d3) -> (d0 * 32768 + d1 * 2048 + d2 * 128 + d3 * 64)>(%arg9, %arg6, %arg7, %arg8)
            %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%15], sizes: [64, 64], strides: [512, 1] {loom.reuse = {spatial = [{depth = 0 : i64, iterator = "%arg6", mapped_to = "x", reuse_type = "no_reuse", volume = 0 : i64}, {depth = 1 : i64, iterator = "%arg7", mapped_to = "y", reuse_type = "no_reuse", volume = 0 : i64}], temporal = [{depth = 2 : i64, iterator = "%arg8", reuse_type = "no_reuse", volume = 0 : i64}, {depth = 3 : i64, iterator = "%arg9", reuse_type = "no_reuse", volume = 0 : i64}]}} : memref<*xf32> to memref<64x64xf32, strided<[512, 1], offset: ?>>
            bufferization.materialize_in_destination %14 in writable %reinterpret_cast : (tensor<64x64xf32>, memref<64x64xf32, strided<[512, 1], offset: ?>>) -> ()
          }
        }
      } {loom.mapped_to = "y"}
    } {loom.mapped_to = "x"}
    return
  }
  func.func @matmul_kernel__d0i1_d1i1__f10__hoist_block_1__c0mem_c1mem(%arg0: memref<*xf32> {tt.divisibility = 16 : i32}, %arg1: memref<*xf32> {tt.divisibility = 16 : i32}, %arg2: memref<*xf32> {tt.divisibility = 16 : i32}, %arg3: index, %arg4: index, %arg5: index) {
    affine.parallel (%arg6) = (0) to (32) {
      affine.parallel (%arg7) = (0) to (2) {
        affine.for %arg8 = 0 to affine_map<()[s0, s1] -> ((s1 ceildiv 32) ceildiv 2)>()[%arg3, %arg4] {
          %12 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 2048 + d1 * 128 + d2 * 64)>(%arg6, %arg7, %arg8)
          %reinterpret_cast = memref.reinterpret_cast %arg1 to offset: [%12], sizes: [8, 64, 64], strides: [32768, 512, 1] {loom.reuse = {spatial = [{depth = 0 : i64, iterator = "%arg6", mapped_to = "x", reuse_type = "no_reuse", volume = 0 : i64}, {depth = 1 : i64, iterator = "%arg7", mapped_to = "y", reuse_type = "no_reuse", volume = 0 : i64}], temporal = [{depth = 2 : i64, iterator = "%arg8", reuse_type = "no_reuse", volume = 0 : i64}]}} : memref<*xf32> to memref<8x64x64xf32, strided<[32768, 512, 1], offset: ?>>
          %alloc = memref.alloc() {loom.alloc = {local = true, memory_name = "L1", size = 131072 : i64}} : memref<8x64x64xf32>
          memref.copy %reinterpret_cast, %alloc {loom.copy.choice = {kind = "mem", memory_name = "L1"}} : memref<8x64x64xf32, strided<[32768, 512, 1], offset: ?>> to memref<8x64x64xf32>
          affine.for %arg9 = 0 to affine_map<()[s0, s1] -> (s0)>()[%arg3, %arg4] {
            %cst = arith.constant 0.000000e+00 : f32
            %13 = tensor.empty() : tensor<64x64xf32>
            %14 = linalg.fill ins(%cst : f32) outs(%13 : tensor<64x64xf32>) -> tensor<64x64xf32>
            %c0 = arith.constant 0 : index
            %c8 = arith.constant 8 : index
            %c1 = arith.constant 1 : index
            %15 = scf.for %arg10 = %c0 to %c8 step %c1 iter_args(%arg11 = %14) -> (tensor<64x64xf32>) {
              %17 = affine.apply affine_map<(d0, d1) -> (d1 * 32768 + d0 * 64)>(%arg10, %arg9)
              %reinterpret_cast_1 = memref.reinterpret_cast %arg0 to offset: [%17], sizes: [64, 64], strides: [512, 1] {loom.reuse = {sequential = [{depth = 4 : i64, iterator = "%arg10", reuse_type = "no_reuse", volume = 0 : i64}], spatial = [{depth = 0 : i64, iterator = "%arg6", mapped_to = "x", reuse_type = "total_reuse", volume = 16384 : i64}, {depth = 1 : i64, iterator = "%arg7", mapped_to = "y", reuse_type = "total_reuse", volume = 16384 : i64}], temporal = [{depth = 2 : i64, iterator = "%arg8", reuse_type = "total_reuse", volume = 16384 : i64}, {depth = 3 : i64, iterator = "%arg9", reuse_type = "no_reuse", volume = 0 : i64}]}} : memref<*xf32> to memref<64x64xf32, strided<[512, 1], offset: ?>>
              %alloc_2 = memref.alloc() {loom.alloc = {local = true, memory_name = "L1", size = 16384 : i64}} : memref<64x64xf32>
              memref.copy %reinterpret_cast_1, %alloc_2 {loom.copy.choice = {kind = "mem", memory_name = "L1"}} : memref<64x64xf32, strided<[512, 1], offset: ?>> to memref<64x64xf32>
              %18 = bufferization.to_tensor %alloc_2 restrict writable : memref<64x64xf32> to tensor<64x64xf32>
              %subview = memref.subview %alloc[%arg10, 0, 0] [1, 64, 64] [1, 1, 1] : memref<8x64x64xf32> to memref<64x64xf32, strided<[64, 1], offset: ?>>
              %19 = bufferization.to_tensor %subview restrict writable : memref<64x64xf32, strided<[64, 1], offset: ?>> to tensor<64x64xf32>
              %20 = linalg.matmul ins(%18, %19 : tensor<64x64xf32>, tensor<64x64xf32>) outs(%14 : tensor<64x64xf32>) -> tensor<64x64xf32>
              %21 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%arg11, %20 : tensor<64x64xf32>, tensor<64x64xf32>) outs(%arg11 : tensor<64x64xf32>) {
              ^bb0(%in: f32, %in_3: f32, %out: f32):
                %22 = arith.addf %in, %in_3 : f32
                linalg.yield %22 : f32
              } -> tensor<64x64xf32>
              scf.yield %21 : tensor<64x64xf32>
            }
            %16 = affine.apply affine_map<(d0, d1, d2, d3) -> (d0 * 32768 + d1 * 2048 + d2 * 128 + d3 * 64)>(%arg9, %arg6, %arg7, %arg8)
            %reinterpret_cast_0 = memref.reinterpret_cast %arg2 to offset: [%16], sizes: [64, 64], strides: [512, 1] {loom.reuse = {spatial = [{depth = 0 : i64, iterator = "%arg6", mapped_to = "x", reuse_type = "no_reuse", volume = 0 : i64}, {depth = 1 : i64, iterator = "%arg7", mapped_to = "y", reuse_type = "no_reuse", volume = 0 : i64}], temporal = [{depth = 2 : i64, iterator = "%arg8", reuse_type = "no_reuse", volume = 0 : i64}, {depth = 3 : i64, iterator = "%arg9", reuse_type = "no_reuse", volume = 0 : i64}]}} : memref<*xf32> to memref<64x64xf32, strided<[512, 1], offset: ?>>
            bufferization.materialize_in_destination %15 in writable %reinterpret_cast_0 : (tensor<64x64xf32>, memref<64x64xf32, strided<[512, 1], offset: ?>>) -> ()
          }
        }
      } {loom.mapped_to = "y"}
    } {loom.mapped_to = "x"}
    return
  }
  func.func @matmul_kernel__d0i1_d1i1__f10__hoist_block_1__c0mem_c1bx(%arg0: memref<*xf32> {tt.divisibility = 16 : i32}, %arg1: memref<*xf32> {tt.divisibility = 16 : i32}, %arg2: memref<*xf32> {tt.divisibility = 16 : i32}, %arg3: index, %arg4: index, %arg5: index) {
    affine.parallel (%arg6) = (0) to (32) {
      affine.parallel (%arg7) = (0) to (2) {
        affine.for %arg8 = 0 to affine_map<()[s0, s1] -> ((s1 ceildiv 32) ceildiv 2)>()[%arg3, %arg4] {
          %12 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 2048 + d1 * 128 + d2 * 64)>(%arg6, %arg7, %arg8)
          %reinterpret_cast = memref.reinterpret_cast %arg1 to offset: [%12], sizes: [8, 64, 64], strides: [32768, 512, 1] {loom.reuse = {spatial = [{depth = 0 : i64, iterator = "%arg6", mapped_to = "x", reuse_type = "no_reuse", volume = 0 : i64}, {depth = 1 : i64, iterator = "%arg7", mapped_to = "y", reuse_type = "no_reuse", volume = 0 : i64}], temporal = [{depth = 2 : i64, iterator = "%arg8", reuse_type = "no_reuse", volume = 0 : i64}]}} : memref<*xf32> to memref<8x64x64xf32, strided<[32768, 512, 1], offset: ?>>
          %alloc = memref.alloc() {loom.alloc = {local = true, memory_name = "L1", size = 131072 : i64}} : memref<8x64x64xf32>
          memref.copy %reinterpret_cast, %alloc {loom.copy.choice = {kind = "mem", memory_name = "L1"}} : memref<8x64x64xf32, strided<[32768, 512, 1], offset: ?>> to memref<8x64x64xf32>
          affine.for %arg9 = 0 to affine_map<()[s0, s1] -> (s0)>()[%arg3, %arg4] {
            %cst = arith.constant 0.000000e+00 : f32
            %13 = tensor.empty() : tensor<64x64xf32>
            %14 = linalg.fill ins(%cst : f32) outs(%13 : tensor<64x64xf32>) -> tensor<64x64xf32>
            %c0 = arith.constant 0 : index
            %c8 = arith.constant 8 : index
            %c1 = arith.constant 1 : index
            %15 = scf.for %arg10 = %c0 to %c8 step %c1 iter_args(%arg11 = %14) -> (tensor<64x64xf32>) {
              %17 = affine.apply affine_map<(d0, d1) -> (d1 * 32768 + d0 * 64)>(%arg10, %arg9)
              %reinterpret_cast_1 = memref.reinterpret_cast %arg0 to offset: [%17], sizes: [64, 64], strides: [512, 1] {loom.reuse = {sequential = [{depth = 4 : i64, iterator = "%arg10", reuse_type = "no_reuse", volume = 0 : i64}], spatial = [{depth = 0 : i64, iterator = "%arg6", mapped_to = "x", reuse_type = "total_reuse", volume = 16384 : i64}, {depth = 1 : i64, iterator = "%arg7", mapped_to = "y", reuse_type = "total_reuse", volume = 16384 : i64}], temporal = [{depth = 2 : i64, iterator = "%arg8", reuse_type = "total_reuse", volume = 16384 : i64}, {depth = 3 : i64, iterator = "%arg9", reuse_type = "no_reuse", volume = 0 : i64}]}} : memref<*xf32> to memref<64x64xf32, strided<[512, 1], offset: ?>>
              %alloc_2 = memref.alloc() {loom.alloc = {local = true, memory_name = "L1", size = 16384 : i64}} : memref<64x64xf32>
              memref.copy %reinterpret_cast_1, %alloc_2 {loom.copy.choice = {dim = "x", interconnect_name = "big_ring", kind = "broadcast"}} : memref<64x64xf32, strided<[512, 1], offset: ?>> to memref<64x64xf32>
              %18 = bufferization.to_tensor %alloc_2 restrict writable : memref<64x64xf32> to tensor<64x64xf32>
              %subview = memref.subview %alloc[%arg10, 0, 0] [1, 64, 64] [1, 1, 1] : memref<8x64x64xf32> to memref<64x64xf32, strided<[64, 1], offset: ?>>
              %19 = bufferization.to_tensor %subview restrict writable : memref<64x64xf32, strided<[64, 1], offset: ?>> to tensor<64x64xf32>
              %20 = linalg.matmul ins(%18, %19 : tensor<64x64xf32>, tensor<64x64xf32>) outs(%14 : tensor<64x64xf32>) -> tensor<64x64xf32>
              %21 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%arg11, %20 : tensor<64x64xf32>, tensor<64x64xf32>) outs(%arg11 : tensor<64x64xf32>) {
              ^bb0(%in: f32, %in_3: f32, %out: f32):
                %22 = arith.addf %in, %in_3 : f32
                linalg.yield %22 : f32
              } -> tensor<64x64xf32>
              scf.yield %21 : tensor<64x64xf32>
            }
            %16 = affine.apply affine_map<(d0, d1, d2, d3) -> (d0 * 32768 + d1 * 2048 + d2 * 128 + d3 * 64)>(%arg9, %arg6, %arg7, %arg8)
            %reinterpret_cast_0 = memref.reinterpret_cast %arg2 to offset: [%16], sizes: [64, 64], strides: [512, 1] {loom.reuse = {spatial = [{depth = 0 : i64, iterator = "%arg6", mapped_to = "x", reuse_type = "no_reuse", volume = 0 : i64}, {depth = 1 : i64, iterator = "%arg7", mapped_to = "y", reuse_type = "no_reuse", volume = 0 : i64}], temporal = [{depth = 2 : i64, iterator = "%arg8", reuse_type = "no_reuse", volume = 0 : i64}, {depth = 3 : i64, iterator = "%arg9", reuse_type = "no_reuse", volume = 0 : i64}]}} : memref<*xf32> to memref<64x64xf32, strided<[512, 1], offset: ?>>
            bufferization.materialize_in_destination %15 in writable %reinterpret_cast_0 : (tensor<64x64xf32>, memref<64x64xf32, strided<[512, 1], offset: ?>>) -> ()
          }
        }
      } {loom.mapped_to = "y"}
    } {loom.mapped_to = "x"}
    return
  }
  func.func @matmul_kernel__d1i1_d0i1__f01__c0mem_c1mem(%arg0: memref<*xf32> {tt.divisibility = 16 : i32}, %arg1: memref<*xf32> {tt.divisibility = 16 : i32}, %arg2: memref<*xf32> {tt.divisibility = 16 : i32}, %arg3: index, %arg4: index, %arg5: index) {
    affine.parallel (%arg6) = (0) to (2) {
      affine.parallel (%arg7) = (0) to (32) {
        affine.for %arg8 = 0 to affine_map<()[s0, s1] -> (s0)>()[%arg3, %arg4] {
          affine.for %arg9 = 0 to affine_map<()[s0, s1] -> ((s1 ceildiv 2) ceildiv 32)>()[%arg3, %arg4] {
            %cst = arith.constant 0.000000e+00 : f32
            %12 = tensor.empty() : tensor<64x64xf32>
            %13 = linalg.fill ins(%cst : f32) outs(%12 : tensor<64x64xf32>) -> tensor<64x64xf32>
            %c0 = arith.constant 0 : index
            %c8 = arith.constant 8 : index
            %c1 = arith.constant 1 : index
            %14 = scf.for %arg10 = %c0 to %c8 step %c1 iter_args(%arg11 = %13) -> (tensor<64x64xf32>) {
              %16 = affine.apply affine_map<(d0, d1) -> (d1 * 32768 + d0 * 64)>(%arg10, %arg8)
              %reinterpret_cast_0 = memref.reinterpret_cast %arg0 to offset: [%16], sizes: [64, 64], strides: [512, 1] {loom.reuse = {sequential = [{depth = 4 : i64, iterator = "%arg10", reuse_type = "no_reuse", volume = 0 : i64}], spatial = [{depth = 0 : i64, iterator = "%arg6", mapped_to = "y", reuse_type = "total_reuse", volume = 16384 : i64}, {depth = 1 : i64, iterator = "%arg7", mapped_to = "x", reuse_type = "total_reuse", volume = 16384 : i64}], temporal = [{depth = 2 : i64, iterator = "%arg8", reuse_type = "no_reuse", volume = 0 : i64}, {depth = 3 : i64, iterator = "%arg9", reuse_type = "total_reuse", volume = 16384 : i64}]}} : memref<*xf32> to memref<64x64xf32, strided<[512, 1], offset: ?>>
              %alloc = memref.alloc() {loom.alloc = {local = true, memory_name = "L1", size = 16384 : i64}} : memref<64x64xf32>
              memref.copy %reinterpret_cast_0, %alloc {loom.copy.choice = {kind = "mem", memory_name = "L1"}} : memref<64x64xf32, strided<[512, 1], offset: ?>> to memref<64x64xf32>
              %17 = bufferization.to_tensor %alloc restrict writable : memref<64x64xf32> to tensor<64x64xf32>
              %18 = affine.apply affine_map<(d0, d1, d2, d3) -> (d0 * 32768 + d1 * 128 + d2 * 2048 + d3 * 64)>(%arg10, %arg6, %arg7, %arg9)
              %reinterpret_cast_1 = memref.reinterpret_cast %arg1 to offset: [%18], sizes: [64, 64], strides: [512, 1] {loom.reuse = {sequential = [{depth = 4 : i64, iterator = "%arg10", reuse_type = "no_reuse", volume = 0 : i64}], spatial = [{depth = 0 : i64, iterator = "%arg6", mapped_to = "y", reuse_type = "no_reuse", volume = 0 : i64}, {depth = 1 : i64, iterator = "%arg7", mapped_to = "x", reuse_type = "no_reuse", volume = 0 : i64}], temporal = [{depth = 2 : i64, iterator = "%arg8", reuse_type = "total_reuse", volume = 16384 : i64}, {depth = 3 : i64, iterator = "%arg9", reuse_type = "no_reuse", volume = 0 : i64}]}} : memref<*xf32> to memref<64x64xf32, strided<[512, 1], offset: ?>>
              %alloc_2 = memref.alloc() {loom.alloc = {local = true, memory_name = "L1", size = 16384 : i64}} : memref<64x64xf32>
              memref.copy %reinterpret_cast_1, %alloc_2 {loom.copy.choice = {kind = "mem", memory_name = "L1"}} : memref<64x64xf32, strided<[512, 1], offset: ?>> to memref<64x64xf32>
              %19 = bufferization.to_tensor %alloc_2 restrict writable : memref<64x64xf32> to tensor<64x64xf32>
              %20 = linalg.matmul ins(%17, %19 : tensor<64x64xf32>, tensor<64x64xf32>) outs(%13 : tensor<64x64xf32>) -> tensor<64x64xf32>
              %21 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%arg11, %20 : tensor<64x64xf32>, tensor<64x64xf32>) outs(%arg11 : tensor<64x64xf32>) {
              ^bb0(%in: f32, %in_3: f32, %out: f32):
                %22 = arith.addf %in, %in_3 : f32
                linalg.yield %22 : f32
              } -> tensor<64x64xf32>
              scf.yield %21 : tensor<64x64xf32>
            }
            %15 = affine.apply affine_map<(d0, d1, d2, d3) -> (d0 * 32768 + d1 * 128 + d2 * 2048 + d3 * 64)>(%arg8, %arg6, %arg7, %arg9)
            %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%15], sizes: [64, 64], strides: [512, 1] {loom.reuse = {spatial = [{depth = 0 : i64, iterator = "%arg6", mapped_to = "y", reuse_type = "no_reuse", volume = 0 : i64}, {depth = 1 : i64, iterator = "%arg7", mapped_to = "x", reuse_type = "no_reuse", volume = 0 : i64}], temporal = [{depth = 2 : i64, iterator = "%arg8", reuse_type = "no_reuse", volume = 0 : i64}, {depth = 3 : i64, iterator = "%arg9", reuse_type = "no_reuse", volume = 0 : i64}]}} : memref<*xf32> to memref<64x64xf32, strided<[512, 1], offset: ?>>
            bufferization.materialize_in_destination %14 in writable %reinterpret_cast : (tensor<64x64xf32>, memref<64x64xf32, strided<[512, 1], offset: ?>>) -> ()
          }
        }
      } {loom.mapped_to = "x"}
    } {loom.mapped_to = "y"}
    return
  }
  func.func @matmul_kernel__d1i1_d0i1__f01__c0bx_c1mem(%arg0: memref<*xf32> {tt.divisibility = 16 : i32}, %arg1: memref<*xf32> {tt.divisibility = 16 : i32}, %arg2: memref<*xf32> {tt.divisibility = 16 : i32}, %arg3: index, %arg4: index, %arg5: index) {
    affine.parallel (%arg6) = (0) to (2) {
      affine.parallel (%arg7) = (0) to (32) {
        affine.for %arg8 = 0 to affine_map<()[s0, s1] -> (s0)>()[%arg3, %arg4] {
          affine.for %arg9 = 0 to affine_map<()[s0, s1] -> ((s1 ceildiv 2) ceildiv 32)>()[%arg3, %arg4] {
            %cst = arith.constant 0.000000e+00 : f32
            %12 = tensor.empty() : tensor<64x64xf32>
            %13 = linalg.fill ins(%cst : f32) outs(%12 : tensor<64x64xf32>) -> tensor<64x64xf32>
            %c0 = arith.constant 0 : index
            %c8 = arith.constant 8 : index
            %c1 = arith.constant 1 : index
            %14 = scf.for %arg10 = %c0 to %c8 step %c1 iter_args(%arg11 = %13) -> (tensor<64x64xf32>) {
              %16 = affine.apply affine_map<(d0, d1) -> (d1 * 32768 + d0 * 64)>(%arg10, %arg8)
              %reinterpret_cast_0 = memref.reinterpret_cast %arg0 to offset: [%16], sizes: [64, 64], strides: [512, 1] {loom.reuse = {sequential = [{depth = 4 : i64, iterator = "%arg10", reuse_type = "no_reuse", volume = 0 : i64}], spatial = [{depth = 0 : i64, iterator = "%arg6", mapped_to = "y", reuse_type = "total_reuse", volume = 16384 : i64}, {depth = 1 : i64, iterator = "%arg7", mapped_to = "x", reuse_type = "total_reuse", volume = 16384 : i64}], temporal = [{depth = 2 : i64, iterator = "%arg8", reuse_type = "no_reuse", volume = 0 : i64}, {depth = 3 : i64, iterator = "%arg9", reuse_type = "total_reuse", volume = 16384 : i64}]}} : memref<*xf32> to memref<64x64xf32, strided<[512, 1], offset: ?>>
              %alloc = memref.alloc() {loom.alloc = {local = true, memory_name = "L1", size = 16384 : i64}} : memref<64x64xf32>
              memref.copy %reinterpret_cast_0, %alloc {loom.copy.choice = {dim = "x", interconnect_name = "big_ring", kind = "broadcast"}} : memref<64x64xf32, strided<[512, 1], offset: ?>> to memref<64x64xf32>
              %17 = bufferization.to_tensor %alloc restrict writable : memref<64x64xf32> to tensor<64x64xf32>
              %18 = affine.apply affine_map<(d0, d1, d2, d3) -> (d0 * 32768 + d1 * 128 + d2 * 2048 + d3 * 64)>(%arg10, %arg6, %arg7, %arg9)
              %reinterpret_cast_1 = memref.reinterpret_cast %arg1 to offset: [%18], sizes: [64, 64], strides: [512, 1] {loom.reuse = {sequential = [{depth = 4 : i64, iterator = "%arg10", reuse_type = "no_reuse", volume = 0 : i64}], spatial = [{depth = 0 : i64, iterator = "%arg6", mapped_to = "y", reuse_type = "no_reuse", volume = 0 : i64}, {depth = 1 : i64, iterator = "%arg7", mapped_to = "x", reuse_type = "no_reuse", volume = 0 : i64}], temporal = [{depth = 2 : i64, iterator = "%arg8", reuse_type = "total_reuse", volume = 16384 : i64}, {depth = 3 : i64, iterator = "%arg9", reuse_type = "no_reuse", volume = 0 : i64}]}} : memref<*xf32> to memref<64x64xf32, strided<[512, 1], offset: ?>>
              %alloc_2 = memref.alloc() {loom.alloc = {local = true, memory_name = "L1", size = 16384 : i64}} : memref<64x64xf32>
              memref.copy %reinterpret_cast_1, %alloc_2 {loom.copy.choice = {kind = "mem", memory_name = "L1"}} : memref<64x64xf32, strided<[512, 1], offset: ?>> to memref<64x64xf32>
              %19 = bufferization.to_tensor %alloc_2 restrict writable : memref<64x64xf32> to tensor<64x64xf32>
              %20 = linalg.matmul ins(%17, %19 : tensor<64x64xf32>, tensor<64x64xf32>) outs(%13 : tensor<64x64xf32>) -> tensor<64x64xf32>
              %21 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%arg11, %20 : tensor<64x64xf32>, tensor<64x64xf32>) outs(%arg11 : tensor<64x64xf32>) {
              ^bb0(%in: f32, %in_3: f32, %out: f32):
                %22 = arith.addf %in, %in_3 : f32
                linalg.yield %22 : f32
              } -> tensor<64x64xf32>
              scf.yield %21 : tensor<64x64xf32>
            }
            %15 = affine.apply affine_map<(d0, d1, d2, d3) -> (d0 * 32768 + d1 * 128 + d2 * 2048 + d3 * 64)>(%arg8, %arg6, %arg7, %arg9)
            %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%15], sizes: [64, 64], strides: [512, 1] {loom.reuse = {spatial = [{depth = 0 : i64, iterator = "%arg6", mapped_to = "y", reuse_type = "no_reuse", volume = 0 : i64}, {depth = 1 : i64, iterator = "%arg7", mapped_to = "x", reuse_type = "no_reuse", volume = 0 : i64}], temporal = [{depth = 2 : i64, iterator = "%arg8", reuse_type = "no_reuse", volume = 0 : i64}, {depth = 3 : i64, iterator = "%arg9", reuse_type = "no_reuse", volume = 0 : i64}]}} : memref<*xf32> to memref<64x64xf32, strided<[512, 1], offset: ?>>
            bufferization.materialize_in_destination %14 in writable %reinterpret_cast : (tensor<64x64xf32>, memref<64x64xf32, strided<[512, 1], offset: ?>>) -> ()
          }
        }
      } {loom.mapped_to = "x"}
    } {loom.mapped_to = "y"}
    return
  }
  func.func @matmul_kernel__d1i1_d0i1__f01__hoist_block_0__c0mem_c1mem(%arg0: memref<*xf32> {tt.divisibility = 16 : i32}, %arg1: memref<*xf32> {tt.divisibility = 16 : i32}, %arg2: memref<*xf32> {tt.divisibility = 16 : i32}, %arg3: index, %arg4: index, %arg5: index) {
    affine.parallel (%arg6) = (0) to (2) {
      affine.parallel (%arg7) = (0) to (32) {
        affine.for %arg8 = 0 to affine_map<()[s0, s1] -> (s0)>()[%arg3, %arg4] {
          %12 = affine.apply affine_map<(d0) -> (d0 * 32768)>(%arg8)
          %reinterpret_cast = memref.reinterpret_cast %arg0 to offset: [%12], sizes: [8, 64, 64], strides: [64, 512, 1] {loom.reuse = {spatial = [{depth = 0 : i64, iterator = "%arg6", mapped_to = "y", reuse_type = "total_reuse", volume = 131072 : i64}, {depth = 1 : i64, iterator = "%arg7", mapped_to = "x", reuse_type = "total_reuse", volume = 131072 : i64}], temporal = [{depth = 2 : i64, iterator = "%arg8", reuse_type = "no_reuse", volume = 0 : i64}]}} : memref<*xf32> to memref<8x64x64xf32, strided<[64, 512, 1], offset: ?>>
          %alloc = memref.alloc() {loom.alloc = {local = true, memory_name = "L1", size = 131072 : i64}} : memref<8x64x64xf32>
          memref.copy %reinterpret_cast, %alloc {loom.copy.choice = {kind = "mem", memory_name = "L1"}} : memref<8x64x64xf32, strided<[64, 512, 1], offset: ?>> to memref<8x64x64xf32>
          affine.for %arg9 = 0 to affine_map<()[s0, s1] -> ((s1 ceildiv 2) ceildiv 32)>()[%arg3, %arg4] {
            %cst = arith.constant 0.000000e+00 : f32
            %13 = tensor.empty() : tensor<64x64xf32>
            %14 = linalg.fill ins(%cst : f32) outs(%13 : tensor<64x64xf32>) -> tensor<64x64xf32>
            %c0 = arith.constant 0 : index
            %c8 = arith.constant 8 : index
            %c1 = arith.constant 1 : index
            %15 = scf.for %arg10 = %c0 to %c8 step %c1 iter_args(%arg11 = %14) -> (tensor<64x64xf32>) {
              %subview = memref.subview %alloc[%arg10, 0, 0] [1, 64, 64] [1, 1, 1] : memref<8x64x64xf32> to memref<64x64xf32, strided<[64, 1], offset: ?>>
              %17 = bufferization.to_tensor %subview restrict writable : memref<64x64xf32, strided<[64, 1], offset: ?>> to tensor<64x64xf32>
              %18 = affine.apply affine_map<(d0, d1, d2, d3) -> (d0 * 32768 + d1 * 128 + d2 * 2048 + d3 * 64)>(%arg10, %arg6, %arg7, %arg9)
              %reinterpret_cast_1 = memref.reinterpret_cast %arg1 to offset: [%18], sizes: [64, 64], strides: [512, 1] {loom.reuse = {sequential = [{depth = 4 : i64, iterator = "%arg10", reuse_type = "no_reuse", volume = 0 : i64}], spatial = [{depth = 0 : i64, iterator = "%arg6", mapped_to = "y", reuse_type = "no_reuse", volume = 0 : i64}, {depth = 1 : i64, iterator = "%arg7", mapped_to = "x", reuse_type = "no_reuse", volume = 0 : i64}], temporal = [{depth = 2 : i64, iterator = "%arg8", reuse_type = "total_reuse", volume = 16384 : i64}, {depth = 3 : i64, iterator = "%arg9", reuse_type = "no_reuse", volume = 0 : i64}]}} : memref<*xf32> to memref<64x64xf32, strided<[512, 1], offset: ?>>
              %alloc_2 = memref.alloc() {loom.alloc = {local = true, memory_name = "L1", size = 16384 : i64}} : memref<64x64xf32>
              memref.copy %reinterpret_cast_1, %alloc_2 {loom.copy.choice = {kind = "mem", memory_name = "L1"}} : memref<64x64xf32, strided<[512, 1], offset: ?>> to memref<64x64xf32>
              %19 = bufferization.to_tensor %alloc_2 restrict writable : memref<64x64xf32> to tensor<64x64xf32>
              %20 = linalg.matmul ins(%17, %19 : tensor<64x64xf32>, tensor<64x64xf32>) outs(%14 : tensor<64x64xf32>) -> tensor<64x64xf32>
              %21 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%arg11, %20 : tensor<64x64xf32>, tensor<64x64xf32>) outs(%arg11 : tensor<64x64xf32>) {
              ^bb0(%in: f32, %in_3: f32, %out: f32):
                %22 = arith.addf %in, %in_3 : f32
                linalg.yield %22 : f32
              } -> tensor<64x64xf32>
              scf.yield %21 : tensor<64x64xf32>
            }
            %16 = affine.apply affine_map<(d0, d1, d2, d3) -> (d0 * 32768 + d1 * 128 + d2 * 2048 + d3 * 64)>(%arg8, %arg6, %arg7, %arg9)
            %reinterpret_cast_0 = memref.reinterpret_cast %arg2 to offset: [%16], sizes: [64, 64], strides: [512, 1] {loom.reuse = {spatial = [{depth = 0 : i64, iterator = "%arg6", mapped_to = "y", reuse_type = "no_reuse", volume = 0 : i64}, {depth = 1 : i64, iterator = "%arg7", mapped_to = "x", reuse_type = "no_reuse", volume = 0 : i64}], temporal = [{depth = 2 : i64, iterator = "%arg8", reuse_type = "no_reuse", volume = 0 : i64}, {depth = 3 : i64, iterator = "%arg9", reuse_type = "no_reuse", volume = 0 : i64}]}} : memref<*xf32> to memref<64x64xf32, strided<[512, 1], offset: ?>>
            bufferization.materialize_in_destination %15 in writable %reinterpret_cast_0 : (tensor<64x64xf32>, memref<64x64xf32, strided<[512, 1], offset: ?>>) -> ()
          }
        }
      } {loom.mapped_to = "x"}
    } {loom.mapped_to = "y"}
    return
  }
  func.func @matmul_kernel__d1i1_d0i1__f01__hoist_block_0__c0bx_c1mem(%arg0: memref<*xf32> {tt.divisibility = 16 : i32}, %arg1: memref<*xf32> {tt.divisibility = 16 : i32}, %arg2: memref<*xf32> {tt.divisibility = 16 : i32}, %arg3: index, %arg4: index, %arg5: index) {
    affine.parallel (%arg6) = (0) to (2) {
      affine.parallel (%arg7) = (0) to (32) {
        affine.for %arg8 = 0 to affine_map<()[s0, s1] -> (s0)>()[%arg3, %arg4] {
          %12 = affine.apply affine_map<(d0) -> (d0 * 32768)>(%arg8)
          %reinterpret_cast = memref.reinterpret_cast %arg0 to offset: [%12], sizes: [8, 64, 64], strides: [64, 512, 1] {loom.reuse = {spatial = [{depth = 0 : i64, iterator = "%arg6", mapped_to = "y", reuse_type = "total_reuse", volume = 131072 : i64}, {depth = 1 : i64, iterator = "%arg7", mapped_to = "x", reuse_type = "total_reuse", volume = 131072 : i64}], temporal = [{depth = 2 : i64, iterator = "%arg8", reuse_type = "no_reuse", volume = 0 : i64}]}} : memref<*xf32> to memref<8x64x64xf32, strided<[64, 512, 1], offset: ?>>
          %alloc = memref.alloc() {loom.alloc = {local = true, memory_name = "L1", size = 131072 : i64}} : memref<8x64x64xf32>
          memref.copy %reinterpret_cast, %alloc {loom.copy.choice = {dim = "x", interconnect_name = "big_ring", kind = "broadcast"}} : memref<8x64x64xf32, strided<[64, 512, 1], offset: ?>> to memref<8x64x64xf32>
          affine.for %arg9 = 0 to affine_map<()[s0, s1] -> ((s1 ceildiv 2) ceildiv 32)>()[%arg3, %arg4] {
            %cst = arith.constant 0.000000e+00 : f32
            %13 = tensor.empty() : tensor<64x64xf32>
            %14 = linalg.fill ins(%cst : f32) outs(%13 : tensor<64x64xf32>) -> tensor<64x64xf32>
            %c0 = arith.constant 0 : index
            %c8 = arith.constant 8 : index
            %c1 = arith.constant 1 : index
            %15 = scf.for %arg10 = %c0 to %c8 step %c1 iter_args(%arg11 = %14) -> (tensor<64x64xf32>) {
              %subview = memref.subview %alloc[%arg10, 0, 0] [1, 64, 64] [1, 1, 1] : memref<8x64x64xf32> to memref<64x64xf32, strided<[64, 1], offset: ?>>
              %17 = bufferization.to_tensor %subview restrict writable : memref<64x64xf32, strided<[64, 1], offset: ?>> to tensor<64x64xf32>
              %18 = affine.apply affine_map<(d0, d1, d2, d3) -> (d0 * 32768 + d1 * 128 + d2 * 2048 + d3 * 64)>(%arg10, %arg6, %arg7, %arg9)
              %reinterpret_cast_1 = memref.reinterpret_cast %arg1 to offset: [%18], sizes: [64, 64], strides: [512, 1] {loom.reuse = {sequential = [{depth = 4 : i64, iterator = "%arg10", reuse_type = "no_reuse", volume = 0 : i64}], spatial = [{depth = 0 : i64, iterator = "%arg6", mapped_to = "y", reuse_type = "no_reuse", volume = 0 : i64}, {depth = 1 : i64, iterator = "%arg7", mapped_to = "x", reuse_type = "no_reuse", volume = 0 : i64}], temporal = [{depth = 2 : i64, iterator = "%arg8", reuse_type = "total_reuse", volume = 16384 : i64}, {depth = 3 : i64, iterator = "%arg9", reuse_type = "no_reuse", volume = 0 : i64}]}} : memref<*xf32> to memref<64x64xf32, strided<[512, 1], offset: ?>>
              %alloc_2 = memref.alloc() {loom.alloc = {local = true, memory_name = "L1", size = 16384 : i64}} : memref<64x64xf32>
              memref.copy %reinterpret_cast_1, %alloc_2 {loom.copy.choice = {kind = "mem", memory_name = "L1"}} : memref<64x64xf32, strided<[512, 1], offset: ?>> to memref<64x64xf32>
              %19 = bufferization.to_tensor %alloc_2 restrict writable : memref<64x64xf32> to tensor<64x64xf32>
              %20 = linalg.matmul ins(%17, %19 : tensor<64x64xf32>, tensor<64x64xf32>) outs(%14 : tensor<64x64xf32>) -> tensor<64x64xf32>
              %21 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%arg11, %20 : tensor<64x64xf32>, tensor<64x64xf32>) outs(%arg11 : tensor<64x64xf32>) {
              ^bb0(%in: f32, %in_3: f32, %out: f32):
                %22 = arith.addf %in, %in_3 : f32
                linalg.yield %22 : f32
              } -> tensor<64x64xf32>
              scf.yield %21 : tensor<64x64xf32>
            }
            %16 = affine.apply affine_map<(d0, d1, d2, d3) -> (d0 * 32768 + d1 * 128 + d2 * 2048 + d3 * 64)>(%arg8, %arg6, %arg7, %arg9)
            %reinterpret_cast_0 = memref.reinterpret_cast %arg2 to offset: [%16], sizes: [64, 64], strides: [512, 1] {loom.reuse = {spatial = [{depth = 0 : i64, iterator = "%arg6", mapped_to = "y", reuse_type = "no_reuse", volume = 0 : i64}, {depth = 1 : i64, iterator = "%arg7", mapped_to = "x", reuse_type = "no_reuse", volume = 0 : i64}], temporal = [{depth = 2 : i64, iterator = "%arg8", reuse_type = "no_reuse", volume = 0 : i64}, {depth = 3 : i64, iterator = "%arg9", reuse_type = "no_reuse", volume = 0 : i64}]}} : memref<*xf32> to memref<64x64xf32, strided<[512, 1], offset: ?>>
            bufferization.materialize_in_destination %15 in writable %reinterpret_cast_0 : (tensor<64x64xf32>, memref<64x64xf32, strided<[512, 1], offset: ?>>) -> ()
          }
        }
      } {loom.mapped_to = "x"}
    } {loom.mapped_to = "y"}
    return
  }
  func.func @matmul_kernel__d1i1_d0i1__f10__c0mem_c1mem(%arg0: memref<*xf32> {tt.divisibility = 16 : i32}, %arg1: memref<*xf32> {tt.divisibility = 16 : i32}, %arg2: memref<*xf32> {tt.divisibility = 16 : i32}, %arg3: index, %arg4: index, %arg5: index) {
    affine.parallel (%arg6) = (0) to (2) {
      affine.parallel (%arg7) = (0) to (32) {
        affine.for %arg8 = 0 to affine_map<()[s0, s1] -> ((s1 ceildiv 2) ceildiv 32)>()[%arg3, %arg4] {
          affine.for %arg9 = 0 to affine_map<()[s0, s1] -> (s0)>()[%arg3, %arg4] {
            %cst = arith.constant 0.000000e+00 : f32
            %12 = tensor.empty() : tensor<64x64xf32>
            %13 = linalg.fill ins(%cst : f32) outs(%12 : tensor<64x64xf32>) -> tensor<64x64xf32>
            %c0 = arith.constant 0 : index
            %c8 = arith.constant 8 : index
            %c1 = arith.constant 1 : index
            %14 = scf.for %arg10 = %c0 to %c8 step %c1 iter_args(%arg11 = %13) -> (tensor<64x64xf32>) {
              %16 = affine.apply affine_map<(d0, d1) -> (d1 * 32768 + d0 * 64)>(%arg10, %arg9)
              %reinterpret_cast_0 = memref.reinterpret_cast %arg0 to offset: [%16], sizes: [64, 64], strides: [512, 1] {loom.reuse = {sequential = [{depth = 4 : i64, iterator = "%arg10", reuse_type = "no_reuse", volume = 0 : i64}], spatial = [{depth = 0 : i64, iterator = "%arg6", mapped_to = "y", reuse_type = "total_reuse", volume = 16384 : i64}, {depth = 1 : i64, iterator = "%arg7", mapped_to = "x", reuse_type = "total_reuse", volume = 16384 : i64}], temporal = [{depth = 2 : i64, iterator = "%arg8", reuse_type = "total_reuse", volume = 16384 : i64}, {depth = 3 : i64, iterator = "%arg9", reuse_type = "no_reuse", volume = 0 : i64}]}} : memref<*xf32> to memref<64x64xf32, strided<[512, 1], offset: ?>>
              %alloc = memref.alloc() {loom.alloc = {local = true, memory_name = "L1", size = 16384 : i64}} : memref<64x64xf32>
              memref.copy %reinterpret_cast_0, %alloc {loom.copy.choice = {kind = "mem", memory_name = "L1"}} : memref<64x64xf32, strided<[512, 1], offset: ?>> to memref<64x64xf32>
              %17 = bufferization.to_tensor %alloc restrict writable : memref<64x64xf32> to tensor<64x64xf32>
              %18 = affine.apply affine_map<(d0, d1, d2, d3) -> (d0 * 32768 + d1 * 128 + d2 * 2048 + d3 * 64)>(%arg10, %arg6, %arg7, %arg8)
              %reinterpret_cast_1 = memref.reinterpret_cast %arg1 to offset: [%18], sizes: [64, 64], strides: [512, 1] {loom.reuse = {sequential = [{depth = 4 : i64, iterator = "%arg10", reuse_type = "no_reuse", volume = 0 : i64}], spatial = [{depth = 0 : i64, iterator = "%arg6", mapped_to = "y", reuse_type = "no_reuse", volume = 0 : i64}, {depth = 1 : i64, iterator = "%arg7", mapped_to = "x", reuse_type = "no_reuse", volume = 0 : i64}], temporal = [{depth = 2 : i64, iterator = "%arg8", reuse_type = "no_reuse", volume = 0 : i64}, {depth = 3 : i64, iterator = "%arg9", reuse_type = "total_reuse", volume = 16384 : i64}]}} : memref<*xf32> to memref<64x64xf32, strided<[512, 1], offset: ?>>
              %alloc_2 = memref.alloc() {loom.alloc = {local = true, memory_name = "L1", size = 16384 : i64}} : memref<64x64xf32>
              memref.copy %reinterpret_cast_1, %alloc_2 {loom.copy.choice = {kind = "mem", memory_name = "L1"}} : memref<64x64xf32, strided<[512, 1], offset: ?>> to memref<64x64xf32>
              %19 = bufferization.to_tensor %alloc_2 restrict writable : memref<64x64xf32> to tensor<64x64xf32>
              %20 = linalg.matmul ins(%17, %19 : tensor<64x64xf32>, tensor<64x64xf32>) outs(%13 : tensor<64x64xf32>) -> tensor<64x64xf32>
              %21 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%arg11, %20 : tensor<64x64xf32>, tensor<64x64xf32>) outs(%arg11 : tensor<64x64xf32>) {
              ^bb0(%in: f32, %in_3: f32, %out: f32):
                %22 = arith.addf %in, %in_3 : f32
                linalg.yield %22 : f32
              } -> tensor<64x64xf32>
              scf.yield %21 : tensor<64x64xf32>
            }
            %15 = affine.apply affine_map<(d0, d1, d2, d3) -> (d0 * 32768 + d1 * 128 + d2 * 2048 + d3 * 64)>(%arg9, %arg6, %arg7, %arg8)
            %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%15], sizes: [64, 64], strides: [512, 1] {loom.reuse = {spatial = [{depth = 0 : i64, iterator = "%arg6", mapped_to = "y", reuse_type = "no_reuse", volume = 0 : i64}, {depth = 1 : i64, iterator = "%arg7", mapped_to = "x", reuse_type = "no_reuse", volume = 0 : i64}], temporal = [{depth = 2 : i64, iterator = "%arg8", reuse_type = "no_reuse", volume = 0 : i64}, {depth = 3 : i64, iterator = "%arg9", reuse_type = "no_reuse", volume = 0 : i64}]}} : memref<*xf32> to memref<64x64xf32, strided<[512, 1], offset: ?>>
            bufferization.materialize_in_destination %14 in writable %reinterpret_cast : (tensor<64x64xf32>, memref<64x64xf32, strided<[512, 1], offset: ?>>) -> ()
          }
        }
      } {loom.mapped_to = "x"}
    } {loom.mapped_to = "y"}
    return
  }
  func.func @matmul_kernel__d1i1_d0i1__f10__c0bx_c1mem(%arg0: memref<*xf32> {tt.divisibility = 16 : i32}, %arg1: memref<*xf32> {tt.divisibility = 16 : i32}, %arg2: memref<*xf32> {tt.divisibility = 16 : i32}, %arg3: index, %arg4: index, %arg5: index) {
    affine.parallel (%arg6) = (0) to (2) {
      affine.parallel (%arg7) = (0) to (32) {
        affine.for %arg8 = 0 to affine_map<()[s0, s1] -> ((s1 ceildiv 2) ceildiv 32)>()[%arg3, %arg4] {
          affine.for %arg9 = 0 to affine_map<()[s0, s1] -> (s0)>()[%arg3, %arg4] {
            %cst = arith.constant 0.000000e+00 : f32
            %12 = tensor.empty() : tensor<64x64xf32>
            %13 = linalg.fill ins(%cst : f32) outs(%12 : tensor<64x64xf32>) -> tensor<64x64xf32>
            %c0 = arith.constant 0 : index
            %c8 = arith.constant 8 : index
            %c1 = arith.constant 1 : index
            %14 = scf.for %arg10 = %c0 to %c8 step %c1 iter_args(%arg11 = %13) -> (tensor<64x64xf32>) {
              %16 = affine.apply affine_map<(d0, d1) -> (d1 * 32768 + d0 * 64)>(%arg10, %arg9)
              %reinterpret_cast_0 = memref.reinterpret_cast %arg0 to offset: [%16], sizes: [64, 64], strides: [512, 1] {loom.reuse = {sequential = [{depth = 4 : i64, iterator = "%arg10", reuse_type = "no_reuse", volume = 0 : i64}], spatial = [{depth = 0 : i64, iterator = "%arg6", mapped_to = "y", reuse_type = "total_reuse", volume = 16384 : i64}, {depth = 1 : i64, iterator = "%arg7", mapped_to = "x", reuse_type = "total_reuse", volume = 16384 : i64}], temporal = [{depth = 2 : i64, iterator = "%arg8", reuse_type = "total_reuse", volume = 16384 : i64}, {depth = 3 : i64, iterator = "%arg9", reuse_type = "no_reuse", volume = 0 : i64}]}} : memref<*xf32> to memref<64x64xf32, strided<[512, 1], offset: ?>>
              %alloc = memref.alloc() {loom.alloc = {local = true, memory_name = "L1", size = 16384 : i64}} : memref<64x64xf32>
              memref.copy %reinterpret_cast_0, %alloc {loom.copy.choice = {dim = "x", interconnect_name = "big_ring", kind = "broadcast"}} : memref<64x64xf32, strided<[512, 1], offset: ?>> to memref<64x64xf32>
              %17 = bufferization.to_tensor %alloc restrict writable : memref<64x64xf32> to tensor<64x64xf32>
              %18 = affine.apply affine_map<(d0, d1, d2, d3) -> (d0 * 32768 + d1 * 128 + d2 * 2048 + d3 * 64)>(%arg10, %arg6, %arg7, %arg8)
              %reinterpret_cast_1 = memref.reinterpret_cast %arg1 to offset: [%18], sizes: [64, 64], strides: [512, 1] {loom.reuse = {sequential = [{depth = 4 : i64, iterator = "%arg10", reuse_type = "no_reuse", volume = 0 : i64}], spatial = [{depth = 0 : i64, iterator = "%arg6", mapped_to = "y", reuse_type = "no_reuse", volume = 0 : i64}, {depth = 1 : i64, iterator = "%arg7", mapped_to = "x", reuse_type = "no_reuse", volume = 0 : i64}], temporal = [{depth = 2 : i64, iterator = "%arg8", reuse_type = "no_reuse", volume = 0 : i64}, {depth = 3 : i64, iterator = "%arg9", reuse_type = "total_reuse", volume = 16384 : i64}]}} : memref<*xf32> to memref<64x64xf32, strided<[512, 1], offset: ?>>
              %alloc_2 = memref.alloc() {loom.alloc = {local = true, memory_name = "L1", size = 16384 : i64}} : memref<64x64xf32>
              memref.copy %reinterpret_cast_1, %alloc_2 {loom.copy.choice = {kind = "mem", memory_name = "L1"}} : memref<64x64xf32, strided<[512, 1], offset: ?>> to memref<64x64xf32>
              %19 = bufferization.to_tensor %alloc_2 restrict writable : memref<64x64xf32> to tensor<64x64xf32>
              %20 = linalg.matmul ins(%17, %19 : tensor<64x64xf32>, tensor<64x64xf32>) outs(%13 : tensor<64x64xf32>) -> tensor<64x64xf32>
              %21 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%arg11, %20 : tensor<64x64xf32>, tensor<64x64xf32>) outs(%arg11 : tensor<64x64xf32>) {
              ^bb0(%in: f32, %in_3: f32, %out: f32):
                %22 = arith.addf %in, %in_3 : f32
                linalg.yield %22 : f32
              } -> tensor<64x64xf32>
              scf.yield %21 : tensor<64x64xf32>
            }
            %15 = affine.apply affine_map<(d0, d1, d2, d3) -> (d0 * 32768 + d1 * 128 + d2 * 2048 + d3 * 64)>(%arg9, %arg6, %arg7, %arg8)
            %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%15], sizes: [64, 64], strides: [512, 1] {loom.reuse = {spatial = [{depth = 0 : i64, iterator = "%arg6", mapped_to = "y", reuse_type = "no_reuse", volume = 0 : i64}, {depth = 1 : i64, iterator = "%arg7", mapped_to = "x", reuse_type = "no_reuse", volume = 0 : i64}], temporal = [{depth = 2 : i64, iterator = "%arg8", reuse_type = "no_reuse", volume = 0 : i64}, {depth = 3 : i64, iterator = "%arg9", reuse_type = "no_reuse", volume = 0 : i64}]}} : memref<*xf32> to memref<64x64xf32, strided<[512, 1], offset: ?>>
            bufferization.materialize_in_destination %14 in writable %reinterpret_cast : (tensor<64x64xf32>, memref<64x64xf32, strided<[512, 1], offset: ?>>) -> ()
          }
        }
      } {loom.mapped_to = "x"}
    } {loom.mapped_to = "y"}
    return
  }
  func.func @matmul_kernel__d1i1_d0i1__f10__hoist_block_1__c0mem_c1mem(%arg0: memref<*xf32> {tt.divisibility = 16 : i32}, %arg1: memref<*xf32> {tt.divisibility = 16 : i32}, %arg2: memref<*xf32> {tt.divisibility = 16 : i32}, %arg3: index, %arg4: index, %arg5: index) {
    affine.parallel (%arg6) = (0) to (2) {
      affine.parallel (%arg7) = (0) to (32) {
        affine.for %arg8 = 0 to affine_map<()[s0, s1] -> ((s1 ceildiv 2) ceildiv 32)>()[%arg3, %arg4] {
          %12 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 128 + d1 * 2048 + d2 * 64)>(%arg6, %arg7, %arg8)
          %reinterpret_cast = memref.reinterpret_cast %arg1 to offset: [%12], sizes: [8, 64, 64], strides: [32768, 512, 1] {loom.reuse = {spatial = [{depth = 0 : i64, iterator = "%arg6", mapped_to = "y", reuse_type = "no_reuse", volume = 0 : i64}, {depth = 1 : i64, iterator = "%arg7", mapped_to = "x", reuse_type = "no_reuse", volume = 0 : i64}], temporal = [{depth = 2 : i64, iterator = "%arg8", reuse_type = "no_reuse", volume = 0 : i64}]}} : memref<*xf32> to memref<8x64x64xf32, strided<[32768, 512, 1], offset: ?>>
          %alloc = memref.alloc() {loom.alloc = {local = true, memory_name = "L1", size = 131072 : i64}} : memref<8x64x64xf32>
          memref.copy %reinterpret_cast, %alloc {loom.copy.choice = {kind = "mem", memory_name = "L1"}} : memref<8x64x64xf32, strided<[32768, 512, 1], offset: ?>> to memref<8x64x64xf32>
          affine.for %arg9 = 0 to affine_map<()[s0, s1] -> (s0)>()[%arg3, %arg4] {
            %cst = arith.constant 0.000000e+00 : f32
            %13 = tensor.empty() : tensor<64x64xf32>
            %14 = linalg.fill ins(%cst : f32) outs(%13 : tensor<64x64xf32>) -> tensor<64x64xf32>
            %c0 = arith.constant 0 : index
            %c8 = arith.constant 8 : index
            %c1 = arith.constant 1 : index
            %15 = scf.for %arg10 = %c0 to %c8 step %c1 iter_args(%arg11 = %14) -> (tensor<64x64xf32>) {
              %17 = affine.apply affine_map<(d0, d1) -> (d1 * 32768 + d0 * 64)>(%arg10, %arg9)
              %reinterpret_cast_1 = memref.reinterpret_cast %arg0 to offset: [%17], sizes: [64, 64], strides: [512, 1] {loom.reuse = {sequential = [{depth = 4 : i64, iterator = "%arg10", reuse_type = "no_reuse", volume = 0 : i64}], spatial = [{depth = 0 : i64, iterator = "%arg6", mapped_to = "y", reuse_type = "total_reuse", volume = 16384 : i64}, {depth = 1 : i64, iterator = "%arg7", mapped_to = "x", reuse_type = "total_reuse", volume = 16384 : i64}], temporal = [{depth = 2 : i64, iterator = "%arg8", reuse_type = "total_reuse", volume = 16384 : i64}, {depth = 3 : i64, iterator = "%arg9", reuse_type = "no_reuse", volume = 0 : i64}]}} : memref<*xf32> to memref<64x64xf32, strided<[512, 1], offset: ?>>
              %alloc_2 = memref.alloc() {loom.alloc = {local = true, memory_name = "L1", size = 16384 : i64}} : memref<64x64xf32>
              memref.copy %reinterpret_cast_1, %alloc_2 {loom.copy.choice = {kind = "mem", memory_name = "L1"}} : memref<64x64xf32, strided<[512, 1], offset: ?>> to memref<64x64xf32>
              %18 = bufferization.to_tensor %alloc_2 restrict writable : memref<64x64xf32> to tensor<64x64xf32>
              %subview = memref.subview %alloc[%arg10, 0, 0] [1, 64, 64] [1, 1, 1] : memref<8x64x64xf32> to memref<64x64xf32, strided<[64, 1], offset: ?>>
              %19 = bufferization.to_tensor %subview restrict writable : memref<64x64xf32, strided<[64, 1], offset: ?>> to tensor<64x64xf32>
              %20 = linalg.matmul ins(%18, %19 : tensor<64x64xf32>, tensor<64x64xf32>) outs(%14 : tensor<64x64xf32>) -> tensor<64x64xf32>
              %21 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%arg11, %20 : tensor<64x64xf32>, tensor<64x64xf32>) outs(%arg11 : tensor<64x64xf32>) {
              ^bb0(%in: f32, %in_3: f32, %out: f32):
                %22 = arith.addf %in, %in_3 : f32
                linalg.yield %22 : f32
              } -> tensor<64x64xf32>
              scf.yield %21 : tensor<64x64xf32>
            }
            %16 = affine.apply affine_map<(d0, d1, d2, d3) -> (d0 * 32768 + d1 * 128 + d2 * 2048 + d3 * 64)>(%arg9, %arg6, %arg7, %arg8)
            %reinterpret_cast_0 = memref.reinterpret_cast %arg2 to offset: [%16], sizes: [64, 64], strides: [512, 1] {loom.reuse = {spatial = [{depth = 0 : i64, iterator = "%arg6", mapped_to = "y", reuse_type = "no_reuse", volume = 0 : i64}, {depth = 1 : i64, iterator = "%arg7", mapped_to = "x", reuse_type = "no_reuse", volume = 0 : i64}], temporal = [{depth = 2 : i64, iterator = "%arg8", reuse_type = "no_reuse", volume = 0 : i64}, {depth = 3 : i64, iterator = "%arg9", reuse_type = "no_reuse", volume = 0 : i64}]}} : memref<*xf32> to memref<64x64xf32, strided<[512, 1], offset: ?>>
            bufferization.materialize_in_destination %15 in writable %reinterpret_cast_0 : (tensor<64x64xf32>, memref<64x64xf32, strided<[512, 1], offset: ?>>) -> ()
          }
        }
      } {loom.mapped_to = "x"}
    } {loom.mapped_to = "y"}
    return
  }
  func.func @matmul_kernel__d1i1_d0i1__f10__hoist_block_1__c0mem_c1bx(%arg0: memref<*xf32> {tt.divisibility = 16 : i32}, %arg1: memref<*xf32> {tt.divisibility = 16 : i32}, %arg2: memref<*xf32> {tt.divisibility = 16 : i32}, %arg3: index, %arg4: index, %arg5: index) {
    affine.parallel (%arg6) = (0) to (2) {
      affine.parallel (%arg7) = (0) to (32) {
        affine.for %arg8 = 0 to affine_map<()[s0, s1] -> ((s1 ceildiv 2) ceildiv 32)>()[%arg3, %arg4] {
          %12 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 128 + d1 * 2048 + d2 * 64)>(%arg6, %arg7, %arg8)
          %reinterpret_cast = memref.reinterpret_cast %arg1 to offset: [%12], sizes: [8, 64, 64], strides: [32768, 512, 1] {loom.reuse = {spatial = [{depth = 0 : i64, iterator = "%arg6", mapped_to = "y", reuse_type = "no_reuse", volume = 0 : i64}, {depth = 1 : i64, iterator = "%arg7", mapped_to = "x", reuse_type = "no_reuse", volume = 0 : i64}], temporal = [{depth = 2 : i64, iterator = "%arg8", reuse_type = "no_reuse", volume = 0 : i64}]}} : memref<*xf32> to memref<8x64x64xf32, strided<[32768, 512, 1], offset: ?>>
          %alloc = memref.alloc() {loom.alloc = {local = true, memory_name = "L1", size = 131072 : i64}} : memref<8x64x64xf32>
          memref.copy %reinterpret_cast, %alloc {loom.copy.choice = {kind = "mem", memory_name = "L1"}} : memref<8x64x64xf32, strided<[32768, 512, 1], offset: ?>> to memref<8x64x64xf32>
          affine.for %arg9 = 0 to affine_map<()[s0, s1] -> (s0)>()[%arg3, %arg4] {
            %cst = arith.constant 0.000000e+00 : f32
            %13 = tensor.empty() : tensor<64x64xf32>
            %14 = linalg.fill ins(%cst : f32) outs(%13 : tensor<64x64xf32>) -> tensor<64x64xf32>
            %c0 = arith.constant 0 : index
            %c8 = arith.constant 8 : index
            %c1 = arith.constant 1 : index
            %15 = scf.for %arg10 = %c0 to %c8 step %c1 iter_args(%arg11 = %14) -> (tensor<64x64xf32>) {
              %17 = affine.apply affine_map<(d0, d1) -> (d1 * 32768 + d0 * 64)>(%arg10, %arg9)
              %reinterpret_cast_1 = memref.reinterpret_cast %arg0 to offset: [%17], sizes: [64, 64], strides: [512, 1] {loom.reuse = {sequential = [{depth = 4 : i64, iterator = "%arg10", reuse_type = "no_reuse", volume = 0 : i64}], spatial = [{depth = 0 : i64, iterator = "%arg6", mapped_to = "y", reuse_type = "total_reuse", volume = 16384 : i64}, {depth = 1 : i64, iterator = "%arg7", mapped_to = "x", reuse_type = "total_reuse", volume = 16384 : i64}], temporal = [{depth = 2 : i64, iterator = "%arg8", reuse_type = "total_reuse", volume = 16384 : i64}, {depth = 3 : i64, iterator = "%arg9", reuse_type = "no_reuse", volume = 0 : i64}]}} : memref<*xf32> to memref<64x64xf32, strided<[512, 1], offset: ?>>
              %alloc_2 = memref.alloc() {loom.alloc = {local = true, memory_name = "L1", size = 16384 : i64}} : memref<64x64xf32>
              memref.copy %reinterpret_cast_1, %alloc_2 {loom.copy.choice = {dim = "x", interconnect_name = "big_ring", kind = "broadcast"}} : memref<64x64xf32, strided<[512, 1], offset: ?>> to memref<64x64xf32>
              %18 = bufferization.to_tensor %alloc_2 restrict writable : memref<64x64xf32> to tensor<64x64xf32>
              %subview = memref.subview %alloc[%arg10, 0, 0] [1, 64, 64] [1, 1, 1] : memref<8x64x64xf32> to memref<64x64xf32, strided<[64, 1], offset: ?>>
              %19 = bufferization.to_tensor %subview restrict writable : memref<64x64xf32, strided<[64, 1], offset: ?>> to tensor<64x64xf32>
              %20 = linalg.matmul ins(%18, %19 : tensor<64x64xf32>, tensor<64x64xf32>) outs(%14 : tensor<64x64xf32>) -> tensor<64x64xf32>
              %21 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%arg11, %20 : tensor<64x64xf32>, tensor<64x64xf32>) outs(%arg11 : tensor<64x64xf32>) {
              ^bb0(%in: f32, %in_3: f32, %out: f32):
                %22 = arith.addf %in, %in_3 : f32
                linalg.yield %22 : f32
              } -> tensor<64x64xf32>
              scf.yield %21 : tensor<64x64xf32>
            }
            %16 = affine.apply affine_map<(d0, d1, d2, d3) -> (d0 * 32768 + d1 * 128 + d2 * 2048 + d3 * 64)>(%arg9, %arg6, %arg7, %arg8)
            %reinterpret_cast_0 = memref.reinterpret_cast %arg2 to offset: [%16], sizes: [64, 64], strides: [512, 1] {loom.reuse = {spatial = [{depth = 0 : i64, iterator = "%arg6", mapped_to = "y", reuse_type = "no_reuse", volume = 0 : i64}, {depth = 1 : i64, iterator = "%arg7", mapped_to = "x", reuse_type = "no_reuse", volume = 0 : i64}], temporal = [{depth = 2 : i64, iterator = "%arg8", reuse_type = "no_reuse", volume = 0 : i64}, {depth = 3 : i64, iterator = "%arg9", reuse_type = "no_reuse", volume = 0 : i64}]}} : memref<*xf32> to memref<64x64xf32, strided<[512, 1], offset: ?>>
            bufferization.materialize_in_destination %15 in writable %reinterpret_cast_0 : (tensor<64x64xf32>, memref<64x64xf32, strided<[512, 1], offset: ?>>) -> ()
          }
        }
      } {loom.mapped_to = "x"}
    } {loom.mapped_to = "y"}
    return
  }
}
