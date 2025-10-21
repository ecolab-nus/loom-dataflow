module {
  %0 = df.spatial_dim "x", 8
  %1 = df.spatial_dim "y", 8
  %2 = df.compute "cores", %0, %1 {map = affine_map<(d0, d1) -> (d0, d1)>}
  %3 = df.memory "L1", %0, %1 {map = affine_map<(d0, d1) -> (d0, d1)>}
  %4 = df.mux %2 : !df.compute, %3 : !df.memory, %0, %1 {map = affine_map<(d0, d1) -> (d0, d1)>}
  %5 = df.interconnects "horizontal_links" %3 : !df.memory, %3 : !df.memory, %0, %1 {map = affine_map<(d0, d1) -> (d0 + 1, d1)>} : !df.interconnect
  %6 = df.interconnects "vertical_links" %3 : !df.memory, %3 : !df.memory, %0, %1 {map = affine_map<(d0, d1) -> (d0, d1 + 1)>} : !df.interconnect
  func.func @matmul_kernel__d0i0_d1i0_f0_f1_f2(%arg0: memref<*xf32> {tt.divisibility = 16 : i32}, %arg1: memref<*xf32> {tt.divisibility = 16 : i32}, %arg2: memref<*xf32> {tt.divisibility = 16 : i32}, %arg3: index {tt.divisibility = 16 : i32}, %arg4: index {tt.divisibility = 16 : i32}, %arg5: index {tt.divisibility = 16 : i32}, %arg6: index {tt.divisibility = 16 : i32}, %arg7: index {tt.divisibility = 16 : i32}, %arg8: index {tt.divisibility = 16 : i32}, %arg9: index, %arg10: index, %arg11: index) {
    affine.for %arg12 = 0 to affine_map<(d0, d1, d2) -> ((d0 ceildiv 8) ceildiv 8)>(%arg9, %arg10, %arg11) {
      affine.for %arg13 = 0 to affine_map<(d0, d1, d2) -> (d1)>(%arg9, %arg10, %arg11) {
        affine.for %arg14 = 0 to affine_map<(d0, d1, d2) -> (d2)>(%arg9, %arg10, %arg11) {
          affine.parallel (%arg15) = (0) to (8) {
            affine.parallel (%arg16) = (0) to (8) {
              %7 = affine.apply affine_map<(d0, d1, d2) -> (d0 + d1 * 64 + d2 * 8)>(%arg16, %arg12, %arg15)
              %cst = arith.constant 0.000000e+00 : f32
              %8 = tensor.empty() : tensor<64x64xf32>
              %9 = linalg.fill ins(%cst : f32) outs(%8 : tensor<64x64xf32>) -> tensor<64x64xf32>
              %c64 = arith.constant 64 : index
              %10 = arith.muli %7, %c64 : index
              %11 = affine.apply affine_map<()[s0] -> ((s0 + 31) floordiv 32)>()[%arg5]
              %c0 = arith.constant 0 : index
              %c1 = arith.constant 1 : index
              %12 = scf.for %arg17 = %c0 to %11 step %c1 iter_args(%arg18 = %9) -> (tensor<64x64xf32>) {
                %c32 = arith.constant 32 : index
                %19 = arith.muli %arg17, %c32 : index
                %20 = arith.muli %10, %arg6 : index
                %21 = affine.apply affine_map<(d0)[s0] -> (d0 * 32 + s0)>(%arg17)[%20]
                %reinterpret_cast_0 = memref.reinterpret_cast %arg0 to offset: [%21], sizes: [64, 32], strides: [%arg6, 1] {tmd.reuse = {sequential = [{depth = 5 : i64, iterator = "%arg17", reuse_type = "no_reuse", volume = 0 : i64}], spatial = [{depth = 3 : i64, iterator = "%arg15", mapped_to = "y", reuse_type = "no_reuse", volume = 0 : i64}, {depth = 4 : i64, iterator = "%arg16", mapped_to = "x", reuse_type = "no_reuse", volume = 0 : i64}], temporal = [{depth = 0 : i64, iterator = "%arg12", reuse_type = "no_reuse", volume = 0 : i64}, {depth = 1 : i64, iterator = "%arg13", reuse_type = "total_reuse", volume = 8192 : i64}, {depth = 2 : i64, iterator = "%arg14", reuse_type = "total_reuse", volume = 8192 : i64}]}} : memref<*xf32> to memref<64x32xf32, strided<[?, 1], offset: ?>>
                %alloc = memref.alloc() : memref<64x32xf32>
                memref.copy %reinterpret_cast_0, %alloc : memref<64x32xf32, strided<[?, 1], offset: ?>> to memref<64x32xf32>
                %22 = bufferization.to_tensor %alloc restrict writable : memref<64x32xf32> to tensor<64x32xf32>
                %23 = arith.muli %19, %arg7 : index
                %24 = affine.apply affine_map<(d0)[s0] -> (d0 * 64 + s0)>(%arg13)[%23]
                %reinterpret_cast_1 = memref.reinterpret_cast %arg1 to offset: [%24], sizes: [32, 64], strides: [%arg7, 1] {tmd.reuse = {sequential = [{depth = 5 : i64, iterator = "%arg17", reuse_type = "no_reuse", volume = 0 : i64}], spatial = [{depth = 3 : i64, iterator = "%arg15", mapped_to = "y", reuse_type = "total_reuse", volume = 8192 : i64}, {depth = 4 : i64, iterator = "%arg16", mapped_to = "x", reuse_type = "total_reuse", volume = 8192 : i64}], temporal = [{depth = 0 : i64, iterator = "%arg12", reuse_type = "total_reuse", volume = 8192 : i64}, {depth = 1 : i64, iterator = "%arg13", reuse_type = "no_reuse", volume = 0 : i64}, {depth = 2 : i64, iterator = "%arg14", reuse_type = "total_reuse", volume = 8192 : i64}]}} : memref<*xf32> to memref<32x64xf32, strided<[?, 1], offset: ?>>
                %alloc_2 = memref.alloc() : memref<32x64xf32>
                memref.copy %reinterpret_cast_1, %alloc_2 : memref<32x64xf32, strided<[?, 1], offset: ?>> to memref<32x64xf32>
                %25 = bufferization.to_tensor %alloc_2 restrict writable : memref<32x64xf32> to tensor<32x64xf32>
                %26 = linalg.matmul ins(%22, %25 : tensor<64x32xf32>, tensor<32x64xf32>) outs(%9 : tensor<64x64xf32>) -> tensor<64x64xf32>
                %27 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%arg18, %26 : tensor<64x64xf32>, tensor<64x64xf32>) outs(%arg18 : tensor<64x64xf32>) {
                ^bb0(%in: f32, %in_3: f32, %out: f32):
                  %28 = arith.addf %in, %in_3 : f32
                  linalg.yield %28 : f32
                } -> tensor<64x64xf32>
                scf.yield %27 : tensor<64x64xf32>
              }
              %13 = arith.muli %10, %arg8 : index
              %14 = affine.apply affine_map<(d0)[s0] -> (d0 * 64 + s0)>(%arg13)[%13]
              %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%14], sizes: [64, 64], strides: [%arg8, 1] {tmd.reuse = {spatial = [{depth = 3 : i64, iterator = "%arg15", mapped_to = "y", reuse_type = "no_reuse", volume = 0 : i64}, {depth = 4 : i64, iterator = "%arg16", mapped_to = "x", reuse_type = "no_reuse", volume = 0 : i64}], temporal = [{depth = 0 : i64, iterator = "%arg12", reuse_type = "no_reuse", volume = 0 : i64}, {depth = 1 : i64, iterator = "%arg13", reuse_type = "no_reuse", volume = 0 : i64}, {depth = 2 : i64, iterator = "%arg14", reuse_type = "total_reuse", volume = 16384 : i64}]}} : memref<*xf32> to memref<64x64xf32, strided<[?, 1], offset: ?>>
              %15 = affine.min affine_map<(d0, d1, d2)[s0] -> (d0 * -64 + s0, 64)>(%7, %arg13, %arg14)[%arg3]
              %16 = affine.min affine_map<(d0, d1, d2)[s0] -> (d1 * -64 + s0, 64)>(%7, %arg13, %arg14)[%arg4]
              %extracted_slice = tensor.extract_slice %12[0, 0] [%15, %16] [1, 1] : tensor<64x64xf32> to tensor<?x?xf32>
              %17 = affine.min affine_map<(d0, d1, d2)[s0] -> (d0 * -64 + s0, 64)>(%7, %arg13, %arg14)[%arg3]
              %18 = affine.min affine_map<(d0, d1, d2)[s0] -> (d1 * -64 + s0, 64)>(%7, %arg13, %arg14)[%arg4]
              %subview = memref.subview %reinterpret_cast[0, 0] [%17, %18] [1, 1] : memref<64x64xf32, strided<[?, 1], offset: ?>> to memref<?x?xf32, strided<[?, 1], offset: ?>>
              bufferization.materialize_in_destination %extracted_slice in writable %subview : (tensor<?x?xf32>, memref<?x?xf32, strided<[?, 1], offset: ?>>) -> ()
            } {tmd.mapped_to = "x"}
          } {tmd.mapped_to = "y"}
        }
      }
    }
    return
  }
  func.func @matmul_kernel__d0i0_d1i0_f0_f2_f1(%arg0: memref<*xf32> {tt.divisibility = 16 : i32}, %arg1: memref<*xf32> {tt.divisibility = 16 : i32}, %arg2: memref<*xf32> {tt.divisibility = 16 : i32}, %arg3: index {tt.divisibility = 16 : i32}, %arg4: index {tt.divisibility = 16 : i32}, %arg5: index {tt.divisibility = 16 : i32}, %arg6: index {tt.divisibility = 16 : i32}, %arg7: index {tt.divisibility = 16 : i32}, %arg8: index {tt.divisibility = 16 : i32}, %arg9: index, %arg10: index, %arg11: index) {
    affine.for %arg12 = 0 to affine_map<(d0, d1, d2) -> ((d0 ceildiv 8) ceildiv 8)>(%arg9, %arg10, %arg11) {
      affine.for %arg13 = 0 to affine_map<(d0, d1, d2) -> (d2)>(%arg9, %arg10, %arg11) {
        affine.for %arg14 = 0 to affine_map<(d0, d1, d2) -> (d1)>(%arg9, %arg10, %arg11) {
          affine.parallel (%arg15) = (0) to (8) {
            affine.parallel (%arg16) = (0) to (8) {
              %7 = affine.apply affine_map<(d0, d1, d2) -> (d0 + d1 * 64 + d2 * 8)>(%arg16, %arg12, %arg15)
              %cst = arith.constant 0.000000e+00 : f32
              %8 = tensor.empty() : tensor<64x64xf32>
              %9 = linalg.fill ins(%cst : f32) outs(%8 : tensor<64x64xf32>) -> tensor<64x64xf32>
              %c64 = arith.constant 64 : index
              %10 = arith.muli %7, %c64 : index
              %11 = affine.apply affine_map<()[s0] -> ((s0 + 31) floordiv 32)>()[%arg5]
              %c0 = arith.constant 0 : index
              %c1 = arith.constant 1 : index
              %12 = scf.for %arg17 = %c0 to %11 step %c1 iter_args(%arg18 = %9) -> (tensor<64x64xf32>) {
                %c32 = arith.constant 32 : index
                %19 = arith.muli %arg17, %c32 : index
                %20 = arith.muli %10, %arg6 : index
                %21 = affine.apply affine_map<(d0)[s0] -> (d0 * 32 + s0)>(%arg17)[%20]
                %reinterpret_cast_0 = memref.reinterpret_cast %arg0 to offset: [%21], sizes: [64, 32], strides: [%arg6, 1] {tmd.reuse = {sequential = [{depth = 5 : i64, iterator = "%arg17", reuse_type = "no_reuse", volume = 0 : i64}], spatial = [{depth = 3 : i64, iterator = "%arg15", mapped_to = "y", reuse_type = "no_reuse", volume = 0 : i64}, {depth = 4 : i64, iterator = "%arg16", mapped_to = "x", reuse_type = "no_reuse", volume = 0 : i64}], temporal = [{depth = 0 : i64, iterator = "%arg12", reuse_type = "no_reuse", volume = 0 : i64}, {depth = 1 : i64, iterator = "%arg13", reuse_type = "total_reuse", volume = 8192 : i64}, {depth = 2 : i64, iterator = "%arg14", reuse_type = "total_reuse", volume = 8192 : i64}]}} : memref<*xf32> to memref<64x32xf32, strided<[?, 1], offset: ?>>
                %alloc = memref.alloc() : memref<64x32xf32>
                memref.copy %reinterpret_cast_0, %alloc : memref<64x32xf32, strided<[?, 1], offset: ?>> to memref<64x32xf32>
                %22 = bufferization.to_tensor %alloc restrict writable : memref<64x32xf32> to tensor<64x32xf32>
                %23 = arith.muli %19, %arg7 : index
                %24 = affine.apply affine_map<(d0)[s0] -> (d0 * 64 + s0)>(%arg14)[%23]
                %reinterpret_cast_1 = memref.reinterpret_cast %arg1 to offset: [%24], sizes: [32, 64], strides: [%arg7, 1] {tmd.reuse = {sequential = [{depth = 5 : i64, iterator = "%arg17", reuse_type = "no_reuse", volume = 0 : i64}], spatial = [{depth = 3 : i64, iterator = "%arg15", mapped_to = "y", reuse_type = "total_reuse", volume = 8192 : i64}, {depth = 4 : i64, iterator = "%arg16", mapped_to = "x", reuse_type = "total_reuse", volume = 8192 : i64}], temporal = [{depth = 0 : i64, iterator = "%arg12", reuse_type = "total_reuse", volume = 8192 : i64}, {depth = 1 : i64, iterator = "%arg13", reuse_type = "total_reuse", volume = 8192 : i64}, {depth = 2 : i64, iterator = "%arg14", reuse_type = "no_reuse", volume = 0 : i64}]}} : memref<*xf32> to memref<32x64xf32, strided<[?, 1], offset: ?>>
                %alloc_2 = memref.alloc() : memref<32x64xf32>
                memref.copy %reinterpret_cast_1, %alloc_2 : memref<32x64xf32, strided<[?, 1], offset: ?>> to memref<32x64xf32>
                %25 = bufferization.to_tensor %alloc_2 restrict writable : memref<32x64xf32> to tensor<32x64xf32>
                %26 = linalg.matmul ins(%22, %25 : tensor<64x32xf32>, tensor<32x64xf32>) outs(%9 : tensor<64x64xf32>) -> tensor<64x64xf32>
                %27 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%arg18, %26 : tensor<64x64xf32>, tensor<64x64xf32>) outs(%arg18 : tensor<64x64xf32>) {
                ^bb0(%in: f32, %in_3: f32, %out: f32):
                  %28 = arith.addf %in, %in_3 : f32
                  linalg.yield %28 : f32
                } -> tensor<64x64xf32>
                scf.yield %27 : tensor<64x64xf32>
              }
              %13 = arith.muli %10, %arg8 : index
              %14 = affine.apply affine_map<(d0)[s0] -> (d0 * 64 + s0)>(%arg14)[%13]
              %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%14], sizes: [64, 64], strides: [%arg8, 1] {tmd.reuse = {spatial = [{depth = 3 : i64, iterator = "%arg15", mapped_to = "y", reuse_type = "no_reuse", volume = 0 : i64}, {depth = 4 : i64, iterator = "%arg16", mapped_to = "x", reuse_type = "no_reuse", volume = 0 : i64}], temporal = [{depth = 0 : i64, iterator = "%arg12", reuse_type = "no_reuse", volume = 0 : i64}, {depth = 1 : i64, iterator = "%arg13", reuse_type = "total_reuse", volume = 16384 : i64}, {depth = 2 : i64, iterator = "%arg14", reuse_type = "no_reuse", volume = 0 : i64}]}} : memref<*xf32> to memref<64x64xf32, strided<[?, 1], offset: ?>>
              %15 = affine.min affine_map<(d0, d1, d2)[s0] -> (d0 * -64 + s0, 64)>(%7, %arg14, %arg13)[%arg3]
              %16 = affine.min affine_map<(d0, d1, d2)[s0] -> (d1 * -64 + s0, 64)>(%7, %arg14, %arg13)[%arg4]
              %extracted_slice = tensor.extract_slice %12[0, 0] [%15, %16] [1, 1] : tensor<64x64xf32> to tensor<?x?xf32>
              %17 = affine.min affine_map<(d0, d1, d2)[s0] -> (d0 * -64 + s0, 64)>(%7, %arg14, %arg13)[%arg3]
              %18 = affine.min affine_map<(d0, d1, d2)[s0] -> (d1 * -64 + s0, 64)>(%7, %arg14, %arg13)[%arg4]
              %subview = memref.subview %reinterpret_cast[0, 0] [%17, %18] [1, 1] : memref<64x64xf32, strided<[?, 1], offset: ?>> to memref<?x?xf32, strided<[?, 1], offset: ?>>
              bufferization.materialize_in_destination %extracted_slice in writable %subview : (tensor<?x?xf32>, memref<?x?xf32, strided<[?, 1], offset: ?>>) -> ()
            } {tmd.mapped_to = "x"}
          } {tmd.mapped_to = "y"}
        }
      }
    }
    return
  }
  func.func @matmul_kernel__d0i0_d1i0_f1_f0_f2(%arg0: memref<*xf32> {tt.divisibility = 16 : i32}, %arg1: memref<*xf32> {tt.divisibility = 16 : i32}, %arg2: memref<*xf32> {tt.divisibility = 16 : i32}, %arg3: index {tt.divisibility = 16 : i32}, %arg4: index {tt.divisibility = 16 : i32}, %arg5: index {tt.divisibility = 16 : i32}, %arg6: index {tt.divisibility = 16 : i32}, %arg7: index {tt.divisibility = 16 : i32}, %arg8: index {tt.divisibility = 16 : i32}, %arg9: index, %arg10: index, %arg11: index) {
    affine.for %arg12 = 0 to affine_map<(d0, d1, d2) -> (d1)>(%arg9, %arg10, %arg11) {
      affine.for %arg13 = 0 to affine_map<(d0, d1, d2) -> ((d0 ceildiv 8) ceildiv 8)>(%arg9, %arg10, %arg11) {
        affine.for %arg14 = 0 to affine_map<(d0, d1, d2) -> (d2)>(%arg9, %arg10, %arg11) {
          affine.parallel (%arg15) = (0) to (8) {
            affine.parallel (%arg16) = (0) to (8) {
              %7 = affine.apply affine_map<(d0, d1, d2) -> (d0 + d1 * 64 + d2 * 8)>(%arg16, %arg13, %arg15)
              %cst = arith.constant 0.000000e+00 : f32
              %8 = tensor.empty() : tensor<64x64xf32>
              %9 = linalg.fill ins(%cst : f32) outs(%8 : tensor<64x64xf32>) -> tensor<64x64xf32>
              %c64 = arith.constant 64 : index
              %10 = arith.muli %7, %c64 : index
              %11 = affine.apply affine_map<()[s0] -> ((s0 + 31) floordiv 32)>()[%arg5]
              %c0 = arith.constant 0 : index
              %c1 = arith.constant 1 : index
              %12 = scf.for %arg17 = %c0 to %11 step %c1 iter_args(%arg18 = %9) -> (tensor<64x64xf32>) {
                %c32 = arith.constant 32 : index
                %19 = arith.muli %arg17, %c32 : index
                %20 = arith.muli %10, %arg6 : index
                %21 = affine.apply affine_map<(d0)[s0] -> (d0 * 32 + s0)>(%arg17)[%20]
                %reinterpret_cast_0 = memref.reinterpret_cast %arg0 to offset: [%21], sizes: [64, 32], strides: [%arg6, 1] {tmd.reuse = {sequential = [{depth = 5 : i64, iterator = "%arg17", reuse_type = "no_reuse", volume = 0 : i64}], spatial = [{depth = 3 : i64, iterator = "%arg15", mapped_to = "y", reuse_type = "no_reuse", volume = 0 : i64}, {depth = 4 : i64, iterator = "%arg16", mapped_to = "x", reuse_type = "no_reuse", volume = 0 : i64}], temporal = [{depth = 0 : i64, iterator = "%arg12", reuse_type = "total_reuse", volume = 8192 : i64}, {depth = 1 : i64, iterator = "%arg13", reuse_type = "no_reuse", volume = 0 : i64}, {depth = 2 : i64, iterator = "%arg14", reuse_type = "total_reuse", volume = 8192 : i64}]}} : memref<*xf32> to memref<64x32xf32, strided<[?, 1], offset: ?>>
                %alloc = memref.alloc() : memref<64x32xf32>
                memref.copy %reinterpret_cast_0, %alloc : memref<64x32xf32, strided<[?, 1], offset: ?>> to memref<64x32xf32>
                %22 = bufferization.to_tensor %alloc restrict writable : memref<64x32xf32> to tensor<64x32xf32>
                %23 = arith.muli %19, %arg7 : index
                %24 = affine.apply affine_map<(d0)[s0] -> (d0 * 64 + s0)>(%arg12)[%23]
                %reinterpret_cast_1 = memref.reinterpret_cast %arg1 to offset: [%24], sizes: [32, 64], strides: [%arg7, 1] {tmd.reuse = {sequential = [{depth = 5 : i64, iterator = "%arg17", reuse_type = "no_reuse", volume = 0 : i64}], spatial = [{depth = 3 : i64, iterator = "%arg15", mapped_to = "y", reuse_type = "total_reuse", volume = 8192 : i64}, {depth = 4 : i64, iterator = "%arg16", mapped_to = "x", reuse_type = "total_reuse", volume = 8192 : i64}], temporal = [{depth = 0 : i64, iterator = "%arg12", reuse_type = "no_reuse", volume = 0 : i64}, {depth = 1 : i64, iterator = "%arg13", reuse_type = "total_reuse", volume = 8192 : i64}, {depth = 2 : i64, iterator = "%arg14", reuse_type = "total_reuse", volume = 8192 : i64}]}} : memref<*xf32> to memref<32x64xf32, strided<[?, 1], offset: ?>>
                %alloc_2 = memref.alloc() : memref<32x64xf32>
                memref.copy %reinterpret_cast_1, %alloc_2 : memref<32x64xf32, strided<[?, 1], offset: ?>> to memref<32x64xf32>
                %25 = bufferization.to_tensor %alloc_2 restrict writable : memref<32x64xf32> to tensor<32x64xf32>
                %26 = linalg.matmul ins(%22, %25 : tensor<64x32xf32>, tensor<32x64xf32>) outs(%9 : tensor<64x64xf32>) -> tensor<64x64xf32>
                %27 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%arg18, %26 : tensor<64x64xf32>, tensor<64x64xf32>) outs(%arg18 : tensor<64x64xf32>) {
                ^bb0(%in: f32, %in_3: f32, %out: f32):
                  %28 = arith.addf %in, %in_3 : f32
                  linalg.yield %28 : f32
                } -> tensor<64x64xf32>
                scf.yield %27 : tensor<64x64xf32>
              }
              %13 = arith.muli %10, %arg8 : index
              %14 = affine.apply affine_map<(d0)[s0] -> (d0 * 64 + s0)>(%arg12)[%13]
              %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%14], sizes: [64, 64], strides: [%arg8, 1] {tmd.reuse = {spatial = [{depth = 3 : i64, iterator = "%arg15", mapped_to = "y", reuse_type = "no_reuse", volume = 0 : i64}, {depth = 4 : i64, iterator = "%arg16", mapped_to = "x", reuse_type = "no_reuse", volume = 0 : i64}], temporal = [{depth = 0 : i64, iterator = "%arg12", reuse_type = "no_reuse", volume = 0 : i64}, {depth = 1 : i64, iterator = "%arg13", reuse_type = "no_reuse", volume = 0 : i64}, {depth = 2 : i64, iterator = "%arg14", reuse_type = "total_reuse", volume = 16384 : i64}]}} : memref<*xf32> to memref<64x64xf32, strided<[?, 1], offset: ?>>
              %15 = affine.min affine_map<(d0, d1, d2)[s0] -> (d0 * -64 + s0, 64)>(%7, %arg12, %arg14)[%arg3]
              %16 = affine.min affine_map<(d0, d1, d2)[s0] -> (d1 * -64 + s0, 64)>(%7, %arg12, %arg14)[%arg4]
              %extracted_slice = tensor.extract_slice %12[0, 0] [%15, %16] [1, 1] : tensor<64x64xf32> to tensor<?x?xf32>
              %17 = affine.min affine_map<(d0, d1, d2)[s0] -> (d0 * -64 + s0, 64)>(%7, %arg12, %arg14)[%arg3]
              %18 = affine.min affine_map<(d0, d1, d2)[s0] -> (d1 * -64 + s0, 64)>(%7, %arg12, %arg14)[%arg4]
              %subview = memref.subview %reinterpret_cast[0, 0] [%17, %18] [1, 1] : memref<64x64xf32, strided<[?, 1], offset: ?>> to memref<?x?xf32, strided<[?, 1], offset: ?>>
              bufferization.materialize_in_destination %extracted_slice in writable %subview : (tensor<?x?xf32>, memref<?x?xf32, strided<[?, 1], offset: ?>>) -> ()
            } {tmd.mapped_to = "x"}
          } {tmd.mapped_to = "y"}
        }
      }
    }
    return
  }
  func.func @matmul_kernel__d0i0_d1i0_f1_f2_f0(%arg0: memref<*xf32> {tt.divisibility = 16 : i32}, %arg1: memref<*xf32> {tt.divisibility = 16 : i32}, %arg2: memref<*xf32> {tt.divisibility = 16 : i32}, %arg3: index {tt.divisibility = 16 : i32}, %arg4: index {tt.divisibility = 16 : i32}, %arg5: index {tt.divisibility = 16 : i32}, %arg6: index {tt.divisibility = 16 : i32}, %arg7: index {tt.divisibility = 16 : i32}, %arg8: index {tt.divisibility = 16 : i32}, %arg9: index, %arg10: index, %arg11: index) {
    affine.for %arg12 = 0 to affine_map<(d0, d1, d2) -> (d1)>(%arg9, %arg10, %arg11) {
      affine.for %arg13 = 0 to affine_map<(d0, d1, d2) -> (d2)>(%arg9, %arg10, %arg11) {
        affine.for %arg14 = 0 to affine_map<(d0, d1, d2) -> ((d0 ceildiv 8) ceildiv 8)>(%arg9, %arg10, %arg11) {
          affine.parallel (%arg15) = (0) to (8) {
            affine.parallel (%arg16) = (0) to (8) {
              %7 = affine.apply affine_map<(d0, d1, d2) -> (d0 + d1 * 64 + d2 * 8)>(%arg16, %arg14, %arg15)
              %cst = arith.constant 0.000000e+00 : f32
              %8 = tensor.empty() : tensor<64x64xf32>
              %9 = linalg.fill ins(%cst : f32) outs(%8 : tensor<64x64xf32>) -> tensor<64x64xf32>
              %c64 = arith.constant 64 : index
              %10 = arith.muli %7, %c64 : index
              %11 = affine.apply affine_map<()[s0] -> ((s0 + 31) floordiv 32)>()[%arg5]
              %c0 = arith.constant 0 : index
              %c1 = arith.constant 1 : index
              %12 = scf.for %arg17 = %c0 to %11 step %c1 iter_args(%arg18 = %9) -> (tensor<64x64xf32>) {
                %c32 = arith.constant 32 : index
                %19 = arith.muli %arg17, %c32 : index
                %20 = arith.muli %10, %arg6 : index
                %21 = affine.apply affine_map<(d0)[s0] -> (d0 * 32 + s0)>(%arg17)[%20]
                %reinterpret_cast_0 = memref.reinterpret_cast %arg0 to offset: [%21], sizes: [64, 32], strides: [%arg6, 1] {tmd.reuse = {sequential = [{depth = 5 : i64, iterator = "%arg17", reuse_type = "no_reuse", volume = 0 : i64}], spatial = [{depth = 3 : i64, iterator = "%arg15", mapped_to = "y", reuse_type = "no_reuse", volume = 0 : i64}, {depth = 4 : i64, iterator = "%arg16", mapped_to = "x", reuse_type = "no_reuse", volume = 0 : i64}], temporal = [{depth = 0 : i64, iterator = "%arg12", reuse_type = "total_reuse", volume = 8192 : i64}, {depth = 1 : i64, iterator = "%arg13", reuse_type = "total_reuse", volume = 8192 : i64}, {depth = 2 : i64, iterator = "%arg14", reuse_type = "no_reuse", volume = 0 : i64}]}} : memref<*xf32> to memref<64x32xf32, strided<[?, 1], offset: ?>>
                %alloc = memref.alloc() : memref<64x32xf32>
                memref.copy %reinterpret_cast_0, %alloc : memref<64x32xf32, strided<[?, 1], offset: ?>> to memref<64x32xf32>
                %22 = bufferization.to_tensor %alloc restrict writable : memref<64x32xf32> to tensor<64x32xf32>
                %23 = arith.muli %19, %arg7 : index
                %24 = affine.apply affine_map<(d0)[s0] -> (d0 * 64 + s0)>(%arg12)[%23]
                %reinterpret_cast_1 = memref.reinterpret_cast %arg1 to offset: [%24], sizes: [32, 64], strides: [%arg7, 1] {tmd.reuse = {sequential = [{depth = 5 : i64, iterator = "%arg17", reuse_type = "no_reuse", volume = 0 : i64}], spatial = [{depth = 3 : i64, iterator = "%arg15", mapped_to = "y", reuse_type = "total_reuse", volume = 8192 : i64}, {depth = 4 : i64, iterator = "%arg16", mapped_to = "x", reuse_type = "total_reuse", volume = 8192 : i64}], temporal = [{depth = 0 : i64, iterator = "%arg12", reuse_type = "no_reuse", volume = 0 : i64}, {depth = 1 : i64, iterator = "%arg13", reuse_type = "total_reuse", volume = 8192 : i64}, {depth = 2 : i64, iterator = "%arg14", reuse_type = "total_reuse", volume = 8192 : i64}]}} : memref<*xf32> to memref<32x64xf32, strided<[?, 1], offset: ?>>
                %alloc_2 = memref.alloc() : memref<32x64xf32>
                memref.copy %reinterpret_cast_1, %alloc_2 : memref<32x64xf32, strided<[?, 1], offset: ?>> to memref<32x64xf32>
                %25 = bufferization.to_tensor %alloc_2 restrict writable : memref<32x64xf32> to tensor<32x64xf32>
                %26 = linalg.matmul ins(%22, %25 : tensor<64x32xf32>, tensor<32x64xf32>) outs(%9 : tensor<64x64xf32>) -> tensor<64x64xf32>
                %27 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%arg18, %26 : tensor<64x64xf32>, tensor<64x64xf32>) outs(%arg18 : tensor<64x64xf32>) {
                ^bb0(%in: f32, %in_3: f32, %out: f32):
                  %28 = arith.addf %in, %in_3 : f32
                  linalg.yield %28 : f32
                } -> tensor<64x64xf32>
                scf.yield %27 : tensor<64x64xf32>
              }
              %13 = arith.muli %10, %arg8 : index
              %14 = affine.apply affine_map<(d0)[s0] -> (d0 * 64 + s0)>(%arg12)[%13]
              %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%14], sizes: [64, 64], strides: [%arg8, 1] {tmd.reuse = {spatial = [{depth = 3 : i64, iterator = "%arg15", mapped_to = "y", reuse_type = "no_reuse", volume = 0 : i64}, {depth = 4 : i64, iterator = "%arg16", mapped_to = "x", reuse_type = "no_reuse", volume = 0 : i64}], temporal = [{depth = 0 : i64, iterator = "%arg12", reuse_type = "no_reuse", volume = 0 : i64}, {depth = 1 : i64, iterator = "%arg13", reuse_type = "total_reuse", volume = 16384 : i64}, {depth = 2 : i64, iterator = "%arg14", reuse_type = "no_reuse", volume = 0 : i64}]}} : memref<*xf32> to memref<64x64xf32, strided<[?, 1], offset: ?>>
              %15 = affine.min affine_map<(d0, d1, d2)[s0] -> (d0 * -64 + s0, 64)>(%7, %arg12, %arg13)[%arg3]
              %16 = affine.min affine_map<(d0, d1, d2)[s0] -> (d1 * -64 + s0, 64)>(%7, %arg12, %arg13)[%arg4]
              %extracted_slice = tensor.extract_slice %12[0, 0] [%15, %16] [1, 1] : tensor<64x64xf32> to tensor<?x?xf32>
              %17 = affine.min affine_map<(d0, d1, d2)[s0] -> (d0 * -64 + s0, 64)>(%7, %arg12, %arg13)[%arg3]
              %18 = affine.min affine_map<(d0, d1, d2)[s0] -> (d1 * -64 + s0, 64)>(%7, %arg12, %arg13)[%arg4]
              %subview = memref.subview %reinterpret_cast[0, 0] [%17, %18] [1, 1] : memref<64x64xf32, strided<[?, 1], offset: ?>> to memref<?x?xf32, strided<[?, 1], offset: ?>>
              bufferization.materialize_in_destination %extracted_slice in writable %subview : (tensor<?x?xf32>, memref<?x?xf32, strided<[?, 1], offset: ?>>) -> ()
            } {tmd.mapped_to = "x"}
          } {tmd.mapped_to = "y"}
        }
      }
    }
    return
  }
  func.func @matmul_kernel__d0i0_d1i0_f2_f0_f1(%arg0: memref<*xf32> {tt.divisibility = 16 : i32}, %arg1: memref<*xf32> {tt.divisibility = 16 : i32}, %arg2: memref<*xf32> {tt.divisibility = 16 : i32}, %arg3: index {tt.divisibility = 16 : i32}, %arg4: index {tt.divisibility = 16 : i32}, %arg5: index {tt.divisibility = 16 : i32}, %arg6: index {tt.divisibility = 16 : i32}, %arg7: index {tt.divisibility = 16 : i32}, %arg8: index {tt.divisibility = 16 : i32}, %arg9: index, %arg10: index, %arg11: index) {
    affine.for %arg12 = 0 to affine_map<(d0, d1, d2) -> (d2)>(%arg9, %arg10, %arg11) {
      affine.for %arg13 = 0 to affine_map<(d0, d1, d2) -> ((d0 ceildiv 8) ceildiv 8)>(%arg9, %arg10, %arg11) {
        affine.for %arg14 = 0 to affine_map<(d0, d1, d2) -> (d1)>(%arg9, %arg10, %arg11) {
          affine.parallel (%arg15) = (0) to (8) {
            affine.parallel (%arg16) = (0) to (8) {
              %7 = affine.apply affine_map<(d0, d1, d2) -> (d0 + d1 * 64 + d2 * 8)>(%arg16, %arg13, %arg15)
              %cst = arith.constant 0.000000e+00 : f32
              %8 = tensor.empty() : tensor<64x64xf32>
              %9 = linalg.fill ins(%cst : f32) outs(%8 : tensor<64x64xf32>) -> tensor<64x64xf32>
              %c64 = arith.constant 64 : index
              %10 = arith.muli %7, %c64 : index
              %11 = affine.apply affine_map<()[s0] -> ((s0 + 31) floordiv 32)>()[%arg5]
              %c0 = arith.constant 0 : index
              %c1 = arith.constant 1 : index
              %12 = scf.for %arg17 = %c0 to %11 step %c1 iter_args(%arg18 = %9) -> (tensor<64x64xf32>) {
                %c32 = arith.constant 32 : index
                %19 = arith.muli %arg17, %c32 : index
                %20 = arith.muli %10, %arg6 : index
                %21 = affine.apply affine_map<(d0)[s0] -> (d0 * 32 + s0)>(%arg17)[%20]
                %reinterpret_cast_0 = memref.reinterpret_cast %arg0 to offset: [%21], sizes: [64, 32], strides: [%arg6, 1] {tmd.reuse = {sequential = [{depth = 5 : i64, iterator = "%arg17", reuse_type = "no_reuse", volume = 0 : i64}], spatial = [{depth = 3 : i64, iterator = "%arg15", mapped_to = "y", reuse_type = "no_reuse", volume = 0 : i64}, {depth = 4 : i64, iterator = "%arg16", mapped_to = "x", reuse_type = "no_reuse", volume = 0 : i64}], temporal = [{depth = 0 : i64, iterator = "%arg12", reuse_type = "total_reuse", volume = 8192 : i64}, {depth = 1 : i64, iterator = "%arg13", reuse_type = "no_reuse", volume = 0 : i64}, {depth = 2 : i64, iterator = "%arg14", reuse_type = "total_reuse", volume = 8192 : i64}]}} : memref<*xf32> to memref<64x32xf32, strided<[?, 1], offset: ?>>
                %alloc = memref.alloc() : memref<64x32xf32>
                memref.copy %reinterpret_cast_0, %alloc : memref<64x32xf32, strided<[?, 1], offset: ?>> to memref<64x32xf32>
                %22 = bufferization.to_tensor %alloc restrict writable : memref<64x32xf32> to tensor<64x32xf32>
                %23 = arith.muli %19, %arg7 : index
                %24 = affine.apply affine_map<(d0)[s0] -> (d0 * 64 + s0)>(%arg14)[%23]
                %reinterpret_cast_1 = memref.reinterpret_cast %arg1 to offset: [%24], sizes: [32, 64], strides: [%arg7, 1] {tmd.reuse = {sequential = [{depth = 5 : i64, iterator = "%arg17", reuse_type = "no_reuse", volume = 0 : i64}], spatial = [{depth = 3 : i64, iterator = "%arg15", mapped_to = "y", reuse_type = "total_reuse", volume = 8192 : i64}, {depth = 4 : i64, iterator = "%arg16", mapped_to = "x", reuse_type = "total_reuse", volume = 8192 : i64}], temporal = [{depth = 0 : i64, iterator = "%arg12", reuse_type = "total_reuse", volume = 8192 : i64}, {depth = 1 : i64, iterator = "%arg13", reuse_type = "total_reuse", volume = 8192 : i64}, {depth = 2 : i64, iterator = "%arg14", reuse_type = "no_reuse", volume = 0 : i64}]}} : memref<*xf32> to memref<32x64xf32, strided<[?, 1], offset: ?>>
                %alloc_2 = memref.alloc() : memref<32x64xf32>
                memref.copy %reinterpret_cast_1, %alloc_2 : memref<32x64xf32, strided<[?, 1], offset: ?>> to memref<32x64xf32>
                %25 = bufferization.to_tensor %alloc_2 restrict writable : memref<32x64xf32> to tensor<32x64xf32>
                %26 = linalg.matmul ins(%22, %25 : tensor<64x32xf32>, tensor<32x64xf32>) outs(%9 : tensor<64x64xf32>) -> tensor<64x64xf32>
                %27 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%arg18, %26 : tensor<64x64xf32>, tensor<64x64xf32>) outs(%arg18 : tensor<64x64xf32>) {
                ^bb0(%in: f32, %in_3: f32, %out: f32):
                  %28 = arith.addf %in, %in_3 : f32
                  linalg.yield %28 : f32
                } -> tensor<64x64xf32>
                scf.yield %27 : tensor<64x64xf32>
              }
              %13 = arith.muli %10, %arg8 : index
              %14 = affine.apply affine_map<(d0)[s0] -> (d0 * 64 + s0)>(%arg14)[%13]
              %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%14], sizes: [64, 64], strides: [%arg8, 1] {tmd.reuse = {spatial = [{depth = 3 : i64, iterator = "%arg15", mapped_to = "y", reuse_type = "no_reuse", volume = 0 : i64}, {depth = 4 : i64, iterator = "%arg16", mapped_to = "x", reuse_type = "no_reuse", volume = 0 : i64}], temporal = [{depth = 0 : i64, iterator = "%arg12", reuse_type = "total_reuse", volume = 16384 : i64}, {depth = 1 : i64, iterator = "%arg13", reuse_type = "no_reuse", volume = 0 : i64}, {depth = 2 : i64, iterator = "%arg14", reuse_type = "no_reuse", volume = 0 : i64}]}} : memref<*xf32> to memref<64x64xf32, strided<[?, 1], offset: ?>>
              %15 = affine.min affine_map<(d0, d1, d2)[s0] -> (d0 * -64 + s0, 64)>(%7, %arg14, %arg12)[%arg3]
              %16 = affine.min affine_map<(d0, d1, d2)[s0] -> (d1 * -64 + s0, 64)>(%7, %arg14, %arg12)[%arg4]
              %extracted_slice = tensor.extract_slice %12[0, 0] [%15, %16] [1, 1] : tensor<64x64xf32> to tensor<?x?xf32>
              %17 = affine.min affine_map<(d0, d1, d2)[s0] -> (d0 * -64 + s0, 64)>(%7, %arg14, %arg12)[%arg3]
              %18 = affine.min affine_map<(d0, d1, d2)[s0] -> (d1 * -64 + s0, 64)>(%7, %arg14, %arg12)[%arg4]
              %subview = memref.subview %reinterpret_cast[0, 0] [%17, %18] [1, 1] : memref<64x64xf32, strided<[?, 1], offset: ?>> to memref<?x?xf32, strided<[?, 1], offset: ?>>
              bufferization.materialize_in_destination %extracted_slice in writable %subview : (tensor<?x?xf32>, memref<?x?xf32, strided<[?, 1], offset: ?>>) -> ()
            } {tmd.mapped_to = "x"}
          } {tmd.mapped_to = "y"}
        }
      }
    }
    return
  }
  func.func @matmul_kernel__d0i0_d1i0_f2_f1_f0(%arg0: memref<*xf32> {tt.divisibility = 16 : i32}, %arg1: memref<*xf32> {tt.divisibility = 16 : i32}, %arg2: memref<*xf32> {tt.divisibility = 16 : i32}, %arg3: index {tt.divisibility = 16 : i32}, %arg4: index {tt.divisibility = 16 : i32}, %arg5: index {tt.divisibility = 16 : i32}, %arg6: index {tt.divisibility = 16 : i32}, %arg7: index {tt.divisibility = 16 : i32}, %arg8: index {tt.divisibility = 16 : i32}, %arg9: index, %arg10: index, %arg11: index) {
    affine.for %arg12 = 0 to affine_map<(d0, d1, d2) -> (d2)>(%arg9, %arg10, %arg11) {
      affine.for %arg13 = 0 to affine_map<(d0, d1, d2) -> (d1)>(%arg9, %arg10, %arg11) {
        affine.for %arg14 = 0 to affine_map<(d0, d1, d2) -> ((d0 ceildiv 8) ceildiv 8)>(%arg9, %arg10, %arg11) {
          affine.parallel (%arg15) = (0) to (8) {
            affine.parallel (%arg16) = (0) to (8) {
              %7 = affine.apply affine_map<(d0, d1, d2) -> (d0 + d1 * 64 + d2 * 8)>(%arg16, %arg14, %arg15)
              %cst = arith.constant 0.000000e+00 : f32
              %8 = tensor.empty() : tensor<64x64xf32>
              %9 = linalg.fill ins(%cst : f32) outs(%8 : tensor<64x64xf32>) -> tensor<64x64xf32>
              %c64 = arith.constant 64 : index
              %10 = arith.muli %7, %c64 : index
              %11 = affine.apply affine_map<()[s0] -> ((s0 + 31) floordiv 32)>()[%arg5]
              %c0 = arith.constant 0 : index
              %c1 = arith.constant 1 : index
              %12 = scf.for %arg17 = %c0 to %11 step %c1 iter_args(%arg18 = %9) -> (tensor<64x64xf32>) {
                %c32 = arith.constant 32 : index
                %19 = arith.muli %arg17, %c32 : index
                %20 = arith.muli %10, %arg6 : index
                %21 = affine.apply affine_map<(d0)[s0] -> (d0 * 32 + s0)>(%arg17)[%20]
                %reinterpret_cast_0 = memref.reinterpret_cast %arg0 to offset: [%21], sizes: [64, 32], strides: [%arg6, 1] {tmd.reuse = {sequential = [{depth = 5 : i64, iterator = "%arg17", reuse_type = "no_reuse", volume = 0 : i64}], spatial = [{depth = 3 : i64, iterator = "%arg15", mapped_to = "y", reuse_type = "no_reuse", volume = 0 : i64}, {depth = 4 : i64, iterator = "%arg16", mapped_to = "x", reuse_type = "no_reuse", volume = 0 : i64}], temporal = [{depth = 0 : i64, iterator = "%arg12", reuse_type = "total_reuse", volume = 8192 : i64}, {depth = 1 : i64, iterator = "%arg13", reuse_type = "total_reuse", volume = 8192 : i64}, {depth = 2 : i64, iterator = "%arg14", reuse_type = "no_reuse", volume = 0 : i64}]}} : memref<*xf32> to memref<64x32xf32, strided<[?, 1], offset: ?>>
                %alloc = memref.alloc() : memref<64x32xf32>
                memref.copy %reinterpret_cast_0, %alloc : memref<64x32xf32, strided<[?, 1], offset: ?>> to memref<64x32xf32>
                %22 = bufferization.to_tensor %alloc restrict writable : memref<64x32xf32> to tensor<64x32xf32>
                %23 = arith.muli %19, %arg7 : index
                %24 = affine.apply affine_map<(d0)[s0] -> (d0 * 64 + s0)>(%arg13)[%23]
                %reinterpret_cast_1 = memref.reinterpret_cast %arg1 to offset: [%24], sizes: [32, 64], strides: [%arg7, 1] {tmd.reuse = {sequential = [{depth = 5 : i64, iterator = "%arg17", reuse_type = "no_reuse", volume = 0 : i64}], spatial = [{depth = 3 : i64, iterator = "%arg15", mapped_to = "y", reuse_type = "total_reuse", volume = 8192 : i64}, {depth = 4 : i64, iterator = "%arg16", mapped_to = "x", reuse_type = "total_reuse", volume = 8192 : i64}], temporal = [{depth = 0 : i64, iterator = "%arg12", reuse_type = "total_reuse", volume = 8192 : i64}, {depth = 1 : i64, iterator = "%arg13", reuse_type = "no_reuse", volume = 0 : i64}, {depth = 2 : i64, iterator = "%arg14", reuse_type = "total_reuse", volume = 8192 : i64}]}} : memref<*xf32> to memref<32x64xf32, strided<[?, 1], offset: ?>>
                %alloc_2 = memref.alloc() : memref<32x64xf32>
                memref.copy %reinterpret_cast_1, %alloc_2 : memref<32x64xf32, strided<[?, 1], offset: ?>> to memref<32x64xf32>
                %25 = bufferization.to_tensor %alloc_2 restrict writable : memref<32x64xf32> to tensor<32x64xf32>
                %26 = linalg.matmul ins(%22, %25 : tensor<64x32xf32>, tensor<32x64xf32>) outs(%9 : tensor<64x64xf32>) -> tensor<64x64xf32>
                %27 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%arg18, %26 : tensor<64x64xf32>, tensor<64x64xf32>) outs(%arg18 : tensor<64x64xf32>) {
                ^bb0(%in: f32, %in_3: f32, %out: f32):
                  %28 = arith.addf %in, %in_3 : f32
                  linalg.yield %28 : f32
                } -> tensor<64x64xf32>
                scf.yield %27 : tensor<64x64xf32>
              }
              %13 = arith.muli %10, %arg8 : index
              %14 = affine.apply affine_map<(d0)[s0] -> (d0 * 64 + s0)>(%arg13)[%13]
              %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%14], sizes: [64, 64], strides: [%arg8, 1] {tmd.reuse = {spatial = [{depth = 3 : i64, iterator = "%arg15", mapped_to = "y", reuse_type = "no_reuse", volume = 0 : i64}, {depth = 4 : i64, iterator = "%arg16", mapped_to = "x", reuse_type = "no_reuse", volume = 0 : i64}], temporal = [{depth = 0 : i64, iterator = "%arg12", reuse_type = "total_reuse", volume = 16384 : i64}, {depth = 1 : i64, iterator = "%arg13", reuse_type = "no_reuse", volume = 0 : i64}, {depth = 2 : i64, iterator = "%arg14", reuse_type = "no_reuse", volume = 0 : i64}]}} : memref<*xf32> to memref<64x64xf32, strided<[?, 1], offset: ?>>
              %15 = affine.min affine_map<(d0, d1, d2)[s0] -> (d0 * -64 + s0, 64)>(%7, %arg13, %arg12)[%arg3]
              %16 = affine.min affine_map<(d0, d1, d2)[s0] -> (d1 * -64 + s0, 64)>(%7, %arg13, %arg12)[%arg4]
              %extracted_slice = tensor.extract_slice %12[0, 0] [%15, %16] [1, 1] : tensor<64x64xf32> to tensor<?x?xf32>
              %17 = affine.min affine_map<(d0, d1, d2)[s0] -> (d0 * -64 + s0, 64)>(%7, %arg13, %arg12)[%arg3]
              %18 = affine.min affine_map<(d0, d1, d2)[s0] -> (d1 * -64 + s0, 64)>(%7, %arg13, %arg12)[%arg4]
              %subview = memref.subview %reinterpret_cast[0, 0] [%17, %18] [1, 1] : memref<64x64xf32, strided<[?, 1], offset: ?>> to memref<?x?xf32, strided<[?, 1], offset: ?>>
              bufferization.materialize_in_destination %extracted_slice in writable %subview : (tensor<?x?xf32>, memref<?x?xf32, strided<[?, 1], offset: ?>>) -> ()
            } {tmd.mapped_to = "x"}
          } {tmd.mapped_to = "y"}
        }
      }
    }
    return
  }
  func.func @matmul_kernel__d1i0_d0i0_f0_f1_f2(%arg0: memref<*xf32> {tt.divisibility = 16 : i32}, %arg1: memref<*xf32> {tt.divisibility = 16 : i32}, %arg2: memref<*xf32> {tt.divisibility = 16 : i32}, %arg3: index {tt.divisibility = 16 : i32}, %arg4: index {tt.divisibility = 16 : i32}, %arg5: index {tt.divisibility = 16 : i32}, %arg6: index {tt.divisibility = 16 : i32}, %arg7: index {tt.divisibility = 16 : i32}, %arg8: index {tt.divisibility = 16 : i32}, %arg9: index, %arg10: index, %arg11: index) {
    affine.for %arg12 = 0 to affine_map<(d0, d1, d2) -> ((d0 ceildiv 8) ceildiv 8)>(%arg9, %arg10, %arg11) {
      affine.for %arg13 = 0 to affine_map<(d0, d1, d2) -> (d1)>(%arg9, %arg10, %arg11) {
        affine.for %arg14 = 0 to affine_map<(d0, d1, d2) -> (d2)>(%arg9, %arg10, %arg11) {
          affine.parallel (%arg15) = (0) to (8) {
            affine.parallel (%arg16) = (0) to (8) {
              %7 = affine.apply affine_map<(d0, d1, d2) -> (d0 + d1 * 64 + d2 * 8)>(%arg16, %arg12, %arg15)
              %cst = arith.constant 0.000000e+00 : f32
              %8 = tensor.empty() : tensor<64x64xf32>
              %9 = linalg.fill ins(%cst : f32) outs(%8 : tensor<64x64xf32>) -> tensor<64x64xf32>
              %c64 = arith.constant 64 : index
              %10 = arith.muli %7, %c64 : index
              %11 = affine.apply affine_map<()[s0] -> ((s0 + 31) floordiv 32)>()[%arg5]
              %c0 = arith.constant 0 : index
              %c1 = arith.constant 1 : index
              %12 = scf.for %arg17 = %c0 to %11 step %c1 iter_args(%arg18 = %9) -> (tensor<64x64xf32>) {
                %c32 = arith.constant 32 : index
                %19 = arith.muli %arg17, %c32 : index
                %20 = arith.muli %10, %arg6 : index
                %21 = affine.apply affine_map<(d0)[s0] -> (d0 * 32 + s0)>(%arg17)[%20]
                %reinterpret_cast_0 = memref.reinterpret_cast %arg0 to offset: [%21], sizes: [64, 32], strides: [%arg6, 1] {tmd.reuse = {sequential = [{depth = 5 : i64, iterator = "%arg17", reuse_type = "no_reuse", volume = 0 : i64}], spatial = [{depth = 3 : i64, iterator = "%arg15", mapped_to = "x", reuse_type = "no_reuse", volume = 0 : i64}, {depth = 4 : i64, iterator = "%arg16", mapped_to = "y", reuse_type = "no_reuse", volume = 0 : i64}], temporal = [{depth = 0 : i64, iterator = "%arg12", reuse_type = "no_reuse", volume = 0 : i64}, {depth = 1 : i64, iterator = "%arg13", reuse_type = "total_reuse", volume = 8192 : i64}, {depth = 2 : i64, iterator = "%arg14", reuse_type = "total_reuse", volume = 8192 : i64}]}} : memref<*xf32> to memref<64x32xf32, strided<[?, 1], offset: ?>>
                %alloc = memref.alloc() : memref<64x32xf32>
                memref.copy %reinterpret_cast_0, %alloc : memref<64x32xf32, strided<[?, 1], offset: ?>> to memref<64x32xf32>
                %22 = bufferization.to_tensor %alloc restrict writable : memref<64x32xf32> to tensor<64x32xf32>
                %23 = arith.muli %19, %arg7 : index
                %24 = affine.apply affine_map<(d0)[s0] -> (d0 * 64 + s0)>(%arg13)[%23]
                %reinterpret_cast_1 = memref.reinterpret_cast %arg1 to offset: [%24], sizes: [32, 64], strides: [%arg7, 1] {tmd.reuse = {sequential = [{depth = 5 : i64, iterator = "%arg17", reuse_type = "no_reuse", volume = 0 : i64}], spatial = [{depth = 3 : i64, iterator = "%arg15", mapped_to = "x", reuse_type = "total_reuse", volume = 8192 : i64}, {depth = 4 : i64, iterator = "%arg16", mapped_to = "y", reuse_type = "total_reuse", volume = 8192 : i64}], temporal = [{depth = 0 : i64, iterator = "%arg12", reuse_type = "total_reuse", volume = 8192 : i64}, {depth = 1 : i64, iterator = "%arg13", reuse_type = "no_reuse", volume = 0 : i64}, {depth = 2 : i64, iterator = "%arg14", reuse_type = "total_reuse", volume = 8192 : i64}]}} : memref<*xf32> to memref<32x64xf32, strided<[?, 1], offset: ?>>
                %alloc_2 = memref.alloc() : memref<32x64xf32>
                memref.copy %reinterpret_cast_1, %alloc_2 : memref<32x64xf32, strided<[?, 1], offset: ?>> to memref<32x64xf32>
                %25 = bufferization.to_tensor %alloc_2 restrict writable : memref<32x64xf32> to tensor<32x64xf32>
                %26 = linalg.matmul ins(%22, %25 : tensor<64x32xf32>, tensor<32x64xf32>) outs(%9 : tensor<64x64xf32>) -> tensor<64x64xf32>
                %27 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%arg18, %26 : tensor<64x64xf32>, tensor<64x64xf32>) outs(%arg18 : tensor<64x64xf32>) {
                ^bb0(%in: f32, %in_3: f32, %out: f32):
                  %28 = arith.addf %in, %in_3 : f32
                  linalg.yield %28 : f32
                } -> tensor<64x64xf32>
                scf.yield %27 : tensor<64x64xf32>
              }
              %13 = arith.muli %10, %arg8 : index
              %14 = affine.apply affine_map<(d0)[s0] -> (d0 * 64 + s0)>(%arg13)[%13]
              %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%14], sizes: [64, 64], strides: [%arg8, 1] {tmd.reuse = {spatial = [{depth = 3 : i64, iterator = "%arg15", mapped_to = "x", reuse_type = "no_reuse", volume = 0 : i64}, {depth = 4 : i64, iterator = "%arg16", mapped_to = "y", reuse_type = "no_reuse", volume = 0 : i64}], temporal = [{depth = 0 : i64, iterator = "%arg12", reuse_type = "no_reuse", volume = 0 : i64}, {depth = 1 : i64, iterator = "%arg13", reuse_type = "no_reuse", volume = 0 : i64}, {depth = 2 : i64, iterator = "%arg14", reuse_type = "total_reuse", volume = 16384 : i64}]}} : memref<*xf32> to memref<64x64xf32, strided<[?, 1], offset: ?>>
              %15 = affine.min affine_map<(d0, d1, d2)[s0] -> (d0 * -64 + s0, 64)>(%7, %arg13, %arg14)[%arg3]
              %16 = affine.min affine_map<(d0, d1, d2)[s0] -> (d1 * -64 + s0, 64)>(%7, %arg13, %arg14)[%arg4]
              %extracted_slice = tensor.extract_slice %12[0, 0] [%15, %16] [1, 1] : tensor<64x64xf32> to tensor<?x?xf32>
              %17 = affine.min affine_map<(d0, d1, d2)[s0] -> (d0 * -64 + s0, 64)>(%7, %arg13, %arg14)[%arg3]
              %18 = affine.min affine_map<(d0, d1, d2)[s0] -> (d1 * -64 + s0, 64)>(%7, %arg13, %arg14)[%arg4]
              %subview = memref.subview %reinterpret_cast[0, 0] [%17, %18] [1, 1] : memref<64x64xf32, strided<[?, 1], offset: ?>> to memref<?x?xf32, strided<[?, 1], offset: ?>>
              bufferization.materialize_in_destination %extracted_slice in writable %subview : (tensor<?x?xf32>, memref<?x?xf32, strided<[?, 1], offset: ?>>) -> ()
            } {tmd.mapped_to = "y"}
          } {tmd.mapped_to = "x"}
        }
      }
    }
    return
  }
  func.func @matmul_kernel__d1i0_d0i0_f0_f2_f1(%arg0: memref<*xf32> {tt.divisibility = 16 : i32}, %arg1: memref<*xf32> {tt.divisibility = 16 : i32}, %arg2: memref<*xf32> {tt.divisibility = 16 : i32}, %arg3: index {tt.divisibility = 16 : i32}, %arg4: index {tt.divisibility = 16 : i32}, %arg5: index {tt.divisibility = 16 : i32}, %arg6: index {tt.divisibility = 16 : i32}, %arg7: index {tt.divisibility = 16 : i32}, %arg8: index {tt.divisibility = 16 : i32}, %arg9: index, %arg10: index, %arg11: index) {
    affine.for %arg12 = 0 to affine_map<(d0, d1, d2) -> ((d0 ceildiv 8) ceildiv 8)>(%arg9, %arg10, %arg11) {
      affine.for %arg13 = 0 to affine_map<(d0, d1, d2) -> (d2)>(%arg9, %arg10, %arg11) {
        affine.for %arg14 = 0 to affine_map<(d0, d1, d2) -> (d1)>(%arg9, %arg10, %arg11) {
          affine.parallel (%arg15) = (0) to (8) {
            affine.parallel (%arg16) = (0) to (8) {
              %7 = affine.apply affine_map<(d0, d1, d2) -> (d0 + d1 * 64 + d2 * 8)>(%arg16, %arg12, %arg15)
              %cst = arith.constant 0.000000e+00 : f32
              %8 = tensor.empty() : tensor<64x64xf32>
              %9 = linalg.fill ins(%cst : f32) outs(%8 : tensor<64x64xf32>) -> tensor<64x64xf32>
              %c64 = arith.constant 64 : index
              %10 = arith.muli %7, %c64 : index
              %11 = affine.apply affine_map<()[s0] -> ((s0 + 31) floordiv 32)>()[%arg5]
              %c0 = arith.constant 0 : index
              %c1 = arith.constant 1 : index
              %12 = scf.for %arg17 = %c0 to %11 step %c1 iter_args(%arg18 = %9) -> (tensor<64x64xf32>) {
                %c32 = arith.constant 32 : index
                %19 = arith.muli %arg17, %c32 : index
                %20 = arith.muli %10, %arg6 : index
                %21 = affine.apply affine_map<(d0)[s0] -> (d0 * 32 + s0)>(%arg17)[%20]
                %reinterpret_cast_0 = memref.reinterpret_cast %arg0 to offset: [%21], sizes: [64, 32], strides: [%arg6, 1] {tmd.reuse = {sequential = [{depth = 5 : i64, iterator = "%arg17", reuse_type = "no_reuse", volume = 0 : i64}], spatial = [{depth = 3 : i64, iterator = "%arg15", mapped_to = "x", reuse_type = "no_reuse", volume = 0 : i64}, {depth = 4 : i64, iterator = "%arg16", mapped_to = "y", reuse_type = "no_reuse", volume = 0 : i64}], temporal = [{depth = 0 : i64, iterator = "%arg12", reuse_type = "no_reuse", volume = 0 : i64}, {depth = 1 : i64, iterator = "%arg13", reuse_type = "total_reuse", volume = 8192 : i64}, {depth = 2 : i64, iterator = "%arg14", reuse_type = "total_reuse", volume = 8192 : i64}]}} : memref<*xf32> to memref<64x32xf32, strided<[?, 1], offset: ?>>
                %alloc = memref.alloc() : memref<64x32xf32>
                memref.copy %reinterpret_cast_0, %alloc : memref<64x32xf32, strided<[?, 1], offset: ?>> to memref<64x32xf32>
                %22 = bufferization.to_tensor %alloc restrict writable : memref<64x32xf32> to tensor<64x32xf32>
                %23 = arith.muli %19, %arg7 : index
                %24 = affine.apply affine_map<(d0)[s0] -> (d0 * 64 + s0)>(%arg14)[%23]
                %reinterpret_cast_1 = memref.reinterpret_cast %arg1 to offset: [%24], sizes: [32, 64], strides: [%arg7, 1] {tmd.reuse = {sequential = [{depth = 5 : i64, iterator = "%arg17", reuse_type = "no_reuse", volume = 0 : i64}], spatial = [{depth = 3 : i64, iterator = "%arg15", mapped_to = "x", reuse_type = "total_reuse", volume = 8192 : i64}, {depth = 4 : i64, iterator = "%arg16", mapped_to = "y", reuse_type = "total_reuse", volume = 8192 : i64}], temporal = [{depth = 0 : i64, iterator = "%arg12", reuse_type = "total_reuse", volume = 8192 : i64}, {depth = 1 : i64, iterator = "%arg13", reuse_type = "total_reuse", volume = 8192 : i64}, {depth = 2 : i64, iterator = "%arg14", reuse_type = "no_reuse", volume = 0 : i64}]}} : memref<*xf32> to memref<32x64xf32, strided<[?, 1], offset: ?>>
                %alloc_2 = memref.alloc() : memref<32x64xf32>
                memref.copy %reinterpret_cast_1, %alloc_2 : memref<32x64xf32, strided<[?, 1], offset: ?>> to memref<32x64xf32>
                %25 = bufferization.to_tensor %alloc_2 restrict writable : memref<32x64xf32> to tensor<32x64xf32>
                %26 = linalg.matmul ins(%22, %25 : tensor<64x32xf32>, tensor<32x64xf32>) outs(%9 : tensor<64x64xf32>) -> tensor<64x64xf32>
                %27 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%arg18, %26 : tensor<64x64xf32>, tensor<64x64xf32>) outs(%arg18 : tensor<64x64xf32>) {
                ^bb0(%in: f32, %in_3: f32, %out: f32):
                  %28 = arith.addf %in, %in_3 : f32
                  linalg.yield %28 : f32
                } -> tensor<64x64xf32>
                scf.yield %27 : tensor<64x64xf32>
              }
              %13 = arith.muli %10, %arg8 : index
              %14 = affine.apply affine_map<(d0)[s0] -> (d0 * 64 + s0)>(%arg14)[%13]
              %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%14], sizes: [64, 64], strides: [%arg8, 1] {tmd.reuse = {spatial = [{depth = 3 : i64, iterator = "%arg15", mapped_to = "x", reuse_type = "no_reuse", volume = 0 : i64}, {depth = 4 : i64, iterator = "%arg16", mapped_to = "y", reuse_type = "no_reuse", volume = 0 : i64}], temporal = [{depth = 0 : i64, iterator = "%arg12", reuse_type = "no_reuse", volume = 0 : i64}, {depth = 1 : i64, iterator = "%arg13", reuse_type = "total_reuse", volume = 16384 : i64}, {depth = 2 : i64, iterator = "%arg14", reuse_type = "no_reuse", volume = 0 : i64}]}} : memref<*xf32> to memref<64x64xf32, strided<[?, 1], offset: ?>>
              %15 = affine.min affine_map<(d0, d1, d2)[s0] -> (d0 * -64 + s0, 64)>(%7, %arg14, %arg13)[%arg3]
              %16 = affine.min affine_map<(d0, d1, d2)[s0] -> (d1 * -64 + s0, 64)>(%7, %arg14, %arg13)[%arg4]
              %extracted_slice = tensor.extract_slice %12[0, 0] [%15, %16] [1, 1] : tensor<64x64xf32> to tensor<?x?xf32>
              %17 = affine.min affine_map<(d0, d1, d2)[s0] -> (d0 * -64 + s0, 64)>(%7, %arg14, %arg13)[%arg3]
              %18 = affine.min affine_map<(d0, d1, d2)[s0] -> (d1 * -64 + s0, 64)>(%7, %arg14, %arg13)[%arg4]
              %subview = memref.subview %reinterpret_cast[0, 0] [%17, %18] [1, 1] : memref<64x64xf32, strided<[?, 1], offset: ?>> to memref<?x?xf32, strided<[?, 1], offset: ?>>
              bufferization.materialize_in_destination %extracted_slice in writable %subview : (tensor<?x?xf32>, memref<?x?xf32, strided<[?, 1], offset: ?>>) -> ()
            } {tmd.mapped_to = "y"}
          } {tmd.mapped_to = "x"}
        }
      }
    }
    return
  }
  func.func @matmul_kernel__d1i0_d0i0_f1_f0_f2(%arg0: memref<*xf32> {tt.divisibility = 16 : i32}, %arg1: memref<*xf32> {tt.divisibility = 16 : i32}, %arg2: memref<*xf32> {tt.divisibility = 16 : i32}, %arg3: index {tt.divisibility = 16 : i32}, %arg4: index {tt.divisibility = 16 : i32}, %arg5: index {tt.divisibility = 16 : i32}, %arg6: index {tt.divisibility = 16 : i32}, %arg7: index {tt.divisibility = 16 : i32}, %arg8: index {tt.divisibility = 16 : i32}, %arg9: index, %arg10: index, %arg11: index) {
    affine.for %arg12 = 0 to affine_map<(d0, d1, d2) -> (d1)>(%arg9, %arg10, %arg11) {
      affine.for %arg13 = 0 to affine_map<(d0, d1, d2) -> ((d0 ceildiv 8) ceildiv 8)>(%arg9, %arg10, %arg11) {
        affine.for %arg14 = 0 to affine_map<(d0, d1, d2) -> (d2)>(%arg9, %arg10, %arg11) {
          affine.parallel (%arg15) = (0) to (8) {
            affine.parallel (%arg16) = (0) to (8) {
              %7 = affine.apply affine_map<(d0, d1, d2) -> (d0 + d1 * 64 + d2 * 8)>(%arg16, %arg13, %arg15)
              %cst = arith.constant 0.000000e+00 : f32
              %8 = tensor.empty() : tensor<64x64xf32>
              %9 = linalg.fill ins(%cst : f32) outs(%8 : tensor<64x64xf32>) -> tensor<64x64xf32>
              %c64 = arith.constant 64 : index
              %10 = arith.muli %7, %c64 : index
              %11 = affine.apply affine_map<()[s0] -> ((s0 + 31) floordiv 32)>()[%arg5]
              %c0 = arith.constant 0 : index
              %c1 = arith.constant 1 : index
              %12 = scf.for %arg17 = %c0 to %11 step %c1 iter_args(%arg18 = %9) -> (tensor<64x64xf32>) {
                %c32 = arith.constant 32 : index
                %19 = arith.muli %arg17, %c32 : index
                %20 = arith.muli %10, %arg6 : index
                %21 = affine.apply affine_map<(d0)[s0] -> (d0 * 32 + s0)>(%arg17)[%20]
                %reinterpret_cast_0 = memref.reinterpret_cast %arg0 to offset: [%21], sizes: [64, 32], strides: [%arg6, 1] {tmd.reuse = {sequential = [{depth = 5 : i64, iterator = "%arg17", reuse_type = "no_reuse", volume = 0 : i64}], spatial = [{depth = 3 : i64, iterator = "%arg15", mapped_to = "x", reuse_type = "no_reuse", volume = 0 : i64}, {depth = 4 : i64, iterator = "%arg16", mapped_to = "y", reuse_type = "no_reuse", volume = 0 : i64}], temporal = [{depth = 0 : i64, iterator = "%arg12", reuse_type = "total_reuse", volume = 8192 : i64}, {depth = 1 : i64, iterator = "%arg13", reuse_type = "no_reuse", volume = 0 : i64}, {depth = 2 : i64, iterator = "%arg14", reuse_type = "total_reuse", volume = 8192 : i64}]}} : memref<*xf32> to memref<64x32xf32, strided<[?, 1], offset: ?>>
                %alloc = memref.alloc() : memref<64x32xf32>
                memref.copy %reinterpret_cast_0, %alloc : memref<64x32xf32, strided<[?, 1], offset: ?>> to memref<64x32xf32>
                %22 = bufferization.to_tensor %alloc restrict writable : memref<64x32xf32> to tensor<64x32xf32>
                %23 = arith.muli %19, %arg7 : index
                %24 = affine.apply affine_map<(d0)[s0] -> (d0 * 64 + s0)>(%arg12)[%23]
                %reinterpret_cast_1 = memref.reinterpret_cast %arg1 to offset: [%24], sizes: [32, 64], strides: [%arg7, 1] {tmd.reuse = {sequential = [{depth = 5 : i64, iterator = "%arg17", reuse_type = "no_reuse", volume = 0 : i64}], spatial = [{depth = 3 : i64, iterator = "%arg15", mapped_to = "x", reuse_type = "total_reuse", volume = 8192 : i64}, {depth = 4 : i64, iterator = "%arg16", mapped_to = "y", reuse_type = "total_reuse", volume = 8192 : i64}], temporal = [{depth = 0 : i64, iterator = "%arg12", reuse_type = "no_reuse", volume = 0 : i64}, {depth = 1 : i64, iterator = "%arg13", reuse_type = "total_reuse", volume = 8192 : i64}, {depth = 2 : i64, iterator = "%arg14", reuse_type = "total_reuse", volume = 8192 : i64}]}} : memref<*xf32> to memref<32x64xf32, strided<[?, 1], offset: ?>>
                %alloc_2 = memref.alloc() : memref<32x64xf32>
                memref.copy %reinterpret_cast_1, %alloc_2 : memref<32x64xf32, strided<[?, 1], offset: ?>> to memref<32x64xf32>
                %25 = bufferization.to_tensor %alloc_2 restrict writable : memref<32x64xf32> to tensor<32x64xf32>
                %26 = linalg.matmul ins(%22, %25 : tensor<64x32xf32>, tensor<32x64xf32>) outs(%9 : tensor<64x64xf32>) -> tensor<64x64xf32>
                %27 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%arg18, %26 : tensor<64x64xf32>, tensor<64x64xf32>) outs(%arg18 : tensor<64x64xf32>) {
                ^bb0(%in: f32, %in_3: f32, %out: f32):
                  %28 = arith.addf %in, %in_3 : f32
                  linalg.yield %28 : f32
                } -> tensor<64x64xf32>
                scf.yield %27 : tensor<64x64xf32>
              }
              %13 = arith.muli %10, %arg8 : index
              %14 = affine.apply affine_map<(d0)[s0] -> (d0 * 64 + s0)>(%arg12)[%13]
              %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%14], sizes: [64, 64], strides: [%arg8, 1] {tmd.reuse = {spatial = [{depth = 3 : i64, iterator = "%arg15", mapped_to = "x", reuse_type = "no_reuse", volume = 0 : i64}, {depth = 4 : i64, iterator = "%arg16", mapped_to = "y", reuse_type = "no_reuse", volume = 0 : i64}], temporal = [{depth = 0 : i64, iterator = "%arg12", reuse_type = "no_reuse", volume = 0 : i64}, {depth = 1 : i64, iterator = "%arg13", reuse_type = "no_reuse", volume = 0 : i64}, {depth = 2 : i64, iterator = "%arg14", reuse_type = "total_reuse", volume = 16384 : i64}]}} : memref<*xf32> to memref<64x64xf32, strided<[?, 1], offset: ?>>
              %15 = affine.min affine_map<(d0, d1, d2)[s0] -> (d0 * -64 + s0, 64)>(%7, %arg12, %arg14)[%arg3]
              %16 = affine.min affine_map<(d0, d1, d2)[s0] -> (d1 * -64 + s0, 64)>(%7, %arg12, %arg14)[%arg4]
              %extracted_slice = tensor.extract_slice %12[0, 0] [%15, %16] [1, 1] : tensor<64x64xf32> to tensor<?x?xf32>
              %17 = affine.min affine_map<(d0, d1, d2)[s0] -> (d0 * -64 + s0, 64)>(%7, %arg12, %arg14)[%arg3]
              %18 = affine.min affine_map<(d0, d1, d2)[s0] -> (d1 * -64 + s0, 64)>(%7, %arg12, %arg14)[%arg4]
              %subview = memref.subview %reinterpret_cast[0, 0] [%17, %18] [1, 1] : memref<64x64xf32, strided<[?, 1], offset: ?>> to memref<?x?xf32, strided<[?, 1], offset: ?>>
              bufferization.materialize_in_destination %extracted_slice in writable %subview : (tensor<?x?xf32>, memref<?x?xf32, strided<[?, 1], offset: ?>>) -> ()
            } {tmd.mapped_to = "y"}
          } {tmd.mapped_to = "x"}
        }
      }
    }
    return
  }
  func.func @matmul_kernel__d1i0_d0i0_f1_f2_f0(%arg0: memref<*xf32> {tt.divisibility = 16 : i32}, %arg1: memref<*xf32> {tt.divisibility = 16 : i32}, %arg2: memref<*xf32> {tt.divisibility = 16 : i32}, %arg3: index {tt.divisibility = 16 : i32}, %arg4: index {tt.divisibility = 16 : i32}, %arg5: index {tt.divisibility = 16 : i32}, %arg6: index {tt.divisibility = 16 : i32}, %arg7: index {tt.divisibility = 16 : i32}, %arg8: index {tt.divisibility = 16 : i32}, %arg9: index, %arg10: index, %arg11: index) {
    affine.for %arg12 = 0 to affine_map<(d0, d1, d2) -> (d1)>(%arg9, %arg10, %arg11) {
      affine.for %arg13 = 0 to affine_map<(d0, d1, d2) -> (d2)>(%arg9, %arg10, %arg11) {
        affine.for %arg14 = 0 to affine_map<(d0, d1, d2) -> ((d0 ceildiv 8) ceildiv 8)>(%arg9, %arg10, %arg11) {
          affine.parallel (%arg15) = (0) to (8) {
            affine.parallel (%arg16) = (0) to (8) {
              %7 = affine.apply affine_map<(d0, d1, d2) -> (d0 + d1 * 64 + d2 * 8)>(%arg16, %arg14, %arg15)
              %cst = arith.constant 0.000000e+00 : f32
              %8 = tensor.empty() : tensor<64x64xf32>
              %9 = linalg.fill ins(%cst : f32) outs(%8 : tensor<64x64xf32>) -> tensor<64x64xf32>
              %c64 = arith.constant 64 : index
              %10 = arith.muli %7, %c64 : index
              %11 = affine.apply affine_map<()[s0] -> ((s0 + 31) floordiv 32)>()[%arg5]
              %c0 = arith.constant 0 : index
              %c1 = arith.constant 1 : index
              %12 = scf.for %arg17 = %c0 to %11 step %c1 iter_args(%arg18 = %9) -> (tensor<64x64xf32>) {
                %c32 = arith.constant 32 : index
                %19 = arith.muli %arg17, %c32 : index
                %20 = arith.muli %10, %arg6 : index
                %21 = affine.apply affine_map<(d0)[s0] -> (d0 * 32 + s0)>(%arg17)[%20]
                %reinterpret_cast_0 = memref.reinterpret_cast %arg0 to offset: [%21], sizes: [64, 32], strides: [%arg6, 1] {tmd.reuse = {sequential = [{depth = 5 : i64, iterator = "%arg17", reuse_type = "no_reuse", volume = 0 : i64}], spatial = [{depth = 3 : i64, iterator = "%arg15", mapped_to = "x", reuse_type = "no_reuse", volume = 0 : i64}, {depth = 4 : i64, iterator = "%arg16", mapped_to = "y", reuse_type = "no_reuse", volume = 0 : i64}], temporal = [{depth = 0 : i64, iterator = "%arg12", reuse_type = "total_reuse", volume = 8192 : i64}, {depth = 1 : i64, iterator = "%arg13", reuse_type = "total_reuse", volume = 8192 : i64}, {depth = 2 : i64, iterator = "%arg14", reuse_type = "no_reuse", volume = 0 : i64}]}} : memref<*xf32> to memref<64x32xf32, strided<[?, 1], offset: ?>>
                %alloc = memref.alloc() : memref<64x32xf32>
                memref.copy %reinterpret_cast_0, %alloc : memref<64x32xf32, strided<[?, 1], offset: ?>> to memref<64x32xf32>
                %22 = bufferization.to_tensor %alloc restrict writable : memref<64x32xf32> to tensor<64x32xf32>
                %23 = arith.muli %19, %arg7 : index
                %24 = affine.apply affine_map<(d0)[s0] -> (d0 * 64 + s0)>(%arg12)[%23]
                %reinterpret_cast_1 = memref.reinterpret_cast %arg1 to offset: [%24], sizes: [32, 64], strides: [%arg7, 1] {tmd.reuse = {sequential = [{depth = 5 : i64, iterator = "%arg17", reuse_type = "no_reuse", volume = 0 : i64}], spatial = [{depth = 3 : i64, iterator = "%arg15", mapped_to = "x", reuse_type = "total_reuse", volume = 8192 : i64}, {depth = 4 : i64, iterator = "%arg16", mapped_to = "y", reuse_type = "total_reuse", volume = 8192 : i64}], temporal = [{depth = 0 : i64, iterator = "%arg12", reuse_type = "no_reuse", volume = 0 : i64}, {depth = 1 : i64, iterator = "%arg13", reuse_type = "total_reuse", volume = 8192 : i64}, {depth = 2 : i64, iterator = "%arg14", reuse_type = "total_reuse", volume = 8192 : i64}]}} : memref<*xf32> to memref<32x64xf32, strided<[?, 1], offset: ?>>
                %alloc_2 = memref.alloc() : memref<32x64xf32>
                memref.copy %reinterpret_cast_1, %alloc_2 : memref<32x64xf32, strided<[?, 1], offset: ?>> to memref<32x64xf32>
                %25 = bufferization.to_tensor %alloc_2 restrict writable : memref<32x64xf32> to tensor<32x64xf32>
                %26 = linalg.matmul ins(%22, %25 : tensor<64x32xf32>, tensor<32x64xf32>) outs(%9 : tensor<64x64xf32>) -> tensor<64x64xf32>
                %27 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%arg18, %26 : tensor<64x64xf32>, tensor<64x64xf32>) outs(%arg18 : tensor<64x64xf32>) {
                ^bb0(%in: f32, %in_3: f32, %out: f32):
                  %28 = arith.addf %in, %in_3 : f32
                  linalg.yield %28 : f32
                } -> tensor<64x64xf32>
                scf.yield %27 : tensor<64x64xf32>
              }
              %13 = arith.muli %10, %arg8 : index
              %14 = affine.apply affine_map<(d0)[s0] -> (d0 * 64 + s0)>(%arg12)[%13]
              %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%14], sizes: [64, 64], strides: [%arg8, 1] {tmd.reuse = {spatial = [{depth = 3 : i64, iterator = "%arg15", mapped_to = "x", reuse_type = "no_reuse", volume = 0 : i64}, {depth = 4 : i64, iterator = "%arg16", mapped_to = "y", reuse_type = "no_reuse", volume = 0 : i64}], temporal = [{depth = 0 : i64, iterator = "%arg12", reuse_type = "no_reuse", volume = 0 : i64}, {depth = 1 : i64, iterator = "%arg13", reuse_type = "total_reuse", volume = 16384 : i64}, {depth = 2 : i64, iterator = "%arg14", reuse_type = "no_reuse", volume = 0 : i64}]}} : memref<*xf32> to memref<64x64xf32, strided<[?, 1], offset: ?>>
              %15 = affine.min affine_map<(d0, d1, d2)[s0] -> (d0 * -64 + s0, 64)>(%7, %arg12, %arg13)[%arg3]
              %16 = affine.min affine_map<(d0, d1, d2)[s0] -> (d1 * -64 + s0, 64)>(%7, %arg12, %arg13)[%arg4]
              %extracted_slice = tensor.extract_slice %12[0, 0] [%15, %16] [1, 1] : tensor<64x64xf32> to tensor<?x?xf32>
              %17 = affine.min affine_map<(d0, d1, d2)[s0] -> (d0 * -64 + s0, 64)>(%7, %arg12, %arg13)[%arg3]
              %18 = affine.min affine_map<(d0, d1, d2)[s0] -> (d1 * -64 + s0, 64)>(%7, %arg12, %arg13)[%arg4]
              %subview = memref.subview %reinterpret_cast[0, 0] [%17, %18] [1, 1] : memref<64x64xf32, strided<[?, 1], offset: ?>> to memref<?x?xf32, strided<[?, 1], offset: ?>>
              bufferization.materialize_in_destination %extracted_slice in writable %subview : (tensor<?x?xf32>, memref<?x?xf32, strided<[?, 1], offset: ?>>) -> ()
            } {tmd.mapped_to = "y"}
          } {tmd.mapped_to = "x"}
        }
      }
    }
    return
  }
  func.func @matmul_kernel__d1i0_d0i0_f2_f0_f1(%arg0: memref<*xf32> {tt.divisibility = 16 : i32}, %arg1: memref<*xf32> {tt.divisibility = 16 : i32}, %arg2: memref<*xf32> {tt.divisibility = 16 : i32}, %arg3: index {tt.divisibility = 16 : i32}, %arg4: index {tt.divisibility = 16 : i32}, %arg5: index {tt.divisibility = 16 : i32}, %arg6: index {tt.divisibility = 16 : i32}, %arg7: index {tt.divisibility = 16 : i32}, %arg8: index {tt.divisibility = 16 : i32}, %arg9: index, %arg10: index, %arg11: index) {
    affine.for %arg12 = 0 to affine_map<(d0, d1, d2) -> (d2)>(%arg9, %arg10, %arg11) {
      affine.for %arg13 = 0 to affine_map<(d0, d1, d2) -> ((d0 ceildiv 8) ceildiv 8)>(%arg9, %arg10, %arg11) {
        affine.for %arg14 = 0 to affine_map<(d0, d1, d2) -> (d1)>(%arg9, %arg10, %arg11) {
          affine.parallel (%arg15) = (0) to (8) {
            affine.parallel (%arg16) = (0) to (8) {
              %7 = affine.apply affine_map<(d0, d1, d2) -> (d0 + d1 * 64 + d2 * 8)>(%arg16, %arg13, %arg15)
              %cst = arith.constant 0.000000e+00 : f32
              %8 = tensor.empty() : tensor<64x64xf32>
              %9 = linalg.fill ins(%cst : f32) outs(%8 : tensor<64x64xf32>) -> tensor<64x64xf32>
              %c64 = arith.constant 64 : index
              %10 = arith.muli %7, %c64 : index
              %11 = affine.apply affine_map<()[s0] -> ((s0 + 31) floordiv 32)>()[%arg5]
              %c0 = arith.constant 0 : index
              %c1 = arith.constant 1 : index
              %12 = scf.for %arg17 = %c0 to %11 step %c1 iter_args(%arg18 = %9) -> (tensor<64x64xf32>) {
                %c32 = arith.constant 32 : index
                %19 = arith.muli %arg17, %c32 : index
                %20 = arith.muli %10, %arg6 : index
                %21 = affine.apply affine_map<(d0)[s0] -> (d0 * 32 + s0)>(%arg17)[%20]
                %reinterpret_cast_0 = memref.reinterpret_cast %arg0 to offset: [%21], sizes: [64, 32], strides: [%arg6, 1] {tmd.reuse = {sequential = [{depth = 5 : i64, iterator = "%arg17", reuse_type = "no_reuse", volume = 0 : i64}], spatial = [{depth = 3 : i64, iterator = "%arg15", mapped_to = "x", reuse_type = "no_reuse", volume = 0 : i64}, {depth = 4 : i64, iterator = "%arg16", mapped_to = "y", reuse_type = "no_reuse", volume = 0 : i64}], temporal = [{depth = 0 : i64, iterator = "%arg12", reuse_type = "total_reuse", volume = 8192 : i64}, {depth = 1 : i64, iterator = "%arg13", reuse_type = "no_reuse", volume = 0 : i64}, {depth = 2 : i64, iterator = "%arg14", reuse_type = "total_reuse", volume = 8192 : i64}]}} : memref<*xf32> to memref<64x32xf32, strided<[?, 1], offset: ?>>
                %alloc = memref.alloc() : memref<64x32xf32>
                memref.copy %reinterpret_cast_0, %alloc : memref<64x32xf32, strided<[?, 1], offset: ?>> to memref<64x32xf32>
                %22 = bufferization.to_tensor %alloc restrict writable : memref<64x32xf32> to tensor<64x32xf32>
                %23 = arith.muli %19, %arg7 : index
                %24 = affine.apply affine_map<(d0)[s0] -> (d0 * 64 + s0)>(%arg14)[%23]
                %reinterpret_cast_1 = memref.reinterpret_cast %arg1 to offset: [%24], sizes: [32, 64], strides: [%arg7, 1] {tmd.reuse = {sequential = [{depth = 5 : i64, iterator = "%arg17", reuse_type = "no_reuse", volume = 0 : i64}], spatial = [{depth = 3 : i64, iterator = "%arg15", mapped_to = "x", reuse_type = "total_reuse", volume = 8192 : i64}, {depth = 4 : i64, iterator = "%arg16", mapped_to = "y", reuse_type = "total_reuse", volume = 8192 : i64}], temporal = [{depth = 0 : i64, iterator = "%arg12", reuse_type = "total_reuse", volume = 8192 : i64}, {depth = 1 : i64, iterator = "%arg13", reuse_type = "total_reuse", volume = 8192 : i64}, {depth = 2 : i64, iterator = "%arg14", reuse_type = "no_reuse", volume = 0 : i64}]}} : memref<*xf32> to memref<32x64xf32, strided<[?, 1], offset: ?>>
                %alloc_2 = memref.alloc() : memref<32x64xf32>
                memref.copy %reinterpret_cast_1, %alloc_2 : memref<32x64xf32, strided<[?, 1], offset: ?>> to memref<32x64xf32>
                %25 = bufferization.to_tensor %alloc_2 restrict writable : memref<32x64xf32> to tensor<32x64xf32>
                %26 = linalg.matmul ins(%22, %25 : tensor<64x32xf32>, tensor<32x64xf32>) outs(%9 : tensor<64x64xf32>) -> tensor<64x64xf32>
                %27 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%arg18, %26 : tensor<64x64xf32>, tensor<64x64xf32>) outs(%arg18 : tensor<64x64xf32>) {
                ^bb0(%in: f32, %in_3: f32, %out: f32):
                  %28 = arith.addf %in, %in_3 : f32
                  linalg.yield %28 : f32
                } -> tensor<64x64xf32>
                scf.yield %27 : tensor<64x64xf32>
              }
              %13 = arith.muli %10, %arg8 : index
              %14 = affine.apply affine_map<(d0)[s0] -> (d0 * 64 + s0)>(%arg14)[%13]
              %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%14], sizes: [64, 64], strides: [%arg8, 1] {tmd.reuse = {spatial = [{depth = 3 : i64, iterator = "%arg15", mapped_to = "x", reuse_type = "no_reuse", volume = 0 : i64}, {depth = 4 : i64, iterator = "%arg16", mapped_to = "y", reuse_type = "no_reuse", volume = 0 : i64}], temporal = [{depth = 0 : i64, iterator = "%arg12", reuse_type = "total_reuse", volume = 16384 : i64}, {depth = 1 : i64, iterator = "%arg13", reuse_type = "no_reuse", volume = 0 : i64}, {depth = 2 : i64, iterator = "%arg14", reuse_type = "no_reuse", volume = 0 : i64}]}} : memref<*xf32> to memref<64x64xf32, strided<[?, 1], offset: ?>>
              %15 = affine.min affine_map<(d0, d1, d2)[s0] -> (d0 * -64 + s0, 64)>(%7, %arg14, %arg12)[%arg3]
              %16 = affine.min affine_map<(d0, d1, d2)[s0] -> (d1 * -64 + s0, 64)>(%7, %arg14, %arg12)[%arg4]
              %extracted_slice = tensor.extract_slice %12[0, 0] [%15, %16] [1, 1] : tensor<64x64xf32> to tensor<?x?xf32>
              %17 = affine.min affine_map<(d0, d1, d2)[s0] -> (d0 * -64 + s0, 64)>(%7, %arg14, %arg12)[%arg3]
              %18 = affine.min affine_map<(d0, d1, d2)[s0] -> (d1 * -64 + s0, 64)>(%7, %arg14, %arg12)[%arg4]
              %subview = memref.subview %reinterpret_cast[0, 0] [%17, %18] [1, 1] : memref<64x64xf32, strided<[?, 1], offset: ?>> to memref<?x?xf32, strided<[?, 1], offset: ?>>
              bufferization.materialize_in_destination %extracted_slice in writable %subview : (tensor<?x?xf32>, memref<?x?xf32, strided<[?, 1], offset: ?>>) -> ()
            } {tmd.mapped_to = "y"}
          } {tmd.mapped_to = "x"}
        }
      }
    }
    return
  }
  func.func @matmul_kernel__d1i0_d0i0_f2_f1_f0(%arg0: memref<*xf32> {tt.divisibility = 16 : i32}, %arg1: memref<*xf32> {tt.divisibility = 16 : i32}, %arg2: memref<*xf32> {tt.divisibility = 16 : i32}, %arg3: index {tt.divisibility = 16 : i32}, %arg4: index {tt.divisibility = 16 : i32}, %arg5: index {tt.divisibility = 16 : i32}, %arg6: index {tt.divisibility = 16 : i32}, %arg7: index {tt.divisibility = 16 : i32}, %arg8: index {tt.divisibility = 16 : i32}, %arg9: index, %arg10: index, %arg11: index) {
    affine.for %arg12 = 0 to affine_map<(d0, d1, d2) -> (d2)>(%arg9, %arg10, %arg11) {
      affine.for %arg13 = 0 to affine_map<(d0, d1, d2) -> (d1)>(%arg9, %arg10, %arg11) {
        affine.for %arg14 = 0 to affine_map<(d0, d1, d2) -> ((d0 ceildiv 8) ceildiv 8)>(%arg9, %arg10, %arg11) {
          affine.parallel (%arg15) = (0) to (8) {
            affine.parallel (%arg16) = (0) to (8) {
              %7 = affine.apply affine_map<(d0, d1, d2) -> (d0 + d1 * 64 + d2 * 8)>(%arg16, %arg14, %arg15)
              %cst = arith.constant 0.000000e+00 : f32
              %8 = tensor.empty() : tensor<64x64xf32>
              %9 = linalg.fill ins(%cst : f32) outs(%8 : tensor<64x64xf32>) -> tensor<64x64xf32>
              %c64 = arith.constant 64 : index
              %10 = arith.muli %7, %c64 : index
              %11 = affine.apply affine_map<()[s0] -> ((s0 + 31) floordiv 32)>()[%arg5]
              %c0 = arith.constant 0 : index
              %c1 = arith.constant 1 : index
              %12 = scf.for %arg17 = %c0 to %11 step %c1 iter_args(%arg18 = %9) -> (tensor<64x64xf32>) {
                %c32 = arith.constant 32 : index
                %19 = arith.muli %arg17, %c32 : index
                %20 = arith.muli %10, %arg6 : index
                %21 = affine.apply affine_map<(d0)[s0] -> (d0 * 32 + s0)>(%arg17)[%20]
                %reinterpret_cast_0 = memref.reinterpret_cast %arg0 to offset: [%21], sizes: [64, 32], strides: [%arg6, 1] {tmd.reuse = {sequential = [{depth = 5 : i64, iterator = "%arg17", reuse_type = "no_reuse", volume = 0 : i64}], spatial = [{depth = 3 : i64, iterator = "%arg15", mapped_to = "x", reuse_type = "no_reuse", volume = 0 : i64}, {depth = 4 : i64, iterator = "%arg16", mapped_to = "y", reuse_type = "no_reuse", volume = 0 : i64}], temporal = [{depth = 0 : i64, iterator = "%arg12", reuse_type = "total_reuse", volume = 8192 : i64}, {depth = 1 : i64, iterator = "%arg13", reuse_type = "total_reuse", volume = 8192 : i64}, {depth = 2 : i64, iterator = "%arg14", reuse_type = "no_reuse", volume = 0 : i64}]}} : memref<*xf32> to memref<64x32xf32, strided<[?, 1], offset: ?>>
                %alloc = memref.alloc() : memref<64x32xf32>
                memref.copy %reinterpret_cast_0, %alloc : memref<64x32xf32, strided<[?, 1], offset: ?>> to memref<64x32xf32>
                %22 = bufferization.to_tensor %alloc restrict writable : memref<64x32xf32> to tensor<64x32xf32>
                %23 = arith.muli %19, %arg7 : index
                %24 = affine.apply affine_map<(d0)[s0] -> (d0 * 64 + s0)>(%arg13)[%23]
                %reinterpret_cast_1 = memref.reinterpret_cast %arg1 to offset: [%24], sizes: [32, 64], strides: [%arg7, 1] {tmd.reuse = {sequential = [{depth = 5 : i64, iterator = "%arg17", reuse_type = "no_reuse", volume = 0 : i64}], spatial = [{depth = 3 : i64, iterator = "%arg15", mapped_to = "x", reuse_type = "total_reuse", volume = 8192 : i64}, {depth = 4 : i64, iterator = "%arg16", mapped_to = "y", reuse_type = "total_reuse", volume = 8192 : i64}], temporal = [{depth = 0 : i64, iterator = "%arg12", reuse_type = "total_reuse", volume = 8192 : i64}, {depth = 1 : i64, iterator = "%arg13", reuse_type = "no_reuse", volume = 0 : i64}, {depth = 2 : i64, iterator = "%arg14", reuse_type = "total_reuse", volume = 8192 : i64}]}} : memref<*xf32> to memref<32x64xf32, strided<[?, 1], offset: ?>>
                %alloc_2 = memref.alloc() : memref<32x64xf32>
                memref.copy %reinterpret_cast_1, %alloc_2 : memref<32x64xf32, strided<[?, 1], offset: ?>> to memref<32x64xf32>
                %25 = bufferization.to_tensor %alloc_2 restrict writable : memref<32x64xf32> to tensor<32x64xf32>
                %26 = linalg.matmul ins(%22, %25 : tensor<64x32xf32>, tensor<32x64xf32>) outs(%9 : tensor<64x64xf32>) -> tensor<64x64xf32>
                %27 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%arg18, %26 : tensor<64x64xf32>, tensor<64x64xf32>) outs(%arg18 : tensor<64x64xf32>) {
                ^bb0(%in: f32, %in_3: f32, %out: f32):
                  %28 = arith.addf %in, %in_3 : f32
                  linalg.yield %28 : f32
                } -> tensor<64x64xf32>
                scf.yield %27 : tensor<64x64xf32>
              }
              %13 = arith.muli %10, %arg8 : index
              %14 = affine.apply affine_map<(d0)[s0] -> (d0 * 64 + s0)>(%arg13)[%13]
              %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%14], sizes: [64, 64], strides: [%arg8, 1] {tmd.reuse = {spatial = [{depth = 3 : i64, iterator = "%arg15", mapped_to = "x", reuse_type = "no_reuse", volume = 0 : i64}, {depth = 4 : i64, iterator = "%arg16", mapped_to = "y", reuse_type = "no_reuse", volume = 0 : i64}], temporal = [{depth = 0 : i64, iterator = "%arg12", reuse_type = "total_reuse", volume = 16384 : i64}, {depth = 1 : i64, iterator = "%arg13", reuse_type = "no_reuse", volume = 0 : i64}, {depth = 2 : i64, iterator = "%arg14", reuse_type = "no_reuse", volume = 0 : i64}]}} : memref<*xf32> to memref<64x64xf32, strided<[?, 1], offset: ?>>
              %15 = affine.min affine_map<(d0, d1, d2)[s0] -> (d0 * -64 + s0, 64)>(%7, %arg13, %arg12)[%arg3]
              %16 = affine.min affine_map<(d0, d1, d2)[s0] -> (d1 * -64 + s0, 64)>(%7, %arg13, %arg12)[%arg4]
              %extracted_slice = tensor.extract_slice %12[0, 0] [%15, %16] [1, 1] : tensor<64x64xf32> to tensor<?x?xf32>
              %17 = affine.min affine_map<(d0, d1, d2)[s0] -> (d0 * -64 + s0, 64)>(%7, %arg13, %arg12)[%arg3]
              %18 = affine.min affine_map<(d0, d1, d2)[s0] -> (d1 * -64 + s0, 64)>(%7, %arg13, %arg12)[%arg4]
              %subview = memref.subview %reinterpret_cast[0, 0] [%17, %18] [1, 1] : memref<64x64xf32, strided<[?, 1], offset: ?>> to memref<?x?xf32, strided<[?, 1], offset: ?>>
              bufferization.materialize_in_destination %extracted_slice in writable %subview : (tensor<?x?xf32>, memref<?x?xf32, strided<[?, 1], offset: ?>>) -> ()
            } {tmd.mapped_to = "y"}
          } {tmd.mapped_to = "x"}
        }
      }
    }
    return
  }
  func.func @matmul_kernel__d0i0_d1i1_f0_f1_f2(%arg0: memref<*xf32> {tt.divisibility = 16 : i32}, %arg1: memref<*xf32> {tt.divisibility = 16 : i32}, %arg2: memref<*xf32> {tt.divisibility = 16 : i32}, %arg3: index {tt.divisibility = 16 : i32}, %arg4: index {tt.divisibility = 16 : i32}, %arg5: index {tt.divisibility = 16 : i32}, %arg6: index {tt.divisibility = 16 : i32}, %arg7: index {tt.divisibility = 16 : i32}, %arg8: index {tt.divisibility = 16 : i32}, %arg9: index, %arg10: index, %arg11: index) {
    affine.for %arg12 = 0 to affine_map<(d0, d1, d2) -> (d0 ceildiv 8)>(%arg9, %arg10, %arg11) {
      affine.for %arg13 = 0 to affine_map<(d0, d1, d2) -> (d1 ceildiv 8)>(%arg9, %arg10, %arg11) {
        affine.for %arg14 = 0 to affine_map<(d0, d1, d2) -> (d2)>(%arg9, %arg10, %arg11) {
          affine.parallel (%arg15) = (0) to (8) {
            %7 = affine.apply affine_map<(d0, d1) -> (d0 * 8 + d1)>(%arg13, %arg15)
            affine.parallel (%arg16) = (0) to (8) {
              %8 = affine.apply affine_map<(d0, d1) -> (d0 * 8 + d1)>(%arg12, %arg16)
              %cst = arith.constant 0.000000e+00 : f32
              %9 = tensor.empty() : tensor<64x64xf32>
              %10 = linalg.fill ins(%cst : f32) outs(%9 : tensor<64x64xf32>) -> tensor<64x64xf32>
              %c64 = arith.constant 64 : index
              %11 = arith.muli %8, %c64 : index
              %12 = affine.apply affine_map<()[s0] -> ((s0 + 31) floordiv 32)>()[%arg5]
              %c0 = arith.constant 0 : index
              %c1 = arith.constant 1 : index
              %13 = scf.for %arg17 = %c0 to %12 step %c1 iter_args(%arg18 = %10) -> (tensor<64x64xf32>) {
                %c32 = arith.constant 32 : index
                %20 = arith.muli %arg17, %c32 : index
                %21 = arith.muli %11, %arg6 : index
                %22 = affine.apply affine_map<(d0)[s0] -> (d0 * 32 + s0)>(%arg17)[%21]
                %reinterpret_cast_0 = memref.reinterpret_cast %arg0 to offset: [%22], sizes: [64, 32], strides: [%arg6, 1] {tmd.reuse = {sequential = [{depth = 5 : i64, iterator = "%arg17", reuse_type = "no_reuse", volume = 0 : i64}], spatial = [{depth = 3 : i64, iterator = "%arg15", mapped_to = "y", reuse_type = "total_reuse", volume = 8192 : i64}, {depth = 4 : i64, iterator = "%arg16", mapped_to = "x", reuse_type = "no_reuse", volume = 0 : i64}], temporal = [{depth = 0 : i64, iterator = "%arg12", reuse_type = "no_reuse", volume = 0 : i64}, {depth = 1 : i64, iterator = "%arg13", reuse_type = "total_reuse", volume = 8192 : i64}, {depth = 2 : i64, iterator = "%arg14", reuse_type = "total_reuse", volume = 8192 : i64}]}} : memref<*xf32> to memref<64x32xf32, strided<[?, 1], offset: ?>>
                %alloc = memref.alloc() : memref<64x32xf32>
                memref.copy %reinterpret_cast_0, %alloc : memref<64x32xf32, strided<[?, 1], offset: ?>> to memref<64x32xf32>
                %23 = bufferization.to_tensor %alloc restrict writable : memref<64x32xf32> to tensor<64x32xf32>
                %24 = arith.muli %20, %arg7 : index
                %25 = affine.apply affine_map<(d0, d1)[s0] -> (d0 * 512 + d1 * 64 + s0)>(%arg13, %arg15)[%24]
                %reinterpret_cast_1 = memref.reinterpret_cast %arg1 to offset: [%25], sizes: [32, 64], strides: [%arg7, 1] {tmd.reuse = {sequential = [{depth = 5 : i64, iterator = "%arg17", reuse_type = "no_reuse", volume = 0 : i64}], spatial = [{depth = 3 : i64, iterator = "%arg15", mapped_to = "y", reuse_type = "no_reuse", volume = 0 : i64}, {depth = 4 : i64, iterator = "%arg16", mapped_to = "x", reuse_type = "total_reuse", volume = 8192 : i64}], temporal = [{depth = 0 : i64, iterator = "%arg12", reuse_type = "total_reuse", volume = 8192 : i64}, {depth = 1 : i64, iterator = "%arg13", reuse_type = "no_reuse", volume = 0 : i64}, {depth = 2 : i64, iterator = "%arg14", reuse_type = "total_reuse", volume = 8192 : i64}]}} : memref<*xf32> to memref<32x64xf32, strided<[?, 1], offset: ?>>
                %alloc_2 = memref.alloc() : memref<32x64xf32>
                memref.copy %reinterpret_cast_1, %alloc_2 : memref<32x64xf32, strided<[?, 1], offset: ?>> to memref<32x64xf32>
                %26 = bufferization.to_tensor %alloc_2 restrict writable : memref<32x64xf32> to tensor<32x64xf32>
                %27 = linalg.matmul ins(%23, %26 : tensor<64x32xf32>, tensor<32x64xf32>) outs(%10 : tensor<64x64xf32>) -> tensor<64x64xf32>
                %28 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%arg18, %27 : tensor<64x64xf32>, tensor<64x64xf32>) outs(%arg18 : tensor<64x64xf32>) {
                ^bb0(%in: f32, %in_3: f32, %out: f32):
                  %29 = arith.addf %in, %in_3 : f32
                  linalg.yield %29 : f32
                } -> tensor<64x64xf32>
                scf.yield %28 : tensor<64x64xf32>
              }
              %14 = arith.muli %11, %arg8 : index
              %15 = affine.apply affine_map<(d0, d1)[s0] -> (d0 * 512 + d1 * 64 + s0)>(%arg13, %arg15)[%14]
              %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%15], sizes: [64, 64], strides: [%arg8, 1] {tmd.reuse = {spatial = [{depth = 3 : i64, iterator = "%arg15", mapped_to = "y", reuse_type = "no_reuse", volume = 0 : i64}, {depth = 4 : i64, iterator = "%arg16", mapped_to = "x", reuse_type = "no_reuse", volume = 0 : i64}], temporal = [{depth = 0 : i64, iterator = "%arg12", reuse_type = "no_reuse", volume = 0 : i64}, {depth = 1 : i64, iterator = "%arg13", reuse_type = "no_reuse", volume = 0 : i64}, {depth = 2 : i64, iterator = "%arg14", reuse_type = "total_reuse", volume = 16384 : i64}]}} : memref<*xf32> to memref<64x64xf32, strided<[?, 1], offset: ?>>
              %16 = affine.min affine_map<(d0, d1, d2)[s0] -> (d0 * -64 + s0, 64)>(%8, %7, %arg14)[%arg3]
              %17 = affine.min affine_map<(d0, d1, d2)[s0] -> (d1 * -64 + s0, 64)>(%8, %7, %arg14)[%arg4]
              %extracted_slice = tensor.extract_slice %13[0, 0] [%16, %17] [1, 1] : tensor<64x64xf32> to tensor<?x?xf32>
              %18 = affine.min affine_map<(d0, d1, d2)[s0] -> (d0 * -64 + s0, 64)>(%8, %7, %arg14)[%arg3]
              %19 = affine.min affine_map<(d0, d1, d2)[s0] -> (d1 * -64 + s0, 64)>(%8, %7, %arg14)[%arg4]
              %subview = memref.subview %reinterpret_cast[0, 0] [%18, %19] [1, 1] : memref<64x64xf32, strided<[?, 1], offset: ?>> to memref<?x?xf32, strided<[?, 1], offset: ?>>
              bufferization.materialize_in_destination %extracted_slice in writable %subview : (tensor<?x?xf32>, memref<?x?xf32, strided<[?, 1], offset: ?>>) -> ()
            } {tmd.mapped_to = "x"}
          } {tmd.mapped_to = "y"}
        }
      }
    }
    return
  }
  func.func @matmul_kernel__d0i0_d1i1_f0_f2_f1(%arg0: memref<*xf32> {tt.divisibility = 16 : i32}, %arg1: memref<*xf32> {tt.divisibility = 16 : i32}, %arg2: memref<*xf32> {tt.divisibility = 16 : i32}, %arg3: index {tt.divisibility = 16 : i32}, %arg4: index {tt.divisibility = 16 : i32}, %arg5: index {tt.divisibility = 16 : i32}, %arg6: index {tt.divisibility = 16 : i32}, %arg7: index {tt.divisibility = 16 : i32}, %arg8: index {tt.divisibility = 16 : i32}, %arg9: index, %arg10: index, %arg11: index) {
    affine.for %arg12 = 0 to affine_map<(d0, d1, d2) -> (d0 ceildiv 8)>(%arg9, %arg10, %arg11) {
      affine.for %arg13 = 0 to affine_map<(d0, d1, d2) -> (d2)>(%arg9, %arg10, %arg11) {
        affine.for %arg14 = 0 to affine_map<(d0, d1, d2) -> (d1 ceildiv 8)>(%arg9, %arg10, %arg11) {
          affine.parallel (%arg15) = (0) to (8) {
            %7 = affine.apply affine_map<(d0, d1) -> (d0 * 8 + d1)>(%arg14, %arg15)
            affine.parallel (%arg16) = (0) to (8) {
              %8 = affine.apply affine_map<(d0, d1) -> (d0 * 8 + d1)>(%arg12, %arg16)
              %cst = arith.constant 0.000000e+00 : f32
              %9 = tensor.empty() : tensor<64x64xf32>
              %10 = linalg.fill ins(%cst : f32) outs(%9 : tensor<64x64xf32>) -> tensor<64x64xf32>
              %c64 = arith.constant 64 : index
              %11 = arith.muli %8, %c64 : index
              %12 = affine.apply affine_map<()[s0] -> ((s0 + 31) floordiv 32)>()[%arg5]
              %c0 = arith.constant 0 : index
              %c1 = arith.constant 1 : index
              %13 = scf.for %arg17 = %c0 to %12 step %c1 iter_args(%arg18 = %10) -> (tensor<64x64xf32>) {
                %c32 = arith.constant 32 : index
                %20 = arith.muli %arg17, %c32 : index
                %21 = arith.muli %11, %arg6 : index
                %22 = affine.apply affine_map<(d0)[s0] -> (d0 * 32 + s0)>(%arg17)[%21]
                %reinterpret_cast_0 = memref.reinterpret_cast %arg0 to offset: [%22], sizes: [64, 32], strides: [%arg6, 1] {tmd.reuse = {sequential = [{depth = 5 : i64, iterator = "%arg17", reuse_type = "no_reuse", volume = 0 : i64}], spatial = [{depth = 3 : i64, iterator = "%arg15", mapped_to = "y", reuse_type = "total_reuse", volume = 8192 : i64}, {depth = 4 : i64, iterator = "%arg16", mapped_to = "x", reuse_type = "no_reuse", volume = 0 : i64}], temporal = [{depth = 0 : i64, iterator = "%arg12", reuse_type = "no_reuse", volume = 0 : i64}, {depth = 1 : i64, iterator = "%arg13", reuse_type = "total_reuse", volume = 8192 : i64}, {depth = 2 : i64, iterator = "%arg14", reuse_type = "total_reuse", volume = 8192 : i64}]}} : memref<*xf32> to memref<64x32xf32, strided<[?, 1], offset: ?>>
                %alloc = memref.alloc() : memref<64x32xf32>
                memref.copy %reinterpret_cast_0, %alloc : memref<64x32xf32, strided<[?, 1], offset: ?>> to memref<64x32xf32>
                %23 = bufferization.to_tensor %alloc restrict writable : memref<64x32xf32> to tensor<64x32xf32>
                %24 = arith.muli %20, %arg7 : index
                %25 = affine.apply affine_map<(d0, d1)[s0] -> (d0 * 512 + d1 * 64 + s0)>(%arg14, %arg15)[%24]
                %reinterpret_cast_1 = memref.reinterpret_cast %arg1 to offset: [%25], sizes: [32, 64], strides: [%arg7, 1] {tmd.reuse = {sequential = [{depth = 5 : i64, iterator = "%arg17", reuse_type = "no_reuse", volume = 0 : i64}], spatial = [{depth = 3 : i64, iterator = "%arg15", mapped_to = "y", reuse_type = "no_reuse", volume = 0 : i64}, {depth = 4 : i64, iterator = "%arg16", mapped_to = "x", reuse_type = "total_reuse", volume = 8192 : i64}], temporal = [{depth = 0 : i64, iterator = "%arg12", reuse_type = "total_reuse", volume = 8192 : i64}, {depth = 1 : i64, iterator = "%arg13", reuse_type = "total_reuse", volume = 8192 : i64}, {depth = 2 : i64, iterator = "%arg14", reuse_type = "no_reuse", volume = 0 : i64}]}} : memref<*xf32> to memref<32x64xf32, strided<[?, 1], offset: ?>>
                %alloc_2 = memref.alloc() : memref<32x64xf32>
                memref.copy %reinterpret_cast_1, %alloc_2 : memref<32x64xf32, strided<[?, 1], offset: ?>> to memref<32x64xf32>
                %26 = bufferization.to_tensor %alloc_2 restrict writable : memref<32x64xf32> to tensor<32x64xf32>
                %27 = linalg.matmul ins(%23, %26 : tensor<64x32xf32>, tensor<32x64xf32>) outs(%10 : tensor<64x64xf32>) -> tensor<64x64xf32>
                %28 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%arg18, %27 : tensor<64x64xf32>, tensor<64x64xf32>) outs(%arg18 : tensor<64x64xf32>) {
                ^bb0(%in: f32, %in_3: f32, %out: f32):
                  %29 = arith.addf %in, %in_3 : f32
                  linalg.yield %29 : f32
                } -> tensor<64x64xf32>
                scf.yield %28 : tensor<64x64xf32>
              }
              %14 = arith.muli %11, %arg8 : index
              %15 = affine.apply affine_map<(d0, d1)[s0] -> (d0 * 512 + d1 * 64 + s0)>(%arg14, %arg15)[%14]
              %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%15], sizes: [64, 64], strides: [%arg8, 1] {tmd.reuse = {spatial = [{depth = 3 : i64, iterator = "%arg15", mapped_to = "y", reuse_type = "no_reuse", volume = 0 : i64}, {depth = 4 : i64, iterator = "%arg16", mapped_to = "x", reuse_type = "no_reuse", volume = 0 : i64}], temporal = [{depth = 0 : i64, iterator = "%arg12", reuse_type = "no_reuse", volume = 0 : i64}, {depth = 1 : i64, iterator = "%arg13", reuse_type = "total_reuse", volume = 16384 : i64}, {depth = 2 : i64, iterator = "%arg14", reuse_type = "no_reuse", volume = 0 : i64}]}} : memref<*xf32> to memref<64x64xf32, strided<[?, 1], offset: ?>>
              %16 = affine.min affine_map<(d0, d1, d2)[s0] -> (d0 * -64 + s0, 64)>(%8, %7, %arg13)[%arg3]
              %17 = affine.min affine_map<(d0, d1, d2)[s0] -> (d1 * -64 + s0, 64)>(%8, %7, %arg13)[%arg4]
              %extracted_slice = tensor.extract_slice %13[0, 0] [%16, %17] [1, 1] : tensor<64x64xf32> to tensor<?x?xf32>
              %18 = affine.min affine_map<(d0, d1, d2)[s0] -> (d0 * -64 + s0, 64)>(%8, %7, %arg13)[%arg3]
              %19 = affine.min affine_map<(d0, d1, d2)[s0] -> (d1 * -64 + s0, 64)>(%8, %7, %arg13)[%arg4]
              %subview = memref.subview %reinterpret_cast[0, 0] [%18, %19] [1, 1] : memref<64x64xf32, strided<[?, 1], offset: ?>> to memref<?x?xf32, strided<[?, 1], offset: ?>>
              bufferization.materialize_in_destination %extracted_slice in writable %subview : (tensor<?x?xf32>, memref<?x?xf32, strided<[?, 1], offset: ?>>) -> ()
            } {tmd.mapped_to = "x"}
          } {tmd.mapped_to = "y"}
        }
      }
    }
    return
  }
  func.func @matmul_kernel__d0i0_d1i1_f1_f0_f2(%arg0: memref<*xf32> {tt.divisibility = 16 : i32}, %arg1: memref<*xf32> {tt.divisibility = 16 : i32}, %arg2: memref<*xf32> {tt.divisibility = 16 : i32}, %arg3: index {tt.divisibility = 16 : i32}, %arg4: index {tt.divisibility = 16 : i32}, %arg5: index {tt.divisibility = 16 : i32}, %arg6: index {tt.divisibility = 16 : i32}, %arg7: index {tt.divisibility = 16 : i32}, %arg8: index {tt.divisibility = 16 : i32}, %arg9: index, %arg10: index, %arg11: index) {
    affine.for %arg12 = 0 to affine_map<(d0, d1, d2) -> (d1 ceildiv 8)>(%arg9, %arg10, %arg11) {
      affine.for %arg13 = 0 to affine_map<(d0, d1, d2) -> (d0 ceildiv 8)>(%arg9, %arg10, %arg11) {
        affine.for %arg14 = 0 to affine_map<(d0, d1, d2) -> (d2)>(%arg9, %arg10, %arg11) {
          affine.parallel (%arg15) = (0) to (8) {
            %7 = affine.apply affine_map<(d0, d1) -> (d0 * 8 + d1)>(%arg12, %arg15)
            affine.parallel (%arg16) = (0) to (8) {
              %8 = affine.apply affine_map<(d0, d1) -> (d0 * 8 + d1)>(%arg13, %arg16)
              %cst = arith.constant 0.000000e+00 : f32
              %9 = tensor.empty() : tensor<64x64xf32>
              %10 = linalg.fill ins(%cst : f32) outs(%9 : tensor<64x64xf32>) -> tensor<64x64xf32>
              %c64 = arith.constant 64 : index
              %11 = arith.muli %8, %c64 : index
              %12 = affine.apply affine_map<()[s0] -> ((s0 + 31) floordiv 32)>()[%arg5]
              %c0 = arith.constant 0 : index
              %c1 = arith.constant 1 : index
              %13 = scf.for %arg17 = %c0 to %12 step %c1 iter_args(%arg18 = %10) -> (tensor<64x64xf32>) {
                %c32 = arith.constant 32 : index
                %20 = arith.muli %arg17, %c32 : index
                %21 = arith.muli %11, %arg6 : index
                %22 = affine.apply affine_map<(d0)[s0] -> (d0 * 32 + s0)>(%arg17)[%21]
                %reinterpret_cast_0 = memref.reinterpret_cast %arg0 to offset: [%22], sizes: [64, 32], strides: [%arg6, 1] {tmd.reuse = {sequential = [{depth = 5 : i64, iterator = "%arg17", reuse_type = "no_reuse", volume = 0 : i64}], spatial = [{depth = 3 : i64, iterator = "%arg15", mapped_to = "y", reuse_type = "total_reuse", volume = 8192 : i64}, {depth = 4 : i64, iterator = "%arg16", mapped_to = "x", reuse_type = "no_reuse", volume = 0 : i64}], temporal = [{depth = 0 : i64, iterator = "%arg12", reuse_type = "total_reuse", volume = 8192 : i64}, {depth = 1 : i64, iterator = "%arg13", reuse_type = "no_reuse", volume = 0 : i64}, {depth = 2 : i64, iterator = "%arg14", reuse_type = "total_reuse", volume = 8192 : i64}]}} : memref<*xf32> to memref<64x32xf32, strided<[?, 1], offset: ?>>
                %alloc = memref.alloc() : memref<64x32xf32>
                memref.copy %reinterpret_cast_0, %alloc : memref<64x32xf32, strided<[?, 1], offset: ?>> to memref<64x32xf32>
                %23 = bufferization.to_tensor %alloc restrict writable : memref<64x32xf32> to tensor<64x32xf32>
                %24 = arith.muli %20, %arg7 : index
                %25 = affine.apply affine_map<(d0, d1)[s0] -> (d0 * 512 + d1 * 64 + s0)>(%arg12, %arg15)[%24]
                %reinterpret_cast_1 = memref.reinterpret_cast %arg1 to offset: [%25], sizes: [32, 64], strides: [%arg7, 1] {tmd.reuse = {sequential = [{depth = 5 : i64, iterator = "%arg17", reuse_type = "no_reuse", volume = 0 : i64}], spatial = [{depth = 3 : i64, iterator = "%arg15", mapped_to = "y", reuse_type = "no_reuse", volume = 0 : i64}, {depth = 4 : i64, iterator = "%arg16", mapped_to = "x", reuse_type = "total_reuse", volume = 8192 : i64}], temporal = [{depth = 0 : i64, iterator = "%arg12", reuse_type = "no_reuse", volume = 0 : i64}, {depth = 1 : i64, iterator = "%arg13", reuse_type = "total_reuse", volume = 8192 : i64}, {depth = 2 : i64, iterator = "%arg14", reuse_type = "total_reuse", volume = 8192 : i64}]}} : memref<*xf32> to memref<32x64xf32, strided<[?, 1], offset: ?>>
                %alloc_2 = memref.alloc() : memref<32x64xf32>
                memref.copy %reinterpret_cast_1, %alloc_2 : memref<32x64xf32, strided<[?, 1], offset: ?>> to memref<32x64xf32>
                %26 = bufferization.to_tensor %alloc_2 restrict writable : memref<32x64xf32> to tensor<32x64xf32>
                %27 = linalg.matmul ins(%23, %26 : tensor<64x32xf32>, tensor<32x64xf32>) outs(%10 : tensor<64x64xf32>) -> tensor<64x64xf32>
                %28 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%arg18, %27 : tensor<64x64xf32>, tensor<64x64xf32>) outs(%arg18 : tensor<64x64xf32>) {
                ^bb0(%in: f32, %in_3: f32, %out: f32):
                  %29 = arith.addf %in, %in_3 : f32
                  linalg.yield %29 : f32
                } -> tensor<64x64xf32>
                scf.yield %28 : tensor<64x64xf32>
              }
              %14 = arith.muli %11, %arg8 : index
              %15 = affine.apply affine_map<(d0, d1)[s0] -> (d0 * 512 + d1 * 64 + s0)>(%arg12, %arg15)[%14]
              %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%15], sizes: [64, 64], strides: [%arg8, 1] {tmd.reuse = {spatial = [{depth = 3 : i64, iterator = "%arg15", mapped_to = "y", reuse_type = "no_reuse", volume = 0 : i64}, {depth = 4 : i64, iterator = "%arg16", mapped_to = "x", reuse_type = "no_reuse", volume = 0 : i64}], temporal = [{depth = 0 : i64, iterator = "%arg12", reuse_type = "no_reuse", volume = 0 : i64}, {depth = 1 : i64, iterator = "%arg13", reuse_type = "no_reuse", volume = 0 : i64}, {depth = 2 : i64, iterator = "%arg14", reuse_type = "total_reuse", volume = 16384 : i64}]}} : memref<*xf32> to memref<64x64xf32, strided<[?, 1], offset: ?>>
              %16 = affine.min affine_map<(d0, d1, d2)[s0] -> (d0 * -64 + s0, 64)>(%8, %7, %arg14)[%arg3]
              %17 = affine.min affine_map<(d0, d1, d2)[s0] -> (d1 * -64 + s0, 64)>(%8, %7, %arg14)[%arg4]
              %extracted_slice = tensor.extract_slice %13[0, 0] [%16, %17] [1, 1] : tensor<64x64xf32> to tensor<?x?xf32>
              %18 = affine.min affine_map<(d0, d1, d2)[s0] -> (d0 * -64 + s0, 64)>(%8, %7, %arg14)[%arg3]
              %19 = affine.min affine_map<(d0, d1, d2)[s0] -> (d1 * -64 + s0, 64)>(%8, %7, %arg14)[%arg4]
              %subview = memref.subview %reinterpret_cast[0, 0] [%18, %19] [1, 1] : memref<64x64xf32, strided<[?, 1], offset: ?>> to memref<?x?xf32, strided<[?, 1], offset: ?>>
              bufferization.materialize_in_destination %extracted_slice in writable %subview : (tensor<?x?xf32>, memref<?x?xf32, strided<[?, 1], offset: ?>>) -> ()
            } {tmd.mapped_to = "x"}
          } {tmd.mapped_to = "y"}
        }
      }
    }
    return
  }
  func.func @matmul_kernel__d0i0_d1i1_f1_f2_f0(%arg0: memref<*xf32> {tt.divisibility = 16 : i32}, %arg1: memref<*xf32> {tt.divisibility = 16 : i32}, %arg2: memref<*xf32> {tt.divisibility = 16 : i32}, %arg3: index {tt.divisibility = 16 : i32}, %arg4: index {tt.divisibility = 16 : i32}, %arg5: index {tt.divisibility = 16 : i32}, %arg6: index {tt.divisibility = 16 : i32}, %arg7: index {tt.divisibility = 16 : i32}, %arg8: index {tt.divisibility = 16 : i32}, %arg9: index, %arg10: index, %arg11: index) {
    affine.for %arg12 = 0 to affine_map<(d0, d1, d2) -> (d1 ceildiv 8)>(%arg9, %arg10, %arg11) {
      affine.for %arg13 = 0 to affine_map<(d0, d1, d2) -> (d2)>(%arg9, %arg10, %arg11) {
        affine.for %arg14 = 0 to affine_map<(d0, d1, d2) -> (d0 ceildiv 8)>(%arg9, %arg10, %arg11) {
          affine.parallel (%arg15) = (0) to (8) {
            %7 = affine.apply affine_map<(d0, d1) -> (d0 * 8 + d1)>(%arg12, %arg15)
            affine.parallel (%arg16) = (0) to (8) {
              %8 = affine.apply affine_map<(d0, d1) -> (d0 * 8 + d1)>(%arg14, %arg16)
              %cst = arith.constant 0.000000e+00 : f32
              %9 = tensor.empty() : tensor<64x64xf32>
              %10 = linalg.fill ins(%cst : f32) outs(%9 : tensor<64x64xf32>) -> tensor<64x64xf32>
              %c64 = arith.constant 64 : index
              %11 = arith.muli %8, %c64 : index
              %12 = affine.apply affine_map<()[s0] -> ((s0 + 31) floordiv 32)>()[%arg5]
              %c0 = arith.constant 0 : index
              %c1 = arith.constant 1 : index
              %13 = scf.for %arg17 = %c0 to %12 step %c1 iter_args(%arg18 = %10) -> (tensor<64x64xf32>) {
                %c32 = arith.constant 32 : index
                %20 = arith.muli %arg17, %c32 : index
                %21 = arith.muli %11, %arg6 : index
                %22 = affine.apply affine_map<(d0)[s0] -> (d0 * 32 + s0)>(%arg17)[%21]
                %reinterpret_cast_0 = memref.reinterpret_cast %arg0 to offset: [%22], sizes: [64, 32], strides: [%arg6, 1] {tmd.reuse = {sequential = [{depth = 5 : i64, iterator = "%arg17", reuse_type = "no_reuse", volume = 0 : i64}], spatial = [{depth = 3 : i64, iterator = "%arg15", mapped_to = "y", reuse_type = "total_reuse", volume = 8192 : i64}, {depth = 4 : i64, iterator = "%arg16", mapped_to = "x", reuse_type = "no_reuse", volume = 0 : i64}], temporal = [{depth = 0 : i64, iterator = "%arg12", reuse_type = "total_reuse", volume = 8192 : i64}, {depth = 1 : i64, iterator = "%arg13", reuse_type = "total_reuse", volume = 8192 : i64}, {depth = 2 : i64, iterator = "%arg14", reuse_type = "no_reuse", volume = 0 : i64}]}} : memref<*xf32> to memref<64x32xf32, strided<[?, 1], offset: ?>>
                %alloc = memref.alloc() : memref<64x32xf32>
                memref.copy %reinterpret_cast_0, %alloc : memref<64x32xf32, strided<[?, 1], offset: ?>> to memref<64x32xf32>
                %23 = bufferization.to_tensor %alloc restrict writable : memref<64x32xf32> to tensor<64x32xf32>
                %24 = arith.muli %20, %arg7 : index
                %25 = affine.apply affine_map<(d0, d1)[s0] -> (d0 * 512 + d1 * 64 + s0)>(%arg12, %arg15)[%24]
                %reinterpret_cast_1 = memref.reinterpret_cast %arg1 to offset: [%25], sizes: [32, 64], strides: [%arg7, 1] {tmd.reuse = {sequential = [{depth = 5 : i64, iterator = "%arg17", reuse_type = "no_reuse", volume = 0 : i64}], spatial = [{depth = 3 : i64, iterator = "%arg15", mapped_to = "y", reuse_type = "no_reuse", volume = 0 : i64}, {depth = 4 : i64, iterator = "%arg16", mapped_to = "x", reuse_type = "total_reuse", volume = 8192 : i64}], temporal = [{depth = 0 : i64, iterator = "%arg12", reuse_type = "no_reuse", volume = 0 : i64}, {depth = 1 : i64, iterator = "%arg13", reuse_type = "total_reuse", volume = 8192 : i64}, {depth = 2 : i64, iterator = "%arg14", reuse_type = "total_reuse", volume = 8192 : i64}]}} : memref<*xf32> to memref<32x64xf32, strided<[?, 1], offset: ?>>
                %alloc_2 = memref.alloc() : memref<32x64xf32>
                memref.copy %reinterpret_cast_1, %alloc_2 : memref<32x64xf32, strided<[?, 1], offset: ?>> to memref<32x64xf32>
                %26 = bufferization.to_tensor %alloc_2 restrict writable : memref<32x64xf32> to tensor<32x64xf32>
                %27 = linalg.matmul ins(%23, %26 : tensor<64x32xf32>, tensor<32x64xf32>) outs(%10 : tensor<64x64xf32>) -> tensor<64x64xf32>
                %28 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%arg18, %27 : tensor<64x64xf32>, tensor<64x64xf32>) outs(%arg18 : tensor<64x64xf32>) {
                ^bb0(%in: f32, %in_3: f32, %out: f32):
                  %29 = arith.addf %in, %in_3 : f32
                  linalg.yield %29 : f32
                } -> tensor<64x64xf32>
                scf.yield %28 : tensor<64x64xf32>
              }
              %14 = arith.muli %11, %arg8 : index
              %15 = affine.apply affine_map<(d0, d1)[s0] -> (d0 * 512 + d1 * 64 + s0)>(%arg12, %arg15)[%14]
              %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%15], sizes: [64, 64], strides: [%arg8, 1] {tmd.reuse = {spatial = [{depth = 3 : i64, iterator = "%arg15", mapped_to = "y", reuse_type = "no_reuse", volume = 0 : i64}, {depth = 4 : i64, iterator = "%arg16", mapped_to = "x", reuse_type = "no_reuse", volume = 0 : i64}], temporal = [{depth = 0 : i64, iterator = "%arg12", reuse_type = "no_reuse", volume = 0 : i64}, {depth = 1 : i64, iterator = "%arg13", reuse_type = "total_reuse", volume = 16384 : i64}, {depth = 2 : i64, iterator = "%arg14", reuse_type = "no_reuse", volume = 0 : i64}]}} : memref<*xf32> to memref<64x64xf32, strided<[?, 1], offset: ?>>
              %16 = affine.min affine_map<(d0, d1, d2)[s0] -> (d0 * -64 + s0, 64)>(%8, %7, %arg13)[%arg3]
              %17 = affine.min affine_map<(d0, d1, d2)[s0] -> (d1 * -64 + s0, 64)>(%8, %7, %arg13)[%arg4]
              %extracted_slice = tensor.extract_slice %13[0, 0] [%16, %17] [1, 1] : tensor<64x64xf32> to tensor<?x?xf32>
              %18 = affine.min affine_map<(d0, d1, d2)[s0] -> (d0 * -64 + s0, 64)>(%8, %7, %arg13)[%arg3]
              %19 = affine.min affine_map<(d0, d1, d2)[s0] -> (d1 * -64 + s0, 64)>(%8, %7, %arg13)[%arg4]
              %subview = memref.subview %reinterpret_cast[0, 0] [%18, %19] [1, 1] : memref<64x64xf32, strided<[?, 1], offset: ?>> to memref<?x?xf32, strided<[?, 1], offset: ?>>
              bufferization.materialize_in_destination %extracted_slice in writable %subview : (tensor<?x?xf32>, memref<?x?xf32, strided<[?, 1], offset: ?>>) -> ()
            } {tmd.mapped_to = "x"}
          } {tmd.mapped_to = "y"}
        }
      }
    }
    return
  }
  func.func @matmul_kernel__d0i0_d1i1_f2_f0_f1(%arg0: memref<*xf32> {tt.divisibility = 16 : i32}, %arg1: memref<*xf32> {tt.divisibility = 16 : i32}, %arg2: memref<*xf32> {tt.divisibility = 16 : i32}, %arg3: index {tt.divisibility = 16 : i32}, %arg4: index {tt.divisibility = 16 : i32}, %arg5: index {tt.divisibility = 16 : i32}, %arg6: index {tt.divisibility = 16 : i32}, %arg7: index {tt.divisibility = 16 : i32}, %arg8: index {tt.divisibility = 16 : i32}, %arg9: index, %arg10: index, %arg11: index) {
    affine.for %arg12 = 0 to affine_map<(d0, d1, d2) -> (d2)>(%arg9, %arg10, %arg11) {
      affine.for %arg13 = 0 to affine_map<(d0, d1, d2) -> (d0 ceildiv 8)>(%arg9, %arg10, %arg11) {
        affine.for %arg14 = 0 to affine_map<(d0, d1, d2) -> (d1 ceildiv 8)>(%arg9, %arg10, %arg11) {
          affine.parallel (%arg15) = (0) to (8) {
            %7 = affine.apply affine_map<(d0, d1) -> (d0 * 8 + d1)>(%arg14, %arg15)
            affine.parallel (%arg16) = (0) to (8) {
              %8 = affine.apply affine_map<(d0, d1) -> (d0 * 8 + d1)>(%arg13, %arg16)
              %cst = arith.constant 0.000000e+00 : f32
              %9 = tensor.empty() : tensor<64x64xf32>
              %10 = linalg.fill ins(%cst : f32) outs(%9 : tensor<64x64xf32>) -> tensor<64x64xf32>
              %c64 = arith.constant 64 : index
              %11 = arith.muli %8, %c64 : index
              %12 = affine.apply affine_map<()[s0] -> ((s0 + 31) floordiv 32)>()[%arg5]
              %c0 = arith.constant 0 : index
              %c1 = arith.constant 1 : index
              %13 = scf.for %arg17 = %c0 to %12 step %c1 iter_args(%arg18 = %10) -> (tensor<64x64xf32>) {
                %c32 = arith.constant 32 : index
                %20 = arith.muli %arg17, %c32 : index
                %21 = arith.muli %11, %arg6 : index
                %22 = affine.apply affine_map<(d0)[s0] -> (d0 * 32 + s0)>(%arg17)[%21]
                %reinterpret_cast_0 = memref.reinterpret_cast %arg0 to offset: [%22], sizes: [64, 32], strides: [%arg6, 1] {tmd.reuse = {sequential = [{depth = 5 : i64, iterator = "%arg17", reuse_type = "no_reuse", volume = 0 : i64}], spatial = [{depth = 3 : i64, iterator = "%arg15", mapped_to = "y", reuse_type = "total_reuse", volume = 8192 : i64}, {depth = 4 : i64, iterator = "%arg16", mapped_to = "x", reuse_type = "no_reuse", volume = 0 : i64}], temporal = [{depth = 0 : i64, iterator = "%arg12", reuse_type = "total_reuse", volume = 8192 : i64}, {depth = 1 : i64, iterator = "%arg13", reuse_type = "no_reuse", volume = 0 : i64}, {depth = 2 : i64, iterator = "%arg14", reuse_type = "total_reuse", volume = 8192 : i64}]}} : memref<*xf32> to memref<64x32xf32, strided<[?, 1], offset: ?>>
                %alloc = memref.alloc() : memref<64x32xf32>
                memref.copy %reinterpret_cast_0, %alloc : memref<64x32xf32, strided<[?, 1], offset: ?>> to memref<64x32xf32>
                %23 = bufferization.to_tensor %alloc restrict writable : memref<64x32xf32> to tensor<64x32xf32>
                %24 = arith.muli %20, %arg7 : index
                %25 = affine.apply affine_map<(d0, d1)[s0] -> (d0 * 512 + d1 * 64 + s0)>(%arg14, %arg15)[%24]
                %reinterpret_cast_1 = memref.reinterpret_cast %arg1 to offset: [%25], sizes: [32, 64], strides: [%arg7, 1] {tmd.reuse = {sequential = [{depth = 5 : i64, iterator = "%arg17", reuse_type = "no_reuse", volume = 0 : i64}], spatial = [{depth = 3 : i64, iterator = "%arg15", mapped_to = "y", reuse_type = "no_reuse", volume = 0 : i64}, {depth = 4 : i64, iterator = "%arg16", mapped_to = "x", reuse_type = "total_reuse", volume = 8192 : i64}], temporal = [{depth = 0 : i64, iterator = "%arg12", reuse_type = "total_reuse", volume = 8192 : i64}, {depth = 1 : i64, iterator = "%arg13", reuse_type = "total_reuse", volume = 8192 : i64}, {depth = 2 : i64, iterator = "%arg14", reuse_type = "no_reuse", volume = 0 : i64}]}} : memref<*xf32> to memref<32x64xf32, strided<[?, 1], offset: ?>>
                %alloc_2 = memref.alloc() : memref<32x64xf32>
                memref.copy %reinterpret_cast_1, %alloc_2 : memref<32x64xf32, strided<[?, 1], offset: ?>> to memref<32x64xf32>
                %26 = bufferization.to_tensor %alloc_2 restrict writable : memref<32x64xf32> to tensor<32x64xf32>
                %27 = linalg.matmul ins(%23, %26 : tensor<64x32xf32>, tensor<32x64xf32>) outs(%10 : tensor<64x64xf32>) -> tensor<64x64xf32>
                %28 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%arg18, %27 : tensor<64x64xf32>, tensor<64x64xf32>) outs(%arg18 : tensor<64x64xf32>) {
                ^bb0(%in: f32, %in_3: f32, %out: f32):
                  %29 = arith.addf %in, %in_3 : f32
                  linalg.yield %29 : f32
                } -> tensor<64x64xf32>
                scf.yield %28 : tensor<64x64xf32>
              }
              %14 = arith.muli %11, %arg8 : index
              %15 = affine.apply affine_map<(d0, d1)[s0] -> (d0 * 512 + d1 * 64 + s0)>(%arg14, %arg15)[%14]
              %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%15], sizes: [64, 64], strides: [%arg8, 1] {tmd.reuse = {spatial = [{depth = 3 : i64, iterator = "%arg15", mapped_to = "y", reuse_type = "no_reuse", volume = 0 : i64}, {depth = 4 : i64, iterator = "%arg16", mapped_to = "x", reuse_type = "no_reuse", volume = 0 : i64}], temporal = [{depth = 0 : i64, iterator = "%arg12", reuse_type = "total_reuse", volume = 16384 : i64}, {depth = 1 : i64, iterator = "%arg13", reuse_type = "no_reuse", volume = 0 : i64}, {depth = 2 : i64, iterator = "%arg14", reuse_type = "no_reuse", volume = 0 : i64}]}} : memref<*xf32> to memref<64x64xf32, strided<[?, 1], offset: ?>>
              %16 = affine.min affine_map<(d0, d1, d2)[s0] -> (d0 * -64 + s0, 64)>(%8, %7, %arg12)[%arg3]
              %17 = affine.min affine_map<(d0, d1, d2)[s0] -> (d1 * -64 + s0, 64)>(%8, %7, %arg12)[%arg4]
              %extracted_slice = tensor.extract_slice %13[0, 0] [%16, %17] [1, 1] : tensor<64x64xf32> to tensor<?x?xf32>
              %18 = affine.min affine_map<(d0, d1, d2)[s0] -> (d0 * -64 + s0, 64)>(%8, %7, %arg12)[%arg3]
              %19 = affine.min affine_map<(d0, d1, d2)[s0] -> (d1 * -64 + s0, 64)>(%8, %7, %arg12)[%arg4]
              %subview = memref.subview %reinterpret_cast[0, 0] [%18, %19] [1, 1] : memref<64x64xf32, strided<[?, 1], offset: ?>> to memref<?x?xf32, strided<[?, 1], offset: ?>>
              bufferization.materialize_in_destination %extracted_slice in writable %subview : (tensor<?x?xf32>, memref<?x?xf32, strided<[?, 1], offset: ?>>) -> ()
            } {tmd.mapped_to = "x"}
          } {tmd.mapped_to = "y"}
        }
      }
    }
    return
  }
  func.func @matmul_kernel__d0i0_d1i1_f2_f1_f0(%arg0: memref<*xf32> {tt.divisibility = 16 : i32}, %arg1: memref<*xf32> {tt.divisibility = 16 : i32}, %arg2: memref<*xf32> {tt.divisibility = 16 : i32}, %arg3: index {tt.divisibility = 16 : i32}, %arg4: index {tt.divisibility = 16 : i32}, %arg5: index {tt.divisibility = 16 : i32}, %arg6: index {tt.divisibility = 16 : i32}, %arg7: index {tt.divisibility = 16 : i32}, %arg8: index {tt.divisibility = 16 : i32}, %arg9: index, %arg10: index, %arg11: index) {
    affine.for %arg12 = 0 to affine_map<(d0, d1, d2) -> (d2)>(%arg9, %arg10, %arg11) {
      affine.for %arg13 = 0 to affine_map<(d0, d1, d2) -> (d1 ceildiv 8)>(%arg9, %arg10, %arg11) {
        affine.for %arg14 = 0 to affine_map<(d0, d1, d2) -> (d0 ceildiv 8)>(%arg9, %arg10, %arg11) {
          affine.parallel (%arg15) = (0) to (8) {
            %7 = affine.apply affine_map<(d0, d1) -> (d0 * 8 + d1)>(%arg13, %arg15)
            affine.parallel (%arg16) = (0) to (8) {
              %8 = affine.apply affine_map<(d0, d1) -> (d0 * 8 + d1)>(%arg14, %arg16)
              %cst = arith.constant 0.000000e+00 : f32
              %9 = tensor.empty() : tensor<64x64xf32>
              %10 = linalg.fill ins(%cst : f32) outs(%9 : tensor<64x64xf32>) -> tensor<64x64xf32>
              %c64 = arith.constant 64 : index
              %11 = arith.muli %8, %c64 : index
              %12 = affine.apply affine_map<()[s0] -> ((s0 + 31) floordiv 32)>()[%arg5]
              %c0 = arith.constant 0 : index
              %c1 = arith.constant 1 : index
              %13 = scf.for %arg17 = %c0 to %12 step %c1 iter_args(%arg18 = %10) -> (tensor<64x64xf32>) {
                %c32 = arith.constant 32 : index
                %20 = arith.muli %arg17, %c32 : index
                %21 = arith.muli %11, %arg6 : index
                %22 = affine.apply affine_map<(d0)[s0] -> (d0 * 32 + s0)>(%arg17)[%21]
                %reinterpret_cast_0 = memref.reinterpret_cast %arg0 to offset: [%22], sizes: [64, 32], strides: [%arg6, 1] {tmd.reuse = {sequential = [{depth = 5 : i64, iterator = "%arg17", reuse_type = "no_reuse", volume = 0 : i64}], spatial = [{depth = 3 : i64, iterator = "%arg15", mapped_to = "y", reuse_type = "total_reuse", volume = 8192 : i64}, {depth = 4 : i64, iterator = "%arg16", mapped_to = "x", reuse_type = "no_reuse", volume = 0 : i64}], temporal = [{depth = 0 : i64, iterator = "%arg12", reuse_type = "total_reuse", volume = 8192 : i64}, {depth = 1 : i64, iterator = "%arg13", reuse_type = "total_reuse", volume = 8192 : i64}, {depth = 2 : i64, iterator = "%arg14", reuse_type = "no_reuse", volume = 0 : i64}]}} : memref<*xf32> to memref<64x32xf32, strided<[?, 1], offset: ?>>
                %alloc = memref.alloc() : memref<64x32xf32>
                memref.copy %reinterpret_cast_0, %alloc : memref<64x32xf32, strided<[?, 1], offset: ?>> to memref<64x32xf32>
                %23 = bufferization.to_tensor %alloc restrict writable : memref<64x32xf32> to tensor<64x32xf32>
                %24 = arith.muli %20, %arg7 : index
                %25 = affine.apply affine_map<(d0, d1)[s0] -> (d0 * 512 + d1 * 64 + s0)>(%arg13, %arg15)[%24]
                %reinterpret_cast_1 = memref.reinterpret_cast %arg1 to offset: [%25], sizes: [32, 64], strides: [%arg7, 1] {tmd.reuse = {sequential = [{depth = 5 : i64, iterator = "%arg17", reuse_type = "no_reuse", volume = 0 : i64}], spatial = [{depth = 3 : i64, iterator = "%arg15", mapped_to = "y", reuse_type = "no_reuse", volume = 0 : i64}, {depth = 4 : i64, iterator = "%arg16", mapped_to = "x", reuse_type = "total_reuse", volume = 8192 : i64}], temporal = [{depth = 0 : i64, iterator = "%arg12", reuse_type = "total_reuse", volume = 8192 : i64}, {depth = 1 : i64, iterator = "%arg13", reuse_type = "no_reuse", volume = 0 : i64}, {depth = 2 : i64, iterator = "%arg14", reuse_type = "total_reuse", volume = 8192 : i64}]}} : memref<*xf32> to memref<32x64xf32, strided<[?, 1], offset: ?>>
                %alloc_2 = memref.alloc() : memref<32x64xf32>
                memref.copy %reinterpret_cast_1, %alloc_2 : memref<32x64xf32, strided<[?, 1], offset: ?>> to memref<32x64xf32>
                %26 = bufferization.to_tensor %alloc_2 restrict writable : memref<32x64xf32> to tensor<32x64xf32>
                %27 = linalg.matmul ins(%23, %26 : tensor<64x32xf32>, tensor<32x64xf32>) outs(%10 : tensor<64x64xf32>) -> tensor<64x64xf32>
                %28 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%arg18, %27 : tensor<64x64xf32>, tensor<64x64xf32>) outs(%arg18 : tensor<64x64xf32>) {
                ^bb0(%in: f32, %in_3: f32, %out: f32):
                  %29 = arith.addf %in, %in_3 : f32
                  linalg.yield %29 : f32
                } -> tensor<64x64xf32>
                scf.yield %28 : tensor<64x64xf32>
              }
              %14 = arith.muli %11, %arg8 : index
              %15 = affine.apply affine_map<(d0, d1)[s0] -> (d0 * 512 + d1 * 64 + s0)>(%arg13, %arg15)[%14]
              %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%15], sizes: [64, 64], strides: [%arg8, 1] {tmd.reuse = {spatial = [{depth = 3 : i64, iterator = "%arg15", mapped_to = "y", reuse_type = "no_reuse", volume = 0 : i64}, {depth = 4 : i64, iterator = "%arg16", mapped_to = "x", reuse_type = "no_reuse", volume = 0 : i64}], temporal = [{depth = 0 : i64, iterator = "%arg12", reuse_type = "total_reuse", volume = 16384 : i64}, {depth = 1 : i64, iterator = "%arg13", reuse_type = "no_reuse", volume = 0 : i64}, {depth = 2 : i64, iterator = "%arg14", reuse_type = "no_reuse", volume = 0 : i64}]}} : memref<*xf32> to memref<64x64xf32, strided<[?, 1], offset: ?>>
              %16 = affine.min affine_map<(d0, d1, d2)[s0] -> (d0 * -64 + s0, 64)>(%8, %7, %arg12)[%arg3]
              %17 = affine.min affine_map<(d0, d1, d2)[s0] -> (d1 * -64 + s0, 64)>(%8, %7, %arg12)[%arg4]
              %extracted_slice = tensor.extract_slice %13[0, 0] [%16, %17] [1, 1] : tensor<64x64xf32> to tensor<?x?xf32>
              %18 = affine.min affine_map<(d0, d1, d2)[s0] -> (d0 * -64 + s0, 64)>(%8, %7, %arg12)[%arg3]
              %19 = affine.min affine_map<(d0, d1, d2)[s0] -> (d1 * -64 + s0, 64)>(%8, %7, %arg12)[%arg4]
              %subview = memref.subview %reinterpret_cast[0, 0] [%18, %19] [1, 1] : memref<64x64xf32, strided<[?, 1], offset: ?>> to memref<?x?xf32, strided<[?, 1], offset: ?>>
              bufferization.materialize_in_destination %extracted_slice in writable %subview : (tensor<?x?xf32>, memref<?x?xf32, strided<[?, 1], offset: ?>>) -> ()
            } {tmd.mapped_to = "x"}
          } {tmd.mapped_to = "y"}
        }
      }
    }
    return
  }
  func.func @matmul_kernel__d0i0_d1i2_f0_f1_f2(%arg0: memref<*xf32> {tt.divisibility = 16 : i32}, %arg1: memref<*xf32> {tt.divisibility = 16 : i32}, %arg2: memref<*xf32> {tt.divisibility = 16 : i32}, %arg3: index {tt.divisibility = 16 : i32}, %arg4: index {tt.divisibility = 16 : i32}, %arg5: index {tt.divisibility = 16 : i32}, %arg6: index {tt.divisibility = 16 : i32}, %arg7: index {tt.divisibility = 16 : i32}, %arg8: index {tt.divisibility = 16 : i32}, %arg9: index, %arg10: index, %arg11: index) {
    affine.for %arg12 = 0 to affine_map<(d0, d1, d2) -> (d0 ceildiv 8)>(%arg9, %arg10, %arg11) {
      affine.for %arg13 = 0 to affine_map<(d0, d1, d2) -> (d1)>(%arg9, %arg10, %arg11) {
        affine.for %arg14 = 0 to affine_map<(d0, d1, d2) -> (d2 ceildiv 8)>(%arg9, %arg10, %arg11) {
          affine.parallel (%arg15) = (0) to (8) {
            %7 = affine.apply affine_map<(d0, d1) -> (d0 * 8 + d1)>(%arg14, %arg15)
            affine.parallel (%arg16) = (0) to (8) {
              %8 = affine.apply affine_map<(d0, d1) -> (d0 * 8 + d1)>(%arg12, %arg16)
              %cst = arith.constant 0.000000e+00 : f32
              %9 = tensor.empty() : tensor<64x64xf32>
              %10 = linalg.fill ins(%cst : f32) outs(%9 : tensor<64x64xf32>) -> tensor<64x64xf32>
              %c64 = arith.constant 64 : index
              %11 = arith.muli %8, %c64 : index
              %12 = affine.apply affine_map<()[s0] -> ((s0 + 31) floordiv 32)>()[%arg5]
              %c0 = arith.constant 0 : index
              %c1 = arith.constant 1 : index
              %13 = scf.for %arg17 = %c0 to %12 step %c1 iter_args(%arg18 = %10) -> (tensor<64x64xf32>) {
                %c32 = arith.constant 32 : index
                %20 = arith.muli %arg17, %c32 : index
                %21 = arith.muli %11, %arg6 : index
                %22 = affine.apply affine_map<(d0)[s0] -> (d0 * 32 + s0)>(%arg17)[%21]
                %reinterpret_cast_0 = memref.reinterpret_cast %arg0 to offset: [%22], sizes: [64, 32], strides: [%arg6, 1] {tmd.reuse = {sequential = [{depth = 5 : i64, iterator = "%arg17", reuse_type = "no_reuse", volume = 0 : i64}], spatial = [{depth = 3 : i64, iterator = "%arg15", mapped_to = "y", reuse_type = "total_reuse", volume = 8192 : i64}, {depth = 4 : i64, iterator = "%arg16", mapped_to = "x", reuse_type = "no_reuse", volume = 0 : i64}], temporal = [{depth = 0 : i64, iterator = "%arg12", reuse_type = "no_reuse", volume = 0 : i64}, {depth = 1 : i64, iterator = "%arg13", reuse_type = "total_reuse", volume = 8192 : i64}, {depth = 2 : i64, iterator = "%arg14", reuse_type = "total_reuse", volume = 8192 : i64}]}} : memref<*xf32> to memref<64x32xf32, strided<[?, 1], offset: ?>>
                %alloc = memref.alloc() : memref<64x32xf32>
                memref.copy %reinterpret_cast_0, %alloc : memref<64x32xf32, strided<[?, 1], offset: ?>> to memref<64x32xf32>
                %23 = bufferization.to_tensor %alloc restrict writable : memref<64x32xf32> to tensor<64x32xf32>
                %24 = arith.muli %20, %arg7 : index
                %25 = affine.apply affine_map<(d0)[s0] -> (d0 * 64 + s0)>(%arg13)[%24]
                %reinterpret_cast_1 = memref.reinterpret_cast %arg1 to offset: [%25], sizes: [32, 64], strides: [%arg7, 1] {tmd.reuse = {sequential = [{depth = 5 : i64, iterator = "%arg17", reuse_type = "no_reuse", volume = 0 : i64}], spatial = [{depth = 3 : i64, iterator = "%arg15", mapped_to = "y", reuse_type = "total_reuse", volume = 8192 : i64}, {depth = 4 : i64, iterator = "%arg16", mapped_to = "x", reuse_type = "total_reuse", volume = 8192 : i64}], temporal = [{depth = 0 : i64, iterator = "%arg12", reuse_type = "total_reuse", volume = 8192 : i64}, {depth = 1 : i64, iterator = "%arg13", reuse_type = "no_reuse", volume = 0 : i64}, {depth = 2 : i64, iterator = "%arg14", reuse_type = "total_reuse", volume = 8192 : i64}]}} : memref<*xf32> to memref<32x64xf32, strided<[?, 1], offset: ?>>
                %alloc_2 = memref.alloc() : memref<32x64xf32>
                memref.copy %reinterpret_cast_1, %alloc_2 : memref<32x64xf32, strided<[?, 1], offset: ?>> to memref<32x64xf32>
                %26 = bufferization.to_tensor %alloc_2 restrict writable : memref<32x64xf32> to tensor<32x64xf32>
                %27 = linalg.matmul ins(%23, %26 : tensor<64x32xf32>, tensor<32x64xf32>) outs(%10 : tensor<64x64xf32>) -> tensor<64x64xf32>
                %28 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%arg18, %27 : tensor<64x64xf32>, tensor<64x64xf32>) outs(%arg18 : tensor<64x64xf32>) {
                ^bb0(%in: f32, %in_3: f32, %out: f32):
                  %29 = arith.addf %in, %in_3 : f32
                  linalg.yield %29 : f32
                } -> tensor<64x64xf32>
                scf.yield %28 : tensor<64x64xf32>
              }
              %14 = arith.muli %11, %arg8 : index
              %15 = affine.apply affine_map<(d0)[s0] -> (d0 * 64 + s0)>(%arg13)[%14]
              %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%15], sizes: [64, 64], strides: [%arg8, 1] {tmd.reuse = {spatial = [{depth = 3 : i64, iterator = "%arg15", mapped_to = "y", reuse_type = "total_reuse", volume = 16384 : i64}, {depth = 4 : i64, iterator = "%arg16", mapped_to = "x", reuse_type = "no_reuse", volume = 0 : i64}], temporal = [{depth = 0 : i64, iterator = "%arg12", reuse_type = "no_reuse", volume = 0 : i64}, {depth = 1 : i64, iterator = "%arg13", reuse_type = "no_reuse", volume = 0 : i64}, {depth = 2 : i64, iterator = "%arg14", reuse_type = "total_reuse", volume = 16384 : i64}]}} : memref<*xf32> to memref<64x64xf32, strided<[?, 1], offset: ?>>
              %16 = affine.min affine_map<(d0, d1, d2)[s0] -> (d0 * -64 + s0, 64)>(%8, %arg13, %7)[%arg3]
              %17 = affine.min affine_map<(d0, d1, d2)[s0] -> (d1 * -64 + s0, 64)>(%8, %arg13, %7)[%arg4]
              %extracted_slice = tensor.extract_slice %13[0, 0] [%16, %17] [1, 1] : tensor<64x64xf32> to tensor<?x?xf32>
              %18 = affine.min affine_map<(d0, d1, d2)[s0] -> (d0 * -64 + s0, 64)>(%8, %arg13, %7)[%arg3]
              %19 = affine.min affine_map<(d0, d1, d2)[s0] -> (d1 * -64 + s0, 64)>(%8, %arg13, %7)[%arg4]
              %subview = memref.subview %reinterpret_cast[0, 0] [%18, %19] [1, 1] : memref<64x64xf32, strided<[?, 1], offset: ?>> to memref<?x?xf32, strided<[?, 1], offset: ?>>
              bufferization.materialize_in_destination %extracted_slice in writable %subview : (tensor<?x?xf32>, memref<?x?xf32, strided<[?, 1], offset: ?>>) -> ()
            } {tmd.mapped_to = "x"}
          } {tmd.mapped_to = "y"}
        }
      }
    }
    return
  }
  func.func @matmul_kernel__d0i0_d1i2_f0_f2_f1(%arg0: memref<*xf32> {tt.divisibility = 16 : i32}, %arg1: memref<*xf32> {tt.divisibility = 16 : i32}, %arg2: memref<*xf32> {tt.divisibility = 16 : i32}, %arg3: index {tt.divisibility = 16 : i32}, %arg4: index {tt.divisibility = 16 : i32}, %arg5: index {tt.divisibility = 16 : i32}, %arg6: index {tt.divisibility = 16 : i32}, %arg7: index {tt.divisibility = 16 : i32}, %arg8: index {tt.divisibility = 16 : i32}, %arg9: index, %arg10: index, %arg11: index) {
    affine.for %arg12 = 0 to affine_map<(d0, d1, d2) -> (d0 ceildiv 8)>(%arg9, %arg10, %arg11) {
      affine.for %arg13 = 0 to affine_map<(d0, d1, d2) -> (d2 ceildiv 8)>(%arg9, %arg10, %arg11) {
        affine.for %arg14 = 0 to affine_map<(d0, d1, d2) -> (d1)>(%arg9, %arg10, %arg11) {
          affine.parallel (%arg15) = (0) to (8) {
            %7 = affine.apply affine_map<(d0, d1) -> (d0 * 8 + d1)>(%arg13, %arg15)
            affine.parallel (%arg16) = (0) to (8) {
              %8 = affine.apply affine_map<(d0, d1) -> (d0 * 8 + d1)>(%arg12, %arg16)
              %cst = arith.constant 0.000000e+00 : f32
              %9 = tensor.empty() : tensor<64x64xf32>
              %10 = linalg.fill ins(%cst : f32) outs(%9 : tensor<64x64xf32>) -> tensor<64x64xf32>
              %c64 = arith.constant 64 : index
              %11 = arith.muli %8, %c64 : index
              %12 = affine.apply affine_map<()[s0] -> ((s0 + 31) floordiv 32)>()[%arg5]
              %c0 = arith.constant 0 : index
              %c1 = arith.constant 1 : index
              %13 = scf.for %arg17 = %c0 to %12 step %c1 iter_args(%arg18 = %10) -> (tensor<64x64xf32>) {
                %c32 = arith.constant 32 : index
                %20 = arith.muli %arg17, %c32 : index
                %21 = arith.muli %11, %arg6 : index
                %22 = affine.apply affine_map<(d0)[s0] -> (d0 * 32 + s0)>(%arg17)[%21]
                %reinterpret_cast_0 = memref.reinterpret_cast %arg0 to offset: [%22], sizes: [64, 32], strides: [%arg6, 1] {tmd.reuse = {sequential = [{depth = 5 : i64, iterator = "%arg17", reuse_type = "no_reuse", volume = 0 : i64}], spatial = [{depth = 3 : i64, iterator = "%arg15", mapped_to = "y", reuse_type = "total_reuse", volume = 8192 : i64}, {depth = 4 : i64, iterator = "%arg16", mapped_to = "x", reuse_type = "no_reuse", volume = 0 : i64}], temporal = [{depth = 0 : i64, iterator = "%arg12", reuse_type = "no_reuse", volume = 0 : i64}, {depth = 1 : i64, iterator = "%arg13", reuse_type = "total_reuse", volume = 8192 : i64}, {depth = 2 : i64, iterator = "%arg14", reuse_type = "total_reuse", volume = 8192 : i64}]}} : memref<*xf32> to memref<64x32xf32, strided<[?, 1], offset: ?>>
                %alloc = memref.alloc() : memref<64x32xf32>
                memref.copy %reinterpret_cast_0, %alloc : memref<64x32xf32, strided<[?, 1], offset: ?>> to memref<64x32xf32>
                %23 = bufferization.to_tensor %alloc restrict writable : memref<64x32xf32> to tensor<64x32xf32>
                %24 = arith.muli %20, %arg7 : index
                %25 = affine.apply affine_map<(d0)[s0] -> (d0 * 64 + s0)>(%arg14)[%24]
                %reinterpret_cast_1 = memref.reinterpret_cast %arg1 to offset: [%25], sizes: [32, 64], strides: [%arg7, 1] {tmd.reuse = {sequential = [{depth = 5 : i64, iterator = "%arg17", reuse_type = "no_reuse", volume = 0 : i64}], spatial = [{depth = 3 : i64, iterator = "%arg15", mapped_to = "y", reuse_type = "total_reuse", volume = 8192 : i64}, {depth = 4 : i64, iterator = "%arg16", mapped_to = "x", reuse_type = "total_reuse", volume = 8192 : i64}], temporal = [{depth = 0 : i64, iterator = "%arg12", reuse_type = "total_reuse", volume = 8192 : i64}, {depth = 1 : i64, iterator = "%arg13", reuse_type = "total_reuse", volume = 8192 : i64}, {depth = 2 : i64, iterator = "%arg14", reuse_type = "no_reuse", volume = 0 : i64}]}} : memref<*xf32> to memref<32x64xf32, strided<[?, 1], offset: ?>>
                %alloc_2 = memref.alloc() : memref<32x64xf32>
                memref.copy %reinterpret_cast_1, %alloc_2 : memref<32x64xf32, strided<[?, 1], offset: ?>> to memref<32x64xf32>
                %26 = bufferization.to_tensor %alloc_2 restrict writable : memref<32x64xf32> to tensor<32x64xf32>
                %27 = linalg.matmul ins(%23, %26 : tensor<64x32xf32>, tensor<32x64xf32>) outs(%10 : tensor<64x64xf32>) -> tensor<64x64xf32>
                %28 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%arg18, %27 : tensor<64x64xf32>, tensor<64x64xf32>) outs(%arg18 : tensor<64x64xf32>) {
                ^bb0(%in: f32, %in_3: f32, %out: f32):
                  %29 = arith.addf %in, %in_3 : f32
                  linalg.yield %29 : f32
                } -> tensor<64x64xf32>
                scf.yield %28 : tensor<64x64xf32>
              }
              %14 = arith.muli %11, %arg8 : index
              %15 = affine.apply affine_map<(d0)[s0] -> (d0 * 64 + s0)>(%arg14)[%14]
              %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%15], sizes: [64, 64], strides: [%arg8, 1] {tmd.reuse = {spatial = [{depth = 3 : i64, iterator = "%arg15", mapped_to = "y", reuse_type = "total_reuse", volume = 16384 : i64}, {depth = 4 : i64, iterator = "%arg16", mapped_to = "x", reuse_type = "no_reuse", volume = 0 : i64}], temporal = [{depth = 0 : i64, iterator = "%arg12", reuse_type = "no_reuse", volume = 0 : i64}, {depth = 1 : i64, iterator = "%arg13", reuse_type = "total_reuse", volume = 16384 : i64}, {depth = 2 : i64, iterator = "%arg14", reuse_type = "no_reuse", volume = 0 : i64}]}} : memref<*xf32> to memref<64x64xf32, strided<[?, 1], offset: ?>>
              %16 = affine.min affine_map<(d0, d1, d2)[s0] -> (d0 * -64 + s0, 64)>(%8, %arg14, %7)[%arg3]
              %17 = affine.min affine_map<(d0, d1, d2)[s0] -> (d1 * -64 + s0, 64)>(%8, %arg14, %7)[%arg4]
              %extracted_slice = tensor.extract_slice %13[0, 0] [%16, %17] [1, 1] : tensor<64x64xf32> to tensor<?x?xf32>
              %18 = affine.min affine_map<(d0, d1, d2)[s0] -> (d0 * -64 + s0, 64)>(%8, %arg14, %7)[%arg3]
              %19 = affine.min affine_map<(d0, d1, d2)[s0] -> (d1 * -64 + s0, 64)>(%8, %arg14, %7)[%arg4]
              %subview = memref.subview %reinterpret_cast[0, 0] [%18, %19] [1, 1] : memref<64x64xf32, strided<[?, 1], offset: ?>> to memref<?x?xf32, strided<[?, 1], offset: ?>>
              bufferization.materialize_in_destination %extracted_slice in writable %subview : (tensor<?x?xf32>, memref<?x?xf32, strided<[?, 1], offset: ?>>) -> ()
            } {tmd.mapped_to = "x"}
          } {tmd.mapped_to = "y"}
        }
      }
    }
    return
  }
  func.func @matmul_kernel__d0i0_d1i2_f1_f0_f2(%arg0: memref<*xf32> {tt.divisibility = 16 : i32}, %arg1: memref<*xf32> {tt.divisibility = 16 : i32}, %arg2: memref<*xf32> {tt.divisibility = 16 : i32}, %arg3: index {tt.divisibility = 16 : i32}, %arg4: index {tt.divisibility = 16 : i32}, %arg5: index {tt.divisibility = 16 : i32}, %arg6: index {tt.divisibility = 16 : i32}, %arg7: index {tt.divisibility = 16 : i32}, %arg8: index {tt.divisibility = 16 : i32}, %arg9: index, %arg10: index, %arg11: index) {
    affine.for %arg12 = 0 to affine_map<(d0, d1, d2) -> (d1)>(%arg9, %arg10, %arg11) {
      affine.for %arg13 = 0 to affine_map<(d0, d1, d2) -> (d0 ceildiv 8)>(%arg9, %arg10, %arg11) {
        affine.for %arg14 = 0 to affine_map<(d0, d1, d2) -> (d2 ceildiv 8)>(%arg9, %arg10, %arg11) {
          affine.parallel (%arg15) = (0) to (8) {
            %7 = affine.apply affine_map<(d0, d1) -> (d0 * 8 + d1)>(%arg14, %arg15)
            affine.parallel (%arg16) = (0) to (8) {
              %8 = affine.apply affine_map<(d0, d1) -> (d0 * 8 + d1)>(%arg13, %arg16)
              %cst = arith.constant 0.000000e+00 : f32
              %9 = tensor.empty() : tensor<64x64xf32>
              %10 = linalg.fill ins(%cst : f32) outs(%9 : tensor<64x64xf32>) -> tensor<64x64xf32>
              %c64 = arith.constant 64 : index
              %11 = arith.muli %8, %c64 : index
              %12 = affine.apply affine_map<()[s0] -> ((s0 + 31) floordiv 32)>()[%arg5]
              %c0 = arith.constant 0 : index
              %c1 = arith.constant 1 : index
              %13 = scf.for %arg17 = %c0 to %12 step %c1 iter_args(%arg18 = %10) -> (tensor<64x64xf32>) {
                %c32 = arith.constant 32 : index
                %20 = arith.muli %arg17, %c32 : index
                %21 = arith.muli %11, %arg6 : index
                %22 = affine.apply affine_map<(d0)[s0] -> (d0 * 32 + s0)>(%arg17)[%21]
                %reinterpret_cast_0 = memref.reinterpret_cast %arg0 to offset: [%22], sizes: [64, 32], strides: [%arg6, 1] {tmd.reuse = {sequential = [{depth = 5 : i64, iterator = "%arg17", reuse_type = "no_reuse", volume = 0 : i64}], spatial = [{depth = 3 : i64, iterator = "%arg15", mapped_to = "y", reuse_type = "total_reuse", volume = 8192 : i64}, {depth = 4 : i64, iterator = "%arg16", mapped_to = "x", reuse_type = "no_reuse", volume = 0 : i64}], temporal = [{depth = 0 : i64, iterator = "%arg12", reuse_type = "total_reuse", volume = 8192 : i64}, {depth = 1 : i64, iterator = "%arg13", reuse_type = "no_reuse", volume = 0 : i64}, {depth = 2 : i64, iterator = "%arg14", reuse_type = "total_reuse", volume = 8192 : i64}]}} : memref<*xf32> to memref<64x32xf32, strided<[?, 1], offset: ?>>
                %alloc = memref.alloc() : memref<64x32xf32>
                memref.copy %reinterpret_cast_0, %alloc : memref<64x32xf32, strided<[?, 1], offset: ?>> to memref<64x32xf32>
                %23 = bufferization.to_tensor %alloc restrict writable : memref<64x32xf32> to tensor<64x32xf32>
                %24 = arith.muli %20, %arg7 : index
                %25 = affine.apply affine_map<(d0)[s0] -> (d0 * 64 + s0)>(%arg12)[%24]
                %reinterpret_cast_1 = memref.reinterpret_cast %arg1 to offset: [%25], sizes: [32, 64], strides: [%arg7, 1] {tmd.reuse = {sequential = [{depth = 5 : i64, iterator = "%arg17", reuse_type = "no_reuse", volume = 0 : i64}], spatial = [{depth = 3 : i64, iterator = "%arg15", mapped_to = "y", reuse_type = "total_reuse", volume = 8192 : i64}, {depth = 4 : i64, iterator = "%arg16", mapped_to = "x", reuse_type = "total_reuse", volume = 8192 : i64}], temporal = [{depth = 0 : i64, iterator = "%arg12", reuse_type = "no_reuse", volume = 0 : i64}, {depth = 1 : i64, iterator = "%arg13", reuse_type = "total_reuse", volume = 8192 : i64}, {depth = 2 : i64, iterator = "%arg14", reuse_type = "total_reuse", volume = 8192 : i64}]}} : memref<*xf32> to memref<32x64xf32, strided<[?, 1], offset: ?>>
                %alloc_2 = memref.alloc() : memref<32x64xf32>
                memref.copy %reinterpret_cast_1, %alloc_2 : memref<32x64xf32, strided<[?, 1], offset: ?>> to memref<32x64xf32>
                %26 = bufferization.to_tensor %alloc_2 restrict writable : memref<32x64xf32> to tensor<32x64xf32>
                %27 = linalg.matmul ins(%23, %26 : tensor<64x32xf32>, tensor<32x64xf32>) outs(%10 : tensor<64x64xf32>) -> tensor<64x64xf32>
                %28 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%arg18, %27 : tensor<64x64xf32>, tensor<64x64xf32>) outs(%arg18 : tensor<64x64xf32>) {
                ^bb0(%in: f32, %in_3: f32, %out: f32):
                  %29 = arith.addf %in, %in_3 : f32
                  linalg.yield %29 : f32
                } -> tensor<64x64xf32>
                scf.yield %28 : tensor<64x64xf32>
              }
              %14 = arith.muli %11, %arg8 : index
              %15 = affine.apply affine_map<(d0)[s0] -> (d0 * 64 + s0)>(%arg12)[%14]
              %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%15], sizes: [64, 64], strides: [%arg8, 1] {tmd.reuse = {spatial = [{depth = 3 : i64, iterator = "%arg15", mapped_to = "y", reuse_type = "total_reuse", volume = 16384 : i64}, {depth = 4 : i64, iterator = "%arg16", mapped_to = "x", reuse_type = "no_reuse", volume = 0 : i64}], temporal = [{depth = 0 : i64, iterator = "%arg12", reuse_type = "no_reuse", volume = 0 : i64}, {depth = 1 : i64, iterator = "%arg13", reuse_type = "no_reuse", volume = 0 : i64}, {depth = 2 : i64, iterator = "%arg14", reuse_type = "total_reuse", volume = 16384 : i64}]}} : memref<*xf32> to memref<64x64xf32, strided<[?, 1], offset: ?>>
              %16 = affine.min affine_map<(d0, d1, d2)[s0] -> (d0 * -64 + s0, 64)>(%8, %arg12, %7)[%arg3]
              %17 = affine.min affine_map<(d0, d1, d2)[s0] -> (d1 * -64 + s0, 64)>(%8, %arg12, %7)[%arg4]
              %extracted_slice = tensor.extract_slice %13[0, 0] [%16, %17] [1, 1] : tensor<64x64xf32> to tensor<?x?xf32>
              %18 = affine.min affine_map<(d0, d1, d2)[s0] -> (d0 * -64 + s0, 64)>(%8, %arg12, %7)[%arg3]
              %19 = affine.min affine_map<(d0, d1, d2)[s0] -> (d1 * -64 + s0, 64)>(%8, %arg12, %7)[%arg4]
              %subview = memref.subview %reinterpret_cast[0, 0] [%18, %19] [1, 1] : memref<64x64xf32, strided<[?, 1], offset: ?>> to memref<?x?xf32, strided<[?, 1], offset: ?>>
              bufferization.materialize_in_destination %extracted_slice in writable %subview : (tensor<?x?xf32>, memref<?x?xf32, strided<[?, 1], offset: ?>>) -> ()
            } {tmd.mapped_to = "x"}
          } {tmd.mapped_to = "y"}
        }
      }
    }
    return
  }
  func.func @matmul_kernel__d0i0_d1i2_f1_f2_f0(%arg0: memref<*xf32> {tt.divisibility = 16 : i32}, %arg1: memref<*xf32> {tt.divisibility = 16 : i32}, %arg2: memref<*xf32> {tt.divisibility = 16 : i32}, %arg3: index {tt.divisibility = 16 : i32}, %arg4: index {tt.divisibility = 16 : i32}, %arg5: index {tt.divisibility = 16 : i32}, %arg6: index {tt.divisibility = 16 : i32}, %arg7: index {tt.divisibility = 16 : i32}, %arg8: index {tt.divisibility = 16 : i32}, %arg9: index, %arg10: index, %arg11: index) {
    affine.for %arg12 = 0 to affine_map<(d0, d1, d2) -> (d1)>(%arg9, %arg10, %arg11) {
      affine.for %arg13 = 0 to affine_map<(d0, d1, d2) -> (d2 ceildiv 8)>(%arg9, %arg10, %arg11) {
        affine.for %arg14 = 0 to affine_map<(d0, d1, d2) -> (d0 ceildiv 8)>(%arg9, %arg10, %arg11) {
          affine.parallel (%arg15) = (0) to (8) {
            %7 = affine.apply affine_map<(d0, d1) -> (d0 * 8 + d1)>(%arg13, %arg15)
            affine.parallel (%arg16) = (0) to (8) {
              %8 = affine.apply affine_map<(d0, d1) -> (d0 * 8 + d1)>(%arg14, %arg16)
              %cst = arith.constant 0.000000e+00 : f32
              %9 = tensor.empty() : tensor<64x64xf32>
              %10 = linalg.fill ins(%cst : f32) outs(%9 : tensor<64x64xf32>) -> tensor<64x64xf32>
              %c64 = arith.constant 64 : index
              %11 = arith.muli %8, %c64 : index
              %12 = affine.apply affine_map<()[s0] -> ((s0 + 31) floordiv 32)>()[%arg5]
              %c0 = arith.constant 0 : index
              %c1 = arith.constant 1 : index
              %13 = scf.for %arg17 = %c0 to %12 step %c1 iter_args(%arg18 = %10) -> (tensor<64x64xf32>) {
                %c32 = arith.constant 32 : index
                %20 = arith.muli %arg17, %c32 : index
                %21 = arith.muli %11, %arg6 : index
                %22 = affine.apply affine_map<(d0)[s0] -> (d0 * 32 + s0)>(%arg17)[%21]
                %reinterpret_cast_0 = memref.reinterpret_cast %arg0 to offset: [%22], sizes: [64, 32], strides: [%arg6, 1] {tmd.reuse = {sequential = [{depth = 5 : i64, iterator = "%arg17", reuse_type = "no_reuse", volume = 0 : i64}], spatial = [{depth = 3 : i64, iterator = "%arg15", mapped_to = "y", reuse_type = "total_reuse", volume = 8192 : i64}, {depth = 4 : i64, iterator = "%arg16", mapped_to = "x", reuse_type = "no_reuse", volume = 0 : i64}], temporal = [{depth = 0 : i64, iterator = "%arg12", reuse_type = "total_reuse", volume = 8192 : i64}, {depth = 1 : i64, iterator = "%arg13", reuse_type = "total_reuse", volume = 8192 : i64}, {depth = 2 : i64, iterator = "%arg14", reuse_type = "no_reuse", volume = 0 : i64}]}} : memref<*xf32> to memref<64x32xf32, strided<[?, 1], offset: ?>>
                %alloc = memref.alloc() : memref<64x32xf32>
                memref.copy %reinterpret_cast_0, %alloc : memref<64x32xf32, strided<[?, 1], offset: ?>> to memref<64x32xf32>
                %23 = bufferization.to_tensor %alloc restrict writable : memref<64x32xf32> to tensor<64x32xf32>
                %24 = arith.muli %20, %arg7 : index
                %25 = affine.apply affine_map<(d0)[s0] -> (d0 * 64 + s0)>(%arg12)[%24]
                %reinterpret_cast_1 = memref.reinterpret_cast %arg1 to offset: [%25], sizes: [32, 64], strides: [%arg7, 1] {tmd.reuse = {sequential = [{depth = 5 : i64, iterator = "%arg17", reuse_type = "no_reuse", volume = 0 : i64}], spatial = [{depth = 3 : i64, iterator = "%arg15", mapped_to = "y", reuse_type = "total_reuse", volume = 8192 : i64}, {depth = 4 : i64, iterator = "%arg16", mapped_to = "x", reuse_type = "total_reuse", volume = 8192 : i64}], temporal = [{depth = 0 : i64, iterator = "%arg12", reuse_type = "no_reuse", volume = 0 : i64}, {depth = 1 : i64, iterator = "%arg13", reuse_type = "total_reuse", volume = 8192 : i64}, {depth = 2 : i64, iterator = "%arg14", reuse_type = "total_reuse", volume = 8192 : i64}]}} : memref<*xf32> to memref<32x64xf32, strided<[?, 1], offset: ?>>
                %alloc_2 = memref.alloc() : memref<32x64xf32>
                memref.copy %reinterpret_cast_1, %alloc_2 : memref<32x64xf32, strided<[?, 1], offset: ?>> to memref<32x64xf32>
                %26 = bufferization.to_tensor %alloc_2 restrict writable : memref<32x64xf32> to tensor<32x64xf32>
                %27 = linalg.matmul ins(%23, %26 : tensor<64x32xf32>, tensor<32x64xf32>) outs(%10 : tensor<64x64xf32>) -> tensor<64x64xf32>
                %28 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%arg18, %27 : tensor<64x64xf32>, tensor<64x64xf32>) outs(%arg18 : tensor<64x64xf32>) {
                ^bb0(%in: f32, %in_3: f32, %out: f32):
                  %29 = arith.addf %in, %in_3 : f32
                  linalg.yield %29 : f32
                } -> tensor<64x64xf32>
                scf.yield %28 : tensor<64x64xf32>
              }
              %14 = arith.muli %11, %arg8 : index
              %15 = affine.apply affine_map<(d0)[s0] -> (d0 * 64 + s0)>(%arg12)[%14]
              %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%15], sizes: [64, 64], strides: [%arg8, 1] {tmd.reuse = {spatial = [{depth = 3 : i64, iterator = "%arg15", mapped_to = "y", reuse_type = "total_reuse", volume = 16384 : i64}, {depth = 4 : i64, iterator = "%arg16", mapped_to = "x", reuse_type = "no_reuse", volume = 0 : i64}], temporal = [{depth = 0 : i64, iterator = "%arg12", reuse_type = "no_reuse", volume = 0 : i64}, {depth = 1 : i64, iterator = "%arg13", reuse_type = "total_reuse", volume = 16384 : i64}, {depth = 2 : i64, iterator = "%arg14", reuse_type = "no_reuse", volume = 0 : i64}]}} : memref<*xf32> to memref<64x64xf32, strided<[?, 1], offset: ?>>
              %16 = affine.min affine_map<(d0, d1, d2)[s0] -> (d0 * -64 + s0, 64)>(%8, %arg12, %7)[%arg3]
              %17 = affine.min affine_map<(d0, d1, d2)[s0] -> (d1 * -64 + s0, 64)>(%8, %arg12, %7)[%arg4]
              %extracted_slice = tensor.extract_slice %13[0, 0] [%16, %17] [1, 1] : tensor<64x64xf32> to tensor<?x?xf32>
              %18 = affine.min affine_map<(d0, d1, d2)[s0] -> (d0 * -64 + s0, 64)>(%8, %arg12, %7)[%arg3]
              %19 = affine.min affine_map<(d0, d1, d2)[s0] -> (d1 * -64 + s0, 64)>(%8, %arg12, %7)[%arg4]
              %subview = memref.subview %reinterpret_cast[0, 0] [%18, %19] [1, 1] : memref<64x64xf32, strided<[?, 1], offset: ?>> to memref<?x?xf32, strided<[?, 1], offset: ?>>
              bufferization.materialize_in_destination %extracted_slice in writable %subview : (tensor<?x?xf32>, memref<?x?xf32, strided<[?, 1], offset: ?>>) -> ()
            } {tmd.mapped_to = "x"}
          } {tmd.mapped_to = "y"}
        }
      }
    }
    return
  }
  func.func @matmul_kernel__d0i0_d1i2_f2_f0_f1(%arg0: memref<*xf32> {tt.divisibility = 16 : i32}, %arg1: memref<*xf32> {tt.divisibility = 16 : i32}, %arg2: memref<*xf32> {tt.divisibility = 16 : i32}, %arg3: index {tt.divisibility = 16 : i32}, %arg4: index {tt.divisibility = 16 : i32}, %arg5: index {tt.divisibility = 16 : i32}, %arg6: index {tt.divisibility = 16 : i32}, %arg7: index {tt.divisibility = 16 : i32}, %arg8: index {tt.divisibility = 16 : i32}, %arg9: index, %arg10: index, %arg11: index) {
    affine.for %arg12 = 0 to affine_map<(d0, d1, d2) -> (d2 ceildiv 8)>(%arg9, %arg10, %arg11) {
      affine.for %arg13 = 0 to affine_map<(d0, d1, d2) -> (d0 ceildiv 8)>(%arg9, %arg10, %arg11) {
        affine.for %arg14 = 0 to affine_map<(d0, d1, d2) -> (d1)>(%arg9, %arg10, %arg11) {
          affine.parallel (%arg15) = (0) to (8) {
            %7 = affine.apply affine_map<(d0, d1) -> (d0 * 8 + d1)>(%arg12, %arg15)
            affine.parallel (%arg16) = (0) to (8) {
              %8 = affine.apply affine_map<(d0, d1) -> (d0 * 8 + d1)>(%arg13, %arg16)
              %cst = arith.constant 0.000000e+00 : f32
              %9 = tensor.empty() : tensor<64x64xf32>
              %10 = linalg.fill ins(%cst : f32) outs(%9 : tensor<64x64xf32>) -> tensor<64x64xf32>
              %c64 = arith.constant 64 : index
              %11 = arith.muli %8, %c64 : index
              %12 = affine.apply affine_map<()[s0] -> ((s0 + 31) floordiv 32)>()[%arg5]
              %c0 = arith.constant 0 : index
              %c1 = arith.constant 1 : index
              %13 = scf.for %arg17 = %c0 to %12 step %c1 iter_args(%arg18 = %10) -> (tensor<64x64xf32>) {
                %c32 = arith.constant 32 : index
                %20 = arith.muli %arg17, %c32 : index
                %21 = arith.muli %11, %arg6 : index
                %22 = affine.apply affine_map<(d0)[s0] -> (d0 * 32 + s0)>(%arg17)[%21]
                %reinterpret_cast_0 = memref.reinterpret_cast %arg0 to offset: [%22], sizes: [64, 32], strides: [%arg6, 1] {tmd.reuse = {sequential = [{depth = 5 : i64, iterator = "%arg17", reuse_type = "no_reuse", volume = 0 : i64}], spatial = [{depth = 3 : i64, iterator = "%arg15", mapped_to = "y", reuse_type = "total_reuse", volume = 8192 : i64}, {depth = 4 : i64, iterator = "%arg16", mapped_to = "x", reuse_type = "no_reuse", volume = 0 : i64}], temporal = [{depth = 0 : i64, iterator = "%arg12", reuse_type = "total_reuse", volume = 8192 : i64}, {depth = 1 : i64, iterator = "%arg13", reuse_type = "no_reuse", volume = 0 : i64}, {depth = 2 : i64, iterator = "%arg14", reuse_type = "total_reuse", volume = 8192 : i64}]}} : memref<*xf32> to memref<64x32xf32, strided<[?, 1], offset: ?>>
                %alloc = memref.alloc() : memref<64x32xf32>
                memref.copy %reinterpret_cast_0, %alloc : memref<64x32xf32, strided<[?, 1], offset: ?>> to memref<64x32xf32>
                %23 = bufferization.to_tensor %alloc restrict writable : memref<64x32xf32> to tensor<64x32xf32>
                %24 = arith.muli %20, %arg7 : index
                %25 = affine.apply affine_map<(d0)[s0] -> (d0 * 64 + s0)>(%arg14)[%24]
                %reinterpret_cast_1 = memref.reinterpret_cast %arg1 to offset: [%25], sizes: [32, 64], strides: [%arg7, 1] {tmd.reuse = {sequential = [{depth = 5 : i64, iterator = "%arg17", reuse_type = "no_reuse", volume = 0 : i64}], spatial = [{depth = 3 : i64, iterator = "%arg15", mapped_to = "y", reuse_type = "total_reuse", volume = 8192 : i64}, {depth = 4 : i64, iterator = "%arg16", mapped_to = "x", reuse_type = "total_reuse", volume = 8192 : i64}], temporal = [{depth = 0 : i64, iterator = "%arg12", reuse_type = "total_reuse", volume = 8192 : i64}, {depth = 1 : i64, iterator = "%arg13", reuse_type = "total_reuse", volume = 8192 : i64}, {depth = 2 : i64, iterator = "%arg14", reuse_type = "no_reuse", volume = 0 : i64}]}} : memref<*xf32> to memref<32x64xf32, strided<[?, 1], offset: ?>>
                %alloc_2 = memref.alloc() : memref<32x64xf32>
                memref.copy %reinterpret_cast_1, %alloc_2 : memref<32x64xf32, strided<[?, 1], offset: ?>> to memref<32x64xf32>
                %26 = bufferization.to_tensor %alloc_2 restrict writable : memref<32x64xf32> to tensor<32x64xf32>
                %27 = linalg.matmul ins(%23, %26 : tensor<64x32xf32>, tensor<32x64xf32>) outs(%10 : tensor<64x64xf32>) -> tensor<64x64xf32>
                %28 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%arg18, %27 : tensor<64x64xf32>, tensor<64x64xf32>) outs(%arg18 : tensor<64x64xf32>) {
                ^bb0(%in: f32, %in_3: f32, %out: f32):
                  %29 = arith.addf %in, %in_3 : f32
                  linalg.yield %29 : f32
                } -> tensor<64x64xf32>
                scf.yield %28 : tensor<64x64xf32>
              }
              %14 = arith.muli %11, %arg8 : index
              %15 = affine.apply affine_map<(d0)[s0] -> (d0 * 64 + s0)>(%arg14)[%14]
              %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%15], sizes: [64, 64], strides: [%arg8, 1] {tmd.reuse = {spatial = [{depth = 3 : i64, iterator = "%arg15", mapped_to = "y", reuse_type = "total_reuse", volume = 16384 : i64}, {depth = 4 : i64, iterator = "%arg16", mapped_to = "x", reuse_type = "no_reuse", volume = 0 : i64}], temporal = [{depth = 0 : i64, iterator = "%arg12", reuse_type = "total_reuse", volume = 16384 : i64}, {depth = 1 : i64, iterator = "%arg13", reuse_type = "no_reuse", volume = 0 : i64}, {depth = 2 : i64, iterator = "%arg14", reuse_type = "no_reuse", volume = 0 : i64}]}} : memref<*xf32> to memref<64x64xf32, strided<[?, 1], offset: ?>>
              %16 = affine.min affine_map<(d0, d1, d2)[s0] -> (d0 * -64 + s0, 64)>(%8, %arg14, %7)[%arg3]
              %17 = affine.min affine_map<(d0, d1, d2)[s0] -> (d1 * -64 + s0, 64)>(%8, %arg14, %7)[%arg4]
              %extracted_slice = tensor.extract_slice %13[0, 0] [%16, %17] [1, 1] : tensor<64x64xf32> to tensor<?x?xf32>
              %18 = affine.min affine_map<(d0, d1, d2)[s0] -> (d0 * -64 + s0, 64)>(%8, %arg14, %7)[%arg3]
              %19 = affine.min affine_map<(d0, d1, d2)[s0] -> (d1 * -64 + s0, 64)>(%8, %arg14, %7)[%arg4]
              %subview = memref.subview %reinterpret_cast[0, 0] [%18, %19] [1, 1] : memref<64x64xf32, strided<[?, 1], offset: ?>> to memref<?x?xf32, strided<[?, 1], offset: ?>>
              bufferization.materialize_in_destination %extracted_slice in writable %subview : (tensor<?x?xf32>, memref<?x?xf32, strided<[?, 1], offset: ?>>) -> ()
            } {tmd.mapped_to = "x"}
          } {tmd.mapped_to = "y"}
        }
      }
    }
    return
  }
  func.func @matmul_kernel__d0i0_d1i2_f2_f1_f0(%arg0: memref<*xf32> {tt.divisibility = 16 : i32}, %arg1: memref<*xf32> {tt.divisibility = 16 : i32}, %arg2: memref<*xf32> {tt.divisibility = 16 : i32}, %arg3: index {tt.divisibility = 16 : i32}, %arg4: index {tt.divisibility = 16 : i32}, %arg5: index {tt.divisibility = 16 : i32}, %arg6: index {tt.divisibility = 16 : i32}, %arg7: index {tt.divisibility = 16 : i32}, %arg8: index {tt.divisibility = 16 : i32}, %arg9: index, %arg10: index, %arg11: index) {
    affine.for %arg12 = 0 to affine_map<(d0, d1, d2) -> (d2 ceildiv 8)>(%arg9, %arg10, %arg11) {
      affine.for %arg13 = 0 to affine_map<(d0, d1, d2) -> (d1)>(%arg9, %arg10, %arg11) {
        affine.for %arg14 = 0 to affine_map<(d0, d1, d2) -> (d0 ceildiv 8)>(%arg9, %arg10, %arg11) {
          affine.parallel (%arg15) = (0) to (8) {
            %7 = affine.apply affine_map<(d0, d1) -> (d0 * 8 + d1)>(%arg12, %arg15)
            affine.parallel (%arg16) = (0) to (8) {
              %8 = affine.apply affine_map<(d0, d1) -> (d0 * 8 + d1)>(%arg14, %arg16)
              %cst = arith.constant 0.000000e+00 : f32
              %9 = tensor.empty() : tensor<64x64xf32>
              %10 = linalg.fill ins(%cst : f32) outs(%9 : tensor<64x64xf32>) -> tensor<64x64xf32>
              %c64 = arith.constant 64 : index
              %11 = arith.muli %8, %c64 : index
              %12 = affine.apply affine_map<()[s0] -> ((s0 + 31) floordiv 32)>()[%arg5]
              %c0 = arith.constant 0 : index
              %c1 = arith.constant 1 : index
              %13 = scf.for %arg17 = %c0 to %12 step %c1 iter_args(%arg18 = %10) -> (tensor<64x64xf32>) {
                %c32 = arith.constant 32 : index
                %20 = arith.muli %arg17, %c32 : index
                %21 = arith.muli %11, %arg6 : index
                %22 = affine.apply affine_map<(d0)[s0] -> (d0 * 32 + s0)>(%arg17)[%21]
                %reinterpret_cast_0 = memref.reinterpret_cast %arg0 to offset: [%22], sizes: [64, 32], strides: [%arg6, 1] {tmd.reuse = {sequential = [{depth = 5 : i64, iterator = "%arg17", reuse_type = "no_reuse", volume = 0 : i64}], spatial = [{depth = 3 : i64, iterator = "%arg15", mapped_to = "y", reuse_type = "total_reuse", volume = 8192 : i64}, {depth = 4 : i64, iterator = "%arg16", mapped_to = "x", reuse_type = "no_reuse", volume = 0 : i64}], temporal = [{depth = 0 : i64, iterator = "%arg12", reuse_type = "total_reuse", volume = 8192 : i64}, {depth = 1 : i64, iterator = "%arg13", reuse_type = "total_reuse", volume = 8192 : i64}, {depth = 2 : i64, iterator = "%arg14", reuse_type = "no_reuse", volume = 0 : i64}]}} : memref<*xf32> to memref<64x32xf32, strided<[?, 1], offset: ?>>
                %alloc = memref.alloc() : memref<64x32xf32>
                memref.copy %reinterpret_cast_0, %alloc : memref<64x32xf32, strided<[?, 1], offset: ?>> to memref<64x32xf32>
                %23 = bufferization.to_tensor %alloc restrict writable : memref<64x32xf32> to tensor<64x32xf32>
                %24 = arith.muli %20, %arg7 : index
                %25 = affine.apply affine_map<(d0)[s0] -> (d0 * 64 + s0)>(%arg13)[%24]
                %reinterpret_cast_1 = memref.reinterpret_cast %arg1 to offset: [%25], sizes: [32, 64], strides: [%arg7, 1] {tmd.reuse = {sequential = [{depth = 5 : i64, iterator = "%arg17", reuse_type = "no_reuse", volume = 0 : i64}], spatial = [{depth = 3 : i64, iterator = "%arg15", mapped_to = "y", reuse_type = "total_reuse", volume = 8192 : i64}, {depth = 4 : i64, iterator = "%arg16", mapped_to = "x", reuse_type = "total_reuse", volume = 8192 : i64}], temporal = [{depth = 0 : i64, iterator = "%arg12", reuse_type = "total_reuse", volume = 8192 : i64}, {depth = 1 : i64, iterator = "%arg13", reuse_type = "no_reuse", volume = 0 : i64}, {depth = 2 : i64, iterator = "%arg14", reuse_type = "total_reuse", volume = 8192 : i64}]}} : memref<*xf32> to memref<32x64xf32, strided<[?, 1], offset: ?>>
                %alloc_2 = memref.alloc() : memref<32x64xf32>
                memref.copy %reinterpret_cast_1, %alloc_2 : memref<32x64xf32, strided<[?, 1], offset: ?>> to memref<32x64xf32>
                %26 = bufferization.to_tensor %alloc_2 restrict writable : memref<32x64xf32> to tensor<32x64xf32>
                %27 = linalg.matmul ins(%23, %26 : tensor<64x32xf32>, tensor<32x64xf32>) outs(%10 : tensor<64x64xf32>) -> tensor<64x64xf32>
                %28 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%arg18, %27 : tensor<64x64xf32>, tensor<64x64xf32>) outs(%arg18 : tensor<64x64xf32>) {
                ^bb0(%in: f32, %in_3: f32, %out: f32):
                  %29 = arith.addf %in, %in_3 : f32
                  linalg.yield %29 : f32
                } -> tensor<64x64xf32>
                scf.yield %28 : tensor<64x64xf32>
              }
              %14 = arith.muli %11, %arg8 : index
              %15 = affine.apply affine_map<(d0)[s0] -> (d0 * 64 + s0)>(%arg13)[%14]
              %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%15], sizes: [64, 64], strides: [%arg8, 1] {tmd.reuse = {spatial = [{depth = 3 : i64, iterator = "%arg15", mapped_to = "y", reuse_type = "total_reuse", volume = 16384 : i64}, {depth = 4 : i64, iterator = "%arg16", mapped_to = "x", reuse_type = "no_reuse", volume = 0 : i64}], temporal = [{depth = 0 : i64, iterator = "%arg12", reuse_type = "total_reuse", volume = 16384 : i64}, {depth = 1 : i64, iterator = "%arg13", reuse_type = "no_reuse", volume = 0 : i64}, {depth = 2 : i64, iterator = "%arg14", reuse_type = "no_reuse", volume = 0 : i64}]}} : memref<*xf32> to memref<64x64xf32, strided<[?, 1], offset: ?>>
              %16 = affine.min affine_map<(d0, d1, d2)[s0] -> (d0 * -64 + s0, 64)>(%8, %arg13, %7)[%arg3]
              %17 = affine.min affine_map<(d0, d1, d2)[s0] -> (d1 * -64 + s0, 64)>(%8, %arg13, %7)[%arg4]
              %extracted_slice = tensor.extract_slice %13[0, 0] [%16, %17] [1, 1] : tensor<64x64xf32> to tensor<?x?xf32>
              %18 = affine.min affine_map<(d0, d1, d2)[s0] -> (d0 * -64 + s0, 64)>(%8, %arg13, %7)[%arg3]
              %19 = affine.min affine_map<(d0, d1, d2)[s0] -> (d1 * -64 + s0, 64)>(%8, %arg13, %7)[%arg4]
              %subview = memref.subview %reinterpret_cast[0, 0] [%18, %19] [1, 1] : memref<64x64xf32, strided<[?, 1], offset: ?>> to memref<?x?xf32, strided<[?, 1], offset: ?>>
              bufferization.materialize_in_destination %extracted_slice in writable %subview : (tensor<?x?xf32>, memref<?x?xf32, strided<[?, 1], offset: ?>>) -> ()
            } {tmd.mapped_to = "x"}
          } {tmd.mapped_to = "y"}
        }
      }
    }
    return
  }
  func.func @matmul_kernel__d1i0_d0i1_f0_f1_f2(%arg0: memref<*xf32> {tt.divisibility = 16 : i32}, %arg1: memref<*xf32> {tt.divisibility = 16 : i32}, %arg2: memref<*xf32> {tt.divisibility = 16 : i32}, %arg3: index {tt.divisibility = 16 : i32}, %arg4: index {tt.divisibility = 16 : i32}, %arg5: index {tt.divisibility = 16 : i32}, %arg6: index {tt.divisibility = 16 : i32}, %arg7: index {tt.divisibility = 16 : i32}, %arg8: index {tt.divisibility = 16 : i32}, %arg9: index, %arg10: index, %arg11: index) {
    affine.for %arg12 = 0 to affine_map<(d0, d1, d2) -> (d0 ceildiv 8)>(%arg9, %arg10, %arg11) {
      affine.for %arg13 = 0 to affine_map<(d0, d1, d2) -> (d1 ceildiv 8)>(%arg9, %arg10, %arg11) {
        affine.for %arg14 = 0 to affine_map<(d0, d1, d2) -> (d2)>(%arg9, %arg10, %arg11) {
          affine.parallel (%arg15) = (0) to (8) {
            %7 = affine.apply affine_map<(d0, d1) -> (d0 * 8 + d1)>(%arg13, %arg15)
            affine.parallel (%arg16) = (0) to (8) {
              %8 = affine.apply affine_map<(d0, d1) -> (d0 * 8 + d1)>(%arg12, %arg16)
              %cst = arith.constant 0.000000e+00 : f32
              %9 = tensor.empty() : tensor<64x64xf32>
              %10 = linalg.fill ins(%cst : f32) outs(%9 : tensor<64x64xf32>) -> tensor<64x64xf32>
              %c64 = arith.constant 64 : index
              %11 = arith.muli %8, %c64 : index
              %12 = affine.apply affine_map<()[s0] -> ((s0 + 31) floordiv 32)>()[%arg5]
              %c0 = arith.constant 0 : index
              %c1 = arith.constant 1 : index
              %13 = scf.for %arg17 = %c0 to %12 step %c1 iter_args(%arg18 = %10) -> (tensor<64x64xf32>) {
                %c32 = arith.constant 32 : index
                %20 = arith.muli %arg17, %c32 : index
                %21 = arith.muli %11, %arg6 : index
                %22 = affine.apply affine_map<(d0)[s0] -> (d0 * 32 + s0)>(%arg17)[%21]
                %reinterpret_cast_0 = memref.reinterpret_cast %arg0 to offset: [%22], sizes: [64, 32], strides: [%arg6, 1] {tmd.reuse = {sequential = [{depth = 5 : i64, iterator = "%arg17", reuse_type = "no_reuse", volume = 0 : i64}], spatial = [{depth = 3 : i64, iterator = "%arg15", mapped_to = "x", reuse_type = "total_reuse", volume = 8192 : i64}, {depth = 4 : i64, iterator = "%arg16", mapped_to = "y", reuse_type = "no_reuse", volume = 0 : i64}], temporal = [{depth = 0 : i64, iterator = "%arg12", reuse_type = "no_reuse", volume = 0 : i64}, {depth = 1 : i64, iterator = "%arg13", reuse_type = "total_reuse", volume = 8192 : i64}, {depth = 2 : i64, iterator = "%arg14", reuse_type = "total_reuse", volume = 8192 : i64}]}} : memref<*xf32> to memref<64x32xf32, strided<[?, 1], offset: ?>>
                %alloc = memref.alloc() : memref<64x32xf32>
                memref.copy %reinterpret_cast_0, %alloc : memref<64x32xf32, strided<[?, 1], offset: ?>> to memref<64x32xf32>
                %23 = bufferization.to_tensor %alloc restrict writable : memref<64x32xf32> to tensor<64x32xf32>
                %24 = arith.muli %20, %arg7 : index
                %25 = affine.apply affine_map<(d0, d1)[s0] -> (d0 * 512 + d1 * 64 + s0)>(%arg13, %arg15)[%24]
                %reinterpret_cast_1 = memref.reinterpret_cast %arg1 to offset: [%25], sizes: [32, 64], strides: [%arg7, 1] {tmd.reuse = {sequential = [{depth = 5 : i64, iterator = "%arg17", reuse_type = "no_reuse", volume = 0 : i64}], spatial = [{depth = 3 : i64, iterator = "%arg15", mapped_to = "x", reuse_type = "no_reuse", volume = 0 : i64}, {depth = 4 : i64, iterator = "%arg16", mapped_to = "y", reuse_type = "total_reuse", volume = 8192 : i64}], temporal = [{depth = 0 : i64, iterator = "%arg12", reuse_type = "total_reuse", volume = 8192 : i64}, {depth = 1 : i64, iterator = "%arg13", reuse_type = "no_reuse", volume = 0 : i64}, {depth = 2 : i64, iterator = "%arg14", reuse_type = "total_reuse", volume = 8192 : i64}]}} : memref<*xf32> to memref<32x64xf32, strided<[?, 1], offset: ?>>
                %alloc_2 = memref.alloc() : memref<32x64xf32>
                memref.copy %reinterpret_cast_1, %alloc_2 : memref<32x64xf32, strided<[?, 1], offset: ?>> to memref<32x64xf32>
                %26 = bufferization.to_tensor %alloc_2 restrict writable : memref<32x64xf32> to tensor<32x64xf32>
                %27 = linalg.matmul ins(%23, %26 : tensor<64x32xf32>, tensor<32x64xf32>) outs(%10 : tensor<64x64xf32>) -> tensor<64x64xf32>
                %28 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%arg18, %27 : tensor<64x64xf32>, tensor<64x64xf32>) outs(%arg18 : tensor<64x64xf32>) {
                ^bb0(%in: f32, %in_3: f32, %out: f32):
                  %29 = arith.addf %in, %in_3 : f32
                  linalg.yield %29 : f32
                } -> tensor<64x64xf32>
                scf.yield %28 : tensor<64x64xf32>
              }
              %14 = arith.muli %11, %arg8 : index
              %15 = affine.apply affine_map<(d0, d1)[s0] -> (d0 * 512 + d1 * 64 + s0)>(%arg13, %arg15)[%14]
              %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%15], sizes: [64, 64], strides: [%arg8, 1] {tmd.reuse = {spatial = [{depth = 3 : i64, iterator = "%arg15", mapped_to = "x", reuse_type = "no_reuse", volume = 0 : i64}, {depth = 4 : i64, iterator = "%arg16", mapped_to = "y", reuse_type = "no_reuse", volume = 0 : i64}], temporal = [{depth = 0 : i64, iterator = "%arg12", reuse_type = "no_reuse", volume = 0 : i64}, {depth = 1 : i64, iterator = "%arg13", reuse_type = "no_reuse", volume = 0 : i64}, {depth = 2 : i64, iterator = "%arg14", reuse_type = "total_reuse", volume = 16384 : i64}]}} : memref<*xf32> to memref<64x64xf32, strided<[?, 1], offset: ?>>
              %16 = affine.min affine_map<(d0, d1, d2)[s0] -> (d0 * -64 + s0, 64)>(%8, %7, %arg14)[%arg3]
              %17 = affine.min affine_map<(d0, d1, d2)[s0] -> (d1 * -64 + s0, 64)>(%8, %7, %arg14)[%arg4]
              %extracted_slice = tensor.extract_slice %13[0, 0] [%16, %17] [1, 1] : tensor<64x64xf32> to tensor<?x?xf32>
              %18 = affine.min affine_map<(d0, d1, d2)[s0] -> (d0 * -64 + s0, 64)>(%8, %7, %arg14)[%arg3]
              %19 = affine.min affine_map<(d0, d1, d2)[s0] -> (d1 * -64 + s0, 64)>(%8, %7, %arg14)[%arg4]
              %subview = memref.subview %reinterpret_cast[0, 0] [%18, %19] [1, 1] : memref<64x64xf32, strided<[?, 1], offset: ?>> to memref<?x?xf32, strided<[?, 1], offset: ?>>
              bufferization.materialize_in_destination %extracted_slice in writable %subview : (tensor<?x?xf32>, memref<?x?xf32, strided<[?, 1], offset: ?>>) -> ()
            } {tmd.mapped_to = "y"}
          } {tmd.mapped_to = "x"}
        }
      }
    }
    return
  }
  func.func @matmul_kernel__d1i0_d0i1_f0_f2_f1(%arg0: memref<*xf32> {tt.divisibility = 16 : i32}, %arg1: memref<*xf32> {tt.divisibility = 16 : i32}, %arg2: memref<*xf32> {tt.divisibility = 16 : i32}, %arg3: index {tt.divisibility = 16 : i32}, %arg4: index {tt.divisibility = 16 : i32}, %arg5: index {tt.divisibility = 16 : i32}, %arg6: index {tt.divisibility = 16 : i32}, %arg7: index {tt.divisibility = 16 : i32}, %arg8: index {tt.divisibility = 16 : i32}, %arg9: index, %arg10: index, %arg11: index) {
    affine.for %arg12 = 0 to affine_map<(d0, d1, d2) -> (d0 ceildiv 8)>(%arg9, %arg10, %arg11) {
      affine.for %arg13 = 0 to affine_map<(d0, d1, d2) -> (d2)>(%arg9, %arg10, %arg11) {
        affine.for %arg14 = 0 to affine_map<(d0, d1, d2) -> (d1 ceildiv 8)>(%arg9, %arg10, %arg11) {
          affine.parallel (%arg15) = (0) to (8) {
            %7 = affine.apply affine_map<(d0, d1) -> (d0 * 8 + d1)>(%arg14, %arg15)
            affine.parallel (%arg16) = (0) to (8) {
              %8 = affine.apply affine_map<(d0, d1) -> (d0 * 8 + d1)>(%arg12, %arg16)
              %cst = arith.constant 0.000000e+00 : f32
              %9 = tensor.empty() : tensor<64x64xf32>
              %10 = linalg.fill ins(%cst : f32) outs(%9 : tensor<64x64xf32>) -> tensor<64x64xf32>
              %c64 = arith.constant 64 : index
              %11 = arith.muli %8, %c64 : index
              %12 = affine.apply affine_map<()[s0] -> ((s0 + 31) floordiv 32)>()[%arg5]
              %c0 = arith.constant 0 : index
              %c1 = arith.constant 1 : index
              %13 = scf.for %arg17 = %c0 to %12 step %c1 iter_args(%arg18 = %10) -> (tensor<64x64xf32>) {
                %c32 = arith.constant 32 : index
                %20 = arith.muli %arg17, %c32 : index
                %21 = arith.muli %11, %arg6 : index
                %22 = affine.apply affine_map<(d0)[s0] -> (d0 * 32 + s0)>(%arg17)[%21]
                %reinterpret_cast_0 = memref.reinterpret_cast %arg0 to offset: [%22], sizes: [64, 32], strides: [%arg6, 1] {tmd.reuse = {sequential = [{depth = 5 : i64, iterator = "%arg17", reuse_type = "no_reuse", volume = 0 : i64}], spatial = [{depth = 3 : i64, iterator = "%arg15", mapped_to = "x", reuse_type = "total_reuse", volume = 8192 : i64}, {depth = 4 : i64, iterator = "%arg16", mapped_to = "y", reuse_type = "no_reuse", volume = 0 : i64}], temporal = [{depth = 0 : i64, iterator = "%arg12", reuse_type = "no_reuse", volume = 0 : i64}, {depth = 1 : i64, iterator = "%arg13", reuse_type = "total_reuse", volume = 8192 : i64}, {depth = 2 : i64, iterator = "%arg14", reuse_type = "total_reuse", volume = 8192 : i64}]}} : memref<*xf32> to memref<64x32xf32, strided<[?, 1], offset: ?>>
                %alloc = memref.alloc() : memref<64x32xf32>
                memref.copy %reinterpret_cast_0, %alloc : memref<64x32xf32, strided<[?, 1], offset: ?>> to memref<64x32xf32>
                %23 = bufferization.to_tensor %alloc restrict writable : memref<64x32xf32> to tensor<64x32xf32>
                %24 = arith.muli %20, %arg7 : index
                %25 = affine.apply affine_map<(d0, d1)[s0] -> (d0 * 512 + d1 * 64 + s0)>(%arg14, %arg15)[%24]
                %reinterpret_cast_1 = memref.reinterpret_cast %arg1 to offset: [%25], sizes: [32, 64], strides: [%arg7, 1] {tmd.reuse = {sequential = [{depth = 5 : i64, iterator = "%arg17", reuse_type = "no_reuse", volume = 0 : i64}], spatial = [{depth = 3 : i64, iterator = "%arg15", mapped_to = "x", reuse_type = "no_reuse", volume = 0 : i64}, {depth = 4 : i64, iterator = "%arg16", mapped_to = "y", reuse_type = "total_reuse", volume = 8192 : i64}], temporal = [{depth = 0 : i64, iterator = "%arg12", reuse_type = "total_reuse", volume = 8192 : i64}, {depth = 1 : i64, iterator = "%arg13", reuse_type = "total_reuse", volume = 8192 : i64}, {depth = 2 : i64, iterator = "%arg14", reuse_type = "no_reuse", volume = 0 : i64}]}} : memref<*xf32> to memref<32x64xf32, strided<[?, 1], offset: ?>>
                %alloc_2 = memref.alloc() : memref<32x64xf32>
                memref.copy %reinterpret_cast_1, %alloc_2 : memref<32x64xf32, strided<[?, 1], offset: ?>> to memref<32x64xf32>
                %26 = bufferization.to_tensor %alloc_2 restrict writable : memref<32x64xf32> to tensor<32x64xf32>
                %27 = linalg.matmul ins(%23, %26 : tensor<64x32xf32>, tensor<32x64xf32>) outs(%10 : tensor<64x64xf32>) -> tensor<64x64xf32>
                %28 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%arg18, %27 : tensor<64x64xf32>, tensor<64x64xf32>) outs(%arg18 : tensor<64x64xf32>) {
                ^bb0(%in: f32, %in_3: f32, %out: f32):
                  %29 = arith.addf %in, %in_3 : f32
                  linalg.yield %29 : f32
                } -> tensor<64x64xf32>
                scf.yield %28 : tensor<64x64xf32>
              }
              %14 = arith.muli %11, %arg8 : index
              %15 = affine.apply affine_map<(d0, d1)[s0] -> (d0 * 512 + d1 * 64 + s0)>(%arg14, %arg15)[%14]
              %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%15], sizes: [64, 64], strides: [%arg8, 1] {tmd.reuse = {spatial = [{depth = 3 : i64, iterator = "%arg15", mapped_to = "x", reuse_type = "no_reuse", volume = 0 : i64}, {depth = 4 : i64, iterator = "%arg16", mapped_to = "y", reuse_type = "no_reuse", volume = 0 : i64}], temporal = [{depth = 0 : i64, iterator = "%arg12", reuse_type = "no_reuse", volume = 0 : i64}, {depth = 1 : i64, iterator = "%arg13", reuse_type = "total_reuse", volume = 16384 : i64}, {depth = 2 : i64, iterator = "%arg14", reuse_type = "no_reuse", volume = 0 : i64}]}} : memref<*xf32> to memref<64x64xf32, strided<[?, 1], offset: ?>>
              %16 = affine.min affine_map<(d0, d1, d2)[s0] -> (d0 * -64 + s0, 64)>(%8, %7, %arg13)[%arg3]
              %17 = affine.min affine_map<(d0, d1, d2)[s0] -> (d1 * -64 + s0, 64)>(%8, %7, %arg13)[%arg4]
              %extracted_slice = tensor.extract_slice %13[0, 0] [%16, %17] [1, 1] : tensor<64x64xf32> to tensor<?x?xf32>
              %18 = affine.min affine_map<(d0, d1, d2)[s0] -> (d0 * -64 + s0, 64)>(%8, %7, %arg13)[%arg3]
              %19 = affine.min affine_map<(d0, d1, d2)[s0] -> (d1 * -64 + s0, 64)>(%8, %7, %arg13)[%arg4]
              %subview = memref.subview %reinterpret_cast[0, 0] [%18, %19] [1, 1] : memref<64x64xf32, strided<[?, 1], offset: ?>> to memref<?x?xf32, strided<[?, 1], offset: ?>>
              bufferization.materialize_in_destination %extracted_slice in writable %subview : (tensor<?x?xf32>, memref<?x?xf32, strided<[?, 1], offset: ?>>) -> ()
            } {tmd.mapped_to = "y"}
          } {tmd.mapped_to = "x"}
        }
      }
    }
    return
  }
  func.func @matmul_kernel__d1i0_d0i1_f1_f0_f2(%arg0: memref<*xf32> {tt.divisibility = 16 : i32}, %arg1: memref<*xf32> {tt.divisibility = 16 : i32}, %arg2: memref<*xf32> {tt.divisibility = 16 : i32}, %arg3: index {tt.divisibility = 16 : i32}, %arg4: index {tt.divisibility = 16 : i32}, %arg5: index {tt.divisibility = 16 : i32}, %arg6: index {tt.divisibility = 16 : i32}, %arg7: index {tt.divisibility = 16 : i32}, %arg8: index {tt.divisibility = 16 : i32}, %arg9: index, %arg10: index, %arg11: index) {
    affine.for %arg12 = 0 to affine_map<(d0, d1, d2) -> (d1 ceildiv 8)>(%arg9, %arg10, %arg11) {
      affine.for %arg13 = 0 to affine_map<(d0, d1, d2) -> (d0 ceildiv 8)>(%arg9, %arg10, %arg11) {
        affine.for %arg14 = 0 to affine_map<(d0, d1, d2) -> (d2)>(%arg9, %arg10, %arg11) {
          affine.parallel (%arg15) = (0) to (8) {
            %7 = affine.apply affine_map<(d0, d1) -> (d0 * 8 + d1)>(%arg12, %arg15)
            affine.parallel (%arg16) = (0) to (8) {
              %8 = affine.apply affine_map<(d0, d1) -> (d0 * 8 + d1)>(%arg13, %arg16)
              %cst = arith.constant 0.000000e+00 : f32
              %9 = tensor.empty() : tensor<64x64xf32>
              %10 = linalg.fill ins(%cst : f32) outs(%9 : tensor<64x64xf32>) -> tensor<64x64xf32>
              %c64 = arith.constant 64 : index
              %11 = arith.muli %8, %c64 : index
              %12 = affine.apply affine_map<()[s0] -> ((s0 + 31) floordiv 32)>()[%arg5]
              %c0 = arith.constant 0 : index
              %c1 = arith.constant 1 : index
              %13 = scf.for %arg17 = %c0 to %12 step %c1 iter_args(%arg18 = %10) -> (tensor<64x64xf32>) {
                %c32 = arith.constant 32 : index
                %20 = arith.muli %arg17, %c32 : index
                %21 = arith.muli %11, %arg6 : index
                %22 = affine.apply affine_map<(d0)[s0] -> (d0 * 32 + s0)>(%arg17)[%21]
                %reinterpret_cast_0 = memref.reinterpret_cast %arg0 to offset: [%22], sizes: [64, 32], strides: [%arg6, 1] {tmd.reuse = {sequential = [{depth = 5 : i64, iterator = "%arg17", reuse_type = "no_reuse", volume = 0 : i64}], spatial = [{depth = 3 : i64, iterator = "%arg15", mapped_to = "x", reuse_type = "total_reuse", volume = 8192 : i64}, {depth = 4 : i64, iterator = "%arg16", mapped_to = "y", reuse_type = "no_reuse", volume = 0 : i64}], temporal = [{depth = 0 : i64, iterator = "%arg12", reuse_type = "total_reuse", volume = 8192 : i64}, {depth = 1 : i64, iterator = "%arg13", reuse_type = "no_reuse", volume = 0 : i64}, {depth = 2 : i64, iterator = "%arg14", reuse_type = "total_reuse", volume = 8192 : i64}]}} : memref<*xf32> to memref<64x32xf32, strided<[?, 1], offset: ?>>
                %alloc = memref.alloc() : memref<64x32xf32>
                memref.copy %reinterpret_cast_0, %alloc : memref<64x32xf32, strided<[?, 1], offset: ?>> to memref<64x32xf32>
                %23 = bufferization.to_tensor %alloc restrict writable : memref<64x32xf32> to tensor<64x32xf32>
                %24 = arith.muli %20, %arg7 : index
                %25 = affine.apply affine_map<(d0, d1)[s0] -> (d0 * 512 + d1 * 64 + s0)>(%arg12, %arg15)[%24]
                %reinterpret_cast_1 = memref.reinterpret_cast %arg1 to offset: [%25], sizes: [32, 64], strides: [%arg7, 1] {tmd.reuse = {sequential = [{depth = 5 : i64, iterator = "%arg17", reuse_type = "no_reuse", volume = 0 : i64}], spatial = [{depth = 3 : i64, iterator = "%arg15", mapped_to = "x", reuse_type = "no_reuse", volume = 0 : i64}, {depth = 4 : i64, iterator = "%arg16", mapped_to = "y", reuse_type = "total_reuse", volume = 8192 : i64}], temporal = [{depth = 0 : i64, iterator = "%arg12", reuse_type = "no_reuse", volume = 0 : i64}, {depth = 1 : i64, iterator = "%arg13", reuse_type = "total_reuse", volume = 8192 : i64}, {depth = 2 : i64, iterator = "%arg14", reuse_type = "total_reuse", volume = 8192 : i64}]}} : memref<*xf32> to memref<32x64xf32, strided<[?, 1], offset: ?>>
                %alloc_2 = memref.alloc() : memref<32x64xf32>
                memref.copy %reinterpret_cast_1, %alloc_2 : memref<32x64xf32, strided<[?, 1], offset: ?>> to memref<32x64xf32>
                %26 = bufferization.to_tensor %alloc_2 restrict writable : memref<32x64xf32> to tensor<32x64xf32>
                %27 = linalg.matmul ins(%23, %26 : tensor<64x32xf32>, tensor<32x64xf32>) outs(%10 : tensor<64x64xf32>) -> tensor<64x64xf32>
                %28 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%arg18, %27 : tensor<64x64xf32>, tensor<64x64xf32>) outs(%arg18 : tensor<64x64xf32>) {
                ^bb0(%in: f32, %in_3: f32, %out: f32):
                  %29 = arith.addf %in, %in_3 : f32
                  linalg.yield %29 : f32
                } -> tensor<64x64xf32>
                scf.yield %28 : tensor<64x64xf32>
              }
              %14 = arith.muli %11, %arg8 : index
              %15 = affine.apply affine_map<(d0, d1)[s0] -> (d0 * 512 + d1 * 64 + s0)>(%arg12, %arg15)[%14]
              %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%15], sizes: [64, 64], strides: [%arg8, 1] {tmd.reuse = {spatial = [{depth = 3 : i64, iterator = "%arg15", mapped_to = "x", reuse_type = "no_reuse", volume = 0 : i64}, {depth = 4 : i64, iterator = "%arg16", mapped_to = "y", reuse_type = "no_reuse", volume = 0 : i64}], temporal = [{depth = 0 : i64, iterator = "%arg12", reuse_type = "no_reuse", volume = 0 : i64}, {depth = 1 : i64, iterator = "%arg13", reuse_type = "no_reuse", volume = 0 : i64}, {depth = 2 : i64, iterator = "%arg14", reuse_type = "total_reuse", volume = 16384 : i64}]}} : memref<*xf32> to memref<64x64xf32, strided<[?, 1], offset: ?>>
              %16 = affine.min affine_map<(d0, d1, d2)[s0] -> (d0 * -64 + s0, 64)>(%8, %7, %arg14)[%arg3]
              %17 = affine.min affine_map<(d0, d1, d2)[s0] -> (d1 * -64 + s0, 64)>(%8, %7, %arg14)[%arg4]
              %extracted_slice = tensor.extract_slice %13[0, 0] [%16, %17] [1, 1] : tensor<64x64xf32> to tensor<?x?xf32>
              %18 = affine.min affine_map<(d0, d1, d2)[s0] -> (d0 * -64 + s0, 64)>(%8, %7, %arg14)[%arg3]
              %19 = affine.min affine_map<(d0, d1, d2)[s0] -> (d1 * -64 + s0, 64)>(%8, %7, %arg14)[%arg4]
              %subview = memref.subview %reinterpret_cast[0, 0] [%18, %19] [1, 1] : memref<64x64xf32, strided<[?, 1], offset: ?>> to memref<?x?xf32, strided<[?, 1], offset: ?>>
              bufferization.materialize_in_destination %extracted_slice in writable %subview : (tensor<?x?xf32>, memref<?x?xf32, strided<[?, 1], offset: ?>>) -> ()
            } {tmd.mapped_to = "y"}
          } {tmd.mapped_to = "x"}
        }
      }
    }
    return
  }
  func.func @matmul_kernel__d1i0_d0i1_f1_f2_f0(%arg0: memref<*xf32> {tt.divisibility = 16 : i32}, %arg1: memref<*xf32> {tt.divisibility = 16 : i32}, %arg2: memref<*xf32> {tt.divisibility = 16 : i32}, %arg3: index {tt.divisibility = 16 : i32}, %arg4: index {tt.divisibility = 16 : i32}, %arg5: index {tt.divisibility = 16 : i32}, %arg6: index {tt.divisibility = 16 : i32}, %arg7: index {tt.divisibility = 16 : i32}, %arg8: index {tt.divisibility = 16 : i32}, %arg9: index, %arg10: index, %arg11: index) {
    affine.for %arg12 = 0 to affine_map<(d0, d1, d2) -> (d1 ceildiv 8)>(%arg9, %arg10, %arg11) {
      affine.for %arg13 = 0 to affine_map<(d0, d1, d2) -> (d2)>(%arg9, %arg10, %arg11) {
        affine.for %arg14 = 0 to affine_map<(d0, d1, d2) -> (d0 ceildiv 8)>(%arg9, %arg10, %arg11) {
          affine.parallel (%arg15) = (0) to (8) {
            %7 = affine.apply affine_map<(d0, d1) -> (d0 * 8 + d1)>(%arg12, %arg15)
            affine.parallel (%arg16) = (0) to (8) {
              %8 = affine.apply affine_map<(d0, d1) -> (d0 * 8 + d1)>(%arg14, %arg16)
              %cst = arith.constant 0.000000e+00 : f32
              %9 = tensor.empty() : tensor<64x64xf32>
              %10 = linalg.fill ins(%cst : f32) outs(%9 : tensor<64x64xf32>) -> tensor<64x64xf32>
              %c64 = arith.constant 64 : index
              %11 = arith.muli %8, %c64 : index
              %12 = affine.apply affine_map<()[s0] -> ((s0 + 31) floordiv 32)>()[%arg5]
              %c0 = arith.constant 0 : index
              %c1 = arith.constant 1 : index
              %13 = scf.for %arg17 = %c0 to %12 step %c1 iter_args(%arg18 = %10) -> (tensor<64x64xf32>) {
                %c32 = arith.constant 32 : index
                %20 = arith.muli %arg17, %c32 : index
                %21 = arith.muli %11, %arg6 : index
                %22 = affine.apply affine_map<(d0)[s0] -> (d0 * 32 + s0)>(%arg17)[%21]
                %reinterpret_cast_0 = memref.reinterpret_cast %arg0 to offset: [%22], sizes: [64, 32], strides: [%arg6, 1] {tmd.reuse = {sequential = [{depth = 5 : i64, iterator = "%arg17", reuse_type = "no_reuse", volume = 0 : i64}], spatial = [{depth = 3 : i64, iterator = "%arg15", mapped_to = "x", reuse_type = "total_reuse", volume = 8192 : i64}, {depth = 4 : i64, iterator = "%arg16", mapped_to = "y", reuse_type = "no_reuse", volume = 0 : i64}], temporal = [{depth = 0 : i64, iterator = "%arg12", reuse_type = "total_reuse", volume = 8192 : i64}, {depth = 1 : i64, iterator = "%arg13", reuse_type = "total_reuse", volume = 8192 : i64}, {depth = 2 : i64, iterator = "%arg14", reuse_type = "no_reuse", volume = 0 : i64}]}} : memref<*xf32> to memref<64x32xf32, strided<[?, 1], offset: ?>>
                %alloc = memref.alloc() : memref<64x32xf32>
                memref.copy %reinterpret_cast_0, %alloc : memref<64x32xf32, strided<[?, 1], offset: ?>> to memref<64x32xf32>
                %23 = bufferization.to_tensor %alloc restrict writable : memref<64x32xf32> to tensor<64x32xf32>
                %24 = arith.muli %20, %arg7 : index
                %25 = affine.apply affine_map<(d0, d1)[s0] -> (d0 * 512 + d1 * 64 + s0)>(%arg12, %arg15)[%24]
                %reinterpret_cast_1 = memref.reinterpret_cast %arg1 to offset: [%25], sizes: [32, 64], strides: [%arg7, 1] {tmd.reuse = {sequential = [{depth = 5 : i64, iterator = "%arg17", reuse_type = "no_reuse", volume = 0 : i64}], spatial = [{depth = 3 : i64, iterator = "%arg15", mapped_to = "x", reuse_type = "no_reuse", volume = 0 : i64}, {depth = 4 : i64, iterator = "%arg16", mapped_to = "y", reuse_type = "total_reuse", volume = 8192 : i64}], temporal = [{depth = 0 : i64, iterator = "%arg12", reuse_type = "no_reuse", volume = 0 : i64}, {depth = 1 : i64, iterator = "%arg13", reuse_type = "total_reuse", volume = 8192 : i64}, {depth = 2 : i64, iterator = "%arg14", reuse_type = "total_reuse", volume = 8192 : i64}]}} : memref<*xf32> to memref<32x64xf32, strided<[?, 1], offset: ?>>
                %alloc_2 = memref.alloc() : memref<32x64xf32>
                memref.copy %reinterpret_cast_1, %alloc_2 : memref<32x64xf32, strided<[?, 1], offset: ?>> to memref<32x64xf32>
                %26 = bufferization.to_tensor %alloc_2 restrict writable : memref<32x64xf32> to tensor<32x64xf32>
                %27 = linalg.matmul ins(%23, %26 : tensor<64x32xf32>, tensor<32x64xf32>) outs(%10 : tensor<64x64xf32>) -> tensor<64x64xf32>
                %28 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%arg18, %27 : tensor<64x64xf32>, tensor<64x64xf32>) outs(%arg18 : tensor<64x64xf32>) {
                ^bb0(%in: f32, %in_3: f32, %out: f32):
                  %29 = arith.addf %in, %in_3 : f32
                  linalg.yield %29 : f32
                } -> tensor<64x64xf32>
                scf.yield %28 : tensor<64x64xf32>
              }
              %14 = arith.muli %11, %arg8 : index
              %15 = affine.apply affine_map<(d0, d1)[s0] -> (d0 * 512 + d1 * 64 + s0)>(%arg12, %arg15)[%14]
              %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%15], sizes: [64, 64], strides: [%arg8, 1] {tmd.reuse = {spatial = [{depth = 3 : i64, iterator = "%arg15", mapped_to = "x", reuse_type = "no_reuse", volume = 0 : i64}, {depth = 4 : i64, iterator = "%arg16", mapped_to = "y", reuse_type = "no_reuse", volume = 0 : i64}], temporal = [{depth = 0 : i64, iterator = "%arg12", reuse_type = "no_reuse", volume = 0 : i64}, {depth = 1 : i64, iterator = "%arg13", reuse_type = "total_reuse", volume = 16384 : i64}, {depth = 2 : i64, iterator = "%arg14", reuse_type = "no_reuse", volume = 0 : i64}]}} : memref<*xf32> to memref<64x64xf32, strided<[?, 1], offset: ?>>
              %16 = affine.min affine_map<(d0, d1, d2)[s0] -> (d0 * -64 + s0, 64)>(%8, %7, %arg13)[%arg3]
              %17 = affine.min affine_map<(d0, d1, d2)[s0] -> (d1 * -64 + s0, 64)>(%8, %7, %arg13)[%arg4]
              %extracted_slice = tensor.extract_slice %13[0, 0] [%16, %17] [1, 1] : tensor<64x64xf32> to tensor<?x?xf32>
              %18 = affine.min affine_map<(d0, d1, d2)[s0] -> (d0 * -64 + s0, 64)>(%8, %7, %arg13)[%arg3]
              %19 = affine.min affine_map<(d0, d1, d2)[s0] -> (d1 * -64 + s0, 64)>(%8, %7, %arg13)[%arg4]
              %subview = memref.subview %reinterpret_cast[0, 0] [%18, %19] [1, 1] : memref<64x64xf32, strided<[?, 1], offset: ?>> to memref<?x?xf32, strided<[?, 1], offset: ?>>
              bufferization.materialize_in_destination %extracted_slice in writable %subview : (tensor<?x?xf32>, memref<?x?xf32, strided<[?, 1], offset: ?>>) -> ()
            } {tmd.mapped_to = "y"}
          } {tmd.mapped_to = "x"}
        }
      }
    }
    return
  }
  func.func @matmul_kernel__d1i0_d0i1_f2_f0_f1(%arg0: memref<*xf32> {tt.divisibility = 16 : i32}, %arg1: memref<*xf32> {tt.divisibility = 16 : i32}, %arg2: memref<*xf32> {tt.divisibility = 16 : i32}, %arg3: index {tt.divisibility = 16 : i32}, %arg4: index {tt.divisibility = 16 : i32}, %arg5: index {tt.divisibility = 16 : i32}, %arg6: index {tt.divisibility = 16 : i32}, %arg7: index {tt.divisibility = 16 : i32}, %arg8: index {tt.divisibility = 16 : i32}, %arg9: index, %arg10: index, %arg11: index) {
    affine.for %arg12 = 0 to affine_map<(d0, d1, d2) -> (d2)>(%arg9, %arg10, %arg11) {
      affine.for %arg13 = 0 to affine_map<(d0, d1, d2) -> (d0 ceildiv 8)>(%arg9, %arg10, %arg11) {
        affine.for %arg14 = 0 to affine_map<(d0, d1, d2) -> (d1 ceildiv 8)>(%arg9, %arg10, %arg11) {
          affine.parallel (%arg15) = (0) to (8) {
            %7 = affine.apply affine_map<(d0, d1) -> (d0 * 8 + d1)>(%arg14, %arg15)
            affine.parallel (%arg16) = (0) to (8) {
              %8 = affine.apply affine_map<(d0, d1) -> (d0 * 8 + d1)>(%arg13, %arg16)
              %cst = arith.constant 0.000000e+00 : f32
              %9 = tensor.empty() : tensor<64x64xf32>
              %10 = linalg.fill ins(%cst : f32) outs(%9 : tensor<64x64xf32>) -> tensor<64x64xf32>
              %c64 = arith.constant 64 : index
              %11 = arith.muli %8, %c64 : index
              %12 = affine.apply affine_map<()[s0] -> ((s0 + 31) floordiv 32)>()[%arg5]
              %c0 = arith.constant 0 : index
              %c1 = arith.constant 1 : index
              %13 = scf.for %arg17 = %c0 to %12 step %c1 iter_args(%arg18 = %10) -> (tensor<64x64xf32>) {
                %c32 = arith.constant 32 : index
                %20 = arith.muli %arg17, %c32 : index
                %21 = arith.muli %11, %arg6 : index
                %22 = affine.apply affine_map<(d0)[s0] -> (d0 * 32 + s0)>(%arg17)[%21]
                %reinterpret_cast_0 = memref.reinterpret_cast %arg0 to offset: [%22], sizes: [64, 32], strides: [%arg6, 1] {tmd.reuse = {sequential = [{depth = 5 : i64, iterator = "%arg17", reuse_type = "no_reuse", volume = 0 : i64}], spatial = [{depth = 3 : i64, iterator = "%arg15", mapped_to = "x", reuse_type = "total_reuse", volume = 8192 : i64}, {depth = 4 : i64, iterator = "%arg16", mapped_to = "y", reuse_type = "no_reuse", volume = 0 : i64}], temporal = [{depth = 0 : i64, iterator = "%arg12", reuse_type = "total_reuse", volume = 8192 : i64}, {depth = 1 : i64, iterator = "%arg13", reuse_type = "no_reuse", volume = 0 : i64}, {depth = 2 : i64, iterator = "%arg14", reuse_type = "total_reuse", volume = 8192 : i64}]}} : memref<*xf32> to memref<64x32xf32, strided<[?, 1], offset: ?>>
                %alloc = memref.alloc() : memref<64x32xf32>
                memref.copy %reinterpret_cast_0, %alloc : memref<64x32xf32, strided<[?, 1], offset: ?>> to memref<64x32xf32>
                %23 = bufferization.to_tensor %alloc restrict writable : memref<64x32xf32> to tensor<64x32xf32>
                %24 = arith.muli %20, %arg7 : index
                %25 = affine.apply affine_map<(d0, d1)[s0] -> (d0 * 512 + d1 * 64 + s0)>(%arg14, %arg15)[%24]
                %reinterpret_cast_1 = memref.reinterpret_cast %arg1 to offset: [%25], sizes: [32, 64], strides: [%arg7, 1] {tmd.reuse = {sequential = [{depth = 5 : i64, iterator = "%arg17", reuse_type = "no_reuse", volume = 0 : i64}], spatial = [{depth = 3 : i64, iterator = "%arg15", mapped_to = "x", reuse_type = "no_reuse", volume = 0 : i64}, {depth = 4 : i64, iterator = "%arg16", mapped_to = "y", reuse_type = "total_reuse", volume = 8192 : i64}], temporal = [{depth = 0 : i64, iterator = "%arg12", reuse_type = "total_reuse", volume = 8192 : i64}, {depth = 1 : i64, iterator = "%arg13", reuse_type = "total_reuse", volume = 8192 : i64}, {depth = 2 : i64, iterator = "%arg14", reuse_type = "no_reuse", volume = 0 : i64}]}} : memref<*xf32> to memref<32x64xf32, strided<[?, 1], offset: ?>>
                %alloc_2 = memref.alloc() : memref<32x64xf32>
                memref.copy %reinterpret_cast_1, %alloc_2 : memref<32x64xf32, strided<[?, 1], offset: ?>> to memref<32x64xf32>
                %26 = bufferization.to_tensor %alloc_2 restrict writable : memref<32x64xf32> to tensor<32x64xf32>
                %27 = linalg.matmul ins(%23, %26 : tensor<64x32xf32>, tensor<32x64xf32>) outs(%10 : tensor<64x64xf32>) -> tensor<64x64xf32>
                %28 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%arg18, %27 : tensor<64x64xf32>, tensor<64x64xf32>) outs(%arg18 : tensor<64x64xf32>) {
                ^bb0(%in: f32, %in_3: f32, %out: f32):
                  %29 = arith.addf %in, %in_3 : f32
                  linalg.yield %29 : f32
                } -> tensor<64x64xf32>
                scf.yield %28 : tensor<64x64xf32>
              }
              %14 = arith.muli %11, %arg8 : index
              %15 = affine.apply affine_map<(d0, d1)[s0] -> (d0 * 512 + d1 * 64 + s0)>(%arg14, %arg15)[%14]
              %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%15], sizes: [64, 64], strides: [%arg8, 1] {tmd.reuse = {spatial = [{depth = 3 : i64, iterator = "%arg15", mapped_to = "x", reuse_type = "no_reuse", volume = 0 : i64}, {depth = 4 : i64, iterator = "%arg16", mapped_to = "y", reuse_type = "no_reuse", volume = 0 : i64}], temporal = [{depth = 0 : i64, iterator = "%arg12", reuse_type = "total_reuse", volume = 16384 : i64}, {depth = 1 : i64, iterator = "%arg13", reuse_type = "no_reuse", volume = 0 : i64}, {depth = 2 : i64, iterator = "%arg14", reuse_type = "no_reuse", volume = 0 : i64}]}} : memref<*xf32> to memref<64x64xf32, strided<[?, 1], offset: ?>>
              %16 = affine.min affine_map<(d0, d1, d2)[s0] -> (d0 * -64 + s0, 64)>(%8, %7, %arg12)[%arg3]
              %17 = affine.min affine_map<(d0, d1, d2)[s0] -> (d1 * -64 + s0, 64)>(%8, %7, %arg12)[%arg4]
              %extracted_slice = tensor.extract_slice %13[0, 0] [%16, %17] [1, 1] : tensor<64x64xf32> to tensor<?x?xf32>
              %18 = affine.min affine_map<(d0, d1, d2)[s0] -> (d0 * -64 + s0, 64)>(%8, %7, %arg12)[%arg3]
              %19 = affine.min affine_map<(d0, d1, d2)[s0] -> (d1 * -64 + s0, 64)>(%8, %7, %arg12)[%arg4]
              %subview = memref.subview %reinterpret_cast[0, 0] [%18, %19] [1, 1] : memref<64x64xf32, strided<[?, 1], offset: ?>> to memref<?x?xf32, strided<[?, 1], offset: ?>>
              bufferization.materialize_in_destination %extracted_slice in writable %subview : (tensor<?x?xf32>, memref<?x?xf32, strided<[?, 1], offset: ?>>) -> ()
            } {tmd.mapped_to = "y"}
          } {tmd.mapped_to = "x"}
        }
      }
    }
    return
  }
  func.func @matmul_kernel__d1i0_d0i1_f2_f1_f0(%arg0: memref<*xf32> {tt.divisibility = 16 : i32}, %arg1: memref<*xf32> {tt.divisibility = 16 : i32}, %arg2: memref<*xf32> {tt.divisibility = 16 : i32}, %arg3: index {tt.divisibility = 16 : i32}, %arg4: index {tt.divisibility = 16 : i32}, %arg5: index {tt.divisibility = 16 : i32}, %arg6: index {tt.divisibility = 16 : i32}, %arg7: index {tt.divisibility = 16 : i32}, %arg8: index {tt.divisibility = 16 : i32}, %arg9: index, %arg10: index, %arg11: index) {
    affine.for %arg12 = 0 to affine_map<(d0, d1, d2) -> (d2)>(%arg9, %arg10, %arg11) {
      affine.for %arg13 = 0 to affine_map<(d0, d1, d2) -> (d1 ceildiv 8)>(%arg9, %arg10, %arg11) {
        affine.for %arg14 = 0 to affine_map<(d0, d1, d2) -> (d0 ceildiv 8)>(%arg9, %arg10, %arg11) {
          affine.parallel (%arg15) = (0) to (8) {
            %7 = affine.apply affine_map<(d0, d1) -> (d0 * 8 + d1)>(%arg13, %arg15)
            affine.parallel (%arg16) = (0) to (8) {
              %8 = affine.apply affine_map<(d0, d1) -> (d0 * 8 + d1)>(%arg14, %arg16)
              %cst = arith.constant 0.000000e+00 : f32
              %9 = tensor.empty() : tensor<64x64xf32>
              %10 = linalg.fill ins(%cst : f32) outs(%9 : tensor<64x64xf32>) -> tensor<64x64xf32>
              %c64 = arith.constant 64 : index
              %11 = arith.muli %8, %c64 : index
              %12 = affine.apply affine_map<()[s0] -> ((s0 + 31) floordiv 32)>()[%arg5]
              %c0 = arith.constant 0 : index
              %c1 = arith.constant 1 : index
              %13 = scf.for %arg17 = %c0 to %12 step %c1 iter_args(%arg18 = %10) -> (tensor<64x64xf32>) {
                %c32 = arith.constant 32 : index
                %20 = arith.muli %arg17, %c32 : index
                %21 = arith.muli %11, %arg6 : index
                %22 = affine.apply affine_map<(d0)[s0] -> (d0 * 32 + s0)>(%arg17)[%21]
                %reinterpret_cast_0 = memref.reinterpret_cast %arg0 to offset: [%22], sizes: [64, 32], strides: [%arg6, 1] {tmd.reuse = {sequential = [{depth = 5 : i64, iterator = "%arg17", reuse_type = "no_reuse", volume = 0 : i64}], spatial = [{depth = 3 : i64, iterator = "%arg15", mapped_to = "x", reuse_type = "total_reuse", volume = 8192 : i64}, {depth = 4 : i64, iterator = "%arg16", mapped_to = "y", reuse_type = "no_reuse", volume = 0 : i64}], temporal = [{depth = 0 : i64, iterator = "%arg12", reuse_type = "total_reuse", volume = 8192 : i64}, {depth = 1 : i64, iterator = "%arg13", reuse_type = "total_reuse", volume = 8192 : i64}, {depth = 2 : i64, iterator = "%arg14", reuse_type = "no_reuse", volume = 0 : i64}]}} : memref<*xf32> to memref<64x32xf32, strided<[?, 1], offset: ?>>
                %alloc = memref.alloc() : memref<64x32xf32>
                memref.copy %reinterpret_cast_0, %alloc : memref<64x32xf32, strided<[?, 1], offset: ?>> to memref<64x32xf32>
                %23 = bufferization.to_tensor %alloc restrict writable : memref<64x32xf32> to tensor<64x32xf32>
                %24 = arith.muli %20, %arg7 : index
                %25 = affine.apply affine_map<(d0, d1)[s0] -> (d0 * 512 + d1 * 64 + s0)>(%arg13, %arg15)[%24]
                %reinterpret_cast_1 = memref.reinterpret_cast %arg1 to offset: [%25], sizes: [32, 64], strides: [%arg7, 1] {tmd.reuse = {sequential = [{depth = 5 : i64, iterator = "%arg17", reuse_type = "no_reuse", volume = 0 : i64}], spatial = [{depth = 3 : i64, iterator = "%arg15", mapped_to = "x", reuse_type = "no_reuse", volume = 0 : i64}, {depth = 4 : i64, iterator = "%arg16", mapped_to = "y", reuse_type = "total_reuse", volume = 8192 : i64}], temporal = [{depth = 0 : i64, iterator = "%arg12", reuse_type = "total_reuse", volume = 8192 : i64}, {depth = 1 : i64, iterator = "%arg13", reuse_type = "no_reuse", volume = 0 : i64}, {depth = 2 : i64, iterator = "%arg14", reuse_type = "total_reuse", volume = 8192 : i64}]}} : memref<*xf32> to memref<32x64xf32, strided<[?, 1], offset: ?>>
                %alloc_2 = memref.alloc() : memref<32x64xf32>
                memref.copy %reinterpret_cast_1, %alloc_2 : memref<32x64xf32, strided<[?, 1], offset: ?>> to memref<32x64xf32>
                %26 = bufferization.to_tensor %alloc_2 restrict writable : memref<32x64xf32> to tensor<32x64xf32>
                %27 = linalg.matmul ins(%23, %26 : tensor<64x32xf32>, tensor<32x64xf32>) outs(%10 : tensor<64x64xf32>) -> tensor<64x64xf32>
                %28 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%arg18, %27 : tensor<64x64xf32>, tensor<64x64xf32>) outs(%arg18 : tensor<64x64xf32>) {
                ^bb0(%in: f32, %in_3: f32, %out: f32):
                  %29 = arith.addf %in, %in_3 : f32
                  linalg.yield %29 : f32
                } -> tensor<64x64xf32>
                scf.yield %28 : tensor<64x64xf32>
              }
              %14 = arith.muli %11, %arg8 : index
              %15 = affine.apply affine_map<(d0, d1)[s0] -> (d0 * 512 + d1 * 64 + s0)>(%arg13, %arg15)[%14]
              %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%15], sizes: [64, 64], strides: [%arg8, 1] {tmd.reuse = {spatial = [{depth = 3 : i64, iterator = "%arg15", mapped_to = "x", reuse_type = "no_reuse", volume = 0 : i64}, {depth = 4 : i64, iterator = "%arg16", mapped_to = "y", reuse_type = "no_reuse", volume = 0 : i64}], temporal = [{depth = 0 : i64, iterator = "%arg12", reuse_type = "total_reuse", volume = 16384 : i64}, {depth = 1 : i64, iterator = "%arg13", reuse_type = "no_reuse", volume = 0 : i64}, {depth = 2 : i64, iterator = "%arg14", reuse_type = "no_reuse", volume = 0 : i64}]}} : memref<*xf32> to memref<64x64xf32, strided<[?, 1], offset: ?>>
              %16 = affine.min affine_map<(d0, d1, d2)[s0] -> (d0 * -64 + s0, 64)>(%8, %7, %arg12)[%arg3]
              %17 = affine.min affine_map<(d0, d1, d2)[s0] -> (d1 * -64 + s0, 64)>(%8, %7, %arg12)[%arg4]
              %extracted_slice = tensor.extract_slice %13[0, 0] [%16, %17] [1, 1] : tensor<64x64xf32> to tensor<?x?xf32>
              %18 = affine.min affine_map<(d0, d1, d2)[s0] -> (d0 * -64 + s0, 64)>(%8, %7, %arg12)[%arg3]
              %19 = affine.min affine_map<(d0, d1, d2)[s0] -> (d1 * -64 + s0, 64)>(%8, %7, %arg12)[%arg4]
              %subview = memref.subview %reinterpret_cast[0, 0] [%18, %19] [1, 1] : memref<64x64xf32, strided<[?, 1], offset: ?>> to memref<?x?xf32, strided<[?, 1], offset: ?>>
              bufferization.materialize_in_destination %extracted_slice in writable %subview : (tensor<?x?xf32>, memref<?x?xf32, strided<[?, 1], offset: ?>>) -> ()
            } {tmd.mapped_to = "y"}
          } {tmd.mapped_to = "x"}
        }
      }
    }
    return
  }
  func.func @matmul_kernel__d0i1_d1i1_f0_f1_f2(%arg0: memref<*xf32> {tt.divisibility = 16 : i32}, %arg1: memref<*xf32> {tt.divisibility = 16 : i32}, %arg2: memref<*xf32> {tt.divisibility = 16 : i32}, %arg3: index {tt.divisibility = 16 : i32}, %arg4: index {tt.divisibility = 16 : i32}, %arg5: index {tt.divisibility = 16 : i32}, %arg6: index {tt.divisibility = 16 : i32}, %arg7: index {tt.divisibility = 16 : i32}, %arg8: index {tt.divisibility = 16 : i32}, %arg9: index, %arg10: index, %arg11: index) {
    affine.for %arg12 = 0 to affine_map<(d0, d1, d2) -> (d0)>(%arg9, %arg10, %arg11) {
      affine.for %arg13 = 0 to affine_map<(d0, d1, d2) -> ((d1 ceildiv 8) ceildiv 8)>(%arg9, %arg10, %arg11) {
        affine.for %arg14 = 0 to affine_map<(d0, d1, d2) -> (d2)>(%arg9, %arg10, %arg11) {
          affine.parallel (%arg15) = (0) to (8) {
            affine.parallel (%arg16) = (0) to (8) {
              %7 = affine.apply affine_map<(d0, d1, d2) -> (d0 + d1 * 64 + d2 * 8)>(%arg16, %arg13, %arg15)
              %cst = arith.constant 0.000000e+00 : f32
              %8 = tensor.empty() : tensor<64x64xf32>
              %9 = linalg.fill ins(%cst : f32) outs(%8 : tensor<64x64xf32>) -> tensor<64x64xf32>
              %c64 = arith.constant 64 : index
              %10 = arith.muli %arg12, %c64 : index
              %11 = affine.apply affine_map<()[s0] -> ((s0 + 31) floordiv 32)>()[%arg5]
              %c0 = arith.constant 0 : index
              %c1 = arith.constant 1 : index
              %12 = scf.for %arg17 = %c0 to %11 step %c1 iter_args(%arg18 = %9) -> (tensor<64x64xf32>) {
                %c32 = arith.constant 32 : index
                %19 = arith.muli %arg17, %c32 : index
                %20 = arith.muli %10, %arg6 : index
                %21 = affine.apply affine_map<(d0)[s0] -> (d0 * 32 + s0)>(%arg17)[%20]
                %reinterpret_cast_0 = memref.reinterpret_cast %arg0 to offset: [%21], sizes: [64, 32], strides: [%arg6, 1] {tmd.reuse = {sequential = [{depth = 5 : i64, iterator = "%arg17", reuse_type = "no_reuse", volume = 0 : i64}], spatial = [{depth = 3 : i64, iterator = "%arg15", mapped_to = "y", reuse_type = "total_reuse", volume = 8192 : i64}, {depth = 4 : i64, iterator = "%arg16", mapped_to = "x", reuse_type = "total_reuse", volume = 8192 : i64}], temporal = [{depth = 0 : i64, iterator = "%arg12", reuse_type = "no_reuse", volume = 0 : i64}, {depth = 1 : i64, iterator = "%arg13", reuse_type = "total_reuse", volume = 8192 : i64}, {depth = 2 : i64, iterator = "%arg14", reuse_type = "total_reuse", volume = 8192 : i64}]}} : memref<*xf32> to memref<64x32xf32, strided<[?, 1], offset: ?>>
                %alloc = memref.alloc() : memref<64x32xf32>
                memref.copy %reinterpret_cast_0, %alloc : memref<64x32xf32, strided<[?, 1], offset: ?>> to memref<64x32xf32>
                %22 = bufferization.to_tensor %alloc restrict writable : memref<64x32xf32> to tensor<64x32xf32>
                %23 = arith.muli %19, %arg7 : index
                %24 = affine.apply affine_map<(d0, d1, d2)[s0] -> (d0 * 64 + d1 * 4096 + d2 * 512 + s0)>(%arg16, %arg13, %arg15)[%23]
                %reinterpret_cast_1 = memref.reinterpret_cast %arg1 to offset: [%24], sizes: [32, 64], strides: [%arg7, 1] {tmd.reuse = {sequential = [{depth = 5 : i64, iterator = "%arg17", reuse_type = "no_reuse", volume = 0 : i64}], spatial = [{depth = 3 : i64, iterator = "%arg15", mapped_to = "y", reuse_type = "no_reuse", volume = 0 : i64}, {depth = 4 : i64, iterator = "%arg16", mapped_to = "x", reuse_type = "no_reuse", volume = 0 : i64}], temporal = [{depth = 0 : i64, iterator = "%arg12", reuse_type = "total_reuse", volume = 8192 : i64}, {depth = 1 : i64, iterator = "%arg13", reuse_type = "no_reuse", volume = 0 : i64}, {depth = 2 : i64, iterator = "%arg14", reuse_type = "total_reuse", volume = 8192 : i64}]}} : memref<*xf32> to memref<32x64xf32, strided<[?, 1], offset: ?>>
                %alloc_2 = memref.alloc() : memref<32x64xf32>
                memref.copy %reinterpret_cast_1, %alloc_2 : memref<32x64xf32, strided<[?, 1], offset: ?>> to memref<32x64xf32>
                %25 = bufferization.to_tensor %alloc_2 restrict writable : memref<32x64xf32> to tensor<32x64xf32>
                %26 = linalg.matmul ins(%22, %25 : tensor<64x32xf32>, tensor<32x64xf32>) outs(%9 : tensor<64x64xf32>) -> tensor<64x64xf32>
                %27 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%arg18, %26 : tensor<64x64xf32>, tensor<64x64xf32>) outs(%arg18 : tensor<64x64xf32>) {
                ^bb0(%in: f32, %in_3: f32, %out: f32):
                  %28 = arith.addf %in, %in_3 : f32
                  linalg.yield %28 : f32
                } -> tensor<64x64xf32>
                scf.yield %27 : tensor<64x64xf32>
              }
              %13 = arith.muli %10, %arg8 : index
              %14 = affine.apply affine_map<(d0, d1, d2)[s0] -> (d0 * 64 + d1 * 4096 + d2 * 512 + s0)>(%arg16, %arg13, %arg15)[%13]
              %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%14], sizes: [64, 64], strides: [%arg8, 1] {tmd.reuse = {spatial = [{depth = 3 : i64, iterator = "%arg15", mapped_to = "y", reuse_type = "no_reuse", volume = 0 : i64}, {depth = 4 : i64, iterator = "%arg16", mapped_to = "x", reuse_type = "no_reuse", volume = 0 : i64}], temporal = [{depth = 0 : i64, iterator = "%arg12", reuse_type = "no_reuse", volume = 0 : i64}, {depth = 1 : i64, iterator = "%arg13", reuse_type = "no_reuse", volume = 0 : i64}, {depth = 2 : i64, iterator = "%arg14", reuse_type = "total_reuse", volume = 16384 : i64}]}} : memref<*xf32> to memref<64x64xf32, strided<[?, 1], offset: ?>>
              %15 = affine.min affine_map<(d0, d1, d2)[s0] -> (d0 * -64 + s0, 64)>(%arg12, %7, %arg14)[%arg3]
              %16 = affine.min affine_map<(d0, d1, d2)[s0] -> (d1 * -64 + s0, 64)>(%arg12, %7, %arg14)[%arg4]
              %extracted_slice = tensor.extract_slice %12[0, 0] [%15, %16] [1, 1] : tensor<64x64xf32> to tensor<?x?xf32>
              %17 = affine.min affine_map<(d0, d1, d2)[s0] -> (d0 * -64 + s0, 64)>(%arg12, %7, %arg14)[%arg3]
              %18 = affine.min affine_map<(d0, d1, d2)[s0] -> (d1 * -64 + s0, 64)>(%arg12, %7, %arg14)[%arg4]
              %subview = memref.subview %reinterpret_cast[0, 0] [%17, %18] [1, 1] : memref<64x64xf32, strided<[?, 1], offset: ?>> to memref<?x?xf32, strided<[?, 1], offset: ?>>
              bufferization.materialize_in_destination %extracted_slice in writable %subview : (tensor<?x?xf32>, memref<?x?xf32, strided<[?, 1], offset: ?>>) -> ()
            } {tmd.mapped_to = "x"}
          } {tmd.mapped_to = "y"}
        }
      }
    }
    return
  }
  func.func @matmul_kernel__d0i1_d1i1_f0_f2_f1(%arg0: memref<*xf32> {tt.divisibility = 16 : i32}, %arg1: memref<*xf32> {tt.divisibility = 16 : i32}, %arg2: memref<*xf32> {tt.divisibility = 16 : i32}, %arg3: index {tt.divisibility = 16 : i32}, %arg4: index {tt.divisibility = 16 : i32}, %arg5: index {tt.divisibility = 16 : i32}, %arg6: index {tt.divisibility = 16 : i32}, %arg7: index {tt.divisibility = 16 : i32}, %arg8: index {tt.divisibility = 16 : i32}, %arg9: index, %arg10: index, %arg11: index) {
    affine.for %arg12 = 0 to affine_map<(d0, d1, d2) -> (d0)>(%arg9, %arg10, %arg11) {
      affine.for %arg13 = 0 to affine_map<(d0, d1, d2) -> (d2)>(%arg9, %arg10, %arg11) {
        affine.for %arg14 = 0 to affine_map<(d0, d1, d2) -> ((d1 ceildiv 8) ceildiv 8)>(%arg9, %arg10, %arg11) {
          affine.parallel (%arg15) = (0) to (8) {
            affine.parallel (%arg16) = (0) to (8) {
              %7 = affine.apply affine_map<(d0, d1, d2) -> (d0 + d1 * 64 + d2 * 8)>(%arg16, %arg14, %arg15)
              %cst = arith.constant 0.000000e+00 : f32
              %8 = tensor.empty() : tensor<64x64xf32>
              %9 = linalg.fill ins(%cst : f32) outs(%8 : tensor<64x64xf32>) -> tensor<64x64xf32>
              %c64 = arith.constant 64 : index
              %10 = arith.muli %arg12, %c64 : index
              %11 = affine.apply affine_map<()[s0] -> ((s0 + 31) floordiv 32)>()[%arg5]
              %c0 = arith.constant 0 : index
              %c1 = arith.constant 1 : index
              %12 = scf.for %arg17 = %c0 to %11 step %c1 iter_args(%arg18 = %9) -> (tensor<64x64xf32>) {
                %c32 = arith.constant 32 : index
                %19 = arith.muli %arg17, %c32 : index
                %20 = arith.muli %10, %arg6 : index
                %21 = affine.apply affine_map<(d0)[s0] -> (d0 * 32 + s0)>(%arg17)[%20]
                %reinterpret_cast_0 = memref.reinterpret_cast %arg0 to offset: [%21], sizes: [64, 32], strides: [%arg6, 1] {tmd.reuse = {sequential = [{depth = 5 : i64, iterator = "%arg17", reuse_type = "no_reuse", volume = 0 : i64}], spatial = [{depth = 3 : i64, iterator = "%arg15", mapped_to = "y", reuse_type = "total_reuse", volume = 8192 : i64}, {depth = 4 : i64, iterator = "%arg16", mapped_to = "x", reuse_type = "total_reuse", volume = 8192 : i64}], temporal = [{depth = 0 : i64, iterator = "%arg12", reuse_type = "no_reuse", volume = 0 : i64}, {depth = 1 : i64, iterator = "%arg13", reuse_type = "total_reuse", volume = 8192 : i64}, {depth = 2 : i64, iterator = "%arg14", reuse_type = "total_reuse", volume = 8192 : i64}]}} : memref<*xf32> to memref<64x32xf32, strided<[?, 1], offset: ?>>
                %alloc = memref.alloc() : memref<64x32xf32>
                memref.copy %reinterpret_cast_0, %alloc : memref<64x32xf32, strided<[?, 1], offset: ?>> to memref<64x32xf32>
                %22 = bufferization.to_tensor %alloc restrict writable : memref<64x32xf32> to tensor<64x32xf32>
                %23 = arith.muli %19, %arg7 : index
                %24 = affine.apply affine_map<(d0, d1, d2)[s0] -> (d0 * 64 + d1 * 4096 + d2 * 512 + s0)>(%arg16, %arg14, %arg15)[%23]
                %reinterpret_cast_1 = memref.reinterpret_cast %arg1 to offset: [%24], sizes: [32, 64], strides: [%arg7, 1] {tmd.reuse = {sequential = [{depth = 5 : i64, iterator = "%arg17", reuse_type = "no_reuse", volume = 0 : i64}], spatial = [{depth = 3 : i64, iterator = "%arg15", mapped_to = "y", reuse_type = "no_reuse", volume = 0 : i64}, {depth = 4 : i64, iterator = "%arg16", mapped_to = "x", reuse_type = "no_reuse", volume = 0 : i64}], temporal = [{depth = 0 : i64, iterator = "%arg12", reuse_type = "total_reuse", volume = 8192 : i64}, {depth = 1 : i64, iterator = "%arg13", reuse_type = "total_reuse", volume = 8192 : i64}, {depth = 2 : i64, iterator = "%arg14", reuse_type = "no_reuse", volume = 0 : i64}]}} : memref<*xf32> to memref<32x64xf32, strided<[?, 1], offset: ?>>
                %alloc_2 = memref.alloc() : memref<32x64xf32>
                memref.copy %reinterpret_cast_1, %alloc_2 : memref<32x64xf32, strided<[?, 1], offset: ?>> to memref<32x64xf32>
                %25 = bufferization.to_tensor %alloc_2 restrict writable : memref<32x64xf32> to tensor<32x64xf32>
                %26 = linalg.matmul ins(%22, %25 : tensor<64x32xf32>, tensor<32x64xf32>) outs(%9 : tensor<64x64xf32>) -> tensor<64x64xf32>
                %27 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%arg18, %26 : tensor<64x64xf32>, tensor<64x64xf32>) outs(%arg18 : tensor<64x64xf32>) {
                ^bb0(%in: f32, %in_3: f32, %out: f32):
                  %28 = arith.addf %in, %in_3 : f32
                  linalg.yield %28 : f32
                } -> tensor<64x64xf32>
                scf.yield %27 : tensor<64x64xf32>
              }
              %13 = arith.muli %10, %arg8 : index
              %14 = affine.apply affine_map<(d0, d1, d2)[s0] -> (d0 * 64 + d1 * 4096 + d2 * 512 + s0)>(%arg16, %arg14, %arg15)[%13]
              %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%14], sizes: [64, 64], strides: [%arg8, 1] {tmd.reuse = {spatial = [{depth = 3 : i64, iterator = "%arg15", mapped_to = "y", reuse_type = "no_reuse", volume = 0 : i64}, {depth = 4 : i64, iterator = "%arg16", mapped_to = "x", reuse_type = "no_reuse", volume = 0 : i64}], temporal = [{depth = 0 : i64, iterator = "%arg12", reuse_type = "no_reuse", volume = 0 : i64}, {depth = 1 : i64, iterator = "%arg13", reuse_type = "total_reuse", volume = 16384 : i64}, {depth = 2 : i64, iterator = "%arg14", reuse_type = "no_reuse", volume = 0 : i64}]}} : memref<*xf32> to memref<64x64xf32, strided<[?, 1], offset: ?>>
              %15 = affine.min affine_map<(d0, d1, d2)[s0] -> (d0 * -64 + s0, 64)>(%arg12, %7, %arg13)[%arg3]
              %16 = affine.min affine_map<(d0, d1, d2)[s0] -> (d1 * -64 + s0, 64)>(%arg12, %7, %arg13)[%arg4]
              %extracted_slice = tensor.extract_slice %12[0, 0] [%15, %16] [1, 1] : tensor<64x64xf32> to tensor<?x?xf32>
              %17 = affine.min affine_map<(d0, d1, d2)[s0] -> (d0 * -64 + s0, 64)>(%arg12, %7, %arg13)[%arg3]
              %18 = affine.min affine_map<(d0, d1, d2)[s0] -> (d1 * -64 + s0, 64)>(%arg12, %7, %arg13)[%arg4]
              %subview = memref.subview %reinterpret_cast[0, 0] [%17, %18] [1, 1] : memref<64x64xf32, strided<[?, 1], offset: ?>> to memref<?x?xf32, strided<[?, 1], offset: ?>>
              bufferization.materialize_in_destination %extracted_slice in writable %subview : (tensor<?x?xf32>, memref<?x?xf32, strided<[?, 1], offset: ?>>) -> ()
            } {tmd.mapped_to = "x"}
          } {tmd.mapped_to = "y"}
        }
      }
    }
    return
  }
  func.func @matmul_kernel__d0i1_d1i1_f1_f0_f2(%arg0: memref<*xf32> {tt.divisibility = 16 : i32}, %arg1: memref<*xf32> {tt.divisibility = 16 : i32}, %arg2: memref<*xf32> {tt.divisibility = 16 : i32}, %arg3: index {tt.divisibility = 16 : i32}, %arg4: index {tt.divisibility = 16 : i32}, %arg5: index {tt.divisibility = 16 : i32}, %arg6: index {tt.divisibility = 16 : i32}, %arg7: index {tt.divisibility = 16 : i32}, %arg8: index {tt.divisibility = 16 : i32}, %arg9: index, %arg10: index, %arg11: index) {
    affine.for %arg12 = 0 to affine_map<(d0, d1, d2) -> ((d1 ceildiv 8) ceildiv 8)>(%arg9, %arg10, %arg11) {
      affine.for %arg13 = 0 to affine_map<(d0, d1, d2) -> (d0)>(%arg9, %arg10, %arg11) {
        affine.for %arg14 = 0 to affine_map<(d0, d1, d2) -> (d2)>(%arg9, %arg10, %arg11) {
          affine.parallel (%arg15) = (0) to (8) {
            affine.parallel (%arg16) = (0) to (8) {
              %7 = affine.apply affine_map<(d0, d1, d2) -> (d0 + d1 * 64 + d2 * 8)>(%arg16, %arg12, %arg15)
              %cst = arith.constant 0.000000e+00 : f32
              %8 = tensor.empty() : tensor<64x64xf32>
              %9 = linalg.fill ins(%cst : f32) outs(%8 : tensor<64x64xf32>) -> tensor<64x64xf32>
              %c64 = arith.constant 64 : index
              %10 = arith.muli %arg13, %c64 : index
              %11 = affine.apply affine_map<()[s0] -> ((s0 + 31) floordiv 32)>()[%arg5]
              %c0 = arith.constant 0 : index
              %c1 = arith.constant 1 : index
              %12 = scf.for %arg17 = %c0 to %11 step %c1 iter_args(%arg18 = %9) -> (tensor<64x64xf32>) {
                %c32 = arith.constant 32 : index
                %19 = arith.muli %arg17, %c32 : index
                %20 = arith.muli %10, %arg6 : index
                %21 = affine.apply affine_map<(d0)[s0] -> (d0 * 32 + s0)>(%arg17)[%20]
                %reinterpret_cast_0 = memref.reinterpret_cast %arg0 to offset: [%21], sizes: [64, 32], strides: [%arg6, 1] {tmd.reuse = {sequential = [{depth = 5 : i64, iterator = "%arg17", reuse_type = "no_reuse", volume = 0 : i64}], spatial = [{depth = 3 : i64, iterator = "%arg15", mapped_to = "y", reuse_type = "total_reuse", volume = 8192 : i64}, {depth = 4 : i64, iterator = "%arg16", mapped_to = "x", reuse_type = "total_reuse", volume = 8192 : i64}], temporal = [{depth = 0 : i64, iterator = "%arg12", reuse_type = "total_reuse", volume = 8192 : i64}, {depth = 1 : i64, iterator = "%arg13", reuse_type = "no_reuse", volume = 0 : i64}, {depth = 2 : i64, iterator = "%arg14", reuse_type = "total_reuse", volume = 8192 : i64}]}} : memref<*xf32> to memref<64x32xf32, strided<[?, 1], offset: ?>>
                %alloc = memref.alloc() : memref<64x32xf32>
                memref.copy %reinterpret_cast_0, %alloc : memref<64x32xf32, strided<[?, 1], offset: ?>> to memref<64x32xf32>
                %22 = bufferization.to_tensor %alloc restrict writable : memref<64x32xf32> to tensor<64x32xf32>
                %23 = arith.muli %19, %arg7 : index
                %24 = affine.apply affine_map<(d0, d1, d2)[s0] -> (d0 * 64 + d1 * 4096 + d2 * 512 + s0)>(%arg16, %arg12, %arg15)[%23]
                %reinterpret_cast_1 = memref.reinterpret_cast %arg1 to offset: [%24], sizes: [32, 64], strides: [%arg7, 1] {tmd.reuse = {sequential = [{depth = 5 : i64, iterator = "%arg17", reuse_type = "no_reuse", volume = 0 : i64}], spatial = [{depth = 3 : i64, iterator = "%arg15", mapped_to = "y", reuse_type = "no_reuse", volume = 0 : i64}, {depth = 4 : i64, iterator = "%arg16", mapped_to = "x", reuse_type = "no_reuse", volume = 0 : i64}], temporal = [{depth = 0 : i64, iterator = "%arg12", reuse_type = "no_reuse", volume = 0 : i64}, {depth = 1 : i64, iterator = "%arg13", reuse_type = "total_reuse", volume = 8192 : i64}, {depth = 2 : i64, iterator = "%arg14", reuse_type = "total_reuse", volume = 8192 : i64}]}} : memref<*xf32> to memref<32x64xf32, strided<[?, 1], offset: ?>>
                %alloc_2 = memref.alloc() : memref<32x64xf32>
                memref.copy %reinterpret_cast_1, %alloc_2 : memref<32x64xf32, strided<[?, 1], offset: ?>> to memref<32x64xf32>
                %25 = bufferization.to_tensor %alloc_2 restrict writable : memref<32x64xf32> to tensor<32x64xf32>
                %26 = linalg.matmul ins(%22, %25 : tensor<64x32xf32>, tensor<32x64xf32>) outs(%9 : tensor<64x64xf32>) -> tensor<64x64xf32>
                %27 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%arg18, %26 : tensor<64x64xf32>, tensor<64x64xf32>) outs(%arg18 : tensor<64x64xf32>) {
                ^bb0(%in: f32, %in_3: f32, %out: f32):
                  %28 = arith.addf %in, %in_3 : f32
                  linalg.yield %28 : f32
                } -> tensor<64x64xf32>
                scf.yield %27 : tensor<64x64xf32>
              }
              %13 = arith.muli %10, %arg8 : index
              %14 = affine.apply affine_map<(d0, d1, d2)[s0] -> (d0 * 64 + d1 * 4096 + d2 * 512 + s0)>(%arg16, %arg12, %arg15)[%13]
              %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%14], sizes: [64, 64], strides: [%arg8, 1] {tmd.reuse = {spatial = [{depth = 3 : i64, iterator = "%arg15", mapped_to = "y", reuse_type = "no_reuse", volume = 0 : i64}, {depth = 4 : i64, iterator = "%arg16", mapped_to = "x", reuse_type = "no_reuse", volume = 0 : i64}], temporal = [{depth = 0 : i64, iterator = "%arg12", reuse_type = "no_reuse", volume = 0 : i64}, {depth = 1 : i64, iterator = "%arg13", reuse_type = "no_reuse", volume = 0 : i64}, {depth = 2 : i64, iterator = "%arg14", reuse_type = "total_reuse", volume = 16384 : i64}]}} : memref<*xf32> to memref<64x64xf32, strided<[?, 1], offset: ?>>
              %15 = affine.min affine_map<(d0, d1, d2)[s0] -> (d0 * -64 + s0, 64)>(%arg13, %7, %arg14)[%arg3]
              %16 = affine.min affine_map<(d0, d1, d2)[s0] -> (d1 * -64 + s0, 64)>(%arg13, %7, %arg14)[%arg4]
              %extracted_slice = tensor.extract_slice %12[0, 0] [%15, %16] [1, 1] : tensor<64x64xf32> to tensor<?x?xf32>
              %17 = affine.min affine_map<(d0, d1, d2)[s0] -> (d0 * -64 + s0, 64)>(%arg13, %7, %arg14)[%arg3]
              %18 = affine.min affine_map<(d0, d1, d2)[s0] -> (d1 * -64 + s0, 64)>(%arg13, %7, %arg14)[%arg4]
              %subview = memref.subview %reinterpret_cast[0, 0] [%17, %18] [1, 1] : memref<64x64xf32, strided<[?, 1], offset: ?>> to memref<?x?xf32, strided<[?, 1], offset: ?>>
              bufferization.materialize_in_destination %extracted_slice in writable %subview : (tensor<?x?xf32>, memref<?x?xf32, strided<[?, 1], offset: ?>>) -> ()
            } {tmd.mapped_to = "x"}
          } {tmd.mapped_to = "y"}
        }
      }
    }
    return
  }
  func.func @matmul_kernel__d0i1_d1i1_f1_f2_f0(%arg0: memref<*xf32> {tt.divisibility = 16 : i32}, %arg1: memref<*xf32> {tt.divisibility = 16 : i32}, %arg2: memref<*xf32> {tt.divisibility = 16 : i32}, %arg3: index {tt.divisibility = 16 : i32}, %arg4: index {tt.divisibility = 16 : i32}, %arg5: index {tt.divisibility = 16 : i32}, %arg6: index {tt.divisibility = 16 : i32}, %arg7: index {tt.divisibility = 16 : i32}, %arg8: index {tt.divisibility = 16 : i32}, %arg9: index, %arg10: index, %arg11: index) {
    affine.for %arg12 = 0 to affine_map<(d0, d1, d2) -> ((d1 ceildiv 8) ceildiv 8)>(%arg9, %arg10, %arg11) {
      affine.for %arg13 = 0 to affine_map<(d0, d1, d2) -> (d2)>(%arg9, %arg10, %arg11) {
        affine.for %arg14 = 0 to affine_map<(d0, d1, d2) -> (d0)>(%arg9, %arg10, %arg11) {
          affine.parallel (%arg15) = (0) to (8) {
            affine.parallel (%arg16) = (0) to (8) {
              %7 = affine.apply affine_map<(d0, d1, d2) -> (d0 + d1 * 64 + d2 * 8)>(%arg16, %arg12, %arg15)
              %cst = arith.constant 0.000000e+00 : f32
              %8 = tensor.empty() : tensor<64x64xf32>
              %9 = linalg.fill ins(%cst : f32) outs(%8 : tensor<64x64xf32>) -> tensor<64x64xf32>
              %c64 = arith.constant 64 : index
              %10 = arith.muli %arg14, %c64 : index
              %11 = affine.apply affine_map<()[s0] -> ((s0 + 31) floordiv 32)>()[%arg5]
              %c0 = arith.constant 0 : index
              %c1 = arith.constant 1 : index
              %12 = scf.for %arg17 = %c0 to %11 step %c1 iter_args(%arg18 = %9) -> (tensor<64x64xf32>) {
                %c32 = arith.constant 32 : index
                %19 = arith.muli %arg17, %c32 : index
                %20 = arith.muli %10, %arg6 : index
                %21 = affine.apply affine_map<(d0)[s0] -> (d0 * 32 + s0)>(%arg17)[%20]
                %reinterpret_cast_0 = memref.reinterpret_cast %arg0 to offset: [%21], sizes: [64, 32], strides: [%arg6, 1] {tmd.reuse = {sequential = [{depth = 5 : i64, iterator = "%arg17", reuse_type = "no_reuse", volume = 0 : i64}], spatial = [{depth = 3 : i64, iterator = "%arg15", mapped_to = "y", reuse_type = "total_reuse", volume = 8192 : i64}, {depth = 4 : i64, iterator = "%arg16", mapped_to = "x", reuse_type = "total_reuse", volume = 8192 : i64}], temporal = [{depth = 0 : i64, iterator = "%arg12", reuse_type = "total_reuse", volume = 8192 : i64}, {depth = 1 : i64, iterator = "%arg13", reuse_type = "total_reuse", volume = 8192 : i64}, {depth = 2 : i64, iterator = "%arg14", reuse_type = "no_reuse", volume = 0 : i64}]}} : memref<*xf32> to memref<64x32xf32, strided<[?, 1], offset: ?>>
                %alloc = memref.alloc() : memref<64x32xf32>
                memref.copy %reinterpret_cast_0, %alloc : memref<64x32xf32, strided<[?, 1], offset: ?>> to memref<64x32xf32>
                %22 = bufferization.to_tensor %alloc restrict writable : memref<64x32xf32> to tensor<64x32xf32>
                %23 = arith.muli %19, %arg7 : index
                %24 = affine.apply affine_map<(d0, d1, d2)[s0] -> (d0 * 64 + d1 * 4096 + d2 * 512 + s0)>(%arg16, %arg12, %arg15)[%23]
                %reinterpret_cast_1 = memref.reinterpret_cast %arg1 to offset: [%24], sizes: [32, 64], strides: [%arg7, 1] {tmd.reuse = {sequential = [{depth = 5 : i64, iterator = "%arg17", reuse_type = "no_reuse", volume = 0 : i64}], spatial = [{depth = 3 : i64, iterator = "%arg15", mapped_to = "y", reuse_type = "no_reuse", volume = 0 : i64}, {depth = 4 : i64, iterator = "%arg16", mapped_to = "x", reuse_type = "no_reuse", volume = 0 : i64}], temporal = [{depth = 0 : i64, iterator = "%arg12", reuse_type = "no_reuse", volume = 0 : i64}, {depth = 1 : i64, iterator = "%arg13", reuse_type = "total_reuse", volume = 8192 : i64}, {depth = 2 : i64, iterator = "%arg14", reuse_type = "total_reuse", volume = 8192 : i64}]}} : memref<*xf32> to memref<32x64xf32, strided<[?, 1], offset: ?>>
                %alloc_2 = memref.alloc() : memref<32x64xf32>
                memref.copy %reinterpret_cast_1, %alloc_2 : memref<32x64xf32, strided<[?, 1], offset: ?>> to memref<32x64xf32>
                %25 = bufferization.to_tensor %alloc_2 restrict writable : memref<32x64xf32> to tensor<32x64xf32>
                %26 = linalg.matmul ins(%22, %25 : tensor<64x32xf32>, tensor<32x64xf32>) outs(%9 : tensor<64x64xf32>) -> tensor<64x64xf32>
                %27 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%arg18, %26 : tensor<64x64xf32>, tensor<64x64xf32>) outs(%arg18 : tensor<64x64xf32>) {
                ^bb0(%in: f32, %in_3: f32, %out: f32):
                  %28 = arith.addf %in, %in_3 : f32
                  linalg.yield %28 : f32
                } -> tensor<64x64xf32>
                scf.yield %27 : tensor<64x64xf32>
              }
              %13 = arith.muli %10, %arg8 : index
              %14 = affine.apply affine_map<(d0, d1, d2)[s0] -> (d0 * 64 + d1 * 4096 + d2 * 512 + s0)>(%arg16, %arg12, %arg15)[%13]
              %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%14], sizes: [64, 64], strides: [%arg8, 1] {tmd.reuse = {spatial = [{depth = 3 : i64, iterator = "%arg15", mapped_to = "y", reuse_type = "no_reuse", volume = 0 : i64}, {depth = 4 : i64, iterator = "%arg16", mapped_to = "x", reuse_type = "no_reuse", volume = 0 : i64}], temporal = [{depth = 0 : i64, iterator = "%arg12", reuse_type = "no_reuse", volume = 0 : i64}, {depth = 1 : i64, iterator = "%arg13", reuse_type = "total_reuse", volume = 16384 : i64}, {depth = 2 : i64, iterator = "%arg14", reuse_type = "no_reuse", volume = 0 : i64}]}} : memref<*xf32> to memref<64x64xf32, strided<[?, 1], offset: ?>>
              %15 = affine.min affine_map<(d0, d1, d2)[s0] -> (d0 * -64 + s0, 64)>(%arg14, %7, %arg13)[%arg3]
              %16 = affine.min affine_map<(d0, d1, d2)[s0] -> (d1 * -64 + s0, 64)>(%arg14, %7, %arg13)[%arg4]
              %extracted_slice = tensor.extract_slice %12[0, 0] [%15, %16] [1, 1] : tensor<64x64xf32> to tensor<?x?xf32>
              %17 = affine.min affine_map<(d0, d1, d2)[s0] -> (d0 * -64 + s0, 64)>(%arg14, %7, %arg13)[%arg3]
              %18 = affine.min affine_map<(d0, d1, d2)[s0] -> (d1 * -64 + s0, 64)>(%arg14, %7, %arg13)[%arg4]
              %subview = memref.subview %reinterpret_cast[0, 0] [%17, %18] [1, 1] : memref<64x64xf32, strided<[?, 1], offset: ?>> to memref<?x?xf32, strided<[?, 1], offset: ?>>
              bufferization.materialize_in_destination %extracted_slice in writable %subview : (tensor<?x?xf32>, memref<?x?xf32, strided<[?, 1], offset: ?>>) -> ()
            } {tmd.mapped_to = "x"}
          } {tmd.mapped_to = "y"}
        }
      }
    }
    return
  }
  func.func @matmul_kernel__d0i1_d1i1_f2_f0_f1(%arg0: memref<*xf32> {tt.divisibility = 16 : i32}, %arg1: memref<*xf32> {tt.divisibility = 16 : i32}, %arg2: memref<*xf32> {tt.divisibility = 16 : i32}, %arg3: index {tt.divisibility = 16 : i32}, %arg4: index {tt.divisibility = 16 : i32}, %arg5: index {tt.divisibility = 16 : i32}, %arg6: index {tt.divisibility = 16 : i32}, %arg7: index {tt.divisibility = 16 : i32}, %arg8: index {tt.divisibility = 16 : i32}, %arg9: index, %arg10: index, %arg11: index) {
    affine.for %arg12 = 0 to affine_map<(d0, d1, d2) -> (d2)>(%arg9, %arg10, %arg11) {
      affine.for %arg13 = 0 to affine_map<(d0, d1, d2) -> (d0)>(%arg9, %arg10, %arg11) {
        affine.for %arg14 = 0 to affine_map<(d0, d1, d2) -> ((d1 ceildiv 8) ceildiv 8)>(%arg9, %arg10, %arg11) {
          affine.parallel (%arg15) = (0) to (8) {
            affine.parallel (%arg16) = (0) to (8) {
              %7 = affine.apply affine_map<(d0, d1, d2) -> (d0 + d1 * 64 + d2 * 8)>(%arg16, %arg14, %arg15)
              %cst = arith.constant 0.000000e+00 : f32
              %8 = tensor.empty() : tensor<64x64xf32>
              %9 = linalg.fill ins(%cst : f32) outs(%8 : tensor<64x64xf32>) -> tensor<64x64xf32>
              %c64 = arith.constant 64 : index
              %10 = arith.muli %arg13, %c64 : index
              %11 = affine.apply affine_map<()[s0] -> ((s0 + 31) floordiv 32)>()[%arg5]
              %c0 = arith.constant 0 : index
              %c1 = arith.constant 1 : index
              %12 = scf.for %arg17 = %c0 to %11 step %c1 iter_args(%arg18 = %9) -> (tensor<64x64xf32>) {
                %c32 = arith.constant 32 : index
                %19 = arith.muli %arg17, %c32 : index
                %20 = arith.muli %10, %arg6 : index
                %21 = affine.apply affine_map<(d0)[s0] -> (d0 * 32 + s0)>(%arg17)[%20]
                %reinterpret_cast_0 = memref.reinterpret_cast %arg0 to offset: [%21], sizes: [64, 32], strides: [%arg6, 1] {tmd.reuse = {sequential = [{depth = 5 : i64, iterator = "%arg17", reuse_type = "no_reuse", volume = 0 : i64}], spatial = [{depth = 3 : i64, iterator = "%arg15", mapped_to = "y", reuse_type = "total_reuse", volume = 8192 : i64}, {depth = 4 : i64, iterator = "%arg16", mapped_to = "x", reuse_type = "total_reuse", volume = 8192 : i64}], temporal = [{depth = 0 : i64, iterator = "%arg12", reuse_type = "total_reuse", volume = 8192 : i64}, {depth = 1 : i64, iterator = "%arg13", reuse_type = "no_reuse", volume = 0 : i64}, {depth = 2 : i64, iterator = "%arg14", reuse_type = "total_reuse", volume = 8192 : i64}]}} : memref<*xf32> to memref<64x32xf32, strided<[?, 1], offset: ?>>
                %alloc = memref.alloc() : memref<64x32xf32>
                memref.copy %reinterpret_cast_0, %alloc : memref<64x32xf32, strided<[?, 1], offset: ?>> to memref<64x32xf32>
                %22 = bufferization.to_tensor %alloc restrict writable : memref<64x32xf32> to tensor<64x32xf32>
                %23 = arith.muli %19, %arg7 : index
                %24 = affine.apply affine_map<(d0, d1, d2)[s0] -> (d0 * 64 + d1 * 4096 + d2 * 512 + s0)>(%arg16, %arg14, %arg15)[%23]
                %reinterpret_cast_1 = memref.reinterpret_cast %arg1 to offset: [%24], sizes: [32, 64], strides: [%arg7, 1] {tmd.reuse = {sequential = [{depth = 5 : i64, iterator = "%arg17", reuse_type = "no_reuse", volume = 0 : i64}], spatial = [{depth = 3 : i64, iterator = "%arg15", mapped_to = "y", reuse_type = "no_reuse", volume = 0 : i64}, {depth = 4 : i64, iterator = "%arg16", mapped_to = "x", reuse_type = "no_reuse", volume = 0 : i64}], temporal = [{depth = 0 : i64, iterator = "%arg12", reuse_type = "total_reuse", volume = 8192 : i64}, {depth = 1 : i64, iterator = "%arg13", reuse_type = "total_reuse", volume = 8192 : i64}, {depth = 2 : i64, iterator = "%arg14", reuse_type = "no_reuse", volume = 0 : i64}]}} : memref<*xf32> to memref<32x64xf32, strided<[?, 1], offset: ?>>
                %alloc_2 = memref.alloc() : memref<32x64xf32>
                memref.copy %reinterpret_cast_1, %alloc_2 : memref<32x64xf32, strided<[?, 1], offset: ?>> to memref<32x64xf32>
                %25 = bufferization.to_tensor %alloc_2 restrict writable : memref<32x64xf32> to tensor<32x64xf32>
                %26 = linalg.matmul ins(%22, %25 : tensor<64x32xf32>, tensor<32x64xf32>) outs(%9 : tensor<64x64xf32>) -> tensor<64x64xf32>
                %27 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%arg18, %26 : tensor<64x64xf32>, tensor<64x64xf32>) outs(%arg18 : tensor<64x64xf32>) {
                ^bb0(%in: f32, %in_3: f32, %out: f32):
                  %28 = arith.addf %in, %in_3 : f32
                  linalg.yield %28 : f32
                } -> tensor<64x64xf32>
                scf.yield %27 : tensor<64x64xf32>
              }
              %13 = arith.muli %10, %arg8 : index
              %14 = affine.apply affine_map<(d0, d1, d2)[s0] -> (d0 * 64 + d1 * 4096 + d2 * 512 + s0)>(%arg16, %arg14, %arg15)[%13]
              %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%14], sizes: [64, 64], strides: [%arg8, 1] {tmd.reuse = {spatial = [{depth = 3 : i64, iterator = "%arg15", mapped_to = "y", reuse_type = "no_reuse", volume = 0 : i64}, {depth = 4 : i64, iterator = "%arg16", mapped_to = "x", reuse_type = "no_reuse", volume = 0 : i64}], temporal = [{depth = 0 : i64, iterator = "%arg12", reuse_type = "total_reuse", volume = 16384 : i64}, {depth = 1 : i64, iterator = "%arg13", reuse_type = "no_reuse", volume = 0 : i64}, {depth = 2 : i64, iterator = "%arg14", reuse_type = "no_reuse", volume = 0 : i64}]}} : memref<*xf32> to memref<64x64xf32, strided<[?, 1], offset: ?>>
              %15 = affine.min affine_map<(d0, d1, d2)[s0] -> (d0 * -64 + s0, 64)>(%arg13, %7, %arg12)[%arg3]
              %16 = affine.min affine_map<(d0, d1, d2)[s0] -> (d1 * -64 + s0, 64)>(%arg13, %7, %arg12)[%arg4]
              %extracted_slice = tensor.extract_slice %12[0, 0] [%15, %16] [1, 1] : tensor<64x64xf32> to tensor<?x?xf32>
              %17 = affine.min affine_map<(d0, d1, d2)[s0] -> (d0 * -64 + s0, 64)>(%arg13, %7, %arg12)[%arg3]
              %18 = affine.min affine_map<(d0, d1, d2)[s0] -> (d1 * -64 + s0, 64)>(%arg13, %7, %arg12)[%arg4]
              %subview = memref.subview %reinterpret_cast[0, 0] [%17, %18] [1, 1] : memref<64x64xf32, strided<[?, 1], offset: ?>> to memref<?x?xf32, strided<[?, 1], offset: ?>>
              bufferization.materialize_in_destination %extracted_slice in writable %subview : (tensor<?x?xf32>, memref<?x?xf32, strided<[?, 1], offset: ?>>) -> ()
            } {tmd.mapped_to = "x"}
          } {tmd.mapped_to = "y"}
        }
      }
    }
    return
  }
  func.func @matmul_kernel__d0i1_d1i1_f2_f1_f0(%arg0: memref<*xf32> {tt.divisibility = 16 : i32}, %arg1: memref<*xf32> {tt.divisibility = 16 : i32}, %arg2: memref<*xf32> {tt.divisibility = 16 : i32}, %arg3: index {tt.divisibility = 16 : i32}, %arg4: index {tt.divisibility = 16 : i32}, %arg5: index {tt.divisibility = 16 : i32}, %arg6: index {tt.divisibility = 16 : i32}, %arg7: index {tt.divisibility = 16 : i32}, %arg8: index {tt.divisibility = 16 : i32}, %arg9: index, %arg10: index, %arg11: index) {
    affine.for %arg12 = 0 to affine_map<(d0, d1, d2) -> (d2)>(%arg9, %arg10, %arg11) {
      affine.for %arg13 = 0 to affine_map<(d0, d1, d2) -> ((d1 ceildiv 8) ceildiv 8)>(%arg9, %arg10, %arg11) {
        affine.for %arg14 = 0 to affine_map<(d0, d1, d2) -> (d0)>(%arg9, %arg10, %arg11) {
          affine.parallel (%arg15) = (0) to (8) {
            affine.parallel (%arg16) = (0) to (8) {
              %7 = affine.apply affine_map<(d0, d1, d2) -> (d0 + d1 * 64 + d2 * 8)>(%arg16, %arg13, %arg15)
              %cst = arith.constant 0.000000e+00 : f32
              %8 = tensor.empty() : tensor<64x64xf32>
              %9 = linalg.fill ins(%cst : f32) outs(%8 : tensor<64x64xf32>) -> tensor<64x64xf32>
              %c64 = arith.constant 64 : index
              %10 = arith.muli %arg14, %c64 : index
              %11 = affine.apply affine_map<()[s0] -> ((s0 + 31) floordiv 32)>()[%arg5]
              %c0 = arith.constant 0 : index
              %c1 = arith.constant 1 : index
              %12 = scf.for %arg17 = %c0 to %11 step %c1 iter_args(%arg18 = %9) -> (tensor<64x64xf32>) {
                %c32 = arith.constant 32 : index
                %19 = arith.muli %arg17, %c32 : index
                %20 = arith.muli %10, %arg6 : index
                %21 = affine.apply affine_map<(d0)[s0] -> (d0 * 32 + s0)>(%arg17)[%20]
                %reinterpret_cast_0 = memref.reinterpret_cast %arg0 to offset: [%21], sizes: [64, 32], strides: [%arg6, 1] {tmd.reuse = {sequential = [{depth = 5 : i64, iterator = "%arg17", reuse_type = "no_reuse", volume = 0 : i64}], spatial = [{depth = 3 : i64, iterator = "%arg15", mapped_to = "y", reuse_type = "total_reuse", volume = 8192 : i64}, {depth = 4 : i64, iterator = "%arg16", mapped_to = "x", reuse_type = "total_reuse", volume = 8192 : i64}], temporal = [{depth = 0 : i64, iterator = "%arg12", reuse_type = "total_reuse", volume = 8192 : i64}, {depth = 1 : i64, iterator = "%arg13", reuse_type = "total_reuse", volume = 8192 : i64}, {depth = 2 : i64, iterator = "%arg14", reuse_type = "no_reuse", volume = 0 : i64}]}} : memref<*xf32> to memref<64x32xf32, strided<[?, 1], offset: ?>>
                %alloc = memref.alloc() : memref<64x32xf32>
                memref.copy %reinterpret_cast_0, %alloc : memref<64x32xf32, strided<[?, 1], offset: ?>> to memref<64x32xf32>
                %22 = bufferization.to_tensor %alloc restrict writable : memref<64x32xf32> to tensor<64x32xf32>
                %23 = arith.muli %19, %arg7 : index
                %24 = affine.apply affine_map<(d0, d1, d2)[s0] -> (d0 * 64 + d1 * 4096 + d2 * 512 + s0)>(%arg16, %arg13, %arg15)[%23]
                %reinterpret_cast_1 = memref.reinterpret_cast %arg1 to offset: [%24], sizes: [32, 64], strides: [%arg7, 1] {tmd.reuse = {sequential = [{depth = 5 : i64, iterator = "%arg17", reuse_type = "no_reuse", volume = 0 : i64}], spatial = [{depth = 3 : i64, iterator = "%arg15", mapped_to = "y", reuse_type = "no_reuse", volume = 0 : i64}, {depth = 4 : i64, iterator = "%arg16", mapped_to = "x", reuse_type = "no_reuse", volume = 0 : i64}], temporal = [{depth = 0 : i64, iterator = "%arg12", reuse_type = "total_reuse", volume = 8192 : i64}, {depth = 1 : i64, iterator = "%arg13", reuse_type = "no_reuse", volume = 0 : i64}, {depth = 2 : i64, iterator = "%arg14", reuse_type = "total_reuse", volume = 8192 : i64}]}} : memref<*xf32> to memref<32x64xf32, strided<[?, 1], offset: ?>>
                %alloc_2 = memref.alloc() : memref<32x64xf32>
                memref.copy %reinterpret_cast_1, %alloc_2 : memref<32x64xf32, strided<[?, 1], offset: ?>> to memref<32x64xf32>
                %25 = bufferization.to_tensor %alloc_2 restrict writable : memref<32x64xf32> to tensor<32x64xf32>
                %26 = linalg.matmul ins(%22, %25 : tensor<64x32xf32>, tensor<32x64xf32>) outs(%9 : tensor<64x64xf32>) -> tensor<64x64xf32>
                %27 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%arg18, %26 : tensor<64x64xf32>, tensor<64x64xf32>) outs(%arg18 : tensor<64x64xf32>) {
                ^bb0(%in: f32, %in_3: f32, %out: f32):
                  %28 = arith.addf %in, %in_3 : f32
                  linalg.yield %28 : f32
                } -> tensor<64x64xf32>
                scf.yield %27 : tensor<64x64xf32>
              }
              %13 = arith.muli %10, %arg8 : index
              %14 = affine.apply affine_map<(d0, d1, d2)[s0] -> (d0 * 64 + d1 * 4096 + d2 * 512 + s0)>(%arg16, %arg13, %arg15)[%13]
              %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%14], sizes: [64, 64], strides: [%arg8, 1] {tmd.reuse = {spatial = [{depth = 3 : i64, iterator = "%arg15", mapped_to = "y", reuse_type = "no_reuse", volume = 0 : i64}, {depth = 4 : i64, iterator = "%arg16", mapped_to = "x", reuse_type = "no_reuse", volume = 0 : i64}], temporal = [{depth = 0 : i64, iterator = "%arg12", reuse_type = "total_reuse", volume = 16384 : i64}, {depth = 1 : i64, iterator = "%arg13", reuse_type = "no_reuse", volume = 0 : i64}, {depth = 2 : i64, iterator = "%arg14", reuse_type = "no_reuse", volume = 0 : i64}]}} : memref<*xf32> to memref<64x64xf32, strided<[?, 1], offset: ?>>
              %15 = affine.min affine_map<(d0, d1, d2)[s0] -> (d0 * -64 + s0, 64)>(%arg14, %7, %arg12)[%arg3]
              %16 = affine.min affine_map<(d0, d1, d2)[s0] -> (d1 * -64 + s0, 64)>(%arg14, %7, %arg12)[%arg4]
              %extracted_slice = tensor.extract_slice %12[0, 0] [%15, %16] [1, 1] : tensor<64x64xf32> to tensor<?x?xf32>
              %17 = affine.min affine_map<(d0, d1, d2)[s0] -> (d0 * -64 + s0, 64)>(%arg14, %7, %arg12)[%arg3]
              %18 = affine.min affine_map<(d0, d1, d2)[s0] -> (d1 * -64 + s0, 64)>(%arg14, %7, %arg12)[%arg4]
              %subview = memref.subview %reinterpret_cast[0, 0] [%17, %18] [1, 1] : memref<64x64xf32, strided<[?, 1], offset: ?>> to memref<?x?xf32, strided<[?, 1], offset: ?>>
              bufferization.materialize_in_destination %extracted_slice in writable %subview : (tensor<?x?xf32>, memref<?x?xf32, strided<[?, 1], offset: ?>>) -> ()
            } {tmd.mapped_to = "x"}
          } {tmd.mapped_to = "y"}
        }
      }
    }
    return
  }
  func.func @matmul_kernel__d1i1_d0i1_f0_f1_f2(%arg0: memref<*xf32> {tt.divisibility = 16 : i32}, %arg1: memref<*xf32> {tt.divisibility = 16 : i32}, %arg2: memref<*xf32> {tt.divisibility = 16 : i32}, %arg3: index {tt.divisibility = 16 : i32}, %arg4: index {tt.divisibility = 16 : i32}, %arg5: index {tt.divisibility = 16 : i32}, %arg6: index {tt.divisibility = 16 : i32}, %arg7: index {tt.divisibility = 16 : i32}, %arg8: index {tt.divisibility = 16 : i32}, %arg9: index, %arg10: index, %arg11: index) {
    affine.for %arg12 = 0 to affine_map<(d0, d1, d2) -> (d0)>(%arg9, %arg10, %arg11) {
      affine.for %arg13 = 0 to affine_map<(d0, d1, d2) -> ((d1 ceildiv 8) ceildiv 8)>(%arg9, %arg10, %arg11) {
        affine.for %arg14 = 0 to affine_map<(d0, d1, d2) -> (d2)>(%arg9, %arg10, %arg11) {
          affine.parallel (%arg15) = (0) to (8) {
            affine.parallel (%arg16) = (0) to (8) {
              %7 = affine.apply affine_map<(d0, d1, d2) -> (d0 + d1 * 64 + d2 * 8)>(%arg16, %arg13, %arg15)
              %cst = arith.constant 0.000000e+00 : f32
              %8 = tensor.empty() : tensor<64x64xf32>
              %9 = linalg.fill ins(%cst : f32) outs(%8 : tensor<64x64xf32>) -> tensor<64x64xf32>
              %c64 = arith.constant 64 : index
              %10 = arith.muli %arg12, %c64 : index
              %11 = affine.apply affine_map<()[s0] -> ((s0 + 31) floordiv 32)>()[%arg5]
              %c0 = arith.constant 0 : index
              %c1 = arith.constant 1 : index
              %12 = scf.for %arg17 = %c0 to %11 step %c1 iter_args(%arg18 = %9) -> (tensor<64x64xf32>) {
                %c32 = arith.constant 32 : index
                %19 = arith.muli %arg17, %c32 : index
                %20 = arith.muli %10, %arg6 : index
                %21 = affine.apply affine_map<(d0)[s0] -> (d0 * 32 + s0)>(%arg17)[%20]
                %reinterpret_cast_0 = memref.reinterpret_cast %arg0 to offset: [%21], sizes: [64, 32], strides: [%arg6, 1] {tmd.reuse = {sequential = [{depth = 5 : i64, iterator = "%arg17", reuse_type = "no_reuse", volume = 0 : i64}], spatial = [{depth = 3 : i64, iterator = "%arg15", mapped_to = "x", reuse_type = "total_reuse", volume = 8192 : i64}, {depth = 4 : i64, iterator = "%arg16", mapped_to = "y", reuse_type = "total_reuse", volume = 8192 : i64}], temporal = [{depth = 0 : i64, iterator = "%arg12", reuse_type = "no_reuse", volume = 0 : i64}, {depth = 1 : i64, iterator = "%arg13", reuse_type = "total_reuse", volume = 8192 : i64}, {depth = 2 : i64, iterator = "%arg14", reuse_type = "total_reuse", volume = 8192 : i64}]}} : memref<*xf32> to memref<64x32xf32, strided<[?, 1], offset: ?>>
                %alloc = memref.alloc() : memref<64x32xf32>
                memref.copy %reinterpret_cast_0, %alloc : memref<64x32xf32, strided<[?, 1], offset: ?>> to memref<64x32xf32>
                %22 = bufferization.to_tensor %alloc restrict writable : memref<64x32xf32> to tensor<64x32xf32>
                %23 = arith.muli %19, %arg7 : index
                %24 = affine.apply affine_map<(d0, d1, d2)[s0] -> (d0 * 64 + d1 * 4096 + d2 * 512 + s0)>(%arg16, %arg13, %arg15)[%23]
                %reinterpret_cast_1 = memref.reinterpret_cast %arg1 to offset: [%24], sizes: [32, 64], strides: [%arg7, 1] {tmd.reuse = {sequential = [{depth = 5 : i64, iterator = "%arg17", reuse_type = "no_reuse", volume = 0 : i64}], spatial = [{depth = 3 : i64, iterator = "%arg15", mapped_to = "x", reuse_type = "no_reuse", volume = 0 : i64}, {depth = 4 : i64, iterator = "%arg16", mapped_to = "y", reuse_type = "no_reuse", volume = 0 : i64}], temporal = [{depth = 0 : i64, iterator = "%arg12", reuse_type = "total_reuse", volume = 8192 : i64}, {depth = 1 : i64, iterator = "%arg13", reuse_type = "no_reuse", volume = 0 : i64}, {depth = 2 : i64, iterator = "%arg14", reuse_type = "total_reuse", volume = 8192 : i64}]}} : memref<*xf32> to memref<32x64xf32, strided<[?, 1], offset: ?>>
                %alloc_2 = memref.alloc() : memref<32x64xf32>
                memref.copy %reinterpret_cast_1, %alloc_2 : memref<32x64xf32, strided<[?, 1], offset: ?>> to memref<32x64xf32>
                %25 = bufferization.to_tensor %alloc_2 restrict writable : memref<32x64xf32> to tensor<32x64xf32>
                %26 = linalg.matmul ins(%22, %25 : tensor<64x32xf32>, tensor<32x64xf32>) outs(%9 : tensor<64x64xf32>) -> tensor<64x64xf32>
                %27 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%arg18, %26 : tensor<64x64xf32>, tensor<64x64xf32>) outs(%arg18 : tensor<64x64xf32>) {
                ^bb0(%in: f32, %in_3: f32, %out: f32):
                  %28 = arith.addf %in, %in_3 : f32
                  linalg.yield %28 : f32
                } -> tensor<64x64xf32>
                scf.yield %27 : tensor<64x64xf32>
              }
              %13 = arith.muli %10, %arg8 : index
              %14 = affine.apply affine_map<(d0, d1, d2)[s0] -> (d0 * 64 + d1 * 4096 + d2 * 512 + s0)>(%arg16, %arg13, %arg15)[%13]
              %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%14], sizes: [64, 64], strides: [%arg8, 1] {tmd.reuse = {spatial = [{depth = 3 : i64, iterator = "%arg15", mapped_to = "x", reuse_type = "no_reuse", volume = 0 : i64}, {depth = 4 : i64, iterator = "%arg16", mapped_to = "y", reuse_type = "no_reuse", volume = 0 : i64}], temporal = [{depth = 0 : i64, iterator = "%arg12", reuse_type = "no_reuse", volume = 0 : i64}, {depth = 1 : i64, iterator = "%arg13", reuse_type = "no_reuse", volume = 0 : i64}, {depth = 2 : i64, iterator = "%arg14", reuse_type = "total_reuse", volume = 16384 : i64}]}} : memref<*xf32> to memref<64x64xf32, strided<[?, 1], offset: ?>>
              %15 = affine.min affine_map<(d0, d1, d2)[s0] -> (d0 * -64 + s0, 64)>(%arg12, %7, %arg14)[%arg3]
              %16 = affine.min affine_map<(d0, d1, d2)[s0] -> (d1 * -64 + s0, 64)>(%arg12, %7, %arg14)[%arg4]
              %extracted_slice = tensor.extract_slice %12[0, 0] [%15, %16] [1, 1] : tensor<64x64xf32> to tensor<?x?xf32>
              %17 = affine.min affine_map<(d0, d1, d2)[s0] -> (d0 * -64 + s0, 64)>(%arg12, %7, %arg14)[%arg3]
              %18 = affine.min affine_map<(d0, d1, d2)[s0] -> (d1 * -64 + s0, 64)>(%arg12, %7, %arg14)[%arg4]
              %subview = memref.subview %reinterpret_cast[0, 0] [%17, %18] [1, 1] : memref<64x64xf32, strided<[?, 1], offset: ?>> to memref<?x?xf32, strided<[?, 1], offset: ?>>
              bufferization.materialize_in_destination %extracted_slice in writable %subview : (tensor<?x?xf32>, memref<?x?xf32, strided<[?, 1], offset: ?>>) -> ()
            } {tmd.mapped_to = "y"}
          } {tmd.mapped_to = "x"}
        }
      }
    }
    return
  }
  func.func @matmul_kernel__d1i1_d0i1_f0_f2_f1(%arg0: memref<*xf32> {tt.divisibility = 16 : i32}, %arg1: memref<*xf32> {tt.divisibility = 16 : i32}, %arg2: memref<*xf32> {tt.divisibility = 16 : i32}, %arg3: index {tt.divisibility = 16 : i32}, %arg4: index {tt.divisibility = 16 : i32}, %arg5: index {tt.divisibility = 16 : i32}, %arg6: index {tt.divisibility = 16 : i32}, %arg7: index {tt.divisibility = 16 : i32}, %arg8: index {tt.divisibility = 16 : i32}, %arg9: index, %arg10: index, %arg11: index) {
    affine.for %arg12 = 0 to affine_map<(d0, d1, d2) -> (d0)>(%arg9, %arg10, %arg11) {
      affine.for %arg13 = 0 to affine_map<(d0, d1, d2) -> (d2)>(%arg9, %arg10, %arg11) {
        affine.for %arg14 = 0 to affine_map<(d0, d1, d2) -> ((d1 ceildiv 8) ceildiv 8)>(%arg9, %arg10, %arg11) {
          affine.parallel (%arg15) = (0) to (8) {
            affine.parallel (%arg16) = (0) to (8) {
              %7 = affine.apply affine_map<(d0, d1, d2) -> (d0 + d1 * 64 + d2 * 8)>(%arg16, %arg14, %arg15)
              %cst = arith.constant 0.000000e+00 : f32
              %8 = tensor.empty() : tensor<64x64xf32>
              %9 = linalg.fill ins(%cst : f32) outs(%8 : tensor<64x64xf32>) -> tensor<64x64xf32>
              %c64 = arith.constant 64 : index
              %10 = arith.muli %arg12, %c64 : index
              %11 = affine.apply affine_map<()[s0] -> ((s0 + 31) floordiv 32)>()[%arg5]
              %c0 = arith.constant 0 : index
              %c1 = arith.constant 1 : index
              %12 = scf.for %arg17 = %c0 to %11 step %c1 iter_args(%arg18 = %9) -> (tensor<64x64xf32>) {
                %c32 = arith.constant 32 : index
                %19 = arith.muli %arg17, %c32 : index
                %20 = arith.muli %10, %arg6 : index
                %21 = affine.apply affine_map<(d0)[s0] -> (d0 * 32 + s0)>(%arg17)[%20]
                %reinterpret_cast_0 = memref.reinterpret_cast %arg0 to offset: [%21], sizes: [64, 32], strides: [%arg6, 1] {tmd.reuse = {sequential = [{depth = 5 : i64, iterator = "%arg17", reuse_type = "no_reuse", volume = 0 : i64}], spatial = [{depth = 3 : i64, iterator = "%arg15", mapped_to = "x", reuse_type = "total_reuse", volume = 8192 : i64}, {depth = 4 : i64, iterator = "%arg16", mapped_to = "y", reuse_type = "total_reuse", volume = 8192 : i64}], temporal = [{depth = 0 : i64, iterator = "%arg12", reuse_type = "no_reuse", volume = 0 : i64}, {depth = 1 : i64, iterator = "%arg13", reuse_type = "total_reuse", volume = 8192 : i64}, {depth = 2 : i64, iterator = "%arg14", reuse_type = "total_reuse", volume = 8192 : i64}]}} : memref<*xf32> to memref<64x32xf32, strided<[?, 1], offset: ?>>
                %alloc = memref.alloc() : memref<64x32xf32>
                memref.copy %reinterpret_cast_0, %alloc : memref<64x32xf32, strided<[?, 1], offset: ?>> to memref<64x32xf32>
                %22 = bufferization.to_tensor %alloc restrict writable : memref<64x32xf32> to tensor<64x32xf32>
                %23 = arith.muli %19, %arg7 : index
                %24 = affine.apply affine_map<(d0, d1, d2)[s0] -> (d0 * 64 + d1 * 4096 + d2 * 512 + s0)>(%arg16, %arg14, %arg15)[%23]
                %reinterpret_cast_1 = memref.reinterpret_cast %arg1 to offset: [%24], sizes: [32, 64], strides: [%arg7, 1] {tmd.reuse = {sequential = [{depth = 5 : i64, iterator = "%arg17", reuse_type = "no_reuse", volume = 0 : i64}], spatial = [{depth = 3 : i64, iterator = "%arg15", mapped_to = "x", reuse_type = "no_reuse", volume = 0 : i64}, {depth = 4 : i64, iterator = "%arg16", mapped_to = "y", reuse_type = "no_reuse", volume = 0 : i64}], temporal = [{depth = 0 : i64, iterator = "%arg12", reuse_type = "total_reuse", volume = 8192 : i64}, {depth = 1 : i64, iterator = "%arg13", reuse_type = "total_reuse", volume = 8192 : i64}, {depth = 2 : i64, iterator = "%arg14", reuse_type = "no_reuse", volume = 0 : i64}]}} : memref<*xf32> to memref<32x64xf32, strided<[?, 1], offset: ?>>
                %alloc_2 = memref.alloc() : memref<32x64xf32>
                memref.copy %reinterpret_cast_1, %alloc_2 : memref<32x64xf32, strided<[?, 1], offset: ?>> to memref<32x64xf32>
                %25 = bufferization.to_tensor %alloc_2 restrict writable : memref<32x64xf32> to tensor<32x64xf32>
                %26 = linalg.matmul ins(%22, %25 : tensor<64x32xf32>, tensor<32x64xf32>) outs(%9 : tensor<64x64xf32>) -> tensor<64x64xf32>
                %27 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%arg18, %26 : tensor<64x64xf32>, tensor<64x64xf32>) outs(%arg18 : tensor<64x64xf32>) {
                ^bb0(%in: f32, %in_3: f32, %out: f32):
                  %28 = arith.addf %in, %in_3 : f32
                  linalg.yield %28 : f32
                } -> tensor<64x64xf32>
                scf.yield %27 : tensor<64x64xf32>
              }
              %13 = arith.muli %10, %arg8 : index
              %14 = affine.apply affine_map<(d0, d1, d2)[s0] -> (d0 * 64 + d1 * 4096 + d2 * 512 + s0)>(%arg16, %arg14, %arg15)[%13]
              %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%14], sizes: [64, 64], strides: [%arg8, 1] {tmd.reuse = {spatial = [{depth = 3 : i64, iterator = "%arg15", mapped_to = "x", reuse_type = "no_reuse", volume = 0 : i64}, {depth = 4 : i64, iterator = "%arg16", mapped_to = "y", reuse_type = "no_reuse", volume = 0 : i64}], temporal = [{depth = 0 : i64, iterator = "%arg12", reuse_type = "no_reuse", volume = 0 : i64}, {depth = 1 : i64, iterator = "%arg13", reuse_type = "total_reuse", volume = 16384 : i64}, {depth = 2 : i64, iterator = "%arg14", reuse_type = "no_reuse", volume = 0 : i64}]}} : memref<*xf32> to memref<64x64xf32, strided<[?, 1], offset: ?>>
              %15 = affine.min affine_map<(d0, d1, d2)[s0] -> (d0 * -64 + s0, 64)>(%arg12, %7, %arg13)[%arg3]
              %16 = affine.min affine_map<(d0, d1, d2)[s0] -> (d1 * -64 + s0, 64)>(%arg12, %7, %arg13)[%arg4]
              %extracted_slice = tensor.extract_slice %12[0, 0] [%15, %16] [1, 1] : tensor<64x64xf32> to tensor<?x?xf32>
              %17 = affine.min affine_map<(d0, d1, d2)[s0] -> (d0 * -64 + s0, 64)>(%arg12, %7, %arg13)[%arg3]
              %18 = affine.min affine_map<(d0, d1, d2)[s0] -> (d1 * -64 + s0, 64)>(%arg12, %7, %arg13)[%arg4]
              %subview = memref.subview %reinterpret_cast[0, 0] [%17, %18] [1, 1] : memref<64x64xf32, strided<[?, 1], offset: ?>> to memref<?x?xf32, strided<[?, 1], offset: ?>>
              bufferization.materialize_in_destination %extracted_slice in writable %subview : (tensor<?x?xf32>, memref<?x?xf32, strided<[?, 1], offset: ?>>) -> ()
            } {tmd.mapped_to = "y"}
          } {tmd.mapped_to = "x"}
        }
      }
    }
    return
  }
  func.func @matmul_kernel__d1i1_d0i1_f1_f0_f2(%arg0: memref<*xf32> {tt.divisibility = 16 : i32}, %arg1: memref<*xf32> {tt.divisibility = 16 : i32}, %arg2: memref<*xf32> {tt.divisibility = 16 : i32}, %arg3: index {tt.divisibility = 16 : i32}, %arg4: index {tt.divisibility = 16 : i32}, %arg5: index {tt.divisibility = 16 : i32}, %arg6: index {tt.divisibility = 16 : i32}, %arg7: index {tt.divisibility = 16 : i32}, %arg8: index {tt.divisibility = 16 : i32}, %arg9: index, %arg10: index, %arg11: index) {
    affine.for %arg12 = 0 to affine_map<(d0, d1, d2) -> ((d1 ceildiv 8) ceildiv 8)>(%arg9, %arg10, %arg11) {
      affine.for %arg13 = 0 to affine_map<(d0, d1, d2) -> (d0)>(%arg9, %arg10, %arg11) {
        affine.for %arg14 = 0 to affine_map<(d0, d1, d2) -> (d2)>(%arg9, %arg10, %arg11) {
          affine.parallel (%arg15) = (0) to (8) {
            affine.parallel (%arg16) = (0) to (8) {
              %7 = affine.apply affine_map<(d0, d1, d2) -> (d0 + d1 * 64 + d2 * 8)>(%arg16, %arg12, %arg15)
              %cst = arith.constant 0.000000e+00 : f32
              %8 = tensor.empty() : tensor<64x64xf32>
              %9 = linalg.fill ins(%cst : f32) outs(%8 : tensor<64x64xf32>) -> tensor<64x64xf32>
              %c64 = arith.constant 64 : index
              %10 = arith.muli %arg13, %c64 : index
              %11 = affine.apply affine_map<()[s0] -> ((s0 + 31) floordiv 32)>()[%arg5]
              %c0 = arith.constant 0 : index
              %c1 = arith.constant 1 : index
              %12 = scf.for %arg17 = %c0 to %11 step %c1 iter_args(%arg18 = %9) -> (tensor<64x64xf32>) {
                %c32 = arith.constant 32 : index
                %19 = arith.muli %arg17, %c32 : index
                %20 = arith.muli %10, %arg6 : index
                %21 = affine.apply affine_map<(d0)[s0] -> (d0 * 32 + s0)>(%arg17)[%20]
                %reinterpret_cast_0 = memref.reinterpret_cast %arg0 to offset: [%21], sizes: [64, 32], strides: [%arg6, 1] {tmd.reuse = {sequential = [{depth = 5 : i64, iterator = "%arg17", reuse_type = "no_reuse", volume = 0 : i64}], spatial = [{depth = 3 : i64, iterator = "%arg15", mapped_to = "x", reuse_type = "total_reuse", volume = 8192 : i64}, {depth = 4 : i64, iterator = "%arg16", mapped_to = "y", reuse_type = "total_reuse", volume = 8192 : i64}], temporal = [{depth = 0 : i64, iterator = "%arg12", reuse_type = "total_reuse", volume = 8192 : i64}, {depth = 1 : i64, iterator = "%arg13", reuse_type = "no_reuse", volume = 0 : i64}, {depth = 2 : i64, iterator = "%arg14", reuse_type = "total_reuse", volume = 8192 : i64}]}} : memref<*xf32> to memref<64x32xf32, strided<[?, 1], offset: ?>>
                %alloc = memref.alloc() : memref<64x32xf32>
                memref.copy %reinterpret_cast_0, %alloc : memref<64x32xf32, strided<[?, 1], offset: ?>> to memref<64x32xf32>
                %22 = bufferization.to_tensor %alloc restrict writable : memref<64x32xf32> to tensor<64x32xf32>
                %23 = arith.muli %19, %arg7 : index
                %24 = affine.apply affine_map<(d0, d1, d2)[s0] -> (d0 * 64 + d1 * 4096 + d2 * 512 + s0)>(%arg16, %arg12, %arg15)[%23]
                %reinterpret_cast_1 = memref.reinterpret_cast %arg1 to offset: [%24], sizes: [32, 64], strides: [%arg7, 1] {tmd.reuse = {sequential = [{depth = 5 : i64, iterator = "%arg17", reuse_type = "no_reuse", volume = 0 : i64}], spatial = [{depth = 3 : i64, iterator = "%arg15", mapped_to = "x", reuse_type = "no_reuse", volume = 0 : i64}, {depth = 4 : i64, iterator = "%arg16", mapped_to = "y", reuse_type = "no_reuse", volume = 0 : i64}], temporal = [{depth = 0 : i64, iterator = "%arg12", reuse_type = "no_reuse", volume = 0 : i64}, {depth = 1 : i64, iterator = "%arg13", reuse_type = "total_reuse", volume = 8192 : i64}, {depth = 2 : i64, iterator = "%arg14", reuse_type = "total_reuse", volume = 8192 : i64}]}} : memref<*xf32> to memref<32x64xf32, strided<[?, 1], offset: ?>>
                %alloc_2 = memref.alloc() : memref<32x64xf32>
                memref.copy %reinterpret_cast_1, %alloc_2 : memref<32x64xf32, strided<[?, 1], offset: ?>> to memref<32x64xf32>
                %25 = bufferization.to_tensor %alloc_2 restrict writable : memref<32x64xf32> to tensor<32x64xf32>
                %26 = linalg.matmul ins(%22, %25 : tensor<64x32xf32>, tensor<32x64xf32>) outs(%9 : tensor<64x64xf32>) -> tensor<64x64xf32>
                %27 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%arg18, %26 : tensor<64x64xf32>, tensor<64x64xf32>) outs(%arg18 : tensor<64x64xf32>) {
                ^bb0(%in: f32, %in_3: f32, %out: f32):
                  %28 = arith.addf %in, %in_3 : f32
                  linalg.yield %28 : f32
                } -> tensor<64x64xf32>
                scf.yield %27 : tensor<64x64xf32>
              }
              %13 = arith.muli %10, %arg8 : index
              %14 = affine.apply affine_map<(d0, d1, d2)[s0] -> (d0 * 64 + d1 * 4096 + d2 * 512 + s0)>(%arg16, %arg12, %arg15)[%13]
              %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%14], sizes: [64, 64], strides: [%arg8, 1] {tmd.reuse = {spatial = [{depth = 3 : i64, iterator = "%arg15", mapped_to = "x", reuse_type = "no_reuse", volume = 0 : i64}, {depth = 4 : i64, iterator = "%arg16", mapped_to = "y", reuse_type = "no_reuse", volume = 0 : i64}], temporal = [{depth = 0 : i64, iterator = "%arg12", reuse_type = "no_reuse", volume = 0 : i64}, {depth = 1 : i64, iterator = "%arg13", reuse_type = "no_reuse", volume = 0 : i64}, {depth = 2 : i64, iterator = "%arg14", reuse_type = "total_reuse", volume = 16384 : i64}]}} : memref<*xf32> to memref<64x64xf32, strided<[?, 1], offset: ?>>
              %15 = affine.min affine_map<(d0, d1, d2)[s0] -> (d0 * -64 + s0, 64)>(%arg13, %7, %arg14)[%arg3]
              %16 = affine.min affine_map<(d0, d1, d2)[s0] -> (d1 * -64 + s0, 64)>(%arg13, %7, %arg14)[%arg4]
              %extracted_slice = tensor.extract_slice %12[0, 0] [%15, %16] [1, 1] : tensor<64x64xf32> to tensor<?x?xf32>
              %17 = affine.min affine_map<(d0, d1, d2)[s0] -> (d0 * -64 + s0, 64)>(%arg13, %7, %arg14)[%arg3]
              %18 = affine.min affine_map<(d0, d1, d2)[s0] -> (d1 * -64 + s0, 64)>(%arg13, %7, %arg14)[%arg4]
              %subview = memref.subview %reinterpret_cast[0, 0] [%17, %18] [1, 1] : memref<64x64xf32, strided<[?, 1], offset: ?>> to memref<?x?xf32, strided<[?, 1], offset: ?>>
              bufferization.materialize_in_destination %extracted_slice in writable %subview : (tensor<?x?xf32>, memref<?x?xf32, strided<[?, 1], offset: ?>>) -> ()
            } {tmd.mapped_to = "y"}
          } {tmd.mapped_to = "x"}
        }
      }
    }
    return
  }
  func.func @matmul_kernel__d1i1_d0i1_f1_f2_f0(%arg0: memref<*xf32> {tt.divisibility = 16 : i32}, %arg1: memref<*xf32> {tt.divisibility = 16 : i32}, %arg2: memref<*xf32> {tt.divisibility = 16 : i32}, %arg3: index {tt.divisibility = 16 : i32}, %arg4: index {tt.divisibility = 16 : i32}, %arg5: index {tt.divisibility = 16 : i32}, %arg6: index {tt.divisibility = 16 : i32}, %arg7: index {tt.divisibility = 16 : i32}, %arg8: index {tt.divisibility = 16 : i32}, %arg9: index, %arg10: index, %arg11: index) {
    affine.for %arg12 = 0 to affine_map<(d0, d1, d2) -> ((d1 ceildiv 8) ceildiv 8)>(%arg9, %arg10, %arg11) {
      affine.for %arg13 = 0 to affine_map<(d0, d1, d2) -> (d2)>(%arg9, %arg10, %arg11) {
        affine.for %arg14 = 0 to affine_map<(d0, d1, d2) -> (d0)>(%arg9, %arg10, %arg11) {
          affine.parallel (%arg15) = (0) to (8) {
            affine.parallel (%arg16) = (0) to (8) {
              %7 = affine.apply affine_map<(d0, d1, d2) -> (d0 + d1 * 64 + d2 * 8)>(%arg16, %arg12, %arg15)
              %cst = arith.constant 0.000000e+00 : f32
              %8 = tensor.empty() : tensor<64x64xf32>
              %9 = linalg.fill ins(%cst : f32) outs(%8 : tensor<64x64xf32>) -> tensor<64x64xf32>
              %c64 = arith.constant 64 : index
              %10 = arith.muli %arg14, %c64 : index
              %11 = affine.apply affine_map<()[s0] -> ((s0 + 31) floordiv 32)>()[%arg5]
              %c0 = arith.constant 0 : index
              %c1 = arith.constant 1 : index
              %12 = scf.for %arg17 = %c0 to %11 step %c1 iter_args(%arg18 = %9) -> (tensor<64x64xf32>) {
                %c32 = arith.constant 32 : index
                %19 = arith.muli %arg17, %c32 : index
                %20 = arith.muli %10, %arg6 : index
                %21 = affine.apply affine_map<(d0)[s0] -> (d0 * 32 + s0)>(%arg17)[%20]
                %reinterpret_cast_0 = memref.reinterpret_cast %arg0 to offset: [%21], sizes: [64, 32], strides: [%arg6, 1] {tmd.reuse = {sequential = [{depth = 5 : i64, iterator = "%arg17", reuse_type = "no_reuse", volume = 0 : i64}], spatial = [{depth = 3 : i64, iterator = "%arg15", mapped_to = "x", reuse_type = "total_reuse", volume = 8192 : i64}, {depth = 4 : i64, iterator = "%arg16", mapped_to = "y", reuse_type = "total_reuse", volume = 8192 : i64}], temporal = [{depth = 0 : i64, iterator = "%arg12", reuse_type = "total_reuse", volume = 8192 : i64}, {depth = 1 : i64, iterator = "%arg13", reuse_type = "total_reuse", volume = 8192 : i64}, {depth = 2 : i64, iterator = "%arg14", reuse_type = "no_reuse", volume = 0 : i64}]}} : memref<*xf32> to memref<64x32xf32, strided<[?, 1], offset: ?>>
                %alloc = memref.alloc() : memref<64x32xf32>
                memref.copy %reinterpret_cast_0, %alloc : memref<64x32xf32, strided<[?, 1], offset: ?>> to memref<64x32xf32>
                %22 = bufferization.to_tensor %alloc restrict writable : memref<64x32xf32> to tensor<64x32xf32>
                %23 = arith.muli %19, %arg7 : index
                %24 = affine.apply affine_map<(d0, d1, d2)[s0] -> (d0 * 64 + d1 * 4096 + d2 * 512 + s0)>(%arg16, %arg12, %arg15)[%23]
                %reinterpret_cast_1 = memref.reinterpret_cast %arg1 to offset: [%24], sizes: [32, 64], strides: [%arg7, 1] {tmd.reuse = {sequential = [{depth = 5 : i64, iterator = "%arg17", reuse_type = "no_reuse", volume = 0 : i64}], spatial = [{depth = 3 : i64, iterator = "%arg15", mapped_to = "x", reuse_type = "no_reuse", volume = 0 : i64}, {depth = 4 : i64, iterator = "%arg16", mapped_to = "y", reuse_type = "no_reuse", volume = 0 : i64}], temporal = [{depth = 0 : i64, iterator = "%arg12", reuse_type = "no_reuse", volume = 0 : i64}, {depth = 1 : i64, iterator = "%arg13", reuse_type = "total_reuse", volume = 8192 : i64}, {depth = 2 : i64, iterator = "%arg14", reuse_type = "total_reuse", volume = 8192 : i64}]}} : memref<*xf32> to memref<32x64xf32, strided<[?, 1], offset: ?>>
                %alloc_2 = memref.alloc() : memref<32x64xf32>
                memref.copy %reinterpret_cast_1, %alloc_2 : memref<32x64xf32, strided<[?, 1], offset: ?>> to memref<32x64xf32>
                %25 = bufferization.to_tensor %alloc_2 restrict writable : memref<32x64xf32> to tensor<32x64xf32>
                %26 = linalg.matmul ins(%22, %25 : tensor<64x32xf32>, tensor<32x64xf32>) outs(%9 : tensor<64x64xf32>) -> tensor<64x64xf32>
                %27 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%arg18, %26 : tensor<64x64xf32>, tensor<64x64xf32>) outs(%arg18 : tensor<64x64xf32>) {
                ^bb0(%in: f32, %in_3: f32, %out: f32):
                  %28 = arith.addf %in, %in_3 : f32
                  linalg.yield %28 : f32
                } -> tensor<64x64xf32>
                scf.yield %27 : tensor<64x64xf32>
              }
              %13 = arith.muli %10, %arg8 : index
              %14 = affine.apply affine_map<(d0, d1, d2)[s0] -> (d0 * 64 + d1 * 4096 + d2 * 512 + s0)>(%arg16, %arg12, %arg15)[%13]
              %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%14], sizes: [64, 64], strides: [%arg8, 1] {tmd.reuse = {spatial = [{depth = 3 : i64, iterator = "%arg15", mapped_to = "x", reuse_type = "no_reuse", volume = 0 : i64}, {depth = 4 : i64, iterator = "%arg16", mapped_to = "y", reuse_type = "no_reuse", volume = 0 : i64}], temporal = [{depth = 0 : i64, iterator = "%arg12", reuse_type = "no_reuse", volume = 0 : i64}, {depth = 1 : i64, iterator = "%arg13", reuse_type = "total_reuse", volume = 16384 : i64}, {depth = 2 : i64, iterator = "%arg14", reuse_type = "no_reuse", volume = 0 : i64}]}} : memref<*xf32> to memref<64x64xf32, strided<[?, 1], offset: ?>>
              %15 = affine.min affine_map<(d0, d1, d2)[s0] -> (d0 * -64 + s0, 64)>(%arg14, %7, %arg13)[%arg3]
              %16 = affine.min affine_map<(d0, d1, d2)[s0] -> (d1 * -64 + s0, 64)>(%arg14, %7, %arg13)[%arg4]
              %extracted_slice = tensor.extract_slice %12[0, 0] [%15, %16] [1, 1] : tensor<64x64xf32> to tensor<?x?xf32>
              %17 = affine.min affine_map<(d0, d1, d2)[s0] -> (d0 * -64 + s0, 64)>(%arg14, %7, %arg13)[%arg3]
              %18 = affine.min affine_map<(d0, d1, d2)[s0] -> (d1 * -64 + s0, 64)>(%arg14, %7, %arg13)[%arg4]
              %subview = memref.subview %reinterpret_cast[0, 0] [%17, %18] [1, 1] : memref<64x64xf32, strided<[?, 1], offset: ?>> to memref<?x?xf32, strided<[?, 1], offset: ?>>
              bufferization.materialize_in_destination %extracted_slice in writable %subview : (tensor<?x?xf32>, memref<?x?xf32, strided<[?, 1], offset: ?>>) -> ()
            } {tmd.mapped_to = "y"}
          } {tmd.mapped_to = "x"}
        }
      }
    }
    return
  }
  func.func @matmul_kernel__d1i1_d0i1_f2_f0_f1(%arg0: memref<*xf32> {tt.divisibility = 16 : i32}, %arg1: memref<*xf32> {tt.divisibility = 16 : i32}, %arg2: memref<*xf32> {tt.divisibility = 16 : i32}, %arg3: index {tt.divisibility = 16 : i32}, %arg4: index {tt.divisibility = 16 : i32}, %arg5: index {tt.divisibility = 16 : i32}, %arg6: index {tt.divisibility = 16 : i32}, %arg7: index {tt.divisibility = 16 : i32}, %arg8: index {tt.divisibility = 16 : i32}, %arg9: index, %arg10: index, %arg11: index) {
    affine.for %arg12 = 0 to affine_map<(d0, d1, d2) -> (d2)>(%arg9, %arg10, %arg11) {
      affine.for %arg13 = 0 to affine_map<(d0, d1, d2) -> (d0)>(%arg9, %arg10, %arg11) {
        affine.for %arg14 = 0 to affine_map<(d0, d1, d2) -> ((d1 ceildiv 8) ceildiv 8)>(%arg9, %arg10, %arg11) {
          affine.parallel (%arg15) = (0) to (8) {
            affine.parallel (%arg16) = (0) to (8) {
              %7 = affine.apply affine_map<(d0, d1, d2) -> (d0 + d1 * 64 + d2 * 8)>(%arg16, %arg14, %arg15)
              %cst = arith.constant 0.000000e+00 : f32
              %8 = tensor.empty() : tensor<64x64xf32>
              %9 = linalg.fill ins(%cst : f32) outs(%8 : tensor<64x64xf32>) -> tensor<64x64xf32>
              %c64 = arith.constant 64 : index
              %10 = arith.muli %arg13, %c64 : index
              %11 = affine.apply affine_map<()[s0] -> ((s0 + 31) floordiv 32)>()[%arg5]
              %c0 = arith.constant 0 : index
              %c1 = arith.constant 1 : index
              %12 = scf.for %arg17 = %c0 to %11 step %c1 iter_args(%arg18 = %9) -> (tensor<64x64xf32>) {
                %c32 = arith.constant 32 : index
                %19 = arith.muli %arg17, %c32 : index
                %20 = arith.muli %10, %arg6 : index
                %21 = affine.apply affine_map<(d0)[s0] -> (d0 * 32 + s0)>(%arg17)[%20]
                %reinterpret_cast_0 = memref.reinterpret_cast %arg0 to offset: [%21], sizes: [64, 32], strides: [%arg6, 1] {tmd.reuse = {sequential = [{depth = 5 : i64, iterator = "%arg17", reuse_type = "no_reuse", volume = 0 : i64}], spatial = [{depth = 3 : i64, iterator = "%arg15", mapped_to = "x", reuse_type = "total_reuse", volume = 8192 : i64}, {depth = 4 : i64, iterator = "%arg16", mapped_to = "y", reuse_type = "total_reuse", volume = 8192 : i64}], temporal = [{depth = 0 : i64, iterator = "%arg12", reuse_type = "total_reuse", volume = 8192 : i64}, {depth = 1 : i64, iterator = "%arg13", reuse_type = "no_reuse", volume = 0 : i64}, {depth = 2 : i64, iterator = "%arg14", reuse_type = "total_reuse", volume = 8192 : i64}]}} : memref<*xf32> to memref<64x32xf32, strided<[?, 1], offset: ?>>
                %alloc = memref.alloc() : memref<64x32xf32>
                memref.copy %reinterpret_cast_0, %alloc : memref<64x32xf32, strided<[?, 1], offset: ?>> to memref<64x32xf32>
                %22 = bufferization.to_tensor %alloc restrict writable : memref<64x32xf32> to tensor<64x32xf32>
                %23 = arith.muli %19, %arg7 : index
                %24 = affine.apply affine_map<(d0, d1, d2)[s0] -> (d0 * 64 + d1 * 4096 + d2 * 512 + s0)>(%arg16, %arg14, %arg15)[%23]
                %reinterpret_cast_1 = memref.reinterpret_cast %arg1 to offset: [%24], sizes: [32, 64], strides: [%arg7, 1] {tmd.reuse = {sequential = [{depth = 5 : i64, iterator = "%arg17", reuse_type = "no_reuse", volume = 0 : i64}], spatial = [{depth = 3 : i64, iterator = "%arg15", mapped_to = "x", reuse_type = "no_reuse", volume = 0 : i64}, {depth = 4 : i64, iterator = "%arg16", mapped_to = "y", reuse_type = "no_reuse", volume = 0 : i64}], temporal = [{depth = 0 : i64, iterator = "%arg12", reuse_type = "total_reuse", volume = 8192 : i64}, {depth = 1 : i64, iterator = "%arg13", reuse_type = "total_reuse", volume = 8192 : i64}, {depth = 2 : i64, iterator = "%arg14", reuse_type = "no_reuse", volume = 0 : i64}]}} : memref<*xf32> to memref<32x64xf32, strided<[?, 1], offset: ?>>
                %alloc_2 = memref.alloc() : memref<32x64xf32>
                memref.copy %reinterpret_cast_1, %alloc_2 : memref<32x64xf32, strided<[?, 1], offset: ?>> to memref<32x64xf32>
                %25 = bufferization.to_tensor %alloc_2 restrict writable : memref<32x64xf32> to tensor<32x64xf32>
                %26 = linalg.matmul ins(%22, %25 : tensor<64x32xf32>, tensor<32x64xf32>) outs(%9 : tensor<64x64xf32>) -> tensor<64x64xf32>
                %27 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%arg18, %26 : tensor<64x64xf32>, tensor<64x64xf32>) outs(%arg18 : tensor<64x64xf32>) {
                ^bb0(%in: f32, %in_3: f32, %out: f32):
                  %28 = arith.addf %in, %in_3 : f32
                  linalg.yield %28 : f32
                } -> tensor<64x64xf32>
                scf.yield %27 : tensor<64x64xf32>
              }
              %13 = arith.muli %10, %arg8 : index
              %14 = affine.apply affine_map<(d0, d1, d2)[s0] -> (d0 * 64 + d1 * 4096 + d2 * 512 + s0)>(%arg16, %arg14, %arg15)[%13]
              %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%14], sizes: [64, 64], strides: [%arg8, 1] {tmd.reuse = {spatial = [{depth = 3 : i64, iterator = "%arg15", mapped_to = "x", reuse_type = "no_reuse", volume = 0 : i64}, {depth = 4 : i64, iterator = "%arg16", mapped_to = "y", reuse_type = "no_reuse", volume = 0 : i64}], temporal = [{depth = 0 : i64, iterator = "%arg12", reuse_type = "total_reuse", volume = 16384 : i64}, {depth = 1 : i64, iterator = "%arg13", reuse_type = "no_reuse", volume = 0 : i64}, {depth = 2 : i64, iterator = "%arg14", reuse_type = "no_reuse", volume = 0 : i64}]}} : memref<*xf32> to memref<64x64xf32, strided<[?, 1], offset: ?>>
              %15 = affine.min affine_map<(d0, d1, d2)[s0] -> (d0 * -64 + s0, 64)>(%arg13, %7, %arg12)[%arg3]
              %16 = affine.min affine_map<(d0, d1, d2)[s0] -> (d1 * -64 + s0, 64)>(%arg13, %7, %arg12)[%arg4]
              %extracted_slice = tensor.extract_slice %12[0, 0] [%15, %16] [1, 1] : tensor<64x64xf32> to tensor<?x?xf32>
              %17 = affine.min affine_map<(d0, d1, d2)[s0] -> (d0 * -64 + s0, 64)>(%arg13, %7, %arg12)[%arg3]
              %18 = affine.min affine_map<(d0, d1, d2)[s0] -> (d1 * -64 + s0, 64)>(%arg13, %7, %arg12)[%arg4]
              %subview = memref.subview %reinterpret_cast[0, 0] [%17, %18] [1, 1] : memref<64x64xf32, strided<[?, 1], offset: ?>> to memref<?x?xf32, strided<[?, 1], offset: ?>>
              bufferization.materialize_in_destination %extracted_slice in writable %subview : (tensor<?x?xf32>, memref<?x?xf32, strided<[?, 1], offset: ?>>) -> ()
            } {tmd.mapped_to = "y"}
          } {tmd.mapped_to = "x"}
        }
      }
    }
    return
  }
  func.func @matmul_kernel__d1i1_d0i1_f2_f1_f0(%arg0: memref<*xf32> {tt.divisibility = 16 : i32}, %arg1: memref<*xf32> {tt.divisibility = 16 : i32}, %arg2: memref<*xf32> {tt.divisibility = 16 : i32}, %arg3: index {tt.divisibility = 16 : i32}, %arg4: index {tt.divisibility = 16 : i32}, %arg5: index {tt.divisibility = 16 : i32}, %arg6: index {tt.divisibility = 16 : i32}, %arg7: index {tt.divisibility = 16 : i32}, %arg8: index {tt.divisibility = 16 : i32}, %arg9: index, %arg10: index, %arg11: index) {
    affine.for %arg12 = 0 to affine_map<(d0, d1, d2) -> (d2)>(%arg9, %arg10, %arg11) {
      affine.for %arg13 = 0 to affine_map<(d0, d1, d2) -> ((d1 ceildiv 8) ceildiv 8)>(%arg9, %arg10, %arg11) {
        affine.for %arg14 = 0 to affine_map<(d0, d1, d2) -> (d0)>(%arg9, %arg10, %arg11) {
          affine.parallel (%arg15) = (0) to (8) {
            affine.parallel (%arg16) = (0) to (8) {
              %7 = affine.apply affine_map<(d0, d1, d2) -> (d0 + d1 * 64 + d2 * 8)>(%arg16, %arg13, %arg15)
              %cst = arith.constant 0.000000e+00 : f32
              %8 = tensor.empty() : tensor<64x64xf32>
              %9 = linalg.fill ins(%cst : f32) outs(%8 : tensor<64x64xf32>) -> tensor<64x64xf32>
              %c64 = arith.constant 64 : index
              %10 = arith.muli %arg14, %c64 : index
              %11 = affine.apply affine_map<()[s0] -> ((s0 + 31) floordiv 32)>()[%arg5]
              %c0 = arith.constant 0 : index
              %c1 = arith.constant 1 : index
              %12 = scf.for %arg17 = %c0 to %11 step %c1 iter_args(%arg18 = %9) -> (tensor<64x64xf32>) {
                %c32 = arith.constant 32 : index
                %19 = arith.muli %arg17, %c32 : index
                %20 = arith.muli %10, %arg6 : index
                %21 = affine.apply affine_map<(d0)[s0] -> (d0 * 32 + s0)>(%arg17)[%20]
                %reinterpret_cast_0 = memref.reinterpret_cast %arg0 to offset: [%21], sizes: [64, 32], strides: [%arg6, 1] {tmd.reuse = {sequential = [{depth = 5 : i64, iterator = "%arg17", reuse_type = "no_reuse", volume = 0 : i64}], spatial = [{depth = 3 : i64, iterator = "%arg15", mapped_to = "x", reuse_type = "total_reuse", volume = 8192 : i64}, {depth = 4 : i64, iterator = "%arg16", mapped_to = "y", reuse_type = "total_reuse", volume = 8192 : i64}], temporal = [{depth = 0 : i64, iterator = "%arg12", reuse_type = "total_reuse", volume = 8192 : i64}, {depth = 1 : i64, iterator = "%arg13", reuse_type = "total_reuse", volume = 8192 : i64}, {depth = 2 : i64, iterator = "%arg14", reuse_type = "no_reuse", volume = 0 : i64}]}} : memref<*xf32> to memref<64x32xf32, strided<[?, 1], offset: ?>>
                %alloc = memref.alloc() : memref<64x32xf32>
                memref.copy %reinterpret_cast_0, %alloc : memref<64x32xf32, strided<[?, 1], offset: ?>> to memref<64x32xf32>
                %22 = bufferization.to_tensor %alloc restrict writable : memref<64x32xf32> to tensor<64x32xf32>
                %23 = arith.muli %19, %arg7 : index
                %24 = affine.apply affine_map<(d0, d1, d2)[s0] -> (d0 * 64 + d1 * 4096 + d2 * 512 + s0)>(%arg16, %arg13, %arg15)[%23]
                %reinterpret_cast_1 = memref.reinterpret_cast %arg1 to offset: [%24], sizes: [32, 64], strides: [%arg7, 1] {tmd.reuse = {sequential = [{depth = 5 : i64, iterator = "%arg17", reuse_type = "no_reuse", volume = 0 : i64}], spatial = [{depth = 3 : i64, iterator = "%arg15", mapped_to = "x", reuse_type = "no_reuse", volume = 0 : i64}, {depth = 4 : i64, iterator = "%arg16", mapped_to = "y", reuse_type = "no_reuse", volume = 0 : i64}], temporal = [{depth = 0 : i64, iterator = "%arg12", reuse_type = "total_reuse", volume = 8192 : i64}, {depth = 1 : i64, iterator = "%arg13", reuse_type = "no_reuse", volume = 0 : i64}, {depth = 2 : i64, iterator = "%arg14", reuse_type = "total_reuse", volume = 8192 : i64}]}} : memref<*xf32> to memref<32x64xf32, strided<[?, 1], offset: ?>>
                %alloc_2 = memref.alloc() : memref<32x64xf32>
                memref.copy %reinterpret_cast_1, %alloc_2 : memref<32x64xf32, strided<[?, 1], offset: ?>> to memref<32x64xf32>
                %25 = bufferization.to_tensor %alloc_2 restrict writable : memref<32x64xf32> to tensor<32x64xf32>
                %26 = linalg.matmul ins(%22, %25 : tensor<64x32xf32>, tensor<32x64xf32>) outs(%9 : tensor<64x64xf32>) -> tensor<64x64xf32>
                %27 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%arg18, %26 : tensor<64x64xf32>, tensor<64x64xf32>) outs(%arg18 : tensor<64x64xf32>) {
                ^bb0(%in: f32, %in_3: f32, %out: f32):
                  %28 = arith.addf %in, %in_3 : f32
                  linalg.yield %28 : f32
                } -> tensor<64x64xf32>
                scf.yield %27 : tensor<64x64xf32>
              }
              %13 = arith.muli %10, %arg8 : index
              %14 = affine.apply affine_map<(d0, d1, d2)[s0] -> (d0 * 64 + d1 * 4096 + d2 * 512 + s0)>(%arg16, %arg13, %arg15)[%13]
              %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%14], sizes: [64, 64], strides: [%arg8, 1] {tmd.reuse = {spatial = [{depth = 3 : i64, iterator = "%arg15", mapped_to = "x", reuse_type = "no_reuse", volume = 0 : i64}, {depth = 4 : i64, iterator = "%arg16", mapped_to = "y", reuse_type = "no_reuse", volume = 0 : i64}], temporal = [{depth = 0 : i64, iterator = "%arg12", reuse_type = "total_reuse", volume = 16384 : i64}, {depth = 1 : i64, iterator = "%arg13", reuse_type = "no_reuse", volume = 0 : i64}, {depth = 2 : i64, iterator = "%arg14", reuse_type = "no_reuse", volume = 0 : i64}]}} : memref<*xf32> to memref<64x64xf32, strided<[?, 1], offset: ?>>
              %15 = affine.min affine_map<(d0, d1, d2)[s0] -> (d0 * -64 + s0, 64)>(%arg14, %7, %arg12)[%arg3]
              %16 = affine.min affine_map<(d0, d1, d2)[s0] -> (d1 * -64 + s0, 64)>(%arg14, %7, %arg12)[%arg4]
              %extracted_slice = tensor.extract_slice %12[0, 0] [%15, %16] [1, 1] : tensor<64x64xf32> to tensor<?x?xf32>
              %17 = affine.min affine_map<(d0, d1, d2)[s0] -> (d0 * -64 + s0, 64)>(%arg14, %7, %arg12)[%arg3]
              %18 = affine.min affine_map<(d0, d1, d2)[s0] -> (d1 * -64 + s0, 64)>(%arg14, %7, %arg12)[%arg4]
              %subview = memref.subview %reinterpret_cast[0, 0] [%17, %18] [1, 1] : memref<64x64xf32, strided<[?, 1], offset: ?>> to memref<?x?xf32, strided<[?, 1], offset: ?>>
              bufferization.materialize_in_destination %extracted_slice in writable %subview : (tensor<?x?xf32>, memref<?x?xf32, strided<[?, 1], offset: ?>>) -> ()
            } {tmd.mapped_to = "y"}
          } {tmd.mapped_to = "x"}
        }
      }
    }
    return
  }
  func.func @matmul_kernel__d0i1_d1i2_f0_f1_f2(%arg0: memref<*xf32> {tt.divisibility = 16 : i32}, %arg1: memref<*xf32> {tt.divisibility = 16 : i32}, %arg2: memref<*xf32> {tt.divisibility = 16 : i32}, %arg3: index {tt.divisibility = 16 : i32}, %arg4: index {tt.divisibility = 16 : i32}, %arg5: index {tt.divisibility = 16 : i32}, %arg6: index {tt.divisibility = 16 : i32}, %arg7: index {tt.divisibility = 16 : i32}, %arg8: index {tt.divisibility = 16 : i32}, %arg9: index, %arg10: index, %arg11: index) {
    affine.for %arg12 = 0 to affine_map<(d0, d1, d2) -> (d0)>(%arg9, %arg10, %arg11) {
      affine.for %arg13 = 0 to affine_map<(d0, d1, d2) -> (d1 ceildiv 8)>(%arg9, %arg10, %arg11) {
        affine.for %arg14 = 0 to affine_map<(d0, d1, d2) -> (d2 ceildiv 8)>(%arg9, %arg10, %arg11) {
          affine.parallel (%arg15) = (0) to (8) {
            %7 = affine.apply affine_map<(d0, d1) -> (d0 * 8 + d1)>(%arg14, %arg15)
            affine.parallel (%arg16) = (0) to (8) {
              %8 = affine.apply affine_map<(d0, d1) -> (d0 * 8 + d1)>(%arg13, %arg16)
              %cst = arith.constant 0.000000e+00 : f32
              %9 = tensor.empty() : tensor<64x64xf32>
              %10 = linalg.fill ins(%cst : f32) outs(%9 : tensor<64x64xf32>) -> tensor<64x64xf32>
              %c64 = arith.constant 64 : index
              %11 = arith.muli %arg12, %c64 : index
              %12 = affine.apply affine_map<()[s0] -> ((s0 + 31) floordiv 32)>()[%arg5]
              %c0 = arith.constant 0 : index
              %c1 = arith.constant 1 : index
              %13 = scf.for %arg17 = %c0 to %12 step %c1 iter_args(%arg18 = %10) -> (tensor<64x64xf32>) {
                %c32 = arith.constant 32 : index
                %20 = arith.muli %arg17, %c32 : index
                %21 = arith.muli %11, %arg6 : index
                %22 = affine.apply affine_map<(d0)[s0] -> (d0 * 32 + s0)>(%arg17)[%21]
                %reinterpret_cast_0 = memref.reinterpret_cast %arg0 to offset: [%22], sizes: [64, 32], strides: [%arg6, 1] {tmd.reuse = {sequential = [{depth = 5 : i64, iterator = "%arg17", reuse_type = "no_reuse", volume = 0 : i64}], spatial = [{depth = 3 : i64, iterator = "%arg15", mapped_to = "y", reuse_type = "total_reuse", volume = 8192 : i64}, {depth = 4 : i64, iterator = "%arg16", mapped_to = "x", reuse_type = "total_reuse", volume = 8192 : i64}], temporal = [{depth = 0 : i64, iterator = "%arg12", reuse_type = "no_reuse", volume = 0 : i64}, {depth = 1 : i64, iterator = "%arg13", reuse_type = "total_reuse", volume = 8192 : i64}, {depth = 2 : i64, iterator = "%arg14", reuse_type = "total_reuse", volume = 8192 : i64}]}} : memref<*xf32> to memref<64x32xf32, strided<[?, 1], offset: ?>>
                %alloc = memref.alloc() : memref<64x32xf32>
                memref.copy %reinterpret_cast_0, %alloc : memref<64x32xf32, strided<[?, 1], offset: ?>> to memref<64x32xf32>
                %23 = bufferization.to_tensor %alloc restrict writable : memref<64x32xf32> to tensor<64x32xf32>
                %24 = arith.muli %20, %arg7 : index
                %25 = affine.apply affine_map<(d0, d1)[s0] -> (d0 * 512 + d1 * 64 + s0)>(%arg13, %arg16)[%24]
                %reinterpret_cast_1 = memref.reinterpret_cast %arg1 to offset: [%25], sizes: [32, 64], strides: [%arg7, 1] {tmd.reuse = {sequential = [{depth = 5 : i64, iterator = "%arg17", reuse_type = "no_reuse", volume = 0 : i64}], spatial = [{depth = 3 : i64, iterator = "%arg15", mapped_to = "y", reuse_type = "total_reuse", volume = 8192 : i64}, {depth = 4 : i64, iterator = "%arg16", mapped_to = "x", reuse_type = "no_reuse", volume = 0 : i64}], temporal = [{depth = 0 : i64, iterator = "%arg12", reuse_type = "total_reuse", volume = 8192 : i64}, {depth = 1 : i64, iterator = "%arg13", reuse_type = "no_reuse", volume = 0 : i64}, {depth = 2 : i64, iterator = "%arg14", reuse_type = "total_reuse", volume = 8192 : i64}]}} : memref<*xf32> to memref<32x64xf32, strided<[?, 1], offset: ?>>
                %alloc_2 = memref.alloc() : memref<32x64xf32>
                memref.copy %reinterpret_cast_1, %alloc_2 : memref<32x64xf32, strided<[?, 1], offset: ?>> to memref<32x64xf32>
                %26 = bufferization.to_tensor %alloc_2 restrict writable : memref<32x64xf32> to tensor<32x64xf32>
                %27 = linalg.matmul ins(%23, %26 : tensor<64x32xf32>, tensor<32x64xf32>) outs(%10 : tensor<64x64xf32>) -> tensor<64x64xf32>
                %28 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%arg18, %27 : tensor<64x64xf32>, tensor<64x64xf32>) outs(%arg18 : tensor<64x64xf32>) {
                ^bb0(%in: f32, %in_3: f32, %out: f32):
                  %29 = arith.addf %in, %in_3 : f32
                  linalg.yield %29 : f32
                } -> tensor<64x64xf32>
                scf.yield %28 : tensor<64x64xf32>
              }
              %14 = arith.muli %11, %arg8 : index
              %15 = affine.apply affine_map<(d0, d1)[s0] -> (d0 * 512 + d1 * 64 + s0)>(%arg13, %arg16)[%14]
              %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%15], sizes: [64, 64], strides: [%arg8, 1] {tmd.reuse = {spatial = [{depth = 3 : i64, iterator = "%arg15", mapped_to = "y", reuse_type = "total_reuse", volume = 16384 : i64}, {depth = 4 : i64, iterator = "%arg16", mapped_to = "x", reuse_type = "no_reuse", volume = 0 : i64}], temporal = [{depth = 0 : i64, iterator = "%arg12", reuse_type = "no_reuse", volume = 0 : i64}, {depth = 1 : i64, iterator = "%arg13", reuse_type = "no_reuse", volume = 0 : i64}, {depth = 2 : i64, iterator = "%arg14", reuse_type = "total_reuse", volume = 16384 : i64}]}} : memref<*xf32> to memref<64x64xf32, strided<[?, 1], offset: ?>>
              %16 = affine.min affine_map<(d0, d1, d2)[s0] -> (d0 * -64 + s0, 64)>(%arg12, %8, %7)[%arg3]
              %17 = affine.min affine_map<(d0, d1, d2)[s0] -> (d1 * -64 + s0, 64)>(%arg12, %8, %7)[%arg4]
              %extracted_slice = tensor.extract_slice %13[0, 0] [%16, %17] [1, 1] : tensor<64x64xf32> to tensor<?x?xf32>
              %18 = affine.min affine_map<(d0, d1, d2)[s0] -> (d0 * -64 + s0, 64)>(%arg12, %8, %7)[%arg3]
              %19 = affine.min affine_map<(d0, d1, d2)[s0] -> (d1 * -64 + s0, 64)>(%arg12, %8, %7)[%arg4]
              %subview = memref.subview %reinterpret_cast[0, 0] [%18, %19] [1, 1] : memref<64x64xf32, strided<[?, 1], offset: ?>> to memref<?x?xf32, strided<[?, 1], offset: ?>>
              bufferization.materialize_in_destination %extracted_slice in writable %subview : (tensor<?x?xf32>, memref<?x?xf32, strided<[?, 1], offset: ?>>) -> ()
            } {tmd.mapped_to = "x"}
          } {tmd.mapped_to = "y"}
        }
      }
    }
    return
  }
  func.func @matmul_kernel__d0i1_d1i2_f0_f2_f1(%arg0: memref<*xf32> {tt.divisibility = 16 : i32}, %arg1: memref<*xf32> {tt.divisibility = 16 : i32}, %arg2: memref<*xf32> {tt.divisibility = 16 : i32}, %arg3: index {tt.divisibility = 16 : i32}, %arg4: index {tt.divisibility = 16 : i32}, %arg5: index {tt.divisibility = 16 : i32}, %arg6: index {tt.divisibility = 16 : i32}, %arg7: index {tt.divisibility = 16 : i32}, %arg8: index {tt.divisibility = 16 : i32}, %arg9: index, %arg10: index, %arg11: index) {
    affine.for %arg12 = 0 to affine_map<(d0, d1, d2) -> (d0)>(%arg9, %arg10, %arg11) {
      affine.for %arg13 = 0 to affine_map<(d0, d1, d2) -> (d2 ceildiv 8)>(%arg9, %arg10, %arg11) {
        affine.for %arg14 = 0 to affine_map<(d0, d1, d2) -> (d1 ceildiv 8)>(%arg9, %arg10, %arg11) {
          affine.parallel (%arg15) = (0) to (8) {
            %7 = affine.apply affine_map<(d0, d1) -> (d0 * 8 + d1)>(%arg13, %arg15)
            affine.parallel (%arg16) = (0) to (8) {
              %8 = affine.apply affine_map<(d0, d1) -> (d0 * 8 + d1)>(%arg14, %arg16)
              %cst = arith.constant 0.000000e+00 : f32
              %9 = tensor.empty() : tensor<64x64xf32>
              %10 = linalg.fill ins(%cst : f32) outs(%9 : tensor<64x64xf32>) -> tensor<64x64xf32>
              %c64 = arith.constant 64 : index
              %11 = arith.muli %arg12, %c64 : index
              %12 = affine.apply affine_map<()[s0] -> ((s0 + 31) floordiv 32)>()[%arg5]
              %c0 = arith.constant 0 : index
              %c1 = arith.constant 1 : index
              %13 = scf.for %arg17 = %c0 to %12 step %c1 iter_args(%arg18 = %10) -> (tensor<64x64xf32>) {
                %c32 = arith.constant 32 : index
                %20 = arith.muli %arg17, %c32 : index
                %21 = arith.muli %11, %arg6 : index
                %22 = affine.apply affine_map<(d0)[s0] -> (d0 * 32 + s0)>(%arg17)[%21]
                %reinterpret_cast_0 = memref.reinterpret_cast %arg0 to offset: [%22], sizes: [64, 32], strides: [%arg6, 1] {tmd.reuse = {sequential = [{depth = 5 : i64, iterator = "%arg17", reuse_type = "no_reuse", volume = 0 : i64}], spatial = [{depth = 3 : i64, iterator = "%arg15", mapped_to = "y", reuse_type = "total_reuse", volume = 8192 : i64}, {depth = 4 : i64, iterator = "%arg16", mapped_to = "x", reuse_type = "total_reuse", volume = 8192 : i64}], temporal = [{depth = 0 : i64, iterator = "%arg12", reuse_type = "no_reuse", volume = 0 : i64}, {depth = 1 : i64, iterator = "%arg13", reuse_type = "total_reuse", volume = 8192 : i64}, {depth = 2 : i64, iterator = "%arg14", reuse_type = "total_reuse", volume = 8192 : i64}]}} : memref<*xf32> to memref<64x32xf32, strided<[?, 1], offset: ?>>
                %alloc = memref.alloc() : memref<64x32xf32>
                memref.copy %reinterpret_cast_0, %alloc : memref<64x32xf32, strided<[?, 1], offset: ?>> to memref<64x32xf32>
                %23 = bufferization.to_tensor %alloc restrict writable : memref<64x32xf32> to tensor<64x32xf32>
                %24 = arith.muli %20, %arg7 : index
                %25 = affine.apply affine_map<(d0, d1)[s0] -> (d0 * 512 + d1 * 64 + s0)>(%arg14, %arg16)[%24]
                %reinterpret_cast_1 = memref.reinterpret_cast %arg1 to offset: [%25], sizes: [32, 64], strides: [%arg7, 1] {tmd.reuse = {sequential = [{depth = 5 : i64, iterator = "%arg17", reuse_type = "no_reuse", volume = 0 : i64}], spatial = [{depth = 3 : i64, iterator = "%arg15", mapped_to = "y", reuse_type = "total_reuse", volume = 8192 : i64}, {depth = 4 : i64, iterator = "%arg16", mapped_to = "x", reuse_type = "no_reuse", volume = 0 : i64}], temporal = [{depth = 0 : i64, iterator = "%arg12", reuse_type = "total_reuse", volume = 8192 : i64}, {depth = 1 : i64, iterator = "%arg13", reuse_type = "total_reuse", volume = 8192 : i64}, {depth = 2 : i64, iterator = "%arg14", reuse_type = "no_reuse", volume = 0 : i64}]}} : memref<*xf32> to memref<32x64xf32, strided<[?, 1], offset: ?>>
                %alloc_2 = memref.alloc() : memref<32x64xf32>
                memref.copy %reinterpret_cast_1, %alloc_2 : memref<32x64xf32, strided<[?, 1], offset: ?>> to memref<32x64xf32>
                %26 = bufferization.to_tensor %alloc_2 restrict writable : memref<32x64xf32> to tensor<32x64xf32>
                %27 = linalg.matmul ins(%23, %26 : tensor<64x32xf32>, tensor<32x64xf32>) outs(%10 : tensor<64x64xf32>) -> tensor<64x64xf32>
                %28 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%arg18, %27 : tensor<64x64xf32>, tensor<64x64xf32>) outs(%arg18 : tensor<64x64xf32>) {
                ^bb0(%in: f32, %in_3: f32, %out: f32):
                  %29 = arith.addf %in, %in_3 : f32
                  linalg.yield %29 : f32
                } -> tensor<64x64xf32>
                scf.yield %28 : tensor<64x64xf32>
              }
              %14 = arith.muli %11, %arg8 : index
              %15 = affine.apply affine_map<(d0, d1)[s0] -> (d0 * 512 + d1 * 64 + s0)>(%arg14, %arg16)[%14]
              %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%15], sizes: [64, 64], strides: [%arg8, 1] {tmd.reuse = {spatial = [{depth = 3 : i64, iterator = "%arg15", mapped_to = "y", reuse_type = "total_reuse", volume = 16384 : i64}, {depth = 4 : i64, iterator = "%arg16", mapped_to = "x", reuse_type = "no_reuse", volume = 0 : i64}], temporal = [{depth = 0 : i64, iterator = "%arg12", reuse_type = "no_reuse", volume = 0 : i64}, {depth = 1 : i64, iterator = "%arg13", reuse_type = "total_reuse", volume = 16384 : i64}, {depth = 2 : i64, iterator = "%arg14", reuse_type = "no_reuse", volume = 0 : i64}]}} : memref<*xf32> to memref<64x64xf32, strided<[?, 1], offset: ?>>
              %16 = affine.min affine_map<(d0, d1, d2)[s0] -> (d0 * -64 + s0, 64)>(%arg12, %8, %7)[%arg3]
              %17 = affine.min affine_map<(d0, d1, d2)[s0] -> (d1 * -64 + s0, 64)>(%arg12, %8, %7)[%arg4]
              %extracted_slice = tensor.extract_slice %13[0, 0] [%16, %17] [1, 1] : tensor<64x64xf32> to tensor<?x?xf32>
              %18 = affine.min affine_map<(d0, d1, d2)[s0] -> (d0 * -64 + s0, 64)>(%arg12, %8, %7)[%arg3]
              %19 = affine.min affine_map<(d0, d1, d2)[s0] -> (d1 * -64 + s0, 64)>(%arg12, %8, %7)[%arg4]
              %subview = memref.subview %reinterpret_cast[0, 0] [%18, %19] [1, 1] : memref<64x64xf32, strided<[?, 1], offset: ?>> to memref<?x?xf32, strided<[?, 1], offset: ?>>
              bufferization.materialize_in_destination %extracted_slice in writable %subview : (tensor<?x?xf32>, memref<?x?xf32, strided<[?, 1], offset: ?>>) -> ()
            } {tmd.mapped_to = "x"}
          } {tmd.mapped_to = "y"}
        }
      }
    }
    return
  }
  func.func @matmul_kernel__d0i1_d1i2_f1_f0_f2(%arg0: memref<*xf32> {tt.divisibility = 16 : i32}, %arg1: memref<*xf32> {tt.divisibility = 16 : i32}, %arg2: memref<*xf32> {tt.divisibility = 16 : i32}, %arg3: index {tt.divisibility = 16 : i32}, %arg4: index {tt.divisibility = 16 : i32}, %arg5: index {tt.divisibility = 16 : i32}, %arg6: index {tt.divisibility = 16 : i32}, %arg7: index {tt.divisibility = 16 : i32}, %arg8: index {tt.divisibility = 16 : i32}, %arg9: index, %arg10: index, %arg11: index) {
    affine.for %arg12 = 0 to affine_map<(d0, d1, d2) -> (d1 ceildiv 8)>(%arg9, %arg10, %arg11) {
      affine.for %arg13 = 0 to affine_map<(d0, d1, d2) -> (d0)>(%arg9, %arg10, %arg11) {
        affine.for %arg14 = 0 to affine_map<(d0, d1, d2) -> (d2 ceildiv 8)>(%arg9, %arg10, %arg11) {
          affine.parallel (%arg15) = (0) to (8) {
            %7 = affine.apply affine_map<(d0, d1) -> (d0 * 8 + d1)>(%arg14, %arg15)
            affine.parallel (%arg16) = (0) to (8) {
              %8 = affine.apply affine_map<(d0, d1) -> (d0 * 8 + d1)>(%arg12, %arg16)
              %cst = arith.constant 0.000000e+00 : f32
              %9 = tensor.empty() : tensor<64x64xf32>
              %10 = linalg.fill ins(%cst : f32) outs(%9 : tensor<64x64xf32>) -> tensor<64x64xf32>
              %c64 = arith.constant 64 : index
              %11 = arith.muli %arg13, %c64 : index
              %12 = affine.apply affine_map<()[s0] -> ((s0 + 31) floordiv 32)>()[%arg5]
              %c0 = arith.constant 0 : index
              %c1 = arith.constant 1 : index
              %13 = scf.for %arg17 = %c0 to %12 step %c1 iter_args(%arg18 = %10) -> (tensor<64x64xf32>) {
                %c32 = arith.constant 32 : index
                %20 = arith.muli %arg17, %c32 : index
                %21 = arith.muli %11, %arg6 : index
                %22 = affine.apply affine_map<(d0)[s0] -> (d0 * 32 + s0)>(%arg17)[%21]
                %reinterpret_cast_0 = memref.reinterpret_cast %arg0 to offset: [%22], sizes: [64, 32], strides: [%arg6, 1] {tmd.reuse = {sequential = [{depth = 5 : i64, iterator = "%arg17", reuse_type = "no_reuse", volume = 0 : i64}], spatial = [{depth = 3 : i64, iterator = "%arg15", mapped_to = "y", reuse_type = "total_reuse", volume = 8192 : i64}, {depth = 4 : i64, iterator = "%arg16", mapped_to = "x", reuse_type = "total_reuse", volume = 8192 : i64}], temporal = [{depth = 0 : i64, iterator = "%arg12", reuse_type = "total_reuse", volume = 8192 : i64}, {depth = 1 : i64, iterator = "%arg13", reuse_type = "no_reuse", volume = 0 : i64}, {depth = 2 : i64, iterator = "%arg14", reuse_type = "total_reuse", volume = 8192 : i64}]}} : memref<*xf32> to memref<64x32xf32, strided<[?, 1], offset: ?>>
                %alloc = memref.alloc() : memref<64x32xf32>
                memref.copy %reinterpret_cast_0, %alloc : memref<64x32xf32, strided<[?, 1], offset: ?>> to memref<64x32xf32>
                %23 = bufferization.to_tensor %alloc restrict writable : memref<64x32xf32> to tensor<64x32xf32>
                %24 = arith.muli %20, %arg7 : index
                %25 = affine.apply affine_map<(d0, d1)[s0] -> (d0 * 512 + d1 * 64 + s0)>(%arg12, %arg16)[%24]
                %reinterpret_cast_1 = memref.reinterpret_cast %arg1 to offset: [%25], sizes: [32, 64], strides: [%arg7, 1] {tmd.reuse = {sequential = [{depth = 5 : i64, iterator = "%arg17", reuse_type = "no_reuse", volume = 0 : i64}], spatial = [{depth = 3 : i64, iterator = "%arg15", mapped_to = "y", reuse_type = "total_reuse", volume = 8192 : i64}, {depth = 4 : i64, iterator = "%arg16", mapped_to = "x", reuse_type = "no_reuse", volume = 0 : i64}], temporal = [{depth = 0 : i64, iterator = "%arg12", reuse_type = "no_reuse", volume = 0 : i64}, {depth = 1 : i64, iterator = "%arg13", reuse_type = "total_reuse", volume = 8192 : i64}, {depth = 2 : i64, iterator = "%arg14", reuse_type = "total_reuse", volume = 8192 : i64}]}} : memref<*xf32> to memref<32x64xf32, strided<[?, 1], offset: ?>>
                %alloc_2 = memref.alloc() : memref<32x64xf32>
                memref.copy %reinterpret_cast_1, %alloc_2 : memref<32x64xf32, strided<[?, 1], offset: ?>> to memref<32x64xf32>
                %26 = bufferization.to_tensor %alloc_2 restrict writable : memref<32x64xf32> to tensor<32x64xf32>
                %27 = linalg.matmul ins(%23, %26 : tensor<64x32xf32>, tensor<32x64xf32>) outs(%10 : tensor<64x64xf32>) -> tensor<64x64xf32>
                %28 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%arg18, %27 : tensor<64x64xf32>, tensor<64x64xf32>) outs(%arg18 : tensor<64x64xf32>) {
                ^bb0(%in: f32, %in_3: f32, %out: f32):
                  %29 = arith.addf %in, %in_3 : f32
                  linalg.yield %29 : f32
                } -> tensor<64x64xf32>
                scf.yield %28 : tensor<64x64xf32>
              }
              %14 = arith.muli %11, %arg8 : index
              %15 = affine.apply affine_map<(d0, d1)[s0] -> (d0 * 512 + d1 * 64 + s0)>(%arg12, %arg16)[%14]
              %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%15], sizes: [64, 64], strides: [%arg8, 1] {tmd.reuse = {spatial = [{depth = 3 : i64, iterator = "%arg15", mapped_to = "y", reuse_type = "total_reuse", volume = 16384 : i64}, {depth = 4 : i64, iterator = "%arg16", mapped_to = "x", reuse_type = "no_reuse", volume = 0 : i64}], temporal = [{depth = 0 : i64, iterator = "%arg12", reuse_type = "no_reuse", volume = 0 : i64}, {depth = 1 : i64, iterator = "%arg13", reuse_type = "no_reuse", volume = 0 : i64}, {depth = 2 : i64, iterator = "%arg14", reuse_type = "total_reuse", volume = 16384 : i64}]}} : memref<*xf32> to memref<64x64xf32, strided<[?, 1], offset: ?>>
              %16 = affine.min affine_map<(d0, d1, d2)[s0] -> (d0 * -64 + s0, 64)>(%arg13, %8, %7)[%arg3]
              %17 = affine.min affine_map<(d0, d1, d2)[s0] -> (d1 * -64 + s0, 64)>(%arg13, %8, %7)[%arg4]
              %extracted_slice = tensor.extract_slice %13[0, 0] [%16, %17] [1, 1] : tensor<64x64xf32> to tensor<?x?xf32>
              %18 = affine.min affine_map<(d0, d1, d2)[s0] -> (d0 * -64 + s0, 64)>(%arg13, %8, %7)[%arg3]
              %19 = affine.min affine_map<(d0, d1, d2)[s0] -> (d1 * -64 + s0, 64)>(%arg13, %8, %7)[%arg4]
              %subview = memref.subview %reinterpret_cast[0, 0] [%18, %19] [1, 1] : memref<64x64xf32, strided<[?, 1], offset: ?>> to memref<?x?xf32, strided<[?, 1], offset: ?>>
              bufferization.materialize_in_destination %extracted_slice in writable %subview : (tensor<?x?xf32>, memref<?x?xf32, strided<[?, 1], offset: ?>>) -> ()
            } {tmd.mapped_to = "x"}
          } {tmd.mapped_to = "y"}
        }
      }
    }
    return
  }
  func.func @matmul_kernel__d0i1_d1i2_f1_f2_f0(%arg0: memref<*xf32> {tt.divisibility = 16 : i32}, %arg1: memref<*xf32> {tt.divisibility = 16 : i32}, %arg2: memref<*xf32> {tt.divisibility = 16 : i32}, %arg3: index {tt.divisibility = 16 : i32}, %arg4: index {tt.divisibility = 16 : i32}, %arg5: index {tt.divisibility = 16 : i32}, %arg6: index {tt.divisibility = 16 : i32}, %arg7: index {tt.divisibility = 16 : i32}, %arg8: index {tt.divisibility = 16 : i32}, %arg9: index, %arg10: index, %arg11: index) {
    affine.for %arg12 = 0 to affine_map<(d0, d1, d2) -> (d1 ceildiv 8)>(%arg9, %arg10, %arg11) {
      affine.for %arg13 = 0 to affine_map<(d0, d1, d2) -> (d2 ceildiv 8)>(%arg9, %arg10, %arg11) {
        affine.for %arg14 = 0 to affine_map<(d0, d1, d2) -> (d0)>(%arg9, %arg10, %arg11) {
          affine.parallel (%arg15) = (0) to (8) {
            %7 = affine.apply affine_map<(d0, d1) -> (d0 * 8 + d1)>(%arg13, %arg15)
            affine.parallel (%arg16) = (0) to (8) {
              %8 = affine.apply affine_map<(d0, d1) -> (d0 * 8 + d1)>(%arg12, %arg16)
              %cst = arith.constant 0.000000e+00 : f32
              %9 = tensor.empty() : tensor<64x64xf32>
              %10 = linalg.fill ins(%cst : f32) outs(%9 : tensor<64x64xf32>) -> tensor<64x64xf32>
              %c64 = arith.constant 64 : index
              %11 = arith.muli %arg14, %c64 : index
              %12 = affine.apply affine_map<()[s0] -> ((s0 + 31) floordiv 32)>()[%arg5]
              %c0 = arith.constant 0 : index
              %c1 = arith.constant 1 : index
              %13 = scf.for %arg17 = %c0 to %12 step %c1 iter_args(%arg18 = %10) -> (tensor<64x64xf32>) {
                %c32 = arith.constant 32 : index
                %20 = arith.muli %arg17, %c32 : index
                %21 = arith.muli %11, %arg6 : index
                %22 = affine.apply affine_map<(d0)[s0] -> (d0 * 32 + s0)>(%arg17)[%21]
                %reinterpret_cast_0 = memref.reinterpret_cast %arg0 to offset: [%22], sizes: [64, 32], strides: [%arg6, 1] {tmd.reuse = {sequential = [{depth = 5 : i64, iterator = "%arg17", reuse_type = "no_reuse", volume = 0 : i64}], spatial = [{depth = 3 : i64, iterator = "%arg15", mapped_to = "y", reuse_type = "total_reuse", volume = 8192 : i64}, {depth = 4 : i64, iterator = "%arg16", mapped_to = "x", reuse_type = "total_reuse", volume = 8192 : i64}], temporal = [{depth = 0 : i64, iterator = "%arg12", reuse_type = "total_reuse", volume = 8192 : i64}, {depth = 1 : i64, iterator = "%arg13", reuse_type = "total_reuse", volume = 8192 : i64}, {depth = 2 : i64, iterator = "%arg14", reuse_type = "no_reuse", volume = 0 : i64}]}} : memref<*xf32> to memref<64x32xf32, strided<[?, 1], offset: ?>>
                %alloc = memref.alloc() : memref<64x32xf32>
                memref.copy %reinterpret_cast_0, %alloc : memref<64x32xf32, strided<[?, 1], offset: ?>> to memref<64x32xf32>
                %23 = bufferization.to_tensor %alloc restrict writable : memref<64x32xf32> to tensor<64x32xf32>
                %24 = arith.muli %20, %arg7 : index
                %25 = affine.apply affine_map<(d0, d1)[s0] -> (d0 * 512 + d1 * 64 + s0)>(%arg12, %arg16)[%24]
                %reinterpret_cast_1 = memref.reinterpret_cast %arg1 to offset: [%25], sizes: [32, 64], strides: [%arg7, 1] {tmd.reuse = {sequential = [{depth = 5 : i64, iterator = "%arg17", reuse_type = "no_reuse", volume = 0 : i64}], spatial = [{depth = 3 : i64, iterator = "%arg15", mapped_to = "y", reuse_type = "total_reuse", volume = 8192 : i64}, {depth = 4 : i64, iterator = "%arg16", mapped_to = "x", reuse_type = "no_reuse", volume = 0 : i64}], temporal = [{depth = 0 : i64, iterator = "%arg12", reuse_type = "no_reuse", volume = 0 : i64}, {depth = 1 : i64, iterator = "%arg13", reuse_type = "total_reuse", volume = 8192 : i64}, {depth = 2 : i64, iterator = "%arg14", reuse_type = "total_reuse", volume = 8192 : i64}]}} : memref<*xf32> to memref<32x64xf32, strided<[?, 1], offset: ?>>
                %alloc_2 = memref.alloc() : memref<32x64xf32>
                memref.copy %reinterpret_cast_1, %alloc_2 : memref<32x64xf32, strided<[?, 1], offset: ?>> to memref<32x64xf32>
                %26 = bufferization.to_tensor %alloc_2 restrict writable : memref<32x64xf32> to tensor<32x64xf32>
                %27 = linalg.matmul ins(%23, %26 : tensor<64x32xf32>, tensor<32x64xf32>) outs(%10 : tensor<64x64xf32>) -> tensor<64x64xf32>
                %28 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%arg18, %27 : tensor<64x64xf32>, tensor<64x64xf32>) outs(%arg18 : tensor<64x64xf32>) {
                ^bb0(%in: f32, %in_3: f32, %out: f32):
                  %29 = arith.addf %in, %in_3 : f32
                  linalg.yield %29 : f32
                } -> tensor<64x64xf32>
                scf.yield %28 : tensor<64x64xf32>
              }
              %14 = arith.muli %11, %arg8 : index
              %15 = affine.apply affine_map<(d0, d1)[s0] -> (d0 * 512 + d1 * 64 + s0)>(%arg12, %arg16)[%14]
              %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%15], sizes: [64, 64], strides: [%arg8, 1] {tmd.reuse = {spatial = [{depth = 3 : i64, iterator = "%arg15", mapped_to = "y", reuse_type = "total_reuse", volume = 16384 : i64}, {depth = 4 : i64, iterator = "%arg16", mapped_to = "x", reuse_type = "no_reuse", volume = 0 : i64}], temporal = [{depth = 0 : i64, iterator = "%arg12", reuse_type = "no_reuse", volume = 0 : i64}, {depth = 1 : i64, iterator = "%arg13", reuse_type = "total_reuse", volume = 16384 : i64}, {depth = 2 : i64, iterator = "%arg14", reuse_type = "no_reuse", volume = 0 : i64}]}} : memref<*xf32> to memref<64x64xf32, strided<[?, 1], offset: ?>>
              %16 = affine.min affine_map<(d0, d1, d2)[s0] -> (d0 * -64 + s0, 64)>(%arg14, %8, %7)[%arg3]
              %17 = affine.min affine_map<(d0, d1, d2)[s0] -> (d1 * -64 + s0, 64)>(%arg14, %8, %7)[%arg4]
              %extracted_slice = tensor.extract_slice %13[0, 0] [%16, %17] [1, 1] : tensor<64x64xf32> to tensor<?x?xf32>
              %18 = affine.min affine_map<(d0, d1, d2)[s0] -> (d0 * -64 + s0, 64)>(%arg14, %8, %7)[%arg3]
              %19 = affine.min affine_map<(d0, d1, d2)[s0] -> (d1 * -64 + s0, 64)>(%arg14, %8, %7)[%arg4]
              %subview = memref.subview %reinterpret_cast[0, 0] [%18, %19] [1, 1] : memref<64x64xf32, strided<[?, 1], offset: ?>> to memref<?x?xf32, strided<[?, 1], offset: ?>>
              bufferization.materialize_in_destination %extracted_slice in writable %subview : (tensor<?x?xf32>, memref<?x?xf32, strided<[?, 1], offset: ?>>) -> ()
            } {tmd.mapped_to = "x"}
          } {tmd.mapped_to = "y"}
        }
      }
    }
    return
  }
  func.func @matmul_kernel__d0i1_d1i2_f2_f0_f1(%arg0: memref<*xf32> {tt.divisibility = 16 : i32}, %arg1: memref<*xf32> {tt.divisibility = 16 : i32}, %arg2: memref<*xf32> {tt.divisibility = 16 : i32}, %arg3: index {tt.divisibility = 16 : i32}, %arg4: index {tt.divisibility = 16 : i32}, %arg5: index {tt.divisibility = 16 : i32}, %arg6: index {tt.divisibility = 16 : i32}, %arg7: index {tt.divisibility = 16 : i32}, %arg8: index {tt.divisibility = 16 : i32}, %arg9: index, %arg10: index, %arg11: index) {
    affine.for %arg12 = 0 to affine_map<(d0, d1, d2) -> (d2 ceildiv 8)>(%arg9, %arg10, %arg11) {
      affine.for %arg13 = 0 to affine_map<(d0, d1, d2) -> (d0)>(%arg9, %arg10, %arg11) {
        affine.for %arg14 = 0 to affine_map<(d0, d1, d2) -> (d1 ceildiv 8)>(%arg9, %arg10, %arg11) {
          affine.parallel (%arg15) = (0) to (8) {
            %7 = affine.apply affine_map<(d0, d1) -> (d0 * 8 + d1)>(%arg12, %arg15)
            affine.parallel (%arg16) = (0) to (8) {
              %8 = affine.apply affine_map<(d0, d1) -> (d0 * 8 + d1)>(%arg14, %arg16)
              %cst = arith.constant 0.000000e+00 : f32
              %9 = tensor.empty() : tensor<64x64xf32>
              %10 = linalg.fill ins(%cst : f32) outs(%9 : tensor<64x64xf32>) -> tensor<64x64xf32>
              %c64 = arith.constant 64 : index
              %11 = arith.muli %arg13, %c64 : index
              %12 = affine.apply affine_map<()[s0] -> ((s0 + 31) floordiv 32)>()[%arg5]
              %c0 = arith.constant 0 : index
              %c1 = arith.constant 1 : index
              %13 = scf.for %arg17 = %c0 to %12 step %c1 iter_args(%arg18 = %10) -> (tensor<64x64xf32>) {
                %c32 = arith.constant 32 : index
                %20 = arith.muli %arg17, %c32 : index
                %21 = arith.muli %11, %arg6 : index
                %22 = affine.apply affine_map<(d0)[s0] -> (d0 * 32 + s0)>(%arg17)[%21]
                %reinterpret_cast_0 = memref.reinterpret_cast %arg0 to offset: [%22], sizes: [64, 32], strides: [%arg6, 1] {tmd.reuse = {sequential = [{depth = 5 : i64, iterator = "%arg17", reuse_type = "no_reuse", volume = 0 : i64}], spatial = [{depth = 3 : i64, iterator = "%arg15", mapped_to = "y", reuse_type = "total_reuse", volume = 8192 : i64}, {depth = 4 : i64, iterator = "%arg16", mapped_to = "x", reuse_type = "total_reuse", volume = 8192 : i64}], temporal = [{depth = 0 : i64, iterator = "%arg12", reuse_type = "total_reuse", volume = 8192 : i64}, {depth = 1 : i64, iterator = "%arg13", reuse_type = "no_reuse", volume = 0 : i64}, {depth = 2 : i64, iterator = "%arg14", reuse_type = "total_reuse", volume = 8192 : i64}]}} : memref<*xf32> to memref<64x32xf32, strided<[?, 1], offset: ?>>
                %alloc = memref.alloc() : memref<64x32xf32>
                memref.copy %reinterpret_cast_0, %alloc : memref<64x32xf32, strided<[?, 1], offset: ?>> to memref<64x32xf32>
                %23 = bufferization.to_tensor %alloc restrict writable : memref<64x32xf32> to tensor<64x32xf32>
                %24 = arith.muli %20, %arg7 : index
                %25 = affine.apply affine_map<(d0, d1)[s0] -> (d0 * 512 + d1 * 64 + s0)>(%arg14, %arg16)[%24]
                %reinterpret_cast_1 = memref.reinterpret_cast %arg1 to offset: [%25], sizes: [32, 64], strides: [%arg7, 1] {tmd.reuse = {sequential = [{depth = 5 : i64, iterator = "%arg17", reuse_type = "no_reuse", volume = 0 : i64}], spatial = [{depth = 3 : i64, iterator = "%arg15", mapped_to = "y", reuse_type = "total_reuse", volume = 8192 : i64}, {depth = 4 : i64, iterator = "%arg16", mapped_to = "x", reuse_type = "no_reuse", volume = 0 : i64}], temporal = [{depth = 0 : i64, iterator = "%arg12", reuse_type = "total_reuse", volume = 8192 : i64}, {depth = 1 : i64, iterator = "%arg13", reuse_type = "total_reuse", volume = 8192 : i64}, {depth = 2 : i64, iterator = "%arg14", reuse_type = "no_reuse", volume = 0 : i64}]}} : memref<*xf32> to memref<32x64xf32, strided<[?, 1], offset: ?>>
                %alloc_2 = memref.alloc() : memref<32x64xf32>
                memref.copy %reinterpret_cast_1, %alloc_2 : memref<32x64xf32, strided<[?, 1], offset: ?>> to memref<32x64xf32>
                %26 = bufferization.to_tensor %alloc_2 restrict writable : memref<32x64xf32> to tensor<32x64xf32>
                %27 = linalg.matmul ins(%23, %26 : tensor<64x32xf32>, tensor<32x64xf32>) outs(%10 : tensor<64x64xf32>) -> tensor<64x64xf32>
                %28 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%arg18, %27 : tensor<64x64xf32>, tensor<64x64xf32>) outs(%arg18 : tensor<64x64xf32>) {
                ^bb0(%in: f32, %in_3: f32, %out: f32):
                  %29 = arith.addf %in, %in_3 : f32
                  linalg.yield %29 : f32
                } -> tensor<64x64xf32>
                scf.yield %28 : tensor<64x64xf32>
              }
              %14 = arith.muli %11, %arg8 : index
              %15 = affine.apply affine_map<(d0, d1)[s0] -> (d0 * 512 + d1 * 64 + s0)>(%arg14, %arg16)[%14]
              %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%15], sizes: [64, 64], strides: [%arg8, 1] {tmd.reuse = {spatial = [{depth = 3 : i64, iterator = "%arg15", mapped_to = "y", reuse_type = "total_reuse", volume = 16384 : i64}, {depth = 4 : i64, iterator = "%arg16", mapped_to = "x", reuse_type = "no_reuse", volume = 0 : i64}], temporal = [{depth = 0 : i64, iterator = "%arg12", reuse_type = "total_reuse", volume = 16384 : i64}, {depth = 1 : i64, iterator = "%arg13", reuse_type = "no_reuse", volume = 0 : i64}, {depth = 2 : i64, iterator = "%arg14", reuse_type = "no_reuse", volume = 0 : i64}]}} : memref<*xf32> to memref<64x64xf32, strided<[?, 1], offset: ?>>
              %16 = affine.min affine_map<(d0, d1, d2)[s0] -> (d0 * -64 + s0, 64)>(%arg13, %8, %7)[%arg3]
              %17 = affine.min affine_map<(d0, d1, d2)[s0] -> (d1 * -64 + s0, 64)>(%arg13, %8, %7)[%arg4]
              %extracted_slice = tensor.extract_slice %13[0, 0] [%16, %17] [1, 1] : tensor<64x64xf32> to tensor<?x?xf32>
              %18 = affine.min affine_map<(d0, d1, d2)[s0] -> (d0 * -64 + s0, 64)>(%arg13, %8, %7)[%arg3]
              %19 = affine.min affine_map<(d0, d1, d2)[s0] -> (d1 * -64 + s0, 64)>(%arg13, %8, %7)[%arg4]
              %subview = memref.subview %reinterpret_cast[0, 0] [%18, %19] [1, 1] : memref<64x64xf32, strided<[?, 1], offset: ?>> to memref<?x?xf32, strided<[?, 1], offset: ?>>
              bufferization.materialize_in_destination %extracted_slice in writable %subview : (tensor<?x?xf32>, memref<?x?xf32, strided<[?, 1], offset: ?>>) -> ()
            } {tmd.mapped_to = "x"}
          } {tmd.mapped_to = "y"}
        }
      }
    }
    return
  }
  func.func @matmul_kernel__d0i1_d1i2_f2_f1_f0(%arg0: memref<*xf32> {tt.divisibility = 16 : i32}, %arg1: memref<*xf32> {tt.divisibility = 16 : i32}, %arg2: memref<*xf32> {tt.divisibility = 16 : i32}, %arg3: index {tt.divisibility = 16 : i32}, %arg4: index {tt.divisibility = 16 : i32}, %arg5: index {tt.divisibility = 16 : i32}, %arg6: index {tt.divisibility = 16 : i32}, %arg7: index {tt.divisibility = 16 : i32}, %arg8: index {tt.divisibility = 16 : i32}, %arg9: index, %arg10: index, %arg11: index) {
    affine.for %arg12 = 0 to affine_map<(d0, d1, d2) -> (d2 ceildiv 8)>(%arg9, %arg10, %arg11) {
      affine.for %arg13 = 0 to affine_map<(d0, d1, d2) -> (d1 ceildiv 8)>(%arg9, %arg10, %arg11) {
        affine.for %arg14 = 0 to affine_map<(d0, d1, d2) -> (d0)>(%arg9, %arg10, %arg11) {
          affine.parallel (%arg15) = (0) to (8) {
            %7 = affine.apply affine_map<(d0, d1) -> (d0 * 8 + d1)>(%arg12, %arg15)
            affine.parallel (%arg16) = (0) to (8) {
              %8 = affine.apply affine_map<(d0, d1) -> (d0 * 8 + d1)>(%arg13, %arg16)
              %cst = arith.constant 0.000000e+00 : f32
              %9 = tensor.empty() : tensor<64x64xf32>
              %10 = linalg.fill ins(%cst : f32) outs(%9 : tensor<64x64xf32>) -> tensor<64x64xf32>
              %c64 = arith.constant 64 : index
              %11 = arith.muli %arg14, %c64 : index
              %12 = affine.apply affine_map<()[s0] -> ((s0 + 31) floordiv 32)>()[%arg5]
              %c0 = arith.constant 0 : index
              %c1 = arith.constant 1 : index
              %13 = scf.for %arg17 = %c0 to %12 step %c1 iter_args(%arg18 = %10) -> (tensor<64x64xf32>) {
                %c32 = arith.constant 32 : index
                %20 = arith.muli %arg17, %c32 : index
                %21 = arith.muli %11, %arg6 : index
                %22 = affine.apply affine_map<(d0)[s0] -> (d0 * 32 + s0)>(%arg17)[%21]
                %reinterpret_cast_0 = memref.reinterpret_cast %arg0 to offset: [%22], sizes: [64, 32], strides: [%arg6, 1] {tmd.reuse = {sequential = [{depth = 5 : i64, iterator = "%arg17", reuse_type = "no_reuse", volume = 0 : i64}], spatial = [{depth = 3 : i64, iterator = "%arg15", mapped_to = "y", reuse_type = "total_reuse", volume = 8192 : i64}, {depth = 4 : i64, iterator = "%arg16", mapped_to = "x", reuse_type = "total_reuse", volume = 8192 : i64}], temporal = [{depth = 0 : i64, iterator = "%arg12", reuse_type = "total_reuse", volume = 8192 : i64}, {depth = 1 : i64, iterator = "%arg13", reuse_type = "total_reuse", volume = 8192 : i64}, {depth = 2 : i64, iterator = "%arg14", reuse_type = "no_reuse", volume = 0 : i64}]}} : memref<*xf32> to memref<64x32xf32, strided<[?, 1], offset: ?>>
                %alloc = memref.alloc() : memref<64x32xf32>
                memref.copy %reinterpret_cast_0, %alloc : memref<64x32xf32, strided<[?, 1], offset: ?>> to memref<64x32xf32>
                %23 = bufferization.to_tensor %alloc restrict writable : memref<64x32xf32> to tensor<64x32xf32>
                %24 = arith.muli %20, %arg7 : index
                %25 = affine.apply affine_map<(d0, d1)[s0] -> (d0 * 512 + d1 * 64 + s0)>(%arg13, %arg16)[%24]
                %reinterpret_cast_1 = memref.reinterpret_cast %arg1 to offset: [%25], sizes: [32, 64], strides: [%arg7, 1] {tmd.reuse = {sequential = [{depth = 5 : i64, iterator = "%arg17", reuse_type = "no_reuse", volume = 0 : i64}], spatial = [{depth = 3 : i64, iterator = "%arg15", mapped_to = "y", reuse_type = "total_reuse", volume = 8192 : i64}, {depth = 4 : i64, iterator = "%arg16", mapped_to = "x", reuse_type = "no_reuse", volume = 0 : i64}], temporal = [{depth = 0 : i64, iterator = "%arg12", reuse_type = "total_reuse", volume = 8192 : i64}, {depth = 1 : i64, iterator = "%arg13", reuse_type = "no_reuse", volume = 0 : i64}, {depth = 2 : i64, iterator = "%arg14", reuse_type = "total_reuse", volume = 8192 : i64}]}} : memref<*xf32> to memref<32x64xf32, strided<[?, 1], offset: ?>>
                %alloc_2 = memref.alloc() : memref<32x64xf32>
                memref.copy %reinterpret_cast_1, %alloc_2 : memref<32x64xf32, strided<[?, 1], offset: ?>> to memref<32x64xf32>
                %26 = bufferization.to_tensor %alloc_2 restrict writable : memref<32x64xf32> to tensor<32x64xf32>
                %27 = linalg.matmul ins(%23, %26 : tensor<64x32xf32>, tensor<32x64xf32>) outs(%10 : tensor<64x64xf32>) -> tensor<64x64xf32>
                %28 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%arg18, %27 : tensor<64x64xf32>, tensor<64x64xf32>) outs(%arg18 : tensor<64x64xf32>) {
                ^bb0(%in: f32, %in_3: f32, %out: f32):
                  %29 = arith.addf %in, %in_3 : f32
                  linalg.yield %29 : f32
                } -> tensor<64x64xf32>
                scf.yield %28 : tensor<64x64xf32>
              }
              %14 = arith.muli %11, %arg8 : index
              %15 = affine.apply affine_map<(d0, d1)[s0] -> (d0 * 512 + d1 * 64 + s0)>(%arg13, %arg16)[%14]
              %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%15], sizes: [64, 64], strides: [%arg8, 1] {tmd.reuse = {spatial = [{depth = 3 : i64, iterator = "%arg15", mapped_to = "y", reuse_type = "total_reuse", volume = 16384 : i64}, {depth = 4 : i64, iterator = "%arg16", mapped_to = "x", reuse_type = "no_reuse", volume = 0 : i64}], temporal = [{depth = 0 : i64, iterator = "%arg12", reuse_type = "total_reuse", volume = 16384 : i64}, {depth = 1 : i64, iterator = "%arg13", reuse_type = "no_reuse", volume = 0 : i64}, {depth = 2 : i64, iterator = "%arg14", reuse_type = "no_reuse", volume = 0 : i64}]}} : memref<*xf32> to memref<64x64xf32, strided<[?, 1], offset: ?>>
              %16 = affine.min affine_map<(d0, d1, d2)[s0] -> (d0 * -64 + s0, 64)>(%arg14, %8, %7)[%arg3]
              %17 = affine.min affine_map<(d0, d1, d2)[s0] -> (d1 * -64 + s0, 64)>(%arg14, %8, %7)[%arg4]
              %extracted_slice = tensor.extract_slice %13[0, 0] [%16, %17] [1, 1] : tensor<64x64xf32> to tensor<?x?xf32>
              %18 = affine.min affine_map<(d0, d1, d2)[s0] -> (d0 * -64 + s0, 64)>(%arg14, %8, %7)[%arg3]
              %19 = affine.min affine_map<(d0, d1, d2)[s0] -> (d1 * -64 + s0, 64)>(%arg14, %8, %7)[%arg4]
              %subview = memref.subview %reinterpret_cast[0, 0] [%18, %19] [1, 1] : memref<64x64xf32, strided<[?, 1], offset: ?>> to memref<?x?xf32, strided<[?, 1], offset: ?>>
              bufferization.materialize_in_destination %extracted_slice in writable %subview : (tensor<?x?xf32>, memref<?x?xf32, strided<[?, 1], offset: ?>>) -> ()
            } {tmd.mapped_to = "x"}
          } {tmd.mapped_to = "y"}
        }
      }
    }
    return
  }
  func.func @matmul_kernel__d1i0_d0i2_f0_f1_f2(%arg0: memref<*xf32> {tt.divisibility = 16 : i32}, %arg1: memref<*xf32> {tt.divisibility = 16 : i32}, %arg2: memref<*xf32> {tt.divisibility = 16 : i32}, %arg3: index {tt.divisibility = 16 : i32}, %arg4: index {tt.divisibility = 16 : i32}, %arg5: index {tt.divisibility = 16 : i32}, %arg6: index {tt.divisibility = 16 : i32}, %arg7: index {tt.divisibility = 16 : i32}, %arg8: index {tt.divisibility = 16 : i32}, %arg9: index, %arg10: index, %arg11: index) {
    affine.for %arg12 = 0 to affine_map<(d0, d1, d2) -> (d0 ceildiv 8)>(%arg9, %arg10, %arg11) {
      affine.for %arg13 = 0 to affine_map<(d0, d1, d2) -> (d1)>(%arg9, %arg10, %arg11) {
        affine.for %arg14 = 0 to affine_map<(d0, d1, d2) -> (d2 ceildiv 8)>(%arg9, %arg10, %arg11) {
          affine.parallel (%arg15) = (0) to (8) {
            %7 = affine.apply affine_map<(d0, d1) -> (d0 * 8 + d1)>(%arg14, %arg15)
            affine.parallel (%arg16) = (0) to (8) {
              %8 = affine.apply affine_map<(d0, d1) -> (d0 * 8 + d1)>(%arg12, %arg16)
              %cst = arith.constant 0.000000e+00 : f32
              %9 = tensor.empty() : tensor<64x64xf32>
              %10 = linalg.fill ins(%cst : f32) outs(%9 : tensor<64x64xf32>) -> tensor<64x64xf32>
              %c64 = arith.constant 64 : index
              %11 = arith.muli %8, %c64 : index
              %12 = affine.apply affine_map<()[s0] -> ((s0 + 31) floordiv 32)>()[%arg5]
              %c0 = arith.constant 0 : index
              %c1 = arith.constant 1 : index
              %13 = scf.for %arg17 = %c0 to %12 step %c1 iter_args(%arg18 = %10) -> (tensor<64x64xf32>) {
                %c32 = arith.constant 32 : index
                %20 = arith.muli %arg17, %c32 : index
                %21 = arith.muli %11, %arg6 : index
                %22 = affine.apply affine_map<(d0)[s0] -> (d0 * 32 + s0)>(%arg17)[%21]
                %reinterpret_cast_0 = memref.reinterpret_cast %arg0 to offset: [%22], sizes: [64, 32], strides: [%arg6, 1] {tmd.reuse = {sequential = [{depth = 5 : i64, iterator = "%arg17", reuse_type = "no_reuse", volume = 0 : i64}], spatial = [{depth = 3 : i64, iterator = "%arg15", mapped_to = "x", reuse_type = "total_reuse", volume = 8192 : i64}, {depth = 4 : i64, iterator = "%arg16", mapped_to = "y", reuse_type = "no_reuse", volume = 0 : i64}], temporal = [{depth = 0 : i64, iterator = "%arg12", reuse_type = "no_reuse", volume = 0 : i64}, {depth = 1 : i64, iterator = "%arg13", reuse_type = "total_reuse", volume = 8192 : i64}, {depth = 2 : i64, iterator = "%arg14", reuse_type = "total_reuse", volume = 8192 : i64}]}} : memref<*xf32> to memref<64x32xf32, strided<[?, 1], offset: ?>>
                %alloc = memref.alloc() : memref<64x32xf32>
                memref.copy %reinterpret_cast_0, %alloc : memref<64x32xf32, strided<[?, 1], offset: ?>> to memref<64x32xf32>
                %23 = bufferization.to_tensor %alloc restrict writable : memref<64x32xf32> to tensor<64x32xf32>
                %24 = arith.muli %20, %arg7 : index
                %25 = affine.apply affine_map<(d0)[s0] -> (d0 * 64 + s0)>(%arg13)[%24]
                %reinterpret_cast_1 = memref.reinterpret_cast %arg1 to offset: [%25], sizes: [32, 64], strides: [%arg7, 1] {tmd.reuse = {sequential = [{depth = 5 : i64, iterator = "%arg17", reuse_type = "no_reuse", volume = 0 : i64}], spatial = [{depth = 3 : i64, iterator = "%arg15", mapped_to = "x", reuse_type = "total_reuse", volume = 8192 : i64}, {depth = 4 : i64, iterator = "%arg16", mapped_to = "y", reuse_type = "total_reuse", volume = 8192 : i64}], temporal = [{depth = 0 : i64, iterator = "%arg12", reuse_type = "total_reuse", volume = 8192 : i64}, {depth = 1 : i64, iterator = "%arg13", reuse_type = "no_reuse", volume = 0 : i64}, {depth = 2 : i64, iterator = "%arg14", reuse_type = "total_reuse", volume = 8192 : i64}]}} : memref<*xf32> to memref<32x64xf32, strided<[?, 1], offset: ?>>
                %alloc_2 = memref.alloc() : memref<32x64xf32>
                memref.copy %reinterpret_cast_1, %alloc_2 : memref<32x64xf32, strided<[?, 1], offset: ?>> to memref<32x64xf32>
                %26 = bufferization.to_tensor %alloc_2 restrict writable : memref<32x64xf32> to tensor<32x64xf32>
                %27 = linalg.matmul ins(%23, %26 : tensor<64x32xf32>, tensor<32x64xf32>) outs(%10 : tensor<64x64xf32>) -> tensor<64x64xf32>
                %28 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%arg18, %27 : tensor<64x64xf32>, tensor<64x64xf32>) outs(%arg18 : tensor<64x64xf32>) {
                ^bb0(%in: f32, %in_3: f32, %out: f32):
                  %29 = arith.addf %in, %in_3 : f32
                  linalg.yield %29 : f32
                } -> tensor<64x64xf32>
                scf.yield %28 : tensor<64x64xf32>
              }
              %14 = arith.muli %11, %arg8 : index
              %15 = affine.apply affine_map<(d0)[s0] -> (d0 * 64 + s0)>(%arg13)[%14]
              %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%15], sizes: [64, 64], strides: [%arg8, 1] {tmd.reuse = {spatial = [{depth = 3 : i64, iterator = "%arg15", mapped_to = "x", reuse_type = "total_reuse", volume = 16384 : i64}, {depth = 4 : i64, iterator = "%arg16", mapped_to = "y", reuse_type = "no_reuse", volume = 0 : i64}], temporal = [{depth = 0 : i64, iterator = "%arg12", reuse_type = "no_reuse", volume = 0 : i64}, {depth = 1 : i64, iterator = "%arg13", reuse_type = "no_reuse", volume = 0 : i64}, {depth = 2 : i64, iterator = "%arg14", reuse_type = "total_reuse", volume = 16384 : i64}]}} : memref<*xf32> to memref<64x64xf32, strided<[?, 1], offset: ?>>
              %16 = affine.min affine_map<(d0, d1, d2)[s0] -> (d0 * -64 + s0, 64)>(%8, %arg13, %7)[%arg3]
              %17 = affine.min affine_map<(d0, d1, d2)[s0] -> (d1 * -64 + s0, 64)>(%8, %arg13, %7)[%arg4]
              %extracted_slice = tensor.extract_slice %13[0, 0] [%16, %17] [1, 1] : tensor<64x64xf32> to tensor<?x?xf32>
              %18 = affine.min affine_map<(d0, d1, d2)[s0] -> (d0 * -64 + s0, 64)>(%8, %arg13, %7)[%arg3]
              %19 = affine.min affine_map<(d0, d1, d2)[s0] -> (d1 * -64 + s0, 64)>(%8, %arg13, %7)[%arg4]
              %subview = memref.subview %reinterpret_cast[0, 0] [%18, %19] [1, 1] : memref<64x64xf32, strided<[?, 1], offset: ?>> to memref<?x?xf32, strided<[?, 1], offset: ?>>
              bufferization.materialize_in_destination %extracted_slice in writable %subview : (tensor<?x?xf32>, memref<?x?xf32, strided<[?, 1], offset: ?>>) -> ()
            } {tmd.mapped_to = "y"}
          } {tmd.mapped_to = "x"}
        }
      }
    }
    return
  }
  func.func @matmul_kernel__d1i0_d0i2_f0_f2_f1(%arg0: memref<*xf32> {tt.divisibility = 16 : i32}, %arg1: memref<*xf32> {tt.divisibility = 16 : i32}, %arg2: memref<*xf32> {tt.divisibility = 16 : i32}, %arg3: index {tt.divisibility = 16 : i32}, %arg4: index {tt.divisibility = 16 : i32}, %arg5: index {tt.divisibility = 16 : i32}, %arg6: index {tt.divisibility = 16 : i32}, %arg7: index {tt.divisibility = 16 : i32}, %arg8: index {tt.divisibility = 16 : i32}, %arg9: index, %arg10: index, %arg11: index) {
    affine.for %arg12 = 0 to affine_map<(d0, d1, d2) -> (d0 ceildiv 8)>(%arg9, %arg10, %arg11) {
      affine.for %arg13 = 0 to affine_map<(d0, d1, d2) -> (d2 ceildiv 8)>(%arg9, %arg10, %arg11) {
        affine.for %arg14 = 0 to affine_map<(d0, d1, d2) -> (d1)>(%arg9, %arg10, %arg11) {
          affine.parallel (%arg15) = (0) to (8) {
            %7 = affine.apply affine_map<(d0, d1) -> (d0 * 8 + d1)>(%arg13, %arg15)
            affine.parallel (%arg16) = (0) to (8) {
              %8 = affine.apply affine_map<(d0, d1) -> (d0 * 8 + d1)>(%arg12, %arg16)
              %cst = arith.constant 0.000000e+00 : f32
              %9 = tensor.empty() : tensor<64x64xf32>
              %10 = linalg.fill ins(%cst : f32) outs(%9 : tensor<64x64xf32>) -> tensor<64x64xf32>
              %c64 = arith.constant 64 : index
              %11 = arith.muli %8, %c64 : index
              %12 = affine.apply affine_map<()[s0] -> ((s0 + 31) floordiv 32)>()[%arg5]
              %c0 = arith.constant 0 : index
              %c1 = arith.constant 1 : index
              %13 = scf.for %arg17 = %c0 to %12 step %c1 iter_args(%arg18 = %10) -> (tensor<64x64xf32>) {
                %c32 = arith.constant 32 : index
                %20 = arith.muli %arg17, %c32 : index
                %21 = arith.muli %11, %arg6 : index
                %22 = affine.apply affine_map<(d0)[s0] -> (d0 * 32 + s0)>(%arg17)[%21]
                %reinterpret_cast_0 = memref.reinterpret_cast %arg0 to offset: [%22], sizes: [64, 32], strides: [%arg6, 1] {tmd.reuse = {sequential = [{depth = 5 : i64, iterator = "%arg17", reuse_type = "no_reuse", volume = 0 : i64}], spatial = [{depth = 3 : i64, iterator = "%arg15", mapped_to = "x", reuse_type = "total_reuse", volume = 8192 : i64}, {depth = 4 : i64, iterator = "%arg16", mapped_to = "y", reuse_type = "no_reuse", volume = 0 : i64}], temporal = [{depth = 0 : i64, iterator = "%arg12", reuse_type = "no_reuse", volume = 0 : i64}, {depth = 1 : i64, iterator = "%arg13", reuse_type = "total_reuse", volume = 8192 : i64}, {depth = 2 : i64, iterator = "%arg14", reuse_type = "total_reuse", volume = 8192 : i64}]}} : memref<*xf32> to memref<64x32xf32, strided<[?, 1], offset: ?>>
                %alloc = memref.alloc() : memref<64x32xf32>
                memref.copy %reinterpret_cast_0, %alloc : memref<64x32xf32, strided<[?, 1], offset: ?>> to memref<64x32xf32>
                %23 = bufferization.to_tensor %alloc restrict writable : memref<64x32xf32> to tensor<64x32xf32>
                %24 = arith.muli %20, %arg7 : index
                %25 = affine.apply affine_map<(d0)[s0] -> (d0 * 64 + s0)>(%arg14)[%24]
                %reinterpret_cast_1 = memref.reinterpret_cast %arg1 to offset: [%25], sizes: [32, 64], strides: [%arg7, 1] {tmd.reuse = {sequential = [{depth = 5 : i64, iterator = "%arg17", reuse_type = "no_reuse", volume = 0 : i64}], spatial = [{depth = 3 : i64, iterator = "%arg15", mapped_to = "x", reuse_type = "total_reuse", volume = 8192 : i64}, {depth = 4 : i64, iterator = "%arg16", mapped_to = "y", reuse_type = "total_reuse", volume = 8192 : i64}], temporal = [{depth = 0 : i64, iterator = "%arg12", reuse_type = "total_reuse", volume = 8192 : i64}, {depth = 1 : i64, iterator = "%arg13", reuse_type = "total_reuse", volume = 8192 : i64}, {depth = 2 : i64, iterator = "%arg14", reuse_type = "no_reuse", volume = 0 : i64}]}} : memref<*xf32> to memref<32x64xf32, strided<[?, 1], offset: ?>>
                %alloc_2 = memref.alloc() : memref<32x64xf32>
                memref.copy %reinterpret_cast_1, %alloc_2 : memref<32x64xf32, strided<[?, 1], offset: ?>> to memref<32x64xf32>
                %26 = bufferization.to_tensor %alloc_2 restrict writable : memref<32x64xf32> to tensor<32x64xf32>
                %27 = linalg.matmul ins(%23, %26 : tensor<64x32xf32>, tensor<32x64xf32>) outs(%10 : tensor<64x64xf32>) -> tensor<64x64xf32>
                %28 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%arg18, %27 : tensor<64x64xf32>, tensor<64x64xf32>) outs(%arg18 : tensor<64x64xf32>) {
                ^bb0(%in: f32, %in_3: f32, %out: f32):
                  %29 = arith.addf %in, %in_3 : f32
                  linalg.yield %29 : f32
                } -> tensor<64x64xf32>
                scf.yield %28 : tensor<64x64xf32>
              }
              %14 = arith.muli %11, %arg8 : index
              %15 = affine.apply affine_map<(d0)[s0] -> (d0 * 64 + s0)>(%arg14)[%14]
              %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%15], sizes: [64, 64], strides: [%arg8, 1] {tmd.reuse = {spatial = [{depth = 3 : i64, iterator = "%arg15", mapped_to = "x", reuse_type = "total_reuse", volume = 16384 : i64}, {depth = 4 : i64, iterator = "%arg16", mapped_to = "y", reuse_type = "no_reuse", volume = 0 : i64}], temporal = [{depth = 0 : i64, iterator = "%arg12", reuse_type = "no_reuse", volume = 0 : i64}, {depth = 1 : i64, iterator = "%arg13", reuse_type = "total_reuse", volume = 16384 : i64}, {depth = 2 : i64, iterator = "%arg14", reuse_type = "no_reuse", volume = 0 : i64}]}} : memref<*xf32> to memref<64x64xf32, strided<[?, 1], offset: ?>>
              %16 = affine.min affine_map<(d0, d1, d2)[s0] -> (d0 * -64 + s0, 64)>(%8, %arg14, %7)[%arg3]
              %17 = affine.min affine_map<(d0, d1, d2)[s0] -> (d1 * -64 + s0, 64)>(%8, %arg14, %7)[%arg4]
              %extracted_slice = tensor.extract_slice %13[0, 0] [%16, %17] [1, 1] : tensor<64x64xf32> to tensor<?x?xf32>
              %18 = affine.min affine_map<(d0, d1, d2)[s0] -> (d0 * -64 + s0, 64)>(%8, %arg14, %7)[%arg3]
              %19 = affine.min affine_map<(d0, d1, d2)[s0] -> (d1 * -64 + s0, 64)>(%8, %arg14, %7)[%arg4]
              %subview = memref.subview %reinterpret_cast[0, 0] [%18, %19] [1, 1] : memref<64x64xf32, strided<[?, 1], offset: ?>> to memref<?x?xf32, strided<[?, 1], offset: ?>>
              bufferization.materialize_in_destination %extracted_slice in writable %subview : (tensor<?x?xf32>, memref<?x?xf32, strided<[?, 1], offset: ?>>) -> ()
            } {tmd.mapped_to = "y"}
          } {tmd.mapped_to = "x"}
        }
      }
    }
    return
  }
  func.func @matmul_kernel__d1i0_d0i2_f1_f0_f2(%arg0: memref<*xf32> {tt.divisibility = 16 : i32}, %arg1: memref<*xf32> {tt.divisibility = 16 : i32}, %arg2: memref<*xf32> {tt.divisibility = 16 : i32}, %arg3: index {tt.divisibility = 16 : i32}, %arg4: index {tt.divisibility = 16 : i32}, %arg5: index {tt.divisibility = 16 : i32}, %arg6: index {tt.divisibility = 16 : i32}, %arg7: index {tt.divisibility = 16 : i32}, %arg8: index {tt.divisibility = 16 : i32}, %arg9: index, %arg10: index, %arg11: index) {
    affine.for %arg12 = 0 to affine_map<(d0, d1, d2) -> (d1)>(%arg9, %arg10, %arg11) {
      affine.for %arg13 = 0 to affine_map<(d0, d1, d2) -> (d0 ceildiv 8)>(%arg9, %arg10, %arg11) {
        affine.for %arg14 = 0 to affine_map<(d0, d1, d2) -> (d2 ceildiv 8)>(%arg9, %arg10, %arg11) {
          affine.parallel (%arg15) = (0) to (8) {
            %7 = affine.apply affine_map<(d0, d1) -> (d0 * 8 + d1)>(%arg14, %arg15)
            affine.parallel (%arg16) = (0) to (8) {
              %8 = affine.apply affine_map<(d0, d1) -> (d0 * 8 + d1)>(%arg13, %arg16)
              %cst = arith.constant 0.000000e+00 : f32
              %9 = tensor.empty() : tensor<64x64xf32>
              %10 = linalg.fill ins(%cst : f32) outs(%9 : tensor<64x64xf32>) -> tensor<64x64xf32>
              %c64 = arith.constant 64 : index
              %11 = arith.muli %8, %c64 : index
              %12 = affine.apply affine_map<()[s0] -> ((s0 + 31) floordiv 32)>()[%arg5]
              %c0 = arith.constant 0 : index
              %c1 = arith.constant 1 : index
              %13 = scf.for %arg17 = %c0 to %12 step %c1 iter_args(%arg18 = %10) -> (tensor<64x64xf32>) {
                %c32 = arith.constant 32 : index
                %20 = arith.muli %arg17, %c32 : index
                %21 = arith.muli %11, %arg6 : index
                %22 = affine.apply affine_map<(d0)[s0] -> (d0 * 32 + s0)>(%arg17)[%21]
                %reinterpret_cast_0 = memref.reinterpret_cast %arg0 to offset: [%22], sizes: [64, 32], strides: [%arg6, 1] {tmd.reuse = {sequential = [{depth = 5 : i64, iterator = "%arg17", reuse_type = "no_reuse", volume = 0 : i64}], spatial = [{depth = 3 : i64, iterator = "%arg15", mapped_to = "x", reuse_type = "total_reuse", volume = 8192 : i64}, {depth = 4 : i64, iterator = "%arg16", mapped_to = "y", reuse_type = "no_reuse", volume = 0 : i64}], temporal = [{depth = 0 : i64, iterator = "%arg12", reuse_type = "total_reuse", volume = 8192 : i64}, {depth = 1 : i64, iterator = "%arg13", reuse_type = "no_reuse", volume = 0 : i64}, {depth = 2 : i64, iterator = "%arg14", reuse_type = "total_reuse", volume = 8192 : i64}]}} : memref<*xf32> to memref<64x32xf32, strided<[?, 1], offset: ?>>
                %alloc = memref.alloc() : memref<64x32xf32>
                memref.copy %reinterpret_cast_0, %alloc : memref<64x32xf32, strided<[?, 1], offset: ?>> to memref<64x32xf32>
                %23 = bufferization.to_tensor %alloc restrict writable : memref<64x32xf32> to tensor<64x32xf32>
                %24 = arith.muli %20, %arg7 : index
                %25 = affine.apply affine_map<(d0)[s0] -> (d0 * 64 + s0)>(%arg12)[%24]
                %reinterpret_cast_1 = memref.reinterpret_cast %arg1 to offset: [%25], sizes: [32, 64], strides: [%arg7, 1] {tmd.reuse = {sequential = [{depth = 5 : i64, iterator = "%arg17", reuse_type = "no_reuse", volume = 0 : i64}], spatial = [{depth = 3 : i64, iterator = "%arg15", mapped_to = "x", reuse_type = "total_reuse", volume = 8192 : i64}, {depth = 4 : i64, iterator = "%arg16", mapped_to = "y", reuse_type = "total_reuse", volume = 8192 : i64}], temporal = [{depth = 0 : i64, iterator = "%arg12", reuse_type = "no_reuse", volume = 0 : i64}, {depth = 1 : i64, iterator = "%arg13", reuse_type = "total_reuse", volume = 8192 : i64}, {depth = 2 : i64, iterator = "%arg14", reuse_type = "total_reuse", volume = 8192 : i64}]}} : memref<*xf32> to memref<32x64xf32, strided<[?, 1], offset: ?>>
                %alloc_2 = memref.alloc() : memref<32x64xf32>
                memref.copy %reinterpret_cast_1, %alloc_2 : memref<32x64xf32, strided<[?, 1], offset: ?>> to memref<32x64xf32>
                %26 = bufferization.to_tensor %alloc_2 restrict writable : memref<32x64xf32> to tensor<32x64xf32>
                %27 = linalg.matmul ins(%23, %26 : tensor<64x32xf32>, tensor<32x64xf32>) outs(%10 : tensor<64x64xf32>) -> tensor<64x64xf32>
                %28 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%arg18, %27 : tensor<64x64xf32>, tensor<64x64xf32>) outs(%arg18 : tensor<64x64xf32>) {
                ^bb0(%in: f32, %in_3: f32, %out: f32):
                  %29 = arith.addf %in, %in_3 : f32
                  linalg.yield %29 : f32
                } -> tensor<64x64xf32>
                scf.yield %28 : tensor<64x64xf32>
              }
              %14 = arith.muli %11, %arg8 : index
              %15 = affine.apply affine_map<(d0)[s0] -> (d0 * 64 + s0)>(%arg12)[%14]
              %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%15], sizes: [64, 64], strides: [%arg8, 1] {tmd.reuse = {spatial = [{depth = 3 : i64, iterator = "%arg15", mapped_to = "x", reuse_type = "total_reuse", volume = 16384 : i64}, {depth = 4 : i64, iterator = "%arg16", mapped_to = "y", reuse_type = "no_reuse", volume = 0 : i64}], temporal = [{depth = 0 : i64, iterator = "%arg12", reuse_type = "no_reuse", volume = 0 : i64}, {depth = 1 : i64, iterator = "%arg13", reuse_type = "no_reuse", volume = 0 : i64}, {depth = 2 : i64, iterator = "%arg14", reuse_type = "total_reuse", volume = 16384 : i64}]}} : memref<*xf32> to memref<64x64xf32, strided<[?, 1], offset: ?>>
              %16 = affine.min affine_map<(d0, d1, d2)[s0] -> (d0 * -64 + s0, 64)>(%8, %arg12, %7)[%arg3]
              %17 = affine.min affine_map<(d0, d1, d2)[s0] -> (d1 * -64 + s0, 64)>(%8, %arg12, %7)[%arg4]
              %extracted_slice = tensor.extract_slice %13[0, 0] [%16, %17] [1, 1] : tensor<64x64xf32> to tensor<?x?xf32>
              %18 = affine.min affine_map<(d0, d1, d2)[s0] -> (d0 * -64 + s0, 64)>(%8, %arg12, %7)[%arg3]
              %19 = affine.min affine_map<(d0, d1, d2)[s0] -> (d1 * -64 + s0, 64)>(%8, %arg12, %7)[%arg4]
              %subview = memref.subview %reinterpret_cast[0, 0] [%18, %19] [1, 1] : memref<64x64xf32, strided<[?, 1], offset: ?>> to memref<?x?xf32, strided<[?, 1], offset: ?>>
              bufferization.materialize_in_destination %extracted_slice in writable %subview : (tensor<?x?xf32>, memref<?x?xf32, strided<[?, 1], offset: ?>>) -> ()
            } {tmd.mapped_to = "y"}
          } {tmd.mapped_to = "x"}
        }
      }
    }
    return
  }
  func.func @matmul_kernel__d1i0_d0i2_f1_f2_f0(%arg0: memref<*xf32> {tt.divisibility = 16 : i32}, %arg1: memref<*xf32> {tt.divisibility = 16 : i32}, %arg2: memref<*xf32> {tt.divisibility = 16 : i32}, %arg3: index {tt.divisibility = 16 : i32}, %arg4: index {tt.divisibility = 16 : i32}, %arg5: index {tt.divisibility = 16 : i32}, %arg6: index {tt.divisibility = 16 : i32}, %arg7: index {tt.divisibility = 16 : i32}, %arg8: index {tt.divisibility = 16 : i32}, %arg9: index, %arg10: index, %arg11: index) {
    affine.for %arg12 = 0 to affine_map<(d0, d1, d2) -> (d1)>(%arg9, %arg10, %arg11) {
      affine.for %arg13 = 0 to affine_map<(d0, d1, d2) -> (d2 ceildiv 8)>(%arg9, %arg10, %arg11) {
        affine.for %arg14 = 0 to affine_map<(d0, d1, d2) -> (d0 ceildiv 8)>(%arg9, %arg10, %arg11) {
          affine.parallel (%arg15) = (0) to (8) {
            %7 = affine.apply affine_map<(d0, d1) -> (d0 * 8 + d1)>(%arg13, %arg15)
            affine.parallel (%arg16) = (0) to (8) {
              %8 = affine.apply affine_map<(d0, d1) -> (d0 * 8 + d1)>(%arg14, %arg16)
              %cst = arith.constant 0.000000e+00 : f32
              %9 = tensor.empty() : tensor<64x64xf32>
              %10 = linalg.fill ins(%cst : f32) outs(%9 : tensor<64x64xf32>) -> tensor<64x64xf32>
              %c64 = arith.constant 64 : index
              %11 = arith.muli %8, %c64 : index
              %12 = affine.apply affine_map<()[s0] -> ((s0 + 31) floordiv 32)>()[%arg5]
              %c0 = arith.constant 0 : index
              %c1 = arith.constant 1 : index
              %13 = scf.for %arg17 = %c0 to %12 step %c1 iter_args(%arg18 = %10) -> (tensor<64x64xf32>) {
                %c32 = arith.constant 32 : index
                %20 = arith.muli %arg17, %c32 : index
                %21 = arith.muli %11, %arg6 : index
                %22 = affine.apply affine_map<(d0)[s0] -> (d0 * 32 + s0)>(%arg17)[%21]
                %reinterpret_cast_0 = memref.reinterpret_cast %arg0 to offset: [%22], sizes: [64, 32], strides: [%arg6, 1] {tmd.reuse = {sequential = [{depth = 5 : i64, iterator = "%arg17", reuse_type = "no_reuse", volume = 0 : i64}], spatial = [{depth = 3 : i64, iterator = "%arg15", mapped_to = "x", reuse_type = "total_reuse", volume = 8192 : i64}, {depth = 4 : i64, iterator = "%arg16", mapped_to = "y", reuse_type = "no_reuse", volume = 0 : i64}], temporal = [{depth = 0 : i64, iterator = "%arg12", reuse_type = "total_reuse", volume = 8192 : i64}, {depth = 1 : i64, iterator = "%arg13", reuse_type = "total_reuse", volume = 8192 : i64}, {depth = 2 : i64, iterator = "%arg14", reuse_type = "no_reuse", volume = 0 : i64}]}} : memref<*xf32> to memref<64x32xf32, strided<[?, 1], offset: ?>>
                %alloc = memref.alloc() : memref<64x32xf32>
                memref.copy %reinterpret_cast_0, %alloc : memref<64x32xf32, strided<[?, 1], offset: ?>> to memref<64x32xf32>
                %23 = bufferization.to_tensor %alloc restrict writable : memref<64x32xf32> to tensor<64x32xf32>
                %24 = arith.muli %20, %arg7 : index
                %25 = affine.apply affine_map<(d0)[s0] -> (d0 * 64 + s0)>(%arg12)[%24]
                %reinterpret_cast_1 = memref.reinterpret_cast %arg1 to offset: [%25], sizes: [32, 64], strides: [%arg7, 1] {tmd.reuse = {sequential = [{depth = 5 : i64, iterator = "%arg17", reuse_type = "no_reuse", volume = 0 : i64}], spatial = [{depth = 3 : i64, iterator = "%arg15", mapped_to = "x", reuse_type = "total_reuse", volume = 8192 : i64}, {depth = 4 : i64, iterator = "%arg16", mapped_to = "y", reuse_type = "total_reuse", volume = 8192 : i64}], temporal = [{depth = 0 : i64, iterator = "%arg12", reuse_type = "no_reuse", volume = 0 : i64}, {depth = 1 : i64, iterator = "%arg13", reuse_type = "total_reuse", volume = 8192 : i64}, {depth = 2 : i64, iterator = "%arg14", reuse_type = "total_reuse", volume = 8192 : i64}]}} : memref<*xf32> to memref<32x64xf32, strided<[?, 1], offset: ?>>
                %alloc_2 = memref.alloc() : memref<32x64xf32>
                memref.copy %reinterpret_cast_1, %alloc_2 : memref<32x64xf32, strided<[?, 1], offset: ?>> to memref<32x64xf32>
                %26 = bufferization.to_tensor %alloc_2 restrict writable : memref<32x64xf32> to tensor<32x64xf32>
                %27 = linalg.matmul ins(%23, %26 : tensor<64x32xf32>, tensor<32x64xf32>) outs(%10 : tensor<64x64xf32>) -> tensor<64x64xf32>
                %28 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%arg18, %27 : tensor<64x64xf32>, tensor<64x64xf32>) outs(%arg18 : tensor<64x64xf32>) {
                ^bb0(%in: f32, %in_3: f32, %out: f32):
                  %29 = arith.addf %in, %in_3 : f32
                  linalg.yield %29 : f32
                } -> tensor<64x64xf32>
                scf.yield %28 : tensor<64x64xf32>
              }
              %14 = arith.muli %11, %arg8 : index
              %15 = affine.apply affine_map<(d0)[s0] -> (d0 * 64 + s0)>(%arg12)[%14]
              %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%15], sizes: [64, 64], strides: [%arg8, 1] {tmd.reuse = {spatial = [{depth = 3 : i64, iterator = "%arg15", mapped_to = "x", reuse_type = "total_reuse", volume = 16384 : i64}, {depth = 4 : i64, iterator = "%arg16", mapped_to = "y", reuse_type = "no_reuse", volume = 0 : i64}], temporal = [{depth = 0 : i64, iterator = "%arg12", reuse_type = "no_reuse", volume = 0 : i64}, {depth = 1 : i64, iterator = "%arg13", reuse_type = "total_reuse", volume = 16384 : i64}, {depth = 2 : i64, iterator = "%arg14", reuse_type = "no_reuse", volume = 0 : i64}]}} : memref<*xf32> to memref<64x64xf32, strided<[?, 1], offset: ?>>
              %16 = affine.min affine_map<(d0, d1, d2)[s0] -> (d0 * -64 + s0, 64)>(%8, %arg12, %7)[%arg3]
              %17 = affine.min affine_map<(d0, d1, d2)[s0] -> (d1 * -64 + s0, 64)>(%8, %arg12, %7)[%arg4]
              %extracted_slice = tensor.extract_slice %13[0, 0] [%16, %17] [1, 1] : tensor<64x64xf32> to tensor<?x?xf32>
              %18 = affine.min affine_map<(d0, d1, d2)[s0] -> (d0 * -64 + s0, 64)>(%8, %arg12, %7)[%arg3]
              %19 = affine.min affine_map<(d0, d1, d2)[s0] -> (d1 * -64 + s0, 64)>(%8, %arg12, %7)[%arg4]
              %subview = memref.subview %reinterpret_cast[0, 0] [%18, %19] [1, 1] : memref<64x64xf32, strided<[?, 1], offset: ?>> to memref<?x?xf32, strided<[?, 1], offset: ?>>
              bufferization.materialize_in_destination %extracted_slice in writable %subview : (tensor<?x?xf32>, memref<?x?xf32, strided<[?, 1], offset: ?>>) -> ()
            } {tmd.mapped_to = "y"}
          } {tmd.mapped_to = "x"}
        }
      }
    }
    return
  }
  func.func @matmul_kernel__d1i0_d0i2_f2_f0_f1(%arg0: memref<*xf32> {tt.divisibility = 16 : i32}, %arg1: memref<*xf32> {tt.divisibility = 16 : i32}, %arg2: memref<*xf32> {tt.divisibility = 16 : i32}, %arg3: index {tt.divisibility = 16 : i32}, %arg4: index {tt.divisibility = 16 : i32}, %arg5: index {tt.divisibility = 16 : i32}, %arg6: index {tt.divisibility = 16 : i32}, %arg7: index {tt.divisibility = 16 : i32}, %arg8: index {tt.divisibility = 16 : i32}, %arg9: index, %arg10: index, %arg11: index) {
    affine.for %arg12 = 0 to affine_map<(d0, d1, d2) -> (d2 ceildiv 8)>(%arg9, %arg10, %arg11) {
      affine.for %arg13 = 0 to affine_map<(d0, d1, d2) -> (d0 ceildiv 8)>(%arg9, %arg10, %arg11) {
        affine.for %arg14 = 0 to affine_map<(d0, d1, d2) -> (d1)>(%arg9, %arg10, %arg11) {
          affine.parallel (%arg15) = (0) to (8) {
            %7 = affine.apply affine_map<(d0, d1) -> (d0 * 8 + d1)>(%arg12, %arg15)
            affine.parallel (%arg16) = (0) to (8) {
              %8 = affine.apply affine_map<(d0, d1) -> (d0 * 8 + d1)>(%arg13, %arg16)
              %cst = arith.constant 0.000000e+00 : f32
              %9 = tensor.empty() : tensor<64x64xf32>
              %10 = linalg.fill ins(%cst : f32) outs(%9 : tensor<64x64xf32>) -> tensor<64x64xf32>
              %c64 = arith.constant 64 : index
              %11 = arith.muli %8, %c64 : index
              %12 = affine.apply affine_map<()[s0] -> ((s0 + 31) floordiv 32)>()[%arg5]
              %c0 = arith.constant 0 : index
              %c1 = arith.constant 1 : index
              %13 = scf.for %arg17 = %c0 to %12 step %c1 iter_args(%arg18 = %10) -> (tensor<64x64xf32>) {
                %c32 = arith.constant 32 : index
                %20 = arith.muli %arg17, %c32 : index
                %21 = arith.muli %11, %arg6 : index
                %22 = affine.apply affine_map<(d0)[s0] -> (d0 * 32 + s0)>(%arg17)[%21]
                %reinterpret_cast_0 = memref.reinterpret_cast %arg0 to offset: [%22], sizes: [64, 32], strides: [%arg6, 1] {tmd.reuse = {sequential = [{depth = 5 : i64, iterator = "%arg17", reuse_type = "no_reuse", volume = 0 : i64}], spatial = [{depth = 3 : i64, iterator = "%arg15", mapped_to = "x", reuse_type = "total_reuse", volume = 8192 : i64}, {depth = 4 : i64, iterator = "%arg16", mapped_to = "y", reuse_type = "no_reuse", volume = 0 : i64}], temporal = [{depth = 0 : i64, iterator = "%arg12", reuse_type = "total_reuse", volume = 8192 : i64}, {depth = 1 : i64, iterator = "%arg13", reuse_type = "no_reuse", volume = 0 : i64}, {depth = 2 : i64, iterator = "%arg14", reuse_type = "total_reuse", volume = 8192 : i64}]}} : memref<*xf32> to memref<64x32xf32, strided<[?, 1], offset: ?>>
                %alloc = memref.alloc() : memref<64x32xf32>
                memref.copy %reinterpret_cast_0, %alloc : memref<64x32xf32, strided<[?, 1], offset: ?>> to memref<64x32xf32>
                %23 = bufferization.to_tensor %alloc restrict writable : memref<64x32xf32> to tensor<64x32xf32>
                %24 = arith.muli %20, %arg7 : index
                %25 = affine.apply affine_map<(d0)[s0] -> (d0 * 64 + s0)>(%arg14)[%24]
                %reinterpret_cast_1 = memref.reinterpret_cast %arg1 to offset: [%25], sizes: [32, 64], strides: [%arg7, 1] {tmd.reuse = {sequential = [{depth = 5 : i64, iterator = "%arg17", reuse_type = "no_reuse", volume = 0 : i64}], spatial = [{depth = 3 : i64, iterator = "%arg15", mapped_to = "x", reuse_type = "total_reuse", volume = 8192 : i64}, {depth = 4 : i64, iterator = "%arg16", mapped_to = "y", reuse_type = "total_reuse", volume = 8192 : i64}], temporal = [{depth = 0 : i64, iterator = "%arg12", reuse_type = "total_reuse", volume = 8192 : i64}, {depth = 1 : i64, iterator = "%arg13", reuse_type = "total_reuse", volume = 8192 : i64}, {depth = 2 : i64, iterator = "%arg14", reuse_type = "no_reuse", volume = 0 : i64}]}} : memref<*xf32> to memref<32x64xf32, strided<[?, 1], offset: ?>>
                %alloc_2 = memref.alloc() : memref<32x64xf32>
                memref.copy %reinterpret_cast_1, %alloc_2 : memref<32x64xf32, strided<[?, 1], offset: ?>> to memref<32x64xf32>
                %26 = bufferization.to_tensor %alloc_2 restrict writable : memref<32x64xf32> to tensor<32x64xf32>
                %27 = linalg.matmul ins(%23, %26 : tensor<64x32xf32>, tensor<32x64xf32>) outs(%10 : tensor<64x64xf32>) -> tensor<64x64xf32>
                %28 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%arg18, %27 : tensor<64x64xf32>, tensor<64x64xf32>) outs(%arg18 : tensor<64x64xf32>) {
                ^bb0(%in: f32, %in_3: f32, %out: f32):
                  %29 = arith.addf %in, %in_3 : f32
                  linalg.yield %29 : f32
                } -> tensor<64x64xf32>
                scf.yield %28 : tensor<64x64xf32>
              }
              %14 = arith.muli %11, %arg8 : index
              %15 = affine.apply affine_map<(d0)[s0] -> (d0 * 64 + s0)>(%arg14)[%14]
              %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%15], sizes: [64, 64], strides: [%arg8, 1] {tmd.reuse = {spatial = [{depth = 3 : i64, iterator = "%arg15", mapped_to = "x", reuse_type = "total_reuse", volume = 16384 : i64}, {depth = 4 : i64, iterator = "%arg16", mapped_to = "y", reuse_type = "no_reuse", volume = 0 : i64}], temporal = [{depth = 0 : i64, iterator = "%arg12", reuse_type = "total_reuse", volume = 16384 : i64}, {depth = 1 : i64, iterator = "%arg13", reuse_type = "no_reuse", volume = 0 : i64}, {depth = 2 : i64, iterator = "%arg14", reuse_type = "no_reuse", volume = 0 : i64}]}} : memref<*xf32> to memref<64x64xf32, strided<[?, 1], offset: ?>>
              %16 = affine.min affine_map<(d0, d1, d2)[s0] -> (d0 * -64 + s0, 64)>(%8, %arg14, %7)[%arg3]
              %17 = affine.min affine_map<(d0, d1, d2)[s0] -> (d1 * -64 + s0, 64)>(%8, %arg14, %7)[%arg4]
              %extracted_slice = tensor.extract_slice %13[0, 0] [%16, %17] [1, 1] : tensor<64x64xf32> to tensor<?x?xf32>
              %18 = affine.min affine_map<(d0, d1, d2)[s0] -> (d0 * -64 + s0, 64)>(%8, %arg14, %7)[%arg3]
              %19 = affine.min affine_map<(d0, d1, d2)[s0] -> (d1 * -64 + s0, 64)>(%8, %arg14, %7)[%arg4]
              %subview = memref.subview %reinterpret_cast[0, 0] [%18, %19] [1, 1] : memref<64x64xf32, strided<[?, 1], offset: ?>> to memref<?x?xf32, strided<[?, 1], offset: ?>>
              bufferization.materialize_in_destination %extracted_slice in writable %subview : (tensor<?x?xf32>, memref<?x?xf32, strided<[?, 1], offset: ?>>) -> ()
            } {tmd.mapped_to = "y"}
          } {tmd.mapped_to = "x"}
        }
      }
    }
    return
  }
  func.func @matmul_kernel__d1i0_d0i2_f2_f1_f0(%arg0: memref<*xf32> {tt.divisibility = 16 : i32}, %arg1: memref<*xf32> {tt.divisibility = 16 : i32}, %arg2: memref<*xf32> {tt.divisibility = 16 : i32}, %arg3: index {tt.divisibility = 16 : i32}, %arg4: index {tt.divisibility = 16 : i32}, %arg5: index {tt.divisibility = 16 : i32}, %arg6: index {tt.divisibility = 16 : i32}, %arg7: index {tt.divisibility = 16 : i32}, %arg8: index {tt.divisibility = 16 : i32}, %arg9: index, %arg10: index, %arg11: index) {
    affine.for %arg12 = 0 to affine_map<(d0, d1, d2) -> (d2 ceildiv 8)>(%arg9, %arg10, %arg11) {
      affine.for %arg13 = 0 to affine_map<(d0, d1, d2) -> (d1)>(%arg9, %arg10, %arg11) {
        affine.for %arg14 = 0 to affine_map<(d0, d1, d2) -> (d0 ceildiv 8)>(%arg9, %arg10, %arg11) {
          affine.parallel (%arg15) = (0) to (8) {
            %7 = affine.apply affine_map<(d0, d1) -> (d0 * 8 + d1)>(%arg12, %arg15)
            affine.parallel (%arg16) = (0) to (8) {
              %8 = affine.apply affine_map<(d0, d1) -> (d0 * 8 + d1)>(%arg14, %arg16)
              %cst = arith.constant 0.000000e+00 : f32
              %9 = tensor.empty() : tensor<64x64xf32>
              %10 = linalg.fill ins(%cst : f32) outs(%9 : tensor<64x64xf32>) -> tensor<64x64xf32>
              %c64 = arith.constant 64 : index
              %11 = arith.muli %8, %c64 : index
              %12 = affine.apply affine_map<()[s0] -> ((s0 + 31) floordiv 32)>()[%arg5]
              %c0 = arith.constant 0 : index
              %c1 = arith.constant 1 : index
              %13 = scf.for %arg17 = %c0 to %12 step %c1 iter_args(%arg18 = %10) -> (tensor<64x64xf32>) {
                %c32 = arith.constant 32 : index
                %20 = arith.muli %arg17, %c32 : index
                %21 = arith.muli %11, %arg6 : index
                %22 = affine.apply affine_map<(d0)[s0] -> (d0 * 32 + s0)>(%arg17)[%21]
                %reinterpret_cast_0 = memref.reinterpret_cast %arg0 to offset: [%22], sizes: [64, 32], strides: [%arg6, 1] {tmd.reuse = {sequential = [{depth = 5 : i64, iterator = "%arg17", reuse_type = "no_reuse", volume = 0 : i64}], spatial = [{depth = 3 : i64, iterator = "%arg15", mapped_to = "x", reuse_type = "total_reuse", volume = 8192 : i64}, {depth = 4 : i64, iterator = "%arg16", mapped_to = "y", reuse_type = "no_reuse", volume = 0 : i64}], temporal = [{depth = 0 : i64, iterator = "%arg12", reuse_type = "total_reuse", volume = 8192 : i64}, {depth = 1 : i64, iterator = "%arg13", reuse_type = "total_reuse", volume = 8192 : i64}, {depth = 2 : i64, iterator = "%arg14", reuse_type = "no_reuse", volume = 0 : i64}]}} : memref<*xf32> to memref<64x32xf32, strided<[?, 1], offset: ?>>
                %alloc = memref.alloc() : memref<64x32xf32>
                memref.copy %reinterpret_cast_0, %alloc : memref<64x32xf32, strided<[?, 1], offset: ?>> to memref<64x32xf32>
                %23 = bufferization.to_tensor %alloc restrict writable : memref<64x32xf32> to tensor<64x32xf32>
                %24 = arith.muli %20, %arg7 : index
                %25 = affine.apply affine_map<(d0)[s0] -> (d0 * 64 + s0)>(%arg13)[%24]
                %reinterpret_cast_1 = memref.reinterpret_cast %arg1 to offset: [%25], sizes: [32, 64], strides: [%arg7, 1] {tmd.reuse = {sequential = [{depth = 5 : i64, iterator = "%arg17", reuse_type = "no_reuse", volume = 0 : i64}], spatial = [{depth = 3 : i64, iterator = "%arg15", mapped_to = "x", reuse_type = "total_reuse", volume = 8192 : i64}, {depth = 4 : i64, iterator = "%arg16", mapped_to = "y", reuse_type = "total_reuse", volume = 8192 : i64}], temporal = [{depth = 0 : i64, iterator = "%arg12", reuse_type = "total_reuse", volume = 8192 : i64}, {depth = 1 : i64, iterator = "%arg13", reuse_type = "no_reuse", volume = 0 : i64}, {depth = 2 : i64, iterator = "%arg14", reuse_type = "total_reuse", volume = 8192 : i64}]}} : memref<*xf32> to memref<32x64xf32, strided<[?, 1], offset: ?>>
                %alloc_2 = memref.alloc() : memref<32x64xf32>
                memref.copy %reinterpret_cast_1, %alloc_2 : memref<32x64xf32, strided<[?, 1], offset: ?>> to memref<32x64xf32>
                %26 = bufferization.to_tensor %alloc_2 restrict writable : memref<32x64xf32> to tensor<32x64xf32>
                %27 = linalg.matmul ins(%23, %26 : tensor<64x32xf32>, tensor<32x64xf32>) outs(%10 : tensor<64x64xf32>) -> tensor<64x64xf32>
                %28 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%arg18, %27 : tensor<64x64xf32>, tensor<64x64xf32>) outs(%arg18 : tensor<64x64xf32>) {
                ^bb0(%in: f32, %in_3: f32, %out: f32):
                  %29 = arith.addf %in, %in_3 : f32
                  linalg.yield %29 : f32
                } -> tensor<64x64xf32>
                scf.yield %28 : tensor<64x64xf32>
              }
              %14 = arith.muli %11, %arg8 : index
              %15 = affine.apply affine_map<(d0)[s0] -> (d0 * 64 + s0)>(%arg13)[%14]
              %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%15], sizes: [64, 64], strides: [%arg8, 1] {tmd.reuse = {spatial = [{depth = 3 : i64, iterator = "%arg15", mapped_to = "x", reuse_type = "total_reuse", volume = 16384 : i64}, {depth = 4 : i64, iterator = "%arg16", mapped_to = "y", reuse_type = "no_reuse", volume = 0 : i64}], temporal = [{depth = 0 : i64, iterator = "%arg12", reuse_type = "total_reuse", volume = 16384 : i64}, {depth = 1 : i64, iterator = "%arg13", reuse_type = "no_reuse", volume = 0 : i64}, {depth = 2 : i64, iterator = "%arg14", reuse_type = "no_reuse", volume = 0 : i64}]}} : memref<*xf32> to memref<64x64xf32, strided<[?, 1], offset: ?>>
              %16 = affine.min affine_map<(d0, d1, d2)[s0] -> (d0 * -64 + s0, 64)>(%8, %arg13, %7)[%arg3]
              %17 = affine.min affine_map<(d0, d1, d2)[s0] -> (d1 * -64 + s0, 64)>(%8, %arg13, %7)[%arg4]
              %extracted_slice = tensor.extract_slice %13[0, 0] [%16, %17] [1, 1] : tensor<64x64xf32> to tensor<?x?xf32>
              %18 = affine.min affine_map<(d0, d1, d2)[s0] -> (d0 * -64 + s0, 64)>(%8, %arg13, %7)[%arg3]
              %19 = affine.min affine_map<(d0, d1, d2)[s0] -> (d1 * -64 + s0, 64)>(%8, %arg13, %7)[%arg4]
              %subview = memref.subview %reinterpret_cast[0, 0] [%18, %19] [1, 1] : memref<64x64xf32, strided<[?, 1], offset: ?>> to memref<?x?xf32, strided<[?, 1], offset: ?>>
              bufferization.materialize_in_destination %extracted_slice in writable %subview : (tensor<?x?xf32>, memref<?x?xf32, strided<[?, 1], offset: ?>>) -> ()
            } {tmd.mapped_to = "y"}
          } {tmd.mapped_to = "x"}
        }
      }
    }
    return
  }
  func.func @matmul_kernel__d1i1_d0i2_f0_f1_f2(%arg0: memref<*xf32> {tt.divisibility = 16 : i32}, %arg1: memref<*xf32> {tt.divisibility = 16 : i32}, %arg2: memref<*xf32> {tt.divisibility = 16 : i32}, %arg3: index {tt.divisibility = 16 : i32}, %arg4: index {tt.divisibility = 16 : i32}, %arg5: index {tt.divisibility = 16 : i32}, %arg6: index {tt.divisibility = 16 : i32}, %arg7: index {tt.divisibility = 16 : i32}, %arg8: index {tt.divisibility = 16 : i32}, %arg9: index, %arg10: index, %arg11: index) {
    affine.for %arg12 = 0 to affine_map<(d0, d1, d2) -> (d0)>(%arg9, %arg10, %arg11) {
      affine.for %arg13 = 0 to affine_map<(d0, d1, d2) -> (d1 ceildiv 8)>(%arg9, %arg10, %arg11) {
        affine.for %arg14 = 0 to affine_map<(d0, d1, d2) -> (d2 ceildiv 8)>(%arg9, %arg10, %arg11) {
          affine.parallel (%arg15) = (0) to (8) {
            %7 = affine.apply affine_map<(d0, d1) -> (d0 * 8 + d1)>(%arg14, %arg15)
            affine.parallel (%arg16) = (0) to (8) {
              %8 = affine.apply affine_map<(d0, d1) -> (d0 * 8 + d1)>(%arg13, %arg16)
              %cst = arith.constant 0.000000e+00 : f32
              %9 = tensor.empty() : tensor<64x64xf32>
              %10 = linalg.fill ins(%cst : f32) outs(%9 : tensor<64x64xf32>) -> tensor<64x64xf32>
              %c64 = arith.constant 64 : index
              %11 = arith.muli %arg12, %c64 : index
              %12 = affine.apply affine_map<()[s0] -> ((s0 + 31) floordiv 32)>()[%arg5]
              %c0 = arith.constant 0 : index
              %c1 = arith.constant 1 : index
              %13 = scf.for %arg17 = %c0 to %12 step %c1 iter_args(%arg18 = %10) -> (tensor<64x64xf32>) {
                %c32 = arith.constant 32 : index
                %20 = arith.muli %arg17, %c32 : index
                %21 = arith.muli %11, %arg6 : index
                %22 = affine.apply affine_map<(d0)[s0] -> (d0 * 32 + s0)>(%arg17)[%21]
                %reinterpret_cast_0 = memref.reinterpret_cast %arg0 to offset: [%22], sizes: [64, 32], strides: [%arg6, 1] {tmd.reuse = {sequential = [{depth = 5 : i64, iterator = "%arg17", reuse_type = "no_reuse", volume = 0 : i64}], spatial = [{depth = 3 : i64, iterator = "%arg15", mapped_to = "x", reuse_type = "total_reuse", volume = 8192 : i64}, {depth = 4 : i64, iterator = "%arg16", mapped_to = "y", reuse_type = "total_reuse", volume = 8192 : i64}], temporal = [{depth = 0 : i64, iterator = "%arg12", reuse_type = "no_reuse", volume = 0 : i64}, {depth = 1 : i64, iterator = "%arg13", reuse_type = "total_reuse", volume = 8192 : i64}, {depth = 2 : i64, iterator = "%arg14", reuse_type = "total_reuse", volume = 8192 : i64}]}} : memref<*xf32> to memref<64x32xf32, strided<[?, 1], offset: ?>>
                %alloc = memref.alloc() : memref<64x32xf32>
                memref.copy %reinterpret_cast_0, %alloc : memref<64x32xf32, strided<[?, 1], offset: ?>> to memref<64x32xf32>
                %23 = bufferization.to_tensor %alloc restrict writable : memref<64x32xf32> to tensor<64x32xf32>
                %24 = arith.muli %20, %arg7 : index
                %25 = affine.apply affine_map<(d0, d1)[s0] -> (d0 * 512 + d1 * 64 + s0)>(%arg13, %arg16)[%24]
                %reinterpret_cast_1 = memref.reinterpret_cast %arg1 to offset: [%25], sizes: [32, 64], strides: [%arg7, 1] {tmd.reuse = {sequential = [{depth = 5 : i64, iterator = "%arg17", reuse_type = "no_reuse", volume = 0 : i64}], spatial = [{depth = 3 : i64, iterator = "%arg15", mapped_to = "x", reuse_type = "total_reuse", volume = 8192 : i64}, {depth = 4 : i64, iterator = "%arg16", mapped_to = "y", reuse_type = "no_reuse", volume = 0 : i64}], temporal = [{depth = 0 : i64, iterator = "%arg12", reuse_type = "total_reuse", volume = 8192 : i64}, {depth = 1 : i64, iterator = "%arg13", reuse_type = "no_reuse", volume = 0 : i64}, {depth = 2 : i64, iterator = "%arg14", reuse_type = "total_reuse", volume = 8192 : i64}]}} : memref<*xf32> to memref<32x64xf32, strided<[?, 1], offset: ?>>
                %alloc_2 = memref.alloc() : memref<32x64xf32>
                memref.copy %reinterpret_cast_1, %alloc_2 : memref<32x64xf32, strided<[?, 1], offset: ?>> to memref<32x64xf32>
                %26 = bufferization.to_tensor %alloc_2 restrict writable : memref<32x64xf32> to tensor<32x64xf32>
                %27 = linalg.matmul ins(%23, %26 : tensor<64x32xf32>, tensor<32x64xf32>) outs(%10 : tensor<64x64xf32>) -> tensor<64x64xf32>
                %28 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%arg18, %27 : tensor<64x64xf32>, tensor<64x64xf32>) outs(%arg18 : tensor<64x64xf32>) {
                ^bb0(%in: f32, %in_3: f32, %out: f32):
                  %29 = arith.addf %in, %in_3 : f32
                  linalg.yield %29 : f32
                } -> tensor<64x64xf32>
                scf.yield %28 : tensor<64x64xf32>
              }
              %14 = arith.muli %11, %arg8 : index
              %15 = affine.apply affine_map<(d0, d1)[s0] -> (d0 * 512 + d1 * 64 + s0)>(%arg13, %arg16)[%14]
              %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%15], sizes: [64, 64], strides: [%arg8, 1] {tmd.reuse = {spatial = [{depth = 3 : i64, iterator = "%arg15", mapped_to = "x", reuse_type = "total_reuse", volume = 16384 : i64}, {depth = 4 : i64, iterator = "%arg16", mapped_to = "y", reuse_type = "no_reuse", volume = 0 : i64}], temporal = [{depth = 0 : i64, iterator = "%arg12", reuse_type = "no_reuse", volume = 0 : i64}, {depth = 1 : i64, iterator = "%arg13", reuse_type = "no_reuse", volume = 0 : i64}, {depth = 2 : i64, iterator = "%arg14", reuse_type = "total_reuse", volume = 16384 : i64}]}} : memref<*xf32> to memref<64x64xf32, strided<[?, 1], offset: ?>>
              %16 = affine.min affine_map<(d0, d1, d2)[s0] -> (d0 * -64 + s0, 64)>(%arg12, %8, %7)[%arg3]
              %17 = affine.min affine_map<(d0, d1, d2)[s0] -> (d1 * -64 + s0, 64)>(%arg12, %8, %7)[%arg4]
              %extracted_slice = tensor.extract_slice %13[0, 0] [%16, %17] [1, 1] : tensor<64x64xf32> to tensor<?x?xf32>
              %18 = affine.min affine_map<(d0, d1, d2)[s0] -> (d0 * -64 + s0, 64)>(%arg12, %8, %7)[%arg3]
              %19 = affine.min affine_map<(d0, d1, d2)[s0] -> (d1 * -64 + s0, 64)>(%arg12, %8, %7)[%arg4]
              %subview = memref.subview %reinterpret_cast[0, 0] [%18, %19] [1, 1] : memref<64x64xf32, strided<[?, 1], offset: ?>> to memref<?x?xf32, strided<[?, 1], offset: ?>>
              bufferization.materialize_in_destination %extracted_slice in writable %subview : (tensor<?x?xf32>, memref<?x?xf32, strided<[?, 1], offset: ?>>) -> ()
            } {tmd.mapped_to = "y"}
          } {tmd.mapped_to = "x"}
        }
      }
    }
    return
  }
  func.func @matmul_kernel__d1i1_d0i2_f0_f2_f1(%arg0: memref<*xf32> {tt.divisibility = 16 : i32}, %arg1: memref<*xf32> {tt.divisibility = 16 : i32}, %arg2: memref<*xf32> {tt.divisibility = 16 : i32}, %arg3: index {tt.divisibility = 16 : i32}, %arg4: index {tt.divisibility = 16 : i32}, %arg5: index {tt.divisibility = 16 : i32}, %arg6: index {tt.divisibility = 16 : i32}, %arg7: index {tt.divisibility = 16 : i32}, %arg8: index {tt.divisibility = 16 : i32}, %arg9: index, %arg10: index, %arg11: index) {
    affine.for %arg12 = 0 to affine_map<(d0, d1, d2) -> (d0)>(%arg9, %arg10, %arg11) {
      affine.for %arg13 = 0 to affine_map<(d0, d1, d2) -> (d2 ceildiv 8)>(%arg9, %arg10, %arg11) {
        affine.for %arg14 = 0 to affine_map<(d0, d1, d2) -> (d1 ceildiv 8)>(%arg9, %arg10, %arg11) {
          affine.parallel (%arg15) = (0) to (8) {
            %7 = affine.apply affine_map<(d0, d1) -> (d0 * 8 + d1)>(%arg13, %arg15)
            affine.parallel (%arg16) = (0) to (8) {
              %8 = affine.apply affine_map<(d0, d1) -> (d0 * 8 + d1)>(%arg14, %arg16)
              %cst = arith.constant 0.000000e+00 : f32
              %9 = tensor.empty() : tensor<64x64xf32>
              %10 = linalg.fill ins(%cst : f32) outs(%9 : tensor<64x64xf32>) -> tensor<64x64xf32>
              %c64 = arith.constant 64 : index
              %11 = arith.muli %arg12, %c64 : index
              %12 = affine.apply affine_map<()[s0] -> ((s0 + 31) floordiv 32)>()[%arg5]
              %c0 = arith.constant 0 : index
              %c1 = arith.constant 1 : index
              %13 = scf.for %arg17 = %c0 to %12 step %c1 iter_args(%arg18 = %10) -> (tensor<64x64xf32>) {
                %c32 = arith.constant 32 : index
                %20 = arith.muli %arg17, %c32 : index
                %21 = arith.muli %11, %arg6 : index
                %22 = affine.apply affine_map<(d0)[s0] -> (d0 * 32 + s0)>(%arg17)[%21]
                %reinterpret_cast_0 = memref.reinterpret_cast %arg0 to offset: [%22], sizes: [64, 32], strides: [%arg6, 1] {tmd.reuse = {sequential = [{depth = 5 : i64, iterator = "%arg17", reuse_type = "no_reuse", volume = 0 : i64}], spatial = [{depth = 3 : i64, iterator = "%arg15", mapped_to = "x", reuse_type = "total_reuse", volume = 8192 : i64}, {depth = 4 : i64, iterator = "%arg16", mapped_to = "y", reuse_type = "total_reuse", volume = 8192 : i64}], temporal = [{depth = 0 : i64, iterator = "%arg12", reuse_type = "no_reuse", volume = 0 : i64}, {depth = 1 : i64, iterator = "%arg13", reuse_type = "total_reuse", volume = 8192 : i64}, {depth = 2 : i64, iterator = "%arg14", reuse_type = "total_reuse", volume = 8192 : i64}]}} : memref<*xf32> to memref<64x32xf32, strided<[?, 1], offset: ?>>
                %alloc = memref.alloc() : memref<64x32xf32>
                memref.copy %reinterpret_cast_0, %alloc : memref<64x32xf32, strided<[?, 1], offset: ?>> to memref<64x32xf32>
                %23 = bufferization.to_tensor %alloc restrict writable : memref<64x32xf32> to tensor<64x32xf32>
                %24 = arith.muli %20, %arg7 : index
                %25 = affine.apply affine_map<(d0, d1)[s0] -> (d0 * 512 + d1 * 64 + s0)>(%arg14, %arg16)[%24]
                %reinterpret_cast_1 = memref.reinterpret_cast %arg1 to offset: [%25], sizes: [32, 64], strides: [%arg7, 1] {tmd.reuse = {sequential = [{depth = 5 : i64, iterator = "%arg17", reuse_type = "no_reuse", volume = 0 : i64}], spatial = [{depth = 3 : i64, iterator = "%arg15", mapped_to = "x", reuse_type = "total_reuse", volume = 8192 : i64}, {depth = 4 : i64, iterator = "%arg16", mapped_to = "y", reuse_type = "no_reuse", volume = 0 : i64}], temporal = [{depth = 0 : i64, iterator = "%arg12", reuse_type = "total_reuse", volume = 8192 : i64}, {depth = 1 : i64, iterator = "%arg13", reuse_type = "total_reuse", volume = 8192 : i64}, {depth = 2 : i64, iterator = "%arg14", reuse_type = "no_reuse", volume = 0 : i64}]}} : memref<*xf32> to memref<32x64xf32, strided<[?, 1], offset: ?>>
                %alloc_2 = memref.alloc() : memref<32x64xf32>
                memref.copy %reinterpret_cast_1, %alloc_2 : memref<32x64xf32, strided<[?, 1], offset: ?>> to memref<32x64xf32>
                %26 = bufferization.to_tensor %alloc_2 restrict writable : memref<32x64xf32> to tensor<32x64xf32>
                %27 = linalg.matmul ins(%23, %26 : tensor<64x32xf32>, tensor<32x64xf32>) outs(%10 : tensor<64x64xf32>) -> tensor<64x64xf32>
                %28 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%arg18, %27 : tensor<64x64xf32>, tensor<64x64xf32>) outs(%arg18 : tensor<64x64xf32>) {
                ^bb0(%in: f32, %in_3: f32, %out: f32):
                  %29 = arith.addf %in, %in_3 : f32
                  linalg.yield %29 : f32
                } -> tensor<64x64xf32>
                scf.yield %28 : tensor<64x64xf32>
              }
              %14 = arith.muli %11, %arg8 : index
              %15 = affine.apply affine_map<(d0, d1)[s0] -> (d0 * 512 + d1 * 64 + s0)>(%arg14, %arg16)[%14]
              %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%15], sizes: [64, 64], strides: [%arg8, 1] {tmd.reuse = {spatial = [{depth = 3 : i64, iterator = "%arg15", mapped_to = "x", reuse_type = "total_reuse", volume = 16384 : i64}, {depth = 4 : i64, iterator = "%arg16", mapped_to = "y", reuse_type = "no_reuse", volume = 0 : i64}], temporal = [{depth = 0 : i64, iterator = "%arg12", reuse_type = "no_reuse", volume = 0 : i64}, {depth = 1 : i64, iterator = "%arg13", reuse_type = "total_reuse", volume = 16384 : i64}, {depth = 2 : i64, iterator = "%arg14", reuse_type = "no_reuse", volume = 0 : i64}]}} : memref<*xf32> to memref<64x64xf32, strided<[?, 1], offset: ?>>
              %16 = affine.min affine_map<(d0, d1, d2)[s0] -> (d0 * -64 + s0, 64)>(%arg12, %8, %7)[%arg3]
              %17 = affine.min affine_map<(d0, d1, d2)[s0] -> (d1 * -64 + s0, 64)>(%arg12, %8, %7)[%arg4]
              %extracted_slice = tensor.extract_slice %13[0, 0] [%16, %17] [1, 1] : tensor<64x64xf32> to tensor<?x?xf32>
              %18 = affine.min affine_map<(d0, d1, d2)[s0] -> (d0 * -64 + s0, 64)>(%arg12, %8, %7)[%arg3]
              %19 = affine.min affine_map<(d0, d1, d2)[s0] -> (d1 * -64 + s0, 64)>(%arg12, %8, %7)[%arg4]
              %subview = memref.subview %reinterpret_cast[0, 0] [%18, %19] [1, 1] : memref<64x64xf32, strided<[?, 1], offset: ?>> to memref<?x?xf32, strided<[?, 1], offset: ?>>
              bufferization.materialize_in_destination %extracted_slice in writable %subview : (tensor<?x?xf32>, memref<?x?xf32, strided<[?, 1], offset: ?>>) -> ()
            } {tmd.mapped_to = "y"}
          } {tmd.mapped_to = "x"}
        }
      }
    }
    return
  }
  func.func @matmul_kernel__d1i1_d0i2_f1_f0_f2(%arg0: memref<*xf32> {tt.divisibility = 16 : i32}, %arg1: memref<*xf32> {tt.divisibility = 16 : i32}, %arg2: memref<*xf32> {tt.divisibility = 16 : i32}, %arg3: index {tt.divisibility = 16 : i32}, %arg4: index {tt.divisibility = 16 : i32}, %arg5: index {tt.divisibility = 16 : i32}, %arg6: index {tt.divisibility = 16 : i32}, %arg7: index {tt.divisibility = 16 : i32}, %arg8: index {tt.divisibility = 16 : i32}, %arg9: index, %arg10: index, %arg11: index) {
    affine.for %arg12 = 0 to affine_map<(d0, d1, d2) -> (d1 ceildiv 8)>(%arg9, %arg10, %arg11) {
      affine.for %arg13 = 0 to affine_map<(d0, d1, d2) -> (d0)>(%arg9, %arg10, %arg11) {
        affine.for %arg14 = 0 to affine_map<(d0, d1, d2) -> (d2 ceildiv 8)>(%arg9, %arg10, %arg11) {
          affine.parallel (%arg15) = (0) to (8) {
            %7 = affine.apply affine_map<(d0, d1) -> (d0 * 8 + d1)>(%arg14, %arg15)
            affine.parallel (%arg16) = (0) to (8) {
              %8 = affine.apply affine_map<(d0, d1) -> (d0 * 8 + d1)>(%arg12, %arg16)
              %cst = arith.constant 0.000000e+00 : f32
              %9 = tensor.empty() : tensor<64x64xf32>
              %10 = linalg.fill ins(%cst : f32) outs(%9 : tensor<64x64xf32>) -> tensor<64x64xf32>
              %c64 = arith.constant 64 : index
              %11 = arith.muli %arg13, %c64 : index
              %12 = affine.apply affine_map<()[s0] -> ((s0 + 31) floordiv 32)>()[%arg5]
              %c0 = arith.constant 0 : index
              %c1 = arith.constant 1 : index
              %13 = scf.for %arg17 = %c0 to %12 step %c1 iter_args(%arg18 = %10) -> (tensor<64x64xf32>) {
                %c32 = arith.constant 32 : index
                %20 = arith.muli %arg17, %c32 : index
                %21 = arith.muli %11, %arg6 : index
                %22 = affine.apply affine_map<(d0)[s0] -> (d0 * 32 + s0)>(%arg17)[%21]
                %reinterpret_cast_0 = memref.reinterpret_cast %arg0 to offset: [%22], sizes: [64, 32], strides: [%arg6, 1] {tmd.reuse = {sequential = [{depth = 5 : i64, iterator = "%arg17", reuse_type = "no_reuse", volume = 0 : i64}], spatial = [{depth = 3 : i64, iterator = "%arg15", mapped_to = "x", reuse_type = "total_reuse", volume = 8192 : i64}, {depth = 4 : i64, iterator = "%arg16", mapped_to = "y", reuse_type = "total_reuse", volume = 8192 : i64}], temporal = [{depth = 0 : i64, iterator = "%arg12", reuse_type = "total_reuse", volume = 8192 : i64}, {depth = 1 : i64, iterator = "%arg13", reuse_type = "no_reuse", volume = 0 : i64}, {depth = 2 : i64, iterator = "%arg14", reuse_type = "total_reuse", volume = 8192 : i64}]}} : memref<*xf32> to memref<64x32xf32, strided<[?, 1], offset: ?>>
                %alloc = memref.alloc() : memref<64x32xf32>
                memref.copy %reinterpret_cast_0, %alloc : memref<64x32xf32, strided<[?, 1], offset: ?>> to memref<64x32xf32>
                %23 = bufferization.to_tensor %alloc restrict writable : memref<64x32xf32> to tensor<64x32xf32>
                %24 = arith.muli %20, %arg7 : index
                %25 = affine.apply affine_map<(d0, d1)[s0] -> (d0 * 512 + d1 * 64 + s0)>(%arg12, %arg16)[%24]
                %reinterpret_cast_1 = memref.reinterpret_cast %arg1 to offset: [%25], sizes: [32, 64], strides: [%arg7, 1] {tmd.reuse = {sequential = [{depth = 5 : i64, iterator = "%arg17", reuse_type = "no_reuse", volume = 0 : i64}], spatial = [{depth = 3 : i64, iterator = "%arg15", mapped_to = "x", reuse_type = "total_reuse", volume = 8192 : i64}, {depth = 4 : i64, iterator = "%arg16", mapped_to = "y", reuse_type = "no_reuse", volume = 0 : i64}], temporal = [{depth = 0 : i64, iterator = "%arg12", reuse_type = "no_reuse", volume = 0 : i64}, {depth = 1 : i64, iterator = "%arg13", reuse_type = "total_reuse", volume = 8192 : i64}, {depth = 2 : i64, iterator = "%arg14", reuse_type = "total_reuse", volume = 8192 : i64}]}} : memref<*xf32> to memref<32x64xf32, strided<[?, 1], offset: ?>>
                %alloc_2 = memref.alloc() : memref<32x64xf32>
                memref.copy %reinterpret_cast_1, %alloc_2 : memref<32x64xf32, strided<[?, 1], offset: ?>> to memref<32x64xf32>
                %26 = bufferization.to_tensor %alloc_2 restrict writable : memref<32x64xf32> to tensor<32x64xf32>
                %27 = linalg.matmul ins(%23, %26 : tensor<64x32xf32>, tensor<32x64xf32>) outs(%10 : tensor<64x64xf32>) -> tensor<64x64xf32>
                %28 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%arg18, %27 : tensor<64x64xf32>, tensor<64x64xf32>) outs(%arg18 : tensor<64x64xf32>) {
                ^bb0(%in: f32, %in_3: f32, %out: f32):
                  %29 = arith.addf %in, %in_3 : f32
                  linalg.yield %29 : f32
                } -> tensor<64x64xf32>
                scf.yield %28 : tensor<64x64xf32>
              }
              %14 = arith.muli %11, %arg8 : index
              %15 = affine.apply affine_map<(d0, d1)[s0] -> (d0 * 512 + d1 * 64 + s0)>(%arg12, %arg16)[%14]
              %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%15], sizes: [64, 64], strides: [%arg8, 1] {tmd.reuse = {spatial = [{depth = 3 : i64, iterator = "%arg15", mapped_to = "x", reuse_type = "total_reuse", volume = 16384 : i64}, {depth = 4 : i64, iterator = "%arg16", mapped_to = "y", reuse_type = "no_reuse", volume = 0 : i64}], temporal = [{depth = 0 : i64, iterator = "%arg12", reuse_type = "no_reuse", volume = 0 : i64}, {depth = 1 : i64, iterator = "%arg13", reuse_type = "no_reuse", volume = 0 : i64}, {depth = 2 : i64, iterator = "%arg14", reuse_type = "total_reuse", volume = 16384 : i64}]}} : memref<*xf32> to memref<64x64xf32, strided<[?, 1], offset: ?>>
              %16 = affine.min affine_map<(d0, d1, d2)[s0] -> (d0 * -64 + s0, 64)>(%arg13, %8, %7)[%arg3]
              %17 = affine.min affine_map<(d0, d1, d2)[s0] -> (d1 * -64 + s0, 64)>(%arg13, %8, %7)[%arg4]
              %extracted_slice = tensor.extract_slice %13[0, 0] [%16, %17] [1, 1] : tensor<64x64xf32> to tensor<?x?xf32>
              %18 = affine.min affine_map<(d0, d1, d2)[s0] -> (d0 * -64 + s0, 64)>(%arg13, %8, %7)[%arg3]
              %19 = affine.min affine_map<(d0, d1, d2)[s0] -> (d1 * -64 + s0, 64)>(%arg13, %8, %7)[%arg4]
              %subview = memref.subview %reinterpret_cast[0, 0] [%18, %19] [1, 1] : memref<64x64xf32, strided<[?, 1], offset: ?>> to memref<?x?xf32, strided<[?, 1], offset: ?>>
              bufferization.materialize_in_destination %extracted_slice in writable %subview : (tensor<?x?xf32>, memref<?x?xf32, strided<[?, 1], offset: ?>>) -> ()
            } {tmd.mapped_to = "y"}
          } {tmd.mapped_to = "x"}
        }
      }
    }
    return
  }
  func.func @matmul_kernel__d1i1_d0i2_f1_f2_f0(%arg0: memref<*xf32> {tt.divisibility = 16 : i32}, %arg1: memref<*xf32> {tt.divisibility = 16 : i32}, %arg2: memref<*xf32> {tt.divisibility = 16 : i32}, %arg3: index {tt.divisibility = 16 : i32}, %arg4: index {tt.divisibility = 16 : i32}, %arg5: index {tt.divisibility = 16 : i32}, %arg6: index {tt.divisibility = 16 : i32}, %arg7: index {tt.divisibility = 16 : i32}, %arg8: index {tt.divisibility = 16 : i32}, %arg9: index, %arg10: index, %arg11: index) {
    affine.for %arg12 = 0 to affine_map<(d0, d1, d2) -> (d1 ceildiv 8)>(%arg9, %arg10, %arg11) {
      affine.for %arg13 = 0 to affine_map<(d0, d1, d2) -> (d2 ceildiv 8)>(%arg9, %arg10, %arg11) {
        affine.for %arg14 = 0 to affine_map<(d0, d1, d2) -> (d0)>(%arg9, %arg10, %arg11) {
          affine.parallel (%arg15) = (0) to (8) {
            %7 = affine.apply affine_map<(d0, d1) -> (d0 * 8 + d1)>(%arg13, %arg15)
            affine.parallel (%arg16) = (0) to (8) {
              %8 = affine.apply affine_map<(d0, d1) -> (d0 * 8 + d1)>(%arg12, %arg16)
              %cst = arith.constant 0.000000e+00 : f32
              %9 = tensor.empty() : tensor<64x64xf32>
              %10 = linalg.fill ins(%cst : f32) outs(%9 : tensor<64x64xf32>) -> tensor<64x64xf32>
              %c64 = arith.constant 64 : index
              %11 = arith.muli %arg14, %c64 : index
              %12 = affine.apply affine_map<()[s0] -> ((s0 + 31) floordiv 32)>()[%arg5]
              %c0 = arith.constant 0 : index
              %c1 = arith.constant 1 : index
              %13 = scf.for %arg17 = %c0 to %12 step %c1 iter_args(%arg18 = %10) -> (tensor<64x64xf32>) {
                %c32 = arith.constant 32 : index
                %20 = arith.muli %arg17, %c32 : index
                %21 = arith.muli %11, %arg6 : index
                %22 = affine.apply affine_map<(d0)[s0] -> (d0 * 32 + s0)>(%arg17)[%21]
                %reinterpret_cast_0 = memref.reinterpret_cast %arg0 to offset: [%22], sizes: [64, 32], strides: [%arg6, 1] {tmd.reuse = {sequential = [{depth = 5 : i64, iterator = "%arg17", reuse_type = "no_reuse", volume = 0 : i64}], spatial = [{depth = 3 : i64, iterator = "%arg15", mapped_to = "x", reuse_type = "total_reuse", volume = 8192 : i64}, {depth = 4 : i64, iterator = "%arg16", mapped_to = "y", reuse_type = "total_reuse", volume = 8192 : i64}], temporal = [{depth = 0 : i64, iterator = "%arg12", reuse_type = "total_reuse", volume = 8192 : i64}, {depth = 1 : i64, iterator = "%arg13", reuse_type = "total_reuse", volume = 8192 : i64}, {depth = 2 : i64, iterator = "%arg14", reuse_type = "no_reuse", volume = 0 : i64}]}} : memref<*xf32> to memref<64x32xf32, strided<[?, 1], offset: ?>>
                %alloc = memref.alloc() : memref<64x32xf32>
                memref.copy %reinterpret_cast_0, %alloc : memref<64x32xf32, strided<[?, 1], offset: ?>> to memref<64x32xf32>
                %23 = bufferization.to_tensor %alloc restrict writable : memref<64x32xf32> to tensor<64x32xf32>
                %24 = arith.muli %20, %arg7 : index
                %25 = affine.apply affine_map<(d0, d1)[s0] -> (d0 * 512 + d1 * 64 + s0)>(%arg12, %arg16)[%24]
                %reinterpret_cast_1 = memref.reinterpret_cast %arg1 to offset: [%25], sizes: [32, 64], strides: [%arg7, 1] {tmd.reuse = {sequential = [{depth = 5 : i64, iterator = "%arg17", reuse_type = "no_reuse", volume = 0 : i64}], spatial = [{depth = 3 : i64, iterator = "%arg15", mapped_to = "x", reuse_type = "total_reuse", volume = 8192 : i64}, {depth = 4 : i64, iterator = "%arg16", mapped_to = "y", reuse_type = "no_reuse", volume = 0 : i64}], temporal = [{depth = 0 : i64, iterator = "%arg12", reuse_type = "no_reuse", volume = 0 : i64}, {depth = 1 : i64, iterator = "%arg13", reuse_type = "total_reuse", volume = 8192 : i64}, {depth = 2 : i64, iterator = "%arg14", reuse_type = "total_reuse", volume = 8192 : i64}]}} : memref<*xf32> to memref<32x64xf32, strided<[?, 1], offset: ?>>
                %alloc_2 = memref.alloc() : memref<32x64xf32>
                memref.copy %reinterpret_cast_1, %alloc_2 : memref<32x64xf32, strided<[?, 1], offset: ?>> to memref<32x64xf32>
                %26 = bufferization.to_tensor %alloc_2 restrict writable : memref<32x64xf32> to tensor<32x64xf32>
                %27 = linalg.matmul ins(%23, %26 : tensor<64x32xf32>, tensor<32x64xf32>) outs(%10 : tensor<64x64xf32>) -> tensor<64x64xf32>
                %28 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%arg18, %27 : tensor<64x64xf32>, tensor<64x64xf32>) outs(%arg18 : tensor<64x64xf32>) {
                ^bb0(%in: f32, %in_3: f32, %out: f32):
                  %29 = arith.addf %in, %in_3 : f32
                  linalg.yield %29 : f32
                } -> tensor<64x64xf32>
                scf.yield %28 : tensor<64x64xf32>
              }
              %14 = arith.muli %11, %arg8 : index
              %15 = affine.apply affine_map<(d0, d1)[s0] -> (d0 * 512 + d1 * 64 + s0)>(%arg12, %arg16)[%14]
              %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%15], sizes: [64, 64], strides: [%arg8, 1] {tmd.reuse = {spatial = [{depth = 3 : i64, iterator = "%arg15", mapped_to = "x", reuse_type = "total_reuse", volume = 16384 : i64}, {depth = 4 : i64, iterator = "%arg16", mapped_to = "y", reuse_type = "no_reuse", volume = 0 : i64}], temporal = [{depth = 0 : i64, iterator = "%arg12", reuse_type = "no_reuse", volume = 0 : i64}, {depth = 1 : i64, iterator = "%arg13", reuse_type = "total_reuse", volume = 16384 : i64}, {depth = 2 : i64, iterator = "%arg14", reuse_type = "no_reuse", volume = 0 : i64}]}} : memref<*xf32> to memref<64x64xf32, strided<[?, 1], offset: ?>>
              %16 = affine.min affine_map<(d0, d1, d2)[s0] -> (d0 * -64 + s0, 64)>(%arg14, %8, %7)[%arg3]
              %17 = affine.min affine_map<(d0, d1, d2)[s0] -> (d1 * -64 + s0, 64)>(%arg14, %8, %7)[%arg4]
              %extracted_slice = tensor.extract_slice %13[0, 0] [%16, %17] [1, 1] : tensor<64x64xf32> to tensor<?x?xf32>
              %18 = affine.min affine_map<(d0, d1, d2)[s0] -> (d0 * -64 + s0, 64)>(%arg14, %8, %7)[%arg3]
              %19 = affine.min affine_map<(d0, d1, d2)[s0] -> (d1 * -64 + s0, 64)>(%arg14, %8, %7)[%arg4]
              %subview = memref.subview %reinterpret_cast[0, 0] [%18, %19] [1, 1] : memref<64x64xf32, strided<[?, 1], offset: ?>> to memref<?x?xf32, strided<[?, 1], offset: ?>>
              bufferization.materialize_in_destination %extracted_slice in writable %subview : (tensor<?x?xf32>, memref<?x?xf32, strided<[?, 1], offset: ?>>) -> ()
            } {tmd.mapped_to = "y"}
          } {tmd.mapped_to = "x"}
        }
      }
    }
    return
  }
  func.func @matmul_kernel__d1i1_d0i2_f2_f0_f1(%arg0: memref<*xf32> {tt.divisibility = 16 : i32}, %arg1: memref<*xf32> {tt.divisibility = 16 : i32}, %arg2: memref<*xf32> {tt.divisibility = 16 : i32}, %arg3: index {tt.divisibility = 16 : i32}, %arg4: index {tt.divisibility = 16 : i32}, %arg5: index {tt.divisibility = 16 : i32}, %arg6: index {tt.divisibility = 16 : i32}, %arg7: index {tt.divisibility = 16 : i32}, %arg8: index {tt.divisibility = 16 : i32}, %arg9: index, %arg10: index, %arg11: index) {
    affine.for %arg12 = 0 to affine_map<(d0, d1, d2) -> (d2 ceildiv 8)>(%arg9, %arg10, %arg11) {
      affine.for %arg13 = 0 to affine_map<(d0, d1, d2) -> (d0)>(%arg9, %arg10, %arg11) {
        affine.for %arg14 = 0 to affine_map<(d0, d1, d2) -> (d1 ceildiv 8)>(%arg9, %arg10, %arg11) {
          affine.parallel (%arg15) = (0) to (8) {
            %7 = affine.apply affine_map<(d0, d1) -> (d0 * 8 + d1)>(%arg12, %arg15)
            affine.parallel (%arg16) = (0) to (8) {
              %8 = affine.apply affine_map<(d0, d1) -> (d0 * 8 + d1)>(%arg14, %arg16)
              %cst = arith.constant 0.000000e+00 : f32
              %9 = tensor.empty() : tensor<64x64xf32>
              %10 = linalg.fill ins(%cst : f32) outs(%9 : tensor<64x64xf32>) -> tensor<64x64xf32>
              %c64 = arith.constant 64 : index
              %11 = arith.muli %arg13, %c64 : index
              %12 = affine.apply affine_map<()[s0] -> ((s0 + 31) floordiv 32)>()[%arg5]
              %c0 = arith.constant 0 : index
              %c1 = arith.constant 1 : index
              %13 = scf.for %arg17 = %c0 to %12 step %c1 iter_args(%arg18 = %10) -> (tensor<64x64xf32>) {
                %c32 = arith.constant 32 : index
                %20 = arith.muli %arg17, %c32 : index
                %21 = arith.muli %11, %arg6 : index
                %22 = affine.apply affine_map<(d0)[s0] -> (d0 * 32 + s0)>(%arg17)[%21]
                %reinterpret_cast_0 = memref.reinterpret_cast %arg0 to offset: [%22], sizes: [64, 32], strides: [%arg6, 1] {tmd.reuse = {sequential = [{depth = 5 : i64, iterator = "%arg17", reuse_type = "no_reuse", volume = 0 : i64}], spatial = [{depth = 3 : i64, iterator = "%arg15", mapped_to = "x", reuse_type = "total_reuse", volume = 8192 : i64}, {depth = 4 : i64, iterator = "%arg16", mapped_to = "y", reuse_type = "total_reuse", volume = 8192 : i64}], temporal = [{depth = 0 : i64, iterator = "%arg12", reuse_type = "total_reuse", volume = 8192 : i64}, {depth = 1 : i64, iterator = "%arg13", reuse_type = "no_reuse", volume = 0 : i64}, {depth = 2 : i64, iterator = "%arg14", reuse_type = "total_reuse", volume = 8192 : i64}]}} : memref<*xf32> to memref<64x32xf32, strided<[?, 1], offset: ?>>
                %alloc = memref.alloc() : memref<64x32xf32>
                memref.copy %reinterpret_cast_0, %alloc : memref<64x32xf32, strided<[?, 1], offset: ?>> to memref<64x32xf32>
                %23 = bufferization.to_tensor %alloc restrict writable : memref<64x32xf32> to tensor<64x32xf32>
                %24 = arith.muli %20, %arg7 : index
                %25 = affine.apply affine_map<(d0, d1)[s0] -> (d0 * 512 + d1 * 64 + s0)>(%arg14, %arg16)[%24]
                %reinterpret_cast_1 = memref.reinterpret_cast %arg1 to offset: [%25], sizes: [32, 64], strides: [%arg7, 1] {tmd.reuse = {sequential = [{depth = 5 : i64, iterator = "%arg17", reuse_type = "no_reuse", volume = 0 : i64}], spatial = [{depth = 3 : i64, iterator = "%arg15", mapped_to = "x", reuse_type = "total_reuse", volume = 8192 : i64}, {depth = 4 : i64, iterator = "%arg16", mapped_to = "y", reuse_type = "no_reuse", volume = 0 : i64}], temporal = [{depth = 0 : i64, iterator = "%arg12", reuse_type = "total_reuse", volume = 8192 : i64}, {depth = 1 : i64, iterator = "%arg13", reuse_type = "total_reuse", volume = 8192 : i64}, {depth = 2 : i64, iterator = "%arg14", reuse_type = "no_reuse", volume = 0 : i64}]}} : memref<*xf32> to memref<32x64xf32, strided<[?, 1], offset: ?>>
                %alloc_2 = memref.alloc() : memref<32x64xf32>
                memref.copy %reinterpret_cast_1, %alloc_2 : memref<32x64xf32, strided<[?, 1], offset: ?>> to memref<32x64xf32>
                %26 = bufferization.to_tensor %alloc_2 restrict writable : memref<32x64xf32> to tensor<32x64xf32>
                %27 = linalg.matmul ins(%23, %26 : tensor<64x32xf32>, tensor<32x64xf32>) outs(%10 : tensor<64x64xf32>) -> tensor<64x64xf32>
                %28 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%arg18, %27 : tensor<64x64xf32>, tensor<64x64xf32>) outs(%arg18 : tensor<64x64xf32>) {
                ^bb0(%in: f32, %in_3: f32, %out: f32):
                  %29 = arith.addf %in, %in_3 : f32
                  linalg.yield %29 : f32
                } -> tensor<64x64xf32>
                scf.yield %28 : tensor<64x64xf32>
              }
              %14 = arith.muli %11, %arg8 : index
              %15 = affine.apply affine_map<(d0, d1)[s0] -> (d0 * 512 + d1 * 64 + s0)>(%arg14, %arg16)[%14]
              %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%15], sizes: [64, 64], strides: [%arg8, 1] {tmd.reuse = {spatial = [{depth = 3 : i64, iterator = "%arg15", mapped_to = "x", reuse_type = "total_reuse", volume = 16384 : i64}, {depth = 4 : i64, iterator = "%arg16", mapped_to = "y", reuse_type = "no_reuse", volume = 0 : i64}], temporal = [{depth = 0 : i64, iterator = "%arg12", reuse_type = "total_reuse", volume = 16384 : i64}, {depth = 1 : i64, iterator = "%arg13", reuse_type = "no_reuse", volume = 0 : i64}, {depth = 2 : i64, iterator = "%arg14", reuse_type = "no_reuse", volume = 0 : i64}]}} : memref<*xf32> to memref<64x64xf32, strided<[?, 1], offset: ?>>
              %16 = affine.min affine_map<(d0, d1, d2)[s0] -> (d0 * -64 + s0, 64)>(%arg13, %8, %7)[%arg3]
              %17 = affine.min affine_map<(d0, d1, d2)[s0] -> (d1 * -64 + s0, 64)>(%arg13, %8, %7)[%arg4]
              %extracted_slice = tensor.extract_slice %13[0, 0] [%16, %17] [1, 1] : tensor<64x64xf32> to tensor<?x?xf32>
              %18 = affine.min affine_map<(d0, d1, d2)[s0] -> (d0 * -64 + s0, 64)>(%arg13, %8, %7)[%arg3]
              %19 = affine.min affine_map<(d0, d1, d2)[s0] -> (d1 * -64 + s0, 64)>(%arg13, %8, %7)[%arg4]
              %subview = memref.subview %reinterpret_cast[0, 0] [%18, %19] [1, 1] : memref<64x64xf32, strided<[?, 1], offset: ?>> to memref<?x?xf32, strided<[?, 1], offset: ?>>
              bufferization.materialize_in_destination %extracted_slice in writable %subview : (tensor<?x?xf32>, memref<?x?xf32, strided<[?, 1], offset: ?>>) -> ()
            } {tmd.mapped_to = "y"}
          } {tmd.mapped_to = "x"}
        }
      }
    }
    return
  }
  func.func @matmul_kernel__d1i1_d0i2_f2_f1_f0(%arg0: memref<*xf32> {tt.divisibility = 16 : i32}, %arg1: memref<*xf32> {tt.divisibility = 16 : i32}, %arg2: memref<*xf32> {tt.divisibility = 16 : i32}, %arg3: index {tt.divisibility = 16 : i32}, %arg4: index {tt.divisibility = 16 : i32}, %arg5: index {tt.divisibility = 16 : i32}, %arg6: index {tt.divisibility = 16 : i32}, %arg7: index {tt.divisibility = 16 : i32}, %arg8: index {tt.divisibility = 16 : i32}, %arg9: index, %arg10: index, %arg11: index) {
    affine.for %arg12 = 0 to affine_map<(d0, d1, d2) -> (d2 ceildiv 8)>(%arg9, %arg10, %arg11) {
      affine.for %arg13 = 0 to affine_map<(d0, d1, d2) -> (d1 ceildiv 8)>(%arg9, %arg10, %arg11) {
        affine.for %arg14 = 0 to affine_map<(d0, d1, d2) -> (d0)>(%arg9, %arg10, %arg11) {
          affine.parallel (%arg15) = (0) to (8) {
            %7 = affine.apply affine_map<(d0, d1) -> (d0 * 8 + d1)>(%arg12, %arg15)
            affine.parallel (%arg16) = (0) to (8) {
              %8 = affine.apply affine_map<(d0, d1) -> (d0 * 8 + d1)>(%arg13, %arg16)
              %cst = arith.constant 0.000000e+00 : f32
              %9 = tensor.empty() : tensor<64x64xf32>
              %10 = linalg.fill ins(%cst : f32) outs(%9 : tensor<64x64xf32>) -> tensor<64x64xf32>
              %c64 = arith.constant 64 : index
              %11 = arith.muli %arg14, %c64 : index
              %12 = affine.apply affine_map<()[s0] -> ((s0 + 31) floordiv 32)>()[%arg5]
              %c0 = arith.constant 0 : index
              %c1 = arith.constant 1 : index
              %13 = scf.for %arg17 = %c0 to %12 step %c1 iter_args(%arg18 = %10) -> (tensor<64x64xf32>) {
                %c32 = arith.constant 32 : index
                %20 = arith.muli %arg17, %c32 : index
                %21 = arith.muli %11, %arg6 : index
                %22 = affine.apply affine_map<(d0)[s0] -> (d0 * 32 + s0)>(%arg17)[%21]
                %reinterpret_cast_0 = memref.reinterpret_cast %arg0 to offset: [%22], sizes: [64, 32], strides: [%arg6, 1] {tmd.reuse = {sequential = [{depth = 5 : i64, iterator = "%arg17", reuse_type = "no_reuse", volume = 0 : i64}], spatial = [{depth = 3 : i64, iterator = "%arg15", mapped_to = "x", reuse_type = "total_reuse", volume = 8192 : i64}, {depth = 4 : i64, iterator = "%arg16", mapped_to = "y", reuse_type = "total_reuse", volume = 8192 : i64}], temporal = [{depth = 0 : i64, iterator = "%arg12", reuse_type = "total_reuse", volume = 8192 : i64}, {depth = 1 : i64, iterator = "%arg13", reuse_type = "total_reuse", volume = 8192 : i64}, {depth = 2 : i64, iterator = "%arg14", reuse_type = "no_reuse", volume = 0 : i64}]}} : memref<*xf32> to memref<64x32xf32, strided<[?, 1], offset: ?>>
                %alloc = memref.alloc() : memref<64x32xf32>
                memref.copy %reinterpret_cast_0, %alloc : memref<64x32xf32, strided<[?, 1], offset: ?>> to memref<64x32xf32>
                %23 = bufferization.to_tensor %alloc restrict writable : memref<64x32xf32> to tensor<64x32xf32>
                %24 = arith.muli %20, %arg7 : index
                %25 = affine.apply affine_map<(d0, d1)[s0] -> (d0 * 512 + d1 * 64 + s0)>(%arg13, %arg16)[%24]
                %reinterpret_cast_1 = memref.reinterpret_cast %arg1 to offset: [%25], sizes: [32, 64], strides: [%arg7, 1] {tmd.reuse = {sequential = [{depth = 5 : i64, iterator = "%arg17", reuse_type = "no_reuse", volume = 0 : i64}], spatial = [{depth = 3 : i64, iterator = "%arg15", mapped_to = "x", reuse_type = "total_reuse", volume = 8192 : i64}, {depth = 4 : i64, iterator = "%arg16", mapped_to = "y", reuse_type = "no_reuse", volume = 0 : i64}], temporal = [{depth = 0 : i64, iterator = "%arg12", reuse_type = "total_reuse", volume = 8192 : i64}, {depth = 1 : i64, iterator = "%arg13", reuse_type = "no_reuse", volume = 0 : i64}, {depth = 2 : i64, iterator = "%arg14", reuse_type = "total_reuse", volume = 8192 : i64}]}} : memref<*xf32> to memref<32x64xf32, strided<[?, 1], offset: ?>>
                %alloc_2 = memref.alloc() : memref<32x64xf32>
                memref.copy %reinterpret_cast_1, %alloc_2 : memref<32x64xf32, strided<[?, 1], offset: ?>> to memref<32x64xf32>
                %26 = bufferization.to_tensor %alloc_2 restrict writable : memref<32x64xf32> to tensor<32x64xf32>
                %27 = linalg.matmul ins(%23, %26 : tensor<64x32xf32>, tensor<32x64xf32>) outs(%10 : tensor<64x64xf32>) -> tensor<64x64xf32>
                %28 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%arg18, %27 : tensor<64x64xf32>, tensor<64x64xf32>) outs(%arg18 : tensor<64x64xf32>) {
                ^bb0(%in: f32, %in_3: f32, %out: f32):
                  %29 = arith.addf %in, %in_3 : f32
                  linalg.yield %29 : f32
                } -> tensor<64x64xf32>
                scf.yield %28 : tensor<64x64xf32>
              }
              %14 = arith.muli %11, %arg8 : index
              %15 = affine.apply affine_map<(d0, d1)[s0] -> (d0 * 512 + d1 * 64 + s0)>(%arg13, %arg16)[%14]
              %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%15], sizes: [64, 64], strides: [%arg8, 1] {tmd.reuse = {spatial = [{depth = 3 : i64, iterator = "%arg15", mapped_to = "x", reuse_type = "total_reuse", volume = 16384 : i64}, {depth = 4 : i64, iterator = "%arg16", mapped_to = "y", reuse_type = "no_reuse", volume = 0 : i64}], temporal = [{depth = 0 : i64, iterator = "%arg12", reuse_type = "total_reuse", volume = 16384 : i64}, {depth = 1 : i64, iterator = "%arg13", reuse_type = "no_reuse", volume = 0 : i64}, {depth = 2 : i64, iterator = "%arg14", reuse_type = "no_reuse", volume = 0 : i64}]}} : memref<*xf32> to memref<64x64xf32, strided<[?, 1], offset: ?>>
              %16 = affine.min affine_map<(d0, d1, d2)[s0] -> (d0 * -64 + s0, 64)>(%arg14, %8, %7)[%arg3]
              %17 = affine.min affine_map<(d0, d1, d2)[s0] -> (d1 * -64 + s0, 64)>(%arg14, %8, %7)[%arg4]
              %extracted_slice = tensor.extract_slice %13[0, 0] [%16, %17] [1, 1] : tensor<64x64xf32> to tensor<?x?xf32>
              %18 = affine.min affine_map<(d0, d1, d2)[s0] -> (d0 * -64 + s0, 64)>(%arg14, %8, %7)[%arg3]
              %19 = affine.min affine_map<(d0, d1, d2)[s0] -> (d1 * -64 + s0, 64)>(%arg14, %8, %7)[%arg4]
              %subview = memref.subview %reinterpret_cast[0, 0] [%18, %19] [1, 1] : memref<64x64xf32, strided<[?, 1], offset: ?>> to memref<?x?xf32, strided<[?, 1], offset: ?>>
              bufferization.materialize_in_destination %extracted_slice in writable %subview : (tensor<?x?xf32>, memref<?x?xf32, strided<[?, 1], offset: ?>>) -> ()
            } {tmd.mapped_to = "y"}
          } {tmd.mapped_to = "x"}
        }
      }
    }
    return
  }
  func.func @matmul_kernel__d0i2_d1i2_f0_f1_f2(%arg0: memref<*xf32> {tt.divisibility = 16 : i32}, %arg1: memref<*xf32> {tt.divisibility = 16 : i32}, %arg2: memref<*xf32> {tt.divisibility = 16 : i32}, %arg3: index {tt.divisibility = 16 : i32}, %arg4: index {tt.divisibility = 16 : i32}, %arg5: index {tt.divisibility = 16 : i32}, %arg6: index {tt.divisibility = 16 : i32}, %arg7: index {tt.divisibility = 16 : i32}, %arg8: index {tt.divisibility = 16 : i32}, %arg9: index, %arg10: index, %arg11: index) {
    affine.for %arg12 = 0 to affine_map<(d0, d1, d2) -> (d0)>(%arg9, %arg10, %arg11) {
      affine.for %arg13 = 0 to affine_map<(d0, d1, d2) -> (d1)>(%arg9, %arg10, %arg11) {
        affine.for %arg14 = 0 to affine_map<(d0, d1, d2) -> ((d2 ceildiv 8) ceildiv 8)>(%arg9, %arg10, %arg11) {
          affine.parallel (%arg15) = (0) to (8) {
            affine.parallel (%arg16) = (0) to (8) {
              %7 = affine.apply affine_map<(d0, d1, d2) -> (d0 + d1 * 64 + d2 * 8)>(%arg16, %arg14, %arg15)
              %cst = arith.constant 0.000000e+00 : f32
              %8 = tensor.empty() : tensor<64x64xf32>
              %9 = linalg.fill ins(%cst : f32) outs(%8 : tensor<64x64xf32>) -> tensor<64x64xf32>
              %c64 = arith.constant 64 : index
              %10 = arith.muli %arg12, %c64 : index
              %11 = affine.apply affine_map<()[s0] -> ((s0 + 31) floordiv 32)>()[%arg5]
              %c0 = arith.constant 0 : index
              %c1 = arith.constant 1 : index
              %12 = scf.for %arg17 = %c0 to %11 step %c1 iter_args(%arg18 = %9) -> (tensor<64x64xf32>) {
                %c32 = arith.constant 32 : index
                %19 = arith.muli %arg17, %c32 : index
                %20 = arith.muli %10, %arg6 : index
                %21 = affine.apply affine_map<(d0)[s0] -> (d0 * 32 + s0)>(%arg17)[%20]
                %reinterpret_cast_0 = memref.reinterpret_cast %arg0 to offset: [%21], sizes: [64, 32], strides: [%arg6, 1] {tmd.reuse = {sequential = [{depth = 5 : i64, iterator = "%arg17", reuse_type = "no_reuse", volume = 0 : i64}], spatial = [{depth = 3 : i64, iterator = "%arg15", mapped_to = "y", reuse_type = "total_reuse", volume = 8192 : i64}, {depth = 4 : i64, iterator = "%arg16", mapped_to = "x", reuse_type = "total_reuse", volume = 8192 : i64}], temporal = [{depth = 0 : i64, iterator = "%arg12", reuse_type = "no_reuse", volume = 0 : i64}, {depth = 1 : i64, iterator = "%arg13", reuse_type = "total_reuse", volume = 8192 : i64}, {depth = 2 : i64, iterator = "%arg14", reuse_type = "total_reuse", volume = 8192 : i64}]}} : memref<*xf32> to memref<64x32xf32, strided<[?, 1], offset: ?>>
                %alloc = memref.alloc() : memref<64x32xf32>
                memref.copy %reinterpret_cast_0, %alloc : memref<64x32xf32, strided<[?, 1], offset: ?>> to memref<64x32xf32>
                %22 = bufferization.to_tensor %alloc restrict writable : memref<64x32xf32> to tensor<64x32xf32>
                %23 = arith.muli %19, %arg7 : index
                %24 = affine.apply affine_map<(d0)[s0] -> (d0 * 64 + s0)>(%arg13)[%23]
                %reinterpret_cast_1 = memref.reinterpret_cast %arg1 to offset: [%24], sizes: [32, 64], strides: [%arg7, 1] {tmd.reuse = {sequential = [{depth = 5 : i64, iterator = "%arg17", reuse_type = "no_reuse", volume = 0 : i64}], spatial = [{depth = 3 : i64, iterator = "%arg15", mapped_to = "y", reuse_type = "total_reuse", volume = 8192 : i64}, {depth = 4 : i64, iterator = "%arg16", mapped_to = "x", reuse_type = "total_reuse", volume = 8192 : i64}], temporal = [{depth = 0 : i64, iterator = "%arg12", reuse_type = "total_reuse", volume = 8192 : i64}, {depth = 1 : i64, iterator = "%arg13", reuse_type = "no_reuse", volume = 0 : i64}, {depth = 2 : i64, iterator = "%arg14", reuse_type = "total_reuse", volume = 8192 : i64}]}} : memref<*xf32> to memref<32x64xf32, strided<[?, 1], offset: ?>>
                %alloc_2 = memref.alloc() : memref<32x64xf32>
                memref.copy %reinterpret_cast_1, %alloc_2 : memref<32x64xf32, strided<[?, 1], offset: ?>> to memref<32x64xf32>
                %25 = bufferization.to_tensor %alloc_2 restrict writable : memref<32x64xf32> to tensor<32x64xf32>
                %26 = linalg.matmul ins(%22, %25 : tensor<64x32xf32>, tensor<32x64xf32>) outs(%9 : tensor<64x64xf32>) -> tensor<64x64xf32>
                %27 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%arg18, %26 : tensor<64x64xf32>, tensor<64x64xf32>) outs(%arg18 : tensor<64x64xf32>) {
                ^bb0(%in: f32, %in_3: f32, %out: f32):
                  %28 = arith.addf %in, %in_3 : f32
                  linalg.yield %28 : f32
                } -> tensor<64x64xf32>
                scf.yield %27 : tensor<64x64xf32>
              }
              %13 = arith.muli %10, %arg8 : index
              %14 = affine.apply affine_map<(d0)[s0] -> (d0 * 64 + s0)>(%arg13)[%13]
              %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%14], sizes: [64, 64], strides: [%arg8, 1] {tmd.reuse = {spatial = [{depth = 3 : i64, iterator = "%arg15", mapped_to = "y", reuse_type = "total_reuse", volume = 16384 : i64}, {depth = 4 : i64, iterator = "%arg16", mapped_to = "x", reuse_type = "total_reuse", volume = 16384 : i64}], temporal = [{depth = 0 : i64, iterator = "%arg12", reuse_type = "no_reuse", volume = 0 : i64}, {depth = 1 : i64, iterator = "%arg13", reuse_type = "no_reuse", volume = 0 : i64}, {depth = 2 : i64, iterator = "%arg14", reuse_type = "total_reuse", volume = 16384 : i64}]}} : memref<*xf32> to memref<64x64xf32, strided<[?, 1], offset: ?>>
              %15 = affine.min affine_map<(d0, d1, d2)[s0] -> (d0 * -64 + s0, 64)>(%arg12, %arg13, %7)[%arg3]
              %16 = affine.min affine_map<(d0, d1, d2)[s0] -> (d1 * -64 + s0, 64)>(%arg12, %arg13, %7)[%arg4]
              %extracted_slice = tensor.extract_slice %12[0, 0] [%15, %16] [1, 1] : tensor<64x64xf32> to tensor<?x?xf32>
              %17 = affine.min affine_map<(d0, d1, d2)[s0] -> (d0 * -64 + s0, 64)>(%arg12, %arg13, %7)[%arg3]
              %18 = affine.min affine_map<(d0, d1, d2)[s0] -> (d1 * -64 + s0, 64)>(%arg12, %arg13, %7)[%arg4]
              %subview = memref.subview %reinterpret_cast[0, 0] [%17, %18] [1, 1] : memref<64x64xf32, strided<[?, 1], offset: ?>> to memref<?x?xf32, strided<[?, 1], offset: ?>>
              bufferization.materialize_in_destination %extracted_slice in writable %subview : (tensor<?x?xf32>, memref<?x?xf32, strided<[?, 1], offset: ?>>) -> ()
            } {tmd.mapped_to = "x"}
          } {tmd.mapped_to = "y"}
        }
      }
    }
    return
  }
  func.func @matmul_kernel__d0i2_d1i2_f0_f2_f1(%arg0: memref<*xf32> {tt.divisibility = 16 : i32}, %arg1: memref<*xf32> {tt.divisibility = 16 : i32}, %arg2: memref<*xf32> {tt.divisibility = 16 : i32}, %arg3: index {tt.divisibility = 16 : i32}, %arg4: index {tt.divisibility = 16 : i32}, %arg5: index {tt.divisibility = 16 : i32}, %arg6: index {tt.divisibility = 16 : i32}, %arg7: index {tt.divisibility = 16 : i32}, %arg8: index {tt.divisibility = 16 : i32}, %arg9: index, %arg10: index, %arg11: index) {
    affine.for %arg12 = 0 to affine_map<(d0, d1, d2) -> (d0)>(%arg9, %arg10, %arg11) {
      affine.for %arg13 = 0 to affine_map<(d0, d1, d2) -> ((d2 ceildiv 8) ceildiv 8)>(%arg9, %arg10, %arg11) {
        affine.for %arg14 = 0 to affine_map<(d0, d1, d2) -> (d1)>(%arg9, %arg10, %arg11) {
          affine.parallel (%arg15) = (0) to (8) {
            affine.parallel (%arg16) = (0) to (8) {
              %7 = affine.apply affine_map<(d0, d1, d2) -> (d0 + d1 * 64 + d2 * 8)>(%arg16, %arg13, %arg15)
              %cst = arith.constant 0.000000e+00 : f32
              %8 = tensor.empty() : tensor<64x64xf32>
              %9 = linalg.fill ins(%cst : f32) outs(%8 : tensor<64x64xf32>) -> tensor<64x64xf32>
              %c64 = arith.constant 64 : index
              %10 = arith.muli %arg12, %c64 : index
              %11 = affine.apply affine_map<()[s0] -> ((s0 + 31) floordiv 32)>()[%arg5]
              %c0 = arith.constant 0 : index
              %c1 = arith.constant 1 : index
              %12 = scf.for %arg17 = %c0 to %11 step %c1 iter_args(%arg18 = %9) -> (tensor<64x64xf32>) {
                %c32 = arith.constant 32 : index
                %19 = arith.muli %arg17, %c32 : index
                %20 = arith.muli %10, %arg6 : index
                %21 = affine.apply affine_map<(d0)[s0] -> (d0 * 32 + s0)>(%arg17)[%20]
                %reinterpret_cast_0 = memref.reinterpret_cast %arg0 to offset: [%21], sizes: [64, 32], strides: [%arg6, 1] {tmd.reuse = {sequential = [{depth = 5 : i64, iterator = "%arg17", reuse_type = "no_reuse", volume = 0 : i64}], spatial = [{depth = 3 : i64, iterator = "%arg15", mapped_to = "y", reuse_type = "total_reuse", volume = 8192 : i64}, {depth = 4 : i64, iterator = "%arg16", mapped_to = "x", reuse_type = "total_reuse", volume = 8192 : i64}], temporal = [{depth = 0 : i64, iterator = "%arg12", reuse_type = "no_reuse", volume = 0 : i64}, {depth = 1 : i64, iterator = "%arg13", reuse_type = "total_reuse", volume = 8192 : i64}, {depth = 2 : i64, iterator = "%arg14", reuse_type = "total_reuse", volume = 8192 : i64}]}} : memref<*xf32> to memref<64x32xf32, strided<[?, 1], offset: ?>>
                %alloc = memref.alloc() : memref<64x32xf32>
                memref.copy %reinterpret_cast_0, %alloc : memref<64x32xf32, strided<[?, 1], offset: ?>> to memref<64x32xf32>
                %22 = bufferization.to_tensor %alloc restrict writable : memref<64x32xf32> to tensor<64x32xf32>
                %23 = arith.muli %19, %arg7 : index
                %24 = affine.apply affine_map<(d0)[s0] -> (d0 * 64 + s0)>(%arg14)[%23]
                %reinterpret_cast_1 = memref.reinterpret_cast %arg1 to offset: [%24], sizes: [32, 64], strides: [%arg7, 1] {tmd.reuse = {sequential = [{depth = 5 : i64, iterator = "%arg17", reuse_type = "no_reuse", volume = 0 : i64}], spatial = [{depth = 3 : i64, iterator = "%arg15", mapped_to = "y", reuse_type = "total_reuse", volume = 8192 : i64}, {depth = 4 : i64, iterator = "%arg16", mapped_to = "x", reuse_type = "total_reuse", volume = 8192 : i64}], temporal = [{depth = 0 : i64, iterator = "%arg12", reuse_type = "total_reuse", volume = 8192 : i64}, {depth = 1 : i64, iterator = "%arg13", reuse_type = "total_reuse", volume = 8192 : i64}, {depth = 2 : i64, iterator = "%arg14", reuse_type = "no_reuse", volume = 0 : i64}]}} : memref<*xf32> to memref<32x64xf32, strided<[?, 1], offset: ?>>
                %alloc_2 = memref.alloc() : memref<32x64xf32>
                memref.copy %reinterpret_cast_1, %alloc_2 : memref<32x64xf32, strided<[?, 1], offset: ?>> to memref<32x64xf32>
                %25 = bufferization.to_tensor %alloc_2 restrict writable : memref<32x64xf32> to tensor<32x64xf32>
                %26 = linalg.matmul ins(%22, %25 : tensor<64x32xf32>, tensor<32x64xf32>) outs(%9 : tensor<64x64xf32>) -> tensor<64x64xf32>
                %27 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%arg18, %26 : tensor<64x64xf32>, tensor<64x64xf32>) outs(%arg18 : tensor<64x64xf32>) {
                ^bb0(%in: f32, %in_3: f32, %out: f32):
                  %28 = arith.addf %in, %in_3 : f32
                  linalg.yield %28 : f32
                } -> tensor<64x64xf32>
                scf.yield %27 : tensor<64x64xf32>
              }
              %13 = arith.muli %10, %arg8 : index
              %14 = affine.apply affine_map<(d0)[s0] -> (d0 * 64 + s0)>(%arg14)[%13]
              %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%14], sizes: [64, 64], strides: [%arg8, 1] {tmd.reuse = {spatial = [{depth = 3 : i64, iterator = "%arg15", mapped_to = "y", reuse_type = "total_reuse", volume = 16384 : i64}, {depth = 4 : i64, iterator = "%arg16", mapped_to = "x", reuse_type = "total_reuse", volume = 16384 : i64}], temporal = [{depth = 0 : i64, iterator = "%arg12", reuse_type = "no_reuse", volume = 0 : i64}, {depth = 1 : i64, iterator = "%arg13", reuse_type = "total_reuse", volume = 16384 : i64}, {depth = 2 : i64, iterator = "%arg14", reuse_type = "no_reuse", volume = 0 : i64}]}} : memref<*xf32> to memref<64x64xf32, strided<[?, 1], offset: ?>>
              %15 = affine.min affine_map<(d0, d1, d2)[s0] -> (d0 * -64 + s0, 64)>(%arg12, %arg14, %7)[%arg3]
              %16 = affine.min affine_map<(d0, d1, d2)[s0] -> (d1 * -64 + s0, 64)>(%arg12, %arg14, %7)[%arg4]
              %extracted_slice = tensor.extract_slice %12[0, 0] [%15, %16] [1, 1] : tensor<64x64xf32> to tensor<?x?xf32>
              %17 = affine.min affine_map<(d0, d1, d2)[s0] -> (d0 * -64 + s0, 64)>(%arg12, %arg14, %7)[%arg3]
              %18 = affine.min affine_map<(d0, d1, d2)[s0] -> (d1 * -64 + s0, 64)>(%arg12, %arg14, %7)[%arg4]
              %subview = memref.subview %reinterpret_cast[0, 0] [%17, %18] [1, 1] : memref<64x64xf32, strided<[?, 1], offset: ?>> to memref<?x?xf32, strided<[?, 1], offset: ?>>
              bufferization.materialize_in_destination %extracted_slice in writable %subview : (tensor<?x?xf32>, memref<?x?xf32, strided<[?, 1], offset: ?>>) -> ()
            } {tmd.mapped_to = "x"}
          } {tmd.mapped_to = "y"}
        }
      }
    }
    return
  }
  func.func @matmul_kernel__d0i2_d1i2_f1_f0_f2(%arg0: memref<*xf32> {tt.divisibility = 16 : i32}, %arg1: memref<*xf32> {tt.divisibility = 16 : i32}, %arg2: memref<*xf32> {tt.divisibility = 16 : i32}, %arg3: index {tt.divisibility = 16 : i32}, %arg4: index {tt.divisibility = 16 : i32}, %arg5: index {tt.divisibility = 16 : i32}, %arg6: index {tt.divisibility = 16 : i32}, %arg7: index {tt.divisibility = 16 : i32}, %arg8: index {tt.divisibility = 16 : i32}, %arg9: index, %arg10: index, %arg11: index) {
    affine.for %arg12 = 0 to affine_map<(d0, d1, d2) -> (d1)>(%arg9, %arg10, %arg11) {
      affine.for %arg13 = 0 to affine_map<(d0, d1, d2) -> (d0)>(%arg9, %arg10, %arg11) {
        affine.for %arg14 = 0 to affine_map<(d0, d1, d2) -> ((d2 ceildiv 8) ceildiv 8)>(%arg9, %arg10, %arg11) {
          affine.parallel (%arg15) = (0) to (8) {
            affine.parallel (%arg16) = (0) to (8) {
              %7 = affine.apply affine_map<(d0, d1, d2) -> (d0 + d1 * 64 + d2 * 8)>(%arg16, %arg14, %arg15)
              %cst = arith.constant 0.000000e+00 : f32
              %8 = tensor.empty() : tensor<64x64xf32>
              %9 = linalg.fill ins(%cst : f32) outs(%8 : tensor<64x64xf32>) -> tensor<64x64xf32>
              %c64 = arith.constant 64 : index
              %10 = arith.muli %arg13, %c64 : index
              %11 = affine.apply affine_map<()[s0] -> ((s0 + 31) floordiv 32)>()[%arg5]
              %c0 = arith.constant 0 : index
              %c1 = arith.constant 1 : index
              %12 = scf.for %arg17 = %c0 to %11 step %c1 iter_args(%arg18 = %9) -> (tensor<64x64xf32>) {
                %c32 = arith.constant 32 : index
                %19 = arith.muli %arg17, %c32 : index
                %20 = arith.muli %10, %arg6 : index
                %21 = affine.apply affine_map<(d0)[s0] -> (d0 * 32 + s0)>(%arg17)[%20]
                %reinterpret_cast_0 = memref.reinterpret_cast %arg0 to offset: [%21], sizes: [64, 32], strides: [%arg6, 1] {tmd.reuse = {sequential = [{depth = 5 : i64, iterator = "%arg17", reuse_type = "no_reuse", volume = 0 : i64}], spatial = [{depth = 3 : i64, iterator = "%arg15", mapped_to = "y", reuse_type = "total_reuse", volume = 8192 : i64}, {depth = 4 : i64, iterator = "%arg16", mapped_to = "x", reuse_type = "total_reuse", volume = 8192 : i64}], temporal = [{depth = 0 : i64, iterator = "%arg12", reuse_type = "total_reuse", volume = 8192 : i64}, {depth = 1 : i64, iterator = "%arg13", reuse_type = "no_reuse", volume = 0 : i64}, {depth = 2 : i64, iterator = "%arg14", reuse_type = "total_reuse", volume = 8192 : i64}]}} : memref<*xf32> to memref<64x32xf32, strided<[?, 1], offset: ?>>
                %alloc = memref.alloc() : memref<64x32xf32>
                memref.copy %reinterpret_cast_0, %alloc : memref<64x32xf32, strided<[?, 1], offset: ?>> to memref<64x32xf32>
                %22 = bufferization.to_tensor %alloc restrict writable : memref<64x32xf32> to tensor<64x32xf32>
                %23 = arith.muli %19, %arg7 : index
                %24 = affine.apply affine_map<(d0)[s0] -> (d0 * 64 + s0)>(%arg12)[%23]
                %reinterpret_cast_1 = memref.reinterpret_cast %arg1 to offset: [%24], sizes: [32, 64], strides: [%arg7, 1] {tmd.reuse = {sequential = [{depth = 5 : i64, iterator = "%arg17", reuse_type = "no_reuse", volume = 0 : i64}], spatial = [{depth = 3 : i64, iterator = "%arg15", mapped_to = "y", reuse_type = "total_reuse", volume = 8192 : i64}, {depth = 4 : i64, iterator = "%arg16", mapped_to = "x", reuse_type = "total_reuse", volume = 8192 : i64}], temporal = [{depth = 0 : i64, iterator = "%arg12", reuse_type = "no_reuse", volume = 0 : i64}, {depth = 1 : i64, iterator = "%arg13", reuse_type = "total_reuse", volume = 8192 : i64}, {depth = 2 : i64, iterator = "%arg14", reuse_type = "total_reuse", volume = 8192 : i64}]}} : memref<*xf32> to memref<32x64xf32, strided<[?, 1], offset: ?>>
                %alloc_2 = memref.alloc() : memref<32x64xf32>
                memref.copy %reinterpret_cast_1, %alloc_2 : memref<32x64xf32, strided<[?, 1], offset: ?>> to memref<32x64xf32>
                %25 = bufferization.to_tensor %alloc_2 restrict writable : memref<32x64xf32> to tensor<32x64xf32>
                %26 = linalg.matmul ins(%22, %25 : tensor<64x32xf32>, tensor<32x64xf32>) outs(%9 : tensor<64x64xf32>) -> tensor<64x64xf32>
                %27 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%arg18, %26 : tensor<64x64xf32>, tensor<64x64xf32>) outs(%arg18 : tensor<64x64xf32>) {
                ^bb0(%in: f32, %in_3: f32, %out: f32):
                  %28 = arith.addf %in, %in_3 : f32
                  linalg.yield %28 : f32
                } -> tensor<64x64xf32>
                scf.yield %27 : tensor<64x64xf32>
              }
              %13 = arith.muli %10, %arg8 : index
              %14 = affine.apply affine_map<(d0)[s0] -> (d0 * 64 + s0)>(%arg12)[%13]
              %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%14], sizes: [64, 64], strides: [%arg8, 1] {tmd.reuse = {spatial = [{depth = 3 : i64, iterator = "%arg15", mapped_to = "y", reuse_type = "total_reuse", volume = 16384 : i64}, {depth = 4 : i64, iterator = "%arg16", mapped_to = "x", reuse_type = "total_reuse", volume = 16384 : i64}], temporal = [{depth = 0 : i64, iterator = "%arg12", reuse_type = "no_reuse", volume = 0 : i64}, {depth = 1 : i64, iterator = "%arg13", reuse_type = "no_reuse", volume = 0 : i64}, {depth = 2 : i64, iterator = "%arg14", reuse_type = "total_reuse", volume = 16384 : i64}]}} : memref<*xf32> to memref<64x64xf32, strided<[?, 1], offset: ?>>
              %15 = affine.min affine_map<(d0, d1, d2)[s0] -> (d0 * -64 + s0, 64)>(%arg13, %arg12, %7)[%arg3]
              %16 = affine.min affine_map<(d0, d1, d2)[s0] -> (d1 * -64 + s0, 64)>(%arg13, %arg12, %7)[%arg4]
              %extracted_slice = tensor.extract_slice %12[0, 0] [%15, %16] [1, 1] : tensor<64x64xf32> to tensor<?x?xf32>
              %17 = affine.min affine_map<(d0, d1, d2)[s0] -> (d0 * -64 + s0, 64)>(%arg13, %arg12, %7)[%arg3]
              %18 = affine.min affine_map<(d0, d1, d2)[s0] -> (d1 * -64 + s0, 64)>(%arg13, %arg12, %7)[%arg4]
              %subview = memref.subview %reinterpret_cast[0, 0] [%17, %18] [1, 1] : memref<64x64xf32, strided<[?, 1], offset: ?>> to memref<?x?xf32, strided<[?, 1], offset: ?>>
              bufferization.materialize_in_destination %extracted_slice in writable %subview : (tensor<?x?xf32>, memref<?x?xf32, strided<[?, 1], offset: ?>>) -> ()
            } {tmd.mapped_to = "x"}
          } {tmd.mapped_to = "y"}
        }
      }
    }
    return
  }
  func.func @matmul_kernel__d0i2_d1i2_f1_f2_f0(%arg0: memref<*xf32> {tt.divisibility = 16 : i32}, %arg1: memref<*xf32> {tt.divisibility = 16 : i32}, %arg2: memref<*xf32> {tt.divisibility = 16 : i32}, %arg3: index {tt.divisibility = 16 : i32}, %arg4: index {tt.divisibility = 16 : i32}, %arg5: index {tt.divisibility = 16 : i32}, %arg6: index {tt.divisibility = 16 : i32}, %arg7: index {tt.divisibility = 16 : i32}, %arg8: index {tt.divisibility = 16 : i32}, %arg9: index, %arg10: index, %arg11: index) {
    affine.for %arg12 = 0 to affine_map<(d0, d1, d2) -> (d1)>(%arg9, %arg10, %arg11) {
      affine.for %arg13 = 0 to affine_map<(d0, d1, d2) -> ((d2 ceildiv 8) ceildiv 8)>(%arg9, %arg10, %arg11) {
        affine.for %arg14 = 0 to affine_map<(d0, d1, d2) -> (d0)>(%arg9, %arg10, %arg11) {
          affine.parallel (%arg15) = (0) to (8) {
            affine.parallel (%arg16) = (0) to (8) {
              %7 = affine.apply affine_map<(d0, d1, d2) -> (d0 + d1 * 64 + d2 * 8)>(%arg16, %arg13, %arg15)
              %cst = arith.constant 0.000000e+00 : f32
              %8 = tensor.empty() : tensor<64x64xf32>
              %9 = linalg.fill ins(%cst : f32) outs(%8 : tensor<64x64xf32>) -> tensor<64x64xf32>
              %c64 = arith.constant 64 : index
              %10 = arith.muli %arg14, %c64 : index
              %11 = affine.apply affine_map<()[s0] -> ((s0 + 31) floordiv 32)>()[%arg5]
              %c0 = arith.constant 0 : index
              %c1 = arith.constant 1 : index
              %12 = scf.for %arg17 = %c0 to %11 step %c1 iter_args(%arg18 = %9) -> (tensor<64x64xf32>) {
                %c32 = arith.constant 32 : index
                %19 = arith.muli %arg17, %c32 : index
                %20 = arith.muli %10, %arg6 : index
                %21 = affine.apply affine_map<(d0)[s0] -> (d0 * 32 + s0)>(%arg17)[%20]
                %reinterpret_cast_0 = memref.reinterpret_cast %arg0 to offset: [%21], sizes: [64, 32], strides: [%arg6, 1] {tmd.reuse = {sequential = [{depth = 5 : i64, iterator = "%arg17", reuse_type = "no_reuse", volume = 0 : i64}], spatial = [{depth = 3 : i64, iterator = "%arg15", mapped_to = "y", reuse_type = "total_reuse", volume = 8192 : i64}, {depth = 4 : i64, iterator = "%arg16", mapped_to = "x", reuse_type = "total_reuse", volume = 8192 : i64}], temporal = [{depth = 0 : i64, iterator = "%arg12", reuse_type = "total_reuse", volume = 8192 : i64}, {depth = 1 : i64, iterator = "%arg13", reuse_type = "total_reuse", volume = 8192 : i64}, {depth = 2 : i64, iterator = "%arg14", reuse_type = "no_reuse", volume = 0 : i64}]}} : memref<*xf32> to memref<64x32xf32, strided<[?, 1], offset: ?>>
                %alloc = memref.alloc() : memref<64x32xf32>
                memref.copy %reinterpret_cast_0, %alloc : memref<64x32xf32, strided<[?, 1], offset: ?>> to memref<64x32xf32>
                %22 = bufferization.to_tensor %alloc restrict writable : memref<64x32xf32> to tensor<64x32xf32>
                %23 = arith.muli %19, %arg7 : index
                %24 = affine.apply affine_map<(d0)[s0] -> (d0 * 64 + s0)>(%arg12)[%23]
                %reinterpret_cast_1 = memref.reinterpret_cast %arg1 to offset: [%24], sizes: [32, 64], strides: [%arg7, 1] {tmd.reuse = {sequential = [{depth = 5 : i64, iterator = "%arg17", reuse_type = "no_reuse", volume = 0 : i64}], spatial = [{depth = 3 : i64, iterator = "%arg15", mapped_to = "y", reuse_type = "total_reuse", volume = 8192 : i64}, {depth = 4 : i64, iterator = "%arg16", mapped_to = "x", reuse_type = "total_reuse", volume = 8192 : i64}], temporal = [{depth = 0 : i64, iterator = "%arg12", reuse_type = "no_reuse", volume = 0 : i64}, {depth = 1 : i64, iterator = "%arg13", reuse_type = "total_reuse", volume = 8192 : i64}, {depth = 2 : i64, iterator = "%arg14", reuse_type = "total_reuse", volume = 8192 : i64}]}} : memref<*xf32> to memref<32x64xf32, strided<[?, 1], offset: ?>>
                %alloc_2 = memref.alloc() : memref<32x64xf32>
                memref.copy %reinterpret_cast_1, %alloc_2 : memref<32x64xf32, strided<[?, 1], offset: ?>> to memref<32x64xf32>
                %25 = bufferization.to_tensor %alloc_2 restrict writable : memref<32x64xf32> to tensor<32x64xf32>
                %26 = linalg.matmul ins(%22, %25 : tensor<64x32xf32>, tensor<32x64xf32>) outs(%9 : tensor<64x64xf32>) -> tensor<64x64xf32>
                %27 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%arg18, %26 : tensor<64x64xf32>, tensor<64x64xf32>) outs(%arg18 : tensor<64x64xf32>) {
                ^bb0(%in: f32, %in_3: f32, %out: f32):
                  %28 = arith.addf %in, %in_3 : f32
                  linalg.yield %28 : f32
                } -> tensor<64x64xf32>
                scf.yield %27 : tensor<64x64xf32>
              }
              %13 = arith.muli %10, %arg8 : index
              %14 = affine.apply affine_map<(d0)[s0] -> (d0 * 64 + s0)>(%arg12)[%13]
              %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%14], sizes: [64, 64], strides: [%arg8, 1] {tmd.reuse = {spatial = [{depth = 3 : i64, iterator = "%arg15", mapped_to = "y", reuse_type = "total_reuse", volume = 16384 : i64}, {depth = 4 : i64, iterator = "%arg16", mapped_to = "x", reuse_type = "total_reuse", volume = 16384 : i64}], temporal = [{depth = 0 : i64, iterator = "%arg12", reuse_type = "no_reuse", volume = 0 : i64}, {depth = 1 : i64, iterator = "%arg13", reuse_type = "total_reuse", volume = 16384 : i64}, {depth = 2 : i64, iterator = "%arg14", reuse_type = "no_reuse", volume = 0 : i64}]}} : memref<*xf32> to memref<64x64xf32, strided<[?, 1], offset: ?>>
              %15 = affine.min affine_map<(d0, d1, d2)[s0] -> (d0 * -64 + s0, 64)>(%arg14, %arg12, %7)[%arg3]
              %16 = affine.min affine_map<(d0, d1, d2)[s0] -> (d1 * -64 + s0, 64)>(%arg14, %arg12, %7)[%arg4]
              %extracted_slice = tensor.extract_slice %12[0, 0] [%15, %16] [1, 1] : tensor<64x64xf32> to tensor<?x?xf32>
              %17 = affine.min affine_map<(d0, d1, d2)[s0] -> (d0 * -64 + s0, 64)>(%arg14, %arg12, %7)[%arg3]
              %18 = affine.min affine_map<(d0, d1, d2)[s0] -> (d1 * -64 + s0, 64)>(%arg14, %arg12, %7)[%arg4]
              %subview = memref.subview %reinterpret_cast[0, 0] [%17, %18] [1, 1] : memref<64x64xf32, strided<[?, 1], offset: ?>> to memref<?x?xf32, strided<[?, 1], offset: ?>>
              bufferization.materialize_in_destination %extracted_slice in writable %subview : (tensor<?x?xf32>, memref<?x?xf32, strided<[?, 1], offset: ?>>) -> ()
            } {tmd.mapped_to = "x"}
          } {tmd.mapped_to = "y"}
        }
      }
    }
    return
  }
  func.func @matmul_kernel__d0i2_d1i2_f2_f0_f1(%arg0: memref<*xf32> {tt.divisibility = 16 : i32}, %arg1: memref<*xf32> {tt.divisibility = 16 : i32}, %arg2: memref<*xf32> {tt.divisibility = 16 : i32}, %arg3: index {tt.divisibility = 16 : i32}, %arg4: index {tt.divisibility = 16 : i32}, %arg5: index {tt.divisibility = 16 : i32}, %arg6: index {tt.divisibility = 16 : i32}, %arg7: index {tt.divisibility = 16 : i32}, %arg8: index {tt.divisibility = 16 : i32}, %arg9: index, %arg10: index, %arg11: index) {
    affine.for %arg12 = 0 to affine_map<(d0, d1, d2) -> ((d2 ceildiv 8) ceildiv 8)>(%arg9, %arg10, %arg11) {
      affine.for %arg13 = 0 to affine_map<(d0, d1, d2) -> (d0)>(%arg9, %arg10, %arg11) {
        affine.for %arg14 = 0 to affine_map<(d0, d1, d2) -> (d1)>(%arg9, %arg10, %arg11) {
          affine.parallel (%arg15) = (0) to (8) {
            affine.parallel (%arg16) = (0) to (8) {
              %7 = affine.apply affine_map<(d0, d1, d2) -> (d0 + d1 * 64 + d2 * 8)>(%arg16, %arg12, %arg15)
              %cst = arith.constant 0.000000e+00 : f32
              %8 = tensor.empty() : tensor<64x64xf32>
              %9 = linalg.fill ins(%cst : f32) outs(%8 : tensor<64x64xf32>) -> tensor<64x64xf32>
              %c64 = arith.constant 64 : index
              %10 = arith.muli %arg13, %c64 : index
              %11 = affine.apply affine_map<()[s0] -> ((s0 + 31) floordiv 32)>()[%arg5]
              %c0 = arith.constant 0 : index
              %c1 = arith.constant 1 : index
              %12 = scf.for %arg17 = %c0 to %11 step %c1 iter_args(%arg18 = %9) -> (tensor<64x64xf32>) {
                %c32 = arith.constant 32 : index
                %19 = arith.muli %arg17, %c32 : index
                %20 = arith.muli %10, %arg6 : index
                %21 = affine.apply affine_map<(d0)[s0] -> (d0 * 32 + s0)>(%arg17)[%20]
                %reinterpret_cast_0 = memref.reinterpret_cast %arg0 to offset: [%21], sizes: [64, 32], strides: [%arg6, 1] {tmd.reuse = {sequential = [{depth = 5 : i64, iterator = "%arg17", reuse_type = "no_reuse", volume = 0 : i64}], spatial = [{depth = 3 : i64, iterator = "%arg15", mapped_to = "y", reuse_type = "total_reuse", volume = 8192 : i64}, {depth = 4 : i64, iterator = "%arg16", mapped_to = "x", reuse_type = "total_reuse", volume = 8192 : i64}], temporal = [{depth = 0 : i64, iterator = "%arg12", reuse_type = "total_reuse", volume = 8192 : i64}, {depth = 1 : i64, iterator = "%arg13", reuse_type = "no_reuse", volume = 0 : i64}, {depth = 2 : i64, iterator = "%arg14", reuse_type = "total_reuse", volume = 8192 : i64}]}} : memref<*xf32> to memref<64x32xf32, strided<[?, 1], offset: ?>>
                %alloc = memref.alloc() : memref<64x32xf32>
                memref.copy %reinterpret_cast_0, %alloc : memref<64x32xf32, strided<[?, 1], offset: ?>> to memref<64x32xf32>
                %22 = bufferization.to_tensor %alloc restrict writable : memref<64x32xf32> to tensor<64x32xf32>
                %23 = arith.muli %19, %arg7 : index
                %24 = affine.apply affine_map<(d0)[s0] -> (d0 * 64 + s0)>(%arg14)[%23]
                %reinterpret_cast_1 = memref.reinterpret_cast %arg1 to offset: [%24], sizes: [32, 64], strides: [%arg7, 1] {tmd.reuse = {sequential = [{depth = 5 : i64, iterator = "%arg17", reuse_type = "no_reuse", volume = 0 : i64}], spatial = [{depth = 3 : i64, iterator = "%arg15", mapped_to = "y", reuse_type = "total_reuse", volume = 8192 : i64}, {depth = 4 : i64, iterator = "%arg16", mapped_to = "x", reuse_type = "total_reuse", volume = 8192 : i64}], temporal = [{depth = 0 : i64, iterator = "%arg12", reuse_type = "total_reuse", volume = 8192 : i64}, {depth = 1 : i64, iterator = "%arg13", reuse_type = "total_reuse", volume = 8192 : i64}, {depth = 2 : i64, iterator = "%arg14", reuse_type = "no_reuse", volume = 0 : i64}]}} : memref<*xf32> to memref<32x64xf32, strided<[?, 1], offset: ?>>
                %alloc_2 = memref.alloc() : memref<32x64xf32>
                memref.copy %reinterpret_cast_1, %alloc_2 : memref<32x64xf32, strided<[?, 1], offset: ?>> to memref<32x64xf32>
                %25 = bufferization.to_tensor %alloc_2 restrict writable : memref<32x64xf32> to tensor<32x64xf32>
                %26 = linalg.matmul ins(%22, %25 : tensor<64x32xf32>, tensor<32x64xf32>) outs(%9 : tensor<64x64xf32>) -> tensor<64x64xf32>
                %27 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%arg18, %26 : tensor<64x64xf32>, tensor<64x64xf32>) outs(%arg18 : tensor<64x64xf32>) {
                ^bb0(%in: f32, %in_3: f32, %out: f32):
                  %28 = arith.addf %in, %in_3 : f32
                  linalg.yield %28 : f32
                } -> tensor<64x64xf32>
                scf.yield %27 : tensor<64x64xf32>
              }
              %13 = arith.muli %10, %arg8 : index
              %14 = affine.apply affine_map<(d0)[s0] -> (d0 * 64 + s0)>(%arg14)[%13]
              %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%14], sizes: [64, 64], strides: [%arg8, 1] {tmd.reuse = {spatial = [{depth = 3 : i64, iterator = "%arg15", mapped_to = "y", reuse_type = "total_reuse", volume = 16384 : i64}, {depth = 4 : i64, iterator = "%arg16", mapped_to = "x", reuse_type = "total_reuse", volume = 16384 : i64}], temporal = [{depth = 0 : i64, iterator = "%arg12", reuse_type = "total_reuse", volume = 16384 : i64}, {depth = 1 : i64, iterator = "%arg13", reuse_type = "no_reuse", volume = 0 : i64}, {depth = 2 : i64, iterator = "%arg14", reuse_type = "no_reuse", volume = 0 : i64}]}} : memref<*xf32> to memref<64x64xf32, strided<[?, 1], offset: ?>>
              %15 = affine.min affine_map<(d0, d1, d2)[s0] -> (d0 * -64 + s0, 64)>(%arg13, %arg14, %7)[%arg3]
              %16 = affine.min affine_map<(d0, d1, d2)[s0] -> (d1 * -64 + s0, 64)>(%arg13, %arg14, %7)[%arg4]
              %extracted_slice = tensor.extract_slice %12[0, 0] [%15, %16] [1, 1] : tensor<64x64xf32> to tensor<?x?xf32>
              %17 = affine.min affine_map<(d0, d1, d2)[s0] -> (d0 * -64 + s0, 64)>(%arg13, %arg14, %7)[%arg3]
              %18 = affine.min affine_map<(d0, d1, d2)[s0] -> (d1 * -64 + s0, 64)>(%arg13, %arg14, %7)[%arg4]
              %subview = memref.subview %reinterpret_cast[0, 0] [%17, %18] [1, 1] : memref<64x64xf32, strided<[?, 1], offset: ?>> to memref<?x?xf32, strided<[?, 1], offset: ?>>
              bufferization.materialize_in_destination %extracted_slice in writable %subview : (tensor<?x?xf32>, memref<?x?xf32, strided<[?, 1], offset: ?>>) -> ()
            } {tmd.mapped_to = "x"}
          } {tmd.mapped_to = "y"}
        }
      }
    }
    return
  }
  func.func @matmul_kernel__d0i2_d1i2_f2_f1_f0(%arg0: memref<*xf32> {tt.divisibility = 16 : i32}, %arg1: memref<*xf32> {tt.divisibility = 16 : i32}, %arg2: memref<*xf32> {tt.divisibility = 16 : i32}, %arg3: index {tt.divisibility = 16 : i32}, %arg4: index {tt.divisibility = 16 : i32}, %arg5: index {tt.divisibility = 16 : i32}, %arg6: index {tt.divisibility = 16 : i32}, %arg7: index {tt.divisibility = 16 : i32}, %arg8: index {tt.divisibility = 16 : i32}, %arg9: index, %arg10: index, %arg11: index) {
    affine.for %arg12 = 0 to affine_map<(d0, d1, d2) -> ((d2 ceildiv 8) ceildiv 8)>(%arg9, %arg10, %arg11) {
      affine.for %arg13 = 0 to affine_map<(d0, d1, d2) -> (d1)>(%arg9, %arg10, %arg11) {
        affine.for %arg14 = 0 to affine_map<(d0, d1, d2) -> (d0)>(%arg9, %arg10, %arg11) {
          affine.parallel (%arg15) = (0) to (8) {
            affine.parallel (%arg16) = (0) to (8) {
              %7 = affine.apply affine_map<(d0, d1, d2) -> (d0 + d1 * 64 + d2 * 8)>(%arg16, %arg12, %arg15)
              %cst = arith.constant 0.000000e+00 : f32
              %8 = tensor.empty() : tensor<64x64xf32>
              %9 = linalg.fill ins(%cst : f32) outs(%8 : tensor<64x64xf32>) -> tensor<64x64xf32>
              %c64 = arith.constant 64 : index
              %10 = arith.muli %arg14, %c64 : index
              %11 = affine.apply affine_map<()[s0] -> ((s0 + 31) floordiv 32)>()[%arg5]
              %c0 = arith.constant 0 : index
              %c1 = arith.constant 1 : index
              %12 = scf.for %arg17 = %c0 to %11 step %c1 iter_args(%arg18 = %9) -> (tensor<64x64xf32>) {
                %c32 = arith.constant 32 : index
                %19 = arith.muli %arg17, %c32 : index
                %20 = arith.muli %10, %arg6 : index
                %21 = affine.apply affine_map<(d0)[s0] -> (d0 * 32 + s0)>(%arg17)[%20]
                %reinterpret_cast_0 = memref.reinterpret_cast %arg0 to offset: [%21], sizes: [64, 32], strides: [%arg6, 1] {tmd.reuse = {sequential = [{depth = 5 : i64, iterator = "%arg17", reuse_type = "no_reuse", volume = 0 : i64}], spatial = [{depth = 3 : i64, iterator = "%arg15", mapped_to = "y", reuse_type = "total_reuse", volume = 8192 : i64}, {depth = 4 : i64, iterator = "%arg16", mapped_to = "x", reuse_type = "total_reuse", volume = 8192 : i64}], temporal = [{depth = 0 : i64, iterator = "%arg12", reuse_type = "total_reuse", volume = 8192 : i64}, {depth = 1 : i64, iterator = "%arg13", reuse_type = "total_reuse", volume = 8192 : i64}, {depth = 2 : i64, iterator = "%arg14", reuse_type = "no_reuse", volume = 0 : i64}]}} : memref<*xf32> to memref<64x32xf32, strided<[?, 1], offset: ?>>
                %alloc = memref.alloc() : memref<64x32xf32>
                memref.copy %reinterpret_cast_0, %alloc : memref<64x32xf32, strided<[?, 1], offset: ?>> to memref<64x32xf32>
                %22 = bufferization.to_tensor %alloc restrict writable : memref<64x32xf32> to tensor<64x32xf32>
                %23 = arith.muli %19, %arg7 : index
                %24 = affine.apply affine_map<(d0)[s0] -> (d0 * 64 + s0)>(%arg13)[%23]
                %reinterpret_cast_1 = memref.reinterpret_cast %arg1 to offset: [%24], sizes: [32, 64], strides: [%arg7, 1] {tmd.reuse = {sequential = [{depth = 5 : i64, iterator = "%arg17", reuse_type = "no_reuse", volume = 0 : i64}], spatial = [{depth = 3 : i64, iterator = "%arg15", mapped_to = "y", reuse_type = "total_reuse", volume = 8192 : i64}, {depth = 4 : i64, iterator = "%arg16", mapped_to = "x", reuse_type = "total_reuse", volume = 8192 : i64}], temporal = [{depth = 0 : i64, iterator = "%arg12", reuse_type = "total_reuse", volume = 8192 : i64}, {depth = 1 : i64, iterator = "%arg13", reuse_type = "no_reuse", volume = 0 : i64}, {depth = 2 : i64, iterator = "%arg14", reuse_type = "total_reuse", volume = 8192 : i64}]}} : memref<*xf32> to memref<32x64xf32, strided<[?, 1], offset: ?>>
                %alloc_2 = memref.alloc() : memref<32x64xf32>
                memref.copy %reinterpret_cast_1, %alloc_2 : memref<32x64xf32, strided<[?, 1], offset: ?>> to memref<32x64xf32>
                %25 = bufferization.to_tensor %alloc_2 restrict writable : memref<32x64xf32> to tensor<32x64xf32>
                %26 = linalg.matmul ins(%22, %25 : tensor<64x32xf32>, tensor<32x64xf32>) outs(%9 : tensor<64x64xf32>) -> tensor<64x64xf32>
                %27 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%arg18, %26 : tensor<64x64xf32>, tensor<64x64xf32>) outs(%arg18 : tensor<64x64xf32>) {
                ^bb0(%in: f32, %in_3: f32, %out: f32):
                  %28 = arith.addf %in, %in_3 : f32
                  linalg.yield %28 : f32
                } -> tensor<64x64xf32>
                scf.yield %27 : tensor<64x64xf32>
              }
              %13 = arith.muli %10, %arg8 : index
              %14 = affine.apply affine_map<(d0)[s0] -> (d0 * 64 + s0)>(%arg13)[%13]
              %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%14], sizes: [64, 64], strides: [%arg8, 1] {tmd.reuse = {spatial = [{depth = 3 : i64, iterator = "%arg15", mapped_to = "y", reuse_type = "total_reuse", volume = 16384 : i64}, {depth = 4 : i64, iterator = "%arg16", mapped_to = "x", reuse_type = "total_reuse", volume = 16384 : i64}], temporal = [{depth = 0 : i64, iterator = "%arg12", reuse_type = "total_reuse", volume = 16384 : i64}, {depth = 1 : i64, iterator = "%arg13", reuse_type = "no_reuse", volume = 0 : i64}, {depth = 2 : i64, iterator = "%arg14", reuse_type = "no_reuse", volume = 0 : i64}]}} : memref<*xf32> to memref<64x64xf32, strided<[?, 1], offset: ?>>
              %15 = affine.min affine_map<(d0, d1, d2)[s0] -> (d0 * -64 + s0, 64)>(%arg14, %arg13, %7)[%arg3]
              %16 = affine.min affine_map<(d0, d1, d2)[s0] -> (d1 * -64 + s0, 64)>(%arg14, %arg13, %7)[%arg4]
              %extracted_slice = tensor.extract_slice %12[0, 0] [%15, %16] [1, 1] : tensor<64x64xf32> to tensor<?x?xf32>
              %17 = affine.min affine_map<(d0, d1, d2)[s0] -> (d0 * -64 + s0, 64)>(%arg14, %arg13, %7)[%arg3]
              %18 = affine.min affine_map<(d0, d1, d2)[s0] -> (d1 * -64 + s0, 64)>(%arg14, %arg13, %7)[%arg4]
              %subview = memref.subview %reinterpret_cast[0, 0] [%17, %18] [1, 1] : memref<64x64xf32, strided<[?, 1], offset: ?>> to memref<?x?xf32, strided<[?, 1], offset: ?>>
              bufferization.materialize_in_destination %extracted_slice in writable %subview : (tensor<?x?xf32>, memref<?x?xf32, strided<[?, 1], offset: ?>>) -> ()
            } {tmd.mapped_to = "x"}
          } {tmd.mapped_to = "y"}
        }
      }
    }
    return
  }
  func.func @matmul_kernel__d1i2_d0i2_f0_f1_f2(%arg0: memref<*xf32> {tt.divisibility = 16 : i32}, %arg1: memref<*xf32> {tt.divisibility = 16 : i32}, %arg2: memref<*xf32> {tt.divisibility = 16 : i32}, %arg3: index {tt.divisibility = 16 : i32}, %arg4: index {tt.divisibility = 16 : i32}, %arg5: index {tt.divisibility = 16 : i32}, %arg6: index {tt.divisibility = 16 : i32}, %arg7: index {tt.divisibility = 16 : i32}, %arg8: index {tt.divisibility = 16 : i32}, %arg9: index, %arg10: index, %arg11: index) {
    affine.for %arg12 = 0 to affine_map<(d0, d1, d2) -> (d0)>(%arg9, %arg10, %arg11) {
      affine.for %arg13 = 0 to affine_map<(d0, d1, d2) -> (d1)>(%arg9, %arg10, %arg11) {
        affine.for %arg14 = 0 to affine_map<(d0, d1, d2) -> ((d2 ceildiv 8) ceildiv 8)>(%arg9, %arg10, %arg11) {
          affine.parallel (%arg15) = (0) to (8) {
            affine.parallel (%arg16) = (0) to (8) {
              %7 = affine.apply affine_map<(d0, d1, d2) -> (d0 + d1 * 64 + d2 * 8)>(%arg16, %arg14, %arg15)
              %cst = arith.constant 0.000000e+00 : f32
              %8 = tensor.empty() : tensor<64x64xf32>
              %9 = linalg.fill ins(%cst : f32) outs(%8 : tensor<64x64xf32>) -> tensor<64x64xf32>
              %c64 = arith.constant 64 : index
              %10 = arith.muli %arg12, %c64 : index
              %11 = affine.apply affine_map<()[s0] -> ((s0 + 31) floordiv 32)>()[%arg5]
              %c0 = arith.constant 0 : index
              %c1 = arith.constant 1 : index
              %12 = scf.for %arg17 = %c0 to %11 step %c1 iter_args(%arg18 = %9) -> (tensor<64x64xf32>) {
                %c32 = arith.constant 32 : index
                %19 = arith.muli %arg17, %c32 : index
                %20 = arith.muli %10, %arg6 : index
                %21 = affine.apply affine_map<(d0)[s0] -> (d0 * 32 + s0)>(%arg17)[%20]
                %reinterpret_cast_0 = memref.reinterpret_cast %arg0 to offset: [%21], sizes: [64, 32], strides: [%arg6, 1] {tmd.reuse = {sequential = [{depth = 5 : i64, iterator = "%arg17", reuse_type = "no_reuse", volume = 0 : i64}], spatial = [{depth = 3 : i64, iterator = "%arg15", mapped_to = "x", reuse_type = "total_reuse", volume = 8192 : i64}, {depth = 4 : i64, iterator = "%arg16", mapped_to = "y", reuse_type = "total_reuse", volume = 8192 : i64}], temporal = [{depth = 0 : i64, iterator = "%arg12", reuse_type = "no_reuse", volume = 0 : i64}, {depth = 1 : i64, iterator = "%arg13", reuse_type = "total_reuse", volume = 8192 : i64}, {depth = 2 : i64, iterator = "%arg14", reuse_type = "total_reuse", volume = 8192 : i64}]}} : memref<*xf32> to memref<64x32xf32, strided<[?, 1], offset: ?>>
                %alloc = memref.alloc() : memref<64x32xf32>
                memref.copy %reinterpret_cast_0, %alloc : memref<64x32xf32, strided<[?, 1], offset: ?>> to memref<64x32xf32>
                %22 = bufferization.to_tensor %alloc restrict writable : memref<64x32xf32> to tensor<64x32xf32>
                %23 = arith.muli %19, %arg7 : index
                %24 = affine.apply affine_map<(d0)[s0] -> (d0 * 64 + s0)>(%arg13)[%23]
                %reinterpret_cast_1 = memref.reinterpret_cast %arg1 to offset: [%24], sizes: [32, 64], strides: [%arg7, 1] {tmd.reuse = {sequential = [{depth = 5 : i64, iterator = "%arg17", reuse_type = "no_reuse", volume = 0 : i64}], spatial = [{depth = 3 : i64, iterator = "%arg15", mapped_to = "x", reuse_type = "total_reuse", volume = 8192 : i64}, {depth = 4 : i64, iterator = "%arg16", mapped_to = "y", reuse_type = "total_reuse", volume = 8192 : i64}], temporal = [{depth = 0 : i64, iterator = "%arg12", reuse_type = "total_reuse", volume = 8192 : i64}, {depth = 1 : i64, iterator = "%arg13", reuse_type = "no_reuse", volume = 0 : i64}, {depth = 2 : i64, iterator = "%arg14", reuse_type = "total_reuse", volume = 8192 : i64}]}} : memref<*xf32> to memref<32x64xf32, strided<[?, 1], offset: ?>>
                %alloc_2 = memref.alloc() : memref<32x64xf32>
                memref.copy %reinterpret_cast_1, %alloc_2 : memref<32x64xf32, strided<[?, 1], offset: ?>> to memref<32x64xf32>
                %25 = bufferization.to_tensor %alloc_2 restrict writable : memref<32x64xf32> to tensor<32x64xf32>
                %26 = linalg.matmul ins(%22, %25 : tensor<64x32xf32>, tensor<32x64xf32>) outs(%9 : tensor<64x64xf32>) -> tensor<64x64xf32>
                %27 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%arg18, %26 : tensor<64x64xf32>, tensor<64x64xf32>) outs(%arg18 : tensor<64x64xf32>) {
                ^bb0(%in: f32, %in_3: f32, %out: f32):
                  %28 = arith.addf %in, %in_3 : f32
                  linalg.yield %28 : f32
                } -> tensor<64x64xf32>
                scf.yield %27 : tensor<64x64xf32>
              }
              %13 = arith.muli %10, %arg8 : index
              %14 = affine.apply affine_map<(d0)[s0] -> (d0 * 64 + s0)>(%arg13)[%13]
              %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%14], sizes: [64, 64], strides: [%arg8, 1] {tmd.reuse = {spatial = [{depth = 3 : i64, iterator = "%arg15", mapped_to = "x", reuse_type = "total_reuse", volume = 16384 : i64}, {depth = 4 : i64, iterator = "%arg16", mapped_to = "y", reuse_type = "total_reuse", volume = 16384 : i64}], temporal = [{depth = 0 : i64, iterator = "%arg12", reuse_type = "no_reuse", volume = 0 : i64}, {depth = 1 : i64, iterator = "%arg13", reuse_type = "no_reuse", volume = 0 : i64}, {depth = 2 : i64, iterator = "%arg14", reuse_type = "total_reuse", volume = 16384 : i64}]}} : memref<*xf32> to memref<64x64xf32, strided<[?, 1], offset: ?>>
              %15 = affine.min affine_map<(d0, d1, d2)[s0] -> (d0 * -64 + s0, 64)>(%arg12, %arg13, %7)[%arg3]
              %16 = affine.min affine_map<(d0, d1, d2)[s0] -> (d1 * -64 + s0, 64)>(%arg12, %arg13, %7)[%arg4]
              %extracted_slice = tensor.extract_slice %12[0, 0] [%15, %16] [1, 1] : tensor<64x64xf32> to tensor<?x?xf32>
              %17 = affine.min affine_map<(d0, d1, d2)[s0] -> (d0 * -64 + s0, 64)>(%arg12, %arg13, %7)[%arg3]
              %18 = affine.min affine_map<(d0, d1, d2)[s0] -> (d1 * -64 + s0, 64)>(%arg12, %arg13, %7)[%arg4]
              %subview = memref.subview %reinterpret_cast[0, 0] [%17, %18] [1, 1] : memref<64x64xf32, strided<[?, 1], offset: ?>> to memref<?x?xf32, strided<[?, 1], offset: ?>>
              bufferization.materialize_in_destination %extracted_slice in writable %subview : (tensor<?x?xf32>, memref<?x?xf32, strided<[?, 1], offset: ?>>) -> ()
            } {tmd.mapped_to = "y"}
          } {tmd.mapped_to = "x"}
        }
      }
    }
    return
  }
  func.func @matmul_kernel__d1i2_d0i2_f0_f2_f1(%arg0: memref<*xf32> {tt.divisibility = 16 : i32}, %arg1: memref<*xf32> {tt.divisibility = 16 : i32}, %arg2: memref<*xf32> {tt.divisibility = 16 : i32}, %arg3: index {tt.divisibility = 16 : i32}, %arg4: index {tt.divisibility = 16 : i32}, %arg5: index {tt.divisibility = 16 : i32}, %arg6: index {tt.divisibility = 16 : i32}, %arg7: index {tt.divisibility = 16 : i32}, %arg8: index {tt.divisibility = 16 : i32}, %arg9: index, %arg10: index, %arg11: index) {
    affine.for %arg12 = 0 to affine_map<(d0, d1, d2) -> (d0)>(%arg9, %arg10, %arg11) {
      affine.for %arg13 = 0 to affine_map<(d0, d1, d2) -> ((d2 ceildiv 8) ceildiv 8)>(%arg9, %arg10, %arg11) {
        affine.for %arg14 = 0 to affine_map<(d0, d1, d2) -> (d1)>(%arg9, %arg10, %arg11) {
          affine.parallel (%arg15) = (0) to (8) {
            affine.parallel (%arg16) = (0) to (8) {
              %7 = affine.apply affine_map<(d0, d1, d2) -> (d0 + d1 * 64 + d2 * 8)>(%arg16, %arg13, %arg15)
              %cst = arith.constant 0.000000e+00 : f32
              %8 = tensor.empty() : tensor<64x64xf32>
              %9 = linalg.fill ins(%cst : f32) outs(%8 : tensor<64x64xf32>) -> tensor<64x64xf32>
              %c64 = arith.constant 64 : index
              %10 = arith.muli %arg12, %c64 : index
              %11 = affine.apply affine_map<()[s0] -> ((s0 + 31) floordiv 32)>()[%arg5]
              %c0 = arith.constant 0 : index
              %c1 = arith.constant 1 : index
              %12 = scf.for %arg17 = %c0 to %11 step %c1 iter_args(%arg18 = %9) -> (tensor<64x64xf32>) {
                %c32 = arith.constant 32 : index
                %19 = arith.muli %arg17, %c32 : index
                %20 = arith.muli %10, %arg6 : index
                %21 = affine.apply affine_map<(d0)[s0] -> (d0 * 32 + s0)>(%arg17)[%20]
                %reinterpret_cast_0 = memref.reinterpret_cast %arg0 to offset: [%21], sizes: [64, 32], strides: [%arg6, 1] {tmd.reuse = {sequential = [{depth = 5 : i64, iterator = "%arg17", reuse_type = "no_reuse", volume = 0 : i64}], spatial = [{depth = 3 : i64, iterator = "%arg15", mapped_to = "x", reuse_type = "total_reuse", volume = 8192 : i64}, {depth = 4 : i64, iterator = "%arg16", mapped_to = "y", reuse_type = "total_reuse", volume = 8192 : i64}], temporal = [{depth = 0 : i64, iterator = "%arg12", reuse_type = "no_reuse", volume = 0 : i64}, {depth = 1 : i64, iterator = "%arg13", reuse_type = "total_reuse", volume = 8192 : i64}, {depth = 2 : i64, iterator = "%arg14", reuse_type = "total_reuse", volume = 8192 : i64}]}} : memref<*xf32> to memref<64x32xf32, strided<[?, 1], offset: ?>>
                %alloc = memref.alloc() : memref<64x32xf32>
                memref.copy %reinterpret_cast_0, %alloc : memref<64x32xf32, strided<[?, 1], offset: ?>> to memref<64x32xf32>
                %22 = bufferization.to_tensor %alloc restrict writable : memref<64x32xf32> to tensor<64x32xf32>
                %23 = arith.muli %19, %arg7 : index
                %24 = affine.apply affine_map<(d0)[s0] -> (d0 * 64 + s0)>(%arg14)[%23]
                %reinterpret_cast_1 = memref.reinterpret_cast %arg1 to offset: [%24], sizes: [32, 64], strides: [%arg7, 1] {tmd.reuse = {sequential = [{depth = 5 : i64, iterator = "%arg17", reuse_type = "no_reuse", volume = 0 : i64}], spatial = [{depth = 3 : i64, iterator = "%arg15", mapped_to = "x", reuse_type = "total_reuse", volume = 8192 : i64}, {depth = 4 : i64, iterator = "%arg16", mapped_to = "y", reuse_type = "total_reuse", volume = 8192 : i64}], temporal = [{depth = 0 : i64, iterator = "%arg12", reuse_type = "total_reuse", volume = 8192 : i64}, {depth = 1 : i64, iterator = "%arg13", reuse_type = "total_reuse", volume = 8192 : i64}, {depth = 2 : i64, iterator = "%arg14", reuse_type = "no_reuse", volume = 0 : i64}]}} : memref<*xf32> to memref<32x64xf32, strided<[?, 1], offset: ?>>
                %alloc_2 = memref.alloc() : memref<32x64xf32>
                memref.copy %reinterpret_cast_1, %alloc_2 : memref<32x64xf32, strided<[?, 1], offset: ?>> to memref<32x64xf32>
                %25 = bufferization.to_tensor %alloc_2 restrict writable : memref<32x64xf32> to tensor<32x64xf32>
                %26 = linalg.matmul ins(%22, %25 : tensor<64x32xf32>, tensor<32x64xf32>) outs(%9 : tensor<64x64xf32>) -> tensor<64x64xf32>
                %27 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%arg18, %26 : tensor<64x64xf32>, tensor<64x64xf32>) outs(%arg18 : tensor<64x64xf32>) {
                ^bb0(%in: f32, %in_3: f32, %out: f32):
                  %28 = arith.addf %in, %in_3 : f32
                  linalg.yield %28 : f32
                } -> tensor<64x64xf32>
                scf.yield %27 : tensor<64x64xf32>
              }
              %13 = arith.muli %10, %arg8 : index
              %14 = affine.apply affine_map<(d0)[s0] -> (d0 * 64 + s0)>(%arg14)[%13]
              %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%14], sizes: [64, 64], strides: [%arg8, 1] {tmd.reuse = {spatial = [{depth = 3 : i64, iterator = "%arg15", mapped_to = "x", reuse_type = "total_reuse", volume = 16384 : i64}, {depth = 4 : i64, iterator = "%arg16", mapped_to = "y", reuse_type = "total_reuse", volume = 16384 : i64}], temporal = [{depth = 0 : i64, iterator = "%arg12", reuse_type = "no_reuse", volume = 0 : i64}, {depth = 1 : i64, iterator = "%arg13", reuse_type = "total_reuse", volume = 16384 : i64}, {depth = 2 : i64, iterator = "%arg14", reuse_type = "no_reuse", volume = 0 : i64}]}} : memref<*xf32> to memref<64x64xf32, strided<[?, 1], offset: ?>>
              %15 = affine.min affine_map<(d0, d1, d2)[s0] -> (d0 * -64 + s0, 64)>(%arg12, %arg14, %7)[%arg3]
              %16 = affine.min affine_map<(d0, d1, d2)[s0] -> (d1 * -64 + s0, 64)>(%arg12, %arg14, %7)[%arg4]
              %extracted_slice = tensor.extract_slice %12[0, 0] [%15, %16] [1, 1] : tensor<64x64xf32> to tensor<?x?xf32>
              %17 = affine.min affine_map<(d0, d1, d2)[s0] -> (d0 * -64 + s0, 64)>(%arg12, %arg14, %7)[%arg3]
              %18 = affine.min affine_map<(d0, d1, d2)[s0] -> (d1 * -64 + s0, 64)>(%arg12, %arg14, %7)[%arg4]
              %subview = memref.subview %reinterpret_cast[0, 0] [%17, %18] [1, 1] : memref<64x64xf32, strided<[?, 1], offset: ?>> to memref<?x?xf32, strided<[?, 1], offset: ?>>
              bufferization.materialize_in_destination %extracted_slice in writable %subview : (tensor<?x?xf32>, memref<?x?xf32, strided<[?, 1], offset: ?>>) -> ()
            } {tmd.mapped_to = "y"}
          } {tmd.mapped_to = "x"}
        }
      }
    }
    return
  }
  func.func @matmul_kernel__d1i2_d0i2_f1_f0_f2(%arg0: memref<*xf32> {tt.divisibility = 16 : i32}, %arg1: memref<*xf32> {tt.divisibility = 16 : i32}, %arg2: memref<*xf32> {tt.divisibility = 16 : i32}, %arg3: index {tt.divisibility = 16 : i32}, %arg4: index {tt.divisibility = 16 : i32}, %arg5: index {tt.divisibility = 16 : i32}, %arg6: index {tt.divisibility = 16 : i32}, %arg7: index {tt.divisibility = 16 : i32}, %arg8: index {tt.divisibility = 16 : i32}, %arg9: index, %arg10: index, %arg11: index) {
    affine.for %arg12 = 0 to affine_map<(d0, d1, d2) -> (d1)>(%arg9, %arg10, %arg11) {
      affine.for %arg13 = 0 to affine_map<(d0, d1, d2) -> (d0)>(%arg9, %arg10, %arg11) {
        affine.for %arg14 = 0 to affine_map<(d0, d1, d2) -> ((d2 ceildiv 8) ceildiv 8)>(%arg9, %arg10, %arg11) {
          affine.parallel (%arg15) = (0) to (8) {
            affine.parallel (%arg16) = (0) to (8) {
              %7 = affine.apply affine_map<(d0, d1, d2) -> (d0 + d1 * 64 + d2 * 8)>(%arg16, %arg14, %arg15)
              %cst = arith.constant 0.000000e+00 : f32
              %8 = tensor.empty() : tensor<64x64xf32>
              %9 = linalg.fill ins(%cst : f32) outs(%8 : tensor<64x64xf32>) -> tensor<64x64xf32>
              %c64 = arith.constant 64 : index
              %10 = arith.muli %arg13, %c64 : index
              %11 = affine.apply affine_map<()[s0] -> ((s0 + 31) floordiv 32)>()[%arg5]
              %c0 = arith.constant 0 : index
              %c1 = arith.constant 1 : index
              %12 = scf.for %arg17 = %c0 to %11 step %c1 iter_args(%arg18 = %9) -> (tensor<64x64xf32>) {
                %c32 = arith.constant 32 : index
                %19 = arith.muli %arg17, %c32 : index
                %20 = arith.muli %10, %arg6 : index
                %21 = affine.apply affine_map<(d0)[s0] -> (d0 * 32 + s0)>(%arg17)[%20]
                %reinterpret_cast_0 = memref.reinterpret_cast %arg0 to offset: [%21], sizes: [64, 32], strides: [%arg6, 1] {tmd.reuse = {sequential = [{depth = 5 : i64, iterator = "%arg17", reuse_type = "no_reuse", volume = 0 : i64}], spatial = [{depth = 3 : i64, iterator = "%arg15", mapped_to = "x", reuse_type = "total_reuse", volume = 8192 : i64}, {depth = 4 : i64, iterator = "%arg16", mapped_to = "y", reuse_type = "total_reuse", volume = 8192 : i64}], temporal = [{depth = 0 : i64, iterator = "%arg12", reuse_type = "total_reuse", volume = 8192 : i64}, {depth = 1 : i64, iterator = "%arg13", reuse_type = "no_reuse", volume = 0 : i64}, {depth = 2 : i64, iterator = "%arg14", reuse_type = "total_reuse", volume = 8192 : i64}]}} : memref<*xf32> to memref<64x32xf32, strided<[?, 1], offset: ?>>
                %alloc = memref.alloc() : memref<64x32xf32>
                memref.copy %reinterpret_cast_0, %alloc : memref<64x32xf32, strided<[?, 1], offset: ?>> to memref<64x32xf32>
                %22 = bufferization.to_tensor %alloc restrict writable : memref<64x32xf32> to tensor<64x32xf32>
                %23 = arith.muli %19, %arg7 : index
                %24 = affine.apply affine_map<(d0)[s0] -> (d0 * 64 + s0)>(%arg12)[%23]
                %reinterpret_cast_1 = memref.reinterpret_cast %arg1 to offset: [%24], sizes: [32, 64], strides: [%arg7, 1] {tmd.reuse = {sequential = [{depth = 5 : i64, iterator = "%arg17", reuse_type = "no_reuse", volume = 0 : i64}], spatial = [{depth = 3 : i64, iterator = "%arg15", mapped_to = "x", reuse_type = "total_reuse", volume = 8192 : i64}, {depth = 4 : i64, iterator = "%arg16", mapped_to = "y", reuse_type = "total_reuse", volume = 8192 : i64}], temporal = [{depth = 0 : i64, iterator = "%arg12", reuse_type = "no_reuse", volume = 0 : i64}, {depth = 1 : i64, iterator = "%arg13", reuse_type = "total_reuse", volume = 8192 : i64}, {depth = 2 : i64, iterator = "%arg14", reuse_type = "total_reuse", volume = 8192 : i64}]}} : memref<*xf32> to memref<32x64xf32, strided<[?, 1], offset: ?>>
                %alloc_2 = memref.alloc() : memref<32x64xf32>
                memref.copy %reinterpret_cast_1, %alloc_2 : memref<32x64xf32, strided<[?, 1], offset: ?>> to memref<32x64xf32>
                %25 = bufferization.to_tensor %alloc_2 restrict writable : memref<32x64xf32> to tensor<32x64xf32>
                %26 = linalg.matmul ins(%22, %25 : tensor<64x32xf32>, tensor<32x64xf32>) outs(%9 : tensor<64x64xf32>) -> tensor<64x64xf32>
                %27 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%arg18, %26 : tensor<64x64xf32>, tensor<64x64xf32>) outs(%arg18 : tensor<64x64xf32>) {
                ^bb0(%in: f32, %in_3: f32, %out: f32):
                  %28 = arith.addf %in, %in_3 : f32
                  linalg.yield %28 : f32
                } -> tensor<64x64xf32>
                scf.yield %27 : tensor<64x64xf32>
              }
              %13 = arith.muli %10, %arg8 : index
              %14 = affine.apply affine_map<(d0)[s0] -> (d0 * 64 + s0)>(%arg12)[%13]
              %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%14], sizes: [64, 64], strides: [%arg8, 1] {tmd.reuse = {spatial = [{depth = 3 : i64, iterator = "%arg15", mapped_to = "x", reuse_type = "total_reuse", volume = 16384 : i64}, {depth = 4 : i64, iterator = "%arg16", mapped_to = "y", reuse_type = "total_reuse", volume = 16384 : i64}], temporal = [{depth = 0 : i64, iterator = "%arg12", reuse_type = "no_reuse", volume = 0 : i64}, {depth = 1 : i64, iterator = "%arg13", reuse_type = "no_reuse", volume = 0 : i64}, {depth = 2 : i64, iterator = "%arg14", reuse_type = "total_reuse", volume = 16384 : i64}]}} : memref<*xf32> to memref<64x64xf32, strided<[?, 1], offset: ?>>
              %15 = affine.min affine_map<(d0, d1, d2)[s0] -> (d0 * -64 + s0, 64)>(%arg13, %arg12, %7)[%arg3]
              %16 = affine.min affine_map<(d0, d1, d2)[s0] -> (d1 * -64 + s0, 64)>(%arg13, %arg12, %7)[%arg4]
              %extracted_slice = tensor.extract_slice %12[0, 0] [%15, %16] [1, 1] : tensor<64x64xf32> to tensor<?x?xf32>
              %17 = affine.min affine_map<(d0, d1, d2)[s0] -> (d0 * -64 + s0, 64)>(%arg13, %arg12, %7)[%arg3]
              %18 = affine.min affine_map<(d0, d1, d2)[s0] -> (d1 * -64 + s0, 64)>(%arg13, %arg12, %7)[%arg4]
              %subview = memref.subview %reinterpret_cast[0, 0] [%17, %18] [1, 1] : memref<64x64xf32, strided<[?, 1], offset: ?>> to memref<?x?xf32, strided<[?, 1], offset: ?>>
              bufferization.materialize_in_destination %extracted_slice in writable %subview : (tensor<?x?xf32>, memref<?x?xf32, strided<[?, 1], offset: ?>>) -> ()
            } {tmd.mapped_to = "y"}
          } {tmd.mapped_to = "x"}
        }
      }
    }
    return
  }
  func.func @matmul_kernel__d1i2_d0i2_f1_f2_f0(%arg0: memref<*xf32> {tt.divisibility = 16 : i32}, %arg1: memref<*xf32> {tt.divisibility = 16 : i32}, %arg2: memref<*xf32> {tt.divisibility = 16 : i32}, %arg3: index {tt.divisibility = 16 : i32}, %arg4: index {tt.divisibility = 16 : i32}, %arg5: index {tt.divisibility = 16 : i32}, %arg6: index {tt.divisibility = 16 : i32}, %arg7: index {tt.divisibility = 16 : i32}, %arg8: index {tt.divisibility = 16 : i32}, %arg9: index, %arg10: index, %arg11: index) {
    affine.for %arg12 = 0 to affine_map<(d0, d1, d2) -> (d1)>(%arg9, %arg10, %arg11) {
      affine.for %arg13 = 0 to affine_map<(d0, d1, d2) -> ((d2 ceildiv 8) ceildiv 8)>(%arg9, %arg10, %arg11) {
        affine.for %arg14 = 0 to affine_map<(d0, d1, d2) -> (d0)>(%arg9, %arg10, %arg11) {
          affine.parallel (%arg15) = (0) to (8) {
            affine.parallel (%arg16) = (0) to (8) {
              %7 = affine.apply affine_map<(d0, d1, d2) -> (d0 + d1 * 64 + d2 * 8)>(%arg16, %arg13, %arg15)
              %cst = arith.constant 0.000000e+00 : f32
              %8 = tensor.empty() : tensor<64x64xf32>
              %9 = linalg.fill ins(%cst : f32) outs(%8 : tensor<64x64xf32>) -> tensor<64x64xf32>
              %c64 = arith.constant 64 : index
              %10 = arith.muli %arg14, %c64 : index
              %11 = affine.apply affine_map<()[s0] -> ((s0 + 31) floordiv 32)>()[%arg5]
              %c0 = arith.constant 0 : index
              %c1 = arith.constant 1 : index
              %12 = scf.for %arg17 = %c0 to %11 step %c1 iter_args(%arg18 = %9) -> (tensor<64x64xf32>) {
                %c32 = arith.constant 32 : index
                %19 = arith.muli %arg17, %c32 : index
                %20 = arith.muli %10, %arg6 : index
                %21 = affine.apply affine_map<(d0)[s0] -> (d0 * 32 + s0)>(%arg17)[%20]
                %reinterpret_cast_0 = memref.reinterpret_cast %arg0 to offset: [%21], sizes: [64, 32], strides: [%arg6, 1] {tmd.reuse = {sequential = [{depth = 5 : i64, iterator = "%arg17", reuse_type = "no_reuse", volume = 0 : i64}], spatial = [{depth = 3 : i64, iterator = "%arg15", mapped_to = "x", reuse_type = "total_reuse", volume = 8192 : i64}, {depth = 4 : i64, iterator = "%arg16", mapped_to = "y", reuse_type = "total_reuse", volume = 8192 : i64}], temporal = [{depth = 0 : i64, iterator = "%arg12", reuse_type = "total_reuse", volume = 8192 : i64}, {depth = 1 : i64, iterator = "%arg13", reuse_type = "total_reuse", volume = 8192 : i64}, {depth = 2 : i64, iterator = "%arg14", reuse_type = "no_reuse", volume = 0 : i64}]}} : memref<*xf32> to memref<64x32xf32, strided<[?, 1], offset: ?>>
                %alloc = memref.alloc() : memref<64x32xf32>
                memref.copy %reinterpret_cast_0, %alloc : memref<64x32xf32, strided<[?, 1], offset: ?>> to memref<64x32xf32>
                %22 = bufferization.to_tensor %alloc restrict writable : memref<64x32xf32> to tensor<64x32xf32>
                %23 = arith.muli %19, %arg7 : index
                %24 = affine.apply affine_map<(d0)[s0] -> (d0 * 64 + s0)>(%arg12)[%23]
                %reinterpret_cast_1 = memref.reinterpret_cast %arg1 to offset: [%24], sizes: [32, 64], strides: [%arg7, 1] {tmd.reuse = {sequential = [{depth = 5 : i64, iterator = "%arg17", reuse_type = "no_reuse", volume = 0 : i64}], spatial = [{depth = 3 : i64, iterator = "%arg15", mapped_to = "x", reuse_type = "total_reuse", volume = 8192 : i64}, {depth = 4 : i64, iterator = "%arg16", mapped_to = "y", reuse_type = "total_reuse", volume = 8192 : i64}], temporal = [{depth = 0 : i64, iterator = "%arg12", reuse_type = "no_reuse", volume = 0 : i64}, {depth = 1 : i64, iterator = "%arg13", reuse_type = "total_reuse", volume = 8192 : i64}, {depth = 2 : i64, iterator = "%arg14", reuse_type = "total_reuse", volume = 8192 : i64}]}} : memref<*xf32> to memref<32x64xf32, strided<[?, 1], offset: ?>>
                %alloc_2 = memref.alloc() : memref<32x64xf32>
                memref.copy %reinterpret_cast_1, %alloc_2 : memref<32x64xf32, strided<[?, 1], offset: ?>> to memref<32x64xf32>
                %25 = bufferization.to_tensor %alloc_2 restrict writable : memref<32x64xf32> to tensor<32x64xf32>
                %26 = linalg.matmul ins(%22, %25 : tensor<64x32xf32>, tensor<32x64xf32>) outs(%9 : tensor<64x64xf32>) -> tensor<64x64xf32>
                %27 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%arg18, %26 : tensor<64x64xf32>, tensor<64x64xf32>) outs(%arg18 : tensor<64x64xf32>) {
                ^bb0(%in: f32, %in_3: f32, %out: f32):
                  %28 = arith.addf %in, %in_3 : f32
                  linalg.yield %28 : f32
                } -> tensor<64x64xf32>
                scf.yield %27 : tensor<64x64xf32>
              }
              %13 = arith.muli %10, %arg8 : index
              %14 = affine.apply affine_map<(d0)[s0] -> (d0 * 64 + s0)>(%arg12)[%13]
              %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%14], sizes: [64, 64], strides: [%arg8, 1] {tmd.reuse = {spatial = [{depth = 3 : i64, iterator = "%arg15", mapped_to = "x", reuse_type = "total_reuse", volume = 16384 : i64}, {depth = 4 : i64, iterator = "%arg16", mapped_to = "y", reuse_type = "total_reuse", volume = 16384 : i64}], temporal = [{depth = 0 : i64, iterator = "%arg12", reuse_type = "no_reuse", volume = 0 : i64}, {depth = 1 : i64, iterator = "%arg13", reuse_type = "total_reuse", volume = 16384 : i64}, {depth = 2 : i64, iterator = "%arg14", reuse_type = "no_reuse", volume = 0 : i64}]}} : memref<*xf32> to memref<64x64xf32, strided<[?, 1], offset: ?>>
              %15 = affine.min affine_map<(d0, d1, d2)[s0] -> (d0 * -64 + s0, 64)>(%arg14, %arg12, %7)[%arg3]
              %16 = affine.min affine_map<(d0, d1, d2)[s0] -> (d1 * -64 + s0, 64)>(%arg14, %arg12, %7)[%arg4]
              %extracted_slice = tensor.extract_slice %12[0, 0] [%15, %16] [1, 1] : tensor<64x64xf32> to tensor<?x?xf32>
              %17 = affine.min affine_map<(d0, d1, d2)[s0] -> (d0 * -64 + s0, 64)>(%arg14, %arg12, %7)[%arg3]
              %18 = affine.min affine_map<(d0, d1, d2)[s0] -> (d1 * -64 + s0, 64)>(%arg14, %arg12, %7)[%arg4]
              %subview = memref.subview %reinterpret_cast[0, 0] [%17, %18] [1, 1] : memref<64x64xf32, strided<[?, 1], offset: ?>> to memref<?x?xf32, strided<[?, 1], offset: ?>>
              bufferization.materialize_in_destination %extracted_slice in writable %subview : (tensor<?x?xf32>, memref<?x?xf32, strided<[?, 1], offset: ?>>) -> ()
            } {tmd.mapped_to = "y"}
          } {tmd.mapped_to = "x"}
        }
      }
    }
    return
  }
  func.func @matmul_kernel__d1i2_d0i2_f2_f0_f1(%arg0: memref<*xf32> {tt.divisibility = 16 : i32}, %arg1: memref<*xf32> {tt.divisibility = 16 : i32}, %arg2: memref<*xf32> {tt.divisibility = 16 : i32}, %arg3: index {tt.divisibility = 16 : i32}, %arg4: index {tt.divisibility = 16 : i32}, %arg5: index {tt.divisibility = 16 : i32}, %arg6: index {tt.divisibility = 16 : i32}, %arg7: index {tt.divisibility = 16 : i32}, %arg8: index {tt.divisibility = 16 : i32}, %arg9: index, %arg10: index, %arg11: index) {
    affine.for %arg12 = 0 to affine_map<(d0, d1, d2) -> ((d2 ceildiv 8) ceildiv 8)>(%arg9, %arg10, %arg11) {
      affine.for %arg13 = 0 to affine_map<(d0, d1, d2) -> (d0)>(%arg9, %arg10, %arg11) {
        affine.for %arg14 = 0 to affine_map<(d0, d1, d2) -> (d1)>(%arg9, %arg10, %arg11) {
          affine.parallel (%arg15) = (0) to (8) {
            affine.parallel (%arg16) = (0) to (8) {
              %7 = affine.apply affine_map<(d0, d1, d2) -> (d0 + d1 * 64 + d2 * 8)>(%arg16, %arg12, %arg15)
              %cst = arith.constant 0.000000e+00 : f32
              %8 = tensor.empty() : tensor<64x64xf32>
              %9 = linalg.fill ins(%cst : f32) outs(%8 : tensor<64x64xf32>) -> tensor<64x64xf32>
              %c64 = arith.constant 64 : index
              %10 = arith.muli %arg13, %c64 : index
              %11 = affine.apply affine_map<()[s0] -> ((s0 + 31) floordiv 32)>()[%arg5]
              %c0 = arith.constant 0 : index
              %c1 = arith.constant 1 : index
              %12 = scf.for %arg17 = %c0 to %11 step %c1 iter_args(%arg18 = %9) -> (tensor<64x64xf32>) {
                %c32 = arith.constant 32 : index
                %19 = arith.muli %arg17, %c32 : index
                %20 = arith.muli %10, %arg6 : index
                %21 = affine.apply affine_map<(d0)[s0] -> (d0 * 32 + s0)>(%arg17)[%20]
                %reinterpret_cast_0 = memref.reinterpret_cast %arg0 to offset: [%21], sizes: [64, 32], strides: [%arg6, 1] {tmd.reuse = {sequential = [{depth = 5 : i64, iterator = "%arg17", reuse_type = "no_reuse", volume = 0 : i64}], spatial = [{depth = 3 : i64, iterator = "%arg15", mapped_to = "x", reuse_type = "total_reuse", volume = 8192 : i64}, {depth = 4 : i64, iterator = "%arg16", mapped_to = "y", reuse_type = "total_reuse", volume = 8192 : i64}], temporal = [{depth = 0 : i64, iterator = "%arg12", reuse_type = "total_reuse", volume = 8192 : i64}, {depth = 1 : i64, iterator = "%arg13", reuse_type = "no_reuse", volume = 0 : i64}, {depth = 2 : i64, iterator = "%arg14", reuse_type = "total_reuse", volume = 8192 : i64}]}} : memref<*xf32> to memref<64x32xf32, strided<[?, 1], offset: ?>>
                %alloc = memref.alloc() : memref<64x32xf32>
                memref.copy %reinterpret_cast_0, %alloc : memref<64x32xf32, strided<[?, 1], offset: ?>> to memref<64x32xf32>
                %22 = bufferization.to_tensor %alloc restrict writable : memref<64x32xf32> to tensor<64x32xf32>
                %23 = arith.muli %19, %arg7 : index
                %24 = affine.apply affine_map<(d0)[s0] -> (d0 * 64 + s0)>(%arg14)[%23]
                %reinterpret_cast_1 = memref.reinterpret_cast %arg1 to offset: [%24], sizes: [32, 64], strides: [%arg7, 1] {tmd.reuse = {sequential = [{depth = 5 : i64, iterator = "%arg17", reuse_type = "no_reuse", volume = 0 : i64}], spatial = [{depth = 3 : i64, iterator = "%arg15", mapped_to = "x", reuse_type = "total_reuse", volume = 8192 : i64}, {depth = 4 : i64, iterator = "%arg16", mapped_to = "y", reuse_type = "total_reuse", volume = 8192 : i64}], temporal = [{depth = 0 : i64, iterator = "%arg12", reuse_type = "total_reuse", volume = 8192 : i64}, {depth = 1 : i64, iterator = "%arg13", reuse_type = "total_reuse", volume = 8192 : i64}, {depth = 2 : i64, iterator = "%arg14", reuse_type = "no_reuse", volume = 0 : i64}]}} : memref<*xf32> to memref<32x64xf32, strided<[?, 1], offset: ?>>
                %alloc_2 = memref.alloc() : memref<32x64xf32>
                memref.copy %reinterpret_cast_1, %alloc_2 : memref<32x64xf32, strided<[?, 1], offset: ?>> to memref<32x64xf32>
                %25 = bufferization.to_tensor %alloc_2 restrict writable : memref<32x64xf32> to tensor<32x64xf32>
                %26 = linalg.matmul ins(%22, %25 : tensor<64x32xf32>, tensor<32x64xf32>) outs(%9 : tensor<64x64xf32>) -> tensor<64x64xf32>
                %27 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%arg18, %26 : tensor<64x64xf32>, tensor<64x64xf32>) outs(%arg18 : tensor<64x64xf32>) {
                ^bb0(%in: f32, %in_3: f32, %out: f32):
                  %28 = arith.addf %in, %in_3 : f32
                  linalg.yield %28 : f32
                } -> tensor<64x64xf32>
                scf.yield %27 : tensor<64x64xf32>
              }
              %13 = arith.muli %10, %arg8 : index
              %14 = affine.apply affine_map<(d0)[s0] -> (d0 * 64 + s0)>(%arg14)[%13]
              %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%14], sizes: [64, 64], strides: [%arg8, 1] {tmd.reuse = {spatial = [{depth = 3 : i64, iterator = "%arg15", mapped_to = "x", reuse_type = "total_reuse", volume = 16384 : i64}, {depth = 4 : i64, iterator = "%arg16", mapped_to = "y", reuse_type = "total_reuse", volume = 16384 : i64}], temporal = [{depth = 0 : i64, iterator = "%arg12", reuse_type = "total_reuse", volume = 16384 : i64}, {depth = 1 : i64, iterator = "%arg13", reuse_type = "no_reuse", volume = 0 : i64}, {depth = 2 : i64, iterator = "%arg14", reuse_type = "no_reuse", volume = 0 : i64}]}} : memref<*xf32> to memref<64x64xf32, strided<[?, 1], offset: ?>>
              %15 = affine.min affine_map<(d0, d1, d2)[s0] -> (d0 * -64 + s0, 64)>(%arg13, %arg14, %7)[%arg3]
              %16 = affine.min affine_map<(d0, d1, d2)[s0] -> (d1 * -64 + s0, 64)>(%arg13, %arg14, %7)[%arg4]
              %extracted_slice = tensor.extract_slice %12[0, 0] [%15, %16] [1, 1] : tensor<64x64xf32> to tensor<?x?xf32>
              %17 = affine.min affine_map<(d0, d1, d2)[s0] -> (d0 * -64 + s0, 64)>(%arg13, %arg14, %7)[%arg3]
              %18 = affine.min affine_map<(d0, d1, d2)[s0] -> (d1 * -64 + s0, 64)>(%arg13, %arg14, %7)[%arg4]
              %subview = memref.subview %reinterpret_cast[0, 0] [%17, %18] [1, 1] : memref<64x64xf32, strided<[?, 1], offset: ?>> to memref<?x?xf32, strided<[?, 1], offset: ?>>
              bufferization.materialize_in_destination %extracted_slice in writable %subview : (tensor<?x?xf32>, memref<?x?xf32, strided<[?, 1], offset: ?>>) -> ()
            } {tmd.mapped_to = "y"}
          } {tmd.mapped_to = "x"}
        }
      }
    }
    return
  }
  func.func @matmul_kernel__d1i2_d0i2_f2_f1_f0(%arg0: memref<*xf32> {tt.divisibility = 16 : i32}, %arg1: memref<*xf32> {tt.divisibility = 16 : i32}, %arg2: memref<*xf32> {tt.divisibility = 16 : i32}, %arg3: index {tt.divisibility = 16 : i32}, %arg4: index {tt.divisibility = 16 : i32}, %arg5: index {tt.divisibility = 16 : i32}, %arg6: index {tt.divisibility = 16 : i32}, %arg7: index {tt.divisibility = 16 : i32}, %arg8: index {tt.divisibility = 16 : i32}, %arg9: index, %arg10: index, %arg11: index) {
    affine.for %arg12 = 0 to affine_map<(d0, d1, d2) -> ((d2 ceildiv 8) ceildiv 8)>(%arg9, %arg10, %arg11) {
      affine.for %arg13 = 0 to affine_map<(d0, d1, d2) -> (d1)>(%arg9, %arg10, %arg11) {
        affine.for %arg14 = 0 to affine_map<(d0, d1, d2) -> (d0)>(%arg9, %arg10, %arg11) {
          affine.parallel (%arg15) = (0) to (8) {
            affine.parallel (%arg16) = (0) to (8) {
              %7 = affine.apply affine_map<(d0, d1, d2) -> (d0 + d1 * 64 + d2 * 8)>(%arg16, %arg12, %arg15)
              %cst = arith.constant 0.000000e+00 : f32
              %8 = tensor.empty() : tensor<64x64xf32>
              %9 = linalg.fill ins(%cst : f32) outs(%8 : tensor<64x64xf32>) -> tensor<64x64xf32>
              %c64 = arith.constant 64 : index
              %10 = arith.muli %arg14, %c64 : index
              %11 = affine.apply affine_map<()[s0] -> ((s0 + 31) floordiv 32)>()[%arg5]
              %c0 = arith.constant 0 : index
              %c1 = arith.constant 1 : index
              %12 = scf.for %arg17 = %c0 to %11 step %c1 iter_args(%arg18 = %9) -> (tensor<64x64xf32>) {
                %c32 = arith.constant 32 : index
                %19 = arith.muli %arg17, %c32 : index
                %20 = arith.muli %10, %arg6 : index
                %21 = affine.apply affine_map<(d0)[s0] -> (d0 * 32 + s0)>(%arg17)[%20]
                %reinterpret_cast_0 = memref.reinterpret_cast %arg0 to offset: [%21], sizes: [64, 32], strides: [%arg6, 1] {tmd.reuse = {sequential = [{depth = 5 : i64, iterator = "%arg17", reuse_type = "no_reuse", volume = 0 : i64}], spatial = [{depth = 3 : i64, iterator = "%arg15", mapped_to = "x", reuse_type = "total_reuse", volume = 8192 : i64}, {depth = 4 : i64, iterator = "%arg16", mapped_to = "y", reuse_type = "total_reuse", volume = 8192 : i64}], temporal = [{depth = 0 : i64, iterator = "%arg12", reuse_type = "total_reuse", volume = 8192 : i64}, {depth = 1 : i64, iterator = "%arg13", reuse_type = "total_reuse", volume = 8192 : i64}, {depth = 2 : i64, iterator = "%arg14", reuse_type = "no_reuse", volume = 0 : i64}]}} : memref<*xf32> to memref<64x32xf32, strided<[?, 1], offset: ?>>
                %alloc = memref.alloc() : memref<64x32xf32>
                memref.copy %reinterpret_cast_0, %alloc : memref<64x32xf32, strided<[?, 1], offset: ?>> to memref<64x32xf32>
                %22 = bufferization.to_tensor %alloc restrict writable : memref<64x32xf32> to tensor<64x32xf32>
                %23 = arith.muli %19, %arg7 : index
                %24 = affine.apply affine_map<(d0)[s0] -> (d0 * 64 + s0)>(%arg13)[%23]
                %reinterpret_cast_1 = memref.reinterpret_cast %arg1 to offset: [%24], sizes: [32, 64], strides: [%arg7, 1] {tmd.reuse = {sequential = [{depth = 5 : i64, iterator = "%arg17", reuse_type = "no_reuse", volume = 0 : i64}], spatial = [{depth = 3 : i64, iterator = "%arg15", mapped_to = "x", reuse_type = "total_reuse", volume = 8192 : i64}, {depth = 4 : i64, iterator = "%arg16", mapped_to = "y", reuse_type = "total_reuse", volume = 8192 : i64}], temporal = [{depth = 0 : i64, iterator = "%arg12", reuse_type = "total_reuse", volume = 8192 : i64}, {depth = 1 : i64, iterator = "%arg13", reuse_type = "no_reuse", volume = 0 : i64}, {depth = 2 : i64, iterator = "%arg14", reuse_type = "total_reuse", volume = 8192 : i64}]}} : memref<*xf32> to memref<32x64xf32, strided<[?, 1], offset: ?>>
                %alloc_2 = memref.alloc() : memref<32x64xf32>
                memref.copy %reinterpret_cast_1, %alloc_2 : memref<32x64xf32, strided<[?, 1], offset: ?>> to memref<32x64xf32>
                %25 = bufferization.to_tensor %alloc_2 restrict writable : memref<32x64xf32> to tensor<32x64xf32>
                %26 = linalg.matmul ins(%22, %25 : tensor<64x32xf32>, tensor<32x64xf32>) outs(%9 : tensor<64x64xf32>) -> tensor<64x64xf32>
                %27 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%arg18, %26 : tensor<64x64xf32>, tensor<64x64xf32>) outs(%arg18 : tensor<64x64xf32>) {
                ^bb0(%in: f32, %in_3: f32, %out: f32):
                  %28 = arith.addf %in, %in_3 : f32
                  linalg.yield %28 : f32
                } -> tensor<64x64xf32>
                scf.yield %27 : tensor<64x64xf32>
              }
              %13 = arith.muli %10, %arg8 : index
              %14 = affine.apply affine_map<(d0)[s0] -> (d0 * 64 + s0)>(%arg13)[%13]
              %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%14], sizes: [64, 64], strides: [%arg8, 1] {tmd.reuse = {spatial = [{depth = 3 : i64, iterator = "%arg15", mapped_to = "x", reuse_type = "total_reuse", volume = 16384 : i64}, {depth = 4 : i64, iterator = "%arg16", mapped_to = "y", reuse_type = "total_reuse", volume = 16384 : i64}], temporal = [{depth = 0 : i64, iterator = "%arg12", reuse_type = "total_reuse", volume = 16384 : i64}, {depth = 1 : i64, iterator = "%arg13", reuse_type = "no_reuse", volume = 0 : i64}, {depth = 2 : i64, iterator = "%arg14", reuse_type = "no_reuse", volume = 0 : i64}]}} : memref<*xf32> to memref<64x64xf32, strided<[?, 1], offset: ?>>
              %15 = affine.min affine_map<(d0, d1, d2)[s0] -> (d0 * -64 + s0, 64)>(%arg14, %arg13, %7)[%arg3]
              %16 = affine.min affine_map<(d0, d1, d2)[s0] -> (d1 * -64 + s0, 64)>(%arg14, %arg13, %7)[%arg4]
              %extracted_slice = tensor.extract_slice %12[0, 0] [%15, %16] [1, 1] : tensor<64x64xf32> to tensor<?x?xf32>
              %17 = affine.min affine_map<(d0, d1, d2)[s0] -> (d0 * -64 + s0, 64)>(%arg14, %arg13, %7)[%arg3]
              %18 = affine.min affine_map<(d0, d1, d2)[s0] -> (d1 * -64 + s0, 64)>(%arg14, %arg13, %7)[%arg4]
              %subview = memref.subview %reinterpret_cast[0, 0] [%17, %18] [1, 1] : memref<64x64xf32, strided<[?, 1], offset: ?>> to memref<?x?xf32, strided<[?, 1], offset: ?>>
              bufferization.materialize_in_destination %extracted_slice in writable %subview : (tensor<?x?xf32>, memref<?x?xf32, strided<[?, 1], offset: ?>>) -> ()
            } {tmd.mapped_to = "y"}
          } {tmd.mapped_to = "x"}
        }
      }
    }
    return
  }
}
