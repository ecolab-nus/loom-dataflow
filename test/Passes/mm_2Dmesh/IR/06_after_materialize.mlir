module {
  %0 = df.mat "FPU" {shape = [32, 32, 32], throughput = 128}
  %1 = df.vec "SFPU" {shape = [32]}
  %2 = df.spatial_dim "x", 8
  %3 = df.spatial_dim "y", 8
  %4 = df.core "core" {scaleout=(%2, %3) , scalein=(%0, %1, [8, 1])}
  %5 = df.memory "L1" {scaleout=(%2, %3) , size = 1499136, bandwidth = 15}
  %6 = df.mux %4 : !df.compute, %5 : !df.memory  {map = affine_map<(d0, d1) -> (d0, d1)>}
  %7 = df.interconnects "horizontal_links" %5 : !df.memory, %5 : !df.memory  {bandwidth = 128 : i64, map = affine_map<(d0, d1) -> ((d0 + 1) mod 8, d1)>, spatial_dims = [@y]} : !df.interconnect
  %8 = df.interconnects "vertical_links" %5 : !df.memory, %5 : !df.memory  {bandwidth = 128 : i64, map = affine_map<(d0, d1) -> (d0, (d1 + 1) mod 8)>, spatial_dims = [@x]} : !df.interconnect
  %9 = df.spatial_dim "d", 4
  %10 = df.memory "DRAM" {scaleout=(%9) , size = 34359738368, bandwidth = 288}
  %11 = df.interconnects "NoC" %5 : !df.memory, %10 : !df.memory  {map = affine_map<(d0, d1) -> (d0 ceildiv 4 + (d1 ceildiv 4) * 2)>} : !df.interconnect
  module attributes {loom.pass_name = "Materialize"} {
    func.func @matmul__d0i0_d1i0__f01__d_d__BK32__BM32__BN32(%arg0: memref<128x128xf32>, %arg1: memref<128x256xf32>, %arg2: memref<128x256xf32>) {
      %cst = arith.constant 0.000000e+00 : f32
      %c32 = arith.constant 32 : index
      affine.parallel (%arg3) = (0) to (8) {
        affine.parallel (%arg4) = (0) to (8) {
          affine.for %arg5 = 0 to 1 {
            affine.for %arg6 = 0 to 8 {
              %12 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 8 + d1 + d2 * 64)>(%arg3, %arg4, %arg5)
              %13 = loom.alloc(%c32, %c32) on @L1 : !loom.buffer_token
              %14 = loom.init_tensor %13(%c32, %c32) : !loom.buffer_token -> tensor<?x?xf32>
              %15 = linalg.fill ins(%cst : f32) outs(%14 : tensor<?x?xf32>) -> tensor<?x?xf32>
              %16 = affine.for %arg7 = 0 to 4 iter_args(%arg8 = %15) -> (tensor<?x?xf32>) {
                %20 = arith.muli %12, %c32 : index
                %21 = arith.muli %arg7, %c32 : index
                %22 = loom.view %arg0[%20, %21] [32, 32] [1, 1], reuse : [seq = false, spat = false, temp = true] : memref<128x128xf32> -> !loom.view
                %23 = loom.alloc(%c32, %c32) on @L1 : !loom.buffer_token
                %24 = loom.copy_to_tensor %22, %23 to @L1, interconnect : [], broadcast : [1, 1] : !loom.view, !loom.buffer_token -> tensor<?x?xf32>
                %25 = arith.muli %arg6, %c32 : index
                %26 = loom.view %arg1[%21, %25] [32, 32] [1, 1], reuse : [seq = false, spat = true, temp = true] : memref<128x256xf32> -> !loom.view
                %27 = loom.alloc(%c32, %c32) on @L1 : !loom.buffer_token
                %28 = loom.copy_to_tensor %26, %27 to @L1, interconnect : [], broadcast : [1, 1] : !loom.view, !loom.buffer_token -> tensor<?x?xf32>
                %29 = linalg.matmul ins(%24, %28 : tensor<?x?xf32>, tensor<?x?xf32>) outs(%15 : tensor<?x?xf32>) -> tensor<?x?xf32>
                %30 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%arg8, %29 : tensor<?x?xf32>, tensor<?x?xf32>) outs(%14 : tensor<?x?xf32>) {
                ^bb0(%in: f32, %in_0: f32, %out: f32):
                  %31 = arith.addf %in, %in_0 : f32
                  linalg.yield %31 : f32
                } -> tensor<?x?xf32>
                affine.yield %30 : tensor<?x?xf32>
              } {loom.iter_type = #loom.iter_type<sequential>}
              %17 = arith.muli %12, %c32 : index
              %18 = arith.muli %arg6, %c32 : index
              %19 = loom.view %arg2[%17, %18] [32, 32] [1, 1], reuse : [seq = false, spat = false, temp = false] : memref<128x256xf32> -> !loom.view
              loom.copy_from_tensor %16, %19 to @DRAM : tensor<?x?xf32>, !loom.view
            } {loom.iter_type = #loom.iter_type<temporal>}
          } {loom.iter_type = #loom.iter_type<temporal>}
        } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @y}
      } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @x}
      return
    }
    func.func @matmul__d0i0_d1i0__f01__d_d__BK64__BM32__BN32(%arg0: memref<128x128xf32>, %arg1: memref<128x256xf32>, %arg2: memref<128x256xf32>) {
      %cst = arith.constant 0.000000e+00 : f32
      %c32 = arith.constant 32 : index
      %c64 = arith.constant 64 : index
      affine.parallel (%arg3) = (0) to (8) {
        affine.parallel (%arg4) = (0) to (8) {
          affine.for %arg5 = 0 to 1 {
            affine.for %arg6 = 0 to 8 {
              %12 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 8 + d1 + d2 * 64)>(%arg3, %arg4, %arg5)
              %13 = loom.alloc(%c32, %c32) on @L1 : !loom.buffer_token
              %14 = loom.init_tensor %13(%c32, %c32) : !loom.buffer_token -> tensor<?x?xf32>
              %15 = linalg.fill ins(%cst : f32) outs(%14 : tensor<?x?xf32>) -> tensor<?x?xf32>
              %16 = affine.for %arg7 = 0 to 2 iter_args(%arg8 = %15) -> (tensor<?x?xf32>) {
                %20 = arith.muli %12, %c32 : index
                %21 = arith.muli %arg7, %c64 : index
                %22 = loom.view %arg0[%20, %21] [32, 64] [1, 1], reuse : [seq = false, spat = false, temp = true] : memref<128x128xf32> -> !loom.view
                %23 = loom.alloc(%c32, %c64) on @L1 : !loom.buffer_token
                %24 = loom.copy_to_tensor %22, %23 to @L1, interconnect : [], broadcast : [1, 1] : !loom.view, !loom.buffer_token -> tensor<?x?xf32>
                %25 = arith.muli %arg6, %c32 : index
                %26 = loom.view %arg1[%21, %25] [64, 32] [1, 1], reuse : [seq = false, spat = true, temp = true] : memref<128x256xf32> -> !loom.view
                %27 = loom.alloc(%c64, %c32) on @L1 : !loom.buffer_token
                %28 = loom.copy_to_tensor %26, %27 to @L1, interconnect : [], broadcast : [1, 1] : !loom.view, !loom.buffer_token -> tensor<?x?xf32>
                %29 = linalg.matmul ins(%24, %28 : tensor<?x?xf32>, tensor<?x?xf32>) outs(%15 : tensor<?x?xf32>) -> tensor<?x?xf32>
                %30 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%arg8, %29 : tensor<?x?xf32>, tensor<?x?xf32>) outs(%14 : tensor<?x?xf32>) {
                ^bb0(%in: f32, %in_0: f32, %out: f32):
                  %31 = arith.addf %in, %in_0 : f32
                  linalg.yield %31 : f32
                } -> tensor<?x?xf32>
                affine.yield %30 : tensor<?x?xf32>
              } {loom.iter_type = #loom.iter_type<sequential>}
              %17 = arith.muli %12, %c32 : index
              %18 = arith.muli %arg6, %c32 : index
              %19 = loom.view %arg2[%17, %18] [32, 32] [1, 1], reuse : [seq = false, spat = false, temp = false] : memref<128x256xf32> -> !loom.view
              loom.copy_from_tensor %16, %19 to @DRAM : tensor<?x?xf32>, !loom.view
            } {loom.iter_type = #loom.iter_type<temporal>}
          } {loom.iter_type = #loom.iter_type<temporal>}
        } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @y}
      } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @x}
      return
    }
  }
  module attributes {loom.pass_name = "Materialize"} {
    func.func @matmul__d0i0_d1i0__f01__d_a__BK32__BM32__BN32(%arg0: memref<128x128xf32>, %arg1: memref<128x256xf32>, %arg2: memref<128x256xf32>) {
      %cst = arith.constant 0.000000e+00 : f32
      %c32 = arith.constant 32 : index
      affine.parallel (%arg3) = (0) to (8) {
        affine.parallel (%arg4) = (0) to (8) {
          affine.for %arg5 = 0 to 1 {
            affine.for %arg6 = 0 to 8 {
              %12 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 8 + d1 + d2 * 64)>(%arg3, %arg4, %arg5)
              %13 = loom.alloc(%c32, %c32) on @L1 : !loom.buffer_token
              %14 = loom.init_tensor %13(%c32, %c32) : !loom.buffer_token -> tensor<?x?xf32>
              %15 = linalg.fill ins(%cst : f32) outs(%14 : tensor<?x?xf32>) -> tensor<?x?xf32>
              %16 = affine.for %arg7 = 0 to 4 iter_args(%arg8 = %15) -> (tensor<?x?xf32>) {
                %20 = arith.muli %12, %c32 : index
                %21 = arith.muli %arg7, %c32 : index
                %22 = loom.view %arg0[%20, %21] [32, 32] [1, 1], reuse : [seq = false, spat = false, temp = true] : memref<128x128xf32> -> !loom.view
                %23 = loom.alloc(%c32, %c32) on @L1 : !loom.buffer_token
                %24 = loom.copy_to_tensor %22, %23 to @L1, interconnect : [], broadcast : [1, 1] : !loom.view, !loom.buffer_token -> tensor<?x?xf32>
                %25 = arith.muli %arg6, %c32 : index
                %26 = loom.view %arg1[%21, %25] [32, 32] [1, 1], reuse : [seq = false, spat = true, temp = true] : memref<128x256xf32> -> !loom.view
                %27 = loom.alloc(%c32, %c32) on @L1 : !loom.buffer_token
                %28 = loom.copy_to_tensor %26, %27 to @L1, interconnect : [@horizontal_links, @vertical_links], broadcast : [8, 8] : !loom.view, !loom.buffer_token -> tensor<?x?xf32>
                %29 = linalg.matmul ins(%24, %28 : tensor<?x?xf32>, tensor<?x?xf32>) outs(%15 : tensor<?x?xf32>) -> tensor<?x?xf32>
                %30 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%arg8, %29 : tensor<?x?xf32>, tensor<?x?xf32>) outs(%14 : tensor<?x?xf32>) {
                ^bb0(%in: f32, %in_0: f32, %out: f32):
                  %31 = arith.addf %in, %in_0 : f32
                  linalg.yield %31 : f32
                } -> tensor<?x?xf32>
                affine.yield %30 : tensor<?x?xf32>
              } {loom.iter_type = #loom.iter_type<sequential>}
              %17 = arith.muli %12, %c32 : index
              %18 = arith.muli %arg6, %c32 : index
              %19 = loom.view %arg2[%17, %18] [32, 32] [1, 1], reuse : [seq = false, spat = false, temp = false] : memref<128x256xf32> -> !loom.view
              loom.copy_from_tensor %16, %19 to @DRAM : tensor<?x?xf32>, !loom.view
            } {loom.iter_type = #loom.iter_type<temporal>}
          } {loom.iter_type = #loom.iter_type<temporal>}
        } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @y}
      } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @x}
      return
    }
    func.func @matmul__d0i0_d1i0__f01__d_a__BK64__BM32__BN32(%arg0: memref<128x128xf32>, %arg1: memref<128x256xf32>, %arg2: memref<128x256xf32>) {
      %cst = arith.constant 0.000000e+00 : f32
      %c32 = arith.constant 32 : index
      %c64 = arith.constant 64 : index
      affine.parallel (%arg3) = (0) to (8) {
        affine.parallel (%arg4) = (0) to (8) {
          affine.for %arg5 = 0 to 1 {
            affine.for %arg6 = 0 to 8 {
              %12 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 8 + d1 + d2 * 64)>(%arg3, %arg4, %arg5)
              %13 = loom.alloc(%c32, %c32) on @L1 : !loom.buffer_token
              %14 = loom.init_tensor %13(%c32, %c32) : !loom.buffer_token -> tensor<?x?xf32>
              %15 = linalg.fill ins(%cst : f32) outs(%14 : tensor<?x?xf32>) -> tensor<?x?xf32>
              %16 = affine.for %arg7 = 0 to 2 iter_args(%arg8 = %15) -> (tensor<?x?xf32>) {
                %20 = arith.muli %12, %c32 : index
                %21 = arith.muli %arg7, %c64 : index
                %22 = loom.view %arg0[%20, %21] [32, 64] [1, 1], reuse : [seq = false, spat = false, temp = true] : memref<128x128xf32> -> !loom.view
                %23 = loom.alloc(%c32, %c64) on @L1 : !loom.buffer_token
                %24 = loom.copy_to_tensor %22, %23 to @L1, interconnect : [], broadcast : [1, 1] : !loom.view, !loom.buffer_token -> tensor<?x?xf32>
                %25 = arith.muli %arg6, %c32 : index
                %26 = loom.view %arg1[%21, %25] [64, 32] [1, 1], reuse : [seq = false, spat = true, temp = true] : memref<128x256xf32> -> !loom.view
                %27 = loom.alloc(%c64, %c32) on @L1 : !loom.buffer_token
                %28 = loom.copy_to_tensor %26, %27 to @L1, interconnect : [@horizontal_links, @vertical_links], broadcast : [8, 8] : !loom.view, !loom.buffer_token -> tensor<?x?xf32>
                %29 = linalg.matmul ins(%24, %28 : tensor<?x?xf32>, tensor<?x?xf32>) outs(%15 : tensor<?x?xf32>) -> tensor<?x?xf32>
                %30 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%arg8, %29 : tensor<?x?xf32>, tensor<?x?xf32>) outs(%14 : tensor<?x?xf32>) {
                ^bb0(%in: f32, %in_0: f32, %out: f32):
                  %31 = arith.addf %in, %in_0 : f32
                  linalg.yield %31 : f32
                } -> tensor<?x?xf32>
                affine.yield %30 : tensor<?x?xf32>
              } {loom.iter_type = #loom.iter_type<sequential>}
              %17 = arith.muli %12, %c32 : index
              %18 = arith.muli %arg6, %c32 : index
              %19 = loom.view %arg2[%17, %18] [32, 32] [1, 1], reuse : [seq = false, spat = false, temp = false] : memref<128x256xf32> -> !loom.view
              loom.copy_from_tensor %16, %19 to @DRAM : tensor<?x?xf32>, !loom.view
            } {loom.iter_type = #loom.iter_type<temporal>}
          } {loom.iter_type = #loom.iter_type<temporal>}
        } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @y}
      } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @x}
      return
    }
  }
  module attributes {loom.pass_name = "Materialize"} {
    func.func @matmul__d0i0_d1i0__f01__d_h__BK32__BM32__BN32(%arg0: memref<128x128xf32>, %arg1: memref<128x256xf32>, %arg2: memref<128x256xf32>) {
      %cst = arith.constant 0.000000e+00 : f32
      %c32 = arith.constant 32 : index
      affine.parallel (%arg3) = (0) to (8) {
        affine.parallel (%arg4) = (0) to (8) {
          affine.for %arg5 = 0 to 1 {
            affine.for %arg6 = 0 to 8 {
              %12 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 8 + d1 + d2 * 64)>(%arg3, %arg4, %arg5)
              %13 = loom.alloc(%c32, %c32) on @L1 : !loom.buffer_token
              %14 = loom.init_tensor %13(%c32, %c32) : !loom.buffer_token -> tensor<?x?xf32>
              %15 = linalg.fill ins(%cst : f32) outs(%14 : tensor<?x?xf32>) -> tensor<?x?xf32>
              %16 = affine.for %arg7 = 0 to 4 iter_args(%arg8 = %15) -> (tensor<?x?xf32>) {
                %20 = arith.muli %12, %c32 : index
                %21 = arith.muli %arg7, %c32 : index
                %22 = loom.view %arg0[%20, %21] [32, 32] [1, 1], reuse : [seq = false, spat = false, temp = true] : memref<128x128xf32> -> !loom.view
                %23 = loom.alloc(%c32, %c32) on @L1 : !loom.buffer_token
                %24 = loom.copy_to_tensor %22, %23 to @L1, interconnect : [], broadcast : [1, 1] : !loom.view, !loom.buffer_token -> tensor<?x?xf32>
                %25 = arith.muli %arg6, %c32 : index
                %26 = loom.view %arg1[%21, %25] [32, 32] [1, 1], reuse : [seq = false, spat = true, temp = true] : memref<128x256xf32> -> !loom.view
                %27 = loom.alloc(%c32, %c32) on @L1 : !loom.buffer_token
                %28 = loom.copy_to_tensor %26, %27 to @L1, interconnect : [@horizontal_links], broadcast : [1, 8] : !loom.view, !loom.buffer_token -> tensor<?x?xf32>
                %29 = linalg.matmul ins(%24, %28 : tensor<?x?xf32>, tensor<?x?xf32>) outs(%15 : tensor<?x?xf32>) -> tensor<?x?xf32>
                %30 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%arg8, %29 : tensor<?x?xf32>, tensor<?x?xf32>) outs(%14 : tensor<?x?xf32>) {
                ^bb0(%in: f32, %in_0: f32, %out: f32):
                  %31 = arith.addf %in, %in_0 : f32
                  linalg.yield %31 : f32
                } -> tensor<?x?xf32>
                affine.yield %30 : tensor<?x?xf32>
              } {loom.iter_type = #loom.iter_type<sequential>}
              %17 = arith.muli %12, %c32 : index
              %18 = arith.muli %arg6, %c32 : index
              %19 = loom.view %arg2[%17, %18] [32, 32] [1, 1], reuse : [seq = false, spat = false, temp = false] : memref<128x256xf32> -> !loom.view
              loom.copy_from_tensor %16, %19 to @DRAM : tensor<?x?xf32>, !loom.view
            } {loom.iter_type = #loom.iter_type<temporal>}
          } {loom.iter_type = #loom.iter_type<temporal>}
        } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @y}
      } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @x}
      return
    }
    func.func @matmul__d0i0_d1i0__f01__d_h__BK64__BM32__BN32(%arg0: memref<128x128xf32>, %arg1: memref<128x256xf32>, %arg2: memref<128x256xf32>) {
      %cst = arith.constant 0.000000e+00 : f32
      %c32 = arith.constant 32 : index
      %c64 = arith.constant 64 : index
      affine.parallel (%arg3) = (0) to (8) {
        affine.parallel (%arg4) = (0) to (8) {
          affine.for %arg5 = 0 to 1 {
            affine.for %arg6 = 0 to 8 {
              %12 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 8 + d1 + d2 * 64)>(%arg3, %arg4, %arg5)
              %13 = loom.alloc(%c32, %c32) on @L1 : !loom.buffer_token
              %14 = loom.init_tensor %13(%c32, %c32) : !loom.buffer_token -> tensor<?x?xf32>
              %15 = linalg.fill ins(%cst : f32) outs(%14 : tensor<?x?xf32>) -> tensor<?x?xf32>
              %16 = affine.for %arg7 = 0 to 2 iter_args(%arg8 = %15) -> (tensor<?x?xf32>) {
                %20 = arith.muli %12, %c32 : index
                %21 = arith.muli %arg7, %c64 : index
                %22 = loom.view %arg0[%20, %21] [32, 64] [1, 1], reuse : [seq = false, spat = false, temp = true] : memref<128x128xf32> -> !loom.view
                %23 = loom.alloc(%c32, %c64) on @L1 : !loom.buffer_token
                %24 = loom.copy_to_tensor %22, %23 to @L1, interconnect : [], broadcast : [1, 1] : !loom.view, !loom.buffer_token -> tensor<?x?xf32>
                %25 = arith.muli %arg6, %c32 : index
                %26 = loom.view %arg1[%21, %25] [64, 32] [1, 1], reuse : [seq = false, spat = true, temp = true] : memref<128x256xf32> -> !loom.view
                %27 = loom.alloc(%c64, %c32) on @L1 : !loom.buffer_token
                %28 = loom.copy_to_tensor %26, %27 to @L1, interconnect : [@horizontal_links], broadcast : [1, 8] : !loom.view, !loom.buffer_token -> tensor<?x?xf32>
                %29 = linalg.matmul ins(%24, %28 : tensor<?x?xf32>, tensor<?x?xf32>) outs(%15 : tensor<?x?xf32>) -> tensor<?x?xf32>
                %30 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%arg8, %29 : tensor<?x?xf32>, tensor<?x?xf32>) outs(%14 : tensor<?x?xf32>) {
                ^bb0(%in: f32, %in_0: f32, %out: f32):
                  %31 = arith.addf %in, %in_0 : f32
                  linalg.yield %31 : f32
                } -> tensor<?x?xf32>
                affine.yield %30 : tensor<?x?xf32>
              } {loom.iter_type = #loom.iter_type<sequential>}
              %17 = arith.muli %12, %c32 : index
              %18 = arith.muli %arg6, %c32 : index
              %19 = loom.view %arg2[%17, %18] [32, 32] [1, 1], reuse : [seq = false, spat = false, temp = false] : memref<128x256xf32> -> !loom.view
              loom.copy_from_tensor %16, %19 to @DRAM : tensor<?x?xf32>, !loom.view
            } {loom.iter_type = #loom.iter_type<temporal>}
          } {loom.iter_type = #loom.iter_type<temporal>}
        } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @y}
      } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @x}
      return
    }
  }
  module attributes {loom.pass_name = "Materialize"} {
    func.func @matmul__d0i0_d1i0__f01__d_v__BK32__BM32__BN32(%arg0: memref<128x128xf32>, %arg1: memref<128x256xf32>, %arg2: memref<128x256xf32>) {
      %cst = arith.constant 0.000000e+00 : f32
      %c32 = arith.constant 32 : index
      affine.parallel (%arg3) = (0) to (8) {
        affine.parallel (%arg4) = (0) to (8) {
          affine.for %arg5 = 0 to 1 {
            affine.for %arg6 = 0 to 8 {
              %12 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 8 + d1 + d2 * 64)>(%arg3, %arg4, %arg5)
              %13 = loom.alloc(%c32, %c32) on @L1 : !loom.buffer_token
              %14 = loom.init_tensor %13(%c32, %c32) : !loom.buffer_token -> tensor<?x?xf32>
              %15 = linalg.fill ins(%cst : f32) outs(%14 : tensor<?x?xf32>) -> tensor<?x?xf32>
              %16 = affine.for %arg7 = 0 to 4 iter_args(%arg8 = %15) -> (tensor<?x?xf32>) {
                %20 = arith.muli %12, %c32 : index
                %21 = arith.muli %arg7, %c32 : index
                %22 = loom.view %arg0[%20, %21] [32, 32] [1, 1], reuse : [seq = false, spat = false, temp = true] : memref<128x128xf32> -> !loom.view
                %23 = loom.alloc(%c32, %c32) on @L1 : !loom.buffer_token
                %24 = loom.copy_to_tensor %22, %23 to @L1, interconnect : [], broadcast : [1, 1] : !loom.view, !loom.buffer_token -> tensor<?x?xf32>
                %25 = arith.muli %arg6, %c32 : index
                %26 = loom.view %arg1[%21, %25] [32, 32] [1, 1], reuse : [seq = false, spat = true, temp = true] : memref<128x256xf32> -> !loom.view
                %27 = loom.alloc(%c32, %c32) on @L1 : !loom.buffer_token
                %28 = loom.copy_to_tensor %26, %27 to @L1, interconnect : [@vertical_links], broadcast : [8, 1] : !loom.view, !loom.buffer_token -> tensor<?x?xf32>
                %29 = linalg.matmul ins(%24, %28 : tensor<?x?xf32>, tensor<?x?xf32>) outs(%15 : tensor<?x?xf32>) -> tensor<?x?xf32>
                %30 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%arg8, %29 : tensor<?x?xf32>, tensor<?x?xf32>) outs(%14 : tensor<?x?xf32>) {
                ^bb0(%in: f32, %in_0: f32, %out: f32):
                  %31 = arith.addf %in, %in_0 : f32
                  linalg.yield %31 : f32
                } -> tensor<?x?xf32>
                affine.yield %30 : tensor<?x?xf32>
              } {loom.iter_type = #loom.iter_type<sequential>}
              %17 = arith.muli %12, %c32 : index
              %18 = arith.muli %arg6, %c32 : index
              %19 = loom.view %arg2[%17, %18] [32, 32] [1, 1], reuse : [seq = false, spat = false, temp = false] : memref<128x256xf32> -> !loom.view
              loom.copy_from_tensor %16, %19 to @DRAM : tensor<?x?xf32>, !loom.view
            } {loom.iter_type = #loom.iter_type<temporal>}
          } {loom.iter_type = #loom.iter_type<temporal>}
        } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @y}
      } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @x}
      return
    }
    func.func @matmul__d0i0_d1i0__f01__d_v__BK64__BM32__BN32(%arg0: memref<128x128xf32>, %arg1: memref<128x256xf32>, %arg2: memref<128x256xf32>) {
      %cst = arith.constant 0.000000e+00 : f32
      %c32 = arith.constant 32 : index
      %c64 = arith.constant 64 : index
      affine.parallel (%arg3) = (0) to (8) {
        affine.parallel (%arg4) = (0) to (8) {
          affine.for %arg5 = 0 to 1 {
            affine.for %arg6 = 0 to 8 {
              %12 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 8 + d1 + d2 * 64)>(%arg3, %arg4, %arg5)
              %13 = loom.alloc(%c32, %c32) on @L1 : !loom.buffer_token
              %14 = loom.init_tensor %13(%c32, %c32) : !loom.buffer_token -> tensor<?x?xf32>
              %15 = linalg.fill ins(%cst : f32) outs(%14 : tensor<?x?xf32>) -> tensor<?x?xf32>
              %16 = affine.for %arg7 = 0 to 2 iter_args(%arg8 = %15) -> (tensor<?x?xf32>) {
                %20 = arith.muli %12, %c32 : index
                %21 = arith.muli %arg7, %c64 : index
                %22 = loom.view %arg0[%20, %21] [32, 64] [1, 1], reuse : [seq = false, spat = false, temp = true] : memref<128x128xf32> -> !loom.view
                %23 = loom.alloc(%c32, %c64) on @L1 : !loom.buffer_token
                %24 = loom.copy_to_tensor %22, %23 to @L1, interconnect : [], broadcast : [1, 1] : !loom.view, !loom.buffer_token -> tensor<?x?xf32>
                %25 = arith.muli %arg6, %c32 : index
                %26 = loom.view %arg1[%21, %25] [64, 32] [1, 1], reuse : [seq = false, spat = true, temp = true] : memref<128x256xf32> -> !loom.view
                %27 = loom.alloc(%c64, %c32) on @L1 : !loom.buffer_token
                %28 = loom.copy_to_tensor %26, %27 to @L1, interconnect : [@vertical_links], broadcast : [8, 1] : !loom.view, !loom.buffer_token -> tensor<?x?xf32>
                %29 = linalg.matmul ins(%24, %28 : tensor<?x?xf32>, tensor<?x?xf32>) outs(%15 : tensor<?x?xf32>) -> tensor<?x?xf32>
                %30 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%arg8, %29 : tensor<?x?xf32>, tensor<?x?xf32>) outs(%14 : tensor<?x?xf32>) {
                ^bb0(%in: f32, %in_0: f32, %out: f32):
                  %31 = arith.addf %in, %in_0 : f32
                  linalg.yield %31 : f32
                } -> tensor<?x?xf32>
                affine.yield %30 : tensor<?x?xf32>
              } {loom.iter_type = #loom.iter_type<sequential>}
              %17 = arith.muli %12, %c32 : index
              %18 = arith.muli %arg6, %c32 : index
              %19 = loom.view %arg2[%17, %18] [32, 32] [1, 1], reuse : [seq = false, spat = false, temp = false] : memref<128x256xf32> -> !loom.view
              loom.copy_from_tensor %16, %19 to @DRAM : tensor<?x?xf32>, !loom.view
            } {loom.iter_type = #loom.iter_type<temporal>}
          } {loom.iter_type = #loom.iter_type<temporal>}
        } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @y}
      } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @x}
      return
    }
  }
  module attributes {loom.pass_name = "Materialize"} {
    func.func @matmul__d0i0_d1i0__f10__d_d__BK32__BM32__BN32(%arg0: memref<128x128xf32>, %arg1: memref<128x256xf32>, %arg2: memref<128x256xf32>) {
      %cst = arith.constant 0.000000e+00 : f32
      %c32 = arith.constant 32 : index
      affine.parallel (%arg3) = (0) to (8) {
        affine.parallel (%arg4) = (0) to (8) {
          affine.for %arg5 = 0 to 8 {
            affine.for %arg6 = 0 to 1 {
              %12 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 8 + d1 + d2 * 64)>(%arg3, %arg4, %arg6)
              %13 = loom.alloc(%c32, %c32) on @L1 : !loom.buffer_token
              %14 = loom.init_tensor %13(%c32, %c32) : !loom.buffer_token -> tensor<?x?xf32>
              %15 = linalg.fill ins(%cst : f32) outs(%14 : tensor<?x?xf32>) -> tensor<?x?xf32>
              %16 = affine.for %arg7 = 0 to 4 iter_args(%arg8 = %15) -> (tensor<?x?xf32>) {
                %20 = arith.muli %12, %c32 : index
                %21 = arith.muli %arg7, %c32 : index
                %22 = loom.view %arg0[%20, %21] [32, 32] [1, 1], reuse : [seq = false, spat = false, temp = true] : memref<128x128xf32> -> !loom.view
                %23 = loom.alloc(%c32, %c32) on @L1 : !loom.buffer_token
                %24 = loom.copy_to_tensor %22, %23 to @L1, interconnect : [], broadcast : [1, 1] : !loom.view, !loom.buffer_token -> tensor<?x?xf32>
                %25 = arith.muli %arg5, %c32 : index
                %26 = loom.view %arg1[%21, %25] [32, 32] [1, 1], reuse : [seq = false, spat = true, temp = true] : memref<128x256xf32> -> !loom.view
                %27 = loom.alloc(%c32, %c32) on @L1 : !loom.buffer_token
                %28 = loom.copy_to_tensor %26, %27 to @L1, interconnect : [], broadcast : [1, 1] : !loom.view, !loom.buffer_token -> tensor<?x?xf32>
                %29 = linalg.matmul ins(%24, %28 : tensor<?x?xf32>, tensor<?x?xf32>) outs(%15 : tensor<?x?xf32>) -> tensor<?x?xf32>
                %30 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%arg8, %29 : tensor<?x?xf32>, tensor<?x?xf32>) outs(%14 : tensor<?x?xf32>) {
                ^bb0(%in: f32, %in_0: f32, %out: f32):
                  %31 = arith.addf %in, %in_0 : f32
                  linalg.yield %31 : f32
                } -> tensor<?x?xf32>
                affine.yield %30 : tensor<?x?xf32>
              } {loom.iter_type = #loom.iter_type<sequential>}
              %17 = arith.muli %12, %c32 : index
              %18 = arith.muli %arg5, %c32 : index
              %19 = loom.view %arg2[%17, %18] [32, 32] [1, 1], reuse : [seq = false, spat = false, temp = false] : memref<128x256xf32> -> !loom.view
              loom.copy_from_tensor %16, %19 to @DRAM : tensor<?x?xf32>, !loom.view
            } {loom.iter_type = #loom.iter_type<temporal>}
          } {loom.iter_type = #loom.iter_type<temporal>}
        } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @y}
      } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @x}
      return
    }
    func.func @matmul__d0i0_d1i0__f10__d_d__BK64__BM32__BN32(%arg0: memref<128x128xf32>, %arg1: memref<128x256xf32>, %arg2: memref<128x256xf32>) {
      %cst = arith.constant 0.000000e+00 : f32
      %c32 = arith.constant 32 : index
      %c64 = arith.constant 64 : index
      affine.parallel (%arg3) = (0) to (8) {
        affine.parallel (%arg4) = (0) to (8) {
          affine.for %arg5 = 0 to 8 {
            affine.for %arg6 = 0 to 1 {
              %12 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 8 + d1 + d2 * 64)>(%arg3, %arg4, %arg6)
              %13 = loom.alloc(%c32, %c32) on @L1 : !loom.buffer_token
              %14 = loom.init_tensor %13(%c32, %c32) : !loom.buffer_token -> tensor<?x?xf32>
              %15 = linalg.fill ins(%cst : f32) outs(%14 : tensor<?x?xf32>) -> tensor<?x?xf32>
              %16 = affine.for %arg7 = 0 to 2 iter_args(%arg8 = %15) -> (tensor<?x?xf32>) {
                %20 = arith.muli %12, %c32 : index
                %21 = arith.muli %arg7, %c64 : index
                %22 = loom.view %arg0[%20, %21] [32, 64] [1, 1], reuse : [seq = false, spat = false, temp = true] : memref<128x128xf32> -> !loom.view
                %23 = loom.alloc(%c32, %c64) on @L1 : !loom.buffer_token
                %24 = loom.copy_to_tensor %22, %23 to @L1, interconnect : [], broadcast : [1, 1] : !loom.view, !loom.buffer_token -> tensor<?x?xf32>
                %25 = arith.muli %arg5, %c32 : index
                %26 = loom.view %arg1[%21, %25] [64, 32] [1, 1], reuse : [seq = false, spat = true, temp = true] : memref<128x256xf32> -> !loom.view
                %27 = loom.alloc(%c64, %c32) on @L1 : !loom.buffer_token
                %28 = loom.copy_to_tensor %26, %27 to @L1, interconnect : [], broadcast : [1, 1] : !loom.view, !loom.buffer_token -> tensor<?x?xf32>
                %29 = linalg.matmul ins(%24, %28 : tensor<?x?xf32>, tensor<?x?xf32>) outs(%15 : tensor<?x?xf32>) -> tensor<?x?xf32>
                %30 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%arg8, %29 : tensor<?x?xf32>, tensor<?x?xf32>) outs(%14 : tensor<?x?xf32>) {
                ^bb0(%in: f32, %in_0: f32, %out: f32):
                  %31 = arith.addf %in, %in_0 : f32
                  linalg.yield %31 : f32
                } -> tensor<?x?xf32>
                affine.yield %30 : tensor<?x?xf32>
              } {loom.iter_type = #loom.iter_type<sequential>}
              %17 = arith.muli %12, %c32 : index
              %18 = arith.muli %arg5, %c32 : index
              %19 = loom.view %arg2[%17, %18] [32, 32] [1, 1], reuse : [seq = false, spat = false, temp = false] : memref<128x256xf32> -> !loom.view
              loom.copy_from_tensor %16, %19 to @DRAM : tensor<?x?xf32>, !loom.view
            } {loom.iter_type = #loom.iter_type<temporal>}
          } {loom.iter_type = #loom.iter_type<temporal>}
        } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @y}
      } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @x}
      return
    }
  }
  module attributes {loom.pass_name = "Materialize"} {
    func.func @matmul__d0i0_d1i0__f10__d_a__BK32__BM32__BN32(%arg0: memref<128x128xf32>, %arg1: memref<128x256xf32>, %arg2: memref<128x256xf32>) {
      %cst = arith.constant 0.000000e+00 : f32
      %c32 = arith.constant 32 : index
      affine.parallel (%arg3) = (0) to (8) {
        affine.parallel (%arg4) = (0) to (8) {
          affine.for %arg5 = 0 to 8 {
            affine.for %arg6 = 0 to 1 {
              %12 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 8 + d1 + d2 * 64)>(%arg3, %arg4, %arg6)
              %13 = loom.alloc(%c32, %c32) on @L1 : !loom.buffer_token
              %14 = loom.init_tensor %13(%c32, %c32) : !loom.buffer_token -> tensor<?x?xf32>
              %15 = linalg.fill ins(%cst : f32) outs(%14 : tensor<?x?xf32>) -> tensor<?x?xf32>
              %16 = affine.for %arg7 = 0 to 4 iter_args(%arg8 = %15) -> (tensor<?x?xf32>) {
                %20 = arith.muli %12, %c32 : index
                %21 = arith.muli %arg7, %c32 : index
                %22 = loom.view %arg0[%20, %21] [32, 32] [1, 1], reuse : [seq = false, spat = false, temp = true] : memref<128x128xf32> -> !loom.view
                %23 = loom.alloc(%c32, %c32) on @L1 : !loom.buffer_token
                %24 = loom.copy_to_tensor %22, %23 to @L1, interconnect : [], broadcast : [1, 1] : !loom.view, !loom.buffer_token -> tensor<?x?xf32>
                %25 = arith.muli %arg5, %c32 : index
                %26 = loom.view %arg1[%21, %25] [32, 32] [1, 1], reuse : [seq = false, spat = true, temp = true] : memref<128x256xf32> -> !loom.view
                %27 = loom.alloc(%c32, %c32) on @L1 : !loom.buffer_token
                %28 = loom.copy_to_tensor %26, %27 to @L1, interconnect : [@horizontal_links, @vertical_links], broadcast : [8, 8] : !loom.view, !loom.buffer_token -> tensor<?x?xf32>
                %29 = linalg.matmul ins(%24, %28 : tensor<?x?xf32>, tensor<?x?xf32>) outs(%15 : tensor<?x?xf32>) -> tensor<?x?xf32>
                %30 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%arg8, %29 : tensor<?x?xf32>, tensor<?x?xf32>) outs(%14 : tensor<?x?xf32>) {
                ^bb0(%in: f32, %in_0: f32, %out: f32):
                  %31 = arith.addf %in, %in_0 : f32
                  linalg.yield %31 : f32
                } -> tensor<?x?xf32>
                affine.yield %30 : tensor<?x?xf32>
              } {loom.iter_type = #loom.iter_type<sequential>}
              %17 = arith.muli %12, %c32 : index
              %18 = arith.muli %arg5, %c32 : index
              %19 = loom.view %arg2[%17, %18] [32, 32] [1, 1], reuse : [seq = false, spat = false, temp = false] : memref<128x256xf32> -> !loom.view
              loom.copy_from_tensor %16, %19 to @DRAM : tensor<?x?xf32>, !loom.view
            } {loom.iter_type = #loom.iter_type<temporal>}
          } {loom.iter_type = #loom.iter_type<temporal>}
        } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @y}
      } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @x}
      return
    }
    func.func @matmul__d0i0_d1i0__f10__d_a__BK64__BM32__BN32(%arg0: memref<128x128xf32>, %arg1: memref<128x256xf32>, %arg2: memref<128x256xf32>) {
      %cst = arith.constant 0.000000e+00 : f32
      %c32 = arith.constant 32 : index
      %c64 = arith.constant 64 : index
      affine.parallel (%arg3) = (0) to (8) {
        affine.parallel (%arg4) = (0) to (8) {
          affine.for %arg5 = 0 to 8 {
            affine.for %arg6 = 0 to 1 {
              %12 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 8 + d1 + d2 * 64)>(%arg3, %arg4, %arg6)
              %13 = loom.alloc(%c32, %c32) on @L1 : !loom.buffer_token
              %14 = loom.init_tensor %13(%c32, %c32) : !loom.buffer_token -> tensor<?x?xf32>
              %15 = linalg.fill ins(%cst : f32) outs(%14 : tensor<?x?xf32>) -> tensor<?x?xf32>
              %16 = affine.for %arg7 = 0 to 2 iter_args(%arg8 = %15) -> (tensor<?x?xf32>) {
                %20 = arith.muli %12, %c32 : index
                %21 = arith.muli %arg7, %c64 : index
                %22 = loom.view %arg0[%20, %21] [32, 64] [1, 1], reuse : [seq = false, spat = false, temp = true] : memref<128x128xf32> -> !loom.view
                %23 = loom.alloc(%c32, %c64) on @L1 : !loom.buffer_token
                %24 = loom.copy_to_tensor %22, %23 to @L1, interconnect : [], broadcast : [1, 1] : !loom.view, !loom.buffer_token -> tensor<?x?xf32>
                %25 = arith.muli %arg5, %c32 : index
                %26 = loom.view %arg1[%21, %25] [64, 32] [1, 1], reuse : [seq = false, spat = true, temp = true] : memref<128x256xf32> -> !loom.view
                %27 = loom.alloc(%c64, %c32) on @L1 : !loom.buffer_token
                %28 = loom.copy_to_tensor %26, %27 to @L1, interconnect : [@horizontal_links, @vertical_links], broadcast : [8, 8] : !loom.view, !loom.buffer_token -> tensor<?x?xf32>
                %29 = linalg.matmul ins(%24, %28 : tensor<?x?xf32>, tensor<?x?xf32>) outs(%15 : tensor<?x?xf32>) -> tensor<?x?xf32>
                %30 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%arg8, %29 : tensor<?x?xf32>, tensor<?x?xf32>) outs(%14 : tensor<?x?xf32>) {
                ^bb0(%in: f32, %in_0: f32, %out: f32):
                  %31 = arith.addf %in, %in_0 : f32
                  linalg.yield %31 : f32
                } -> tensor<?x?xf32>
                affine.yield %30 : tensor<?x?xf32>
              } {loom.iter_type = #loom.iter_type<sequential>}
              %17 = arith.muli %12, %c32 : index
              %18 = arith.muli %arg5, %c32 : index
              %19 = loom.view %arg2[%17, %18] [32, 32] [1, 1], reuse : [seq = false, spat = false, temp = false] : memref<128x256xf32> -> !loom.view
              loom.copy_from_tensor %16, %19 to @DRAM : tensor<?x?xf32>, !loom.view
            } {loom.iter_type = #loom.iter_type<temporal>}
          } {loom.iter_type = #loom.iter_type<temporal>}
        } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @y}
      } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @x}
      return
    }
  }
  module attributes {loom.pass_name = "Materialize"} {
    func.func @matmul__d0i0_d1i0__f10__d_h__BK32__BM32__BN32(%arg0: memref<128x128xf32>, %arg1: memref<128x256xf32>, %arg2: memref<128x256xf32>) {
      %cst = arith.constant 0.000000e+00 : f32
      %c32 = arith.constant 32 : index
      affine.parallel (%arg3) = (0) to (8) {
        affine.parallel (%arg4) = (0) to (8) {
          affine.for %arg5 = 0 to 8 {
            affine.for %arg6 = 0 to 1 {
              %12 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 8 + d1 + d2 * 64)>(%arg3, %arg4, %arg6)
              %13 = loom.alloc(%c32, %c32) on @L1 : !loom.buffer_token
              %14 = loom.init_tensor %13(%c32, %c32) : !loom.buffer_token -> tensor<?x?xf32>
              %15 = linalg.fill ins(%cst : f32) outs(%14 : tensor<?x?xf32>) -> tensor<?x?xf32>
              %16 = affine.for %arg7 = 0 to 4 iter_args(%arg8 = %15) -> (tensor<?x?xf32>) {
                %20 = arith.muli %12, %c32 : index
                %21 = arith.muli %arg7, %c32 : index
                %22 = loom.view %arg0[%20, %21] [32, 32] [1, 1], reuse : [seq = false, spat = false, temp = true] : memref<128x128xf32> -> !loom.view
                %23 = loom.alloc(%c32, %c32) on @L1 : !loom.buffer_token
                %24 = loom.copy_to_tensor %22, %23 to @L1, interconnect : [], broadcast : [1, 1] : !loom.view, !loom.buffer_token -> tensor<?x?xf32>
                %25 = arith.muli %arg5, %c32 : index
                %26 = loom.view %arg1[%21, %25] [32, 32] [1, 1], reuse : [seq = false, spat = true, temp = true] : memref<128x256xf32> -> !loom.view
                %27 = loom.alloc(%c32, %c32) on @L1 : !loom.buffer_token
                %28 = loom.copy_to_tensor %26, %27 to @L1, interconnect : [@horizontal_links], broadcast : [1, 8] : !loom.view, !loom.buffer_token -> tensor<?x?xf32>
                %29 = linalg.matmul ins(%24, %28 : tensor<?x?xf32>, tensor<?x?xf32>) outs(%15 : tensor<?x?xf32>) -> tensor<?x?xf32>
                %30 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%arg8, %29 : tensor<?x?xf32>, tensor<?x?xf32>) outs(%14 : tensor<?x?xf32>) {
                ^bb0(%in: f32, %in_0: f32, %out: f32):
                  %31 = arith.addf %in, %in_0 : f32
                  linalg.yield %31 : f32
                } -> tensor<?x?xf32>
                affine.yield %30 : tensor<?x?xf32>
              } {loom.iter_type = #loom.iter_type<sequential>}
              %17 = arith.muli %12, %c32 : index
              %18 = arith.muli %arg5, %c32 : index
              %19 = loom.view %arg2[%17, %18] [32, 32] [1, 1], reuse : [seq = false, spat = false, temp = false] : memref<128x256xf32> -> !loom.view
              loom.copy_from_tensor %16, %19 to @DRAM : tensor<?x?xf32>, !loom.view
            } {loom.iter_type = #loom.iter_type<temporal>}
          } {loom.iter_type = #loom.iter_type<temporal>}
        } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @y}
      } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @x}
      return
    }
    func.func @matmul__d0i0_d1i0__f10__d_h__BK64__BM32__BN32(%arg0: memref<128x128xf32>, %arg1: memref<128x256xf32>, %arg2: memref<128x256xf32>) {
      %cst = arith.constant 0.000000e+00 : f32
      %c32 = arith.constant 32 : index
      %c64 = arith.constant 64 : index
      affine.parallel (%arg3) = (0) to (8) {
        affine.parallel (%arg4) = (0) to (8) {
          affine.for %arg5 = 0 to 8 {
            affine.for %arg6 = 0 to 1 {
              %12 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 8 + d1 + d2 * 64)>(%arg3, %arg4, %arg6)
              %13 = loom.alloc(%c32, %c32) on @L1 : !loom.buffer_token
              %14 = loom.init_tensor %13(%c32, %c32) : !loom.buffer_token -> tensor<?x?xf32>
              %15 = linalg.fill ins(%cst : f32) outs(%14 : tensor<?x?xf32>) -> tensor<?x?xf32>
              %16 = affine.for %arg7 = 0 to 2 iter_args(%arg8 = %15) -> (tensor<?x?xf32>) {
                %20 = arith.muli %12, %c32 : index
                %21 = arith.muli %arg7, %c64 : index
                %22 = loom.view %arg0[%20, %21] [32, 64] [1, 1], reuse : [seq = false, spat = false, temp = true] : memref<128x128xf32> -> !loom.view
                %23 = loom.alloc(%c32, %c64) on @L1 : !loom.buffer_token
                %24 = loom.copy_to_tensor %22, %23 to @L1, interconnect : [], broadcast : [1, 1] : !loom.view, !loom.buffer_token -> tensor<?x?xf32>
                %25 = arith.muli %arg5, %c32 : index
                %26 = loom.view %arg1[%21, %25] [64, 32] [1, 1], reuse : [seq = false, spat = true, temp = true] : memref<128x256xf32> -> !loom.view
                %27 = loom.alloc(%c64, %c32) on @L1 : !loom.buffer_token
                %28 = loom.copy_to_tensor %26, %27 to @L1, interconnect : [@horizontal_links], broadcast : [1, 8] : !loom.view, !loom.buffer_token -> tensor<?x?xf32>
                %29 = linalg.matmul ins(%24, %28 : tensor<?x?xf32>, tensor<?x?xf32>) outs(%15 : tensor<?x?xf32>) -> tensor<?x?xf32>
                %30 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%arg8, %29 : tensor<?x?xf32>, tensor<?x?xf32>) outs(%14 : tensor<?x?xf32>) {
                ^bb0(%in: f32, %in_0: f32, %out: f32):
                  %31 = arith.addf %in, %in_0 : f32
                  linalg.yield %31 : f32
                } -> tensor<?x?xf32>
                affine.yield %30 : tensor<?x?xf32>
              } {loom.iter_type = #loom.iter_type<sequential>}
              %17 = arith.muli %12, %c32 : index
              %18 = arith.muli %arg5, %c32 : index
              %19 = loom.view %arg2[%17, %18] [32, 32] [1, 1], reuse : [seq = false, spat = false, temp = false] : memref<128x256xf32> -> !loom.view
              loom.copy_from_tensor %16, %19 to @DRAM : tensor<?x?xf32>, !loom.view
            } {loom.iter_type = #loom.iter_type<temporal>}
          } {loom.iter_type = #loom.iter_type<temporal>}
        } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @y}
      } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @x}
      return
    }
  }
  module attributes {loom.pass_name = "Materialize"} {
    func.func @matmul__d0i0_d1i0__f10__d_v__BK32__BM32__BN32(%arg0: memref<128x128xf32>, %arg1: memref<128x256xf32>, %arg2: memref<128x256xf32>) {
      %cst = arith.constant 0.000000e+00 : f32
      %c32 = arith.constant 32 : index
      affine.parallel (%arg3) = (0) to (8) {
        affine.parallel (%arg4) = (0) to (8) {
          affine.for %arg5 = 0 to 8 {
            affine.for %arg6 = 0 to 1 {
              %12 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 8 + d1 + d2 * 64)>(%arg3, %arg4, %arg6)
              %13 = loom.alloc(%c32, %c32) on @L1 : !loom.buffer_token
              %14 = loom.init_tensor %13(%c32, %c32) : !loom.buffer_token -> tensor<?x?xf32>
              %15 = linalg.fill ins(%cst : f32) outs(%14 : tensor<?x?xf32>) -> tensor<?x?xf32>
              %16 = affine.for %arg7 = 0 to 4 iter_args(%arg8 = %15) -> (tensor<?x?xf32>) {
                %20 = arith.muli %12, %c32 : index
                %21 = arith.muli %arg7, %c32 : index
                %22 = loom.view %arg0[%20, %21] [32, 32] [1, 1], reuse : [seq = false, spat = false, temp = true] : memref<128x128xf32> -> !loom.view
                %23 = loom.alloc(%c32, %c32) on @L1 : !loom.buffer_token
                %24 = loom.copy_to_tensor %22, %23 to @L1, interconnect : [], broadcast : [1, 1] : !loom.view, !loom.buffer_token -> tensor<?x?xf32>
                %25 = arith.muli %arg5, %c32 : index
                %26 = loom.view %arg1[%21, %25] [32, 32] [1, 1], reuse : [seq = false, spat = true, temp = true] : memref<128x256xf32> -> !loom.view
                %27 = loom.alloc(%c32, %c32) on @L1 : !loom.buffer_token
                %28 = loom.copy_to_tensor %26, %27 to @L1, interconnect : [@vertical_links], broadcast : [8, 1] : !loom.view, !loom.buffer_token -> tensor<?x?xf32>
                %29 = linalg.matmul ins(%24, %28 : tensor<?x?xf32>, tensor<?x?xf32>) outs(%15 : tensor<?x?xf32>) -> tensor<?x?xf32>
                %30 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%arg8, %29 : tensor<?x?xf32>, tensor<?x?xf32>) outs(%14 : tensor<?x?xf32>) {
                ^bb0(%in: f32, %in_0: f32, %out: f32):
                  %31 = arith.addf %in, %in_0 : f32
                  linalg.yield %31 : f32
                } -> tensor<?x?xf32>
                affine.yield %30 : tensor<?x?xf32>
              } {loom.iter_type = #loom.iter_type<sequential>}
              %17 = arith.muli %12, %c32 : index
              %18 = arith.muli %arg5, %c32 : index
              %19 = loom.view %arg2[%17, %18] [32, 32] [1, 1], reuse : [seq = false, spat = false, temp = false] : memref<128x256xf32> -> !loom.view
              loom.copy_from_tensor %16, %19 to @DRAM : tensor<?x?xf32>, !loom.view
            } {loom.iter_type = #loom.iter_type<temporal>}
          } {loom.iter_type = #loom.iter_type<temporal>}
        } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @y}
      } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @x}
      return
    }
    func.func @matmul__d0i0_d1i0__f10__d_v__BK64__BM32__BN32(%arg0: memref<128x128xf32>, %arg1: memref<128x256xf32>, %arg2: memref<128x256xf32>) {
      %cst = arith.constant 0.000000e+00 : f32
      %c32 = arith.constant 32 : index
      %c64 = arith.constant 64 : index
      affine.parallel (%arg3) = (0) to (8) {
        affine.parallel (%arg4) = (0) to (8) {
          affine.for %arg5 = 0 to 8 {
            affine.for %arg6 = 0 to 1 {
              %12 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 8 + d1 + d2 * 64)>(%arg3, %arg4, %arg6)
              %13 = loom.alloc(%c32, %c32) on @L1 : !loom.buffer_token
              %14 = loom.init_tensor %13(%c32, %c32) : !loom.buffer_token -> tensor<?x?xf32>
              %15 = linalg.fill ins(%cst : f32) outs(%14 : tensor<?x?xf32>) -> tensor<?x?xf32>
              %16 = affine.for %arg7 = 0 to 2 iter_args(%arg8 = %15) -> (tensor<?x?xf32>) {
                %20 = arith.muli %12, %c32 : index
                %21 = arith.muli %arg7, %c64 : index
                %22 = loom.view %arg0[%20, %21] [32, 64] [1, 1], reuse : [seq = false, spat = false, temp = true] : memref<128x128xf32> -> !loom.view
                %23 = loom.alloc(%c32, %c64) on @L1 : !loom.buffer_token
                %24 = loom.copy_to_tensor %22, %23 to @L1, interconnect : [], broadcast : [1, 1] : !loom.view, !loom.buffer_token -> tensor<?x?xf32>
                %25 = arith.muli %arg5, %c32 : index
                %26 = loom.view %arg1[%21, %25] [64, 32] [1, 1], reuse : [seq = false, spat = true, temp = true] : memref<128x256xf32> -> !loom.view
                %27 = loom.alloc(%c64, %c32) on @L1 : !loom.buffer_token
                %28 = loom.copy_to_tensor %26, %27 to @L1, interconnect : [@vertical_links], broadcast : [8, 1] : !loom.view, !loom.buffer_token -> tensor<?x?xf32>
                %29 = linalg.matmul ins(%24, %28 : tensor<?x?xf32>, tensor<?x?xf32>) outs(%15 : tensor<?x?xf32>) -> tensor<?x?xf32>
                %30 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%arg8, %29 : tensor<?x?xf32>, tensor<?x?xf32>) outs(%14 : tensor<?x?xf32>) {
                ^bb0(%in: f32, %in_0: f32, %out: f32):
                  %31 = arith.addf %in, %in_0 : f32
                  linalg.yield %31 : f32
                } -> tensor<?x?xf32>
                affine.yield %30 : tensor<?x?xf32>
              } {loom.iter_type = #loom.iter_type<sequential>}
              %17 = arith.muli %12, %c32 : index
              %18 = arith.muli %arg5, %c32 : index
              %19 = loom.view %arg2[%17, %18] [32, 32] [1, 1], reuse : [seq = false, spat = false, temp = false] : memref<128x256xf32> -> !loom.view
              loom.copy_from_tensor %16, %19 to @DRAM : tensor<?x?xf32>, !loom.view
            } {loom.iter_type = #loom.iter_type<temporal>}
          } {loom.iter_type = #loom.iter_type<temporal>}
        } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @y}
      } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @x}
      return
    }
  }
  module attributes {loom.pass_name = "Materialize"} {
    func.func @matmul__d0i0_d1i1__f01__d_d__BK32__BM32__BN32(%arg0: memref<128x128xf32>, %arg1: memref<128x256xf32>, %arg2: memref<128x256xf32>) {
      %cst = arith.constant 0.000000e+00 : f32
      %c32 = arith.constant 32 : index
      affine.parallel (%arg3) = (0) to (8) {
        affine.parallel (%arg4) = (0) to (8) {
          affine.for %arg5 = 0 to 1 {
            affine.for %arg6 = 0 to 1 {
              %12 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 8)>(%arg3, %arg5)
              %13 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 8)>(%arg4, %arg6)
              %14 = loom.alloc(%c32, %c32) on @L1 : !loom.buffer_token
              %15 = loom.init_tensor %14(%c32, %c32) : !loom.buffer_token -> tensor<?x?xf32>
              %16 = linalg.fill ins(%cst : f32) outs(%15 : tensor<?x?xf32>) -> tensor<?x?xf32>
              %17 = affine.for %arg7 = 0 to 4 iter_args(%arg8 = %16) -> (tensor<?x?xf32>) {
                %21 = arith.muli %12, %c32 : index
                %22 = arith.muli %arg7, %c32 : index
                %23 = loom.view %arg0[%21, %22] [32, 32] [1, 1], reuse : [seq = false, spat = true, temp = true] : memref<128x128xf32> -> !loom.view
                %24 = loom.alloc(%c32, %c32) on @L1 : !loom.buffer_token
                %25 = loom.copy_to_tensor %23, %24 to @L1, interconnect : [], broadcast : [1, 1] : !loom.view, !loom.buffer_token -> tensor<?x?xf32>
                %26 = arith.muli %13, %c32 : index
                %27 = loom.view %arg1[%22, %26] [32, 32] [1, 1], reuse : [seq = false, spat = true, temp = true] : memref<128x256xf32> -> !loom.view
                %28 = loom.alloc(%c32, %c32) on @L1 : !loom.buffer_token
                %29 = loom.copy_to_tensor %27, %28 to @L1, interconnect : [], broadcast : [1, 1] : !loom.view, !loom.buffer_token -> tensor<?x?xf32>
                %30 = linalg.matmul ins(%25, %29 : tensor<?x?xf32>, tensor<?x?xf32>) outs(%16 : tensor<?x?xf32>) -> tensor<?x?xf32>
                %31 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%arg8, %30 : tensor<?x?xf32>, tensor<?x?xf32>) outs(%15 : tensor<?x?xf32>) {
                ^bb0(%in: f32, %in_0: f32, %out: f32):
                  %32 = arith.addf %in, %in_0 : f32
                  linalg.yield %32 : f32
                } -> tensor<?x?xf32>
                affine.yield %31 : tensor<?x?xf32>
              } {loom.iter_type = #loom.iter_type<sequential>}
              %18 = arith.muli %12, %c32 : index
              %19 = arith.muli %13, %c32 : index
              %20 = loom.view %arg2[%18, %19] [32, 32] [1, 1], reuse : [seq = false, spat = false, temp = false] : memref<128x256xf32> -> !loom.view
              loom.copy_from_tensor %17, %20 to @DRAM : tensor<?x?xf32>, !loom.view
            } {loom.iter_type = #loom.iter_type<temporal>}
          } {loom.iter_type = #loom.iter_type<temporal>}
        } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @y}
      } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @x}
      return
    }
    func.func @matmul__d0i0_d1i1__f01__d_d__BK64__BM32__BN32(%arg0: memref<128x128xf32>, %arg1: memref<128x256xf32>, %arg2: memref<128x256xf32>) {
      %cst = arith.constant 0.000000e+00 : f32
      %c32 = arith.constant 32 : index
      %c64 = arith.constant 64 : index
      affine.parallel (%arg3) = (0) to (8) {
        affine.parallel (%arg4) = (0) to (8) {
          affine.for %arg5 = 0 to 1 {
            affine.for %arg6 = 0 to 1 {
              %12 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 8)>(%arg3, %arg5)
              %13 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 8)>(%arg4, %arg6)
              %14 = loom.alloc(%c32, %c32) on @L1 : !loom.buffer_token
              %15 = loom.init_tensor %14(%c32, %c32) : !loom.buffer_token -> tensor<?x?xf32>
              %16 = linalg.fill ins(%cst : f32) outs(%15 : tensor<?x?xf32>) -> tensor<?x?xf32>
              %17 = affine.for %arg7 = 0 to 2 iter_args(%arg8 = %16) -> (tensor<?x?xf32>) {
                %21 = arith.muli %12, %c32 : index
                %22 = arith.muli %arg7, %c64 : index
                %23 = loom.view %arg0[%21, %22] [32, 64] [1, 1], reuse : [seq = false, spat = true, temp = true] : memref<128x128xf32> -> !loom.view
                %24 = loom.alloc(%c32, %c64) on @L1 : !loom.buffer_token
                %25 = loom.copy_to_tensor %23, %24 to @L1, interconnect : [], broadcast : [1, 1] : !loom.view, !loom.buffer_token -> tensor<?x?xf32>
                %26 = arith.muli %13, %c32 : index
                %27 = loom.view %arg1[%22, %26] [64, 32] [1, 1], reuse : [seq = false, spat = true, temp = true] : memref<128x256xf32> -> !loom.view
                %28 = loom.alloc(%c64, %c32) on @L1 : !loom.buffer_token
                %29 = loom.copy_to_tensor %27, %28 to @L1, interconnect : [], broadcast : [1, 1] : !loom.view, !loom.buffer_token -> tensor<?x?xf32>
                %30 = linalg.matmul ins(%25, %29 : tensor<?x?xf32>, tensor<?x?xf32>) outs(%16 : tensor<?x?xf32>) -> tensor<?x?xf32>
                %31 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%arg8, %30 : tensor<?x?xf32>, tensor<?x?xf32>) outs(%15 : tensor<?x?xf32>) {
                ^bb0(%in: f32, %in_0: f32, %out: f32):
                  %32 = arith.addf %in, %in_0 : f32
                  linalg.yield %32 : f32
                } -> tensor<?x?xf32>
                affine.yield %31 : tensor<?x?xf32>
              } {loom.iter_type = #loom.iter_type<sequential>}
              %18 = arith.muli %12, %c32 : index
              %19 = arith.muli %13, %c32 : index
              %20 = loom.view %arg2[%18, %19] [32, 32] [1, 1], reuse : [seq = false, spat = false, temp = false] : memref<128x256xf32> -> !loom.view
              loom.copy_from_tensor %17, %20 to @DRAM : tensor<?x?xf32>, !loom.view
            } {loom.iter_type = #loom.iter_type<temporal>}
          } {loom.iter_type = #loom.iter_type<temporal>}
        } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @y}
      } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @x}
      return
    }
  }
  module attributes {loom.pass_name = "Materialize"} {
    func.func @matmul__d0i0_d1i1__f01__d_v__BK32__BM32__BN32(%arg0: memref<128x128xf32>, %arg1: memref<128x256xf32>, %arg2: memref<128x256xf32>) {
      %cst = arith.constant 0.000000e+00 : f32
      %c32 = arith.constant 32 : index
      affine.parallel (%arg3) = (0) to (8) {
        affine.parallel (%arg4) = (0) to (8) {
          affine.for %arg5 = 0 to 1 {
            affine.for %arg6 = 0 to 1 {
              %12 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 8)>(%arg3, %arg5)
              %13 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 8)>(%arg4, %arg6)
              %14 = loom.alloc(%c32, %c32) on @L1 : !loom.buffer_token
              %15 = loom.init_tensor %14(%c32, %c32) : !loom.buffer_token -> tensor<?x?xf32>
              %16 = linalg.fill ins(%cst : f32) outs(%15 : tensor<?x?xf32>) -> tensor<?x?xf32>
              %17 = affine.for %arg7 = 0 to 4 iter_args(%arg8 = %16) -> (tensor<?x?xf32>) {
                %21 = arith.muli %12, %c32 : index
                %22 = arith.muli %arg7, %c32 : index
                %23 = loom.view %arg0[%21, %22] [32, 32] [1, 1], reuse : [seq = false, spat = true, temp = true] : memref<128x128xf32> -> !loom.view
                %24 = loom.alloc(%c32, %c32) on @L1 : !loom.buffer_token
                %25 = loom.copy_to_tensor %23, %24 to @L1, interconnect : [], broadcast : [1, 1] : !loom.view, !loom.buffer_token -> tensor<?x?xf32>
                %26 = arith.muli %13, %c32 : index
                %27 = loom.view %arg1[%22, %26] [32, 32] [1, 1], reuse : [seq = false, spat = true, temp = true] : memref<128x256xf32> -> !loom.view
                %28 = loom.alloc(%c32, %c32) on @L1 : !loom.buffer_token
                %29 = loom.copy_to_tensor %27, %28 to @L1, interconnect : [@vertical_links], broadcast : [8, 1] : !loom.view, !loom.buffer_token -> tensor<?x?xf32>
                %30 = linalg.matmul ins(%25, %29 : tensor<?x?xf32>, tensor<?x?xf32>) outs(%16 : tensor<?x?xf32>) -> tensor<?x?xf32>
                %31 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%arg8, %30 : tensor<?x?xf32>, tensor<?x?xf32>) outs(%15 : tensor<?x?xf32>) {
                ^bb0(%in: f32, %in_0: f32, %out: f32):
                  %32 = arith.addf %in, %in_0 : f32
                  linalg.yield %32 : f32
                } -> tensor<?x?xf32>
                affine.yield %31 : tensor<?x?xf32>
              } {loom.iter_type = #loom.iter_type<sequential>}
              %18 = arith.muli %12, %c32 : index
              %19 = arith.muli %13, %c32 : index
              %20 = loom.view %arg2[%18, %19] [32, 32] [1, 1], reuse : [seq = false, spat = false, temp = false] : memref<128x256xf32> -> !loom.view
              loom.copy_from_tensor %17, %20 to @DRAM : tensor<?x?xf32>, !loom.view
            } {loom.iter_type = #loom.iter_type<temporal>}
          } {loom.iter_type = #loom.iter_type<temporal>}
        } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @y}
      } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @x}
      return
    }
    func.func @matmul__d0i0_d1i1__f01__d_v__BK64__BM32__BN32(%arg0: memref<128x128xf32>, %arg1: memref<128x256xf32>, %arg2: memref<128x256xf32>) {
      %cst = arith.constant 0.000000e+00 : f32
      %c32 = arith.constant 32 : index
      %c64 = arith.constant 64 : index
      affine.parallel (%arg3) = (0) to (8) {
        affine.parallel (%arg4) = (0) to (8) {
          affine.for %arg5 = 0 to 1 {
            affine.for %arg6 = 0 to 1 {
              %12 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 8)>(%arg3, %arg5)
              %13 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 8)>(%arg4, %arg6)
              %14 = loom.alloc(%c32, %c32) on @L1 : !loom.buffer_token
              %15 = loom.init_tensor %14(%c32, %c32) : !loom.buffer_token -> tensor<?x?xf32>
              %16 = linalg.fill ins(%cst : f32) outs(%15 : tensor<?x?xf32>) -> tensor<?x?xf32>
              %17 = affine.for %arg7 = 0 to 2 iter_args(%arg8 = %16) -> (tensor<?x?xf32>) {
                %21 = arith.muli %12, %c32 : index
                %22 = arith.muli %arg7, %c64 : index
                %23 = loom.view %arg0[%21, %22] [32, 64] [1, 1], reuse : [seq = false, spat = true, temp = true] : memref<128x128xf32> -> !loom.view
                %24 = loom.alloc(%c32, %c64) on @L1 : !loom.buffer_token
                %25 = loom.copy_to_tensor %23, %24 to @L1, interconnect : [], broadcast : [1, 1] : !loom.view, !loom.buffer_token -> tensor<?x?xf32>
                %26 = arith.muli %13, %c32 : index
                %27 = loom.view %arg1[%22, %26] [64, 32] [1, 1], reuse : [seq = false, spat = true, temp = true] : memref<128x256xf32> -> !loom.view
                %28 = loom.alloc(%c64, %c32) on @L1 : !loom.buffer_token
                %29 = loom.copy_to_tensor %27, %28 to @L1, interconnect : [@vertical_links], broadcast : [8, 1] : !loom.view, !loom.buffer_token -> tensor<?x?xf32>
                %30 = linalg.matmul ins(%25, %29 : tensor<?x?xf32>, tensor<?x?xf32>) outs(%16 : tensor<?x?xf32>) -> tensor<?x?xf32>
                %31 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%arg8, %30 : tensor<?x?xf32>, tensor<?x?xf32>) outs(%15 : tensor<?x?xf32>) {
                ^bb0(%in: f32, %in_0: f32, %out: f32):
                  %32 = arith.addf %in, %in_0 : f32
                  linalg.yield %32 : f32
                } -> tensor<?x?xf32>
                affine.yield %31 : tensor<?x?xf32>
              } {loom.iter_type = #loom.iter_type<sequential>}
              %18 = arith.muli %12, %c32 : index
              %19 = arith.muli %13, %c32 : index
              %20 = loom.view %arg2[%18, %19] [32, 32] [1, 1], reuse : [seq = false, spat = false, temp = false] : memref<128x256xf32> -> !loom.view
              loom.copy_from_tensor %17, %20 to @DRAM : tensor<?x?xf32>, !loom.view
            } {loom.iter_type = #loom.iter_type<temporal>}
          } {loom.iter_type = #loom.iter_type<temporal>}
        } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @y}
      } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @x}
      return
    }
  }
  module attributes {loom.pass_name = "Materialize"} {
    func.func @matmul__d0i0_d1i1__f01__h_d__BK32__BM32__BN32(%arg0: memref<128x128xf32>, %arg1: memref<128x256xf32>, %arg2: memref<128x256xf32>) {
      %cst = arith.constant 0.000000e+00 : f32
      %c32 = arith.constant 32 : index
      affine.parallel (%arg3) = (0) to (8) {
        affine.parallel (%arg4) = (0) to (8) {
          affine.for %arg5 = 0 to 1 {
            affine.for %arg6 = 0 to 1 {
              %12 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 8)>(%arg3, %arg5)
              %13 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 8)>(%arg4, %arg6)
              %14 = loom.alloc(%c32, %c32) on @L1 : !loom.buffer_token
              %15 = loom.init_tensor %14(%c32, %c32) : !loom.buffer_token -> tensor<?x?xf32>
              %16 = linalg.fill ins(%cst : f32) outs(%15 : tensor<?x?xf32>) -> tensor<?x?xf32>
              %17 = affine.for %arg7 = 0 to 4 iter_args(%arg8 = %16) -> (tensor<?x?xf32>) {
                %21 = arith.muli %12, %c32 : index
                %22 = arith.muli %arg7, %c32 : index
                %23 = loom.view %arg0[%21, %22] [32, 32] [1, 1], reuse : [seq = false, spat = true, temp = true] : memref<128x128xf32> -> !loom.view
                %24 = loom.alloc(%c32, %c32) on @L1 : !loom.buffer_token
                %25 = loom.copy_to_tensor %23, %24 to @L1, interconnect : [@horizontal_links], broadcast : [1, 8] : !loom.view, !loom.buffer_token -> tensor<?x?xf32>
                %26 = arith.muli %13, %c32 : index
                %27 = loom.view %arg1[%22, %26] [32, 32] [1, 1], reuse : [seq = false, spat = true, temp = true] : memref<128x256xf32> -> !loom.view
                %28 = loom.alloc(%c32, %c32) on @L1 : !loom.buffer_token
                %29 = loom.copy_to_tensor %27, %28 to @L1, interconnect : [], broadcast : [1, 1] : !loom.view, !loom.buffer_token -> tensor<?x?xf32>
                %30 = linalg.matmul ins(%25, %29 : tensor<?x?xf32>, tensor<?x?xf32>) outs(%16 : tensor<?x?xf32>) -> tensor<?x?xf32>
                %31 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%arg8, %30 : tensor<?x?xf32>, tensor<?x?xf32>) outs(%15 : tensor<?x?xf32>) {
                ^bb0(%in: f32, %in_0: f32, %out: f32):
                  %32 = arith.addf %in, %in_0 : f32
                  linalg.yield %32 : f32
                } -> tensor<?x?xf32>
                affine.yield %31 : tensor<?x?xf32>
              } {loom.iter_type = #loom.iter_type<sequential>}
              %18 = arith.muli %12, %c32 : index
              %19 = arith.muli %13, %c32 : index
              %20 = loom.view %arg2[%18, %19] [32, 32] [1, 1], reuse : [seq = false, spat = false, temp = false] : memref<128x256xf32> -> !loom.view
              loom.copy_from_tensor %17, %20 to @DRAM : tensor<?x?xf32>, !loom.view
            } {loom.iter_type = #loom.iter_type<temporal>}
          } {loom.iter_type = #loom.iter_type<temporal>}
        } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @y}
      } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @x}
      return
    }
    func.func @matmul__d0i0_d1i1__f01__h_d__BK64__BM32__BN32(%arg0: memref<128x128xf32>, %arg1: memref<128x256xf32>, %arg2: memref<128x256xf32>) {
      %cst = arith.constant 0.000000e+00 : f32
      %c32 = arith.constant 32 : index
      %c64 = arith.constant 64 : index
      affine.parallel (%arg3) = (0) to (8) {
        affine.parallel (%arg4) = (0) to (8) {
          affine.for %arg5 = 0 to 1 {
            affine.for %arg6 = 0 to 1 {
              %12 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 8)>(%arg3, %arg5)
              %13 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 8)>(%arg4, %arg6)
              %14 = loom.alloc(%c32, %c32) on @L1 : !loom.buffer_token
              %15 = loom.init_tensor %14(%c32, %c32) : !loom.buffer_token -> tensor<?x?xf32>
              %16 = linalg.fill ins(%cst : f32) outs(%15 : tensor<?x?xf32>) -> tensor<?x?xf32>
              %17 = affine.for %arg7 = 0 to 2 iter_args(%arg8 = %16) -> (tensor<?x?xf32>) {
                %21 = arith.muli %12, %c32 : index
                %22 = arith.muli %arg7, %c64 : index
                %23 = loom.view %arg0[%21, %22] [32, 64] [1, 1], reuse : [seq = false, spat = true, temp = true] : memref<128x128xf32> -> !loom.view
                %24 = loom.alloc(%c32, %c64) on @L1 : !loom.buffer_token
                %25 = loom.copy_to_tensor %23, %24 to @L1, interconnect : [@horizontal_links], broadcast : [1, 8] : !loom.view, !loom.buffer_token -> tensor<?x?xf32>
                %26 = arith.muli %13, %c32 : index
                %27 = loom.view %arg1[%22, %26] [64, 32] [1, 1], reuse : [seq = false, spat = true, temp = true] : memref<128x256xf32> -> !loom.view
                %28 = loom.alloc(%c64, %c32) on @L1 : !loom.buffer_token
                %29 = loom.copy_to_tensor %27, %28 to @L1, interconnect : [], broadcast : [1, 1] : !loom.view, !loom.buffer_token -> tensor<?x?xf32>
                %30 = linalg.matmul ins(%25, %29 : tensor<?x?xf32>, tensor<?x?xf32>) outs(%16 : tensor<?x?xf32>) -> tensor<?x?xf32>
                %31 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%arg8, %30 : tensor<?x?xf32>, tensor<?x?xf32>) outs(%15 : tensor<?x?xf32>) {
                ^bb0(%in: f32, %in_0: f32, %out: f32):
                  %32 = arith.addf %in, %in_0 : f32
                  linalg.yield %32 : f32
                } -> tensor<?x?xf32>
                affine.yield %31 : tensor<?x?xf32>
              } {loom.iter_type = #loom.iter_type<sequential>}
              %18 = arith.muli %12, %c32 : index
              %19 = arith.muli %13, %c32 : index
              %20 = loom.view %arg2[%18, %19] [32, 32] [1, 1], reuse : [seq = false, spat = false, temp = false] : memref<128x256xf32> -> !loom.view
              loom.copy_from_tensor %17, %20 to @DRAM : tensor<?x?xf32>, !loom.view
            } {loom.iter_type = #loom.iter_type<temporal>}
          } {loom.iter_type = #loom.iter_type<temporal>}
        } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @y}
      } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @x}
      return
    }
  }
  module attributes {loom.pass_name = "Materialize"} {
    func.func @matmul__d0i0_d1i1__f01__h_v__BK32__BM32__BN32(%arg0: memref<128x128xf32>, %arg1: memref<128x256xf32>, %arg2: memref<128x256xf32>) {
      %cst = arith.constant 0.000000e+00 : f32
      %c32 = arith.constant 32 : index
      affine.parallel (%arg3) = (0) to (8) {
        affine.parallel (%arg4) = (0) to (8) {
          affine.for %arg5 = 0 to 1 {
            affine.for %arg6 = 0 to 1 {
              %12 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 8)>(%arg3, %arg5)
              %13 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 8)>(%arg4, %arg6)
              %14 = loom.alloc(%c32, %c32) on @L1 : !loom.buffer_token
              %15 = loom.init_tensor %14(%c32, %c32) : !loom.buffer_token -> tensor<?x?xf32>
              %16 = linalg.fill ins(%cst : f32) outs(%15 : tensor<?x?xf32>) -> tensor<?x?xf32>
              %17 = affine.for %arg7 = 0 to 4 iter_args(%arg8 = %16) -> (tensor<?x?xf32>) {
                %21 = arith.muli %12, %c32 : index
                %22 = arith.muli %arg7, %c32 : index
                %23 = loom.view %arg0[%21, %22] [32, 32] [1, 1], reuse : [seq = false, spat = true, temp = true] : memref<128x128xf32> -> !loom.view
                %24 = loom.alloc(%c32, %c32) on @L1 : !loom.buffer_token
                %25 = loom.copy_to_tensor %23, %24 to @L1, interconnect : [@horizontal_links], broadcast : [1, 8] : !loom.view, !loom.buffer_token -> tensor<?x?xf32>
                %26 = arith.muli %13, %c32 : index
                %27 = loom.view %arg1[%22, %26] [32, 32] [1, 1], reuse : [seq = false, spat = true, temp = true] : memref<128x256xf32> -> !loom.view
                %28 = loom.alloc(%c32, %c32) on @L1 : !loom.buffer_token
                %29 = loom.copy_to_tensor %27, %28 to @L1, interconnect : [@vertical_links], broadcast : [8, 1] : !loom.view, !loom.buffer_token -> tensor<?x?xf32>
                %30 = linalg.matmul ins(%25, %29 : tensor<?x?xf32>, tensor<?x?xf32>) outs(%16 : tensor<?x?xf32>) -> tensor<?x?xf32>
                %31 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%arg8, %30 : tensor<?x?xf32>, tensor<?x?xf32>) outs(%15 : tensor<?x?xf32>) {
                ^bb0(%in: f32, %in_0: f32, %out: f32):
                  %32 = arith.addf %in, %in_0 : f32
                  linalg.yield %32 : f32
                } -> tensor<?x?xf32>
                affine.yield %31 : tensor<?x?xf32>
              } {loom.iter_type = #loom.iter_type<sequential>}
              %18 = arith.muli %12, %c32 : index
              %19 = arith.muli %13, %c32 : index
              %20 = loom.view %arg2[%18, %19] [32, 32] [1, 1], reuse : [seq = false, spat = false, temp = false] : memref<128x256xf32> -> !loom.view
              loom.copy_from_tensor %17, %20 to @DRAM : tensor<?x?xf32>, !loom.view
            } {loom.iter_type = #loom.iter_type<temporal>}
          } {loom.iter_type = #loom.iter_type<temporal>}
        } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @y}
      } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @x}
      return
    }
    func.func @matmul__d0i0_d1i1__f01__h_v__BK64__BM32__BN32(%arg0: memref<128x128xf32>, %arg1: memref<128x256xf32>, %arg2: memref<128x256xf32>) {
      %cst = arith.constant 0.000000e+00 : f32
      %c32 = arith.constant 32 : index
      %c64 = arith.constant 64 : index
      affine.parallel (%arg3) = (0) to (8) {
        affine.parallel (%arg4) = (0) to (8) {
          affine.for %arg5 = 0 to 1 {
            affine.for %arg6 = 0 to 1 {
              %12 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 8)>(%arg3, %arg5)
              %13 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 8)>(%arg4, %arg6)
              %14 = loom.alloc(%c32, %c32) on @L1 : !loom.buffer_token
              %15 = loom.init_tensor %14(%c32, %c32) : !loom.buffer_token -> tensor<?x?xf32>
              %16 = linalg.fill ins(%cst : f32) outs(%15 : tensor<?x?xf32>) -> tensor<?x?xf32>
              %17 = affine.for %arg7 = 0 to 2 iter_args(%arg8 = %16) -> (tensor<?x?xf32>) {
                %21 = arith.muli %12, %c32 : index
                %22 = arith.muli %arg7, %c64 : index
                %23 = loom.view %arg0[%21, %22] [32, 64] [1, 1], reuse : [seq = false, spat = true, temp = true] : memref<128x128xf32> -> !loom.view
                %24 = loom.alloc(%c32, %c64) on @L1 : !loom.buffer_token
                %25 = loom.copy_to_tensor %23, %24 to @L1, interconnect : [@horizontal_links], broadcast : [1, 8] : !loom.view, !loom.buffer_token -> tensor<?x?xf32>
                %26 = arith.muli %13, %c32 : index
                %27 = loom.view %arg1[%22, %26] [64, 32] [1, 1], reuse : [seq = false, spat = true, temp = true] : memref<128x256xf32> -> !loom.view
                %28 = loom.alloc(%c64, %c32) on @L1 : !loom.buffer_token
                %29 = loom.copy_to_tensor %27, %28 to @L1, interconnect : [@vertical_links], broadcast : [8, 1] : !loom.view, !loom.buffer_token -> tensor<?x?xf32>
                %30 = linalg.matmul ins(%25, %29 : tensor<?x?xf32>, tensor<?x?xf32>) outs(%16 : tensor<?x?xf32>) -> tensor<?x?xf32>
                %31 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%arg8, %30 : tensor<?x?xf32>, tensor<?x?xf32>) outs(%15 : tensor<?x?xf32>) {
                ^bb0(%in: f32, %in_0: f32, %out: f32):
                  %32 = arith.addf %in, %in_0 : f32
                  linalg.yield %32 : f32
                } -> tensor<?x?xf32>
                affine.yield %31 : tensor<?x?xf32>
              } {loom.iter_type = #loom.iter_type<sequential>}
              %18 = arith.muli %12, %c32 : index
              %19 = arith.muli %13, %c32 : index
              %20 = loom.view %arg2[%18, %19] [32, 32] [1, 1], reuse : [seq = false, spat = false, temp = false] : memref<128x256xf32> -> !loom.view
              loom.copy_from_tensor %17, %20 to @DRAM : tensor<?x?xf32>, !loom.view
            } {loom.iter_type = #loom.iter_type<temporal>}
          } {loom.iter_type = #loom.iter_type<temporal>}
        } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @y}
      } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @x}
      return
    }
  }
  module attributes {loom.pass_name = "Materialize"} {
    func.func @matmul__d0i0_d1i1__f10__d_d__BK32__BM32__BN32(%arg0: memref<128x128xf32>, %arg1: memref<128x256xf32>, %arg2: memref<128x256xf32>) {
      %cst = arith.constant 0.000000e+00 : f32
      %c32 = arith.constant 32 : index
      affine.parallel (%arg3) = (0) to (8) {
        affine.parallel (%arg4) = (0) to (8) {
          affine.for %arg5 = 0 to 1 {
            affine.for %arg6 = 0 to 1 {
              %12 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 8)>(%arg3, %arg6)
              %13 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 8)>(%arg4, %arg5)
              %14 = loom.alloc(%c32, %c32) on @L1 : !loom.buffer_token
              %15 = loom.init_tensor %14(%c32, %c32) : !loom.buffer_token -> tensor<?x?xf32>
              %16 = linalg.fill ins(%cst : f32) outs(%15 : tensor<?x?xf32>) -> tensor<?x?xf32>
              %17 = affine.for %arg7 = 0 to 4 iter_args(%arg8 = %16) -> (tensor<?x?xf32>) {
                %21 = arith.muli %12, %c32 : index
                %22 = arith.muli %arg7, %c32 : index
                %23 = loom.view %arg0[%21, %22] [32, 32] [1, 1], reuse : [seq = false, spat = true, temp = true] : memref<128x128xf32> -> !loom.view
                %24 = loom.alloc(%c32, %c32) on @L1 : !loom.buffer_token
                %25 = loom.copy_to_tensor %23, %24 to @L1, interconnect : [], broadcast : [1, 1] : !loom.view, !loom.buffer_token -> tensor<?x?xf32>
                %26 = arith.muli %13, %c32 : index
                %27 = loom.view %arg1[%22, %26] [32, 32] [1, 1], reuse : [seq = false, spat = true, temp = true] : memref<128x256xf32> -> !loom.view
                %28 = loom.alloc(%c32, %c32) on @L1 : !loom.buffer_token
                %29 = loom.copy_to_tensor %27, %28 to @L1, interconnect : [], broadcast : [1, 1] : !loom.view, !loom.buffer_token -> tensor<?x?xf32>
                %30 = linalg.matmul ins(%25, %29 : tensor<?x?xf32>, tensor<?x?xf32>) outs(%16 : tensor<?x?xf32>) -> tensor<?x?xf32>
                %31 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%arg8, %30 : tensor<?x?xf32>, tensor<?x?xf32>) outs(%15 : tensor<?x?xf32>) {
                ^bb0(%in: f32, %in_0: f32, %out: f32):
                  %32 = arith.addf %in, %in_0 : f32
                  linalg.yield %32 : f32
                } -> tensor<?x?xf32>
                affine.yield %31 : tensor<?x?xf32>
              } {loom.iter_type = #loom.iter_type<sequential>}
              %18 = arith.muli %12, %c32 : index
              %19 = arith.muli %13, %c32 : index
              %20 = loom.view %arg2[%18, %19] [32, 32] [1, 1], reuse : [seq = false, spat = false, temp = false] : memref<128x256xf32> -> !loom.view
              loom.copy_from_tensor %17, %20 to @DRAM : tensor<?x?xf32>, !loom.view
            } {loom.iter_type = #loom.iter_type<temporal>}
          } {loom.iter_type = #loom.iter_type<temporal>}
        } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @y}
      } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @x}
      return
    }
    func.func @matmul__d0i0_d1i1__f10__d_d__BK64__BM32__BN32(%arg0: memref<128x128xf32>, %arg1: memref<128x256xf32>, %arg2: memref<128x256xf32>) {
      %cst = arith.constant 0.000000e+00 : f32
      %c32 = arith.constant 32 : index
      %c64 = arith.constant 64 : index
      affine.parallel (%arg3) = (0) to (8) {
        affine.parallel (%arg4) = (0) to (8) {
          affine.for %arg5 = 0 to 1 {
            affine.for %arg6 = 0 to 1 {
              %12 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 8)>(%arg3, %arg6)
              %13 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 8)>(%arg4, %arg5)
              %14 = loom.alloc(%c32, %c32) on @L1 : !loom.buffer_token
              %15 = loom.init_tensor %14(%c32, %c32) : !loom.buffer_token -> tensor<?x?xf32>
              %16 = linalg.fill ins(%cst : f32) outs(%15 : tensor<?x?xf32>) -> tensor<?x?xf32>
              %17 = affine.for %arg7 = 0 to 2 iter_args(%arg8 = %16) -> (tensor<?x?xf32>) {
                %21 = arith.muli %12, %c32 : index
                %22 = arith.muli %arg7, %c64 : index
                %23 = loom.view %arg0[%21, %22] [32, 64] [1, 1], reuse : [seq = false, spat = true, temp = true] : memref<128x128xf32> -> !loom.view
                %24 = loom.alloc(%c32, %c64) on @L1 : !loom.buffer_token
                %25 = loom.copy_to_tensor %23, %24 to @L1, interconnect : [], broadcast : [1, 1] : !loom.view, !loom.buffer_token -> tensor<?x?xf32>
                %26 = arith.muli %13, %c32 : index
                %27 = loom.view %arg1[%22, %26] [64, 32] [1, 1], reuse : [seq = false, spat = true, temp = true] : memref<128x256xf32> -> !loom.view
                %28 = loom.alloc(%c64, %c32) on @L1 : !loom.buffer_token
                %29 = loom.copy_to_tensor %27, %28 to @L1, interconnect : [], broadcast : [1, 1] : !loom.view, !loom.buffer_token -> tensor<?x?xf32>
                %30 = linalg.matmul ins(%25, %29 : tensor<?x?xf32>, tensor<?x?xf32>) outs(%16 : tensor<?x?xf32>) -> tensor<?x?xf32>
                %31 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%arg8, %30 : tensor<?x?xf32>, tensor<?x?xf32>) outs(%15 : tensor<?x?xf32>) {
                ^bb0(%in: f32, %in_0: f32, %out: f32):
                  %32 = arith.addf %in, %in_0 : f32
                  linalg.yield %32 : f32
                } -> tensor<?x?xf32>
                affine.yield %31 : tensor<?x?xf32>
              } {loom.iter_type = #loom.iter_type<sequential>}
              %18 = arith.muli %12, %c32 : index
              %19 = arith.muli %13, %c32 : index
              %20 = loom.view %arg2[%18, %19] [32, 32] [1, 1], reuse : [seq = false, spat = false, temp = false] : memref<128x256xf32> -> !loom.view
              loom.copy_from_tensor %17, %20 to @DRAM : tensor<?x?xf32>, !loom.view
            } {loom.iter_type = #loom.iter_type<temporal>}
          } {loom.iter_type = #loom.iter_type<temporal>}
        } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @y}
      } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @x}
      return
    }
  }
  module attributes {loom.pass_name = "Materialize"} {
    func.func @matmul__d0i0_d1i1__f10__d_v__BK32__BM32__BN32(%arg0: memref<128x128xf32>, %arg1: memref<128x256xf32>, %arg2: memref<128x256xf32>) {
      %cst = arith.constant 0.000000e+00 : f32
      %c32 = arith.constant 32 : index
      affine.parallel (%arg3) = (0) to (8) {
        affine.parallel (%arg4) = (0) to (8) {
          affine.for %arg5 = 0 to 1 {
            affine.for %arg6 = 0 to 1 {
              %12 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 8)>(%arg3, %arg6)
              %13 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 8)>(%arg4, %arg5)
              %14 = loom.alloc(%c32, %c32) on @L1 : !loom.buffer_token
              %15 = loom.init_tensor %14(%c32, %c32) : !loom.buffer_token -> tensor<?x?xf32>
              %16 = linalg.fill ins(%cst : f32) outs(%15 : tensor<?x?xf32>) -> tensor<?x?xf32>
              %17 = affine.for %arg7 = 0 to 4 iter_args(%arg8 = %16) -> (tensor<?x?xf32>) {
                %21 = arith.muli %12, %c32 : index
                %22 = arith.muli %arg7, %c32 : index
                %23 = loom.view %arg0[%21, %22] [32, 32] [1, 1], reuse : [seq = false, spat = true, temp = true] : memref<128x128xf32> -> !loom.view
                %24 = loom.alloc(%c32, %c32) on @L1 : !loom.buffer_token
                %25 = loom.copy_to_tensor %23, %24 to @L1, interconnect : [], broadcast : [1, 1] : !loom.view, !loom.buffer_token -> tensor<?x?xf32>
                %26 = arith.muli %13, %c32 : index
                %27 = loom.view %arg1[%22, %26] [32, 32] [1, 1], reuse : [seq = false, spat = true, temp = true] : memref<128x256xf32> -> !loom.view
                %28 = loom.alloc(%c32, %c32) on @L1 : !loom.buffer_token
                %29 = loom.copy_to_tensor %27, %28 to @L1, interconnect : [@vertical_links], broadcast : [8, 1] : !loom.view, !loom.buffer_token -> tensor<?x?xf32>
                %30 = linalg.matmul ins(%25, %29 : tensor<?x?xf32>, tensor<?x?xf32>) outs(%16 : tensor<?x?xf32>) -> tensor<?x?xf32>
                %31 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%arg8, %30 : tensor<?x?xf32>, tensor<?x?xf32>) outs(%15 : tensor<?x?xf32>) {
                ^bb0(%in: f32, %in_0: f32, %out: f32):
                  %32 = arith.addf %in, %in_0 : f32
                  linalg.yield %32 : f32
                } -> tensor<?x?xf32>
                affine.yield %31 : tensor<?x?xf32>
              } {loom.iter_type = #loom.iter_type<sequential>}
              %18 = arith.muli %12, %c32 : index
              %19 = arith.muli %13, %c32 : index
              %20 = loom.view %arg2[%18, %19] [32, 32] [1, 1], reuse : [seq = false, spat = false, temp = false] : memref<128x256xf32> -> !loom.view
              loom.copy_from_tensor %17, %20 to @DRAM : tensor<?x?xf32>, !loom.view
            } {loom.iter_type = #loom.iter_type<temporal>}
          } {loom.iter_type = #loom.iter_type<temporal>}
        } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @y}
      } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @x}
      return
    }
    func.func @matmul__d0i0_d1i1__f10__d_v__BK64__BM32__BN32(%arg0: memref<128x128xf32>, %arg1: memref<128x256xf32>, %arg2: memref<128x256xf32>) {
      %cst = arith.constant 0.000000e+00 : f32
      %c32 = arith.constant 32 : index
      %c64 = arith.constant 64 : index
      affine.parallel (%arg3) = (0) to (8) {
        affine.parallel (%arg4) = (0) to (8) {
          affine.for %arg5 = 0 to 1 {
            affine.for %arg6 = 0 to 1 {
              %12 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 8)>(%arg3, %arg6)
              %13 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 8)>(%arg4, %arg5)
              %14 = loom.alloc(%c32, %c32) on @L1 : !loom.buffer_token
              %15 = loom.init_tensor %14(%c32, %c32) : !loom.buffer_token -> tensor<?x?xf32>
              %16 = linalg.fill ins(%cst : f32) outs(%15 : tensor<?x?xf32>) -> tensor<?x?xf32>
              %17 = affine.for %arg7 = 0 to 2 iter_args(%arg8 = %16) -> (tensor<?x?xf32>) {
                %21 = arith.muli %12, %c32 : index
                %22 = arith.muli %arg7, %c64 : index
                %23 = loom.view %arg0[%21, %22] [32, 64] [1, 1], reuse : [seq = false, spat = true, temp = true] : memref<128x128xf32> -> !loom.view
                %24 = loom.alloc(%c32, %c64) on @L1 : !loom.buffer_token
                %25 = loom.copy_to_tensor %23, %24 to @L1, interconnect : [], broadcast : [1, 1] : !loom.view, !loom.buffer_token -> tensor<?x?xf32>
                %26 = arith.muli %13, %c32 : index
                %27 = loom.view %arg1[%22, %26] [64, 32] [1, 1], reuse : [seq = false, spat = true, temp = true] : memref<128x256xf32> -> !loom.view
                %28 = loom.alloc(%c64, %c32) on @L1 : !loom.buffer_token
                %29 = loom.copy_to_tensor %27, %28 to @L1, interconnect : [@vertical_links], broadcast : [8, 1] : !loom.view, !loom.buffer_token -> tensor<?x?xf32>
                %30 = linalg.matmul ins(%25, %29 : tensor<?x?xf32>, tensor<?x?xf32>) outs(%16 : tensor<?x?xf32>) -> tensor<?x?xf32>
                %31 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%arg8, %30 : tensor<?x?xf32>, tensor<?x?xf32>) outs(%15 : tensor<?x?xf32>) {
                ^bb0(%in: f32, %in_0: f32, %out: f32):
                  %32 = arith.addf %in, %in_0 : f32
                  linalg.yield %32 : f32
                } -> tensor<?x?xf32>
                affine.yield %31 : tensor<?x?xf32>
              } {loom.iter_type = #loom.iter_type<sequential>}
              %18 = arith.muli %12, %c32 : index
              %19 = arith.muli %13, %c32 : index
              %20 = loom.view %arg2[%18, %19] [32, 32] [1, 1], reuse : [seq = false, spat = false, temp = false] : memref<128x256xf32> -> !loom.view
              loom.copy_from_tensor %17, %20 to @DRAM : tensor<?x?xf32>, !loom.view
            } {loom.iter_type = #loom.iter_type<temporal>}
          } {loom.iter_type = #loom.iter_type<temporal>}
        } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @y}
      } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @x}
      return
    }
  }
  module attributes {loom.pass_name = "Materialize"} {
    func.func @matmul__d0i0_d1i1__f10__h_d__BK32__BM32__BN32(%arg0: memref<128x128xf32>, %arg1: memref<128x256xf32>, %arg2: memref<128x256xf32>) {
      %cst = arith.constant 0.000000e+00 : f32
      %c32 = arith.constant 32 : index
      affine.parallel (%arg3) = (0) to (8) {
        affine.parallel (%arg4) = (0) to (8) {
          affine.for %arg5 = 0 to 1 {
            affine.for %arg6 = 0 to 1 {
              %12 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 8)>(%arg3, %arg6)
              %13 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 8)>(%arg4, %arg5)
              %14 = loom.alloc(%c32, %c32) on @L1 : !loom.buffer_token
              %15 = loom.init_tensor %14(%c32, %c32) : !loom.buffer_token -> tensor<?x?xf32>
              %16 = linalg.fill ins(%cst : f32) outs(%15 : tensor<?x?xf32>) -> tensor<?x?xf32>
              %17 = affine.for %arg7 = 0 to 4 iter_args(%arg8 = %16) -> (tensor<?x?xf32>) {
                %21 = arith.muli %12, %c32 : index
                %22 = arith.muli %arg7, %c32 : index
                %23 = loom.view %arg0[%21, %22] [32, 32] [1, 1], reuse : [seq = false, spat = true, temp = true] : memref<128x128xf32> -> !loom.view
                %24 = loom.alloc(%c32, %c32) on @L1 : !loom.buffer_token
                %25 = loom.copy_to_tensor %23, %24 to @L1, interconnect : [@horizontal_links], broadcast : [1, 8] : !loom.view, !loom.buffer_token -> tensor<?x?xf32>
                %26 = arith.muli %13, %c32 : index
                %27 = loom.view %arg1[%22, %26] [32, 32] [1, 1], reuse : [seq = false, spat = true, temp = true] : memref<128x256xf32> -> !loom.view
                %28 = loom.alloc(%c32, %c32) on @L1 : !loom.buffer_token
                %29 = loom.copy_to_tensor %27, %28 to @L1, interconnect : [], broadcast : [1, 1] : !loom.view, !loom.buffer_token -> tensor<?x?xf32>
                %30 = linalg.matmul ins(%25, %29 : tensor<?x?xf32>, tensor<?x?xf32>) outs(%16 : tensor<?x?xf32>) -> tensor<?x?xf32>
                %31 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%arg8, %30 : tensor<?x?xf32>, tensor<?x?xf32>) outs(%15 : tensor<?x?xf32>) {
                ^bb0(%in: f32, %in_0: f32, %out: f32):
                  %32 = arith.addf %in, %in_0 : f32
                  linalg.yield %32 : f32
                } -> tensor<?x?xf32>
                affine.yield %31 : tensor<?x?xf32>
              } {loom.iter_type = #loom.iter_type<sequential>}
              %18 = arith.muli %12, %c32 : index
              %19 = arith.muli %13, %c32 : index
              %20 = loom.view %arg2[%18, %19] [32, 32] [1, 1], reuse : [seq = false, spat = false, temp = false] : memref<128x256xf32> -> !loom.view
              loom.copy_from_tensor %17, %20 to @DRAM : tensor<?x?xf32>, !loom.view
            } {loom.iter_type = #loom.iter_type<temporal>}
          } {loom.iter_type = #loom.iter_type<temporal>}
        } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @y}
      } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @x}
      return
    }
    func.func @matmul__d0i0_d1i1__f10__h_d__BK64__BM32__BN32(%arg0: memref<128x128xf32>, %arg1: memref<128x256xf32>, %arg2: memref<128x256xf32>) {
      %cst = arith.constant 0.000000e+00 : f32
      %c32 = arith.constant 32 : index
      %c64 = arith.constant 64 : index
      affine.parallel (%arg3) = (0) to (8) {
        affine.parallel (%arg4) = (0) to (8) {
          affine.for %arg5 = 0 to 1 {
            affine.for %arg6 = 0 to 1 {
              %12 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 8)>(%arg3, %arg6)
              %13 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 8)>(%arg4, %arg5)
              %14 = loom.alloc(%c32, %c32) on @L1 : !loom.buffer_token
              %15 = loom.init_tensor %14(%c32, %c32) : !loom.buffer_token -> tensor<?x?xf32>
              %16 = linalg.fill ins(%cst : f32) outs(%15 : tensor<?x?xf32>) -> tensor<?x?xf32>
              %17 = affine.for %arg7 = 0 to 2 iter_args(%arg8 = %16) -> (tensor<?x?xf32>) {
                %21 = arith.muli %12, %c32 : index
                %22 = arith.muli %arg7, %c64 : index
                %23 = loom.view %arg0[%21, %22] [32, 64] [1, 1], reuse : [seq = false, spat = true, temp = true] : memref<128x128xf32> -> !loom.view
                %24 = loom.alloc(%c32, %c64) on @L1 : !loom.buffer_token
                %25 = loom.copy_to_tensor %23, %24 to @L1, interconnect : [@horizontal_links], broadcast : [1, 8] : !loom.view, !loom.buffer_token -> tensor<?x?xf32>
                %26 = arith.muli %13, %c32 : index
                %27 = loom.view %arg1[%22, %26] [64, 32] [1, 1], reuse : [seq = false, spat = true, temp = true] : memref<128x256xf32> -> !loom.view
                %28 = loom.alloc(%c64, %c32) on @L1 : !loom.buffer_token
                %29 = loom.copy_to_tensor %27, %28 to @L1, interconnect : [], broadcast : [1, 1] : !loom.view, !loom.buffer_token -> tensor<?x?xf32>
                %30 = linalg.matmul ins(%25, %29 : tensor<?x?xf32>, tensor<?x?xf32>) outs(%16 : tensor<?x?xf32>) -> tensor<?x?xf32>
                %31 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%arg8, %30 : tensor<?x?xf32>, tensor<?x?xf32>) outs(%15 : tensor<?x?xf32>) {
                ^bb0(%in: f32, %in_0: f32, %out: f32):
                  %32 = arith.addf %in, %in_0 : f32
                  linalg.yield %32 : f32
                } -> tensor<?x?xf32>
                affine.yield %31 : tensor<?x?xf32>
              } {loom.iter_type = #loom.iter_type<sequential>}
              %18 = arith.muli %12, %c32 : index
              %19 = arith.muli %13, %c32 : index
              %20 = loom.view %arg2[%18, %19] [32, 32] [1, 1], reuse : [seq = false, spat = false, temp = false] : memref<128x256xf32> -> !loom.view
              loom.copy_from_tensor %17, %20 to @DRAM : tensor<?x?xf32>, !loom.view
            } {loom.iter_type = #loom.iter_type<temporal>}
          } {loom.iter_type = #loom.iter_type<temporal>}
        } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @y}
      } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @x}
      return
    }
  }
  module attributes {loom.pass_name = "Materialize"} {
    func.func @matmul__d0i0_d1i1__f10__h_v__BK32__BM32__BN32(%arg0: memref<128x128xf32>, %arg1: memref<128x256xf32>, %arg2: memref<128x256xf32>) {
      %cst = arith.constant 0.000000e+00 : f32
      %c32 = arith.constant 32 : index
      affine.parallel (%arg3) = (0) to (8) {
        affine.parallel (%arg4) = (0) to (8) {
          affine.for %arg5 = 0 to 1 {
            affine.for %arg6 = 0 to 1 {
              %12 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 8)>(%arg3, %arg6)
              %13 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 8)>(%arg4, %arg5)
              %14 = loom.alloc(%c32, %c32) on @L1 : !loom.buffer_token
              %15 = loom.init_tensor %14(%c32, %c32) : !loom.buffer_token -> tensor<?x?xf32>
              %16 = linalg.fill ins(%cst : f32) outs(%15 : tensor<?x?xf32>) -> tensor<?x?xf32>
              %17 = affine.for %arg7 = 0 to 4 iter_args(%arg8 = %16) -> (tensor<?x?xf32>) {
                %21 = arith.muli %12, %c32 : index
                %22 = arith.muli %arg7, %c32 : index
                %23 = loom.view %arg0[%21, %22] [32, 32] [1, 1], reuse : [seq = false, spat = true, temp = true] : memref<128x128xf32> -> !loom.view
                %24 = loom.alloc(%c32, %c32) on @L1 : !loom.buffer_token
                %25 = loom.copy_to_tensor %23, %24 to @L1, interconnect : [@horizontal_links], broadcast : [1, 8] : !loom.view, !loom.buffer_token -> tensor<?x?xf32>
                %26 = arith.muli %13, %c32 : index
                %27 = loom.view %arg1[%22, %26] [32, 32] [1, 1], reuse : [seq = false, spat = true, temp = true] : memref<128x256xf32> -> !loom.view
                %28 = loom.alloc(%c32, %c32) on @L1 : !loom.buffer_token
                %29 = loom.copy_to_tensor %27, %28 to @L1, interconnect : [@vertical_links], broadcast : [8, 1] : !loom.view, !loom.buffer_token -> tensor<?x?xf32>
                %30 = linalg.matmul ins(%25, %29 : tensor<?x?xf32>, tensor<?x?xf32>) outs(%16 : tensor<?x?xf32>) -> tensor<?x?xf32>
                %31 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%arg8, %30 : tensor<?x?xf32>, tensor<?x?xf32>) outs(%15 : tensor<?x?xf32>) {
                ^bb0(%in: f32, %in_0: f32, %out: f32):
                  %32 = arith.addf %in, %in_0 : f32
                  linalg.yield %32 : f32
                } -> tensor<?x?xf32>
                affine.yield %31 : tensor<?x?xf32>
              } {loom.iter_type = #loom.iter_type<sequential>}
              %18 = arith.muli %12, %c32 : index
              %19 = arith.muli %13, %c32 : index
              %20 = loom.view %arg2[%18, %19] [32, 32] [1, 1], reuse : [seq = false, spat = false, temp = false] : memref<128x256xf32> -> !loom.view
              loom.copy_from_tensor %17, %20 to @DRAM : tensor<?x?xf32>, !loom.view
            } {loom.iter_type = #loom.iter_type<temporal>}
          } {loom.iter_type = #loom.iter_type<temporal>}
        } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @y}
      } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @x}
      return
    }
    func.func @matmul__d0i0_d1i1__f10__h_v__BK64__BM32__BN32(%arg0: memref<128x128xf32>, %arg1: memref<128x256xf32>, %arg2: memref<128x256xf32>) {
      %cst = arith.constant 0.000000e+00 : f32
      %c32 = arith.constant 32 : index
      %c64 = arith.constant 64 : index
      affine.parallel (%arg3) = (0) to (8) {
        affine.parallel (%arg4) = (0) to (8) {
          affine.for %arg5 = 0 to 1 {
            affine.for %arg6 = 0 to 1 {
              %12 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 8)>(%arg3, %arg6)
              %13 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 8)>(%arg4, %arg5)
              %14 = loom.alloc(%c32, %c32) on @L1 : !loom.buffer_token
              %15 = loom.init_tensor %14(%c32, %c32) : !loom.buffer_token -> tensor<?x?xf32>
              %16 = linalg.fill ins(%cst : f32) outs(%15 : tensor<?x?xf32>) -> tensor<?x?xf32>
              %17 = affine.for %arg7 = 0 to 2 iter_args(%arg8 = %16) -> (tensor<?x?xf32>) {
                %21 = arith.muli %12, %c32 : index
                %22 = arith.muli %arg7, %c64 : index
                %23 = loom.view %arg0[%21, %22] [32, 64] [1, 1], reuse : [seq = false, spat = true, temp = true] : memref<128x128xf32> -> !loom.view
                %24 = loom.alloc(%c32, %c64) on @L1 : !loom.buffer_token
                %25 = loom.copy_to_tensor %23, %24 to @L1, interconnect : [@horizontal_links], broadcast : [1, 8] : !loom.view, !loom.buffer_token -> tensor<?x?xf32>
                %26 = arith.muli %13, %c32 : index
                %27 = loom.view %arg1[%22, %26] [64, 32] [1, 1], reuse : [seq = false, spat = true, temp = true] : memref<128x256xf32> -> !loom.view
                %28 = loom.alloc(%c64, %c32) on @L1 : !loom.buffer_token
                %29 = loom.copy_to_tensor %27, %28 to @L1, interconnect : [@vertical_links], broadcast : [8, 1] : !loom.view, !loom.buffer_token -> tensor<?x?xf32>
                %30 = linalg.matmul ins(%25, %29 : tensor<?x?xf32>, tensor<?x?xf32>) outs(%16 : tensor<?x?xf32>) -> tensor<?x?xf32>
                %31 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%arg8, %30 : tensor<?x?xf32>, tensor<?x?xf32>) outs(%15 : tensor<?x?xf32>) {
                ^bb0(%in: f32, %in_0: f32, %out: f32):
                  %32 = arith.addf %in, %in_0 : f32
                  linalg.yield %32 : f32
                } -> tensor<?x?xf32>
                affine.yield %31 : tensor<?x?xf32>
              } {loom.iter_type = #loom.iter_type<sequential>}
              %18 = arith.muli %12, %c32 : index
              %19 = arith.muli %13, %c32 : index
              %20 = loom.view %arg2[%18, %19] [32, 32] [1, 1], reuse : [seq = false, spat = false, temp = false] : memref<128x256xf32> -> !loom.view
              loom.copy_from_tensor %17, %20 to @DRAM : tensor<?x?xf32>, !loom.view
            } {loom.iter_type = #loom.iter_type<temporal>}
          } {loom.iter_type = #loom.iter_type<temporal>}
        } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @y}
      } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @x}
      return
    }
  }
  module attributes {loom.pass_name = "Materialize"} {
    func.func @matmul__d1i0_d0i1__f01__d_d__BK32__BM32__BN32(%arg0: memref<128x128xf32>, %arg1: memref<128x256xf32>, %arg2: memref<128x256xf32>) {
      %cst = arith.constant 0.000000e+00 : f32
      %c32 = arith.constant 32 : index
      affine.parallel (%arg3) = (0) to (8) {
        affine.parallel (%arg4) = (0) to (8) {
          affine.for %arg5 = 0 to 1 {
            affine.for %arg6 = 0 to 1 {
              %12 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 8)>(%arg3, %arg5)
              %13 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 8)>(%arg4, %arg6)
              %14 = loom.alloc(%c32, %c32) on @L1 : !loom.buffer_token
              %15 = loom.init_tensor %14(%c32, %c32) : !loom.buffer_token -> tensor<?x?xf32>
              %16 = linalg.fill ins(%cst : f32) outs(%15 : tensor<?x?xf32>) -> tensor<?x?xf32>
              %17 = affine.for %arg7 = 0 to 4 iter_args(%arg8 = %16) -> (tensor<?x?xf32>) {
                %21 = arith.muli %12, %c32 : index
                %22 = arith.muli %arg7, %c32 : index
                %23 = loom.view %arg0[%21, %22] [32, 32] [1, 1], reuse : [seq = false, spat = true, temp = true] : memref<128x128xf32> -> !loom.view
                %24 = loom.alloc(%c32, %c32) on @L1 : !loom.buffer_token
                %25 = loom.copy_to_tensor %23, %24 to @L1, interconnect : [], broadcast : [1, 1] : !loom.view, !loom.buffer_token -> tensor<?x?xf32>
                %26 = arith.muli %13, %c32 : index
                %27 = loom.view %arg1[%22, %26] [32, 32] [1, 1], reuse : [seq = false, spat = true, temp = true] : memref<128x256xf32> -> !loom.view
                %28 = loom.alloc(%c32, %c32) on @L1 : !loom.buffer_token
                %29 = loom.copy_to_tensor %27, %28 to @L1, interconnect : [], broadcast : [1, 1] : !loom.view, !loom.buffer_token -> tensor<?x?xf32>
                %30 = linalg.matmul ins(%25, %29 : tensor<?x?xf32>, tensor<?x?xf32>) outs(%16 : tensor<?x?xf32>) -> tensor<?x?xf32>
                %31 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%arg8, %30 : tensor<?x?xf32>, tensor<?x?xf32>) outs(%15 : tensor<?x?xf32>) {
                ^bb0(%in: f32, %in_0: f32, %out: f32):
                  %32 = arith.addf %in, %in_0 : f32
                  linalg.yield %32 : f32
                } -> tensor<?x?xf32>
                affine.yield %31 : tensor<?x?xf32>
              } {loom.iter_type = #loom.iter_type<sequential>}
              %18 = arith.muli %12, %c32 : index
              %19 = arith.muli %13, %c32 : index
              %20 = loom.view %arg2[%18, %19] [32, 32] [1, 1], reuse : [seq = false, spat = false, temp = false] : memref<128x256xf32> -> !loom.view
              loom.copy_from_tensor %17, %20 to @DRAM : tensor<?x?xf32>, !loom.view
            } {loom.iter_type = #loom.iter_type<temporal>}
          } {loom.iter_type = #loom.iter_type<temporal>}
        } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @x}
      } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @y}
      return
    }
    func.func @matmul__d1i0_d0i1__f01__d_d__BK64__BM32__BN32(%arg0: memref<128x128xf32>, %arg1: memref<128x256xf32>, %arg2: memref<128x256xf32>) {
      %cst = arith.constant 0.000000e+00 : f32
      %c32 = arith.constant 32 : index
      %c64 = arith.constant 64 : index
      affine.parallel (%arg3) = (0) to (8) {
        affine.parallel (%arg4) = (0) to (8) {
          affine.for %arg5 = 0 to 1 {
            affine.for %arg6 = 0 to 1 {
              %12 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 8)>(%arg3, %arg5)
              %13 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 8)>(%arg4, %arg6)
              %14 = loom.alloc(%c32, %c32) on @L1 : !loom.buffer_token
              %15 = loom.init_tensor %14(%c32, %c32) : !loom.buffer_token -> tensor<?x?xf32>
              %16 = linalg.fill ins(%cst : f32) outs(%15 : tensor<?x?xf32>) -> tensor<?x?xf32>
              %17 = affine.for %arg7 = 0 to 2 iter_args(%arg8 = %16) -> (tensor<?x?xf32>) {
                %21 = arith.muli %12, %c32 : index
                %22 = arith.muli %arg7, %c64 : index
                %23 = loom.view %arg0[%21, %22] [32, 64] [1, 1], reuse : [seq = false, spat = true, temp = true] : memref<128x128xf32> -> !loom.view
                %24 = loom.alloc(%c32, %c64) on @L1 : !loom.buffer_token
                %25 = loom.copy_to_tensor %23, %24 to @L1, interconnect : [], broadcast : [1, 1] : !loom.view, !loom.buffer_token -> tensor<?x?xf32>
                %26 = arith.muli %13, %c32 : index
                %27 = loom.view %arg1[%22, %26] [64, 32] [1, 1], reuse : [seq = false, spat = true, temp = true] : memref<128x256xf32> -> !loom.view
                %28 = loom.alloc(%c64, %c32) on @L1 : !loom.buffer_token
                %29 = loom.copy_to_tensor %27, %28 to @L1, interconnect : [], broadcast : [1, 1] : !loom.view, !loom.buffer_token -> tensor<?x?xf32>
                %30 = linalg.matmul ins(%25, %29 : tensor<?x?xf32>, tensor<?x?xf32>) outs(%16 : tensor<?x?xf32>) -> tensor<?x?xf32>
                %31 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%arg8, %30 : tensor<?x?xf32>, tensor<?x?xf32>) outs(%15 : tensor<?x?xf32>) {
                ^bb0(%in: f32, %in_0: f32, %out: f32):
                  %32 = arith.addf %in, %in_0 : f32
                  linalg.yield %32 : f32
                } -> tensor<?x?xf32>
                affine.yield %31 : tensor<?x?xf32>
              } {loom.iter_type = #loom.iter_type<sequential>}
              %18 = arith.muli %12, %c32 : index
              %19 = arith.muli %13, %c32 : index
              %20 = loom.view %arg2[%18, %19] [32, 32] [1, 1], reuse : [seq = false, spat = false, temp = false] : memref<128x256xf32> -> !loom.view
              loom.copy_from_tensor %17, %20 to @DRAM : tensor<?x?xf32>, !loom.view
            } {loom.iter_type = #loom.iter_type<temporal>}
          } {loom.iter_type = #loom.iter_type<temporal>}
        } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @x}
      } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @y}
      return
    }
  }
  module attributes {loom.pass_name = "Materialize"} {
    func.func @matmul__d1i0_d0i1__f01__d_h__BK32__BM32__BN32(%arg0: memref<128x128xf32>, %arg1: memref<128x256xf32>, %arg2: memref<128x256xf32>) {
      %cst = arith.constant 0.000000e+00 : f32
      %c32 = arith.constant 32 : index
      affine.parallel (%arg3) = (0) to (8) {
        affine.parallel (%arg4) = (0) to (8) {
          affine.for %arg5 = 0 to 1 {
            affine.for %arg6 = 0 to 1 {
              %12 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 8)>(%arg3, %arg5)
              %13 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 8)>(%arg4, %arg6)
              %14 = loom.alloc(%c32, %c32) on @L1 : !loom.buffer_token
              %15 = loom.init_tensor %14(%c32, %c32) : !loom.buffer_token -> tensor<?x?xf32>
              %16 = linalg.fill ins(%cst : f32) outs(%15 : tensor<?x?xf32>) -> tensor<?x?xf32>
              %17 = affine.for %arg7 = 0 to 4 iter_args(%arg8 = %16) -> (tensor<?x?xf32>) {
                %21 = arith.muli %12, %c32 : index
                %22 = arith.muli %arg7, %c32 : index
                %23 = loom.view %arg0[%21, %22] [32, 32] [1, 1], reuse : [seq = false, spat = true, temp = true] : memref<128x128xf32> -> !loom.view
                %24 = loom.alloc(%c32, %c32) on @L1 : !loom.buffer_token
                %25 = loom.copy_to_tensor %23, %24 to @L1, interconnect : [], broadcast : [1, 1] : !loom.view, !loom.buffer_token -> tensor<?x?xf32>
                %26 = arith.muli %13, %c32 : index
                %27 = loom.view %arg1[%22, %26] [32, 32] [1, 1], reuse : [seq = false, spat = true, temp = true] : memref<128x256xf32> -> !loom.view
                %28 = loom.alloc(%c32, %c32) on @L1 : !loom.buffer_token
                %29 = loom.copy_to_tensor %27, %28 to @L1, interconnect : [@horizontal_links], broadcast : [1, 8] : !loom.view, !loom.buffer_token -> tensor<?x?xf32>
                %30 = linalg.matmul ins(%25, %29 : tensor<?x?xf32>, tensor<?x?xf32>) outs(%16 : tensor<?x?xf32>) -> tensor<?x?xf32>
                %31 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%arg8, %30 : tensor<?x?xf32>, tensor<?x?xf32>) outs(%15 : tensor<?x?xf32>) {
                ^bb0(%in: f32, %in_0: f32, %out: f32):
                  %32 = arith.addf %in, %in_0 : f32
                  linalg.yield %32 : f32
                } -> tensor<?x?xf32>
                affine.yield %31 : tensor<?x?xf32>
              } {loom.iter_type = #loom.iter_type<sequential>}
              %18 = arith.muli %12, %c32 : index
              %19 = arith.muli %13, %c32 : index
              %20 = loom.view %arg2[%18, %19] [32, 32] [1, 1], reuse : [seq = false, spat = false, temp = false] : memref<128x256xf32> -> !loom.view
              loom.copy_from_tensor %17, %20 to @DRAM : tensor<?x?xf32>, !loom.view
            } {loom.iter_type = #loom.iter_type<temporal>}
          } {loom.iter_type = #loom.iter_type<temporal>}
        } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @x}
      } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @y}
      return
    }
    func.func @matmul__d1i0_d0i1__f01__d_h__BK64__BM32__BN32(%arg0: memref<128x128xf32>, %arg1: memref<128x256xf32>, %arg2: memref<128x256xf32>) {
      %cst = arith.constant 0.000000e+00 : f32
      %c32 = arith.constant 32 : index
      %c64 = arith.constant 64 : index
      affine.parallel (%arg3) = (0) to (8) {
        affine.parallel (%arg4) = (0) to (8) {
          affine.for %arg5 = 0 to 1 {
            affine.for %arg6 = 0 to 1 {
              %12 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 8)>(%arg3, %arg5)
              %13 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 8)>(%arg4, %arg6)
              %14 = loom.alloc(%c32, %c32) on @L1 : !loom.buffer_token
              %15 = loom.init_tensor %14(%c32, %c32) : !loom.buffer_token -> tensor<?x?xf32>
              %16 = linalg.fill ins(%cst : f32) outs(%15 : tensor<?x?xf32>) -> tensor<?x?xf32>
              %17 = affine.for %arg7 = 0 to 2 iter_args(%arg8 = %16) -> (tensor<?x?xf32>) {
                %21 = arith.muli %12, %c32 : index
                %22 = arith.muli %arg7, %c64 : index
                %23 = loom.view %arg0[%21, %22] [32, 64] [1, 1], reuse : [seq = false, spat = true, temp = true] : memref<128x128xf32> -> !loom.view
                %24 = loom.alloc(%c32, %c64) on @L1 : !loom.buffer_token
                %25 = loom.copy_to_tensor %23, %24 to @L1, interconnect : [], broadcast : [1, 1] : !loom.view, !loom.buffer_token -> tensor<?x?xf32>
                %26 = arith.muli %13, %c32 : index
                %27 = loom.view %arg1[%22, %26] [64, 32] [1, 1], reuse : [seq = false, spat = true, temp = true] : memref<128x256xf32> -> !loom.view
                %28 = loom.alloc(%c64, %c32) on @L1 : !loom.buffer_token
                %29 = loom.copy_to_tensor %27, %28 to @L1, interconnect : [@horizontal_links], broadcast : [1, 8] : !loom.view, !loom.buffer_token -> tensor<?x?xf32>
                %30 = linalg.matmul ins(%25, %29 : tensor<?x?xf32>, tensor<?x?xf32>) outs(%16 : tensor<?x?xf32>) -> tensor<?x?xf32>
                %31 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%arg8, %30 : tensor<?x?xf32>, tensor<?x?xf32>) outs(%15 : tensor<?x?xf32>) {
                ^bb0(%in: f32, %in_0: f32, %out: f32):
                  %32 = arith.addf %in, %in_0 : f32
                  linalg.yield %32 : f32
                } -> tensor<?x?xf32>
                affine.yield %31 : tensor<?x?xf32>
              } {loom.iter_type = #loom.iter_type<sequential>}
              %18 = arith.muli %12, %c32 : index
              %19 = arith.muli %13, %c32 : index
              %20 = loom.view %arg2[%18, %19] [32, 32] [1, 1], reuse : [seq = false, spat = false, temp = false] : memref<128x256xf32> -> !loom.view
              loom.copy_from_tensor %17, %20 to @DRAM : tensor<?x?xf32>, !loom.view
            } {loom.iter_type = #loom.iter_type<temporal>}
          } {loom.iter_type = #loom.iter_type<temporal>}
        } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @x}
      } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @y}
      return
    }
  }
  module attributes {loom.pass_name = "Materialize"} {
    func.func @matmul__d1i0_d0i1__f01__v_d__BK32__BM32__BN32(%arg0: memref<128x128xf32>, %arg1: memref<128x256xf32>, %arg2: memref<128x256xf32>) {
      %cst = arith.constant 0.000000e+00 : f32
      %c32 = arith.constant 32 : index
      affine.parallel (%arg3) = (0) to (8) {
        affine.parallel (%arg4) = (0) to (8) {
          affine.for %arg5 = 0 to 1 {
            affine.for %arg6 = 0 to 1 {
              %12 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 8)>(%arg3, %arg5)
              %13 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 8)>(%arg4, %arg6)
              %14 = loom.alloc(%c32, %c32) on @L1 : !loom.buffer_token
              %15 = loom.init_tensor %14(%c32, %c32) : !loom.buffer_token -> tensor<?x?xf32>
              %16 = linalg.fill ins(%cst : f32) outs(%15 : tensor<?x?xf32>) -> tensor<?x?xf32>
              %17 = affine.for %arg7 = 0 to 4 iter_args(%arg8 = %16) -> (tensor<?x?xf32>) {
                %21 = arith.muli %12, %c32 : index
                %22 = arith.muli %arg7, %c32 : index
                %23 = loom.view %arg0[%21, %22] [32, 32] [1, 1], reuse : [seq = false, spat = true, temp = true] : memref<128x128xf32> -> !loom.view
                %24 = loom.alloc(%c32, %c32) on @L1 : !loom.buffer_token
                %25 = loom.copy_to_tensor %23, %24 to @L1, interconnect : [@vertical_links], broadcast : [8, 1] : !loom.view, !loom.buffer_token -> tensor<?x?xf32>
                %26 = arith.muli %13, %c32 : index
                %27 = loom.view %arg1[%22, %26] [32, 32] [1, 1], reuse : [seq = false, spat = true, temp = true] : memref<128x256xf32> -> !loom.view
                %28 = loom.alloc(%c32, %c32) on @L1 : !loom.buffer_token
                %29 = loom.copy_to_tensor %27, %28 to @L1, interconnect : [], broadcast : [1, 1] : !loom.view, !loom.buffer_token -> tensor<?x?xf32>
                %30 = linalg.matmul ins(%25, %29 : tensor<?x?xf32>, tensor<?x?xf32>) outs(%16 : tensor<?x?xf32>) -> tensor<?x?xf32>
                %31 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%arg8, %30 : tensor<?x?xf32>, tensor<?x?xf32>) outs(%15 : tensor<?x?xf32>) {
                ^bb0(%in: f32, %in_0: f32, %out: f32):
                  %32 = arith.addf %in, %in_0 : f32
                  linalg.yield %32 : f32
                } -> tensor<?x?xf32>
                affine.yield %31 : tensor<?x?xf32>
              } {loom.iter_type = #loom.iter_type<sequential>}
              %18 = arith.muli %12, %c32 : index
              %19 = arith.muli %13, %c32 : index
              %20 = loom.view %arg2[%18, %19] [32, 32] [1, 1], reuse : [seq = false, spat = false, temp = false] : memref<128x256xf32> -> !loom.view
              loom.copy_from_tensor %17, %20 to @DRAM : tensor<?x?xf32>, !loom.view
            } {loom.iter_type = #loom.iter_type<temporal>}
          } {loom.iter_type = #loom.iter_type<temporal>}
        } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @x}
      } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @y}
      return
    }
    func.func @matmul__d1i0_d0i1__f01__v_d__BK64__BM32__BN32(%arg0: memref<128x128xf32>, %arg1: memref<128x256xf32>, %arg2: memref<128x256xf32>) {
      %cst = arith.constant 0.000000e+00 : f32
      %c32 = arith.constant 32 : index
      %c64 = arith.constant 64 : index
      affine.parallel (%arg3) = (0) to (8) {
        affine.parallel (%arg4) = (0) to (8) {
          affine.for %arg5 = 0 to 1 {
            affine.for %arg6 = 0 to 1 {
              %12 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 8)>(%arg3, %arg5)
              %13 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 8)>(%arg4, %arg6)
              %14 = loom.alloc(%c32, %c32) on @L1 : !loom.buffer_token
              %15 = loom.init_tensor %14(%c32, %c32) : !loom.buffer_token -> tensor<?x?xf32>
              %16 = linalg.fill ins(%cst : f32) outs(%15 : tensor<?x?xf32>) -> tensor<?x?xf32>
              %17 = affine.for %arg7 = 0 to 2 iter_args(%arg8 = %16) -> (tensor<?x?xf32>) {
                %21 = arith.muli %12, %c32 : index
                %22 = arith.muli %arg7, %c64 : index
                %23 = loom.view %arg0[%21, %22] [32, 64] [1, 1], reuse : [seq = false, spat = true, temp = true] : memref<128x128xf32> -> !loom.view
                %24 = loom.alloc(%c32, %c64) on @L1 : !loom.buffer_token
                %25 = loom.copy_to_tensor %23, %24 to @L1, interconnect : [@vertical_links], broadcast : [8, 1] : !loom.view, !loom.buffer_token -> tensor<?x?xf32>
                %26 = arith.muli %13, %c32 : index
                %27 = loom.view %arg1[%22, %26] [64, 32] [1, 1], reuse : [seq = false, spat = true, temp = true] : memref<128x256xf32> -> !loom.view
                %28 = loom.alloc(%c64, %c32) on @L1 : !loom.buffer_token
                %29 = loom.copy_to_tensor %27, %28 to @L1, interconnect : [], broadcast : [1, 1] : !loom.view, !loom.buffer_token -> tensor<?x?xf32>
                %30 = linalg.matmul ins(%25, %29 : tensor<?x?xf32>, tensor<?x?xf32>) outs(%16 : tensor<?x?xf32>) -> tensor<?x?xf32>
                %31 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%arg8, %30 : tensor<?x?xf32>, tensor<?x?xf32>) outs(%15 : tensor<?x?xf32>) {
                ^bb0(%in: f32, %in_0: f32, %out: f32):
                  %32 = arith.addf %in, %in_0 : f32
                  linalg.yield %32 : f32
                } -> tensor<?x?xf32>
                affine.yield %31 : tensor<?x?xf32>
              } {loom.iter_type = #loom.iter_type<sequential>}
              %18 = arith.muli %12, %c32 : index
              %19 = arith.muli %13, %c32 : index
              %20 = loom.view %arg2[%18, %19] [32, 32] [1, 1], reuse : [seq = false, spat = false, temp = false] : memref<128x256xf32> -> !loom.view
              loom.copy_from_tensor %17, %20 to @DRAM : tensor<?x?xf32>, !loom.view
            } {loom.iter_type = #loom.iter_type<temporal>}
          } {loom.iter_type = #loom.iter_type<temporal>}
        } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @x}
      } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @y}
      return
    }
  }
  module attributes {loom.pass_name = "Materialize"} {
    func.func @matmul__d1i0_d0i1__f01__v_h__BK32__BM32__BN32(%arg0: memref<128x128xf32>, %arg1: memref<128x256xf32>, %arg2: memref<128x256xf32>) {
      %cst = arith.constant 0.000000e+00 : f32
      %c32 = arith.constant 32 : index
      affine.parallel (%arg3) = (0) to (8) {
        affine.parallel (%arg4) = (0) to (8) {
          affine.for %arg5 = 0 to 1 {
            affine.for %arg6 = 0 to 1 {
              %12 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 8)>(%arg3, %arg5)
              %13 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 8)>(%arg4, %arg6)
              %14 = loom.alloc(%c32, %c32) on @L1 : !loom.buffer_token
              %15 = loom.init_tensor %14(%c32, %c32) : !loom.buffer_token -> tensor<?x?xf32>
              %16 = linalg.fill ins(%cst : f32) outs(%15 : tensor<?x?xf32>) -> tensor<?x?xf32>
              %17 = affine.for %arg7 = 0 to 4 iter_args(%arg8 = %16) -> (tensor<?x?xf32>) {
                %21 = arith.muli %12, %c32 : index
                %22 = arith.muli %arg7, %c32 : index
                %23 = loom.view %arg0[%21, %22] [32, 32] [1, 1], reuse : [seq = false, spat = true, temp = true] : memref<128x128xf32> -> !loom.view
                %24 = loom.alloc(%c32, %c32) on @L1 : !loom.buffer_token
                %25 = loom.copy_to_tensor %23, %24 to @L1, interconnect : [@vertical_links], broadcast : [8, 1] : !loom.view, !loom.buffer_token -> tensor<?x?xf32>
                %26 = arith.muli %13, %c32 : index
                %27 = loom.view %arg1[%22, %26] [32, 32] [1, 1], reuse : [seq = false, spat = true, temp = true] : memref<128x256xf32> -> !loom.view
                %28 = loom.alloc(%c32, %c32) on @L1 : !loom.buffer_token
                %29 = loom.copy_to_tensor %27, %28 to @L1, interconnect : [@horizontal_links], broadcast : [1, 8] : !loom.view, !loom.buffer_token -> tensor<?x?xf32>
                %30 = linalg.matmul ins(%25, %29 : tensor<?x?xf32>, tensor<?x?xf32>) outs(%16 : tensor<?x?xf32>) -> tensor<?x?xf32>
                %31 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%arg8, %30 : tensor<?x?xf32>, tensor<?x?xf32>) outs(%15 : tensor<?x?xf32>) {
                ^bb0(%in: f32, %in_0: f32, %out: f32):
                  %32 = arith.addf %in, %in_0 : f32
                  linalg.yield %32 : f32
                } -> tensor<?x?xf32>
                affine.yield %31 : tensor<?x?xf32>
              } {loom.iter_type = #loom.iter_type<sequential>}
              %18 = arith.muli %12, %c32 : index
              %19 = arith.muli %13, %c32 : index
              %20 = loom.view %arg2[%18, %19] [32, 32] [1, 1], reuse : [seq = false, spat = false, temp = false] : memref<128x256xf32> -> !loom.view
              loom.copy_from_tensor %17, %20 to @DRAM : tensor<?x?xf32>, !loom.view
            } {loom.iter_type = #loom.iter_type<temporal>}
          } {loom.iter_type = #loom.iter_type<temporal>}
        } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @x}
      } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @y}
      return
    }
    func.func @matmul__d1i0_d0i1__f01__v_h__BK64__BM32__BN32(%arg0: memref<128x128xf32>, %arg1: memref<128x256xf32>, %arg2: memref<128x256xf32>) {
      %cst = arith.constant 0.000000e+00 : f32
      %c32 = arith.constant 32 : index
      %c64 = arith.constant 64 : index
      affine.parallel (%arg3) = (0) to (8) {
        affine.parallel (%arg4) = (0) to (8) {
          affine.for %arg5 = 0 to 1 {
            affine.for %arg6 = 0 to 1 {
              %12 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 8)>(%arg3, %arg5)
              %13 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 8)>(%arg4, %arg6)
              %14 = loom.alloc(%c32, %c32) on @L1 : !loom.buffer_token
              %15 = loom.init_tensor %14(%c32, %c32) : !loom.buffer_token -> tensor<?x?xf32>
              %16 = linalg.fill ins(%cst : f32) outs(%15 : tensor<?x?xf32>) -> tensor<?x?xf32>
              %17 = affine.for %arg7 = 0 to 2 iter_args(%arg8 = %16) -> (tensor<?x?xf32>) {
                %21 = arith.muli %12, %c32 : index
                %22 = arith.muli %arg7, %c64 : index
                %23 = loom.view %arg0[%21, %22] [32, 64] [1, 1], reuse : [seq = false, spat = true, temp = true] : memref<128x128xf32> -> !loom.view
                %24 = loom.alloc(%c32, %c64) on @L1 : !loom.buffer_token
                %25 = loom.copy_to_tensor %23, %24 to @L1, interconnect : [@vertical_links], broadcast : [8, 1] : !loom.view, !loom.buffer_token -> tensor<?x?xf32>
                %26 = arith.muli %13, %c32 : index
                %27 = loom.view %arg1[%22, %26] [64, 32] [1, 1], reuse : [seq = false, spat = true, temp = true] : memref<128x256xf32> -> !loom.view
                %28 = loom.alloc(%c64, %c32) on @L1 : !loom.buffer_token
                %29 = loom.copy_to_tensor %27, %28 to @L1, interconnect : [@horizontal_links], broadcast : [1, 8] : !loom.view, !loom.buffer_token -> tensor<?x?xf32>
                %30 = linalg.matmul ins(%25, %29 : tensor<?x?xf32>, tensor<?x?xf32>) outs(%16 : tensor<?x?xf32>) -> tensor<?x?xf32>
                %31 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%arg8, %30 : tensor<?x?xf32>, tensor<?x?xf32>) outs(%15 : tensor<?x?xf32>) {
                ^bb0(%in: f32, %in_0: f32, %out: f32):
                  %32 = arith.addf %in, %in_0 : f32
                  linalg.yield %32 : f32
                } -> tensor<?x?xf32>
                affine.yield %31 : tensor<?x?xf32>
              } {loom.iter_type = #loom.iter_type<sequential>}
              %18 = arith.muli %12, %c32 : index
              %19 = arith.muli %13, %c32 : index
              %20 = loom.view %arg2[%18, %19] [32, 32] [1, 1], reuse : [seq = false, spat = false, temp = false] : memref<128x256xf32> -> !loom.view
              loom.copy_from_tensor %17, %20 to @DRAM : tensor<?x?xf32>, !loom.view
            } {loom.iter_type = #loom.iter_type<temporal>}
          } {loom.iter_type = #loom.iter_type<temporal>}
        } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @x}
      } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @y}
      return
    }
  }
  module attributes {loom.pass_name = "Materialize"} {
    func.func @matmul__d1i0_d0i1__f10__d_d__BK32__BM32__BN32(%arg0: memref<128x128xf32>, %arg1: memref<128x256xf32>, %arg2: memref<128x256xf32>) {
      %cst = arith.constant 0.000000e+00 : f32
      %c32 = arith.constant 32 : index
      affine.parallel (%arg3) = (0) to (8) {
        affine.parallel (%arg4) = (0) to (8) {
          affine.for %arg5 = 0 to 1 {
            affine.for %arg6 = 0 to 1 {
              %12 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 8)>(%arg3, %arg6)
              %13 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 8)>(%arg4, %arg5)
              %14 = loom.alloc(%c32, %c32) on @L1 : !loom.buffer_token
              %15 = loom.init_tensor %14(%c32, %c32) : !loom.buffer_token -> tensor<?x?xf32>
              %16 = linalg.fill ins(%cst : f32) outs(%15 : tensor<?x?xf32>) -> tensor<?x?xf32>
              %17 = affine.for %arg7 = 0 to 4 iter_args(%arg8 = %16) -> (tensor<?x?xf32>) {
                %21 = arith.muli %12, %c32 : index
                %22 = arith.muli %arg7, %c32 : index
                %23 = loom.view %arg0[%21, %22] [32, 32] [1, 1], reuse : [seq = false, spat = true, temp = true] : memref<128x128xf32> -> !loom.view
                %24 = loom.alloc(%c32, %c32) on @L1 : !loom.buffer_token
                %25 = loom.copy_to_tensor %23, %24 to @L1, interconnect : [], broadcast : [1, 1] : !loom.view, !loom.buffer_token -> tensor<?x?xf32>
                %26 = arith.muli %13, %c32 : index
                %27 = loom.view %arg1[%22, %26] [32, 32] [1, 1], reuse : [seq = false, spat = true, temp = true] : memref<128x256xf32> -> !loom.view
                %28 = loom.alloc(%c32, %c32) on @L1 : !loom.buffer_token
                %29 = loom.copy_to_tensor %27, %28 to @L1, interconnect : [], broadcast : [1, 1] : !loom.view, !loom.buffer_token -> tensor<?x?xf32>
                %30 = linalg.matmul ins(%25, %29 : tensor<?x?xf32>, tensor<?x?xf32>) outs(%16 : tensor<?x?xf32>) -> tensor<?x?xf32>
                %31 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%arg8, %30 : tensor<?x?xf32>, tensor<?x?xf32>) outs(%15 : tensor<?x?xf32>) {
                ^bb0(%in: f32, %in_0: f32, %out: f32):
                  %32 = arith.addf %in, %in_0 : f32
                  linalg.yield %32 : f32
                } -> tensor<?x?xf32>
                affine.yield %31 : tensor<?x?xf32>
              } {loom.iter_type = #loom.iter_type<sequential>}
              %18 = arith.muli %12, %c32 : index
              %19 = arith.muli %13, %c32 : index
              %20 = loom.view %arg2[%18, %19] [32, 32] [1, 1], reuse : [seq = false, spat = false, temp = false] : memref<128x256xf32> -> !loom.view
              loom.copy_from_tensor %17, %20 to @DRAM : tensor<?x?xf32>, !loom.view
            } {loom.iter_type = #loom.iter_type<temporal>}
          } {loom.iter_type = #loom.iter_type<temporal>}
        } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @x}
      } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @y}
      return
    }
    func.func @matmul__d1i0_d0i1__f10__d_d__BK64__BM32__BN32(%arg0: memref<128x128xf32>, %arg1: memref<128x256xf32>, %arg2: memref<128x256xf32>) {
      %cst = arith.constant 0.000000e+00 : f32
      %c32 = arith.constant 32 : index
      %c64 = arith.constant 64 : index
      affine.parallel (%arg3) = (0) to (8) {
        affine.parallel (%arg4) = (0) to (8) {
          affine.for %arg5 = 0 to 1 {
            affine.for %arg6 = 0 to 1 {
              %12 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 8)>(%arg3, %arg6)
              %13 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 8)>(%arg4, %arg5)
              %14 = loom.alloc(%c32, %c32) on @L1 : !loom.buffer_token
              %15 = loom.init_tensor %14(%c32, %c32) : !loom.buffer_token -> tensor<?x?xf32>
              %16 = linalg.fill ins(%cst : f32) outs(%15 : tensor<?x?xf32>) -> tensor<?x?xf32>
              %17 = affine.for %arg7 = 0 to 2 iter_args(%arg8 = %16) -> (tensor<?x?xf32>) {
                %21 = arith.muli %12, %c32 : index
                %22 = arith.muli %arg7, %c64 : index
                %23 = loom.view %arg0[%21, %22] [32, 64] [1, 1], reuse : [seq = false, spat = true, temp = true] : memref<128x128xf32> -> !loom.view
                %24 = loom.alloc(%c32, %c64) on @L1 : !loom.buffer_token
                %25 = loom.copy_to_tensor %23, %24 to @L1, interconnect : [], broadcast : [1, 1] : !loom.view, !loom.buffer_token -> tensor<?x?xf32>
                %26 = arith.muli %13, %c32 : index
                %27 = loom.view %arg1[%22, %26] [64, 32] [1, 1], reuse : [seq = false, spat = true, temp = true] : memref<128x256xf32> -> !loom.view
                %28 = loom.alloc(%c64, %c32) on @L1 : !loom.buffer_token
                %29 = loom.copy_to_tensor %27, %28 to @L1, interconnect : [], broadcast : [1, 1] : !loom.view, !loom.buffer_token -> tensor<?x?xf32>
                %30 = linalg.matmul ins(%25, %29 : tensor<?x?xf32>, tensor<?x?xf32>) outs(%16 : tensor<?x?xf32>) -> tensor<?x?xf32>
                %31 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%arg8, %30 : tensor<?x?xf32>, tensor<?x?xf32>) outs(%15 : tensor<?x?xf32>) {
                ^bb0(%in: f32, %in_0: f32, %out: f32):
                  %32 = arith.addf %in, %in_0 : f32
                  linalg.yield %32 : f32
                } -> tensor<?x?xf32>
                affine.yield %31 : tensor<?x?xf32>
              } {loom.iter_type = #loom.iter_type<sequential>}
              %18 = arith.muli %12, %c32 : index
              %19 = arith.muli %13, %c32 : index
              %20 = loom.view %arg2[%18, %19] [32, 32] [1, 1], reuse : [seq = false, spat = false, temp = false] : memref<128x256xf32> -> !loom.view
              loom.copy_from_tensor %17, %20 to @DRAM : tensor<?x?xf32>, !loom.view
            } {loom.iter_type = #loom.iter_type<temporal>}
          } {loom.iter_type = #loom.iter_type<temporal>}
        } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @x}
      } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @y}
      return
    }
  }
  module attributes {loom.pass_name = "Materialize"} {
    func.func @matmul__d1i0_d0i1__f10__d_h__BK32__BM32__BN32(%arg0: memref<128x128xf32>, %arg1: memref<128x256xf32>, %arg2: memref<128x256xf32>) {
      %cst = arith.constant 0.000000e+00 : f32
      %c32 = arith.constant 32 : index
      affine.parallel (%arg3) = (0) to (8) {
        affine.parallel (%arg4) = (0) to (8) {
          affine.for %arg5 = 0 to 1 {
            affine.for %arg6 = 0 to 1 {
              %12 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 8)>(%arg3, %arg6)
              %13 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 8)>(%arg4, %arg5)
              %14 = loom.alloc(%c32, %c32) on @L1 : !loom.buffer_token
              %15 = loom.init_tensor %14(%c32, %c32) : !loom.buffer_token -> tensor<?x?xf32>
              %16 = linalg.fill ins(%cst : f32) outs(%15 : tensor<?x?xf32>) -> tensor<?x?xf32>
              %17 = affine.for %arg7 = 0 to 4 iter_args(%arg8 = %16) -> (tensor<?x?xf32>) {
                %21 = arith.muli %12, %c32 : index
                %22 = arith.muli %arg7, %c32 : index
                %23 = loom.view %arg0[%21, %22] [32, 32] [1, 1], reuse : [seq = false, spat = true, temp = true] : memref<128x128xf32> -> !loom.view
                %24 = loom.alloc(%c32, %c32) on @L1 : !loom.buffer_token
                %25 = loom.copy_to_tensor %23, %24 to @L1, interconnect : [], broadcast : [1, 1] : !loom.view, !loom.buffer_token -> tensor<?x?xf32>
                %26 = arith.muli %13, %c32 : index
                %27 = loom.view %arg1[%22, %26] [32, 32] [1, 1], reuse : [seq = false, spat = true, temp = true] : memref<128x256xf32> -> !loom.view
                %28 = loom.alloc(%c32, %c32) on @L1 : !loom.buffer_token
                %29 = loom.copy_to_tensor %27, %28 to @L1, interconnect : [@horizontal_links], broadcast : [1, 8] : !loom.view, !loom.buffer_token -> tensor<?x?xf32>
                %30 = linalg.matmul ins(%25, %29 : tensor<?x?xf32>, tensor<?x?xf32>) outs(%16 : tensor<?x?xf32>) -> tensor<?x?xf32>
                %31 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%arg8, %30 : tensor<?x?xf32>, tensor<?x?xf32>) outs(%15 : tensor<?x?xf32>) {
                ^bb0(%in: f32, %in_0: f32, %out: f32):
                  %32 = arith.addf %in, %in_0 : f32
                  linalg.yield %32 : f32
                } -> tensor<?x?xf32>
                affine.yield %31 : tensor<?x?xf32>
              } {loom.iter_type = #loom.iter_type<sequential>}
              %18 = arith.muli %12, %c32 : index
              %19 = arith.muli %13, %c32 : index
              %20 = loom.view %arg2[%18, %19] [32, 32] [1, 1], reuse : [seq = false, spat = false, temp = false] : memref<128x256xf32> -> !loom.view
              loom.copy_from_tensor %17, %20 to @DRAM : tensor<?x?xf32>, !loom.view
            } {loom.iter_type = #loom.iter_type<temporal>}
          } {loom.iter_type = #loom.iter_type<temporal>}
        } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @x}
      } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @y}
      return
    }
    func.func @matmul__d1i0_d0i1__f10__d_h__BK64__BM32__BN32(%arg0: memref<128x128xf32>, %arg1: memref<128x256xf32>, %arg2: memref<128x256xf32>) {
      %cst = arith.constant 0.000000e+00 : f32
      %c32 = arith.constant 32 : index
      %c64 = arith.constant 64 : index
      affine.parallel (%arg3) = (0) to (8) {
        affine.parallel (%arg4) = (0) to (8) {
          affine.for %arg5 = 0 to 1 {
            affine.for %arg6 = 0 to 1 {
              %12 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 8)>(%arg3, %arg6)
              %13 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 8)>(%arg4, %arg5)
              %14 = loom.alloc(%c32, %c32) on @L1 : !loom.buffer_token
              %15 = loom.init_tensor %14(%c32, %c32) : !loom.buffer_token -> tensor<?x?xf32>
              %16 = linalg.fill ins(%cst : f32) outs(%15 : tensor<?x?xf32>) -> tensor<?x?xf32>
              %17 = affine.for %arg7 = 0 to 2 iter_args(%arg8 = %16) -> (tensor<?x?xf32>) {
                %21 = arith.muli %12, %c32 : index
                %22 = arith.muli %arg7, %c64 : index
                %23 = loom.view %arg0[%21, %22] [32, 64] [1, 1], reuse : [seq = false, spat = true, temp = true] : memref<128x128xf32> -> !loom.view
                %24 = loom.alloc(%c32, %c64) on @L1 : !loom.buffer_token
                %25 = loom.copy_to_tensor %23, %24 to @L1, interconnect : [], broadcast : [1, 1] : !loom.view, !loom.buffer_token -> tensor<?x?xf32>
                %26 = arith.muli %13, %c32 : index
                %27 = loom.view %arg1[%22, %26] [64, 32] [1, 1], reuse : [seq = false, spat = true, temp = true] : memref<128x256xf32> -> !loom.view
                %28 = loom.alloc(%c64, %c32) on @L1 : !loom.buffer_token
                %29 = loom.copy_to_tensor %27, %28 to @L1, interconnect : [@horizontal_links], broadcast : [1, 8] : !loom.view, !loom.buffer_token -> tensor<?x?xf32>
                %30 = linalg.matmul ins(%25, %29 : tensor<?x?xf32>, tensor<?x?xf32>) outs(%16 : tensor<?x?xf32>) -> tensor<?x?xf32>
                %31 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%arg8, %30 : tensor<?x?xf32>, tensor<?x?xf32>) outs(%15 : tensor<?x?xf32>) {
                ^bb0(%in: f32, %in_0: f32, %out: f32):
                  %32 = arith.addf %in, %in_0 : f32
                  linalg.yield %32 : f32
                } -> tensor<?x?xf32>
                affine.yield %31 : tensor<?x?xf32>
              } {loom.iter_type = #loom.iter_type<sequential>}
              %18 = arith.muli %12, %c32 : index
              %19 = arith.muli %13, %c32 : index
              %20 = loom.view %arg2[%18, %19] [32, 32] [1, 1], reuse : [seq = false, spat = false, temp = false] : memref<128x256xf32> -> !loom.view
              loom.copy_from_tensor %17, %20 to @DRAM : tensor<?x?xf32>, !loom.view
            } {loom.iter_type = #loom.iter_type<temporal>}
          } {loom.iter_type = #loom.iter_type<temporal>}
        } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @x}
      } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @y}
      return
    }
  }
  module attributes {loom.pass_name = "Materialize"} {
    func.func @matmul__d1i0_d0i1__f10__v_d__BK32__BM32__BN32(%arg0: memref<128x128xf32>, %arg1: memref<128x256xf32>, %arg2: memref<128x256xf32>) {
      %cst = arith.constant 0.000000e+00 : f32
      %c32 = arith.constant 32 : index
      affine.parallel (%arg3) = (0) to (8) {
        affine.parallel (%arg4) = (0) to (8) {
          affine.for %arg5 = 0 to 1 {
            affine.for %arg6 = 0 to 1 {
              %12 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 8)>(%arg3, %arg6)
              %13 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 8)>(%arg4, %arg5)
              %14 = loom.alloc(%c32, %c32) on @L1 : !loom.buffer_token
              %15 = loom.init_tensor %14(%c32, %c32) : !loom.buffer_token -> tensor<?x?xf32>
              %16 = linalg.fill ins(%cst : f32) outs(%15 : tensor<?x?xf32>) -> tensor<?x?xf32>
              %17 = affine.for %arg7 = 0 to 4 iter_args(%arg8 = %16) -> (tensor<?x?xf32>) {
                %21 = arith.muli %12, %c32 : index
                %22 = arith.muli %arg7, %c32 : index
                %23 = loom.view %arg0[%21, %22] [32, 32] [1, 1], reuse : [seq = false, spat = true, temp = true] : memref<128x128xf32> -> !loom.view
                %24 = loom.alloc(%c32, %c32) on @L1 : !loom.buffer_token
                %25 = loom.copy_to_tensor %23, %24 to @L1, interconnect : [@vertical_links], broadcast : [8, 1] : !loom.view, !loom.buffer_token -> tensor<?x?xf32>
                %26 = arith.muli %13, %c32 : index
                %27 = loom.view %arg1[%22, %26] [32, 32] [1, 1], reuse : [seq = false, spat = true, temp = true] : memref<128x256xf32> -> !loom.view
                %28 = loom.alloc(%c32, %c32) on @L1 : !loom.buffer_token
                %29 = loom.copy_to_tensor %27, %28 to @L1, interconnect : [], broadcast : [1, 1] : !loom.view, !loom.buffer_token -> tensor<?x?xf32>
                %30 = linalg.matmul ins(%25, %29 : tensor<?x?xf32>, tensor<?x?xf32>) outs(%16 : tensor<?x?xf32>) -> tensor<?x?xf32>
                %31 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%arg8, %30 : tensor<?x?xf32>, tensor<?x?xf32>) outs(%15 : tensor<?x?xf32>) {
                ^bb0(%in: f32, %in_0: f32, %out: f32):
                  %32 = arith.addf %in, %in_0 : f32
                  linalg.yield %32 : f32
                } -> tensor<?x?xf32>
                affine.yield %31 : tensor<?x?xf32>
              } {loom.iter_type = #loom.iter_type<sequential>}
              %18 = arith.muli %12, %c32 : index
              %19 = arith.muli %13, %c32 : index
              %20 = loom.view %arg2[%18, %19] [32, 32] [1, 1], reuse : [seq = false, spat = false, temp = false] : memref<128x256xf32> -> !loom.view
              loom.copy_from_tensor %17, %20 to @DRAM : tensor<?x?xf32>, !loom.view
            } {loom.iter_type = #loom.iter_type<temporal>}
          } {loom.iter_type = #loom.iter_type<temporal>}
        } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @x}
      } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @y}
      return
    }
    func.func @matmul__d1i0_d0i1__f10__v_d__BK64__BM32__BN32(%arg0: memref<128x128xf32>, %arg1: memref<128x256xf32>, %arg2: memref<128x256xf32>) {
      %cst = arith.constant 0.000000e+00 : f32
      %c32 = arith.constant 32 : index
      %c64 = arith.constant 64 : index
      affine.parallel (%arg3) = (0) to (8) {
        affine.parallel (%arg4) = (0) to (8) {
          affine.for %arg5 = 0 to 1 {
            affine.for %arg6 = 0 to 1 {
              %12 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 8)>(%arg3, %arg6)
              %13 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 8)>(%arg4, %arg5)
              %14 = loom.alloc(%c32, %c32) on @L1 : !loom.buffer_token
              %15 = loom.init_tensor %14(%c32, %c32) : !loom.buffer_token -> tensor<?x?xf32>
              %16 = linalg.fill ins(%cst : f32) outs(%15 : tensor<?x?xf32>) -> tensor<?x?xf32>
              %17 = affine.for %arg7 = 0 to 2 iter_args(%arg8 = %16) -> (tensor<?x?xf32>) {
                %21 = arith.muli %12, %c32 : index
                %22 = arith.muli %arg7, %c64 : index
                %23 = loom.view %arg0[%21, %22] [32, 64] [1, 1], reuse : [seq = false, spat = true, temp = true] : memref<128x128xf32> -> !loom.view
                %24 = loom.alloc(%c32, %c64) on @L1 : !loom.buffer_token
                %25 = loom.copy_to_tensor %23, %24 to @L1, interconnect : [@vertical_links], broadcast : [8, 1] : !loom.view, !loom.buffer_token -> tensor<?x?xf32>
                %26 = arith.muli %13, %c32 : index
                %27 = loom.view %arg1[%22, %26] [64, 32] [1, 1], reuse : [seq = false, spat = true, temp = true] : memref<128x256xf32> -> !loom.view
                %28 = loom.alloc(%c64, %c32) on @L1 : !loom.buffer_token
                %29 = loom.copy_to_tensor %27, %28 to @L1, interconnect : [], broadcast : [1, 1] : !loom.view, !loom.buffer_token -> tensor<?x?xf32>
                %30 = linalg.matmul ins(%25, %29 : tensor<?x?xf32>, tensor<?x?xf32>) outs(%16 : tensor<?x?xf32>) -> tensor<?x?xf32>
                %31 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%arg8, %30 : tensor<?x?xf32>, tensor<?x?xf32>) outs(%15 : tensor<?x?xf32>) {
                ^bb0(%in: f32, %in_0: f32, %out: f32):
                  %32 = arith.addf %in, %in_0 : f32
                  linalg.yield %32 : f32
                } -> tensor<?x?xf32>
                affine.yield %31 : tensor<?x?xf32>
              } {loom.iter_type = #loom.iter_type<sequential>}
              %18 = arith.muli %12, %c32 : index
              %19 = arith.muli %13, %c32 : index
              %20 = loom.view %arg2[%18, %19] [32, 32] [1, 1], reuse : [seq = false, spat = false, temp = false] : memref<128x256xf32> -> !loom.view
              loom.copy_from_tensor %17, %20 to @DRAM : tensor<?x?xf32>, !loom.view
            } {loom.iter_type = #loom.iter_type<temporal>}
          } {loom.iter_type = #loom.iter_type<temporal>}
        } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @x}
      } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @y}
      return
    }
  }
  module attributes {loom.pass_name = "Materialize"} {
    func.func @matmul__d1i0_d0i1__f10__v_h__BK32__BM32__BN32(%arg0: memref<128x128xf32>, %arg1: memref<128x256xf32>, %arg2: memref<128x256xf32>) {
      %cst = arith.constant 0.000000e+00 : f32
      %c32 = arith.constant 32 : index
      affine.parallel (%arg3) = (0) to (8) {
        affine.parallel (%arg4) = (0) to (8) {
          affine.for %arg5 = 0 to 1 {
            affine.for %arg6 = 0 to 1 {
              %12 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 8)>(%arg3, %arg6)
              %13 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 8)>(%arg4, %arg5)
              %14 = loom.alloc(%c32, %c32) on @L1 : !loom.buffer_token
              %15 = loom.init_tensor %14(%c32, %c32) : !loom.buffer_token -> tensor<?x?xf32>
              %16 = linalg.fill ins(%cst : f32) outs(%15 : tensor<?x?xf32>) -> tensor<?x?xf32>
              %17 = affine.for %arg7 = 0 to 4 iter_args(%arg8 = %16) -> (tensor<?x?xf32>) {
                %21 = arith.muli %12, %c32 : index
                %22 = arith.muli %arg7, %c32 : index
                %23 = loom.view %arg0[%21, %22] [32, 32] [1, 1], reuse : [seq = false, spat = true, temp = true] : memref<128x128xf32> -> !loom.view
                %24 = loom.alloc(%c32, %c32) on @L1 : !loom.buffer_token
                %25 = loom.copy_to_tensor %23, %24 to @L1, interconnect : [@vertical_links], broadcast : [8, 1] : !loom.view, !loom.buffer_token -> tensor<?x?xf32>
                %26 = arith.muli %13, %c32 : index
                %27 = loom.view %arg1[%22, %26] [32, 32] [1, 1], reuse : [seq = false, spat = true, temp = true] : memref<128x256xf32> -> !loom.view
                %28 = loom.alloc(%c32, %c32) on @L1 : !loom.buffer_token
                %29 = loom.copy_to_tensor %27, %28 to @L1, interconnect : [@horizontal_links], broadcast : [1, 8] : !loom.view, !loom.buffer_token -> tensor<?x?xf32>
                %30 = linalg.matmul ins(%25, %29 : tensor<?x?xf32>, tensor<?x?xf32>) outs(%16 : tensor<?x?xf32>) -> tensor<?x?xf32>
                %31 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%arg8, %30 : tensor<?x?xf32>, tensor<?x?xf32>) outs(%15 : tensor<?x?xf32>) {
                ^bb0(%in: f32, %in_0: f32, %out: f32):
                  %32 = arith.addf %in, %in_0 : f32
                  linalg.yield %32 : f32
                } -> tensor<?x?xf32>
                affine.yield %31 : tensor<?x?xf32>
              } {loom.iter_type = #loom.iter_type<sequential>}
              %18 = arith.muli %12, %c32 : index
              %19 = arith.muli %13, %c32 : index
              %20 = loom.view %arg2[%18, %19] [32, 32] [1, 1], reuse : [seq = false, spat = false, temp = false] : memref<128x256xf32> -> !loom.view
              loom.copy_from_tensor %17, %20 to @DRAM : tensor<?x?xf32>, !loom.view
            } {loom.iter_type = #loom.iter_type<temporal>}
          } {loom.iter_type = #loom.iter_type<temporal>}
        } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @x}
      } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @y}
      return
    }
    func.func @matmul__d1i0_d0i1__f10__v_h__BK64__BM32__BN32(%arg0: memref<128x128xf32>, %arg1: memref<128x256xf32>, %arg2: memref<128x256xf32>) {
      %cst = arith.constant 0.000000e+00 : f32
      %c32 = arith.constant 32 : index
      %c64 = arith.constant 64 : index
      affine.parallel (%arg3) = (0) to (8) {
        affine.parallel (%arg4) = (0) to (8) {
          affine.for %arg5 = 0 to 1 {
            affine.for %arg6 = 0 to 1 {
              %12 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 8)>(%arg3, %arg6)
              %13 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 8)>(%arg4, %arg5)
              %14 = loom.alloc(%c32, %c32) on @L1 : !loom.buffer_token
              %15 = loom.init_tensor %14(%c32, %c32) : !loom.buffer_token -> tensor<?x?xf32>
              %16 = linalg.fill ins(%cst : f32) outs(%15 : tensor<?x?xf32>) -> tensor<?x?xf32>
              %17 = affine.for %arg7 = 0 to 2 iter_args(%arg8 = %16) -> (tensor<?x?xf32>) {
                %21 = arith.muli %12, %c32 : index
                %22 = arith.muli %arg7, %c64 : index
                %23 = loom.view %arg0[%21, %22] [32, 64] [1, 1], reuse : [seq = false, spat = true, temp = true] : memref<128x128xf32> -> !loom.view
                %24 = loom.alloc(%c32, %c64) on @L1 : !loom.buffer_token
                %25 = loom.copy_to_tensor %23, %24 to @L1, interconnect : [@vertical_links], broadcast : [8, 1] : !loom.view, !loom.buffer_token -> tensor<?x?xf32>
                %26 = arith.muli %13, %c32 : index
                %27 = loom.view %arg1[%22, %26] [64, 32] [1, 1], reuse : [seq = false, spat = true, temp = true] : memref<128x256xf32> -> !loom.view
                %28 = loom.alloc(%c64, %c32) on @L1 : !loom.buffer_token
                %29 = loom.copy_to_tensor %27, %28 to @L1, interconnect : [@horizontal_links], broadcast : [1, 8] : !loom.view, !loom.buffer_token -> tensor<?x?xf32>
                %30 = linalg.matmul ins(%25, %29 : tensor<?x?xf32>, tensor<?x?xf32>) outs(%16 : tensor<?x?xf32>) -> tensor<?x?xf32>
                %31 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%arg8, %30 : tensor<?x?xf32>, tensor<?x?xf32>) outs(%15 : tensor<?x?xf32>) {
                ^bb0(%in: f32, %in_0: f32, %out: f32):
                  %32 = arith.addf %in, %in_0 : f32
                  linalg.yield %32 : f32
                } -> tensor<?x?xf32>
                affine.yield %31 : tensor<?x?xf32>
              } {loom.iter_type = #loom.iter_type<sequential>}
              %18 = arith.muli %12, %c32 : index
              %19 = arith.muli %13, %c32 : index
              %20 = loom.view %arg2[%18, %19] [32, 32] [1, 1], reuse : [seq = false, spat = false, temp = false] : memref<128x256xf32> -> !loom.view
              loom.copy_from_tensor %17, %20 to @DRAM : tensor<?x?xf32>, !loom.view
            } {loom.iter_type = #loom.iter_type<temporal>}
          } {loom.iter_type = #loom.iter_type<temporal>}
        } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @x}
      } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @y}
      return
    }
  }
  module attributes {loom.pass_name = "Materialize"} {
    func.func @matmul__d0i1_d1i1__f01__d_d__BK32__BM32__BN32(%arg0: memref<128x128xf32>, %arg1: memref<128x256xf32>, %arg2: memref<128x256xf32>) {
      %cst = arith.constant 0.000000e+00 : f32
      %c32 = arith.constant 32 : index
      affine.parallel (%arg3) = (0) to (8) {
        affine.parallel (%arg4) = (0) to (8) {
          affine.for %arg5 = 0 to 4 {
            affine.for %arg6 = 0 to 1 {
              %12 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 8 + d1 + d2 * 64)>(%arg3, %arg4, %arg6)
              %13 = loom.alloc(%c32, %c32) on @L1 : !loom.buffer_token
              %14 = loom.init_tensor %13(%c32, %c32) : !loom.buffer_token -> tensor<?x?xf32>
              %15 = linalg.fill ins(%cst : f32) outs(%14 : tensor<?x?xf32>) -> tensor<?x?xf32>
              %16 = affine.for %arg7 = 0 to 4 iter_args(%arg8 = %15) -> (tensor<?x?xf32>) {
                %20 = arith.muli %arg5, %c32 : index
                %21 = arith.muli %arg7, %c32 : index
                %22 = loom.view %arg0[%20, %21] [32, 32] [1, 1], reuse : [seq = false, spat = true, temp = true] : memref<128x128xf32> -> !loom.view
                %23 = loom.alloc(%c32, %c32) on @L1 : !loom.buffer_token
                %24 = loom.copy_to_tensor %22, %23 to @L1, interconnect : [], broadcast : [1, 1] : !loom.view, !loom.buffer_token -> tensor<?x?xf32>
                %25 = arith.muli %12, %c32 : index
                %26 = loom.view %arg1[%21, %25] [32, 32] [1, 1], reuse : [seq = false, spat = false, temp = true] : memref<128x256xf32> -> !loom.view
                %27 = loom.alloc(%c32, %c32) on @L1 : !loom.buffer_token
                %28 = loom.copy_to_tensor %26, %27 to @L1, interconnect : [], broadcast : [1, 1] : !loom.view, !loom.buffer_token -> tensor<?x?xf32>
                %29 = linalg.matmul ins(%24, %28 : tensor<?x?xf32>, tensor<?x?xf32>) outs(%15 : tensor<?x?xf32>) -> tensor<?x?xf32>
                %30 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%arg8, %29 : tensor<?x?xf32>, tensor<?x?xf32>) outs(%14 : tensor<?x?xf32>) {
                ^bb0(%in: f32, %in_0: f32, %out: f32):
                  %31 = arith.addf %in, %in_0 : f32
                  linalg.yield %31 : f32
                } -> tensor<?x?xf32>
                affine.yield %30 : tensor<?x?xf32>
              } {loom.iter_type = #loom.iter_type<sequential>}
              %17 = arith.muli %arg5, %c32 : index
              %18 = arith.muli %12, %c32 : index
              %19 = loom.view %arg2[%17, %18] [32, 32] [1, 1], reuse : [seq = false, spat = false, temp = false] : memref<128x256xf32> -> !loom.view
              loom.copy_from_tensor %16, %19 to @DRAM : tensor<?x?xf32>, !loom.view
            } {loom.iter_type = #loom.iter_type<temporal>}
          } {loom.iter_type = #loom.iter_type<temporal>}
        } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @y}
      } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @x}
      return
    }
    func.func @matmul__d0i1_d1i1__f01__d_d__BK64__BM32__BN32(%arg0: memref<128x128xf32>, %arg1: memref<128x256xf32>, %arg2: memref<128x256xf32>) {
      %cst = arith.constant 0.000000e+00 : f32
      %c32 = arith.constant 32 : index
      %c64 = arith.constant 64 : index
      affine.parallel (%arg3) = (0) to (8) {
        affine.parallel (%arg4) = (0) to (8) {
          affine.for %arg5 = 0 to 4 {
            affine.for %arg6 = 0 to 1 {
              %12 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 8 + d1 + d2 * 64)>(%arg3, %arg4, %arg6)
              %13 = loom.alloc(%c32, %c32) on @L1 : !loom.buffer_token
              %14 = loom.init_tensor %13(%c32, %c32) : !loom.buffer_token -> tensor<?x?xf32>
              %15 = linalg.fill ins(%cst : f32) outs(%14 : tensor<?x?xf32>) -> tensor<?x?xf32>
              %16 = affine.for %arg7 = 0 to 2 iter_args(%arg8 = %15) -> (tensor<?x?xf32>) {
                %20 = arith.muli %arg5, %c32 : index
                %21 = arith.muli %arg7, %c64 : index
                %22 = loom.view %arg0[%20, %21] [32, 64] [1, 1], reuse : [seq = false, spat = true, temp = true] : memref<128x128xf32> -> !loom.view
                %23 = loom.alloc(%c32, %c64) on @L1 : !loom.buffer_token
                %24 = loom.copy_to_tensor %22, %23 to @L1, interconnect : [], broadcast : [1, 1] : !loom.view, !loom.buffer_token -> tensor<?x?xf32>
                %25 = arith.muli %12, %c32 : index
                %26 = loom.view %arg1[%21, %25] [64, 32] [1, 1], reuse : [seq = false, spat = false, temp = true] : memref<128x256xf32> -> !loom.view
                %27 = loom.alloc(%c64, %c32) on @L1 : !loom.buffer_token
                %28 = loom.copy_to_tensor %26, %27 to @L1, interconnect : [], broadcast : [1, 1] : !loom.view, !loom.buffer_token -> tensor<?x?xf32>
                %29 = linalg.matmul ins(%24, %28 : tensor<?x?xf32>, tensor<?x?xf32>) outs(%15 : tensor<?x?xf32>) -> tensor<?x?xf32>
                %30 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%arg8, %29 : tensor<?x?xf32>, tensor<?x?xf32>) outs(%14 : tensor<?x?xf32>) {
                ^bb0(%in: f32, %in_0: f32, %out: f32):
                  %31 = arith.addf %in, %in_0 : f32
                  linalg.yield %31 : f32
                } -> tensor<?x?xf32>
                affine.yield %30 : tensor<?x?xf32>
              } {loom.iter_type = #loom.iter_type<sequential>}
              %17 = arith.muli %arg5, %c32 : index
              %18 = arith.muli %12, %c32 : index
              %19 = loom.view %arg2[%17, %18] [32, 32] [1, 1], reuse : [seq = false, spat = false, temp = false] : memref<128x256xf32> -> !loom.view
              loom.copy_from_tensor %16, %19 to @DRAM : tensor<?x?xf32>, !loom.view
            } {loom.iter_type = #loom.iter_type<temporal>}
          } {loom.iter_type = #loom.iter_type<temporal>}
        } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @y}
      } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @x}
      return
    }
  }
  module attributes {loom.pass_name = "Materialize"} {
    func.func @matmul__d0i1_d1i1__f01__a_d__BK32__BM32__BN32(%arg0: memref<128x128xf32>, %arg1: memref<128x256xf32>, %arg2: memref<128x256xf32>) {
      %cst = arith.constant 0.000000e+00 : f32
      %c32 = arith.constant 32 : index
      affine.parallel (%arg3) = (0) to (8) {
        affine.parallel (%arg4) = (0) to (8) {
          affine.for %arg5 = 0 to 4 {
            affine.for %arg6 = 0 to 1 {
              %12 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 8 + d1 + d2 * 64)>(%arg3, %arg4, %arg6)
              %13 = loom.alloc(%c32, %c32) on @L1 : !loom.buffer_token
              %14 = loom.init_tensor %13(%c32, %c32) : !loom.buffer_token -> tensor<?x?xf32>
              %15 = linalg.fill ins(%cst : f32) outs(%14 : tensor<?x?xf32>) -> tensor<?x?xf32>
              %16 = affine.for %arg7 = 0 to 4 iter_args(%arg8 = %15) -> (tensor<?x?xf32>) {
                %20 = arith.muli %arg5, %c32 : index
                %21 = arith.muli %arg7, %c32 : index
                %22 = loom.view %arg0[%20, %21] [32, 32] [1, 1], reuse : [seq = false, spat = true, temp = true] : memref<128x128xf32> -> !loom.view
                %23 = loom.alloc(%c32, %c32) on @L1 : !loom.buffer_token
                %24 = loom.copy_to_tensor %22, %23 to @L1, interconnect : [@horizontal_links, @vertical_links], broadcast : [8, 8] : !loom.view, !loom.buffer_token -> tensor<?x?xf32>
                %25 = arith.muli %12, %c32 : index
                %26 = loom.view %arg1[%21, %25] [32, 32] [1, 1], reuse : [seq = false, spat = false, temp = true] : memref<128x256xf32> -> !loom.view
                %27 = loom.alloc(%c32, %c32) on @L1 : !loom.buffer_token
                %28 = loom.copy_to_tensor %26, %27 to @L1, interconnect : [], broadcast : [1, 1] : !loom.view, !loom.buffer_token -> tensor<?x?xf32>
                %29 = linalg.matmul ins(%24, %28 : tensor<?x?xf32>, tensor<?x?xf32>) outs(%15 : tensor<?x?xf32>) -> tensor<?x?xf32>
                %30 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%arg8, %29 : tensor<?x?xf32>, tensor<?x?xf32>) outs(%14 : tensor<?x?xf32>) {
                ^bb0(%in: f32, %in_0: f32, %out: f32):
                  %31 = arith.addf %in, %in_0 : f32
                  linalg.yield %31 : f32
                } -> tensor<?x?xf32>
                affine.yield %30 : tensor<?x?xf32>
              } {loom.iter_type = #loom.iter_type<sequential>}
              %17 = arith.muli %arg5, %c32 : index
              %18 = arith.muli %12, %c32 : index
              %19 = loom.view %arg2[%17, %18] [32, 32] [1, 1], reuse : [seq = false, spat = false, temp = false] : memref<128x256xf32> -> !loom.view
              loom.copy_from_tensor %16, %19 to @DRAM : tensor<?x?xf32>, !loom.view
            } {loom.iter_type = #loom.iter_type<temporal>}
          } {loom.iter_type = #loom.iter_type<temporal>}
        } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @y}
      } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @x}
      return
    }
    func.func @matmul__d0i1_d1i1__f01__a_d__BK64__BM32__BN32(%arg0: memref<128x128xf32>, %arg1: memref<128x256xf32>, %arg2: memref<128x256xf32>) {
      %cst = arith.constant 0.000000e+00 : f32
      %c32 = arith.constant 32 : index
      %c64 = arith.constant 64 : index
      affine.parallel (%arg3) = (0) to (8) {
        affine.parallel (%arg4) = (0) to (8) {
          affine.for %arg5 = 0 to 4 {
            affine.for %arg6 = 0 to 1 {
              %12 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 8 + d1 + d2 * 64)>(%arg3, %arg4, %arg6)
              %13 = loom.alloc(%c32, %c32) on @L1 : !loom.buffer_token
              %14 = loom.init_tensor %13(%c32, %c32) : !loom.buffer_token -> tensor<?x?xf32>
              %15 = linalg.fill ins(%cst : f32) outs(%14 : tensor<?x?xf32>) -> tensor<?x?xf32>
              %16 = affine.for %arg7 = 0 to 2 iter_args(%arg8 = %15) -> (tensor<?x?xf32>) {
                %20 = arith.muli %arg5, %c32 : index
                %21 = arith.muli %arg7, %c64 : index
                %22 = loom.view %arg0[%20, %21] [32, 64] [1, 1], reuse : [seq = false, spat = true, temp = true] : memref<128x128xf32> -> !loom.view
                %23 = loom.alloc(%c32, %c64) on @L1 : !loom.buffer_token
                %24 = loom.copy_to_tensor %22, %23 to @L1, interconnect : [@horizontal_links, @vertical_links], broadcast : [8, 8] : !loom.view, !loom.buffer_token -> tensor<?x?xf32>
                %25 = arith.muli %12, %c32 : index
                %26 = loom.view %arg1[%21, %25] [64, 32] [1, 1], reuse : [seq = false, spat = false, temp = true] : memref<128x256xf32> -> !loom.view
                %27 = loom.alloc(%c64, %c32) on @L1 : !loom.buffer_token
                %28 = loom.copy_to_tensor %26, %27 to @L1, interconnect : [], broadcast : [1, 1] : !loom.view, !loom.buffer_token -> tensor<?x?xf32>
                %29 = linalg.matmul ins(%24, %28 : tensor<?x?xf32>, tensor<?x?xf32>) outs(%15 : tensor<?x?xf32>) -> tensor<?x?xf32>
                %30 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%arg8, %29 : tensor<?x?xf32>, tensor<?x?xf32>) outs(%14 : tensor<?x?xf32>) {
                ^bb0(%in: f32, %in_0: f32, %out: f32):
                  %31 = arith.addf %in, %in_0 : f32
                  linalg.yield %31 : f32
                } -> tensor<?x?xf32>
                affine.yield %30 : tensor<?x?xf32>
              } {loom.iter_type = #loom.iter_type<sequential>}
              %17 = arith.muli %arg5, %c32 : index
              %18 = arith.muli %12, %c32 : index
              %19 = loom.view %arg2[%17, %18] [32, 32] [1, 1], reuse : [seq = false, spat = false, temp = false] : memref<128x256xf32> -> !loom.view
              loom.copy_from_tensor %16, %19 to @DRAM : tensor<?x?xf32>, !loom.view
            } {loom.iter_type = #loom.iter_type<temporal>}
          } {loom.iter_type = #loom.iter_type<temporal>}
        } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @y}
      } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @x}
      return
    }
  }
  module attributes {loom.pass_name = "Materialize"} {
    func.func @matmul__d0i1_d1i1__f01__h_d__BK32__BM32__BN32(%arg0: memref<128x128xf32>, %arg1: memref<128x256xf32>, %arg2: memref<128x256xf32>) {
      %cst = arith.constant 0.000000e+00 : f32
      %c32 = arith.constant 32 : index
      affine.parallel (%arg3) = (0) to (8) {
        affine.parallel (%arg4) = (0) to (8) {
          affine.for %arg5 = 0 to 4 {
            affine.for %arg6 = 0 to 1 {
              %12 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 8 + d1 + d2 * 64)>(%arg3, %arg4, %arg6)
              %13 = loom.alloc(%c32, %c32) on @L1 : !loom.buffer_token
              %14 = loom.init_tensor %13(%c32, %c32) : !loom.buffer_token -> tensor<?x?xf32>
              %15 = linalg.fill ins(%cst : f32) outs(%14 : tensor<?x?xf32>) -> tensor<?x?xf32>
              %16 = affine.for %arg7 = 0 to 4 iter_args(%arg8 = %15) -> (tensor<?x?xf32>) {
                %20 = arith.muli %arg5, %c32 : index
                %21 = arith.muli %arg7, %c32 : index
                %22 = loom.view %arg0[%20, %21] [32, 32] [1, 1], reuse : [seq = false, spat = true, temp = true] : memref<128x128xf32> -> !loom.view
                %23 = loom.alloc(%c32, %c32) on @L1 : !loom.buffer_token
                %24 = loom.copy_to_tensor %22, %23 to @L1, interconnect : [@horizontal_links], broadcast : [1, 8] : !loom.view, !loom.buffer_token -> tensor<?x?xf32>
                %25 = arith.muli %12, %c32 : index
                %26 = loom.view %arg1[%21, %25] [32, 32] [1, 1], reuse : [seq = false, spat = false, temp = true] : memref<128x256xf32> -> !loom.view
                %27 = loom.alloc(%c32, %c32) on @L1 : !loom.buffer_token
                %28 = loom.copy_to_tensor %26, %27 to @L1, interconnect : [], broadcast : [1, 1] : !loom.view, !loom.buffer_token -> tensor<?x?xf32>
                %29 = linalg.matmul ins(%24, %28 : tensor<?x?xf32>, tensor<?x?xf32>) outs(%15 : tensor<?x?xf32>) -> tensor<?x?xf32>
                %30 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%arg8, %29 : tensor<?x?xf32>, tensor<?x?xf32>) outs(%14 : tensor<?x?xf32>) {
                ^bb0(%in: f32, %in_0: f32, %out: f32):
                  %31 = arith.addf %in, %in_0 : f32
                  linalg.yield %31 : f32
                } -> tensor<?x?xf32>
                affine.yield %30 : tensor<?x?xf32>
              } {loom.iter_type = #loom.iter_type<sequential>}
              %17 = arith.muli %arg5, %c32 : index
              %18 = arith.muli %12, %c32 : index
              %19 = loom.view %arg2[%17, %18] [32, 32] [1, 1], reuse : [seq = false, spat = false, temp = false] : memref<128x256xf32> -> !loom.view
              loom.copy_from_tensor %16, %19 to @DRAM : tensor<?x?xf32>, !loom.view
            } {loom.iter_type = #loom.iter_type<temporal>}
          } {loom.iter_type = #loom.iter_type<temporal>}
        } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @y}
      } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @x}
      return
    }
    func.func @matmul__d0i1_d1i1__f01__h_d__BK64__BM32__BN32(%arg0: memref<128x128xf32>, %arg1: memref<128x256xf32>, %arg2: memref<128x256xf32>) {
      %cst = arith.constant 0.000000e+00 : f32
      %c32 = arith.constant 32 : index
      %c64 = arith.constant 64 : index
      affine.parallel (%arg3) = (0) to (8) {
        affine.parallel (%arg4) = (0) to (8) {
          affine.for %arg5 = 0 to 4 {
            affine.for %arg6 = 0 to 1 {
              %12 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 8 + d1 + d2 * 64)>(%arg3, %arg4, %arg6)
              %13 = loom.alloc(%c32, %c32) on @L1 : !loom.buffer_token
              %14 = loom.init_tensor %13(%c32, %c32) : !loom.buffer_token -> tensor<?x?xf32>
              %15 = linalg.fill ins(%cst : f32) outs(%14 : tensor<?x?xf32>) -> tensor<?x?xf32>
              %16 = affine.for %arg7 = 0 to 2 iter_args(%arg8 = %15) -> (tensor<?x?xf32>) {
                %20 = arith.muli %arg5, %c32 : index
                %21 = arith.muli %arg7, %c64 : index
                %22 = loom.view %arg0[%20, %21] [32, 64] [1, 1], reuse : [seq = false, spat = true, temp = true] : memref<128x128xf32> -> !loom.view
                %23 = loom.alloc(%c32, %c64) on @L1 : !loom.buffer_token
                %24 = loom.copy_to_tensor %22, %23 to @L1, interconnect : [@horizontal_links], broadcast : [1, 8] : !loom.view, !loom.buffer_token -> tensor<?x?xf32>
                %25 = arith.muli %12, %c32 : index
                %26 = loom.view %arg1[%21, %25] [64, 32] [1, 1], reuse : [seq = false, spat = false, temp = true] : memref<128x256xf32> -> !loom.view
                %27 = loom.alloc(%c64, %c32) on @L1 : !loom.buffer_token
                %28 = loom.copy_to_tensor %26, %27 to @L1, interconnect : [], broadcast : [1, 1] : !loom.view, !loom.buffer_token -> tensor<?x?xf32>
                %29 = linalg.matmul ins(%24, %28 : tensor<?x?xf32>, tensor<?x?xf32>) outs(%15 : tensor<?x?xf32>) -> tensor<?x?xf32>
                %30 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%arg8, %29 : tensor<?x?xf32>, tensor<?x?xf32>) outs(%14 : tensor<?x?xf32>) {
                ^bb0(%in: f32, %in_0: f32, %out: f32):
                  %31 = arith.addf %in, %in_0 : f32
                  linalg.yield %31 : f32
                } -> tensor<?x?xf32>
                affine.yield %30 : tensor<?x?xf32>
              } {loom.iter_type = #loom.iter_type<sequential>}
              %17 = arith.muli %arg5, %c32 : index
              %18 = arith.muli %12, %c32 : index
              %19 = loom.view %arg2[%17, %18] [32, 32] [1, 1], reuse : [seq = false, spat = false, temp = false] : memref<128x256xf32> -> !loom.view
              loom.copy_from_tensor %16, %19 to @DRAM : tensor<?x?xf32>, !loom.view
            } {loom.iter_type = #loom.iter_type<temporal>}
          } {loom.iter_type = #loom.iter_type<temporal>}
        } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @y}
      } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @x}
      return
    }
  }
  module attributes {loom.pass_name = "Materialize"} {
    func.func @matmul__d0i1_d1i1__f01__v_d__BK32__BM32__BN32(%arg0: memref<128x128xf32>, %arg1: memref<128x256xf32>, %arg2: memref<128x256xf32>) {
      %cst = arith.constant 0.000000e+00 : f32
      %c32 = arith.constant 32 : index
      affine.parallel (%arg3) = (0) to (8) {
        affine.parallel (%arg4) = (0) to (8) {
          affine.for %arg5 = 0 to 4 {
            affine.for %arg6 = 0 to 1 {
              %12 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 8 + d1 + d2 * 64)>(%arg3, %arg4, %arg6)
              %13 = loom.alloc(%c32, %c32) on @L1 : !loom.buffer_token
              %14 = loom.init_tensor %13(%c32, %c32) : !loom.buffer_token -> tensor<?x?xf32>
              %15 = linalg.fill ins(%cst : f32) outs(%14 : tensor<?x?xf32>) -> tensor<?x?xf32>
              %16 = affine.for %arg7 = 0 to 4 iter_args(%arg8 = %15) -> (tensor<?x?xf32>) {
                %20 = arith.muli %arg5, %c32 : index
                %21 = arith.muli %arg7, %c32 : index
                %22 = loom.view %arg0[%20, %21] [32, 32] [1, 1], reuse : [seq = false, spat = true, temp = true] : memref<128x128xf32> -> !loom.view
                %23 = loom.alloc(%c32, %c32) on @L1 : !loom.buffer_token
                %24 = loom.copy_to_tensor %22, %23 to @L1, interconnect : [@vertical_links], broadcast : [8, 1] : !loom.view, !loom.buffer_token -> tensor<?x?xf32>
                %25 = arith.muli %12, %c32 : index
                %26 = loom.view %arg1[%21, %25] [32, 32] [1, 1], reuse : [seq = false, spat = false, temp = true] : memref<128x256xf32> -> !loom.view
                %27 = loom.alloc(%c32, %c32) on @L1 : !loom.buffer_token
                %28 = loom.copy_to_tensor %26, %27 to @L1, interconnect : [], broadcast : [1, 1] : !loom.view, !loom.buffer_token -> tensor<?x?xf32>
                %29 = linalg.matmul ins(%24, %28 : tensor<?x?xf32>, tensor<?x?xf32>) outs(%15 : tensor<?x?xf32>) -> tensor<?x?xf32>
                %30 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%arg8, %29 : tensor<?x?xf32>, tensor<?x?xf32>) outs(%14 : tensor<?x?xf32>) {
                ^bb0(%in: f32, %in_0: f32, %out: f32):
                  %31 = arith.addf %in, %in_0 : f32
                  linalg.yield %31 : f32
                } -> tensor<?x?xf32>
                affine.yield %30 : tensor<?x?xf32>
              } {loom.iter_type = #loom.iter_type<sequential>}
              %17 = arith.muli %arg5, %c32 : index
              %18 = arith.muli %12, %c32 : index
              %19 = loom.view %arg2[%17, %18] [32, 32] [1, 1], reuse : [seq = false, spat = false, temp = false] : memref<128x256xf32> -> !loom.view
              loom.copy_from_tensor %16, %19 to @DRAM : tensor<?x?xf32>, !loom.view
            } {loom.iter_type = #loom.iter_type<temporal>}
          } {loom.iter_type = #loom.iter_type<temporal>}
        } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @y}
      } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @x}
      return
    }
    func.func @matmul__d0i1_d1i1__f01__v_d__BK64__BM32__BN32(%arg0: memref<128x128xf32>, %arg1: memref<128x256xf32>, %arg2: memref<128x256xf32>) {
      %cst = arith.constant 0.000000e+00 : f32
      %c32 = arith.constant 32 : index
      %c64 = arith.constant 64 : index
      affine.parallel (%arg3) = (0) to (8) {
        affine.parallel (%arg4) = (0) to (8) {
          affine.for %arg5 = 0 to 4 {
            affine.for %arg6 = 0 to 1 {
              %12 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 8 + d1 + d2 * 64)>(%arg3, %arg4, %arg6)
              %13 = loom.alloc(%c32, %c32) on @L1 : !loom.buffer_token
              %14 = loom.init_tensor %13(%c32, %c32) : !loom.buffer_token -> tensor<?x?xf32>
              %15 = linalg.fill ins(%cst : f32) outs(%14 : tensor<?x?xf32>) -> tensor<?x?xf32>
              %16 = affine.for %arg7 = 0 to 2 iter_args(%arg8 = %15) -> (tensor<?x?xf32>) {
                %20 = arith.muli %arg5, %c32 : index
                %21 = arith.muli %arg7, %c64 : index
                %22 = loom.view %arg0[%20, %21] [32, 64] [1, 1], reuse : [seq = false, spat = true, temp = true] : memref<128x128xf32> -> !loom.view
                %23 = loom.alloc(%c32, %c64) on @L1 : !loom.buffer_token
                %24 = loom.copy_to_tensor %22, %23 to @L1, interconnect : [@vertical_links], broadcast : [8, 1] : !loom.view, !loom.buffer_token -> tensor<?x?xf32>
                %25 = arith.muli %12, %c32 : index
                %26 = loom.view %arg1[%21, %25] [64, 32] [1, 1], reuse : [seq = false, spat = false, temp = true] : memref<128x256xf32> -> !loom.view
                %27 = loom.alloc(%c64, %c32) on @L1 : !loom.buffer_token
                %28 = loom.copy_to_tensor %26, %27 to @L1, interconnect : [], broadcast : [1, 1] : !loom.view, !loom.buffer_token -> tensor<?x?xf32>
                %29 = linalg.matmul ins(%24, %28 : tensor<?x?xf32>, tensor<?x?xf32>) outs(%15 : tensor<?x?xf32>) -> tensor<?x?xf32>
                %30 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%arg8, %29 : tensor<?x?xf32>, tensor<?x?xf32>) outs(%14 : tensor<?x?xf32>) {
                ^bb0(%in: f32, %in_0: f32, %out: f32):
                  %31 = arith.addf %in, %in_0 : f32
                  linalg.yield %31 : f32
                } -> tensor<?x?xf32>
                affine.yield %30 : tensor<?x?xf32>
              } {loom.iter_type = #loom.iter_type<sequential>}
              %17 = arith.muli %arg5, %c32 : index
              %18 = arith.muli %12, %c32 : index
              %19 = loom.view %arg2[%17, %18] [32, 32] [1, 1], reuse : [seq = false, spat = false, temp = false] : memref<128x256xf32> -> !loom.view
              loom.copy_from_tensor %16, %19 to @DRAM : tensor<?x?xf32>, !loom.view
            } {loom.iter_type = #loom.iter_type<temporal>}
          } {loom.iter_type = #loom.iter_type<temporal>}
        } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @y}
      } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @x}
      return
    }
  }
  module attributes {loom.pass_name = "Materialize"} {
    func.func @matmul__d0i1_d1i1__f10__d_d__BK32__BM32__BN32(%arg0: memref<128x128xf32>, %arg1: memref<128x256xf32>, %arg2: memref<128x256xf32>) {
      %cst = arith.constant 0.000000e+00 : f32
      %c32 = arith.constant 32 : index
      affine.parallel (%arg3) = (0) to (8) {
        affine.parallel (%arg4) = (0) to (8) {
          affine.for %arg5 = 0 to 1 {
            affine.for %arg6 = 0 to 4 {
              %12 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 8 + d1 + d2 * 64)>(%arg3, %arg4, %arg5)
              %13 = loom.alloc(%c32, %c32) on @L1 : !loom.buffer_token
              %14 = loom.init_tensor %13(%c32, %c32) : !loom.buffer_token -> tensor<?x?xf32>
              %15 = linalg.fill ins(%cst : f32) outs(%14 : tensor<?x?xf32>) -> tensor<?x?xf32>
              %16 = affine.for %arg7 = 0 to 4 iter_args(%arg8 = %15) -> (tensor<?x?xf32>) {
                %20 = arith.muli %arg6, %c32 : index
                %21 = arith.muli %arg7, %c32 : index
                %22 = loom.view %arg0[%20, %21] [32, 32] [1, 1], reuse : [seq = false, spat = true, temp = true] : memref<128x128xf32> -> !loom.view
                %23 = loom.alloc(%c32, %c32) on @L1 : !loom.buffer_token
                %24 = loom.copy_to_tensor %22, %23 to @L1, interconnect : [], broadcast : [1, 1] : !loom.view, !loom.buffer_token -> tensor<?x?xf32>
                %25 = arith.muli %12, %c32 : index
                %26 = loom.view %arg1[%21, %25] [32, 32] [1, 1], reuse : [seq = false, spat = false, temp = true] : memref<128x256xf32> -> !loom.view
                %27 = loom.alloc(%c32, %c32) on @L1 : !loom.buffer_token
                %28 = loom.copy_to_tensor %26, %27 to @L1, interconnect : [], broadcast : [1, 1] : !loom.view, !loom.buffer_token -> tensor<?x?xf32>
                %29 = linalg.matmul ins(%24, %28 : tensor<?x?xf32>, tensor<?x?xf32>) outs(%15 : tensor<?x?xf32>) -> tensor<?x?xf32>
                %30 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%arg8, %29 : tensor<?x?xf32>, tensor<?x?xf32>) outs(%14 : tensor<?x?xf32>) {
                ^bb0(%in: f32, %in_0: f32, %out: f32):
                  %31 = arith.addf %in, %in_0 : f32
                  linalg.yield %31 : f32
                } -> tensor<?x?xf32>
                affine.yield %30 : tensor<?x?xf32>
              } {loom.iter_type = #loom.iter_type<sequential>}
              %17 = arith.muli %arg6, %c32 : index
              %18 = arith.muli %12, %c32 : index
              %19 = loom.view %arg2[%17, %18] [32, 32] [1, 1], reuse : [seq = false, spat = false, temp = false] : memref<128x256xf32> -> !loom.view
              loom.copy_from_tensor %16, %19 to @DRAM : tensor<?x?xf32>, !loom.view
            } {loom.iter_type = #loom.iter_type<temporal>}
          } {loom.iter_type = #loom.iter_type<temporal>}
        } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @y}
      } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @x}
      return
    }
    func.func @matmul__d0i1_d1i1__f10__d_d__BK64__BM32__BN32(%arg0: memref<128x128xf32>, %arg1: memref<128x256xf32>, %arg2: memref<128x256xf32>) {
      %cst = arith.constant 0.000000e+00 : f32
      %c32 = arith.constant 32 : index
      %c64 = arith.constant 64 : index
      affine.parallel (%arg3) = (0) to (8) {
        affine.parallel (%arg4) = (0) to (8) {
          affine.for %arg5 = 0 to 1 {
            affine.for %arg6 = 0 to 4 {
              %12 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 8 + d1 + d2 * 64)>(%arg3, %arg4, %arg5)
              %13 = loom.alloc(%c32, %c32) on @L1 : !loom.buffer_token
              %14 = loom.init_tensor %13(%c32, %c32) : !loom.buffer_token -> tensor<?x?xf32>
              %15 = linalg.fill ins(%cst : f32) outs(%14 : tensor<?x?xf32>) -> tensor<?x?xf32>
              %16 = affine.for %arg7 = 0 to 2 iter_args(%arg8 = %15) -> (tensor<?x?xf32>) {
                %20 = arith.muli %arg6, %c32 : index
                %21 = arith.muli %arg7, %c64 : index
                %22 = loom.view %arg0[%20, %21] [32, 64] [1, 1], reuse : [seq = false, spat = true, temp = true] : memref<128x128xf32> -> !loom.view
                %23 = loom.alloc(%c32, %c64) on @L1 : !loom.buffer_token
                %24 = loom.copy_to_tensor %22, %23 to @L1, interconnect : [], broadcast : [1, 1] : !loom.view, !loom.buffer_token -> tensor<?x?xf32>
                %25 = arith.muli %12, %c32 : index
                %26 = loom.view %arg1[%21, %25] [64, 32] [1, 1], reuse : [seq = false, spat = false, temp = true] : memref<128x256xf32> -> !loom.view
                %27 = loom.alloc(%c64, %c32) on @L1 : !loom.buffer_token
                %28 = loom.copy_to_tensor %26, %27 to @L1, interconnect : [], broadcast : [1, 1] : !loom.view, !loom.buffer_token -> tensor<?x?xf32>
                %29 = linalg.matmul ins(%24, %28 : tensor<?x?xf32>, tensor<?x?xf32>) outs(%15 : tensor<?x?xf32>) -> tensor<?x?xf32>
                %30 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%arg8, %29 : tensor<?x?xf32>, tensor<?x?xf32>) outs(%14 : tensor<?x?xf32>) {
                ^bb0(%in: f32, %in_0: f32, %out: f32):
                  %31 = arith.addf %in, %in_0 : f32
                  linalg.yield %31 : f32
                } -> tensor<?x?xf32>
                affine.yield %30 : tensor<?x?xf32>
              } {loom.iter_type = #loom.iter_type<sequential>}
              %17 = arith.muli %arg6, %c32 : index
              %18 = arith.muli %12, %c32 : index
              %19 = loom.view %arg2[%17, %18] [32, 32] [1, 1], reuse : [seq = false, spat = false, temp = false] : memref<128x256xf32> -> !loom.view
              loom.copy_from_tensor %16, %19 to @DRAM : tensor<?x?xf32>, !loom.view
            } {loom.iter_type = #loom.iter_type<temporal>}
          } {loom.iter_type = #loom.iter_type<temporal>}
        } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @y}
      } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @x}
      return
    }
  }
  module attributes {loom.pass_name = "Materialize"} {
    func.func @matmul__d0i1_d1i1__f10__a_d__BK32__BM32__BN32(%arg0: memref<128x128xf32>, %arg1: memref<128x256xf32>, %arg2: memref<128x256xf32>) {
      %cst = arith.constant 0.000000e+00 : f32
      %c32 = arith.constant 32 : index
      affine.parallel (%arg3) = (0) to (8) {
        affine.parallel (%arg4) = (0) to (8) {
          affine.for %arg5 = 0 to 1 {
            affine.for %arg6 = 0 to 4 {
              %12 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 8 + d1 + d2 * 64)>(%arg3, %arg4, %arg5)
              %13 = loom.alloc(%c32, %c32) on @L1 : !loom.buffer_token
              %14 = loom.init_tensor %13(%c32, %c32) : !loom.buffer_token -> tensor<?x?xf32>
              %15 = linalg.fill ins(%cst : f32) outs(%14 : tensor<?x?xf32>) -> tensor<?x?xf32>
              %16 = affine.for %arg7 = 0 to 4 iter_args(%arg8 = %15) -> (tensor<?x?xf32>) {
                %20 = arith.muli %arg6, %c32 : index
                %21 = arith.muli %arg7, %c32 : index
                %22 = loom.view %arg0[%20, %21] [32, 32] [1, 1], reuse : [seq = false, spat = true, temp = true] : memref<128x128xf32> -> !loom.view
                %23 = loom.alloc(%c32, %c32) on @L1 : !loom.buffer_token
                %24 = loom.copy_to_tensor %22, %23 to @L1, interconnect : [@horizontal_links, @vertical_links], broadcast : [8, 8] : !loom.view, !loom.buffer_token -> tensor<?x?xf32>
                %25 = arith.muli %12, %c32 : index
                %26 = loom.view %arg1[%21, %25] [32, 32] [1, 1], reuse : [seq = false, spat = false, temp = true] : memref<128x256xf32> -> !loom.view
                %27 = loom.alloc(%c32, %c32) on @L1 : !loom.buffer_token
                %28 = loom.copy_to_tensor %26, %27 to @L1, interconnect : [], broadcast : [1, 1] : !loom.view, !loom.buffer_token -> tensor<?x?xf32>
                %29 = linalg.matmul ins(%24, %28 : tensor<?x?xf32>, tensor<?x?xf32>) outs(%15 : tensor<?x?xf32>) -> tensor<?x?xf32>
                %30 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%arg8, %29 : tensor<?x?xf32>, tensor<?x?xf32>) outs(%14 : tensor<?x?xf32>) {
                ^bb0(%in: f32, %in_0: f32, %out: f32):
                  %31 = arith.addf %in, %in_0 : f32
                  linalg.yield %31 : f32
                } -> tensor<?x?xf32>
                affine.yield %30 : tensor<?x?xf32>
              } {loom.iter_type = #loom.iter_type<sequential>}
              %17 = arith.muli %arg6, %c32 : index
              %18 = arith.muli %12, %c32 : index
              %19 = loom.view %arg2[%17, %18] [32, 32] [1, 1], reuse : [seq = false, spat = false, temp = false] : memref<128x256xf32> -> !loom.view
              loom.copy_from_tensor %16, %19 to @DRAM : tensor<?x?xf32>, !loom.view
            } {loom.iter_type = #loom.iter_type<temporal>}
          } {loom.iter_type = #loom.iter_type<temporal>}
        } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @y}
      } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @x}
      return
    }
    func.func @matmul__d0i1_d1i1__f10__a_d__BK64__BM32__BN32(%arg0: memref<128x128xf32>, %arg1: memref<128x256xf32>, %arg2: memref<128x256xf32>) {
      %cst = arith.constant 0.000000e+00 : f32
      %c32 = arith.constant 32 : index
      %c64 = arith.constant 64 : index
      affine.parallel (%arg3) = (0) to (8) {
        affine.parallel (%arg4) = (0) to (8) {
          affine.for %arg5 = 0 to 1 {
            affine.for %arg6 = 0 to 4 {
              %12 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 8 + d1 + d2 * 64)>(%arg3, %arg4, %arg5)
              %13 = loom.alloc(%c32, %c32) on @L1 : !loom.buffer_token
              %14 = loom.init_tensor %13(%c32, %c32) : !loom.buffer_token -> tensor<?x?xf32>
              %15 = linalg.fill ins(%cst : f32) outs(%14 : tensor<?x?xf32>) -> tensor<?x?xf32>
              %16 = affine.for %arg7 = 0 to 2 iter_args(%arg8 = %15) -> (tensor<?x?xf32>) {
                %20 = arith.muli %arg6, %c32 : index
                %21 = arith.muli %arg7, %c64 : index
                %22 = loom.view %arg0[%20, %21] [32, 64] [1, 1], reuse : [seq = false, spat = true, temp = true] : memref<128x128xf32> -> !loom.view
                %23 = loom.alloc(%c32, %c64) on @L1 : !loom.buffer_token
                %24 = loom.copy_to_tensor %22, %23 to @L1, interconnect : [@horizontal_links, @vertical_links], broadcast : [8, 8] : !loom.view, !loom.buffer_token -> tensor<?x?xf32>
                %25 = arith.muli %12, %c32 : index
                %26 = loom.view %arg1[%21, %25] [64, 32] [1, 1], reuse : [seq = false, spat = false, temp = true] : memref<128x256xf32> -> !loom.view
                %27 = loom.alloc(%c64, %c32) on @L1 : !loom.buffer_token
                %28 = loom.copy_to_tensor %26, %27 to @L1, interconnect : [], broadcast : [1, 1] : !loom.view, !loom.buffer_token -> tensor<?x?xf32>
                %29 = linalg.matmul ins(%24, %28 : tensor<?x?xf32>, tensor<?x?xf32>) outs(%15 : tensor<?x?xf32>) -> tensor<?x?xf32>
                %30 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%arg8, %29 : tensor<?x?xf32>, tensor<?x?xf32>) outs(%14 : tensor<?x?xf32>) {
                ^bb0(%in: f32, %in_0: f32, %out: f32):
                  %31 = arith.addf %in, %in_0 : f32
                  linalg.yield %31 : f32
                } -> tensor<?x?xf32>
                affine.yield %30 : tensor<?x?xf32>
              } {loom.iter_type = #loom.iter_type<sequential>}
              %17 = arith.muli %arg6, %c32 : index
              %18 = arith.muli %12, %c32 : index
              %19 = loom.view %arg2[%17, %18] [32, 32] [1, 1], reuse : [seq = false, spat = false, temp = false] : memref<128x256xf32> -> !loom.view
              loom.copy_from_tensor %16, %19 to @DRAM : tensor<?x?xf32>, !loom.view
            } {loom.iter_type = #loom.iter_type<temporal>}
          } {loom.iter_type = #loom.iter_type<temporal>}
        } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @y}
      } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @x}
      return
    }
  }
  module attributes {loom.pass_name = "Materialize"} {
    func.func @matmul__d0i1_d1i1__f10__h_d__BK32__BM32__BN32(%arg0: memref<128x128xf32>, %arg1: memref<128x256xf32>, %arg2: memref<128x256xf32>) {
      %cst = arith.constant 0.000000e+00 : f32
      %c32 = arith.constant 32 : index
      affine.parallel (%arg3) = (0) to (8) {
        affine.parallel (%arg4) = (0) to (8) {
          affine.for %arg5 = 0 to 1 {
            affine.for %arg6 = 0 to 4 {
              %12 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 8 + d1 + d2 * 64)>(%arg3, %arg4, %arg5)
              %13 = loom.alloc(%c32, %c32) on @L1 : !loom.buffer_token
              %14 = loom.init_tensor %13(%c32, %c32) : !loom.buffer_token -> tensor<?x?xf32>
              %15 = linalg.fill ins(%cst : f32) outs(%14 : tensor<?x?xf32>) -> tensor<?x?xf32>
              %16 = affine.for %arg7 = 0 to 4 iter_args(%arg8 = %15) -> (tensor<?x?xf32>) {
                %20 = arith.muli %arg6, %c32 : index
                %21 = arith.muli %arg7, %c32 : index
                %22 = loom.view %arg0[%20, %21] [32, 32] [1, 1], reuse : [seq = false, spat = true, temp = true] : memref<128x128xf32> -> !loom.view
                %23 = loom.alloc(%c32, %c32) on @L1 : !loom.buffer_token
                %24 = loom.copy_to_tensor %22, %23 to @L1, interconnect : [@horizontal_links], broadcast : [1, 8] : !loom.view, !loom.buffer_token -> tensor<?x?xf32>
                %25 = arith.muli %12, %c32 : index
                %26 = loom.view %arg1[%21, %25] [32, 32] [1, 1], reuse : [seq = false, spat = false, temp = true] : memref<128x256xf32> -> !loom.view
                %27 = loom.alloc(%c32, %c32) on @L1 : !loom.buffer_token
                %28 = loom.copy_to_tensor %26, %27 to @L1, interconnect : [], broadcast : [1, 1] : !loom.view, !loom.buffer_token -> tensor<?x?xf32>
                %29 = linalg.matmul ins(%24, %28 : tensor<?x?xf32>, tensor<?x?xf32>) outs(%15 : tensor<?x?xf32>) -> tensor<?x?xf32>
                %30 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%arg8, %29 : tensor<?x?xf32>, tensor<?x?xf32>) outs(%14 : tensor<?x?xf32>) {
                ^bb0(%in: f32, %in_0: f32, %out: f32):
                  %31 = arith.addf %in, %in_0 : f32
                  linalg.yield %31 : f32
                } -> tensor<?x?xf32>
                affine.yield %30 : tensor<?x?xf32>
              } {loom.iter_type = #loom.iter_type<sequential>}
              %17 = arith.muli %arg6, %c32 : index
              %18 = arith.muli %12, %c32 : index
              %19 = loom.view %arg2[%17, %18] [32, 32] [1, 1], reuse : [seq = false, spat = false, temp = false] : memref<128x256xf32> -> !loom.view
              loom.copy_from_tensor %16, %19 to @DRAM : tensor<?x?xf32>, !loom.view
            } {loom.iter_type = #loom.iter_type<temporal>}
          } {loom.iter_type = #loom.iter_type<temporal>}
        } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @y}
      } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @x}
      return
    }
    func.func @matmul__d0i1_d1i1__f10__h_d__BK64__BM32__BN32(%arg0: memref<128x128xf32>, %arg1: memref<128x256xf32>, %arg2: memref<128x256xf32>) {
      %cst = arith.constant 0.000000e+00 : f32
      %c32 = arith.constant 32 : index
      %c64 = arith.constant 64 : index
      affine.parallel (%arg3) = (0) to (8) {
        affine.parallel (%arg4) = (0) to (8) {
          affine.for %arg5 = 0 to 1 {
            affine.for %arg6 = 0 to 4 {
              %12 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 8 + d1 + d2 * 64)>(%arg3, %arg4, %arg5)
              %13 = loom.alloc(%c32, %c32) on @L1 : !loom.buffer_token
              %14 = loom.init_tensor %13(%c32, %c32) : !loom.buffer_token -> tensor<?x?xf32>
              %15 = linalg.fill ins(%cst : f32) outs(%14 : tensor<?x?xf32>) -> tensor<?x?xf32>
              %16 = affine.for %arg7 = 0 to 2 iter_args(%arg8 = %15) -> (tensor<?x?xf32>) {
                %20 = arith.muli %arg6, %c32 : index
                %21 = arith.muli %arg7, %c64 : index
                %22 = loom.view %arg0[%20, %21] [32, 64] [1, 1], reuse : [seq = false, spat = true, temp = true] : memref<128x128xf32> -> !loom.view
                %23 = loom.alloc(%c32, %c64) on @L1 : !loom.buffer_token
                %24 = loom.copy_to_tensor %22, %23 to @L1, interconnect : [@horizontal_links], broadcast : [1, 8] : !loom.view, !loom.buffer_token -> tensor<?x?xf32>
                %25 = arith.muli %12, %c32 : index
                %26 = loom.view %arg1[%21, %25] [64, 32] [1, 1], reuse : [seq = false, spat = false, temp = true] : memref<128x256xf32> -> !loom.view
                %27 = loom.alloc(%c64, %c32) on @L1 : !loom.buffer_token
                %28 = loom.copy_to_tensor %26, %27 to @L1, interconnect : [], broadcast : [1, 1] : !loom.view, !loom.buffer_token -> tensor<?x?xf32>
                %29 = linalg.matmul ins(%24, %28 : tensor<?x?xf32>, tensor<?x?xf32>) outs(%15 : tensor<?x?xf32>) -> tensor<?x?xf32>
                %30 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%arg8, %29 : tensor<?x?xf32>, tensor<?x?xf32>) outs(%14 : tensor<?x?xf32>) {
                ^bb0(%in: f32, %in_0: f32, %out: f32):
                  %31 = arith.addf %in, %in_0 : f32
                  linalg.yield %31 : f32
                } -> tensor<?x?xf32>
                affine.yield %30 : tensor<?x?xf32>
              } {loom.iter_type = #loom.iter_type<sequential>}
              %17 = arith.muli %arg6, %c32 : index
              %18 = arith.muli %12, %c32 : index
              %19 = loom.view %arg2[%17, %18] [32, 32] [1, 1], reuse : [seq = false, spat = false, temp = false] : memref<128x256xf32> -> !loom.view
              loom.copy_from_tensor %16, %19 to @DRAM : tensor<?x?xf32>, !loom.view
            } {loom.iter_type = #loom.iter_type<temporal>}
          } {loom.iter_type = #loom.iter_type<temporal>}
        } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @y}
      } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @x}
      return
    }
  }
  module attributes {loom.pass_name = "Materialize"} {
    func.func @matmul__d0i1_d1i1__f10__v_d__BK32__BM32__BN32(%arg0: memref<128x128xf32>, %arg1: memref<128x256xf32>, %arg2: memref<128x256xf32>) {
      %cst = arith.constant 0.000000e+00 : f32
      %c32 = arith.constant 32 : index
      affine.parallel (%arg3) = (0) to (8) {
        affine.parallel (%arg4) = (0) to (8) {
          affine.for %arg5 = 0 to 1 {
            affine.for %arg6 = 0 to 4 {
              %12 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 8 + d1 + d2 * 64)>(%arg3, %arg4, %arg5)
              %13 = loom.alloc(%c32, %c32) on @L1 : !loom.buffer_token
              %14 = loom.init_tensor %13(%c32, %c32) : !loom.buffer_token -> tensor<?x?xf32>
              %15 = linalg.fill ins(%cst : f32) outs(%14 : tensor<?x?xf32>) -> tensor<?x?xf32>
              %16 = affine.for %arg7 = 0 to 4 iter_args(%arg8 = %15) -> (tensor<?x?xf32>) {
                %20 = arith.muli %arg6, %c32 : index
                %21 = arith.muli %arg7, %c32 : index
                %22 = loom.view %arg0[%20, %21] [32, 32] [1, 1], reuse : [seq = false, spat = true, temp = true] : memref<128x128xf32> -> !loom.view
                %23 = loom.alloc(%c32, %c32) on @L1 : !loom.buffer_token
                %24 = loom.copy_to_tensor %22, %23 to @L1, interconnect : [@vertical_links], broadcast : [8, 1] : !loom.view, !loom.buffer_token -> tensor<?x?xf32>
                %25 = arith.muli %12, %c32 : index
                %26 = loom.view %arg1[%21, %25] [32, 32] [1, 1], reuse : [seq = false, spat = false, temp = true] : memref<128x256xf32> -> !loom.view
                %27 = loom.alloc(%c32, %c32) on @L1 : !loom.buffer_token
                %28 = loom.copy_to_tensor %26, %27 to @L1, interconnect : [], broadcast : [1, 1] : !loom.view, !loom.buffer_token -> tensor<?x?xf32>
                %29 = linalg.matmul ins(%24, %28 : tensor<?x?xf32>, tensor<?x?xf32>) outs(%15 : tensor<?x?xf32>) -> tensor<?x?xf32>
                %30 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%arg8, %29 : tensor<?x?xf32>, tensor<?x?xf32>) outs(%14 : tensor<?x?xf32>) {
                ^bb0(%in: f32, %in_0: f32, %out: f32):
                  %31 = arith.addf %in, %in_0 : f32
                  linalg.yield %31 : f32
                } -> tensor<?x?xf32>
                affine.yield %30 : tensor<?x?xf32>
              } {loom.iter_type = #loom.iter_type<sequential>}
              %17 = arith.muli %arg6, %c32 : index
              %18 = arith.muli %12, %c32 : index
              %19 = loom.view %arg2[%17, %18] [32, 32] [1, 1], reuse : [seq = false, spat = false, temp = false] : memref<128x256xf32> -> !loom.view
              loom.copy_from_tensor %16, %19 to @DRAM : tensor<?x?xf32>, !loom.view
            } {loom.iter_type = #loom.iter_type<temporal>}
          } {loom.iter_type = #loom.iter_type<temporal>}
        } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @y}
      } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @x}
      return
    }
    func.func @matmul__d0i1_d1i1__f10__v_d__BK64__BM32__BN32(%arg0: memref<128x128xf32>, %arg1: memref<128x256xf32>, %arg2: memref<128x256xf32>) {
      %cst = arith.constant 0.000000e+00 : f32
      %c32 = arith.constant 32 : index
      %c64 = arith.constant 64 : index
      affine.parallel (%arg3) = (0) to (8) {
        affine.parallel (%arg4) = (0) to (8) {
          affine.for %arg5 = 0 to 1 {
            affine.for %arg6 = 0 to 4 {
              %12 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 8 + d1 + d2 * 64)>(%arg3, %arg4, %arg5)
              %13 = loom.alloc(%c32, %c32) on @L1 : !loom.buffer_token
              %14 = loom.init_tensor %13(%c32, %c32) : !loom.buffer_token -> tensor<?x?xf32>
              %15 = linalg.fill ins(%cst : f32) outs(%14 : tensor<?x?xf32>) -> tensor<?x?xf32>
              %16 = affine.for %arg7 = 0 to 2 iter_args(%arg8 = %15) -> (tensor<?x?xf32>) {
                %20 = arith.muli %arg6, %c32 : index
                %21 = arith.muli %arg7, %c64 : index
                %22 = loom.view %arg0[%20, %21] [32, 64] [1, 1], reuse : [seq = false, spat = true, temp = true] : memref<128x128xf32> -> !loom.view
                %23 = loom.alloc(%c32, %c64) on @L1 : !loom.buffer_token
                %24 = loom.copy_to_tensor %22, %23 to @L1, interconnect : [@vertical_links], broadcast : [8, 1] : !loom.view, !loom.buffer_token -> tensor<?x?xf32>
                %25 = arith.muli %12, %c32 : index
                %26 = loom.view %arg1[%21, %25] [64, 32] [1, 1], reuse : [seq = false, spat = false, temp = true] : memref<128x256xf32> -> !loom.view
                %27 = loom.alloc(%c64, %c32) on @L1 : !loom.buffer_token
                %28 = loom.copy_to_tensor %26, %27 to @L1, interconnect : [], broadcast : [1, 1] : !loom.view, !loom.buffer_token -> tensor<?x?xf32>
                %29 = linalg.matmul ins(%24, %28 : tensor<?x?xf32>, tensor<?x?xf32>) outs(%15 : tensor<?x?xf32>) -> tensor<?x?xf32>
                %30 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%arg8, %29 : tensor<?x?xf32>, tensor<?x?xf32>) outs(%14 : tensor<?x?xf32>) {
                ^bb0(%in: f32, %in_0: f32, %out: f32):
                  %31 = arith.addf %in, %in_0 : f32
                  linalg.yield %31 : f32
                } -> tensor<?x?xf32>
                affine.yield %30 : tensor<?x?xf32>
              } {loom.iter_type = #loom.iter_type<sequential>}
              %17 = arith.muli %arg6, %c32 : index
              %18 = arith.muli %12, %c32 : index
              %19 = loom.view %arg2[%17, %18] [32, 32] [1, 1], reuse : [seq = false, spat = false, temp = false] : memref<128x256xf32> -> !loom.view
              loom.copy_from_tensor %16, %19 to @DRAM : tensor<?x?xf32>, !loom.view
            } {loom.iter_type = #loom.iter_type<temporal>}
          } {loom.iter_type = #loom.iter_type<temporal>}
        } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @y}
      } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @x}
      return
    }
  }
}
