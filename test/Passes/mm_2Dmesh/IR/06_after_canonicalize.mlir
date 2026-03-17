module attributes {loom.block_size_0 = -1 : index, loom.block_size_1 = -1 : index, loom.block_size_2 = -1 : index} {
  %0 = df.mat "FPU" {shape = [32, 32, 32], throughput = 128}
  %1 = df.vec "SFPU" {shape = [32]}
  %2 = df.spatial_dim "x", 8
  %3 = df.spatial_dim "y", 8
  %4 = df.core "core" {scaleout=(%2, %3) , scalein=(%0, %1, [8, 1])}
  %5 = df.memory "L1" {scaleout=(%2, %3) , size = 1499136, bandwidth = 15}
  %6 = df.mux %4 : !df.compute, %5 : !df.memory  {map = affine_map<(d0, d1) -> (d0, d1)>}
  %7 = df.interconnects "horizontal_links" %5 : !df.memory, %5 : !df.memory  {bandwidth = 128 : i64, map = affine_map<(d0, d1) -> ((d0 + 1) mod 8, d1)>, spatial_dims = [@x]} : !df.interconnect
  %8 = df.interconnects "vertical_links" %5 : !df.memory, %5 : !df.memory  {bandwidth = 128 : i64, map = affine_map<(d0, d1) -> (d0, (d1 + 1) mod 8)>, spatial_dims = [@y]} : !df.interconnect
  %9 = df.spatial_dim "d", 4
  %10 = df.memory "DRAM" {scaleout=(%9) , size = 34359738368, bandwidth = 288}
  %11 = df.interconnects "NoC" %5 : !df.memory, %10 : !df.memory  {map = affine_map<(d0, d1) -> (d0 ceildiv 4 + (d1 ceildiv 4) * 2)>} : !df.interconnect
  module attributes {loom.block_size_0 = -1 : index, loom.block_size_1 = -1 : index, loom.block_size_2 = -1 : index, loom.pass_name = "Materialize"} {
    func.func @matmul__d0i0_d1i0__f01__d_d__block_size_01__block_size_164__block_size_2128(%arg0: memref<4096x512xf32>, %arg1: memref<512x4096xf32>, %arg2: memref<4096x4096xf32>) {
      %cst = arith.constant 0.000000e+00 : f32
      %c64 = arith.constant 64 : index
      %c128 = arith.constant 128 : index
      affine.parallel (%arg3) = (0) to (8) {
        affine.parallel (%arg4) = (0) to (8) {
          affine.for %arg5 = 0 to 64 {
            affine.for %arg6 = 0 to 64 {
              %12 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 8 + d1 + d2 * 64)>(%arg3, %arg4, %arg5)
              %13 = loom.alloc [128, 64] on @L1 : memref<128x64xf32>
              %14 = loom.semaphore_take %13 : memref<128x64xf32> -> memref<128x64xf32>
              %15 = loom.alloc [1, 128] on @L1 : memref<1x128xf32>
              %16 = loom.semaphore_take %15 : memref<1x128xf32> -> memref<1x128xf32>
              %17 = loom.alloc [1, 64] on @L1 : memref<1x64xf32>
              %18 = loom.semaphore_take %17 : memref<1x64xf32> -> memref<1x64xf32>
              %19 = loom.init_tensor %18[1, 64] : memref<1x64xf32> -> tensor<1x64xf32>
              %20 = linalg.fill ins(%cst : f32) outs(%19 : tensor<1x64xf32>) -> tensor<1x64xf32>
              %21 = affine.for %arg7 = 0 to 4 iter_args(%arg8 = %20) -> (tensor<1x64xf32>) {
                %24 = arith.muli %arg7, %c128 : index
                %25 = affine.apply affine_map<(d0, d1) -> (d0 * 512 + d1)>(%12, %24)
                %reinterpret_cast_0 = memref.reinterpret_cast %arg0 to offset: [%25], sizes: [1, 128], strides: [512, 1] : memref<4096x512xf32> to memref<1x128xf32, strided<[512, 1], offset: ?>>
                %26 = loom.copy_to_tensor %reinterpret_cast_0, %16 on @L1, interconnect : [], broadcast : [1, 1] : memref<1x128xf32, strided<[512, 1], offset: ?>>, memref<1x128xf32> -> tensor<1x128xf32>
                %27 = arith.muli %arg7, %c128 : index
                %28 = arith.muli %arg6, %c64 : index
                %29 = affine.apply affine_map<(d0, d1) -> (d0 * 4096 + d1)>(%27, %28)
                %reinterpret_cast_1 = memref.reinterpret_cast %arg1 to offset: [%29], sizes: [128, 64], strides: [4096, 1] : memref<512x4096xf32> to memref<128x64xf32, strided<[4096, 1], offset: ?>>
                %30 = loom.copy_to_tensor %reinterpret_cast_1, %14 on @L1, interconnect : [], broadcast : [1, 1] : memref<128x64xf32, strided<[4096, 1], offset: ?>>, memref<128x64xf32> -> tensor<128x64xf32>
                %31 = linalg.matmul ins(%26, %30 : tensor<1x128xf32>, tensor<128x64xf32>) outs(%arg8 : tensor<1x64xf32>) -> tensor<1x64xf32>
                loom.semaphore_give %14 : memref<128x64xf32>
                loom.semaphore_give %16 : memref<1x128xf32>
                affine.yield %31 : tensor<1x64xf32>
              }
              %22 = arith.muli %arg6, %c64 : index
              %23 = affine.apply affine_map<(d0, d1) -> (d0 * 4096 + d1)>(%12, %22)
              %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%23], sizes: [1, 64], strides: [4096, 1] : memref<4096x4096xf32> to memref<1x64xf32, strided<[4096, 1], offset: ?>>
              loom.copy_from_tensor %21, %reinterpret_cast on @DRAM : tensor<1x64xf32>, memref<1x64xf32, strided<[4096, 1], offset: ?>>
              loom.semaphore_give %18 : memref<1x64xf32>
            } {loom.iter_type = #loom.iter_type<temporal>}
          } {loom.iter_type = #loom.iter_type<temporal>}
        } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @y}
      } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @x}
      return
    }
  }
  module attributes {loom.block_size_0 = -1 : index, loom.block_size_1 = -1 : index, loom.block_size_2 = -1 : index, loom.pass_name = "Materialize"} {
    func.func @matmul__d0i0_d1i0__f01__d_a__block_size_01__block_size_164__block_size_2128(%arg0: memref<4096x512xf32>, %arg1: memref<512x4096xf32>, %arg2: memref<4096x4096xf32>) {
      %cst = arith.constant 0.000000e+00 : f32
      %c64 = arith.constant 64 : index
      %c128 = arith.constant 128 : index
      affine.parallel (%arg3) = (0) to (8) {
        affine.parallel (%arg4) = (0) to (8) {
          affine.for %arg5 = 0 to 64 {
            affine.for %arg6 = 0 to 64 {
              %12 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 8 + d1 + d2 * 64)>(%arg3, %arg4, %arg5)
              %13 = loom.alloc [128, 64] on @L1 : memref<128x64xf32>
              %14 = loom.semaphore_take %13 : memref<128x64xf32> -> memref<128x64xf32>
              %15 = loom.alloc [1, 128] on @L1 : memref<1x128xf32>
              %16 = loom.semaphore_take %15 : memref<1x128xf32> -> memref<1x128xf32>
              %17 = loom.alloc [1, 64] on @L1 : memref<1x64xf32>
              %18 = loom.semaphore_take %17 : memref<1x64xf32> -> memref<1x64xf32>
              %19 = loom.init_tensor %18[1, 64] : memref<1x64xf32> -> tensor<1x64xf32>
              %20 = linalg.fill ins(%cst : f32) outs(%19 : tensor<1x64xf32>) -> tensor<1x64xf32>
              %21 = affine.for %arg7 = 0 to 4 iter_args(%arg8 = %20) -> (tensor<1x64xf32>) {
                %24 = arith.muli %arg7, %c128 : index
                %25 = affine.apply affine_map<(d0, d1) -> (d0 * 512 + d1)>(%12, %24)
                %reinterpret_cast_0 = memref.reinterpret_cast %arg0 to offset: [%25], sizes: [1, 128], strides: [512, 1] : memref<4096x512xf32> to memref<1x128xf32, strided<[512, 1], offset: ?>>
                %26 = loom.copy_to_tensor %reinterpret_cast_0, %16 on @L1, interconnect : [], broadcast : [1, 1] : memref<1x128xf32, strided<[512, 1], offset: ?>>, memref<1x128xf32> -> tensor<1x128xf32>
                %27 = arith.muli %arg7, %c128 : index
                %28 = arith.muli %arg6, %c64 : index
                %29 = affine.apply affine_map<(d0, d1) -> (d0 * 4096 + d1)>(%27, %28)
                %reinterpret_cast_1 = memref.reinterpret_cast %arg1 to offset: [%29], sizes: [128, 64], strides: [4096, 1] : memref<512x4096xf32> to memref<128x64xf32, strided<[4096, 1], offset: ?>>
                %30 = loom.copy_to_tensor %reinterpret_cast_1, %14 on @L1, interconnect : [@horizontal_links, @vertical_links], broadcast : [8, 8] : memref<128x64xf32, strided<[4096, 1], offset: ?>>, memref<128x64xf32> -> tensor<128x64xf32>
                %31 = linalg.matmul ins(%26, %30 : tensor<1x128xf32>, tensor<128x64xf32>) outs(%arg8 : tensor<1x64xf32>) -> tensor<1x64xf32>
                loom.semaphore_give %14 : memref<128x64xf32>
                loom.semaphore_give %16 : memref<1x128xf32>
                affine.yield %31 : tensor<1x64xf32>
              }
              %22 = arith.muli %arg6, %c64 : index
              %23 = affine.apply affine_map<(d0, d1) -> (d0 * 4096 + d1)>(%12, %22)
              %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%23], sizes: [1, 64], strides: [4096, 1] : memref<4096x4096xf32> to memref<1x64xf32, strided<[4096, 1], offset: ?>>
              loom.copy_from_tensor %21, %reinterpret_cast on @DRAM : tensor<1x64xf32>, memref<1x64xf32, strided<[4096, 1], offset: ?>>
              loom.semaphore_give %18 : memref<1x64xf32>
            } {loom.iter_type = #loom.iter_type<temporal>}
          } {loom.iter_type = #loom.iter_type<temporal>}
        } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @y}
      } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @x}
      return
    }
  }
  module attributes {loom.block_size_0 = -1 : index, loom.block_size_1 = -1 : index, loom.block_size_2 = -1 : index, loom.pass_name = "Materialize"} {
    func.func @matmul__d0i0_d1i0__f01__d_h__block_size_01__block_size_164__block_size_2128(%arg0: memref<4096x512xf32>, %arg1: memref<512x4096xf32>, %arg2: memref<4096x4096xf32>) {
      %cst = arith.constant 0.000000e+00 : f32
      %c64 = arith.constant 64 : index
      %c128 = arith.constant 128 : index
      affine.parallel (%arg3) = (0) to (8) {
        affine.parallel (%arg4) = (0) to (8) {
          affine.for %arg5 = 0 to 64 {
            affine.for %arg6 = 0 to 64 {
              %12 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 8 + d1 + d2 * 64)>(%arg3, %arg4, %arg5)
              %13 = loom.alloc [128, 64] on @L1 : memref<128x64xf32>
              %14 = loom.semaphore_take %13 : memref<128x64xf32> -> memref<128x64xf32>
              %15 = loom.alloc [1, 128] on @L1 : memref<1x128xf32>
              %16 = loom.semaphore_take %15 : memref<1x128xf32> -> memref<1x128xf32>
              %17 = loom.alloc [1, 64] on @L1 : memref<1x64xf32>
              %18 = loom.semaphore_take %17 : memref<1x64xf32> -> memref<1x64xf32>
              %19 = loom.init_tensor %18[1, 64] : memref<1x64xf32> -> tensor<1x64xf32>
              %20 = linalg.fill ins(%cst : f32) outs(%19 : tensor<1x64xf32>) -> tensor<1x64xf32>
              %21 = affine.for %arg7 = 0 to 4 iter_args(%arg8 = %20) -> (tensor<1x64xf32>) {
                %24 = arith.muli %arg7, %c128 : index
                %25 = affine.apply affine_map<(d0, d1) -> (d0 * 512 + d1)>(%12, %24)
                %reinterpret_cast_0 = memref.reinterpret_cast %arg0 to offset: [%25], sizes: [1, 128], strides: [512, 1] : memref<4096x512xf32> to memref<1x128xf32, strided<[512, 1], offset: ?>>
                %26 = loom.copy_to_tensor %reinterpret_cast_0, %16 on @L1, interconnect : [], broadcast : [1, 1] : memref<1x128xf32, strided<[512, 1], offset: ?>>, memref<1x128xf32> -> tensor<1x128xf32>
                %27 = arith.muli %arg7, %c128 : index
                %28 = arith.muli %arg6, %c64 : index
                %29 = affine.apply affine_map<(d0, d1) -> (d0 * 4096 + d1)>(%27, %28)
                %reinterpret_cast_1 = memref.reinterpret_cast %arg1 to offset: [%29], sizes: [128, 64], strides: [4096, 1] : memref<512x4096xf32> to memref<128x64xf32, strided<[4096, 1], offset: ?>>
                %30 = loom.copy_to_tensor %reinterpret_cast_1, %14 on @L1, interconnect : [@horizontal_links], broadcast : [1, 8] : memref<128x64xf32, strided<[4096, 1], offset: ?>>, memref<128x64xf32> -> tensor<128x64xf32>
                %31 = linalg.matmul ins(%26, %30 : tensor<1x128xf32>, tensor<128x64xf32>) outs(%arg8 : tensor<1x64xf32>) -> tensor<1x64xf32>
                loom.semaphore_give %14 : memref<128x64xf32>
                loom.semaphore_give %16 : memref<1x128xf32>
                affine.yield %31 : tensor<1x64xf32>
              }
              %22 = arith.muli %arg6, %c64 : index
              %23 = affine.apply affine_map<(d0, d1) -> (d0 * 4096 + d1)>(%12, %22)
              %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%23], sizes: [1, 64], strides: [4096, 1] : memref<4096x4096xf32> to memref<1x64xf32, strided<[4096, 1], offset: ?>>
              loom.copy_from_tensor %21, %reinterpret_cast on @DRAM : tensor<1x64xf32>, memref<1x64xf32, strided<[4096, 1], offset: ?>>
              loom.semaphore_give %18 : memref<1x64xf32>
            } {loom.iter_type = #loom.iter_type<temporal>}
          } {loom.iter_type = #loom.iter_type<temporal>}
        } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @y}
      } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @x}
      return
    }
  }
  module attributes {loom.block_size_0 = -1 : index, loom.block_size_1 = -1 : index, loom.block_size_2 = -1 : index, loom.pass_name = "Materialize"} {
    func.func @matmul__d0i0_d1i0__f01__d_v__block_size_01__block_size_164__block_size_2128(%arg0: memref<4096x512xf32>, %arg1: memref<512x4096xf32>, %arg2: memref<4096x4096xf32>) {
      %cst = arith.constant 0.000000e+00 : f32
      %c64 = arith.constant 64 : index
      %c128 = arith.constant 128 : index
      affine.parallel (%arg3) = (0) to (8) {
        affine.parallel (%arg4) = (0) to (8) {
          affine.for %arg5 = 0 to 64 {
            affine.for %arg6 = 0 to 64 {
              %12 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 8 + d1 + d2 * 64)>(%arg3, %arg4, %arg5)
              %13 = loom.alloc [128, 64] on @L1 : memref<128x64xf32>
              %14 = loom.semaphore_take %13 : memref<128x64xf32> -> memref<128x64xf32>
              %15 = loom.alloc [1, 128] on @L1 : memref<1x128xf32>
              %16 = loom.semaphore_take %15 : memref<1x128xf32> -> memref<1x128xf32>
              %17 = loom.alloc [1, 64] on @L1 : memref<1x64xf32>
              %18 = loom.semaphore_take %17 : memref<1x64xf32> -> memref<1x64xf32>
              %19 = loom.init_tensor %18[1, 64] : memref<1x64xf32> -> tensor<1x64xf32>
              %20 = linalg.fill ins(%cst : f32) outs(%19 : tensor<1x64xf32>) -> tensor<1x64xf32>
              %21 = affine.for %arg7 = 0 to 4 iter_args(%arg8 = %20) -> (tensor<1x64xf32>) {
                %24 = arith.muli %arg7, %c128 : index
                %25 = affine.apply affine_map<(d0, d1) -> (d0 * 512 + d1)>(%12, %24)
                %reinterpret_cast_0 = memref.reinterpret_cast %arg0 to offset: [%25], sizes: [1, 128], strides: [512, 1] : memref<4096x512xf32> to memref<1x128xf32, strided<[512, 1], offset: ?>>
                %26 = loom.copy_to_tensor %reinterpret_cast_0, %16 on @L1, interconnect : [], broadcast : [1, 1] : memref<1x128xf32, strided<[512, 1], offset: ?>>, memref<1x128xf32> -> tensor<1x128xf32>
                %27 = arith.muli %arg7, %c128 : index
                %28 = arith.muli %arg6, %c64 : index
                %29 = affine.apply affine_map<(d0, d1) -> (d0 * 4096 + d1)>(%27, %28)
                %reinterpret_cast_1 = memref.reinterpret_cast %arg1 to offset: [%29], sizes: [128, 64], strides: [4096, 1] : memref<512x4096xf32> to memref<128x64xf32, strided<[4096, 1], offset: ?>>
                %30 = loom.copy_to_tensor %reinterpret_cast_1, %14 on @L1, interconnect : [@vertical_links], broadcast : [8, 1] : memref<128x64xf32, strided<[4096, 1], offset: ?>>, memref<128x64xf32> -> tensor<128x64xf32>
                %31 = linalg.matmul ins(%26, %30 : tensor<1x128xf32>, tensor<128x64xf32>) outs(%arg8 : tensor<1x64xf32>) -> tensor<1x64xf32>
                loom.semaphore_give %14 : memref<128x64xf32>
                loom.semaphore_give %16 : memref<1x128xf32>
                affine.yield %31 : tensor<1x64xf32>
              }
              %22 = arith.muli %arg6, %c64 : index
              %23 = affine.apply affine_map<(d0, d1) -> (d0 * 4096 + d1)>(%12, %22)
              %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%23], sizes: [1, 64], strides: [4096, 1] : memref<4096x4096xf32> to memref<1x64xf32, strided<[4096, 1], offset: ?>>
              loom.copy_from_tensor %21, %reinterpret_cast on @DRAM : tensor<1x64xf32>, memref<1x64xf32, strided<[4096, 1], offset: ?>>
              loom.semaphore_give %18 : memref<1x64xf32>
            } {loom.iter_type = #loom.iter_type<temporal>}
          } {loom.iter_type = #loom.iter_type<temporal>}
        } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @y}
      } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @x}
      return
    }
  }
  module attributes {loom.block_size_0 = -1 : index, loom.block_size_1 = -1 : index, loom.block_size_2 = -1 : index, loom.pass_name = "Materialize"} {
    func.func @matmul__d0i0_d1i0__f10__d_d__block_size_01__block_size_164__block_size_2128(%arg0: memref<4096x512xf32>, %arg1: memref<512x4096xf32>, %arg2: memref<4096x4096xf32>) {
      %cst = arith.constant 0.000000e+00 : f32
      %c64 = arith.constant 64 : index
      %c128 = arith.constant 128 : index
      affine.parallel (%arg3) = (0) to (8) {
        affine.parallel (%arg4) = (0) to (8) {
          affine.for %arg5 = 0 to 64 {
            affine.for %arg6 = 0 to 64 {
              %12 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 8 + d1 + d2 * 64)>(%arg3, %arg4, %arg6)
              %13 = loom.alloc [128, 64] on @L1 : memref<128x64xf32>
              %14 = loom.semaphore_take %13 : memref<128x64xf32> -> memref<128x64xf32>
              %15 = loom.alloc [1, 128] on @L1 : memref<1x128xf32>
              %16 = loom.semaphore_take %15 : memref<1x128xf32> -> memref<1x128xf32>
              %17 = loom.alloc [1, 64] on @L1 : memref<1x64xf32>
              %18 = loom.semaphore_take %17 : memref<1x64xf32> -> memref<1x64xf32>
              %19 = loom.init_tensor %18[1, 64] : memref<1x64xf32> -> tensor<1x64xf32>
              %20 = linalg.fill ins(%cst : f32) outs(%19 : tensor<1x64xf32>) -> tensor<1x64xf32>
              %21 = affine.for %arg7 = 0 to 4 iter_args(%arg8 = %20) -> (tensor<1x64xf32>) {
                %24 = arith.muli %arg7, %c128 : index
                %25 = affine.apply affine_map<(d0, d1) -> (d0 * 512 + d1)>(%12, %24)
                %reinterpret_cast_0 = memref.reinterpret_cast %arg0 to offset: [%25], sizes: [1, 128], strides: [512, 1] : memref<4096x512xf32> to memref<1x128xf32, strided<[512, 1], offset: ?>>
                %26 = loom.copy_to_tensor %reinterpret_cast_0, %16 on @L1, interconnect : [], broadcast : [1, 1] : memref<1x128xf32, strided<[512, 1], offset: ?>>, memref<1x128xf32> -> tensor<1x128xf32>
                %27 = arith.muli %arg7, %c128 : index
                %28 = arith.muli %arg5, %c64 : index
                %29 = affine.apply affine_map<(d0, d1) -> (d0 * 4096 + d1)>(%27, %28)
                %reinterpret_cast_1 = memref.reinterpret_cast %arg1 to offset: [%29], sizes: [128, 64], strides: [4096, 1] : memref<512x4096xf32> to memref<128x64xf32, strided<[4096, 1], offset: ?>>
                %30 = loom.copy_to_tensor %reinterpret_cast_1, %14 on @L1, interconnect : [], broadcast : [1, 1] : memref<128x64xf32, strided<[4096, 1], offset: ?>>, memref<128x64xf32> -> tensor<128x64xf32>
                %31 = linalg.matmul ins(%26, %30 : tensor<1x128xf32>, tensor<128x64xf32>) outs(%arg8 : tensor<1x64xf32>) -> tensor<1x64xf32>
                loom.semaphore_give %14 : memref<128x64xf32>
                loom.semaphore_give %16 : memref<1x128xf32>
                affine.yield %31 : tensor<1x64xf32>
              }
              %22 = arith.muli %arg5, %c64 : index
              %23 = affine.apply affine_map<(d0, d1) -> (d0 * 4096 + d1)>(%12, %22)
              %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%23], sizes: [1, 64], strides: [4096, 1] : memref<4096x4096xf32> to memref<1x64xf32, strided<[4096, 1], offset: ?>>
              loom.copy_from_tensor %21, %reinterpret_cast on @DRAM : tensor<1x64xf32>, memref<1x64xf32, strided<[4096, 1], offset: ?>>
              loom.semaphore_give %18 : memref<1x64xf32>
            } {loom.iter_type = #loom.iter_type<temporal>}
          } {loom.iter_type = #loom.iter_type<temporal>}
        } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @y}
      } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @x}
      return
    }
  }
  module attributes {loom.block_size_0 = -1 : index, loom.block_size_1 = -1 : index, loom.block_size_2 = -1 : index, loom.pass_name = "Materialize"} {
    func.func @matmul__d0i0_d1i0__f10__d_a__block_size_01__block_size_164__block_size_2128(%arg0: memref<4096x512xf32>, %arg1: memref<512x4096xf32>, %arg2: memref<4096x4096xf32>) {
      %cst = arith.constant 0.000000e+00 : f32
      %c64 = arith.constant 64 : index
      %c128 = arith.constant 128 : index
      affine.parallel (%arg3) = (0) to (8) {
        affine.parallel (%arg4) = (0) to (8) {
          affine.for %arg5 = 0 to 64 {
            affine.for %arg6 = 0 to 64 {
              %12 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 8 + d1 + d2 * 64)>(%arg3, %arg4, %arg6)
              %13 = loom.alloc [128, 64] on @L1 : memref<128x64xf32>
              %14 = loom.semaphore_take %13 : memref<128x64xf32> -> memref<128x64xf32>
              %15 = loom.alloc [1, 128] on @L1 : memref<1x128xf32>
              %16 = loom.semaphore_take %15 : memref<1x128xf32> -> memref<1x128xf32>
              %17 = loom.alloc [1, 64] on @L1 : memref<1x64xf32>
              %18 = loom.semaphore_take %17 : memref<1x64xf32> -> memref<1x64xf32>
              %19 = loom.init_tensor %18[1, 64] : memref<1x64xf32> -> tensor<1x64xf32>
              %20 = linalg.fill ins(%cst : f32) outs(%19 : tensor<1x64xf32>) -> tensor<1x64xf32>
              %21 = affine.for %arg7 = 0 to 4 iter_args(%arg8 = %20) -> (tensor<1x64xf32>) {
                %24 = arith.muli %arg7, %c128 : index
                %25 = affine.apply affine_map<(d0, d1) -> (d0 * 512 + d1)>(%12, %24)
                %reinterpret_cast_0 = memref.reinterpret_cast %arg0 to offset: [%25], sizes: [1, 128], strides: [512, 1] : memref<4096x512xf32> to memref<1x128xf32, strided<[512, 1], offset: ?>>
                %26 = loom.copy_to_tensor %reinterpret_cast_0, %16 on @L1, interconnect : [], broadcast : [1, 1] : memref<1x128xf32, strided<[512, 1], offset: ?>>, memref<1x128xf32> -> tensor<1x128xf32>
                %27 = arith.muli %arg7, %c128 : index
                %28 = arith.muli %arg5, %c64 : index
                %29 = affine.apply affine_map<(d0, d1) -> (d0 * 4096 + d1)>(%27, %28)
                %reinterpret_cast_1 = memref.reinterpret_cast %arg1 to offset: [%29], sizes: [128, 64], strides: [4096, 1] : memref<512x4096xf32> to memref<128x64xf32, strided<[4096, 1], offset: ?>>
                %30 = loom.copy_to_tensor %reinterpret_cast_1, %14 on @L1, interconnect : [@horizontal_links, @vertical_links], broadcast : [8, 8] : memref<128x64xf32, strided<[4096, 1], offset: ?>>, memref<128x64xf32> -> tensor<128x64xf32>
                %31 = linalg.matmul ins(%26, %30 : tensor<1x128xf32>, tensor<128x64xf32>) outs(%arg8 : tensor<1x64xf32>) -> tensor<1x64xf32>
                loom.semaphore_give %14 : memref<128x64xf32>
                loom.semaphore_give %16 : memref<1x128xf32>
                affine.yield %31 : tensor<1x64xf32>
              }
              %22 = arith.muli %arg5, %c64 : index
              %23 = affine.apply affine_map<(d0, d1) -> (d0 * 4096 + d1)>(%12, %22)
              %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%23], sizes: [1, 64], strides: [4096, 1] : memref<4096x4096xf32> to memref<1x64xf32, strided<[4096, 1], offset: ?>>
              loom.copy_from_tensor %21, %reinterpret_cast on @DRAM : tensor<1x64xf32>, memref<1x64xf32, strided<[4096, 1], offset: ?>>
              loom.semaphore_give %18 : memref<1x64xf32>
            } {loom.iter_type = #loom.iter_type<temporal>}
          } {loom.iter_type = #loom.iter_type<temporal>}
        } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @y}
      } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @x}
      return
    }
  }
  module attributes {loom.block_size_0 = -1 : index, loom.block_size_1 = -1 : index, loom.block_size_2 = -1 : index, loom.pass_name = "Materialize"} {
    func.func @matmul__d0i0_d1i0__f10__d_h__block_size_01__block_size_164__block_size_2128(%arg0: memref<4096x512xf32>, %arg1: memref<512x4096xf32>, %arg2: memref<4096x4096xf32>) {
      %cst = arith.constant 0.000000e+00 : f32
      %c64 = arith.constant 64 : index
      %c128 = arith.constant 128 : index
      affine.parallel (%arg3) = (0) to (8) {
        affine.parallel (%arg4) = (0) to (8) {
          affine.for %arg5 = 0 to 64 {
            affine.for %arg6 = 0 to 64 {
              %12 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 8 + d1 + d2 * 64)>(%arg3, %arg4, %arg6)
              %13 = loom.alloc [128, 64] on @L1 : memref<128x64xf32>
              %14 = loom.semaphore_take %13 : memref<128x64xf32> -> memref<128x64xf32>
              %15 = loom.alloc [1, 128] on @L1 : memref<1x128xf32>
              %16 = loom.semaphore_take %15 : memref<1x128xf32> -> memref<1x128xf32>
              %17 = loom.alloc [1, 64] on @L1 : memref<1x64xf32>
              %18 = loom.semaphore_take %17 : memref<1x64xf32> -> memref<1x64xf32>
              %19 = loom.init_tensor %18[1, 64] : memref<1x64xf32> -> tensor<1x64xf32>
              %20 = linalg.fill ins(%cst : f32) outs(%19 : tensor<1x64xf32>) -> tensor<1x64xf32>
              %21 = affine.for %arg7 = 0 to 4 iter_args(%arg8 = %20) -> (tensor<1x64xf32>) {
                %24 = arith.muli %arg7, %c128 : index
                %25 = affine.apply affine_map<(d0, d1) -> (d0 * 512 + d1)>(%12, %24)
                %reinterpret_cast_0 = memref.reinterpret_cast %arg0 to offset: [%25], sizes: [1, 128], strides: [512, 1] : memref<4096x512xf32> to memref<1x128xf32, strided<[512, 1], offset: ?>>
                %26 = loom.copy_to_tensor %reinterpret_cast_0, %16 on @L1, interconnect : [], broadcast : [1, 1] : memref<1x128xf32, strided<[512, 1], offset: ?>>, memref<1x128xf32> -> tensor<1x128xf32>
                %27 = arith.muli %arg7, %c128 : index
                %28 = arith.muli %arg5, %c64 : index
                %29 = affine.apply affine_map<(d0, d1) -> (d0 * 4096 + d1)>(%27, %28)
                %reinterpret_cast_1 = memref.reinterpret_cast %arg1 to offset: [%29], sizes: [128, 64], strides: [4096, 1] : memref<512x4096xf32> to memref<128x64xf32, strided<[4096, 1], offset: ?>>
                %30 = loom.copy_to_tensor %reinterpret_cast_1, %14 on @L1, interconnect : [@horizontal_links], broadcast : [1, 8] : memref<128x64xf32, strided<[4096, 1], offset: ?>>, memref<128x64xf32> -> tensor<128x64xf32>
                %31 = linalg.matmul ins(%26, %30 : tensor<1x128xf32>, tensor<128x64xf32>) outs(%arg8 : tensor<1x64xf32>) -> tensor<1x64xf32>
                loom.semaphore_give %14 : memref<128x64xf32>
                loom.semaphore_give %16 : memref<1x128xf32>
                affine.yield %31 : tensor<1x64xf32>
              }
              %22 = arith.muli %arg5, %c64 : index
              %23 = affine.apply affine_map<(d0, d1) -> (d0 * 4096 + d1)>(%12, %22)
              %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%23], sizes: [1, 64], strides: [4096, 1] : memref<4096x4096xf32> to memref<1x64xf32, strided<[4096, 1], offset: ?>>
              loom.copy_from_tensor %21, %reinterpret_cast on @DRAM : tensor<1x64xf32>, memref<1x64xf32, strided<[4096, 1], offset: ?>>
              loom.semaphore_give %18 : memref<1x64xf32>
            } {loom.iter_type = #loom.iter_type<temporal>}
          } {loom.iter_type = #loom.iter_type<temporal>}
        } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @y}
      } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @x}
      return
    }
  }
  module attributes {loom.block_size_0 = -1 : index, loom.block_size_1 = -1 : index, loom.block_size_2 = -1 : index, loom.pass_name = "Materialize"} {
    func.func @matmul__d0i0_d1i0__f10__d_v__block_size_01__block_size_164__block_size_2128(%arg0: memref<4096x512xf32>, %arg1: memref<512x4096xf32>, %arg2: memref<4096x4096xf32>) {
      %cst = arith.constant 0.000000e+00 : f32
      %c64 = arith.constant 64 : index
      %c128 = arith.constant 128 : index
      affine.parallel (%arg3) = (0) to (8) {
        affine.parallel (%arg4) = (0) to (8) {
          affine.for %arg5 = 0 to 64 {
            affine.for %arg6 = 0 to 64 {
              %12 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 8 + d1 + d2 * 64)>(%arg3, %arg4, %arg6)
              %13 = loom.alloc [128, 64] on @L1 : memref<128x64xf32>
              %14 = loom.semaphore_take %13 : memref<128x64xf32> -> memref<128x64xf32>
              %15 = loom.alloc [1, 128] on @L1 : memref<1x128xf32>
              %16 = loom.semaphore_take %15 : memref<1x128xf32> -> memref<1x128xf32>
              %17 = loom.alloc [1, 64] on @L1 : memref<1x64xf32>
              %18 = loom.semaphore_take %17 : memref<1x64xf32> -> memref<1x64xf32>
              %19 = loom.init_tensor %18[1, 64] : memref<1x64xf32> -> tensor<1x64xf32>
              %20 = linalg.fill ins(%cst : f32) outs(%19 : tensor<1x64xf32>) -> tensor<1x64xf32>
              %21 = affine.for %arg7 = 0 to 4 iter_args(%arg8 = %20) -> (tensor<1x64xf32>) {
                %24 = arith.muli %arg7, %c128 : index
                %25 = affine.apply affine_map<(d0, d1) -> (d0 * 512 + d1)>(%12, %24)
                %reinterpret_cast_0 = memref.reinterpret_cast %arg0 to offset: [%25], sizes: [1, 128], strides: [512, 1] : memref<4096x512xf32> to memref<1x128xf32, strided<[512, 1], offset: ?>>
                %26 = loom.copy_to_tensor %reinterpret_cast_0, %16 on @L1, interconnect : [], broadcast : [1, 1] : memref<1x128xf32, strided<[512, 1], offset: ?>>, memref<1x128xf32> -> tensor<1x128xf32>
                %27 = arith.muli %arg7, %c128 : index
                %28 = arith.muli %arg5, %c64 : index
                %29 = affine.apply affine_map<(d0, d1) -> (d0 * 4096 + d1)>(%27, %28)
                %reinterpret_cast_1 = memref.reinterpret_cast %arg1 to offset: [%29], sizes: [128, 64], strides: [4096, 1] : memref<512x4096xf32> to memref<128x64xf32, strided<[4096, 1], offset: ?>>
                %30 = loom.copy_to_tensor %reinterpret_cast_1, %14 on @L1, interconnect : [@vertical_links], broadcast : [8, 1] : memref<128x64xf32, strided<[4096, 1], offset: ?>>, memref<128x64xf32> -> tensor<128x64xf32>
                %31 = linalg.matmul ins(%26, %30 : tensor<1x128xf32>, tensor<128x64xf32>) outs(%arg8 : tensor<1x64xf32>) -> tensor<1x64xf32>
                loom.semaphore_give %14 : memref<128x64xf32>
                loom.semaphore_give %16 : memref<1x128xf32>
                affine.yield %31 : tensor<1x64xf32>
              }
              %22 = arith.muli %arg5, %c64 : index
              %23 = affine.apply affine_map<(d0, d1) -> (d0 * 4096 + d1)>(%12, %22)
              %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%23], sizes: [1, 64], strides: [4096, 1] : memref<4096x4096xf32> to memref<1x64xf32, strided<[4096, 1], offset: ?>>
              loom.copy_from_tensor %21, %reinterpret_cast on @DRAM : tensor<1x64xf32>, memref<1x64xf32, strided<[4096, 1], offset: ?>>
              loom.semaphore_give %18 : memref<1x64xf32>
            } {loom.iter_type = #loom.iter_type<temporal>}
          } {loom.iter_type = #loom.iter_type<temporal>}
        } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @y}
      } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @x}
      return
    }
  }
  module attributes {loom.block_size_0 = -1 : index, loom.block_size_1 = -1 : index, loom.block_size_2 = -1 : index, loom.pass_name = "Materialize"} {
    func.func @matmul__d1i0_d0i0__f01__d_d__block_size_01__block_size_164__block_size_2128(%arg0: memref<4096x512xf32>, %arg1: memref<512x4096xf32>, %arg2: memref<4096x4096xf32>) {
      %cst = arith.constant 0.000000e+00 : f32
      %c64 = arith.constant 64 : index
      %c128 = arith.constant 128 : index
      affine.parallel (%arg3) = (0) to (8) {
        affine.parallel (%arg4) = (0) to (8) {
          affine.for %arg5 = 0 to 64 {
            affine.for %arg6 = 0 to 64 {
              %12 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 8 + d1 + d2 * 64)>(%arg3, %arg4, %arg5)
              %13 = loom.alloc [128, 64] on @L1 : memref<128x64xf32>
              %14 = loom.semaphore_take %13 : memref<128x64xf32> -> memref<128x64xf32>
              %15 = loom.alloc [1, 128] on @L1 : memref<1x128xf32>
              %16 = loom.semaphore_take %15 : memref<1x128xf32> -> memref<1x128xf32>
              %17 = loom.alloc [1, 64] on @L1 : memref<1x64xf32>
              %18 = loom.semaphore_take %17 : memref<1x64xf32> -> memref<1x64xf32>
              %19 = loom.init_tensor %18[1, 64] : memref<1x64xf32> -> tensor<1x64xf32>
              %20 = linalg.fill ins(%cst : f32) outs(%19 : tensor<1x64xf32>) -> tensor<1x64xf32>
              %21 = affine.for %arg7 = 0 to 4 iter_args(%arg8 = %20) -> (tensor<1x64xf32>) {
                %24 = arith.muli %arg7, %c128 : index
                %25 = affine.apply affine_map<(d0, d1) -> (d0 * 512 + d1)>(%12, %24)
                %reinterpret_cast_0 = memref.reinterpret_cast %arg0 to offset: [%25], sizes: [1, 128], strides: [512, 1] : memref<4096x512xf32> to memref<1x128xf32, strided<[512, 1], offset: ?>>
                %26 = loom.copy_to_tensor %reinterpret_cast_0, %16 on @L1, interconnect : [], broadcast : [1, 1] : memref<1x128xf32, strided<[512, 1], offset: ?>>, memref<1x128xf32> -> tensor<1x128xf32>
                %27 = arith.muli %arg7, %c128 : index
                %28 = arith.muli %arg6, %c64 : index
                %29 = affine.apply affine_map<(d0, d1) -> (d0 * 4096 + d1)>(%27, %28)
                %reinterpret_cast_1 = memref.reinterpret_cast %arg1 to offset: [%29], sizes: [128, 64], strides: [4096, 1] : memref<512x4096xf32> to memref<128x64xf32, strided<[4096, 1], offset: ?>>
                %30 = loom.copy_to_tensor %reinterpret_cast_1, %14 on @L1, interconnect : [], broadcast : [1, 1] : memref<128x64xf32, strided<[4096, 1], offset: ?>>, memref<128x64xf32> -> tensor<128x64xf32>
                %31 = linalg.matmul ins(%26, %30 : tensor<1x128xf32>, tensor<128x64xf32>) outs(%arg8 : tensor<1x64xf32>) -> tensor<1x64xf32>
                loom.semaphore_give %14 : memref<128x64xf32>
                loom.semaphore_give %16 : memref<1x128xf32>
                affine.yield %31 : tensor<1x64xf32>
              }
              %22 = arith.muli %arg6, %c64 : index
              %23 = affine.apply affine_map<(d0, d1) -> (d0 * 4096 + d1)>(%12, %22)
              %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%23], sizes: [1, 64], strides: [4096, 1] : memref<4096x4096xf32> to memref<1x64xf32, strided<[4096, 1], offset: ?>>
              loom.copy_from_tensor %21, %reinterpret_cast on @DRAM : tensor<1x64xf32>, memref<1x64xf32, strided<[4096, 1], offset: ?>>
              loom.semaphore_give %18 : memref<1x64xf32>
            } {loom.iter_type = #loom.iter_type<temporal>}
          } {loom.iter_type = #loom.iter_type<temporal>}
        } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @x}
      } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @y}
      return
    }
  }
  module attributes {loom.block_size_0 = -1 : index, loom.block_size_1 = -1 : index, loom.block_size_2 = -1 : index, loom.pass_name = "Materialize"} {
    func.func @matmul__d1i0_d0i0__f01__d_a__block_size_01__block_size_164__block_size_2128(%arg0: memref<4096x512xf32>, %arg1: memref<512x4096xf32>, %arg2: memref<4096x4096xf32>) {
      %cst = arith.constant 0.000000e+00 : f32
      %c64 = arith.constant 64 : index
      %c128 = arith.constant 128 : index
      affine.parallel (%arg3) = (0) to (8) {
        affine.parallel (%arg4) = (0) to (8) {
          affine.for %arg5 = 0 to 64 {
            affine.for %arg6 = 0 to 64 {
              %12 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 8 + d1 + d2 * 64)>(%arg3, %arg4, %arg5)
              %13 = loom.alloc [128, 64] on @L1 : memref<128x64xf32>
              %14 = loom.semaphore_take %13 : memref<128x64xf32> -> memref<128x64xf32>
              %15 = loom.alloc [1, 128] on @L1 : memref<1x128xf32>
              %16 = loom.semaphore_take %15 : memref<1x128xf32> -> memref<1x128xf32>
              %17 = loom.alloc [1, 64] on @L1 : memref<1x64xf32>
              %18 = loom.semaphore_take %17 : memref<1x64xf32> -> memref<1x64xf32>
              %19 = loom.init_tensor %18[1, 64] : memref<1x64xf32> -> tensor<1x64xf32>
              %20 = linalg.fill ins(%cst : f32) outs(%19 : tensor<1x64xf32>) -> tensor<1x64xf32>
              %21 = affine.for %arg7 = 0 to 4 iter_args(%arg8 = %20) -> (tensor<1x64xf32>) {
                %24 = arith.muli %arg7, %c128 : index
                %25 = affine.apply affine_map<(d0, d1) -> (d0 * 512 + d1)>(%12, %24)
                %reinterpret_cast_0 = memref.reinterpret_cast %arg0 to offset: [%25], sizes: [1, 128], strides: [512, 1] : memref<4096x512xf32> to memref<1x128xf32, strided<[512, 1], offset: ?>>
                %26 = loom.copy_to_tensor %reinterpret_cast_0, %16 on @L1, interconnect : [], broadcast : [1, 1] : memref<1x128xf32, strided<[512, 1], offset: ?>>, memref<1x128xf32> -> tensor<1x128xf32>
                %27 = arith.muli %arg7, %c128 : index
                %28 = arith.muli %arg6, %c64 : index
                %29 = affine.apply affine_map<(d0, d1) -> (d0 * 4096 + d1)>(%27, %28)
                %reinterpret_cast_1 = memref.reinterpret_cast %arg1 to offset: [%29], sizes: [128, 64], strides: [4096, 1] : memref<512x4096xf32> to memref<128x64xf32, strided<[4096, 1], offset: ?>>
                %30 = loom.copy_to_tensor %reinterpret_cast_1, %14 on @L1, interconnect : [@horizontal_links, @vertical_links], broadcast : [8, 8] : memref<128x64xf32, strided<[4096, 1], offset: ?>>, memref<128x64xf32> -> tensor<128x64xf32>
                %31 = linalg.matmul ins(%26, %30 : tensor<1x128xf32>, tensor<128x64xf32>) outs(%arg8 : tensor<1x64xf32>) -> tensor<1x64xf32>
                loom.semaphore_give %14 : memref<128x64xf32>
                loom.semaphore_give %16 : memref<1x128xf32>
                affine.yield %31 : tensor<1x64xf32>
              }
              %22 = arith.muli %arg6, %c64 : index
              %23 = affine.apply affine_map<(d0, d1) -> (d0 * 4096 + d1)>(%12, %22)
              %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%23], sizes: [1, 64], strides: [4096, 1] : memref<4096x4096xf32> to memref<1x64xf32, strided<[4096, 1], offset: ?>>
              loom.copy_from_tensor %21, %reinterpret_cast on @DRAM : tensor<1x64xf32>, memref<1x64xf32, strided<[4096, 1], offset: ?>>
              loom.semaphore_give %18 : memref<1x64xf32>
            } {loom.iter_type = #loom.iter_type<temporal>}
          } {loom.iter_type = #loom.iter_type<temporal>}
        } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @x}
      } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @y}
      return
    }
  }
  module attributes {loom.block_size_0 = -1 : index, loom.block_size_1 = -1 : index, loom.block_size_2 = -1 : index, loom.pass_name = "Materialize"} {
    func.func @matmul__d1i0_d0i0__f01__d_h__block_size_01__block_size_164__block_size_2128(%arg0: memref<4096x512xf32>, %arg1: memref<512x4096xf32>, %arg2: memref<4096x4096xf32>) {
      %cst = arith.constant 0.000000e+00 : f32
      %c64 = arith.constant 64 : index
      %c128 = arith.constant 128 : index
      affine.parallel (%arg3) = (0) to (8) {
        affine.parallel (%arg4) = (0) to (8) {
          affine.for %arg5 = 0 to 64 {
            affine.for %arg6 = 0 to 64 {
              %12 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 8 + d1 + d2 * 64)>(%arg3, %arg4, %arg5)
              %13 = loom.alloc [128, 64] on @L1 : memref<128x64xf32>
              %14 = loom.semaphore_take %13 : memref<128x64xf32> -> memref<128x64xf32>
              %15 = loom.alloc [1, 128] on @L1 : memref<1x128xf32>
              %16 = loom.semaphore_take %15 : memref<1x128xf32> -> memref<1x128xf32>
              %17 = loom.alloc [1, 64] on @L1 : memref<1x64xf32>
              %18 = loom.semaphore_take %17 : memref<1x64xf32> -> memref<1x64xf32>
              %19 = loom.init_tensor %18[1, 64] : memref<1x64xf32> -> tensor<1x64xf32>
              %20 = linalg.fill ins(%cst : f32) outs(%19 : tensor<1x64xf32>) -> tensor<1x64xf32>
              %21 = affine.for %arg7 = 0 to 4 iter_args(%arg8 = %20) -> (tensor<1x64xf32>) {
                %24 = arith.muli %arg7, %c128 : index
                %25 = affine.apply affine_map<(d0, d1) -> (d0 * 512 + d1)>(%12, %24)
                %reinterpret_cast_0 = memref.reinterpret_cast %arg0 to offset: [%25], sizes: [1, 128], strides: [512, 1] : memref<4096x512xf32> to memref<1x128xf32, strided<[512, 1], offset: ?>>
                %26 = loom.copy_to_tensor %reinterpret_cast_0, %16 on @L1, interconnect : [], broadcast : [1, 1] : memref<1x128xf32, strided<[512, 1], offset: ?>>, memref<1x128xf32> -> tensor<1x128xf32>
                %27 = arith.muli %arg7, %c128 : index
                %28 = arith.muli %arg6, %c64 : index
                %29 = affine.apply affine_map<(d0, d1) -> (d0 * 4096 + d1)>(%27, %28)
                %reinterpret_cast_1 = memref.reinterpret_cast %arg1 to offset: [%29], sizes: [128, 64], strides: [4096, 1] : memref<512x4096xf32> to memref<128x64xf32, strided<[4096, 1], offset: ?>>
                %30 = loom.copy_to_tensor %reinterpret_cast_1, %14 on @L1, interconnect : [@horizontal_links], broadcast : [1, 8] : memref<128x64xf32, strided<[4096, 1], offset: ?>>, memref<128x64xf32> -> tensor<128x64xf32>
                %31 = linalg.matmul ins(%26, %30 : tensor<1x128xf32>, tensor<128x64xf32>) outs(%arg8 : tensor<1x64xf32>) -> tensor<1x64xf32>
                loom.semaphore_give %14 : memref<128x64xf32>
                loom.semaphore_give %16 : memref<1x128xf32>
                affine.yield %31 : tensor<1x64xf32>
              }
              %22 = arith.muli %arg6, %c64 : index
              %23 = affine.apply affine_map<(d0, d1) -> (d0 * 4096 + d1)>(%12, %22)
              %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%23], sizes: [1, 64], strides: [4096, 1] : memref<4096x4096xf32> to memref<1x64xf32, strided<[4096, 1], offset: ?>>
              loom.copy_from_tensor %21, %reinterpret_cast on @DRAM : tensor<1x64xf32>, memref<1x64xf32, strided<[4096, 1], offset: ?>>
              loom.semaphore_give %18 : memref<1x64xf32>
            } {loom.iter_type = #loom.iter_type<temporal>}
          } {loom.iter_type = #loom.iter_type<temporal>}
        } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @x}
      } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @y}
      return
    }
  }
  module attributes {loom.block_size_0 = -1 : index, loom.block_size_1 = -1 : index, loom.block_size_2 = -1 : index, loom.pass_name = "Materialize"} {
    func.func @matmul__d1i0_d0i0__f01__d_v__block_size_01__block_size_164__block_size_2128(%arg0: memref<4096x512xf32>, %arg1: memref<512x4096xf32>, %arg2: memref<4096x4096xf32>) {
      %cst = arith.constant 0.000000e+00 : f32
      %c64 = arith.constant 64 : index
      %c128 = arith.constant 128 : index
      affine.parallel (%arg3) = (0) to (8) {
        affine.parallel (%arg4) = (0) to (8) {
          affine.for %arg5 = 0 to 64 {
            affine.for %arg6 = 0 to 64 {
              %12 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 8 + d1 + d2 * 64)>(%arg3, %arg4, %arg5)
              %13 = loom.alloc [128, 64] on @L1 : memref<128x64xf32>
              %14 = loom.semaphore_take %13 : memref<128x64xf32> -> memref<128x64xf32>
              %15 = loom.alloc [1, 128] on @L1 : memref<1x128xf32>
              %16 = loom.semaphore_take %15 : memref<1x128xf32> -> memref<1x128xf32>
              %17 = loom.alloc [1, 64] on @L1 : memref<1x64xf32>
              %18 = loom.semaphore_take %17 : memref<1x64xf32> -> memref<1x64xf32>
              %19 = loom.init_tensor %18[1, 64] : memref<1x64xf32> -> tensor<1x64xf32>
              %20 = linalg.fill ins(%cst : f32) outs(%19 : tensor<1x64xf32>) -> tensor<1x64xf32>
              %21 = affine.for %arg7 = 0 to 4 iter_args(%arg8 = %20) -> (tensor<1x64xf32>) {
                %24 = arith.muli %arg7, %c128 : index
                %25 = affine.apply affine_map<(d0, d1) -> (d0 * 512 + d1)>(%12, %24)
                %reinterpret_cast_0 = memref.reinterpret_cast %arg0 to offset: [%25], sizes: [1, 128], strides: [512, 1] : memref<4096x512xf32> to memref<1x128xf32, strided<[512, 1], offset: ?>>
                %26 = loom.copy_to_tensor %reinterpret_cast_0, %16 on @L1, interconnect : [], broadcast : [1, 1] : memref<1x128xf32, strided<[512, 1], offset: ?>>, memref<1x128xf32> -> tensor<1x128xf32>
                %27 = arith.muli %arg7, %c128 : index
                %28 = arith.muli %arg6, %c64 : index
                %29 = affine.apply affine_map<(d0, d1) -> (d0 * 4096 + d1)>(%27, %28)
                %reinterpret_cast_1 = memref.reinterpret_cast %arg1 to offset: [%29], sizes: [128, 64], strides: [4096, 1] : memref<512x4096xf32> to memref<128x64xf32, strided<[4096, 1], offset: ?>>
                %30 = loom.copy_to_tensor %reinterpret_cast_1, %14 on @L1, interconnect : [@vertical_links], broadcast : [8, 1] : memref<128x64xf32, strided<[4096, 1], offset: ?>>, memref<128x64xf32> -> tensor<128x64xf32>
                %31 = linalg.matmul ins(%26, %30 : tensor<1x128xf32>, tensor<128x64xf32>) outs(%arg8 : tensor<1x64xf32>) -> tensor<1x64xf32>
                loom.semaphore_give %14 : memref<128x64xf32>
                loom.semaphore_give %16 : memref<1x128xf32>
                affine.yield %31 : tensor<1x64xf32>
              }
              %22 = arith.muli %arg6, %c64 : index
              %23 = affine.apply affine_map<(d0, d1) -> (d0 * 4096 + d1)>(%12, %22)
              %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%23], sizes: [1, 64], strides: [4096, 1] : memref<4096x4096xf32> to memref<1x64xf32, strided<[4096, 1], offset: ?>>
              loom.copy_from_tensor %21, %reinterpret_cast on @DRAM : tensor<1x64xf32>, memref<1x64xf32, strided<[4096, 1], offset: ?>>
              loom.semaphore_give %18 : memref<1x64xf32>
            } {loom.iter_type = #loom.iter_type<temporal>}
          } {loom.iter_type = #loom.iter_type<temporal>}
        } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @x}
      } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @y}
      return
    }
  }
  module attributes {loom.block_size_0 = -1 : index, loom.block_size_1 = -1 : index, loom.block_size_2 = -1 : index, loom.pass_name = "Materialize"} {
    func.func @matmul__d1i0_d0i0__f10__d_d__block_size_01__block_size_164__block_size_2128(%arg0: memref<4096x512xf32>, %arg1: memref<512x4096xf32>, %arg2: memref<4096x4096xf32>) {
      %cst = arith.constant 0.000000e+00 : f32
      %c64 = arith.constant 64 : index
      %c128 = arith.constant 128 : index
      affine.parallel (%arg3) = (0) to (8) {
        affine.parallel (%arg4) = (0) to (8) {
          affine.for %arg5 = 0 to 64 {
            affine.for %arg6 = 0 to 64 {
              %12 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 8 + d1 + d2 * 64)>(%arg3, %arg4, %arg6)
              %13 = loom.alloc [128, 64] on @L1 : memref<128x64xf32>
              %14 = loom.semaphore_take %13 : memref<128x64xf32> -> memref<128x64xf32>
              %15 = loom.alloc [1, 128] on @L1 : memref<1x128xf32>
              %16 = loom.semaphore_take %15 : memref<1x128xf32> -> memref<1x128xf32>
              %17 = loom.alloc [1, 64] on @L1 : memref<1x64xf32>
              %18 = loom.semaphore_take %17 : memref<1x64xf32> -> memref<1x64xf32>
              %19 = loom.init_tensor %18[1, 64] : memref<1x64xf32> -> tensor<1x64xf32>
              %20 = linalg.fill ins(%cst : f32) outs(%19 : tensor<1x64xf32>) -> tensor<1x64xf32>
              %21 = affine.for %arg7 = 0 to 4 iter_args(%arg8 = %20) -> (tensor<1x64xf32>) {
                %24 = arith.muli %arg7, %c128 : index
                %25 = affine.apply affine_map<(d0, d1) -> (d0 * 512 + d1)>(%12, %24)
                %reinterpret_cast_0 = memref.reinterpret_cast %arg0 to offset: [%25], sizes: [1, 128], strides: [512, 1] : memref<4096x512xf32> to memref<1x128xf32, strided<[512, 1], offset: ?>>
                %26 = loom.copy_to_tensor %reinterpret_cast_0, %16 on @L1, interconnect : [], broadcast : [1, 1] : memref<1x128xf32, strided<[512, 1], offset: ?>>, memref<1x128xf32> -> tensor<1x128xf32>
                %27 = arith.muli %arg7, %c128 : index
                %28 = arith.muli %arg5, %c64 : index
                %29 = affine.apply affine_map<(d0, d1) -> (d0 * 4096 + d1)>(%27, %28)
                %reinterpret_cast_1 = memref.reinterpret_cast %arg1 to offset: [%29], sizes: [128, 64], strides: [4096, 1] : memref<512x4096xf32> to memref<128x64xf32, strided<[4096, 1], offset: ?>>
                %30 = loom.copy_to_tensor %reinterpret_cast_1, %14 on @L1, interconnect : [], broadcast : [1, 1] : memref<128x64xf32, strided<[4096, 1], offset: ?>>, memref<128x64xf32> -> tensor<128x64xf32>
                %31 = linalg.matmul ins(%26, %30 : tensor<1x128xf32>, tensor<128x64xf32>) outs(%arg8 : tensor<1x64xf32>) -> tensor<1x64xf32>
                loom.semaphore_give %14 : memref<128x64xf32>
                loom.semaphore_give %16 : memref<1x128xf32>
                affine.yield %31 : tensor<1x64xf32>
              }
              %22 = arith.muli %arg5, %c64 : index
              %23 = affine.apply affine_map<(d0, d1) -> (d0 * 4096 + d1)>(%12, %22)
              %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%23], sizes: [1, 64], strides: [4096, 1] : memref<4096x4096xf32> to memref<1x64xf32, strided<[4096, 1], offset: ?>>
              loom.copy_from_tensor %21, %reinterpret_cast on @DRAM : tensor<1x64xf32>, memref<1x64xf32, strided<[4096, 1], offset: ?>>
              loom.semaphore_give %18 : memref<1x64xf32>
            } {loom.iter_type = #loom.iter_type<temporal>}
          } {loom.iter_type = #loom.iter_type<temporal>}
        } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @x}
      } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @y}
      return
    }
  }
  module attributes {loom.block_size_0 = -1 : index, loom.block_size_1 = -1 : index, loom.block_size_2 = -1 : index, loom.pass_name = "Materialize"} {
    func.func @matmul__d1i0_d0i0__f10__d_a__block_size_01__block_size_164__block_size_2128(%arg0: memref<4096x512xf32>, %arg1: memref<512x4096xf32>, %arg2: memref<4096x4096xf32>) {
      %cst = arith.constant 0.000000e+00 : f32
      %c64 = arith.constant 64 : index
      %c128 = arith.constant 128 : index
      affine.parallel (%arg3) = (0) to (8) {
        affine.parallel (%arg4) = (0) to (8) {
          affine.for %arg5 = 0 to 64 {
            affine.for %arg6 = 0 to 64 {
              %12 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 8 + d1 + d2 * 64)>(%arg3, %arg4, %arg6)
              %13 = loom.alloc [128, 64] on @L1 : memref<128x64xf32>
              %14 = loom.semaphore_take %13 : memref<128x64xf32> -> memref<128x64xf32>
              %15 = loom.alloc [1, 128] on @L1 : memref<1x128xf32>
              %16 = loom.semaphore_take %15 : memref<1x128xf32> -> memref<1x128xf32>
              %17 = loom.alloc [1, 64] on @L1 : memref<1x64xf32>
              %18 = loom.semaphore_take %17 : memref<1x64xf32> -> memref<1x64xf32>
              %19 = loom.init_tensor %18[1, 64] : memref<1x64xf32> -> tensor<1x64xf32>
              %20 = linalg.fill ins(%cst : f32) outs(%19 : tensor<1x64xf32>) -> tensor<1x64xf32>
              %21 = affine.for %arg7 = 0 to 4 iter_args(%arg8 = %20) -> (tensor<1x64xf32>) {
                %24 = arith.muli %arg7, %c128 : index
                %25 = affine.apply affine_map<(d0, d1) -> (d0 * 512 + d1)>(%12, %24)
                %reinterpret_cast_0 = memref.reinterpret_cast %arg0 to offset: [%25], sizes: [1, 128], strides: [512, 1] : memref<4096x512xf32> to memref<1x128xf32, strided<[512, 1], offset: ?>>
                %26 = loom.copy_to_tensor %reinterpret_cast_0, %16 on @L1, interconnect : [], broadcast : [1, 1] : memref<1x128xf32, strided<[512, 1], offset: ?>>, memref<1x128xf32> -> tensor<1x128xf32>
                %27 = arith.muli %arg7, %c128 : index
                %28 = arith.muli %arg5, %c64 : index
                %29 = affine.apply affine_map<(d0, d1) -> (d0 * 4096 + d1)>(%27, %28)
                %reinterpret_cast_1 = memref.reinterpret_cast %arg1 to offset: [%29], sizes: [128, 64], strides: [4096, 1] : memref<512x4096xf32> to memref<128x64xf32, strided<[4096, 1], offset: ?>>
                %30 = loom.copy_to_tensor %reinterpret_cast_1, %14 on @L1, interconnect : [@horizontal_links, @vertical_links], broadcast : [8, 8] : memref<128x64xf32, strided<[4096, 1], offset: ?>>, memref<128x64xf32> -> tensor<128x64xf32>
                %31 = linalg.matmul ins(%26, %30 : tensor<1x128xf32>, tensor<128x64xf32>) outs(%arg8 : tensor<1x64xf32>) -> tensor<1x64xf32>
                loom.semaphore_give %14 : memref<128x64xf32>
                loom.semaphore_give %16 : memref<1x128xf32>
                affine.yield %31 : tensor<1x64xf32>
              }
              %22 = arith.muli %arg5, %c64 : index
              %23 = affine.apply affine_map<(d0, d1) -> (d0 * 4096 + d1)>(%12, %22)
              %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%23], sizes: [1, 64], strides: [4096, 1] : memref<4096x4096xf32> to memref<1x64xf32, strided<[4096, 1], offset: ?>>
              loom.copy_from_tensor %21, %reinterpret_cast on @DRAM : tensor<1x64xf32>, memref<1x64xf32, strided<[4096, 1], offset: ?>>
              loom.semaphore_give %18 : memref<1x64xf32>
            } {loom.iter_type = #loom.iter_type<temporal>}
          } {loom.iter_type = #loom.iter_type<temporal>}
        } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @x}
      } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @y}
      return
    }
  }
  module attributes {loom.block_size_0 = -1 : index, loom.block_size_1 = -1 : index, loom.block_size_2 = -1 : index, loom.pass_name = "Materialize"} {
    func.func @matmul__d1i0_d0i0__f10__d_h__block_size_01__block_size_164__block_size_2128(%arg0: memref<4096x512xf32>, %arg1: memref<512x4096xf32>, %arg2: memref<4096x4096xf32>) {
      %cst = arith.constant 0.000000e+00 : f32
      %c64 = arith.constant 64 : index
      %c128 = arith.constant 128 : index
      affine.parallel (%arg3) = (0) to (8) {
        affine.parallel (%arg4) = (0) to (8) {
          affine.for %arg5 = 0 to 64 {
            affine.for %arg6 = 0 to 64 {
              %12 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 8 + d1 + d2 * 64)>(%arg3, %arg4, %arg6)
              %13 = loom.alloc [128, 64] on @L1 : memref<128x64xf32>
              %14 = loom.semaphore_take %13 : memref<128x64xf32> -> memref<128x64xf32>
              %15 = loom.alloc [1, 128] on @L1 : memref<1x128xf32>
              %16 = loom.semaphore_take %15 : memref<1x128xf32> -> memref<1x128xf32>
              %17 = loom.alloc [1, 64] on @L1 : memref<1x64xf32>
              %18 = loom.semaphore_take %17 : memref<1x64xf32> -> memref<1x64xf32>
              %19 = loom.init_tensor %18[1, 64] : memref<1x64xf32> -> tensor<1x64xf32>
              %20 = linalg.fill ins(%cst : f32) outs(%19 : tensor<1x64xf32>) -> tensor<1x64xf32>
              %21 = affine.for %arg7 = 0 to 4 iter_args(%arg8 = %20) -> (tensor<1x64xf32>) {
                %24 = arith.muli %arg7, %c128 : index
                %25 = affine.apply affine_map<(d0, d1) -> (d0 * 512 + d1)>(%12, %24)
                %reinterpret_cast_0 = memref.reinterpret_cast %arg0 to offset: [%25], sizes: [1, 128], strides: [512, 1] : memref<4096x512xf32> to memref<1x128xf32, strided<[512, 1], offset: ?>>
                %26 = loom.copy_to_tensor %reinterpret_cast_0, %16 on @L1, interconnect : [], broadcast : [1, 1] : memref<1x128xf32, strided<[512, 1], offset: ?>>, memref<1x128xf32> -> tensor<1x128xf32>
                %27 = arith.muli %arg7, %c128 : index
                %28 = arith.muli %arg5, %c64 : index
                %29 = affine.apply affine_map<(d0, d1) -> (d0 * 4096 + d1)>(%27, %28)
                %reinterpret_cast_1 = memref.reinterpret_cast %arg1 to offset: [%29], sizes: [128, 64], strides: [4096, 1] : memref<512x4096xf32> to memref<128x64xf32, strided<[4096, 1], offset: ?>>
                %30 = loom.copy_to_tensor %reinterpret_cast_1, %14 on @L1, interconnect : [@horizontal_links], broadcast : [1, 8] : memref<128x64xf32, strided<[4096, 1], offset: ?>>, memref<128x64xf32> -> tensor<128x64xf32>
                %31 = linalg.matmul ins(%26, %30 : tensor<1x128xf32>, tensor<128x64xf32>) outs(%arg8 : tensor<1x64xf32>) -> tensor<1x64xf32>
                loom.semaphore_give %14 : memref<128x64xf32>
                loom.semaphore_give %16 : memref<1x128xf32>
                affine.yield %31 : tensor<1x64xf32>
              }
              %22 = arith.muli %arg5, %c64 : index
              %23 = affine.apply affine_map<(d0, d1) -> (d0 * 4096 + d1)>(%12, %22)
              %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%23], sizes: [1, 64], strides: [4096, 1] : memref<4096x4096xf32> to memref<1x64xf32, strided<[4096, 1], offset: ?>>
              loom.copy_from_tensor %21, %reinterpret_cast on @DRAM : tensor<1x64xf32>, memref<1x64xf32, strided<[4096, 1], offset: ?>>
              loom.semaphore_give %18 : memref<1x64xf32>
            } {loom.iter_type = #loom.iter_type<temporal>}
          } {loom.iter_type = #loom.iter_type<temporal>}
        } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @x}
      } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @y}
      return
    }
  }
  module attributes {loom.block_size_0 = -1 : index, loom.block_size_1 = -1 : index, loom.block_size_2 = -1 : index, loom.pass_name = "Materialize"} {
    func.func @matmul__d1i0_d0i0__f10__d_v__block_size_01__block_size_164__block_size_2128(%arg0: memref<4096x512xf32>, %arg1: memref<512x4096xf32>, %arg2: memref<4096x4096xf32>) {
      %cst = arith.constant 0.000000e+00 : f32
      %c64 = arith.constant 64 : index
      %c128 = arith.constant 128 : index
      affine.parallel (%arg3) = (0) to (8) {
        affine.parallel (%arg4) = (0) to (8) {
          affine.for %arg5 = 0 to 64 {
            affine.for %arg6 = 0 to 64 {
              %12 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 8 + d1 + d2 * 64)>(%arg3, %arg4, %arg6)
              %13 = loom.alloc [128, 64] on @L1 : memref<128x64xf32>
              %14 = loom.semaphore_take %13 : memref<128x64xf32> -> memref<128x64xf32>
              %15 = loom.alloc [1, 128] on @L1 : memref<1x128xf32>
              %16 = loom.semaphore_take %15 : memref<1x128xf32> -> memref<1x128xf32>
              %17 = loom.alloc [1, 64] on @L1 : memref<1x64xf32>
              %18 = loom.semaphore_take %17 : memref<1x64xf32> -> memref<1x64xf32>
              %19 = loom.init_tensor %18[1, 64] : memref<1x64xf32> -> tensor<1x64xf32>
              %20 = linalg.fill ins(%cst : f32) outs(%19 : tensor<1x64xf32>) -> tensor<1x64xf32>
              %21 = affine.for %arg7 = 0 to 4 iter_args(%arg8 = %20) -> (tensor<1x64xf32>) {
                %24 = arith.muli %arg7, %c128 : index
                %25 = affine.apply affine_map<(d0, d1) -> (d0 * 512 + d1)>(%12, %24)
                %reinterpret_cast_0 = memref.reinterpret_cast %arg0 to offset: [%25], sizes: [1, 128], strides: [512, 1] : memref<4096x512xf32> to memref<1x128xf32, strided<[512, 1], offset: ?>>
                %26 = loom.copy_to_tensor %reinterpret_cast_0, %16 on @L1, interconnect : [], broadcast : [1, 1] : memref<1x128xf32, strided<[512, 1], offset: ?>>, memref<1x128xf32> -> tensor<1x128xf32>
                %27 = arith.muli %arg7, %c128 : index
                %28 = arith.muli %arg5, %c64 : index
                %29 = affine.apply affine_map<(d0, d1) -> (d0 * 4096 + d1)>(%27, %28)
                %reinterpret_cast_1 = memref.reinterpret_cast %arg1 to offset: [%29], sizes: [128, 64], strides: [4096, 1] : memref<512x4096xf32> to memref<128x64xf32, strided<[4096, 1], offset: ?>>
                %30 = loom.copy_to_tensor %reinterpret_cast_1, %14 on @L1, interconnect : [@vertical_links], broadcast : [8, 1] : memref<128x64xf32, strided<[4096, 1], offset: ?>>, memref<128x64xf32> -> tensor<128x64xf32>
                %31 = linalg.matmul ins(%26, %30 : tensor<1x128xf32>, tensor<128x64xf32>) outs(%arg8 : tensor<1x64xf32>) -> tensor<1x64xf32>
                loom.semaphore_give %14 : memref<128x64xf32>
                loom.semaphore_give %16 : memref<1x128xf32>
                affine.yield %31 : tensor<1x64xf32>
              }
              %22 = arith.muli %arg5, %c64 : index
              %23 = affine.apply affine_map<(d0, d1) -> (d0 * 4096 + d1)>(%12, %22)
              %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%23], sizes: [1, 64], strides: [4096, 1] : memref<4096x4096xf32> to memref<1x64xf32, strided<[4096, 1], offset: ?>>
              loom.copy_from_tensor %21, %reinterpret_cast on @DRAM : tensor<1x64xf32>, memref<1x64xf32, strided<[4096, 1], offset: ?>>
              loom.semaphore_give %18 : memref<1x64xf32>
            } {loom.iter_type = #loom.iter_type<temporal>}
          } {loom.iter_type = #loom.iter_type<temporal>}
        } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @x}
      } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @y}
      return
    }
  }
  module attributes {loom.block_size_0 = -1 : index, loom.block_size_1 = -1 : index, loom.block_size_2 = -1 : index, loom.pass_name = "Materialize"} {
    func.func @matmul__d0i0_d1i1__f01__d_d__block_size_01__block_size_164__block_size_2128(%arg0: memref<4096x512xf32>, %arg1: memref<512x4096xf32>, %arg2: memref<4096x4096xf32>) {
      %cst = arith.constant 0.000000e+00 : f32
      %c64 = arith.constant 64 : index
      %c128 = arith.constant 128 : index
      affine.parallel (%arg3) = (0) to (8) {
        affine.parallel (%arg4) = (0) to (8) {
          affine.for %arg5 = 0 to 512 {
            affine.for %arg6 = 0 to 8 {
              %12 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 8)>(%arg3, %arg5)
              %13 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 8)>(%arg4, %arg6)
              %14 = loom.alloc [128, 64] on @L1 : memref<128x64xf32>
              %15 = loom.semaphore_take %14 : memref<128x64xf32> -> memref<128x64xf32>
              %16 = loom.alloc [1, 128] on @L1 : memref<1x128xf32>
              %17 = loom.semaphore_take %16 : memref<1x128xf32> -> memref<1x128xf32>
              %18 = loom.alloc [1, 64] on @L1 : memref<1x64xf32>
              %19 = loom.semaphore_take %18 : memref<1x64xf32> -> memref<1x64xf32>
              %20 = loom.init_tensor %19[1, 64] : memref<1x64xf32> -> tensor<1x64xf32>
              %21 = linalg.fill ins(%cst : f32) outs(%20 : tensor<1x64xf32>) -> tensor<1x64xf32>
              %22 = affine.for %arg7 = 0 to 4 iter_args(%arg8 = %21) -> (tensor<1x64xf32>) {
                %25 = arith.muli %arg7, %c128 : index
                %26 = affine.apply affine_map<(d0, d1) -> (d0 * 512 + d1)>(%12, %25)
                %reinterpret_cast_0 = memref.reinterpret_cast %arg0 to offset: [%26], sizes: [1, 128], strides: [512, 1] : memref<4096x512xf32> to memref<1x128xf32, strided<[512, 1], offset: ?>>
                %27 = loom.copy_to_tensor %reinterpret_cast_0, %17 on @L1, interconnect : [], broadcast : [1, 1] : memref<1x128xf32, strided<[512, 1], offset: ?>>, memref<1x128xf32> -> tensor<1x128xf32>
                %28 = arith.muli %arg7, %c128 : index
                %29 = arith.muli %13, %c64 : index
                %30 = affine.apply affine_map<(d0, d1) -> (d0 * 4096 + d1)>(%28, %29)
                %reinterpret_cast_1 = memref.reinterpret_cast %arg1 to offset: [%30], sizes: [128, 64], strides: [4096, 1] : memref<512x4096xf32> to memref<128x64xf32, strided<[4096, 1], offset: ?>>
                %31 = loom.copy_to_tensor %reinterpret_cast_1, %15 on @L1, interconnect : [], broadcast : [1, 1] : memref<128x64xf32, strided<[4096, 1], offset: ?>>, memref<128x64xf32> -> tensor<128x64xf32>
                %32 = linalg.matmul ins(%27, %31 : tensor<1x128xf32>, tensor<128x64xf32>) outs(%arg8 : tensor<1x64xf32>) -> tensor<1x64xf32>
                loom.semaphore_give %15 : memref<128x64xf32>
                loom.semaphore_give %17 : memref<1x128xf32>
                affine.yield %32 : tensor<1x64xf32>
              }
              %23 = arith.muli %13, %c64 : index
              %24 = affine.apply affine_map<(d0, d1) -> (d0 * 4096 + d1)>(%12, %23)
              %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%24], sizes: [1, 64], strides: [4096, 1] : memref<4096x4096xf32> to memref<1x64xf32, strided<[4096, 1], offset: ?>>
              loom.copy_from_tensor %22, %reinterpret_cast on @DRAM : tensor<1x64xf32>, memref<1x64xf32, strided<[4096, 1], offset: ?>>
              loom.semaphore_give %19 : memref<1x64xf32>
            } {loom.iter_type = #loom.iter_type<temporal>}
          } {loom.iter_type = #loom.iter_type<temporal>}
        } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @y}
      } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @x}
      return
    }
  }
  module attributes {loom.block_size_0 = -1 : index, loom.block_size_1 = -1 : index, loom.block_size_2 = -1 : index, loom.pass_name = "Materialize"} {
    func.func @matmul__d0i0_d1i1__f01__d_h__block_size_01__block_size_164__block_size_2128(%arg0: memref<4096x512xf32>, %arg1: memref<512x4096xf32>, %arg2: memref<4096x4096xf32>) {
      %cst = arith.constant 0.000000e+00 : f32
      %c64 = arith.constant 64 : index
      %c128 = arith.constant 128 : index
      affine.parallel (%arg3) = (0) to (8) {
        affine.parallel (%arg4) = (0) to (8) {
          affine.for %arg5 = 0 to 512 {
            affine.for %arg6 = 0 to 8 {
              %12 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 8)>(%arg3, %arg5)
              %13 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 8)>(%arg4, %arg6)
              %14 = loom.alloc [128, 64] on @L1 : memref<128x64xf32>
              %15 = loom.semaphore_take %14 : memref<128x64xf32> -> memref<128x64xf32>
              %16 = loom.alloc [1, 128] on @L1 : memref<1x128xf32>
              %17 = loom.semaphore_take %16 : memref<1x128xf32> -> memref<1x128xf32>
              %18 = loom.alloc [1, 64] on @L1 : memref<1x64xf32>
              %19 = loom.semaphore_take %18 : memref<1x64xf32> -> memref<1x64xf32>
              %20 = loom.init_tensor %19[1, 64] : memref<1x64xf32> -> tensor<1x64xf32>
              %21 = linalg.fill ins(%cst : f32) outs(%20 : tensor<1x64xf32>) -> tensor<1x64xf32>
              %22 = affine.for %arg7 = 0 to 4 iter_args(%arg8 = %21) -> (tensor<1x64xf32>) {
                %25 = arith.muli %arg7, %c128 : index
                %26 = affine.apply affine_map<(d0, d1) -> (d0 * 512 + d1)>(%12, %25)
                %reinterpret_cast_0 = memref.reinterpret_cast %arg0 to offset: [%26], sizes: [1, 128], strides: [512, 1] : memref<4096x512xf32> to memref<1x128xf32, strided<[512, 1], offset: ?>>
                %27 = loom.copy_to_tensor %reinterpret_cast_0, %17 on @L1, interconnect : [], broadcast : [1, 1] : memref<1x128xf32, strided<[512, 1], offset: ?>>, memref<1x128xf32> -> tensor<1x128xf32>
                %28 = arith.muli %arg7, %c128 : index
                %29 = arith.muli %13, %c64 : index
                %30 = affine.apply affine_map<(d0, d1) -> (d0 * 4096 + d1)>(%28, %29)
                %reinterpret_cast_1 = memref.reinterpret_cast %arg1 to offset: [%30], sizes: [128, 64], strides: [4096, 1] : memref<512x4096xf32> to memref<128x64xf32, strided<[4096, 1], offset: ?>>
                %31 = loom.copy_to_tensor %reinterpret_cast_1, %15 on @L1, interconnect : [@horizontal_links], broadcast : [1, 8] : memref<128x64xf32, strided<[4096, 1], offset: ?>>, memref<128x64xf32> -> tensor<128x64xf32>
                %32 = linalg.matmul ins(%27, %31 : tensor<1x128xf32>, tensor<128x64xf32>) outs(%arg8 : tensor<1x64xf32>) -> tensor<1x64xf32>
                loom.semaphore_give %15 : memref<128x64xf32>
                loom.semaphore_give %17 : memref<1x128xf32>
                affine.yield %32 : tensor<1x64xf32>
              }
              %23 = arith.muli %13, %c64 : index
              %24 = affine.apply affine_map<(d0, d1) -> (d0 * 4096 + d1)>(%12, %23)
              %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%24], sizes: [1, 64], strides: [4096, 1] : memref<4096x4096xf32> to memref<1x64xf32, strided<[4096, 1], offset: ?>>
              loom.copy_from_tensor %22, %reinterpret_cast on @DRAM : tensor<1x64xf32>, memref<1x64xf32, strided<[4096, 1], offset: ?>>
              loom.semaphore_give %19 : memref<1x64xf32>
            } {loom.iter_type = #loom.iter_type<temporal>}
          } {loom.iter_type = #loom.iter_type<temporal>}
        } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @y}
      } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @x}
      return
    }
  }
  module attributes {loom.block_size_0 = -1 : index, loom.block_size_1 = -1 : index, loom.block_size_2 = -1 : index, loom.pass_name = "Materialize"} {
    func.func @matmul__d0i0_d1i1__f01__v_d__block_size_01__block_size_164__block_size_2128(%arg0: memref<4096x512xf32>, %arg1: memref<512x4096xf32>, %arg2: memref<4096x4096xf32>) {
      %cst = arith.constant 0.000000e+00 : f32
      %c64 = arith.constant 64 : index
      %c128 = arith.constant 128 : index
      affine.parallel (%arg3) = (0) to (8) {
        affine.parallel (%arg4) = (0) to (8) {
          affine.for %arg5 = 0 to 512 {
            affine.for %arg6 = 0 to 8 {
              %12 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 8)>(%arg3, %arg5)
              %13 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 8)>(%arg4, %arg6)
              %14 = loom.alloc [128, 64] on @L1 : memref<128x64xf32>
              %15 = loom.semaphore_take %14 : memref<128x64xf32> -> memref<128x64xf32>
              %16 = loom.alloc [1, 128] on @L1 : memref<1x128xf32>
              %17 = loom.semaphore_take %16 : memref<1x128xf32> -> memref<1x128xf32>
              %18 = loom.alloc [1, 64] on @L1 : memref<1x64xf32>
              %19 = loom.semaphore_take %18 : memref<1x64xf32> -> memref<1x64xf32>
              %20 = loom.init_tensor %19[1, 64] : memref<1x64xf32> -> tensor<1x64xf32>
              %21 = linalg.fill ins(%cst : f32) outs(%20 : tensor<1x64xf32>) -> tensor<1x64xf32>
              %22 = affine.for %arg7 = 0 to 4 iter_args(%arg8 = %21) -> (tensor<1x64xf32>) {
                %25 = arith.muli %arg7, %c128 : index
                %26 = affine.apply affine_map<(d0, d1) -> (d0 * 512 + d1)>(%12, %25)
                %reinterpret_cast_0 = memref.reinterpret_cast %arg0 to offset: [%26], sizes: [1, 128], strides: [512, 1] : memref<4096x512xf32> to memref<1x128xf32, strided<[512, 1], offset: ?>>
                %27 = loom.copy_to_tensor %reinterpret_cast_0, %17 on @L1, interconnect : [@vertical_links], broadcast : [8, 1] : memref<1x128xf32, strided<[512, 1], offset: ?>>, memref<1x128xf32> -> tensor<1x128xf32>
                %28 = arith.muli %arg7, %c128 : index
                %29 = arith.muli %13, %c64 : index
                %30 = affine.apply affine_map<(d0, d1) -> (d0 * 4096 + d1)>(%28, %29)
                %reinterpret_cast_1 = memref.reinterpret_cast %arg1 to offset: [%30], sizes: [128, 64], strides: [4096, 1] : memref<512x4096xf32> to memref<128x64xf32, strided<[4096, 1], offset: ?>>
                %31 = loom.copy_to_tensor %reinterpret_cast_1, %15 on @L1, interconnect : [], broadcast : [1, 1] : memref<128x64xf32, strided<[4096, 1], offset: ?>>, memref<128x64xf32> -> tensor<128x64xf32>
                %32 = linalg.matmul ins(%27, %31 : tensor<1x128xf32>, tensor<128x64xf32>) outs(%arg8 : tensor<1x64xf32>) -> tensor<1x64xf32>
                loom.semaphore_give %15 : memref<128x64xf32>
                loom.semaphore_give %17 : memref<1x128xf32>
                affine.yield %32 : tensor<1x64xf32>
              }
              %23 = arith.muli %13, %c64 : index
              %24 = affine.apply affine_map<(d0, d1) -> (d0 * 4096 + d1)>(%12, %23)
              %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%24], sizes: [1, 64], strides: [4096, 1] : memref<4096x4096xf32> to memref<1x64xf32, strided<[4096, 1], offset: ?>>
              loom.copy_from_tensor %22, %reinterpret_cast on @DRAM : tensor<1x64xf32>, memref<1x64xf32, strided<[4096, 1], offset: ?>>
              loom.semaphore_give %19 : memref<1x64xf32>
            } {loom.iter_type = #loom.iter_type<temporal>}
          } {loom.iter_type = #loom.iter_type<temporal>}
        } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @y}
      } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @x}
      return
    }
  }
  module attributes {loom.block_size_0 = -1 : index, loom.block_size_1 = -1 : index, loom.block_size_2 = -1 : index, loom.pass_name = "Materialize"} {
    func.func @matmul__d0i0_d1i1__f01__v_h__block_size_01__block_size_164__block_size_2128(%arg0: memref<4096x512xf32>, %arg1: memref<512x4096xf32>, %arg2: memref<4096x4096xf32>) {
      %cst = arith.constant 0.000000e+00 : f32
      %c64 = arith.constant 64 : index
      %c128 = arith.constant 128 : index
      affine.parallel (%arg3) = (0) to (8) {
        affine.parallel (%arg4) = (0) to (8) {
          affine.for %arg5 = 0 to 512 {
            affine.for %arg6 = 0 to 8 {
              %12 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 8)>(%arg3, %arg5)
              %13 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 8)>(%arg4, %arg6)
              %14 = loom.alloc [128, 64] on @L1 : memref<128x64xf32>
              %15 = loom.semaphore_take %14 : memref<128x64xf32> -> memref<128x64xf32>
              %16 = loom.alloc [1, 128] on @L1 : memref<1x128xf32>
              %17 = loom.semaphore_take %16 : memref<1x128xf32> -> memref<1x128xf32>
              %18 = loom.alloc [1, 64] on @L1 : memref<1x64xf32>
              %19 = loom.semaphore_take %18 : memref<1x64xf32> -> memref<1x64xf32>
              %20 = loom.init_tensor %19[1, 64] : memref<1x64xf32> -> tensor<1x64xf32>
              %21 = linalg.fill ins(%cst : f32) outs(%20 : tensor<1x64xf32>) -> tensor<1x64xf32>
              %22 = affine.for %arg7 = 0 to 4 iter_args(%arg8 = %21) -> (tensor<1x64xf32>) {
                %25 = arith.muli %arg7, %c128 : index
                %26 = affine.apply affine_map<(d0, d1) -> (d0 * 512 + d1)>(%12, %25)
                %reinterpret_cast_0 = memref.reinterpret_cast %arg0 to offset: [%26], sizes: [1, 128], strides: [512, 1] : memref<4096x512xf32> to memref<1x128xf32, strided<[512, 1], offset: ?>>
                %27 = loom.copy_to_tensor %reinterpret_cast_0, %17 on @L1, interconnect : [@vertical_links], broadcast : [8, 1] : memref<1x128xf32, strided<[512, 1], offset: ?>>, memref<1x128xf32> -> tensor<1x128xf32>
                %28 = arith.muli %arg7, %c128 : index
                %29 = arith.muli %13, %c64 : index
                %30 = affine.apply affine_map<(d0, d1) -> (d0 * 4096 + d1)>(%28, %29)
                %reinterpret_cast_1 = memref.reinterpret_cast %arg1 to offset: [%30], sizes: [128, 64], strides: [4096, 1] : memref<512x4096xf32> to memref<128x64xf32, strided<[4096, 1], offset: ?>>
                %31 = loom.copy_to_tensor %reinterpret_cast_1, %15 on @L1, interconnect : [@horizontal_links], broadcast : [1, 8] : memref<128x64xf32, strided<[4096, 1], offset: ?>>, memref<128x64xf32> -> tensor<128x64xf32>
                %32 = linalg.matmul ins(%27, %31 : tensor<1x128xf32>, tensor<128x64xf32>) outs(%arg8 : tensor<1x64xf32>) -> tensor<1x64xf32>
                loom.semaphore_give %15 : memref<128x64xf32>
                loom.semaphore_give %17 : memref<1x128xf32>
                affine.yield %32 : tensor<1x64xf32>
              }
              %23 = arith.muli %13, %c64 : index
              %24 = affine.apply affine_map<(d0, d1) -> (d0 * 4096 + d1)>(%12, %23)
              %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%24], sizes: [1, 64], strides: [4096, 1] : memref<4096x4096xf32> to memref<1x64xf32, strided<[4096, 1], offset: ?>>
              loom.copy_from_tensor %22, %reinterpret_cast on @DRAM : tensor<1x64xf32>, memref<1x64xf32, strided<[4096, 1], offset: ?>>
              loom.semaphore_give %19 : memref<1x64xf32>
            } {loom.iter_type = #loom.iter_type<temporal>}
          } {loom.iter_type = #loom.iter_type<temporal>}
        } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @y}
      } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @x}
      return
    }
  }
  module attributes {loom.block_size_0 = -1 : index, loom.block_size_1 = -1 : index, loom.block_size_2 = -1 : index, loom.pass_name = "Materialize"} {
    func.func @matmul__d0i0_d1i1__f10__d_d__block_size_01__block_size_164__block_size_2128(%arg0: memref<4096x512xf32>, %arg1: memref<512x4096xf32>, %arg2: memref<4096x4096xf32>) {
      %cst = arith.constant 0.000000e+00 : f32
      %c64 = arith.constant 64 : index
      %c128 = arith.constant 128 : index
      affine.parallel (%arg3) = (0) to (8) {
        affine.parallel (%arg4) = (0) to (8) {
          affine.for %arg5 = 0 to 8 {
            affine.for %arg6 = 0 to 512 {
              %12 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 8)>(%arg3, %arg6)
              %13 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 8)>(%arg4, %arg5)
              %14 = loom.alloc [128, 64] on @L1 : memref<128x64xf32>
              %15 = loom.semaphore_take %14 : memref<128x64xf32> -> memref<128x64xf32>
              %16 = loom.alloc [1, 128] on @L1 : memref<1x128xf32>
              %17 = loom.semaphore_take %16 : memref<1x128xf32> -> memref<1x128xf32>
              %18 = loom.alloc [1, 64] on @L1 : memref<1x64xf32>
              %19 = loom.semaphore_take %18 : memref<1x64xf32> -> memref<1x64xf32>
              %20 = loom.init_tensor %19[1, 64] : memref<1x64xf32> -> tensor<1x64xf32>
              %21 = linalg.fill ins(%cst : f32) outs(%20 : tensor<1x64xf32>) -> tensor<1x64xf32>
              %22 = affine.for %arg7 = 0 to 4 iter_args(%arg8 = %21) -> (tensor<1x64xf32>) {
                %25 = arith.muli %arg7, %c128 : index
                %26 = affine.apply affine_map<(d0, d1) -> (d0 * 512 + d1)>(%12, %25)
                %reinterpret_cast_0 = memref.reinterpret_cast %arg0 to offset: [%26], sizes: [1, 128], strides: [512, 1] : memref<4096x512xf32> to memref<1x128xf32, strided<[512, 1], offset: ?>>
                %27 = loom.copy_to_tensor %reinterpret_cast_0, %17 on @L1, interconnect : [], broadcast : [1, 1] : memref<1x128xf32, strided<[512, 1], offset: ?>>, memref<1x128xf32> -> tensor<1x128xf32>
                %28 = arith.muli %arg7, %c128 : index
                %29 = arith.muli %13, %c64 : index
                %30 = affine.apply affine_map<(d0, d1) -> (d0 * 4096 + d1)>(%28, %29)
                %reinterpret_cast_1 = memref.reinterpret_cast %arg1 to offset: [%30], sizes: [128, 64], strides: [4096, 1] : memref<512x4096xf32> to memref<128x64xf32, strided<[4096, 1], offset: ?>>
                %31 = loom.copy_to_tensor %reinterpret_cast_1, %15 on @L1, interconnect : [], broadcast : [1, 1] : memref<128x64xf32, strided<[4096, 1], offset: ?>>, memref<128x64xf32> -> tensor<128x64xf32>
                %32 = linalg.matmul ins(%27, %31 : tensor<1x128xf32>, tensor<128x64xf32>) outs(%arg8 : tensor<1x64xf32>) -> tensor<1x64xf32>
                loom.semaphore_give %15 : memref<128x64xf32>
                loom.semaphore_give %17 : memref<1x128xf32>
                affine.yield %32 : tensor<1x64xf32>
              }
              %23 = arith.muli %13, %c64 : index
              %24 = affine.apply affine_map<(d0, d1) -> (d0 * 4096 + d1)>(%12, %23)
              %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%24], sizes: [1, 64], strides: [4096, 1] : memref<4096x4096xf32> to memref<1x64xf32, strided<[4096, 1], offset: ?>>
              loom.copy_from_tensor %22, %reinterpret_cast on @DRAM : tensor<1x64xf32>, memref<1x64xf32, strided<[4096, 1], offset: ?>>
              loom.semaphore_give %19 : memref<1x64xf32>
            } {loom.iter_type = #loom.iter_type<temporal>}
          } {loom.iter_type = #loom.iter_type<temporal>}
        } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @y}
      } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @x}
      return
    }
  }
  module attributes {loom.block_size_0 = -1 : index, loom.block_size_1 = -1 : index, loom.block_size_2 = -1 : index, loom.pass_name = "Materialize"} {
    func.func @matmul__d0i0_d1i1__f10__d_h__block_size_01__block_size_164__block_size_2128(%arg0: memref<4096x512xf32>, %arg1: memref<512x4096xf32>, %arg2: memref<4096x4096xf32>) {
      %cst = arith.constant 0.000000e+00 : f32
      %c64 = arith.constant 64 : index
      %c128 = arith.constant 128 : index
      affine.parallel (%arg3) = (0) to (8) {
        affine.parallel (%arg4) = (0) to (8) {
          affine.for %arg5 = 0 to 8 {
            affine.for %arg6 = 0 to 512 {
              %12 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 8)>(%arg3, %arg6)
              %13 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 8)>(%arg4, %arg5)
              %14 = loom.alloc [128, 64] on @L1 : memref<128x64xf32>
              %15 = loom.semaphore_take %14 : memref<128x64xf32> -> memref<128x64xf32>
              %16 = loom.alloc [1, 128] on @L1 : memref<1x128xf32>
              %17 = loom.semaphore_take %16 : memref<1x128xf32> -> memref<1x128xf32>
              %18 = loom.alloc [1, 64] on @L1 : memref<1x64xf32>
              %19 = loom.semaphore_take %18 : memref<1x64xf32> -> memref<1x64xf32>
              %20 = loom.init_tensor %19[1, 64] : memref<1x64xf32> -> tensor<1x64xf32>
              %21 = linalg.fill ins(%cst : f32) outs(%20 : tensor<1x64xf32>) -> tensor<1x64xf32>
              %22 = affine.for %arg7 = 0 to 4 iter_args(%arg8 = %21) -> (tensor<1x64xf32>) {
                %25 = arith.muli %arg7, %c128 : index
                %26 = affine.apply affine_map<(d0, d1) -> (d0 * 512 + d1)>(%12, %25)
                %reinterpret_cast_0 = memref.reinterpret_cast %arg0 to offset: [%26], sizes: [1, 128], strides: [512, 1] : memref<4096x512xf32> to memref<1x128xf32, strided<[512, 1], offset: ?>>
                %27 = loom.copy_to_tensor %reinterpret_cast_0, %17 on @L1, interconnect : [], broadcast : [1, 1] : memref<1x128xf32, strided<[512, 1], offset: ?>>, memref<1x128xf32> -> tensor<1x128xf32>
                %28 = arith.muli %arg7, %c128 : index
                %29 = arith.muli %13, %c64 : index
                %30 = affine.apply affine_map<(d0, d1) -> (d0 * 4096 + d1)>(%28, %29)
                %reinterpret_cast_1 = memref.reinterpret_cast %arg1 to offset: [%30], sizes: [128, 64], strides: [4096, 1] : memref<512x4096xf32> to memref<128x64xf32, strided<[4096, 1], offset: ?>>
                %31 = loom.copy_to_tensor %reinterpret_cast_1, %15 on @L1, interconnect : [@horizontal_links], broadcast : [1, 8] : memref<128x64xf32, strided<[4096, 1], offset: ?>>, memref<128x64xf32> -> tensor<128x64xf32>
                %32 = linalg.matmul ins(%27, %31 : tensor<1x128xf32>, tensor<128x64xf32>) outs(%arg8 : tensor<1x64xf32>) -> tensor<1x64xf32>
                loom.semaphore_give %15 : memref<128x64xf32>
                loom.semaphore_give %17 : memref<1x128xf32>
                affine.yield %32 : tensor<1x64xf32>
              }
              %23 = arith.muli %13, %c64 : index
              %24 = affine.apply affine_map<(d0, d1) -> (d0 * 4096 + d1)>(%12, %23)
              %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%24], sizes: [1, 64], strides: [4096, 1] : memref<4096x4096xf32> to memref<1x64xf32, strided<[4096, 1], offset: ?>>
              loom.copy_from_tensor %22, %reinterpret_cast on @DRAM : tensor<1x64xf32>, memref<1x64xf32, strided<[4096, 1], offset: ?>>
              loom.semaphore_give %19 : memref<1x64xf32>
            } {loom.iter_type = #loom.iter_type<temporal>}
          } {loom.iter_type = #loom.iter_type<temporal>}
        } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @y}
      } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @x}
      return
    }
  }
  module attributes {loom.block_size_0 = -1 : index, loom.block_size_1 = -1 : index, loom.block_size_2 = -1 : index, loom.pass_name = "Materialize"} {
    func.func @matmul__d0i0_d1i1__f10__v_d__block_size_01__block_size_164__block_size_2128(%arg0: memref<4096x512xf32>, %arg1: memref<512x4096xf32>, %arg2: memref<4096x4096xf32>) {
      %cst = arith.constant 0.000000e+00 : f32
      %c64 = arith.constant 64 : index
      %c128 = arith.constant 128 : index
      affine.parallel (%arg3) = (0) to (8) {
        affine.parallel (%arg4) = (0) to (8) {
          affine.for %arg5 = 0 to 8 {
            affine.for %arg6 = 0 to 512 {
              %12 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 8)>(%arg3, %arg6)
              %13 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 8)>(%arg4, %arg5)
              %14 = loom.alloc [128, 64] on @L1 : memref<128x64xf32>
              %15 = loom.semaphore_take %14 : memref<128x64xf32> -> memref<128x64xf32>
              %16 = loom.alloc [1, 128] on @L1 : memref<1x128xf32>
              %17 = loom.semaphore_take %16 : memref<1x128xf32> -> memref<1x128xf32>
              %18 = loom.alloc [1, 64] on @L1 : memref<1x64xf32>
              %19 = loom.semaphore_take %18 : memref<1x64xf32> -> memref<1x64xf32>
              %20 = loom.init_tensor %19[1, 64] : memref<1x64xf32> -> tensor<1x64xf32>
              %21 = linalg.fill ins(%cst : f32) outs(%20 : tensor<1x64xf32>) -> tensor<1x64xf32>
              %22 = affine.for %arg7 = 0 to 4 iter_args(%arg8 = %21) -> (tensor<1x64xf32>) {
                %25 = arith.muli %arg7, %c128 : index
                %26 = affine.apply affine_map<(d0, d1) -> (d0 * 512 + d1)>(%12, %25)
                %reinterpret_cast_0 = memref.reinterpret_cast %arg0 to offset: [%26], sizes: [1, 128], strides: [512, 1] : memref<4096x512xf32> to memref<1x128xf32, strided<[512, 1], offset: ?>>
                %27 = loom.copy_to_tensor %reinterpret_cast_0, %17 on @L1, interconnect : [@vertical_links], broadcast : [8, 1] : memref<1x128xf32, strided<[512, 1], offset: ?>>, memref<1x128xf32> -> tensor<1x128xf32>
                %28 = arith.muli %arg7, %c128 : index
                %29 = arith.muli %13, %c64 : index
                %30 = affine.apply affine_map<(d0, d1) -> (d0 * 4096 + d1)>(%28, %29)
                %reinterpret_cast_1 = memref.reinterpret_cast %arg1 to offset: [%30], sizes: [128, 64], strides: [4096, 1] : memref<512x4096xf32> to memref<128x64xf32, strided<[4096, 1], offset: ?>>
                %31 = loom.copy_to_tensor %reinterpret_cast_1, %15 on @L1, interconnect : [], broadcast : [1, 1] : memref<128x64xf32, strided<[4096, 1], offset: ?>>, memref<128x64xf32> -> tensor<128x64xf32>
                %32 = linalg.matmul ins(%27, %31 : tensor<1x128xf32>, tensor<128x64xf32>) outs(%arg8 : tensor<1x64xf32>) -> tensor<1x64xf32>
                loom.semaphore_give %15 : memref<128x64xf32>
                loom.semaphore_give %17 : memref<1x128xf32>
                affine.yield %32 : tensor<1x64xf32>
              }
              %23 = arith.muli %13, %c64 : index
              %24 = affine.apply affine_map<(d0, d1) -> (d0 * 4096 + d1)>(%12, %23)
              %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%24], sizes: [1, 64], strides: [4096, 1] : memref<4096x4096xf32> to memref<1x64xf32, strided<[4096, 1], offset: ?>>
              loom.copy_from_tensor %22, %reinterpret_cast on @DRAM : tensor<1x64xf32>, memref<1x64xf32, strided<[4096, 1], offset: ?>>
              loom.semaphore_give %19 : memref<1x64xf32>
            } {loom.iter_type = #loom.iter_type<temporal>}
          } {loom.iter_type = #loom.iter_type<temporal>}
        } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @y}
      } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @x}
      return
    }
  }
  module attributes {loom.block_size_0 = -1 : index, loom.block_size_1 = -1 : index, loom.block_size_2 = -1 : index, loom.pass_name = "Materialize"} {
    func.func @matmul__d0i0_d1i1__f10__v_h__block_size_01__block_size_164__block_size_2128(%arg0: memref<4096x512xf32>, %arg1: memref<512x4096xf32>, %arg2: memref<4096x4096xf32>) {
      %cst = arith.constant 0.000000e+00 : f32
      %c64 = arith.constant 64 : index
      %c128 = arith.constant 128 : index
      affine.parallel (%arg3) = (0) to (8) {
        affine.parallel (%arg4) = (0) to (8) {
          affine.for %arg5 = 0 to 8 {
            affine.for %arg6 = 0 to 512 {
              %12 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 8)>(%arg3, %arg6)
              %13 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 8)>(%arg4, %arg5)
              %14 = loom.alloc [128, 64] on @L1 : memref<128x64xf32>
              %15 = loom.semaphore_take %14 : memref<128x64xf32> -> memref<128x64xf32>
              %16 = loom.alloc [1, 128] on @L1 : memref<1x128xf32>
              %17 = loom.semaphore_take %16 : memref<1x128xf32> -> memref<1x128xf32>
              %18 = loom.alloc [1, 64] on @L1 : memref<1x64xf32>
              %19 = loom.semaphore_take %18 : memref<1x64xf32> -> memref<1x64xf32>
              %20 = loom.init_tensor %19[1, 64] : memref<1x64xf32> -> tensor<1x64xf32>
              %21 = linalg.fill ins(%cst : f32) outs(%20 : tensor<1x64xf32>) -> tensor<1x64xf32>
              %22 = affine.for %arg7 = 0 to 4 iter_args(%arg8 = %21) -> (tensor<1x64xf32>) {
                %25 = arith.muli %arg7, %c128 : index
                %26 = affine.apply affine_map<(d0, d1) -> (d0 * 512 + d1)>(%12, %25)
                %reinterpret_cast_0 = memref.reinterpret_cast %arg0 to offset: [%26], sizes: [1, 128], strides: [512, 1] : memref<4096x512xf32> to memref<1x128xf32, strided<[512, 1], offset: ?>>
                %27 = loom.copy_to_tensor %reinterpret_cast_0, %17 on @L1, interconnect : [@vertical_links], broadcast : [8, 1] : memref<1x128xf32, strided<[512, 1], offset: ?>>, memref<1x128xf32> -> tensor<1x128xf32>
                %28 = arith.muli %arg7, %c128 : index
                %29 = arith.muli %13, %c64 : index
                %30 = affine.apply affine_map<(d0, d1) -> (d0 * 4096 + d1)>(%28, %29)
                %reinterpret_cast_1 = memref.reinterpret_cast %arg1 to offset: [%30], sizes: [128, 64], strides: [4096, 1] : memref<512x4096xf32> to memref<128x64xf32, strided<[4096, 1], offset: ?>>
                %31 = loom.copy_to_tensor %reinterpret_cast_1, %15 on @L1, interconnect : [@horizontal_links], broadcast : [1, 8] : memref<128x64xf32, strided<[4096, 1], offset: ?>>, memref<128x64xf32> -> tensor<128x64xf32>
                %32 = linalg.matmul ins(%27, %31 : tensor<1x128xf32>, tensor<128x64xf32>) outs(%arg8 : tensor<1x64xf32>) -> tensor<1x64xf32>
                loom.semaphore_give %15 : memref<128x64xf32>
                loom.semaphore_give %17 : memref<1x128xf32>
                affine.yield %32 : tensor<1x64xf32>
              }
              %23 = arith.muli %13, %c64 : index
              %24 = affine.apply affine_map<(d0, d1) -> (d0 * 4096 + d1)>(%12, %23)
              %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%24], sizes: [1, 64], strides: [4096, 1] : memref<4096x4096xf32> to memref<1x64xf32, strided<[4096, 1], offset: ?>>
              loom.copy_from_tensor %22, %reinterpret_cast on @DRAM : tensor<1x64xf32>, memref<1x64xf32, strided<[4096, 1], offset: ?>>
              loom.semaphore_give %19 : memref<1x64xf32>
            } {loom.iter_type = #loom.iter_type<temporal>}
          } {loom.iter_type = #loom.iter_type<temporal>}
        } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @y}
      } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @x}
      return
    }
  }
  module attributes {loom.block_size_0 = -1 : index, loom.block_size_1 = -1 : index, loom.block_size_2 = -1 : index, loom.pass_name = "Materialize"} {
    func.func @matmul__d1i0_d0i1__f01__d_d__block_size_01__block_size_164__block_size_2128(%arg0: memref<4096x512xf32>, %arg1: memref<512x4096xf32>, %arg2: memref<4096x4096xf32>) {
      %cst = arith.constant 0.000000e+00 : f32
      %c64 = arith.constant 64 : index
      %c128 = arith.constant 128 : index
      affine.parallel (%arg3) = (0) to (8) {
        affine.parallel (%arg4) = (0) to (8) {
          affine.for %arg5 = 0 to 512 {
            affine.for %arg6 = 0 to 8 {
              %12 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 8)>(%arg3, %arg5)
              %13 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 8)>(%arg4, %arg6)
              %14 = loom.alloc [128, 64] on @L1 : memref<128x64xf32>
              %15 = loom.semaphore_take %14 : memref<128x64xf32> -> memref<128x64xf32>
              %16 = loom.alloc [1, 128] on @L1 : memref<1x128xf32>
              %17 = loom.semaphore_take %16 : memref<1x128xf32> -> memref<1x128xf32>
              %18 = loom.alloc [1, 64] on @L1 : memref<1x64xf32>
              %19 = loom.semaphore_take %18 : memref<1x64xf32> -> memref<1x64xf32>
              %20 = loom.init_tensor %19[1, 64] : memref<1x64xf32> -> tensor<1x64xf32>
              %21 = linalg.fill ins(%cst : f32) outs(%20 : tensor<1x64xf32>) -> tensor<1x64xf32>
              %22 = affine.for %arg7 = 0 to 4 iter_args(%arg8 = %21) -> (tensor<1x64xf32>) {
                %25 = arith.muli %arg7, %c128 : index
                %26 = affine.apply affine_map<(d0, d1) -> (d0 * 512 + d1)>(%12, %25)
                %reinterpret_cast_0 = memref.reinterpret_cast %arg0 to offset: [%26], sizes: [1, 128], strides: [512, 1] : memref<4096x512xf32> to memref<1x128xf32, strided<[512, 1], offset: ?>>
                %27 = loom.copy_to_tensor %reinterpret_cast_0, %17 on @L1, interconnect : [], broadcast : [1, 1] : memref<1x128xf32, strided<[512, 1], offset: ?>>, memref<1x128xf32> -> tensor<1x128xf32>
                %28 = arith.muli %arg7, %c128 : index
                %29 = arith.muli %13, %c64 : index
                %30 = affine.apply affine_map<(d0, d1) -> (d0 * 4096 + d1)>(%28, %29)
                %reinterpret_cast_1 = memref.reinterpret_cast %arg1 to offset: [%30], sizes: [128, 64], strides: [4096, 1] : memref<512x4096xf32> to memref<128x64xf32, strided<[4096, 1], offset: ?>>
                %31 = loom.copy_to_tensor %reinterpret_cast_1, %15 on @L1, interconnect : [], broadcast : [1, 1] : memref<128x64xf32, strided<[4096, 1], offset: ?>>, memref<128x64xf32> -> tensor<128x64xf32>
                %32 = linalg.matmul ins(%27, %31 : tensor<1x128xf32>, tensor<128x64xf32>) outs(%arg8 : tensor<1x64xf32>) -> tensor<1x64xf32>
                loom.semaphore_give %15 : memref<128x64xf32>
                loom.semaphore_give %17 : memref<1x128xf32>
                affine.yield %32 : tensor<1x64xf32>
              }
              %23 = arith.muli %13, %c64 : index
              %24 = affine.apply affine_map<(d0, d1) -> (d0 * 4096 + d1)>(%12, %23)
              %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%24], sizes: [1, 64], strides: [4096, 1] : memref<4096x4096xf32> to memref<1x64xf32, strided<[4096, 1], offset: ?>>
              loom.copy_from_tensor %22, %reinterpret_cast on @DRAM : tensor<1x64xf32>, memref<1x64xf32, strided<[4096, 1], offset: ?>>
              loom.semaphore_give %19 : memref<1x64xf32>
            } {loom.iter_type = #loom.iter_type<temporal>}
          } {loom.iter_type = #loom.iter_type<temporal>}
        } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @x}
      } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @y}
      return
    }
  }
  module attributes {loom.block_size_0 = -1 : index, loom.block_size_1 = -1 : index, loom.block_size_2 = -1 : index, loom.pass_name = "Materialize"} {
    func.func @matmul__d1i0_d0i1__f01__d_v__block_size_01__block_size_164__block_size_2128(%arg0: memref<4096x512xf32>, %arg1: memref<512x4096xf32>, %arg2: memref<4096x4096xf32>) {
      %cst = arith.constant 0.000000e+00 : f32
      %c64 = arith.constant 64 : index
      %c128 = arith.constant 128 : index
      affine.parallel (%arg3) = (0) to (8) {
        affine.parallel (%arg4) = (0) to (8) {
          affine.for %arg5 = 0 to 512 {
            affine.for %arg6 = 0 to 8 {
              %12 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 8)>(%arg3, %arg5)
              %13 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 8)>(%arg4, %arg6)
              %14 = loom.alloc [128, 64] on @L1 : memref<128x64xf32>
              %15 = loom.semaphore_take %14 : memref<128x64xf32> -> memref<128x64xf32>
              %16 = loom.alloc [1, 128] on @L1 : memref<1x128xf32>
              %17 = loom.semaphore_take %16 : memref<1x128xf32> -> memref<1x128xf32>
              %18 = loom.alloc [1, 64] on @L1 : memref<1x64xf32>
              %19 = loom.semaphore_take %18 : memref<1x64xf32> -> memref<1x64xf32>
              %20 = loom.init_tensor %19[1, 64] : memref<1x64xf32> -> tensor<1x64xf32>
              %21 = linalg.fill ins(%cst : f32) outs(%20 : tensor<1x64xf32>) -> tensor<1x64xf32>
              %22 = affine.for %arg7 = 0 to 4 iter_args(%arg8 = %21) -> (tensor<1x64xf32>) {
                %25 = arith.muli %arg7, %c128 : index
                %26 = affine.apply affine_map<(d0, d1) -> (d0 * 512 + d1)>(%12, %25)
                %reinterpret_cast_0 = memref.reinterpret_cast %arg0 to offset: [%26], sizes: [1, 128], strides: [512, 1] : memref<4096x512xf32> to memref<1x128xf32, strided<[512, 1], offset: ?>>
                %27 = loom.copy_to_tensor %reinterpret_cast_0, %17 on @L1, interconnect : [], broadcast : [1, 1] : memref<1x128xf32, strided<[512, 1], offset: ?>>, memref<1x128xf32> -> tensor<1x128xf32>
                %28 = arith.muli %arg7, %c128 : index
                %29 = arith.muli %13, %c64 : index
                %30 = affine.apply affine_map<(d0, d1) -> (d0 * 4096 + d1)>(%28, %29)
                %reinterpret_cast_1 = memref.reinterpret_cast %arg1 to offset: [%30], sizes: [128, 64], strides: [4096, 1] : memref<512x4096xf32> to memref<128x64xf32, strided<[4096, 1], offset: ?>>
                %31 = loom.copy_to_tensor %reinterpret_cast_1, %15 on @L1, interconnect : [@vertical_links], broadcast : [8, 1] : memref<128x64xf32, strided<[4096, 1], offset: ?>>, memref<128x64xf32> -> tensor<128x64xf32>
                %32 = linalg.matmul ins(%27, %31 : tensor<1x128xf32>, tensor<128x64xf32>) outs(%arg8 : tensor<1x64xf32>) -> tensor<1x64xf32>
                loom.semaphore_give %15 : memref<128x64xf32>
                loom.semaphore_give %17 : memref<1x128xf32>
                affine.yield %32 : tensor<1x64xf32>
              }
              %23 = arith.muli %13, %c64 : index
              %24 = affine.apply affine_map<(d0, d1) -> (d0 * 4096 + d1)>(%12, %23)
              %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%24], sizes: [1, 64], strides: [4096, 1] : memref<4096x4096xf32> to memref<1x64xf32, strided<[4096, 1], offset: ?>>
              loom.copy_from_tensor %22, %reinterpret_cast on @DRAM : tensor<1x64xf32>, memref<1x64xf32, strided<[4096, 1], offset: ?>>
              loom.semaphore_give %19 : memref<1x64xf32>
            } {loom.iter_type = #loom.iter_type<temporal>}
          } {loom.iter_type = #loom.iter_type<temporal>}
        } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @x}
      } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @y}
      return
    }
  }
  module attributes {loom.block_size_0 = -1 : index, loom.block_size_1 = -1 : index, loom.block_size_2 = -1 : index, loom.pass_name = "Materialize"} {
    func.func @matmul__d1i0_d0i1__f01__h_d__block_size_01__block_size_164__block_size_2128(%arg0: memref<4096x512xf32>, %arg1: memref<512x4096xf32>, %arg2: memref<4096x4096xf32>) {
      %cst = arith.constant 0.000000e+00 : f32
      %c64 = arith.constant 64 : index
      %c128 = arith.constant 128 : index
      affine.parallel (%arg3) = (0) to (8) {
        affine.parallel (%arg4) = (0) to (8) {
          affine.for %arg5 = 0 to 512 {
            affine.for %arg6 = 0 to 8 {
              %12 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 8)>(%arg3, %arg5)
              %13 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 8)>(%arg4, %arg6)
              %14 = loom.alloc [128, 64] on @L1 : memref<128x64xf32>
              %15 = loom.semaphore_take %14 : memref<128x64xf32> -> memref<128x64xf32>
              %16 = loom.alloc [1, 128] on @L1 : memref<1x128xf32>
              %17 = loom.semaphore_take %16 : memref<1x128xf32> -> memref<1x128xf32>
              %18 = loom.alloc [1, 64] on @L1 : memref<1x64xf32>
              %19 = loom.semaphore_take %18 : memref<1x64xf32> -> memref<1x64xf32>
              %20 = loom.init_tensor %19[1, 64] : memref<1x64xf32> -> tensor<1x64xf32>
              %21 = linalg.fill ins(%cst : f32) outs(%20 : tensor<1x64xf32>) -> tensor<1x64xf32>
              %22 = affine.for %arg7 = 0 to 4 iter_args(%arg8 = %21) -> (tensor<1x64xf32>) {
                %25 = arith.muli %arg7, %c128 : index
                %26 = affine.apply affine_map<(d0, d1) -> (d0 * 512 + d1)>(%12, %25)
                %reinterpret_cast_0 = memref.reinterpret_cast %arg0 to offset: [%26], sizes: [1, 128], strides: [512, 1] : memref<4096x512xf32> to memref<1x128xf32, strided<[512, 1], offset: ?>>
                %27 = loom.copy_to_tensor %reinterpret_cast_0, %17 on @L1, interconnect : [@horizontal_links], broadcast : [1, 8] : memref<1x128xf32, strided<[512, 1], offset: ?>>, memref<1x128xf32> -> tensor<1x128xf32>
                %28 = arith.muli %arg7, %c128 : index
                %29 = arith.muli %13, %c64 : index
                %30 = affine.apply affine_map<(d0, d1) -> (d0 * 4096 + d1)>(%28, %29)
                %reinterpret_cast_1 = memref.reinterpret_cast %arg1 to offset: [%30], sizes: [128, 64], strides: [4096, 1] : memref<512x4096xf32> to memref<128x64xf32, strided<[4096, 1], offset: ?>>
                %31 = loom.copy_to_tensor %reinterpret_cast_1, %15 on @L1, interconnect : [], broadcast : [1, 1] : memref<128x64xf32, strided<[4096, 1], offset: ?>>, memref<128x64xf32> -> tensor<128x64xf32>
                %32 = linalg.matmul ins(%27, %31 : tensor<1x128xf32>, tensor<128x64xf32>) outs(%arg8 : tensor<1x64xf32>) -> tensor<1x64xf32>
                loom.semaphore_give %15 : memref<128x64xf32>
                loom.semaphore_give %17 : memref<1x128xf32>
                affine.yield %32 : tensor<1x64xf32>
              }
              %23 = arith.muli %13, %c64 : index
              %24 = affine.apply affine_map<(d0, d1) -> (d0 * 4096 + d1)>(%12, %23)
              %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%24], sizes: [1, 64], strides: [4096, 1] : memref<4096x4096xf32> to memref<1x64xf32, strided<[4096, 1], offset: ?>>
              loom.copy_from_tensor %22, %reinterpret_cast on @DRAM : tensor<1x64xf32>, memref<1x64xf32, strided<[4096, 1], offset: ?>>
              loom.semaphore_give %19 : memref<1x64xf32>
            } {loom.iter_type = #loom.iter_type<temporal>}
          } {loom.iter_type = #loom.iter_type<temporal>}
        } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @x}
      } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @y}
      return
    }
  }
  module attributes {loom.block_size_0 = -1 : index, loom.block_size_1 = -1 : index, loom.block_size_2 = -1 : index, loom.pass_name = "Materialize"} {
    func.func @matmul__d1i0_d0i1__f01__h_v__block_size_01__block_size_164__block_size_2128(%arg0: memref<4096x512xf32>, %arg1: memref<512x4096xf32>, %arg2: memref<4096x4096xf32>) {
      %cst = arith.constant 0.000000e+00 : f32
      %c64 = arith.constant 64 : index
      %c128 = arith.constant 128 : index
      affine.parallel (%arg3) = (0) to (8) {
        affine.parallel (%arg4) = (0) to (8) {
          affine.for %arg5 = 0 to 512 {
            affine.for %arg6 = 0 to 8 {
              %12 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 8)>(%arg3, %arg5)
              %13 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 8)>(%arg4, %arg6)
              %14 = loom.alloc [128, 64] on @L1 : memref<128x64xf32>
              %15 = loom.semaphore_take %14 : memref<128x64xf32> -> memref<128x64xf32>
              %16 = loom.alloc [1, 128] on @L1 : memref<1x128xf32>
              %17 = loom.semaphore_take %16 : memref<1x128xf32> -> memref<1x128xf32>
              %18 = loom.alloc [1, 64] on @L1 : memref<1x64xf32>
              %19 = loom.semaphore_take %18 : memref<1x64xf32> -> memref<1x64xf32>
              %20 = loom.init_tensor %19[1, 64] : memref<1x64xf32> -> tensor<1x64xf32>
              %21 = linalg.fill ins(%cst : f32) outs(%20 : tensor<1x64xf32>) -> tensor<1x64xf32>
              %22 = affine.for %arg7 = 0 to 4 iter_args(%arg8 = %21) -> (tensor<1x64xf32>) {
                %25 = arith.muli %arg7, %c128 : index
                %26 = affine.apply affine_map<(d0, d1) -> (d0 * 512 + d1)>(%12, %25)
                %reinterpret_cast_0 = memref.reinterpret_cast %arg0 to offset: [%26], sizes: [1, 128], strides: [512, 1] : memref<4096x512xf32> to memref<1x128xf32, strided<[512, 1], offset: ?>>
                %27 = loom.copy_to_tensor %reinterpret_cast_0, %17 on @L1, interconnect : [@horizontal_links], broadcast : [1, 8] : memref<1x128xf32, strided<[512, 1], offset: ?>>, memref<1x128xf32> -> tensor<1x128xf32>
                %28 = arith.muli %arg7, %c128 : index
                %29 = arith.muli %13, %c64 : index
                %30 = affine.apply affine_map<(d0, d1) -> (d0 * 4096 + d1)>(%28, %29)
                %reinterpret_cast_1 = memref.reinterpret_cast %arg1 to offset: [%30], sizes: [128, 64], strides: [4096, 1] : memref<512x4096xf32> to memref<128x64xf32, strided<[4096, 1], offset: ?>>
                %31 = loom.copy_to_tensor %reinterpret_cast_1, %15 on @L1, interconnect : [@vertical_links], broadcast : [8, 1] : memref<128x64xf32, strided<[4096, 1], offset: ?>>, memref<128x64xf32> -> tensor<128x64xf32>
                %32 = linalg.matmul ins(%27, %31 : tensor<1x128xf32>, tensor<128x64xf32>) outs(%arg8 : tensor<1x64xf32>) -> tensor<1x64xf32>
                loom.semaphore_give %15 : memref<128x64xf32>
                loom.semaphore_give %17 : memref<1x128xf32>
                affine.yield %32 : tensor<1x64xf32>
              }
              %23 = arith.muli %13, %c64 : index
              %24 = affine.apply affine_map<(d0, d1) -> (d0 * 4096 + d1)>(%12, %23)
              %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%24], sizes: [1, 64], strides: [4096, 1] : memref<4096x4096xf32> to memref<1x64xf32, strided<[4096, 1], offset: ?>>
              loom.copy_from_tensor %22, %reinterpret_cast on @DRAM : tensor<1x64xf32>, memref<1x64xf32, strided<[4096, 1], offset: ?>>
              loom.semaphore_give %19 : memref<1x64xf32>
            } {loom.iter_type = #loom.iter_type<temporal>}
          } {loom.iter_type = #loom.iter_type<temporal>}
        } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @x}
      } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @y}
      return
    }
  }
  module attributes {loom.block_size_0 = -1 : index, loom.block_size_1 = -1 : index, loom.block_size_2 = -1 : index, loom.pass_name = "Materialize"} {
    func.func @matmul__d1i0_d0i1__f10__d_d__block_size_01__block_size_164__block_size_2128(%arg0: memref<4096x512xf32>, %arg1: memref<512x4096xf32>, %arg2: memref<4096x4096xf32>) {
      %cst = arith.constant 0.000000e+00 : f32
      %c64 = arith.constant 64 : index
      %c128 = arith.constant 128 : index
      affine.parallel (%arg3) = (0) to (8) {
        affine.parallel (%arg4) = (0) to (8) {
          affine.for %arg5 = 0 to 8 {
            affine.for %arg6 = 0 to 512 {
              %12 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 8)>(%arg3, %arg6)
              %13 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 8)>(%arg4, %arg5)
              %14 = loom.alloc [128, 64] on @L1 : memref<128x64xf32>
              %15 = loom.semaphore_take %14 : memref<128x64xf32> -> memref<128x64xf32>
              %16 = loom.alloc [1, 128] on @L1 : memref<1x128xf32>
              %17 = loom.semaphore_take %16 : memref<1x128xf32> -> memref<1x128xf32>
              %18 = loom.alloc [1, 64] on @L1 : memref<1x64xf32>
              %19 = loom.semaphore_take %18 : memref<1x64xf32> -> memref<1x64xf32>
              %20 = loom.init_tensor %19[1, 64] : memref<1x64xf32> -> tensor<1x64xf32>
              %21 = linalg.fill ins(%cst : f32) outs(%20 : tensor<1x64xf32>) -> tensor<1x64xf32>
              %22 = affine.for %arg7 = 0 to 4 iter_args(%arg8 = %21) -> (tensor<1x64xf32>) {
                %25 = arith.muli %arg7, %c128 : index
                %26 = affine.apply affine_map<(d0, d1) -> (d0 * 512 + d1)>(%12, %25)
                %reinterpret_cast_0 = memref.reinterpret_cast %arg0 to offset: [%26], sizes: [1, 128], strides: [512, 1] : memref<4096x512xf32> to memref<1x128xf32, strided<[512, 1], offset: ?>>
                %27 = loom.copy_to_tensor %reinterpret_cast_0, %17 on @L1, interconnect : [], broadcast : [1, 1] : memref<1x128xf32, strided<[512, 1], offset: ?>>, memref<1x128xf32> -> tensor<1x128xf32>
                %28 = arith.muli %arg7, %c128 : index
                %29 = arith.muli %13, %c64 : index
                %30 = affine.apply affine_map<(d0, d1) -> (d0 * 4096 + d1)>(%28, %29)
                %reinterpret_cast_1 = memref.reinterpret_cast %arg1 to offset: [%30], sizes: [128, 64], strides: [4096, 1] : memref<512x4096xf32> to memref<128x64xf32, strided<[4096, 1], offset: ?>>
                %31 = loom.copy_to_tensor %reinterpret_cast_1, %15 on @L1, interconnect : [], broadcast : [1, 1] : memref<128x64xf32, strided<[4096, 1], offset: ?>>, memref<128x64xf32> -> tensor<128x64xf32>
                %32 = linalg.matmul ins(%27, %31 : tensor<1x128xf32>, tensor<128x64xf32>) outs(%arg8 : tensor<1x64xf32>) -> tensor<1x64xf32>
                loom.semaphore_give %15 : memref<128x64xf32>
                loom.semaphore_give %17 : memref<1x128xf32>
                affine.yield %32 : tensor<1x64xf32>
              }
              %23 = arith.muli %13, %c64 : index
              %24 = affine.apply affine_map<(d0, d1) -> (d0 * 4096 + d1)>(%12, %23)
              %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%24], sizes: [1, 64], strides: [4096, 1] : memref<4096x4096xf32> to memref<1x64xf32, strided<[4096, 1], offset: ?>>
              loom.copy_from_tensor %22, %reinterpret_cast on @DRAM : tensor<1x64xf32>, memref<1x64xf32, strided<[4096, 1], offset: ?>>
              loom.semaphore_give %19 : memref<1x64xf32>
            } {loom.iter_type = #loom.iter_type<temporal>}
          } {loom.iter_type = #loom.iter_type<temporal>}
        } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @x}
      } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @y}
      return
    }
  }
  module attributes {loom.block_size_0 = -1 : index, loom.block_size_1 = -1 : index, loom.block_size_2 = -1 : index, loom.pass_name = "Materialize"} {
    func.func @matmul__d1i0_d0i1__f10__d_v__block_size_01__block_size_164__block_size_2128(%arg0: memref<4096x512xf32>, %arg1: memref<512x4096xf32>, %arg2: memref<4096x4096xf32>) {
      %cst = arith.constant 0.000000e+00 : f32
      %c64 = arith.constant 64 : index
      %c128 = arith.constant 128 : index
      affine.parallel (%arg3) = (0) to (8) {
        affine.parallel (%arg4) = (0) to (8) {
          affine.for %arg5 = 0 to 8 {
            affine.for %arg6 = 0 to 512 {
              %12 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 8)>(%arg3, %arg6)
              %13 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 8)>(%arg4, %arg5)
              %14 = loom.alloc [128, 64] on @L1 : memref<128x64xf32>
              %15 = loom.semaphore_take %14 : memref<128x64xf32> -> memref<128x64xf32>
              %16 = loom.alloc [1, 128] on @L1 : memref<1x128xf32>
              %17 = loom.semaphore_take %16 : memref<1x128xf32> -> memref<1x128xf32>
              %18 = loom.alloc [1, 64] on @L1 : memref<1x64xf32>
              %19 = loom.semaphore_take %18 : memref<1x64xf32> -> memref<1x64xf32>
              %20 = loom.init_tensor %19[1, 64] : memref<1x64xf32> -> tensor<1x64xf32>
              %21 = linalg.fill ins(%cst : f32) outs(%20 : tensor<1x64xf32>) -> tensor<1x64xf32>
              %22 = affine.for %arg7 = 0 to 4 iter_args(%arg8 = %21) -> (tensor<1x64xf32>) {
                %25 = arith.muli %arg7, %c128 : index
                %26 = affine.apply affine_map<(d0, d1) -> (d0 * 512 + d1)>(%12, %25)
                %reinterpret_cast_0 = memref.reinterpret_cast %arg0 to offset: [%26], sizes: [1, 128], strides: [512, 1] : memref<4096x512xf32> to memref<1x128xf32, strided<[512, 1], offset: ?>>
                %27 = loom.copy_to_tensor %reinterpret_cast_0, %17 on @L1, interconnect : [], broadcast : [1, 1] : memref<1x128xf32, strided<[512, 1], offset: ?>>, memref<1x128xf32> -> tensor<1x128xf32>
                %28 = arith.muli %arg7, %c128 : index
                %29 = arith.muli %13, %c64 : index
                %30 = affine.apply affine_map<(d0, d1) -> (d0 * 4096 + d1)>(%28, %29)
                %reinterpret_cast_1 = memref.reinterpret_cast %arg1 to offset: [%30], sizes: [128, 64], strides: [4096, 1] : memref<512x4096xf32> to memref<128x64xf32, strided<[4096, 1], offset: ?>>
                %31 = loom.copy_to_tensor %reinterpret_cast_1, %15 on @L1, interconnect : [@vertical_links], broadcast : [8, 1] : memref<128x64xf32, strided<[4096, 1], offset: ?>>, memref<128x64xf32> -> tensor<128x64xf32>
                %32 = linalg.matmul ins(%27, %31 : tensor<1x128xf32>, tensor<128x64xf32>) outs(%arg8 : tensor<1x64xf32>) -> tensor<1x64xf32>
                loom.semaphore_give %15 : memref<128x64xf32>
                loom.semaphore_give %17 : memref<1x128xf32>
                affine.yield %32 : tensor<1x64xf32>
              }
              %23 = arith.muli %13, %c64 : index
              %24 = affine.apply affine_map<(d0, d1) -> (d0 * 4096 + d1)>(%12, %23)
              %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%24], sizes: [1, 64], strides: [4096, 1] : memref<4096x4096xf32> to memref<1x64xf32, strided<[4096, 1], offset: ?>>
              loom.copy_from_tensor %22, %reinterpret_cast on @DRAM : tensor<1x64xf32>, memref<1x64xf32, strided<[4096, 1], offset: ?>>
              loom.semaphore_give %19 : memref<1x64xf32>
            } {loom.iter_type = #loom.iter_type<temporal>}
          } {loom.iter_type = #loom.iter_type<temporal>}
        } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @x}
      } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @y}
      return
    }
  }
  module attributes {loom.block_size_0 = -1 : index, loom.block_size_1 = -1 : index, loom.block_size_2 = -1 : index, loom.pass_name = "Materialize"} {
    func.func @matmul__d1i0_d0i1__f10__h_d__block_size_01__block_size_164__block_size_2128(%arg0: memref<4096x512xf32>, %arg1: memref<512x4096xf32>, %arg2: memref<4096x4096xf32>) {
      %cst = arith.constant 0.000000e+00 : f32
      %c64 = arith.constant 64 : index
      %c128 = arith.constant 128 : index
      affine.parallel (%arg3) = (0) to (8) {
        affine.parallel (%arg4) = (0) to (8) {
          affine.for %arg5 = 0 to 8 {
            affine.for %arg6 = 0 to 512 {
              %12 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 8)>(%arg3, %arg6)
              %13 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 8)>(%arg4, %arg5)
              %14 = loom.alloc [128, 64] on @L1 : memref<128x64xf32>
              %15 = loom.semaphore_take %14 : memref<128x64xf32> -> memref<128x64xf32>
              %16 = loom.alloc [1, 128] on @L1 : memref<1x128xf32>
              %17 = loom.semaphore_take %16 : memref<1x128xf32> -> memref<1x128xf32>
              %18 = loom.alloc [1, 64] on @L1 : memref<1x64xf32>
              %19 = loom.semaphore_take %18 : memref<1x64xf32> -> memref<1x64xf32>
              %20 = loom.init_tensor %19[1, 64] : memref<1x64xf32> -> tensor<1x64xf32>
              %21 = linalg.fill ins(%cst : f32) outs(%20 : tensor<1x64xf32>) -> tensor<1x64xf32>
              %22 = affine.for %arg7 = 0 to 4 iter_args(%arg8 = %21) -> (tensor<1x64xf32>) {
                %25 = arith.muli %arg7, %c128 : index
                %26 = affine.apply affine_map<(d0, d1) -> (d0 * 512 + d1)>(%12, %25)
                %reinterpret_cast_0 = memref.reinterpret_cast %arg0 to offset: [%26], sizes: [1, 128], strides: [512, 1] : memref<4096x512xf32> to memref<1x128xf32, strided<[512, 1], offset: ?>>
                %27 = loom.copy_to_tensor %reinterpret_cast_0, %17 on @L1, interconnect : [@horizontal_links], broadcast : [1, 8] : memref<1x128xf32, strided<[512, 1], offset: ?>>, memref<1x128xf32> -> tensor<1x128xf32>
                %28 = arith.muli %arg7, %c128 : index
                %29 = arith.muli %13, %c64 : index
                %30 = affine.apply affine_map<(d0, d1) -> (d0 * 4096 + d1)>(%28, %29)
                %reinterpret_cast_1 = memref.reinterpret_cast %arg1 to offset: [%30], sizes: [128, 64], strides: [4096, 1] : memref<512x4096xf32> to memref<128x64xf32, strided<[4096, 1], offset: ?>>
                %31 = loom.copy_to_tensor %reinterpret_cast_1, %15 on @L1, interconnect : [], broadcast : [1, 1] : memref<128x64xf32, strided<[4096, 1], offset: ?>>, memref<128x64xf32> -> tensor<128x64xf32>
                %32 = linalg.matmul ins(%27, %31 : tensor<1x128xf32>, tensor<128x64xf32>) outs(%arg8 : tensor<1x64xf32>) -> tensor<1x64xf32>
                loom.semaphore_give %15 : memref<128x64xf32>
                loom.semaphore_give %17 : memref<1x128xf32>
                affine.yield %32 : tensor<1x64xf32>
              }
              %23 = arith.muli %13, %c64 : index
              %24 = affine.apply affine_map<(d0, d1) -> (d0 * 4096 + d1)>(%12, %23)
              %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%24], sizes: [1, 64], strides: [4096, 1] : memref<4096x4096xf32> to memref<1x64xf32, strided<[4096, 1], offset: ?>>
              loom.copy_from_tensor %22, %reinterpret_cast on @DRAM : tensor<1x64xf32>, memref<1x64xf32, strided<[4096, 1], offset: ?>>
              loom.semaphore_give %19 : memref<1x64xf32>
            } {loom.iter_type = #loom.iter_type<temporal>}
          } {loom.iter_type = #loom.iter_type<temporal>}
        } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @x}
      } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @y}
      return
    }
  }
  module attributes {loom.block_size_0 = -1 : index, loom.block_size_1 = -1 : index, loom.block_size_2 = -1 : index, loom.pass_name = "Materialize"} {
    func.func @matmul__d1i0_d0i1__f10__h_v__block_size_01__block_size_164__block_size_2128(%arg0: memref<4096x512xf32>, %arg1: memref<512x4096xf32>, %arg2: memref<4096x4096xf32>) {
      %cst = arith.constant 0.000000e+00 : f32
      %c64 = arith.constant 64 : index
      %c128 = arith.constant 128 : index
      affine.parallel (%arg3) = (0) to (8) {
        affine.parallel (%arg4) = (0) to (8) {
          affine.for %arg5 = 0 to 8 {
            affine.for %arg6 = 0 to 512 {
              %12 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 8)>(%arg3, %arg6)
              %13 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 8)>(%arg4, %arg5)
              %14 = loom.alloc [128, 64] on @L1 : memref<128x64xf32>
              %15 = loom.semaphore_take %14 : memref<128x64xf32> -> memref<128x64xf32>
              %16 = loom.alloc [1, 128] on @L1 : memref<1x128xf32>
              %17 = loom.semaphore_take %16 : memref<1x128xf32> -> memref<1x128xf32>
              %18 = loom.alloc [1, 64] on @L1 : memref<1x64xf32>
              %19 = loom.semaphore_take %18 : memref<1x64xf32> -> memref<1x64xf32>
              %20 = loom.init_tensor %19[1, 64] : memref<1x64xf32> -> tensor<1x64xf32>
              %21 = linalg.fill ins(%cst : f32) outs(%20 : tensor<1x64xf32>) -> tensor<1x64xf32>
              %22 = affine.for %arg7 = 0 to 4 iter_args(%arg8 = %21) -> (tensor<1x64xf32>) {
                %25 = arith.muli %arg7, %c128 : index
                %26 = affine.apply affine_map<(d0, d1) -> (d0 * 512 + d1)>(%12, %25)
                %reinterpret_cast_0 = memref.reinterpret_cast %arg0 to offset: [%26], sizes: [1, 128], strides: [512, 1] : memref<4096x512xf32> to memref<1x128xf32, strided<[512, 1], offset: ?>>
                %27 = loom.copy_to_tensor %reinterpret_cast_0, %17 on @L1, interconnect : [@horizontal_links], broadcast : [1, 8] : memref<1x128xf32, strided<[512, 1], offset: ?>>, memref<1x128xf32> -> tensor<1x128xf32>
                %28 = arith.muli %arg7, %c128 : index
                %29 = arith.muli %13, %c64 : index
                %30 = affine.apply affine_map<(d0, d1) -> (d0 * 4096 + d1)>(%28, %29)
                %reinterpret_cast_1 = memref.reinterpret_cast %arg1 to offset: [%30], sizes: [128, 64], strides: [4096, 1] : memref<512x4096xf32> to memref<128x64xf32, strided<[4096, 1], offset: ?>>
                %31 = loom.copy_to_tensor %reinterpret_cast_1, %15 on @L1, interconnect : [@vertical_links], broadcast : [8, 1] : memref<128x64xf32, strided<[4096, 1], offset: ?>>, memref<128x64xf32> -> tensor<128x64xf32>
                %32 = linalg.matmul ins(%27, %31 : tensor<1x128xf32>, tensor<128x64xf32>) outs(%arg8 : tensor<1x64xf32>) -> tensor<1x64xf32>
                loom.semaphore_give %15 : memref<128x64xf32>
                loom.semaphore_give %17 : memref<1x128xf32>
                affine.yield %32 : tensor<1x64xf32>
              }
              %23 = arith.muli %13, %c64 : index
              %24 = affine.apply affine_map<(d0, d1) -> (d0 * 4096 + d1)>(%12, %23)
              %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%24], sizes: [1, 64], strides: [4096, 1] : memref<4096x4096xf32> to memref<1x64xf32, strided<[4096, 1], offset: ?>>
              loom.copy_from_tensor %22, %reinterpret_cast on @DRAM : tensor<1x64xf32>, memref<1x64xf32, strided<[4096, 1], offset: ?>>
              loom.semaphore_give %19 : memref<1x64xf32>
            } {loom.iter_type = #loom.iter_type<temporal>}
          } {loom.iter_type = #loom.iter_type<temporal>}
        } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @x}
      } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @y}
      return
    }
  }
  module attributes {loom.block_size_0 = -1 : index, loom.block_size_1 = -1 : index, loom.block_size_2 = -1 : index, loom.pass_name = "Materialize"} {
    func.func @matmul__d0i1_d1i1__f01__d_d__block_size_01__block_size_164__block_size_2128(%arg0: memref<4096x512xf32>, %arg1: memref<512x4096xf32>, %arg2: memref<4096x4096xf32>) {
      %cst = arith.constant 0.000000e+00 : f32
      %c64 = arith.constant 64 : index
      %c128 = arith.constant 128 : index
      affine.parallel (%arg3) = (0) to (8) {
        affine.parallel (%arg4) = (0) to (8) {
          affine.for %arg5 = 0 to 4096 {
            affine.for %arg6 = 0 to 1 {
              %12 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 8 + d1 + d2 * 64)>(%arg3, %arg4, %arg6)
              %13 = loom.alloc [128, 64] on @L1 : memref<128x64xf32>
              %14 = loom.semaphore_take %13 : memref<128x64xf32> -> memref<128x64xf32>
              %15 = loom.alloc [1, 128] on @L1 : memref<1x128xf32>
              %16 = loom.semaphore_take %15 : memref<1x128xf32> -> memref<1x128xf32>
              %17 = loom.alloc [1, 64] on @L1 : memref<1x64xf32>
              %18 = loom.semaphore_take %17 : memref<1x64xf32> -> memref<1x64xf32>
              %19 = loom.init_tensor %18[1, 64] : memref<1x64xf32> -> tensor<1x64xf32>
              %20 = linalg.fill ins(%cst : f32) outs(%19 : tensor<1x64xf32>) -> tensor<1x64xf32>
              %21 = affine.for %arg7 = 0 to 4 iter_args(%arg8 = %20) -> (tensor<1x64xf32>) {
                %24 = arith.muli %arg7, %c128 : index
                %25 = affine.apply affine_map<(d0, d1) -> (d0 * 512 + d1)>(%arg5, %24)
                %reinterpret_cast_0 = memref.reinterpret_cast %arg0 to offset: [%25], sizes: [1, 128], strides: [512, 1] : memref<4096x512xf32> to memref<1x128xf32, strided<[512, 1], offset: ?>>
                %26 = loom.copy_to_tensor %reinterpret_cast_0, %16 on @L1, interconnect : [], broadcast : [1, 1] : memref<1x128xf32, strided<[512, 1], offset: ?>>, memref<1x128xf32> -> tensor<1x128xf32>
                %27 = arith.muli %arg7, %c128 : index
                %28 = arith.muli %12, %c64 : index
                %29 = affine.apply affine_map<(d0, d1) -> (d0 * 4096 + d1)>(%27, %28)
                %reinterpret_cast_1 = memref.reinterpret_cast %arg1 to offset: [%29], sizes: [128, 64], strides: [4096, 1] : memref<512x4096xf32> to memref<128x64xf32, strided<[4096, 1], offset: ?>>
                %30 = loom.copy_to_tensor %reinterpret_cast_1, %14 on @L1, interconnect : [], broadcast : [1, 1] : memref<128x64xf32, strided<[4096, 1], offset: ?>>, memref<128x64xf32> -> tensor<128x64xf32>
                %31 = linalg.matmul ins(%26, %30 : tensor<1x128xf32>, tensor<128x64xf32>) outs(%arg8 : tensor<1x64xf32>) -> tensor<1x64xf32>
                loom.semaphore_give %14 : memref<128x64xf32>
                loom.semaphore_give %16 : memref<1x128xf32>
                affine.yield %31 : tensor<1x64xf32>
              }
              %22 = arith.muli %12, %c64 : index
              %23 = affine.apply affine_map<(d0, d1) -> (d0 * 4096 + d1)>(%arg5, %22)
              %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%23], sizes: [1, 64], strides: [4096, 1] : memref<4096x4096xf32> to memref<1x64xf32, strided<[4096, 1], offset: ?>>
              loom.copy_from_tensor %21, %reinterpret_cast on @DRAM : tensor<1x64xf32>, memref<1x64xf32, strided<[4096, 1], offset: ?>>
              loom.semaphore_give %18 : memref<1x64xf32>
            } {loom.iter_type = #loom.iter_type<temporal>}
          } {loom.iter_type = #loom.iter_type<temporal>}
        } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @y}
      } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @x}
      return
    }
  }
  module attributes {loom.block_size_0 = -1 : index, loom.block_size_1 = -1 : index, loom.block_size_2 = -1 : index, loom.pass_name = "Materialize"} {
    func.func @matmul__d0i1_d1i1__f01__a_d__block_size_01__block_size_164__block_size_2128(%arg0: memref<4096x512xf32>, %arg1: memref<512x4096xf32>, %arg2: memref<4096x4096xf32>) {
      %cst = arith.constant 0.000000e+00 : f32
      %c64 = arith.constant 64 : index
      %c128 = arith.constant 128 : index
      affine.parallel (%arg3) = (0) to (8) {
        affine.parallel (%arg4) = (0) to (8) {
          affine.for %arg5 = 0 to 4096 {
            affine.for %arg6 = 0 to 1 {
              %12 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 8 + d1 + d2 * 64)>(%arg3, %arg4, %arg6)
              %13 = loom.alloc [128, 64] on @L1 : memref<128x64xf32>
              %14 = loom.semaphore_take %13 : memref<128x64xf32> -> memref<128x64xf32>
              %15 = loom.alloc [1, 128] on @L1 : memref<1x128xf32>
              %16 = loom.semaphore_take %15 : memref<1x128xf32> -> memref<1x128xf32>
              %17 = loom.alloc [1, 64] on @L1 : memref<1x64xf32>
              %18 = loom.semaphore_take %17 : memref<1x64xf32> -> memref<1x64xf32>
              %19 = loom.init_tensor %18[1, 64] : memref<1x64xf32> -> tensor<1x64xf32>
              %20 = linalg.fill ins(%cst : f32) outs(%19 : tensor<1x64xf32>) -> tensor<1x64xf32>
              %21 = affine.for %arg7 = 0 to 4 iter_args(%arg8 = %20) -> (tensor<1x64xf32>) {
                %24 = arith.muli %arg7, %c128 : index
                %25 = affine.apply affine_map<(d0, d1) -> (d0 * 512 + d1)>(%arg5, %24)
                %reinterpret_cast_0 = memref.reinterpret_cast %arg0 to offset: [%25], sizes: [1, 128], strides: [512, 1] : memref<4096x512xf32> to memref<1x128xf32, strided<[512, 1], offset: ?>>
                %26 = loom.copy_to_tensor %reinterpret_cast_0, %16 on @L1, interconnect : [@horizontal_links, @vertical_links], broadcast : [8, 8] : memref<1x128xf32, strided<[512, 1], offset: ?>>, memref<1x128xf32> -> tensor<1x128xf32>
                %27 = arith.muli %arg7, %c128 : index
                %28 = arith.muli %12, %c64 : index
                %29 = affine.apply affine_map<(d0, d1) -> (d0 * 4096 + d1)>(%27, %28)
                %reinterpret_cast_1 = memref.reinterpret_cast %arg1 to offset: [%29], sizes: [128, 64], strides: [4096, 1] : memref<512x4096xf32> to memref<128x64xf32, strided<[4096, 1], offset: ?>>
                %30 = loom.copy_to_tensor %reinterpret_cast_1, %14 on @L1, interconnect : [], broadcast : [1, 1] : memref<128x64xf32, strided<[4096, 1], offset: ?>>, memref<128x64xf32> -> tensor<128x64xf32>
                %31 = linalg.matmul ins(%26, %30 : tensor<1x128xf32>, tensor<128x64xf32>) outs(%arg8 : tensor<1x64xf32>) -> tensor<1x64xf32>
                loom.semaphore_give %14 : memref<128x64xf32>
                loom.semaphore_give %16 : memref<1x128xf32>
                affine.yield %31 : tensor<1x64xf32>
              }
              %22 = arith.muli %12, %c64 : index
              %23 = affine.apply affine_map<(d0, d1) -> (d0 * 4096 + d1)>(%arg5, %22)
              %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%23], sizes: [1, 64], strides: [4096, 1] : memref<4096x4096xf32> to memref<1x64xf32, strided<[4096, 1], offset: ?>>
              loom.copy_from_tensor %21, %reinterpret_cast on @DRAM : tensor<1x64xf32>, memref<1x64xf32, strided<[4096, 1], offset: ?>>
              loom.semaphore_give %18 : memref<1x64xf32>
            } {loom.iter_type = #loom.iter_type<temporal>}
          } {loom.iter_type = #loom.iter_type<temporal>}
        } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @y}
      } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @x}
      return
    }
  }
  module attributes {loom.block_size_0 = -1 : index, loom.block_size_1 = -1 : index, loom.block_size_2 = -1 : index, loom.pass_name = "Materialize"} {
    func.func @matmul__d0i1_d1i1__f01__h_d__block_size_01__block_size_164__block_size_2128(%arg0: memref<4096x512xf32>, %arg1: memref<512x4096xf32>, %arg2: memref<4096x4096xf32>) {
      %cst = arith.constant 0.000000e+00 : f32
      %c64 = arith.constant 64 : index
      %c128 = arith.constant 128 : index
      affine.parallel (%arg3) = (0) to (8) {
        affine.parallel (%arg4) = (0) to (8) {
          affine.for %arg5 = 0 to 4096 {
            affine.for %arg6 = 0 to 1 {
              %12 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 8 + d1 + d2 * 64)>(%arg3, %arg4, %arg6)
              %13 = loom.alloc [128, 64] on @L1 : memref<128x64xf32>
              %14 = loom.semaphore_take %13 : memref<128x64xf32> -> memref<128x64xf32>
              %15 = loom.alloc [1, 128] on @L1 : memref<1x128xf32>
              %16 = loom.semaphore_take %15 : memref<1x128xf32> -> memref<1x128xf32>
              %17 = loom.alloc [1, 64] on @L1 : memref<1x64xf32>
              %18 = loom.semaphore_take %17 : memref<1x64xf32> -> memref<1x64xf32>
              %19 = loom.init_tensor %18[1, 64] : memref<1x64xf32> -> tensor<1x64xf32>
              %20 = linalg.fill ins(%cst : f32) outs(%19 : tensor<1x64xf32>) -> tensor<1x64xf32>
              %21 = affine.for %arg7 = 0 to 4 iter_args(%arg8 = %20) -> (tensor<1x64xf32>) {
                %24 = arith.muli %arg7, %c128 : index
                %25 = affine.apply affine_map<(d0, d1) -> (d0 * 512 + d1)>(%arg5, %24)
                %reinterpret_cast_0 = memref.reinterpret_cast %arg0 to offset: [%25], sizes: [1, 128], strides: [512, 1] : memref<4096x512xf32> to memref<1x128xf32, strided<[512, 1], offset: ?>>
                %26 = loom.copy_to_tensor %reinterpret_cast_0, %16 on @L1, interconnect : [@horizontal_links], broadcast : [1, 8] : memref<1x128xf32, strided<[512, 1], offset: ?>>, memref<1x128xf32> -> tensor<1x128xf32>
                %27 = arith.muli %arg7, %c128 : index
                %28 = arith.muli %12, %c64 : index
                %29 = affine.apply affine_map<(d0, d1) -> (d0 * 4096 + d1)>(%27, %28)
                %reinterpret_cast_1 = memref.reinterpret_cast %arg1 to offset: [%29], sizes: [128, 64], strides: [4096, 1] : memref<512x4096xf32> to memref<128x64xf32, strided<[4096, 1], offset: ?>>
                %30 = loom.copy_to_tensor %reinterpret_cast_1, %14 on @L1, interconnect : [], broadcast : [1, 1] : memref<128x64xf32, strided<[4096, 1], offset: ?>>, memref<128x64xf32> -> tensor<128x64xf32>
                %31 = linalg.matmul ins(%26, %30 : tensor<1x128xf32>, tensor<128x64xf32>) outs(%arg8 : tensor<1x64xf32>) -> tensor<1x64xf32>
                loom.semaphore_give %14 : memref<128x64xf32>
                loom.semaphore_give %16 : memref<1x128xf32>
                affine.yield %31 : tensor<1x64xf32>
              }
              %22 = arith.muli %12, %c64 : index
              %23 = affine.apply affine_map<(d0, d1) -> (d0 * 4096 + d1)>(%arg5, %22)
              %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%23], sizes: [1, 64], strides: [4096, 1] : memref<4096x4096xf32> to memref<1x64xf32, strided<[4096, 1], offset: ?>>
              loom.copy_from_tensor %21, %reinterpret_cast on @DRAM : tensor<1x64xf32>, memref<1x64xf32, strided<[4096, 1], offset: ?>>
              loom.semaphore_give %18 : memref<1x64xf32>
            } {loom.iter_type = #loom.iter_type<temporal>}
          } {loom.iter_type = #loom.iter_type<temporal>}
        } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @y}
      } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @x}
      return
    }
  }
  module attributes {loom.block_size_0 = -1 : index, loom.block_size_1 = -1 : index, loom.block_size_2 = -1 : index, loom.pass_name = "Materialize"} {
    func.func @matmul__d0i1_d1i1__f01__v_d__block_size_01__block_size_164__block_size_2128(%arg0: memref<4096x512xf32>, %arg1: memref<512x4096xf32>, %arg2: memref<4096x4096xf32>) {
      %cst = arith.constant 0.000000e+00 : f32
      %c64 = arith.constant 64 : index
      %c128 = arith.constant 128 : index
      affine.parallel (%arg3) = (0) to (8) {
        affine.parallel (%arg4) = (0) to (8) {
          affine.for %arg5 = 0 to 4096 {
            affine.for %arg6 = 0 to 1 {
              %12 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 8 + d1 + d2 * 64)>(%arg3, %arg4, %arg6)
              %13 = loom.alloc [128, 64] on @L1 : memref<128x64xf32>
              %14 = loom.semaphore_take %13 : memref<128x64xf32> -> memref<128x64xf32>
              %15 = loom.alloc [1, 128] on @L1 : memref<1x128xf32>
              %16 = loom.semaphore_take %15 : memref<1x128xf32> -> memref<1x128xf32>
              %17 = loom.alloc [1, 64] on @L1 : memref<1x64xf32>
              %18 = loom.semaphore_take %17 : memref<1x64xf32> -> memref<1x64xf32>
              %19 = loom.init_tensor %18[1, 64] : memref<1x64xf32> -> tensor<1x64xf32>
              %20 = linalg.fill ins(%cst : f32) outs(%19 : tensor<1x64xf32>) -> tensor<1x64xf32>
              %21 = affine.for %arg7 = 0 to 4 iter_args(%arg8 = %20) -> (tensor<1x64xf32>) {
                %24 = arith.muli %arg7, %c128 : index
                %25 = affine.apply affine_map<(d0, d1) -> (d0 * 512 + d1)>(%arg5, %24)
                %reinterpret_cast_0 = memref.reinterpret_cast %arg0 to offset: [%25], sizes: [1, 128], strides: [512, 1] : memref<4096x512xf32> to memref<1x128xf32, strided<[512, 1], offset: ?>>
                %26 = loom.copy_to_tensor %reinterpret_cast_0, %16 on @L1, interconnect : [@vertical_links], broadcast : [8, 1] : memref<1x128xf32, strided<[512, 1], offset: ?>>, memref<1x128xf32> -> tensor<1x128xf32>
                %27 = arith.muli %arg7, %c128 : index
                %28 = arith.muli %12, %c64 : index
                %29 = affine.apply affine_map<(d0, d1) -> (d0 * 4096 + d1)>(%27, %28)
                %reinterpret_cast_1 = memref.reinterpret_cast %arg1 to offset: [%29], sizes: [128, 64], strides: [4096, 1] : memref<512x4096xf32> to memref<128x64xf32, strided<[4096, 1], offset: ?>>
                %30 = loom.copy_to_tensor %reinterpret_cast_1, %14 on @L1, interconnect : [], broadcast : [1, 1] : memref<128x64xf32, strided<[4096, 1], offset: ?>>, memref<128x64xf32> -> tensor<128x64xf32>
                %31 = linalg.matmul ins(%26, %30 : tensor<1x128xf32>, tensor<128x64xf32>) outs(%arg8 : tensor<1x64xf32>) -> tensor<1x64xf32>
                loom.semaphore_give %14 : memref<128x64xf32>
                loom.semaphore_give %16 : memref<1x128xf32>
                affine.yield %31 : tensor<1x64xf32>
              }
              %22 = arith.muli %12, %c64 : index
              %23 = affine.apply affine_map<(d0, d1) -> (d0 * 4096 + d1)>(%arg5, %22)
              %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%23], sizes: [1, 64], strides: [4096, 1] : memref<4096x4096xf32> to memref<1x64xf32, strided<[4096, 1], offset: ?>>
              loom.copy_from_tensor %21, %reinterpret_cast on @DRAM : tensor<1x64xf32>, memref<1x64xf32, strided<[4096, 1], offset: ?>>
              loom.semaphore_give %18 : memref<1x64xf32>
            } {loom.iter_type = #loom.iter_type<temporal>}
          } {loom.iter_type = #loom.iter_type<temporal>}
        } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @y}
      } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @x}
      return
    }
  }
  module attributes {loom.block_size_0 = -1 : index, loom.block_size_1 = -1 : index, loom.block_size_2 = -1 : index, loom.pass_name = "Materialize"} {
    func.func @matmul__d0i1_d1i1__f10__d_d__block_size_01__block_size_164__block_size_2128(%arg0: memref<4096x512xf32>, %arg1: memref<512x4096xf32>, %arg2: memref<4096x4096xf32>) {
      %cst = arith.constant 0.000000e+00 : f32
      %c64 = arith.constant 64 : index
      %c128 = arith.constant 128 : index
      affine.parallel (%arg3) = (0) to (8) {
        affine.parallel (%arg4) = (0) to (8) {
          affine.for %arg5 = 0 to 1 {
            affine.for %arg6 = 0 to 4096 {
              %12 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 8 + d1 + d2 * 64)>(%arg3, %arg4, %arg5)
              %13 = loom.alloc [128, 64] on @L1 : memref<128x64xf32>
              %14 = loom.semaphore_take %13 : memref<128x64xf32> -> memref<128x64xf32>
              %15 = loom.alloc [1, 128] on @L1 : memref<1x128xf32>
              %16 = loom.semaphore_take %15 : memref<1x128xf32> -> memref<1x128xf32>
              %17 = loom.alloc [1, 64] on @L1 : memref<1x64xf32>
              %18 = loom.semaphore_take %17 : memref<1x64xf32> -> memref<1x64xf32>
              %19 = loom.init_tensor %18[1, 64] : memref<1x64xf32> -> tensor<1x64xf32>
              %20 = linalg.fill ins(%cst : f32) outs(%19 : tensor<1x64xf32>) -> tensor<1x64xf32>
              %21 = affine.for %arg7 = 0 to 4 iter_args(%arg8 = %20) -> (tensor<1x64xf32>) {
                %24 = arith.muli %arg7, %c128 : index
                %25 = affine.apply affine_map<(d0, d1) -> (d0 * 512 + d1)>(%arg6, %24)
                %reinterpret_cast_0 = memref.reinterpret_cast %arg0 to offset: [%25], sizes: [1, 128], strides: [512, 1] : memref<4096x512xf32> to memref<1x128xf32, strided<[512, 1], offset: ?>>
                %26 = loom.copy_to_tensor %reinterpret_cast_0, %16 on @L1, interconnect : [], broadcast : [1, 1] : memref<1x128xf32, strided<[512, 1], offset: ?>>, memref<1x128xf32> -> tensor<1x128xf32>
                %27 = arith.muli %arg7, %c128 : index
                %28 = arith.muli %12, %c64 : index
                %29 = affine.apply affine_map<(d0, d1) -> (d0 * 4096 + d1)>(%27, %28)
                %reinterpret_cast_1 = memref.reinterpret_cast %arg1 to offset: [%29], sizes: [128, 64], strides: [4096, 1] : memref<512x4096xf32> to memref<128x64xf32, strided<[4096, 1], offset: ?>>
                %30 = loom.copy_to_tensor %reinterpret_cast_1, %14 on @L1, interconnect : [], broadcast : [1, 1] : memref<128x64xf32, strided<[4096, 1], offset: ?>>, memref<128x64xf32> -> tensor<128x64xf32>
                %31 = linalg.matmul ins(%26, %30 : tensor<1x128xf32>, tensor<128x64xf32>) outs(%arg8 : tensor<1x64xf32>) -> tensor<1x64xf32>
                loom.semaphore_give %14 : memref<128x64xf32>
                loom.semaphore_give %16 : memref<1x128xf32>
                affine.yield %31 : tensor<1x64xf32>
              }
              %22 = arith.muli %12, %c64 : index
              %23 = affine.apply affine_map<(d0, d1) -> (d0 * 4096 + d1)>(%arg6, %22)
              %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%23], sizes: [1, 64], strides: [4096, 1] : memref<4096x4096xf32> to memref<1x64xf32, strided<[4096, 1], offset: ?>>
              loom.copy_from_tensor %21, %reinterpret_cast on @DRAM : tensor<1x64xf32>, memref<1x64xf32, strided<[4096, 1], offset: ?>>
              loom.semaphore_give %18 : memref<1x64xf32>
            } {loom.iter_type = #loom.iter_type<temporal>}
          } {loom.iter_type = #loom.iter_type<temporal>}
        } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @y}
      } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @x}
      return
    }
  }
  module attributes {loom.block_size_0 = -1 : index, loom.block_size_1 = -1 : index, loom.block_size_2 = -1 : index, loom.pass_name = "Materialize"} {
    func.func @matmul__d0i1_d1i1__f10__a_d__block_size_01__block_size_164__block_size_2128(%arg0: memref<4096x512xf32>, %arg1: memref<512x4096xf32>, %arg2: memref<4096x4096xf32>) {
      %cst = arith.constant 0.000000e+00 : f32
      %c64 = arith.constant 64 : index
      %c128 = arith.constant 128 : index
      affine.parallel (%arg3) = (0) to (8) {
        affine.parallel (%arg4) = (0) to (8) {
          affine.for %arg5 = 0 to 1 {
            affine.for %arg6 = 0 to 4096 {
              %12 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 8 + d1 + d2 * 64)>(%arg3, %arg4, %arg5)
              %13 = loom.alloc [128, 64] on @L1 : memref<128x64xf32>
              %14 = loom.semaphore_take %13 : memref<128x64xf32> -> memref<128x64xf32>
              %15 = loom.alloc [1, 128] on @L1 : memref<1x128xf32>
              %16 = loom.semaphore_take %15 : memref<1x128xf32> -> memref<1x128xf32>
              %17 = loom.alloc [1, 64] on @L1 : memref<1x64xf32>
              %18 = loom.semaphore_take %17 : memref<1x64xf32> -> memref<1x64xf32>
              %19 = loom.init_tensor %18[1, 64] : memref<1x64xf32> -> tensor<1x64xf32>
              %20 = linalg.fill ins(%cst : f32) outs(%19 : tensor<1x64xf32>) -> tensor<1x64xf32>
              %21 = affine.for %arg7 = 0 to 4 iter_args(%arg8 = %20) -> (tensor<1x64xf32>) {
                %24 = arith.muli %arg7, %c128 : index
                %25 = affine.apply affine_map<(d0, d1) -> (d0 * 512 + d1)>(%arg6, %24)
                %reinterpret_cast_0 = memref.reinterpret_cast %arg0 to offset: [%25], sizes: [1, 128], strides: [512, 1] : memref<4096x512xf32> to memref<1x128xf32, strided<[512, 1], offset: ?>>
                %26 = loom.copy_to_tensor %reinterpret_cast_0, %16 on @L1, interconnect : [@horizontal_links, @vertical_links], broadcast : [8, 8] : memref<1x128xf32, strided<[512, 1], offset: ?>>, memref<1x128xf32> -> tensor<1x128xf32>
                %27 = arith.muli %arg7, %c128 : index
                %28 = arith.muli %12, %c64 : index
                %29 = affine.apply affine_map<(d0, d1) -> (d0 * 4096 + d1)>(%27, %28)
                %reinterpret_cast_1 = memref.reinterpret_cast %arg1 to offset: [%29], sizes: [128, 64], strides: [4096, 1] : memref<512x4096xf32> to memref<128x64xf32, strided<[4096, 1], offset: ?>>
                %30 = loom.copy_to_tensor %reinterpret_cast_1, %14 on @L1, interconnect : [], broadcast : [1, 1] : memref<128x64xf32, strided<[4096, 1], offset: ?>>, memref<128x64xf32> -> tensor<128x64xf32>
                %31 = linalg.matmul ins(%26, %30 : tensor<1x128xf32>, tensor<128x64xf32>) outs(%arg8 : tensor<1x64xf32>) -> tensor<1x64xf32>
                loom.semaphore_give %14 : memref<128x64xf32>
                loom.semaphore_give %16 : memref<1x128xf32>
                affine.yield %31 : tensor<1x64xf32>
              }
              %22 = arith.muli %12, %c64 : index
              %23 = affine.apply affine_map<(d0, d1) -> (d0 * 4096 + d1)>(%arg6, %22)
              %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%23], sizes: [1, 64], strides: [4096, 1] : memref<4096x4096xf32> to memref<1x64xf32, strided<[4096, 1], offset: ?>>
              loom.copy_from_tensor %21, %reinterpret_cast on @DRAM : tensor<1x64xf32>, memref<1x64xf32, strided<[4096, 1], offset: ?>>
              loom.semaphore_give %18 : memref<1x64xf32>
            } {loom.iter_type = #loom.iter_type<temporal>}
          } {loom.iter_type = #loom.iter_type<temporal>}
        } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @y}
      } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @x}
      return
    }
  }
  module attributes {loom.block_size_0 = -1 : index, loom.block_size_1 = -1 : index, loom.block_size_2 = -1 : index, loom.pass_name = "Materialize"} {
    func.func @matmul__d0i1_d1i1__f10__h_d__block_size_01__block_size_164__block_size_2128(%arg0: memref<4096x512xf32>, %arg1: memref<512x4096xf32>, %arg2: memref<4096x4096xf32>) {
      %cst = arith.constant 0.000000e+00 : f32
      %c64 = arith.constant 64 : index
      %c128 = arith.constant 128 : index
      affine.parallel (%arg3) = (0) to (8) {
        affine.parallel (%arg4) = (0) to (8) {
          affine.for %arg5 = 0 to 1 {
            affine.for %arg6 = 0 to 4096 {
              %12 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 8 + d1 + d2 * 64)>(%arg3, %arg4, %arg5)
              %13 = loom.alloc [128, 64] on @L1 : memref<128x64xf32>
              %14 = loom.semaphore_take %13 : memref<128x64xf32> -> memref<128x64xf32>
              %15 = loom.alloc [1, 128] on @L1 : memref<1x128xf32>
              %16 = loom.semaphore_take %15 : memref<1x128xf32> -> memref<1x128xf32>
              %17 = loom.alloc [1, 64] on @L1 : memref<1x64xf32>
              %18 = loom.semaphore_take %17 : memref<1x64xf32> -> memref<1x64xf32>
              %19 = loom.init_tensor %18[1, 64] : memref<1x64xf32> -> tensor<1x64xf32>
              %20 = linalg.fill ins(%cst : f32) outs(%19 : tensor<1x64xf32>) -> tensor<1x64xf32>
              %21 = affine.for %arg7 = 0 to 4 iter_args(%arg8 = %20) -> (tensor<1x64xf32>) {
                %24 = arith.muli %arg7, %c128 : index
                %25 = affine.apply affine_map<(d0, d1) -> (d0 * 512 + d1)>(%arg6, %24)
                %reinterpret_cast_0 = memref.reinterpret_cast %arg0 to offset: [%25], sizes: [1, 128], strides: [512, 1] : memref<4096x512xf32> to memref<1x128xf32, strided<[512, 1], offset: ?>>
                %26 = loom.copy_to_tensor %reinterpret_cast_0, %16 on @L1, interconnect : [@horizontal_links], broadcast : [1, 8] : memref<1x128xf32, strided<[512, 1], offset: ?>>, memref<1x128xf32> -> tensor<1x128xf32>
                %27 = arith.muli %arg7, %c128 : index
                %28 = arith.muli %12, %c64 : index
                %29 = affine.apply affine_map<(d0, d1) -> (d0 * 4096 + d1)>(%27, %28)
                %reinterpret_cast_1 = memref.reinterpret_cast %arg1 to offset: [%29], sizes: [128, 64], strides: [4096, 1] : memref<512x4096xf32> to memref<128x64xf32, strided<[4096, 1], offset: ?>>
                %30 = loom.copy_to_tensor %reinterpret_cast_1, %14 on @L1, interconnect : [], broadcast : [1, 1] : memref<128x64xf32, strided<[4096, 1], offset: ?>>, memref<128x64xf32> -> tensor<128x64xf32>
                %31 = linalg.matmul ins(%26, %30 : tensor<1x128xf32>, tensor<128x64xf32>) outs(%arg8 : tensor<1x64xf32>) -> tensor<1x64xf32>
                loom.semaphore_give %14 : memref<128x64xf32>
                loom.semaphore_give %16 : memref<1x128xf32>
                affine.yield %31 : tensor<1x64xf32>
              }
              %22 = arith.muli %12, %c64 : index
              %23 = affine.apply affine_map<(d0, d1) -> (d0 * 4096 + d1)>(%arg6, %22)
              %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%23], sizes: [1, 64], strides: [4096, 1] : memref<4096x4096xf32> to memref<1x64xf32, strided<[4096, 1], offset: ?>>
              loom.copy_from_tensor %21, %reinterpret_cast on @DRAM : tensor<1x64xf32>, memref<1x64xf32, strided<[4096, 1], offset: ?>>
              loom.semaphore_give %18 : memref<1x64xf32>
            } {loom.iter_type = #loom.iter_type<temporal>}
          } {loom.iter_type = #loom.iter_type<temporal>}
        } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @y}
      } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @x}
      return
    }
  }
  module attributes {loom.block_size_0 = -1 : index, loom.block_size_1 = -1 : index, loom.block_size_2 = -1 : index, loom.pass_name = "Materialize"} {
    func.func @matmul__d0i1_d1i1__f10__v_d__block_size_01__block_size_164__block_size_2128(%arg0: memref<4096x512xf32>, %arg1: memref<512x4096xf32>, %arg2: memref<4096x4096xf32>) {
      %cst = arith.constant 0.000000e+00 : f32
      %c64 = arith.constant 64 : index
      %c128 = arith.constant 128 : index
      affine.parallel (%arg3) = (0) to (8) {
        affine.parallel (%arg4) = (0) to (8) {
          affine.for %arg5 = 0 to 1 {
            affine.for %arg6 = 0 to 4096 {
              %12 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 8 + d1 + d2 * 64)>(%arg3, %arg4, %arg5)
              %13 = loom.alloc [128, 64] on @L1 : memref<128x64xf32>
              %14 = loom.semaphore_take %13 : memref<128x64xf32> -> memref<128x64xf32>
              %15 = loom.alloc [1, 128] on @L1 : memref<1x128xf32>
              %16 = loom.semaphore_take %15 : memref<1x128xf32> -> memref<1x128xf32>
              %17 = loom.alloc [1, 64] on @L1 : memref<1x64xf32>
              %18 = loom.semaphore_take %17 : memref<1x64xf32> -> memref<1x64xf32>
              %19 = loom.init_tensor %18[1, 64] : memref<1x64xf32> -> tensor<1x64xf32>
              %20 = linalg.fill ins(%cst : f32) outs(%19 : tensor<1x64xf32>) -> tensor<1x64xf32>
              %21 = affine.for %arg7 = 0 to 4 iter_args(%arg8 = %20) -> (tensor<1x64xf32>) {
                %24 = arith.muli %arg7, %c128 : index
                %25 = affine.apply affine_map<(d0, d1) -> (d0 * 512 + d1)>(%arg6, %24)
                %reinterpret_cast_0 = memref.reinterpret_cast %arg0 to offset: [%25], sizes: [1, 128], strides: [512, 1] : memref<4096x512xf32> to memref<1x128xf32, strided<[512, 1], offset: ?>>
                %26 = loom.copy_to_tensor %reinterpret_cast_0, %16 on @L1, interconnect : [@vertical_links], broadcast : [8, 1] : memref<1x128xf32, strided<[512, 1], offset: ?>>, memref<1x128xf32> -> tensor<1x128xf32>
                %27 = arith.muli %arg7, %c128 : index
                %28 = arith.muli %12, %c64 : index
                %29 = affine.apply affine_map<(d0, d1) -> (d0 * 4096 + d1)>(%27, %28)
                %reinterpret_cast_1 = memref.reinterpret_cast %arg1 to offset: [%29], sizes: [128, 64], strides: [4096, 1] : memref<512x4096xf32> to memref<128x64xf32, strided<[4096, 1], offset: ?>>
                %30 = loom.copy_to_tensor %reinterpret_cast_1, %14 on @L1, interconnect : [], broadcast : [1, 1] : memref<128x64xf32, strided<[4096, 1], offset: ?>>, memref<128x64xf32> -> tensor<128x64xf32>
                %31 = linalg.matmul ins(%26, %30 : tensor<1x128xf32>, tensor<128x64xf32>) outs(%arg8 : tensor<1x64xf32>) -> tensor<1x64xf32>
                loom.semaphore_give %14 : memref<128x64xf32>
                loom.semaphore_give %16 : memref<1x128xf32>
                affine.yield %31 : tensor<1x64xf32>
              }
              %22 = arith.muli %12, %c64 : index
              %23 = affine.apply affine_map<(d0, d1) -> (d0 * 4096 + d1)>(%arg6, %22)
              %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%23], sizes: [1, 64], strides: [4096, 1] : memref<4096x4096xf32> to memref<1x64xf32, strided<[4096, 1], offset: ?>>
              loom.copy_from_tensor %21, %reinterpret_cast on @DRAM : tensor<1x64xf32>, memref<1x64xf32, strided<[4096, 1], offset: ?>>
              loom.semaphore_give %18 : memref<1x64xf32>
            } {loom.iter_type = #loom.iter_type<temporal>}
          } {loom.iter_type = #loom.iter_type<temporal>}
        } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @y}
      } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @x}
      return
    }
  }
  module attributes {loom.block_size_0 = -1 : index, loom.block_size_1 = -1 : index, loom.block_size_2 = -1 : index, loom.pass_name = "Materialize"} {
    func.func @matmul__d1i1_d0i1__f01__d_d__block_size_01__block_size_164__block_size_2128(%arg0: memref<4096x512xf32>, %arg1: memref<512x4096xf32>, %arg2: memref<4096x4096xf32>) {
      %cst = arith.constant 0.000000e+00 : f32
      %c64 = arith.constant 64 : index
      %c128 = arith.constant 128 : index
      affine.parallel (%arg3) = (0) to (8) {
        affine.parallel (%arg4) = (0) to (8) {
          affine.for %arg5 = 0 to 4096 {
            affine.for %arg6 = 0 to 1 {
              %12 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 8 + d1 + d2 * 64)>(%arg3, %arg4, %arg6)
              %13 = loom.alloc [128, 64] on @L1 : memref<128x64xf32>
              %14 = loom.semaphore_take %13 : memref<128x64xf32> -> memref<128x64xf32>
              %15 = loom.alloc [1, 128] on @L1 : memref<1x128xf32>
              %16 = loom.semaphore_take %15 : memref<1x128xf32> -> memref<1x128xf32>
              %17 = loom.alloc [1, 64] on @L1 : memref<1x64xf32>
              %18 = loom.semaphore_take %17 : memref<1x64xf32> -> memref<1x64xf32>
              %19 = loom.init_tensor %18[1, 64] : memref<1x64xf32> -> tensor<1x64xf32>
              %20 = linalg.fill ins(%cst : f32) outs(%19 : tensor<1x64xf32>) -> tensor<1x64xf32>
              %21 = affine.for %arg7 = 0 to 4 iter_args(%arg8 = %20) -> (tensor<1x64xf32>) {
                %24 = arith.muli %arg7, %c128 : index
                %25 = affine.apply affine_map<(d0, d1) -> (d0 * 512 + d1)>(%arg5, %24)
                %reinterpret_cast_0 = memref.reinterpret_cast %arg0 to offset: [%25], sizes: [1, 128], strides: [512, 1] : memref<4096x512xf32> to memref<1x128xf32, strided<[512, 1], offset: ?>>
                %26 = loom.copy_to_tensor %reinterpret_cast_0, %16 on @L1, interconnect : [], broadcast : [1, 1] : memref<1x128xf32, strided<[512, 1], offset: ?>>, memref<1x128xf32> -> tensor<1x128xf32>
                %27 = arith.muli %arg7, %c128 : index
                %28 = arith.muli %12, %c64 : index
                %29 = affine.apply affine_map<(d0, d1) -> (d0 * 4096 + d1)>(%27, %28)
                %reinterpret_cast_1 = memref.reinterpret_cast %arg1 to offset: [%29], sizes: [128, 64], strides: [4096, 1] : memref<512x4096xf32> to memref<128x64xf32, strided<[4096, 1], offset: ?>>
                %30 = loom.copy_to_tensor %reinterpret_cast_1, %14 on @L1, interconnect : [], broadcast : [1, 1] : memref<128x64xf32, strided<[4096, 1], offset: ?>>, memref<128x64xf32> -> tensor<128x64xf32>
                %31 = linalg.matmul ins(%26, %30 : tensor<1x128xf32>, tensor<128x64xf32>) outs(%arg8 : tensor<1x64xf32>) -> tensor<1x64xf32>
                loom.semaphore_give %14 : memref<128x64xf32>
                loom.semaphore_give %16 : memref<1x128xf32>
                affine.yield %31 : tensor<1x64xf32>
              }
              %22 = arith.muli %12, %c64 : index
              %23 = affine.apply affine_map<(d0, d1) -> (d0 * 4096 + d1)>(%arg5, %22)
              %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%23], sizes: [1, 64], strides: [4096, 1] : memref<4096x4096xf32> to memref<1x64xf32, strided<[4096, 1], offset: ?>>
              loom.copy_from_tensor %21, %reinterpret_cast on @DRAM : tensor<1x64xf32>, memref<1x64xf32, strided<[4096, 1], offset: ?>>
              loom.semaphore_give %18 : memref<1x64xf32>
            } {loom.iter_type = #loom.iter_type<temporal>}
          } {loom.iter_type = #loom.iter_type<temporal>}
        } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @x}
      } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @y}
      return
    }
  }
  module attributes {loom.block_size_0 = -1 : index, loom.block_size_1 = -1 : index, loom.block_size_2 = -1 : index, loom.pass_name = "Materialize"} {
    func.func @matmul__d1i1_d0i1__f01__a_d__block_size_01__block_size_164__block_size_2128(%arg0: memref<4096x512xf32>, %arg1: memref<512x4096xf32>, %arg2: memref<4096x4096xf32>) {
      %cst = arith.constant 0.000000e+00 : f32
      %c64 = arith.constant 64 : index
      %c128 = arith.constant 128 : index
      affine.parallel (%arg3) = (0) to (8) {
        affine.parallel (%arg4) = (0) to (8) {
          affine.for %arg5 = 0 to 4096 {
            affine.for %arg6 = 0 to 1 {
              %12 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 8 + d1 + d2 * 64)>(%arg3, %arg4, %arg6)
              %13 = loom.alloc [128, 64] on @L1 : memref<128x64xf32>
              %14 = loom.semaphore_take %13 : memref<128x64xf32> -> memref<128x64xf32>
              %15 = loom.alloc [1, 128] on @L1 : memref<1x128xf32>
              %16 = loom.semaphore_take %15 : memref<1x128xf32> -> memref<1x128xf32>
              %17 = loom.alloc [1, 64] on @L1 : memref<1x64xf32>
              %18 = loom.semaphore_take %17 : memref<1x64xf32> -> memref<1x64xf32>
              %19 = loom.init_tensor %18[1, 64] : memref<1x64xf32> -> tensor<1x64xf32>
              %20 = linalg.fill ins(%cst : f32) outs(%19 : tensor<1x64xf32>) -> tensor<1x64xf32>
              %21 = affine.for %arg7 = 0 to 4 iter_args(%arg8 = %20) -> (tensor<1x64xf32>) {
                %24 = arith.muli %arg7, %c128 : index
                %25 = affine.apply affine_map<(d0, d1) -> (d0 * 512 + d1)>(%arg5, %24)
                %reinterpret_cast_0 = memref.reinterpret_cast %arg0 to offset: [%25], sizes: [1, 128], strides: [512, 1] : memref<4096x512xf32> to memref<1x128xf32, strided<[512, 1], offset: ?>>
                %26 = loom.copy_to_tensor %reinterpret_cast_0, %16 on @L1, interconnect : [@horizontal_links, @vertical_links], broadcast : [8, 8] : memref<1x128xf32, strided<[512, 1], offset: ?>>, memref<1x128xf32> -> tensor<1x128xf32>
                %27 = arith.muli %arg7, %c128 : index
                %28 = arith.muli %12, %c64 : index
                %29 = affine.apply affine_map<(d0, d1) -> (d0 * 4096 + d1)>(%27, %28)
                %reinterpret_cast_1 = memref.reinterpret_cast %arg1 to offset: [%29], sizes: [128, 64], strides: [4096, 1] : memref<512x4096xf32> to memref<128x64xf32, strided<[4096, 1], offset: ?>>
                %30 = loom.copy_to_tensor %reinterpret_cast_1, %14 on @L1, interconnect : [], broadcast : [1, 1] : memref<128x64xf32, strided<[4096, 1], offset: ?>>, memref<128x64xf32> -> tensor<128x64xf32>
                %31 = linalg.matmul ins(%26, %30 : tensor<1x128xf32>, tensor<128x64xf32>) outs(%arg8 : tensor<1x64xf32>) -> tensor<1x64xf32>
                loom.semaphore_give %14 : memref<128x64xf32>
                loom.semaphore_give %16 : memref<1x128xf32>
                affine.yield %31 : tensor<1x64xf32>
              }
              %22 = arith.muli %12, %c64 : index
              %23 = affine.apply affine_map<(d0, d1) -> (d0 * 4096 + d1)>(%arg5, %22)
              %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%23], sizes: [1, 64], strides: [4096, 1] : memref<4096x4096xf32> to memref<1x64xf32, strided<[4096, 1], offset: ?>>
              loom.copy_from_tensor %21, %reinterpret_cast on @DRAM : tensor<1x64xf32>, memref<1x64xf32, strided<[4096, 1], offset: ?>>
              loom.semaphore_give %18 : memref<1x64xf32>
            } {loom.iter_type = #loom.iter_type<temporal>}
          } {loom.iter_type = #loom.iter_type<temporal>}
        } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @x}
      } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @y}
      return
    }
  }
  module attributes {loom.block_size_0 = -1 : index, loom.block_size_1 = -1 : index, loom.block_size_2 = -1 : index, loom.pass_name = "Materialize"} {
    func.func @matmul__d1i1_d0i1__f01__h_d__block_size_01__block_size_164__block_size_2128(%arg0: memref<4096x512xf32>, %arg1: memref<512x4096xf32>, %arg2: memref<4096x4096xf32>) {
      %cst = arith.constant 0.000000e+00 : f32
      %c64 = arith.constant 64 : index
      %c128 = arith.constant 128 : index
      affine.parallel (%arg3) = (0) to (8) {
        affine.parallel (%arg4) = (0) to (8) {
          affine.for %arg5 = 0 to 4096 {
            affine.for %arg6 = 0 to 1 {
              %12 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 8 + d1 + d2 * 64)>(%arg3, %arg4, %arg6)
              %13 = loom.alloc [128, 64] on @L1 : memref<128x64xf32>
              %14 = loom.semaphore_take %13 : memref<128x64xf32> -> memref<128x64xf32>
              %15 = loom.alloc [1, 128] on @L1 : memref<1x128xf32>
              %16 = loom.semaphore_take %15 : memref<1x128xf32> -> memref<1x128xf32>
              %17 = loom.alloc [1, 64] on @L1 : memref<1x64xf32>
              %18 = loom.semaphore_take %17 : memref<1x64xf32> -> memref<1x64xf32>
              %19 = loom.init_tensor %18[1, 64] : memref<1x64xf32> -> tensor<1x64xf32>
              %20 = linalg.fill ins(%cst : f32) outs(%19 : tensor<1x64xf32>) -> tensor<1x64xf32>
              %21 = affine.for %arg7 = 0 to 4 iter_args(%arg8 = %20) -> (tensor<1x64xf32>) {
                %24 = arith.muli %arg7, %c128 : index
                %25 = affine.apply affine_map<(d0, d1) -> (d0 * 512 + d1)>(%arg5, %24)
                %reinterpret_cast_0 = memref.reinterpret_cast %arg0 to offset: [%25], sizes: [1, 128], strides: [512, 1] : memref<4096x512xf32> to memref<1x128xf32, strided<[512, 1], offset: ?>>
                %26 = loom.copy_to_tensor %reinterpret_cast_0, %16 on @L1, interconnect : [@horizontal_links], broadcast : [1, 8] : memref<1x128xf32, strided<[512, 1], offset: ?>>, memref<1x128xf32> -> tensor<1x128xf32>
                %27 = arith.muli %arg7, %c128 : index
                %28 = arith.muli %12, %c64 : index
                %29 = affine.apply affine_map<(d0, d1) -> (d0 * 4096 + d1)>(%27, %28)
                %reinterpret_cast_1 = memref.reinterpret_cast %arg1 to offset: [%29], sizes: [128, 64], strides: [4096, 1] : memref<512x4096xf32> to memref<128x64xf32, strided<[4096, 1], offset: ?>>
                %30 = loom.copy_to_tensor %reinterpret_cast_1, %14 on @L1, interconnect : [], broadcast : [1, 1] : memref<128x64xf32, strided<[4096, 1], offset: ?>>, memref<128x64xf32> -> tensor<128x64xf32>
                %31 = linalg.matmul ins(%26, %30 : tensor<1x128xf32>, tensor<128x64xf32>) outs(%arg8 : tensor<1x64xf32>) -> tensor<1x64xf32>
                loom.semaphore_give %14 : memref<128x64xf32>
                loom.semaphore_give %16 : memref<1x128xf32>
                affine.yield %31 : tensor<1x64xf32>
              }
              %22 = arith.muli %12, %c64 : index
              %23 = affine.apply affine_map<(d0, d1) -> (d0 * 4096 + d1)>(%arg5, %22)
              %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%23], sizes: [1, 64], strides: [4096, 1] : memref<4096x4096xf32> to memref<1x64xf32, strided<[4096, 1], offset: ?>>
              loom.copy_from_tensor %21, %reinterpret_cast on @DRAM : tensor<1x64xf32>, memref<1x64xf32, strided<[4096, 1], offset: ?>>
              loom.semaphore_give %18 : memref<1x64xf32>
            } {loom.iter_type = #loom.iter_type<temporal>}
          } {loom.iter_type = #loom.iter_type<temporal>}
        } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @x}
      } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @y}
      return
    }
  }
  module attributes {loom.block_size_0 = -1 : index, loom.block_size_1 = -1 : index, loom.block_size_2 = -1 : index, loom.pass_name = "Materialize"} {
    func.func @matmul__d1i1_d0i1__f01__v_d__block_size_01__block_size_164__block_size_2128(%arg0: memref<4096x512xf32>, %arg1: memref<512x4096xf32>, %arg2: memref<4096x4096xf32>) {
      %cst = arith.constant 0.000000e+00 : f32
      %c64 = arith.constant 64 : index
      %c128 = arith.constant 128 : index
      affine.parallel (%arg3) = (0) to (8) {
        affine.parallel (%arg4) = (0) to (8) {
          affine.for %arg5 = 0 to 4096 {
            affine.for %arg6 = 0 to 1 {
              %12 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 8 + d1 + d2 * 64)>(%arg3, %arg4, %arg6)
              %13 = loom.alloc [128, 64] on @L1 : memref<128x64xf32>
              %14 = loom.semaphore_take %13 : memref<128x64xf32> -> memref<128x64xf32>
              %15 = loom.alloc [1, 128] on @L1 : memref<1x128xf32>
              %16 = loom.semaphore_take %15 : memref<1x128xf32> -> memref<1x128xf32>
              %17 = loom.alloc [1, 64] on @L1 : memref<1x64xf32>
              %18 = loom.semaphore_take %17 : memref<1x64xf32> -> memref<1x64xf32>
              %19 = loom.init_tensor %18[1, 64] : memref<1x64xf32> -> tensor<1x64xf32>
              %20 = linalg.fill ins(%cst : f32) outs(%19 : tensor<1x64xf32>) -> tensor<1x64xf32>
              %21 = affine.for %arg7 = 0 to 4 iter_args(%arg8 = %20) -> (tensor<1x64xf32>) {
                %24 = arith.muli %arg7, %c128 : index
                %25 = affine.apply affine_map<(d0, d1) -> (d0 * 512 + d1)>(%arg5, %24)
                %reinterpret_cast_0 = memref.reinterpret_cast %arg0 to offset: [%25], sizes: [1, 128], strides: [512, 1] : memref<4096x512xf32> to memref<1x128xf32, strided<[512, 1], offset: ?>>
                %26 = loom.copy_to_tensor %reinterpret_cast_0, %16 on @L1, interconnect : [@vertical_links], broadcast : [8, 1] : memref<1x128xf32, strided<[512, 1], offset: ?>>, memref<1x128xf32> -> tensor<1x128xf32>
                %27 = arith.muli %arg7, %c128 : index
                %28 = arith.muli %12, %c64 : index
                %29 = affine.apply affine_map<(d0, d1) -> (d0 * 4096 + d1)>(%27, %28)
                %reinterpret_cast_1 = memref.reinterpret_cast %arg1 to offset: [%29], sizes: [128, 64], strides: [4096, 1] : memref<512x4096xf32> to memref<128x64xf32, strided<[4096, 1], offset: ?>>
                %30 = loom.copy_to_tensor %reinterpret_cast_1, %14 on @L1, interconnect : [], broadcast : [1, 1] : memref<128x64xf32, strided<[4096, 1], offset: ?>>, memref<128x64xf32> -> tensor<128x64xf32>
                %31 = linalg.matmul ins(%26, %30 : tensor<1x128xf32>, tensor<128x64xf32>) outs(%arg8 : tensor<1x64xf32>) -> tensor<1x64xf32>
                loom.semaphore_give %14 : memref<128x64xf32>
                loom.semaphore_give %16 : memref<1x128xf32>
                affine.yield %31 : tensor<1x64xf32>
              }
              %22 = arith.muli %12, %c64 : index
              %23 = affine.apply affine_map<(d0, d1) -> (d0 * 4096 + d1)>(%arg5, %22)
              %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%23], sizes: [1, 64], strides: [4096, 1] : memref<4096x4096xf32> to memref<1x64xf32, strided<[4096, 1], offset: ?>>
              loom.copy_from_tensor %21, %reinterpret_cast on @DRAM : tensor<1x64xf32>, memref<1x64xf32, strided<[4096, 1], offset: ?>>
              loom.semaphore_give %18 : memref<1x64xf32>
            } {loom.iter_type = #loom.iter_type<temporal>}
          } {loom.iter_type = #loom.iter_type<temporal>}
        } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @x}
      } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @y}
      return
    }
  }
  module attributes {loom.block_size_0 = -1 : index, loom.block_size_1 = -1 : index, loom.block_size_2 = -1 : index, loom.pass_name = "Materialize"} {
    func.func @matmul__d1i1_d0i1__f10__d_d__block_size_01__block_size_164__block_size_2128(%arg0: memref<4096x512xf32>, %arg1: memref<512x4096xf32>, %arg2: memref<4096x4096xf32>) {
      %cst = arith.constant 0.000000e+00 : f32
      %c64 = arith.constant 64 : index
      %c128 = arith.constant 128 : index
      affine.parallel (%arg3) = (0) to (8) {
        affine.parallel (%arg4) = (0) to (8) {
          affine.for %arg5 = 0 to 1 {
            affine.for %arg6 = 0 to 4096 {
              %12 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 8 + d1 + d2 * 64)>(%arg3, %arg4, %arg5)
              %13 = loom.alloc [128, 64] on @L1 : memref<128x64xf32>
              %14 = loom.semaphore_take %13 : memref<128x64xf32> -> memref<128x64xf32>
              %15 = loom.alloc [1, 128] on @L1 : memref<1x128xf32>
              %16 = loom.semaphore_take %15 : memref<1x128xf32> -> memref<1x128xf32>
              %17 = loom.alloc [1, 64] on @L1 : memref<1x64xf32>
              %18 = loom.semaphore_take %17 : memref<1x64xf32> -> memref<1x64xf32>
              %19 = loom.init_tensor %18[1, 64] : memref<1x64xf32> -> tensor<1x64xf32>
              %20 = linalg.fill ins(%cst : f32) outs(%19 : tensor<1x64xf32>) -> tensor<1x64xf32>
              %21 = affine.for %arg7 = 0 to 4 iter_args(%arg8 = %20) -> (tensor<1x64xf32>) {
                %24 = arith.muli %arg7, %c128 : index
                %25 = affine.apply affine_map<(d0, d1) -> (d0 * 512 + d1)>(%arg6, %24)
                %reinterpret_cast_0 = memref.reinterpret_cast %arg0 to offset: [%25], sizes: [1, 128], strides: [512, 1] : memref<4096x512xf32> to memref<1x128xf32, strided<[512, 1], offset: ?>>
                %26 = loom.copy_to_tensor %reinterpret_cast_0, %16 on @L1, interconnect : [], broadcast : [1, 1] : memref<1x128xf32, strided<[512, 1], offset: ?>>, memref<1x128xf32> -> tensor<1x128xf32>
                %27 = arith.muli %arg7, %c128 : index
                %28 = arith.muli %12, %c64 : index
                %29 = affine.apply affine_map<(d0, d1) -> (d0 * 4096 + d1)>(%27, %28)
                %reinterpret_cast_1 = memref.reinterpret_cast %arg1 to offset: [%29], sizes: [128, 64], strides: [4096, 1] : memref<512x4096xf32> to memref<128x64xf32, strided<[4096, 1], offset: ?>>
                %30 = loom.copy_to_tensor %reinterpret_cast_1, %14 on @L1, interconnect : [], broadcast : [1, 1] : memref<128x64xf32, strided<[4096, 1], offset: ?>>, memref<128x64xf32> -> tensor<128x64xf32>
                %31 = linalg.matmul ins(%26, %30 : tensor<1x128xf32>, tensor<128x64xf32>) outs(%arg8 : tensor<1x64xf32>) -> tensor<1x64xf32>
                loom.semaphore_give %14 : memref<128x64xf32>
                loom.semaphore_give %16 : memref<1x128xf32>
                affine.yield %31 : tensor<1x64xf32>
              }
              %22 = arith.muli %12, %c64 : index
              %23 = affine.apply affine_map<(d0, d1) -> (d0 * 4096 + d1)>(%arg6, %22)
              %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%23], sizes: [1, 64], strides: [4096, 1] : memref<4096x4096xf32> to memref<1x64xf32, strided<[4096, 1], offset: ?>>
              loom.copy_from_tensor %21, %reinterpret_cast on @DRAM : tensor<1x64xf32>, memref<1x64xf32, strided<[4096, 1], offset: ?>>
              loom.semaphore_give %18 : memref<1x64xf32>
            } {loom.iter_type = #loom.iter_type<temporal>}
          } {loom.iter_type = #loom.iter_type<temporal>}
        } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @x}
      } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @y}
      return
    }
  }
  module attributes {loom.block_size_0 = -1 : index, loom.block_size_1 = -1 : index, loom.block_size_2 = -1 : index, loom.pass_name = "Materialize"} {
    func.func @matmul__d1i1_d0i1__f10__a_d__block_size_01__block_size_164__block_size_2128(%arg0: memref<4096x512xf32>, %arg1: memref<512x4096xf32>, %arg2: memref<4096x4096xf32>) {
      %cst = arith.constant 0.000000e+00 : f32
      %c64 = arith.constant 64 : index
      %c128 = arith.constant 128 : index
      affine.parallel (%arg3) = (0) to (8) {
        affine.parallel (%arg4) = (0) to (8) {
          affine.for %arg5 = 0 to 1 {
            affine.for %arg6 = 0 to 4096 {
              %12 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 8 + d1 + d2 * 64)>(%arg3, %arg4, %arg5)
              %13 = loom.alloc [128, 64] on @L1 : memref<128x64xf32>
              %14 = loom.semaphore_take %13 : memref<128x64xf32> -> memref<128x64xf32>
              %15 = loom.alloc [1, 128] on @L1 : memref<1x128xf32>
              %16 = loom.semaphore_take %15 : memref<1x128xf32> -> memref<1x128xf32>
              %17 = loom.alloc [1, 64] on @L1 : memref<1x64xf32>
              %18 = loom.semaphore_take %17 : memref<1x64xf32> -> memref<1x64xf32>
              %19 = loom.init_tensor %18[1, 64] : memref<1x64xf32> -> tensor<1x64xf32>
              %20 = linalg.fill ins(%cst : f32) outs(%19 : tensor<1x64xf32>) -> tensor<1x64xf32>
              %21 = affine.for %arg7 = 0 to 4 iter_args(%arg8 = %20) -> (tensor<1x64xf32>) {
                %24 = arith.muli %arg7, %c128 : index
                %25 = affine.apply affine_map<(d0, d1) -> (d0 * 512 + d1)>(%arg6, %24)
                %reinterpret_cast_0 = memref.reinterpret_cast %arg0 to offset: [%25], sizes: [1, 128], strides: [512, 1] : memref<4096x512xf32> to memref<1x128xf32, strided<[512, 1], offset: ?>>
                %26 = loom.copy_to_tensor %reinterpret_cast_0, %16 on @L1, interconnect : [@horizontal_links, @vertical_links], broadcast : [8, 8] : memref<1x128xf32, strided<[512, 1], offset: ?>>, memref<1x128xf32> -> tensor<1x128xf32>
                %27 = arith.muli %arg7, %c128 : index
                %28 = arith.muli %12, %c64 : index
                %29 = affine.apply affine_map<(d0, d1) -> (d0 * 4096 + d1)>(%27, %28)
                %reinterpret_cast_1 = memref.reinterpret_cast %arg1 to offset: [%29], sizes: [128, 64], strides: [4096, 1] : memref<512x4096xf32> to memref<128x64xf32, strided<[4096, 1], offset: ?>>
                %30 = loom.copy_to_tensor %reinterpret_cast_1, %14 on @L1, interconnect : [], broadcast : [1, 1] : memref<128x64xf32, strided<[4096, 1], offset: ?>>, memref<128x64xf32> -> tensor<128x64xf32>
                %31 = linalg.matmul ins(%26, %30 : tensor<1x128xf32>, tensor<128x64xf32>) outs(%arg8 : tensor<1x64xf32>) -> tensor<1x64xf32>
                loom.semaphore_give %14 : memref<128x64xf32>
                loom.semaphore_give %16 : memref<1x128xf32>
                affine.yield %31 : tensor<1x64xf32>
              }
              %22 = arith.muli %12, %c64 : index
              %23 = affine.apply affine_map<(d0, d1) -> (d0 * 4096 + d1)>(%arg6, %22)
              %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%23], sizes: [1, 64], strides: [4096, 1] : memref<4096x4096xf32> to memref<1x64xf32, strided<[4096, 1], offset: ?>>
              loom.copy_from_tensor %21, %reinterpret_cast on @DRAM : tensor<1x64xf32>, memref<1x64xf32, strided<[4096, 1], offset: ?>>
              loom.semaphore_give %18 : memref<1x64xf32>
            } {loom.iter_type = #loom.iter_type<temporal>}
          } {loom.iter_type = #loom.iter_type<temporal>}
        } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @x}
      } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @y}
      return
    }
  }
  module attributes {loom.block_size_0 = -1 : index, loom.block_size_1 = -1 : index, loom.block_size_2 = -1 : index, loom.pass_name = "Materialize"} {
    func.func @matmul__d1i1_d0i1__f10__h_d__block_size_01__block_size_164__block_size_2128(%arg0: memref<4096x512xf32>, %arg1: memref<512x4096xf32>, %arg2: memref<4096x4096xf32>) {
      %cst = arith.constant 0.000000e+00 : f32
      %c64 = arith.constant 64 : index
      %c128 = arith.constant 128 : index
      affine.parallel (%arg3) = (0) to (8) {
        affine.parallel (%arg4) = (0) to (8) {
          affine.for %arg5 = 0 to 1 {
            affine.for %arg6 = 0 to 4096 {
              %12 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 8 + d1 + d2 * 64)>(%arg3, %arg4, %arg5)
              %13 = loom.alloc [128, 64] on @L1 : memref<128x64xf32>
              %14 = loom.semaphore_take %13 : memref<128x64xf32> -> memref<128x64xf32>
              %15 = loom.alloc [1, 128] on @L1 : memref<1x128xf32>
              %16 = loom.semaphore_take %15 : memref<1x128xf32> -> memref<1x128xf32>
              %17 = loom.alloc [1, 64] on @L1 : memref<1x64xf32>
              %18 = loom.semaphore_take %17 : memref<1x64xf32> -> memref<1x64xf32>
              %19 = loom.init_tensor %18[1, 64] : memref<1x64xf32> -> tensor<1x64xf32>
              %20 = linalg.fill ins(%cst : f32) outs(%19 : tensor<1x64xf32>) -> tensor<1x64xf32>
              %21 = affine.for %arg7 = 0 to 4 iter_args(%arg8 = %20) -> (tensor<1x64xf32>) {
                %24 = arith.muli %arg7, %c128 : index
                %25 = affine.apply affine_map<(d0, d1) -> (d0 * 512 + d1)>(%arg6, %24)
                %reinterpret_cast_0 = memref.reinterpret_cast %arg0 to offset: [%25], sizes: [1, 128], strides: [512, 1] : memref<4096x512xf32> to memref<1x128xf32, strided<[512, 1], offset: ?>>
                %26 = loom.copy_to_tensor %reinterpret_cast_0, %16 on @L1, interconnect : [@horizontal_links], broadcast : [1, 8] : memref<1x128xf32, strided<[512, 1], offset: ?>>, memref<1x128xf32> -> tensor<1x128xf32>
                %27 = arith.muli %arg7, %c128 : index
                %28 = arith.muli %12, %c64 : index
                %29 = affine.apply affine_map<(d0, d1) -> (d0 * 4096 + d1)>(%27, %28)
                %reinterpret_cast_1 = memref.reinterpret_cast %arg1 to offset: [%29], sizes: [128, 64], strides: [4096, 1] : memref<512x4096xf32> to memref<128x64xf32, strided<[4096, 1], offset: ?>>
                %30 = loom.copy_to_tensor %reinterpret_cast_1, %14 on @L1, interconnect : [], broadcast : [1, 1] : memref<128x64xf32, strided<[4096, 1], offset: ?>>, memref<128x64xf32> -> tensor<128x64xf32>
                %31 = linalg.matmul ins(%26, %30 : tensor<1x128xf32>, tensor<128x64xf32>) outs(%arg8 : tensor<1x64xf32>) -> tensor<1x64xf32>
                loom.semaphore_give %14 : memref<128x64xf32>
                loom.semaphore_give %16 : memref<1x128xf32>
                affine.yield %31 : tensor<1x64xf32>
              }
              %22 = arith.muli %12, %c64 : index
              %23 = affine.apply affine_map<(d0, d1) -> (d0 * 4096 + d1)>(%arg6, %22)
              %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%23], sizes: [1, 64], strides: [4096, 1] : memref<4096x4096xf32> to memref<1x64xf32, strided<[4096, 1], offset: ?>>
              loom.copy_from_tensor %21, %reinterpret_cast on @DRAM : tensor<1x64xf32>, memref<1x64xf32, strided<[4096, 1], offset: ?>>
              loom.semaphore_give %18 : memref<1x64xf32>
            } {loom.iter_type = #loom.iter_type<temporal>}
          } {loom.iter_type = #loom.iter_type<temporal>}
        } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @x}
      } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @y}
      return
    }
  }
  module attributes {loom.block_size_0 = -1 : index, loom.block_size_1 = -1 : index, loom.block_size_2 = -1 : index, loom.pass_name = "Materialize"} {
    func.func @matmul__d1i1_d0i1__f10__v_d__block_size_01__block_size_164__block_size_2128(%arg0: memref<4096x512xf32>, %arg1: memref<512x4096xf32>, %arg2: memref<4096x4096xf32>) {
      %cst = arith.constant 0.000000e+00 : f32
      %c64 = arith.constant 64 : index
      %c128 = arith.constant 128 : index
      affine.parallel (%arg3) = (0) to (8) {
        affine.parallel (%arg4) = (0) to (8) {
          affine.for %arg5 = 0 to 1 {
            affine.for %arg6 = 0 to 4096 {
              %12 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 8 + d1 + d2 * 64)>(%arg3, %arg4, %arg5)
              %13 = loom.alloc [128, 64] on @L1 : memref<128x64xf32>
              %14 = loom.semaphore_take %13 : memref<128x64xf32> -> memref<128x64xf32>
              %15 = loom.alloc [1, 128] on @L1 : memref<1x128xf32>
              %16 = loom.semaphore_take %15 : memref<1x128xf32> -> memref<1x128xf32>
              %17 = loom.alloc [1, 64] on @L1 : memref<1x64xf32>
              %18 = loom.semaphore_take %17 : memref<1x64xf32> -> memref<1x64xf32>
              %19 = loom.init_tensor %18[1, 64] : memref<1x64xf32> -> tensor<1x64xf32>
              %20 = linalg.fill ins(%cst : f32) outs(%19 : tensor<1x64xf32>) -> tensor<1x64xf32>
              %21 = affine.for %arg7 = 0 to 4 iter_args(%arg8 = %20) -> (tensor<1x64xf32>) {
                %24 = arith.muli %arg7, %c128 : index
                %25 = affine.apply affine_map<(d0, d1) -> (d0 * 512 + d1)>(%arg6, %24)
                %reinterpret_cast_0 = memref.reinterpret_cast %arg0 to offset: [%25], sizes: [1, 128], strides: [512, 1] : memref<4096x512xf32> to memref<1x128xf32, strided<[512, 1], offset: ?>>
                %26 = loom.copy_to_tensor %reinterpret_cast_0, %16 on @L1, interconnect : [@vertical_links], broadcast : [8, 1] : memref<1x128xf32, strided<[512, 1], offset: ?>>, memref<1x128xf32> -> tensor<1x128xf32>
                %27 = arith.muli %arg7, %c128 : index
                %28 = arith.muli %12, %c64 : index
                %29 = affine.apply affine_map<(d0, d1) -> (d0 * 4096 + d1)>(%27, %28)
                %reinterpret_cast_1 = memref.reinterpret_cast %arg1 to offset: [%29], sizes: [128, 64], strides: [4096, 1] : memref<512x4096xf32> to memref<128x64xf32, strided<[4096, 1], offset: ?>>
                %30 = loom.copy_to_tensor %reinterpret_cast_1, %14 on @L1, interconnect : [], broadcast : [1, 1] : memref<128x64xf32, strided<[4096, 1], offset: ?>>, memref<128x64xf32> -> tensor<128x64xf32>
                %31 = linalg.matmul ins(%26, %30 : tensor<1x128xf32>, tensor<128x64xf32>) outs(%arg8 : tensor<1x64xf32>) -> tensor<1x64xf32>
                loom.semaphore_give %14 : memref<128x64xf32>
                loom.semaphore_give %16 : memref<1x128xf32>
                affine.yield %31 : tensor<1x64xf32>
              }
              %22 = arith.muli %12, %c64 : index
              %23 = affine.apply affine_map<(d0, d1) -> (d0 * 4096 + d1)>(%arg6, %22)
              %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%23], sizes: [1, 64], strides: [4096, 1] : memref<4096x4096xf32> to memref<1x64xf32, strided<[4096, 1], offset: ?>>
              loom.copy_from_tensor %21, %reinterpret_cast on @DRAM : tensor<1x64xf32>, memref<1x64xf32, strided<[4096, 1], offset: ?>>
              loom.semaphore_give %18 : memref<1x64xf32>
            } {loom.iter_type = #loom.iter_type<temporal>}
          } {loom.iter_type = #loom.iter_type<temporal>}
        } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @x}
      } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @y}
      return
    }
  }
}
