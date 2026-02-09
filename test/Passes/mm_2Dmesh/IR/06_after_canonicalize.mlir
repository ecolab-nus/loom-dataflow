module {
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
  module attributes {loom.pass_name = "Materialize"} {
    func.func @matmul__d0i0_d1i0__f01__d_d__BK64__BM64__BN64(%arg0: memref<4096x512xf32>, %arg1: memref<512x4096xf32>, %arg2: memref<4096x4096xf32>) {
      %cst = arith.constant 0.000000e+00 : f32
      %c64 = arith.constant 64 : index
      affine.parallel (%arg3) = (0) to (8) {
        affine.parallel (%arg4) = (0) to (8) {
          affine.for %arg5 = 0 to 1 {
            affine.for %arg6 = 0 to 64 {
              %12 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 8 + d1 + d2 * 64)>(%arg3, %arg4, %arg5)
              %13 = loom.alloc [64, 64] on @L1 : memref<64x64xf32>
              %14 = loom.init_tensor %13[64, 64] : memref<64x64xf32> -> tensor<64x64xf32>
              %15 = linalg.fill ins(%cst : f32) outs(%14 : tensor<64x64xf32>) -> tensor<64x64xf32>
              %16 = affine.for %arg7 = 0 to 8 iter_args(%arg8 = %15) -> (tensor<64x64xf32>) {
                %20 = arith.muli %12, %c64 : index
                %21 = arith.muli %arg7, %c64 : index
                %22 = affine.apply affine_map<(d0, d1) -> (d0 * 512 + d1)>(%20, %21)
                %reinterpret_cast_0 = memref.reinterpret_cast %arg0 to offset: [%22], sizes: [64, 64], strides: [512, 1] : memref<4096x512xf32> to memref<64x64xf32, strided<[512, 1], offset: ?>>
                %23 = loom.alloc [64, 64] on @L1 : memref<64x64xf32>
                %24 = loom.copy_to_tensor %reinterpret_cast_0, %23 on @L1, interconnect : [], broadcast : [1, 1] : memref<64x64xf32, strided<[512, 1], offset: ?>>, memref<64x64xf32> -> tensor<64x64xf32>
                %25 = arith.muli %arg6, %c64 : index
                %26 = affine.apply affine_map<(d0, d1) -> (d0 * 4096 + d1)>(%21, %25)
                %reinterpret_cast_1 = memref.reinterpret_cast %arg1 to offset: [%26], sizes: [64, 64], strides: [4096, 1] : memref<512x4096xf32> to memref<64x64xf32, strided<[4096, 1], offset: ?>>
                %27 = loom.alloc [64, 64] on @L1 : memref<64x64xf32>
                %28 = loom.copy_to_tensor %reinterpret_cast_1, %27 on @L1, interconnect : [], broadcast : [1, 1] : memref<64x64xf32, strided<[4096, 1], offset: ?>>, memref<64x64xf32> -> tensor<64x64xf32>
                %29 = linalg.matmul ins(%24, %28 : tensor<64x64xf32>, tensor<64x64xf32>) outs(%arg8 : tensor<64x64xf32>) -> tensor<64x64xf32>
                affine.yield %29 : tensor<64x64xf32>
              }
              %17 = arith.muli %12, %c64 : index
              %18 = arith.muli %arg6, %c64 : index
              %19 = affine.apply affine_map<(d0, d1) -> (d0 * 4096 + d1)>(%17, %18)
              %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%19], sizes: [64, 64], strides: [4096, 1] : memref<4096x4096xf32> to memref<64x64xf32, strided<[4096, 1], offset: ?>>
              loom.copy_from_tensor %16, %reinterpret_cast on @DRAM : tensor<64x64xf32>, memref<64x64xf32, strided<[4096, 1], offset: ?>>
            } {loom.iter_type = #loom.iter_type<temporal>}
          } {loom.iter_type = #loom.iter_type<temporal>}
        } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @y}
      } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @x}
      return
    }
    func.func @matmul__d0i0_d1i0__f01__d_d__BK256__BM32__BN32(%arg0: memref<4096x512xf32>, %arg1: memref<512x4096xf32>, %arg2: memref<4096x4096xf32>) {
      %cst = arith.constant 0.000000e+00 : f32
      %c32 = arith.constant 32 : index
      %c256 = arith.constant 256 : index
      affine.parallel (%arg3) = (0) to (8) {
        affine.parallel (%arg4) = (0) to (8) {
          affine.for %arg5 = 0 to 2 {
            affine.for %arg6 = 0 to 128 {
              %12 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 8 + d1 + d2 * 64)>(%arg3, %arg4, %arg5)
              %13 = loom.alloc [32, 32] on @L1 : memref<32x32xf32>
              %14 = loom.init_tensor %13[32, 32] : memref<32x32xf32> -> tensor<32x32xf32>
              %15 = linalg.fill ins(%cst : f32) outs(%14 : tensor<32x32xf32>) -> tensor<32x32xf32>
              %16 = affine.for %arg7 = 0 to 2 iter_args(%arg8 = %15) -> (tensor<32x32xf32>) {
                %20 = arith.muli %12, %c32 : index
                %21 = arith.muli %arg7, %c256 : index
                %22 = affine.apply affine_map<(d0, d1) -> (d0 * 512 + d1)>(%20, %21)
                %reinterpret_cast_0 = memref.reinterpret_cast %arg0 to offset: [%22], sizes: [32, 256], strides: [512, 1] : memref<4096x512xf32> to memref<32x256xf32, strided<[512, 1], offset: ?>>
                %23 = loom.alloc [32, 256] on @L1 : memref<32x256xf32>
                %24 = loom.copy_to_tensor %reinterpret_cast_0, %23 on @L1, interconnect : [], broadcast : [1, 1] : memref<32x256xf32, strided<[512, 1], offset: ?>>, memref<32x256xf32> -> tensor<32x256xf32>
                %25 = arith.muli %arg6, %c32 : index
                %26 = affine.apply affine_map<(d0, d1) -> (d0 * 4096 + d1)>(%21, %25)
                %reinterpret_cast_1 = memref.reinterpret_cast %arg1 to offset: [%26], sizes: [256, 32], strides: [4096, 1] : memref<512x4096xf32> to memref<256x32xf32, strided<[4096, 1], offset: ?>>
                %27 = loom.alloc [256, 32] on @L1 : memref<256x32xf32>
                %28 = loom.copy_to_tensor %reinterpret_cast_1, %27 on @L1, interconnect : [], broadcast : [1, 1] : memref<256x32xf32, strided<[4096, 1], offset: ?>>, memref<256x32xf32> -> tensor<256x32xf32>
                %29 = linalg.matmul ins(%24, %28 : tensor<32x256xf32>, tensor<256x32xf32>) outs(%arg8 : tensor<32x32xf32>) -> tensor<32x32xf32>
                affine.yield %29 : tensor<32x32xf32>
              }
              %17 = arith.muli %12, %c32 : index
              %18 = arith.muli %arg6, %c32 : index
              %19 = affine.apply affine_map<(d0, d1) -> (d0 * 4096 + d1)>(%17, %18)
              %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%19], sizes: [32, 32], strides: [4096, 1] : memref<4096x4096xf32> to memref<32x32xf32, strided<[4096, 1], offset: ?>>
              loom.copy_from_tensor %16, %reinterpret_cast on @DRAM : tensor<32x32xf32>, memref<32x32xf32, strided<[4096, 1], offset: ?>>
            } {loom.iter_type = #loom.iter_type<temporal>}
          } {loom.iter_type = #loom.iter_type<temporal>}
        } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @y}
      } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @x}
      return
    }
  }
  module attributes {loom.pass_name = "Materialize"} {
    func.func @matmul__d0i0_d1i0__f01__d_a__BK64__BM64__BN64(%arg0: memref<4096x512xf32>, %arg1: memref<512x4096xf32>, %arg2: memref<4096x4096xf32>) {
      %cst = arith.constant 0.000000e+00 : f32
      %c64 = arith.constant 64 : index
      affine.parallel (%arg3) = (0) to (8) {
        affine.parallel (%arg4) = (0) to (8) {
          affine.for %arg5 = 0 to 1 {
            affine.for %arg6 = 0 to 64 {
              %12 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 8 + d1 + d2 * 64)>(%arg3, %arg4, %arg5)
              %13 = loom.alloc [64, 64] on @L1 : memref<64x64xf32>
              %14 = loom.init_tensor %13[64, 64] : memref<64x64xf32> -> tensor<64x64xf32>
              %15 = linalg.fill ins(%cst : f32) outs(%14 : tensor<64x64xf32>) -> tensor<64x64xf32>
              %16 = affine.for %arg7 = 0 to 8 iter_args(%arg8 = %15) -> (tensor<64x64xf32>) {
                %20 = arith.muli %12, %c64 : index
                %21 = arith.muli %arg7, %c64 : index
                %22 = affine.apply affine_map<(d0, d1) -> (d0 * 512 + d1)>(%20, %21)
                %reinterpret_cast_0 = memref.reinterpret_cast %arg0 to offset: [%22], sizes: [64, 64], strides: [512, 1] : memref<4096x512xf32> to memref<64x64xf32, strided<[512, 1], offset: ?>>
                %23 = loom.alloc [64, 64] on @L1 : memref<64x64xf32>
                %24 = loom.copy_to_tensor %reinterpret_cast_0, %23 on @L1, interconnect : [], broadcast : [1, 1] : memref<64x64xf32, strided<[512, 1], offset: ?>>, memref<64x64xf32> -> tensor<64x64xf32>
                %25 = arith.muli %arg6, %c64 : index
                %26 = affine.apply affine_map<(d0, d1) -> (d0 * 4096 + d1)>(%21, %25)
                %reinterpret_cast_1 = memref.reinterpret_cast %arg1 to offset: [%26], sizes: [64, 64], strides: [4096, 1] : memref<512x4096xf32> to memref<64x64xf32, strided<[4096, 1], offset: ?>>
                %27 = loom.alloc [64, 64] on @L1 : memref<64x64xf32>
                %28 = loom.copy_to_tensor %reinterpret_cast_1, %27 on @L1, interconnect : [@horizontal_links, @vertical_links], broadcast : [8, 8] : memref<64x64xf32, strided<[4096, 1], offset: ?>>, memref<64x64xf32> -> tensor<64x64xf32>
                %29 = linalg.matmul ins(%24, %28 : tensor<64x64xf32>, tensor<64x64xf32>) outs(%arg8 : tensor<64x64xf32>) -> tensor<64x64xf32>
                affine.yield %29 : tensor<64x64xf32>
              }
              %17 = arith.muli %12, %c64 : index
              %18 = arith.muli %arg6, %c64 : index
              %19 = affine.apply affine_map<(d0, d1) -> (d0 * 4096 + d1)>(%17, %18)
              %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%19], sizes: [64, 64], strides: [4096, 1] : memref<4096x4096xf32> to memref<64x64xf32, strided<[4096, 1], offset: ?>>
              loom.copy_from_tensor %16, %reinterpret_cast on @DRAM : tensor<64x64xf32>, memref<64x64xf32, strided<[4096, 1], offset: ?>>
            } {loom.iter_type = #loom.iter_type<temporal>}
          } {loom.iter_type = #loom.iter_type<temporal>}
        } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @y}
      } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @x}
      return
    }
    func.func @matmul__d0i0_d1i0__f01__d_a__BK256__BM32__BN32(%arg0: memref<4096x512xf32>, %arg1: memref<512x4096xf32>, %arg2: memref<4096x4096xf32>) {
      %cst = arith.constant 0.000000e+00 : f32
      %c32 = arith.constant 32 : index
      %c256 = arith.constant 256 : index
      affine.parallel (%arg3) = (0) to (8) {
        affine.parallel (%arg4) = (0) to (8) {
          affine.for %arg5 = 0 to 2 {
            affine.for %arg6 = 0 to 128 {
              %12 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 8 + d1 + d2 * 64)>(%arg3, %arg4, %arg5)
              %13 = loom.alloc [32, 32] on @L1 : memref<32x32xf32>
              %14 = loom.init_tensor %13[32, 32] : memref<32x32xf32> -> tensor<32x32xf32>
              %15 = linalg.fill ins(%cst : f32) outs(%14 : tensor<32x32xf32>) -> tensor<32x32xf32>
              %16 = affine.for %arg7 = 0 to 2 iter_args(%arg8 = %15) -> (tensor<32x32xf32>) {
                %20 = arith.muli %12, %c32 : index
                %21 = arith.muli %arg7, %c256 : index
                %22 = affine.apply affine_map<(d0, d1) -> (d0 * 512 + d1)>(%20, %21)
                %reinterpret_cast_0 = memref.reinterpret_cast %arg0 to offset: [%22], sizes: [32, 256], strides: [512, 1] : memref<4096x512xf32> to memref<32x256xf32, strided<[512, 1], offset: ?>>
                %23 = loom.alloc [32, 256] on @L1 : memref<32x256xf32>
                %24 = loom.copy_to_tensor %reinterpret_cast_0, %23 on @L1, interconnect : [], broadcast : [1, 1] : memref<32x256xf32, strided<[512, 1], offset: ?>>, memref<32x256xf32> -> tensor<32x256xf32>
                %25 = arith.muli %arg6, %c32 : index
                %26 = affine.apply affine_map<(d0, d1) -> (d0 * 4096 + d1)>(%21, %25)
                %reinterpret_cast_1 = memref.reinterpret_cast %arg1 to offset: [%26], sizes: [256, 32], strides: [4096, 1] : memref<512x4096xf32> to memref<256x32xf32, strided<[4096, 1], offset: ?>>
                %27 = loom.alloc [256, 32] on @L1 : memref<256x32xf32>
                %28 = loom.copy_to_tensor %reinterpret_cast_1, %27 on @L1, interconnect : [@horizontal_links, @vertical_links], broadcast : [8, 8] : memref<256x32xf32, strided<[4096, 1], offset: ?>>, memref<256x32xf32> -> tensor<256x32xf32>
                %29 = linalg.matmul ins(%24, %28 : tensor<32x256xf32>, tensor<256x32xf32>) outs(%arg8 : tensor<32x32xf32>) -> tensor<32x32xf32>
                affine.yield %29 : tensor<32x32xf32>
              }
              %17 = arith.muli %12, %c32 : index
              %18 = arith.muli %arg6, %c32 : index
              %19 = affine.apply affine_map<(d0, d1) -> (d0 * 4096 + d1)>(%17, %18)
              %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%19], sizes: [32, 32], strides: [4096, 1] : memref<4096x4096xf32> to memref<32x32xf32, strided<[4096, 1], offset: ?>>
              loom.copy_from_tensor %16, %reinterpret_cast on @DRAM : tensor<32x32xf32>, memref<32x32xf32, strided<[4096, 1], offset: ?>>
            } {loom.iter_type = #loom.iter_type<temporal>}
          } {loom.iter_type = #loom.iter_type<temporal>}
        } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @y}
      } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @x}
      return
    }
  }
  module attributes {loom.pass_name = "Materialize"} {
    func.func @matmul__d0i0_d1i0__f01__d_h__BK64__BM64__BN64(%arg0: memref<4096x512xf32>, %arg1: memref<512x4096xf32>, %arg2: memref<4096x4096xf32>) {
      %cst = arith.constant 0.000000e+00 : f32
      %c64 = arith.constant 64 : index
      affine.parallel (%arg3) = (0) to (8) {
        affine.parallel (%arg4) = (0) to (8) {
          affine.for %arg5 = 0 to 1 {
            affine.for %arg6 = 0 to 64 {
              %12 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 8 + d1 + d2 * 64)>(%arg3, %arg4, %arg5)
              %13 = loom.alloc [64, 64] on @L1 : memref<64x64xf32>
              %14 = loom.init_tensor %13[64, 64] : memref<64x64xf32> -> tensor<64x64xf32>
              %15 = linalg.fill ins(%cst : f32) outs(%14 : tensor<64x64xf32>) -> tensor<64x64xf32>
              %16 = affine.for %arg7 = 0 to 8 iter_args(%arg8 = %15) -> (tensor<64x64xf32>) {
                %20 = arith.muli %12, %c64 : index
                %21 = arith.muli %arg7, %c64 : index
                %22 = affine.apply affine_map<(d0, d1) -> (d0 * 512 + d1)>(%20, %21)
                %reinterpret_cast_0 = memref.reinterpret_cast %arg0 to offset: [%22], sizes: [64, 64], strides: [512, 1] : memref<4096x512xf32> to memref<64x64xf32, strided<[512, 1], offset: ?>>
                %23 = loom.alloc [64, 64] on @L1 : memref<64x64xf32>
                %24 = loom.copy_to_tensor %reinterpret_cast_0, %23 on @L1, interconnect : [], broadcast : [1, 1] : memref<64x64xf32, strided<[512, 1], offset: ?>>, memref<64x64xf32> -> tensor<64x64xf32>
                %25 = arith.muli %arg6, %c64 : index
                %26 = affine.apply affine_map<(d0, d1) -> (d0 * 4096 + d1)>(%21, %25)
                %reinterpret_cast_1 = memref.reinterpret_cast %arg1 to offset: [%26], sizes: [64, 64], strides: [4096, 1] : memref<512x4096xf32> to memref<64x64xf32, strided<[4096, 1], offset: ?>>
                %27 = loom.alloc [64, 64] on @L1 : memref<64x64xf32>
                %28 = loom.copy_to_tensor %reinterpret_cast_1, %27 on @L1, interconnect : [@horizontal_links], broadcast : [1, 8] : memref<64x64xf32, strided<[4096, 1], offset: ?>>, memref<64x64xf32> -> tensor<64x64xf32>
                %29 = linalg.matmul ins(%24, %28 : tensor<64x64xf32>, tensor<64x64xf32>) outs(%arg8 : tensor<64x64xf32>) -> tensor<64x64xf32>
                affine.yield %29 : tensor<64x64xf32>
              }
              %17 = arith.muli %12, %c64 : index
              %18 = arith.muli %arg6, %c64 : index
              %19 = affine.apply affine_map<(d0, d1) -> (d0 * 4096 + d1)>(%17, %18)
              %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%19], sizes: [64, 64], strides: [4096, 1] : memref<4096x4096xf32> to memref<64x64xf32, strided<[4096, 1], offset: ?>>
              loom.copy_from_tensor %16, %reinterpret_cast on @DRAM : tensor<64x64xf32>, memref<64x64xf32, strided<[4096, 1], offset: ?>>
            } {loom.iter_type = #loom.iter_type<temporal>}
          } {loom.iter_type = #loom.iter_type<temporal>}
        } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @y}
      } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @x}
      return
    }
    func.func @matmul__d0i0_d1i0__f01__d_h__BK256__BM32__BN32(%arg0: memref<4096x512xf32>, %arg1: memref<512x4096xf32>, %arg2: memref<4096x4096xf32>) {
      %cst = arith.constant 0.000000e+00 : f32
      %c32 = arith.constant 32 : index
      %c256 = arith.constant 256 : index
      affine.parallel (%arg3) = (0) to (8) {
        affine.parallel (%arg4) = (0) to (8) {
          affine.for %arg5 = 0 to 2 {
            affine.for %arg6 = 0 to 128 {
              %12 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 8 + d1 + d2 * 64)>(%arg3, %arg4, %arg5)
              %13 = loom.alloc [32, 32] on @L1 : memref<32x32xf32>
              %14 = loom.init_tensor %13[32, 32] : memref<32x32xf32> -> tensor<32x32xf32>
              %15 = linalg.fill ins(%cst : f32) outs(%14 : tensor<32x32xf32>) -> tensor<32x32xf32>
              %16 = affine.for %arg7 = 0 to 2 iter_args(%arg8 = %15) -> (tensor<32x32xf32>) {
                %20 = arith.muli %12, %c32 : index
                %21 = arith.muli %arg7, %c256 : index
                %22 = affine.apply affine_map<(d0, d1) -> (d0 * 512 + d1)>(%20, %21)
                %reinterpret_cast_0 = memref.reinterpret_cast %arg0 to offset: [%22], sizes: [32, 256], strides: [512, 1] : memref<4096x512xf32> to memref<32x256xf32, strided<[512, 1], offset: ?>>
                %23 = loom.alloc [32, 256] on @L1 : memref<32x256xf32>
                %24 = loom.copy_to_tensor %reinterpret_cast_0, %23 on @L1, interconnect : [], broadcast : [1, 1] : memref<32x256xf32, strided<[512, 1], offset: ?>>, memref<32x256xf32> -> tensor<32x256xf32>
                %25 = arith.muli %arg6, %c32 : index
                %26 = affine.apply affine_map<(d0, d1) -> (d0 * 4096 + d1)>(%21, %25)
                %reinterpret_cast_1 = memref.reinterpret_cast %arg1 to offset: [%26], sizes: [256, 32], strides: [4096, 1] : memref<512x4096xf32> to memref<256x32xf32, strided<[4096, 1], offset: ?>>
                %27 = loom.alloc [256, 32] on @L1 : memref<256x32xf32>
                %28 = loom.copy_to_tensor %reinterpret_cast_1, %27 on @L1, interconnect : [@horizontal_links], broadcast : [1, 8] : memref<256x32xf32, strided<[4096, 1], offset: ?>>, memref<256x32xf32> -> tensor<256x32xf32>
                %29 = linalg.matmul ins(%24, %28 : tensor<32x256xf32>, tensor<256x32xf32>) outs(%arg8 : tensor<32x32xf32>) -> tensor<32x32xf32>
                affine.yield %29 : tensor<32x32xf32>
              }
              %17 = arith.muli %12, %c32 : index
              %18 = arith.muli %arg6, %c32 : index
              %19 = affine.apply affine_map<(d0, d1) -> (d0 * 4096 + d1)>(%17, %18)
              %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%19], sizes: [32, 32], strides: [4096, 1] : memref<4096x4096xf32> to memref<32x32xf32, strided<[4096, 1], offset: ?>>
              loom.copy_from_tensor %16, %reinterpret_cast on @DRAM : tensor<32x32xf32>, memref<32x32xf32, strided<[4096, 1], offset: ?>>
            } {loom.iter_type = #loom.iter_type<temporal>}
          } {loom.iter_type = #loom.iter_type<temporal>}
        } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @y}
      } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @x}
      return
    }
  }
  module attributes {loom.pass_name = "Materialize"} {
    func.func @matmul__d0i0_d1i0__f01__d_v__BK64__BM64__BN64(%arg0: memref<4096x512xf32>, %arg1: memref<512x4096xf32>, %arg2: memref<4096x4096xf32>) {
      %cst = arith.constant 0.000000e+00 : f32
      %c64 = arith.constant 64 : index
      affine.parallel (%arg3) = (0) to (8) {
        affine.parallel (%arg4) = (0) to (8) {
          affine.for %arg5 = 0 to 1 {
            affine.for %arg6 = 0 to 64 {
              %12 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 8 + d1 + d2 * 64)>(%arg3, %arg4, %arg5)
              %13 = loom.alloc [64, 64] on @L1 : memref<64x64xf32>
              %14 = loom.init_tensor %13[64, 64] : memref<64x64xf32> -> tensor<64x64xf32>
              %15 = linalg.fill ins(%cst : f32) outs(%14 : tensor<64x64xf32>) -> tensor<64x64xf32>
              %16 = affine.for %arg7 = 0 to 8 iter_args(%arg8 = %15) -> (tensor<64x64xf32>) {
                %20 = arith.muli %12, %c64 : index
                %21 = arith.muli %arg7, %c64 : index
                %22 = affine.apply affine_map<(d0, d1) -> (d0 * 512 + d1)>(%20, %21)
                %reinterpret_cast_0 = memref.reinterpret_cast %arg0 to offset: [%22], sizes: [64, 64], strides: [512, 1] : memref<4096x512xf32> to memref<64x64xf32, strided<[512, 1], offset: ?>>
                %23 = loom.alloc [64, 64] on @L1 : memref<64x64xf32>
                %24 = loom.copy_to_tensor %reinterpret_cast_0, %23 on @L1, interconnect : [], broadcast : [1, 1] : memref<64x64xf32, strided<[512, 1], offset: ?>>, memref<64x64xf32> -> tensor<64x64xf32>
                %25 = arith.muli %arg6, %c64 : index
                %26 = affine.apply affine_map<(d0, d1) -> (d0 * 4096 + d1)>(%21, %25)
                %reinterpret_cast_1 = memref.reinterpret_cast %arg1 to offset: [%26], sizes: [64, 64], strides: [4096, 1] : memref<512x4096xf32> to memref<64x64xf32, strided<[4096, 1], offset: ?>>
                %27 = loom.alloc [64, 64] on @L1 : memref<64x64xf32>
                %28 = loom.copy_to_tensor %reinterpret_cast_1, %27 on @L1, interconnect : [@vertical_links], broadcast : [8, 1] : memref<64x64xf32, strided<[4096, 1], offset: ?>>, memref<64x64xf32> -> tensor<64x64xf32>
                %29 = linalg.matmul ins(%24, %28 : tensor<64x64xf32>, tensor<64x64xf32>) outs(%arg8 : tensor<64x64xf32>) -> tensor<64x64xf32>
                affine.yield %29 : tensor<64x64xf32>
              }
              %17 = arith.muli %12, %c64 : index
              %18 = arith.muli %arg6, %c64 : index
              %19 = affine.apply affine_map<(d0, d1) -> (d0 * 4096 + d1)>(%17, %18)
              %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%19], sizes: [64, 64], strides: [4096, 1] : memref<4096x4096xf32> to memref<64x64xf32, strided<[4096, 1], offset: ?>>
              loom.copy_from_tensor %16, %reinterpret_cast on @DRAM : tensor<64x64xf32>, memref<64x64xf32, strided<[4096, 1], offset: ?>>
            } {loom.iter_type = #loom.iter_type<temporal>}
          } {loom.iter_type = #loom.iter_type<temporal>}
        } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @y}
      } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @x}
      return
    }
    func.func @matmul__d0i0_d1i0__f01__d_v__BK256__BM32__BN32(%arg0: memref<4096x512xf32>, %arg1: memref<512x4096xf32>, %arg2: memref<4096x4096xf32>) {
      %cst = arith.constant 0.000000e+00 : f32
      %c32 = arith.constant 32 : index
      %c256 = arith.constant 256 : index
      affine.parallel (%arg3) = (0) to (8) {
        affine.parallel (%arg4) = (0) to (8) {
          affine.for %arg5 = 0 to 2 {
            affine.for %arg6 = 0 to 128 {
              %12 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 8 + d1 + d2 * 64)>(%arg3, %arg4, %arg5)
              %13 = loom.alloc [32, 32] on @L1 : memref<32x32xf32>
              %14 = loom.init_tensor %13[32, 32] : memref<32x32xf32> -> tensor<32x32xf32>
              %15 = linalg.fill ins(%cst : f32) outs(%14 : tensor<32x32xf32>) -> tensor<32x32xf32>
              %16 = affine.for %arg7 = 0 to 2 iter_args(%arg8 = %15) -> (tensor<32x32xf32>) {
                %20 = arith.muli %12, %c32 : index
                %21 = arith.muli %arg7, %c256 : index
                %22 = affine.apply affine_map<(d0, d1) -> (d0 * 512 + d1)>(%20, %21)
                %reinterpret_cast_0 = memref.reinterpret_cast %arg0 to offset: [%22], sizes: [32, 256], strides: [512, 1] : memref<4096x512xf32> to memref<32x256xf32, strided<[512, 1], offset: ?>>
                %23 = loom.alloc [32, 256] on @L1 : memref<32x256xf32>
                %24 = loom.copy_to_tensor %reinterpret_cast_0, %23 on @L1, interconnect : [], broadcast : [1, 1] : memref<32x256xf32, strided<[512, 1], offset: ?>>, memref<32x256xf32> -> tensor<32x256xf32>
                %25 = arith.muli %arg6, %c32 : index
                %26 = affine.apply affine_map<(d0, d1) -> (d0 * 4096 + d1)>(%21, %25)
                %reinterpret_cast_1 = memref.reinterpret_cast %arg1 to offset: [%26], sizes: [256, 32], strides: [4096, 1] : memref<512x4096xf32> to memref<256x32xf32, strided<[4096, 1], offset: ?>>
                %27 = loom.alloc [256, 32] on @L1 : memref<256x32xf32>
                %28 = loom.copy_to_tensor %reinterpret_cast_1, %27 on @L1, interconnect : [@vertical_links], broadcast : [8, 1] : memref<256x32xf32, strided<[4096, 1], offset: ?>>, memref<256x32xf32> -> tensor<256x32xf32>
                %29 = linalg.matmul ins(%24, %28 : tensor<32x256xf32>, tensor<256x32xf32>) outs(%arg8 : tensor<32x32xf32>) -> tensor<32x32xf32>
                affine.yield %29 : tensor<32x32xf32>
              }
              %17 = arith.muli %12, %c32 : index
              %18 = arith.muli %arg6, %c32 : index
              %19 = affine.apply affine_map<(d0, d1) -> (d0 * 4096 + d1)>(%17, %18)
              %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%19], sizes: [32, 32], strides: [4096, 1] : memref<4096x4096xf32> to memref<32x32xf32, strided<[4096, 1], offset: ?>>
              loom.copy_from_tensor %16, %reinterpret_cast on @DRAM : tensor<32x32xf32>, memref<32x32xf32, strided<[4096, 1], offset: ?>>
            } {loom.iter_type = #loom.iter_type<temporal>}
          } {loom.iter_type = #loom.iter_type<temporal>}
        } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @y}
      } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @x}
      return
    }
  }
  module attributes {loom.pass_name = "Materialize"} {
    func.func @matmul__d0i0_d1i0__f10__d_d__BK64__BM64__BN64(%arg0: memref<4096x512xf32>, %arg1: memref<512x4096xf32>, %arg2: memref<4096x4096xf32>) {
      %cst = arith.constant 0.000000e+00 : f32
      %c64 = arith.constant 64 : index
      affine.parallel (%arg3) = (0) to (8) {
        affine.parallel (%arg4) = (0) to (8) {
          affine.for %arg5 = 0 to 64 {
            affine.for %arg6 = 0 to 1 {
              %12 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 8 + d1 + d2 * 64)>(%arg3, %arg4, %arg6)
              %13 = loom.alloc [64, 64] on @L1 : memref<64x64xf32>
              %14 = loom.init_tensor %13[64, 64] : memref<64x64xf32> -> tensor<64x64xf32>
              %15 = linalg.fill ins(%cst : f32) outs(%14 : tensor<64x64xf32>) -> tensor<64x64xf32>
              %16 = affine.for %arg7 = 0 to 8 iter_args(%arg8 = %15) -> (tensor<64x64xf32>) {
                %20 = arith.muli %12, %c64 : index
                %21 = arith.muli %arg7, %c64 : index
                %22 = affine.apply affine_map<(d0, d1) -> (d0 * 512 + d1)>(%20, %21)
                %reinterpret_cast_0 = memref.reinterpret_cast %arg0 to offset: [%22], sizes: [64, 64], strides: [512, 1] : memref<4096x512xf32> to memref<64x64xf32, strided<[512, 1], offset: ?>>
                %23 = loom.alloc [64, 64] on @L1 : memref<64x64xf32>
                %24 = loom.copy_to_tensor %reinterpret_cast_0, %23 on @L1, interconnect : [], broadcast : [1, 1] : memref<64x64xf32, strided<[512, 1], offset: ?>>, memref<64x64xf32> -> tensor<64x64xf32>
                %25 = arith.muli %arg5, %c64 : index
                %26 = affine.apply affine_map<(d0, d1) -> (d0 * 4096 + d1)>(%21, %25)
                %reinterpret_cast_1 = memref.reinterpret_cast %arg1 to offset: [%26], sizes: [64, 64], strides: [4096, 1] : memref<512x4096xf32> to memref<64x64xf32, strided<[4096, 1], offset: ?>>
                %27 = loom.alloc [64, 64] on @L1 : memref<64x64xf32>
                %28 = loom.copy_to_tensor %reinterpret_cast_1, %27 on @L1, interconnect : [], broadcast : [1, 1] : memref<64x64xf32, strided<[4096, 1], offset: ?>>, memref<64x64xf32> -> tensor<64x64xf32>
                %29 = linalg.matmul ins(%24, %28 : tensor<64x64xf32>, tensor<64x64xf32>) outs(%arg8 : tensor<64x64xf32>) -> tensor<64x64xf32>
                affine.yield %29 : tensor<64x64xf32>
              }
              %17 = arith.muli %12, %c64 : index
              %18 = arith.muli %arg5, %c64 : index
              %19 = affine.apply affine_map<(d0, d1) -> (d0 * 4096 + d1)>(%17, %18)
              %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%19], sizes: [64, 64], strides: [4096, 1] : memref<4096x4096xf32> to memref<64x64xf32, strided<[4096, 1], offset: ?>>
              loom.copy_from_tensor %16, %reinterpret_cast on @DRAM : tensor<64x64xf32>, memref<64x64xf32, strided<[4096, 1], offset: ?>>
            } {loom.iter_type = #loom.iter_type<temporal>}
          } {loom.iter_type = #loom.iter_type<temporal>}
        } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @y}
      } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @x}
      return
    }
    func.func @matmul__d0i0_d1i0__f10__d_d__BK256__BM32__BN32(%arg0: memref<4096x512xf32>, %arg1: memref<512x4096xf32>, %arg2: memref<4096x4096xf32>) {
      %cst = arith.constant 0.000000e+00 : f32
      %c32 = arith.constant 32 : index
      %c256 = arith.constant 256 : index
      affine.parallel (%arg3) = (0) to (8) {
        affine.parallel (%arg4) = (0) to (8) {
          affine.for %arg5 = 0 to 128 {
            affine.for %arg6 = 0 to 2 {
              %12 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 8 + d1 + d2 * 64)>(%arg3, %arg4, %arg6)
              %13 = loom.alloc [32, 32] on @L1 : memref<32x32xf32>
              %14 = loom.init_tensor %13[32, 32] : memref<32x32xf32> -> tensor<32x32xf32>
              %15 = linalg.fill ins(%cst : f32) outs(%14 : tensor<32x32xf32>) -> tensor<32x32xf32>
              %16 = affine.for %arg7 = 0 to 2 iter_args(%arg8 = %15) -> (tensor<32x32xf32>) {
                %20 = arith.muli %12, %c32 : index
                %21 = arith.muli %arg7, %c256 : index
                %22 = affine.apply affine_map<(d0, d1) -> (d0 * 512 + d1)>(%20, %21)
                %reinterpret_cast_0 = memref.reinterpret_cast %arg0 to offset: [%22], sizes: [32, 256], strides: [512, 1] : memref<4096x512xf32> to memref<32x256xf32, strided<[512, 1], offset: ?>>
                %23 = loom.alloc [32, 256] on @L1 : memref<32x256xf32>
                %24 = loom.copy_to_tensor %reinterpret_cast_0, %23 on @L1, interconnect : [], broadcast : [1, 1] : memref<32x256xf32, strided<[512, 1], offset: ?>>, memref<32x256xf32> -> tensor<32x256xf32>
                %25 = arith.muli %arg5, %c32 : index
                %26 = affine.apply affine_map<(d0, d1) -> (d0 * 4096 + d1)>(%21, %25)
                %reinterpret_cast_1 = memref.reinterpret_cast %arg1 to offset: [%26], sizes: [256, 32], strides: [4096, 1] : memref<512x4096xf32> to memref<256x32xf32, strided<[4096, 1], offset: ?>>
                %27 = loom.alloc [256, 32] on @L1 : memref<256x32xf32>
                %28 = loom.copy_to_tensor %reinterpret_cast_1, %27 on @L1, interconnect : [], broadcast : [1, 1] : memref<256x32xf32, strided<[4096, 1], offset: ?>>, memref<256x32xf32> -> tensor<256x32xf32>
                %29 = linalg.matmul ins(%24, %28 : tensor<32x256xf32>, tensor<256x32xf32>) outs(%arg8 : tensor<32x32xf32>) -> tensor<32x32xf32>
                affine.yield %29 : tensor<32x32xf32>
              }
              %17 = arith.muli %12, %c32 : index
              %18 = arith.muli %arg5, %c32 : index
              %19 = affine.apply affine_map<(d0, d1) -> (d0 * 4096 + d1)>(%17, %18)
              %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%19], sizes: [32, 32], strides: [4096, 1] : memref<4096x4096xf32> to memref<32x32xf32, strided<[4096, 1], offset: ?>>
              loom.copy_from_tensor %16, %reinterpret_cast on @DRAM : tensor<32x32xf32>, memref<32x32xf32, strided<[4096, 1], offset: ?>>
            } {loom.iter_type = #loom.iter_type<temporal>}
          } {loom.iter_type = #loom.iter_type<temporal>}
        } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @y}
      } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @x}
      return
    }
  }
  module attributes {loom.pass_name = "Materialize"} {
    func.func @matmul__d0i0_d1i0__f10__d_a__BK64__BM64__BN64(%arg0: memref<4096x512xf32>, %arg1: memref<512x4096xf32>, %arg2: memref<4096x4096xf32>) {
      %cst = arith.constant 0.000000e+00 : f32
      %c64 = arith.constant 64 : index
      affine.parallel (%arg3) = (0) to (8) {
        affine.parallel (%arg4) = (0) to (8) {
          affine.for %arg5 = 0 to 64 {
            affine.for %arg6 = 0 to 1 {
              %12 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 8 + d1 + d2 * 64)>(%arg3, %arg4, %arg6)
              %13 = loom.alloc [64, 64] on @L1 : memref<64x64xf32>
              %14 = loom.init_tensor %13[64, 64] : memref<64x64xf32> -> tensor<64x64xf32>
              %15 = linalg.fill ins(%cst : f32) outs(%14 : tensor<64x64xf32>) -> tensor<64x64xf32>
              %16 = affine.for %arg7 = 0 to 8 iter_args(%arg8 = %15) -> (tensor<64x64xf32>) {
                %20 = arith.muli %12, %c64 : index
                %21 = arith.muli %arg7, %c64 : index
                %22 = affine.apply affine_map<(d0, d1) -> (d0 * 512 + d1)>(%20, %21)
                %reinterpret_cast_0 = memref.reinterpret_cast %arg0 to offset: [%22], sizes: [64, 64], strides: [512, 1] : memref<4096x512xf32> to memref<64x64xf32, strided<[512, 1], offset: ?>>
                %23 = loom.alloc [64, 64] on @L1 : memref<64x64xf32>
                %24 = loom.copy_to_tensor %reinterpret_cast_0, %23 on @L1, interconnect : [], broadcast : [1, 1] : memref<64x64xf32, strided<[512, 1], offset: ?>>, memref<64x64xf32> -> tensor<64x64xf32>
                %25 = arith.muli %arg5, %c64 : index
                %26 = affine.apply affine_map<(d0, d1) -> (d0 * 4096 + d1)>(%21, %25)
                %reinterpret_cast_1 = memref.reinterpret_cast %arg1 to offset: [%26], sizes: [64, 64], strides: [4096, 1] : memref<512x4096xf32> to memref<64x64xf32, strided<[4096, 1], offset: ?>>
                %27 = loom.alloc [64, 64] on @L1 : memref<64x64xf32>
                %28 = loom.copy_to_tensor %reinterpret_cast_1, %27 on @L1, interconnect : [@horizontal_links, @vertical_links], broadcast : [8, 8] : memref<64x64xf32, strided<[4096, 1], offset: ?>>, memref<64x64xf32> -> tensor<64x64xf32>
                %29 = linalg.matmul ins(%24, %28 : tensor<64x64xf32>, tensor<64x64xf32>) outs(%arg8 : tensor<64x64xf32>) -> tensor<64x64xf32>
                affine.yield %29 : tensor<64x64xf32>
              }
              %17 = arith.muli %12, %c64 : index
              %18 = arith.muli %arg5, %c64 : index
              %19 = affine.apply affine_map<(d0, d1) -> (d0 * 4096 + d1)>(%17, %18)
              %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%19], sizes: [64, 64], strides: [4096, 1] : memref<4096x4096xf32> to memref<64x64xf32, strided<[4096, 1], offset: ?>>
              loom.copy_from_tensor %16, %reinterpret_cast on @DRAM : tensor<64x64xf32>, memref<64x64xf32, strided<[4096, 1], offset: ?>>
            } {loom.iter_type = #loom.iter_type<temporal>}
          } {loom.iter_type = #loom.iter_type<temporal>}
        } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @y}
      } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @x}
      return
    }
    func.func @matmul__d0i0_d1i0__f10__d_a__BK256__BM32__BN32(%arg0: memref<4096x512xf32>, %arg1: memref<512x4096xf32>, %arg2: memref<4096x4096xf32>) {
      %cst = arith.constant 0.000000e+00 : f32
      %c32 = arith.constant 32 : index
      %c256 = arith.constant 256 : index
      affine.parallel (%arg3) = (0) to (8) {
        affine.parallel (%arg4) = (0) to (8) {
          affine.for %arg5 = 0 to 128 {
            affine.for %arg6 = 0 to 2 {
              %12 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 8 + d1 + d2 * 64)>(%arg3, %arg4, %arg6)
              %13 = loom.alloc [32, 32] on @L1 : memref<32x32xf32>
              %14 = loom.init_tensor %13[32, 32] : memref<32x32xf32> -> tensor<32x32xf32>
              %15 = linalg.fill ins(%cst : f32) outs(%14 : tensor<32x32xf32>) -> tensor<32x32xf32>
              %16 = affine.for %arg7 = 0 to 2 iter_args(%arg8 = %15) -> (tensor<32x32xf32>) {
                %20 = arith.muli %12, %c32 : index
                %21 = arith.muli %arg7, %c256 : index
                %22 = affine.apply affine_map<(d0, d1) -> (d0 * 512 + d1)>(%20, %21)
                %reinterpret_cast_0 = memref.reinterpret_cast %arg0 to offset: [%22], sizes: [32, 256], strides: [512, 1] : memref<4096x512xf32> to memref<32x256xf32, strided<[512, 1], offset: ?>>
                %23 = loom.alloc [32, 256] on @L1 : memref<32x256xf32>
                %24 = loom.copy_to_tensor %reinterpret_cast_0, %23 on @L1, interconnect : [], broadcast : [1, 1] : memref<32x256xf32, strided<[512, 1], offset: ?>>, memref<32x256xf32> -> tensor<32x256xf32>
                %25 = arith.muli %arg5, %c32 : index
                %26 = affine.apply affine_map<(d0, d1) -> (d0 * 4096 + d1)>(%21, %25)
                %reinterpret_cast_1 = memref.reinterpret_cast %arg1 to offset: [%26], sizes: [256, 32], strides: [4096, 1] : memref<512x4096xf32> to memref<256x32xf32, strided<[4096, 1], offset: ?>>
                %27 = loom.alloc [256, 32] on @L1 : memref<256x32xf32>
                %28 = loom.copy_to_tensor %reinterpret_cast_1, %27 on @L1, interconnect : [@horizontal_links, @vertical_links], broadcast : [8, 8] : memref<256x32xf32, strided<[4096, 1], offset: ?>>, memref<256x32xf32> -> tensor<256x32xf32>
                %29 = linalg.matmul ins(%24, %28 : tensor<32x256xf32>, tensor<256x32xf32>) outs(%arg8 : tensor<32x32xf32>) -> tensor<32x32xf32>
                affine.yield %29 : tensor<32x32xf32>
              }
              %17 = arith.muli %12, %c32 : index
              %18 = arith.muli %arg5, %c32 : index
              %19 = affine.apply affine_map<(d0, d1) -> (d0 * 4096 + d1)>(%17, %18)
              %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%19], sizes: [32, 32], strides: [4096, 1] : memref<4096x4096xf32> to memref<32x32xf32, strided<[4096, 1], offset: ?>>
              loom.copy_from_tensor %16, %reinterpret_cast on @DRAM : tensor<32x32xf32>, memref<32x32xf32, strided<[4096, 1], offset: ?>>
            } {loom.iter_type = #loom.iter_type<temporal>}
          } {loom.iter_type = #loom.iter_type<temporal>}
        } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @y}
      } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @x}
      return
    }
  }
  module attributes {loom.pass_name = "Materialize"} {
    func.func @matmul__d0i0_d1i0__f10__d_h__BK64__BM64__BN64(%arg0: memref<4096x512xf32>, %arg1: memref<512x4096xf32>, %arg2: memref<4096x4096xf32>) {
      %cst = arith.constant 0.000000e+00 : f32
      %c64 = arith.constant 64 : index
      affine.parallel (%arg3) = (0) to (8) {
        affine.parallel (%arg4) = (0) to (8) {
          affine.for %arg5 = 0 to 64 {
            affine.for %arg6 = 0 to 1 {
              %12 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 8 + d1 + d2 * 64)>(%arg3, %arg4, %arg6)
              %13 = loom.alloc [64, 64] on @L1 : memref<64x64xf32>
              %14 = loom.init_tensor %13[64, 64] : memref<64x64xf32> -> tensor<64x64xf32>
              %15 = linalg.fill ins(%cst : f32) outs(%14 : tensor<64x64xf32>) -> tensor<64x64xf32>
              %16 = affine.for %arg7 = 0 to 8 iter_args(%arg8 = %15) -> (tensor<64x64xf32>) {
                %20 = arith.muli %12, %c64 : index
                %21 = arith.muli %arg7, %c64 : index
                %22 = affine.apply affine_map<(d0, d1) -> (d0 * 512 + d1)>(%20, %21)
                %reinterpret_cast_0 = memref.reinterpret_cast %arg0 to offset: [%22], sizes: [64, 64], strides: [512, 1] : memref<4096x512xf32> to memref<64x64xf32, strided<[512, 1], offset: ?>>
                %23 = loom.alloc [64, 64] on @L1 : memref<64x64xf32>
                %24 = loom.copy_to_tensor %reinterpret_cast_0, %23 on @L1, interconnect : [], broadcast : [1, 1] : memref<64x64xf32, strided<[512, 1], offset: ?>>, memref<64x64xf32> -> tensor<64x64xf32>
                %25 = arith.muli %arg5, %c64 : index
                %26 = affine.apply affine_map<(d0, d1) -> (d0 * 4096 + d1)>(%21, %25)
                %reinterpret_cast_1 = memref.reinterpret_cast %arg1 to offset: [%26], sizes: [64, 64], strides: [4096, 1] : memref<512x4096xf32> to memref<64x64xf32, strided<[4096, 1], offset: ?>>
                %27 = loom.alloc [64, 64] on @L1 : memref<64x64xf32>
                %28 = loom.copy_to_tensor %reinterpret_cast_1, %27 on @L1, interconnect : [@horizontal_links], broadcast : [1, 8] : memref<64x64xf32, strided<[4096, 1], offset: ?>>, memref<64x64xf32> -> tensor<64x64xf32>
                %29 = linalg.matmul ins(%24, %28 : tensor<64x64xf32>, tensor<64x64xf32>) outs(%arg8 : tensor<64x64xf32>) -> tensor<64x64xf32>
                affine.yield %29 : tensor<64x64xf32>
              }
              %17 = arith.muli %12, %c64 : index
              %18 = arith.muli %arg5, %c64 : index
              %19 = affine.apply affine_map<(d0, d1) -> (d0 * 4096 + d1)>(%17, %18)
              %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%19], sizes: [64, 64], strides: [4096, 1] : memref<4096x4096xf32> to memref<64x64xf32, strided<[4096, 1], offset: ?>>
              loom.copy_from_tensor %16, %reinterpret_cast on @DRAM : tensor<64x64xf32>, memref<64x64xf32, strided<[4096, 1], offset: ?>>
            } {loom.iter_type = #loom.iter_type<temporal>}
          } {loom.iter_type = #loom.iter_type<temporal>}
        } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @y}
      } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @x}
      return
    }
    func.func @matmul__d0i0_d1i0__f10__d_h__BK256__BM32__BN32(%arg0: memref<4096x512xf32>, %arg1: memref<512x4096xf32>, %arg2: memref<4096x4096xf32>) {
      %cst = arith.constant 0.000000e+00 : f32
      %c32 = arith.constant 32 : index
      %c256 = arith.constant 256 : index
      affine.parallel (%arg3) = (0) to (8) {
        affine.parallel (%arg4) = (0) to (8) {
          affine.for %arg5 = 0 to 128 {
            affine.for %arg6 = 0 to 2 {
              %12 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 8 + d1 + d2 * 64)>(%arg3, %arg4, %arg6)
              %13 = loom.alloc [32, 32] on @L1 : memref<32x32xf32>
              %14 = loom.init_tensor %13[32, 32] : memref<32x32xf32> -> tensor<32x32xf32>
              %15 = linalg.fill ins(%cst : f32) outs(%14 : tensor<32x32xf32>) -> tensor<32x32xf32>
              %16 = affine.for %arg7 = 0 to 2 iter_args(%arg8 = %15) -> (tensor<32x32xf32>) {
                %20 = arith.muli %12, %c32 : index
                %21 = arith.muli %arg7, %c256 : index
                %22 = affine.apply affine_map<(d0, d1) -> (d0 * 512 + d1)>(%20, %21)
                %reinterpret_cast_0 = memref.reinterpret_cast %arg0 to offset: [%22], sizes: [32, 256], strides: [512, 1] : memref<4096x512xf32> to memref<32x256xf32, strided<[512, 1], offset: ?>>
                %23 = loom.alloc [32, 256] on @L1 : memref<32x256xf32>
                %24 = loom.copy_to_tensor %reinterpret_cast_0, %23 on @L1, interconnect : [], broadcast : [1, 1] : memref<32x256xf32, strided<[512, 1], offset: ?>>, memref<32x256xf32> -> tensor<32x256xf32>
                %25 = arith.muli %arg5, %c32 : index
                %26 = affine.apply affine_map<(d0, d1) -> (d0 * 4096 + d1)>(%21, %25)
                %reinterpret_cast_1 = memref.reinterpret_cast %arg1 to offset: [%26], sizes: [256, 32], strides: [4096, 1] : memref<512x4096xf32> to memref<256x32xf32, strided<[4096, 1], offset: ?>>
                %27 = loom.alloc [256, 32] on @L1 : memref<256x32xf32>
                %28 = loom.copy_to_tensor %reinterpret_cast_1, %27 on @L1, interconnect : [@horizontal_links], broadcast : [1, 8] : memref<256x32xf32, strided<[4096, 1], offset: ?>>, memref<256x32xf32> -> tensor<256x32xf32>
                %29 = linalg.matmul ins(%24, %28 : tensor<32x256xf32>, tensor<256x32xf32>) outs(%arg8 : tensor<32x32xf32>) -> tensor<32x32xf32>
                affine.yield %29 : tensor<32x32xf32>
              }
              %17 = arith.muli %12, %c32 : index
              %18 = arith.muli %arg5, %c32 : index
              %19 = affine.apply affine_map<(d0, d1) -> (d0 * 4096 + d1)>(%17, %18)
              %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%19], sizes: [32, 32], strides: [4096, 1] : memref<4096x4096xf32> to memref<32x32xf32, strided<[4096, 1], offset: ?>>
              loom.copy_from_tensor %16, %reinterpret_cast on @DRAM : tensor<32x32xf32>, memref<32x32xf32, strided<[4096, 1], offset: ?>>
            } {loom.iter_type = #loom.iter_type<temporal>}
          } {loom.iter_type = #loom.iter_type<temporal>}
        } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @y}
      } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @x}
      return
    }
  }
  module attributes {loom.pass_name = "Materialize"} {
    func.func @matmul__d0i0_d1i0__f10__d_v__BK64__BM64__BN64(%arg0: memref<4096x512xf32>, %arg1: memref<512x4096xf32>, %arg2: memref<4096x4096xf32>) {
      %cst = arith.constant 0.000000e+00 : f32
      %c64 = arith.constant 64 : index
      affine.parallel (%arg3) = (0) to (8) {
        affine.parallel (%arg4) = (0) to (8) {
          affine.for %arg5 = 0 to 64 {
            affine.for %arg6 = 0 to 1 {
              %12 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 8 + d1 + d2 * 64)>(%arg3, %arg4, %arg6)
              %13 = loom.alloc [64, 64] on @L1 : memref<64x64xf32>
              %14 = loom.init_tensor %13[64, 64] : memref<64x64xf32> -> tensor<64x64xf32>
              %15 = linalg.fill ins(%cst : f32) outs(%14 : tensor<64x64xf32>) -> tensor<64x64xf32>
              %16 = affine.for %arg7 = 0 to 8 iter_args(%arg8 = %15) -> (tensor<64x64xf32>) {
                %20 = arith.muli %12, %c64 : index
                %21 = arith.muli %arg7, %c64 : index
                %22 = affine.apply affine_map<(d0, d1) -> (d0 * 512 + d1)>(%20, %21)
                %reinterpret_cast_0 = memref.reinterpret_cast %arg0 to offset: [%22], sizes: [64, 64], strides: [512, 1] : memref<4096x512xf32> to memref<64x64xf32, strided<[512, 1], offset: ?>>
                %23 = loom.alloc [64, 64] on @L1 : memref<64x64xf32>
                %24 = loom.copy_to_tensor %reinterpret_cast_0, %23 on @L1, interconnect : [], broadcast : [1, 1] : memref<64x64xf32, strided<[512, 1], offset: ?>>, memref<64x64xf32> -> tensor<64x64xf32>
                %25 = arith.muli %arg5, %c64 : index
                %26 = affine.apply affine_map<(d0, d1) -> (d0 * 4096 + d1)>(%21, %25)
                %reinterpret_cast_1 = memref.reinterpret_cast %arg1 to offset: [%26], sizes: [64, 64], strides: [4096, 1] : memref<512x4096xf32> to memref<64x64xf32, strided<[4096, 1], offset: ?>>
                %27 = loom.alloc [64, 64] on @L1 : memref<64x64xf32>
                %28 = loom.copy_to_tensor %reinterpret_cast_1, %27 on @L1, interconnect : [@vertical_links], broadcast : [8, 1] : memref<64x64xf32, strided<[4096, 1], offset: ?>>, memref<64x64xf32> -> tensor<64x64xf32>
                %29 = linalg.matmul ins(%24, %28 : tensor<64x64xf32>, tensor<64x64xf32>) outs(%arg8 : tensor<64x64xf32>) -> tensor<64x64xf32>
                affine.yield %29 : tensor<64x64xf32>
              }
              %17 = arith.muli %12, %c64 : index
              %18 = arith.muli %arg5, %c64 : index
              %19 = affine.apply affine_map<(d0, d1) -> (d0 * 4096 + d1)>(%17, %18)
              %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%19], sizes: [64, 64], strides: [4096, 1] : memref<4096x4096xf32> to memref<64x64xf32, strided<[4096, 1], offset: ?>>
              loom.copy_from_tensor %16, %reinterpret_cast on @DRAM : tensor<64x64xf32>, memref<64x64xf32, strided<[4096, 1], offset: ?>>
            } {loom.iter_type = #loom.iter_type<temporal>}
          } {loom.iter_type = #loom.iter_type<temporal>}
        } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @y}
      } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @x}
      return
    }
    func.func @matmul__d0i0_d1i0__f10__d_v__BK256__BM32__BN32(%arg0: memref<4096x512xf32>, %arg1: memref<512x4096xf32>, %arg2: memref<4096x4096xf32>) {
      %cst = arith.constant 0.000000e+00 : f32
      %c32 = arith.constant 32 : index
      %c256 = arith.constant 256 : index
      affine.parallel (%arg3) = (0) to (8) {
        affine.parallel (%arg4) = (0) to (8) {
          affine.for %arg5 = 0 to 128 {
            affine.for %arg6 = 0 to 2 {
              %12 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 8 + d1 + d2 * 64)>(%arg3, %arg4, %arg6)
              %13 = loom.alloc [32, 32] on @L1 : memref<32x32xf32>
              %14 = loom.init_tensor %13[32, 32] : memref<32x32xf32> -> tensor<32x32xf32>
              %15 = linalg.fill ins(%cst : f32) outs(%14 : tensor<32x32xf32>) -> tensor<32x32xf32>
              %16 = affine.for %arg7 = 0 to 2 iter_args(%arg8 = %15) -> (tensor<32x32xf32>) {
                %20 = arith.muli %12, %c32 : index
                %21 = arith.muli %arg7, %c256 : index
                %22 = affine.apply affine_map<(d0, d1) -> (d0 * 512 + d1)>(%20, %21)
                %reinterpret_cast_0 = memref.reinterpret_cast %arg0 to offset: [%22], sizes: [32, 256], strides: [512, 1] : memref<4096x512xf32> to memref<32x256xf32, strided<[512, 1], offset: ?>>
                %23 = loom.alloc [32, 256] on @L1 : memref<32x256xf32>
                %24 = loom.copy_to_tensor %reinterpret_cast_0, %23 on @L1, interconnect : [], broadcast : [1, 1] : memref<32x256xf32, strided<[512, 1], offset: ?>>, memref<32x256xf32> -> tensor<32x256xf32>
                %25 = arith.muli %arg5, %c32 : index
                %26 = affine.apply affine_map<(d0, d1) -> (d0 * 4096 + d1)>(%21, %25)
                %reinterpret_cast_1 = memref.reinterpret_cast %arg1 to offset: [%26], sizes: [256, 32], strides: [4096, 1] : memref<512x4096xf32> to memref<256x32xf32, strided<[4096, 1], offset: ?>>
                %27 = loom.alloc [256, 32] on @L1 : memref<256x32xf32>
                %28 = loom.copy_to_tensor %reinterpret_cast_1, %27 on @L1, interconnect : [@vertical_links], broadcast : [8, 1] : memref<256x32xf32, strided<[4096, 1], offset: ?>>, memref<256x32xf32> -> tensor<256x32xf32>
                %29 = linalg.matmul ins(%24, %28 : tensor<32x256xf32>, tensor<256x32xf32>) outs(%arg8 : tensor<32x32xf32>) -> tensor<32x32xf32>
                affine.yield %29 : tensor<32x32xf32>
              }
              %17 = arith.muli %12, %c32 : index
              %18 = arith.muli %arg5, %c32 : index
              %19 = affine.apply affine_map<(d0, d1) -> (d0 * 4096 + d1)>(%17, %18)
              %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%19], sizes: [32, 32], strides: [4096, 1] : memref<4096x4096xf32> to memref<32x32xf32, strided<[4096, 1], offset: ?>>
              loom.copy_from_tensor %16, %reinterpret_cast on @DRAM : tensor<32x32xf32>, memref<32x32xf32, strided<[4096, 1], offset: ?>>
            } {loom.iter_type = #loom.iter_type<temporal>}
          } {loom.iter_type = #loom.iter_type<temporal>}
        } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @y}
      } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @x}
      return
    }
  }
  module attributes {loom.pass_name = "Materialize"} {
    func.func @matmul__d0i0_d1i1__f01__d_d__BK64__BM64__BN64(%arg0: memref<4096x512xf32>, %arg1: memref<512x4096xf32>, %arg2: memref<4096x4096xf32>) {
      %cst = arith.constant 0.000000e+00 : f32
      %c64 = arith.constant 64 : index
      affine.parallel (%arg3) = (0) to (8) {
        affine.parallel (%arg4) = (0) to (8) {
          affine.for %arg5 = 0 to 8 {
            affine.for %arg6 = 0 to 8 {
              %12 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 8)>(%arg3, %arg5)
              %13 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 8)>(%arg4, %arg6)
              %14 = loom.alloc [64, 64] on @L1 : memref<64x64xf32>
              %15 = loom.init_tensor %14[64, 64] : memref<64x64xf32> -> tensor<64x64xf32>
              %16 = linalg.fill ins(%cst : f32) outs(%15 : tensor<64x64xf32>) -> tensor<64x64xf32>
              %17 = affine.for %arg7 = 0 to 8 iter_args(%arg8 = %16) -> (tensor<64x64xf32>) {
                %21 = arith.muli %12, %c64 : index
                %22 = arith.muli %arg7, %c64 : index
                %23 = affine.apply affine_map<(d0, d1) -> (d0 * 512 + d1)>(%21, %22)
                %reinterpret_cast_0 = memref.reinterpret_cast %arg0 to offset: [%23], sizes: [64, 64], strides: [512, 1] : memref<4096x512xf32> to memref<64x64xf32, strided<[512, 1], offset: ?>>
                %24 = loom.alloc [64, 64] on @L1 : memref<64x64xf32>
                %25 = loom.copy_to_tensor %reinterpret_cast_0, %24 on @L1, interconnect : [], broadcast : [1, 1] : memref<64x64xf32, strided<[512, 1], offset: ?>>, memref<64x64xf32> -> tensor<64x64xf32>
                %26 = arith.muli %13, %c64 : index
                %27 = affine.apply affine_map<(d0, d1) -> (d0 * 4096 + d1)>(%22, %26)
                %reinterpret_cast_1 = memref.reinterpret_cast %arg1 to offset: [%27], sizes: [64, 64], strides: [4096, 1] : memref<512x4096xf32> to memref<64x64xf32, strided<[4096, 1], offset: ?>>
                %28 = loom.alloc [64, 64] on @L1 : memref<64x64xf32>
                %29 = loom.copy_to_tensor %reinterpret_cast_1, %28 on @L1, interconnect : [], broadcast : [1, 1] : memref<64x64xf32, strided<[4096, 1], offset: ?>>, memref<64x64xf32> -> tensor<64x64xf32>
                %30 = linalg.matmul ins(%25, %29 : tensor<64x64xf32>, tensor<64x64xf32>) outs(%arg8 : tensor<64x64xf32>) -> tensor<64x64xf32>
                affine.yield %30 : tensor<64x64xf32>
              }
              %18 = arith.muli %12, %c64 : index
              %19 = arith.muli %13, %c64 : index
              %20 = affine.apply affine_map<(d0, d1) -> (d0 * 4096 + d1)>(%18, %19)
              %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%20], sizes: [64, 64], strides: [4096, 1] : memref<4096x4096xf32> to memref<64x64xf32, strided<[4096, 1], offset: ?>>
              loom.copy_from_tensor %17, %reinterpret_cast on @DRAM : tensor<64x64xf32>, memref<64x64xf32, strided<[4096, 1], offset: ?>>
            } {loom.iter_type = #loom.iter_type<temporal>}
          } {loom.iter_type = #loom.iter_type<temporal>}
        } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @y}
      } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @x}
      return
    }
    func.func @matmul__d0i0_d1i1__f01__d_d__BK256__BM32__BN32(%arg0: memref<4096x512xf32>, %arg1: memref<512x4096xf32>, %arg2: memref<4096x4096xf32>) {
      %cst = arith.constant 0.000000e+00 : f32
      %c32 = arith.constant 32 : index
      %c256 = arith.constant 256 : index
      affine.parallel (%arg3) = (0) to (8) {
        affine.parallel (%arg4) = (0) to (8) {
          affine.for %arg5 = 0 to 16 {
            affine.for %arg6 = 0 to 16 {
              %12 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 8)>(%arg3, %arg5)
              %13 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 8)>(%arg4, %arg6)
              %14 = loom.alloc [32, 32] on @L1 : memref<32x32xf32>
              %15 = loom.init_tensor %14[32, 32] : memref<32x32xf32> -> tensor<32x32xf32>
              %16 = linalg.fill ins(%cst : f32) outs(%15 : tensor<32x32xf32>) -> tensor<32x32xf32>
              %17 = affine.for %arg7 = 0 to 2 iter_args(%arg8 = %16) -> (tensor<32x32xf32>) {
                %21 = arith.muli %12, %c32 : index
                %22 = arith.muli %arg7, %c256 : index
                %23 = affine.apply affine_map<(d0, d1) -> (d0 * 512 + d1)>(%21, %22)
                %reinterpret_cast_0 = memref.reinterpret_cast %arg0 to offset: [%23], sizes: [32, 256], strides: [512, 1] : memref<4096x512xf32> to memref<32x256xf32, strided<[512, 1], offset: ?>>
                %24 = loom.alloc [32, 256] on @L1 : memref<32x256xf32>
                %25 = loom.copy_to_tensor %reinterpret_cast_0, %24 on @L1, interconnect : [], broadcast : [1, 1] : memref<32x256xf32, strided<[512, 1], offset: ?>>, memref<32x256xf32> -> tensor<32x256xf32>
                %26 = arith.muli %13, %c32 : index
                %27 = affine.apply affine_map<(d0, d1) -> (d0 * 4096 + d1)>(%22, %26)
                %reinterpret_cast_1 = memref.reinterpret_cast %arg1 to offset: [%27], sizes: [256, 32], strides: [4096, 1] : memref<512x4096xf32> to memref<256x32xf32, strided<[4096, 1], offset: ?>>
                %28 = loom.alloc [256, 32] on @L1 : memref<256x32xf32>
                %29 = loom.copy_to_tensor %reinterpret_cast_1, %28 on @L1, interconnect : [], broadcast : [1, 1] : memref<256x32xf32, strided<[4096, 1], offset: ?>>, memref<256x32xf32> -> tensor<256x32xf32>
                %30 = linalg.matmul ins(%25, %29 : tensor<32x256xf32>, tensor<256x32xf32>) outs(%arg8 : tensor<32x32xf32>) -> tensor<32x32xf32>
                affine.yield %30 : tensor<32x32xf32>
              }
              %18 = arith.muli %12, %c32 : index
              %19 = arith.muli %13, %c32 : index
              %20 = affine.apply affine_map<(d0, d1) -> (d0 * 4096 + d1)>(%18, %19)
              %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%20], sizes: [32, 32], strides: [4096, 1] : memref<4096x4096xf32> to memref<32x32xf32, strided<[4096, 1], offset: ?>>
              loom.copy_from_tensor %17, %reinterpret_cast on @DRAM : tensor<32x32xf32>, memref<32x32xf32, strided<[4096, 1], offset: ?>>
            } {loom.iter_type = #loom.iter_type<temporal>}
          } {loom.iter_type = #loom.iter_type<temporal>}
        } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @y}
      } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @x}
      return
    }
  }
  module attributes {loom.pass_name = "Materialize"} {
    func.func @matmul__d0i0_d1i1__f01__d_h__BK64__BM64__BN64(%arg0: memref<4096x512xf32>, %arg1: memref<512x4096xf32>, %arg2: memref<4096x4096xf32>) {
      %cst = arith.constant 0.000000e+00 : f32
      %c64 = arith.constant 64 : index
      affine.parallel (%arg3) = (0) to (8) {
        affine.parallel (%arg4) = (0) to (8) {
          affine.for %arg5 = 0 to 8 {
            affine.for %arg6 = 0 to 8 {
              %12 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 8)>(%arg3, %arg5)
              %13 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 8)>(%arg4, %arg6)
              %14 = loom.alloc [64, 64] on @L1 : memref<64x64xf32>
              %15 = loom.init_tensor %14[64, 64] : memref<64x64xf32> -> tensor<64x64xf32>
              %16 = linalg.fill ins(%cst : f32) outs(%15 : tensor<64x64xf32>) -> tensor<64x64xf32>
              %17 = affine.for %arg7 = 0 to 8 iter_args(%arg8 = %16) -> (tensor<64x64xf32>) {
                %21 = arith.muli %12, %c64 : index
                %22 = arith.muli %arg7, %c64 : index
                %23 = affine.apply affine_map<(d0, d1) -> (d0 * 512 + d1)>(%21, %22)
                %reinterpret_cast_0 = memref.reinterpret_cast %arg0 to offset: [%23], sizes: [64, 64], strides: [512, 1] : memref<4096x512xf32> to memref<64x64xf32, strided<[512, 1], offset: ?>>
                %24 = loom.alloc [64, 64] on @L1 : memref<64x64xf32>
                %25 = loom.copy_to_tensor %reinterpret_cast_0, %24 on @L1, interconnect : [], broadcast : [1, 1] : memref<64x64xf32, strided<[512, 1], offset: ?>>, memref<64x64xf32> -> tensor<64x64xf32>
                %26 = arith.muli %13, %c64 : index
                %27 = affine.apply affine_map<(d0, d1) -> (d0 * 4096 + d1)>(%22, %26)
                %reinterpret_cast_1 = memref.reinterpret_cast %arg1 to offset: [%27], sizes: [64, 64], strides: [4096, 1] : memref<512x4096xf32> to memref<64x64xf32, strided<[4096, 1], offset: ?>>
                %28 = loom.alloc [64, 64] on @L1 : memref<64x64xf32>
                %29 = loom.copy_to_tensor %reinterpret_cast_1, %28 on @L1, interconnect : [@horizontal_links], broadcast : [1, 8] : memref<64x64xf32, strided<[4096, 1], offset: ?>>, memref<64x64xf32> -> tensor<64x64xf32>
                %30 = linalg.matmul ins(%25, %29 : tensor<64x64xf32>, tensor<64x64xf32>) outs(%arg8 : tensor<64x64xf32>) -> tensor<64x64xf32>
                affine.yield %30 : tensor<64x64xf32>
              }
              %18 = arith.muli %12, %c64 : index
              %19 = arith.muli %13, %c64 : index
              %20 = affine.apply affine_map<(d0, d1) -> (d0 * 4096 + d1)>(%18, %19)
              %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%20], sizes: [64, 64], strides: [4096, 1] : memref<4096x4096xf32> to memref<64x64xf32, strided<[4096, 1], offset: ?>>
              loom.copy_from_tensor %17, %reinterpret_cast on @DRAM : tensor<64x64xf32>, memref<64x64xf32, strided<[4096, 1], offset: ?>>
            } {loom.iter_type = #loom.iter_type<temporal>}
          } {loom.iter_type = #loom.iter_type<temporal>}
        } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @y}
      } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @x}
      return
    }
    func.func @matmul__d0i0_d1i1__f01__d_h__BK256__BM32__BN32(%arg0: memref<4096x512xf32>, %arg1: memref<512x4096xf32>, %arg2: memref<4096x4096xf32>) {
      %cst = arith.constant 0.000000e+00 : f32
      %c32 = arith.constant 32 : index
      %c256 = arith.constant 256 : index
      affine.parallel (%arg3) = (0) to (8) {
        affine.parallel (%arg4) = (0) to (8) {
          affine.for %arg5 = 0 to 16 {
            affine.for %arg6 = 0 to 16 {
              %12 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 8)>(%arg3, %arg5)
              %13 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 8)>(%arg4, %arg6)
              %14 = loom.alloc [32, 32] on @L1 : memref<32x32xf32>
              %15 = loom.init_tensor %14[32, 32] : memref<32x32xf32> -> tensor<32x32xf32>
              %16 = linalg.fill ins(%cst : f32) outs(%15 : tensor<32x32xf32>) -> tensor<32x32xf32>
              %17 = affine.for %arg7 = 0 to 2 iter_args(%arg8 = %16) -> (tensor<32x32xf32>) {
                %21 = arith.muli %12, %c32 : index
                %22 = arith.muli %arg7, %c256 : index
                %23 = affine.apply affine_map<(d0, d1) -> (d0 * 512 + d1)>(%21, %22)
                %reinterpret_cast_0 = memref.reinterpret_cast %arg0 to offset: [%23], sizes: [32, 256], strides: [512, 1] : memref<4096x512xf32> to memref<32x256xf32, strided<[512, 1], offset: ?>>
                %24 = loom.alloc [32, 256] on @L1 : memref<32x256xf32>
                %25 = loom.copy_to_tensor %reinterpret_cast_0, %24 on @L1, interconnect : [], broadcast : [1, 1] : memref<32x256xf32, strided<[512, 1], offset: ?>>, memref<32x256xf32> -> tensor<32x256xf32>
                %26 = arith.muli %13, %c32 : index
                %27 = affine.apply affine_map<(d0, d1) -> (d0 * 4096 + d1)>(%22, %26)
                %reinterpret_cast_1 = memref.reinterpret_cast %arg1 to offset: [%27], sizes: [256, 32], strides: [4096, 1] : memref<512x4096xf32> to memref<256x32xf32, strided<[4096, 1], offset: ?>>
                %28 = loom.alloc [256, 32] on @L1 : memref<256x32xf32>
                %29 = loom.copy_to_tensor %reinterpret_cast_1, %28 on @L1, interconnect : [@horizontal_links], broadcast : [1, 8] : memref<256x32xf32, strided<[4096, 1], offset: ?>>, memref<256x32xf32> -> tensor<256x32xf32>
                %30 = linalg.matmul ins(%25, %29 : tensor<32x256xf32>, tensor<256x32xf32>) outs(%arg8 : tensor<32x32xf32>) -> tensor<32x32xf32>
                affine.yield %30 : tensor<32x32xf32>
              }
              %18 = arith.muli %12, %c32 : index
              %19 = arith.muli %13, %c32 : index
              %20 = affine.apply affine_map<(d0, d1) -> (d0 * 4096 + d1)>(%18, %19)
              %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%20], sizes: [32, 32], strides: [4096, 1] : memref<4096x4096xf32> to memref<32x32xf32, strided<[4096, 1], offset: ?>>
              loom.copy_from_tensor %17, %reinterpret_cast on @DRAM : tensor<32x32xf32>, memref<32x32xf32, strided<[4096, 1], offset: ?>>
            } {loom.iter_type = #loom.iter_type<temporal>}
          } {loom.iter_type = #loom.iter_type<temporal>}
        } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @y}
      } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @x}
      return
    }
  }
  module attributes {loom.pass_name = "Materialize"} {
    func.func @matmul__d0i0_d1i1__f01__v_d__BK64__BM64__BN64(%arg0: memref<4096x512xf32>, %arg1: memref<512x4096xf32>, %arg2: memref<4096x4096xf32>) {
      %cst = arith.constant 0.000000e+00 : f32
      %c64 = arith.constant 64 : index
      affine.parallel (%arg3) = (0) to (8) {
        affine.parallel (%arg4) = (0) to (8) {
          affine.for %arg5 = 0 to 8 {
            affine.for %arg6 = 0 to 8 {
              %12 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 8)>(%arg3, %arg5)
              %13 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 8)>(%arg4, %arg6)
              %14 = loom.alloc [64, 64] on @L1 : memref<64x64xf32>
              %15 = loom.init_tensor %14[64, 64] : memref<64x64xf32> -> tensor<64x64xf32>
              %16 = linalg.fill ins(%cst : f32) outs(%15 : tensor<64x64xf32>) -> tensor<64x64xf32>
              %17 = affine.for %arg7 = 0 to 8 iter_args(%arg8 = %16) -> (tensor<64x64xf32>) {
                %21 = arith.muli %12, %c64 : index
                %22 = arith.muli %arg7, %c64 : index
                %23 = affine.apply affine_map<(d0, d1) -> (d0 * 512 + d1)>(%21, %22)
                %reinterpret_cast_0 = memref.reinterpret_cast %arg0 to offset: [%23], sizes: [64, 64], strides: [512, 1] : memref<4096x512xf32> to memref<64x64xf32, strided<[512, 1], offset: ?>>
                %24 = loom.alloc [64, 64] on @L1 : memref<64x64xf32>
                %25 = loom.copy_to_tensor %reinterpret_cast_0, %24 on @L1, interconnect : [@vertical_links], broadcast : [8, 1] : memref<64x64xf32, strided<[512, 1], offset: ?>>, memref<64x64xf32> -> tensor<64x64xf32>
                %26 = arith.muli %13, %c64 : index
                %27 = affine.apply affine_map<(d0, d1) -> (d0 * 4096 + d1)>(%22, %26)
                %reinterpret_cast_1 = memref.reinterpret_cast %arg1 to offset: [%27], sizes: [64, 64], strides: [4096, 1] : memref<512x4096xf32> to memref<64x64xf32, strided<[4096, 1], offset: ?>>
                %28 = loom.alloc [64, 64] on @L1 : memref<64x64xf32>
                %29 = loom.copy_to_tensor %reinterpret_cast_1, %28 on @L1, interconnect : [], broadcast : [1, 1] : memref<64x64xf32, strided<[4096, 1], offset: ?>>, memref<64x64xf32> -> tensor<64x64xf32>
                %30 = linalg.matmul ins(%25, %29 : tensor<64x64xf32>, tensor<64x64xf32>) outs(%arg8 : tensor<64x64xf32>) -> tensor<64x64xf32>
                affine.yield %30 : tensor<64x64xf32>
              }
              %18 = arith.muli %12, %c64 : index
              %19 = arith.muli %13, %c64 : index
              %20 = affine.apply affine_map<(d0, d1) -> (d0 * 4096 + d1)>(%18, %19)
              %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%20], sizes: [64, 64], strides: [4096, 1] : memref<4096x4096xf32> to memref<64x64xf32, strided<[4096, 1], offset: ?>>
              loom.copy_from_tensor %17, %reinterpret_cast on @DRAM : tensor<64x64xf32>, memref<64x64xf32, strided<[4096, 1], offset: ?>>
            } {loom.iter_type = #loom.iter_type<temporal>}
          } {loom.iter_type = #loom.iter_type<temporal>}
        } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @y}
      } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @x}
      return
    }
    func.func @matmul__d0i0_d1i1__f01__v_d__BK256__BM32__BN32(%arg0: memref<4096x512xf32>, %arg1: memref<512x4096xf32>, %arg2: memref<4096x4096xf32>) {
      %cst = arith.constant 0.000000e+00 : f32
      %c32 = arith.constant 32 : index
      %c256 = arith.constant 256 : index
      affine.parallel (%arg3) = (0) to (8) {
        affine.parallel (%arg4) = (0) to (8) {
          affine.for %arg5 = 0 to 16 {
            affine.for %arg6 = 0 to 16 {
              %12 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 8)>(%arg3, %arg5)
              %13 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 8)>(%arg4, %arg6)
              %14 = loom.alloc [32, 32] on @L1 : memref<32x32xf32>
              %15 = loom.init_tensor %14[32, 32] : memref<32x32xf32> -> tensor<32x32xf32>
              %16 = linalg.fill ins(%cst : f32) outs(%15 : tensor<32x32xf32>) -> tensor<32x32xf32>
              %17 = affine.for %arg7 = 0 to 2 iter_args(%arg8 = %16) -> (tensor<32x32xf32>) {
                %21 = arith.muli %12, %c32 : index
                %22 = arith.muli %arg7, %c256 : index
                %23 = affine.apply affine_map<(d0, d1) -> (d0 * 512 + d1)>(%21, %22)
                %reinterpret_cast_0 = memref.reinterpret_cast %arg0 to offset: [%23], sizes: [32, 256], strides: [512, 1] : memref<4096x512xf32> to memref<32x256xf32, strided<[512, 1], offset: ?>>
                %24 = loom.alloc [32, 256] on @L1 : memref<32x256xf32>
                %25 = loom.copy_to_tensor %reinterpret_cast_0, %24 on @L1, interconnect : [@vertical_links], broadcast : [8, 1] : memref<32x256xf32, strided<[512, 1], offset: ?>>, memref<32x256xf32> -> tensor<32x256xf32>
                %26 = arith.muli %13, %c32 : index
                %27 = affine.apply affine_map<(d0, d1) -> (d0 * 4096 + d1)>(%22, %26)
                %reinterpret_cast_1 = memref.reinterpret_cast %arg1 to offset: [%27], sizes: [256, 32], strides: [4096, 1] : memref<512x4096xf32> to memref<256x32xf32, strided<[4096, 1], offset: ?>>
                %28 = loom.alloc [256, 32] on @L1 : memref<256x32xf32>
                %29 = loom.copy_to_tensor %reinterpret_cast_1, %28 on @L1, interconnect : [], broadcast : [1, 1] : memref<256x32xf32, strided<[4096, 1], offset: ?>>, memref<256x32xf32> -> tensor<256x32xf32>
                %30 = linalg.matmul ins(%25, %29 : tensor<32x256xf32>, tensor<256x32xf32>) outs(%arg8 : tensor<32x32xf32>) -> tensor<32x32xf32>
                affine.yield %30 : tensor<32x32xf32>
              }
              %18 = arith.muli %12, %c32 : index
              %19 = arith.muli %13, %c32 : index
              %20 = affine.apply affine_map<(d0, d1) -> (d0 * 4096 + d1)>(%18, %19)
              %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%20], sizes: [32, 32], strides: [4096, 1] : memref<4096x4096xf32> to memref<32x32xf32, strided<[4096, 1], offset: ?>>
              loom.copy_from_tensor %17, %reinterpret_cast on @DRAM : tensor<32x32xf32>, memref<32x32xf32, strided<[4096, 1], offset: ?>>
            } {loom.iter_type = #loom.iter_type<temporal>}
          } {loom.iter_type = #loom.iter_type<temporal>}
        } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @y}
      } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @x}
      return
    }
  }
  module attributes {loom.pass_name = "Materialize"} {
    func.func @matmul__d0i0_d1i1__f01__v_h__BK64__BM64__BN64(%arg0: memref<4096x512xf32>, %arg1: memref<512x4096xf32>, %arg2: memref<4096x4096xf32>) {
      %cst = arith.constant 0.000000e+00 : f32
      %c64 = arith.constant 64 : index
      affine.parallel (%arg3) = (0) to (8) {
        affine.parallel (%arg4) = (0) to (8) {
          affine.for %arg5 = 0 to 8 {
            affine.for %arg6 = 0 to 8 {
              %12 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 8)>(%arg3, %arg5)
              %13 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 8)>(%arg4, %arg6)
              %14 = loom.alloc [64, 64] on @L1 : memref<64x64xf32>
              %15 = loom.init_tensor %14[64, 64] : memref<64x64xf32> -> tensor<64x64xf32>
              %16 = linalg.fill ins(%cst : f32) outs(%15 : tensor<64x64xf32>) -> tensor<64x64xf32>
              %17 = affine.for %arg7 = 0 to 8 iter_args(%arg8 = %16) -> (tensor<64x64xf32>) {
                %21 = arith.muli %12, %c64 : index
                %22 = arith.muli %arg7, %c64 : index
                %23 = affine.apply affine_map<(d0, d1) -> (d0 * 512 + d1)>(%21, %22)
                %reinterpret_cast_0 = memref.reinterpret_cast %arg0 to offset: [%23], sizes: [64, 64], strides: [512, 1] : memref<4096x512xf32> to memref<64x64xf32, strided<[512, 1], offset: ?>>
                %24 = loom.alloc [64, 64] on @L1 : memref<64x64xf32>
                %25 = loom.copy_to_tensor %reinterpret_cast_0, %24 on @L1, interconnect : [@vertical_links], broadcast : [8, 1] : memref<64x64xf32, strided<[512, 1], offset: ?>>, memref<64x64xf32> -> tensor<64x64xf32>
                %26 = arith.muli %13, %c64 : index
                %27 = affine.apply affine_map<(d0, d1) -> (d0 * 4096 + d1)>(%22, %26)
                %reinterpret_cast_1 = memref.reinterpret_cast %arg1 to offset: [%27], sizes: [64, 64], strides: [4096, 1] : memref<512x4096xf32> to memref<64x64xf32, strided<[4096, 1], offset: ?>>
                %28 = loom.alloc [64, 64] on @L1 : memref<64x64xf32>
                %29 = loom.copy_to_tensor %reinterpret_cast_1, %28 on @L1, interconnect : [@horizontal_links], broadcast : [1, 8] : memref<64x64xf32, strided<[4096, 1], offset: ?>>, memref<64x64xf32> -> tensor<64x64xf32>
                %30 = linalg.matmul ins(%25, %29 : tensor<64x64xf32>, tensor<64x64xf32>) outs(%arg8 : tensor<64x64xf32>) -> tensor<64x64xf32>
                affine.yield %30 : tensor<64x64xf32>
              }
              %18 = arith.muli %12, %c64 : index
              %19 = arith.muli %13, %c64 : index
              %20 = affine.apply affine_map<(d0, d1) -> (d0 * 4096 + d1)>(%18, %19)
              %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%20], sizes: [64, 64], strides: [4096, 1] : memref<4096x4096xf32> to memref<64x64xf32, strided<[4096, 1], offset: ?>>
              loom.copy_from_tensor %17, %reinterpret_cast on @DRAM : tensor<64x64xf32>, memref<64x64xf32, strided<[4096, 1], offset: ?>>
            } {loom.iter_type = #loom.iter_type<temporal>}
          } {loom.iter_type = #loom.iter_type<temporal>}
        } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @y}
      } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @x}
      return
    }
    func.func @matmul__d0i0_d1i1__f01__v_h__BK256__BM32__BN32(%arg0: memref<4096x512xf32>, %arg1: memref<512x4096xf32>, %arg2: memref<4096x4096xf32>) {
      %cst = arith.constant 0.000000e+00 : f32
      %c32 = arith.constant 32 : index
      %c256 = arith.constant 256 : index
      affine.parallel (%arg3) = (0) to (8) {
        affine.parallel (%arg4) = (0) to (8) {
          affine.for %arg5 = 0 to 16 {
            affine.for %arg6 = 0 to 16 {
              %12 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 8)>(%arg3, %arg5)
              %13 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 8)>(%arg4, %arg6)
              %14 = loom.alloc [32, 32] on @L1 : memref<32x32xf32>
              %15 = loom.init_tensor %14[32, 32] : memref<32x32xf32> -> tensor<32x32xf32>
              %16 = linalg.fill ins(%cst : f32) outs(%15 : tensor<32x32xf32>) -> tensor<32x32xf32>
              %17 = affine.for %arg7 = 0 to 2 iter_args(%arg8 = %16) -> (tensor<32x32xf32>) {
                %21 = arith.muli %12, %c32 : index
                %22 = arith.muli %arg7, %c256 : index
                %23 = affine.apply affine_map<(d0, d1) -> (d0 * 512 + d1)>(%21, %22)
                %reinterpret_cast_0 = memref.reinterpret_cast %arg0 to offset: [%23], sizes: [32, 256], strides: [512, 1] : memref<4096x512xf32> to memref<32x256xf32, strided<[512, 1], offset: ?>>
                %24 = loom.alloc [32, 256] on @L1 : memref<32x256xf32>
                %25 = loom.copy_to_tensor %reinterpret_cast_0, %24 on @L1, interconnect : [@vertical_links], broadcast : [8, 1] : memref<32x256xf32, strided<[512, 1], offset: ?>>, memref<32x256xf32> -> tensor<32x256xf32>
                %26 = arith.muli %13, %c32 : index
                %27 = affine.apply affine_map<(d0, d1) -> (d0 * 4096 + d1)>(%22, %26)
                %reinterpret_cast_1 = memref.reinterpret_cast %arg1 to offset: [%27], sizes: [256, 32], strides: [4096, 1] : memref<512x4096xf32> to memref<256x32xf32, strided<[4096, 1], offset: ?>>
                %28 = loom.alloc [256, 32] on @L1 : memref<256x32xf32>
                %29 = loom.copy_to_tensor %reinterpret_cast_1, %28 on @L1, interconnect : [@horizontal_links], broadcast : [1, 8] : memref<256x32xf32, strided<[4096, 1], offset: ?>>, memref<256x32xf32> -> tensor<256x32xf32>
                %30 = linalg.matmul ins(%25, %29 : tensor<32x256xf32>, tensor<256x32xf32>) outs(%arg8 : tensor<32x32xf32>) -> tensor<32x32xf32>
                affine.yield %30 : tensor<32x32xf32>
              }
              %18 = arith.muli %12, %c32 : index
              %19 = arith.muli %13, %c32 : index
              %20 = affine.apply affine_map<(d0, d1) -> (d0 * 4096 + d1)>(%18, %19)
              %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%20], sizes: [32, 32], strides: [4096, 1] : memref<4096x4096xf32> to memref<32x32xf32, strided<[4096, 1], offset: ?>>
              loom.copy_from_tensor %17, %reinterpret_cast on @DRAM : tensor<32x32xf32>, memref<32x32xf32, strided<[4096, 1], offset: ?>>
            } {loom.iter_type = #loom.iter_type<temporal>}
          } {loom.iter_type = #loom.iter_type<temporal>}
        } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @y}
      } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @x}
      return
    }
  }
  module attributes {loom.pass_name = "Materialize"} {
    func.func @matmul__d0i0_d1i1__f10__d_d__BK64__BM64__BN64(%arg0: memref<4096x512xf32>, %arg1: memref<512x4096xf32>, %arg2: memref<4096x4096xf32>) {
      %cst = arith.constant 0.000000e+00 : f32
      %c64 = arith.constant 64 : index
      affine.parallel (%arg3) = (0) to (8) {
        affine.parallel (%arg4) = (0) to (8) {
          affine.for %arg5 = 0 to 8 {
            affine.for %arg6 = 0 to 8 {
              %12 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 8)>(%arg3, %arg6)
              %13 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 8)>(%arg4, %arg5)
              %14 = loom.alloc [64, 64] on @L1 : memref<64x64xf32>
              %15 = loom.init_tensor %14[64, 64] : memref<64x64xf32> -> tensor<64x64xf32>
              %16 = linalg.fill ins(%cst : f32) outs(%15 : tensor<64x64xf32>) -> tensor<64x64xf32>
              %17 = affine.for %arg7 = 0 to 8 iter_args(%arg8 = %16) -> (tensor<64x64xf32>) {
                %21 = arith.muli %12, %c64 : index
                %22 = arith.muli %arg7, %c64 : index
                %23 = affine.apply affine_map<(d0, d1) -> (d0 * 512 + d1)>(%21, %22)
                %reinterpret_cast_0 = memref.reinterpret_cast %arg0 to offset: [%23], sizes: [64, 64], strides: [512, 1] : memref<4096x512xf32> to memref<64x64xf32, strided<[512, 1], offset: ?>>
                %24 = loom.alloc [64, 64] on @L1 : memref<64x64xf32>
                %25 = loom.copy_to_tensor %reinterpret_cast_0, %24 on @L1, interconnect : [], broadcast : [1, 1] : memref<64x64xf32, strided<[512, 1], offset: ?>>, memref<64x64xf32> -> tensor<64x64xf32>
                %26 = arith.muli %13, %c64 : index
                %27 = affine.apply affine_map<(d0, d1) -> (d0 * 4096 + d1)>(%22, %26)
                %reinterpret_cast_1 = memref.reinterpret_cast %arg1 to offset: [%27], sizes: [64, 64], strides: [4096, 1] : memref<512x4096xf32> to memref<64x64xf32, strided<[4096, 1], offset: ?>>
                %28 = loom.alloc [64, 64] on @L1 : memref<64x64xf32>
                %29 = loom.copy_to_tensor %reinterpret_cast_1, %28 on @L1, interconnect : [], broadcast : [1, 1] : memref<64x64xf32, strided<[4096, 1], offset: ?>>, memref<64x64xf32> -> tensor<64x64xf32>
                %30 = linalg.matmul ins(%25, %29 : tensor<64x64xf32>, tensor<64x64xf32>) outs(%arg8 : tensor<64x64xf32>) -> tensor<64x64xf32>
                affine.yield %30 : tensor<64x64xf32>
              }
              %18 = arith.muli %12, %c64 : index
              %19 = arith.muli %13, %c64 : index
              %20 = affine.apply affine_map<(d0, d1) -> (d0 * 4096 + d1)>(%18, %19)
              %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%20], sizes: [64, 64], strides: [4096, 1] : memref<4096x4096xf32> to memref<64x64xf32, strided<[4096, 1], offset: ?>>
              loom.copy_from_tensor %17, %reinterpret_cast on @DRAM : tensor<64x64xf32>, memref<64x64xf32, strided<[4096, 1], offset: ?>>
            } {loom.iter_type = #loom.iter_type<temporal>}
          } {loom.iter_type = #loom.iter_type<temporal>}
        } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @y}
      } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @x}
      return
    }
    func.func @matmul__d0i0_d1i1__f10__d_d__BK256__BM32__BN32(%arg0: memref<4096x512xf32>, %arg1: memref<512x4096xf32>, %arg2: memref<4096x4096xf32>) {
      %cst = arith.constant 0.000000e+00 : f32
      %c32 = arith.constant 32 : index
      %c256 = arith.constant 256 : index
      affine.parallel (%arg3) = (0) to (8) {
        affine.parallel (%arg4) = (0) to (8) {
          affine.for %arg5 = 0 to 16 {
            affine.for %arg6 = 0 to 16 {
              %12 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 8)>(%arg3, %arg6)
              %13 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 8)>(%arg4, %arg5)
              %14 = loom.alloc [32, 32] on @L1 : memref<32x32xf32>
              %15 = loom.init_tensor %14[32, 32] : memref<32x32xf32> -> tensor<32x32xf32>
              %16 = linalg.fill ins(%cst : f32) outs(%15 : tensor<32x32xf32>) -> tensor<32x32xf32>
              %17 = affine.for %arg7 = 0 to 2 iter_args(%arg8 = %16) -> (tensor<32x32xf32>) {
                %21 = arith.muli %12, %c32 : index
                %22 = arith.muli %arg7, %c256 : index
                %23 = affine.apply affine_map<(d0, d1) -> (d0 * 512 + d1)>(%21, %22)
                %reinterpret_cast_0 = memref.reinterpret_cast %arg0 to offset: [%23], sizes: [32, 256], strides: [512, 1] : memref<4096x512xf32> to memref<32x256xf32, strided<[512, 1], offset: ?>>
                %24 = loom.alloc [32, 256] on @L1 : memref<32x256xf32>
                %25 = loom.copy_to_tensor %reinterpret_cast_0, %24 on @L1, interconnect : [], broadcast : [1, 1] : memref<32x256xf32, strided<[512, 1], offset: ?>>, memref<32x256xf32> -> tensor<32x256xf32>
                %26 = arith.muli %13, %c32 : index
                %27 = affine.apply affine_map<(d0, d1) -> (d0 * 4096 + d1)>(%22, %26)
                %reinterpret_cast_1 = memref.reinterpret_cast %arg1 to offset: [%27], sizes: [256, 32], strides: [4096, 1] : memref<512x4096xf32> to memref<256x32xf32, strided<[4096, 1], offset: ?>>
                %28 = loom.alloc [256, 32] on @L1 : memref<256x32xf32>
                %29 = loom.copy_to_tensor %reinterpret_cast_1, %28 on @L1, interconnect : [], broadcast : [1, 1] : memref<256x32xf32, strided<[4096, 1], offset: ?>>, memref<256x32xf32> -> tensor<256x32xf32>
                %30 = linalg.matmul ins(%25, %29 : tensor<32x256xf32>, tensor<256x32xf32>) outs(%arg8 : tensor<32x32xf32>) -> tensor<32x32xf32>
                affine.yield %30 : tensor<32x32xf32>
              }
              %18 = arith.muli %12, %c32 : index
              %19 = arith.muli %13, %c32 : index
              %20 = affine.apply affine_map<(d0, d1) -> (d0 * 4096 + d1)>(%18, %19)
              %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%20], sizes: [32, 32], strides: [4096, 1] : memref<4096x4096xf32> to memref<32x32xf32, strided<[4096, 1], offset: ?>>
              loom.copy_from_tensor %17, %reinterpret_cast on @DRAM : tensor<32x32xf32>, memref<32x32xf32, strided<[4096, 1], offset: ?>>
            } {loom.iter_type = #loom.iter_type<temporal>}
          } {loom.iter_type = #loom.iter_type<temporal>}
        } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @y}
      } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @x}
      return
    }
  }
  module attributes {loom.pass_name = "Materialize"} {
    func.func @matmul__d0i0_d1i1__f10__d_h__BK64__BM64__BN64(%arg0: memref<4096x512xf32>, %arg1: memref<512x4096xf32>, %arg2: memref<4096x4096xf32>) {
      %cst = arith.constant 0.000000e+00 : f32
      %c64 = arith.constant 64 : index
      affine.parallel (%arg3) = (0) to (8) {
        affine.parallel (%arg4) = (0) to (8) {
          affine.for %arg5 = 0 to 8 {
            affine.for %arg6 = 0 to 8 {
              %12 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 8)>(%arg3, %arg6)
              %13 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 8)>(%arg4, %arg5)
              %14 = loom.alloc [64, 64] on @L1 : memref<64x64xf32>
              %15 = loom.init_tensor %14[64, 64] : memref<64x64xf32> -> tensor<64x64xf32>
              %16 = linalg.fill ins(%cst : f32) outs(%15 : tensor<64x64xf32>) -> tensor<64x64xf32>
              %17 = affine.for %arg7 = 0 to 8 iter_args(%arg8 = %16) -> (tensor<64x64xf32>) {
                %21 = arith.muli %12, %c64 : index
                %22 = arith.muli %arg7, %c64 : index
                %23 = affine.apply affine_map<(d0, d1) -> (d0 * 512 + d1)>(%21, %22)
                %reinterpret_cast_0 = memref.reinterpret_cast %arg0 to offset: [%23], sizes: [64, 64], strides: [512, 1] : memref<4096x512xf32> to memref<64x64xf32, strided<[512, 1], offset: ?>>
                %24 = loom.alloc [64, 64] on @L1 : memref<64x64xf32>
                %25 = loom.copy_to_tensor %reinterpret_cast_0, %24 on @L1, interconnect : [], broadcast : [1, 1] : memref<64x64xf32, strided<[512, 1], offset: ?>>, memref<64x64xf32> -> tensor<64x64xf32>
                %26 = arith.muli %13, %c64 : index
                %27 = affine.apply affine_map<(d0, d1) -> (d0 * 4096 + d1)>(%22, %26)
                %reinterpret_cast_1 = memref.reinterpret_cast %arg1 to offset: [%27], sizes: [64, 64], strides: [4096, 1] : memref<512x4096xf32> to memref<64x64xf32, strided<[4096, 1], offset: ?>>
                %28 = loom.alloc [64, 64] on @L1 : memref<64x64xf32>
                %29 = loom.copy_to_tensor %reinterpret_cast_1, %28 on @L1, interconnect : [@horizontal_links], broadcast : [1, 8] : memref<64x64xf32, strided<[4096, 1], offset: ?>>, memref<64x64xf32> -> tensor<64x64xf32>
                %30 = linalg.matmul ins(%25, %29 : tensor<64x64xf32>, tensor<64x64xf32>) outs(%arg8 : tensor<64x64xf32>) -> tensor<64x64xf32>
                affine.yield %30 : tensor<64x64xf32>
              }
              %18 = arith.muli %12, %c64 : index
              %19 = arith.muli %13, %c64 : index
              %20 = affine.apply affine_map<(d0, d1) -> (d0 * 4096 + d1)>(%18, %19)
              %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%20], sizes: [64, 64], strides: [4096, 1] : memref<4096x4096xf32> to memref<64x64xf32, strided<[4096, 1], offset: ?>>
              loom.copy_from_tensor %17, %reinterpret_cast on @DRAM : tensor<64x64xf32>, memref<64x64xf32, strided<[4096, 1], offset: ?>>
            } {loom.iter_type = #loom.iter_type<temporal>}
          } {loom.iter_type = #loom.iter_type<temporal>}
        } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @y}
      } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @x}
      return
    }
    func.func @matmul__d0i0_d1i1__f10__d_h__BK256__BM32__BN32(%arg0: memref<4096x512xf32>, %arg1: memref<512x4096xf32>, %arg2: memref<4096x4096xf32>) {
      %cst = arith.constant 0.000000e+00 : f32
      %c32 = arith.constant 32 : index
      %c256 = arith.constant 256 : index
      affine.parallel (%arg3) = (0) to (8) {
        affine.parallel (%arg4) = (0) to (8) {
          affine.for %arg5 = 0 to 16 {
            affine.for %arg6 = 0 to 16 {
              %12 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 8)>(%arg3, %arg6)
              %13 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 8)>(%arg4, %arg5)
              %14 = loom.alloc [32, 32] on @L1 : memref<32x32xf32>
              %15 = loom.init_tensor %14[32, 32] : memref<32x32xf32> -> tensor<32x32xf32>
              %16 = linalg.fill ins(%cst : f32) outs(%15 : tensor<32x32xf32>) -> tensor<32x32xf32>
              %17 = affine.for %arg7 = 0 to 2 iter_args(%arg8 = %16) -> (tensor<32x32xf32>) {
                %21 = arith.muli %12, %c32 : index
                %22 = arith.muli %arg7, %c256 : index
                %23 = affine.apply affine_map<(d0, d1) -> (d0 * 512 + d1)>(%21, %22)
                %reinterpret_cast_0 = memref.reinterpret_cast %arg0 to offset: [%23], sizes: [32, 256], strides: [512, 1] : memref<4096x512xf32> to memref<32x256xf32, strided<[512, 1], offset: ?>>
                %24 = loom.alloc [32, 256] on @L1 : memref<32x256xf32>
                %25 = loom.copy_to_tensor %reinterpret_cast_0, %24 on @L1, interconnect : [], broadcast : [1, 1] : memref<32x256xf32, strided<[512, 1], offset: ?>>, memref<32x256xf32> -> tensor<32x256xf32>
                %26 = arith.muli %13, %c32 : index
                %27 = affine.apply affine_map<(d0, d1) -> (d0 * 4096 + d1)>(%22, %26)
                %reinterpret_cast_1 = memref.reinterpret_cast %arg1 to offset: [%27], sizes: [256, 32], strides: [4096, 1] : memref<512x4096xf32> to memref<256x32xf32, strided<[4096, 1], offset: ?>>
                %28 = loom.alloc [256, 32] on @L1 : memref<256x32xf32>
                %29 = loom.copy_to_tensor %reinterpret_cast_1, %28 on @L1, interconnect : [@horizontal_links], broadcast : [1, 8] : memref<256x32xf32, strided<[4096, 1], offset: ?>>, memref<256x32xf32> -> tensor<256x32xf32>
                %30 = linalg.matmul ins(%25, %29 : tensor<32x256xf32>, tensor<256x32xf32>) outs(%arg8 : tensor<32x32xf32>) -> tensor<32x32xf32>
                affine.yield %30 : tensor<32x32xf32>
              }
              %18 = arith.muli %12, %c32 : index
              %19 = arith.muli %13, %c32 : index
              %20 = affine.apply affine_map<(d0, d1) -> (d0 * 4096 + d1)>(%18, %19)
              %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%20], sizes: [32, 32], strides: [4096, 1] : memref<4096x4096xf32> to memref<32x32xf32, strided<[4096, 1], offset: ?>>
              loom.copy_from_tensor %17, %reinterpret_cast on @DRAM : tensor<32x32xf32>, memref<32x32xf32, strided<[4096, 1], offset: ?>>
            } {loom.iter_type = #loom.iter_type<temporal>}
          } {loom.iter_type = #loom.iter_type<temporal>}
        } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @y}
      } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @x}
      return
    }
  }
  module attributes {loom.pass_name = "Materialize"} {
    func.func @matmul__d0i0_d1i1__f10__v_d__BK64__BM64__BN64(%arg0: memref<4096x512xf32>, %arg1: memref<512x4096xf32>, %arg2: memref<4096x4096xf32>) {
      %cst = arith.constant 0.000000e+00 : f32
      %c64 = arith.constant 64 : index
      affine.parallel (%arg3) = (0) to (8) {
        affine.parallel (%arg4) = (0) to (8) {
          affine.for %arg5 = 0 to 8 {
            affine.for %arg6 = 0 to 8 {
              %12 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 8)>(%arg3, %arg6)
              %13 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 8)>(%arg4, %arg5)
              %14 = loom.alloc [64, 64] on @L1 : memref<64x64xf32>
              %15 = loom.init_tensor %14[64, 64] : memref<64x64xf32> -> tensor<64x64xf32>
              %16 = linalg.fill ins(%cst : f32) outs(%15 : tensor<64x64xf32>) -> tensor<64x64xf32>
              %17 = affine.for %arg7 = 0 to 8 iter_args(%arg8 = %16) -> (tensor<64x64xf32>) {
                %21 = arith.muli %12, %c64 : index
                %22 = arith.muli %arg7, %c64 : index
                %23 = affine.apply affine_map<(d0, d1) -> (d0 * 512 + d1)>(%21, %22)
                %reinterpret_cast_0 = memref.reinterpret_cast %arg0 to offset: [%23], sizes: [64, 64], strides: [512, 1] : memref<4096x512xf32> to memref<64x64xf32, strided<[512, 1], offset: ?>>
                %24 = loom.alloc [64, 64] on @L1 : memref<64x64xf32>
                %25 = loom.copy_to_tensor %reinterpret_cast_0, %24 on @L1, interconnect : [@vertical_links], broadcast : [8, 1] : memref<64x64xf32, strided<[512, 1], offset: ?>>, memref<64x64xf32> -> tensor<64x64xf32>
                %26 = arith.muli %13, %c64 : index
                %27 = affine.apply affine_map<(d0, d1) -> (d0 * 4096 + d1)>(%22, %26)
                %reinterpret_cast_1 = memref.reinterpret_cast %arg1 to offset: [%27], sizes: [64, 64], strides: [4096, 1] : memref<512x4096xf32> to memref<64x64xf32, strided<[4096, 1], offset: ?>>
                %28 = loom.alloc [64, 64] on @L1 : memref<64x64xf32>
                %29 = loom.copy_to_tensor %reinterpret_cast_1, %28 on @L1, interconnect : [], broadcast : [1, 1] : memref<64x64xf32, strided<[4096, 1], offset: ?>>, memref<64x64xf32> -> tensor<64x64xf32>
                %30 = linalg.matmul ins(%25, %29 : tensor<64x64xf32>, tensor<64x64xf32>) outs(%arg8 : tensor<64x64xf32>) -> tensor<64x64xf32>
                affine.yield %30 : tensor<64x64xf32>
              }
              %18 = arith.muli %12, %c64 : index
              %19 = arith.muli %13, %c64 : index
              %20 = affine.apply affine_map<(d0, d1) -> (d0 * 4096 + d1)>(%18, %19)
              %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%20], sizes: [64, 64], strides: [4096, 1] : memref<4096x4096xf32> to memref<64x64xf32, strided<[4096, 1], offset: ?>>
              loom.copy_from_tensor %17, %reinterpret_cast on @DRAM : tensor<64x64xf32>, memref<64x64xf32, strided<[4096, 1], offset: ?>>
            } {loom.iter_type = #loom.iter_type<temporal>}
          } {loom.iter_type = #loom.iter_type<temporal>}
        } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @y}
      } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @x}
      return
    }
    func.func @matmul__d0i0_d1i1__f10__v_d__BK256__BM32__BN32(%arg0: memref<4096x512xf32>, %arg1: memref<512x4096xf32>, %arg2: memref<4096x4096xf32>) {
      %cst = arith.constant 0.000000e+00 : f32
      %c32 = arith.constant 32 : index
      %c256 = arith.constant 256 : index
      affine.parallel (%arg3) = (0) to (8) {
        affine.parallel (%arg4) = (0) to (8) {
          affine.for %arg5 = 0 to 16 {
            affine.for %arg6 = 0 to 16 {
              %12 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 8)>(%arg3, %arg6)
              %13 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 8)>(%arg4, %arg5)
              %14 = loom.alloc [32, 32] on @L1 : memref<32x32xf32>
              %15 = loom.init_tensor %14[32, 32] : memref<32x32xf32> -> tensor<32x32xf32>
              %16 = linalg.fill ins(%cst : f32) outs(%15 : tensor<32x32xf32>) -> tensor<32x32xf32>
              %17 = affine.for %arg7 = 0 to 2 iter_args(%arg8 = %16) -> (tensor<32x32xf32>) {
                %21 = arith.muli %12, %c32 : index
                %22 = arith.muli %arg7, %c256 : index
                %23 = affine.apply affine_map<(d0, d1) -> (d0 * 512 + d1)>(%21, %22)
                %reinterpret_cast_0 = memref.reinterpret_cast %arg0 to offset: [%23], sizes: [32, 256], strides: [512, 1] : memref<4096x512xf32> to memref<32x256xf32, strided<[512, 1], offset: ?>>
                %24 = loom.alloc [32, 256] on @L1 : memref<32x256xf32>
                %25 = loom.copy_to_tensor %reinterpret_cast_0, %24 on @L1, interconnect : [@vertical_links], broadcast : [8, 1] : memref<32x256xf32, strided<[512, 1], offset: ?>>, memref<32x256xf32> -> tensor<32x256xf32>
                %26 = arith.muli %13, %c32 : index
                %27 = affine.apply affine_map<(d0, d1) -> (d0 * 4096 + d1)>(%22, %26)
                %reinterpret_cast_1 = memref.reinterpret_cast %arg1 to offset: [%27], sizes: [256, 32], strides: [4096, 1] : memref<512x4096xf32> to memref<256x32xf32, strided<[4096, 1], offset: ?>>
                %28 = loom.alloc [256, 32] on @L1 : memref<256x32xf32>
                %29 = loom.copy_to_tensor %reinterpret_cast_1, %28 on @L1, interconnect : [], broadcast : [1, 1] : memref<256x32xf32, strided<[4096, 1], offset: ?>>, memref<256x32xf32> -> tensor<256x32xf32>
                %30 = linalg.matmul ins(%25, %29 : tensor<32x256xf32>, tensor<256x32xf32>) outs(%arg8 : tensor<32x32xf32>) -> tensor<32x32xf32>
                affine.yield %30 : tensor<32x32xf32>
              }
              %18 = arith.muli %12, %c32 : index
              %19 = arith.muli %13, %c32 : index
              %20 = affine.apply affine_map<(d0, d1) -> (d0 * 4096 + d1)>(%18, %19)
              %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%20], sizes: [32, 32], strides: [4096, 1] : memref<4096x4096xf32> to memref<32x32xf32, strided<[4096, 1], offset: ?>>
              loom.copy_from_tensor %17, %reinterpret_cast on @DRAM : tensor<32x32xf32>, memref<32x32xf32, strided<[4096, 1], offset: ?>>
            } {loom.iter_type = #loom.iter_type<temporal>}
          } {loom.iter_type = #loom.iter_type<temporal>}
        } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @y}
      } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @x}
      return
    }
  }
  module attributes {loom.pass_name = "Materialize"} {
    func.func @matmul__d0i0_d1i1__f10__v_h__BK64__BM64__BN64(%arg0: memref<4096x512xf32>, %arg1: memref<512x4096xf32>, %arg2: memref<4096x4096xf32>) {
      %cst = arith.constant 0.000000e+00 : f32
      %c64 = arith.constant 64 : index
      affine.parallel (%arg3) = (0) to (8) {
        affine.parallel (%arg4) = (0) to (8) {
          affine.for %arg5 = 0 to 8 {
            affine.for %arg6 = 0 to 8 {
              %12 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 8)>(%arg3, %arg6)
              %13 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 8)>(%arg4, %arg5)
              %14 = loom.alloc [64, 64] on @L1 : memref<64x64xf32>
              %15 = loom.init_tensor %14[64, 64] : memref<64x64xf32> -> tensor<64x64xf32>
              %16 = linalg.fill ins(%cst : f32) outs(%15 : tensor<64x64xf32>) -> tensor<64x64xf32>
              %17 = affine.for %arg7 = 0 to 8 iter_args(%arg8 = %16) -> (tensor<64x64xf32>) {
                %21 = arith.muli %12, %c64 : index
                %22 = arith.muli %arg7, %c64 : index
                %23 = affine.apply affine_map<(d0, d1) -> (d0 * 512 + d1)>(%21, %22)
                %reinterpret_cast_0 = memref.reinterpret_cast %arg0 to offset: [%23], sizes: [64, 64], strides: [512, 1] : memref<4096x512xf32> to memref<64x64xf32, strided<[512, 1], offset: ?>>
                %24 = loom.alloc [64, 64] on @L1 : memref<64x64xf32>
                %25 = loom.copy_to_tensor %reinterpret_cast_0, %24 on @L1, interconnect : [@vertical_links], broadcast : [8, 1] : memref<64x64xf32, strided<[512, 1], offset: ?>>, memref<64x64xf32> -> tensor<64x64xf32>
                %26 = arith.muli %13, %c64 : index
                %27 = affine.apply affine_map<(d0, d1) -> (d0 * 4096 + d1)>(%22, %26)
                %reinterpret_cast_1 = memref.reinterpret_cast %arg1 to offset: [%27], sizes: [64, 64], strides: [4096, 1] : memref<512x4096xf32> to memref<64x64xf32, strided<[4096, 1], offset: ?>>
                %28 = loom.alloc [64, 64] on @L1 : memref<64x64xf32>
                %29 = loom.copy_to_tensor %reinterpret_cast_1, %28 on @L1, interconnect : [@horizontal_links], broadcast : [1, 8] : memref<64x64xf32, strided<[4096, 1], offset: ?>>, memref<64x64xf32> -> tensor<64x64xf32>
                %30 = linalg.matmul ins(%25, %29 : tensor<64x64xf32>, tensor<64x64xf32>) outs(%arg8 : tensor<64x64xf32>) -> tensor<64x64xf32>
                affine.yield %30 : tensor<64x64xf32>
              }
              %18 = arith.muli %12, %c64 : index
              %19 = arith.muli %13, %c64 : index
              %20 = affine.apply affine_map<(d0, d1) -> (d0 * 4096 + d1)>(%18, %19)
              %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%20], sizes: [64, 64], strides: [4096, 1] : memref<4096x4096xf32> to memref<64x64xf32, strided<[4096, 1], offset: ?>>
              loom.copy_from_tensor %17, %reinterpret_cast on @DRAM : tensor<64x64xf32>, memref<64x64xf32, strided<[4096, 1], offset: ?>>
            } {loom.iter_type = #loom.iter_type<temporal>}
          } {loom.iter_type = #loom.iter_type<temporal>}
        } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @y}
      } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @x}
      return
    }
    func.func @matmul__d0i0_d1i1__f10__v_h__BK256__BM32__BN32(%arg0: memref<4096x512xf32>, %arg1: memref<512x4096xf32>, %arg2: memref<4096x4096xf32>) {
      %cst = arith.constant 0.000000e+00 : f32
      %c32 = arith.constant 32 : index
      %c256 = arith.constant 256 : index
      affine.parallel (%arg3) = (0) to (8) {
        affine.parallel (%arg4) = (0) to (8) {
          affine.for %arg5 = 0 to 16 {
            affine.for %arg6 = 0 to 16 {
              %12 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 8)>(%arg3, %arg6)
              %13 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 8)>(%arg4, %arg5)
              %14 = loom.alloc [32, 32] on @L1 : memref<32x32xf32>
              %15 = loom.init_tensor %14[32, 32] : memref<32x32xf32> -> tensor<32x32xf32>
              %16 = linalg.fill ins(%cst : f32) outs(%15 : tensor<32x32xf32>) -> tensor<32x32xf32>
              %17 = affine.for %arg7 = 0 to 2 iter_args(%arg8 = %16) -> (tensor<32x32xf32>) {
                %21 = arith.muli %12, %c32 : index
                %22 = arith.muli %arg7, %c256 : index
                %23 = affine.apply affine_map<(d0, d1) -> (d0 * 512 + d1)>(%21, %22)
                %reinterpret_cast_0 = memref.reinterpret_cast %arg0 to offset: [%23], sizes: [32, 256], strides: [512, 1] : memref<4096x512xf32> to memref<32x256xf32, strided<[512, 1], offset: ?>>
                %24 = loom.alloc [32, 256] on @L1 : memref<32x256xf32>
                %25 = loom.copy_to_tensor %reinterpret_cast_0, %24 on @L1, interconnect : [@vertical_links], broadcast : [8, 1] : memref<32x256xf32, strided<[512, 1], offset: ?>>, memref<32x256xf32> -> tensor<32x256xf32>
                %26 = arith.muli %13, %c32 : index
                %27 = affine.apply affine_map<(d0, d1) -> (d0 * 4096 + d1)>(%22, %26)
                %reinterpret_cast_1 = memref.reinterpret_cast %arg1 to offset: [%27], sizes: [256, 32], strides: [4096, 1] : memref<512x4096xf32> to memref<256x32xf32, strided<[4096, 1], offset: ?>>
                %28 = loom.alloc [256, 32] on @L1 : memref<256x32xf32>
                %29 = loom.copy_to_tensor %reinterpret_cast_1, %28 on @L1, interconnect : [@horizontal_links], broadcast : [1, 8] : memref<256x32xf32, strided<[4096, 1], offset: ?>>, memref<256x32xf32> -> tensor<256x32xf32>
                %30 = linalg.matmul ins(%25, %29 : tensor<32x256xf32>, tensor<256x32xf32>) outs(%arg8 : tensor<32x32xf32>) -> tensor<32x32xf32>
                affine.yield %30 : tensor<32x32xf32>
              }
              %18 = arith.muli %12, %c32 : index
              %19 = arith.muli %13, %c32 : index
              %20 = affine.apply affine_map<(d0, d1) -> (d0 * 4096 + d1)>(%18, %19)
              %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%20], sizes: [32, 32], strides: [4096, 1] : memref<4096x4096xf32> to memref<32x32xf32, strided<[4096, 1], offset: ?>>
              loom.copy_from_tensor %17, %reinterpret_cast on @DRAM : tensor<32x32xf32>, memref<32x32xf32, strided<[4096, 1], offset: ?>>
            } {loom.iter_type = #loom.iter_type<temporal>}
          } {loom.iter_type = #loom.iter_type<temporal>}
        } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @y}
      } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @x}
      return
    }
  }
  module attributes {loom.pass_name = "Materialize"} {
    func.func @matmul__d1i0_d0i1__f01__d_d__BK64__BM64__BN64(%arg0: memref<4096x512xf32>, %arg1: memref<512x4096xf32>, %arg2: memref<4096x4096xf32>) {
      %cst = arith.constant 0.000000e+00 : f32
      %c64 = arith.constant 64 : index
      affine.parallel (%arg3) = (0) to (8) {
        affine.parallel (%arg4) = (0) to (8) {
          affine.for %arg5 = 0 to 8 {
            affine.for %arg6 = 0 to 8 {
              %12 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 8)>(%arg3, %arg5)
              %13 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 8)>(%arg4, %arg6)
              %14 = loom.alloc [64, 64] on @L1 : memref<64x64xf32>
              %15 = loom.init_tensor %14[64, 64] : memref<64x64xf32> -> tensor<64x64xf32>
              %16 = linalg.fill ins(%cst : f32) outs(%15 : tensor<64x64xf32>) -> tensor<64x64xf32>
              %17 = affine.for %arg7 = 0 to 8 iter_args(%arg8 = %16) -> (tensor<64x64xf32>) {
                %21 = arith.muli %12, %c64 : index
                %22 = arith.muli %arg7, %c64 : index
                %23 = affine.apply affine_map<(d0, d1) -> (d0 * 512 + d1)>(%21, %22)
                %reinterpret_cast_0 = memref.reinterpret_cast %arg0 to offset: [%23], sizes: [64, 64], strides: [512, 1] : memref<4096x512xf32> to memref<64x64xf32, strided<[512, 1], offset: ?>>
                %24 = loom.alloc [64, 64] on @L1 : memref<64x64xf32>
                %25 = loom.copy_to_tensor %reinterpret_cast_0, %24 on @L1, interconnect : [], broadcast : [1, 1] : memref<64x64xf32, strided<[512, 1], offset: ?>>, memref<64x64xf32> -> tensor<64x64xf32>
                %26 = arith.muli %13, %c64 : index
                %27 = affine.apply affine_map<(d0, d1) -> (d0 * 4096 + d1)>(%22, %26)
                %reinterpret_cast_1 = memref.reinterpret_cast %arg1 to offset: [%27], sizes: [64, 64], strides: [4096, 1] : memref<512x4096xf32> to memref<64x64xf32, strided<[4096, 1], offset: ?>>
                %28 = loom.alloc [64, 64] on @L1 : memref<64x64xf32>
                %29 = loom.copy_to_tensor %reinterpret_cast_1, %28 on @L1, interconnect : [], broadcast : [1, 1] : memref<64x64xf32, strided<[4096, 1], offset: ?>>, memref<64x64xf32> -> tensor<64x64xf32>
                %30 = linalg.matmul ins(%25, %29 : tensor<64x64xf32>, tensor<64x64xf32>) outs(%arg8 : tensor<64x64xf32>) -> tensor<64x64xf32>
                affine.yield %30 : tensor<64x64xf32>
              }
              %18 = arith.muli %12, %c64 : index
              %19 = arith.muli %13, %c64 : index
              %20 = affine.apply affine_map<(d0, d1) -> (d0 * 4096 + d1)>(%18, %19)
              %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%20], sizes: [64, 64], strides: [4096, 1] : memref<4096x4096xf32> to memref<64x64xf32, strided<[4096, 1], offset: ?>>
              loom.copy_from_tensor %17, %reinterpret_cast on @DRAM : tensor<64x64xf32>, memref<64x64xf32, strided<[4096, 1], offset: ?>>
            } {loom.iter_type = #loom.iter_type<temporal>}
          } {loom.iter_type = #loom.iter_type<temporal>}
        } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @x}
      } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @y}
      return
    }
    func.func @matmul__d1i0_d0i1__f01__d_d__BK256__BM32__BN32(%arg0: memref<4096x512xf32>, %arg1: memref<512x4096xf32>, %arg2: memref<4096x4096xf32>) {
      %cst = arith.constant 0.000000e+00 : f32
      %c32 = arith.constant 32 : index
      %c256 = arith.constant 256 : index
      affine.parallel (%arg3) = (0) to (8) {
        affine.parallel (%arg4) = (0) to (8) {
          affine.for %arg5 = 0 to 16 {
            affine.for %arg6 = 0 to 16 {
              %12 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 8)>(%arg3, %arg5)
              %13 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 8)>(%arg4, %arg6)
              %14 = loom.alloc [32, 32] on @L1 : memref<32x32xf32>
              %15 = loom.init_tensor %14[32, 32] : memref<32x32xf32> -> tensor<32x32xf32>
              %16 = linalg.fill ins(%cst : f32) outs(%15 : tensor<32x32xf32>) -> tensor<32x32xf32>
              %17 = affine.for %arg7 = 0 to 2 iter_args(%arg8 = %16) -> (tensor<32x32xf32>) {
                %21 = arith.muli %12, %c32 : index
                %22 = arith.muli %arg7, %c256 : index
                %23 = affine.apply affine_map<(d0, d1) -> (d0 * 512 + d1)>(%21, %22)
                %reinterpret_cast_0 = memref.reinterpret_cast %arg0 to offset: [%23], sizes: [32, 256], strides: [512, 1] : memref<4096x512xf32> to memref<32x256xf32, strided<[512, 1], offset: ?>>
                %24 = loom.alloc [32, 256] on @L1 : memref<32x256xf32>
                %25 = loom.copy_to_tensor %reinterpret_cast_0, %24 on @L1, interconnect : [], broadcast : [1, 1] : memref<32x256xf32, strided<[512, 1], offset: ?>>, memref<32x256xf32> -> tensor<32x256xf32>
                %26 = arith.muli %13, %c32 : index
                %27 = affine.apply affine_map<(d0, d1) -> (d0 * 4096 + d1)>(%22, %26)
                %reinterpret_cast_1 = memref.reinterpret_cast %arg1 to offset: [%27], sizes: [256, 32], strides: [4096, 1] : memref<512x4096xf32> to memref<256x32xf32, strided<[4096, 1], offset: ?>>
                %28 = loom.alloc [256, 32] on @L1 : memref<256x32xf32>
                %29 = loom.copy_to_tensor %reinterpret_cast_1, %28 on @L1, interconnect : [], broadcast : [1, 1] : memref<256x32xf32, strided<[4096, 1], offset: ?>>, memref<256x32xf32> -> tensor<256x32xf32>
                %30 = linalg.matmul ins(%25, %29 : tensor<32x256xf32>, tensor<256x32xf32>) outs(%arg8 : tensor<32x32xf32>) -> tensor<32x32xf32>
                affine.yield %30 : tensor<32x32xf32>
              }
              %18 = arith.muli %12, %c32 : index
              %19 = arith.muli %13, %c32 : index
              %20 = affine.apply affine_map<(d0, d1) -> (d0 * 4096 + d1)>(%18, %19)
              %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%20], sizes: [32, 32], strides: [4096, 1] : memref<4096x4096xf32> to memref<32x32xf32, strided<[4096, 1], offset: ?>>
              loom.copy_from_tensor %17, %reinterpret_cast on @DRAM : tensor<32x32xf32>, memref<32x32xf32, strided<[4096, 1], offset: ?>>
            } {loom.iter_type = #loom.iter_type<temporal>}
          } {loom.iter_type = #loom.iter_type<temporal>}
        } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @x}
      } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @y}
      return
    }
  }
  module attributes {loom.pass_name = "Materialize"} {
    func.func @matmul__d1i0_d0i1__f01__d_v__BK64__BM64__BN64(%arg0: memref<4096x512xf32>, %arg1: memref<512x4096xf32>, %arg2: memref<4096x4096xf32>) {
      %cst = arith.constant 0.000000e+00 : f32
      %c64 = arith.constant 64 : index
      affine.parallel (%arg3) = (0) to (8) {
        affine.parallel (%arg4) = (0) to (8) {
          affine.for %arg5 = 0 to 8 {
            affine.for %arg6 = 0 to 8 {
              %12 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 8)>(%arg3, %arg5)
              %13 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 8)>(%arg4, %arg6)
              %14 = loom.alloc [64, 64] on @L1 : memref<64x64xf32>
              %15 = loom.init_tensor %14[64, 64] : memref<64x64xf32> -> tensor<64x64xf32>
              %16 = linalg.fill ins(%cst : f32) outs(%15 : tensor<64x64xf32>) -> tensor<64x64xf32>
              %17 = affine.for %arg7 = 0 to 8 iter_args(%arg8 = %16) -> (tensor<64x64xf32>) {
                %21 = arith.muli %12, %c64 : index
                %22 = arith.muli %arg7, %c64 : index
                %23 = affine.apply affine_map<(d0, d1) -> (d0 * 512 + d1)>(%21, %22)
                %reinterpret_cast_0 = memref.reinterpret_cast %arg0 to offset: [%23], sizes: [64, 64], strides: [512, 1] : memref<4096x512xf32> to memref<64x64xf32, strided<[512, 1], offset: ?>>
                %24 = loom.alloc [64, 64] on @L1 : memref<64x64xf32>
                %25 = loom.copy_to_tensor %reinterpret_cast_0, %24 on @L1, interconnect : [], broadcast : [1, 1] : memref<64x64xf32, strided<[512, 1], offset: ?>>, memref<64x64xf32> -> tensor<64x64xf32>
                %26 = arith.muli %13, %c64 : index
                %27 = affine.apply affine_map<(d0, d1) -> (d0 * 4096 + d1)>(%22, %26)
                %reinterpret_cast_1 = memref.reinterpret_cast %arg1 to offset: [%27], sizes: [64, 64], strides: [4096, 1] : memref<512x4096xf32> to memref<64x64xf32, strided<[4096, 1], offset: ?>>
                %28 = loom.alloc [64, 64] on @L1 : memref<64x64xf32>
                %29 = loom.copy_to_tensor %reinterpret_cast_1, %28 on @L1, interconnect : [@vertical_links], broadcast : [8, 1] : memref<64x64xf32, strided<[4096, 1], offset: ?>>, memref<64x64xf32> -> tensor<64x64xf32>
                %30 = linalg.matmul ins(%25, %29 : tensor<64x64xf32>, tensor<64x64xf32>) outs(%arg8 : tensor<64x64xf32>) -> tensor<64x64xf32>
                affine.yield %30 : tensor<64x64xf32>
              }
              %18 = arith.muli %12, %c64 : index
              %19 = arith.muli %13, %c64 : index
              %20 = affine.apply affine_map<(d0, d1) -> (d0 * 4096 + d1)>(%18, %19)
              %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%20], sizes: [64, 64], strides: [4096, 1] : memref<4096x4096xf32> to memref<64x64xf32, strided<[4096, 1], offset: ?>>
              loom.copy_from_tensor %17, %reinterpret_cast on @DRAM : tensor<64x64xf32>, memref<64x64xf32, strided<[4096, 1], offset: ?>>
            } {loom.iter_type = #loom.iter_type<temporal>}
          } {loom.iter_type = #loom.iter_type<temporal>}
        } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @x}
      } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @y}
      return
    }
    func.func @matmul__d1i0_d0i1__f01__d_v__BK256__BM32__BN32(%arg0: memref<4096x512xf32>, %arg1: memref<512x4096xf32>, %arg2: memref<4096x4096xf32>) {
      %cst = arith.constant 0.000000e+00 : f32
      %c32 = arith.constant 32 : index
      %c256 = arith.constant 256 : index
      affine.parallel (%arg3) = (0) to (8) {
        affine.parallel (%arg4) = (0) to (8) {
          affine.for %arg5 = 0 to 16 {
            affine.for %arg6 = 0 to 16 {
              %12 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 8)>(%arg3, %arg5)
              %13 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 8)>(%arg4, %arg6)
              %14 = loom.alloc [32, 32] on @L1 : memref<32x32xf32>
              %15 = loom.init_tensor %14[32, 32] : memref<32x32xf32> -> tensor<32x32xf32>
              %16 = linalg.fill ins(%cst : f32) outs(%15 : tensor<32x32xf32>) -> tensor<32x32xf32>
              %17 = affine.for %arg7 = 0 to 2 iter_args(%arg8 = %16) -> (tensor<32x32xf32>) {
                %21 = arith.muli %12, %c32 : index
                %22 = arith.muli %arg7, %c256 : index
                %23 = affine.apply affine_map<(d0, d1) -> (d0 * 512 + d1)>(%21, %22)
                %reinterpret_cast_0 = memref.reinterpret_cast %arg0 to offset: [%23], sizes: [32, 256], strides: [512, 1] : memref<4096x512xf32> to memref<32x256xf32, strided<[512, 1], offset: ?>>
                %24 = loom.alloc [32, 256] on @L1 : memref<32x256xf32>
                %25 = loom.copy_to_tensor %reinterpret_cast_0, %24 on @L1, interconnect : [], broadcast : [1, 1] : memref<32x256xf32, strided<[512, 1], offset: ?>>, memref<32x256xf32> -> tensor<32x256xf32>
                %26 = arith.muli %13, %c32 : index
                %27 = affine.apply affine_map<(d0, d1) -> (d0 * 4096 + d1)>(%22, %26)
                %reinterpret_cast_1 = memref.reinterpret_cast %arg1 to offset: [%27], sizes: [256, 32], strides: [4096, 1] : memref<512x4096xf32> to memref<256x32xf32, strided<[4096, 1], offset: ?>>
                %28 = loom.alloc [256, 32] on @L1 : memref<256x32xf32>
                %29 = loom.copy_to_tensor %reinterpret_cast_1, %28 on @L1, interconnect : [@vertical_links], broadcast : [8, 1] : memref<256x32xf32, strided<[4096, 1], offset: ?>>, memref<256x32xf32> -> tensor<256x32xf32>
                %30 = linalg.matmul ins(%25, %29 : tensor<32x256xf32>, tensor<256x32xf32>) outs(%arg8 : tensor<32x32xf32>) -> tensor<32x32xf32>
                affine.yield %30 : tensor<32x32xf32>
              }
              %18 = arith.muli %12, %c32 : index
              %19 = arith.muli %13, %c32 : index
              %20 = affine.apply affine_map<(d0, d1) -> (d0 * 4096 + d1)>(%18, %19)
              %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%20], sizes: [32, 32], strides: [4096, 1] : memref<4096x4096xf32> to memref<32x32xf32, strided<[4096, 1], offset: ?>>
              loom.copy_from_tensor %17, %reinterpret_cast on @DRAM : tensor<32x32xf32>, memref<32x32xf32, strided<[4096, 1], offset: ?>>
            } {loom.iter_type = #loom.iter_type<temporal>}
          } {loom.iter_type = #loom.iter_type<temporal>}
        } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @x}
      } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @y}
      return
    }
  }
  module attributes {loom.pass_name = "Materialize"} {
    func.func @matmul__d1i0_d0i1__f01__h_d__BK64__BM64__BN64(%arg0: memref<4096x512xf32>, %arg1: memref<512x4096xf32>, %arg2: memref<4096x4096xf32>) {
      %cst = arith.constant 0.000000e+00 : f32
      %c64 = arith.constant 64 : index
      affine.parallel (%arg3) = (0) to (8) {
        affine.parallel (%arg4) = (0) to (8) {
          affine.for %arg5 = 0 to 8 {
            affine.for %arg6 = 0 to 8 {
              %12 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 8)>(%arg3, %arg5)
              %13 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 8)>(%arg4, %arg6)
              %14 = loom.alloc [64, 64] on @L1 : memref<64x64xf32>
              %15 = loom.init_tensor %14[64, 64] : memref<64x64xf32> -> tensor<64x64xf32>
              %16 = linalg.fill ins(%cst : f32) outs(%15 : tensor<64x64xf32>) -> tensor<64x64xf32>
              %17 = affine.for %arg7 = 0 to 8 iter_args(%arg8 = %16) -> (tensor<64x64xf32>) {
                %21 = arith.muli %12, %c64 : index
                %22 = arith.muli %arg7, %c64 : index
                %23 = affine.apply affine_map<(d0, d1) -> (d0 * 512 + d1)>(%21, %22)
                %reinterpret_cast_0 = memref.reinterpret_cast %arg0 to offset: [%23], sizes: [64, 64], strides: [512, 1] : memref<4096x512xf32> to memref<64x64xf32, strided<[512, 1], offset: ?>>
                %24 = loom.alloc [64, 64] on @L1 : memref<64x64xf32>
                %25 = loom.copy_to_tensor %reinterpret_cast_0, %24 on @L1, interconnect : [@horizontal_links], broadcast : [1, 8] : memref<64x64xf32, strided<[512, 1], offset: ?>>, memref<64x64xf32> -> tensor<64x64xf32>
                %26 = arith.muli %13, %c64 : index
                %27 = affine.apply affine_map<(d0, d1) -> (d0 * 4096 + d1)>(%22, %26)
                %reinterpret_cast_1 = memref.reinterpret_cast %arg1 to offset: [%27], sizes: [64, 64], strides: [4096, 1] : memref<512x4096xf32> to memref<64x64xf32, strided<[4096, 1], offset: ?>>
                %28 = loom.alloc [64, 64] on @L1 : memref<64x64xf32>
                %29 = loom.copy_to_tensor %reinterpret_cast_1, %28 on @L1, interconnect : [], broadcast : [1, 1] : memref<64x64xf32, strided<[4096, 1], offset: ?>>, memref<64x64xf32> -> tensor<64x64xf32>
                %30 = linalg.matmul ins(%25, %29 : tensor<64x64xf32>, tensor<64x64xf32>) outs(%arg8 : tensor<64x64xf32>) -> tensor<64x64xf32>
                affine.yield %30 : tensor<64x64xf32>
              }
              %18 = arith.muli %12, %c64 : index
              %19 = arith.muli %13, %c64 : index
              %20 = affine.apply affine_map<(d0, d1) -> (d0 * 4096 + d1)>(%18, %19)
              %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%20], sizes: [64, 64], strides: [4096, 1] : memref<4096x4096xf32> to memref<64x64xf32, strided<[4096, 1], offset: ?>>
              loom.copy_from_tensor %17, %reinterpret_cast on @DRAM : tensor<64x64xf32>, memref<64x64xf32, strided<[4096, 1], offset: ?>>
            } {loom.iter_type = #loom.iter_type<temporal>}
          } {loom.iter_type = #loom.iter_type<temporal>}
        } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @x}
      } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @y}
      return
    }
    func.func @matmul__d1i0_d0i1__f01__h_d__BK256__BM32__BN32(%arg0: memref<4096x512xf32>, %arg1: memref<512x4096xf32>, %arg2: memref<4096x4096xf32>) {
      %cst = arith.constant 0.000000e+00 : f32
      %c32 = arith.constant 32 : index
      %c256 = arith.constant 256 : index
      affine.parallel (%arg3) = (0) to (8) {
        affine.parallel (%arg4) = (0) to (8) {
          affine.for %arg5 = 0 to 16 {
            affine.for %arg6 = 0 to 16 {
              %12 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 8)>(%arg3, %arg5)
              %13 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 8)>(%arg4, %arg6)
              %14 = loom.alloc [32, 32] on @L1 : memref<32x32xf32>
              %15 = loom.init_tensor %14[32, 32] : memref<32x32xf32> -> tensor<32x32xf32>
              %16 = linalg.fill ins(%cst : f32) outs(%15 : tensor<32x32xf32>) -> tensor<32x32xf32>
              %17 = affine.for %arg7 = 0 to 2 iter_args(%arg8 = %16) -> (tensor<32x32xf32>) {
                %21 = arith.muli %12, %c32 : index
                %22 = arith.muli %arg7, %c256 : index
                %23 = affine.apply affine_map<(d0, d1) -> (d0 * 512 + d1)>(%21, %22)
                %reinterpret_cast_0 = memref.reinterpret_cast %arg0 to offset: [%23], sizes: [32, 256], strides: [512, 1] : memref<4096x512xf32> to memref<32x256xf32, strided<[512, 1], offset: ?>>
                %24 = loom.alloc [32, 256] on @L1 : memref<32x256xf32>
                %25 = loom.copy_to_tensor %reinterpret_cast_0, %24 on @L1, interconnect : [@horizontal_links], broadcast : [1, 8] : memref<32x256xf32, strided<[512, 1], offset: ?>>, memref<32x256xf32> -> tensor<32x256xf32>
                %26 = arith.muli %13, %c32 : index
                %27 = affine.apply affine_map<(d0, d1) -> (d0 * 4096 + d1)>(%22, %26)
                %reinterpret_cast_1 = memref.reinterpret_cast %arg1 to offset: [%27], sizes: [256, 32], strides: [4096, 1] : memref<512x4096xf32> to memref<256x32xf32, strided<[4096, 1], offset: ?>>
                %28 = loom.alloc [256, 32] on @L1 : memref<256x32xf32>
                %29 = loom.copy_to_tensor %reinterpret_cast_1, %28 on @L1, interconnect : [], broadcast : [1, 1] : memref<256x32xf32, strided<[4096, 1], offset: ?>>, memref<256x32xf32> -> tensor<256x32xf32>
                %30 = linalg.matmul ins(%25, %29 : tensor<32x256xf32>, tensor<256x32xf32>) outs(%arg8 : tensor<32x32xf32>) -> tensor<32x32xf32>
                affine.yield %30 : tensor<32x32xf32>
              }
              %18 = arith.muli %12, %c32 : index
              %19 = arith.muli %13, %c32 : index
              %20 = affine.apply affine_map<(d0, d1) -> (d0 * 4096 + d1)>(%18, %19)
              %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%20], sizes: [32, 32], strides: [4096, 1] : memref<4096x4096xf32> to memref<32x32xf32, strided<[4096, 1], offset: ?>>
              loom.copy_from_tensor %17, %reinterpret_cast on @DRAM : tensor<32x32xf32>, memref<32x32xf32, strided<[4096, 1], offset: ?>>
            } {loom.iter_type = #loom.iter_type<temporal>}
          } {loom.iter_type = #loom.iter_type<temporal>}
        } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @x}
      } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @y}
      return
    }
  }
  module attributes {loom.pass_name = "Materialize"} {
    func.func @matmul__d1i0_d0i1__f01__h_v__BK64__BM64__BN64(%arg0: memref<4096x512xf32>, %arg1: memref<512x4096xf32>, %arg2: memref<4096x4096xf32>) {
      %cst = arith.constant 0.000000e+00 : f32
      %c64 = arith.constant 64 : index
      affine.parallel (%arg3) = (0) to (8) {
        affine.parallel (%arg4) = (0) to (8) {
          affine.for %arg5 = 0 to 8 {
            affine.for %arg6 = 0 to 8 {
              %12 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 8)>(%arg3, %arg5)
              %13 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 8)>(%arg4, %arg6)
              %14 = loom.alloc [64, 64] on @L1 : memref<64x64xf32>
              %15 = loom.init_tensor %14[64, 64] : memref<64x64xf32> -> tensor<64x64xf32>
              %16 = linalg.fill ins(%cst : f32) outs(%15 : tensor<64x64xf32>) -> tensor<64x64xf32>
              %17 = affine.for %arg7 = 0 to 8 iter_args(%arg8 = %16) -> (tensor<64x64xf32>) {
                %21 = arith.muli %12, %c64 : index
                %22 = arith.muli %arg7, %c64 : index
                %23 = affine.apply affine_map<(d0, d1) -> (d0 * 512 + d1)>(%21, %22)
                %reinterpret_cast_0 = memref.reinterpret_cast %arg0 to offset: [%23], sizes: [64, 64], strides: [512, 1] : memref<4096x512xf32> to memref<64x64xf32, strided<[512, 1], offset: ?>>
                %24 = loom.alloc [64, 64] on @L1 : memref<64x64xf32>
                %25 = loom.copy_to_tensor %reinterpret_cast_0, %24 on @L1, interconnect : [@horizontal_links], broadcast : [1, 8] : memref<64x64xf32, strided<[512, 1], offset: ?>>, memref<64x64xf32> -> tensor<64x64xf32>
                %26 = arith.muli %13, %c64 : index
                %27 = affine.apply affine_map<(d0, d1) -> (d0 * 4096 + d1)>(%22, %26)
                %reinterpret_cast_1 = memref.reinterpret_cast %arg1 to offset: [%27], sizes: [64, 64], strides: [4096, 1] : memref<512x4096xf32> to memref<64x64xf32, strided<[4096, 1], offset: ?>>
                %28 = loom.alloc [64, 64] on @L1 : memref<64x64xf32>
                %29 = loom.copy_to_tensor %reinterpret_cast_1, %28 on @L1, interconnect : [@vertical_links], broadcast : [8, 1] : memref<64x64xf32, strided<[4096, 1], offset: ?>>, memref<64x64xf32> -> tensor<64x64xf32>
                %30 = linalg.matmul ins(%25, %29 : tensor<64x64xf32>, tensor<64x64xf32>) outs(%arg8 : tensor<64x64xf32>) -> tensor<64x64xf32>
                affine.yield %30 : tensor<64x64xf32>
              }
              %18 = arith.muli %12, %c64 : index
              %19 = arith.muli %13, %c64 : index
              %20 = affine.apply affine_map<(d0, d1) -> (d0 * 4096 + d1)>(%18, %19)
              %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%20], sizes: [64, 64], strides: [4096, 1] : memref<4096x4096xf32> to memref<64x64xf32, strided<[4096, 1], offset: ?>>
              loom.copy_from_tensor %17, %reinterpret_cast on @DRAM : tensor<64x64xf32>, memref<64x64xf32, strided<[4096, 1], offset: ?>>
            } {loom.iter_type = #loom.iter_type<temporal>}
          } {loom.iter_type = #loom.iter_type<temporal>}
        } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @x}
      } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @y}
      return
    }
    func.func @matmul__d1i0_d0i1__f01__h_v__BK256__BM32__BN32(%arg0: memref<4096x512xf32>, %arg1: memref<512x4096xf32>, %arg2: memref<4096x4096xf32>) {
      %cst = arith.constant 0.000000e+00 : f32
      %c32 = arith.constant 32 : index
      %c256 = arith.constant 256 : index
      affine.parallel (%arg3) = (0) to (8) {
        affine.parallel (%arg4) = (0) to (8) {
          affine.for %arg5 = 0 to 16 {
            affine.for %arg6 = 0 to 16 {
              %12 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 8)>(%arg3, %arg5)
              %13 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 8)>(%arg4, %arg6)
              %14 = loom.alloc [32, 32] on @L1 : memref<32x32xf32>
              %15 = loom.init_tensor %14[32, 32] : memref<32x32xf32> -> tensor<32x32xf32>
              %16 = linalg.fill ins(%cst : f32) outs(%15 : tensor<32x32xf32>) -> tensor<32x32xf32>
              %17 = affine.for %arg7 = 0 to 2 iter_args(%arg8 = %16) -> (tensor<32x32xf32>) {
                %21 = arith.muli %12, %c32 : index
                %22 = arith.muli %arg7, %c256 : index
                %23 = affine.apply affine_map<(d0, d1) -> (d0 * 512 + d1)>(%21, %22)
                %reinterpret_cast_0 = memref.reinterpret_cast %arg0 to offset: [%23], sizes: [32, 256], strides: [512, 1] : memref<4096x512xf32> to memref<32x256xf32, strided<[512, 1], offset: ?>>
                %24 = loom.alloc [32, 256] on @L1 : memref<32x256xf32>
                %25 = loom.copy_to_tensor %reinterpret_cast_0, %24 on @L1, interconnect : [@horizontal_links], broadcast : [1, 8] : memref<32x256xf32, strided<[512, 1], offset: ?>>, memref<32x256xf32> -> tensor<32x256xf32>
                %26 = arith.muli %13, %c32 : index
                %27 = affine.apply affine_map<(d0, d1) -> (d0 * 4096 + d1)>(%22, %26)
                %reinterpret_cast_1 = memref.reinterpret_cast %arg1 to offset: [%27], sizes: [256, 32], strides: [4096, 1] : memref<512x4096xf32> to memref<256x32xf32, strided<[4096, 1], offset: ?>>
                %28 = loom.alloc [256, 32] on @L1 : memref<256x32xf32>
                %29 = loom.copy_to_tensor %reinterpret_cast_1, %28 on @L1, interconnect : [@vertical_links], broadcast : [8, 1] : memref<256x32xf32, strided<[4096, 1], offset: ?>>, memref<256x32xf32> -> tensor<256x32xf32>
                %30 = linalg.matmul ins(%25, %29 : tensor<32x256xf32>, tensor<256x32xf32>) outs(%arg8 : tensor<32x32xf32>) -> tensor<32x32xf32>
                affine.yield %30 : tensor<32x32xf32>
              }
              %18 = arith.muli %12, %c32 : index
              %19 = arith.muli %13, %c32 : index
              %20 = affine.apply affine_map<(d0, d1) -> (d0 * 4096 + d1)>(%18, %19)
              %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%20], sizes: [32, 32], strides: [4096, 1] : memref<4096x4096xf32> to memref<32x32xf32, strided<[4096, 1], offset: ?>>
              loom.copy_from_tensor %17, %reinterpret_cast on @DRAM : tensor<32x32xf32>, memref<32x32xf32, strided<[4096, 1], offset: ?>>
            } {loom.iter_type = #loom.iter_type<temporal>}
          } {loom.iter_type = #loom.iter_type<temporal>}
        } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @x}
      } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @y}
      return
    }
  }
  module attributes {loom.pass_name = "Materialize"} {
    func.func @matmul__d1i0_d0i1__f10__d_d__BK64__BM64__BN64(%arg0: memref<4096x512xf32>, %arg1: memref<512x4096xf32>, %arg2: memref<4096x4096xf32>) {
      %cst = arith.constant 0.000000e+00 : f32
      %c64 = arith.constant 64 : index
      affine.parallel (%arg3) = (0) to (8) {
        affine.parallel (%arg4) = (0) to (8) {
          affine.for %arg5 = 0 to 8 {
            affine.for %arg6 = 0 to 8 {
              %12 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 8)>(%arg3, %arg6)
              %13 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 8)>(%arg4, %arg5)
              %14 = loom.alloc [64, 64] on @L1 : memref<64x64xf32>
              %15 = loom.init_tensor %14[64, 64] : memref<64x64xf32> -> tensor<64x64xf32>
              %16 = linalg.fill ins(%cst : f32) outs(%15 : tensor<64x64xf32>) -> tensor<64x64xf32>
              %17 = affine.for %arg7 = 0 to 8 iter_args(%arg8 = %16) -> (tensor<64x64xf32>) {
                %21 = arith.muli %12, %c64 : index
                %22 = arith.muli %arg7, %c64 : index
                %23 = affine.apply affine_map<(d0, d1) -> (d0 * 512 + d1)>(%21, %22)
                %reinterpret_cast_0 = memref.reinterpret_cast %arg0 to offset: [%23], sizes: [64, 64], strides: [512, 1] : memref<4096x512xf32> to memref<64x64xf32, strided<[512, 1], offset: ?>>
                %24 = loom.alloc [64, 64] on @L1 : memref<64x64xf32>
                %25 = loom.copy_to_tensor %reinterpret_cast_0, %24 on @L1, interconnect : [], broadcast : [1, 1] : memref<64x64xf32, strided<[512, 1], offset: ?>>, memref<64x64xf32> -> tensor<64x64xf32>
                %26 = arith.muli %13, %c64 : index
                %27 = affine.apply affine_map<(d0, d1) -> (d0 * 4096 + d1)>(%22, %26)
                %reinterpret_cast_1 = memref.reinterpret_cast %arg1 to offset: [%27], sizes: [64, 64], strides: [4096, 1] : memref<512x4096xf32> to memref<64x64xf32, strided<[4096, 1], offset: ?>>
                %28 = loom.alloc [64, 64] on @L1 : memref<64x64xf32>
                %29 = loom.copy_to_tensor %reinterpret_cast_1, %28 on @L1, interconnect : [], broadcast : [1, 1] : memref<64x64xf32, strided<[4096, 1], offset: ?>>, memref<64x64xf32> -> tensor<64x64xf32>
                %30 = linalg.matmul ins(%25, %29 : tensor<64x64xf32>, tensor<64x64xf32>) outs(%arg8 : tensor<64x64xf32>) -> tensor<64x64xf32>
                affine.yield %30 : tensor<64x64xf32>
              }
              %18 = arith.muli %12, %c64 : index
              %19 = arith.muli %13, %c64 : index
              %20 = affine.apply affine_map<(d0, d1) -> (d0 * 4096 + d1)>(%18, %19)
              %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%20], sizes: [64, 64], strides: [4096, 1] : memref<4096x4096xf32> to memref<64x64xf32, strided<[4096, 1], offset: ?>>
              loom.copy_from_tensor %17, %reinterpret_cast on @DRAM : tensor<64x64xf32>, memref<64x64xf32, strided<[4096, 1], offset: ?>>
            } {loom.iter_type = #loom.iter_type<temporal>}
          } {loom.iter_type = #loom.iter_type<temporal>}
        } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @x}
      } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @y}
      return
    }
    func.func @matmul__d1i0_d0i1__f10__d_d__BK256__BM32__BN32(%arg0: memref<4096x512xf32>, %arg1: memref<512x4096xf32>, %arg2: memref<4096x4096xf32>) {
      %cst = arith.constant 0.000000e+00 : f32
      %c32 = arith.constant 32 : index
      %c256 = arith.constant 256 : index
      affine.parallel (%arg3) = (0) to (8) {
        affine.parallel (%arg4) = (0) to (8) {
          affine.for %arg5 = 0 to 16 {
            affine.for %arg6 = 0 to 16 {
              %12 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 8)>(%arg3, %arg6)
              %13 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 8)>(%arg4, %arg5)
              %14 = loom.alloc [32, 32] on @L1 : memref<32x32xf32>
              %15 = loom.init_tensor %14[32, 32] : memref<32x32xf32> -> tensor<32x32xf32>
              %16 = linalg.fill ins(%cst : f32) outs(%15 : tensor<32x32xf32>) -> tensor<32x32xf32>
              %17 = affine.for %arg7 = 0 to 2 iter_args(%arg8 = %16) -> (tensor<32x32xf32>) {
                %21 = arith.muli %12, %c32 : index
                %22 = arith.muli %arg7, %c256 : index
                %23 = affine.apply affine_map<(d0, d1) -> (d0 * 512 + d1)>(%21, %22)
                %reinterpret_cast_0 = memref.reinterpret_cast %arg0 to offset: [%23], sizes: [32, 256], strides: [512, 1] : memref<4096x512xf32> to memref<32x256xf32, strided<[512, 1], offset: ?>>
                %24 = loom.alloc [32, 256] on @L1 : memref<32x256xf32>
                %25 = loom.copy_to_tensor %reinterpret_cast_0, %24 on @L1, interconnect : [], broadcast : [1, 1] : memref<32x256xf32, strided<[512, 1], offset: ?>>, memref<32x256xf32> -> tensor<32x256xf32>
                %26 = arith.muli %13, %c32 : index
                %27 = affine.apply affine_map<(d0, d1) -> (d0 * 4096 + d1)>(%22, %26)
                %reinterpret_cast_1 = memref.reinterpret_cast %arg1 to offset: [%27], sizes: [256, 32], strides: [4096, 1] : memref<512x4096xf32> to memref<256x32xf32, strided<[4096, 1], offset: ?>>
                %28 = loom.alloc [256, 32] on @L1 : memref<256x32xf32>
                %29 = loom.copy_to_tensor %reinterpret_cast_1, %28 on @L1, interconnect : [], broadcast : [1, 1] : memref<256x32xf32, strided<[4096, 1], offset: ?>>, memref<256x32xf32> -> tensor<256x32xf32>
                %30 = linalg.matmul ins(%25, %29 : tensor<32x256xf32>, tensor<256x32xf32>) outs(%arg8 : tensor<32x32xf32>) -> tensor<32x32xf32>
                affine.yield %30 : tensor<32x32xf32>
              }
              %18 = arith.muli %12, %c32 : index
              %19 = arith.muli %13, %c32 : index
              %20 = affine.apply affine_map<(d0, d1) -> (d0 * 4096 + d1)>(%18, %19)
              %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%20], sizes: [32, 32], strides: [4096, 1] : memref<4096x4096xf32> to memref<32x32xf32, strided<[4096, 1], offset: ?>>
              loom.copy_from_tensor %17, %reinterpret_cast on @DRAM : tensor<32x32xf32>, memref<32x32xf32, strided<[4096, 1], offset: ?>>
            } {loom.iter_type = #loom.iter_type<temporal>}
          } {loom.iter_type = #loom.iter_type<temporal>}
        } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @x}
      } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @y}
      return
    }
  }
  module attributes {loom.pass_name = "Materialize"} {
    func.func @matmul__d1i0_d0i1__f10__d_v__BK64__BM64__BN64(%arg0: memref<4096x512xf32>, %arg1: memref<512x4096xf32>, %arg2: memref<4096x4096xf32>) {
      %cst = arith.constant 0.000000e+00 : f32
      %c64 = arith.constant 64 : index
      affine.parallel (%arg3) = (0) to (8) {
        affine.parallel (%arg4) = (0) to (8) {
          affine.for %arg5 = 0 to 8 {
            affine.for %arg6 = 0 to 8 {
              %12 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 8)>(%arg3, %arg6)
              %13 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 8)>(%arg4, %arg5)
              %14 = loom.alloc [64, 64] on @L1 : memref<64x64xf32>
              %15 = loom.init_tensor %14[64, 64] : memref<64x64xf32> -> tensor<64x64xf32>
              %16 = linalg.fill ins(%cst : f32) outs(%15 : tensor<64x64xf32>) -> tensor<64x64xf32>
              %17 = affine.for %arg7 = 0 to 8 iter_args(%arg8 = %16) -> (tensor<64x64xf32>) {
                %21 = arith.muli %12, %c64 : index
                %22 = arith.muli %arg7, %c64 : index
                %23 = affine.apply affine_map<(d0, d1) -> (d0 * 512 + d1)>(%21, %22)
                %reinterpret_cast_0 = memref.reinterpret_cast %arg0 to offset: [%23], sizes: [64, 64], strides: [512, 1] : memref<4096x512xf32> to memref<64x64xf32, strided<[512, 1], offset: ?>>
                %24 = loom.alloc [64, 64] on @L1 : memref<64x64xf32>
                %25 = loom.copy_to_tensor %reinterpret_cast_0, %24 on @L1, interconnect : [], broadcast : [1, 1] : memref<64x64xf32, strided<[512, 1], offset: ?>>, memref<64x64xf32> -> tensor<64x64xf32>
                %26 = arith.muli %13, %c64 : index
                %27 = affine.apply affine_map<(d0, d1) -> (d0 * 4096 + d1)>(%22, %26)
                %reinterpret_cast_1 = memref.reinterpret_cast %arg1 to offset: [%27], sizes: [64, 64], strides: [4096, 1] : memref<512x4096xf32> to memref<64x64xf32, strided<[4096, 1], offset: ?>>
                %28 = loom.alloc [64, 64] on @L1 : memref<64x64xf32>
                %29 = loom.copy_to_tensor %reinterpret_cast_1, %28 on @L1, interconnect : [@vertical_links], broadcast : [8, 1] : memref<64x64xf32, strided<[4096, 1], offset: ?>>, memref<64x64xf32> -> tensor<64x64xf32>
                %30 = linalg.matmul ins(%25, %29 : tensor<64x64xf32>, tensor<64x64xf32>) outs(%arg8 : tensor<64x64xf32>) -> tensor<64x64xf32>
                affine.yield %30 : tensor<64x64xf32>
              }
              %18 = arith.muli %12, %c64 : index
              %19 = arith.muli %13, %c64 : index
              %20 = affine.apply affine_map<(d0, d1) -> (d0 * 4096 + d1)>(%18, %19)
              %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%20], sizes: [64, 64], strides: [4096, 1] : memref<4096x4096xf32> to memref<64x64xf32, strided<[4096, 1], offset: ?>>
              loom.copy_from_tensor %17, %reinterpret_cast on @DRAM : tensor<64x64xf32>, memref<64x64xf32, strided<[4096, 1], offset: ?>>
            } {loom.iter_type = #loom.iter_type<temporal>}
          } {loom.iter_type = #loom.iter_type<temporal>}
        } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @x}
      } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @y}
      return
    }
    func.func @matmul__d1i0_d0i1__f10__d_v__BK256__BM32__BN32(%arg0: memref<4096x512xf32>, %arg1: memref<512x4096xf32>, %arg2: memref<4096x4096xf32>) {
      %cst = arith.constant 0.000000e+00 : f32
      %c32 = arith.constant 32 : index
      %c256 = arith.constant 256 : index
      affine.parallel (%arg3) = (0) to (8) {
        affine.parallel (%arg4) = (0) to (8) {
          affine.for %arg5 = 0 to 16 {
            affine.for %arg6 = 0 to 16 {
              %12 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 8)>(%arg3, %arg6)
              %13 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 8)>(%arg4, %arg5)
              %14 = loom.alloc [32, 32] on @L1 : memref<32x32xf32>
              %15 = loom.init_tensor %14[32, 32] : memref<32x32xf32> -> tensor<32x32xf32>
              %16 = linalg.fill ins(%cst : f32) outs(%15 : tensor<32x32xf32>) -> tensor<32x32xf32>
              %17 = affine.for %arg7 = 0 to 2 iter_args(%arg8 = %16) -> (tensor<32x32xf32>) {
                %21 = arith.muli %12, %c32 : index
                %22 = arith.muli %arg7, %c256 : index
                %23 = affine.apply affine_map<(d0, d1) -> (d0 * 512 + d1)>(%21, %22)
                %reinterpret_cast_0 = memref.reinterpret_cast %arg0 to offset: [%23], sizes: [32, 256], strides: [512, 1] : memref<4096x512xf32> to memref<32x256xf32, strided<[512, 1], offset: ?>>
                %24 = loom.alloc [32, 256] on @L1 : memref<32x256xf32>
                %25 = loom.copy_to_tensor %reinterpret_cast_0, %24 on @L1, interconnect : [], broadcast : [1, 1] : memref<32x256xf32, strided<[512, 1], offset: ?>>, memref<32x256xf32> -> tensor<32x256xf32>
                %26 = arith.muli %13, %c32 : index
                %27 = affine.apply affine_map<(d0, d1) -> (d0 * 4096 + d1)>(%22, %26)
                %reinterpret_cast_1 = memref.reinterpret_cast %arg1 to offset: [%27], sizes: [256, 32], strides: [4096, 1] : memref<512x4096xf32> to memref<256x32xf32, strided<[4096, 1], offset: ?>>
                %28 = loom.alloc [256, 32] on @L1 : memref<256x32xf32>
                %29 = loom.copy_to_tensor %reinterpret_cast_1, %28 on @L1, interconnect : [@vertical_links], broadcast : [8, 1] : memref<256x32xf32, strided<[4096, 1], offset: ?>>, memref<256x32xf32> -> tensor<256x32xf32>
                %30 = linalg.matmul ins(%25, %29 : tensor<32x256xf32>, tensor<256x32xf32>) outs(%arg8 : tensor<32x32xf32>) -> tensor<32x32xf32>
                affine.yield %30 : tensor<32x32xf32>
              }
              %18 = arith.muli %12, %c32 : index
              %19 = arith.muli %13, %c32 : index
              %20 = affine.apply affine_map<(d0, d1) -> (d0 * 4096 + d1)>(%18, %19)
              %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%20], sizes: [32, 32], strides: [4096, 1] : memref<4096x4096xf32> to memref<32x32xf32, strided<[4096, 1], offset: ?>>
              loom.copy_from_tensor %17, %reinterpret_cast on @DRAM : tensor<32x32xf32>, memref<32x32xf32, strided<[4096, 1], offset: ?>>
            } {loom.iter_type = #loom.iter_type<temporal>}
          } {loom.iter_type = #loom.iter_type<temporal>}
        } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @x}
      } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @y}
      return
    }
  }
  module attributes {loom.pass_name = "Materialize"} {
    func.func @matmul__d1i0_d0i1__f10__h_d__BK64__BM64__BN64(%arg0: memref<4096x512xf32>, %arg1: memref<512x4096xf32>, %arg2: memref<4096x4096xf32>) {
      %cst = arith.constant 0.000000e+00 : f32
      %c64 = arith.constant 64 : index
      affine.parallel (%arg3) = (0) to (8) {
        affine.parallel (%arg4) = (0) to (8) {
          affine.for %arg5 = 0 to 8 {
            affine.for %arg6 = 0 to 8 {
              %12 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 8)>(%arg3, %arg6)
              %13 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 8)>(%arg4, %arg5)
              %14 = loom.alloc [64, 64] on @L1 : memref<64x64xf32>
              %15 = loom.init_tensor %14[64, 64] : memref<64x64xf32> -> tensor<64x64xf32>
              %16 = linalg.fill ins(%cst : f32) outs(%15 : tensor<64x64xf32>) -> tensor<64x64xf32>
              %17 = affine.for %arg7 = 0 to 8 iter_args(%arg8 = %16) -> (tensor<64x64xf32>) {
                %21 = arith.muli %12, %c64 : index
                %22 = arith.muli %arg7, %c64 : index
                %23 = affine.apply affine_map<(d0, d1) -> (d0 * 512 + d1)>(%21, %22)
                %reinterpret_cast_0 = memref.reinterpret_cast %arg0 to offset: [%23], sizes: [64, 64], strides: [512, 1] : memref<4096x512xf32> to memref<64x64xf32, strided<[512, 1], offset: ?>>
                %24 = loom.alloc [64, 64] on @L1 : memref<64x64xf32>
                %25 = loom.copy_to_tensor %reinterpret_cast_0, %24 on @L1, interconnect : [@horizontal_links], broadcast : [1, 8] : memref<64x64xf32, strided<[512, 1], offset: ?>>, memref<64x64xf32> -> tensor<64x64xf32>
                %26 = arith.muli %13, %c64 : index
                %27 = affine.apply affine_map<(d0, d1) -> (d0 * 4096 + d1)>(%22, %26)
                %reinterpret_cast_1 = memref.reinterpret_cast %arg1 to offset: [%27], sizes: [64, 64], strides: [4096, 1] : memref<512x4096xf32> to memref<64x64xf32, strided<[4096, 1], offset: ?>>
                %28 = loom.alloc [64, 64] on @L1 : memref<64x64xf32>
                %29 = loom.copy_to_tensor %reinterpret_cast_1, %28 on @L1, interconnect : [], broadcast : [1, 1] : memref<64x64xf32, strided<[4096, 1], offset: ?>>, memref<64x64xf32> -> tensor<64x64xf32>
                %30 = linalg.matmul ins(%25, %29 : tensor<64x64xf32>, tensor<64x64xf32>) outs(%arg8 : tensor<64x64xf32>) -> tensor<64x64xf32>
                affine.yield %30 : tensor<64x64xf32>
              }
              %18 = arith.muli %12, %c64 : index
              %19 = arith.muli %13, %c64 : index
              %20 = affine.apply affine_map<(d0, d1) -> (d0 * 4096 + d1)>(%18, %19)
              %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%20], sizes: [64, 64], strides: [4096, 1] : memref<4096x4096xf32> to memref<64x64xf32, strided<[4096, 1], offset: ?>>
              loom.copy_from_tensor %17, %reinterpret_cast on @DRAM : tensor<64x64xf32>, memref<64x64xf32, strided<[4096, 1], offset: ?>>
            } {loom.iter_type = #loom.iter_type<temporal>}
          } {loom.iter_type = #loom.iter_type<temporal>}
        } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @x}
      } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @y}
      return
    }
    func.func @matmul__d1i0_d0i1__f10__h_d__BK256__BM32__BN32(%arg0: memref<4096x512xf32>, %arg1: memref<512x4096xf32>, %arg2: memref<4096x4096xf32>) {
      %cst = arith.constant 0.000000e+00 : f32
      %c32 = arith.constant 32 : index
      %c256 = arith.constant 256 : index
      affine.parallel (%arg3) = (0) to (8) {
        affine.parallel (%arg4) = (0) to (8) {
          affine.for %arg5 = 0 to 16 {
            affine.for %arg6 = 0 to 16 {
              %12 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 8)>(%arg3, %arg6)
              %13 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 8)>(%arg4, %arg5)
              %14 = loom.alloc [32, 32] on @L1 : memref<32x32xf32>
              %15 = loom.init_tensor %14[32, 32] : memref<32x32xf32> -> tensor<32x32xf32>
              %16 = linalg.fill ins(%cst : f32) outs(%15 : tensor<32x32xf32>) -> tensor<32x32xf32>
              %17 = affine.for %arg7 = 0 to 2 iter_args(%arg8 = %16) -> (tensor<32x32xf32>) {
                %21 = arith.muli %12, %c32 : index
                %22 = arith.muli %arg7, %c256 : index
                %23 = affine.apply affine_map<(d0, d1) -> (d0 * 512 + d1)>(%21, %22)
                %reinterpret_cast_0 = memref.reinterpret_cast %arg0 to offset: [%23], sizes: [32, 256], strides: [512, 1] : memref<4096x512xf32> to memref<32x256xf32, strided<[512, 1], offset: ?>>
                %24 = loom.alloc [32, 256] on @L1 : memref<32x256xf32>
                %25 = loom.copy_to_tensor %reinterpret_cast_0, %24 on @L1, interconnect : [@horizontal_links], broadcast : [1, 8] : memref<32x256xf32, strided<[512, 1], offset: ?>>, memref<32x256xf32> -> tensor<32x256xf32>
                %26 = arith.muli %13, %c32 : index
                %27 = affine.apply affine_map<(d0, d1) -> (d0 * 4096 + d1)>(%22, %26)
                %reinterpret_cast_1 = memref.reinterpret_cast %arg1 to offset: [%27], sizes: [256, 32], strides: [4096, 1] : memref<512x4096xf32> to memref<256x32xf32, strided<[4096, 1], offset: ?>>
                %28 = loom.alloc [256, 32] on @L1 : memref<256x32xf32>
                %29 = loom.copy_to_tensor %reinterpret_cast_1, %28 on @L1, interconnect : [], broadcast : [1, 1] : memref<256x32xf32, strided<[4096, 1], offset: ?>>, memref<256x32xf32> -> tensor<256x32xf32>
                %30 = linalg.matmul ins(%25, %29 : tensor<32x256xf32>, tensor<256x32xf32>) outs(%arg8 : tensor<32x32xf32>) -> tensor<32x32xf32>
                affine.yield %30 : tensor<32x32xf32>
              }
              %18 = arith.muli %12, %c32 : index
              %19 = arith.muli %13, %c32 : index
              %20 = affine.apply affine_map<(d0, d1) -> (d0 * 4096 + d1)>(%18, %19)
              %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%20], sizes: [32, 32], strides: [4096, 1] : memref<4096x4096xf32> to memref<32x32xf32, strided<[4096, 1], offset: ?>>
              loom.copy_from_tensor %17, %reinterpret_cast on @DRAM : tensor<32x32xf32>, memref<32x32xf32, strided<[4096, 1], offset: ?>>
            } {loom.iter_type = #loom.iter_type<temporal>}
          } {loom.iter_type = #loom.iter_type<temporal>}
        } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @x}
      } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @y}
      return
    }
  }
  module attributes {loom.pass_name = "Materialize"} {
    func.func @matmul__d1i0_d0i1__f10__h_v__BK64__BM64__BN64(%arg0: memref<4096x512xf32>, %arg1: memref<512x4096xf32>, %arg2: memref<4096x4096xf32>) {
      %cst = arith.constant 0.000000e+00 : f32
      %c64 = arith.constant 64 : index
      affine.parallel (%arg3) = (0) to (8) {
        affine.parallel (%arg4) = (0) to (8) {
          affine.for %arg5 = 0 to 8 {
            affine.for %arg6 = 0 to 8 {
              %12 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 8)>(%arg3, %arg6)
              %13 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 8)>(%arg4, %arg5)
              %14 = loom.alloc [64, 64] on @L1 : memref<64x64xf32>
              %15 = loom.init_tensor %14[64, 64] : memref<64x64xf32> -> tensor<64x64xf32>
              %16 = linalg.fill ins(%cst : f32) outs(%15 : tensor<64x64xf32>) -> tensor<64x64xf32>
              %17 = affine.for %arg7 = 0 to 8 iter_args(%arg8 = %16) -> (tensor<64x64xf32>) {
                %21 = arith.muli %12, %c64 : index
                %22 = arith.muli %arg7, %c64 : index
                %23 = affine.apply affine_map<(d0, d1) -> (d0 * 512 + d1)>(%21, %22)
                %reinterpret_cast_0 = memref.reinterpret_cast %arg0 to offset: [%23], sizes: [64, 64], strides: [512, 1] : memref<4096x512xf32> to memref<64x64xf32, strided<[512, 1], offset: ?>>
                %24 = loom.alloc [64, 64] on @L1 : memref<64x64xf32>
                %25 = loom.copy_to_tensor %reinterpret_cast_0, %24 on @L1, interconnect : [@horizontal_links], broadcast : [1, 8] : memref<64x64xf32, strided<[512, 1], offset: ?>>, memref<64x64xf32> -> tensor<64x64xf32>
                %26 = arith.muli %13, %c64 : index
                %27 = affine.apply affine_map<(d0, d1) -> (d0 * 4096 + d1)>(%22, %26)
                %reinterpret_cast_1 = memref.reinterpret_cast %arg1 to offset: [%27], sizes: [64, 64], strides: [4096, 1] : memref<512x4096xf32> to memref<64x64xf32, strided<[4096, 1], offset: ?>>
                %28 = loom.alloc [64, 64] on @L1 : memref<64x64xf32>
                %29 = loom.copy_to_tensor %reinterpret_cast_1, %28 on @L1, interconnect : [@vertical_links], broadcast : [8, 1] : memref<64x64xf32, strided<[4096, 1], offset: ?>>, memref<64x64xf32> -> tensor<64x64xf32>
                %30 = linalg.matmul ins(%25, %29 : tensor<64x64xf32>, tensor<64x64xf32>) outs(%arg8 : tensor<64x64xf32>) -> tensor<64x64xf32>
                affine.yield %30 : tensor<64x64xf32>
              }
              %18 = arith.muli %12, %c64 : index
              %19 = arith.muli %13, %c64 : index
              %20 = affine.apply affine_map<(d0, d1) -> (d0 * 4096 + d1)>(%18, %19)
              %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%20], sizes: [64, 64], strides: [4096, 1] : memref<4096x4096xf32> to memref<64x64xf32, strided<[4096, 1], offset: ?>>
              loom.copy_from_tensor %17, %reinterpret_cast on @DRAM : tensor<64x64xf32>, memref<64x64xf32, strided<[4096, 1], offset: ?>>
            } {loom.iter_type = #loom.iter_type<temporal>}
          } {loom.iter_type = #loom.iter_type<temporal>}
        } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @x}
      } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @y}
      return
    }
    func.func @matmul__d1i0_d0i1__f10__h_v__BK256__BM32__BN32(%arg0: memref<4096x512xf32>, %arg1: memref<512x4096xf32>, %arg2: memref<4096x4096xf32>) {
      %cst = arith.constant 0.000000e+00 : f32
      %c32 = arith.constant 32 : index
      %c256 = arith.constant 256 : index
      affine.parallel (%arg3) = (0) to (8) {
        affine.parallel (%arg4) = (0) to (8) {
          affine.for %arg5 = 0 to 16 {
            affine.for %arg6 = 0 to 16 {
              %12 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 8)>(%arg3, %arg6)
              %13 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 8)>(%arg4, %arg5)
              %14 = loom.alloc [32, 32] on @L1 : memref<32x32xf32>
              %15 = loom.init_tensor %14[32, 32] : memref<32x32xf32> -> tensor<32x32xf32>
              %16 = linalg.fill ins(%cst : f32) outs(%15 : tensor<32x32xf32>) -> tensor<32x32xf32>
              %17 = affine.for %arg7 = 0 to 2 iter_args(%arg8 = %16) -> (tensor<32x32xf32>) {
                %21 = arith.muli %12, %c32 : index
                %22 = arith.muli %arg7, %c256 : index
                %23 = affine.apply affine_map<(d0, d1) -> (d0 * 512 + d1)>(%21, %22)
                %reinterpret_cast_0 = memref.reinterpret_cast %arg0 to offset: [%23], sizes: [32, 256], strides: [512, 1] : memref<4096x512xf32> to memref<32x256xf32, strided<[512, 1], offset: ?>>
                %24 = loom.alloc [32, 256] on @L1 : memref<32x256xf32>
                %25 = loom.copy_to_tensor %reinterpret_cast_0, %24 on @L1, interconnect : [@horizontal_links], broadcast : [1, 8] : memref<32x256xf32, strided<[512, 1], offset: ?>>, memref<32x256xf32> -> tensor<32x256xf32>
                %26 = arith.muli %13, %c32 : index
                %27 = affine.apply affine_map<(d0, d1) -> (d0 * 4096 + d1)>(%22, %26)
                %reinterpret_cast_1 = memref.reinterpret_cast %arg1 to offset: [%27], sizes: [256, 32], strides: [4096, 1] : memref<512x4096xf32> to memref<256x32xf32, strided<[4096, 1], offset: ?>>
                %28 = loom.alloc [256, 32] on @L1 : memref<256x32xf32>
                %29 = loom.copy_to_tensor %reinterpret_cast_1, %28 on @L1, interconnect : [@vertical_links], broadcast : [8, 1] : memref<256x32xf32, strided<[4096, 1], offset: ?>>, memref<256x32xf32> -> tensor<256x32xf32>
                %30 = linalg.matmul ins(%25, %29 : tensor<32x256xf32>, tensor<256x32xf32>) outs(%arg8 : tensor<32x32xf32>) -> tensor<32x32xf32>
                affine.yield %30 : tensor<32x32xf32>
              }
              %18 = arith.muli %12, %c32 : index
              %19 = arith.muli %13, %c32 : index
              %20 = affine.apply affine_map<(d0, d1) -> (d0 * 4096 + d1)>(%18, %19)
              %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%20], sizes: [32, 32], strides: [4096, 1] : memref<4096x4096xf32> to memref<32x32xf32, strided<[4096, 1], offset: ?>>
              loom.copy_from_tensor %17, %reinterpret_cast on @DRAM : tensor<32x32xf32>, memref<32x32xf32, strided<[4096, 1], offset: ?>>
            } {loom.iter_type = #loom.iter_type<temporal>}
          } {loom.iter_type = #loom.iter_type<temporal>}
        } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @x}
      } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @y}
      return
    }
  }
  module attributes {loom.pass_name = "Materialize"} {
    func.func @matmul__d0i1_d1i1__f01__d_d__BK64__BM64__BN64(%arg0: memref<4096x512xf32>, %arg1: memref<512x4096xf32>, %arg2: memref<4096x4096xf32>) {
      %cst = arith.constant 0.000000e+00 : f32
      %c64 = arith.constant 64 : index
      affine.parallel (%arg3) = (0) to (8) {
        affine.parallel (%arg4) = (0) to (8) {
          affine.for %arg5 = 0 to 64 {
            affine.for %arg6 = 0 to 1 {
              %12 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 8 + d1 + d2 * 64)>(%arg3, %arg4, %arg6)
              %13 = loom.alloc [64, 64] on @L1 : memref<64x64xf32>
              %14 = loom.init_tensor %13[64, 64] : memref<64x64xf32> -> tensor<64x64xf32>
              %15 = linalg.fill ins(%cst : f32) outs(%14 : tensor<64x64xf32>) -> tensor<64x64xf32>
              %16 = affine.for %arg7 = 0 to 8 iter_args(%arg8 = %15) -> (tensor<64x64xf32>) {
                %20 = arith.muli %arg5, %c64 : index
                %21 = arith.muli %arg7, %c64 : index
                %22 = affine.apply affine_map<(d0, d1) -> (d0 * 512 + d1)>(%20, %21)
                %reinterpret_cast_0 = memref.reinterpret_cast %arg0 to offset: [%22], sizes: [64, 64], strides: [512, 1] : memref<4096x512xf32> to memref<64x64xf32, strided<[512, 1], offset: ?>>
                %23 = loom.alloc [64, 64] on @L1 : memref<64x64xf32>
                %24 = loom.copy_to_tensor %reinterpret_cast_0, %23 on @L1, interconnect : [], broadcast : [1, 1] : memref<64x64xf32, strided<[512, 1], offset: ?>>, memref<64x64xf32> -> tensor<64x64xf32>
                %25 = arith.muli %12, %c64 : index
                %26 = affine.apply affine_map<(d0, d1) -> (d0 * 4096 + d1)>(%21, %25)
                %reinterpret_cast_1 = memref.reinterpret_cast %arg1 to offset: [%26], sizes: [64, 64], strides: [4096, 1] : memref<512x4096xf32> to memref<64x64xf32, strided<[4096, 1], offset: ?>>
                %27 = loom.alloc [64, 64] on @L1 : memref<64x64xf32>
                %28 = loom.copy_to_tensor %reinterpret_cast_1, %27 on @L1, interconnect : [], broadcast : [1, 1] : memref<64x64xf32, strided<[4096, 1], offset: ?>>, memref<64x64xf32> -> tensor<64x64xf32>
                %29 = linalg.matmul ins(%24, %28 : tensor<64x64xf32>, tensor<64x64xf32>) outs(%arg8 : tensor<64x64xf32>) -> tensor<64x64xf32>
                affine.yield %29 : tensor<64x64xf32>
              }
              %17 = arith.muli %arg5, %c64 : index
              %18 = arith.muli %12, %c64 : index
              %19 = affine.apply affine_map<(d0, d1) -> (d0 * 4096 + d1)>(%17, %18)
              %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%19], sizes: [64, 64], strides: [4096, 1] : memref<4096x4096xf32> to memref<64x64xf32, strided<[4096, 1], offset: ?>>
              loom.copy_from_tensor %16, %reinterpret_cast on @DRAM : tensor<64x64xf32>, memref<64x64xf32, strided<[4096, 1], offset: ?>>
            } {loom.iter_type = #loom.iter_type<temporal>}
          } {loom.iter_type = #loom.iter_type<temporal>}
        } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @y}
      } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @x}
      return
    }
    func.func @matmul__d0i1_d1i1__f01__d_d__BK256__BM32__BN32(%arg0: memref<4096x512xf32>, %arg1: memref<512x4096xf32>, %arg2: memref<4096x4096xf32>) {
      %cst = arith.constant 0.000000e+00 : f32
      %c32 = arith.constant 32 : index
      %c256 = arith.constant 256 : index
      affine.parallel (%arg3) = (0) to (8) {
        affine.parallel (%arg4) = (0) to (8) {
          affine.for %arg5 = 0 to 128 {
            affine.for %arg6 = 0 to 2 {
              %12 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 8 + d1 + d2 * 64)>(%arg3, %arg4, %arg6)
              %13 = loom.alloc [32, 32] on @L1 : memref<32x32xf32>
              %14 = loom.init_tensor %13[32, 32] : memref<32x32xf32> -> tensor<32x32xf32>
              %15 = linalg.fill ins(%cst : f32) outs(%14 : tensor<32x32xf32>) -> tensor<32x32xf32>
              %16 = affine.for %arg7 = 0 to 2 iter_args(%arg8 = %15) -> (tensor<32x32xf32>) {
                %20 = arith.muli %arg5, %c32 : index
                %21 = arith.muli %arg7, %c256 : index
                %22 = affine.apply affine_map<(d0, d1) -> (d0 * 512 + d1)>(%20, %21)
                %reinterpret_cast_0 = memref.reinterpret_cast %arg0 to offset: [%22], sizes: [32, 256], strides: [512, 1] : memref<4096x512xf32> to memref<32x256xf32, strided<[512, 1], offset: ?>>
                %23 = loom.alloc [32, 256] on @L1 : memref<32x256xf32>
                %24 = loom.copy_to_tensor %reinterpret_cast_0, %23 on @L1, interconnect : [], broadcast : [1, 1] : memref<32x256xf32, strided<[512, 1], offset: ?>>, memref<32x256xf32> -> tensor<32x256xf32>
                %25 = arith.muli %12, %c32 : index
                %26 = affine.apply affine_map<(d0, d1) -> (d0 * 4096 + d1)>(%21, %25)
                %reinterpret_cast_1 = memref.reinterpret_cast %arg1 to offset: [%26], sizes: [256, 32], strides: [4096, 1] : memref<512x4096xf32> to memref<256x32xf32, strided<[4096, 1], offset: ?>>
                %27 = loom.alloc [256, 32] on @L1 : memref<256x32xf32>
                %28 = loom.copy_to_tensor %reinterpret_cast_1, %27 on @L1, interconnect : [], broadcast : [1, 1] : memref<256x32xf32, strided<[4096, 1], offset: ?>>, memref<256x32xf32> -> tensor<256x32xf32>
                %29 = linalg.matmul ins(%24, %28 : tensor<32x256xf32>, tensor<256x32xf32>) outs(%arg8 : tensor<32x32xf32>) -> tensor<32x32xf32>
                affine.yield %29 : tensor<32x32xf32>
              }
              %17 = arith.muli %arg5, %c32 : index
              %18 = arith.muli %12, %c32 : index
              %19 = affine.apply affine_map<(d0, d1) -> (d0 * 4096 + d1)>(%17, %18)
              %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%19], sizes: [32, 32], strides: [4096, 1] : memref<4096x4096xf32> to memref<32x32xf32, strided<[4096, 1], offset: ?>>
              loom.copy_from_tensor %16, %reinterpret_cast on @DRAM : tensor<32x32xf32>, memref<32x32xf32, strided<[4096, 1], offset: ?>>
            } {loom.iter_type = #loom.iter_type<temporal>}
          } {loom.iter_type = #loom.iter_type<temporal>}
        } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @y}
      } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @x}
      return
    }
  }
  module attributes {loom.pass_name = "Materialize"} {
    func.func @matmul__d0i1_d1i1__f01__a_d__BK64__BM64__BN64(%arg0: memref<4096x512xf32>, %arg1: memref<512x4096xf32>, %arg2: memref<4096x4096xf32>) {
      %cst = arith.constant 0.000000e+00 : f32
      %c64 = arith.constant 64 : index
      affine.parallel (%arg3) = (0) to (8) {
        affine.parallel (%arg4) = (0) to (8) {
          affine.for %arg5 = 0 to 64 {
            affine.for %arg6 = 0 to 1 {
              %12 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 8 + d1 + d2 * 64)>(%arg3, %arg4, %arg6)
              %13 = loom.alloc [64, 64] on @L1 : memref<64x64xf32>
              %14 = loom.init_tensor %13[64, 64] : memref<64x64xf32> -> tensor<64x64xf32>
              %15 = linalg.fill ins(%cst : f32) outs(%14 : tensor<64x64xf32>) -> tensor<64x64xf32>
              %16 = affine.for %arg7 = 0 to 8 iter_args(%arg8 = %15) -> (tensor<64x64xf32>) {
                %20 = arith.muli %arg5, %c64 : index
                %21 = arith.muli %arg7, %c64 : index
                %22 = affine.apply affine_map<(d0, d1) -> (d0 * 512 + d1)>(%20, %21)
                %reinterpret_cast_0 = memref.reinterpret_cast %arg0 to offset: [%22], sizes: [64, 64], strides: [512, 1] : memref<4096x512xf32> to memref<64x64xf32, strided<[512, 1], offset: ?>>
                %23 = loom.alloc [64, 64] on @L1 : memref<64x64xf32>
                %24 = loom.copy_to_tensor %reinterpret_cast_0, %23 on @L1, interconnect : [@horizontal_links, @vertical_links], broadcast : [8, 8] : memref<64x64xf32, strided<[512, 1], offset: ?>>, memref<64x64xf32> -> tensor<64x64xf32>
                %25 = arith.muli %12, %c64 : index
                %26 = affine.apply affine_map<(d0, d1) -> (d0 * 4096 + d1)>(%21, %25)
                %reinterpret_cast_1 = memref.reinterpret_cast %arg1 to offset: [%26], sizes: [64, 64], strides: [4096, 1] : memref<512x4096xf32> to memref<64x64xf32, strided<[4096, 1], offset: ?>>
                %27 = loom.alloc [64, 64] on @L1 : memref<64x64xf32>
                %28 = loom.copy_to_tensor %reinterpret_cast_1, %27 on @L1, interconnect : [], broadcast : [1, 1] : memref<64x64xf32, strided<[4096, 1], offset: ?>>, memref<64x64xf32> -> tensor<64x64xf32>
                %29 = linalg.matmul ins(%24, %28 : tensor<64x64xf32>, tensor<64x64xf32>) outs(%arg8 : tensor<64x64xf32>) -> tensor<64x64xf32>
                affine.yield %29 : tensor<64x64xf32>
              }
              %17 = arith.muli %arg5, %c64 : index
              %18 = arith.muli %12, %c64 : index
              %19 = affine.apply affine_map<(d0, d1) -> (d0 * 4096 + d1)>(%17, %18)
              %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%19], sizes: [64, 64], strides: [4096, 1] : memref<4096x4096xf32> to memref<64x64xf32, strided<[4096, 1], offset: ?>>
              loom.copy_from_tensor %16, %reinterpret_cast on @DRAM : tensor<64x64xf32>, memref<64x64xf32, strided<[4096, 1], offset: ?>>
            } {loom.iter_type = #loom.iter_type<temporal>}
          } {loom.iter_type = #loom.iter_type<temporal>}
        } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @y}
      } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @x}
      return
    }
    func.func @matmul__d0i1_d1i1__f01__a_d__BK256__BM32__BN32(%arg0: memref<4096x512xf32>, %arg1: memref<512x4096xf32>, %arg2: memref<4096x4096xf32>) {
      %cst = arith.constant 0.000000e+00 : f32
      %c32 = arith.constant 32 : index
      %c256 = arith.constant 256 : index
      affine.parallel (%arg3) = (0) to (8) {
        affine.parallel (%arg4) = (0) to (8) {
          affine.for %arg5 = 0 to 128 {
            affine.for %arg6 = 0 to 2 {
              %12 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 8 + d1 + d2 * 64)>(%arg3, %arg4, %arg6)
              %13 = loom.alloc [32, 32] on @L1 : memref<32x32xf32>
              %14 = loom.init_tensor %13[32, 32] : memref<32x32xf32> -> tensor<32x32xf32>
              %15 = linalg.fill ins(%cst : f32) outs(%14 : tensor<32x32xf32>) -> tensor<32x32xf32>
              %16 = affine.for %arg7 = 0 to 2 iter_args(%arg8 = %15) -> (tensor<32x32xf32>) {
                %20 = arith.muli %arg5, %c32 : index
                %21 = arith.muli %arg7, %c256 : index
                %22 = affine.apply affine_map<(d0, d1) -> (d0 * 512 + d1)>(%20, %21)
                %reinterpret_cast_0 = memref.reinterpret_cast %arg0 to offset: [%22], sizes: [32, 256], strides: [512, 1] : memref<4096x512xf32> to memref<32x256xf32, strided<[512, 1], offset: ?>>
                %23 = loom.alloc [32, 256] on @L1 : memref<32x256xf32>
                %24 = loom.copy_to_tensor %reinterpret_cast_0, %23 on @L1, interconnect : [@horizontal_links, @vertical_links], broadcast : [8, 8] : memref<32x256xf32, strided<[512, 1], offset: ?>>, memref<32x256xf32> -> tensor<32x256xf32>
                %25 = arith.muli %12, %c32 : index
                %26 = affine.apply affine_map<(d0, d1) -> (d0 * 4096 + d1)>(%21, %25)
                %reinterpret_cast_1 = memref.reinterpret_cast %arg1 to offset: [%26], sizes: [256, 32], strides: [4096, 1] : memref<512x4096xf32> to memref<256x32xf32, strided<[4096, 1], offset: ?>>
                %27 = loom.alloc [256, 32] on @L1 : memref<256x32xf32>
                %28 = loom.copy_to_tensor %reinterpret_cast_1, %27 on @L1, interconnect : [], broadcast : [1, 1] : memref<256x32xf32, strided<[4096, 1], offset: ?>>, memref<256x32xf32> -> tensor<256x32xf32>
                %29 = linalg.matmul ins(%24, %28 : tensor<32x256xf32>, tensor<256x32xf32>) outs(%arg8 : tensor<32x32xf32>) -> tensor<32x32xf32>
                affine.yield %29 : tensor<32x32xf32>
              }
              %17 = arith.muli %arg5, %c32 : index
              %18 = arith.muli %12, %c32 : index
              %19 = affine.apply affine_map<(d0, d1) -> (d0 * 4096 + d1)>(%17, %18)
              %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%19], sizes: [32, 32], strides: [4096, 1] : memref<4096x4096xf32> to memref<32x32xf32, strided<[4096, 1], offset: ?>>
              loom.copy_from_tensor %16, %reinterpret_cast on @DRAM : tensor<32x32xf32>, memref<32x32xf32, strided<[4096, 1], offset: ?>>
            } {loom.iter_type = #loom.iter_type<temporal>}
          } {loom.iter_type = #loom.iter_type<temporal>}
        } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @y}
      } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @x}
      return
    }
  }
  module attributes {loom.pass_name = "Materialize"} {
    func.func @matmul__d0i1_d1i1__f01__h_d__BK64__BM64__BN64(%arg0: memref<4096x512xf32>, %arg1: memref<512x4096xf32>, %arg2: memref<4096x4096xf32>) {
      %cst = arith.constant 0.000000e+00 : f32
      %c64 = arith.constant 64 : index
      affine.parallel (%arg3) = (0) to (8) {
        affine.parallel (%arg4) = (0) to (8) {
          affine.for %arg5 = 0 to 64 {
            affine.for %arg6 = 0 to 1 {
              %12 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 8 + d1 + d2 * 64)>(%arg3, %arg4, %arg6)
              %13 = loom.alloc [64, 64] on @L1 : memref<64x64xf32>
              %14 = loom.init_tensor %13[64, 64] : memref<64x64xf32> -> tensor<64x64xf32>
              %15 = linalg.fill ins(%cst : f32) outs(%14 : tensor<64x64xf32>) -> tensor<64x64xf32>
              %16 = affine.for %arg7 = 0 to 8 iter_args(%arg8 = %15) -> (tensor<64x64xf32>) {
                %20 = arith.muli %arg5, %c64 : index
                %21 = arith.muli %arg7, %c64 : index
                %22 = affine.apply affine_map<(d0, d1) -> (d0 * 512 + d1)>(%20, %21)
                %reinterpret_cast_0 = memref.reinterpret_cast %arg0 to offset: [%22], sizes: [64, 64], strides: [512, 1] : memref<4096x512xf32> to memref<64x64xf32, strided<[512, 1], offset: ?>>
                %23 = loom.alloc [64, 64] on @L1 : memref<64x64xf32>
                %24 = loom.copy_to_tensor %reinterpret_cast_0, %23 on @L1, interconnect : [@horizontal_links], broadcast : [1, 8] : memref<64x64xf32, strided<[512, 1], offset: ?>>, memref<64x64xf32> -> tensor<64x64xf32>
                %25 = arith.muli %12, %c64 : index
                %26 = affine.apply affine_map<(d0, d1) -> (d0 * 4096 + d1)>(%21, %25)
                %reinterpret_cast_1 = memref.reinterpret_cast %arg1 to offset: [%26], sizes: [64, 64], strides: [4096, 1] : memref<512x4096xf32> to memref<64x64xf32, strided<[4096, 1], offset: ?>>
                %27 = loom.alloc [64, 64] on @L1 : memref<64x64xf32>
                %28 = loom.copy_to_tensor %reinterpret_cast_1, %27 on @L1, interconnect : [], broadcast : [1, 1] : memref<64x64xf32, strided<[4096, 1], offset: ?>>, memref<64x64xf32> -> tensor<64x64xf32>
                %29 = linalg.matmul ins(%24, %28 : tensor<64x64xf32>, tensor<64x64xf32>) outs(%arg8 : tensor<64x64xf32>) -> tensor<64x64xf32>
                affine.yield %29 : tensor<64x64xf32>
              }
              %17 = arith.muli %arg5, %c64 : index
              %18 = arith.muli %12, %c64 : index
              %19 = affine.apply affine_map<(d0, d1) -> (d0 * 4096 + d1)>(%17, %18)
              %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%19], sizes: [64, 64], strides: [4096, 1] : memref<4096x4096xf32> to memref<64x64xf32, strided<[4096, 1], offset: ?>>
              loom.copy_from_tensor %16, %reinterpret_cast on @DRAM : tensor<64x64xf32>, memref<64x64xf32, strided<[4096, 1], offset: ?>>
            } {loom.iter_type = #loom.iter_type<temporal>}
          } {loom.iter_type = #loom.iter_type<temporal>}
        } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @y}
      } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @x}
      return
    }
    func.func @matmul__d0i1_d1i1__f01__h_d__BK256__BM32__BN32(%arg0: memref<4096x512xf32>, %arg1: memref<512x4096xf32>, %arg2: memref<4096x4096xf32>) {
      %cst = arith.constant 0.000000e+00 : f32
      %c32 = arith.constant 32 : index
      %c256 = arith.constant 256 : index
      affine.parallel (%arg3) = (0) to (8) {
        affine.parallel (%arg4) = (0) to (8) {
          affine.for %arg5 = 0 to 128 {
            affine.for %arg6 = 0 to 2 {
              %12 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 8 + d1 + d2 * 64)>(%arg3, %arg4, %arg6)
              %13 = loom.alloc [32, 32] on @L1 : memref<32x32xf32>
              %14 = loom.init_tensor %13[32, 32] : memref<32x32xf32> -> tensor<32x32xf32>
              %15 = linalg.fill ins(%cst : f32) outs(%14 : tensor<32x32xf32>) -> tensor<32x32xf32>
              %16 = affine.for %arg7 = 0 to 2 iter_args(%arg8 = %15) -> (tensor<32x32xf32>) {
                %20 = arith.muli %arg5, %c32 : index
                %21 = arith.muli %arg7, %c256 : index
                %22 = affine.apply affine_map<(d0, d1) -> (d0 * 512 + d1)>(%20, %21)
                %reinterpret_cast_0 = memref.reinterpret_cast %arg0 to offset: [%22], sizes: [32, 256], strides: [512, 1] : memref<4096x512xf32> to memref<32x256xf32, strided<[512, 1], offset: ?>>
                %23 = loom.alloc [32, 256] on @L1 : memref<32x256xf32>
                %24 = loom.copy_to_tensor %reinterpret_cast_0, %23 on @L1, interconnect : [@horizontal_links], broadcast : [1, 8] : memref<32x256xf32, strided<[512, 1], offset: ?>>, memref<32x256xf32> -> tensor<32x256xf32>
                %25 = arith.muli %12, %c32 : index
                %26 = affine.apply affine_map<(d0, d1) -> (d0 * 4096 + d1)>(%21, %25)
                %reinterpret_cast_1 = memref.reinterpret_cast %arg1 to offset: [%26], sizes: [256, 32], strides: [4096, 1] : memref<512x4096xf32> to memref<256x32xf32, strided<[4096, 1], offset: ?>>
                %27 = loom.alloc [256, 32] on @L1 : memref<256x32xf32>
                %28 = loom.copy_to_tensor %reinterpret_cast_1, %27 on @L1, interconnect : [], broadcast : [1, 1] : memref<256x32xf32, strided<[4096, 1], offset: ?>>, memref<256x32xf32> -> tensor<256x32xf32>
                %29 = linalg.matmul ins(%24, %28 : tensor<32x256xf32>, tensor<256x32xf32>) outs(%arg8 : tensor<32x32xf32>) -> tensor<32x32xf32>
                affine.yield %29 : tensor<32x32xf32>
              }
              %17 = arith.muli %arg5, %c32 : index
              %18 = arith.muli %12, %c32 : index
              %19 = affine.apply affine_map<(d0, d1) -> (d0 * 4096 + d1)>(%17, %18)
              %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%19], sizes: [32, 32], strides: [4096, 1] : memref<4096x4096xf32> to memref<32x32xf32, strided<[4096, 1], offset: ?>>
              loom.copy_from_tensor %16, %reinterpret_cast on @DRAM : tensor<32x32xf32>, memref<32x32xf32, strided<[4096, 1], offset: ?>>
            } {loom.iter_type = #loom.iter_type<temporal>}
          } {loom.iter_type = #loom.iter_type<temporal>}
        } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @y}
      } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @x}
      return
    }
  }
  module attributes {loom.pass_name = "Materialize"} {
    func.func @matmul__d0i1_d1i1__f01__v_d__BK64__BM64__BN64(%arg0: memref<4096x512xf32>, %arg1: memref<512x4096xf32>, %arg2: memref<4096x4096xf32>) {
      %cst = arith.constant 0.000000e+00 : f32
      %c64 = arith.constant 64 : index
      affine.parallel (%arg3) = (0) to (8) {
        affine.parallel (%arg4) = (0) to (8) {
          affine.for %arg5 = 0 to 64 {
            affine.for %arg6 = 0 to 1 {
              %12 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 8 + d1 + d2 * 64)>(%arg3, %arg4, %arg6)
              %13 = loom.alloc [64, 64] on @L1 : memref<64x64xf32>
              %14 = loom.init_tensor %13[64, 64] : memref<64x64xf32> -> tensor<64x64xf32>
              %15 = linalg.fill ins(%cst : f32) outs(%14 : tensor<64x64xf32>) -> tensor<64x64xf32>
              %16 = affine.for %arg7 = 0 to 8 iter_args(%arg8 = %15) -> (tensor<64x64xf32>) {
                %20 = arith.muli %arg5, %c64 : index
                %21 = arith.muli %arg7, %c64 : index
                %22 = affine.apply affine_map<(d0, d1) -> (d0 * 512 + d1)>(%20, %21)
                %reinterpret_cast_0 = memref.reinterpret_cast %arg0 to offset: [%22], sizes: [64, 64], strides: [512, 1] : memref<4096x512xf32> to memref<64x64xf32, strided<[512, 1], offset: ?>>
                %23 = loom.alloc [64, 64] on @L1 : memref<64x64xf32>
                %24 = loom.copy_to_tensor %reinterpret_cast_0, %23 on @L1, interconnect : [@vertical_links], broadcast : [8, 1] : memref<64x64xf32, strided<[512, 1], offset: ?>>, memref<64x64xf32> -> tensor<64x64xf32>
                %25 = arith.muli %12, %c64 : index
                %26 = affine.apply affine_map<(d0, d1) -> (d0 * 4096 + d1)>(%21, %25)
                %reinterpret_cast_1 = memref.reinterpret_cast %arg1 to offset: [%26], sizes: [64, 64], strides: [4096, 1] : memref<512x4096xf32> to memref<64x64xf32, strided<[4096, 1], offset: ?>>
                %27 = loom.alloc [64, 64] on @L1 : memref<64x64xf32>
                %28 = loom.copy_to_tensor %reinterpret_cast_1, %27 on @L1, interconnect : [], broadcast : [1, 1] : memref<64x64xf32, strided<[4096, 1], offset: ?>>, memref<64x64xf32> -> tensor<64x64xf32>
                %29 = linalg.matmul ins(%24, %28 : tensor<64x64xf32>, tensor<64x64xf32>) outs(%arg8 : tensor<64x64xf32>) -> tensor<64x64xf32>
                affine.yield %29 : tensor<64x64xf32>
              }
              %17 = arith.muli %arg5, %c64 : index
              %18 = arith.muli %12, %c64 : index
              %19 = affine.apply affine_map<(d0, d1) -> (d0 * 4096 + d1)>(%17, %18)
              %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%19], sizes: [64, 64], strides: [4096, 1] : memref<4096x4096xf32> to memref<64x64xf32, strided<[4096, 1], offset: ?>>
              loom.copy_from_tensor %16, %reinterpret_cast on @DRAM : tensor<64x64xf32>, memref<64x64xf32, strided<[4096, 1], offset: ?>>
            } {loom.iter_type = #loom.iter_type<temporal>}
          } {loom.iter_type = #loom.iter_type<temporal>}
        } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @y}
      } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @x}
      return
    }
    func.func @matmul__d0i1_d1i1__f01__v_d__BK256__BM32__BN32(%arg0: memref<4096x512xf32>, %arg1: memref<512x4096xf32>, %arg2: memref<4096x4096xf32>) {
      %cst = arith.constant 0.000000e+00 : f32
      %c32 = arith.constant 32 : index
      %c256 = arith.constant 256 : index
      affine.parallel (%arg3) = (0) to (8) {
        affine.parallel (%arg4) = (0) to (8) {
          affine.for %arg5 = 0 to 128 {
            affine.for %arg6 = 0 to 2 {
              %12 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 8 + d1 + d2 * 64)>(%arg3, %arg4, %arg6)
              %13 = loom.alloc [32, 32] on @L1 : memref<32x32xf32>
              %14 = loom.init_tensor %13[32, 32] : memref<32x32xf32> -> tensor<32x32xf32>
              %15 = linalg.fill ins(%cst : f32) outs(%14 : tensor<32x32xf32>) -> tensor<32x32xf32>
              %16 = affine.for %arg7 = 0 to 2 iter_args(%arg8 = %15) -> (tensor<32x32xf32>) {
                %20 = arith.muli %arg5, %c32 : index
                %21 = arith.muli %arg7, %c256 : index
                %22 = affine.apply affine_map<(d0, d1) -> (d0 * 512 + d1)>(%20, %21)
                %reinterpret_cast_0 = memref.reinterpret_cast %arg0 to offset: [%22], sizes: [32, 256], strides: [512, 1] : memref<4096x512xf32> to memref<32x256xf32, strided<[512, 1], offset: ?>>
                %23 = loom.alloc [32, 256] on @L1 : memref<32x256xf32>
                %24 = loom.copy_to_tensor %reinterpret_cast_0, %23 on @L1, interconnect : [@vertical_links], broadcast : [8, 1] : memref<32x256xf32, strided<[512, 1], offset: ?>>, memref<32x256xf32> -> tensor<32x256xf32>
                %25 = arith.muli %12, %c32 : index
                %26 = affine.apply affine_map<(d0, d1) -> (d0 * 4096 + d1)>(%21, %25)
                %reinterpret_cast_1 = memref.reinterpret_cast %arg1 to offset: [%26], sizes: [256, 32], strides: [4096, 1] : memref<512x4096xf32> to memref<256x32xf32, strided<[4096, 1], offset: ?>>
                %27 = loom.alloc [256, 32] on @L1 : memref<256x32xf32>
                %28 = loom.copy_to_tensor %reinterpret_cast_1, %27 on @L1, interconnect : [], broadcast : [1, 1] : memref<256x32xf32, strided<[4096, 1], offset: ?>>, memref<256x32xf32> -> tensor<256x32xf32>
                %29 = linalg.matmul ins(%24, %28 : tensor<32x256xf32>, tensor<256x32xf32>) outs(%arg8 : tensor<32x32xf32>) -> tensor<32x32xf32>
                affine.yield %29 : tensor<32x32xf32>
              }
              %17 = arith.muli %arg5, %c32 : index
              %18 = arith.muli %12, %c32 : index
              %19 = affine.apply affine_map<(d0, d1) -> (d0 * 4096 + d1)>(%17, %18)
              %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%19], sizes: [32, 32], strides: [4096, 1] : memref<4096x4096xf32> to memref<32x32xf32, strided<[4096, 1], offset: ?>>
              loom.copy_from_tensor %16, %reinterpret_cast on @DRAM : tensor<32x32xf32>, memref<32x32xf32, strided<[4096, 1], offset: ?>>
            } {loom.iter_type = #loom.iter_type<temporal>}
          } {loom.iter_type = #loom.iter_type<temporal>}
        } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @y}
      } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @x}
      return
    }
  }
  module attributes {loom.pass_name = "Materialize"} {
    func.func @matmul__d0i1_d1i1__f10__d_d__BK64__BM64__BN64(%arg0: memref<4096x512xf32>, %arg1: memref<512x4096xf32>, %arg2: memref<4096x4096xf32>) {
      %cst = arith.constant 0.000000e+00 : f32
      %c64 = arith.constant 64 : index
      affine.parallel (%arg3) = (0) to (8) {
        affine.parallel (%arg4) = (0) to (8) {
          affine.for %arg5 = 0 to 1 {
            affine.for %arg6 = 0 to 64 {
              %12 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 8 + d1 + d2 * 64)>(%arg3, %arg4, %arg5)
              %13 = loom.alloc [64, 64] on @L1 : memref<64x64xf32>
              %14 = loom.init_tensor %13[64, 64] : memref<64x64xf32> -> tensor<64x64xf32>
              %15 = linalg.fill ins(%cst : f32) outs(%14 : tensor<64x64xf32>) -> tensor<64x64xf32>
              %16 = affine.for %arg7 = 0 to 8 iter_args(%arg8 = %15) -> (tensor<64x64xf32>) {
                %20 = arith.muli %arg6, %c64 : index
                %21 = arith.muli %arg7, %c64 : index
                %22 = affine.apply affine_map<(d0, d1) -> (d0 * 512 + d1)>(%20, %21)
                %reinterpret_cast_0 = memref.reinterpret_cast %arg0 to offset: [%22], sizes: [64, 64], strides: [512, 1] : memref<4096x512xf32> to memref<64x64xf32, strided<[512, 1], offset: ?>>
                %23 = loom.alloc [64, 64] on @L1 : memref<64x64xf32>
                %24 = loom.copy_to_tensor %reinterpret_cast_0, %23 on @L1, interconnect : [], broadcast : [1, 1] : memref<64x64xf32, strided<[512, 1], offset: ?>>, memref<64x64xf32> -> tensor<64x64xf32>
                %25 = arith.muli %12, %c64 : index
                %26 = affine.apply affine_map<(d0, d1) -> (d0 * 4096 + d1)>(%21, %25)
                %reinterpret_cast_1 = memref.reinterpret_cast %arg1 to offset: [%26], sizes: [64, 64], strides: [4096, 1] : memref<512x4096xf32> to memref<64x64xf32, strided<[4096, 1], offset: ?>>
                %27 = loom.alloc [64, 64] on @L1 : memref<64x64xf32>
                %28 = loom.copy_to_tensor %reinterpret_cast_1, %27 on @L1, interconnect : [], broadcast : [1, 1] : memref<64x64xf32, strided<[4096, 1], offset: ?>>, memref<64x64xf32> -> tensor<64x64xf32>
                %29 = linalg.matmul ins(%24, %28 : tensor<64x64xf32>, tensor<64x64xf32>) outs(%arg8 : tensor<64x64xf32>) -> tensor<64x64xf32>
                affine.yield %29 : tensor<64x64xf32>
              }
              %17 = arith.muli %arg6, %c64 : index
              %18 = arith.muli %12, %c64 : index
              %19 = affine.apply affine_map<(d0, d1) -> (d0 * 4096 + d1)>(%17, %18)
              %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%19], sizes: [64, 64], strides: [4096, 1] : memref<4096x4096xf32> to memref<64x64xf32, strided<[4096, 1], offset: ?>>
              loom.copy_from_tensor %16, %reinterpret_cast on @DRAM : tensor<64x64xf32>, memref<64x64xf32, strided<[4096, 1], offset: ?>>
            } {loom.iter_type = #loom.iter_type<temporal>}
          } {loom.iter_type = #loom.iter_type<temporal>}
        } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @y}
      } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @x}
      return
    }
    func.func @matmul__d0i1_d1i1__f10__d_d__BK256__BM32__BN32(%arg0: memref<4096x512xf32>, %arg1: memref<512x4096xf32>, %arg2: memref<4096x4096xf32>) {
      %cst = arith.constant 0.000000e+00 : f32
      %c32 = arith.constant 32 : index
      %c256 = arith.constant 256 : index
      affine.parallel (%arg3) = (0) to (8) {
        affine.parallel (%arg4) = (0) to (8) {
          affine.for %arg5 = 0 to 2 {
            affine.for %arg6 = 0 to 128 {
              %12 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 8 + d1 + d2 * 64)>(%arg3, %arg4, %arg5)
              %13 = loom.alloc [32, 32] on @L1 : memref<32x32xf32>
              %14 = loom.init_tensor %13[32, 32] : memref<32x32xf32> -> tensor<32x32xf32>
              %15 = linalg.fill ins(%cst : f32) outs(%14 : tensor<32x32xf32>) -> tensor<32x32xf32>
              %16 = affine.for %arg7 = 0 to 2 iter_args(%arg8 = %15) -> (tensor<32x32xf32>) {
                %20 = arith.muli %arg6, %c32 : index
                %21 = arith.muli %arg7, %c256 : index
                %22 = affine.apply affine_map<(d0, d1) -> (d0 * 512 + d1)>(%20, %21)
                %reinterpret_cast_0 = memref.reinterpret_cast %arg0 to offset: [%22], sizes: [32, 256], strides: [512, 1] : memref<4096x512xf32> to memref<32x256xf32, strided<[512, 1], offset: ?>>
                %23 = loom.alloc [32, 256] on @L1 : memref<32x256xf32>
                %24 = loom.copy_to_tensor %reinterpret_cast_0, %23 on @L1, interconnect : [], broadcast : [1, 1] : memref<32x256xf32, strided<[512, 1], offset: ?>>, memref<32x256xf32> -> tensor<32x256xf32>
                %25 = arith.muli %12, %c32 : index
                %26 = affine.apply affine_map<(d0, d1) -> (d0 * 4096 + d1)>(%21, %25)
                %reinterpret_cast_1 = memref.reinterpret_cast %arg1 to offset: [%26], sizes: [256, 32], strides: [4096, 1] : memref<512x4096xf32> to memref<256x32xf32, strided<[4096, 1], offset: ?>>
                %27 = loom.alloc [256, 32] on @L1 : memref<256x32xf32>
                %28 = loom.copy_to_tensor %reinterpret_cast_1, %27 on @L1, interconnect : [], broadcast : [1, 1] : memref<256x32xf32, strided<[4096, 1], offset: ?>>, memref<256x32xf32> -> tensor<256x32xf32>
                %29 = linalg.matmul ins(%24, %28 : tensor<32x256xf32>, tensor<256x32xf32>) outs(%arg8 : tensor<32x32xf32>) -> tensor<32x32xf32>
                affine.yield %29 : tensor<32x32xf32>
              }
              %17 = arith.muli %arg6, %c32 : index
              %18 = arith.muli %12, %c32 : index
              %19 = affine.apply affine_map<(d0, d1) -> (d0 * 4096 + d1)>(%17, %18)
              %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%19], sizes: [32, 32], strides: [4096, 1] : memref<4096x4096xf32> to memref<32x32xf32, strided<[4096, 1], offset: ?>>
              loom.copy_from_tensor %16, %reinterpret_cast on @DRAM : tensor<32x32xf32>, memref<32x32xf32, strided<[4096, 1], offset: ?>>
            } {loom.iter_type = #loom.iter_type<temporal>}
          } {loom.iter_type = #loom.iter_type<temporal>}
        } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @y}
      } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @x}
      return
    }
  }
  module attributes {loom.pass_name = "Materialize"} {
    func.func @matmul__d0i1_d1i1__f10__a_d__BK64__BM64__BN64(%arg0: memref<4096x512xf32>, %arg1: memref<512x4096xf32>, %arg2: memref<4096x4096xf32>) {
      %cst = arith.constant 0.000000e+00 : f32
      %c64 = arith.constant 64 : index
      affine.parallel (%arg3) = (0) to (8) {
        affine.parallel (%arg4) = (0) to (8) {
          affine.for %arg5 = 0 to 1 {
            affine.for %arg6 = 0 to 64 {
              %12 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 8 + d1 + d2 * 64)>(%arg3, %arg4, %arg5)
              %13 = loom.alloc [64, 64] on @L1 : memref<64x64xf32>
              %14 = loom.init_tensor %13[64, 64] : memref<64x64xf32> -> tensor<64x64xf32>
              %15 = linalg.fill ins(%cst : f32) outs(%14 : tensor<64x64xf32>) -> tensor<64x64xf32>
              %16 = affine.for %arg7 = 0 to 8 iter_args(%arg8 = %15) -> (tensor<64x64xf32>) {
                %20 = arith.muli %arg6, %c64 : index
                %21 = arith.muli %arg7, %c64 : index
                %22 = affine.apply affine_map<(d0, d1) -> (d0 * 512 + d1)>(%20, %21)
                %reinterpret_cast_0 = memref.reinterpret_cast %arg0 to offset: [%22], sizes: [64, 64], strides: [512, 1] : memref<4096x512xf32> to memref<64x64xf32, strided<[512, 1], offset: ?>>
                %23 = loom.alloc [64, 64] on @L1 : memref<64x64xf32>
                %24 = loom.copy_to_tensor %reinterpret_cast_0, %23 on @L1, interconnect : [@horizontal_links, @vertical_links], broadcast : [8, 8] : memref<64x64xf32, strided<[512, 1], offset: ?>>, memref<64x64xf32> -> tensor<64x64xf32>
                %25 = arith.muli %12, %c64 : index
                %26 = affine.apply affine_map<(d0, d1) -> (d0 * 4096 + d1)>(%21, %25)
                %reinterpret_cast_1 = memref.reinterpret_cast %arg1 to offset: [%26], sizes: [64, 64], strides: [4096, 1] : memref<512x4096xf32> to memref<64x64xf32, strided<[4096, 1], offset: ?>>
                %27 = loom.alloc [64, 64] on @L1 : memref<64x64xf32>
                %28 = loom.copy_to_tensor %reinterpret_cast_1, %27 on @L1, interconnect : [], broadcast : [1, 1] : memref<64x64xf32, strided<[4096, 1], offset: ?>>, memref<64x64xf32> -> tensor<64x64xf32>
                %29 = linalg.matmul ins(%24, %28 : tensor<64x64xf32>, tensor<64x64xf32>) outs(%arg8 : tensor<64x64xf32>) -> tensor<64x64xf32>
                affine.yield %29 : tensor<64x64xf32>
              }
              %17 = arith.muli %arg6, %c64 : index
              %18 = arith.muli %12, %c64 : index
              %19 = affine.apply affine_map<(d0, d1) -> (d0 * 4096 + d1)>(%17, %18)
              %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%19], sizes: [64, 64], strides: [4096, 1] : memref<4096x4096xf32> to memref<64x64xf32, strided<[4096, 1], offset: ?>>
              loom.copy_from_tensor %16, %reinterpret_cast on @DRAM : tensor<64x64xf32>, memref<64x64xf32, strided<[4096, 1], offset: ?>>
            } {loom.iter_type = #loom.iter_type<temporal>}
          } {loom.iter_type = #loom.iter_type<temporal>}
        } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @y}
      } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @x}
      return
    }
    func.func @matmul__d0i1_d1i1__f10__a_d__BK256__BM32__BN32(%arg0: memref<4096x512xf32>, %arg1: memref<512x4096xf32>, %arg2: memref<4096x4096xf32>) {
      %cst = arith.constant 0.000000e+00 : f32
      %c32 = arith.constant 32 : index
      %c256 = arith.constant 256 : index
      affine.parallel (%arg3) = (0) to (8) {
        affine.parallel (%arg4) = (0) to (8) {
          affine.for %arg5 = 0 to 2 {
            affine.for %arg6 = 0 to 128 {
              %12 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 8 + d1 + d2 * 64)>(%arg3, %arg4, %arg5)
              %13 = loom.alloc [32, 32] on @L1 : memref<32x32xf32>
              %14 = loom.init_tensor %13[32, 32] : memref<32x32xf32> -> tensor<32x32xf32>
              %15 = linalg.fill ins(%cst : f32) outs(%14 : tensor<32x32xf32>) -> tensor<32x32xf32>
              %16 = affine.for %arg7 = 0 to 2 iter_args(%arg8 = %15) -> (tensor<32x32xf32>) {
                %20 = arith.muli %arg6, %c32 : index
                %21 = arith.muli %arg7, %c256 : index
                %22 = affine.apply affine_map<(d0, d1) -> (d0 * 512 + d1)>(%20, %21)
                %reinterpret_cast_0 = memref.reinterpret_cast %arg0 to offset: [%22], sizes: [32, 256], strides: [512, 1] : memref<4096x512xf32> to memref<32x256xf32, strided<[512, 1], offset: ?>>
                %23 = loom.alloc [32, 256] on @L1 : memref<32x256xf32>
                %24 = loom.copy_to_tensor %reinterpret_cast_0, %23 on @L1, interconnect : [@horizontal_links, @vertical_links], broadcast : [8, 8] : memref<32x256xf32, strided<[512, 1], offset: ?>>, memref<32x256xf32> -> tensor<32x256xf32>
                %25 = arith.muli %12, %c32 : index
                %26 = affine.apply affine_map<(d0, d1) -> (d0 * 4096 + d1)>(%21, %25)
                %reinterpret_cast_1 = memref.reinterpret_cast %arg1 to offset: [%26], sizes: [256, 32], strides: [4096, 1] : memref<512x4096xf32> to memref<256x32xf32, strided<[4096, 1], offset: ?>>
                %27 = loom.alloc [256, 32] on @L1 : memref<256x32xf32>
                %28 = loom.copy_to_tensor %reinterpret_cast_1, %27 on @L1, interconnect : [], broadcast : [1, 1] : memref<256x32xf32, strided<[4096, 1], offset: ?>>, memref<256x32xf32> -> tensor<256x32xf32>
                %29 = linalg.matmul ins(%24, %28 : tensor<32x256xf32>, tensor<256x32xf32>) outs(%arg8 : tensor<32x32xf32>) -> tensor<32x32xf32>
                affine.yield %29 : tensor<32x32xf32>
              }
              %17 = arith.muli %arg6, %c32 : index
              %18 = arith.muli %12, %c32 : index
              %19 = affine.apply affine_map<(d0, d1) -> (d0 * 4096 + d1)>(%17, %18)
              %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%19], sizes: [32, 32], strides: [4096, 1] : memref<4096x4096xf32> to memref<32x32xf32, strided<[4096, 1], offset: ?>>
              loom.copy_from_tensor %16, %reinterpret_cast on @DRAM : tensor<32x32xf32>, memref<32x32xf32, strided<[4096, 1], offset: ?>>
            } {loom.iter_type = #loom.iter_type<temporal>}
          } {loom.iter_type = #loom.iter_type<temporal>}
        } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @y}
      } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @x}
      return
    }
  }
  module attributes {loom.pass_name = "Materialize"} {
    func.func @matmul__d0i1_d1i1__f10__h_d__BK64__BM64__BN64(%arg0: memref<4096x512xf32>, %arg1: memref<512x4096xf32>, %arg2: memref<4096x4096xf32>) {
      %cst = arith.constant 0.000000e+00 : f32
      %c64 = arith.constant 64 : index
      affine.parallel (%arg3) = (0) to (8) {
        affine.parallel (%arg4) = (0) to (8) {
          affine.for %arg5 = 0 to 1 {
            affine.for %arg6 = 0 to 64 {
              %12 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 8 + d1 + d2 * 64)>(%arg3, %arg4, %arg5)
              %13 = loom.alloc [64, 64] on @L1 : memref<64x64xf32>
              %14 = loom.init_tensor %13[64, 64] : memref<64x64xf32> -> tensor<64x64xf32>
              %15 = linalg.fill ins(%cst : f32) outs(%14 : tensor<64x64xf32>) -> tensor<64x64xf32>
              %16 = affine.for %arg7 = 0 to 8 iter_args(%arg8 = %15) -> (tensor<64x64xf32>) {
                %20 = arith.muli %arg6, %c64 : index
                %21 = arith.muli %arg7, %c64 : index
                %22 = affine.apply affine_map<(d0, d1) -> (d0 * 512 + d1)>(%20, %21)
                %reinterpret_cast_0 = memref.reinterpret_cast %arg0 to offset: [%22], sizes: [64, 64], strides: [512, 1] : memref<4096x512xf32> to memref<64x64xf32, strided<[512, 1], offset: ?>>
                %23 = loom.alloc [64, 64] on @L1 : memref<64x64xf32>
                %24 = loom.copy_to_tensor %reinterpret_cast_0, %23 on @L1, interconnect : [@horizontal_links], broadcast : [1, 8] : memref<64x64xf32, strided<[512, 1], offset: ?>>, memref<64x64xf32> -> tensor<64x64xf32>
                %25 = arith.muli %12, %c64 : index
                %26 = affine.apply affine_map<(d0, d1) -> (d0 * 4096 + d1)>(%21, %25)
                %reinterpret_cast_1 = memref.reinterpret_cast %arg1 to offset: [%26], sizes: [64, 64], strides: [4096, 1] : memref<512x4096xf32> to memref<64x64xf32, strided<[4096, 1], offset: ?>>
                %27 = loom.alloc [64, 64] on @L1 : memref<64x64xf32>
                %28 = loom.copy_to_tensor %reinterpret_cast_1, %27 on @L1, interconnect : [], broadcast : [1, 1] : memref<64x64xf32, strided<[4096, 1], offset: ?>>, memref<64x64xf32> -> tensor<64x64xf32>
                %29 = linalg.matmul ins(%24, %28 : tensor<64x64xf32>, tensor<64x64xf32>) outs(%arg8 : tensor<64x64xf32>) -> tensor<64x64xf32>
                affine.yield %29 : tensor<64x64xf32>
              }
              %17 = arith.muli %arg6, %c64 : index
              %18 = arith.muli %12, %c64 : index
              %19 = affine.apply affine_map<(d0, d1) -> (d0 * 4096 + d1)>(%17, %18)
              %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%19], sizes: [64, 64], strides: [4096, 1] : memref<4096x4096xf32> to memref<64x64xf32, strided<[4096, 1], offset: ?>>
              loom.copy_from_tensor %16, %reinterpret_cast on @DRAM : tensor<64x64xf32>, memref<64x64xf32, strided<[4096, 1], offset: ?>>
            } {loom.iter_type = #loom.iter_type<temporal>}
          } {loom.iter_type = #loom.iter_type<temporal>}
        } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @y}
      } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @x}
      return
    }
    func.func @matmul__d0i1_d1i1__f10__h_d__BK256__BM32__BN32(%arg0: memref<4096x512xf32>, %arg1: memref<512x4096xf32>, %arg2: memref<4096x4096xf32>) {
      %cst = arith.constant 0.000000e+00 : f32
      %c32 = arith.constant 32 : index
      %c256 = arith.constant 256 : index
      affine.parallel (%arg3) = (0) to (8) {
        affine.parallel (%arg4) = (0) to (8) {
          affine.for %arg5 = 0 to 2 {
            affine.for %arg6 = 0 to 128 {
              %12 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 8 + d1 + d2 * 64)>(%arg3, %arg4, %arg5)
              %13 = loom.alloc [32, 32] on @L1 : memref<32x32xf32>
              %14 = loom.init_tensor %13[32, 32] : memref<32x32xf32> -> tensor<32x32xf32>
              %15 = linalg.fill ins(%cst : f32) outs(%14 : tensor<32x32xf32>) -> tensor<32x32xf32>
              %16 = affine.for %arg7 = 0 to 2 iter_args(%arg8 = %15) -> (tensor<32x32xf32>) {
                %20 = arith.muli %arg6, %c32 : index
                %21 = arith.muli %arg7, %c256 : index
                %22 = affine.apply affine_map<(d0, d1) -> (d0 * 512 + d1)>(%20, %21)
                %reinterpret_cast_0 = memref.reinterpret_cast %arg0 to offset: [%22], sizes: [32, 256], strides: [512, 1] : memref<4096x512xf32> to memref<32x256xf32, strided<[512, 1], offset: ?>>
                %23 = loom.alloc [32, 256] on @L1 : memref<32x256xf32>
                %24 = loom.copy_to_tensor %reinterpret_cast_0, %23 on @L1, interconnect : [@horizontal_links], broadcast : [1, 8] : memref<32x256xf32, strided<[512, 1], offset: ?>>, memref<32x256xf32> -> tensor<32x256xf32>
                %25 = arith.muli %12, %c32 : index
                %26 = affine.apply affine_map<(d0, d1) -> (d0 * 4096 + d1)>(%21, %25)
                %reinterpret_cast_1 = memref.reinterpret_cast %arg1 to offset: [%26], sizes: [256, 32], strides: [4096, 1] : memref<512x4096xf32> to memref<256x32xf32, strided<[4096, 1], offset: ?>>
                %27 = loom.alloc [256, 32] on @L1 : memref<256x32xf32>
                %28 = loom.copy_to_tensor %reinterpret_cast_1, %27 on @L1, interconnect : [], broadcast : [1, 1] : memref<256x32xf32, strided<[4096, 1], offset: ?>>, memref<256x32xf32> -> tensor<256x32xf32>
                %29 = linalg.matmul ins(%24, %28 : tensor<32x256xf32>, tensor<256x32xf32>) outs(%arg8 : tensor<32x32xf32>) -> tensor<32x32xf32>
                affine.yield %29 : tensor<32x32xf32>
              }
              %17 = arith.muli %arg6, %c32 : index
              %18 = arith.muli %12, %c32 : index
              %19 = affine.apply affine_map<(d0, d1) -> (d0 * 4096 + d1)>(%17, %18)
              %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%19], sizes: [32, 32], strides: [4096, 1] : memref<4096x4096xf32> to memref<32x32xf32, strided<[4096, 1], offset: ?>>
              loom.copy_from_tensor %16, %reinterpret_cast on @DRAM : tensor<32x32xf32>, memref<32x32xf32, strided<[4096, 1], offset: ?>>
            } {loom.iter_type = #loom.iter_type<temporal>}
          } {loom.iter_type = #loom.iter_type<temporal>}
        } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @y}
      } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @x}
      return
    }
  }
  module attributes {loom.pass_name = "Materialize"} {
    func.func @matmul__d0i1_d1i1__f10__v_d__BK64__BM64__BN64(%arg0: memref<4096x512xf32>, %arg1: memref<512x4096xf32>, %arg2: memref<4096x4096xf32>) {
      %cst = arith.constant 0.000000e+00 : f32
      %c64 = arith.constant 64 : index
      affine.parallel (%arg3) = (0) to (8) {
        affine.parallel (%arg4) = (0) to (8) {
          affine.for %arg5 = 0 to 1 {
            affine.for %arg6 = 0 to 64 {
              %12 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 8 + d1 + d2 * 64)>(%arg3, %arg4, %arg5)
              %13 = loom.alloc [64, 64] on @L1 : memref<64x64xf32>
              %14 = loom.init_tensor %13[64, 64] : memref<64x64xf32> -> tensor<64x64xf32>
              %15 = linalg.fill ins(%cst : f32) outs(%14 : tensor<64x64xf32>) -> tensor<64x64xf32>
              %16 = affine.for %arg7 = 0 to 8 iter_args(%arg8 = %15) -> (tensor<64x64xf32>) {
                %20 = arith.muli %arg6, %c64 : index
                %21 = arith.muli %arg7, %c64 : index
                %22 = affine.apply affine_map<(d0, d1) -> (d0 * 512 + d1)>(%20, %21)
                %reinterpret_cast_0 = memref.reinterpret_cast %arg0 to offset: [%22], sizes: [64, 64], strides: [512, 1] : memref<4096x512xf32> to memref<64x64xf32, strided<[512, 1], offset: ?>>
                %23 = loom.alloc [64, 64] on @L1 : memref<64x64xf32>
                %24 = loom.copy_to_tensor %reinterpret_cast_0, %23 on @L1, interconnect : [@vertical_links], broadcast : [8, 1] : memref<64x64xf32, strided<[512, 1], offset: ?>>, memref<64x64xf32> -> tensor<64x64xf32>
                %25 = arith.muli %12, %c64 : index
                %26 = affine.apply affine_map<(d0, d1) -> (d0 * 4096 + d1)>(%21, %25)
                %reinterpret_cast_1 = memref.reinterpret_cast %arg1 to offset: [%26], sizes: [64, 64], strides: [4096, 1] : memref<512x4096xf32> to memref<64x64xf32, strided<[4096, 1], offset: ?>>
                %27 = loom.alloc [64, 64] on @L1 : memref<64x64xf32>
                %28 = loom.copy_to_tensor %reinterpret_cast_1, %27 on @L1, interconnect : [], broadcast : [1, 1] : memref<64x64xf32, strided<[4096, 1], offset: ?>>, memref<64x64xf32> -> tensor<64x64xf32>
                %29 = linalg.matmul ins(%24, %28 : tensor<64x64xf32>, tensor<64x64xf32>) outs(%arg8 : tensor<64x64xf32>) -> tensor<64x64xf32>
                affine.yield %29 : tensor<64x64xf32>
              }
              %17 = arith.muli %arg6, %c64 : index
              %18 = arith.muli %12, %c64 : index
              %19 = affine.apply affine_map<(d0, d1) -> (d0 * 4096 + d1)>(%17, %18)
              %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%19], sizes: [64, 64], strides: [4096, 1] : memref<4096x4096xf32> to memref<64x64xf32, strided<[4096, 1], offset: ?>>
              loom.copy_from_tensor %16, %reinterpret_cast on @DRAM : tensor<64x64xf32>, memref<64x64xf32, strided<[4096, 1], offset: ?>>
            } {loom.iter_type = #loom.iter_type<temporal>}
          } {loom.iter_type = #loom.iter_type<temporal>}
        } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @y}
      } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @x}
      return
    }
    func.func @matmul__d0i1_d1i1__f10__v_d__BK256__BM32__BN32(%arg0: memref<4096x512xf32>, %arg1: memref<512x4096xf32>, %arg2: memref<4096x4096xf32>) {
      %cst = arith.constant 0.000000e+00 : f32
      %c32 = arith.constant 32 : index
      %c256 = arith.constant 256 : index
      affine.parallel (%arg3) = (0) to (8) {
        affine.parallel (%arg4) = (0) to (8) {
          affine.for %arg5 = 0 to 2 {
            affine.for %arg6 = 0 to 128 {
              %12 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 8 + d1 + d2 * 64)>(%arg3, %arg4, %arg5)
              %13 = loom.alloc [32, 32] on @L1 : memref<32x32xf32>
              %14 = loom.init_tensor %13[32, 32] : memref<32x32xf32> -> tensor<32x32xf32>
              %15 = linalg.fill ins(%cst : f32) outs(%14 : tensor<32x32xf32>) -> tensor<32x32xf32>
              %16 = affine.for %arg7 = 0 to 2 iter_args(%arg8 = %15) -> (tensor<32x32xf32>) {
                %20 = arith.muli %arg6, %c32 : index
                %21 = arith.muli %arg7, %c256 : index
                %22 = affine.apply affine_map<(d0, d1) -> (d0 * 512 + d1)>(%20, %21)
                %reinterpret_cast_0 = memref.reinterpret_cast %arg0 to offset: [%22], sizes: [32, 256], strides: [512, 1] : memref<4096x512xf32> to memref<32x256xf32, strided<[512, 1], offset: ?>>
                %23 = loom.alloc [32, 256] on @L1 : memref<32x256xf32>
                %24 = loom.copy_to_tensor %reinterpret_cast_0, %23 on @L1, interconnect : [@vertical_links], broadcast : [8, 1] : memref<32x256xf32, strided<[512, 1], offset: ?>>, memref<32x256xf32> -> tensor<32x256xf32>
                %25 = arith.muli %12, %c32 : index
                %26 = affine.apply affine_map<(d0, d1) -> (d0 * 4096 + d1)>(%21, %25)
                %reinterpret_cast_1 = memref.reinterpret_cast %arg1 to offset: [%26], sizes: [256, 32], strides: [4096, 1] : memref<512x4096xf32> to memref<256x32xf32, strided<[4096, 1], offset: ?>>
                %27 = loom.alloc [256, 32] on @L1 : memref<256x32xf32>
                %28 = loom.copy_to_tensor %reinterpret_cast_1, %27 on @L1, interconnect : [], broadcast : [1, 1] : memref<256x32xf32, strided<[4096, 1], offset: ?>>, memref<256x32xf32> -> tensor<256x32xf32>
                %29 = linalg.matmul ins(%24, %28 : tensor<32x256xf32>, tensor<256x32xf32>) outs(%arg8 : tensor<32x32xf32>) -> tensor<32x32xf32>
                affine.yield %29 : tensor<32x32xf32>
              }
              %17 = arith.muli %arg6, %c32 : index
              %18 = arith.muli %12, %c32 : index
              %19 = affine.apply affine_map<(d0, d1) -> (d0 * 4096 + d1)>(%17, %18)
              %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%19], sizes: [32, 32], strides: [4096, 1] : memref<4096x4096xf32> to memref<32x32xf32, strided<[4096, 1], offset: ?>>
              loom.copy_from_tensor %16, %reinterpret_cast on @DRAM : tensor<32x32xf32>, memref<32x32xf32, strided<[4096, 1], offset: ?>>
            } {loom.iter_type = #loom.iter_type<temporal>}
          } {loom.iter_type = #loom.iter_type<temporal>}
        } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @y}
      } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @x}
      return
    }
  }
}
