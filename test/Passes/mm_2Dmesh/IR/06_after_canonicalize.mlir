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
    func.func @matmul__d0i0_d1i0__f01__d_d__BK64__BM64__BN64(%arg0: memref<4096x512xf16>, %arg1: memref<512x4096xf16>, %arg2: memref<4096x4096xf16>) {
      %cst = arith.constant 0.000000e+00 : f16
      %c64 = arith.constant 64 : index
      affine.parallel (%arg3) = (0) to (8) {
        affine.parallel (%arg4) = (0) to (8) {
          affine.for %arg5 = 0 to 1 {
            affine.for %arg6 = 0 to 64 {
              %12 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 8 + d1 + d2 * 64)>(%arg3, %arg4, %arg5)
              %13 = loom.alloc [64, 64] on @L1 : memref<64x64xf16>
              %14 = loom.semaphore_take %13 : memref<64x64xf16> -> memref<64x64xf16>
              %15 = loom.alloc [64, 64] on @L1 : memref<64x64xf16>
              %16 = loom.semaphore_take %15 : memref<64x64xf16> -> memref<64x64xf16>
              %17 = loom.alloc [64, 64] on @L1 : memref<64x64xf16>
              %18 = loom.semaphore_take %17 : memref<64x64xf16> -> memref<64x64xf16>
              %19 = loom.init_tensor %18[64, 64] : memref<64x64xf16> -> tensor<64x64xf16>
              %20 = linalg.fill ins(%cst : f16) outs(%19 : tensor<64x64xf16>) -> tensor<64x64xf16>
              %21 = affine.for %arg7 = 0 to 8 iter_args(%arg8 = %20) -> (tensor<64x64xf16>) {
                %25 = arith.muli %12, %c64 : index
                %26 = arith.muli %arg7, %c64 : index
                %27 = affine.apply affine_map<(d0, d1) -> (d0 * 512 + d1)>(%25, %26)
                %reinterpret_cast_0 = memref.reinterpret_cast %arg0 to offset: [%27], sizes: [64, 64], strides: [512, 1] : memref<4096x512xf16> to memref<64x64xf16, strided<[512, 1], offset: ?>>
                %28 = loom.copy_to_tensor %reinterpret_cast_0, %16 on @L1, interconnect : [], broadcast : [1, 1] : memref<64x64xf16, strided<[512, 1], offset: ?>>, memref<64x64xf16> -> tensor<64x64xf16>
                %29 = arith.muli %arg6, %c64 : index
                %30 = affine.apply affine_map<(d0, d1) -> (d0 * 4096 + d1)>(%26, %29)
                %reinterpret_cast_1 = memref.reinterpret_cast %arg1 to offset: [%30], sizes: [64, 64], strides: [4096, 1] : memref<512x4096xf16> to memref<64x64xf16, strided<[4096, 1], offset: ?>>
                %31 = loom.copy_to_tensor %reinterpret_cast_1, %14 on @L1, interconnect : [], broadcast : [1, 1] : memref<64x64xf16, strided<[4096, 1], offset: ?>>, memref<64x64xf16> -> tensor<64x64xf16>
                %32 = linalg.matmul ins(%28, %31 : tensor<64x64xf16>, tensor<64x64xf16>) outs(%arg8 : tensor<64x64xf16>) -> tensor<64x64xf16>
                loom.semaphore_give %14 : memref<64x64xf16>
                loom.semaphore_give %16 : memref<64x64xf16>
                affine.yield %32 : tensor<64x64xf16>
              }
              %22 = arith.muli %12, %c64 : index
              %23 = arith.muli %arg6, %c64 : index
              %24 = affine.apply affine_map<(d0, d1) -> (d0 * 4096 + d1)>(%22, %23)
              %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%24], sizes: [64, 64], strides: [4096, 1] : memref<4096x4096xf16> to memref<64x64xf16, strided<[4096, 1], offset: ?>>
              loom.copy_from_tensor %21, %reinterpret_cast on @DRAM : tensor<64x64xf16>, memref<64x64xf16, strided<[4096, 1], offset: ?>>
              loom.semaphore_give %18 : memref<64x64xf16>
            } {loom.iter_type = #loom.iter_type<temporal>}
          } {loom.iter_type = #loom.iter_type<temporal>}
        } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @y}
      } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @x}
      return
    }
  }
  module attributes {loom.pass_name = "Materialize"} {
    func.func @matmul__d0i0_d1i0__f01__d_a__BK64__BM64__BN64(%arg0: memref<4096x512xf16>, %arg1: memref<512x4096xf16>, %arg2: memref<4096x4096xf16>) {
      %cst = arith.constant 0.000000e+00 : f16
      %c64 = arith.constant 64 : index
      affine.parallel (%arg3) = (0) to (8) {
        affine.parallel (%arg4) = (0) to (8) {
          affine.for %arg5 = 0 to 1 {
            affine.for %arg6 = 0 to 64 {
              %12 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 8 + d1 + d2 * 64)>(%arg3, %arg4, %arg5)
              %13 = loom.alloc [64, 64] on @L1 : memref<64x64xf16>
              %14 = loom.semaphore_take %13 : memref<64x64xf16> -> memref<64x64xf16>
              %15 = loom.alloc [64, 64] on @L1 : memref<64x64xf16>
              %16 = loom.semaphore_take %15 : memref<64x64xf16> -> memref<64x64xf16>
              %17 = loom.alloc [64, 64] on @L1 : memref<64x64xf16>
              %18 = loom.semaphore_take %17 : memref<64x64xf16> -> memref<64x64xf16>
              %19 = loom.init_tensor %18[64, 64] : memref<64x64xf16> -> tensor<64x64xf16>
              %20 = linalg.fill ins(%cst : f16) outs(%19 : tensor<64x64xf16>) -> tensor<64x64xf16>
              %21 = affine.for %arg7 = 0 to 8 iter_args(%arg8 = %20) -> (tensor<64x64xf16>) {
                %25 = arith.muli %12, %c64 : index
                %26 = arith.muli %arg7, %c64 : index
                %27 = affine.apply affine_map<(d0, d1) -> (d0 * 512 + d1)>(%25, %26)
                %reinterpret_cast_0 = memref.reinterpret_cast %arg0 to offset: [%27], sizes: [64, 64], strides: [512, 1] : memref<4096x512xf16> to memref<64x64xf16, strided<[512, 1], offset: ?>>
                %28 = loom.copy_to_tensor %reinterpret_cast_0, %16 on @L1, interconnect : [], broadcast : [1, 1] : memref<64x64xf16, strided<[512, 1], offset: ?>>, memref<64x64xf16> -> tensor<64x64xf16>
                %29 = arith.muli %arg6, %c64 : index
                %30 = affine.apply affine_map<(d0, d1) -> (d0 * 4096 + d1)>(%26, %29)
                %reinterpret_cast_1 = memref.reinterpret_cast %arg1 to offset: [%30], sizes: [64, 64], strides: [4096, 1] : memref<512x4096xf16> to memref<64x64xf16, strided<[4096, 1], offset: ?>>
                %31 = loom.copy_to_tensor %reinterpret_cast_1, %14 on @L1, interconnect : [@horizontal_links, @vertical_links], broadcast : [8, 8] : memref<64x64xf16, strided<[4096, 1], offset: ?>>, memref<64x64xf16> -> tensor<64x64xf16>
                %32 = linalg.matmul ins(%28, %31 : tensor<64x64xf16>, tensor<64x64xf16>) outs(%arg8 : tensor<64x64xf16>) -> tensor<64x64xf16>
                loom.semaphore_give %14 : memref<64x64xf16>
                loom.semaphore_give %16 : memref<64x64xf16>
                affine.yield %32 : tensor<64x64xf16>
              }
              %22 = arith.muli %12, %c64 : index
              %23 = arith.muli %arg6, %c64 : index
              %24 = affine.apply affine_map<(d0, d1) -> (d0 * 4096 + d1)>(%22, %23)
              %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%24], sizes: [64, 64], strides: [4096, 1] : memref<4096x4096xf16> to memref<64x64xf16, strided<[4096, 1], offset: ?>>
              loom.copy_from_tensor %21, %reinterpret_cast on @DRAM : tensor<64x64xf16>, memref<64x64xf16, strided<[4096, 1], offset: ?>>
              loom.semaphore_give %18 : memref<64x64xf16>
            } {loom.iter_type = #loom.iter_type<temporal>}
          } {loom.iter_type = #loom.iter_type<temporal>}
        } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @y}
      } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @x}
      return
    }
  }
  module attributes {loom.pass_name = "Materialize"} {
    func.func @matmul__d0i0_d1i0__f01__d_h__BK64__BM64__BN64(%arg0: memref<4096x512xf16>, %arg1: memref<512x4096xf16>, %arg2: memref<4096x4096xf16>) {
      %cst = arith.constant 0.000000e+00 : f16
      %c64 = arith.constant 64 : index
      affine.parallel (%arg3) = (0) to (8) {
        affine.parallel (%arg4) = (0) to (8) {
          affine.for %arg5 = 0 to 1 {
            affine.for %arg6 = 0 to 64 {
              %12 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 8 + d1 + d2 * 64)>(%arg3, %arg4, %arg5)
              %13 = loom.alloc [64, 64] on @L1 : memref<64x64xf16>
              %14 = loom.semaphore_take %13 : memref<64x64xf16> -> memref<64x64xf16>
              %15 = loom.alloc [64, 64] on @L1 : memref<64x64xf16>
              %16 = loom.semaphore_take %15 : memref<64x64xf16> -> memref<64x64xf16>
              %17 = loom.alloc [64, 64] on @L1 : memref<64x64xf16>
              %18 = loom.semaphore_take %17 : memref<64x64xf16> -> memref<64x64xf16>
              %19 = loom.init_tensor %18[64, 64] : memref<64x64xf16> -> tensor<64x64xf16>
              %20 = linalg.fill ins(%cst : f16) outs(%19 : tensor<64x64xf16>) -> tensor<64x64xf16>
              %21 = affine.for %arg7 = 0 to 8 iter_args(%arg8 = %20) -> (tensor<64x64xf16>) {
                %25 = arith.muli %12, %c64 : index
                %26 = arith.muli %arg7, %c64 : index
                %27 = affine.apply affine_map<(d0, d1) -> (d0 * 512 + d1)>(%25, %26)
                %reinterpret_cast_0 = memref.reinterpret_cast %arg0 to offset: [%27], sizes: [64, 64], strides: [512, 1] : memref<4096x512xf16> to memref<64x64xf16, strided<[512, 1], offset: ?>>
                %28 = loom.copy_to_tensor %reinterpret_cast_0, %16 on @L1, interconnect : [], broadcast : [1, 1] : memref<64x64xf16, strided<[512, 1], offset: ?>>, memref<64x64xf16> -> tensor<64x64xf16>
                %29 = arith.muli %arg6, %c64 : index
                %30 = affine.apply affine_map<(d0, d1) -> (d0 * 4096 + d1)>(%26, %29)
                %reinterpret_cast_1 = memref.reinterpret_cast %arg1 to offset: [%30], sizes: [64, 64], strides: [4096, 1] : memref<512x4096xf16> to memref<64x64xf16, strided<[4096, 1], offset: ?>>
                %31 = loom.copy_to_tensor %reinterpret_cast_1, %14 on @L1, interconnect : [@horizontal_links], broadcast : [1, 8] : memref<64x64xf16, strided<[4096, 1], offset: ?>>, memref<64x64xf16> -> tensor<64x64xf16>
                %32 = linalg.matmul ins(%28, %31 : tensor<64x64xf16>, tensor<64x64xf16>) outs(%arg8 : tensor<64x64xf16>) -> tensor<64x64xf16>
                loom.semaphore_give %14 : memref<64x64xf16>
                loom.semaphore_give %16 : memref<64x64xf16>
                affine.yield %32 : tensor<64x64xf16>
              }
              %22 = arith.muli %12, %c64 : index
              %23 = arith.muli %arg6, %c64 : index
              %24 = affine.apply affine_map<(d0, d1) -> (d0 * 4096 + d1)>(%22, %23)
              %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%24], sizes: [64, 64], strides: [4096, 1] : memref<4096x4096xf16> to memref<64x64xf16, strided<[4096, 1], offset: ?>>
              loom.copy_from_tensor %21, %reinterpret_cast on @DRAM : tensor<64x64xf16>, memref<64x64xf16, strided<[4096, 1], offset: ?>>
              loom.semaphore_give %18 : memref<64x64xf16>
            } {loom.iter_type = #loom.iter_type<temporal>}
          } {loom.iter_type = #loom.iter_type<temporal>}
        } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @y}
      } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @x}
      return
    }
  }
  module attributes {loom.pass_name = "Materialize"} {
    func.func @matmul__d0i0_d1i0__f01__d_v__BK64__BM64__BN64(%arg0: memref<4096x512xf16>, %arg1: memref<512x4096xf16>, %arg2: memref<4096x4096xf16>) {
      %cst = arith.constant 0.000000e+00 : f16
      %c64 = arith.constant 64 : index
      affine.parallel (%arg3) = (0) to (8) {
        affine.parallel (%arg4) = (0) to (8) {
          affine.for %arg5 = 0 to 1 {
            affine.for %arg6 = 0 to 64 {
              %12 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 8 + d1 + d2 * 64)>(%arg3, %arg4, %arg5)
              %13 = loom.alloc [64, 64] on @L1 : memref<64x64xf16>
              %14 = loom.semaphore_take %13 : memref<64x64xf16> -> memref<64x64xf16>
              %15 = loom.alloc [64, 64] on @L1 : memref<64x64xf16>
              %16 = loom.semaphore_take %15 : memref<64x64xf16> -> memref<64x64xf16>
              %17 = loom.alloc [64, 64] on @L1 : memref<64x64xf16>
              %18 = loom.semaphore_take %17 : memref<64x64xf16> -> memref<64x64xf16>
              %19 = loom.init_tensor %18[64, 64] : memref<64x64xf16> -> tensor<64x64xf16>
              %20 = linalg.fill ins(%cst : f16) outs(%19 : tensor<64x64xf16>) -> tensor<64x64xf16>
              %21 = affine.for %arg7 = 0 to 8 iter_args(%arg8 = %20) -> (tensor<64x64xf16>) {
                %25 = arith.muli %12, %c64 : index
                %26 = arith.muli %arg7, %c64 : index
                %27 = affine.apply affine_map<(d0, d1) -> (d0 * 512 + d1)>(%25, %26)
                %reinterpret_cast_0 = memref.reinterpret_cast %arg0 to offset: [%27], sizes: [64, 64], strides: [512, 1] : memref<4096x512xf16> to memref<64x64xf16, strided<[512, 1], offset: ?>>
                %28 = loom.copy_to_tensor %reinterpret_cast_0, %16 on @L1, interconnect : [], broadcast : [1, 1] : memref<64x64xf16, strided<[512, 1], offset: ?>>, memref<64x64xf16> -> tensor<64x64xf16>
                %29 = arith.muli %arg6, %c64 : index
                %30 = affine.apply affine_map<(d0, d1) -> (d0 * 4096 + d1)>(%26, %29)
                %reinterpret_cast_1 = memref.reinterpret_cast %arg1 to offset: [%30], sizes: [64, 64], strides: [4096, 1] : memref<512x4096xf16> to memref<64x64xf16, strided<[4096, 1], offset: ?>>
                %31 = loom.copy_to_tensor %reinterpret_cast_1, %14 on @L1, interconnect : [@vertical_links], broadcast : [8, 1] : memref<64x64xf16, strided<[4096, 1], offset: ?>>, memref<64x64xf16> -> tensor<64x64xf16>
                %32 = linalg.matmul ins(%28, %31 : tensor<64x64xf16>, tensor<64x64xf16>) outs(%arg8 : tensor<64x64xf16>) -> tensor<64x64xf16>
                loom.semaphore_give %14 : memref<64x64xf16>
                loom.semaphore_give %16 : memref<64x64xf16>
                affine.yield %32 : tensor<64x64xf16>
              }
              %22 = arith.muli %12, %c64 : index
              %23 = arith.muli %arg6, %c64 : index
              %24 = affine.apply affine_map<(d0, d1) -> (d0 * 4096 + d1)>(%22, %23)
              %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%24], sizes: [64, 64], strides: [4096, 1] : memref<4096x4096xf16> to memref<64x64xf16, strided<[4096, 1], offset: ?>>
              loom.copy_from_tensor %21, %reinterpret_cast on @DRAM : tensor<64x64xf16>, memref<64x64xf16, strided<[4096, 1], offset: ?>>
              loom.semaphore_give %18 : memref<64x64xf16>
            } {loom.iter_type = #loom.iter_type<temporal>}
          } {loom.iter_type = #loom.iter_type<temporal>}
        } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @y}
      } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @x}
      return
    }
  }
  module attributes {loom.pass_name = "Materialize"} {
    func.func @matmul__d0i0_d1i0__f10__d_d__BK64__BM64__BN64(%arg0: memref<4096x512xf16>, %arg1: memref<512x4096xf16>, %arg2: memref<4096x4096xf16>) {
      %cst = arith.constant 0.000000e+00 : f16
      %c64 = arith.constant 64 : index
      affine.parallel (%arg3) = (0) to (8) {
        affine.parallel (%arg4) = (0) to (8) {
          affine.for %arg5 = 0 to 64 {
            affine.for %arg6 = 0 to 1 {
              %12 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 8 + d1 + d2 * 64)>(%arg3, %arg4, %arg6)
              %13 = loom.alloc [64, 64] on @L1 : memref<64x64xf16>
              %14 = loom.semaphore_take %13 : memref<64x64xf16> -> memref<64x64xf16>
              %15 = loom.alloc [64, 64] on @L1 : memref<64x64xf16>
              %16 = loom.semaphore_take %15 : memref<64x64xf16> -> memref<64x64xf16>
              %17 = loom.alloc [64, 64] on @L1 : memref<64x64xf16>
              %18 = loom.semaphore_take %17 : memref<64x64xf16> -> memref<64x64xf16>
              %19 = loom.init_tensor %18[64, 64] : memref<64x64xf16> -> tensor<64x64xf16>
              %20 = linalg.fill ins(%cst : f16) outs(%19 : tensor<64x64xf16>) -> tensor<64x64xf16>
              %21 = affine.for %arg7 = 0 to 8 iter_args(%arg8 = %20) -> (tensor<64x64xf16>) {
                %25 = arith.muli %12, %c64 : index
                %26 = arith.muli %arg7, %c64 : index
                %27 = affine.apply affine_map<(d0, d1) -> (d0 * 512 + d1)>(%25, %26)
                %reinterpret_cast_0 = memref.reinterpret_cast %arg0 to offset: [%27], sizes: [64, 64], strides: [512, 1] : memref<4096x512xf16> to memref<64x64xf16, strided<[512, 1], offset: ?>>
                %28 = loom.copy_to_tensor %reinterpret_cast_0, %16 on @L1, interconnect : [], broadcast : [1, 1] : memref<64x64xf16, strided<[512, 1], offset: ?>>, memref<64x64xf16> -> tensor<64x64xf16>
                %29 = arith.muli %arg5, %c64 : index
                %30 = affine.apply affine_map<(d0, d1) -> (d0 * 4096 + d1)>(%26, %29)
                %reinterpret_cast_1 = memref.reinterpret_cast %arg1 to offset: [%30], sizes: [64, 64], strides: [4096, 1] : memref<512x4096xf16> to memref<64x64xf16, strided<[4096, 1], offset: ?>>
                %31 = loom.copy_to_tensor %reinterpret_cast_1, %14 on @L1, interconnect : [], broadcast : [1, 1] : memref<64x64xf16, strided<[4096, 1], offset: ?>>, memref<64x64xf16> -> tensor<64x64xf16>
                %32 = linalg.matmul ins(%28, %31 : tensor<64x64xf16>, tensor<64x64xf16>) outs(%arg8 : tensor<64x64xf16>) -> tensor<64x64xf16>
                loom.semaphore_give %14 : memref<64x64xf16>
                loom.semaphore_give %16 : memref<64x64xf16>
                affine.yield %32 : tensor<64x64xf16>
              }
              %22 = arith.muli %12, %c64 : index
              %23 = arith.muli %arg5, %c64 : index
              %24 = affine.apply affine_map<(d0, d1) -> (d0 * 4096 + d1)>(%22, %23)
              %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%24], sizes: [64, 64], strides: [4096, 1] : memref<4096x4096xf16> to memref<64x64xf16, strided<[4096, 1], offset: ?>>
              loom.copy_from_tensor %21, %reinterpret_cast on @DRAM : tensor<64x64xf16>, memref<64x64xf16, strided<[4096, 1], offset: ?>>
              loom.semaphore_give %18 : memref<64x64xf16>
            } {loom.iter_type = #loom.iter_type<temporal>}
          } {loom.iter_type = #loom.iter_type<temporal>}
        } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @y}
      } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @x}
      return
    }
  }
  module attributes {loom.pass_name = "Materialize"} {
    func.func @matmul__d0i0_d1i0__f10__d_a__BK64__BM64__BN64(%arg0: memref<4096x512xf16>, %arg1: memref<512x4096xf16>, %arg2: memref<4096x4096xf16>) {
      %cst = arith.constant 0.000000e+00 : f16
      %c64 = arith.constant 64 : index
      affine.parallel (%arg3) = (0) to (8) {
        affine.parallel (%arg4) = (0) to (8) {
          affine.for %arg5 = 0 to 64 {
            affine.for %arg6 = 0 to 1 {
              %12 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 8 + d1 + d2 * 64)>(%arg3, %arg4, %arg6)
              %13 = loom.alloc [64, 64] on @L1 : memref<64x64xf16>
              %14 = loom.semaphore_take %13 : memref<64x64xf16> -> memref<64x64xf16>
              %15 = loom.alloc [64, 64] on @L1 : memref<64x64xf16>
              %16 = loom.semaphore_take %15 : memref<64x64xf16> -> memref<64x64xf16>
              %17 = loom.alloc [64, 64] on @L1 : memref<64x64xf16>
              %18 = loom.semaphore_take %17 : memref<64x64xf16> -> memref<64x64xf16>
              %19 = loom.init_tensor %18[64, 64] : memref<64x64xf16> -> tensor<64x64xf16>
              %20 = linalg.fill ins(%cst : f16) outs(%19 : tensor<64x64xf16>) -> tensor<64x64xf16>
              %21 = affine.for %arg7 = 0 to 8 iter_args(%arg8 = %20) -> (tensor<64x64xf16>) {
                %25 = arith.muli %12, %c64 : index
                %26 = arith.muli %arg7, %c64 : index
                %27 = affine.apply affine_map<(d0, d1) -> (d0 * 512 + d1)>(%25, %26)
                %reinterpret_cast_0 = memref.reinterpret_cast %arg0 to offset: [%27], sizes: [64, 64], strides: [512, 1] : memref<4096x512xf16> to memref<64x64xf16, strided<[512, 1], offset: ?>>
                %28 = loom.copy_to_tensor %reinterpret_cast_0, %16 on @L1, interconnect : [], broadcast : [1, 1] : memref<64x64xf16, strided<[512, 1], offset: ?>>, memref<64x64xf16> -> tensor<64x64xf16>
                %29 = arith.muli %arg5, %c64 : index
                %30 = affine.apply affine_map<(d0, d1) -> (d0 * 4096 + d1)>(%26, %29)
                %reinterpret_cast_1 = memref.reinterpret_cast %arg1 to offset: [%30], sizes: [64, 64], strides: [4096, 1] : memref<512x4096xf16> to memref<64x64xf16, strided<[4096, 1], offset: ?>>
                %31 = loom.copy_to_tensor %reinterpret_cast_1, %14 on @L1, interconnect : [@horizontal_links, @vertical_links], broadcast : [8, 8] : memref<64x64xf16, strided<[4096, 1], offset: ?>>, memref<64x64xf16> -> tensor<64x64xf16>
                %32 = linalg.matmul ins(%28, %31 : tensor<64x64xf16>, tensor<64x64xf16>) outs(%arg8 : tensor<64x64xf16>) -> tensor<64x64xf16>
                loom.semaphore_give %14 : memref<64x64xf16>
                loom.semaphore_give %16 : memref<64x64xf16>
                affine.yield %32 : tensor<64x64xf16>
              }
              %22 = arith.muli %12, %c64 : index
              %23 = arith.muli %arg5, %c64 : index
              %24 = affine.apply affine_map<(d0, d1) -> (d0 * 4096 + d1)>(%22, %23)
              %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%24], sizes: [64, 64], strides: [4096, 1] : memref<4096x4096xf16> to memref<64x64xf16, strided<[4096, 1], offset: ?>>
              loom.copy_from_tensor %21, %reinterpret_cast on @DRAM : tensor<64x64xf16>, memref<64x64xf16, strided<[4096, 1], offset: ?>>
              loom.semaphore_give %18 : memref<64x64xf16>
            } {loom.iter_type = #loom.iter_type<temporal>}
          } {loom.iter_type = #loom.iter_type<temporal>}
        } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @y}
      } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @x}
      return
    }
  }
  module attributes {loom.pass_name = "Materialize"} {
    func.func @matmul__d0i0_d1i0__f10__d_h__BK64__BM64__BN64(%arg0: memref<4096x512xf16>, %arg1: memref<512x4096xf16>, %arg2: memref<4096x4096xf16>) {
      %cst = arith.constant 0.000000e+00 : f16
      %c64 = arith.constant 64 : index
      affine.parallel (%arg3) = (0) to (8) {
        affine.parallel (%arg4) = (0) to (8) {
          affine.for %arg5 = 0 to 64 {
            affine.for %arg6 = 0 to 1 {
              %12 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 8 + d1 + d2 * 64)>(%arg3, %arg4, %arg6)
              %13 = loom.alloc [64, 64] on @L1 : memref<64x64xf16>
              %14 = loom.semaphore_take %13 : memref<64x64xf16> -> memref<64x64xf16>
              %15 = loom.alloc [64, 64] on @L1 : memref<64x64xf16>
              %16 = loom.semaphore_take %15 : memref<64x64xf16> -> memref<64x64xf16>
              %17 = loom.alloc [64, 64] on @L1 : memref<64x64xf16>
              %18 = loom.semaphore_take %17 : memref<64x64xf16> -> memref<64x64xf16>
              %19 = loom.init_tensor %18[64, 64] : memref<64x64xf16> -> tensor<64x64xf16>
              %20 = linalg.fill ins(%cst : f16) outs(%19 : tensor<64x64xf16>) -> tensor<64x64xf16>
              %21 = affine.for %arg7 = 0 to 8 iter_args(%arg8 = %20) -> (tensor<64x64xf16>) {
                %25 = arith.muli %12, %c64 : index
                %26 = arith.muli %arg7, %c64 : index
                %27 = affine.apply affine_map<(d0, d1) -> (d0 * 512 + d1)>(%25, %26)
                %reinterpret_cast_0 = memref.reinterpret_cast %arg0 to offset: [%27], sizes: [64, 64], strides: [512, 1] : memref<4096x512xf16> to memref<64x64xf16, strided<[512, 1], offset: ?>>
                %28 = loom.copy_to_tensor %reinterpret_cast_0, %16 on @L1, interconnect : [], broadcast : [1, 1] : memref<64x64xf16, strided<[512, 1], offset: ?>>, memref<64x64xf16> -> tensor<64x64xf16>
                %29 = arith.muli %arg5, %c64 : index
                %30 = affine.apply affine_map<(d0, d1) -> (d0 * 4096 + d1)>(%26, %29)
                %reinterpret_cast_1 = memref.reinterpret_cast %arg1 to offset: [%30], sizes: [64, 64], strides: [4096, 1] : memref<512x4096xf16> to memref<64x64xf16, strided<[4096, 1], offset: ?>>
                %31 = loom.copy_to_tensor %reinterpret_cast_1, %14 on @L1, interconnect : [@horizontal_links], broadcast : [1, 8] : memref<64x64xf16, strided<[4096, 1], offset: ?>>, memref<64x64xf16> -> tensor<64x64xf16>
                %32 = linalg.matmul ins(%28, %31 : tensor<64x64xf16>, tensor<64x64xf16>) outs(%arg8 : tensor<64x64xf16>) -> tensor<64x64xf16>
                loom.semaphore_give %14 : memref<64x64xf16>
                loom.semaphore_give %16 : memref<64x64xf16>
                affine.yield %32 : tensor<64x64xf16>
              }
              %22 = arith.muli %12, %c64 : index
              %23 = arith.muli %arg5, %c64 : index
              %24 = affine.apply affine_map<(d0, d1) -> (d0 * 4096 + d1)>(%22, %23)
              %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%24], sizes: [64, 64], strides: [4096, 1] : memref<4096x4096xf16> to memref<64x64xf16, strided<[4096, 1], offset: ?>>
              loom.copy_from_tensor %21, %reinterpret_cast on @DRAM : tensor<64x64xf16>, memref<64x64xf16, strided<[4096, 1], offset: ?>>
              loom.semaphore_give %18 : memref<64x64xf16>
            } {loom.iter_type = #loom.iter_type<temporal>}
          } {loom.iter_type = #loom.iter_type<temporal>}
        } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @y}
      } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @x}
      return
    }
  }
  module attributes {loom.pass_name = "Materialize"} {
    func.func @matmul__d0i0_d1i0__f10__d_v__BK64__BM64__BN64(%arg0: memref<4096x512xf16>, %arg1: memref<512x4096xf16>, %arg2: memref<4096x4096xf16>) {
      %cst = arith.constant 0.000000e+00 : f16
      %c64 = arith.constant 64 : index
      affine.parallel (%arg3) = (0) to (8) {
        affine.parallel (%arg4) = (0) to (8) {
          affine.for %arg5 = 0 to 64 {
            affine.for %arg6 = 0 to 1 {
              %12 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 8 + d1 + d2 * 64)>(%arg3, %arg4, %arg6)
              %13 = loom.alloc [64, 64] on @L1 : memref<64x64xf16>
              %14 = loom.semaphore_take %13 : memref<64x64xf16> -> memref<64x64xf16>
              %15 = loom.alloc [64, 64] on @L1 : memref<64x64xf16>
              %16 = loom.semaphore_take %15 : memref<64x64xf16> -> memref<64x64xf16>
              %17 = loom.alloc [64, 64] on @L1 : memref<64x64xf16>
              %18 = loom.semaphore_take %17 : memref<64x64xf16> -> memref<64x64xf16>
              %19 = loom.init_tensor %18[64, 64] : memref<64x64xf16> -> tensor<64x64xf16>
              %20 = linalg.fill ins(%cst : f16) outs(%19 : tensor<64x64xf16>) -> tensor<64x64xf16>
              %21 = affine.for %arg7 = 0 to 8 iter_args(%arg8 = %20) -> (tensor<64x64xf16>) {
                %25 = arith.muli %12, %c64 : index
                %26 = arith.muli %arg7, %c64 : index
                %27 = affine.apply affine_map<(d0, d1) -> (d0 * 512 + d1)>(%25, %26)
                %reinterpret_cast_0 = memref.reinterpret_cast %arg0 to offset: [%27], sizes: [64, 64], strides: [512, 1] : memref<4096x512xf16> to memref<64x64xf16, strided<[512, 1], offset: ?>>
                %28 = loom.copy_to_tensor %reinterpret_cast_0, %16 on @L1, interconnect : [], broadcast : [1, 1] : memref<64x64xf16, strided<[512, 1], offset: ?>>, memref<64x64xf16> -> tensor<64x64xf16>
                %29 = arith.muli %arg5, %c64 : index
                %30 = affine.apply affine_map<(d0, d1) -> (d0 * 4096 + d1)>(%26, %29)
                %reinterpret_cast_1 = memref.reinterpret_cast %arg1 to offset: [%30], sizes: [64, 64], strides: [4096, 1] : memref<512x4096xf16> to memref<64x64xf16, strided<[4096, 1], offset: ?>>
                %31 = loom.copy_to_tensor %reinterpret_cast_1, %14 on @L1, interconnect : [@vertical_links], broadcast : [8, 1] : memref<64x64xf16, strided<[4096, 1], offset: ?>>, memref<64x64xf16> -> tensor<64x64xf16>
                %32 = linalg.matmul ins(%28, %31 : tensor<64x64xf16>, tensor<64x64xf16>) outs(%arg8 : tensor<64x64xf16>) -> tensor<64x64xf16>
                loom.semaphore_give %14 : memref<64x64xf16>
                loom.semaphore_give %16 : memref<64x64xf16>
                affine.yield %32 : tensor<64x64xf16>
              }
              %22 = arith.muli %12, %c64 : index
              %23 = arith.muli %arg5, %c64 : index
              %24 = affine.apply affine_map<(d0, d1) -> (d0 * 4096 + d1)>(%22, %23)
              %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%24], sizes: [64, 64], strides: [4096, 1] : memref<4096x4096xf16> to memref<64x64xf16, strided<[4096, 1], offset: ?>>
              loom.copy_from_tensor %21, %reinterpret_cast on @DRAM : tensor<64x64xf16>, memref<64x64xf16, strided<[4096, 1], offset: ?>>
              loom.semaphore_give %18 : memref<64x64xf16>
            } {loom.iter_type = #loom.iter_type<temporal>}
          } {loom.iter_type = #loom.iter_type<temporal>}
        } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @y}
      } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @x}
      return
    }
  }
  module attributes {loom.pass_name = "Materialize"} {
    func.func @matmul__d1i0_d0i0__f01__d_d__BK64__BM64__BN64(%arg0: memref<4096x512xf16>, %arg1: memref<512x4096xf16>, %arg2: memref<4096x4096xf16>) {
      %cst = arith.constant 0.000000e+00 : f16
      %c64 = arith.constant 64 : index
      affine.parallel (%arg3) = (0) to (8) {
        affine.parallel (%arg4) = (0) to (8) {
          affine.for %arg5 = 0 to 1 {
            affine.for %arg6 = 0 to 64 {
              %12 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 8 + d1 + d2 * 64)>(%arg3, %arg4, %arg5)
              %13 = loom.alloc [64, 64] on @L1 : memref<64x64xf16>
              %14 = loom.semaphore_take %13 : memref<64x64xf16> -> memref<64x64xf16>
              %15 = loom.alloc [64, 64] on @L1 : memref<64x64xf16>
              %16 = loom.semaphore_take %15 : memref<64x64xf16> -> memref<64x64xf16>
              %17 = loom.alloc [64, 64] on @L1 : memref<64x64xf16>
              %18 = loom.semaphore_take %17 : memref<64x64xf16> -> memref<64x64xf16>
              %19 = loom.init_tensor %18[64, 64] : memref<64x64xf16> -> tensor<64x64xf16>
              %20 = linalg.fill ins(%cst : f16) outs(%19 : tensor<64x64xf16>) -> tensor<64x64xf16>
              %21 = affine.for %arg7 = 0 to 8 iter_args(%arg8 = %20) -> (tensor<64x64xf16>) {
                %25 = arith.muli %12, %c64 : index
                %26 = arith.muli %arg7, %c64 : index
                %27 = affine.apply affine_map<(d0, d1) -> (d0 * 512 + d1)>(%25, %26)
                %reinterpret_cast_0 = memref.reinterpret_cast %arg0 to offset: [%27], sizes: [64, 64], strides: [512, 1] : memref<4096x512xf16> to memref<64x64xf16, strided<[512, 1], offset: ?>>
                %28 = loom.copy_to_tensor %reinterpret_cast_0, %16 on @L1, interconnect : [], broadcast : [1, 1] : memref<64x64xf16, strided<[512, 1], offset: ?>>, memref<64x64xf16> -> tensor<64x64xf16>
                %29 = arith.muli %arg6, %c64 : index
                %30 = affine.apply affine_map<(d0, d1) -> (d0 * 4096 + d1)>(%26, %29)
                %reinterpret_cast_1 = memref.reinterpret_cast %arg1 to offset: [%30], sizes: [64, 64], strides: [4096, 1] : memref<512x4096xf16> to memref<64x64xf16, strided<[4096, 1], offset: ?>>
                %31 = loom.copy_to_tensor %reinterpret_cast_1, %14 on @L1, interconnect : [], broadcast : [1, 1] : memref<64x64xf16, strided<[4096, 1], offset: ?>>, memref<64x64xf16> -> tensor<64x64xf16>
                %32 = linalg.matmul ins(%28, %31 : tensor<64x64xf16>, tensor<64x64xf16>) outs(%arg8 : tensor<64x64xf16>) -> tensor<64x64xf16>
                loom.semaphore_give %14 : memref<64x64xf16>
                loom.semaphore_give %16 : memref<64x64xf16>
                affine.yield %32 : tensor<64x64xf16>
              }
              %22 = arith.muli %12, %c64 : index
              %23 = arith.muli %arg6, %c64 : index
              %24 = affine.apply affine_map<(d0, d1) -> (d0 * 4096 + d1)>(%22, %23)
              %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%24], sizes: [64, 64], strides: [4096, 1] : memref<4096x4096xf16> to memref<64x64xf16, strided<[4096, 1], offset: ?>>
              loom.copy_from_tensor %21, %reinterpret_cast on @DRAM : tensor<64x64xf16>, memref<64x64xf16, strided<[4096, 1], offset: ?>>
              loom.semaphore_give %18 : memref<64x64xf16>
            } {loom.iter_type = #loom.iter_type<temporal>}
          } {loom.iter_type = #loom.iter_type<temporal>}
        } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @x}
      } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @y}
      return
    }
  }
  module attributes {loom.pass_name = "Materialize"} {
    func.func @matmul__d1i0_d0i0__f01__d_a__BK64__BM64__BN64(%arg0: memref<4096x512xf16>, %arg1: memref<512x4096xf16>, %arg2: memref<4096x4096xf16>) {
      %cst = arith.constant 0.000000e+00 : f16
      %c64 = arith.constant 64 : index
      affine.parallel (%arg3) = (0) to (8) {
        affine.parallel (%arg4) = (0) to (8) {
          affine.for %arg5 = 0 to 1 {
            affine.for %arg6 = 0 to 64 {
              %12 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 8 + d1 + d2 * 64)>(%arg3, %arg4, %arg5)
              %13 = loom.alloc [64, 64] on @L1 : memref<64x64xf16>
              %14 = loom.semaphore_take %13 : memref<64x64xf16> -> memref<64x64xf16>
              %15 = loom.alloc [64, 64] on @L1 : memref<64x64xf16>
              %16 = loom.semaphore_take %15 : memref<64x64xf16> -> memref<64x64xf16>
              %17 = loom.alloc [64, 64] on @L1 : memref<64x64xf16>
              %18 = loom.semaphore_take %17 : memref<64x64xf16> -> memref<64x64xf16>
              %19 = loom.init_tensor %18[64, 64] : memref<64x64xf16> -> tensor<64x64xf16>
              %20 = linalg.fill ins(%cst : f16) outs(%19 : tensor<64x64xf16>) -> tensor<64x64xf16>
              %21 = affine.for %arg7 = 0 to 8 iter_args(%arg8 = %20) -> (tensor<64x64xf16>) {
                %25 = arith.muli %12, %c64 : index
                %26 = arith.muli %arg7, %c64 : index
                %27 = affine.apply affine_map<(d0, d1) -> (d0 * 512 + d1)>(%25, %26)
                %reinterpret_cast_0 = memref.reinterpret_cast %arg0 to offset: [%27], sizes: [64, 64], strides: [512, 1] : memref<4096x512xf16> to memref<64x64xf16, strided<[512, 1], offset: ?>>
                %28 = loom.copy_to_tensor %reinterpret_cast_0, %16 on @L1, interconnect : [], broadcast : [1, 1] : memref<64x64xf16, strided<[512, 1], offset: ?>>, memref<64x64xf16> -> tensor<64x64xf16>
                %29 = arith.muli %arg6, %c64 : index
                %30 = affine.apply affine_map<(d0, d1) -> (d0 * 4096 + d1)>(%26, %29)
                %reinterpret_cast_1 = memref.reinterpret_cast %arg1 to offset: [%30], sizes: [64, 64], strides: [4096, 1] : memref<512x4096xf16> to memref<64x64xf16, strided<[4096, 1], offset: ?>>
                %31 = loom.copy_to_tensor %reinterpret_cast_1, %14 on @L1, interconnect : [@horizontal_links, @vertical_links], broadcast : [8, 8] : memref<64x64xf16, strided<[4096, 1], offset: ?>>, memref<64x64xf16> -> tensor<64x64xf16>
                %32 = linalg.matmul ins(%28, %31 : tensor<64x64xf16>, tensor<64x64xf16>) outs(%arg8 : tensor<64x64xf16>) -> tensor<64x64xf16>
                loom.semaphore_give %14 : memref<64x64xf16>
                loom.semaphore_give %16 : memref<64x64xf16>
                affine.yield %32 : tensor<64x64xf16>
              }
              %22 = arith.muli %12, %c64 : index
              %23 = arith.muli %arg6, %c64 : index
              %24 = affine.apply affine_map<(d0, d1) -> (d0 * 4096 + d1)>(%22, %23)
              %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%24], sizes: [64, 64], strides: [4096, 1] : memref<4096x4096xf16> to memref<64x64xf16, strided<[4096, 1], offset: ?>>
              loom.copy_from_tensor %21, %reinterpret_cast on @DRAM : tensor<64x64xf16>, memref<64x64xf16, strided<[4096, 1], offset: ?>>
              loom.semaphore_give %18 : memref<64x64xf16>
            } {loom.iter_type = #loom.iter_type<temporal>}
          } {loom.iter_type = #loom.iter_type<temporal>}
        } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @x}
      } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @y}
      return
    }
  }
  module attributes {loom.pass_name = "Materialize"} {
    func.func @matmul__d1i0_d0i0__f01__d_h__BK64__BM64__BN64(%arg0: memref<4096x512xf16>, %arg1: memref<512x4096xf16>, %arg2: memref<4096x4096xf16>) {
      %cst = arith.constant 0.000000e+00 : f16
      %c64 = arith.constant 64 : index
      affine.parallel (%arg3) = (0) to (8) {
        affine.parallel (%arg4) = (0) to (8) {
          affine.for %arg5 = 0 to 1 {
            affine.for %arg6 = 0 to 64 {
              %12 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 8 + d1 + d2 * 64)>(%arg3, %arg4, %arg5)
              %13 = loom.alloc [64, 64] on @L1 : memref<64x64xf16>
              %14 = loom.semaphore_take %13 : memref<64x64xf16> -> memref<64x64xf16>
              %15 = loom.alloc [64, 64] on @L1 : memref<64x64xf16>
              %16 = loom.semaphore_take %15 : memref<64x64xf16> -> memref<64x64xf16>
              %17 = loom.alloc [64, 64] on @L1 : memref<64x64xf16>
              %18 = loom.semaphore_take %17 : memref<64x64xf16> -> memref<64x64xf16>
              %19 = loom.init_tensor %18[64, 64] : memref<64x64xf16> -> tensor<64x64xf16>
              %20 = linalg.fill ins(%cst : f16) outs(%19 : tensor<64x64xf16>) -> tensor<64x64xf16>
              %21 = affine.for %arg7 = 0 to 8 iter_args(%arg8 = %20) -> (tensor<64x64xf16>) {
                %25 = arith.muli %12, %c64 : index
                %26 = arith.muli %arg7, %c64 : index
                %27 = affine.apply affine_map<(d0, d1) -> (d0 * 512 + d1)>(%25, %26)
                %reinterpret_cast_0 = memref.reinterpret_cast %arg0 to offset: [%27], sizes: [64, 64], strides: [512, 1] : memref<4096x512xf16> to memref<64x64xf16, strided<[512, 1], offset: ?>>
                %28 = loom.copy_to_tensor %reinterpret_cast_0, %16 on @L1, interconnect : [], broadcast : [1, 1] : memref<64x64xf16, strided<[512, 1], offset: ?>>, memref<64x64xf16> -> tensor<64x64xf16>
                %29 = arith.muli %arg6, %c64 : index
                %30 = affine.apply affine_map<(d0, d1) -> (d0 * 4096 + d1)>(%26, %29)
                %reinterpret_cast_1 = memref.reinterpret_cast %arg1 to offset: [%30], sizes: [64, 64], strides: [4096, 1] : memref<512x4096xf16> to memref<64x64xf16, strided<[4096, 1], offset: ?>>
                %31 = loom.copy_to_tensor %reinterpret_cast_1, %14 on @L1, interconnect : [@horizontal_links], broadcast : [1, 8] : memref<64x64xf16, strided<[4096, 1], offset: ?>>, memref<64x64xf16> -> tensor<64x64xf16>
                %32 = linalg.matmul ins(%28, %31 : tensor<64x64xf16>, tensor<64x64xf16>) outs(%arg8 : tensor<64x64xf16>) -> tensor<64x64xf16>
                loom.semaphore_give %14 : memref<64x64xf16>
                loom.semaphore_give %16 : memref<64x64xf16>
                affine.yield %32 : tensor<64x64xf16>
              }
              %22 = arith.muli %12, %c64 : index
              %23 = arith.muli %arg6, %c64 : index
              %24 = affine.apply affine_map<(d0, d1) -> (d0 * 4096 + d1)>(%22, %23)
              %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%24], sizes: [64, 64], strides: [4096, 1] : memref<4096x4096xf16> to memref<64x64xf16, strided<[4096, 1], offset: ?>>
              loom.copy_from_tensor %21, %reinterpret_cast on @DRAM : tensor<64x64xf16>, memref<64x64xf16, strided<[4096, 1], offset: ?>>
              loom.semaphore_give %18 : memref<64x64xf16>
            } {loom.iter_type = #loom.iter_type<temporal>}
          } {loom.iter_type = #loom.iter_type<temporal>}
        } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @x}
      } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @y}
      return
    }
  }
  module attributes {loom.pass_name = "Materialize"} {
    func.func @matmul__d1i0_d0i0__f01__d_v__BK64__BM64__BN64(%arg0: memref<4096x512xf16>, %arg1: memref<512x4096xf16>, %arg2: memref<4096x4096xf16>) {
      %cst = arith.constant 0.000000e+00 : f16
      %c64 = arith.constant 64 : index
      affine.parallel (%arg3) = (0) to (8) {
        affine.parallel (%arg4) = (0) to (8) {
          affine.for %arg5 = 0 to 1 {
            affine.for %arg6 = 0 to 64 {
              %12 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 8 + d1 + d2 * 64)>(%arg3, %arg4, %arg5)
              %13 = loom.alloc [64, 64] on @L1 : memref<64x64xf16>
              %14 = loom.semaphore_take %13 : memref<64x64xf16> -> memref<64x64xf16>
              %15 = loom.alloc [64, 64] on @L1 : memref<64x64xf16>
              %16 = loom.semaphore_take %15 : memref<64x64xf16> -> memref<64x64xf16>
              %17 = loom.alloc [64, 64] on @L1 : memref<64x64xf16>
              %18 = loom.semaphore_take %17 : memref<64x64xf16> -> memref<64x64xf16>
              %19 = loom.init_tensor %18[64, 64] : memref<64x64xf16> -> tensor<64x64xf16>
              %20 = linalg.fill ins(%cst : f16) outs(%19 : tensor<64x64xf16>) -> tensor<64x64xf16>
              %21 = affine.for %arg7 = 0 to 8 iter_args(%arg8 = %20) -> (tensor<64x64xf16>) {
                %25 = arith.muli %12, %c64 : index
                %26 = arith.muli %arg7, %c64 : index
                %27 = affine.apply affine_map<(d0, d1) -> (d0 * 512 + d1)>(%25, %26)
                %reinterpret_cast_0 = memref.reinterpret_cast %arg0 to offset: [%27], sizes: [64, 64], strides: [512, 1] : memref<4096x512xf16> to memref<64x64xf16, strided<[512, 1], offset: ?>>
                %28 = loom.copy_to_tensor %reinterpret_cast_0, %16 on @L1, interconnect : [], broadcast : [1, 1] : memref<64x64xf16, strided<[512, 1], offset: ?>>, memref<64x64xf16> -> tensor<64x64xf16>
                %29 = arith.muli %arg6, %c64 : index
                %30 = affine.apply affine_map<(d0, d1) -> (d0 * 4096 + d1)>(%26, %29)
                %reinterpret_cast_1 = memref.reinterpret_cast %arg1 to offset: [%30], sizes: [64, 64], strides: [4096, 1] : memref<512x4096xf16> to memref<64x64xf16, strided<[4096, 1], offset: ?>>
                %31 = loom.copy_to_tensor %reinterpret_cast_1, %14 on @L1, interconnect : [@vertical_links], broadcast : [8, 1] : memref<64x64xf16, strided<[4096, 1], offset: ?>>, memref<64x64xf16> -> tensor<64x64xf16>
                %32 = linalg.matmul ins(%28, %31 : tensor<64x64xf16>, tensor<64x64xf16>) outs(%arg8 : tensor<64x64xf16>) -> tensor<64x64xf16>
                loom.semaphore_give %14 : memref<64x64xf16>
                loom.semaphore_give %16 : memref<64x64xf16>
                affine.yield %32 : tensor<64x64xf16>
              }
              %22 = arith.muli %12, %c64 : index
              %23 = arith.muli %arg6, %c64 : index
              %24 = affine.apply affine_map<(d0, d1) -> (d0 * 4096 + d1)>(%22, %23)
              %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%24], sizes: [64, 64], strides: [4096, 1] : memref<4096x4096xf16> to memref<64x64xf16, strided<[4096, 1], offset: ?>>
              loom.copy_from_tensor %21, %reinterpret_cast on @DRAM : tensor<64x64xf16>, memref<64x64xf16, strided<[4096, 1], offset: ?>>
              loom.semaphore_give %18 : memref<64x64xf16>
            } {loom.iter_type = #loom.iter_type<temporal>}
          } {loom.iter_type = #loom.iter_type<temporal>}
        } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @x}
      } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @y}
      return
    }
  }
  module attributes {loom.pass_name = "Materialize"} {
    func.func @matmul__d1i0_d0i0__f10__d_d__BK64__BM64__BN64(%arg0: memref<4096x512xf16>, %arg1: memref<512x4096xf16>, %arg2: memref<4096x4096xf16>) {
      %cst = arith.constant 0.000000e+00 : f16
      %c64 = arith.constant 64 : index
      affine.parallel (%arg3) = (0) to (8) {
        affine.parallel (%arg4) = (0) to (8) {
          affine.for %arg5 = 0 to 64 {
            affine.for %arg6 = 0 to 1 {
              %12 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 8 + d1 + d2 * 64)>(%arg3, %arg4, %arg6)
              %13 = loom.alloc [64, 64] on @L1 : memref<64x64xf16>
              %14 = loom.semaphore_take %13 : memref<64x64xf16> -> memref<64x64xf16>
              %15 = loom.alloc [64, 64] on @L1 : memref<64x64xf16>
              %16 = loom.semaphore_take %15 : memref<64x64xf16> -> memref<64x64xf16>
              %17 = loom.alloc [64, 64] on @L1 : memref<64x64xf16>
              %18 = loom.semaphore_take %17 : memref<64x64xf16> -> memref<64x64xf16>
              %19 = loom.init_tensor %18[64, 64] : memref<64x64xf16> -> tensor<64x64xf16>
              %20 = linalg.fill ins(%cst : f16) outs(%19 : tensor<64x64xf16>) -> tensor<64x64xf16>
              %21 = affine.for %arg7 = 0 to 8 iter_args(%arg8 = %20) -> (tensor<64x64xf16>) {
                %25 = arith.muli %12, %c64 : index
                %26 = arith.muli %arg7, %c64 : index
                %27 = affine.apply affine_map<(d0, d1) -> (d0 * 512 + d1)>(%25, %26)
                %reinterpret_cast_0 = memref.reinterpret_cast %arg0 to offset: [%27], sizes: [64, 64], strides: [512, 1] : memref<4096x512xf16> to memref<64x64xf16, strided<[512, 1], offset: ?>>
                %28 = loom.copy_to_tensor %reinterpret_cast_0, %16 on @L1, interconnect : [], broadcast : [1, 1] : memref<64x64xf16, strided<[512, 1], offset: ?>>, memref<64x64xf16> -> tensor<64x64xf16>
                %29 = arith.muli %arg5, %c64 : index
                %30 = affine.apply affine_map<(d0, d1) -> (d0 * 4096 + d1)>(%26, %29)
                %reinterpret_cast_1 = memref.reinterpret_cast %arg1 to offset: [%30], sizes: [64, 64], strides: [4096, 1] : memref<512x4096xf16> to memref<64x64xf16, strided<[4096, 1], offset: ?>>
                %31 = loom.copy_to_tensor %reinterpret_cast_1, %14 on @L1, interconnect : [], broadcast : [1, 1] : memref<64x64xf16, strided<[4096, 1], offset: ?>>, memref<64x64xf16> -> tensor<64x64xf16>
                %32 = linalg.matmul ins(%28, %31 : tensor<64x64xf16>, tensor<64x64xf16>) outs(%arg8 : tensor<64x64xf16>) -> tensor<64x64xf16>
                loom.semaphore_give %14 : memref<64x64xf16>
                loom.semaphore_give %16 : memref<64x64xf16>
                affine.yield %32 : tensor<64x64xf16>
              }
              %22 = arith.muli %12, %c64 : index
              %23 = arith.muli %arg5, %c64 : index
              %24 = affine.apply affine_map<(d0, d1) -> (d0 * 4096 + d1)>(%22, %23)
              %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%24], sizes: [64, 64], strides: [4096, 1] : memref<4096x4096xf16> to memref<64x64xf16, strided<[4096, 1], offset: ?>>
              loom.copy_from_tensor %21, %reinterpret_cast on @DRAM : tensor<64x64xf16>, memref<64x64xf16, strided<[4096, 1], offset: ?>>
              loom.semaphore_give %18 : memref<64x64xf16>
            } {loom.iter_type = #loom.iter_type<temporal>}
          } {loom.iter_type = #loom.iter_type<temporal>}
        } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @x}
      } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @y}
      return
    }
  }
  module attributes {loom.pass_name = "Materialize"} {
    func.func @matmul__d1i0_d0i0__f10__d_a__BK64__BM64__BN64(%arg0: memref<4096x512xf16>, %arg1: memref<512x4096xf16>, %arg2: memref<4096x4096xf16>) {
      %cst = arith.constant 0.000000e+00 : f16
      %c64 = arith.constant 64 : index
      affine.parallel (%arg3) = (0) to (8) {
        affine.parallel (%arg4) = (0) to (8) {
          affine.for %arg5 = 0 to 64 {
            affine.for %arg6 = 0 to 1 {
              %12 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 8 + d1 + d2 * 64)>(%arg3, %arg4, %arg6)
              %13 = loom.alloc [64, 64] on @L1 : memref<64x64xf16>
              %14 = loom.semaphore_take %13 : memref<64x64xf16> -> memref<64x64xf16>
              %15 = loom.alloc [64, 64] on @L1 : memref<64x64xf16>
              %16 = loom.semaphore_take %15 : memref<64x64xf16> -> memref<64x64xf16>
              %17 = loom.alloc [64, 64] on @L1 : memref<64x64xf16>
              %18 = loom.semaphore_take %17 : memref<64x64xf16> -> memref<64x64xf16>
              %19 = loom.init_tensor %18[64, 64] : memref<64x64xf16> -> tensor<64x64xf16>
              %20 = linalg.fill ins(%cst : f16) outs(%19 : tensor<64x64xf16>) -> tensor<64x64xf16>
              %21 = affine.for %arg7 = 0 to 8 iter_args(%arg8 = %20) -> (tensor<64x64xf16>) {
                %25 = arith.muli %12, %c64 : index
                %26 = arith.muli %arg7, %c64 : index
                %27 = affine.apply affine_map<(d0, d1) -> (d0 * 512 + d1)>(%25, %26)
                %reinterpret_cast_0 = memref.reinterpret_cast %arg0 to offset: [%27], sizes: [64, 64], strides: [512, 1] : memref<4096x512xf16> to memref<64x64xf16, strided<[512, 1], offset: ?>>
                %28 = loom.copy_to_tensor %reinterpret_cast_0, %16 on @L1, interconnect : [], broadcast : [1, 1] : memref<64x64xf16, strided<[512, 1], offset: ?>>, memref<64x64xf16> -> tensor<64x64xf16>
                %29 = arith.muli %arg5, %c64 : index
                %30 = affine.apply affine_map<(d0, d1) -> (d0 * 4096 + d1)>(%26, %29)
                %reinterpret_cast_1 = memref.reinterpret_cast %arg1 to offset: [%30], sizes: [64, 64], strides: [4096, 1] : memref<512x4096xf16> to memref<64x64xf16, strided<[4096, 1], offset: ?>>
                %31 = loom.copy_to_tensor %reinterpret_cast_1, %14 on @L1, interconnect : [@horizontal_links, @vertical_links], broadcast : [8, 8] : memref<64x64xf16, strided<[4096, 1], offset: ?>>, memref<64x64xf16> -> tensor<64x64xf16>
                %32 = linalg.matmul ins(%28, %31 : tensor<64x64xf16>, tensor<64x64xf16>) outs(%arg8 : tensor<64x64xf16>) -> tensor<64x64xf16>
                loom.semaphore_give %14 : memref<64x64xf16>
                loom.semaphore_give %16 : memref<64x64xf16>
                affine.yield %32 : tensor<64x64xf16>
              }
              %22 = arith.muli %12, %c64 : index
              %23 = arith.muli %arg5, %c64 : index
              %24 = affine.apply affine_map<(d0, d1) -> (d0 * 4096 + d1)>(%22, %23)
              %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%24], sizes: [64, 64], strides: [4096, 1] : memref<4096x4096xf16> to memref<64x64xf16, strided<[4096, 1], offset: ?>>
              loom.copy_from_tensor %21, %reinterpret_cast on @DRAM : tensor<64x64xf16>, memref<64x64xf16, strided<[4096, 1], offset: ?>>
              loom.semaphore_give %18 : memref<64x64xf16>
            } {loom.iter_type = #loom.iter_type<temporal>}
          } {loom.iter_type = #loom.iter_type<temporal>}
        } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @x}
      } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @y}
      return
    }
  }
  module attributes {loom.pass_name = "Materialize"} {
    func.func @matmul__d1i0_d0i0__f10__d_h__BK64__BM64__BN64(%arg0: memref<4096x512xf16>, %arg1: memref<512x4096xf16>, %arg2: memref<4096x4096xf16>) {
      %cst = arith.constant 0.000000e+00 : f16
      %c64 = arith.constant 64 : index
      affine.parallel (%arg3) = (0) to (8) {
        affine.parallel (%arg4) = (0) to (8) {
          affine.for %arg5 = 0 to 64 {
            affine.for %arg6 = 0 to 1 {
              %12 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 8 + d1 + d2 * 64)>(%arg3, %arg4, %arg6)
              %13 = loom.alloc [64, 64] on @L1 : memref<64x64xf16>
              %14 = loom.semaphore_take %13 : memref<64x64xf16> -> memref<64x64xf16>
              %15 = loom.alloc [64, 64] on @L1 : memref<64x64xf16>
              %16 = loom.semaphore_take %15 : memref<64x64xf16> -> memref<64x64xf16>
              %17 = loom.alloc [64, 64] on @L1 : memref<64x64xf16>
              %18 = loom.semaphore_take %17 : memref<64x64xf16> -> memref<64x64xf16>
              %19 = loom.init_tensor %18[64, 64] : memref<64x64xf16> -> tensor<64x64xf16>
              %20 = linalg.fill ins(%cst : f16) outs(%19 : tensor<64x64xf16>) -> tensor<64x64xf16>
              %21 = affine.for %arg7 = 0 to 8 iter_args(%arg8 = %20) -> (tensor<64x64xf16>) {
                %25 = arith.muli %12, %c64 : index
                %26 = arith.muli %arg7, %c64 : index
                %27 = affine.apply affine_map<(d0, d1) -> (d0 * 512 + d1)>(%25, %26)
                %reinterpret_cast_0 = memref.reinterpret_cast %arg0 to offset: [%27], sizes: [64, 64], strides: [512, 1] : memref<4096x512xf16> to memref<64x64xf16, strided<[512, 1], offset: ?>>
                %28 = loom.copy_to_tensor %reinterpret_cast_0, %16 on @L1, interconnect : [], broadcast : [1, 1] : memref<64x64xf16, strided<[512, 1], offset: ?>>, memref<64x64xf16> -> tensor<64x64xf16>
                %29 = arith.muli %arg5, %c64 : index
                %30 = affine.apply affine_map<(d0, d1) -> (d0 * 4096 + d1)>(%26, %29)
                %reinterpret_cast_1 = memref.reinterpret_cast %arg1 to offset: [%30], sizes: [64, 64], strides: [4096, 1] : memref<512x4096xf16> to memref<64x64xf16, strided<[4096, 1], offset: ?>>
                %31 = loom.copy_to_tensor %reinterpret_cast_1, %14 on @L1, interconnect : [@horizontal_links], broadcast : [1, 8] : memref<64x64xf16, strided<[4096, 1], offset: ?>>, memref<64x64xf16> -> tensor<64x64xf16>
                %32 = linalg.matmul ins(%28, %31 : tensor<64x64xf16>, tensor<64x64xf16>) outs(%arg8 : tensor<64x64xf16>) -> tensor<64x64xf16>
                loom.semaphore_give %14 : memref<64x64xf16>
                loom.semaphore_give %16 : memref<64x64xf16>
                affine.yield %32 : tensor<64x64xf16>
              }
              %22 = arith.muli %12, %c64 : index
              %23 = arith.muli %arg5, %c64 : index
              %24 = affine.apply affine_map<(d0, d1) -> (d0 * 4096 + d1)>(%22, %23)
              %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%24], sizes: [64, 64], strides: [4096, 1] : memref<4096x4096xf16> to memref<64x64xf16, strided<[4096, 1], offset: ?>>
              loom.copy_from_tensor %21, %reinterpret_cast on @DRAM : tensor<64x64xf16>, memref<64x64xf16, strided<[4096, 1], offset: ?>>
              loom.semaphore_give %18 : memref<64x64xf16>
            } {loom.iter_type = #loom.iter_type<temporal>}
          } {loom.iter_type = #loom.iter_type<temporal>}
        } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @x}
      } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @y}
      return
    }
  }
  module attributes {loom.pass_name = "Materialize"} {
    func.func @matmul__d1i0_d0i0__f10__d_v__BK64__BM64__BN64(%arg0: memref<4096x512xf16>, %arg1: memref<512x4096xf16>, %arg2: memref<4096x4096xf16>) {
      %cst = arith.constant 0.000000e+00 : f16
      %c64 = arith.constant 64 : index
      affine.parallel (%arg3) = (0) to (8) {
        affine.parallel (%arg4) = (0) to (8) {
          affine.for %arg5 = 0 to 64 {
            affine.for %arg6 = 0 to 1 {
              %12 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 8 + d1 + d2 * 64)>(%arg3, %arg4, %arg6)
              %13 = loom.alloc [64, 64] on @L1 : memref<64x64xf16>
              %14 = loom.semaphore_take %13 : memref<64x64xf16> -> memref<64x64xf16>
              %15 = loom.alloc [64, 64] on @L1 : memref<64x64xf16>
              %16 = loom.semaphore_take %15 : memref<64x64xf16> -> memref<64x64xf16>
              %17 = loom.alloc [64, 64] on @L1 : memref<64x64xf16>
              %18 = loom.semaphore_take %17 : memref<64x64xf16> -> memref<64x64xf16>
              %19 = loom.init_tensor %18[64, 64] : memref<64x64xf16> -> tensor<64x64xf16>
              %20 = linalg.fill ins(%cst : f16) outs(%19 : tensor<64x64xf16>) -> tensor<64x64xf16>
              %21 = affine.for %arg7 = 0 to 8 iter_args(%arg8 = %20) -> (tensor<64x64xf16>) {
                %25 = arith.muli %12, %c64 : index
                %26 = arith.muli %arg7, %c64 : index
                %27 = affine.apply affine_map<(d0, d1) -> (d0 * 512 + d1)>(%25, %26)
                %reinterpret_cast_0 = memref.reinterpret_cast %arg0 to offset: [%27], sizes: [64, 64], strides: [512, 1] : memref<4096x512xf16> to memref<64x64xf16, strided<[512, 1], offset: ?>>
                %28 = loom.copy_to_tensor %reinterpret_cast_0, %16 on @L1, interconnect : [], broadcast : [1, 1] : memref<64x64xf16, strided<[512, 1], offset: ?>>, memref<64x64xf16> -> tensor<64x64xf16>
                %29 = arith.muli %arg5, %c64 : index
                %30 = affine.apply affine_map<(d0, d1) -> (d0 * 4096 + d1)>(%26, %29)
                %reinterpret_cast_1 = memref.reinterpret_cast %arg1 to offset: [%30], sizes: [64, 64], strides: [4096, 1] : memref<512x4096xf16> to memref<64x64xf16, strided<[4096, 1], offset: ?>>
                %31 = loom.copy_to_tensor %reinterpret_cast_1, %14 on @L1, interconnect : [@vertical_links], broadcast : [8, 1] : memref<64x64xf16, strided<[4096, 1], offset: ?>>, memref<64x64xf16> -> tensor<64x64xf16>
                %32 = linalg.matmul ins(%28, %31 : tensor<64x64xf16>, tensor<64x64xf16>) outs(%arg8 : tensor<64x64xf16>) -> tensor<64x64xf16>
                loom.semaphore_give %14 : memref<64x64xf16>
                loom.semaphore_give %16 : memref<64x64xf16>
                affine.yield %32 : tensor<64x64xf16>
              }
              %22 = arith.muli %12, %c64 : index
              %23 = arith.muli %arg5, %c64 : index
              %24 = affine.apply affine_map<(d0, d1) -> (d0 * 4096 + d1)>(%22, %23)
              %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%24], sizes: [64, 64], strides: [4096, 1] : memref<4096x4096xf16> to memref<64x64xf16, strided<[4096, 1], offset: ?>>
              loom.copy_from_tensor %21, %reinterpret_cast on @DRAM : tensor<64x64xf16>, memref<64x64xf16, strided<[4096, 1], offset: ?>>
              loom.semaphore_give %18 : memref<64x64xf16>
            } {loom.iter_type = #loom.iter_type<temporal>}
          } {loom.iter_type = #loom.iter_type<temporal>}
        } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @x}
      } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @y}
      return
    }
  }
  module attributes {loom.pass_name = "Materialize"} {
    func.func @matmul__d0i0_d1i1__f01__d_d__BK64__BM64__BN64(%arg0: memref<4096x512xf16>, %arg1: memref<512x4096xf16>, %arg2: memref<4096x4096xf16>) {
      %cst = arith.constant 0.000000e+00 : f16
      %c64 = arith.constant 64 : index
      affine.parallel (%arg3) = (0) to (8) {
        affine.parallel (%arg4) = (0) to (8) {
          affine.for %arg5 = 0 to 8 {
            affine.for %arg6 = 0 to 8 {
              %12 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 8)>(%arg3, %arg5)
              %13 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 8)>(%arg4, %arg6)
              %14 = loom.alloc [64, 64] on @L1 : memref<64x64xf16>
              %15 = loom.semaphore_take %14 : memref<64x64xf16> -> memref<64x64xf16>
              %16 = loom.alloc [64, 64] on @L1 : memref<64x64xf16>
              %17 = loom.semaphore_take %16 : memref<64x64xf16> -> memref<64x64xf16>
              %18 = loom.alloc [64, 64] on @L1 : memref<64x64xf16>
              %19 = loom.semaphore_take %18 : memref<64x64xf16> -> memref<64x64xf16>
              %20 = loom.init_tensor %19[64, 64] : memref<64x64xf16> -> tensor<64x64xf16>
              %21 = linalg.fill ins(%cst : f16) outs(%20 : tensor<64x64xf16>) -> tensor<64x64xf16>
              %22 = affine.for %arg7 = 0 to 8 iter_args(%arg8 = %21) -> (tensor<64x64xf16>) {
                %26 = arith.muli %12, %c64 : index
                %27 = arith.muli %arg7, %c64 : index
                %28 = affine.apply affine_map<(d0, d1) -> (d0 * 512 + d1)>(%26, %27)
                %reinterpret_cast_0 = memref.reinterpret_cast %arg0 to offset: [%28], sizes: [64, 64], strides: [512, 1] : memref<4096x512xf16> to memref<64x64xf16, strided<[512, 1], offset: ?>>
                %29 = loom.copy_to_tensor %reinterpret_cast_0, %17 on @L1, interconnect : [], broadcast : [1, 1] : memref<64x64xf16, strided<[512, 1], offset: ?>>, memref<64x64xf16> -> tensor<64x64xf16>
                %30 = arith.muli %13, %c64 : index
                %31 = affine.apply affine_map<(d0, d1) -> (d0 * 4096 + d1)>(%27, %30)
                %reinterpret_cast_1 = memref.reinterpret_cast %arg1 to offset: [%31], sizes: [64, 64], strides: [4096, 1] : memref<512x4096xf16> to memref<64x64xf16, strided<[4096, 1], offset: ?>>
                %32 = loom.copy_to_tensor %reinterpret_cast_1, %15 on @L1, interconnect : [], broadcast : [1, 1] : memref<64x64xf16, strided<[4096, 1], offset: ?>>, memref<64x64xf16> -> tensor<64x64xf16>
                %33 = linalg.matmul ins(%29, %32 : tensor<64x64xf16>, tensor<64x64xf16>) outs(%arg8 : tensor<64x64xf16>) -> tensor<64x64xf16>
                loom.semaphore_give %15 : memref<64x64xf16>
                loom.semaphore_give %17 : memref<64x64xf16>
                affine.yield %33 : tensor<64x64xf16>
              }
              %23 = arith.muli %12, %c64 : index
              %24 = arith.muli %13, %c64 : index
              %25 = affine.apply affine_map<(d0, d1) -> (d0 * 4096 + d1)>(%23, %24)
              %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%25], sizes: [64, 64], strides: [4096, 1] : memref<4096x4096xf16> to memref<64x64xf16, strided<[4096, 1], offset: ?>>
              loom.copy_from_tensor %22, %reinterpret_cast on @DRAM : tensor<64x64xf16>, memref<64x64xf16, strided<[4096, 1], offset: ?>>
              loom.semaphore_give %19 : memref<64x64xf16>
            } {loom.iter_type = #loom.iter_type<temporal>}
          } {loom.iter_type = #loom.iter_type<temporal>}
        } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @y}
      } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @x}
      return
    }
  }
  module attributes {loom.pass_name = "Materialize"} {
    func.func @matmul__d0i0_d1i1__f01__d_h__BK64__BM64__BN64(%arg0: memref<4096x512xf16>, %arg1: memref<512x4096xf16>, %arg2: memref<4096x4096xf16>) {
      %cst = arith.constant 0.000000e+00 : f16
      %c64 = arith.constant 64 : index
      affine.parallel (%arg3) = (0) to (8) {
        affine.parallel (%arg4) = (0) to (8) {
          affine.for %arg5 = 0 to 8 {
            affine.for %arg6 = 0 to 8 {
              %12 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 8)>(%arg3, %arg5)
              %13 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 8)>(%arg4, %arg6)
              %14 = loom.alloc [64, 64] on @L1 : memref<64x64xf16>
              %15 = loom.semaphore_take %14 : memref<64x64xf16> -> memref<64x64xf16>
              %16 = loom.alloc [64, 64] on @L1 : memref<64x64xf16>
              %17 = loom.semaphore_take %16 : memref<64x64xf16> -> memref<64x64xf16>
              %18 = loom.alloc [64, 64] on @L1 : memref<64x64xf16>
              %19 = loom.semaphore_take %18 : memref<64x64xf16> -> memref<64x64xf16>
              %20 = loom.init_tensor %19[64, 64] : memref<64x64xf16> -> tensor<64x64xf16>
              %21 = linalg.fill ins(%cst : f16) outs(%20 : tensor<64x64xf16>) -> tensor<64x64xf16>
              %22 = affine.for %arg7 = 0 to 8 iter_args(%arg8 = %21) -> (tensor<64x64xf16>) {
                %26 = arith.muli %12, %c64 : index
                %27 = arith.muli %arg7, %c64 : index
                %28 = affine.apply affine_map<(d0, d1) -> (d0 * 512 + d1)>(%26, %27)
                %reinterpret_cast_0 = memref.reinterpret_cast %arg0 to offset: [%28], sizes: [64, 64], strides: [512, 1] : memref<4096x512xf16> to memref<64x64xf16, strided<[512, 1], offset: ?>>
                %29 = loom.copy_to_tensor %reinterpret_cast_0, %17 on @L1, interconnect : [], broadcast : [1, 1] : memref<64x64xf16, strided<[512, 1], offset: ?>>, memref<64x64xf16> -> tensor<64x64xf16>
                %30 = arith.muli %13, %c64 : index
                %31 = affine.apply affine_map<(d0, d1) -> (d0 * 4096 + d1)>(%27, %30)
                %reinterpret_cast_1 = memref.reinterpret_cast %arg1 to offset: [%31], sizes: [64, 64], strides: [4096, 1] : memref<512x4096xf16> to memref<64x64xf16, strided<[4096, 1], offset: ?>>
                %32 = loom.copy_to_tensor %reinterpret_cast_1, %15 on @L1, interconnect : [@horizontal_links], broadcast : [1, 8] : memref<64x64xf16, strided<[4096, 1], offset: ?>>, memref<64x64xf16> -> tensor<64x64xf16>
                %33 = linalg.matmul ins(%29, %32 : tensor<64x64xf16>, tensor<64x64xf16>) outs(%arg8 : tensor<64x64xf16>) -> tensor<64x64xf16>
                loom.semaphore_give %15 : memref<64x64xf16>
                loom.semaphore_give %17 : memref<64x64xf16>
                affine.yield %33 : tensor<64x64xf16>
              }
              %23 = arith.muli %12, %c64 : index
              %24 = arith.muli %13, %c64 : index
              %25 = affine.apply affine_map<(d0, d1) -> (d0 * 4096 + d1)>(%23, %24)
              %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%25], sizes: [64, 64], strides: [4096, 1] : memref<4096x4096xf16> to memref<64x64xf16, strided<[4096, 1], offset: ?>>
              loom.copy_from_tensor %22, %reinterpret_cast on @DRAM : tensor<64x64xf16>, memref<64x64xf16, strided<[4096, 1], offset: ?>>
              loom.semaphore_give %19 : memref<64x64xf16>
            } {loom.iter_type = #loom.iter_type<temporal>}
          } {loom.iter_type = #loom.iter_type<temporal>}
        } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @y}
      } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @x}
      return
    }
  }
  module attributes {loom.pass_name = "Materialize"} {
    func.func @matmul__d0i0_d1i1__f01__v_d__BK64__BM64__BN64(%arg0: memref<4096x512xf16>, %arg1: memref<512x4096xf16>, %arg2: memref<4096x4096xf16>) {
      %cst = arith.constant 0.000000e+00 : f16
      %c64 = arith.constant 64 : index
      affine.parallel (%arg3) = (0) to (8) {
        affine.parallel (%arg4) = (0) to (8) {
          affine.for %arg5 = 0 to 8 {
            affine.for %arg6 = 0 to 8 {
              %12 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 8)>(%arg3, %arg5)
              %13 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 8)>(%arg4, %arg6)
              %14 = loom.alloc [64, 64] on @L1 : memref<64x64xf16>
              %15 = loom.semaphore_take %14 : memref<64x64xf16> -> memref<64x64xf16>
              %16 = loom.alloc [64, 64] on @L1 : memref<64x64xf16>
              %17 = loom.semaphore_take %16 : memref<64x64xf16> -> memref<64x64xf16>
              %18 = loom.alloc [64, 64] on @L1 : memref<64x64xf16>
              %19 = loom.semaphore_take %18 : memref<64x64xf16> -> memref<64x64xf16>
              %20 = loom.init_tensor %19[64, 64] : memref<64x64xf16> -> tensor<64x64xf16>
              %21 = linalg.fill ins(%cst : f16) outs(%20 : tensor<64x64xf16>) -> tensor<64x64xf16>
              %22 = affine.for %arg7 = 0 to 8 iter_args(%arg8 = %21) -> (tensor<64x64xf16>) {
                %26 = arith.muli %12, %c64 : index
                %27 = arith.muli %arg7, %c64 : index
                %28 = affine.apply affine_map<(d0, d1) -> (d0 * 512 + d1)>(%26, %27)
                %reinterpret_cast_0 = memref.reinterpret_cast %arg0 to offset: [%28], sizes: [64, 64], strides: [512, 1] : memref<4096x512xf16> to memref<64x64xf16, strided<[512, 1], offset: ?>>
                %29 = loom.copy_to_tensor %reinterpret_cast_0, %17 on @L1, interconnect : [@vertical_links], broadcast : [8, 1] : memref<64x64xf16, strided<[512, 1], offset: ?>>, memref<64x64xf16> -> tensor<64x64xf16>
                %30 = arith.muli %13, %c64 : index
                %31 = affine.apply affine_map<(d0, d1) -> (d0 * 4096 + d1)>(%27, %30)
                %reinterpret_cast_1 = memref.reinterpret_cast %arg1 to offset: [%31], sizes: [64, 64], strides: [4096, 1] : memref<512x4096xf16> to memref<64x64xf16, strided<[4096, 1], offset: ?>>
                %32 = loom.copy_to_tensor %reinterpret_cast_1, %15 on @L1, interconnect : [], broadcast : [1, 1] : memref<64x64xf16, strided<[4096, 1], offset: ?>>, memref<64x64xf16> -> tensor<64x64xf16>
                %33 = linalg.matmul ins(%29, %32 : tensor<64x64xf16>, tensor<64x64xf16>) outs(%arg8 : tensor<64x64xf16>) -> tensor<64x64xf16>
                loom.semaphore_give %15 : memref<64x64xf16>
                loom.semaphore_give %17 : memref<64x64xf16>
                affine.yield %33 : tensor<64x64xf16>
              }
              %23 = arith.muli %12, %c64 : index
              %24 = arith.muli %13, %c64 : index
              %25 = affine.apply affine_map<(d0, d1) -> (d0 * 4096 + d1)>(%23, %24)
              %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%25], sizes: [64, 64], strides: [4096, 1] : memref<4096x4096xf16> to memref<64x64xf16, strided<[4096, 1], offset: ?>>
              loom.copy_from_tensor %22, %reinterpret_cast on @DRAM : tensor<64x64xf16>, memref<64x64xf16, strided<[4096, 1], offset: ?>>
              loom.semaphore_give %19 : memref<64x64xf16>
            } {loom.iter_type = #loom.iter_type<temporal>}
          } {loom.iter_type = #loom.iter_type<temporal>}
        } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @y}
      } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @x}
      return
    }
  }
  module attributes {loom.pass_name = "Materialize"} {
    func.func @matmul__d0i0_d1i1__f01__v_h__BK64__BM64__BN64(%arg0: memref<4096x512xf16>, %arg1: memref<512x4096xf16>, %arg2: memref<4096x4096xf16>) {
      %cst = arith.constant 0.000000e+00 : f16
      %c64 = arith.constant 64 : index
      affine.parallel (%arg3) = (0) to (8) {
        affine.parallel (%arg4) = (0) to (8) {
          affine.for %arg5 = 0 to 8 {
            affine.for %arg6 = 0 to 8 {
              %12 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 8)>(%arg3, %arg5)
              %13 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 8)>(%arg4, %arg6)
              %14 = loom.alloc [64, 64] on @L1 : memref<64x64xf16>
              %15 = loom.semaphore_take %14 : memref<64x64xf16> -> memref<64x64xf16>
              %16 = loom.alloc [64, 64] on @L1 : memref<64x64xf16>
              %17 = loom.semaphore_take %16 : memref<64x64xf16> -> memref<64x64xf16>
              %18 = loom.alloc [64, 64] on @L1 : memref<64x64xf16>
              %19 = loom.semaphore_take %18 : memref<64x64xf16> -> memref<64x64xf16>
              %20 = loom.init_tensor %19[64, 64] : memref<64x64xf16> -> tensor<64x64xf16>
              %21 = linalg.fill ins(%cst : f16) outs(%20 : tensor<64x64xf16>) -> tensor<64x64xf16>
              %22 = affine.for %arg7 = 0 to 8 iter_args(%arg8 = %21) -> (tensor<64x64xf16>) {
                %26 = arith.muli %12, %c64 : index
                %27 = arith.muli %arg7, %c64 : index
                %28 = affine.apply affine_map<(d0, d1) -> (d0 * 512 + d1)>(%26, %27)
                %reinterpret_cast_0 = memref.reinterpret_cast %arg0 to offset: [%28], sizes: [64, 64], strides: [512, 1] : memref<4096x512xf16> to memref<64x64xf16, strided<[512, 1], offset: ?>>
                %29 = loom.copy_to_tensor %reinterpret_cast_0, %17 on @L1, interconnect : [@vertical_links], broadcast : [8, 1] : memref<64x64xf16, strided<[512, 1], offset: ?>>, memref<64x64xf16> -> tensor<64x64xf16>
                %30 = arith.muli %13, %c64 : index
                %31 = affine.apply affine_map<(d0, d1) -> (d0 * 4096 + d1)>(%27, %30)
                %reinterpret_cast_1 = memref.reinterpret_cast %arg1 to offset: [%31], sizes: [64, 64], strides: [4096, 1] : memref<512x4096xf16> to memref<64x64xf16, strided<[4096, 1], offset: ?>>
                %32 = loom.copy_to_tensor %reinterpret_cast_1, %15 on @L1, interconnect : [@horizontal_links], broadcast : [1, 8] : memref<64x64xf16, strided<[4096, 1], offset: ?>>, memref<64x64xf16> -> tensor<64x64xf16>
                %33 = linalg.matmul ins(%29, %32 : tensor<64x64xf16>, tensor<64x64xf16>) outs(%arg8 : tensor<64x64xf16>) -> tensor<64x64xf16>
                loom.semaphore_give %15 : memref<64x64xf16>
                loom.semaphore_give %17 : memref<64x64xf16>
                affine.yield %33 : tensor<64x64xf16>
              }
              %23 = arith.muli %12, %c64 : index
              %24 = arith.muli %13, %c64 : index
              %25 = affine.apply affine_map<(d0, d1) -> (d0 * 4096 + d1)>(%23, %24)
              %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%25], sizes: [64, 64], strides: [4096, 1] : memref<4096x4096xf16> to memref<64x64xf16, strided<[4096, 1], offset: ?>>
              loom.copy_from_tensor %22, %reinterpret_cast on @DRAM : tensor<64x64xf16>, memref<64x64xf16, strided<[4096, 1], offset: ?>>
              loom.semaphore_give %19 : memref<64x64xf16>
            } {loom.iter_type = #loom.iter_type<temporal>}
          } {loom.iter_type = #loom.iter_type<temporal>}
        } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @y}
      } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @x}
      return
    }
  }
  module attributes {loom.pass_name = "Materialize"} {
    func.func @matmul__d0i0_d1i1__f10__d_d__BK64__BM64__BN64(%arg0: memref<4096x512xf16>, %arg1: memref<512x4096xf16>, %arg2: memref<4096x4096xf16>) {
      %cst = arith.constant 0.000000e+00 : f16
      %c64 = arith.constant 64 : index
      affine.parallel (%arg3) = (0) to (8) {
        affine.parallel (%arg4) = (0) to (8) {
          affine.for %arg5 = 0 to 8 {
            affine.for %arg6 = 0 to 8 {
              %12 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 8)>(%arg3, %arg6)
              %13 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 8)>(%arg4, %arg5)
              %14 = loom.alloc [64, 64] on @L1 : memref<64x64xf16>
              %15 = loom.semaphore_take %14 : memref<64x64xf16> -> memref<64x64xf16>
              %16 = loom.alloc [64, 64] on @L1 : memref<64x64xf16>
              %17 = loom.semaphore_take %16 : memref<64x64xf16> -> memref<64x64xf16>
              %18 = loom.alloc [64, 64] on @L1 : memref<64x64xf16>
              %19 = loom.semaphore_take %18 : memref<64x64xf16> -> memref<64x64xf16>
              %20 = loom.init_tensor %19[64, 64] : memref<64x64xf16> -> tensor<64x64xf16>
              %21 = linalg.fill ins(%cst : f16) outs(%20 : tensor<64x64xf16>) -> tensor<64x64xf16>
              %22 = affine.for %arg7 = 0 to 8 iter_args(%arg8 = %21) -> (tensor<64x64xf16>) {
                %26 = arith.muli %12, %c64 : index
                %27 = arith.muli %arg7, %c64 : index
                %28 = affine.apply affine_map<(d0, d1) -> (d0 * 512 + d1)>(%26, %27)
                %reinterpret_cast_0 = memref.reinterpret_cast %arg0 to offset: [%28], sizes: [64, 64], strides: [512, 1] : memref<4096x512xf16> to memref<64x64xf16, strided<[512, 1], offset: ?>>
                %29 = loom.copy_to_tensor %reinterpret_cast_0, %17 on @L1, interconnect : [], broadcast : [1, 1] : memref<64x64xf16, strided<[512, 1], offset: ?>>, memref<64x64xf16> -> tensor<64x64xf16>
                %30 = arith.muli %13, %c64 : index
                %31 = affine.apply affine_map<(d0, d1) -> (d0 * 4096 + d1)>(%27, %30)
                %reinterpret_cast_1 = memref.reinterpret_cast %arg1 to offset: [%31], sizes: [64, 64], strides: [4096, 1] : memref<512x4096xf16> to memref<64x64xf16, strided<[4096, 1], offset: ?>>
                %32 = loom.copy_to_tensor %reinterpret_cast_1, %15 on @L1, interconnect : [], broadcast : [1, 1] : memref<64x64xf16, strided<[4096, 1], offset: ?>>, memref<64x64xf16> -> tensor<64x64xf16>
                %33 = linalg.matmul ins(%29, %32 : tensor<64x64xf16>, tensor<64x64xf16>) outs(%arg8 : tensor<64x64xf16>) -> tensor<64x64xf16>
                loom.semaphore_give %15 : memref<64x64xf16>
                loom.semaphore_give %17 : memref<64x64xf16>
                affine.yield %33 : tensor<64x64xf16>
              }
              %23 = arith.muli %12, %c64 : index
              %24 = arith.muli %13, %c64 : index
              %25 = affine.apply affine_map<(d0, d1) -> (d0 * 4096 + d1)>(%23, %24)
              %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%25], sizes: [64, 64], strides: [4096, 1] : memref<4096x4096xf16> to memref<64x64xf16, strided<[4096, 1], offset: ?>>
              loom.copy_from_tensor %22, %reinterpret_cast on @DRAM : tensor<64x64xf16>, memref<64x64xf16, strided<[4096, 1], offset: ?>>
              loom.semaphore_give %19 : memref<64x64xf16>
            } {loom.iter_type = #loom.iter_type<temporal>}
          } {loom.iter_type = #loom.iter_type<temporal>}
        } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @y}
      } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @x}
      return
    }
  }
  module attributes {loom.pass_name = "Materialize"} {
    func.func @matmul__d0i0_d1i1__f10__d_h__BK64__BM64__BN64(%arg0: memref<4096x512xf16>, %arg1: memref<512x4096xf16>, %arg2: memref<4096x4096xf16>) {
      %cst = arith.constant 0.000000e+00 : f16
      %c64 = arith.constant 64 : index
      affine.parallel (%arg3) = (0) to (8) {
        affine.parallel (%arg4) = (0) to (8) {
          affine.for %arg5 = 0 to 8 {
            affine.for %arg6 = 0 to 8 {
              %12 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 8)>(%arg3, %arg6)
              %13 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 8)>(%arg4, %arg5)
              %14 = loom.alloc [64, 64] on @L1 : memref<64x64xf16>
              %15 = loom.semaphore_take %14 : memref<64x64xf16> -> memref<64x64xf16>
              %16 = loom.alloc [64, 64] on @L1 : memref<64x64xf16>
              %17 = loom.semaphore_take %16 : memref<64x64xf16> -> memref<64x64xf16>
              %18 = loom.alloc [64, 64] on @L1 : memref<64x64xf16>
              %19 = loom.semaphore_take %18 : memref<64x64xf16> -> memref<64x64xf16>
              %20 = loom.init_tensor %19[64, 64] : memref<64x64xf16> -> tensor<64x64xf16>
              %21 = linalg.fill ins(%cst : f16) outs(%20 : tensor<64x64xf16>) -> tensor<64x64xf16>
              %22 = affine.for %arg7 = 0 to 8 iter_args(%arg8 = %21) -> (tensor<64x64xf16>) {
                %26 = arith.muli %12, %c64 : index
                %27 = arith.muli %arg7, %c64 : index
                %28 = affine.apply affine_map<(d0, d1) -> (d0 * 512 + d1)>(%26, %27)
                %reinterpret_cast_0 = memref.reinterpret_cast %arg0 to offset: [%28], sizes: [64, 64], strides: [512, 1] : memref<4096x512xf16> to memref<64x64xf16, strided<[512, 1], offset: ?>>
                %29 = loom.copy_to_tensor %reinterpret_cast_0, %17 on @L1, interconnect : [], broadcast : [1, 1] : memref<64x64xf16, strided<[512, 1], offset: ?>>, memref<64x64xf16> -> tensor<64x64xf16>
                %30 = arith.muli %13, %c64 : index
                %31 = affine.apply affine_map<(d0, d1) -> (d0 * 4096 + d1)>(%27, %30)
                %reinterpret_cast_1 = memref.reinterpret_cast %arg1 to offset: [%31], sizes: [64, 64], strides: [4096, 1] : memref<512x4096xf16> to memref<64x64xf16, strided<[4096, 1], offset: ?>>
                %32 = loom.copy_to_tensor %reinterpret_cast_1, %15 on @L1, interconnect : [@horizontal_links], broadcast : [1, 8] : memref<64x64xf16, strided<[4096, 1], offset: ?>>, memref<64x64xf16> -> tensor<64x64xf16>
                %33 = linalg.matmul ins(%29, %32 : tensor<64x64xf16>, tensor<64x64xf16>) outs(%arg8 : tensor<64x64xf16>) -> tensor<64x64xf16>
                loom.semaphore_give %15 : memref<64x64xf16>
                loom.semaphore_give %17 : memref<64x64xf16>
                affine.yield %33 : tensor<64x64xf16>
              }
              %23 = arith.muli %12, %c64 : index
              %24 = arith.muli %13, %c64 : index
              %25 = affine.apply affine_map<(d0, d1) -> (d0 * 4096 + d1)>(%23, %24)
              %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%25], sizes: [64, 64], strides: [4096, 1] : memref<4096x4096xf16> to memref<64x64xf16, strided<[4096, 1], offset: ?>>
              loom.copy_from_tensor %22, %reinterpret_cast on @DRAM : tensor<64x64xf16>, memref<64x64xf16, strided<[4096, 1], offset: ?>>
              loom.semaphore_give %19 : memref<64x64xf16>
            } {loom.iter_type = #loom.iter_type<temporal>}
          } {loom.iter_type = #loom.iter_type<temporal>}
        } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @y}
      } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @x}
      return
    }
  }
  module attributes {loom.pass_name = "Materialize"} {
    func.func @matmul__d0i0_d1i1__f10__v_d__BK64__BM64__BN64(%arg0: memref<4096x512xf16>, %arg1: memref<512x4096xf16>, %arg2: memref<4096x4096xf16>) {
      %cst = arith.constant 0.000000e+00 : f16
      %c64 = arith.constant 64 : index
      affine.parallel (%arg3) = (0) to (8) {
        affine.parallel (%arg4) = (0) to (8) {
          affine.for %arg5 = 0 to 8 {
            affine.for %arg6 = 0 to 8 {
              %12 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 8)>(%arg3, %arg6)
              %13 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 8)>(%arg4, %arg5)
              %14 = loom.alloc [64, 64] on @L1 : memref<64x64xf16>
              %15 = loom.semaphore_take %14 : memref<64x64xf16> -> memref<64x64xf16>
              %16 = loom.alloc [64, 64] on @L1 : memref<64x64xf16>
              %17 = loom.semaphore_take %16 : memref<64x64xf16> -> memref<64x64xf16>
              %18 = loom.alloc [64, 64] on @L1 : memref<64x64xf16>
              %19 = loom.semaphore_take %18 : memref<64x64xf16> -> memref<64x64xf16>
              %20 = loom.init_tensor %19[64, 64] : memref<64x64xf16> -> tensor<64x64xf16>
              %21 = linalg.fill ins(%cst : f16) outs(%20 : tensor<64x64xf16>) -> tensor<64x64xf16>
              %22 = affine.for %arg7 = 0 to 8 iter_args(%arg8 = %21) -> (tensor<64x64xf16>) {
                %26 = arith.muli %12, %c64 : index
                %27 = arith.muli %arg7, %c64 : index
                %28 = affine.apply affine_map<(d0, d1) -> (d0 * 512 + d1)>(%26, %27)
                %reinterpret_cast_0 = memref.reinterpret_cast %arg0 to offset: [%28], sizes: [64, 64], strides: [512, 1] : memref<4096x512xf16> to memref<64x64xf16, strided<[512, 1], offset: ?>>
                %29 = loom.copy_to_tensor %reinterpret_cast_0, %17 on @L1, interconnect : [@vertical_links], broadcast : [8, 1] : memref<64x64xf16, strided<[512, 1], offset: ?>>, memref<64x64xf16> -> tensor<64x64xf16>
                %30 = arith.muli %13, %c64 : index
                %31 = affine.apply affine_map<(d0, d1) -> (d0 * 4096 + d1)>(%27, %30)
                %reinterpret_cast_1 = memref.reinterpret_cast %arg1 to offset: [%31], sizes: [64, 64], strides: [4096, 1] : memref<512x4096xf16> to memref<64x64xf16, strided<[4096, 1], offset: ?>>
                %32 = loom.copy_to_tensor %reinterpret_cast_1, %15 on @L1, interconnect : [], broadcast : [1, 1] : memref<64x64xf16, strided<[4096, 1], offset: ?>>, memref<64x64xf16> -> tensor<64x64xf16>
                %33 = linalg.matmul ins(%29, %32 : tensor<64x64xf16>, tensor<64x64xf16>) outs(%arg8 : tensor<64x64xf16>) -> tensor<64x64xf16>
                loom.semaphore_give %15 : memref<64x64xf16>
                loom.semaphore_give %17 : memref<64x64xf16>
                affine.yield %33 : tensor<64x64xf16>
              }
              %23 = arith.muli %12, %c64 : index
              %24 = arith.muli %13, %c64 : index
              %25 = affine.apply affine_map<(d0, d1) -> (d0 * 4096 + d1)>(%23, %24)
              %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%25], sizes: [64, 64], strides: [4096, 1] : memref<4096x4096xf16> to memref<64x64xf16, strided<[4096, 1], offset: ?>>
              loom.copy_from_tensor %22, %reinterpret_cast on @DRAM : tensor<64x64xf16>, memref<64x64xf16, strided<[4096, 1], offset: ?>>
              loom.semaphore_give %19 : memref<64x64xf16>
            } {loom.iter_type = #loom.iter_type<temporal>}
          } {loom.iter_type = #loom.iter_type<temporal>}
        } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @y}
      } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @x}
      return
    }
  }
  module attributes {loom.pass_name = "Materialize"} {
    func.func @matmul__d0i0_d1i1__f10__v_h__BK64__BM64__BN64(%arg0: memref<4096x512xf16>, %arg1: memref<512x4096xf16>, %arg2: memref<4096x4096xf16>) {
      %cst = arith.constant 0.000000e+00 : f16
      %c64 = arith.constant 64 : index
      affine.parallel (%arg3) = (0) to (8) {
        affine.parallel (%arg4) = (0) to (8) {
          affine.for %arg5 = 0 to 8 {
            affine.for %arg6 = 0 to 8 {
              %12 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 8)>(%arg3, %arg6)
              %13 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 8)>(%arg4, %arg5)
              %14 = loom.alloc [64, 64] on @L1 : memref<64x64xf16>
              %15 = loom.semaphore_take %14 : memref<64x64xf16> -> memref<64x64xf16>
              %16 = loom.alloc [64, 64] on @L1 : memref<64x64xf16>
              %17 = loom.semaphore_take %16 : memref<64x64xf16> -> memref<64x64xf16>
              %18 = loom.alloc [64, 64] on @L1 : memref<64x64xf16>
              %19 = loom.semaphore_take %18 : memref<64x64xf16> -> memref<64x64xf16>
              %20 = loom.init_tensor %19[64, 64] : memref<64x64xf16> -> tensor<64x64xf16>
              %21 = linalg.fill ins(%cst : f16) outs(%20 : tensor<64x64xf16>) -> tensor<64x64xf16>
              %22 = affine.for %arg7 = 0 to 8 iter_args(%arg8 = %21) -> (tensor<64x64xf16>) {
                %26 = arith.muli %12, %c64 : index
                %27 = arith.muli %arg7, %c64 : index
                %28 = affine.apply affine_map<(d0, d1) -> (d0 * 512 + d1)>(%26, %27)
                %reinterpret_cast_0 = memref.reinterpret_cast %arg0 to offset: [%28], sizes: [64, 64], strides: [512, 1] : memref<4096x512xf16> to memref<64x64xf16, strided<[512, 1], offset: ?>>
                %29 = loom.copy_to_tensor %reinterpret_cast_0, %17 on @L1, interconnect : [@vertical_links], broadcast : [8, 1] : memref<64x64xf16, strided<[512, 1], offset: ?>>, memref<64x64xf16> -> tensor<64x64xf16>
                %30 = arith.muli %13, %c64 : index
                %31 = affine.apply affine_map<(d0, d1) -> (d0 * 4096 + d1)>(%27, %30)
                %reinterpret_cast_1 = memref.reinterpret_cast %arg1 to offset: [%31], sizes: [64, 64], strides: [4096, 1] : memref<512x4096xf16> to memref<64x64xf16, strided<[4096, 1], offset: ?>>
                %32 = loom.copy_to_tensor %reinterpret_cast_1, %15 on @L1, interconnect : [@horizontal_links], broadcast : [1, 8] : memref<64x64xf16, strided<[4096, 1], offset: ?>>, memref<64x64xf16> -> tensor<64x64xf16>
                %33 = linalg.matmul ins(%29, %32 : tensor<64x64xf16>, tensor<64x64xf16>) outs(%arg8 : tensor<64x64xf16>) -> tensor<64x64xf16>
                loom.semaphore_give %15 : memref<64x64xf16>
                loom.semaphore_give %17 : memref<64x64xf16>
                affine.yield %33 : tensor<64x64xf16>
              }
              %23 = arith.muli %12, %c64 : index
              %24 = arith.muli %13, %c64 : index
              %25 = affine.apply affine_map<(d0, d1) -> (d0 * 4096 + d1)>(%23, %24)
              %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%25], sizes: [64, 64], strides: [4096, 1] : memref<4096x4096xf16> to memref<64x64xf16, strided<[4096, 1], offset: ?>>
              loom.copy_from_tensor %22, %reinterpret_cast on @DRAM : tensor<64x64xf16>, memref<64x64xf16, strided<[4096, 1], offset: ?>>
              loom.semaphore_give %19 : memref<64x64xf16>
            } {loom.iter_type = #loom.iter_type<temporal>}
          } {loom.iter_type = #loom.iter_type<temporal>}
        } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @y}
      } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @x}
      return
    }
  }
  module attributes {loom.pass_name = "Materialize"} {
    func.func @matmul__d1i0_d0i1__f01__d_d__BK64__BM64__BN64(%arg0: memref<4096x512xf16>, %arg1: memref<512x4096xf16>, %arg2: memref<4096x4096xf16>) {
      %cst = arith.constant 0.000000e+00 : f16
      %c64 = arith.constant 64 : index
      affine.parallel (%arg3) = (0) to (8) {
        affine.parallel (%arg4) = (0) to (8) {
          affine.for %arg5 = 0 to 8 {
            affine.for %arg6 = 0 to 8 {
              %12 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 8)>(%arg3, %arg5)
              %13 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 8)>(%arg4, %arg6)
              %14 = loom.alloc [64, 64] on @L1 : memref<64x64xf16>
              %15 = loom.semaphore_take %14 : memref<64x64xf16> -> memref<64x64xf16>
              %16 = loom.alloc [64, 64] on @L1 : memref<64x64xf16>
              %17 = loom.semaphore_take %16 : memref<64x64xf16> -> memref<64x64xf16>
              %18 = loom.alloc [64, 64] on @L1 : memref<64x64xf16>
              %19 = loom.semaphore_take %18 : memref<64x64xf16> -> memref<64x64xf16>
              %20 = loom.init_tensor %19[64, 64] : memref<64x64xf16> -> tensor<64x64xf16>
              %21 = linalg.fill ins(%cst : f16) outs(%20 : tensor<64x64xf16>) -> tensor<64x64xf16>
              %22 = affine.for %arg7 = 0 to 8 iter_args(%arg8 = %21) -> (tensor<64x64xf16>) {
                %26 = arith.muli %12, %c64 : index
                %27 = arith.muli %arg7, %c64 : index
                %28 = affine.apply affine_map<(d0, d1) -> (d0 * 512 + d1)>(%26, %27)
                %reinterpret_cast_0 = memref.reinterpret_cast %arg0 to offset: [%28], sizes: [64, 64], strides: [512, 1] : memref<4096x512xf16> to memref<64x64xf16, strided<[512, 1], offset: ?>>
                %29 = loom.copy_to_tensor %reinterpret_cast_0, %17 on @L1, interconnect : [], broadcast : [1, 1] : memref<64x64xf16, strided<[512, 1], offset: ?>>, memref<64x64xf16> -> tensor<64x64xf16>
                %30 = arith.muli %13, %c64 : index
                %31 = affine.apply affine_map<(d0, d1) -> (d0 * 4096 + d1)>(%27, %30)
                %reinterpret_cast_1 = memref.reinterpret_cast %arg1 to offset: [%31], sizes: [64, 64], strides: [4096, 1] : memref<512x4096xf16> to memref<64x64xf16, strided<[4096, 1], offset: ?>>
                %32 = loom.copy_to_tensor %reinterpret_cast_1, %15 on @L1, interconnect : [], broadcast : [1, 1] : memref<64x64xf16, strided<[4096, 1], offset: ?>>, memref<64x64xf16> -> tensor<64x64xf16>
                %33 = linalg.matmul ins(%29, %32 : tensor<64x64xf16>, tensor<64x64xf16>) outs(%arg8 : tensor<64x64xf16>) -> tensor<64x64xf16>
                loom.semaphore_give %15 : memref<64x64xf16>
                loom.semaphore_give %17 : memref<64x64xf16>
                affine.yield %33 : tensor<64x64xf16>
              }
              %23 = arith.muli %12, %c64 : index
              %24 = arith.muli %13, %c64 : index
              %25 = affine.apply affine_map<(d0, d1) -> (d0 * 4096 + d1)>(%23, %24)
              %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%25], sizes: [64, 64], strides: [4096, 1] : memref<4096x4096xf16> to memref<64x64xf16, strided<[4096, 1], offset: ?>>
              loom.copy_from_tensor %22, %reinterpret_cast on @DRAM : tensor<64x64xf16>, memref<64x64xf16, strided<[4096, 1], offset: ?>>
              loom.semaphore_give %19 : memref<64x64xf16>
            } {loom.iter_type = #loom.iter_type<temporal>}
          } {loom.iter_type = #loom.iter_type<temporal>}
        } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @x}
      } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @y}
      return
    }
  }
  module attributes {loom.pass_name = "Materialize"} {
    func.func @matmul__d1i0_d0i1__f01__d_v__BK64__BM64__BN64(%arg0: memref<4096x512xf16>, %arg1: memref<512x4096xf16>, %arg2: memref<4096x4096xf16>) {
      %cst = arith.constant 0.000000e+00 : f16
      %c64 = arith.constant 64 : index
      affine.parallel (%arg3) = (0) to (8) {
        affine.parallel (%arg4) = (0) to (8) {
          affine.for %arg5 = 0 to 8 {
            affine.for %arg6 = 0 to 8 {
              %12 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 8)>(%arg3, %arg5)
              %13 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 8)>(%arg4, %arg6)
              %14 = loom.alloc [64, 64] on @L1 : memref<64x64xf16>
              %15 = loom.semaphore_take %14 : memref<64x64xf16> -> memref<64x64xf16>
              %16 = loom.alloc [64, 64] on @L1 : memref<64x64xf16>
              %17 = loom.semaphore_take %16 : memref<64x64xf16> -> memref<64x64xf16>
              %18 = loom.alloc [64, 64] on @L1 : memref<64x64xf16>
              %19 = loom.semaphore_take %18 : memref<64x64xf16> -> memref<64x64xf16>
              %20 = loom.init_tensor %19[64, 64] : memref<64x64xf16> -> tensor<64x64xf16>
              %21 = linalg.fill ins(%cst : f16) outs(%20 : tensor<64x64xf16>) -> tensor<64x64xf16>
              %22 = affine.for %arg7 = 0 to 8 iter_args(%arg8 = %21) -> (tensor<64x64xf16>) {
                %26 = arith.muli %12, %c64 : index
                %27 = arith.muli %arg7, %c64 : index
                %28 = affine.apply affine_map<(d0, d1) -> (d0 * 512 + d1)>(%26, %27)
                %reinterpret_cast_0 = memref.reinterpret_cast %arg0 to offset: [%28], sizes: [64, 64], strides: [512, 1] : memref<4096x512xf16> to memref<64x64xf16, strided<[512, 1], offset: ?>>
                %29 = loom.copy_to_tensor %reinterpret_cast_0, %17 on @L1, interconnect : [], broadcast : [1, 1] : memref<64x64xf16, strided<[512, 1], offset: ?>>, memref<64x64xf16> -> tensor<64x64xf16>
                %30 = arith.muli %13, %c64 : index
                %31 = affine.apply affine_map<(d0, d1) -> (d0 * 4096 + d1)>(%27, %30)
                %reinterpret_cast_1 = memref.reinterpret_cast %arg1 to offset: [%31], sizes: [64, 64], strides: [4096, 1] : memref<512x4096xf16> to memref<64x64xf16, strided<[4096, 1], offset: ?>>
                %32 = loom.copy_to_tensor %reinterpret_cast_1, %15 on @L1, interconnect : [@vertical_links], broadcast : [8, 1] : memref<64x64xf16, strided<[4096, 1], offset: ?>>, memref<64x64xf16> -> tensor<64x64xf16>
                %33 = linalg.matmul ins(%29, %32 : tensor<64x64xf16>, tensor<64x64xf16>) outs(%arg8 : tensor<64x64xf16>) -> tensor<64x64xf16>
                loom.semaphore_give %15 : memref<64x64xf16>
                loom.semaphore_give %17 : memref<64x64xf16>
                affine.yield %33 : tensor<64x64xf16>
              }
              %23 = arith.muli %12, %c64 : index
              %24 = arith.muli %13, %c64 : index
              %25 = affine.apply affine_map<(d0, d1) -> (d0 * 4096 + d1)>(%23, %24)
              %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%25], sizes: [64, 64], strides: [4096, 1] : memref<4096x4096xf16> to memref<64x64xf16, strided<[4096, 1], offset: ?>>
              loom.copy_from_tensor %22, %reinterpret_cast on @DRAM : tensor<64x64xf16>, memref<64x64xf16, strided<[4096, 1], offset: ?>>
              loom.semaphore_give %19 : memref<64x64xf16>
            } {loom.iter_type = #loom.iter_type<temporal>}
          } {loom.iter_type = #loom.iter_type<temporal>}
        } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @x}
      } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @y}
      return
    }
  }
  module attributes {loom.pass_name = "Materialize"} {
    func.func @matmul__d1i0_d0i1__f01__h_d__BK64__BM64__BN64(%arg0: memref<4096x512xf16>, %arg1: memref<512x4096xf16>, %arg2: memref<4096x4096xf16>) {
      %cst = arith.constant 0.000000e+00 : f16
      %c64 = arith.constant 64 : index
      affine.parallel (%arg3) = (0) to (8) {
        affine.parallel (%arg4) = (0) to (8) {
          affine.for %arg5 = 0 to 8 {
            affine.for %arg6 = 0 to 8 {
              %12 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 8)>(%arg3, %arg5)
              %13 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 8)>(%arg4, %arg6)
              %14 = loom.alloc [64, 64] on @L1 : memref<64x64xf16>
              %15 = loom.semaphore_take %14 : memref<64x64xf16> -> memref<64x64xf16>
              %16 = loom.alloc [64, 64] on @L1 : memref<64x64xf16>
              %17 = loom.semaphore_take %16 : memref<64x64xf16> -> memref<64x64xf16>
              %18 = loom.alloc [64, 64] on @L1 : memref<64x64xf16>
              %19 = loom.semaphore_take %18 : memref<64x64xf16> -> memref<64x64xf16>
              %20 = loom.init_tensor %19[64, 64] : memref<64x64xf16> -> tensor<64x64xf16>
              %21 = linalg.fill ins(%cst : f16) outs(%20 : tensor<64x64xf16>) -> tensor<64x64xf16>
              %22 = affine.for %arg7 = 0 to 8 iter_args(%arg8 = %21) -> (tensor<64x64xf16>) {
                %26 = arith.muli %12, %c64 : index
                %27 = arith.muli %arg7, %c64 : index
                %28 = affine.apply affine_map<(d0, d1) -> (d0 * 512 + d1)>(%26, %27)
                %reinterpret_cast_0 = memref.reinterpret_cast %arg0 to offset: [%28], sizes: [64, 64], strides: [512, 1] : memref<4096x512xf16> to memref<64x64xf16, strided<[512, 1], offset: ?>>
                %29 = loom.copy_to_tensor %reinterpret_cast_0, %17 on @L1, interconnect : [@horizontal_links], broadcast : [1, 8] : memref<64x64xf16, strided<[512, 1], offset: ?>>, memref<64x64xf16> -> tensor<64x64xf16>
                %30 = arith.muli %13, %c64 : index
                %31 = affine.apply affine_map<(d0, d1) -> (d0 * 4096 + d1)>(%27, %30)
                %reinterpret_cast_1 = memref.reinterpret_cast %arg1 to offset: [%31], sizes: [64, 64], strides: [4096, 1] : memref<512x4096xf16> to memref<64x64xf16, strided<[4096, 1], offset: ?>>
                %32 = loom.copy_to_tensor %reinterpret_cast_1, %15 on @L1, interconnect : [], broadcast : [1, 1] : memref<64x64xf16, strided<[4096, 1], offset: ?>>, memref<64x64xf16> -> tensor<64x64xf16>
                %33 = linalg.matmul ins(%29, %32 : tensor<64x64xf16>, tensor<64x64xf16>) outs(%arg8 : tensor<64x64xf16>) -> tensor<64x64xf16>
                loom.semaphore_give %15 : memref<64x64xf16>
                loom.semaphore_give %17 : memref<64x64xf16>
                affine.yield %33 : tensor<64x64xf16>
              }
              %23 = arith.muli %12, %c64 : index
              %24 = arith.muli %13, %c64 : index
              %25 = affine.apply affine_map<(d0, d1) -> (d0 * 4096 + d1)>(%23, %24)
              %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%25], sizes: [64, 64], strides: [4096, 1] : memref<4096x4096xf16> to memref<64x64xf16, strided<[4096, 1], offset: ?>>
              loom.copy_from_tensor %22, %reinterpret_cast on @DRAM : tensor<64x64xf16>, memref<64x64xf16, strided<[4096, 1], offset: ?>>
              loom.semaphore_give %19 : memref<64x64xf16>
            } {loom.iter_type = #loom.iter_type<temporal>}
          } {loom.iter_type = #loom.iter_type<temporal>}
        } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @x}
      } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @y}
      return
    }
  }
  module attributes {loom.pass_name = "Materialize"} {
    func.func @matmul__d1i0_d0i1__f01__h_v__BK64__BM64__BN64(%arg0: memref<4096x512xf16>, %arg1: memref<512x4096xf16>, %arg2: memref<4096x4096xf16>) {
      %cst = arith.constant 0.000000e+00 : f16
      %c64 = arith.constant 64 : index
      affine.parallel (%arg3) = (0) to (8) {
        affine.parallel (%arg4) = (0) to (8) {
          affine.for %arg5 = 0 to 8 {
            affine.for %arg6 = 0 to 8 {
              %12 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 8)>(%arg3, %arg5)
              %13 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 8)>(%arg4, %arg6)
              %14 = loom.alloc [64, 64] on @L1 : memref<64x64xf16>
              %15 = loom.semaphore_take %14 : memref<64x64xf16> -> memref<64x64xf16>
              %16 = loom.alloc [64, 64] on @L1 : memref<64x64xf16>
              %17 = loom.semaphore_take %16 : memref<64x64xf16> -> memref<64x64xf16>
              %18 = loom.alloc [64, 64] on @L1 : memref<64x64xf16>
              %19 = loom.semaphore_take %18 : memref<64x64xf16> -> memref<64x64xf16>
              %20 = loom.init_tensor %19[64, 64] : memref<64x64xf16> -> tensor<64x64xf16>
              %21 = linalg.fill ins(%cst : f16) outs(%20 : tensor<64x64xf16>) -> tensor<64x64xf16>
              %22 = affine.for %arg7 = 0 to 8 iter_args(%arg8 = %21) -> (tensor<64x64xf16>) {
                %26 = arith.muli %12, %c64 : index
                %27 = arith.muli %arg7, %c64 : index
                %28 = affine.apply affine_map<(d0, d1) -> (d0 * 512 + d1)>(%26, %27)
                %reinterpret_cast_0 = memref.reinterpret_cast %arg0 to offset: [%28], sizes: [64, 64], strides: [512, 1] : memref<4096x512xf16> to memref<64x64xf16, strided<[512, 1], offset: ?>>
                %29 = loom.copy_to_tensor %reinterpret_cast_0, %17 on @L1, interconnect : [@horizontal_links], broadcast : [1, 8] : memref<64x64xf16, strided<[512, 1], offset: ?>>, memref<64x64xf16> -> tensor<64x64xf16>
                %30 = arith.muli %13, %c64 : index
                %31 = affine.apply affine_map<(d0, d1) -> (d0 * 4096 + d1)>(%27, %30)
                %reinterpret_cast_1 = memref.reinterpret_cast %arg1 to offset: [%31], sizes: [64, 64], strides: [4096, 1] : memref<512x4096xf16> to memref<64x64xf16, strided<[4096, 1], offset: ?>>
                %32 = loom.copy_to_tensor %reinterpret_cast_1, %15 on @L1, interconnect : [@vertical_links], broadcast : [8, 1] : memref<64x64xf16, strided<[4096, 1], offset: ?>>, memref<64x64xf16> -> tensor<64x64xf16>
                %33 = linalg.matmul ins(%29, %32 : tensor<64x64xf16>, tensor<64x64xf16>) outs(%arg8 : tensor<64x64xf16>) -> tensor<64x64xf16>
                loom.semaphore_give %15 : memref<64x64xf16>
                loom.semaphore_give %17 : memref<64x64xf16>
                affine.yield %33 : tensor<64x64xf16>
              }
              %23 = arith.muli %12, %c64 : index
              %24 = arith.muli %13, %c64 : index
              %25 = affine.apply affine_map<(d0, d1) -> (d0 * 4096 + d1)>(%23, %24)
              %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%25], sizes: [64, 64], strides: [4096, 1] : memref<4096x4096xf16> to memref<64x64xf16, strided<[4096, 1], offset: ?>>
              loom.copy_from_tensor %22, %reinterpret_cast on @DRAM : tensor<64x64xf16>, memref<64x64xf16, strided<[4096, 1], offset: ?>>
              loom.semaphore_give %19 : memref<64x64xf16>
            } {loom.iter_type = #loom.iter_type<temporal>}
          } {loom.iter_type = #loom.iter_type<temporal>}
        } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @x}
      } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @y}
      return
    }
  }
  module attributes {loom.pass_name = "Materialize"} {
    func.func @matmul__d1i0_d0i1__f10__d_d__BK64__BM64__BN64(%arg0: memref<4096x512xf16>, %arg1: memref<512x4096xf16>, %arg2: memref<4096x4096xf16>) {
      %cst = arith.constant 0.000000e+00 : f16
      %c64 = arith.constant 64 : index
      affine.parallel (%arg3) = (0) to (8) {
        affine.parallel (%arg4) = (0) to (8) {
          affine.for %arg5 = 0 to 8 {
            affine.for %arg6 = 0 to 8 {
              %12 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 8)>(%arg3, %arg6)
              %13 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 8)>(%arg4, %arg5)
              %14 = loom.alloc [64, 64] on @L1 : memref<64x64xf16>
              %15 = loom.semaphore_take %14 : memref<64x64xf16> -> memref<64x64xf16>
              %16 = loom.alloc [64, 64] on @L1 : memref<64x64xf16>
              %17 = loom.semaphore_take %16 : memref<64x64xf16> -> memref<64x64xf16>
              %18 = loom.alloc [64, 64] on @L1 : memref<64x64xf16>
              %19 = loom.semaphore_take %18 : memref<64x64xf16> -> memref<64x64xf16>
              %20 = loom.init_tensor %19[64, 64] : memref<64x64xf16> -> tensor<64x64xf16>
              %21 = linalg.fill ins(%cst : f16) outs(%20 : tensor<64x64xf16>) -> tensor<64x64xf16>
              %22 = affine.for %arg7 = 0 to 8 iter_args(%arg8 = %21) -> (tensor<64x64xf16>) {
                %26 = arith.muli %12, %c64 : index
                %27 = arith.muli %arg7, %c64 : index
                %28 = affine.apply affine_map<(d0, d1) -> (d0 * 512 + d1)>(%26, %27)
                %reinterpret_cast_0 = memref.reinterpret_cast %arg0 to offset: [%28], sizes: [64, 64], strides: [512, 1] : memref<4096x512xf16> to memref<64x64xf16, strided<[512, 1], offset: ?>>
                %29 = loom.copy_to_tensor %reinterpret_cast_0, %17 on @L1, interconnect : [], broadcast : [1, 1] : memref<64x64xf16, strided<[512, 1], offset: ?>>, memref<64x64xf16> -> tensor<64x64xf16>
                %30 = arith.muli %13, %c64 : index
                %31 = affine.apply affine_map<(d0, d1) -> (d0 * 4096 + d1)>(%27, %30)
                %reinterpret_cast_1 = memref.reinterpret_cast %arg1 to offset: [%31], sizes: [64, 64], strides: [4096, 1] : memref<512x4096xf16> to memref<64x64xf16, strided<[4096, 1], offset: ?>>
                %32 = loom.copy_to_tensor %reinterpret_cast_1, %15 on @L1, interconnect : [], broadcast : [1, 1] : memref<64x64xf16, strided<[4096, 1], offset: ?>>, memref<64x64xf16> -> tensor<64x64xf16>
                %33 = linalg.matmul ins(%29, %32 : tensor<64x64xf16>, tensor<64x64xf16>) outs(%arg8 : tensor<64x64xf16>) -> tensor<64x64xf16>
                loom.semaphore_give %15 : memref<64x64xf16>
                loom.semaphore_give %17 : memref<64x64xf16>
                affine.yield %33 : tensor<64x64xf16>
              }
              %23 = arith.muli %12, %c64 : index
              %24 = arith.muli %13, %c64 : index
              %25 = affine.apply affine_map<(d0, d1) -> (d0 * 4096 + d1)>(%23, %24)
              %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%25], sizes: [64, 64], strides: [4096, 1] : memref<4096x4096xf16> to memref<64x64xf16, strided<[4096, 1], offset: ?>>
              loom.copy_from_tensor %22, %reinterpret_cast on @DRAM : tensor<64x64xf16>, memref<64x64xf16, strided<[4096, 1], offset: ?>>
              loom.semaphore_give %19 : memref<64x64xf16>
            } {loom.iter_type = #loom.iter_type<temporal>}
          } {loom.iter_type = #loom.iter_type<temporal>}
        } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @x}
      } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @y}
      return
    }
  }
  module attributes {loom.pass_name = "Materialize"} {
    func.func @matmul__d1i0_d0i1__f10__d_v__BK64__BM64__BN64(%arg0: memref<4096x512xf16>, %arg1: memref<512x4096xf16>, %arg2: memref<4096x4096xf16>) {
      %cst = arith.constant 0.000000e+00 : f16
      %c64 = arith.constant 64 : index
      affine.parallel (%arg3) = (0) to (8) {
        affine.parallel (%arg4) = (0) to (8) {
          affine.for %arg5 = 0 to 8 {
            affine.for %arg6 = 0 to 8 {
              %12 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 8)>(%arg3, %arg6)
              %13 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 8)>(%arg4, %arg5)
              %14 = loom.alloc [64, 64] on @L1 : memref<64x64xf16>
              %15 = loom.semaphore_take %14 : memref<64x64xf16> -> memref<64x64xf16>
              %16 = loom.alloc [64, 64] on @L1 : memref<64x64xf16>
              %17 = loom.semaphore_take %16 : memref<64x64xf16> -> memref<64x64xf16>
              %18 = loom.alloc [64, 64] on @L1 : memref<64x64xf16>
              %19 = loom.semaphore_take %18 : memref<64x64xf16> -> memref<64x64xf16>
              %20 = loom.init_tensor %19[64, 64] : memref<64x64xf16> -> tensor<64x64xf16>
              %21 = linalg.fill ins(%cst : f16) outs(%20 : tensor<64x64xf16>) -> tensor<64x64xf16>
              %22 = affine.for %arg7 = 0 to 8 iter_args(%arg8 = %21) -> (tensor<64x64xf16>) {
                %26 = arith.muli %12, %c64 : index
                %27 = arith.muli %arg7, %c64 : index
                %28 = affine.apply affine_map<(d0, d1) -> (d0 * 512 + d1)>(%26, %27)
                %reinterpret_cast_0 = memref.reinterpret_cast %arg0 to offset: [%28], sizes: [64, 64], strides: [512, 1] : memref<4096x512xf16> to memref<64x64xf16, strided<[512, 1], offset: ?>>
                %29 = loom.copy_to_tensor %reinterpret_cast_0, %17 on @L1, interconnect : [], broadcast : [1, 1] : memref<64x64xf16, strided<[512, 1], offset: ?>>, memref<64x64xf16> -> tensor<64x64xf16>
                %30 = arith.muli %13, %c64 : index
                %31 = affine.apply affine_map<(d0, d1) -> (d0 * 4096 + d1)>(%27, %30)
                %reinterpret_cast_1 = memref.reinterpret_cast %arg1 to offset: [%31], sizes: [64, 64], strides: [4096, 1] : memref<512x4096xf16> to memref<64x64xf16, strided<[4096, 1], offset: ?>>
                %32 = loom.copy_to_tensor %reinterpret_cast_1, %15 on @L1, interconnect : [@vertical_links], broadcast : [8, 1] : memref<64x64xf16, strided<[4096, 1], offset: ?>>, memref<64x64xf16> -> tensor<64x64xf16>
                %33 = linalg.matmul ins(%29, %32 : tensor<64x64xf16>, tensor<64x64xf16>) outs(%arg8 : tensor<64x64xf16>) -> tensor<64x64xf16>
                loom.semaphore_give %15 : memref<64x64xf16>
                loom.semaphore_give %17 : memref<64x64xf16>
                affine.yield %33 : tensor<64x64xf16>
              }
              %23 = arith.muli %12, %c64 : index
              %24 = arith.muli %13, %c64 : index
              %25 = affine.apply affine_map<(d0, d1) -> (d0 * 4096 + d1)>(%23, %24)
              %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%25], sizes: [64, 64], strides: [4096, 1] : memref<4096x4096xf16> to memref<64x64xf16, strided<[4096, 1], offset: ?>>
              loom.copy_from_tensor %22, %reinterpret_cast on @DRAM : tensor<64x64xf16>, memref<64x64xf16, strided<[4096, 1], offset: ?>>
              loom.semaphore_give %19 : memref<64x64xf16>
            } {loom.iter_type = #loom.iter_type<temporal>}
          } {loom.iter_type = #loom.iter_type<temporal>}
        } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @x}
      } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @y}
      return
    }
  }
  module attributes {loom.pass_name = "Materialize"} {
    func.func @matmul__d1i0_d0i1__f10__h_d__BK64__BM64__BN64(%arg0: memref<4096x512xf16>, %arg1: memref<512x4096xf16>, %arg2: memref<4096x4096xf16>) {
      %cst = arith.constant 0.000000e+00 : f16
      %c64 = arith.constant 64 : index
      affine.parallel (%arg3) = (0) to (8) {
        affine.parallel (%arg4) = (0) to (8) {
          affine.for %arg5 = 0 to 8 {
            affine.for %arg6 = 0 to 8 {
              %12 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 8)>(%arg3, %arg6)
              %13 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 8)>(%arg4, %arg5)
              %14 = loom.alloc [64, 64] on @L1 : memref<64x64xf16>
              %15 = loom.semaphore_take %14 : memref<64x64xf16> -> memref<64x64xf16>
              %16 = loom.alloc [64, 64] on @L1 : memref<64x64xf16>
              %17 = loom.semaphore_take %16 : memref<64x64xf16> -> memref<64x64xf16>
              %18 = loom.alloc [64, 64] on @L1 : memref<64x64xf16>
              %19 = loom.semaphore_take %18 : memref<64x64xf16> -> memref<64x64xf16>
              %20 = loom.init_tensor %19[64, 64] : memref<64x64xf16> -> tensor<64x64xf16>
              %21 = linalg.fill ins(%cst : f16) outs(%20 : tensor<64x64xf16>) -> tensor<64x64xf16>
              %22 = affine.for %arg7 = 0 to 8 iter_args(%arg8 = %21) -> (tensor<64x64xf16>) {
                %26 = arith.muli %12, %c64 : index
                %27 = arith.muli %arg7, %c64 : index
                %28 = affine.apply affine_map<(d0, d1) -> (d0 * 512 + d1)>(%26, %27)
                %reinterpret_cast_0 = memref.reinterpret_cast %arg0 to offset: [%28], sizes: [64, 64], strides: [512, 1] : memref<4096x512xf16> to memref<64x64xf16, strided<[512, 1], offset: ?>>
                %29 = loom.copy_to_tensor %reinterpret_cast_0, %17 on @L1, interconnect : [@horizontal_links], broadcast : [1, 8] : memref<64x64xf16, strided<[512, 1], offset: ?>>, memref<64x64xf16> -> tensor<64x64xf16>
                %30 = arith.muli %13, %c64 : index
                %31 = affine.apply affine_map<(d0, d1) -> (d0 * 4096 + d1)>(%27, %30)
                %reinterpret_cast_1 = memref.reinterpret_cast %arg1 to offset: [%31], sizes: [64, 64], strides: [4096, 1] : memref<512x4096xf16> to memref<64x64xf16, strided<[4096, 1], offset: ?>>
                %32 = loom.copy_to_tensor %reinterpret_cast_1, %15 on @L1, interconnect : [], broadcast : [1, 1] : memref<64x64xf16, strided<[4096, 1], offset: ?>>, memref<64x64xf16> -> tensor<64x64xf16>
                %33 = linalg.matmul ins(%29, %32 : tensor<64x64xf16>, tensor<64x64xf16>) outs(%arg8 : tensor<64x64xf16>) -> tensor<64x64xf16>
                loom.semaphore_give %15 : memref<64x64xf16>
                loom.semaphore_give %17 : memref<64x64xf16>
                affine.yield %33 : tensor<64x64xf16>
              }
              %23 = arith.muli %12, %c64 : index
              %24 = arith.muli %13, %c64 : index
              %25 = affine.apply affine_map<(d0, d1) -> (d0 * 4096 + d1)>(%23, %24)
              %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%25], sizes: [64, 64], strides: [4096, 1] : memref<4096x4096xf16> to memref<64x64xf16, strided<[4096, 1], offset: ?>>
              loom.copy_from_tensor %22, %reinterpret_cast on @DRAM : tensor<64x64xf16>, memref<64x64xf16, strided<[4096, 1], offset: ?>>
              loom.semaphore_give %19 : memref<64x64xf16>
            } {loom.iter_type = #loom.iter_type<temporal>}
          } {loom.iter_type = #loom.iter_type<temporal>}
        } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @x}
      } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @y}
      return
    }
  }
  module attributes {loom.pass_name = "Materialize"} {
    func.func @matmul__d1i0_d0i1__f10__h_v__BK64__BM64__BN64(%arg0: memref<4096x512xf16>, %arg1: memref<512x4096xf16>, %arg2: memref<4096x4096xf16>) {
      %cst = arith.constant 0.000000e+00 : f16
      %c64 = arith.constant 64 : index
      affine.parallel (%arg3) = (0) to (8) {
        affine.parallel (%arg4) = (0) to (8) {
          affine.for %arg5 = 0 to 8 {
            affine.for %arg6 = 0 to 8 {
              %12 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 8)>(%arg3, %arg6)
              %13 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 8)>(%arg4, %arg5)
              %14 = loom.alloc [64, 64] on @L1 : memref<64x64xf16>
              %15 = loom.semaphore_take %14 : memref<64x64xf16> -> memref<64x64xf16>
              %16 = loom.alloc [64, 64] on @L1 : memref<64x64xf16>
              %17 = loom.semaphore_take %16 : memref<64x64xf16> -> memref<64x64xf16>
              %18 = loom.alloc [64, 64] on @L1 : memref<64x64xf16>
              %19 = loom.semaphore_take %18 : memref<64x64xf16> -> memref<64x64xf16>
              %20 = loom.init_tensor %19[64, 64] : memref<64x64xf16> -> tensor<64x64xf16>
              %21 = linalg.fill ins(%cst : f16) outs(%20 : tensor<64x64xf16>) -> tensor<64x64xf16>
              %22 = affine.for %arg7 = 0 to 8 iter_args(%arg8 = %21) -> (tensor<64x64xf16>) {
                %26 = arith.muli %12, %c64 : index
                %27 = arith.muli %arg7, %c64 : index
                %28 = affine.apply affine_map<(d0, d1) -> (d0 * 512 + d1)>(%26, %27)
                %reinterpret_cast_0 = memref.reinterpret_cast %arg0 to offset: [%28], sizes: [64, 64], strides: [512, 1] : memref<4096x512xf16> to memref<64x64xf16, strided<[512, 1], offset: ?>>
                %29 = loom.copy_to_tensor %reinterpret_cast_0, %17 on @L1, interconnect : [@horizontal_links], broadcast : [1, 8] : memref<64x64xf16, strided<[512, 1], offset: ?>>, memref<64x64xf16> -> tensor<64x64xf16>
                %30 = arith.muli %13, %c64 : index
                %31 = affine.apply affine_map<(d0, d1) -> (d0 * 4096 + d1)>(%27, %30)
                %reinterpret_cast_1 = memref.reinterpret_cast %arg1 to offset: [%31], sizes: [64, 64], strides: [4096, 1] : memref<512x4096xf16> to memref<64x64xf16, strided<[4096, 1], offset: ?>>
                %32 = loom.copy_to_tensor %reinterpret_cast_1, %15 on @L1, interconnect : [@vertical_links], broadcast : [8, 1] : memref<64x64xf16, strided<[4096, 1], offset: ?>>, memref<64x64xf16> -> tensor<64x64xf16>
                %33 = linalg.matmul ins(%29, %32 : tensor<64x64xf16>, tensor<64x64xf16>) outs(%arg8 : tensor<64x64xf16>) -> tensor<64x64xf16>
                loom.semaphore_give %15 : memref<64x64xf16>
                loom.semaphore_give %17 : memref<64x64xf16>
                affine.yield %33 : tensor<64x64xf16>
              }
              %23 = arith.muli %12, %c64 : index
              %24 = arith.muli %13, %c64 : index
              %25 = affine.apply affine_map<(d0, d1) -> (d0 * 4096 + d1)>(%23, %24)
              %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%25], sizes: [64, 64], strides: [4096, 1] : memref<4096x4096xf16> to memref<64x64xf16, strided<[4096, 1], offset: ?>>
              loom.copy_from_tensor %22, %reinterpret_cast on @DRAM : tensor<64x64xf16>, memref<64x64xf16, strided<[4096, 1], offset: ?>>
              loom.semaphore_give %19 : memref<64x64xf16>
            } {loom.iter_type = #loom.iter_type<temporal>}
          } {loom.iter_type = #loom.iter_type<temporal>}
        } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @x}
      } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @y}
      return
    }
  }
  module attributes {loom.pass_name = "Materialize"} {
    func.func @matmul__d0i1_d1i1__f01__d_d__BK64__BM64__BN64(%arg0: memref<4096x512xf16>, %arg1: memref<512x4096xf16>, %arg2: memref<4096x4096xf16>) {
      %cst = arith.constant 0.000000e+00 : f16
      %c64 = arith.constant 64 : index
      affine.parallel (%arg3) = (0) to (8) {
        affine.parallel (%arg4) = (0) to (8) {
          affine.for %arg5 = 0 to 64 {
            affine.for %arg6 = 0 to 1 {
              %12 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 8 + d1 + d2 * 64)>(%arg3, %arg4, %arg6)
              %13 = loom.alloc [64, 64] on @L1 : memref<64x64xf16>
              %14 = loom.semaphore_take %13 : memref<64x64xf16> -> memref<64x64xf16>
              %15 = loom.alloc [64, 64] on @L1 : memref<64x64xf16>
              %16 = loom.semaphore_take %15 : memref<64x64xf16> -> memref<64x64xf16>
              %17 = loom.alloc [64, 64] on @L1 : memref<64x64xf16>
              %18 = loom.semaphore_take %17 : memref<64x64xf16> -> memref<64x64xf16>
              %19 = loom.init_tensor %18[64, 64] : memref<64x64xf16> -> tensor<64x64xf16>
              %20 = linalg.fill ins(%cst : f16) outs(%19 : tensor<64x64xf16>) -> tensor<64x64xf16>
              %21 = affine.for %arg7 = 0 to 8 iter_args(%arg8 = %20) -> (tensor<64x64xf16>) {
                %25 = arith.muli %arg5, %c64 : index
                %26 = arith.muli %arg7, %c64 : index
                %27 = affine.apply affine_map<(d0, d1) -> (d0 * 512 + d1)>(%25, %26)
                %reinterpret_cast_0 = memref.reinterpret_cast %arg0 to offset: [%27], sizes: [64, 64], strides: [512, 1] : memref<4096x512xf16> to memref<64x64xf16, strided<[512, 1], offset: ?>>
                %28 = loom.copy_to_tensor %reinterpret_cast_0, %16 on @L1, interconnect : [], broadcast : [1, 1] : memref<64x64xf16, strided<[512, 1], offset: ?>>, memref<64x64xf16> -> tensor<64x64xf16>
                %29 = arith.muli %12, %c64 : index
                %30 = affine.apply affine_map<(d0, d1) -> (d0 * 4096 + d1)>(%26, %29)
                %reinterpret_cast_1 = memref.reinterpret_cast %arg1 to offset: [%30], sizes: [64, 64], strides: [4096, 1] : memref<512x4096xf16> to memref<64x64xf16, strided<[4096, 1], offset: ?>>
                %31 = loom.copy_to_tensor %reinterpret_cast_1, %14 on @L1, interconnect : [], broadcast : [1, 1] : memref<64x64xf16, strided<[4096, 1], offset: ?>>, memref<64x64xf16> -> tensor<64x64xf16>
                %32 = linalg.matmul ins(%28, %31 : tensor<64x64xf16>, tensor<64x64xf16>) outs(%arg8 : tensor<64x64xf16>) -> tensor<64x64xf16>
                loom.semaphore_give %14 : memref<64x64xf16>
                loom.semaphore_give %16 : memref<64x64xf16>
                affine.yield %32 : tensor<64x64xf16>
              }
              %22 = arith.muli %arg5, %c64 : index
              %23 = arith.muli %12, %c64 : index
              %24 = affine.apply affine_map<(d0, d1) -> (d0 * 4096 + d1)>(%22, %23)
              %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%24], sizes: [64, 64], strides: [4096, 1] : memref<4096x4096xf16> to memref<64x64xf16, strided<[4096, 1], offset: ?>>
              loom.copy_from_tensor %21, %reinterpret_cast on @DRAM : tensor<64x64xf16>, memref<64x64xf16, strided<[4096, 1], offset: ?>>
              loom.semaphore_give %18 : memref<64x64xf16>
            } {loom.iter_type = #loom.iter_type<temporal>}
          } {loom.iter_type = #loom.iter_type<temporal>}
        } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @y}
      } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @x}
      return
    }
  }
  module attributes {loom.pass_name = "Materialize"} {
    func.func @matmul__d0i1_d1i1__f01__a_d__BK64__BM64__BN64(%arg0: memref<4096x512xf16>, %arg1: memref<512x4096xf16>, %arg2: memref<4096x4096xf16>) {
      %cst = arith.constant 0.000000e+00 : f16
      %c64 = arith.constant 64 : index
      affine.parallel (%arg3) = (0) to (8) {
        affine.parallel (%arg4) = (0) to (8) {
          affine.for %arg5 = 0 to 64 {
            affine.for %arg6 = 0 to 1 {
              %12 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 8 + d1 + d2 * 64)>(%arg3, %arg4, %arg6)
              %13 = loom.alloc [64, 64] on @L1 : memref<64x64xf16>
              %14 = loom.semaphore_take %13 : memref<64x64xf16> -> memref<64x64xf16>
              %15 = loom.alloc [64, 64] on @L1 : memref<64x64xf16>
              %16 = loom.semaphore_take %15 : memref<64x64xf16> -> memref<64x64xf16>
              %17 = loom.alloc [64, 64] on @L1 : memref<64x64xf16>
              %18 = loom.semaphore_take %17 : memref<64x64xf16> -> memref<64x64xf16>
              %19 = loom.init_tensor %18[64, 64] : memref<64x64xf16> -> tensor<64x64xf16>
              %20 = linalg.fill ins(%cst : f16) outs(%19 : tensor<64x64xf16>) -> tensor<64x64xf16>
              %21 = affine.for %arg7 = 0 to 8 iter_args(%arg8 = %20) -> (tensor<64x64xf16>) {
                %25 = arith.muli %arg5, %c64 : index
                %26 = arith.muli %arg7, %c64 : index
                %27 = affine.apply affine_map<(d0, d1) -> (d0 * 512 + d1)>(%25, %26)
                %reinterpret_cast_0 = memref.reinterpret_cast %arg0 to offset: [%27], sizes: [64, 64], strides: [512, 1] : memref<4096x512xf16> to memref<64x64xf16, strided<[512, 1], offset: ?>>
                %28 = loom.copy_to_tensor %reinterpret_cast_0, %16 on @L1, interconnect : [@horizontal_links, @vertical_links], broadcast : [8, 8] : memref<64x64xf16, strided<[512, 1], offset: ?>>, memref<64x64xf16> -> tensor<64x64xf16>
                %29 = arith.muli %12, %c64 : index
                %30 = affine.apply affine_map<(d0, d1) -> (d0 * 4096 + d1)>(%26, %29)
                %reinterpret_cast_1 = memref.reinterpret_cast %arg1 to offset: [%30], sizes: [64, 64], strides: [4096, 1] : memref<512x4096xf16> to memref<64x64xf16, strided<[4096, 1], offset: ?>>
                %31 = loom.copy_to_tensor %reinterpret_cast_1, %14 on @L1, interconnect : [], broadcast : [1, 1] : memref<64x64xf16, strided<[4096, 1], offset: ?>>, memref<64x64xf16> -> tensor<64x64xf16>
                %32 = linalg.matmul ins(%28, %31 : tensor<64x64xf16>, tensor<64x64xf16>) outs(%arg8 : tensor<64x64xf16>) -> tensor<64x64xf16>
                loom.semaphore_give %14 : memref<64x64xf16>
                loom.semaphore_give %16 : memref<64x64xf16>
                affine.yield %32 : tensor<64x64xf16>
              }
              %22 = arith.muli %arg5, %c64 : index
              %23 = arith.muli %12, %c64 : index
              %24 = affine.apply affine_map<(d0, d1) -> (d0 * 4096 + d1)>(%22, %23)
              %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%24], sizes: [64, 64], strides: [4096, 1] : memref<4096x4096xf16> to memref<64x64xf16, strided<[4096, 1], offset: ?>>
              loom.copy_from_tensor %21, %reinterpret_cast on @DRAM : tensor<64x64xf16>, memref<64x64xf16, strided<[4096, 1], offset: ?>>
              loom.semaphore_give %18 : memref<64x64xf16>
            } {loom.iter_type = #loom.iter_type<temporal>}
          } {loom.iter_type = #loom.iter_type<temporal>}
        } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @y}
      } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @x}
      return
    }
  }
  module attributes {loom.pass_name = "Materialize"} {
    func.func @matmul__d0i1_d1i1__f01__h_d__BK64__BM64__BN64(%arg0: memref<4096x512xf16>, %arg1: memref<512x4096xf16>, %arg2: memref<4096x4096xf16>) {
      %cst = arith.constant 0.000000e+00 : f16
      %c64 = arith.constant 64 : index
      affine.parallel (%arg3) = (0) to (8) {
        affine.parallel (%arg4) = (0) to (8) {
          affine.for %arg5 = 0 to 64 {
            affine.for %arg6 = 0 to 1 {
              %12 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 8 + d1 + d2 * 64)>(%arg3, %arg4, %arg6)
              %13 = loom.alloc [64, 64] on @L1 : memref<64x64xf16>
              %14 = loom.semaphore_take %13 : memref<64x64xf16> -> memref<64x64xf16>
              %15 = loom.alloc [64, 64] on @L1 : memref<64x64xf16>
              %16 = loom.semaphore_take %15 : memref<64x64xf16> -> memref<64x64xf16>
              %17 = loom.alloc [64, 64] on @L1 : memref<64x64xf16>
              %18 = loom.semaphore_take %17 : memref<64x64xf16> -> memref<64x64xf16>
              %19 = loom.init_tensor %18[64, 64] : memref<64x64xf16> -> tensor<64x64xf16>
              %20 = linalg.fill ins(%cst : f16) outs(%19 : tensor<64x64xf16>) -> tensor<64x64xf16>
              %21 = affine.for %arg7 = 0 to 8 iter_args(%arg8 = %20) -> (tensor<64x64xf16>) {
                %25 = arith.muli %arg5, %c64 : index
                %26 = arith.muli %arg7, %c64 : index
                %27 = affine.apply affine_map<(d0, d1) -> (d0 * 512 + d1)>(%25, %26)
                %reinterpret_cast_0 = memref.reinterpret_cast %arg0 to offset: [%27], sizes: [64, 64], strides: [512, 1] : memref<4096x512xf16> to memref<64x64xf16, strided<[512, 1], offset: ?>>
                %28 = loom.copy_to_tensor %reinterpret_cast_0, %16 on @L1, interconnect : [@horizontal_links], broadcast : [1, 8] : memref<64x64xf16, strided<[512, 1], offset: ?>>, memref<64x64xf16> -> tensor<64x64xf16>
                %29 = arith.muli %12, %c64 : index
                %30 = affine.apply affine_map<(d0, d1) -> (d0 * 4096 + d1)>(%26, %29)
                %reinterpret_cast_1 = memref.reinterpret_cast %arg1 to offset: [%30], sizes: [64, 64], strides: [4096, 1] : memref<512x4096xf16> to memref<64x64xf16, strided<[4096, 1], offset: ?>>
                %31 = loom.copy_to_tensor %reinterpret_cast_1, %14 on @L1, interconnect : [], broadcast : [1, 1] : memref<64x64xf16, strided<[4096, 1], offset: ?>>, memref<64x64xf16> -> tensor<64x64xf16>
                %32 = linalg.matmul ins(%28, %31 : tensor<64x64xf16>, tensor<64x64xf16>) outs(%arg8 : tensor<64x64xf16>) -> tensor<64x64xf16>
                loom.semaphore_give %14 : memref<64x64xf16>
                loom.semaphore_give %16 : memref<64x64xf16>
                affine.yield %32 : tensor<64x64xf16>
              }
              %22 = arith.muli %arg5, %c64 : index
              %23 = arith.muli %12, %c64 : index
              %24 = affine.apply affine_map<(d0, d1) -> (d0 * 4096 + d1)>(%22, %23)
              %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%24], sizes: [64, 64], strides: [4096, 1] : memref<4096x4096xf16> to memref<64x64xf16, strided<[4096, 1], offset: ?>>
              loom.copy_from_tensor %21, %reinterpret_cast on @DRAM : tensor<64x64xf16>, memref<64x64xf16, strided<[4096, 1], offset: ?>>
              loom.semaphore_give %18 : memref<64x64xf16>
            } {loom.iter_type = #loom.iter_type<temporal>}
          } {loom.iter_type = #loom.iter_type<temporal>}
        } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @y}
      } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @x}
      return
    }
  }
  module attributes {loom.pass_name = "Materialize"} {
    func.func @matmul__d0i1_d1i1__f01__v_d__BK64__BM64__BN64(%arg0: memref<4096x512xf16>, %arg1: memref<512x4096xf16>, %arg2: memref<4096x4096xf16>) {
      %cst = arith.constant 0.000000e+00 : f16
      %c64 = arith.constant 64 : index
      affine.parallel (%arg3) = (0) to (8) {
        affine.parallel (%arg4) = (0) to (8) {
          affine.for %arg5 = 0 to 64 {
            affine.for %arg6 = 0 to 1 {
              %12 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 8 + d1 + d2 * 64)>(%arg3, %arg4, %arg6)
              %13 = loom.alloc [64, 64] on @L1 : memref<64x64xf16>
              %14 = loom.semaphore_take %13 : memref<64x64xf16> -> memref<64x64xf16>
              %15 = loom.alloc [64, 64] on @L1 : memref<64x64xf16>
              %16 = loom.semaphore_take %15 : memref<64x64xf16> -> memref<64x64xf16>
              %17 = loom.alloc [64, 64] on @L1 : memref<64x64xf16>
              %18 = loom.semaphore_take %17 : memref<64x64xf16> -> memref<64x64xf16>
              %19 = loom.init_tensor %18[64, 64] : memref<64x64xf16> -> tensor<64x64xf16>
              %20 = linalg.fill ins(%cst : f16) outs(%19 : tensor<64x64xf16>) -> tensor<64x64xf16>
              %21 = affine.for %arg7 = 0 to 8 iter_args(%arg8 = %20) -> (tensor<64x64xf16>) {
                %25 = arith.muli %arg5, %c64 : index
                %26 = arith.muli %arg7, %c64 : index
                %27 = affine.apply affine_map<(d0, d1) -> (d0 * 512 + d1)>(%25, %26)
                %reinterpret_cast_0 = memref.reinterpret_cast %arg0 to offset: [%27], sizes: [64, 64], strides: [512, 1] : memref<4096x512xf16> to memref<64x64xf16, strided<[512, 1], offset: ?>>
                %28 = loom.copy_to_tensor %reinterpret_cast_0, %16 on @L1, interconnect : [@vertical_links], broadcast : [8, 1] : memref<64x64xf16, strided<[512, 1], offset: ?>>, memref<64x64xf16> -> tensor<64x64xf16>
                %29 = arith.muli %12, %c64 : index
                %30 = affine.apply affine_map<(d0, d1) -> (d0 * 4096 + d1)>(%26, %29)
                %reinterpret_cast_1 = memref.reinterpret_cast %arg1 to offset: [%30], sizes: [64, 64], strides: [4096, 1] : memref<512x4096xf16> to memref<64x64xf16, strided<[4096, 1], offset: ?>>
                %31 = loom.copy_to_tensor %reinterpret_cast_1, %14 on @L1, interconnect : [], broadcast : [1, 1] : memref<64x64xf16, strided<[4096, 1], offset: ?>>, memref<64x64xf16> -> tensor<64x64xf16>
                %32 = linalg.matmul ins(%28, %31 : tensor<64x64xf16>, tensor<64x64xf16>) outs(%arg8 : tensor<64x64xf16>) -> tensor<64x64xf16>
                loom.semaphore_give %14 : memref<64x64xf16>
                loom.semaphore_give %16 : memref<64x64xf16>
                affine.yield %32 : tensor<64x64xf16>
              }
              %22 = arith.muli %arg5, %c64 : index
              %23 = arith.muli %12, %c64 : index
              %24 = affine.apply affine_map<(d0, d1) -> (d0 * 4096 + d1)>(%22, %23)
              %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%24], sizes: [64, 64], strides: [4096, 1] : memref<4096x4096xf16> to memref<64x64xf16, strided<[4096, 1], offset: ?>>
              loom.copy_from_tensor %21, %reinterpret_cast on @DRAM : tensor<64x64xf16>, memref<64x64xf16, strided<[4096, 1], offset: ?>>
              loom.semaphore_give %18 : memref<64x64xf16>
            } {loom.iter_type = #loom.iter_type<temporal>}
          } {loom.iter_type = #loom.iter_type<temporal>}
        } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @y}
      } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @x}
      return
    }
  }
  module attributes {loom.pass_name = "Materialize"} {
    func.func @matmul__d0i1_d1i1__f10__d_d__BK64__BM64__BN64(%arg0: memref<4096x512xf16>, %arg1: memref<512x4096xf16>, %arg2: memref<4096x4096xf16>) {
      %cst = arith.constant 0.000000e+00 : f16
      %c64 = arith.constant 64 : index
      affine.parallel (%arg3) = (0) to (8) {
        affine.parallel (%arg4) = (0) to (8) {
          affine.for %arg5 = 0 to 1 {
            affine.for %arg6 = 0 to 64 {
              %12 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 8 + d1 + d2 * 64)>(%arg3, %arg4, %arg5)
              %13 = loom.alloc [64, 64] on @L1 : memref<64x64xf16>
              %14 = loom.semaphore_take %13 : memref<64x64xf16> -> memref<64x64xf16>
              %15 = loom.alloc [64, 64] on @L1 : memref<64x64xf16>
              %16 = loom.semaphore_take %15 : memref<64x64xf16> -> memref<64x64xf16>
              %17 = loom.alloc [64, 64] on @L1 : memref<64x64xf16>
              %18 = loom.semaphore_take %17 : memref<64x64xf16> -> memref<64x64xf16>
              %19 = loom.init_tensor %18[64, 64] : memref<64x64xf16> -> tensor<64x64xf16>
              %20 = linalg.fill ins(%cst : f16) outs(%19 : tensor<64x64xf16>) -> tensor<64x64xf16>
              %21 = affine.for %arg7 = 0 to 8 iter_args(%arg8 = %20) -> (tensor<64x64xf16>) {
                %25 = arith.muli %arg6, %c64 : index
                %26 = arith.muli %arg7, %c64 : index
                %27 = affine.apply affine_map<(d0, d1) -> (d0 * 512 + d1)>(%25, %26)
                %reinterpret_cast_0 = memref.reinterpret_cast %arg0 to offset: [%27], sizes: [64, 64], strides: [512, 1] : memref<4096x512xf16> to memref<64x64xf16, strided<[512, 1], offset: ?>>
                %28 = loom.copy_to_tensor %reinterpret_cast_0, %16 on @L1, interconnect : [], broadcast : [1, 1] : memref<64x64xf16, strided<[512, 1], offset: ?>>, memref<64x64xf16> -> tensor<64x64xf16>
                %29 = arith.muli %12, %c64 : index
                %30 = affine.apply affine_map<(d0, d1) -> (d0 * 4096 + d1)>(%26, %29)
                %reinterpret_cast_1 = memref.reinterpret_cast %arg1 to offset: [%30], sizes: [64, 64], strides: [4096, 1] : memref<512x4096xf16> to memref<64x64xf16, strided<[4096, 1], offset: ?>>
                %31 = loom.copy_to_tensor %reinterpret_cast_1, %14 on @L1, interconnect : [], broadcast : [1, 1] : memref<64x64xf16, strided<[4096, 1], offset: ?>>, memref<64x64xf16> -> tensor<64x64xf16>
                %32 = linalg.matmul ins(%28, %31 : tensor<64x64xf16>, tensor<64x64xf16>) outs(%arg8 : tensor<64x64xf16>) -> tensor<64x64xf16>
                loom.semaphore_give %14 : memref<64x64xf16>
                loom.semaphore_give %16 : memref<64x64xf16>
                affine.yield %32 : tensor<64x64xf16>
              }
              %22 = arith.muli %arg6, %c64 : index
              %23 = arith.muli %12, %c64 : index
              %24 = affine.apply affine_map<(d0, d1) -> (d0 * 4096 + d1)>(%22, %23)
              %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%24], sizes: [64, 64], strides: [4096, 1] : memref<4096x4096xf16> to memref<64x64xf16, strided<[4096, 1], offset: ?>>
              loom.copy_from_tensor %21, %reinterpret_cast on @DRAM : tensor<64x64xf16>, memref<64x64xf16, strided<[4096, 1], offset: ?>>
              loom.semaphore_give %18 : memref<64x64xf16>
            } {loom.iter_type = #loom.iter_type<temporal>}
          } {loom.iter_type = #loom.iter_type<temporal>}
        } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @y}
      } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @x}
      return
    }
  }
  module attributes {loom.pass_name = "Materialize"} {
    func.func @matmul__d0i1_d1i1__f10__a_d__BK64__BM64__BN64(%arg0: memref<4096x512xf16>, %arg1: memref<512x4096xf16>, %arg2: memref<4096x4096xf16>) {
      %cst = arith.constant 0.000000e+00 : f16
      %c64 = arith.constant 64 : index
      affine.parallel (%arg3) = (0) to (8) {
        affine.parallel (%arg4) = (0) to (8) {
          affine.for %arg5 = 0 to 1 {
            affine.for %arg6 = 0 to 64 {
              %12 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 8 + d1 + d2 * 64)>(%arg3, %arg4, %arg5)
              %13 = loom.alloc [64, 64] on @L1 : memref<64x64xf16>
              %14 = loom.semaphore_take %13 : memref<64x64xf16> -> memref<64x64xf16>
              %15 = loom.alloc [64, 64] on @L1 : memref<64x64xf16>
              %16 = loom.semaphore_take %15 : memref<64x64xf16> -> memref<64x64xf16>
              %17 = loom.alloc [64, 64] on @L1 : memref<64x64xf16>
              %18 = loom.semaphore_take %17 : memref<64x64xf16> -> memref<64x64xf16>
              %19 = loom.init_tensor %18[64, 64] : memref<64x64xf16> -> tensor<64x64xf16>
              %20 = linalg.fill ins(%cst : f16) outs(%19 : tensor<64x64xf16>) -> tensor<64x64xf16>
              %21 = affine.for %arg7 = 0 to 8 iter_args(%arg8 = %20) -> (tensor<64x64xf16>) {
                %25 = arith.muli %arg6, %c64 : index
                %26 = arith.muli %arg7, %c64 : index
                %27 = affine.apply affine_map<(d0, d1) -> (d0 * 512 + d1)>(%25, %26)
                %reinterpret_cast_0 = memref.reinterpret_cast %arg0 to offset: [%27], sizes: [64, 64], strides: [512, 1] : memref<4096x512xf16> to memref<64x64xf16, strided<[512, 1], offset: ?>>
                %28 = loom.copy_to_tensor %reinterpret_cast_0, %16 on @L1, interconnect : [@horizontal_links, @vertical_links], broadcast : [8, 8] : memref<64x64xf16, strided<[512, 1], offset: ?>>, memref<64x64xf16> -> tensor<64x64xf16>
                %29 = arith.muli %12, %c64 : index
                %30 = affine.apply affine_map<(d0, d1) -> (d0 * 4096 + d1)>(%26, %29)
                %reinterpret_cast_1 = memref.reinterpret_cast %arg1 to offset: [%30], sizes: [64, 64], strides: [4096, 1] : memref<512x4096xf16> to memref<64x64xf16, strided<[4096, 1], offset: ?>>
                %31 = loom.copy_to_tensor %reinterpret_cast_1, %14 on @L1, interconnect : [], broadcast : [1, 1] : memref<64x64xf16, strided<[4096, 1], offset: ?>>, memref<64x64xf16> -> tensor<64x64xf16>
                %32 = linalg.matmul ins(%28, %31 : tensor<64x64xf16>, tensor<64x64xf16>) outs(%arg8 : tensor<64x64xf16>) -> tensor<64x64xf16>
                loom.semaphore_give %14 : memref<64x64xf16>
                loom.semaphore_give %16 : memref<64x64xf16>
                affine.yield %32 : tensor<64x64xf16>
              }
              %22 = arith.muli %arg6, %c64 : index
              %23 = arith.muli %12, %c64 : index
              %24 = affine.apply affine_map<(d0, d1) -> (d0 * 4096 + d1)>(%22, %23)
              %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%24], sizes: [64, 64], strides: [4096, 1] : memref<4096x4096xf16> to memref<64x64xf16, strided<[4096, 1], offset: ?>>
              loom.copy_from_tensor %21, %reinterpret_cast on @DRAM : tensor<64x64xf16>, memref<64x64xf16, strided<[4096, 1], offset: ?>>
              loom.semaphore_give %18 : memref<64x64xf16>
            } {loom.iter_type = #loom.iter_type<temporal>}
          } {loom.iter_type = #loom.iter_type<temporal>}
        } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @y}
      } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @x}
      return
    }
  }
  module attributes {loom.pass_name = "Materialize"} {
    func.func @matmul__d0i1_d1i1__f10__h_d__BK64__BM64__BN64(%arg0: memref<4096x512xf16>, %arg1: memref<512x4096xf16>, %arg2: memref<4096x4096xf16>) {
      %cst = arith.constant 0.000000e+00 : f16
      %c64 = arith.constant 64 : index
      affine.parallel (%arg3) = (0) to (8) {
        affine.parallel (%arg4) = (0) to (8) {
          affine.for %arg5 = 0 to 1 {
            affine.for %arg6 = 0 to 64 {
              %12 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 8 + d1 + d2 * 64)>(%arg3, %arg4, %arg5)
              %13 = loom.alloc [64, 64] on @L1 : memref<64x64xf16>
              %14 = loom.semaphore_take %13 : memref<64x64xf16> -> memref<64x64xf16>
              %15 = loom.alloc [64, 64] on @L1 : memref<64x64xf16>
              %16 = loom.semaphore_take %15 : memref<64x64xf16> -> memref<64x64xf16>
              %17 = loom.alloc [64, 64] on @L1 : memref<64x64xf16>
              %18 = loom.semaphore_take %17 : memref<64x64xf16> -> memref<64x64xf16>
              %19 = loom.init_tensor %18[64, 64] : memref<64x64xf16> -> tensor<64x64xf16>
              %20 = linalg.fill ins(%cst : f16) outs(%19 : tensor<64x64xf16>) -> tensor<64x64xf16>
              %21 = affine.for %arg7 = 0 to 8 iter_args(%arg8 = %20) -> (tensor<64x64xf16>) {
                %25 = arith.muli %arg6, %c64 : index
                %26 = arith.muli %arg7, %c64 : index
                %27 = affine.apply affine_map<(d0, d1) -> (d0 * 512 + d1)>(%25, %26)
                %reinterpret_cast_0 = memref.reinterpret_cast %arg0 to offset: [%27], sizes: [64, 64], strides: [512, 1] : memref<4096x512xf16> to memref<64x64xf16, strided<[512, 1], offset: ?>>
                %28 = loom.copy_to_tensor %reinterpret_cast_0, %16 on @L1, interconnect : [@horizontal_links], broadcast : [1, 8] : memref<64x64xf16, strided<[512, 1], offset: ?>>, memref<64x64xf16> -> tensor<64x64xf16>
                %29 = arith.muli %12, %c64 : index
                %30 = affine.apply affine_map<(d0, d1) -> (d0 * 4096 + d1)>(%26, %29)
                %reinterpret_cast_1 = memref.reinterpret_cast %arg1 to offset: [%30], sizes: [64, 64], strides: [4096, 1] : memref<512x4096xf16> to memref<64x64xf16, strided<[4096, 1], offset: ?>>
                %31 = loom.copy_to_tensor %reinterpret_cast_1, %14 on @L1, interconnect : [], broadcast : [1, 1] : memref<64x64xf16, strided<[4096, 1], offset: ?>>, memref<64x64xf16> -> tensor<64x64xf16>
                %32 = linalg.matmul ins(%28, %31 : tensor<64x64xf16>, tensor<64x64xf16>) outs(%arg8 : tensor<64x64xf16>) -> tensor<64x64xf16>
                loom.semaphore_give %14 : memref<64x64xf16>
                loom.semaphore_give %16 : memref<64x64xf16>
                affine.yield %32 : tensor<64x64xf16>
              }
              %22 = arith.muli %arg6, %c64 : index
              %23 = arith.muli %12, %c64 : index
              %24 = affine.apply affine_map<(d0, d1) -> (d0 * 4096 + d1)>(%22, %23)
              %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%24], sizes: [64, 64], strides: [4096, 1] : memref<4096x4096xf16> to memref<64x64xf16, strided<[4096, 1], offset: ?>>
              loom.copy_from_tensor %21, %reinterpret_cast on @DRAM : tensor<64x64xf16>, memref<64x64xf16, strided<[4096, 1], offset: ?>>
              loom.semaphore_give %18 : memref<64x64xf16>
            } {loom.iter_type = #loom.iter_type<temporal>}
          } {loom.iter_type = #loom.iter_type<temporal>}
        } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @y}
      } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @x}
      return
    }
  }
  module attributes {loom.pass_name = "Materialize"} {
    func.func @matmul__d0i1_d1i1__f10__v_d__BK64__BM64__BN64(%arg0: memref<4096x512xf16>, %arg1: memref<512x4096xf16>, %arg2: memref<4096x4096xf16>) {
      %cst = arith.constant 0.000000e+00 : f16
      %c64 = arith.constant 64 : index
      affine.parallel (%arg3) = (0) to (8) {
        affine.parallel (%arg4) = (0) to (8) {
          affine.for %arg5 = 0 to 1 {
            affine.for %arg6 = 0 to 64 {
              %12 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 8 + d1 + d2 * 64)>(%arg3, %arg4, %arg5)
              %13 = loom.alloc [64, 64] on @L1 : memref<64x64xf16>
              %14 = loom.semaphore_take %13 : memref<64x64xf16> -> memref<64x64xf16>
              %15 = loom.alloc [64, 64] on @L1 : memref<64x64xf16>
              %16 = loom.semaphore_take %15 : memref<64x64xf16> -> memref<64x64xf16>
              %17 = loom.alloc [64, 64] on @L1 : memref<64x64xf16>
              %18 = loom.semaphore_take %17 : memref<64x64xf16> -> memref<64x64xf16>
              %19 = loom.init_tensor %18[64, 64] : memref<64x64xf16> -> tensor<64x64xf16>
              %20 = linalg.fill ins(%cst : f16) outs(%19 : tensor<64x64xf16>) -> tensor<64x64xf16>
              %21 = affine.for %arg7 = 0 to 8 iter_args(%arg8 = %20) -> (tensor<64x64xf16>) {
                %25 = arith.muli %arg6, %c64 : index
                %26 = arith.muli %arg7, %c64 : index
                %27 = affine.apply affine_map<(d0, d1) -> (d0 * 512 + d1)>(%25, %26)
                %reinterpret_cast_0 = memref.reinterpret_cast %arg0 to offset: [%27], sizes: [64, 64], strides: [512, 1] : memref<4096x512xf16> to memref<64x64xf16, strided<[512, 1], offset: ?>>
                %28 = loom.copy_to_tensor %reinterpret_cast_0, %16 on @L1, interconnect : [@vertical_links], broadcast : [8, 1] : memref<64x64xf16, strided<[512, 1], offset: ?>>, memref<64x64xf16> -> tensor<64x64xf16>
                %29 = arith.muli %12, %c64 : index
                %30 = affine.apply affine_map<(d0, d1) -> (d0 * 4096 + d1)>(%26, %29)
                %reinterpret_cast_1 = memref.reinterpret_cast %arg1 to offset: [%30], sizes: [64, 64], strides: [4096, 1] : memref<512x4096xf16> to memref<64x64xf16, strided<[4096, 1], offset: ?>>
                %31 = loom.copy_to_tensor %reinterpret_cast_1, %14 on @L1, interconnect : [], broadcast : [1, 1] : memref<64x64xf16, strided<[4096, 1], offset: ?>>, memref<64x64xf16> -> tensor<64x64xf16>
                %32 = linalg.matmul ins(%28, %31 : tensor<64x64xf16>, tensor<64x64xf16>) outs(%arg8 : tensor<64x64xf16>) -> tensor<64x64xf16>
                loom.semaphore_give %14 : memref<64x64xf16>
                loom.semaphore_give %16 : memref<64x64xf16>
                affine.yield %32 : tensor<64x64xf16>
              }
              %22 = arith.muli %arg6, %c64 : index
              %23 = arith.muli %12, %c64 : index
              %24 = affine.apply affine_map<(d0, d1) -> (d0 * 4096 + d1)>(%22, %23)
              %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%24], sizes: [64, 64], strides: [4096, 1] : memref<4096x4096xf16> to memref<64x64xf16, strided<[4096, 1], offset: ?>>
              loom.copy_from_tensor %21, %reinterpret_cast on @DRAM : tensor<64x64xf16>, memref<64x64xf16, strided<[4096, 1], offset: ?>>
              loom.semaphore_give %18 : memref<64x64xf16>
            } {loom.iter_type = #loom.iter_type<temporal>}
          } {loom.iter_type = #loom.iter_type<temporal>}
        } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @y}
      } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @x}
      return
    }
  }
  module attributes {loom.pass_name = "Materialize"} {
    func.func @matmul__d1i1_d0i1__f01__d_d__BK64__BM64__BN64(%arg0: memref<4096x512xf16>, %arg1: memref<512x4096xf16>, %arg2: memref<4096x4096xf16>) {
      %cst = arith.constant 0.000000e+00 : f16
      %c64 = arith.constant 64 : index
      affine.parallel (%arg3) = (0) to (8) {
        affine.parallel (%arg4) = (0) to (8) {
          affine.for %arg5 = 0 to 64 {
            affine.for %arg6 = 0 to 1 {
              %12 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 8 + d1 + d2 * 64)>(%arg3, %arg4, %arg6)
              %13 = loom.alloc [64, 64] on @L1 : memref<64x64xf16>
              %14 = loom.semaphore_take %13 : memref<64x64xf16> -> memref<64x64xf16>
              %15 = loom.alloc [64, 64] on @L1 : memref<64x64xf16>
              %16 = loom.semaphore_take %15 : memref<64x64xf16> -> memref<64x64xf16>
              %17 = loom.alloc [64, 64] on @L1 : memref<64x64xf16>
              %18 = loom.semaphore_take %17 : memref<64x64xf16> -> memref<64x64xf16>
              %19 = loom.init_tensor %18[64, 64] : memref<64x64xf16> -> tensor<64x64xf16>
              %20 = linalg.fill ins(%cst : f16) outs(%19 : tensor<64x64xf16>) -> tensor<64x64xf16>
              %21 = affine.for %arg7 = 0 to 8 iter_args(%arg8 = %20) -> (tensor<64x64xf16>) {
                %25 = arith.muli %arg5, %c64 : index
                %26 = arith.muli %arg7, %c64 : index
                %27 = affine.apply affine_map<(d0, d1) -> (d0 * 512 + d1)>(%25, %26)
                %reinterpret_cast_0 = memref.reinterpret_cast %arg0 to offset: [%27], sizes: [64, 64], strides: [512, 1] : memref<4096x512xf16> to memref<64x64xf16, strided<[512, 1], offset: ?>>
                %28 = loom.copy_to_tensor %reinterpret_cast_0, %16 on @L1, interconnect : [], broadcast : [1, 1] : memref<64x64xf16, strided<[512, 1], offset: ?>>, memref<64x64xf16> -> tensor<64x64xf16>
                %29 = arith.muli %12, %c64 : index
                %30 = affine.apply affine_map<(d0, d1) -> (d0 * 4096 + d1)>(%26, %29)
                %reinterpret_cast_1 = memref.reinterpret_cast %arg1 to offset: [%30], sizes: [64, 64], strides: [4096, 1] : memref<512x4096xf16> to memref<64x64xf16, strided<[4096, 1], offset: ?>>
                %31 = loom.copy_to_tensor %reinterpret_cast_1, %14 on @L1, interconnect : [], broadcast : [1, 1] : memref<64x64xf16, strided<[4096, 1], offset: ?>>, memref<64x64xf16> -> tensor<64x64xf16>
                %32 = linalg.matmul ins(%28, %31 : tensor<64x64xf16>, tensor<64x64xf16>) outs(%arg8 : tensor<64x64xf16>) -> tensor<64x64xf16>
                loom.semaphore_give %14 : memref<64x64xf16>
                loom.semaphore_give %16 : memref<64x64xf16>
                affine.yield %32 : tensor<64x64xf16>
              }
              %22 = arith.muli %arg5, %c64 : index
              %23 = arith.muli %12, %c64 : index
              %24 = affine.apply affine_map<(d0, d1) -> (d0 * 4096 + d1)>(%22, %23)
              %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%24], sizes: [64, 64], strides: [4096, 1] : memref<4096x4096xf16> to memref<64x64xf16, strided<[4096, 1], offset: ?>>
              loom.copy_from_tensor %21, %reinterpret_cast on @DRAM : tensor<64x64xf16>, memref<64x64xf16, strided<[4096, 1], offset: ?>>
              loom.semaphore_give %18 : memref<64x64xf16>
            } {loom.iter_type = #loom.iter_type<temporal>}
          } {loom.iter_type = #loom.iter_type<temporal>}
        } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @x}
      } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @y}
      return
    }
  }
  module attributes {loom.pass_name = "Materialize"} {
    func.func @matmul__d1i1_d0i1__f01__a_d__BK64__BM64__BN64(%arg0: memref<4096x512xf16>, %arg1: memref<512x4096xf16>, %arg2: memref<4096x4096xf16>) {
      %cst = arith.constant 0.000000e+00 : f16
      %c64 = arith.constant 64 : index
      affine.parallel (%arg3) = (0) to (8) {
        affine.parallel (%arg4) = (0) to (8) {
          affine.for %arg5 = 0 to 64 {
            affine.for %arg6 = 0 to 1 {
              %12 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 8 + d1 + d2 * 64)>(%arg3, %arg4, %arg6)
              %13 = loom.alloc [64, 64] on @L1 : memref<64x64xf16>
              %14 = loom.semaphore_take %13 : memref<64x64xf16> -> memref<64x64xf16>
              %15 = loom.alloc [64, 64] on @L1 : memref<64x64xf16>
              %16 = loom.semaphore_take %15 : memref<64x64xf16> -> memref<64x64xf16>
              %17 = loom.alloc [64, 64] on @L1 : memref<64x64xf16>
              %18 = loom.semaphore_take %17 : memref<64x64xf16> -> memref<64x64xf16>
              %19 = loom.init_tensor %18[64, 64] : memref<64x64xf16> -> tensor<64x64xf16>
              %20 = linalg.fill ins(%cst : f16) outs(%19 : tensor<64x64xf16>) -> tensor<64x64xf16>
              %21 = affine.for %arg7 = 0 to 8 iter_args(%arg8 = %20) -> (tensor<64x64xf16>) {
                %25 = arith.muli %arg5, %c64 : index
                %26 = arith.muli %arg7, %c64 : index
                %27 = affine.apply affine_map<(d0, d1) -> (d0 * 512 + d1)>(%25, %26)
                %reinterpret_cast_0 = memref.reinterpret_cast %arg0 to offset: [%27], sizes: [64, 64], strides: [512, 1] : memref<4096x512xf16> to memref<64x64xf16, strided<[512, 1], offset: ?>>
                %28 = loom.copy_to_tensor %reinterpret_cast_0, %16 on @L1, interconnect : [@horizontal_links, @vertical_links], broadcast : [8, 8] : memref<64x64xf16, strided<[512, 1], offset: ?>>, memref<64x64xf16> -> tensor<64x64xf16>
                %29 = arith.muli %12, %c64 : index
                %30 = affine.apply affine_map<(d0, d1) -> (d0 * 4096 + d1)>(%26, %29)
                %reinterpret_cast_1 = memref.reinterpret_cast %arg1 to offset: [%30], sizes: [64, 64], strides: [4096, 1] : memref<512x4096xf16> to memref<64x64xf16, strided<[4096, 1], offset: ?>>
                %31 = loom.copy_to_tensor %reinterpret_cast_1, %14 on @L1, interconnect : [], broadcast : [1, 1] : memref<64x64xf16, strided<[4096, 1], offset: ?>>, memref<64x64xf16> -> tensor<64x64xf16>
                %32 = linalg.matmul ins(%28, %31 : tensor<64x64xf16>, tensor<64x64xf16>) outs(%arg8 : tensor<64x64xf16>) -> tensor<64x64xf16>
                loom.semaphore_give %14 : memref<64x64xf16>
                loom.semaphore_give %16 : memref<64x64xf16>
                affine.yield %32 : tensor<64x64xf16>
              }
              %22 = arith.muli %arg5, %c64 : index
              %23 = arith.muli %12, %c64 : index
              %24 = affine.apply affine_map<(d0, d1) -> (d0 * 4096 + d1)>(%22, %23)
              %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%24], sizes: [64, 64], strides: [4096, 1] : memref<4096x4096xf16> to memref<64x64xf16, strided<[4096, 1], offset: ?>>
              loom.copy_from_tensor %21, %reinterpret_cast on @DRAM : tensor<64x64xf16>, memref<64x64xf16, strided<[4096, 1], offset: ?>>
              loom.semaphore_give %18 : memref<64x64xf16>
            } {loom.iter_type = #loom.iter_type<temporal>}
          } {loom.iter_type = #loom.iter_type<temporal>}
        } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @x}
      } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @y}
      return
    }
  }
  module attributes {loom.pass_name = "Materialize"} {
    func.func @matmul__d1i1_d0i1__f01__h_d__BK64__BM64__BN64(%arg0: memref<4096x512xf16>, %arg1: memref<512x4096xf16>, %arg2: memref<4096x4096xf16>) {
      %cst = arith.constant 0.000000e+00 : f16
      %c64 = arith.constant 64 : index
      affine.parallel (%arg3) = (0) to (8) {
        affine.parallel (%arg4) = (0) to (8) {
          affine.for %arg5 = 0 to 64 {
            affine.for %arg6 = 0 to 1 {
              %12 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 8 + d1 + d2 * 64)>(%arg3, %arg4, %arg6)
              %13 = loom.alloc [64, 64] on @L1 : memref<64x64xf16>
              %14 = loom.semaphore_take %13 : memref<64x64xf16> -> memref<64x64xf16>
              %15 = loom.alloc [64, 64] on @L1 : memref<64x64xf16>
              %16 = loom.semaphore_take %15 : memref<64x64xf16> -> memref<64x64xf16>
              %17 = loom.alloc [64, 64] on @L1 : memref<64x64xf16>
              %18 = loom.semaphore_take %17 : memref<64x64xf16> -> memref<64x64xf16>
              %19 = loom.init_tensor %18[64, 64] : memref<64x64xf16> -> tensor<64x64xf16>
              %20 = linalg.fill ins(%cst : f16) outs(%19 : tensor<64x64xf16>) -> tensor<64x64xf16>
              %21 = affine.for %arg7 = 0 to 8 iter_args(%arg8 = %20) -> (tensor<64x64xf16>) {
                %25 = arith.muli %arg5, %c64 : index
                %26 = arith.muli %arg7, %c64 : index
                %27 = affine.apply affine_map<(d0, d1) -> (d0 * 512 + d1)>(%25, %26)
                %reinterpret_cast_0 = memref.reinterpret_cast %arg0 to offset: [%27], sizes: [64, 64], strides: [512, 1] : memref<4096x512xf16> to memref<64x64xf16, strided<[512, 1], offset: ?>>
                %28 = loom.copy_to_tensor %reinterpret_cast_0, %16 on @L1, interconnect : [@horizontal_links], broadcast : [1, 8] : memref<64x64xf16, strided<[512, 1], offset: ?>>, memref<64x64xf16> -> tensor<64x64xf16>
                %29 = arith.muli %12, %c64 : index
                %30 = affine.apply affine_map<(d0, d1) -> (d0 * 4096 + d1)>(%26, %29)
                %reinterpret_cast_1 = memref.reinterpret_cast %arg1 to offset: [%30], sizes: [64, 64], strides: [4096, 1] : memref<512x4096xf16> to memref<64x64xf16, strided<[4096, 1], offset: ?>>
                %31 = loom.copy_to_tensor %reinterpret_cast_1, %14 on @L1, interconnect : [], broadcast : [1, 1] : memref<64x64xf16, strided<[4096, 1], offset: ?>>, memref<64x64xf16> -> tensor<64x64xf16>
                %32 = linalg.matmul ins(%28, %31 : tensor<64x64xf16>, tensor<64x64xf16>) outs(%arg8 : tensor<64x64xf16>) -> tensor<64x64xf16>
                loom.semaphore_give %14 : memref<64x64xf16>
                loom.semaphore_give %16 : memref<64x64xf16>
                affine.yield %32 : tensor<64x64xf16>
              }
              %22 = arith.muli %arg5, %c64 : index
              %23 = arith.muli %12, %c64 : index
              %24 = affine.apply affine_map<(d0, d1) -> (d0 * 4096 + d1)>(%22, %23)
              %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%24], sizes: [64, 64], strides: [4096, 1] : memref<4096x4096xf16> to memref<64x64xf16, strided<[4096, 1], offset: ?>>
              loom.copy_from_tensor %21, %reinterpret_cast on @DRAM : tensor<64x64xf16>, memref<64x64xf16, strided<[4096, 1], offset: ?>>
              loom.semaphore_give %18 : memref<64x64xf16>
            } {loom.iter_type = #loom.iter_type<temporal>}
          } {loom.iter_type = #loom.iter_type<temporal>}
        } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @x}
      } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @y}
      return
    }
  }
  module attributes {loom.pass_name = "Materialize"} {
    func.func @matmul__d1i1_d0i1__f01__v_d__BK64__BM64__BN64(%arg0: memref<4096x512xf16>, %arg1: memref<512x4096xf16>, %arg2: memref<4096x4096xf16>) {
      %cst = arith.constant 0.000000e+00 : f16
      %c64 = arith.constant 64 : index
      affine.parallel (%arg3) = (0) to (8) {
        affine.parallel (%arg4) = (0) to (8) {
          affine.for %arg5 = 0 to 64 {
            affine.for %arg6 = 0 to 1 {
              %12 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 8 + d1 + d2 * 64)>(%arg3, %arg4, %arg6)
              %13 = loom.alloc [64, 64] on @L1 : memref<64x64xf16>
              %14 = loom.semaphore_take %13 : memref<64x64xf16> -> memref<64x64xf16>
              %15 = loom.alloc [64, 64] on @L1 : memref<64x64xf16>
              %16 = loom.semaphore_take %15 : memref<64x64xf16> -> memref<64x64xf16>
              %17 = loom.alloc [64, 64] on @L1 : memref<64x64xf16>
              %18 = loom.semaphore_take %17 : memref<64x64xf16> -> memref<64x64xf16>
              %19 = loom.init_tensor %18[64, 64] : memref<64x64xf16> -> tensor<64x64xf16>
              %20 = linalg.fill ins(%cst : f16) outs(%19 : tensor<64x64xf16>) -> tensor<64x64xf16>
              %21 = affine.for %arg7 = 0 to 8 iter_args(%arg8 = %20) -> (tensor<64x64xf16>) {
                %25 = arith.muli %arg5, %c64 : index
                %26 = arith.muli %arg7, %c64 : index
                %27 = affine.apply affine_map<(d0, d1) -> (d0 * 512 + d1)>(%25, %26)
                %reinterpret_cast_0 = memref.reinterpret_cast %arg0 to offset: [%27], sizes: [64, 64], strides: [512, 1] : memref<4096x512xf16> to memref<64x64xf16, strided<[512, 1], offset: ?>>
                %28 = loom.copy_to_tensor %reinterpret_cast_0, %16 on @L1, interconnect : [@vertical_links], broadcast : [8, 1] : memref<64x64xf16, strided<[512, 1], offset: ?>>, memref<64x64xf16> -> tensor<64x64xf16>
                %29 = arith.muli %12, %c64 : index
                %30 = affine.apply affine_map<(d0, d1) -> (d0 * 4096 + d1)>(%26, %29)
                %reinterpret_cast_1 = memref.reinterpret_cast %arg1 to offset: [%30], sizes: [64, 64], strides: [4096, 1] : memref<512x4096xf16> to memref<64x64xf16, strided<[4096, 1], offset: ?>>
                %31 = loom.copy_to_tensor %reinterpret_cast_1, %14 on @L1, interconnect : [], broadcast : [1, 1] : memref<64x64xf16, strided<[4096, 1], offset: ?>>, memref<64x64xf16> -> tensor<64x64xf16>
                %32 = linalg.matmul ins(%28, %31 : tensor<64x64xf16>, tensor<64x64xf16>) outs(%arg8 : tensor<64x64xf16>) -> tensor<64x64xf16>
                loom.semaphore_give %14 : memref<64x64xf16>
                loom.semaphore_give %16 : memref<64x64xf16>
                affine.yield %32 : tensor<64x64xf16>
              }
              %22 = arith.muli %arg5, %c64 : index
              %23 = arith.muli %12, %c64 : index
              %24 = affine.apply affine_map<(d0, d1) -> (d0 * 4096 + d1)>(%22, %23)
              %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%24], sizes: [64, 64], strides: [4096, 1] : memref<4096x4096xf16> to memref<64x64xf16, strided<[4096, 1], offset: ?>>
              loom.copy_from_tensor %21, %reinterpret_cast on @DRAM : tensor<64x64xf16>, memref<64x64xf16, strided<[4096, 1], offset: ?>>
              loom.semaphore_give %18 : memref<64x64xf16>
            } {loom.iter_type = #loom.iter_type<temporal>}
          } {loom.iter_type = #loom.iter_type<temporal>}
        } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @x}
      } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @y}
      return
    }
  }
  module attributes {loom.pass_name = "Materialize"} {
    func.func @matmul__d1i1_d0i1__f10__d_d__BK64__BM64__BN64(%arg0: memref<4096x512xf16>, %arg1: memref<512x4096xf16>, %arg2: memref<4096x4096xf16>) {
      %cst = arith.constant 0.000000e+00 : f16
      %c64 = arith.constant 64 : index
      affine.parallel (%arg3) = (0) to (8) {
        affine.parallel (%arg4) = (0) to (8) {
          affine.for %arg5 = 0 to 1 {
            affine.for %arg6 = 0 to 64 {
              %12 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 8 + d1 + d2 * 64)>(%arg3, %arg4, %arg5)
              %13 = loom.alloc [64, 64] on @L1 : memref<64x64xf16>
              %14 = loom.semaphore_take %13 : memref<64x64xf16> -> memref<64x64xf16>
              %15 = loom.alloc [64, 64] on @L1 : memref<64x64xf16>
              %16 = loom.semaphore_take %15 : memref<64x64xf16> -> memref<64x64xf16>
              %17 = loom.alloc [64, 64] on @L1 : memref<64x64xf16>
              %18 = loom.semaphore_take %17 : memref<64x64xf16> -> memref<64x64xf16>
              %19 = loom.init_tensor %18[64, 64] : memref<64x64xf16> -> tensor<64x64xf16>
              %20 = linalg.fill ins(%cst : f16) outs(%19 : tensor<64x64xf16>) -> tensor<64x64xf16>
              %21 = affine.for %arg7 = 0 to 8 iter_args(%arg8 = %20) -> (tensor<64x64xf16>) {
                %25 = arith.muli %arg6, %c64 : index
                %26 = arith.muli %arg7, %c64 : index
                %27 = affine.apply affine_map<(d0, d1) -> (d0 * 512 + d1)>(%25, %26)
                %reinterpret_cast_0 = memref.reinterpret_cast %arg0 to offset: [%27], sizes: [64, 64], strides: [512, 1] : memref<4096x512xf16> to memref<64x64xf16, strided<[512, 1], offset: ?>>
                %28 = loom.copy_to_tensor %reinterpret_cast_0, %16 on @L1, interconnect : [], broadcast : [1, 1] : memref<64x64xf16, strided<[512, 1], offset: ?>>, memref<64x64xf16> -> tensor<64x64xf16>
                %29 = arith.muli %12, %c64 : index
                %30 = affine.apply affine_map<(d0, d1) -> (d0 * 4096 + d1)>(%26, %29)
                %reinterpret_cast_1 = memref.reinterpret_cast %arg1 to offset: [%30], sizes: [64, 64], strides: [4096, 1] : memref<512x4096xf16> to memref<64x64xf16, strided<[4096, 1], offset: ?>>
                %31 = loom.copy_to_tensor %reinterpret_cast_1, %14 on @L1, interconnect : [], broadcast : [1, 1] : memref<64x64xf16, strided<[4096, 1], offset: ?>>, memref<64x64xf16> -> tensor<64x64xf16>
                %32 = linalg.matmul ins(%28, %31 : tensor<64x64xf16>, tensor<64x64xf16>) outs(%arg8 : tensor<64x64xf16>) -> tensor<64x64xf16>
                loom.semaphore_give %14 : memref<64x64xf16>
                loom.semaphore_give %16 : memref<64x64xf16>
                affine.yield %32 : tensor<64x64xf16>
              }
              %22 = arith.muli %arg6, %c64 : index
              %23 = arith.muli %12, %c64 : index
              %24 = affine.apply affine_map<(d0, d1) -> (d0 * 4096 + d1)>(%22, %23)
              %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%24], sizes: [64, 64], strides: [4096, 1] : memref<4096x4096xf16> to memref<64x64xf16, strided<[4096, 1], offset: ?>>
              loom.copy_from_tensor %21, %reinterpret_cast on @DRAM : tensor<64x64xf16>, memref<64x64xf16, strided<[4096, 1], offset: ?>>
              loom.semaphore_give %18 : memref<64x64xf16>
            } {loom.iter_type = #loom.iter_type<temporal>}
          } {loom.iter_type = #loom.iter_type<temporal>}
        } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @x}
      } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @y}
      return
    }
  }
  module attributes {loom.pass_name = "Materialize"} {
    func.func @matmul__d1i1_d0i1__f10__a_d__BK64__BM64__BN64(%arg0: memref<4096x512xf16>, %arg1: memref<512x4096xf16>, %arg2: memref<4096x4096xf16>) {
      %cst = arith.constant 0.000000e+00 : f16
      %c64 = arith.constant 64 : index
      affine.parallel (%arg3) = (0) to (8) {
        affine.parallel (%arg4) = (0) to (8) {
          affine.for %arg5 = 0 to 1 {
            affine.for %arg6 = 0 to 64 {
              %12 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 8 + d1 + d2 * 64)>(%arg3, %arg4, %arg5)
              %13 = loom.alloc [64, 64] on @L1 : memref<64x64xf16>
              %14 = loom.semaphore_take %13 : memref<64x64xf16> -> memref<64x64xf16>
              %15 = loom.alloc [64, 64] on @L1 : memref<64x64xf16>
              %16 = loom.semaphore_take %15 : memref<64x64xf16> -> memref<64x64xf16>
              %17 = loom.alloc [64, 64] on @L1 : memref<64x64xf16>
              %18 = loom.semaphore_take %17 : memref<64x64xf16> -> memref<64x64xf16>
              %19 = loom.init_tensor %18[64, 64] : memref<64x64xf16> -> tensor<64x64xf16>
              %20 = linalg.fill ins(%cst : f16) outs(%19 : tensor<64x64xf16>) -> tensor<64x64xf16>
              %21 = affine.for %arg7 = 0 to 8 iter_args(%arg8 = %20) -> (tensor<64x64xf16>) {
                %25 = arith.muli %arg6, %c64 : index
                %26 = arith.muli %arg7, %c64 : index
                %27 = affine.apply affine_map<(d0, d1) -> (d0 * 512 + d1)>(%25, %26)
                %reinterpret_cast_0 = memref.reinterpret_cast %arg0 to offset: [%27], sizes: [64, 64], strides: [512, 1] : memref<4096x512xf16> to memref<64x64xf16, strided<[512, 1], offset: ?>>
                %28 = loom.copy_to_tensor %reinterpret_cast_0, %16 on @L1, interconnect : [@horizontal_links, @vertical_links], broadcast : [8, 8] : memref<64x64xf16, strided<[512, 1], offset: ?>>, memref<64x64xf16> -> tensor<64x64xf16>
                %29 = arith.muli %12, %c64 : index
                %30 = affine.apply affine_map<(d0, d1) -> (d0 * 4096 + d1)>(%26, %29)
                %reinterpret_cast_1 = memref.reinterpret_cast %arg1 to offset: [%30], sizes: [64, 64], strides: [4096, 1] : memref<512x4096xf16> to memref<64x64xf16, strided<[4096, 1], offset: ?>>
                %31 = loom.copy_to_tensor %reinterpret_cast_1, %14 on @L1, interconnect : [], broadcast : [1, 1] : memref<64x64xf16, strided<[4096, 1], offset: ?>>, memref<64x64xf16> -> tensor<64x64xf16>
                %32 = linalg.matmul ins(%28, %31 : tensor<64x64xf16>, tensor<64x64xf16>) outs(%arg8 : tensor<64x64xf16>) -> tensor<64x64xf16>
                loom.semaphore_give %14 : memref<64x64xf16>
                loom.semaphore_give %16 : memref<64x64xf16>
                affine.yield %32 : tensor<64x64xf16>
              }
              %22 = arith.muli %arg6, %c64 : index
              %23 = arith.muli %12, %c64 : index
              %24 = affine.apply affine_map<(d0, d1) -> (d0 * 4096 + d1)>(%22, %23)
              %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%24], sizes: [64, 64], strides: [4096, 1] : memref<4096x4096xf16> to memref<64x64xf16, strided<[4096, 1], offset: ?>>
              loom.copy_from_tensor %21, %reinterpret_cast on @DRAM : tensor<64x64xf16>, memref<64x64xf16, strided<[4096, 1], offset: ?>>
              loom.semaphore_give %18 : memref<64x64xf16>
            } {loom.iter_type = #loom.iter_type<temporal>}
          } {loom.iter_type = #loom.iter_type<temporal>}
        } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @x}
      } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @y}
      return
    }
  }
  module attributes {loom.pass_name = "Materialize"} {
    func.func @matmul__d1i1_d0i1__f10__h_d__BK64__BM64__BN64(%arg0: memref<4096x512xf16>, %arg1: memref<512x4096xf16>, %arg2: memref<4096x4096xf16>) {
      %cst = arith.constant 0.000000e+00 : f16
      %c64 = arith.constant 64 : index
      affine.parallel (%arg3) = (0) to (8) {
        affine.parallel (%arg4) = (0) to (8) {
          affine.for %arg5 = 0 to 1 {
            affine.for %arg6 = 0 to 64 {
              %12 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 8 + d1 + d2 * 64)>(%arg3, %arg4, %arg5)
              %13 = loom.alloc [64, 64] on @L1 : memref<64x64xf16>
              %14 = loom.semaphore_take %13 : memref<64x64xf16> -> memref<64x64xf16>
              %15 = loom.alloc [64, 64] on @L1 : memref<64x64xf16>
              %16 = loom.semaphore_take %15 : memref<64x64xf16> -> memref<64x64xf16>
              %17 = loom.alloc [64, 64] on @L1 : memref<64x64xf16>
              %18 = loom.semaphore_take %17 : memref<64x64xf16> -> memref<64x64xf16>
              %19 = loom.init_tensor %18[64, 64] : memref<64x64xf16> -> tensor<64x64xf16>
              %20 = linalg.fill ins(%cst : f16) outs(%19 : tensor<64x64xf16>) -> tensor<64x64xf16>
              %21 = affine.for %arg7 = 0 to 8 iter_args(%arg8 = %20) -> (tensor<64x64xf16>) {
                %25 = arith.muli %arg6, %c64 : index
                %26 = arith.muli %arg7, %c64 : index
                %27 = affine.apply affine_map<(d0, d1) -> (d0 * 512 + d1)>(%25, %26)
                %reinterpret_cast_0 = memref.reinterpret_cast %arg0 to offset: [%27], sizes: [64, 64], strides: [512, 1] : memref<4096x512xf16> to memref<64x64xf16, strided<[512, 1], offset: ?>>
                %28 = loom.copy_to_tensor %reinterpret_cast_0, %16 on @L1, interconnect : [@horizontal_links], broadcast : [1, 8] : memref<64x64xf16, strided<[512, 1], offset: ?>>, memref<64x64xf16> -> tensor<64x64xf16>
                %29 = arith.muli %12, %c64 : index
                %30 = affine.apply affine_map<(d0, d1) -> (d0 * 4096 + d1)>(%26, %29)
                %reinterpret_cast_1 = memref.reinterpret_cast %arg1 to offset: [%30], sizes: [64, 64], strides: [4096, 1] : memref<512x4096xf16> to memref<64x64xf16, strided<[4096, 1], offset: ?>>
                %31 = loom.copy_to_tensor %reinterpret_cast_1, %14 on @L1, interconnect : [], broadcast : [1, 1] : memref<64x64xf16, strided<[4096, 1], offset: ?>>, memref<64x64xf16> -> tensor<64x64xf16>
                %32 = linalg.matmul ins(%28, %31 : tensor<64x64xf16>, tensor<64x64xf16>) outs(%arg8 : tensor<64x64xf16>) -> tensor<64x64xf16>
                loom.semaphore_give %14 : memref<64x64xf16>
                loom.semaphore_give %16 : memref<64x64xf16>
                affine.yield %32 : tensor<64x64xf16>
              }
              %22 = arith.muli %arg6, %c64 : index
              %23 = arith.muli %12, %c64 : index
              %24 = affine.apply affine_map<(d0, d1) -> (d0 * 4096 + d1)>(%22, %23)
              %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%24], sizes: [64, 64], strides: [4096, 1] : memref<4096x4096xf16> to memref<64x64xf16, strided<[4096, 1], offset: ?>>
              loom.copy_from_tensor %21, %reinterpret_cast on @DRAM : tensor<64x64xf16>, memref<64x64xf16, strided<[4096, 1], offset: ?>>
              loom.semaphore_give %18 : memref<64x64xf16>
            } {loom.iter_type = #loom.iter_type<temporal>}
          } {loom.iter_type = #loom.iter_type<temporal>}
        } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @x}
      } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @y}
      return
    }
  }
  module attributes {loom.pass_name = "Materialize"} {
    func.func @matmul__d1i1_d0i1__f10__v_d__BK64__BM64__BN64(%arg0: memref<4096x512xf16>, %arg1: memref<512x4096xf16>, %arg2: memref<4096x4096xf16>) {
      %cst = arith.constant 0.000000e+00 : f16
      %c64 = arith.constant 64 : index
      affine.parallel (%arg3) = (0) to (8) {
        affine.parallel (%arg4) = (0) to (8) {
          affine.for %arg5 = 0 to 1 {
            affine.for %arg6 = 0 to 64 {
              %12 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 8 + d1 + d2 * 64)>(%arg3, %arg4, %arg5)
              %13 = loom.alloc [64, 64] on @L1 : memref<64x64xf16>
              %14 = loom.semaphore_take %13 : memref<64x64xf16> -> memref<64x64xf16>
              %15 = loom.alloc [64, 64] on @L1 : memref<64x64xf16>
              %16 = loom.semaphore_take %15 : memref<64x64xf16> -> memref<64x64xf16>
              %17 = loom.alloc [64, 64] on @L1 : memref<64x64xf16>
              %18 = loom.semaphore_take %17 : memref<64x64xf16> -> memref<64x64xf16>
              %19 = loom.init_tensor %18[64, 64] : memref<64x64xf16> -> tensor<64x64xf16>
              %20 = linalg.fill ins(%cst : f16) outs(%19 : tensor<64x64xf16>) -> tensor<64x64xf16>
              %21 = affine.for %arg7 = 0 to 8 iter_args(%arg8 = %20) -> (tensor<64x64xf16>) {
                %25 = arith.muli %arg6, %c64 : index
                %26 = arith.muli %arg7, %c64 : index
                %27 = affine.apply affine_map<(d0, d1) -> (d0 * 512 + d1)>(%25, %26)
                %reinterpret_cast_0 = memref.reinterpret_cast %arg0 to offset: [%27], sizes: [64, 64], strides: [512, 1] : memref<4096x512xf16> to memref<64x64xf16, strided<[512, 1], offset: ?>>
                %28 = loom.copy_to_tensor %reinterpret_cast_0, %16 on @L1, interconnect : [@vertical_links], broadcast : [8, 1] : memref<64x64xf16, strided<[512, 1], offset: ?>>, memref<64x64xf16> -> tensor<64x64xf16>
                %29 = arith.muli %12, %c64 : index
                %30 = affine.apply affine_map<(d0, d1) -> (d0 * 4096 + d1)>(%26, %29)
                %reinterpret_cast_1 = memref.reinterpret_cast %arg1 to offset: [%30], sizes: [64, 64], strides: [4096, 1] : memref<512x4096xf16> to memref<64x64xf16, strided<[4096, 1], offset: ?>>
                %31 = loom.copy_to_tensor %reinterpret_cast_1, %14 on @L1, interconnect : [], broadcast : [1, 1] : memref<64x64xf16, strided<[4096, 1], offset: ?>>, memref<64x64xf16> -> tensor<64x64xf16>
                %32 = linalg.matmul ins(%28, %31 : tensor<64x64xf16>, tensor<64x64xf16>) outs(%arg8 : tensor<64x64xf16>) -> tensor<64x64xf16>
                loom.semaphore_give %14 : memref<64x64xf16>
                loom.semaphore_give %16 : memref<64x64xf16>
                affine.yield %32 : tensor<64x64xf16>
              }
              %22 = arith.muli %arg6, %c64 : index
              %23 = arith.muli %12, %c64 : index
              %24 = affine.apply affine_map<(d0, d1) -> (d0 * 4096 + d1)>(%22, %23)
              %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%24], sizes: [64, 64], strides: [4096, 1] : memref<4096x4096xf16> to memref<64x64xf16, strided<[4096, 1], offset: ?>>
              loom.copy_from_tensor %21, %reinterpret_cast on @DRAM : tensor<64x64xf16>, memref<64x64xf16, strided<[4096, 1], offset: ?>>
              loom.semaphore_give %18 : memref<64x64xf16>
            } {loom.iter_type = #loom.iter_type<temporal>}
          } {loom.iter_type = #loom.iter_type<temporal>}
        } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @x}
      } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @y}
      return
    }
  }
}
