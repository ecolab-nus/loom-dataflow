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
  module {
    func.func @matmul__d0i0_d1i0__f01__d_d(%arg0: memref<4096x512xf16>, %arg1: memref<512x4096xf16>, %arg2: memref<4096x4096xf16>) {
      %cst = arith.constant 0.000000e+00 : f16
      %12 = loom.get_symbolic_block_size @constraints::@BM : index
      %13 = loom.get_symbolic_block_size @constraints::@BN : index
      %14 = loom.get_symbolic_block_size @constraints::@BK : index
      affine.parallel (%arg3) = (0) to (8) {
        affine.parallel (%arg4) = (0) to (8) {
          affine.for %arg5 = 0 to affine_map<()[s0, s1] -> (4096 ceildiv (s0 * 64))>()[%12, %13] {
            affine.for %arg6 = 0 to affine_map<()[s0, s1] -> (4096 ceildiv s1)>()[%12, %13] {
              %15 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 8 + d1 + d2 * 64)>(%arg3, %arg4, %arg5)
              %16 = loom.alloc [%14, %13] on @L1 : memref<?x?xf16>
              %17 = loom.semaphore_take %16 : memref<?x?xf16> -> memref<?x?xf16>
              %18 = loom.alloc [%12, %14] on @L1 : memref<?x?xf16>
              %19 = loom.semaphore_take %18 : memref<?x?xf16> -> memref<?x?xf16>
              %20 = loom.alloc [%12, %13] on @L1 : memref<?x?xf16>
              %21 = loom.semaphore_take %20 : memref<?x?xf16> -> memref<?x?xf16>
              %22 = loom.init_tensor %21[%12, %13] : memref<?x?xf16> -> tensor<?x?xf16>
              %23 = linalg.fill ins(%cst : f16) outs(%22 : tensor<?x?xf16>) -> tensor<?x?xf16>
              %24 = affine.for %arg7 = 0 to affine_map<()[s0] -> (512 ceildiv s0)>()[%14] iter_args(%arg8 = %23) -> (tensor<?x?xf16>) {
                %28 = arith.muli %15, %12 : index
                %29 = arith.muli %arg7, %14 : index
                %30 = loom.subview %arg0[%28, %29] [%12, %14] [1, 1], reuse : [seq = false, spat = false, temp = true] : memref<4096x512xf16> to memref<?x?xf16, strided<[512, 1], offset: ?>>
                %31 = loom.copy_to_tensor %30, %19 on @L1, interconnect : [], broadcast : [1, 1] : memref<?x?xf16, strided<[512, 1], offset: ?>>, memref<?x?xf16> -> tensor<?x?xf16>
                %32 = arith.muli %arg6, %13 : index
                %33 = loom.subview %arg1[%29, %32] [%14, %13] [1, 1], reuse : [seq = false, spat = true, temp = true] : memref<512x4096xf16> to memref<?x?xf16, strided<[4096, 1], offset: ?>>
                %34 = loom.copy_to_tensor %33, %17 on @L1, interconnect : [], broadcast : [1, 1] : memref<?x?xf16, strided<[4096, 1], offset: ?>>, memref<?x?xf16> -> tensor<?x?xf16>
                %35 = linalg.matmul ins(%31, %34 : tensor<?x?xf16>, tensor<?x?xf16>) outs(%arg8 : tensor<?x?xf16>) -> tensor<?x?xf16>
                loom.semaphore_give %17 : memref<?x?xf16>
                loom.semaphore_give %19 : memref<?x?xf16>
                affine.yield %35 : tensor<?x?xf16>
              } {loom.iter_type = #loom.iter_type<sequential>}
              %25 = arith.muli %15, %12 : index
              %26 = arith.muli %arg6, %13 : index
              %27 = loom.subview %arg2[%25, %26] [%12, %13] [1, 1], reuse : [seq = false, spat = false, temp = false] : memref<4096x4096xf16> to memref<?x?xf16, strided<[4096, 1], offset: ?>>
              loom.copy_from_tensor %24, %27 on @DRAM : tensor<?x?xf16>, memref<?x?xf16, strided<[4096, 1], offset: ?>>
              loom.semaphore_give %21 : memref<?x?xf16>
            } {loom.iter_type = #loom.iter_type<temporal>}
          } {loom.iter_type = #loom.iter_type<temporal>}
        } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @y}
      } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @x}
      return
    }
  }
  module {
    func.func @matmul__d0i0_d1i0__f01__d_a(%arg0: memref<4096x512xf16>, %arg1: memref<512x4096xf16>, %arg2: memref<4096x4096xf16>) {
      %cst = arith.constant 0.000000e+00 : f16
      %12 = loom.get_symbolic_block_size @constraints::@BM : index
      %13 = loom.get_symbolic_block_size @constraints::@BN : index
      %14 = loom.get_symbolic_block_size @constraints::@BK : index
      affine.parallel (%arg3) = (0) to (8) {
        affine.parallel (%arg4) = (0) to (8) {
          affine.for %arg5 = 0 to affine_map<()[s0, s1] -> (4096 ceildiv (s0 * 64))>()[%12, %13] {
            affine.for %arg6 = 0 to affine_map<()[s0, s1] -> (4096 ceildiv s1)>()[%12, %13] {
              %15 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 8 + d1 + d2 * 64)>(%arg3, %arg4, %arg5)
              %16 = loom.alloc [%14, %13] on @L1 : memref<?x?xf16>
              %17 = loom.semaphore_take %16 : memref<?x?xf16> -> memref<?x?xf16>
              %18 = loom.alloc [%12, %14] on @L1 : memref<?x?xf16>
              %19 = loom.semaphore_take %18 : memref<?x?xf16> -> memref<?x?xf16>
              %20 = loom.alloc [%12, %13] on @L1 : memref<?x?xf16>
              %21 = loom.semaphore_take %20 : memref<?x?xf16> -> memref<?x?xf16>
              %22 = loom.init_tensor %21[%12, %13] : memref<?x?xf16> -> tensor<?x?xf16>
              %23 = linalg.fill ins(%cst : f16) outs(%22 : tensor<?x?xf16>) -> tensor<?x?xf16>
              %24 = affine.for %arg7 = 0 to affine_map<()[s0] -> (512 ceildiv s0)>()[%14] iter_args(%arg8 = %23) -> (tensor<?x?xf16>) {
                %28 = arith.muli %15, %12 : index
                %29 = arith.muli %arg7, %14 : index
                %30 = loom.subview %arg0[%28, %29] [%12, %14] [1, 1], reuse : [seq = false, spat = false, temp = true] : memref<4096x512xf16> to memref<?x?xf16, strided<[512, 1], offset: ?>>
                %31 = loom.copy_to_tensor %30, %19 on @L1, interconnect : [], broadcast : [1, 1] : memref<?x?xf16, strided<[512, 1], offset: ?>>, memref<?x?xf16> -> tensor<?x?xf16>
                %32 = arith.muli %arg6, %13 : index
                %33 = loom.subview %arg1[%29, %32] [%14, %13] [1, 1], reuse : [seq = false, spat = true, temp = true] : memref<512x4096xf16> to memref<?x?xf16, strided<[4096, 1], offset: ?>>
                %34 = loom.copy_to_tensor %33, %17 on @L1, interconnect : [@horizontal_links, @vertical_links], broadcast : [8, 8] : memref<?x?xf16, strided<[4096, 1], offset: ?>>, memref<?x?xf16> -> tensor<?x?xf16>
                %35 = linalg.matmul ins(%31, %34 : tensor<?x?xf16>, tensor<?x?xf16>) outs(%arg8 : tensor<?x?xf16>) -> tensor<?x?xf16>
                loom.semaphore_give %17 : memref<?x?xf16>
                loom.semaphore_give %19 : memref<?x?xf16>
                affine.yield %35 : tensor<?x?xf16>
              } {loom.iter_type = #loom.iter_type<sequential>}
              %25 = arith.muli %15, %12 : index
              %26 = arith.muli %arg6, %13 : index
              %27 = loom.subview %arg2[%25, %26] [%12, %13] [1, 1], reuse : [seq = false, spat = false, temp = false] : memref<4096x4096xf16> to memref<?x?xf16, strided<[4096, 1], offset: ?>>
              loom.copy_from_tensor %24, %27 on @DRAM : tensor<?x?xf16>, memref<?x?xf16, strided<[4096, 1], offset: ?>>
              loom.semaphore_give %21 : memref<?x?xf16>
            } {loom.iter_type = #loom.iter_type<temporal>}
          } {loom.iter_type = #loom.iter_type<temporal>}
        } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @y}
      } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @x}
      return
    }
  }
  module {
    func.func @matmul__d0i0_d1i0__f01__d_h(%arg0: memref<4096x512xf16>, %arg1: memref<512x4096xf16>, %arg2: memref<4096x4096xf16>) {
      %cst = arith.constant 0.000000e+00 : f16
      %12 = loom.get_symbolic_block_size @constraints::@BM : index
      %13 = loom.get_symbolic_block_size @constraints::@BN : index
      %14 = loom.get_symbolic_block_size @constraints::@BK : index
      affine.parallel (%arg3) = (0) to (8) {
        affine.parallel (%arg4) = (0) to (8) {
          affine.for %arg5 = 0 to affine_map<()[s0, s1] -> (4096 ceildiv (s0 * 64))>()[%12, %13] {
            affine.for %arg6 = 0 to affine_map<()[s0, s1] -> (4096 ceildiv s1)>()[%12, %13] {
              %15 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 8 + d1 + d2 * 64)>(%arg3, %arg4, %arg5)
              %16 = loom.alloc [%14, %13] on @L1 : memref<?x?xf16>
              %17 = loom.semaphore_take %16 : memref<?x?xf16> -> memref<?x?xf16>
              %18 = loom.alloc [%12, %14] on @L1 : memref<?x?xf16>
              %19 = loom.semaphore_take %18 : memref<?x?xf16> -> memref<?x?xf16>
              %20 = loom.alloc [%12, %13] on @L1 : memref<?x?xf16>
              %21 = loom.semaphore_take %20 : memref<?x?xf16> -> memref<?x?xf16>
              %22 = loom.init_tensor %21[%12, %13] : memref<?x?xf16> -> tensor<?x?xf16>
              %23 = linalg.fill ins(%cst : f16) outs(%22 : tensor<?x?xf16>) -> tensor<?x?xf16>
              %24 = affine.for %arg7 = 0 to affine_map<()[s0] -> (512 ceildiv s0)>()[%14] iter_args(%arg8 = %23) -> (tensor<?x?xf16>) {
                %28 = arith.muli %15, %12 : index
                %29 = arith.muli %arg7, %14 : index
                %30 = loom.subview %arg0[%28, %29] [%12, %14] [1, 1], reuse : [seq = false, spat = false, temp = true] : memref<4096x512xf16> to memref<?x?xf16, strided<[512, 1], offset: ?>>
                %31 = loom.copy_to_tensor %30, %19 on @L1, interconnect : [], broadcast : [1, 1] : memref<?x?xf16, strided<[512, 1], offset: ?>>, memref<?x?xf16> -> tensor<?x?xf16>
                %32 = arith.muli %arg6, %13 : index
                %33 = loom.subview %arg1[%29, %32] [%14, %13] [1, 1], reuse : [seq = false, spat = true, temp = true] : memref<512x4096xf16> to memref<?x?xf16, strided<[4096, 1], offset: ?>>
                %34 = loom.copy_to_tensor %33, %17 on @L1, interconnect : [@horizontal_links], broadcast : [1, 8] : memref<?x?xf16, strided<[4096, 1], offset: ?>>, memref<?x?xf16> -> tensor<?x?xf16>
                %35 = linalg.matmul ins(%31, %34 : tensor<?x?xf16>, tensor<?x?xf16>) outs(%arg8 : tensor<?x?xf16>) -> tensor<?x?xf16>
                loom.semaphore_give %17 : memref<?x?xf16>
                loom.semaphore_give %19 : memref<?x?xf16>
                affine.yield %35 : tensor<?x?xf16>
              } {loom.iter_type = #loom.iter_type<sequential>}
              %25 = arith.muli %15, %12 : index
              %26 = arith.muli %arg6, %13 : index
              %27 = loom.subview %arg2[%25, %26] [%12, %13] [1, 1], reuse : [seq = false, spat = false, temp = false] : memref<4096x4096xf16> to memref<?x?xf16, strided<[4096, 1], offset: ?>>
              loom.copy_from_tensor %24, %27 on @DRAM : tensor<?x?xf16>, memref<?x?xf16, strided<[4096, 1], offset: ?>>
              loom.semaphore_give %21 : memref<?x?xf16>
            } {loom.iter_type = #loom.iter_type<temporal>}
          } {loom.iter_type = #loom.iter_type<temporal>}
        } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @y}
      } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @x}
      return
    }
  }
  module {
    func.func @matmul__d0i0_d1i0__f01__d_v(%arg0: memref<4096x512xf16>, %arg1: memref<512x4096xf16>, %arg2: memref<4096x4096xf16>) {
      %cst = arith.constant 0.000000e+00 : f16
      %12 = loom.get_symbolic_block_size @constraints::@BM : index
      %13 = loom.get_symbolic_block_size @constraints::@BN : index
      %14 = loom.get_symbolic_block_size @constraints::@BK : index
      affine.parallel (%arg3) = (0) to (8) {
        affine.parallel (%arg4) = (0) to (8) {
          affine.for %arg5 = 0 to affine_map<()[s0, s1] -> (4096 ceildiv (s0 * 64))>()[%12, %13] {
            affine.for %arg6 = 0 to affine_map<()[s0, s1] -> (4096 ceildiv s1)>()[%12, %13] {
              %15 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 8 + d1 + d2 * 64)>(%arg3, %arg4, %arg5)
              %16 = loom.alloc [%14, %13] on @L1 : memref<?x?xf16>
              %17 = loom.semaphore_take %16 : memref<?x?xf16> -> memref<?x?xf16>
              %18 = loom.alloc [%12, %14] on @L1 : memref<?x?xf16>
              %19 = loom.semaphore_take %18 : memref<?x?xf16> -> memref<?x?xf16>
              %20 = loom.alloc [%12, %13] on @L1 : memref<?x?xf16>
              %21 = loom.semaphore_take %20 : memref<?x?xf16> -> memref<?x?xf16>
              %22 = loom.init_tensor %21[%12, %13] : memref<?x?xf16> -> tensor<?x?xf16>
              %23 = linalg.fill ins(%cst : f16) outs(%22 : tensor<?x?xf16>) -> tensor<?x?xf16>
              %24 = affine.for %arg7 = 0 to affine_map<()[s0] -> (512 ceildiv s0)>()[%14] iter_args(%arg8 = %23) -> (tensor<?x?xf16>) {
                %28 = arith.muli %15, %12 : index
                %29 = arith.muli %arg7, %14 : index
                %30 = loom.subview %arg0[%28, %29] [%12, %14] [1, 1], reuse : [seq = false, spat = false, temp = true] : memref<4096x512xf16> to memref<?x?xf16, strided<[512, 1], offset: ?>>
                %31 = loom.copy_to_tensor %30, %19 on @L1, interconnect : [], broadcast : [1, 1] : memref<?x?xf16, strided<[512, 1], offset: ?>>, memref<?x?xf16> -> tensor<?x?xf16>
                %32 = arith.muli %arg6, %13 : index
                %33 = loom.subview %arg1[%29, %32] [%14, %13] [1, 1], reuse : [seq = false, spat = true, temp = true] : memref<512x4096xf16> to memref<?x?xf16, strided<[4096, 1], offset: ?>>
                %34 = loom.copy_to_tensor %33, %17 on @L1, interconnect : [@vertical_links], broadcast : [8, 1] : memref<?x?xf16, strided<[4096, 1], offset: ?>>, memref<?x?xf16> -> tensor<?x?xf16>
                %35 = linalg.matmul ins(%31, %34 : tensor<?x?xf16>, tensor<?x?xf16>) outs(%arg8 : tensor<?x?xf16>) -> tensor<?x?xf16>
                loom.semaphore_give %17 : memref<?x?xf16>
                loom.semaphore_give %19 : memref<?x?xf16>
                affine.yield %35 : tensor<?x?xf16>
              } {loom.iter_type = #loom.iter_type<sequential>}
              %25 = arith.muli %15, %12 : index
              %26 = arith.muli %arg6, %13 : index
              %27 = loom.subview %arg2[%25, %26] [%12, %13] [1, 1], reuse : [seq = false, spat = false, temp = false] : memref<4096x4096xf16> to memref<?x?xf16, strided<[4096, 1], offset: ?>>
              loom.copy_from_tensor %24, %27 on @DRAM : tensor<?x?xf16>, memref<?x?xf16, strided<[4096, 1], offset: ?>>
              loom.semaphore_give %21 : memref<?x?xf16>
            } {loom.iter_type = #loom.iter_type<temporal>}
          } {loom.iter_type = #loom.iter_type<temporal>}
        } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @y}
      } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @x}
      return
    }
  }
  module {
    func.func @matmul__d0i0_d1i0__f10__d_d(%arg0: memref<4096x512xf16>, %arg1: memref<512x4096xf16>, %arg2: memref<4096x4096xf16>) {
      %cst = arith.constant 0.000000e+00 : f16
      %12 = loom.get_symbolic_block_size @constraints::@BM : index
      %13 = loom.get_symbolic_block_size @constraints::@BN : index
      %14 = loom.get_symbolic_block_size @constraints::@BK : index
      affine.parallel (%arg3) = (0) to (8) {
        affine.parallel (%arg4) = (0) to (8) {
          affine.for %arg5 = 0 to affine_map<()[s0, s1] -> (4096 ceildiv s1)>()[%12, %13] {
            affine.for %arg6 = 0 to affine_map<()[s0, s1] -> (4096 ceildiv (s0 * 64))>()[%12, %13] {
              %15 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 8 + d1 + d2 * 64)>(%arg3, %arg4, %arg6)
              %16 = loom.alloc [%14, %13] on @L1 : memref<?x?xf16>
              %17 = loom.semaphore_take %16 : memref<?x?xf16> -> memref<?x?xf16>
              %18 = loom.alloc [%12, %14] on @L1 : memref<?x?xf16>
              %19 = loom.semaphore_take %18 : memref<?x?xf16> -> memref<?x?xf16>
              %20 = loom.alloc [%12, %13] on @L1 : memref<?x?xf16>
              %21 = loom.semaphore_take %20 : memref<?x?xf16> -> memref<?x?xf16>
              %22 = loom.init_tensor %21[%12, %13] : memref<?x?xf16> -> tensor<?x?xf16>
              %23 = linalg.fill ins(%cst : f16) outs(%22 : tensor<?x?xf16>) -> tensor<?x?xf16>
              %24 = affine.for %arg7 = 0 to affine_map<()[s0] -> (512 ceildiv s0)>()[%14] iter_args(%arg8 = %23) -> (tensor<?x?xf16>) {
                %28 = arith.muli %15, %12 : index
                %29 = arith.muli %arg7, %14 : index
                %30 = loom.subview %arg0[%28, %29] [%12, %14] [1, 1], reuse : [seq = false, spat = false, temp = true] : memref<4096x512xf16> to memref<?x?xf16, strided<[512, 1], offset: ?>>
                %31 = loom.copy_to_tensor %30, %19 on @L1, interconnect : [], broadcast : [1, 1] : memref<?x?xf16, strided<[512, 1], offset: ?>>, memref<?x?xf16> -> tensor<?x?xf16>
                %32 = arith.muli %arg5, %13 : index
                %33 = loom.subview %arg1[%29, %32] [%14, %13] [1, 1], reuse : [seq = false, spat = true, temp = true] : memref<512x4096xf16> to memref<?x?xf16, strided<[4096, 1], offset: ?>>
                %34 = loom.copy_to_tensor %33, %17 on @L1, interconnect : [], broadcast : [1, 1] : memref<?x?xf16, strided<[4096, 1], offset: ?>>, memref<?x?xf16> -> tensor<?x?xf16>
                %35 = linalg.matmul ins(%31, %34 : tensor<?x?xf16>, tensor<?x?xf16>) outs(%arg8 : tensor<?x?xf16>) -> tensor<?x?xf16>
                loom.semaphore_give %17 : memref<?x?xf16>
                loom.semaphore_give %19 : memref<?x?xf16>
                affine.yield %35 : tensor<?x?xf16>
              } {loom.iter_type = #loom.iter_type<sequential>}
              %25 = arith.muli %15, %12 : index
              %26 = arith.muli %arg5, %13 : index
              %27 = loom.subview %arg2[%25, %26] [%12, %13] [1, 1], reuse : [seq = false, spat = false, temp = false] : memref<4096x4096xf16> to memref<?x?xf16, strided<[4096, 1], offset: ?>>
              loom.copy_from_tensor %24, %27 on @DRAM : tensor<?x?xf16>, memref<?x?xf16, strided<[4096, 1], offset: ?>>
              loom.semaphore_give %21 : memref<?x?xf16>
            } {loom.iter_type = #loom.iter_type<temporal>}
          } {loom.iter_type = #loom.iter_type<temporal>}
        } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @y}
      } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @x}
      return
    }
  }
  module {
    func.func @matmul__d0i0_d1i0__f10__d_a(%arg0: memref<4096x512xf16>, %arg1: memref<512x4096xf16>, %arg2: memref<4096x4096xf16>) {
      %cst = arith.constant 0.000000e+00 : f16
      %12 = loom.get_symbolic_block_size @constraints::@BM : index
      %13 = loom.get_symbolic_block_size @constraints::@BN : index
      %14 = loom.get_symbolic_block_size @constraints::@BK : index
      affine.parallel (%arg3) = (0) to (8) {
        affine.parallel (%arg4) = (0) to (8) {
          affine.for %arg5 = 0 to affine_map<()[s0, s1] -> (4096 ceildiv s1)>()[%12, %13] {
            affine.for %arg6 = 0 to affine_map<()[s0, s1] -> (4096 ceildiv (s0 * 64))>()[%12, %13] {
              %15 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 8 + d1 + d2 * 64)>(%arg3, %arg4, %arg6)
              %16 = loom.alloc [%14, %13] on @L1 : memref<?x?xf16>
              %17 = loom.semaphore_take %16 : memref<?x?xf16> -> memref<?x?xf16>
              %18 = loom.alloc [%12, %14] on @L1 : memref<?x?xf16>
              %19 = loom.semaphore_take %18 : memref<?x?xf16> -> memref<?x?xf16>
              %20 = loom.alloc [%12, %13] on @L1 : memref<?x?xf16>
              %21 = loom.semaphore_take %20 : memref<?x?xf16> -> memref<?x?xf16>
              %22 = loom.init_tensor %21[%12, %13] : memref<?x?xf16> -> tensor<?x?xf16>
              %23 = linalg.fill ins(%cst : f16) outs(%22 : tensor<?x?xf16>) -> tensor<?x?xf16>
              %24 = affine.for %arg7 = 0 to affine_map<()[s0] -> (512 ceildiv s0)>()[%14] iter_args(%arg8 = %23) -> (tensor<?x?xf16>) {
                %28 = arith.muli %15, %12 : index
                %29 = arith.muli %arg7, %14 : index
                %30 = loom.subview %arg0[%28, %29] [%12, %14] [1, 1], reuse : [seq = false, spat = false, temp = true] : memref<4096x512xf16> to memref<?x?xf16, strided<[512, 1], offset: ?>>
                %31 = loom.copy_to_tensor %30, %19 on @L1, interconnect : [], broadcast : [1, 1] : memref<?x?xf16, strided<[512, 1], offset: ?>>, memref<?x?xf16> -> tensor<?x?xf16>
                %32 = arith.muli %arg5, %13 : index
                %33 = loom.subview %arg1[%29, %32] [%14, %13] [1, 1], reuse : [seq = false, spat = true, temp = true] : memref<512x4096xf16> to memref<?x?xf16, strided<[4096, 1], offset: ?>>
                %34 = loom.copy_to_tensor %33, %17 on @L1, interconnect : [@horizontal_links, @vertical_links], broadcast : [8, 8] : memref<?x?xf16, strided<[4096, 1], offset: ?>>, memref<?x?xf16> -> tensor<?x?xf16>
                %35 = linalg.matmul ins(%31, %34 : tensor<?x?xf16>, tensor<?x?xf16>) outs(%arg8 : tensor<?x?xf16>) -> tensor<?x?xf16>
                loom.semaphore_give %17 : memref<?x?xf16>
                loom.semaphore_give %19 : memref<?x?xf16>
                affine.yield %35 : tensor<?x?xf16>
              } {loom.iter_type = #loom.iter_type<sequential>}
              %25 = arith.muli %15, %12 : index
              %26 = arith.muli %arg5, %13 : index
              %27 = loom.subview %arg2[%25, %26] [%12, %13] [1, 1], reuse : [seq = false, spat = false, temp = false] : memref<4096x4096xf16> to memref<?x?xf16, strided<[4096, 1], offset: ?>>
              loom.copy_from_tensor %24, %27 on @DRAM : tensor<?x?xf16>, memref<?x?xf16, strided<[4096, 1], offset: ?>>
              loom.semaphore_give %21 : memref<?x?xf16>
            } {loom.iter_type = #loom.iter_type<temporal>}
          } {loom.iter_type = #loom.iter_type<temporal>}
        } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @y}
      } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @x}
      return
    }
  }
  module {
    func.func @matmul__d0i0_d1i0__f10__d_h(%arg0: memref<4096x512xf16>, %arg1: memref<512x4096xf16>, %arg2: memref<4096x4096xf16>) {
      %cst = arith.constant 0.000000e+00 : f16
      %12 = loom.get_symbolic_block_size @constraints::@BM : index
      %13 = loom.get_symbolic_block_size @constraints::@BN : index
      %14 = loom.get_symbolic_block_size @constraints::@BK : index
      affine.parallel (%arg3) = (0) to (8) {
        affine.parallel (%arg4) = (0) to (8) {
          affine.for %arg5 = 0 to affine_map<()[s0, s1] -> (4096 ceildiv s1)>()[%12, %13] {
            affine.for %arg6 = 0 to affine_map<()[s0, s1] -> (4096 ceildiv (s0 * 64))>()[%12, %13] {
              %15 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 8 + d1 + d2 * 64)>(%arg3, %arg4, %arg6)
              %16 = loom.alloc [%14, %13] on @L1 : memref<?x?xf16>
              %17 = loom.semaphore_take %16 : memref<?x?xf16> -> memref<?x?xf16>
              %18 = loom.alloc [%12, %14] on @L1 : memref<?x?xf16>
              %19 = loom.semaphore_take %18 : memref<?x?xf16> -> memref<?x?xf16>
              %20 = loom.alloc [%12, %13] on @L1 : memref<?x?xf16>
              %21 = loom.semaphore_take %20 : memref<?x?xf16> -> memref<?x?xf16>
              %22 = loom.init_tensor %21[%12, %13] : memref<?x?xf16> -> tensor<?x?xf16>
              %23 = linalg.fill ins(%cst : f16) outs(%22 : tensor<?x?xf16>) -> tensor<?x?xf16>
              %24 = affine.for %arg7 = 0 to affine_map<()[s0] -> (512 ceildiv s0)>()[%14] iter_args(%arg8 = %23) -> (tensor<?x?xf16>) {
                %28 = arith.muli %15, %12 : index
                %29 = arith.muli %arg7, %14 : index
                %30 = loom.subview %arg0[%28, %29] [%12, %14] [1, 1], reuse : [seq = false, spat = false, temp = true] : memref<4096x512xf16> to memref<?x?xf16, strided<[512, 1], offset: ?>>
                %31 = loom.copy_to_tensor %30, %19 on @L1, interconnect : [], broadcast : [1, 1] : memref<?x?xf16, strided<[512, 1], offset: ?>>, memref<?x?xf16> -> tensor<?x?xf16>
                %32 = arith.muli %arg5, %13 : index
                %33 = loom.subview %arg1[%29, %32] [%14, %13] [1, 1], reuse : [seq = false, spat = true, temp = true] : memref<512x4096xf16> to memref<?x?xf16, strided<[4096, 1], offset: ?>>
                %34 = loom.copy_to_tensor %33, %17 on @L1, interconnect : [@horizontal_links], broadcast : [1, 8] : memref<?x?xf16, strided<[4096, 1], offset: ?>>, memref<?x?xf16> -> tensor<?x?xf16>
                %35 = linalg.matmul ins(%31, %34 : tensor<?x?xf16>, tensor<?x?xf16>) outs(%arg8 : tensor<?x?xf16>) -> tensor<?x?xf16>
                loom.semaphore_give %17 : memref<?x?xf16>
                loom.semaphore_give %19 : memref<?x?xf16>
                affine.yield %35 : tensor<?x?xf16>
              } {loom.iter_type = #loom.iter_type<sequential>}
              %25 = arith.muli %15, %12 : index
              %26 = arith.muli %arg5, %13 : index
              %27 = loom.subview %arg2[%25, %26] [%12, %13] [1, 1], reuse : [seq = false, spat = false, temp = false] : memref<4096x4096xf16> to memref<?x?xf16, strided<[4096, 1], offset: ?>>
              loom.copy_from_tensor %24, %27 on @DRAM : tensor<?x?xf16>, memref<?x?xf16, strided<[4096, 1], offset: ?>>
              loom.semaphore_give %21 : memref<?x?xf16>
            } {loom.iter_type = #loom.iter_type<temporal>}
          } {loom.iter_type = #loom.iter_type<temporal>}
        } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @y}
      } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @x}
      return
    }
  }
  module {
    func.func @matmul__d0i0_d1i0__f10__d_v(%arg0: memref<4096x512xf16>, %arg1: memref<512x4096xf16>, %arg2: memref<4096x4096xf16>) {
      %cst = arith.constant 0.000000e+00 : f16
      %12 = loom.get_symbolic_block_size @constraints::@BM : index
      %13 = loom.get_symbolic_block_size @constraints::@BN : index
      %14 = loom.get_symbolic_block_size @constraints::@BK : index
      affine.parallel (%arg3) = (0) to (8) {
        affine.parallel (%arg4) = (0) to (8) {
          affine.for %arg5 = 0 to affine_map<()[s0, s1] -> (4096 ceildiv s1)>()[%12, %13] {
            affine.for %arg6 = 0 to affine_map<()[s0, s1] -> (4096 ceildiv (s0 * 64))>()[%12, %13] {
              %15 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 8 + d1 + d2 * 64)>(%arg3, %arg4, %arg6)
              %16 = loom.alloc [%14, %13] on @L1 : memref<?x?xf16>
              %17 = loom.semaphore_take %16 : memref<?x?xf16> -> memref<?x?xf16>
              %18 = loom.alloc [%12, %14] on @L1 : memref<?x?xf16>
              %19 = loom.semaphore_take %18 : memref<?x?xf16> -> memref<?x?xf16>
              %20 = loom.alloc [%12, %13] on @L1 : memref<?x?xf16>
              %21 = loom.semaphore_take %20 : memref<?x?xf16> -> memref<?x?xf16>
              %22 = loom.init_tensor %21[%12, %13] : memref<?x?xf16> -> tensor<?x?xf16>
              %23 = linalg.fill ins(%cst : f16) outs(%22 : tensor<?x?xf16>) -> tensor<?x?xf16>
              %24 = affine.for %arg7 = 0 to affine_map<()[s0] -> (512 ceildiv s0)>()[%14] iter_args(%arg8 = %23) -> (tensor<?x?xf16>) {
                %28 = arith.muli %15, %12 : index
                %29 = arith.muli %arg7, %14 : index
                %30 = loom.subview %arg0[%28, %29] [%12, %14] [1, 1], reuse : [seq = false, spat = false, temp = true] : memref<4096x512xf16> to memref<?x?xf16, strided<[512, 1], offset: ?>>
                %31 = loom.copy_to_tensor %30, %19 on @L1, interconnect : [], broadcast : [1, 1] : memref<?x?xf16, strided<[512, 1], offset: ?>>, memref<?x?xf16> -> tensor<?x?xf16>
                %32 = arith.muli %arg5, %13 : index
                %33 = loom.subview %arg1[%29, %32] [%14, %13] [1, 1], reuse : [seq = false, spat = true, temp = true] : memref<512x4096xf16> to memref<?x?xf16, strided<[4096, 1], offset: ?>>
                %34 = loom.copy_to_tensor %33, %17 on @L1, interconnect : [@vertical_links], broadcast : [8, 1] : memref<?x?xf16, strided<[4096, 1], offset: ?>>, memref<?x?xf16> -> tensor<?x?xf16>
                %35 = linalg.matmul ins(%31, %34 : tensor<?x?xf16>, tensor<?x?xf16>) outs(%arg8 : tensor<?x?xf16>) -> tensor<?x?xf16>
                loom.semaphore_give %17 : memref<?x?xf16>
                loom.semaphore_give %19 : memref<?x?xf16>
                affine.yield %35 : tensor<?x?xf16>
              } {loom.iter_type = #loom.iter_type<sequential>}
              %25 = arith.muli %15, %12 : index
              %26 = arith.muli %arg5, %13 : index
              %27 = loom.subview %arg2[%25, %26] [%12, %13] [1, 1], reuse : [seq = false, spat = false, temp = false] : memref<4096x4096xf16> to memref<?x?xf16, strided<[4096, 1], offset: ?>>
              loom.copy_from_tensor %24, %27 on @DRAM : tensor<?x?xf16>, memref<?x?xf16, strided<[4096, 1], offset: ?>>
              loom.semaphore_give %21 : memref<?x?xf16>
            } {loom.iter_type = #loom.iter_type<temporal>}
          } {loom.iter_type = #loom.iter_type<temporal>}
        } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @y}
      } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @x}
      return
    }
  }
  module {
    func.func @matmul__d1i0_d0i0__f01__d_d(%arg0: memref<4096x512xf16>, %arg1: memref<512x4096xf16>, %arg2: memref<4096x4096xf16>) {
      %cst = arith.constant 0.000000e+00 : f16
      %12 = loom.get_symbolic_block_size @constraints::@BM : index
      %13 = loom.get_symbolic_block_size @constraints::@BN : index
      %14 = loom.get_symbolic_block_size @constraints::@BK : index
      affine.parallel (%arg3) = (0) to (8) {
        affine.parallel (%arg4) = (0) to (8) {
          affine.for %arg5 = 0 to affine_map<()[s0, s1] -> (4096 ceildiv (s0 * 64))>()[%12, %13] {
            affine.for %arg6 = 0 to affine_map<()[s0, s1] -> (4096 ceildiv s1)>()[%12, %13] {
              %15 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 8 + d1 + d2 * 64)>(%arg3, %arg4, %arg5)
              %16 = loom.alloc [%14, %13] on @L1 : memref<?x?xf16>
              %17 = loom.semaphore_take %16 : memref<?x?xf16> -> memref<?x?xf16>
              %18 = loom.alloc [%12, %14] on @L1 : memref<?x?xf16>
              %19 = loom.semaphore_take %18 : memref<?x?xf16> -> memref<?x?xf16>
              %20 = loom.alloc [%12, %13] on @L1 : memref<?x?xf16>
              %21 = loom.semaphore_take %20 : memref<?x?xf16> -> memref<?x?xf16>
              %22 = loom.init_tensor %21[%12, %13] : memref<?x?xf16> -> tensor<?x?xf16>
              %23 = linalg.fill ins(%cst : f16) outs(%22 : tensor<?x?xf16>) -> tensor<?x?xf16>
              %24 = affine.for %arg7 = 0 to affine_map<()[s0] -> (512 ceildiv s0)>()[%14] iter_args(%arg8 = %23) -> (tensor<?x?xf16>) {
                %28 = arith.muli %15, %12 : index
                %29 = arith.muli %arg7, %14 : index
                %30 = loom.subview %arg0[%28, %29] [%12, %14] [1, 1], reuse : [seq = false, spat = false, temp = true] : memref<4096x512xf16> to memref<?x?xf16, strided<[512, 1], offset: ?>>
                %31 = loom.copy_to_tensor %30, %19 on @L1, interconnect : [], broadcast : [1, 1] : memref<?x?xf16, strided<[512, 1], offset: ?>>, memref<?x?xf16> -> tensor<?x?xf16>
                %32 = arith.muli %arg6, %13 : index
                %33 = loom.subview %arg1[%29, %32] [%14, %13] [1, 1], reuse : [seq = false, spat = true, temp = true] : memref<512x4096xf16> to memref<?x?xf16, strided<[4096, 1], offset: ?>>
                %34 = loom.copy_to_tensor %33, %17 on @L1, interconnect : [], broadcast : [1, 1] : memref<?x?xf16, strided<[4096, 1], offset: ?>>, memref<?x?xf16> -> tensor<?x?xf16>
                %35 = linalg.matmul ins(%31, %34 : tensor<?x?xf16>, tensor<?x?xf16>) outs(%arg8 : tensor<?x?xf16>) -> tensor<?x?xf16>
                loom.semaphore_give %17 : memref<?x?xf16>
                loom.semaphore_give %19 : memref<?x?xf16>
                affine.yield %35 : tensor<?x?xf16>
              } {loom.iter_type = #loom.iter_type<sequential>}
              %25 = arith.muli %15, %12 : index
              %26 = arith.muli %arg6, %13 : index
              %27 = loom.subview %arg2[%25, %26] [%12, %13] [1, 1], reuse : [seq = false, spat = false, temp = false] : memref<4096x4096xf16> to memref<?x?xf16, strided<[4096, 1], offset: ?>>
              loom.copy_from_tensor %24, %27 on @DRAM : tensor<?x?xf16>, memref<?x?xf16, strided<[4096, 1], offset: ?>>
              loom.semaphore_give %21 : memref<?x?xf16>
            } {loom.iter_type = #loom.iter_type<temporal>}
          } {loom.iter_type = #loom.iter_type<temporal>}
        } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @x}
      } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @y}
      return
    }
  }
  module {
    func.func @matmul__d1i0_d0i0__f01__d_a(%arg0: memref<4096x512xf16>, %arg1: memref<512x4096xf16>, %arg2: memref<4096x4096xf16>) {
      %cst = arith.constant 0.000000e+00 : f16
      %12 = loom.get_symbolic_block_size @constraints::@BM : index
      %13 = loom.get_symbolic_block_size @constraints::@BN : index
      %14 = loom.get_symbolic_block_size @constraints::@BK : index
      affine.parallel (%arg3) = (0) to (8) {
        affine.parallel (%arg4) = (0) to (8) {
          affine.for %arg5 = 0 to affine_map<()[s0, s1] -> (4096 ceildiv (s0 * 64))>()[%12, %13] {
            affine.for %arg6 = 0 to affine_map<()[s0, s1] -> (4096 ceildiv s1)>()[%12, %13] {
              %15 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 8 + d1 + d2 * 64)>(%arg3, %arg4, %arg5)
              %16 = loom.alloc [%14, %13] on @L1 : memref<?x?xf16>
              %17 = loom.semaphore_take %16 : memref<?x?xf16> -> memref<?x?xf16>
              %18 = loom.alloc [%12, %14] on @L1 : memref<?x?xf16>
              %19 = loom.semaphore_take %18 : memref<?x?xf16> -> memref<?x?xf16>
              %20 = loom.alloc [%12, %13] on @L1 : memref<?x?xf16>
              %21 = loom.semaphore_take %20 : memref<?x?xf16> -> memref<?x?xf16>
              %22 = loom.init_tensor %21[%12, %13] : memref<?x?xf16> -> tensor<?x?xf16>
              %23 = linalg.fill ins(%cst : f16) outs(%22 : tensor<?x?xf16>) -> tensor<?x?xf16>
              %24 = affine.for %arg7 = 0 to affine_map<()[s0] -> (512 ceildiv s0)>()[%14] iter_args(%arg8 = %23) -> (tensor<?x?xf16>) {
                %28 = arith.muli %15, %12 : index
                %29 = arith.muli %arg7, %14 : index
                %30 = loom.subview %arg0[%28, %29] [%12, %14] [1, 1], reuse : [seq = false, spat = false, temp = true] : memref<4096x512xf16> to memref<?x?xf16, strided<[512, 1], offset: ?>>
                %31 = loom.copy_to_tensor %30, %19 on @L1, interconnect : [], broadcast : [1, 1] : memref<?x?xf16, strided<[512, 1], offset: ?>>, memref<?x?xf16> -> tensor<?x?xf16>
                %32 = arith.muli %arg6, %13 : index
                %33 = loom.subview %arg1[%29, %32] [%14, %13] [1, 1], reuse : [seq = false, spat = true, temp = true] : memref<512x4096xf16> to memref<?x?xf16, strided<[4096, 1], offset: ?>>
                %34 = loom.copy_to_tensor %33, %17 on @L1, interconnect : [@horizontal_links, @vertical_links], broadcast : [8, 8] : memref<?x?xf16, strided<[4096, 1], offset: ?>>, memref<?x?xf16> -> tensor<?x?xf16>
                %35 = linalg.matmul ins(%31, %34 : tensor<?x?xf16>, tensor<?x?xf16>) outs(%arg8 : tensor<?x?xf16>) -> tensor<?x?xf16>
                loom.semaphore_give %17 : memref<?x?xf16>
                loom.semaphore_give %19 : memref<?x?xf16>
                affine.yield %35 : tensor<?x?xf16>
              } {loom.iter_type = #loom.iter_type<sequential>}
              %25 = arith.muli %15, %12 : index
              %26 = arith.muli %arg6, %13 : index
              %27 = loom.subview %arg2[%25, %26] [%12, %13] [1, 1], reuse : [seq = false, spat = false, temp = false] : memref<4096x4096xf16> to memref<?x?xf16, strided<[4096, 1], offset: ?>>
              loom.copy_from_tensor %24, %27 on @DRAM : tensor<?x?xf16>, memref<?x?xf16, strided<[4096, 1], offset: ?>>
              loom.semaphore_give %21 : memref<?x?xf16>
            } {loom.iter_type = #loom.iter_type<temporal>}
          } {loom.iter_type = #loom.iter_type<temporal>}
        } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @x}
      } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @y}
      return
    }
  }
  module {
    func.func @matmul__d1i0_d0i0__f01__d_h(%arg0: memref<4096x512xf16>, %arg1: memref<512x4096xf16>, %arg2: memref<4096x4096xf16>) {
      %cst = arith.constant 0.000000e+00 : f16
      %12 = loom.get_symbolic_block_size @constraints::@BM : index
      %13 = loom.get_symbolic_block_size @constraints::@BN : index
      %14 = loom.get_symbolic_block_size @constraints::@BK : index
      affine.parallel (%arg3) = (0) to (8) {
        affine.parallel (%arg4) = (0) to (8) {
          affine.for %arg5 = 0 to affine_map<()[s0, s1] -> (4096 ceildiv (s0 * 64))>()[%12, %13] {
            affine.for %arg6 = 0 to affine_map<()[s0, s1] -> (4096 ceildiv s1)>()[%12, %13] {
              %15 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 8 + d1 + d2 * 64)>(%arg3, %arg4, %arg5)
              %16 = loom.alloc [%14, %13] on @L1 : memref<?x?xf16>
              %17 = loom.semaphore_take %16 : memref<?x?xf16> -> memref<?x?xf16>
              %18 = loom.alloc [%12, %14] on @L1 : memref<?x?xf16>
              %19 = loom.semaphore_take %18 : memref<?x?xf16> -> memref<?x?xf16>
              %20 = loom.alloc [%12, %13] on @L1 : memref<?x?xf16>
              %21 = loom.semaphore_take %20 : memref<?x?xf16> -> memref<?x?xf16>
              %22 = loom.init_tensor %21[%12, %13] : memref<?x?xf16> -> tensor<?x?xf16>
              %23 = linalg.fill ins(%cst : f16) outs(%22 : tensor<?x?xf16>) -> tensor<?x?xf16>
              %24 = affine.for %arg7 = 0 to affine_map<()[s0] -> (512 ceildiv s0)>()[%14] iter_args(%arg8 = %23) -> (tensor<?x?xf16>) {
                %28 = arith.muli %15, %12 : index
                %29 = arith.muli %arg7, %14 : index
                %30 = loom.subview %arg0[%28, %29] [%12, %14] [1, 1], reuse : [seq = false, spat = false, temp = true] : memref<4096x512xf16> to memref<?x?xf16, strided<[512, 1], offset: ?>>
                %31 = loom.copy_to_tensor %30, %19 on @L1, interconnect : [], broadcast : [1, 1] : memref<?x?xf16, strided<[512, 1], offset: ?>>, memref<?x?xf16> -> tensor<?x?xf16>
                %32 = arith.muli %arg6, %13 : index
                %33 = loom.subview %arg1[%29, %32] [%14, %13] [1, 1], reuse : [seq = false, spat = true, temp = true] : memref<512x4096xf16> to memref<?x?xf16, strided<[4096, 1], offset: ?>>
                %34 = loom.copy_to_tensor %33, %17 on @L1, interconnect : [@horizontal_links], broadcast : [1, 8] : memref<?x?xf16, strided<[4096, 1], offset: ?>>, memref<?x?xf16> -> tensor<?x?xf16>
                %35 = linalg.matmul ins(%31, %34 : tensor<?x?xf16>, tensor<?x?xf16>) outs(%arg8 : tensor<?x?xf16>) -> tensor<?x?xf16>
                loom.semaphore_give %17 : memref<?x?xf16>
                loom.semaphore_give %19 : memref<?x?xf16>
                affine.yield %35 : tensor<?x?xf16>
              } {loom.iter_type = #loom.iter_type<sequential>}
              %25 = arith.muli %15, %12 : index
              %26 = arith.muli %arg6, %13 : index
              %27 = loom.subview %arg2[%25, %26] [%12, %13] [1, 1], reuse : [seq = false, spat = false, temp = false] : memref<4096x4096xf16> to memref<?x?xf16, strided<[4096, 1], offset: ?>>
              loom.copy_from_tensor %24, %27 on @DRAM : tensor<?x?xf16>, memref<?x?xf16, strided<[4096, 1], offset: ?>>
              loom.semaphore_give %21 : memref<?x?xf16>
            } {loom.iter_type = #loom.iter_type<temporal>}
          } {loom.iter_type = #loom.iter_type<temporal>}
        } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @x}
      } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @y}
      return
    }
  }
  module {
    func.func @matmul__d1i0_d0i0__f01__d_v(%arg0: memref<4096x512xf16>, %arg1: memref<512x4096xf16>, %arg2: memref<4096x4096xf16>) {
      %cst = arith.constant 0.000000e+00 : f16
      %12 = loom.get_symbolic_block_size @constraints::@BM : index
      %13 = loom.get_symbolic_block_size @constraints::@BN : index
      %14 = loom.get_symbolic_block_size @constraints::@BK : index
      affine.parallel (%arg3) = (0) to (8) {
        affine.parallel (%arg4) = (0) to (8) {
          affine.for %arg5 = 0 to affine_map<()[s0, s1] -> (4096 ceildiv (s0 * 64))>()[%12, %13] {
            affine.for %arg6 = 0 to affine_map<()[s0, s1] -> (4096 ceildiv s1)>()[%12, %13] {
              %15 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 8 + d1 + d2 * 64)>(%arg3, %arg4, %arg5)
              %16 = loom.alloc [%14, %13] on @L1 : memref<?x?xf16>
              %17 = loom.semaphore_take %16 : memref<?x?xf16> -> memref<?x?xf16>
              %18 = loom.alloc [%12, %14] on @L1 : memref<?x?xf16>
              %19 = loom.semaphore_take %18 : memref<?x?xf16> -> memref<?x?xf16>
              %20 = loom.alloc [%12, %13] on @L1 : memref<?x?xf16>
              %21 = loom.semaphore_take %20 : memref<?x?xf16> -> memref<?x?xf16>
              %22 = loom.init_tensor %21[%12, %13] : memref<?x?xf16> -> tensor<?x?xf16>
              %23 = linalg.fill ins(%cst : f16) outs(%22 : tensor<?x?xf16>) -> tensor<?x?xf16>
              %24 = affine.for %arg7 = 0 to affine_map<()[s0] -> (512 ceildiv s0)>()[%14] iter_args(%arg8 = %23) -> (tensor<?x?xf16>) {
                %28 = arith.muli %15, %12 : index
                %29 = arith.muli %arg7, %14 : index
                %30 = loom.subview %arg0[%28, %29] [%12, %14] [1, 1], reuse : [seq = false, spat = false, temp = true] : memref<4096x512xf16> to memref<?x?xf16, strided<[512, 1], offset: ?>>
                %31 = loom.copy_to_tensor %30, %19 on @L1, interconnect : [], broadcast : [1, 1] : memref<?x?xf16, strided<[512, 1], offset: ?>>, memref<?x?xf16> -> tensor<?x?xf16>
                %32 = arith.muli %arg6, %13 : index
                %33 = loom.subview %arg1[%29, %32] [%14, %13] [1, 1], reuse : [seq = false, spat = true, temp = true] : memref<512x4096xf16> to memref<?x?xf16, strided<[4096, 1], offset: ?>>
                %34 = loom.copy_to_tensor %33, %17 on @L1, interconnect : [@vertical_links], broadcast : [8, 1] : memref<?x?xf16, strided<[4096, 1], offset: ?>>, memref<?x?xf16> -> tensor<?x?xf16>
                %35 = linalg.matmul ins(%31, %34 : tensor<?x?xf16>, tensor<?x?xf16>) outs(%arg8 : tensor<?x?xf16>) -> tensor<?x?xf16>
                loom.semaphore_give %17 : memref<?x?xf16>
                loom.semaphore_give %19 : memref<?x?xf16>
                affine.yield %35 : tensor<?x?xf16>
              } {loom.iter_type = #loom.iter_type<sequential>}
              %25 = arith.muli %15, %12 : index
              %26 = arith.muli %arg6, %13 : index
              %27 = loom.subview %arg2[%25, %26] [%12, %13] [1, 1], reuse : [seq = false, spat = false, temp = false] : memref<4096x4096xf16> to memref<?x?xf16, strided<[4096, 1], offset: ?>>
              loom.copy_from_tensor %24, %27 on @DRAM : tensor<?x?xf16>, memref<?x?xf16, strided<[4096, 1], offset: ?>>
              loom.semaphore_give %21 : memref<?x?xf16>
            } {loom.iter_type = #loom.iter_type<temporal>}
          } {loom.iter_type = #loom.iter_type<temporal>}
        } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @x}
      } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @y}
      return
    }
  }
  module {
    func.func @matmul__d1i0_d0i0__f10__d_d(%arg0: memref<4096x512xf16>, %arg1: memref<512x4096xf16>, %arg2: memref<4096x4096xf16>) {
      %cst = arith.constant 0.000000e+00 : f16
      %12 = loom.get_symbolic_block_size @constraints::@BM : index
      %13 = loom.get_symbolic_block_size @constraints::@BN : index
      %14 = loom.get_symbolic_block_size @constraints::@BK : index
      affine.parallel (%arg3) = (0) to (8) {
        affine.parallel (%arg4) = (0) to (8) {
          affine.for %arg5 = 0 to affine_map<()[s0, s1] -> (4096 ceildiv s1)>()[%12, %13] {
            affine.for %arg6 = 0 to affine_map<()[s0, s1] -> (4096 ceildiv (s0 * 64))>()[%12, %13] {
              %15 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 8 + d1 + d2 * 64)>(%arg3, %arg4, %arg6)
              %16 = loom.alloc [%14, %13] on @L1 : memref<?x?xf16>
              %17 = loom.semaphore_take %16 : memref<?x?xf16> -> memref<?x?xf16>
              %18 = loom.alloc [%12, %14] on @L1 : memref<?x?xf16>
              %19 = loom.semaphore_take %18 : memref<?x?xf16> -> memref<?x?xf16>
              %20 = loom.alloc [%12, %13] on @L1 : memref<?x?xf16>
              %21 = loom.semaphore_take %20 : memref<?x?xf16> -> memref<?x?xf16>
              %22 = loom.init_tensor %21[%12, %13] : memref<?x?xf16> -> tensor<?x?xf16>
              %23 = linalg.fill ins(%cst : f16) outs(%22 : tensor<?x?xf16>) -> tensor<?x?xf16>
              %24 = affine.for %arg7 = 0 to affine_map<()[s0] -> (512 ceildiv s0)>()[%14] iter_args(%arg8 = %23) -> (tensor<?x?xf16>) {
                %28 = arith.muli %15, %12 : index
                %29 = arith.muli %arg7, %14 : index
                %30 = loom.subview %arg0[%28, %29] [%12, %14] [1, 1], reuse : [seq = false, spat = false, temp = true] : memref<4096x512xf16> to memref<?x?xf16, strided<[512, 1], offset: ?>>
                %31 = loom.copy_to_tensor %30, %19 on @L1, interconnect : [], broadcast : [1, 1] : memref<?x?xf16, strided<[512, 1], offset: ?>>, memref<?x?xf16> -> tensor<?x?xf16>
                %32 = arith.muli %arg5, %13 : index
                %33 = loom.subview %arg1[%29, %32] [%14, %13] [1, 1], reuse : [seq = false, spat = true, temp = true] : memref<512x4096xf16> to memref<?x?xf16, strided<[4096, 1], offset: ?>>
                %34 = loom.copy_to_tensor %33, %17 on @L1, interconnect : [], broadcast : [1, 1] : memref<?x?xf16, strided<[4096, 1], offset: ?>>, memref<?x?xf16> -> tensor<?x?xf16>
                %35 = linalg.matmul ins(%31, %34 : tensor<?x?xf16>, tensor<?x?xf16>) outs(%arg8 : tensor<?x?xf16>) -> tensor<?x?xf16>
                loom.semaphore_give %17 : memref<?x?xf16>
                loom.semaphore_give %19 : memref<?x?xf16>
                affine.yield %35 : tensor<?x?xf16>
              } {loom.iter_type = #loom.iter_type<sequential>}
              %25 = arith.muli %15, %12 : index
              %26 = arith.muli %arg5, %13 : index
              %27 = loom.subview %arg2[%25, %26] [%12, %13] [1, 1], reuse : [seq = false, spat = false, temp = false] : memref<4096x4096xf16> to memref<?x?xf16, strided<[4096, 1], offset: ?>>
              loom.copy_from_tensor %24, %27 on @DRAM : tensor<?x?xf16>, memref<?x?xf16, strided<[4096, 1], offset: ?>>
              loom.semaphore_give %21 : memref<?x?xf16>
            } {loom.iter_type = #loom.iter_type<temporal>}
          } {loom.iter_type = #loom.iter_type<temporal>}
        } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @x}
      } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @y}
      return
    }
  }
  module {
    func.func @matmul__d1i0_d0i0__f10__d_a(%arg0: memref<4096x512xf16>, %arg1: memref<512x4096xf16>, %arg2: memref<4096x4096xf16>) {
      %cst = arith.constant 0.000000e+00 : f16
      %12 = loom.get_symbolic_block_size @constraints::@BM : index
      %13 = loom.get_symbolic_block_size @constraints::@BN : index
      %14 = loom.get_symbolic_block_size @constraints::@BK : index
      affine.parallel (%arg3) = (0) to (8) {
        affine.parallel (%arg4) = (0) to (8) {
          affine.for %arg5 = 0 to affine_map<()[s0, s1] -> (4096 ceildiv s1)>()[%12, %13] {
            affine.for %arg6 = 0 to affine_map<()[s0, s1] -> (4096 ceildiv (s0 * 64))>()[%12, %13] {
              %15 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 8 + d1 + d2 * 64)>(%arg3, %arg4, %arg6)
              %16 = loom.alloc [%14, %13] on @L1 : memref<?x?xf16>
              %17 = loom.semaphore_take %16 : memref<?x?xf16> -> memref<?x?xf16>
              %18 = loom.alloc [%12, %14] on @L1 : memref<?x?xf16>
              %19 = loom.semaphore_take %18 : memref<?x?xf16> -> memref<?x?xf16>
              %20 = loom.alloc [%12, %13] on @L1 : memref<?x?xf16>
              %21 = loom.semaphore_take %20 : memref<?x?xf16> -> memref<?x?xf16>
              %22 = loom.init_tensor %21[%12, %13] : memref<?x?xf16> -> tensor<?x?xf16>
              %23 = linalg.fill ins(%cst : f16) outs(%22 : tensor<?x?xf16>) -> tensor<?x?xf16>
              %24 = affine.for %arg7 = 0 to affine_map<()[s0] -> (512 ceildiv s0)>()[%14] iter_args(%arg8 = %23) -> (tensor<?x?xf16>) {
                %28 = arith.muli %15, %12 : index
                %29 = arith.muli %arg7, %14 : index
                %30 = loom.subview %arg0[%28, %29] [%12, %14] [1, 1], reuse : [seq = false, spat = false, temp = true] : memref<4096x512xf16> to memref<?x?xf16, strided<[512, 1], offset: ?>>
                %31 = loom.copy_to_tensor %30, %19 on @L1, interconnect : [], broadcast : [1, 1] : memref<?x?xf16, strided<[512, 1], offset: ?>>, memref<?x?xf16> -> tensor<?x?xf16>
                %32 = arith.muli %arg5, %13 : index
                %33 = loom.subview %arg1[%29, %32] [%14, %13] [1, 1], reuse : [seq = false, spat = true, temp = true] : memref<512x4096xf16> to memref<?x?xf16, strided<[4096, 1], offset: ?>>
                %34 = loom.copy_to_tensor %33, %17 on @L1, interconnect : [@horizontal_links, @vertical_links], broadcast : [8, 8] : memref<?x?xf16, strided<[4096, 1], offset: ?>>, memref<?x?xf16> -> tensor<?x?xf16>
                %35 = linalg.matmul ins(%31, %34 : tensor<?x?xf16>, tensor<?x?xf16>) outs(%arg8 : tensor<?x?xf16>) -> tensor<?x?xf16>
                loom.semaphore_give %17 : memref<?x?xf16>
                loom.semaphore_give %19 : memref<?x?xf16>
                affine.yield %35 : tensor<?x?xf16>
              } {loom.iter_type = #loom.iter_type<sequential>}
              %25 = arith.muli %15, %12 : index
              %26 = arith.muli %arg5, %13 : index
              %27 = loom.subview %arg2[%25, %26] [%12, %13] [1, 1], reuse : [seq = false, spat = false, temp = false] : memref<4096x4096xf16> to memref<?x?xf16, strided<[4096, 1], offset: ?>>
              loom.copy_from_tensor %24, %27 on @DRAM : tensor<?x?xf16>, memref<?x?xf16, strided<[4096, 1], offset: ?>>
              loom.semaphore_give %21 : memref<?x?xf16>
            } {loom.iter_type = #loom.iter_type<temporal>}
          } {loom.iter_type = #loom.iter_type<temporal>}
        } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @x}
      } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @y}
      return
    }
  }
  module {
    func.func @matmul__d1i0_d0i0__f10__d_h(%arg0: memref<4096x512xf16>, %arg1: memref<512x4096xf16>, %arg2: memref<4096x4096xf16>) {
      %cst = arith.constant 0.000000e+00 : f16
      %12 = loom.get_symbolic_block_size @constraints::@BM : index
      %13 = loom.get_symbolic_block_size @constraints::@BN : index
      %14 = loom.get_symbolic_block_size @constraints::@BK : index
      affine.parallel (%arg3) = (0) to (8) {
        affine.parallel (%arg4) = (0) to (8) {
          affine.for %arg5 = 0 to affine_map<()[s0, s1] -> (4096 ceildiv s1)>()[%12, %13] {
            affine.for %arg6 = 0 to affine_map<()[s0, s1] -> (4096 ceildiv (s0 * 64))>()[%12, %13] {
              %15 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 8 + d1 + d2 * 64)>(%arg3, %arg4, %arg6)
              %16 = loom.alloc [%14, %13] on @L1 : memref<?x?xf16>
              %17 = loom.semaphore_take %16 : memref<?x?xf16> -> memref<?x?xf16>
              %18 = loom.alloc [%12, %14] on @L1 : memref<?x?xf16>
              %19 = loom.semaphore_take %18 : memref<?x?xf16> -> memref<?x?xf16>
              %20 = loom.alloc [%12, %13] on @L1 : memref<?x?xf16>
              %21 = loom.semaphore_take %20 : memref<?x?xf16> -> memref<?x?xf16>
              %22 = loom.init_tensor %21[%12, %13] : memref<?x?xf16> -> tensor<?x?xf16>
              %23 = linalg.fill ins(%cst : f16) outs(%22 : tensor<?x?xf16>) -> tensor<?x?xf16>
              %24 = affine.for %arg7 = 0 to affine_map<()[s0] -> (512 ceildiv s0)>()[%14] iter_args(%arg8 = %23) -> (tensor<?x?xf16>) {
                %28 = arith.muli %15, %12 : index
                %29 = arith.muli %arg7, %14 : index
                %30 = loom.subview %arg0[%28, %29] [%12, %14] [1, 1], reuse : [seq = false, spat = false, temp = true] : memref<4096x512xf16> to memref<?x?xf16, strided<[512, 1], offset: ?>>
                %31 = loom.copy_to_tensor %30, %19 on @L1, interconnect : [], broadcast : [1, 1] : memref<?x?xf16, strided<[512, 1], offset: ?>>, memref<?x?xf16> -> tensor<?x?xf16>
                %32 = arith.muli %arg5, %13 : index
                %33 = loom.subview %arg1[%29, %32] [%14, %13] [1, 1], reuse : [seq = false, spat = true, temp = true] : memref<512x4096xf16> to memref<?x?xf16, strided<[4096, 1], offset: ?>>
                %34 = loom.copy_to_tensor %33, %17 on @L1, interconnect : [@horizontal_links], broadcast : [1, 8] : memref<?x?xf16, strided<[4096, 1], offset: ?>>, memref<?x?xf16> -> tensor<?x?xf16>
                %35 = linalg.matmul ins(%31, %34 : tensor<?x?xf16>, tensor<?x?xf16>) outs(%arg8 : tensor<?x?xf16>) -> tensor<?x?xf16>
                loom.semaphore_give %17 : memref<?x?xf16>
                loom.semaphore_give %19 : memref<?x?xf16>
                affine.yield %35 : tensor<?x?xf16>
              } {loom.iter_type = #loom.iter_type<sequential>}
              %25 = arith.muli %15, %12 : index
              %26 = arith.muli %arg5, %13 : index
              %27 = loom.subview %arg2[%25, %26] [%12, %13] [1, 1], reuse : [seq = false, spat = false, temp = false] : memref<4096x4096xf16> to memref<?x?xf16, strided<[4096, 1], offset: ?>>
              loom.copy_from_tensor %24, %27 on @DRAM : tensor<?x?xf16>, memref<?x?xf16, strided<[4096, 1], offset: ?>>
              loom.semaphore_give %21 : memref<?x?xf16>
            } {loom.iter_type = #loom.iter_type<temporal>}
          } {loom.iter_type = #loom.iter_type<temporal>}
        } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @x}
      } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @y}
      return
    }
  }
  module {
    func.func @matmul__d1i0_d0i0__f10__d_v(%arg0: memref<4096x512xf16>, %arg1: memref<512x4096xf16>, %arg2: memref<4096x4096xf16>) {
      %cst = arith.constant 0.000000e+00 : f16
      %12 = loom.get_symbolic_block_size @constraints::@BM : index
      %13 = loom.get_symbolic_block_size @constraints::@BN : index
      %14 = loom.get_symbolic_block_size @constraints::@BK : index
      affine.parallel (%arg3) = (0) to (8) {
        affine.parallel (%arg4) = (0) to (8) {
          affine.for %arg5 = 0 to affine_map<()[s0, s1] -> (4096 ceildiv s1)>()[%12, %13] {
            affine.for %arg6 = 0 to affine_map<()[s0, s1] -> (4096 ceildiv (s0 * 64))>()[%12, %13] {
              %15 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 8 + d1 + d2 * 64)>(%arg3, %arg4, %arg6)
              %16 = loom.alloc [%14, %13] on @L1 : memref<?x?xf16>
              %17 = loom.semaphore_take %16 : memref<?x?xf16> -> memref<?x?xf16>
              %18 = loom.alloc [%12, %14] on @L1 : memref<?x?xf16>
              %19 = loom.semaphore_take %18 : memref<?x?xf16> -> memref<?x?xf16>
              %20 = loom.alloc [%12, %13] on @L1 : memref<?x?xf16>
              %21 = loom.semaphore_take %20 : memref<?x?xf16> -> memref<?x?xf16>
              %22 = loom.init_tensor %21[%12, %13] : memref<?x?xf16> -> tensor<?x?xf16>
              %23 = linalg.fill ins(%cst : f16) outs(%22 : tensor<?x?xf16>) -> tensor<?x?xf16>
              %24 = affine.for %arg7 = 0 to affine_map<()[s0] -> (512 ceildiv s0)>()[%14] iter_args(%arg8 = %23) -> (tensor<?x?xf16>) {
                %28 = arith.muli %15, %12 : index
                %29 = arith.muli %arg7, %14 : index
                %30 = loom.subview %arg0[%28, %29] [%12, %14] [1, 1], reuse : [seq = false, spat = false, temp = true] : memref<4096x512xf16> to memref<?x?xf16, strided<[512, 1], offset: ?>>
                %31 = loom.copy_to_tensor %30, %19 on @L1, interconnect : [], broadcast : [1, 1] : memref<?x?xf16, strided<[512, 1], offset: ?>>, memref<?x?xf16> -> tensor<?x?xf16>
                %32 = arith.muli %arg5, %13 : index
                %33 = loom.subview %arg1[%29, %32] [%14, %13] [1, 1], reuse : [seq = false, spat = true, temp = true] : memref<512x4096xf16> to memref<?x?xf16, strided<[4096, 1], offset: ?>>
                %34 = loom.copy_to_tensor %33, %17 on @L1, interconnect : [@vertical_links], broadcast : [8, 1] : memref<?x?xf16, strided<[4096, 1], offset: ?>>, memref<?x?xf16> -> tensor<?x?xf16>
                %35 = linalg.matmul ins(%31, %34 : tensor<?x?xf16>, tensor<?x?xf16>) outs(%arg8 : tensor<?x?xf16>) -> tensor<?x?xf16>
                loom.semaphore_give %17 : memref<?x?xf16>
                loom.semaphore_give %19 : memref<?x?xf16>
                affine.yield %35 : tensor<?x?xf16>
              } {loom.iter_type = #loom.iter_type<sequential>}
              %25 = arith.muli %15, %12 : index
              %26 = arith.muli %arg5, %13 : index
              %27 = loom.subview %arg2[%25, %26] [%12, %13] [1, 1], reuse : [seq = false, spat = false, temp = false] : memref<4096x4096xf16> to memref<?x?xf16, strided<[4096, 1], offset: ?>>
              loom.copy_from_tensor %24, %27 on @DRAM : tensor<?x?xf16>, memref<?x?xf16, strided<[4096, 1], offset: ?>>
              loom.semaphore_give %21 : memref<?x?xf16>
            } {loom.iter_type = #loom.iter_type<temporal>}
          } {loom.iter_type = #loom.iter_type<temporal>}
        } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @x}
      } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @y}
      return
    }
  }
  module {
    func.func @matmul__d0i0_d1i1__f01__d_d(%arg0: memref<4096x512xf16>, %arg1: memref<512x4096xf16>, %arg2: memref<4096x4096xf16>) {
      %cst = arith.constant 0.000000e+00 : f16
      %12 = loom.get_symbolic_block_size @constraints::@BM : index
      %13 = loom.get_symbolic_block_size @constraints::@BN : index
      %14 = loom.get_symbolic_block_size @constraints::@BK : index
      affine.parallel (%arg3) = (0) to (8) {
        affine.parallel (%arg4) = (0) to (8) {
          affine.for %arg5 = 0 to affine_map<()[s0, s1] -> (4096 ceildiv (s0 * 8))>()[%12, %13] {
            affine.for %arg6 = 0 to affine_map<()[s0, s1] -> (4096 ceildiv (s1 * 8))>()[%12, %13] {
              %15 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 8)>(%arg3, %arg5)
              %16 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 8)>(%arg4, %arg6)
              %17 = loom.alloc [%14, %13] on @L1 : memref<?x?xf16>
              %18 = loom.semaphore_take %17 : memref<?x?xf16> -> memref<?x?xf16>
              %19 = loom.alloc [%12, %14] on @L1 : memref<?x?xf16>
              %20 = loom.semaphore_take %19 : memref<?x?xf16> -> memref<?x?xf16>
              %21 = loom.alloc [%12, %13] on @L1 : memref<?x?xf16>
              %22 = loom.semaphore_take %21 : memref<?x?xf16> -> memref<?x?xf16>
              %23 = loom.init_tensor %22[%12, %13] : memref<?x?xf16> -> tensor<?x?xf16>
              %24 = linalg.fill ins(%cst : f16) outs(%23 : tensor<?x?xf16>) -> tensor<?x?xf16>
              %25 = affine.for %arg7 = 0 to affine_map<()[s0] -> (512 ceildiv s0)>()[%14] iter_args(%arg8 = %24) -> (tensor<?x?xf16>) {
                %29 = arith.muli %15, %12 : index
                %30 = arith.muli %arg7, %14 : index
                %31 = loom.subview %arg0[%29, %30] [%12, %14] [1, 1], reuse : [seq = false, spat = true, temp = true] : memref<4096x512xf16> to memref<?x?xf16, strided<[512, 1], offset: ?>>
                %32 = loom.copy_to_tensor %31, %20 on @L1, interconnect : [], broadcast : [1, 1] : memref<?x?xf16, strided<[512, 1], offset: ?>>, memref<?x?xf16> -> tensor<?x?xf16>
                %33 = arith.muli %16, %13 : index
                %34 = loom.subview %arg1[%30, %33] [%14, %13] [1, 1], reuse : [seq = false, spat = true, temp = true] : memref<512x4096xf16> to memref<?x?xf16, strided<[4096, 1], offset: ?>>
                %35 = loom.copy_to_tensor %34, %18 on @L1, interconnect : [], broadcast : [1, 1] : memref<?x?xf16, strided<[4096, 1], offset: ?>>, memref<?x?xf16> -> tensor<?x?xf16>
                %36 = linalg.matmul ins(%32, %35 : tensor<?x?xf16>, tensor<?x?xf16>) outs(%arg8 : tensor<?x?xf16>) -> tensor<?x?xf16>
                loom.semaphore_give %18 : memref<?x?xf16>
                loom.semaphore_give %20 : memref<?x?xf16>
                affine.yield %36 : tensor<?x?xf16>
              } {loom.iter_type = #loom.iter_type<sequential>}
              %26 = arith.muli %15, %12 : index
              %27 = arith.muli %16, %13 : index
              %28 = loom.subview %arg2[%26, %27] [%12, %13] [1, 1], reuse : [seq = false, spat = false, temp = false] : memref<4096x4096xf16> to memref<?x?xf16, strided<[4096, 1], offset: ?>>
              loom.copy_from_tensor %25, %28 on @DRAM : tensor<?x?xf16>, memref<?x?xf16, strided<[4096, 1], offset: ?>>
              loom.semaphore_give %22 : memref<?x?xf16>
            } {loom.iter_type = #loom.iter_type<temporal>}
          } {loom.iter_type = #loom.iter_type<temporal>}
        } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @y}
      } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @x}
      return
    }
  }
  module {
    func.func @matmul__d0i0_d1i1__f01__d_h(%arg0: memref<4096x512xf16>, %arg1: memref<512x4096xf16>, %arg2: memref<4096x4096xf16>) {
      %cst = arith.constant 0.000000e+00 : f16
      %12 = loom.get_symbolic_block_size @constraints::@BM : index
      %13 = loom.get_symbolic_block_size @constraints::@BN : index
      %14 = loom.get_symbolic_block_size @constraints::@BK : index
      affine.parallel (%arg3) = (0) to (8) {
        affine.parallel (%arg4) = (0) to (8) {
          affine.for %arg5 = 0 to affine_map<()[s0, s1] -> (4096 ceildiv (s0 * 8))>()[%12, %13] {
            affine.for %arg6 = 0 to affine_map<()[s0, s1] -> (4096 ceildiv (s1 * 8))>()[%12, %13] {
              %15 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 8)>(%arg3, %arg5)
              %16 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 8)>(%arg4, %arg6)
              %17 = loom.alloc [%14, %13] on @L1 : memref<?x?xf16>
              %18 = loom.semaphore_take %17 : memref<?x?xf16> -> memref<?x?xf16>
              %19 = loom.alloc [%12, %14] on @L1 : memref<?x?xf16>
              %20 = loom.semaphore_take %19 : memref<?x?xf16> -> memref<?x?xf16>
              %21 = loom.alloc [%12, %13] on @L1 : memref<?x?xf16>
              %22 = loom.semaphore_take %21 : memref<?x?xf16> -> memref<?x?xf16>
              %23 = loom.init_tensor %22[%12, %13] : memref<?x?xf16> -> tensor<?x?xf16>
              %24 = linalg.fill ins(%cst : f16) outs(%23 : tensor<?x?xf16>) -> tensor<?x?xf16>
              %25 = affine.for %arg7 = 0 to affine_map<()[s0] -> (512 ceildiv s0)>()[%14] iter_args(%arg8 = %24) -> (tensor<?x?xf16>) {
                %29 = arith.muli %15, %12 : index
                %30 = arith.muli %arg7, %14 : index
                %31 = loom.subview %arg0[%29, %30] [%12, %14] [1, 1], reuse : [seq = false, spat = true, temp = true] : memref<4096x512xf16> to memref<?x?xf16, strided<[512, 1], offset: ?>>
                %32 = loom.copy_to_tensor %31, %20 on @L1, interconnect : [], broadcast : [1, 1] : memref<?x?xf16, strided<[512, 1], offset: ?>>, memref<?x?xf16> -> tensor<?x?xf16>
                %33 = arith.muli %16, %13 : index
                %34 = loom.subview %arg1[%30, %33] [%14, %13] [1, 1], reuse : [seq = false, spat = true, temp = true] : memref<512x4096xf16> to memref<?x?xf16, strided<[4096, 1], offset: ?>>
                %35 = loom.copy_to_tensor %34, %18 on @L1, interconnect : [@horizontal_links], broadcast : [1, 8] : memref<?x?xf16, strided<[4096, 1], offset: ?>>, memref<?x?xf16> -> tensor<?x?xf16>
                %36 = linalg.matmul ins(%32, %35 : tensor<?x?xf16>, tensor<?x?xf16>) outs(%arg8 : tensor<?x?xf16>) -> tensor<?x?xf16>
                loom.semaphore_give %18 : memref<?x?xf16>
                loom.semaphore_give %20 : memref<?x?xf16>
                affine.yield %36 : tensor<?x?xf16>
              } {loom.iter_type = #loom.iter_type<sequential>}
              %26 = arith.muli %15, %12 : index
              %27 = arith.muli %16, %13 : index
              %28 = loom.subview %arg2[%26, %27] [%12, %13] [1, 1], reuse : [seq = false, spat = false, temp = false] : memref<4096x4096xf16> to memref<?x?xf16, strided<[4096, 1], offset: ?>>
              loom.copy_from_tensor %25, %28 on @DRAM : tensor<?x?xf16>, memref<?x?xf16, strided<[4096, 1], offset: ?>>
              loom.semaphore_give %22 : memref<?x?xf16>
            } {loom.iter_type = #loom.iter_type<temporal>}
          } {loom.iter_type = #loom.iter_type<temporal>}
        } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @y}
      } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @x}
      return
    }
  }
  module {
    func.func @matmul__d0i0_d1i1__f01__v_d(%arg0: memref<4096x512xf16>, %arg1: memref<512x4096xf16>, %arg2: memref<4096x4096xf16>) {
      %cst = arith.constant 0.000000e+00 : f16
      %12 = loom.get_symbolic_block_size @constraints::@BM : index
      %13 = loom.get_symbolic_block_size @constraints::@BN : index
      %14 = loom.get_symbolic_block_size @constraints::@BK : index
      affine.parallel (%arg3) = (0) to (8) {
        affine.parallel (%arg4) = (0) to (8) {
          affine.for %arg5 = 0 to affine_map<()[s0, s1] -> (4096 ceildiv (s0 * 8))>()[%12, %13] {
            affine.for %arg6 = 0 to affine_map<()[s0, s1] -> (4096 ceildiv (s1 * 8))>()[%12, %13] {
              %15 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 8)>(%arg3, %arg5)
              %16 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 8)>(%arg4, %arg6)
              %17 = loom.alloc [%14, %13] on @L1 : memref<?x?xf16>
              %18 = loom.semaphore_take %17 : memref<?x?xf16> -> memref<?x?xf16>
              %19 = loom.alloc [%12, %14] on @L1 : memref<?x?xf16>
              %20 = loom.semaphore_take %19 : memref<?x?xf16> -> memref<?x?xf16>
              %21 = loom.alloc [%12, %13] on @L1 : memref<?x?xf16>
              %22 = loom.semaphore_take %21 : memref<?x?xf16> -> memref<?x?xf16>
              %23 = loom.init_tensor %22[%12, %13] : memref<?x?xf16> -> tensor<?x?xf16>
              %24 = linalg.fill ins(%cst : f16) outs(%23 : tensor<?x?xf16>) -> tensor<?x?xf16>
              %25 = affine.for %arg7 = 0 to affine_map<()[s0] -> (512 ceildiv s0)>()[%14] iter_args(%arg8 = %24) -> (tensor<?x?xf16>) {
                %29 = arith.muli %15, %12 : index
                %30 = arith.muli %arg7, %14 : index
                %31 = loom.subview %arg0[%29, %30] [%12, %14] [1, 1], reuse : [seq = false, spat = true, temp = true] : memref<4096x512xf16> to memref<?x?xf16, strided<[512, 1], offset: ?>>
                %32 = loom.copy_to_tensor %31, %20 on @L1, interconnect : [@vertical_links], broadcast : [8, 1] : memref<?x?xf16, strided<[512, 1], offset: ?>>, memref<?x?xf16> -> tensor<?x?xf16>
                %33 = arith.muli %16, %13 : index
                %34 = loom.subview %arg1[%30, %33] [%14, %13] [1, 1], reuse : [seq = false, spat = true, temp = true] : memref<512x4096xf16> to memref<?x?xf16, strided<[4096, 1], offset: ?>>
                %35 = loom.copy_to_tensor %34, %18 on @L1, interconnect : [], broadcast : [1, 1] : memref<?x?xf16, strided<[4096, 1], offset: ?>>, memref<?x?xf16> -> tensor<?x?xf16>
                %36 = linalg.matmul ins(%32, %35 : tensor<?x?xf16>, tensor<?x?xf16>) outs(%arg8 : tensor<?x?xf16>) -> tensor<?x?xf16>
                loom.semaphore_give %18 : memref<?x?xf16>
                loom.semaphore_give %20 : memref<?x?xf16>
                affine.yield %36 : tensor<?x?xf16>
              } {loom.iter_type = #loom.iter_type<sequential>}
              %26 = arith.muli %15, %12 : index
              %27 = arith.muli %16, %13 : index
              %28 = loom.subview %arg2[%26, %27] [%12, %13] [1, 1], reuse : [seq = false, spat = false, temp = false] : memref<4096x4096xf16> to memref<?x?xf16, strided<[4096, 1], offset: ?>>
              loom.copy_from_tensor %25, %28 on @DRAM : tensor<?x?xf16>, memref<?x?xf16, strided<[4096, 1], offset: ?>>
              loom.semaphore_give %22 : memref<?x?xf16>
            } {loom.iter_type = #loom.iter_type<temporal>}
          } {loom.iter_type = #loom.iter_type<temporal>}
        } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @y}
      } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @x}
      return
    }
  }
  module {
    func.func @matmul__d0i0_d1i1__f01__v_h(%arg0: memref<4096x512xf16>, %arg1: memref<512x4096xf16>, %arg2: memref<4096x4096xf16>) {
      %cst = arith.constant 0.000000e+00 : f16
      %12 = loom.get_symbolic_block_size @constraints::@BM : index
      %13 = loom.get_symbolic_block_size @constraints::@BN : index
      %14 = loom.get_symbolic_block_size @constraints::@BK : index
      affine.parallel (%arg3) = (0) to (8) {
        affine.parallel (%arg4) = (0) to (8) {
          affine.for %arg5 = 0 to affine_map<()[s0, s1] -> (4096 ceildiv (s0 * 8))>()[%12, %13] {
            affine.for %arg6 = 0 to affine_map<()[s0, s1] -> (4096 ceildiv (s1 * 8))>()[%12, %13] {
              %15 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 8)>(%arg3, %arg5)
              %16 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 8)>(%arg4, %arg6)
              %17 = loom.alloc [%14, %13] on @L1 : memref<?x?xf16>
              %18 = loom.semaphore_take %17 : memref<?x?xf16> -> memref<?x?xf16>
              %19 = loom.alloc [%12, %14] on @L1 : memref<?x?xf16>
              %20 = loom.semaphore_take %19 : memref<?x?xf16> -> memref<?x?xf16>
              %21 = loom.alloc [%12, %13] on @L1 : memref<?x?xf16>
              %22 = loom.semaphore_take %21 : memref<?x?xf16> -> memref<?x?xf16>
              %23 = loom.init_tensor %22[%12, %13] : memref<?x?xf16> -> tensor<?x?xf16>
              %24 = linalg.fill ins(%cst : f16) outs(%23 : tensor<?x?xf16>) -> tensor<?x?xf16>
              %25 = affine.for %arg7 = 0 to affine_map<()[s0] -> (512 ceildiv s0)>()[%14] iter_args(%arg8 = %24) -> (tensor<?x?xf16>) {
                %29 = arith.muli %15, %12 : index
                %30 = arith.muli %arg7, %14 : index
                %31 = loom.subview %arg0[%29, %30] [%12, %14] [1, 1], reuse : [seq = false, spat = true, temp = true] : memref<4096x512xf16> to memref<?x?xf16, strided<[512, 1], offset: ?>>
                %32 = loom.copy_to_tensor %31, %20 on @L1, interconnect : [@vertical_links], broadcast : [8, 1] : memref<?x?xf16, strided<[512, 1], offset: ?>>, memref<?x?xf16> -> tensor<?x?xf16>
                %33 = arith.muli %16, %13 : index
                %34 = loom.subview %arg1[%30, %33] [%14, %13] [1, 1], reuse : [seq = false, spat = true, temp = true] : memref<512x4096xf16> to memref<?x?xf16, strided<[4096, 1], offset: ?>>
                %35 = loom.copy_to_tensor %34, %18 on @L1, interconnect : [@horizontal_links], broadcast : [1, 8] : memref<?x?xf16, strided<[4096, 1], offset: ?>>, memref<?x?xf16> -> tensor<?x?xf16>
                %36 = linalg.matmul ins(%32, %35 : tensor<?x?xf16>, tensor<?x?xf16>) outs(%arg8 : tensor<?x?xf16>) -> tensor<?x?xf16>
                loom.semaphore_give %18 : memref<?x?xf16>
                loom.semaphore_give %20 : memref<?x?xf16>
                affine.yield %36 : tensor<?x?xf16>
              } {loom.iter_type = #loom.iter_type<sequential>}
              %26 = arith.muli %15, %12 : index
              %27 = arith.muli %16, %13 : index
              %28 = loom.subview %arg2[%26, %27] [%12, %13] [1, 1], reuse : [seq = false, spat = false, temp = false] : memref<4096x4096xf16> to memref<?x?xf16, strided<[4096, 1], offset: ?>>
              loom.copy_from_tensor %25, %28 on @DRAM : tensor<?x?xf16>, memref<?x?xf16, strided<[4096, 1], offset: ?>>
              loom.semaphore_give %22 : memref<?x?xf16>
            } {loom.iter_type = #loom.iter_type<temporal>}
          } {loom.iter_type = #loom.iter_type<temporal>}
        } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @y}
      } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @x}
      return
    }
  }
  module {
    func.func @matmul__d0i0_d1i1__f10__d_d(%arg0: memref<4096x512xf16>, %arg1: memref<512x4096xf16>, %arg2: memref<4096x4096xf16>) {
      %cst = arith.constant 0.000000e+00 : f16
      %12 = loom.get_symbolic_block_size @constraints::@BM : index
      %13 = loom.get_symbolic_block_size @constraints::@BN : index
      %14 = loom.get_symbolic_block_size @constraints::@BK : index
      affine.parallel (%arg3) = (0) to (8) {
        affine.parallel (%arg4) = (0) to (8) {
          affine.for %arg5 = 0 to affine_map<()[s0, s1] -> (4096 ceildiv (s1 * 8))>()[%12, %13] {
            affine.for %arg6 = 0 to affine_map<()[s0, s1] -> (4096 ceildiv (s0 * 8))>()[%12, %13] {
              %15 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 8)>(%arg3, %arg6)
              %16 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 8)>(%arg4, %arg5)
              %17 = loom.alloc [%14, %13] on @L1 : memref<?x?xf16>
              %18 = loom.semaphore_take %17 : memref<?x?xf16> -> memref<?x?xf16>
              %19 = loom.alloc [%12, %14] on @L1 : memref<?x?xf16>
              %20 = loom.semaphore_take %19 : memref<?x?xf16> -> memref<?x?xf16>
              %21 = loom.alloc [%12, %13] on @L1 : memref<?x?xf16>
              %22 = loom.semaphore_take %21 : memref<?x?xf16> -> memref<?x?xf16>
              %23 = loom.init_tensor %22[%12, %13] : memref<?x?xf16> -> tensor<?x?xf16>
              %24 = linalg.fill ins(%cst : f16) outs(%23 : tensor<?x?xf16>) -> tensor<?x?xf16>
              %25 = affine.for %arg7 = 0 to affine_map<()[s0] -> (512 ceildiv s0)>()[%14] iter_args(%arg8 = %24) -> (tensor<?x?xf16>) {
                %29 = arith.muli %15, %12 : index
                %30 = arith.muli %arg7, %14 : index
                %31 = loom.subview %arg0[%29, %30] [%12, %14] [1, 1], reuse : [seq = false, spat = true, temp = true] : memref<4096x512xf16> to memref<?x?xf16, strided<[512, 1], offset: ?>>
                %32 = loom.copy_to_tensor %31, %20 on @L1, interconnect : [], broadcast : [1, 1] : memref<?x?xf16, strided<[512, 1], offset: ?>>, memref<?x?xf16> -> tensor<?x?xf16>
                %33 = arith.muli %16, %13 : index
                %34 = loom.subview %arg1[%30, %33] [%14, %13] [1, 1], reuse : [seq = false, spat = true, temp = true] : memref<512x4096xf16> to memref<?x?xf16, strided<[4096, 1], offset: ?>>
                %35 = loom.copy_to_tensor %34, %18 on @L1, interconnect : [], broadcast : [1, 1] : memref<?x?xf16, strided<[4096, 1], offset: ?>>, memref<?x?xf16> -> tensor<?x?xf16>
                %36 = linalg.matmul ins(%32, %35 : tensor<?x?xf16>, tensor<?x?xf16>) outs(%arg8 : tensor<?x?xf16>) -> tensor<?x?xf16>
                loom.semaphore_give %18 : memref<?x?xf16>
                loom.semaphore_give %20 : memref<?x?xf16>
                affine.yield %36 : tensor<?x?xf16>
              } {loom.iter_type = #loom.iter_type<sequential>}
              %26 = arith.muli %15, %12 : index
              %27 = arith.muli %16, %13 : index
              %28 = loom.subview %arg2[%26, %27] [%12, %13] [1, 1], reuse : [seq = false, spat = false, temp = false] : memref<4096x4096xf16> to memref<?x?xf16, strided<[4096, 1], offset: ?>>
              loom.copy_from_tensor %25, %28 on @DRAM : tensor<?x?xf16>, memref<?x?xf16, strided<[4096, 1], offset: ?>>
              loom.semaphore_give %22 : memref<?x?xf16>
            } {loom.iter_type = #loom.iter_type<temporal>}
          } {loom.iter_type = #loom.iter_type<temporal>}
        } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @y}
      } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @x}
      return
    }
  }
  module {
    func.func @matmul__d0i0_d1i1__f10__d_h(%arg0: memref<4096x512xf16>, %arg1: memref<512x4096xf16>, %arg2: memref<4096x4096xf16>) {
      %cst = arith.constant 0.000000e+00 : f16
      %12 = loom.get_symbolic_block_size @constraints::@BM : index
      %13 = loom.get_symbolic_block_size @constraints::@BN : index
      %14 = loom.get_symbolic_block_size @constraints::@BK : index
      affine.parallel (%arg3) = (0) to (8) {
        affine.parallel (%arg4) = (0) to (8) {
          affine.for %arg5 = 0 to affine_map<()[s0, s1] -> (4096 ceildiv (s1 * 8))>()[%12, %13] {
            affine.for %arg6 = 0 to affine_map<()[s0, s1] -> (4096 ceildiv (s0 * 8))>()[%12, %13] {
              %15 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 8)>(%arg3, %arg6)
              %16 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 8)>(%arg4, %arg5)
              %17 = loom.alloc [%14, %13] on @L1 : memref<?x?xf16>
              %18 = loom.semaphore_take %17 : memref<?x?xf16> -> memref<?x?xf16>
              %19 = loom.alloc [%12, %14] on @L1 : memref<?x?xf16>
              %20 = loom.semaphore_take %19 : memref<?x?xf16> -> memref<?x?xf16>
              %21 = loom.alloc [%12, %13] on @L1 : memref<?x?xf16>
              %22 = loom.semaphore_take %21 : memref<?x?xf16> -> memref<?x?xf16>
              %23 = loom.init_tensor %22[%12, %13] : memref<?x?xf16> -> tensor<?x?xf16>
              %24 = linalg.fill ins(%cst : f16) outs(%23 : tensor<?x?xf16>) -> tensor<?x?xf16>
              %25 = affine.for %arg7 = 0 to affine_map<()[s0] -> (512 ceildiv s0)>()[%14] iter_args(%arg8 = %24) -> (tensor<?x?xf16>) {
                %29 = arith.muli %15, %12 : index
                %30 = arith.muli %arg7, %14 : index
                %31 = loom.subview %arg0[%29, %30] [%12, %14] [1, 1], reuse : [seq = false, spat = true, temp = true] : memref<4096x512xf16> to memref<?x?xf16, strided<[512, 1], offset: ?>>
                %32 = loom.copy_to_tensor %31, %20 on @L1, interconnect : [], broadcast : [1, 1] : memref<?x?xf16, strided<[512, 1], offset: ?>>, memref<?x?xf16> -> tensor<?x?xf16>
                %33 = arith.muli %16, %13 : index
                %34 = loom.subview %arg1[%30, %33] [%14, %13] [1, 1], reuse : [seq = false, spat = true, temp = true] : memref<512x4096xf16> to memref<?x?xf16, strided<[4096, 1], offset: ?>>
                %35 = loom.copy_to_tensor %34, %18 on @L1, interconnect : [@horizontal_links], broadcast : [1, 8] : memref<?x?xf16, strided<[4096, 1], offset: ?>>, memref<?x?xf16> -> tensor<?x?xf16>
                %36 = linalg.matmul ins(%32, %35 : tensor<?x?xf16>, tensor<?x?xf16>) outs(%arg8 : tensor<?x?xf16>) -> tensor<?x?xf16>
                loom.semaphore_give %18 : memref<?x?xf16>
                loom.semaphore_give %20 : memref<?x?xf16>
                affine.yield %36 : tensor<?x?xf16>
              } {loom.iter_type = #loom.iter_type<sequential>}
              %26 = arith.muli %15, %12 : index
              %27 = arith.muli %16, %13 : index
              %28 = loom.subview %arg2[%26, %27] [%12, %13] [1, 1], reuse : [seq = false, spat = false, temp = false] : memref<4096x4096xf16> to memref<?x?xf16, strided<[4096, 1], offset: ?>>
              loom.copy_from_tensor %25, %28 on @DRAM : tensor<?x?xf16>, memref<?x?xf16, strided<[4096, 1], offset: ?>>
              loom.semaphore_give %22 : memref<?x?xf16>
            } {loom.iter_type = #loom.iter_type<temporal>}
          } {loom.iter_type = #loom.iter_type<temporal>}
        } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @y}
      } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @x}
      return
    }
  }
  module {
    func.func @matmul__d0i0_d1i1__f10__v_d(%arg0: memref<4096x512xf16>, %arg1: memref<512x4096xf16>, %arg2: memref<4096x4096xf16>) {
      %cst = arith.constant 0.000000e+00 : f16
      %12 = loom.get_symbolic_block_size @constraints::@BM : index
      %13 = loom.get_symbolic_block_size @constraints::@BN : index
      %14 = loom.get_symbolic_block_size @constraints::@BK : index
      affine.parallel (%arg3) = (0) to (8) {
        affine.parallel (%arg4) = (0) to (8) {
          affine.for %arg5 = 0 to affine_map<()[s0, s1] -> (4096 ceildiv (s1 * 8))>()[%12, %13] {
            affine.for %arg6 = 0 to affine_map<()[s0, s1] -> (4096 ceildiv (s0 * 8))>()[%12, %13] {
              %15 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 8)>(%arg3, %arg6)
              %16 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 8)>(%arg4, %arg5)
              %17 = loom.alloc [%14, %13] on @L1 : memref<?x?xf16>
              %18 = loom.semaphore_take %17 : memref<?x?xf16> -> memref<?x?xf16>
              %19 = loom.alloc [%12, %14] on @L1 : memref<?x?xf16>
              %20 = loom.semaphore_take %19 : memref<?x?xf16> -> memref<?x?xf16>
              %21 = loom.alloc [%12, %13] on @L1 : memref<?x?xf16>
              %22 = loom.semaphore_take %21 : memref<?x?xf16> -> memref<?x?xf16>
              %23 = loom.init_tensor %22[%12, %13] : memref<?x?xf16> -> tensor<?x?xf16>
              %24 = linalg.fill ins(%cst : f16) outs(%23 : tensor<?x?xf16>) -> tensor<?x?xf16>
              %25 = affine.for %arg7 = 0 to affine_map<()[s0] -> (512 ceildiv s0)>()[%14] iter_args(%arg8 = %24) -> (tensor<?x?xf16>) {
                %29 = arith.muli %15, %12 : index
                %30 = arith.muli %arg7, %14 : index
                %31 = loom.subview %arg0[%29, %30] [%12, %14] [1, 1], reuse : [seq = false, spat = true, temp = true] : memref<4096x512xf16> to memref<?x?xf16, strided<[512, 1], offset: ?>>
                %32 = loom.copy_to_tensor %31, %20 on @L1, interconnect : [@vertical_links], broadcast : [8, 1] : memref<?x?xf16, strided<[512, 1], offset: ?>>, memref<?x?xf16> -> tensor<?x?xf16>
                %33 = arith.muli %16, %13 : index
                %34 = loom.subview %arg1[%30, %33] [%14, %13] [1, 1], reuse : [seq = false, spat = true, temp = true] : memref<512x4096xf16> to memref<?x?xf16, strided<[4096, 1], offset: ?>>
                %35 = loom.copy_to_tensor %34, %18 on @L1, interconnect : [], broadcast : [1, 1] : memref<?x?xf16, strided<[4096, 1], offset: ?>>, memref<?x?xf16> -> tensor<?x?xf16>
                %36 = linalg.matmul ins(%32, %35 : tensor<?x?xf16>, tensor<?x?xf16>) outs(%arg8 : tensor<?x?xf16>) -> tensor<?x?xf16>
                loom.semaphore_give %18 : memref<?x?xf16>
                loom.semaphore_give %20 : memref<?x?xf16>
                affine.yield %36 : tensor<?x?xf16>
              } {loom.iter_type = #loom.iter_type<sequential>}
              %26 = arith.muli %15, %12 : index
              %27 = arith.muli %16, %13 : index
              %28 = loom.subview %arg2[%26, %27] [%12, %13] [1, 1], reuse : [seq = false, spat = false, temp = false] : memref<4096x4096xf16> to memref<?x?xf16, strided<[4096, 1], offset: ?>>
              loom.copy_from_tensor %25, %28 on @DRAM : tensor<?x?xf16>, memref<?x?xf16, strided<[4096, 1], offset: ?>>
              loom.semaphore_give %22 : memref<?x?xf16>
            } {loom.iter_type = #loom.iter_type<temporal>}
          } {loom.iter_type = #loom.iter_type<temporal>}
        } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @y}
      } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @x}
      return
    }
  }
  module {
    func.func @matmul__d0i0_d1i1__f10__v_h(%arg0: memref<4096x512xf16>, %arg1: memref<512x4096xf16>, %arg2: memref<4096x4096xf16>) {
      %cst = arith.constant 0.000000e+00 : f16
      %12 = loom.get_symbolic_block_size @constraints::@BM : index
      %13 = loom.get_symbolic_block_size @constraints::@BN : index
      %14 = loom.get_symbolic_block_size @constraints::@BK : index
      affine.parallel (%arg3) = (0) to (8) {
        affine.parallel (%arg4) = (0) to (8) {
          affine.for %arg5 = 0 to affine_map<()[s0, s1] -> (4096 ceildiv (s1 * 8))>()[%12, %13] {
            affine.for %arg6 = 0 to affine_map<()[s0, s1] -> (4096 ceildiv (s0 * 8))>()[%12, %13] {
              %15 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 8)>(%arg3, %arg6)
              %16 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 8)>(%arg4, %arg5)
              %17 = loom.alloc [%14, %13] on @L1 : memref<?x?xf16>
              %18 = loom.semaphore_take %17 : memref<?x?xf16> -> memref<?x?xf16>
              %19 = loom.alloc [%12, %14] on @L1 : memref<?x?xf16>
              %20 = loom.semaphore_take %19 : memref<?x?xf16> -> memref<?x?xf16>
              %21 = loom.alloc [%12, %13] on @L1 : memref<?x?xf16>
              %22 = loom.semaphore_take %21 : memref<?x?xf16> -> memref<?x?xf16>
              %23 = loom.init_tensor %22[%12, %13] : memref<?x?xf16> -> tensor<?x?xf16>
              %24 = linalg.fill ins(%cst : f16) outs(%23 : tensor<?x?xf16>) -> tensor<?x?xf16>
              %25 = affine.for %arg7 = 0 to affine_map<()[s0] -> (512 ceildiv s0)>()[%14] iter_args(%arg8 = %24) -> (tensor<?x?xf16>) {
                %29 = arith.muli %15, %12 : index
                %30 = arith.muli %arg7, %14 : index
                %31 = loom.subview %arg0[%29, %30] [%12, %14] [1, 1], reuse : [seq = false, spat = true, temp = true] : memref<4096x512xf16> to memref<?x?xf16, strided<[512, 1], offset: ?>>
                %32 = loom.copy_to_tensor %31, %20 on @L1, interconnect : [@vertical_links], broadcast : [8, 1] : memref<?x?xf16, strided<[512, 1], offset: ?>>, memref<?x?xf16> -> tensor<?x?xf16>
                %33 = arith.muli %16, %13 : index
                %34 = loom.subview %arg1[%30, %33] [%14, %13] [1, 1], reuse : [seq = false, spat = true, temp = true] : memref<512x4096xf16> to memref<?x?xf16, strided<[4096, 1], offset: ?>>
                %35 = loom.copy_to_tensor %34, %18 on @L1, interconnect : [@horizontal_links], broadcast : [1, 8] : memref<?x?xf16, strided<[4096, 1], offset: ?>>, memref<?x?xf16> -> tensor<?x?xf16>
                %36 = linalg.matmul ins(%32, %35 : tensor<?x?xf16>, tensor<?x?xf16>) outs(%arg8 : tensor<?x?xf16>) -> tensor<?x?xf16>
                loom.semaphore_give %18 : memref<?x?xf16>
                loom.semaphore_give %20 : memref<?x?xf16>
                affine.yield %36 : tensor<?x?xf16>
              } {loom.iter_type = #loom.iter_type<sequential>}
              %26 = arith.muli %15, %12 : index
              %27 = arith.muli %16, %13 : index
              %28 = loom.subview %arg2[%26, %27] [%12, %13] [1, 1], reuse : [seq = false, spat = false, temp = false] : memref<4096x4096xf16> to memref<?x?xf16, strided<[4096, 1], offset: ?>>
              loom.copy_from_tensor %25, %28 on @DRAM : tensor<?x?xf16>, memref<?x?xf16, strided<[4096, 1], offset: ?>>
              loom.semaphore_give %22 : memref<?x?xf16>
            } {loom.iter_type = #loom.iter_type<temporal>}
          } {loom.iter_type = #loom.iter_type<temporal>}
        } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @y}
      } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @x}
      return
    }
  }
  module {
    func.func @matmul__d1i0_d0i1__f01__d_d(%arg0: memref<4096x512xf16>, %arg1: memref<512x4096xf16>, %arg2: memref<4096x4096xf16>) {
      %cst = arith.constant 0.000000e+00 : f16
      %12 = loom.get_symbolic_block_size @constraints::@BM : index
      %13 = loom.get_symbolic_block_size @constraints::@BN : index
      %14 = loom.get_symbolic_block_size @constraints::@BK : index
      affine.parallel (%arg3) = (0) to (8) {
        affine.parallel (%arg4) = (0) to (8) {
          affine.for %arg5 = 0 to affine_map<()[s0, s1] -> (4096 ceildiv (s0 * 8))>()[%12, %13] {
            affine.for %arg6 = 0 to affine_map<()[s0, s1] -> (4096 ceildiv (s1 * 8))>()[%12, %13] {
              %15 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 8)>(%arg3, %arg5)
              %16 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 8)>(%arg4, %arg6)
              %17 = loom.alloc [%14, %13] on @L1 : memref<?x?xf16>
              %18 = loom.semaphore_take %17 : memref<?x?xf16> -> memref<?x?xf16>
              %19 = loom.alloc [%12, %14] on @L1 : memref<?x?xf16>
              %20 = loom.semaphore_take %19 : memref<?x?xf16> -> memref<?x?xf16>
              %21 = loom.alloc [%12, %13] on @L1 : memref<?x?xf16>
              %22 = loom.semaphore_take %21 : memref<?x?xf16> -> memref<?x?xf16>
              %23 = loom.init_tensor %22[%12, %13] : memref<?x?xf16> -> tensor<?x?xf16>
              %24 = linalg.fill ins(%cst : f16) outs(%23 : tensor<?x?xf16>) -> tensor<?x?xf16>
              %25 = affine.for %arg7 = 0 to affine_map<()[s0] -> (512 ceildiv s0)>()[%14] iter_args(%arg8 = %24) -> (tensor<?x?xf16>) {
                %29 = arith.muli %15, %12 : index
                %30 = arith.muli %arg7, %14 : index
                %31 = loom.subview %arg0[%29, %30] [%12, %14] [1, 1], reuse : [seq = false, spat = true, temp = true] : memref<4096x512xf16> to memref<?x?xf16, strided<[512, 1], offset: ?>>
                %32 = loom.copy_to_tensor %31, %20 on @L1, interconnect : [], broadcast : [1, 1] : memref<?x?xf16, strided<[512, 1], offset: ?>>, memref<?x?xf16> -> tensor<?x?xf16>
                %33 = arith.muli %16, %13 : index
                %34 = loom.subview %arg1[%30, %33] [%14, %13] [1, 1], reuse : [seq = false, spat = true, temp = true] : memref<512x4096xf16> to memref<?x?xf16, strided<[4096, 1], offset: ?>>
                %35 = loom.copy_to_tensor %34, %18 on @L1, interconnect : [], broadcast : [1, 1] : memref<?x?xf16, strided<[4096, 1], offset: ?>>, memref<?x?xf16> -> tensor<?x?xf16>
                %36 = linalg.matmul ins(%32, %35 : tensor<?x?xf16>, tensor<?x?xf16>) outs(%arg8 : tensor<?x?xf16>) -> tensor<?x?xf16>
                loom.semaphore_give %18 : memref<?x?xf16>
                loom.semaphore_give %20 : memref<?x?xf16>
                affine.yield %36 : tensor<?x?xf16>
              } {loom.iter_type = #loom.iter_type<sequential>}
              %26 = arith.muli %15, %12 : index
              %27 = arith.muli %16, %13 : index
              %28 = loom.subview %arg2[%26, %27] [%12, %13] [1, 1], reuse : [seq = false, spat = false, temp = false] : memref<4096x4096xf16> to memref<?x?xf16, strided<[4096, 1], offset: ?>>
              loom.copy_from_tensor %25, %28 on @DRAM : tensor<?x?xf16>, memref<?x?xf16, strided<[4096, 1], offset: ?>>
              loom.semaphore_give %22 : memref<?x?xf16>
            } {loom.iter_type = #loom.iter_type<temporal>}
          } {loom.iter_type = #loom.iter_type<temporal>}
        } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @x}
      } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @y}
      return
    }
  }
  module {
    func.func @matmul__d1i0_d0i1__f01__d_v(%arg0: memref<4096x512xf16>, %arg1: memref<512x4096xf16>, %arg2: memref<4096x4096xf16>) {
      %cst = arith.constant 0.000000e+00 : f16
      %12 = loom.get_symbolic_block_size @constraints::@BM : index
      %13 = loom.get_symbolic_block_size @constraints::@BN : index
      %14 = loom.get_symbolic_block_size @constraints::@BK : index
      affine.parallel (%arg3) = (0) to (8) {
        affine.parallel (%arg4) = (0) to (8) {
          affine.for %arg5 = 0 to affine_map<()[s0, s1] -> (4096 ceildiv (s0 * 8))>()[%12, %13] {
            affine.for %arg6 = 0 to affine_map<()[s0, s1] -> (4096 ceildiv (s1 * 8))>()[%12, %13] {
              %15 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 8)>(%arg3, %arg5)
              %16 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 8)>(%arg4, %arg6)
              %17 = loom.alloc [%14, %13] on @L1 : memref<?x?xf16>
              %18 = loom.semaphore_take %17 : memref<?x?xf16> -> memref<?x?xf16>
              %19 = loom.alloc [%12, %14] on @L1 : memref<?x?xf16>
              %20 = loom.semaphore_take %19 : memref<?x?xf16> -> memref<?x?xf16>
              %21 = loom.alloc [%12, %13] on @L1 : memref<?x?xf16>
              %22 = loom.semaphore_take %21 : memref<?x?xf16> -> memref<?x?xf16>
              %23 = loom.init_tensor %22[%12, %13] : memref<?x?xf16> -> tensor<?x?xf16>
              %24 = linalg.fill ins(%cst : f16) outs(%23 : tensor<?x?xf16>) -> tensor<?x?xf16>
              %25 = affine.for %arg7 = 0 to affine_map<()[s0] -> (512 ceildiv s0)>()[%14] iter_args(%arg8 = %24) -> (tensor<?x?xf16>) {
                %29 = arith.muli %15, %12 : index
                %30 = arith.muli %arg7, %14 : index
                %31 = loom.subview %arg0[%29, %30] [%12, %14] [1, 1], reuse : [seq = false, spat = true, temp = true] : memref<4096x512xf16> to memref<?x?xf16, strided<[512, 1], offset: ?>>
                %32 = loom.copy_to_tensor %31, %20 on @L1, interconnect : [], broadcast : [1, 1] : memref<?x?xf16, strided<[512, 1], offset: ?>>, memref<?x?xf16> -> tensor<?x?xf16>
                %33 = arith.muli %16, %13 : index
                %34 = loom.subview %arg1[%30, %33] [%14, %13] [1, 1], reuse : [seq = false, spat = true, temp = true] : memref<512x4096xf16> to memref<?x?xf16, strided<[4096, 1], offset: ?>>
                %35 = loom.copy_to_tensor %34, %18 on @L1, interconnect : [@vertical_links], broadcast : [8, 1] : memref<?x?xf16, strided<[4096, 1], offset: ?>>, memref<?x?xf16> -> tensor<?x?xf16>
                %36 = linalg.matmul ins(%32, %35 : tensor<?x?xf16>, tensor<?x?xf16>) outs(%arg8 : tensor<?x?xf16>) -> tensor<?x?xf16>
                loom.semaphore_give %18 : memref<?x?xf16>
                loom.semaphore_give %20 : memref<?x?xf16>
                affine.yield %36 : tensor<?x?xf16>
              } {loom.iter_type = #loom.iter_type<sequential>}
              %26 = arith.muli %15, %12 : index
              %27 = arith.muli %16, %13 : index
              %28 = loom.subview %arg2[%26, %27] [%12, %13] [1, 1], reuse : [seq = false, spat = false, temp = false] : memref<4096x4096xf16> to memref<?x?xf16, strided<[4096, 1], offset: ?>>
              loom.copy_from_tensor %25, %28 on @DRAM : tensor<?x?xf16>, memref<?x?xf16, strided<[4096, 1], offset: ?>>
              loom.semaphore_give %22 : memref<?x?xf16>
            } {loom.iter_type = #loom.iter_type<temporal>}
          } {loom.iter_type = #loom.iter_type<temporal>}
        } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @x}
      } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @y}
      return
    }
  }
  module {
    func.func @matmul__d1i0_d0i1__f01__h_d(%arg0: memref<4096x512xf16>, %arg1: memref<512x4096xf16>, %arg2: memref<4096x4096xf16>) {
      %cst = arith.constant 0.000000e+00 : f16
      %12 = loom.get_symbolic_block_size @constraints::@BM : index
      %13 = loom.get_symbolic_block_size @constraints::@BN : index
      %14 = loom.get_symbolic_block_size @constraints::@BK : index
      affine.parallel (%arg3) = (0) to (8) {
        affine.parallel (%arg4) = (0) to (8) {
          affine.for %arg5 = 0 to affine_map<()[s0, s1] -> (4096 ceildiv (s0 * 8))>()[%12, %13] {
            affine.for %arg6 = 0 to affine_map<()[s0, s1] -> (4096 ceildiv (s1 * 8))>()[%12, %13] {
              %15 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 8)>(%arg3, %arg5)
              %16 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 8)>(%arg4, %arg6)
              %17 = loom.alloc [%14, %13] on @L1 : memref<?x?xf16>
              %18 = loom.semaphore_take %17 : memref<?x?xf16> -> memref<?x?xf16>
              %19 = loom.alloc [%12, %14] on @L1 : memref<?x?xf16>
              %20 = loom.semaphore_take %19 : memref<?x?xf16> -> memref<?x?xf16>
              %21 = loom.alloc [%12, %13] on @L1 : memref<?x?xf16>
              %22 = loom.semaphore_take %21 : memref<?x?xf16> -> memref<?x?xf16>
              %23 = loom.init_tensor %22[%12, %13] : memref<?x?xf16> -> tensor<?x?xf16>
              %24 = linalg.fill ins(%cst : f16) outs(%23 : tensor<?x?xf16>) -> tensor<?x?xf16>
              %25 = affine.for %arg7 = 0 to affine_map<()[s0] -> (512 ceildiv s0)>()[%14] iter_args(%arg8 = %24) -> (tensor<?x?xf16>) {
                %29 = arith.muli %15, %12 : index
                %30 = arith.muli %arg7, %14 : index
                %31 = loom.subview %arg0[%29, %30] [%12, %14] [1, 1], reuse : [seq = false, spat = true, temp = true] : memref<4096x512xf16> to memref<?x?xf16, strided<[512, 1], offset: ?>>
                %32 = loom.copy_to_tensor %31, %20 on @L1, interconnect : [@horizontal_links], broadcast : [1, 8] : memref<?x?xf16, strided<[512, 1], offset: ?>>, memref<?x?xf16> -> tensor<?x?xf16>
                %33 = arith.muli %16, %13 : index
                %34 = loom.subview %arg1[%30, %33] [%14, %13] [1, 1], reuse : [seq = false, spat = true, temp = true] : memref<512x4096xf16> to memref<?x?xf16, strided<[4096, 1], offset: ?>>
                %35 = loom.copy_to_tensor %34, %18 on @L1, interconnect : [], broadcast : [1, 1] : memref<?x?xf16, strided<[4096, 1], offset: ?>>, memref<?x?xf16> -> tensor<?x?xf16>
                %36 = linalg.matmul ins(%32, %35 : tensor<?x?xf16>, tensor<?x?xf16>) outs(%arg8 : tensor<?x?xf16>) -> tensor<?x?xf16>
                loom.semaphore_give %18 : memref<?x?xf16>
                loom.semaphore_give %20 : memref<?x?xf16>
                affine.yield %36 : tensor<?x?xf16>
              } {loom.iter_type = #loom.iter_type<sequential>}
              %26 = arith.muli %15, %12 : index
              %27 = arith.muli %16, %13 : index
              %28 = loom.subview %arg2[%26, %27] [%12, %13] [1, 1], reuse : [seq = false, spat = false, temp = false] : memref<4096x4096xf16> to memref<?x?xf16, strided<[4096, 1], offset: ?>>
              loom.copy_from_tensor %25, %28 on @DRAM : tensor<?x?xf16>, memref<?x?xf16, strided<[4096, 1], offset: ?>>
              loom.semaphore_give %22 : memref<?x?xf16>
            } {loom.iter_type = #loom.iter_type<temporal>}
          } {loom.iter_type = #loom.iter_type<temporal>}
        } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @x}
      } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @y}
      return
    }
  }
  module {
    func.func @matmul__d1i0_d0i1__f01__h_v(%arg0: memref<4096x512xf16>, %arg1: memref<512x4096xf16>, %arg2: memref<4096x4096xf16>) {
      %cst = arith.constant 0.000000e+00 : f16
      %12 = loom.get_symbolic_block_size @constraints::@BM : index
      %13 = loom.get_symbolic_block_size @constraints::@BN : index
      %14 = loom.get_symbolic_block_size @constraints::@BK : index
      affine.parallel (%arg3) = (0) to (8) {
        affine.parallel (%arg4) = (0) to (8) {
          affine.for %arg5 = 0 to affine_map<()[s0, s1] -> (4096 ceildiv (s0 * 8))>()[%12, %13] {
            affine.for %arg6 = 0 to affine_map<()[s0, s1] -> (4096 ceildiv (s1 * 8))>()[%12, %13] {
              %15 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 8)>(%arg3, %arg5)
              %16 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 8)>(%arg4, %arg6)
              %17 = loom.alloc [%14, %13] on @L1 : memref<?x?xf16>
              %18 = loom.semaphore_take %17 : memref<?x?xf16> -> memref<?x?xf16>
              %19 = loom.alloc [%12, %14] on @L1 : memref<?x?xf16>
              %20 = loom.semaphore_take %19 : memref<?x?xf16> -> memref<?x?xf16>
              %21 = loom.alloc [%12, %13] on @L1 : memref<?x?xf16>
              %22 = loom.semaphore_take %21 : memref<?x?xf16> -> memref<?x?xf16>
              %23 = loom.init_tensor %22[%12, %13] : memref<?x?xf16> -> tensor<?x?xf16>
              %24 = linalg.fill ins(%cst : f16) outs(%23 : tensor<?x?xf16>) -> tensor<?x?xf16>
              %25 = affine.for %arg7 = 0 to affine_map<()[s0] -> (512 ceildiv s0)>()[%14] iter_args(%arg8 = %24) -> (tensor<?x?xf16>) {
                %29 = arith.muli %15, %12 : index
                %30 = arith.muli %arg7, %14 : index
                %31 = loom.subview %arg0[%29, %30] [%12, %14] [1, 1], reuse : [seq = false, spat = true, temp = true] : memref<4096x512xf16> to memref<?x?xf16, strided<[512, 1], offset: ?>>
                %32 = loom.copy_to_tensor %31, %20 on @L1, interconnect : [@horizontal_links], broadcast : [1, 8] : memref<?x?xf16, strided<[512, 1], offset: ?>>, memref<?x?xf16> -> tensor<?x?xf16>
                %33 = arith.muli %16, %13 : index
                %34 = loom.subview %arg1[%30, %33] [%14, %13] [1, 1], reuse : [seq = false, spat = true, temp = true] : memref<512x4096xf16> to memref<?x?xf16, strided<[4096, 1], offset: ?>>
                %35 = loom.copy_to_tensor %34, %18 on @L1, interconnect : [@vertical_links], broadcast : [8, 1] : memref<?x?xf16, strided<[4096, 1], offset: ?>>, memref<?x?xf16> -> tensor<?x?xf16>
                %36 = linalg.matmul ins(%32, %35 : tensor<?x?xf16>, tensor<?x?xf16>) outs(%arg8 : tensor<?x?xf16>) -> tensor<?x?xf16>
                loom.semaphore_give %18 : memref<?x?xf16>
                loom.semaphore_give %20 : memref<?x?xf16>
                affine.yield %36 : tensor<?x?xf16>
              } {loom.iter_type = #loom.iter_type<sequential>}
              %26 = arith.muli %15, %12 : index
              %27 = arith.muli %16, %13 : index
              %28 = loom.subview %arg2[%26, %27] [%12, %13] [1, 1], reuse : [seq = false, spat = false, temp = false] : memref<4096x4096xf16> to memref<?x?xf16, strided<[4096, 1], offset: ?>>
              loom.copy_from_tensor %25, %28 on @DRAM : tensor<?x?xf16>, memref<?x?xf16, strided<[4096, 1], offset: ?>>
              loom.semaphore_give %22 : memref<?x?xf16>
            } {loom.iter_type = #loom.iter_type<temporal>}
          } {loom.iter_type = #loom.iter_type<temporal>}
        } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @x}
      } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @y}
      return
    }
  }
  module {
    func.func @matmul__d1i0_d0i1__f10__d_d(%arg0: memref<4096x512xf16>, %arg1: memref<512x4096xf16>, %arg2: memref<4096x4096xf16>) {
      %cst = arith.constant 0.000000e+00 : f16
      %12 = loom.get_symbolic_block_size @constraints::@BM : index
      %13 = loom.get_symbolic_block_size @constraints::@BN : index
      %14 = loom.get_symbolic_block_size @constraints::@BK : index
      affine.parallel (%arg3) = (0) to (8) {
        affine.parallel (%arg4) = (0) to (8) {
          affine.for %arg5 = 0 to affine_map<()[s0, s1] -> (4096 ceildiv (s1 * 8))>()[%12, %13] {
            affine.for %arg6 = 0 to affine_map<()[s0, s1] -> (4096 ceildiv (s0 * 8))>()[%12, %13] {
              %15 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 8)>(%arg3, %arg6)
              %16 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 8)>(%arg4, %arg5)
              %17 = loom.alloc [%14, %13] on @L1 : memref<?x?xf16>
              %18 = loom.semaphore_take %17 : memref<?x?xf16> -> memref<?x?xf16>
              %19 = loom.alloc [%12, %14] on @L1 : memref<?x?xf16>
              %20 = loom.semaphore_take %19 : memref<?x?xf16> -> memref<?x?xf16>
              %21 = loom.alloc [%12, %13] on @L1 : memref<?x?xf16>
              %22 = loom.semaphore_take %21 : memref<?x?xf16> -> memref<?x?xf16>
              %23 = loom.init_tensor %22[%12, %13] : memref<?x?xf16> -> tensor<?x?xf16>
              %24 = linalg.fill ins(%cst : f16) outs(%23 : tensor<?x?xf16>) -> tensor<?x?xf16>
              %25 = affine.for %arg7 = 0 to affine_map<()[s0] -> (512 ceildiv s0)>()[%14] iter_args(%arg8 = %24) -> (tensor<?x?xf16>) {
                %29 = arith.muli %15, %12 : index
                %30 = arith.muli %arg7, %14 : index
                %31 = loom.subview %arg0[%29, %30] [%12, %14] [1, 1], reuse : [seq = false, spat = true, temp = true] : memref<4096x512xf16> to memref<?x?xf16, strided<[512, 1], offset: ?>>
                %32 = loom.copy_to_tensor %31, %20 on @L1, interconnect : [], broadcast : [1, 1] : memref<?x?xf16, strided<[512, 1], offset: ?>>, memref<?x?xf16> -> tensor<?x?xf16>
                %33 = arith.muli %16, %13 : index
                %34 = loom.subview %arg1[%30, %33] [%14, %13] [1, 1], reuse : [seq = false, spat = true, temp = true] : memref<512x4096xf16> to memref<?x?xf16, strided<[4096, 1], offset: ?>>
                %35 = loom.copy_to_tensor %34, %18 on @L1, interconnect : [], broadcast : [1, 1] : memref<?x?xf16, strided<[4096, 1], offset: ?>>, memref<?x?xf16> -> tensor<?x?xf16>
                %36 = linalg.matmul ins(%32, %35 : tensor<?x?xf16>, tensor<?x?xf16>) outs(%arg8 : tensor<?x?xf16>) -> tensor<?x?xf16>
                loom.semaphore_give %18 : memref<?x?xf16>
                loom.semaphore_give %20 : memref<?x?xf16>
                affine.yield %36 : tensor<?x?xf16>
              } {loom.iter_type = #loom.iter_type<sequential>}
              %26 = arith.muli %15, %12 : index
              %27 = arith.muli %16, %13 : index
              %28 = loom.subview %arg2[%26, %27] [%12, %13] [1, 1], reuse : [seq = false, spat = false, temp = false] : memref<4096x4096xf16> to memref<?x?xf16, strided<[4096, 1], offset: ?>>
              loom.copy_from_tensor %25, %28 on @DRAM : tensor<?x?xf16>, memref<?x?xf16, strided<[4096, 1], offset: ?>>
              loom.semaphore_give %22 : memref<?x?xf16>
            } {loom.iter_type = #loom.iter_type<temporal>}
          } {loom.iter_type = #loom.iter_type<temporal>}
        } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @x}
      } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @y}
      return
    }
  }
  module {
    func.func @matmul__d1i0_d0i1__f10__d_v(%arg0: memref<4096x512xf16>, %arg1: memref<512x4096xf16>, %arg2: memref<4096x4096xf16>) {
      %cst = arith.constant 0.000000e+00 : f16
      %12 = loom.get_symbolic_block_size @constraints::@BM : index
      %13 = loom.get_symbolic_block_size @constraints::@BN : index
      %14 = loom.get_symbolic_block_size @constraints::@BK : index
      affine.parallel (%arg3) = (0) to (8) {
        affine.parallel (%arg4) = (0) to (8) {
          affine.for %arg5 = 0 to affine_map<()[s0, s1] -> (4096 ceildiv (s1 * 8))>()[%12, %13] {
            affine.for %arg6 = 0 to affine_map<()[s0, s1] -> (4096 ceildiv (s0 * 8))>()[%12, %13] {
              %15 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 8)>(%arg3, %arg6)
              %16 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 8)>(%arg4, %arg5)
              %17 = loom.alloc [%14, %13] on @L1 : memref<?x?xf16>
              %18 = loom.semaphore_take %17 : memref<?x?xf16> -> memref<?x?xf16>
              %19 = loom.alloc [%12, %14] on @L1 : memref<?x?xf16>
              %20 = loom.semaphore_take %19 : memref<?x?xf16> -> memref<?x?xf16>
              %21 = loom.alloc [%12, %13] on @L1 : memref<?x?xf16>
              %22 = loom.semaphore_take %21 : memref<?x?xf16> -> memref<?x?xf16>
              %23 = loom.init_tensor %22[%12, %13] : memref<?x?xf16> -> tensor<?x?xf16>
              %24 = linalg.fill ins(%cst : f16) outs(%23 : tensor<?x?xf16>) -> tensor<?x?xf16>
              %25 = affine.for %arg7 = 0 to affine_map<()[s0] -> (512 ceildiv s0)>()[%14] iter_args(%arg8 = %24) -> (tensor<?x?xf16>) {
                %29 = arith.muli %15, %12 : index
                %30 = arith.muli %arg7, %14 : index
                %31 = loom.subview %arg0[%29, %30] [%12, %14] [1, 1], reuse : [seq = false, spat = true, temp = true] : memref<4096x512xf16> to memref<?x?xf16, strided<[512, 1], offset: ?>>
                %32 = loom.copy_to_tensor %31, %20 on @L1, interconnect : [], broadcast : [1, 1] : memref<?x?xf16, strided<[512, 1], offset: ?>>, memref<?x?xf16> -> tensor<?x?xf16>
                %33 = arith.muli %16, %13 : index
                %34 = loom.subview %arg1[%30, %33] [%14, %13] [1, 1], reuse : [seq = false, spat = true, temp = true] : memref<512x4096xf16> to memref<?x?xf16, strided<[4096, 1], offset: ?>>
                %35 = loom.copy_to_tensor %34, %18 on @L1, interconnect : [@vertical_links], broadcast : [8, 1] : memref<?x?xf16, strided<[4096, 1], offset: ?>>, memref<?x?xf16> -> tensor<?x?xf16>
                %36 = linalg.matmul ins(%32, %35 : tensor<?x?xf16>, tensor<?x?xf16>) outs(%arg8 : tensor<?x?xf16>) -> tensor<?x?xf16>
                loom.semaphore_give %18 : memref<?x?xf16>
                loom.semaphore_give %20 : memref<?x?xf16>
                affine.yield %36 : tensor<?x?xf16>
              } {loom.iter_type = #loom.iter_type<sequential>}
              %26 = arith.muli %15, %12 : index
              %27 = arith.muli %16, %13 : index
              %28 = loom.subview %arg2[%26, %27] [%12, %13] [1, 1], reuse : [seq = false, spat = false, temp = false] : memref<4096x4096xf16> to memref<?x?xf16, strided<[4096, 1], offset: ?>>
              loom.copy_from_tensor %25, %28 on @DRAM : tensor<?x?xf16>, memref<?x?xf16, strided<[4096, 1], offset: ?>>
              loom.semaphore_give %22 : memref<?x?xf16>
            } {loom.iter_type = #loom.iter_type<temporal>}
          } {loom.iter_type = #loom.iter_type<temporal>}
        } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @x}
      } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @y}
      return
    }
  }
  module {
    func.func @matmul__d1i0_d0i1__f10__h_d(%arg0: memref<4096x512xf16>, %arg1: memref<512x4096xf16>, %arg2: memref<4096x4096xf16>) {
      %cst = arith.constant 0.000000e+00 : f16
      %12 = loom.get_symbolic_block_size @constraints::@BM : index
      %13 = loom.get_symbolic_block_size @constraints::@BN : index
      %14 = loom.get_symbolic_block_size @constraints::@BK : index
      affine.parallel (%arg3) = (0) to (8) {
        affine.parallel (%arg4) = (0) to (8) {
          affine.for %arg5 = 0 to affine_map<()[s0, s1] -> (4096 ceildiv (s1 * 8))>()[%12, %13] {
            affine.for %arg6 = 0 to affine_map<()[s0, s1] -> (4096 ceildiv (s0 * 8))>()[%12, %13] {
              %15 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 8)>(%arg3, %arg6)
              %16 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 8)>(%arg4, %arg5)
              %17 = loom.alloc [%14, %13] on @L1 : memref<?x?xf16>
              %18 = loom.semaphore_take %17 : memref<?x?xf16> -> memref<?x?xf16>
              %19 = loom.alloc [%12, %14] on @L1 : memref<?x?xf16>
              %20 = loom.semaphore_take %19 : memref<?x?xf16> -> memref<?x?xf16>
              %21 = loom.alloc [%12, %13] on @L1 : memref<?x?xf16>
              %22 = loom.semaphore_take %21 : memref<?x?xf16> -> memref<?x?xf16>
              %23 = loom.init_tensor %22[%12, %13] : memref<?x?xf16> -> tensor<?x?xf16>
              %24 = linalg.fill ins(%cst : f16) outs(%23 : tensor<?x?xf16>) -> tensor<?x?xf16>
              %25 = affine.for %arg7 = 0 to affine_map<()[s0] -> (512 ceildiv s0)>()[%14] iter_args(%arg8 = %24) -> (tensor<?x?xf16>) {
                %29 = arith.muli %15, %12 : index
                %30 = arith.muli %arg7, %14 : index
                %31 = loom.subview %arg0[%29, %30] [%12, %14] [1, 1], reuse : [seq = false, spat = true, temp = true] : memref<4096x512xf16> to memref<?x?xf16, strided<[512, 1], offset: ?>>
                %32 = loom.copy_to_tensor %31, %20 on @L1, interconnect : [@horizontal_links], broadcast : [1, 8] : memref<?x?xf16, strided<[512, 1], offset: ?>>, memref<?x?xf16> -> tensor<?x?xf16>
                %33 = arith.muli %16, %13 : index
                %34 = loom.subview %arg1[%30, %33] [%14, %13] [1, 1], reuse : [seq = false, spat = true, temp = true] : memref<512x4096xf16> to memref<?x?xf16, strided<[4096, 1], offset: ?>>
                %35 = loom.copy_to_tensor %34, %18 on @L1, interconnect : [], broadcast : [1, 1] : memref<?x?xf16, strided<[4096, 1], offset: ?>>, memref<?x?xf16> -> tensor<?x?xf16>
                %36 = linalg.matmul ins(%32, %35 : tensor<?x?xf16>, tensor<?x?xf16>) outs(%arg8 : tensor<?x?xf16>) -> tensor<?x?xf16>
                loom.semaphore_give %18 : memref<?x?xf16>
                loom.semaphore_give %20 : memref<?x?xf16>
                affine.yield %36 : tensor<?x?xf16>
              } {loom.iter_type = #loom.iter_type<sequential>}
              %26 = arith.muli %15, %12 : index
              %27 = arith.muli %16, %13 : index
              %28 = loom.subview %arg2[%26, %27] [%12, %13] [1, 1], reuse : [seq = false, spat = false, temp = false] : memref<4096x4096xf16> to memref<?x?xf16, strided<[4096, 1], offset: ?>>
              loom.copy_from_tensor %25, %28 on @DRAM : tensor<?x?xf16>, memref<?x?xf16, strided<[4096, 1], offset: ?>>
              loom.semaphore_give %22 : memref<?x?xf16>
            } {loom.iter_type = #loom.iter_type<temporal>}
          } {loom.iter_type = #loom.iter_type<temporal>}
        } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @x}
      } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @y}
      return
    }
  }
  module {
    func.func @matmul__d1i0_d0i1__f10__h_v(%arg0: memref<4096x512xf16>, %arg1: memref<512x4096xf16>, %arg2: memref<4096x4096xf16>) {
      %cst = arith.constant 0.000000e+00 : f16
      %12 = loom.get_symbolic_block_size @constraints::@BM : index
      %13 = loom.get_symbolic_block_size @constraints::@BN : index
      %14 = loom.get_symbolic_block_size @constraints::@BK : index
      affine.parallel (%arg3) = (0) to (8) {
        affine.parallel (%arg4) = (0) to (8) {
          affine.for %arg5 = 0 to affine_map<()[s0, s1] -> (4096 ceildiv (s1 * 8))>()[%12, %13] {
            affine.for %arg6 = 0 to affine_map<()[s0, s1] -> (4096 ceildiv (s0 * 8))>()[%12, %13] {
              %15 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 8)>(%arg3, %arg6)
              %16 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 8)>(%arg4, %arg5)
              %17 = loom.alloc [%14, %13] on @L1 : memref<?x?xf16>
              %18 = loom.semaphore_take %17 : memref<?x?xf16> -> memref<?x?xf16>
              %19 = loom.alloc [%12, %14] on @L1 : memref<?x?xf16>
              %20 = loom.semaphore_take %19 : memref<?x?xf16> -> memref<?x?xf16>
              %21 = loom.alloc [%12, %13] on @L1 : memref<?x?xf16>
              %22 = loom.semaphore_take %21 : memref<?x?xf16> -> memref<?x?xf16>
              %23 = loom.init_tensor %22[%12, %13] : memref<?x?xf16> -> tensor<?x?xf16>
              %24 = linalg.fill ins(%cst : f16) outs(%23 : tensor<?x?xf16>) -> tensor<?x?xf16>
              %25 = affine.for %arg7 = 0 to affine_map<()[s0] -> (512 ceildiv s0)>()[%14] iter_args(%arg8 = %24) -> (tensor<?x?xf16>) {
                %29 = arith.muli %15, %12 : index
                %30 = arith.muli %arg7, %14 : index
                %31 = loom.subview %arg0[%29, %30] [%12, %14] [1, 1], reuse : [seq = false, spat = true, temp = true] : memref<4096x512xf16> to memref<?x?xf16, strided<[512, 1], offset: ?>>
                %32 = loom.copy_to_tensor %31, %20 on @L1, interconnect : [@horizontal_links], broadcast : [1, 8] : memref<?x?xf16, strided<[512, 1], offset: ?>>, memref<?x?xf16> -> tensor<?x?xf16>
                %33 = arith.muli %16, %13 : index
                %34 = loom.subview %arg1[%30, %33] [%14, %13] [1, 1], reuse : [seq = false, spat = true, temp = true] : memref<512x4096xf16> to memref<?x?xf16, strided<[4096, 1], offset: ?>>
                %35 = loom.copy_to_tensor %34, %18 on @L1, interconnect : [@vertical_links], broadcast : [8, 1] : memref<?x?xf16, strided<[4096, 1], offset: ?>>, memref<?x?xf16> -> tensor<?x?xf16>
                %36 = linalg.matmul ins(%32, %35 : tensor<?x?xf16>, tensor<?x?xf16>) outs(%arg8 : tensor<?x?xf16>) -> tensor<?x?xf16>
                loom.semaphore_give %18 : memref<?x?xf16>
                loom.semaphore_give %20 : memref<?x?xf16>
                affine.yield %36 : tensor<?x?xf16>
              } {loom.iter_type = #loom.iter_type<sequential>}
              %26 = arith.muli %15, %12 : index
              %27 = arith.muli %16, %13 : index
              %28 = loom.subview %arg2[%26, %27] [%12, %13] [1, 1], reuse : [seq = false, spat = false, temp = false] : memref<4096x4096xf16> to memref<?x?xf16, strided<[4096, 1], offset: ?>>
              loom.copy_from_tensor %25, %28 on @DRAM : tensor<?x?xf16>, memref<?x?xf16, strided<[4096, 1], offset: ?>>
              loom.semaphore_give %22 : memref<?x?xf16>
            } {loom.iter_type = #loom.iter_type<temporal>}
          } {loom.iter_type = #loom.iter_type<temporal>}
        } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @x}
      } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @y}
      return
    }
  }
  module {
    func.func @matmul__d0i1_d1i1__f01__d_d(%arg0: memref<4096x512xf16>, %arg1: memref<512x4096xf16>, %arg2: memref<4096x4096xf16>) {
      %cst = arith.constant 0.000000e+00 : f16
      %12 = loom.get_symbolic_block_size @constraints::@BM : index
      %13 = loom.get_symbolic_block_size @constraints::@BN : index
      %14 = loom.get_symbolic_block_size @constraints::@BK : index
      affine.parallel (%arg3) = (0) to (8) {
        affine.parallel (%arg4) = (0) to (8) {
          affine.for %arg5 = 0 to affine_map<()[s0, s1] -> (4096 ceildiv s0)>()[%12, %13] {
            affine.for %arg6 = 0 to affine_map<()[s0, s1] -> (4096 ceildiv (s1 * 64))>()[%12, %13] {
              %15 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 8 + d1 + d2 * 64)>(%arg3, %arg4, %arg6)
              %16 = loom.alloc [%14, %13] on @L1 : memref<?x?xf16>
              %17 = loom.semaphore_take %16 : memref<?x?xf16> -> memref<?x?xf16>
              %18 = loom.alloc [%12, %14] on @L1 : memref<?x?xf16>
              %19 = loom.semaphore_take %18 : memref<?x?xf16> -> memref<?x?xf16>
              %20 = loom.alloc [%12, %13] on @L1 : memref<?x?xf16>
              %21 = loom.semaphore_take %20 : memref<?x?xf16> -> memref<?x?xf16>
              %22 = loom.init_tensor %21[%12, %13] : memref<?x?xf16> -> tensor<?x?xf16>
              %23 = linalg.fill ins(%cst : f16) outs(%22 : tensor<?x?xf16>) -> tensor<?x?xf16>
              %24 = affine.for %arg7 = 0 to affine_map<()[s0] -> (512 ceildiv s0)>()[%14] iter_args(%arg8 = %23) -> (tensor<?x?xf16>) {
                %28 = arith.muli %arg5, %12 : index
                %29 = arith.muli %arg7, %14 : index
                %30 = loom.subview %arg0[%28, %29] [%12, %14] [1, 1], reuse : [seq = false, spat = true, temp = true] : memref<4096x512xf16> to memref<?x?xf16, strided<[512, 1], offset: ?>>
                %31 = loom.copy_to_tensor %30, %19 on @L1, interconnect : [], broadcast : [1, 1] : memref<?x?xf16, strided<[512, 1], offset: ?>>, memref<?x?xf16> -> tensor<?x?xf16>
                %32 = arith.muli %15, %13 : index
                %33 = loom.subview %arg1[%29, %32] [%14, %13] [1, 1], reuse : [seq = false, spat = false, temp = true] : memref<512x4096xf16> to memref<?x?xf16, strided<[4096, 1], offset: ?>>
                %34 = loom.copy_to_tensor %33, %17 on @L1, interconnect : [], broadcast : [1, 1] : memref<?x?xf16, strided<[4096, 1], offset: ?>>, memref<?x?xf16> -> tensor<?x?xf16>
                %35 = linalg.matmul ins(%31, %34 : tensor<?x?xf16>, tensor<?x?xf16>) outs(%arg8 : tensor<?x?xf16>) -> tensor<?x?xf16>
                loom.semaphore_give %17 : memref<?x?xf16>
                loom.semaphore_give %19 : memref<?x?xf16>
                affine.yield %35 : tensor<?x?xf16>
              } {loom.iter_type = #loom.iter_type<sequential>}
              %25 = arith.muli %arg5, %12 : index
              %26 = arith.muli %15, %13 : index
              %27 = loom.subview %arg2[%25, %26] [%12, %13] [1, 1], reuse : [seq = false, spat = false, temp = false] : memref<4096x4096xf16> to memref<?x?xf16, strided<[4096, 1], offset: ?>>
              loom.copy_from_tensor %24, %27 on @DRAM : tensor<?x?xf16>, memref<?x?xf16, strided<[4096, 1], offset: ?>>
              loom.semaphore_give %21 : memref<?x?xf16>
            } {loom.iter_type = #loom.iter_type<temporal>}
          } {loom.iter_type = #loom.iter_type<temporal>}
        } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @y}
      } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @x}
      return
    }
  }
  module {
    func.func @matmul__d0i1_d1i1__f01__a_d(%arg0: memref<4096x512xf16>, %arg1: memref<512x4096xf16>, %arg2: memref<4096x4096xf16>) {
      %cst = arith.constant 0.000000e+00 : f16
      %12 = loom.get_symbolic_block_size @constraints::@BM : index
      %13 = loom.get_symbolic_block_size @constraints::@BN : index
      %14 = loom.get_symbolic_block_size @constraints::@BK : index
      affine.parallel (%arg3) = (0) to (8) {
        affine.parallel (%arg4) = (0) to (8) {
          affine.for %arg5 = 0 to affine_map<()[s0, s1] -> (4096 ceildiv s0)>()[%12, %13] {
            affine.for %arg6 = 0 to affine_map<()[s0, s1] -> (4096 ceildiv (s1 * 64))>()[%12, %13] {
              %15 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 8 + d1 + d2 * 64)>(%arg3, %arg4, %arg6)
              %16 = loom.alloc [%14, %13] on @L1 : memref<?x?xf16>
              %17 = loom.semaphore_take %16 : memref<?x?xf16> -> memref<?x?xf16>
              %18 = loom.alloc [%12, %14] on @L1 : memref<?x?xf16>
              %19 = loom.semaphore_take %18 : memref<?x?xf16> -> memref<?x?xf16>
              %20 = loom.alloc [%12, %13] on @L1 : memref<?x?xf16>
              %21 = loom.semaphore_take %20 : memref<?x?xf16> -> memref<?x?xf16>
              %22 = loom.init_tensor %21[%12, %13] : memref<?x?xf16> -> tensor<?x?xf16>
              %23 = linalg.fill ins(%cst : f16) outs(%22 : tensor<?x?xf16>) -> tensor<?x?xf16>
              %24 = affine.for %arg7 = 0 to affine_map<()[s0] -> (512 ceildiv s0)>()[%14] iter_args(%arg8 = %23) -> (tensor<?x?xf16>) {
                %28 = arith.muli %arg5, %12 : index
                %29 = arith.muli %arg7, %14 : index
                %30 = loom.subview %arg0[%28, %29] [%12, %14] [1, 1], reuse : [seq = false, spat = true, temp = true] : memref<4096x512xf16> to memref<?x?xf16, strided<[512, 1], offset: ?>>
                %31 = loom.copy_to_tensor %30, %19 on @L1, interconnect : [@horizontal_links, @vertical_links], broadcast : [8, 8] : memref<?x?xf16, strided<[512, 1], offset: ?>>, memref<?x?xf16> -> tensor<?x?xf16>
                %32 = arith.muli %15, %13 : index
                %33 = loom.subview %arg1[%29, %32] [%14, %13] [1, 1], reuse : [seq = false, spat = false, temp = true] : memref<512x4096xf16> to memref<?x?xf16, strided<[4096, 1], offset: ?>>
                %34 = loom.copy_to_tensor %33, %17 on @L1, interconnect : [], broadcast : [1, 1] : memref<?x?xf16, strided<[4096, 1], offset: ?>>, memref<?x?xf16> -> tensor<?x?xf16>
                %35 = linalg.matmul ins(%31, %34 : tensor<?x?xf16>, tensor<?x?xf16>) outs(%arg8 : tensor<?x?xf16>) -> tensor<?x?xf16>
                loom.semaphore_give %17 : memref<?x?xf16>
                loom.semaphore_give %19 : memref<?x?xf16>
                affine.yield %35 : tensor<?x?xf16>
              } {loom.iter_type = #loom.iter_type<sequential>}
              %25 = arith.muli %arg5, %12 : index
              %26 = arith.muli %15, %13 : index
              %27 = loom.subview %arg2[%25, %26] [%12, %13] [1, 1], reuse : [seq = false, spat = false, temp = false] : memref<4096x4096xf16> to memref<?x?xf16, strided<[4096, 1], offset: ?>>
              loom.copy_from_tensor %24, %27 on @DRAM : tensor<?x?xf16>, memref<?x?xf16, strided<[4096, 1], offset: ?>>
              loom.semaphore_give %21 : memref<?x?xf16>
            } {loom.iter_type = #loom.iter_type<temporal>}
          } {loom.iter_type = #loom.iter_type<temporal>}
        } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @y}
      } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @x}
      return
    }
  }
  module {
    func.func @matmul__d0i1_d1i1__f01__h_d(%arg0: memref<4096x512xf16>, %arg1: memref<512x4096xf16>, %arg2: memref<4096x4096xf16>) {
      %cst = arith.constant 0.000000e+00 : f16
      %12 = loom.get_symbolic_block_size @constraints::@BM : index
      %13 = loom.get_symbolic_block_size @constraints::@BN : index
      %14 = loom.get_symbolic_block_size @constraints::@BK : index
      affine.parallel (%arg3) = (0) to (8) {
        affine.parallel (%arg4) = (0) to (8) {
          affine.for %arg5 = 0 to affine_map<()[s0, s1] -> (4096 ceildiv s0)>()[%12, %13] {
            affine.for %arg6 = 0 to affine_map<()[s0, s1] -> (4096 ceildiv (s1 * 64))>()[%12, %13] {
              %15 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 8 + d1 + d2 * 64)>(%arg3, %arg4, %arg6)
              %16 = loom.alloc [%14, %13] on @L1 : memref<?x?xf16>
              %17 = loom.semaphore_take %16 : memref<?x?xf16> -> memref<?x?xf16>
              %18 = loom.alloc [%12, %14] on @L1 : memref<?x?xf16>
              %19 = loom.semaphore_take %18 : memref<?x?xf16> -> memref<?x?xf16>
              %20 = loom.alloc [%12, %13] on @L1 : memref<?x?xf16>
              %21 = loom.semaphore_take %20 : memref<?x?xf16> -> memref<?x?xf16>
              %22 = loom.init_tensor %21[%12, %13] : memref<?x?xf16> -> tensor<?x?xf16>
              %23 = linalg.fill ins(%cst : f16) outs(%22 : tensor<?x?xf16>) -> tensor<?x?xf16>
              %24 = affine.for %arg7 = 0 to affine_map<()[s0] -> (512 ceildiv s0)>()[%14] iter_args(%arg8 = %23) -> (tensor<?x?xf16>) {
                %28 = arith.muli %arg5, %12 : index
                %29 = arith.muli %arg7, %14 : index
                %30 = loom.subview %arg0[%28, %29] [%12, %14] [1, 1], reuse : [seq = false, spat = true, temp = true] : memref<4096x512xf16> to memref<?x?xf16, strided<[512, 1], offset: ?>>
                %31 = loom.copy_to_tensor %30, %19 on @L1, interconnect : [@horizontal_links], broadcast : [1, 8] : memref<?x?xf16, strided<[512, 1], offset: ?>>, memref<?x?xf16> -> tensor<?x?xf16>
                %32 = arith.muli %15, %13 : index
                %33 = loom.subview %arg1[%29, %32] [%14, %13] [1, 1], reuse : [seq = false, spat = false, temp = true] : memref<512x4096xf16> to memref<?x?xf16, strided<[4096, 1], offset: ?>>
                %34 = loom.copy_to_tensor %33, %17 on @L1, interconnect : [], broadcast : [1, 1] : memref<?x?xf16, strided<[4096, 1], offset: ?>>, memref<?x?xf16> -> tensor<?x?xf16>
                %35 = linalg.matmul ins(%31, %34 : tensor<?x?xf16>, tensor<?x?xf16>) outs(%arg8 : tensor<?x?xf16>) -> tensor<?x?xf16>
                loom.semaphore_give %17 : memref<?x?xf16>
                loom.semaphore_give %19 : memref<?x?xf16>
                affine.yield %35 : tensor<?x?xf16>
              } {loom.iter_type = #loom.iter_type<sequential>}
              %25 = arith.muli %arg5, %12 : index
              %26 = arith.muli %15, %13 : index
              %27 = loom.subview %arg2[%25, %26] [%12, %13] [1, 1], reuse : [seq = false, spat = false, temp = false] : memref<4096x4096xf16> to memref<?x?xf16, strided<[4096, 1], offset: ?>>
              loom.copy_from_tensor %24, %27 on @DRAM : tensor<?x?xf16>, memref<?x?xf16, strided<[4096, 1], offset: ?>>
              loom.semaphore_give %21 : memref<?x?xf16>
            } {loom.iter_type = #loom.iter_type<temporal>}
          } {loom.iter_type = #loom.iter_type<temporal>}
        } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @y}
      } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @x}
      return
    }
  }
  module {
    func.func @matmul__d0i1_d1i1__f01__v_d(%arg0: memref<4096x512xf16>, %arg1: memref<512x4096xf16>, %arg2: memref<4096x4096xf16>) {
      %cst = arith.constant 0.000000e+00 : f16
      %12 = loom.get_symbolic_block_size @constraints::@BM : index
      %13 = loom.get_symbolic_block_size @constraints::@BN : index
      %14 = loom.get_symbolic_block_size @constraints::@BK : index
      affine.parallel (%arg3) = (0) to (8) {
        affine.parallel (%arg4) = (0) to (8) {
          affine.for %arg5 = 0 to affine_map<()[s0, s1] -> (4096 ceildiv s0)>()[%12, %13] {
            affine.for %arg6 = 0 to affine_map<()[s0, s1] -> (4096 ceildiv (s1 * 64))>()[%12, %13] {
              %15 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 8 + d1 + d2 * 64)>(%arg3, %arg4, %arg6)
              %16 = loom.alloc [%14, %13] on @L1 : memref<?x?xf16>
              %17 = loom.semaphore_take %16 : memref<?x?xf16> -> memref<?x?xf16>
              %18 = loom.alloc [%12, %14] on @L1 : memref<?x?xf16>
              %19 = loom.semaphore_take %18 : memref<?x?xf16> -> memref<?x?xf16>
              %20 = loom.alloc [%12, %13] on @L1 : memref<?x?xf16>
              %21 = loom.semaphore_take %20 : memref<?x?xf16> -> memref<?x?xf16>
              %22 = loom.init_tensor %21[%12, %13] : memref<?x?xf16> -> tensor<?x?xf16>
              %23 = linalg.fill ins(%cst : f16) outs(%22 : tensor<?x?xf16>) -> tensor<?x?xf16>
              %24 = affine.for %arg7 = 0 to affine_map<()[s0] -> (512 ceildiv s0)>()[%14] iter_args(%arg8 = %23) -> (tensor<?x?xf16>) {
                %28 = arith.muli %arg5, %12 : index
                %29 = arith.muli %arg7, %14 : index
                %30 = loom.subview %arg0[%28, %29] [%12, %14] [1, 1], reuse : [seq = false, spat = true, temp = true] : memref<4096x512xf16> to memref<?x?xf16, strided<[512, 1], offset: ?>>
                %31 = loom.copy_to_tensor %30, %19 on @L1, interconnect : [@vertical_links], broadcast : [8, 1] : memref<?x?xf16, strided<[512, 1], offset: ?>>, memref<?x?xf16> -> tensor<?x?xf16>
                %32 = arith.muli %15, %13 : index
                %33 = loom.subview %arg1[%29, %32] [%14, %13] [1, 1], reuse : [seq = false, spat = false, temp = true] : memref<512x4096xf16> to memref<?x?xf16, strided<[4096, 1], offset: ?>>
                %34 = loom.copy_to_tensor %33, %17 on @L1, interconnect : [], broadcast : [1, 1] : memref<?x?xf16, strided<[4096, 1], offset: ?>>, memref<?x?xf16> -> tensor<?x?xf16>
                %35 = linalg.matmul ins(%31, %34 : tensor<?x?xf16>, tensor<?x?xf16>) outs(%arg8 : tensor<?x?xf16>) -> tensor<?x?xf16>
                loom.semaphore_give %17 : memref<?x?xf16>
                loom.semaphore_give %19 : memref<?x?xf16>
                affine.yield %35 : tensor<?x?xf16>
              } {loom.iter_type = #loom.iter_type<sequential>}
              %25 = arith.muli %arg5, %12 : index
              %26 = arith.muli %15, %13 : index
              %27 = loom.subview %arg2[%25, %26] [%12, %13] [1, 1], reuse : [seq = false, spat = false, temp = false] : memref<4096x4096xf16> to memref<?x?xf16, strided<[4096, 1], offset: ?>>
              loom.copy_from_tensor %24, %27 on @DRAM : tensor<?x?xf16>, memref<?x?xf16, strided<[4096, 1], offset: ?>>
              loom.semaphore_give %21 : memref<?x?xf16>
            } {loom.iter_type = #loom.iter_type<temporal>}
          } {loom.iter_type = #loom.iter_type<temporal>}
        } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @y}
      } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @x}
      return
    }
  }
  module {
    func.func @matmul__d0i1_d1i1__f10__d_d(%arg0: memref<4096x512xf16>, %arg1: memref<512x4096xf16>, %arg2: memref<4096x4096xf16>) {
      %cst = arith.constant 0.000000e+00 : f16
      %12 = loom.get_symbolic_block_size @constraints::@BM : index
      %13 = loom.get_symbolic_block_size @constraints::@BN : index
      %14 = loom.get_symbolic_block_size @constraints::@BK : index
      affine.parallel (%arg3) = (0) to (8) {
        affine.parallel (%arg4) = (0) to (8) {
          affine.for %arg5 = 0 to affine_map<()[s0, s1] -> (4096 ceildiv (s1 * 64))>()[%12, %13] {
            affine.for %arg6 = 0 to affine_map<()[s0, s1] -> (4096 ceildiv s0)>()[%12, %13] {
              %15 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 8 + d1 + d2 * 64)>(%arg3, %arg4, %arg5)
              %16 = loom.alloc [%14, %13] on @L1 : memref<?x?xf16>
              %17 = loom.semaphore_take %16 : memref<?x?xf16> -> memref<?x?xf16>
              %18 = loom.alloc [%12, %14] on @L1 : memref<?x?xf16>
              %19 = loom.semaphore_take %18 : memref<?x?xf16> -> memref<?x?xf16>
              %20 = loom.alloc [%12, %13] on @L1 : memref<?x?xf16>
              %21 = loom.semaphore_take %20 : memref<?x?xf16> -> memref<?x?xf16>
              %22 = loom.init_tensor %21[%12, %13] : memref<?x?xf16> -> tensor<?x?xf16>
              %23 = linalg.fill ins(%cst : f16) outs(%22 : tensor<?x?xf16>) -> tensor<?x?xf16>
              %24 = affine.for %arg7 = 0 to affine_map<()[s0] -> (512 ceildiv s0)>()[%14] iter_args(%arg8 = %23) -> (tensor<?x?xf16>) {
                %28 = arith.muli %arg6, %12 : index
                %29 = arith.muli %arg7, %14 : index
                %30 = loom.subview %arg0[%28, %29] [%12, %14] [1, 1], reuse : [seq = false, spat = true, temp = true] : memref<4096x512xf16> to memref<?x?xf16, strided<[512, 1], offset: ?>>
                %31 = loom.copy_to_tensor %30, %19 on @L1, interconnect : [], broadcast : [1, 1] : memref<?x?xf16, strided<[512, 1], offset: ?>>, memref<?x?xf16> -> tensor<?x?xf16>
                %32 = arith.muli %15, %13 : index
                %33 = loom.subview %arg1[%29, %32] [%14, %13] [1, 1], reuse : [seq = false, spat = false, temp = true] : memref<512x4096xf16> to memref<?x?xf16, strided<[4096, 1], offset: ?>>
                %34 = loom.copy_to_tensor %33, %17 on @L1, interconnect : [], broadcast : [1, 1] : memref<?x?xf16, strided<[4096, 1], offset: ?>>, memref<?x?xf16> -> tensor<?x?xf16>
                %35 = linalg.matmul ins(%31, %34 : tensor<?x?xf16>, tensor<?x?xf16>) outs(%arg8 : tensor<?x?xf16>) -> tensor<?x?xf16>
                loom.semaphore_give %17 : memref<?x?xf16>
                loom.semaphore_give %19 : memref<?x?xf16>
                affine.yield %35 : tensor<?x?xf16>
              } {loom.iter_type = #loom.iter_type<sequential>}
              %25 = arith.muli %arg6, %12 : index
              %26 = arith.muli %15, %13 : index
              %27 = loom.subview %arg2[%25, %26] [%12, %13] [1, 1], reuse : [seq = false, spat = false, temp = false] : memref<4096x4096xf16> to memref<?x?xf16, strided<[4096, 1], offset: ?>>
              loom.copy_from_tensor %24, %27 on @DRAM : tensor<?x?xf16>, memref<?x?xf16, strided<[4096, 1], offset: ?>>
              loom.semaphore_give %21 : memref<?x?xf16>
            } {loom.iter_type = #loom.iter_type<temporal>}
          } {loom.iter_type = #loom.iter_type<temporal>}
        } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @y}
      } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @x}
      return
    }
  }
  module {
    func.func @matmul__d0i1_d1i1__f10__a_d(%arg0: memref<4096x512xf16>, %arg1: memref<512x4096xf16>, %arg2: memref<4096x4096xf16>) {
      %cst = arith.constant 0.000000e+00 : f16
      %12 = loom.get_symbolic_block_size @constraints::@BM : index
      %13 = loom.get_symbolic_block_size @constraints::@BN : index
      %14 = loom.get_symbolic_block_size @constraints::@BK : index
      affine.parallel (%arg3) = (0) to (8) {
        affine.parallel (%arg4) = (0) to (8) {
          affine.for %arg5 = 0 to affine_map<()[s0, s1] -> (4096 ceildiv (s1 * 64))>()[%12, %13] {
            affine.for %arg6 = 0 to affine_map<()[s0, s1] -> (4096 ceildiv s0)>()[%12, %13] {
              %15 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 8 + d1 + d2 * 64)>(%arg3, %arg4, %arg5)
              %16 = loom.alloc [%14, %13] on @L1 : memref<?x?xf16>
              %17 = loom.semaphore_take %16 : memref<?x?xf16> -> memref<?x?xf16>
              %18 = loom.alloc [%12, %14] on @L1 : memref<?x?xf16>
              %19 = loom.semaphore_take %18 : memref<?x?xf16> -> memref<?x?xf16>
              %20 = loom.alloc [%12, %13] on @L1 : memref<?x?xf16>
              %21 = loom.semaphore_take %20 : memref<?x?xf16> -> memref<?x?xf16>
              %22 = loom.init_tensor %21[%12, %13] : memref<?x?xf16> -> tensor<?x?xf16>
              %23 = linalg.fill ins(%cst : f16) outs(%22 : tensor<?x?xf16>) -> tensor<?x?xf16>
              %24 = affine.for %arg7 = 0 to affine_map<()[s0] -> (512 ceildiv s0)>()[%14] iter_args(%arg8 = %23) -> (tensor<?x?xf16>) {
                %28 = arith.muli %arg6, %12 : index
                %29 = arith.muli %arg7, %14 : index
                %30 = loom.subview %arg0[%28, %29] [%12, %14] [1, 1], reuse : [seq = false, spat = true, temp = true] : memref<4096x512xf16> to memref<?x?xf16, strided<[512, 1], offset: ?>>
                %31 = loom.copy_to_tensor %30, %19 on @L1, interconnect : [@horizontal_links, @vertical_links], broadcast : [8, 8] : memref<?x?xf16, strided<[512, 1], offset: ?>>, memref<?x?xf16> -> tensor<?x?xf16>
                %32 = arith.muli %15, %13 : index
                %33 = loom.subview %arg1[%29, %32] [%14, %13] [1, 1], reuse : [seq = false, spat = false, temp = true] : memref<512x4096xf16> to memref<?x?xf16, strided<[4096, 1], offset: ?>>
                %34 = loom.copy_to_tensor %33, %17 on @L1, interconnect : [], broadcast : [1, 1] : memref<?x?xf16, strided<[4096, 1], offset: ?>>, memref<?x?xf16> -> tensor<?x?xf16>
                %35 = linalg.matmul ins(%31, %34 : tensor<?x?xf16>, tensor<?x?xf16>) outs(%arg8 : tensor<?x?xf16>) -> tensor<?x?xf16>
                loom.semaphore_give %17 : memref<?x?xf16>
                loom.semaphore_give %19 : memref<?x?xf16>
                affine.yield %35 : tensor<?x?xf16>
              } {loom.iter_type = #loom.iter_type<sequential>}
              %25 = arith.muli %arg6, %12 : index
              %26 = arith.muli %15, %13 : index
              %27 = loom.subview %arg2[%25, %26] [%12, %13] [1, 1], reuse : [seq = false, spat = false, temp = false] : memref<4096x4096xf16> to memref<?x?xf16, strided<[4096, 1], offset: ?>>
              loom.copy_from_tensor %24, %27 on @DRAM : tensor<?x?xf16>, memref<?x?xf16, strided<[4096, 1], offset: ?>>
              loom.semaphore_give %21 : memref<?x?xf16>
            } {loom.iter_type = #loom.iter_type<temporal>}
          } {loom.iter_type = #loom.iter_type<temporal>}
        } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @y}
      } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @x}
      return
    }
  }
  module {
    func.func @matmul__d0i1_d1i1__f10__h_d(%arg0: memref<4096x512xf16>, %arg1: memref<512x4096xf16>, %arg2: memref<4096x4096xf16>) {
      %cst = arith.constant 0.000000e+00 : f16
      %12 = loom.get_symbolic_block_size @constraints::@BM : index
      %13 = loom.get_symbolic_block_size @constraints::@BN : index
      %14 = loom.get_symbolic_block_size @constraints::@BK : index
      affine.parallel (%arg3) = (0) to (8) {
        affine.parallel (%arg4) = (0) to (8) {
          affine.for %arg5 = 0 to affine_map<()[s0, s1] -> (4096 ceildiv (s1 * 64))>()[%12, %13] {
            affine.for %arg6 = 0 to affine_map<()[s0, s1] -> (4096 ceildiv s0)>()[%12, %13] {
              %15 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 8 + d1 + d2 * 64)>(%arg3, %arg4, %arg5)
              %16 = loom.alloc [%14, %13] on @L1 : memref<?x?xf16>
              %17 = loom.semaphore_take %16 : memref<?x?xf16> -> memref<?x?xf16>
              %18 = loom.alloc [%12, %14] on @L1 : memref<?x?xf16>
              %19 = loom.semaphore_take %18 : memref<?x?xf16> -> memref<?x?xf16>
              %20 = loom.alloc [%12, %13] on @L1 : memref<?x?xf16>
              %21 = loom.semaphore_take %20 : memref<?x?xf16> -> memref<?x?xf16>
              %22 = loom.init_tensor %21[%12, %13] : memref<?x?xf16> -> tensor<?x?xf16>
              %23 = linalg.fill ins(%cst : f16) outs(%22 : tensor<?x?xf16>) -> tensor<?x?xf16>
              %24 = affine.for %arg7 = 0 to affine_map<()[s0] -> (512 ceildiv s0)>()[%14] iter_args(%arg8 = %23) -> (tensor<?x?xf16>) {
                %28 = arith.muli %arg6, %12 : index
                %29 = arith.muli %arg7, %14 : index
                %30 = loom.subview %arg0[%28, %29] [%12, %14] [1, 1], reuse : [seq = false, spat = true, temp = true] : memref<4096x512xf16> to memref<?x?xf16, strided<[512, 1], offset: ?>>
                %31 = loom.copy_to_tensor %30, %19 on @L1, interconnect : [@horizontal_links], broadcast : [1, 8] : memref<?x?xf16, strided<[512, 1], offset: ?>>, memref<?x?xf16> -> tensor<?x?xf16>
                %32 = arith.muli %15, %13 : index
                %33 = loom.subview %arg1[%29, %32] [%14, %13] [1, 1], reuse : [seq = false, spat = false, temp = true] : memref<512x4096xf16> to memref<?x?xf16, strided<[4096, 1], offset: ?>>
                %34 = loom.copy_to_tensor %33, %17 on @L1, interconnect : [], broadcast : [1, 1] : memref<?x?xf16, strided<[4096, 1], offset: ?>>, memref<?x?xf16> -> tensor<?x?xf16>
                %35 = linalg.matmul ins(%31, %34 : tensor<?x?xf16>, tensor<?x?xf16>) outs(%arg8 : tensor<?x?xf16>) -> tensor<?x?xf16>
                loom.semaphore_give %17 : memref<?x?xf16>
                loom.semaphore_give %19 : memref<?x?xf16>
                affine.yield %35 : tensor<?x?xf16>
              } {loom.iter_type = #loom.iter_type<sequential>}
              %25 = arith.muli %arg6, %12 : index
              %26 = arith.muli %15, %13 : index
              %27 = loom.subview %arg2[%25, %26] [%12, %13] [1, 1], reuse : [seq = false, spat = false, temp = false] : memref<4096x4096xf16> to memref<?x?xf16, strided<[4096, 1], offset: ?>>
              loom.copy_from_tensor %24, %27 on @DRAM : tensor<?x?xf16>, memref<?x?xf16, strided<[4096, 1], offset: ?>>
              loom.semaphore_give %21 : memref<?x?xf16>
            } {loom.iter_type = #loom.iter_type<temporal>}
          } {loom.iter_type = #loom.iter_type<temporal>}
        } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @y}
      } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @x}
      return
    }
  }
  module {
    func.func @matmul__d0i1_d1i1__f10__v_d(%arg0: memref<4096x512xf16>, %arg1: memref<512x4096xf16>, %arg2: memref<4096x4096xf16>) {
      %cst = arith.constant 0.000000e+00 : f16
      %12 = loom.get_symbolic_block_size @constraints::@BM : index
      %13 = loom.get_symbolic_block_size @constraints::@BN : index
      %14 = loom.get_symbolic_block_size @constraints::@BK : index
      affine.parallel (%arg3) = (0) to (8) {
        affine.parallel (%arg4) = (0) to (8) {
          affine.for %arg5 = 0 to affine_map<()[s0, s1] -> (4096 ceildiv (s1 * 64))>()[%12, %13] {
            affine.for %arg6 = 0 to affine_map<()[s0, s1] -> (4096 ceildiv s0)>()[%12, %13] {
              %15 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 8 + d1 + d2 * 64)>(%arg3, %arg4, %arg5)
              %16 = loom.alloc [%14, %13] on @L1 : memref<?x?xf16>
              %17 = loom.semaphore_take %16 : memref<?x?xf16> -> memref<?x?xf16>
              %18 = loom.alloc [%12, %14] on @L1 : memref<?x?xf16>
              %19 = loom.semaphore_take %18 : memref<?x?xf16> -> memref<?x?xf16>
              %20 = loom.alloc [%12, %13] on @L1 : memref<?x?xf16>
              %21 = loom.semaphore_take %20 : memref<?x?xf16> -> memref<?x?xf16>
              %22 = loom.init_tensor %21[%12, %13] : memref<?x?xf16> -> tensor<?x?xf16>
              %23 = linalg.fill ins(%cst : f16) outs(%22 : tensor<?x?xf16>) -> tensor<?x?xf16>
              %24 = affine.for %arg7 = 0 to affine_map<()[s0] -> (512 ceildiv s0)>()[%14] iter_args(%arg8 = %23) -> (tensor<?x?xf16>) {
                %28 = arith.muli %arg6, %12 : index
                %29 = arith.muli %arg7, %14 : index
                %30 = loom.subview %arg0[%28, %29] [%12, %14] [1, 1], reuse : [seq = false, spat = true, temp = true] : memref<4096x512xf16> to memref<?x?xf16, strided<[512, 1], offset: ?>>
                %31 = loom.copy_to_tensor %30, %19 on @L1, interconnect : [@vertical_links], broadcast : [8, 1] : memref<?x?xf16, strided<[512, 1], offset: ?>>, memref<?x?xf16> -> tensor<?x?xf16>
                %32 = arith.muli %15, %13 : index
                %33 = loom.subview %arg1[%29, %32] [%14, %13] [1, 1], reuse : [seq = false, spat = false, temp = true] : memref<512x4096xf16> to memref<?x?xf16, strided<[4096, 1], offset: ?>>
                %34 = loom.copy_to_tensor %33, %17 on @L1, interconnect : [], broadcast : [1, 1] : memref<?x?xf16, strided<[4096, 1], offset: ?>>, memref<?x?xf16> -> tensor<?x?xf16>
                %35 = linalg.matmul ins(%31, %34 : tensor<?x?xf16>, tensor<?x?xf16>) outs(%arg8 : tensor<?x?xf16>) -> tensor<?x?xf16>
                loom.semaphore_give %17 : memref<?x?xf16>
                loom.semaphore_give %19 : memref<?x?xf16>
                affine.yield %35 : tensor<?x?xf16>
              } {loom.iter_type = #loom.iter_type<sequential>}
              %25 = arith.muli %arg6, %12 : index
              %26 = arith.muli %15, %13 : index
              %27 = loom.subview %arg2[%25, %26] [%12, %13] [1, 1], reuse : [seq = false, spat = false, temp = false] : memref<4096x4096xf16> to memref<?x?xf16, strided<[4096, 1], offset: ?>>
              loom.copy_from_tensor %24, %27 on @DRAM : tensor<?x?xf16>, memref<?x?xf16, strided<[4096, 1], offset: ?>>
              loom.semaphore_give %21 : memref<?x?xf16>
            } {loom.iter_type = #loom.iter_type<temporal>}
          } {loom.iter_type = #loom.iter_type<temporal>}
        } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @y}
      } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @x}
      return
    }
  }
  module {
    func.func @matmul__d1i1_d0i1__f01__d_d(%arg0: memref<4096x512xf16>, %arg1: memref<512x4096xf16>, %arg2: memref<4096x4096xf16>) {
      %cst = arith.constant 0.000000e+00 : f16
      %12 = loom.get_symbolic_block_size @constraints::@BM : index
      %13 = loom.get_symbolic_block_size @constraints::@BN : index
      %14 = loom.get_symbolic_block_size @constraints::@BK : index
      affine.parallel (%arg3) = (0) to (8) {
        affine.parallel (%arg4) = (0) to (8) {
          affine.for %arg5 = 0 to affine_map<()[s0, s1] -> (4096 ceildiv s0)>()[%12, %13] {
            affine.for %arg6 = 0 to affine_map<()[s0, s1] -> (4096 ceildiv (s1 * 64))>()[%12, %13] {
              %15 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 8 + d1 + d2 * 64)>(%arg3, %arg4, %arg6)
              %16 = loom.alloc [%14, %13] on @L1 : memref<?x?xf16>
              %17 = loom.semaphore_take %16 : memref<?x?xf16> -> memref<?x?xf16>
              %18 = loom.alloc [%12, %14] on @L1 : memref<?x?xf16>
              %19 = loom.semaphore_take %18 : memref<?x?xf16> -> memref<?x?xf16>
              %20 = loom.alloc [%12, %13] on @L1 : memref<?x?xf16>
              %21 = loom.semaphore_take %20 : memref<?x?xf16> -> memref<?x?xf16>
              %22 = loom.init_tensor %21[%12, %13] : memref<?x?xf16> -> tensor<?x?xf16>
              %23 = linalg.fill ins(%cst : f16) outs(%22 : tensor<?x?xf16>) -> tensor<?x?xf16>
              %24 = affine.for %arg7 = 0 to affine_map<()[s0] -> (512 ceildiv s0)>()[%14] iter_args(%arg8 = %23) -> (tensor<?x?xf16>) {
                %28 = arith.muli %arg5, %12 : index
                %29 = arith.muli %arg7, %14 : index
                %30 = loom.subview %arg0[%28, %29] [%12, %14] [1, 1], reuse : [seq = false, spat = true, temp = true] : memref<4096x512xf16> to memref<?x?xf16, strided<[512, 1], offset: ?>>
                %31 = loom.copy_to_tensor %30, %19 on @L1, interconnect : [], broadcast : [1, 1] : memref<?x?xf16, strided<[512, 1], offset: ?>>, memref<?x?xf16> -> tensor<?x?xf16>
                %32 = arith.muli %15, %13 : index
                %33 = loom.subview %arg1[%29, %32] [%14, %13] [1, 1], reuse : [seq = false, spat = false, temp = true] : memref<512x4096xf16> to memref<?x?xf16, strided<[4096, 1], offset: ?>>
                %34 = loom.copy_to_tensor %33, %17 on @L1, interconnect : [], broadcast : [1, 1] : memref<?x?xf16, strided<[4096, 1], offset: ?>>, memref<?x?xf16> -> tensor<?x?xf16>
                %35 = linalg.matmul ins(%31, %34 : tensor<?x?xf16>, tensor<?x?xf16>) outs(%arg8 : tensor<?x?xf16>) -> tensor<?x?xf16>
                loom.semaphore_give %17 : memref<?x?xf16>
                loom.semaphore_give %19 : memref<?x?xf16>
                affine.yield %35 : tensor<?x?xf16>
              } {loom.iter_type = #loom.iter_type<sequential>}
              %25 = arith.muli %arg5, %12 : index
              %26 = arith.muli %15, %13 : index
              %27 = loom.subview %arg2[%25, %26] [%12, %13] [1, 1], reuse : [seq = false, spat = false, temp = false] : memref<4096x4096xf16> to memref<?x?xf16, strided<[4096, 1], offset: ?>>
              loom.copy_from_tensor %24, %27 on @DRAM : tensor<?x?xf16>, memref<?x?xf16, strided<[4096, 1], offset: ?>>
              loom.semaphore_give %21 : memref<?x?xf16>
            } {loom.iter_type = #loom.iter_type<temporal>}
          } {loom.iter_type = #loom.iter_type<temporal>}
        } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @x}
      } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @y}
      return
    }
  }
  module {
    func.func @matmul__d1i1_d0i1__f01__a_d(%arg0: memref<4096x512xf16>, %arg1: memref<512x4096xf16>, %arg2: memref<4096x4096xf16>) {
      %cst = arith.constant 0.000000e+00 : f16
      %12 = loom.get_symbolic_block_size @constraints::@BM : index
      %13 = loom.get_symbolic_block_size @constraints::@BN : index
      %14 = loom.get_symbolic_block_size @constraints::@BK : index
      affine.parallel (%arg3) = (0) to (8) {
        affine.parallel (%arg4) = (0) to (8) {
          affine.for %arg5 = 0 to affine_map<()[s0, s1] -> (4096 ceildiv s0)>()[%12, %13] {
            affine.for %arg6 = 0 to affine_map<()[s0, s1] -> (4096 ceildiv (s1 * 64))>()[%12, %13] {
              %15 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 8 + d1 + d2 * 64)>(%arg3, %arg4, %arg6)
              %16 = loom.alloc [%14, %13] on @L1 : memref<?x?xf16>
              %17 = loom.semaphore_take %16 : memref<?x?xf16> -> memref<?x?xf16>
              %18 = loom.alloc [%12, %14] on @L1 : memref<?x?xf16>
              %19 = loom.semaphore_take %18 : memref<?x?xf16> -> memref<?x?xf16>
              %20 = loom.alloc [%12, %13] on @L1 : memref<?x?xf16>
              %21 = loom.semaphore_take %20 : memref<?x?xf16> -> memref<?x?xf16>
              %22 = loom.init_tensor %21[%12, %13] : memref<?x?xf16> -> tensor<?x?xf16>
              %23 = linalg.fill ins(%cst : f16) outs(%22 : tensor<?x?xf16>) -> tensor<?x?xf16>
              %24 = affine.for %arg7 = 0 to affine_map<()[s0] -> (512 ceildiv s0)>()[%14] iter_args(%arg8 = %23) -> (tensor<?x?xf16>) {
                %28 = arith.muli %arg5, %12 : index
                %29 = arith.muli %arg7, %14 : index
                %30 = loom.subview %arg0[%28, %29] [%12, %14] [1, 1], reuse : [seq = false, spat = true, temp = true] : memref<4096x512xf16> to memref<?x?xf16, strided<[512, 1], offset: ?>>
                %31 = loom.copy_to_tensor %30, %19 on @L1, interconnect : [@horizontal_links, @vertical_links], broadcast : [8, 8] : memref<?x?xf16, strided<[512, 1], offset: ?>>, memref<?x?xf16> -> tensor<?x?xf16>
                %32 = arith.muli %15, %13 : index
                %33 = loom.subview %arg1[%29, %32] [%14, %13] [1, 1], reuse : [seq = false, spat = false, temp = true] : memref<512x4096xf16> to memref<?x?xf16, strided<[4096, 1], offset: ?>>
                %34 = loom.copy_to_tensor %33, %17 on @L1, interconnect : [], broadcast : [1, 1] : memref<?x?xf16, strided<[4096, 1], offset: ?>>, memref<?x?xf16> -> tensor<?x?xf16>
                %35 = linalg.matmul ins(%31, %34 : tensor<?x?xf16>, tensor<?x?xf16>) outs(%arg8 : tensor<?x?xf16>) -> tensor<?x?xf16>
                loom.semaphore_give %17 : memref<?x?xf16>
                loom.semaphore_give %19 : memref<?x?xf16>
                affine.yield %35 : tensor<?x?xf16>
              } {loom.iter_type = #loom.iter_type<sequential>}
              %25 = arith.muli %arg5, %12 : index
              %26 = arith.muli %15, %13 : index
              %27 = loom.subview %arg2[%25, %26] [%12, %13] [1, 1], reuse : [seq = false, spat = false, temp = false] : memref<4096x4096xf16> to memref<?x?xf16, strided<[4096, 1], offset: ?>>
              loom.copy_from_tensor %24, %27 on @DRAM : tensor<?x?xf16>, memref<?x?xf16, strided<[4096, 1], offset: ?>>
              loom.semaphore_give %21 : memref<?x?xf16>
            } {loom.iter_type = #loom.iter_type<temporal>}
          } {loom.iter_type = #loom.iter_type<temporal>}
        } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @x}
      } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @y}
      return
    }
  }
  module {
    func.func @matmul__d1i1_d0i1__f01__h_d(%arg0: memref<4096x512xf16>, %arg1: memref<512x4096xf16>, %arg2: memref<4096x4096xf16>) {
      %cst = arith.constant 0.000000e+00 : f16
      %12 = loom.get_symbolic_block_size @constraints::@BM : index
      %13 = loom.get_symbolic_block_size @constraints::@BN : index
      %14 = loom.get_symbolic_block_size @constraints::@BK : index
      affine.parallel (%arg3) = (0) to (8) {
        affine.parallel (%arg4) = (0) to (8) {
          affine.for %arg5 = 0 to affine_map<()[s0, s1] -> (4096 ceildiv s0)>()[%12, %13] {
            affine.for %arg6 = 0 to affine_map<()[s0, s1] -> (4096 ceildiv (s1 * 64))>()[%12, %13] {
              %15 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 8 + d1 + d2 * 64)>(%arg3, %arg4, %arg6)
              %16 = loom.alloc [%14, %13] on @L1 : memref<?x?xf16>
              %17 = loom.semaphore_take %16 : memref<?x?xf16> -> memref<?x?xf16>
              %18 = loom.alloc [%12, %14] on @L1 : memref<?x?xf16>
              %19 = loom.semaphore_take %18 : memref<?x?xf16> -> memref<?x?xf16>
              %20 = loom.alloc [%12, %13] on @L1 : memref<?x?xf16>
              %21 = loom.semaphore_take %20 : memref<?x?xf16> -> memref<?x?xf16>
              %22 = loom.init_tensor %21[%12, %13] : memref<?x?xf16> -> tensor<?x?xf16>
              %23 = linalg.fill ins(%cst : f16) outs(%22 : tensor<?x?xf16>) -> tensor<?x?xf16>
              %24 = affine.for %arg7 = 0 to affine_map<()[s0] -> (512 ceildiv s0)>()[%14] iter_args(%arg8 = %23) -> (tensor<?x?xf16>) {
                %28 = arith.muli %arg5, %12 : index
                %29 = arith.muli %arg7, %14 : index
                %30 = loom.subview %arg0[%28, %29] [%12, %14] [1, 1], reuse : [seq = false, spat = true, temp = true] : memref<4096x512xf16> to memref<?x?xf16, strided<[512, 1], offset: ?>>
                %31 = loom.copy_to_tensor %30, %19 on @L1, interconnect : [@horizontal_links], broadcast : [1, 8] : memref<?x?xf16, strided<[512, 1], offset: ?>>, memref<?x?xf16> -> tensor<?x?xf16>
                %32 = arith.muli %15, %13 : index
                %33 = loom.subview %arg1[%29, %32] [%14, %13] [1, 1], reuse : [seq = false, spat = false, temp = true] : memref<512x4096xf16> to memref<?x?xf16, strided<[4096, 1], offset: ?>>
                %34 = loom.copy_to_tensor %33, %17 on @L1, interconnect : [], broadcast : [1, 1] : memref<?x?xf16, strided<[4096, 1], offset: ?>>, memref<?x?xf16> -> tensor<?x?xf16>
                %35 = linalg.matmul ins(%31, %34 : tensor<?x?xf16>, tensor<?x?xf16>) outs(%arg8 : tensor<?x?xf16>) -> tensor<?x?xf16>
                loom.semaphore_give %17 : memref<?x?xf16>
                loom.semaphore_give %19 : memref<?x?xf16>
                affine.yield %35 : tensor<?x?xf16>
              } {loom.iter_type = #loom.iter_type<sequential>}
              %25 = arith.muli %arg5, %12 : index
              %26 = arith.muli %15, %13 : index
              %27 = loom.subview %arg2[%25, %26] [%12, %13] [1, 1], reuse : [seq = false, spat = false, temp = false] : memref<4096x4096xf16> to memref<?x?xf16, strided<[4096, 1], offset: ?>>
              loom.copy_from_tensor %24, %27 on @DRAM : tensor<?x?xf16>, memref<?x?xf16, strided<[4096, 1], offset: ?>>
              loom.semaphore_give %21 : memref<?x?xf16>
            } {loom.iter_type = #loom.iter_type<temporal>}
          } {loom.iter_type = #loom.iter_type<temporal>}
        } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @x}
      } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @y}
      return
    }
  }
  module {
    func.func @matmul__d1i1_d0i1__f01__v_d(%arg0: memref<4096x512xf16>, %arg1: memref<512x4096xf16>, %arg2: memref<4096x4096xf16>) {
      %cst = arith.constant 0.000000e+00 : f16
      %12 = loom.get_symbolic_block_size @constraints::@BM : index
      %13 = loom.get_symbolic_block_size @constraints::@BN : index
      %14 = loom.get_symbolic_block_size @constraints::@BK : index
      affine.parallel (%arg3) = (0) to (8) {
        affine.parallel (%arg4) = (0) to (8) {
          affine.for %arg5 = 0 to affine_map<()[s0, s1] -> (4096 ceildiv s0)>()[%12, %13] {
            affine.for %arg6 = 0 to affine_map<()[s0, s1] -> (4096 ceildiv (s1 * 64))>()[%12, %13] {
              %15 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 8 + d1 + d2 * 64)>(%arg3, %arg4, %arg6)
              %16 = loom.alloc [%14, %13] on @L1 : memref<?x?xf16>
              %17 = loom.semaphore_take %16 : memref<?x?xf16> -> memref<?x?xf16>
              %18 = loom.alloc [%12, %14] on @L1 : memref<?x?xf16>
              %19 = loom.semaphore_take %18 : memref<?x?xf16> -> memref<?x?xf16>
              %20 = loom.alloc [%12, %13] on @L1 : memref<?x?xf16>
              %21 = loom.semaphore_take %20 : memref<?x?xf16> -> memref<?x?xf16>
              %22 = loom.init_tensor %21[%12, %13] : memref<?x?xf16> -> tensor<?x?xf16>
              %23 = linalg.fill ins(%cst : f16) outs(%22 : tensor<?x?xf16>) -> tensor<?x?xf16>
              %24 = affine.for %arg7 = 0 to affine_map<()[s0] -> (512 ceildiv s0)>()[%14] iter_args(%arg8 = %23) -> (tensor<?x?xf16>) {
                %28 = arith.muli %arg5, %12 : index
                %29 = arith.muli %arg7, %14 : index
                %30 = loom.subview %arg0[%28, %29] [%12, %14] [1, 1], reuse : [seq = false, spat = true, temp = true] : memref<4096x512xf16> to memref<?x?xf16, strided<[512, 1], offset: ?>>
                %31 = loom.copy_to_tensor %30, %19 on @L1, interconnect : [@vertical_links], broadcast : [8, 1] : memref<?x?xf16, strided<[512, 1], offset: ?>>, memref<?x?xf16> -> tensor<?x?xf16>
                %32 = arith.muli %15, %13 : index
                %33 = loom.subview %arg1[%29, %32] [%14, %13] [1, 1], reuse : [seq = false, spat = false, temp = true] : memref<512x4096xf16> to memref<?x?xf16, strided<[4096, 1], offset: ?>>
                %34 = loom.copy_to_tensor %33, %17 on @L1, interconnect : [], broadcast : [1, 1] : memref<?x?xf16, strided<[4096, 1], offset: ?>>, memref<?x?xf16> -> tensor<?x?xf16>
                %35 = linalg.matmul ins(%31, %34 : tensor<?x?xf16>, tensor<?x?xf16>) outs(%arg8 : tensor<?x?xf16>) -> tensor<?x?xf16>
                loom.semaphore_give %17 : memref<?x?xf16>
                loom.semaphore_give %19 : memref<?x?xf16>
                affine.yield %35 : tensor<?x?xf16>
              } {loom.iter_type = #loom.iter_type<sequential>}
              %25 = arith.muli %arg5, %12 : index
              %26 = arith.muli %15, %13 : index
              %27 = loom.subview %arg2[%25, %26] [%12, %13] [1, 1], reuse : [seq = false, spat = false, temp = false] : memref<4096x4096xf16> to memref<?x?xf16, strided<[4096, 1], offset: ?>>
              loom.copy_from_tensor %24, %27 on @DRAM : tensor<?x?xf16>, memref<?x?xf16, strided<[4096, 1], offset: ?>>
              loom.semaphore_give %21 : memref<?x?xf16>
            } {loom.iter_type = #loom.iter_type<temporal>}
          } {loom.iter_type = #loom.iter_type<temporal>}
        } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @x}
      } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @y}
      return
    }
  }
  module {
    func.func @matmul__d1i1_d0i1__f10__d_d(%arg0: memref<4096x512xf16>, %arg1: memref<512x4096xf16>, %arg2: memref<4096x4096xf16>) {
      %cst = arith.constant 0.000000e+00 : f16
      %12 = loom.get_symbolic_block_size @constraints::@BM : index
      %13 = loom.get_symbolic_block_size @constraints::@BN : index
      %14 = loom.get_symbolic_block_size @constraints::@BK : index
      affine.parallel (%arg3) = (0) to (8) {
        affine.parallel (%arg4) = (0) to (8) {
          affine.for %arg5 = 0 to affine_map<()[s0, s1] -> (4096 ceildiv (s1 * 64))>()[%12, %13] {
            affine.for %arg6 = 0 to affine_map<()[s0, s1] -> (4096 ceildiv s0)>()[%12, %13] {
              %15 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 8 + d1 + d2 * 64)>(%arg3, %arg4, %arg5)
              %16 = loom.alloc [%14, %13] on @L1 : memref<?x?xf16>
              %17 = loom.semaphore_take %16 : memref<?x?xf16> -> memref<?x?xf16>
              %18 = loom.alloc [%12, %14] on @L1 : memref<?x?xf16>
              %19 = loom.semaphore_take %18 : memref<?x?xf16> -> memref<?x?xf16>
              %20 = loom.alloc [%12, %13] on @L1 : memref<?x?xf16>
              %21 = loom.semaphore_take %20 : memref<?x?xf16> -> memref<?x?xf16>
              %22 = loom.init_tensor %21[%12, %13] : memref<?x?xf16> -> tensor<?x?xf16>
              %23 = linalg.fill ins(%cst : f16) outs(%22 : tensor<?x?xf16>) -> tensor<?x?xf16>
              %24 = affine.for %arg7 = 0 to affine_map<()[s0] -> (512 ceildiv s0)>()[%14] iter_args(%arg8 = %23) -> (tensor<?x?xf16>) {
                %28 = arith.muli %arg6, %12 : index
                %29 = arith.muli %arg7, %14 : index
                %30 = loom.subview %arg0[%28, %29] [%12, %14] [1, 1], reuse : [seq = false, spat = true, temp = true] : memref<4096x512xf16> to memref<?x?xf16, strided<[512, 1], offset: ?>>
                %31 = loom.copy_to_tensor %30, %19 on @L1, interconnect : [], broadcast : [1, 1] : memref<?x?xf16, strided<[512, 1], offset: ?>>, memref<?x?xf16> -> tensor<?x?xf16>
                %32 = arith.muli %15, %13 : index
                %33 = loom.subview %arg1[%29, %32] [%14, %13] [1, 1], reuse : [seq = false, spat = false, temp = true] : memref<512x4096xf16> to memref<?x?xf16, strided<[4096, 1], offset: ?>>
                %34 = loom.copy_to_tensor %33, %17 on @L1, interconnect : [], broadcast : [1, 1] : memref<?x?xf16, strided<[4096, 1], offset: ?>>, memref<?x?xf16> -> tensor<?x?xf16>
                %35 = linalg.matmul ins(%31, %34 : tensor<?x?xf16>, tensor<?x?xf16>) outs(%arg8 : tensor<?x?xf16>) -> tensor<?x?xf16>
                loom.semaphore_give %17 : memref<?x?xf16>
                loom.semaphore_give %19 : memref<?x?xf16>
                affine.yield %35 : tensor<?x?xf16>
              } {loom.iter_type = #loom.iter_type<sequential>}
              %25 = arith.muli %arg6, %12 : index
              %26 = arith.muli %15, %13 : index
              %27 = loom.subview %arg2[%25, %26] [%12, %13] [1, 1], reuse : [seq = false, spat = false, temp = false] : memref<4096x4096xf16> to memref<?x?xf16, strided<[4096, 1], offset: ?>>
              loom.copy_from_tensor %24, %27 on @DRAM : tensor<?x?xf16>, memref<?x?xf16, strided<[4096, 1], offset: ?>>
              loom.semaphore_give %21 : memref<?x?xf16>
            } {loom.iter_type = #loom.iter_type<temporal>}
          } {loom.iter_type = #loom.iter_type<temporal>}
        } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @x}
      } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @y}
      return
    }
  }
  module {
    func.func @matmul__d1i1_d0i1__f10__a_d(%arg0: memref<4096x512xf16>, %arg1: memref<512x4096xf16>, %arg2: memref<4096x4096xf16>) {
      %cst = arith.constant 0.000000e+00 : f16
      %12 = loom.get_symbolic_block_size @constraints::@BM : index
      %13 = loom.get_symbolic_block_size @constraints::@BN : index
      %14 = loom.get_symbolic_block_size @constraints::@BK : index
      affine.parallel (%arg3) = (0) to (8) {
        affine.parallel (%arg4) = (0) to (8) {
          affine.for %arg5 = 0 to affine_map<()[s0, s1] -> (4096 ceildiv (s1 * 64))>()[%12, %13] {
            affine.for %arg6 = 0 to affine_map<()[s0, s1] -> (4096 ceildiv s0)>()[%12, %13] {
              %15 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 8 + d1 + d2 * 64)>(%arg3, %arg4, %arg5)
              %16 = loom.alloc [%14, %13] on @L1 : memref<?x?xf16>
              %17 = loom.semaphore_take %16 : memref<?x?xf16> -> memref<?x?xf16>
              %18 = loom.alloc [%12, %14] on @L1 : memref<?x?xf16>
              %19 = loom.semaphore_take %18 : memref<?x?xf16> -> memref<?x?xf16>
              %20 = loom.alloc [%12, %13] on @L1 : memref<?x?xf16>
              %21 = loom.semaphore_take %20 : memref<?x?xf16> -> memref<?x?xf16>
              %22 = loom.init_tensor %21[%12, %13] : memref<?x?xf16> -> tensor<?x?xf16>
              %23 = linalg.fill ins(%cst : f16) outs(%22 : tensor<?x?xf16>) -> tensor<?x?xf16>
              %24 = affine.for %arg7 = 0 to affine_map<()[s0] -> (512 ceildiv s0)>()[%14] iter_args(%arg8 = %23) -> (tensor<?x?xf16>) {
                %28 = arith.muli %arg6, %12 : index
                %29 = arith.muli %arg7, %14 : index
                %30 = loom.subview %arg0[%28, %29] [%12, %14] [1, 1], reuse : [seq = false, spat = true, temp = true] : memref<4096x512xf16> to memref<?x?xf16, strided<[512, 1], offset: ?>>
                %31 = loom.copy_to_tensor %30, %19 on @L1, interconnect : [@horizontal_links, @vertical_links], broadcast : [8, 8] : memref<?x?xf16, strided<[512, 1], offset: ?>>, memref<?x?xf16> -> tensor<?x?xf16>
                %32 = arith.muli %15, %13 : index
                %33 = loom.subview %arg1[%29, %32] [%14, %13] [1, 1], reuse : [seq = false, spat = false, temp = true] : memref<512x4096xf16> to memref<?x?xf16, strided<[4096, 1], offset: ?>>
                %34 = loom.copy_to_tensor %33, %17 on @L1, interconnect : [], broadcast : [1, 1] : memref<?x?xf16, strided<[4096, 1], offset: ?>>, memref<?x?xf16> -> tensor<?x?xf16>
                %35 = linalg.matmul ins(%31, %34 : tensor<?x?xf16>, tensor<?x?xf16>) outs(%arg8 : tensor<?x?xf16>) -> tensor<?x?xf16>
                loom.semaphore_give %17 : memref<?x?xf16>
                loom.semaphore_give %19 : memref<?x?xf16>
                affine.yield %35 : tensor<?x?xf16>
              } {loom.iter_type = #loom.iter_type<sequential>}
              %25 = arith.muli %arg6, %12 : index
              %26 = arith.muli %15, %13 : index
              %27 = loom.subview %arg2[%25, %26] [%12, %13] [1, 1], reuse : [seq = false, spat = false, temp = false] : memref<4096x4096xf16> to memref<?x?xf16, strided<[4096, 1], offset: ?>>
              loom.copy_from_tensor %24, %27 on @DRAM : tensor<?x?xf16>, memref<?x?xf16, strided<[4096, 1], offset: ?>>
              loom.semaphore_give %21 : memref<?x?xf16>
            } {loom.iter_type = #loom.iter_type<temporal>}
          } {loom.iter_type = #loom.iter_type<temporal>}
        } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @x}
      } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @y}
      return
    }
  }
  module {
    func.func @matmul__d1i1_d0i1__f10__h_d(%arg0: memref<4096x512xf16>, %arg1: memref<512x4096xf16>, %arg2: memref<4096x4096xf16>) {
      %cst = arith.constant 0.000000e+00 : f16
      %12 = loom.get_symbolic_block_size @constraints::@BM : index
      %13 = loom.get_symbolic_block_size @constraints::@BN : index
      %14 = loom.get_symbolic_block_size @constraints::@BK : index
      affine.parallel (%arg3) = (0) to (8) {
        affine.parallel (%arg4) = (0) to (8) {
          affine.for %arg5 = 0 to affine_map<()[s0, s1] -> (4096 ceildiv (s1 * 64))>()[%12, %13] {
            affine.for %arg6 = 0 to affine_map<()[s0, s1] -> (4096 ceildiv s0)>()[%12, %13] {
              %15 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 8 + d1 + d2 * 64)>(%arg3, %arg4, %arg5)
              %16 = loom.alloc [%14, %13] on @L1 : memref<?x?xf16>
              %17 = loom.semaphore_take %16 : memref<?x?xf16> -> memref<?x?xf16>
              %18 = loom.alloc [%12, %14] on @L1 : memref<?x?xf16>
              %19 = loom.semaphore_take %18 : memref<?x?xf16> -> memref<?x?xf16>
              %20 = loom.alloc [%12, %13] on @L1 : memref<?x?xf16>
              %21 = loom.semaphore_take %20 : memref<?x?xf16> -> memref<?x?xf16>
              %22 = loom.init_tensor %21[%12, %13] : memref<?x?xf16> -> tensor<?x?xf16>
              %23 = linalg.fill ins(%cst : f16) outs(%22 : tensor<?x?xf16>) -> tensor<?x?xf16>
              %24 = affine.for %arg7 = 0 to affine_map<()[s0] -> (512 ceildiv s0)>()[%14] iter_args(%arg8 = %23) -> (tensor<?x?xf16>) {
                %28 = arith.muli %arg6, %12 : index
                %29 = arith.muli %arg7, %14 : index
                %30 = loom.subview %arg0[%28, %29] [%12, %14] [1, 1], reuse : [seq = false, spat = true, temp = true] : memref<4096x512xf16> to memref<?x?xf16, strided<[512, 1], offset: ?>>
                %31 = loom.copy_to_tensor %30, %19 on @L1, interconnect : [@horizontal_links], broadcast : [1, 8] : memref<?x?xf16, strided<[512, 1], offset: ?>>, memref<?x?xf16> -> tensor<?x?xf16>
                %32 = arith.muli %15, %13 : index
                %33 = loom.subview %arg1[%29, %32] [%14, %13] [1, 1], reuse : [seq = false, spat = false, temp = true] : memref<512x4096xf16> to memref<?x?xf16, strided<[4096, 1], offset: ?>>
                %34 = loom.copy_to_tensor %33, %17 on @L1, interconnect : [], broadcast : [1, 1] : memref<?x?xf16, strided<[4096, 1], offset: ?>>, memref<?x?xf16> -> tensor<?x?xf16>
                %35 = linalg.matmul ins(%31, %34 : tensor<?x?xf16>, tensor<?x?xf16>) outs(%arg8 : tensor<?x?xf16>) -> tensor<?x?xf16>
                loom.semaphore_give %17 : memref<?x?xf16>
                loom.semaphore_give %19 : memref<?x?xf16>
                affine.yield %35 : tensor<?x?xf16>
              } {loom.iter_type = #loom.iter_type<sequential>}
              %25 = arith.muli %arg6, %12 : index
              %26 = arith.muli %15, %13 : index
              %27 = loom.subview %arg2[%25, %26] [%12, %13] [1, 1], reuse : [seq = false, spat = false, temp = false] : memref<4096x4096xf16> to memref<?x?xf16, strided<[4096, 1], offset: ?>>
              loom.copy_from_tensor %24, %27 on @DRAM : tensor<?x?xf16>, memref<?x?xf16, strided<[4096, 1], offset: ?>>
              loom.semaphore_give %21 : memref<?x?xf16>
            } {loom.iter_type = #loom.iter_type<temporal>}
          } {loom.iter_type = #loom.iter_type<temporal>}
        } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @x}
      } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @y}
      return
    }
  }
  module {
    func.func @matmul__d1i1_d0i1__f10__v_d(%arg0: memref<4096x512xf16>, %arg1: memref<512x4096xf16>, %arg2: memref<4096x4096xf16>) {
      %cst = arith.constant 0.000000e+00 : f16
      %12 = loom.get_symbolic_block_size @constraints::@BM : index
      %13 = loom.get_symbolic_block_size @constraints::@BN : index
      %14 = loom.get_symbolic_block_size @constraints::@BK : index
      affine.parallel (%arg3) = (0) to (8) {
        affine.parallel (%arg4) = (0) to (8) {
          affine.for %arg5 = 0 to affine_map<()[s0, s1] -> (4096 ceildiv (s1 * 64))>()[%12, %13] {
            affine.for %arg6 = 0 to affine_map<()[s0, s1] -> (4096 ceildiv s0)>()[%12, %13] {
              %15 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 8 + d1 + d2 * 64)>(%arg3, %arg4, %arg5)
              %16 = loom.alloc [%14, %13] on @L1 : memref<?x?xf16>
              %17 = loom.semaphore_take %16 : memref<?x?xf16> -> memref<?x?xf16>
              %18 = loom.alloc [%12, %14] on @L1 : memref<?x?xf16>
              %19 = loom.semaphore_take %18 : memref<?x?xf16> -> memref<?x?xf16>
              %20 = loom.alloc [%12, %13] on @L1 : memref<?x?xf16>
              %21 = loom.semaphore_take %20 : memref<?x?xf16> -> memref<?x?xf16>
              %22 = loom.init_tensor %21[%12, %13] : memref<?x?xf16> -> tensor<?x?xf16>
              %23 = linalg.fill ins(%cst : f16) outs(%22 : tensor<?x?xf16>) -> tensor<?x?xf16>
              %24 = affine.for %arg7 = 0 to affine_map<()[s0] -> (512 ceildiv s0)>()[%14] iter_args(%arg8 = %23) -> (tensor<?x?xf16>) {
                %28 = arith.muli %arg6, %12 : index
                %29 = arith.muli %arg7, %14 : index
                %30 = loom.subview %arg0[%28, %29] [%12, %14] [1, 1], reuse : [seq = false, spat = true, temp = true] : memref<4096x512xf16> to memref<?x?xf16, strided<[512, 1], offset: ?>>
                %31 = loom.copy_to_tensor %30, %19 on @L1, interconnect : [@vertical_links], broadcast : [8, 1] : memref<?x?xf16, strided<[512, 1], offset: ?>>, memref<?x?xf16> -> tensor<?x?xf16>
                %32 = arith.muli %15, %13 : index
                %33 = loom.subview %arg1[%29, %32] [%14, %13] [1, 1], reuse : [seq = false, spat = false, temp = true] : memref<512x4096xf16> to memref<?x?xf16, strided<[4096, 1], offset: ?>>
                %34 = loom.copy_to_tensor %33, %17 on @L1, interconnect : [], broadcast : [1, 1] : memref<?x?xf16, strided<[4096, 1], offset: ?>>, memref<?x?xf16> -> tensor<?x?xf16>
                %35 = linalg.matmul ins(%31, %34 : tensor<?x?xf16>, tensor<?x?xf16>) outs(%arg8 : tensor<?x?xf16>) -> tensor<?x?xf16>
                loom.semaphore_give %17 : memref<?x?xf16>
                loom.semaphore_give %19 : memref<?x?xf16>
                affine.yield %35 : tensor<?x?xf16>
              } {loom.iter_type = #loom.iter_type<sequential>}
              %25 = arith.muli %arg6, %12 : index
              %26 = arith.muli %15, %13 : index
              %27 = loom.subview %arg2[%25, %26] [%12, %13] [1, 1], reuse : [seq = false, spat = false, temp = false] : memref<4096x4096xf16> to memref<?x?xf16, strided<[4096, 1], offset: ?>>
              loom.copy_from_tensor %24, %27 on @DRAM : tensor<?x?xf16>, memref<?x?xf16, strided<[4096, 1], offset: ?>>
              loom.semaphore_give %21 : memref<?x?xf16>
            } {loom.iter_type = #loom.iter_type<temporal>}
          } {loom.iter_type = #loom.iter_type<temporal>}
        } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @x}
      } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @y}
      return
    }
  }
}
