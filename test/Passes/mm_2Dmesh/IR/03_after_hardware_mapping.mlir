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
  module attributes {loom.pass_name = "EnumerateHWMapping"} {
    loom.constraint_space @constraints {
      %12 = loom.symbolic_var "BM" : index
      %13 = loom.symbolic_var "BN" : index
      %14 = loom.symbolic_var "BK" : index
      loom.range %12[32, 1024]
      loom.range %13[32, 1024]
      loom.range %14[32, 1024]
      loom.align %12 by 32
      loom.align %13 by 32
      loom.align %14 by 32
      loom.polynomial_constraint(%12, %13, %14) {monomials = [{coeff = -1 : i64, vars = [0, 1, 2]}], upper_bound = -262144 : i64}
      loom.polynomial_constraint(%12, %13, %14) {monomials = [{coeff = 2 : i64, vars = [0, 1]}, {coeff = 2 : i64, vars = [0, 2]}, {coeff = 2 : i64, vars = [2, 1]}], upper_bound = 1499136 : i64}
      loom.polynomial_constraint(%12) {monomials = [{coeff = 64 : i64, vars = [0]}], upper_bound = 1024 : i64}
    }
    func.func @matmul__d0i0_d1i0__f01(%arg0: memref<4096x512xf16>, %arg1: memref<512x4096xf16>, %arg2: memref<4096x4096xf16>) {
      %cst = arith.constant 0.000000e+00 : f16
      %12 = loom.get_symbolic_block_size @constraints::@BM : index
      %13 = loom.get_symbolic_block_size @constraints::@BN : index
      %14 = loom.get_symbolic_block_size @constraints::@BK : index
      affine.parallel (%arg3) = (0) to (8) {
        affine.parallel (%arg4) = (0) to (8) {
          affine.for %arg5 = 0 to affine_map<()[s0, s1] -> (((4096 ceildiv s0) ceildiv 8) ceildiv 8)>()[%12, %13] {
            affine.for %arg6 = 0 to affine_map<()[s0, s1] -> (4096 ceildiv s1)>()[%12, %13] {
              %15 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 8 + d1 + d2 * 64)>(%arg3, %arg4, %arg5)
              %16 = loom.alloc [%12, %13] on @L1 : memref<?x?xf16>
              %17 = loom.init_tensor %16[%12, %13] : memref<?x?xf16> -> tensor<?x?xf16>
              %18 = linalg.fill ins(%cst : f16) outs(%17 : tensor<?x?xf16>) -> tensor<?x?xf16>
              %19 = affine.for %arg7 = 0 to affine_map<()[s0] -> (512 ceildiv s0)>()[%14] iter_args(%arg8 = %18) -> (tensor<?x?xf16>) {
                %23 = arith.muli %15, %12 : index
                %24 = arith.muli %arg7, %14 : index
                %25 = loom.subview %arg0[%23, %24] [%12, %14] [1, 1], reuse : [seq = false, spat = false, temp = false] : memref<4096x512xf16> to memref<?x?xf16, strided<[512, 1], offset: ?>>
                %26 = loom.alloc [%12, %14] on @L1 : memref<?x?xf16>
                %27 = loom.copy_to_tensor %25, %26 on @L1, interconnect : [], broadcast : [1, 1] : memref<?x?xf16, strided<[512, 1], offset: ?>>, memref<?x?xf16> -> tensor<?x?xf16>
                %28 = arith.muli %arg6, %13 : index
                %29 = loom.subview %arg1[%24, %28] [%14, %13] [1, 1], reuse : [seq = false, spat = false, temp = false] : memref<512x4096xf16> to memref<?x?xf16, strided<[4096, 1], offset: ?>>
                %30 = loom.alloc [%14, %13] on @L1 : memref<?x?xf16>
                %31 = loom.copy_to_tensor %29, %30 on @L1, interconnect : [], broadcast : [1, 1] : memref<?x?xf16, strided<[4096, 1], offset: ?>>, memref<?x?xf16> -> tensor<?x?xf16>
                %32 = linalg.matmul ins(%27, %31 : tensor<?x?xf16>, tensor<?x?xf16>) outs(%arg8 : tensor<?x?xf16>) -> tensor<?x?xf16>
                affine.yield %32 : tensor<?x?xf16>
              } {loom.iter_type = #loom.iter_type<sequential>}
              %20 = arith.muli %15, %12 : index
              %21 = arith.muli %arg6, %13 : index
              %22 = loom.subview %arg2[%20, %21] [%12, %13] [1, 1], reuse : [seq = false, spat = false, temp = false] : memref<4096x4096xf16> to memref<?x?xf16, strided<[4096, 1], offset: ?>>
              loom.copy_from_tensor %19, %22 on @DRAM : tensor<?x?xf16>, memref<?x?xf16, strided<[4096, 1], offset: ?>>
            } {loom.iter_type = #loom.iter_type<temporal>}
          } {loom.iter_type = #loom.iter_type<temporal>}
        } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @y}
      } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @x}
      return
    }
  }
  module attributes {loom.pass_name = "EnumerateHWMapping"} {
    loom.constraint_space @constraints {
      %12 = loom.symbolic_var "BM" : index
      %13 = loom.symbolic_var "BN" : index
      %14 = loom.symbolic_var "BK" : index
      loom.range %12[32, 1024]
      loom.range %13[32, 1024]
      loom.range %14[32, 1024]
      loom.align %12 by 32
      loom.align %13 by 32
      loom.align %14 by 32
      loom.polynomial_constraint(%12, %13, %14) {monomials = [{coeff = -1 : i64, vars = [0, 1, 2]}], upper_bound = -262144 : i64}
      loom.polynomial_constraint(%12, %13, %14) {monomials = [{coeff = 2 : i64, vars = [0, 1]}, {coeff = 2 : i64, vars = [0, 2]}, {coeff = 2 : i64, vars = [2, 1]}], upper_bound = 1499136 : i64}
      loom.polynomial_constraint(%12) {monomials = [{coeff = 64 : i64, vars = [0]}], upper_bound = 1024 : i64}
    }
    func.func @matmul__d0i0_d1i0__f10(%arg0: memref<4096x512xf16>, %arg1: memref<512x4096xf16>, %arg2: memref<4096x4096xf16>) {
      %cst = arith.constant 0.000000e+00 : f16
      %12 = loom.get_symbolic_block_size @constraints::@BM : index
      %13 = loom.get_symbolic_block_size @constraints::@BN : index
      %14 = loom.get_symbolic_block_size @constraints::@BK : index
      affine.parallel (%arg3) = (0) to (8) {
        affine.parallel (%arg4) = (0) to (8) {
          affine.for %arg5 = 0 to affine_map<()[s0, s1] -> (4096 ceildiv s1)>()[%12, %13] {
            affine.for %arg6 = 0 to affine_map<()[s0, s1] -> (((4096 ceildiv s0) ceildiv 8) ceildiv 8)>()[%12, %13] {
              %15 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 8 + d1 + d2 * 64)>(%arg3, %arg4, %arg6)
              %16 = loom.alloc [%12, %13] on @L1 : memref<?x?xf16>
              %17 = loom.init_tensor %16[%12, %13] : memref<?x?xf16> -> tensor<?x?xf16>
              %18 = linalg.fill ins(%cst : f16) outs(%17 : tensor<?x?xf16>) -> tensor<?x?xf16>
              %19 = affine.for %arg7 = 0 to affine_map<()[s0] -> (512 ceildiv s0)>()[%14] iter_args(%arg8 = %18) -> (tensor<?x?xf16>) {
                %23 = arith.muli %15, %12 : index
                %24 = arith.muli %arg7, %14 : index
                %25 = loom.subview %arg0[%23, %24] [%12, %14] [1, 1], reuse : [seq = false, spat = false, temp = false] : memref<4096x512xf16> to memref<?x?xf16, strided<[512, 1], offset: ?>>
                %26 = loom.alloc [%12, %14] on @L1 : memref<?x?xf16>
                %27 = loom.copy_to_tensor %25, %26 on @L1, interconnect : [], broadcast : [1, 1] : memref<?x?xf16, strided<[512, 1], offset: ?>>, memref<?x?xf16> -> tensor<?x?xf16>
                %28 = arith.muli %arg5, %13 : index
                %29 = loom.subview %arg1[%24, %28] [%14, %13] [1, 1], reuse : [seq = false, spat = false, temp = false] : memref<512x4096xf16> to memref<?x?xf16, strided<[4096, 1], offset: ?>>
                %30 = loom.alloc [%14, %13] on @L1 : memref<?x?xf16>
                %31 = loom.copy_to_tensor %29, %30 on @L1, interconnect : [], broadcast : [1, 1] : memref<?x?xf16, strided<[4096, 1], offset: ?>>, memref<?x?xf16> -> tensor<?x?xf16>
                %32 = linalg.matmul ins(%27, %31 : tensor<?x?xf16>, tensor<?x?xf16>) outs(%arg8 : tensor<?x?xf16>) -> tensor<?x?xf16>
                affine.yield %32 : tensor<?x?xf16>
              } {loom.iter_type = #loom.iter_type<sequential>}
              %20 = arith.muli %15, %12 : index
              %21 = arith.muli %arg5, %13 : index
              %22 = loom.subview %arg2[%20, %21] [%12, %13] [1, 1], reuse : [seq = false, spat = false, temp = false] : memref<4096x4096xf16> to memref<?x?xf16, strided<[4096, 1], offset: ?>>
              loom.copy_from_tensor %19, %22 on @DRAM : tensor<?x?xf16>, memref<?x?xf16, strided<[4096, 1], offset: ?>>
            } {loom.iter_type = #loom.iter_type<temporal>}
          } {loom.iter_type = #loom.iter_type<temporal>}
        } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @y}
      } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @x}
      return
    }
  }
  module attributes {loom.pass_name = "EnumerateHWMapping"} {
    loom.constraint_space @constraints {
      %12 = loom.symbolic_var "BM" : index
      %13 = loom.symbolic_var "BN" : index
      %14 = loom.symbolic_var "BK" : index
      loom.range %12[32, 1024]
      loom.range %13[32, 1024]
      loom.range %14[32, 1024]
      loom.align %12 by 32
      loom.align %13 by 32
      loom.align %14 by 32
      loom.polynomial_constraint(%12, %13, %14) {monomials = [{coeff = -1 : i64, vars = [0, 1, 2]}], upper_bound = -262144 : i64}
      loom.polynomial_constraint(%12, %13, %14) {monomials = [{coeff = 2 : i64, vars = [0, 1]}, {coeff = 2 : i64, vars = [0, 2]}, {coeff = 2 : i64, vars = [2, 1]}], upper_bound = 1499136 : i64}
      loom.polynomial_constraint(%12) {monomials = [{coeff = 8 : i64, vars = [0]}], upper_bound = 1024 : i64}
      loom.polynomial_constraint(%13) {monomials = [{coeff = 8 : i64, vars = [0]}], upper_bound = 1024 : i64}
    }
    func.func @matmul__d0i0_d1i1__f01(%arg0: memref<4096x512xf16>, %arg1: memref<512x4096xf16>, %arg2: memref<4096x4096xf16>) {
      %cst = arith.constant 0.000000e+00 : f16
      %12 = loom.get_symbolic_block_size @constraints::@BM : index
      %13 = loom.get_symbolic_block_size @constraints::@BN : index
      %14 = loom.get_symbolic_block_size @constraints::@BK : index
      affine.parallel (%arg3) = (0) to (8) {
        affine.parallel (%arg4) = (0) to (8) {
          affine.for %arg5 = 0 to affine_map<()[s0, s1] -> ((4096 ceildiv s0) ceildiv 8)>()[%12, %13] {
            affine.for %arg6 = 0 to affine_map<()[s0, s1] -> ((4096 ceildiv s1) ceildiv 8)>()[%12, %13] {
              %15 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 8)>(%arg3, %arg5)
              %16 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 8)>(%arg4, %arg6)
              %17 = loom.alloc [%12, %13] on @L1 : memref<?x?xf16>
              %18 = loom.init_tensor %17[%12, %13] : memref<?x?xf16> -> tensor<?x?xf16>
              %19 = linalg.fill ins(%cst : f16) outs(%18 : tensor<?x?xf16>) -> tensor<?x?xf16>
              %20 = affine.for %arg7 = 0 to affine_map<()[s0] -> (512 ceildiv s0)>()[%14] iter_args(%arg8 = %19) -> (tensor<?x?xf16>) {
                %24 = arith.muli %15, %12 : index
                %25 = arith.muli %arg7, %14 : index
                %26 = loom.subview %arg0[%24, %25] [%12, %14] [1, 1], reuse : [seq = false, spat = false, temp = false] : memref<4096x512xf16> to memref<?x?xf16, strided<[512, 1], offset: ?>>
                %27 = loom.alloc [%12, %14] on @L1 : memref<?x?xf16>
                %28 = loom.copy_to_tensor %26, %27 on @L1, interconnect : [], broadcast : [1, 1] : memref<?x?xf16, strided<[512, 1], offset: ?>>, memref<?x?xf16> -> tensor<?x?xf16>
                %29 = arith.muli %16, %13 : index
                %30 = loom.subview %arg1[%25, %29] [%14, %13] [1, 1], reuse : [seq = false, spat = false, temp = false] : memref<512x4096xf16> to memref<?x?xf16, strided<[4096, 1], offset: ?>>
                %31 = loom.alloc [%14, %13] on @L1 : memref<?x?xf16>
                %32 = loom.copy_to_tensor %30, %31 on @L1, interconnect : [], broadcast : [1, 1] : memref<?x?xf16, strided<[4096, 1], offset: ?>>, memref<?x?xf16> -> tensor<?x?xf16>
                %33 = linalg.matmul ins(%28, %32 : tensor<?x?xf16>, tensor<?x?xf16>) outs(%arg8 : tensor<?x?xf16>) -> tensor<?x?xf16>
                affine.yield %33 : tensor<?x?xf16>
              } {loom.iter_type = #loom.iter_type<sequential>}
              %21 = arith.muli %15, %12 : index
              %22 = arith.muli %16, %13 : index
              %23 = loom.subview %arg2[%21, %22] [%12, %13] [1, 1], reuse : [seq = false, spat = false, temp = false] : memref<4096x4096xf16> to memref<?x?xf16, strided<[4096, 1], offset: ?>>
              loom.copy_from_tensor %20, %23 on @DRAM : tensor<?x?xf16>, memref<?x?xf16, strided<[4096, 1], offset: ?>>
            } {loom.iter_type = #loom.iter_type<temporal>}
          } {loom.iter_type = #loom.iter_type<temporal>}
        } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @y}
      } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @x}
      return
    }
  }
  module attributes {loom.pass_name = "EnumerateHWMapping"} {
    loom.constraint_space @constraints {
      %12 = loom.symbolic_var "BM" : index
      %13 = loom.symbolic_var "BN" : index
      %14 = loom.symbolic_var "BK" : index
      loom.range %12[32, 1024]
      loom.range %13[32, 1024]
      loom.range %14[32, 1024]
      loom.align %12 by 32
      loom.align %13 by 32
      loom.align %14 by 32
      loom.polynomial_constraint(%12, %13, %14) {monomials = [{coeff = -1 : i64, vars = [0, 1, 2]}], upper_bound = -262144 : i64}
      loom.polynomial_constraint(%12, %13, %14) {monomials = [{coeff = 2 : i64, vars = [0, 1]}, {coeff = 2 : i64, vars = [0, 2]}, {coeff = 2 : i64, vars = [2, 1]}], upper_bound = 1499136 : i64}
      loom.polynomial_constraint(%12) {monomials = [{coeff = 8 : i64, vars = [0]}], upper_bound = 1024 : i64}
      loom.polynomial_constraint(%13) {monomials = [{coeff = 8 : i64, vars = [0]}], upper_bound = 1024 : i64}
    }
    func.func @matmul__d0i0_d1i1__f10(%arg0: memref<4096x512xf16>, %arg1: memref<512x4096xf16>, %arg2: memref<4096x4096xf16>) {
      %cst = arith.constant 0.000000e+00 : f16
      %12 = loom.get_symbolic_block_size @constraints::@BM : index
      %13 = loom.get_symbolic_block_size @constraints::@BN : index
      %14 = loom.get_symbolic_block_size @constraints::@BK : index
      affine.parallel (%arg3) = (0) to (8) {
        affine.parallel (%arg4) = (0) to (8) {
          affine.for %arg5 = 0 to affine_map<()[s0, s1] -> ((4096 ceildiv s1) ceildiv 8)>()[%12, %13] {
            affine.for %arg6 = 0 to affine_map<()[s0, s1] -> ((4096 ceildiv s0) ceildiv 8)>()[%12, %13] {
              %15 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 8)>(%arg3, %arg6)
              %16 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 8)>(%arg4, %arg5)
              %17 = loom.alloc [%12, %13] on @L1 : memref<?x?xf16>
              %18 = loom.init_tensor %17[%12, %13] : memref<?x?xf16> -> tensor<?x?xf16>
              %19 = linalg.fill ins(%cst : f16) outs(%18 : tensor<?x?xf16>) -> tensor<?x?xf16>
              %20 = affine.for %arg7 = 0 to affine_map<()[s0] -> (512 ceildiv s0)>()[%14] iter_args(%arg8 = %19) -> (tensor<?x?xf16>) {
                %24 = arith.muli %15, %12 : index
                %25 = arith.muli %arg7, %14 : index
                %26 = loom.subview %arg0[%24, %25] [%12, %14] [1, 1], reuse : [seq = false, spat = false, temp = false] : memref<4096x512xf16> to memref<?x?xf16, strided<[512, 1], offset: ?>>
                %27 = loom.alloc [%12, %14] on @L1 : memref<?x?xf16>
                %28 = loom.copy_to_tensor %26, %27 on @L1, interconnect : [], broadcast : [1, 1] : memref<?x?xf16, strided<[512, 1], offset: ?>>, memref<?x?xf16> -> tensor<?x?xf16>
                %29 = arith.muli %16, %13 : index
                %30 = loom.subview %arg1[%25, %29] [%14, %13] [1, 1], reuse : [seq = false, spat = false, temp = false] : memref<512x4096xf16> to memref<?x?xf16, strided<[4096, 1], offset: ?>>
                %31 = loom.alloc [%14, %13] on @L1 : memref<?x?xf16>
                %32 = loom.copy_to_tensor %30, %31 on @L1, interconnect : [], broadcast : [1, 1] : memref<?x?xf16, strided<[4096, 1], offset: ?>>, memref<?x?xf16> -> tensor<?x?xf16>
                %33 = linalg.matmul ins(%28, %32 : tensor<?x?xf16>, tensor<?x?xf16>) outs(%arg8 : tensor<?x?xf16>) -> tensor<?x?xf16>
                affine.yield %33 : tensor<?x?xf16>
              } {loom.iter_type = #loom.iter_type<sequential>}
              %21 = arith.muli %15, %12 : index
              %22 = arith.muli %16, %13 : index
              %23 = loom.subview %arg2[%21, %22] [%12, %13] [1, 1], reuse : [seq = false, spat = false, temp = false] : memref<4096x4096xf16> to memref<?x?xf16, strided<[4096, 1], offset: ?>>
              loom.copy_from_tensor %20, %23 on @DRAM : tensor<?x?xf16>, memref<?x?xf16, strided<[4096, 1], offset: ?>>
            } {loom.iter_type = #loom.iter_type<temporal>}
          } {loom.iter_type = #loom.iter_type<temporal>}
        } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @y}
      } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @x}
      return
    }
  }
  module attributes {loom.pass_name = "EnumerateHWMapping"} {
    loom.constraint_space @constraints {
      %12 = loom.symbolic_var "BM" : index
      %13 = loom.symbolic_var "BN" : index
      %14 = loom.symbolic_var "BK" : index
      loom.range %12[32, 1024]
      loom.range %13[32, 1024]
      loom.range %14[32, 1024]
      loom.align %12 by 32
      loom.align %13 by 32
      loom.align %14 by 32
      loom.polynomial_constraint(%12, %13, %14) {monomials = [{coeff = -1 : i64, vars = [0, 1, 2]}], upper_bound = -262144 : i64}
      loom.polynomial_constraint(%12, %13, %14) {monomials = [{coeff = 2 : i64, vars = [0, 1]}, {coeff = 2 : i64, vars = [0, 2]}, {coeff = 2 : i64, vars = [2, 1]}], upper_bound = 1499136 : i64}
      loom.polynomial_constraint(%12) {monomials = [{coeff = 8 : i64, vars = [0]}], upper_bound = 1024 : i64}
      loom.polynomial_constraint(%13) {monomials = [{coeff = 8 : i64, vars = [0]}], upper_bound = 1024 : i64}
    }
    func.func @matmul__d1i0_d0i1__f01(%arg0: memref<4096x512xf16>, %arg1: memref<512x4096xf16>, %arg2: memref<4096x4096xf16>) {
      %cst = arith.constant 0.000000e+00 : f16
      %12 = loom.get_symbolic_block_size @constraints::@BM : index
      %13 = loom.get_symbolic_block_size @constraints::@BN : index
      %14 = loom.get_symbolic_block_size @constraints::@BK : index
      affine.parallel (%arg3) = (0) to (8) {
        affine.parallel (%arg4) = (0) to (8) {
          affine.for %arg5 = 0 to affine_map<()[s0, s1] -> ((4096 ceildiv s0) ceildiv 8)>()[%12, %13] {
            affine.for %arg6 = 0 to affine_map<()[s0, s1] -> ((4096 ceildiv s1) ceildiv 8)>()[%12, %13] {
              %15 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 8)>(%arg3, %arg5)
              %16 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 8)>(%arg4, %arg6)
              %17 = loom.alloc [%12, %13] on @L1 : memref<?x?xf16>
              %18 = loom.init_tensor %17[%12, %13] : memref<?x?xf16> -> tensor<?x?xf16>
              %19 = linalg.fill ins(%cst : f16) outs(%18 : tensor<?x?xf16>) -> tensor<?x?xf16>
              %20 = affine.for %arg7 = 0 to affine_map<()[s0] -> (512 ceildiv s0)>()[%14] iter_args(%arg8 = %19) -> (tensor<?x?xf16>) {
                %24 = arith.muli %15, %12 : index
                %25 = arith.muli %arg7, %14 : index
                %26 = loom.subview %arg0[%24, %25] [%12, %14] [1, 1], reuse : [seq = false, spat = false, temp = false] : memref<4096x512xf16> to memref<?x?xf16, strided<[512, 1], offset: ?>>
                %27 = loom.alloc [%12, %14] on @L1 : memref<?x?xf16>
                %28 = loom.copy_to_tensor %26, %27 on @L1, interconnect : [], broadcast : [1, 1] : memref<?x?xf16, strided<[512, 1], offset: ?>>, memref<?x?xf16> -> tensor<?x?xf16>
                %29 = arith.muli %16, %13 : index
                %30 = loom.subview %arg1[%25, %29] [%14, %13] [1, 1], reuse : [seq = false, spat = false, temp = false] : memref<512x4096xf16> to memref<?x?xf16, strided<[4096, 1], offset: ?>>
                %31 = loom.alloc [%14, %13] on @L1 : memref<?x?xf16>
                %32 = loom.copy_to_tensor %30, %31 on @L1, interconnect : [], broadcast : [1, 1] : memref<?x?xf16, strided<[4096, 1], offset: ?>>, memref<?x?xf16> -> tensor<?x?xf16>
                %33 = linalg.matmul ins(%28, %32 : tensor<?x?xf16>, tensor<?x?xf16>) outs(%arg8 : tensor<?x?xf16>) -> tensor<?x?xf16>
                affine.yield %33 : tensor<?x?xf16>
              } {loom.iter_type = #loom.iter_type<sequential>}
              %21 = arith.muli %15, %12 : index
              %22 = arith.muli %16, %13 : index
              %23 = loom.subview %arg2[%21, %22] [%12, %13] [1, 1], reuse : [seq = false, spat = false, temp = false] : memref<4096x4096xf16> to memref<?x?xf16, strided<[4096, 1], offset: ?>>
              loom.copy_from_tensor %20, %23 on @DRAM : tensor<?x?xf16>, memref<?x?xf16, strided<[4096, 1], offset: ?>>
            } {loom.iter_type = #loom.iter_type<temporal>}
          } {loom.iter_type = #loom.iter_type<temporal>}
        } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @x}
      } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @y}
      return
    }
  }
  module attributes {loom.pass_name = "EnumerateHWMapping"} {
    loom.constraint_space @constraints {
      %12 = loom.symbolic_var "BM" : index
      %13 = loom.symbolic_var "BN" : index
      %14 = loom.symbolic_var "BK" : index
      loom.range %12[32, 1024]
      loom.range %13[32, 1024]
      loom.range %14[32, 1024]
      loom.align %12 by 32
      loom.align %13 by 32
      loom.align %14 by 32
      loom.polynomial_constraint(%12, %13, %14) {monomials = [{coeff = -1 : i64, vars = [0, 1, 2]}], upper_bound = -262144 : i64}
      loom.polynomial_constraint(%12, %13, %14) {monomials = [{coeff = 2 : i64, vars = [0, 1]}, {coeff = 2 : i64, vars = [0, 2]}, {coeff = 2 : i64, vars = [2, 1]}], upper_bound = 1499136 : i64}
      loom.polynomial_constraint(%12) {monomials = [{coeff = 8 : i64, vars = [0]}], upper_bound = 1024 : i64}
      loom.polynomial_constraint(%13) {monomials = [{coeff = 8 : i64, vars = [0]}], upper_bound = 1024 : i64}
    }
    func.func @matmul__d1i0_d0i1__f10(%arg0: memref<4096x512xf16>, %arg1: memref<512x4096xf16>, %arg2: memref<4096x4096xf16>) {
      %cst = arith.constant 0.000000e+00 : f16
      %12 = loom.get_symbolic_block_size @constraints::@BM : index
      %13 = loom.get_symbolic_block_size @constraints::@BN : index
      %14 = loom.get_symbolic_block_size @constraints::@BK : index
      affine.parallel (%arg3) = (0) to (8) {
        affine.parallel (%arg4) = (0) to (8) {
          affine.for %arg5 = 0 to affine_map<()[s0, s1] -> ((4096 ceildiv s1) ceildiv 8)>()[%12, %13] {
            affine.for %arg6 = 0 to affine_map<()[s0, s1] -> ((4096 ceildiv s0) ceildiv 8)>()[%12, %13] {
              %15 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 8)>(%arg3, %arg6)
              %16 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 8)>(%arg4, %arg5)
              %17 = loom.alloc [%12, %13] on @L1 : memref<?x?xf16>
              %18 = loom.init_tensor %17[%12, %13] : memref<?x?xf16> -> tensor<?x?xf16>
              %19 = linalg.fill ins(%cst : f16) outs(%18 : tensor<?x?xf16>) -> tensor<?x?xf16>
              %20 = affine.for %arg7 = 0 to affine_map<()[s0] -> (512 ceildiv s0)>()[%14] iter_args(%arg8 = %19) -> (tensor<?x?xf16>) {
                %24 = arith.muli %15, %12 : index
                %25 = arith.muli %arg7, %14 : index
                %26 = loom.subview %arg0[%24, %25] [%12, %14] [1, 1], reuse : [seq = false, spat = false, temp = false] : memref<4096x512xf16> to memref<?x?xf16, strided<[512, 1], offset: ?>>
                %27 = loom.alloc [%12, %14] on @L1 : memref<?x?xf16>
                %28 = loom.copy_to_tensor %26, %27 on @L1, interconnect : [], broadcast : [1, 1] : memref<?x?xf16, strided<[512, 1], offset: ?>>, memref<?x?xf16> -> tensor<?x?xf16>
                %29 = arith.muli %16, %13 : index
                %30 = loom.subview %arg1[%25, %29] [%14, %13] [1, 1], reuse : [seq = false, spat = false, temp = false] : memref<512x4096xf16> to memref<?x?xf16, strided<[4096, 1], offset: ?>>
                %31 = loom.alloc [%14, %13] on @L1 : memref<?x?xf16>
                %32 = loom.copy_to_tensor %30, %31 on @L1, interconnect : [], broadcast : [1, 1] : memref<?x?xf16, strided<[4096, 1], offset: ?>>, memref<?x?xf16> -> tensor<?x?xf16>
                %33 = linalg.matmul ins(%28, %32 : tensor<?x?xf16>, tensor<?x?xf16>) outs(%arg8 : tensor<?x?xf16>) -> tensor<?x?xf16>
                affine.yield %33 : tensor<?x?xf16>
              } {loom.iter_type = #loom.iter_type<sequential>}
              %21 = arith.muli %15, %12 : index
              %22 = arith.muli %16, %13 : index
              %23 = loom.subview %arg2[%21, %22] [%12, %13] [1, 1], reuse : [seq = false, spat = false, temp = false] : memref<4096x4096xf16> to memref<?x?xf16, strided<[4096, 1], offset: ?>>
              loom.copy_from_tensor %20, %23 on @DRAM : tensor<?x?xf16>, memref<?x?xf16, strided<[4096, 1], offset: ?>>
            } {loom.iter_type = #loom.iter_type<temporal>}
          } {loom.iter_type = #loom.iter_type<temporal>}
        } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @x}
      } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @y}
      return
    }
  }
  module attributes {loom.pass_name = "EnumerateHWMapping"} {
    loom.constraint_space @constraints {
      %12 = loom.symbolic_var "BM" : index
      %13 = loom.symbolic_var "BN" : index
      %14 = loom.symbolic_var "BK" : index
      loom.range %12[32, 1024]
      loom.range %13[32, 1024]
      loom.range %14[32, 1024]
      loom.align %12 by 32
      loom.align %13 by 32
      loom.align %14 by 32
      loom.polynomial_constraint(%12, %13, %14) {monomials = [{coeff = -1 : i64, vars = [0, 1, 2]}], upper_bound = -262144 : i64}
      loom.polynomial_constraint(%12, %13, %14) {monomials = [{coeff = 2 : i64, vars = [0, 1]}, {coeff = 2 : i64, vars = [0, 2]}, {coeff = 2 : i64, vars = [2, 1]}], upper_bound = 1499136 : i64}
      loom.polynomial_constraint(%13) {monomials = [{coeff = 64 : i64, vars = [0]}], upper_bound = 1024 : i64}
    }
    func.func @matmul__d0i1_d1i1__f01(%arg0: memref<4096x512xf16>, %arg1: memref<512x4096xf16>, %arg2: memref<4096x4096xf16>) {
      %cst = arith.constant 0.000000e+00 : f16
      %12 = loom.get_symbolic_block_size @constraints::@BM : index
      %13 = loom.get_symbolic_block_size @constraints::@BN : index
      %14 = loom.get_symbolic_block_size @constraints::@BK : index
      affine.parallel (%arg3) = (0) to (8) {
        affine.parallel (%arg4) = (0) to (8) {
          affine.for %arg5 = 0 to affine_map<()[s0, s1] -> (4096 ceildiv s0)>()[%12, %13] {
            affine.for %arg6 = 0 to affine_map<()[s0, s1] -> (((4096 ceildiv s1) ceildiv 8) ceildiv 8)>()[%12, %13] {
              %15 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 8 + d1 + d2 * 64)>(%arg3, %arg4, %arg6)
              %16 = loom.alloc [%12, %13] on @L1 : memref<?x?xf16>
              %17 = loom.init_tensor %16[%12, %13] : memref<?x?xf16> -> tensor<?x?xf16>
              %18 = linalg.fill ins(%cst : f16) outs(%17 : tensor<?x?xf16>) -> tensor<?x?xf16>
              %19 = affine.for %arg7 = 0 to affine_map<()[s0] -> (512 ceildiv s0)>()[%14] iter_args(%arg8 = %18) -> (tensor<?x?xf16>) {
                %23 = arith.muli %arg5, %12 : index
                %24 = arith.muli %arg7, %14 : index
                %25 = loom.subview %arg0[%23, %24] [%12, %14] [1, 1], reuse : [seq = false, spat = false, temp = false] : memref<4096x512xf16> to memref<?x?xf16, strided<[512, 1], offset: ?>>
                %26 = loom.alloc [%12, %14] on @L1 : memref<?x?xf16>
                %27 = loom.copy_to_tensor %25, %26 on @L1, interconnect : [], broadcast : [1, 1] : memref<?x?xf16, strided<[512, 1], offset: ?>>, memref<?x?xf16> -> tensor<?x?xf16>
                %28 = arith.muli %15, %13 : index
                %29 = loom.subview %arg1[%24, %28] [%14, %13] [1, 1], reuse : [seq = false, spat = false, temp = false] : memref<512x4096xf16> to memref<?x?xf16, strided<[4096, 1], offset: ?>>
                %30 = loom.alloc [%14, %13] on @L1 : memref<?x?xf16>
                %31 = loom.copy_to_tensor %29, %30 on @L1, interconnect : [], broadcast : [1, 1] : memref<?x?xf16, strided<[4096, 1], offset: ?>>, memref<?x?xf16> -> tensor<?x?xf16>
                %32 = linalg.matmul ins(%27, %31 : tensor<?x?xf16>, tensor<?x?xf16>) outs(%arg8 : tensor<?x?xf16>) -> tensor<?x?xf16>
                affine.yield %32 : tensor<?x?xf16>
              } {loom.iter_type = #loom.iter_type<sequential>}
              %20 = arith.muli %arg5, %12 : index
              %21 = arith.muli %15, %13 : index
              %22 = loom.subview %arg2[%20, %21] [%12, %13] [1, 1], reuse : [seq = false, spat = false, temp = false] : memref<4096x4096xf16> to memref<?x?xf16, strided<[4096, 1], offset: ?>>
              loom.copy_from_tensor %19, %22 on @DRAM : tensor<?x?xf16>, memref<?x?xf16, strided<[4096, 1], offset: ?>>
            } {loom.iter_type = #loom.iter_type<temporal>}
          } {loom.iter_type = #loom.iter_type<temporal>}
        } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @y}
      } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @x}
      return
    }
  }
  module attributes {loom.pass_name = "EnumerateHWMapping"} {
    loom.constraint_space @constraints {
      %12 = loom.symbolic_var "BM" : index
      %13 = loom.symbolic_var "BN" : index
      %14 = loom.symbolic_var "BK" : index
      loom.range %12[32, 1024]
      loom.range %13[32, 1024]
      loom.range %14[32, 1024]
      loom.align %12 by 32
      loom.align %13 by 32
      loom.align %14 by 32
      loom.polynomial_constraint(%12, %13, %14) {monomials = [{coeff = -1 : i64, vars = [0, 1, 2]}], upper_bound = -262144 : i64}
      loom.polynomial_constraint(%12, %13, %14) {monomials = [{coeff = 2 : i64, vars = [0, 1]}, {coeff = 2 : i64, vars = [0, 2]}, {coeff = 2 : i64, vars = [2, 1]}], upper_bound = 1499136 : i64}
      loom.polynomial_constraint(%13) {monomials = [{coeff = 64 : i64, vars = [0]}], upper_bound = 1024 : i64}
    }
    func.func @matmul__d0i1_d1i1__f10(%arg0: memref<4096x512xf16>, %arg1: memref<512x4096xf16>, %arg2: memref<4096x4096xf16>) {
      %cst = arith.constant 0.000000e+00 : f16
      %12 = loom.get_symbolic_block_size @constraints::@BM : index
      %13 = loom.get_symbolic_block_size @constraints::@BN : index
      %14 = loom.get_symbolic_block_size @constraints::@BK : index
      affine.parallel (%arg3) = (0) to (8) {
        affine.parallel (%arg4) = (0) to (8) {
          affine.for %arg5 = 0 to affine_map<()[s0, s1] -> (((4096 ceildiv s1) ceildiv 8) ceildiv 8)>()[%12, %13] {
            affine.for %arg6 = 0 to affine_map<()[s0, s1] -> (4096 ceildiv s0)>()[%12, %13] {
              %15 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 8 + d1 + d2 * 64)>(%arg3, %arg4, %arg5)
              %16 = loom.alloc [%12, %13] on @L1 : memref<?x?xf16>
              %17 = loom.init_tensor %16[%12, %13] : memref<?x?xf16> -> tensor<?x?xf16>
              %18 = linalg.fill ins(%cst : f16) outs(%17 : tensor<?x?xf16>) -> tensor<?x?xf16>
              %19 = affine.for %arg7 = 0 to affine_map<()[s0] -> (512 ceildiv s0)>()[%14] iter_args(%arg8 = %18) -> (tensor<?x?xf16>) {
                %23 = arith.muli %arg6, %12 : index
                %24 = arith.muli %arg7, %14 : index
                %25 = loom.subview %arg0[%23, %24] [%12, %14] [1, 1], reuse : [seq = false, spat = false, temp = false] : memref<4096x512xf16> to memref<?x?xf16, strided<[512, 1], offset: ?>>
                %26 = loom.alloc [%12, %14] on @L1 : memref<?x?xf16>
                %27 = loom.copy_to_tensor %25, %26 on @L1, interconnect : [], broadcast : [1, 1] : memref<?x?xf16, strided<[512, 1], offset: ?>>, memref<?x?xf16> -> tensor<?x?xf16>
                %28 = arith.muli %15, %13 : index
                %29 = loom.subview %arg1[%24, %28] [%14, %13] [1, 1], reuse : [seq = false, spat = false, temp = false] : memref<512x4096xf16> to memref<?x?xf16, strided<[4096, 1], offset: ?>>
                %30 = loom.alloc [%14, %13] on @L1 : memref<?x?xf16>
                %31 = loom.copy_to_tensor %29, %30 on @L1, interconnect : [], broadcast : [1, 1] : memref<?x?xf16, strided<[4096, 1], offset: ?>>, memref<?x?xf16> -> tensor<?x?xf16>
                %32 = linalg.matmul ins(%27, %31 : tensor<?x?xf16>, tensor<?x?xf16>) outs(%arg8 : tensor<?x?xf16>) -> tensor<?x?xf16>
                affine.yield %32 : tensor<?x?xf16>
              } {loom.iter_type = #loom.iter_type<sequential>}
              %20 = arith.muli %arg6, %12 : index
              %21 = arith.muli %15, %13 : index
              %22 = loom.subview %arg2[%20, %21] [%12, %13] [1, 1], reuse : [seq = false, spat = false, temp = false] : memref<4096x4096xf16> to memref<?x?xf16, strided<[4096, 1], offset: ?>>
              loom.copy_from_tensor %19, %22 on @DRAM : tensor<?x?xf16>, memref<?x?xf16, strided<[4096, 1], offset: ?>>
            } {loom.iter_type = #loom.iter_type<temporal>}
          } {loom.iter_type = #loom.iter_type<temporal>}
        } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @y}
      } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @x}
      return
    }
  }
}
