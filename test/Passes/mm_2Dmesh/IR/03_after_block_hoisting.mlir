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
      loom.polynomial_constraint(%12, %13, %14) {monomials = [{coeff = 4 : i64, vars = [0, 1]}, {coeff = 4 : i64, vars = [0, 2]}, {coeff = 4 : i64, vars = [2, 1]}], upper_bound = 1499136 : i64}
      loom.polynomial_constraint(%12) {monomials = [{coeff = 64 : i64, vars = [0]}], upper_bound = 1024 : i64}
    }
    func.func @matmul__d0i0_d1i0__f01(%arg0: memref<128x128xf32>, %arg1: memref<128x256xf32>, %arg2: memref<128x256xf32>) {
      %cst = arith.constant 0.000000e+00 : f32
      %12 = loom.get_symbolic_block_size @constraints::@BM : index
      %13 = loom.get_symbolic_block_size @constraints::@BN : index
      %14 = loom.get_symbolic_block_size @constraints::@BK : index
      affine.parallel (%arg3) = (0) to (8) {
        affine.parallel (%arg4) = (0) to (8) {
          affine.for %arg5 = 0 to affine_map<()[s0, s1] -> (((128 ceildiv s0) ceildiv 8) ceildiv 8)>()[%12, %13] {
            affine.for %arg6 = 0 to affine_map<()[s0, s1] -> (256 ceildiv s1)>()[%12, %13] {
              %15 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 8 + d1 + d2 * 64)>(%arg3, %arg4, %arg5)
              %16 = loom.alloc(%12, %13) on @L1 : !loom.buffer_token
              %17 = loom.init_tensor %16(%12, %13) : !loom.buffer_token -> tensor<?x?xf32>
              %18 = linalg.fill ins(%cst : f32) outs(%17 : tensor<?x?xf32>) -> tensor<?x?xf32>
              %19 = affine.for %arg7 = 0 to affine_map<()[s0] -> (128 ceildiv s0)>()[%14] iter_args(%arg8 = %18) -> (tensor<?x?xf32>) {
                %23 = arith.muli %15, %12 : index
                %24 = arith.muli %arg7, %14 : index
                %25 = loom.view %arg0[%23, %24] [%12, %14] [1, 1], reuse : [seq = false, spat = false, temp = false] : memref<128x128xf32> -> !loom.view
                %26 = loom.alloc(%12, %14) on @L1 : !loom.buffer_token
                %27 = loom.copy_to_tensor %25, %26, interconnect : [], broadcast : [1, 1] : !loom.view, !loom.buffer_token -> tensor<?x?xf32>
                %28 = arith.muli %arg6, %13 : index
                %29 = loom.view %arg1[%24, %28] [%14, %13] [1, 1], reuse : [seq = false, spat = false, temp = false] : memref<128x256xf32> -> !loom.view
                %30 = loom.alloc(%14, %13) on @L1 : !loom.buffer_token
                %31 = loom.copy_to_tensor %29, %30, interconnect : [], broadcast : [1, 1] : !loom.view, !loom.buffer_token -> tensor<?x?xf32>
                %32 = linalg.matmul ins(%27, %31 : tensor<?x?xf32>, tensor<?x?xf32>) outs(%18 : tensor<?x?xf32>) -> tensor<?x?xf32>
                %33 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%arg8, %32 : tensor<?x?xf32>, tensor<?x?xf32>) outs(%17 : tensor<?x?xf32>) {
                ^bb0(%in: f32, %in_0: f32, %out: f32):
                  %34 = arith.addf %in, %in_0 : f32
                  linalg.yield %34 : f32
                } -> tensor<?x?xf32>
                affine.yield %33 : tensor<?x?xf32>
              } {loom.iter_type = #loom.iter_type<sequential>}
              %20 = arith.muli %15, %12 : index
              %21 = arith.muli %arg6, %13 : index
              %22 = loom.view %arg2[%20, %21] [%12, %13] [1, 1], reuse : [seq = false, spat = false, temp = false] : memref<128x256xf32> -> !loom.view
              loom.copy_from_tensor %19, %22 : tensor<?x?xf32>, !loom.view
            } {loom.iter_type = #loom.iter_type<temporal>}
          } {loom.iter_type = #loom.iter_type<temporal>}
        } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @y}
      } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @x}
      return
    }
  }
  module attributes {loom.pass_name = "HoistBlockLoading"} {
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
      loom.polynomial_constraint(%12, %13, %14) {monomials = [{coeff = 4 : i64, vars = [0, 1]}, {coeff = 4 : i64, vars = [0, 2]}, {coeff = 4 : i64, vars = [2, 1]}], upper_bound = 1499136 : i64}
      loom.polynomial_constraint(%12) {monomials = [{coeff = 64 : i64, vars = [0]}], upper_bound = 1024 : i64}
      loom.polynomial_constraint(%12, %13, %14) {monomials = [{coeff = 4 : i64, vars = [0, 1]}, {coeff = 4 : i64, vars = [2, 1]}], upper_bound = 1499136 : i64}
    }
    func.func @matmul__d0i0_d1i0__f01__hoist_block_0(%arg0: memref<128x128xf32>, %arg1: memref<128x256xf32>, %arg2: memref<128x256xf32>) {
      %cst = arith.constant 0.000000e+00 : f32
      %12 = loom.get_symbolic_block_size @constraints::@BM : index
      %13 = loom.get_symbolic_block_size @constraints::@BN : index
      %14 = loom.get_symbolic_block_size @constraints::@BK : index
      affine.parallel (%arg3) = (0) to (8) {
        affine.parallel (%arg4) = (0) to (8) {
          affine.for %arg5 = 0 to affine_map<()[s0, s1] -> (((128 ceildiv s0) ceildiv 8) ceildiv 8)>()[%12, %13] {
            affine.for %arg6 = 0 to affine_map<()[s0, s1] -> (256 ceildiv s1)>()[%12, %13] {
              %15 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 8 + d1 + d2 * 64)>(%arg3, %arg4, %arg5)
              %16 = loom.alloc(%12, %13) on @L1 : !loom.buffer_token
              %17 = loom.init_tensor %16(%12, %13) : !loom.buffer_token -> tensor<?x?xf32>
              %18 = linalg.fill ins(%cst : f32) outs(%17 : tensor<?x?xf32>) -> tensor<?x?xf32>
              %19 = arith.muli %15, %12 : index
              %c0 = arith.constant 0 : index
              %c128 = arith.constant 128 : index
              %20 = loom.view %arg0[%19, %c0] [%12, %c128] [1, 1], reuse : [seq = false, spat = false, temp = false] : memref<128x128xf32> -> !loom.view
              %21 = loom.alloc(%12, %c128) on @L1 : !loom.buffer_token
              %22 = loom.pack_to_tensor %20, %21 inner_tiles = [%12, %14] outer_dims_perm = [1, 0] : !loom.view, !loom.buffer_token -> tensor<?x?x?xf32>
              %23 = affine.for %arg7 = 0 to affine_map<()[s0] -> (128 ceildiv s0)>()[%14] iter_args(%arg8 = %18) -> (tensor<?x?xf32>) {
                %27 = arith.muli %15, %12 : index
                %28 = arith.muli %arg7, %14 : index
                %extracted_slice = tensor.extract_slice %22[%arg7, 0, 0] [1, %12, %14] [1, 1, 1] : tensor<?x?x?xf32> to tensor<1x?x?xf32>
                %collapsed = tensor.collapse_shape %extracted_slice [[0, 1], [2]] : tensor<1x?x?xf32> into tensor<?x?xf32>
                %29 = arith.muli %arg6, %13 : index
                %30 = loom.view %arg1[%28, %29] [%14, %13] [1, 1], reuse : [seq = false, spat = false, temp = false] : memref<128x256xf32> -> !loom.view
                %31 = loom.alloc(%14, %13) on @L1 : !loom.buffer_token
                %32 = loom.copy_to_tensor %30, %31, interconnect : [], broadcast : [1, 1] : !loom.view, !loom.buffer_token -> tensor<?x?xf32>
                %33 = linalg.matmul ins(%collapsed, %32 : tensor<?x?xf32>, tensor<?x?xf32>) outs(%18 : tensor<?x?xf32>) -> tensor<?x?xf32>
                %34 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%arg8, %33 : tensor<?x?xf32>, tensor<?x?xf32>) outs(%17 : tensor<?x?xf32>) {
                ^bb0(%in: f32, %in_0: f32, %out: f32):
                  %35 = arith.addf %in, %in_0 : f32
                  linalg.yield %35 : f32
                } -> tensor<?x?xf32>
                affine.yield %34 : tensor<?x?xf32>
              } {loom.iter_type = #loom.iter_type<sequential>}
              %24 = arith.muli %15, %12 : index
              %25 = arith.muli %arg6, %13 : index
              %26 = loom.view %arg2[%24, %25] [%12, %13] [1, 1], reuse : [seq = false, spat = false, temp = false] : memref<128x256xf32> -> !loom.view
              loom.copy_from_tensor %23, %26 : tensor<?x?xf32>, !loom.view
            } {loom.iter_type = #loom.iter_type<temporal>}
          } {loom.iter_type = #loom.iter_type<temporal>}
        } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @y}
      } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @x}
      return
    }
  }
  module attributes {loom.pass_name = "HoistBlockLoading"} {
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
      loom.polynomial_constraint(%12, %13, %14) {monomials = [{coeff = 4 : i64, vars = [0, 1]}, {coeff = 4 : i64, vars = [0, 2]}, {coeff = 4 : i64, vars = [2, 1]}], upper_bound = 1499136 : i64}
      loom.polynomial_constraint(%12) {monomials = [{coeff = 64 : i64, vars = [0]}], upper_bound = 1024 : i64}
      loom.polynomial_constraint(%12, %13, %14) {monomials = [{coeff = 4 : i64, vars = [0, 1]}, {coeff = 4 : i64, vars = [0, 2]}], upper_bound = 1499136 : i64}
    }
    func.func @matmul__d0i0_d1i0__f01__hoist_block_1(%arg0: memref<128x128xf32>, %arg1: memref<128x256xf32>, %arg2: memref<128x256xf32>) {
      %cst = arith.constant 0.000000e+00 : f32
      %12 = loom.get_symbolic_block_size @constraints::@BM : index
      %13 = loom.get_symbolic_block_size @constraints::@BN : index
      %14 = loom.get_symbolic_block_size @constraints::@BK : index
      affine.parallel (%arg3) = (0) to (8) {
        affine.parallel (%arg4) = (0) to (8) {
          affine.for %arg5 = 0 to affine_map<()[s0, s1] -> (((128 ceildiv s0) ceildiv 8) ceildiv 8)>()[%12, %13] {
            affine.for %arg6 = 0 to affine_map<()[s0, s1] -> (256 ceildiv s1)>()[%12, %13] {
              %15 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 8 + d1 + d2 * 64)>(%arg3, %arg4, %arg5)
              %16 = loom.alloc(%12, %13) on @L1 : !loom.buffer_token
              %17 = loom.init_tensor %16(%12, %13) : !loom.buffer_token -> tensor<?x?xf32>
              %18 = linalg.fill ins(%cst : f32) outs(%17 : tensor<?x?xf32>) -> tensor<?x?xf32>
              %c0 = arith.constant 0 : index
              %19 = arith.muli %arg6, %13 : index
              %c128 = arith.constant 128 : index
              %20 = loom.view %arg1[%c0, %19] [%c128, %13] [1, 1], reuse : [seq = false, spat = false, temp = false] : memref<128x256xf32> -> !loom.view
              %21 = loom.alloc(%c128, %13) on @L1 : !loom.buffer_token
              %22 = loom.pack_to_tensor %20, %21 inner_tiles = [%14, %13] outer_dims_perm = [0, 1] : !loom.view, !loom.buffer_token -> tensor<?x?x?xf32>
              %23 = affine.for %arg7 = 0 to affine_map<()[s0] -> (128 ceildiv s0)>()[%14] iter_args(%arg8 = %18) -> (tensor<?x?xf32>) {
                %27 = arith.muli %15, %12 : index
                %28 = arith.muli %arg7, %14 : index
                %29 = loom.view %arg0[%27, %28] [%12, %14] [1, 1], reuse : [seq = false, spat = false, temp = false] : memref<128x128xf32> -> !loom.view
                %30 = loom.alloc(%12, %14) on @L1 : !loom.buffer_token
                %31 = loom.copy_to_tensor %29, %30, interconnect : [], broadcast : [1, 1] : !loom.view, !loom.buffer_token -> tensor<?x?xf32>
                %32 = arith.muli %arg6, %13 : index
                %extracted_slice = tensor.extract_slice %22[%arg7, 0, 0] [1, %14, %13] [1, 1, 1] : tensor<?x?x?xf32> to tensor<1x?x?xf32>
                %collapsed = tensor.collapse_shape %extracted_slice [[0, 1], [2]] : tensor<1x?x?xf32> into tensor<?x?xf32>
                %33 = linalg.matmul ins(%31, %collapsed : tensor<?x?xf32>, tensor<?x?xf32>) outs(%18 : tensor<?x?xf32>) -> tensor<?x?xf32>
                %34 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%arg8, %33 : tensor<?x?xf32>, tensor<?x?xf32>) outs(%17 : tensor<?x?xf32>) {
                ^bb0(%in: f32, %in_0: f32, %out: f32):
                  %35 = arith.addf %in, %in_0 : f32
                  linalg.yield %35 : f32
                } -> tensor<?x?xf32>
                affine.yield %34 : tensor<?x?xf32>
              } {loom.iter_type = #loom.iter_type<sequential>}
              %24 = arith.muli %15, %12 : index
              %25 = arith.muli %arg6, %13 : index
              %26 = loom.view %arg2[%24, %25] [%12, %13] [1, 1], reuse : [seq = false, spat = false, temp = false] : memref<128x256xf32> -> !loom.view
              loom.copy_from_tensor %23, %26 : tensor<?x?xf32>, !loom.view
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
      loom.polynomial_constraint(%12, %13, %14) {monomials = [{coeff = 4 : i64, vars = [0, 1]}, {coeff = 4 : i64, vars = [0, 2]}, {coeff = 4 : i64, vars = [2, 1]}], upper_bound = 1499136 : i64}
      loom.polynomial_constraint(%12) {monomials = [{coeff = 64 : i64, vars = [0]}], upper_bound = 1024 : i64}
    }
    func.func @matmul__d0i0_d1i0__f10(%arg0: memref<128x128xf32>, %arg1: memref<128x256xf32>, %arg2: memref<128x256xf32>) {
      %cst = arith.constant 0.000000e+00 : f32
      %12 = loom.get_symbolic_block_size @constraints::@BM : index
      %13 = loom.get_symbolic_block_size @constraints::@BN : index
      %14 = loom.get_symbolic_block_size @constraints::@BK : index
      affine.parallel (%arg3) = (0) to (8) {
        affine.parallel (%arg4) = (0) to (8) {
          affine.for %arg5 = 0 to affine_map<()[s0, s1] -> (256 ceildiv s1)>()[%12, %13] {
            affine.for %arg6 = 0 to affine_map<()[s0, s1] -> (((128 ceildiv s0) ceildiv 8) ceildiv 8)>()[%12, %13] {
              %15 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 8 + d1 + d2 * 64)>(%arg3, %arg4, %arg6)
              %16 = loom.alloc(%12, %13) on @L1 : !loom.buffer_token
              %17 = loom.init_tensor %16(%12, %13) : !loom.buffer_token -> tensor<?x?xf32>
              %18 = linalg.fill ins(%cst : f32) outs(%17 : tensor<?x?xf32>) -> tensor<?x?xf32>
              %19 = affine.for %arg7 = 0 to affine_map<()[s0] -> (128 ceildiv s0)>()[%14] iter_args(%arg8 = %18) -> (tensor<?x?xf32>) {
                %23 = arith.muli %15, %12 : index
                %24 = arith.muli %arg7, %14 : index
                %25 = loom.view %arg0[%23, %24] [%12, %14] [1, 1], reuse : [seq = false, spat = false, temp = false] : memref<128x128xf32> -> !loom.view
                %26 = loom.alloc(%12, %14) on @L1 : !loom.buffer_token
                %27 = loom.copy_to_tensor %25, %26, interconnect : [], broadcast : [1, 1] : !loom.view, !loom.buffer_token -> tensor<?x?xf32>
                %28 = arith.muli %arg5, %13 : index
                %29 = loom.view %arg1[%24, %28] [%14, %13] [1, 1], reuse : [seq = false, spat = false, temp = false] : memref<128x256xf32> -> !loom.view
                %30 = loom.alloc(%14, %13) on @L1 : !loom.buffer_token
                %31 = loom.copy_to_tensor %29, %30, interconnect : [], broadcast : [1, 1] : !loom.view, !loom.buffer_token -> tensor<?x?xf32>
                %32 = linalg.matmul ins(%27, %31 : tensor<?x?xf32>, tensor<?x?xf32>) outs(%18 : tensor<?x?xf32>) -> tensor<?x?xf32>
                %33 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%arg8, %32 : tensor<?x?xf32>, tensor<?x?xf32>) outs(%17 : tensor<?x?xf32>) {
                ^bb0(%in: f32, %in_0: f32, %out: f32):
                  %34 = arith.addf %in, %in_0 : f32
                  linalg.yield %34 : f32
                } -> tensor<?x?xf32>
                affine.yield %33 : tensor<?x?xf32>
              } {loom.iter_type = #loom.iter_type<sequential>}
              %20 = arith.muli %15, %12 : index
              %21 = arith.muli %arg5, %13 : index
              %22 = loom.view %arg2[%20, %21] [%12, %13] [1, 1], reuse : [seq = false, spat = false, temp = false] : memref<128x256xf32> -> !loom.view
              loom.copy_from_tensor %19, %22 : tensor<?x?xf32>, !loom.view
            } {loom.iter_type = #loom.iter_type<temporal>}
          } {loom.iter_type = #loom.iter_type<temporal>}
        } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @y}
      } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @x}
      return
    }
  }
  module attributes {loom.pass_name = "HoistBlockLoading"} {
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
      loom.polynomial_constraint(%12, %13, %14) {monomials = [{coeff = 4 : i64, vars = [0, 1]}, {coeff = 4 : i64, vars = [0, 2]}, {coeff = 4 : i64, vars = [2, 1]}], upper_bound = 1499136 : i64}
      loom.polynomial_constraint(%12) {monomials = [{coeff = 64 : i64, vars = [0]}], upper_bound = 1024 : i64}
      loom.polynomial_constraint(%12, %13, %14) {monomials = [{coeff = 4 : i64, vars = [0, 1]}, {coeff = 4 : i64, vars = [2, 1]}], upper_bound = 1499136 : i64}
    }
    func.func @matmul__d0i0_d1i0__f10__hoist_block_0(%arg0: memref<128x128xf32>, %arg1: memref<128x256xf32>, %arg2: memref<128x256xf32>) {
      %cst = arith.constant 0.000000e+00 : f32
      %12 = loom.get_symbolic_block_size @constraints::@BM : index
      %13 = loom.get_symbolic_block_size @constraints::@BN : index
      %14 = loom.get_symbolic_block_size @constraints::@BK : index
      affine.parallel (%arg3) = (0) to (8) {
        affine.parallel (%arg4) = (0) to (8) {
          affine.for %arg5 = 0 to affine_map<()[s0, s1] -> (256 ceildiv s1)>()[%12, %13] {
            affine.for %arg6 = 0 to affine_map<()[s0, s1] -> (((128 ceildiv s0) ceildiv 8) ceildiv 8)>()[%12, %13] {
              %15 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 8 + d1 + d2 * 64)>(%arg3, %arg4, %arg6)
              %16 = loom.alloc(%12, %13) on @L1 : !loom.buffer_token
              %17 = loom.init_tensor %16(%12, %13) : !loom.buffer_token -> tensor<?x?xf32>
              %18 = linalg.fill ins(%cst : f32) outs(%17 : tensor<?x?xf32>) -> tensor<?x?xf32>
              %19 = arith.muli %15, %12 : index
              %c0 = arith.constant 0 : index
              %c128 = arith.constant 128 : index
              %20 = loom.view %arg0[%19, %c0] [%12, %c128] [1, 1], reuse : [seq = false, spat = false, temp = false] : memref<128x128xf32> -> !loom.view
              %21 = loom.alloc(%12, %c128) on @L1 : !loom.buffer_token
              %22 = loom.pack_to_tensor %20, %21 inner_tiles = [%12, %14] outer_dims_perm = [1, 0] : !loom.view, !loom.buffer_token -> tensor<?x?x?xf32>
              %23 = affine.for %arg7 = 0 to affine_map<()[s0] -> (128 ceildiv s0)>()[%14] iter_args(%arg8 = %18) -> (tensor<?x?xf32>) {
                %27 = arith.muli %15, %12 : index
                %28 = arith.muli %arg7, %14 : index
                %extracted_slice = tensor.extract_slice %22[%arg7, 0, 0] [1, %12, %14] [1, 1, 1] : tensor<?x?x?xf32> to tensor<1x?x?xf32>
                %collapsed = tensor.collapse_shape %extracted_slice [[0, 1], [2]] : tensor<1x?x?xf32> into tensor<?x?xf32>
                %29 = arith.muli %arg5, %13 : index
                %30 = loom.view %arg1[%28, %29] [%14, %13] [1, 1], reuse : [seq = false, spat = false, temp = false] : memref<128x256xf32> -> !loom.view
                %31 = loom.alloc(%14, %13) on @L1 : !loom.buffer_token
                %32 = loom.copy_to_tensor %30, %31, interconnect : [], broadcast : [1, 1] : !loom.view, !loom.buffer_token -> tensor<?x?xf32>
                %33 = linalg.matmul ins(%collapsed, %32 : tensor<?x?xf32>, tensor<?x?xf32>) outs(%18 : tensor<?x?xf32>) -> tensor<?x?xf32>
                %34 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%arg8, %33 : tensor<?x?xf32>, tensor<?x?xf32>) outs(%17 : tensor<?x?xf32>) {
                ^bb0(%in: f32, %in_0: f32, %out: f32):
                  %35 = arith.addf %in, %in_0 : f32
                  linalg.yield %35 : f32
                } -> tensor<?x?xf32>
                affine.yield %34 : tensor<?x?xf32>
              } {loom.iter_type = #loom.iter_type<sequential>}
              %24 = arith.muli %15, %12 : index
              %25 = arith.muli %arg5, %13 : index
              %26 = loom.view %arg2[%24, %25] [%12, %13] [1, 1], reuse : [seq = false, spat = false, temp = false] : memref<128x256xf32> -> !loom.view
              loom.copy_from_tensor %23, %26 : tensor<?x?xf32>, !loom.view
            } {loom.iter_type = #loom.iter_type<temporal>}
          } {loom.iter_type = #loom.iter_type<temporal>}
        } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @y}
      } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @x}
      return
    }
  }
  module attributes {loom.pass_name = "HoistBlockLoading"} {
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
      loom.polynomial_constraint(%12, %13, %14) {monomials = [{coeff = 4 : i64, vars = [0, 1]}, {coeff = 4 : i64, vars = [0, 2]}, {coeff = 4 : i64, vars = [2, 1]}], upper_bound = 1499136 : i64}
      loom.polynomial_constraint(%12) {monomials = [{coeff = 64 : i64, vars = [0]}], upper_bound = 1024 : i64}
      loom.polynomial_constraint(%12, %13, %14) {monomials = [{coeff = 4 : i64, vars = [0, 1]}, {coeff = 4 : i64, vars = [0, 2]}], upper_bound = 1499136 : i64}
    }
    func.func @matmul__d0i0_d1i0__f10__hoist_block_1(%arg0: memref<128x128xf32>, %arg1: memref<128x256xf32>, %arg2: memref<128x256xf32>) {
      %cst = arith.constant 0.000000e+00 : f32
      %12 = loom.get_symbolic_block_size @constraints::@BM : index
      %13 = loom.get_symbolic_block_size @constraints::@BN : index
      %14 = loom.get_symbolic_block_size @constraints::@BK : index
      affine.parallel (%arg3) = (0) to (8) {
        affine.parallel (%arg4) = (0) to (8) {
          affine.for %arg5 = 0 to affine_map<()[s0, s1] -> (256 ceildiv s1)>()[%12, %13] {
            affine.for %arg6 = 0 to affine_map<()[s0, s1] -> (((128 ceildiv s0) ceildiv 8) ceildiv 8)>()[%12, %13] {
              %15 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 8 + d1 + d2 * 64)>(%arg3, %arg4, %arg6)
              %16 = loom.alloc(%12, %13) on @L1 : !loom.buffer_token
              %17 = loom.init_tensor %16(%12, %13) : !loom.buffer_token -> tensor<?x?xf32>
              %18 = linalg.fill ins(%cst : f32) outs(%17 : tensor<?x?xf32>) -> tensor<?x?xf32>
              %c0 = arith.constant 0 : index
              %19 = arith.muli %arg5, %13 : index
              %c128 = arith.constant 128 : index
              %20 = loom.view %arg1[%c0, %19] [%c128, %13] [1, 1], reuse : [seq = false, spat = false, temp = false] : memref<128x256xf32> -> !loom.view
              %21 = loom.alloc(%c128, %13) on @L1 : !loom.buffer_token
              %22 = loom.pack_to_tensor %20, %21 inner_tiles = [%14, %13] outer_dims_perm = [0, 1] : !loom.view, !loom.buffer_token -> tensor<?x?x?xf32>
              %23 = affine.for %arg7 = 0 to affine_map<()[s0] -> (128 ceildiv s0)>()[%14] iter_args(%arg8 = %18) -> (tensor<?x?xf32>) {
                %27 = arith.muli %15, %12 : index
                %28 = arith.muli %arg7, %14 : index
                %29 = loom.view %arg0[%27, %28] [%12, %14] [1, 1], reuse : [seq = false, spat = false, temp = false] : memref<128x128xf32> -> !loom.view
                %30 = loom.alloc(%12, %14) on @L1 : !loom.buffer_token
                %31 = loom.copy_to_tensor %29, %30, interconnect : [], broadcast : [1, 1] : !loom.view, !loom.buffer_token -> tensor<?x?xf32>
                %32 = arith.muli %arg5, %13 : index
                %extracted_slice = tensor.extract_slice %22[%arg7, 0, 0] [1, %14, %13] [1, 1, 1] : tensor<?x?x?xf32> to tensor<1x?x?xf32>
                %collapsed = tensor.collapse_shape %extracted_slice [[0, 1], [2]] : tensor<1x?x?xf32> into tensor<?x?xf32>
                %33 = linalg.matmul ins(%31, %collapsed : tensor<?x?xf32>, tensor<?x?xf32>) outs(%18 : tensor<?x?xf32>) -> tensor<?x?xf32>
                %34 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%arg8, %33 : tensor<?x?xf32>, tensor<?x?xf32>) outs(%17 : tensor<?x?xf32>) {
                ^bb0(%in: f32, %in_0: f32, %out: f32):
                  %35 = arith.addf %in, %in_0 : f32
                  linalg.yield %35 : f32
                } -> tensor<?x?xf32>
                affine.yield %34 : tensor<?x?xf32>
              } {loom.iter_type = #loom.iter_type<sequential>}
              %24 = arith.muli %15, %12 : index
              %25 = arith.muli %arg5, %13 : index
              %26 = loom.view %arg2[%24, %25] [%12, %13] [1, 1], reuse : [seq = false, spat = false, temp = false] : memref<128x256xf32> -> !loom.view
              loom.copy_from_tensor %23, %26 : tensor<?x?xf32>, !loom.view
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
      loom.polynomial_constraint(%12, %13, %14) {monomials = [{coeff = 4 : i64, vars = [0, 1]}, {coeff = 4 : i64, vars = [0, 2]}, {coeff = 4 : i64, vars = [2, 1]}], upper_bound = 1499136 : i64}
      loom.polynomial_constraint(%12) {monomials = [{coeff = 8 : i64, vars = [0]}], upper_bound = 1024 : i64}
      loom.polynomial_constraint(%13) {monomials = [{coeff = 8 : i64, vars = [0]}], upper_bound = 1024 : i64}
    }
    func.func @matmul__d0i0_d1i1__f01(%arg0: memref<128x128xf32>, %arg1: memref<128x256xf32>, %arg2: memref<128x256xf32>) {
      %cst = arith.constant 0.000000e+00 : f32
      %12 = loom.get_symbolic_block_size @constraints::@BM : index
      %13 = loom.get_symbolic_block_size @constraints::@BN : index
      %14 = loom.get_symbolic_block_size @constraints::@BK : index
      affine.parallel (%arg3) = (0) to (8) {
        affine.parallel (%arg4) = (0) to (8) {
          affine.for %arg5 = 0 to affine_map<()[s0, s1] -> ((128 ceildiv s0) ceildiv 8)>()[%12, %13] {
            affine.for %arg6 = 0 to affine_map<()[s0, s1] -> ((256 ceildiv s1) ceildiv 8)>()[%12, %13] {
              %15 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 8)>(%arg3, %arg5)
              %16 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 8)>(%arg4, %arg6)
              %17 = loom.alloc(%12, %13) on @L1 : !loom.buffer_token
              %18 = loom.init_tensor %17(%12, %13) : !loom.buffer_token -> tensor<?x?xf32>
              %19 = linalg.fill ins(%cst : f32) outs(%18 : tensor<?x?xf32>) -> tensor<?x?xf32>
              %20 = affine.for %arg7 = 0 to affine_map<()[s0] -> (128 ceildiv s0)>()[%14] iter_args(%arg8 = %19) -> (tensor<?x?xf32>) {
                %24 = arith.muli %15, %12 : index
                %25 = arith.muli %arg7, %14 : index
                %26 = loom.view %arg0[%24, %25] [%12, %14] [1, 1], reuse : [seq = false, spat = false, temp = false] : memref<128x128xf32> -> !loom.view
                %27 = loom.alloc(%12, %14) on @L1 : !loom.buffer_token
                %28 = loom.copy_to_tensor %26, %27, interconnect : [], broadcast : [1, 1] : !loom.view, !loom.buffer_token -> tensor<?x?xf32>
                %29 = arith.muli %16, %13 : index
                %30 = loom.view %arg1[%25, %29] [%14, %13] [1, 1], reuse : [seq = false, spat = false, temp = false] : memref<128x256xf32> -> !loom.view
                %31 = loom.alloc(%14, %13) on @L1 : !loom.buffer_token
                %32 = loom.copy_to_tensor %30, %31, interconnect : [], broadcast : [1, 1] : !loom.view, !loom.buffer_token -> tensor<?x?xf32>
                %33 = linalg.matmul ins(%28, %32 : tensor<?x?xf32>, tensor<?x?xf32>) outs(%19 : tensor<?x?xf32>) -> tensor<?x?xf32>
                %34 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%arg8, %33 : tensor<?x?xf32>, tensor<?x?xf32>) outs(%18 : tensor<?x?xf32>) {
                ^bb0(%in: f32, %in_0: f32, %out: f32):
                  %35 = arith.addf %in, %in_0 : f32
                  linalg.yield %35 : f32
                } -> tensor<?x?xf32>
                affine.yield %34 : tensor<?x?xf32>
              } {loom.iter_type = #loom.iter_type<sequential>}
              %21 = arith.muli %15, %12 : index
              %22 = arith.muli %16, %13 : index
              %23 = loom.view %arg2[%21, %22] [%12, %13] [1, 1], reuse : [seq = false, spat = false, temp = false] : memref<128x256xf32> -> !loom.view
              loom.copy_from_tensor %20, %23 : tensor<?x?xf32>, !loom.view
            } {loom.iter_type = #loom.iter_type<temporal>}
          } {loom.iter_type = #loom.iter_type<temporal>}
        } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @y}
      } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @x}
      return
    }
  }
  module attributes {loom.pass_name = "HoistBlockLoading"} {
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
      loom.polynomial_constraint(%12, %13, %14) {monomials = [{coeff = 4 : i64, vars = [0, 1]}, {coeff = 4 : i64, vars = [0, 2]}, {coeff = 4 : i64, vars = [2, 1]}], upper_bound = 1499136 : i64}
      loom.polynomial_constraint(%12) {monomials = [{coeff = 8 : i64, vars = [0]}], upper_bound = 1024 : i64}
      loom.polynomial_constraint(%13) {monomials = [{coeff = 8 : i64, vars = [0]}], upper_bound = 1024 : i64}
      loom.polynomial_constraint(%12, %13, %14) {monomials = [{coeff = 4 : i64, vars = [0, 1]}, {coeff = 4 : i64, vars = [2, 1]}], upper_bound = 1499136 : i64}
    }
    func.func @matmul__d0i0_d1i1__f01__hoist_block_0(%arg0: memref<128x128xf32>, %arg1: memref<128x256xf32>, %arg2: memref<128x256xf32>) {
      %cst = arith.constant 0.000000e+00 : f32
      %12 = loom.get_symbolic_block_size @constraints::@BM : index
      %13 = loom.get_symbolic_block_size @constraints::@BN : index
      %14 = loom.get_symbolic_block_size @constraints::@BK : index
      affine.parallel (%arg3) = (0) to (8) {
        affine.parallel (%arg4) = (0) to (8) {
          affine.for %arg5 = 0 to affine_map<()[s0, s1] -> ((128 ceildiv s0) ceildiv 8)>()[%12, %13] {
            affine.for %arg6 = 0 to affine_map<()[s0, s1] -> ((256 ceildiv s1) ceildiv 8)>()[%12, %13] {
              %15 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 8)>(%arg3, %arg5)
              %16 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 8)>(%arg4, %arg6)
              %17 = loom.alloc(%12, %13) on @L1 : !loom.buffer_token
              %18 = loom.init_tensor %17(%12, %13) : !loom.buffer_token -> tensor<?x?xf32>
              %19 = linalg.fill ins(%cst : f32) outs(%18 : tensor<?x?xf32>) -> tensor<?x?xf32>
              %20 = arith.muli %15, %12 : index
              %c0 = arith.constant 0 : index
              %c128 = arith.constant 128 : index
              %21 = loom.view %arg0[%20, %c0] [%12, %c128] [1, 1], reuse : [seq = false, spat = false, temp = false] : memref<128x128xf32> -> !loom.view
              %22 = loom.alloc(%12, %c128) on @L1 : !loom.buffer_token
              %23 = loom.pack_to_tensor %21, %22 inner_tiles = [%12, %14] outer_dims_perm = [1, 0] : !loom.view, !loom.buffer_token -> tensor<?x?x?xf32>
              %24 = affine.for %arg7 = 0 to affine_map<()[s0] -> (128 ceildiv s0)>()[%14] iter_args(%arg8 = %19) -> (tensor<?x?xf32>) {
                %28 = arith.muli %15, %12 : index
                %29 = arith.muli %arg7, %14 : index
                %extracted_slice = tensor.extract_slice %23[%arg7, 0, 0] [1, %12, %14] [1, 1, 1] : tensor<?x?x?xf32> to tensor<1x?x?xf32>
                %collapsed = tensor.collapse_shape %extracted_slice [[0, 1], [2]] : tensor<1x?x?xf32> into tensor<?x?xf32>
                %30 = arith.muli %16, %13 : index
                %31 = loom.view %arg1[%29, %30] [%14, %13] [1, 1], reuse : [seq = false, spat = false, temp = false] : memref<128x256xf32> -> !loom.view
                %32 = loom.alloc(%14, %13) on @L1 : !loom.buffer_token
                %33 = loom.copy_to_tensor %31, %32, interconnect : [], broadcast : [1, 1] : !loom.view, !loom.buffer_token -> tensor<?x?xf32>
                %34 = linalg.matmul ins(%collapsed, %33 : tensor<?x?xf32>, tensor<?x?xf32>) outs(%19 : tensor<?x?xf32>) -> tensor<?x?xf32>
                %35 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%arg8, %34 : tensor<?x?xf32>, tensor<?x?xf32>) outs(%18 : tensor<?x?xf32>) {
                ^bb0(%in: f32, %in_0: f32, %out: f32):
                  %36 = arith.addf %in, %in_0 : f32
                  linalg.yield %36 : f32
                } -> tensor<?x?xf32>
                affine.yield %35 : tensor<?x?xf32>
              } {loom.iter_type = #loom.iter_type<sequential>}
              %25 = arith.muli %15, %12 : index
              %26 = arith.muli %16, %13 : index
              %27 = loom.view %arg2[%25, %26] [%12, %13] [1, 1], reuse : [seq = false, spat = false, temp = false] : memref<128x256xf32> -> !loom.view
              loom.copy_from_tensor %24, %27 : tensor<?x?xf32>, !loom.view
            } {loom.iter_type = #loom.iter_type<temporal>}
          } {loom.iter_type = #loom.iter_type<temporal>}
        } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @y}
      } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @x}
      return
    }
  }
  module attributes {loom.pass_name = "HoistBlockLoading"} {
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
      loom.polynomial_constraint(%12, %13, %14) {monomials = [{coeff = 4 : i64, vars = [0, 1]}, {coeff = 4 : i64, vars = [0, 2]}, {coeff = 4 : i64, vars = [2, 1]}], upper_bound = 1499136 : i64}
      loom.polynomial_constraint(%12) {monomials = [{coeff = 8 : i64, vars = [0]}], upper_bound = 1024 : i64}
      loom.polynomial_constraint(%13) {monomials = [{coeff = 8 : i64, vars = [0]}], upper_bound = 1024 : i64}
      loom.polynomial_constraint(%12, %13, %14) {monomials = [{coeff = 4 : i64, vars = [0, 1]}, {coeff = 4 : i64, vars = [0, 2]}], upper_bound = 1499136 : i64}
    }
    func.func @matmul__d0i0_d1i1__f01__hoist_block_1(%arg0: memref<128x128xf32>, %arg1: memref<128x256xf32>, %arg2: memref<128x256xf32>) {
      %cst = arith.constant 0.000000e+00 : f32
      %12 = loom.get_symbolic_block_size @constraints::@BM : index
      %13 = loom.get_symbolic_block_size @constraints::@BN : index
      %14 = loom.get_symbolic_block_size @constraints::@BK : index
      affine.parallel (%arg3) = (0) to (8) {
        affine.parallel (%arg4) = (0) to (8) {
          affine.for %arg5 = 0 to affine_map<()[s0, s1] -> ((128 ceildiv s0) ceildiv 8)>()[%12, %13] {
            affine.for %arg6 = 0 to affine_map<()[s0, s1] -> ((256 ceildiv s1) ceildiv 8)>()[%12, %13] {
              %15 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 8)>(%arg3, %arg5)
              %16 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 8)>(%arg4, %arg6)
              %17 = loom.alloc(%12, %13) on @L1 : !loom.buffer_token
              %18 = loom.init_tensor %17(%12, %13) : !loom.buffer_token -> tensor<?x?xf32>
              %19 = linalg.fill ins(%cst : f32) outs(%18 : tensor<?x?xf32>) -> tensor<?x?xf32>
              %c0 = arith.constant 0 : index
              %20 = arith.muli %16, %13 : index
              %c128 = arith.constant 128 : index
              %21 = loom.view %arg1[%c0, %20] [%c128, %13] [1, 1], reuse : [seq = false, spat = false, temp = false] : memref<128x256xf32> -> !loom.view
              %22 = loom.alloc(%c128, %13) on @L1 : !loom.buffer_token
              %23 = loom.pack_to_tensor %21, %22 inner_tiles = [%14, %13] outer_dims_perm = [0, 1] : !loom.view, !loom.buffer_token -> tensor<?x?x?xf32>
              %24 = affine.for %arg7 = 0 to affine_map<()[s0] -> (128 ceildiv s0)>()[%14] iter_args(%arg8 = %19) -> (tensor<?x?xf32>) {
                %28 = arith.muli %15, %12 : index
                %29 = arith.muli %arg7, %14 : index
                %30 = loom.view %arg0[%28, %29] [%12, %14] [1, 1], reuse : [seq = false, spat = false, temp = false] : memref<128x128xf32> -> !loom.view
                %31 = loom.alloc(%12, %14) on @L1 : !loom.buffer_token
                %32 = loom.copy_to_tensor %30, %31, interconnect : [], broadcast : [1, 1] : !loom.view, !loom.buffer_token -> tensor<?x?xf32>
                %33 = arith.muli %16, %13 : index
                %extracted_slice = tensor.extract_slice %23[%arg7, 0, 0] [1, %14, %13] [1, 1, 1] : tensor<?x?x?xf32> to tensor<1x?x?xf32>
                %collapsed = tensor.collapse_shape %extracted_slice [[0, 1], [2]] : tensor<1x?x?xf32> into tensor<?x?xf32>
                %34 = linalg.matmul ins(%32, %collapsed : tensor<?x?xf32>, tensor<?x?xf32>) outs(%19 : tensor<?x?xf32>) -> tensor<?x?xf32>
                %35 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%arg8, %34 : tensor<?x?xf32>, tensor<?x?xf32>) outs(%18 : tensor<?x?xf32>) {
                ^bb0(%in: f32, %in_0: f32, %out: f32):
                  %36 = arith.addf %in, %in_0 : f32
                  linalg.yield %36 : f32
                } -> tensor<?x?xf32>
                affine.yield %35 : tensor<?x?xf32>
              } {loom.iter_type = #loom.iter_type<sequential>}
              %25 = arith.muli %15, %12 : index
              %26 = arith.muli %16, %13 : index
              %27 = loom.view %arg2[%25, %26] [%12, %13] [1, 1], reuse : [seq = false, spat = false, temp = false] : memref<128x256xf32> -> !loom.view
              loom.copy_from_tensor %24, %27 : tensor<?x?xf32>, !loom.view
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
      loom.polynomial_constraint(%12, %13, %14) {monomials = [{coeff = 4 : i64, vars = [0, 1]}, {coeff = 4 : i64, vars = [0, 2]}, {coeff = 4 : i64, vars = [2, 1]}], upper_bound = 1499136 : i64}
      loom.polynomial_constraint(%12) {monomials = [{coeff = 8 : i64, vars = [0]}], upper_bound = 1024 : i64}
      loom.polynomial_constraint(%13) {monomials = [{coeff = 8 : i64, vars = [0]}], upper_bound = 1024 : i64}
    }
    func.func @matmul__d0i0_d1i1__f10(%arg0: memref<128x128xf32>, %arg1: memref<128x256xf32>, %arg2: memref<128x256xf32>) {
      %cst = arith.constant 0.000000e+00 : f32
      %12 = loom.get_symbolic_block_size @constraints::@BM : index
      %13 = loom.get_symbolic_block_size @constraints::@BN : index
      %14 = loom.get_symbolic_block_size @constraints::@BK : index
      affine.parallel (%arg3) = (0) to (8) {
        affine.parallel (%arg4) = (0) to (8) {
          affine.for %arg5 = 0 to affine_map<()[s0, s1] -> ((256 ceildiv s1) ceildiv 8)>()[%12, %13] {
            affine.for %arg6 = 0 to affine_map<()[s0, s1] -> ((128 ceildiv s0) ceildiv 8)>()[%12, %13] {
              %15 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 8)>(%arg3, %arg6)
              %16 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 8)>(%arg4, %arg5)
              %17 = loom.alloc(%12, %13) on @L1 : !loom.buffer_token
              %18 = loom.init_tensor %17(%12, %13) : !loom.buffer_token -> tensor<?x?xf32>
              %19 = linalg.fill ins(%cst : f32) outs(%18 : tensor<?x?xf32>) -> tensor<?x?xf32>
              %20 = affine.for %arg7 = 0 to affine_map<()[s0] -> (128 ceildiv s0)>()[%14] iter_args(%arg8 = %19) -> (tensor<?x?xf32>) {
                %24 = arith.muli %15, %12 : index
                %25 = arith.muli %arg7, %14 : index
                %26 = loom.view %arg0[%24, %25] [%12, %14] [1, 1], reuse : [seq = false, spat = false, temp = false] : memref<128x128xf32> -> !loom.view
                %27 = loom.alloc(%12, %14) on @L1 : !loom.buffer_token
                %28 = loom.copy_to_tensor %26, %27, interconnect : [], broadcast : [1, 1] : !loom.view, !loom.buffer_token -> tensor<?x?xf32>
                %29 = arith.muli %16, %13 : index
                %30 = loom.view %arg1[%25, %29] [%14, %13] [1, 1], reuse : [seq = false, spat = false, temp = false] : memref<128x256xf32> -> !loom.view
                %31 = loom.alloc(%14, %13) on @L1 : !loom.buffer_token
                %32 = loom.copy_to_tensor %30, %31, interconnect : [], broadcast : [1, 1] : !loom.view, !loom.buffer_token -> tensor<?x?xf32>
                %33 = linalg.matmul ins(%28, %32 : tensor<?x?xf32>, tensor<?x?xf32>) outs(%19 : tensor<?x?xf32>) -> tensor<?x?xf32>
                %34 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%arg8, %33 : tensor<?x?xf32>, tensor<?x?xf32>) outs(%18 : tensor<?x?xf32>) {
                ^bb0(%in: f32, %in_0: f32, %out: f32):
                  %35 = arith.addf %in, %in_0 : f32
                  linalg.yield %35 : f32
                } -> tensor<?x?xf32>
                affine.yield %34 : tensor<?x?xf32>
              } {loom.iter_type = #loom.iter_type<sequential>}
              %21 = arith.muli %15, %12 : index
              %22 = arith.muli %16, %13 : index
              %23 = loom.view %arg2[%21, %22] [%12, %13] [1, 1], reuse : [seq = false, spat = false, temp = false] : memref<128x256xf32> -> !loom.view
              loom.copy_from_tensor %20, %23 : tensor<?x?xf32>, !loom.view
            } {loom.iter_type = #loom.iter_type<temporal>}
          } {loom.iter_type = #loom.iter_type<temporal>}
        } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @y}
      } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @x}
      return
    }
  }
  module attributes {loom.pass_name = "HoistBlockLoading"} {
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
      loom.polynomial_constraint(%12, %13, %14) {monomials = [{coeff = 4 : i64, vars = [0, 1]}, {coeff = 4 : i64, vars = [0, 2]}, {coeff = 4 : i64, vars = [2, 1]}], upper_bound = 1499136 : i64}
      loom.polynomial_constraint(%12) {monomials = [{coeff = 8 : i64, vars = [0]}], upper_bound = 1024 : i64}
      loom.polynomial_constraint(%13) {monomials = [{coeff = 8 : i64, vars = [0]}], upper_bound = 1024 : i64}
      loom.polynomial_constraint(%12, %13, %14) {monomials = [{coeff = 4 : i64, vars = [0, 1]}, {coeff = 4 : i64, vars = [2, 1]}], upper_bound = 1499136 : i64}
    }
    func.func @matmul__d0i0_d1i1__f10__hoist_block_0(%arg0: memref<128x128xf32>, %arg1: memref<128x256xf32>, %arg2: memref<128x256xf32>) {
      %cst = arith.constant 0.000000e+00 : f32
      %12 = loom.get_symbolic_block_size @constraints::@BM : index
      %13 = loom.get_symbolic_block_size @constraints::@BN : index
      %14 = loom.get_symbolic_block_size @constraints::@BK : index
      affine.parallel (%arg3) = (0) to (8) {
        affine.parallel (%arg4) = (0) to (8) {
          affine.for %arg5 = 0 to affine_map<()[s0, s1] -> ((256 ceildiv s1) ceildiv 8)>()[%12, %13] {
            affine.for %arg6 = 0 to affine_map<()[s0, s1] -> ((128 ceildiv s0) ceildiv 8)>()[%12, %13] {
              %15 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 8)>(%arg3, %arg6)
              %16 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 8)>(%arg4, %arg5)
              %17 = loom.alloc(%12, %13) on @L1 : !loom.buffer_token
              %18 = loom.init_tensor %17(%12, %13) : !loom.buffer_token -> tensor<?x?xf32>
              %19 = linalg.fill ins(%cst : f32) outs(%18 : tensor<?x?xf32>) -> tensor<?x?xf32>
              %20 = arith.muli %15, %12 : index
              %c0 = arith.constant 0 : index
              %c128 = arith.constant 128 : index
              %21 = loom.view %arg0[%20, %c0] [%12, %c128] [1, 1], reuse : [seq = false, spat = false, temp = false] : memref<128x128xf32> -> !loom.view
              %22 = loom.alloc(%12, %c128) on @L1 : !loom.buffer_token
              %23 = loom.pack_to_tensor %21, %22 inner_tiles = [%12, %14] outer_dims_perm = [1, 0] : !loom.view, !loom.buffer_token -> tensor<?x?x?xf32>
              %24 = affine.for %arg7 = 0 to affine_map<()[s0] -> (128 ceildiv s0)>()[%14] iter_args(%arg8 = %19) -> (tensor<?x?xf32>) {
                %28 = arith.muli %15, %12 : index
                %29 = arith.muli %arg7, %14 : index
                %extracted_slice = tensor.extract_slice %23[%arg7, 0, 0] [1, %12, %14] [1, 1, 1] : tensor<?x?x?xf32> to tensor<1x?x?xf32>
                %collapsed = tensor.collapse_shape %extracted_slice [[0, 1], [2]] : tensor<1x?x?xf32> into tensor<?x?xf32>
                %30 = arith.muli %16, %13 : index
                %31 = loom.view %arg1[%29, %30] [%14, %13] [1, 1], reuse : [seq = false, spat = false, temp = false] : memref<128x256xf32> -> !loom.view
                %32 = loom.alloc(%14, %13) on @L1 : !loom.buffer_token
                %33 = loom.copy_to_tensor %31, %32, interconnect : [], broadcast : [1, 1] : !loom.view, !loom.buffer_token -> tensor<?x?xf32>
                %34 = linalg.matmul ins(%collapsed, %33 : tensor<?x?xf32>, tensor<?x?xf32>) outs(%19 : tensor<?x?xf32>) -> tensor<?x?xf32>
                %35 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%arg8, %34 : tensor<?x?xf32>, tensor<?x?xf32>) outs(%18 : tensor<?x?xf32>) {
                ^bb0(%in: f32, %in_0: f32, %out: f32):
                  %36 = arith.addf %in, %in_0 : f32
                  linalg.yield %36 : f32
                } -> tensor<?x?xf32>
                affine.yield %35 : tensor<?x?xf32>
              } {loom.iter_type = #loom.iter_type<sequential>}
              %25 = arith.muli %15, %12 : index
              %26 = arith.muli %16, %13 : index
              %27 = loom.view %arg2[%25, %26] [%12, %13] [1, 1], reuse : [seq = false, spat = false, temp = false] : memref<128x256xf32> -> !loom.view
              loom.copy_from_tensor %24, %27 : tensor<?x?xf32>, !loom.view
            } {loom.iter_type = #loom.iter_type<temporal>}
          } {loom.iter_type = #loom.iter_type<temporal>}
        } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @y}
      } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @x}
      return
    }
  }
  module attributes {loom.pass_name = "HoistBlockLoading"} {
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
      loom.polynomial_constraint(%12, %13, %14) {monomials = [{coeff = 4 : i64, vars = [0, 1]}, {coeff = 4 : i64, vars = [0, 2]}, {coeff = 4 : i64, vars = [2, 1]}], upper_bound = 1499136 : i64}
      loom.polynomial_constraint(%12) {monomials = [{coeff = 8 : i64, vars = [0]}], upper_bound = 1024 : i64}
      loom.polynomial_constraint(%13) {monomials = [{coeff = 8 : i64, vars = [0]}], upper_bound = 1024 : i64}
      loom.polynomial_constraint(%12, %13, %14) {monomials = [{coeff = 4 : i64, vars = [0, 1]}, {coeff = 4 : i64, vars = [0, 2]}], upper_bound = 1499136 : i64}
    }
    func.func @matmul__d0i0_d1i1__f10__hoist_block_1(%arg0: memref<128x128xf32>, %arg1: memref<128x256xf32>, %arg2: memref<128x256xf32>) {
      %cst = arith.constant 0.000000e+00 : f32
      %12 = loom.get_symbolic_block_size @constraints::@BM : index
      %13 = loom.get_symbolic_block_size @constraints::@BN : index
      %14 = loom.get_symbolic_block_size @constraints::@BK : index
      affine.parallel (%arg3) = (0) to (8) {
        affine.parallel (%arg4) = (0) to (8) {
          affine.for %arg5 = 0 to affine_map<()[s0, s1] -> ((256 ceildiv s1) ceildiv 8)>()[%12, %13] {
            affine.for %arg6 = 0 to affine_map<()[s0, s1] -> ((128 ceildiv s0) ceildiv 8)>()[%12, %13] {
              %15 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 8)>(%arg3, %arg6)
              %16 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 8)>(%arg4, %arg5)
              %17 = loom.alloc(%12, %13) on @L1 : !loom.buffer_token
              %18 = loom.init_tensor %17(%12, %13) : !loom.buffer_token -> tensor<?x?xf32>
              %19 = linalg.fill ins(%cst : f32) outs(%18 : tensor<?x?xf32>) -> tensor<?x?xf32>
              %c0 = arith.constant 0 : index
              %20 = arith.muli %16, %13 : index
              %c128 = arith.constant 128 : index
              %21 = loom.view %arg1[%c0, %20] [%c128, %13] [1, 1], reuse : [seq = false, spat = false, temp = false] : memref<128x256xf32> -> !loom.view
              %22 = loom.alloc(%c128, %13) on @L1 : !loom.buffer_token
              %23 = loom.pack_to_tensor %21, %22 inner_tiles = [%14, %13] outer_dims_perm = [0, 1] : !loom.view, !loom.buffer_token -> tensor<?x?x?xf32>
              %24 = affine.for %arg7 = 0 to affine_map<()[s0] -> (128 ceildiv s0)>()[%14] iter_args(%arg8 = %19) -> (tensor<?x?xf32>) {
                %28 = arith.muli %15, %12 : index
                %29 = arith.muli %arg7, %14 : index
                %30 = loom.view %arg0[%28, %29] [%12, %14] [1, 1], reuse : [seq = false, spat = false, temp = false] : memref<128x128xf32> -> !loom.view
                %31 = loom.alloc(%12, %14) on @L1 : !loom.buffer_token
                %32 = loom.copy_to_tensor %30, %31, interconnect : [], broadcast : [1, 1] : !loom.view, !loom.buffer_token -> tensor<?x?xf32>
                %33 = arith.muli %16, %13 : index
                %extracted_slice = tensor.extract_slice %23[%arg7, 0, 0] [1, %14, %13] [1, 1, 1] : tensor<?x?x?xf32> to tensor<1x?x?xf32>
                %collapsed = tensor.collapse_shape %extracted_slice [[0, 1], [2]] : tensor<1x?x?xf32> into tensor<?x?xf32>
                %34 = linalg.matmul ins(%32, %collapsed : tensor<?x?xf32>, tensor<?x?xf32>) outs(%19 : tensor<?x?xf32>) -> tensor<?x?xf32>
                %35 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%arg8, %34 : tensor<?x?xf32>, tensor<?x?xf32>) outs(%18 : tensor<?x?xf32>) {
                ^bb0(%in: f32, %in_0: f32, %out: f32):
                  %36 = arith.addf %in, %in_0 : f32
                  linalg.yield %36 : f32
                } -> tensor<?x?xf32>
                affine.yield %35 : tensor<?x?xf32>
              } {loom.iter_type = #loom.iter_type<sequential>}
              %25 = arith.muli %15, %12 : index
              %26 = arith.muli %16, %13 : index
              %27 = loom.view %arg2[%25, %26] [%12, %13] [1, 1], reuse : [seq = false, spat = false, temp = false] : memref<128x256xf32> -> !loom.view
              loom.copy_from_tensor %24, %27 : tensor<?x?xf32>, !loom.view
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
      loom.polynomial_constraint(%12, %13, %14) {monomials = [{coeff = 4 : i64, vars = [0, 1]}, {coeff = 4 : i64, vars = [0, 2]}, {coeff = 4 : i64, vars = [2, 1]}], upper_bound = 1499136 : i64}
      loom.polynomial_constraint(%12) {monomials = [{coeff = 8 : i64, vars = [0]}], upper_bound = 1024 : i64}
      loom.polynomial_constraint(%13) {monomials = [{coeff = 8 : i64, vars = [0]}], upper_bound = 1024 : i64}
    }
    func.func @matmul__d1i0_d0i1__f01(%arg0: memref<128x128xf32>, %arg1: memref<128x256xf32>, %arg2: memref<128x256xf32>) {
      %cst = arith.constant 0.000000e+00 : f32
      %12 = loom.get_symbolic_block_size @constraints::@BM : index
      %13 = loom.get_symbolic_block_size @constraints::@BN : index
      %14 = loom.get_symbolic_block_size @constraints::@BK : index
      affine.parallel (%arg3) = (0) to (8) {
        affine.parallel (%arg4) = (0) to (8) {
          affine.for %arg5 = 0 to affine_map<()[s0, s1] -> ((128 ceildiv s0) ceildiv 8)>()[%12, %13] {
            affine.for %arg6 = 0 to affine_map<()[s0, s1] -> ((256 ceildiv s1) ceildiv 8)>()[%12, %13] {
              %15 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 8)>(%arg3, %arg5)
              %16 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 8)>(%arg4, %arg6)
              %17 = loom.alloc(%12, %13) on @L1 : !loom.buffer_token
              %18 = loom.init_tensor %17(%12, %13) : !loom.buffer_token -> tensor<?x?xf32>
              %19 = linalg.fill ins(%cst : f32) outs(%18 : tensor<?x?xf32>) -> tensor<?x?xf32>
              %20 = affine.for %arg7 = 0 to affine_map<()[s0] -> (128 ceildiv s0)>()[%14] iter_args(%arg8 = %19) -> (tensor<?x?xf32>) {
                %24 = arith.muli %15, %12 : index
                %25 = arith.muli %arg7, %14 : index
                %26 = loom.view %arg0[%24, %25] [%12, %14] [1, 1], reuse : [seq = false, spat = false, temp = false] : memref<128x128xf32> -> !loom.view
                %27 = loom.alloc(%12, %14) on @L1 : !loom.buffer_token
                %28 = loom.copy_to_tensor %26, %27, interconnect : [], broadcast : [1, 1] : !loom.view, !loom.buffer_token -> tensor<?x?xf32>
                %29 = arith.muli %16, %13 : index
                %30 = loom.view %arg1[%25, %29] [%14, %13] [1, 1], reuse : [seq = false, spat = false, temp = false] : memref<128x256xf32> -> !loom.view
                %31 = loom.alloc(%14, %13) on @L1 : !loom.buffer_token
                %32 = loom.copy_to_tensor %30, %31, interconnect : [], broadcast : [1, 1] : !loom.view, !loom.buffer_token -> tensor<?x?xf32>
                %33 = linalg.matmul ins(%28, %32 : tensor<?x?xf32>, tensor<?x?xf32>) outs(%19 : tensor<?x?xf32>) -> tensor<?x?xf32>
                %34 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%arg8, %33 : tensor<?x?xf32>, tensor<?x?xf32>) outs(%18 : tensor<?x?xf32>) {
                ^bb0(%in: f32, %in_0: f32, %out: f32):
                  %35 = arith.addf %in, %in_0 : f32
                  linalg.yield %35 : f32
                } -> tensor<?x?xf32>
                affine.yield %34 : tensor<?x?xf32>
              } {loom.iter_type = #loom.iter_type<sequential>}
              %21 = arith.muli %15, %12 : index
              %22 = arith.muli %16, %13 : index
              %23 = loom.view %arg2[%21, %22] [%12, %13] [1, 1], reuse : [seq = false, spat = false, temp = false] : memref<128x256xf32> -> !loom.view
              loom.copy_from_tensor %20, %23 : tensor<?x?xf32>, !loom.view
            } {loom.iter_type = #loom.iter_type<temporal>}
          } {loom.iter_type = #loom.iter_type<temporal>}
        } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @x}
      } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @y}
      return
    }
  }
  module attributes {loom.pass_name = "HoistBlockLoading"} {
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
      loom.polynomial_constraint(%12, %13, %14) {monomials = [{coeff = 4 : i64, vars = [0, 1]}, {coeff = 4 : i64, vars = [0, 2]}, {coeff = 4 : i64, vars = [2, 1]}], upper_bound = 1499136 : i64}
      loom.polynomial_constraint(%12) {monomials = [{coeff = 8 : i64, vars = [0]}], upper_bound = 1024 : i64}
      loom.polynomial_constraint(%13) {monomials = [{coeff = 8 : i64, vars = [0]}], upper_bound = 1024 : i64}
      loom.polynomial_constraint(%12, %13, %14) {monomials = [{coeff = 4 : i64, vars = [0, 1]}, {coeff = 4 : i64, vars = [2, 1]}], upper_bound = 1499136 : i64}
    }
    func.func @matmul__d1i0_d0i1__f01__hoist_block_0(%arg0: memref<128x128xf32>, %arg1: memref<128x256xf32>, %arg2: memref<128x256xf32>) {
      %cst = arith.constant 0.000000e+00 : f32
      %12 = loom.get_symbolic_block_size @constraints::@BM : index
      %13 = loom.get_symbolic_block_size @constraints::@BN : index
      %14 = loom.get_symbolic_block_size @constraints::@BK : index
      affine.parallel (%arg3) = (0) to (8) {
        affine.parallel (%arg4) = (0) to (8) {
          affine.for %arg5 = 0 to affine_map<()[s0, s1] -> ((128 ceildiv s0) ceildiv 8)>()[%12, %13] {
            affine.for %arg6 = 0 to affine_map<()[s0, s1] -> ((256 ceildiv s1) ceildiv 8)>()[%12, %13] {
              %15 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 8)>(%arg3, %arg5)
              %16 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 8)>(%arg4, %arg6)
              %17 = loom.alloc(%12, %13) on @L1 : !loom.buffer_token
              %18 = loom.init_tensor %17(%12, %13) : !loom.buffer_token -> tensor<?x?xf32>
              %19 = linalg.fill ins(%cst : f32) outs(%18 : tensor<?x?xf32>) -> tensor<?x?xf32>
              %20 = arith.muli %15, %12 : index
              %c0 = arith.constant 0 : index
              %c128 = arith.constant 128 : index
              %21 = loom.view %arg0[%20, %c0] [%12, %c128] [1, 1], reuse : [seq = false, spat = false, temp = false] : memref<128x128xf32> -> !loom.view
              %22 = loom.alloc(%12, %c128) on @L1 : !loom.buffer_token
              %23 = loom.pack_to_tensor %21, %22 inner_tiles = [%12, %14] outer_dims_perm = [1, 0] : !loom.view, !loom.buffer_token -> tensor<?x?x?xf32>
              %24 = affine.for %arg7 = 0 to affine_map<()[s0] -> (128 ceildiv s0)>()[%14] iter_args(%arg8 = %19) -> (tensor<?x?xf32>) {
                %28 = arith.muli %15, %12 : index
                %29 = arith.muli %arg7, %14 : index
                %extracted_slice = tensor.extract_slice %23[%arg7, 0, 0] [1, %12, %14] [1, 1, 1] : tensor<?x?x?xf32> to tensor<1x?x?xf32>
                %collapsed = tensor.collapse_shape %extracted_slice [[0, 1], [2]] : tensor<1x?x?xf32> into tensor<?x?xf32>
                %30 = arith.muli %16, %13 : index
                %31 = loom.view %arg1[%29, %30] [%14, %13] [1, 1], reuse : [seq = false, spat = false, temp = false] : memref<128x256xf32> -> !loom.view
                %32 = loom.alloc(%14, %13) on @L1 : !loom.buffer_token
                %33 = loom.copy_to_tensor %31, %32, interconnect : [], broadcast : [1, 1] : !loom.view, !loom.buffer_token -> tensor<?x?xf32>
                %34 = linalg.matmul ins(%collapsed, %33 : tensor<?x?xf32>, tensor<?x?xf32>) outs(%19 : tensor<?x?xf32>) -> tensor<?x?xf32>
                %35 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%arg8, %34 : tensor<?x?xf32>, tensor<?x?xf32>) outs(%18 : tensor<?x?xf32>) {
                ^bb0(%in: f32, %in_0: f32, %out: f32):
                  %36 = arith.addf %in, %in_0 : f32
                  linalg.yield %36 : f32
                } -> tensor<?x?xf32>
                affine.yield %35 : tensor<?x?xf32>
              } {loom.iter_type = #loom.iter_type<sequential>}
              %25 = arith.muli %15, %12 : index
              %26 = arith.muli %16, %13 : index
              %27 = loom.view %arg2[%25, %26] [%12, %13] [1, 1], reuse : [seq = false, spat = false, temp = false] : memref<128x256xf32> -> !loom.view
              loom.copy_from_tensor %24, %27 : tensor<?x?xf32>, !loom.view
            } {loom.iter_type = #loom.iter_type<temporal>}
          } {loom.iter_type = #loom.iter_type<temporal>}
        } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @x}
      } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @y}
      return
    }
  }
  module attributes {loom.pass_name = "HoistBlockLoading"} {
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
      loom.polynomial_constraint(%12, %13, %14) {monomials = [{coeff = 4 : i64, vars = [0, 1]}, {coeff = 4 : i64, vars = [0, 2]}, {coeff = 4 : i64, vars = [2, 1]}], upper_bound = 1499136 : i64}
      loom.polynomial_constraint(%12) {monomials = [{coeff = 8 : i64, vars = [0]}], upper_bound = 1024 : i64}
      loom.polynomial_constraint(%13) {monomials = [{coeff = 8 : i64, vars = [0]}], upper_bound = 1024 : i64}
      loom.polynomial_constraint(%12, %13, %14) {monomials = [{coeff = 4 : i64, vars = [0, 1]}, {coeff = 4 : i64, vars = [0, 2]}], upper_bound = 1499136 : i64}
    }
    func.func @matmul__d1i0_d0i1__f01__hoist_block_1(%arg0: memref<128x128xf32>, %arg1: memref<128x256xf32>, %arg2: memref<128x256xf32>) {
      %cst = arith.constant 0.000000e+00 : f32
      %12 = loom.get_symbolic_block_size @constraints::@BM : index
      %13 = loom.get_symbolic_block_size @constraints::@BN : index
      %14 = loom.get_symbolic_block_size @constraints::@BK : index
      affine.parallel (%arg3) = (0) to (8) {
        affine.parallel (%arg4) = (0) to (8) {
          affine.for %arg5 = 0 to affine_map<()[s0, s1] -> ((128 ceildiv s0) ceildiv 8)>()[%12, %13] {
            affine.for %arg6 = 0 to affine_map<()[s0, s1] -> ((256 ceildiv s1) ceildiv 8)>()[%12, %13] {
              %15 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 8)>(%arg3, %arg5)
              %16 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 8)>(%arg4, %arg6)
              %17 = loom.alloc(%12, %13) on @L1 : !loom.buffer_token
              %18 = loom.init_tensor %17(%12, %13) : !loom.buffer_token -> tensor<?x?xf32>
              %19 = linalg.fill ins(%cst : f32) outs(%18 : tensor<?x?xf32>) -> tensor<?x?xf32>
              %c0 = arith.constant 0 : index
              %20 = arith.muli %16, %13 : index
              %c128 = arith.constant 128 : index
              %21 = loom.view %arg1[%c0, %20] [%c128, %13] [1, 1], reuse : [seq = false, spat = false, temp = false] : memref<128x256xf32> -> !loom.view
              %22 = loom.alloc(%c128, %13) on @L1 : !loom.buffer_token
              %23 = loom.pack_to_tensor %21, %22 inner_tiles = [%14, %13] outer_dims_perm = [0, 1] : !loom.view, !loom.buffer_token -> tensor<?x?x?xf32>
              %24 = affine.for %arg7 = 0 to affine_map<()[s0] -> (128 ceildiv s0)>()[%14] iter_args(%arg8 = %19) -> (tensor<?x?xf32>) {
                %28 = arith.muli %15, %12 : index
                %29 = arith.muli %arg7, %14 : index
                %30 = loom.view %arg0[%28, %29] [%12, %14] [1, 1], reuse : [seq = false, spat = false, temp = false] : memref<128x128xf32> -> !loom.view
                %31 = loom.alloc(%12, %14) on @L1 : !loom.buffer_token
                %32 = loom.copy_to_tensor %30, %31, interconnect : [], broadcast : [1, 1] : !loom.view, !loom.buffer_token -> tensor<?x?xf32>
                %33 = arith.muli %16, %13 : index
                %extracted_slice = tensor.extract_slice %23[%arg7, 0, 0] [1, %14, %13] [1, 1, 1] : tensor<?x?x?xf32> to tensor<1x?x?xf32>
                %collapsed = tensor.collapse_shape %extracted_slice [[0, 1], [2]] : tensor<1x?x?xf32> into tensor<?x?xf32>
                %34 = linalg.matmul ins(%32, %collapsed : tensor<?x?xf32>, tensor<?x?xf32>) outs(%19 : tensor<?x?xf32>) -> tensor<?x?xf32>
                %35 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%arg8, %34 : tensor<?x?xf32>, tensor<?x?xf32>) outs(%18 : tensor<?x?xf32>) {
                ^bb0(%in: f32, %in_0: f32, %out: f32):
                  %36 = arith.addf %in, %in_0 : f32
                  linalg.yield %36 : f32
                } -> tensor<?x?xf32>
                affine.yield %35 : tensor<?x?xf32>
              } {loom.iter_type = #loom.iter_type<sequential>}
              %25 = arith.muli %15, %12 : index
              %26 = arith.muli %16, %13 : index
              %27 = loom.view %arg2[%25, %26] [%12, %13] [1, 1], reuse : [seq = false, spat = false, temp = false] : memref<128x256xf32> -> !loom.view
              loom.copy_from_tensor %24, %27 : tensor<?x?xf32>, !loom.view
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
      loom.polynomial_constraint(%12, %13, %14) {monomials = [{coeff = 4 : i64, vars = [0, 1]}, {coeff = 4 : i64, vars = [0, 2]}, {coeff = 4 : i64, vars = [2, 1]}], upper_bound = 1499136 : i64}
      loom.polynomial_constraint(%12) {monomials = [{coeff = 8 : i64, vars = [0]}], upper_bound = 1024 : i64}
      loom.polynomial_constraint(%13) {monomials = [{coeff = 8 : i64, vars = [0]}], upper_bound = 1024 : i64}
    }
    func.func @matmul__d1i0_d0i1__f10(%arg0: memref<128x128xf32>, %arg1: memref<128x256xf32>, %arg2: memref<128x256xf32>) {
      %cst = arith.constant 0.000000e+00 : f32
      %12 = loom.get_symbolic_block_size @constraints::@BM : index
      %13 = loom.get_symbolic_block_size @constraints::@BN : index
      %14 = loom.get_symbolic_block_size @constraints::@BK : index
      affine.parallel (%arg3) = (0) to (8) {
        affine.parallel (%arg4) = (0) to (8) {
          affine.for %arg5 = 0 to affine_map<()[s0, s1] -> ((256 ceildiv s1) ceildiv 8)>()[%12, %13] {
            affine.for %arg6 = 0 to affine_map<()[s0, s1] -> ((128 ceildiv s0) ceildiv 8)>()[%12, %13] {
              %15 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 8)>(%arg3, %arg6)
              %16 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 8)>(%arg4, %arg5)
              %17 = loom.alloc(%12, %13) on @L1 : !loom.buffer_token
              %18 = loom.init_tensor %17(%12, %13) : !loom.buffer_token -> tensor<?x?xf32>
              %19 = linalg.fill ins(%cst : f32) outs(%18 : tensor<?x?xf32>) -> tensor<?x?xf32>
              %20 = affine.for %arg7 = 0 to affine_map<()[s0] -> (128 ceildiv s0)>()[%14] iter_args(%arg8 = %19) -> (tensor<?x?xf32>) {
                %24 = arith.muli %15, %12 : index
                %25 = arith.muli %arg7, %14 : index
                %26 = loom.view %arg0[%24, %25] [%12, %14] [1, 1], reuse : [seq = false, spat = false, temp = false] : memref<128x128xf32> -> !loom.view
                %27 = loom.alloc(%12, %14) on @L1 : !loom.buffer_token
                %28 = loom.copy_to_tensor %26, %27, interconnect : [], broadcast : [1, 1] : !loom.view, !loom.buffer_token -> tensor<?x?xf32>
                %29 = arith.muli %16, %13 : index
                %30 = loom.view %arg1[%25, %29] [%14, %13] [1, 1], reuse : [seq = false, spat = false, temp = false] : memref<128x256xf32> -> !loom.view
                %31 = loom.alloc(%14, %13) on @L1 : !loom.buffer_token
                %32 = loom.copy_to_tensor %30, %31, interconnect : [], broadcast : [1, 1] : !loom.view, !loom.buffer_token -> tensor<?x?xf32>
                %33 = linalg.matmul ins(%28, %32 : tensor<?x?xf32>, tensor<?x?xf32>) outs(%19 : tensor<?x?xf32>) -> tensor<?x?xf32>
                %34 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%arg8, %33 : tensor<?x?xf32>, tensor<?x?xf32>) outs(%18 : tensor<?x?xf32>) {
                ^bb0(%in: f32, %in_0: f32, %out: f32):
                  %35 = arith.addf %in, %in_0 : f32
                  linalg.yield %35 : f32
                } -> tensor<?x?xf32>
                affine.yield %34 : tensor<?x?xf32>
              } {loom.iter_type = #loom.iter_type<sequential>}
              %21 = arith.muli %15, %12 : index
              %22 = arith.muli %16, %13 : index
              %23 = loom.view %arg2[%21, %22] [%12, %13] [1, 1], reuse : [seq = false, spat = false, temp = false] : memref<128x256xf32> -> !loom.view
              loom.copy_from_tensor %20, %23 : tensor<?x?xf32>, !loom.view
            } {loom.iter_type = #loom.iter_type<temporal>}
          } {loom.iter_type = #loom.iter_type<temporal>}
        } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @x}
      } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @y}
      return
    }
  }
  module attributes {loom.pass_name = "HoistBlockLoading"} {
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
      loom.polynomial_constraint(%12, %13, %14) {monomials = [{coeff = 4 : i64, vars = [0, 1]}, {coeff = 4 : i64, vars = [0, 2]}, {coeff = 4 : i64, vars = [2, 1]}], upper_bound = 1499136 : i64}
      loom.polynomial_constraint(%12) {monomials = [{coeff = 8 : i64, vars = [0]}], upper_bound = 1024 : i64}
      loom.polynomial_constraint(%13) {monomials = [{coeff = 8 : i64, vars = [0]}], upper_bound = 1024 : i64}
      loom.polynomial_constraint(%12, %13, %14) {monomials = [{coeff = 4 : i64, vars = [0, 1]}, {coeff = 4 : i64, vars = [2, 1]}], upper_bound = 1499136 : i64}
    }
    func.func @matmul__d1i0_d0i1__f10__hoist_block_0(%arg0: memref<128x128xf32>, %arg1: memref<128x256xf32>, %arg2: memref<128x256xf32>) {
      %cst = arith.constant 0.000000e+00 : f32
      %12 = loom.get_symbolic_block_size @constraints::@BM : index
      %13 = loom.get_symbolic_block_size @constraints::@BN : index
      %14 = loom.get_symbolic_block_size @constraints::@BK : index
      affine.parallel (%arg3) = (0) to (8) {
        affine.parallel (%arg4) = (0) to (8) {
          affine.for %arg5 = 0 to affine_map<()[s0, s1] -> ((256 ceildiv s1) ceildiv 8)>()[%12, %13] {
            affine.for %arg6 = 0 to affine_map<()[s0, s1] -> ((128 ceildiv s0) ceildiv 8)>()[%12, %13] {
              %15 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 8)>(%arg3, %arg6)
              %16 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 8)>(%arg4, %arg5)
              %17 = loom.alloc(%12, %13) on @L1 : !loom.buffer_token
              %18 = loom.init_tensor %17(%12, %13) : !loom.buffer_token -> tensor<?x?xf32>
              %19 = linalg.fill ins(%cst : f32) outs(%18 : tensor<?x?xf32>) -> tensor<?x?xf32>
              %20 = arith.muli %15, %12 : index
              %c0 = arith.constant 0 : index
              %c128 = arith.constant 128 : index
              %21 = loom.view %arg0[%20, %c0] [%12, %c128] [1, 1], reuse : [seq = false, spat = false, temp = false] : memref<128x128xf32> -> !loom.view
              %22 = loom.alloc(%12, %c128) on @L1 : !loom.buffer_token
              %23 = loom.pack_to_tensor %21, %22 inner_tiles = [%12, %14] outer_dims_perm = [1, 0] : !loom.view, !loom.buffer_token -> tensor<?x?x?xf32>
              %24 = affine.for %arg7 = 0 to affine_map<()[s0] -> (128 ceildiv s0)>()[%14] iter_args(%arg8 = %19) -> (tensor<?x?xf32>) {
                %28 = arith.muli %15, %12 : index
                %29 = arith.muli %arg7, %14 : index
                %extracted_slice = tensor.extract_slice %23[%arg7, 0, 0] [1, %12, %14] [1, 1, 1] : tensor<?x?x?xf32> to tensor<1x?x?xf32>
                %collapsed = tensor.collapse_shape %extracted_slice [[0, 1], [2]] : tensor<1x?x?xf32> into tensor<?x?xf32>
                %30 = arith.muli %16, %13 : index
                %31 = loom.view %arg1[%29, %30] [%14, %13] [1, 1], reuse : [seq = false, spat = false, temp = false] : memref<128x256xf32> -> !loom.view
                %32 = loom.alloc(%14, %13) on @L1 : !loom.buffer_token
                %33 = loom.copy_to_tensor %31, %32, interconnect : [], broadcast : [1, 1] : !loom.view, !loom.buffer_token -> tensor<?x?xf32>
                %34 = linalg.matmul ins(%collapsed, %33 : tensor<?x?xf32>, tensor<?x?xf32>) outs(%19 : tensor<?x?xf32>) -> tensor<?x?xf32>
                %35 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%arg8, %34 : tensor<?x?xf32>, tensor<?x?xf32>) outs(%18 : tensor<?x?xf32>) {
                ^bb0(%in: f32, %in_0: f32, %out: f32):
                  %36 = arith.addf %in, %in_0 : f32
                  linalg.yield %36 : f32
                } -> tensor<?x?xf32>
                affine.yield %35 : tensor<?x?xf32>
              } {loom.iter_type = #loom.iter_type<sequential>}
              %25 = arith.muli %15, %12 : index
              %26 = arith.muli %16, %13 : index
              %27 = loom.view %arg2[%25, %26] [%12, %13] [1, 1], reuse : [seq = false, spat = false, temp = false] : memref<128x256xf32> -> !loom.view
              loom.copy_from_tensor %24, %27 : tensor<?x?xf32>, !loom.view
            } {loom.iter_type = #loom.iter_type<temporal>}
          } {loom.iter_type = #loom.iter_type<temporal>}
        } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @x}
      } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @y}
      return
    }
  }
  module attributes {loom.pass_name = "HoistBlockLoading"} {
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
      loom.polynomial_constraint(%12, %13, %14) {monomials = [{coeff = 4 : i64, vars = [0, 1]}, {coeff = 4 : i64, vars = [0, 2]}, {coeff = 4 : i64, vars = [2, 1]}], upper_bound = 1499136 : i64}
      loom.polynomial_constraint(%12) {monomials = [{coeff = 8 : i64, vars = [0]}], upper_bound = 1024 : i64}
      loom.polynomial_constraint(%13) {monomials = [{coeff = 8 : i64, vars = [0]}], upper_bound = 1024 : i64}
      loom.polynomial_constraint(%12, %13, %14) {monomials = [{coeff = 4 : i64, vars = [0, 1]}, {coeff = 4 : i64, vars = [0, 2]}], upper_bound = 1499136 : i64}
    }
    func.func @matmul__d1i0_d0i1__f10__hoist_block_1(%arg0: memref<128x128xf32>, %arg1: memref<128x256xf32>, %arg2: memref<128x256xf32>) {
      %cst = arith.constant 0.000000e+00 : f32
      %12 = loom.get_symbolic_block_size @constraints::@BM : index
      %13 = loom.get_symbolic_block_size @constraints::@BN : index
      %14 = loom.get_symbolic_block_size @constraints::@BK : index
      affine.parallel (%arg3) = (0) to (8) {
        affine.parallel (%arg4) = (0) to (8) {
          affine.for %arg5 = 0 to affine_map<()[s0, s1] -> ((256 ceildiv s1) ceildiv 8)>()[%12, %13] {
            affine.for %arg6 = 0 to affine_map<()[s0, s1] -> ((128 ceildiv s0) ceildiv 8)>()[%12, %13] {
              %15 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 8)>(%arg3, %arg6)
              %16 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 8)>(%arg4, %arg5)
              %17 = loom.alloc(%12, %13) on @L1 : !loom.buffer_token
              %18 = loom.init_tensor %17(%12, %13) : !loom.buffer_token -> tensor<?x?xf32>
              %19 = linalg.fill ins(%cst : f32) outs(%18 : tensor<?x?xf32>) -> tensor<?x?xf32>
              %c0 = arith.constant 0 : index
              %20 = arith.muli %16, %13 : index
              %c128 = arith.constant 128 : index
              %21 = loom.view %arg1[%c0, %20] [%c128, %13] [1, 1], reuse : [seq = false, spat = false, temp = false] : memref<128x256xf32> -> !loom.view
              %22 = loom.alloc(%c128, %13) on @L1 : !loom.buffer_token
              %23 = loom.pack_to_tensor %21, %22 inner_tiles = [%14, %13] outer_dims_perm = [0, 1] : !loom.view, !loom.buffer_token -> tensor<?x?x?xf32>
              %24 = affine.for %arg7 = 0 to affine_map<()[s0] -> (128 ceildiv s0)>()[%14] iter_args(%arg8 = %19) -> (tensor<?x?xf32>) {
                %28 = arith.muli %15, %12 : index
                %29 = arith.muli %arg7, %14 : index
                %30 = loom.view %arg0[%28, %29] [%12, %14] [1, 1], reuse : [seq = false, spat = false, temp = false] : memref<128x128xf32> -> !loom.view
                %31 = loom.alloc(%12, %14) on @L1 : !loom.buffer_token
                %32 = loom.copy_to_tensor %30, %31, interconnect : [], broadcast : [1, 1] : !loom.view, !loom.buffer_token -> tensor<?x?xf32>
                %33 = arith.muli %16, %13 : index
                %extracted_slice = tensor.extract_slice %23[%arg7, 0, 0] [1, %14, %13] [1, 1, 1] : tensor<?x?x?xf32> to tensor<1x?x?xf32>
                %collapsed = tensor.collapse_shape %extracted_slice [[0, 1], [2]] : tensor<1x?x?xf32> into tensor<?x?xf32>
                %34 = linalg.matmul ins(%32, %collapsed : tensor<?x?xf32>, tensor<?x?xf32>) outs(%19 : tensor<?x?xf32>) -> tensor<?x?xf32>
                %35 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%arg8, %34 : tensor<?x?xf32>, tensor<?x?xf32>) outs(%18 : tensor<?x?xf32>) {
                ^bb0(%in: f32, %in_0: f32, %out: f32):
                  %36 = arith.addf %in, %in_0 : f32
                  linalg.yield %36 : f32
                } -> tensor<?x?xf32>
                affine.yield %35 : tensor<?x?xf32>
              } {loom.iter_type = #loom.iter_type<sequential>}
              %25 = arith.muli %15, %12 : index
              %26 = arith.muli %16, %13 : index
              %27 = loom.view %arg2[%25, %26] [%12, %13] [1, 1], reuse : [seq = false, spat = false, temp = false] : memref<128x256xf32> -> !loom.view
              loom.copy_from_tensor %24, %27 : tensor<?x?xf32>, !loom.view
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
      loom.polynomial_constraint(%12, %13, %14) {monomials = [{coeff = 4 : i64, vars = [0, 1]}, {coeff = 4 : i64, vars = [0, 2]}, {coeff = 4 : i64, vars = [2, 1]}], upper_bound = 1499136 : i64}
      loom.polynomial_constraint(%13) {monomials = [{coeff = 64 : i64, vars = [0]}], upper_bound = 1024 : i64}
    }
    func.func @matmul__d0i1_d1i1__f01(%arg0: memref<128x128xf32>, %arg1: memref<128x256xf32>, %arg2: memref<128x256xf32>) {
      %cst = arith.constant 0.000000e+00 : f32
      %12 = loom.get_symbolic_block_size @constraints::@BM : index
      %13 = loom.get_symbolic_block_size @constraints::@BN : index
      %14 = loom.get_symbolic_block_size @constraints::@BK : index
      affine.parallel (%arg3) = (0) to (8) {
        affine.parallel (%arg4) = (0) to (8) {
          affine.for %arg5 = 0 to affine_map<()[s0, s1] -> (128 ceildiv s0)>()[%12, %13] {
            affine.for %arg6 = 0 to affine_map<()[s0, s1] -> (((256 ceildiv s1) ceildiv 8) ceildiv 8)>()[%12, %13] {
              %15 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 8 + d1 + d2 * 64)>(%arg3, %arg4, %arg6)
              %16 = loom.alloc(%12, %13) on @L1 : !loom.buffer_token
              %17 = loom.init_tensor %16(%12, %13) : !loom.buffer_token -> tensor<?x?xf32>
              %18 = linalg.fill ins(%cst : f32) outs(%17 : tensor<?x?xf32>) -> tensor<?x?xf32>
              %19 = affine.for %arg7 = 0 to affine_map<()[s0] -> (128 ceildiv s0)>()[%14] iter_args(%arg8 = %18) -> (tensor<?x?xf32>) {
                %23 = arith.muli %arg5, %12 : index
                %24 = arith.muli %arg7, %14 : index
                %25 = loom.view %arg0[%23, %24] [%12, %14] [1, 1], reuse : [seq = false, spat = false, temp = false] : memref<128x128xf32> -> !loom.view
                %26 = loom.alloc(%12, %14) on @L1 : !loom.buffer_token
                %27 = loom.copy_to_tensor %25, %26, interconnect : [], broadcast : [1, 1] : !loom.view, !loom.buffer_token -> tensor<?x?xf32>
                %28 = arith.muli %15, %13 : index
                %29 = loom.view %arg1[%24, %28] [%14, %13] [1, 1], reuse : [seq = false, spat = false, temp = false] : memref<128x256xf32> -> !loom.view
                %30 = loom.alloc(%14, %13) on @L1 : !loom.buffer_token
                %31 = loom.copy_to_tensor %29, %30, interconnect : [], broadcast : [1, 1] : !loom.view, !loom.buffer_token -> tensor<?x?xf32>
                %32 = linalg.matmul ins(%27, %31 : tensor<?x?xf32>, tensor<?x?xf32>) outs(%18 : tensor<?x?xf32>) -> tensor<?x?xf32>
                %33 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%arg8, %32 : tensor<?x?xf32>, tensor<?x?xf32>) outs(%17 : tensor<?x?xf32>) {
                ^bb0(%in: f32, %in_0: f32, %out: f32):
                  %34 = arith.addf %in, %in_0 : f32
                  linalg.yield %34 : f32
                } -> tensor<?x?xf32>
                affine.yield %33 : tensor<?x?xf32>
              } {loom.iter_type = #loom.iter_type<sequential>}
              %20 = arith.muli %arg5, %12 : index
              %21 = arith.muli %15, %13 : index
              %22 = loom.view %arg2[%20, %21] [%12, %13] [1, 1], reuse : [seq = false, spat = false, temp = false] : memref<128x256xf32> -> !loom.view
              loom.copy_from_tensor %19, %22 : tensor<?x?xf32>, !loom.view
            } {loom.iter_type = #loom.iter_type<temporal>}
          } {loom.iter_type = #loom.iter_type<temporal>}
        } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @y}
      } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @x}
      return
    }
  }
  module attributes {loom.pass_name = "HoistBlockLoading"} {
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
      loom.polynomial_constraint(%12, %13, %14) {monomials = [{coeff = 4 : i64, vars = [0, 1]}, {coeff = 4 : i64, vars = [0, 2]}, {coeff = 4 : i64, vars = [2, 1]}], upper_bound = 1499136 : i64}
      loom.polynomial_constraint(%13) {monomials = [{coeff = 64 : i64, vars = [0]}], upper_bound = 1024 : i64}
      loom.polynomial_constraint(%12, %13, %14) {monomials = [{coeff = 4 : i64, vars = [0, 1]}, {coeff = 4 : i64, vars = [2, 1]}], upper_bound = 1499136 : i64}
    }
    func.func @matmul__d0i1_d1i1__f01__hoist_block_0(%arg0: memref<128x128xf32>, %arg1: memref<128x256xf32>, %arg2: memref<128x256xf32>) {
      %cst = arith.constant 0.000000e+00 : f32
      %12 = loom.get_symbolic_block_size @constraints::@BM : index
      %13 = loom.get_symbolic_block_size @constraints::@BN : index
      %14 = loom.get_symbolic_block_size @constraints::@BK : index
      affine.parallel (%arg3) = (0) to (8) {
        affine.parallel (%arg4) = (0) to (8) {
          affine.for %arg5 = 0 to affine_map<()[s0, s1] -> (128 ceildiv s0)>()[%12, %13] {
            affine.for %arg6 = 0 to affine_map<()[s0, s1] -> (((256 ceildiv s1) ceildiv 8) ceildiv 8)>()[%12, %13] {
              %15 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 8 + d1 + d2 * 64)>(%arg3, %arg4, %arg6)
              %16 = loom.alloc(%12, %13) on @L1 : !loom.buffer_token
              %17 = loom.init_tensor %16(%12, %13) : !loom.buffer_token -> tensor<?x?xf32>
              %18 = linalg.fill ins(%cst : f32) outs(%17 : tensor<?x?xf32>) -> tensor<?x?xf32>
              %19 = arith.muli %arg5, %12 : index
              %c0 = arith.constant 0 : index
              %c128 = arith.constant 128 : index
              %20 = loom.view %arg0[%19, %c0] [%12, %c128] [1, 1], reuse : [seq = false, spat = false, temp = false] : memref<128x128xf32> -> !loom.view
              %21 = loom.alloc(%12, %c128) on @L1 : !loom.buffer_token
              %22 = loom.pack_to_tensor %20, %21 inner_tiles = [%12, %14] outer_dims_perm = [1, 0] : !loom.view, !loom.buffer_token -> tensor<?x?x?xf32>
              %23 = affine.for %arg7 = 0 to affine_map<()[s0] -> (128 ceildiv s0)>()[%14] iter_args(%arg8 = %18) -> (tensor<?x?xf32>) {
                %27 = arith.muli %arg5, %12 : index
                %28 = arith.muli %arg7, %14 : index
                %extracted_slice = tensor.extract_slice %22[%arg7, 0, 0] [1, %12, %14] [1, 1, 1] : tensor<?x?x?xf32> to tensor<1x?x?xf32>
                %collapsed = tensor.collapse_shape %extracted_slice [[0, 1], [2]] : tensor<1x?x?xf32> into tensor<?x?xf32>
                %29 = arith.muli %15, %13 : index
                %30 = loom.view %arg1[%28, %29] [%14, %13] [1, 1], reuse : [seq = false, spat = false, temp = false] : memref<128x256xf32> -> !loom.view
                %31 = loom.alloc(%14, %13) on @L1 : !loom.buffer_token
                %32 = loom.copy_to_tensor %30, %31, interconnect : [], broadcast : [1, 1] : !loom.view, !loom.buffer_token -> tensor<?x?xf32>
                %33 = linalg.matmul ins(%collapsed, %32 : tensor<?x?xf32>, tensor<?x?xf32>) outs(%18 : tensor<?x?xf32>) -> tensor<?x?xf32>
                %34 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%arg8, %33 : tensor<?x?xf32>, tensor<?x?xf32>) outs(%17 : tensor<?x?xf32>) {
                ^bb0(%in: f32, %in_0: f32, %out: f32):
                  %35 = arith.addf %in, %in_0 : f32
                  linalg.yield %35 : f32
                } -> tensor<?x?xf32>
                affine.yield %34 : tensor<?x?xf32>
              } {loom.iter_type = #loom.iter_type<sequential>}
              %24 = arith.muli %arg5, %12 : index
              %25 = arith.muli %15, %13 : index
              %26 = loom.view %arg2[%24, %25] [%12, %13] [1, 1], reuse : [seq = false, spat = false, temp = false] : memref<128x256xf32> -> !loom.view
              loom.copy_from_tensor %23, %26 : tensor<?x?xf32>, !loom.view
            } {loom.iter_type = #loom.iter_type<temporal>}
          } {loom.iter_type = #loom.iter_type<temporal>}
        } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @y}
      } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @x}
      return
    }
  }
  module attributes {loom.pass_name = "HoistBlockLoading"} {
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
      loom.polynomial_constraint(%12, %13, %14) {monomials = [{coeff = 4 : i64, vars = [0, 1]}, {coeff = 4 : i64, vars = [0, 2]}, {coeff = 4 : i64, vars = [2, 1]}], upper_bound = 1499136 : i64}
      loom.polynomial_constraint(%13) {monomials = [{coeff = 64 : i64, vars = [0]}], upper_bound = 1024 : i64}
      loom.polynomial_constraint(%12, %13, %14) {monomials = [{coeff = 4 : i64, vars = [0, 1]}, {coeff = 4 : i64, vars = [0, 2]}], upper_bound = 1499136 : i64}
    }
    func.func @matmul__d0i1_d1i1__f01__hoist_block_1(%arg0: memref<128x128xf32>, %arg1: memref<128x256xf32>, %arg2: memref<128x256xf32>) {
      %cst = arith.constant 0.000000e+00 : f32
      %12 = loom.get_symbolic_block_size @constraints::@BM : index
      %13 = loom.get_symbolic_block_size @constraints::@BN : index
      %14 = loom.get_symbolic_block_size @constraints::@BK : index
      affine.parallel (%arg3) = (0) to (8) {
        affine.parallel (%arg4) = (0) to (8) {
          affine.for %arg5 = 0 to affine_map<()[s0, s1] -> (128 ceildiv s0)>()[%12, %13] {
            affine.for %arg6 = 0 to affine_map<()[s0, s1] -> (((256 ceildiv s1) ceildiv 8) ceildiv 8)>()[%12, %13] {
              %15 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 8 + d1 + d2 * 64)>(%arg3, %arg4, %arg6)
              %16 = loom.alloc(%12, %13) on @L1 : !loom.buffer_token
              %17 = loom.init_tensor %16(%12, %13) : !loom.buffer_token -> tensor<?x?xf32>
              %18 = linalg.fill ins(%cst : f32) outs(%17 : tensor<?x?xf32>) -> tensor<?x?xf32>
              %c0 = arith.constant 0 : index
              %19 = arith.muli %15, %13 : index
              %c128 = arith.constant 128 : index
              %20 = loom.view %arg1[%c0, %19] [%c128, %13] [1, 1], reuse : [seq = false, spat = false, temp = false] : memref<128x256xf32> -> !loom.view
              %21 = loom.alloc(%c128, %13) on @L1 : !loom.buffer_token
              %22 = loom.pack_to_tensor %20, %21 inner_tiles = [%14, %13] outer_dims_perm = [0, 1] : !loom.view, !loom.buffer_token -> tensor<?x?x?xf32>
              %23 = affine.for %arg7 = 0 to affine_map<()[s0] -> (128 ceildiv s0)>()[%14] iter_args(%arg8 = %18) -> (tensor<?x?xf32>) {
                %27 = arith.muli %arg5, %12 : index
                %28 = arith.muli %arg7, %14 : index
                %29 = loom.view %arg0[%27, %28] [%12, %14] [1, 1], reuse : [seq = false, spat = false, temp = false] : memref<128x128xf32> -> !loom.view
                %30 = loom.alloc(%12, %14) on @L1 : !loom.buffer_token
                %31 = loom.copy_to_tensor %29, %30, interconnect : [], broadcast : [1, 1] : !loom.view, !loom.buffer_token -> tensor<?x?xf32>
                %32 = arith.muli %15, %13 : index
                %extracted_slice = tensor.extract_slice %22[%arg7, 0, 0] [1, %14, %13] [1, 1, 1] : tensor<?x?x?xf32> to tensor<1x?x?xf32>
                %collapsed = tensor.collapse_shape %extracted_slice [[0, 1], [2]] : tensor<1x?x?xf32> into tensor<?x?xf32>
                %33 = linalg.matmul ins(%31, %collapsed : tensor<?x?xf32>, tensor<?x?xf32>) outs(%18 : tensor<?x?xf32>) -> tensor<?x?xf32>
                %34 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%arg8, %33 : tensor<?x?xf32>, tensor<?x?xf32>) outs(%17 : tensor<?x?xf32>) {
                ^bb0(%in: f32, %in_0: f32, %out: f32):
                  %35 = arith.addf %in, %in_0 : f32
                  linalg.yield %35 : f32
                } -> tensor<?x?xf32>
                affine.yield %34 : tensor<?x?xf32>
              } {loom.iter_type = #loom.iter_type<sequential>}
              %24 = arith.muli %arg5, %12 : index
              %25 = arith.muli %15, %13 : index
              %26 = loom.view %arg2[%24, %25] [%12, %13] [1, 1], reuse : [seq = false, spat = false, temp = false] : memref<128x256xf32> -> !loom.view
              loom.copy_from_tensor %23, %26 : tensor<?x?xf32>, !loom.view
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
      loom.polynomial_constraint(%12, %13, %14) {monomials = [{coeff = 4 : i64, vars = [0, 1]}, {coeff = 4 : i64, vars = [0, 2]}, {coeff = 4 : i64, vars = [2, 1]}], upper_bound = 1499136 : i64}
      loom.polynomial_constraint(%13) {monomials = [{coeff = 64 : i64, vars = [0]}], upper_bound = 1024 : i64}
    }
    func.func @matmul__d0i1_d1i1__f10(%arg0: memref<128x128xf32>, %arg1: memref<128x256xf32>, %arg2: memref<128x256xf32>) {
      %cst = arith.constant 0.000000e+00 : f32
      %12 = loom.get_symbolic_block_size @constraints::@BM : index
      %13 = loom.get_symbolic_block_size @constraints::@BN : index
      %14 = loom.get_symbolic_block_size @constraints::@BK : index
      affine.parallel (%arg3) = (0) to (8) {
        affine.parallel (%arg4) = (0) to (8) {
          affine.for %arg5 = 0 to affine_map<()[s0, s1] -> (((256 ceildiv s1) ceildiv 8) ceildiv 8)>()[%12, %13] {
            affine.for %arg6 = 0 to affine_map<()[s0, s1] -> (128 ceildiv s0)>()[%12, %13] {
              %15 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 8 + d1 + d2 * 64)>(%arg3, %arg4, %arg5)
              %16 = loom.alloc(%12, %13) on @L1 : !loom.buffer_token
              %17 = loom.init_tensor %16(%12, %13) : !loom.buffer_token -> tensor<?x?xf32>
              %18 = linalg.fill ins(%cst : f32) outs(%17 : tensor<?x?xf32>) -> tensor<?x?xf32>
              %19 = affine.for %arg7 = 0 to affine_map<()[s0] -> (128 ceildiv s0)>()[%14] iter_args(%arg8 = %18) -> (tensor<?x?xf32>) {
                %23 = arith.muli %arg6, %12 : index
                %24 = arith.muli %arg7, %14 : index
                %25 = loom.view %arg0[%23, %24] [%12, %14] [1, 1], reuse : [seq = false, spat = false, temp = false] : memref<128x128xf32> -> !loom.view
                %26 = loom.alloc(%12, %14) on @L1 : !loom.buffer_token
                %27 = loom.copy_to_tensor %25, %26, interconnect : [], broadcast : [1, 1] : !loom.view, !loom.buffer_token -> tensor<?x?xf32>
                %28 = arith.muli %15, %13 : index
                %29 = loom.view %arg1[%24, %28] [%14, %13] [1, 1], reuse : [seq = false, spat = false, temp = false] : memref<128x256xf32> -> !loom.view
                %30 = loom.alloc(%14, %13) on @L1 : !loom.buffer_token
                %31 = loom.copy_to_tensor %29, %30, interconnect : [], broadcast : [1, 1] : !loom.view, !loom.buffer_token -> tensor<?x?xf32>
                %32 = linalg.matmul ins(%27, %31 : tensor<?x?xf32>, tensor<?x?xf32>) outs(%18 : tensor<?x?xf32>) -> tensor<?x?xf32>
                %33 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%arg8, %32 : tensor<?x?xf32>, tensor<?x?xf32>) outs(%17 : tensor<?x?xf32>) {
                ^bb0(%in: f32, %in_0: f32, %out: f32):
                  %34 = arith.addf %in, %in_0 : f32
                  linalg.yield %34 : f32
                } -> tensor<?x?xf32>
                affine.yield %33 : tensor<?x?xf32>
              } {loom.iter_type = #loom.iter_type<sequential>}
              %20 = arith.muli %arg6, %12 : index
              %21 = arith.muli %15, %13 : index
              %22 = loom.view %arg2[%20, %21] [%12, %13] [1, 1], reuse : [seq = false, spat = false, temp = false] : memref<128x256xf32> -> !loom.view
              loom.copy_from_tensor %19, %22 : tensor<?x?xf32>, !loom.view
            } {loom.iter_type = #loom.iter_type<temporal>}
          } {loom.iter_type = #loom.iter_type<temporal>}
        } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @y}
      } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @x}
      return
    }
  }
  module attributes {loom.pass_name = "HoistBlockLoading"} {
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
      loom.polynomial_constraint(%12, %13, %14) {monomials = [{coeff = 4 : i64, vars = [0, 1]}, {coeff = 4 : i64, vars = [0, 2]}, {coeff = 4 : i64, vars = [2, 1]}], upper_bound = 1499136 : i64}
      loom.polynomial_constraint(%13) {monomials = [{coeff = 64 : i64, vars = [0]}], upper_bound = 1024 : i64}
      loom.polynomial_constraint(%12, %13, %14) {monomials = [{coeff = 4 : i64, vars = [0, 1]}, {coeff = 4 : i64, vars = [2, 1]}], upper_bound = 1499136 : i64}
    }
    func.func @matmul__d0i1_d1i1__f10__hoist_block_0(%arg0: memref<128x128xf32>, %arg1: memref<128x256xf32>, %arg2: memref<128x256xf32>) {
      %cst = arith.constant 0.000000e+00 : f32
      %12 = loom.get_symbolic_block_size @constraints::@BM : index
      %13 = loom.get_symbolic_block_size @constraints::@BN : index
      %14 = loom.get_symbolic_block_size @constraints::@BK : index
      affine.parallel (%arg3) = (0) to (8) {
        affine.parallel (%arg4) = (0) to (8) {
          affine.for %arg5 = 0 to affine_map<()[s0, s1] -> (((256 ceildiv s1) ceildiv 8) ceildiv 8)>()[%12, %13] {
            affine.for %arg6 = 0 to affine_map<()[s0, s1] -> (128 ceildiv s0)>()[%12, %13] {
              %15 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 8 + d1 + d2 * 64)>(%arg3, %arg4, %arg5)
              %16 = loom.alloc(%12, %13) on @L1 : !loom.buffer_token
              %17 = loom.init_tensor %16(%12, %13) : !loom.buffer_token -> tensor<?x?xf32>
              %18 = linalg.fill ins(%cst : f32) outs(%17 : tensor<?x?xf32>) -> tensor<?x?xf32>
              %19 = arith.muli %arg6, %12 : index
              %c0 = arith.constant 0 : index
              %c128 = arith.constant 128 : index
              %20 = loom.view %arg0[%19, %c0] [%12, %c128] [1, 1], reuse : [seq = false, spat = false, temp = false] : memref<128x128xf32> -> !loom.view
              %21 = loom.alloc(%12, %c128) on @L1 : !loom.buffer_token
              %22 = loom.pack_to_tensor %20, %21 inner_tiles = [%12, %14] outer_dims_perm = [1, 0] : !loom.view, !loom.buffer_token -> tensor<?x?x?xf32>
              %23 = affine.for %arg7 = 0 to affine_map<()[s0] -> (128 ceildiv s0)>()[%14] iter_args(%arg8 = %18) -> (tensor<?x?xf32>) {
                %27 = arith.muli %arg6, %12 : index
                %28 = arith.muli %arg7, %14 : index
                %extracted_slice = tensor.extract_slice %22[%arg7, 0, 0] [1, %12, %14] [1, 1, 1] : tensor<?x?x?xf32> to tensor<1x?x?xf32>
                %collapsed = tensor.collapse_shape %extracted_slice [[0, 1], [2]] : tensor<1x?x?xf32> into tensor<?x?xf32>
                %29 = arith.muli %15, %13 : index
                %30 = loom.view %arg1[%28, %29] [%14, %13] [1, 1], reuse : [seq = false, spat = false, temp = false] : memref<128x256xf32> -> !loom.view
                %31 = loom.alloc(%14, %13) on @L1 : !loom.buffer_token
                %32 = loom.copy_to_tensor %30, %31, interconnect : [], broadcast : [1, 1] : !loom.view, !loom.buffer_token -> tensor<?x?xf32>
                %33 = linalg.matmul ins(%collapsed, %32 : tensor<?x?xf32>, tensor<?x?xf32>) outs(%18 : tensor<?x?xf32>) -> tensor<?x?xf32>
                %34 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%arg8, %33 : tensor<?x?xf32>, tensor<?x?xf32>) outs(%17 : tensor<?x?xf32>) {
                ^bb0(%in: f32, %in_0: f32, %out: f32):
                  %35 = arith.addf %in, %in_0 : f32
                  linalg.yield %35 : f32
                } -> tensor<?x?xf32>
                affine.yield %34 : tensor<?x?xf32>
              } {loom.iter_type = #loom.iter_type<sequential>}
              %24 = arith.muli %arg6, %12 : index
              %25 = arith.muli %15, %13 : index
              %26 = loom.view %arg2[%24, %25] [%12, %13] [1, 1], reuse : [seq = false, spat = false, temp = false] : memref<128x256xf32> -> !loom.view
              loom.copy_from_tensor %23, %26 : tensor<?x?xf32>, !loom.view
            } {loom.iter_type = #loom.iter_type<temporal>}
          } {loom.iter_type = #loom.iter_type<temporal>}
        } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @y}
      } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @x}
      return
    }
  }
  module attributes {loom.pass_name = "HoistBlockLoading"} {
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
      loom.polynomial_constraint(%12, %13, %14) {monomials = [{coeff = 4 : i64, vars = [0, 1]}, {coeff = 4 : i64, vars = [0, 2]}, {coeff = 4 : i64, vars = [2, 1]}], upper_bound = 1499136 : i64}
      loom.polynomial_constraint(%13) {monomials = [{coeff = 64 : i64, vars = [0]}], upper_bound = 1024 : i64}
      loom.polynomial_constraint(%12, %13, %14) {monomials = [{coeff = 4 : i64, vars = [0, 1]}, {coeff = 4 : i64, vars = [0, 2]}], upper_bound = 1499136 : i64}
    }
    func.func @matmul__d0i1_d1i1__f10__hoist_block_1(%arg0: memref<128x128xf32>, %arg1: memref<128x256xf32>, %arg2: memref<128x256xf32>) {
      %cst = arith.constant 0.000000e+00 : f32
      %12 = loom.get_symbolic_block_size @constraints::@BM : index
      %13 = loom.get_symbolic_block_size @constraints::@BN : index
      %14 = loom.get_symbolic_block_size @constraints::@BK : index
      affine.parallel (%arg3) = (0) to (8) {
        affine.parallel (%arg4) = (0) to (8) {
          affine.for %arg5 = 0 to affine_map<()[s0, s1] -> (((256 ceildiv s1) ceildiv 8) ceildiv 8)>()[%12, %13] {
            affine.for %arg6 = 0 to affine_map<()[s0, s1] -> (128 ceildiv s0)>()[%12, %13] {
              %15 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 8 + d1 + d2 * 64)>(%arg3, %arg4, %arg5)
              %16 = loom.alloc(%12, %13) on @L1 : !loom.buffer_token
              %17 = loom.init_tensor %16(%12, %13) : !loom.buffer_token -> tensor<?x?xf32>
              %18 = linalg.fill ins(%cst : f32) outs(%17 : tensor<?x?xf32>) -> tensor<?x?xf32>
              %c0 = arith.constant 0 : index
              %19 = arith.muli %15, %13 : index
              %c128 = arith.constant 128 : index
              %20 = loom.view %arg1[%c0, %19] [%c128, %13] [1, 1], reuse : [seq = false, spat = false, temp = false] : memref<128x256xf32> -> !loom.view
              %21 = loom.alloc(%c128, %13) on @L1 : !loom.buffer_token
              %22 = loom.pack_to_tensor %20, %21 inner_tiles = [%14, %13] outer_dims_perm = [0, 1] : !loom.view, !loom.buffer_token -> tensor<?x?x?xf32>
              %23 = affine.for %arg7 = 0 to affine_map<()[s0] -> (128 ceildiv s0)>()[%14] iter_args(%arg8 = %18) -> (tensor<?x?xf32>) {
                %27 = arith.muli %arg6, %12 : index
                %28 = arith.muli %arg7, %14 : index
                %29 = loom.view %arg0[%27, %28] [%12, %14] [1, 1], reuse : [seq = false, spat = false, temp = false] : memref<128x128xf32> -> !loom.view
                %30 = loom.alloc(%12, %14) on @L1 : !loom.buffer_token
                %31 = loom.copy_to_tensor %29, %30, interconnect : [], broadcast : [1, 1] : !loom.view, !loom.buffer_token -> tensor<?x?xf32>
                %32 = arith.muli %15, %13 : index
                %extracted_slice = tensor.extract_slice %22[%arg7, 0, 0] [1, %14, %13] [1, 1, 1] : tensor<?x?x?xf32> to tensor<1x?x?xf32>
                %collapsed = tensor.collapse_shape %extracted_slice [[0, 1], [2]] : tensor<1x?x?xf32> into tensor<?x?xf32>
                %33 = linalg.matmul ins(%31, %collapsed : tensor<?x?xf32>, tensor<?x?xf32>) outs(%18 : tensor<?x?xf32>) -> tensor<?x?xf32>
                %34 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%arg8, %33 : tensor<?x?xf32>, tensor<?x?xf32>) outs(%17 : tensor<?x?xf32>) {
                ^bb0(%in: f32, %in_0: f32, %out: f32):
                  %35 = arith.addf %in, %in_0 : f32
                  linalg.yield %35 : f32
                } -> tensor<?x?xf32>
                affine.yield %34 : tensor<?x?xf32>
              } {loom.iter_type = #loom.iter_type<sequential>}
              %24 = arith.muli %arg6, %12 : index
              %25 = arith.muli %15, %13 : index
              %26 = loom.view %arg2[%24, %25] [%12, %13] [1, 1], reuse : [seq = false, spat = false, temp = false] : memref<128x256xf32> -> !loom.view
              loom.copy_from_tensor %23, %26 : tensor<?x?xf32>, !loom.view
            } {loom.iter_type = #loom.iter_type<temporal>}
          } {loom.iter_type = #loom.iter_type<temporal>}
        } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @y}
      } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @x}
      return
    }
  }
}
