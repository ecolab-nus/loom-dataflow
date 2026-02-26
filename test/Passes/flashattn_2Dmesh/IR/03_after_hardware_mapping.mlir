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
      %12 = loom.symbolic_var "BB" : index
      %13 = loom.symbolic_var "BM" : index
      %14 = loom.symbolic_var "BN" : index
      loom.range %12[32, 32]
      loom.range %13[32, 4096]
      loom.range %14[32, 4096]
      loom.align %12 by 32
      loom.align %13 by 32
      loom.align %14 by 32
      loom.polynomial_constraint(%12, %13, %14) {monomials = [{coeff = -1 : i64, vars = [0, 1, 2]}], upper_bound = -262144 : i64}
      loom.polynomial_constraint(%12, %14, %13) {monomials = [{coeff = 4 : i64, vars = [0, 1]}, {coeff = 4 : i64, vars = [0, 2, 1]}, {coeff = 4 : i64, vars = [0, 1]}, {coeff = 4 : i64, vars = [0, 2]}, {coeff = 4 : i64, vars = [0, 2]}, {coeff = 4 : i64, vars = [0, 2]}, {coeff = 4 : i64, vars = [0, 2]}, {coeff = 4 : i64, vars = [0, 2]}, {coeff = 4 : i64, vars = [0, 2]}, {coeff = 4 : i64, vars = [0, 2]}, {coeff = 4 : i64, vars = [0, 2]}], upper_bound = 1499136 : i64}
      loom.polynomial_constraint(%12) {monomials = [{coeff = 64 : i64, vars = [0]}], upper_bound = 32 : i64}
    }
    func.func @attention__d0i0_d1i0__f01(%arg0: memref<32x128x4096xf32>, %arg1: memref<32x4096x128xf32>, %arg2: memref<32x4096x128xf32>, %arg3: memref<32x4096x128xf32>) {
      %cst = arith.constant 2.000000e+00 : f32
      %cst_0 = arith.constant 0.12751743074602467 : f64
      %cst_1 = arith.constant 0.000000e+00 : f32
      %cst_2 = arith.constant 1.000000e+00 : f32
      %cst_3 = arith.constant 0xFF800000 : f32
      %12 = loom.get_symbolic_block_size @constraints::@BB : index
      %13 = loom.get_symbolic_block_size @constraints::@BM : index
      %14 = loom.get_symbolic_block_size @constraints::@BN : index
      affine.parallel (%arg4) = (0) to (8) {
        affine.parallel (%arg5) = (0) to (8) {
          affine.for %arg6 = 0 to affine_map<()[s0, s1] -> (((32 ceildiv s0) ceildiv 8) ceildiv 8)>()[%12, %13] {
            affine.for %arg7 = 0 to affine_map<()[s0, s1] -> (4096 ceildiv s1)>()[%12, %13] {
              %15 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 8 + d1 + d2 * 64)>(%arg4, %arg5, %arg6)
              %16 = loom.alloc [%12, %14, 128] on @L1 : memref<?x?x128xf32>
              %17 = loom.alloc [%12, %13, %14] on @L1 : memref<?x?x?xf32>
              %18 = loom.init_tensor %17[%12, %13, %14] : memref<?x?x?xf32> -> tensor<?x?x?xf32>
              %19 = loom.alloc [%12, 128, %14] on @L1 : memref<?x128x?xf32>
              %20 = loom.alloc [%12, %13] on @L1 : memref<?x?xf32>
              %21 = loom.init_tensor %20[%12, %13] : memref<?x?xf32> -> tensor<?x?xf32>
              %22 = loom.alloc [%12, %13] on @L1 : memref<?x?xf32>
              %23 = loom.init_tensor %22[%12, %13] : memref<?x?xf32> -> tensor<?x?xf32>
              %24 = loom.alloc [%12, %13] on @L1 : memref<?x?xf32>
              %25 = loom.init_tensor %24[%12, %13] : memref<?x?xf32> -> tensor<?x?xf32>
              %26 = loom.alloc [%12, %13] on @L1 : memref<?x?xf32>
              %27 = loom.init_tensor %26[%12, %13] : memref<?x?xf32> -> tensor<?x?xf32>
              %28 = loom.alloc [%12, %13] on @L1 : memref<?x?xf32>
              %29 = loom.init_tensor %28[%12, %13] : memref<?x?xf32> -> tensor<?x?xf32>
              %30 = loom.alloc [%12, %13, 128] on @L1 : memref<?x?x128xf32>
              %31 = loom.init_tensor %30[%12, %13, 128] : memref<?x?x128xf32> -> tensor<?x?x128xf32>
              %32 = loom.alloc [%12, %13, 128] on @L1 : memref<?x?x128xf32>
              %33 = loom.init_tensor %32[%12, %13, 128] : memref<?x?x128xf32> -> tensor<?x?x128xf32>
              %34 = loom.alloc [%12, %13, 128] on @L1 : memref<?x?x128xf32>
              %35 = loom.init_tensor %34[%12, %13, 128] : memref<?x?x128xf32> -> tensor<?x?x128xf32>
              %36 = arith.muli %15, %12 : index
              %37 = arith.muli %arg7, %13 : index
              %38 = loom.subview %arg2[%36, %37, 0] [%12, %13, 128] [1, 1, 1], reuse : [seq = false, spat = false, temp = false] : memref<32x4096x128xf32> to memref<?x?x128xf32, strided<[524288, 128, 1], offset: ?>>
              %39 = loom.copy_to_tensor %38, %30 on @L1, interconnect : [], broadcast : [1, 1] : memref<?x?x128xf32, strided<[524288, 128, 1], offset: ?>>, memref<?x?x128xf32> -> tensor<?x?x128xf32>
              %40 = linalg.fill ins(%cst_1 : f32) outs(%33 : tensor<?x?x128xf32>) -> tensor<?x?x128xf32>
              %41 = linalg.fill ins(%cst_2 : f32) outs(%21 : tensor<?x?xf32>) -> tensor<?x?xf32>
              %42 = linalg.fill ins(%cst_3 : f32) outs(%23 : tensor<?x?xf32>) -> tensor<?x?xf32>
              %43:3 = affine.for %arg8 = 0 to affine_map<()[s0] -> (4096 ceildiv s0)>()[%14] iter_args(%arg9 = %42, %arg10 = %41, %arg11 = %40) -> (tensor<?x?xf32>, tensor<?x?xf32>, tensor<?x?x128xf32>) {
                %46 = arith.muli %arg8, %14 : index
                %47 = loom.subview %arg0[%36, 0, %46] [%12, 128, %14] [1, 1, 1], reuse : [seq = false, spat = false, temp = false] : memref<32x128x4096xf32> to memref<?x128x?xf32, strided<[524288, 4096, 1], offset: ?>>
                %48 = loom.copy_to_tensor %47, %19 on @L1, interconnect : [], broadcast : [1, 1] : memref<?x128x?xf32, strided<[524288, 4096, 1], offset: ?>>, memref<?x128x?xf32> -> tensor<?x128x?xf32>
                %49 = linalg.fill ins(%cst_1 : f32) outs(%18 : tensor<?x?x?xf32>) -> tensor<?x?x?xf32>
                %50 = linalg.batch_matmul ins(%39, %48 : tensor<?x?x128xf32>, tensor<?x128x?xf32>) outs(%49 : tensor<?x?x?xf32>) -> tensor<?x?x?xf32>
                %51 = linalg.fill ins(%cst_3 : f32) outs(%25 : tensor<?x?xf32>) -> tensor<?x?xf32>
                %52 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%50 : tensor<?x?x?xf32>) outs(%51 : tensor<?x?xf32>) {
                ^bb0(%in: f32, %out: f32):
                  %65 = arith.maximumf %in, %out : f32
                  linalg.yield %65 : f32
                } -> tensor<?x?xf32>
                %53 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%arg9, %52 : tensor<?x?xf32>, tensor<?x?xf32>) outs(%25 : tensor<?x?xf32>) {
                ^bb0(%in: f32, %in_4: f32, %out: f32):
                  %65 = arith.truncf %cst_0 : f64 to f32
                  %66 = arith.mulf %in_4, %65 : f32
                  %67 = arith.cmpf ogt, %in, %66 : f32
                  %68 = arith.select %67, %in, %66 : f32
                  linalg.yield %68 : f32
                } -> tensor<?x?xf32>
                %54 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%50, %53 : tensor<?x?x?xf32>, tensor<?x?xf32>) outs(%18 : tensor<?x?x?xf32>) {
                ^bb0(%in: f32, %in_4: f32, %out: f32):
                  %65 = arith.truncf %cst_0 : f64 to f32
                  %66 = arith.mulf %in, %65 : f32
                  %67 = arith.subf %66, %in_4 : f32
                  %68 = math.powf %cst, %67 : f32
                  linalg.yield %68 : f32
                } -> tensor<?x?x?xf32>
                %55 = linalg.fill ins(%cst_1 : f32) outs(%27 : tensor<?x?xf32>) -> tensor<?x?xf32>
                %56 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%54 : tensor<?x?x?xf32>) outs(%55 : tensor<?x?xf32>) {
                ^bb0(%in: f32, %out: f32):
                  %65 = arith.addf %in, %out : f32
                  linalg.yield %65 : f32
                } -> tensor<?x?xf32>
                %57 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%arg9, %53 : tensor<?x?xf32>, tensor<?x?xf32>) outs(%29 : tensor<?x?xf32>) {
                ^bb0(%in: f32, %in_4: f32, %out: f32):
                  %65 = arith.subf %in, %in_4 : f32
                  %66 = math.powf %cst, %65 : f32
                  linalg.yield %66 : f32
                } -> tensor<?x?xf32>
                %58 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%arg10, %57, %56 : tensor<?x?xf32>, tensor<?x?xf32>, tensor<?x?xf32>) outs(%arg10 : tensor<?x?xf32>) {
                ^bb0(%in: f32, %in_4: f32, %in_5: f32, %out: f32):
                  %65 = arith.mulf %in, %in_4 : f32
                  %66 = arith.addf %65, %in_5 : f32
                  linalg.yield %66 : f32
                } -> tensor<?x?xf32>
                %59 = loom.subview %arg1[%36, %46, 0] [%12, %14, 128] [1, 1, 1], reuse : [seq = false, spat = false, temp = false] : memref<32x4096x128xf32> to memref<?x?x128xf32, strided<[524288, 128, 1], offset: ?>>
                %60 = loom.copy_to_tensor %59, %16 on @L1, interconnect : [], broadcast : [1, 1] : memref<?x?x128xf32, strided<[524288, 128, 1], offset: ?>>, memref<?x?x128xf32> -> tensor<?x?x128xf32>
                %61 = linalg.fill ins(%cst_1 : f32) outs(%35 : tensor<?x?x128xf32>) -> tensor<?x?x128xf32>
                %62 = linalg.batch_matmul ins(%54, %60 : tensor<?x?x?xf32>, tensor<?x?x128xf32>) outs(%61 : tensor<?x?x128xf32>) -> tensor<?x?x128xf32>
                %63 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%62, %arg11, %57 : tensor<?x?x128xf32>, tensor<?x?x128xf32>, tensor<?x?xf32>) outs(%arg11 : tensor<?x?x128xf32>) {
                ^bb0(%in: f32, %in_4: f32, %in_5: f32, %out: f32):
                  %65 = arith.mulf %in_4, %in_5 : f32
                  %66 = arith.addf %in, %65 : f32
                  linalg.yield %66 : f32
                } -> tensor<?x?x128xf32>
                %64 = linalg.copy ins(%53 : tensor<?x?xf32>) outs(%arg9 : tensor<?x?xf32>) -> tensor<?x?xf32>
                affine.yield %64, %58, %63 : tensor<?x?xf32>, tensor<?x?xf32>, tensor<?x?x128xf32>
              } {loom.iter_type = #loom.iter_type<sequential>}
              %44 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%43#2, %43#1 : tensor<?x?x128xf32>, tensor<?x?xf32>) outs(%31 : tensor<?x?x128xf32>) {
              ^bb0(%in: f32, %in_4: f32, %out: f32):
                %46 = arith.divf %in, %in_4 : f32
                linalg.yield %46 : f32
              } -> tensor<?x?x128xf32>
              %45 = loom.subview %arg3[%36, %37, 0] [%12, %13, 128] [1, 1, 1], reuse : [seq = false, spat = false, temp = false] : memref<32x4096x128xf32> to memref<?x?x128xf32, strided<[524288, 128, 1], offset: ?>>
              loom.copy_from_tensor %44, %45 on @DRAM : tensor<?x?x128xf32>, memref<?x?x128xf32, strided<[524288, 128, 1], offset: ?>>
            } {loom.iter_type = #loom.iter_type<temporal>}
          } {loom.iter_type = #loom.iter_type<temporal>}
        } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @y}
      } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @x}
      return
    }
  }
  module attributes {loom.pass_name = "EnumerateHWMapping"} {
    loom.constraint_space @constraints {
      %12 = loom.symbolic_var "BB" : index
      %13 = loom.symbolic_var "BM" : index
      %14 = loom.symbolic_var "BN" : index
      loom.range %12[32, 32]
      loom.range %13[32, 4096]
      loom.range %14[32, 4096]
      loom.align %12 by 32
      loom.align %13 by 32
      loom.align %14 by 32
      loom.polynomial_constraint(%12, %13, %14) {monomials = [{coeff = -1 : i64, vars = [0, 1, 2]}], upper_bound = -262144 : i64}
      loom.polynomial_constraint(%12, %14, %13) {monomials = [{coeff = 4 : i64, vars = [0, 1]}, {coeff = 4 : i64, vars = [0, 2, 1]}, {coeff = 4 : i64, vars = [0, 1]}, {coeff = 4 : i64, vars = [0, 2]}, {coeff = 4 : i64, vars = [0, 2]}, {coeff = 4 : i64, vars = [0, 2]}, {coeff = 4 : i64, vars = [0, 2]}, {coeff = 4 : i64, vars = [0, 2]}, {coeff = 4 : i64, vars = [0, 2]}, {coeff = 4 : i64, vars = [0, 2]}, {coeff = 4 : i64, vars = [0, 2]}], upper_bound = 1499136 : i64}
      loom.polynomial_constraint(%12) {monomials = [{coeff = 64 : i64, vars = [0]}], upper_bound = 32 : i64}
    }
    func.func @attention__d0i0_d1i0__f10(%arg0: memref<32x128x4096xf32>, %arg1: memref<32x4096x128xf32>, %arg2: memref<32x4096x128xf32>, %arg3: memref<32x4096x128xf32>) {
      %cst = arith.constant 2.000000e+00 : f32
      %cst_0 = arith.constant 0.12751743074602467 : f64
      %cst_1 = arith.constant 0.000000e+00 : f32
      %cst_2 = arith.constant 1.000000e+00 : f32
      %cst_3 = arith.constant 0xFF800000 : f32
      %12 = loom.get_symbolic_block_size @constraints::@BB : index
      %13 = loom.get_symbolic_block_size @constraints::@BM : index
      %14 = loom.get_symbolic_block_size @constraints::@BN : index
      affine.parallel (%arg4) = (0) to (8) {
        affine.parallel (%arg5) = (0) to (8) {
          affine.for %arg6 = 0 to affine_map<()[s0, s1] -> (4096 ceildiv s1)>()[%12, %13] {
            affine.for %arg7 = 0 to affine_map<()[s0, s1] -> (((32 ceildiv s0) ceildiv 8) ceildiv 8)>()[%12, %13] {
              %15 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 8 + d1 + d2 * 64)>(%arg4, %arg5, %arg7)
              %16 = loom.alloc [%12, %14, 128] on @L1 : memref<?x?x128xf32>
              %17 = loom.alloc [%12, %13, %14] on @L1 : memref<?x?x?xf32>
              %18 = loom.init_tensor %17[%12, %13, %14] : memref<?x?x?xf32> -> tensor<?x?x?xf32>
              %19 = loom.alloc [%12, 128, %14] on @L1 : memref<?x128x?xf32>
              %20 = loom.alloc [%12, %13] on @L1 : memref<?x?xf32>
              %21 = loom.init_tensor %20[%12, %13] : memref<?x?xf32> -> tensor<?x?xf32>
              %22 = loom.alloc [%12, %13] on @L1 : memref<?x?xf32>
              %23 = loom.init_tensor %22[%12, %13] : memref<?x?xf32> -> tensor<?x?xf32>
              %24 = loom.alloc [%12, %13] on @L1 : memref<?x?xf32>
              %25 = loom.init_tensor %24[%12, %13] : memref<?x?xf32> -> tensor<?x?xf32>
              %26 = loom.alloc [%12, %13] on @L1 : memref<?x?xf32>
              %27 = loom.init_tensor %26[%12, %13] : memref<?x?xf32> -> tensor<?x?xf32>
              %28 = loom.alloc [%12, %13] on @L1 : memref<?x?xf32>
              %29 = loom.init_tensor %28[%12, %13] : memref<?x?xf32> -> tensor<?x?xf32>
              %30 = loom.alloc [%12, %13, 128] on @L1 : memref<?x?x128xf32>
              %31 = loom.init_tensor %30[%12, %13, 128] : memref<?x?x128xf32> -> tensor<?x?x128xf32>
              %32 = loom.alloc [%12, %13, 128] on @L1 : memref<?x?x128xf32>
              %33 = loom.init_tensor %32[%12, %13, 128] : memref<?x?x128xf32> -> tensor<?x?x128xf32>
              %34 = loom.alloc [%12, %13, 128] on @L1 : memref<?x?x128xf32>
              %35 = loom.init_tensor %34[%12, %13, 128] : memref<?x?x128xf32> -> tensor<?x?x128xf32>
              %36 = arith.muli %15, %12 : index
              %37 = arith.muli %arg6, %13 : index
              %38 = loom.subview %arg2[%36, %37, 0] [%12, %13, 128] [1, 1, 1], reuse : [seq = false, spat = false, temp = false] : memref<32x4096x128xf32> to memref<?x?x128xf32, strided<[524288, 128, 1], offset: ?>>
              %39 = loom.copy_to_tensor %38, %30 on @L1, interconnect : [], broadcast : [1, 1] : memref<?x?x128xf32, strided<[524288, 128, 1], offset: ?>>, memref<?x?x128xf32> -> tensor<?x?x128xf32>
              %40 = linalg.fill ins(%cst_1 : f32) outs(%33 : tensor<?x?x128xf32>) -> tensor<?x?x128xf32>
              %41 = linalg.fill ins(%cst_2 : f32) outs(%21 : tensor<?x?xf32>) -> tensor<?x?xf32>
              %42 = linalg.fill ins(%cst_3 : f32) outs(%23 : tensor<?x?xf32>) -> tensor<?x?xf32>
              %43:3 = affine.for %arg8 = 0 to affine_map<()[s0] -> (4096 ceildiv s0)>()[%14] iter_args(%arg9 = %42, %arg10 = %41, %arg11 = %40) -> (tensor<?x?xf32>, tensor<?x?xf32>, tensor<?x?x128xf32>) {
                %46 = arith.muli %arg8, %14 : index
                %47 = loom.subview %arg0[%36, 0, %46] [%12, 128, %14] [1, 1, 1], reuse : [seq = false, spat = false, temp = false] : memref<32x128x4096xf32> to memref<?x128x?xf32, strided<[524288, 4096, 1], offset: ?>>
                %48 = loom.copy_to_tensor %47, %19 on @L1, interconnect : [], broadcast : [1, 1] : memref<?x128x?xf32, strided<[524288, 4096, 1], offset: ?>>, memref<?x128x?xf32> -> tensor<?x128x?xf32>
                %49 = linalg.fill ins(%cst_1 : f32) outs(%18 : tensor<?x?x?xf32>) -> tensor<?x?x?xf32>
                %50 = linalg.batch_matmul ins(%39, %48 : tensor<?x?x128xf32>, tensor<?x128x?xf32>) outs(%49 : tensor<?x?x?xf32>) -> tensor<?x?x?xf32>
                %51 = linalg.fill ins(%cst_3 : f32) outs(%25 : tensor<?x?xf32>) -> tensor<?x?xf32>
                %52 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%50 : tensor<?x?x?xf32>) outs(%51 : tensor<?x?xf32>) {
                ^bb0(%in: f32, %out: f32):
                  %65 = arith.maximumf %in, %out : f32
                  linalg.yield %65 : f32
                } -> tensor<?x?xf32>
                %53 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%arg9, %52 : tensor<?x?xf32>, tensor<?x?xf32>) outs(%25 : tensor<?x?xf32>) {
                ^bb0(%in: f32, %in_4: f32, %out: f32):
                  %65 = arith.truncf %cst_0 : f64 to f32
                  %66 = arith.mulf %in_4, %65 : f32
                  %67 = arith.cmpf ogt, %in, %66 : f32
                  %68 = arith.select %67, %in, %66 : f32
                  linalg.yield %68 : f32
                } -> tensor<?x?xf32>
                %54 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%50, %53 : tensor<?x?x?xf32>, tensor<?x?xf32>) outs(%18 : tensor<?x?x?xf32>) {
                ^bb0(%in: f32, %in_4: f32, %out: f32):
                  %65 = arith.truncf %cst_0 : f64 to f32
                  %66 = arith.mulf %in, %65 : f32
                  %67 = arith.subf %66, %in_4 : f32
                  %68 = math.powf %cst, %67 : f32
                  linalg.yield %68 : f32
                } -> tensor<?x?x?xf32>
                %55 = linalg.fill ins(%cst_1 : f32) outs(%27 : tensor<?x?xf32>) -> tensor<?x?xf32>
                %56 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%54 : tensor<?x?x?xf32>) outs(%55 : tensor<?x?xf32>) {
                ^bb0(%in: f32, %out: f32):
                  %65 = arith.addf %in, %out : f32
                  linalg.yield %65 : f32
                } -> tensor<?x?xf32>
                %57 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%arg9, %53 : tensor<?x?xf32>, tensor<?x?xf32>) outs(%29 : tensor<?x?xf32>) {
                ^bb0(%in: f32, %in_4: f32, %out: f32):
                  %65 = arith.subf %in, %in_4 : f32
                  %66 = math.powf %cst, %65 : f32
                  linalg.yield %66 : f32
                } -> tensor<?x?xf32>
                %58 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%arg10, %57, %56 : tensor<?x?xf32>, tensor<?x?xf32>, tensor<?x?xf32>) outs(%arg10 : tensor<?x?xf32>) {
                ^bb0(%in: f32, %in_4: f32, %in_5: f32, %out: f32):
                  %65 = arith.mulf %in, %in_4 : f32
                  %66 = arith.addf %65, %in_5 : f32
                  linalg.yield %66 : f32
                } -> tensor<?x?xf32>
                %59 = loom.subview %arg1[%36, %46, 0] [%12, %14, 128] [1, 1, 1], reuse : [seq = false, spat = false, temp = false] : memref<32x4096x128xf32> to memref<?x?x128xf32, strided<[524288, 128, 1], offset: ?>>
                %60 = loom.copy_to_tensor %59, %16 on @L1, interconnect : [], broadcast : [1, 1] : memref<?x?x128xf32, strided<[524288, 128, 1], offset: ?>>, memref<?x?x128xf32> -> tensor<?x?x128xf32>
                %61 = linalg.fill ins(%cst_1 : f32) outs(%35 : tensor<?x?x128xf32>) -> tensor<?x?x128xf32>
                %62 = linalg.batch_matmul ins(%54, %60 : tensor<?x?x?xf32>, tensor<?x?x128xf32>) outs(%61 : tensor<?x?x128xf32>) -> tensor<?x?x128xf32>
                %63 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%62, %arg11, %57 : tensor<?x?x128xf32>, tensor<?x?x128xf32>, tensor<?x?xf32>) outs(%arg11 : tensor<?x?x128xf32>) {
                ^bb0(%in: f32, %in_4: f32, %in_5: f32, %out: f32):
                  %65 = arith.mulf %in_4, %in_5 : f32
                  %66 = arith.addf %in, %65 : f32
                  linalg.yield %66 : f32
                } -> tensor<?x?x128xf32>
                %64 = linalg.copy ins(%53 : tensor<?x?xf32>) outs(%arg9 : tensor<?x?xf32>) -> tensor<?x?xf32>
                affine.yield %64, %58, %63 : tensor<?x?xf32>, tensor<?x?xf32>, tensor<?x?x128xf32>
              } {loom.iter_type = #loom.iter_type<sequential>}
              %44 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%43#2, %43#1 : tensor<?x?x128xf32>, tensor<?x?xf32>) outs(%31 : tensor<?x?x128xf32>) {
              ^bb0(%in: f32, %in_4: f32, %out: f32):
                %46 = arith.divf %in, %in_4 : f32
                linalg.yield %46 : f32
              } -> tensor<?x?x128xf32>
              %45 = loom.subview %arg3[%36, %37, 0] [%12, %13, 128] [1, 1, 1], reuse : [seq = false, spat = false, temp = false] : memref<32x4096x128xf32> to memref<?x?x128xf32, strided<[524288, 128, 1], offset: ?>>
              loom.copy_from_tensor %44, %45 on @DRAM : tensor<?x?x128xf32>, memref<?x?x128xf32, strided<[524288, 128, 1], offset: ?>>
            } {loom.iter_type = #loom.iter_type<temporal>}
          } {loom.iter_type = #loom.iter_type<temporal>}
        } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @y}
      } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @x}
      return
    }
  }
  module attributes {loom.pass_name = "EnumerateHWMapping"} {
    loom.constraint_space @constraints {
      %12 = loom.symbolic_var "BB" : index
      %13 = loom.symbolic_var "BM" : index
      %14 = loom.symbolic_var "BN" : index
      loom.range %12[32, 32]
      loom.range %13[32, 4096]
      loom.range %14[32, 4096]
      loom.align %12 by 32
      loom.align %13 by 32
      loom.align %14 by 32
      loom.polynomial_constraint(%12, %13, %14) {monomials = [{coeff = -1 : i64, vars = [0, 1, 2]}], upper_bound = -262144 : i64}
      loom.polynomial_constraint(%12, %14, %13) {monomials = [{coeff = 4 : i64, vars = [0, 1]}, {coeff = 4 : i64, vars = [0, 2, 1]}, {coeff = 4 : i64, vars = [0, 1]}, {coeff = 4 : i64, vars = [0, 2]}, {coeff = 4 : i64, vars = [0, 2]}, {coeff = 4 : i64, vars = [0, 2]}, {coeff = 4 : i64, vars = [0, 2]}, {coeff = 4 : i64, vars = [0, 2]}, {coeff = 4 : i64, vars = [0, 2]}, {coeff = 4 : i64, vars = [0, 2]}, {coeff = 4 : i64, vars = [0, 2]}], upper_bound = 1499136 : i64}
      loom.polynomial_constraint(%12) {monomials = [{coeff = 8 : i64, vars = [0]}], upper_bound = 32 : i64}
      loom.polynomial_constraint(%13) {monomials = [{coeff = 8 : i64, vars = [0]}], upper_bound = 4096 : i64}
    }
    func.func @attention__d0i0_d1i1__f01(%arg0: memref<32x128x4096xf32>, %arg1: memref<32x4096x128xf32>, %arg2: memref<32x4096x128xf32>, %arg3: memref<32x4096x128xf32>) {
      %cst = arith.constant 2.000000e+00 : f32
      %cst_0 = arith.constant 0.12751743074602467 : f64
      %cst_1 = arith.constant 0.000000e+00 : f32
      %cst_2 = arith.constant 1.000000e+00 : f32
      %cst_3 = arith.constant 0xFF800000 : f32
      %12 = loom.get_symbolic_block_size @constraints::@BB : index
      %13 = loom.get_symbolic_block_size @constraints::@BM : index
      %14 = loom.get_symbolic_block_size @constraints::@BN : index
      affine.parallel (%arg4) = (0) to (8) {
        affine.parallel (%arg5) = (0) to (8) {
          affine.for %arg6 = 0 to affine_map<()[s0, s1] -> ((32 ceildiv s0) ceildiv 8)>()[%12, %13] {
            affine.for %arg7 = 0 to affine_map<()[s0, s1] -> ((4096 ceildiv s1) ceildiv 8)>()[%12, %13] {
              %15 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 8)>(%arg4, %arg6)
              %16 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 8)>(%arg5, %arg7)
              %17 = loom.alloc [%12, %14, 128] on @L1 : memref<?x?x128xf32>
              %18 = loom.alloc [%12, %13, %14] on @L1 : memref<?x?x?xf32>
              %19 = loom.init_tensor %18[%12, %13, %14] : memref<?x?x?xf32> -> tensor<?x?x?xf32>
              %20 = loom.alloc [%12, 128, %14] on @L1 : memref<?x128x?xf32>
              %21 = loom.alloc [%12, %13] on @L1 : memref<?x?xf32>
              %22 = loom.init_tensor %21[%12, %13] : memref<?x?xf32> -> tensor<?x?xf32>
              %23 = loom.alloc [%12, %13] on @L1 : memref<?x?xf32>
              %24 = loom.init_tensor %23[%12, %13] : memref<?x?xf32> -> tensor<?x?xf32>
              %25 = loom.alloc [%12, %13] on @L1 : memref<?x?xf32>
              %26 = loom.init_tensor %25[%12, %13] : memref<?x?xf32> -> tensor<?x?xf32>
              %27 = loom.alloc [%12, %13] on @L1 : memref<?x?xf32>
              %28 = loom.init_tensor %27[%12, %13] : memref<?x?xf32> -> tensor<?x?xf32>
              %29 = loom.alloc [%12, %13] on @L1 : memref<?x?xf32>
              %30 = loom.init_tensor %29[%12, %13] : memref<?x?xf32> -> tensor<?x?xf32>
              %31 = loom.alloc [%12, %13, 128] on @L1 : memref<?x?x128xf32>
              %32 = loom.init_tensor %31[%12, %13, 128] : memref<?x?x128xf32> -> tensor<?x?x128xf32>
              %33 = loom.alloc [%12, %13, 128] on @L1 : memref<?x?x128xf32>
              %34 = loom.init_tensor %33[%12, %13, 128] : memref<?x?x128xf32> -> tensor<?x?x128xf32>
              %35 = loom.alloc [%12, %13, 128] on @L1 : memref<?x?x128xf32>
              %36 = loom.init_tensor %35[%12, %13, 128] : memref<?x?x128xf32> -> tensor<?x?x128xf32>
              %37 = arith.muli %15, %12 : index
              %38 = arith.muli %16, %13 : index
              %39 = loom.subview %arg2[%37, %38, 0] [%12, %13, 128] [1, 1, 1], reuse : [seq = false, spat = false, temp = false] : memref<32x4096x128xf32> to memref<?x?x128xf32, strided<[524288, 128, 1], offset: ?>>
              %40 = loom.copy_to_tensor %39, %31 on @L1, interconnect : [], broadcast : [1, 1] : memref<?x?x128xf32, strided<[524288, 128, 1], offset: ?>>, memref<?x?x128xf32> -> tensor<?x?x128xf32>
              %41 = linalg.fill ins(%cst_1 : f32) outs(%34 : tensor<?x?x128xf32>) -> tensor<?x?x128xf32>
              %42 = linalg.fill ins(%cst_2 : f32) outs(%22 : tensor<?x?xf32>) -> tensor<?x?xf32>
              %43 = linalg.fill ins(%cst_3 : f32) outs(%24 : tensor<?x?xf32>) -> tensor<?x?xf32>
              %44:3 = affine.for %arg8 = 0 to affine_map<()[s0] -> (4096 ceildiv s0)>()[%14] iter_args(%arg9 = %43, %arg10 = %42, %arg11 = %41) -> (tensor<?x?xf32>, tensor<?x?xf32>, tensor<?x?x128xf32>) {
                %47 = arith.muli %arg8, %14 : index
                %48 = loom.subview %arg0[%37, 0, %47] [%12, 128, %14] [1, 1, 1], reuse : [seq = false, spat = false, temp = false] : memref<32x128x4096xf32> to memref<?x128x?xf32, strided<[524288, 4096, 1], offset: ?>>
                %49 = loom.copy_to_tensor %48, %20 on @L1, interconnect : [], broadcast : [1, 1] : memref<?x128x?xf32, strided<[524288, 4096, 1], offset: ?>>, memref<?x128x?xf32> -> tensor<?x128x?xf32>
                %50 = linalg.fill ins(%cst_1 : f32) outs(%19 : tensor<?x?x?xf32>) -> tensor<?x?x?xf32>
                %51 = linalg.batch_matmul ins(%40, %49 : tensor<?x?x128xf32>, tensor<?x128x?xf32>) outs(%50 : tensor<?x?x?xf32>) -> tensor<?x?x?xf32>
                %52 = linalg.fill ins(%cst_3 : f32) outs(%26 : tensor<?x?xf32>) -> tensor<?x?xf32>
                %53 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%51 : tensor<?x?x?xf32>) outs(%52 : tensor<?x?xf32>) {
                ^bb0(%in: f32, %out: f32):
                  %66 = arith.maximumf %in, %out : f32
                  linalg.yield %66 : f32
                } -> tensor<?x?xf32>
                %54 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%arg9, %53 : tensor<?x?xf32>, tensor<?x?xf32>) outs(%26 : tensor<?x?xf32>) {
                ^bb0(%in: f32, %in_4: f32, %out: f32):
                  %66 = arith.truncf %cst_0 : f64 to f32
                  %67 = arith.mulf %in_4, %66 : f32
                  %68 = arith.cmpf ogt, %in, %67 : f32
                  %69 = arith.select %68, %in, %67 : f32
                  linalg.yield %69 : f32
                } -> tensor<?x?xf32>
                %55 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%51, %54 : tensor<?x?x?xf32>, tensor<?x?xf32>) outs(%19 : tensor<?x?x?xf32>) {
                ^bb0(%in: f32, %in_4: f32, %out: f32):
                  %66 = arith.truncf %cst_0 : f64 to f32
                  %67 = arith.mulf %in, %66 : f32
                  %68 = arith.subf %67, %in_4 : f32
                  %69 = math.powf %cst, %68 : f32
                  linalg.yield %69 : f32
                } -> tensor<?x?x?xf32>
                %56 = linalg.fill ins(%cst_1 : f32) outs(%28 : tensor<?x?xf32>) -> tensor<?x?xf32>
                %57 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%55 : tensor<?x?x?xf32>) outs(%56 : tensor<?x?xf32>) {
                ^bb0(%in: f32, %out: f32):
                  %66 = arith.addf %in, %out : f32
                  linalg.yield %66 : f32
                } -> tensor<?x?xf32>
                %58 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%arg9, %54 : tensor<?x?xf32>, tensor<?x?xf32>) outs(%30 : tensor<?x?xf32>) {
                ^bb0(%in: f32, %in_4: f32, %out: f32):
                  %66 = arith.subf %in, %in_4 : f32
                  %67 = math.powf %cst, %66 : f32
                  linalg.yield %67 : f32
                } -> tensor<?x?xf32>
                %59 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%arg10, %58, %57 : tensor<?x?xf32>, tensor<?x?xf32>, tensor<?x?xf32>) outs(%arg10 : tensor<?x?xf32>) {
                ^bb0(%in: f32, %in_4: f32, %in_5: f32, %out: f32):
                  %66 = arith.mulf %in, %in_4 : f32
                  %67 = arith.addf %66, %in_5 : f32
                  linalg.yield %67 : f32
                } -> tensor<?x?xf32>
                %60 = loom.subview %arg1[%37, %47, 0] [%12, %14, 128] [1, 1, 1], reuse : [seq = false, spat = false, temp = false] : memref<32x4096x128xf32> to memref<?x?x128xf32, strided<[524288, 128, 1], offset: ?>>
                %61 = loom.copy_to_tensor %60, %17 on @L1, interconnect : [], broadcast : [1, 1] : memref<?x?x128xf32, strided<[524288, 128, 1], offset: ?>>, memref<?x?x128xf32> -> tensor<?x?x128xf32>
                %62 = linalg.fill ins(%cst_1 : f32) outs(%36 : tensor<?x?x128xf32>) -> tensor<?x?x128xf32>
                %63 = linalg.batch_matmul ins(%55, %61 : tensor<?x?x?xf32>, tensor<?x?x128xf32>) outs(%62 : tensor<?x?x128xf32>) -> tensor<?x?x128xf32>
                %64 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%63, %arg11, %58 : tensor<?x?x128xf32>, tensor<?x?x128xf32>, tensor<?x?xf32>) outs(%arg11 : tensor<?x?x128xf32>) {
                ^bb0(%in: f32, %in_4: f32, %in_5: f32, %out: f32):
                  %66 = arith.mulf %in_4, %in_5 : f32
                  %67 = arith.addf %in, %66 : f32
                  linalg.yield %67 : f32
                } -> tensor<?x?x128xf32>
                %65 = linalg.copy ins(%54 : tensor<?x?xf32>) outs(%arg9 : tensor<?x?xf32>) -> tensor<?x?xf32>
                affine.yield %65, %59, %64 : tensor<?x?xf32>, tensor<?x?xf32>, tensor<?x?x128xf32>
              } {loom.iter_type = #loom.iter_type<sequential>}
              %45 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%44#2, %44#1 : tensor<?x?x128xf32>, tensor<?x?xf32>) outs(%32 : tensor<?x?x128xf32>) {
              ^bb0(%in: f32, %in_4: f32, %out: f32):
                %47 = arith.divf %in, %in_4 : f32
                linalg.yield %47 : f32
              } -> tensor<?x?x128xf32>
              %46 = loom.subview %arg3[%37, %38, 0] [%12, %13, 128] [1, 1, 1], reuse : [seq = false, spat = false, temp = false] : memref<32x4096x128xf32> to memref<?x?x128xf32, strided<[524288, 128, 1], offset: ?>>
              loom.copy_from_tensor %45, %46 on @DRAM : tensor<?x?x128xf32>, memref<?x?x128xf32, strided<[524288, 128, 1], offset: ?>>
            } {loom.iter_type = #loom.iter_type<temporal>}
          } {loom.iter_type = #loom.iter_type<temporal>}
        } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @y}
      } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @x}
      return
    }
  }
  module attributes {loom.pass_name = "EnumerateHWMapping"} {
    loom.constraint_space @constraints {
      %12 = loom.symbolic_var "BB" : index
      %13 = loom.symbolic_var "BM" : index
      %14 = loom.symbolic_var "BN" : index
      loom.range %12[32, 32]
      loom.range %13[32, 4096]
      loom.range %14[32, 4096]
      loom.align %12 by 32
      loom.align %13 by 32
      loom.align %14 by 32
      loom.polynomial_constraint(%12, %13, %14) {monomials = [{coeff = -1 : i64, vars = [0, 1, 2]}], upper_bound = -262144 : i64}
      loom.polynomial_constraint(%12, %14, %13) {monomials = [{coeff = 4 : i64, vars = [0, 1]}, {coeff = 4 : i64, vars = [0, 2, 1]}, {coeff = 4 : i64, vars = [0, 1]}, {coeff = 4 : i64, vars = [0, 2]}, {coeff = 4 : i64, vars = [0, 2]}, {coeff = 4 : i64, vars = [0, 2]}, {coeff = 4 : i64, vars = [0, 2]}, {coeff = 4 : i64, vars = [0, 2]}, {coeff = 4 : i64, vars = [0, 2]}, {coeff = 4 : i64, vars = [0, 2]}, {coeff = 4 : i64, vars = [0, 2]}], upper_bound = 1499136 : i64}
      loom.polynomial_constraint(%12) {monomials = [{coeff = 8 : i64, vars = [0]}], upper_bound = 32 : i64}
      loom.polynomial_constraint(%13) {monomials = [{coeff = 8 : i64, vars = [0]}], upper_bound = 4096 : i64}
    }
    func.func @attention__d0i0_d1i1__f10(%arg0: memref<32x128x4096xf32>, %arg1: memref<32x4096x128xf32>, %arg2: memref<32x4096x128xf32>, %arg3: memref<32x4096x128xf32>) {
      %cst = arith.constant 2.000000e+00 : f32
      %cst_0 = arith.constant 0.12751743074602467 : f64
      %cst_1 = arith.constant 0.000000e+00 : f32
      %cst_2 = arith.constant 1.000000e+00 : f32
      %cst_3 = arith.constant 0xFF800000 : f32
      %12 = loom.get_symbolic_block_size @constraints::@BB : index
      %13 = loom.get_symbolic_block_size @constraints::@BM : index
      %14 = loom.get_symbolic_block_size @constraints::@BN : index
      affine.parallel (%arg4) = (0) to (8) {
        affine.parallel (%arg5) = (0) to (8) {
          affine.for %arg6 = 0 to affine_map<()[s0, s1] -> ((4096 ceildiv s1) ceildiv 8)>()[%12, %13] {
            affine.for %arg7 = 0 to affine_map<()[s0, s1] -> ((32 ceildiv s0) ceildiv 8)>()[%12, %13] {
              %15 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 8)>(%arg4, %arg7)
              %16 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 8)>(%arg5, %arg6)
              %17 = loom.alloc [%12, %14, 128] on @L1 : memref<?x?x128xf32>
              %18 = loom.alloc [%12, %13, %14] on @L1 : memref<?x?x?xf32>
              %19 = loom.init_tensor %18[%12, %13, %14] : memref<?x?x?xf32> -> tensor<?x?x?xf32>
              %20 = loom.alloc [%12, 128, %14] on @L1 : memref<?x128x?xf32>
              %21 = loom.alloc [%12, %13] on @L1 : memref<?x?xf32>
              %22 = loom.init_tensor %21[%12, %13] : memref<?x?xf32> -> tensor<?x?xf32>
              %23 = loom.alloc [%12, %13] on @L1 : memref<?x?xf32>
              %24 = loom.init_tensor %23[%12, %13] : memref<?x?xf32> -> tensor<?x?xf32>
              %25 = loom.alloc [%12, %13] on @L1 : memref<?x?xf32>
              %26 = loom.init_tensor %25[%12, %13] : memref<?x?xf32> -> tensor<?x?xf32>
              %27 = loom.alloc [%12, %13] on @L1 : memref<?x?xf32>
              %28 = loom.init_tensor %27[%12, %13] : memref<?x?xf32> -> tensor<?x?xf32>
              %29 = loom.alloc [%12, %13] on @L1 : memref<?x?xf32>
              %30 = loom.init_tensor %29[%12, %13] : memref<?x?xf32> -> tensor<?x?xf32>
              %31 = loom.alloc [%12, %13, 128] on @L1 : memref<?x?x128xf32>
              %32 = loom.init_tensor %31[%12, %13, 128] : memref<?x?x128xf32> -> tensor<?x?x128xf32>
              %33 = loom.alloc [%12, %13, 128] on @L1 : memref<?x?x128xf32>
              %34 = loom.init_tensor %33[%12, %13, 128] : memref<?x?x128xf32> -> tensor<?x?x128xf32>
              %35 = loom.alloc [%12, %13, 128] on @L1 : memref<?x?x128xf32>
              %36 = loom.init_tensor %35[%12, %13, 128] : memref<?x?x128xf32> -> tensor<?x?x128xf32>
              %37 = arith.muli %15, %12 : index
              %38 = arith.muli %16, %13 : index
              %39 = loom.subview %arg2[%37, %38, 0] [%12, %13, 128] [1, 1, 1], reuse : [seq = false, spat = false, temp = false] : memref<32x4096x128xf32> to memref<?x?x128xf32, strided<[524288, 128, 1], offset: ?>>
              %40 = loom.copy_to_tensor %39, %31 on @L1, interconnect : [], broadcast : [1, 1] : memref<?x?x128xf32, strided<[524288, 128, 1], offset: ?>>, memref<?x?x128xf32> -> tensor<?x?x128xf32>
              %41 = linalg.fill ins(%cst_1 : f32) outs(%34 : tensor<?x?x128xf32>) -> tensor<?x?x128xf32>
              %42 = linalg.fill ins(%cst_2 : f32) outs(%22 : tensor<?x?xf32>) -> tensor<?x?xf32>
              %43 = linalg.fill ins(%cst_3 : f32) outs(%24 : tensor<?x?xf32>) -> tensor<?x?xf32>
              %44:3 = affine.for %arg8 = 0 to affine_map<()[s0] -> (4096 ceildiv s0)>()[%14] iter_args(%arg9 = %43, %arg10 = %42, %arg11 = %41) -> (tensor<?x?xf32>, tensor<?x?xf32>, tensor<?x?x128xf32>) {
                %47 = arith.muli %arg8, %14 : index
                %48 = loom.subview %arg0[%37, 0, %47] [%12, 128, %14] [1, 1, 1], reuse : [seq = false, spat = false, temp = false] : memref<32x128x4096xf32> to memref<?x128x?xf32, strided<[524288, 4096, 1], offset: ?>>
                %49 = loom.copy_to_tensor %48, %20 on @L1, interconnect : [], broadcast : [1, 1] : memref<?x128x?xf32, strided<[524288, 4096, 1], offset: ?>>, memref<?x128x?xf32> -> tensor<?x128x?xf32>
                %50 = linalg.fill ins(%cst_1 : f32) outs(%19 : tensor<?x?x?xf32>) -> tensor<?x?x?xf32>
                %51 = linalg.batch_matmul ins(%40, %49 : tensor<?x?x128xf32>, tensor<?x128x?xf32>) outs(%50 : tensor<?x?x?xf32>) -> tensor<?x?x?xf32>
                %52 = linalg.fill ins(%cst_3 : f32) outs(%26 : tensor<?x?xf32>) -> tensor<?x?xf32>
                %53 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%51 : tensor<?x?x?xf32>) outs(%52 : tensor<?x?xf32>) {
                ^bb0(%in: f32, %out: f32):
                  %66 = arith.maximumf %in, %out : f32
                  linalg.yield %66 : f32
                } -> tensor<?x?xf32>
                %54 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%arg9, %53 : tensor<?x?xf32>, tensor<?x?xf32>) outs(%26 : tensor<?x?xf32>) {
                ^bb0(%in: f32, %in_4: f32, %out: f32):
                  %66 = arith.truncf %cst_0 : f64 to f32
                  %67 = arith.mulf %in_4, %66 : f32
                  %68 = arith.cmpf ogt, %in, %67 : f32
                  %69 = arith.select %68, %in, %67 : f32
                  linalg.yield %69 : f32
                } -> tensor<?x?xf32>
                %55 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%51, %54 : tensor<?x?x?xf32>, tensor<?x?xf32>) outs(%19 : tensor<?x?x?xf32>) {
                ^bb0(%in: f32, %in_4: f32, %out: f32):
                  %66 = arith.truncf %cst_0 : f64 to f32
                  %67 = arith.mulf %in, %66 : f32
                  %68 = arith.subf %67, %in_4 : f32
                  %69 = math.powf %cst, %68 : f32
                  linalg.yield %69 : f32
                } -> tensor<?x?x?xf32>
                %56 = linalg.fill ins(%cst_1 : f32) outs(%28 : tensor<?x?xf32>) -> tensor<?x?xf32>
                %57 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%55 : tensor<?x?x?xf32>) outs(%56 : tensor<?x?xf32>) {
                ^bb0(%in: f32, %out: f32):
                  %66 = arith.addf %in, %out : f32
                  linalg.yield %66 : f32
                } -> tensor<?x?xf32>
                %58 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%arg9, %54 : tensor<?x?xf32>, tensor<?x?xf32>) outs(%30 : tensor<?x?xf32>) {
                ^bb0(%in: f32, %in_4: f32, %out: f32):
                  %66 = arith.subf %in, %in_4 : f32
                  %67 = math.powf %cst, %66 : f32
                  linalg.yield %67 : f32
                } -> tensor<?x?xf32>
                %59 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%arg10, %58, %57 : tensor<?x?xf32>, tensor<?x?xf32>, tensor<?x?xf32>) outs(%arg10 : tensor<?x?xf32>) {
                ^bb0(%in: f32, %in_4: f32, %in_5: f32, %out: f32):
                  %66 = arith.mulf %in, %in_4 : f32
                  %67 = arith.addf %66, %in_5 : f32
                  linalg.yield %67 : f32
                } -> tensor<?x?xf32>
                %60 = loom.subview %arg1[%37, %47, 0] [%12, %14, 128] [1, 1, 1], reuse : [seq = false, spat = false, temp = false] : memref<32x4096x128xf32> to memref<?x?x128xf32, strided<[524288, 128, 1], offset: ?>>
                %61 = loom.copy_to_tensor %60, %17 on @L1, interconnect : [], broadcast : [1, 1] : memref<?x?x128xf32, strided<[524288, 128, 1], offset: ?>>, memref<?x?x128xf32> -> tensor<?x?x128xf32>
                %62 = linalg.fill ins(%cst_1 : f32) outs(%36 : tensor<?x?x128xf32>) -> tensor<?x?x128xf32>
                %63 = linalg.batch_matmul ins(%55, %61 : tensor<?x?x?xf32>, tensor<?x?x128xf32>) outs(%62 : tensor<?x?x128xf32>) -> tensor<?x?x128xf32>
                %64 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%63, %arg11, %58 : tensor<?x?x128xf32>, tensor<?x?x128xf32>, tensor<?x?xf32>) outs(%arg11 : tensor<?x?x128xf32>) {
                ^bb0(%in: f32, %in_4: f32, %in_5: f32, %out: f32):
                  %66 = arith.mulf %in_4, %in_5 : f32
                  %67 = arith.addf %in, %66 : f32
                  linalg.yield %67 : f32
                } -> tensor<?x?x128xf32>
                %65 = linalg.copy ins(%54 : tensor<?x?xf32>) outs(%arg9 : tensor<?x?xf32>) -> tensor<?x?xf32>
                affine.yield %65, %59, %64 : tensor<?x?xf32>, tensor<?x?xf32>, tensor<?x?x128xf32>
              } {loom.iter_type = #loom.iter_type<sequential>}
              %45 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%44#2, %44#1 : tensor<?x?x128xf32>, tensor<?x?xf32>) outs(%32 : tensor<?x?x128xf32>) {
              ^bb0(%in: f32, %in_4: f32, %out: f32):
                %47 = arith.divf %in, %in_4 : f32
                linalg.yield %47 : f32
              } -> tensor<?x?x128xf32>
              %46 = loom.subview %arg3[%37, %38, 0] [%12, %13, 128] [1, 1, 1], reuse : [seq = false, spat = false, temp = false] : memref<32x4096x128xf32> to memref<?x?x128xf32, strided<[524288, 128, 1], offset: ?>>
              loom.copy_from_tensor %45, %46 on @DRAM : tensor<?x?x128xf32>, memref<?x?x128xf32, strided<[524288, 128, 1], offset: ?>>
            } {loom.iter_type = #loom.iter_type<temporal>}
          } {loom.iter_type = #loom.iter_type<temporal>}
        } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @y}
      } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @x}
      return
    }
  }
  module attributes {loom.pass_name = "EnumerateHWMapping"} {
    loom.constraint_space @constraints {
      %12 = loom.symbolic_var "BB" : index
      %13 = loom.symbolic_var "BM" : index
      %14 = loom.symbolic_var "BN" : index
      loom.range %12[32, 32]
      loom.range %13[32, 4096]
      loom.range %14[32, 4096]
      loom.align %12 by 32
      loom.align %13 by 32
      loom.align %14 by 32
      loom.polynomial_constraint(%12, %13, %14) {monomials = [{coeff = -1 : i64, vars = [0, 1, 2]}], upper_bound = -262144 : i64}
      loom.polynomial_constraint(%12, %14, %13) {monomials = [{coeff = 4 : i64, vars = [0, 1]}, {coeff = 4 : i64, vars = [0, 2, 1]}, {coeff = 4 : i64, vars = [0, 1]}, {coeff = 4 : i64, vars = [0, 2]}, {coeff = 4 : i64, vars = [0, 2]}, {coeff = 4 : i64, vars = [0, 2]}, {coeff = 4 : i64, vars = [0, 2]}, {coeff = 4 : i64, vars = [0, 2]}, {coeff = 4 : i64, vars = [0, 2]}, {coeff = 4 : i64, vars = [0, 2]}, {coeff = 4 : i64, vars = [0, 2]}], upper_bound = 1499136 : i64}
      loom.polynomial_constraint(%12) {monomials = [{coeff = 8 : i64, vars = [0]}], upper_bound = 32 : i64}
      loom.polynomial_constraint(%13) {monomials = [{coeff = 8 : i64, vars = [0]}], upper_bound = 4096 : i64}
    }
    func.func @attention__d1i0_d0i1__f01(%arg0: memref<32x128x4096xf32>, %arg1: memref<32x4096x128xf32>, %arg2: memref<32x4096x128xf32>, %arg3: memref<32x4096x128xf32>) {
      %cst = arith.constant 2.000000e+00 : f32
      %cst_0 = arith.constant 0.12751743074602467 : f64
      %cst_1 = arith.constant 0.000000e+00 : f32
      %cst_2 = arith.constant 1.000000e+00 : f32
      %cst_3 = arith.constant 0xFF800000 : f32
      %12 = loom.get_symbolic_block_size @constraints::@BB : index
      %13 = loom.get_symbolic_block_size @constraints::@BM : index
      %14 = loom.get_symbolic_block_size @constraints::@BN : index
      affine.parallel (%arg4) = (0) to (8) {
        affine.parallel (%arg5) = (0) to (8) {
          affine.for %arg6 = 0 to affine_map<()[s0, s1] -> ((32 ceildiv s0) ceildiv 8)>()[%12, %13] {
            affine.for %arg7 = 0 to affine_map<()[s0, s1] -> ((4096 ceildiv s1) ceildiv 8)>()[%12, %13] {
              %15 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 8)>(%arg4, %arg6)
              %16 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 8)>(%arg5, %arg7)
              %17 = loom.alloc [%12, %14, 128] on @L1 : memref<?x?x128xf32>
              %18 = loom.alloc [%12, %13, %14] on @L1 : memref<?x?x?xf32>
              %19 = loom.init_tensor %18[%12, %13, %14] : memref<?x?x?xf32> -> tensor<?x?x?xf32>
              %20 = loom.alloc [%12, 128, %14] on @L1 : memref<?x128x?xf32>
              %21 = loom.alloc [%12, %13] on @L1 : memref<?x?xf32>
              %22 = loom.init_tensor %21[%12, %13] : memref<?x?xf32> -> tensor<?x?xf32>
              %23 = loom.alloc [%12, %13] on @L1 : memref<?x?xf32>
              %24 = loom.init_tensor %23[%12, %13] : memref<?x?xf32> -> tensor<?x?xf32>
              %25 = loom.alloc [%12, %13] on @L1 : memref<?x?xf32>
              %26 = loom.init_tensor %25[%12, %13] : memref<?x?xf32> -> tensor<?x?xf32>
              %27 = loom.alloc [%12, %13] on @L1 : memref<?x?xf32>
              %28 = loom.init_tensor %27[%12, %13] : memref<?x?xf32> -> tensor<?x?xf32>
              %29 = loom.alloc [%12, %13] on @L1 : memref<?x?xf32>
              %30 = loom.init_tensor %29[%12, %13] : memref<?x?xf32> -> tensor<?x?xf32>
              %31 = loom.alloc [%12, %13, 128] on @L1 : memref<?x?x128xf32>
              %32 = loom.init_tensor %31[%12, %13, 128] : memref<?x?x128xf32> -> tensor<?x?x128xf32>
              %33 = loom.alloc [%12, %13, 128] on @L1 : memref<?x?x128xf32>
              %34 = loom.init_tensor %33[%12, %13, 128] : memref<?x?x128xf32> -> tensor<?x?x128xf32>
              %35 = loom.alloc [%12, %13, 128] on @L1 : memref<?x?x128xf32>
              %36 = loom.init_tensor %35[%12, %13, 128] : memref<?x?x128xf32> -> tensor<?x?x128xf32>
              %37 = arith.muli %15, %12 : index
              %38 = arith.muli %16, %13 : index
              %39 = loom.subview %arg2[%37, %38, 0] [%12, %13, 128] [1, 1, 1], reuse : [seq = false, spat = false, temp = false] : memref<32x4096x128xf32> to memref<?x?x128xf32, strided<[524288, 128, 1], offset: ?>>
              %40 = loom.copy_to_tensor %39, %31 on @L1, interconnect : [], broadcast : [1, 1] : memref<?x?x128xf32, strided<[524288, 128, 1], offset: ?>>, memref<?x?x128xf32> -> tensor<?x?x128xf32>
              %41 = linalg.fill ins(%cst_1 : f32) outs(%34 : tensor<?x?x128xf32>) -> tensor<?x?x128xf32>
              %42 = linalg.fill ins(%cst_2 : f32) outs(%22 : tensor<?x?xf32>) -> tensor<?x?xf32>
              %43 = linalg.fill ins(%cst_3 : f32) outs(%24 : tensor<?x?xf32>) -> tensor<?x?xf32>
              %44:3 = affine.for %arg8 = 0 to affine_map<()[s0] -> (4096 ceildiv s0)>()[%14] iter_args(%arg9 = %43, %arg10 = %42, %arg11 = %41) -> (tensor<?x?xf32>, tensor<?x?xf32>, tensor<?x?x128xf32>) {
                %47 = arith.muli %arg8, %14 : index
                %48 = loom.subview %arg0[%37, 0, %47] [%12, 128, %14] [1, 1, 1], reuse : [seq = false, spat = false, temp = false] : memref<32x128x4096xf32> to memref<?x128x?xf32, strided<[524288, 4096, 1], offset: ?>>
                %49 = loom.copy_to_tensor %48, %20 on @L1, interconnect : [], broadcast : [1, 1] : memref<?x128x?xf32, strided<[524288, 4096, 1], offset: ?>>, memref<?x128x?xf32> -> tensor<?x128x?xf32>
                %50 = linalg.fill ins(%cst_1 : f32) outs(%19 : tensor<?x?x?xf32>) -> tensor<?x?x?xf32>
                %51 = linalg.batch_matmul ins(%40, %49 : tensor<?x?x128xf32>, tensor<?x128x?xf32>) outs(%50 : tensor<?x?x?xf32>) -> tensor<?x?x?xf32>
                %52 = linalg.fill ins(%cst_3 : f32) outs(%26 : tensor<?x?xf32>) -> tensor<?x?xf32>
                %53 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%51 : tensor<?x?x?xf32>) outs(%52 : tensor<?x?xf32>) {
                ^bb0(%in: f32, %out: f32):
                  %66 = arith.maximumf %in, %out : f32
                  linalg.yield %66 : f32
                } -> tensor<?x?xf32>
                %54 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%arg9, %53 : tensor<?x?xf32>, tensor<?x?xf32>) outs(%26 : tensor<?x?xf32>) {
                ^bb0(%in: f32, %in_4: f32, %out: f32):
                  %66 = arith.truncf %cst_0 : f64 to f32
                  %67 = arith.mulf %in_4, %66 : f32
                  %68 = arith.cmpf ogt, %in, %67 : f32
                  %69 = arith.select %68, %in, %67 : f32
                  linalg.yield %69 : f32
                } -> tensor<?x?xf32>
                %55 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%51, %54 : tensor<?x?x?xf32>, tensor<?x?xf32>) outs(%19 : tensor<?x?x?xf32>) {
                ^bb0(%in: f32, %in_4: f32, %out: f32):
                  %66 = arith.truncf %cst_0 : f64 to f32
                  %67 = arith.mulf %in, %66 : f32
                  %68 = arith.subf %67, %in_4 : f32
                  %69 = math.powf %cst, %68 : f32
                  linalg.yield %69 : f32
                } -> tensor<?x?x?xf32>
                %56 = linalg.fill ins(%cst_1 : f32) outs(%28 : tensor<?x?xf32>) -> tensor<?x?xf32>
                %57 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%55 : tensor<?x?x?xf32>) outs(%56 : tensor<?x?xf32>) {
                ^bb0(%in: f32, %out: f32):
                  %66 = arith.addf %in, %out : f32
                  linalg.yield %66 : f32
                } -> tensor<?x?xf32>
                %58 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%arg9, %54 : tensor<?x?xf32>, tensor<?x?xf32>) outs(%30 : tensor<?x?xf32>) {
                ^bb0(%in: f32, %in_4: f32, %out: f32):
                  %66 = arith.subf %in, %in_4 : f32
                  %67 = math.powf %cst, %66 : f32
                  linalg.yield %67 : f32
                } -> tensor<?x?xf32>
                %59 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%arg10, %58, %57 : tensor<?x?xf32>, tensor<?x?xf32>, tensor<?x?xf32>) outs(%arg10 : tensor<?x?xf32>) {
                ^bb0(%in: f32, %in_4: f32, %in_5: f32, %out: f32):
                  %66 = arith.mulf %in, %in_4 : f32
                  %67 = arith.addf %66, %in_5 : f32
                  linalg.yield %67 : f32
                } -> tensor<?x?xf32>
                %60 = loom.subview %arg1[%37, %47, 0] [%12, %14, 128] [1, 1, 1], reuse : [seq = false, spat = false, temp = false] : memref<32x4096x128xf32> to memref<?x?x128xf32, strided<[524288, 128, 1], offset: ?>>
                %61 = loom.copy_to_tensor %60, %17 on @L1, interconnect : [], broadcast : [1, 1] : memref<?x?x128xf32, strided<[524288, 128, 1], offset: ?>>, memref<?x?x128xf32> -> tensor<?x?x128xf32>
                %62 = linalg.fill ins(%cst_1 : f32) outs(%36 : tensor<?x?x128xf32>) -> tensor<?x?x128xf32>
                %63 = linalg.batch_matmul ins(%55, %61 : tensor<?x?x?xf32>, tensor<?x?x128xf32>) outs(%62 : tensor<?x?x128xf32>) -> tensor<?x?x128xf32>
                %64 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%63, %arg11, %58 : tensor<?x?x128xf32>, tensor<?x?x128xf32>, tensor<?x?xf32>) outs(%arg11 : tensor<?x?x128xf32>) {
                ^bb0(%in: f32, %in_4: f32, %in_5: f32, %out: f32):
                  %66 = arith.mulf %in_4, %in_5 : f32
                  %67 = arith.addf %in, %66 : f32
                  linalg.yield %67 : f32
                } -> tensor<?x?x128xf32>
                %65 = linalg.copy ins(%54 : tensor<?x?xf32>) outs(%arg9 : tensor<?x?xf32>) -> tensor<?x?xf32>
                affine.yield %65, %59, %64 : tensor<?x?xf32>, tensor<?x?xf32>, tensor<?x?x128xf32>
              } {loom.iter_type = #loom.iter_type<sequential>}
              %45 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%44#2, %44#1 : tensor<?x?x128xf32>, tensor<?x?xf32>) outs(%32 : tensor<?x?x128xf32>) {
              ^bb0(%in: f32, %in_4: f32, %out: f32):
                %47 = arith.divf %in, %in_4 : f32
                linalg.yield %47 : f32
              } -> tensor<?x?x128xf32>
              %46 = loom.subview %arg3[%37, %38, 0] [%12, %13, 128] [1, 1, 1], reuse : [seq = false, spat = false, temp = false] : memref<32x4096x128xf32> to memref<?x?x128xf32, strided<[524288, 128, 1], offset: ?>>
              loom.copy_from_tensor %45, %46 on @DRAM : tensor<?x?x128xf32>, memref<?x?x128xf32, strided<[524288, 128, 1], offset: ?>>
            } {loom.iter_type = #loom.iter_type<temporal>}
          } {loom.iter_type = #loom.iter_type<temporal>}
        } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @x}
      } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @y}
      return
    }
  }
  module attributes {loom.pass_name = "EnumerateHWMapping"} {
    loom.constraint_space @constraints {
      %12 = loom.symbolic_var "BB" : index
      %13 = loom.symbolic_var "BM" : index
      %14 = loom.symbolic_var "BN" : index
      loom.range %12[32, 32]
      loom.range %13[32, 4096]
      loom.range %14[32, 4096]
      loom.align %12 by 32
      loom.align %13 by 32
      loom.align %14 by 32
      loom.polynomial_constraint(%12, %13, %14) {monomials = [{coeff = -1 : i64, vars = [0, 1, 2]}], upper_bound = -262144 : i64}
      loom.polynomial_constraint(%12, %14, %13) {monomials = [{coeff = 4 : i64, vars = [0, 1]}, {coeff = 4 : i64, vars = [0, 2, 1]}, {coeff = 4 : i64, vars = [0, 1]}, {coeff = 4 : i64, vars = [0, 2]}, {coeff = 4 : i64, vars = [0, 2]}, {coeff = 4 : i64, vars = [0, 2]}, {coeff = 4 : i64, vars = [0, 2]}, {coeff = 4 : i64, vars = [0, 2]}, {coeff = 4 : i64, vars = [0, 2]}, {coeff = 4 : i64, vars = [0, 2]}, {coeff = 4 : i64, vars = [0, 2]}], upper_bound = 1499136 : i64}
      loom.polynomial_constraint(%12) {monomials = [{coeff = 8 : i64, vars = [0]}], upper_bound = 32 : i64}
      loom.polynomial_constraint(%13) {monomials = [{coeff = 8 : i64, vars = [0]}], upper_bound = 4096 : i64}
    }
    func.func @attention__d1i0_d0i1__f10(%arg0: memref<32x128x4096xf32>, %arg1: memref<32x4096x128xf32>, %arg2: memref<32x4096x128xf32>, %arg3: memref<32x4096x128xf32>) {
      %cst = arith.constant 2.000000e+00 : f32
      %cst_0 = arith.constant 0.12751743074602467 : f64
      %cst_1 = arith.constant 0.000000e+00 : f32
      %cst_2 = arith.constant 1.000000e+00 : f32
      %cst_3 = arith.constant 0xFF800000 : f32
      %12 = loom.get_symbolic_block_size @constraints::@BB : index
      %13 = loom.get_symbolic_block_size @constraints::@BM : index
      %14 = loom.get_symbolic_block_size @constraints::@BN : index
      affine.parallel (%arg4) = (0) to (8) {
        affine.parallel (%arg5) = (0) to (8) {
          affine.for %arg6 = 0 to affine_map<()[s0, s1] -> ((4096 ceildiv s1) ceildiv 8)>()[%12, %13] {
            affine.for %arg7 = 0 to affine_map<()[s0, s1] -> ((32 ceildiv s0) ceildiv 8)>()[%12, %13] {
              %15 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 8)>(%arg4, %arg7)
              %16 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 8)>(%arg5, %arg6)
              %17 = loom.alloc [%12, %14, 128] on @L1 : memref<?x?x128xf32>
              %18 = loom.alloc [%12, %13, %14] on @L1 : memref<?x?x?xf32>
              %19 = loom.init_tensor %18[%12, %13, %14] : memref<?x?x?xf32> -> tensor<?x?x?xf32>
              %20 = loom.alloc [%12, 128, %14] on @L1 : memref<?x128x?xf32>
              %21 = loom.alloc [%12, %13] on @L1 : memref<?x?xf32>
              %22 = loom.init_tensor %21[%12, %13] : memref<?x?xf32> -> tensor<?x?xf32>
              %23 = loom.alloc [%12, %13] on @L1 : memref<?x?xf32>
              %24 = loom.init_tensor %23[%12, %13] : memref<?x?xf32> -> tensor<?x?xf32>
              %25 = loom.alloc [%12, %13] on @L1 : memref<?x?xf32>
              %26 = loom.init_tensor %25[%12, %13] : memref<?x?xf32> -> tensor<?x?xf32>
              %27 = loom.alloc [%12, %13] on @L1 : memref<?x?xf32>
              %28 = loom.init_tensor %27[%12, %13] : memref<?x?xf32> -> tensor<?x?xf32>
              %29 = loom.alloc [%12, %13] on @L1 : memref<?x?xf32>
              %30 = loom.init_tensor %29[%12, %13] : memref<?x?xf32> -> tensor<?x?xf32>
              %31 = loom.alloc [%12, %13, 128] on @L1 : memref<?x?x128xf32>
              %32 = loom.init_tensor %31[%12, %13, 128] : memref<?x?x128xf32> -> tensor<?x?x128xf32>
              %33 = loom.alloc [%12, %13, 128] on @L1 : memref<?x?x128xf32>
              %34 = loom.init_tensor %33[%12, %13, 128] : memref<?x?x128xf32> -> tensor<?x?x128xf32>
              %35 = loom.alloc [%12, %13, 128] on @L1 : memref<?x?x128xf32>
              %36 = loom.init_tensor %35[%12, %13, 128] : memref<?x?x128xf32> -> tensor<?x?x128xf32>
              %37 = arith.muli %15, %12 : index
              %38 = arith.muli %16, %13 : index
              %39 = loom.subview %arg2[%37, %38, 0] [%12, %13, 128] [1, 1, 1], reuse : [seq = false, spat = false, temp = false] : memref<32x4096x128xf32> to memref<?x?x128xf32, strided<[524288, 128, 1], offset: ?>>
              %40 = loom.copy_to_tensor %39, %31 on @L1, interconnect : [], broadcast : [1, 1] : memref<?x?x128xf32, strided<[524288, 128, 1], offset: ?>>, memref<?x?x128xf32> -> tensor<?x?x128xf32>
              %41 = linalg.fill ins(%cst_1 : f32) outs(%34 : tensor<?x?x128xf32>) -> tensor<?x?x128xf32>
              %42 = linalg.fill ins(%cst_2 : f32) outs(%22 : tensor<?x?xf32>) -> tensor<?x?xf32>
              %43 = linalg.fill ins(%cst_3 : f32) outs(%24 : tensor<?x?xf32>) -> tensor<?x?xf32>
              %44:3 = affine.for %arg8 = 0 to affine_map<()[s0] -> (4096 ceildiv s0)>()[%14] iter_args(%arg9 = %43, %arg10 = %42, %arg11 = %41) -> (tensor<?x?xf32>, tensor<?x?xf32>, tensor<?x?x128xf32>) {
                %47 = arith.muli %arg8, %14 : index
                %48 = loom.subview %arg0[%37, 0, %47] [%12, 128, %14] [1, 1, 1], reuse : [seq = false, spat = false, temp = false] : memref<32x128x4096xf32> to memref<?x128x?xf32, strided<[524288, 4096, 1], offset: ?>>
                %49 = loom.copy_to_tensor %48, %20 on @L1, interconnect : [], broadcast : [1, 1] : memref<?x128x?xf32, strided<[524288, 4096, 1], offset: ?>>, memref<?x128x?xf32> -> tensor<?x128x?xf32>
                %50 = linalg.fill ins(%cst_1 : f32) outs(%19 : tensor<?x?x?xf32>) -> tensor<?x?x?xf32>
                %51 = linalg.batch_matmul ins(%40, %49 : tensor<?x?x128xf32>, tensor<?x128x?xf32>) outs(%50 : tensor<?x?x?xf32>) -> tensor<?x?x?xf32>
                %52 = linalg.fill ins(%cst_3 : f32) outs(%26 : tensor<?x?xf32>) -> tensor<?x?xf32>
                %53 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%51 : tensor<?x?x?xf32>) outs(%52 : tensor<?x?xf32>) {
                ^bb0(%in: f32, %out: f32):
                  %66 = arith.maximumf %in, %out : f32
                  linalg.yield %66 : f32
                } -> tensor<?x?xf32>
                %54 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%arg9, %53 : tensor<?x?xf32>, tensor<?x?xf32>) outs(%26 : tensor<?x?xf32>) {
                ^bb0(%in: f32, %in_4: f32, %out: f32):
                  %66 = arith.truncf %cst_0 : f64 to f32
                  %67 = arith.mulf %in_4, %66 : f32
                  %68 = arith.cmpf ogt, %in, %67 : f32
                  %69 = arith.select %68, %in, %67 : f32
                  linalg.yield %69 : f32
                } -> tensor<?x?xf32>
                %55 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%51, %54 : tensor<?x?x?xf32>, tensor<?x?xf32>) outs(%19 : tensor<?x?x?xf32>) {
                ^bb0(%in: f32, %in_4: f32, %out: f32):
                  %66 = arith.truncf %cst_0 : f64 to f32
                  %67 = arith.mulf %in, %66 : f32
                  %68 = arith.subf %67, %in_4 : f32
                  %69 = math.powf %cst, %68 : f32
                  linalg.yield %69 : f32
                } -> tensor<?x?x?xf32>
                %56 = linalg.fill ins(%cst_1 : f32) outs(%28 : tensor<?x?xf32>) -> tensor<?x?xf32>
                %57 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%55 : tensor<?x?x?xf32>) outs(%56 : tensor<?x?xf32>) {
                ^bb0(%in: f32, %out: f32):
                  %66 = arith.addf %in, %out : f32
                  linalg.yield %66 : f32
                } -> tensor<?x?xf32>
                %58 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%arg9, %54 : tensor<?x?xf32>, tensor<?x?xf32>) outs(%30 : tensor<?x?xf32>) {
                ^bb0(%in: f32, %in_4: f32, %out: f32):
                  %66 = arith.subf %in, %in_4 : f32
                  %67 = math.powf %cst, %66 : f32
                  linalg.yield %67 : f32
                } -> tensor<?x?xf32>
                %59 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%arg10, %58, %57 : tensor<?x?xf32>, tensor<?x?xf32>, tensor<?x?xf32>) outs(%arg10 : tensor<?x?xf32>) {
                ^bb0(%in: f32, %in_4: f32, %in_5: f32, %out: f32):
                  %66 = arith.mulf %in, %in_4 : f32
                  %67 = arith.addf %66, %in_5 : f32
                  linalg.yield %67 : f32
                } -> tensor<?x?xf32>
                %60 = loom.subview %arg1[%37, %47, 0] [%12, %14, 128] [1, 1, 1], reuse : [seq = false, spat = false, temp = false] : memref<32x4096x128xf32> to memref<?x?x128xf32, strided<[524288, 128, 1], offset: ?>>
                %61 = loom.copy_to_tensor %60, %17 on @L1, interconnect : [], broadcast : [1, 1] : memref<?x?x128xf32, strided<[524288, 128, 1], offset: ?>>, memref<?x?x128xf32> -> tensor<?x?x128xf32>
                %62 = linalg.fill ins(%cst_1 : f32) outs(%36 : tensor<?x?x128xf32>) -> tensor<?x?x128xf32>
                %63 = linalg.batch_matmul ins(%55, %61 : tensor<?x?x?xf32>, tensor<?x?x128xf32>) outs(%62 : tensor<?x?x128xf32>) -> tensor<?x?x128xf32>
                %64 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%63, %arg11, %58 : tensor<?x?x128xf32>, tensor<?x?x128xf32>, tensor<?x?xf32>) outs(%arg11 : tensor<?x?x128xf32>) {
                ^bb0(%in: f32, %in_4: f32, %in_5: f32, %out: f32):
                  %66 = arith.mulf %in_4, %in_5 : f32
                  %67 = arith.addf %in, %66 : f32
                  linalg.yield %67 : f32
                } -> tensor<?x?x128xf32>
                %65 = linalg.copy ins(%54 : tensor<?x?xf32>) outs(%arg9 : tensor<?x?xf32>) -> tensor<?x?xf32>
                affine.yield %65, %59, %64 : tensor<?x?xf32>, tensor<?x?xf32>, tensor<?x?x128xf32>
              } {loom.iter_type = #loom.iter_type<sequential>}
              %45 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%44#2, %44#1 : tensor<?x?x128xf32>, tensor<?x?xf32>) outs(%32 : tensor<?x?x128xf32>) {
              ^bb0(%in: f32, %in_4: f32, %out: f32):
                %47 = arith.divf %in, %in_4 : f32
                linalg.yield %47 : f32
              } -> tensor<?x?x128xf32>
              %46 = loom.subview %arg3[%37, %38, 0] [%12, %13, 128] [1, 1, 1], reuse : [seq = false, spat = false, temp = false] : memref<32x4096x128xf32> to memref<?x?x128xf32, strided<[524288, 128, 1], offset: ?>>
              loom.copy_from_tensor %45, %46 on @DRAM : tensor<?x?x128xf32>, memref<?x?x128xf32, strided<[524288, 128, 1], offset: ?>>
            } {loom.iter_type = #loom.iter_type<temporal>}
          } {loom.iter_type = #loom.iter_type<temporal>}
        } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @x}
      } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @y}
      return
    }
  }
  module attributes {loom.pass_name = "EnumerateHWMapping"} {
    loom.constraint_space @constraints {
      %12 = loom.symbolic_var "BB" : index
      %13 = loom.symbolic_var "BM" : index
      %14 = loom.symbolic_var "BN" : index
      loom.range %12[32, 32]
      loom.range %13[32, 4096]
      loom.range %14[32, 4096]
      loom.align %12 by 32
      loom.align %13 by 32
      loom.align %14 by 32
      loom.polynomial_constraint(%12, %13, %14) {monomials = [{coeff = -1 : i64, vars = [0, 1, 2]}], upper_bound = -262144 : i64}
      loom.polynomial_constraint(%12, %14, %13) {monomials = [{coeff = 4 : i64, vars = [0, 1]}, {coeff = 4 : i64, vars = [0, 2, 1]}, {coeff = 4 : i64, vars = [0, 1]}, {coeff = 4 : i64, vars = [0, 2]}, {coeff = 4 : i64, vars = [0, 2]}, {coeff = 4 : i64, vars = [0, 2]}, {coeff = 4 : i64, vars = [0, 2]}, {coeff = 4 : i64, vars = [0, 2]}, {coeff = 4 : i64, vars = [0, 2]}, {coeff = 4 : i64, vars = [0, 2]}, {coeff = 4 : i64, vars = [0, 2]}], upper_bound = 1499136 : i64}
      loom.polynomial_constraint(%13) {monomials = [{coeff = 64 : i64, vars = [0]}], upper_bound = 4096 : i64}
    }
    func.func @attention__d0i1_d1i1__f01(%arg0: memref<32x128x4096xf32>, %arg1: memref<32x4096x128xf32>, %arg2: memref<32x4096x128xf32>, %arg3: memref<32x4096x128xf32>) {
      %cst = arith.constant 2.000000e+00 : f32
      %cst_0 = arith.constant 0.12751743074602467 : f64
      %cst_1 = arith.constant 0.000000e+00 : f32
      %cst_2 = arith.constant 1.000000e+00 : f32
      %cst_3 = arith.constant 0xFF800000 : f32
      %12 = loom.get_symbolic_block_size @constraints::@BB : index
      %13 = loom.get_symbolic_block_size @constraints::@BM : index
      %14 = loom.get_symbolic_block_size @constraints::@BN : index
      affine.parallel (%arg4) = (0) to (8) {
        affine.parallel (%arg5) = (0) to (8) {
          affine.for %arg6 = 0 to affine_map<()[s0, s1] -> (32 ceildiv s0)>()[%12, %13] {
            affine.for %arg7 = 0 to affine_map<()[s0, s1] -> (((4096 ceildiv s1) ceildiv 8) ceildiv 8)>()[%12, %13] {
              %15 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 8 + d1 + d2 * 64)>(%arg4, %arg5, %arg7)
              %16 = loom.alloc [%12, %14, 128] on @L1 : memref<?x?x128xf32>
              %17 = loom.alloc [%12, %13, %14] on @L1 : memref<?x?x?xf32>
              %18 = loom.init_tensor %17[%12, %13, %14] : memref<?x?x?xf32> -> tensor<?x?x?xf32>
              %19 = loom.alloc [%12, 128, %14] on @L1 : memref<?x128x?xf32>
              %20 = loom.alloc [%12, %13] on @L1 : memref<?x?xf32>
              %21 = loom.init_tensor %20[%12, %13] : memref<?x?xf32> -> tensor<?x?xf32>
              %22 = loom.alloc [%12, %13] on @L1 : memref<?x?xf32>
              %23 = loom.init_tensor %22[%12, %13] : memref<?x?xf32> -> tensor<?x?xf32>
              %24 = loom.alloc [%12, %13] on @L1 : memref<?x?xf32>
              %25 = loom.init_tensor %24[%12, %13] : memref<?x?xf32> -> tensor<?x?xf32>
              %26 = loom.alloc [%12, %13] on @L1 : memref<?x?xf32>
              %27 = loom.init_tensor %26[%12, %13] : memref<?x?xf32> -> tensor<?x?xf32>
              %28 = loom.alloc [%12, %13] on @L1 : memref<?x?xf32>
              %29 = loom.init_tensor %28[%12, %13] : memref<?x?xf32> -> tensor<?x?xf32>
              %30 = loom.alloc [%12, %13, 128] on @L1 : memref<?x?x128xf32>
              %31 = loom.init_tensor %30[%12, %13, 128] : memref<?x?x128xf32> -> tensor<?x?x128xf32>
              %32 = loom.alloc [%12, %13, 128] on @L1 : memref<?x?x128xf32>
              %33 = loom.init_tensor %32[%12, %13, 128] : memref<?x?x128xf32> -> tensor<?x?x128xf32>
              %34 = loom.alloc [%12, %13, 128] on @L1 : memref<?x?x128xf32>
              %35 = loom.init_tensor %34[%12, %13, 128] : memref<?x?x128xf32> -> tensor<?x?x128xf32>
              %36 = arith.muli %arg6, %12 : index
              %37 = arith.muli %15, %13 : index
              %38 = loom.subview %arg2[%36, %37, 0] [%12, %13, 128] [1, 1, 1], reuse : [seq = false, spat = false, temp = false] : memref<32x4096x128xf32> to memref<?x?x128xf32, strided<[524288, 128, 1], offset: ?>>
              %39 = loom.copy_to_tensor %38, %30 on @L1, interconnect : [], broadcast : [1, 1] : memref<?x?x128xf32, strided<[524288, 128, 1], offset: ?>>, memref<?x?x128xf32> -> tensor<?x?x128xf32>
              %40 = linalg.fill ins(%cst_1 : f32) outs(%33 : tensor<?x?x128xf32>) -> tensor<?x?x128xf32>
              %41 = linalg.fill ins(%cst_2 : f32) outs(%21 : tensor<?x?xf32>) -> tensor<?x?xf32>
              %42 = linalg.fill ins(%cst_3 : f32) outs(%23 : tensor<?x?xf32>) -> tensor<?x?xf32>
              %43:3 = affine.for %arg8 = 0 to affine_map<()[s0] -> (4096 ceildiv s0)>()[%14] iter_args(%arg9 = %42, %arg10 = %41, %arg11 = %40) -> (tensor<?x?xf32>, tensor<?x?xf32>, tensor<?x?x128xf32>) {
                %46 = arith.muli %arg8, %14 : index
                %47 = loom.subview %arg0[%36, 0, %46] [%12, 128, %14] [1, 1, 1], reuse : [seq = false, spat = false, temp = false] : memref<32x128x4096xf32> to memref<?x128x?xf32, strided<[524288, 4096, 1], offset: ?>>
                %48 = loom.copy_to_tensor %47, %19 on @L1, interconnect : [], broadcast : [1, 1] : memref<?x128x?xf32, strided<[524288, 4096, 1], offset: ?>>, memref<?x128x?xf32> -> tensor<?x128x?xf32>
                %49 = linalg.fill ins(%cst_1 : f32) outs(%18 : tensor<?x?x?xf32>) -> tensor<?x?x?xf32>
                %50 = linalg.batch_matmul ins(%39, %48 : tensor<?x?x128xf32>, tensor<?x128x?xf32>) outs(%49 : tensor<?x?x?xf32>) -> tensor<?x?x?xf32>
                %51 = linalg.fill ins(%cst_3 : f32) outs(%25 : tensor<?x?xf32>) -> tensor<?x?xf32>
                %52 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%50 : tensor<?x?x?xf32>) outs(%51 : tensor<?x?xf32>) {
                ^bb0(%in: f32, %out: f32):
                  %65 = arith.maximumf %in, %out : f32
                  linalg.yield %65 : f32
                } -> tensor<?x?xf32>
                %53 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%arg9, %52 : tensor<?x?xf32>, tensor<?x?xf32>) outs(%25 : tensor<?x?xf32>) {
                ^bb0(%in: f32, %in_4: f32, %out: f32):
                  %65 = arith.truncf %cst_0 : f64 to f32
                  %66 = arith.mulf %in_4, %65 : f32
                  %67 = arith.cmpf ogt, %in, %66 : f32
                  %68 = arith.select %67, %in, %66 : f32
                  linalg.yield %68 : f32
                } -> tensor<?x?xf32>
                %54 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%50, %53 : tensor<?x?x?xf32>, tensor<?x?xf32>) outs(%18 : tensor<?x?x?xf32>) {
                ^bb0(%in: f32, %in_4: f32, %out: f32):
                  %65 = arith.truncf %cst_0 : f64 to f32
                  %66 = arith.mulf %in, %65 : f32
                  %67 = arith.subf %66, %in_4 : f32
                  %68 = math.powf %cst, %67 : f32
                  linalg.yield %68 : f32
                } -> tensor<?x?x?xf32>
                %55 = linalg.fill ins(%cst_1 : f32) outs(%27 : tensor<?x?xf32>) -> tensor<?x?xf32>
                %56 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%54 : tensor<?x?x?xf32>) outs(%55 : tensor<?x?xf32>) {
                ^bb0(%in: f32, %out: f32):
                  %65 = arith.addf %in, %out : f32
                  linalg.yield %65 : f32
                } -> tensor<?x?xf32>
                %57 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%arg9, %53 : tensor<?x?xf32>, tensor<?x?xf32>) outs(%29 : tensor<?x?xf32>) {
                ^bb0(%in: f32, %in_4: f32, %out: f32):
                  %65 = arith.subf %in, %in_4 : f32
                  %66 = math.powf %cst, %65 : f32
                  linalg.yield %66 : f32
                } -> tensor<?x?xf32>
                %58 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%arg10, %57, %56 : tensor<?x?xf32>, tensor<?x?xf32>, tensor<?x?xf32>) outs(%arg10 : tensor<?x?xf32>) {
                ^bb0(%in: f32, %in_4: f32, %in_5: f32, %out: f32):
                  %65 = arith.mulf %in, %in_4 : f32
                  %66 = arith.addf %65, %in_5 : f32
                  linalg.yield %66 : f32
                } -> tensor<?x?xf32>
                %59 = loom.subview %arg1[%36, %46, 0] [%12, %14, 128] [1, 1, 1], reuse : [seq = false, spat = false, temp = false] : memref<32x4096x128xf32> to memref<?x?x128xf32, strided<[524288, 128, 1], offset: ?>>
                %60 = loom.copy_to_tensor %59, %16 on @L1, interconnect : [], broadcast : [1, 1] : memref<?x?x128xf32, strided<[524288, 128, 1], offset: ?>>, memref<?x?x128xf32> -> tensor<?x?x128xf32>
                %61 = linalg.fill ins(%cst_1 : f32) outs(%35 : tensor<?x?x128xf32>) -> tensor<?x?x128xf32>
                %62 = linalg.batch_matmul ins(%54, %60 : tensor<?x?x?xf32>, tensor<?x?x128xf32>) outs(%61 : tensor<?x?x128xf32>) -> tensor<?x?x128xf32>
                %63 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%62, %arg11, %57 : tensor<?x?x128xf32>, tensor<?x?x128xf32>, tensor<?x?xf32>) outs(%arg11 : tensor<?x?x128xf32>) {
                ^bb0(%in: f32, %in_4: f32, %in_5: f32, %out: f32):
                  %65 = arith.mulf %in_4, %in_5 : f32
                  %66 = arith.addf %in, %65 : f32
                  linalg.yield %66 : f32
                } -> tensor<?x?x128xf32>
                %64 = linalg.copy ins(%53 : tensor<?x?xf32>) outs(%arg9 : tensor<?x?xf32>) -> tensor<?x?xf32>
                affine.yield %64, %58, %63 : tensor<?x?xf32>, tensor<?x?xf32>, tensor<?x?x128xf32>
              } {loom.iter_type = #loom.iter_type<sequential>}
              %44 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%43#2, %43#1 : tensor<?x?x128xf32>, tensor<?x?xf32>) outs(%31 : tensor<?x?x128xf32>) {
              ^bb0(%in: f32, %in_4: f32, %out: f32):
                %46 = arith.divf %in, %in_4 : f32
                linalg.yield %46 : f32
              } -> tensor<?x?x128xf32>
              %45 = loom.subview %arg3[%36, %37, 0] [%12, %13, 128] [1, 1, 1], reuse : [seq = false, spat = false, temp = false] : memref<32x4096x128xf32> to memref<?x?x128xf32, strided<[524288, 128, 1], offset: ?>>
              loom.copy_from_tensor %44, %45 on @DRAM : tensor<?x?x128xf32>, memref<?x?x128xf32, strided<[524288, 128, 1], offset: ?>>
            } {loom.iter_type = #loom.iter_type<temporal>}
          } {loom.iter_type = #loom.iter_type<temporal>}
        } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @y}
      } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @x}
      return
    }
  }
  module attributes {loom.pass_name = "EnumerateHWMapping"} {
    loom.constraint_space @constraints {
      %12 = loom.symbolic_var "BB" : index
      %13 = loom.symbolic_var "BM" : index
      %14 = loom.symbolic_var "BN" : index
      loom.range %12[32, 32]
      loom.range %13[32, 4096]
      loom.range %14[32, 4096]
      loom.align %12 by 32
      loom.align %13 by 32
      loom.align %14 by 32
      loom.polynomial_constraint(%12, %13, %14) {monomials = [{coeff = -1 : i64, vars = [0, 1, 2]}], upper_bound = -262144 : i64}
      loom.polynomial_constraint(%12, %14, %13) {monomials = [{coeff = 4 : i64, vars = [0, 1]}, {coeff = 4 : i64, vars = [0, 2, 1]}, {coeff = 4 : i64, vars = [0, 1]}, {coeff = 4 : i64, vars = [0, 2]}, {coeff = 4 : i64, vars = [0, 2]}, {coeff = 4 : i64, vars = [0, 2]}, {coeff = 4 : i64, vars = [0, 2]}, {coeff = 4 : i64, vars = [0, 2]}, {coeff = 4 : i64, vars = [0, 2]}, {coeff = 4 : i64, vars = [0, 2]}, {coeff = 4 : i64, vars = [0, 2]}], upper_bound = 1499136 : i64}
      loom.polynomial_constraint(%13) {monomials = [{coeff = 64 : i64, vars = [0]}], upper_bound = 4096 : i64}
    }
    func.func @attention__d0i1_d1i1__f10(%arg0: memref<32x128x4096xf32>, %arg1: memref<32x4096x128xf32>, %arg2: memref<32x4096x128xf32>, %arg3: memref<32x4096x128xf32>) {
      %cst = arith.constant 2.000000e+00 : f32
      %cst_0 = arith.constant 0.12751743074602467 : f64
      %cst_1 = arith.constant 0.000000e+00 : f32
      %cst_2 = arith.constant 1.000000e+00 : f32
      %cst_3 = arith.constant 0xFF800000 : f32
      %12 = loom.get_symbolic_block_size @constraints::@BB : index
      %13 = loom.get_symbolic_block_size @constraints::@BM : index
      %14 = loom.get_symbolic_block_size @constraints::@BN : index
      affine.parallel (%arg4) = (0) to (8) {
        affine.parallel (%arg5) = (0) to (8) {
          affine.for %arg6 = 0 to affine_map<()[s0, s1] -> (((4096 ceildiv s1) ceildiv 8) ceildiv 8)>()[%12, %13] {
            affine.for %arg7 = 0 to affine_map<()[s0, s1] -> (32 ceildiv s0)>()[%12, %13] {
              %15 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 8 + d1 + d2 * 64)>(%arg4, %arg5, %arg6)
              %16 = loom.alloc [%12, %14, 128] on @L1 : memref<?x?x128xf32>
              %17 = loom.alloc [%12, %13, %14] on @L1 : memref<?x?x?xf32>
              %18 = loom.init_tensor %17[%12, %13, %14] : memref<?x?x?xf32> -> tensor<?x?x?xf32>
              %19 = loom.alloc [%12, 128, %14] on @L1 : memref<?x128x?xf32>
              %20 = loom.alloc [%12, %13] on @L1 : memref<?x?xf32>
              %21 = loom.init_tensor %20[%12, %13] : memref<?x?xf32> -> tensor<?x?xf32>
              %22 = loom.alloc [%12, %13] on @L1 : memref<?x?xf32>
              %23 = loom.init_tensor %22[%12, %13] : memref<?x?xf32> -> tensor<?x?xf32>
              %24 = loom.alloc [%12, %13] on @L1 : memref<?x?xf32>
              %25 = loom.init_tensor %24[%12, %13] : memref<?x?xf32> -> tensor<?x?xf32>
              %26 = loom.alloc [%12, %13] on @L1 : memref<?x?xf32>
              %27 = loom.init_tensor %26[%12, %13] : memref<?x?xf32> -> tensor<?x?xf32>
              %28 = loom.alloc [%12, %13] on @L1 : memref<?x?xf32>
              %29 = loom.init_tensor %28[%12, %13] : memref<?x?xf32> -> tensor<?x?xf32>
              %30 = loom.alloc [%12, %13, 128] on @L1 : memref<?x?x128xf32>
              %31 = loom.init_tensor %30[%12, %13, 128] : memref<?x?x128xf32> -> tensor<?x?x128xf32>
              %32 = loom.alloc [%12, %13, 128] on @L1 : memref<?x?x128xf32>
              %33 = loom.init_tensor %32[%12, %13, 128] : memref<?x?x128xf32> -> tensor<?x?x128xf32>
              %34 = loom.alloc [%12, %13, 128] on @L1 : memref<?x?x128xf32>
              %35 = loom.init_tensor %34[%12, %13, 128] : memref<?x?x128xf32> -> tensor<?x?x128xf32>
              %36 = arith.muli %arg7, %12 : index
              %37 = arith.muli %15, %13 : index
              %38 = loom.subview %arg2[%36, %37, 0] [%12, %13, 128] [1, 1, 1], reuse : [seq = false, spat = false, temp = false] : memref<32x4096x128xf32> to memref<?x?x128xf32, strided<[524288, 128, 1], offset: ?>>
              %39 = loom.copy_to_tensor %38, %30 on @L1, interconnect : [], broadcast : [1, 1] : memref<?x?x128xf32, strided<[524288, 128, 1], offset: ?>>, memref<?x?x128xf32> -> tensor<?x?x128xf32>
              %40 = linalg.fill ins(%cst_1 : f32) outs(%33 : tensor<?x?x128xf32>) -> tensor<?x?x128xf32>
              %41 = linalg.fill ins(%cst_2 : f32) outs(%21 : tensor<?x?xf32>) -> tensor<?x?xf32>
              %42 = linalg.fill ins(%cst_3 : f32) outs(%23 : tensor<?x?xf32>) -> tensor<?x?xf32>
              %43:3 = affine.for %arg8 = 0 to affine_map<()[s0] -> (4096 ceildiv s0)>()[%14] iter_args(%arg9 = %42, %arg10 = %41, %arg11 = %40) -> (tensor<?x?xf32>, tensor<?x?xf32>, tensor<?x?x128xf32>) {
                %46 = arith.muli %arg8, %14 : index
                %47 = loom.subview %arg0[%36, 0, %46] [%12, 128, %14] [1, 1, 1], reuse : [seq = false, spat = false, temp = false] : memref<32x128x4096xf32> to memref<?x128x?xf32, strided<[524288, 4096, 1], offset: ?>>
                %48 = loom.copy_to_tensor %47, %19 on @L1, interconnect : [], broadcast : [1, 1] : memref<?x128x?xf32, strided<[524288, 4096, 1], offset: ?>>, memref<?x128x?xf32> -> tensor<?x128x?xf32>
                %49 = linalg.fill ins(%cst_1 : f32) outs(%18 : tensor<?x?x?xf32>) -> tensor<?x?x?xf32>
                %50 = linalg.batch_matmul ins(%39, %48 : tensor<?x?x128xf32>, tensor<?x128x?xf32>) outs(%49 : tensor<?x?x?xf32>) -> tensor<?x?x?xf32>
                %51 = linalg.fill ins(%cst_3 : f32) outs(%25 : tensor<?x?xf32>) -> tensor<?x?xf32>
                %52 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%50 : tensor<?x?x?xf32>) outs(%51 : tensor<?x?xf32>) {
                ^bb0(%in: f32, %out: f32):
                  %65 = arith.maximumf %in, %out : f32
                  linalg.yield %65 : f32
                } -> tensor<?x?xf32>
                %53 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%arg9, %52 : tensor<?x?xf32>, tensor<?x?xf32>) outs(%25 : tensor<?x?xf32>) {
                ^bb0(%in: f32, %in_4: f32, %out: f32):
                  %65 = arith.truncf %cst_0 : f64 to f32
                  %66 = arith.mulf %in_4, %65 : f32
                  %67 = arith.cmpf ogt, %in, %66 : f32
                  %68 = arith.select %67, %in, %66 : f32
                  linalg.yield %68 : f32
                } -> tensor<?x?xf32>
                %54 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%50, %53 : tensor<?x?x?xf32>, tensor<?x?xf32>) outs(%18 : tensor<?x?x?xf32>) {
                ^bb0(%in: f32, %in_4: f32, %out: f32):
                  %65 = arith.truncf %cst_0 : f64 to f32
                  %66 = arith.mulf %in, %65 : f32
                  %67 = arith.subf %66, %in_4 : f32
                  %68 = math.powf %cst, %67 : f32
                  linalg.yield %68 : f32
                } -> tensor<?x?x?xf32>
                %55 = linalg.fill ins(%cst_1 : f32) outs(%27 : tensor<?x?xf32>) -> tensor<?x?xf32>
                %56 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%54 : tensor<?x?x?xf32>) outs(%55 : tensor<?x?xf32>) {
                ^bb0(%in: f32, %out: f32):
                  %65 = arith.addf %in, %out : f32
                  linalg.yield %65 : f32
                } -> tensor<?x?xf32>
                %57 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%arg9, %53 : tensor<?x?xf32>, tensor<?x?xf32>) outs(%29 : tensor<?x?xf32>) {
                ^bb0(%in: f32, %in_4: f32, %out: f32):
                  %65 = arith.subf %in, %in_4 : f32
                  %66 = math.powf %cst, %65 : f32
                  linalg.yield %66 : f32
                } -> tensor<?x?xf32>
                %58 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%arg10, %57, %56 : tensor<?x?xf32>, tensor<?x?xf32>, tensor<?x?xf32>) outs(%arg10 : tensor<?x?xf32>) {
                ^bb0(%in: f32, %in_4: f32, %in_5: f32, %out: f32):
                  %65 = arith.mulf %in, %in_4 : f32
                  %66 = arith.addf %65, %in_5 : f32
                  linalg.yield %66 : f32
                } -> tensor<?x?xf32>
                %59 = loom.subview %arg1[%36, %46, 0] [%12, %14, 128] [1, 1, 1], reuse : [seq = false, spat = false, temp = false] : memref<32x4096x128xf32> to memref<?x?x128xf32, strided<[524288, 128, 1], offset: ?>>
                %60 = loom.copy_to_tensor %59, %16 on @L1, interconnect : [], broadcast : [1, 1] : memref<?x?x128xf32, strided<[524288, 128, 1], offset: ?>>, memref<?x?x128xf32> -> tensor<?x?x128xf32>
                %61 = linalg.fill ins(%cst_1 : f32) outs(%35 : tensor<?x?x128xf32>) -> tensor<?x?x128xf32>
                %62 = linalg.batch_matmul ins(%54, %60 : tensor<?x?x?xf32>, tensor<?x?x128xf32>) outs(%61 : tensor<?x?x128xf32>) -> tensor<?x?x128xf32>
                %63 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%62, %arg11, %57 : tensor<?x?x128xf32>, tensor<?x?x128xf32>, tensor<?x?xf32>) outs(%arg11 : tensor<?x?x128xf32>) {
                ^bb0(%in: f32, %in_4: f32, %in_5: f32, %out: f32):
                  %65 = arith.mulf %in_4, %in_5 : f32
                  %66 = arith.addf %in, %65 : f32
                  linalg.yield %66 : f32
                } -> tensor<?x?x128xf32>
                %64 = linalg.copy ins(%53 : tensor<?x?xf32>) outs(%arg9 : tensor<?x?xf32>) -> tensor<?x?xf32>
                affine.yield %64, %58, %63 : tensor<?x?xf32>, tensor<?x?xf32>, tensor<?x?x128xf32>
              } {loom.iter_type = #loom.iter_type<sequential>}
              %44 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%43#2, %43#1 : tensor<?x?x128xf32>, tensor<?x?xf32>) outs(%31 : tensor<?x?x128xf32>) {
              ^bb0(%in: f32, %in_4: f32, %out: f32):
                %46 = arith.divf %in, %in_4 : f32
                linalg.yield %46 : f32
              } -> tensor<?x?x128xf32>
              %45 = loom.subview %arg3[%36, %37, 0] [%12, %13, 128] [1, 1, 1], reuse : [seq = false, spat = false, temp = false] : memref<32x4096x128xf32> to memref<?x?x128xf32, strided<[524288, 128, 1], offset: ?>>
              loom.copy_from_tensor %44, %45 on @DRAM : tensor<?x?x128xf32>, memref<?x?x128xf32, strided<[524288, 128, 1], offset: ?>>
            } {loom.iter_type = #loom.iter_type<temporal>}
          } {loom.iter_type = #loom.iter_type<temporal>}
        } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @y}
      } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @x}
      return
    }
  }
}
