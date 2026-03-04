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
              %18 = loom.semaphore %17 : memref<?x?x?xf32> -> memref<?x?x?xf32>
              %19 = loom.init_tensor %18[%12, %13, %14] : memref<?x?x?xf32> -> tensor<?x?x?xf32>
              %20 = loom.alloc [%12, 128, %14] on @L1 : memref<?x128x?xf32>
              %21 = loom.alloc [%12, %13] on @L1 : memref<?x?xf32>
              %22 = loom.semaphore %21 : memref<?x?xf32> -> memref<?x?xf32>
              %23 = loom.init_tensor %22[%12, %13] : memref<?x?xf32> -> tensor<?x?xf32>
              %24 = loom.alloc [%12, %13] on @L1 : memref<?x?xf32>
              %25 = loom.semaphore %24 : memref<?x?xf32> -> memref<?x?xf32>
              %26 = loom.init_tensor %25[%12, %13] : memref<?x?xf32> -> tensor<?x?xf32>
              %27 = loom.alloc [%12, %13] on @L1 : memref<?x?xf32>
              %28 = loom.semaphore %27 : memref<?x?xf32> -> memref<?x?xf32>
              %29 = loom.init_tensor %28[%12, %13] : memref<?x?xf32> -> tensor<?x?xf32>
              %30 = loom.alloc [%12, %13] on @L1 : memref<?x?xf32>
              %31 = loom.semaphore %30 : memref<?x?xf32> -> memref<?x?xf32>
              %32 = loom.init_tensor %31[%12, %13] : memref<?x?xf32> -> tensor<?x?xf32>
              %33 = loom.alloc [%12, %13] on @L1 : memref<?x?xf32>
              %34 = loom.semaphore %33 : memref<?x?xf32> -> memref<?x?xf32>
              %35 = loom.init_tensor %34[%12, %13] : memref<?x?xf32> -> tensor<?x?xf32>
              %36 = loom.alloc [%12, %13, 128] on @L1 : memref<?x?x128xf32>
              %37 = loom.semaphore %36 : memref<?x?x128xf32> -> memref<?x?x128xf32>
              %38 = loom.init_tensor %37[%12, %13, 128] : memref<?x?x128xf32> -> tensor<?x?x128xf32>
              %39 = loom.alloc [%12, %13, 128] on @L1 : memref<?x?x128xf32>
              %40 = loom.semaphore %39 : memref<?x?x128xf32> -> memref<?x?x128xf32>
              %41 = loom.init_tensor %40[%12, %13, 128] : memref<?x?x128xf32> -> tensor<?x?x128xf32>
              %42 = loom.alloc [%12, %13, 128] on @L1 : memref<?x?x128xf32>
              %43 = loom.semaphore %42 : memref<?x?x128xf32> -> memref<?x?x128xf32>
              %44 = loom.init_tensor %43[%12, %13, 128] : memref<?x?x128xf32> -> tensor<?x?x128xf32>
              %45 = arith.muli %15, %12 : index
              %46 = arith.muli %arg7, %13 : index
              %47 = loom.subview %arg2[%45, %46, 0] [%12, %13, 128] [1, 1, 1], reuse : [seq = false, spat = false, temp = false] : memref<32x4096x128xf32> to memref<?x?x128xf32, strided<[524288, 128, 1], offset: ?>>
              %48 = loom.semaphore %36 : memref<?x?x128xf32> -> memref<?x?x128xf32>
              %49 = loom.copy_to_tensor %47, %48 on @L1, interconnect : [], broadcast : [1, 1] : memref<?x?x128xf32, strided<[524288, 128, 1], offset: ?>>, memref<?x?x128xf32> -> tensor<?x?x128xf32>
              %50 = linalg.fill ins(%cst_1 : f32) outs(%41 : tensor<?x?x128xf32>) -> tensor<?x?x128xf32>
              %51 = linalg.fill ins(%cst_2 : f32) outs(%23 : tensor<?x?xf32>) -> tensor<?x?xf32>
              %52 = linalg.fill ins(%cst_3 : f32) outs(%26 : tensor<?x?xf32>) -> tensor<?x?xf32>
              %53:3 = affine.for %arg8 = 0 to affine_map<()[s0] -> (4096 ceildiv s0)>()[%14] iter_args(%arg9 = %52, %arg10 = %51, %arg11 = %50) -> (tensor<?x?xf32>, tensor<?x?xf32>, tensor<?x?x128xf32>) {
                %56 = arith.muli %arg8, %14 : index
                %57 = loom.subview %arg0[%45, 0, %56] [%12, 128, %14] [1, 1, 1], reuse : [seq = false, spat = false, temp = false] : memref<32x128x4096xf32> to memref<?x128x?xf32, strided<[524288, 4096, 1], offset: ?>>
                %58 = loom.semaphore %20 : memref<?x128x?xf32> -> memref<?x128x?xf32>
                %59 = loom.copy_to_tensor %57, %58 on @L1, interconnect : [], broadcast : [1, 1] : memref<?x128x?xf32, strided<[524288, 4096, 1], offset: ?>>, memref<?x128x?xf32> -> tensor<?x128x?xf32>
                %60 = linalg.fill ins(%cst_1 : f32) outs(%19 : tensor<?x?x?xf32>) -> tensor<?x?x?xf32>
                %61 = linalg.batch_matmul ins(%49, %59 : tensor<?x?x128xf32>, tensor<?x128x?xf32>) outs(%60 : tensor<?x?x?xf32>) -> tensor<?x?x?xf32>
                %62 = linalg.fill ins(%cst_3 : f32) outs(%29 : tensor<?x?xf32>) -> tensor<?x?xf32>
                %63 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%61 : tensor<?x?x?xf32>) outs(%62 : tensor<?x?xf32>) {
                ^bb0(%in: f32, %out: f32):
                  %77 = arith.maximumf %in, %out : f32
                  linalg.yield %77 : f32
                } -> tensor<?x?xf32>
                %64 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%arg9, %63 : tensor<?x?xf32>, tensor<?x?xf32>) outs(%29 : tensor<?x?xf32>) {
                ^bb0(%in: f32, %in_4: f32, %out: f32):
                  %77 = arith.truncf %cst_0 : f64 to f32
                  %78 = arith.mulf %in_4, %77 : f32
                  %79 = arith.cmpf ogt, %in, %78 : f32
                  %80 = arith.select %79, %in, %78 : f32
                  linalg.yield %80 : f32
                } -> tensor<?x?xf32>
                %65 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%61, %64 : tensor<?x?x?xf32>, tensor<?x?xf32>) outs(%19 : tensor<?x?x?xf32>) {
                ^bb0(%in: f32, %in_4: f32, %out: f32):
                  %77 = arith.truncf %cst_0 : f64 to f32
                  %78 = arith.mulf %in, %77 : f32
                  %79 = arith.subf %78, %in_4 : f32
                  %80 = math.powf %cst, %79 : f32
                  linalg.yield %80 : f32
                } -> tensor<?x?x?xf32>
                %66 = linalg.fill ins(%cst_1 : f32) outs(%32 : tensor<?x?xf32>) -> tensor<?x?xf32>
                %67 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%65 : tensor<?x?x?xf32>) outs(%66 : tensor<?x?xf32>) {
                ^bb0(%in: f32, %out: f32):
                  %77 = arith.addf %in, %out : f32
                  linalg.yield %77 : f32
                } -> tensor<?x?xf32>
                %68 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%arg9, %64 : tensor<?x?xf32>, tensor<?x?xf32>) outs(%35 : tensor<?x?xf32>) {
                ^bb0(%in: f32, %in_4: f32, %out: f32):
                  %77 = arith.subf %in, %in_4 : f32
                  %78 = math.powf %cst, %77 : f32
                  linalg.yield %78 : f32
                } -> tensor<?x?xf32>
                %69 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%arg10, %68, %67 : tensor<?x?xf32>, tensor<?x?xf32>, tensor<?x?xf32>) outs(%arg10 : tensor<?x?xf32>) {
                ^bb0(%in: f32, %in_4: f32, %in_5: f32, %out: f32):
                  %77 = arith.mulf %in, %in_4 : f32
                  %78 = arith.addf %77, %in_5 : f32
                  linalg.yield %78 : f32
                } -> tensor<?x?xf32>
                %70 = loom.subview %arg1[%45, %56, 0] [%12, %14, 128] [1, 1, 1], reuse : [seq = false, spat = false, temp = false] : memref<32x4096x128xf32> to memref<?x?x128xf32, strided<[524288, 128, 1], offset: ?>>
                %71 = loom.semaphore %16 : memref<?x?x128xf32> -> memref<?x?x128xf32>
                %72 = loom.copy_to_tensor %70, %71 on @L1, interconnect : [], broadcast : [1, 1] : memref<?x?x128xf32, strided<[524288, 128, 1], offset: ?>>, memref<?x?x128xf32> -> tensor<?x?x128xf32>
                %73 = linalg.fill ins(%cst_1 : f32) outs(%44 : tensor<?x?x128xf32>) -> tensor<?x?x128xf32>
                %74 = linalg.batch_matmul ins(%65, %72 : tensor<?x?x?xf32>, tensor<?x?x128xf32>) outs(%73 : tensor<?x?x128xf32>) -> tensor<?x?x128xf32>
                %75 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%74, %arg11, %68 : tensor<?x?x128xf32>, tensor<?x?x128xf32>, tensor<?x?xf32>) outs(%arg11 : tensor<?x?x128xf32>) {
                ^bb0(%in: f32, %in_4: f32, %in_5: f32, %out: f32):
                  %77 = arith.mulf %in_4, %in_5 : f32
                  %78 = arith.addf %in, %77 : f32
                  linalg.yield %78 : f32
                } -> tensor<?x?x128xf32>
                %76 = linalg.copy ins(%64 : tensor<?x?xf32>) outs(%arg9 : tensor<?x?xf32>) -> tensor<?x?xf32>
                affine.yield %76, %69, %75 : tensor<?x?xf32>, tensor<?x?xf32>, tensor<?x?x128xf32>
              } {loom.iter_type = #loom.iter_type<sequential>}
              %54 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%53#2, %53#1 : tensor<?x?x128xf32>, tensor<?x?xf32>) outs(%38 : tensor<?x?x128xf32>) {
              ^bb0(%in: f32, %in_4: f32, %out: f32):
                %56 = arith.divf %in, %in_4 : f32
                linalg.yield %56 : f32
              } -> tensor<?x?x128xf32>
              %55 = loom.subview %arg3[%45, %46, 0] [%12, %13, 128] [1, 1, 1], reuse : [seq = false, spat = false, temp = false] : memref<32x4096x128xf32> to memref<?x?x128xf32, strided<[524288, 128, 1], offset: ?>>
              loom.copy_from_tensor %54, %55 on @DRAM : tensor<?x?x128xf32>, memref<?x?x128xf32, strided<[524288, 128, 1], offset: ?>>
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
              %18 = loom.semaphore %17 : memref<?x?x?xf32> -> memref<?x?x?xf32>
              %19 = loom.init_tensor %18[%12, %13, %14] : memref<?x?x?xf32> -> tensor<?x?x?xf32>
              %20 = loom.alloc [%12, 128, %14] on @L1 : memref<?x128x?xf32>
              %21 = loom.alloc [%12, %13] on @L1 : memref<?x?xf32>
              %22 = loom.semaphore %21 : memref<?x?xf32> -> memref<?x?xf32>
              %23 = loom.init_tensor %22[%12, %13] : memref<?x?xf32> -> tensor<?x?xf32>
              %24 = loom.alloc [%12, %13] on @L1 : memref<?x?xf32>
              %25 = loom.semaphore %24 : memref<?x?xf32> -> memref<?x?xf32>
              %26 = loom.init_tensor %25[%12, %13] : memref<?x?xf32> -> tensor<?x?xf32>
              %27 = loom.alloc [%12, %13] on @L1 : memref<?x?xf32>
              %28 = loom.semaphore %27 : memref<?x?xf32> -> memref<?x?xf32>
              %29 = loom.init_tensor %28[%12, %13] : memref<?x?xf32> -> tensor<?x?xf32>
              %30 = loom.alloc [%12, %13] on @L1 : memref<?x?xf32>
              %31 = loom.semaphore %30 : memref<?x?xf32> -> memref<?x?xf32>
              %32 = loom.init_tensor %31[%12, %13] : memref<?x?xf32> -> tensor<?x?xf32>
              %33 = loom.alloc [%12, %13] on @L1 : memref<?x?xf32>
              %34 = loom.semaphore %33 : memref<?x?xf32> -> memref<?x?xf32>
              %35 = loom.init_tensor %34[%12, %13] : memref<?x?xf32> -> tensor<?x?xf32>
              %36 = loom.alloc [%12, %13, 128] on @L1 : memref<?x?x128xf32>
              %37 = loom.semaphore %36 : memref<?x?x128xf32> -> memref<?x?x128xf32>
              %38 = loom.init_tensor %37[%12, %13, 128] : memref<?x?x128xf32> -> tensor<?x?x128xf32>
              %39 = loom.alloc [%12, %13, 128] on @L1 : memref<?x?x128xf32>
              %40 = loom.semaphore %39 : memref<?x?x128xf32> -> memref<?x?x128xf32>
              %41 = loom.init_tensor %40[%12, %13, 128] : memref<?x?x128xf32> -> tensor<?x?x128xf32>
              %42 = loom.alloc [%12, %13, 128] on @L1 : memref<?x?x128xf32>
              %43 = loom.semaphore %42 : memref<?x?x128xf32> -> memref<?x?x128xf32>
              %44 = loom.init_tensor %43[%12, %13, 128] : memref<?x?x128xf32> -> tensor<?x?x128xf32>
              %45 = arith.muli %15, %12 : index
              %46 = arith.muli %arg6, %13 : index
              %47 = loom.subview %arg2[%45, %46, 0] [%12, %13, 128] [1, 1, 1], reuse : [seq = false, spat = false, temp = false] : memref<32x4096x128xf32> to memref<?x?x128xf32, strided<[524288, 128, 1], offset: ?>>
              %48 = loom.semaphore %36 : memref<?x?x128xf32> -> memref<?x?x128xf32>
              %49 = loom.copy_to_tensor %47, %48 on @L1, interconnect : [], broadcast : [1, 1] : memref<?x?x128xf32, strided<[524288, 128, 1], offset: ?>>, memref<?x?x128xf32> -> tensor<?x?x128xf32>
              %50 = linalg.fill ins(%cst_1 : f32) outs(%41 : tensor<?x?x128xf32>) -> tensor<?x?x128xf32>
              %51 = linalg.fill ins(%cst_2 : f32) outs(%23 : tensor<?x?xf32>) -> tensor<?x?xf32>
              %52 = linalg.fill ins(%cst_3 : f32) outs(%26 : tensor<?x?xf32>) -> tensor<?x?xf32>
              %53:3 = affine.for %arg8 = 0 to affine_map<()[s0] -> (4096 ceildiv s0)>()[%14] iter_args(%arg9 = %52, %arg10 = %51, %arg11 = %50) -> (tensor<?x?xf32>, tensor<?x?xf32>, tensor<?x?x128xf32>) {
                %56 = arith.muli %arg8, %14 : index
                %57 = loom.subview %arg0[%45, 0, %56] [%12, 128, %14] [1, 1, 1], reuse : [seq = false, spat = false, temp = false] : memref<32x128x4096xf32> to memref<?x128x?xf32, strided<[524288, 4096, 1], offset: ?>>
                %58 = loom.semaphore %20 : memref<?x128x?xf32> -> memref<?x128x?xf32>
                %59 = loom.copy_to_tensor %57, %58 on @L1, interconnect : [], broadcast : [1, 1] : memref<?x128x?xf32, strided<[524288, 4096, 1], offset: ?>>, memref<?x128x?xf32> -> tensor<?x128x?xf32>
                %60 = linalg.fill ins(%cst_1 : f32) outs(%19 : tensor<?x?x?xf32>) -> tensor<?x?x?xf32>
                %61 = linalg.batch_matmul ins(%49, %59 : tensor<?x?x128xf32>, tensor<?x128x?xf32>) outs(%60 : tensor<?x?x?xf32>) -> tensor<?x?x?xf32>
                %62 = linalg.fill ins(%cst_3 : f32) outs(%29 : tensor<?x?xf32>) -> tensor<?x?xf32>
                %63 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%61 : tensor<?x?x?xf32>) outs(%62 : tensor<?x?xf32>) {
                ^bb0(%in: f32, %out: f32):
                  %77 = arith.maximumf %in, %out : f32
                  linalg.yield %77 : f32
                } -> tensor<?x?xf32>
                %64 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%arg9, %63 : tensor<?x?xf32>, tensor<?x?xf32>) outs(%29 : tensor<?x?xf32>) {
                ^bb0(%in: f32, %in_4: f32, %out: f32):
                  %77 = arith.truncf %cst_0 : f64 to f32
                  %78 = arith.mulf %in_4, %77 : f32
                  %79 = arith.cmpf ogt, %in, %78 : f32
                  %80 = arith.select %79, %in, %78 : f32
                  linalg.yield %80 : f32
                } -> tensor<?x?xf32>
                %65 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%61, %64 : tensor<?x?x?xf32>, tensor<?x?xf32>) outs(%19 : tensor<?x?x?xf32>) {
                ^bb0(%in: f32, %in_4: f32, %out: f32):
                  %77 = arith.truncf %cst_0 : f64 to f32
                  %78 = arith.mulf %in, %77 : f32
                  %79 = arith.subf %78, %in_4 : f32
                  %80 = math.powf %cst, %79 : f32
                  linalg.yield %80 : f32
                } -> tensor<?x?x?xf32>
                %66 = linalg.fill ins(%cst_1 : f32) outs(%32 : tensor<?x?xf32>) -> tensor<?x?xf32>
                %67 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%65 : tensor<?x?x?xf32>) outs(%66 : tensor<?x?xf32>) {
                ^bb0(%in: f32, %out: f32):
                  %77 = arith.addf %in, %out : f32
                  linalg.yield %77 : f32
                } -> tensor<?x?xf32>
                %68 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%arg9, %64 : tensor<?x?xf32>, tensor<?x?xf32>) outs(%35 : tensor<?x?xf32>) {
                ^bb0(%in: f32, %in_4: f32, %out: f32):
                  %77 = arith.subf %in, %in_4 : f32
                  %78 = math.powf %cst, %77 : f32
                  linalg.yield %78 : f32
                } -> tensor<?x?xf32>
                %69 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%arg10, %68, %67 : tensor<?x?xf32>, tensor<?x?xf32>, tensor<?x?xf32>) outs(%arg10 : tensor<?x?xf32>) {
                ^bb0(%in: f32, %in_4: f32, %in_5: f32, %out: f32):
                  %77 = arith.mulf %in, %in_4 : f32
                  %78 = arith.addf %77, %in_5 : f32
                  linalg.yield %78 : f32
                } -> tensor<?x?xf32>
                %70 = loom.subview %arg1[%45, %56, 0] [%12, %14, 128] [1, 1, 1], reuse : [seq = false, spat = false, temp = false] : memref<32x4096x128xf32> to memref<?x?x128xf32, strided<[524288, 128, 1], offset: ?>>
                %71 = loom.semaphore %16 : memref<?x?x128xf32> -> memref<?x?x128xf32>
                %72 = loom.copy_to_tensor %70, %71 on @L1, interconnect : [], broadcast : [1, 1] : memref<?x?x128xf32, strided<[524288, 128, 1], offset: ?>>, memref<?x?x128xf32> -> tensor<?x?x128xf32>
                %73 = linalg.fill ins(%cst_1 : f32) outs(%44 : tensor<?x?x128xf32>) -> tensor<?x?x128xf32>
                %74 = linalg.batch_matmul ins(%65, %72 : tensor<?x?x?xf32>, tensor<?x?x128xf32>) outs(%73 : tensor<?x?x128xf32>) -> tensor<?x?x128xf32>
                %75 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%74, %arg11, %68 : tensor<?x?x128xf32>, tensor<?x?x128xf32>, tensor<?x?xf32>) outs(%arg11 : tensor<?x?x128xf32>) {
                ^bb0(%in: f32, %in_4: f32, %in_5: f32, %out: f32):
                  %77 = arith.mulf %in_4, %in_5 : f32
                  %78 = arith.addf %in, %77 : f32
                  linalg.yield %78 : f32
                } -> tensor<?x?x128xf32>
                %76 = linalg.copy ins(%64 : tensor<?x?xf32>) outs(%arg9 : tensor<?x?xf32>) -> tensor<?x?xf32>
                affine.yield %76, %69, %75 : tensor<?x?xf32>, tensor<?x?xf32>, tensor<?x?x128xf32>
              } {loom.iter_type = #loom.iter_type<sequential>}
              %54 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%53#2, %53#1 : tensor<?x?x128xf32>, tensor<?x?xf32>) outs(%38 : tensor<?x?x128xf32>) {
              ^bb0(%in: f32, %in_4: f32, %out: f32):
                %56 = arith.divf %in, %in_4 : f32
                linalg.yield %56 : f32
              } -> tensor<?x?x128xf32>
              %55 = loom.subview %arg3[%45, %46, 0] [%12, %13, 128] [1, 1, 1], reuse : [seq = false, spat = false, temp = false] : memref<32x4096x128xf32> to memref<?x?x128xf32, strided<[524288, 128, 1], offset: ?>>
              loom.copy_from_tensor %54, %55 on @DRAM : tensor<?x?x128xf32>, memref<?x?x128xf32, strided<[524288, 128, 1], offset: ?>>
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
      loom.polynomial_constraint(%12) {monomials = [{coeff = 64 : i64, vars = [0]}], upper_bound = 32 : i64}
    }
    func.func @attention__d1i0_d0i0__f01(%arg0: memref<32x128x4096xf32>, %arg1: memref<32x4096x128xf32>, %arg2: memref<32x4096x128xf32>, %arg3: memref<32x4096x128xf32>) {
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
              %18 = loom.semaphore %17 : memref<?x?x?xf32> -> memref<?x?x?xf32>
              %19 = loom.init_tensor %18[%12, %13, %14] : memref<?x?x?xf32> -> tensor<?x?x?xf32>
              %20 = loom.alloc [%12, 128, %14] on @L1 : memref<?x128x?xf32>
              %21 = loom.alloc [%12, %13] on @L1 : memref<?x?xf32>
              %22 = loom.semaphore %21 : memref<?x?xf32> -> memref<?x?xf32>
              %23 = loom.init_tensor %22[%12, %13] : memref<?x?xf32> -> tensor<?x?xf32>
              %24 = loom.alloc [%12, %13] on @L1 : memref<?x?xf32>
              %25 = loom.semaphore %24 : memref<?x?xf32> -> memref<?x?xf32>
              %26 = loom.init_tensor %25[%12, %13] : memref<?x?xf32> -> tensor<?x?xf32>
              %27 = loom.alloc [%12, %13] on @L1 : memref<?x?xf32>
              %28 = loom.semaphore %27 : memref<?x?xf32> -> memref<?x?xf32>
              %29 = loom.init_tensor %28[%12, %13] : memref<?x?xf32> -> tensor<?x?xf32>
              %30 = loom.alloc [%12, %13] on @L1 : memref<?x?xf32>
              %31 = loom.semaphore %30 : memref<?x?xf32> -> memref<?x?xf32>
              %32 = loom.init_tensor %31[%12, %13] : memref<?x?xf32> -> tensor<?x?xf32>
              %33 = loom.alloc [%12, %13] on @L1 : memref<?x?xf32>
              %34 = loom.semaphore %33 : memref<?x?xf32> -> memref<?x?xf32>
              %35 = loom.init_tensor %34[%12, %13] : memref<?x?xf32> -> tensor<?x?xf32>
              %36 = loom.alloc [%12, %13, 128] on @L1 : memref<?x?x128xf32>
              %37 = loom.semaphore %36 : memref<?x?x128xf32> -> memref<?x?x128xf32>
              %38 = loom.init_tensor %37[%12, %13, 128] : memref<?x?x128xf32> -> tensor<?x?x128xf32>
              %39 = loom.alloc [%12, %13, 128] on @L1 : memref<?x?x128xf32>
              %40 = loom.semaphore %39 : memref<?x?x128xf32> -> memref<?x?x128xf32>
              %41 = loom.init_tensor %40[%12, %13, 128] : memref<?x?x128xf32> -> tensor<?x?x128xf32>
              %42 = loom.alloc [%12, %13, 128] on @L1 : memref<?x?x128xf32>
              %43 = loom.semaphore %42 : memref<?x?x128xf32> -> memref<?x?x128xf32>
              %44 = loom.init_tensor %43[%12, %13, 128] : memref<?x?x128xf32> -> tensor<?x?x128xf32>
              %45 = arith.muli %15, %12 : index
              %46 = arith.muli %arg7, %13 : index
              %47 = loom.subview %arg2[%45, %46, 0] [%12, %13, 128] [1, 1, 1], reuse : [seq = false, spat = false, temp = false] : memref<32x4096x128xf32> to memref<?x?x128xf32, strided<[524288, 128, 1], offset: ?>>
              %48 = loom.semaphore %36 : memref<?x?x128xf32> -> memref<?x?x128xf32>
              %49 = loom.copy_to_tensor %47, %48 on @L1, interconnect : [], broadcast : [1, 1] : memref<?x?x128xf32, strided<[524288, 128, 1], offset: ?>>, memref<?x?x128xf32> -> tensor<?x?x128xf32>
              %50 = linalg.fill ins(%cst_1 : f32) outs(%41 : tensor<?x?x128xf32>) -> tensor<?x?x128xf32>
              %51 = linalg.fill ins(%cst_2 : f32) outs(%23 : tensor<?x?xf32>) -> tensor<?x?xf32>
              %52 = linalg.fill ins(%cst_3 : f32) outs(%26 : tensor<?x?xf32>) -> tensor<?x?xf32>
              %53:3 = affine.for %arg8 = 0 to affine_map<()[s0] -> (4096 ceildiv s0)>()[%14] iter_args(%arg9 = %52, %arg10 = %51, %arg11 = %50) -> (tensor<?x?xf32>, tensor<?x?xf32>, tensor<?x?x128xf32>) {
                %56 = arith.muli %arg8, %14 : index
                %57 = loom.subview %arg0[%45, 0, %56] [%12, 128, %14] [1, 1, 1], reuse : [seq = false, spat = false, temp = false] : memref<32x128x4096xf32> to memref<?x128x?xf32, strided<[524288, 4096, 1], offset: ?>>
                %58 = loom.semaphore %20 : memref<?x128x?xf32> -> memref<?x128x?xf32>
                %59 = loom.copy_to_tensor %57, %58 on @L1, interconnect : [], broadcast : [1, 1] : memref<?x128x?xf32, strided<[524288, 4096, 1], offset: ?>>, memref<?x128x?xf32> -> tensor<?x128x?xf32>
                %60 = linalg.fill ins(%cst_1 : f32) outs(%19 : tensor<?x?x?xf32>) -> tensor<?x?x?xf32>
                %61 = linalg.batch_matmul ins(%49, %59 : tensor<?x?x128xf32>, tensor<?x128x?xf32>) outs(%60 : tensor<?x?x?xf32>) -> tensor<?x?x?xf32>
                %62 = linalg.fill ins(%cst_3 : f32) outs(%29 : tensor<?x?xf32>) -> tensor<?x?xf32>
                %63 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%61 : tensor<?x?x?xf32>) outs(%62 : tensor<?x?xf32>) {
                ^bb0(%in: f32, %out: f32):
                  %77 = arith.maximumf %in, %out : f32
                  linalg.yield %77 : f32
                } -> tensor<?x?xf32>
                %64 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%arg9, %63 : tensor<?x?xf32>, tensor<?x?xf32>) outs(%29 : tensor<?x?xf32>) {
                ^bb0(%in: f32, %in_4: f32, %out: f32):
                  %77 = arith.truncf %cst_0 : f64 to f32
                  %78 = arith.mulf %in_4, %77 : f32
                  %79 = arith.cmpf ogt, %in, %78 : f32
                  %80 = arith.select %79, %in, %78 : f32
                  linalg.yield %80 : f32
                } -> tensor<?x?xf32>
                %65 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%61, %64 : tensor<?x?x?xf32>, tensor<?x?xf32>) outs(%19 : tensor<?x?x?xf32>) {
                ^bb0(%in: f32, %in_4: f32, %out: f32):
                  %77 = arith.truncf %cst_0 : f64 to f32
                  %78 = arith.mulf %in, %77 : f32
                  %79 = arith.subf %78, %in_4 : f32
                  %80 = math.powf %cst, %79 : f32
                  linalg.yield %80 : f32
                } -> tensor<?x?x?xf32>
                %66 = linalg.fill ins(%cst_1 : f32) outs(%32 : tensor<?x?xf32>) -> tensor<?x?xf32>
                %67 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%65 : tensor<?x?x?xf32>) outs(%66 : tensor<?x?xf32>) {
                ^bb0(%in: f32, %out: f32):
                  %77 = arith.addf %in, %out : f32
                  linalg.yield %77 : f32
                } -> tensor<?x?xf32>
                %68 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%arg9, %64 : tensor<?x?xf32>, tensor<?x?xf32>) outs(%35 : tensor<?x?xf32>) {
                ^bb0(%in: f32, %in_4: f32, %out: f32):
                  %77 = arith.subf %in, %in_4 : f32
                  %78 = math.powf %cst, %77 : f32
                  linalg.yield %78 : f32
                } -> tensor<?x?xf32>
                %69 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%arg10, %68, %67 : tensor<?x?xf32>, tensor<?x?xf32>, tensor<?x?xf32>) outs(%arg10 : tensor<?x?xf32>) {
                ^bb0(%in: f32, %in_4: f32, %in_5: f32, %out: f32):
                  %77 = arith.mulf %in, %in_4 : f32
                  %78 = arith.addf %77, %in_5 : f32
                  linalg.yield %78 : f32
                } -> tensor<?x?xf32>
                %70 = loom.subview %arg1[%45, %56, 0] [%12, %14, 128] [1, 1, 1], reuse : [seq = false, spat = false, temp = false] : memref<32x4096x128xf32> to memref<?x?x128xf32, strided<[524288, 128, 1], offset: ?>>
                %71 = loom.semaphore %16 : memref<?x?x128xf32> -> memref<?x?x128xf32>
                %72 = loom.copy_to_tensor %70, %71 on @L1, interconnect : [], broadcast : [1, 1] : memref<?x?x128xf32, strided<[524288, 128, 1], offset: ?>>, memref<?x?x128xf32> -> tensor<?x?x128xf32>
                %73 = linalg.fill ins(%cst_1 : f32) outs(%44 : tensor<?x?x128xf32>) -> tensor<?x?x128xf32>
                %74 = linalg.batch_matmul ins(%65, %72 : tensor<?x?x?xf32>, tensor<?x?x128xf32>) outs(%73 : tensor<?x?x128xf32>) -> tensor<?x?x128xf32>
                %75 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%74, %arg11, %68 : tensor<?x?x128xf32>, tensor<?x?x128xf32>, tensor<?x?xf32>) outs(%arg11 : tensor<?x?x128xf32>) {
                ^bb0(%in: f32, %in_4: f32, %in_5: f32, %out: f32):
                  %77 = arith.mulf %in_4, %in_5 : f32
                  %78 = arith.addf %in, %77 : f32
                  linalg.yield %78 : f32
                } -> tensor<?x?x128xf32>
                %76 = linalg.copy ins(%64 : tensor<?x?xf32>) outs(%arg9 : tensor<?x?xf32>) -> tensor<?x?xf32>
                affine.yield %76, %69, %75 : tensor<?x?xf32>, tensor<?x?xf32>, tensor<?x?x128xf32>
              } {loom.iter_type = #loom.iter_type<sequential>}
              %54 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%53#2, %53#1 : tensor<?x?x128xf32>, tensor<?x?xf32>) outs(%38 : tensor<?x?x128xf32>) {
              ^bb0(%in: f32, %in_4: f32, %out: f32):
                %56 = arith.divf %in, %in_4 : f32
                linalg.yield %56 : f32
              } -> tensor<?x?x128xf32>
              %55 = loom.subview %arg3[%45, %46, 0] [%12, %13, 128] [1, 1, 1], reuse : [seq = false, spat = false, temp = false] : memref<32x4096x128xf32> to memref<?x?x128xf32, strided<[524288, 128, 1], offset: ?>>
              loom.copy_from_tensor %54, %55 on @DRAM : tensor<?x?x128xf32>, memref<?x?x128xf32, strided<[524288, 128, 1], offset: ?>>
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
      loom.polynomial_constraint(%12) {monomials = [{coeff = 64 : i64, vars = [0]}], upper_bound = 32 : i64}
    }
    func.func @attention__d1i0_d0i0__f10(%arg0: memref<32x128x4096xf32>, %arg1: memref<32x4096x128xf32>, %arg2: memref<32x4096x128xf32>, %arg3: memref<32x4096x128xf32>) {
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
              %18 = loom.semaphore %17 : memref<?x?x?xf32> -> memref<?x?x?xf32>
              %19 = loom.init_tensor %18[%12, %13, %14] : memref<?x?x?xf32> -> tensor<?x?x?xf32>
              %20 = loom.alloc [%12, 128, %14] on @L1 : memref<?x128x?xf32>
              %21 = loom.alloc [%12, %13] on @L1 : memref<?x?xf32>
              %22 = loom.semaphore %21 : memref<?x?xf32> -> memref<?x?xf32>
              %23 = loom.init_tensor %22[%12, %13] : memref<?x?xf32> -> tensor<?x?xf32>
              %24 = loom.alloc [%12, %13] on @L1 : memref<?x?xf32>
              %25 = loom.semaphore %24 : memref<?x?xf32> -> memref<?x?xf32>
              %26 = loom.init_tensor %25[%12, %13] : memref<?x?xf32> -> tensor<?x?xf32>
              %27 = loom.alloc [%12, %13] on @L1 : memref<?x?xf32>
              %28 = loom.semaphore %27 : memref<?x?xf32> -> memref<?x?xf32>
              %29 = loom.init_tensor %28[%12, %13] : memref<?x?xf32> -> tensor<?x?xf32>
              %30 = loom.alloc [%12, %13] on @L1 : memref<?x?xf32>
              %31 = loom.semaphore %30 : memref<?x?xf32> -> memref<?x?xf32>
              %32 = loom.init_tensor %31[%12, %13] : memref<?x?xf32> -> tensor<?x?xf32>
              %33 = loom.alloc [%12, %13] on @L1 : memref<?x?xf32>
              %34 = loom.semaphore %33 : memref<?x?xf32> -> memref<?x?xf32>
              %35 = loom.init_tensor %34[%12, %13] : memref<?x?xf32> -> tensor<?x?xf32>
              %36 = loom.alloc [%12, %13, 128] on @L1 : memref<?x?x128xf32>
              %37 = loom.semaphore %36 : memref<?x?x128xf32> -> memref<?x?x128xf32>
              %38 = loom.init_tensor %37[%12, %13, 128] : memref<?x?x128xf32> -> tensor<?x?x128xf32>
              %39 = loom.alloc [%12, %13, 128] on @L1 : memref<?x?x128xf32>
              %40 = loom.semaphore %39 : memref<?x?x128xf32> -> memref<?x?x128xf32>
              %41 = loom.init_tensor %40[%12, %13, 128] : memref<?x?x128xf32> -> tensor<?x?x128xf32>
              %42 = loom.alloc [%12, %13, 128] on @L1 : memref<?x?x128xf32>
              %43 = loom.semaphore %42 : memref<?x?x128xf32> -> memref<?x?x128xf32>
              %44 = loom.init_tensor %43[%12, %13, 128] : memref<?x?x128xf32> -> tensor<?x?x128xf32>
              %45 = arith.muli %15, %12 : index
              %46 = arith.muli %arg6, %13 : index
              %47 = loom.subview %arg2[%45, %46, 0] [%12, %13, 128] [1, 1, 1], reuse : [seq = false, spat = false, temp = false] : memref<32x4096x128xf32> to memref<?x?x128xf32, strided<[524288, 128, 1], offset: ?>>
              %48 = loom.semaphore %36 : memref<?x?x128xf32> -> memref<?x?x128xf32>
              %49 = loom.copy_to_tensor %47, %48 on @L1, interconnect : [], broadcast : [1, 1] : memref<?x?x128xf32, strided<[524288, 128, 1], offset: ?>>, memref<?x?x128xf32> -> tensor<?x?x128xf32>
              %50 = linalg.fill ins(%cst_1 : f32) outs(%41 : tensor<?x?x128xf32>) -> tensor<?x?x128xf32>
              %51 = linalg.fill ins(%cst_2 : f32) outs(%23 : tensor<?x?xf32>) -> tensor<?x?xf32>
              %52 = linalg.fill ins(%cst_3 : f32) outs(%26 : tensor<?x?xf32>) -> tensor<?x?xf32>
              %53:3 = affine.for %arg8 = 0 to affine_map<()[s0] -> (4096 ceildiv s0)>()[%14] iter_args(%arg9 = %52, %arg10 = %51, %arg11 = %50) -> (tensor<?x?xf32>, tensor<?x?xf32>, tensor<?x?x128xf32>) {
                %56 = arith.muli %arg8, %14 : index
                %57 = loom.subview %arg0[%45, 0, %56] [%12, 128, %14] [1, 1, 1], reuse : [seq = false, spat = false, temp = false] : memref<32x128x4096xf32> to memref<?x128x?xf32, strided<[524288, 4096, 1], offset: ?>>
                %58 = loom.semaphore %20 : memref<?x128x?xf32> -> memref<?x128x?xf32>
                %59 = loom.copy_to_tensor %57, %58 on @L1, interconnect : [], broadcast : [1, 1] : memref<?x128x?xf32, strided<[524288, 4096, 1], offset: ?>>, memref<?x128x?xf32> -> tensor<?x128x?xf32>
                %60 = linalg.fill ins(%cst_1 : f32) outs(%19 : tensor<?x?x?xf32>) -> tensor<?x?x?xf32>
                %61 = linalg.batch_matmul ins(%49, %59 : tensor<?x?x128xf32>, tensor<?x128x?xf32>) outs(%60 : tensor<?x?x?xf32>) -> tensor<?x?x?xf32>
                %62 = linalg.fill ins(%cst_3 : f32) outs(%29 : tensor<?x?xf32>) -> tensor<?x?xf32>
                %63 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%61 : tensor<?x?x?xf32>) outs(%62 : tensor<?x?xf32>) {
                ^bb0(%in: f32, %out: f32):
                  %77 = arith.maximumf %in, %out : f32
                  linalg.yield %77 : f32
                } -> tensor<?x?xf32>
                %64 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%arg9, %63 : tensor<?x?xf32>, tensor<?x?xf32>) outs(%29 : tensor<?x?xf32>) {
                ^bb0(%in: f32, %in_4: f32, %out: f32):
                  %77 = arith.truncf %cst_0 : f64 to f32
                  %78 = arith.mulf %in_4, %77 : f32
                  %79 = arith.cmpf ogt, %in, %78 : f32
                  %80 = arith.select %79, %in, %78 : f32
                  linalg.yield %80 : f32
                } -> tensor<?x?xf32>
                %65 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%61, %64 : tensor<?x?x?xf32>, tensor<?x?xf32>) outs(%19 : tensor<?x?x?xf32>) {
                ^bb0(%in: f32, %in_4: f32, %out: f32):
                  %77 = arith.truncf %cst_0 : f64 to f32
                  %78 = arith.mulf %in, %77 : f32
                  %79 = arith.subf %78, %in_4 : f32
                  %80 = math.powf %cst, %79 : f32
                  linalg.yield %80 : f32
                } -> tensor<?x?x?xf32>
                %66 = linalg.fill ins(%cst_1 : f32) outs(%32 : tensor<?x?xf32>) -> tensor<?x?xf32>
                %67 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%65 : tensor<?x?x?xf32>) outs(%66 : tensor<?x?xf32>) {
                ^bb0(%in: f32, %out: f32):
                  %77 = arith.addf %in, %out : f32
                  linalg.yield %77 : f32
                } -> tensor<?x?xf32>
                %68 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%arg9, %64 : tensor<?x?xf32>, tensor<?x?xf32>) outs(%35 : tensor<?x?xf32>) {
                ^bb0(%in: f32, %in_4: f32, %out: f32):
                  %77 = arith.subf %in, %in_4 : f32
                  %78 = math.powf %cst, %77 : f32
                  linalg.yield %78 : f32
                } -> tensor<?x?xf32>
                %69 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%arg10, %68, %67 : tensor<?x?xf32>, tensor<?x?xf32>, tensor<?x?xf32>) outs(%arg10 : tensor<?x?xf32>) {
                ^bb0(%in: f32, %in_4: f32, %in_5: f32, %out: f32):
                  %77 = arith.mulf %in, %in_4 : f32
                  %78 = arith.addf %77, %in_5 : f32
                  linalg.yield %78 : f32
                } -> tensor<?x?xf32>
                %70 = loom.subview %arg1[%45, %56, 0] [%12, %14, 128] [1, 1, 1], reuse : [seq = false, spat = false, temp = false] : memref<32x4096x128xf32> to memref<?x?x128xf32, strided<[524288, 128, 1], offset: ?>>
                %71 = loom.semaphore %16 : memref<?x?x128xf32> -> memref<?x?x128xf32>
                %72 = loom.copy_to_tensor %70, %71 on @L1, interconnect : [], broadcast : [1, 1] : memref<?x?x128xf32, strided<[524288, 128, 1], offset: ?>>, memref<?x?x128xf32> -> tensor<?x?x128xf32>
                %73 = linalg.fill ins(%cst_1 : f32) outs(%44 : tensor<?x?x128xf32>) -> tensor<?x?x128xf32>
                %74 = linalg.batch_matmul ins(%65, %72 : tensor<?x?x?xf32>, tensor<?x?x128xf32>) outs(%73 : tensor<?x?x128xf32>) -> tensor<?x?x128xf32>
                %75 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%74, %arg11, %68 : tensor<?x?x128xf32>, tensor<?x?x128xf32>, tensor<?x?xf32>) outs(%arg11 : tensor<?x?x128xf32>) {
                ^bb0(%in: f32, %in_4: f32, %in_5: f32, %out: f32):
                  %77 = arith.mulf %in_4, %in_5 : f32
                  %78 = arith.addf %in, %77 : f32
                  linalg.yield %78 : f32
                } -> tensor<?x?x128xf32>
                %76 = linalg.copy ins(%64 : tensor<?x?xf32>) outs(%arg9 : tensor<?x?xf32>) -> tensor<?x?xf32>
                affine.yield %76, %69, %75 : tensor<?x?xf32>, tensor<?x?xf32>, tensor<?x?x128xf32>
              } {loom.iter_type = #loom.iter_type<sequential>}
              %54 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%53#2, %53#1 : tensor<?x?x128xf32>, tensor<?x?xf32>) outs(%38 : tensor<?x?x128xf32>) {
              ^bb0(%in: f32, %in_4: f32, %out: f32):
                %56 = arith.divf %in, %in_4 : f32
                linalg.yield %56 : f32
              } -> tensor<?x?x128xf32>
              %55 = loom.subview %arg3[%45, %46, 0] [%12, %13, 128] [1, 1, 1], reuse : [seq = false, spat = false, temp = false] : memref<32x4096x128xf32> to memref<?x?x128xf32, strided<[524288, 128, 1], offset: ?>>
              loom.copy_from_tensor %54, %55 on @DRAM : tensor<?x?x128xf32>, memref<?x?x128xf32, strided<[524288, 128, 1], offset: ?>>
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
              %19 = loom.semaphore %18 : memref<?x?x?xf32> -> memref<?x?x?xf32>
              %20 = loom.init_tensor %19[%12, %13, %14] : memref<?x?x?xf32> -> tensor<?x?x?xf32>
              %21 = loom.alloc [%12, 128, %14] on @L1 : memref<?x128x?xf32>
              %22 = loom.alloc [%12, %13] on @L1 : memref<?x?xf32>
              %23 = loom.semaphore %22 : memref<?x?xf32> -> memref<?x?xf32>
              %24 = loom.init_tensor %23[%12, %13] : memref<?x?xf32> -> tensor<?x?xf32>
              %25 = loom.alloc [%12, %13] on @L1 : memref<?x?xf32>
              %26 = loom.semaphore %25 : memref<?x?xf32> -> memref<?x?xf32>
              %27 = loom.init_tensor %26[%12, %13] : memref<?x?xf32> -> tensor<?x?xf32>
              %28 = loom.alloc [%12, %13] on @L1 : memref<?x?xf32>
              %29 = loom.semaphore %28 : memref<?x?xf32> -> memref<?x?xf32>
              %30 = loom.init_tensor %29[%12, %13] : memref<?x?xf32> -> tensor<?x?xf32>
              %31 = loom.alloc [%12, %13] on @L1 : memref<?x?xf32>
              %32 = loom.semaphore %31 : memref<?x?xf32> -> memref<?x?xf32>
              %33 = loom.init_tensor %32[%12, %13] : memref<?x?xf32> -> tensor<?x?xf32>
              %34 = loom.alloc [%12, %13] on @L1 : memref<?x?xf32>
              %35 = loom.semaphore %34 : memref<?x?xf32> -> memref<?x?xf32>
              %36 = loom.init_tensor %35[%12, %13] : memref<?x?xf32> -> tensor<?x?xf32>
              %37 = loom.alloc [%12, %13, 128] on @L1 : memref<?x?x128xf32>
              %38 = loom.semaphore %37 : memref<?x?x128xf32> -> memref<?x?x128xf32>
              %39 = loom.init_tensor %38[%12, %13, 128] : memref<?x?x128xf32> -> tensor<?x?x128xf32>
              %40 = loom.alloc [%12, %13, 128] on @L1 : memref<?x?x128xf32>
              %41 = loom.semaphore %40 : memref<?x?x128xf32> -> memref<?x?x128xf32>
              %42 = loom.init_tensor %41[%12, %13, 128] : memref<?x?x128xf32> -> tensor<?x?x128xf32>
              %43 = loom.alloc [%12, %13, 128] on @L1 : memref<?x?x128xf32>
              %44 = loom.semaphore %43 : memref<?x?x128xf32> -> memref<?x?x128xf32>
              %45 = loom.init_tensor %44[%12, %13, 128] : memref<?x?x128xf32> -> tensor<?x?x128xf32>
              %46 = arith.muli %15, %12 : index
              %47 = arith.muli %16, %13 : index
              %48 = loom.subview %arg2[%46, %47, 0] [%12, %13, 128] [1, 1, 1], reuse : [seq = false, spat = false, temp = false] : memref<32x4096x128xf32> to memref<?x?x128xf32, strided<[524288, 128, 1], offset: ?>>
              %49 = loom.semaphore %37 : memref<?x?x128xf32> -> memref<?x?x128xf32>
              %50 = loom.copy_to_tensor %48, %49 on @L1, interconnect : [], broadcast : [1, 1] : memref<?x?x128xf32, strided<[524288, 128, 1], offset: ?>>, memref<?x?x128xf32> -> tensor<?x?x128xf32>
              %51 = linalg.fill ins(%cst_1 : f32) outs(%42 : tensor<?x?x128xf32>) -> tensor<?x?x128xf32>
              %52 = linalg.fill ins(%cst_2 : f32) outs(%24 : tensor<?x?xf32>) -> tensor<?x?xf32>
              %53 = linalg.fill ins(%cst_3 : f32) outs(%27 : tensor<?x?xf32>) -> tensor<?x?xf32>
              %54:3 = affine.for %arg8 = 0 to affine_map<()[s0] -> (4096 ceildiv s0)>()[%14] iter_args(%arg9 = %53, %arg10 = %52, %arg11 = %51) -> (tensor<?x?xf32>, tensor<?x?xf32>, tensor<?x?x128xf32>) {
                %57 = arith.muli %arg8, %14 : index
                %58 = loom.subview %arg0[%46, 0, %57] [%12, 128, %14] [1, 1, 1], reuse : [seq = false, spat = false, temp = false] : memref<32x128x4096xf32> to memref<?x128x?xf32, strided<[524288, 4096, 1], offset: ?>>
                %59 = loom.semaphore %21 : memref<?x128x?xf32> -> memref<?x128x?xf32>
                %60 = loom.copy_to_tensor %58, %59 on @L1, interconnect : [], broadcast : [1, 1] : memref<?x128x?xf32, strided<[524288, 4096, 1], offset: ?>>, memref<?x128x?xf32> -> tensor<?x128x?xf32>
                %61 = linalg.fill ins(%cst_1 : f32) outs(%20 : tensor<?x?x?xf32>) -> tensor<?x?x?xf32>
                %62 = linalg.batch_matmul ins(%50, %60 : tensor<?x?x128xf32>, tensor<?x128x?xf32>) outs(%61 : tensor<?x?x?xf32>) -> tensor<?x?x?xf32>
                %63 = linalg.fill ins(%cst_3 : f32) outs(%30 : tensor<?x?xf32>) -> tensor<?x?xf32>
                %64 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%62 : tensor<?x?x?xf32>) outs(%63 : tensor<?x?xf32>) {
                ^bb0(%in: f32, %out: f32):
                  %78 = arith.maximumf %in, %out : f32
                  linalg.yield %78 : f32
                } -> tensor<?x?xf32>
                %65 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%arg9, %64 : tensor<?x?xf32>, tensor<?x?xf32>) outs(%30 : tensor<?x?xf32>) {
                ^bb0(%in: f32, %in_4: f32, %out: f32):
                  %78 = arith.truncf %cst_0 : f64 to f32
                  %79 = arith.mulf %in_4, %78 : f32
                  %80 = arith.cmpf ogt, %in, %79 : f32
                  %81 = arith.select %80, %in, %79 : f32
                  linalg.yield %81 : f32
                } -> tensor<?x?xf32>
                %66 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%62, %65 : tensor<?x?x?xf32>, tensor<?x?xf32>) outs(%20 : tensor<?x?x?xf32>) {
                ^bb0(%in: f32, %in_4: f32, %out: f32):
                  %78 = arith.truncf %cst_0 : f64 to f32
                  %79 = arith.mulf %in, %78 : f32
                  %80 = arith.subf %79, %in_4 : f32
                  %81 = math.powf %cst, %80 : f32
                  linalg.yield %81 : f32
                } -> tensor<?x?x?xf32>
                %67 = linalg.fill ins(%cst_1 : f32) outs(%33 : tensor<?x?xf32>) -> tensor<?x?xf32>
                %68 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%66 : tensor<?x?x?xf32>) outs(%67 : tensor<?x?xf32>) {
                ^bb0(%in: f32, %out: f32):
                  %78 = arith.addf %in, %out : f32
                  linalg.yield %78 : f32
                } -> tensor<?x?xf32>
                %69 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%arg9, %65 : tensor<?x?xf32>, tensor<?x?xf32>) outs(%36 : tensor<?x?xf32>) {
                ^bb0(%in: f32, %in_4: f32, %out: f32):
                  %78 = arith.subf %in, %in_4 : f32
                  %79 = math.powf %cst, %78 : f32
                  linalg.yield %79 : f32
                } -> tensor<?x?xf32>
                %70 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%arg10, %69, %68 : tensor<?x?xf32>, tensor<?x?xf32>, tensor<?x?xf32>) outs(%arg10 : tensor<?x?xf32>) {
                ^bb0(%in: f32, %in_4: f32, %in_5: f32, %out: f32):
                  %78 = arith.mulf %in, %in_4 : f32
                  %79 = arith.addf %78, %in_5 : f32
                  linalg.yield %79 : f32
                } -> tensor<?x?xf32>
                %71 = loom.subview %arg1[%46, %57, 0] [%12, %14, 128] [1, 1, 1], reuse : [seq = false, spat = false, temp = false] : memref<32x4096x128xf32> to memref<?x?x128xf32, strided<[524288, 128, 1], offset: ?>>
                %72 = loom.semaphore %17 : memref<?x?x128xf32> -> memref<?x?x128xf32>
                %73 = loom.copy_to_tensor %71, %72 on @L1, interconnect : [], broadcast : [1, 1] : memref<?x?x128xf32, strided<[524288, 128, 1], offset: ?>>, memref<?x?x128xf32> -> tensor<?x?x128xf32>
                %74 = linalg.fill ins(%cst_1 : f32) outs(%45 : tensor<?x?x128xf32>) -> tensor<?x?x128xf32>
                %75 = linalg.batch_matmul ins(%66, %73 : tensor<?x?x?xf32>, tensor<?x?x128xf32>) outs(%74 : tensor<?x?x128xf32>) -> tensor<?x?x128xf32>
                %76 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%75, %arg11, %69 : tensor<?x?x128xf32>, tensor<?x?x128xf32>, tensor<?x?xf32>) outs(%arg11 : tensor<?x?x128xf32>) {
                ^bb0(%in: f32, %in_4: f32, %in_5: f32, %out: f32):
                  %78 = arith.mulf %in_4, %in_5 : f32
                  %79 = arith.addf %in, %78 : f32
                  linalg.yield %79 : f32
                } -> tensor<?x?x128xf32>
                %77 = linalg.copy ins(%65 : tensor<?x?xf32>) outs(%arg9 : tensor<?x?xf32>) -> tensor<?x?xf32>
                affine.yield %77, %70, %76 : tensor<?x?xf32>, tensor<?x?xf32>, tensor<?x?x128xf32>
              } {loom.iter_type = #loom.iter_type<sequential>}
              %55 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%54#2, %54#1 : tensor<?x?x128xf32>, tensor<?x?xf32>) outs(%39 : tensor<?x?x128xf32>) {
              ^bb0(%in: f32, %in_4: f32, %out: f32):
                %57 = arith.divf %in, %in_4 : f32
                linalg.yield %57 : f32
              } -> tensor<?x?x128xf32>
              %56 = loom.subview %arg3[%46, %47, 0] [%12, %13, 128] [1, 1, 1], reuse : [seq = false, spat = false, temp = false] : memref<32x4096x128xf32> to memref<?x?x128xf32, strided<[524288, 128, 1], offset: ?>>
              loom.copy_from_tensor %55, %56 on @DRAM : tensor<?x?x128xf32>, memref<?x?x128xf32, strided<[524288, 128, 1], offset: ?>>
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
              %19 = loom.semaphore %18 : memref<?x?x?xf32> -> memref<?x?x?xf32>
              %20 = loom.init_tensor %19[%12, %13, %14] : memref<?x?x?xf32> -> tensor<?x?x?xf32>
              %21 = loom.alloc [%12, 128, %14] on @L1 : memref<?x128x?xf32>
              %22 = loom.alloc [%12, %13] on @L1 : memref<?x?xf32>
              %23 = loom.semaphore %22 : memref<?x?xf32> -> memref<?x?xf32>
              %24 = loom.init_tensor %23[%12, %13] : memref<?x?xf32> -> tensor<?x?xf32>
              %25 = loom.alloc [%12, %13] on @L1 : memref<?x?xf32>
              %26 = loom.semaphore %25 : memref<?x?xf32> -> memref<?x?xf32>
              %27 = loom.init_tensor %26[%12, %13] : memref<?x?xf32> -> tensor<?x?xf32>
              %28 = loom.alloc [%12, %13] on @L1 : memref<?x?xf32>
              %29 = loom.semaphore %28 : memref<?x?xf32> -> memref<?x?xf32>
              %30 = loom.init_tensor %29[%12, %13] : memref<?x?xf32> -> tensor<?x?xf32>
              %31 = loom.alloc [%12, %13] on @L1 : memref<?x?xf32>
              %32 = loom.semaphore %31 : memref<?x?xf32> -> memref<?x?xf32>
              %33 = loom.init_tensor %32[%12, %13] : memref<?x?xf32> -> tensor<?x?xf32>
              %34 = loom.alloc [%12, %13] on @L1 : memref<?x?xf32>
              %35 = loom.semaphore %34 : memref<?x?xf32> -> memref<?x?xf32>
              %36 = loom.init_tensor %35[%12, %13] : memref<?x?xf32> -> tensor<?x?xf32>
              %37 = loom.alloc [%12, %13, 128] on @L1 : memref<?x?x128xf32>
              %38 = loom.semaphore %37 : memref<?x?x128xf32> -> memref<?x?x128xf32>
              %39 = loom.init_tensor %38[%12, %13, 128] : memref<?x?x128xf32> -> tensor<?x?x128xf32>
              %40 = loom.alloc [%12, %13, 128] on @L1 : memref<?x?x128xf32>
              %41 = loom.semaphore %40 : memref<?x?x128xf32> -> memref<?x?x128xf32>
              %42 = loom.init_tensor %41[%12, %13, 128] : memref<?x?x128xf32> -> tensor<?x?x128xf32>
              %43 = loom.alloc [%12, %13, 128] on @L1 : memref<?x?x128xf32>
              %44 = loom.semaphore %43 : memref<?x?x128xf32> -> memref<?x?x128xf32>
              %45 = loom.init_tensor %44[%12, %13, 128] : memref<?x?x128xf32> -> tensor<?x?x128xf32>
              %46 = arith.muli %15, %12 : index
              %47 = arith.muli %16, %13 : index
              %48 = loom.subview %arg2[%46, %47, 0] [%12, %13, 128] [1, 1, 1], reuse : [seq = false, spat = false, temp = false] : memref<32x4096x128xf32> to memref<?x?x128xf32, strided<[524288, 128, 1], offset: ?>>
              %49 = loom.semaphore %37 : memref<?x?x128xf32> -> memref<?x?x128xf32>
              %50 = loom.copy_to_tensor %48, %49 on @L1, interconnect : [], broadcast : [1, 1] : memref<?x?x128xf32, strided<[524288, 128, 1], offset: ?>>, memref<?x?x128xf32> -> tensor<?x?x128xf32>
              %51 = linalg.fill ins(%cst_1 : f32) outs(%42 : tensor<?x?x128xf32>) -> tensor<?x?x128xf32>
              %52 = linalg.fill ins(%cst_2 : f32) outs(%24 : tensor<?x?xf32>) -> tensor<?x?xf32>
              %53 = linalg.fill ins(%cst_3 : f32) outs(%27 : tensor<?x?xf32>) -> tensor<?x?xf32>
              %54:3 = affine.for %arg8 = 0 to affine_map<()[s0] -> (4096 ceildiv s0)>()[%14] iter_args(%arg9 = %53, %arg10 = %52, %arg11 = %51) -> (tensor<?x?xf32>, tensor<?x?xf32>, tensor<?x?x128xf32>) {
                %57 = arith.muli %arg8, %14 : index
                %58 = loom.subview %arg0[%46, 0, %57] [%12, 128, %14] [1, 1, 1], reuse : [seq = false, spat = false, temp = false] : memref<32x128x4096xf32> to memref<?x128x?xf32, strided<[524288, 4096, 1], offset: ?>>
                %59 = loom.semaphore %21 : memref<?x128x?xf32> -> memref<?x128x?xf32>
                %60 = loom.copy_to_tensor %58, %59 on @L1, interconnect : [], broadcast : [1, 1] : memref<?x128x?xf32, strided<[524288, 4096, 1], offset: ?>>, memref<?x128x?xf32> -> tensor<?x128x?xf32>
                %61 = linalg.fill ins(%cst_1 : f32) outs(%20 : tensor<?x?x?xf32>) -> tensor<?x?x?xf32>
                %62 = linalg.batch_matmul ins(%50, %60 : tensor<?x?x128xf32>, tensor<?x128x?xf32>) outs(%61 : tensor<?x?x?xf32>) -> tensor<?x?x?xf32>
                %63 = linalg.fill ins(%cst_3 : f32) outs(%30 : tensor<?x?xf32>) -> tensor<?x?xf32>
                %64 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%62 : tensor<?x?x?xf32>) outs(%63 : tensor<?x?xf32>) {
                ^bb0(%in: f32, %out: f32):
                  %78 = arith.maximumf %in, %out : f32
                  linalg.yield %78 : f32
                } -> tensor<?x?xf32>
                %65 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%arg9, %64 : tensor<?x?xf32>, tensor<?x?xf32>) outs(%30 : tensor<?x?xf32>) {
                ^bb0(%in: f32, %in_4: f32, %out: f32):
                  %78 = arith.truncf %cst_0 : f64 to f32
                  %79 = arith.mulf %in_4, %78 : f32
                  %80 = arith.cmpf ogt, %in, %79 : f32
                  %81 = arith.select %80, %in, %79 : f32
                  linalg.yield %81 : f32
                } -> tensor<?x?xf32>
                %66 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%62, %65 : tensor<?x?x?xf32>, tensor<?x?xf32>) outs(%20 : tensor<?x?x?xf32>) {
                ^bb0(%in: f32, %in_4: f32, %out: f32):
                  %78 = arith.truncf %cst_0 : f64 to f32
                  %79 = arith.mulf %in, %78 : f32
                  %80 = arith.subf %79, %in_4 : f32
                  %81 = math.powf %cst, %80 : f32
                  linalg.yield %81 : f32
                } -> tensor<?x?x?xf32>
                %67 = linalg.fill ins(%cst_1 : f32) outs(%33 : tensor<?x?xf32>) -> tensor<?x?xf32>
                %68 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%66 : tensor<?x?x?xf32>) outs(%67 : tensor<?x?xf32>) {
                ^bb0(%in: f32, %out: f32):
                  %78 = arith.addf %in, %out : f32
                  linalg.yield %78 : f32
                } -> tensor<?x?xf32>
                %69 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%arg9, %65 : tensor<?x?xf32>, tensor<?x?xf32>) outs(%36 : tensor<?x?xf32>) {
                ^bb0(%in: f32, %in_4: f32, %out: f32):
                  %78 = arith.subf %in, %in_4 : f32
                  %79 = math.powf %cst, %78 : f32
                  linalg.yield %79 : f32
                } -> tensor<?x?xf32>
                %70 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%arg10, %69, %68 : tensor<?x?xf32>, tensor<?x?xf32>, tensor<?x?xf32>) outs(%arg10 : tensor<?x?xf32>) {
                ^bb0(%in: f32, %in_4: f32, %in_5: f32, %out: f32):
                  %78 = arith.mulf %in, %in_4 : f32
                  %79 = arith.addf %78, %in_5 : f32
                  linalg.yield %79 : f32
                } -> tensor<?x?xf32>
                %71 = loom.subview %arg1[%46, %57, 0] [%12, %14, 128] [1, 1, 1], reuse : [seq = false, spat = false, temp = false] : memref<32x4096x128xf32> to memref<?x?x128xf32, strided<[524288, 128, 1], offset: ?>>
                %72 = loom.semaphore %17 : memref<?x?x128xf32> -> memref<?x?x128xf32>
                %73 = loom.copy_to_tensor %71, %72 on @L1, interconnect : [], broadcast : [1, 1] : memref<?x?x128xf32, strided<[524288, 128, 1], offset: ?>>, memref<?x?x128xf32> -> tensor<?x?x128xf32>
                %74 = linalg.fill ins(%cst_1 : f32) outs(%45 : tensor<?x?x128xf32>) -> tensor<?x?x128xf32>
                %75 = linalg.batch_matmul ins(%66, %73 : tensor<?x?x?xf32>, tensor<?x?x128xf32>) outs(%74 : tensor<?x?x128xf32>) -> tensor<?x?x128xf32>
                %76 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%75, %arg11, %69 : tensor<?x?x128xf32>, tensor<?x?x128xf32>, tensor<?x?xf32>) outs(%arg11 : tensor<?x?x128xf32>) {
                ^bb0(%in: f32, %in_4: f32, %in_5: f32, %out: f32):
                  %78 = arith.mulf %in_4, %in_5 : f32
                  %79 = arith.addf %in, %78 : f32
                  linalg.yield %79 : f32
                } -> tensor<?x?x128xf32>
                %77 = linalg.copy ins(%65 : tensor<?x?xf32>) outs(%arg9 : tensor<?x?xf32>) -> tensor<?x?xf32>
                affine.yield %77, %70, %76 : tensor<?x?xf32>, tensor<?x?xf32>, tensor<?x?x128xf32>
              } {loom.iter_type = #loom.iter_type<sequential>}
              %55 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%54#2, %54#1 : tensor<?x?x128xf32>, tensor<?x?xf32>) outs(%39 : tensor<?x?x128xf32>) {
              ^bb0(%in: f32, %in_4: f32, %out: f32):
                %57 = arith.divf %in, %in_4 : f32
                linalg.yield %57 : f32
              } -> tensor<?x?x128xf32>
              %56 = loom.subview %arg3[%46, %47, 0] [%12, %13, 128] [1, 1, 1], reuse : [seq = false, spat = false, temp = false] : memref<32x4096x128xf32> to memref<?x?x128xf32, strided<[524288, 128, 1], offset: ?>>
              loom.copy_from_tensor %55, %56 on @DRAM : tensor<?x?x128xf32>, memref<?x?x128xf32, strided<[524288, 128, 1], offset: ?>>
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
              %19 = loom.semaphore %18 : memref<?x?x?xf32> -> memref<?x?x?xf32>
              %20 = loom.init_tensor %19[%12, %13, %14] : memref<?x?x?xf32> -> tensor<?x?x?xf32>
              %21 = loom.alloc [%12, 128, %14] on @L1 : memref<?x128x?xf32>
              %22 = loom.alloc [%12, %13] on @L1 : memref<?x?xf32>
              %23 = loom.semaphore %22 : memref<?x?xf32> -> memref<?x?xf32>
              %24 = loom.init_tensor %23[%12, %13] : memref<?x?xf32> -> tensor<?x?xf32>
              %25 = loom.alloc [%12, %13] on @L1 : memref<?x?xf32>
              %26 = loom.semaphore %25 : memref<?x?xf32> -> memref<?x?xf32>
              %27 = loom.init_tensor %26[%12, %13] : memref<?x?xf32> -> tensor<?x?xf32>
              %28 = loom.alloc [%12, %13] on @L1 : memref<?x?xf32>
              %29 = loom.semaphore %28 : memref<?x?xf32> -> memref<?x?xf32>
              %30 = loom.init_tensor %29[%12, %13] : memref<?x?xf32> -> tensor<?x?xf32>
              %31 = loom.alloc [%12, %13] on @L1 : memref<?x?xf32>
              %32 = loom.semaphore %31 : memref<?x?xf32> -> memref<?x?xf32>
              %33 = loom.init_tensor %32[%12, %13] : memref<?x?xf32> -> tensor<?x?xf32>
              %34 = loom.alloc [%12, %13] on @L1 : memref<?x?xf32>
              %35 = loom.semaphore %34 : memref<?x?xf32> -> memref<?x?xf32>
              %36 = loom.init_tensor %35[%12, %13] : memref<?x?xf32> -> tensor<?x?xf32>
              %37 = loom.alloc [%12, %13, 128] on @L1 : memref<?x?x128xf32>
              %38 = loom.semaphore %37 : memref<?x?x128xf32> -> memref<?x?x128xf32>
              %39 = loom.init_tensor %38[%12, %13, 128] : memref<?x?x128xf32> -> tensor<?x?x128xf32>
              %40 = loom.alloc [%12, %13, 128] on @L1 : memref<?x?x128xf32>
              %41 = loom.semaphore %40 : memref<?x?x128xf32> -> memref<?x?x128xf32>
              %42 = loom.init_tensor %41[%12, %13, 128] : memref<?x?x128xf32> -> tensor<?x?x128xf32>
              %43 = loom.alloc [%12, %13, 128] on @L1 : memref<?x?x128xf32>
              %44 = loom.semaphore %43 : memref<?x?x128xf32> -> memref<?x?x128xf32>
              %45 = loom.init_tensor %44[%12, %13, 128] : memref<?x?x128xf32> -> tensor<?x?x128xf32>
              %46 = arith.muli %15, %12 : index
              %47 = arith.muli %16, %13 : index
              %48 = loom.subview %arg2[%46, %47, 0] [%12, %13, 128] [1, 1, 1], reuse : [seq = false, spat = false, temp = false] : memref<32x4096x128xf32> to memref<?x?x128xf32, strided<[524288, 128, 1], offset: ?>>
              %49 = loom.semaphore %37 : memref<?x?x128xf32> -> memref<?x?x128xf32>
              %50 = loom.copy_to_tensor %48, %49 on @L1, interconnect : [], broadcast : [1, 1] : memref<?x?x128xf32, strided<[524288, 128, 1], offset: ?>>, memref<?x?x128xf32> -> tensor<?x?x128xf32>
              %51 = linalg.fill ins(%cst_1 : f32) outs(%42 : tensor<?x?x128xf32>) -> tensor<?x?x128xf32>
              %52 = linalg.fill ins(%cst_2 : f32) outs(%24 : tensor<?x?xf32>) -> tensor<?x?xf32>
              %53 = linalg.fill ins(%cst_3 : f32) outs(%27 : tensor<?x?xf32>) -> tensor<?x?xf32>
              %54:3 = affine.for %arg8 = 0 to affine_map<()[s0] -> (4096 ceildiv s0)>()[%14] iter_args(%arg9 = %53, %arg10 = %52, %arg11 = %51) -> (tensor<?x?xf32>, tensor<?x?xf32>, tensor<?x?x128xf32>) {
                %57 = arith.muli %arg8, %14 : index
                %58 = loom.subview %arg0[%46, 0, %57] [%12, 128, %14] [1, 1, 1], reuse : [seq = false, spat = false, temp = false] : memref<32x128x4096xf32> to memref<?x128x?xf32, strided<[524288, 4096, 1], offset: ?>>
                %59 = loom.semaphore %21 : memref<?x128x?xf32> -> memref<?x128x?xf32>
                %60 = loom.copy_to_tensor %58, %59 on @L1, interconnect : [], broadcast : [1, 1] : memref<?x128x?xf32, strided<[524288, 4096, 1], offset: ?>>, memref<?x128x?xf32> -> tensor<?x128x?xf32>
                %61 = linalg.fill ins(%cst_1 : f32) outs(%20 : tensor<?x?x?xf32>) -> tensor<?x?x?xf32>
                %62 = linalg.batch_matmul ins(%50, %60 : tensor<?x?x128xf32>, tensor<?x128x?xf32>) outs(%61 : tensor<?x?x?xf32>) -> tensor<?x?x?xf32>
                %63 = linalg.fill ins(%cst_3 : f32) outs(%30 : tensor<?x?xf32>) -> tensor<?x?xf32>
                %64 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%62 : tensor<?x?x?xf32>) outs(%63 : tensor<?x?xf32>) {
                ^bb0(%in: f32, %out: f32):
                  %78 = arith.maximumf %in, %out : f32
                  linalg.yield %78 : f32
                } -> tensor<?x?xf32>
                %65 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%arg9, %64 : tensor<?x?xf32>, tensor<?x?xf32>) outs(%30 : tensor<?x?xf32>) {
                ^bb0(%in: f32, %in_4: f32, %out: f32):
                  %78 = arith.truncf %cst_0 : f64 to f32
                  %79 = arith.mulf %in_4, %78 : f32
                  %80 = arith.cmpf ogt, %in, %79 : f32
                  %81 = arith.select %80, %in, %79 : f32
                  linalg.yield %81 : f32
                } -> tensor<?x?xf32>
                %66 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%62, %65 : tensor<?x?x?xf32>, tensor<?x?xf32>) outs(%20 : tensor<?x?x?xf32>) {
                ^bb0(%in: f32, %in_4: f32, %out: f32):
                  %78 = arith.truncf %cst_0 : f64 to f32
                  %79 = arith.mulf %in, %78 : f32
                  %80 = arith.subf %79, %in_4 : f32
                  %81 = math.powf %cst, %80 : f32
                  linalg.yield %81 : f32
                } -> tensor<?x?x?xf32>
                %67 = linalg.fill ins(%cst_1 : f32) outs(%33 : tensor<?x?xf32>) -> tensor<?x?xf32>
                %68 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%66 : tensor<?x?x?xf32>) outs(%67 : tensor<?x?xf32>) {
                ^bb0(%in: f32, %out: f32):
                  %78 = arith.addf %in, %out : f32
                  linalg.yield %78 : f32
                } -> tensor<?x?xf32>
                %69 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%arg9, %65 : tensor<?x?xf32>, tensor<?x?xf32>) outs(%36 : tensor<?x?xf32>) {
                ^bb0(%in: f32, %in_4: f32, %out: f32):
                  %78 = arith.subf %in, %in_4 : f32
                  %79 = math.powf %cst, %78 : f32
                  linalg.yield %79 : f32
                } -> tensor<?x?xf32>
                %70 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%arg10, %69, %68 : tensor<?x?xf32>, tensor<?x?xf32>, tensor<?x?xf32>) outs(%arg10 : tensor<?x?xf32>) {
                ^bb0(%in: f32, %in_4: f32, %in_5: f32, %out: f32):
                  %78 = arith.mulf %in, %in_4 : f32
                  %79 = arith.addf %78, %in_5 : f32
                  linalg.yield %79 : f32
                } -> tensor<?x?xf32>
                %71 = loom.subview %arg1[%46, %57, 0] [%12, %14, 128] [1, 1, 1], reuse : [seq = false, spat = false, temp = false] : memref<32x4096x128xf32> to memref<?x?x128xf32, strided<[524288, 128, 1], offset: ?>>
                %72 = loom.semaphore %17 : memref<?x?x128xf32> -> memref<?x?x128xf32>
                %73 = loom.copy_to_tensor %71, %72 on @L1, interconnect : [], broadcast : [1, 1] : memref<?x?x128xf32, strided<[524288, 128, 1], offset: ?>>, memref<?x?x128xf32> -> tensor<?x?x128xf32>
                %74 = linalg.fill ins(%cst_1 : f32) outs(%45 : tensor<?x?x128xf32>) -> tensor<?x?x128xf32>
                %75 = linalg.batch_matmul ins(%66, %73 : tensor<?x?x?xf32>, tensor<?x?x128xf32>) outs(%74 : tensor<?x?x128xf32>) -> tensor<?x?x128xf32>
                %76 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%75, %arg11, %69 : tensor<?x?x128xf32>, tensor<?x?x128xf32>, tensor<?x?xf32>) outs(%arg11 : tensor<?x?x128xf32>) {
                ^bb0(%in: f32, %in_4: f32, %in_5: f32, %out: f32):
                  %78 = arith.mulf %in_4, %in_5 : f32
                  %79 = arith.addf %in, %78 : f32
                  linalg.yield %79 : f32
                } -> tensor<?x?x128xf32>
                %77 = linalg.copy ins(%65 : tensor<?x?xf32>) outs(%arg9 : tensor<?x?xf32>) -> tensor<?x?xf32>
                affine.yield %77, %70, %76 : tensor<?x?xf32>, tensor<?x?xf32>, tensor<?x?x128xf32>
              } {loom.iter_type = #loom.iter_type<sequential>}
              %55 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%54#2, %54#1 : tensor<?x?x128xf32>, tensor<?x?xf32>) outs(%39 : tensor<?x?x128xf32>) {
              ^bb0(%in: f32, %in_4: f32, %out: f32):
                %57 = arith.divf %in, %in_4 : f32
                linalg.yield %57 : f32
              } -> tensor<?x?x128xf32>
              %56 = loom.subview %arg3[%46, %47, 0] [%12, %13, 128] [1, 1, 1], reuse : [seq = false, spat = false, temp = false] : memref<32x4096x128xf32> to memref<?x?x128xf32, strided<[524288, 128, 1], offset: ?>>
              loom.copy_from_tensor %55, %56 on @DRAM : tensor<?x?x128xf32>, memref<?x?x128xf32, strided<[524288, 128, 1], offset: ?>>
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
              %19 = loom.semaphore %18 : memref<?x?x?xf32> -> memref<?x?x?xf32>
              %20 = loom.init_tensor %19[%12, %13, %14] : memref<?x?x?xf32> -> tensor<?x?x?xf32>
              %21 = loom.alloc [%12, 128, %14] on @L1 : memref<?x128x?xf32>
              %22 = loom.alloc [%12, %13] on @L1 : memref<?x?xf32>
              %23 = loom.semaphore %22 : memref<?x?xf32> -> memref<?x?xf32>
              %24 = loom.init_tensor %23[%12, %13] : memref<?x?xf32> -> tensor<?x?xf32>
              %25 = loom.alloc [%12, %13] on @L1 : memref<?x?xf32>
              %26 = loom.semaphore %25 : memref<?x?xf32> -> memref<?x?xf32>
              %27 = loom.init_tensor %26[%12, %13] : memref<?x?xf32> -> tensor<?x?xf32>
              %28 = loom.alloc [%12, %13] on @L1 : memref<?x?xf32>
              %29 = loom.semaphore %28 : memref<?x?xf32> -> memref<?x?xf32>
              %30 = loom.init_tensor %29[%12, %13] : memref<?x?xf32> -> tensor<?x?xf32>
              %31 = loom.alloc [%12, %13] on @L1 : memref<?x?xf32>
              %32 = loom.semaphore %31 : memref<?x?xf32> -> memref<?x?xf32>
              %33 = loom.init_tensor %32[%12, %13] : memref<?x?xf32> -> tensor<?x?xf32>
              %34 = loom.alloc [%12, %13] on @L1 : memref<?x?xf32>
              %35 = loom.semaphore %34 : memref<?x?xf32> -> memref<?x?xf32>
              %36 = loom.init_tensor %35[%12, %13] : memref<?x?xf32> -> tensor<?x?xf32>
              %37 = loom.alloc [%12, %13, 128] on @L1 : memref<?x?x128xf32>
              %38 = loom.semaphore %37 : memref<?x?x128xf32> -> memref<?x?x128xf32>
              %39 = loom.init_tensor %38[%12, %13, 128] : memref<?x?x128xf32> -> tensor<?x?x128xf32>
              %40 = loom.alloc [%12, %13, 128] on @L1 : memref<?x?x128xf32>
              %41 = loom.semaphore %40 : memref<?x?x128xf32> -> memref<?x?x128xf32>
              %42 = loom.init_tensor %41[%12, %13, 128] : memref<?x?x128xf32> -> tensor<?x?x128xf32>
              %43 = loom.alloc [%12, %13, 128] on @L1 : memref<?x?x128xf32>
              %44 = loom.semaphore %43 : memref<?x?x128xf32> -> memref<?x?x128xf32>
              %45 = loom.init_tensor %44[%12, %13, 128] : memref<?x?x128xf32> -> tensor<?x?x128xf32>
              %46 = arith.muli %15, %12 : index
              %47 = arith.muli %16, %13 : index
              %48 = loom.subview %arg2[%46, %47, 0] [%12, %13, 128] [1, 1, 1], reuse : [seq = false, spat = false, temp = false] : memref<32x4096x128xf32> to memref<?x?x128xf32, strided<[524288, 128, 1], offset: ?>>
              %49 = loom.semaphore %37 : memref<?x?x128xf32> -> memref<?x?x128xf32>
              %50 = loom.copy_to_tensor %48, %49 on @L1, interconnect : [], broadcast : [1, 1] : memref<?x?x128xf32, strided<[524288, 128, 1], offset: ?>>, memref<?x?x128xf32> -> tensor<?x?x128xf32>
              %51 = linalg.fill ins(%cst_1 : f32) outs(%42 : tensor<?x?x128xf32>) -> tensor<?x?x128xf32>
              %52 = linalg.fill ins(%cst_2 : f32) outs(%24 : tensor<?x?xf32>) -> tensor<?x?xf32>
              %53 = linalg.fill ins(%cst_3 : f32) outs(%27 : tensor<?x?xf32>) -> tensor<?x?xf32>
              %54:3 = affine.for %arg8 = 0 to affine_map<()[s0] -> (4096 ceildiv s0)>()[%14] iter_args(%arg9 = %53, %arg10 = %52, %arg11 = %51) -> (tensor<?x?xf32>, tensor<?x?xf32>, tensor<?x?x128xf32>) {
                %57 = arith.muli %arg8, %14 : index
                %58 = loom.subview %arg0[%46, 0, %57] [%12, 128, %14] [1, 1, 1], reuse : [seq = false, spat = false, temp = false] : memref<32x128x4096xf32> to memref<?x128x?xf32, strided<[524288, 4096, 1], offset: ?>>
                %59 = loom.semaphore %21 : memref<?x128x?xf32> -> memref<?x128x?xf32>
                %60 = loom.copy_to_tensor %58, %59 on @L1, interconnect : [], broadcast : [1, 1] : memref<?x128x?xf32, strided<[524288, 4096, 1], offset: ?>>, memref<?x128x?xf32> -> tensor<?x128x?xf32>
                %61 = linalg.fill ins(%cst_1 : f32) outs(%20 : tensor<?x?x?xf32>) -> tensor<?x?x?xf32>
                %62 = linalg.batch_matmul ins(%50, %60 : tensor<?x?x128xf32>, tensor<?x128x?xf32>) outs(%61 : tensor<?x?x?xf32>) -> tensor<?x?x?xf32>
                %63 = linalg.fill ins(%cst_3 : f32) outs(%30 : tensor<?x?xf32>) -> tensor<?x?xf32>
                %64 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%62 : tensor<?x?x?xf32>) outs(%63 : tensor<?x?xf32>) {
                ^bb0(%in: f32, %out: f32):
                  %78 = arith.maximumf %in, %out : f32
                  linalg.yield %78 : f32
                } -> tensor<?x?xf32>
                %65 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%arg9, %64 : tensor<?x?xf32>, tensor<?x?xf32>) outs(%30 : tensor<?x?xf32>) {
                ^bb0(%in: f32, %in_4: f32, %out: f32):
                  %78 = arith.truncf %cst_0 : f64 to f32
                  %79 = arith.mulf %in_4, %78 : f32
                  %80 = arith.cmpf ogt, %in, %79 : f32
                  %81 = arith.select %80, %in, %79 : f32
                  linalg.yield %81 : f32
                } -> tensor<?x?xf32>
                %66 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%62, %65 : tensor<?x?x?xf32>, tensor<?x?xf32>) outs(%20 : tensor<?x?x?xf32>) {
                ^bb0(%in: f32, %in_4: f32, %out: f32):
                  %78 = arith.truncf %cst_0 : f64 to f32
                  %79 = arith.mulf %in, %78 : f32
                  %80 = arith.subf %79, %in_4 : f32
                  %81 = math.powf %cst, %80 : f32
                  linalg.yield %81 : f32
                } -> tensor<?x?x?xf32>
                %67 = linalg.fill ins(%cst_1 : f32) outs(%33 : tensor<?x?xf32>) -> tensor<?x?xf32>
                %68 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%66 : tensor<?x?x?xf32>) outs(%67 : tensor<?x?xf32>) {
                ^bb0(%in: f32, %out: f32):
                  %78 = arith.addf %in, %out : f32
                  linalg.yield %78 : f32
                } -> tensor<?x?xf32>
                %69 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%arg9, %65 : tensor<?x?xf32>, tensor<?x?xf32>) outs(%36 : tensor<?x?xf32>) {
                ^bb0(%in: f32, %in_4: f32, %out: f32):
                  %78 = arith.subf %in, %in_4 : f32
                  %79 = math.powf %cst, %78 : f32
                  linalg.yield %79 : f32
                } -> tensor<?x?xf32>
                %70 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%arg10, %69, %68 : tensor<?x?xf32>, tensor<?x?xf32>, tensor<?x?xf32>) outs(%arg10 : tensor<?x?xf32>) {
                ^bb0(%in: f32, %in_4: f32, %in_5: f32, %out: f32):
                  %78 = arith.mulf %in, %in_4 : f32
                  %79 = arith.addf %78, %in_5 : f32
                  linalg.yield %79 : f32
                } -> tensor<?x?xf32>
                %71 = loom.subview %arg1[%46, %57, 0] [%12, %14, 128] [1, 1, 1], reuse : [seq = false, spat = false, temp = false] : memref<32x4096x128xf32> to memref<?x?x128xf32, strided<[524288, 128, 1], offset: ?>>
                %72 = loom.semaphore %17 : memref<?x?x128xf32> -> memref<?x?x128xf32>
                %73 = loom.copy_to_tensor %71, %72 on @L1, interconnect : [], broadcast : [1, 1] : memref<?x?x128xf32, strided<[524288, 128, 1], offset: ?>>, memref<?x?x128xf32> -> tensor<?x?x128xf32>
                %74 = linalg.fill ins(%cst_1 : f32) outs(%45 : tensor<?x?x128xf32>) -> tensor<?x?x128xf32>
                %75 = linalg.batch_matmul ins(%66, %73 : tensor<?x?x?xf32>, tensor<?x?x128xf32>) outs(%74 : tensor<?x?x128xf32>) -> tensor<?x?x128xf32>
                %76 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%75, %arg11, %69 : tensor<?x?x128xf32>, tensor<?x?x128xf32>, tensor<?x?xf32>) outs(%arg11 : tensor<?x?x128xf32>) {
                ^bb0(%in: f32, %in_4: f32, %in_5: f32, %out: f32):
                  %78 = arith.mulf %in_4, %in_5 : f32
                  %79 = arith.addf %in, %78 : f32
                  linalg.yield %79 : f32
                } -> tensor<?x?x128xf32>
                %77 = linalg.copy ins(%65 : tensor<?x?xf32>) outs(%arg9 : tensor<?x?xf32>) -> tensor<?x?xf32>
                affine.yield %77, %70, %76 : tensor<?x?xf32>, tensor<?x?xf32>, tensor<?x?x128xf32>
              } {loom.iter_type = #loom.iter_type<sequential>}
              %55 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%54#2, %54#1 : tensor<?x?x128xf32>, tensor<?x?xf32>) outs(%39 : tensor<?x?x128xf32>) {
              ^bb0(%in: f32, %in_4: f32, %out: f32):
                %57 = arith.divf %in, %in_4 : f32
                linalg.yield %57 : f32
              } -> tensor<?x?x128xf32>
              %56 = loom.subview %arg3[%46, %47, 0] [%12, %13, 128] [1, 1, 1], reuse : [seq = false, spat = false, temp = false] : memref<32x4096x128xf32> to memref<?x?x128xf32, strided<[524288, 128, 1], offset: ?>>
              loom.copy_from_tensor %55, %56 on @DRAM : tensor<?x?x128xf32>, memref<?x?x128xf32, strided<[524288, 128, 1], offset: ?>>
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
              %18 = loom.semaphore %17 : memref<?x?x?xf32> -> memref<?x?x?xf32>
              %19 = loom.init_tensor %18[%12, %13, %14] : memref<?x?x?xf32> -> tensor<?x?x?xf32>
              %20 = loom.alloc [%12, 128, %14] on @L1 : memref<?x128x?xf32>
              %21 = loom.alloc [%12, %13] on @L1 : memref<?x?xf32>
              %22 = loom.semaphore %21 : memref<?x?xf32> -> memref<?x?xf32>
              %23 = loom.init_tensor %22[%12, %13] : memref<?x?xf32> -> tensor<?x?xf32>
              %24 = loom.alloc [%12, %13] on @L1 : memref<?x?xf32>
              %25 = loom.semaphore %24 : memref<?x?xf32> -> memref<?x?xf32>
              %26 = loom.init_tensor %25[%12, %13] : memref<?x?xf32> -> tensor<?x?xf32>
              %27 = loom.alloc [%12, %13] on @L1 : memref<?x?xf32>
              %28 = loom.semaphore %27 : memref<?x?xf32> -> memref<?x?xf32>
              %29 = loom.init_tensor %28[%12, %13] : memref<?x?xf32> -> tensor<?x?xf32>
              %30 = loom.alloc [%12, %13] on @L1 : memref<?x?xf32>
              %31 = loom.semaphore %30 : memref<?x?xf32> -> memref<?x?xf32>
              %32 = loom.init_tensor %31[%12, %13] : memref<?x?xf32> -> tensor<?x?xf32>
              %33 = loom.alloc [%12, %13] on @L1 : memref<?x?xf32>
              %34 = loom.semaphore %33 : memref<?x?xf32> -> memref<?x?xf32>
              %35 = loom.init_tensor %34[%12, %13] : memref<?x?xf32> -> tensor<?x?xf32>
              %36 = loom.alloc [%12, %13, 128] on @L1 : memref<?x?x128xf32>
              %37 = loom.semaphore %36 : memref<?x?x128xf32> -> memref<?x?x128xf32>
              %38 = loom.init_tensor %37[%12, %13, 128] : memref<?x?x128xf32> -> tensor<?x?x128xf32>
              %39 = loom.alloc [%12, %13, 128] on @L1 : memref<?x?x128xf32>
              %40 = loom.semaphore %39 : memref<?x?x128xf32> -> memref<?x?x128xf32>
              %41 = loom.init_tensor %40[%12, %13, 128] : memref<?x?x128xf32> -> tensor<?x?x128xf32>
              %42 = loom.alloc [%12, %13, 128] on @L1 : memref<?x?x128xf32>
              %43 = loom.semaphore %42 : memref<?x?x128xf32> -> memref<?x?x128xf32>
              %44 = loom.init_tensor %43[%12, %13, 128] : memref<?x?x128xf32> -> tensor<?x?x128xf32>
              %45 = arith.muli %arg6, %12 : index
              %46 = arith.muli %15, %13 : index
              %47 = loom.subview %arg2[%45, %46, 0] [%12, %13, 128] [1, 1, 1], reuse : [seq = false, spat = false, temp = false] : memref<32x4096x128xf32> to memref<?x?x128xf32, strided<[524288, 128, 1], offset: ?>>
              %48 = loom.semaphore %36 : memref<?x?x128xf32> -> memref<?x?x128xf32>
              %49 = loom.copy_to_tensor %47, %48 on @L1, interconnect : [], broadcast : [1, 1] : memref<?x?x128xf32, strided<[524288, 128, 1], offset: ?>>, memref<?x?x128xf32> -> tensor<?x?x128xf32>
              %50 = linalg.fill ins(%cst_1 : f32) outs(%41 : tensor<?x?x128xf32>) -> tensor<?x?x128xf32>
              %51 = linalg.fill ins(%cst_2 : f32) outs(%23 : tensor<?x?xf32>) -> tensor<?x?xf32>
              %52 = linalg.fill ins(%cst_3 : f32) outs(%26 : tensor<?x?xf32>) -> tensor<?x?xf32>
              %53:3 = affine.for %arg8 = 0 to affine_map<()[s0] -> (4096 ceildiv s0)>()[%14] iter_args(%arg9 = %52, %arg10 = %51, %arg11 = %50) -> (tensor<?x?xf32>, tensor<?x?xf32>, tensor<?x?x128xf32>) {
                %56 = arith.muli %arg8, %14 : index
                %57 = loom.subview %arg0[%45, 0, %56] [%12, 128, %14] [1, 1, 1], reuse : [seq = false, spat = false, temp = false] : memref<32x128x4096xf32> to memref<?x128x?xf32, strided<[524288, 4096, 1], offset: ?>>
                %58 = loom.semaphore %20 : memref<?x128x?xf32> -> memref<?x128x?xf32>
                %59 = loom.copy_to_tensor %57, %58 on @L1, interconnect : [], broadcast : [1, 1] : memref<?x128x?xf32, strided<[524288, 4096, 1], offset: ?>>, memref<?x128x?xf32> -> tensor<?x128x?xf32>
                %60 = linalg.fill ins(%cst_1 : f32) outs(%19 : tensor<?x?x?xf32>) -> tensor<?x?x?xf32>
                %61 = linalg.batch_matmul ins(%49, %59 : tensor<?x?x128xf32>, tensor<?x128x?xf32>) outs(%60 : tensor<?x?x?xf32>) -> tensor<?x?x?xf32>
                %62 = linalg.fill ins(%cst_3 : f32) outs(%29 : tensor<?x?xf32>) -> tensor<?x?xf32>
                %63 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%61 : tensor<?x?x?xf32>) outs(%62 : tensor<?x?xf32>) {
                ^bb0(%in: f32, %out: f32):
                  %77 = arith.maximumf %in, %out : f32
                  linalg.yield %77 : f32
                } -> tensor<?x?xf32>
                %64 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%arg9, %63 : tensor<?x?xf32>, tensor<?x?xf32>) outs(%29 : tensor<?x?xf32>) {
                ^bb0(%in: f32, %in_4: f32, %out: f32):
                  %77 = arith.truncf %cst_0 : f64 to f32
                  %78 = arith.mulf %in_4, %77 : f32
                  %79 = arith.cmpf ogt, %in, %78 : f32
                  %80 = arith.select %79, %in, %78 : f32
                  linalg.yield %80 : f32
                } -> tensor<?x?xf32>
                %65 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%61, %64 : tensor<?x?x?xf32>, tensor<?x?xf32>) outs(%19 : tensor<?x?x?xf32>) {
                ^bb0(%in: f32, %in_4: f32, %out: f32):
                  %77 = arith.truncf %cst_0 : f64 to f32
                  %78 = arith.mulf %in, %77 : f32
                  %79 = arith.subf %78, %in_4 : f32
                  %80 = math.powf %cst, %79 : f32
                  linalg.yield %80 : f32
                } -> tensor<?x?x?xf32>
                %66 = linalg.fill ins(%cst_1 : f32) outs(%32 : tensor<?x?xf32>) -> tensor<?x?xf32>
                %67 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%65 : tensor<?x?x?xf32>) outs(%66 : tensor<?x?xf32>) {
                ^bb0(%in: f32, %out: f32):
                  %77 = arith.addf %in, %out : f32
                  linalg.yield %77 : f32
                } -> tensor<?x?xf32>
                %68 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%arg9, %64 : tensor<?x?xf32>, tensor<?x?xf32>) outs(%35 : tensor<?x?xf32>) {
                ^bb0(%in: f32, %in_4: f32, %out: f32):
                  %77 = arith.subf %in, %in_4 : f32
                  %78 = math.powf %cst, %77 : f32
                  linalg.yield %78 : f32
                } -> tensor<?x?xf32>
                %69 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%arg10, %68, %67 : tensor<?x?xf32>, tensor<?x?xf32>, tensor<?x?xf32>) outs(%arg10 : tensor<?x?xf32>) {
                ^bb0(%in: f32, %in_4: f32, %in_5: f32, %out: f32):
                  %77 = arith.mulf %in, %in_4 : f32
                  %78 = arith.addf %77, %in_5 : f32
                  linalg.yield %78 : f32
                } -> tensor<?x?xf32>
                %70 = loom.subview %arg1[%45, %56, 0] [%12, %14, 128] [1, 1, 1], reuse : [seq = false, spat = false, temp = false] : memref<32x4096x128xf32> to memref<?x?x128xf32, strided<[524288, 128, 1], offset: ?>>
                %71 = loom.semaphore %16 : memref<?x?x128xf32> -> memref<?x?x128xf32>
                %72 = loom.copy_to_tensor %70, %71 on @L1, interconnect : [], broadcast : [1, 1] : memref<?x?x128xf32, strided<[524288, 128, 1], offset: ?>>, memref<?x?x128xf32> -> tensor<?x?x128xf32>
                %73 = linalg.fill ins(%cst_1 : f32) outs(%44 : tensor<?x?x128xf32>) -> tensor<?x?x128xf32>
                %74 = linalg.batch_matmul ins(%65, %72 : tensor<?x?x?xf32>, tensor<?x?x128xf32>) outs(%73 : tensor<?x?x128xf32>) -> tensor<?x?x128xf32>
                %75 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%74, %arg11, %68 : tensor<?x?x128xf32>, tensor<?x?x128xf32>, tensor<?x?xf32>) outs(%arg11 : tensor<?x?x128xf32>) {
                ^bb0(%in: f32, %in_4: f32, %in_5: f32, %out: f32):
                  %77 = arith.mulf %in_4, %in_5 : f32
                  %78 = arith.addf %in, %77 : f32
                  linalg.yield %78 : f32
                } -> tensor<?x?x128xf32>
                %76 = linalg.copy ins(%64 : tensor<?x?xf32>) outs(%arg9 : tensor<?x?xf32>) -> tensor<?x?xf32>
                affine.yield %76, %69, %75 : tensor<?x?xf32>, tensor<?x?xf32>, tensor<?x?x128xf32>
              } {loom.iter_type = #loom.iter_type<sequential>}
              %54 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%53#2, %53#1 : tensor<?x?x128xf32>, tensor<?x?xf32>) outs(%38 : tensor<?x?x128xf32>) {
              ^bb0(%in: f32, %in_4: f32, %out: f32):
                %56 = arith.divf %in, %in_4 : f32
                linalg.yield %56 : f32
              } -> tensor<?x?x128xf32>
              %55 = loom.subview %arg3[%45, %46, 0] [%12, %13, 128] [1, 1, 1], reuse : [seq = false, spat = false, temp = false] : memref<32x4096x128xf32> to memref<?x?x128xf32, strided<[524288, 128, 1], offset: ?>>
              loom.copy_from_tensor %54, %55 on @DRAM : tensor<?x?x128xf32>, memref<?x?x128xf32, strided<[524288, 128, 1], offset: ?>>
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
              %18 = loom.semaphore %17 : memref<?x?x?xf32> -> memref<?x?x?xf32>
              %19 = loom.init_tensor %18[%12, %13, %14] : memref<?x?x?xf32> -> tensor<?x?x?xf32>
              %20 = loom.alloc [%12, 128, %14] on @L1 : memref<?x128x?xf32>
              %21 = loom.alloc [%12, %13] on @L1 : memref<?x?xf32>
              %22 = loom.semaphore %21 : memref<?x?xf32> -> memref<?x?xf32>
              %23 = loom.init_tensor %22[%12, %13] : memref<?x?xf32> -> tensor<?x?xf32>
              %24 = loom.alloc [%12, %13] on @L1 : memref<?x?xf32>
              %25 = loom.semaphore %24 : memref<?x?xf32> -> memref<?x?xf32>
              %26 = loom.init_tensor %25[%12, %13] : memref<?x?xf32> -> tensor<?x?xf32>
              %27 = loom.alloc [%12, %13] on @L1 : memref<?x?xf32>
              %28 = loom.semaphore %27 : memref<?x?xf32> -> memref<?x?xf32>
              %29 = loom.init_tensor %28[%12, %13] : memref<?x?xf32> -> tensor<?x?xf32>
              %30 = loom.alloc [%12, %13] on @L1 : memref<?x?xf32>
              %31 = loom.semaphore %30 : memref<?x?xf32> -> memref<?x?xf32>
              %32 = loom.init_tensor %31[%12, %13] : memref<?x?xf32> -> tensor<?x?xf32>
              %33 = loom.alloc [%12, %13] on @L1 : memref<?x?xf32>
              %34 = loom.semaphore %33 : memref<?x?xf32> -> memref<?x?xf32>
              %35 = loom.init_tensor %34[%12, %13] : memref<?x?xf32> -> tensor<?x?xf32>
              %36 = loom.alloc [%12, %13, 128] on @L1 : memref<?x?x128xf32>
              %37 = loom.semaphore %36 : memref<?x?x128xf32> -> memref<?x?x128xf32>
              %38 = loom.init_tensor %37[%12, %13, 128] : memref<?x?x128xf32> -> tensor<?x?x128xf32>
              %39 = loom.alloc [%12, %13, 128] on @L1 : memref<?x?x128xf32>
              %40 = loom.semaphore %39 : memref<?x?x128xf32> -> memref<?x?x128xf32>
              %41 = loom.init_tensor %40[%12, %13, 128] : memref<?x?x128xf32> -> tensor<?x?x128xf32>
              %42 = loom.alloc [%12, %13, 128] on @L1 : memref<?x?x128xf32>
              %43 = loom.semaphore %42 : memref<?x?x128xf32> -> memref<?x?x128xf32>
              %44 = loom.init_tensor %43[%12, %13, 128] : memref<?x?x128xf32> -> tensor<?x?x128xf32>
              %45 = arith.muli %arg7, %12 : index
              %46 = arith.muli %15, %13 : index
              %47 = loom.subview %arg2[%45, %46, 0] [%12, %13, 128] [1, 1, 1], reuse : [seq = false, spat = false, temp = false] : memref<32x4096x128xf32> to memref<?x?x128xf32, strided<[524288, 128, 1], offset: ?>>
              %48 = loom.semaphore %36 : memref<?x?x128xf32> -> memref<?x?x128xf32>
              %49 = loom.copy_to_tensor %47, %48 on @L1, interconnect : [], broadcast : [1, 1] : memref<?x?x128xf32, strided<[524288, 128, 1], offset: ?>>, memref<?x?x128xf32> -> tensor<?x?x128xf32>
              %50 = linalg.fill ins(%cst_1 : f32) outs(%41 : tensor<?x?x128xf32>) -> tensor<?x?x128xf32>
              %51 = linalg.fill ins(%cst_2 : f32) outs(%23 : tensor<?x?xf32>) -> tensor<?x?xf32>
              %52 = linalg.fill ins(%cst_3 : f32) outs(%26 : tensor<?x?xf32>) -> tensor<?x?xf32>
              %53:3 = affine.for %arg8 = 0 to affine_map<()[s0] -> (4096 ceildiv s0)>()[%14] iter_args(%arg9 = %52, %arg10 = %51, %arg11 = %50) -> (tensor<?x?xf32>, tensor<?x?xf32>, tensor<?x?x128xf32>) {
                %56 = arith.muli %arg8, %14 : index
                %57 = loom.subview %arg0[%45, 0, %56] [%12, 128, %14] [1, 1, 1], reuse : [seq = false, spat = false, temp = false] : memref<32x128x4096xf32> to memref<?x128x?xf32, strided<[524288, 4096, 1], offset: ?>>
                %58 = loom.semaphore %20 : memref<?x128x?xf32> -> memref<?x128x?xf32>
                %59 = loom.copy_to_tensor %57, %58 on @L1, interconnect : [], broadcast : [1, 1] : memref<?x128x?xf32, strided<[524288, 4096, 1], offset: ?>>, memref<?x128x?xf32> -> tensor<?x128x?xf32>
                %60 = linalg.fill ins(%cst_1 : f32) outs(%19 : tensor<?x?x?xf32>) -> tensor<?x?x?xf32>
                %61 = linalg.batch_matmul ins(%49, %59 : tensor<?x?x128xf32>, tensor<?x128x?xf32>) outs(%60 : tensor<?x?x?xf32>) -> tensor<?x?x?xf32>
                %62 = linalg.fill ins(%cst_3 : f32) outs(%29 : tensor<?x?xf32>) -> tensor<?x?xf32>
                %63 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%61 : tensor<?x?x?xf32>) outs(%62 : tensor<?x?xf32>) {
                ^bb0(%in: f32, %out: f32):
                  %77 = arith.maximumf %in, %out : f32
                  linalg.yield %77 : f32
                } -> tensor<?x?xf32>
                %64 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%arg9, %63 : tensor<?x?xf32>, tensor<?x?xf32>) outs(%29 : tensor<?x?xf32>) {
                ^bb0(%in: f32, %in_4: f32, %out: f32):
                  %77 = arith.truncf %cst_0 : f64 to f32
                  %78 = arith.mulf %in_4, %77 : f32
                  %79 = arith.cmpf ogt, %in, %78 : f32
                  %80 = arith.select %79, %in, %78 : f32
                  linalg.yield %80 : f32
                } -> tensor<?x?xf32>
                %65 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%61, %64 : tensor<?x?x?xf32>, tensor<?x?xf32>) outs(%19 : tensor<?x?x?xf32>) {
                ^bb0(%in: f32, %in_4: f32, %out: f32):
                  %77 = arith.truncf %cst_0 : f64 to f32
                  %78 = arith.mulf %in, %77 : f32
                  %79 = arith.subf %78, %in_4 : f32
                  %80 = math.powf %cst, %79 : f32
                  linalg.yield %80 : f32
                } -> tensor<?x?x?xf32>
                %66 = linalg.fill ins(%cst_1 : f32) outs(%32 : tensor<?x?xf32>) -> tensor<?x?xf32>
                %67 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%65 : tensor<?x?x?xf32>) outs(%66 : tensor<?x?xf32>) {
                ^bb0(%in: f32, %out: f32):
                  %77 = arith.addf %in, %out : f32
                  linalg.yield %77 : f32
                } -> tensor<?x?xf32>
                %68 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%arg9, %64 : tensor<?x?xf32>, tensor<?x?xf32>) outs(%35 : tensor<?x?xf32>) {
                ^bb0(%in: f32, %in_4: f32, %out: f32):
                  %77 = arith.subf %in, %in_4 : f32
                  %78 = math.powf %cst, %77 : f32
                  linalg.yield %78 : f32
                } -> tensor<?x?xf32>
                %69 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%arg10, %68, %67 : tensor<?x?xf32>, tensor<?x?xf32>, tensor<?x?xf32>) outs(%arg10 : tensor<?x?xf32>) {
                ^bb0(%in: f32, %in_4: f32, %in_5: f32, %out: f32):
                  %77 = arith.mulf %in, %in_4 : f32
                  %78 = arith.addf %77, %in_5 : f32
                  linalg.yield %78 : f32
                } -> tensor<?x?xf32>
                %70 = loom.subview %arg1[%45, %56, 0] [%12, %14, 128] [1, 1, 1], reuse : [seq = false, spat = false, temp = false] : memref<32x4096x128xf32> to memref<?x?x128xf32, strided<[524288, 128, 1], offset: ?>>
                %71 = loom.semaphore %16 : memref<?x?x128xf32> -> memref<?x?x128xf32>
                %72 = loom.copy_to_tensor %70, %71 on @L1, interconnect : [], broadcast : [1, 1] : memref<?x?x128xf32, strided<[524288, 128, 1], offset: ?>>, memref<?x?x128xf32> -> tensor<?x?x128xf32>
                %73 = linalg.fill ins(%cst_1 : f32) outs(%44 : tensor<?x?x128xf32>) -> tensor<?x?x128xf32>
                %74 = linalg.batch_matmul ins(%65, %72 : tensor<?x?x?xf32>, tensor<?x?x128xf32>) outs(%73 : tensor<?x?x128xf32>) -> tensor<?x?x128xf32>
                %75 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%74, %arg11, %68 : tensor<?x?x128xf32>, tensor<?x?x128xf32>, tensor<?x?xf32>) outs(%arg11 : tensor<?x?x128xf32>) {
                ^bb0(%in: f32, %in_4: f32, %in_5: f32, %out: f32):
                  %77 = arith.mulf %in_4, %in_5 : f32
                  %78 = arith.addf %in, %77 : f32
                  linalg.yield %78 : f32
                } -> tensor<?x?x128xf32>
                %76 = linalg.copy ins(%64 : tensor<?x?xf32>) outs(%arg9 : tensor<?x?xf32>) -> tensor<?x?xf32>
                affine.yield %76, %69, %75 : tensor<?x?xf32>, tensor<?x?xf32>, tensor<?x?x128xf32>
              } {loom.iter_type = #loom.iter_type<sequential>}
              %54 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%53#2, %53#1 : tensor<?x?x128xf32>, tensor<?x?xf32>) outs(%38 : tensor<?x?x128xf32>) {
              ^bb0(%in: f32, %in_4: f32, %out: f32):
                %56 = arith.divf %in, %in_4 : f32
                linalg.yield %56 : f32
              } -> tensor<?x?x128xf32>
              %55 = loom.subview %arg3[%45, %46, 0] [%12, %13, 128] [1, 1, 1], reuse : [seq = false, spat = false, temp = false] : memref<32x4096x128xf32> to memref<?x?x128xf32, strided<[524288, 128, 1], offset: ?>>
              loom.copy_from_tensor %54, %55 on @DRAM : tensor<?x?x128xf32>, memref<?x?x128xf32, strided<[524288, 128, 1], offset: ?>>
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
      loom.polynomial_constraint(%13) {monomials = [{coeff = 64 : i64, vars = [0]}], upper_bound = 4096 : i64}
    }
    func.func @attention__d1i1_d0i1__f01(%arg0: memref<32x128x4096xf32>, %arg1: memref<32x4096x128xf32>, %arg2: memref<32x4096x128xf32>, %arg3: memref<32x4096x128xf32>) {
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
              %18 = loom.semaphore %17 : memref<?x?x?xf32> -> memref<?x?x?xf32>
              %19 = loom.init_tensor %18[%12, %13, %14] : memref<?x?x?xf32> -> tensor<?x?x?xf32>
              %20 = loom.alloc [%12, 128, %14] on @L1 : memref<?x128x?xf32>
              %21 = loom.alloc [%12, %13] on @L1 : memref<?x?xf32>
              %22 = loom.semaphore %21 : memref<?x?xf32> -> memref<?x?xf32>
              %23 = loom.init_tensor %22[%12, %13] : memref<?x?xf32> -> tensor<?x?xf32>
              %24 = loom.alloc [%12, %13] on @L1 : memref<?x?xf32>
              %25 = loom.semaphore %24 : memref<?x?xf32> -> memref<?x?xf32>
              %26 = loom.init_tensor %25[%12, %13] : memref<?x?xf32> -> tensor<?x?xf32>
              %27 = loom.alloc [%12, %13] on @L1 : memref<?x?xf32>
              %28 = loom.semaphore %27 : memref<?x?xf32> -> memref<?x?xf32>
              %29 = loom.init_tensor %28[%12, %13] : memref<?x?xf32> -> tensor<?x?xf32>
              %30 = loom.alloc [%12, %13] on @L1 : memref<?x?xf32>
              %31 = loom.semaphore %30 : memref<?x?xf32> -> memref<?x?xf32>
              %32 = loom.init_tensor %31[%12, %13] : memref<?x?xf32> -> tensor<?x?xf32>
              %33 = loom.alloc [%12, %13] on @L1 : memref<?x?xf32>
              %34 = loom.semaphore %33 : memref<?x?xf32> -> memref<?x?xf32>
              %35 = loom.init_tensor %34[%12, %13] : memref<?x?xf32> -> tensor<?x?xf32>
              %36 = loom.alloc [%12, %13, 128] on @L1 : memref<?x?x128xf32>
              %37 = loom.semaphore %36 : memref<?x?x128xf32> -> memref<?x?x128xf32>
              %38 = loom.init_tensor %37[%12, %13, 128] : memref<?x?x128xf32> -> tensor<?x?x128xf32>
              %39 = loom.alloc [%12, %13, 128] on @L1 : memref<?x?x128xf32>
              %40 = loom.semaphore %39 : memref<?x?x128xf32> -> memref<?x?x128xf32>
              %41 = loom.init_tensor %40[%12, %13, 128] : memref<?x?x128xf32> -> tensor<?x?x128xf32>
              %42 = loom.alloc [%12, %13, 128] on @L1 : memref<?x?x128xf32>
              %43 = loom.semaphore %42 : memref<?x?x128xf32> -> memref<?x?x128xf32>
              %44 = loom.init_tensor %43[%12, %13, 128] : memref<?x?x128xf32> -> tensor<?x?x128xf32>
              %45 = arith.muli %arg6, %12 : index
              %46 = arith.muli %15, %13 : index
              %47 = loom.subview %arg2[%45, %46, 0] [%12, %13, 128] [1, 1, 1], reuse : [seq = false, spat = false, temp = false] : memref<32x4096x128xf32> to memref<?x?x128xf32, strided<[524288, 128, 1], offset: ?>>
              %48 = loom.semaphore %36 : memref<?x?x128xf32> -> memref<?x?x128xf32>
              %49 = loom.copy_to_tensor %47, %48 on @L1, interconnect : [], broadcast : [1, 1] : memref<?x?x128xf32, strided<[524288, 128, 1], offset: ?>>, memref<?x?x128xf32> -> tensor<?x?x128xf32>
              %50 = linalg.fill ins(%cst_1 : f32) outs(%41 : tensor<?x?x128xf32>) -> tensor<?x?x128xf32>
              %51 = linalg.fill ins(%cst_2 : f32) outs(%23 : tensor<?x?xf32>) -> tensor<?x?xf32>
              %52 = linalg.fill ins(%cst_3 : f32) outs(%26 : tensor<?x?xf32>) -> tensor<?x?xf32>
              %53:3 = affine.for %arg8 = 0 to affine_map<()[s0] -> (4096 ceildiv s0)>()[%14] iter_args(%arg9 = %52, %arg10 = %51, %arg11 = %50) -> (tensor<?x?xf32>, tensor<?x?xf32>, tensor<?x?x128xf32>) {
                %56 = arith.muli %arg8, %14 : index
                %57 = loom.subview %arg0[%45, 0, %56] [%12, 128, %14] [1, 1, 1], reuse : [seq = false, spat = false, temp = false] : memref<32x128x4096xf32> to memref<?x128x?xf32, strided<[524288, 4096, 1], offset: ?>>
                %58 = loom.semaphore %20 : memref<?x128x?xf32> -> memref<?x128x?xf32>
                %59 = loom.copy_to_tensor %57, %58 on @L1, interconnect : [], broadcast : [1, 1] : memref<?x128x?xf32, strided<[524288, 4096, 1], offset: ?>>, memref<?x128x?xf32> -> tensor<?x128x?xf32>
                %60 = linalg.fill ins(%cst_1 : f32) outs(%19 : tensor<?x?x?xf32>) -> tensor<?x?x?xf32>
                %61 = linalg.batch_matmul ins(%49, %59 : tensor<?x?x128xf32>, tensor<?x128x?xf32>) outs(%60 : tensor<?x?x?xf32>) -> tensor<?x?x?xf32>
                %62 = linalg.fill ins(%cst_3 : f32) outs(%29 : tensor<?x?xf32>) -> tensor<?x?xf32>
                %63 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%61 : tensor<?x?x?xf32>) outs(%62 : tensor<?x?xf32>) {
                ^bb0(%in: f32, %out: f32):
                  %77 = arith.maximumf %in, %out : f32
                  linalg.yield %77 : f32
                } -> tensor<?x?xf32>
                %64 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%arg9, %63 : tensor<?x?xf32>, tensor<?x?xf32>) outs(%29 : tensor<?x?xf32>) {
                ^bb0(%in: f32, %in_4: f32, %out: f32):
                  %77 = arith.truncf %cst_0 : f64 to f32
                  %78 = arith.mulf %in_4, %77 : f32
                  %79 = arith.cmpf ogt, %in, %78 : f32
                  %80 = arith.select %79, %in, %78 : f32
                  linalg.yield %80 : f32
                } -> tensor<?x?xf32>
                %65 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%61, %64 : tensor<?x?x?xf32>, tensor<?x?xf32>) outs(%19 : tensor<?x?x?xf32>) {
                ^bb0(%in: f32, %in_4: f32, %out: f32):
                  %77 = arith.truncf %cst_0 : f64 to f32
                  %78 = arith.mulf %in, %77 : f32
                  %79 = arith.subf %78, %in_4 : f32
                  %80 = math.powf %cst, %79 : f32
                  linalg.yield %80 : f32
                } -> tensor<?x?x?xf32>
                %66 = linalg.fill ins(%cst_1 : f32) outs(%32 : tensor<?x?xf32>) -> tensor<?x?xf32>
                %67 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%65 : tensor<?x?x?xf32>) outs(%66 : tensor<?x?xf32>) {
                ^bb0(%in: f32, %out: f32):
                  %77 = arith.addf %in, %out : f32
                  linalg.yield %77 : f32
                } -> tensor<?x?xf32>
                %68 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%arg9, %64 : tensor<?x?xf32>, tensor<?x?xf32>) outs(%35 : tensor<?x?xf32>) {
                ^bb0(%in: f32, %in_4: f32, %out: f32):
                  %77 = arith.subf %in, %in_4 : f32
                  %78 = math.powf %cst, %77 : f32
                  linalg.yield %78 : f32
                } -> tensor<?x?xf32>
                %69 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%arg10, %68, %67 : tensor<?x?xf32>, tensor<?x?xf32>, tensor<?x?xf32>) outs(%arg10 : tensor<?x?xf32>) {
                ^bb0(%in: f32, %in_4: f32, %in_5: f32, %out: f32):
                  %77 = arith.mulf %in, %in_4 : f32
                  %78 = arith.addf %77, %in_5 : f32
                  linalg.yield %78 : f32
                } -> tensor<?x?xf32>
                %70 = loom.subview %arg1[%45, %56, 0] [%12, %14, 128] [1, 1, 1], reuse : [seq = false, spat = false, temp = false] : memref<32x4096x128xf32> to memref<?x?x128xf32, strided<[524288, 128, 1], offset: ?>>
                %71 = loom.semaphore %16 : memref<?x?x128xf32> -> memref<?x?x128xf32>
                %72 = loom.copy_to_tensor %70, %71 on @L1, interconnect : [], broadcast : [1, 1] : memref<?x?x128xf32, strided<[524288, 128, 1], offset: ?>>, memref<?x?x128xf32> -> tensor<?x?x128xf32>
                %73 = linalg.fill ins(%cst_1 : f32) outs(%44 : tensor<?x?x128xf32>) -> tensor<?x?x128xf32>
                %74 = linalg.batch_matmul ins(%65, %72 : tensor<?x?x?xf32>, tensor<?x?x128xf32>) outs(%73 : tensor<?x?x128xf32>) -> tensor<?x?x128xf32>
                %75 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%74, %arg11, %68 : tensor<?x?x128xf32>, tensor<?x?x128xf32>, tensor<?x?xf32>) outs(%arg11 : tensor<?x?x128xf32>) {
                ^bb0(%in: f32, %in_4: f32, %in_5: f32, %out: f32):
                  %77 = arith.mulf %in_4, %in_5 : f32
                  %78 = arith.addf %in, %77 : f32
                  linalg.yield %78 : f32
                } -> tensor<?x?x128xf32>
                %76 = linalg.copy ins(%64 : tensor<?x?xf32>) outs(%arg9 : tensor<?x?xf32>) -> tensor<?x?xf32>
                affine.yield %76, %69, %75 : tensor<?x?xf32>, tensor<?x?xf32>, tensor<?x?x128xf32>
              } {loom.iter_type = #loom.iter_type<sequential>}
              %54 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%53#2, %53#1 : tensor<?x?x128xf32>, tensor<?x?xf32>) outs(%38 : tensor<?x?x128xf32>) {
              ^bb0(%in: f32, %in_4: f32, %out: f32):
                %56 = arith.divf %in, %in_4 : f32
                linalg.yield %56 : f32
              } -> tensor<?x?x128xf32>
              %55 = loom.subview %arg3[%45, %46, 0] [%12, %13, 128] [1, 1, 1], reuse : [seq = false, spat = false, temp = false] : memref<32x4096x128xf32> to memref<?x?x128xf32, strided<[524288, 128, 1], offset: ?>>
              loom.copy_from_tensor %54, %55 on @DRAM : tensor<?x?x128xf32>, memref<?x?x128xf32, strided<[524288, 128, 1], offset: ?>>
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
      loom.polynomial_constraint(%13) {monomials = [{coeff = 64 : i64, vars = [0]}], upper_bound = 4096 : i64}
    }
    func.func @attention__d1i1_d0i1__f10(%arg0: memref<32x128x4096xf32>, %arg1: memref<32x4096x128xf32>, %arg2: memref<32x4096x128xf32>, %arg3: memref<32x4096x128xf32>) {
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
              %18 = loom.semaphore %17 : memref<?x?x?xf32> -> memref<?x?x?xf32>
              %19 = loom.init_tensor %18[%12, %13, %14] : memref<?x?x?xf32> -> tensor<?x?x?xf32>
              %20 = loom.alloc [%12, 128, %14] on @L1 : memref<?x128x?xf32>
              %21 = loom.alloc [%12, %13] on @L1 : memref<?x?xf32>
              %22 = loom.semaphore %21 : memref<?x?xf32> -> memref<?x?xf32>
              %23 = loom.init_tensor %22[%12, %13] : memref<?x?xf32> -> tensor<?x?xf32>
              %24 = loom.alloc [%12, %13] on @L1 : memref<?x?xf32>
              %25 = loom.semaphore %24 : memref<?x?xf32> -> memref<?x?xf32>
              %26 = loom.init_tensor %25[%12, %13] : memref<?x?xf32> -> tensor<?x?xf32>
              %27 = loom.alloc [%12, %13] on @L1 : memref<?x?xf32>
              %28 = loom.semaphore %27 : memref<?x?xf32> -> memref<?x?xf32>
              %29 = loom.init_tensor %28[%12, %13] : memref<?x?xf32> -> tensor<?x?xf32>
              %30 = loom.alloc [%12, %13] on @L1 : memref<?x?xf32>
              %31 = loom.semaphore %30 : memref<?x?xf32> -> memref<?x?xf32>
              %32 = loom.init_tensor %31[%12, %13] : memref<?x?xf32> -> tensor<?x?xf32>
              %33 = loom.alloc [%12, %13] on @L1 : memref<?x?xf32>
              %34 = loom.semaphore %33 : memref<?x?xf32> -> memref<?x?xf32>
              %35 = loom.init_tensor %34[%12, %13] : memref<?x?xf32> -> tensor<?x?xf32>
              %36 = loom.alloc [%12, %13, 128] on @L1 : memref<?x?x128xf32>
              %37 = loom.semaphore %36 : memref<?x?x128xf32> -> memref<?x?x128xf32>
              %38 = loom.init_tensor %37[%12, %13, 128] : memref<?x?x128xf32> -> tensor<?x?x128xf32>
              %39 = loom.alloc [%12, %13, 128] on @L1 : memref<?x?x128xf32>
              %40 = loom.semaphore %39 : memref<?x?x128xf32> -> memref<?x?x128xf32>
              %41 = loom.init_tensor %40[%12, %13, 128] : memref<?x?x128xf32> -> tensor<?x?x128xf32>
              %42 = loom.alloc [%12, %13, 128] on @L1 : memref<?x?x128xf32>
              %43 = loom.semaphore %42 : memref<?x?x128xf32> -> memref<?x?x128xf32>
              %44 = loom.init_tensor %43[%12, %13, 128] : memref<?x?x128xf32> -> tensor<?x?x128xf32>
              %45 = arith.muli %arg7, %12 : index
              %46 = arith.muli %15, %13 : index
              %47 = loom.subview %arg2[%45, %46, 0] [%12, %13, 128] [1, 1, 1], reuse : [seq = false, spat = false, temp = false] : memref<32x4096x128xf32> to memref<?x?x128xf32, strided<[524288, 128, 1], offset: ?>>
              %48 = loom.semaphore %36 : memref<?x?x128xf32> -> memref<?x?x128xf32>
              %49 = loom.copy_to_tensor %47, %48 on @L1, interconnect : [], broadcast : [1, 1] : memref<?x?x128xf32, strided<[524288, 128, 1], offset: ?>>, memref<?x?x128xf32> -> tensor<?x?x128xf32>
              %50 = linalg.fill ins(%cst_1 : f32) outs(%41 : tensor<?x?x128xf32>) -> tensor<?x?x128xf32>
              %51 = linalg.fill ins(%cst_2 : f32) outs(%23 : tensor<?x?xf32>) -> tensor<?x?xf32>
              %52 = linalg.fill ins(%cst_3 : f32) outs(%26 : tensor<?x?xf32>) -> tensor<?x?xf32>
              %53:3 = affine.for %arg8 = 0 to affine_map<()[s0] -> (4096 ceildiv s0)>()[%14] iter_args(%arg9 = %52, %arg10 = %51, %arg11 = %50) -> (tensor<?x?xf32>, tensor<?x?xf32>, tensor<?x?x128xf32>) {
                %56 = arith.muli %arg8, %14 : index
                %57 = loom.subview %arg0[%45, 0, %56] [%12, 128, %14] [1, 1, 1], reuse : [seq = false, spat = false, temp = false] : memref<32x128x4096xf32> to memref<?x128x?xf32, strided<[524288, 4096, 1], offset: ?>>
                %58 = loom.semaphore %20 : memref<?x128x?xf32> -> memref<?x128x?xf32>
                %59 = loom.copy_to_tensor %57, %58 on @L1, interconnect : [], broadcast : [1, 1] : memref<?x128x?xf32, strided<[524288, 4096, 1], offset: ?>>, memref<?x128x?xf32> -> tensor<?x128x?xf32>
                %60 = linalg.fill ins(%cst_1 : f32) outs(%19 : tensor<?x?x?xf32>) -> tensor<?x?x?xf32>
                %61 = linalg.batch_matmul ins(%49, %59 : tensor<?x?x128xf32>, tensor<?x128x?xf32>) outs(%60 : tensor<?x?x?xf32>) -> tensor<?x?x?xf32>
                %62 = linalg.fill ins(%cst_3 : f32) outs(%29 : tensor<?x?xf32>) -> tensor<?x?xf32>
                %63 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%61 : tensor<?x?x?xf32>) outs(%62 : tensor<?x?xf32>) {
                ^bb0(%in: f32, %out: f32):
                  %77 = arith.maximumf %in, %out : f32
                  linalg.yield %77 : f32
                } -> tensor<?x?xf32>
                %64 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%arg9, %63 : tensor<?x?xf32>, tensor<?x?xf32>) outs(%29 : tensor<?x?xf32>) {
                ^bb0(%in: f32, %in_4: f32, %out: f32):
                  %77 = arith.truncf %cst_0 : f64 to f32
                  %78 = arith.mulf %in_4, %77 : f32
                  %79 = arith.cmpf ogt, %in, %78 : f32
                  %80 = arith.select %79, %in, %78 : f32
                  linalg.yield %80 : f32
                } -> tensor<?x?xf32>
                %65 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%61, %64 : tensor<?x?x?xf32>, tensor<?x?xf32>) outs(%19 : tensor<?x?x?xf32>) {
                ^bb0(%in: f32, %in_4: f32, %out: f32):
                  %77 = arith.truncf %cst_0 : f64 to f32
                  %78 = arith.mulf %in, %77 : f32
                  %79 = arith.subf %78, %in_4 : f32
                  %80 = math.powf %cst, %79 : f32
                  linalg.yield %80 : f32
                } -> tensor<?x?x?xf32>
                %66 = linalg.fill ins(%cst_1 : f32) outs(%32 : tensor<?x?xf32>) -> tensor<?x?xf32>
                %67 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%65 : tensor<?x?x?xf32>) outs(%66 : tensor<?x?xf32>) {
                ^bb0(%in: f32, %out: f32):
                  %77 = arith.addf %in, %out : f32
                  linalg.yield %77 : f32
                } -> tensor<?x?xf32>
                %68 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%arg9, %64 : tensor<?x?xf32>, tensor<?x?xf32>) outs(%35 : tensor<?x?xf32>) {
                ^bb0(%in: f32, %in_4: f32, %out: f32):
                  %77 = arith.subf %in, %in_4 : f32
                  %78 = math.powf %cst, %77 : f32
                  linalg.yield %78 : f32
                } -> tensor<?x?xf32>
                %69 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%arg10, %68, %67 : tensor<?x?xf32>, tensor<?x?xf32>, tensor<?x?xf32>) outs(%arg10 : tensor<?x?xf32>) {
                ^bb0(%in: f32, %in_4: f32, %in_5: f32, %out: f32):
                  %77 = arith.mulf %in, %in_4 : f32
                  %78 = arith.addf %77, %in_5 : f32
                  linalg.yield %78 : f32
                } -> tensor<?x?xf32>
                %70 = loom.subview %arg1[%45, %56, 0] [%12, %14, 128] [1, 1, 1], reuse : [seq = false, spat = false, temp = false] : memref<32x4096x128xf32> to memref<?x?x128xf32, strided<[524288, 128, 1], offset: ?>>
                %71 = loom.semaphore %16 : memref<?x?x128xf32> -> memref<?x?x128xf32>
                %72 = loom.copy_to_tensor %70, %71 on @L1, interconnect : [], broadcast : [1, 1] : memref<?x?x128xf32, strided<[524288, 128, 1], offset: ?>>, memref<?x?x128xf32> -> tensor<?x?x128xf32>
                %73 = linalg.fill ins(%cst_1 : f32) outs(%44 : tensor<?x?x128xf32>) -> tensor<?x?x128xf32>
                %74 = linalg.batch_matmul ins(%65, %72 : tensor<?x?x?xf32>, tensor<?x?x128xf32>) outs(%73 : tensor<?x?x128xf32>) -> tensor<?x?x128xf32>
                %75 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%74, %arg11, %68 : tensor<?x?x128xf32>, tensor<?x?x128xf32>, tensor<?x?xf32>) outs(%arg11 : tensor<?x?x128xf32>) {
                ^bb0(%in: f32, %in_4: f32, %in_5: f32, %out: f32):
                  %77 = arith.mulf %in_4, %in_5 : f32
                  %78 = arith.addf %in, %77 : f32
                  linalg.yield %78 : f32
                } -> tensor<?x?x128xf32>
                %76 = linalg.copy ins(%64 : tensor<?x?xf32>) outs(%arg9 : tensor<?x?xf32>) -> tensor<?x?xf32>
                affine.yield %76, %69, %75 : tensor<?x?xf32>, tensor<?x?xf32>, tensor<?x?x128xf32>
              } {loom.iter_type = #loom.iter_type<sequential>}
              %54 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%53#2, %53#1 : tensor<?x?x128xf32>, tensor<?x?xf32>) outs(%38 : tensor<?x?x128xf32>) {
              ^bb0(%in: f32, %in_4: f32, %out: f32):
                %56 = arith.divf %in, %in_4 : f32
                linalg.yield %56 : f32
              } -> tensor<?x?x128xf32>
              %55 = loom.subview %arg3[%45, %46, 0] [%12, %13, 128] [1, 1, 1], reuse : [seq = false, spat = false, temp = false] : memref<32x4096x128xf32> to memref<?x?x128xf32, strided<[524288, 128, 1], offset: ?>>
              loom.copy_from_tensor %54, %55 on @DRAM : tensor<?x?x128xf32>, memref<?x?x128xf32, strided<[524288, 128, 1], offset: ?>>
            } {loom.iter_type = #loom.iter_type<temporal>}
          } {loom.iter_type = #loom.iter_type<temporal>}
        } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @x}
      } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @y}
      return
    }
  }
}
