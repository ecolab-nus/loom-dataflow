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
  module attributes {loom.pass_name = "EnumerateCopyBroadcast"} {
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
      %15 = loom.expression(%13, %12) {coeffs = [1, 1], logic = "mul"} : index
      %16 = loom.expression(%14, %15) {coeffs = [1, 1], logic = "mul"} : index
      loom.polynomial_constraint(%12, %13, %14, %15, %16) {monomials = [{coeff = -1 : i64, vars = [4]}], upper_bound = -262144 : i64}
      %17 = loom.expression(%12, %13) {coeffs = [1, 1], logic = "add"} : index
      %18 = loom.expression(%14, %17) {coeffs = [1, 1], logic = "mul"} : index
      loom.polynomial_constraint(%12, %14, %13, %17, %18) {monomials = [{coeff = 1 : i64, vars = [4]}], upper_bound = 374784 : i64}
      loom.polynomial_constraint(%12) {monomials = [{coeff = 1 : i64, vars = [0]}], upper_bound = 16 : i64}
      %19 = loom.expression(%14, %12) {coeffs = [1, 1], logic = "mul"} : index
      %20 = loom.expression(%14, %13) {coeffs = [1, 1], logic = "mul"} : index
      %21 = loom.expression(%13, %12) {coeffs = [1, 1], logic = "mul"} : index
      %22 = loom.expression(%14, %21) {coeffs = [1, 1], logic = "mul"} : index
      loom.polynomial_constraint(%12, %13, %14, %19, %20, %21, %22) {monomials = [{coeff = 64 : i64, vars = [3]}, {coeff = 64 : i64, vars = [4]}, {coeff = -1 : i64, vars = [6]}], upper_bound = 0 : i64}
    }
    func.func @matmul_kernel__d0i0_d1i0__f01__d_d(%arg0: memref<*xf32> {tt.divisibility = 16 : i32}, %arg1: memref<*xf32> {tt.divisibility = 16 : i32}, %arg2: memref<*xf32> {tt.divisibility = 16 : i32}, %arg3: index, %arg4: index, %arg5: index, %arg6: index, %arg7: index, %arg8: index) {
      affine.parallel (%arg9) = (0) to (8) {
        affine.parallel (%arg10) = (0) to (8) {
          affine.for %arg11 = 0 to affine_map<()[s0, s1] -> ((s0 ceildiv 8) ceildiv 8)>()[%arg3, %arg4] {
            affine.for %arg12 = 0 to affine_map<()[s0, s1] -> (s1)>()[%arg3, %arg4] {
              %12 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 8 + d1 * 8 + d2)>(%arg9, %arg10, %arg11)
              %c1024 = arith.constant 1024 : index
              %c1024_0 = arith.constant 1024 : index
              %c1 = arith.constant 1 : index
              %13 = loom.get_symbolic_block_size @constraints::@BM : index
              %14 = loom.get_symbolic_block_size @constraints::@BN : index
              %15 = loom.get_symbolic_block_size @constraints::@BK : index
              %16 = arith.ceildivsi %c1024_0, %15 : index
              %cst = arith.constant 0.000000e+00 : f32
              %c1024_1 = arith.constant 1024 : index
              %c0 = arith.constant 0 : index
              %17 = tensor.empty(%13, %14) : tensor<?x?xf32>
              %18 = linalg.fill ins(%cst : f32) outs(%17 : tensor<?x?xf32>) -> tensor<?x?xf32>
              %19 = arith.muli %12, %13 : index
              %20 = arith.muli %arg12, %14 : index
              %21 = scf.for %arg13 = %c0 to %16 step %c1 iter_args(%arg14 = %18) -> (tensor<?x?xf32>) {
                %25 = arith.muli %arg13, %15 : index
                %26 = arith.muli %19, %c1024_1 : index
                %27 = arith.addi %26, %25 : index
                %28 = loom.reinterpret_cast %arg0 to offset : [%27], sizes : [%13, %15], strides : [%c1024_0, %c1], reuse : [seq = false, spat = false, temp = true] : memref<*xf32> to memref<?x?xf32, strided<[?, ?], offset: ?>>
                %29 = loom.alloc(%13, %15) on @L1 : memref<?x?xf32>
                loom.copy %28, %29, interconnect : [], broadcast : [1, 1] : memref<?x?xf32, strided<[?, ?], offset: ?>> to memref<?x?xf32>
                %30 = bufferization.to_tensor %29 restrict writable : memref<?x?xf32> to tensor<?x?xf32>
                %31 = arith.muli %25, %c1024_1 : index
                %32 = arith.addi %31, %20 : index
                %33 = loom.reinterpret_cast %arg1 to offset : [%32], sizes : [%15, %14], strides : [%c1024, %c1], reuse : [seq = false, spat = true, temp = true] : memref<*xf32> to memref<?x?xf32, strided<[?, ?], offset: ?>>
                %34 = loom.alloc(%15, %14) on @L1 : memref<?x?xf32>
                loom.copy %33, %34, interconnect : [], broadcast : [1, 1] : memref<?x?xf32, strided<[?, ?], offset: ?>> to memref<?x?xf32>
                %35 = bufferization.to_tensor %34 restrict writable : memref<?x?xf32> to tensor<?x?xf32>
                %36 = linalg.matmul ins(%30, %35 : tensor<?x?xf32>, tensor<?x?xf32>) outs(%18 : tensor<?x?xf32>) -> tensor<?x?xf32>
                %37 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%arg14, %36 : tensor<?x?xf32>, tensor<?x?xf32>) outs(%arg14 : tensor<?x?xf32>) {
                ^bb0(%in: f32, %in_2: f32, %out: f32):
                  %38 = arith.addf %in, %in_2 : f32
                  linalg.yield %38 : f32
                } -> tensor<?x?xf32>
                scf.yield %37 : tensor<?x?xf32>
              }
              %22 = arith.muli %19, %c1024_1 : index
              %23 = arith.addi %22, %20 : index
              %24 = loom.reinterpret_cast %arg2 to offset : [%23], sizes : [%13, %14], strides : [%c1024, %c1], reuse : [seq = false, spat = false, temp = false] : memref<*xf32> to memref<?x?xf32, strided<[?, ?], offset: ?>>
              bufferization.materialize_in_destination %21 in writable %24 : (tensor<?x?xf32>, memref<?x?xf32, strided<[?, ?], offset: ?>>) -> ()
            }
          }
        } {loom.mapped_to = @y}
      } {loom.mapped_to = @x}
      return
    }
  }
  module attributes {loom.pass_name = "EnumerateCopyBroadcast"} {
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
      %15 = loom.expression(%13, %12) {coeffs = [1, 1], logic = "mul"} : index
      %16 = loom.expression(%14, %15) {coeffs = [1, 1], logic = "mul"} : index
      loom.polynomial_constraint(%12, %13, %14, %15, %16) {monomials = [{coeff = -1 : i64, vars = [4]}], upper_bound = -262144 : i64}
      %17 = loom.expression(%12, %13) {coeffs = [1, 1], logic = "add"} : index
      %18 = loom.expression(%17, %14) {coeffs = [1, 1], logic = "mul"} : index
      loom.polynomial_constraint(%12, %14, %13, %17, %18) {monomials = [{coeff = 1 : i64, vars = [4]}], upper_bound = 374784 : i64}
      loom.polynomial_constraint(%12) {monomials = [{coeff = 1 : i64, vars = [0]}], upper_bound = 16 : i64}
      %19 = loom.expression(%14, %12) {coeffs = [1, 1], logic = "mul"} : index
      %20 = loom.expression(%13, %14) {coeffs = [1, 1], logic = "mul"} : index
      %21 = loom.expression(%13, %12) {coeffs = [1, 1], logic = "mul"} : index
      %22 = loom.expression(%14, %21) {coeffs = [1, 1], logic = "mul"} : index
      loom.polynomial_constraint(%12, %13, %14, %19, %20, %21, %22) {monomials = [{coeff = 960 : i64, vars = [3]}, {coeff = 256 : i64, vars = [4]}, {coeff = -15 : i64, vars = [6]}], upper_bound = 0 : i64}
    }
    func.func @matmul_kernel__d0i0_d1i0__f01__d_a(%arg0: memref<*xf32> {tt.divisibility = 16 : i32}, %arg1: memref<*xf32> {tt.divisibility = 16 : i32}, %arg2: memref<*xf32> {tt.divisibility = 16 : i32}, %arg3: index, %arg4: index, %arg5: index, %arg6: index, %arg7: index, %arg8: index) {
      affine.parallel (%arg9) = (0) to (8) {
        affine.parallel (%arg10) = (0) to (8) {
          affine.for %arg11 = 0 to affine_map<()[s0, s1] -> ((s0 ceildiv 8) ceildiv 8)>()[%arg3, %arg4] {
            affine.for %arg12 = 0 to affine_map<()[s0, s1] -> (s1)>()[%arg3, %arg4] {
              %12 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 8 + d1 * 8 + d2)>(%arg9, %arg10, %arg11)
              %c1024 = arith.constant 1024 : index
              %c1024_0 = arith.constant 1024 : index
              %c1 = arith.constant 1 : index
              %13 = loom.get_symbolic_block_size @constraints::@BM : index
              %14 = loom.get_symbolic_block_size @constraints::@BN : index
              %15 = loom.get_symbolic_block_size @constraints::@BK : index
              %16 = arith.ceildivsi %c1024_0, %15 : index
              %cst = arith.constant 0.000000e+00 : f32
              %c1024_1 = arith.constant 1024 : index
              %c0 = arith.constant 0 : index
              %17 = tensor.empty(%13, %14) : tensor<?x?xf32>
              %18 = linalg.fill ins(%cst : f32) outs(%17 : tensor<?x?xf32>) -> tensor<?x?xf32>
              %19 = arith.muli %12, %13 : index
              %20 = arith.muli %arg12, %14 : index
              %21 = scf.for %arg13 = %c0 to %16 step %c1 iter_args(%arg14 = %18) -> (tensor<?x?xf32>) {
                %25 = arith.muli %arg13, %15 : index
                %26 = arith.muli %19, %c1024_1 : index
                %27 = arith.addi %26, %25 : index
                %28 = loom.reinterpret_cast %arg0 to offset : [%27], sizes : [%13, %15], strides : [%c1024_0, %c1], reuse : [seq = false, spat = false, temp = true] : memref<*xf32> to memref<?x?xf32, strided<[?, ?], offset: ?>>
                %29 = loom.alloc(%13, %15) on @L1 : memref<?x?xf32>
                loom.copy %28, %29, interconnect : [], broadcast : [1, 1] : memref<?x?xf32, strided<[?, ?], offset: ?>> to memref<?x?xf32>
                %30 = bufferization.to_tensor %29 restrict writable : memref<?x?xf32> to tensor<?x?xf32>
                %31 = arith.muli %25, %c1024_1 : index
                %32 = arith.addi %31, %20 : index
                %33 = loom.reinterpret_cast %arg1 to offset : [%32], sizes : [%15, %14], strides : [%c1024, %c1], reuse : [seq = false, spat = true, temp = true] : memref<*xf32> to memref<?x?xf32, strided<[?, ?], offset: ?>>
                %34 = loom.alloc(%15, %14) on @L1 : memref<?x?xf32>
                loom.copy %33, %34, interconnect : [@horizontal_links, @vertical_links], broadcast : [8, 8] : memref<?x?xf32, strided<[?, ?], offset: ?>> to memref<?x?xf32>
                %35 = bufferization.to_tensor %34 restrict writable : memref<?x?xf32> to tensor<?x?xf32>
                %36 = linalg.matmul ins(%30, %35 : tensor<?x?xf32>, tensor<?x?xf32>) outs(%18 : tensor<?x?xf32>) -> tensor<?x?xf32>
                %37 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%arg14, %36 : tensor<?x?xf32>, tensor<?x?xf32>) outs(%arg14 : tensor<?x?xf32>) {
                ^bb0(%in: f32, %in_2: f32, %out: f32):
                  %38 = arith.addf %in, %in_2 : f32
                  linalg.yield %38 : f32
                } -> tensor<?x?xf32>
                scf.yield %37 : tensor<?x?xf32>
              }
              %22 = arith.muli %19, %c1024_1 : index
              %23 = arith.addi %22, %20 : index
              %24 = loom.reinterpret_cast %arg2 to offset : [%23], sizes : [%13, %14], strides : [%c1024, %c1], reuse : [seq = false, spat = false, temp = false] : memref<*xf32> to memref<?x?xf32, strided<[?, ?], offset: ?>>
              bufferization.materialize_in_destination %21 in writable %24 : (tensor<?x?xf32>, memref<?x?xf32, strided<[?, ?], offset: ?>>) -> ()
            }
          }
        } {loom.mapped_to = @y}
      } {loom.mapped_to = @x}
      return
    }
  }
  module attributes {loom.pass_name = "EnumerateCopyBroadcast"} {
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
      %15 = loom.expression(%12, %13) {coeffs = [1, 1], logic = "mul"} : index
      %16 = loom.expression(%14, %15) {coeffs = [1, 1], logic = "mul"} : index
      loom.polynomial_constraint(%12, %13, %14, %15, %16) {monomials = [{coeff = -1 : i64, vars = [4]}], upper_bound = -262144 : i64}
      %17 = loom.expression(%12, %13) {coeffs = [1, 1], logic = "add"} : index
      %18 = loom.expression(%14, %17) {coeffs = [1, 1], logic = "mul"} : index
      loom.polynomial_constraint(%12, %14, %13, %17, %18) {monomials = [{coeff = 1 : i64, vars = [4]}], upper_bound = 374784 : i64}
      loom.polynomial_constraint(%12) {monomials = [{coeff = 1 : i64, vars = [0]}], upper_bound = 16 : i64}
      %19 = loom.expression(%12, %14) {coeffs = [1, 1], logic = "mul"} : index
      %20 = loom.expression(%13, %14) {coeffs = [1, 1], logic = "mul"} : index
      %21 = loom.expression(%12, %13) {coeffs = [1, 1], logic = "mul"} : index
      %22 = loom.expression(%14, %21) {coeffs = [1, 1], logic = "mul"} : index
      loom.polynomial_constraint(%12, %13, %14, %19, %20, %21, %22) {monomials = [{coeff = 960 : i64, vars = [3]}, {coeff = 256 : i64, vars = [4]}, {coeff = -15 : i64, vars = [6]}], upper_bound = 0 : i64}
    }
    func.func @matmul_kernel__d0i0_d1i0__f01__d_h(%arg0: memref<*xf32> {tt.divisibility = 16 : i32}, %arg1: memref<*xf32> {tt.divisibility = 16 : i32}, %arg2: memref<*xf32> {tt.divisibility = 16 : i32}, %arg3: index, %arg4: index, %arg5: index, %arg6: index, %arg7: index, %arg8: index) {
      affine.parallel (%arg9) = (0) to (8) {
        affine.parallel (%arg10) = (0) to (8) {
          affine.for %arg11 = 0 to affine_map<()[s0, s1] -> ((s0 ceildiv 8) ceildiv 8)>()[%arg3, %arg4] {
            affine.for %arg12 = 0 to affine_map<()[s0, s1] -> (s1)>()[%arg3, %arg4] {
              %12 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 8 + d1 * 8 + d2)>(%arg9, %arg10, %arg11)
              %c1024 = arith.constant 1024 : index
              %c1024_0 = arith.constant 1024 : index
              %c1 = arith.constant 1 : index
              %13 = loom.get_symbolic_block_size @constraints::@BM : index
              %14 = loom.get_symbolic_block_size @constraints::@BN : index
              %15 = loom.get_symbolic_block_size @constraints::@BK : index
              %16 = arith.ceildivsi %c1024_0, %15 : index
              %cst = arith.constant 0.000000e+00 : f32
              %c1024_1 = arith.constant 1024 : index
              %c0 = arith.constant 0 : index
              %17 = tensor.empty(%13, %14) : tensor<?x?xf32>
              %18 = linalg.fill ins(%cst : f32) outs(%17 : tensor<?x?xf32>) -> tensor<?x?xf32>
              %19 = arith.muli %12, %13 : index
              %20 = arith.muli %arg12, %14 : index
              %21 = scf.for %arg13 = %c0 to %16 step %c1 iter_args(%arg14 = %18) -> (tensor<?x?xf32>) {
                %25 = arith.muli %arg13, %15 : index
                %26 = arith.muli %19, %c1024_1 : index
                %27 = arith.addi %26, %25 : index
                %28 = loom.reinterpret_cast %arg0 to offset : [%27], sizes : [%13, %15], strides : [%c1024_0, %c1], reuse : [seq = false, spat = false, temp = true] : memref<*xf32> to memref<?x?xf32, strided<[?, ?], offset: ?>>
                %29 = loom.alloc(%13, %15) on @L1 : memref<?x?xf32>
                loom.copy %28, %29, interconnect : [], broadcast : [1, 1] : memref<?x?xf32, strided<[?, ?], offset: ?>> to memref<?x?xf32>
                %30 = bufferization.to_tensor %29 restrict writable : memref<?x?xf32> to tensor<?x?xf32>
                %31 = arith.muli %25, %c1024_1 : index
                %32 = arith.addi %31, %20 : index
                %33 = loom.reinterpret_cast %arg1 to offset : [%32], sizes : [%15, %14], strides : [%c1024, %c1], reuse : [seq = false, spat = true, temp = true] : memref<*xf32> to memref<?x?xf32, strided<[?, ?], offset: ?>>
                %34 = loom.alloc(%15, %14) on @L1 : memref<?x?xf32>
                loom.copy %33, %34, interconnect : [@horizontal_links], broadcast : [1, 8] : memref<?x?xf32, strided<[?, ?], offset: ?>> to memref<?x?xf32>
                %35 = bufferization.to_tensor %34 restrict writable : memref<?x?xf32> to tensor<?x?xf32>
                %36 = linalg.matmul ins(%30, %35 : tensor<?x?xf32>, tensor<?x?xf32>) outs(%18 : tensor<?x?xf32>) -> tensor<?x?xf32>
                %37 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%arg14, %36 : tensor<?x?xf32>, tensor<?x?xf32>) outs(%arg14 : tensor<?x?xf32>) {
                ^bb0(%in: f32, %in_2: f32, %out: f32):
                  %38 = arith.addf %in, %in_2 : f32
                  linalg.yield %38 : f32
                } -> tensor<?x?xf32>
                scf.yield %37 : tensor<?x?xf32>
              }
              %22 = arith.muli %19, %c1024_1 : index
              %23 = arith.addi %22, %20 : index
              %24 = loom.reinterpret_cast %arg2 to offset : [%23], sizes : [%13, %14], strides : [%c1024, %c1], reuse : [seq = false, spat = false, temp = false] : memref<*xf32> to memref<?x?xf32, strided<[?, ?], offset: ?>>
              bufferization.materialize_in_destination %21 in writable %24 : (tensor<?x?xf32>, memref<?x?xf32, strided<[?, ?], offset: ?>>) -> ()
            }
          }
        } {loom.mapped_to = @y}
      } {loom.mapped_to = @x}
      return
    }
  }
  module attributes {loom.pass_name = "EnumerateCopyBroadcast"} {
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
      %15 = loom.expression(%12, %13) {coeffs = [1, 1], logic = "mul"} : index
      %16 = loom.expression(%14, %15) {coeffs = [1, 1], logic = "mul"} : index
      loom.polynomial_constraint(%12, %13, %14, %15, %16) {monomials = [{coeff = -1 : i64, vars = [4]}], upper_bound = -262144 : i64}
      %17 = loom.expression(%12, %13) {coeffs = [1, 1], logic = "add"} : index
      %18 = loom.expression(%14, %17) {coeffs = [1, 1], logic = "mul"} : index
      loom.polynomial_constraint(%12, %14, %13, %17, %18) {monomials = [{coeff = 1 : i64, vars = [4]}], upper_bound = 374784 : i64}
      loom.polynomial_constraint(%12) {monomials = [{coeff = 1 : i64, vars = [0]}], upper_bound = 16 : i64}
      %19 = loom.expression(%12, %14) {coeffs = [1, 1], logic = "mul"} : index
      %20 = loom.expression(%13, %14) {coeffs = [1, 1], logic = "mul"} : index
      %21 = loom.expression(%12, %13) {coeffs = [1, 1], logic = "mul"} : index
      %22 = loom.expression(%14, %21) {coeffs = [1, 1], logic = "mul"} : index
      loom.polynomial_constraint(%12, %13, %14, %19, %20, %21, %22) {monomials = [{coeff = 960 : i64, vars = [3]}, {coeff = 256 : i64, vars = [4]}, {coeff = -15 : i64, vars = [6]}], upper_bound = 0 : i64}
    }
    func.func @matmul_kernel__d0i0_d1i0__f01__d_v(%arg0: memref<*xf32> {tt.divisibility = 16 : i32}, %arg1: memref<*xf32> {tt.divisibility = 16 : i32}, %arg2: memref<*xf32> {tt.divisibility = 16 : i32}, %arg3: index, %arg4: index, %arg5: index, %arg6: index, %arg7: index, %arg8: index) {
      affine.parallel (%arg9) = (0) to (8) {
        affine.parallel (%arg10) = (0) to (8) {
          affine.for %arg11 = 0 to affine_map<()[s0, s1] -> ((s0 ceildiv 8) ceildiv 8)>()[%arg3, %arg4] {
            affine.for %arg12 = 0 to affine_map<()[s0, s1] -> (s1)>()[%arg3, %arg4] {
              %12 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 8 + d1 * 8 + d2)>(%arg9, %arg10, %arg11)
              %c1024 = arith.constant 1024 : index
              %c1024_0 = arith.constant 1024 : index
              %c1 = arith.constant 1 : index
              %13 = loom.get_symbolic_block_size @constraints::@BM : index
              %14 = loom.get_symbolic_block_size @constraints::@BN : index
              %15 = loom.get_symbolic_block_size @constraints::@BK : index
              %16 = arith.ceildivsi %c1024_0, %15 : index
              %cst = arith.constant 0.000000e+00 : f32
              %c1024_1 = arith.constant 1024 : index
              %c0 = arith.constant 0 : index
              %17 = tensor.empty(%13, %14) : tensor<?x?xf32>
              %18 = linalg.fill ins(%cst : f32) outs(%17 : tensor<?x?xf32>) -> tensor<?x?xf32>
              %19 = arith.muli %12, %13 : index
              %20 = arith.muli %arg12, %14 : index
              %21 = scf.for %arg13 = %c0 to %16 step %c1 iter_args(%arg14 = %18) -> (tensor<?x?xf32>) {
                %25 = arith.muli %arg13, %15 : index
                %26 = arith.muli %19, %c1024_1 : index
                %27 = arith.addi %26, %25 : index
                %28 = loom.reinterpret_cast %arg0 to offset : [%27], sizes : [%13, %15], strides : [%c1024_0, %c1], reuse : [seq = false, spat = false, temp = true] : memref<*xf32> to memref<?x?xf32, strided<[?, ?], offset: ?>>
                %29 = loom.alloc(%13, %15) on @L1 : memref<?x?xf32>
                loom.copy %28, %29, interconnect : [], broadcast : [1, 1] : memref<?x?xf32, strided<[?, ?], offset: ?>> to memref<?x?xf32>
                %30 = bufferization.to_tensor %29 restrict writable : memref<?x?xf32> to tensor<?x?xf32>
                %31 = arith.muli %25, %c1024_1 : index
                %32 = arith.addi %31, %20 : index
                %33 = loom.reinterpret_cast %arg1 to offset : [%32], sizes : [%15, %14], strides : [%c1024, %c1], reuse : [seq = false, spat = true, temp = true] : memref<*xf32> to memref<?x?xf32, strided<[?, ?], offset: ?>>
                %34 = loom.alloc(%15, %14) on @L1 : memref<?x?xf32>
                loom.copy %33, %34, interconnect : [@vertical_links], broadcast : [8, 1] : memref<?x?xf32, strided<[?, ?], offset: ?>> to memref<?x?xf32>
                %35 = bufferization.to_tensor %34 restrict writable : memref<?x?xf32> to tensor<?x?xf32>
                %36 = linalg.matmul ins(%30, %35 : tensor<?x?xf32>, tensor<?x?xf32>) outs(%18 : tensor<?x?xf32>) -> tensor<?x?xf32>
                %37 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%arg14, %36 : tensor<?x?xf32>, tensor<?x?xf32>) outs(%arg14 : tensor<?x?xf32>) {
                ^bb0(%in: f32, %in_2: f32, %out: f32):
                  %38 = arith.addf %in, %in_2 : f32
                  linalg.yield %38 : f32
                } -> tensor<?x?xf32>
                scf.yield %37 : tensor<?x?xf32>
              }
              %22 = arith.muli %19, %c1024_1 : index
              %23 = arith.addi %22, %20 : index
              %24 = loom.reinterpret_cast %arg2 to offset : [%23], sizes : [%13, %14], strides : [%c1024, %c1], reuse : [seq = false, spat = false, temp = false] : memref<*xf32> to memref<?x?xf32, strided<[?, ?], offset: ?>>
              bufferization.materialize_in_destination %21 in writable %24 : (tensor<?x?xf32>, memref<?x?xf32, strided<[?, ?], offset: ?>>) -> ()
            }
          }
        } {loom.mapped_to = @y}
      } {loom.mapped_to = @x}
      return
    }
  }
  module attributes {loom.pass_name = "EnumerateCopyBroadcast"} {
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
      %15 = loom.expression(%12, %13) {coeffs = [1, 1], logic = "mul"} : index
      %16 = loom.expression(%14, %15) {coeffs = [1, 1], logic = "mul"} : index
      loom.polynomial_constraint(%12, %13, %14, %15, %16) {monomials = [{coeff = -1 : i64, vars = [4]}], upper_bound = -262144 : i64}
      %17 = loom.expression(%12, %13) {coeffs = [1, 1], logic = "add"} : index
      %18 = loom.expression(%14, %17) {coeffs = [1, 1], logic = "mul"} : index
      loom.polynomial_constraint(%12, %14, %13, %17, %18) {monomials = [{coeff = 1 : i64, vars = [4]}], upper_bound = 374784 : i64}
      loom.polynomial_constraint(%12) {monomials = [{coeff = 1 : i64, vars = [0]}], upper_bound = 16 : i64}
      %19 = loom.expression(%13, %14) {coeffs = [1, 1], logic = "mul"} : index
      loom.polynomial_constraint(%12, %14, %13, %19) {monomials = [{coeff = 1024 : i64, vars = [0]}, {coeff = 1 : i64, vars = [3]}], upper_bound = 374784 : i64}
      %20 = loom.expression(%12, %14) {coeffs = [1, 1], logic = "mul"} : index
      %21 = loom.expression(%13, %14) {coeffs = [1, 1], logic = "mul"} : index
      %22 = loom.expression(%12, %13) {coeffs = [1, 1], logic = "mul"} : index
      %23 = loom.expression(%14, %22) {coeffs = [1, 1], logic = "mul"} : index
      loom.polynomial_constraint(%12, %13, %14, %20, %21, %22, %23) {monomials = [{coeff = 64 : i64, vars = [3]}, {coeff = 64 : i64, vars = [4]}, {coeff = -1 : i64, vars = [6]}], upper_bound = 0 : i64}
    }
    func.func @matmul_kernel__d0i0_d1i0__f01__hoist_block_0__d_d(%arg0: memref<*xf32> {tt.divisibility = 16 : i32}, %arg1: memref<*xf32> {tt.divisibility = 16 : i32}, %arg2: memref<*xf32> {tt.divisibility = 16 : i32}, %arg3: index, %arg4: index, %arg5: index, %arg6: index, %arg7: index, %arg8: index) {
      affine.parallel (%arg9) = (0) to (8) {
        affine.parallel (%arg10) = (0) to (8) {
          affine.for %arg11 = 0 to affine_map<()[s0, s1] -> ((s0 ceildiv 8) ceildiv 8)>()[%arg3, %arg4] {
            affine.for %arg12 = 0 to affine_map<()[s0, s1] -> (s1)>()[%arg3, %arg4] {
              %12 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 8 + d1 * 8 + d2)>(%arg9, %arg10, %arg11)
              %c1024 = arith.constant 1024 : index
              %c1024_0 = arith.constant 1024 : index
              %c1 = arith.constant 1 : index
              %13 = loom.get_symbolic_block_size @constraints::@BM : index
              %14 = loom.get_symbolic_block_size @constraints::@BN : index
              %15 = loom.get_symbolic_block_size @constraints::@BK : index
              %16 = arith.ceildivsi %c1024_0, %15 : index
              %cst = arith.constant 0.000000e+00 : f32
              %c1024_1 = arith.constant 1024 : index
              %c0 = arith.constant 0 : index
              %17 = tensor.empty(%13, %14) : tensor<?x?xf32>
              %18 = linalg.fill ins(%cst : f32) outs(%17 : tensor<?x?xf32>) -> tensor<?x?xf32>
              %19 = arith.muli %12, %13 : index
              %20 = arith.muli %arg12, %14 : index
              %21 = arith.muli %13, %c1024_0 : index
              %c0_2 = arith.constant 0 : index
              %22 = arith.muli %19, %c1024_1 : index
              %23 = arith.muli %c0_2, %15 : index
              %24 = arith.addi %22, %23 : index
              %25 = loom.reinterpret_cast %arg0 to offset : [%24], sizes : [%16, %13, %15], strides : [%21, %c1024_0, %c1], reuse : [seq = false, spat = false, temp = true] : memref<*xf32> to memref<?x?x?xf32, strided<[?, ?, ?], offset: ?>>
              %26 = loom.alloc(%16, %13, %15) on @L1 : memref<?x?x?xf32>
              loom.copy %25, %26, interconnect : [], broadcast : [1, 1] : memref<?x?x?xf32, strided<[?, ?, ?], offset: ?>> to memref<?x?x?xf32>
              %27 = scf.for %arg13 = %c0 to %16 step %c1 iter_args(%arg14 = %18) -> (tensor<?x?xf32>) {
                %31 = arith.muli %arg13, %15 : index
                %32 = arith.muli %19, %c1024_1 : index
                %subview = memref.subview %26[%arg13, 0, 0] [1, %13, %15] [1, 1, 1] : memref<?x?x?xf32> to memref<?x?xf32, strided<[?, 1], offset: ?>>
                %33 = bufferization.to_tensor %subview restrict writable : memref<?x?xf32, strided<[?, 1], offset: ?>> to tensor<?x?xf32>
                %34 = arith.muli %31, %c1024_1 : index
                %35 = arith.addi %34, %20 : index
                %36 = loom.reinterpret_cast %arg1 to offset : [%35], sizes : [%15, %14], strides : [%c1024, %c1], reuse : [seq = false, spat = true, temp = true] : memref<*xf32> to memref<?x?xf32, strided<[?, ?], offset: ?>>
                %37 = loom.alloc(%15, %14) on @L1 : memref<?x?xf32>
                loom.copy %36, %37, interconnect : [], broadcast : [1, 1] : memref<?x?xf32, strided<[?, ?], offset: ?>> to memref<?x?xf32>
                %38 = bufferization.to_tensor %37 restrict writable : memref<?x?xf32> to tensor<?x?xf32>
                %39 = linalg.matmul ins(%33, %38 : tensor<?x?xf32>, tensor<?x?xf32>) outs(%18 : tensor<?x?xf32>) -> tensor<?x?xf32>
                %40 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%arg14, %39 : tensor<?x?xf32>, tensor<?x?xf32>) outs(%arg14 : tensor<?x?xf32>) {
                ^bb0(%in: f32, %in_3: f32, %out: f32):
                  %41 = arith.addf %in, %in_3 : f32
                  linalg.yield %41 : f32
                } -> tensor<?x?xf32>
                scf.yield %40 : tensor<?x?xf32>
              }
              %28 = arith.muli %19, %c1024_1 : index
              %29 = arith.addi %28, %20 : index
              %30 = loom.reinterpret_cast %arg2 to offset : [%29], sizes : [%13, %14], strides : [%c1024, %c1], reuse : [seq = false, spat = false, temp = false] : memref<*xf32> to memref<?x?xf32, strided<[?, ?], offset: ?>>
              bufferization.materialize_in_destination %27 in writable %30 : (tensor<?x?xf32>, memref<?x?xf32, strided<[?, ?], offset: ?>>) -> ()
            }
          }
        } {loom.mapped_to = @y}
      } {loom.mapped_to = @x}
      return
    }
  }
  module attributes {loom.pass_name = "EnumerateCopyBroadcast"} {
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
      %15 = loom.expression(%12, %13) {coeffs = [1, 1], logic = "mul"} : index
      %16 = loom.expression(%15, %14) {coeffs = [1, 1], logic = "mul"} : index
      loom.polynomial_constraint(%12, %13, %14, %15, %16) {monomials = [{coeff = -1 : i64, vars = [4]}], upper_bound = -262144 : i64}
      %17 = loom.expression(%12, %13) {coeffs = [1, 1], logic = "add"} : index
      %18 = loom.expression(%14, %17) {coeffs = [1, 1], logic = "mul"} : index
      loom.polynomial_constraint(%12, %14, %13, %17, %18) {monomials = [{coeff = 1 : i64, vars = [4]}], upper_bound = 374784 : i64}
      loom.polynomial_constraint(%12) {monomials = [{coeff = 1 : i64, vars = [0]}], upper_bound = 16 : i64}
      %19 = loom.expression(%13, %14) {coeffs = [1, 1], logic = "mul"} : index
      loom.polynomial_constraint(%12, %14, %13, %19) {monomials = [{coeff = 1024 : i64, vars = [0]}, {coeff = 1 : i64, vars = [3]}], upper_bound = 374784 : i64}
      %20 = loom.expression(%12, %14) {coeffs = [1, 1], logic = "mul"} : index
      %21 = loom.expression(%13, %14) {coeffs = [1, 1], logic = "mul"} : index
      %22 = loom.expression(%12, %13) {coeffs = [1, 1], logic = "mul"} : index
      %23 = loom.expression(%14, %22) {coeffs = [1, 1], logic = "mul"} : index
      loom.polynomial_constraint(%12, %13, %14, %20, %21, %22, %23) {monomials = [{coeff = 960 : i64, vars = [3]}, {coeff = 256 : i64, vars = [4]}, {coeff = -15 : i64, vars = [6]}], upper_bound = 0 : i64}
    }
    func.func @matmul_kernel__d0i0_d1i0__f01__hoist_block_0__d_a(%arg0: memref<*xf32> {tt.divisibility = 16 : i32}, %arg1: memref<*xf32> {tt.divisibility = 16 : i32}, %arg2: memref<*xf32> {tt.divisibility = 16 : i32}, %arg3: index, %arg4: index, %arg5: index, %arg6: index, %arg7: index, %arg8: index) {
      affine.parallel (%arg9) = (0) to (8) {
        affine.parallel (%arg10) = (0) to (8) {
          affine.for %arg11 = 0 to affine_map<()[s0, s1] -> ((s0 ceildiv 8) ceildiv 8)>()[%arg3, %arg4] {
            affine.for %arg12 = 0 to affine_map<()[s0, s1] -> (s1)>()[%arg3, %arg4] {
              %12 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 8 + d1 * 8 + d2)>(%arg9, %arg10, %arg11)
              %c1024 = arith.constant 1024 : index
              %c1024_0 = arith.constant 1024 : index
              %c1 = arith.constant 1 : index
              %13 = loom.get_symbolic_block_size @constraints::@BM : index
              %14 = loom.get_symbolic_block_size @constraints::@BN : index
              %15 = loom.get_symbolic_block_size @constraints::@BK : index
              %16 = arith.ceildivsi %c1024_0, %15 : index
              %cst = arith.constant 0.000000e+00 : f32
              %c1024_1 = arith.constant 1024 : index
              %c0 = arith.constant 0 : index
              %17 = tensor.empty(%13, %14) : tensor<?x?xf32>
              %18 = linalg.fill ins(%cst : f32) outs(%17 : tensor<?x?xf32>) -> tensor<?x?xf32>
              %19 = arith.muli %12, %13 : index
              %20 = arith.muli %arg12, %14 : index
              %21 = arith.muli %13, %c1024_0 : index
              %c0_2 = arith.constant 0 : index
              %22 = arith.muli %19, %c1024_1 : index
              %23 = arith.muli %c0_2, %15 : index
              %24 = arith.addi %22, %23 : index
              %25 = loom.reinterpret_cast %arg0 to offset : [%24], sizes : [%16, %13, %15], strides : [%21, %c1024_0, %c1], reuse : [seq = false, spat = false, temp = true] : memref<*xf32> to memref<?x?x?xf32, strided<[?, ?, ?], offset: ?>>
              %26 = loom.alloc(%16, %13, %15) on @L1 : memref<?x?x?xf32>
              loom.copy %25, %26, interconnect : [], broadcast : [1, 1] : memref<?x?x?xf32, strided<[?, ?, ?], offset: ?>> to memref<?x?x?xf32>
              %27 = scf.for %arg13 = %c0 to %16 step %c1 iter_args(%arg14 = %18) -> (tensor<?x?xf32>) {
                %31 = arith.muli %arg13, %15 : index
                %32 = arith.muli %19, %c1024_1 : index
                %subview = memref.subview %26[%arg13, 0, 0] [1, %13, %15] [1, 1, 1] : memref<?x?x?xf32> to memref<?x?xf32, strided<[?, 1], offset: ?>>
                %33 = bufferization.to_tensor %subview restrict writable : memref<?x?xf32, strided<[?, 1], offset: ?>> to tensor<?x?xf32>
                %34 = arith.muli %31, %c1024_1 : index
                %35 = arith.addi %34, %20 : index
                %36 = loom.reinterpret_cast %arg1 to offset : [%35], sizes : [%15, %14], strides : [%c1024, %c1], reuse : [seq = false, spat = true, temp = true] : memref<*xf32> to memref<?x?xf32, strided<[?, ?], offset: ?>>
                %37 = loom.alloc(%15, %14) on @L1 : memref<?x?xf32>
                loom.copy %36, %37, interconnect : [@horizontal_links, @vertical_links], broadcast : [8, 8] : memref<?x?xf32, strided<[?, ?], offset: ?>> to memref<?x?xf32>
                %38 = bufferization.to_tensor %37 restrict writable : memref<?x?xf32> to tensor<?x?xf32>
                %39 = linalg.matmul ins(%33, %38 : tensor<?x?xf32>, tensor<?x?xf32>) outs(%18 : tensor<?x?xf32>) -> tensor<?x?xf32>
                %40 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%arg14, %39 : tensor<?x?xf32>, tensor<?x?xf32>) outs(%arg14 : tensor<?x?xf32>) {
                ^bb0(%in: f32, %in_3: f32, %out: f32):
                  %41 = arith.addf %in, %in_3 : f32
                  linalg.yield %41 : f32
                } -> tensor<?x?xf32>
                scf.yield %40 : tensor<?x?xf32>
              }
              %28 = arith.muli %19, %c1024_1 : index
              %29 = arith.addi %28, %20 : index
              %30 = loom.reinterpret_cast %arg2 to offset : [%29], sizes : [%13, %14], strides : [%c1024, %c1], reuse : [seq = false, spat = false, temp = false] : memref<*xf32> to memref<?x?xf32, strided<[?, ?], offset: ?>>
              bufferization.materialize_in_destination %27 in writable %30 : (tensor<?x?xf32>, memref<?x?xf32, strided<[?, ?], offset: ?>>) -> ()
            }
          }
        } {loom.mapped_to = @y}
      } {loom.mapped_to = @x}
      return
    }
  }
  module attributes {loom.pass_name = "EnumerateCopyBroadcast"} {
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
      %15 = loom.expression(%12, %13) {coeffs = [1, 1], logic = "mul"} : index
      %16 = loom.expression(%15, %14) {coeffs = [1, 1], logic = "mul"} : index
      loom.polynomial_constraint(%12, %13, %14, %15, %16) {monomials = [{coeff = -1 : i64, vars = [4]}], upper_bound = -262144 : i64}
      %17 = loom.expression(%12, %13) {coeffs = [1, 1], logic = "add"} : index
      %18 = loom.expression(%14, %17) {coeffs = [1, 1], logic = "mul"} : index
      loom.polynomial_constraint(%12, %14, %13, %17, %18) {monomials = [{coeff = 1 : i64, vars = [4]}], upper_bound = 374784 : i64}
      loom.polynomial_constraint(%12) {monomials = [{coeff = 1 : i64, vars = [0]}], upper_bound = 16 : i64}
      %19 = loom.expression(%13, %14) {coeffs = [1, 1], logic = "mul"} : index
      loom.polynomial_constraint(%12, %14, %13, %19) {monomials = [{coeff = 1024 : i64, vars = [0]}, {coeff = 1 : i64, vars = [3]}], upper_bound = 374784 : i64}
      %20 = loom.expression(%12, %14) {coeffs = [1, 1], logic = "mul"} : index
      %21 = loom.expression(%13, %14) {coeffs = [1, 1], logic = "mul"} : index
      %22 = loom.expression(%12, %13) {coeffs = [1, 1], logic = "mul"} : index
      %23 = loom.expression(%14, %22) {coeffs = [1, 1], logic = "mul"} : index
      loom.polynomial_constraint(%12, %13, %14, %20, %21, %22, %23) {monomials = [{coeff = 960 : i64, vars = [3]}, {coeff = 256 : i64, vars = [4]}, {coeff = -15 : i64, vars = [6]}], upper_bound = 0 : i64}
    }
    func.func @matmul_kernel__d0i0_d1i0__f01__hoist_block_0__d_h(%arg0: memref<*xf32> {tt.divisibility = 16 : i32}, %arg1: memref<*xf32> {tt.divisibility = 16 : i32}, %arg2: memref<*xf32> {tt.divisibility = 16 : i32}, %arg3: index, %arg4: index, %arg5: index, %arg6: index, %arg7: index, %arg8: index) {
      affine.parallel (%arg9) = (0) to (8) {
        affine.parallel (%arg10) = (0) to (8) {
          affine.for %arg11 = 0 to affine_map<()[s0, s1] -> ((s0 ceildiv 8) ceildiv 8)>()[%arg3, %arg4] {
            affine.for %arg12 = 0 to affine_map<()[s0, s1] -> (s1)>()[%arg3, %arg4] {
              %12 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 8 + d1 * 8 + d2)>(%arg9, %arg10, %arg11)
              %c1024 = arith.constant 1024 : index
              %c1024_0 = arith.constant 1024 : index
              %c1 = arith.constant 1 : index
              %13 = loom.get_symbolic_block_size @constraints::@BM : index
              %14 = loom.get_symbolic_block_size @constraints::@BN : index
              %15 = loom.get_symbolic_block_size @constraints::@BK : index
              %16 = arith.ceildivsi %c1024_0, %15 : index
              %cst = arith.constant 0.000000e+00 : f32
              %c1024_1 = arith.constant 1024 : index
              %c0 = arith.constant 0 : index
              %17 = tensor.empty(%13, %14) : tensor<?x?xf32>
              %18 = linalg.fill ins(%cst : f32) outs(%17 : tensor<?x?xf32>) -> tensor<?x?xf32>
              %19 = arith.muli %12, %13 : index
              %20 = arith.muli %arg12, %14 : index
              %21 = arith.muli %13, %c1024_0 : index
              %c0_2 = arith.constant 0 : index
              %22 = arith.muli %19, %c1024_1 : index
              %23 = arith.muli %c0_2, %15 : index
              %24 = arith.addi %22, %23 : index
              %25 = loom.reinterpret_cast %arg0 to offset : [%24], sizes : [%16, %13, %15], strides : [%21, %c1024_0, %c1], reuse : [seq = false, spat = false, temp = true] : memref<*xf32> to memref<?x?x?xf32, strided<[?, ?, ?], offset: ?>>
              %26 = loom.alloc(%16, %13, %15) on @L1 : memref<?x?x?xf32>
              loom.copy %25, %26, interconnect : [], broadcast : [1, 1] : memref<?x?x?xf32, strided<[?, ?, ?], offset: ?>> to memref<?x?x?xf32>
              %27 = scf.for %arg13 = %c0 to %16 step %c1 iter_args(%arg14 = %18) -> (tensor<?x?xf32>) {
                %31 = arith.muli %arg13, %15 : index
                %32 = arith.muli %19, %c1024_1 : index
                %subview = memref.subview %26[%arg13, 0, 0] [1, %13, %15] [1, 1, 1] : memref<?x?x?xf32> to memref<?x?xf32, strided<[?, 1], offset: ?>>
                %33 = bufferization.to_tensor %subview restrict writable : memref<?x?xf32, strided<[?, 1], offset: ?>> to tensor<?x?xf32>
                %34 = arith.muli %31, %c1024_1 : index
                %35 = arith.addi %34, %20 : index
                %36 = loom.reinterpret_cast %arg1 to offset : [%35], sizes : [%15, %14], strides : [%c1024, %c1], reuse : [seq = false, spat = true, temp = true] : memref<*xf32> to memref<?x?xf32, strided<[?, ?], offset: ?>>
                %37 = loom.alloc(%15, %14) on @L1 : memref<?x?xf32>
                loom.copy %36, %37, interconnect : [@horizontal_links], broadcast : [1, 8] : memref<?x?xf32, strided<[?, ?], offset: ?>> to memref<?x?xf32>
                %38 = bufferization.to_tensor %37 restrict writable : memref<?x?xf32> to tensor<?x?xf32>
                %39 = linalg.matmul ins(%33, %38 : tensor<?x?xf32>, tensor<?x?xf32>) outs(%18 : tensor<?x?xf32>) -> tensor<?x?xf32>
                %40 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%arg14, %39 : tensor<?x?xf32>, tensor<?x?xf32>) outs(%arg14 : tensor<?x?xf32>) {
                ^bb0(%in: f32, %in_3: f32, %out: f32):
                  %41 = arith.addf %in, %in_3 : f32
                  linalg.yield %41 : f32
                } -> tensor<?x?xf32>
                scf.yield %40 : tensor<?x?xf32>
              }
              %28 = arith.muli %19, %c1024_1 : index
              %29 = arith.addi %28, %20 : index
              %30 = loom.reinterpret_cast %arg2 to offset : [%29], sizes : [%13, %14], strides : [%c1024, %c1], reuse : [seq = false, spat = false, temp = false] : memref<*xf32> to memref<?x?xf32, strided<[?, ?], offset: ?>>
              bufferization.materialize_in_destination %27 in writable %30 : (tensor<?x?xf32>, memref<?x?xf32, strided<[?, ?], offset: ?>>) -> ()
            }
          }
        } {loom.mapped_to = @y}
      } {loom.mapped_to = @x}
      return
    }
  }
  module attributes {loom.pass_name = "EnumerateCopyBroadcast"} {
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
      %15 = loom.expression(%12, %13) {coeffs = [1, 1], logic = "mul"} : index
      %16 = loom.expression(%15, %14) {coeffs = [1, 1], logic = "mul"} : index
      loom.polynomial_constraint(%12, %13, %14, %15, %16) {monomials = [{coeff = -1 : i64, vars = [4]}], upper_bound = -262144 : i64}
      %17 = loom.expression(%12, %13) {coeffs = [1, 1], logic = "add"} : index
      %18 = loom.expression(%14, %17) {coeffs = [1, 1], logic = "mul"} : index
      loom.polynomial_constraint(%12, %14, %13, %17, %18) {monomials = [{coeff = 1 : i64, vars = [4]}], upper_bound = 374784 : i64}
      loom.polynomial_constraint(%12) {monomials = [{coeff = 1 : i64, vars = [0]}], upper_bound = 16 : i64}
      %19 = loom.expression(%13, %14) {coeffs = [1, 1], logic = "mul"} : index
      loom.polynomial_constraint(%12, %14, %13, %19) {monomials = [{coeff = 1024 : i64, vars = [0]}, {coeff = 1 : i64, vars = [3]}], upper_bound = 374784 : i64}
      %20 = loom.expression(%12, %14) {coeffs = [1, 1], logic = "mul"} : index
      %21 = loom.expression(%13, %14) {coeffs = [1, 1], logic = "mul"} : index
      %22 = loom.expression(%12, %13) {coeffs = [1, 1], logic = "mul"} : index
      %23 = loom.expression(%14, %22) {coeffs = [1, 1], logic = "mul"} : index
      loom.polynomial_constraint(%12, %13, %14, %20, %21, %22, %23) {monomials = [{coeff = 960 : i64, vars = [3]}, {coeff = 256 : i64, vars = [4]}, {coeff = -15 : i64, vars = [6]}], upper_bound = 0 : i64}
    }
    func.func @matmul_kernel__d0i0_d1i0__f01__hoist_block_0__d_v(%arg0: memref<*xf32> {tt.divisibility = 16 : i32}, %arg1: memref<*xf32> {tt.divisibility = 16 : i32}, %arg2: memref<*xf32> {tt.divisibility = 16 : i32}, %arg3: index, %arg4: index, %arg5: index, %arg6: index, %arg7: index, %arg8: index) {
      affine.parallel (%arg9) = (0) to (8) {
        affine.parallel (%arg10) = (0) to (8) {
          affine.for %arg11 = 0 to affine_map<()[s0, s1] -> ((s0 ceildiv 8) ceildiv 8)>()[%arg3, %arg4] {
            affine.for %arg12 = 0 to affine_map<()[s0, s1] -> (s1)>()[%arg3, %arg4] {
              %12 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 8 + d1 * 8 + d2)>(%arg9, %arg10, %arg11)
              %c1024 = arith.constant 1024 : index
              %c1024_0 = arith.constant 1024 : index
              %c1 = arith.constant 1 : index
              %13 = loom.get_symbolic_block_size @constraints::@BM : index
              %14 = loom.get_symbolic_block_size @constraints::@BN : index
              %15 = loom.get_symbolic_block_size @constraints::@BK : index
              %16 = arith.ceildivsi %c1024_0, %15 : index
              %cst = arith.constant 0.000000e+00 : f32
              %c1024_1 = arith.constant 1024 : index
              %c0 = arith.constant 0 : index
              %17 = tensor.empty(%13, %14) : tensor<?x?xf32>
              %18 = linalg.fill ins(%cst : f32) outs(%17 : tensor<?x?xf32>) -> tensor<?x?xf32>
              %19 = arith.muli %12, %13 : index
              %20 = arith.muli %arg12, %14 : index
              %21 = arith.muli %13, %c1024_0 : index
              %c0_2 = arith.constant 0 : index
              %22 = arith.muli %19, %c1024_1 : index
              %23 = arith.muli %c0_2, %15 : index
              %24 = arith.addi %22, %23 : index
              %25 = loom.reinterpret_cast %arg0 to offset : [%24], sizes : [%16, %13, %15], strides : [%21, %c1024_0, %c1], reuse : [seq = false, spat = false, temp = true] : memref<*xf32> to memref<?x?x?xf32, strided<[?, ?, ?], offset: ?>>
              %26 = loom.alloc(%16, %13, %15) on @L1 : memref<?x?x?xf32>
              loom.copy %25, %26, interconnect : [], broadcast : [1, 1] : memref<?x?x?xf32, strided<[?, ?, ?], offset: ?>> to memref<?x?x?xf32>
              %27 = scf.for %arg13 = %c0 to %16 step %c1 iter_args(%arg14 = %18) -> (tensor<?x?xf32>) {
                %31 = arith.muli %arg13, %15 : index
                %32 = arith.muli %19, %c1024_1 : index
                %subview = memref.subview %26[%arg13, 0, 0] [1, %13, %15] [1, 1, 1] : memref<?x?x?xf32> to memref<?x?xf32, strided<[?, 1], offset: ?>>
                %33 = bufferization.to_tensor %subview restrict writable : memref<?x?xf32, strided<[?, 1], offset: ?>> to tensor<?x?xf32>
                %34 = arith.muli %31, %c1024_1 : index
                %35 = arith.addi %34, %20 : index
                %36 = loom.reinterpret_cast %arg1 to offset : [%35], sizes : [%15, %14], strides : [%c1024, %c1], reuse : [seq = false, spat = true, temp = true] : memref<*xf32> to memref<?x?xf32, strided<[?, ?], offset: ?>>
                %37 = loom.alloc(%15, %14) on @L1 : memref<?x?xf32>
                loom.copy %36, %37, interconnect : [@vertical_links], broadcast : [8, 1] : memref<?x?xf32, strided<[?, ?], offset: ?>> to memref<?x?xf32>
                %38 = bufferization.to_tensor %37 restrict writable : memref<?x?xf32> to tensor<?x?xf32>
                %39 = linalg.matmul ins(%33, %38 : tensor<?x?xf32>, tensor<?x?xf32>) outs(%18 : tensor<?x?xf32>) -> tensor<?x?xf32>
                %40 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%arg14, %39 : tensor<?x?xf32>, tensor<?x?xf32>) outs(%arg14 : tensor<?x?xf32>) {
                ^bb0(%in: f32, %in_3: f32, %out: f32):
                  %41 = arith.addf %in, %in_3 : f32
                  linalg.yield %41 : f32
                } -> tensor<?x?xf32>
                scf.yield %40 : tensor<?x?xf32>
              }
              %28 = arith.muli %19, %c1024_1 : index
              %29 = arith.addi %28, %20 : index
              %30 = loom.reinterpret_cast %arg2 to offset : [%29], sizes : [%13, %14], strides : [%c1024, %c1], reuse : [seq = false, spat = false, temp = false] : memref<*xf32> to memref<?x?xf32, strided<[?, ?], offset: ?>>
              bufferization.materialize_in_destination %27 in writable %30 : (tensor<?x?xf32>, memref<?x?xf32, strided<[?, ?], offset: ?>>) -> ()
            }
          }
        } {loom.mapped_to = @y}
      } {loom.mapped_to = @x}
      return
    }
  }
  module attributes {loom.pass_name = "EnumerateCopyBroadcast"} {
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
      %15 = loom.expression(%12, %13) {coeffs = [1, 1], logic = "mul"} : index
      %16 = loom.expression(%14, %15) {coeffs = [1, 1], logic = "mul"} : index
      loom.polynomial_constraint(%12, %13, %14, %15, %16) {monomials = [{coeff = -1 : i64, vars = [4]}], upper_bound = -262144 : i64}
      %17 = loom.expression(%12, %13) {coeffs = [1, 1], logic = "add"} : index
      %18 = loom.expression(%14, %17) {coeffs = [1, 1], logic = "mul"} : index
      loom.polynomial_constraint(%12, %14, %13, %17, %18) {monomials = [{coeff = 1 : i64, vars = [4]}], upper_bound = 374784 : i64}
      loom.polynomial_constraint(%12) {monomials = [{coeff = 1 : i64, vars = [0]}], upper_bound = 16 : i64}
      %19 = loom.expression(%12, %14) {coeffs = [1, 1], logic = "mul"} : index
      loom.polynomial_constraint(%13, %12, %14, %19) {monomials = [{coeff = 1024 : i64, vars = [0]}, {coeff = 1 : i64, vars = [3]}], upper_bound = 374784 : i64}
      %20 = loom.expression(%12, %14) {coeffs = [1, 1], logic = "mul"} : index
      %21 = loom.expression(%13, %14) {coeffs = [1, 1], logic = "mul"} : index
      %22 = loom.expression(%12, %13) {coeffs = [1, 1], logic = "mul"} : index
      %23 = loom.expression(%14, %22) {coeffs = [1, 1], logic = "mul"} : index
      loom.polynomial_constraint(%12, %13, %14, %20, %21, %22, %23) {monomials = [{coeff = 64 : i64, vars = [3]}, {coeff = 64 : i64, vars = [4]}, {coeff = -1 : i64, vars = [6]}], upper_bound = 0 : i64}
    }
    func.func @matmul_kernel__d0i0_d1i0__f01__hoist_block_1__d_d(%arg0: memref<*xf32> {tt.divisibility = 16 : i32}, %arg1: memref<*xf32> {tt.divisibility = 16 : i32}, %arg2: memref<*xf32> {tt.divisibility = 16 : i32}, %arg3: index, %arg4: index, %arg5: index, %arg6: index, %arg7: index, %arg8: index) {
      affine.parallel (%arg9) = (0) to (8) {
        affine.parallel (%arg10) = (0) to (8) {
          affine.for %arg11 = 0 to affine_map<()[s0, s1] -> ((s0 ceildiv 8) ceildiv 8)>()[%arg3, %arg4] {
            affine.for %arg12 = 0 to affine_map<()[s0, s1] -> (s1)>()[%arg3, %arg4] {
              %12 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 8 + d1 * 8 + d2)>(%arg9, %arg10, %arg11)
              %c1024 = arith.constant 1024 : index
              %c1024_0 = arith.constant 1024 : index
              %c1 = arith.constant 1 : index
              %13 = loom.get_symbolic_block_size @constraints::@BM : index
              %14 = loom.get_symbolic_block_size @constraints::@BN : index
              %15 = loom.get_symbolic_block_size @constraints::@BK : index
              %16 = arith.ceildivsi %c1024_0, %15 : index
              %cst = arith.constant 0.000000e+00 : f32
              %c1024_1 = arith.constant 1024 : index
              %c0 = arith.constant 0 : index
              %17 = tensor.empty(%13, %14) : tensor<?x?xf32>
              %18 = linalg.fill ins(%cst : f32) outs(%17 : tensor<?x?xf32>) -> tensor<?x?xf32>
              %19 = arith.muli %12, %13 : index
              %20 = arith.muli %arg12, %14 : index
              %21 = arith.muli %15, %c1024 : index
              %c0_2 = arith.constant 0 : index
              %22 = arith.muli %c0_2, %15 : index
              %23 = arith.muli %22, %c1024_1 : index
              %24 = arith.addi %23, %20 : index
              %25 = loom.reinterpret_cast %arg1 to offset : [%24], sizes : [%16, %15, %14], strides : [%21, %c1024, %c1], reuse : [seq = false, spat = true, temp = true] : memref<*xf32> to memref<?x?x?xf32, strided<[?, ?, ?], offset: ?>>
              %26 = loom.alloc(%16, %15, %14) on @L1 : memref<?x?x?xf32>
              loom.copy %25, %26, interconnect : [], broadcast : [1, 1] : memref<?x?x?xf32, strided<[?, ?, ?], offset: ?>> to memref<?x?x?xf32>
              %27 = scf.for %arg13 = %c0 to %16 step %c1 iter_args(%arg14 = %18) -> (tensor<?x?xf32>) {
                %31 = arith.muli %arg13, %15 : index
                %32 = arith.muli %19, %c1024_1 : index
                %33 = arith.addi %32, %31 : index
                %34 = loom.reinterpret_cast %arg0 to offset : [%33], sizes : [%13, %15], strides : [%c1024_0, %c1], reuse : [seq = false, spat = false, temp = true] : memref<*xf32> to memref<?x?xf32, strided<[?, ?], offset: ?>>
                %35 = loom.alloc(%13, %15) on @L1 : memref<?x?xf32>
                loom.copy %34, %35, interconnect : [], broadcast : [1, 1] : memref<?x?xf32, strided<[?, ?], offset: ?>> to memref<?x?xf32>
                %36 = bufferization.to_tensor %35 restrict writable : memref<?x?xf32> to tensor<?x?xf32>
                %37 = arith.muli %31, %c1024_1 : index
                %subview = memref.subview %26[%arg13, 0, 0] [1, %15, %14] [1, 1, 1] : memref<?x?x?xf32> to memref<?x?xf32, strided<[?, 1], offset: ?>>
                %38 = bufferization.to_tensor %subview restrict writable : memref<?x?xf32, strided<[?, 1], offset: ?>> to tensor<?x?xf32>
                %39 = linalg.matmul ins(%36, %38 : tensor<?x?xf32>, tensor<?x?xf32>) outs(%18 : tensor<?x?xf32>) -> tensor<?x?xf32>
                %40 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%arg14, %39 : tensor<?x?xf32>, tensor<?x?xf32>) outs(%arg14 : tensor<?x?xf32>) {
                ^bb0(%in: f32, %in_3: f32, %out: f32):
                  %41 = arith.addf %in, %in_3 : f32
                  linalg.yield %41 : f32
                } -> tensor<?x?xf32>
                scf.yield %40 : tensor<?x?xf32>
              }
              %28 = arith.muli %19, %c1024_1 : index
              %29 = arith.addi %28, %20 : index
              %30 = loom.reinterpret_cast %arg2 to offset : [%29], sizes : [%13, %14], strides : [%c1024, %c1], reuse : [seq = false, spat = false, temp = false] : memref<*xf32> to memref<?x?xf32, strided<[?, ?], offset: ?>>
              bufferization.materialize_in_destination %27 in writable %30 : (tensor<?x?xf32>, memref<?x?xf32, strided<[?, ?], offset: ?>>) -> ()
            }
          }
        } {loom.mapped_to = @y}
      } {loom.mapped_to = @x}
      return
    }
  }
  module attributes {loom.pass_name = "EnumerateCopyBroadcast"} {
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
      %15 = loom.expression(%13, %12) {coeffs = [1, 1], logic = "mul"} : index
      %16 = loom.expression(%15, %14) {coeffs = [1, 1], logic = "mul"} : index
      loom.polynomial_constraint(%12, %13, %14, %15, %16) {monomials = [{coeff = -1 : i64, vars = [4]}], upper_bound = -262144 : i64}
      %17 = loom.expression(%12, %13) {coeffs = [1, 1], logic = "add"} : index
      %18 = loom.expression(%17, %14) {coeffs = [1, 1], logic = "mul"} : index
      loom.polynomial_constraint(%12, %14, %13, %17, %18) {monomials = [{coeff = 1 : i64, vars = [4]}], upper_bound = 374784 : i64}
      loom.polynomial_constraint(%12) {monomials = [{coeff = 1 : i64, vars = [0]}], upper_bound = 16 : i64}
      %19 = loom.expression(%14, %12) {coeffs = [1, 1], logic = "mul"} : index
      loom.polynomial_constraint(%13, %12, %14, %19) {monomials = [{coeff = 1024 : i64, vars = [0]}, {coeff = 1 : i64, vars = [3]}], upper_bound = 374784 : i64}
      %20 = loom.expression(%14, %12) {coeffs = [1, 1], logic = "mul"} : index
      %21 = loom.expression(%13, %14) {coeffs = [1, 1], logic = "mul"} : index
      %22 = loom.expression(%13, %12) {coeffs = [1, 1], logic = "mul"} : index
      %23 = loom.expression(%14, %22) {coeffs = [1, 1], logic = "mul"} : index
      loom.polynomial_constraint(%12, %13, %14, %20, %21, %22, %23) {monomials = [{coeff = 256 : i64, vars = [3]}, {coeff = 960 : i64, vars = [4]}, {coeff = -15 : i64, vars = [6]}], upper_bound = 0 : i64}
    }
    func.func @matmul_kernel__d0i0_d1i0__f01__hoist_block_1__a_d(%arg0: memref<*xf32> {tt.divisibility = 16 : i32}, %arg1: memref<*xf32> {tt.divisibility = 16 : i32}, %arg2: memref<*xf32> {tt.divisibility = 16 : i32}, %arg3: index, %arg4: index, %arg5: index, %arg6: index, %arg7: index, %arg8: index) {
      affine.parallel (%arg9) = (0) to (8) {
        affine.parallel (%arg10) = (0) to (8) {
          affine.for %arg11 = 0 to affine_map<()[s0, s1] -> ((s0 ceildiv 8) ceildiv 8)>()[%arg3, %arg4] {
            affine.for %arg12 = 0 to affine_map<()[s0, s1] -> (s1)>()[%arg3, %arg4] {
              %12 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 8 + d1 * 8 + d2)>(%arg9, %arg10, %arg11)
              %c1024 = arith.constant 1024 : index
              %c1024_0 = arith.constant 1024 : index
              %c1 = arith.constant 1 : index
              %13 = loom.get_symbolic_block_size @constraints::@BM : index
              %14 = loom.get_symbolic_block_size @constraints::@BN : index
              %15 = loom.get_symbolic_block_size @constraints::@BK : index
              %16 = arith.ceildivsi %c1024_0, %15 : index
              %cst = arith.constant 0.000000e+00 : f32
              %c1024_1 = arith.constant 1024 : index
              %c0 = arith.constant 0 : index
              %17 = tensor.empty(%13, %14) : tensor<?x?xf32>
              %18 = linalg.fill ins(%cst : f32) outs(%17 : tensor<?x?xf32>) -> tensor<?x?xf32>
              %19 = arith.muli %12, %13 : index
              %20 = arith.muli %arg12, %14 : index
              %21 = arith.muli %15, %c1024 : index
              %c0_2 = arith.constant 0 : index
              %22 = arith.muli %c0_2, %15 : index
              %23 = arith.muli %22, %c1024_1 : index
              %24 = arith.addi %23, %20 : index
              %25 = loom.reinterpret_cast %arg1 to offset : [%24], sizes : [%16, %15, %14], strides : [%21, %c1024, %c1], reuse : [seq = false, spat = true, temp = true] : memref<*xf32> to memref<?x?x?xf32, strided<[?, ?, ?], offset: ?>>
              %26 = loom.alloc(%16, %15, %14) on @L1 : memref<?x?x?xf32>
              loom.copy %25, %26, interconnect : [@horizontal_links, @vertical_links], broadcast : [8, 8] : memref<?x?x?xf32, strided<[?, ?, ?], offset: ?>> to memref<?x?x?xf32>
              %27 = scf.for %arg13 = %c0 to %16 step %c1 iter_args(%arg14 = %18) -> (tensor<?x?xf32>) {
                %31 = arith.muli %arg13, %15 : index
                %32 = arith.muli %19, %c1024_1 : index
                %33 = arith.addi %32, %31 : index
                %34 = loom.reinterpret_cast %arg0 to offset : [%33], sizes : [%13, %15], strides : [%c1024_0, %c1], reuse : [seq = false, spat = false, temp = true] : memref<*xf32> to memref<?x?xf32, strided<[?, ?], offset: ?>>
                %35 = loom.alloc(%13, %15) on @L1 : memref<?x?xf32>
                loom.copy %34, %35, interconnect : [], broadcast : [1, 1] : memref<?x?xf32, strided<[?, ?], offset: ?>> to memref<?x?xf32>
                %36 = bufferization.to_tensor %35 restrict writable : memref<?x?xf32> to tensor<?x?xf32>
                %37 = arith.muli %31, %c1024_1 : index
                %subview = memref.subview %26[%arg13, 0, 0] [1, %15, %14] [1, 1, 1] : memref<?x?x?xf32> to memref<?x?xf32, strided<[?, 1], offset: ?>>
                %38 = bufferization.to_tensor %subview restrict writable : memref<?x?xf32, strided<[?, 1], offset: ?>> to tensor<?x?xf32>
                %39 = linalg.matmul ins(%36, %38 : tensor<?x?xf32>, tensor<?x?xf32>) outs(%18 : tensor<?x?xf32>) -> tensor<?x?xf32>
                %40 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%arg14, %39 : tensor<?x?xf32>, tensor<?x?xf32>) outs(%arg14 : tensor<?x?xf32>) {
                ^bb0(%in: f32, %in_3: f32, %out: f32):
                  %41 = arith.addf %in, %in_3 : f32
                  linalg.yield %41 : f32
                } -> tensor<?x?xf32>
                scf.yield %40 : tensor<?x?xf32>
              }
              %28 = arith.muli %19, %c1024_1 : index
              %29 = arith.addi %28, %20 : index
              %30 = loom.reinterpret_cast %arg2 to offset : [%29], sizes : [%13, %14], strides : [%c1024, %c1], reuse : [seq = false, spat = false, temp = false] : memref<*xf32> to memref<?x?xf32, strided<[?, ?], offset: ?>>
              bufferization.materialize_in_destination %27 in writable %30 : (tensor<?x?xf32>, memref<?x?xf32, strided<[?, ?], offset: ?>>) -> ()
            }
          }
        } {loom.mapped_to = @y}
      } {loom.mapped_to = @x}
      return
    }
  }
  module attributes {loom.pass_name = "EnumerateCopyBroadcast"} {
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
      %15 = loom.expression(%13, %12) {coeffs = [1, 1], logic = "mul"} : index
      %16 = loom.expression(%14, %15) {coeffs = [1, 1], logic = "mul"} : index
      loom.polynomial_constraint(%12, %13, %14, %15, %16) {monomials = [{coeff = -1 : i64, vars = [4]}], upper_bound = -262144 : i64}
      %17 = loom.expression(%12, %13) {coeffs = [1, 1], logic = "add"} : index
      %18 = loom.expression(%14, %17) {coeffs = [1, 1], logic = "mul"} : index
      loom.polynomial_constraint(%12, %14, %13, %17, %18) {monomials = [{coeff = 1 : i64, vars = [4]}], upper_bound = 374784 : i64}
      loom.polynomial_constraint(%12) {monomials = [{coeff = 1 : i64, vars = [0]}], upper_bound = 16 : i64}
      %19 = loom.expression(%14, %12) {coeffs = [1, 1], logic = "mul"} : index
      loom.polynomial_constraint(%13, %12, %14, %19) {monomials = [{coeff = 1024 : i64, vars = [0]}, {coeff = 1 : i64, vars = [3]}], upper_bound = 374784 : i64}
      %20 = loom.expression(%14, %12) {coeffs = [1, 1], logic = "mul"} : index
      %21 = loom.expression(%13, %14) {coeffs = [1, 1], logic = "mul"} : index
      %22 = loom.expression(%13, %12) {coeffs = [1, 1], logic = "mul"} : index
      %23 = loom.expression(%14, %22) {coeffs = [1, 1], logic = "mul"} : index
      loom.polynomial_constraint(%12, %13, %14, %20, %21, %22, %23) {monomials = [{coeff = 256 : i64, vars = [3]}, {coeff = 960 : i64, vars = [4]}, {coeff = -15 : i64, vars = [6]}], upper_bound = 0 : i64}
    }
    func.func @matmul_kernel__d0i0_d1i0__f01__hoist_block_1__h_d(%arg0: memref<*xf32> {tt.divisibility = 16 : i32}, %arg1: memref<*xf32> {tt.divisibility = 16 : i32}, %arg2: memref<*xf32> {tt.divisibility = 16 : i32}, %arg3: index, %arg4: index, %arg5: index, %arg6: index, %arg7: index, %arg8: index) {
      affine.parallel (%arg9) = (0) to (8) {
        affine.parallel (%arg10) = (0) to (8) {
          affine.for %arg11 = 0 to affine_map<()[s0, s1] -> ((s0 ceildiv 8) ceildiv 8)>()[%arg3, %arg4] {
            affine.for %arg12 = 0 to affine_map<()[s0, s1] -> (s1)>()[%arg3, %arg4] {
              %12 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 8 + d1 * 8 + d2)>(%arg9, %arg10, %arg11)
              %c1024 = arith.constant 1024 : index
              %c1024_0 = arith.constant 1024 : index
              %c1 = arith.constant 1 : index
              %13 = loom.get_symbolic_block_size @constraints::@BM : index
              %14 = loom.get_symbolic_block_size @constraints::@BN : index
              %15 = loom.get_symbolic_block_size @constraints::@BK : index
              %16 = arith.ceildivsi %c1024_0, %15 : index
              %cst = arith.constant 0.000000e+00 : f32
              %c1024_1 = arith.constant 1024 : index
              %c0 = arith.constant 0 : index
              %17 = tensor.empty(%13, %14) : tensor<?x?xf32>
              %18 = linalg.fill ins(%cst : f32) outs(%17 : tensor<?x?xf32>) -> tensor<?x?xf32>
              %19 = arith.muli %12, %13 : index
              %20 = arith.muli %arg12, %14 : index
              %21 = arith.muli %15, %c1024 : index
              %c0_2 = arith.constant 0 : index
              %22 = arith.muli %c0_2, %15 : index
              %23 = arith.muli %22, %c1024_1 : index
              %24 = arith.addi %23, %20 : index
              %25 = loom.reinterpret_cast %arg1 to offset : [%24], sizes : [%16, %15, %14], strides : [%21, %c1024, %c1], reuse : [seq = false, spat = true, temp = true] : memref<*xf32> to memref<?x?x?xf32, strided<[?, ?, ?], offset: ?>>
              %26 = loom.alloc(%16, %15, %14) on @L1 : memref<?x?x?xf32>
              loom.copy %25, %26, interconnect : [@horizontal_links], broadcast : [1, 8] : memref<?x?x?xf32, strided<[?, ?, ?], offset: ?>> to memref<?x?x?xf32>
              %27 = scf.for %arg13 = %c0 to %16 step %c1 iter_args(%arg14 = %18) -> (tensor<?x?xf32>) {
                %31 = arith.muli %arg13, %15 : index
                %32 = arith.muli %19, %c1024_1 : index
                %33 = arith.addi %32, %31 : index
                %34 = loom.reinterpret_cast %arg0 to offset : [%33], sizes : [%13, %15], strides : [%c1024_0, %c1], reuse : [seq = false, spat = false, temp = true] : memref<*xf32> to memref<?x?xf32, strided<[?, ?], offset: ?>>
                %35 = loom.alloc(%13, %15) on @L1 : memref<?x?xf32>
                loom.copy %34, %35, interconnect : [], broadcast : [1, 1] : memref<?x?xf32, strided<[?, ?], offset: ?>> to memref<?x?xf32>
                %36 = bufferization.to_tensor %35 restrict writable : memref<?x?xf32> to tensor<?x?xf32>
                %37 = arith.muli %31, %c1024_1 : index
                %subview = memref.subview %26[%arg13, 0, 0] [1, %15, %14] [1, 1, 1] : memref<?x?x?xf32> to memref<?x?xf32, strided<[?, 1], offset: ?>>
                %38 = bufferization.to_tensor %subview restrict writable : memref<?x?xf32, strided<[?, 1], offset: ?>> to tensor<?x?xf32>
                %39 = linalg.matmul ins(%36, %38 : tensor<?x?xf32>, tensor<?x?xf32>) outs(%18 : tensor<?x?xf32>) -> tensor<?x?xf32>
                %40 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%arg14, %39 : tensor<?x?xf32>, tensor<?x?xf32>) outs(%arg14 : tensor<?x?xf32>) {
                ^bb0(%in: f32, %in_3: f32, %out: f32):
                  %41 = arith.addf %in, %in_3 : f32
                  linalg.yield %41 : f32
                } -> tensor<?x?xf32>
                scf.yield %40 : tensor<?x?xf32>
              }
              %28 = arith.muli %19, %c1024_1 : index
              %29 = arith.addi %28, %20 : index
              %30 = loom.reinterpret_cast %arg2 to offset : [%29], sizes : [%13, %14], strides : [%c1024, %c1], reuse : [seq = false, spat = false, temp = false] : memref<*xf32> to memref<?x?xf32, strided<[?, ?], offset: ?>>
              bufferization.materialize_in_destination %27 in writable %30 : (tensor<?x?xf32>, memref<?x?xf32, strided<[?, ?], offset: ?>>) -> ()
            }
          }
        } {loom.mapped_to = @y}
      } {loom.mapped_to = @x}
      return
    }
  }
  module attributes {loom.pass_name = "EnumerateCopyBroadcast"} {
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
      %15 = loom.expression(%12, %13) {coeffs = [1, 1], logic = "mul"} : index
      %16 = loom.expression(%15, %14) {coeffs = [1, 1], logic = "mul"} : index
      loom.polynomial_constraint(%12, %13, %14, %15, %16) {monomials = [{coeff = -1 : i64, vars = [4]}], upper_bound = -262144 : i64}
      %17 = loom.expression(%12, %13) {coeffs = [1, 1], logic = "add"} : index
      %18 = loom.expression(%17, %14) {coeffs = [1, 1], logic = "mul"} : index
      loom.polynomial_constraint(%12, %14, %13, %17, %18) {monomials = [{coeff = 1 : i64, vars = [4]}], upper_bound = 374784 : i64}
      loom.polynomial_constraint(%12) {monomials = [{coeff = 1 : i64, vars = [0]}], upper_bound = 16 : i64}
      %19 = loom.expression(%12, %14) {coeffs = [1, 1], logic = "mul"} : index
      loom.polynomial_constraint(%13, %12, %14, %19) {monomials = [{coeff = 1024 : i64, vars = [0]}, {coeff = 1 : i64, vars = [3]}], upper_bound = 374784 : i64}
      %20 = loom.expression(%12, %14) {coeffs = [1, 1], logic = "mul"} : index
      %21 = loom.expression(%13, %14) {coeffs = [1, 1], logic = "mul"} : index
      %22 = loom.expression(%12, %13) {coeffs = [1, 1], logic = "mul"} : index
      %23 = loom.expression(%14, %22) {coeffs = [1, 1], logic = "mul"} : index
      loom.polynomial_constraint(%12, %13, %14, %20, %21, %22, %23) {monomials = [{coeff = 256 : i64, vars = [3]}, {coeff = 960 : i64, vars = [4]}, {coeff = -15 : i64, vars = [6]}], upper_bound = 0 : i64}
    }
    func.func @matmul_kernel__d0i0_d1i0__f01__hoist_block_1__v_d(%arg0: memref<*xf32> {tt.divisibility = 16 : i32}, %arg1: memref<*xf32> {tt.divisibility = 16 : i32}, %arg2: memref<*xf32> {tt.divisibility = 16 : i32}, %arg3: index, %arg4: index, %arg5: index, %arg6: index, %arg7: index, %arg8: index) {
      affine.parallel (%arg9) = (0) to (8) {
        affine.parallel (%arg10) = (0) to (8) {
          affine.for %arg11 = 0 to affine_map<()[s0, s1] -> ((s0 ceildiv 8) ceildiv 8)>()[%arg3, %arg4] {
            affine.for %arg12 = 0 to affine_map<()[s0, s1] -> (s1)>()[%arg3, %arg4] {
              %12 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 8 + d1 * 8 + d2)>(%arg9, %arg10, %arg11)
              %c1024 = arith.constant 1024 : index
              %c1024_0 = arith.constant 1024 : index
              %c1 = arith.constant 1 : index
              %13 = loom.get_symbolic_block_size @constraints::@BM : index
              %14 = loom.get_symbolic_block_size @constraints::@BN : index
              %15 = loom.get_symbolic_block_size @constraints::@BK : index
              %16 = arith.ceildivsi %c1024_0, %15 : index
              %cst = arith.constant 0.000000e+00 : f32
              %c1024_1 = arith.constant 1024 : index
              %c0 = arith.constant 0 : index
              %17 = tensor.empty(%13, %14) : tensor<?x?xf32>
              %18 = linalg.fill ins(%cst : f32) outs(%17 : tensor<?x?xf32>) -> tensor<?x?xf32>
              %19 = arith.muli %12, %13 : index
              %20 = arith.muli %arg12, %14 : index
              %21 = arith.muli %15, %c1024 : index
              %c0_2 = arith.constant 0 : index
              %22 = arith.muli %c0_2, %15 : index
              %23 = arith.muli %22, %c1024_1 : index
              %24 = arith.addi %23, %20 : index
              %25 = loom.reinterpret_cast %arg1 to offset : [%24], sizes : [%16, %15, %14], strides : [%21, %c1024, %c1], reuse : [seq = false, spat = true, temp = true] : memref<*xf32> to memref<?x?x?xf32, strided<[?, ?, ?], offset: ?>>
              %26 = loom.alloc(%16, %15, %14) on @L1 : memref<?x?x?xf32>
              loom.copy %25, %26, interconnect : [@vertical_links], broadcast : [8, 1] : memref<?x?x?xf32, strided<[?, ?, ?], offset: ?>> to memref<?x?x?xf32>
              %27 = scf.for %arg13 = %c0 to %16 step %c1 iter_args(%arg14 = %18) -> (tensor<?x?xf32>) {
                %31 = arith.muli %arg13, %15 : index
                %32 = arith.muli %19, %c1024_1 : index
                %33 = arith.addi %32, %31 : index
                %34 = loom.reinterpret_cast %arg0 to offset : [%33], sizes : [%13, %15], strides : [%c1024_0, %c1], reuse : [seq = false, spat = false, temp = true] : memref<*xf32> to memref<?x?xf32, strided<[?, ?], offset: ?>>
                %35 = loom.alloc(%13, %15) on @L1 : memref<?x?xf32>
                loom.copy %34, %35, interconnect : [], broadcast : [1, 1] : memref<?x?xf32, strided<[?, ?], offset: ?>> to memref<?x?xf32>
                %36 = bufferization.to_tensor %35 restrict writable : memref<?x?xf32> to tensor<?x?xf32>
                %37 = arith.muli %31, %c1024_1 : index
                %subview = memref.subview %26[%arg13, 0, 0] [1, %15, %14] [1, 1, 1] : memref<?x?x?xf32> to memref<?x?xf32, strided<[?, 1], offset: ?>>
                %38 = bufferization.to_tensor %subview restrict writable : memref<?x?xf32, strided<[?, 1], offset: ?>> to tensor<?x?xf32>
                %39 = linalg.matmul ins(%36, %38 : tensor<?x?xf32>, tensor<?x?xf32>) outs(%18 : tensor<?x?xf32>) -> tensor<?x?xf32>
                %40 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%arg14, %39 : tensor<?x?xf32>, tensor<?x?xf32>) outs(%arg14 : tensor<?x?xf32>) {
                ^bb0(%in: f32, %in_3: f32, %out: f32):
                  %41 = arith.addf %in, %in_3 : f32
                  linalg.yield %41 : f32
                } -> tensor<?x?xf32>
                scf.yield %40 : tensor<?x?xf32>
              }
              %28 = arith.muli %19, %c1024_1 : index
              %29 = arith.addi %28, %20 : index
              %30 = loom.reinterpret_cast %arg2 to offset : [%29], sizes : [%13, %14], strides : [%c1024, %c1], reuse : [seq = false, spat = false, temp = false] : memref<*xf32> to memref<?x?xf32, strided<[?, ?], offset: ?>>
              bufferization.materialize_in_destination %27 in writable %30 : (tensor<?x?xf32>, memref<?x?xf32, strided<[?, ?], offset: ?>>) -> ()
            }
          }
        } {loom.mapped_to = @y}
      } {loom.mapped_to = @x}
      return
    }
  }
  module attributes {loom.pass_name = "EnumerateCopyBroadcast"} {
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
      %15 = loom.expression(%12, %13) {coeffs = [1, 1], logic = "mul"} : index
      %16 = loom.expression(%15, %14) {coeffs = [1, 1], logic = "mul"} : index
      loom.polynomial_constraint(%12, %13, %14, %15, %16) {monomials = [{coeff = -1 : i64, vars = [4]}], upper_bound = -262144 : i64}
      %17 = loom.expression(%12, %13) {coeffs = [1, 1], logic = "add"} : index
      %18 = loom.expression(%17, %14) {coeffs = [1, 1], logic = "mul"} : index
      loom.polynomial_constraint(%12, %14, %13, %17, %18) {monomials = [{coeff = 1 : i64, vars = [4]}], upper_bound = 374784 : i64}
      loom.polynomial_constraint(%12) {monomials = [{coeff = 1 : i64, vars = [0]}], upper_bound = 16 : i64}
      %19 = loom.expression(%12, %14) {coeffs = [1, 1], logic = "mul"} : index
      %20 = loom.expression(%13, %14) {coeffs = [1, 1], logic = "mul"} : index
      %21 = loom.expression(%12, %13) {coeffs = [1, 1], logic = "mul"} : index
      %22 = loom.expression(%14, %21) {coeffs = [1, 1], logic = "mul"} : index
      loom.polynomial_constraint(%12, %13, %14, %19, %20, %21, %22) {monomials = [{coeff = 64 : i64, vars = [3]}, {coeff = 64 : i64, vars = [4]}, {coeff = -1 : i64, vars = [6]}], upper_bound = 0 : i64}
    }
    func.func @matmul_kernel__d0i0_d1i0__f10__d_d(%arg0: memref<*xf32> {tt.divisibility = 16 : i32}, %arg1: memref<*xf32> {tt.divisibility = 16 : i32}, %arg2: memref<*xf32> {tt.divisibility = 16 : i32}, %arg3: index, %arg4: index, %arg5: index, %arg6: index, %arg7: index, %arg8: index) {
      affine.parallel (%arg9) = (0) to (8) {
        affine.parallel (%arg10) = (0) to (8) {
          affine.for %arg11 = 0 to affine_map<()[s0, s1] -> (s1)>()[%arg3, %arg4] {
            affine.for %arg12 = 0 to affine_map<()[s0, s1] -> ((s0 ceildiv 8) ceildiv 8)>()[%arg3, %arg4] {
              %12 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 8 + d1 * 8 + d2)>(%arg9, %arg10, %arg12)
              %c1024 = arith.constant 1024 : index
              %c1024_0 = arith.constant 1024 : index
              %c1 = arith.constant 1 : index
              %13 = loom.get_symbolic_block_size @constraints::@BM : index
              %14 = loom.get_symbolic_block_size @constraints::@BN : index
              %15 = loom.get_symbolic_block_size @constraints::@BK : index
              %16 = arith.ceildivsi %c1024_0, %15 : index
              %cst = arith.constant 0.000000e+00 : f32
              %c1024_1 = arith.constant 1024 : index
              %c0 = arith.constant 0 : index
              %17 = tensor.empty(%13, %14) : tensor<?x?xf32>
              %18 = linalg.fill ins(%cst : f32) outs(%17 : tensor<?x?xf32>) -> tensor<?x?xf32>
              %19 = arith.muli %12, %13 : index
              %20 = arith.muli %arg11, %14 : index
              %21 = scf.for %arg13 = %c0 to %16 step %c1 iter_args(%arg14 = %18) -> (tensor<?x?xf32>) {
                %25 = arith.muli %arg13, %15 : index
                %26 = arith.muli %19, %c1024_1 : index
                %27 = arith.addi %26, %25 : index
                %28 = loom.reinterpret_cast %arg0 to offset : [%27], sizes : [%13, %15], strides : [%c1024_0, %c1], reuse : [seq = false, spat = false, temp = true] : memref<*xf32> to memref<?x?xf32, strided<[?, ?], offset: ?>>
                %29 = loom.alloc(%13, %15) on @L1 : memref<?x?xf32>
                loom.copy %28, %29, interconnect : [], broadcast : [1, 1] : memref<?x?xf32, strided<[?, ?], offset: ?>> to memref<?x?xf32>
                %30 = bufferization.to_tensor %29 restrict writable : memref<?x?xf32> to tensor<?x?xf32>
                %31 = arith.muli %25, %c1024_1 : index
                %32 = arith.addi %31, %20 : index
                %33 = loom.reinterpret_cast %arg1 to offset : [%32], sizes : [%15, %14], strides : [%c1024, %c1], reuse : [seq = false, spat = true, temp = true] : memref<*xf32> to memref<?x?xf32, strided<[?, ?], offset: ?>>
                %34 = loom.alloc(%15, %14) on @L1 : memref<?x?xf32>
                loom.copy %33, %34, interconnect : [], broadcast : [1, 1] : memref<?x?xf32, strided<[?, ?], offset: ?>> to memref<?x?xf32>
                %35 = bufferization.to_tensor %34 restrict writable : memref<?x?xf32> to tensor<?x?xf32>
                %36 = linalg.matmul ins(%30, %35 : tensor<?x?xf32>, tensor<?x?xf32>) outs(%18 : tensor<?x?xf32>) -> tensor<?x?xf32>
                %37 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%arg14, %36 : tensor<?x?xf32>, tensor<?x?xf32>) outs(%arg14 : tensor<?x?xf32>) {
                ^bb0(%in: f32, %in_2: f32, %out: f32):
                  %38 = arith.addf %in, %in_2 : f32
                  linalg.yield %38 : f32
                } -> tensor<?x?xf32>
                scf.yield %37 : tensor<?x?xf32>
              }
              %22 = arith.muli %19, %c1024_1 : index
              %23 = arith.addi %22, %20 : index
              %24 = loom.reinterpret_cast %arg2 to offset : [%23], sizes : [%13, %14], strides : [%c1024, %c1], reuse : [seq = false, spat = false, temp = false] : memref<*xf32> to memref<?x?xf32, strided<[?, ?], offset: ?>>
              bufferization.materialize_in_destination %21 in writable %24 : (tensor<?x?xf32>, memref<?x?xf32, strided<[?, ?], offset: ?>>) -> ()
            }
          }
        } {loom.mapped_to = @y}
      } {loom.mapped_to = @x}
      return
    }
  }
  module attributes {loom.pass_name = "EnumerateCopyBroadcast"} {
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
      %15 = loom.expression(%12, %13) {coeffs = [1, 1], logic = "mul"} : index
      %16 = loom.expression(%15, %14) {coeffs = [1, 1], logic = "mul"} : index
      loom.polynomial_constraint(%12, %13, %14, %15, %16) {monomials = [{coeff = -1 : i64, vars = [4]}], upper_bound = -262144 : i64}
      %17 = loom.expression(%12, %13) {coeffs = [1, 1], logic = "add"} : index
      %18 = loom.expression(%17, %14) {coeffs = [1, 1], logic = "mul"} : index
      loom.polynomial_constraint(%12, %14, %13, %17, %18) {monomials = [{coeff = 1 : i64, vars = [4]}], upper_bound = 374784 : i64}
      loom.polynomial_constraint(%12) {monomials = [{coeff = 1 : i64, vars = [0]}], upper_bound = 16 : i64}
      %19 = loom.expression(%12, %14) {coeffs = [1, 1], logic = "mul"} : index
      %20 = loom.expression(%13, %14) {coeffs = [1, 1], logic = "mul"} : index
      %21 = loom.expression(%12, %13) {coeffs = [1, 1], logic = "mul"} : index
      %22 = loom.expression(%14, %21) {coeffs = [1, 1], logic = "mul"} : index
      loom.polynomial_constraint(%12, %13, %14, %19, %20, %21, %22) {monomials = [{coeff = 960 : i64, vars = [3]}, {coeff = 256 : i64, vars = [4]}, {coeff = -15 : i64, vars = [6]}], upper_bound = 0 : i64}
    }
    func.func @matmul_kernel__d0i0_d1i0__f10__d_a(%arg0: memref<*xf32> {tt.divisibility = 16 : i32}, %arg1: memref<*xf32> {tt.divisibility = 16 : i32}, %arg2: memref<*xf32> {tt.divisibility = 16 : i32}, %arg3: index, %arg4: index, %arg5: index, %arg6: index, %arg7: index, %arg8: index) {
      affine.parallel (%arg9) = (0) to (8) {
        affine.parallel (%arg10) = (0) to (8) {
          affine.for %arg11 = 0 to affine_map<()[s0, s1] -> (s1)>()[%arg3, %arg4] {
            affine.for %arg12 = 0 to affine_map<()[s0, s1] -> ((s0 ceildiv 8) ceildiv 8)>()[%arg3, %arg4] {
              %12 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 8 + d1 * 8 + d2)>(%arg9, %arg10, %arg12)
              %c1024 = arith.constant 1024 : index
              %c1024_0 = arith.constant 1024 : index
              %c1 = arith.constant 1 : index
              %13 = loom.get_symbolic_block_size @constraints::@BM : index
              %14 = loom.get_symbolic_block_size @constraints::@BN : index
              %15 = loom.get_symbolic_block_size @constraints::@BK : index
              %16 = arith.ceildivsi %c1024_0, %15 : index
              %cst = arith.constant 0.000000e+00 : f32
              %c1024_1 = arith.constant 1024 : index
              %c0 = arith.constant 0 : index
              %17 = tensor.empty(%13, %14) : tensor<?x?xf32>
              %18 = linalg.fill ins(%cst : f32) outs(%17 : tensor<?x?xf32>) -> tensor<?x?xf32>
              %19 = arith.muli %12, %13 : index
              %20 = arith.muli %arg11, %14 : index
              %21 = scf.for %arg13 = %c0 to %16 step %c1 iter_args(%arg14 = %18) -> (tensor<?x?xf32>) {
                %25 = arith.muli %arg13, %15 : index
                %26 = arith.muli %19, %c1024_1 : index
                %27 = arith.addi %26, %25 : index
                %28 = loom.reinterpret_cast %arg0 to offset : [%27], sizes : [%13, %15], strides : [%c1024_0, %c1], reuse : [seq = false, spat = false, temp = true] : memref<*xf32> to memref<?x?xf32, strided<[?, ?], offset: ?>>
                %29 = loom.alloc(%13, %15) on @L1 : memref<?x?xf32>
                loom.copy %28, %29, interconnect : [], broadcast : [1, 1] : memref<?x?xf32, strided<[?, ?], offset: ?>> to memref<?x?xf32>
                %30 = bufferization.to_tensor %29 restrict writable : memref<?x?xf32> to tensor<?x?xf32>
                %31 = arith.muli %25, %c1024_1 : index
                %32 = arith.addi %31, %20 : index
                %33 = loom.reinterpret_cast %arg1 to offset : [%32], sizes : [%15, %14], strides : [%c1024, %c1], reuse : [seq = false, spat = true, temp = true] : memref<*xf32> to memref<?x?xf32, strided<[?, ?], offset: ?>>
                %34 = loom.alloc(%15, %14) on @L1 : memref<?x?xf32>
                loom.copy %33, %34, interconnect : [@horizontal_links, @vertical_links], broadcast : [8, 8] : memref<?x?xf32, strided<[?, ?], offset: ?>> to memref<?x?xf32>
                %35 = bufferization.to_tensor %34 restrict writable : memref<?x?xf32> to tensor<?x?xf32>
                %36 = linalg.matmul ins(%30, %35 : tensor<?x?xf32>, tensor<?x?xf32>) outs(%18 : tensor<?x?xf32>) -> tensor<?x?xf32>
                %37 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%arg14, %36 : tensor<?x?xf32>, tensor<?x?xf32>) outs(%arg14 : tensor<?x?xf32>) {
                ^bb0(%in: f32, %in_2: f32, %out: f32):
                  %38 = arith.addf %in, %in_2 : f32
                  linalg.yield %38 : f32
                } -> tensor<?x?xf32>
                scf.yield %37 : tensor<?x?xf32>
              }
              %22 = arith.muli %19, %c1024_1 : index
              %23 = arith.addi %22, %20 : index
              %24 = loom.reinterpret_cast %arg2 to offset : [%23], sizes : [%13, %14], strides : [%c1024, %c1], reuse : [seq = false, spat = false, temp = false] : memref<*xf32> to memref<?x?xf32, strided<[?, ?], offset: ?>>
              bufferization.materialize_in_destination %21 in writable %24 : (tensor<?x?xf32>, memref<?x?xf32, strided<[?, ?], offset: ?>>) -> ()
            }
          }
        } {loom.mapped_to = @y}
      } {loom.mapped_to = @x}
      return
    }
  }
  module attributes {loom.pass_name = "EnumerateCopyBroadcast"} {
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
      %15 = loom.expression(%12, %13) {coeffs = [1, 1], logic = "mul"} : index
      %16 = loom.expression(%15, %14) {coeffs = [1, 1], logic = "mul"} : index
      loom.polynomial_constraint(%12, %13, %14, %15, %16) {monomials = [{coeff = -1 : i64, vars = [4]}], upper_bound = -262144 : i64}
      %17 = loom.expression(%12, %13) {coeffs = [1, 1], logic = "add"} : index
      %18 = loom.expression(%17, %14) {coeffs = [1, 1], logic = "mul"} : index
      loom.polynomial_constraint(%12, %14, %13, %17, %18) {monomials = [{coeff = 1 : i64, vars = [4]}], upper_bound = 374784 : i64}
      loom.polynomial_constraint(%12) {monomials = [{coeff = 1 : i64, vars = [0]}], upper_bound = 16 : i64}
      %19 = loom.expression(%12, %14) {coeffs = [1, 1], logic = "mul"} : index
      %20 = loom.expression(%13, %14) {coeffs = [1, 1], logic = "mul"} : index
      %21 = loom.expression(%12, %13) {coeffs = [1, 1], logic = "mul"} : index
      %22 = loom.expression(%14, %21) {coeffs = [1, 1], logic = "mul"} : index
      loom.polynomial_constraint(%12, %13, %14, %19, %20, %21, %22) {monomials = [{coeff = 960 : i64, vars = [3]}, {coeff = 256 : i64, vars = [4]}, {coeff = -15 : i64, vars = [6]}], upper_bound = 0 : i64}
    }
    func.func @matmul_kernel__d0i0_d1i0__f10__d_h(%arg0: memref<*xf32> {tt.divisibility = 16 : i32}, %arg1: memref<*xf32> {tt.divisibility = 16 : i32}, %arg2: memref<*xf32> {tt.divisibility = 16 : i32}, %arg3: index, %arg4: index, %arg5: index, %arg6: index, %arg7: index, %arg8: index) {
      affine.parallel (%arg9) = (0) to (8) {
        affine.parallel (%arg10) = (0) to (8) {
          affine.for %arg11 = 0 to affine_map<()[s0, s1] -> (s1)>()[%arg3, %arg4] {
            affine.for %arg12 = 0 to affine_map<()[s0, s1] -> ((s0 ceildiv 8) ceildiv 8)>()[%arg3, %arg4] {
              %12 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 8 + d1 * 8 + d2)>(%arg9, %arg10, %arg12)
              %c1024 = arith.constant 1024 : index
              %c1024_0 = arith.constant 1024 : index
              %c1 = arith.constant 1 : index
              %13 = loom.get_symbolic_block_size @constraints::@BM : index
              %14 = loom.get_symbolic_block_size @constraints::@BN : index
              %15 = loom.get_symbolic_block_size @constraints::@BK : index
              %16 = arith.ceildivsi %c1024_0, %15 : index
              %cst = arith.constant 0.000000e+00 : f32
              %c1024_1 = arith.constant 1024 : index
              %c0 = arith.constant 0 : index
              %17 = tensor.empty(%13, %14) : tensor<?x?xf32>
              %18 = linalg.fill ins(%cst : f32) outs(%17 : tensor<?x?xf32>) -> tensor<?x?xf32>
              %19 = arith.muli %12, %13 : index
              %20 = arith.muli %arg11, %14 : index
              %21 = scf.for %arg13 = %c0 to %16 step %c1 iter_args(%arg14 = %18) -> (tensor<?x?xf32>) {
                %25 = arith.muli %arg13, %15 : index
                %26 = arith.muli %19, %c1024_1 : index
                %27 = arith.addi %26, %25 : index
                %28 = loom.reinterpret_cast %arg0 to offset : [%27], sizes : [%13, %15], strides : [%c1024_0, %c1], reuse : [seq = false, spat = false, temp = true] : memref<*xf32> to memref<?x?xf32, strided<[?, ?], offset: ?>>
                %29 = loom.alloc(%13, %15) on @L1 : memref<?x?xf32>
                loom.copy %28, %29, interconnect : [], broadcast : [1, 1] : memref<?x?xf32, strided<[?, ?], offset: ?>> to memref<?x?xf32>
                %30 = bufferization.to_tensor %29 restrict writable : memref<?x?xf32> to tensor<?x?xf32>
                %31 = arith.muli %25, %c1024_1 : index
                %32 = arith.addi %31, %20 : index
                %33 = loom.reinterpret_cast %arg1 to offset : [%32], sizes : [%15, %14], strides : [%c1024, %c1], reuse : [seq = false, spat = true, temp = true] : memref<*xf32> to memref<?x?xf32, strided<[?, ?], offset: ?>>
                %34 = loom.alloc(%15, %14) on @L1 : memref<?x?xf32>
                loom.copy %33, %34, interconnect : [@horizontal_links], broadcast : [1, 8] : memref<?x?xf32, strided<[?, ?], offset: ?>> to memref<?x?xf32>
                %35 = bufferization.to_tensor %34 restrict writable : memref<?x?xf32> to tensor<?x?xf32>
                %36 = linalg.matmul ins(%30, %35 : tensor<?x?xf32>, tensor<?x?xf32>) outs(%18 : tensor<?x?xf32>) -> tensor<?x?xf32>
                %37 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%arg14, %36 : tensor<?x?xf32>, tensor<?x?xf32>) outs(%arg14 : tensor<?x?xf32>) {
                ^bb0(%in: f32, %in_2: f32, %out: f32):
                  %38 = arith.addf %in, %in_2 : f32
                  linalg.yield %38 : f32
                } -> tensor<?x?xf32>
                scf.yield %37 : tensor<?x?xf32>
              }
              %22 = arith.muli %19, %c1024_1 : index
              %23 = arith.addi %22, %20 : index
              %24 = loom.reinterpret_cast %arg2 to offset : [%23], sizes : [%13, %14], strides : [%c1024, %c1], reuse : [seq = false, spat = false, temp = false] : memref<*xf32> to memref<?x?xf32, strided<[?, ?], offset: ?>>
              bufferization.materialize_in_destination %21 in writable %24 : (tensor<?x?xf32>, memref<?x?xf32, strided<[?, ?], offset: ?>>) -> ()
            }
          }
        } {loom.mapped_to = @y}
      } {loom.mapped_to = @x}
      return
    }
  }
  module attributes {loom.pass_name = "EnumerateCopyBroadcast"} {
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
      %15 = loom.expression(%12, %13) {coeffs = [1, 1], logic = "mul"} : index
      %16 = loom.expression(%15, %14) {coeffs = [1, 1], logic = "mul"} : index
      loom.polynomial_constraint(%12, %13, %14, %15, %16) {monomials = [{coeff = -1 : i64, vars = [4]}], upper_bound = -262144 : i64}
      %17 = loom.expression(%12, %13) {coeffs = [1, 1], logic = "add"} : index
      %18 = loom.expression(%17, %14) {coeffs = [1, 1], logic = "mul"} : index
      loom.polynomial_constraint(%12, %14, %13, %17, %18) {monomials = [{coeff = 1 : i64, vars = [4]}], upper_bound = 374784 : i64}
      loom.polynomial_constraint(%12) {monomials = [{coeff = 1 : i64, vars = [0]}], upper_bound = 16 : i64}
      %19 = loom.expression(%12, %14) {coeffs = [1, 1], logic = "mul"} : index
      %20 = loom.expression(%13, %14) {coeffs = [1, 1], logic = "mul"} : index
      %21 = loom.expression(%12, %13) {coeffs = [1, 1], logic = "mul"} : index
      %22 = loom.expression(%14, %21) {coeffs = [1, 1], logic = "mul"} : index
      loom.polynomial_constraint(%12, %13, %14, %19, %20, %21, %22) {monomials = [{coeff = 960 : i64, vars = [3]}, {coeff = 256 : i64, vars = [4]}, {coeff = -15 : i64, vars = [6]}], upper_bound = 0 : i64}
    }
    func.func @matmul_kernel__d0i0_d1i0__f10__d_v(%arg0: memref<*xf32> {tt.divisibility = 16 : i32}, %arg1: memref<*xf32> {tt.divisibility = 16 : i32}, %arg2: memref<*xf32> {tt.divisibility = 16 : i32}, %arg3: index, %arg4: index, %arg5: index, %arg6: index, %arg7: index, %arg8: index) {
      affine.parallel (%arg9) = (0) to (8) {
        affine.parallel (%arg10) = (0) to (8) {
          affine.for %arg11 = 0 to affine_map<()[s0, s1] -> (s1)>()[%arg3, %arg4] {
            affine.for %arg12 = 0 to affine_map<()[s0, s1] -> ((s0 ceildiv 8) ceildiv 8)>()[%arg3, %arg4] {
              %12 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 8 + d1 * 8 + d2)>(%arg9, %arg10, %arg12)
              %c1024 = arith.constant 1024 : index
              %c1024_0 = arith.constant 1024 : index
              %c1 = arith.constant 1 : index
              %13 = loom.get_symbolic_block_size @constraints::@BM : index
              %14 = loom.get_symbolic_block_size @constraints::@BN : index
              %15 = loom.get_symbolic_block_size @constraints::@BK : index
              %16 = arith.ceildivsi %c1024_0, %15 : index
              %cst = arith.constant 0.000000e+00 : f32
              %c1024_1 = arith.constant 1024 : index
              %c0 = arith.constant 0 : index
              %17 = tensor.empty(%13, %14) : tensor<?x?xf32>
              %18 = linalg.fill ins(%cst : f32) outs(%17 : tensor<?x?xf32>) -> tensor<?x?xf32>
              %19 = arith.muli %12, %13 : index
              %20 = arith.muli %arg11, %14 : index
              %21 = scf.for %arg13 = %c0 to %16 step %c1 iter_args(%arg14 = %18) -> (tensor<?x?xf32>) {
                %25 = arith.muli %arg13, %15 : index
                %26 = arith.muli %19, %c1024_1 : index
                %27 = arith.addi %26, %25 : index
                %28 = loom.reinterpret_cast %arg0 to offset : [%27], sizes : [%13, %15], strides : [%c1024_0, %c1], reuse : [seq = false, spat = false, temp = true] : memref<*xf32> to memref<?x?xf32, strided<[?, ?], offset: ?>>
                %29 = loom.alloc(%13, %15) on @L1 : memref<?x?xf32>
                loom.copy %28, %29, interconnect : [], broadcast : [1, 1] : memref<?x?xf32, strided<[?, ?], offset: ?>> to memref<?x?xf32>
                %30 = bufferization.to_tensor %29 restrict writable : memref<?x?xf32> to tensor<?x?xf32>
                %31 = arith.muli %25, %c1024_1 : index
                %32 = arith.addi %31, %20 : index
                %33 = loom.reinterpret_cast %arg1 to offset : [%32], sizes : [%15, %14], strides : [%c1024, %c1], reuse : [seq = false, spat = true, temp = true] : memref<*xf32> to memref<?x?xf32, strided<[?, ?], offset: ?>>
                %34 = loom.alloc(%15, %14) on @L1 : memref<?x?xf32>
                loom.copy %33, %34, interconnect : [@vertical_links], broadcast : [8, 1] : memref<?x?xf32, strided<[?, ?], offset: ?>> to memref<?x?xf32>
                %35 = bufferization.to_tensor %34 restrict writable : memref<?x?xf32> to tensor<?x?xf32>
                %36 = linalg.matmul ins(%30, %35 : tensor<?x?xf32>, tensor<?x?xf32>) outs(%18 : tensor<?x?xf32>) -> tensor<?x?xf32>
                %37 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%arg14, %36 : tensor<?x?xf32>, tensor<?x?xf32>) outs(%arg14 : tensor<?x?xf32>) {
                ^bb0(%in: f32, %in_2: f32, %out: f32):
                  %38 = arith.addf %in, %in_2 : f32
                  linalg.yield %38 : f32
                } -> tensor<?x?xf32>
                scf.yield %37 : tensor<?x?xf32>
              }
              %22 = arith.muli %19, %c1024_1 : index
              %23 = arith.addi %22, %20 : index
              %24 = loom.reinterpret_cast %arg2 to offset : [%23], sizes : [%13, %14], strides : [%c1024, %c1], reuse : [seq = false, spat = false, temp = false] : memref<*xf32> to memref<?x?xf32, strided<[?, ?], offset: ?>>
              bufferization.materialize_in_destination %21 in writable %24 : (tensor<?x?xf32>, memref<?x?xf32, strided<[?, ?], offset: ?>>) -> ()
            }
          }
        } {loom.mapped_to = @y}
      } {loom.mapped_to = @x}
      return
    }
  }
  module attributes {loom.pass_name = "EnumerateCopyBroadcast"} {
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
      %15 = loom.expression(%12, %13) {coeffs = [1, 1], logic = "mul"} : index
      %16 = loom.expression(%14, %15) {coeffs = [1, 1], logic = "mul"} : index
      loom.polynomial_constraint(%12, %13, %14, %15, %16) {monomials = [{coeff = -1 : i64, vars = [4]}], upper_bound = -262144 : i64}
      %17 = loom.expression(%12, %13) {coeffs = [1, 1], logic = "add"} : index
      %18 = loom.expression(%14, %17) {coeffs = [1, 1], logic = "mul"} : index
      loom.polynomial_constraint(%12, %14, %13, %17, %18) {monomials = [{coeff = 1 : i64, vars = [4]}], upper_bound = 374784 : i64}
      loom.polynomial_constraint(%12) {monomials = [{coeff = 1 : i64, vars = [0]}], upper_bound = 16 : i64}
      %19 = loom.expression(%14, %13) {coeffs = [1, 1], logic = "mul"} : index
      loom.polynomial_constraint(%12, %14, %13, %19) {monomials = [{coeff = 1024 : i64, vars = [0]}, {coeff = 1 : i64, vars = [3]}], upper_bound = 374784 : i64}
      %20 = loom.expression(%14, %12) {coeffs = [1, 1], logic = "mul"} : index
      %21 = loom.expression(%14, %13) {coeffs = [1, 1], logic = "mul"} : index
      %22 = loom.expression(%12, %13) {coeffs = [1, 1], logic = "mul"} : index
      %23 = loom.expression(%14, %22) {coeffs = [1, 1], logic = "mul"} : index
      loom.polynomial_constraint(%12, %13, %14, %20, %21, %22, %23) {monomials = [{coeff = 64 : i64, vars = [3]}, {coeff = 64 : i64, vars = [4]}, {coeff = -1 : i64, vars = [6]}], upper_bound = 0 : i64}
    }
    func.func @matmul_kernel__d0i0_d1i0__f10__hoist_block_0__d_d(%arg0: memref<*xf32> {tt.divisibility = 16 : i32}, %arg1: memref<*xf32> {tt.divisibility = 16 : i32}, %arg2: memref<*xf32> {tt.divisibility = 16 : i32}, %arg3: index, %arg4: index, %arg5: index, %arg6: index, %arg7: index, %arg8: index) {
      affine.parallel (%arg9) = (0) to (8) {
        affine.parallel (%arg10) = (0) to (8) {
          affine.for %arg11 = 0 to affine_map<()[s0, s1] -> (s1)>()[%arg3, %arg4] {
            affine.for %arg12 = 0 to affine_map<()[s0, s1] -> ((s0 ceildiv 8) ceildiv 8)>()[%arg3, %arg4] {
              %12 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 8 + d1 * 8 + d2)>(%arg9, %arg10, %arg12)
              %c1024 = arith.constant 1024 : index
              %c1024_0 = arith.constant 1024 : index
              %c1 = arith.constant 1 : index
              %13 = loom.get_symbolic_block_size @constraints::@BM : index
              %14 = loom.get_symbolic_block_size @constraints::@BN : index
              %15 = loom.get_symbolic_block_size @constraints::@BK : index
              %16 = arith.ceildivsi %c1024_0, %15 : index
              %cst = arith.constant 0.000000e+00 : f32
              %c1024_1 = arith.constant 1024 : index
              %c0 = arith.constant 0 : index
              %17 = tensor.empty(%13, %14) : tensor<?x?xf32>
              %18 = linalg.fill ins(%cst : f32) outs(%17 : tensor<?x?xf32>) -> tensor<?x?xf32>
              %19 = arith.muli %12, %13 : index
              %20 = arith.muli %arg11, %14 : index
              %21 = arith.muli %13, %c1024_0 : index
              %c0_2 = arith.constant 0 : index
              %22 = arith.muli %19, %c1024_1 : index
              %23 = arith.muli %c0_2, %15 : index
              %24 = arith.addi %22, %23 : index
              %25 = loom.reinterpret_cast %arg0 to offset : [%24], sizes : [%16, %13, %15], strides : [%21, %c1024_0, %c1], reuse : [seq = false, spat = false, temp = true] : memref<*xf32> to memref<?x?x?xf32, strided<[?, ?, ?], offset: ?>>
              %26 = loom.alloc(%16, %13, %15) on @L1 : memref<?x?x?xf32>
              loom.copy %25, %26, interconnect : [], broadcast : [1, 1] : memref<?x?x?xf32, strided<[?, ?, ?], offset: ?>> to memref<?x?x?xf32>
              %27 = scf.for %arg13 = %c0 to %16 step %c1 iter_args(%arg14 = %18) -> (tensor<?x?xf32>) {
                %31 = arith.muli %arg13, %15 : index
                %32 = arith.muli %19, %c1024_1 : index
                %subview = memref.subview %26[%arg13, 0, 0] [1, %13, %15] [1, 1, 1] : memref<?x?x?xf32> to memref<?x?xf32, strided<[?, 1], offset: ?>>
                %33 = bufferization.to_tensor %subview restrict writable : memref<?x?xf32, strided<[?, 1], offset: ?>> to tensor<?x?xf32>
                %34 = arith.muli %31, %c1024_1 : index
                %35 = arith.addi %34, %20 : index
                %36 = loom.reinterpret_cast %arg1 to offset : [%35], sizes : [%15, %14], strides : [%c1024, %c1], reuse : [seq = false, spat = true, temp = true] : memref<*xf32> to memref<?x?xf32, strided<[?, ?], offset: ?>>
                %37 = loom.alloc(%15, %14) on @L1 : memref<?x?xf32>
                loom.copy %36, %37, interconnect : [], broadcast : [1, 1] : memref<?x?xf32, strided<[?, ?], offset: ?>> to memref<?x?xf32>
                %38 = bufferization.to_tensor %37 restrict writable : memref<?x?xf32> to tensor<?x?xf32>
                %39 = linalg.matmul ins(%33, %38 : tensor<?x?xf32>, tensor<?x?xf32>) outs(%18 : tensor<?x?xf32>) -> tensor<?x?xf32>
                %40 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%arg14, %39 : tensor<?x?xf32>, tensor<?x?xf32>) outs(%arg14 : tensor<?x?xf32>) {
                ^bb0(%in: f32, %in_3: f32, %out: f32):
                  %41 = arith.addf %in, %in_3 : f32
                  linalg.yield %41 : f32
                } -> tensor<?x?xf32>
                scf.yield %40 : tensor<?x?xf32>
              }
              %28 = arith.muli %19, %c1024_1 : index
              %29 = arith.addi %28, %20 : index
              %30 = loom.reinterpret_cast %arg2 to offset : [%29], sizes : [%13, %14], strides : [%c1024, %c1], reuse : [seq = false, spat = false, temp = false] : memref<*xf32> to memref<?x?xf32, strided<[?, ?], offset: ?>>
              bufferization.materialize_in_destination %27 in writable %30 : (tensor<?x?xf32>, memref<?x?xf32, strided<[?, ?], offset: ?>>) -> ()
            }
          }
        } {loom.mapped_to = @y}
      } {loom.mapped_to = @x}
      return
    }
  }
  module attributes {loom.pass_name = "EnumerateCopyBroadcast"} {
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
      %15 = loom.expression(%12, %13) {coeffs = [1, 1], logic = "mul"} : index
      %16 = loom.expression(%15, %14) {coeffs = [1, 1], logic = "mul"} : index
      loom.polynomial_constraint(%12, %13, %14, %15, %16) {monomials = [{coeff = -1 : i64, vars = [4]}], upper_bound = -262144 : i64}
      %17 = loom.expression(%12, %13) {coeffs = [1, 1], logic = "add"} : index
      %18 = loom.expression(%17, %14) {coeffs = [1, 1], logic = "mul"} : index
      loom.polynomial_constraint(%12, %14, %13, %17, %18) {monomials = [{coeff = 1 : i64, vars = [4]}], upper_bound = 374784 : i64}
      loom.polynomial_constraint(%12) {monomials = [{coeff = 1 : i64, vars = [0]}], upper_bound = 16 : i64}
      %19 = loom.expression(%13, %14) {coeffs = [1, 1], logic = "mul"} : index
      loom.polynomial_constraint(%12, %14, %13, %19) {monomials = [{coeff = 1024 : i64, vars = [0]}, {coeff = 1 : i64, vars = [3]}], upper_bound = 374784 : i64}
      %20 = loom.expression(%12, %14) {coeffs = [1, 1], logic = "mul"} : index
      %21 = loom.expression(%13, %14) {coeffs = [1, 1], logic = "mul"} : index
      %22 = loom.expression(%12, %13) {coeffs = [1, 1], logic = "mul"} : index
      %23 = loom.expression(%14, %22) {coeffs = [1, 1], logic = "mul"} : index
      loom.polynomial_constraint(%12, %13, %14, %20, %21, %22, %23) {monomials = [{coeff = 960 : i64, vars = [3]}, {coeff = 256 : i64, vars = [4]}, {coeff = -15 : i64, vars = [6]}], upper_bound = 0 : i64}
    }
    func.func @matmul_kernel__d0i0_d1i0__f10__hoist_block_0__d_a(%arg0: memref<*xf32> {tt.divisibility = 16 : i32}, %arg1: memref<*xf32> {tt.divisibility = 16 : i32}, %arg2: memref<*xf32> {tt.divisibility = 16 : i32}, %arg3: index, %arg4: index, %arg5: index, %arg6: index, %arg7: index, %arg8: index) {
      affine.parallel (%arg9) = (0) to (8) {
        affine.parallel (%arg10) = (0) to (8) {
          affine.for %arg11 = 0 to affine_map<()[s0, s1] -> (s1)>()[%arg3, %arg4] {
            affine.for %arg12 = 0 to affine_map<()[s0, s1] -> ((s0 ceildiv 8) ceildiv 8)>()[%arg3, %arg4] {
              %12 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 8 + d1 * 8 + d2)>(%arg9, %arg10, %arg12)
              %c1024 = arith.constant 1024 : index
              %c1024_0 = arith.constant 1024 : index
              %c1 = arith.constant 1 : index
              %13 = loom.get_symbolic_block_size @constraints::@BM : index
              %14 = loom.get_symbolic_block_size @constraints::@BN : index
              %15 = loom.get_symbolic_block_size @constraints::@BK : index
              %16 = arith.ceildivsi %c1024_0, %15 : index
              %cst = arith.constant 0.000000e+00 : f32
              %c1024_1 = arith.constant 1024 : index
              %c0 = arith.constant 0 : index
              %17 = tensor.empty(%13, %14) : tensor<?x?xf32>
              %18 = linalg.fill ins(%cst : f32) outs(%17 : tensor<?x?xf32>) -> tensor<?x?xf32>
              %19 = arith.muli %12, %13 : index
              %20 = arith.muli %arg11, %14 : index
              %21 = arith.muli %13, %c1024_0 : index
              %c0_2 = arith.constant 0 : index
              %22 = arith.muli %19, %c1024_1 : index
              %23 = arith.muli %c0_2, %15 : index
              %24 = arith.addi %22, %23 : index
              %25 = loom.reinterpret_cast %arg0 to offset : [%24], sizes : [%16, %13, %15], strides : [%21, %c1024_0, %c1], reuse : [seq = false, spat = false, temp = true] : memref<*xf32> to memref<?x?x?xf32, strided<[?, ?, ?], offset: ?>>
              %26 = loom.alloc(%16, %13, %15) on @L1 : memref<?x?x?xf32>
              loom.copy %25, %26, interconnect : [], broadcast : [1, 1] : memref<?x?x?xf32, strided<[?, ?, ?], offset: ?>> to memref<?x?x?xf32>
              %27 = scf.for %arg13 = %c0 to %16 step %c1 iter_args(%arg14 = %18) -> (tensor<?x?xf32>) {
                %31 = arith.muli %arg13, %15 : index
                %32 = arith.muli %19, %c1024_1 : index
                %subview = memref.subview %26[%arg13, 0, 0] [1, %13, %15] [1, 1, 1] : memref<?x?x?xf32> to memref<?x?xf32, strided<[?, 1], offset: ?>>
                %33 = bufferization.to_tensor %subview restrict writable : memref<?x?xf32, strided<[?, 1], offset: ?>> to tensor<?x?xf32>
                %34 = arith.muli %31, %c1024_1 : index
                %35 = arith.addi %34, %20 : index
                %36 = loom.reinterpret_cast %arg1 to offset : [%35], sizes : [%15, %14], strides : [%c1024, %c1], reuse : [seq = false, spat = true, temp = true] : memref<*xf32> to memref<?x?xf32, strided<[?, ?], offset: ?>>
                %37 = loom.alloc(%15, %14) on @L1 : memref<?x?xf32>
                loom.copy %36, %37, interconnect : [@horizontal_links, @vertical_links], broadcast : [8, 8] : memref<?x?xf32, strided<[?, ?], offset: ?>> to memref<?x?xf32>
                %38 = bufferization.to_tensor %37 restrict writable : memref<?x?xf32> to tensor<?x?xf32>
                %39 = linalg.matmul ins(%33, %38 : tensor<?x?xf32>, tensor<?x?xf32>) outs(%18 : tensor<?x?xf32>) -> tensor<?x?xf32>
                %40 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%arg14, %39 : tensor<?x?xf32>, tensor<?x?xf32>) outs(%arg14 : tensor<?x?xf32>) {
                ^bb0(%in: f32, %in_3: f32, %out: f32):
                  %41 = arith.addf %in, %in_3 : f32
                  linalg.yield %41 : f32
                } -> tensor<?x?xf32>
                scf.yield %40 : tensor<?x?xf32>
              }
              %28 = arith.muli %19, %c1024_1 : index
              %29 = arith.addi %28, %20 : index
              %30 = loom.reinterpret_cast %arg2 to offset : [%29], sizes : [%13, %14], strides : [%c1024, %c1], reuse : [seq = false, spat = false, temp = false] : memref<*xf32> to memref<?x?xf32, strided<[?, ?], offset: ?>>
              bufferization.materialize_in_destination %27 in writable %30 : (tensor<?x?xf32>, memref<?x?xf32, strided<[?, ?], offset: ?>>) -> ()
            }
          }
        } {loom.mapped_to = @y}
      } {loom.mapped_to = @x}
      return
    }
  }
  module attributes {loom.pass_name = "EnumerateCopyBroadcast"} {
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
      %15 = loom.expression(%12, %13) {coeffs = [1, 1], logic = "mul"} : index
      %16 = loom.expression(%14, %15) {coeffs = [1, 1], logic = "mul"} : index
      loom.polynomial_constraint(%12, %13, %14, %15, %16) {monomials = [{coeff = -1 : i64, vars = [4]}], upper_bound = -262144 : i64}
      %17 = loom.expression(%12, %13) {coeffs = [1, 1], logic = "add"} : index
      %18 = loom.expression(%14, %17) {coeffs = [1, 1], logic = "mul"} : index
      loom.polynomial_constraint(%12, %14, %13, %17, %18) {monomials = [{coeff = 1 : i64, vars = [4]}], upper_bound = 374784 : i64}
      loom.polynomial_constraint(%12) {monomials = [{coeff = 1 : i64, vars = [0]}], upper_bound = 16 : i64}
      %19 = loom.expression(%13, %14) {coeffs = [1, 1], logic = "mul"} : index
      loom.polynomial_constraint(%12, %14, %13, %19) {monomials = [{coeff = 1024 : i64, vars = [0]}, {coeff = 1 : i64, vars = [3]}], upper_bound = 374784 : i64}
      %20 = loom.expression(%12, %14) {coeffs = [1, 1], logic = "mul"} : index
      %21 = loom.expression(%13, %14) {coeffs = [1, 1], logic = "mul"} : index
      %22 = loom.expression(%12, %13) {coeffs = [1, 1], logic = "mul"} : index
      %23 = loom.expression(%14, %22) {coeffs = [1, 1], logic = "mul"} : index
      loom.polynomial_constraint(%12, %13, %14, %20, %21, %22, %23) {monomials = [{coeff = 960 : i64, vars = [3]}, {coeff = 256 : i64, vars = [4]}, {coeff = -15 : i64, vars = [6]}], upper_bound = 0 : i64}
    }
    func.func @matmul_kernel__d0i0_d1i0__f10__hoist_block_0__d_h(%arg0: memref<*xf32> {tt.divisibility = 16 : i32}, %arg1: memref<*xf32> {tt.divisibility = 16 : i32}, %arg2: memref<*xf32> {tt.divisibility = 16 : i32}, %arg3: index, %arg4: index, %arg5: index, %arg6: index, %arg7: index, %arg8: index) {
      affine.parallel (%arg9) = (0) to (8) {
        affine.parallel (%arg10) = (0) to (8) {
          affine.for %arg11 = 0 to affine_map<()[s0, s1] -> (s1)>()[%arg3, %arg4] {
            affine.for %arg12 = 0 to affine_map<()[s0, s1] -> ((s0 ceildiv 8) ceildiv 8)>()[%arg3, %arg4] {
              %12 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 8 + d1 * 8 + d2)>(%arg9, %arg10, %arg12)
              %c1024 = arith.constant 1024 : index
              %c1024_0 = arith.constant 1024 : index
              %c1 = arith.constant 1 : index
              %13 = loom.get_symbolic_block_size @constraints::@BM : index
              %14 = loom.get_symbolic_block_size @constraints::@BN : index
              %15 = loom.get_symbolic_block_size @constraints::@BK : index
              %16 = arith.ceildivsi %c1024_0, %15 : index
              %cst = arith.constant 0.000000e+00 : f32
              %c1024_1 = arith.constant 1024 : index
              %c0 = arith.constant 0 : index
              %17 = tensor.empty(%13, %14) : tensor<?x?xf32>
              %18 = linalg.fill ins(%cst : f32) outs(%17 : tensor<?x?xf32>) -> tensor<?x?xf32>
              %19 = arith.muli %12, %13 : index
              %20 = arith.muli %arg11, %14 : index
              %21 = arith.muli %13, %c1024_0 : index
              %c0_2 = arith.constant 0 : index
              %22 = arith.muli %19, %c1024_1 : index
              %23 = arith.muli %c0_2, %15 : index
              %24 = arith.addi %22, %23 : index
              %25 = loom.reinterpret_cast %arg0 to offset : [%24], sizes : [%16, %13, %15], strides : [%21, %c1024_0, %c1], reuse : [seq = false, spat = false, temp = true] : memref<*xf32> to memref<?x?x?xf32, strided<[?, ?, ?], offset: ?>>
              %26 = loom.alloc(%16, %13, %15) on @L1 : memref<?x?x?xf32>
              loom.copy %25, %26, interconnect : [], broadcast : [1, 1] : memref<?x?x?xf32, strided<[?, ?, ?], offset: ?>> to memref<?x?x?xf32>
              %27 = scf.for %arg13 = %c0 to %16 step %c1 iter_args(%arg14 = %18) -> (tensor<?x?xf32>) {
                %31 = arith.muli %arg13, %15 : index
                %32 = arith.muli %19, %c1024_1 : index
                %subview = memref.subview %26[%arg13, 0, 0] [1, %13, %15] [1, 1, 1] : memref<?x?x?xf32> to memref<?x?xf32, strided<[?, 1], offset: ?>>
                %33 = bufferization.to_tensor %subview restrict writable : memref<?x?xf32, strided<[?, 1], offset: ?>> to tensor<?x?xf32>
                %34 = arith.muli %31, %c1024_1 : index
                %35 = arith.addi %34, %20 : index
                %36 = loom.reinterpret_cast %arg1 to offset : [%35], sizes : [%15, %14], strides : [%c1024, %c1], reuse : [seq = false, spat = true, temp = true] : memref<*xf32> to memref<?x?xf32, strided<[?, ?], offset: ?>>
                %37 = loom.alloc(%15, %14) on @L1 : memref<?x?xf32>
                loom.copy %36, %37, interconnect : [@horizontal_links], broadcast : [1, 8] : memref<?x?xf32, strided<[?, ?], offset: ?>> to memref<?x?xf32>
                %38 = bufferization.to_tensor %37 restrict writable : memref<?x?xf32> to tensor<?x?xf32>
                %39 = linalg.matmul ins(%33, %38 : tensor<?x?xf32>, tensor<?x?xf32>) outs(%18 : tensor<?x?xf32>) -> tensor<?x?xf32>
                %40 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%arg14, %39 : tensor<?x?xf32>, tensor<?x?xf32>) outs(%arg14 : tensor<?x?xf32>) {
                ^bb0(%in: f32, %in_3: f32, %out: f32):
                  %41 = arith.addf %in, %in_3 : f32
                  linalg.yield %41 : f32
                } -> tensor<?x?xf32>
                scf.yield %40 : tensor<?x?xf32>
              }
              %28 = arith.muli %19, %c1024_1 : index
              %29 = arith.addi %28, %20 : index
              %30 = loom.reinterpret_cast %arg2 to offset : [%29], sizes : [%13, %14], strides : [%c1024, %c1], reuse : [seq = false, spat = false, temp = false] : memref<*xf32> to memref<?x?xf32, strided<[?, ?], offset: ?>>
              bufferization.materialize_in_destination %27 in writable %30 : (tensor<?x?xf32>, memref<?x?xf32, strided<[?, ?], offset: ?>>) -> ()
            }
          }
        } {loom.mapped_to = @y}
      } {loom.mapped_to = @x}
      return
    }
  }
  module attributes {loom.pass_name = "EnumerateCopyBroadcast"} {
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
      %15 = loom.expression(%12, %13) {coeffs = [1, 1], logic = "mul"} : index
      %16 = loom.expression(%15, %14) {coeffs = [1, 1], logic = "mul"} : index
      loom.polynomial_constraint(%12, %13, %14, %15, %16) {monomials = [{coeff = -1 : i64, vars = [4]}], upper_bound = -262144 : i64}
      %17 = loom.expression(%12, %13) {coeffs = [1, 1], logic = "add"} : index
      %18 = loom.expression(%17, %14) {coeffs = [1, 1], logic = "mul"} : index
      loom.polynomial_constraint(%12, %14, %13, %17, %18) {monomials = [{coeff = 1 : i64, vars = [4]}], upper_bound = 374784 : i64}
      loom.polynomial_constraint(%12) {monomials = [{coeff = 1 : i64, vars = [0]}], upper_bound = 16 : i64}
      %19 = loom.expression(%13, %14) {coeffs = [1, 1], logic = "mul"} : index
      loom.polynomial_constraint(%12, %14, %13, %19) {monomials = [{coeff = 1024 : i64, vars = [0]}, {coeff = 1 : i64, vars = [3]}], upper_bound = 374784 : i64}
      %20 = loom.expression(%12, %14) {coeffs = [1, 1], logic = "mul"} : index
      %21 = loom.expression(%13, %14) {coeffs = [1, 1], logic = "mul"} : index
      %22 = loom.expression(%12, %13) {coeffs = [1, 1], logic = "mul"} : index
      %23 = loom.expression(%14, %22) {coeffs = [1, 1], logic = "mul"} : index
      loom.polynomial_constraint(%12, %13, %14, %20, %21, %22, %23) {monomials = [{coeff = 960 : i64, vars = [3]}, {coeff = 256 : i64, vars = [4]}, {coeff = -15 : i64, vars = [6]}], upper_bound = 0 : i64}
    }
    func.func @matmul_kernel__d0i0_d1i0__f10__hoist_block_0__d_v(%arg0: memref<*xf32> {tt.divisibility = 16 : i32}, %arg1: memref<*xf32> {tt.divisibility = 16 : i32}, %arg2: memref<*xf32> {tt.divisibility = 16 : i32}, %arg3: index, %arg4: index, %arg5: index, %arg6: index, %arg7: index, %arg8: index) {
      affine.parallel (%arg9) = (0) to (8) {
        affine.parallel (%arg10) = (0) to (8) {
          affine.for %arg11 = 0 to affine_map<()[s0, s1] -> (s1)>()[%arg3, %arg4] {
            affine.for %arg12 = 0 to affine_map<()[s0, s1] -> ((s0 ceildiv 8) ceildiv 8)>()[%arg3, %arg4] {
              %12 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 8 + d1 * 8 + d2)>(%arg9, %arg10, %arg12)
              %c1024 = arith.constant 1024 : index
              %c1024_0 = arith.constant 1024 : index
              %c1 = arith.constant 1 : index
              %13 = loom.get_symbolic_block_size @constraints::@BM : index
              %14 = loom.get_symbolic_block_size @constraints::@BN : index
              %15 = loom.get_symbolic_block_size @constraints::@BK : index
              %16 = arith.ceildivsi %c1024_0, %15 : index
              %cst = arith.constant 0.000000e+00 : f32
              %c1024_1 = arith.constant 1024 : index
              %c0 = arith.constant 0 : index
              %17 = tensor.empty(%13, %14) : tensor<?x?xf32>
              %18 = linalg.fill ins(%cst : f32) outs(%17 : tensor<?x?xf32>) -> tensor<?x?xf32>
              %19 = arith.muli %12, %13 : index
              %20 = arith.muli %arg11, %14 : index
              %21 = arith.muli %13, %c1024_0 : index
              %c0_2 = arith.constant 0 : index
              %22 = arith.muli %19, %c1024_1 : index
              %23 = arith.muli %c0_2, %15 : index
              %24 = arith.addi %22, %23 : index
              %25 = loom.reinterpret_cast %arg0 to offset : [%24], sizes : [%16, %13, %15], strides : [%21, %c1024_0, %c1], reuse : [seq = false, spat = false, temp = true] : memref<*xf32> to memref<?x?x?xf32, strided<[?, ?, ?], offset: ?>>
              %26 = loom.alloc(%16, %13, %15) on @L1 : memref<?x?x?xf32>
              loom.copy %25, %26, interconnect : [], broadcast : [1, 1] : memref<?x?x?xf32, strided<[?, ?, ?], offset: ?>> to memref<?x?x?xf32>
              %27 = scf.for %arg13 = %c0 to %16 step %c1 iter_args(%arg14 = %18) -> (tensor<?x?xf32>) {
                %31 = arith.muli %arg13, %15 : index
                %32 = arith.muli %19, %c1024_1 : index
                %subview = memref.subview %26[%arg13, 0, 0] [1, %13, %15] [1, 1, 1] : memref<?x?x?xf32> to memref<?x?xf32, strided<[?, 1], offset: ?>>
                %33 = bufferization.to_tensor %subview restrict writable : memref<?x?xf32, strided<[?, 1], offset: ?>> to tensor<?x?xf32>
                %34 = arith.muli %31, %c1024_1 : index
                %35 = arith.addi %34, %20 : index
                %36 = loom.reinterpret_cast %arg1 to offset : [%35], sizes : [%15, %14], strides : [%c1024, %c1], reuse : [seq = false, spat = true, temp = true] : memref<*xf32> to memref<?x?xf32, strided<[?, ?], offset: ?>>
                %37 = loom.alloc(%15, %14) on @L1 : memref<?x?xf32>
                loom.copy %36, %37, interconnect : [@vertical_links], broadcast : [8, 1] : memref<?x?xf32, strided<[?, ?], offset: ?>> to memref<?x?xf32>
                %38 = bufferization.to_tensor %37 restrict writable : memref<?x?xf32> to tensor<?x?xf32>
                %39 = linalg.matmul ins(%33, %38 : tensor<?x?xf32>, tensor<?x?xf32>) outs(%18 : tensor<?x?xf32>) -> tensor<?x?xf32>
                %40 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%arg14, %39 : tensor<?x?xf32>, tensor<?x?xf32>) outs(%arg14 : tensor<?x?xf32>) {
                ^bb0(%in: f32, %in_3: f32, %out: f32):
                  %41 = arith.addf %in, %in_3 : f32
                  linalg.yield %41 : f32
                } -> tensor<?x?xf32>
                scf.yield %40 : tensor<?x?xf32>
              }
              %28 = arith.muli %19, %c1024_1 : index
              %29 = arith.addi %28, %20 : index
              %30 = loom.reinterpret_cast %arg2 to offset : [%29], sizes : [%13, %14], strides : [%c1024, %c1], reuse : [seq = false, spat = false, temp = false] : memref<*xf32> to memref<?x?xf32, strided<[?, ?], offset: ?>>
              bufferization.materialize_in_destination %27 in writable %30 : (tensor<?x?xf32>, memref<?x?xf32, strided<[?, ?], offset: ?>>) -> ()
            }
          }
        } {loom.mapped_to = @y}
      } {loom.mapped_to = @x}
      return
    }
  }
  module attributes {loom.pass_name = "EnumerateCopyBroadcast"} {
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
      %15 = loom.expression(%12, %13) {coeffs = [1, 1], logic = "mul"} : index
      %16 = loom.expression(%15, %14) {coeffs = [1, 1], logic = "mul"} : index
      loom.polynomial_constraint(%12, %13, %14, %15, %16) {monomials = [{coeff = -1 : i64, vars = [4]}], upper_bound = -262144 : i64}
      %17 = loom.expression(%12, %13) {coeffs = [1, 1], logic = "add"} : index
      %18 = loom.expression(%17, %14) {coeffs = [1, 1], logic = "mul"} : index
      loom.polynomial_constraint(%12, %14, %13, %17, %18) {monomials = [{coeff = 1 : i64, vars = [4]}], upper_bound = 374784 : i64}
      loom.polynomial_constraint(%12) {monomials = [{coeff = 1 : i64, vars = [0]}], upper_bound = 16 : i64}
      %19 = loom.expression(%12, %14) {coeffs = [1, 1], logic = "mul"} : index
      loom.polynomial_constraint(%13, %12, %14, %19) {monomials = [{coeff = 1024 : i64, vars = [0]}, {coeff = 1 : i64, vars = [3]}], upper_bound = 374784 : i64}
      %20 = loom.expression(%12, %14) {coeffs = [1, 1], logic = "mul"} : index
      %21 = loom.expression(%13, %14) {coeffs = [1, 1], logic = "mul"} : index
      %22 = loom.expression(%12, %13) {coeffs = [1, 1], logic = "mul"} : index
      %23 = loom.expression(%14, %22) {coeffs = [1, 1], logic = "mul"} : index
      loom.polynomial_constraint(%12, %13, %14, %20, %21, %22, %23) {monomials = [{coeff = 64 : i64, vars = [3]}, {coeff = 64 : i64, vars = [4]}, {coeff = -1 : i64, vars = [6]}], upper_bound = 0 : i64}
    }
    func.func @matmul_kernel__d0i0_d1i0__f10__hoist_block_1__d_d(%arg0: memref<*xf32> {tt.divisibility = 16 : i32}, %arg1: memref<*xf32> {tt.divisibility = 16 : i32}, %arg2: memref<*xf32> {tt.divisibility = 16 : i32}, %arg3: index, %arg4: index, %arg5: index, %arg6: index, %arg7: index, %arg8: index) {
      affine.parallel (%arg9) = (0) to (8) {
        affine.parallel (%arg10) = (0) to (8) {
          affine.for %arg11 = 0 to affine_map<()[s0, s1] -> (s1)>()[%arg3, %arg4] {
            affine.for %arg12 = 0 to affine_map<()[s0, s1] -> ((s0 ceildiv 8) ceildiv 8)>()[%arg3, %arg4] {
              %12 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 8 + d1 * 8 + d2)>(%arg9, %arg10, %arg12)
              %c1024 = arith.constant 1024 : index
              %c1024_0 = arith.constant 1024 : index
              %c1 = arith.constant 1 : index
              %13 = loom.get_symbolic_block_size @constraints::@BM : index
              %14 = loom.get_symbolic_block_size @constraints::@BN : index
              %15 = loom.get_symbolic_block_size @constraints::@BK : index
              %16 = arith.ceildivsi %c1024_0, %15 : index
              %cst = arith.constant 0.000000e+00 : f32
              %c1024_1 = arith.constant 1024 : index
              %c0 = arith.constant 0 : index
              %17 = tensor.empty(%13, %14) : tensor<?x?xf32>
              %18 = linalg.fill ins(%cst : f32) outs(%17 : tensor<?x?xf32>) -> tensor<?x?xf32>
              %19 = arith.muli %12, %13 : index
              %20 = arith.muli %arg11, %14 : index
              %21 = arith.muli %15, %c1024 : index
              %c0_2 = arith.constant 0 : index
              %22 = arith.muli %c0_2, %15 : index
              %23 = arith.muli %22, %c1024_1 : index
              %24 = arith.addi %23, %20 : index
              %25 = loom.reinterpret_cast %arg1 to offset : [%24], sizes : [%16, %15, %14], strides : [%21, %c1024, %c1], reuse : [seq = false, spat = true, temp = true] : memref<*xf32> to memref<?x?x?xf32, strided<[?, ?, ?], offset: ?>>
              %26 = loom.alloc(%16, %15, %14) on @L1 : memref<?x?x?xf32>
              loom.copy %25, %26, interconnect : [], broadcast : [1, 1] : memref<?x?x?xf32, strided<[?, ?, ?], offset: ?>> to memref<?x?x?xf32>
              %27 = scf.for %arg13 = %c0 to %16 step %c1 iter_args(%arg14 = %18) -> (tensor<?x?xf32>) {
                %31 = arith.muli %arg13, %15 : index
                %32 = arith.muli %19, %c1024_1 : index
                %33 = arith.addi %32, %31 : index
                %34 = loom.reinterpret_cast %arg0 to offset : [%33], sizes : [%13, %15], strides : [%c1024_0, %c1], reuse : [seq = false, spat = false, temp = true] : memref<*xf32> to memref<?x?xf32, strided<[?, ?], offset: ?>>
                %35 = loom.alloc(%13, %15) on @L1 : memref<?x?xf32>
                loom.copy %34, %35, interconnect : [], broadcast : [1, 1] : memref<?x?xf32, strided<[?, ?], offset: ?>> to memref<?x?xf32>
                %36 = bufferization.to_tensor %35 restrict writable : memref<?x?xf32> to tensor<?x?xf32>
                %37 = arith.muli %31, %c1024_1 : index
                %subview = memref.subview %26[%arg13, 0, 0] [1, %15, %14] [1, 1, 1] : memref<?x?x?xf32> to memref<?x?xf32, strided<[?, 1], offset: ?>>
                %38 = bufferization.to_tensor %subview restrict writable : memref<?x?xf32, strided<[?, 1], offset: ?>> to tensor<?x?xf32>
                %39 = linalg.matmul ins(%36, %38 : tensor<?x?xf32>, tensor<?x?xf32>) outs(%18 : tensor<?x?xf32>) -> tensor<?x?xf32>
                %40 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%arg14, %39 : tensor<?x?xf32>, tensor<?x?xf32>) outs(%arg14 : tensor<?x?xf32>) {
                ^bb0(%in: f32, %in_3: f32, %out: f32):
                  %41 = arith.addf %in, %in_3 : f32
                  linalg.yield %41 : f32
                } -> tensor<?x?xf32>
                scf.yield %40 : tensor<?x?xf32>
              }
              %28 = arith.muli %19, %c1024_1 : index
              %29 = arith.addi %28, %20 : index
              %30 = loom.reinterpret_cast %arg2 to offset : [%29], sizes : [%13, %14], strides : [%c1024, %c1], reuse : [seq = false, spat = false, temp = false] : memref<*xf32> to memref<?x?xf32, strided<[?, ?], offset: ?>>
              bufferization.materialize_in_destination %27 in writable %30 : (tensor<?x?xf32>, memref<?x?xf32, strided<[?, ?], offset: ?>>) -> ()
            }
          }
        } {loom.mapped_to = @y}
      } {loom.mapped_to = @x}
      return
    }
  }
  module attributes {loom.pass_name = "EnumerateCopyBroadcast"} {
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
      %15 = loom.expression(%12, %13) {coeffs = [1, 1], logic = "mul"} : index
      %16 = loom.expression(%15, %14) {coeffs = [1, 1], logic = "mul"} : index
      loom.polynomial_constraint(%12, %13, %14, %15, %16) {monomials = [{coeff = -1 : i64, vars = [4]}], upper_bound = -262144 : i64}
      %17 = loom.expression(%12, %13) {coeffs = [1, 1], logic = "add"} : index
      %18 = loom.expression(%17, %14) {coeffs = [1, 1], logic = "mul"} : index
      loom.polynomial_constraint(%12, %14, %13, %17, %18) {monomials = [{coeff = 1 : i64, vars = [4]}], upper_bound = 374784 : i64}
      loom.polynomial_constraint(%12) {monomials = [{coeff = 1 : i64, vars = [0]}], upper_bound = 16 : i64}
      %19 = loom.expression(%12, %14) {coeffs = [1, 1], logic = "mul"} : index
      loom.polynomial_constraint(%13, %12, %14, %19) {monomials = [{coeff = 1024 : i64, vars = [0]}, {coeff = 1 : i64, vars = [3]}], upper_bound = 374784 : i64}
      %20 = loom.expression(%12, %14) {coeffs = [1, 1], logic = "mul"} : index
      %21 = loom.expression(%13, %14) {coeffs = [1, 1], logic = "mul"} : index
      %22 = loom.expression(%12, %13) {coeffs = [1, 1], logic = "mul"} : index
      %23 = loom.expression(%14, %22) {coeffs = [1, 1], logic = "mul"} : index
      loom.polynomial_constraint(%12, %13, %14, %20, %21, %22, %23) {monomials = [{coeff = 256 : i64, vars = [3]}, {coeff = 960 : i64, vars = [4]}, {coeff = -15 : i64, vars = [6]}], upper_bound = 0 : i64}
    }
    func.func @matmul_kernel__d0i0_d1i0__f10__hoist_block_1__a_d(%arg0: memref<*xf32> {tt.divisibility = 16 : i32}, %arg1: memref<*xf32> {tt.divisibility = 16 : i32}, %arg2: memref<*xf32> {tt.divisibility = 16 : i32}, %arg3: index, %arg4: index, %arg5: index, %arg6: index, %arg7: index, %arg8: index) {
      affine.parallel (%arg9) = (0) to (8) {
        affine.parallel (%arg10) = (0) to (8) {
          affine.for %arg11 = 0 to affine_map<()[s0, s1] -> (s1)>()[%arg3, %arg4] {
            affine.for %arg12 = 0 to affine_map<()[s0, s1] -> ((s0 ceildiv 8) ceildiv 8)>()[%arg3, %arg4] {
              %12 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 8 + d1 * 8 + d2)>(%arg9, %arg10, %arg12)
              %c1024 = arith.constant 1024 : index
              %c1024_0 = arith.constant 1024 : index
              %c1 = arith.constant 1 : index
              %13 = loom.get_symbolic_block_size @constraints::@BM : index
              %14 = loom.get_symbolic_block_size @constraints::@BN : index
              %15 = loom.get_symbolic_block_size @constraints::@BK : index
              %16 = arith.ceildivsi %c1024_0, %15 : index
              %cst = arith.constant 0.000000e+00 : f32
              %c1024_1 = arith.constant 1024 : index
              %c0 = arith.constant 0 : index
              %17 = tensor.empty(%13, %14) : tensor<?x?xf32>
              %18 = linalg.fill ins(%cst : f32) outs(%17 : tensor<?x?xf32>) -> tensor<?x?xf32>
              %19 = arith.muli %12, %13 : index
              %20 = arith.muli %arg11, %14 : index
              %21 = arith.muli %15, %c1024 : index
              %c0_2 = arith.constant 0 : index
              %22 = arith.muli %c0_2, %15 : index
              %23 = arith.muli %22, %c1024_1 : index
              %24 = arith.addi %23, %20 : index
              %25 = loom.reinterpret_cast %arg1 to offset : [%24], sizes : [%16, %15, %14], strides : [%21, %c1024, %c1], reuse : [seq = false, spat = true, temp = true] : memref<*xf32> to memref<?x?x?xf32, strided<[?, ?, ?], offset: ?>>
              %26 = loom.alloc(%16, %15, %14) on @L1 : memref<?x?x?xf32>
              loom.copy %25, %26, interconnect : [@horizontal_links, @vertical_links], broadcast : [8, 8] : memref<?x?x?xf32, strided<[?, ?, ?], offset: ?>> to memref<?x?x?xf32>
              %27 = scf.for %arg13 = %c0 to %16 step %c1 iter_args(%arg14 = %18) -> (tensor<?x?xf32>) {
                %31 = arith.muli %arg13, %15 : index
                %32 = arith.muli %19, %c1024_1 : index
                %33 = arith.addi %32, %31 : index
                %34 = loom.reinterpret_cast %arg0 to offset : [%33], sizes : [%13, %15], strides : [%c1024_0, %c1], reuse : [seq = false, spat = false, temp = true] : memref<*xf32> to memref<?x?xf32, strided<[?, ?], offset: ?>>
                %35 = loom.alloc(%13, %15) on @L1 : memref<?x?xf32>
                loom.copy %34, %35, interconnect : [], broadcast : [1, 1] : memref<?x?xf32, strided<[?, ?], offset: ?>> to memref<?x?xf32>
                %36 = bufferization.to_tensor %35 restrict writable : memref<?x?xf32> to tensor<?x?xf32>
                %37 = arith.muli %31, %c1024_1 : index
                %subview = memref.subview %26[%arg13, 0, 0] [1, %15, %14] [1, 1, 1] : memref<?x?x?xf32> to memref<?x?xf32, strided<[?, 1], offset: ?>>
                %38 = bufferization.to_tensor %subview restrict writable : memref<?x?xf32, strided<[?, 1], offset: ?>> to tensor<?x?xf32>
                %39 = linalg.matmul ins(%36, %38 : tensor<?x?xf32>, tensor<?x?xf32>) outs(%18 : tensor<?x?xf32>) -> tensor<?x?xf32>
                %40 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%arg14, %39 : tensor<?x?xf32>, tensor<?x?xf32>) outs(%arg14 : tensor<?x?xf32>) {
                ^bb0(%in: f32, %in_3: f32, %out: f32):
                  %41 = arith.addf %in, %in_3 : f32
                  linalg.yield %41 : f32
                } -> tensor<?x?xf32>
                scf.yield %40 : tensor<?x?xf32>
              }
              %28 = arith.muli %19, %c1024_1 : index
              %29 = arith.addi %28, %20 : index
              %30 = loom.reinterpret_cast %arg2 to offset : [%29], sizes : [%13, %14], strides : [%c1024, %c1], reuse : [seq = false, spat = false, temp = false] : memref<*xf32> to memref<?x?xf32, strided<[?, ?], offset: ?>>
              bufferization.materialize_in_destination %27 in writable %30 : (tensor<?x?xf32>, memref<?x?xf32, strided<[?, ?], offset: ?>>) -> ()
            }
          }
        } {loom.mapped_to = @y}
      } {loom.mapped_to = @x}
      return
    }
  }
  module attributes {loom.pass_name = "EnumerateCopyBroadcast"} {
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
      %15 = loom.expression(%13, %12) {coeffs = [1, 1], logic = "mul"} : index
      %16 = loom.expression(%15, %14) {coeffs = [1, 1], logic = "mul"} : index
      loom.polynomial_constraint(%12, %13, %14, %15, %16) {monomials = [{coeff = -1 : i64, vars = [4]}], upper_bound = -262144 : i64}
      %17 = loom.expression(%12, %13) {coeffs = [1, 1], logic = "add"} : index
      %18 = loom.expression(%14, %17) {coeffs = [1, 1], logic = "mul"} : index
      loom.polynomial_constraint(%12, %14, %13, %17, %18) {monomials = [{coeff = 1 : i64, vars = [4]}], upper_bound = 374784 : i64}
      loom.polynomial_constraint(%12) {monomials = [{coeff = 1 : i64, vars = [0]}], upper_bound = 16 : i64}
      %19 = loom.expression(%14, %12) {coeffs = [1, 1], logic = "mul"} : index
      loom.polynomial_constraint(%13, %12, %14, %19) {monomials = [{coeff = 1024 : i64, vars = [0]}, {coeff = 1 : i64, vars = [3]}], upper_bound = 374784 : i64}
      %20 = loom.expression(%14, %12) {coeffs = [1, 1], logic = "mul"} : index
      %21 = loom.expression(%13, %14) {coeffs = [1, 1], logic = "mul"} : index
      %22 = loom.expression(%13, %12) {coeffs = [1, 1], logic = "mul"} : index
      %23 = loom.expression(%14, %22) {coeffs = [1, 1], logic = "mul"} : index
      loom.polynomial_constraint(%12, %13, %14, %20, %21, %22, %23) {monomials = [{coeff = 256 : i64, vars = [3]}, {coeff = 960 : i64, vars = [4]}, {coeff = -15 : i64, vars = [6]}], upper_bound = 0 : i64}
    }
    func.func @matmul_kernel__d0i0_d1i0__f10__hoist_block_1__h_d(%arg0: memref<*xf32> {tt.divisibility = 16 : i32}, %arg1: memref<*xf32> {tt.divisibility = 16 : i32}, %arg2: memref<*xf32> {tt.divisibility = 16 : i32}, %arg3: index, %arg4: index, %arg5: index, %arg6: index, %arg7: index, %arg8: index) {
      affine.parallel (%arg9) = (0) to (8) {
        affine.parallel (%arg10) = (0) to (8) {
          affine.for %arg11 = 0 to affine_map<()[s0, s1] -> (s1)>()[%arg3, %arg4] {
            affine.for %arg12 = 0 to affine_map<()[s0, s1] -> ((s0 ceildiv 8) ceildiv 8)>()[%arg3, %arg4] {
              %12 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 8 + d1 * 8 + d2)>(%arg9, %arg10, %arg12)
              %c1024 = arith.constant 1024 : index
              %c1024_0 = arith.constant 1024 : index
              %c1 = arith.constant 1 : index
              %13 = loom.get_symbolic_block_size @constraints::@BM : index
              %14 = loom.get_symbolic_block_size @constraints::@BN : index
              %15 = loom.get_symbolic_block_size @constraints::@BK : index
              %16 = arith.ceildivsi %c1024_0, %15 : index
              %cst = arith.constant 0.000000e+00 : f32
              %c1024_1 = arith.constant 1024 : index
              %c0 = arith.constant 0 : index
              %17 = tensor.empty(%13, %14) : tensor<?x?xf32>
              %18 = linalg.fill ins(%cst : f32) outs(%17 : tensor<?x?xf32>) -> tensor<?x?xf32>
              %19 = arith.muli %12, %13 : index
              %20 = arith.muli %arg11, %14 : index
              %21 = arith.muli %15, %c1024 : index
              %c0_2 = arith.constant 0 : index
              %22 = arith.muli %c0_2, %15 : index
              %23 = arith.muli %22, %c1024_1 : index
              %24 = arith.addi %23, %20 : index
              %25 = loom.reinterpret_cast %arg1 to offset : [%24], sizes : [%16, %15, %14], strides : [%21, %c1024, %c1], reuse : [seq = false, spat = true, temp = true] : memref<*xf32> to memref<?x?x?xf32, strided<[?, ?, ?], offset: ?>>
              %26 = loom.alloc(%16, %15, %14) on @L1 : memref<?x?x?xf32>
              loom.copy %25, %26, interconnect : [@horizontal_links], broadcast : [1, 8] : memref<?x?x?xf32, strided<[?, ?, ?], offset: ?>> to memref<?x?x?xf32>
              %27 = scf.for %arg13 = %c0 to %16 step %c1 iter_args(%arg14 = %18) -> (tensor<?x?xf32>) {
                %31 = arith.muli %arg13, %15 : index
                %32 = arith.muli %19, %c1024_1 : index
                %33 = arith.addi %32, %31 : index
                %34 = loom.reinterpret_cast %arg0 to offset : [%33], sizes : [%13, %15], strides : [%c1024_0, %c1], reuse : [seq = false, spat = false, temp = true] : memref<*xf32> to memref<?x?xf32, strided<[?, ?], offset: ?>>
                %35 = loom.alloc(%13, %15) on @L1 : memref<?x?xf32>
                loom.copy %34, %35, interconnect : [], broadcast : [1, 1] : memref<?x?xf32, strided<[?, ?], offset: ?>> to memref<?x?xf32>
                %36 = bufferization.to_tensor %35 restrict writable : memref<?x?xf32> to tensor<?x?xf32>
                %37 = arith.muli %31, %c1024_1 : index
                %subview = memref.subview %26[%arg13, 0, 0] [1, %15, %14] [1, 1, 1] : memref<?x?x?xf32> to memref<?x?xf32, strided<[?, 1], offset: ?>>
                %38 = bufferization.to_tensor %subview restrict writable : memref<?x?xf32, strided<[?, 1], offset: ?>> to tensor<?x?xf32>
                %39 = linalg.matmul ins(%36, %38 : tensor<?x?xf32>, tensor<?x?xf32>) outs(%18 : tensor<?x?xf32>) -> tensor<?x?xf32>
                %40 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%arg14, %39 : tensor<?x?xf32>, tensor<?x?xf32>) outs(%arg14 : tensor<?x?xf32>) {
                ^bb0(%in: f32, %in_3: f32, %out: f32):
                  %41 = arith.addf %in, %in_3 : f32
                  linalg.yield %41 : f32
                } -> tensor<?x?xf32>
                scf.yield %40 : tensor<?x?xf32>
              }
              %28 = arith.muli %19, %c1024_1 : index
              %29 = arith.addi %28, %20 : index
              %30 = loom.reinterpret_cast %arg2 to offset : [%29], sizes : [%13, %14], strides : [%c1024, %c1], reuse : [seq = false, spat = false, temp = false] : memref<*xf32> to memref<?x?xf32, strided<[?, ?], offset: ?>>
              bufferization.materialize_in_destination %27 in writable %30 : (tensor<?x?xf32>, memref<?x?xf32, strided<[?, ?], offset: ?>>) -> ()
            }
          }
        } {loom.mapped_to = @y}
      } {loom.mapped_to = @x}
      return
    }
  }
  module attributes {loom.pass_name = "EnumerateCopyBroadcast"} {
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
      %15 = loom.expression(%13, %12) {coeffs = [1, 1], logic = "mul"} : index
      %16 = loom.expression(%15, %14) {coeffs = [1, 1], logic = "mul"} : index
      loom.polynomial_constraint(%12, %13, %14, %15, %16) {monomials = [{coeff = -1 : i64, vars = [4]}], upper_bound = -262144 : i64}
      %17 = loom.expression(%12, %13) {coeffs = [1, 1], logic = "add"} : index
      %18 = loom.expression(%14, %17) {coeffs = [1, 1], logic = "mul"} : index
      loom.polynomial_constraint(%12, %14, %13, %17, %18) {monomials = [{coeff = 1 : i64, vars = [4]}], upper_bound = 374784 : i64}
      loom.polynomial_constraint(%12) {monomials = [{coeff = 1 : i64, vars = [0]}], upper_bound = 16 : i64}
      %19 = loom.expression(%14, %12) {coeffs = [1, 1], logic = "mul"} : index
      loom.polynomial_constraint(%13, %12, %14, %19) {monomials = [{coeff = 1024 : i64, vars = [0]}, {coeff = 1 : i64, vars = [3]}], upper_bound = 374784 : i64}
      %20 = loom.expression(%14, %12) {coeffs = [1, 1], logic = "mul"} : index
      %21 = loom.expression(%13, %14) {coeffs = [1, 1], logic = "mul"} : index
      %22 = loom.expression(%13, %12) {coeffs = [1, 1], logic = "mul"} : index
      %23 = loom.expression(%14, %22) {coeffs = [1, 1], logic = "mul"} : index
      loom.polynomial_constraint(%12, %13, %14, %20, %21, %22, %23) {monomials = [{coeff = 256 : i64, vars = [3]}, {coeff = 960 : i64, vars = [4]}, {coeff = -15 : i64, vars = [6]}], upper_bound = 0 : i64}
    }
    func.func @matmul_kernel__d0i0_d1i0__f10__hoist_block_1__v_d(%arg0: memref<*xf32> {tt.divisibility = 16 : i32}, %arg1: memref<*xf32> {tt.divisibility = 16 : i32}, %arg2: memref<*xf32> {tt.divisibility = 16 : i32}, %arg3: index, %arg4: index, %arg5: index, %arg6: index, %arg7: index, %arg8: index) {
      affine.parallel (%arg9) = (0) to (8) {
        affine.parallel (%arg10) = (0) to (8) {
          affine.for %arg11 = 0 to affine_map<()[s0, s1] -> (s1)>()[%arg3, %arg4] {
            affine.for %arg12 = 0 to affine_map<()[s0, s1] -> ((s0 ceildiv 8) ceildiv 8)>()[%arg3, %arg4] {
              %12 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 8 + d1 * 8 + d2)>(%arg9, %arg10, %arg12)
              %c1024 = arith.constant 1024 : index
              %c1024_0 = arith.constant 1024 : index
              %c1 = arith.constant 1 : index
              %13 = loom.get_symbolic_block_size @constraints::@BM : index
              %14 = loom.get_symbolic_block_size @constraints::@BN : index
              %15 = loom.get_symbolic_block_size @constraints::@BK : index
              %16 = arith.ceildivsi %c1024_0, %15 : index
              %cst = arith.constant 0.000000e+00 : f32
              %c1024_1 = arith.constant 1024 : index
              %c0 = arith.constant 0 : index
              %17 = tensor.empty(%13, %14) : tensor<?x?xf32>
              %18 = linalg.fill ins(%cst : f32) outs(%17 : tensor<?x?xf32>) -> tensor<?x?xf32>
              %19 = arith.muli %12, %13 : index
              %20 = arith.muli %arg11, %14 : index
              %21 = arith.muli %15, %c1024 : index
              %c0_2 = arith.constant 0 : index
              %22 = arith.muli %c0_2, %15 : index
              %23 = arith.muli %22, %c1024_1 : index
              %24 = arith.addi %23, %20 : index
              %25 = loom.reinterpret_cast %arg1 to offset : [%24], sizes : [%16, %15, %14], strides : [%21, %c1024, %c1], reuse : [seq = false, spat = true, temp = true] : memref<*xf32> to memref<?x?x?xf32, strided<[?, ?, ?], offset: ?>>
              %26 = loom.alloc(%16, %15, %14) on @L1 : memref<?x?x?xf32>
              loom.copy %25, %26, interconnect : [@vertical_links], broadcast : [8, 1] : memref<?x?x?xf32, strided<[?, ?, ?], offset: ?>> to memref<?x?x?xf32>
              %27 = scf.for %arg13 = %c0 to %16 step %c1 iter_args(%arg14 = %18) -> (tensor<?x?xf32>) {
                %31 = arith.muli %arg13, %15 : index
                %32 = arith.muli %19, %c1024_1 : index
                %33 = arith.addi %32, %31 : index
                %34 = loom.reinterpret_cast %arg0 to offset : [%33], sizes : [%13, %15], strides : [%c1024_0, %c1], reuse : [seq = false, spat = false, temp = true] : memref<*xf32> to memref<?x?xf32, strided<[?, ?], offset: ?>>
                %35 = loom.alloc(%13, %15) on @L1 : memref<?x?xf32>
                loom.copy %34, %35, interconnect : [], broadcast : [1, 1] : memref<?x?xf32, strided<[?, ?], offset: ?>> to memref<?x?xf32>
                %36 = bufferization.to_tensor %35 restrict writable : memref<?x?xf32> to tensor<?x?xf32>
                %37 = arith.muli %31, %c1024_1 : index
                %subview = memref.subview %26[%arg13, 0, 0] [1, %15, %14] [1, 1, 1] : memref<?x?x?xf32> to memref<?x?xf32, strided<[?, 1], offset: ?>>
                %38 = bufferization.to_tensor %subview restrict writable : memref<?x?xf32, strided<[?, 1], offset: ?>> to tensor<?x?xf32>
                %39 = linalg.matmul ins(%36, %38 : tensor<?x?xf32>, tensor<?x?xf32>) outs(%18 : tensor<?x?xf32>) -> tensor<?x?xf32>
                %40 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%arg14, %39 : tensor<?x?xf32>, tensor<?x?xf32>) outs(%arg14 : tensor<?x?xf32>) {
                ^bb0(%in: f32, %in_3: f32, %out: f32):
                  %41 = arith.addf %in, %in_3 : f32
                  linalg.yield %41 : f32
                } -> tensor<?x?xf32>
                scf.yield %40 : tensor<?x?xf32>
              }
              %28 = arith.muli %19, %c1024_1 : index
              %29 = arith.addi %28, %20 : index
              %30 = loom.reinterpret_cast %arg2 to offset : [%29], sizes : [%13, %14], strides : [%c1024, %c1], reuse : [seq = false, spat = false, temp = false] : memref<*xf32> to memref<?x?xf32, strided<[?, ?], offset: ?>>
              bufferization.materialize_in_destination %27 in writable %30 : (tensor<?x?xf32>, memref<?x?xf32, strided<[?, ?], offset: ?>>) -> ()
            }
          }
        } {loom.mapped_to = @y}
      } {loom.mapped_to = @x}
      return
    }
  }
  module attributes {loom.pass_name = "EnumerateCopyBroadcast"} {
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
      %15 = loom.expression(%13, %12) {coeffs = [1, 1], logic = "mul"} : index
      %16 = loom.expression(%15, %14) {coeffs = [1, 1], logic = "mul"} : index
      loom.polynomial_constraint(%12, %13, %14, %15, %16) {monomials = [{coeff = -1 : i64, vars = [4]}], upper_bound = -262144 : i64}
      %17 = loom.expression(%12, %13) {coeffs = [1, 1], logic = "add"} : index
      %18 = loom.expression(%14, %17) {coeffs = [1, 1], logic = "mul"} : index
      loom.polynomial_constraint(%12, %14, %13, %17, %18) {monomials = [{coeff = 1 : i64, vars = [4]}], upper_bound = 374784 : i64}
      loom.polynomial_constraint(%12) {monomials = [{coeff = 1 : i64, vars = [0]}], upper_bound = 128 : i64}
      loom.polynomial_constraint(%13) {monomials = [{coeff = 1 : i64, vars = [0]}], upper_bound = 128 : i64}
      %19 = loom.expression(%14, %12) {coeffs = [1, 1], logic = "mul"} : index
      %20 = loom.expression(%13, %14) {coeffs = [1, 1], logic = "mul"} : index
      %21 = loom.expression(%13, %12) {coeffs = [1, 1], logic = "mul"} : index
      %22 = loom.expression(%14, %21) {coeffs = [1, 1], logic = "mul"} : index
      loom.polynomial_constraint(%12, %13, %14, %19, %20, %21, %22) {monomials = [{coeff = 64 : i64, vars = [3]}, {coeff = 64 : i64, vars = [4]}, {coeff = -1 : i64, vars = [6]}], upper_bound = 0 : i64}
    }
    func.func @matmul_kernel__d0i0_d1i1__f01__d_d(%arg0: memref<*xf32> {tt.divisibility = 16 : i32}, %arg1: memref<*xf32> {tt.divisibility = 16 : i32}, %arg2: memref<*xf32> {tt.divisibility = 16 : i32}, %arg3: index, %arg4: index, %arg5: index, %arg6: index, %arg7: index, %arg8: index) {
      affine.parallel (%arg9) = (0) to (8) {
        affine.parallel (%arg10) = (0) to (8) {
          affine.for %arg11 = 0 to affine_map<()[s0, s1] -> (s0 ceildiv 8)>()[%arg3, %arg4] {
            affine.for %arg12 = 0 to affine_map<()[s0, s1] -> (s1 ceildiv 8)>()[%arg3, %arg4] {
              %12 = affine.apply affine_map<(d0, d1) -> (d0 * 8 + d1)>(%arg10, %arg12)
              %13 = affine.apply affine_map<(d0, d1) -> (d0 * 8 + d1)>(%arg9, %arg11)
              %c1024 = arith.constant 1024 : index
              %c1024_0 = arith.constant 1024 : index
              %c1 = arith.constant 1 : index
              %14 = loom.get_symbolic_block_size @constraints::@BM : index
              %15 = loom.get_symbolic_block_size @constraints::@BN : index
              %16 = loom.get_symbolic_block_size @constraints::@BK : index
              %17 = arith.ceildivsi %c1024_0, %16 : index
              %cst = arith.constant 0.000000e+00 : f32
              %c1024_1 = arith.constant 1024 : index
              %c0 = arith.constant 0 : index
              %18 = tensor.empty(%14, %15) : tensor<?x?xf32>
              %19 = linalg.fill ins(%cst : f32) outs(%18 : tensor<?x?xf32>) -> tensor<?x?xf32>
              %20 = arith.muli %13, %14 : index
              %21 = arith.muli %12, %15 : index
              %22 = scf.for %arg13 = %c0 to %17 step %c1 iter_args(%arg14 = %19) -> (tensor<?x?xf32>) {
                %26 = arith.muli %arg13, %16 : index
                %27 = arith.muli %20, %c1024_1 : index
                %28 = arith.addi %27, %26 : index
                %29 = loom.reinterpret_cast %arg0 to offset : [%28], sizes : [%14, %16], strides : [%c1024_0, %c1], reuse : [seq = false, spat = true, temp = true] : memref<*xf32> to memref<?x?xf32, strided<[?, ?], offset: ?>>
                %30 = loom.alloc(%14, %16) on @L1 : memref<?x?xf32>
                loom.copy %29, %30, interconnect : [], broadcast : [1, 1] : memref<?x?xf32, strided<[?, ?], offset: ?>> to memref<?x?xf32>
                %31 = bufferization.to_tensor %30 restrict writable : memref<?x?xf32> to tensor<?x?xf32>
                %32 = arith.muli %26, %c1024_1 : index
                %33 = arith.addi %32, %21 : index
                %34 = loom.reinterpret_cast %arg1 to offset : [%33], sizes : [%16, %15], strides : [%c1024, %c1], reuse : [seq = false, spat = true, temp = true] : memref<*xf32> to memref<?x?xf32, strided<[?, ?], offset: ?>>
                %35 = loom.alloc(%16, %15) on @L1 : memref<?x?xf32>
                loom.copy %34, %35, interconnect : [], broadcast : [1, 1] : memref<?x?xf32, strided<[?, ?], offset: ?>> to memref<?x?xf32>
                %36 = bufferization.to_tensor %35 restrict writable : memref<?x?xf32> to tensor<?x?xf32>
                %37 = linalg.matmul ins(%31, %36 : tensor<?x?xf32>, tensor<?x?xf32>) outs(%19 : tensor<?x?xf32>) -> tensor<?x?xf32>
                %38 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%arg14, %37 : tensor<?x?xf32>, tensor<?x?xf32>) outs(%arg14 : tensor<?x?xf32>) {
                ^bb0(%in: f32, %in_2: f32, %out: f32):
                  %39 = arith.addf %in, %in_2 : f32
                  linalg.yield %39 : f32
                } -> tensor<?x?xf32>
                scf.yield %38 : tensor<?x?xf32>
              }
              %23 = arith.muli %20, %c1024_1 : index
              %24 = arith.addi %23, %21 : index
              %25 = loom.reinterpret_cast %arg2 to offset : [%24], sizes : [%14, %15], strides : [%c1024, %c1], reuse : [seq = false, spat = false, temp = false] : memref<*xf32> to memref<?x?xf32, strided<[?, ?], offset: ?>>
              bufferization.materialize_in_destination %22 in writable %25 : (tensor<?x?xf32>, memref<?x?xf32, strided<[?, ?], offset: ?>>) -> ()
            }
          }
        } {loom.mapped_to = @y}
      } {loom.mapped_to = @x}
      return
    }
  }
  module attributes {loom.pass_name = "EnumerateCopyBroadcast"} {
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
      %15 = loom.expression(%12, %13) {coeffs = [1, 1], logic = "mul"} : index
      %16 = loom.expression(%15, %14) {coeffs = [1, 1], logic = "mul"} : index
      loom.polynomial_constraint(%12, %13, %14, %15, %16) {monomials = [{coeff = -1 : i64, vars = [4]}], upper_bound = -262144 : i64}
      %17 = loom.expression(%12, %13) {coeffs = [1, 1], logic = "add"} : index
      %18 = loom.expression(%17, %14) {coeffs = [1, 1], logic = "mul"} : index
      loom.polynomial_constraint(%12, %14, %13, %17, %18) {monomials = [{coeff = 1 : i64, vars = [4]}], upper_bound = 374784 : i64}
      loom.polynomial_constraint(%12) {monomials = [{coeff = 1 : i64, vars = [0]}], upper_bound = 128 : i64}
      loom.polynomial_constraint(%13) {monomials = [{coeff = 1 : i64, vars = [0]}], upper_bound = 128 : i64}
      %19 = loom.expression(%12, %14) {coeffs = [1, 1], logic = "mul"} : index
      %20 = loom.expression(%13, %14) {coeffs = [1, 1], logic = "mul"} : index
      %21 = loom.expression(%12, %13) {coeffs = [1, 1], logic = "mul"} : index
      %22 = loom.expression(%14, %21) {coeffs = [1, 1], logic = "mul"} : index
      loom.polynomial_constraint(%12, %13, %14, %19, %20, %21, %22) {monomials = [{coeff = 960 : i64, vars = [3]}, {coeff = 256 : i64, vars = [4]}, {coeff = -15 : i64, vars = [6]}], upper_bound = 0 : i64}
    }
    func.func @matmul_kernel__d0i0_d1i1__f01__d_v(%arg0: memref<*xf32> {tt.divisibility = 16 : i32}, %arg1: memref<*xf32> {tt.divisibility = 16 : i32}, %arg2: memref<*xf32> {tt.divisibility = 16 : i32}, %arg3: index, %arg4: index, %arg5: index, %arg6: index, %arg7: index, %arg8: index) {
      affine.parallel (%arg9) = (0) to (8) {
        affine.parallel (%arg10) = (0) to (8) {
          affine.for %arg11 = 0 to affine_map<()[s0, s1] -> (s0 ceildiv 8)>()[%arg3, %arg4] {
            affine.for %arg12 = 0 to affine_map<()[s0, s1] -> (s1 ceildiv 8)>()[%arg3, %arg4] {
              %12 = affine.apply affine_map<(d0, d1) -> (d0 * 8 + d1)>(%arg10, %arg12)
              %13 = affine.apply affine_map<(d0, d1) -> (d0 * 8 + d1)>(%arg9, %arg11)
              %c1024 = arith.constant 1024 : index
              %c1024_0 = arith.constant 1024 : index
              %c1 = arith.constant 1 : index
              %14 = loom.get_symbolic_block_size @constraints::@BM : index
              %15 = loom.get_symbolic_block_size @constraints::@BN : index
              %16 = loom.get_symbolic_block_size @constraints::@BK : index
              %17 = arith.ceildivsi %c1024_0, %16 : index
              %cst = arith.constant 0.000000e+00 : f32
              %c1024_1 = arith.constant 1024 : index
              %c0 = arith.constant 0 : index
              %18 = tensor.empty(%14, %15) : tensor<?x?xf32>
              %19 = linalg.fill ins(%cst : f32) outs(%18 : tensor<?x?xf32>) -> tensor<?x?xf32>
              %20 = arith.muli %13, %14 : index
              %21 = arith.muli %12, %15 : index
              %22 = scf.for %arg13 = %c0 to %17 step %c1 iter_args(%arg14 = %19) -> (tensor<?x?xf32>) {
                %26 = arith.muli %arg13, %16 : index
                %27 = arith.muli %20, %c1024_1 : index
                %28 = arith.addi %27, %26 : index
                %29 = loom.reinterpret_cast %arg0 to offset : [%28], sizes : [%14, %16], strides : [%c1024_0, %c1], reuse : [seq = false, spat = true, temp = true] : memref<*xf32> to memref<?x?xf32, strided<[?, ?], offset: ?>>
                %30 = loom.alloc(%14, %16) on @L1 : memref<?x?xf32>
                loom.copy %29, %30, interconnect : [], broadcast : [1, 1] : memref<?x?xf32, strided<[?, ?], offset: ?>> to memref<?x?xf32>
                %31 = bufferization.to_tensor %30 restrict writable : memref<?x?xf32> to tensor<?x?xf32>
                %32 = arith.muli %26, %c1024_1 : index
                %33 = arith.addi %32, %21 : index
                %34 = loom.reinterpret_cast %arg1 to offset : [%33], sizes : [%16, %15], strides : [%c1024, %c1], reuse : [seq = false, spat = true, temp = true] : memref<*xf32> to memref<?x?xf32, strided<[?, ?], offset: ?>>
                %35 = loom.alloc(%16, %15) on @L1 : memref<?x?xf32>
                loom.copy %34, %35, interconnect : [@vertical_links], broadcast : [8, 1] : memref<?x?xf32, strided<[?, ?], offset: ?>> to memref<?x?xf32>
                %36 = bufferization.to_tensor %35 restrict writable : memref<?x?xf32> to tensor<?x?xf32>
                %37 = linalg.matmul ins(%31, %36 : tensor<?x?xf32>, tensor<?x?xf32>) outs(%19 : tensor<?x?xf32>) -> tensor<?x?xf32>
                %38 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%arg14, %37 : tensor<?x?xf32>, tensor<?x?xf32>) outs(%arg14 : tensor<?x?xf32>) {
                ^bb0(%in: f32, %in_2: f32, %out: f32):
                  %39 = arith.addf %in, %in_2 : f32
                  linalg.yield %39 : f32
                } -> tensor<?x?xf32>
                scf.yield %38 : tensor<?x?xf32>
              }
              %23 = arith.muli %20, %c1024_1 : index
              %24 = arith.addi %23, %21 : index
              %25 = loom.reinterpret_cast %arg2 to offset : [%24], sizes : [%14, %15], strides : [%c1024, %c1], reuse : [seq = false, spat = false, temp = false] : memref<*xf32> to memref<?x?xf32, strided<[?, ?], offset: ?>>
              bufferization.materialize_in_destination %22 in writable %25 : (tensor<?x?xf32>, memref<?x?xf32, strided<[?, ?], offset: ?>>) -> ()
            }
          }
        } {loom.mapped_to = @y}
      } {loom.mapped_to = @x}
      return
    }
  }
  module attributes {loom.pass_name = "EnumerateCopyBroadcast"} {
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
      %15 = loom.expression(%12, %13) {coeffs = [1, 1], logic = "mul"} : index
      %16 = loom.expression(%15, %14) {coeffs = [1, 1], logic = "mul"} : index
      loom.polynomial_constraint(%12, %13, %14, %15, %16) {monomials = [{coeff = -1 : i64, vars = [4]}], upper_bound = -262144 : i64}
      %17 = loom.expression(%12, %13) {coeffs = [1, 1], logic = "add"} : index
      %18 = loom.expression(%17, %14) {coeffs = [1, 1], logic = "mul"} : index
      loom.polynomial_constraint(%12, %14, %13, %17, %18) {monomials = [{coeff = 1 : i64, vars = [4]}], upper_bound = 374784 : i64}
      loom.polynomial_constraint(%12) {monomials = [{coeff = 1 : i64, vars = [0]}], upper_bound = 128 : i64}
      loom.polynomial_constraint(%13) {monomials = [{coeff = 1 : i64, vars = [0]}], upper_bound = 128 : i64}
      %19 = loom.expression(%12, %14) {coeffs = [1, 1], logic = "mul"} : index
      %20 = loom.expression(%13, %14) {coeffs = [1, 1], logic = "mul"} : index
      %21 = loom.expression(%12, %13) {coeffs = [1, 1], logic = "mul"} : index
      %22 = loom.expression(%14, %21) {coeffs = [1, 1], logic = "mul"} : index
      loom.polynomial_constraint(%12, %13, %14, %19, %20, %21, %22) {monomials = [{coeff = 256 : i64, vars = [3]}, {coeff = 960 : i64, vars = [4]}, {coeff = -15 : i64, vars = [6]}], upper_bound = 0 : i64}
    }
    func.func @matmul_kernel__d0i0_d1i1__f01__h_d(%arg0: memref<*xf32> {tt.divisibility = 16 : i32}, %arg1: memref<*xf32> {tt.divisibility = 16 : i32}, %arg2: memref<*xf32> {tt.divisibility = 16 : i32}, %arg3: index, %arg4: index, %arg5: index, %arg6: index, %arg7: index, %arg8: index) {
      affine.parallel (%arg9) = (0) to (8) {
        affine.parallel (%arg10) = (0) to (8) {
          affine.for %arg11 = 0 to affine_map<()[s0, s1] -> (s0 ceildiv 8)>()[%arg3, %arg4] {
            affine.for %arg12 = 0 to affine_map<()[s0, s1] -> (s1 ceildiv 8)>()[%arg3, %arg4] {
              %12 = affine.apply affine_map<(d0, d1) -> (d0 * 8 + d1)>(%arg10, %arg12)
              %13 = affine.apply affine_map<(d0, d1) -> (d0 * 8 + d1)>(%arg9, %arg11)
              %c1024 = arith.constant 1024 : index
              %c1024_0 = arith.constant 1024 : index
              %c1 = arith.constant 1 : index
              %14 = loom.get_symbolic_block_size @constraints::@BM : index
              %15 = loom.get_symbolic_block_size @constraints::@BN : index
              %16 = loom.get_symbolic_block_size @constraints::@BK : index
              %17 = arith.ceildivsi %c1024_0, %16 : index
              %cst = arith.constant 0.000000e+00 : f32
              %c1024_1 = arith.constant 1024 : index
              %c0 = arith.constant 0 : index
              %18 = tensor.empty(%14, %15) : tensor<?x?xf32>
              %19 = linalg.fill ins(%cst : f32) outs(%18 : tensor<?x?xf32>) -> tensor<?x?xf32>
              %20 = arith.muli %13, %14 : index
              %21 = arith.muli %12, %15 : index
              %22 = scf.for %arg13 = %c0 to %17 step %c1 iter_args(%arg14 = %19) -> (tensor<?x?xf32>) {
                %26 = arith.muli %arg13, %16 : index
                %27 = arith.muli %20, %c1024_1 : index
                %28 = arith.addi %27, %26 : index
                %29 = loom.reinterpret_cast %arg0 to offset : [%28], sizes : [%14, %16], strides : [%c1024_0, %c1], reuse : [seq = false, spat = true, temp = true] : memref<*xf32> to memref<?x?xf32, strided<[?, ?], offset: ?>>
                %30 = loom.alloc(%14, %16) on @L1 : memref<?x?xf32>
                loom.copy %29, %30, interconnect : [@horizontal_links], broadcast : [1, 8] : memref<?x?xf32, strided<[?, ?], offset: ?>> to memref<?x?xf32>
                %31 = bufferization.to_tensor %30 restrict writable : memref<?x?xf32> to tensor<?x?xf32>
                %32 = arith.muli %26, %c1024_1 : index
                %33 = arith.addi %32, %21 : index
                %34 = loom.reinterpret_cast %arg1 to offset : [%33], sizes : [%16, %15], strides : [%c1024, %c1], reuse : [seq = false, spat = true, temp = true] : memref<*xf32> to memref<?x?xf32, strided<[?, ?], offset: ?>>
                %35 = loom.alloc(%16, %15) on @L1 : memref<?x?xf32>
                loom.copy %34, %35, interconnect : [], broadcast : [1, 1] : memref<?x?xf32, strided<[?, ?], offset: ?>> to memref<?x?xf32>
                %36 = bufferization.to_tensor %35 restrict writable : memref<?x?xf32> to tensor<?x?xf32>
                %37 = linalg.matmul ins(%31, %36 : tensor<?x?xf32>, tensor<?x?xf32>) outs(%19 : tensor<?x?xf32>) -> tensor<?x?xf32>
                %38 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%arg14, %37 : tensor<?x?xf32>, tensor<?x?xf32>) outs(%arg14 : tensor<?x?xf32>) {
                ^bb0(%in: f32, %in_2: f32, %out: f32):
                  %39 = arith.addf %in, %in_2 : f32
                  linalg.yield %39 : f32
                } -> tensor<?x?xf32>
                scf.yield %38 : tensor<?x?xf32>
              }
              %23 = arith.muli %20, %c1024_1 : index
              %24 = arith.addi %23, %21 : index
              %25 = loom.reinterpret_cast %arg2 to offset : [%24], sizes : [%14, %15], strides : [%c1024, %c1], reuse : [seq = false, spat = false, temp = false] : memref<*xf32> to memref<?x?xf32, strided<[?, ?], offset: ?>>
              bufferization.materialize_in_destination %22 in writable %25 : (tensor<?x?xf32>, memref<?x?xf32, strided<[?, ?], offset: ?>>) -> ()
            }
          }
        } {loom.mapped_to = @y}
      } {loom.mapped_to = @x}
      return
    }
  }
  module attributes {loom.pass_name = "EnumerateCopyBroadcast"} {
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
      %15 = loom.expression(%12, %13) {coeffs = [1, 1], logic = "mul"} : index
      %16 = loom.expression(%15, %14) {coeffs = [1, 1], logic = "mul"} : index
      loom.polynomial_constraint(%12, %13, %14, %15, %16) {monomials = [{coeff = -1 : i64, vars = [4]}], upper_bound = -262144 : i64}
      %17 = loom.expression(%12, %13) {coeffs = [1, 1], logic = "add"} : index
      %18 = loom.expression(%17, %14) {coeffs = [1, 1], logic = "mul"} : index
      loom.polynomial_constraint(%12, %14, %13, %17, %18) {monomials = [{coeff = 1 : i64, vars = [4]}], upper_bound = 374784 : i64}
      loom.polynomial_constraint(%12) {monomials = [{coeff = 1 : i64, vars = [0]}], upper_bound = 128 : i64}
      loom.polynomial_constraint(%13) {monomials = [{coeff = 1 : i64, vars = [0]}], upper_bound = 128 : i64}
      %19 = loom.expression(%12, %14) {coeffs = [1, 1], logic = "mul"} : index
      %20 = loom.expression(%13, %14) {coeffs = [1, 1], logic = "mul"} : index
      %21 = loom.expression(%12, %13) {coeffs = [1, 1], logic = "mul"} : index
      %22 = loom.expression(%14, %21) {coeffs = [1, 1], logic = "mul"} : index
      loom.polynomial_constraint(%12, %13, %14, %19, %20, %21, %22) {monomials = [{coeff = 256 : i64, vars = [3]}, {coeff = 256 : i64, vars = [4]}, {coeff = -15 : i64, vars = [6]}], upper_bound = 0 : i64}
    }
    func.func @matmul_kernel__d0i0_d1i1__f01__h_v(%arg0: memref<*xf32> {tt.divisibility = 16 : i32}, %arg1: memref<*xf32> {tt.divisibility = 16 : i32}, %arg2: memref<*xf32> {tt.divisibility = 16 : i32}, %arg3: index, %arg4: index, %arg5: index, %arg6: index, %arg7: index, %arg8: index) {
      affine.parallel (%arg9) = (0) to (8) {
        affine.parallel (%arg10) = (0) to (8) {
          affine.for %arg11 = 0 to affine_map<()[s0, s1] -> (s0 ceildiv 8)>()[%arg3, %arg4] {
            affine.for %arg12 = 0 to affine_map<()[s0, s1] -> (s1 ceildiv 8)>()[%arg3, %arg4] {
              %12 = affine.apply affine_map<(d0, d1) -> (d0 * 8 + d1)>(%arg10, %arg12)
              %13 = affine.apply affine_map<(d0, d1) -> (d0 * 8 + d1)>(%arg9, %arg11)
              %c1024 = arith.constant 1024 : index
              %c1024_0 = arith.constant 1024 : index
              %c1 = arith.constant 1 : index
              %14 = loom.get_symbolic_block_size @constraints::@BM : index
              %15 = loom.get_symbolic_block_size @constraints::@BN : index
              %16 = loom.get_symbolic_block_size @constraints::@BK : index
              %17 = arith.ceildivsi %c1024_0, %16 : index
              %cst = arith.constant 0.000000e+00 : f32
              %c1024_1 = arith.constant 1024 : index
              %c0 = arith.constant 0 : index
              %18 = tensor.empty(%14, %15) : tensor<?x?xf32>
              %19 = linalg.fill ins(%cst : f32) outs(%18 : tensor<?x?xf32>) -> tensor<?x?xf32>
              %20 = arith.muli %13, %14 : index
              %21 = arith.muli %12, %15 : index
              %22 = scf.for %arg13 = %c0 to %17 step %c1 iter_args(%arg14 = %19) -> (tensor<?x?xf32>) {
                %26 = arith.muli %arg13, %16 : index
                %27 = arith.muli %20, %c1024_1 : index
                %28 = arith.addi %27, %26 : index
                %29 = loom.reinterpret_cast %arg0 to offset : [%28], sizes : [%14, %16], strides : [%c1024_0, %c1], reuse : [seq = false, spat = true, temp = true] : memref<*xf32> to memref<?x?xf32, strided<[?, ?], offset: ?>>
                %30 = loom.alloc(%14, %16) on @L1 : memref<?x?xf32>
                loom.copy %29, %30, interconnect : [@horizontal_links], broadcast : [1, 8] : memref<?x?xf32, strided<[?, ?], offset: ?>> to memref<?x?xf32>
                %31 = bufferization.to_tensor %30 restrict writable : memref<?x?xf32> to tensor<?x?xf32>
                %32 = arith.muli %26, %c1024_1 : index
                %33 = arith.addi %32, %21 : index
                %34 = loom.reinterpret_cast %arg1 to offset : [%33], sizes : [%16, %15], strides : [%c1024, %c1], reuse : [seq = false, spat = true, temp = true] : memref<*xf32> to memref<?x?xf32, strided<[?, ?], offset: ?>>
                %35 = loom.alloc(%16, %15) on @L1 : memref<?x?xf32>
                loom.copy %34, %35, interconnect : [@vertical_links], broadcast : [8, 1] : memref<?x?xf32, strided<[?, ?], offset: ?>> to memref<?x?xf32>
                %36 = bufferization.to_tensor %35 restrict writable : memref<?x?xf32> to tensor<?x?xf32>
                %37 = linalg.matmul ins(%31, %36 : tensor<?x?xf32>, tensor<?x?xf32>) outs(%19 : tensor<?x?xf32>) -> tensor<?x?xf32>
                %38 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%arg14, %37 : tensor<?x?xf32>, tensor<?x?xf32>) outs(%arg14 : tensor<?x?xf32>) {
                ^bb0(%in: f32, %in_2: f32, %out: f32):
                  %39 = arith.addf %in, %in_2 : f32
                  linalg.yield %39 : f32
                } -> tensor<?x?xf32>
                scf.yield %38 : tensor<?x?xf32>
              }
              %23 = arith.muli %20, %c1024_1 : index
              %24 = arith.addi %23, %21 : index
              %25 = loom.reinterpret_cast %arg2 to offset : [%24], sizes : [%14, %15], strides : [%c1024, %c1], reuse : [seq = false, spat = false, temp = false] : memref<*xf32> to memref<?x?xf32, strided<[?, ?], offset: ?>>
              bufferization.materialize_in_destination %22 in writable %25 : (tensor<?x?xf32>, memref<?x?xf32, strided<[?, ?], offset: ?>>) -> ()
            }
          }
        } {loom.mapped_to = @y}
      } {loom.mapped_to = @x}
      return
    }
  }
  module attributes {loom.pass_name = "EnumerateCopyBroadcast"} {
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
      %15 = loom.expression(%12, %13) {coeffs = [1, 1], logic = "mul"} : index
      %16 = loom.expression(%15, %14) {coeffs = [1, 1], logic = "mul"} : index
      loom.polynomial_constraint(%12, %13, %14, %15, %16) {monomials = [{coeff = -1 : i64, vars = [4]}], upper_bound = -262144 : i64}
      %17 = loom.expression(%12, %13) {coeffs = [1, 1], logic = "add"} : index
      %18 = loom.expression(%17, %14) {coeffs = [1, 1], logic = "mul"} : index
      loom.polynomial_constraint(%12, %14, %13, %17, %18) {monomials = [{coeff = 1 : i64, vars = [4]}], upper_bound = 374784 : i64}
      loom.polynomial_constraint(%12) {monomials = [{coeff = 1 : i64, vars = [0]}], upper_bound = 128 : i64}
      loom.polynomial_constraint(%13) {monomials = [{coeff = 1 : i64, vars = [0]}], upper_bound = 128 : i64}
      %19 = loom.expression(%13, %14) {coeffs = [1, 1], logic = "mul"} : index
      loom.polynomial_constraint(%12, %14, %13, %19) {monomials = [{coeff = 1024 : i64, vars = [0]}, {coeff = 1 : i64, vars = [3]}], upper_bound = 374784 : i64}
      %20 = loom.expression(%12, %14) {coeffs = [1, 1], logic = "mul"} : index
      %21 = loom.expression(%13, %14) {coeffs = [1, 1], logic = "mul"} : index
      %22 = loom.expression(%12, %13) {coeffs = [1, 1], logic = "mul"} : index
      %23 = loom.expression(%14, %22) {coeffs = [1, 1], logic = "mul"} : index
      loom.polynomial_constraint(%12, %13, %14, %20, %21, %22, %23) {monomials = [{coeff = 64 : i64, vars = [3]}, {coeff = 64 : i64, vars = [4]}, {coeff = -1 : i64, vars = [6]}], upper_bound = 0 : i64}
    }
    func.func @matmul_kernel__d0i0_d1i1__f01__hoist_block_0__d_d(%arg0: memref<*xf32> {tt.divisibility = 16 : i32}, %arg1: memref<*xf32> {tt.divisibility = 16 : i32}, %arg2: memref<*xf32> {tt.divisibility = 16 : i32}, %arg3: index, %arg4: index, %arg5: index, %arg6: index, %arg7: index, %arg8: index) {
      affine.parallel (%arg9) = (0) to (8) {
        affine.parallel (%arg10) = (0) to (8) {
          affine.for %arg11 = 0 to affine_map<()[s0, s1] -> (s0 ceildiv 8)>()[%arg3, %arg4] {
            affine.for %arg12 = 0 to affine_map<()[s0, s1] -> (s1 ceildiv 8)>()[%arg3, %arg4] {
              %12 = affine.apply affine_map<(d0, d1) -> (d0 * 8 + d1)>(%arg10, %arg12)
              %13 = affine.apply affine_map<(d0, d1) -> (d0 * 8 + d1)>(%arg9, %arg11)
              %c1024 = arith.constant 1024 : index
              %c1024_0 = arith.constant 1024 : index
              %c1 = arith.constant 1 : index
              %14 = loom.get_symbolic_block_size @constraints::@BM : index
              %15 = loom.get_symbolic_block_size @constraints::@BN : index
              %16 = loom.get_symbolic_block_size @constraints::@BK : index
              %17 = arith.ceildivsi %c1024_0, %16 : index
              %cst = arith.constant 0.000000e+00 : f32
              %c1024_1 = arith.constant 1024 : index
              %c0 = arith.constant 0 : index
              %18 = tensor.empty(%14, %15) : tensor<?x?xf32>
              %19 = linalg.fill ins(%cst : f32) outs(%18 : tensor<?x?xf32>) -> tensor<?x?xf32>
              %20 = arith.muli %13, %14 : index
              %21 = arith.muli %12, %15 : index
              %22 = arith.muli %14, %c1024_0 : index
              %c0_2 = arith.constant 0 : index
              %23 = arith.muli %20, %c1024_1 : index
              %24 = arith.muli %c0_2, %16 : index
              %25 = arith.addi %23, %24 : index
              %26 = loom.reinterpret_cast %arg0 to offset : [%25], sizes : [%17, %14, %16], strides : [%22, %c1024_0, %c1], reuse : [seq = false, spat = true, temp = true] : memref<*xf32> to memref<?x?x?xf32, strided<[?, ?, ?], offset: ?>>
              %27 = loom.alloc(%17, %14, %16) on @L1 : memref<?x?x?xf32>
              loom.copy %26, %27, interconnect : [], broadcast : [1, 1] : memref<?x?x?xf32, strided<[?, ?, ?], offset: ?>> to memref<?x?x?xf32>
              %28 = scf.for %arg13 = %c0 to %17 step %c1 iter_args(%arg14 = %19) -> (tensor<?x?xf32>) {
                %32 = arith.muli %arg13, %16 : index
                %33 = arith.muli %20, %c1024_1 : index
                %subview = memref.subview %27[%arg13, 0, 0] [1, %14, %16] [1, 1, 1] : memref<?x?x?xf32> to memref<?x?xf32, strided<[?, 1], offset: ?>>
                %34 = bufferization.to_tensor %subview restrict writable : memref<?x?xf32, strided<[?, 1], offset: ?>> to tensor<?x?xf32>
                %35 = arith.muli %32, %c1024_1 : index
                %36 = arith.addi %35, %21 : index
                %37 = loom.reinterpret_cast %arg1 to offset : [%36], sizes : [%16, %15], strides : [%c1024, %c1], reuse : [seq = false, spat = true, temp = true] : memref<*xf32> to memref<?x?xf32, strided<[?, ?], offset: ?>>
                %38 = loom.alloc(%16, %15) on @L1 : memref<?x?xf32>
                loom.copy %37, %38, interconnect : [], broadcast : [1, 1] : memref<?x?xf32, strided<[?, ?], offset: ?>> to memref<?x?xf32>
                %39 = bufferization.to_tensor %38 restrict writable : memref<?x?xf32> to tensor<?x?xf32>
                %40 = linalg.matmul ins(%34, %39 : tensor<?x?xf32>, tensor<?x?xf32>) outs(%19 : tensor<?x?xf32>) -> tensor<?x?xf32>
                %41 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%arg14, %40 : tensor<?x?xf32>, tensor<?x?xf32>) outs(%arg14 : tensor<?x?xf32>) {
                ^bb0(%in: f32, %in_3: f32, %out: f32):
                  %42 = arith.addf %in, %in_3 : f32
                  linalg.yield %42 : f32
                } -> tensor<?x?xf32>
                scf.yield %41 : tensor<?x?xf32>
              }
              %29 = arith.muli %20, %c1024_1 : index
              %30 = arith.addi %29, %21 : index
              %31 = loom.reinterpret_cast %arg2 to offset : [%30], sizes : [%14, %15], strides : [%c1024, %c1], reuse : [seq = false, spat = false, temp = false] : memref<*xf32> to memref<?x?xf32, strided<[?, ?], offset: ?>>
              bufferization.materialize_in_destination %28 in writable %31 : (tensor<?x?xf32>, memref<?x?xf32, strided<[?, ?], offset: ?>>) -> ()
            }
          }
        } {loom.mapped_to = @y}
      } {loom.mapped_to = @x}
      return
    }
  }
  module attributes {loom.pass_name = "EnumerateCopyBroadcast"} {
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
      %15 = loom.expression(%13, %12) {coeffs = [1, 1], logic = "mul"} : index
      %16 = loom.expression(%14, %15) {coeffs = [1, 1], logic = "mul"} : index
      loom.polynomial_constraint(%12, %13, %14, %15, %16) {monomials = [{coeff = -1 : i64, vars = [4]}], upper_bound = -262144 : i64}
      %17 = loom.expression(%12, %13) {coeffs = [1, 1], logic = "add"} : index
      %18 = loom.expression(%14, %17) {coeffs = [1, 1], logic = "mul"} : index
      loom.polynomial_constraint(%12, %14, %13, %17, %18) {monomials = [{coeff = 1 : i64, vars = [4]}], upper_bound = 374784 : i64}
      loom.polynomial_constraint(%12) {monomials = [{coeff = 1 : i64, vars = [0]}], upper_bound = 128 : i64}
      loom.polynomial_constraint(%13) {monomials = [{coeff = 1 : i64, vars = [0]}], upper_bound = 128 : i64}
      %19 = loom.expression(%14, %13) {coeffs = [1, 1], logic = "mul"} : index
      loom.polynomial_constraint(%12, %14, %13, %19) {monomials = [{coeff = 1024 : i64, vars = [0]}, {coeff = 1 : i64, vars = [3]}], upper_bound = 374784 : i64}
      %20 = loom.expression(%14, %12) {coeffs = [1, 1], logic = "mul"} : index
      %21 = loom.expression(%14, %13) {coeffs = [1, 1], logic = "mul"} : index
      %22 = loom.expression(%13, %12) {coeffs = [1, 1], logic = "mul"} : index
      %23 = loom.expression(%14, %22) {coeffs = [1, 1], logic = "mul"} : index
      loom.polynomial_constraint(%12, %13, %14, %20, %21, %22, %23) {monomials = [{coeff = 960 : i64, vars = [3]}, {coeff = 256 : i64, vars = [4]}, {coeff = -15 : i64, vars = [6]}], upper_bound = 0 : i64}
    }
    func.func @matmul_kernel__d0i0_d1i1__f01__hoist_block_0__d_v(%arg0: memref<*xf32> {tt.divisibility = 16 : i32}, %arg1: memref<*xf32> {tt.divisibility = 16 : i32}, %arg2: memref<*xf32> {tt.divisibility = 16 : i32}, %arg3: index, %arg4: index, %arg5: index, %arg6: index, %arg7: index, %arg8: index) {
      affine.parallel (%arg9) = (0) to (8) {
        affine.parallel (%arg10) = (0) to (8) {
          affine.for %arg11 = 0 to affine_map<()[s0, s1] -> (s0 ceildiv 8)>()[%arg3, %arg4] {
            affine.for %arg12 = 0 to affine_map<()[s0, s1] -> (s1 ceildiv 8)>()[%arg3, %arg4] {
              %12 = affine.apply affine_map<(d0, d1) -> (d0 * 8 + d1)>(%arg10, %arg12)
              %13 = affine.apply affine_map<(d0, d1) -> (d0 * 8 + d1)>(%arg9, %arg11)
              %c1024 = arith.constant 1024 : index
              %c1024_0 = arith.constant 1024 : index
              %c1 = arith.constant 1 : index
              %14 = loom.get_symbolic_block_size @constraints::@BM : index
              %15 = loom.get_symbolic_block_size @constraints::@BN : index
              %16 = loom.get_symbolic_block_size @constraints::@BK : index
              %17 = arith.ceildivsi %c1024_0, %16 : index
              %cst = arith.constant 0.000000e+00 : f32
              %c1024_1 = arith.constant 1024 : index
              %c0 = arith.constant 0 : index
              %18 = tensor.empty(%14, %15) : tensor<?x?xf32>
              %19 = linalg.fill ins(%cst : f32) outs(%18 : tensor<?x?xf32>) -> tensor<?x?xf32>
              %20 = arith.muli %13, %14 : index
              %21 = arith.muli %12, %15 : index
              %22 = arith.muli %14, %c1024_0 : index
              %c0_2 = arith.constant 0 : index
              %23 = arith.muli %20, %c1024_1 : index
              %24 = arith.muli %c0_2, %16 : index
              %25 = arith.addi %23, %24 : index
              %26 = loom.reinterpret_cast %arg0 to offset : [%25], sizes : [%17, %14, %16], strides : [%22, %c1024_0, %c1], reuse : [seq = false, spat = true, temp = true] : memref<*xf32> to memref<?x?x?xf32, strided<[?, ?, ?], offset: ?>>
              %27 = loom.alloc(%17, %14, %16) on @L1 : memref<?x?x?xf32>
              loom.copy %26, %27, interconnect : [], broadcast : [1, 1] : memref<?x?x?xf32, strided<[?, ?, ?], offset: ?>> to memref<?x?x?xf32>
              %28 = scf.for %arg13 = %c0 to %17 step %c1 iter_args(%arg14 = %19) -> (tensor<?x?xf32>) {
                %32 = arith.muli %arg13, %16 : index
                %33 = arith.muli %20, %c1024_1 : index
                %subview = memref.subview %27[%arg13, 0, 0] [1, %14, %16] [1, 1, 1] : memref<?x?x?xf32> to memref<?x?xf32, strided<[?, 1], offset: ?>>
                %34 = bufferization.to_tensor %subview restrict writable : memref<?x?xf32, strided<[?, 1], offset: ?>> to tensor<?x?xf32>
                %35 = arith.muli %32, %c1024_1 : index
                %36 = arith.addi %35, %21 : index
                %37 = loom.reinterpret_cast %arg1 to offset : [%36], sizes : [%16, %15], strides : [%c1024, %c1], reuse : [seq = false, spat = true, temp = true] : memref<*xf32> to memref<?x?xf32, strided<[?, ?], offset: ?>>
                %38 = loom.alloc(%16, %15) on @L1 : memref<?x?xf32>
                loom.copy %37, %38, interconnect : [@vertical_links], broadcast : [8, 1] : memref<?x?xf32, strided<[?, ?], offset: ?>> to memref<?x?xf32>
                %39 = bufferization.to_tensor %38 restrict writable : memref<?x?xf32> to tensor<?x?xf32>
                %40 = linalg.matmul ins(%34, %39 : tensor<?x?xf32>, tensor<?x?xf32>) outs(%19 : tensor<?x?xf32>) -> tensor<?x?xf32>
                %41 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%arg14, %40 : tensor<?x?xf32>, tensor<?x?xf32>) outs(%arg14 : tensor<?x?xf32>) {
                ^bb0(%in: f32, %in_3: f32, %out: f32):
                  %42 = arith.addf %in, %in_3 : f32
                  linalg.yield %42 : f32
                } -> tensor<?x?xf32>
                scf.yield %41 : tensor<?x?xf32>
              }
              %29 = arith.muli %20, %c1024_1 : index
              %30 = arith.addi %29, %21 : index
              %31 = loom.reinterpret_cast %arg2 to offset : [%30], sizes : [%14, %15], strides : [%c1024, %c1], reuse : [seq = false, spat = false, temp = false] : memref<*xf32> to memref<?x?xf32, strided<[?, ?], offset: ?>>
              bufferization.materialize_in_destination %28 in writable %31 : (tensor<?x?xf32>, memref<?x?xf32, strided<[?, ?], offset: ?>>) -> ()
            }
          }
        } {loom.mapped_to = @y}
      } {loom.mapped_to = @x}
      return
    }
  }
  module attributes {loom.pass_name = "EnumerateCopyBroadcast"} {
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
      %15 = loom.expression(%13, %12) {coeffs = [1, 1], logic = "mul"} : index
      %16 = loom.expression(%15, %14) {coeffs = [1, 1], logic = "mul"} : index
      loom.polynomial_constraint(%12, %13, %14, %15, %16) {monomials = [{coeff = -1 : i64, vars = [4]}], upper_bound = -262144 : i64}
      %17 = loom.expression(%12, %13) {coeffs = [1, 1], logic = "add"} : index
      %18 = loom.expression(%14, %17) {coeffs = [1, 1], logic = "mul"} : index
      loom.polynomial_constraint(%12, %14, %13, %17, %18) {monomials = [{coeff = 1 : i64, vars = [4]}], upper_bound = 374784 : i64}
      loom.polynomial_constraint(%12) {monomials = [{coeff = 1 : i64, vars = [0]}], upper_bound = 128 : i64}
      loom.polynomial_constraint(%13) {monomials = [{coeff = 1 : i64, vars = [0]}], upper_bound = 128 : i64}
      %19 = loom.expression(%13, %14) {coeffs = [1, 1], logic = "mul"} : index
      loom.polynomial_constraint(%12, %14, %13, %19) {monomials = [{coeff = 1024 : i64, vars = [0]}, {coeff = 1 : i64, vars = [3]}], upper_bound = 374784 : i64}
      %20 = loom.expression(%14, %12) {coeffs = [1, 1], logic = "mul"} : index
      %21 = loom.expression(%13, %14) {coeffs = [1, 1], logic = "mul"} : index
      %22 = loom.expression(%13, %12) {coeffs = [1, 1], logic = "mul"} : index
      %23 = loom.expression(%14, %22) {coeffs = [1, 1], logic = "mul"} : index
      loom.polynomial_constraint(%12, %13, %14, %20, %21, %22, %23) {monomials = [{coeff = 256 : i64, vars = [3]}, {coeff = 960 : i64, vars = [4]}, {coeff = -15 : i64, vars = [6]}], upper_bound = 0 : i64}
    }
    func.func @matmul_kernel__d0i0_d1i1__f01__hoist_block_0__h_d(%arg0: memref<*xf32> {tt.divisibility = 16 : i32}, %arg1: memref<*xf32> {tt.divisibility = 16 : i32}, %arg2: memref<*xf32> {tt.divisibility = 16 : i32}, %arg3: index, %arg4: index, %arg5: index, %arg6: index, %arg7: index, %arg8: index) {
      affine.parallel (%arg9) = (0) to (8) {
        affine.parallel (%arg10) = (0) to (8) {
          affine.for %arg11 = 0 to affine_map<()[s0, s1] -> (s0 ceildiv 8)>()[%arg3, %arg4] {
            affine.for %arg12 = 0 to affine_map<()[s0, s1] -> (s1 ceildiv 8)>()[%arg3, %arg4] {
              %12 = affine.apply affine_map<(d0, d1) -> (d0 * 8 + d1)>(%arg10, %arg12)
              %13 = affine.apply affine_map<(d0, d1) -> (d0 * 8 + d1)>(%arg9, %arg11)
              %c1024 = arith.constant 1024 : index
              %c1024_0 = arith.constant 1024 : index
              %c1 = arith.constant 1 : index
              %14 = loom.get_symbolic_block_size @constraints::@BM : index
              %15 = loom.get_symbolic_block_size @constraints::@BN : index
              %16 = loom.get_symbolic_block_size @constraints::@BK : index
              %17 = arith.ceildivsi %c1024_0, %16 : index
              %cst = arith.constant 0.000000e+00 : f32
              %c1024_1 = arith.constant 1024 : index
              %c0 = arith.constant 0 : index
              %18 = tensor.empty(%14, %15) : tensor<?x?xf32>
              %19 = linalg.fill ins(%cst : f32) outs(%18 : tensor<?x?xf32>) -> tensor<?x?xf32>
              %20 = arith.muli %13, %14 : index
              %21 = arith.muli %12, %15 : index
              %22 = arith.muli %14, %c1024_0 : index
              %c0_2 = arith.constant 0 : index
              %23 = arith.muli %20, %c1024_1 : index
              %24 = arith.muli %c0_2, %16 : index
              %25 = arith.addi %23, %24 : index
              %26 = loom.reinterpret_cast %arg0 to offset : [%25], sizes : [%17, %14, %16], strides : [%22, %c1024_0, %c1], reuse : [seq = false, spat = true, temp = true] : memref<*xf32> to memref<?x?x?xf32, strided<[?, ?, ?], offset: ?>>
              %27 = loom.alloc(%17, %14, %16) on @L1 : memref<?x?x?xf32>
              loom.copy %26, %27, interconnect : [@horizontal_links], broadcast : [1, 8] : memref<?x?x?xf32, strided<[?, ?, ?], offset: ?>> to memref<?x?x?xf32>
              %28 = scf.for %arg13 = %c0 to %17 step %c1 iter_args(%arg14 = %19) -> (tensor<?x?xf32>) {
                %32 = arith.muli %arg13, %16 : index
                %33 = arith.muli %20, %c1024_1 : index
                %subview = memref.subview %27[%arg13, 0, 0] [1, %14, %16] [1, 1, 1] : memref<?x?x?xf32> to memref<?x?xf32, strided<[?, 1], offset: ?>>
                %34 = bufferization.to_tensor %subview restrict writable : memref<?x?xf32, strided<[?, 1], offset: ?>> to tensor<?x?xf32>
                %35 = arith.muli %32, %c1024_1 : index
                %36 = arith.addi %35, %21 : index
                %37 = loom.reinterpret_cast %arg1 to offset : [%36], sizes : [%16, %15], strides : [%c1024, %c1], reuse : [seq = false, spat = true, temp = true] : memref<*xf32> to memref<?x?xf32, strided<[?, ?], offset: ?>>
                %38 = loom.alloc(%16, %15) on @L1 : memref<?x?xf32>
                loom.copy %37, %38, interconnect : [], broadcast : [1, 1] : memref<?x?xf32, strided<[?, ?], offset: ?>> to memref<?x?xf32>
                %39 = bufferization.to_tensor %38 restrict writable : memref<?x?xf32> to tensor<?x?xf32>
                %40 = linalg.matmul ins(%34, %39 : tensor<?x?xf32>, tensor<?x?xf32>) outs(%19 : tensor<?x?xf32>) -> tensor<?x?xf32>
                %41 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%arg14, %40 : tensor<?x?xf32>, tensor<?x?xf32>) outs(%arg14 : tensor<?x?xf32>) {
                ^bb0(%in: f32, %in_3: f32, %out: f32):
                  %42 = arith.addf %in, %in_3 : f32
                  linalg.yield %42 : f32
                } -> tensor<?x?xf32>
                scf.yield %41 : tensor<?x?xf32>
              }
              %29 = arith.muli %20, %c1024_1 : index
              %30 = arith.addi %29, %21 : index
              %31 = loom.reinterpret_cast %arg2 to offset : [%30], sizes : [%14, %15], strides : [%c1024, %c1], reuse : [seq = false, spat = false, temp = false] : memref<*xf32> to memref<?x?xf32, strided<[?, ?], offset: ?>>
              bufferization.materialize_in_destination %28 in writable %31 : (tensor<?x?xf32>, memref<?x?xf32, strided<[?, ?], offset: ?>>) -> ()
            }
          }
        } {loom.mapped_to = @y}
      } {loom.mapped_to = @x}
      return
    }
  }
  module attributes {loom.pass_name = "EnumerateCopyBroadcast"} {
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
      %15 = loom.expression(%13, %12) {coeffs = [1, 1], logic = "mul"} : index
      %16 = loom.expression(%15, %14) {coeffs = [1, 1], logic = "mul"} : index
      loom.polynomial_constraint(%12, %13, %14, %15, %16) {monomials = [{coeff = -1 : i64, vars = [4]}], upper_bound = -262144 : i64}
      %17 = loom.expression(%12, %13) {coeffs = [1, 1], logic = "add"} : index
      %18 = loom.expression(%14, %17) {coeffs = [1, 1], logic = "mul"} : index
      loom.polynomial_constraint(%12, %14, %13, %17, %18) {monomials = [{coeff = 1 : i64, vars = [4]}], upper_bound = 374784 : i64}
      loom.polynomial_constraint(%12) {monomials = [{coeff = 1 : i64, vars = [0]}], upper_bound = 128 : i64}
      loom.polynomial_constraint(%13) {monomials = [{coeff = 1 : i64, vars = [0]}], upper_bound = 128 : i64}
      %19 = loom.expression(%13, %14) {coeffs = [1, 1], logic = "mul"} : index
      loom.polynomial_constraint(%12, %14, %13, %19) {monomials = [{coeff = 1024 : i64, vars = [0]}, {coeff = 1 : i64, vars = [3]}], upper_bound = 374784 : i64}
      %20 = loom.expression(%14, %12) {coeffs = [1, 1], logic = "mul"} : index
      %21 = loom.expression(%13, %14) {coeffs = [1, 1], logic = "mul"} : index
      %22 = loom.expression(%13, %12) {coeffs = [1, 1], logic = "mul"} : index
      %23 = loom.expression(%14, %22) {coeffs = [1, 1], logic = "mul"} : index
      loom.polynomial_constraint(%12, %13, %14, %20, %21, %22, %23) {monomials = [{coeff = 256 : i64, vars = [3]}, {coeff = 256 : i64, vars = [4]}, {coeff = -15 : i64, vars = [6]}], upper_bound = 0 : i64}
    }
    func.func @matmul_kernel__d0i0_d1i1__f01__hoist_block_0__h_v(%arg0: memref<*xf32> {tt.divisibility = 16 : i32}, %arg1: memref<*xf32> {tt.divisibility = 16 : i32}, %arg2: memref<*xf32> {tt.divisibility = 16 : i32}, %arg3: index, %arg4: index, %arg5: index, %arg6: index, %arg7: index, %arg8: index) {
      affine.parallel (%arg9) = (0) to (8) {
        affine.parallel (%arg10) = (0) to (8) {
          affine.for %arg11 = 0 to affine_map<()[s0, s1] -> (s0 ceildiv 8)>()[%arg3, %arg4] {
            affine.for %arg12 = 0 to affine_map<()[s0, s1] -> (s1 ceildiv 8)>()[%arg3, %arg4] {
              %12 = affine.apply affine_map<(d0, d1) -> (d0 * 8 + d1)>(%arg10, %arg12)
              %13 = affine.apply affine_map<(d0, d1) -> (d0 * 8 + d1)>(%arg9, %arg11)
              %c1024 = arith.constant 1024 : index
              %c1024_0 = arith.constant 1024 : index
              %c1 = arith.constant 1 : index
              %14 = loom.get_symbolic_block_size @constraints::@BM : index
              %15 = loom.get_symbolic_block_size @constraints::@BN : index
              %16 = loom.get_symbolic_block_size @constraints::@BK : index
              %17 = arith.ceildivsi %c1024_0, %16 : index
              %cst = arith.constant 0.000000e+00 : f32
              %c1024_1 = arith.constant 1024 : index
              %c0 = arith.constant 0 : index
              %18 = tensor.empty(%14, %15) : tensor<?x?xf32>
              %19 = linalg.fill ins(%cst : f32) outs(%18 : tensor<?x?xf32>) -> tensor<?x?xf32>
              %20 = arith.muli %13, %14 : index
              %21 = arith.muli %12, %15 : index
              %22 = arith.muli %14, %c1024_0 : index
              %c0_2 = arith.constant 0 : index
              %23 = arith.muli %20, %c1024_1 : index
              %24 = arith.muli %c0_2, %16 : index
              %25 = arith.addi %23, %24 : index
              %26 = loom.reinterpret_cast %arg0 to offset : [%25], sizes : [%17, %14, %16], strides : [%22, %c1024_0, %c1], reuse : [seq = false, spat = true, temp = true] : memref<*xf32> to memref<?x?x?xf32, strided<[?, ?, ?], offset: ?>>
              %27 = loom.alloc(%17, %14, %16) on @L1 : memref<?x?x?xf32>
              loom.copy %26, %27, interconnect : [@horizontal_links], broadcast : [1, 8] : memref<?x?x?xf32, strided<[?, ?, ?], offset: ?>> to memref<?x?x?xf32>
              %28 = scf.for %arg13 = %c0 to %17 step %c1 iter_args(%arg14 = %19) -> (tensor<?x?xf32>) {
                %32 = arith.muli %arg13, %16 : index
                %33 = arith.muli %20, %c1024_1 : index
                %subview = memref.subview %27[%arg13, 0, 0] [1, %14, %16] [1, 1, 1] : memref<?x?x?xf32> to memref<?x?xf32, strided<[?, 1], offset: ?>>
                %34 = bufferization.to_tensor %subview restrict writable : memref<?x?xf32, strided<[?, 1], offset: ?>> to tensor<?x?xf32>
                %35 = arith.muli %32, %c1024_1 : index
                %36 = arith.addi %35, %21 : index
                %37 = loom.reinterpret_cast %arg1 to offset : [%36], sizes : [%16, %15], strides : [%c1024, %c1], reuse : [seq = false, spat = true, temp = true] : memref<*xf32> to memref<?x?xf32, strided<[?, ?], offset: ?>>
                %38 = loom.alloc(%16, %15) on @L1 : memref<?x?xf32>
                loom.copy %37, %38, interconnect : [@vertical_links], broadcast : [8, 1] : memref<?x?xf32, strided<[?, ?], offset: ?>> to memref<?x?xf32>
                %39 = bufferization.to_tensor %38 restrict writable : memref<?x?xf32> to tensor<?x?xf32>
                %40 = linalg.matmul ins(%34, %39 : tensor<?x?xf32>, tensor<?x?xf32>) outs(%19 : tensor<?x?xf32>) -> tensor<?x?xf32>
                %41 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%arg14, %40 : tensor<?x?xf32>, tensor<?x?xf32>) outs(%arg14 : tensor<?x?xf32>) {
                ^bb0(%in: f32, %in_3: f32, %out: f32):
                  %42 = arith.addf %in, %in_3 : f32
                  linalg.yield %42 : f32
                } -> tensor<?x?xf32>
                scf.yield %41 : tensor<?x?xf32>
              }
              %29 = arith.muli %20, %c1024_1 : index
              %30 = arith.addi %29, %21 : index
              %31 = loom.reinterpret_cast %arg2 to offset : [%30], sizes : [%14, %15], strides : [%c1024, %c1], reuse : [seq = false, spat = false, temp = false] : memref<*xf32> to memref<?x?xf32, strided<[?, ?], offset: ?>>
              bufferization.materialize_in_destination %28 in writable %31 : (tensor<?x?xf32>, memref<?x?xf32, strided<[?, ?], offset: ?>>) -> ()
            }
          }
        } {loom.mapped_to = @y}
      } {loom.mapped_to = @x}
      return
    }
  }
  module attributes {loom.pass_name = "EnumerateCopyBroadcast"} {
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
      %15 = loom.expression(%13, %12) {coeffs = [1, 1], logic = "mul"} : index
      %16 = loom.expression(%15, %14) {coeffs = [1, 1], logic = "mul"} : index
      loom.polynomial_constraint(%12, %13, %14, %15, %16) {monomials = [{coeff = -1 : i64, vars = [4]}], upper_bound = -262144 : i64}
      %17 = loom.expression(%12, %13) {coeffs = [1, 1], logic = "add"} : index
      %18 = loom.expression(%14, %17) {coeffs = [1, 1], logic = "mul"} : index
      loom.polynomial_constraint(%12, %14, %13, %17, %18) {monomials = [{coeff = 1 : i64, vars = [4]}], upper_bound = 374784 : i64}
      loom.polynomial_constraint(%12) {monomials = [{coeff = 1 : i64, vars = [0]}], upper_bound = 128 : i64}
      loom.polynomial_constraint(%13) {monomials = [{coeff = 1 : i64, vars = [0]}], upper_bound = 128 : i64}
      %19 = loom.expression(%14, %12) {coeffs = [1, 1], logic = "mul"} : index
      loom.polynomial_constraint(%13, %12, %14, %19) {monomials = [{coeff = 1024 : i64, vars = [0]}, {coeff = 1 : i64, vars = [3]}], upper_bound = 374784 : i64}
      %20 = loom.expression(%14, %12) {coeffs = [1, 1], logic = "mul"} : index
      %21 = loom.expression(%13, %14) {coeffs = [1, 1], logic = "mul"} : index
      %22 = loom.expression(%13, %12) {coeffs = [1, 1], logic = "mul"} : index
      %23 = loom.expression(%14, %22) {coeffs = [1, 1], logic = "mul"} : index
      loom.polynomial_constraint(%12, %13, %14, %20, %21, %22, %23) {monomials = [{coeff = 64 : i64, vars = [3]}, {coeff = 64 : i64, vars = [4]}, {coeff = -1 : i64, vars = [6]}], upper_bound = 0 : i64}
    }
    func.func @matmul_kernel__d0i0_d1i1__f01__hoist_block_1__d_d(%arg0: memref<*xf32> {tt.divisibility = 16 : i32}, %arg1: memref<*xf32> {tt.divisibility = 16 : i32}, %arg2: memref<*xf32> {tt.divisibility = 16 : i32}, %arg3: index, %arg4: index, %arg5: index, %arg6: index, %arg7: index, %arg8: index) {
      affine.parallel (%arg9) = (0) to (8) {
        affine.parallel (%arg10) = (0) to (8) {
          affine.for %arg11 = 0 to affine_map<()[s0, s1] -> (s0 ceildiv 8)>()[%arg3, %arg4] {
            affine.for %arg12 = 0 to affine_map<()[s0, s1] -> (s1 ceildiv 8)>()[%arg3, %arg4] {
              %12 = affine.apply affine_map<(d0, d1) -> (d0 * 8 + d1)>(%arg10, %arg12)
              %13 = affine.apply affine_map<(d0, d1) -> (d0 * 8 + d1)>(%arg9, %arg11)
              %c1024 = arith.constant 1024 : index
              %c1024_0 = arith.constant 1024 : index
              %c1 = arith.constant 1 : index
              %14 = loom.get_symbolic_block_size @constraints::@BM : index
              %15 = loom.get_symbolic_block_size @constraints::@BN : index
              %16 = loom.get_symbolic_block_size @constraints::@BK : index
              %17 = arith.ceildivsi %c1024_0, %16 : index
              %cst = arith.constant 0.000000e+00 : f32
              %c1024_1 = arith.constant 1024 : index
              %c0 = arith.constant 0 : index
              %18 = tensor.empty(%14, %15) : tensor<?x?xf32>
              %19 = linalg.fill ins(%cst : f32) outs(%18 : tensor<?x?xf32>) -> tensor<?x?xf32>
              %20 = arith.muli %13, %14 : index
              %21 = arith.muli %12, %15 : index
              %22 = arith.muli %16, %c1024 : index
              %c0_2 = arith.constant 0 : index
              %23 = arith.muli %c0_2, %16 : index
              %24 = arith.muli %23, %c1024_1 : index
              %25 = arith.addi %24, %21 : index
              %26 = loom.reinterpret_cast %arg1 to offset : [%25], sizes : [%17, %16, %15], strides : [%22, %c1024, %c1], reuse : [seq = false, spat = true, temp = true] : memref<*xf32> to memref<?x?x?xf32, strided<[?, ?, ?], offset: ?>>
              %27 = loom.alloc(%17, %16, %15) on @L1 : memref<?x?x?xf32>
              loom.copy %26, %27, interconnect : [], broadcast : [1, 1] : memref<?x?x?xf32, strided<[?, ?, ?], offset: ?>> to memref<?x?x?xf32>
              %28 = scf.for %arg13 = %c0 to %17 step %c1 iter_args(%arg14 = %19) -> (tensor<?x?xf32>) {
                %32 = arith.muli %arg13, %16 : index
                %33 = arith.muli %20, %c1024_1 : index
                %34 = arith.addi %33, %32 : index
                %35 = loom.reinterpret_cast %arg0 to offset : [%34], sizes : [%14, %16], strides : [%c1024_0, %c1], reuse : [seq = false, spat = true, temp = true] : memref<*xf32> to memref<?x?xf32, strided<[?, ?], offset: ?>>
                %36 = loom.alloc(%14, %16) on @L1 : memref<?x?xf32>
                loom.copy %35, %36, interconnect : [], broadcast : [1, 1] : memref<?x?xf32, strided<[?, ?], offset: ?>> to memref<?x?xf32>
                %37 = bufferization.to_tensor %36 restrict writable : memref<?x?xf32> to tensor<?x?xf32>
                %38 = arith.muli %32, %c1024_1 : index
                %subview = memref.subview %27[%arg13, 0, 0] [1, %16, %15] [1, 1, 1] : memref<?x?x?xf32> to memref<?x?xf32, strided<[?, 1], offset: ?>>
                %39 = bufferization.to_tensor %subview restrict writable : memref<?x?xf32, strided<[?, 1], offset: ?>> to tensor<?x?xf32>
                %40 = linalg.matmul ins(%37, %39 : tensor<?x?xf32>, tensor<?x?xf32>) outs(%19 : tensor<?x?xf32>) -> tensor<?x?xf32>
                %41 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%arg14, %40 : tensor<?x?xf32>, tensor<?x?xf32>) outs(%arg14 : tensor<?x?xf32>) {
                ^bb0(%in: f32, %in_3: f32, %out: f32):
                  %42 = arith.addf %in, %in_3 : f32
                  linalg.yield %42 : f32
                } -> tensor<?x?xf32>
                scf.yield %41 : tensor<?x?xf32>
              }
              %29 = arith.muli %20, %c1024_1 : index
              %30 = arith.addi %29, %21 : index
              %31 = loom.reinterpret_cast %arg2 to offset : [%30], sizes : [%14, %15], strides : [%c1024, %c1], reuse : [seq = false, spat = false, temp = false] : memref<*xf32> to memref<?x?xf32, strided<[?, ?], offset: ?>>
              bufferization.materialize_in_destination %28 in writable %31 : (tensor<?x?xf32>, memref<?x?xf32, strided<[?, ?], offset: ?>>) -> ()
            }
          }
        } {loom.mapped_to = @y}
      } {loom.mapped_to = @x}
      return
    }
  }
  module attributes {loom.pass_name = "EnumerateCopyBroadcast"} {
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
      %15 = loom.expression(%13, %12) {coeffs = [1, 1], logic = "mul"} : index
      %16 = loom.expression(%15, %14) {coeffs = [1, 1], logic = "mul"} : index
      loom.polynomial_constraint(%12, %13, %14, %15, %16) {monomials = [{coeff = -1 : i64, vars = [4]}], upper_bound = -262144 : i64}
      %17 = loom.expression(%12, %13) {coeffs = [1, 1], logic = "add"} : index
      %18 = loom.expression(%14, %17) {coeffs = [1, 1], logic = "mul"} : index
      loom.polynomial_constraint(%12, %14, %13, %17, %18) {monomials = [{coeff = 1 : i64, vars = [4]}], upper_bound = 374784 : i64}
      loom.polynomial_constraint(%12) {monomials = [{coeff = 1 : i64, vars = [0]}], upper_bound = 128 : i64}
      loom.polynomial_constraint(%13) {monomials = [{coeff = 1 : i64, vars = [0]}], upper_bound = 128 : i64}
      %19 = loom.expression(%14, %12) {coeffs = [1, 1], logic = "mul"} : index
      loom.polynomial_constraint(%13, %12, %14, %19) {monomials = [{coeff = 1024 : i64, vars = [0]}, {coeff = 1 : i64, vars = [3]}], upper_bound = 374784 : i64}
      %20 = loom.expression(%14, %12) {coeffs = [1, 1], logic = "mul"} : index
      %21 = loom.expression(%13, %14) {coeffs = [1, 1], logic = "mul"} : index
      %22 = loom.expression(%13, %12) {coeffs = [1, 1], logic = "mul"} : index
      %23 = loom.expression(%14, %22) {coeffs = [1, 1], logic = "mul"} : index
      loom.polynomial_constraint(%12, %13, %14, %20, %21, %22, %23) {monomials = [{coeff = 960 : i64, vars = [3]}, {coeff = 256 : i64, vars = [4]}, {coeff = -15 : i64, vars = [6]}], upper_bound = 0 : i64}
    }
    func.func @matmul_kernel__d0i0_d1i1__f01__hoist_block_1__d_h(%arg0: memref<*xf32> {tt.divisibility = 16 : i32}, %arg1: memref<*xf32> {tt.divisibility = 16 : i32}, %arg2: memref<*xf32> {tt.divisibility = 16 : i32}, %arg3: index, %arg4: index, %arg5: index, %arg6: index, %arg7: index, %arg8: index) {
      affine.parallel (%arg9) = (0) to (8) {
        affine.parallel (%arg10) = (0) to (8) {
          affine.for %arg11 = 0 to affine_map<()[s0, s1] -> (s0 ceildiv 8)>()[%arg3, %arg4] {
            affine.for %arg12 = 0 to affine_map<()[s0, s1] -> (s1 ceildiv 8)>()[%arg3, %arg4] {
              %12 = affine.apply affine_map<(d0, d1) -> (d0 * 8 + d1)>(%arg10, %arg12)
              %13 = affine.apply affine_map<(d0, d1) -> (d0 * 8 + d1)>(%arg9, %arg11)
              %c1024 = arith.constant 1024 : index
              %c1024_0 = arith.constant 1024 : index
              %c1 = arith.constant 1 : index
              %14 = loom.get_symbolic_block_size @constraints::@BM : index
              %15 = loom.get_symbolic_block_size @constraints::@BN : index
              %16 = loom.get_symbolic_block_size @constraints::@BK : index
              %17 = arith.ceildivsi %c1024_0, %16 : index
              %cst = arith.constant 0.000000e+00 : f32
              %c1024_1 = arith.constant 1024 : index
              %c0 = arith.constant 0 : index
              %18 = tensor.empty(%14, %15) : tensor<?x?xf32>
              %19 = linalg.fill ins(%cst : f32) outs(%18 : tensor<?x?xf32>) -> tensor<?x?xf32>
              %20 = arith.muli %13, %14 : index
              %21 = arith.muli %12, %15 : index
              %22 = arith.muli %16, %c1024 : index
              %c0_2 = arith.constant 0 : index
              %23 = arith.muli %c0_2, %16 : index
              %24 = arith.muli %23, %c1024_1 : index
              %25 = arith.addi %24, %21 : index
              %26 = loom.reinterpret_cast %arg1 to offset : [%25], sizes : [%17, %16, %15], strides : [%22, %c1024, %c1], reuse : [seq = false, spat = true, temp = true] : memref<*xf32> to memref<?x?x?xf32, strided<[?, ?, ?], offset: ?>>
              %27 = loom.alloc(%17, %16, %15) on @L1 : memref<?x?x?xf32>
              loom.copy %26, %27, interconnect : [], broadcast : [1, 1] : memref<?x?x?xf32, strided<[?, ?, ?], offset: ?>> to memref<?x?x?xf32>
              %28 = scf.for %arg13 = %c0 to %17 step %c1 iter_args(%arg14 = %19) -> (tensor<?x?xf32>) {
                %32 = arith.muli %arg13, %16 : index
                %33 = arith.muli %20, %c1024_1 : index
                %34 = arith.addi %33, %32 : index
                %35 = loom.reinterpret_cast %arg0 to offset : [%34], sizes : [%14, %16], strides : [%c1024_0, %c1], reuse : [seq = false, spat = true, temp = true] : memref<*xf32> to memref<?x?xf32, strided<[?, ?], offset: ?>>
                %36 = loom.alloc(%14, %16) on @L1 : memref<?x?xf32>
                loom.copy %35, %36, interconnect : [@horizontal_links], broadcast : [1, 8] : memref<?x?xf32, strided<[?, ?], offset: ?>> to memref<?x?xf32>
                %37 = bufferization.to_tensor %36 restrict writable : memref<?x?xf32> to tensor<?x?xf32>
                %38 = arith.muli %32, %c1024_1 : index
                %subview = memref.subview %27[%arg13, 0, 0] [1, %16, %15] [1, 1, 1] : memref<?x?x?xf32> to memref<?x?xf32, strided<[?, 1], offset: ?>>
                %39 = bufferization.to_tensor %subview restrict writable : memref<?x?xf32, strided<[?, 1], offset: ?>> to tensor<?x?xf32>
                %40 = linalg.matmul ins(%37, %39 : tensor<?x?xf32>, tensor<?x?xf32>) outs(%19 : tensor<?x?xf32>) -> tensor<?x?xf32>
                %41 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%arg14, %40 : tensor<?x?xf32>, tensor<?x?xf32>) outs(%arg14 : tensor<?x?xf32>) {
                ^bb0(%in: f32, %in_3: f32, %out: f32):
                  %42 = arith.addf %in, %in_3 : f32
                  linalg.yield %42 : f32
                } -> tensor<?x?xf32>
                scf.yield %41 : tensor<?x?xf32>
              }
              %29 = arith.muli %20, %c1024_1 : index
              %30 = arith.addi %29, %21 : index
              %31 = loom.reinterpret_cast %arg2 to offset : [%30], sizes : [%14, %15], strides : [%c1024, %c1], reuse : [seq = false, spat = false, temp = false] : memref<*xf32> to memref<?x?xf32, strided<[?, ?], offset: ?>>
              bufferization.materialize_in_destination %28 in writable %31 : (tensor<?x?xf32>, memref<?x?xf32, strided<[?, ?], offset: ?>>) -> ()
            }
          }
        } {loom.mapped_to = @y}
      } {loom.mapped_to = @x}
      return
    }
  }
  module attributes {loom.pass_name = "EnumerateCopyBroadcast"} {
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
      %15 = loom.expression(%12, %13) {coeffs = [1, 1], logic = "mul"} : index
      %16 = loom.expression(%15, %14) {coeffs = [1, 1], logic = "mul"} : index
      loom.polynomial_constraint(%12, %13, %14, %15, %16) {monomials = [{coeff = -1 : i64, vars = [4]}], upper_bound = -262144 : i64}
      %17 = loom.expression(%12, %13) {coeffs = [1, 1], logic = "add"} : index
      %18 = loom.expression(%17, %14) {coeffs = [1, 1], logic = "mul"} : index
      loom.polynomial_constraint(%12, %14, %13, %17, %18) {monomials = [{coeff = 1 : i64, vars = [4]}], upper_bound = 374784 : i64}
      loom.polynomial_constraint(%12) {monomials = [{coeff = 1 : i64, vars = [0]}], upper_bound = 128 : i64}
      loom.polynomial_constraint(%13) {monomials = [{coeff = 1 : i64, vars = [0]}], upper_bound = 128 : i64}
      %19 = loom.expression(%12, %14) {coeffs = [1, 1], logic = "mul"} : index
      loom.polynomial_constraint(%13, %12, %14, %19) {monomials = [{coeff = 1024 : i64, vars = [0]}, {coeff = 1 : i64, vars = [3]}], upper_bound = 374784 : i64}
      %20 = loom.expression(%12, %14) {coeffs = [1, 1], logic = "mul"} : index
      %21 = loom.expression(%13, %14) {coeffs = [1, 1], logic = "mul"} : index
      %22 = loom.expression(%12, %13) {coeffs = [1, 1], logic = "mul"} : index
      %23 = loom.expression(%14, %22) {coeffs = [1, 1], logic = "mul"} : index
      loom.polynomial_constraint(%12, %13, %14, %20, %21, %22, %23) {monomials = [{coeff = 256 : i64, vars = [3]}, {coeff = 960 : i64, vars = [4]}, {coeff = -15 : i64, vars = [6]}], upper_bound = 0 : i64}
    }
    func.func @matmul_kernel__d0i0_d1i1__f01__hoist_block_1__v_d(%arg0: memref<*xf32> {tt.divisibility = 16 : i32}, %arg1: memref<*xf32> {tt.divisibility = 16 : i32}, %arg2: memref<*xf32> {tt.divisibility = 16 : i32}, %arg3: index, %arg4: index, %arg5: index, %arg6: index, %arg7: index, %arg8: index) {
      affine.parallel (%arg9) = (0) to (8) {
        affine.parallel (%arg10) = (0) to (8) {
          affine.for %arg11 = 0 to affine_map<()[s0, s1] -> (s0 ceildiv 8)>()[%arg3, %arg4] {
            affine.for %arg12 = 0 to affine_map<()[s0, s1] -> (s1 ceildiv 8)>()[%arg3, %arg4] {
              %12 = affine.apply affine_map<(d0, d1) -> (d0 * 8 + d1)>(%arg10, %arg12)
              %13 = affine.apply affine_map<(d0, d1) -> (d0 * 8 + d1)>(%arg9, %arg11)
              %c1024 = arith.constant 1024 : index
              %c1024_0 = arith.constant 1024 : index
              %c1 = arith.constant 1 : index
              %14 = loom.get_symbolic_block_size @constraints::@BM : index
              %15 = loom.get_symbolic_block_size @constraints::@BN : index
              %16 = loom.get_symbolic_block_size @constraints::@BK : index
              %17 = arith.ceildivsi %c1024_0, %16 : index
              %cst = arith.constant 0.000000e+00 : f32
              %c1024_1 = arith.constant 1024 : index
              %c0 = arith.constant 0 : index
              %18 = tensor.empty(%14, %15) : tensor<?x?xf32>
              %19 = linalg.fill ins(%cst : f32) outs(%18 : tensor<?x?xf32>) -> tensor<?x?xf32>
              %20 = arith.muli %13, %14 : index
              %21 = arith.muli %12, %15 : index
              %22 = arith.muli %16, %c1024 : index
              %c0_2 = arith.constant 0 : index
              %23 = arith.muli %c0_2, %16 : index
              %24 = arith.muli %23, %c1024_1 : index
              %25 = arith.addi %24, %21 : index
              %26 = loom.reinterpret_cast %arg1 to offset : [%25], sizes : [%17, %16, %15], strides : [%22, %c1024, %c1], reuse : [seq = false, spat = true, temp = true] : memref<*xf32> to memref<?x?x?xf32, strided<[?, ?, ?], offset: ?>>
              %27 = loom.alloc(%17, %16, %15) on @L1 : memref<?x?x?xf32>
              loom.copy %26, %27, interconnect : [@vertical_links], broadcast : [8, 1] : memref<?x?x?xf32, strided<[?, ?, ?], offset: ?>> to memref<?x?x?xf32>
              %28 = scf.for %arg13 = %c0 to %17 step %c1 iter_args(%arg14 = %19) -> (tensor<?x?xf32>) {
                %32 = arith.muli %arg13, %16 : index
                %33 = arith.muli %20, %c1024_1 : index
                %34 = arith.addi %33, %32 : index
                %35 = loom.reinterpret_cast %arg0 to offset : [%34], sizes : [%14, %16], strides : [%c1024_0, %c1], reuse : [seq = false, spat = true, temp = true] : memref<*xf32> to memref<?x?xf32, strided<[?, ?], offset: ?>>
                %36 = loom.alloc(%14, %16) on @L1 : memref<?x?xf32>
                loom.copy %35, %36, interconnect : [], broadcast : [1, 1] : memref<?x?xf32, strided<[?, ?], offset: ?>> to memref<?x?xf32>
                %37 = bufferization.to_tensor %36 restrict writable : memref<?x?xf32> to tensor<?x?xf32>
                %38 = arith.muli %32, %c1024_1 : index
                %subview = memref.subview %27[%arg13, 0, 0] [1, %16, %15] [1, 1, 1] : memref<?x?x?xf32> to memref<?x?xf32, strided<[?, 1], offset: ?>>
                %39 = bufferization.to_tensor %subview restrict writable : memref<?x?xf32, strided<[?, 1], offset: ?>> to tensor<?x?xf32>
                %40 = linalg.matmul ins(%37, %39 : tensor<?x?xf32>, tensor<?x?xf32>) outs(%19 : tensor<?x?xf32>) -> tensor<?x?xf32>
                %41 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%arg14, %40 : tensor<?x?xf32>, tensor<?x?xf32>) outs(%arg14 : tensor<?x?xf32>) {
                ^bb0(%in: f32, %in_3: f32, %out: f32):
                  %42 = arith.addf %in, %in_3 : f32
                  linalg.yield %42 : f32
                } -> tensor<?x?xf32>
                scf.yield %41 : tensor<?x?xf32>
              }
              %29 = arith.muli %20, %c1024_1 : index
              %30 = arith.addi %29, %21 : index
              %31 = loom.reinterpret_cast %arg2 to offset : [%30], sizes : [%14, %15], strides : [%c1024, %c1], reuse : [seq = false, spat = false, temp = false] : memref<*xf32> to memref<?x?xf32, strided<[?, ?], offset: ?>>
              bufferization.materialize_in_destination %28 in writable %31 : (tensor<?x?xf32>, memref<?x?xf32, strided<[?, ?], offset: ?>>) -> ()
            }
          }
        } {loom.mapped_to = @y}
      } {loom.mapped_to = @x}
      return
    }
  }
  module attributes {loom.pass_name = "EnumerateCopyBroadcast"} {
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
      %15 = loom.expression(%12, %13) {coeffs = [1, 1], logic = "mul"} : index
      %16 = loom.expression(%15, %14) {coeffs = [1, 1], logic = "mul"} : index
      loom.polynomial_constraint(%12, %13, %14, %15, %16) {monomials = [{coeff = -1 : i64, vars = [4]}], upper_bound = -262144 : i64}
      %17 = loom.expression(%12, %13) {coeffs = [1, 1], logic = "add"} : index
      %18 = loom.expression(%17, %14) {coeffs = [1, 1], logic = "mul"} : index
      loom.polynomial_constraint(%12, %14, %13, %17, %18) {monomials = [{coeff = 1 : i64, vars = [4]}], upper_bound = 374784 : i64}
      loom.polynomial_constraint(%12) {monomials = [{coeff = 1 : i64, vars = [0]}], upper_bound = 128 : i64}
      loom.polynomial_constraint(%13) {monomials = [{coeff = 1 : i64, vars = [0]}], upper_bound = 128 : i64}
      %19 = loom.expression(%12, %14) {coeffs = [1, 1], logic = "mul"} : index
      loom.polynomial_constraint(%13, %12, %14, %19) {monomials = [{coeff = 1024 : i64, vars = [0]}, {coeff = 1 : i64, vars = [3]}], upper_bound = 374784 : i64}
      %20 = loom.expression(%12, %14) {coeffs = [1, 1], logic = "mul"} : index
      %21 = loom.expression(%13, %14) {coeffs = [1, 1], logic = "mul"} : index
      %22 = loom.expression(%12, %13) {coeffs = [1, 1], logic = "mul"} : index
      %23 = loom.expression(%14, %22) {coeffs = [1, 1], logic = "mul"} : index
      loom.polynomial_constraint(%12, %13, %14, %20, %21, %22, %23) {monomials = [{coeff = 256 : i64, vars = [3]}, {coeff = 256 : i64, vars = [4]}, {coeff = -15 : i64, vars = [6]}], upper_bound = 0 : i64}
    }
    func.func @matmul_kernel__d0i0_d1i1__f01__hoist_block_1__v_h(%arg0: memref<*xf32> {tt.divisibility = 16 : i32}, %arg1: memref<*xf32> {tt.divisibility = 16 : i32}, %arg2: memref<*xf32> {tt.divisibility = 16 : i32}, %arg3: index, %arg4: index, %arg5: index, %arg6: index, %arg7: index, %arg8: index) {
      affine.parallel (%arg9) = (0) to (8) {
        affine.parallel (%arg10) = (0) to (8) {
          affine.for %arg11 = 0 to affine_map<()[s0, s1] -> (s0 ceildiv 8)>()[%arg3, %arg4] {
            affine.for %arg12 = 0 to affine_map<()[s0, s1] -> (s1 ceildiv 8)>()[%arg3, %arg4] {
              %12 = affine.apply affine_map<(d0, d1) -> (d0 * 8 + d1)>(%arg10, %arg12)
              %13 = affine.apply affine_map<(d0, d1) -> (d0 * 8 + d1)>(%arg9, %arg11)
              %c1024 = arith.constant 1024 : index
              %c1024_0 = arith.constant 1024 : index
              %c1 = arith.constant 1 : index
              %14 = loom.get_symbolic_block_size @constraints::@BM : index
              %15 = loom.get_symbolic_block_size @constraints::@BN : index
              %16 = loom.get_symbolic_block_size @constraints::@BK : index
              %17 = arith.ceildivsi %c1024_0, %16 : index
              %cst = arith.constant 0.000000e+00 : f32
              %c1024_1 = arith.constant 1024 : index
              %c0 = arith.constant 0 : index
              %18 = tensor.empty(%14, %15) : tensor<?x?xf32>
              %19 = linalg.fill ins(%cst : f32) outs(%18 : tensor<?x?xf32>) -> tensor<?x?xf32>
              %20 = arith.muli %13, %14 : index
              %21 = arith.muli %12, %15 : index
              %22 = arith.muli %16, %c1024 : index
              %c0_2 = arith.constant 0 : index
              %23 = arith.muli %c0_2, %16 : index
              %24 = arith.muli %23, %c1024_1 : index
              %25 = arith.addi %24, %21 : index
              %26 = loom.reinterpret_cast %arg1 to offset : [%25], sizes : [%17, %16, %15], strides : [%22, %c1024, %c1], reuse : [seq = false, spat = true, temp = true] : memref<*xf32> to memref<?x?x?xf32, strided<[?, ?, ?], offset: ?>>
              %27 = loom.alloc(%17, %16, %15) on @L1 : memref<?x?x?xf32>
              loom.copy %26, %27, interconnect : [@vertical_links], broadcast : [8, 1] : memref<?x?x?xf32, strided<[?, ?, ?], offset: ?>> to memref<?x?x?xf32>
              %28 = scf.for %arg13 = %c0 to %17 step %c1 iter_args(%arg14 = %19) -> (tensor<?x?xf32>) {
                %32 = arith.muli %arg13, %16 : index
                %33 = arith.muli %20, %c1024_1 : index
                %34 = arith.addi %33, %32 : index
                %35 = loom.reinterpret_cast %arg0 to offset : [%34], sizes : [%14, %16], strides : [%c1024_0, %c1], reuse : [seq = false, spat = true, temp = true] : memref<*xf32> to memref<?x?xf32, strided<[?, ?], offset: ?>>
                %36 = loom.alloc(%14, %16) on @L1 : memref<?x?xf32>
                loom.copy %35, %36, interconnect : [@horizontal_links], broadcast : [1, 8] : memref<?x?xf32, strided<[?, ?], offset: ?>> to memref<?x?xf32>
                %37 = bufferization.to_tensor %36 restrict writable : memref<?x?xf32> to tensor<?x?xf32>
                %38 = arith.muli %32, %c1024_1 : index
                %subview = memref.subview %27[%arg13, 0, 0] [1, %16, %15] [1, 1, 1] : memref<?x?x?xf32> to memref<?x?xf32, strided<[?, 1], offset: ?>>
                %39 = bufferization.to_tensor %subview restrict writable : memref<?x?xf32, strided<[?, 1], offset: ?>> to tensor<?x?xf32>
                %40 = linalg.matmul ins(%37, %39 : tensor<?x?xf32>, tensor<?x?xf32>) outs(%19 : tensor<?x?xf32>) -> tensor<?x?xf32>
                %41 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%arg14, %40 : tensor<?x?xf32>, tensor<?x?xf32>) outs(%arg14 : tensor<?x?xf32>) {
                ^bb0(%in: f32, %in_3: f32, %out: f32):
                  %42 = arith.addf %in, %in_3 : f32
                  linalg.yield %42 : f32
                } -> tensor<?x?xf32>
                scf.yield %41 : tensor<?x?xf32>
              }
              %29 = arith.muli %20, %c1024_1 : index
              %30 = arith.addi %29, %21 : index
              %31 = loom.reinterpret_cast %arg2 to offset : [%30], sizes : [%14, %15], strides : [%c1024, %c1], reuse : [seq = false, spat = false, temp = false] : memref<*xf32> to memref<?x?xf32, strided<[?, ?], offset: ?>>
              bufferization.materialize_in_destination %28 in writable %31 : (tensor<?x?xf32>, memref<?x?xf32, strided<[?, ?], offset: ?>>) -> ()
            }
          }
        } {loom.mapped_to = @y}
      } {loom.mapped_to = @x}
      return
    }
  }
  module attributes {loom.pass_name = "EnumerateCopyBroadcast"} {
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
      %15 = loom.expression(%13, %12) {coeffs = [1, 1], logic = "mul"} : index
      %16 = loom.expression(%15, %14) {coeffs = [1, 1], logic = "mul"} : index
      loom.polynomial_constraint(%12, %13, %14, %15, %16) {monomials = [{coeff = -1 : i64, vars = [4]}], upper_bound = -262144 : i64}
      %17 = loom.expression(%12, %13) {coeffs = [1, 1], logic = "add"} : index
      %18 = loom.expression(%17, %14) {coeffs = [1, 1], logic = "mul"} : index
      loom.polynomial_constraint(%12, %14, %13, %17, %18) {monomials = [{coeff = 1 : i64, vars = [4]}], upper_bound = 374784 : i64}
      loom.polynomial_constraint(%12) {monomials = [{coeff = 1 : i64, vars = [0]}], upper_bound = 128 : i64}
      loom.polynomial_constraint(%13) {monomials = [{coeff = 1 : i64, vars = [0]}], upper_bound = 128 : i64}
      %19 = loom.expression(%12, %14) {coeffs = [1, 1], logic = "mul"} : index
      %20 = loom.expression(%13, %14) {coeffs = [1, 1], logic = "mul"} : index
      %21 = loom.expression(%13, %12) {coeffs = [1, 1], logic = "mul"} : index
      %22 = loom.expression(%14, %21) {coeffs = [1, 1], logic = "mul"} : index
      loom.polynomial_constraint(%12, %13, %14, %19, %20, %21, %22) {monomials = [{coeff = 64 : i64, vars = [3]}, {coeff = 64 : i64, vars = [4]}, {coeff = -1 : i64, vars = [6]}], upper_bound = 0 : i64}
    }
    func.func @matmul_kernel__d0i0_d1i1__f10__d_d(%arg0: memref<*xf32> {tt.divisibility = 16 : i32}, %arg1: memref<*xf32> {tt.divisibility = 16 : i32}, %arg2: memref<*xf32> {tt.divisibility = 16 : i32}, %arg3: index, %arg4: index, %arg5: index, %arg6: index, %arg7: index, %arg8: index) {
      affine.parallel (%arg9) = (0) to (8) {
        affine.parallel (%arg10) = (0) to (8) {
          affine.for %arg11 = 0 to affine_map<()[s0, s1] -> (s1 ceildiv 8)>()[%arg3, %arg4] {
            affine.for %arg12 = 0 to affine_map<()[s0, s1] -> (s0 ceildiv 8)>()[%arg3, %arg4] {
              %12 = affine.apply affine_map<(d0, d1) -> (d0 * 8 + d1)>(%arg10, %arg11)
              %13 = affine.apply affine_map<(d0, d1) -> (d0 * 8 + d1)>(%arg9, %arg12)
              %c1024 = arith.constant 1024 : index
              %c1024_0 = arith.constant 1024 : index
              %c1 = arith.constant 1 : index
              %14 = loom.get_symbolic_block_size @constraints::@BM : index
              %15 = loom.get_symbolic_block_size @constraints::@BN : index
              %16 = loom.get_symbolic_block_size @constraints::@BK : index
              %17 = arith.ceildivsi %c1024_0, %16 : index
              %cst = arith.constant 0.000000e+00 : f32
              %c1024_1 = arith.constant 1024 : index
              %c0 = arith.constant 0 : index
              %18 = tensor.empty(%14, %15) : tensor<?x?xf32>
              %19 = linalg.fill ins(%cst : f32) outs(%18 : tensor<?x?xf32>) -> tensor<?x?xf32>
              %20 = arith.muli %13, %14 : index
              %21 = arith.muli %12, %15 : index
              %22 = scf.for %arg13 = %c0 to %17 step %c1 iter_args(%arg14 = %19) -> (tensor<?x?xf32>) {
                %26 = arith.muli %arg13, %16 : index
                %27 = arith.muli %20, %c1024_1 : index
                %28 = arith.addi %27, %26 : index
                %29 = loom.reinterpret_cast %arg0 to offset : [%28], sizes : [%14, %16], strides : [%c1024_0, %c1], reuse : [seq = false, spat = true, temp = true] : memref<*xf32> to memref<?x?xf32, strided<[?, ?], offset: ?>>
                %30 = loom.alloc(%14, %16) on @L1 : memref<?x?xf32>
                loom.copy %29, %30, interconnect : [], broadcast : [1, 1] : memref<?x?xf32, strided<[?, ?], offset: ?>> to memref<?x?xf32>
                %31 = bufferization.to_tensor %30 restrict writable : memref<?x?xf32> to tensor<?x?xf32>
                %32 = arith.muli %26, %c1024_1 : index
                %33 = arith.addi %32, %21 : index
                %34 = loom.reinterpret_cast %arg1 to offset : [%33], sizes : [%16, %15], strides : [%c1024, %c1], reuse : [seq = false, spat = true, temp = true] : memref<*xf32> to memref<?x?xf32, strided<[?, ?], offset: ?>>
                %35 = loom.alloc(%16, %15) on @L1 : memref<?x?xf32>
                loom.copy %34, %35, interconnect : [], broadcast : [1, 1] : memref<?x?xf32, strided<[?, ?], offset: ?>> to memref<?x?xf32>
                %36 = bufferization.to_tensor %35 restrict writable : memref<?x?xf32> to tensor<?x?xf32>
                %37 = linalg.matmul ins(%31, %36 : tensor<?x?xf32>, tensor<?x?xf32>) outs(%19 : tensor<?x?xf32>) -> tensor<?x?xf32>
                %38 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%arg14, %37 : tensor<?x?xf32>, tensor<?x?xf32>) outs(%arg14 : tensor<?x?xf32>) {
                ^bb0(%in: f32, %in_2: f32, %out: f32):
                  %39 = arith.addf %in, %in_2 : f32
                  linalg.yield %39 : f32
                } -> tensor<?x?xf32>
                scf.yield %38 : tensor<?x?xf32>
              }
              %23 = arith.muli %20, %c1024_1 : index
              %24 = arith.addi %23, %21 : index
              %25 = loom.reinterpret_cast %arg2 to offset : [%24], sizes : [%14, %15], strides : [%c1024, %c1], reuse : [seq = false, spat = false, temp = false] : memref<*xf32> to memref<?x?xf32, strided<[?, ?], offset: ?>>
              bufferization.materialize_in_destination %22 in writable %25 : (tensor<?x?xf32>, memref<?x?xf32, strided<[?, ?], offset: ?>>) -> ()
            }
          }
        } {loom.mapped_to = @y}
      } {loom.mapped_to = @x}
      return
    }
  }
  module attributes {loom.pass_name = "EnumerateCopyBroadcast"} {
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
      %15 = loom.expression(%13, %12) {coeffs = [1, 1], logic = "mul"} : index
      %16 = loom.expression(%14, %15) {coeffs = [1, 1], logic = "mul"} : index
      loom.polynomial_constraint(%12, %13, %14, %15, %16) {monomials = [{coeff = -1 : i64, vars = [4]}], upper_bound = -262144 : i64}
      %17 = loom.expression(%12, %13) {coeffs = [1, 1], logic = "add"} : index
      %18 = loom.expression(%14, %17) {coeffs = [1, 1], logic = "mul"} : index
      loom.polynomial_constraint(%12, %14, %13, %17, %18) {monomials = [{coeff = 1 : i64, vars = [4]}], upper_bound = 374784 : i64}
      loom.polynomial_constraint(%12) {monomials = [{coeff = 1 : i64, vars = [0]}], upper_bound = 128 : i64}
      loom.polynomial_constraint(%13) {monomials = [{coeff = 1 : i64, vars = [0]}], upper_bound = 128 : i64}
      %19 = loom.expression(%12, %14) {coeffs = [1, 1], logic = "mul"} : index
      %20 = loom.expression(%13, %14) {coeffs = [1, 1], logic = "mul"} : index
      %21 = loom.expression(%13, %12) {coeffs = [1, 1], logic = "mul"} : index
      %22 = loom.expression(%14, %21) {coeffs = [1, 1], logic = "mul"} : index
      loom.polynomial_constraint(%12, %13, %14, %19, %20, %21, %22) {monomials = [{coeff = 960 : i64, vars = [3]}, {coeff = 256 : i64, vars = [4]}, {coeff = -15 : i64, vars = [6]}], upper_bound = 0 : i64}
    }
    func.func @matmul_kernel__d0i0_d1i1__f10__d_v(%arg0: memref<*xf32> {tt.divisibility = 16 : i32}, %arg1: memref<*xf32> {tt.divisibility = 16 : i32}, %arg2: memref<*xf32> {tt.divisibility = 16 : i32}, %arg3: index, %arg4: index, %arg5: index, %arg6: index, %arg7: index, %arg8: index) {
      affine.parallel (%arg9) = (0) to (8) {
        affine.parallel (%arg10) = (0) to (8) {
          affine.for %arg11 = 0 to affine_map<()[s0, s1] -> (s1 ceildiv 8)>()[%arg3, %arg4] {
            affine.for %arg12 = 0 to affine_map<()[s0, s1] -> (s0 ceildiv 8)>()[%arg3, %arg4] {
              %12 = affine.apply affine_map<(d0, d1) -> (d0 * 8 + d1)>(%arg10, %arg11)
              %13 = affine.apply affine_map<(d0, d1) -> (d0 * 8 + d1)>(%arg9, %arg12)
              %c1024 = arith.constant 1024 : index
              %c1024_0 = arith.constant 1024 : index
              %c1 = arith.constant 1 : index
              %14 = loom.get_symbolic_block_size @constraints::@BM : index
              %15 = loom.get_symbolic_block_size @constraints::@BN : index
              %16 = loom.get_symbolic_block_size @constraints::@BK : index
              %17 = arith.ceildivsi %c1024_0, %16 : index
              %cst = arith.constant 0.000000e+00 : f32
              %c1024_1 = arith.constant 1024 : index
              %c0 = arith.constant 0 : index
              %18 = tensor.empty(%14, %15) : tensor<?x?xf32>
              %19 = linalg.fill ins(%cst : f32) outs(%18 : tensor<?x?xf32>) -> tensor<?x?xf32>
              %20 = arith.muli %13, %14 : index
              %21 = arith.muli %12, %15 : index
              %22 = scf.for %arg13 = %c0 to %17 step %c1 iter_args(%arg14 = %19) -> (tensor<?x?xf32>) {
                %26 = arith.muli %arg13, %16 : index
                %27 = arith.muli %20, %c1024_1 : index
                %28 = arith.addi %27, %26 : index
                %29 = loom.reinterpret_cast %arg0 to offset : [%28], sizes : [%14, %16], strides : [%c1024_0, %c1], reuse : [seq = false, spat = true, temp = true] : memref<*xf32> to memref<?x?xf32, strided<[?, ?], offset: ?>>
                %30 = loom.alloc(%14, %16) on @L1 : memref<?x?xf32>
                loom.copy %29, %30, interconnect : [], broadcast : [1, 1] : memref<?x?xf32, strided<[?, ?], offset: ?>> to memref<?x?xf32>
                %31 = bufferization.to_tensor %30 restrict writable : memref<?x?xf32> to tensor<?x?xf32>
                %32 = arith.muli %26, %c1024_1 : index
                %33 = arith.addi %32, %21 : index
                %34 = loom.reinterpret_cast %arg1 to offset : [%33], sizes : [%16, %15], strides : [%c1024, %c1], reuse : [seq = false, spat = true, temp = true] : memref<*xf32> to memref<?x?xf32, strided<[?, ?], offset: ?>>
                %35 = loom.alloc(%16, %15) on @L1 : memref<?x?xf32>
                loom.copy %34, %35, interconnect : [@vertical_links], broadcast : [8, 1] : memref<?x?xf32, strided<[?, ?], offset: ?>> to memref<?x?xf32>
                %36 = bufferization.to_tensor %35 restrict writable : memref<?x?xf32> to tensor<?x?xf32>
                %37 = linalg.matmul ins(%31, %36 : tensor<?x?xf32>, tensor<?x?xf32>) outs(%19 : tensor<?x?xf32>) -> tensor<?x?xf32>
                %38 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%arg14, %37 : tensor<?x?xf32>, tensor<?x?xf32>) outs(%arg14 : tensor<?x?xf32>) {
                ^bb0(%in: f32, %in_2: f32, %out: f32):
                  %39 = arith.addf %in, %in_2 : f32
                  linalg.yield %39 : f32
                } -> tensor<?x?xf32>
                scf.yield %38 : tensor<?x?xf32>
              }
              %23 = arith.muli %20, %c1024_1 : index
              %24 = arith.addi %23, %21 : index
              %25 = loom.reinterpret_cast %arg2 to offset : [%24], sizes : [%14, %15], strides : [%c1024, %c1], reuse : [seq = false, spat = false, temp = false] : memref<*xf32> to memref<?x?xf32, strided<[?, ?], offset: ?>>
              bufferization.materialize_in_destination %22 in writable %25 : (tensor<?x?xf32>, memref<?x?xf32, strided<[?, ?], offset: ?>>) -> ()
            }
          }
        } {loom.mapped_to = @y}
      } {loom.mapped_to = @x}
      return
    }
  }
  module attributes {loom.pass_name = "EnumerateCopyBroadcast"} {
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
      %15 = loom.expression(%12, %13) {coeffs = [1, 1], logic = "mul"} : index
      %16 = loom.expression(%15, %14) {coeffs = [1, 1], logic = "mul"} : index
      loom.polynomial_constraint(%12, %13, %14, %15, %16) {monomials = [{coeff = -1 : i64, vars = [4]}], upper_bound = -262144 : i64}
      %17 = loom.expression(%12, %13) {coeffs = [1, 1], logic = "add"} : index
      %18 = loom.expression(%17, %14) {coeffs = [1, 1], logic = "mul"} : index
      loom.polynomial_constraint(%12, %14, %13, %17, %18) {monomials = [{coeff = 1 : i64, vars = [4]}], upper_bound = 374784 : i64}
      loom.polynomial_constraint(%12) {monomials = [{coeff = 1 : i64, vars = [0]}], upper_bound = 128 : i64}
      loom.polynomial_constraint(%13) {monomials = [{coeff = 1 : i64, vars = [0]}], upper_bound = 128 : i64}
      %19 = loom.expression(%12, %14) {coeffs = [1, 1], logic = "mul"} : index
      %20 = loom.expression(%13, %14) {coeffs = [1, 1], logic = "mul"} : index
      %21 = loom.expression(%12, %13) {coeffs = [1, 1], logic = "mul"} : index
      %22 = loom.expression(%14, %21) {coeffs = [1, 1], logic = "mul"} : index
      loom.polynomial_constraint(%12, %13, %14, %19, %20, %21, %22) {monomials = [{coeff = 256 : i64, vars = [3]}, {coeff = 960 : i64, vars = [4]}, {coeff = -15 : i64, vars = [6]}], upper_bound = 0 : i64}
    }
    func.func @matmul_kernel__d0i0_d1i1__f10__h_d(%arg0: memref<*xf32> {tt.divisibility = 16 : i32}, %arg1: memref<*xf32> {tt.divisibility = 16 : i32}, %arg2: memref<*xf32> {tt.divisibility = 16 : i32}, %arg3: index, %arg4: index, %arg5: index, %arg6: index, %arg7: index, %arg8: index) {
      affine.parallel (%arg9) = (0) to (8) {
        affine.parallel (%arg10) = (0) to (8) {
          affine.for %arg11 = 0 to affine_map<()[s0, s1] -> (s1 ceildiv 8)>()[%arg3, %arg4] {
            affine.for %arg12 = 0 to affine_map<()[s0, s1] -> (s0 ceildiv 8)>()[%arg3, %arg4] {
              %12 = affine.apply affine_map<(d0, d1) -> (d0 * 8 + d1)>(%arg10, %arg11)
              %13 = affine.apply affine_map<(d0, d1) -> (d0 * 8 + d1)>(%arg9, %arg12)
              %c1024 = arith.constant 1024 : index
              %c1024_0 = arith.constant 1024 : index
              %c1 = arith.constant 1 : index
              %14 = loom.get_symbolic_block_size @constraints::@BM : index
              %15 = loom.get_symbolic_block_size @constraints::@BN : index
              %16 = loom.get_symbolic_block_size @constraints::@BK : index
              %17 = arith.ceildivsi %c1024_0, %16 : index
              %cst = arith.constant 0.000000e+00 : f32
              %c1024_1 = arith.constant 1024 : index
              %c0 = arith.constant 0 : index
              %18 = tensor.empty(%14, %15) : tensor<?x?xf32>
              %19 = linalg.fill ins(%cst : f32) outs(%18 : tensor<?x?xf32>) -> tensor<?x?xf32>
              %20 = arith.muli %13, %14 : index
              %21 = arith.muli %12, %15 : index
              %22 = scf.for %arg13 = %c0 to %17 step %c1 iter_args(%arg14 = %19) -> (tensor<?x?xf32>) {
                %26 = arith.muli %arg13, %16 : index
                %27 = arith.muli %20, %c1024_1 : index
                %28 = arith.addi %27, %26 : index
                %29 = loom.reinterpret_cast %arg0 to offset : [%28], sizes : [%14, %16], strides : [%c1024_0, %c1], reuse : [seq = false, spat = true, temp = true] : memref<*xf32> to memref<?x?xf32, strided<[?, ?], offset: ?>>
                %30 = loom.alloc(%14, %16) on @L1 : memref<?x?xf32>
                loom.copy %29, %30, interconnect : [@horizontal_links], broadcast : [1, 8] : memref<?x?xf32, strided<[?, ?], offset: ?>> to memref<?x?xf32>
                %31 = bufferization.to_tensor %30 restrict writable : memref<?x?xf32> to tensor<?x?xf32>
                %32 = arith.muli %26, %c1024_1 : index
                %33 = arith.addi %32, %21 : index
                %34 = loom.reinterpret_cast %arg1 to offset : [%33], sizes : [%16, %15], strides : [%c1024, %c1], reuse : [seq = false, spat = true, temp = true] : memref<*xf32> to memref<?x?xf32, strided<[?, ?], offset: ?>>
                %35 = loom.alloc(%16, %15) on @L1 : memref<?x?xf32>
                loom.copy %34, %35, interconnect : [], broadcast : [1, 1] : memref<?x?xf32, strided<[?, ?], offset: ?>> to memref<?x?xf32>
                %36 = bufferization.to_tensor %35 restrict writable : memref<?x?xf32> to tensor<?x?xf32>
                %37 = linalg.matmul ins(%31, %36 : tensor<?x?xf32>, tensor<?x?xf32>) outs(%19 : tensor<?x?xf32>) -> tensor<?x?xf32>
                %38 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%arg14, %37 : tensor<?x?xf32>, tensor<?x?xf32>) outs(%arg14 : tensor<?x?xf32>) {
                ^bb0(%in: f32, %in_2: f32, %out: f32):
                  %39 = arith.addf %in, %in_2 : f32
                  linalg.yield %39 : f32
                } -> tensor<?x?xf32>
                scf.yield %38 : tensor<?x?xf32>
              }
              %23 = arith.muli %20, %c1024_1 : index
              %24 = arith.addi %23, %21 : index
              %25 = loom.reinterpret_cast %arg2 to offset : [%24], sizes : [%14, %15], strides : [%c1024, %c1], reuse : [seq = false, spat = false, temp = false] : memref<*xf32> to memref<?x?xf32, strided<[?, ?], offset: ?>>
              bufferization.materialize_in_destination %22 in writable %25 : (tensor<?x?xf32>, memref<?x?xf32, strided<[?, ?], offset: ?>>) -> ()
            }
          }
        } {loom.mapped_to = @y}
      } {loom.mapped_to = @x}
      return
    }
  }
  module attributes {loom.pass_name = "EnumerateCopyBroadcast"} {
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
      %15 = loom.expression(%13, %12) {coeffs = [1, 1], logic = "mul"} : index
      %16 = loom.expression(%14, %15) {coeffs = [1, 1], logic = "mul"} : index
      loom.polynomial_constraint(%12, %13, %14, %15, %16) {monomials = [{coeff = -1 : i64, vars = [4]}], upper_bound = -262144 : i64}
      %17 = loom.expression(%12, %13) {coeffs = [1, 1], logic = "add"} : index
      %18 = loom.expression(%14, %17) {coeffs = [1, 1], logic = "mul"} : index
      loom.polynomial_constraint(%12, %14, %13, %17, %18) {monomials = [{coeff = 1 : i64, vars = [4]}], upper_bound = 374784 : i64}
      loom.polynomial_constraint(%12) {monomials = [{coeff = 1 : i64, vars = [0]}], upper_bound = 128 : i64}
      loom.polynomial_constraint(%13) {monomials = [{coeff = 1 : i64, vars = [0]}], upper_bound = 128 : i64}
      %19 = loom.expression(%14, %12) {coeffs = [1, 1], logic = "mul"} : index
      %20 = loom.expression(%13, %14) {coeffs = [1, 1], logic = "mul"} : index
      %21 = loom.expression(%13, %12) {coeffs = [1, 1], logic = "mul"} : index
      %22 = loom.expression(%14, %21) {coeffs = [1, 1], logic = "mul"} : index
      loom.polynomial_constraint(%12, %13, %14, %19, %20, %21, %22) {monomials = [{coeff = 256 : i64, vars = [3]}, {coeff = 256 : i64, vars = [4]}, {coeff = -15 : i64, vars = [6]}], upper_bound = 0 : i64}
    }
    func.func @matmul_kernel__d0i0_d1i1__f10__h_v(%arg0: memref<*xf32> {tt.divisibility = 16 : i32}, %arg1: memref<*xf32> {tt.divisibility = 16 : i32}, %arg2: memref<*xf32> {tt.divisibility = 16 : i32}, %arg3: index, %arg4: index, %arg5: index, %arg6: index, %arg7: index, %arg8: index) {
      affine.parallel (%arg9) = (0) to (8) {
        affine.parallel (%arg10) = (0) to (8) {
          affine.for %arg11 = 0 to affine_map<()[s0, s1] -> (s1 ceildiv 8)>()[%arg3, %arg4] {
            affine.for %arg12 = 0 to affine_map<()[s0, s1] -> (s0 ceildiv 8)>()[%arg3, %arg4] {
              %12 = affine.apply affine_map<(d0, d1) -> (d0 * 8 + d1)>(%arg10, %arg11)
              %13 = affine.apply affine_map<(d0, d1) -> (d0 * 8 + d1)>(%arg9, %arg12)
              %c1024 = arith.constant 1024 : index
              %c1024_0 = arith.constant 1024 : index
              %c1 = arith.constant 1 : index
              %14 = loom.get_symbolic_block_size @constraints::@BM : index
              %15 = loom.get_symbolic_block_size @constraints::@BN : index
              %16 = loom.get_symbolic_block_size @constraints::@BK : index
              %17 = arith.ceildivsi %c1024_0, %16 : index
              %cst = arith.constant 0.000000e+00 : f32
              %c1024_1 = arith.constant 1024 : index
              %c0 = arith.constant 0 : index
              %18 = tensor.empty(%14, %15) : tensor<?x?xf32>
              %19 = linalg.fill ins(%cst : f32) outs(%18 : tensor<?x?xf32>) -> tensor<?x?xf32>
              %20 = arith.muli %13, %14 : index
              %21 = arith.muli %12, %15 : index
              %22 = scf.for %arg13 = %c0 to %17 step %c1 iter_args(%arg14 = %19) -> (tensor<?x?xf32>) {
                %26 = arith.muli %arg13, %16 : index
                %27 = arith.muli %20, %c1024_1 : index
                %28 = arith.addi %27, %26 : index
                %29 = loom.reinterpret_cast %arg0 to offset : [%28], sizes : [%14, %16], strides : [%c1024_0, %c1], reuse : [seq = false, spat = true, temp = true] : memref<*xf32> to memref<?x?xf32, strided<[?, ?], offset: ?>>
                %30 = loom.alloc(%14, %16) on @L1 : memref<?x?xf32>
                loom.copy %29, %30, interconnect : [@horizontal_links], broadcast : [1, 8] : memref<?x?xf32, strided<[?, ?], offset: ?>> to memref<?x?xf32>
                %31 = bufferization.to_tensor %30 restrict writable : memref<?x?xf32> to tensor<?x?xf32>
                %32 = arith.muli %26, %c1024_1 : index
                %33 = arith.addi %32, %21 : index
                %34 = loom.reinterpret_cast %arg1 to offset : [%33], sizes : [%16, %15], strides : [%c1024, %c1], reuse : [seq = false, spat = true, temp = true] : memref<*xf32> to memref<?x?xf32, strided<[?, ?], offset: ?>>
                %35 = loom.alloc(%16, %15) on @L1 : memref<?x?xf32>
                loom.copy %34, %35, interconnect : [@vertical_links], broadcast : [8, 1] : memref<?x?xf32, strided<[?, ?], offset: ?>> to memref<?x?xf32>
                %36 = bufferization.to_tensor %35 restrict writable : memref<?x?xf32> to tensor<?x?xf32>
                %37 = linalg.matmul ins(%31, %36 : tensor<?x?xf32>, tensor<?x?xf32>) outs(%19 : tensor<?x?xf32>) -> tensor<?x?xf32>
                %38 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%arg14, %37 : tensor<?x?xf32>, tensor<?x?xf32>) outs(%arg14 : tensor<?x?xf32>) {
                ^bb0(%in: f32, %in_2: f32, %out: f32):
                  %39 = arith.addf %in, %in_2 : f32
                  linalg.yield %39 : f32
                } -> tensor<?x?xf32>
                scf.yield %38 : tensor<?x?xf32>
              }
              %23 = arith.muli %20, %c1024_1 : index
              %24 = arith.addi %23, %21 : index
              %25 = loom.reinterpret_cast %arg2 to offset : [%24], sizes : [%14, %15], strides : [%c1024, %c1], reuse : [seq = false, spat = false, temp = false] : memref<*xf32> to memref<?x?xf32, strided<[?, ?], offset: ?>>
              bufferization.materialize_in_destination %22 in writable %25 : (tensor<?x?xf32>, memref<?x?xf32, strided<[?, ?], offset: ?>>) -> ()
            }
          }
        } {loom.mapped_to = @y}
      } {loom.mapped_to = @x}
      return
    }
  }
  module attributes {loom.pass_name = "EnumerateCopyBroadcast"} {
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
      %15 = loom.expression(%12, %13) {coeffs = [1, 1], logic = "mul"} : index
      %16 = loom.expression(%15, %14) {coeffs = [1, 1], logic = "mul"} : index
      loom.polynomial_constraint(%12, %13, %14, %15, %16) {monomials = [{coeff = -1 : i64, vars = [4]}], upper_bound = -262144 : i64}
      %17 = loom.expression(%12, %13) {coeffs = [1, 1], logic = "add"} : index
      %18 = loom.expression(%17, %14) {coeffs = [1, 1], logic = "mul"} : index
      loom.polynomial_constraint(%12, %14, %13, %17, %18) {monomials = [{coeff = 1 : i64, vars = [4]}], upper_bound = 374784 : i64}
      loom.polynomial_constraint(%12) {monomials = [{coeff = 1 : i64, vars = [0]}], upper_bound = 128 : i64}
      loom.polynomial_constraint(%13) {monomials = [{coeff = 1 : i64, vars = [0]}], upper_bound = 128 : i64}
      %19 = loom.expression(%13, %14) {coeffs = [1, 1], logic = "mul"} : index
      loom.polynomial_constraint(%12, %14, %13, %19) {monomials = [{coeff = 1024 : i64, vars = [0]}, {coeff = 1 : i64, vars = [3]}], upper_bound = 374784 : i64}
      %20 = loom.expression(%12, %14) {coeffs = [1, 1], logic = "mul"} : index
      %21 = loom.expression(%13, %14) {coeffs = [1, 1], logic = "mul"} : index
      %22 = loom.expression(%12, %13) {coeffs = [1, 1], logic = "mul"} : index
      %23 = loom.expression(%14, %22) {coeffs = [1, 1], logic = "mul"} : index
      loom.polynomial_constraint(%12, %13, %14, %20, %21, %22, %23) {monomials = [{coeff = 64 : i64, vars = [3]}, {coeff = 64 : i64, vars = [4]}, {coeff = -1 : i64, vars = [6]}], upper_bound = 0 : i64}
    }
    func.func @matmul_kernel__d0i0_d1i1__f10__hoist_block_0__d_d(%arg0: memref<*xf32> {tt.divisibility = 16 : i32}, %arg1: memref<*xf32> {tt.divisibility = 16 : i32}, %arg2: memref<*xf32> {tt.divisibility = 16 : i32}, %arg3: index, %arg4: index, %arg5: index, %arg6: index, %arg7: index, %arg8: index) {
      affine.parallel (%arg9) = (0) to (8) {
        affine.parallel (%arg10) = (0) to (8) {
          affine.for %arg11 = 0 to affine_map<()[s0, s1] -> (s1 ceildiv 8)>()[%arg3, %arg4] {
            affine.for %arg12 = 0 to affine_map<()[s0, s1] -> (s0 ceildiv 8)>()[%arg3, %arg4] {
              %12 = affine.apply affine_map<(d0, d1) -> (d0 * 8 + d1)>(%arg10, %arg11)
              %13 = affine.apply affine_map<(d0, d1) -> (d0 * 8 + d1)>(%arg9, %arg12)
              %c1024 = arith.constant 1024 : index
              %c1024_0 = arith.constant 1024 : index
              %c1 = arith.constant 1 : index
              %14 = loom.get_symbolic_block_size @constraints::@BM : index
              %15 = loom.get_symbolic_block_size @constraints::@BN : index
              %16 = loom.get_symbolic_block_size @constraints::@BK : index
              %17 = arith.ceildivsi %c1024_0, %16 : index
              %cst = arith.constant 0.000000e+00 : f32
              %c1024_1 = arith.constant 1024 : index
              %c0 = arith.constant 0 : index
              %18 = tensor.empty(%14, %15) : tensor<?x?xf32>
              %19 = linalg.fill ins(%cst : f32) outs(%18 : tensor<?x?xf32>) -> tensor<?x?xf32>
              %20 = arith.muli %13, %14 : index
              %21 = arith.muli %12, %15 : index
              %22 = arith.muli %14, %c1024_0 : index
              %c0_2 = arith.constant 0 : index
              %23 = arith.muli %20, %c1024_1 : index
              %24 = arith.muli %c0_2, %16 : index
              %25 = arith.addi %23, %24 : index
              %26 = loom.reinterpret_cast %arg0 to offset : [%25], sizes : [%17, %14, %16], strides : [%22, %c1024_0, %c1], reuse : [seq = false, spat = true, temp = true] : memref<*xf32> to memref<?x?x?xf32, strided<[?, ?, ?], offset: ?>>
              %27 = loom.alloc(%17, %14, %16) on @L1 : memref<?x?x?xf32>
              loom.copy %26, %27, interconnect : [], broadcast : [1, 1] : memref<?x?x?xf32, strided<[?, ?, ?], offset: ?>> to memref<?x?x?xf32>
              %28 = scf.for %arg13 = %c0 to %17 step %c1 iter_args(%arg14 = %19) -> (tensor<?x?xf32>) {
                %32 = arith.muli %arg13, %16 : index
                %33 = arith.muli %20, %c1024_1 : index
                %subview = memref.subview %27[%arg13, 0, 0] [1, %14, %16] [1, 1, 1] : memref<?x?x?xf32> to memref<?x?xf32, strided<[?, 1], offset: ?>>
                %34 = bufferization.to_tensor %subview restrict writable : memref<?x?xf32, strided<[?, 1], offset: ?>> to tensor<?x?xf32>
                %35 = arith.muli %32, %c1024_1 : index
                %36 = arith.addi %35, %21 : index
                %37 = loom.reinterpret_cast %arg1 to offset : [%36], sizes : [%16, %15], strides : [%c1024, %c1], reuse : [seq = false, spat = true, temp = true] : memref<*xf32> to memref<?x?xf32, strided<[?, ?], offset: ?>>
                %38 = loom.alloc(%16, %15) on @L1 : memref<?x?xf32>
                loom.copy %37, %38, interconnect : [], broadcast : [1, 1] : memref<?x?xf32, strided<[?, ?], offset: ?>> to memref<?x?xf32>
                %39 = bufferization.to_tensor %38 restrict writable : memref<?x?xf32> to tensor<?x?xf32>
                %40 = linalg.matmul ins(%34, %39 : tensor<?x?xf32>, tensor<?x?xf32>) outs(%19 : tensor<?x?xf32>) -> tensor<?x?xf32>
                %41 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%arg14, %40 : tensor<?x?xf32>, tensor<?x?xf32>) outs(%arg14 : tensor<?x?xf32>) {
                ^bb0(%in: f32, %in_3: f32, %out: f32):
                  %42 = arith.addf %in, %in_3 : f32
                  linalg.yield %42 : f32
                } -> tensor<?x?xf32>
                scf.yield %41 : tensor<?x?xf32>
              }
              %29 = arith.muli %20, %c1024_1 : index
              %30 = arith.addi %29, %21 : index
              %31 = loom.reinterpret_cast %arg2 to offset : [%30], sizes : [%14, %15], strides : [%c1024, %c1], reuse : [seq = false, spat = false, temp = false] : memref<*xf32> to memref<?x?xf32, strided<[?, ?], offset: ?>>
              bufferization.materialize_in_destination %28 in writable %31 : (tensor<?x?xf32>, memref<?x?xf32, strided<[?, ?], offset: ?>>) -> ()
            }
          }
        } {loom.mapped_to = @y}
      } {loom.mapped_to = @x}
      return
    }
  }
  module attributes {loom.pass_name = "EnumerateCopyBroadcast"} {
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
      %15 = loom.expression(%13, %12) {coeffs = [1, 1], logic = "mul"} : index
      %16 = loom.expression(%15, %14) {coeffs = [1, 1], logic = "mul"} : index
      loom.polynomial_constraint(%12, %13, %14, %15, %16) {monomials = [{coeff = -1 : i64, vars = [4]}], upper_bound = -262144 : i64}
      %17 = loom.expression(%12, %13) {coeffs = [1, 1], logic = "add"} : index
      %18 = loom.expression(%14, %17) {coeffs = [1, 1], logic = "mul"} : index
      loom.polynomial_constraint(%12, %14, %13, %17, %18) {monomials = [{coeff = 1 : i64, vars = [4]}], upper_bound = 374784 : i64}
      loom.polynomial_constraint(%12) {monomials = [{coeff = 1 : i64, vars = [0]}], upper_bound = 128 : i64}
      loom.polynomial_constraint(%13) {monomials = [{coeff = 1 : i64, vars = [0]}], upper_bound = 128 : i64}
      %19 = loom.expression(%13, %14) {coeffs = [1, 1], logic = "mul"} : index
      loom.polynomial_constraint(%12, %14, %13, %19) {monomials = [{coeff = 1024 : i64, vars = [0]}, {coeff = 1 : i64, vars = [3]}], upper_bound = 374784 : i64}
      %20 = loom.expression(%14, %12) {coeffs = [1, 1], logic = "mul"} : index
      %21 = loom.expression(%13, %14) {coeffs = [1, 1], logic = "mul"} : index
      %22 = loom.expression(%13, %12) {coeffs = [1, 1], logic = "mul"} : index
      %23 = loom.expression(%14, %22) {coeffs = [1, 1], logic = "mul"} : index
      loom.polynomial_constraint(%12, %13, %14, %20, %21, %22, %23) {monomials = [{coeff = 960 : i64, vars = [3]}, {coeff = 256 : i64, vars = [4]}, {coeff = -15 : i64, vars = [6]}], upper_bound = 0 : i64}
    }
    func.func @matmul_kernel__d0i0_d1i1__f10__hoist_block_0__d_v(%arg0: memref<*xf32> {tt.divisibility = 16 : i32}, %arg1: memref<*xf32> {tt.divisibility = 16 : i32}, %arg2: memref<*xf32> {tt.divisibility = 16 : i32}, %arg3: index, %arg4: index, %arg5: index, %arg6: index, %arg7: index, %arg8: index) {
      affine.parallel (%arg9) = (0) to (8) {
        affine.parallel (%arg10) = (0) to (8) {
          affine.for %arg11 = 0 to affine_map<()[s0, s1] -> (s1 ceildiv 8)>()[%arg3, %arg4] {
            affine.for %arg12 = 0 to affine_map<()[s0, s1] -> (s0 ceildiv 8)>()[%arg3, %arg4] {
              %12 = affine.apply affine_map<(d0, d1) -> (d0 * 8 + d1)>(%arg10, %arg11)
              %13 = affine.apply affine_map<(d0, d1) -> (d0 * 8 + d1)>(%arg9, %arg12)
              %c1024 = arith.constant 1024 : index
              %c1024_0 = arith.constant 1024 : index
              %c1 = arith.constant 1 : index
              %14 = loom.get_symbolic_block_size @constraints::@BM : index
              %15 = loom.get_symbolic_block_size @constraints::@BN : index
              %16 = loom.get_symbolic_block_size @constraints::@BK : index
              %17 = arith.ceildivsi %c1024_0, %16 : index
              %cst = arith.constant 0.000000e+00 : f32
              %c1024_1 = arith.constant 1024 : index
              %c0 = arith.constant 0 : index
              %18 = tensor.empty(%14, %15) : tensor<?x?xf32>
              %19 = linalg.fill ins(%cst : f32) outs(%18 : tensor<?x?xf32>) -> tensor<?x?xf32>
              %20 = arith.muli %13, %14 : index
              %21 = arith.muli %12, %15 : index
              %22 = arith.muli %14, %c1024_0 : index
              %c0_2 = arith.constant 0 : index
              %23 = arith.muli %20, %c1024_1 : index
              %24 = arith.muli %c0_2, %16 : index
              %25 = arith.addi %23, %24 : index
              %26 = loom.reinterpret_cast %arg0 to offset : [%25], sizes : [%17, %14, %16], strides : [%22, %c1024_0, %c1], reuse : [seq = false, spat = true, temp = true] : memref<*xf32> to memref<?x?x?xf32, strided<[?, ?, ?], offset: ?>>
              %27 = loom.alloc(%17, %14, %16) on @L1 : memref<?x?x?xf32>
              loom.copy %26, %27, interconnect : [], broadcast : [1, 1] : memref<?x?x?xf32, strided<[?, ?, ?], offset: ?>> to memref<?x?x?xf32>
              %28 = scf.for %arg13 = %c0 to %17 step %c1 iter_args(%arg14 = %19) -> (tensor<?x?xf32>) {
                %32 = arith.muli %arg13, %16 : index
                %33 = arith.muli %20, %c1024_1 : index
                %subview = memref.subview %27[%arg13, 0, 0] [1, %14, %16] [1, 1, 1] : memref<?x?x?xf32> to memref<?x?xf32, strided<[?, 1], offset: ?>>
                %34 = bufferization.to_tensor %subview restrict writable : memref<?x?xf32, strided<[?, 1], offset: ?>> to tensor<?x?xf32>
                %35 = arith.muli %32, %c1024_1 : index
                %36 = arith.addi %35, %21 : index
                %37 = loom.reinterpret_cast %arg1 to offset : [%36], sizes : [%16, %15], strides : [%c1024, %c1], reuse : [seq = false, spat = true, temp = true] : memref<*xf32> to memref<?x?xf32, strided<[?, ?], offset: ?>>
                %38 = loom.alloc(%16, %15) on @L1 : memref<?x?xf32>
                loom.copy %37, %38, interconnect : [@vertical_links], broadcast : [8, 1] : memref<?x?xf32, strided<[?, ?], offset: ?>> to memref<?x?xf32>
                %39 = bufferization.to_tensor %38 restrict writable : memref<?x?xf32> to tensor<?x?xf32>
                %40 = linalg.matmul ins(%34, %39 : tensor<?x?xf32>, tensor<?x?xf32>) outs(%19 : tensor<?x?xf32>) -> tensor<?x?xf32>
                %41 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%arg14, %40 : tensor<?x?xf32>, tensor<?x?xf32>) outs(%arg14 : tensor<?x?xf32>) {
                ^bb0(%in: f32, %in_3: f32, %out: f32):
                  %42 = arith.addf %in, %in_3 : f32
                  linalg.yield %42 : f32
                } -> tensor<?x?xf32>
                scf.yield %41 : tensor<?x?xf32>
              }
              %29 = arith.muli %20, %c1024_1 : index
              %30 = arith.addi %29, %21 : index
              %31 = loom.reinterpret_cast %arg2 to offset : [%30], sizes : [%14, %15], strides : [%c1024, %c1], reuse : [seq = false, spat = false, temp = false] : memref<*xf32> to memref<?x?xf32, strided<[?, ?], offset: ?>>
              bufferization.materialize_in_destination %28 in writable %31 : (tensor<?x?xf32>, memref<?x?xf32, strided<[?, ?], offset: ?>>) -> ()
            }
          }
        } {loom.mapped_to = @y}
      } {loom.mapped_to = @x}
      return
    }
  }
  module attributes {loom.pass_name = "EnumerateCopyBroadcast"} {
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
      %15 = loom.expression(%13, %12) {coeffs = [1, 1], logic = "mul"} : index
      %16 = loom.expression(%15, %14) {coeffs = [1, 1], logic = "mul"} : index
      loom.polynomial_constraint(%12, %13, %14, %15, %16) {monomials = [{coeff = -1 : i64, vars = [4]}], upper_bound = -262144 : i64}
      %17 = loom.expression(%12, %13) {coeffs = [1, 1], logic = "add"} : index
      %18 = loom.expression(%14, %17) {coeffs = [1, 1], logic = "mul"} : index
      loom.polynomial_constraint(%12, %14, %13, %17, %18) {monomials = [{coeff = 1 : i64, vars = [4]}], upper_bound = 374784 : i64}
      loom.polynomial_constraint(%12) {monomials = [{coeff = 1 : i64, vars = [0]}], upper_bound = 128 : i64}
      loom.polynomial_constraint(%13) {monomials = [{coeff = 1 : i64, vars = [0]}], upper_bound = 128 : i64}
      %19 = loom.expression(%13, %14) {coeffs = [1, 1], logic = "mul"} : index
      loom.polynomial_constraint(%12, %14, %13, %19) {monomials = [{coeff = 1024 : i64, vars = [0]}, {coeff = 1 : i64, vars = [3]}], upper_bound = 374784 : i64}
      %20 = loom.expression(%14, %12) {coeffs = [1, 1], logic = "mul"} : index
      %21 = loom.expression(%13, %14) {coeffs = [1, 1], logic = "mul"} : index
      %22 = loom.expression(%13, %12) {coeffs = [1, 1], logic = "mul"} : index
      %23 = loom.expression(%14, %22) {coeffs = [1, 1], logic = "mul"} : index
      loom.polynomial_constraint(%12, %13, %14, %20, %21, %22, %23) {monomials = [{coeff = 256 : i64, vars = [3]}, {coeff = 960 : i64, vars = [4]}, {coeff = -15 : i64, vars = [6]}], upper_bound = 0 : i64}
    }
    func.func @matmul_kernel__d0i0_d1i1__f10__hoist_block_0__h_d(%arg0: memref<*xf32> {tt.divisibility = 16 : i32}, %arg1: memref<*xf32> {tt.divisibility = 16 : i32}, %arg2: memref<*xf32> {tt.divisibility = 16 : i32}, %arg3: index, %arg4: index, %arg5: index, %arg6: index, %arg7: index, %arg8: index) {
      affine.parallel (%arg9) = (0) to (8) {
        affine.parallel (%arg10) = (0) to (8) {
          affine.for %arg11 = 0 to affine_map<()[s0, s1] -> (s1 ceildiv 8)>()[%arg3, %arg4] {
            affine.for %arg12 = 0 to affine_map<()[s0, s1] -> (s0 ceildiv 8)>()[%arg3, %arg4] {
              %12 = affine.apply affine_map<(d0, d1) -> (d0 * 8 + d1)>(%arg10, %arg11)
              %13 = affine.apply affine_map<(d0, d1) -> (d0 * 8 + d1)>(%arg9, %arg12)
              %c1024 = arith.constant 1024 : index
              %c1024_0 = arith.constant 1024 : index
              %c1 = arith.constant 1 : index
              %14 = loom.get_symbolic_block_size @constraints::@BM : index
              %15 = loom.get_symbolic_block_size @constraints::@BN : index
              %16 = loom.get_symbolic_block_size @constraints::@BK : index
              %17 = arith.ceildivsi %c1024_0, %16 : index
              %cst = arith.constant 0.000000e+00 : f32
              %c1024_1 = arith.constant 1024 : index
              %c0 = arith.constant 0 : index
              %18 = tensor.empty(%14, %15) : tensor<?x?xf32>
              %19 = linalg.fill ins(%cst : f32) outs(%18 : tensor<?x?xf32>) -> tensor<?x?xf32>
              %20 = arith.muli %13, %14 : index
              %21 = arith.muli %12, %15 : index
              %22 = arith.muli %14, %c1024_0 : index
              %c0_2 = arith.constant 0 : index
              %23 = arith.muli %20, %c1024_1 : index
              %24 = arith.muli %c0_2, %16 : index
              %25 = arith.addi %23, %24 : index
              %26 = loom.reinterpret_cast %arg0 to offset : [%25], sizes : [%17, %14, %16], strides : [%22, %c1024_0, %c1], reuse : [seq = false, spat = true, temp = true] : memref<*xf32> to memref<?x?x?xf32, strided<[?, ?, ?], offset: ?>>
              %27 = loom.alloc(%17, %14, %16) on @L1 : memref<?x?x?xf32>
              loom.copy %26, %27, interconnect : [@horizontal_links], broadcast : [1, 8] : memref<?x?x?xf32, strided<[?, ?, ?], offset: ?>> to memref<?x?x?xf32>
              %28 = scf.for %arg13 = %c0 to %17 step %c1 iter_args(%arg14 = %19) -> (tensor<?x?xf32>) {
                %32 = arith.muli %arg13, %16 : index
                %33 = arith.muli %20, %c1024_1 : index
                %subview = memref.subview %27[%arg13, 0, 0] [1, %14, %16] [1, 1, 1] : memref<?x?x?xf32> to memref<?x?xf32, strided<[?, 1], offset: ?>>
                %34 = bufferization.to_tensor %subview restrict writable : memref<?x?xf32, strided<[?, 1], offset: ?>> to tensor<?x?xf32>
                %35 = arith.muli %32, %c1024_1 : index
                %36 = arith.addi %35, %21 : index
                %37 = loom.reinterpret_cast %arg1 to offset : [%36], sizes : [%16, %15], strides : [%c1024, %c1], reuse : [seq = false, spat = true, temp = true] : memref<*xf32> to memref<?x?xf32, strided<[?, ?], offset: ?>>
                %38 = loom.alloc(%16, %15) on @L1 : memref<?x?xf32>
                loom.copy %37, %38, interconnect : [], broadcast : [1, 1] : memref<?x?xf32, strided<[?, ?], offset: ?>> to memref<?x?xf32>
                %39 = bufferization.to_tensor %38 restrict writable : memref<?x?xf32> to tensor<?x?xf32>
                %40 = linalg.matmul ins(%34, %39 : tensor<?x?xf32>, tensor<?x?xf32>) outs(%19 : tensor<?x?xf32>) -> tensor<?x?xf32>
                %41 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%arg14, %40 : tensor<?x?xf32>, tensor<?x?xf32>) outs(%arg14 : tensor<?x?xf32>) {
                ^bb0(%in: f32, %in_3: f32, %out: f32):
                  %42 = arith.addf %in, %in_3 : f32
                  linalg.yield %42 : f32
                } -> tensor<?x?xf32>
                scf.yield %41 : tensor<?x?xf32>
              }
              %29 = arith.muli %20, %c1024_1 : index
              %30 = arith.addi %29, %21 : index
              %31 = loom.reinterpret_cast %arg2 to offset : [%30], sizes : [%14, %15], strides : [%c1024, %c1], reuse : [seq = false, spat = false, temp = false] : memref<*xf32> to memref<?x?xf32, strided<[?, ?], offset: ?>>
              bufferization.materialize_in_destination %28 in writable %31 : (tensor<?x?xf32>, memref<?x?xf32, strided<[?, ?], offset: ?>>) -> ()
            }
          }
        } {loom.mapped_to = @y}
      } {loom.mapped_to = @x}
      return
    }
  }
  module attributes {loom.pass_name = "EnumerateCopyBroadcast"} {
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
      %15 = loom.expression(%13, %12) {coeffs = [1, 1], logic = "mul"} : index
      %16 = loom.expression(%15, %14) {coeffs = [1, 1], logic = "mul"} : index
      loom.polynomial_constraint(%12, %13, %14, %15, %16) {monomials = [{coeff = -1 : i64, vars = [4]}], upper_bound = -262144 : i64}
      %17 = loom.expression(%12, %13) {coeffs = [1, 1], logic = "add"} : index
      %18 = loom.expression(%14, %17) {coeffs = [1, 1], logic = "mul"} : index
      loom.polynomial_constraint(%12, %14, %13, %17, %18) {monomials = [{coeff = 1 : i64, vars = [4]}], upper_bound = 374784 : i64}
      loom.polynomial_constraint(%12) {monomials = [{coeff = 1 : i64, vars = [0]}], upper_bound = 128 : i64}
      loom.polynomial_constraint(%13) {monomials = [{coeff = 1 : i64, vars = [0]}], upper_bound = 128 : i64}
      %19 = loom.expression(%13, %14) {coeffs = [1, 1], logic = "mul"} : index
      loom.polynomial_constraint(%12, %14, %13, %19) {monomials = [{coeff = 1024 : i64, vars = [0]}, {coeff = 1 : i64, vars = [3]}], upper_bound = 374784 : i64}
      %20 = loom.expression(%14, %12) {coeffs = [1, 1], logic = "mul"} : index
      %21 = loom.expression(%13, %14) {coeffs = [1, 1], logic = "mul"} : index
      %22 = loom.expression(%13, %12) {coeffs = [1, 1], logic = "mul"} : index
      %23 = loom.expression(%14, %22) {coeffs = [1, 1], logic = "mul"} : index
      loom.polynomial_constraint(%12, %13, %14, %20, %21, %22, %23) {monomials = [{coeff = 256 : i64, vars = [3]}, {coeff = 256 : i64, vars = [4]}, {coeff = -15 : i64, vars = [6]}], upper_bound = 0 : i64}
    }
    func.func @matmul_kernel__d0i0_d1i1__f10__hoist_block_0__h_v(%arg0: memref<*xf32> {tt.divisibility = 16 : i32}, %arg1: memref<*xf32> {tt.divisibility = 16 : i32}, %arg2: memref<*xf32> {tt.divisibility = 16 : i32}, %arg3: index, %arg4: index, %arg5: index, %arg6: index, %arg7: index, %arg8: index) {
      affine.parallel (%arg9) = (0) to (8) {
        affine.parallel (%arg10) = (0) to (8) {
          affine.for %arg11 = 0 to affine_map<()[s0, s1] -> (s1 ceildiv 8)>()[%arg3, %arg4] {
            affine.for %arg12 = 0 to affine_map<()[s0, s1] -> (s0 ceildiv 8)>()[%arg3, %arg4] {
              %12 = affine.apply affine_map<(d0, d1) -> (d0 * 8 + d1)>(%arg10, %arg11)
              %13 = affine.apply affine_map<(d0, d1) -> (d0 * 8 + d1)>(%arg9, %arg12)
              %c1024 = arith.constant 1024 : index
              %c1024_0 = arith.constant 1024 : index
              %c1 = arith.constant 1 : index
              %14 = loom.get_symbolic_block_size @constraints::@BM : index
              %15 = loom.get_symbolic_block_size @constraints::@BN : index
              %16 = loom.get_symbolic_block_size @constraints::@BK : index
              %17 = arith.ceildivsi %c1024_0, %16 : index
              %cst = arith.constant 0.000000e+00 : f32
              %c1024_1 = arith.constant 1024 : index
              %c0 = arith.constant 0 : index
              %18 = tensor.empty(%14, %15) : tensor<?x?xf32>
              %19 = linalg.fill ins(%cst : f32) outs(%18 : tensor<?x?xf32>) -> tensor<?x?xf32>
              %20 = arith.muli %13, %14 : index
              %21 = arith.muli %12, %15 : index
              %22 = arith.muli %14, %c1024_0 : index
              %c0_2 = arith.constant 0 : index
              %23 = arith.muli %20, %c1024_1 : index
              %24 = arith.muli %c0_2, %16 : index
              %25 = arith.addi %23, %24 : index
              %26 = loom.reinterpret_cast %arg0 to offset : [%25], sizes : [%17, %14, %16], strides : [%22, %c1024_0, %c1], reuse : [seq = false, spat = true, temp = true] : memref<*xf32> to memref<?x?x?xf32, strided<[?, ?, ?], offset: ?>>
              %27 = loom.alloc(%17, %14, %16) on @L1 : memref<?x?x?xf32>
              loom.copy %26, %27, interconnect : [@horizontal_links], broadcast : [1, 8] : memref<?x?x?xf32, strided<[?, ?, ?], offset: ?>> to memref<?x?x?xf32>
              %28 = scf.for %arg13 = %c0 to %17 step %c1 iter_args(%arg14 = %19) -> (tensor<?x?xf32>) {
                %32 = arith.muli %arg13, %16 : index
                %33 = arith.muli %20, %c1024_1 : index
                %subview = memref.subview %27[%arg13, 0, 0] [1, %14, %16] [1, 1, 1] : memref<?x?x?xf32> to memref<?x?xf32, strided<[?, 1], offset: ?>>
                %34 = bufferization.to_tensor %subview restrict writable : memref<?x?xf32, strided<[?, 1], offset: ?>> to tensor<?x?xf32>
                %35 = arith.muli %32, %c1024_1 : index
                %36 = arith.addi %35, %21 : index
                %37 = loom.reinterpret_cast %arg1 to offset : [%36], sizes : [%16, %15], strides : [%c1024, %c1], reuse : [seq = false, spat = true, temp = true] : memref<*xf32> to memref<?x?xf32, strided<[?, ?], offset: ?>>
                %38 = loom.alloc(%16, %15) on @L1 : memref<?x?xf32>
                loom.copy %37, %38, interconnect : [@vertical_links], broadcast : [8, 1] : memref<?x?xf32, strided<[?, ?], offset: ?>> to memref<?x?xf32>
                %39 = bufferization.to_tensor %38 restrict writable : memref<?x?xf32> to tensor<?x?xf32>
                %40 = linalg.matmul ins(%34, %39 : tensor<?x?xf32>, tensor<?x?xf32>) outs(%19 : tensor<?x?xf32>) -> tensor<?x?xf32>
                %41 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%arg14, %40 : tensor<?x?xf32>, tensor<?x?xf32>) outs(%arg14 : tensor<?x?xf32>) {
                ^bb0(%in: f32, %in_3: f32, %out: f32):
                  %42 = arith.addf %in, %in_3 : f32
                  linalg.yield %42 : f32
                } -> tensor<?x?xf32>
                scf.yield %41 : tensor<?x?xf32>
              }
              %29 = arith.muli %20, %c1024_1 : index
              %30 = arith.addi %29, %21 : index
              %31 = loom.reinterpret_cast %arg2 to offset : [%30], sizes : [%14, %15], strides : [%c1024, %c1], reuse : [seq = false, spat = false, temp = false] : memref<*xf32> to memref<?x?xf32, strided<[?, ?], offset: ?>>
              bufferization.materialize_in_destination %28 in writable %31 : (tensor<?x?xf32>, memref<?x?xf32, strided<[?, ?], offset: ?>>) -> ()
            }
          }
        } {loom.mapped_to = @y}
      } {loom.mapped_to = @x}
      return
    }
  }
  module attributes {loom.pass_name = "EnumerateCopyBroadcast"} {
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
      %15 = loom.expression(%13, %12) {coeffs = [1, 1], logic = "mul"} : index
      %16 = loom.expression(%15, %14) {coeffs = [1, 1], logic = "mul"} : index
      loom.polynomial_constraint(%12, %13, %14, %15, %16) {monomials = [{coeff = -1 : i64, vars = [4]}], upper_bound = -262144 : i64}
      %17 = loom.expression(%12, %13) {coeffs = [1, 1], logic = "add"} : index
      %18 = loom.expression(%14, %17) {coeffs = [1, 1], logic = "mul"} : index
      loom.polynomial_constraint(%12, %14, %13, %17, %18) {monomials = [{coeff = 1 : i64, vars = [4]}], upper_bound = 374784 : i64}
      loom.polynomial_constraint(%12) {monomials = [{coeff = 1 : i64, vars = [0]}], upper_bound = 128 : i64}
      loom.polynomial_constraint(%13) {monomials = [{coeff = 1 : i64, vars = [0]}], upper_bound = 128 : i64}
      %19 = loom.expression(%14, %12) {coeffs = [1, 1], logic = "mul"} : index
      loom.polynomial_constraint(%13, %12, %14, %19) {monomials = [{coeff = 1024 : i64, vars = [0]}, {coeff = 1 : i64, vars = [3]}], upper_bound = 374784 : i64}
      %20 = loom.expression(%14, %12) {coeffs = [1, 1], logic = "mul"} : index
      %21 = loom.expression(%13, %14) {coeffs = [1, 1], logic = "mul"} : index
      %22 = loom.expression(%13, %12) {coeffs = [1, 1], logic = "mul"} : index
      %23 = loom.expression(%14, %22) {coeffs = [1, 1], logic = "mul"} : index
      loom.polynomial_constraint(%12, %13, %14, %20, %21, %22, %23) {monomials = [{coeff = 64 : i64, vars = [3]}, {coeff = 64 : i64, vars = [4]}, {coeff = -1 : i64, vars = [6]}], upper_bound = 0 : i64}
    }
    func.func @matmul_kernel__d0i0_d1i1__f10__hoist_block_1__d_d(%arg0: memref<*xf32> {tt.divisibility = 16 : i32}, %arg1: memref<*xf32> {tt.divisibility = 16 : i32}, %arg2: memref<*xf32> {tt.divisibility = 16 : i32}, %arg3: index, %arg4: index, %arg5: index, %arg6: index, %arg7: index, %arg8: index) {
      affine.parallel (%arg9) = (0) to (8) {
        affine.parallel (%arg10) = (0) to (8) {
          affine.for %arg11 = 0 to affine_map<()[s0, s1] -> (s1 ceildiv 8)>()[%arg3, %arg4] {
            affine.for %arg12 = 0 to affine_map<()[s0, s1] -> (s0 ceildiv 8)>()[%arg3, %arg4] {
              %12 = affine.apply affine_map<(d0, d1) -> (d0 * 8 + d1)>(%arg10, %arg11)
              %13 = affine.apply affine_map<(d0, d1) -> (d0 * 8 + d1)>(%arg9, %arg12)
              %c1024 = arith.constant 1024 : index
              %c1024_0 = arith.constant 1024 : index
              %c1 = arith.constant 1 : index
              %14 = loom.get_symbolic_block_size @constraints::@BM : index
              %15 = loom.get_symbolic_block_size @constraints::@BN : index
              %16 = loom.get_symbolic_block_size @constraints::@BK : index
              %17 = arith.ceildivsi %c1024_0, %16 : index
              %cst = arith.constant 0.000000e+00 : f32
              %c1024_1 = arith.constant 1024 : index
              %c0 = arith.constant 0 : index
              %18 = tensor.empty(%14, %15) : tensor<?x?xf32>
              %19 = linalg.fill ins(%cst : f32) outs(%18 : tensor<?x?xf32>) -> tensor<?x?xf32>
              %20 = arith.muli %13, %14 : index
              %21 = arith.muli %12, %15 : index
              %22 = arith.muli %16, %c1024 : index
              %c0_2 = arith.constant 0 : index
              %23 = arith.muli %c0_2, %16 : index
              %24 = arith.muli %23, %c1024_1 : index
              %25 = arith.addi %24, %21 : index
              %26 = loom.reinterpret_cast %arg1 to offset : [%25], sizes : [%17, %16, %15], strides : [%22, %c1024, %c1], reuse : [seq = false, spat = true, temp = true] : memref<*xf32> to memref<?x?x?xf32, strided<[?, ?, ?], offset: ?>>
              %27 = loom.alloc(%17, %16, %15) on @L1 : memref<?x?x?xf32>
              loom.copy %26, %27, interconnect : [], broadcast : [1, 1] : memref<?x?x?xf32, strided<[?, ?, ?], offset: ?>> to memref<?x?x?xf32>
              %28 = scf.for %arg13 = %c0 to %17 step %c1 iter_args(%arg14 = %19) -> (tensor<?x?xf32>) {
                %32 = arith.muli %arg13, %16 : index
                %33 = arith.muli %20, %c1024_1 : index
                %34 = arith.addi %33, %32 : index
                %35 = loom.reinterpret_cast %arg0 to offset : [%34], sizes : [%14, %16], strides : [%c1024_0, %c1], reuse : [seq = false, spat = true, temp = true] : memref<*xf32> to memref<?x?xf32, strided<[?, ?], offset: ?>>
                %36 = loom.alloc(%14, %16) on @L1 : memref<?x?xf32>
                loom.copy %35, %36, interconnect : [], broadcast : [1, 1] : memref<?x?xf32, strided<[?, ?], offset: ?>> to memref<?x?xf32>
                %37 = bufferization.to_tensor %36 restrict writable : memref<?x?xf32> to tensor<?x?xf32>
                %38 = arith.muli %32, %c1024_1 : index
                %subview = memref.subview %27[%arg13, 0, 0] [1, %16, %15] [1, 1, 1] : memref<?x?x?xf32> to memref<?x?xf32, strided<[?, 1], offset: ?>>
                %39 = bufferization.to_tensor %subview restrict writable : memref<?x?xf32, strided<[?, 1], offset: ?>> to tensor<?x?xf32>
                %40 = linalg.matmul ins(%37, %39 : tensor<?x?xf32>, tensor<?x?xf32>) outs(%19 : tensor<?x?xf32>) -> tensor<?x?xf32>
                %41 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%arg14, %40 : tensor<?x?xf32>, tensor<?x?xf32>) outs(%arg14 : tensor<?x?xf32>) {
                ^bb0(%in: f32, %in_3: f32, %out: f32):
                  %42 = arith.addf %in, %in_3 : f32
                  linalg.yield %42 : f32
                } -> tensor<?x?xf32>
                scf.yield %41 : tensor<?x?xf32>
              }
              %29 = arith.muli %20, %c1024_1 : index
              %30 = arith.addi %29, %21 : index
              %31 = loom.reinterpret_cast %arg2 to offset : [%30], sizes : [%14, %15], strides : [%c1024, %c1], reuse : [seq = false, spat = false, temp = false] : memref<*xf32> to memref<?x?xf32, strided<[?, ?], offset: ?>>
              bufferization.materialize_in_destination %28 in writable %31 : (tensor<?x?xf32>, memref<?x?xf32, strided<[?, ?], offset: ?>>) -> ()
            }
          }
        } {loom.mapped_to = @y}
      } {loom.mapped_to = @x}
      return
    }
  }
  module attributes {loom.pass_name = "EnumerateCopyBroadcast"} {
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
      %15 = loom.expression(%13, %12) {coeffs = [1, 1], logic = "mul"} : index
      %16 = loom.expression(%15, %14) {coeffs = [1, 1], logic = "mul"} : index
      loom.polynomial_constraint(%12, %13, %14, %15, %16) {monomials = [{coeff = -1 : i64, vars = [4]}], upper_bound = -262144 : i64}
      %17 = loom.expression(%12, %13) {coeffs = [1, 1], logic = "add"} : index
      %18 = loom.expression(%14, %17) {coeffs = [1, 1], logic = "mul"} : index
      loom.polynomial_constraint(%12, %14, %13, %17, %18) {monomials = [{coeff = 1 : i64, vars = [4]}], upper_bound = 374784 : i64}
      loom.polynomial_constraint(%12) {monomials = [{coeff = 1 : i64, vars = [0]}], upper_bound = 128 : i64}
      loom.polynomial_constraint(%13) {monomials = [{coeff = 1 : i64, vars = [0]}], upper_bound = 128 : i64}
      %19 = loom.expression(%14, %12) {coeffs = [1, 1], logic = "mul"} : index
      loom.polynomial_constraint(%13, %12, %14, %19) {monomials = [{coeff = 1024 : i64, vars = [0]}, {coeff = 1 : i64, vars = [3]}], upper_bound = 374784 : i64}
      %20 = loom.expression(%14, %12) {coeffs = [1, 1], logic = "mul"} : index
      %21 = loom.expression(%13, %14) {coeffs = [1, 1], logic = "mul"} : index
      %22 = loom.expression(%13, %12) {coeffs = [1, 1], logic = "mul"} : index
      %23 = loom.expression(%14, %22) {coeffs = [1, 1], logic = "mul"} : index
      loom.polynomial_constraint(%12, %13, %14, %20, %21, %22, %23) {monomials = [{coeff = 960 : i64, vars = [3]}, {coeff = 256 : i64, vars = [4]}, {coeff = -15 : i64, vars = [6]}], upper_bound = 0 : i64}
    }
    func.func @matmul_kernel__d0i0_d1i1__f10__hoist_block_1__d_h(%arg0: memref<*xf32> {tt.divisibility = 16 : i32}, %arg1: memref<*xf32> {tt.divisibility = 16 : i32}, %arg2: memref<*xf32> {tt.divisibility = 16 : i32}, %arg3: index, %arg4: index, %arg5: index, %arg6: index, %arg7: index, %arg8: index) {
      affine.parallel (%arg9) = (0) to (8) {
        affine.parallel (%arg10) = (0) to (8) {
          affine.for %arg11 = 0 to affine_map<()[s0, s1] -> (s1 ceildiv 8)>()[%arg3, %arg4] {
            affine.for %arg12 = 0 to affine_map<()[s0, s1] -> (s0 ceildiv 8)>()[%arg3, %arg4] {
              %12 = affine.apply affine_map<(d0, d1) -> (d0 * 8 + d1)>(%arg10, %arg11)
              %13 = affine.apply affine_map<(d0, d1) -> (d0 * 8 + d1)>(%arg9, %arg12)
              %c1024 = arith.constant 1024 : index
              %c1024_0 = arith.constant 1024 : index
              %c1 = arith.constant 1 : index
              %14 = loom.get_symbolic_block_size @constraints::@BM : index
              %15 = loom.get_symbolic_block_size @constraints::@BN : index
              %16 = loom.get_symbolic_block_size @constraints::@BK : index
              %17 = arith.ceildivsi %c1024_0, %16 : index
              %cst = arith.constant 0.000000e+00 : f32
              %c1024_1 = arith.constant 1024 : index
              %c0 = arith.constant 0 : index
              %18 = tensor.empty(%14, %15) : tensor<?x?xf32>
              %19 = linalg.fill ins(%cst : f32) outs(%18 : tensor<?x?xf32>) -> tensor<?x?xf32>
              %20 = arith.muli %13, %14 : index
              %21 = arith.muli %12, %15 : index
              %22 = arith.muli %16, %c1024 : index
              %c0_2 = arith.constant 0 : index
              %23 = arith.muli %c0_2, %16 : index
              %24 = arith.muli %23, %c1024_1 : index
              %25 = arith.addi %24, %21 : index
              %26 = loom.reinterpret_cast %arg1 to offset : [%25], sizes : [%17, %16, %15], strides : [%22, %c1024, %c1], reuse : [seq = false, spat = true, temp = true] : memref<*xf32> to memref<?x?x?xf32, strided<[?, ?, ?], offset: ?>>
              %27 = loom.alloc(%17, %16, %15) on @L1 : memref<?x?x?xf32>
              loom.copy %26, %27, interconnect : [], broadcast : [1, 1] : memref<?x?x?xf32, strided<[?, ?, ?], offset: ?>> to memref<?x?x?xf32>
              %28 = scf.for %arg13 = %c0 to %17 step %c1 iter_args(%arg14 = %19) -> (tensor<?x?xf32>) {
                %32 = arith.muli %arg13, %16 : index
                %33 = arith.muli %20, %c1024_1 : index
                %34 = arith.addi %33, %32 : index
                %35 = loom.reinterpret_cast %arg0 to offset : [%34], sizes : [%14, %16], strides : [%c1024_0, %c1], reuse : [seq = false, spat = true, temp = true] : memref<*xf32> to memref<?x?xf32, strided<[?, ?], offset: ?>>
                %36 = loom.alloc(%14, %16) on @L1 : memref<?x?xf32>
                loom.copy %35, %36, interconnect : [@horizontal_links], broadcast : [1, 8] : memref<?x?xf32, strided<[?, ?], offset: ?>> to memref<?x?xf32>
                %37 = bufferization.to_tensor %36 restrict writable : memref<?x?xf32> to tensor<?x?xf32>
                %38 = arith.muli %32, %c1024_1 : index
                %subview = memref.subview %27[%arg13, 0, 0] [1, %16, %15] [1, 1, 1] : memref<?x?x?xf32> to memref<?x?xf32, strided<[?, 1], offset: ?>>
                %39 = bufferization.to_tensor %subview restrict writable : memref<?x?xf32, strided<[?, 1], offset: ?>> to tensor<?x?xf32>
                %40 = linalg.matmul ins(%37, %39 : tensor<?x?xf32>, tensor<?x?xf32>) outs(%19 : tensor<?x?xf32>) -> tensor<?x?xf32>
                %41 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%arg14, %40 : tensor<?x?xf32>, tensor<?x?xf32>) outs(%arg14 : tensor<?x?xf32>) {
                ^bb0(%in: f32, %in_3: f32, %out: f32):
                  %42 = arith.addf %in, %in_3 : f32
                  linalg.yield %42 : f32
                } -> tensor<?x?xf32>
                scf.yield %41 : tensor<?x?xf32>
              }
              %29 = arith.muli %20, %c1024_1 : index
              %30 = arith.addi %29, %21 : index
              %31 = loom.reinterpret_cast %arg2 to offset : [%30], sizes : [%14, %15], strides : [%c1024, %c1], reuse : [seq = false, spat = false, temp = false] : memref<*xf32> to memref<?x?xf32, strided<[?, ?], offset: ?>>
              bufferization.materialize_in_destination %28 in writable %31 : (tensor<?x?xf32>, memref<?x?xf32, strided<[?, ?], offset: ?>>) -> ()
            }
          }
        } {loom.mapped_to = @y}
      } {loom.mapped_to = @x}
      return
    }
  }
  module attributes {loom.pass_name = "EnumerateCopyBroadcast"} {
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
      %15 = loom.expression(%13, %12) {coeffs = [1, 1], logic = "mul"} : index
      %16 = loom.expression(%15, %14) {coeffs = [1, 1], logic = "mul"} : index
      loom.polynomial_constraint(%12, %13, %14, %15, %16) {monomials = [{coeff = -1 : i64, vars = [4]}], upper_bound = -262144 : i64}
      %17 = loom.expression(%12, %13) {coeffs = [1, 1], logic = "add"} : index
      %18 = loom.expression(%14, %17) {coeffs = [1, 1], logic = "mul"} : index
      loom.polynomial_constraint(%12, %14, %13, %17, %18) {monomials = [{coeff = 1 : i64, vars = [4]}], upper_bound = 374784 : i64}
      loom.polynomial_constraint(%12) {monomials = [{coeff = 1 : i64, vars = [0]}], upper_bound = 128 : i64}
      loom.polynomial_constraint(%13) {monomials = [{coeff = 1 : i64, vars = [0]}], upper_bound = 128 : i64}
      %19 = loom.expression(%14, %12) {coeffs = [1, 1], logic = "mul"} : index
      loom.polynomial_constraint(%13, %12, %14, %19) {monomials = [{coeff = 1024 : i64, vars = [0]}, {coeff = 1 : i64, vars = [3]}], upper_bound = 374784 : i64}
      %20 = loom.expression(%14, %12) {coeffs = [1, 1], logic = "mul"} : index
      %21 = loom.expression(%13, %14) {coeffs = [1, 1], logic = "mul"} : index
      %22 = loom.expression(%13, %12) {coeffs = [1, 1], logic = "mul"} : index
      %23 = loom.expression(%14, %22) {coeffs = [1, 1], logic = "mul"} : index
      loom.polynomial_constraint(%12, %13, %14, %20, %21, %22, %23) {monomials = [{coeff = 256 : i64, vars = [3]}, {coeff = 960 : i64, vars = [4]}, {coeff = -15 : i64, vars = [6]}], upper_bound = 0 : i64}
    }
    func.func @matmul_kernel__d0i0_d1i1__f10__hoist_block_1__v_d(%arg0: memref<*xf32> {tt.divisibility = 16 : i32}, %arg1: memref<*xf32> {tt.divisibility = 16 : i32}, %arg2: memref<*xf32> {tt.divisibility = 16 : i32}, %arg3: index, %arg4: index, %arg5: index, %arg6: index, %arg7: index, %arg8: index) {
      affine.parallel (%arg9) = (0) to (8) {
        affine.parallel (%arg10) = (0) to (8) {
          affine.for %arg11 = 0 to affine_map<()[s0, s1] -> (s1 ceildiv 8)>()[%arg3, %arg4] {
            affine.for %arg12 = 0 to affine_map<()[s0, s1] -> (s0 ceildiv 8)>()[%arg3, %arg4] {
              %12 = affine.apply affine_map<(d0, d1) -> (d0 * 8 + d1)>(%arg10, %arg11)
              %13 = affine.apply affine_map<(d0, d1) -> (d0 * 8 + d1)>(%arg9, %arg12)
              %c1024 = arith.constant 1024 : index
              %c1024_0 = arith.constant 1024 : index
              %c1 = arith.constant 1 : index
              %14 = loom.get_symbolic_block_size @constraints::@BM : index
              %15 = loom.get_symbolic_block_size @constraints::@BN : index
              %16 = loom.get_symbolic_block_size @constraints::@BK : index
              %17 = arith.ceildivsi %c1024_0, %16 : index
              %cst = arith.constant 0.000000e+00 : f32
              %c1024_1 = arith.constant 1024 : index
              %c0 = arith.constant 0 : index
              %18 = tensor.empty(%14, %15) : tensor<?x?xf32>
              %19 = linalg.fill ins(%cst : f32) outs(%18 : tensor<?x?xf32>) -> tensor<?x?xf32>
              %20 = arith.muli %13, %14 : index
              %21 = arith.muli %12, %15 : index
              %22 = arith.muli %16, %c1024 : index
              %c0_2 = arith.constant 0 : index
              %23 = arith.muli %c0_2, %16 : index
              %24 = arith.muli %23, %c1024_1 : index
              %25 = arith.addi %24, %21 : index
              %26 = loom.reinterpret_cast %arg1 to offset : [%25], sizes : [%17, %16, %15], strides : [%22, %c1024, %c1], reuse : [seq = false, spat = true, temp = true] : memref<*xf32> to memref<?x?x?xf32, strided<[?, ?, ?], offset: ?>>
              %27 = loom.alloc(%17, %16, %15) on @L1 : memref<?x?x?xf32>
              loom.copy %26, %27, interconnect : [@vertical_links], broadcast : [8, 1] : memref<?x?x?xf32, strided<[?, ?, ?], offset: ?>> to memref<?x?x?xf32>
              %28 = scf.for %arg13 = %c0 to %17 step %c1 iter_args(%arg14 = %19) -> (tensor<?x?xf32>) {
                %32 = arith.muli %arg13, %16 : index
                %33 = arith.muli %20, %c1024_1 : index
                %34 = arith.addi %33, %32 : index
                %35 = loom.reinterpret_cast %arg0 to offset : [%34], sizes : [%14, %16], strides : [%c1024_0, %c1], reuse : [seq = false, spat = true, temp = true] : memref<*xf32> to memref<?x?xf32, strided<[?, ?], offset: ?>>
                %36 = loom.alloc(%14, %16) on @L1 : memref<?x?xf32>
                loom.copy %35, %36, interconnect : [], broadcast : [1, 1] : memref<?x?xf32, strided<[?, ?], offset: ?>> to memref<?x?xf32>
                %37 = bufferization.to_tensor %36 restrict writable : memref<?x?xf32> to tensor<?x?xf32>
                %38 = arith.muli %32, %c1024_1 : index
                %subview = memref.subview %27[%arg13, 0, 0] [1, %16, %15] [1, 1, 1] : memref<?x?x?xf32> to memref<?x?xf32, strided<[?, 1], offset: ?>>
                %39 = bufferization.to_tensor %subview restrict writable : memref<?x?xf32, strided<[?, 1], offset: ?>> to tensor<?x?xf32>
                %40 = linalg.matmul ins(%37, %39 : tensor<?x?xf32>, tensor<?x?xf32>) outs(%19 : tensor<?x?xf32>) -> tensor<?x?xf32>
                %41 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%arg14, %40 : tensor<?x?xf32>, tensor<?x?xf32>) outs(%arg14 : tensor<?x?xf32>) {
                ^bb0(%in: f32, %in_3: f32, %out: f32):
                  %42 = arith.addf %in, %in_3 : f32
                  linalg.yield %42 : f32
                } -> tensor<?x?xf32>
                scf.yield %41 : tensor<?x?xf32>
              }
              %29 = arith.muli %20, %c1024_1 : index
              %30 = arith.addi %29, %21 : index
              %31 = loom.reinterpret_cast %arg2 to offset : [%30], sizes : [%14, %15], strides : [%c1024, %c1], reuse : [seq = false, spat = false, temp = false] : memref<*xf32> to memref<?x?xf32, strided<[?, ?], offset: ?>>
              bufferization.materialize_in_destination %28 in writable %31 : (tensor<?x?xf32>, memref<?x?xf32, strided<[?, ?], offset: ?>>) -> ()
            }
          }
        } {loom.mapped_to = @y}
      } {loom.mapped_to = @x}
      return
    }
  }
  module attributes {loom.pass_name = "EnumerateCopyBroadcast"} {
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
      %15 = loom.expression(%13, %12) {coeffs = [1, 1], logic = "mul"} : index
      %16 = loom.expression(%15, %14) {coeffs = [1, 1], logic = "mul"} : index
      loom.polynomial_constraint(%12, %13, %14, %15, %16) {monomials = [{coeff = -1 : i64, vars = [4]}], upper_bound = -262144 : i64}
      %17 = loom.expression(%12, %13) {coeffs = [1, 1], logic = "add"} : index
      %18 = loom.expression(%14, %17) {coeffs = [1, 1], logic = "mul"} : index
      loom.polynomial_constraint(%12, %14, %13, %17, %18) {monomials = [{coeff = 1 : i64, vars = [4]}], upper_bound = 374784 : i64}
      loom.polynomial_constraint(%12) {monomials = [{coeff = 1 : i64, vars = [0]}], upper_bound = 128 : i64}
      loom.polynomial_constraint(%13) {monomials = [{coeff = 1 : i64, vars = [0]}], upper_bound = 128 : i64}
      %19 = loom.expression(%14, %12) {coeffs = [1, 1], logic = "mul"} : index
      loom.polynomial_constraint(%13, %12, %14, %19) {monomials = [{coeff = 1024 : i64, vars = [0]}, {coeff = 1 : i64, vars = [3]}], upper_bound = 374784 : i64}
      %20 = loom.expression(%14, %12) {coeffs = [1, 1], logic = "mul"} : index
      %21 = loom.expression(%13, %14) {coeffs = [1, 1], logic = "mul"} : index
      %22 = loom.expression(%13, %12) {coeffs = [1, 1], logic = "mul"} : index
      %23 = loom.expression(%14, %22) {coeffs = [1, 1], logic = "mul"} : index
      loom.polynomial_constraint(%12, %13, %14, %20, %21, %22, %23) {monomials = [{coeff = 256 : i64, vars = [3]}, {coeff = 256 : i64, vars = [4]}, {coeff = -15 : i64, vars = [6]}], upper_bound = 0 : i64}
    }
    func.func @matmul_kernel__d0i0_d1i1__f10__hoist_block_1__v_h(%arg0: memref<*xf32> {tt.divisibility = 16 : i32}, %arg1: memref<*xf32> {tt.divisibility = 16 : i32}, %arg2: memref<*xf32> {tt.divisibility = 16 : i32}, %arg3: index, %arg4: index, %arg5: index, %arg6: index, %arg7: index, %arg8: index) {
      affine.parallel (%arg9) = (0) to (8) {
        affine.parallel (%arg10) = (0) to (8) {
          affine.for %arg11 = 0 to affine_map<()[s0, s1] -> (s1 ceildiv 8)>()[%arg3, %arg4] {
            affine.for %arg12 = 0 to affine_map<()[s0, s1] -> (s0 ceildiv 8)>()[%arg3, %arg4] {
              %12 = affine.apply affine_map<(d0, d1) -> (d0 * 8 + d1)>(%arg10, %arg11)
              %13 = affine.apply affine_map<(d0, d1) -> (d0 * 8 + d1)>(%arg9, %arg12)
              %c1024 = arith.constant 1024 : index
              %c1024_0 = arith.constant 1024 : index
              %c1 = arith.constant 1 : index
              %14 = loom.get_symbolic_block_size @constraints::@BM : index
              %15 = loom.get_symbolic_block_size @constraints::@BN : index
              %16 = loom.get_symbolic_block_size @constraints::@BK : index
              %17 = arith.ceildivsi %c1024_0, %16 : index
              %cst = arith.constant 0.000000e+00 : f32
              %c1024_1 = arith.constant 1024 : index
              %c0 = arith.constant 0 : index
              %18 = tensor.empty(%14, %15) : tensor<?x?xf32>
              %19 = linalg.fill ins(%cst : f32) outs(%18 : tensor<?x?xf32>) -> tensor<?x?xf32>
              %20 = arith.muli %13, %14 : index
              %21 = arith.muli %12, %15 : index
              %22 = arith.muli %16, %c1024 : index
              %c0_2 = arith.constant 0 : index
              %23 = arith.muli %c0_2, %16 : index
              %24 = arith.muli %23, %c1024_1 : index
              %25 = arith.addi %24, %21 : index
              %26 = loom.reinterpret_cast %arg1 to offset : [%25], sizes : [%17, %16, %15], strides : [%22, %c1024, %c1], reuse : [seq = false, spat = true, temp = true] : memref<*xf32> to memref<?x?x?xf32, strided<[?, ?, ?], offset: ?>>
              %27 = loom.alloc(%17, %16, %15) on @L1 : memref<?x?x?xf32>
              loom.copy %26, %27, interconnect : [@vertical_links], broadcast : [8, 1] : memref<?x?x?xf32, strided<[?, ?, ?], offset: ?>> to memref<?x?x?xf32>
              %28 = scf.for %arg13 = %c0 to %17 step %c1 iter_args(%arg14 = %19) -> (tensor<?x?xf32>) {
                %32 = arith.muli %arg13, %16 : index
                %33 = arith.muli %20, %c1024_1 : index
                %34 = arith.addi %33, %32 : index
                %35 = loom.reinterpret_cast %arg0 to offset : [%34], sizes : [%14, %16], strides : [%c1024_0, %c1], reuse : [seq = false, spat = true, temp = true] : memref<*xf32> to memref<?x?xf32, strided<[?, ?], offset: ?>>
                %36 = loom.alloc(%14, %16) on @L1 : memref<?x?xf32>
                loom.copy %35, %36, interconnect : [@horizontal_links], broadcast : [1, 8] : memref<?x?xf32, strided<[?, ?], offset: ?>> to memref<?x?xf32>
                %37 = bufferization.to_tensor %36 restrict writable : memref<?x?xf32> to tensor<?x?xf32>
                %38 = arith.muli %32, %c1024_1 : index
                %subview = memref.subview %27[%arg13, 0, 0] [1, %16, %15] [1, 1, 1] : memref<?x?x?xf32> to memref<?x?xf32, strided<[?, 1], offset: ?>>
                %39 = bufferization.to_tensor %subview restrict writable : memref<?x?xf32, strided<[?, 1], offset: ?>> to tensor<?x?xf32>
                %40 = linalg.matmul ins(%37, %39 : tensor<?x?xf32>, tensor<?x?xf32>) outs(%19 : tensor<?x?xf32>) -> tensor<?x?xf32>
                %41 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%arg14, %40 : tensor<?x?xf32>, tensor<?x?xf32>) outs(%arg14 : tensor<?x?xf32>) {
                ^bb0(%in: f32, %in_3: f32, %out: f32):
                  %42 = arith.addf %in, %in_3 : f32
                  linalg.yield %42 : f32
                } -> tensor<?x?xf32>
                scf.yield %41 : tensor<?x?xf32>
              }
              %29 = arith.muli %20, %c1024_1 : index
              %30 = arith.addi %29, %21 : index
              %31 = loom.reinterpret_cast %arg2 to offset : [%30], sizes : [%14, %15], strides : [%c1024, %c1], reuse : [seq = false, spat = false, temp = false] : memref<*xf32> to memref<?x?xf32, strided<[?, ?], offset: ?>>
              bufferization.materialize_in_destination %28 in writable %31 : (tensor<?x?xf32>, memref<?x?xf32, strided<[?, ?], offset: ?>>) -> ()
            }
          }
        } {loom.mapped_to = @y}
      } {loom.mapped_to = @x}
      return
    }
  }
  module attributes {loom.pass_name = "EnumerateCopyBroadcast"} {
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
      %15 = loom.expression(%13, %12) {coeffs = [1, 1], logic = "mul"} : index
      %16 = loom.expression(%15, %14) {coeffs = [1, 1], logic = "mul"} : index
      loom.polynomial_constraint(%12, %13, %14, %15, %16) {monomials = [{coeff = -1 : i64, vars = [4]}], upper_bound = -262144 : i64}
      %17 = loom.expression(%12, %13) {coeffs = [1, 1], logic = "add"} : index
      %18 = loom.expression(%17, %14) {coeffs = [1, 1], logic = "mul"} : index
      loom.polynomial_constraint(%12, %14, %13, %17, %18) {monomials = [{coeff = 1 : i64, vars = [4]}], upper_bound = 374784 : i64}
      loom.polynomial_constraint(%12) {monomials = [{coeff = 1 : i64, vars = [0]}], upper_bound = 128 : i64}
      loom.polynomial_constraint(%13) {monomials = [{coeff = 1 : i64, vars = [0]}], upper_bound = 128 : i64}
      %19 = loom.expression(%12, %14) {coeffs = [1, 1], logic = "mul"} : index
      %20 = loom.expression(%13, %14) {coeffs = [1, 1], logic = "mul"} : index
      %21 = loom.expression(%13, %12) {coeffs = [1, 1], logic = "mul"} : index
      %22 = loom.expression(%14, %21) {coeffs = [1, 1], logic = "mul"} : index
      loom.polynomial_constraint(%12, %13, %14, %19, %20, %21, %22) {monomials = [{coeff = 64 : i64, vars = [3]}, {coeff = 64 : i64, vars = [4]}, {coeff = -1 : i64, vars = [6]}], upper_bound = 0 : i64}
    }
    func.func @matmul_kernel__d1i0_d0i1__f01__d_d(%arg0: memref<*xf32> {tt.divisibility = 16 : i32}, %arg1: memref<*xf32> {tt.divisibility = 16 : i32}, %arg2: memref<*xf32> {tt.divisibility = 16 : i32}, %arg3: index, %arg4: index, %arg5: index, %arg6: index, %arg7: index, %arg8: index) {
      affine.parallel (%arg9) = (0) to (8) {
        affine.parallel (%arg10) = (0) to (8) {
          affine.for %arg11 = 0 to affine_map<()[s0, s1] -> (s0 ceildiv 8)>()[%arg3, %arg4] {
            affine.for %arg12 = 0 to affine_map<()[s0, s1] -> (s1 ceildiv 8)>()[%arg3, %arg4] {
              %12 = affine.apply affine_map<(d0, d1) -> (d0 * 8 + d1)>(%arg10, %arg12)
              %13 = affine.apply affine_map<(d0, d1) -> (d0 * 8 + d1)>(%arg9, %arg11)
              %c1024 = arith.constant 1024 : index
              %c1024_0 = arith.constant 1024 : index
              %c1 = arith.constant 1 : index
              %14 = loom.get_symbolic_block_size @constraints::@BM : index
              %15 = loom.get_symbolic_block_size @constraints::@BN : index
              %16 = loom.get_symbolic_block_size @constraints::@BK : index
              %17 = arith.ceildivsi %c1024_0, %16 : index
              %cst = arith.constant 0.000000e+00 : f32
              %c1024_1 = arith.constant 1024 : index
              %c0 = arith.constant 0 : index
              %18 = tensor.empty(%14, %15) : tensor<?x?xf32>
              %19 = linalg.fill ins(%cst : f32) outs(%18 : tensor<?x?xf32>) -> tensor<?x?xf32>
              %20 = arith.muli %13, %14 : index
              %21 = arith.muli %12, %15 : index
              %22 = scf.for %arg13 = %c0 to %17 step %c1 iter_args(%arg14 = %19) -> (tensor<?x?xf32>) {
                %26 = arith.muli %arg13, %16 : index
                %27 = arith.muli %20, %c1024_1 : index
                %28 = arith.addi %27, %26 : index
                %29 = loom.reinterpret_cast %arg0 to offset : [%28], sizes : [%14, %16], strides : [%c1024_0, %c1], reuse : [seq = false, spat = true, temp = true] : memref<*xf32> to memref<?x?xf32, strided<[?, ?], offset: ?>>
                %30 = loom.alloc(%14, %16) on @L1 : memref<?x?xf32>
                loom.copy %29, %30, interconnect : [], broadcast : [1, 1] : memref<?x?xf32, strided<[?, ?], offset: ?>> to memref<?x?xf32>
                %31 = bufferization.to_tensor %30 restrict writable : memref<?x?xf32> to tensor<?x?xf32>
                %32 = arith.muli %26, %c1024_1 : index
                %33 = arith.addi %32, %21 : index
                %34 = loom.reinterpret_cast %arg1 to offset : [%33], sizes : [%16, %15], strides : [%c1024, %c1], reuse : [seq = false, spat = true, temp = true] : memref<*xf32> to memref<?x?xf32, strided<[?, ?], offset: ?>>
                %35 = loom.alloc(%16, %15) on @L1 : memref<?x?xf32>
                loom.copy %34, %35, interconnect : [], broadcast : [1, 1] : memref<?x?xf32, strided<[?, ?], offset: ?>> to memref<?x?xf32>
                %36 = bufferization.to_tensor %35 restrict writable : memref<?x?xf32> to tensor<?x?xf32>
                %37 = linalg.matmul ins(%31, %36 : tensor<?x?xf32>, tensor<?x?xf32>) outs(%19 : tensor<?x?xf32>) -> tensor<?x?xf32>
                %38 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%arg14, %37 : tensor<?x?xf32>, tensor<?x?xf32>) outs(%arg14 : tensor<?x?xf32>) {
                ^bb0(%in: f32, %in_2: f32, %out: f32):
                  %39 = arith.addf %in, %in_2 : f32
                  linalg.yield %39 : f32
                } -> tensor<?x?xf32>
                scf.yield %38 : tensor<?x?xf32>
              }
              %23 = arith.muli %20, %c1024_1 : index
              %24 = arith.addi %23, %21 : index
              %25 = loom.reinterpret_cast %arg2 to offset : [%24], sizes : [%14, %15], strides : [%c1024, %c1], reuse : [seq = false, spat = false, temp = false] : memref<*xf32> to memref<?x?xf32, strided<[?, ?], offset: ?>>
              bufferization.materialize_in_destination %22 in writable %25 : (tensor<?x?xf32>, memref<?x?xf32, strided<[?, ?], offset: ?>>) -> ()
            }
          }
        } {loom.mapped_to = @x}
      } {loom.mapped_to = @y}
      return
    }
  }
  module attributes {loom.pass_name = "EnumerateCopyBroadcast"} {
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
      %15 = loom.expression(%12, %13) {coeffs = [1, 1], logic = "mul"} : index
      %16 = loom.expression(%15, %14) {coeffs = [1, 1], logic = "mul"} : index
      loom.polynomial_constraint(%12, %13, %14, %15, %16) {monomials = [{coeff = -1 : i64, vars = [4]}], upper_bound = -262144 : i64}
      %17 = loom.expression(%12, %13) {coeffs = [1, 1], logic = "add"} : index
      %18 = loom.expression(%14, %17) {coeffs = [1, 1], logic = "mul"} : index
      loom.polynomial_constraint(%12, %14, %13, %17, %18) {monomials = [{coeff = 1 : i64, vars = [4]}], upper_bound = 374784 : i64}
      loom.polynomial_constraint(%12) {monomials = [{coeff = 1 : i64, vars = [0]}], upper_bound = 128 : i64}
      loom.polynomial_constraint(%13) {monomials = [{coeff = 1 : i64, vars = [0]}], upper_bound = 128 : i64}
      %19 = loom.expression(%12, %14) {coeffs = [1, 1], logic = "mul"} : index
      %20 = loom.expression(%14, %13) {coeffs = [1, 1], logic = "mul"} : index
      %21 = loom.expression(%12, %13) {coeffs = [1, 1], logic = "mul"} : index
      %22 = loom.expression(%14, %21) {coeffs = [1, 1], logic = "mul"} : index
      loom.polynomial_constraint(%12, %13, %14, %19, %20, %21, %22) {monomials = [{coeff = 960 : i64, vars = [3]}, {coeff = 256 : i64, vars = [4]}, {coeff = -15 : i64, vars = [6]}], upper_bound = 0 : i64}
    }
    func.func @matmul_kernel__d1i0_d0i1__f01__d_h(%arg0: memref<*xf32> {tt.divisibility = 16 : i32}, %arg1: memref<*xf32> {tt.divisibility = 16 : i32}, %arg2: memref<*xf32> {tt.divisibility = 16 : i32}, %arg3: index, %arg4: index, %arg5: index, %arg6: index, %arg7: index, %arg8: index) {
      affine.parallel (%arg9) = (0) to (8) {
        affine.parallel (%arg10) = (0) to (8) {
          affine.for %arg11 = 0 to affine_map<()[s0, s1] -> (s0 ceildiv 8)>()[%arg3, %arg4] {
            affine.for %arg12 = 0 to affine_map<()[s0, s1] -> (s1 ceildiv 8)>()[%arg3, %arg4] {
              %12 = affine.apply affine_map<(d0, d1) -> (d0 * 8 + d1)>(%arg10, %arg12)
              %13 = affine.apply affine_map<(d0, d1) -> (d0 * 8 + d1)>(%arg9, %arg11)
              %c1024 = arith.constant 1024 : index
              %c1024_0 = arith.constant 1024 : index
              %c1 = arith.constant 1 : index
              %14 = loom.get_symbolic_block_size @constraints::@BM : index
              %15 = loom.get_symbolic_block_size @constraints::@BN : index
              %16 = loom.get_symbolic_block_size @constraints::@BK : index
              %17 = arith.ceildivsi %c1024_0, %16 : index
              %cst = arith.constant 0.000000e+00 : f32
              %c1024_1 = arith.constant 1024 : index
              %c0 = arith.constant 0 : index
              %18 = tensor.empty(%14, %15) : tensor<?x?xf32>
              %19 = linalg.fill ins(%cst : f32) outs(%18 : tensor<?x?xf32>) -> tensor<?x?xf32>
              %20 = arith.muli %13, %14 : index
              %21 = arith.muli %12, %15 : index
              %22 = scf.for %arg13 = %c0 to %17 step %c1 iter_args(%arg14 = %19) -> (tensor<?x?xf32>) {
                %26 = arith.muli %arg13, %16 : index
                %27 = arith.muli %20, %c1024_1 : index
                %28 = arith.addi %27, %26 : index
                %29 = loom.reinterpret_cast %arg0 to offset : [%28], sizes : [%14, %16], strides : [%c1024_0, %c1], reuse : [seq = false, spat = true, temp = true] : memref<*xf32> to memref<?x?xf32, strided<[?, ?], offset: ?>>
                %30 = loom.alloc(%14, %16) on @L1 : memref<?x?xf32>
                loom.copy %29, %30, interconnect : [], broadcast : [1, 1] : memref<?x?xf32, strided<[?, ?], offset: ?>> to memref<?x?xf32>
                %31 = bufferization.to_tensor %30 restrict writable : memref<?x?xf32> to tensor<?x?xf32>
                %32 = arith.muli %26, %c1024_1 : index
                %33 = arith.addi %32, %21 : index
                %34 = loom.reinterpret_cast %arg1 to offset : [%33], sizes : [%16, %15], strides : [%c1024, %c1], reuse : [seq = false, spat = true, temp = true] : memref<*xf32> to memref<?x?xf32, strided<[?, ?], offset: ?>>
                %35 = loom.alloc(%16, %15) on @L1 : memref<?x?xf32>
                loom.copy %34, %35, interconnect : [@horizontal_links], broadcast : [1, 8] : memref<?x?xf32, strided<[?, ?], offset: ?>> to memref<?x?xf32>
                %36 = bufferization.to_tensor %35 restrict writable : memref<?x?xf32> to tensor<?x?xf32>
                %37 = linalg.matmul ins(%31, %36 : tensor<?x?xf32>, tensor<?x?xf32>) outs(%19 : tensor<?x?xf32>) -> tensor<?x?xf32>
                %38 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%arg14, %37 : tensor<?x?xf32>, tensor<?x?xf32>) outs(%arg14 : tensor<?x?xf32>) {
                ^bb0(%in: f32, %in_2: f32, %out: f32):
                  %39 = arith.addf %in, %in_2 : f32
                  linalg.yield %39 : f32
                } -> tensor<?x?xf32>
                scf.yield %38 : tensor<?x?xf32>
              }
              %23 = arith.muli %20, %c1024_1 : index
              %24 = arith.addi %23, %21 : index
              %25 = loom.reinterpret_cast %arg2 to offset : [%24], sizes : [%14, %15], strides : [%c1024, %c1], reuse : [seq = false, spat = false, temp = false] : memref<*xf32> to memref<?x?xf32, strided<[?, ?], offset: ?>>
              bufferization.materialize_in_destination %22 in writable %25 : (tensor<?x?xf32>, memref<?x?xf32, strided<[?, ?], offset: ?>>) -> ()
            }
          }
        } {loom.mapped_to = @x}
      } {loom.mapped_to = @y}
      return
    }
  }
  module attributes {loom.pass_name = "EnumerateCopyBroadcast"} {
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
      %15 = loom.expression(%12, %13) {coeffs = [1, 1], logic = "mul"} : index
      %16 = loom.expression(%15, %14) {coeffs = [1, 1], logic = "mul"} : index
      loom.polynomial_constraint(%12, %13, %14, %15, %16) {monomials = [{coeff = -1 : i64, vars = [4]}], upper_bound = -262144 : i64}
      %17 = loom.expression(%12, %13) {coeffs = [1, 1], logic = "add"} : index
      %18 = loom.expression(%17, %14) {coeffs = [1, 1], logic = "mul"} : index
      loom.polynomial_constraint(%12, %14, %13, %17, %18) {monomials = [{coeff = 1 : i64, vars = [4]}], upper_bound = 374784 : i64}
      loom.polynomial_constraint(%12) {monomials = [{coeff = 1 : i64, vars = [0]}], upper_bound = 128 : i64}
      loom.polynomial_constraint(%13) {monomials = [{coeff = 1 : i64, vars = [0]}], upper_bound = 128 : i64}
      %19 = loom.expression(%12, %14) {coeffs = [1, 1], logic = "mul"} : index
      %20 = loom.expression(%13, %14) {coeffs = [1, 1], logic = "mul"} : index
      %21 = loom.expression(%12, %13) {coeffs = [1, 1], logic = "mul"} : index
      %22 = loom.expression(%14, %21) {coeffs = [1, 1], logic = "mul"} : index
      loom.polynomial_constraint(%12, %13, %14, %19, %20, %21, %22) {monomials = [{coeff = 256 : i64, vars = [3]}, {coeff = 960 : i64, vars = [4]}, {coeff = -15 : i64, vars = [6]}], upper_bound = 0 : i64}
    }
    func.func @matmul_kernel__d1i0_d0i1__f01__v_d(%arg0: memref<*xf32> {tt.divisibility = 16 : i32}, %arg1: memref<*xf32> {tt.divisibility = 16 : i32}, %arg2: memref<*xf32> {tt.divisibility = 16 : i32}, %arg3: index, %arg4: index, %arg5: index, %arg6: index, %arg7: index, %arg8: index) {
      affine.parallel (%arg9) = (0) to (8) {
        affine.parallel (%arg10) = (0) to (8) {
          affine.for %arg11 = 0 to affine_map<()[s0, s1] -> (s0 ceildiv 8)>()[%arg3, %arg4] {
            affine.for %arg12 = 0 to affine_map<()[s0, s1] -> (s1 ceildiv 8)>()[%arg3, %arg4] {
              %12 = affine.apply affine_map<(d0, d1) -> (d0 * 8 + d1)>(%arg10, %arg12)
              %13 = affine.apply affine_map<(d0, d1) -> (d0 * 8 + d1)>(%arg9, %arg11)
              %c1024 = arith.constant 1024 : index
              %c1024_0 = arith.constant 1024 : index
              %c1 = arith.constant 1 : index
              %14 = loom.get_symbolic_block_size @constraints::@BM : index
              %15 = loom.get_symbolic_block_size @constraints::@BN : index
              %16 = loom.get_symbolic_block_size @constraints::@BK : index
              %17 = arith.ceildivsi %c1024_0, %16 : index
              %cst = arith.constant 0.000000e+00 : f32
              %c1024_1 = arith.constant 1024 : index
              %c0 = arith.constant 0 : index
              %18 = tensor.empty(%14, %15) : tensor<?x?xf32>
              %19 = linalg.fill ins(%cst : f32) outs(%18 : tensor<?x?xf32>) -> tensor<?x?xf32>
              %20 = arith.muli %13, %14 : index
              %21 = arith.muli %12, %15 : index
              %22 = scf.for %arg13 = %c0 to %17 step %c1 iter_args(%arg14 = %19) -> (tensor<?x?xf32>) {
                %26 = arith.muli %arg13, %16 : index
                %27 = arith.muli %20, %c1024_1 : index
                %28 = arith.addi %27, %26 : index
                %29 = loom.reinterpret_cast %arg0 to offset : [%28], sizes : [%14, %16], strides : [%c1024_0, %c1], reuse : [seq = false, spat = true, temp = true] : memref<*xf32> to memref<?x?xf32, strided<[?, ?], offset: ?>>
                %30 = loom.alloc(%14, %16) on @L1 : memref<?x?xf32>
                loom.copy %29, %30, interconnect : [@vertical_links], broadcast : [8, 1] : memref<?x?xf32, strided<[?, ?], offset: ?>> to memref<?x?xf32>
                %31 = bufferization.to_tensor %30 restrict writable : memref<?x?xf32> to tensor<?x?xf32>
                %32 = arith.muli %26, %c1024_1 : index
                %33 = arith.addi %32, %21 : index
                %34 = loom.reinterpret_cast %arg1 to offset : [%33], sizes : [%16, %15], strides : [%c1024, %c1], reuse : [seq = false, spat = true, temp = true] : memref<*xf32> to memref<?x?xf32, strided<[?, ?], offset: ?>>
                %35 = loom.alloc(%16, %15) on @L1 : memref<?x?xf32>
                loom.copy %34, %35, interconnect : [], broadcast : [1, 1] : memref<?x?xf32, strided<[?, ?], offset: ?>> to memref<?x?xf32>
                %36 = bufferization.to_tensor %35 restrict writable : memref<?x?xf32> to tensor<?x?xf32>
                %37 = linalg.matmul ins(%31, %36 : tensor<?x?xf32>, tensor<?x?xf32>) outs(%19 : tensor<?x?xf32>) -> tensor<?x?xf32>
                %38 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%arg14, %37 : tensor<?x?xf32>, tensor<?x?xf32>) outs(%arg14 : tensor<?x?xf32>) {
                ^bb0(%in: f32, %in_2: f32, %out: f32):
                  %39 = arith.addf %in, %in_2 : f32
                  linalg.yield %39 : f32
                } -> tensor<?x?xf32>
                scf.yield %38 : tensor<?x?xf32>
              }
              %23 = arith.muli %20, %c1024_1 : index
              %24 = arith.addi %23, %21 : index
              %25 = loom.reinterpret_cast %arg2 to offset : [%24], sizes : [%14, %15], strides : [%c1024, %c1], reuse : [seq = false, spat = false, temp = false] : memref<*xf32> to memref<?x?xf32, strided<[?, ?], offset: ?>>
              bufferization.materialize_in_destination %22 in writable %25 : (tensor<?x?xf32>, memref<?x?xf32, strided<[?, ?], offset: ?>>) -> ()
            }
          }
        } {loom.mapped_to = @x}
      } {loom.mapped_to = @y}
      return
    }
  }
  module attributes {loom.pass_name = "EnumerateCopyBroadcast"} {
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
      %15 = loom.expression(%12, %13) {coeffs = [1, 1], logic = "mul"} : index
      %16 = loom.expression(%15, %14) {coeffs = [1, 1], logic = "mul"} : index
      loom.polynomial_constraint(%12, %13, %14, %15, %16) {monomials = [{coeff = -1 : i64, vars = [4]}], upper_bound = -262144 : i64}
      %17 = loom.expression(%12, %13) {coeffs = [1, 1], logic = "add"} : index
      %18 = loom.expression(%17, %14) {coeffs = [1, 1], logic = "mul"} : index
      loom.polynomial_constraint(%12, %14, %13, %17, %18) {monomials = [{coeff = 1 : i64, vars = [4]}], upper_bound = 374784 : i64}
      loom.polynomial_constraint(%12) {monomials = [{coeff = 1 : i64, vars = [0]}], upper_bound = 128 : i64}
      loom.polynomial_constraint(%13) {monomials = [{coeff = 1 : i64, vars = [0]}], upper_bound = 128 : i64}
      %19 = loom.expression(%12, %14) {coeffs = [1, 1], logic = "mul"} : index
      %20 = loom.expression(%13, %14) {coeffs = [1, 1], logic = "mul"} : index
      %21 = loom.expression(%12, %13) {coeffs = [1, 1], logic = "mul"} : index
      %22 = loom.expression(%14, %21) {coeffs = [1, 1], logic = "mul"} : index
      loom.polynomial_constraint(%12, %13, %14, %19, %20, %21, %22) {monomials = [{coeff = 256 : i64, vars = [3]}, {coeff = 256 : i64, vars = [4]}, {coeff = -15 : i64, vars = [6]}], upper_bound = 0 : i64}
    }
    func.func @matmul_kernel__d1i0_d0i1__f01__v_h(%arg0: memref<*xf32> {tt.divisibility = 16 : i32}, %arg1: memref<*xf32> {tt.divisibility = 16 : i32}, %arg2: memref<*xf32> {tt.divisibility = 16 : i32}, %arg3: index, %arg4: index, %arg5: index, %arg6: index, %arg7: index, %arg8: index) {
      affine.parallel (%arg9) = (0) to (8) {
        affine.parallel (%arg10) = (0) to (8) {
          affine.for %arg11 = 0 to affine_map<()[s0, s1] -> (s0 ceildiv 8)>()[%arg3, %arg4] {
            affine.for %arg12 = 0 to affine_map<()[s0, s1] -> (s1 ceildiv 8)>()[%arg3, %arg4] {
              %12 = affine.apply affine_map<(d0, d1) -> (d0 * 8 + d1)>(%arg10, %arg12)
              %13 = affine.apply affine_map<(d0, d1) -> (d0 * 8 + d1)>(%arg9, %arg11)
              %c1024 = arith.constant 1024 : index
              %c1024_0 = arith.constant 1024 : index
              %c1 = arith.constant 1 : index
              %14 = loom.get_symbolic_block_size @constraints::@BM : index
              %15 = loom.get_symbolic_block_size @constraints::@BN : index
              %16 = loom.get_symbolic_block_size @constraints::@BK : index
              %17 = arith.ceildivsi %c1024_0, %16 : index
              %cst = arith.constant 0.000000e+00 : f32
              %c1024_1 = arith.constant 1024 : index
              %c0 = arith.constant 0 : index
              %18 = tensor.empty(%14, %15) : tensor<?x?xf32>
              %19 = linalg.fill ins(%cst : f32) outs(%18 : tensor<?x?xf32>) -> tensor<?x?xf32>
              %20 = arith.muli %13, %14 : index
              %21 = arith.muli %12, %15 : index
              %22 = scf.for %arg13 = %c0 to %17 step %c1 iter_args(%arg14 = %19) -> (tensor<?x?xf32>) {
                %26 = arith.muli %arg13, %16 : index
                %27 = arith.muli %20, %c1024_1 : index
                %28 = arith.addi %27, %26 : index
                %29 = loom.reinterpret_cast %arg0 to offset : [%28], sizes : [%14, %16], strides : [%c1024_0, %c1], reuse : [seq = false, spat = true, temp = true] : memref<*xf32> to memref<?x?xf32, strided<[?, ?], offset: ?>>
                %30 = loom.alloc(%14, %16) on @L1 : memref<?x?xf32>
                loom.copy %29, %30, interconnect : [@vertical_links], broadcast : [8, 1] : memref<?x?xf32, strided<[?, ?], offset: ?>> to memref<?x?xf32>
                %31 = bufferization.to_tensor %30 restrict writable : memref<?x?xf32> to tensor<?x?xf32>
                %32 = arith.muli %26, %c1024_1 : index
                %33 = arith.addi %32, %21 : index
                %34 = loom.reinterpret_cast %arg1 to offset : [%33], sizes : [%16, %15], strides : [%c1024, %c1], reuse : [seq = false, spat = true, temp = true] : memref<*xf32> to memref<?x?xf32, strided<[?, ?], offset: ?>>
                %35 = loom.alloc(%16, %15) on @L1 : memref<?x?xf32>
                loom.copy %34, %35, interconnect : [@horizontal_links], broadcast : [1, 8] : memref<?x?xf32, strided<[?, ?], offset: ?>> to memref<?x?xf32>
                %36 = bufferization.to_tensor %35 restrict writable : memref<?x?xf32> to tensor<?x?xf32>
                %37 = linalg.matmul ins(%31, %36 : tensor<?x?xf32>, tensor<?x?xf32>) outs(%19 : tensor<?x?xf32>) -> tensor<?x?xf32>
                %38 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%arg14, %37 : tensor<?x?xf32>, tensor<?x?xf32>) outs(%arg14 : tensor<?x?xf32>) {
                ^bb0(%in: f32, %in_2: f32, %out: f32):
                  %39 = arith.addf %in, %in_2 : f32
                  linalg.yield %39 : f32
                } -> tensor<?x?xf32>
                scf.yield %38 : tensor<?x?xf32>
              }
              %23 = arith.muli %20, %c1024_1 : index
              %24 = arith.addi %23, %21 : index
              %25 = loom.reinterpret_cast %arg2 to offset : [%24], sizes : [%14, %15], strides : [%c1024, %c1], reuse : [seq = false, spat = false, temp = false] : memref<*xf32> to memref<?x?xf32, strided<[?, ?], offset: ?>>
              bufferization.materialize_in_destination %22 in writable %25 : (tensor<?x?xf32>, memref<?x?xf32, strided<[?, ?], offset: ?>>) -> ()
            }
          }
        } {loom.mapped_to = @x}
      } {loom.mapped_to = @y}
      return
    }
  }
  module attributes {loom.pass_name = "EnumerateCopyBroadcast"} {
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
      %15 = loom.expression(%12, %13) {coeffs = [1, 1], logic = "mul"} : index
      %16 = loom.expression(%14, %15) {coeffs = [1, 1], logic = "mul"} : index
      loom.polynomial_constraint(%12, %13, %14, %15, %16) {monomials = [{coeff = -1 : i64, vars = [4]}], upper_bound = -262144 : i64}
      %17 = loom.expression(%12, %13) {coeffs = [1, 1], logic = "add"} : index
      %18 = loom.expression(%14, %17) {coeffs = [1, 1], logic = "mul"} : index
      loom.polynomial_constraint(%12, %14, %13, %17, %18) {monomials = [{coeff = 1 : i64, vars = [4]}], upper_bound = 374784 : i64}
      loom.polynomial_constraint(%12) {monomials = [{coeff = 1 : i64, vars = [0]}], upper_bound = 128 : i64}
      loom.polynomial_constraint(%13) {monomials = [{coeff = 1 : i64, vars = [0]}], upper_bound = 128 : i64}
      %19 = loom.expression(%14, %13) {coeffs = [1, 1], logic = "mul"} : index
      loom.polynomial_constraint(%12, %14, %13, %19) {monomials = [{coeff = 1024 : i64, vars = [0]}, {coeff = 1 : i64, vars = [3]}], upper_bound = 374784 : i64}
      %20 = loom.expression(%12, %14) {coeffs = [1, 1], logic = "mul"} : index
      %21 = loom.expression(%14, %13) {coeffs = [1, 1], logic = "mul"} : index
      %22 = loom.expression(%12, %13) {coeffs = [1, 1], logic = "mul"} : index
      %23 = loom.expression(%14, %22) {coeffs = [1, 1], logic = "mul"} : index
      loom.polynomial_constraint(%12, %13, %14, %20, %21, %22, %23) {monomials = [{coeff = 64 : i64, vars = [3]}, {coeff = 64 : i64, vars = [4]}, {coeff = -1 : i64, vars = [6]}], upper_bound = 0 : i64}
    }
    func.func @matmul_kernel__d1i0_d0i1__f01__hoist_block_0__d_d(%arg0: memref<*xf32> {tt.divisibility = 16 : i32}, %arg1: memref<*xf32> {tt.divisibility = 16 : i32}, %arg2: memref<*xf32> {tt.divisibility = 16 : i32}, %arg3: index, %arg4: index, %arg5: index, %arg6: index, %arg7: index, %arg8: index) {
      affine.parallel (%arg9) = (0) to (8) {
        affine.parallel (%arg10) = (0) to (8) {
          affine.for %arg11 = 0 to affine_map<()[s0, s1] -> (s0 ceildiv 8)>()[%arg3, %arg4] {
            affine.for %arg12 = 0 to affine_map<()[s0, s1] -> (s1 ceildiv 8)>()[%arg3, %arg4] {
              %12 = affine.apply affine_map<(d0, d1) -> (d0 * 8 + d1)>(%arg10, %arg12)
              %13 = affine.apply affine_map<(d0, d1) -> (d0 * 8 + d1)>(%arg9, %arg11)
              %c1024 = arith.constant 1024 : index
              %c1024_0 = arith.constant 1024 : index
              %c1 = arith.constant 1 : index
              %14 = loom.get_symbolic_block_size @constraints::@BM : index
              %15 = loom.get_symbolic_block_size @constraints::@BN : index
              %16 = loom.get_symbolic_block_size @constraints::@BK : index
              %17 = arith.ceildivsi %c1024_0, %16 : index
              %cst = arith.constant 0.000000e+00 : f32
              %c1024_1 = arith.constant 1024 : index
              %c0 = arith.constant 0 : index
              %18 = tensor.empty(%14, %15) : tensor<?x?xf32>
              %19 = linalg.fill ins(%cst : f32) outs(%18 : tensor<?x?xf32>) -> tensor<?x?xf32>
              %20 = arith.muli %13, %14 : index
              %21 = arith.muli %12, %15 : index
              %22 = arith.muli %14, %c1024_0 : index
              %c0_2 = arith.constant 0 : index
              %23 = arith.muli %20, %c1024_1 : index
              %24 = arith.muli %c0_2, %16 : index
              %25 = arith.addi %23, %24 : index
              %26 = loom.reinterpret_cast %arg0 to offset : [%25], sizes : [%17, %14, %16], strides : [%22, %c1024_0, %c1], reuse : [seq = false, spat = true, temp = true] : memref<*xf32> to memref<?x?x?xf32, strided<[?, ?, ?], offset: ?>>
              %27 = loom.alloc(%17, %14, %16) on @L1 : memref<?x?x?xf32>
              loom.copy %26, %27, interconnect : [], broadcast : [1, 1] : memref<?x?x?xf32, strided<[?, ?, ?], offset: ?>> to memref<?x?x?xf32>
              %28 = scf.for %arg13 = %c0 to %17 step %c1 iter_args(%arg14 = %19) -> (tensor<?x?xf32>) {
                %32 = arith.muli %arg13, %16 : index
                %33 = arith.muli %20, %c1024_1 : index
                %subview = memref.subview %27[%arg13, 0, 0] [1, %14, %16] [1, 1, 1] : memref<?x?x?xf32> to memref<?x?xf32, strided<[?, 1], offset: ?>>
                %34 = bufferization.to_tensor %subview restrict writable : memref<?x?xf32, strided<[?, 1], offset: ?>> to tensor<?x?xf32>
                %35 = arith.muli %32, %c1024_1 : index
                %36 = arith.addi %35, %21 : index
                %37 = loom.reinterpret_cast %arg1 to offset : [%36], sizes : [%16, %15], strides : [%c1024, %c1], reuse : [seq = false, spat = true, temp = true] : memref<*xf32> to memref<?x?xf32, strided<[?, ?], offset: ?>>
                %38 = loom.alloc(%16, %15) on @L1 : memref<?x?xf32>
                loom.copy %37, %38, interconnect : [], broadcast : [1, 1] : memref<?x?xf32, strided<[?, ?], offset: ?>> to memref<?x?xf32>
                %39 = bufferization.to_tensor %38 restrict writable : memref<?x?xf32> to tensor<?x?xf32>
                %40 = linalg.matmul ins(%34, %39 : tensor<?x?xf32>, tensor<?x?xf32>) outs(%19 : tensor<?x?xf32>) -> tensor<?x?xf32>
                %41 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%arg14, %40 : tensor<?x?xf32>, tensor<?x?xf32>) outs(%arg14 : tensor<?x?xf32>) {
                ^bb0(%in: f32, %in_3: f32, %out: f32):
                  %42 = arith.addf %in, %in_3 : f32
                  linalg.yield %42 : f32
                } -> tensor<?x?xf32>
                scf.yield %41 : tensor<?x?xf32>
              }
              %29 = arith.muli %20, %c1024_1 : index
              %30 = arith.addi %29, %21 : index
              %31 = loom.reinterpret_cast %arg2 to offset : [%30], sizes : [%14, %15], strides : [%c1024, %c1], reuse : [seq = false, spat = false, temp = false] : memref<*xf32> to memref<?x?xf32, strided<[?, ?], offset: ?>>
              bufferization.materialize_in_destination %28 in writable %31 : (tensor<?x?xf32>, memref<?x?xf32, strided<[?, ?], offset: ?>>) -> ()
            }
          }
        } {loom.mapped_to = @x}
      } {loom.mapped_to = @y}
      return
    }
  }
  module attributes {loom.pass_name = "EnumerateCopyBroadcast"} {
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
      %15 = loom.expression(%13, %12) {coeffs = [1, 1], logic = "mul"} : index
      %16 = loom.expression(%15, %14) {coeffs = [1, 1], logic = "mul"} : index
      loom.polynomial_constraint(%12, %13, %14, %15, %16) {monomials = [{coeff = -1 : i64, vars = [4]}], upper_bound = -262144 : i64}
      %17 = loom.expression(%12, %13) {coeffs = [1, 1], logic = "add"} : index
      %18 = loom.expression(%14, %17) {coeffs = [1, 1], logic = "mul"} : index
      loom.polynomial_constraint(%12, %14, %13, %17, %18) {monomials = [{coeff = 1 : i64, vars = [4]}], upper_bound = 374784 : i64}
      loom.polynomial_constraint(%12) {monomials = [{coeff = 1 : i64, vars = [0]}], upper_bound = 128 : i64}
      loom.polynomial_constraint(%13) {monomials = [{coeff = 1 : i64, vars = [0]}], upper_bound = 128 : i64}
      %19 = loom.expression(%13, %14) {coeffs = [1, 1], logic = "mul"} : index
      loom.polynomial_constraint(%12, %14, %13, %19) {monomials = [{coeff = 1024 : i64, vars = [0]}, {coeff = 1 : i64, vars = [3]}], upper_bound = 374784 : i64}
      %20 = loom.expression(%14, %12) {coeffs = [1, 1], logic = "mul"} : index
      %21 = loom.expression(%13, %14) {coeffs = [1, 1], logic = "mul"} : index
      %22 = loom.expression(%13, %12) {coeffs = [1, 1], logic = "mul"} : index
      %23 = loom.expression(%14, %22) {coeffs = [1, 1], logic = "mul"} : index
      loom.polynomial_constraint(%12, %13, %14, %20, %21, %22, %23) {monomials = [{coeff = 960 : i64, vars = [3]}, {coeff = 256 : i64, vars = [4]}, {coeff = -15 : i64, vars = [6]}], upper_bound = 0 : i64}
    }
    func.func @matmul_kernel__d1i0_d0i1__f01__hoist_block_0__d_h(%arg0: memref<*xf32> {tt.divisibility = 16 : i32}, %arg1: memref<*xf32> {tt.divisibility = 16 : i32}, %arg2: memref<*xf32> {tt.divisibility = 16 : i32}, %arg3: index, %arg4: index, %arg5: index, %arg6: index, %arg7: index, %arg8: index) {
      affine.parallel (%arg9) = (0) to (8) {
        affine.parallel (%arg10) = (0) to (8) {
          affine.for %arg11 = 0 to affine_map<()[s0, s1] -> (s0 ceildiv 8)>()[%arg3, %arg4] {
            affine.for %arg12 = 0 to affine_map<()[s0, s1] -> (s1 ceildiv 8)>()[%arg3, %arg4] {
              %12 = affine.apply affine_map<(d0, d1) -> (d0 * 8 + d1)>(%arg10, %arg12)
              %13 = affine.apply affine_map<(d0, d1) -> (d0 * 8 + d1)>(%arg9, %arg11)
              %c1024 = arith.constant 1024 : index
              %c1024_0 = arith.constant 1024 : index
              %c1 = arith.constant 1 : index
              %14 = loom.get_symbolic_block_size @constraints::@BM : index
              %15 = loom.get_symbolic_block_size @constraints::@BN : index
              %16 = loom.get_symbolic_block_size @constraints::@BK : index
              %17 = arith.ceildivsi %c1024_0, %16 : index
              %cst = arith.constant 0.000000e+00 : f32
              %c1024_1 = arith.constant 1024 : index
              %c0 = arith.constant 0 : index
              %18 = tensor.empty(%14, %15) : tensor<?x?xf32>
              %19 = linalg.fill ins(%cst : f32) outs(%18 : tensor<?x?xf32>) -> tensor<?x?xf32>
              %20 = arith.muli %13, %14 : index
              %21 = arith.muli %12, %15 : index
              %22 = arith.muli %14, %c1024_0 : index
              %c0_2 = arith.constant 0 : index
              %23 = arith.muli %20, %c1024_1 : index
              %24 = arith.muli %c0_2, %16 : index
              %25 = arith.addi %23, %24 : index
              %26 = loom.reinterpret_cast %arg0 to offset : [%25], sizes : [%17, %14, %16], strides : [%22, %c1024_0, %c1], reuse : [seq = false, spat = true, temp = true] : memref<*xf32> to memref<?x?x?xf32, strided<[?, ?, ?], offset: ?>>
              %27 = loom.alloc(%17, %14, %16) on @L1 : memref<?x?x?xf32>
              loom.copy %26, %27, interconnect : [], broadcast : [1, 1] : memref<?x?x?xf32, strided<[?, ?, ?], offset: ?>> to memref<?x?x?xf32>
              %28 = scf.for %arg13 = %c0 to %17 step %c1 iter_args(%arg14 = %19) -> (tensor<?x?xf32>) {
                %32 = arith.muli %arg13, %16 : index
                %33 = arith.muli %20, %c1024_1 : index
                %subview = memref.subview %27[%arg13, 0, 0] [1, %14, %16] [1, 1, 1] : memref<?x?x?xf32> to memref<?x?xf32, strided<[?, 1], offset: ?>>
                %34 = bufferization.to_tensor %subview restrict writable : memref<?x?xf32, strided<[?, 1], offset: ?>> to tensor<?x?xf32>
                %35 = arith.muli %32, %c1024_1 : index
                %36 = arith.addi %35, %21 : index
                %37 = loom.reinterpret_cast %arg1 to offset : [%36], sizes : [%16, %15], strides : [%c1024, %c1], reuse : [seq = false, spat = true, temp = true] : memref<*xf32> to memref<?x?xf32, strided<[?, ?], offset: ?>>
                %38 = loom.alloc(%16, %15) on @L1 : memref<?x?xf32>
                loom.copy %37, %38, interconnect : [@horizontal_links], broadcast : [1, 8] : memref<?x?xf32, strided<[?, ?], offset: ?>> to memref<?x?xf32>
                %39 = bufferization.to_tensor %38 restrict writable : memref<?x?xf32> to tensor<?x?xf32>
                %40 = linalg.matmul ins(%34, %39 : tensor<?x?xf32>, tensor<?x?xf32>) outs(%19 : tensor<?x?xf32>) -> tensor<?x?xf32>
                %41 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%arg14, %40 : tensor<?x?xf32>, tensor<?x?xf32>) outs(%arg14 : tensor<?x?xf32>) {
                ^bb0(%in: f32, %in_3: f32, %out: f32):
                  %42 = arith.addf %in, %in_3 : f32
                  linalg.yield %42 : f32
                } -> tensor<?x?xf32>
                scf.yield %41 : tensor<?x?xf32>
              }
              %29 = arith.muli %20, %c1024_1 : index
              %30 = arith.addi %29, %21 : index
              %31 = loom.reinterpret_cast %arg2 to offset : [%30], sizes : [%14, %15], strides : [%c1024, %c1], reuse : [seq = false, spat = false, temp = false] : memref<*xf32> to memref<?x?xf32, strided<[?, ?], offset: ?>>
              bufferization.materialize_in_destination %28 in writable %31 : (tensor<?x?xf32>, memref<?x?xf32, strided<[?, ?], offset: ?>>) -> ()
            }
          }
        } {loom.mapped_to = @x}
      } {loom.mapped_to = @y}
      return
    }
  }
  module attributes {loom.pass_name = "EnumerateCopyBroadcast"} {
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
      %15 = loom.expression(%12, %13) {coeffs = [1, 1], logic = "mul"} : index
      %16 = loom.expression(%15, %14) {coeffs = [1, 1], logic = "mul"} : index
      loom.polynomial_constraint(%12, %13, %14, %15, %16) {monomials = [{coeff = -1 : i64, vars = [4]}], upper_bound = -262144 : i64}
      %17 = loom.expression(%12, %13) {coeffs = [1, 1], logic = "add"} : index
      %18 = loom.expression(%14, %17) {coeffs = [1, 1], logic = "mul"} : index
      loom.polynomial_constraint(%12, %14, %13, %17, %18) {monomials = [{coeff = 1 : i64, vars = [4]}], upper_bound = 374784 : i64}
      loom.polynomial_constraint(%12) {monomials = [{coeff = 1 : i64, vars = [0]}], upper_bound = 128 : i64}
      loom.polynomial_constraint(%13) {monomials = [{coeff = 1 : i64, vars = [0]}], upper_bound = 128 : i64}
      %19 = loom.expression(%14, %13) {coeffs = [1, 1], logic = "mul"} : index
      loom.polynomial_constraint(%12, %14, %13, %19) {monomials = [{coeff = 1024 : i64, vars = [0]}, {coeff = 1 : i64, vars = [3]}], upper_bound = 374784 : i64}
      %20 = loom.expression(%14, %12) {coeffs = [1, 1], logic = "mul"} : index
      %21 = loom.expression(%14, %13) {coeffs = [1, 1], logic = "mul"} : index
      %22 = loom.expression(%12, %13) {coeffs = [1, 1], logic = "mul"} : index
      %23 = loom.expression(%14, %22) {coeffs = [1, 1], logic = "mul"} : index
      loom.polynomial_constraint(%12, %13, %14, %20, %21, %22, %23) {monomials = [{coeff = 256 : i64, vars = [3]}, {coeff = 960 : i64, vars = [4]}, {coeff = -15 : i64, vars = [6]}], upper_bound = 0 : i64}
    }
    func.func @matmul_kernel__d1i0_d0i1__f01__hoist_block_0__v_d(%arg0: memref<*xf32> {tt.divisibility = 16 : i32}, %arg1: memref<*xf32> {tt.divisibility = 16 : i32}, %arg2: memref<*xf32> {tt.divisibility = 16 : i32}, %arg3: index, %arg4: index, %arg5: index, %arg6: index, %arg7: index, %arg8: index) {
      affine.parallel (%arg9) = (0) to (8) {
        affine.parallel (%arg10) = (0) to (8) {
          affine.for %arg11 = 0 to affine_map<()[s0, s1] -> (s0 ceildiv 8)>()[%arg3, %arg4] {
            affine.for %arg12 = 0 to affine_map<()[s0, s1] -> (s1 ceildiv 8)>()[%arg3, %arg4] {
              %12 = affine.apply affine_map<(d0, d1) -> (d0 * 8 + d1)>(%arg10, %arg12)
              %13 = affine.apply affine_map<(d0, d1) -> (d0 * 8 + d1)>(%arg9, %arg11)
              %c1024 = arith.constant 1024 : index
              %c1024_0 = arith.constant 1024 : index
              %c1 = arith.constant 1 : index
              %14 = loom.get_symbolic_block_size @constraints::@BM : index
              %15 = loom.get_symbolic_block_size @constraints::@BN : index
              %16 = loom.get_symbolic_block_size @constraints::@BK : index
              %17 = arith.ceildivsi %c1024_0, %16 : index
              %cst = arith.constant 0.000000e+00 : f32
              %c1024_1 = arith.constant 1024 : index
              %c0 = arith.constant 0 : index
              %18 = tensor.empty(%14, %15) : tensor<?x?xf32>
              %19 = linalg.fill ins(%cst : f32) outs(%18 : tensor<?x?xf32>) -> tensor<?x?xf32>
              %20 = arith.muli %13, %14 : index
              %21 = arith.muli %12, %15 : index
              %22 = arith.muli %14, %c1024_0 : index
              %c0_2 = arith.constant 0 : index
              %23 = arith.muli %20, %c1024_1 : index
              %24 = arith.muli %c0_2, %16 : index
              %25 = arith.addi %23, %24 : index
              %26 = loom.reinterpret_cast %arg0 to offset : [%25], sizes : [%17, %14, %16], strides : [%22, %c1024_0, %c1], reuse : [seq = false, spat = true, temp = true] : memref<*xf32> to memref<?x?x?xf32, strided<[?, ?, ?], offset: ?>>
              %27 = loom.alloc(%17, %14, %16) on @L1 : memref<?x?x?xf32>
              loom.copy %26, %27, interconnect : [@vertical_links], broadcast : [8, 1] : memref<?x?x?xf32, strided<[?, ?, ?], offset: ?>> to memref<?x?x?xf32>
              %28 = scf.for %arg13 = %c0 to %17 step %c1 iter_args(%arg14 = %19) -> (tensor<?x?xf32>) {
                %32 = arith.muli %arg13, %16 : index
                %33 = arith.muli %20, %c1024_1 : index
                %subview = memref.subview %27[%arg13, 0, 0] [1, %14, %16] [1, 1, 1] : memref<?x?x?xf32> to memref<?x?xf32, strided<[?, 1], offset: ?>>
                %34 = bufferization.to_tensor %subview restrict writable : memref<?x?xf32, strided<[?, 1], offset: ?>> to tensor<?x?xf32>
                %35 = arith.muli %32, %c1024_1 : index
                %36 = arith.addi %35, %21 : index
                %37 = loom.reinterpret_cast %arg1 to offset : [%36], sizes : [%16, %15], strides : [%c1024, %c1], reuse : [seq = false, spat = true, temp = true] : memref<*xf32> to memref<?x?xf32, strided<[?, ?], offset: ?>>
                %38 = loom.alloc(%16, %15) on @L1 : memref<?x?xf32>
                loom.copy %37, %38, interconnect : [], broadcast : [1, 1] : memref<?x?xf32, strided<[?, ?], offset: ?>> to memref<?x?xf32>
                %39 = bufferization.to_tensor %38 restrict writable : memref<?x?xf32> to tensor<?x?xf32>
                %40 = linalg.matmul ins(%34, %39 : tensor<?x?xf32>, tensor<?x?xf32>) outs(%19 : tensor<?x?xf32>) -> tensor<?x?xf32>
                %41 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%arg14, %40 : tensor<?x?xf32>, tensor<?x?xf32>) outs(%arg14 : tensor<?x?xf32>) {
                ^bb0(%in: f32, %in_3: f32, %out: f32):
                  %42 = arith.addf %in, %in_3 : f32
                  linalg.yield %42 : f32
                } -> tensor<?x?xf32>
                scf.yield %41 : tensor<?x?xf32>
              }
              %29 = arith.muli %20, %c1024_1 : index
              %30 = arith.addi %29, %21 : index
              %31 = loom.reinterpret_cast %arg2 to offset : [%30], sizes : [%14, %15], strides : [%c1024, %c1], reuse : [seq = false, spat = false, temp = false] : memref<*xf32> to memref<?x?xf32, strided<[?, ?], offset: ?>>
              bufferization.materialize_in_destination %28 in writable %31 : (tensor<?x?xf32>, memref<?x?xf32, strided<[?, ?], offset: ?>>) -> ()
            }
          }
        } {loom.mapped_to = @x}
      } {loom.mapped_to = @y}
      return
    }
  }
  module attributes {loom.pass_name = "EnumerateCopyBroadcast"} {
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
      %15 = loom.expression(%13, %12) {coeffs = [1, 1], logic = "mul"} : index
      %16 = loom.expression(%15, %14) {coeffs = [1, 1], logic = "mul"} : index
      loom.polynomial_constraint(%12, %13, %14, %15, %16) {monomials = [{coeff = -1 : i64, vars = [4]}], upper_bound = -262144 : i64}
      %17 = loom.expression(%12, %13) {coeffs = [1, 1], logic = "add"} : index
      %18 = loom.expression(%14, %17) {coeffs = [1, 1], logic = "mul"} : index
      loom.polynomial_constraint(%12, %14, %13, %17, %18) {monomials = [{coeff = 1 : i64, vars = [4]}], upper_bound = 374784 : i64}
      loom.polynomial_constraint(%12) {monomials = [{coeff = 1 : i64, vars = [0]}], upper_bound = 128 : i64}
      loom.polynomial_constraint(%13) {monomials = [{coeff = 1 : i64, vars = [0]}], upper_bound = 128 : i64}
      %19 = loom.expression(%13, %14) {coeffs = [1, 1], logic = "mul"} : index
      loom.polynomial_constraint(%12, %14, %13, %19) {monomials = [{coeff = 1024 : i64, vars = [0]}, {coeff = 1 : i64, vars = [3]}], upper_bound = 374784 : i64}
      %20 = loom.expression(%14, %12) {coeffs = [1, 1], logic = "mul"} : index
      %21 = loom.expression(%13, %14) {coeffs = [1, 1], logic = "mul"} : index
      %22 = loom.expression(%13, %12) {coeffs = [1, 1], logic = "mul"} : index
      %23 = loom.expression(%14, %22) {coeffs = [1, 1], logic = "mul"} : index
      loom.polynomial_constraint(%12, %13, %14, %20, %21, %22, %23) {monomials = [{coeff = 256 : i64, vars = [3]}, {coeff = 256 : i64, vars = [4]}, {coeff = -15 : i64, vars = [6]}], upper_bound = 0 : i64}
    }
    func.func @matmul_kernel__d1i0_d0i1__f01__hoist_block_0__v_h(%arg0: memref<*xf32> {tt.divisibility = 16 : i32}, %arg1: memref<*xf32> {tt.divisibility = 16 : i32}, %arg2: memref<*xf32> {tt.divisibility = 16 : i32}, %arg3: index, %arg4: index, %arg5: index, %arg6: index, %arg7: index, %arg8: index) {
      affine.parallel (%arg9) = (0) to (8) {
        affine.parallel (%arg10) = (0) to (8) {
          affine.for %arg11 = 0 to affine_map<()[s0, s1] -> (s0 ceildiv 8)>()[%arg3, %arg4] {
            affine.for %arg12 = 0 to affine_map<()[s0, s1] -> (s1 ceildiv 8)>()[%arg3, %arg4] {
              %12 = affine.apply affine_map<(d0, d1) -> (d0 * 8 + d1)>(%arg10, %arg12)
              %13 = affine.apply affine_map<(d0, d1) -> (d0 * 8 + d1)>(%arg9, %arg11)
              %c1024 = arith.constant 1024 : index
              %c1024_0 = arith.constant 1024 : index
              %c1 = arith.constant 1 : index
              %14 = loom.get_symbolic_block_size @constraints::@BM : index
              %15 = loom.get_symbolic_block_size @constraints::@BN : index
              %16 = loom.get_symbolic_block_size @constraints::@BK : index
              %17 = arith.ceildivsi %c1024_0, %16 : index
              %cst = arith.constant 0.000000e+00 : f32
              %c1024_1 = arith.constant 1024 : index
              %c0 = arith.constant 0 : index
              %18 = tensor.empty(%14, %15) : tensor<?x?xf32>
              %19 = linalg.fill ins(%cst : f32) outs(%18 : tensor<?x?xf32>) -> tensor<?x?xf32>
              %20 = arith.muli %13, %14 : index
              %21 = arith.muli %12, %15 : index
              %22 = arith.muli %14, %c1024_0 : index
              %c0_2 = arith.constant 0 : index
              %23 = arith.muli %20, %c1024_1 : index
              %24 = arith.muli %c0_2, %16 : index
              %25 = arith.addi %23, %24 : index
              %26 = loom.reinterpret_cast %arg0 to offset : [%25], sizes : [%17, %14, %16], strides : [%22, %c1024_0, %c1], reuse : [seq = false, spat = true, temp = true] : memref<*xf32> to memref<?x?x?xf32, strided<[?, ?, ?], offset: ?>>
              %27 = loom.alloc(%17, %14, %16) on @L1 : memref<?x?x?xf32>
              loom.copy %26, %27, interconnect : [@vertical_links], broadcast : [8, 1] : memref<?x?x?xf32, strided<[?, ?, ?], offset: ?>> to memref<?x?x?xf32>
              %28 = scf.for %arg13 = %c0 to %17 step %c1 iter_args(%arg14 = %19) -> (tensor<?x?xf32>) {
                %32 = arith.muli %arg13, %16 : index
                %33 = arith.muli %20, %c1024_1 : index
                %subview = memref.subview %27[%arg13, 0, 0] [1, %14, %16] [1, 1, 1] : memref<?x?x?xf32> to memref<?x?xf32, strided<[?, 1], offset: ?>>
                %34 = bufferization.to_tensor %subview restrict writable : memref<?x?xf32, strided<[?, 1], offset: ?>> to tensor<?x?xf32>
                %35 = arith.muli %32, %c1024_1 : index
                %36 = arith.addi %35, %21 : index
                %37 = loom.reinterpret_cast %arg1 to offset : [%36], sizes : [%16, %15], strides : [%c1024, %c1], reuse : [seq = false, spat = true, temp = true] : memref<*xf32> to memref<?x?xf32, strided<[?, ?], offset: ?>>
                %38 = loom.alloc(%16, %15) on @L1 : memref<?x?xf32>
                loom.copy %37, %38, interconnect : [@horizontal_links], broadcast : [1, 8] : memref<?x?xf32, strided<[?, ?], offset: ?>> to memref<?x?xf32>
                %39 = bufferization.to_tensor %38 restrict writable : memref<?x?xf32> to tensor<?x?xf32>
                %40 = linalg.matmul ins(%34, %39 : tensor<?x?xf32>, tensor<?x?xf32>) outs(%19 : tensor<?x?xf32>) -> tensor<?x?xf32>
                %41 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%arg14, %40 : tensor<?x?xf32>, tensor<?x?xf32>) outs(%arg14 : tensor<?x?xf32>) {
                ^bb0(%in: f32, %in_3: f32, %out: f32):
                  %42 = arith.addf %in, %in_3 : f32
                  linalg.yield %42 : f32
                } -> tensor<?x?xf32>
                scf.yield %41 : tensor<?x?xf32>
              }
              %29 = arith.muli %20, %c1024_1 : index
              %30 = arith.addi %29, %21 : index
              %31 = loom.reinterpret_cast %arg2 to offset : [%30], sizes : [%14, %15], strides : [%c1024, %c1], reuse : [seq = false, spat = false, temp = false] : memref<*xf32> to memref<?x?xf32, strided<[?, ?], offset: ?>>
              bufferization.materialize_in_destination %28 in writable %31 : (tensor<?x?xf32>, memref<?x?xf32, strided<[?, ?], offset: ?>>) -> ()
            }
          }
        } {loom.mapped_to = @x}
      } {loom.mapped_to = @y}
      return
    }
  }
  module attributes {loom.pass_name = "EnumerateCopyBroadcast"} {
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
      %15 = loom.expression(%13, %12) {coeffs = [1, 1], logic = "mul"} : index
      %16 = loom.expression(%15, %14) {coeffs = [1, 1], logic = "mul"} : index
      loom.polynomial_constraint(%12, %13, %14, %15, %16) {monomials = [{coeff = -1 : i64, vars = [4]}], upper_bound = -262144 : i64}
      %17 = loom.expression(%12, %13) {coeffs = [1, 1], logic = "add"} : index
      %18 = loom.expression(%14, %17) {coeffs = [1, 1], logic = "mul"} : index
      loom.polynomial_constraint(%12, %14, %13, %17, %18) {monomials = [{coeff = 1 : i64, vars = [4]}], upper_bound = 374784 : i64}
      loom.polynomial_constraint(%12) {monomials = [{coeff = 1 : i64, vars = [0]}], upper_bound = 128 : i64}
      loom.polynomial_constraint(%13) {monomials = [{coeff = 1 : i64, vars = [0]}], upper_bound = 128 : i64}
      %19 = loom.expression(%14, %12) {coeffs = [1, 1], logic = "mul"} : index
      loom.polynomial_constraint(%13, %12, %14, %19) {monomials = [{coeff = 1024 : i64, vars = [0]}, {coeff = 1 : i64, vars = [3]}], upper_bound = 374784 : i64}
      %20 = loom.expression(%14, %12) {coeffs = [1, 1], logic = "mul"} : index
      %21 = loom.expression(%13, %14) {coeffs = [1, 1], logic = "mul"} : index
      %22 = loom.expression(%13, %12) {coeffs = [1, 1], logic = "mul"} : index
      %23 = loom.expression(%14, %22) {coeffs = [1, 1], logic = "mul"} : index
      loom.polynomial_constraint(%12, %13, %14, %20, %21, %22, %23) {monomials = [{coeff = 64 : i64, vars = [3]}, {coeff = 64 : i64, vars = [4]}, {coeff = -1 : i64, vars = [6]}], upper_bound = 0 : i64}
    }
    func.func @matmul_kernel__d1i0_d0i1__f01__hoist_block_1__d_d(%arg0: memref<*xf32> {tt.divisibility = 16 : i32}, %arg1: memref<*xf32> {tt.divisibility = 16 : i32}, %arg2: memref<*xf32> {tt.divisibility = 16 : i32}, %arg3: index, %arg4: index, %arg5: index, %arg6: index, %arg7: index, %arg8: index) {
      affine.parallel (%arg9) = (0) to (8) {
        affine.parallel (%arg10) = (0) to (8) {
          affine.for %arg11 = 0 to affine_map<()[s0, s1] -> (s0 ceildiv 8)>()[%arg3, %arg4] {
            affine.for %arg12 = 0 to affine_map<()[s0, s1] -> (s1 ceildiv 8)>()[%arg3, %arg4] {
              %12 = affine.apply affine_map<(d0, d1) -> (d0 * 8 + d1)>(%arg10, %arg12)
              %13 = affine.apply affine_map<(d0, d1) -> (d0 * 8 + d1)>(%arg9, %arg11)
              %c1024 = arith.constant 1024 : index
              %c1024_0 = arith.constant 1024 : index
              %c1 = arith.constant 1 : index
              %14 = loom.get_symbolic_block_size @constraints::@BM : index
              %15 = loom.get_symbolic_block_size @constraints::@BN : index
              %16 = loom.get_symbolic_block_size @constraints::@BK : index
              %17 = arith.ceildivsi %c1024_0, %16 : index
              %cst = arith.constant 0.000000e+00 : f32
              %c1024_1 = arith.constant 1024 : index
              %c0 = arith.constant 0 : index
              %18 = tensor.empty(%14, %15) : tensor<?x?xf32>
              %19 = linalg.fill ins(%cst : f32) outs(%18 : tensor<?x?xf32>) -> tensor<?x?xf32>
              %20 = arith.muli %13, %14 : index
              %21 = arith.muli %12, %15 : index
              %22 = arith.muli %16, %c1024 : index
              %c0_2 = arith.constant 0 : index
              %23 = arith.muli %c0_2, %16 : index
              %24 = arith.muli %23, %c1024_1 : index
              %25 = arith.addi %24, %21 : index
              %26 = loom.reinterpret_cast %arg1 to offset : [%25], sizes : [%17, %16, %15], strides : [%22, %c1024, %c1], reuse : [seq = false, spat = true, temp = true] : memref<*xf32> to memref<?x?x?xf32, strided<[?, ?, ?], offset: ?>>
              %27 = loom.alloc(%17, %16, %15) on @L1 : memref<?x?x?xf32>
              loom.copy %26, %27, interconnect : [], broadcast : [1, 1] : memref<?x?x?xf32, strided<[?, ?, ?], offset: ?>> to memref<?x?x?xf32>
              %28 = scf.for %arg13 = %c0 to %17 step %c1 iter_args(%arg14 = %19) -> (tensor<?x?xf32>) {
                %32 = arith.muli %arg13, %16 : index
                %33 = arith.muli %20, %c1024_1 : index
                %34 = arith.addi %33, %32 : index
                %35 = loom.reinterpret_cast %arg0 to offset : [%34], sizes : [%14, %16], strides : [%c1024_0, %c1], reuse : [seq = false, spat = true, temp = true] : memref<*xf32> to memref<?x?xf32, strided<[?, ?], offset: ?>>
                %36 = loom.alloc(%14, %16) on @L1 : memref<?x?xf32>
                loom.copy %35, %36, interconnect : [], broadcast : [1, 1] : memref<?x?xf32, strided<[?, ?], offset: ?>> to memref<?x?xf32>
                %37 = bufferization.to_tensor %36 restrict writable : memref<?x?xf32> to tensor<?x?xf32>
                %38 = arith.muli %32, %c1024_1 : index
                %subview = memref.subview %27[%arg13, 0, 0] [1, %16, %15] [1, 1, 1] : memref<?x?x?xf32> to memref<?x?xf32, strided<[?, 1], offset: ?>>
                %39 = bufferization.to_tensor %subview restrict writable : memref<?x?xf32, strided<[?, 1], offset: ?>> to tensor<?x?xf32>
                %40 = linalg.matmul ins(%37, %39 : tensor<?x?xf32>, tensor<?x?xf32>) outs(%19 : tensor<?x?xf32>) -> tensor<?x?xf32>
                %41 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%arg14, %40 : tensor<?x?xf32>, tensor<?x?xf32>) outs(%arg14 : tensor<?x?xf32>) {
                ^bb0(%in: f32, %in_3: f32, %out: f32):
                  %42 = arith.addf %in, %in_3 : f32
                  linalg.yield %42 : f32
                } -> tensor<?x?xf32>
                scf.yield %41 : tensor<?x?xf32>
              }
              %29 = arith.muli %20, %c1024_1 : index
              %30 = arith.addi %29, %21 : index
              %31 = loom.reinterpret_cast %arg2 to offset : [%30], sizes : [%14, %15], strides : [%c1024, %c1], reuse : [seq = false, spat = false, temp = false] : memref<*xf32> to memref<?x?xf32, strided<[?, ?], offset: ?>>
              bufferization.materialize_in_destination %28 in writable %31 : (tensor<?x?xf32>, memref<?x?xf32, strided<[?, ?], offset: ?>>) -> ()
            }
          }
        } {loom.mapped_to = @x}
      } {loom.mapped_to = @y}
      return
    }
  }
  module attributes {loom.pass_name = "EnumerateCopyBroadcast"} {
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
      %15 = loom.expression(%12, %13) {coeffs = [1, 1], logic = "mul"} : index
      %16 = loom.expression(%15, %14) {coeffs = [1, 1], logic = "mul"} : index
      loom.polynomial_constraint(%12, %13, %14, %15, %16) {monomials = [{coeff = -1 : i64, vars = [4]}], upper_bound = -262144 : i64}
      %17 = loom.expression(%12, %13) {coeffs = [1, 1], logic = "add"} : index
      %18 = loom.expression(%14, %17) {coeffs = [1, 1], logic = "mul"} : index
      loom.polynomial_constraint(%12, %14, %13, %17, %18) {monomials = [{coeff = 1 : i64, vars = [4]}], upper_bound = 374784 : i64}
      loom.polynomial_constraint(%12) {monomials = [{coeff = 1 : i64, vars = [0]}], upper_bound = 128 : i64}
      loom.polynomial_constraint(%13) {monomials = [{coeff = 1 : i64, vars = [0]}], upper_bound = 128 : i64}
      %19 = loom.expression(%12, %14) {coeffs = [1, 1], logic = "mul"} : index
      loom.polynomial_constraint(%13, %12, %14, %19) {monomials = [{coeff = 1024 : i64, vars = [0]}, {coeff = 1 : i64, vars = [3]}], upper_bound = 374784 : i64}
      %20 = loom.expression(%12, %14) {coeffs = [1, 1], logic = "mul"} : index
      %21 = loom.expression(%13, %14) {coeffs = [1, 1], logic = "mul"} : index
      %22 = loom.expression(%12, %13) {coeffs = [1, 1], logic = "mul"} : index
      %23 = loom.expression(%14, %22) {coeffs = [1, 1], logic = "mul"} : index
      loom.polynomial_constraint(%12, %13, %14, %20, %21, %22, %23) {monomials = [{coeff = 960 : i64, vars = [3]}, {coeff = 256 : i64, vars = [4]}, {coeff = -15 : i64, vars = [6]}], upper_bound = 0 : i64}
    }
    func.func @matmul_kernel__d1i0_d0i1__f01__hoist_block_1__d_v(%arg0: memref<*xf32> {tt.divisibility = 16 : i32}, %arg1: memref<*xf32> {tt.divisibility = 16 : i32}, %arg2: memref<*xf32> {tt.divisibility = 16 : i32}, %arg3: index, %arg4: index, %arg5: index, %arg6: index, %arg7: index, %arg8: index) {
      affine.parallel (%arg9) = (0) to (8) {
        affine.parallel (%arg10) = (0) to (8) {
          affine.for %arg11 = 0 to affine_map<()[s0, s1] -> (s0 ceildiv 8)>()[%arg3, %arg4] {
            affine.for %arg12 = 0 to affine_map<()[s0, s1] -> (s1 ceildiv 8)>()[%arg3, %arg4] {
              %12 = affine.apply affine_map<(d0, d1) -> (d0 * 8 + d1)>(%arg10, %arg12)
              %13 = affine.apply affine_map<(d0, d1) -> (d0 * 8 + d1)>(%arg9, %arg11)
              %c1024 = arith.constant 1024 : index
              %c1024_0 = arith.constant 1024 : index
              %c1 = arith.constant 1 : index
              %14 = loom.get_symbolic_block_size @constraints::@BM : index
              %15 = loom.get_symbolic_block_size @constraints::@BN : index
              %16 = loom.get_symbolic_block_size @constraints::@BK : index
              %17 = arith.ceildivsi %c1024_0, %16 : index
              %cst = arith.constant 0.000000e+00 : f32
              %c1024_1 = arith.constant 1024 : index
              %c0 = arith.constant 0 : index
              %18 = tensor.empty(%14, %15) : tensor<?x?xf32>
              %19 = linalg.fill ins(%cst : f32) outs(%18 : tensor<?x?xf32>) -> tensor<?x?xf32>
              %20 = arith.muli %13, %14 : index
              %21 = arith.muli %12, %15 : index
              %22 = arith.muli %16, %c1024 : index
              %c0_2 = arith.constant 0 : index
              %23 = arith.muli %c0_2, %16 : index
              %24 = arith.muli %23, %c1024_1 : index
              %25 = arith.addi %24, %21 : index
              %26 = loom.reinterpret_cast %arg1 to offset : [%25], sizes : [%17, %16, %15], strides : [%22, %c1024, %c1], reuse : [seq = false, spat = true, temp = true] : memref<*xf32> to memref<?x?x?xf32, strided<[?, ?, ?], offset: ?>>
              %27 = loom.alloc(%17, %16, %15) on @L1 : memref<?x?x?xf32>
              loom.copy %26, %27, interconnect : [], broadcast : [1, 1] : memref<?x?x?xf32, strided<[?, ?, ?], offset: ?>> to memref<?x?x?xf32>
              %28 = scf.for %arg13 = %c0 to %17 step %c1 iter_args(%arg14 = %19) -> (tensor<?x?xf32>) {
                %32 = arith.muli %arg13, %16 : index
                %33 = arith.muli %20, %c1024_1 : index
                %34 = arith.addi %33, %32 : index
                %35 = loom.reinterpret_cast %arg0 to offset : [%34], sizes : [%14, %16], strides : [%c1024_0, %c1], reuse : [seq = false, spat = true, temp = true] : memref<*xf32> to memref<?x?xf32, strided<[?, ?], offset: ?>>
                %36 = loom.alloc(%14, %16) on @L1 : memref<?x?xf32>
                loom.copy %35, %36, interconnect : [@vertical_links], broadcast : [8, 1] : memref<?x?xf32, strided<[?, ?], offset: ?>> to memref<?x?xf32>
                %37 = bufferization.to_tensor %36 restrict writable : memref<?x?xf32> to tensor<?x?xf32>
                %38 = arith.muli %32, %c1024_1 : index
                %subview = memref.subview %27[%arg13, 0, 0] [1, %16, %15] [1, 1, 1] : memref<?x?x?xf32> to memref<?x?xf32, strided<[?, 1], offset: ?>>
                %39 = bufferization.to_tensor %subview restrict writable : memref<?x?xf32, strided<[?, 1], offset: ?>> to tensor<?x?xf32>
                %40 = linalg.matmul ins(%37, %39 : tensor<?x?xf32>, tensor<?x?xf32>) outs(%19 : tensor<?x?xf32>) -> tensor<?x?xf32>
                %41 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%arg14, %40 : tensor<?x?xf32>, tensor<?x?xf32>) outs(%arg14 : tensor<?x?xf32>) {
                ^bb0(%in: f32, %in_3: f32, %out: f32):
                  %42 = arith.addf %in, %in_3 : f32
                  linalg.yield %42 : f32
                } -> tensor<?x?xf32>
                scf.yield %41 : tensor<?x?xf32>
              }
              %29 = arith.muli %20, %c1024_1 : index
              %30 = arith.addi %29, %21 : index
              %31 = loom.reinterpret_cast %arg2 to offset : [%30], sizes : [%14, %15], strides : [%c1024, %c1], reuse : [seq = false, spat = false, temp = false] : memref<*xf32> to memref<?x?xf32, strided<[?, ?], offset: ?>>
              bufferization.materialize_in_destination %28 in writable %31 : (tensor<?x?xf32>, memref<?x?xf32, strided<[?, ?], offset: ?>>) -> ()
            }
          }
        } {loom.mapped_to = @x}
      } {loom.mapped_to = @y}
      return
    }
  }
  module attributes {loom.pass_name = "EnumerateCopyBroadcast"} {
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
      %15 = loom.expression(%12, %13) {coeffs = [1, 1], logic = "mul"} : index
      %16 = loom.expression(%15, %14) {coeffs = [1, 1], logic = "mul"} : index
      loom.polynomial_constraint(%12, %13, %14, %15, %16) {monomials = [{coeff = -1 : i64, vars = [4]}], upper_bound = -262144 : i64}
      %17 = loom.expression(%12, %13) {coeffs = [1, 1], logic = "add"} : index
      %18 = loom.expression(%14, %17) {coeffs = [1, 1], logic = "mul"} : index
      loom.polynomial_constraint(%12, %14, %13, %17, %18) {monomials = [{coeff = 1 : i64, vars = [4]}], upper_bound = 374784 : i64}
      loom.polynomial_constraint(%12) {monomials = [{coeff = 1 : i64, vars = [0]}], upper_bound = 128 : i64}
      loom.polynomial_constraint(%13) {monomials = [{coeff = 1 : i64, vars = [0]}], upper_bound = 128 : i64}
      %19 = loom.expression(%12, %14) {coeffs = [1, 1], logic = "mul"} : index
      loom.polynomial_constraint(%13, %12, %14, %19) {monomials = [{coeff = 1024 : i64, vars = [0]}, {coeff = 1 : i64, vars = [3]}], upper_bound = 374784 : i64}
      %20 = loom.expression(%12, %14) {coeffs = [1, 1], logic = "mul"} : index
      %21 = loom.expression(%13, %14) {coeffs = [1, 1], logic = "mul"} : index
      %22 = loom.expression(%12, %13) {coeffs = [1, 1], logic = "mul"} : index
      %23 = loom.expression(%14, %22) {coeffs = [1, 1], logic = "mul"} : index
      loom.polynomial_constraint(%12, %13, %14, %20, %21, %22, %23) {monomials = [{coeff = 256 : i64, vars = [3]}, {coeff = 960 : i64, vars = [4]}, {coeff = -15 : i64, vars = [6]}], upper_bound = 0 : i64}
    }
    func.func @matmul_kernel__d1i0_d0i1__f01__hoist_block_1__h_d(%arg0: memref<*xf32> {tt.divisibility = 16 : i32}, %arg1: memref<*xf32> {tt.divisibility = 16 : i32}, %arg2: memref<*xf32> {tt.divisibility = 16 : i32}, %arg3: index, %arg4: index, %arg5: index, %arg6: index, %arg7: index, %arg8: index) {
      affine.parallel (%arg9) = (0) to (8) {
        affine.parallel (%arg10) = (0) to (8) {
          affine.for %arg11 = 0 to affine_map<()[s0, s1] -> (s0 ceildiv 8)>()[%arg3, %arg4] {
            affine.for %arg12 = 0 to affine_map<()[s0, s1] -> (s1 ceildiv 8)>()[%arg3, %arg4] {
              %12 = affine.apply affine_map<(d0, d1) -> (d0 * 8 + d1)>(%arg10, %arg12)
              %13 = affine.apply affine_map<(d0, d1) -> (d0 * 8 + d1)>(%arg9, %arg11)
              %c1024 = arith.constant 1024 : index
              %c1024_0 = arith.constant 1024 : index
              %c1 = arith.constant 1 : index
              %14 = loom.get_symbolic_block_size @constraints::@BM : index
              %15 = loom.get_symbolic_block_size @constraints::@BN : index
              %16 = loom.get_symbolic_block_size @constraints::@BK : index
              %17 = arith.ceildivsi %c1024_0, %16 : index
              %cst = arith.constant 0.000000e+00 : f32
              %c1024_1 = arith.constant 1024 : index
              %c0 = arith.constant 0 : index
              %18 = tensor.empty(%14, %15) : tensor<?x?xf32>
              %19 = linalg.fill ins(%cst : f32) outs(%18 : tensor<?x?xf32>) -> tensor<?x?xf32>
              %20 = arith.muli %13, %14 : index
              %21 = arith.muli %12, %15 : index
              %22 = arith.muli %16, %c1024 : index
              %c0_2 = arith.constant 0 : index
              %23 = arith.muli %c0_2, %16 : index
              %24 = arith.muli %23, %c1024_1 : index
              %25 = arith.addi %24, %21 : index
              %26 = loom.reinterpret_cast %arg1 to offset : [%25], sizes : [%17, %16, %15], strides : [%22, %c1024, %c1], reuse : [seq = false, spat = true, temp = true] : memref<*xf32> to memref<?x?x?xf32, strided<[?, ?, ?], offset: ?>>
              %27 = loom.alloc(%17, %16, %15) on @L1 : memref<?x?x?xf32>
              loom.copy %26, %27, interconnect : [@horizontal_links], broadcast : [1, 8] : memref<?x?x?xf32, strided<[?, ?, ?], offset: ?>> to memref<?x?x?xf32>
              %28 = scf.for %arg13 = %c0 to %17 step %c1 iter_args(%arg14 = %19) -> (tensor<?x?xf32>) {
                %32 = arith.muli %arg13, %16 : index
                %33 = arith.muli %20, %c1024_1 : index
                %34 = arith.addi %33, %32 : index
                %35 = loom.reinterpret_cast %arg0 to offset : [%34], sizes : [%14, %16], strides : [%c1024_0, %c1], reuse : [seq = false, spat = true, temp = true] : memref<*xf32> to memref<?x?xf32, strided<[?, ?], offset: ?>>
                %36 = loom.alloc(%14, %16) on @L1 : memref<?x?xf32>
                loom.copy %35, %36, interconnect : [], broadcast : [1, 1] : memref<?x?xf32, strided<[?, ?], offset: ?>> to memref<?x?xf32>
                %37 = bufferization.to_tensor %36 restrict writable : memref<?x?xf32> to tensor<?x?xf32>
                %38 = arith.muli %32, %c1024_1 : index
                %subview = memref.subview %27[%arg13, 0, 0] [1, %16, %15] [1, 1, 1] : memref<?x?x?xf32> to memref<?x?xf32, strided<[?, 1], offset: ?>>
                %39 = bufferization.to_tensor %subview restrict writable : memref<?x?xf32, strided<[?, 1], offset: ?>> to tensor<?x?xf32>
                %40 = linalg.matmul ins(%37, %39 : tensor<?x?xf32>, tensor<?x?xf32>) outs(%19 : tensor<?x?xf32>) -> tensor<?x?xf32>
                %41 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%arg14, %40 : tensor<?x?xf32>, tensor<?x?xf32>) outs(%arg14 : tensor<?x?xf32>) {
                ^bb0(%in: f32, %in_3: f32, %out: f32):
                  %42 = arith.addf %in, %in_3 : f32
                  linalg.yield %42 : f32
                } -> tensor<?x?xf32>
                scf.yield %41 : tensor<?x?xf32>
              }
              %29 = arith.muli %20, %c1024_1 : index
              %30 = arith.addi %29, %21 : index
              %31 = loom.reinterpret_cast %arg2 to offset : [%30], sizes : [%14, %15], strides : [%c1024, %c1], reuse : [seq = false, spat = false, temp = false] : memref<*xf32> to memref<?x?xf32, strided<[?, ?], offset: ?>>
              bufferization.materialize_in_destination %28 in writable %31 : (tensor<?x?xf32>, memref<?x?xf32, strided<[?, ?], offset: ?>>) -> ()
            }
          }
        } {loom.mapped_to = @x}
      } {loom.mapped_to = @y}
      return
    }
  }
  module attributes {loom.pass_name = "EnumerateCopyBroadcast"} {
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
      %15 = loom.expression(%13, %12) {coeffs = [1, 1], logic = "mul"} : index
      %16 = loom.expression(%14, %15) {coeffs = [1, 1], logic = "mul"} : index
      loom.polynomial_constraint(%12, %13, %14, %15, %16) {monomials = [{coeff = -1 : i64, vars = [4]}], upper_bound = -262144 : i64}
      %17 = loom.expression(%12, %13) {coeffs = [1, 1], logic = "add"} : index
      %18 = loom.expression(%14, %17) {coeffs = [1, 1], logic = "mul"} : index
      loom.polynomial_constraint(%12, %14, %13, %17, %18) {monomials = [{coeff = 1 : i64, vars = [4]}], upper_bound = 374784 : i64}
      loom.polynomial_constraint(%12) {monomials = [{coeff = 1 : i64, vars = [0]}], upper_bound = 128 : i64}
      loom.polynomial_constraint(%13) {monomials = [{coeff = 1 : i64, vars = [0]}], upper_bound = 128 : i64}
      %19 = loom.expression(%14, %12) {coeffs = [1, 1], logic = "mul"} : index
      loom.polynomial_constraint(%13, %12, %14, %19) {monomials = [{coeff = 1024 : i64, vars = [0]}, {coeff = 1 : i64, vars = [3]}], upper_bound = 374784 : i64}
      %20 = loom.expression(%14, %12) {coeffs = [1, 1], logic = "mul"} : index
      %21 = loom.expression(%14, %13) {coeffs = [1, 1], logic = "mul"} : index
      %22 = loom.expression(%13, %12) {coeffs = [1, 1], logic = "mul"} : index
      %23 = loom.expression(%14, %22) {coeffs = [1, 1], logic = "mul"} : index
      loom.polynomial_constraint(%12, %13, %14, %20, %21, %22, %23) {monomials = [{coeff = 256 : i64, vars = [3]}, {coeff = 256 : i64, vars = [4]}, {coeff = -15 : i64, vars = [6]}], upper_bound = 0 : i64}
    }
    func.func @matmul_kernel__d1i0_d0i1__f01__hoist_block_1__h_v(%arg0: memref<*xf32> {tt.divisibility = 16 : i32}, %arg1: memref<*xf32> {tt.divisibility = 16 : i32}, %arg2: memref<*xf32> {tt.divisibility = 16 : i32}, %arg3: index, %arg4: index, %arg5: index, %arg6: index, %arg7: index, %arg8: index) {
      affine.parallel (%arg9) = (0) to (8) {
        affine.parallel (%arg10) = (0) to (8) {
          affine.for %arg11 = 0 to affine_map<()[s0, s1] -> (s0 ceildiv 8)>()[%arg3, %arg4] {
            affine.for %arg12 = 0 to affine_map<()[s0, s1] -> (s1 ceildiv 8)>()[%arg3, %arg4] {
              %12 = affine.apply affine_map<(d0, d1) -> (d0 * 8 + d1)>(%arg10, %arg12)
              %13 = affine.apply affine_map<(d0, d1) -> (d0 * 8 + d1)>(%arg9, %arg11)
              %c1024 = arith.constant 1024 : index
              %c1024_0 = arith.constant 1024 : index
              %c1 = arith.constant 1 : index
              %14 = loom.get_symbolic_block_size @constraints::@BM : index
              %15 = loom.get_symbolic_block_size @constraints::@BN : index
              %16 = loom.get_symbolic_block_size @constraints::@BK : index
              %17 = arith.ceildivsi %c1024_0, %16 : index
              %cst = arith.constant 0.000000e+00 : f32
              %c1024_1 = arith.constant 1024 : index
              %c0 = arith.constant 0 : index
              %18 = tensor.empty(%14, %15) : tensor<?x?xf32>
              %19 = linalg.fill ins(%cst : f32) outs(%18 : tensor<?x?xf32>) -> tensor<?x?xf32>
              %20 = arith.muli %13, %14 : index
              %21 = arith.muli %12, %15 : index
              %22 = arith.muli %16, %c1024 : index
              %c0_2 = arith.constant 0 : index
              %23 = arith.muli %c0_2, %16 : index
              %24 = arith.muli %23, %c1024_1 : index
              %25 = arith.addi %24, %21 : index
              %26 = loom.reinterpret_cast %arg1 to offset : [%25], sizes : [%17, %16, %15], strides : [%22, %c1024, %c1], reuse : [seq = false, spat = true, temp = true] : memref<*xf32> to memref<?x?x?xf32, strided<[?, ?, ?], offset: ?>>
              %27 = loom.alloc(%17, %16, %15) on @L1 : memref<?x?x?xf32>
              loom.copy %26, %27, interconnect : [@horizontal_links], broadcast : [1, 8] : memref<?x?x?xf32, strided<[?, ?, ?], offset: ?>> to memref<?x?x?xf32>
              %28 = scf.for %arg13 = %c0 to %17 step %c1 iter_args(%arg14 = %19) -> (tensor<?x?xf32>) {
                %32 = arith.muli %arg13, %16 : index
                %33 = arith.muli %20, %c1024_1 : index
                %34 = arith.addi %33, %32 : index
                %35 = loom.reinterpret_cast %arg0 to offset : [%34], sizes : [%14, %16], strides : [%c1024_0, %c1], reuse : [seq = false, spat = true, temp = true] : memref<*xf32> to memref<?x?xf32, strided<[?, ?], offset: ?>>
                %36 = loom.alloc(%14, %16) on @L1 : memref<?x?xf32>
                loom.copy %35, %36, interconnect : [@vertical_links], broadcast : [8, 1] : memref<?x?xf32, strided<[?, ?], offset: ?>> to memref<?x?xf32>
                %37 = bufferization.to_tensor %36 restrict writable : memref<?x?xf32> to tensor<?x?xf32>
                %38 = arith.muli %32, %c1024_1 : index
                %subview = memref.subview %27[%arg13, 0, 0] [1, %16, %15] [1, 1, 1] : memref<?x?x?xf32> to memref<?x?xf32, strided<[?, 1], offset: ?>>
                %39 = bufferization.to_tensor %subview restrict writable : memref<?x?xf32, strided<[?, 1], offset: ?>> to tensor<?x?xf32>
                %40 = linalg.matmul ins(%37, %39 : tensor<?x?xf32>, tensor<?x?xf32>) outs(%19 : tensor<?x?xf32>) -> tensor<?x?xf32>
                %41 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%arg14, %40 : tensor<?x?xf32>, tensor<?x?xf32>) outs(%arg14 : tensor<?x?xf32>) {
                ^bb0(%in: f32, %in_3: f32, %out: f32):
                  %42 = arith.addf %in, %in_3 : f32
                  linalg.yield %42 : f32
                } -> tensor<?x?xf32>
                scf.yield %41 : tensor<?x?xf32>
              }
              %29 = arith.muli %20, %c1024_1 : index
              %30 = arith.addi %29, %21 : index
              %31 = loom.reinterpret_cast %arg2 to offset : [%30], sizes : [%14, %15], strides : [%c1024, %c1], reuse : [seq = false, spat = false, temp = false] : memref<*xf32> to memref<?x?xf32, strided<[?, ?], offset: ?>>
              bufferization.materialize_in_destination %28 in writable %31 : (tensor<?x?xf32>, memref<?x?xf32, strided<[?, ?], offset: ?>>) -> ()
            }
          }
        } {loom.mapped_to = @x}
      } {loom.mapped_to = @y}
      return
    }
  }
  module attributes {loom.pass_name = "EnumerateCopyBroadcast"} {
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
      %15 = loom.expression(%13, %12) {coeffs = [1, 1], logic = "mul"} : index
      %16 = loom.expression(%15, %14) {coeffs = [1, 1], logic = "mul"} : index
      loom.polynomial_constraint(%12, %13, %14, %15, %16) {monomials = [{coeff = -1 : i64, vars = [4]}], upper_bound = -262144 : i64}
      %17 = loom.expression(%12, %13) {coeffs = [1, 1], logic = "add"} : index
      %18 = loom.expression(%14, %17) {coeffs = [1, 1], logic = "mul"} : index
      loom.polynomial_constraint(%12, %14, %13, %17, %18) {monomials = [{coeff = 1 : i64, vars = [4]}], upper_bound = 374784 : i64}
      loom.polynomial_constraint(%12) {monomials = [{coeff = 1 : i64, vars = [0]}], upper_bound = 128 : i64}
      loom.polynomial_constraint(%13) {monomials = [{coeff = 1 : i64, vars = [0]}], upper_bound = 128 : i64}
      %19 = loom.expression(%14, %12) {coeffs = [1, 1], logic = "mul"} : index
      %20 = loom.expression(%13, %14) {coeffs = [1, 1], logic = "mul"} : index
      %21 = loom.expression(%13, %12) {coeffs = [1, 1], logic = "mul"} : index
      %22 = loom.expression(%14, %21) {coeffs = [1, 1], logic = "mul"} : index
      loom.polynomial_constraint(%12, %13, %14, %19, %20, %21, %22) {monomials = [{coeff = 64 : i64, vars = [3]}, {coeff = 64 : i64, vars = [4]}, {coeff = -1 : i64, vars = [6]}], upper_bound = 0 : i64}
    }
    func.func @matmul_kernel__d1i0_d0i1__f10__d_d(%arg0: memref<*xf32> {tt.divisibility = 16 : i32}, %arg1: memref<*xf32> {tt.divisibility = 16 : i32}, %arg2: memref<*xf32> {tt.divisibility = 16 : i32}, %arg3: index, %arg4: index, %arg5: index, %arg6: index, %arg7: index, %arg8: index) {
      affine.parallel (%arg9) = (0) to (8) {
        affine.parallel (%arg10) = (0) to (8) {
          affine.for %arg11 = 0 to affine_map<()[s0, s1] -> (s1 ceildiv 8)>()[%arg3, %arg4] {
            affine.for %arg12 = 0 to affine_map<()[s0, s1] -> (s0 ceildiv 8)>()[%arg3, %arg4] {
              %12 = affine.apply affine_map<(d0, d1) -> (d0 * 8 + d1)>(%arg10, %arg11)
              %13 = affine.apply affine_map<(d0, d1) -> (d0 * 8 + d1)>(%arg9, %arg12)
              %c1024 = arith.constant 1024 : index
              %c1024_0 = arith.constant 1024 : index
              %c1 = arith.constant 1 : index
              %14 = loom.get_symbolic_block_size @constraints::@BM : index
              %15 = loom.get_symbolic_block_size @constraints::@BN : index
              %16 = loom.get_symbolic_block_size @constraints::@BK : index
              %17 = arith.ceildivsi %c1024_0, %16 : index
              %cst = arith.constant 0.000000e+00 : f32
              %c1024_1 = arith.constant 1024 : index
              %c0 = arith.constant 0 : index
              %18 = tensor.empty(%14, %15) : tensor<?x?xf32>
              %19 = linalg.fill ins(%cst : f32) outs(%18 : tensor<?x?xf32>) -> tensor<?x?xf32>
              %20 = arith.muli %13, %14 : index
              %21 = arith.muli %12, %15 : index
              %22 = scf.for %arg13 = %c0 to %17 step %c1 iter_args(%arg14 = %19) -> (tensor<?x?xf32>) {
                %26 = arith.muli %arg13, %16 : index
                %27 = arith.muli %20, %c1024_1 : index
                %28 = arith.addi %27, %26 : index
                %29 = loom.reinterpret_cast %arg0 to offset : [%28], sizes : [%14, %16], strides : [%c1024_0, %c1], reuse : [seq = false, spat = true, temp = true] : memref<*xf32> to memref<?x?xf32, strided<[?, ?], offset: ?>>
                %30 = loom.alloc(%14, %16) on @L1 : memref<?x?xf32>
                loom.copy %29, %30, interconnect : [], broadcast : [1, 1] : memref<?x?xf32, strided<[?, ?], offset: ?>> to memref<?x?xf32>
                %31 = bufferization.to_tensor %30 restrict writable : memref<?x?xf32> to tensor<?x?xf32>
                %32 = arith.muli %26, %c1024_1 : index
                %33 = arith.addi %32, %21 : index
                %34 = loom.reinterpret_cast %arg1 to offset : [%33], sizes : [%16, %15], strides : [%c1024, %c1], reuse : [seq = false, spat = true, temp = true] : memref<*xf32> to memref<?x?xf32, strided<[?, ?], offset: ?>>
                %35 = loom.alloc(%16, %15) on @L1 : memref<?x?xf32>
                loom.copy %34, %35, interconnect : [], broadcast : [1, 1] : memref<?x?xf32, strided<[?, ?], offset: ?>> to memref<?x?xf32>
                %36 = bufferization.to_tensor %35 restrict writable : memref<?x?xf32> to tensor<?x?xf32>
                %37 = linalg.matmul ins(%31, %36 : tensor<?x?xf32>, tensor<?x?xf32>) outs(%19 : tensor<?x?xf32>) -> tensor<?x?xf32>
                %38 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%arg14, %37 : tensor<?x?xf32>, tensor<?x?xf32>) outs(%arg14 : tensor<?x?xf32>) {
                ^bb0(%in: f32, %in_2: f32, %out: f32):
                  %39 = arith.addf %in, %in_2 : f32
                  linalg.yield %39 : f32
                } -> tensor<?x?xf32>
                scf.yield %38 : tensor<?x?xf32>
              }
              %23 = arith.muli %20, %c1024_1 : index
              %24 = arith.addi %23, %21 : index
              %25 = loom.reinterpret_cast %arg2 to offset : [%24], sizes : [%14, %15], strides : [%c1024, %c1], reuse : [seq = false, spat = false, temp = false] : memref<*xf32> to memref<?x?xf32, strided<[?, ?], offset: ?>>
              bufferization.materialize_in_destination %22 in writable %25 : (tensor<?x?xf32>, memref<?x?xf32, strided<[?, ?], offset: ?>>) -> ()
            }
          }
        } {loom.mapped_to = @x}
      } {loom.mapped_to = @y}
      return
    }
  }
  module attributes {loom.pass_name = "EnumerateCopyBroadcast"} {
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
      %15 = loom.expression(%12, %13) {coeffs = [1, 1], logic = "mul"} : index
      %16 = loom.expression(%15, %14) {coeffs = [1, 1], logic = "mul"} : index
      loom.polynomial_constraint(%12, %13, %14, %15, %16) {monomials = [{coeff = -1 : i64, vars = [4]}], upper_bound = -262144 : i64}
      %17 = loom.expression(%12, %13) {coeffs = [1, 1], logic = "add"} : index
      %18 = loom.expression(%17, %14) {coeffs = [1, 1], logic = "mul"} : index
      loom.polynomial_constraint(%12, %14, %13, %17, %18) {monomials = [{coeff = 1 : i64, vars = [4]}], upper_bound = 374784 : i64}
      loom.polynomial_constraint(%12) {monomials = [{coeff = 1 : i64, vars = [0]}], upper_bound = 128 : i64}
      loom.polynomial_constraint(%13) {monomials = [{coeff = 1 : i64, vars = [0]}], upper_bound = 128 : i64}
      %19 = loom.expression(%12, %14) {coeffs = [1, 1], logic = "mul"} : index
      %20 = loom.expression(%13, %14) {coeffs = [1, 1], logic = "mul"} : index
      %21 = loom.expression(%12, %13) {coeffs = [1, 1], logic = "mul"} : index
      %22 = loom.expression(%14, %21) {coeffs = [1, 1], logic = "mul"} : index
      loom.polynomial_constraint(%12, %13, %14, %19, %20, %21, %22) {monomials = [{coeff = 960 : i64, vars = [3]}, {coeff = 256 : i64, vars = [4]}, {coeff = -15 : i64, vars = [6]}], upper_bound = 0 : i64}
    }
    func.func @matmul_kernel__d1i0_d0i1__f10__d_h(%arg0: memref<*xf32> {tt.divisibility = 16 : i32}, %arg1: memref<*xf32> {tt.divisibility = 16 : i32}, %arg2: memref<*xf32> {tt.divisibility = 16 : i32}, %arg3: index, %arg4: index, %arg5: index, %arg6: index, %arg7: index, %arg8: index) {
      affine.parallel (%arg9) = (0) to (8) {
        affine.parallel (%arg10) = (0) to (8) {
          affine.for %arg11 = 0 to affine_map<()[s0, s1] -> (s1 ceildiv 8)>()[%arg3, %arg4] {
            affine.for %arg12 = 0 to affine_map<()[s0, s1] -> (s0 ceildiv 8)>()[%arg3, %arg4] {
              %12 = affine.apply affine_map<(d0, d1) -> (d0 * 8 + d1)>(%arg10, %arg11)
              %13 = affine.apply affine_map<(d0, d1) -> (d0 * 8 + d1)>(%arg9, %arg12)
              %c1024 = arith.constant 1024 : index
              %c1024_0 = arith.constant 1024 : index
              %c1 = arith.constant 1 : index
              %14 = loom.get_symbolic_block_size @constraints::@BM : index
              %15 = loom.get_symbolic_block_size @constraints::@BN : index
              %16 = loom.get_symbolic_block_size @constraints::@BK : index
              %17 = arith.ceildivsi %c1024_0, %16 : index
              %cst = arith.constant 0.000000e+00 : f32
              %c1024_1 = arith.constant 1024 : index
              %c0 = arith.constant 0 : index
              %18 = tensor.empty(%14, %15) : tensor<?x?xf32>
              %19 = linalg.fill ins(%cst : f32) outs(%18 : tensor<?x?xf32>) -> tensor<?x?xf32>
              %20 = arith.muli %13, %14 : index
              %21 = arith.muli %12, %15 : index
              %22 = scf.for %arg13 = %c0 to %17 step %c1 iter_args(%arg14 = %19) -> (tensor<?x?xf32>) {
                %26 = arith.muli %arg13, %16 : index
                %27 = arith.muli %20, %c1024_1 : index
                %28 = arith.addi %27, %26 : index
                %29 = loom.reinterpret_cast %arg0 to offset : [%28], sizes : [%14, %16], strides : [%c1024_0, %c1], reuse : [seq = false, spat = true, temp = true] : memref<*xf32> to memref<?x?xf32, strided<[?, ?], offset: ?>>
                %30 = loom.alloc(%14, %16) on @L1 : memref<?x?xf32>
                loom.copy %29, %30, interconnect : [], broadcast : [1, 1] : memref<?x?xf32, strided<[?, ?], offset: ?>> to memref<?x?xf32>
                %31 = bufferization.to_tensor %30 restrict writable : memref<?x?xf32> to tensor<?x?xf32>
                %32 = arith.muli %26, %c1024_1 : index
                %33 = arith.addi %32, %21 : index
                %34 = loom.reinterpret_cast %arg1 to offset : [%33], sizes : [%16, %15], strides : [%c1024, %c1], reuse : [seq = false, spat = true, temp = true] : memref<*xf32> to memref<?x?xf32, strided<[?, ?], offset: ?>>
                %35 = loom.alloc(%16, %15) on @L1 : memref<?x?xf32>
                loom.copy %34, %35, interconnect : [@horizontal_links], broadcast : [1, 8] : memref<?x?xf32, strided<[?, ?], offset: ?>> to memref<?x?xf32>
                %36 = bufferization.to_tensor %35 restrict writable : memref<?x?xf32> to tensor<?x?xf32>
                %37 = linalg.matmul ins(%31, %36 : tensor<?x?xf32>, tensor<?x?xf32>) outs(%19 : tensor<?x?xf32>) -> tensor<?x?xf32>
                %38 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%arg14, %37 : tensor<?x?xf32>, tensor<?x?xf32>) outs(%arg14 : tensor<?x?xf32>) {
                ^bb0(%in: f32, %in_2: f32, %out: f32):
                  %39 = arith.addf %in, %in_2 : f32
                  linalg.yield %39 : f32
                } -> tensor<?x?xf32>
                scf.yield %38 : tensor<?x?xf32>
              }
              %23 = arith.muli %20, %c1024_1 : index
              %24 = arith.addi %23, %21 : index
              %25 = loom.reinterpret_cast %arg2 to offset : [%24], sizes : [%14, %15], strides : [%c1024, %c1], reuse : [seq = false, spat = false, temp = false] : memref<*xf32> to memref<?x?xf32, strided<[?, ?], offset: ?>>
              bufferization.materialize_in_destination %22 in writable %25 : (tensor<?x?xf32>, memref<?x?xf32, strided<[?, ?], offset: ?>>) -> ()
            }
          }
        } {loom.mapped_to = @x}
      } {loom.mapped_to = @y}
      return
    }
  }
  module attributes {loom.pass_name = "EnumerateCopyBroadcast"} {
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
      %15 = loom.expression(%13, %12) {coeffs = [1, 1], logic = "mul"} : index
      %16 = loom.expression(%14, %15) {coeffs = [1, 1], logic = "mul"} : index
      loom.polynomial_constraint(%12, %13, %14, %15, %16) {monomials = [{coeff = -1 : i64, vars = [4]}], upper_bound = -262144 : i64}
      %17 = loom.expression(%12, %13) {coeffs = [1, 1], logic = "add"} : index
      %18 = loom.expression(%14, %17) {coeffs = [1, 1], logic = "mul"} : index
      loom.polynomial_constraint(%12, %14, %13, %17, %18) {monomials = [{coeff = 1 : i64, vars = [4]}], upper_bound = 374784 : i64}
      loom.polynomial_constraint(%12) {monomials = [{coeff = 1 : i64, vars = [0]}], upper_bound = 128 : i64}
      loom.polynomial_constraint(%13) {monomials = [{coeff = 1 : i64, vars = [0]}], upper_bound = 128 : i64}
      %19 = loom.expression(%14, %12) {coeffs = [1, 1], logic = "mul"} : index
      %20 = loom.expression(%13, %14) {coeffs = [1, 1], logic = "mul"} : index
      %21 = loom.expression(%13, %12) {coeffs = [1, 1], logic = "mul"} : index
      %22 = loom.expression(%14, %21) {coeffs = [1, 1], logic = "mul"} : index
      loom.polynomial_constraint(%12, %13, %14, %19, %20, %21, %22) {monomials = [{coeff = 256 : i64, vars = [3]}, {coeff = 960 : i64, vars = [4]}, {coeff = -15 : i64, vars = [6]}], upper_bound = 0 : i64}
    }
    func.func @matmul_kernel__d1i0_d0i1__f10__v_d(%arg0: memref<*xf32> {tt.divisibility = 16 : i32}, %arg1: memref<*xf32> {tt.divisibility = 16 : i32}, %arg2: memref<*xf32> {tt.divisibility = 16 : i32}, %arg3: index, %arg4: index, %arg5: index, %arg6: index, %arg7: index, %arg8: index) {
      affine.parallel (%arg9) = (0) to (8) {
        affine.parallel (%arg10) = (0) to (8) {
          affine.for %arg11 = 0 to affine_map<()[s0, s1] -> (s1 ceildiv 8)>()[%arg3, %arg4] {
            affine.for %arg12 = 0 to affine_map<()[s0, s1] -> (s0 ceildiv 8)>()[%arg3, %arg4] {
              %12 = affine.apply affine_map<(d0, d1) -> (d0 * 8 + d1)>(%arg10, %arg11)
              %13 = affine.apply affine_map<(d0, d1) -> (d0 * 8 + d1)>(%arg9, %arg12)
              %c1024 = arith.constant 1024 : index
              %c1024_0 = arith.constant 1024 : index
              %c1 = arith.constant 1 : index
              %14 = loom.get_symbolic_block_size @constraints::@BM : index
              %15 = loom.get_symbolic_block_size @constraints::@BN : index
              %16 = loom.get_symbolic_block_size @constraints::@BK : index
              %17 = arith.ceildivsi %c1024_0, %16 : index
              %cst = arith.constant 0.000000e+00 : f32
              %c1024_1 = arith.constant 1024 : index
              %c0 = arith.constant 0 : index
              %18 = tensor.empty(%14, %15) : tensor<?x?xf32>
              %19 = linalg.fill ins(%cst : f32) outs(%18 : tensor<?x?xf32>) -> tensor<?x?xf32>
              %20 = arith.muli %13, %14 : index
              %21 = arith.muli %12, %15 : index
              %22 = scf.for %arg13 = %c0 to %17 step %c1 iter_args(%arg14 = %19) -> (tensor<?x?xf32>) {
                %26 = arith.muli %arg13, %16 : index
                %27 = arith.muli %20, %c1024_1 : index
                %28 = arith.addi %27, %26 : index
                %29 = loom.reinterpret_cast %arg0 to offset : [%28], sizes : [%14, %16], strides : [%c1024_0, %c1], reuse : [seq = false, spat = true, temp = true] : memref<*xf32> to memref<?x?xf32, strided<[?, ?], offset: ?>>
                %30 = loom.alloc(%14, %16) on @L1 : memref<?x?xf32>
                loom.copy %29, %30, interconnect : [@vertical_links], broadcast : [8, 1] : memref<?x?xf32, strided<[?, ?], offset: ?>> to memref<?x?xf32>
                %31 = bufferization.to_tensor %30 restrict writable : memref<?x?xf32> to tensor<?x?xf32>
                %32 = arith.muli %26, %c1024_1 : index
                %33 = arith.addi %32, %21 : index
                %34 = loom.reinterpret_cast %arg1 to offset : [%33], sizes : [%16, %15], strides : [%c1024, %c1], reuse : [seq = false, spat = true, temp = true] : memref<*xf32> to memref<?x?xf32, strided<[?, ?], offset: ?>>
                %35 = loom.alloc(%16, %15) on @L1 : memref<?x?xf32>
                loom.copy %34, %35, interconnect : [], broadcast : [1, 1] : memref<?x?xf32, strided<[?, ?], offset: ?>> to memref<?x?xf32>
                %36 = bufferization.to_tensor %35 restrict writable : memref<?x?xf32> to tensor<?x?xf32>
                %37 = linalg.matmul ins(%31, %36 : tensor<?x?xf32>, tensor<?x?xf32>) outs(%19 : tensor<?x?xf32>) -> tensor<?x?xf32>
                %38 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%arg14, %37 : tensor<?x?xf32>, tensor<?x?xf32>) outs(%arg14 : tensor<?x?xf32>) {
                ^bb0(%in: f32, %in_2: f32, %out: f32):
                  %39 = arith.addf %in, %in_2 : f32
                  linalg.yield %39 : f32
                } -> tensor<?x?xf32>
                scf.yield %38 : tensor<?x?xf32>
              }
              %23 = arith.muli %20, %c1024_1 : index
              %24 = arith.addi %23, %21 : index
              %25 = loom.reinterpret_cast %arg2 to offset : [%24], sizes : [%14, %15], strides : [%c1024, %c1], reuse : [seq = false, spat = false, temp = false] : memref<*xf32> to memref<?x?xf32, strided<[?, ?], offset: ?>>
              bufferization.materialize_in_destination %22 in writable %25 : (tensor<?x?xf32>, memref<?x?xf32, strided<[?, ?], offset: ?>>) -> ()
            }
          }
        } {loom.mapped_to = @x}
      } {loom.mapped_to = @y}
      return
    }
  }
  module attributes {loom.pass_name = "EnumerateCopyBroadcast"} {
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
      %15 = loom.expression(%12, %13) {coeffs = [1, 1], logic = "mul"} : index
      %16 = loom.expression(%15, %14) {coeffs = [1, 1], logic = "mul"} : index
      loom.polynomial_constraint(%12, %13, %14, %15, %16) {monomials = [{coeff = -1 : i64, vars = [4]}], upper_bound = -262144 : i64}
      %17 = loom.expression(%12, %13) {coeffs = [1, 1], logic = "add"} : index
      %18 = loom.expression(%17, %14) {coeffs = [1, 1], logic = "mul"} : index
      loom.polynomial_constraint(%12, %14, %13, %17, %18) {monomials = [{coeff = 1 : i64, vars = [4]}], upper_bound = 374784 : i64}
      loom.polynomial_constraint(%12) {monomials = [{coeff = 1 : i64, vars = [0]}], upper_bound = 128 : i64}
      loom.polynomial_constraint(%13) {monomials = [{coeff = 1 : i64, vars = [0]}], upper_bound = 128 : i64}
      %19 = loom.expression(%12, %14) {coeffs = [1, 1], logic = "mul"} : index
      %20 = loom.expression(%13, %14) {coeffs = [1, 1], logic = "mul"} : index
      %21 = loom.expression(%12, %13) {coeffs = [1, 1], logic = "mul"} : index
      %22 = loom.expression(%14, %21) {coeffs = [1, 1], logic = "mul"} : index
      loom.polynomial_constraint(%12, %13, %14, %19, %20, %21, %22) {monomials = [{coeff = 256 : i64, vars = [3]}, {coeff = 256 : i64, vars = [4]}, {coeff = -15 : i64, vars = [6]}], upper_bound = 0 : i64}
    }
    func.func @matmul_kernel__d1i0_d0i1__f10__v_h(%arg0: memref<*xf32> {tt.divisibility = 16 : i32}, %arg1: memref<*xf32> {tt.divisibility = 16 : i32}, %arg2: memref<*xf32> {tt.divisibility = 16 : i32}, %arg3: index, %arg4: index, %arg5: index, %arg6: index, %arg7: index, %arg8: index) {
      affine.parallel (%arg9) = (0) to (8) {
        affine.parallel (%arg10) = (0) to (8) {
          affine.for %arg11 = 0 to affine_map<()[s0, s1] -> (s1 ceildiv 8)>()[%arg3, %arg4] {
            affine.for %arg12 = 0 to affine_map<()[s0, s1] -> (s0 ceildiv 8)>()[%arg3, %arg4] {
              %12 = affine.apply affine_map<(d0, d1) -> (d0 * 8 + d1)>(%arg10, %arg11)
              %13 = affine.apply affine_map<(d0, d1) -> (d0 * 8 + d1)>(%arg9, %arg12)
              %c1024 = arith.constant 1024 : index
              %c1024_0 = arith.constant 1024 : index
              %c1 = arith.constant 1 : index
              %14 = loom.get_symbolic_block_size @constraints::@BM : index
              %15 = loom.get_symbolic_block_size @constraints::@BN : index
              %16 = loom.get_symbolic_block_size @constraints::@BK : index
              %17 = arith.ceildivsi %c1024_0, %16 : index
              %cst = arith.constant 0.000000e+00 : f32
              %c1024_1 = arith.constant 1024 : index
              %c0 = arith.constant 0 : index
              %18 = tensor.empty(%14, %15) : tensor<?x?xf32>
              %19 = linalg.fill ins(%cst : f32) outs(%18 : tensor<?x?xf32>) -> tensor<?x?xf32>
              %20 = arith.muli %13, %14 : index
              %21 = arith.muli %12, %15 : index
              %22 = scf.for %arg13 = %c0 to %17 step %c1 iter_args(%arg14 = %19) -> (tensor<?x?xf32>) {
                %26 = arith.muli %arg13, %16 : index
                %27 = arith.muli %20, %c1024_1 : index
                %28 = arith.addi %27, %26 : index
                %29 = loom.reinterpret_cast %arg0 to offset : [%28], sizes : [%14, %16], strides : [%c1024_0, %c1], reuse : [seq = false, spat = true, temp = true] : memref<*xf32> to memref<?x?xf32, strided<[?, ?], offset: ?>>
                %30 = loom.alloc(%14, %16) on @L1 : memref<?x?xf32>
                loom.copy %29, %30, interconnect : [@vertical_links], broadcast : [8, 1] : memref<?x?xf32, strided<[?, ?], offset: ?>> to memref<?x?xf32>
                %31 = bufferization.to_tensor %30 restrict writable : memref<?x?xf32> to tensor<?x?xf32>
                %32 = arith.muli %26, %c1024_1 : index
                %33 = arith.addi %32, %21 : index
                %34 = loom.reinterpret_cast %arg1 to offset : [%33], sizes : [%16, %15], strides : [%c1024, %c1], reuse : [seq = false, spat = true, temp = true] : memref<*xf32> to memref<?x?xf32, strided<[?, ?], offset: ?>>
                %35 = loom.alloc(%16, %15) on @L1 : memref<?x?xf32>
                loom.copy %34, %35, interconnect : [@horizontal_links], broadcast : [1, 8] : memref<?x?xf32, strided<[?, ?], offset: ?>> to memref<?x?xf32>
                %36 = bufferization.to_tensor %35 restrict writable : memref<?x?xf32> to tensor<?x?xf32>
                %37 = linalg.matmul ins(%31, %36 : tensor<?x?xf32>, tensor<?x?xf32>) outs(%19 : tensor<?x?xf32>) -> tensor<?x?xf32>
                %38 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%arg14, %37 : tensor<?x?xf32>, tensor<?x?xf32>) outs(%arg14 : tensor<?x?xf32>) {
                ^bb0(%in: f32, %in_2: f32, %out: f32):
                  %39 = arith.addf %in, %in_2 : f32
                  linalg.yield %39 : f32
                } -> tensor<?x?xf32>
                scf.yield %38 : tensor<?x?xf32>
              }
              %23 = arith.muli %20, %c1024_1 : index
              %24 = arith.addi %23, %21 : index
              %25 = loom.reinterpret_cast %arg2 to offset : [%24], sizes : [%14, %15], strides : [%c1024, %c1], reuse : [seq = false, spat = false, temp = false] : memref<*xf32> to memref<?x?xf32, strided<[?, ?], offset: ?>>
              bufferization.materialize_in_destination %22 in writable %25 : (tensor<?x?xf32>, memref<?x?xf32, strided<[?, ?], offset: ?>>) -> ()
            }
          }
        } {loom.mapped_to = @x}
      } {loom.mapped_to = @y}
      return
    }
  }
  module attributes {loom.pass_name = "EnumerateCopyBroadcast"} {
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
      %15 = loom.expression(%12, %13) {coeffs = [1, 1], logic = "mul"} : index
      %16 = loom.expression(%15, %14) {coeffs = [1, 1], logic = "mul"} : index
      loom.polynomial_constraint(%12, %13, %14, %15, %16) {monomials = [{coeff = -1 : i64, vars = [4]}], upper_bound = -262144 : i64}
      %17 = loom.expression(%12, %13) {coeffs = [1, 1], logic = "add"} : index
      %18 = loom.expression(%17, %14) {coeffs = [1, 1], logic = "mul"} : index
      loom.polynomial_constraint(%12, %14, %13, %17, %18) {monomials = [{coeff = 1 : i64, vars = [4]}], upper_bound = 374784 : i64}
      loom.polynomial_constraint(%12) {monomials = [{coeff = 1 : i64, vars = [0]}], upper_bound = 128 : i64}
      loom.polynomial_constraint(%13) {monomials = [{coeff = 1 : i64, vars = [0]}], upper_bound = 128 : i64}
      %19 = loom.expression(%13, %14) {coeffs = [1, 1], logic = "mul"} : index
      loom.polynomial_constraint(%12, %14, %13, %19) {monomials = [{coeff = 1024 : i64, vars = [0]}, {coeff = 1 : i64, vars = [3]}], upper_bound = 374784 : i64}
      %20 = loom.expression(%12, %14) {coeffs = [1, 1], logic = "mul"} : index
      %21 = loom.expression(%13, %14) {coeffs = [1, 1], logic = "mul"} : index
      %22 = loom.expression(%12, %13) {coeffs = [1, 1], logic = "mul"} : index
      %23 = loom.expression(%14, %22) {coeffs = [1, 1], logic = "mul"} : index
      loom.polynomial_constraint(%12, %13, %14, %20, %21, %22, %23) {monomials = [{coeff = 64 : i64, vars = [3]}, {coeff = 64 : i64, vars = [4]}, {coeff = -1 : i64, vars = [6]}], upper_bound = 0 : i64}
    }
    func.func @matmul_kernel__d1i0_d0i1__f10__hoist_block_0__d_d(%arg0: memref<*xf32> {tt.divisibility = 16 : i32}, %arg1: memref<*xf32> {tt.divisibility = 16 : i32}, %arg2: memref<*xf32> {tt.divisibility = 16 : i32}, %arg3: index, %arg4: index, %arg5: index, %arg6: index, %arg7: index, %arg8: index) {
      affine.parallel (%arg9) = (0) to (8) {
        affine.parallel (%arg10) = (0) to (8) {
          affine.for %arg11 = 0 to affine_map<()[s0, s1] -> (s1 ceildiv 8)>()[%arg3, %arg4] {
            affine.for %arg12 = 0 to affine_map<()[s0, s1] -> (s0 ceildiv 8)>()[%arg3, %arg4] {
              %12 = affine.apply affine_map<(d0, d1) -> (d0 * 8 + d1)>(%arg10, %arg11)
              %13 = affine.apply affine_map<(d0, d1) -> (d0 * 8 + d1)>(%arg9, %arg12)
              %c1024 = arith.constant 1024 : index
              %c1024_0 = arith.constant 1024 : index
              %c1 = arith.constant 1 : index
              %14 = loom.get_symbolic_block_size @constraints::@BM : index
              %15 = loom.get_symbolic_block_size @constraints::@BN : index
              %16 = loom.get_symbolic_block_size @constraints::@BK : index
              %17 = arith.ceildivsi %c1024_0, %16 : index
              %cst = arith.constant 0.000000e+00 : f32
              %c1024_1 = arith.constant 1024 : index
              %c0 = arith.constant 0 : index
              %18 = tensor.empty(%14, %15) : tensor<?x?xf32>
              %19 = linalg.fill ins(%cst : f32) outs(%18 : tensor<?x?xf32>) -> tensor<?x?xf32>
              %20 = arith.muli %13, %14 : index
              %21 = arith.muli %12, %15 : index
              %22 = arith.muli %14, %c1024_0 : index
              %c0_2 = arith.constant 0 : index
              %23 = arith.muli %20, %c1024_1 : index
              %24 = arith.muli %c0_2, %16 : index
              %25 = arith.addi %23, %24 : index
              %26 = loom.reinterpret_cast %arg0 to offset : [%25], sizes : [%17, %14, %16], strides : [%22, %c1024_0, %c1], reuse : [seq = false, spat = true, temp = true] : memref<*xf32> to memref<?x?x?xf32, strided<[?, ?, ?], offset: ?>>
              %27 = loom.alloc(%17, %14, %16) on @L1 : memref<?x?x?xf32>
              loom.copy %26, %27, interconnect : [], broadcast : [1, 1] : memref<?x?x?xf32, strided<[?, ?, ?], offset: ?>> to memref<?x?x?xf32>
              %28 = scf.for %arg13 = %c0 to %17 step %c1 iter_args(%arg14 = %19) -> (tensor<?x?xf32>) {
                %32 = arith.muli %arg13, %16 : index
                %33 = arith.muli %20, %c1024_1 : index
                %subview = memref.subview %27[%arg13, 0, 0] [1, %14, %16] [1, 1, 1] : memref<?x?x?xf32> to memref<?x?xf32, strided<[?, 1], offset: ?>>
                %34 = bufferization.to_tensor %subview restrict writable : memref<?x?xf32, strided<[?, 1], offset: ?>> to tensor<?x?xf32>
                %35 = arith.muli %32, %c1024_1 : index
                %36 = arith.addi %35, %21 : index
                %37 = loom.reinterpret_cast %arg1 to offset : [%36], sizes : [%16, %15], strides : [%c1024, %c1], reuse : [seq = false, spat = true, temp = true] : memref<*xf32> to memref<?x?xf32, strided<[?, ?], offset: ?>>
                %38 = loom.alloc(%16, %15) on @L1 : memref<?x?xf32>
                loom.copy %37, %38, interconnect : [], broadcast : [1, 1] : memref<?x?xf32, strided<[?, ?], offset: ?>> to memref<?x?xf32>
                %39 = bufferization.to_tensor %38 restrict writable : memref<?x?xf32> to tensor<?x?xf32>
                %40 = linalg.matmul ins(%34, %39 : tensor<?x?xf32>, tensor<?x?xf32>) outs(%19 : tensor<?x?xf32>) -> tensor<?x?xf32>
                %41 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%arg14, %40 : tensor<?x?xf32>, tensor<?x?xf32>) outs(%arg14 : tensor<?x?xf32>) {
                ^bb0(%in: f32, %in_3: f32, %out: f32):
                  %42 = arith.addf %in, %in_3 : f32
                  linalg.yield %42 : f32
                } -> tensor<?x?xf32>
                scf.yield %41 : tensor<?x?xf32>
              }
              %29 = arith.muli %20, %c1024_1 : index
              %30 = arith.addi %29, %21 : index
              %31 = loom.reinterpret_cast %arg2 to offset : [%30], sizes : [%14, %15], strides : [%c1024, %c1], reuse : [seq = false, spat = false, temp = false] : memref<*xf32> to memref<?x?xf32, strided<[?, ?], offset: ?>>
              bufferization.materialize_in_destination %28 in writable %31 : (tensor<?x?xf32>, memref<?x?xf32, strided<[?, ?], offset: ?>>) -> ()
            }
          }
        } {loom.mapped_to = @x}
      } {loom.mapped_to = @y}
      return
    }
  }
  module attributes {loom.pass_name = "EnumerateCopyBroadcast"} {
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
      %15 = loom.expression(%13, %12) {coeffs = [1, 1], logic = "mul"} : index
      %16 = loom.expression(%15, %14) {coeffs = [1, 1], logic = "mul"} : index
      loom.polynomial_constraint(%12, %13, %14, %15, %16) {monomials = [{coeff = -1 : i64, vars = [4]}], upper_bound = -262144 : i64}
      %17 = loom.expression(%12, %13) {coeffs = [1, 1], logic = "add"} : index
      %18 = loom.expression(%14, %17) {coeffs = [1, 1], logic = "mul"} : index
      loom.polynomial_constraint(%12, %14, %13, %17, %18) {monomials = [{coeff = 1 : i64, vars = [4]}], upper_bound = 374784 : i64}
      loom.polynomial_constraint(%12) {monomials = [{coeff = 1 : i64, vars = [0]}], upper_bound = 128 : i64}
      loom.polynomial_constraint(%13) {monomials = [{coeff = 1 : i64, vars = [0]}], upper_bound = 128 : i64}
      %19 = loom.expression(%13, %14) {coeffs = [1, 1], logic = "mul"} : index
      loom.polynomial_constraint(%12, %14, %13, %19) {monomials = [{coeff = 1024 : i64, vars = [0]}, {coeff = 1 : i64, vars = [3]}], upper_bound = 374784 : i64}
      %20 = loom.expression(%14, %12) {coeffs = [1, 1], logic = "mul"} : index
      %21 = loom.expression(%13, %14) {coeffs = [1, 1], logic = "mul"} : index
      %22 = loom.expression(%13, %12) {coeffs = [1, 1], logic = "mul"} : index
      %23 = loom.expression(%14, %22) {coeffs = [1, 1], logic = "mul"} : index
      loom.polynomial_constraint(%12, %13, %14, %20, %21, %22, %23) {monomials = [{coeff = 960 : i64, vars = [3]}, {coeff = 256 : i64, vars = [4]}, {coeff = -15 : i64, vars = [6]}], upper_bound = 0 : i64}
    }
    func.func @matmul_kernel__d1i0_d0i1__f10__hoist_block_0__d_h(%arg0: memref<*xf32> {tt.divisibility = 16 : i32}, %arg1: memref<*xf32> {tt.divisibility = 16 : i32}, %arg2: memref<*xf32> {tt.divisibility = 16 : i32}, %arg3: index, %arg4: index, %arg5: index, %arg6: index, %arg7: index, %arg8: index) {
      affine.parallel (%arg9) = (0) to (8) {
        affine.parallel (%arg10) = (0) to (8) {
          affine.for %arg11 = 0 to affine_map<()[s0, s1] -> (s1 ceildiv 8)>()[%arg3, %arg4] {
            affine.for %arg12 = 0 to affine_map<()[s0, s1] -> (s0 ceildiv 8)>()[%arg3, %arg4] {
              %12 = affine.apply affine_map<(d0, d1) -> (d0 * 8 + d1)>(%arg10, %arg11)
              %13 = affine.apply affine_map<(d0, d1) -> (d0 * 8 + d1)>(%arg9, %arg12)
              %c1024 = arith.constant 1024 : index
              %c1024_0 = arith.constant 1024 : index
              %c1 = arith.constant 1 : index
              %14 = loom.get_symbolic_block_size @constraints::@BM : index
              %15 = loom.get_symbolic_block_size @constraints::@BN : index
              %16 = loom.get_symbolic_block_size @constraints::@BK : index
              %17 = arith.ceildivsi %c1024_0, %16 : index
              %cst = arith.constant 0.000000e+00 : f32
              %c1024_1 = arith.constant 1024 : index
              %c0 = arith.constant 0 : index
              %18 = tensor.empty(%14, %15) : tensor<?x?xf32>
              %19 = linalg.fill ins(%cst : f32) outs(%18 : tensor<?x?xf32>) -> tensor<?x?xf32>
              %20 = arith.muli %13, %14 : index
              %21 = arith.muli %12, %15 : index
              %22 = arith.muli %14, %c1024_0 : index
              %c0_2 = arith.constant 0 : index
              %23 = arith.muli %20, %c1024_1 : index
              %24 = arith.muli %c0_2, %16 : index
              %25 = arith.addi %23, %24 : index
              %26 = loom.reinterpret_cast %arg0 to offset : [%25], sizes : [%17, %14, %16], strides : [%22, %c1024_0, %c1], reuse : [seq = false, spat = true, temp = true] : memref<*xf32> to memref<?x?x?xf32, strided<[?, ?, ?], offset: ?>>
              %27 = loom.alloc(%17, %14, %16) on @L1 : memref<?x?x?xf32>
              loom.copy %26, %27, interconnect : [], broadcast : [1, 1] : memref<?x?x?xf32, strided<[?, ?, ?], offset: ?>> to memref<?x?x?xf32>
              %28 = scf.for %arg13 = %c0 to %17 step %c1 iter_args(%arg14 = %19) -> (tensor<?x?xf32>) {
                %32 = arith.muli %arg13, %16 : index
                %33 = arith.muli %20, %c1024_1 : index
                %subview = memref.subview %27[%arg13, 0, 0] [1, %14, %16] [1, 1, 1] : memref<?x?x?xf32> to memref<?x?xf32, strided<[?, 1], offset: ?>>
                %34 = bufferization.to_tensor %subview restrict writable : memref<?x?xf32, strided<[?, 1], offset: ?>> to tensor<?x?xf32>
                %35 = arith.muli %32, %c1024_1 : index
                %36 = arith.addi %35, %21 : index
                %37 = loom.reinterpret_cast %arg1 to offset : [%36], sizes : [%16, %15], strides : [%c1024, %c1], reuse : [seq = false, spat = true, temp = true] : memref<*xf32> to memref<?x?xf32, strided<[?, ?], offset: ?>>
                %38 = loom.alloc(%16, %15) on @L1 : memref<?x?xf32>
                loom.copy %37, %38, interconnect : [@horizontal_links], broadcast : [1, 8] : memref<?x?xf32, strided<[?, ?], offset: ?>> to memref<?x?xf32>
                %39 = bufferization.to_tensor %38 restrict writable : memref<?x?xf32> to tensor<?x?xf32>
                %40 = linalg.matmul ins(%34, %39 : tensor<?x?xf32>, tensor<?x?xf32>) outs(%19 : tensor<?x?xf32>) -> tensor<?x?xf32>
                %41 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%arg14, %40 : tensor<?x?xf32>, tensor<?x?xf32>) outs(%arg14 : tensor<?x?xf32>) {
                ^bb0(%in: f32, %in_3: f32, %out: f32):
                  %42 = arith.addf %in, %in_3 : f32
                  linalg.yield %42 : f32
                } -> tensor<?x?xf32>
                scf.yield %41 : tensor<?x?xf32>
              }
              %29 = arith.muli %20, %c1024_1 : index
              %30 = arith.addi %29, %21 : index
              %31 = loom.reinterpret_cast %arg2 to offset : [%30], sizes : [%14, %15], strides : [%c1024, %c1], reuse : [seq = false, spat = false, temp = false] : memref<*xf32> to memref<?x?xf32, strided<[?, ?], offset: ?>>
              bufferization.materialize_in_destination %28 in writable %31 : (tensor<?x?xf32>, memref<?x?xf32, strided<[?, ?], offset: ?>>) -> ()
            }
          }
        } {loom.mapped_to = @x}
      } {loom.mapped_to = @y}
      return
    }
  }
  module attributes {loom.pass_name = "EnumerateCopyBroadcast"} {
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
      %15 = loom.expression(%13, %12) {coeffs = [1, 1], logic = "mul"} : index
      %16 = loom.expression(%15, %14) {coeffs = [1, 1], logic = "mul"} : index
      loom.polynomial_constraint(%12, %13, %14, %15, %16) {monomials = [{coeff = -1 : i64, vars = [4]}], upper_bound = -262144 : i64}
      %17 = loom.expression(%12, %13) {coeffs = [1, 1], logic = "add"} : index
      %18 = loom.expression(%14, %17) {coeffs = [1, 1], logic = "mul"} : index
      loom.polynomial_constraint(%12, %14, %13, %17, %18) {monomials = [{coeff = 1 : i64, vars = [4]}], upper_bound = 374784 : i64}
      loom.polynomial_constraint(%12) {monomials = [{coeff = 1 : i64, vars = [0]}], upper_bound = 128 : i64}
      loom.polynomial_constraint(%13) {monomials = [{coeff = 1 : i64, vars = [0]}], upper_bound = 128 : i64}
      %19 = loom.expression(%13, %14) {coeffs = [1, 1], logic = "mul"} : index
      loom.polynomial_constraint(%12, %14, %13, %19) {monomials = [{coeff = 1024 : i64, vars = [0]}, {coeff = 1 : i64, vars = [3]}], upper_bound = 374784 : i64}
      %20 = loom.expression(%14, %12) {coeffs = [1, 1], logic = "mul"} : index
      %21 = loom.expression(%13, %14) {coeffs = [1, 1], logic = "mul"} : index
      %22 = loom.expression(%13, %12) {coeffs = [1, 1], logic = "mul"} : index
      %23 = loom.expression(%14, %22) {coeffs = [1, 1], logic = "mul"} : index
      loom.polynomial_constraint(%12, %13, %14, %20, %21, %22, %23) {monomials = [{coeff = 256 : i64, vars = [3]}, {coeff = 960 : i64, vars = [4]}, {coeff = -15 : i64, vars = [6]}], upper_bound = 0 : i64}
    }
    func.func @matmul_kernel__d1i0_d0i1__f10__hoist_block_0__v_d(%arg0: memref<*xf32> {tt.divisibility = 16 : i32}, %arg1: memref<*xf32> {tt.divisibility = 16 : i32}, %arg2: memref<*xf32> {tt.divisibility = 16 : i32}, %arg3: index, %arg4: index, %arg5: index, %arg6: index, %arg7: index, %arg8: index) {
      affine.parallel (%arg9) = (0) to (8) {
        affine.parallel (%arg10) = (0) to (8) {
          affine.for %arg11 = 0 to affine_map<()[s0, s1] -> (s1 ceildiv 8)>()[%arg3, %arg4] {
            affine.for %arg12 = 0 to affine_map<()[s0, s1] -> (s0 ceildiv 8)>()[%arg3, %arg4] {
              %12 = affine.apply affine_map<(d0, d1) -> (d0 * 8 + d1)>(%arg10, %arg11)
              %13 = affine.apply affine_map<(d0, d1) -> (d0 * 8 + d1)>(%arg9, %arg12)
              %c1024 = arith.constant 1024 : index
              %c1024_0 = arith.constant 1024 : index
              %c1 = arith.constant 1 : index
              %14 = loom.get_symbolic_block_size @constraints::@BM : index
              %15 = loom.get_symbolic_block_size @constraints::@BN : index
              %16 = loom.get_symbolic_block_size @constraints::@BK : index
              %17 = arith.ceildivsi %c1024_0, %16 : index
              %cst = arith.constant 0.000000e+00 : f32
              %c1024_1 = arith.constant 1024 : index
              %c0 = arith.constant 0 : index
              %18 = tensor.empty(%14, %15) : tensor<?x?xf32>
              %19 = linalg.fill ins(%cst : f32) outs(%18 : tensor<?x?xf32>) -> tensor<?x?xf32>
              %20 = arith.muli %13, %14 : index
              %21 = arith.muli %12, %15 : index
              %22 = arith.muli %14, %c1024_0 : index
              %c0_2 = arith.constant 0 : index
              %23 = arith.muli %20, %c1024_1 : index
              %24 = arith.muli %c0_2, %16 : index
              %25 = arith.addi %23, %24 : index
              %26 = loom.reinterpret_cast %arg0 to offset : [%25], sizes : [%17, %14, %16], strides : [%22, %c1024_0, %c1], reuse : [seq = false, spat = true, temp = true] : memref<*xf32> to memref<?x?x?xf32, strided<[?, ?, ?], offset: ?>>
              %27 = loom.alloc(%17, %14, %16) on @L1 : memref<?x?x?xf32>
              loom.copy %26, %27, interconnect : [@vertical_links], broadcast : [8, 1] : memref<?x?x?xf32, strided<[?, ?, ?], offset: ?>> to memref<?x?x?xf32>
              %28 = scf.for %arg13 = %c0 to %17 step %c1 iter_args(%arg14 = %19) -> (tensor<?x?xf32>) {
                %32 = arith.muli %arg13, %16 : index
                %33 = arith.muli %20, %c1024_1 : index
                %subview = memref.subview %27[%arg13, 0, 0] [1, %14, %16] [1, 1, 1] : memref<?x?x?xf32> to memref<?x?xf32, strided<[?, 1], offset: ?>>
                %34 = bufferization.to_tensor %subview restrict writable : memref<?x?xf32, strided<[?, 1], offset: ?>> to tensor<?x?xf32>
                %35 = arith.muli %32, %c1024_1 : index
                %36 = arith.addi %35, %21 : index
                %37 = loom.reinterpret_cast %arg1 to offset : [%36], sizes : [%16, %15], strides : [%c1024, %c1], reuse : [seq = false, spat = true, temp = true] : memref<*xf32> to memref<?x?xf32, strided<[?, ?], offset: ?>>
                %38 = loom.alloc(%16, %15) on @L1 : memref<?x?xf32>
                loom.copy %37, %38, interconnect : [], broadcast : [1, 1] : memref<?x?xf32, strided<[?, ?], offset: ?>> to memref<?x?xf32>
                %39 = bufferization.to_tensor %38 restrict writable : memref<?x?xf32> to tensor<?x?xf32>
                %40 = linalg.matmul ins(%34, %39 : tensor<?x?xf32>, tensor<?x?xf32>) outs(%19 : tensor<?x?xf32>) -> tensor<?x?xf32>
                %41 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%arg14, %40 : tensor<?x?xf32>, tensor<?x?xf32>) outs(%arg14 : tensor<?x?xf32>) {
                ^bb0(%in: f32, %in_3: f32, %out: f32):
                  %42 = arith.addf %in, %in_3 : f32
                  linalg.yield %42 : f32
                } -> tensor<?x?xf32>
                scf.yield %41 : tensor<?x?xf32>
              }
              %29 = arith.muli %20, %c1024_1 : index
              %30 = arith.addi %29, %21 : index
              %31 = loom.reinterpret_cast %arg2 to offset : [%30], sizes : [%14, %15], strides : [%c1024, %c1], reuse : [seq = false, spat = false, temp = false] : memref<*xf32> to memref<?x?xf32, strided<[?, ?], offset: ?>>
              bufferization.materialize_in_destination %28 in writable %31 : (tensor<?x?xf32>, memref<?x?xf32, strided<[?, ?], offset: ?>>) -> ()
            }
          }
        } {loom.mapped_to = @x}
      } {loom.mapped_to = @y}
      return
    }
  }
  module attributes {loom.pass_name = "EnumerateCopyBroadcast"} {
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
      %15 = loom.expression(%12, %13) {coeffs = [1, 1], logic = "mul"} : index
      %16 = loom.expression(%15, %14) {coeffs = [1, 1], logic = "mul"} : index
      loom.polynomial_constraint(%12, %13, %14, %15, %16) {monomials = [{coeff = -1 : i64, vars = [4]}], upper_bound = -262144 : i64}
      %17 = loom.expression(%12, %13) {coeffs = [1, 1], logic = "add"} : index
      %18 = loom.expression(%14, %17) {coeffs = [1, 1], logic = "mul"} : index
      loom.polynomial_constraint(%12, %14, %13, %17, %18) {monomials = [{coeff = 1 : i64, vars = [4]}], upper_bound = 374784 : i64}
      loom.polynomial_constraint(%12) {monomials = [{coeff = 1 : i64, vars = [0]}], upper_bound = 128 : i64}
      loom.polynomial_constraint(%13) {monomials = [{coeff = 1 : i64, vars = [0]}], upper_bound = 128 : i64}
      %19 = loom.expression(%14, %13) {coeffs = [1, 1], logic = "mul"} : index
      loom.polynomial_constraint(%12, %14, %13, %19) {monomials = [{coeff = 1024 : i64, vars = [0]}, {coeff = 1 : i64, vars = [3]}], upper_bound = 374784 : i64}
      %20 = loom.expression(%14, %12) {coeffs = [1, 1], logic = "mul"} : index
      %21 = loom.expression(%14, %13) {coeffs = [1, 1], logic = "mul"} : index
      %22 = loom.expression(%12, %13) {coeffs = [1, 1], logic = "mul"} : index
      %23 = loom.expression(%14, %22) {coeffs = [1, 1], logic = "mul"} : index
      loom.polynomial_constraint(%12, %13, %14, %20, %21, %22, %23) {monomials = [{coeff = 256 : i64, vars = [3]}, {coeff = 256 : i64, vars = [4]}, {coeff = -15 : i64, vars = [6]}], upper_bound = 0 : i64}
    }
    func.func @matmul_kernel__d1i0_d0i1__f10__hoist_block_0__v_h(%arg0: memref<*xf32> {tt.divisibility = 16 : i32}, %arg1: memref<*xf32> {tt.divisibility = 16 : i32}, %arg2: memref<*xf32> {tt.divisibility = 16 : i32}, %arg3: index, %arg4: index, %arg5: index, %arg6: index, %arg7: index, %arg8: index) {
      affine.parallel (%arg9) = (0) to (8) {
        affine.parallel (%arg10) = (0) to (8) {
          affine.for %arg11 = 0 to affine_map<()[s0, s1] -> (s1 ceildiv 8)>()[%arg3, %arg4] {
            affine.for %arg12 = 0 to affine_map<()[s0, s1] -> (s0 ceildiv 8)>()[%arg3, %arg4] {
              %12 = affine.apply affine_map<(d0, d1) -> (d0 * 8 + d1)>(%arg10, %arg11)
              %13 = affine.apply affine_map<(d0, d1) -> (d0 * 8 + d1)>(%arg9, %arg12)
              %c1024 = arith.constant 1024 : index
              %c1024_0 = arith.constant 1024 : index
              %c1 = arith.constant 1 : index
              %14 = loom.get_symbolic_block_size @constraints::@BM : index
              %15 = loom.get_symbolic_block_size @constraints::@BN : index
              %16 = loom.get_symbolic_block_size @constraints::@BK : index
              %17 = arith.ceildivsi %c1024_0, %16 : index
              %cst = arith.constant 0.000000e+00 : f32
              %c1024_1 = arith.constant 1024 : index
              %c0 = arith.constant 0 : index
              %18 = tensor.empty(%14, %15) : tensor<?x?xf32>
              %19 = linalg.fill ins(%cst : f32) outs(%18 : tensor<?x?xf32>) -> tensor<?x?xf32>
              %20 = arith.muli %13, %14 : index
              %21 = arith.muli %12, %15 : index
              %22 = arith.muli %14, %c1024_0 : index
              %c0_2 = arith.constant 0 : index
              %23 = arith.muli %20, %c1024_1 : index
              %24 = arith.muli %c0_2, %16 : index
              %25 = arith.addi %23, %24 : index
              %26 = loom.reinterpret_cast %arg0 to offset : [%25], sizes : [%17, %14, %16], strides : [%22, %c1024_0, %c1], reuse : [seq = false, spat = true, temp = true] : memref<*xf32> to memref<?x?x?xf32, strided<[?, ?, ?], offset: ?>>
              %27 = loom.alloc(%17, %14, %16) on @L1 : memref<?x?x?xf32>
              loom.copy %26, %27, interconnect : [@vertical_links], broadcast : [8, 1] : memref<?x?x?xf32, strided<[?, ?, ?], offset: ?>> to memref<?x?x?xf32>
              %28 = scf.for %arg13 = %c0 to %17 step %c1 iter_args(%arg14 = %19) -> (tensor<?x?xf32>) {
                %32 = arith.muli %arg13, %16 : index
                %33 = arith.muli %20, %c1024_1 : index
                %subview = memref.subview %27[%arg13, 0, 0] [1, %14, %16] [1, 1, 1] : memref<?x?x?xf32> to memref<?x?xf32, strided<[?, 1], offset: ?>>
                %34 = bufferization.to_tensor %subview restrict writable : memref<?x?xf32, strided<[?, 1], offset: ?>> to tensor<?x?xf32>
                %35 = arith.muli %32, %c1024_1 : index
                %36 = arith.addi %35, %21 : index
                %37 = loom.reinterpret_cast %arg1 to offset : [%36], sizes : [%16, %15], strides : [%c1024, %c1], reuse : [seq = false, spat = true, temp = true] : memref<*xf32> to memref<?x?xf32, strided<[?, ?], offset: ?>>
                %38 = loom.alloc(%16, %15) on @L1 : memref<?x?xf32>
                loom.copy %37, %38, interconnect : [@horizontal_links], broadcast : [1, 8] : memref<?x?xf32, strided<[?, ?], offset: ?>> to memref<?x?xf32>
                %39 = bufferization.to_tensor %38 restrict writable : memref<?x?xf32> to tensor<?x?xf32>
                %40 = linalg.matmul ins(%34, %39 : tensor<?x?xf32>, tensor<?x?xf32>) outs(%19 : tensor<?x?xf32>) -> tensor<?x?xf32>
                %41 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%arg14, %40 : tensor<?x?xf32>, tensor<?x?xf32>) outs(%arg14 : tensor<?x?xf32>) {
                ^bb0(%in: f32, %in_3: f32, %out: f32):
                  %42 = arith.addf %in, %in_3 : f32
                  linalg.yield %42 : f32
                } -> tensor<?x?xf32>
                scf.yield %41 : tensor<?x?xf32>
              }
              %29 = arith.muli %20, %c1024_1 : index
              %30 = arith.addi %29, %21 : index
              %31 = loom.reinterpret_cast %arg2 to offset : [%30], sizes : [%14, %15], strides : [%c1024, %c1], reuse : [seq = false, spat = false, temp = false] : memref<*xf32> to memref<?x?xf32, strided<[?, ?], offset: ?>>
              bufferization.materialize_in_destination %28 in writable %31 : (tensor<?x?xf32>, memref<?x?xf32, strided<[?, ?], offset: ?>>) -> ()
            }
          }
        } {loom.mapped_to = @x}
      } {loom.mapped_to = @y}
      return
    }
  }
  module attributes {loom.pass_name = "EnumerateCopyBroadcast"} {
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
      %15 = loom.expression(%13, %12) {coeffs = [1, 1], logic = "mul"} : index
      %16 = loom.expression(%15, %14) {coeffs = [1, 1], logic = "mul"} : index
      loom.polynomial_constraint(%12, %13, %14, %15, %16) {monomials = [{coeff = -1 : i64, vars = [4]}], upper_bound = -262144 : i64}
      %17 = loom.expression(%12, %13) {coeffs = [1, 1], logic = "add"} : index
      %18 = loom.expression(%14, %17) {coeffs = [1, 1], logic = "mul"} : index
      loom.polynomial_constraint(%12, %14, %13, %17, %18) {monomials = [{coeff = 1 : i64, vars = [4]}], upper_bound = 374784 : i64}
      loom.polynomial_constraint(%12) {monomials = [{coeff = 1 : i64, vars = [0]}], upper_bound = 128 : i64}
      loom.polynomial_constraint(%13) {monomials = [{coeff = 1 : i64, vars = [0]}], upper_bound = 128 : i64}
      %19 = loom.expression(%14, %12) {coeffs = [1, 1], logic = "mul"} : index
      loom.polynomial_constraint(%13, %12, %14, %19) {monomials = [{coeff = 1024 : i64, vars = [0]}, {coeff = 1 : i64, vars = [3]}], upper_bound = 374784 : i64}
      %20 = loom.expression(%14, %12) {coeffs = [1, 1], logic = "mul"} : index
      %21 = loom.expression(%13, %14) {coeffs = [1, 1], logic = "mul"} : index
      %22 = loom.expression(%13, %12) {coeffs = [1, 1], logic = "mul"} : index
      %23 = loom.expression(%14, %22) {coeffs = [1, 1], logic = "mul"} : index
      loom.polynomial_constraint(%12, %13, %14, %20, %21, %22, %23) {monomials = [{coeff = 64 : i64, vars = [3]}, {coeff = 64 : i64, vars = [4]}, {coeff = -1 : i64, vars = [6]}], upper_bound = 0 : i64}
    }
    func.func @matmul_kernel__d1i0_d0i1__f10__hoist_block_1__d_d(%arg0: memref<*xf32> {tt.divisibility = 16 : i32}, %arg1: memref<*xf32> {tt.divisibility = 16 : i32}, %arg2: memref<*xf32> {tt.divisibility = 16 : i32}, %arg3: index, %arg4: index, %arg5: index, %arg6: index, %arg7: index, %arg8: index) {
      affine.parallel (%arg9) = (0) to (8) {
        affine.parallel (%arg10) = (0) to (8) {
          affine.for %arg11 = 0 to affine_map<()[s0, s1] -> (s1 ceildiv 8)>()[%arg3, %arg4] {
            affine.for %arg12 = 0 to affine_map<()[s0, s1] -> (s0 ceildiv 8)>()[%arg3, %arg4] {
              %12 = affine.apply affine_map<(d0, d1) -> (d0 * 8 + d1)>(%arg10, %arg11)
              %13 = affine.apply affine_map<(d0, d1) -> (d0 * 8 + d1)>(%arg9, %arg12)
              %c1024 = arith.constant 1024 : index
              %c1024_0 = arith.constant 1024 : index
              %c1 = arith.constant 1 : index
              %14 = loom.get_symbolic_block_size @constraints::@BM : index
              %15 = loom.get_symbolic_block_size @constraints::@BN : index
              %16 = loom.get_symbolic_block_size @constraints::@BK : index
              %17 = arith.ceildivsi %c1024_0, %16 : index
              %cst = arith.constant 0.000000e+00 : f32
              %c1024_1 = arith.constant 1024 : index
              %c0 = arith.constant 0 : index
              %18 = tensor.empty(%14, %15) : tensor<?x?xf32>
              %19 = linalg.fill ins(%cst : f32) outs(%18 : tensor<?x?xf32>) -> tensor<?x?xf32>
              %20 = arith.muli %13, %14 : index
              %21 = arith.muli %12, %15 : index
              %22 = arith.muli %16, %c1024 : index
              %c0_2 = arith.constant 0 : index
              %23 = arith.muli %c0_2, %16 : index
              %24 = arith.muli %23, %c1024_1 : index
              %25 = arith.addi %24, %21 : index
              %26 = loom.reinterpret_cast %arg1 to offset : [%25], sizes : [%17, %16, %15], strides : [%22, %c1024, %c1], reuse : [seq = false, spat = true, temp = true] : memref<*xf32> to memref<?x?x?xf32, strided<[?, ?, ?], offset: ?>>
              %27 = loom.alloc(%17, %16, %15) on @L1 : memref<?x?x?xf32>
              loom.copy %26, %27, interconnect : [], broadcast : [1, 1] : memref<?x?x?xf32, strided<[?, ?, ?], offset: ?>> to memref<?x?x?xf32>
              %28 = scf.for %arg13 = %c0 to %17 step %c1 iter_args(%arg14 = %19) -> (tensor<?x?xf32>) {
                %32 = arith.muli %arg13, %16 : index
                %33 = arith.muli %20, %c1024_1 : index
                %34 = arith.addi %33, %32 : index
                %35 = loom.reinterpret_cast %arg0 to offset : [%34], sizes : [%14, %16], strides : [%c1024_0, %c1], reuse : [seq = false, spat = true, temp = true] : memref<*xf32> to memref<?x?xf32, strided<[?, ?], offset: ?>>
                %36 = loom.alloc(%14, %16) on @L1 : memref<?x?xf32>
                loom.copy %35, %36, interconnect : [], broadcast : [1, 1] : memref<?x?xf32, strided<[?, ?], offset: ?>> to memref<?x?xf32>
                %37 = bufferization.to_tensor %36 restrict writable : memref<?x?xf32> to tensor<?x?xf32>
                %38 = arith.muli %32, %c1024_1 : index
                %subview = memref.subview %27[%arg13, 0, 0] [1, %16, %15] [1, 1, 1] : memref<?x?x?xf32> to memref<?x?xf32, strided<[?, 1], offset: ?>>
                %39 = bufferization.to_tensor %subview restrict writable : memref<?x?xf32, strided<[?, 1], offset: ?>> to tensor<?x?xf32>
                %40 = linalg.matmul ins(%37, %39 : tensor<?x?xf32>, tensor<?x?xf32>) outs(%19 : tensor<?x?xf32>) -> tensor<?x?xf32>
                %41 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%arg14, %40 : tensor<?x?xf32>, tensor<?x?xf32>) outs(%arg14 : tensor<?x?xf32>) {
                ^bb0(%in: f32, %in_3: f32, %out: f32):
                  %42 = arith.addf %in, %in_3 : f32
                  linalg.yield %42 : f32
                } -> tensor<?x?xf32>
                scf.yield %41 : tensor<?x?xf32>
              }
              %29 = arith.muli %20, %c1024_1 : index
              %30 = arith.addi %29, %21 : index
              %31 = loom.reinterpret_cast %arg2 to offset : [%30], sizes : [%14, %15], strides : [%c1024, %c1], reuse : [seq = false, spat = false, temp = false] : memref<*xf32> to memref<?x?xf32, strided<[?, ?], offset: ?>>
              bufferization.materialize_in_destination %28 in writable %31 : (tensor<?x?xf32>, memref<?x?xf32, strided<[?, ?], offset: ?>>) -> ()
            }
          }
        } {loom.mapped_to = @x}
      } {loom.mapped_to = @y}
      return
    }
  }
  module attributes {loom.pass_name = "EnumerateCopyBroadcast"} {
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
      %15 = loom.expression(%13, %12) {coeffs = [1, 1], logic = "mul"} : index
      %16 = loom.expression(%15, %14) {coeffs = [1, 1], logic = "mul"} : index
      loom.polynomial_constraint(%12, %13, %14, %15, %16) {monomials = [{coeff = -1 : i64, vars = [4]}], upper_bound = -262144 : i64}
      %17 = loom.expression(%12, %13) {coeffs = [1, 1], logic = "add"} : index
      %18 = loom.expression(%14, %17) {coeffs = [1, 1], logic = "mul"} : index
      loom.polynomial_constraint(%12, %14, %13, %17, %18) {monomials = [{coeff = 1 : i64, vars = [4]}], upper_bound = 374784 : i64}
      loom.polynomial_constraint(%12) {monomials = [{coeff = 1 : i64, vars = [0]}], upper_bound = 128 : i64}
      loom.polynomial_constraint(%13) {monomials = [{coeff = 1 : i64, vars = [0]}], upper_bound = 128 : i64}
      %19 = loom.expression(%14, %12) {coeffs = [1, 1], logic = "mul"} : index
      loom.polynomial_constraint(%13, %12, %14, %19) {monomials = [{coeff = 1024 : i64, vars = [0]}, {coeff = 1 : i64, vars = [3]}], upper_bound = 374784 : i64}
      %20 = loom.expression(%14, %12) {coeffs = [1, 1], logic = "mul"} : index
      %21 = loom.expression(%13, %14) {coeffs = [1, 1], logic = "mul"} : index
      %22 = loom.expression(%13, %12) {coeffs = [1, 1], logic = "mul"} : index
      %23 = loom.expression(%14, %22) {coeffs = [1, 1], logic = "mul"} : index
      loom.polynomial_constraint(%12, %13, %14, %20, %21, %22, %23) {monomials = [{coeff = 960 : i64, vars = [3]}, {coeff = 256 : i64, vars = [4]}, {coeff = -15 : i64, vars = [6]}], upper_bound = 0 : i64}
    }
    func.func @matmul_kernel__d1i0_d0i1__f10__hoist_block_1__d_v(%arg0: memref<*xf32> {tt.divisibility = 16 : i32}, %arg1: memref<*xf32> {tt.divisibility = 16 : i32}, %arg2: memref<*xf32> {tt.divisibility = 16 : i32}, %arg3: index, %arg4: index, %arg5: index, %arg6: index, %arg7: index, %arg8: index) {
      affine.parallel (%arg9) = (0) to (8) {
        affine.parallel (%arg10) = (0) to (8) {
          affine.for %arg11 = 0 to affine_map<()[s0, s1] -> (s1 ceildiv 8)>()[%arg3, %arg4] {
            affine.for %arg12 = 0 to affine_map<()[s0, s1] -> (s0 ceildiv 8)>()[%arg3, %arg4] {
              %12 = affine.apply affine_map<(d0, d1) -> (d0 * 8 + d1)>(%arg10, %arg11)
              %13 = affine.apply affine_map<(d0, d1) -> (d0 * 8 + d1)>(%arg9, %arg12)
              %c1024 = arith.constant 1024 : index
              %c1024_0 = arith.constant 1024 : index
              %c1 = arith.constant 1 : index
              %14 = loom.get_symbolic_block_size @constraints::@BM : index
              %15 = loom.get_symbolic_block_size @constraints::@BN : index
              %16 = loom.get_symbolic_block_size @constraints::@BK : index
              %17 = arith.ceildivsi %c1024_0, %16 : index
              %cst = arith.constant 0.000000e+00 : f32
              %c1024_1 = arith.constant 1024 : index
              %c0 = arith.constant 0 : index
              %18 = tensor.empty(%14, %15) : tensor<?x?xf32>
              %19 = linalg.fill ins(%cst : f32) outs(%18 : tensor<?x?xf32>) -> tensor<?x?xf32>
              %20 = arith.muli %13, %14 : index
              %21 = arith.muli %12, %15 : index
              %22 = arith.muli %16, %c1024 : index
              %c0_2 = arith.constant 0 : index
              %23 = arith.muli %c0_2, %16 : index
              %24 = arith.muli %23, %c1024_1 : index
              %25 = arith.addi %24, %21 : index
              %26 = loom.reinterpret_cast %arg1 to offset : [%25], sizes : [%17, %16, %15], strides : [%22, %c1024, %c1], reuse : [seq = false, spat = true, temp = true] : memref<*xf32> to memref<?x?x?xf32, strided<[?, ?, ?], offset: ?>>
              %27 = loom.alloc(%17, %16, %15) on @L1 : memref<?x?x?xf32>
              loom.copy %26, %27, interconnect : [], broadcast : [1, 1] : memref<?x?x?xf32, strided<[?, ?, ?], offset: ?>> to memref<?x?x?xf32>
              %28 = scf.for %arg13 = %c0 to %17 step %c1 iter_args(%arg14 = %19) -> (tensor<?x?xf32>) {
                %32 = arith.muli %arg13, %16 : index
                %33 = arith.muli %20, %c1024_1 : index
                %34 = arith.addi %33, %32 : index
                %35 = loom.reinterpret_cast %arg0 to offset : [%34], sizes : [%14, %16], strides : [%c1024_0, %c1], reuse : [seq = false, spat = true, temp = true] : memref<*xf32> to memref<?x?xf32, strided<[?, ?], offset: ?>>
                %36 = loom.alloc(%14, %16) on @L1 : memref<?x?xf32>
                loom.copy %35, %36, interconnect : [@vertical_links], broadcast : [8, 1] : memref<?x?xf32, strided<[?, ?], offset: ?>> to memref<?x?xf32>
                %37 = bufferization.to_tensor %36 restrict writable : memref<?x?xf32> to tensor<?x?xf32>
                %38 = arith.muli %32, %c1024_1 : index
                %subview = memref.subview %27[%arg13, 0, 0] [1, %16, %15] [1, 1, 1] : memref<?x?x?xf32> to memref<?x?xf32, strided<[?, 1], offset: ?>>
                %39 = bufferization.to_tensor %subview restrict writable : memref<?x?xf32, strided<[?, 1], offset: ?>> to tensor<?x?xf32>
                %40 = linalg.matmul ins(%37, %39 : tensor<?x?xf32>, tensor<?x?xf32>) outs(%19 : tensor<?x?xf32>) -> tensor<?x?xf32>
                %41 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%arg14, %40 : tensor<?x?xf32>, tensor<?x?xf32>) outs(%arg14 : tensor<?x?xf32>) {
                ^bb0(%in: f32, %in_3: f32, %out: f32):
                  %42 = arith.addf %in, %in_3 : f32
                  linalg.yield %42 : f32
                } -> tensor<?x?xf32>
                scf.yield %41 : tensor<?x?xf32>
              }
              %29 = arith.muli %20, %c1024_1 : index
              %30 = arith.addi %29, %21 : index
              %31 = loom.reinterpret_cast %arg2 to offset : [%30], sizes : [%14, %15], strides : [%c1024, %c1], reuse : [seq = false, spat = false, temp = false] : memref<*xf32> to memref<?x?xf32, strided<[?, ?], offset: ?>>
              bufferization.materialize_in_destination %28 in writable %31 : (tensor<?x?xf32>, memref<?x?xf32, strided<[?, ?], offset: ?>>) -> ()
            }
          }
        } {loom.mapped_to = @x}
      } {loom.mapped_to = @y}
      return
    }
  }
  module attributes {loom.pass_name = "EnumerateCopyBroadcast"} {
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
      %15 = loom.expression(%12, %13) {coeffs = [1, 1], logic = "mul"} : index
      %16 = loom.expression(%15, %14) {coeffs = [1, 1], logic = "mul"} : index
      loom.polynomial_constraint(%12, %13, %14, %15, %16) {monomials = [{coeff = -1 : i64, vars = [4]}], upper_bound = -262144 : i64}
      %17 = loom.expression(%12, %13) {coeffs = [1, 1], logic = "add"} : index
      %18 = loom.expression(%14, %17) {coeffs = [1, 1], logic = "mul"} : index
      loom.polynomial_constraint(%12, %14, %13, %17, %18) {monomials = [{coeff = 1 : i64, vars = [4]}], upper_bound = 374784 : i64}
      loom.polynomial_constraint(%12) {monomials = [{coeff = 1 : i64, vars = [0]}], upper_bound = 128 : i64}
      loom.polynomial_constraint(%13) {monomials = [{coeff = 1 : i64, vars = [0]}], upper_bound = 128 : i64}
      %19 = loom.expression(%12, %14) {coeffs = [1, 1], logic = "mul"} : index
      loom.polynomial_constraint(%13, %12, %14, %19) {monomials = [{coeff = 1024 : i64, vars = [0]}, {coeff = 1 : i64, vars = [3]}], upper_bound = 374784 : i64}
      %20 = loom.expression(%12, %14) {coeffs = [1, 1], logic = "mul"} : index
      %21 = loom.expression(%13, %14) {coeffs = [1, 1], logic = "mul"} : index
      %22 = loom.expression(%12, %13) {coeffs = [1, 1], logic = "mul"} : index
      %23 = loom.expression(%14, %22) {coeffs = [1, 1], logic = "mul"} : index
      loom.polynomial_constraint(%12, %13, %14, %20, %21, %22, %23) {monomials = [{coeff = 256 : i64, vars = [3]}, {coeff = 960 : i64, vars = [4]}, {coeff = -15 : i64, vars = [6]}], upper_bound = 0 : i64}
    }
    func.func @matmul_kernel__d1i0_d0i1__f10__hoist_block_1__h_d(%arg0: memref<*xf32> {tt.divisibility = 16 : i32}, %arg1: memref<*xf32> {tt.divisibility = 16 : i32}, %arg2: memref<*xf32> {tt.divisibility = 16 : i32}, %arg3: index, %arg4: index, %arg5: index, %arg6: index, %arg7: index, %arg8: index) {
      affine.parallel (%arg9) = (0) to (8) {
        affine.parallel (%arg10) = (0) to (8) {
          affine.for %arg11 = 0 to affine_map<()[s0, s1] -> (s1 ceildiv 8)>()[%arg3, %arg4] {
            affine.for %arg12 = 0 to affine_map<()[s0, s1] -> (s0 ceildiv 8)>()[%arg3, %arg4] {
              %12 = affine.apply affine_map<(d0, d1) -> (d0 * 8 + d1)>(%arg10, %arg11)
              %13 = affine.apply affine_map<(d0, d1) -> (d0 * 8 + d1)>(%arg9, %arg12)
              %c1024 = arith.constant 1024 : index
              %c1024_0 = arith.constant 1024 : index
              %c1 = arith.constant 1 : index
              %14 = loom.get_symbolic_block_size @constraints::@BM : index
              %15 = loom.get_symbolic_block_size @constraints::@BN : index
              %16 = loom.get_symbolic_block_size @constraints::@BK : index
              %17 = arith.ceildivsi %c1024_0, %16 : index
              %cst = arith.constant 0.000000e+00 : f32
              %c1024_1 = arith.constant 1024 : index
              %c0 = arith.constant 0 : index
              %18 = tensor.empty(%14, %15) : tensor<?x?xf32>
              %19 = linalg.fill ins(%cst : f32) outs(%18 : tensor<?x?xf32>) -> tensor<?x?xf32>
              %20 = arith.muli %13, %14 : index
              %21 = arith.muli %12, %15 : index
              %22 = arith.muli %16, %c1024 : index
              %c0_2 = arith.constant 0 : index
              %23 = arith.muli %c0_2, %16 : index
              %24 = arith.muli %23, %c1024_1 : index
              %25 = arith.addi %24, %21 : index
              %26 = loom.reinterpret_cast %arg1 to offset : [%25], sizes : [%17, %16, %15], strides : [%22, %c1024, %c1], reuse : [seq = false, spat = true, temp = true] : memref<*xf32> to memref<?x?x?xf32, strided<[?, ?, ?], offset: ?>>
              %27 = loom.alloc(%17, %16, %15) on @L1 : memref<?x?x?xf32>
              loom.copy %26, %27, interconnect : [@horizontal_links], broadcast : [1, 8] : memref<?x?x?xf32, strided<[?, ?, ?], offset: ?>> to memref<?x?x?xf32>
              %28 = scf.for %arg13 = %c0 to %17 step %c1 iter_args(%arg14 = %19) -> (tensor<?x?xf32>) {
                %32 = arith.muli %arg13, %16 : index
                %33 = arith.muli %20, %c1024_1 : index
                %34 = arith.addi %33, %32 : index
                %35 = loom.reinterpret_cast %arg0 to offset : [%34], sizes : [%14, %16], strides : [%c1024_0, %c1], reuse : [seq = false, spat = true, temp = true] : memref<*xf32> to memref<?x?xf32, strided<[?, ?], offset: ?>>
                %36 = loom.alloc(%14, %16) on @L1 : memref<?x?xf32>
                loom.copy %35, %36, interconnect : [], broadcast : [1, 1] : memref<?x?xf32, strided<[?, ?], offset: ?>> to memref<?x?xf32>
                %37 = bufferization.to_tensor %36 restrict writable : memref<?x?xf32> to tensor<?x?xf32>
                %38 = arith.muli %32, %c1024_1 : index
                %subview = memref.subview %27[%arg13, 0, 0] [1, %16, %15] [1, 1, 1] : memref<?x?x?xf32> to memref<?x?xf32, strided<[?, 1], offset: ?>>
                %39 = bufferization.to_tensor %subview restrict writable : memref<?x?xf32, strided<[?, 1], offset: ?>> to tensor<?x?xf32>
                %40 = linalg.matmul ins(%37, %39 : tensor<?x?xf32>, tensor<?x?xf32>) outs(%19 : tensor<?x?xf32>) -> tensor<?x?xf32>
                %41 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%arg14, %40 : tensor<?x?xf32>, tensor<?x?xf32>) outs(%arg14 : tensor<?x?xf32>) {
                ^bb0(%in: f32, %in_3: f32, %out: f32):
                  %42 = arith.addf %in, %in_3 : f32
                  linalg.yield %42 : f32
                } -> tensor<?x?xf32>
                scf.yield %41 : tensor<?x?xf32>
              }
              %29 = arith.muli %20, %c1024_1 : index
              %30 = arith.addi %29, %21 : index
              %31 = loom.reinterpret_cast %arg2 to offset : [%30], sizes : [%14, %15], strides : [%c1024, %c1], reuse : [seq = false, spat = false, temp = false] : memref<*xf32> to memref<?x?xf32, strided<[?, ?], offset: ?>>
              bufferization.materialize_in_destination %28 in writable %31 : (tensor<?x?xf32>, memref<?x?xf32, strided<[?, ?], offset: ?>>) -> ()
            }
          }
        } {loom.mapped_to = @x}
      } {loom.mapped_to = @y}
      return
    }
  }
  module attributes {loom.pass_name = "EnumerateCopyBroadcast"} {
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
      %15 = loom.expression(%13, %12) {coeffs = [1, 1], logic = "mul"} : index
      %16 = loom.expression(%15, %14) {coeffs = [1, 1], logic = "mul"} : index
      loom.polynomial_constraint(%12, %13, %14, %15, %16) {monomials = [{coeff = -1 : i64, vars = [4]}], upper_bound = -262144 : i64}
      %17 = loom.expression(%12, %13) {coeffs = [1, 1], logic = "add"} : index
      %18 = loom.expression(%14, %17) {coeffs = [1, 1], logic = "mul"} : index
      loom.polynomial_constraint(%12, %14, %13, %17, %18) {monomials = [{coeff = 1 : i64, vars = [4]}], upper_bound = 374784 : i64}
      loom.polynomial_constraint(%12) {monomials = [{coeff = 1 : i64, vars = [0]}], upper_bound = 128 : i64}
      loom.polynomial_constraint(%13) {monomials = [{coeff = 1 : i64, vars = [0]}], upper_bound = 128 : i64}
      %19 = loom.expression(%14, %12) {coeffs = [1, 1], logic = "mul"} : index
      loom.polynomial_constraint(%13, %12, %14, %19) {monomials = [{coeff = 1024 : i64, vars = [0]}, {coeff = 1 : i64, vars = [3]}], upper_bound = 374784 : i64}
      %20 = loom.expression(%14, %12) {coeffs = [1, 1], logic = "mul"} : index
      %21 = loom.expression(%13, %14) {coeffs = [1, 1], logic = "mul"} : index
      %22 = loom.expression(%13, %12) {coeffs = [1, 1], logic = "mul"} : index
      %23 = loom.expression(%14, %22) {coeffs = [1, 1], logic = "mul"} : index
      loom.polynomial_constraint(%12, %13, %14, %20, %21, %22, %23) {monomials = [{coeff = 256 : i64, vars = [3]}, {coeff = 256 : i64, vars = [4]}, {coeff = -15 : i64, vars = [6]}], upper_bound = 0 : i64}
    }
    func.func @matmul_kernel__d1i0_d0i1__f10__hoist_block_1__h_v(%arg0: memref<*xf32> {tt.divisibility = 16 : i32}, %arg1: memref<*xf32> {tt.divisibility = 16 : i32}, %arg2: memref<*xf32> {tt.divisibility = 16 : i32}, %arg3: index, %arg4: index, %arg5: index, %arg6: index, %arg7: index, %arg8: index) {
      affine.parallel (%arg9) = (0) to (8) {
        affine.parallel (%arg10) = (0) to (8) {
          affine.for %arg11 = 0 to affine_map<()[s0, s1] -> (s1 ceildiv 8)>()[%arg3, %arg4] {
            affine.for %arg12 = 0 to affine_map<()[s0, s1] -> (s0 ceildiv 8)>()[%arg3, %arg4] {
              %12 = affine.apply affine_map<(d0, d1) -> (d0 * 8 + d1)>(%arg10, %arg11)
              %13 = affine.apply affine_map<(d0, d1) -> (d0 * 8 + d1)>(%arg9, %arg12)
              %c1024 = arith.constant 1024 : index
              %c1024_0 = arith.constant 1024 : index
              %c1 = arith.constant 1 : index
              %14 = loom.get_symbolic_block_size @constraints::@BM : index
              %15 = loom.get_symbolic_block_size @constraints::@BN : index
              %16 = loom.get_symbolic_block_size @constraints::@BK : index
              %17 = arith.ceildivsi %c1024_0, %16 : index
              %cst = arith.constant 0.000000e+00 : f32
              %c1024_1 = arith.constant 1024 : index
              %c0 = arith.constant 0 : index
              %18 = tensor.empty(%14, %15) : tensor<?x?xf32>
              %19 = linalg.fill ins(%cst : f32) outs(%18 : tensor<?x?xf32>) -> tensor<?x?xf32>
              %20 = arith.muli %13, %14 : index
              %21 = arith.muli %12, %15 : index
              %22 = arith.muli %16, %c1024 : index
              %c0_2 = arith.constant 0 : index
              %23 = arith.muli %c0_2, %16 : index
              %24 = arith.muli %23, %c1024_1 : index
              %25 = arith.addi %24, %21 : index
              %26 = loom.reinterpret_cast %arg1 to offset : [%25], sizes : [%17, %16, %15], strides : [%22, %c1024, %c1], reuse : [seq = false, spat = true, temp = true] : memref<*xf32> to memref<?x?x?xf32, strided<[?, ?, ?], offset: ?>>
              %27 = loom.alloc(%17, %16, %15) on @L1 : memref<?x?x?xf32>
              loom.copy %26, %27, interconnect : [@horizontal_links], broadcast : [1, 8] : memref<?x?x?xf32, strided<[?, ?, ?], offset: ?>> to memref<?x?x?xf32>
              %28 = scf.for %arg13 = %c0 to %17 step %c1 iter_args(%arg14 = %19) -> (tensor<?x?xf32>) {
                %32 = arith.muli %arg13, %16 : index
                %33 = arith.muli %20, %c1024_1 : index
                %34 = arith.addi %33, %32 : index
                %35 = loom.reinterpret_cast %arg0 to offset : [%34], sizes : [%14, %16], strides : [%c1024_0, %c1], reuse : [seq = false, spat = true, temp = true] : memref<*xf32> to memref<?x?xf32, strided<[?, ?], offset: ?>>
                %36 = loom.alloc(%14, %16) on @L1 : memref<?x?xf32>
                loom.copy %35, %36, interconnect : [@vertical_links], broadcast : [8, 1] : memref<?x?xf32, strided<[?, ?], offset: ?>> to memref<?x?xf32>
                %37 = bufferization.to_tensor %36 restrict writable : memref<?x?xf32> to tensor<?x?xf32>
                %38 = arith.muli %32, %c1024_1 : index
                %subview = memref.subview %27[%arg13, 0, 0] [1, %16, %15] [1, 1, 1] : memref<?x?x?xf32> to memref<?x?xf32, strided<[?, 1], offset: ?>>
                %39 = bufferization.to_tensor %subview restrict writable : memref<?x?xf32, strided<[?, 1], offset: ?>> to tensor<?x?xf32>
                %40 = linalg.matmul ins(%37, %39 : tensor<?x?xf32>, tensor<?x?xf32>) outs(%19 : tensor<?x?xf32>) -> tensor<?x?xf32>
                %41 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%arg14, %40 : tensor<?x?xf32>, tensor<?x?xf32>) outs(%arg14 : tensor<?x?xf32>) {
                ^bb0(%in: f32, %in_3: f32, %out: f32):
                  %42 = arith.addf %in, %in_3 : f32
                  linalg.yield %42 : f32
                } -> tensor<?x?xf32>
                scf.yield %41 : tensor<?x?xf32>
              }
              %29 = arith.muli %20, %c1024_1 : index
              %30 = arith.addi %29, %21 : index
              %31 = loom.reinterpret_cast %arg2 to offset : [%30], sizes : [%14, %15], strides : [%c1024, %c1], reuse : [seq = false, spat = false, temp = false] : memref<*xf32> to memref<?x?xf32, strided<[?, ?], offset: ?>>
              bufferization.materialize_in_destination %28 in writable %31 : (tensor<?x?xf32>, memref<?x?xf32, strided<[?, ?], offset: ?>>) -> ()
            }
          }
        } {loom.mapped_to = @x}
      } {loom.mapped_to = @y}
      return
    }
  }
  module attributes {loom.pass_name = "EnumerateCopyBroadcast"} {
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
      %15 = loom.expression(%13, %12) {coeffs = [1, 1], logic = "mul"} : index
      %16 = loom.expression(%14, %15) {coeffs = [1, 1], logic = "mul"} : index
      loom.polynomial_constraint(%12, %13, %14, %15, %16) {monomials = [{coeff = -1 : i64, vars = [4]}], upper_bound = -262144 : i64}
      %17 = loom.expression(%12, %13) {coeffs = [1, 1], logic = "add"} : index
      %18 = loom.expression(%14, %17) {coeffs = [1, 1], logic = "mul"} : index
      loom.polynomial_constraint(%12, %14, %13, %17, %18) {monomials = [{coeff = 1 : i64, vars = [4]}], upper_bound = 374784 : i64}
      loom.polynomial_constraint(%13) {monomials = [{coeff = 1 : i64, vars = [0]}], upper_bound = 16 : i64}
      %19 = loom.expression(%14, %12) {coeffs = [1, 1], logic = "mul"} : index
      %20 = loom.expression(%13, %14) {coeffs = [1, 1], logic = "mul"} : index
      %21 = loom.expression(%13, %12) {coeffs = [1, 1], logic = "mul"} : index
      %22 = loom.expression(%14, %21) {coeffs = [1, 1], logic = "mul"} : index
      loom.polynomial_constraint(%12, %13, %14, %19, %20, %21, %22) {monomials = [{coeff = 64 : i64, vars = [3]}, {coeff = 64 : i64, vars = [4]}, {coeff = -1 : i64, vars = [6]}], upper_bound = 0 : i64}
    }
    func.func @matmul_kernel__d0i1_d1i1__f01__d_d(%arg0: memref<*xf32> {tt.divisibility = 16 : i32}, %arg1: memref<*xf32> {tt.divisibility = 16 : i32}, %arg2: memref<*xf32> {tt.divisibility = 16 : i32}, %arg3: index, %arg4: index, %arg5: index, %arg6: index, %arg7: index, %arg8: index) {
      affine.parallel (%arg9) = (0) to (8) {
        affine.parallel (%arg10) = (0) to (8) {
          affine.for %arg11 = 0 to affine_map<()[s0, s1] -> (s0)>()[%arg3, %arg4] {
            affine.for %arg12 = 0 to affine_map<()[s0, s1] -> ((s1 ceildiv 8) ceildiv 8)>()[%arg3, %arg4] {
              %12 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 8 + d1 * 8 + d2)>(%arg9, %arg10, %arg12)
              %c1024 = arith.constant 1024 : index
              %c1024_0 = arith.constant 1024 : index
              %c1 = arith.constant 1 : index
              %13 = loom.get_symbolic_block_size @constraints::@BM : index
              %14 = loom.get_symbolic_block_size @constraints::@BN : index
              %15 = loom.get_symbolic_block_size @constraints::@BK : index
              %16 = arith.ceildivsi %c1024_0, %15 : index
              %cst = arith.constant 0.000000e+00 : f32
              %c1024_1 = arith.constant 1024 : index
              %c0 = arith.constant 0 : index
              %17 = tensor.empty(%13, %14) : tensor<?x?xf32>
              %18 = linalg.fill ins(%cst : f32) outs(%17 : tensor<?x?xf32>) -> tensor<?x?xf32>
              %19 = arith.muli %arg11, %13 : index
              %20 = arith.muli %12, %14 : index
              %21 = scf.for %arg13 = %c0 to %16 step %c1 iter_args(%arg14 = %18) -> (tensor<?x?xf32>) {
                %25 = arith.muli %arg13, %15 : index
                %26 = arith.muli %19, %c1024_1 : index
                %27 = arith.addi %26, %25 : index
                %28 = loom.reinterpret_cast %arg0 to offset : [%27], sizes : [%13, %15], strides : [%c1024_0, %c1], reuse : [seq = false, spat = true, temp = true] : memref<*xf32> to memref<?x?xf32, strided<[?, ?], offset: ?>>
                %29 = loom.alloc(%13, %15) on @L1 : memref<?x?xf32>
                loom.copy %28, %29, interconnect : [], broadcast : [1, 1] : memref<?x?xf32, strided<[?, ?], offset: ?>> to memref<?x?xf32>
                %30 = bufferization.to_tensor %29 restrict writable : memref<?x?xf32> to tensor<?x?xf32>
                %31 = arith.muli %25, %c1024_1 : index
                %32 = arith.addi %31, %20 : index
                %33 = loom.reinterpret_cast %arg1 to offset : [%32], sizes : [%15, %14], strides : [%c1024, %c1], reuse : [seq = false, spat = false, temp = true] : memref<*xf32> to memref<?x?xf32, strided<[?, ?], offset: ?>>
                %34 = loom.alloc(%15, %14) on @L1 : memref<?x?xf32>
                loom.copy %33, %34, interconnect : [], broadcast : [1, 1] : memref<?x?xf32, strided<[?, ?], offset: ?>> to memref<?x?xf32>
                %35 = bufferization.to_tensor %34 restrict writable : memref<?x?xf32> to tensor<?x?xf32>
                %36 = linalg.matmul ins(%30, %35 : tensor<?x?xf32>, tensor<?x?xf32>) outs(%18 : tensor<?x?xf32>) -> tensor<?x?xf32>
                %37 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%arg14, %36 : tensor<?x?xf32>, tensor<?x?xf32>) outs(%arg14 : tensor<?x?xf32>) {
                ^bb0(%in: f32, %in_2: f32, %out: f32):
                  %38 = arith.addf %in, %in_2 : f32
                  linalg.yield %38 : f32
                } -> tensor<?x?xf32>
                scf.yield %37 : tensor<?x?xf32>
              }
              %22 = arith.muli %19, %c1024_1 : index
              %23 = arith.addi %22, %20 : index
              %24 = loom.reinterpret_cast %arg2 to offset : [%23], sizes : [%13, %14], strides : [%c1024, %c1], reuse : [seq = false, spat = false, temp = false] : memref<*xf32> to memref<?x?xf32, strided<[?, ?], offset: ?>>
              bufferization.materialize_in_destination %21 in writable %24 : (tensor<?x?xf32>, memref<?x?xf32, strided<[?, ?], offset: ?>>) -> ()
            }
          }
        } {loom.mapped_to = @y}
      } {loom.mapped_to = @x}
      return
    }
  }
  module attributes {loom.pass_name = "EnumerateCopyBroadcast"} {
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
      %15 = loom.expression(%12, %13) {coeffs = [1, 1], logic = "mul"} : index
      %16 = loom.expression(%15, %14) {coeffs = [1, 1], logic = "mul"} : index
      loom.polynomial_constraint(%12, %13, %14, %15, %16) {monomials = [{coeff = -1 : i64, vars = [4]}], upper_bound = -262144 : i64}
      %17 = loom.expression(%12, %13) {coeffs = [1, 1], logic = "add"} : index
      %18 = loom.expression(%17, %14) {coeffs = [1, 1], logic = "mul"} : index
      loom.polynomial_constraint(%12, %14, %13, %17, %18) {monomials = [{coeff = 1 : i64, vars = [4]}], upper_bound = 374784 : i64}
      loom.polynomial_constraint(%13) {monomials = [{coeff = 1 : i64, vars = [0]}], upper_bound = 16 : i64}
      %19 = loom.expression(%12, %14) {coeffs = [1, 1], logic = "mul"} : index
      %20 = loom.expression(%13, %14) {coeffs = [1, 1], logic = "mul"} : index
      %21 = loom.expression(%12, %13) {coeffs = [1, 1], logic = "mul"} : index
      %22 = loom.expression(%14, %21) {coeffs = [1, 1], logic = "mul"} : index
      loom.polynomial_constraint(%12, %13, %14, %19, %20, %21, %22) {monomials = [{coeff = 256 : i64, vars = [3]}, {coeff = 960 : i64, vars = [4]}, {coeff = -15 : i64, vars = [6]}], upper_bound = 0 : i64}
    }
    func.func @matmul_kernel__d0i1_d1i1__f01__a_d(%arg0: memref<*xf32> {tt.divisibility = 16 : i32}, %arg1: memref<*xf32> {tt.divisibility = 16 : i32}, %arg2: memref<*xf32> {tt.divisibility = 16 : i32}, %arg3: index, %arg4: index, %arg5: index, %arg6: index, %arg7: index, %arg8: index) {
      affine.parallel (%arg9) = (0) to (8) {
        affine.parallel (%arg10) = (0) to (8) {
          affine.for %arg11 = 0 to affine_map<()[s0, s1] -> (s0)>()[%arg3, %arg4] {
            affine.for %arg12 = 0 to affine_map<()[s0, s1] -> ((s1 ceildiv 8) ceildiv 8)>()[%arg3, %arg4] {
              %12 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 8 + d1 * 8 + d2)>(%arg9, %arg10, %arg12)
              %c1024 = arith.constant 1024 : index
              %c1024_0 = arith.constant 1024 : index
              %c1 = arith.constant 1 : index
              %13 = loom.get_symbolic_block_size @constraints::@BM : index
              %14 = loom.get_symbolic_block_size @constraints::@BN : index
              %15 = loom.get_symbolic_block_size @constraints::@BK : index
              %16 = arith.ceildivsi %c1024_0, %15 : index
              %cst = arith.constant 0.000000e+00 : f32
              %c1024_1 = arith.constant 1024 : index
              %c0 = arith.constant 0 : index
              %17 = tensor.empty(%13, %14) : tensor<?x?xf32>
              %18 = linalg.fill ins(%cst : f32) outs(%17 : tensor<?x?xf32>) -> tensor<?x?xf32>
              %19 = arith.muli %arg11, %13 : index
              %20 = arith.muli %12, %14 : index
              %21 = scf.for %arg13 = %c0 to %16 step %c1 iter_args(%arg14 = %18) -> (tensor<?x?xf32>) {
                %25 = arith.muli %arg13, %15 : index
                %26 = arith.muli %19, %c1024_1 : index
                %27 = arith.addi %26, %25 : index
                %28 = loom.reinterpret_cast %arg0 to offset : [%27], sizes : [%13, %15], strides : [%c1024_0, %c1], reuse : [seq = false, spat = true, temp = true] : memref<*xf32> to memref<?x?xf32, strided<[?, ?], offset: ?>>
                %29 = loom.alloc(%13, %15) on @L1 : memref<?x?xf32>
                loom.copy %28, %29, interconnect : [@horizontal_links, @vertical_links], broadcast : [8, 8] : memref<?x?xf32, strided<[?, ?], offset: ?>> to memref<?x?xf32>
                %30 = bufferization.to_tensor %29 restrict writable : memref<?x?xf32> to tensor<?x?xf32>
                %31 = arith.muli %25, %c1024_1 : index
                %32 = arith.addi %31, %20 : index
                %33 = loom.reinterpret_cast %arg1 to offset : [%32], sizes : [%15, %14], strides : [%c1024, %c1], reuse : [seq = false, spat = false, temp = true] : memref<*xf32> to memref<?x?xf32, strided<[?, ?], offset: ?>>
                %34 = loom.alloc(%15, %14) on @L1 : memref<?x?xf32>
                loom.copy %33, %34, interconnect : [], broadcast : [1, 1] : memref<?x?xf32, strided<[?, ?], offset: ?>> to memref<?x?xf32>
                %35 = bufferization.to_tensor %34 restrict writable : memref<?x?xf32> to tensor<?x?xf32>
                %36 = linalg.matmul ins(%30, %35 : tensor<?x?xf32>, tensor<?x?xf32>) outs(%18 : tensor<?x?xf32>) -> tensor<?x?xf32>
                %37 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%arg14, %36 : tensor<?x?xf32>, tensor<?x?xf32>) outs(%arg14 : tensor<?x?xf32>) {
                ^bb0(%in: f32, %in_2: f32, %out: f32):
                  %38 = arith.addf %in, %in_2 : f32
                  linalg.yield %38 : f32
                } -> tensor<?x?xf32>
                scf.yield %37 : tensor<?x?xf32>
              }
              %22 = arith.muli %19, %c1024_1 : index
              %23 = arith.addi %22, %20 : index
              %24 = loom.reinterpret_cast %arg2 to offset : [%23], sizes : [%13, %14], strides : [%c1024, %c1], reuse : [seq = false, spat = false, temp = false] : memref<*xf32> to memref<?x?xf32, strided<[?, ?], offset: ?>>
              bufferization.materialize_in_destination %21 in writable %24 : (tensor<?x?xf32>, memref<?x?xf32, strided<[?, ?], offset: ?>>) -> ()
            }
          }
        } {loom.mapped_to = @y}
      } {loom.mapped_to = @x}
      return
    }
  }
  module attributes {loom.pass_name = "EnumerateCopyBroadcast"} {
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
      %15 = loom.expression(%12, %13) {coeffs = [1, 1], logic = "mul"} : index
      %16 = loom.expression(%15, %14) {coeffs = [1, 1], logic = "mul"} : index
      loom.polynomial_constraint(%12, %13, %14, %15, %16) {monomials = [{coeff = -1 : i64, vars = [4]}], upper_bound = -262144 : i64}
      %17 = loom.expression(%12, %13) {coeffs = [1, 1], logic = "add"} : index
      %18 = loom.expression(%17, %14) {coeffs = [1, 1], logic = "mul"} : index
      loom.polynomial_constraint(%12, %14, %13, %17, %18) {monomials = [{coeff = 1 : i64, vars = [4]}], upper_bound = 374784 : i64}
      loom.polynomial_constraint(%13) {monomials = [{coeff = 1 : i64, vars = [0]}], upper_bound = 16 : i64}
      %19 = loom.expression(%12, %14) {coeffs = [1, 1], logic = "mul"} : index
      %20 = loom.expression(%13, %14) {coeffs = [1, 1], logic = "mul"} : index
      %21 = loom.expression(%12, %13) {coeffs = [1, 1], logic = "mul"} : index
      %22 = loom.expression(%14, %21) {coeffs = [1, 1], logic = "mul"} : index
      loom.polynomial_constraint(%12, %13, %14, %19, %20, %21, %22) {monomials = [{coeff = 256 : i64, vars = [3]}, {coeff = 960 : i64, vars = [4]}, {coeff = -15 : i64, vars = [6]}], upper_bound = 0 : i64}
    }
    func.func @matmul_kernel__d0i1_d1i1__f01__h_d(%arg0: memref<*xf32> {tt.divisibility = 16 : i32}, %arg1: memref<*xf32> {tt.divisibility = 16 : i32}, %arg2: memref<*xf32> {tt.divisibility = 16 : i32}, %arg3: index, %arg4: index, %arg5: index, %arg6: index, %arg7: index, %arg8: index) {
      affine.parallel (%arg9) = (0) to (8) {
        affine.parallel (%arg10) = (0) to (8) {
          affine.for %arg11 = 0 to affine_map<()[s0, s1] -> (s0)>()[%arg3, %arg4] {
            affine.for %arg12 = 0 to affine_map<()[s0, s1] -> ((s1 ceildiv 8) ceildiv 8)>()[%arg3, %arg4] {
              %12 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 8 + d1 * 8 + d2)>(%arg9, %arg10, %arg12)
              %c1024 = arith.constant 1024 : index
              %c1024_0 = arith.constant 1024 : index
              %c1 = arith.constant 1 : index
              %13 = loom.get_symbolic_block_size @constraints::@BM : index
              %14 = loom.get_symbolic_block_size @constraints::@BN : index
              %15 = loom.get_symbolic_block_size @constraints::@BK : index
              %16 = arith.ceildivsi %c1024_0, %15 : index
              %cst = arith.constant 0.000000e+00 : f32
              %c1024_1 = arith.constant 1024 : index
              %c0 = arith.constant 0 : index
              %17 = tensor.empty(%13, %14) : tensor<?x?xf32>
              %18 = linalg.fill ins(%cst : f32) outs(%17 : tensor<?x?xf32>) -> tensor<?x?xf32>
              %19 = arith.muli %arg11, %13 : index
              %20 = arith.muli %12, %14 : index
              %21 = scf.for %arg13 = %c0 to %16 step %c1 iter_args(%arg14 = %18) -> (tensor<?x?xf32>) {
                %25 = arith.muli %arg13, %15 : index
                %26 = arith.muli %19, %c1024_1 : index
                %27 = arith.addi %26, %25 : index
                %28 = loom.reinterpret_cast %arg0 to offset : [%27], sizes : [%13, %15], strides : [%c1024_0, %c1], reuse : [seq = false, spat = true, temp = true] : memref<*xf32> to memref<?x?xf32, strided<[?, ?], offset: ?>>
                %29 = loom.alloc(%13, %15) on @L1 : memref<?x?xf32>
                loom.copy %28, %29, interconnect : [@horizontal_links], broadcast : [1, 8] : memref<?x?xf32, strided<[?, ?], offset: ?>> to memref<?x?xf32>
                %30 = bufferization.to_tensor %29 restrict writable : memref<?x?xf32> to tensor<?x?xf32>
                %31 = arith.muli %25, %c1024_1 : index
                %32 = arith.addi %31, %20 : index
                %33 = loom.reinterpret_cast %arg1 to offset : [%32], sizes : [%15, %14], strides : [%c1024, %c1], reuse : [seq = false, spat = false, temp = true] : memref<*xf32> to memref<?x?xf32, strided<[?, ?], offset: ?>>
                %34 = loom.alloc(%15, %14) on @L1 : memref<?x?xf32>
                loom.copy %33, %34, interconnect : [], broadcast : [1, 1] : memref<?x?xf32, strided<[?, ?], offset: ?>> to memref<?x?xf32>
                %35 = bufferization.to_tensor %34 restrict writable : memref<?x?xf32> to tensor<?x?xf32>
                %36 = linalg.matmul ins(%30, %35 : tensor<?x?xf32>, tensor<?x?xf32>) outs(%18 : tensor<?x?xf32>) -> tensor<?x?xf32>
                %37 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%arg14, %36 : tensor<?x?xf32>, tensor<?x?xf32>) outs(%arg14 : tensor<?x?xf32>) {
                ^bb0(%in: f32, %in_2: f32, %out: f32):
                  %38 = arith.addf %in, %in_2 : f32
                  linalg.yield %38 : f32
                } -> tensor<?x?xf32>
                scf.yield %37 : tensor<?x?xf32>
              }
              %22 = arith.muli %19, %c1024_1 : index
              %23 = arith.addi %22, %20 : index
              %24 = loom.reinterpret_cast %arg2 to offset : [%23], sizes : [%13, %14], strides : [%c1024, %c1], reuse : [seq = false, spat = false, temp = false] : memref<*xf32> to memref<?x?xf32, strided<[?, ?], offset: ?>>
              bufferization.materialize_in_destination %21 in writable %24 : (tensor<?x?xf32>, memref<?x?xf32, strided<[?, ?], offset: ?>>) -> ()
            }
          }
        } {loom.mapped_to = @y}
      } {loom.mapped_to = @x}
      return
    }
  }
  module attributes {loom.pass_name = "EnumerateCopyBroadcast"} {
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
      %15 = loom.expression(%12, %13) {coeffs = [1, 1], logic = "mul"} : index
      %16 = loom.expression(%15, %14) {coeffs = [1, 1], logic = "mul"} : index
      loom.polynomial_constraint(%12, %13, %14, %15, %16) {monomials = [{coeff = -1 : i64, vars = [4]}], upper_bound = -262144 : i64}
      %17 = loom.expression(%12, %13) {coeffs = [1, 1], logic = "add"} : index
      %18 = loom.expression(%17, %14) {coeffs = [1, 1], logic = "mul"} : index
      loom.polynomial_constraint(%12, %14, %13, %17, %18) {monomials = [{coeff = 1 : i64, vars = [4]}], upper_bound = 374784 : i64}
      loom.polynomial_constraint(%13) {monomials = [{coeff = 1 : i64, vars = [0]}], upper_bound = 16 : i64}
      %19 = loom.expression(%12, %14) {coeffs = [1, 1], logic = "mul"} : index
      %20 = loom.expression(%14, %13) {coeffs = [1, 1], logic = "mul"} : index
      %21 = loom.expression(%12, %13) {coeffs = [1, 1], logic = "mul"} : index
      %22 = loom.expression(%14, %21) {coeffs = [1, 1], logic = "mul"} : index
      loom.polynomial_constraint(%12, %13, %14, %19, %20, %21, %22) {monomials = [{coeff = 256 : i64, vars = [3]}, {coeff = 960 : i64, vars = [4]}, {coeff = -15 : i64, vars = [6]}], upper_bound = 0 : i64}
    }
    func.func @matmul_kernel__d0i1_d1i1__f01__v_d(%arg0: memref<*xf32> {tt.divisibility = 16 : i32}, %arg1: memref<*xf32> {tt.divisibility = 16 : i32}, %arg2: memref<*xf32> {tt.divisibility = 16 : i32}, %arg3: index, %arg4: index, %arg5: index, %arg6: index, %arg7: index, %arg8: index) {
      affine.parallel (%arg9) = (0) to (8) {
        affine.parallel (%arg10) = (0) to (8) {
          affine.for %arg11 = 0 to affine_map<()[s0, s1] -> (s0)>()[%arg3, %arg4] {
            affine.for %arg12 = 0 to affine_map<()[s0, s1] -> ((s1 ceildiv 8) ceildiv 8)>()[%arg3, %arg4] {
              %12 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 8 + d1 * 8 + d2)>(%arg9, %arg10, %arg12)
              %c1024 = arith.constant 1024 : index
              %c1024_0 = arith.constant 1024 : index
              %c1 = arith.constant 1 : index
              %13 = loom.get_symbolic_block_size @constraints::@BM : index
              %14 = loom.get_symbolic_block_size @constraints::@BN : index
              %15 = loom.get_symbolic_block_size @constraints::@BK : index
              %16 = arith.ceildivsi %c1024_0, %15 : index
              %cst = arith.constant 0.000000e+00 : f32
              %c1024_1 = arith.constant 1024 : index
              %c0 = arith.constant 0 : index
              %17 = tensor.empty(%13, %14) : tensor<?x?xf32>
              %18 = linalg.fill ins(%cst : f32) outs(%17 : tensor<?x?xf32>) -> tensor<?x?xf32>
              %19 = arith.muli %arg11, %13 : index
              %20 = arith.muli %12, %14 : index
              %21 = scf.for %arg13 = %c0 to %16 step %c1 iter_args(%arg14 = %18) -> (tensor<?x?xf32>) {
                %25 = arith.muli %arg13, %15 : index
                %26 = arith.muli %19, %c1024_1 : index
                %27 = arith.addi %26, %25 : index
                %28 = loom.reinterpret_cast %arg0 to offset : [%27], sizes : [%13, %15], strides : [%c1024_0, %c1], reuse : [seq = false, spat = true, temp = true] : memref<*xf32> to memref<?x?xf32, strided<[?, ?], offset: ?>>
                %29 = loom.alloc(%13, %15) on @L1 : memref<?x?xf32>
                loom.copy %28, %29, interconnect : [@vertical_links], broadcast : [8, 1] : memref<?x?xf32, strided<[?, ?], offset: ?>> to memref<?x?xf32>
                %30 = bufferization.to_tensor %29 restrict writable : memref<?x?xf32> to tensor<?x?xf32>
                %31 = arith.muli %25, %c1024_1 : index
                %32 = arith.addi %31, %20 : index
                %33 = loom.reinterpret_cast %arg1 to offset : [%32], sizes : [%15, %14], strides : [%c1024, %c1], reuse : [seq = false, spat = false, temp = true] : memref<*xf32> to memref<?x?xf32, strided<[?, ?], offset: ?>>
                %34 = loom.alloc(%15, %14) on @L1 : memref<?x?xf32>
                loom.copy %33, %34, interconnect : [], broadcast : [1, 1] : memref<?x?xf32, strided<[?, ?], offset: ?>> to memref<?x?xf32>
                %35 = bufferization.to_tensor %34 restrict writable : memref<?x?xf32> to tensor<?x?xf32>
                %36 = linalg.matmul ins(%30, %35 : tensor<?x?xf32>, tensor<?x?xf32>) outs(%18 : tensor<?x?xf32>) -> tensor<?x?xf32>
                %37 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%arg14, %36 : tensor<?x?xf32>, tensor<?x?xf32>) outs(%arg14 : tensor<?x?xf32>) {
                ^bb0(%in: f32, %in_2: f32, %out: f32):
                  %38 = arith.addf %in, %in_2 : f32
                  linalg.yield %38 : f32
                } -> tensor<?x?xf32>
                scf.yield %37 : tensor<?x?xf32>
              }
              %22 = arith.muli %19, %c1024_1 : index
              %23 = arith.addi %22, %20 : index
              %24 = loom.reinterpret_cast %arg2 to offset : [%23], sizes : [%13, %14], strides : [%c1024, %c1], reuse : [seq = false, spat = false, temp = false] : memref<*xf32> to memref<?x?xf32, strided<[?, ?], offset: ?>>
              bufferization.materialize_in_destination %21 in writable %24 : (tensor<?x?xf32>, memref<?x?xf32, strided<[?, ?], offset: ?>>) -> ()
            }
          }
        } {loom.mapped_to = @y}
      } {loom.mapped_to = @x}
      return
    }
  }
  module attributes {loom.pass_name = "EnumerateCopyBroadcast"} {
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
      %15 = loom.expression(%12, %13) {coeffs = [1, 1], logic = "mul"} : index
      %16 = loom.expression(%15, %14) {coeffs = [1, 1], logic = "mul"} : index
      loom.polynomial_constraint(%12, %13, %14, %15, %16) {monomials = [{coeff = -1 : i64, vars = [4]}], upper_bound = -262144 : i64}
      %17 = loom.expression(%12, %13) {coeffs = [1, 1], logic = "add"} : index
      %18 = loom.expression(%17, %14) {coeffs = [1, 1], logic = "mul"} : index
      loom.polynomial_constraint(%12, %14, %13, %17, %18) {monomials = [{coeff = 1 : i64, vars = [4]}], upper_bound = 374784 : i64}
      loom.polynomial_constraint(%13) {monomials = [{coeff = 1 : i64, vars = [0]}], upper_bound = 16 : i64}
      %19 = loom.expression(%13, %14) {coeffs = [1, 1], logic = "mul"} : index
      loom.polynomial_constraint(%12, %14, %13, %19) {monomials = [{coeff = 1024 : i64, vars = [0]}, {coeff = 1 : i64, vars = [3]}], upper_bound = 374784 : i64}
      %20 = loom.expression(%12, %14) {coeffs = [1, 1], logic = "mul"} : index
      %21 = loom.expression(%13, %14) {coeffs = [1, 1], logic = "mul"} : index
      %22 = loom.expression(%12, %13) {coeffs = [1, 1], logic = "mul"} : index
      %23 = loom.expression(%14, %22) {coeffs = [1, 1], logic = "mul"} : index
      loom.polynomial_constraint(%12, %13, %14, %20, %21, %22, %23) {monomials = [{coeff = 64 : i64, vars = [3]}, {coeff = 64 : i64, vars = [4]}, {coeff = -1 : i64, vars = [6]}], upper_bound = 0 : i64}
    }
    func.func @matmul_kernel__d0i1_d1i1__f01__hoist_block_0__d_d(%arg0: memref<*xf32> {tt.divisibility = 16 : i32}, %arg1: memref<*xf32> {tt.divisibility = 16 : i32}, %arg2: memref<*xf32> {tt.divisibility = 16 : i32}, %arg3: index, %arg4: index, %arg5: index, %arg6: index, %arg7: index, %arg8: index) {
      affine.parallel (%arg9) = (0) to (8) {
        affine.parallel (%arg10) = (0) to (8) {
          affine.for %arg11 = 0 to affine_map<()[s0, s1] -> (s0)>()[%arg3, %arg4] {
            affine.for %arg12 = 0 to affine_map<()[s0, s1] -> ((s1 ceildiv 8) ceildiv 8)>()[%arg3, %arg4] {
              %12 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 8 + d1 * 8 + d2)>(%arg9, %arg10, %arg12)
              %c1024 = arith.constant 1024 : index
              %c1024_0 = arith.constant 1024 : index
              %c1 = arith.constant 1 : index
              %13 = loom.get_symbolic_block_size @constraints::@BM : index
              %14 = loom.get_symbolic_block_size @constraints::@BN : index
              %15 = loom.get_symbolic_block_size @constraints::@BK : index
              %16 = arith.ceildivsi %c1024_0, %15 : index
              %cst = arith.constant 0.000000e+00 : f32
              %c1024_1 = arith.constant 1024 : index
              %c0 = arith.constant 0 : index
              %17 = tensor.empty(%13, %14) : tensor<?x?xf32>
              %18 = linalg.fill ins(%cst : f32) outs(%17 : tensor<?x?xf32>) -> tensor<?x?xf32>
              %19 = arith.muli %arg11, %13 : index
              %20 = arith.muli %12, %14 : index
              %21 = arith.muli %13, %c1024_0 : index
              %c0_2 = arith.constant 0 : index
              %22 = arith.muli %19, %c1024_1 : index
              %23 = arith.muli %c0_2, %15 : index
              %24 = arith.addi %22, %23 : index
              %25 = loom.reinterpret_cast %arg0 to offset : [%24], sizes : [%16, %13, %15], strides : [%21, %c1024_0, %c1], reuse : [seq = false, spat = true, temp = true] : memref<*xf32> to memref<?x?x?xf32, strided<[?, ?, ?], offset: ?>>
              %26 = loom.alloc(%16, %13, %15) on @L1 : memref<?x?x?xf32>
              loom.copy %25, %26, interconnect : [], broadcast : [1, 1] : memref<?x?x?xf32, strided<[?, ?, ?], offset: ?>> to memref<?x?x?xf32>
              %27 = scf.for %arg13 = %c0 to %16 step %c1 iter_args(%arg14 = %18) -> (tensor<?x?xf32>) {
                %31 = arith.muli %arg13, %15 : index
                %32 = arith.muli %19, %c1024_1 : index
                %subview = memref.subview %26[%arg13, 0, 0] [1, %13, %15] [1, 1, 1] : memref<?x?x?xf32> to memref<?x?xf32, strided<[?, 1], offset: ?>>
                %33 = bufferization.to_tensor %subview restrict writable : memref<?x?xf32, strided<[?, 1], offset: ?>> to tensor<?x?xf32>
                %34 = arith.muli %31, %c1024_1 : index
                %35 = arith.addi %34, %20 : index
                %36 = loom.reinterpret_cast %arg1 to offset : [%35], sizes : [%15, %14], strides : [%c1024, %c1], reuse : [seq = false, spat = false, temp = true] : memref<*xf32> to memref<?x?xf32, strided<[?, ?], offset: ?>>
                %37 = loom.alloc(%15, %14) on @L1 : memref<?x?xf32>
                loom.copy %36, %37, interconnect : [], broadcast : [1, 1] : memref<?x?xf32, strided<[?, ?], offset: ?>> to memref<?x?xf32>
                %38 = bufferization.to_tensor %37 restrict writable : memref<?x?xf32> to tensor<?x?xf32>
                %39 = linalg.matmul ins(%33, %38 : tensor<?x?xf32>, tensor<?x?xf32>) outs(%18 : tensor<?x?xf32>) -> tensor<?x?xf32>
                %40 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%arg14, %39 : tensor<?x?xf32>, tensor<?x?xf32>) outs(%arg14 : tensor<?x?xf32>) {
                ^bb0(%in: f32, %in_3: f32, %out: f32):
                  %41 = arith.addf %in, %in_3 : f32
                  linalg.yield %41 : f32
                } -> tensor<?x?xf32>
                scf.yield %40 : tensor<?x?xf32>
              }
              %28 = arith.muli %19, %c1024_1 : index
              %29 = arith.addi %28, %20 : index
              %30 = loom.reinterpret_cast %arg2 to offset : [%29], sizes : [%13, %14], strides : [%c1024, %c1], reuse : [seq = false, spat = false, temp = false] : memref<*xf32> to memref<?x?xf32, strided<[?, ?], offset: ?>>
              bufferization.materialize_in_destination %27 in writable %30 : (tensor<?x?xf32>, memref<?x?xf32, strided<[?, ?], offset: ?>>) -> ()
            }
          }
        } {loom.mapped_to = @y}
      } {loom.mapped_to = @x}
      return
    }
  }
  module attributes {loom.pass_name = "EnumerateCopyBroadcast"} {
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
      %15 = loom.expression(%12, %13) {coeffs = [1, 1], logic = "mul"} : index
      %16 = loom.expression(%15, %14) {coeffs = [1, 1], logic = "mul"} : index
      loom.polynomial_constraint(%12, %13, %14, %15, %16) {monomials = [{coeff = -1 : i64, vars = [4]}], upper_bound = -262144 : i64}
      %17 = loom.expression(%12, %13) {coeffs = [1, 1], logic = "add"} : index
      %18 = loom.expression(%17, %14) {coeffs = [1, 1], logic = "mul"} : index
      loom.polynomial_constraint(%12, %14, %13, %17, %18) {monomials = [{coeff = 1 : i64, vars = [4]}], upper_bound = 374784 : i64}
      loom.polynomial_constraint(%13) {monomials = [{coeff = 1 : i64, vars = [0]}], upper_bound = 16 : i64}
      %19 = loom.expression(%13, %14) {coeffs = [1, 1], logic = "mul"} : index
      loom.polynomial_constraint(%12, %14, %13, %19) {monomials = [{coeff = 1024 : i64, vars = [0]}, {coeff = 1 : i64, vars = [3]}], upper_bound = 374784 : i64}
      %20 = loom.expression(%12, %14) {coeffs = [1, 1], logic = "mul"} : index
      %21 = loom.expression(%13, %14) {coeffs = [1, 1], logic = "mul"} : index
      %22 = loom.expression(%12, %13) {coeffs = [1, 1], logic = "mul"} : index
      %23 = loom.expression(%14, %22) {coeffs = [1, 1], logic = "mul"} : index
      loom.polynomial_constraint(%12, %13, %14, %20, %21, %22, %23) {monomials = [{coeff = 256 : i64, vars = [3]}, {coeff = 960 : i64, vars = [4]}, {coeff = -15 : i64, vars = [6]}], upper_bound = 0 : i64}
    }
    func.func @matmul_kernel__d0i1_d1i1__f01__hoist_block_0__a_d(%arg0: memref<*xf32> {tt.divisibility = 16 : i32}, %arg1: memref<*xf32> {tt.divisibility = 16 : i32}, %arg2: memref<*xf32> {tt.divisibility = 16 : i32}, %arg3: index, %arg4: index, %arg5: index, %arg6: index, %arg7: index, %arg8: index) {
      affine.parallel (%arg9) = (0) to (8) {
        affine.parallel (%arg10) = (0) to (8) {
          affine.for %arg11 = 0 to affine_map<()[s0, s1] -> (s0)>()[%arg3, %arg4] {
            affine.for %arg12 = 0 to affine_map<()[s0, s1] -> ((s1 ceildiv 8) ceildiv 8)>()[%arg3, %arg4] {
              %12 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 8 + d1 * 8 + d2)>(%arg9, %arg10, %arg12)
              %c1024 = arith.constant 1024 : index
              %c1024_0 = arith.constant 1024 : index
              %c1 = arith.constant 1 : index
              %13 = loom.get_symbolic_block_size @constraints::@BM : index
              %14 = loom.get_symbolic_block_size @constraints::@BN : index
              %15 = loom.get_symbolic_block_size @constraints::@BK : index
              %16 = arith.ceildivsi %c1024_0, %15 : index
              %cst = arith.constant 0.000000e+00 : f32
              %c1024_1 = arith.constant 1024 : index
              %c0 = arith.constant 0 : index
              %17 = tensor.empty(%13, %14) : tensor<?x?xf32>
              %18 = linalg.fill ins(%cst : f32) outs(%17 : tensor<?x?xf32>) -> tensor<?x?xf32>
              %19 = arith.muli %arg11, %13 : index
              %20 = arith.muli %12, %14 : index
              %21 = arith.muli %13, %c1024_0 : index
              %c0_2 = arith.constant 0 : index
              %22 = arith.muli %19, %c1024_1 : index
              %23 = arith.muli %c0_2, %15 : index
              %24 = arith.addi %22, %23 : index
              %25 = loom.reinterpret_cast %arg0 to offset : [%24], sizes : [%16, %13, %15], strides : [%21, %c1024_0, %c1], reuse : [seq = false, spat = true, temp = true] : memref<*xf32> to memref<?x?x?xf32, strided<[?, ?, ?], offset: ?>>
              %26 = loom.alloc(%16, %13, %15) on @L1 : memref<?x?x?xf32>
              loom.copy %25, %26, interconnect : [@horizontal_links, @vertical_links], broadcast : [8, 8] : memref<?x?x?xf32, strided<[?, ?, ?], offset: ?>> to memref<?x?x?xf32>
              %27 = scf.for %arg13 = %c0 to %16 step %c1 iter_args(%arg14 = %18) -> (tensor<?x?xf32>) {
                %31 = arith.muli %arg13, %15 : index
                %32 = arith.muli %19, %c1024_1 : index
                %subview = memref.subview %26[%arg13, 0, 0] [1, %13, %15] [1, 1, 1] : memref<?x?x?xf32> to memref<?x?xf32, strided<[?, 1], offset: ?>>
                %33 = bufferization.to_tensor %subview restrict writable : memref<?x?xf32, strided<[?, 1], offset: ?>> to tensor<?x?xf32>
                %34 = arith.muli %31, %c1024_1 : index
                %35 = arith.addi %34, %20 : index
                %36 = loom.reinterpret_cast %arg1 to offset : [%35], sizes : [%15, %14], strides : [%c1024, %c1], reuse : [seq = false, spat = false, temp = true] : memref<*xf32> to memref<?x?xf32, strided<[?, ?], offset: ?>>
                %37 = loom.alloc(%15, %14) on @L1 : memref<?x?xf32>
                loom.copy %36, %37, interconnect : [], broadcast : [1, 1] : memref<?x?xf32, strided<[?, ?], offset: ?>> to memref<?x?xf32>
                %38 = bufferization.to_tensor %37 restrict writable : memref<?x?xf32> to tensor<?x?xf32>
                %39 = linalg.matmul ins(%33, %38 : tensor<?x?xf32>, tensor<?x?xf32>) outs(%18 : tensor<?x?xf32>) -> tensor<?x?xf32>
                %40 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%arg14, %39 : tensor<?x?xf32>, tensor<?x?xf32>) outs(%arg14 : tensor<?x?xf32>) {
                ^bb0(%in: f32, %in_3: f32, %out: f32):
                  %41 = arith.addf %in, %in_3 : f32
                  linalg.yield %41 : f32
                } -> tensor<?x?xf32>
                scf.yield %40 : tensor<?x?xf32>
              }
              %28 = arith.muli %19, %c1024_1 : index
              %29 = arith.addi %28, %20 : index
              %30 = loom.reinterpret_cast %arg2 to offset : [%29], sizes : [%13, %14], strides : [%c1024, %c1], reuse : [seq = false, spat = false, temp = false] : memref<*xf32> to memref<?x?xf32, strided<[?, ?], offset: ?>>
              bufferization.materialize_in_destination %27 in writable %30 : (tensor<?x?xf32>, memref<?x?xf32, strided<[?, ?], offset: ?>>) -> ()
            }
          }
        } {loom.mapped_to = @y}
      } {loom.mapped_to = @x}
      return
    }
  }
  module attributes {loom.pass_name = "EnumerateCopyBroadcast"} {
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
      %15 = loom.expression(%13, %12) {coeffs = [1, 1], logic = "mul"} : index
      %16 = loom.expression(%15, %14) {coeffs = [1, 1], logic = "mul"} : index
      loom.polynomial_constraint(%12, %13, %14, %15, %16) {monomials = [{coeff = -1 : i64, vars = [4]}], upper_bound = -262144 : i64}
      %17 = loom.expression(%12, %13) {coeffs = [1, 1], logic = "add"} : index
      %18 = loom.expression(%14, %17) {coeffs = [1, 1], logic = "mul"} : index
      loom.polynomial_constraint(%12, %14, %13, %17, %18) {monomials = [{coeff = 1 : i64, vars = [4]}], upper_bound = 374784 : i64}
      loom.polynomial_constraint(%13) {monomials = [{coeff = 1 : i64, vars = [0]}], upper_bound = 16 : i64}
      %19 = loom.expression(%13, %14) {coeffs = [1, 1], logic = "mul"} : index
      loom.polynomial_constraint(%12, %14, %13, %19) {monomials = [{coeff = 1024 : i64, vars = [0]}, {coeff = 1 : i64, vars = [3]}], upper_bound = 374784 : i64}
      %20 = loom.expression(%14, %12) {coeffs = [1, 1], logic = "mul"} : index
      %21 = loom.expression(%13, %14) {coeffs = [1, 1], logic = "mul"} : index
      %22 = loom.expression(%13, %12) {coeffs = [1, 1], logic = "mul"} : index
      %23 = loom.expression(%14, %22) {coeffs = [1, 1], logic = "mul"} : index
      loom.polynomial_constraint(%12, %13, %14, %20, %21, %22, %23) {monomials = [{coeff = 256 : i64, vars = [3]}, {coeff = 960 : i64, vars = [4]}, {coeff = -15 : i64, vars = [6]}], upper_bound = 0 : i64}
    }
    func.func @matmul_kernel__d0i1_d1i1__f01__hoist_block_0__h_d(%arg0: memref<*xf32> {tt.divisibility = 16 : i32}, %arg1: memref<*xf32> {tt.divisibility = 16 : i32}, %arg2: memref<*xf32> {tt.divisibility = 16 : i32}, %arg3: index, %arg4: index, %arg5: index, %arg6: index, %arg7: index, %arg8: index) {
      affine.parallel (%arg9) = (0) to (8) {
        affine.parallel (%arg10) = (0) to (8) {
          affine.for %arg11 = 0 to affine_map<()[s0, s1] -> (s0)>()[%arg3, %arg4] {
            affine.for %arg12 = 0 to affine_map<()[s0, s1] -> ((s1 ceildiv 8) ceildiv 8)>()[%arg3, %arg4] {
              %12 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 8 + d1 * 8 + d2)>(%arg9, %arg10, %arg12)
              %c1024 = arith.constant 1024 : index
              %c1024_0 = arith.constant 1024 : index
              %c1 = arith.constant 1 : index
              %13 = loom.get_symbolic_block_size @constraints::@BM : index
              %14 = loom.get_symbolic_block_size @constraints::@BN : index
              %15 = loom.get_symbolic_block_size @constraints::@BK : index
              %16 = arith.ceildivsi %c1024_0, %15 : index
              %cst = arith.constant 0.000000e+00 : f32
              %c1024_1 = arith.constant 1024 : index
              %c0 = arith.constant 0 : index
              %17 = tensor.empty(%13, %14) : tensor<?x?xf32>
              %18 = linalg.fill ins(%cst : f32) outs(%17 : tensor<?x?xf32>) -> tensor<?x?xf32>
              %19 = arith.muli %arg11, %13 : index
              %20 = arith.muli %12, %14 : index
              %21 = arith.muli %13, %c1024_0 : index
              %c0_2 = arith.constant 0 : index
              %22 = arith.muli %19, %c1024_1 : index
              %23 = arith.muli %c0_2, %15 : index
              %24 = arith.addi %22, %23 : index
              %25 = loom.reinterpret_cast %arg0 to offset : [%24], sizes : [%16, %13, %15], strides : [%21, %c1024_0, %c1], reuse : [seq = false, spat = true, temp = true] : memref<*xf32> to memref<?x?x?xf32, strided<[?, ?, ?], offset: ?>>
              %26 = loom.alloc(%16, %13, %15) on @L1 : memref<?x?x?xf32>
              loom.copy %25, %26, interconnect : [@horizontal_links], broadcast : [1, 8] : memref<?x?x?xf32, strided<[?, ?, ?], offset: ?>> to memref<?x?x?xf32>
              %27 = scf.for %arg13 = %c0 to %16 step %c1 iter_args(%arg14 = %18) -> (tensor<?x?xf32>) {
                %31 = arith.muli %arg13, %15 : index
                %32 = arith.muli %19, %c1024_1 : index
                %subview = memref.subview %26[%arg13, 0, 0] [1, %13, %15] [1, 1, 1] : memref<?x?x?xf32> to memref<?x?xf32, strided<[?, 1], offset: ?>>
                %33 = bufferization.to_tensor %subview restrict writable : memref<?x?xf32, strided<[?, 1], offset: ?>> to tensor<?x?xf32>
                %34 = arith.muli %31, %c1024_1 : index
                %35 = arith.addi %34, %20 : index
                %36 = loom.reinterpret_cast %arg1 to offset : [%35], sizes : [%15, %14], strides : [%c1024, %c1], reuse : [seq = false, spat = false, temp = true] : memref<*xf32> to memref<?x?xf32, strided<[?, ?], offset: ?>>
                %37 = loom.alloc(%15, %14) on @L1 : memref<?x?xf32>
                loom.copy %36, %37, interconnect : [], broadcast : [1, 1] : memref<?x?xf32, strided<[?, ?], offset: ?>> to memref<?x?xf32>
                %38 = bufferization.to_tensor %37 restrict writable : memref<?x?xf32> to tensor<?x?xf32>
                %39 = linalg.matmul ins(%33, %38 : tensor<?x?xf32>, tensor<?x?xf32>) outs(%18 : tensor<?x?xf32>) -> tensor<?x?xf32>
                %40 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%arg14, %39 : tensor<?x?xf32>, tensor<?x?xf32>) outs(%arg14 : tensor<?x?xf32>) {
                ^bb0(%in: f32, %in_3: f32, %out: f32):
                  %41 = arith.addf %in, %in_3 : f32
                  linalg.yield %41 : f32
                } -> tensor<?x?xf32>
                scf.yield %40 : tensor<?x?xf32>
              }
              %28 = arith.muli %19, %c1024_1 : index
              %29 = arith.addi %28, %20 : index
              %30 = loom.reinterpret_cast %arg2 to offset : [%29], sizes : [%13, %14], strides : [%c1024, %c1], reuse : [seq = false, spat = false, temp = false] : memref<*xf32> to memref<?x?xf32, strided<[?, ?], offset: ?>>
              bufferization.materialize_in_destination %27 in writable %30 : (tensor<?x?xf32>, memref<?x?xf32, strided<[?, ?], offset: ?>>) -> ()
            }
          }
        } {loom.mapped_to = @y}
      } {loom.mapped_to = @x}
      return
    }
  }
  module attributes {loom.pass_name = "EnumerateCopyBroadcast"} {
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
      %15 = loom.expression(%13, %12) {coeffs = [1, 1], logic = "mul"} : index
      %16 = loom.expression(%15, %14) {coeffs = [1, 1], logic = "mul"} : index
      loom.polynomial_constraint(%12, %13, %14, %15, %16) {monomials = [{coeff = -1 : i64, vars = [4]}], upper_bound = -262144 : i64}
      %17 = loom.expression(%12, %13) {coeffs = [1, 1], logic = "add"} : index
      %18 = loom.expression(%14, %17) {coeffs = [1, 1], logic = "mul"} : index
      loom.polynomial_constraint(%12, %14, %13, %17, %18) {monomials = [{coeff = 1 : i64, vars = [4]}], upper_bound = 374784 : i64}
      loom.polynomial_constraint(%13) {monomials = [{coeff = 1 : i64, vars = [0]}], upper_bound = 16 : i64}
      %19 = loom.expression(%13, %14) {coeffs = [1, 1], logic = "mul"} : index
      loom.polynomial_constraint(%12, %14, %13, %19) {monomials = [{coeff = 1024 : i64, vars = [0]}, {coeff = 1 : i64, vars = [3]}], upper_bound = 374784 : i64}
      %20 = loom.expression(%14, %12) {coeffs = [1, 1], logic = "mul"} : index
      %21 = loom.expression(%13, %14) {coeffs = [1, 1], logic = "mul"} : index
      %22 = loom.expression(%13, %12) {coeffs = [1, 1], logic = "mul"} : index
      %23 = loom.expression(%14, %22) {coeffs = [1, 1], logic = "mul"} : index
      loom.polynomial_constraint(%12, %13, %14, %20, %21, %22, %23) {monomials = [{coeff = 256 : i64, vars = [3]}, {coeff = 960 : i64, vars = [4]}, {coeff = -15 : i64, vars = [6]}], upper_bound = 0 : i64}
    }
    func.func @matmul_kernel__d0i1_d1i1__f01__hoist_block_0__v_d(%arg0: memref<*xf32> {tt.divisibility = 16 : i32}, %arg1: memref<*xf32> {tt.divisibility = 16 : i32}, %arg2: memref<*xf32> {tt.divisibility = 16 : i32}, %arg3: index, %arg4: index, %arg5: index, %arg6: index, %arg7: index, %arg8: index) {
      affine.parallel (%arg9) = (0) to (8) {
        affine.parallel (%arg10) = (0) to (8) {
          affine.for %arg11 = 0 to affine_map<()[s0, s1] -> (s0)>()[%arg3, %arg4] {
            affine.for %arg12 = 0 to affine_map<()[s0, s1] -> ((s1 ceildiv 8) ceildiv 8)>()[%arg3, %arg4] {
              %12 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 8 + d1 * 8 + d2)>(%arg9, %arg10, %arg12)
              %c1024 = arith.constant 1024 : index
              %c1024_0 = arith.constant 1024 : index
              %c1 = arith.constant 1 : index
              %13 = loom.get_symbolic_block_size @constraints::@BM : index
              %14 = loom.get_symbolic_block_size @constraints::@BN : index
              %15 = loom.get_symbolic_block_size @constraints::@BK : index
              %16 = arith.ceildivsi %c1024_0, %15 : index
              %cst = arith.constant 0.000000e+00 : f32
              %c1024_1 = arith.constant 1024 : index
              %c0 = arith.constant 0 : index
              %17 = tensor.empty(%13, %14) : tensor<?x?xf32>
              %18 = linalg.fill ins(%cst : f32) outs(%17 : tensor<?x?xf32>) -> tensor<?x?xf32>
              %19 = arith.muli %arg11, %13 : index
              %20 = arith.muli %12, %14 : index
              %21 = arith.muli %13, %c1024_0 : index
              %c0_2 = arith.constant 0 : index
              %22 = arith.muli %19, %c1024_1 : index
              %23 = arith.muli %c0_2, %15 : index
              %24 = arith.addi %22, %23 : index
              %25 = loom.reinterpret_cast %arg0 to offset : [%24], sizes : [%16, %13, %15], strides : [%21, %c1024_0, %c1], reuse : [seq = false, spat = true, temp = true] : memref<*xf32> to memref<?x?x?xf32, strided<[?, ?, ?], offset: ?>>
              %26 = loom.alloc(%16, %13, %15) on @L1 : memref<?x?x?xf32>
              loom.copy %25, %26, interconnect : [@vertical_links], broadcast : [8, 1] : memref<?x?x?xf32, strided<[?, ?, ?], offset: ?>> to memref<?x?x?xf32>
              %27 = scf.for %arg13 = %c0 to %16 step %c1 iter_args(%arg14 = %18) -> (tensor<?x?xf32>) {
                %31 = arith.muli %arg13, %15 : index
                %32 = arith.muli %19, %c1024_1 : index
                %subview = memref.subview %26[%arg13, 0, 0] [1, %13, %15] [1, 1, 1] : memref<?x?x?xf32> to memref<?x?xf32, strided<[?, 1], offset: ?>>
                %33 = bufferization.to_tensor %subview restrict writable : memref<?x?xf32, strided<[?, 1], offset: ?>> to tensor<?x?xf32>
                %34 = arith.muli %31, %c1024_1 : index
                %35 = arith.addi %34, %20 : index
                %36 = loom.reinterpret_cast %arg1 to offset : [%35], sizes : [%15, %14], strides : [%c1024, %c1], reuse : [seq = false, spat = false, temp = true] : memref<*xf32> to memref<?x?xf32, strided<[?, ?], offset: ?>>
                %37 = loom.alloc(%15, %14) on @L1 : memref<?x?xf32>
                loom.copy %36, %37, interconnect : [], broadcast : [1, 1] : memref<?x?xf32, strided<[?, ?], offset: ?>> to memref<?x?xf32>
                %38 = bufferization.to_tensor %37 restrict writable : memref<?x?xf32> to tensor<?x?xf32>
                %39 = linalg.matmul ins(%33, %38 : tensor<?x?xf32>, tensor<?x?xf32>) outs(%18 : tensor<?x?xf32>) -> tensor<?x?xf32>
                %40 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%arg14, %39 : tensor<?x?xf32>, tensor<?x?xf32>) outs(%arg14 : tensor<?x?xf32>) {
                ^bb0(%in: f32, %in_3: f32, %out: f32):
                  %41 = arith.addf %in, %in_3 : f32
                  linalg.yield %41 : f32
                } -> tensor<?x?xf32>
                scf.yield %40 : tensor<?x?xf32>
              }
              %28 = arith.muli %19, %c1024_1 : index
              %29 = arith.addi %28, %20 : index
              %30 = loom.reinterpret_cast %arg2 to offset : [%29], sizes : [%13, %14], strides : [%c1024, %c1], reuse : [seq = false, spat = false, temp = false] : memref<*xf32> to memref<?x?xf32, strided<[?, ?], offset: ?>>
              bufferization.materialize_in_destination %27 in writable %30 : (tensor<?x?xf32>, memref<?x?xf32, strided<[?, ?], offset: ?>>) -> ()
            }
          }
        } {loom.mapped_to = @y}
      } {loom.mapped_to = @x}
      return
    }
  }
  module attributes {loom.pass_name = "EnumerateCopyBroadcast"} {
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
      %15 = loom.expression(%13, %12) {coeffs = [1, 1], logic = "mul"} : index
      %16 = loom.expression(%15, %14) {coeffs = [1, 1], logic = "mul"} : index
      loom.polynomial_constraint(%12, %13, %14, %15, %16) {monomials = [{coeff = -1 : i64, vars = [4]}], upper_bound = -262144 : i64}
      %17 = loom.expression(%12, %13) {coeffs = [1, 1], logic = "add"} : index
      %18 = loom.expression(%14, %17) {coeffs = [1, 1], logic = "mul"} : index
      loom.polynomial_constraint(%12, %14, %13, %17, %18) {monomials = [{coeff = 1 : i64, vars = [4]}], upper_bound = 374784 : i64}
      loom.polynomial_constraint(%13) {monomials = [{coeff = 1 : i64, vars = [0]}], upper_bound = 16 : i64}
      %19 = loom.expression(%14, %12) {coeffs = [1, 1], logic = "mul"} : index
      loom.polynomial_constraint(%13, %12, %14, %19) {monomials = [{coeff = 1024 : i64, vars = [0]}, {coeff = 1 : i64, vars = [3]}], upper_bound = 374784 : i64}
      %20 = loom.expression(%14, %12) {coeffs = [1, 1], logic = "mul"} : index
      %21 = loom.expression(%13, %14) {coeffs = [1, 1], logic = "mul"} : index
      %22 = loom.expression(%13, %12) {coeffs = [1, 1], logic = "mul"} : index
      %23 = loom.expression(%14, %22) {coeffs = [1, 1], logic = "mul"} : index
      loom.polynomial_constraint(%12, %13, %14, %20, %21, %22, %23) {monomials = [{coeff = 64 : i64, vars = [3]}, {coeff = 64 : i64, vars = [4]}, {coeff = -1 : i64, vars = [6]}], upper_bound = 0 : i64}
    }
    func.func @matmul_kernel__d0i1_d1i1__f01__hoist_block_1__d_d(%arg0: memref<*xf32> {tt.divisibility = 16 : i32}, %arg1: memref<*xf32> {tt.divisibility = 16 : i32}, %arg2: memref<*xf32> {tt.divisibility = 16 : i32}, %arg3: index, %arg4: index, %arg5: index, %arg6: index, %arg7: index, %arg8: index) {
      affine.parallel (%arg9) = (0) to (8) {
        affine.parallel (%arg10) = (0) to (8) {
          affine.for %arg11 = 0 to affine_map<()[s0, s1] -> (s0)>()[%arg3, %arg4] {
            affine.for %arg12 = 0 to affine_map<()[s0, s1] -> ((s1 ceildiv 8) ceildiv 8)>()[%arg3, %arg4] {
              %12 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 8 + d1 * 8 + d2)>(%arg9, %arg10, %arg12)
              %c1024 = arith.constant 1024 : index
              %c1024_0 = arith.constant 1024 : index
              %c1 = arith.constant 1 : index
              %13 = loom.get_symbolic_block_size @constraints::@BM : index
              %14 = loom.get_symbolic_block_size @constraints::@BN : index
              %15 = loom.get_symbolic_block_size @constraints::@BK : index
              %16 = arith.ceildivsi %c1024_0, %15 : index
              %cst = arith.constant 0.000000e+00 : f32
              %c1024_1 = arith.constant 1024 : index
              %c0 = arith.constant 0 : index
              %17 = tensor.empty(%13, %14) : tensor<?x?xf32>
              %18 = linalg.fill ins(%cst : f32) outs(%17 : tensor<?x?xf32>) -> tensor<?x?xf32>
              %19 = arith.muli %arg11, %13 : index
              %20 = arith.muli %12, %14 : index
              %21 = arith.muli %15, %c1024 : index
              %c0_2 = arith.constant 0 : index
              %22 = arith.muli %c0_2, %15 : index
              %23 = arith.muli %22, %c1024_1 : index
              %24 = arith.addi %23, %20 : index
              %25 = loom.reinterpret_cast %arg1 to offset : [%24], sizes : [%16, %15, %14], strides : [%21, %c1024, %c1], reuse : [seq = false, spat = false, temp = true] : memref<*xf32> to memref<?x?x?xf32, strided<[?, ?, ?], offset: ?>>
              %26 = loom.alloc(%16, %15, %14) on @L1 : memref<?x?x?xf32>
              loom.copy %25, %26, interconnect : [], broadcast : [1, 1] : memref<?x?x?xf32, strided<[?, ?, ?], offset: ?>> to memref<?x?x?xf32>
              %27 = scf.for %arg13 = %c0 to %16 step %c1 iter_args(%arg14 = %18) -> (tensor<?x?xf32>) {
                %31 = arith.muli %arg13, %15 : index
                %32 = arith.muli %19, %c1024_1 : index
                %33 = arith.addi %32, %31 : index
                %34 = loom.reinterpret_cast %arg0 to offset : [%33], sizes : [%13, %15], strides : [%c1024_0, %c1], reuse : [seq = false, spat = true, temp = true] : memref<*xf32> to memref<?x?xf32, strided<[?, ?], offset: ?>>
                %35 = loom.alloc(%13, %15) on @L1 : memref<?x?xf32>
                loom.copy %34, %35, interconnect : [], broadcast : [1, 1] : memref<?x?xf32, strided<[?, ?], offset: ?>> to memref<?x?xf32>
                %36 = bufferization.to_tensor %35 restrict writable : memref<?x?xf32> to tensor<?x?xf32>
                %37 = arith.muli %31, %c1024_1 : index
                %subview = memref.subview %26[%arg13, 0, 0] [1, %15, %14] [1, 1, 1] : memref<?x?x?xf32> to memref<?x?xf32, strided<[?, 1], offset: ?>>
                %38 = bufferization.to_tensor %subview restrict writable : memref<?x?xf32, strided<[?, 1], offset: ?>> to tensor<?x?xf32>
                %39 = linalg.matmul ins(%36, %38 : tensor<?x?xf32>, tensor<?x?xf32>) outs(%18 : tensor<?x?xf32>) -> tensor<?x?xf32>
                %40 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%arg14, %39 : tensor<?x?xf32>, tensor<?x?xf32>) outs(%arg14 : tensor<?x?xf32>) {
                ^bb0(%in: f32, %in_3: f32, %out: f32):
                  %41 = arith.addf %in, %in_3 : f32
                  linalg.yield %41 : f32
                } -> tensor<?x?xf32>
                scf.yield %40 : tensor<?x?xf32>
              }
              %28 = arith.muli %19, %c1024_1 : index
              %29 = arith.addi %28, %20 : index
              %30 = loom.reinterpret_cast %arg2 to offset : [%29], sizes : [%13, %14], strides : [%c1024, %c1], reuse : [seq = false, spat = false, temp = false] : memref<*xf32> to memref<?x?xf32, strided<[?, ?], offset: ?>>
              bufferization.materialize_in_destination %27 in writable %30 : (tensor<?x?xf32>, memref<?x?xf32, strided<[?, ?], offset: ?>>) -> ()
            }
          }
        } {loom.mapped_to = @y}
      } {loom.mapped_to = @x}
      return
    }
  }
  module attributes {loom.pass_name = "EnumerateCopyBroadcast"} {
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
      %15 = loom.expression(%13, %12) {coeffs = [1, 1], logic = "mul"} : index
      %16 = loom.expression(%15, %14) {coeffs = [1, 1], logic = "mul"} : index
      loom.polynomial_constraint(%12, %13, %14, %15, %16) {monomials = [{coeff = -1 : i64, vars = [4]}], upper_bound = -262144 : i64}
      %17 = loom.expression(%12, %13) {coeffs = [1, 1], logic = "add"} : index
      %18 = loom.expression(%14, %17) {coeffs = [1, 1], logic = "mul"} : index
      loom.polynomial_constraint(%12, %14, %13, %17, %18) {monomials = [{coeff = 1 : i64, vars = [4]}], upper_bound = 374784 : i64}
      loom.polynomial_constraint(%13) {monomials = [{coeff = 1 : i64, vars = [0]}], upper_bound = 16 : i64}
      %19 = loom.expression(%14, %12) {coeffs = [1, 1], logic = "mul"} : index
      loom.polynomial_constraint(%13, %12, %14, %19) {monomials = [{coeff = 1024 : i64, vars = [0]}, {coeff = 1 : i64, vars = [3]}], upper_bound = 374784 : i64}
      %20 = loom.expression(%14, %12) {coeffs = [1, 1], logic = "mul"} : index
      %21 = loom.expression(%13, %14) {coeffs = [1, 1], logic = "mul"} : index
      %22 = loom.expression(%13, %12) {coeffs = [1, 1], logic = "mul"} : index
      %23 = loom.expression(%14, %22) {coeffs = [1, 1], logic = "mul"} : index
      loom.polynomial_constraint(%12, %13, %14, %20, %21, %22, %23) {monomials = [{coeff = 960 : i64, vars = [3]}, {coeff = 256 : i64, vars = [4]}, {coeff = -15 : i64, vars = [6]}], upper_bound = 0 : i64}
    }
    func.func @matmul_kernel__d0i1_d1i1__f01__hoist_block_1__d_a(%arg0: memref<*xf32> {tt.divisibility = 16 : i32}, %arg1: memref<*xf32> {tt.divisibility = 16 : i32}, %arg2: memref<*xf32> {tt.divisibility = 16 : i32}, %arg3: index, %arg4: index, %arg5: index, %arg6: index, %arg7: index, %arg8: index) {
      affine.parallel (%arg9) = (0) to (8) {
        affine.parallel (%arg10) = (0) to (8) {
          affine.for %arg11 = 0 to affine_map<()[s0, s1] -> (s0)>()[%arg3, %arg4] {
            affine.for %arg12 = 0 to affine_map<()[s0, s1] -> ((s1 ceildiv 8) ceildiv 8)>()[%arg3, %arg4] {
              %12 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 8 + d1 * 8 + d2)>(%arg9, %arg10, %arg12)
              %c1024 = arith.constant 1024 : index
              %c1024_0 = arith.constant 1024 : index
              %c1 = arith.constant 1 : index
              %13 = loom.get_symbolic_block_size @constraints::@BM : index
              %14 = loom.get_symbolic_block_size @constraints::@BN : index
              %15 = loom.get_symbolic_block_size @constraints::@BK : index
              %16 = arith.ceildivsi %c1024_0, %15 : index
              %cst = arith.constant 0.000000e+00 : f32
              %c1024_1 = arith.constant 1024 : index
              %c0 = arith.constant 0 : index
              %17 = tensor.empty(%13, %14) : tensor<?x?xf32>
              %18 = linalg.fill ins(%cst : f32) outs(%17 : tensor<?x?xf32>) -> tensor<?x?xf32>
              %19 = arith.muli %arg11, %13 : index
              %20 = arith.muli %12, %14 : index
              %21 = arith.muli %15, %c1024 : index
              %c0_2 = arith.constant 0 : index
              %22 = arith.muli %c0_2, %15 : index
              %23 = arith.muli %22, %c1024_1 : index
              %24 = arith.addi %23, %20 : index
              %25 = loom.reinterpret_cast %arg1 to offset : [%24], sizes : [%16, %15, %14], strides : [%21, %c1024, %c1], reuse : [seq = false, spat = false, temp = true] : memref<*xf32> to memref<?x?x?xf32, strided<[?, ?, ?], offset: ?>>
              %26 = loom.alloc(%16, %15, %14) on @L1 : memref<?x?x?xf32>
              loom.copy %25, %26, interconnect : [], broadcast : [1, 1] : memref<?x?x?xf32, strided<[?, ?, ?], offset: ?>> to memref<?x?x?xf32>
              %27 = scf.for %arg13 = %c0 to %16 step %c1 iter_args(%arg14 = %18) -> (tensor<?x?xf32>) {
                %31 = arith.muli %arg13, %15 : index
                %32 = arith.muli %19, %c1024_1 : index
                %33 = arith.addi %32, %31 : index
                %34 = loom.reinterpret_cast %arg0 to offset : [%33], sizes : [%13, %15], strides : [%c1024_0, %c1], reuse : [seq = false, spat = true, temp = true] : memref<*xf32> to memref<?x?xf32, strided<[?, ?], offset: ?>>
                %35 = loom.alloc(%13, %15) on @L1 : memref<?x?xf32>
                loom.copy %34, %35, interconnect : [@horizontal_links, @vertical_links], broadcast : [8, 8] : memref<?x?xf32, strided<[?, ?], offset: ?>> to memref<?x?xf32>
                %36 = bufferization.to_tensor %35 restrict writable : memref<?x?xf32> to tensor<?x?xf32>
                %37 = arith.muli %31, %c1024_1 : index
                %subview = memref.subview %26[%arg13, 0, 0] [1, %15, %14] [1, 1, 1] : memref<?x?x?xf32> to memref<?x?xf32, strided<[?, 1], offset: ?>>
                %38 = bufferization.to_tensor %subview restrict writable : memref<?x?xf32, strided<[?, 1], offset: ?>> to tensor<?x?xf32>
                %39 = linalg.matmul ins(%36, %38 : tensor<?x?xf32>, tensor<?x?xf32>) outs(%18 : tensor<?x?xf32>) -> tensor<?x?xf32>
                %40 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%arg14, %39 : tensor<?x?xf32>, tensor<?x?xf32>) outs(%arg14 : tensor<?x?xf32>) {
                ^bb0(%in: f32, %in_3: f32, %out: f32):
                  %41 = arith.addf %in, %in_3 : f32
                  linalg.yield %41 : f32
                } -> tensor<?x?xf32>
                scf.yield %40 : tensor<?x?xf32>
              }
              %28 = arith.muli %19, %c1024_1 : index
              %29 = arith.addi %28, %20 : index
              %30 = loom.reinterpret_cast %arg2 to offset : [%29], sizes : [%13, %14], strides : [%c1024, %c1], reuse : [seq = false, spat = false, temp = false] : memref<*xf32> to memref<?x?xf32, strided<[?, ?], offset: ?>>
              bufferization.materialize_in_destination %27 in writable %30 : (tensor<?x?xf32>, memref<?x?xf32, strided<[?, ?], offset: ?>>) -> ()
            }
          }
        } {loom.mapped_to = @y}
      } {loom.mapped_to = @x}
      return
    }
  }
  module attributes {loom.pass_name = "EnumerateCopyBroadcast"} {
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
      %15 = loom.expression(%13, %12) {coeffs = [1, 1], logic = "mul"} : index
      %16 = loom.expression(%15, %14) {coeffs = [1, 1], logic = "mul"} : index
      loom.polynomial_constraint(%12, %13, %14, %15, %16) {monomials = [{coeff = -1 : i64, vars = [4]}], upper_bound = -262144 : i64}
      %17 = loom.expression(%12, %13) {coeffs = [1, 1], logic = "add"} : index
      %18 = loom.expression(%14, %17) {coeffs = [1, 1], logic = "mul"} : index
      loom.polynomial_constraint(%12, %14, %13, %17, %18) {monomials = [{coeff = 1 : i64, vars = [4]}], upper_bound = 374784 : i64}
      loom.polynomial_constraint(%13) {monomials = [{coeff = 1 : i64, vars = [0]}], upper_bound = 16 : i64}
      %19 = loom.expression(%14, %12) {coeffs = [1, 1], logic = "mul"} : index
      loom.polynomial_constraint(%13, %12, %14, %19) {monomials = [{coeff = 1024 : i64, vars = [0]}, {coeff = 1 : i64, vars = [3]}], upper_bound = 374784 : i64}
      %20 = loom.expression(%14, %12) {coeffs = [1, 1], logic = "mul"} : index
      %21 = loom.expression(%13, %14) {coeffs = [1, 1], logic = "mul"} : index
      %22 = loom.expression(%13, %12) {coeffs = [1, 1], logic = "mul"} : index
      %23 = loom.expression(%14, %22) {coeffs = [1, 1], logic = "mul"} : index
      loom.polynomial_constraint(%12, %13, %14, %20, %21, %22, %23) {monomials = [{coeff = 960 : i64, vars = [3]}, {coeff = 256 : i64, vars = [4]}, {coeff = -15 : i64, vars = [6]}], upper_bound = 0 : i64}
    }
    func.func @matmul_kernel__d0i1_d1i1__f01__hoist_block_1__d_h(%arg0: memref<*xf32> {tt.divisibility = 16 : i32}, %arg1: memref<*xf32> {tt.divisibility = 16 : i32}, %arg2: memref<*xf32> {tt.divisibility = 16 : i32}, %arg3: index, %arg4: index, %arg5: index, %arg6: index, %arg7: index, %arg8: index) {
      affine.parallel (%arg9) = (0) to (8) {
        affine.parallel (%arg10) = (0) to (8) {
          affine.for %arg11 = 0 to affine_map<()[s0, s1] -> (s0)>()[%arg3, %arg4] {
            affine.for %arg12 = 0 to affine_map<()[s0, s1] -> ((s1 ceildiv 8) ceildiv 8)>()[%arg3, %arg4] {
              %12 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 8 + d1 * 8 + d2)>(%arg9, %arg10, %arg12)
              %c1024 = arith.constant 1024 : index
              %c1024_0 = arith.constant 1024 : index
              %c1 = arith.constant 1 : index
              %13 = loom.get_symbolic_block_size @constraints::@BM : index
              %14 = loom.get_symbolic_block_size @constraints::@BN : index
              %15 = loom.get_symbolic_block_size @constraints::@BK : index
              %16 = arith.ceildivsi %c1024_0, %15 : index
              %cst = arith.constant 0.000000e+00 : f32
              %c1024_1 = arith.constant 1024 : index
              %c0 = arith.constant 0 : index
              %17 = tensor.empty(%13, %14) : tensor<?x?xf32>
              %18 = linalg.fill ins(%cst : f32) outs(%17 : tensor<?x?xf32>) -> tensor<?x?xf32>
              %19 = arith.muli %arg11, %13 : index
              %20 = arith.muli %12, %14 : index
              %21 = arith.muli %15, %c1024 : index
              %c0_2 = arith.constant 0 : index
              %22 = arith.muli %c0_2, %15 : index
              %23 = arith.muli %22, %c1024_1 : index
              %24 = arith.addi %23, %20 : index
              %25 = loom.reinterpret_cast %arg1 to offset : [%24], sizes : [%16, %15, %14], strides : [%21, %c1024, %c1], reuse : [seq = false, spat = false, temp = true] : memref<*xf32> to memref<?x?x?xf32, strided<[?, ?, ?], offset: ?>>
              %26 = loom.alloc(%16, %15, %14) on @L1 : memref<?x?x?xf32>
              loom.copy %25, %26, interconnect : [], broadcast : [1, 1] : memref<?x?x?xf32, strided<[?, ?, ?], offset: ?>> to memref<?x?x?xf32>
              %27 = scf.for %arg13 = %c0 to %16 step %c1 iter_args(%arg14 = %18) -> (tensor<?x?xf32>) {
                %31 = arith.muli %arg13, %15 : index
                %32 = arith.muli %19, %c1024_1 : index
                %33 = arith.addi %32, %31 : index
                %34 = loom.reinterpret_cast %arg0 to offset : [%33], sizes : [%13, %15], strides : [%c1024_0, %c1], reuse : [seq = false, spat = true, temp = true] : memref<*xf32> to memref<?x?xf32, strided<[?, ?], offset: ?>>
                %35 = loom.alloc(%13, %15) on @L1 : memref<?x?xf32>
                loom.copy %34, %35, interconnect : [@horizontal_links], broadcast : [1, 8] : memref<?x?xf32, strided<[?, ?], offset: ?>> to memref<?x?xf32>
                %36 = bufferization.to_tensor %35 restrict writable : memref<?x?xf32> to tensor<?x?xf32>
                %37 = arith.muli %31, %c1024_1 : index
                %subview = memref.subview %26[%arg13, 0, 0] [1, %15, %14] [1, 1, 1] : memref<?x?x?xf32> to memref<?x?xf32, strided<[?, 1], offset: ?>>
                %38 = bufferization.to_tensor %subview restrict writable : memref<?x?xf32, strided<[?, 1], offset: ?>> to tensor<?x?xf32>
                %39 = linalg.matmul ins(%36, %38 : tensor<?x?xf32>, tensor<?x?xf32>) outs(%18 : tensor<?x?xf32>) -> tensor<?x?xf32>
                %40 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%arg14, %39 : tensor<?x?xf32>, tensor<?x?xf32>) outs(%arg14 : tensor<?x?xf32>) {
                ^bb0(%in: f32, %in_3: f32, %out: f32):
                  %41 = arith.addf %in, %in_3 : f32
                  linalg.yield %41 : f32
                } -> tensor<?x?xf32>
                scf.yield %40 : tensor<?x?xf32>
              }
              %28 = arith.muli %19, %c1024_1 : index
              %29 = arith.addi %28, %20 : index
              %30 = loom.reinterpret_cast %arg2 to offset : [%29], sizes : [%13, %14], strides : [%c1024, %c1], reuse : [seq = false, spat = false, temp = false] : memref<*xf32> to memref<?x?xf32, strided<[?, ?], offset: ?>>
              bufferization.materialize_in_destination %27 in writable %30 : (tensor<?x?xf32>, memref<?x?xf32, strided<[?, ?], offset: ?>>) -> ()
            }
          }
        } {loom.mapped_to = @y}
      } {loom.mapped_to = @x}
      return
    }
  }
  module attributes {loom.pass_name = "EnumerateCopyBroadcast"} {
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
      %15 = loom.expression(%12, %13) {coeffs = [1, 1], logic = "mul"} : index
      %16 = loom.expression(%15, %14) {coeffs = [1, 1], logic = "mul"} : index
      loom.polynomial_constraint(%12, %13, %14, %15, %16) {monomials = [{coeff = -1 : i64, vars = [4]}], upper_bound = -262144 : i64}
      %17 = loom.expression(%12, %13) {coeffs = [1, 1], logic = "add"} : index
      %18 = loom.expression(%14, %17) {coeffs = [1, 1], logic = "mul"} : index
      loom.polynomial_constraint(%12, %14, %13, %17, %18) {monomials = [{coeff = 1 : i64, vars = [4]}], upper_bound = 374784 : i64}
      loom.polynomial_constraint(%13) {monomials = [{coeff = 1 : i64, vars = [0]}], upper_bound = 16 : i64}
      %19 = loom.expression(%14, %12) {coeffs = [1, 1], logic = "mul"} : index
      loom.polynomial_constraint(%13, %12, %14, %19) {monomials = [{coeff = 1024 : i64, vars = [0]}, {coeff = 1 : i64, vars = [3]}], upper_bound = 374784 : i64}
      %20 = loom.expression(%14, %12) {coeffs = [1, 1], logic = "mul"} : index
      %21 = loom.expression(%14, %13) {coeffs = [1, 1], logic = "mul"} : index
      %22 = loom.expression(%12, %13) {coeffs = [1, 1], logic = "mul"} : index
      %23 = loom.expression(%14, %22) {coeffs = [1, 1], logic = "mul"} : index
      loom.polynomial_constraint(%12, %13, %14, %20, %21, %22, %23) {monomials = [{coeff = 960 : i64, vars = [3]}, {coeff = 256 : i64, vars = [4]}, {coeff = -15 : i64, vars = [6]}], upper_bound = 0 : i64}
    }
    func.func @matmul_kernel__d0i1_d1i1__f01__hoist_block_1__d_v(%arg0: memref<*xf32> {tt.divisibility = 16 : i32}, %arg1: memref<*xf32> {tt.divisibility = 16 : i32}, %arg2: memref<*xf32> {tt.divisibility = 16 : i32}, %arg3: index, %arg4: index, %arg5: index, %arg6: index, %arg7: index, %arg8: index) {
      affine.parallel (%arg9) = (0) to (8) {
        affine.parallel (%arg10) = (0) to (8) {
          affine.for %arg11 = 0 to affine_map<()[s0, s1] -> (s0)>()[%arg3, %arg4] {
            affine.for %arg12 = 0 to affine_map<()[s0, s1] -> ((s1 ceildiv 8) ceildiv 8)>()[%arg3, %arg4] {
              %12 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 8 + d1 * 8 + d2)>(%arg9, %arg10, %arg12)
              %c1024 = arith.constant 1024 : index
              %c1024_0 = arith.constant 1024 : index
              %c1 = arith.constant 1 : index
              %13 = loom.get_symbolic_block_size @constraints::@BM : index
              %14 = loom.get_symbolic_block_size @constraints::@BN : index
              %15 = loom.get_symbolic_block_size @constraints::@BK : index
              %16 = arith.ceildivsi %c1024_0, %15 : index
              %cst = arith.constant 0.000000e+00 : f32
              %c1024_1 = arith.constant 1024 : index
              %c0 = arith.constant 0 : index
              %17 = tensor.empty(%13, %14) : tensor<?x?xf32>
              %18 = linalg.fill ins(%cst : f32) outs(%17 : tensor<?x?xf32>) -> tensor<?x?xf32>
              %19 = arith.muli %arg11, %13 : index
              %20 = arith.muli %12, %14 : index
              %21 = arith.muli %15, %c1024 : index
              %c0_2 = arith.constant 0 : index
              %22 = arith.muli %c0_2, %15 : index
              %23 = arith.muli %22, %c1024_1 : index
              %24 = arith.addi %23, %20 : index
              %25 = loom.reinterpret_cast %arg1 to offset : [%24], sizes : [%16, %15, %14], strides : [%21, %c1024, %c1], reuse : [seq = false, spat = false, temp = true] : memref<*xf32> to memref<?x?x?xf32, strided<[?, ?, ?], offset: ?>>
              %26 = loom.alloc(%16, %15, %14) on @L1 : memref<?x?x?xf32>
              loom.copy %25, %26, interconnect : [], broadcast : [1, 1] : memref<?x?x?xf32, strided<[?, ?, ?], offset: ?>> to memref<?x?x?xf32>
              %27 = scf.for %arg13 = %c0 to %16 step %c1 iter_args(%arg14 = %18) -> (tensor<?x?xf32>) {
                %31 = arith.muli %arg13, %15 : index
                %32 = arith.muli %19, %c1024_1 : index
                %33 = arith.addi %32, %31 : index
                %34 = loom.reinterpret_cast %arg0 to offset : [%33], sizes : [%13, %15], strides : [%c1024_0, %c1], reuse : [seq = false, spat = true, temp = true] : memref<*xf32> to memref<?x?xf32, strided<[?, ?], offset: ?>>
                %35 = loom.alloc(%13, %15) on @L1 : memref<?x?xf32>
                loom.copy %34, %35, interconnect : [@vertical_links], broadcast : [8, 1] : memref<?x?xf32, strided<[?, ?], offset: ?>> to memref<?x?xf32>
                %36 = bufferization.to_tensor %35 restrict writable : memref<?x?xf32> to tensor<?x?xf32>
                %37 = arith.muli %31, %c1024_1 : index
                %subview = memref.subview %26[%arg13, 0, 0] [1, %15, %14] [1, 1, 1] : memref<?x?x?xf32> to memref<?x?xf32, strided<[?, 1], offset: ?>>
                %38 = bufferization.to_tensor %subview restrict writable : memref<?x?xf32, strided<[?, 1], offset: ?>> to tensor<?x?xf32>
                %39 = linalg.matmul ins(%36, %38 : tensor<?x?xf32>, tensor<?x?xf32>) outs(%18 : tensor<?x?xf32>) -> tensor<?x?xf32>
                %40 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%arg14, %39 : tensor<?x?xf32>, tensor<?x?xf32>) outs(%arg14 : tensor<?x?xf32>) {
                ^bb0(%in: f32, %in_3: f32, %out: f32):
                  %41 = arith.addf %in, %in_3 : f32
                  linalg.yield %41 : f32
                } -> tensor<?x?xf32>
                scf.yield %40 : tensor<?x?xf32>
              }
              %28 = arith.muli %19, %c1024_1 : index
              %29 = arith.addi %28, %20 : index
              %30 = loom.reinterpret_cast %arg2 to offset : [%29], sizes : [%13, %14], strides : [%c1024, %c1], reuse : [seq = false, spat = false, temp = false] : memref<*xf32> to memref<?x?xf32, strided<[?, ?], offset: ?>>
              bufferization.materialize_in_destination %27 in writable %30 : (tensor<?x?xf32>, memref<?x?xf32, strided<[?, ?], offset: ?>>) -> ()
            }
          }
        } {loom.mapped_to = @y}
      } {loom.mapped_to = @x}
      return
    }
  }
  module attributes {loom.pass_name = "EnumerateCopyBroadcast"} {
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
      %15 = loom.expression(%12, %13) {coeffs = [1, 1], logic = "mul"} : index
      %16 = loom.expression(%15, %14) {coeffs = [1, 1], logic = "mul"} : index
      loom.polynomial_constraint(%12, %13, %14, %15, %16) {monomials = [{coeff = -1 : i64, vars = [4]}], upper_bound = -262144 : i64}
      %17 = loom.expression(%12, %13) {coeffs = [1, 1], logic = "add"} : index
      %18 = loom.expression(%17, %14) {coeffs = [1, 1], logic = "mul"} : index
      loom.polynomial_constraint(%12, %14, %13, %17, %18) {monomials = [{coeff = 1 : i64, vars = [4]}], upper_bound = 374784 : i64}
      loom.polynomial_constraint(%13) {monomials = [{coeff = 1 : i64, vars = [0]}], upper_bound = 16 : i64}
      %19 = loom.expression(%12, %14) {coeffs = [1, 1], logic = "mul"} : index
      %20 = loom.expression(%13, %14) {coeffs = [1, 1], logic = "mul"} : index
      %21 = loom.expression(%12, %13) {coeffs = [1, 1], logic = "mul"} : index
      %22 = loom.expression(%14, %21) {coeffs = [1, 1], logic = "mul"} : index
      loom.polynomial_constraint(%12, %13, %14, %19, %20, %21, %22) {monomials = [{coeff = 64 : i64, vars = [3]}, {coeff = 64 : i64, vars = [4]}, {coeff = -1 : i64, vars = [6]}], upper_bound = 0 : i64}
    }
    func.func @matmul_kernel__d0i1_d1i1__f10__d_d(%arg0: memref<*xf32> {tt.divisibility = 16 : i32}, %arg1: memref<*xf32> {tt.divisibility = 16 : i32}, %arg2: memref<*xf32> {tt.divisibility = 16 : i32}, %arg3: index, %arg4: index, %arg5: index, %arg6: index, %arg7: index, %arg8: index) {
      affine.parallel (%arg9) = (0) to (8) {
        affine.parallel (%arg10) = (0) to (8) {
          affine.for %arg11 = 0 to affine_map<()[s0, s1] -> ((s1 ceildiv 8) ceildiv 8)>()[%arg3, %arg4] {
            affine.for %arg12 = 0 to affine_map<()[s0, s1] -> (s0)>()[%arg3, %arg4] {
              %12 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 8 + d1 * 8 + d2)>(%arg9, %arg10, %arg11)
              %c1024 = arith.constant 1024 : index
              %c1024_0 = arith.constant 1024 : index
              %c1 = arith.constant 1 : index
              %13 = loom.get_symbolic_block_size @constraints::@BM : index
              %14 = loom.get_symbolic_block_size @constraints::@BN : index
              %15 = loom.get_symbolic_block_size @constraints::@BK : index
              %16 = arith.ceildivsi %c1024_0, %15 : index
              %cst = arith.constant 0.000000e+00 : f32
              %c1024_1 = arith.constant 1024 : index
              %c0 = arith.constant 0 : index
              %17 = tensor.empty(%13, %14) : tensor<?x?xf32>
              %18 = linalg.fill ins(%cst : f32) outs(%17 : tensor<?x?xf32>) -> tensor<?x?xf32>
              %19 = arith.muli %arg12, %13 : index
              %20 = arith.muli %12, %14 : index
              %21 = scf.for %arg13 = %c0 to %16 step %c1 iter_args(%arg14 = %18) -> (tensor<?x?xf32>) {
                %25 = arith.muli %arg13, %15 : index
                %26 = arith.muli %19, %c1024_1 : index
                %27 = arith.addi %26, %25 : index
                %28 = loom.reinterpret_cast %arg0 to offset : [%27], sizes : [%13, %15], strides : [%c1024_0, %c1], reuse : [seq = false, spat = true, temp = true] : memref<*xf32> to memref<?x?xf32, strided<[?, ?], offset: ?>>
                %29 = loom.alloc(%13, %15) on @L1 : memref<?x?xf32>
                loom.copy %28, %29, interconnect : [], broadcast : [1, 1] : memref<?x?xf32, strided<[?, ?], offset: ?>> to memref<?x?xf32>
                %30 = bufferization.to_tensor %29 restrict writable : memref<?x?xf32> to tensor<?x?xf32>
                %31 = arith.muli %25, %c1024_1 : index
                %32 = arith.addi %31, %20 : index
                %33 = loom.reinterpret_cast %arg1 to offset : [%32], sizes : [%15, %14], strides : [%c1024, %c1], reuse : [seq = false, spat = false, temp = true] : memref<*xf32> to memref<?x?xf32, strided<[?, ?], offset: ?>>
                %34 = loom.alloc(%15, %14) on @L1 : memref<?x?xf32>
                loom.copy %33, %34, interconnect : [], broadcast : [1, 1] : memref<?x?xf32, strided<[?, ?], offset: ?>> to memref<?x?xf32>
                %35 = bufferization.to_tensor %34 restrict writable : memref<?x?xf32> to tensor<?x?xf32>
                %36 = linalg.matmul ins(%30, %35 : tensor<?x?xf32>, tensor<?x?xf32>) outs(%18 : tensor<?x?xf32>) -> tensor<?x?xf32>
                %37 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%arg14, %36 : tensor<?x?xf32>, tensor<?x?xf32>) outs(%arg14 : tensor<?x?xf32>) {
                ^bb0(%in: f32, %in_2: f32, %out: f32):
                  %38 = arith.addf %in, %in_2 : f32
                  linalg.yield %38 : f32
                } -> tensor<?x?xf32>
                scf.yield %37 : tensor<?x?xf32>
              }
              %22 = arith.muli %19, %c1024_1 : index
              %23 = arith.addi %22, %20 : index
              %24 = loom.reinterpret_cast %arg2 to offset : [%23], sizes : [%13, %14], strides : [%c1024, %c1], reuse : [seq = false, spat = false, temp = false] : memref<*xf32> to memref<?x?xf32, strided<[?, ?], offset: ?>>
              bufferization.materialize_in_destination %21 in writable %24 : (tensor<?x?xf32>, memref<?x?xf32, strided<[?, ?], offset: ?>>) -> ()
            }
          }
        } {loom.mapped_to = @y}
      } {loom.mapped_to = @x}
      return
    }
  }
  module attributes {loom.pass_name = "EnumerateCopyBroadcast"} {
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
      %15 = loom.expression(%12, %13) {coeffs = [1, 1], logic = "mul"} : index
      %16 = loom.expression(%15, %14) {coeffs = [1, 1], logic = "mul"} : index
      loom.polynomial_constraint(%12, %13, %14, %15, %16) {monomials = [{coeff = -1 : i64, vars = [4]}], upper_bound = -262144 : i64}
      %17 = loom.expression(%12, %13) {coeffs = [1, 1], logic = "add"} : index
      %18 = loom.expression(%17, %14) {coeffs = [1, 1], logic = "mul"} : index
      loom.polynomial_constraint(%12, %14, %13, %17, %18) {monomials = [{coeff = 1 : i64, vars = [4]}], upper_bound = 374784 : i64}
      loom.polynomial_constraint(%13) {monomials = [{coeff = 1 : i64, vars = [0]}], upper_bound = 16 : i64}
      %19 = loom.expression(%12, %14) {coeffs = [1, 1], logic = "mul"} : index
      %20 = loom.expression(%13, %14) {coeffs = [1, 1], logic = "mul"} : index
      %21 = loom.expression(%12, %13) {coeffs = [1, 1], logic = "mul"} : index
      %22 = loom.expression(%14, %21) {coeffs = [1, 1], logic = "mul"} : index
      loom.polynomial_constraint(%12, %13, %14, %19, %20, %21, %22) {monomials = [{coeff = 256 : i64, vars = [3]}, {coeff = 960 : i64, vars = [4]}, {coeff = -15 : i64, vars = [6]}], upper_bound = 0 : i64}
    }
    func.func @matmul_kernel__d0i1_d1i1__f10__a_d(%arg0: memref<*xf32> {tt.divisibility = 16 : i32}, %arg1: memref<*xf32> {tt.divisibility = 16 : i32}, %arg2: memref<*xf32> {tt.divisibility = 16 : i32}, %arg3: index, %arg4: index, %arg5: index, %arg6: index, %arg7: index, %arg8: index) {
      affine.parallel (%arg9) = (0) to (8) {
        affine.parallel (%arg10) = (0) to (8) {
          affine.for %arg11 = 0 to affine_map<()[s0, s1] -> ((s1 ceildiv 8) ceildiv 8)>()[%arg3, %arg4] {
            affine.for %arg12 = 0 to affine_map<()[s0, s1] -> (s0)>()[%arg3, %arg4] {
              %12 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 8 + d1 * 8 + d2)>(%arg9, %arg10, %arg11)
              %c1024 = arith.constant 1024 : index
              %c1024_0 = arith.constant 1024 : index
              %c1 = arith.constant 1 : index
              %13 = loom.get_symbolic_block_size @constraints::@BM : index
              %14 = loom.get_symbolic_block_size @constraints::@BN : index
              %15 = loom.get_symbolic_block_size @constraints::@BK : index
              %16 = arith.ceildivsi %c1024_0, %15 : index
              %cst = arith.constant 0.000000e+00 : f32
              %c1024_1 = arith.constant 1024 : index
              %c0 = arith.constant 0 : index
              %17 = tensor.empty(%13, %14) : tensor<?x?xf32>
              %18 = linalg.fill ins(%cst : f32) outs(%17 : tensor<?x?xf32>) -> tensor<?x?xf32>
              %19 = arith.muli %arg12, %13 : index
              %20 = arith.muli %12, %14 : index
              %21 = scf.for %arg13 = %c0 to %16 step %c1 iter_args(%arg14 = %18) -> (tensor<?x?xf32>) {
                %25 = arith.muli %arg13, %15 : index
                %26 = arith.muli %19, %c1024_1 : index
                %27 = arith.addi %26, %25 : index
                %28 = loom.reinterpret_cast %arg0 to offset : [%27], sizes : [%13, %15], strides : [%c1024_0, %c1], reuse : [seq = false, spat = true, temp = true] : memref<*xf32> to memref<?x?xf32, strided<[?, ?], offset: ?>>
                %29 = loom.alloc(%13, %15) on @L1 : memref<?x?xf32>
                loom.copy %28, %29, interconnect : [@horizontal_links, @vertical_links], broadcast : [8, 8] : memref<?x?xf32, strided<[?, ?], offset: ?>> to memref<?x?xf32>
                %30 = bufferization.to_tensor %29 restrict writable : memref<?x?xf32> to tensor<?x?xf32>
                %31 = arith.muli %25, %c1024_1 : index
                %32 = arith.addi %31, %20 : index
                %33 = loom.reinterpret_cast %arg1 to offset : [%32], sizes : [%15, %14], strides : [%c1024, %c1], reuse : [seq = false, spat = false, temp = true] : memref<*xf32> to memref<?x?xf32, strided<[?, ?], offset: ?>>
                %34 = loom.alloc(%15, %14) on @L1 : memref<?x?xf32>
                loom.copy %33, %34, interconnect : [], broadcast : [1, 1] : memref<?x?xf32, strided<[?, ?], offset: ?>> to memref<?x?xf32>
                %35 = bufferization.to_tensor %34 restrict writable : memref<?x?xf32> to tensor<?x?xf32>
                %36 = linalg.matmul ins(%30, %35 : tensor<?x?xf32>, tensor<?x?xf32>) outs(%18 : tensor<?x?xf32>) -> tensor<?x?xf32>
                %37 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%arg14, %36 : tensor<?x?xf32>, tensor<?x?xf32>) outs(%arg14 : tensor<?x?xf32>) {
                ^bb0(%in: f32, %in_2: f32, %out: f32):
                  %38 = arith.addf %in, %in_2 : f32
                  linalg.yield %38 : f32
                } -> tensor<?x?xf32>
                scf.yield %37 : tensor<?x?xf32>
              }
              %22 = arith.muli %19, %c1024_1 : index
              %23 = arith.addi %22, %20 : index
              %24 = loom.reinterpret_cast %arg2 to offset : [%23], sizes : [%13, %14], strides : [%c1024, %c1], reuse : [seq = false, spat = false, temp = false] : memref<*xf32> to memref<?x?xf32, strided<[?, ?], offset: ?>>
              bufferization.materialize_in_destination %21 in writable %24 : (tensor<?x?xf32>, memref<?x?xf32, strided<[?, ?], offset: ?>>) -> ()
            }
          }
        } {loom.mapped_to = @y}
      } {loom.mapped_to = @x}
      return
    }
  }
  module attributes {loom.pass_name = "EnumerateCopyBroadcast"} {
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
      %15 = loom.expression(%12, %13) {coeffs = [1, 1], logic = "mul"} : index
      %16 = loom.expression(%15, %14) {coeffs = [1, 1], logic = "mul"} : index
      loom.polynomial_constraint(%12, %13, %14, %15, %16) {monomials = [{coeff = -1 : i64, vars = [4]}], upper_bound = -262144 : i64}
      %17 = loom.expression(%12, %13) {coeffs = [1, 1], logic = "add"} : index
      %18 = loom.expression(%17, %14) {coeffs = [1, 1], logic = "mul"} : index
      loom.polynomial_constraint(%12, %14, %13, %17, %18) {monomials = [{coeff = 1 : i64, vars = [4]}], upper_bound = 374784 : i64}
      loom.polynomial_constraint(%13) {monomials = [{coeff = 1 : i64, vars = [0]}], upper_bound = 16 : i64}
      %19 = loom.expression(%12, %14) {coeffs = [1, 1], logic = "mul"} : index
      %20 = loom.expression(%13, %14) {coeffs = [1, 1], logic = "mul"} : index
      %21 = loom.expression(%12, %13) {coeffs = [1, 1], logic = "mul"} : index
      %22 = loom.expression(%14, %21) {coeffs = [1, 1], logic = "mul"} : index
      loom.polynomial_constraint(%12, %13, %14, %19, %20, %21, %22) {monomials = [{coeff = 256 : i64, vars = [3]}, {coeff = 960 : i64, vars = [4]}, {coeff = -15 : i64, vars = [6]}], upper_bound = 0 : i64}
    }
    func.func @matmul_kernel__d0i1_d1i1__f10__h_d(%arg0: memref<*xf32> {tt.divisibility = 16 : i32}, %arg1: memref<*xf32> {tt.divisibility = 16 : i32}, %arg2: memref<*xf32> {tt.divisibility = 16 : i32}, %arg3: index, %arg4: index, %arg5: index, %arg6: index, %arg7: index, %arg8: index) {
      affine.parallel (%arg9) = (0) to (8) {
        affine.parallel (%arg10) = (0) to (8) {
          affine.for %arg11 = 0 to affine_map<()[s0, s1] -> ((s1 ceildiv 8) ceildiv 8)>()[%arg3, %arg4] {
            affine.for %arg12 = 0 to affine_map<()[s0, s1] -> (s0)>()[%arg3, %arg4] {
              %12 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 8 + d1 * 8 + d2)>(%arg9, %arg10, %arg11)
              %c1024 = arith.constant 1024 : index
              %c1024_0 = arith.constant 1024 : index
              %c1 = arith.constant 1 : index
              %13 = loom.get_symbolic_block_size @constraints::@BM : index
              %14 = loom.get_symbolic_block_size @constraints::@BN : index
              %15 = loom.get_symbolic_block_size @constraints::@BK : index
              %16 = arith.ceildivsi %c1024_0, %15 : index
              %cst = arith.constant 0.000000e+00 : f32
              %c1024_1 = arith.constant 1024 : index
              %c0 = arith.constant 0 : index
              %17 = tensor.empty(%13, %14) : tensor<?x?xf32>
              %18 = linalg.fill ins(%cst : f32) outs(%17 : tensor<?x?xf32>) -> tensor<?x?xf32>
              %19 = arith.muli %arg12, %13 : index
              %20 = arith.muli %12, %14 : index
              %21 = scf.for %arg13 = %c0 to %16 step %c1 iter_args(%arg14 = %18) -> (tensor<?x?xf32>) {
                %25 = arith.muli %arg13, %15 : index
                %26 = arith.muli %19, %c1024_1 : index
                %27 = arith.addi %26, %25 : index
                %28 = loom.reinterpret_cast %arg0 to offset : [%27], sizes : [%13, %15], strides : [%c1024_0, %c1], reuse : [seq = false, spat = true, temp = true] : memref<*xf32> to memref<?x?xf32, strided<[?, ?], offset: ?>>
                %29 = loom.alloc(%13, %15) on @L1 : memref<?x?xf32>
                loom.copy %28, %29, interconnect : [@horizontal_links], broadcast : [1, 8] : memref<?x?xf32, strided<[?, ?], offset: ?>> to memref<?x?xf32>
                %30 = bufferization.to_tensor %29 restrict writable : memref<?x?xf32> to tensor<?x?xf32>
                %31 = arith.muli %25, %c1024_1 : index
                %32 = arith.addi %31, %20 : index
                %33 = loom.reinterpret_cast %arg1 to offset : [%32], sizes : [%15, %14], strides : [%c1024, %c1], reuse : [seq = false, spat = false, temp = true] : memref<*xf32> to memref<?x?xf32, strided<[?, ?], offset: ?>>
                %34 = loom.alloc(%15, %14) on @L1 : memref<?x?xf32>
                loom.copy %33, %34, interconnect : [], broadcast : [1, 1] : memref<?x?xf32, strided<[?, ?], offset: ?>> to memref<?x?xf32>
                %35 = bufferization.to_tensor %34 restrict writable : memref<?x?xf32> to tensor<?x?xf32>
                %36 = linalg.matmul ins(%30, %35 : tensor<?x?xf32>, tensor<?x?xf32>) outs(%18 : tensor<?x?xf32>) -> tensor<?x?xf32>
                %37 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%arg14, %36 : tensor<?x?xf32>, tensor<?x?xf32>) outs(%arg14 : tensor<?x?xf32>) {
                ^bb0(%in: f32, %in_2: f32, %out: f32):
                  %38 = arith.addf %in, %in_2 : f32
                  linalg.yield %38 : f32
                } -> tensor<?x?xf32>
                scf.yield %37 : tensor<?x?xf32>
              }
              %22 = arith.muli %19, %c1024_1 : index
              %23 = arith.addi %22, %20 : index
              %24 = loom.reinterpret_cast %arg2 to offset : [%23], sizes : [%13, %14], strides : [%c1024, %c1], reuse : [seq = false, spat = false, temp = false] : memref<*xf32> to memref<?x?xf32, strided<[?, ?], offset: ?>>
              bufferization.materialize_in_destination %21 in writable %24 : (tensor<?x?xf32>, memref<?x?xf32, strided<[?, ?], offset: ?>>) -> ()
            }
          }
        } {loom.mapped_to = @y}
      } {loom.mapped_to = @x}
      return
    }
  }
  module attributes {loom.pass_name = "EnumerateCopyBroadcast"} {
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
      %15 = loom.expression(%12, %13) {coeffs = [1, 1], logic = "mul"} : index
      %16 = loom.expression(%15, %14) {coeffs = [1, 1], logic = "mul"} : index
      loom.polynomial_constraint(%12, %13, %14, %15, %16) {monomials = [{coeff = -1 : i64, vars = [4]}], upper_bound = -262144 : i64}
      %17 = loom.expression(%12, %13) {coeffs = [1, 1], logic = "add"} : index
      %18 = loom.expression(%17, %14) {coeffs = [1, 1], logic = "mul"} : index
      loom.polynomial_constraint(%12, %14, %13, %17, %18) {monomials = [{coeff = 1 : i64, vars = [4]}], upper_bound = 374784 : i64}
      loom.polynomial_constraint(%13) {monomials = [{coeff = 1 : i64, vars = [0]}], upper_bound = 16 : i64}
      %19 = loom.expression(%12, %14) {coeffs = [1, 1], logic = "mul"} : index
      %20 = loom.expression(%13, %14) {coeffs = [1, 1], logic = "mul"} : index
      %21 = loom.expression(%12, %13) {coeffs = [1, 1], logic = "mul"} : index
      %22 = loom.expression(%14, %21) {coeffs = [1, 1], logic = "mul"} : index
      loom.polynomial_constraint(%12, %13, %14, %19, %20, %21, %22) {monomials = [{coeff = 256 : i64, vars = [3]}, {coeff = 960 : i64, vars = [4]}, {coeff = -15 : i64, vars = [6]}], upper_bound = 0 : i64}
    }
    func.func @matmul_kernel__d0i1_d1i1__f10__v_d(%arg0: memref<*xf32> {tt.divisibility = 16 : i32}, %arg1: memref<*xf32> {tt.divisibility = 16 : i32}, %arg2: memref<*xf32> {tt.divisibility = 16 : i32}, %arg3: index, %arg4: index, %arg5: index, %arg6: index, %arg7: index, %arg8: index) {
      affine.parallel (%arg9) = (0) to (8) {
        affine.parallel (%arg10) = (0) to (8) {
          affine.for %arg11 = 0 to affine_map<()[s0, s1] -> ((s1 ceildiv 8) ceildiv 8)>()[%arg3, %arg4] {
            affine.for %arg12 = 0 to affine_map<()[s0, s1] -> (s0)>()[%arg3, %arg4] {
              %12 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 8 + d1 * 8 + d2)>(%arg9, %arg10, %arg11)
              %c1024 = arith.constant 1024 : index
              %c1024_0 = arith.constant 1024 : index
              %c1 = arith.constant 1 : index
              %13 = loom.get_symbolic_block_size @constraints::@BM : index
              %14 = loom.get_symbolic_block_size @constraints::@BN : index
              %15 = loom.get_symbolic_block_size @constraints::@BK : index
              %16 = arith.ceildivsi %c1024_0, %15 : index
              %cst = arith.constant 0.000000e+00 : f32
              %c1024_1 = arith.constant 1024 : index
              %c0 = arith.constant 0 : index
              %17 = tensor.empty(%13, %14) : tensor<?x?xf32>
              %18 = linalg.fill ins(%cst : f32) outs(%17 : tensor<?x?xf32>) -> tensor<?x?xf32>
              %19 = arith.muli %arg12, %13 : index
              %20 = arith.muli %12, %14 : index
              %21 = scf.for %arg13 = %c0 to %16 step %c1 iter_args(%arg14 = %18) -> (tensor<?x?xf32>) {
                %25 = arith.muli %arg13, %15 : index
                %26 = arith.muli %19, %c1024_1 : index
                %27 = arith.addi %26, %25 : index
                %28 = loom.reinterpret_cast %arg0 to offset : [%27], sizes : [%13, %15], strides : [%c1024_0, %c1], reuse : [seq = false, spat = true, temp = true] : memref<*xf32> to memref<?x?xf32, strided<[?, ?], offset: ?>>
                %29 = loom.alloc(%13, %15) on @L1 : memref<?x?xf32>
                loom.copy %28, %29, interconnect : [@vertical_links], broadcast : [8, 1] : memref<?x?xf32, strided<[?, ?], offset: ?>> to memref<?x?xf32>
                %30 = bufferization.to_tensor %29 restrict writable : memref<?x?xf32> to tensor<?x?xf32>
                %31 = arith.muli %25, %c1024_1 : index
                %32 = arith.addi %31, %20 : index
                %33 = loom.reinterpret_cast %arg1 to offset : [%32], sizes : [%15, %14], strides : [%c1024, %c1], reuse : [seq = false, spat = false, temp = true] : memref<*xf32> to memref<?x?xf32, strided<[?, ?], offset: ?>>
                %34 = loom.alloc(%15, %14) on @L1 : memref<?x?xf32>
                loom.copy %33, %34, interconnect : [], broadcast : [1, 1] : memref<?x?xf32, strided<[?, ?], offset: ?>> to memref<?x?xf32>
                %35 = bufferization.to_tensor %34 restrict writable : memref<?x?xf32> to tensor<?x?xf32>
                %36 = linalg.matmul ins(%30, %35 : tensor<?x?xf32>, tensor<?x?xf32>) outs(%18 : tensor<?x?xf32>) -> tensor<?x?xf32>
                %37 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%arg14, %36 : tensor<?x?xf32>, tensor<?x?xf32>) outs(%arg14 : tensor<?x?xf32>) {
                ^bb0(%in: f32, %in_2: f32, %out: f32):
                  %38 = arith.addf %in, %in_2 : f32
                  linalg.yield %38 : f32
                } -> tensor<?x?xf32>
                scf.yield %37 : tensor<?x?xf32>
              }
              %22 = arith.muli %19, %c1024_1 : index
              %23 = arith.addi %22, %20 : index
              %24 = loom.reinterpret_cast %arg2 to offset : [%23], sizes : [%13, %14], strides : [%c1024, %c1], reuse : [seq = false, spat = false, temp = false] : memref<*xf32> to memref<?x?xf32, strided<[?, ?], offset: ?>>
              bufferization.materialize_in_destination %21 in writable %24 : (tensor<?x?xf32>, memref<?x?xf32, strided<[?, ?], offset: ?>>) -> ()
            }
          }
        } {loom.mapped_to = @y}
      } {loom.mapped_to = @x}
      return
    }
  }
  module attributes {loom.pass_name = "EnumerateCopyBroadcast"} {
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
      %15 = loom.expression(%12, %13) {coeffs = [1, 1], logic = "mul"} : index
      %16 = loom.expression(%15, %14) {coeffs = [1, 1], logic = "mul"} : index
      loom.polynomial_constraint(%12, %13, %14, %15, %16) {monomials = [{coeff = -1 : i64, vars = [4]}], upper_bound = -262144 : i64}
      %17 = loom.expression(%12, %13) {coeffs = [1, 1], logic = "add"} : index
      %18 = loom.expression(%17, %14) {coeffs = [1, 1], logic = "mul"} : index
      loom.polynomial_constraint(%12, %14, %13, %17, %18) {monomials = [{coeff = 1 : i64, vars = [4]}], upper_bound = 374784 : i64}
      loom.polynomial_constraint(%13) {monomials = [{coeff = 1 : i64, vars = [0]}], upper_bound = 16 : i64}
      %19 = loom.expression(%13, %14) {coeffs = [1, 1], logic = "mul"} : index
      loom.polynomial_constraint(%12, %14, %13, %19) {monomials = [{coeff = 1024 : i64, vars = [0]}, {coeff = 1 : i64, vars = [3]}], upper_bound = 374784 : i64}
      %20 = loom.expression(%12, %14) {coeffs = [1, 1], logic = "mul"} : index
      %21 = loom.expression(%13, %14) {coeffs = [1, 1], logic = "mul"} : index
      %22 = loom.expression(%12, %13) {coeffs = [1, 1], logic = "mul"} : index
      %23 = loom.expression(%14, %22) {coeffs = [1, 1], logic = "mul"} : index
      loom.polynomial_constraint(%12, %13, %14, %20, %21, %22, %23) {monomials = [{coeff = 64 : i64, vars = [3]}, {coeff = 64 : i64, vars = [4]}, {coeff = -1 : i64, vars = [6]}], upper_bound = 0 : i64}
    }
    func.func @matmul_kernel__d0i1_d1i1__f10__hoist_block_0__d_d(%arg0: memref<*xf32> {tt.divisibility = 16 : i32}, %arg1: memref<*xf32> {tt.divisibility = 16 : i32}, %arg2: memref<*xf32> {tt.divisibility = 16 : i32}, %arg3: index, %arg4: index, %arg5: index, %arg6: index, %arg7: index, %arg8: index) {
      affine.parallel (%arg9) = (0) to (8) {
        affine.parallel (%arg10) = (0) to (8) {
          affine.for %arg11 = 0 to affine_map<()[s0, s1] -> ((s1 ceildiv 8) ceildiv 8)>()[%arg3, %arg4] {
            affine.for %arg12 = 0 to affine_map<()[s0, s1] -> (s0)>()[%arg3, %arg4] {
              %12 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 8 + d1 * 8 + d2)>(%arg9, %arg10, %arg11)
              %c1024 = arith.constant 1024 : index
              %c1024_0 = arith.constant 1024 : index
              %c1 = arith.constant 1 : index
              %13 = loom.get_symbolic_block_size @constraints::@BM : index
              %14 = loom.get_symbolic_block_size @constraints::@BN : index
              %15 = loom.get_symbolic_block_size @constraints::@BK : index
              %16 = arith.ceildivsi %c1024_0, %15 : index
              %cst = arith.constant 0.000000e+00 : f32
              %c1024_1 = arith.constant 1024 : index
              %c0 = arith.constant 0 : index
              %17 = tensor.empty(%13, %14) : tensor<?x?xf32>
              %18 = linalg.fill ins(%cst : f32) outs(%17 : tensor<?x?xf32>) -> tensor<?x?xf32>
              %19 = arith.muli %arg12, %13 : index
              %20 = arith.muli %12, %14 : index
              %21 = arith.muli %13, %c1024_0 : index
              %c0_2 = arith.constant 0 : index
              %22 = arith.muli %19, %c1024_1 : index
              %23 = arith.muli %c0_2, %15 : index
              %24 = arith.addi %22, %23 : index
              %25 = loom.reinterpret_cast %arg0 to offset : [%24], sizes : [%16, %13, %15], strides : [%21, %c1024_0, %c1], reuse : [seq = false, spat = true, temp = true] : memref<*xf32> to memref<?x?x?xf32, strided<[?, ?, ?], offset: ?>>
              %26 = loom.alloc(%16, %13, %15) on @L1 : memref<?x?x?xf32>
              loom.copy %25, %26, interconnect : [], broadcast : [1, 1] : memref<?x?x?xf32, strided<[?, ?, ?], offset: ?>> to memref<?x?x?xf32>
              %27 = scf.for %arg13 = %c0 to %16 step %c1 iter_args(%arg14 = %18) -> (tensor<?x?xf32>) {
                %31 = arith.muli %arg13, %15 : index
                %32 = arith.muli %19, %c1024_1 : index
                %subview = memref.subview %26[%arg13, 0, 0] [1, %13, %15] [1, 1, 1] : memref<?x?x?xf32> to memref<?x?xf32, strided<[?, 1], offset: ?>>
                %33 = bufferization.to_tensor %subview restrict writable : memref<?x?xf32, strided<[?, 1], offset: ?>> to tensor<?x?xf32>
                %34 = arith.muli %31, %c1024_1 : index
                %35 = arith.addi %34, %20 : index
                %36 = loom.reinterpret_cast %arg1 to offset : [%35], sizes : [%15, %14], strides : [%c1024, %c1], reuse : [seq = false, spat = false, temp = true] : memref<*xf32> to memref<?x?xf32, strided<[?, ?], offset: ?>>
                %37 = loom.alloc(%15, %14) on @L1 : memref<?x?xf32>
                loom.copy %36, %37, interconnect : [], broadcast : [1, 1] : memref<?x?xf32, strided<[?, ?], offset: ?>> to memref<?x?xf32>
                %38 = bufferization.to_tensor %37 restrict writable : memref<?x?xf32> to tensor<?x?xf32>
                %39 = linalg.matmul ins(%33, %38 : tensor<?x?xf32>, tensor<?x?xf32>) outs(%18 : tensor<?x?xf32>) -> tensor<?x?xf32>
                %40 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%arg14, %39 : tensor<?x?xf32>, tensor<?x?xf32>) outs(%arg14 : tensor<?x?xf32>) {
                ^bb0(%in: f32, %in_3: f32, %out: f32):
                  %41 = arith.addf %in, %in_3 : f32
                  linalg.yield %41 : f32
                } -> tensor<?x?xf32>
                scf.yield %40 : tensor<?x?xf32>
              }
              %28 = arith.muli %19, %c1024_1 : index
              %29 = arith.addi %28, %20 : index
              %30 = loom.reinterpret_cast %arg2 to offset : [%29], sizes : [%13, %14], strides : [%c1024, %c1], reuse : [seq = false, spat = false, temp = false] : memref<*xf32> to memref<?x?xf32, strided<[?, ?], offset: ?>>
              bufferization.materialize_in_destination %27 in writable %30 : (tensor<?x?xf32>, memref<?x?xf32, strided<[?, ?], offset: ?>>) -> ()
            }
          }
        } {loom.mapped_to = @y}
      } {loom.mapped_to = @x}
      return
    }
  }
  module attributes {loom.pass_name = "EnumerateCopyBroadcast"} {
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
      %15 = loom.expression(%12, %13) {coeffs = [1, 1], logic = "mul"} : index
      %16 = loom.expression(%15, %14) {coeffs = [1, 1], logic = "mul"} : index
      loom.polynomial_constraint(%12, %13, %14, %15, %16) {monomials = [{coeff = -1 : i64, vars = [4]}], upper_bound = -262144 : i64}
      %17 = loom.expression(%12, %13) {coeffs = [1, 1], logic = "add"} : index
      %18 = loom.expression(%14, %17) {coeffs = [1, 1], logic = "mul"} : index
      loom.polynomial_constraint(%12, %14, %13, %17, %18) {monomials = [{coeff = 1 : i64, vars = [4]}], upper_bound = 374784 : i64}
      loom.polynomial_constraint(%13) {monomials = [{coeff = 1 : i64, vars = [0]}], upper_bound = 16 : i64}
      %19 = loom.expression(%14, %13) {coeffs = [1, 1], logic = "mul"} : index
      loom.polynomial_constraint(%12, %14, %13, %19) {monomials = [{coeff = 1024 : i64, vars = [0]}, {coeff = 1 : i64, vars = [3]}], upper_bound = 374784 : i64}
      %20 = loom.expression(%14, %12) {coeffs = [1, 1], logic = "mul"} : index
      %21 = loom.expression(%14, %13) {coeffs = [1, 1], logic = "mul"} : index
      %22 = loom.expression(%12, %13) {coeffs = [1, 1], logic = "mul"} : index
      %23 = loom.expression(%14, %22) {coeffs = [1, 1], logic = "mul"} : index
      loom.polynomial_constraint(%12, %13, %14, %20, %21, %22, %23) {monomials = [{coeff = 256 : i64, vars = [3]}, {coeff = 960 : i64, vars = [4]}, {coeff = -15 : i64, vars = [6]}], upper_bound = 0 : i64}
    }
    func.func @matmul_kernel__d0i1_d1i1__f10__hoist_block_0__a_d(%arg0: memref<*xf32> {tt.divisibility = 16 : i32}, %arg1: memref<*xf32> {tt.divisibility = 16 : i32}, %arg2: memref<*xf32> {tt.divisibility = 16 : i32}, %arg3: index, %arg4: index, %arg5: index, %arg6: index, %arg7: index, %arg8: index) {
      affine.parallel (%arg9) = (0) to (8) {
        affine.parallel (%arg10) = (0) to (8) {
          affine.for %arg11 = 0 to affine_map<()[s0, s1] -> ((s1 ceildiv 8) ceildiv 8)>()[%arg3, %arg4] {
            affine.for %arg12 = 0 to affine_map<()[s0, s1] -> (s0)>()[%arg3, %arg4] {
              %12 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 8 + d1 * 8 + d2)>(%arg9, %arg10, %arg11)
              %c1024 = arith.constant 1024 : index
              %c1024_0 = arith.constant 1024 : index
              %c1 = arith.constant 1 : index
              %13 = loom.get_symbolic_block_size @constraints::@BM : index
              %14 = loom.get_symbolic_block_size @constraints::@BN : index
              %15 = loom.get_symbolic_block_size @constraints::@BK : index
              %16 = arith.ceildivsi %c1024_0, %15 : index
              %cst = arith.constant 0.000000e+00 : f32
              %c1024_1 = arith.constant 1024 : index
              %c0 = arith.constant 0 : index
              %17 = tensor.empty(%13, %14) : tensor<?x?xf32>
              %18 = linalg.fill ins(%cst : f32) outs(%17 : tensor<?x?xf32>) -> tensor<?x?xf32>
              %19 = arith.muli %arg12, %13 : index
              %20 = arith.muli %12, %14 : index
              %21 = arith.muli %13, %c1024_0 : index
              %c0_2 = arith.constant 0 : index
              %22 = arith.muli %19, %c1024_1 : index
              %23 = arith.muli %c0_2, %15 : index
              %24 = arith.addi %22, %23 : index
              %25 = loom.reinterpret_cast %arg0 to offset : [%24], sizes : [%16, %13, %15], strides : [%21, %c1024_0, %c1], reuse : [seq = false, spat = true, temp = true] : memref<*xf32> to memref<?x?x?xf32, strided<[?, ?, ?], offset: ?>>
              %26 = loom.alloc(%16, %13, %15) on @L1 : memref<?x?x?xf32>
              loom.copy %25, %26, interconnect : [@horizontal_links, @vertical_links], broadcast : [8, 8] : memref<?x?x?xf32, strided<[?, ?, ?], offset: ?>> to memref<?x?x?xf32>
              %27 = scf.for %arg13 = %c0 to %16 step %c1 iter_args(%arg14 = %18) -> (tensor<?x?xf32>) {
                %31 = arith.muli %arg13, %15 : index
                %32 = arith.muli %19, %c1024_1 : index
                %subview = memref.subview %26[%arg13, 0, 0] [1, %13, %15] [1, 1, 1] : memref<?x?x?xf32> to memref<?x?xf32, strided<[?, 1], offset: ?>>
                %33 = bufferization.to_tensor %subview restrict writable : memref<?x?xf32, strided<[?, 1], offset: ?>> to tensor<?x?xf32>
                %34 = arith.muli %31, %c1024_1 : index
                %35 = arith.addi %34, %20 : index
                %36 = loom.reinterpret_cast %arg1 to offset : [%35], sizes : [%15, %14], strides : [%c1024, %c1], reuse : [seq = false, spat = false, temp = true] : memref<*xf32> to memref<?x?xf32, strided<[?, ?], offset: ?>>
                %37 = loom.alloc(%15, %14) on @L1 : memref<?x?xf32>
                loom.copy %36, %37, interconnect : [], broadcast : [1, 1] : memref<?x?xf32, strided<[?, ?], offset: ?>> to memref<?x?xf32>
                %38 = bufferization.to_tensor %37 restrict writable : memref<?x?xf32> to tensor<?x?xf32>
                %39 = linalg.matmul ins(%33, %38 : tensor<?x?xf32>, tensor<?x?xf32>) outs(%18 : tensor<?x?xf32>) -> tensor<?x?xf32>
                %40 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%arg14, %39 : tensor<?x?xf32>, tensor<?x?xf32>) outs(%arg14 : tensor<?x?xf32>) {
                ^bb0(%in: f32, %in_3: f32, %out: f32):
                  %41 = arith.addf %in, %in_3 : f32
                  linalg.yield %41 : f32
                } -> tensor<?x?xf32>
                scf.yield %40 : tensor<?x?xf32>
              }
              %28 = arith.muli %19, %c1024_1 : index
              %29 = arith.addi %28, %20 : index
              %30 = loom.reinterpret_cast %arg2 to offset : [%29], sizes : [%13, %14], strides : [%c1024, %c1], reuse : [seq = false, spat = false, temp = false] : memref<*xf32> to memref<?x?xf32, strided<[?, ?], offset: ?>>
              bufferization.materialize_in_destination %27 in writable %30 : (tensor<?x?xf32>, memref<?x?xf32, strided<[?, ?], offset: ?>>) -> ()
            }
          }
        } {loom.mapped_to = @y}
      } {loom.mapped_to = @x}
      return
    }
  }
  module attributes {loom.pass_name = "EnumerateCopyBroadcast"} {
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
      %15 = loom.expression(%12, %13) {coeffs = [1, 1], logic = "mul"} : index
      %16 = loom.expression(%15, %14) {coeffs = [1, 1], logic = "mul"} : index
      loom.polynomial_constraint(%12, %13, %14, %15, %16) {monomials = [{coeff = -1 : i64, vars = [4]}], upper_bound = -262144 : i64}
      %17 = loom.expression(%12, %13) {coeffs = [1, 1], logic = "add"} : index
      %18 = loom.expression(%17, %14) {coeffs = [1, 1], logic = "mul"} : index
      loom.polynomial_constraint(%12, %14, %13, %17, %18) {monomials = [{coeff = 1 : i64, vars = [4]}], upper_bound = 374784 : i64}
      loom.polynomial_constraint(%13) {monomials = [{coeff = 1 : i64, vars = [0]}], upper_bound = 16 : i64}
      %19 = loom.expression(%13, %14) {coeffs = [1, 1], logic = "mul"} : index
      loom.polynomial_constraint(%12, %14, %13, %19) {monomials = [{coeff = 1024 : i64, vars = [0]}, {coeff = 1 : i64, vars = [3]}], upper_bound = 374784 : i64}
      %20 = loom.expression(%12, %14) {coeffs = [1, 1], logic = "mul"} : index
      %21 = loom.expression(%13, %14) {coeffs = [1, 1], logic = "mul"} : index
      %22 = loom.expression(%12, %13) {coeffs = [1, 1], logic = "mul"} : index
      %23 = loom.expression(%14, %22) {coeffs = [1, 1], logic = "mul"} : index
      loom.polynomial_constraint(%12, %13, %14, %20, %21, %22, %23) {monomials = [{coeff = 256 : i64, vars = [3]}, {coeff = 960 : i64, vars = [4]}, {coeff = -15 : i64, vars = [6]}], upper_bound = 0 : i64}
    }
    func.func @matmul_kernel__d0i1_d1i1__f10__hoist_block_0__h_d(%arg0: memref<*xf32> {tt.divisibility = 16 : i32}, %arg1: memref<*xf32> {tt.divisibility = 16 : i32}, %arg2: memref<*xf32> {tt.divisibility = 16 : i32}, %arg3: index, %arg4: index, %arg5: index, %arg6: index, %arg7: index, %arg8: index) {
      affine.parallel (%arg9) = (0) to (8) {
        affine.parallel (%arg10) = (0) to (8) {
          affine.for %arg11 = 0 to affine_map<()[s0, s1] -> ((s1 ceildiv 8) ceildiv 8)>()[%arg3, %arg4] {
            affine.for %arg12 = 0 to affine_map<()[s0, s1] -> (s0)>()[%arg3, %arg4] {
              %12 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 8 + d1 * 8 + d2)>(%arg9, %arg10, %arg11)
              %c1024 = arith.constant 1024 : index
              %c1024_0 = arith.constant 1024 : index
              %c1 = arith.constant 1 : index
              %13 = loom.get_symbolic_block_size @constraints::@BM : index
              %14 = loom.get_symbolic_block_size @constraints::@BN : index
              %15 = loom.get_symbolic_block_size @constraints::@BK : index
              %16 = arith.ceildivsi %c1024_0, %15 : index
              %cst = arith.constant 0.000000e+00 : f32
              %c1024_1 = arith.constant 1024 : index
              %c0 = arith.constant 0 : index
              %17 = tensor.empty(%13, %14) : tensor<?x?xf32>
              %18 = linalg.fill ins(%cst : f32) outs(%17 : tensor<?x?xf32>) -> tensor<?x?xf32>
              %19 = arith.muli %arg12, %13 : index
              %20 = arith.muli %12, %14 : index
              %21 = arith.muli %13, %c1024_0 : index
              %c0_2 = arith.constant 0 : index
              %22 = arith.muli %19, %c1024_1 : index
              %23 = arith.muli %c0_2, %15 : index
              %24 = arith.addi %22, %23 : index
              %25 = loom.reinterpret_cast %arg0 to offset : [%24], sizes : [%16, %13, %15], strides : [%21, %c1024_0, %c1], reuse : [seq = false, spat = true, temp = true] : memref<*xf32> to memref<?x?x?xf32, strided<[?, ?, ?], offset: ?>>
              %26 = loom.alloc(%16, %13, %15) on @L1 : memref<?x?x?xf32>
              loom.copy %25, %26, interconnect : [@horizontal_links], broadcast : [1, 8] : memref<?x?x?xf32, strided<[?, ?, ?], offset: ?>> to memref<?x?x?xf32>
              %27 = scf.for %arg13 = %c0 to %16 step %c1 iter_args(%arg14 = %18) -> (tensor<?x?xf32>) {
                %31 = arith.muli %arg13, %15 : index
                %32 = arith.muli %19, %c1024_1 : index
                %subview = memref.subview %26[%arg13, 0, 0] [1, %13, %15] [1, 1, 1] : memref<?x?x?xf32> to memref<?x?xf32, strided<[?, 1], offset: ?>>
                %33 = bufferization.to_tensor %subview restrict writable : memref<?x?xf32, strided<[?, 1], offset: ?>> to tensor<?x?xf32>
                %34 = arith.muli %31, %c1024_1 : index
                %35 = arith.addi %34, %20 : index
                %36 = loom.reinterpret_cast %arg1 to offset : [%35], sizes : [%15, %14], strides : [%c1024, %c1], reuse : [seq = false, spat = false, temp = true] : memref<*xf32> to memref<?x?xf32, strided<[?, ?], offset: ?>>
                %37 = loom.alloc(%15, %14) on @L1 : memref<?x?xf32>
                loom.copy %36, %37, interconnect : [], broadcast : [1, 1] : memref<?x?xf32, strided<[?, ?], offset: ?>> to memref<?x?xf32>
                %38 = bufferization.to_tensor %37 restrict writable : memref<?x?xf32> to tensor<?x?xf32>
                %39 = linalg.matmul ins(%33, %38 : tensor<?x?xf32>, tensor<?x?xf32>) outs(%18 : tensor<?x?xf32>) -> tensor<?x?xf32>
                %40 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%arg14, %39 : tensor<?x?xf32>, tensor<?x?xf32>) outs(%arg14 : tensor<?x?xf32>) {
                ^bb0(%in: f32, %in_3: f32, %out: f32):
                  %41 = arith.addf %in, %in_3 : f32
                  linalg.yield %41 : f32
                } -> tensor<?x?xf32>
                scf.yield %40 : tensor<?x?xf32>
              }
              %28 = arith.muli %19, %c1024_1 : index
              %29 = arith.addi %28, %20 : index
              %30 = loom.reinterpret_cast %arg2 to offset : [%29], sizes : [%13, %14], strides : [%c1024, %c1], reuse : [seq = false, spat = false, temp = false] : memref<*xf32> to memref<?x?xf32, strided<[?, ?], offset: ?>>
              bufferization.materialize_in_destination %27 in writable %30 : (tensor<?x?xf32>, memref<?x?xf32, strided<[?, ?], offset: ?>>) -> ()
            }
          }
        } {loom.mapped_to = @y}
      } {loom.mapped_to = @x}
      return
    }
  }
  module attributes {loom.pass_name = "EnumerateCopyBroadcast"} {
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
      %15 = loom.expression(%13, %12) {coeffs = [1, 1], logic = "mul"} : index
      %16 = loom.expression(%15, %14) {coeffs = [1, 1], logic = "mul"} : index
      loom.polynomial_constraint(%12, %13, %14, %15, %16) {monomials = [{coeff = -1 : i64, vars = [4]}], upper_bound = -262144 : i64}
      %17 = loom.expression(%12, %13) {coeffs = [1, 1], logic = "add"} : index
      %18 = loom.expression(%14, %17) {coeffs = [1, 1], logic = "mul"} : index
      loom.polynomial_constraint(%12, %14, %13, %17, %18) {monomials = [{coeff = 1 : i64, vars = [4]}], upper_bound = 374784 : i64}
      loom.polynomial_constraint(%13) {monomials = [{coeff = 1 : i64, vars = [0]}], upper_bound = 16 : i64}
      %19 = loom.expression(%13, %14) {coeffs = [1, 1], logic = "mul"} : index
      loom.polynomial_constraint(%12, %14, %13, %19) {monomials = [{coeff = 1024 : i64, vars = [0]}, {coeff = 1 : i64, vars = [3]}], upper_bound = 374784 : i64}
      %20 = loom.expression(%14, %12) {coeffs = [1, 1], logic = "mul"} : index
      %21 = loom.expression(%13, %14) {coeffs = [1, 1], logic = "mul"} : index
      %22 = loom.expression(%13, %12) {coeffs = [1, 1], logic = "mul"} : index
      %23 = loom.expression(%14, %22) {coeffs = [1, 1], logic = "mul"} : index
      loom.polynomial_constraint(%12, %13, %14, %20, %21, %22, %23) {monomials = [{coeff = 256 : i64, vars = [3]}, {coeff = 960 : i64, vars = [4]}, {coeff = -15 : i64, vars = [6]}], upper_bound = 0 : i64}
    }
    func.func @matmul_kernel__d0i1_d1i1__f10__hoist_block_0__v_d(%arg0: memref<*xf32> {tt.divisibility = 16 : i32}, %arg1: memref<*xf32> {tt.divisibility = 16 : i32}, %arg2: memref<*xf32> {tt.divisibility = 16 : i32}, %arg3: index, %arg4: index, %arg5: index, %arg6: index, %arg7: index, %arg8: index) {
      affine.parallel (%arg9) = (0) to (8) {
        affine.parallel (%arg10) = (0) to (8) {
          affine.for %arg11 = 0 to affine_map<()[s0, s1] -> ((s1 ceildiv 8) ceildiv 8)>()[%arg3, %arg4] {
            affine.for %arg12 = 0 to affine_map<()[s0, s1] -> (s0)>()[%arg3, %arg4] {
              %12 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 8 + d1 * 8 + d2)>(%arg9, %arg10, %arg11)
              %c1024 = arith.constant 1024 : index
              %c1024_0 = arith.constant 1024 : index
              %c1 = arith.constant 1 : index
              %13 = loom.get_symbolic_block_size @constraints::@BM : index
              %14 = loom.get_symbolic_block_size @constraints::@BN : index
              %15 = loom.get_symbolic_block_size @constraints::@BK : index
              %16 = arith.ceildivsi %c1024_0, %15 : index
              %cst = arith.constant 0.000000e+00 : f32
              %c1024_1 = arith.constant 1024 : index
              %c0 = arith.constant 0 : index
              %17 = tensor.empty(%13, %14) : tensor<?x?xf32>
              %18 = linalg.fill ins(%cst : f32) outs(%17 : tensor<?x?xf32>) -> tensor<?x?xf32>
              %19 = arith.muli %arg12, %13 : index
              %20 = arith.muli %12, %14 : index
              %21 = arith.muli %13, %c1024_0 : index
              %c0_2 = arith.constant 0 : index
              %22 = arith.muli %19, %c1024_1 : index
              %23 = arith.muli %c0_2, %15 : index
              %24 = arith.addi %22, %23 : index
              %25 = loom.reinterpret_cast %arg0 to offset : [%24], sizes : [%16, %13, %15], strides : [%21, %c1024_0, %c1], reuse : [seq = false, spat = true, temp = true] : memref<*xf32> to memref<?x?x?xf32, strided<[?, ?, ?], offset: ?>>
              %26 = loom.alloc(%16, %13, %15) on @L1 : memref<?x?x?xf32>
              loom.copy %25, %26, interconnect : [@vertical_links], broadcast : [8, 1] : memref<?x?x?xf32, strided<[?, ?, ?], offset: ?>> to memref<?x?x?xf32>
              %27 = scf.for %arg13 = %c0 to %16 step %c1 iter_args(%arg14 = %18) -> (tensor<?x?xf32>) {
                %31 = arith.muli %arg13, %15 : index
                %32 = arith.muli %19, %c1024_1 : index
                %subview = memref.subview %26[%arg13, 0, 0] [1, %13, %15] [1, 1, 1] : memref<?x?x?xf32> to memref<?x?xf32, strided<[?, 1], offset: ?>>
                %33 = bufferization.to_tensor %subview restrict writable : memref<?x?xf32, strided<[?, 1], offset: ?>> to tensor<?x?xf32>
                %34 = arith.muli %31, %c1024_1 : index
                %35 = arith.addi %34, %20 : index
                %36 = loom.reinterpret_cast %arg1 to offset : [%35], sizes : [%15, %14], strides : [%c1024, %c1], reuse : [seq = false, spat = false, temp = true] : memref<*xf32> to memref<?x?xf32, strided<[?, ?], offset: ?>>
                %37 = loom.alloc(%15, %14) on @L1 : memref<?x?xf32>
                loom.copy %36, %37, interconnect : [], broadcast : [1, 1] : memref<?x?xf32, strided<[?, ?], offset: ?>> to memref<?x?xf32>
                %38 = bufferization.to_tensor %37 restrict writable : memref<?x?xf32> to tensor<?x?xf32>
                %39 = linalg.matmul ins(%33, %38 : tensor<?x?xf32>, tensor<?x?xf32>) outs(%18 : tensor<?x?xf32>) -> tensor<?x?xf32>
                %40 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%arg14, %39 : tensor<?x?xf32>, tensor<?x?xf32>) outs(%arg14 : tensor<?x?xf32>) {
                ^bb0(%in: f32, %in_3: f32, %out: f32):
                  %41 = arith.addf %in, %in_3 : f32
                  linalg.yield %41 : f32
                } -> tensor<?x?xf32>
                scf.yield %40 : tensor<?x?xf32>
              }
              %28 = arith.muli %19, %c1024_1 : index
              %29 = arith.addi %28, %20 : index
              %30 = loom.reinterpret_cast %arg2 to offset : [%29], sizes : [%13, %14], strides : [%c1024, %c1], reuse : [seq = false, spat = false, temp = false] : memref<*xf32> to memref<?x?xf32, strided<[?, ?], offset: ?>>
              bufferization.materialize_in_destination %27 in writable %30 : (tensor<?x?xf32>, memref<?x?xf32, strided<[?, ?], offset: ?>>) -> ()
            }
          }
        } {loom.mapped_to = @y}
      } {loom.mapped_to = @x}
      return
    }
  }
  module attributes {loom.pass_name = "EnumerateCopyBroadcast"} {
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
      %15 = loom.expression(%13, %12) {coeffs = [1, 1], logic = "mul"} : index
      %16 = loom.expression(%15, %14) {coeffs = [1, 1], logic = "mul"} : index
      loom.polynomial_constraint(%12, %13, %14, %15, %16) {monomials = [{coeff = -1 : i64, vars = [4]}], upper_bound = -262144 : i64}
      %17 = loom.expression(%12, %13) {coeffs = [1, 1], logic = "add"} : index
      %18 = loom.expression(%14, %17) {coeffs = [1, 1], logic = "mul"} : index
      loom.polynomial_constraint(%12, %14, %13, %17, %18) {monomials = [{coeff = 1 : i64, vars = [4]}], upper_bound = 374784 : i64}
      loom.polynomial_constraint(%13) {monomials = [{coeff = 1 : i64, vars = [0]}], upper_bound = 16 : i64}
      %19 = loom.expression(%14, %12) {coeffs = [1, 1], logic = "mul"} : index
      loom.polynomial_constraint(%13, %12, %14, %19) {monomials = [{coeff = 1024 : i64, vars = [0]}, {coeff = 1 : i64, vars = [3]}], upper_bound = 374784 : i64}
      %20 = loom.expression(%14, %12) {coeffs = [1, 1], logic = "mul"} : index
      %21 = loom.expression(%13, %14) {coeffs = [1, 1], logic = "mul"} : index
      %22 = loom.expression(%13, %12) {coeffs = [1, 1], logic = "mul"} : index
      %23 = loom.expression(%14, %22) {coeffs = [1, 1], logic = "mul"} : index
      loom.polynomial_constraint(%12, %13, %14, %20, %21, %22, %23) {monomials = [{coeff = 64 : i64, vars = [3]}, {coeff = 64 : i64, vars = [4]}, {coeff = -1 : i64, vars = [6]}], upper_bound = 0 : i64}
    }
    func.func @matmul_kernel__d0i1_d1i1__f10__hoist_block_1__d_d(%arg0: memref<*xf32> {tt.divisibility = 16 : i32}, %arg1: memref<*xf32> {tt.divisibility = 16 : i32}, %arg2: memref<*xf32> {tt.divisibility = 16 : i32}, %arg3: index, %arg4: index, %arg5: index, %arg6: index, %arg7: index, %arg8: index) {
      affine.parallel (%arg9) = (0) to (8) {
        affine.parallel (%arg10) = (0) to (8) {
          affine.for %arg11 = 0 to affine_map<()[s0, s1] -> ((s1 ceildiv 8) ceildiv 8)>()[%arg3, %arg4] {
            affine.for %arg12 = 0 to affine_map<()[s0, s1] -> (s0)>()[%arg3, %arg4] {
              %12 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 8 + d1 * 8 + d2)>(%arg9, %arg10, %arg11)
              %c1024 = arith.constant 1024 : index
              %c1024_0 = arith.constant 1024 : index
              %c1 = arith.constant 1 : index
              %13 = loom.get_symbolic_block_size @constraints::@BM : index
              %14 = loom.get_symbolic_block_size @constraints::@BN : index
              %15 = loom.get_symbolic_block_size @constraints::@BK : index
              %16 = arith.ceildivsi %c1024_0, %15 : index
              %cst = arith.constant 0.000000e+00 : f32
              %c1024_1 = arith.constant 1024 : index
              %c0 = arith.constant 0 : index
              %17 = tensor.empty(%13, %14) : tensor<?x?xf32>
              %18 = linalg.fill ins(%cst : f32) outs(%17 : tensor<?x?xf32>) -> tensor<?x?xf32>
              %19 = arith.muli %arg12, %13 : index
              %20 = arith.muli %12, %14 : index
              %21 = arith.muli %15, %c1024 : index
              %c0_2 = arith.constant 0 : index
              %22 = arith.muli %c0_2, %15 : index
              %23 = arith.muli %22, %c1024_1 : index
              %24 = arith.addi %23, %20 : index
              %25 = loom.reinterpret_cast %arg1 to offset : [%24], sizes : [%16, %15, %14], strides : [%21, %c1024, %c1], reuse : [seq = false, spat = false, temp = true] : memref<*xf32> to memref<?x?x?xf32, strided<[?, ?, ?], offset: ?>>
              %26 = loom.alloc(%16, %15, %14) on @L1 : memref<?x?x?xf32>
              loom.copy %25, %26, interconnect : [], broadcast : [1, 1] : memref<?x?x?xf32, strided<[?, ?, ?], offset: ?>> to memref<?x?x?xf32>
              %27 = scf.for %arg13 = %c0 to %16 step %c1 iter_args(%arg14 = %18) -> (tensor<?x?xf32>) {
                %31 = arith.muli %arg13, %15 : index
                %32 = arith.muli %19, %c1024_1 : index
                %33 = arith.addi %32, %31 : index
                %34 = loom.reinterpret_cast %arg0 to offset : [%33], sizes : [%13, %15], strides : [%c1024_0, %c1], reuse : [seq = false, spat = true, temp = true] : memref<*xf32> to memref<?x?xf32, strided<[?, ?], offset: ?>>
                %35 = loom.alloc(%13, %15) on @L1 : memref<?x?xf32>
                loom.copy %34, %35, interconnect : [], broadcast : [1, 1] : memref<?x?xf32, strided<[?, ?], offset: ?>> to memref<?x?xf32>
                %36 = bufferization.to_tensor %35 restrict writable : memref<?x?xf32> to tensor<?x?xf32>
                %37 = arith.muli %31, %c1024_1 : index
                %subview = memref.subview %26[%arg13, 0, 0] [1, %15, %14] [1, 1, 1] : memref<?x?x?xf32> to memref<?x?xf32, strided<[?, 1], offset: ?>>
                %38 = bufferization.to_tensor %subview restrict writable : memref<?x?xf32, strided<[?, 1], offset: ?>> to tensor<?x?xf32>
                %39 = linalg.matmul ins(%36, %38 : tensor<?x?xf32>, tensor<?x?xf32>) outs(%18 : tensor<?x?xf32>) -> tensor<?x?xf32>
                %40 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%arg14, %39 : tensor<?x?xf32>, tensor<?x?xf32>) outs(%arg14 : tensor<?x?xf32>) {
                ^bb0(%in: f32, %in_3: f32, %out: f32):
                  %41 = arith.addf %in, %in_3 : f32
                  linalg.yield %41 : f32
                } -> tensor<?x?xf32>
                scf.yield %40 : tensor<?x?xf32>
              }
              %28 = arith.muli %19, %c1024_1 : index
              %29 = arith.addi %28, %20 : index
              %30 = loom.reinterpret_cast %arg2 to offset : [%29], sizes : [%13, %14], strides : [%c1024, %c1], reuse : [seq = false, spat = false, temp = false] : memref<*xf32> to memref<?x?xf32, strided<[?, ?], offset: ?>>
              bufferization.materialize_in_destination %27 in writable %30 : (tensor<?x?xf32>, memref<?x?xf32, strided<[?, ?], offset: ?>>) -> ()
            }
          }
        } {loom.mapped_to = @y}
      } {loom.mapped_to = @x}
      return
    }
  }
  module attributes {loom.pass_name = "EnumerateCopyBroadcast"} {
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
      %15 = loom.expression(%12, %13) {coeffs = [1, 1], logic = "mul"} : index
      %16 = loom.expression(%15, %14) {coeffs = [1, 1], logic = "mul"} : index
      loom.polynomial_constraint(%12, %13, %14, %15, %16) {monomials = [{coeff = -1 : i64, vars = [4]}], upper_bound = -262144 : i64}
      %17 = loom.expression(%12, %13) {coeffs = [1, 1], logic = "add"} : index
      %18 = loom.expression(%14, %17) {coeffs = [1, 1], logic = "mul"} : index
      loom.polynomial_constraint(%12, %14, %13, %17, %18) {monomials = [{coeff = 1 : i64, vars = [4]}], upper_bound = 374784 : i64}
      loom.polynomial_constraint(%13) {monomials = [{coeff = 1 : i64, vars = [0]}], upper_bound = 16 : i64}
      %19 = loom.expression(%12, %14) {coeffs = [1, 1], logic = "mul"} : index
      loom.polynomial_constraint(%13, %12, %14, %19) {monomials = [{coeff = 1024 : i64, vars = [0]}, {coeff = 1 : i64, vars = [3]}], upper_bound = 374784 : i64}
      %20 = loom.expression(%12, %14) {coeffs = [1, 1], logic = "mul"} : index
      %21 = loom.expression(%13, %14) {coeffs = [1, 1], logic = "mul"} : index
      %22 = loom.expression(%12, %13) {coeffs = [1, 1], logic = "mul"} : index
      %23 = loom.expression(%14, %22) {coeffs = [1, 1], logic = "mul"} : index
      loom.polynomial_constraint(%12, %13, %14, %20, %21, %22, %23) {monomials = [{coeff = 960 : i64, vars = [3]}, {coeff = 256 : i64, vars = [4]}, {coeff = -15 : i64, vars = [6]}], upper_bound = 0 : i64}
    }
    func.func @matmul_kernel__d0i1_d1i1__f10__hoist_block_1__d_a(%arg0: memref<*xf32> {tt.divisibility = 16 : i32}, %arg1: memref<*xf32> {tt.divisibility = 16 : i32}, %arg2: memref<*xf32> {tt.divisibility = 16 : i32}, %arg3: index, %arg4: index, %arg5: index, %arg6: index, %arg7: index, %arg8: index) {
      affine.parallel (%arg9) = (0) to (8) {
        affine.parallel (%arg10) = (0) to (8) {
          affine.for %arg11 = 0 to affine_map<()[s0, s1] -> ((s1 ceildiv 8) ceildiv 8)>()[%arg3, %arg4] {
            affine.for %arg12 = 0 to affine_map<()[s0, s1] -> (s0)>()[%arg3, %arg4] {
              %12 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 8 + d1 * 8 + d2)>(%arg9, %arg10, %arg11)
              %c1024 = arith.constant 1024 : index
              %c1024_0 = arith.constant 1024 : index
              %c1 = arith.constant 1 : index
              %13 = loom.get_symbolic_block_size @constraints::@BM : index
              %14 = loom.get_symbolic_block_size @constraints::@BN : index
              %15 = loom.get_symbolic_block_size @constraints::@BK : index
              %16 = arith.ceildivsi %c1024_0, %15 : index
              %cst = arith.constant 0.000000e+00 : f32
              %c1024_1 = arith.constant 1024 : index
              %c0 = arith.constant 0 : index
              %17 = tensor.empty(%13, %14) : tensor<?x?xf32>
              %18 = linalg.fill ins(%cst : f32) outs(%17 : tensor<?x?xf32>) -> tensor<?x?xf32>
              %19 = arith.muli %arg12, %13 : index
              %20 = arith.muli %12, %14 : index
              %21 = arith.muli %15, %c1024 : index
              %c0_2 = arith.constant 0 : index
              %22 = arith.muli %c0_2, %15 : index
              %23 = arith.muli %22, %c1024_1 : index
              %24 = arith.addi %23, %20 : index
              %25 = loom.reinterpret_cast %arg1 to offset : [%24], sizes : [%16, %15, %14], strides : [%21, %c1024, %c1], reuse : [seq = false, spat = false, temp = true] : memref<*xf32> to memref<?x?x?xf32, strided<[?, ?, ?], offset: ?>>
              %26 = loom.alloc(%16, %15, %14) on @L1 : memref<?x?x?xf32>
              loom.copy %25, %26, interconnect : [], broadcast : [1, 1] : memref<?x?x?xf32, strided<[?, ?, ?], offset: ?>> to memref<?x?x?xf32>
              %27 = scf.for %arg13 = %c0 to %16 step %c1 iter_args(%arg14 = %18) -> (tensor<?x?xf32>) {
                %31 = arith.muli %arg13, %15 : index
                %32 = arith.muli %19, %c1024_1 : index
                %33 = arith.addi %32, %31 : index
                %34 = loom.reinterpret_cast %arg0 to offset : [%33], sizes : [%13, %15], strides : [%c1024_0, %c1], reuse : [seq = false, spat = true, temp = true] : memref<*xf32> to memref<?x?xf32, strided<[?, ?], offset: ?>>
                %35 = loom.alloc(%13, %15) on @L1 : memref<?x?xf32>
                loom.copy %34, %35, interconnect : [@horizontal_links, @vertical_links], broadcast : [8, 8] : memref<?x?xf32, strided<[?, ?], offset: ?>> to memref<?x?xf32>
                %36 = bufferization.to_tensor %35 restrict writable : memref<?x?xf32> to tensor<?x?xf32>
                %37 = arith.muli %31, %c1024_1 : index
                %subview = memref.subview %26[%arg13, 0, 0] [1, %15, %14] [1, 1, 1] : memref<?x?x?xf32> to memref<?x?xf32, strided<[?, 1], offset: ?>>
                %38 = bufferization.to_tensor %subview restrict writable : memref<?x?xf32, strided<[?, 1], offset: ?>> to tensor<?x?xf32>
                %39 = linalg.matmul ins(%36, %38 : tensor<?x?xf32>, tensor<?x?xf32>) outs(%18 : tensor<?x?xf32>) -> tensor<?x?xf32>
                %40 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%arg14, %39 : tensor<?x?xf32>, tensor<?x?xf32>) outs(%arg14 : tensor<?x?xf32>) {
                ^bb0(%in: f32, %in_3: f32, %out: f32):
                  %41 = arith.addf %in, %in_3 : f32
                  linalg.yield %41 : f32
                } -> tensor<?x?xf32>
                scf.yield %40 : tensor<?x?xf32>
              }
              %28 = arith.muli %19, %c1024_1 : index
              %29 = arith.addi %28, %20 : index
              %30 = loom.reinterpret_cast %arg2 to offset : [%29], sizes : [%13, %14], strides : [%c1024, %c1], reuse : [seq = false, spat = false, temp = false] : memref<*xf32> to memref<?x?xf32, strided<[?, ?], offset: ?>>
              bufferization.materialize_in_destination %27 in writable %30 : (tensor<?x?xf32>, memref<?x?xf32, strided<[?, ?], offset: ?>>) -> ()
            }
          }
        } {loom.mapped_to = @y}
      } {loom.mapped_to = @x}
      return
    }
  }
  module attributes {loom.pass_name = "EnumerateCopyBroadcast"} {
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
      %15 = loom.expression(%13, %12) {coeffs = [1, 1], logic = "mul"} : index
      %16 = loom.expression(%14, %15) {coeffs = [1, 1], logic = "mul"} : index
      loom.polynomial_constraint(%12, %13, %14, %15, %16) {monomials = [{coeff = -1 : i64, vars = [4]}], upper_bound = -262144 : i64}
      %17 = loom.expression(%12, %13) {coeffs = [1, 1], logic = "add"} : index
      %18 = loom.expression(%14, %17) {coeffs = [1, 1], logic = "mul"} : index
      loom.polynomial_constraint(%12, %14, %13, %17, %18) {monomials = [{coeff = 1 : i64, vars = [4]}], upper_bound = 374784 : i64}
      loom.polynomial_constraint(%13) {monomials = [{coeff = 1 : i64, vars = [0]}], upper_bound = 16 : i64}
      %19 = loom.expression(%14, %12) {coeffs = [1, 1], logic = "mul"} : index
      loom.polynomial_constraint(%13, %12, %14, %19) {monomials = [{coeff = 1024 : i64, vars = [0]}, {coeff = 1 : i64, vars = [3]}], upper_bound = 374784 : i64}
      %20 = loom.expression(%14, %12) {coeffs = [1, 1], logic = "mul"} : index
      %21 = loom.expression(%13, %14) {coeffs = [1, 1], logic = "mul"} : index
      %22 = loom.expression(%13, %12) {coeffs = [1, 1], logic = "mul"} : index
      %23 = loom.expression(%14, %22) {coeffs = [1, 1], logic = "mul"} : index
      loom.polynomial_constraint(%12, %13, %14, %20, %21, %22, %23) {monomials = [{coeff = 960 : i64, vars = [3]}, {coeff = 256 : i64, vars = [4]}, {coeff = -15 : i64, vars = [6]}], upper_bound = 0 : i64}
    }
    func.func @matmul_kernel__d0i1_d1i1__f10__hoist_block_1__d_h(%arg0: memref<*xf32> {tt.divisibility = 16 : i32}, %arg1: memref<*xf32> {tt.divisibility = 16 : i32}, %arg2: memref<*xf32> {tt.divisibility = 16 : i32}, %arg3: index, %arg4: index, %arg5: index, %arg6: index, %arg7: index, %arg8: index) {
      affine.parallel (%arg9) = (0) to (8) {
        affine.parallel (%arg10) = (0) to (8) {
          affine.for %arg11 = 0 to affine_map<()[s0, s1] -> ((s1 ceildiv 8) ceildiv 8)>()[%arg3, %arg4] {
            affine.for %arg12 = 0 to affine_map<()[s0, s1] -> (s0)>()[%arg3, %arg4] {
              %12 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 8 + d1 * 8 + d2)>(%arg9, %arg10, %arg11)
              %c1024 = arith.constant 1024 : index
              %c1024_0 = arith.constant 1024 : index
              %c1 = arith.constant 1 : index
              %13 = loom.get_symbolic_block_size @constraints::@BM : index
              %14 = loom.get_symbolic_block_size @constraints::@BN : index
              %15 = loom.get_symbolic_block_size @constraints::@BK : index
              %16 = arith.ceildivsi %c1024_0, %15 : index
              %cst = arith.constant 0.000000e+00 : f32
              %c1024_1 = arith.constant 1024 : index
              %c0 = arith.constant 0 : index
              %17 = tensor.empty(%13, %14) : tensor<?x?xf32>
              %18 = linalg.fill ins(%cst : f32) outs(%17 : tensor<?x?xf32>) -> tensor<?x?xf32>
              %19 = arith.muli %arg12, %13 : index
              %20 = arith.muli %12, %14 : index
              %21 = arith.muli %15, %c1024 : index
              %c0_2 = arith.constant 0 : index
              %22 = arith.muli %c0_2, %15 : index
              %23 = arith.muli %22, %c1024_1 : index
              %24 = arith.addi %23, %20 : index
              %25 = loom.reinterpret_cast %arg1 to offset : [%24], sizes : [%16, %15, %14], strides : [%21, %c1024, %c1], reuse : [seq = false, spat = false, temp = true] : memref<*xf32> to memref<?x?x?xf32, strided<[?, ?, ?], offset: ?>>
              %26 = loom.alloc(%16, %15, %14) on @L1 : memref<?x?x?xf32>
              loom.copy %25, %26, interconnect : [], broadcast : [1, 1] : memref<?x?x?xf32, strided<[?, ?, ?], offset: ?>> to memref<?x?x?xf32>
              %27 = scf.for %arg13 = %c0 to %16 step %c1 iter_args(%arg14 = %18) -> (tensor<?x?xf32>) {
                %31 = arith.muli %arg13, %15 : index
                %32 = arith.muli %19, %c1024_1 : index
                %33 = arith.addi %32, %31 : index
                %34 = loom.reinterpret_cast %arg0 to offset : [%33], sizes : [%13, %15], strides : [%c1024_0, %c1], reuse : [seq = false, spat = true, temp = true] : memref<*xf32> to memref<?x?xf32, strided<[?, ?], offset: ?>>
                %35 = loom.alloc(%13, %15) on @L1 : memref<?x?xf32>
                loom.copy %34, %35, interconnect : [@horizontal_links], broadcast : [1, 8] : memref<?x?xf32, strided<[?, ?], offset: ?>> to memref<?x?xf32>
                %36 = bufferization.to_tensor %35 restrict writable : memref<?x?xf32> to tensor<?x?xf32>
                %37 = arith.muli %31, %c1024_1 : index
                %subview = memref.subview %26[%arg13, 0, 0] [1, %15, %14] [1, 1, 1] : memref<?x?x?xf32> to memref<?x?xf32, strided<[?, 1], offset: ?>>
                %38 = bufferization.to_tensor %subview restrict writable : memref<?x?xf32, strided<[?, 1], offset: ?>> to tensor<?x?xf32>
                %39 = linalg.matmul ins(%36, %38 : tensor<?x?xf32>, tensor<?x?xf32>) outs(%18 : tensor<?x?xf32>) -> tensor<?x?xf32>
                %40 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%arg14, %39 : tensor<?x?xf32>, tensor<?x?xf32>) outs(%arg14 : tensor<?x?xf32>) {
                ^bb0(%in: f32, %in_3: f32, %out: f32):
                  %41 = arith.addf %in, %in_3 : f32
                  linalg.yield %41 : f32
                } -> tensor<?x?xf32>
                scf.yield %40 : tensor<?x?xf32>
              }
              %28 = arith.muli %19, %c1024_1 : index
              %29 = arith.addi %28, %20 : index
              %30 = loom.reinterpret_cast %arg2 to offset : [%29], sizes : [%13, %14], strides : [%c1024, %c1], reuse : [seq = false, spat = false, temp = false] : memref<*xf32> to memref<?x?xf32, strided<[?, ?], offset: ?>>
              bufferization.materialize_in_destination %27 in writable %30 : (tensor<?x?xf32>, memref<?x?xf32, strided<[?, ?], offset: ?>>) -> ()
            }
          }
        } {loom.mapped_to = @y}
      } {loom.mapped_to = @x}
      return
    }
  }
  module attributes {loom.pass_name = "EnumerateCopyBroadcast"} {
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
      %15 = loom.expression(%13, %12) {coeffs = [1, 1], logic = "mul"} : index
      %16 = loom.expression(%15, %14) {coeffs = [1, 1], logic = "mul"} : index
      loom.polynomial_constraint(%12, %13, %14, %15, %16) {monomials = [{coeff = -1 : i64, vars = [4]}], upper_bound = -262144 : i64}
      %17 = loom.expression(%12, %13) {coeffs = [1, 1], logic = "add"} : index
      %18 = loom.expression(%14, %17) {coeffs = [1, 1], logic = "mul"} : index
      loom.polynomial_constraint(%12, %14, %13, %17, %18) {monomials = [{coeff = 1 : i64, vars = [4]}], upper_bound = 374784 : i64}
      loom.polynomial_constraint(%13) {monomials = [{coeff = 1 : i64, vars = [0]}], upper_bound = 16 : i64}
      %19 = loom.expression(%14, %12) {coeffs = [1, 1], logic = "mul"} : index
      loom.polynomial_constraint(%13, %12, %14, %19) {monomials = [{coeff = 1024 : i64, vars = [0]}, {coeff = 1 : i64, vars = [3]}], upper_bound = 374784 : i64}
      %20 = loom.expression(%14, %12) {coeffs = [1, 1], logic = "mul"} : index
      %21 = loom.expression(%13, %14) {coeffs = [1, 1], logic = "mul"} : index
      %22 = loom.expression(%13, %12) {coeffs = [1, 1], logic = "mul"} : index
      %23 = loom.expression(%14, %22) {coeffs = [1, 1], logic = "mul"} : index
      loom.polynomial_constraint(%12, %13, %14, %20, %21, %22, %23) {monomials = [{coeff = 960 : i64, vars = [3]}, {coeff = 256 : i64, vars = [4]}, {coeff = -15 : i64, vars = [6]}], upper_bound = 0 : i64}
    }
    func.func @matmul_kernel__d0i1_d1i1__f10__hoist_block_1__d_v(%arg0: memref<*xf32> {tt.divisibility = 16 : i32}, %arg1: memref<*xf32> {tt.divisibility = 16 : i32}, %arg2: memref<*xf32> {tt.divisibility = 16 : i32}, %arg3: index, %arg4: index, %arg5: index, %arg6: index, %arg7: index, %arg8: index) {
      affine.parallel (%arg9) = (0) to (8) {
        affine.parallel (%arg10) = (0) to (8) {
          affine.for %arg11 = 0 to affine_map<()[s0, s1] -> ((s1 ceildiv 8) ceildiv 8)>()[%arg3, %arg4] {
            affine.for %arg12 = 0 to affine_map<()[s0, s1] -> (s0)>()[%arg3, %arg4] {
              %12 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 8 + d1 * 8 + d2)>(%arg9, %arg10, %arg11)
              %c1024 = arith.constant 1024 : index
              %c1024_0 = arith.constant 1024 : index
              %c1 = arith.constant 1 : index
              %13 = loom.get_symbolic_block_size @constraints::@BM : index
              %14 = loom.get_symbolic_block_size @constraints::@BN : index
              %15 = loom.get_symbolic_block_size @constraints::@BK : index
              %16 = arith.ceildivsi %c1024_0, %15 : index
              %cst = arith.constant 0.000000e+00 : f32
              %c1024_1 = arith.constant 1024 : index
              %c0 = arith.constant 0 : index
              %17 = tensor.empty(%13, %14) : tensor<?x?xf32>
              %18 = linalg.fill ins(%cst : f32) outs(%17 : tensor<?x?xf32>) -> tensor<?x?xf32>
              %19 = arith.muli %arg12, %13 : index
              %20 = arith.muli %12, %14 : index
              %21 = arith.muli %15, %c1024 : index
              %c0_2 = arith.constant 0 : index
              %22 = arith.muli %c0_2, %15 : index
              %23 = arith.muli %22, %c1024_1 : index
              %24 = arith.addi %23, %20 : index
              %25 = loom.reinterpret_cast %arg1 to offset : [%24], sizes : [%16, %15, %14], strides : [%21, %c1024, %c1], reuse : [seq = false, spat = false, temp = true] : memref<*xf32> to memref<?x?x?xf32, strided<[?, ?, ?], offset: ?>>
              %26 = loom.alloc(%16, %15, %14) on @L1 : memref<?x?x?xf32>
              loom.copy %25, %26, interconnect : [], broadcast : [1, 1] : memref<?x?x?xf32, strided<[?, ?, ?], offset: ?>> to memref<?x?x?xf32>
              %27 = scf.for %arg13 = %c0 to %16 step %c1 iter_args(%arg14 = %18) -> (tensor<?x?xf32>) {
                %31 = arith.muli %arg13, %15 : index
                %32 = arith.muli %19, %c1024_1 : index
                %33 = arith.addi %32, %31 : index
                %34 = loom.reinterpret_cast %arg0 to offset : [%33], sizes : [%13, %15], strides : [%c1024_0, %c1], reuse : [seq = false, spat = true, temp = true] : memref<*xf32> to memref<?x?xf32, strided<[?, ?], offset: ?>>
                %35 = loom.alloc(%13, %15) on @L1 : memref<?x?xf32>
                loom.copy %34, %35, interconnect : [@vertical_links], broadcast : [8, 1] : memref<?x?xf32, strided<[?, ?], offset: ?>> to memref<?x?xf32>
                %36 = bufferization.to_tensor %35 restrict writable : memref<?x?xf32> to tensor<?x?xf32>
                %37 = arith.muli %31, %c1024_1 : index
                %subview = memref.subview %26[%arg13, 0, 0] [1, %15, %14] [1, 1, 1] : memref<?x?x?xf32> to memref<?x?xf32, strided<[?, 1], offset: ?>>
                %38 = bufferization.to_tensor %subview restrict writable : memref<?x?xf32, strided<[?, 1], offset: ?>> to tensor<?x?xf32>
                %39 = linalg.matmul ins(%36, %38 : tensor<?x?xf32>, tensor<?x?xf32>) outs(%18 : tensor<?x?xf32>) -> tensor<?x?xf32>
                %40 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%arg14, %39 : tensor<?x?xf32>, tensor<?x?xf32>) outs(%arg14 : tensor<?x?xf32>) {
                ^bb0(%in: f32, %in_3: f32, %out: f32):
                  %41 = arith.addf %in, %in_3 : f32
                  linalg.yield %41 : f32
                } -> tensor<?x?xf32>
                scf.yield %40 : tensor<?x?xf32>
              }
              %28 = arith.muli %19, %c1024_1 : index
              %29 = arith.addi %28, %20 : index
              %30 = loom.reinterpret_cast %arg2 to offset : [%29], sizes : [%13, %14], strides : [%c1024, %c1], reuse : [seq = false, spat = false, temp = false] : memref<*xf32> to memref<?x?xf32, strided<[?, ?], offset: ?>>
              bufferization.materialize_in_destination %27 in writable %30 : (tensor<?x?xf32>, memref<?x?xf32, strided<[?, ?], offset: ?>>) -> ()
            }
          }
        } {loom.mapped_to = @y}
      } {loom.mapped_to = @x}
      return
    }
  }
}
