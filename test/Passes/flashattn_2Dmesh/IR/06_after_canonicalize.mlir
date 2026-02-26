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
    func.func @attention__d0i0_d1i0__f01__d_d_d__BB32__BM32__BN32(%arg0: memref<32x128x4096xf32>, %arg1: memref<32x4096x128xf32>, %arg2: memref<32x4096x128xf32>, %arg3: memref<32x4096x128xf32>) {
      %cst = arith.constant 2.000000e+00 : f32
      %cst_0 = arith.constant 0.12751743074602467 : f64
      %cst_1 = arith.constant 0.000000e+00 : f32
      %cst_2 = arith.constant 1.000000e+00 : f32
      %cst_3 = arith.constant 0xFF800000 : f32
      %c32 = arith.constant 32 : index
      affine.parallel (%arg4) = (0) to (8) {
        affine.parallel (%arg5) = (0) to (8) {
          affine.for %arg6 = 0 to 1 {
            affine.for %arg7 = 0 to 128 {
              %12 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 8 + d1 + d2 * 64)>(%arg4, %arg5, %arg6)
              %13 = loom.alloc [32, 32, 128] on @L1 : memref<32x32x128xf32>
              %14 = loom.alloc [32, 32, 32] on @L1 : memref<32x32x32xf32>
              %15 = loom.init_tensor %14[32, 32, 32] : memref<32x32x32xf32> -> tensor<32x32x32xf32>
              %16 = loom.alloc [32, 128, 32] on @L1 : memref<32x128x32xf32>
              %17 = loom.alloc [32, 32] on @L1 : memref<32x32xf32>
              %18 = loom.init_tensor %17[32, 32] : memref<32x32xf32> -> tensor<32x32xf32>
              %19 = loom.alloc [32, 32] on @L1 : memref<32x32xf32>
              %20 = loom.init_tensor %19[32, 32] : memref<32x32xf32> -> tensor<32x32xf32>
              %21 = loom.alloc [32, 32] on @L1 : memref<32x32xf32>
              %22 = loom.init_tensor %21[32, 32] : memref<32x32xf32> -> tensor<32x32xf32>
              %23 = loom.alloc [32, 32] on @L1 : memref<32x32xf32>
              %24 = loom.init_tensor %23[32, 32] : memref<32x32xf32> -> tensor<32x32xf32>
              %25 = loom.alloc [32, 32] on @L1 : memref<32x32xf32>
              %26 = loom.init_tensor %25[32, 32] : memref<32x32xf32> -> tensor<32x32xf32>
              %27 = loom.alloc [32, 32, 128] on @L1 : memref<32x32x128xf32>
              %28 = loom.init_tensor %27[32, 32, 128] : memref<32x32x128xf32> -> tensor<32x32x128xf32>
              %29 = loom.alloc [32, 32, 128] on @L1 : memref<32x32x128xf32>
              %30 = loom.init_tensor %29[32, 32, 128] : memref<32x32x128xf32> -> tensor<32x32x128xf32>
              %31 = loom.alloc [32, 32, 128] on @L1 : memref<32x32x128xf32>
              %32 = loom.init_tensor %31[32, 32, 128] : memref<32x32x128xf32> -> tensor<32x32x128xf32>
              %33 = arith.muli %12, %c32 : index
              %34 = arith.muli %arg7, %c32 : index
              %c0 = arith.constant 0 : index
              %35 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 524288 + d1 * 128 + d2)>(%33, %34, %c0)
              %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%35], sizes: [32, 32, 128], strides: [524288, 128, 1] : memref<32x4096x128xf32> to memref<32x32x128xf32, strided<[524288, 128, 1], offset: ?>>
              %36 = loom.copy_to_tensor %reinterpret_cast, %27 on @L1, interconnect : [], broadcast : [1, 1] : memref<32x32x128xf32, strided<[524288, 128, 1], offset: ?>>, memref<32x32x128xf32> -> tensor<32x32x128xf32>
              %37 = linalg.fill ins(%cst_1 : f32) outs(%30 : tensor<32x32x128xf32>) -> tensor<32x32x128xf32>
              %38 = linalg.fill ins(%cst_2 : f32) outs(%18 : tensor<32x32xf32>) -> tensor<32x32xf32>
              %39 = linalg.fill ins(%cst_3 : f32) outs(%20 : tensor<32x32xf32>) -> tensor<32x32xf32>
              %40:3 = affine.for %arg8 = 0 to 128 iter_args(%arg9 = %39, %arg10 = %38, %arg11 = %37) -> (tensor<32x32xf32>, tensor<32x32xf32>, tensor<32x32x128xf32>) {
                %43 = arith.muli %arg8, %c32 : index
                %c0_6 = arith.constant 0 : index
                %44 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 524288 + d1 * 4096 + d2)>(%33, %c0_6, %43)
                %reinterpret_cast_7 = memref.reinterpret_cast %arg0 to offset: [%44], sizes: [32, 128, 32], strides: [524288, 4096, 1] : memref<32x128x4096xf32> to memref<32x128x32xf32, strided<[524288, 4096, 1], offset: ?>>
                %45 = loom.copy_to_tensor %reinterpret_cast_7, %16 on @L1, interconnect : [], broadcast : [1, 1] : memref<32x128x32xf32, strided<[524288, 4096, 1], offset: ?>>, memref<32x128x32xf32> -> tensor<32x128x32xf32>
                %46 = linalg.fill ins(%cst_1 : f32) outs(%15 : tensor<32x32x32xf32>) -> tensor<32x32x32xf32>
                %47 = linalg.batch_matmul ins(%36, %45 : tensor<32x32x128xf32>, tensor<32x128x32xf32>) outs(%46 : tensor<32x32x32xf32>) -> tensor<32x32x32xf32>
                %48 = linalg.fill ins(%cst_3 : f32) outs(%22 : tensor<32x32xf32>) -> tensor<32x32xf32>
                %49 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%47 : tensor<32x32x32xf32>) outs(%48 : tensor<32x32xf32>) {
                ^bb0(%in: f32, %out: f32):
                  %62 = arith.maximumf %in, %out : f32
                  linalg.yield %62 : f32
                } -> tensor<32x32xf32>
                %50 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%arg9, %49 : tensor<32x32xf32>, tensor<32x32xf32>) outs(%22 : tensor<32x32xf32>) {
                ^bb0(%in: f32, %in_10: f32, %out: f32):
                  %62 = arith.truncf %cst_0 : f64 to f32
                  %63 = arith.mulf %in_10, %62 : f32
                  %64 = arith.cmpf ogt, %in, %63 : f32
                  %65 = arith.select %64, %in, %63 : f32
                  linalg.yield %65 : f32
                } -> tensor<32x32xf32>
                %51 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%47, %50 : tensor<32x32x32xf32>, tensor<32x32xf32>) outs(%15 : tensor<32x32x32xf32>) {
                ^bb0(%in: f32, %in_10: f32, %out: f32):
                  %62 = arith.truncf %cst_0 : f64 to f32
                  %63 = arith.mulf %in, %62 : f32
                  %64 = arith.subf %63, %in_10 : f32
                  %65 = math.powf %cst, %64 : f32
                  linalg.yield %65 : f32
                } -> tensor<32x32x32xf32>
                %52 = linalg.fill ins(%cst_1 : f32) outs(%24 : tensor<32x32xf32>) -> tensor<32x32xf32>
                %53 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%51 : tensor<32x32x32xf32>) outs(%52 : tensor<32x32xf32>) {
                ^bb0(%in: f32, %out: f32):
                  %62 = arith.addf %in, %out : f32
                  linalg.yield %62 : f32
                } -> tensor<32x32xf32>
                %54 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%arg9, %50 : tensor<32x32xf32>, tensor<32x32xf32>) outs(%26 : tensor<32x32xf32>) {
                ^bb0(%in: f32, %in_10: f32, %out: f32):
                  %62 = arith.subf %in, %in_10 : f32
                  %63 = math.powf %cst, %62 : f32
                  linalg.yield %63 : f32
                } -> tensor<32x32xf32>
                %55 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%arg10, %54, %53 : tensor<32x32xf32>, tensor<32x32xf32>, tensor<32x32xf32>) outs(%arg10 : tensor<32x32xf32>) {
                ^bb0(%in: f32, %in_10: f32, %in_11: f32, %out: f32):
                  %62 = arith.mulf %in, %in_10 : f32
                  %63 = arith.addf %62, %in_11 : f32
                  linalg.yield %63 : f32
                } -> tensor<32x32xf32>
                %c0_8 = arith.constant 0 : index
                %56 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 524288 + d1 * 128 + d2)>(%33, %43, %c0_8)
                %reinterpret_cast_9 = memref.reinterpret_cast %arg1 to offset: [%56], sizes: [32, 32, 128], strides: [524288, 128, 1] : memref<32x4096x128xf32> to memref<32x32x128xf32, strided<[524288, 128, 1], offset: ?>>
                %57 = loom.copy_to_tensor %reinterpret_cast_9, %13 on @L1, interconnect : [], broadcast : [1, 1] : memref<32x32x128xf32, strided<[524288, 128, 1], offset: ?>>, memref<32x32x128xf32> -> tensor<32x32x128xf32>
                %58 = linalg.fill ins(%cst_1 : f32) outs(%32 : tensor<32x32x128xf32>) -> tensor<32x32x128xf32>
                %59 = linalg.batch_matmul ins(%51, %57 : tensor<32x32x32xf32>, tensor<32x32x128xf32>) outs(%58 : tensor<32x32x128xf32>) -> tensor<32x32x128xf32>
                %60 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%59, %arg11, %54 : tensor<32x32x128xf32>, tensor<32x32x128xf32>, tensor<32x32xf32>) outs(%arg11 : tensor<32x32x128xf32>) {
                ^bb0(%in: f32, %in_10: f32, %in_11: f32, %out: f32):
                  %62 = arith.mulf %in_10, %in_11 : f32
                  %63 = arith.addf %in, %62 : f32
                  linalg.yield %63 : f32
                } -> tensor<32x32x128xf32>
                %61 = linalg.copy ins(%50 : tensor<32x32xf32>) outs(%arg9 : tensor<32x32xf32>) -> tensor<32x32xf32>
                affine.yield %61, %55, %60 : tensor<32x32xf32>, tensor<32x32xf32>, tensor<32x32x128xf32>
              }
              %41 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%40#2, %40#1 : tensor<32x32x128xf32>, tensor<32x32xf32>) outs(%28 : tensor<32x32x128xf32>) {
              ^bb0(%in: f32, %in_6: f32, %out: f32):
                %43 = arith.divf %in, %in_6 : f32
                linalg.yield %43 : f32
              } -> tensor<32x32x128xf32>
              %c0_4 = arith.constant 0 : index
              %42 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 524288 + d1 * 128 + d2)>(%33, %34, %c0_4)
              %reinterpret_cast_5 = memref.reinterpret_cast %arg3 to offset: [%42], sizes: [32, 32, 128], strides: [524288, 128, 1] : memref<32x4096x128xf32> to memref<32x32x128xf32, strided<[524288, 128, 1], offset: ?>>
              loom.copy_from_tensor %41, %reinterpret_cast_5 on @DRAM : tensor<32x32x128xf32>, memref<32x32x128xf32, strided<[524288, 128, 1], offset: ?>>
            } {loom.iter_type = #loom.iter_type<temporal>}
          } {loom.iter_type = #loom.iter_type<temporal>}
        } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @y}
      } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @x}
      return
    }
  }
  module attributes {loom.pass_name = "Materialize"} {
    func.func @attention__d0i0_d1i0__f10__d_d_d__BB32__BM32__BN32(%arg0: memref<32x128x4096xf32>, %arg1: memref<32x4096x128xf32>, %arg2: memref<32x4096x128xf32>, %arg3: memref<32x4096x128xf32>) {
      %cst = arith.constant 2.000000e+00 : f32
      %cst_0 = arith.constant 0.12751743074602467 : f64
      %cst_1 = arith.constant 0.000000e+00 : f32
      %cst_2 = arith.constant 1.000000e+00 : f32
      %cst_3 = arith.constant 0xFF800000 : f32
      %c32 = arith.constant 32 : index
      affine.parallel (%arg4) = (0) to (8) {
        affine.parallel (%arg5) = (0) to (8) {
          affine.for %arg6 = 0 to 128 {
            affine.for %arg7 = 0 to 1 {
              %12 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 8 + d1 + d2 * 64)>(%arg4, %arg5, %arg7)
              %13 = loom.alloc [32, 32, 128] on @L1 : memref<32x32x128xf32>
              %14 = loom.alloc [32, 32, 32] on @L1 : memref<32x32x32xf32>
              %15 = loom.init_tensor %14[32, 32, 32] : memref<32x32x32xf32> -> tensor<32x32x32xf32>
              %16 = loom.alloc [32, 128, 32] on @L1 : memref<32x128x32xf32>
              %17 = loom.alloc [32, 32] on @L1 : memref<32x32xf32>
              %18 = loom.init_tensor %17[32, 32] : memref<32x32xf32> -> tensor<32x32xf32>
              %19 = loom.alloc [32, 32] on @L1 : memref<32x32xf32>
              %20 = loom.init_tensor %19[32, 32] : memref<32x32xf32> -> tensor<32x32xf32>
              %21 = loom.alloc [32, 32] on @L1 : memref<32x32xf32>
              %22 = loom.init_tensor %21[32, 32] : memref<32x32xf32> -> tensor<32x32xf32>
              %23 = loom.alloc [32, 32] on @L1 : memref<32x32xf32>
              %24 = loom.init_tensor %23[32, 32] : memref<32x32xf32> -> tensor<32x32xf32>
              %25 = loom.alloc [32, 32] on @L1 : memref<32x32xf32>
              %26 = loom.init_tensor %25[32, 32] : memref<32x32xf32> -> tensor<32x32xf32>
              %27 = loom.alloc [32, 32, 128] on @L1 : memref<32x32x128xf32>
              %28 = loom.init_tensor %27[32, 32, 128] : memref<32x32x128xf32> -> tensor<32x32x128xf32>
              %29 = loom.alloc [32, 32, 128] on @L1 : memref<32x32x128xf32>
              %30 = loom.init_tensor %29[32, 32, 128] : memref<32x32x128xf32> -> tensor<32x32x128xf32>
              %31 = loom.alloc [32, 32, 128] on @L1 : memref<32x32x128xf32>
              %32 = loom.init_tensor %31[32, 32, 128] : memref<32x32x128xf32> -> tensor<32x32x128xf32>
              %33 = arith.muli %12, %c32 : index
              %34 = arith.muli %arg6, %c32 : index
              %c0 = arith.constant 0 : index
              %35 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 524288 + d1 * 128 + d2)>(%33, %34, %c0)
              %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%35], sizes: [32, 32, 128], strides: [524288, 128, 1] : memref<32x4096x128xf32> to memref<32x32x128xf32, strided<[524288, 128, 1], offset: ?>>
              %36 = loom.copy_to_tensor %reinterpret_cast, %27 on @L1, interconnect : [], broadcast : [1, 1] : memref<32x32x128xf32, strided<[524288, 128, 1], offset: ?>>, memref<32x32x128xf32> -> tensor<32x32x128xf32>
              %37 = linalg.fill ins(%cst_1 : f32) outs(%30 : tensor<32x32x128xf32>) -> tensor<32x32x128xf32>
              %38 = linalg.fill ins(%cst_2 : f32) outs(%18 : tensor<32x32xf32>) -> tensor<32x32xf32>
              %39 = linalg.fill ins(%cst_3 : f32) outs(%20 : tensor<32x32xf32>) -> tensor<32x32xf32>
              %40:3 = affine.for %arg8 = 0 to 128 iter_args(%arg9 = %39, %arg10 = %38, %arg11 = %37) -> (tensor<32x32xf32>, tensor<32x32xf32>, tensor<32x32x128xf32>) {
                %43 = arith.muli %arg8, %c32 : index
                %c0_6 = arith.constant 0 : index
                %44 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 524288 + d1 * 4096 + d2)>(%33, %c0_6, %43)
                %reinterpret_cast_7 = memref.reinterpret_cast %arg0 to offset: [%44], sizes: [32, 128, 32], strides: [524288, 4096, 1] : memref<32x128x4096xf32> to memref<32x128x32xf32, strided<[524288, 4096, 1], offset: ?>>
                %45 = loom.copy_to_tensor %reinterpret_cast_7, %16 on @L1, interconnect : [], broadcast : [1, 1] : memref<32x128x32xf32, strided<[524288, 4096, 1], offset: ?>>, memref<32x128x32xf32> -> tensor<32x128x32xf32>
                %46 = linalg.fill ins(%cst_1 : f32) outs(%15 : tensor<32x32x32xf32>) -> tensor<32x32x32xf32>
                %47 = linalg.batch_matmul ins(%36, %45 : tensor<32x32x128xf32>, tensor<32x128x32xf32>) outs(%46 : tensor<32x32x32xf32>) -> tensor<32x32x32xf32>
                %48 = linalg.fill ins(%cst_3 : f32) outs(%22 : tensor<32x32xf32>) -> tensor<32x32xf32>
                %49 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%47 : tensor<32x32x32xf32>) outs(%48 : tensor<32x32xf32>) {
                ^bb0(%in: f32, %out: f32):
                  %62 = arith.maximumf %in, %out : f32
                  linalg.yield %62 : f32
                } -> tensor<32x32xf32>
                %50 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%arg9, %49 : tensor<32x32xf32>, tensor<32x32xf32>) outs(%22 : tensor<32x32xf32>) {
                ^bb0(%in: f32, %in_10: f32, %out: f32):
                  %62 = arith.truncf %cst_0 : f64 to f32
                  %63 = arith.mulf %in_10, %62 : f32
                  %64 = arith.cmpf ogt, %in, %63 : f32
                  %65 = arith.select %64, %in, %63 : f32
                  linalg.yield %65 : f32
                } -> tensor<32x32xf32>
                %51 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%47, %50 : tensor<32x32x32xf32>, tensor<32x32xf32>) outs(%15 : tensor<32x32x32xf32>) {
                ^bb0(%in: f32, %in_10: f32, %out: f32):
                  %62 = arith.truncf %cst_0 : f64 to f32
                  %63 = arith.mulf %in, %62 : f32
                  %64 = arith.subf %63, %in_10 : f32
                  %65 = math.powf %cst, %64 : f32
                  linalg.yield %65 : f32
                } -> tensor<32x32x32xf32>
                %52 = linalg.fill ins(%cst_1 : f32) outs(%24 : tensor<32x32xf32>) -> tensor<32x32xf32>
                %53 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%51 : tensor<32x32x32xf32>) outs(%52 : tensor<32x32xf32>) {
                ^bb0(%in: f32, %out: f32):
                  %62 = arith.addf %in, %out : f32
                  linalg.yield %62 : f32
                } -> tensor<32x32xf32>
                %54 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%arg9, %50 : tensor<32x32xf32>, tensor<32x32xf32>) outs(%26 : tensor<32x32xf32>) {
                ^bb0(%in: f32, %in_10: f32, %out: f32):
                  %62 = arith.subf %in, %in_10 : f32
                  %63 = math.powf %cst, %62 : f32
                  linalg.yield %63 : f32
                } -> tensor<32x32xf32>
                %55 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%arg10, %54, %53 : tensor<32x32xf32>, tensor<32x32xf32>, tensor<32x32xf32>) outs(%arg10 : tensor<32x32xf32>) {
                ^bb0(%in: f32, %in_10: f32, %in_11: f32, %out: f32):
                  %62 = arith.mulf %in, %in_10 : f32
                  %63 = arith.addf %62, %in_11 : f32
                  linalg.yield %63 : f32
                } -> tensor<32x32xf32>
                %c0_8 = arith.constant 0 : index
                %56 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 524288 + d1 * 128 + d2)>(%33, %43, %c0_8)
                %reinterpret_cast_9 = memref.reinterpret_cast %arg1 to offset: [%56], sizes: [32, 32, 128], strides: [524288, 128, 1] : memref<32x4096x128xf32> to memref<32x32x128xf32, strided<[524288, 128, 1], offset: ?>>
                %57 = loom.copy_to_tensor %reinterpret_cast_9, %13 on @L1, interconnect : [], broadcast : [1, 1] : memref<32x32x128xf32, strided<[524288, 128, 1], offset: ?>>, memref<32x32x128xf32> -> tensor<32x32x128xf32>
                %58 = linalg.fill ins(%cst_1 : f32) outs(%32 : tensor<32x32x128xf32>) -> tensor<32x32x128xf32>
                %59 = linalg.batch_matmul ins(%51, %57 : tensor<32x32x32xf32>, tensor<32x32x128xf32>) outs(%58 : tensor<32x32x128xf32>) -> tensor<32x32x128xf32>
                %60 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%59, %arg11, %54 : tensor<32x32x128xf32>, tensor<32x32x128xf32>, tensor<32x32xf32>) outs(%arg11 : tensor<32x32x128xf32>) {
                ^bb0(%in: f32, %in_10: f32, %in_11: f32, %out: f32):
                  %62 = arith.mulf %in_10, %in_11 : f32
                  %63 = arith.addf %in, %62 : f32
                  linalg.yield %63 : f32
                } -> tensor<32x32x128xf32>
                %61 = linalg.copy ins(%50 : tensor<32x32xf32>) outs(%arg9 : tensor<32x32xf32>) -> tensor<32x32xf32>
                affine.yield %61, %55, %60 : tensor<32x32xf32>, tensor<32x32xf32>, tensor<32x32x128xf32>
              }
              %41 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%40#2, %40#1 : tensor<32x32x128xf32>, tensor<32x32xf32>) outs(%28 : tensor<32x32x128xf32>) {
              ^bb0(%in: f32, %in_6: f32, %out: f32):
                %43 = arith.divf %in, %in_6 : f32
                linalg.yield %43 : f32
              } -> tensor<32x32x128xf32>
              %c0_4 = arith.constant 0 : index
              %42 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 524288 + d1 * 128 + d2)>(%33, %34, %c0_4)
              %reinterpret_cast_5 = memref.reinterpret_cast %arg3 to offset: [%42], sizes: [32, 32, 128], strides: [524288, 128, 1] : memref<32x4096x128xf32> to memref<32x32x128xf32, strided<[524288, 128, 1], offset: ?>>
              loom.copy_from_tensor %41, %reinterpret_cast_5 on @DRAM : tensor<32x32x128xf32>, memref<32x32x128xf32, strided<[524288, 128, 1], offset: ?>>
            } {loom.iter_type = #loom.iter_type<temporal>}
          } {loom.iter_type = #loom.iter_type<temporal>}
        } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @y}
      } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @x}
      return
    }
  }
  module attributes {loom.pass_name = "Materialize"} {
    func.func @attention__d0i0_d1i1__f01__d_d_d__BB32__BM32__BN32(%arg0: memref<32x128x4096xf32>, %arg1: memref<32x4096x128xf32>, %arg2: memref<32x4096x128xf32>, %arg3: memref<32x4096x128xf32>) {
      %cst = arith.constant 2.000000e+00 : f32
      %cst_0 = arith.constant 0.12751743074602467 : f64
      %cst_1 = arith.constant 0.000000e+00 : f32
      %cst_2 = arith.constant 1.000000e+00 : f32
      %cst_3 = arith.constant 0xFF800000 : f32
      %c32 = arith.constant 32 : index
      affine.parallel (%arg4) = (0) to (8) {
        affine.parallel (%arg5) = (0) to (8) {
          affine.for %arg6 = 0 to 1 {
            affine.for %arg7 = 0 to 16 {
              %12 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 8)>(%arg4, %arg6)
              %13 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 8)>(%arg5, %arg7)
              %14 = loom.alloc [32, 32, 128] on @L1 : memref<32x32x128xf32>
              %15 = loom.alloc [32, 32, 32] on @L1 : memref<32x32x32xf32>
              %16 = loom.init_tensor %15[32, 32, 32] : memref<32x32x32xf32> -> tensor<32x32x32xf32>
              %17 = loom.alloc [32, 128, 32] on @L1 : memref<32x128x32xf32>
              %18 = loom.alloc [32, 32] on @L1 : memref<32x32xf32>
              %19 = loom.init_tensor %18[32, 32] : memref<32x32xf32> -> tensor<32x32xf32>
              %20 = loom.alloc [32, 32] on @L1 : memref<32x32xf32>
              %21 = loom.init_tensor %20[32, 32] : memref<32x32xf32> -> tensor<32x32xf32>
              %22 = loom.alloc [32, 32] on @L1 : memref<32x32xf32>
              %23 = loom.init_tensor %22[32, 32] : memref<32x32xf32> -> tensor<32x32xf32>
              %24 = loom.alloc [32, 32] on @L1 : memref<32x32xf32>
              %25 = loom.init_tensor %24[32, 32] : memref<32x32xf32> -> tensor<32x32xf32>
              %26 = loom.alloc [32, 32] on @L1 : memref<32x32xf32>
              %27 = loom.init_tensor %26[32, 32] : memref<32x32xf32> -> tensor<32x32xf32>
              %28 = loom.alloc [32, 32, 128] on @L1 : memref<32x32x128xf32>
              %29 = loom.init_tensor %28[32, 32, 128] : memref<32x32x128xf32> -> tensor<32x32x128xf32>
              %30 = loom.alloc [32, 32, 128] on @L1 : memref<32x32x128xf32>
              %31 = loom.init_tensor %30[32, 32, 128] : memref<32x32x128xf32> -> tensor<32x32x128xf32>
              %32 = loom.alloc [32, 32, 128] on @L1 : memref<32x32x128xf32>
              %33 = loom.init_tensor %32[32, 32, 128] : memref<32x32x128xf32> -> tensor<32x32x128xf32>
              %34 = arith.muli %12, %c32 : index
              %35 = arith.muli %13, %c32 : index
              %c0 = arith.constant 0 : index
              %36 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 524288 + d1 * 128 + d2)>(%34, %35, %c0)
              %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%36], sizes: [32, 32, 128], strides: [524288, 128, 1] : memref<32x4096x128xf32> to memref<32x32x128xf32, strided<[524288, 128, 1], offset: ?>>
              %37 = loom.copy_to_tensor %reinterpret_cast, %28 on @L1, interconnect : [], broadcast : [1, 1] : memref<32x32x128xf32, strided<[524288, 128, 1], offset: ?>>, memref<32x32x128xf32> -> tensor<32x32x128xf32>
              %38 = linalg.fill ins(%cst_1 : f32) outs(%31 : tensor<32x32x128xf32>) -> tensor<32x32x128xf32>
              %39 = linalg.fill ins(%cst_2 : f32) outs(%19 : tensor<32x32xf32>) -> tensor<32x32xf32>
              %40 = linalg.fill ins(%cst_3 : f32) outs(%21 : tensor<32x32xf32>) -> tensor<32x32xf32>
              %41:3 = affine.for %arg8 = 0 to 128 iter_args(%arg9 = %40, %arg10 = %39, %arg11 = %38) -> (tensor<32x32xf32>, tensor<32x32xf32>, tensor<32x32x128xf32>) {
                %44 = arith.muli %arg8, %c32 : index
                %c0_6 = arith.constant 0 : index
                %45 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 524288 + d1 * 4096 + d2)>(%34, %c0_6, %44)
                %reinterpret_cast_7 = memref.reinterpret_cast %arg0 to offset: [%45], sizes: [32, 128, 32], strides: [524288, 4096, 1] : memref<32x128x4096xf32> to memref<32x128x32xf32, strided<[524288, 4096, 1], offset: ?>>
                %46 = loom.copy_to_tensor %reinterpret_cast_7, %17 on @L1, interconnect : [], broadcast : [1, 1] : memref<32x128x32xf32, strided<[524288, 4096, 1], offset: ?>>, memref<32x128x32xf32> -> tensor<32x128x32xf32>
                %47 = linalg.fill ins(%cst_1 : f32) outs(%16 : tensor<32x32x32xf32>) -> tensor<32x32x32xf32>
                %48 = linalg.batch_matmul ins(%37, %46 : tensor<32x32x128xf32>, tensor<32x128x32xf32>) outs(%47 : tensor<32x32x32xf32>) -> tensor<32x32x32xf32>
                %49 = linalg.fill ins(%cst_3 : f32) outs(%23 : tensor<32x32xf32>) -> tensor<32x32xf32>
                %50 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%48 : tensor<32x32x32xf32>) outs(%49 : tensor<32x32xf32>) {
                ^bb0(%in: f32, %out: f32):
                  %63 = arith.maximumf %in, %out : f32
                  linalg.yield %63 : f32
                } -> tensor<32x32xf32>
                %51 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%arg9, %50 : tensor<32x32xf32>, tensor<32x32xf32>) outs(%23 : tensor<32x32xf32>) {
                ^bb0(%in: f32, %in_10: f32, %out: f32):
                  %63 = arith.truncf %cst_0 : f64 to f32
                  %64 = arith.mulf %in_10, %63 : f32
                  %65 = arith.cmpf ogt, %in, %64 : f32
                  %66 = arith.select %65, %in, %64 : f32
                  linalg.yield %66 : f32
                } -> tensor<32x32xf32>
                %52 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%48, %51 : tensor<32x32x32xf32>, tensor<32x32xf32>) outs(%16 : tensor<32x32x32xf32>) {
                ^bb0(%in: f32, %in_10: f32, %out: f32):
                  %63 = arith.truncf %cst_0 : f64 to f32
                  %64 = arith.mulf %in, %63 : f32
                  %65 = arith.subf %64, %in_10 : f32
                  %66 = math.powf %cst, %65 : f32
                  linalg.yield %66 : f32
                } -> tensor<32x32x32xf32>
                %53 = linalg.fill ins(%cst_1 : f32) outs(%25 : tensor<32x32xf32>) -> tensor<32x32xf32>
                %54 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%52 : tensor<32x32x32xf32>) outs(%53 : tensor<32x32xf32>) {
                ^bb0(%in: f32, %out: f32):
                  %63 = arith.addf %in, %out : f32
                  linalg.yield %63 : f32
                } -> tensor<32x32xf32>
                %55 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%arg9, %51 : tensor<32x32xf32>, tensor<32x32xf32>) outs(%27 : tensor<32x32xf32>) {
                ^bb0(%in: f32, %in_10: f32, %out: f32):
                  %63 = arith.subf %in, %in_10 : f32
                  %64 = math.powf %cst, %63 : f32
                  linalg.yield %64 : f32
                } -> tensor<32x32xf32>
                %56 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%arg10, %55, %54 : tensor<32x32xf32>, tensor<32x32xf32>, tensor<32x32xf32>) outs(%arg10 : tensor<32x32xf32>) {
                ^bb0(%in: f32, %in_10: f32, %in_11: f32, %out: f32):
                  %63 = arith.mulf %in, %in_10 : f32
                  %64 = arith.addf %63, %in_11 : f32
                  linalg.yield %64 : f32
                } -> tensor<32x32xf32>
                %c0_8 = arith.constant 0 : index
                %57 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 524288 + d1 * 128 + d2)>(%34, %44, %c0_8)
                %reinterpret_cast_9 = memref.reinterpret_cast %arg1 to offset: [%57], sizes: [32, 32, 128], strides: [524288, 128, 1] : memref<32x4096x128xf32> to memref<32x32x128xf32, strided<[524288, 128, 1], offset: ?>>
                %58 = loom.copy_to_tensor %reinterpret_cast_9, %14 on @L1, interconnect : [], broadcast : [1, 1] : memref<32x32x128xf32, strided<[524288, 128, 1], offset: ?>>, memref<32x32x128xf32> -> tensor<32x32x128xf32>
                %59 = linalg.fill ins(%cst_1 : f32) outs(%33 : tensor<32x32x128xf32>) -> tensor<32x32x128xf32>
                %60 = linalg.batch_matmul ins(%52, %58 : tensor<32x32x32xf32>, tensor<32x32x128xf32>) outs(%59 : tensor<32x32x128xf32>) -> tensor<32x32x128xf32>
                %61 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%60, %arg11, %55 : tensor<32x32x128xf32>, tensor<32x32x128xf32>, tensor<32x32xf32>) outs(%arg11 : tensor<32x32x128xf32>) {
                ^bb0(%in: f32, %in_10: f32, %in_11: f32, %out: f32):
                  %63 = arith.mulf %in_10, %in_11 : f32
                  %64 = arith.addf %in, %63 : f32
                  linalg.yield %64 : f32
                } -> tensor<32x32x128xf32>
                %62 = linalg.copy ins(%51 : tensor<32x32xf32>) outs(%arg9 : tensor<32x32xf32>) -> tensor<32x32xf32>
                affine.yield %62, %56, %61 : tensor<32x32xf32>, tensor<32x32xf32>, tensor<32x32x128xf32>
              }
              %42 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%41#2, %41#1 : tensor<32x32x128xf32>, tensor<32x32xf32>) outs(%29 : tensor<32x32x128xf32>) {
              ^bb0(%in: f32, %in_6: f32, %out: f32):
                %44 = arith.divf %in, %in_6 : f32
                linalg.yield %44 : f32
              } -> tensor<32x32x128xf32>
              %c0_4 = arith.constant 0 : index
              %43 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 524288 + d1 * 128 + d2)>(%34, %35, %c0_4)
              %reinterpret_cast_5 = memref.reinterpret_cast %arg3 to offset: [%43], sizes: [32, 32, 128], strides: [524288, 128, 1] : memref<32x4096x128xf32> to memref<32x32x128xf32, strided<[524288, 128, 1], offset: ?>>
              loom.copy_from_tensor %42, %reinterpret_cast_5 on @DRAM : tensor<32x32x128xf32>, memref<32x32x128xf32, strided<[524288, 128, 1], offset: ?>>
            } {loom.iter_type = #loom.iter_type<temporal>}
          } {loom.iter_type = #loom.iter_type<temporal>}
        } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @y}
      } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @x}
      return
    }
  }
  module attributes {loom.pass_name = "Materialize"} {
    func.func @attention__d0i0_d1i1__f01__d_d_v__BB32__BM32__BN32(%arg0: memref<32x128x4096xf32>, %arg1: memref<32x4096x128xf32>, %arg2: memref<32x4096x128xf32>, %arg3: memref<32x4096x128xf32>) {
      %cst = arith.constant 2.000000e+00 : f32
      %cst_0 = arith.constant 0.12751743074602467 : f64
      %cst_1 = arith.constant 0.000000e+00 : f32
      %cst_2 = arith.constant 1.000000e+00 : f32
      %cst_3 = arith.constant 0xFF800000 : f32
      %c32 = arith.constant 32 : index
      affine.parallel (%arg4) = (0) to (8) {
        affine.parallel (%arg5) = (0) to (8) {
          affine.for %arg6 = 0 to 1 {
            affine.for %arg7 = 0 to 16 {
              %12 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 8)>(%arg4, %arg6)
              %13 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 8)>(%arg5, %arg7)
              %14 = loom.alloc [32, 32, 128] on @L1 : memref<32x32x128xf32>
              %15 = loom.alloc [32, 32, 32] on @L1 : memref<32x32x32xf32>
              %16 = loom.init_tensor %15[32, 32, 32] : memref<32x32x32xf32> -> tensor<32x32x32xf32>
              %17 = loom.alloc [32, 128, 32] on @L1 : memref<32x128x32xf32>
              %18 = loom.alloc [32, 32] on @L1 : memref<32x32xf32>
              %19 = loom.init_tensor %18[32, 32] : memref<32x32xf32> -> tensor<32x32xf32>
              %20 = loom.alloc [32, 32] on @L1 : memref<32x32xf32>
              %21 = loom.init_tensor %20[32, 32] : memref<32x32xf32> -> tensor<32x32xf32>
              %22 = loom.alloc [32, 32] on @L1 : memref<32x32xf32>
              %23 = loom.init_tensor %22[32, 32] : memref<32x32xf32> -> tensor<32x32xf32>
              %24 = loom.alloc [32, 32] on @L1 : memref<32x32xf32>
              %25 = loom.init_tensor %24[32, 32] : memref<32x32xf32> -> tensor<32x32xf32>
              %26 = loom.alloc [32, 32] on @L1 : memref<32x32xf32>
              %27 = loom.init_tensor %26[32, 32] : memref<32x32xf32> -> tensor<32x32xf32>
              %28 = loom.alloc [32, 32, 128] on @L1 : memref<32x32x128xf32>
              %29 = loom.init_tensor %28[32, 32, 128] : memref<32x32x128xf32> -> tensor<32x32x128xf32>
              %30 = loom.alloc [32, 32, 128] on @L1 : memref<32x32x128xf32>
              %31 = loom.init_tensor %30[32, 32, 128] : memref<32x32x128xf32> -> tensor<32x32x128xf32>
              %32 = loom.alloc [32, 32, 128] on @L1 : memref<32x32x128xf32>
              %33 = loom.init_tensor %32[32, 32, 128] : memref<32x32x128xf32> -> tensor<32x32x128xf32>
              %34 = arith.muli %12, %c32 : index
              %35 = arith.muli %13, %c32 : index
              %c0 = arith.constant 0 : index
              %36 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 524288 + d1 * 128 + d2)>(%34, %35, %c0)
              %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%36], sizes: [32, 32, 128], strides: [524288, 128, 1] : memref<32x4096x128xf32> to memref<32x32x128xf32, strided<[524288, 128, 1], offset: ?>>
              %37 = loom.copy_to_tensor %reinterpret_cast, %28 on @L1, interconnect : [], broadcast : [1, 1] : memref<32x32x128xf32, strided<[524288, 128, 1], offset: ?>>, memref<32x32x128xf32> -> tensor<32x32x128xf32>
              %38 = linalg.fill ins(%cst_1 : f32) outs(%31 : tensor<32x32x128xf32>) -> tensor<32x32x128xf32>
              %39 = linalg.fill ins(%cst_2 : f32) outs(%19 : tensor<32x32xf32>) -> tensor<32x32xf32>
              %40 = linalg.fill ins(%cst_3 : f32) outs(%21 : tensor<32x32xf32>) -> tensor<32x32xf32>
              %41:3 = affine.for %arg8 = 0 to 128 iter_args(%arg9 = %40, %arg10 = %39, %arg11 = %38) -> (tensor<32x32xf32>, tensor<32x32xf32>, tensor<32x32x128xf32>) {
                %44 = arith.muli %arg8, %c32 : index
                %c0_6 = arith.constant 0 : index
                %45 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 524288 + d1 * 4096 + d2)>(%34, %c0_6, %44)
                %reinterpret_cast_7 = memref.reinterpret_cast %arg0 to offset: [%45], sizes: [32, 128, 32], strides: [524288, 4096, 1] : memref<32x128x4096xf32> to memref<32x128x32xf32, strided<[524288, 4096, 1], offset: ?>>
                %46 = loom.copy_to_tensor %reinterpret_cast_7, %17 on @L1, interconnect : [], broadcast : [1, 1] : memref<32x128x32xf32, strided<[524288, 4096, 1], offset: ?>>, memref<32x128x32xf32> -> tensor<32x128x32xf32>
                %47 = linalg.fill ins(%cst_1 : f32) outs(%16 : tensor<32x32x32xf32>) -> tensor<32x32x32xf32>
                %48 = linalg.batch_matmul ins(%37, %46 : tensor<32x32x128xf32>, tensor<32x128x32xf32>) outs(%47 : tensor<32x32x32xf32>) -> tensor<32x32x32xf32>
                %49 = linalg.fill ins(%cst_3 : f32) outs(%23 : tensor<32x32xf32>) -> tensor<32x32xf32>
                %50 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%48 : tensor<32x32x32xf32>) outs(%49 : tensor<32x32xf32>) {
                ^bb0(%in: f32, %out: f32):
                  %63 = arith.maximumf %in, %out : f32
                  linalg.yield %63 : f32
                } -> tensor<32x32xf32>
                %51 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%arg9, %50 : tensor<32x32xf32>, tensor<32x32xf32>) outs(%23 : tensor<32x32xf32>) {
                ^bb0(%in: f32, %in_10: f32, %out: f32):
                  %63 = arith.truncf %cst_0 : f64 to f32
                  %64 = arith.mulf %in_10, %63 : f32
                  %65 = arith.cmpf ogt, %in, %64 : f32
                  %66 = arith.select %65, %in, %64 : f32
                  linalg.yield %66 : f32
                } -> tensor<32x32xf32>
                %52 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%48, %51 : tensor<32x32x32xf32>, tensor<32x32xf32>) outs(%16 : tensor<32x32x32xf32>) {
                ^bb0(%in: f32, %in_10: f32, %out: f32):
                  %63 = arith.truncf %cst_0 : f64 to f32
                  %64 = arith.mulf %in, %63 : f32
                  %65 = arith.subf %64, %in_10 : f32
                  %66 = math.powf %cst, %65 : f32
                  linalg.yield %66 : f32
                } -> tensor<32x32x32xf32>
                %53 = linalg.fill ins(%cst_1 : f32) outs(%25 : tensor<32x32xf32>) -> tensor<32x32xf32>
                %54 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%52 : tensor<32x32x32xf32>) outs(%53 : tensor<32x32xf32>) {
                ^bb0(%in: f32, %out: f32):
                  %63 = arith.addf %in, %out : f32
                  linalg.yield %63 : f32
                } -> tensor<32x32xf32>
                %55 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%arg9, %51 : tensor<32x32xf32>, tensor<32x32xf32>) outs(%27 : tensor<32x32xf32>) {
                ^bb0(%in: f32, %in_10: f32, %out: f32):
                  %63 = arith.subf %in, %in_10 : f32
                  %64 = math.powf %cst, %63 : f32
                  linalg.yield %64 : f32
                } -> tensor<32x32xf32>
                %56 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%arg10, %55, %54 : tensor<32x32xf32>, tensor<32x32xf32>, tensor<32x32xf32>) outs(%arg10 : tensor<32x32xf32>) {
                ^bb0(%in: f32, %in_10: f32, %in_11: f32, %out: f32):
                  %63 = arith.mulf %in, %in_10 : f32
                  %64 = arith.addf %63, %in_11 : f32
                  linalg.yield %64 : f32
                } -> tensor<32x32xf32>
                %c0_8 = arith.constant 0 : index
                %57 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 524288 + d1 * 128 + d2)>(%34, %44, %c0_8)
                %reinterpret_cast_9 = memref.reinterpret_cast %arg1 to offset: [%57], sizes: [32, 32, 128], strides: [524288, 128, 1] : memref<32x4096x128xf32> to memref<32x32x128xf32, strided<[524288, 128, 1], offset: ?>>
                %58 = loom.copy_to_tensor %reinterpret_cast_9, %14 on @L1, interconnect : [@vertical_links], broadcast : [8, 1] : memref<32x32x128xf32, strided<[524288, 128, 1], offset: ?>>, memref<32x32x128xf32> -> tensor<32x32x128xf32>
                %59 = linalg.fill ins(%cst_1 : f32) outs(%33 : tensor<32x32x128xf32>) -> tensor<32x32x128xf32>
                %60 = linalg.batch_matmul ins(%52, %58 : tensor<32x32x32xf32>, tensor<32x32x128xf32>) outs(%59 : tensor<32x32x128xf32>) -> tensor<32x32x128xf32>
                %61 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%60, %arg11, %55 : tensor<32x32x128xf32>, tensor<32x32x128xf32>, tensor<32x32xf32>) outs(%arg11 : tensor<32x32x128xf32>) {
                ^bb0(%in: f32, %in_10: f32, %in_11: f32, %out: f32):
                  %63 = arith.mulf %in_10, %in_11 : f32
                  %64 = arith.addf %in, %63 : f32
                  linalg.yield %64 : f32
                } -> tensor<32x32x128xf32>
                %62 = linalg.copy ins(%51 : tensor<32x32xf32>) outs(%arg9 : tensor<32x32xf32>) -> tensor<32x32xf32>
                affine.yield %62, %56, %61 : tensor<32x32xf32>, tensor<32x32xf32>, tensor<32x32x128xf32>
              }
              %42 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%41#2, %41#1 : tensor<32x32x128xf32>, tensor<32x32xf32>) outs(%29 : tensor<32x32x128xf32>) {
              ^bb0(%in: f32, %in_6: f32, %out: f32):
                %44 = arith.divf %in, %in_6 : f32
                linalg.yield %44 : f32
              } -> tensor<32x32x128xf32>
              %c0_4 = arith.constant 0 : index
              %43 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 524288 + d1 * 128 + d2)>(%34, %35, %c0_4)
              %reinterpret_cast_5 = memref.reinterpret_cast %arg3 to offset: [%43], sizes: [32, 32, 128], strides: [524288, 128, 1] : memref<32x4096x128xf32> to memref<32x32x128xf32, strided<[524288, 128, 1], offset: ?>>
              loom.copy_from_tensor %42, %reinterpret_cast_5 on @DRAM : tensor<32x32x128xf32>, memref<32x32x128xf32, strided<[524288, 128, 1], offset: ?>>
            } {loom.iter_type = #loom.iter_type<temporal>}
          } {loom.iter_type = #loom.iter_type<temporal>}
        } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @y}
      } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @x}
      return
    }
  }
  module attributes {loom.pass_name = "Materialize"} {
    func.func @attention__d0i0_d1i1__f01__d_v_d__BB32__BM32__BN32(%arg0: memref<32x128x4096xf32>, %arg1: memref<32x4096x128xf32>, %arg2: memref<32x4096x128xf32>, %arg3: memref<32x4096x128xf32>) {
      %cst = arith.constant 2.000000e+00 : f32
      %cst_0 = arith.constant 0.12751743074602467 : f64
      %cst_1 = arith.constant 0.000000e+00 : f32
      %cst_2 = arith.constant 1.000000e+00 : f32
      %cst_3 = arith.constant 0xFF800000 : f32
      %c32 = arith.constant 32 : index
      affine.parallel (%arg4) = (0) to (8) {
        affine.parallel (%arg5) = (0) to (8) {
          affine.for %arg6 = 0 to 1 {
            affine.for %arg7 = 0 to 16 {
              %12 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 8)>(%arg4, %arg6)
              %13 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 8)>(%arg5, %arg7)
              %14 = loom.alloc [32, 32, 128] on @L1 : memref<32x32x128xf32>
              %15 = loom.alloc [32, 32, 32] on @L1 : memref<32x32x32xf32>
              %16 = loom.init_tensor %15[32, 32, 32] : memref<32x32x32xf32> -> tensor<32x32x32xf32>
              %17 = loom.alloc [32, 128, 32] on @L1 : memref<32x128x32xf32>
              %18 = loom.alloc [32, 32] on @L1 : memref<32x32xf32>
              %19 = loom.init_tensor %18[32, 32] : memref<32x32xf32> -> tensor<32x32xf32>
              %20 = loom.alloc [32, 32] on @L1 : memref<32x32xf32>
              %21 = loom.init_tensor %20[32, 32] : memref<32x32xf32> -> tensor<32x32xf32>
              %22 = loom.alloc [32, 32] on @L1 : memref<32x32xf32>
              %23 = loom.init_tensor %22[32, 32] : memref<32x32xf32> -> tensor<32x32xf32>
              %24 = loom.alloc [32, 32] on @L1 : memref<32x32xf32>
              %25 = loom.init_tensor %24[32, 32] : memref<32x32xf32> -> tensor<32x32xf32>
              %26 = loom.alloc [32, 32] on @L1 : memref<32x32xf32>
              %27 = loom.init_tensor %26[32, 32] : memref<32x32xf32> -> tensor<32x32xf32>
              %28 = loom.alloc [32, 32, 128] on @L1 : memref<32x32x128xf32>
              %29 = loom.init_tensor %28[32, 32, 128] : memref<32x32x128xf32> -> tensor<32x32x128xf32>
              %30 = loom.alloc [32, 32, 128] on @L1 : memref<32x32x128xf32>
              %31 = loom.init_tensor %30[32, 32, 128] : memref<32x32x128xf32> -> tensor<32x32x128xf32>
              %32 = loom.alloc [32, 32, 128] on @L1 : memref<32x32x128xf32>
              %33 = loom.init_tensor %32[32, 32, 128] : memref<32x32x128xf32> -> tensor<32x32x128xf32>
              %34 = arith.muli %12, %c32 : index
              %35 = arith.muli %13, %c32 : index
              %c0 = arith.constant 0 : index
              %36 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 524288 + d1 * 128 + d2)>(%34, %35, %c0)
              %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%36], sizes: [32, 32, 128], strides: [524288, 128, 1] : memref<32x4096x128xf32> to memref<32x32x128xf32, strided<[524288, 128, 1], offset: ?>>
              %37 = loom.copy_to_tensor %reinterpret_cast, %28 on @L1, interconnect : [], broadcast : [1, 1] : memref<32x32x128xf32, strided<[524288, 128, 1], offset: ?>>, memref<32x32x128xf32> -> tensor<32x32x128xf32>
              %38 = linalg.fill ins(%cst_1 : f32) outs(%31 : tensor<32x32x128xf32>) -> tensor<32x32x128xf32>
              %39 = linalg.fill ins(%cst_2 : f32) outs(%19 : tensor<32x32xf32>) -> tensor<32x32xf32>
              %40 = linalg.fill ins(%cst_3 : f32) outs(%21 : tensor<32x32xf32>) -> tensor<32x32xf32>
              %41:3 = affine.for %arg8 = 0 to 128 iter_args(%arg9 = %40, %arg10 = %39, %arg11 = %38) -> (tensor<32x32xf32>, tensor<32x32xf32>, tensor<32x32x128xf32>) {
                %44 = arith.muli %arg8, %c32 : index
                %c0_6 = arith.constant 0 : index
                %45 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 524288 + d1 * 4096 + d2)>(%34, %c0_6, %44)
                %reinterpret_cast_7 = memref.reinterpret_cast %arg0 to offset: [%45], sizes: [32, 128, 32], strides: [524288, 4096, 1] : memref<32x128x4096xf32> to memref<32x128x32xf32, strided<[524288, 4096, 1], offset: ?>>
                %46 = loom.copy_to_tensor %reinterpret_cast_7, %17 on @L1, interconnect : [@vertical_links], broadcast : [8, 1] : memref<32x128x32xf32, strided<[524288, 4096, 1], offset: ?>>, memref<32x128x32xf32> -> tensor<32x128x32xf32>
                %47 = linalg.fill ins(%cst_1 : f32) outs(%16 : tensor<32x32x32xf32>) -> tensor<32x32x32xf32>
                %48 = linalg.batch_matmul ins(%37, %46 : tensor<32x32x128xf32>, tensor<32x128x32xf32>) outs(%47 : tensor<32x32x32xf32>) -> tensor<32x32x32xf32>
                %49 = linalg.fill ins(%cst_3 : f32) outs(%23 : tensor<32x32xf32>) -> tensor<32x32xf32>
                %50 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%48 : tensor<32x32x32xf32>) outs(%49 : tensor<32x32xf32>) {
                ^bb0(%in: f32, %out: f32):
                  %63 = arith.maximumf %in, %out : f32
                  linalg.yield %63 : f32
                } -> tensor<32x32xf32>
                %51 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%arg9, %50 : tensor<32x32xf32>, tensor<32x32xf32>) outs(%23 : tensor<32x32xf32>) {
                ^bb0(%in: f32, %in_10: f32, %out: f32):
                  %63 = arith.truncf %cst_0 : f64 to f32
                  %64 = arith.mulf %in_10, %63 : f32
                  %65 = arith.cmpf ogt, %in, %64 : f32
                  %66 = arith.select %65, %in, %64 : f32
                  linalg.yield %66 : f32
                } -> tensor<32x32xf32>
                %52 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%48, %51 : tensor<32x32x32xf32>, tensor<32x32xf32>) outs(%16 : tensor<32x32x32xf32>) {
                ^bb0(%in: f32, %in_10: f32, %out: f32):
                  %63 = arith.truncf %cst_0 : f64 to f32
                  %64 = arith.mulf %in, %63 : f32
                  %65 = arith.subf %64, %in_10 : f32
                  %66 = math.powf %cst, %65 : f32
                  linalg.yield %66 : f32
                } -> tensor<32x32x32xf32>
                %53 = linalg.fill ins(%cst_1 : f32) outs(%25 : tensor<32x32xf32>) -> tensor<32x32xf32>
                %54 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%52 : tensor<32x32x32xf32>) outs(%53 : tensor<32x32xf32>) {
                ^bb0(%in: f32, %out: f32):
                  %63 = arith.addf %in, %out : f32
                  linalg.yield %63 : f32
                } -> tensor<32x32xf32>
                %55 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%arg9, %51 : tensor<32x32xf32>, tensor<32x32xf32>) outs(%27 : tensor<32x32xf32>) {
                ^bb0(%in: f32, %in_10: f32, %out: f32):
                  %63 = arith.subf %in, %in_10 : f32
                  %64 = math.powf %cst, %63 : f32
                  linalg.yield %64 : f32
                } -> tensor<32x32xf32>
                %56 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%arg10, %55, %54 : tensor<32x32xf32>, tensor<32x32xf32>, tensor<32x32xf32>) outs(%arg10 : tensor<32x32xf32>) {
                ^bb0(%in: f32, %in_10: f32, %in_11: f32, %out: f32):
                  %63 = arith.mulf %in, %in_10 : f32
                  %64 = arith.addf %63, %in_11 : f32
                  linalg.yield %64 : f32
                } -> tensor<32x32xf32>
                %c0_8 = arith.constant 0 : index
                %57 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 524288 + d1 * 128 + d2)>(%34, %44, %c0_8)
                %reinterpret_cast_9 = memref.reinterpret_cast %arg1 to offset: [%57], sizes: [32, 32, 128], strides: [524288, 128, 1] : memref<32x4096x128xf32> to memref<32x32x128xf32, strided<[524288, 128, 1], offset: ?>>
                %58 = loom.copy_to_tensor %reinterpret_cast_9, %14 on @L1, interconnect : [], broadcast : [1, 1] : memref<32x32x128xf32, strided<[524288, 128, 1], offset: ?>>, memref<32x32x128xf32> -> tensor<32x32x128xf32>
                %59 = linalg.fill ins(%cst_1 : f32) outs(%33 : tensor<32x32x128xf32>) -> tensor<32x32x128xf32>
                %60 = linalg.batch_matmul ins(%52, %58 : tensor<32x32x32xf32>, tensor<32x32x128xf32>) outs(%59 : tensor<32x32x128xf32>) -> tensor<32x32x128xf32>
                %61 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%60, %arg11, %55 : tensor<32x32x128xf32>, tensor<32x32x128xf32>, tensor<32x32xf32>) outs(%arg11 : tensor<32x32x128xf32>) {
                ^bb0(%in: f32, %in_10: f32, %in_11: f32, %out: f32):
                  %63 = arith.mulf %in_10, %in_11 : f32
                  %64 = arith.addf %in, %63 : f32
                  linalg.yield %64 : f32
                } -> tensor<32x32x128xf32>
                %62 = linalg.copy ins(%51 : tensor<32x32xf32>) outs(%arg9 : tensor<32x32xf32>) -> tensor<32x32xf32>
                affine.yield %62, %56, %61 : tensor<32x32xf32>, tensor<32x32xf32>, tensor<32x32x128xf32>
              }
              %42 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%41#2, %41#1 : tensor<32x32x128xf32>, tensor<32x32xf32>) outs(%29 : tensor<32x32x128xf32>) {
              ^bb0(%in: f32, %in_6: f32, %out: f32):
                %44 = arith.divf %in, %in_6 : f32
                linalg.yield %44 : f32
              } -> tensor<32x32x128xf32>
              %c0_4 = arith.constant 0 : index
              %43 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 524288 + d1 * 128 + d2)>(%34, %35, %c0_4)
              %reinterpret_cast_5 = memref.reinterpret_cast %arg3 to offset: [%43], sizes: [32, 32, 128], strides: [524288, 128, 1] : memref<32x4096x128xf32> to memref<32x32x128xf32, strided<[524288, 128, 1], offset: ?>>
              loom.copy_from_tensor %42, %reinterpret_cast_5 on @DRAM : tensor<32x32x128xf32>, memref<32x32x128xf32, strided<[524288, 128, 1], offset: ?>>
            } {loom.iter_type = #loom.iter_type<temporal>}
          } {loom.iter_type = #loom.iter_type<temporal>}
        } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @y}
      } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @x}
      return
    }
  }
  module attributes {loom.pass_name = "Materialize"} {
    func.func @attention__d0i0_d1i1__f01__d_v_v__BB32__BM32__BN32(%arg0: memref<32x128x4096xf32>, %arg1: memref<32x4096x128xf32>, %arg2: memref<32x4096x128xf32>, %arg3: memref<32x4096x128xf32>) {
      %cst = arith.constant 2.000000e+00 : f32
      %cst_0 = arith.constant 0.12751743074602467 : f64
      %cst_1 = arith.constant 0.000000e+00 : f32
      %cst_2 = arith.constant 1.000000e+00 : f32
      %cst_3 = arith.constant 0xFF800000 : f32
      %c32 = arith.constant 32 : index
      affine.parallel (%arg4) = (0) to (8) {
        affine.parallel (%arg5) = (0) to (8) {
          affine.for %arg6 = 0 to 1 {
            affine.for %arg7 = 0 to 16 {
              %12 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 8)>(%arg4, %arg6)
              %13 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 8)>(%arg5, %arg7)
              %14 = loom.alloc [32, 32, 128] on @L1 : memref<32x32x128xf32>
              %15 = loom.alloc [32, 32, 32] on @L1 : memref<32x32x32xf32>
              %16 = loom.init_tensor %15[32, 32, 32] : memref<32x32x32xf32> -> tensor<32x32x32xf32>
              %17 = loom.alloc [32, 128, 32] on @L1 : memref<32x128x32xf32>
              %18 = loom.alloc [32, 32] on @L1 : memref<32x32xf32>
              %19 = loom.init_tensor %18[32, 32] : memref<32x32xf32> -> tensor<32x32xf32>
              %20 = loom.alloc [32, 32] on @L1 : memref<32x32xf32>
              %21 = loom.init_tensor %20[32, 32] : memref<32x32xf32> -> tensor<32x32xf32>
              %22 = loom.alloc [32, 32] on @L1 : memref<32x32xf32>
              %23 = loom.init_tensor %22[32, 32] : memref<32x32xf32> -> tensor<32x32xf32>
              %24 = loom.alloc [32, 32] on @L1 : memref<32x32xf32>
              %25 = loom.init_tensor %24[32, 32] : memref<32x32xf32> -> tensor<32x32xf32>
              %26 = loom.alloc [32, 32] on @L1 : memref<32x32xf32>
              %27 = loom.init_tensor %26[32, 32] : memref<32x32xf32> -> tensor<32x32xf32>
              %28 = loom.alloc [32, 32, 128] on @L1 : memref<32x32x128xf32>
              %29 = loom.init_tensor %28[32, 32, 128] : memref<32x32x128xf32> -> tensor<32x32x128xf32>
              %30 = loom.alloc [32, 32, 128] on @L1 : memref<32x32x128xf32>
              %31 = loom.init_tensor %30[32, 32, 128] : memref<32x32x128xf32> -> tensor<32x32x128xf32>
              %32 = loom.alloc [32, 32, 128] on @L1 : memref<32x32x128xf32>
              %33 = loom.init_tensor %32[32, 32, 128] : memref<32x32x128xf32> -> tensor<32x32x128xf32>
              %34 = arith.muli %12, %c32 : index
              %35 = arith.muli %13, %c32 : index
              %c0 = arith.constant 0 : index
              %36 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 524288 + d1 * 128 + d2)>(%34, %35, %c0)
              %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%36], sizes: [32, 32, 128], strides: [524288, 128, 1] : memref<32x4096x128xf32> to memref<32x32x128xf32, strided<[524288, 128, 1], offset: ?>>
              %37 = loom.copy_to_tensor %reinterpret_cast, %28 on @L1, interconnect : [], broadcast : [1, 1] : memref<32x32x128xf32, strided<[524288, 128, 1], offset: ?>>, memref<32x32x128xf32> -> tensor<32x32x128xf32>
              %38 = linalg.fill ins(%cst_1 : f32) outs(%31 : tensor<32x32x128xf32>) -> tensor<32x32x128xf32>
              %39 = linalg.fill ins(%cst_2 : f32) outs(%19 : tensor<32x32xf32>) -> tensor<32x32xf32>
              %40 = linalg.fill ins(%cst_3 : f32) outs(%21 : tensor<32x32xf32>) -> tensor<32x32xf32>
              %41:3 = affine.for %arg8 = 0 to 128 iter_args(%arg9 = %40, %arg10 = %39, %arg11 = %38) -> (tensor<32x32xf32>, tensor<32x32xf32>, tensor<32x32x128xf32>) {
                %44 = arith.muli %arg8, %c32 : index
                %c0_6 = arith.constant 0 : index
                %45 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 524288 + d1 * 4096 + d2)>(%34, %c0_6, %44)
                %reinterpret_cast_7 = memref.reinterpret_cast %arg0 to offset: [%45], sizes: [32, 128, 32], strides: [524288, 4096, 1] : memref<32x128x4096xf32> to memref<32x128x32xf32, strided<[524288, 4096, 1], offset: ?>>
                %46 = loom.copy_to_tensor %reinterpret_cast_7, %17 on @L1, interconnect : [@vertical_links], broadcast : [8, 1] : memref<32x128x32xf32, strided<[524288, 4096, 1], offset: ?>>, memref<32x128x32xf32> -> tensor<32x128x32xf32>
                %47 = linalg.fill ins(%cst_1 : f32) outs(%16 : tensor<32x32x32xf32>) -> tensor<32x32x32xf32>
                %48 = linalg.batch_matmul ins(%37, %46 : tensor<32x32x128xf32>, tensor<32x128x32xf32>) outs(%47 : tensor<32x32x32xf32>) -> tensor<32x32x32xf32>
                %49 = linalg.fill ins(%cst_3 : f32) outs(%23 : tensor<32x32xf32>) -> tensor<32x32xf32>
                %50 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%48 : tensor<32x32x32xf32>) outs(%49 : tensor<32x32xf32>) {
                ^bb0(%in: f32, %out: f32):
                  %63 = arith.maximumf %in, %out : f32
                  linalg.yield %63 : f32
                } -> tensor<32x32xf32>
                %51 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%arg9, %50 : tensor<32x32xf32>, tensor<32x32xf32>) outs(%23 : tensor<32x32xf32>) {
                ^bb0(%in: f32, %in_10: f32, %out: f32):
                  %63 = arith.truncf %cst_0 : f64 to f32
                  %64 = arith.mulf %in_10, %63 : f32
                  %65 = arith.cmpf ogt, %in, %64 : f32
                  %66 = arith.select %65, %in, %64 : f32
                  linalg.yield %66 : f32
                } -> tensor<32x32xf32>
                %52 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%48, %51 : tensor<32x32x32xf32>, tensor<32x32xf32>) outs(%16 : tensor<32x32x32xf32>) {
                ^bb0(%in: f32, %in_10: f32, %out: f32):
                  %63 = arith.truncf %cst_0 : f64 to f32
                  %64 = arith.mulf %in, %63 : f32
                  %65 = arith.subf %64, %in_10 : f32
                  %66 = math.powf %cst, %65 : f32
                  linalg.yield %66 : f32
                } -> tensor<32x32x32xf32>
                %53 = linalg.fill ins(%cst_1 : f32) outs(%25 : tensor<32x32xf32>) -> tensor<32x32xf32>
                %54 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%52 : tensor<32x32x32xf32>) outs(%53 : tensor<32x32xf32>) {
                ^bb0(%in: f32, %out: f32):
                  %63 = arith.addf %in, %out : f32
                  linalg.yield %63 : f32
                } -> tensor<32x32xf32>
                %55 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%arg9, %51 : tensor<32x32xf32>, tensor<32x32xf32>) outs(%27 : tensor<32x32xf32>) {
                ^bb0(%in: f32, %in_10: f32, %out: f32):
                  %63 = arith.subf %in, %in_10 : f32
                  %64 = math.powf %cst, %63 : f32
                  linalg.yield %64 : f32
                } -> tensor<32x32xf32>
                %56 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%arg10, %55, %54 : tensor<32x32xf32>, tensor<32x32xf32>, tensor<32x32xf32>) outs(%arg10 : tensor<32x32xf32>) {
                ^bb0(%in: f32, %in_10: f32, %in_11: f32, %out: f32):
                  %63 = arith.mulf %in, %in_10 : f32
                  %64 = arith.addf %63, %in_11 : f32
                  linalg.yield %64 : f32
                } -> tensor<32x32xf32>
                %c0_8 = arith.constant 0 : index
                %57 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 524288 + d1 * 128 + d2)>(%34, %44, %c0_8)
                %reinterpret_cast_9 = memref.reinterpret_cast %arg1 to offset: [%57], sizes: [32, 32, 128], strides: [524288, 128, 1] : memref<32x4096x128xf32> to memref<32x32x128xf32, strided<[524288, 128, 1], offset: ?>>
                %58 = loom.copy_to_tensor %reinterpret_cast_9, %14 on @L1, interconnect : [@vertical_links], broadcast : [8, 1] : memref<32x32x128xf32, strided<[524288, 128, 1], offset: ?>>, memref<32x32x128xf32> -> tensor<32x32x128xf32>
                %59 = linalg.fill ins(%cst_1 : f32) outs(%33 : tensor<32x32x128xf32>) -> tensor<32x32x128xf32>
                %60 = linalg.batch_matmul ins(%52, %58 : tensor<32x32x32xf32>, tensor<32x32x128xf32>) outs(%59 : tensor<32x32x128xf32>) -> tensor<32x32x128xf32>
                %61 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%60, %arg11, %55 : tensor<32x32x128xf32>, tensor<32x32x128xf32>, tensor<32x32xf32>) outs(%arg11 : tensor<32x32x128xf32>) {
                ^bb0(%in: f32, %in_10: f32, %in_11: f32, %out: f32):
                  %63 = arith.mulf %in_10, %in_11 : f32
                  %64 = arith.addf %in, %63 : f32
                  linalg.yield %64 : f32
                } -> tensor<32x32x128xf32>
                %62 = linalg.copy ins(%51 : tensor<32x32xf32>) outs(%arg9 : tensor<32x32xf32>) -> tensor<32x32xf32>
                affine.yield %62, %56, %61 : tensor<32x32xf32>, tensor<32x32xf32>, tensor<32x32x128xf32>
              }
              %42 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%41#2, %41#1 : tensor<32x32x128xf32>, tensor<32x32xf32>) outs(%29 : tensor<32x32x128xf32>) {
              ^bb0(%in: f32, %in_6: f32, %out: f32):
                %44 = arith.divf %in, %in_6 : f32
                linalg.yield %44 : f32
              } -> tensor<32x32x128xf32>
              %c0_4 = arith.constant 0 : index
              %43 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 524288 + d1 * 128 + d2)>(%34, %35, %c0_4)
              %reinterpret_cast_5 = memref.reinterpret_cast %arg3 to offset: [%43], sizes: [32, 32, 128], strides: [524288, 128, 1] : memref<32x4096x128xf32> to memref<32x32x128xf32, strided<[524288, 128, 1], offset: ?>>
              loom.copy_from_tensor %42, %reinterpret_cast_5 on @DRAM : tensor<32x32x128xf32>, memref<32x32x128xf32, strided<[524288, 128, 1], offset: ?>>
            } {loom.iter_type = #loom.iter_type<temporal>}
          } {loom.iter_type = #loom.iter_type<temporal>}
        } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @y}
      } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @x}
      return
    }
  }
  module attributes {loom.pass_name = "Materialize"} {
    func.func @attention__d0i0_d1i1__f10__d_d_d__BB32__BM32__BN32(%arg0: memref<32x128x4096xf32>, %arg1: memref<32x4096x128xf32>, %arg2: memref<32x4096x128xf32>, %arg3: memref<32x4096x128xf32>) {
      %cst = arith.constant 2.000000e+00 : f32
      %cst_0 = arith.constant 0.12751743074602467 : f64
      %cst_1 = arith.constant 0.000000e+00 : f32
      %cst_2 = arith.constant 1.000000e+00 : f32
      %cst_3 = arith.constant 0xFF800000 : f32
      %c32 = arith.constant 32 : index
      affine.parallel (%arg4) = (0) to (8) {
        affine.parallel (%arg5) = (0) to (8) {
          affine.for %arg6 = 0 to 16 {
            affine.for %arg7 = 0 to 1 {
              %12 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 8)>(%arg4, %arg7)
              %13 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 8)>(%arg5, %arg6)
              %14 = loom.alloc [32, 32, 128] on @L1 : memref<32x32x128xf32>
              %15 = loom.alloc [32, 32, 32] on @L1 : memref<32x32x32xf32>
              %16 = loom.init_tensor %15[32, 32, 32] : memref<32x32x32xf32> -> tensor<32x32x32xf32>
              %17 = loom.alloc [32, 128, 32] on @L1 : memref<32x128x32xf32>
              %18 = loom.alloc [32, 32] on @L1 : memref<32x32xf32>
              %19 = loom.init_tensor %18[32, 32] : memref<32x32xf32> -> tensor<32x32xf32>
              %20 = loom.alloc [32, 32] on @L1 : memref<32x32xf32>
              %21 = loom.init_tensor %20[32, 32] : memref<32x32xf32> -> tensor<32x32xf32>
              %22 = loom.alloc [32, 32] on @L1 : memref<32x32xf32>
              %23 = loom.init_tensor %22[32, 32] : memref<32x32xf32> -> tensor<32x32xf32>
              %24 = loom.alloc [32, 32] on @L1 : memref<32x32xf32>
              %25 = loom.init_tensor %24[32, 32] : memref<32x32xf32> -> tensor<32x32xf32>
              %26 = loom.alloc [32, 32] on @L1 : memref<32x32xf32>
              %27 = loom.init_tensor %26[32, 32] : memref<32x32xf32> -> tensor<32x32xf32>
              %28 = loom.alloc [32, 32, 128] on @L1 : memref<32x32x128xf32>
              %29 = loom.init_tensor %28[32, 32, 128] : memref<32x32x128xf32> -> tensor<32x32x128xf32>
              %30 = loom.alloc [32, 32, 128] on @L1 : memref<32x32x128xf32>
              %31 = loom.init_tensor %30[32, 32, 128] : memref<32x32x128xf32> -> tensor<32x32x128xf32>
              %32 = loom.alloc [32, 32, 128] on @L1 : memref<32x32x128xf32>
              %33 = loom.init_tensor %32[32, 32, 128] : memref<32x32x128xf32> -> tensor<32x32x128xf32>
              %34 = arith.muli %12, %c32 : index
              %35 = arith.muli %13, %c32 : index
              %c0 = arith.constant 0 : index
              %36 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 524288 + d1 * 128 + d2)>(%34, %35, %c0)
              %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%36], sizes: [32, 32, 128], strides: [524288, 128, 1] : memref<32x4096x128xf32> to memref<32x32x128xf32, strided<[524288, 128, 1], offset: ?>>
              %37 = loom.copy_to_tensor %reinterpret_cast, %28 on @L1, interconnect : [], broadcast : [1, 1] : memref<32x32x128xf32, strided<[524288, 128, 1], offset: ?>>, memref<32x32x128xf32> -> tensor<32x32x128xf32>
              %38 = linalg.fill ins(%cst_1 : f32) outs(%31 : tensor<32x32x128xf32>) -> tensor<32x32x128xf32>
              %39 = linalg.fill ins(%cst_2 : f32) outs(%19 : tensor<32x32xf32>) -> tensor<32x32xf32>
              %40 = linalg.fill ins(%cst_3 : f32) outs(%21 : tensor<32x32xf32>) -> tensor<32x32xf32>
              %41:3 = affine.for %arg8 = 0 to 128 iter_args(%arg9 = %40, %arg10 = %39, %arg11 = %38) -> (tensor<32x32xf32>, tensor<32x32xf32>, tensor<32x32x128xf32>) {
                %44 = arith.muli %arg8, %c32 : index
                %c0_6 = arith.constant 0 : index
                %45 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 524288 + d1 * 4096 + d2)>(%34, %c0_6, %44)
                %reinterpret_cast_7 = memref.reinterpret_cast %arg0 to offset: [%45], sizes: [32, 128, 32], strides: [524288, 4096, 1] : memref<32x128x4096xf32> to memref<32x128x32xf32, strided<[524288, 4096, 1], offset: ?>>
                %46 = loom.copy_to_tensor %reinterpret_cast_7, %17 on @L1, interconnect : [], broadcast : [1, 1] : memref<32x128x32xf32, strided<[524288, 4096, 1], offset: ?>>, memref<32x128x32xf32> -> tensor<32x128x32xf32>
                %47 = linalg.fill ins(%cst_1 : f32) outs(%16 : tensor<32x32x32xf32>) -> tensor<32x32x32xf32>
                %48 = linalg.batch_matmul ins(%37, %46 : tensor<32x32x128xf32>, tensor<32x128x32xf32>) outs(%47 : tensor<32x32x32xf32>) -> tensor<32x32x32xf32>
                %49 = linalg.fill ins(%cst_3 : f32) outs(%23 : tensor<32x32xf32>) -> tensor<32x32xf32>
                %50 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%48 : tensor<32x32x32xf32>) outs(%49 : tensor<32x32xf32>) {
                ^bb0(%in: f32, %out: f32):
                  %63 = arith.maximumf %in, %out : f32
                  linalg.yield %63 : f32
                } -> tensor<32x32xf32>
                %51 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%arg9, %50 : tensor<32x32xf32>, tensor<32x32xf32>) outs(%23 : tensor<32x32xf32>) {
                ^bb0(%in: f32, %in_10: f32, %out: f32):
                  %63 = arith.truncf %cst_0 : f64 to f32
                  %64 = arith.mulf %in_10, %63 : f32
                  %65 = arith.cmpf ogt, %in, %64 : f32
                  %66 = arith.select %65, %in, %64 : f32
                  linalg.yield %66 : f32
                } -> tensor<32x32xf32>
                %52 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%48, %51 : tensor<32x32x32xf32>, tensor<32x32xf32>) outs(%16 : tensor<32x32x32xf32>) {
                ^bb0(%in: f32, %in_10: f32, %out: f32):
                  %63 = arith.truncf %cst_0 : f64 to f32
                  %64 = arith.mulf %in, %63 : f32
                  %65 = arith.subf %64, %in_10 : f32
                  %66 = math.powf %cst, %65 : f32
                  linalg.yield %66 : f32
                } -> tensor<32x32x32xf32>
                %53 = linalg.fill ins(%cst_1 : f32) outs(%25 : tensor<32x32xf32>) -> tensor<32x32xf32>
                %54 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%52 : tensor<32x32x32xf32>) outs(%53 : tensor<32x32xf32>) {
                ^bb0(%in: f32, %out: f32):
                  %63 = arith.addf %in, %out : f32
                  linalg.yield %63 : f32
                } -> tensor<32x32xf32>
                %55 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%arg9, %51 : tensor<32x32xf32>, tensor<32x32xf32>) outs(%27 : tensor<32x32xf32>) {
                ^bb0(%in: f32, %in_10: f32, %out: f32):
                  %63 = arith.subf %in, %in_10 : f32
                  %64 = math.powf %cst, %63 : f32
                  linalg.yield %64 : f32
                } -> tensor<32x32xf32>
                %56 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%arg10, %55, %54 : tensor<32x32xf32>, tensor<32x32xf32>, tensor<32x32xf32>) outs(%arg10 : tensor<32x32xf32>) {
                ^bb0(%in: f32, %in_10: f32, %in_11: f32, %out: f32):
                  %63 = arith.mulf %in, %in_10 : f32
                  %64 = arith.addf %63, %in_11 : f32
                  linalg.yield %64 : f32
                } -> tensor<32x32xf32>
                %c0_8 = arith.constant 0 : index
                %57 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 524288 + d1 * 128 + d2)>(%34, %44, %c0_8)
                %reinterpret_cast_9 = memref.reinterpret_cast %arg1 to offset: [%57], sizes: [32, 32, 128], strides: [524288, 128, 1] : memref<32x4096x128xf32> to memref<32x32x128xf32, strided<[524288, 128, 1], offset: ?>>
                %58 = loom.copy_to_tensor %reinterpret_cast_9, %14 on @L1, interconnect : [], broadcast : [1, 1] : memref<32x32x128xf32, strided<[524288, 128, 1], offset: ?>>, memref<32x32x128xf32> -> tensor<32x32x128xf32>
                %59 = linalg.fill ins(%cst_1 : f32) outs(%33 : tensor<32x32x128xf32>) -> tensor<32x32x128xf32>
                %60 = linalg.batch_matmul ins(%52, %58 : tensor<32x32x32xf32>, tensor<32x32x128xf32>) outs(%59 : tensor<32x32x128xf32>) -> tensor<32x32x128xf32>
                %61 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%60, %arg11, %55 : tensor<32x32x128xf32>, tensor<32x32x128xf32>, tensor<32x32xf32>) outs(%arg11 : tensor<32x32x128xf32>) {
                ^bb0(%in: f32, %in_10: f32, %in_11: f32, %out: f32):
                  %63 = arith.mulf %in_10, %in_11 : f32
                  %64 = arith.addf %in, %63 : f32
                  linalg.yield %64 : f32
                } -> tensor<32x32x128xf32>
                %62 = linalg.copy ins(%51 : tensor<32x32xf32>) outs(%arg9 : tensor<32x32xf32>) -> tensor<32x32xf32>
                affine.yield %62, %56, %61 : tensor<32x32xf32>, tensor<32x32xf32>, tensor<32x32x128xf32>
              }
              %42 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%41#2, %41#1 : tensor<32x32x128xf32>, tensor<32x32xf32>) outs(%29 : tensor<32x32x128xf32>) {
              ^bb0(%in: f32, %in_6: f32, %out: f32):
                %44 = arith.divf %in, %in_6 : f32
                linalg.yield %44 : f32
              } -> tensor<32x32x128xf32>
              %c0_4 = arith.constant 0 : index
              %43 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 524288 + d1 * 128 + d2)>(%34, %35, %c0_4)
              %reinterpret_cast_5 = memref.reinterpret_cast %arg3 to offset: [%43], sizes: [32, 32, 128], strides: [524288, 128, 1] : memref<32x4096x128xf32> to memref<32x32x128xf32, strided<[524288, 128, 1], offset: ?>>
              loom.copy_from_tensor %42, %reinterpret_cast_5 on @DRAM : tensor<32x32x128xf32>, memref<32x32x128xf32, strided<[524288, 128, 1], offset: ?>>
            } {loom.iter_type = #loom.iter_type<temporal>}
          } {loom.iter_type = #loom.iter_type<temporal>}
        } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @y}
      } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @x}
      return
    }
  }
  module attributes {loom.pass_name = "Materialize"} {
    func.func @attention__d0i0_d1i1__f10__d_d_v__BB32__BM32__BN32(%arg0: memref<32x128x4096xf32>, %arg1: memref<32x4096x128xf32>, %arg2: memref<32x4096x128xf32>, %arg3: memref<32x4096x128xf32>) {
      %cst = arith.constant 2.000000e+00 : f32
      %cst_0 = arith.constant 0.12751743074602467 : f64
      %cst_1 = arith.constant 0.000000e+00 : f32
      %cst_2 = arith.constant 1.000000e+00 : f32
      %cst_3 = arith.constant 0xFF800000 : f32
      %c32 = arith.constant 32 : index
      affine.parallel (%arg4) = (0) to (8) {
        affine.parallel (%arg5) = (0) to (8) {
          affine.for %arg6 = 0 to 16 {
            affine.for %arg7 = 0 to 1 {
              %12 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 8)>(%arg4, %arg7)
              %13 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 8)>(%arg5, %arg6)
              %14 = loom.alloc [32, 32, 128] on @L1 : memref<32x32x128xf32>
              %15 = loom.alloc [32, 32, 32] on @L1 : memref<32x32x32xf32>
              %16 = loom.init_tensor %15[32, 32, 32] : memref<32x32x32xf32> -> tensor<32x32x32xf32>
              %17 = loom.alloc [32, 128, 32] on @L1 : memref<32x128x32xf32>
              %18 = loom.alloc [32, 32] on @L1 : memref<32x32xf32>
              %19 = loom.init_tensor %18[32, 32] : memref<32x32xf32> -> tensor<32x32xf32>
              %20 = loom.alloc [32, 32] on @L1 : memref<32x32xf32>
              %21 = loom.init_tensor %20[32, 32] : memref<32x32xf32> -> tensor<32x32xf32>
              %22 = loom.alloc [32, 32] on @L1 : memref<32x32xf32>
              %23 = loom.init_tensor %22[32, 32] : memref<32x32xf32> -> tensor<32x32xf32>
              %24 = loom.alloc [32, 32] on @L1 : memref<32x32xf32>
              %25 = loom.init_tensor %24[32, 32] : memref<32x32xf32> -> tensor<32x32xf32>
              %26 = loom.alloc [32, 32] on @L1 : memref<32x32xf32>
              %27 = loom.init_tensor %26[32, 32] : memref<32x32xf32> -> tensor<32x32xf32>
              %28 = loom.alloc [32, 32, 128] on @L1 : memref<32x32x128xf32>
              %29 = loom.init_tensor %28[32, 32, 128] : memref<32x32x128xf32> -> tensor<32x32x128xf32>
              %30 = loom.alloc [32, 32, 128] on @L1 : memref<32x32x128xf32>
              %31 = loom.init_tensor %30[32, 32, 128] : memref<32x32x128xf32> -> tensor<32x32x128xf32>
              %32 = loom.alloc [32, 32, 128] on @L1 : memref<32x32x128xf32>
              %33 = loom.init_tensor %32[32, 32, 128] : memref<32x32x128xf32> -> tensor<32x32x128xf32>
              %34 = arith.muli %12, %c32 : index
              %35 = arith.muli %13, %c32 : index
              %c0 = arith.constant 0 : index
              %36 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 524288 + d1 * 128 + d2)>(%34, %35, %c0)
              %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%36], sizes: [32, 32, 128], strides: [524288, 128, 1] : memref<32x4096x128xf32> to memref<32x32x128xf32, strided<[524288, 128, 1], offset: ?>>
              %37 = loom.copy_to_tensor %reinterpret_cast, %28 on @L1, interconnect : [], broadcast : [1, 1] : memref<32x32x128xf32, strided<[524288, 128, 1], offset: ?>>, memref<32x32x128xf32> -> tensor<32x32x128xf32>
              %38 = linalg.fill ins(%cst_1 : f32) outs(%31 : tensor<32x32x128xf32>) -> tensor<32x32x128xf32>
              %39 = linalg.fill ins(%cst_2 : f32) outs(%19 : tensor<32x32xf32>) -> tensor<32x32xf32>
              %40 = linalg.fill ins(%cst_3 : f32) outs(%21 : tensor<32x32xf32>) -> tensor<32x32xf32>
              %41:3 = affine.for %arg8 = 0 to 128 iter_args(%arg9 = %40, %arg10 = %39, %arg11 = %38) -> (tensor<32x32xf32>, tensor<32x32xf32>, tensor<32x32x128xf32>) {
                %44 = arith.muli %arg8, %c32 : index
                %c0_6 = arith.constant 0 : index
                %45 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 524288 + d1 * 4096 + d2)>(%34, %c0_6, %44)
                %reinterpret_cast_7 = memref.reinterpret_cast %arg0 to offset: [%45], sizes: [32, 128, 32], strides: [524288, 4096, 1] : memref<32x128x4096xf32> to memref<32x128x32xf32, strided<[524288, 4096, 1], offset: ?>>
                %46 = loom.copy_to_tensor %reinterpret_cast_7, %17 on @L1, interconnect : [], broadcast : [1, 1] : memref<32x128x32xf32, strided<[524288, 4096, 1], offset: ?>>, memref<32x128x32xf32> -> tensor<32x128x32xf32>
                %47 = linalg.fill ins(%cst_1 : f32) outs(%16 : tensor<32x32x32xf32>) -> tensor<32x32x32xf32>
                %48 = linalg.batch_matmul ins(%37, %46 : tensor<32x32x128xf32>, tensor<32x128x32xf32>) outs(%47 : tensor<32x32x32xf32>) -> tensor<32x32x32xf32>
                %49 = linalg.fill ins(%cst_3 : f32) outs(%23 : tensor<32x32xf32>) -> tensor<32x32xf32>
                %50 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%48 : tensor<32x32x32xf32>) outs(%49 : tensor<32x32xf32>) {
                ^bb0(%in: f32, %out: f32):
                  %63 = arith.maximumf %in, %out : f32
                  linalg.yield %63 : f32
                } -> tensor<32x32xf32>
                %51 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%arg9, %50 : tensor<32x32xf32>, tensor<32x32xf32>) outs(%23 : tensor<32x32xf32>) {
                ^bb0(%in: f32, %in_10: f32, %out: f32):
                  %63 = arith.truncf %cst_0 : f64 to f32
                  %64 = arith.mulf %in_10, %63 : f32
                  %65 = arith.cmpf ogt, %in, %64 : f32
                  %66 = arith.select %65, %in, %64 : f32
                  linalg.yield %66 : f32
                } -> tensor<32x32xf32>
                %52 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%48, %51 : tensor<32x32x32xf32>, tensor<32x32xf32>) outs(%16 : tensor<32x32x32xf32>) {
                ^bb0(%in: f32, %in_10: f32, %out: f32):
                  %63 = arith.truncf %cst_0 : f64 to f32
                  %64 = arith.mulf %in, %63 : f32
                  %65 = arith.subf %64, %in_10 : f32
                  %66 = math.powf %cst, %65 : f32
                  linalg.yield %66 : f32
                } -> tensor<32x32x32xf32>
                %53 = linalg.fill ins(%cst_1 : f32) outs(%25 : tensor<32x32xf32>) -> tensor<32x32xf32>
                %54 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%52 : tensor<32x32x32xf32>) outs(%53 : tensor<32x32xf32>) {
                ^bb0(%in: f32, %out: f32):
                  %63 = arith.addf %in, %out : f32
                  linalg.yield %63 : f32
                } -> tensor<32x32xf32>
                %55 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%arg9, %51 : tensor<32x32xf32>, tensor<32x32xf32>) outs(%27 : tensor<32x32xf32>) {
                ^bb0(%in: f32, %in_10: f32, %out: f32):
                  %63 = arith.subf %in, %in_10 : f32
                  %64 = math.powf %cst, %63 : f32
                  linalg.yield %64 : f32
                } -> tensor<32x32xf32>
                %56 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%arg10, %55, %54 : tensor<32x32xf32>, tensor<32x32xf32>, tensor<32x32xf32>) outs(%arg10 : tensor<32x32xf32>) {
                ^bb0(%in: f32, %in_10: f32, %in_11: f32, %out: f32):
                  %63 = arith.mulf %in, %in_10 : f32
                  %64 = arith.addf %63, %in_11 : f32
                  linalg.yield %64 : f32
                } -> tensor<32x32xf32>
                %c0_8 = arith.constant 0 : index
                %57 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 524288 + d1 * 128 + d2)>(%34, %44, %c0_8)
                %reinterpret_cast_9 = memref.reinterpret_cast %arg1 to offset: [%57], sizes: [32, 32, 128], strides: [524288, 128, 1] : memref<32x4096x128xf32> to memref<32x32x128xf32, strided<[524288, 128, 1], offset: ?>>
                %58 = loom.copy_to_tensor %reinterpret_cast_9, %14 on @L1, interconnect : [@vertical_links], broadcast : [8, 1] : memref<32x32x128xf32, strided<[524288, 128, 1], offset: ?>>, memref<32x32x128xf32> -> tensor<32x32x128xf32>
                %59 = linalg.fill ins(%cst_1 : f32) outs(%33 : tensor<32x32x128xf32>) -> tensor<32x32x128xf32>
                %60 = linalg.batch_matmul ins(%52, %58 : tensor<32x32x32xf32>, tensor<32x32x128xf32>) outs(%59 : tensor<32x32x128xf32>) -> tensor<32x32x128xf32>
                %61 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%60, %arg11, %55 : tensor<32x32x128xf32>, tensor<32x32x128xf32>, tensor<32x32xf32>) outs(%arg11 : tensor<32x32x128xf32>) {
                ^bb0(%in: f32, %in_10: f32, %in_11: f32, %out: f32):
                  %63 = arith.mulf %in_10, %in_11 : f32
                  %64 = arith.addf %in, %63 : f32
                  linalg.yield %64 : f32
                } -> tensor<32x32x128xf32>
                %62 = linalg.copy ins(%51 : tensor<32x32xf32>) outs(%arg9 : tensor<32x32xf32>) -> tensor<32x32xf32>
                affine.yield %62, %56, %61 : tensor<32x32xf32>, tensor<32x32xf32>, tensor<32x32x128xf32>
              }
              %42 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%41#2, %41#1 : tensor<32x32x128xf32>, tensor<32x32xf32>) outs(%29 : tensor<32x32x128xf32>) {
              ^bb0(%in: f32, %in_6: f32, %out: f32):
                %44 = arith.divf %in, %in_6 : f32
                linalg.yield %44 : f32
              } -> tensor<32x32x128xf32>
              %c0_4 = arith.constant 0 : index
              %43 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 524288 + d1 * 128 + d2)>(%34, %35, %c0_4)
              %reinterpret_cast_5 = memref.reinterpret_cast %arg3 to offset: [%43], sizes: [32, 32, 128], strides: [524288, 128, 1] : memref<32x4096x128xf32> to memref<32x32x128xf32, strided<[524288, 128, 1], offset: ?>>
              loom.copy_from_tensor %42, %reinterpret_cast_5 on @DRAM : tensor<32x32x128xf32>, memref<32x32x128xf32, strided<[524288, 128, 1], offset: ?>>
            } {loom.iter_type = #loom.iter_type<temporal>}
          } {loom.iter_type = #loom.iter_type<temporal>}
        } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @y}
      } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @x}
      return
    }
  }
  module attributes {loom.pass_name = "Materialize"} {
    func.func @attention__d0i0_d1i1__f10__d_v_d__BB32__BM32__BN32(%arg0: memref<32x128x4096xf32>, %arg1: memref<32x4096x128xf32>, %arg2: memref<32x4096x128xf32>, %arg3: memref<32x4096x128xf32>) {
      %cst = arith.constant 2.000000e+00 : f32
      %cst_0 = arith.constant 0.12751743074602467 : f64
      %cst_1 = arith.constant 0.000000e+00 : f32
      %cst_2 = arith.constant 1.000000e+00 : f32
      %cst_3 = arith.constant 0xFF800000 : f32
      %c32 = arith.constant 32 : index
      affine.parallel (%arg4) = (0) to (8) {
        affine.parallel (%arg5) = (0) to (8) {
          affine.for %arg6 = 0 to 16 {
            affine.for %arg7 = 0 to 1 {
              %12 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 8)>(%arg4, %arg7)
              %13 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 8)>(%arg5, %arg6)
              %14 = loom.alloc [32, 32, 128] on @L1 : memref<32x32x128xf32>
              %15 = loom.alloc [32, 32, 32] on @L1 : memref<32x32x32xf32>
              %16 = loom.init_tensor %15[32, 32, 32] : memref<32x32x32xf32> -> tensor<32x32x32xf32>
              %17 = loom.alloc [32, 128, 32] on @L1 : memref<32x128x32xf32>
              %18 = loom.alloc [32, 32] on @L1 : memref<32x32xf32>
              %19 = loom.init_tensor %18[32, 32] : memref<32x32xf32> -> tensor<32x32xf32>
              %20 = loom.alloc [32, 32] on @L1 : memref<32x32xf32>
              %21 = loom.init_tensor %20[32, 32] : memref<32x32xf32> -> tensor<32x32xf32>
              %22 = loom.alloc [32, 32] on @L1 : memref<32x32xf32>
              %23 = loom.init_tensor %22[32, 32] : memref<32x32xf32> -> tensor<32x32xf32>
              %24 = loom.alloc [32, 32] on @L1 : memref<32x32xf32>
              %25 = loom.init_tensor %24[32, 32] : memref<32x32xf32> -> tensor<32x32xf32>
              %26 = loom.alloc [32, 32] on @L1 : memref<32x32xf32>
              %27 = loom.init_tensor %26[32, 32] : memref<32x32xf32> -> tensor<32x32xf32>
              %28 = loom.alloc [32, 32, 128] on @L1 : memref<32x32x128xf32>
              %29 = loom.init_tensor %28[32, 32, 128] : memref<32x32x128xf32> -> tensor<32x32x128xf32>
              %30 = loom.alloc [32, 32, 128] on @L1 : memref<32x32x128xf32>
              %31 = loom.init_tensor %30[32, 32, 128] : memref<32x32x128xf32> -> tensor<32x32x128xf32>
              %32 = loom.alloc [32, 32, 128] on @L1 : memref<32x32x128xf32>
              %33 = loom.init_tensor %32[32, 32, 128] : memref<32x32x128xf32> -> tensor<32x32x128xf32>
              %34 = arith.muli %12, %c32 : index
              %35 = arith.muli %13, %c32 : index
              %c0 = arith.constant 0 : index
              %36 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 524288 + d1 * 128 + d2)>(%34, %35, %c0)
              %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%36], sizes: [32, 32, 128], strides: [524288, 128, 1] : memref<32x4096x128xf32> to memref<32x32x128xf32, strided<[524288, 128, 1], offset: ?>>
              %37 = loom.copy_to_tensor %reinterpret_cast, %28 on @L1, interconnect : [], broadcast : [1, 1] : memref<32x32x128xf32, strided<[524288, 128, 1], offset: ?>>, memref<32x32x128xf32> -> tensor<32x32x128xf32>
              %38 = linalg.fill ins(%cst_1 : f32) outs(%31 : tensor<32x32x128xf32>) -> tensor<32x32x128xf32>
              %39 = linalg.fill ins(%cst_2 : f32) outs(%19 : tensor<32x32xf32>) -> tensor<32x32xf32>
              %40 = linalg.fill ins(%cst_3 : f32) outs(%21 : tensor<32x32xf32>) -> tensor<32x32xf32>
              %41:3 = affine.for %arg8 = 0 to 128 iter_args(%arg9 = %40, %arg10 = %39, %arg11 = %38) -> (tensor<32x32xf32>, tensor<32x32xf32>, tensor<32x32x128xf32>) {
                %44 = arith.muli %arg8, %c32 : index
                %c0_6 = arith.constant 0 : index
                %45 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 524288 + d1 * 4096 + d2)>(%34, %c0_6, %44)
                %reinterpret_cast_7 = memref.reinterpret_cast %arg0 to offset: [%45], sizes: [32, 128, 32], strides: [524288, 4096, 1] : memref<32x128x4096xf32> to memref<32x128x32xf32, strided<[524288, 4096, 1], offset: ?>>
                %46 = loom.copy_to_tensor %reinterpret_cast_7, %17 on @L1, interconnect : [@vertical_links], broadcast : [8, 1] : memref<32x128x32xf32, strided<[524288, 4096, 1], offset: ?>>, memref<32x128x32xf32> -> tensor<32x128x32xf32>
                %47 = linalg.fill ins(%cst_1 : f32) outs(%16 : tensor<32x32x32xf32>) -> tensor<32x32x32xf32>
                %48 = linalg.batch_matmul ins(%37, %46 : tensor<32x32x128xf32>, tensor<32x128x32xf32>) outs(%47 : tensor<32x32x32xf32>) -> tensor<32x32x32xf32>
                %49 = linalg.fill ins(%cst_3 : f32) outs(%23 : tensor<32x32xf32>) -> tensor<32x32xf32>
                %50 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%48 : tensor<32x32x32xf32>) outs(%49 : tensor<32x32xf32>) {
                ^bb0(%in: f32, %out: f32):
                  %63 = arith.maximumf %in, %out : f32
                  linalg.yield %63 : f32
                } -> tensor<32x32xf32>
                %51 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%arg9, %50 : tensor<32x32xf32>, tensor<32x32xf32>) outs(%23 : tensor<32x32xf32>) {
                ^bb0(%in: f32, %in_10: f32, %out: f32):
                  %63 = arith.truncf %cst_0 : f64 to f32
                  %64 = arith.mulf %in_10, %63 : f32
                  %65 = arith.cmpf ogt, %in, %64 : f32
                  %66 = arith.select %65, %in, %64 : f32
                  linalg.yield %66 : f32
                } -> tensor<32x32xf32>
                %52 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%48, %51 : tensor<32x32x32xf32>, tensor<32x32xf32>) outs(%16 : tensor<32x32x32xf32>) {
                ^bb0(%in: f32, %in_10: f32, %out: f32):
                  %63 = arith.truncf %cst_0 : f64 to f32
                  %64 = arith.mulf %in, %63 : f32
                  %65 = arith.subf %64, %in_10 : f32
                  %66 = math.powf %cst, %65 : f32
                  linalg.yield %66 : f32
                } -> tensor<32x32x32xf32>
                %53 = linalg.fill ins(%cst_1 : f32) outs(%25 : tensor<32x32xf32>) -> tensor<32x32xf32>
                %54 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%52 : tensor<32x32x32xf32>) outs(%53 : tensor<32x32xf32>) {
                ^bb0(%in: f32, %out: f32):
                  %63 = arith.addf %in, %out : f32
                  linalg.yield %63 : f32
                } -> tensor<32x32xf32>
                %55 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%arg9, %51 : tensor<32x32xf32>, tensor<32x32xf32>) outs(%27 : tensor<32x32xf32>) {
                ^bb0(%in: f32, %in_10: f32, %out: f32):
                  %63 = arith.subf %in, %in_10 : f32
                  %64 = math.powf %cst, %63 : f32
                  linalg.yield %64 : f32
                } -> tensor<32x32xf32>
                %56 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%arg10, %55, %54 : tensor<32x32xf32>, tensor<32x32xf32>, tensor<32x32xf32>) outs(%arg10 : tensor<32x32xf32>) {
                ^bb0(%in: f32, %in_10: f32, %in_11: f32, %out: f32):
                  %63 = arith.mulf %in, %in_10 : f32
                  %64 = arith.addf %63, %in_11 : f32
                  linalg.yield %64 : f32
                } -> tensor<32x32xf32>
                %c0_8 = arith.constant 0 : index
                %57 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 524288 + d1 * 128 + d2)>(%34, %44, %c0_8)
                %reinterpret_cast_9 = memref.reinterpret_cast %arg1 to offset: [%57], sizes: [32, 32, 128], strides: [524288, 128, 1] : memref<32x4096x128xf32> to memref<32x32x128xf32, strided<[524288, 128, 1], offset: ?>>
                %58 = loom.copy_to_tensor %reinterpret_cast_9, %14 on @L1, interconnect : [], broadcast : [1, 1] : memref<32x32x128xf32, strided<[524288, 128, 1], offset: ?>>, memref<32x32x128xf32> -> tensor<32x32x128xf32>
                %59 = linalg.fill ins(%cst_1 : f32) outs(%33 : tensor<32x32x128xf32>) -> tensor<32x32x128xf32>
                %60 = linalg.batch_matmul ins(%52, %58 : tensor<32x32x32xf32>, tensor<32x32x128xf32>) outs(%59 : tensor<32x32x128xf32>) -> tensor<32x32x128xf32>
                %61 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%60, %arg11, %55 : tensor<32x32x128xf32>, tensor<32x32x128xf32>, tensor<32x32xf32>) outs(%arg11 : tensor<32x32x128xf32>) {
                ^bb0(%in: f32, %in_10: f32, %in_11: f32, %out: f32):
                  %63 = arith.mulf %in_10, %in_11 : f32
                  %64 = arith.addf %in, %63 : f32
                  linalg.yield %64 : f32
                } -> tensor<32x32x128xf32>
                %62 = linalg.copy ins(%51 : tensor<32x32xf32>) outs(%arg9 : tensor<32x32xf32>) -> tensor<32x32xf32>
                affine.yield %62, %56, %61 : tensor<32x32xf32>, tensor<32x32xf32>, tensor<32x32x128xf32>
              }
              %42 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%41#2, %41#1 : tensor<32x32x128xf32>, tensor<32x32xf32>) outs(%29 : tensor<32x32x128xf32>) {
              ^bb0(%in: f32, %in_6: f32, %out: f32):
                %44 = arith.divf %in, %in_6 : f32
                linalg.yield %44 : f32
              } -> tensor<32x32x128xf32>
              %c0_4 = arith.constant 0 : index
              %43 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 524288 + d1 * 128 + d2)>(%34, %35, %c0_4)
              %reinterpret_cast_5 = memref.reinterpret_cast %arg3 to offset: [%43], sizes: [32, 32, 128], strides: [524288, 128, 1] : memref<32x4096x128xf32> to memref<32x32x128xf32, strided<[524288, 128, 1], offset: ?>>
              loom.copy_from_tensor %42, %reinterpret_cast_5 on @DRAM : tensor<32x32x128xf32>, memref<32x32x128xf32, strided<[524288, 128, 1], offset: ?>>
            } {loom.iter_type = #loom.iter_type<temporal>}
          } {loom.iter_type = #loom.iter_type<temporal>}
        } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @y}
      } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @x}
      return
    }
  }
  module attributes {loom.pass_name = "Materialize"} {
    func.func @attention__d0i0_d1i1__f10__d_v_v__BB32__BM32__BN32(%arg0: memref<32x128x4096xf32>, %arg1: memref<32x4096x128xf32>, %arg2: memref<32x4096x128xf32>, %arg3: memref<32x4096x128xf32>) {
      %cst = arith.constant 2.000000e+00 : f32
      %cst_0 = arith.constant 0.12751743074602467 : f64
      %cst_1 = arith.constant 0.000000e+00 : f32
      %cst_2 = arith.constant 1.000000e+00 : f32
      %cst_3 = arith.constant 0xFF800000 : f32
      %c32 = arith.constant 32 : index
      affine.parallel (%arg4) = (0) to (8) {
        affine.parallel (%arg5) = (0) to (8) {
          affine.for %arg6 = 0 to 16 {
            affine.for %arg7 = 0 to 1 {
              %12 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 8)>(%arg4, %arg7)
              %13 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 8)>(%arg5, %arg6)
              %14 = loom.alloc [32, 32, 128] on @L1 : memref<32x32x128xf32>
              %15 = loom.alloc [32, 32, 32] on @L1 : memref<32x32x32xf32>
              %16 = loom.init_tensor %15[32, 32, 32] : memref<32x32x32xf32> -> tensor<32x32x32xf32>
              %17 = loom.alloc [32, 128, 32] on @L1 : memref<32x128x32xf32>
              %18 = loom.alloc [32, 32] on @L1 : memref<32x32xf32>
              %19 = loom.init_tensor %18[32, 32] : memref<32x32xf32> -> tensor<32x32xf32>
              %20 = loom.alloc [32, 32] on @L1 : memref<32x32xf32>
              %21 = loom.init_tensor %20[32, 32] : memref<32x32xf32> -> tensor<32x32xf32>
              %22 = loom.alloc [32, 32] on @L1 : memref<32x32xf32>
              %23 = loom.init_tensor %22[32, 32] : memref<32x32xf32> -> tensor<32x32xf32>
              %24 = loom.alloc [32, 32] on @L1 : memref<32x32xf32>
              %25 = loom.init_tensor %24[32, 32] : memref<32x32xf32> -> tensor<32x32xf32>
              %26 = loom.alloc [32, 32] on @L1 : memref<32x32xf32>
              %27 = loom.init_tensor %26[32, 32] : memref<32x32xf32> -> tensor<32x32xf32>
              %28 = loom.alloc [32, 32, 128] on @L1 : memref<32x32x128xf32>
              %29 = loom.init_tensor %28[32, 32, 128] : memref<32x32x128xf32> -> tensor<32x32x128xf32>
              %30 = loom.alloc [32, 32, 128] on @L1 : memref<32x32x128xf32>
              %31 = loom.init_tensor %30[32, 32, 128] : memref<32x32x128xf32> -> tensor<32x32x128xf32>
              %32 = loom.alloc [32, 32, 128] on @L1 : memref<32x32x128xf32>
              %33 = loom.init_tensor %32[32, 32, 128] : memref<32x32x128xf32> -> tensor<32x32x128xf32>
              %34 = arith.muli %12, %c32 : index
              %35 = arith.muli %13, %c32 : index
              %c0 = arith.constant 0 : index
              %36 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 524288 + d1 * 128 + d2)>(%34, %35, %c0)
              %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%36], sizes: [32, 32, 128], strides: [524288, 128, 1] : memref<32x4096x128xf32> to memref<32x32x128xf32, strided<[524288, 128, 1], offset: ?>>
              %37 = loom.copy_to_tensor %reinterpret_cast, %28 on @L1, interconnect : [], broadcast : [1, 1] : memref<32x32x128xf32, strided<[524288, 128, 1], offset: ?>>, memref<32x32x128xf32> -> tensor<32x32x128xf32>
              %38 = linalg.fill ins(%cst_1 : f32) outs(%31 : tensor<32x32x128xf32>) -> tensor<32x32x128xf32>
              %39 = linalg.fill ins(%cst_2 : f32) outs(%19 : tensor<32x32xf32>) -> tensor<32x32xf32>
              %40 = linalg.fill ins(%cst_3 : f32) outs(%21 : tensor<32x32xf32>) -> tensor<32x32xf32>
              %41:3 = affine.for %arg8 = 0 to 128 iter_args(%arg9 = %40, %arg10 = %39, %arg11 = %38) -> (tensor<32x32xf32>, tensor<32x32xf32>, tensor<32x32x128xf32>) {
                %44 = arith.muli %arg8, %c32 : index
                %c0_6 = arith.constant 0 : index
                %45 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 524288 + d1 * 4096 + d2)>(%34, %c0_6, %44)
                %reinterpret_cast_7 = memref.reinterpret_cast %arg0 to offset: [%45], sizes: [32, 128, 32], strides: [524288, 4096, 1] : memref<32x128x4096xf32> to memref<32x128x32xf32, strided<[524288, 4096, 1], offset: ?>>
                %46 = loom.copy_to_tensor %reinterpret_cast_7, %17 on @L1, interconnect : [@vertical_links], broadcast : [8, 1] : memref<32x128x32xf32, strided<[524288, 4096, 1], offset: ?>>, memref<32x128x32xf32> -> tensor<32x128x32xf32>
                %47 = linalg.fill ins(%cst_1 : f32) outs(%16 : tensor<32x32x32xf32>) -> tensor<32x32x32xf32>
                %48 = linalg.batch_matmul ins(%37, %46 : tensor<32x32x128xf32>, tensor<32x128x32xf32>) outs(%47 : tensor<32x32x32xf32>) -> tensor<32x32x32xf32>
                %49 = linalg.fill ins(%cst_3 : f32) outs(%23 : tensor<32x32xf32>) -> tensor<32x32xf32>
                %50 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%48 : tensor<32x32x32xf32>) outs(%49 : tensor<32x32xf32>) {
                ^bb0(%in: f32, %out: f32):
                  %63 = arith.maximumf %in, %out : f32
                  linalg.yield %63 : f32
                } -> tensor<32x32xf32>
                %51 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%arg9, %50 : tensor<32x32xf32>, tensor<32x32xf32>) outs(%23 : tensor<32x32xf32>) {
                ^bb0(%in: f32, %in_10: f32, %out: f32):
                  %63 = arith.truncf %cst_0 : f64 to f32
                  %64 = arith.mulf %in_10, %63 : f32
                  %65 = arith.cmpf ogt, %in, %64 : f32
                  %66 = arith.select %65, %in, %64 : f32
                  linalg.yield %66 : f32
                } -> tensor<32x32xf32>
                %52 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%48, %51 : tensor<32x32x32xf32>, tensor<32x32xf32>) outs(%16 : tensor<32x32x32xf32>) {
                ^bb0(%in: f32, %in_10: f32, %out: f32):
                  %63 = arith.truncf %cst_0 : f64 to f32
                  %64 = arith.mulf %in, %63 : f32
                  %65 = arith.subf %64, %in_10 : f32
                  %66 = math.powf %cst, %65 : f32
                  linalg.yield %66 : f32
                } -> tensor<32x32x32xf32>
                %53 = linalg.fill ins(%cst_1 : f32) outs(%25 : tensor<32x32xf32>) -> tensor<32x32xf32>
                %54 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%52 : tensor<32x32x32xf32>) outs(%53 : tensor<32x32xf32>) {
                ^bb0(%in: f32, %out: f32):
                  %63 = arith.addf %in, %out : f32
                  linalg.yield %63 : f32
                } -> tensor<32x32xf32>
                %55 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%arg9, %51 : tensor<32x32xf32>, tensor<32x32xf32>) outs(%27 : tensor<32x32xf32>) {
                ^bb0(%in: f32, %in_10: f32, %out: f32):
                  %63 = arith.subf %in, %in_10 : f32
                  %64 = math.powf %cst, %63 : f32
                  linalg.yield %64 : f32
                } -> tensor<32x32xf32>
                %56 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%arg10, %55, %54 : tensor<32x32xf32>, tensor<32x32xf32>, tensor<32x32xf32>) outs(%arg10 : tensor<32x32xf32>) {
                ^bb0(%in: f32, %in_10: f32, %in_11: f32, %out: f32):
                  %63 = arith.mulf %in, %in_10 : f32
                  %64 = arith.addf %63, %in_11 : f32
                  linalg.yield %64 : f32
                } -> tensor<32x32xf32>
                %c0_8 = arith.constant 0 : index
                %57 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 524288 + d1 * 128 + d2)>(%34, %44, %c0_8)
                %reinterpret_cast_9 = memref.reinterpret_cast %arg1 to offset: [%57], sizes: [32, 32, 128], strides: [524288, 128, 1] : memref<32x4096x128xf32> to memref<32x32x128xf32, strided<[524288, 128, 1], offset: ?>>
                %58 = loom.copy_to_tensor %reinterpret_cast_9, %14 on @L1, interconnect : [@vertical_links], broadcast : [8, 1] : memref<32x32x128xf32, strided<[524288, 128, 1], offset: ?>>, memref<32x32x128xf32> -> tensor<32x32x128xf32>
                %59 = linalg.fill ins(%cst_1 : f32) outs(%33 : tensor<32x32x128xf32>) -> tensor<32x32x128xf32>
                %60 = linalg.batch_matmul ins(%52, %58 : tensor<32x32x32xf32>, tensor<32x32x128xf32>) outs(%59 : tensor<32x32x128xf32>) -> tensor<32x32x128xf32>
                %61 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%60, %arg11, %55 : tensor<32x32x128xf32>, tensor<32x32x128xf32>, tensor<32x32xf32>) outs(%arg11 : tensor<32x32x128xf32>) {
                ^bb0(%in: f32, %in_10: f32, %in_11: f32, %out: f32):
                  %63 = arith.mulf %in_10, %in_11 : f32
                  %64 = arith.addf %in, %63 : f32
                  linalg.yield %64 : f32
                } -> tensor<32x32x128xf32>
                %62 = linalg.copy ins(%51 : tensor<32x32xf32>) outs(%arg9 : tensor<32x32xf32>) -> tensor<32x32xf32>
                affine.yield %62, %56, %61 : tensor<32x32xf32>, tensor<32x32xf32>, tensor<32x32x128xf32>
              }
              %42 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%41#2, %41#1 : tensor<32x32x128xf32>, tensor<32x32xf32>) outs(%29 : tensor<32x32x128xf32>) {
              ^bb0(%in: f32, %in_6: f32, %out: f32):
                %44 = arith.divf %in, %in_6 : f32
                linalg.yield %44 : f32
              } -> tensor<32x32x128xf32>
              %c0_4 = arith.constant 0 : index
              %43 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 524288 + d1 * 128 + d2)>(%34, %35, %c0_4)
              %reinterpret_cast_5 = memref.reinterpret_cast %arg3 to offset: [%43], sizes: [32, 32, 128], strides: [524288, 128, 1] : memref<32x4096x128xf32> to memref<32x32x128xf32, strided<[524288, 128, 1], offset: ?>>
              loom.copy_from_tensor %42, %reinterpret_cast_5 on @DRAM : tensor<32x32x128xf32>, memref<32x32x128xf32, strided<[524288, 128, 1], offset: ?>>
            } {loom.iter_type = #loom.iter_type<temporal>}
          } {loom.iter_type = #loom.iter_type<temporal>}
        } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @y}
      } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @x}
      return
    }
  }
  module attributes {loom.pass_name = "Materialize"} {
    func.func @attention__d1i0_d0i1__f01__d_d_d__BB32__BM32__BN32(%arg0: memref<32x128x4096xf32>, %arg1: memref<32x4096x128xf32>, %arg2: memref<32x4096x128xf32>, %arg3: memref<32x4096x128xf32>) {
      %cst = arith.constant 2.000000e+00 : f32
      %cst_0 = arith.constant 0.12751743074602467 : f64
      %cst_1 = arith.constant 0.000000e+00 : f32
      %cst_2 = arith.constant 1.000000e+00 : f32
      %cst_3 = arith.constant 0xFF800000 : f32
      %c32 = arith.constant 32 : index
      affine.parallel (%arg4) = (0) to (8) {
        affine.parallel (%arg5) = (0) to (8) {
          affine.for %arg6 = 0 to 1 {
            affine.for %arg7 = 0 to 16 {
              %12 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 8)>(%arg4, %arg6)
              %13 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 8)>(%arg5, %arg7)
              %14 = loom.alloc [32, 32, 128] on @L1 : memref<32x32x128xf32>
              %15 = loom.alloc [32, 32, 32] on @L1 : memref<32x32x32xf32>
              %16 = loom.init_tensor %15[32, 32, 32] : memref<32x32x32xf32> -> tensor<32x32x32xf32>
              %17 = loom.alloc [32, 128, 32] on @L1 : memref<32x128x32xf32>
              %18 = loom.alloc [32, 32] on @L1 : memref<32x32xf32>
              %19 = loom.init_tensor %18[32, 32] : memref<32x32xf32> -> tensor<32x32xf32>
              %20 = loom.alloc [32, 32] on @L1 : memref<32x32xf32>
              %21 = loom.init_tensor %20[32, 32] : memref<32x32xf32> -> tensor<32x32xf32>
              %22 = loom.alloc [32, 32] on @L1 : memref<32x32xf32>
              %23 = loom.init_tensor %22[32, 32] : memref<32x32xf32> -> tensor<32x32xf32>
              %24 = loom.alloc [32, 32] on @L1 : memref<32x32xf32>
              %25 = loom.init_tensor %24[32, 32] : memref<32x32xf32> -> tensor<32x32xf32>
              %26 = loom.alloc [32, 32] on @L1 : memref<32x32xf32>
              %27 = loom.init_tensor %26[32, 32] : memref<32x32xf32> -> tensor<32x32xf32>
              %28 = loom.alloc [32, 32, 128] on @L1 : memref<32x32x128xf32>
              %29 = loom.init_tensor %28[32, 32, 128] : memref<32x32x128xf32> -> tensor<32x32x128xf32>
              %30 = loom.alloc [32, 32, 128] on @L1 : memref<32x32x128xf32>
              %31 = loom.init_tensor %30[32, 32, 128] : memref<32x32x128xf32> -> tensor<32x32x128xf32>
              %32 = loom.alloc [32, 32, 128] on @L1 : memref<32x32x128xf32>
              %33 = loom.init_tensor %32[32, 32, 128] : memref<32x32x128xf32> -> tensor<32x32x128xf32>
              %34 = arith.muli %12, %c32 : index
              %35 = arith.muli %13, %c32 : index
              %c0 = arith.constant 0 : index
              %36 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 524288 + d1 * 128 + d2)>(%34, %35, %c0)
              %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%36], sizes: [32, 32, 128], strides: [524288, 128, 1] : memref<32x4096x128xf32> to memref<32x32x128xf32, strided<[524288, 128, 1], offset: ?>>
              %37 = loom.copy_to_tensor %reinterpret_cast, %28 on @L1, interconnect : [], broadcast : [1, 1] : memref<32x32x128xf32, strided<[524288, 128, 1], offset: ?>>, memref<32x32x128xf32> -> tensor<32x32x128xf32>
              %38 = linalg.fill ins(%cst_1 : f32) outs(%31 : tensor<32x32x128xf32>) -> tensor<32x32x128xf32>
              %39 = linalg.fill ins(%cst_2 : f32) outs(%19 : tensor<32x32xf32>) -> tensor<32x32xf32>
              %40 = linalg.fill ins(%cst_3 : f32) outs(%21 : tensor<32x32xf32>) -> tensor<32x32xf32>
              %41:3 = affine.for %arg8 = 0 to 128 iter_args(%arg9 = %40, %arg10 = %39, %arg11 = %38) -> (tensor<32x32xf32>, tensor<32x32xf32>, tensor<32x32x128xf32>) {
                %44 = arith.muli %arg8, %c32 : index
                %c0_6 = arith.constant 0 : index
                %45 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 524288 + d1 * 4096 + d2)>(%34, %c0_6, %44)
                %reinterpret_cast_7 = memref.reinterpret_cast %arg0 to offset: [%45], sizes: [32, 128, 32], strides: [524288, 4096, 1] : memref<32x128x4096xf32> to memref<32x128x32xf32, strided<[524288, 4096, 1], offset: ?>>
                %46 = loom.copy_to_tensor %reinterpret_cast_7, %17 on @L1, interconnect : [], broadcast : [1, 1] : memref<32x128x32xf32, strided<[524288, 4096, 1], offset: ?>>, memref<32x128x32xf32> -> tensor<32x128x32xf32>
                %47 = linalg.fill ins(%cst_1 : f32) outs(%16 : tensor<32x32x32xf32>) -> tensor<32x32x32xf32>
                %48 = linalg.batch_matmul ins(%37, %46 : tensor<32x32x128xf32>, tensor<32x128x32xf32>) outs(%47 : tensor<32x32x32xf32>) -> tensor<32x32x32xf32>
                %49 = linalg.fill ins(%cst_3 : f32) outs(%23 : tensor<32x32xf32>) -> tensor<32x32xf32>
                %50 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%48 : tensor<32x32x32xf32>) outs(%49 : tensor<32x32xf32>) {
                ^bb0(%in: f32, %out: f32):
                  %63 = arith.maximumf %in, %out : f32
                  linalg.yield %63 : f32
                } -> tensor<32x32xf32>
                %51 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%arg9, %50 : tensor<32x32xf32>, tensor<32x32xf32>) outs(%23 : tensor<32x32xf32>) {
                ^bb0(%in: f32, %in_10: f32, %out: f32):
                  %63 = arith.truncf %cst_0 : f64 to f32
                  %64 = arith.mulf %in_10, %63 : f32
                  %65 = arith.cmpf ogt, %in, %64 : f32
                  %66 = arith.select %65, %in, %64 : f32
                  linalg.yield %66 : f32
                } -> tensor<32x32xf32>
                %52 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%48, %51 : tensor<32x32x32xf32>, tensor<32x32xf32>) outs(%16 : tensor<32x32x32xf32>) {
                ^bb0(%in: f32, %in_10: f32, %out: f32):
                  %63 = arith.truncf %cst_0 : f64 to f32
                  %64 = arith.mulf %in, %63 : f32
                  %65 = arith.subf %64, %in_10 : f32
                  %66 = math.powf %cst, %65 : f32
                  linalg.yield %66 : f32
                } -> tensor<32x32x32xf32>
                %53 = linalg.fill ins(%cst_1 : f32) outs(%25 : tensor<32x32xf32>) -> tensor<32x32xf32>
                %54 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%52 : tensor<32x32x32xf32>) outs(%53 : tensor<32x32xf32>) {
                ^bb0(%in: f32, %out: f32):
                  %63 = arith.addf %in, %out : f32
                  linalg.yield %63 : f32
                } -> tensor<32x32xf32>
                %55 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%arg9, %51 : tensor<32x32xf32>, tensor<32x32xf32>) outs(%27 : tensor<32x32xf32>) {
                ^bb0(%in: f32, %in_10: f32, %out: f32):
                  %63 = arith.subf %in, %in_10 : f32
                  %64 = math.powf %cst, %63 : f32
                  linalg.yield %64 : f32
                } -> tensor<32x32xf32>
                %56 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%arg10, %55, %54 : tensor<32x32xf32>, tensor<32x32xf32>, tensor<32x32xf32>) outs(%arg10 : tensor<32x32xf32>) {
                ^bb0(%in: f32, %in_10: f32, %in_11: f32, %out: f32):
                  %63 = arith.mulf %in, %in_10 : f32
                  %64 = arith.addf %63, %in_11 : f32
                  linalg.yield %64 : f32
                } -> tensor<32x32xf32>
                %c0_8 = arith.constant 0 : index
                %57 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 524288 + d1 * 128 + d2)>(%34, %44, %c0_8)
                %reinterpret_cast_9 = memref.reinterpret_cast %arg1 to offset: [%57], sizes: [32, 32, 128], strides: [524288, 128, 1] : memref<32x4096x128xf32> to memref<32x32x128xf32, strided<[524288, 128, 1], offset: ?>>
                %58 = loom.copy_to_tensor %reinterpret_cast_9, %14 on @L1, interconnect : [], broadcast : [1, 1] : memref<32x32x128xf32, strided<[524288, 128, 1], offset: ?>>, memref<32x32x128xf32> -> tensor<32x32x128xf32>
                %59 = linalg.fill ins(%cst_1 : f32) outs(%33 : tensor<32x32x128xf32>) -> tensor<32x32x128xf32>
                %60 = linalg.batch_matmul ins(%52, %58 : tensor<32x32x32xf32>, tensor<32x32x128xf32>) outs(%59 : tensor<32x32x128xf32>) -> tensor<32x32x128xf32>
                %61 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%60, %arg11, %55 : tensor<32x32x128xf32>, tensor<32x32x128xf32>, tensor<32x32xf32>) outs(%arg11 : tensor<32x32x128xf32>) {
                ^bb0(%in: f32, %in_10: f32, %in_11: f32, %out: f32):
                  %63 = arith.mulf %in_10, %in_11 : f32
                  %64 = arith.addf %in, %63 : f32
                  linalg.yield %64 : f32
                } -> tensor<32x32x128xf32>
                %62 = linalg.copy ins(%51 : tensor<32x32xf32>) outs(%arg9 : tensor<32x32xf32>) -> tensor<32x32xf32>
                affine.yield %62, %56, %61 : tensor<32x32xf32>, tensor<32x32xf32>, tensor<32x32x128xf32>
              }
              %42 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%41#2, %41#1 : tensor<32x32x128xf32>, tensor<32x32xf32>) outs(%29 : tensor<32x32x128xf32>) {
              ^bb0(%in: f32, %in_6: f32, %out: f32):
                %44 = arith.divf %in, %in_6 : f32
                linalg.yield %44 : f32
              } -> tensor<32x32x128xf32>
              %c0_4 = arith.constant 0 : index
              %43 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 524288 + d1 * 128 + d2)>(%34, %35, %c0_4)
              %reinterpret_cast_5 = memref.reinterpret_cast %arg3 to offset: [%43], sizes: [32, 32, 128], strides: [524288, 128, 1] : memref<32x4096x128xf32> to memref<32x32x128xf32, strided<[524288, 128, 1], offset: ?>>
              loom.copy_from_tensor %42, %reinterpret_cast_5 on @DRAM : tensor<32x32x128xf32>, memref<32x32x128xf32, strided<[524288, 128, 1], offset: ?>>
            } {loom.iter_type = #loom.iter_type<temporal>}
          } {loom.iter_type = #loom.iter_type<temporal>}
        } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @x}
      } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @y}
      return
    }
  }
  module attributes {loom.pass_name = "Materialize"} {
    func.func @attention__d1i0_d0i1__f01__d_d_h__BB32__BM32__BN32(%arg0: memref<32x128x4096xf32>, %arg1: memref<32x4096x128xf32>, %arg2: memref<32x4096x128xf32>, %arg3: memref<32x4096x128xf32>) {
      %cst = arith.constant 2.000000e+00 : f32
      %cst_0 = arith.constant 0.12751743074602467 : f64
      %cst_1 = arith.constant 0.000000e+00 : f32
      %cst_2 = arith.constant 1.000000e+00 : f32
      %cst_3 = arith.constant 0xFF800000 : f32
      %c32 = arith.constant 32 : index
      affine.parallel (%arg4) = (0) to (8) {
        affine.parallel (%arg5) = (0) to (8) {
          affine.for %arg6 = 0 to 1 {
            affine.for %arg7 = 0 to 16 {
              %12 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 8)>(%arg4, %arg6)
              %13 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 8)>(%arg5, %arg7)
              %14 = loom.alloc [32, 32, 128] on @L1 : memref<32x32x128xf32>
              %15 = loom.alloc [32, 32, 32] on @L1 : memref<32x32x32xf32>
              %16 = loom.init_tensor %15[32, 32, 32] : memref<32x32x32xf32> -> tensor<32x32x32xf32>
              %17 = loom.alloc [32, 128, 32] on @L1 : memref<32x128x32xf32>
              %18 = loom.alloc [32, 32] on @L1 : memref<32x32xf32>
              %19 = loom.init_tensor %18[32, 32] : memref<32x32xf32> -> tensor<32x32xf32>
              %20 = loom.alloc [32, 32] on @L1 : memref<32x32xf32>
              %21 = loom.init_tensor %20[32, 32] : memref<32x32xf32> -> tensor<32x32xf32>
              %22 = loom.alloc [32, 32] on @L1 : memref<32x32xf32>
              %23 = loom.init_tensor %22[32, 32] : memref<32x32xf32> -> tensor<32x32xf32>
              %24 = loom.alloc [32, 32] on @L1 : memref<32x32xf32>
              %25 = loom.init_tensor %24[32, 32] : memref<32x32xf32> -> tensor<32x32xf32>
              %26 = loom.alloc [32, 32] on @L1 : memref<32x32xf32>
              %27 = loom.init_tensor %26[32, 32] : memref<32x32xf32> -> tensor<32x32xf32>
              %28 = loom.alloc [32, 32, 128] on @L1 : memref<32x32x128xf32>
              %29 = loom.init_tensor %28[32, 32, 128] : memref<32x32x128xf32> -> tensor<32x32x128xf32>
              %30 = loom.alloc [32, 32, 128] on @L1 : memref<32x32x128xf32>
              %31 = loom.init_tensor %30[32, 32, 128] : memref<32x32x128xf32> -> tensor<32x32x128xf32>
              %32 = loom.alloc [32, 32, 128] on @L1 : memref<32x32x128xf32>
              %33 = loom.init_tensor %32[32, 32, 128] : memref<32x32x128xf32> -> tensor<32x32x128xf32>
              %34 = arith.muli %12, %c32 : index
              %35 = arith.muli %13, %c32 : index
              %c0 = arith.constant 0 : index
              %36 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 524288 + d1 * 128 + d2)>(%34, %35, %c0)
              %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%36], sizes: [32, 32, 128], strides: [524288, 128, 1] : memref<32x4096x128xf32> to memref<32x32x128xf32, strided<[524288, 128, 1], offset: ?>>
              %37 = loom.copy_to_tensor %reinterpret_cast, %28 on @L1, interconnect : [], broadcast : [1, 1] : memref<32x32x128xf32, strided<[524288, 128, 1], offset: ?>>, memref<32x32x128xf32> -> tensor<32x32x128xf32>
              %38 = linalg.fill ins(%cst_1 : f32) outs(%31 : tensor<32x32x128xf32>) -> tensor<32x32x128xf32>
              %39 = linalg.fill ins(%cst_2 : f32) outs(%19 : tensor<32x32xf32>) -> tensor<32x32xf32>
              %40 = linalg.fill ins(%cst_3 : f32) outs(%21 : tensor<32x32xf32>) -> tensor<32x32xf32>
              %41:3 = affine.for %arg8 = 0 to 128 iter_args(%arg9 = %40, %arg10 = %39, %arg11 = %38) -> (tensor<32x32xf32>, tensor<32x32xf32>, tensor<32x32x128xf32>) {
                %44 = arith.muli %arg8, %c32 : index
                %c0_6 = arith.constant 0 : index
                %45 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 524288 + d1 * 4096 + d2)>(%34, %c0_6, %44)
                %reinterpret_cast_7 = memref.reinterpret_cast %arg0 to offset: [%45], sizes: [32, 128, 32], strides: [524288, 4096, 1] : memref<32x128x4096xf32> to memref<32x128x32xf32, strided<[524288, 4096, 1], offset: ?>>
                %46 = loom.copy_to_tensor %reinterpret_cast_7, %17 on @L1, interconnect : [], broadcast : [1, 1] : memref<32x128x32xf32, strided<[524288, 4096, 1], offset: ?>>, memref<32x128x32xf32> -> tensor<32x128x32xf32>
                %47 = linalg.fill ins(%cst_1 : f32) outs(%16 : tensor<32x32x32xf32>) -> tensor<32x32x32xf32>
                %48 = linalg.batch_matmul ins(%37, %46 : tensor<32x32x128xf32>, tensor<32x128x32xf32>) outs(%47 : tensor<32x32x32xf32>) -> tensor<32x32x32xf32>
                %49 = linalg.fill ins(%cst_3 : f32) outs(%23 : tensor<32x32xf32>) -> tensor<32x32xf32>
                %50 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%48 : tensor<32x32x32xf32>) outs(%49 : tensor<32x32xf32>) {
                ^bb0(%in: f32, %out: f32):
                  %63 = arith.maximumf %in, %out : f32
                  linalg.yield %63 : f32
                } -> tensor<32x32xf32>
                %51 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%arg9, %50 : tensor<32x32xf32>, tensor<32x32xf32>) outs(%23 : tensor<32x32xf32>) {
                ^bb0(%in: f32, %in_10: f32, %out: f32):
                  %63 = arith.truncf %cst_0 : f64 to f32
                  %64 = arith.mulf %in_10, %63 : f32
                  %65 = arith.cmpf ogt, %in, %64 : f32
                  %66 = arith.select %65, %in, %64 : f32
                  linalg.yield %66 : f32
                } -> tensor<32x32xf32>
                %52 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%48, %51 : tensor<32x32x32xf32>, tensor<32x32xf32>) outs(%16 : tensor<32x32x32xf32>) {
                ^bb0(%in: f32, %in_10: f32, %out: f32):
                  %63 = arith.truncf %cst_0 : f64 to f32
                  %64 = arith.mulf %in, %63 : f32
                  %65 = arith.subf %64, %in_10 : f32
                  %66 = math.powf %cst, %65 : f32
                  linalg.yield %66 : f32
                } -> tensor<32x32x32xf32>
                %53 = linalg.fill ins(%cst_1 : f32) outs(%25 : tensor<32x32xf32>) -> tensor<32x32xf32>
                %54 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%52 : tensor<32x32x32xf32>) outs(%53 : tensor<32x32xf32>) {
                ^bb0(%in: f32, %out: f32):
                  %63 = arith.addf %in, %out : f32
                  linalg.yield %63 : f32
                } -> tensor<32x32xf32>
                %55 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%arg9, %51 : tensor<32x32xf32>, tensor<32x32xf32>) outs(%27 : tensor<32x32xf32>) {
                ^bb0(%in: f32, %in_10: f32, %out: f32):
                  %63 = arith.subf %in, %in_10 : f32
                  %64 = math.powf %cst, %63 : f32
                  linalg.yield %64 : f32
                } -> tensor<32x32xf32>
                %56 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%arg10, %55, %54 : tensor<32x32xf32>, tensor<32x32xf32>, tensor<32x32xf32>) outs(%arg10 : tensor<32x32xf32>) {
                ^bb0(%in: f32, %in_10: f32, %in_11: f32, %out: f32):
                  %63 = arith.mulf %in, %in_10 : f32
                  %64 = arith.addf %63, %in_11 : f32
                  linalg.yield %64 : f32
                } -> tensor<32x32xf32>
                %c0_8 = arith.constant 0 : index
                %57 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 524288 + d1 * 128 + d2)>(%34, %44, %c0_8)
                %reinterpret_cast_9 = memref.reinterpret_cast %arg1 to offset: [%57], sizes: [32, 32, 128], strides: [524288, 128, 1] : memref<32x4096x128xf32> to memref<32x32x128xf32, strided<[524288, 128, 1], offset: ?>>
                %58 = loom.copy_to_tensor %reinterpret_cast_9, %14 on @L1, interconnect : [@horizontal_links], broadcast : [1, 8] : memref<32x32x128xf32, strided<[524288, 128, 1], offset: ?>>, memref<32x32x128xf32> -> tensor<32x32x128xf32>
                %59 = linalg.fill ins(%cst_1 : f32) outs(%33 : tensor<32x32x128xf32>) -> tensor<32x32x128xf32>
                %60 = linalg.batch_matmul ins(%52, %58 : tensor<32x32x32xf32>, tensor<32x32x128xf32>) outs(%59 : tensor<32x32x128xf32>) -> tensor<32x32x128xf32>
                %61 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%60, %arg11, %55 : tensor<32x32x128xf32>, tensor<32x32x128xf32>, tensor<32x32xf32>) outs(%arg11 : tensor<32x32x128xf32>) {
                ^bb0(%in: f32, %in_10: f32, %in_11: f32, %out: f32):
                  %63 = arith.mulf %in_10, %in_11 : f32
                  %64 = arith.addf %in, %63 : f32
                  linalg.yield %64 : f32
                } -> tensor<32x32x128xf32>
                %62 = linalg.copy ins(%51 : tensor<32x32xf32>) outs(%arg9 : tensor<32x32xf32>) -> tensor<32x32xf32>
                affine.yield %62, %56, %61 : tensor<32x32xf32>, tensor<32x32xf32>, tensor<32x32x128xf32>
              }
              %42 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%41#2, %41#1 : tensor<32x32x128xf32>, tensor<32x32xf32>) outs(%29 : tensor<32x32x128xf32>) {
              ^bb0(%in: f32, %in_6: f32, %out: f32):
                %44 = arith.divf %in, %in_6 : f32
                linalg.yield %44 : f32
              } -> tensor<32x32x128xf32>
              %c0_4 = arith.constant 0 : index
              %43 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 524288 + d1 * 128 + d2)>(%34, %35, %c0_4)
              %reinterpret_cast_5 = memref.reinterpret_cast %arg3 to offset: [%43], sizes: [32, 32, 128], strides: [524288, 128, 1] : memref<32x4096x128xf32> to memref<32x32x128xf32, strided<[524288, 128, 1], offset: ?>>
              loom.copy_from_tensor %42, %reinterpret_cast_5 on @DRAM : tensor<32x32x128xf32>, memref<32x32x128xf32, strided<[524288, 128, 1], offset: ?>>
            } {loom.iter_type = #loom.iter_type<temporal>}
          } {loom.iter_type = #loom.iter_type<temporal>}
        } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @x}
      } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @y}
      return
    }
  }
  module attributes {loom.pass_name = "Materialize"} {
    func.func @attention__d1i0_d0i1__f01__d_h_d__BB32__BM32__BN32(%arg0: memref<32x128x4096xf32>, %arg1: memref<32x4096x128xf32>, %arg2: memref<32x4096x128xf32>, %arg3: memref<32x4096x128xf32>) {
      %cst = arith.constant 2.000000e+00 : f32
      %cst_0 = arith.constant 0.12751743074602467 : f64
      %cst_1 = arith.constant 0.000000e+00 : f32
      %cst_2 = arith.constant 1.000000e+00 : f32
      %cst_3 = arith.constant 0xFF800000 : f32
      %c32 = arith.constant 32 : index
      affine.parallel (%arg4) = (0) to (8) {
        affine.parallel (%arg5) = (0) to (8) {
          affine.for %arg6 = 0 to 1 {
            affine.for %arg7 = 0 to 16 {
              %12 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 8)>(%arg4, %arg6)
              %13 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 8)>(%arg5, %arg7)
              %14 = loom.alloc [32, 32, 128] on @L1 : memref<32x32x128xf32>
              %15 = loom.alloc [32, 32, 32] on @L1 : memref<32x32x32xf32>
              %16 = loom.init_tensor %15[32, 32, 32] : memref<32x32x32xf32> -> tensor<32x32x32xf32>
              %17 = loom.alloc [32, 128, 32] on @L1 : memref<32x128x32xf32>
              %18 = loom.alloc [32, 32] on @L1 : memref<32x32xf32>
              %19 = loom.init_tensor %18[32, 32] : memref<32x32xf32> -> tensor<32x32xf32>
              %20 = loom.alloc [32, 32] on @L1 : memref<32x32xf32>
              %21 = loom.init_tensor %20[32, 32] : memref<32x32xf32> -> tensor<32x32xf32>
              %22 = loom.alloc [32, 32] on @L1 : memref<32x32xf32>
              %23 = loom.init_tensor %22[32, 32] : memref<32x32xf32> -> tensor<32x32xf32>
              %24 = loom.alloc [32, 32] on @L1 : memref<32x32xf32>
              %25 = loom.init_tensor %24[32, 32] : memref<32x32xf32> -> tensor<32x32xf32>
              %26 = loom.alloc [32, 32] on @L1 : memref<32x32xf32>
              %27 = loom.init_tensor %26[32, 32] : memref<32x32xf32> -> tensor<32x32xf32>
              %28 = loom.alloc [32, 32, 128] on @L1 : memref<32x32x128xf32>
              %29 = loom.init_tensor %28[32, 32, 128] : memref<32x32x128xf32> -> tensor<32x32x128xf32>
              %30 = loom.alloc [32, 32, 128] on @L1 : memref<32x32x128xf32>
              %31 = loom.init_tensor %30[32, 32, 128] : memref<32x32x128xf32> -> tensor<32x32x128xf32>
              %32 = loom.alloc [32, 32, 128] on @L1 : memref<32x32x128xf32>
              %33 = loom.init_tensor %32[32, 32, 128] : memref<32x32x128xf32> -> tensor<32x32x128xf32>
              %34 = arith.muli %12, %c32 : index
              %35 = arith.muli %13, %c32 : index
              %c0 = arith.constant 0 : index
              %36 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 524288 + d1 * 128 + d2)>(%34, %35, %c0)
              %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%36], sizes: [32, 32, 128], strides: [524288, 128, 1] : memref<32x4096x128xf32> to memref<32x32x128xf32, strided<[524288, 128, 1], offset: ?>>
              %37 = loom.copy_to_tensor %reinterpret_cast, %28 on @L1, interconnect : [], broadcast : [1, 1] : memref<32x32x128xf32, strided<[524288, 128, 1], offset: ?>>, memref<32x32x128xf32> -> tensor<32x32x128xf32>
              %38 = linalg.fill ins(%cst_1 : f32) outs(%31 : tensor<32x32x128xf32>) -> tensor<32x32x128xf32>
              %39 = linalg.fill ins(%cst_2 : f32) outs(%19 : tensor<32x32xf32>) -> tensor<32x32xf32>
              %40 = linalg.fill ins(%cst_3 : f32) outs(%21 : tensor<32x32xf32>) -> tensor<32x32xf32>
              %41:3 = affine.for %arg8 = 0 to 128 iter_args(%arg9 = %40, %arg10 = %39, %arg11 = %38) -> (tensor<32x32xf32>, tensor<32x32xf32>, tensor<32x32x128xf32>) {
                %44 = arith.muli %arg8, %c32 : index
                %c0_6 = arith.constant 0 : index
                %45 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 524288 + d1 * 4096 + d2)>(%34, %c0_6, %44)
                %reinterpret_cast_7 = memref.reinterpret_cast %arg0 to offset: [%45], sizes: [32, 128, 32], strides: [524288, 4096, 1] : memref<32x128x4096xf32> to memref<32x128x32xf32, strided<[524288, 4096, 1], offset: ?>>
                %46 = loom.copy_to_tensor %reinterpret_cast_7, %17 on @L1, interconnect : [@horizontal_links], broadcast : [1, 8] : memref<32x128x32xf32, strided<[524288, 4096, 1], offset: ?>>, memref<32x128x32xf32> -> tensor<32x128x32xf32>
                %47 = linalg.fill ins(%cst_1 : f32) outs(%16 : tensor<32x32x32xf32>) -> tensor<32x32x32xf32>
                %48 = linalg.batch_matmul ins(%37, %46 : tensor<32x32x128xf32>, tensor<32x128x32xf32>) outs(%47 : tensor<32x32x32xf32>) -> tensor<32x32x32xf32>
                %49 = linalg.fill ins(%cst_3 : f32) outs(%23 : tensor<32x32xf32>) -> tensor<32x32xf32>
                %50 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%48 : tensor<32x32x32xf32>) outs(%49 : tensor<32x32xf32>) {
                ^bb0(%in: f32, %out: f32):
                  %63 = arith.maximumf %in, %out : f32
                  linalg.yield %63 : f32
                } -> tensor<32x32xf32>
                %51 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%arg9, %50 : tensor<32x32xf32>, tensor<32x32xf32>) outs(%23 : tensor<32x32xf32>) {
                ^bb0(%in: f32, %in_10: f32, %out: f32):
                  %63 = arith.truncf %cst_0 : f64 to f32
                  %64 = arith.mulf %in_10, %63 : f32
                  %65 = arith.cmpf ogt, %in, %64 : f32
                  %66 = arith.select %65, %in, %64 : f32
                  linalg.yield %66 : f32
                } -> tensor<32x32xf32>
                %52 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%48, %51 : tensor<32x32x32xf32>, tensor<32x32xf32>) outs(%16 : tensor<32x32x32xf32>) {
                ^bb0(%in: f32, %in_10: f32, %out: f32):
                  %63 = arith.truncf %cst_0 : f64 to f32
                  %64 = arith.mulf %in, %63 : f32
                  %65 = arith.subf %64, %in_10 : f32
                  %66 = math.powf %cst, %65 : f32
                  linalg.yield %66 : f32
                } -> tensor<32x32x32xf32>
                %53 = linalg.fill ins(%cst_1 : f32) outs(%25 : tensor<32x32xf32>) -> tensor<32x32xf32>
                %54 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%52 : tensor<32x32x32xf32>) outs(%53 : tensor<32x32xf32>) {
                ^bb0(%in: f32, %out: f32):
                  %63 = arith.addf %in, %out : f32
                  linalg.yield %63 : f32
                } -> tensor<32x32xf32>
                %55 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%arg9, %51 : tensor<32x32xf32>, tensor<32x32xf32>) outs(%27 : tensor<32x32xf32>) {
                ^bb0(%in: f32, %in_10: f32, %out: f32):
                  %63 = arith.subf %in, %in_10 : f32
                  %64 = math.powf %cst, %63 : f32
                  linalg.yield %64 : f32
                } -> tensor<32x32xf32>
                %56 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%arg10, %55, %54 : tensor<32x32xf32>, tensor<32x32xf32>, tensor<32x32xf32>) outs(%arg10 : tensor<32x32xf32>) {
                ^bb0(%in: f32, %in_10: f32, %in_11: f32, %out: f32):
                  %63 = arith.mulf %in, %in_10 : f32
                  %64 = arith.addf %63, %in_11 : f32
                  linalg.yield %64 : f32
                } -> tensor<32x32xf32>
                %c0_8 = arith.constant 0 : index
                %57 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 524288 + d1 * 128 + d2)>(%34, %44, %c0_8)
                %reinterpret_cast_9 = memref.reinterpret_cast %arg1 to offset: [%57], sizes: [32, 32, 128], strides: [524288, 128, 1] : memref<32x4096x128xf32> to memref<32x32x128xf32, strided<[524288, 128, 1], offset: ?>>
                %58 = loom.copy_to_tensor %reinterpret_cast_9, %14 on @L1, interconnect : [], broadcast : [1, 1] : memref<32x32x128xf32, strided<[524288, 128, 1], offset: ?>>, memref<32x32x128xf32> -> tensor<32x32x128xf32>
                %59 = linalg.fill ins(%cst_1 : f32) outs(%33 : tensor<32x32x128xf32>) -> tensor<32x32x128xf32>
                %60 = linalg.batch_matmul ins(%52, %58 : tensor<32x32x32xf32>, tensor<32x32x128xf32>) outs(%59 : tensor<32x32x128xf32>) -> tensor<32x32x128xf32>
                %61 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%60, %arg11, %55 : tensor<32x32x128xf32>, tensor<32x32x128xf32>, tensor<32x32xf32>) outs(%arg11 : tensor<32x32x128xf32>) {
                ^bb0(%in: f32, %in_10: f32, %in_11: f32, %out: f32):
                  %63 = arith.mulf %in_10, %in_11 : f32
                  %64 = arith.addf %in, %63 : f32
                  linalg.yield %64 : f32
                } -> tensor<32x32x128xf32>
                %62 = linalg.copy ins(%51 : tensor<32x32xf32>) outs(%arg9 : tensor<32x32xf32>) -> tensor<32x32xf32>
                affine.yield %62, %56, %61 : tensor<32x32xf32>, tensor<32x32xf32>, tensor<32x32x128xf32>
              }
              %42 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%41#2, %41#1 : tensor<32x32x128xf32>, tensor<32x32xf32>) outs(%29 : tensor<32x32x128xf32>) {
              ^bb0(%in: f32, %in_6: f32, %out: f32):
                %44 = arith.divf %in, %in_6 : f32
                linalg.yield %44 : f32
              } -> tensor<32x32x128xf32>
              %c0_4 = arith.constant 0 : index
              %43 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 524288 + d1 * 128 + d2)>(%34, %35, %c0_4)
              %reinterpret_cast_5 = memref.reinterpret_cast %arg3 to offset: [%43], sizes: [32, 32, 128], strides: [524288, 128, 1] : memref<32x4096x128xf32> to memref<32x32x128xf32, strided<[524288, 128, 1], offset: ?>>
              loom.copy_from_tensor %42, %reinterpret_cast_5 on @DRAM : tensor<32x32x128xf32>, memref<32x32x128xf32, strided<[524288, 128, 1], offset: ?>>
            } {loom.iter_type = #loom.iter_type<temporal>}
          } {loom.iter_type = #loom.iter_type<temporal>}
        } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @x}
      } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @y}
      return
    }
  }
  module attributes {loom.pass_name = "Materialize"} {
    func.func @attention__d1i0_d0i1__f01__d_h_h__BB32__BM32__BN32(%arg0: memref<32x128x4096xf32>, %arg1: memref<32x4096x128xf32>, %arg2: memref<32x4096x128xf32>, %arg3: memref<32x4096x128xf32>) {
      %cst = arith.constant 2.000000e+00 : f32
      %cst_0 = arith.constant 0.12751743074602467 : f64
      %cst_1 = arith.constant 0.000000e+00 : f32
      %cst_2 = arith.constant 1.000000e+00 : f32
      %cst_3 = arith.constant 0xFF800000 : f32
      %c32 = arith.constant 32 : index
      affine.parallel (%arg4) = (0) to (8) {
        affine.parallel (%arg5) = (0) to (8) {
          affine.for %arg6 = 0 to 1 {
            affine.for %arg7 = 0 to 16 {
              %12 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 8)>(%arg4, %arg6)
              %13 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 8)>(%arg5, %arg7)
              %14 = loom.alloc [32, 32, 128] on @L1 : memref<32x32x128xf32>
              %15 = loom.alloc [32, 32, 32] on @L1 : memref<32x32x32xf32>
              %16 = loom.init_tensor %15[32, 32, 32] : memref<32x32x32xf32> -> tensor<32x32x32xf32>
              %17 = loom.alloc [32, 128, 32] on @L1 : memref<32x128x32xf32>
              %18 = loom.alloc [32, 32] on @L1 : memref<32x32xf32>
              %19 = loom.init_tensor %18[32, 32] : memref<32x32xf32> -> tensor<32x32xf32>
              %20 = loom.alloc [32, 32] on @L1 : memref<32x32xf32>
              %21 = loom.init_tensor %20[32, 32] : memref<32x32xf32> -> tensor<32x32xf32>
              %22 = loom.alloc [32, 32] on @L1 : memref<32x32xf32>
              %23 = loom.init_tensor %22[32, 32] : memref<32x32xf32> -> tensor<32x32xf32>
              %24 = loom.alloc [32, 32] on @L1 : memref<32x32xf32>
              %25 = loom.init_tensor %24[32, 32] : memref<32x32xf32> -> tensor<32x32xf32>
              %26 = loom.alloc [32, 32] on @L1 : memref<32x32xf32>
              %27 = loom.init_tensor %26[32, 32] : memref<32x32xf32> -> tensor<32x32xf32>
              %28 = loom.alloc [32, 32, 128] on @L1 : memref<32x32x128xf32>
              %29 = loom.init_tensor %28[32, 32, 128] : memref<32x32x128xf32> -> tensor<32x32x128xf32>
              %30 = loom.alloc [32, 32, 128] on @L1 : memref<32x32x128xf32>
              %31 = loom.init_tensor %30[32, 32, 128] : memref<32x32x128xf32> -> tensor<32x32x128xf32>
              %32 = loom.alloc [32, 32, 128] on @L1 : memref<32x32x128xf32>
              %33 = loom.init_tensor %32[32, 32, 128] : memref<32x32x128xf32> -> tensor<32x32x128xf32>
              %34 = arith.muli %12, %c32 : index
              %35 = arith.muli %13, %c32 : index
              %c0 = arith.constant 0 : index
              %36 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 524288 + d1 * 128 + d2)>(%34, %35, %c0)
              %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%36], sizes: [32, 32, 128], strides: [524288, 128, 1] : memref<32x4096x128xf32> to memref<32x32x128xf32, strided<[524288, 128, 1], offset: ?>>
              %37 = loom.copy_to_tensor %reinterpret_cast, %28 on @L1, interconnect : [], broadcast : [1, 1] : memref<32x32x128xf32, strided<[524288, 128, 1], offset: ?>>, memref<32x32x128xf32> -> tensor<32x32x128xf32>
              %38 = linalg.fill ins(%cst_1 : f32) outs(%31 : tensor<32x32x128xf32>) -> tensor<32x32x128xf32>
              %39 = linalg.fill ins(%cst_2 : f32) outs(%19 : tensor<32x32xf32>) -> tensor<32x32xf32>
              %40 = linalg.fill ins(%cst_3 : f32) outs(%21 : tensor<32x32xf32>) -> tensor<32x32xf32>
              %41:3 = affine.for %arg8 = 0 to 128 iter_args(%arg9 = %40, %arg10 = %39, %arg11 = %38) -> (tensor<32x32xf32>, tensor<32x32xf32>, tensor<32x32x128xf32>) {
                %44 = arith.muli %arg8, %c32 : index
                %c0_6 = arith.constant 0 : index
                %45 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 524288 + d1 * 4096 + d2)>(%34, %c0_6, %44)
                %reinterpret_cast_7 = memref.reinterpret_cast %arg0 to offset: [%45], sizes: [32, 128, 32], strides: [524288, 4096, 1] : memref<32x128x4096xf32> to memref<32x128x32xf32, strided<[524288, 4096, 1], offset: ?>>
                %46 = loom.copy_to_tensor %reinterpret_cast_7, %17 on @L1, interconnect : [@horizontal_links], broadcast : [1, 8] : memref<32x128x32xf32, strided<[524288, 4096, 1], offset: ?>>, memref<32x128x32xf32> -> tensor<32x128x32xf32>
                %47 = linalg.fill ins(%cst_1 : f32) outs(%16 : tensor<32x32x32xf32>) -> tensor<32x32x32xf32>
                %48 = linalg.batch_matmul ins(%37, %46 : tensor<32x32x128xf32>, tensor<32x128x32xf32>) outs(%47 : tensor<32x32x32xf32>) -> tensor<32x32x32xf32>
                %49 = linalg.fill ins(%cst_3 : f32) outs(%23 : tensor<32x32xf32>) -> tensor<32x32xf32>
                %50 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%48 : tensor<32x32x32xf32>) outs(%49 : tensor<32x32xf32>) {
                ^bb0(%in: f32, %out: f32):
                  %63 = arith.maximumf %in, %out : f32
                  linalg.yield %63 : f32
                } -> tensor<32x32xf32>
                %51 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%arg9, %50 : tensor<32x32xf32>, tensor<32x32xf32>) outs(%23 : tensor<32x32xf32>) {
                ^bb0(%in: f32, %in_10: f32, %out: f32):
                  %63 = arith.truncf %cst_0 : f64 to f32
                  %64 = arith.mulf %in_10, %63 : f32
                  %65 = arith.cmpf ogt, %in, %64 : f32
                  %66 = arith.select %65, %in, %64 : f32
                  linalg.yield %66 : f32
                } -> tensor<32x32xf32>
                %52 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%48, %51 : tensor<32x32x32xf32>, tensor<32x32xf32>) outs(%16 : tensor<32x32x32xf32>) {
                ^bb0(%in: f32, %in_10: f32, %out: f32):
                  %63 = arith.truncf %cst_0 : f64 to f32
                  %64 = arith.mulf %in, %63 : f32
                  %65 = arith.subf %64, %in_10 : f32
                  %66 = math.powf %cst, %65 : f32
                  linalg.yield %66 : f32
                } -> tensor<32x32x32xf32>
                %53 = linalg.fill ins(%cst_1 : f32) outs(%25 : tensor<32x32xf32>) -> tensor<32x32xf32>
                %54 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%52 : tensor<32x32x32xf32>) outs(%53 : tensor<32x32xf32>) {
                ^bb0(%in: f32, %out: f32):
                  %63 = arith.addf %in, %out : f32
                  linalg.yield %63 : f32
                } -> tensor<32x32xf32>
                %55 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%arg9, %51 : tensor<32x32xf32>, tensor<32x32xf32>) outs(%27 : tensor<32x32xf32>) {
                ^bb0(%in: f32, %in_10: f32, %out: f32):
                  %63 = arith.subf %in, %in_10 : f32
                  %64 = math.powf %cst, %63 : f32
                  linalg.yield %64 : f32
                } -> tensor<32x32xf32>
                %56 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%arg10, %55, %54 : tensor<32x32xf32>, tensor<32x32xf32>, tensor<32x32xf32>) outs(%arg10 : tensor<32x32xf32>) {
                ^bb0(%in: f32, %in_10: f32, %in_11: f32, %out: f32):
                  %63 = arith.mulf %in, %in_10 : f32
                  %64 = arith.addf %63, %in_11 : f32
                  linalg.yield %64 : f32
                } -> tensor<32x32xf32>
                %c0_8 = arith.constant 0 : index
                %57 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 524288 + d1 * 128 + d2)>(%34, %44, %c0_8)
                %reinterpret_cast_9 = memref.reinterpret_cast %arg1 to offset: [%57], sizes: [32, 32, 128], strides: [524288, 128, 1] : memref<32x4096x128xf32> to memref<32x32x128xf32, strided<[524288, 128, 1], offset: ?>>
                %58 = loom.copy_to_tensor %reinterpret_cast_9, %14 on @L1, interconnect : [@horizontal_links], broadcast : [1, 8] : memref<32x32x128xf32, strided<[524288, 128, 1], offset: ?>>, memref<32x32x128xf32> -> tensor<32x32x128xf32>
                %59 = linalg.fill ins(%cst_1 : f32) outs(%33 : tensor<32x32x128xf32>) -> tensor<32x32x128xf32>
                %60 = linalg.batch_matmul ins(%52, %58 : tensor<32x32x32xf32>, tensor<32x32x128xf32>) outs(%59 : tensor<32x32x128xf32>) -> tensor<32x32x128xf32>
                %61 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%60, %arg11, %55 : tensor<32x32x128xf32>, tensor<32x32x128xf32>, tensor<32x32xf32>) outs(%arg11 : tensor<32x32x128xf32>) {
                ^bb0(%in: f32, %in_10: f32, %in_11: f32, %out: f32):
                  %63 = arith.mulf %in_10, %in_11 : f32
                  %64 = arith.addf %in, %63 : f32
                  linalg.yield %64 : f32
                } -> tensor<32x32x128xf32>
                %62 = linalg.copy ins(%51 : tensor<32x32xf32>) outs(%arg9 : tensor<32x32xf32>) -> tensor<32x32xf32>
                affine.yield %62, %56, %61 : tensor<32x32xf32>, tensor<32x32xf32>, tensor<32x32x128xf32>
              }
              %42 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%41#2, %41#1 : tensor<32x32x128xf32>, tensor<32x32xf32>) outs(%29 : tensor<32x32x128xf32>) {
              ^bb0(%in: f32, %in_6: f32, %out: f32):
                %44 = arith.divf %in, %in_6 : f32
                linalg.yield %44 : f32
              } -> tensor<32x32x128xf32>
              %c0_4 = arith.constant 0 : index
              %43 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 524288 + d1 * 128 + d2)>(%34, %35, %c0_4)
              %reinterpret_cast_5 = memref.reinterpret_cast %arg3 to offset: [%43], sizes: [32, 32, 128], strides: [524288, 128, 1] : memref<32x4096x128xf32> to memref<32x32x128xf32, strided<[524288, 128, 1], offset: ?>>
              loom.copy_from_tensor %42, %reinterpret_cast_5 on @DRAM : tensor<32x32x128xf32>, memref<32x32x128xf32, strided<[524288, 128, 1], offset: ?>>
            } {loom.iter_type = #loom.iter_type<temporal>}
          } {loom.iter_type = #loom.iter_type<temporal>}
        } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @x}
      } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @y}
      return
    }
  }
  module attributes {loom.pass_name = "Materialize"} {
    func.func @attention__d1i0_d0i1__f10__d_d_d__BB32__BM32__BN32(%arg0: memref<32x128x4096xf32>, %arg1: memref<32x4096x128xf32>, %arg2: memref<32x4096x128xf32>, %arg3: memref<32x4096x128xf32>) {
      %cst = arith.constant 2.000000e+00 : f32
      %cst_0 = arith.constant 0.12751743074602467 : f64
      %cst_1 = arith.constant 0.000000e+00 : f32
      %cst_2 = arith.constant 1.000000e+00 : f32
      %cst_3 = arith.constant 0xFF800000 : f32
      %c32 = arith.constant 32 : index
      affine.parallel (%arg4) = (0) to (8) {
        affine.parallel (%arg5) = (0) to (8) {
          affine.for %arg6 = 0 to 16 {
            affine.for %arg7 = 0 to 1 {
              %12 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 8)>(%arg4, %arg7)
              %13 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 8)>(%arg5, %arg6)
              %14 = loom.alloc [32, 32, 128] on @L1 : memref<32x32x128xf32>
              %15 = loom.alloc [32, 32, 32] on @L1 : memref<32x32x32xf32>
              %16 = loom.init_tensor %15[32, 32, 32] : memref<32x32x32xf32> -> tensor<32x32x32xf32>
              %17 = loom.alloc [32, 128, 32] on @L1 : memref<32x128x32xf32>
              %18 = loom.alloc [32, 32] on @L1 : memref<32x32xf32>
              %19 = loom.init_tensor %18[32, 32] : memref<32x32xf32> -> tensor<32x32xf32>
              %20 = loom.alloc [32, 32] on @L1 : memref<32x32xf32>
              %21 = loom.init_tensor %20[32, 32] : memref<32x32xf32> -> tensor<32x32xf32>
              %22 = loom.alloc [32, 32] on @L1 : memref<32x32xf32>
              %23 = loom.init_tensor %22[32, 32] : memref<32x32xf32> -> tensor<32x32xf32>
              %24 = loom.alloc [32, 32] on @L1 : memref<32x32xf32>
              %25 = loom.init_tensor %24[32, 32] : memref<32x32xf32> -> tensor<32x32xf32>
              %26 = loom.alloc [32, 32] on @L1 : memref<32x32xf32>
              %27 = loom.init_tensor %26[32, 32] : memref<32x32xf32> -> tensor<32x32xf32>
              %28 = loom.alloc [32, 32, 128] on @L1 : memref<32x32x128xf32>
              %29 = loom.init_tensor %28[32, 32, 128] : memref<32x32x128xf32> -> tensor<32x32x128xf32>
              %30 = loom.alloc [32, 32, 128] on @L1 : memref<32x32x128xf32>
              %31 = loom.init_tensor %30[32, 32, 128] : memref<32x32x128xf32> -> tensor<32x32x128xf32>
              %32 = loom.alloc [32, 32, 128] on @L1 : memref<32x32x128xf32>
              %33 = loom.init_tensor %32[32, 32, 128] : memref<32x32x128xf32> -> tensor<32x32x128xf32>
              %34 = arith.muli %12, %c32 : index
              %35 = arith.muli %13, %c32 : index
              %c0 = arith.constant 0 : index
              %36 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 524288 + d1 * 128 + d2)>(%34, %35, %c0)
              %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%36], sizes: [32, 32, 128], strides: [524288, 128, 1] : memref<32x4096x128xf32> to memref<32x32x128xf32, strided<[524288, 128, 1], offset: ?>>
              %37 = loom.copy_to_tensor %reinterpret_cast, %28 on @L1, interconnect : [], broadcast : [1, 1] : memref<32x32x128xf32, strided<[524288, 128, 1], offset: ?>>, memref<32x32x128xf32> -> tensor<32x32x128xf32>
              %38 = linalg.fill ins(%cst_1 : f32) outs(%31 : tensor<32x32x128xf32>) -> tensor<32x32x128xf32>
              %39 = linalg.fill ins(%cst_2 : f32) outs(%19 : tensor<32x32xf32>) -> tensor<32x32xf32>
              %40 = linalg.fill ins(%cst_3 : f32) outs(%21 : tensor<32x32xf32>) -> tensor<32x32xf32>
              %41:3 = affine.for %arg8 = 0 to 128 iter_args(%arg9 = %40, %arg10 = %39, %arg11 = %38) -> (tensor<32x32xf32>, tensor<32x32xf32>, tensor<32x32x128xf32>) {
                %44 = arith.muli %arg8, %c32 : index
                %c0_6 = arith.constant 0 : index
                %45 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 524288 + d1 * 4096 + d2)>(%34, %c0_6, %44)
                %reinterpret_cast_7 = memref.reinterpret_cast %arg0 to offset: [%45], sizes: [32, 128, 32], strides: [524288, 4096, 1] : memref<32x128x4096xf32> to memref<32x128x32xf32, strided<[524288, 4096, 1], offset: ?>>
                %46 = loom.copy_to_tensor %reinterpret_cast_7, %17 on @L1, interconnect : [], broadcast : [1, 1] : memref<32x128x32xf32, strided<[524288, 4096, 1], offset: ?>>, memref<32x128x32xf32> -> tensor<32x128x32xf32>
                %47 = linalg.fill ins(%cst_1 : f32) outs(%16 : tensor<32x32x32xf32>) -> tensor<32x32x32xf32>
                %48 = linalg.batch_matmul ins(%37, %46 : tensor<32x32x128xf32>, tensor<32x128x32xf32>) outs(%47 : tensor<32x32x32xf32>) -> tensor<32x32x32xf32>
                %49 = linalg.fill ins(%cst_3 : f32) outs(%23 : tensor<32x32xf32>) -> tensor<32x32xf32>
                %50 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%48 : tensor<32x32x32xf32>) outs(%49 : tensor<32x32xf32>) {
                ^bb0(%in: f32, %out: f32):
                  %63 = arith.maximumf %in, %out : f32
                  linalg.yield %63 : f32
                } -> tensor<32x32xf32>
                %51 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%arg9, %50 : tensor<32x32xf32>, tensor<32x32xf32>) outs(%23 : tensor<32x32xf32>) {
                ^bb0(%in: f32, %in_10: f32, %out: f32):
                  %63 = arith.truncf %cst_0 : f64 to f32
                  %64 = arith.mulf %in_10, %63 : f32
                  %65 = arith.cmpf ogt, %in, %64 : f32
                  %66 = arith.select %65, %in, %64 : f32
                  linalg.yield %66 : f32
                } -> tensor<32x32xf32>
                %52 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%48, %51 : tensor<32x32x32xf32>, tensor<32x32xf32>) outs(%16 : tensor<32x32x32xf32>) {
                ^bb0(%in: f32, %in_10: f32, %out: f32):
                  %63 = arith.truncf %cst_0 : f64 to f32
                  %64 = arith.mulf %in, %63 : f32
                  %65 = arith.subf %64, %in_10 : f32
                  %66 = math.powf %cst, %65 : f32
                  linalg.yield %66 : f32
                } -> tensor<32x32x32xf32>
                %53 = linalg.fill ins(%cst_1 : f32) outs(%25 : tensor<32x32xf32>) -> tensor<32x32xf32>
                %54 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%52 : tensor<32x32x32xf32>) outs(%53 : tensor<32x32xf32>) {
                ^bb0(%in: f32, %out: f32):
                  %63 = arith.addf %in, %out : f32
                  linalg.yield %63 : f32
                } -> tensor<32x32xf32>
                %55 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%arg9, %51 : tensor<32x32xf32>, tensor<32x32xf32>) outs(%27 : tensor<32x32xf32>) {
                ^bb0(%in: f32, %in_10: f32, %out: f32):
                  %63 = arith.subf %in, %in_10 : f32
                  %64 = math.powf %cst, %63 : f32
                  linalg.yield %64 : f32
                } -> tensor<32x32xf32>
                %56 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%arg10, %55, %54 : tensor<32x32xf32>, tensor<32x32xf32>, tensor<32x32xf32>) outs(%arg10 : tensor<32x32xf32>) {
                ^bb0(%in: f32, %in_10: f32, %in_11: f32, %out: f32):
                  %63 = arith.mulf %in, %in_10 : f32
                  %64 = arith.addf %63, %in_11 : f32
                  linalg.yield %64 : f32
                } -> tensor<32x32xf32>
                %c0_8 = arith.constant 0 : index
                %57 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 524288 + d1 * 128 + d2)>(%34, %44, %c0_8)
                %reinterpret_cast_9 = memref.reinterpret_cast %arg1 to offset: [%57], sizes: [32, 32, 128], strides: [524288, 128, 1] : memref<32x4096x128xf32> to memref<32x32x128xf32, strided<[524288, 128, 1], offset: ?>>
                %58 = loom.copy_to_tensor %reinterpret_cast_9, %14 on @L1, interconnect : [], broadcast : [1, 1] : memref<32x32x128xf32, strided<[524288, 128, 1], offset: ?>>, memref<32x32x128xf32> -> tensor<32x32x128xf32>
                %59 = linalg.fill ins(%cst_1 : f32) outs(%33 : tensor<32x32x128xf32>) -> tensor<32x32x128xf32>
                %60 = linalg.batch_matmul ins(%52, %58 : tensor<32x32x32xf32>, tensor<32x32x128xf32>) outs(%59 : tensor<32x32x128xf32>) -> tensor<32x32x128xf32>
                %61 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%60, %arg11, %55 : tensor<32x32x128xf32>, tensor<32x32x128xf32>, tensor<32x32xf32>) outs(%arg11 : tensor<32x32x128xf32>) {
                ^bb0(%in: f32, %in_10: f32, %in_11: f32, %out: f32):
                  %63 = arith.mulf %in_10, %in_11 : f32
                  %64 = arith.addf %in, %63 : f32
                  linalg.yield %64 : f32
                } -> tensor<32x32x128xf32>
                %62 = linalg.copy ins(%51 : tensor<32x32xf32>) outs(%arg9 : tensor<32x32xf32>) -> tensor<32x32xf32>
                affine.yield %62, %56, %61 : tensor<32x32xf32>, tensor<32x32xf32>, tensor<32x32x128xf32>
              }
              %42 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%41#2, %41#1 : tensor<32x32x128xf32>, tensor<32x32xf32>) outs(%29 : tensor<32x32x128xf32>) {
              ^bb0(%in: f32, %in_6: f32, %out: f32):
                %44 = arith.divf %in, %in_6 : f32
                linalg.yield %44 : f32
              } -> tensor<32x32x128xf32>
              %c0_4 = arith.constant 0 : index
              %43 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 524288 + d1 * 128 + d2)>(%34, %35, %c0_4)
              %reinterpret_cast_5 = memref.reinterpret_cast %arg3 to offset: [%43], sizes: [32, 32, 128], strides: [524288, 128, 1] : memref<32x4096x128xf32> to memref<32x32x128xf32, strided<[524288, 128, 1], offset: ?>>
              loom.copy_from_tensor %42, %reinterpret_cast_5 on @DRAM : tensor<32x32x128xf32>, memref<32x32x128xf32, strided<[524288, 128, 1], offset: ?>>
            } {loom.iter_type = #loom.iter_type<temporal>}
          } {loom.iter_type = #loom.iter_type<temporal>}
        } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @x}
      } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @y}
      return
    }
  }
  module attributes {loom.pass_name = "Materialize"} {
    func.func @attention__d1i0_d0i1__f10__d_d_h__BB32__BM32__BN32(%arg0: memref<32x128x4096xf32>, %arg1: memref<32x4096x128xf32>, %arg2: memref<32x4096x128xf32>, %arg3: memref<32x4096x128xf32>) {
      %cst = arith.constant 2.000000e+00 : f32
      %cst_0 = arith.constant 0.12751743074602467 : f64
      %cst_1 = arith.constant 0.000000e+00 : f32
      %cst_2 = arith.constant 1.000000e+00 : f32
      %cst_3 = arith.constant 0xFF800000 : f32
      %c32 = arith.constant 32 : index
      affine.parallel (%arg4) = (0) to (8) {
        affine.parallel (%arg5) = (0) to (8) {
          affine.for %arg6 = 0 to 16 {
            affine.for %arg7 = 0 to 1 {
              %12 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 8)>(%arg4, %arg7)
              %13 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 8)>(%arg5, %arg6)
              %14 = loom.alloc [32, 32, 128] on @L1 : memref<32x32x128xf32>
              %15 = loom.alloc [32, 32, 32] on @L1 : memref<32x32x32xf32>
              %16 = loom.init_tensor %15[32, 32, 32] : memref<32x32x32xf32> -> tensor<32x32x32xf32>
              %17 = loom.alloc [32, 128, 32] on @L1 : memref<32x128x32xf32>
              %18 = loom.alloc [32, 32] on @L1 : memref<32x32xf32>
              %19 = loom.init_tensor %18[32, 32] : memref<32x32xf32> -> tensor<32x32xf32>
              %20 = loom.alloc [32, 32] on @L1 : memref<32x32xf32>
              %21 = loom.init_tensor %20[32, 32] : memref<32x32xf32> -> tensor<32x32xf32>
              %22 = loom.alloc [32, 32] on @L1 : memref<32x32xf32>
              %23 = loom.init_tensor %22[32, 32] : memref<32x32xf32> -> tensor<32x32xf32>
              %24 = loom.alloc [32, 32] on @L1 : memref<32x32xf32>
              %25 = loom.init_tensor %24[32, 32] : memref<32x32xf32> -> tensor<32x32xf32>
              %26 = loom.alloc [32, 32] on @L1 : memref<32x32xf32>
              %27 = loom.init_tensor %26[32, 32] : memref<32x32xf32> -> tensor<32x32xf32>
              %28 = loom.alloc [32, 32, 128] on @L1 : memref<32x32x128xf32>
              %29 = loom.init_tensor %28[32, 32, 128] : memref<32x32x128xf32> -> tensor<32x32x128xf32>
              %30 = loom.alloc [32, 32, 128] on @L1 : memref<32x32x128xf32>
              %31 = loom.init_tensor %30[32, 32, 128] : memref<32x32x128xf32> -> tensor<32x32x128xf32>
              %32 = loom.alloc [32, 32, 128] on @L1 : memref<32x32x128xf32>
              %33 = loom.init_tensor %32[32, 32, 128] : memref<32x32x128xf32> -> tensor<32x32x128xf32>
              %34 = arith.muli %12, %c32 : index
              %35 = arith.muli %13, %c32 : index
              %c0 = arith.constant 0 : index
              %36 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 524288 + d1 * 128 + d2)>(%34, %35, %c0)
              %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%36], sizes: [32, 32, 128], strides: [524288, 128, 1] : memref<32x4096x128xf32> to memref<32x32x128xf32, strided<[524288, 128, 1], offset: ?>>
              %37 = loom.copy_to_tensor %reinterpret_cast, %28 on @L1, interconnect : [], broadcast : [1, 1] : memref<32x32x128xf32, strided<[524288, 128, 1], offset: ?>>, memref<32x32x128xf32> -> tensor<32x32x128xf32>
              %38 = linalg.fill ins(%cst_1 : f32) outs(%31 : tensor<32x32x128xf32>) -> tensor<32x32x128xf32>
              %39 = linalg.fill ins(%cst_2 : f32) outs(%19 : tensor<32x32xf32>) -> tensor<32x32xf32>
              %40 = linalg.fill ins(%cst_3 : f32) outs(%21 : tensor<32x32xf32>) -> tensor<32x32xf32>
              %41:3 = affine.for %arg8 = 0 to 128 iter_args(%arg9 = %40, %arg10 = %39, %arg11 = %38) -> (tensor<32x32xf32>, tensor<32x32xf32>, tensor<32x32x128xf32>) {
                %44 = arith.muli %arg8, %c32 : index
                %c0_6 = arith.constant 0 : index
                %45 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 524288 + d1 * 4096 + d2)>(%34, %c0_6, %44)
                %reinterpret_cast_7 = memref.reinterpret_cast %arg0 to offset: [%45], sizes: [32, 128, 32], strides: [524288, 4096, 1] : memref<32x128x4096xf32> to memref<32x128x32xf32, strided<[524288, 4096, 1], offset: ?>>
                %46 = loom.copy_to_tensor %reinterpret_cast_7, %17 on @L1, interconnect : [], broadcast : [1, 1] : memref<32x128x32xf32, strided<[524288, 4096, 1], offset: ?>>, memref<32x128x32xf32> -> tensor<32x128x32xf32>
                %47 = linalg.fill ins(%cst_1 : f32) outs(%16 : tensor<32x32x32xf32>) -> tensor<32x32x32xf32>
                %48 = linalg.batch_matmul ins(%37, %46 : tensor<32x32x128xf32>, tensor<32x128x32xf32>) outs(%47 : tensor<32x32x32xf32>) -> tensor<32x32x32xf32>
                %49 = linalg.fill ins(%cst_3 : f32) outs(%23 : tensor<32x32xf32>) -> tensor<32x32xf32>
                %50 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%48 : tensor<32x32x32xf32>) outs(%49 : tensor<32x32xf32>) {
                ^bb0(%in: f32, %out: f32):
                  %63 = arith.maximumf %in, %out : f32
                  linalg.yield %63 : f32
                } -> tensor<32x32xf32>
                %51 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%arg9, %50 : tensor<32x32xf32>, tensor<32x32xf32>) outs(%23 : tensor<32x32xf32>) {
                ^bb0(%in: f32, %in_10: f32, %out: f32):
                  %63 = arith.truncf %cst_0 : f64 to f32
                  %64 = arith.mulf %in_10, %63 : f32
                  %65 = arith.cmpf ogt, %in, %64 : f32
                  %66 = arith.select %65, %in, %64 : f32
                  linalg.yield %66 : f32
                } -> tensor<32x32xf32>
                %52 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%48, %51 : tensor<32x32x32xf32>, tensor<32x32xf32>) outs(%16 : tensor<32x32x32xf32>) {
                ^bb0(%in: f32, %in_10: f32, %out: f32):
                  %63 = arith.truncf %cst_0 : f64 to f32
                  %64 = arith.mulf %in, %63 : f32
                  %65 = arith.subf %64, %in_10 : f32
                  %66 = math.powf %cst, %65 : f32
                  linalg.yield %66 : f32
                } -> tensor<32x32x32xf32>
                %53 = linalg.fill ins(%cst_1 : f32) outs(%25 : tensor<32x32xf32>) -> tensor<32x32xf32>
                %54 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%52 : tensor<32x32x32xf32>) outs(%53 : tensor<32x32xf32>) {
                ^bb0(%in: f32, %out: f32):
                  %63 = arith.addf %in, %out : f32
                  linalg.yield %63 : f32
                } -> tensor<32x32xf32>
                %55 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%arg9, %51 : tensor<32x32xf32>, tensor<32x32xf32>) outs(%27 : tensor<32x32xf32>) {
                ^bb0(%in: f32, %in_10: f32, %out: f32):
                  %63 = arith.subf %in, %in_10 : f32
                  %64 = math.powf %cst, %63 : f32
                  linalg.yield %64 : f32
                } -> tensor<32x32xf32>
                %56 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%arg10, %55, %54 : tensor<32x32xf32>, tensor<32x32xf32>, tensor<32x32xf32>) outs(%arg10 : tensor<32x32xf32>) {
                ^bb0(%in: f32, %in_10: f32, %in_11: f32, %out: f32):
                  %63 = arith.mulf %in, %in_10 : f32
                  %64 = arith.addf %63, %in_11 : f32
                  linalg.yield %64 : f32
                } -> tensor<32x32xf32>
                %c0_8 = arith.constant 0 : index
                %57 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 524288 + d1 * 128 + d2)>(%34, %44, %c0_8)
                %reinterpret_cast_9 = memref.reinterpret_cast %arg1 to offset: [%57], sizes: [32, 32, 128], strides: [524288, 128, 1] : memref<32x4096x128xf32> to memref<32x32x128xf32, strided<[524288, 128, 1], offset: ?>>
                %58 = loom.copy_to_tensor %reinterpret_cast_9, %14 on @L1, interconnect : [@horizontal_links], broadcast : [1, 8] : memref<32x32x128xf32, strided<[524288, 128, 1], offset: ?>>, memref<32x32x128xf32> -> tensor<32x32x128xf32>
                %59 = linalg.fill ins(%cst_1 : f32) outs(%33 : tensor<32x32x128xf32>) -> tensor<32x32x128xf32>
                %60 = linalg.batch_matmul ins(%52, %58 : tensor<32x32x32xf32>, tensor<32x32x128xf32>) outs(%59 : tensor<32x32x128xf32>) -> tensor<32x32x128xf32>
                %61 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%60, %arg11, %55 : tensor<32x32x128xf32>, tensor<32x32x128xf32>, tensor<32x32xf32>) outs(%arg11 : tensor<32x32x128xf32>) {
                ^bb0(%in: f32, %in_10: f32, %in_11: f32, %out: f32):
                  %63 = arith.mulf %in_10, %in_11 : f32
                  %64 = arith.addf %in, %63 : f32
                  linalg.yield %64 : f32
                } -> tensor<32x32x128xf32>
                %62 = linalg.copy ins(%51 : tensor<32x32xf32>) outs(%arg9 : tensor<32x32xf32>) -> tensor<32x32xf32>
                affine.yield %62, %56, %61 : tensor<32x32xf32>, tensor<32x32xf32>, tensor<32x32x128xf32>
              }
              %42 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%41#2, %41#1 : tensor<32x32x128xf32>, tensor<32x32xf32>) outs(%29 : tensor<32x32x128xf32>) {
              ^bb0(%in: f32, %in_6: f32, %out: f32):
                %44 = arith.divf %in, %in_6 : f32
                linalg.yield %44 : f32
              } -> tensor<32x32x128xf32>
              %c0_4 = arith.constant 0 : index
              %43 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 524288 + d1 * 128 + d2)>(%34, %35, %c0_4)
              %reinterpret_cast_5 = memref.reinterpret_cast %arg3 to offset: [%43], sizes: [32, 32, 128], strides: [524288, 128, 1] : memref<32x4096x128xf32> to memref<32x32x128xf32, strided<[524288, 128, 1], offset: ?>>
              loom.copy_from_tensor %42, %reinterpret_cast_5 on @DRAM : tensor<32x32x128xf32>, memref<32x32x128xf32, strided<[524288, 128, 1], offset: ?>>
            } {loom.iter_type = #loom.iter_type<temporal>}
          } {loom.iter_type = #loom.iter_type<temporal>}
        } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @x}
      } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @y}
      return
    }
  }
  module attributes {loom.pass_name = "Materialize"} {
    func.func @attention__d1i0_d0i1__f10__d_h_d__BB32__BM32__BN32(%arg0: memref<32x128x4096xf32>, %arg1: memref<32x4096x128xf32>, %arg2: memref<32x4096x128xf32>, %arg3: memref<32x4096x128xf32>) {
      %cst = arith.constant 2.000000e+00 : f32
      %cst_0 = arith.constant 0.12751743074602467 : f64
      %cst_1 = arith.constant 0.000000e+00 : f32
      %cst_2 = arith.constant 1.000000e+00 : f32
      %cst_3 = arith.constant 0xFF800000 : f32
      %c32 = arith.constant 32 : index
      affine.parallel (%arg4) = (0) to (8) {
        affine.parallel (%arg5) = (0) to (8) {
          affine.for %arg6 = 0 to 16 {
            affine.for %arg7 = 0 to 1 {
              %12 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 8)>(%arg4, %arg7)
              %13 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 8)>(%arg5, %arg6)
              %14 = loom.alloc [32, 32, 128] on @L1 : memref<32x32x128xf32>
              %15 = loom.alloc [32, 32, 32] on @L1 : memref<32x32x32xf32>
              %16 = loom.init_tensor %15[32, 32, 32] : memref<32x32x32xf32> -> tensor<32x32x32xf32>
              %17 = loom.alloc [32, 128, 32] on @L1 : memref<32x128x32xf32>
              %18 = loom.alloc [32, 32] on @L1 : memref<32x32xf32>
              %19 = loom.init_tensor %18[32, 32] : memref<32x32xf32> -> tensor<32x32xf32>
              %20 = loom.alloc [32, 32] on @L1 : memref<32x32xf32>
              %21 = loom.init_tensor %20[32, 32] : memref<32x32xf32> -> tensor<32x32xf32>
              %22 = loom.alloc [32, 32] on @L1 : memref<32x32xf32>
              %23 = loom.init_tensor %22[32, 32] : memref<32x32xf32> -> tensor<32x32xf32>
              %24 = loom.alloc [32, 32] on @L1 : memref<32x32xf32>
              %25 = loom.init_tensor %24[32, 32] : memref<32x32xf32> -> tensor<32x32xf32>
              %26 = loom.alloc [32, 32] on @L1 : memref<32x32xf32>
              %27 = loom.init_tensor %26[32, 32] : memref<32x32xf32> -> tensor<32x32xf32>
              %28 = loom.alloc [32, 32, 128] on @L1 : memref<32x32x128xf32>
              %29 = loom.init_tensor %28[32, 32, 128] : memref<32x32x128xf32> -> tensor<32x32x128xf32>
              %30 = loom.alloc [32, 32, 128] on @L1 : memref<32x32x128xf32>
              %31 = loom.init_tensor %30[32, 32, 128] : memref<32x32x128xf32> -> tensor<32x32x128xf32>
              %32 = loom.alloc [32, 32, 128] on @L1 : memref<32x32x128xf32>
              %33 = loom.init_tensor %32[32, 32, 128] : memref<32x32x128xf32> -> tensor<32x32x128xf32>
              %34 = arith.muli %12, %c32 : index
              %35 = arith.muli %13, %c32 : index
              %c0 = arith.constant 0 : index
              %36 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 524288 + d1 * 128 + d2)>(%34, %35, %c0)
              %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%36], sizes: [32, 32, 128], strides: [524288, 128, 1] : memref<32x4096x128xf32> to memref<32x32x128xf32, strided<[524288, 128, 1], offset: ?>>
              %37 = loom.copy_to_tensor %reinterpret_cast, %28 on @L1, interconnect : [], broadcast : [1, 1] : memref<32x32x128xf32, strided<[524288, 128, 1], offset: ?>>, memref<32x32x128xf32> -> tensor<32x32x128xf32>
              %38 = linalg.fill ins(%cst_1 : f32) outs(%31 : tensor<32x32x128xf32>) -> tensor<32x32x128xf32>
              %39 = linalg.fill ins(%cst_2 : f32) outs(%19 : tensor<32x32xf32>) -> tensor<32x32xf32>
              %40 = linalg.fill ins(%cst_3 : f32) outs(%21 : tensor<32x32xf32>) -> tensor<32x32xf32>
              %41:3 = affine.for %arg8 = 0 to 128 iter_args(%arg9 = %40, %arg10 = %39, %arg11 = %38) -> (tensor<32x32xf32>, tensor<32x32xf32>, tensor<32x32x128xf32>) {
                %44 = arith.muli %arg8, %c32 : index
                %c0_6 = arith.constant 0 : index
                %45 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 524288 + d1 * 4096 + d2)>(%34, %c0_6, %44)
                %reinterpret_cast_7 = memref.reinterpret_cast %arg0 to offset: [%45], sizes: [32, 128, 32], strides: [524288, 4096, 1] : memref<32x128x4096xf32> to memref<32x128x32xf32, strided<[524288, 4096, 1], offset: ?>>
                %46 = loom.copy_to_tensor %reinterpret_cast_7, %17 on @L1, interconnect : [@horizontal_links], broadcast : [1, 8] : memref<32x128x32xf32, strided<[524288, 4096, 1], offset: ?>>, memref<32x128x32xf32> -> tensor<32x128x32xf32>
                %47 = linalg.fill ins(%cst_1 : f32) outs(%16 : tensor<32x32x32xf32>) -> tensor<32x32x32xf32>
                %48 = linalg.batch_matmul ins(%37, %46 : tensor<32x32x128xf32>, tensor<32x128x32xf32>) outs(%47 : tensor<32x32x32xf32>) -> tensor<32x32x32xf32>
                %49 = linalg.fill ins(%cst_3 : f32) outs(%23 : tensor<32x32xf32>) -> tensor<32x32xf32>
                %50 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%48 : tensor<32x32x32xf32>) outs(%49 : tensor<32x32xf32>) {
                ^bb0(%in: f32, %out: f32):
                  %63 = arith.maximumf %in, %out : f32
                  linalg.yield %63 : f32
                } -> tensor<32x32xf32>
                %51 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%arg9, %50 : tensor<32x32xf32>, tensor<32x32xf32>) outs(%23 : tensor<32x32xf32>) {
                ^bb0(%in: f32, %in_10: f32, %out: f32):
                  %63 = arith.truncf %cst_0 : f64 to f32
                  %64 = arith.mulf %in_10, %63 : f32
                  %65 = arith.cmpf ogt, %in, %64 : f32
                  %66 = arith.select %65, %in, %64 : f32
                  linalg.yield %66 : f32
                } -> tensor<32x32xf32>
                %52 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%48, %51 : tensor<32x32x32xf32>, tensor<32x32xf32>) outs(%16 : tensor<32x32x32xf32>) {
                ^bb0(%in: f32, %in_10: f32, %out: f32):
                  %63 = arith.truncf %cst_0 : f64 to f32
                  %64 = arith.mulf %in, %63 : f32
                  %65 = arith.subf %64, %in_10 : f32
                  %66 = math.powf %cst, %65 : f32
                  linalg.yield %66 : f32
                } -> tensor<32x32x32xf32>
                %53 = linalg.fill ins(%cst_1 : f32) outs(%25 : tensor<32x32xf32>) -> tensor<32x32xf32>
                %54 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%52 : tensor<32x32x32xf32>) outs(%53 : tensor<32x32xf32>) {
                ^bb0(%in: f32, %out: f32):
                  %63 = arith.addf %in, %out : f32
                  linalg.yield %63 : f32
                } -> tensor<32x32xf32>
                %55 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%arg9, %51 : tensor<32x32xf32>, tensor<32x32xf32>) outs(%27 : tensor<32x32xf32>) {
                ^bb0(%in: f32, %in_10: f32, %out: f32):
                  %63 = arith.subf %in, %in_10 : f32
                  %64 = math.powf %cst, %63 : f32
                  linalg.yield %64 : f32
                } -> tensor<32x32xf32>
                %56 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%arg10, %55, %54 : tensor<32x32xf32>, tensor<32x32xf32>, tensor<32x32xf32>) outs(%arg10 : tensor<32x32xf32>) {
                ^bb0(%in: f32, %in_10: f32, %in_11: f32, %out: f32):
                  %63 = arith.mulf %in, %in_10 : f32
                  %64 = arith.addf %63, %in_11 : f32
                  linalg.yield %64 : f32
                } -> tensor<32x32xf32>
                %c0_8 = arith.constant 0 : index
                %57 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 524288 + d1 * 128 + d2)>(%34, %44, %c0_8)
                %reinterpret_cast_9 = memref.reinterpret_cast %arg1 to offset: [%57], sizes: [32, 32, 128], strides: [524288, 128, 1] : memref<32x4096x128xf32> to memref<32x32x128xf32, strided<[524288, 128, 1], offset: ?>>
                %58 = loom.copy_to_tensor %reinterpret_cast_9, %14 on @L1, interconnect : [], broadcast : [1, 1] : memref<32x32x128xf32, strided<[524288, 128, 1], offset: ?>>, memref<32x32x128xf32> -> tensor<32x32x128xf32>
                %59 = linalg.fill ins(%cst_1 : f32) outs(%33 : tensor<32x32x128xf32>) -> tensor<32x32x128xf32>
                %60 = linalg.batch_matmul ins(%52, %58 : tensor<32x32x32xf32>, tensor<32x32x128xf32>) outs(%59 : tensor<32x32x128xf32>) -> tensor<32x32x128xf32>
                %61 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%60, %arg11, %55 : tensor<32x32x128xf32>, tensor<32x32x128xf32>, tensor<32x32xf32>) outs(%arg11 : tensor<32x32x128xf32>) {
                ^bb0(%in: f32, %in_10: f32, %in_11: f32, %out: f32):
                  %63 = arith.mulf %in_10, %in_11 : f32
                  %64 = arith.addf %in, %63 : f32
                  linalg.yield %64 : f32
                } -> tensor<32x32x128xf32>
                %62 = linalg.copy ins(%51 : tensor<32x32xf32>) outs(%arg9 : tensor<32x32xf32>) -> tensor<32x32xf32>
                affine.yield %62, %56, %61 : tensor<32x32xf32>, tensor<32x32xf32>, tensor<32x32x128xf32>
              }
              %42 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%41#2, %41#1 : tensor<32x32x128xf32>, tensor<32x32xf32>) outs(%29 : tensor<32x32x128xf32>) {
              ^bb0(%in: f32, %in_6: f32, %out: f32):
                %44 = arith.divf %in, %in_6 : f32
                linalg.yield %44 : f32
              } -> tensor<32x32x128xf32>
              %c0_4 = arith.constant 0 : index
              %43 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 524288 + d1 * 128 + d2)>(%34, %35, %c0_4)
              %reinterpret_cast_5 = memref.reinterpret_cast %arg3 to offset: [%43], sizes: [32, 32, 128], strides: [524288, 128, 1] : memref<32x4096x128xf32> to memref<32x32x128xf32, strided<[524288, 128, 1], offset: ?>>
              loom.copy_from_tensor %42, %reinterpret_cast_5 on @DRAM : tensor<32x32x128xf32>, memref<32x32x128xf32, strided<[524288, 128, 1], offset: ?>>
            } {loom.iter_type = #loom.iter_type<temporal>}
          } {loom.iter_type = #loom.iter_type<temporal>}
        } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @x}
      } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @y}
      return
    }
  }
  module attributes {loom.pass_name = "Materialize"} {
    func.func @attention__d1i0_d0i1__f10__d_h_h__BB32__BM32__BN32(%arg0: memref<32x128x4096xf32>, %arg1: memref<32x4096x128xf32>, %arg2: memref<32x4096x128xf32>, %arg3: memref<32x4096x128xf32>) {
      %cst = arith.constant 2.000000e+00 : f32
      %cst_0 = arith.constant 0.12751743074602467 : f64
      %cst_1 = arith.constant 0.000000e+00 : f32
      %cst_2 = arith.constant 1.000000e+00 : f32
      %cst_3 = arith.constant 0xFF800000 : f32
      %c32 = arith.constant 32 : index
      affine.parallel (%arg4) = (0) to (8) {
        affine.parallel (%arg5) = (0) to (8) {
          affine.for %arg6 = 0 to 16 {
            affine.for %arg7 = 0 to 1 {
              %12 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 8)>(%arg4, %arg7)
              %13 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 8)>(%arg5, %arg6)
              %14 = loom.alloc [32, 32, 128] on @L1 : memref<32x32x128xf32>
              %15 = loom.alloc [32, 32, 32] on @L1 : memref<32x32x32xf32>
              %16 = loom.init_tensor %15[32, 32, 32] : memref<32x32x32xf32> -> tensor<32x32x32xf32>
              %17 = loom.alloc [32, 128, 32] on @L1 : memref<32x128x32xf32>
              %18 = loom.alloc [32, 32] on @L1 : memref<32x32xf32>
              %19 = loom.init_tensor %18[32, 32] : memref<32x32xf32> -> tensor<32x32xf32>
              %20 = loom.alloc [32, 32] on @L1 : memref<32x32xf32>
              %21 = loom.init_tensor %20[32, 32] : memref<32x32xf32> -> tensor<32x32xf32>
              %22 = loom.alloc [32, 32] on @L1 : memref<32x32xf32>
              %23 = loom.init_tensor %22[32, 32] : memref<32x32xf32> -> tensor<32x32xf32>
              %24 = loom.alloc [32, 32] on @L1 : memref<32x32xf32>
              %25 = loom.init_tensor %24[32, 32] : memref<32x32xf32> -> tensor<32x32xf32>
              %26 = loom.alloc [32, 32] on @L1 : memref<32x32xf32>
              %27 = loom.init_tensor %26[32, 32] : memref<32x32xf32> -> tensor<32x32xf32>
              %28 = loom.alloc [32, 32, 128] on @L1 : memref<32x32x128xf32>
              %29 = loom.init_tensor %28[32, 32, 128] : memref<32x32x128xf32> -> tensor<32x32x128xf32>
              %30 = loom.alloc [32, 32, 128] on @L1 : memref<32x32x128xf32>
              %31 = loom.init_tensor %30[32, 32, 128] : memref<32x32x128xf32> -> tensor<32x32x128xf32>
              %32 = loom.alloc [32, 32, 128] on @L1 : memref<32x32x128xf32>
              %33 = loom.init_tensor %32[32, 32, 128] : memref<32x32x128xf32> -> tensor<32x32x128xf32>
              %34 = arith.muli %12, %c32 : index
              %35 = arith.muli %13, %c32 : index
              %c0 = arith.constant 0 : index
              %36 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 524288 + d1 * 128 + d2)>(%34, %35, %c0)
              %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%36], sizes: [32, 32, 128], strides: [524288, 128, 1] : memref<32x4096x128xf32> to memref<32x32x128xf32, strided<[524288, 128, 1], offset: ?>>
              %37 = loom.copy_to_tensor %reinterpret_cast, %28 on @L1, interconnect : [], broadcast : [1, 1] : memref<32x32x128xf32, strided<[524288, 128, 1], offset: ?>>, memref<32x32x128xf32> -> tensor<32x32x128xf32>
              %38 = linalg.fill ins(%cst_1 : f32) outs(%31 : tensor<32x32x128xf32>) -> tensor<32x32x128xf32>
              %39 = linalg.fill ins(%cst_2 : f32) outs(%19 : tensor<32x32xf32>) -> tensor<32x32xf32>
              %40 = linalg.fill ins(%cst_3 : f32) outs(%21 : tensor<32x32xf32>) -> tensor<32x32xf32>
              %41:3 = affine.for %arg8 = 0 to 128 iter_args(%arg9 = %40, %arg10 = %39, %arg11 = %38) -> (tensor<32x32xf32>, tensor<32x32xf32>, tensor<32x32x128xf32>) {
                %44 = arith.muli %arg8, %c32 : index
                %c0_6 = arith.constant 0 : index
                %45 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 524288 + d1 * 4096 + d2)>(%34, %c0_6, %44)
                %reinterpret_cast_7 = memref.reinterpret_cast %arg0 to offset: [%45], sizes: [32, 128, 32], strides: [524288, 4096, 1] : memref<32x128x4096xf32> to memref<32x128x32xf32, strided<[524288, 4096, 1], offset: ?>>
                %46 = loom.copy_to_tensor %reinterpret_cast_7, %17 on @L1, interconnect : [@horizontal_links], broadcast : [1, 8] : memref<32x128x32xf32, strided<[524288, 4096, 1], offset: ?>>, memref<32x128x32xf32> -> tensor<32x128x32xf32>
                %47 = linalg.fill ins(%cst_1 : f32) outs(%16 : tensor<32x32x32xf32>) -> tensor<32x32x32xf32>
                %48 = linalg.batch_matmul ins(%37, %46 : tensor<32x32x128xf32>, tensor<32x128x32xf32>) outs(%47 : tensor<32x32x32xf32>) -> tensor<32x32x32xf32>
                %49 = linalg.fill ins(%cst_3 : f32) outs(%23 : tensor<32x32xf32>) -> tensor<32x32xf32>
                %50 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%48 : tensor<32x32x32xf32>) outs(%49 : tensor<32x32xf32>) {
                ^bb0(%in: f32, %out: f32):
                  %63 = arith.maximumf %in, %out : f32
                  linalg.yield %63 : f32
                } -> tensor<32x32xf32>
                %51 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%arg9, %50 : tensor<32x32xf32>, tensor<32x32xf32>) outs(%23 : tensor<32x32xf32>) {
                ^bb0(%in: f32, %in_10: f32, %out: f32):
                  %63 = arith.truncf %cst_0 : f64 to f32
                  %64 = arith.mulf %in_10, %63 : f32
                  %65 = arith.cmpf ogt, %in, %64 : f32
                  %66 = arith.select %65, %in, %64 : f32
                  linalg.yield %66 : f32
                } -> tensor<32x32xf32>
                %52 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%48, %51 : tensor<32x32x32xf32>, tensor<32x32xf32>) outs(%16 : tensor<32x32x32xf32>) {
                ^bb0(%in: f32, %in_10: f32, %out: f32):
                  %63 = arith.truncf %cst_0 : f64 to f32
                  %64 = arith.mulf %in, %63 : f32
                  %65 = arith.subf %64, %in_10 : f32
                  %66 = math.powf %cst, %65 : f32
                  linalg.yield %66 : f32
                } -> tensor<32x32x32xf32>
                %53 = linalg.fill ins(%cst_1 : f32) outs(%25 : tensor<32x32xf32>) -> tensor<32x32xf32>
                %54 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%52 : tensor<32x32x32xf32>) outs(%53 : tensor<32x32xf32>) {
                ^bb0(%in: f32, %out: f32):
                  %63 = arith.addf %in, %out : f32
                  linalg.yield %63 : f32
                } -> tensor<32x32xf32>
                %55 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%arg9, %51 : tensor<32x32xf32>, tensor<32x32xf32>) outs(%27 : tensor<32x32xf32>) {
                ^bb0(%in: f32, %in_10: f32, %out: f32):
                  %63 = arith.subf %in, %in_10 : f32
                  %64 = math.powf %cst, %63 : f32
                  linalg.yield %64 : f32
                } -> tensor<32x32xf32>
                %56 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%arg10, %55, %54 : tensor<32x32xf32>, tensor<32x32xf32>, tensor<32x32xf32>) outs(%arg10 : tensor<32x32xf32>) {
                ^bb0(%in: f32, %in_10: f32, %in_11: f32, %out: f32):
                  %63 = arith.mulf %in, %in_10 : f32
                  %64 = arith.addf %63, %in_11 : f32
                  linalg.yield %64 : f32
                } -> tensor<32x32xf32>
                %c0_8 = arith.constant 0 : index
                %57 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 524288 + d1 * 128 + d2)>(%34, %44, %c0_8)
                %reinterpret_cast_9 = memref.reinterpret_cast %arg1 to offset: [%57], sizes: [32, 32, 128], strides: [524288, 128, 1] : memref<32x4096x128xf32> to memref<32x32x128xf32, strided<[524288, 128, 1], offset: ?>>
                %58 = loom.copy_to_tensor %reinterpret_cast_9, %14 on @L1, interconnect : [@horizontal_links], broadcast : [1, 8] : memref<32x32x128xf32, strided<[524288, 128, 1], offset: ?>>, memref<32x32x128xf32> -> tensor<32x32x128xf32>
                %59 = linalg.fill ins(%cst_1 : f32) outs(%33 : tensor<32x32x128xf32>) -> tensor<32x32x128xf32>
                %60 = linalg.batch_matmul ins(%52, %58 : tensor<32x32x32xf32>, tensor<32x32x128xf32>) outs(%59 : tensor<32x32x128xf32>) -> tensor<32x32x128xf32>
                %61 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%60, %arg11, %55 : tensor<32x32x128xf32>, tensor<32x32x128xf32>, tensor<32x32xf32>) outs(%arg11 : tensor<32x32x128xf32>) {
                ^bb0(%in: f32, %in_10: f32, %in_11: f32, %out: f32):
                  %63 = arith.mulf %in_10, %in_11 : f32
                  %64 = arith.addf %in, %63 : f32
                  linalg.yield %64 : f32
                } -> tensor<32x32x128xf32>
                %62 = linalg.copy ins(%51 : tensor<32x32xf32>) outs(%arg9 : tensor<32x32xf32>) -> tensor<32x32xf32>
                affine.yield %62, %56, %61 : tensor<32x32xf32>, tensor<32x32xf32>, tensor<32x32x128xf32>
              }
              %42 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%41#2, %41#1 : tensor<32x32x128xf32>, tensor<32x32xf32>) outs(%29 : tensor<32x32x128xf32>) {
              ^bb0(%in: f32, %in_6: f32, %out: f32):
                %44 = arith.divf %in, %in_6 : f32
                linalg.yield %44 : f32
              } -> tensor<32x32x128xf32>
              %c0_4 = arith.constant 0 : index
              %43 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 524288 + d1 * 128 + d2)>(%34, %35, %c0_4)
              %reinterpret_cast_5 = memref.reinterpret_cast %arg3 to offset: [%43], sizes: [32, 32, 128], strides: [524288, 128, 1] : memref<32x4096x128xf32> to memref<32x32x128xf32, strided<[524288, 128, 1], offset: ?>>
              loom.copy_from_tensor %42, %reinterpret_cast_5 on @DRAM : tensor<32x32x128xf32>, memref<32x32x128xf32, strided<[524288, 128, 1], offset: ?>>
            } {loom.iter_type = #loom.iter_type<temporal>}
          } {loom.iter_type = #loom.iter_type<temporal>}
        } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @x}
      } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @y}
      return
    }
  }
  module attributes {loom.pass_name = "Materialize"} {
    func.func @attention__d0i1_d1i1__f01__d_d_d__BB32__BM32__BN32(%arg0: memref<32x128x4096xf32>, %arg1: memref<32x4096x128xf32>, %arg2: memref<32x4096x128xf32>, %arg3: memref<32x4096x128xf32>) {
      %cst = arith.constant 2.000000e+00 : f32
      %cst_0 = arith.constant 0.12751743074602467 : f64
      %cst_1 = arith.constant 0.000000e+00 : f32
      %cst_2 = arith.constant 1.000000e+00 : f32
      %cst_3 = arith.constant 0xFF800000 : f32
      %c32 = arith.constant 32 : index
      affine.parallel (%arg4) = (0) to (8) {
        affine.parallel (%arg5) = (0) to (8) {
          affine.for %arg6 = 0 to 1 {
            affine.for %arg7 = 0 to 2 {
              %12 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 8 + d1 + d2 * 64)>(%arg4, %arg5, %arg7)
              %13 = loom.alloc [32, 32, 128] on @L1 : memref<32x32x128xf32>
              %14 = loom.alloc [32, 32, 32] on @L1 : memref<32x32x32xf32>
              %15 = loom.init_tensor %14[32, 32, 32] : memref<32x32x32xf32> -> tensor<32x32x32xf32>
              %16 = loom.alloc [32, 128, 32] on @L1 : memref<32x128x32xf32>
              %17 = loom.alloc [32, 32] on @L1 : memref<32x32xf32>
              %18 = loom.init_tensor %17[32, 32] : memref<32x32xf32> -> tensor<32x32xf32>
              %19 = loom.alloc [32, 32] on @L1 : memref<32x32xf32>
              %20 = loom.init_tensor %19[32, 32] : memref<32x32xf32> -> tensor<32x32xf32>
              %21 = loom.alloc [32, 32] on @L1 : memref<32x32xf32>
              %22 = loom.init_tensor %21[32, 32] : memref<32x32xf32> -> tensor<32x32xf32>
              %23 = loom.alloc [32, 32] on @L1 : memref<32x32xf32>
              %24 = loom.init_tensor %23[32, 32] : memref<32x32xf32> -> tensor<32x32xf32>
              %25 = loom.alloc [32, 32] on @L1 : memref<32x32xf32>
              %26 = loom.init_tensor %25[32, 32] : memref<32x32xf32> -> tensor<32x32xf32>
              %27 = loom.alloc [32, 32, 128] on @L1 : memref<32x32x128xf32>
              %28 = loom.init_tensor %27[32, 32, 128] : memref<32x32x128xf32> -> tensor<32x32x128xf32>
              %29 = loom.alloc [32, 32, 128] on @L1 : memref<32x32x128xf32>
              %30 = loom.init_tensor %29[32, 32, 128] : memref<32x32x128xf32> -> tensor<32x32x128xf32>
              %31 = loom.alloc [32, 32, 128] on @L1 : memref<32x32x128xf32>
              %32 = loom.init_tensor %31[32, 32, 128] : memref<32x32x128xf32> -> tensor<32x32x128xf32>
              %33 = arith.muli %arg6, %c32 : index
              %34 = arith.muli %12, %c32 : index
              %c0 = arith.constant 0 : index
              %35 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 524288 + d1 * 128 + d2)>(%33, %34, %c0)
              %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%35], sizes: [32, 32, 128], strides: [524288, 128, 1] : memref<32x4096x128xf32> to memref<32x32x128xf32, strided<[524288, 128, 1], offset: ?>>
              %36 = loom.copy_to_tensor %reinterpret_cast, %27 on @L1, interconnect : [], broadcast : [1, 1] : memref<32x32x128xf32, strided<[524288, 128, 1], offset: ?>>, memref<32x32x128xf32> -> tensor<32x32x128xf32>
              %37 = linalg.fill ins(%cst_1 : f32) outs(%30 : tensor<32x32x128xf32>) -> tensor<32x32x128xf32>
              %38 = linalg.fill ins(%cst_2 : f32) outs(%18 : tensor<32x32xf32>) -> tensor<32x32xf32>
              %39 = linalg.fill ins(%cst_3 : f32) outs(%20 : tensor<32x32xf32>) -> tensor<32x32xf32>
              %40:3 = affine.for %arg8 = 0 to 128 iter_args(%arg9 = %39, %arg10 = %38, %arg11 = %37) -> (tensor<32x32xf32>, tensor<32x32xf32>, tensor<32x32x128xf32>) {
                %43 = arith.muli %arg8, %c32 : index
                %c0_6 = arith.constant 0 : index
                %44 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 524288 + d1 * 4096 + d2)>(%33, %c0_6, %43)
                %reinterpret_cast_7 = memref.reinterpret_cast %arg0 to offset: [%44], sizes: [32, 128, 32], strides: [524288, 4096, 1] : memref<32x128x4096xf32> to memref<32x128x32xf32, strided<[524288, 4096, 1], offset: ?>>
                %45 = loom.copy_to_tensor %reinterpret_cast_7, %16 on @L1, interconnect : [], broadcast : [1, 1] : memref<32x128x32xf32, strided<[524288, 4096, 1], offset: ?>>, memref<32x128x32xf32> -> tensor<32x128x32xf32>
                %46 = linalg.fill ins(%cst_1 : f32) outs(%15 : tensor<32x32x32xf32>) -> tensor<32x32x32xf32>
                %47 = linalg.batch_matmul ins(%36, %45 : tensor<32x32x128xf32>, tensor<32x128x32xf32>) outs(%46 : tensor<32x32x32xf32>) -> tensor<32x32x32xf32>
                %48 = linalg.fill ins(%cst_3 : f32) outs(%22 : tensor<32x32xf32>) -> tensor<32x32xf32>
                %49 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%47 : tensor<32x32x32xf32>) outs(%48 : tensor<32x32xf32>) {
                ^bb0(%in: f32, %out: f32):
                  %62 = arith.maximumf %in, %out : f32
                  linalg.yield %62 : f32
                } -> tensor<32x32xf32>
                %50 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%arg9, %49 : tensor<32x32xf32>, tensor<32x32xf32>) outs(%22 : tensor<32x32xf32>) {
                ^bb0(%in: f32, %in_10: f32, %out: f32):
                  %62 = arith.truncf %cst_0 : f64 to f32
                  %63 = arith.mulf %in_10, %62 : f32
                  %64 = arith.cmpf ogt, %in, %63 : f32
                  %65 = arith.select %64, %in, %63 : f32
                  linalg.yield %65 : f32
                } -> tensor<32x32xf32>
                %51 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%47, %50 : tensor<32x32x32xf32>, tensor<32x32xf32>) outs(%15 : tensor<32x32x32xf32>) {
                ^bb0(%in: f32, %in_10: f32, %out: f32):
                  %62 = arith.truncf %cst_0 : f64 to f32
                  %63 = arith.mulf %in, %62 : f32
                  %64 = arith.subf %63, %in_10 : f32
                  %65 = math.powf %cst, %64 : f32
                  linalg.yield %65 : f32
                } -> tensor<32x32x32xf32>
                %52 = linalg.fill ins(%cst_1 : f32) outs(%24 : tensor<32x32xf32>) -> tensor<32x32xf32>
                %53 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%51 : tensor<32x32x32xf32>) outs(%52 : tensor<32x32xf32>) {
                ^bb0(%in: f32, %out: f32):
                  %62 = arith.addf %in, %out : f32
                  linalg.yield %62 : f32
                } -> tensor<32x32xf32>
                %54 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%arg9, %50 : tensor<32x32xf32>, tensor<32x32xf32>) outs(%26 : tensor<32x32xf32>) {
                ^bb0(%in: f32, %in_10: f32, %out: f32):
                  %62 = arith.subf %in, %in_10 : f32
                  %63 = math.powf %cst, %62 : f32
                  linalg.yield %63 : f32
                } -> tensor<32x32xf32>
                %55 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%arg10, %54, %53 : tensor<32x32xf32>, tensor<32x32xf32>, tensor<32x32xf32>) outs(%arg10 : tensor<32x32xf32>) {
                ^bb0(%in: f32, %in_10: f32, %in_11: f32, %out: f32):
                  %62 = arith.mulf %in, %in_10 : f32
                  %63 = arith.addf %62, %in_11 : f32
                  linalg.yield %63 : f32
                } -> tensor<32x32xf32>
                %c0_8 = arith.constant 0 : index
                %56 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 524288 + d1 * 128 + d2)>(%33, %43, %c0_8)
                %reinterpret_cast_9 = memref.reinterpret_cast %arg1 to offset: [%56], sizes: [32, 32, 128], strides: [524288, 128, 1] : memref<32x4096x128xf32> to memref<32x32x128xf32, strided<[524288, 128, 1], offset: ?>>
                %57 = loom.copy_to_tensor %reinterpret_cast_9, %13 on @L1, interconnect : [], broadcast : [1, 1] : memref<32x32x128xf32, strided<[524288, 128, 1], offset: ?>>, memref<32x32x128xf32> -> tensor<32x32x128xf32>
                %58 = linalg.fill ins(%cst_1 : f32) outs(%32 : tensor<32x32x128xf32>) -> tensor<32x32x128xf32>
                %59 = linalg.batch_matmul ins(%51, %57 : tensor<32x32x32xf32>, tensor<32x32x128xf32>) outs(%58 : tensor<32x32x128xf32>) -> tensor<32x32x128xf32>
                %60 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%59, %arg11, %54 : tensor<32x32x128xf32>, tensor<32x32x128xf32>, tensor<32x32xf32>) outs(%arg11 : tensor<32x32x128xf32>) {
                ^bb0(%in: f32, %in_10: f32, %in_11: f32, %out: f32):
                  %62 = arith.mulf %in_10, %in_11 : f32
                  %63 = arith.addf %in, %62 : f32
                  linalg.yield %63 : f32
                } -> tensor<32x32x128xf32>
                %61 = linalg.copy ins(%50 : tensor<32x32xf32>) outs(%arg9 : tensor<32x32xf32>) -> tensor<32x32xf32>
                affine.yield %61, %55, %60 : tensor<32x32xf32>, tensor<32x32xf32>, tensor<32x32x128xf32>
              }
              %41 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%40#2, %40#1 : tensor<32x32x128xf32>, tensor<32x32xf32>) outs(%28 : tensor<32x32x128xf32>) {
              ^bb0(%in: f32, %in_6: f32, %out: f32):
                %43 = arith.divf %in, %in_6 : f32
                linalg.yield %43 : f32
              } -> tensor<32x32x128xf32>
              %c0_4 = arith.constant 0 : index
              %42 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 524288 + d1 * 128 + d2)>(%33, %34, %c0_4)
              %reinterpret_cast_5 = memref.reinterpret_cast %arg3 to offset: [%42], sizes: [32, 32, 128], strides: [524288, 128, 1] : memref<32x4096x128xf32> to memref<32x32x128xf32, strided<[524288, 128, 1], offset: ?>>
              loom.copy_from_tensor %41, %reinterpret_cast_5 on @DRAM : tensor<32x32x128xf32>, memref<32x32x128xf32, strided<[524288, 128, 1], offset: ?>>
            } {loom.iter_type = #loom.iter_type<temporal>}
          } {loom.iter_type = #loom.iter_type<temporal>}
        } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @y}
      } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @x}
      return
    }
  }
  module attributes {loom.pass_name = "Materialize"} {
    func.func @attention__d0i1_d1i1__f01__d_d_a__BB32__BM32__BN32(%arg0: memref<32x128x4096xf32>, %arg1: memref<32x4096x128xf32>, %arg2: memref<32x4096x128xf32>, %arg3: memref<32x4096x128xf32>) {
      %cst = arith.constant 2.000000e+00 : f32
      %cst_0 = arith.constant 0.12751743074602467 : f64
      %cst_1 = arith.constant 0.000000e+00 : f32
      %cst_2 = arith.constant 1.000000e+00 : f32
      %cst_3 = arith.constant 0xFF800000 : f32
      %c32 = arith.constant 32 : index
      affine.parallel (%arg4) = (0) to (8) {
        affine.parallel (%arg5) = (0) to (8) {
          affine.for %arg6 = 0 to 1 {
            affine.for %arg7 = 0 to 2 {
              %12 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 8 + d1 + d2 * 64)>(%arg4, %arg5, %arg7)
              %13 = loom.alloc [32, 32, 128] on @L1 : memref<32x32x128xf32>
              %14 = loom.alloc [32, 32, 32] on @L1 : memref<32x32x32xf32>
              %15 = loom.init_tensor %14[32, 32, 32] : memref<32x32x32xf32> -> tensor<32x32x32xf32>
              %16 = loom.alloc [32, 128, 32] on @L1 : memref<32x128x32xf32>
              %17 = loom.alloc [32, 32] on @L1 : memref<32x32xf32>
              %18 = loom.init_tensor %17[32, 32] : memref<32x32xf32> -> tensor<32x32xf32>
              %19 = loom.alloc [32, 32] on @L1 : memref<32x32xf32>
              %20 = loom.init_tensor %19[32, 32] : memref<32x32xf32> -> tensor<32x32xf32>
              %21 = loom.alloc [32, 32] on @L1 : memref<32x32xf32>
              %22 = loom.init_tensor %21[32, 32] : memref<32x32xf32> -> tensor<32x32xf32>
              %23 = loom.alloc [32, 32] on @L1 : memref<32x32xf32>
              %24 = loom.init_tensor %23[32, 32] : memref<32x32xf32> -> tensor<32x32xf32>
              %25 = loom.alloc [32, 32] on @L1 : memref<32x32xf32>
              %26 = loom.init_tensor %25[32, 32] : memref<32x32xf32> -> tensor<32x32xf32>
              %27 = loom.alloc [32, 32, 128] on @L1 : memref<32x32x128xf32>
              %28 = loom.init_tensor %27[32, 32, 128] : memref<32x32x128xf32> -> tensor<32x32x128xf32>
              %29 = loom.alloc [32, 32, 128] on @L1 : memref<32x32x128xf32>
              %30 = loom.init_tensor %29[32, 32, 128] : memref<32x32x128xf32> -> tensor<32x32x128xf32>
              %31 = loom.alloc [32, 32, 128] on @L1 : memref<32x32x128xf32>
              %32 = loom.init_tensor %31[32, 32, 128] : memref<32x32x128xf32> -> tensor<32x32x128xf32>
              %33 = arith.muli %arg6, %c32 : index
              %34 = arith.muli %12, %c32 : index
              %c0 = arith.constant 0 : index
              %35 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 524288 + d1 * 128 + d2)>(%33, %34, %c0)
              %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%35], sizes: [32, 32, 128], strides: [524288, 128, 1] : memref<32x4096x128xf32> to memref<32x32x128xf32, strided<[524288, 128, 1], offset: ?>>
              %36 = loom.copy_to_tensor %reinterpret_cast, %27 on @L1, interconnect : [], broadcast : [1, 1] : memref<32x32x128xf32, strided<[524288, 128, 1], offset: ?>>, memref<32x32x128xf32> -> tensor<32x32x128xf32>
              %37 = linalg.fill ins(%cst_1 : f32) outs(%30 : tensor<32x32x128xf32>) -> tensor<32x32x128xf32>
              %38 = linalg.fill ins(%cst_2 : f32) outs(%18 : tensor<32x32xf32>) -> tensor<32x32xf32>
              %39 = linalg.fill ins(%cst_3 : f32) outs(%20 : tensor<32x32xf32>) -> tensor<32x32xf32>
              %40:3 = affine.for %arg8 = 0 to 128 iter_args(%arg9 = %39, %arg10 = %38, %arg11 = %37) -> (tensor<32x32xf32>, tensor<32x32xf32>, tensor<32x32x128xf32>) {
                %43 = arith.muli %arg8, %c32 : index
                %c0_6 = arith.constant 0 : index
                %44 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 524288 + d1 * 4096 + d2)>(%33, %c0_6, %43)
                %reinterpret_cast_7 = memref.reinterpret_cast %arg0 to offset: [%44], sizes: [32, 128, 32], strides: [524288, 4096, 1] : memref<32x128x4096xf32> to memref<32x128x32xf32, strided<[524288, 4096, 1], offset: ?>>
                %45 = loom.copy_to_tensor %reinterpret_cast_7, %16 on @L1, interconnect : [], broadcast : [1, 1] : memref<32x128x32xf32, strided<[524288, 4096, 1], offset: ?>>, memref<32x128x32xf32> -> tensor<32x128x32xf32>
                %46 = linalg.fill ins(%cst_1 : f32) outs(%15 : tensor<32x32x32xf32>) -> tensor<32x32x32xf32>
                %47 = linalg.batch_matmul ins(%36, %45 : tensor<32x32x128xf32>, tensor<32x128x32xf32>) outs(%46 : tensor<32x32x32xf32>) -> tensor<32x32x32xf32>
                %48 = linalg.fill ins(%cst_3 : f32) outs(%22 : tensor<32x32xf32>) -> tensor<32x32xf32>
                %49 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%47 : tensor<32x32x32xf32>) outs(%48 : tensor<32x32xf32>) {
                ^bb0(%in: f32, %out: f32):
                  %62 = arith.maximumf %in, %out : f32
                  linalg.yield %62 : f32
                } -> tensor<32x32xf32>
                %50 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%arg9, %49 : tensor<32x32xf32>, tensor<32x32xf32>) outs(%22 : tensor<32x32xf32>) {
                ^bb0(%in: f32, %in_10: f32, %out: f32):
                  %62 = arith.truncf %cst_0 : f64 to f32
                  %63 = arith.mulf %in_10, %62 : f32
                  %64 = arith.cmpf ogt, %in, %63 : f32
                  %65 = arith.select %64, %in, %63 : f32
                  linalg.yield %65 : f32
                } -> tensor<32x32xf32>
                %51 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%47, %50 : tensor<32x32x32xf32>, tensor<32x32xf32>) outs(%15 : tensor<32x32x32xf32>) {
                ^bb0(%in: f32, %in_10: f32, %out: f32):
                  %62 = arith.truncf %cst_0 : f64 to f32
                  %63 = arith.mulf %in, %62 : f32
                  %64 = arith.subf %63, %in_10 : f32
                  %65 = math.powf %cst, %64 : f32
                  linalg.yield %65 : f32
                } -> tensor<32x32x32xf32>
                %52 = linalg.fill ins(%cst_1 : f32) outs(%24 : tensor<32x32xf32>) -> tensor<32x32xf32>
                %53 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%51 : tensor<32x32x32xf32>) outs(%52 : tensor<32x32xf32>) {
                ^bb0(%in: f32, %out: f32):
                  %62 = arith.addf %in, %out : f32
                  linalg.yield %62 : f32
                } -> tensor<32x32xf32>
                %54 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%arg9, %50 : tensor<32x32xf32>, tensor<32x32xf32>) outs(%26 : tensor<32x32xf32>) {
                ^bb0(%in: f32, %in_10: f32, %out: f32):
                  %62 = arith.subf %in, %in_10 : f32
                  %63 = math.powf %cst, %62 : f32
                  linalg.yield %63 : f32
                } -> tensor<32x32xf32>
                %55 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%arg10, %54, %53 : tensor<32x32xf32>, tensor<32x32xf32>, tensor<32x32xf32>) outs(%arg10 : tensor<32x32xf32>) {
                ^bb0(%in: f32, %in_10: f32, %in_11: f32, %out: f32):
                  %62 = arith.mulf %in, %in_10 : f32
                  %63 = arith.addf %62, %in_11 : f32
                  linalg.yield %63 : f32
                } -> tensor<32x32xf32>
                %c0_8 = arith.constant 0 : index
                %56 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 524288 + d1 * 128 + d2)>(%33, %43, %c0_8)
                %reinterpret_cast_9 = memref.reinterpret_cast %arg1 to offset: [%56], sizes: [32, 32, 128], strides: [524288, 128, 1] : memref<32x4096x128xf32> to memref<32x32x128xf32, strided<[524288, 128, 1], offset: ?>>
                %57 = loom.copy_to_tensor %reinterpret_cast_9, %13 on @L1, interconnect : [@horizontal_links, @vertical_links], broadcast : [8, 8] : memref<32x32x128xf32, strided<[524288, 128, 1], offset: ?>>, memref<32x32x128xf32> -> tensor<32x32x128xf32>
                %58 = linalg.fill ins(%cst_1 : f32) outs(%32 : tensor<32x32x128xf32>) -> tensor<32x32x128xf32>
                %59 = linalg.batch_matmul ins(%51, %57 : tensor<32x32x32xf32>, tensor<32x32x128xf32>) outs(%58 : tensor<32x32x128xf32>) -> tensor<32x32x128xf32>
                %60 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%59, %arg11, %54 : tensor<32x32x128xf32>, tensor<32x32x128xf32>, tensor<32x32xf32>) outs(%arg11 : tensor<32x32x128xf32>) {
                ^bb0(%in: f32, %in_10: f32, %in_11: f32, %out: f32):
                  %62 = arith.mulf %in_10, %in_11 : f32
                  %63 = arith.addf %in, %62 : f32
                  linalg.yield %63 : f32
                } -> tensor<32x32x128xf32>
                %61 = linalg.copy ins(%50 : tensor<32x32xf32>) outs(%arg9 : tensor<32x32xf32>) -> tensor<32x32xf32>
                affine.yield %61, %55, %60 : tensor<32x32xf32>, tensor<32x32xf32>, tensor<32x32x128xf32>
              }
              %41 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%40#2, %40#1 : tensor<32x32x128xf32>, tensor<32x32xf32>) outs(%28 : tensor<32x32x128xf32>) {
              ^bb0(%in: f32, %in_6: f32, %out: f32):
                %43 = arith.divf %in, %in_6 : f32
                linalg.yield %43 : f32
              } -> tensor<32x32x128xf32>
              %c0_4 = arith.constant 0 : index
              %42 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 524288 + d1 * 128 + d2)>(%33, %34, %c0_4)
              %reinterpret_cast_5 = memref.reinterpret_cast %arg3 to offset: [%42], sizes: [32, 32, 128], strides: [524288, 128, 1] : memref<32x4096x128xf32> to memref<32x32x128xf32, strided<[524288, 128, 1], offset: ?>>
              loom.copy_from_tensor %41, %reinterpret_cast_5 on @DRAM : tensor<32x32x128xf32>, memref<32x32x128xf32, strided<[524288, 128, 1], offset: ?>>
            } {loom.iter_type = #loom.iter_type<temporal>}
          } {loom.iter_type = #loom.iter_type<temporal>}
        } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @y}
      } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @x}
      return
    }
  }
  module attributes {loom.pass_name = "Materialize"} {
    func.func @attention__d0i1_d1i1__f01__d_d_h__BB32__BM32__BN32(%arg0: memref<32x128x4096xf32>, %arg1: memref<32x4096x128xf32>, %arg2: memref<32x4096x128xf32>, %arg3: memref<32x4096x128xf32>) {
      %cst = arith.constant 2.000000e+00 : f32
      %cst_0 = arith.constant 0.12751743074602467 : f64
      %cst_1 = arith.constant 0.000000e+00 : f32
      %cst_2 = arith.constant 1.000000e+00 : f32
      %cst_3 = arith.constant 0xFF800000 : f32
      %c32 = arith.constant 32 : index
      affine.parallel (%arg4) = (0) to (8) {
        affine.parallel (%arg5) = (0) to (8) {
          affine.for %arg6 = 0 to 1 {
            affine.for %arg7 = 0 to 2 {
              %12 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 8 + d1 + d2 * 64)>(%arg4, %arg5, %arg7)
              %13 = loom.alloc [32, 32, 128] on @L1 : memref<32x32x128xf32>
              %14 = loom.alloc [32, 32, 32] on @L1 : memref<32x32x32xf32>
              %15 = loom.init_tensor %14[32, 32, 32] : memref<32x32x32xf32> -> tensor<32x32x32xf32>
              %16 = loom.alloc [32, 128, 32] on @L1 : memref<32x128x32xf32>
              %17 = loom.alloc [32, 32] on @L1 : memref<32x32xf32>
              %18 = loom.init_tensor %17[32, 32] : memref<32x32xf32> -> tensor<32x32xf32>
              %19 = loom.alloc [32, 32] on @L1 : memref<32x32xf32>
              %20 = loom.init_tensor %19[32, 32] : memref<32x32xf32> -> tensor<32x32xf32>
              %21 = loom.alloc [32, 32] on @L1 : memref<32x32xf32>
              %22 = loom.init_tensor %21[32, 32] : memref<32x32xf32> -> tensor<32x32xf32>
              %23 = loom.alloc [32, 32] on @L1 : memref<32x32xf32>
              %24 = loom.init_tensor %23[32, 32] : memref<32x32xf32> -> tensor<32x32xf32>
              %25 = loom.alloc [32, 32] on @L1 : memref<32x32xf32>
              %26 = loom.init_tensor %25[32, 32] : memref<32x32xf32> -> tensor<32x32xf32>
              %27 = loom.alloc [32, 32, 128] on @L1 : memref<32x32x128xf32>
              %28 = loom.init_tensor %27[32, 32, 128] : memref<32x32x128xf32> -> tensor<32x32x128xf32>
              %29 = loom.alloc [32, 32, 128] on @L1 : memref<32x32x128xf32>
              %30 = loom.init_tensor %29[32, 32, 128] : memref<32x32x128xf32> -> tensor<32x32x128xf32>
              %31 = loom.alloc [32, 32, 128] on @L1 : memref<32x32x128xf32>
              %32 = loom.init_tensor %31[32, 32, 128] : memref<32x32x128xf32> -> tensor<32x32x128xf32>
              %33 = arith.muli %arg6, %c32 : index
              %34 = arith.muli %12, %c32 : index
              %c0 = arith.constant 0 : index
              %35 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 524288 + d1 * 128 + d2)>(%33, %34, %c0)
              %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%35], sizes: [32, 32, 128], strides: [524288, 128, 1] : memref<32x4096x128xf32> to memref<32x32x128xf32, strided<[524288, 128, 1], offset: ?>>
              %36 = loom.copy_to_tensor %reinterpret_cast, %27 on @L1, interconnect : [], broadcast : [1, 1] : memref<32x32x128xf32, strided<[524288, 128, 1], offset: ?>>, memref<32x32x128xf32> -> tensor<32x32x128xf32>
              %37 = linalg.fill ins(%cst_1 : f32) outs(%30 : tensor<32x32x128xf32>) -> tensor<32x32x128xf32>
              %38 = linalg.fill ins(%cst_2 : f32) outs(%18 : tensor<32x32xf32>) -> tensor<32x32xf32>
              %39 = linalg.fill ins(%cst_3 : f32) outs(%20 : tensor<32x32xf32>) -> tensor<32x32xf32>
              %40:3 = affine.for %arg8 = 0 to 128 iter_args(%arg9 = %39, %arg10 = %38, %arg11 = %37) -> (tensor<32x32xf32>, tensor<32x32xf32>, tensor<32x32x128xf32>) {
                %43 = arith.muli %arg8, %c32 : index
                %c0_6 = arith.constant 0 : index
                %44 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 524288 + d1 * 4096 + d2)>(%33, %c0_6, %43)
                %reinterpret_cast_7 = memref.reinterpret_cast %arg0 to offset: [%44], sizes: [32, 128, 32], strides: [524288, 4096, 1] : memref<32x128x4096xf32> to memref<32x128x32xf32, strided<[524288, 4096, 1], offset: ?>>
                %45 = loom.copy_to_tensor %reinterpret_cast_7, %16 on @L1, interconnect : [], broadcast : [1, 1] : memref<32x128x32xf32, strided<[524288, 4096, 1], offset: ?>>, memref<32x128x32xf32> -> tensor<32x128x32xf32>
                %46 = linalg.fill ins(%cst_1 : f32) outs(%15 : tensor<32x32x32xf32>) -> tensor<32x32x32xf32>
                %47 = linalg.batch_matmul ins(%36, %45 : tensor<32x32x128xf32>, tensor<32x128x32xf32>) outs(%46 : tensor<32x32x32xf32>) -> tensor<32x32x32xf32>
                %48 = linalg.fill ins(%cst_3 : f32) outs(%22 : tensor<32x32xf32>) -> tensor<32x32xf32>
                %49 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%47 : tensor<32x32x32xf32>) outs(%48 : tensor<32x32xf32>) {
                ^bb0(%in: f32, %out: f32):
                  %62 = arith.maximumf %in, %out : f32
                  linalg.yield %62 : f32
                } -> tensor<32x32xf32>
                %50 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%arg9, %49 : tensor<32x32xf32>, tensor<32x32xf32>) outs(%22 : tensor<32x32xf32>) {
                ^bb0(%in: f32, %in_10: f32, %out: f32):
                  %62 = arith.truncf %cst_0 : f64 to f32
                  %63 = arith.mulf %in_10, %62 : f32
                  %64 = arith.cmpf ogt, %in, %63 : f32
                  %65 = arith.select %64, %in, %63 : f32
                  linalg.yield %65 : f32
                } -> tensor<32x32xf32>
                %51 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%47, %50 : tensor<32x32x32xf32>, tensor<32x32xf32>) outs(%15 : tensor<32x32x32xf32>) {
                ^bb0(%in: f32, %in_10: f32, %out: f32):
                  %62 = arith.truncf %cst_0 : f64 to f32
                  %63 = arith.mulf %in, %62 : f32
                  %64 = arith.subf %63, %in_10 : f32
                  %65 = math.powf %cst, %64 : f32
                  linalg.yield %65 : f32
                } -> tensor<32x32x32xf32>
                %52 = linalg.fill ins(%cst_1 : f32) outs(%24 : tensor<32x32xf32>) -> tensor<32x32xf32>
                %53 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%51 : tensor<32x32x32xf32>) outs(%52 : tensor<32x32xf32>) {
                ^bb0(%in: f32, %out: f32):
                  %62 = arith.addf %in, %out : f32
                  linalg.yield %62 : f32
                } -> tensor<32x32xf32>
                %54 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%arg9, %50 : tensor<32x32xf32>, tensor<32x32xf32>) outs(%26 : tensor<32x32xf32>) {
                ^bb0(%in: f32, %in_10: f32, %out: f32):
                  %62 = arith.subf %in, %in_10 : f32
                  %63 = math.powf %cst, %62 : f32
                  linalg.yield %63 : f32
                } -> tensor<32x32xf32>
                %55 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%arg10, %54, %53 : tensor<32x32xf32>, tensor<32x32xf32>, tensor<32x32xf32>) outs(%arg10 : tensor<32x32xf32>) {
                ^bb0(%in: f32, %in_10: f32, %in_11: f32, %out: f32):
                  %62 = arith.mulf %in, %in_10 : f32
                  %63 = arith.addf %62, %in_11 : f32
                  linalg.yield %63 : f32
                } -> tensor<32x32xf32>
                %c0_8 = arith.constant 0 : index
                %56 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 524288 + d1 * 128 + d2)>(%33, %43, %c0_8)
                %reinterpret_cast_9 = memref.reinterpret_cast %arg1 to offset: [%56], sizes: [32, 32, 128], strides: [524288, 128, 1] : memref<32x4096x128xf32> to memref<32x32x128xf32, strided<[524288, 128, 1], offset: ?>>
                %57 = loom.copy_to_tensor %reinterpret_cast_9, %13 on @L1, interconnect : [@horizontal_links], broadcast : [1, 8] : memref<32x32x128xf32, strided<[524288, 128, 1], offset: ?>>, memref<32x32x128xf32> -> tensor<32x32x128xf32>
                %58 = linalg.fill ins(%cst_1 : f32) outs(%32 : tensor<32x32x128xf32>) -> tensor<32x32x128xf32>
                %59 = linalg.batch_matmul ins(%51, %57 : tensor<32x32x32xf32>, tensor<32x32x128xf32>) outs(%58 : tensor<32x32x128xf32>) -> tensor<32x32x128xf32>
                %60 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%59, %arg11, %54 : tensor<32x32x128xf32>, tensor<32x32x128xf32>, tensor<32x32xf32>) outs(%arg11 : tensor<32x32x128xf32>) {
                ^bb0(%in: f32, %in_10: f32, %in_11: f32, %out: f32):
                  %62 = arith.mulf %in_10, %in_11 : f32
                  %63 = arith.addf %in, %62 : f32
                  linalg.yield %63 : f32
                } -> tensor<32x32x128xf32>
                %61 = linalg.copy ins(%50 : tensor<32x32xf32>) outs(%arg9 : tensor<32x32xf32>) -> tensor<32x32xf32>
                affine.yield %61, %55, %60 : tensor<32x32xf32>, tensor<32x32xf32>, tensor<32x32x128xf32>
              }
              %41 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%40#2, %40#1 : tensor<32x32x128xf32>, tensor<32x32xf32>) outs(%28 : tensor<32x32x128xf32>) {
              ^bb0(%in: f32, %in_6: f32, %out: f32):
                %43 = arith.divf %in, %in_6 : f32
                linalg.yield %43 : f32
              } -> tensor<32x32x128xf32>
              %c0_4 = arith.constant 0 : index
              %42 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 524288 + d1 * 128 + d2)>(%33, %34, %c0_4)
              %reinterpret_cast_5 = memref.reinterpret_cast %arg3 to offset: [%42], sizes: [32, 32, 128], strides: [524288, 128, 1] : memref<32x4096x128xf32> to memref<32x32x128xf32, strided<[524288, 128, 1], offset: ?>>
              loom.copy_from_tensor %41, %reinterpret_cast_5 on @DRAM : tensor<32x32x128xf32>, memref<32x32x128xf32, strided<[524288, 128, 1], offset: ?>>
            } {loom.iter_type = #loom.iter_type<temporal>}
          } {loom.iter_type = #loom.iter_type<temporal>}
        } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @y}
      } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @x}
      return
    }
  }
  module attributes {loom.pass_name = "Materialize"} {
    func.func @attention__d0i1_d1i1__f01__d_d_v__BB32__BM32__BN32(%arg0: memref<32x128x4096xf32>, %arg1: memref<32x4096x128xf32>, %arg2: memref<32x4096x128xf32>, %arg3: memref<32x4096x128xf32>) {
      %cst = arith.constant 2.000000e+00 : f32
      %cst_0 = arith.constant 0.12751743074602467 : f64
      %cst_1 = arith.constant 0.000000e+00 : f32
      %cst_2 = arith.constant 1.000000e+00 : f32
      %cst_3 = arith.constant 0xFF800000 : f32
      %c32 = arith.constant 32 : index
      affine.parallel (%arg4) = (0) to (8) {
        affine.parallel (%arg5) = (0) to (8) {
          affine.for %arg6 = 0 to 1 {
            affine.for %arg7 = 0 to 2 {
              %12 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 8 + d1 + d2 * 64)>(%arg4, %arg5, %arg7)
              %13 = loom.alloc [32, 32, 128] on @L1 : memref<32x32x128xf32>
              %14 = loom.alloc [32, 32, 32] on @L1 : memref<32x32x32xf32>
              %15 = loom.init_tensor %14[32, 32, 32] : memref<32x32x32xf32> -> tensor<32x32x32xf32>
              %16 = loom.alloc [32, 128, 32] on @L1 : memref<32x128x32xf32>
              %17 = loom.alloc [32, 32] on @L1 : memref<32x32xf32>
              %18 = loom.init_tensor %17[32, 32] : memref<32x32xf32> -> tensor<32x32xf32>
              %19 = loom.alloc [32, 32] on @L1 : memref<32x32xf32>
              %20 = loom.init_tensor %19[32, 32] : memref<32x32xf32> -> tensor<32x32xf32>
              %21 = loom.alloc [32, 32] on @L1 : memref<32x32xf32>
              %22 = loom.init_tensor %21[32, 32] : memref<32x32xf32> -> tensor<32x32xf32>
              %23 = loom.alloc [32, 32] on @L1 : memref<32x32xf32>
              %24 = loom.init_tensor %23[32, 32] : memref<32x32xf32> -> tensor<32x32xf32>
              %25 = loom.alloc [32, 32] on @L1 : memref<32x32xf32>
              %26 = loom.init_tensor %25[32, 32] : memref<32x32xf32> -> tensor<32x32xf32>
              %27 = loom.alloc [32, 32, 128] on @L1 : memref<32x32x128xf32>
              %28 = loom.init_tensor %27[32, 32, 128] : memref<32x32x128xf32> -> tensor<32x32x128xf32>
              %29 = loom.alloc [32, 32, 128] on @L1 : memref<32x32x128xf32>
              %30 = loom.init_tensor %29[32, 32, 128] : memref<32x32x128xf32> -> tensor<32x32x128xf32>
              %31 = loom.alloc [32, 32, 128] on @L1 : memref<32x32x128xf32>
              %32 = loom.init_tensor %31[32, 32, 128] : memref<32x32x128xf32> -> tensor<32x32x128xf32>
              %33 = arith.muli %arg6, %c32 : index
              %34 = arith.muli %12, %c32 : index
              %c0 = arith.constant 0 : index
              %35 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 524288 + d1 * 128 + d2)>(%33, %34, %c0)
              %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%35], sizes: [32, 32, 128], strides: [524288, 128, 1] : memref<32x4096x128xf32> to memref<32x32x128xf32, strided<[524288, 128, 1], offset: ?>>
              %36 = loom.copy_to_tensor %reinterpret_cast, %27 on @L1, interconnect : [], broadcast : [1, 1] : memref<32x32x128xf32, strided<[524288, 128, 1], offset: ?>>, memref<32x32x128xf32> -> tensor<32x32x128xf32>
              %37 = linalg.fill ins(%cst_1 : f32) outs(%30 : tensor<32x32x128xf32>) -> tensor<32x32x128xf32>
              %38 = linalg.fill ins(%cst_2 : f32) outs(%18 : tensor<32x32xf32>) -> tensor<32x32xf32>
              %39 = linalg.fill ins(%cst_3 : f32) outs(%20 : tensor<32x32xf32>) -> tensor<32x32xf32>
              %40:3 = affine.for %arg8 = 0 to 128 iter_args(%arg9 = %39, %arg10 = %38, %arg11 = %37) -> (tensor<32x32xf32>, tensor<32x32xf32>, tensor<32x32x128xf32>) {
                %43 = arith.muli %arg8, %c32 : index
                %c0_6 = arith.constant 0 : index
                %44 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 524288 + d1 * 4096 + d2)>(%33, %c0_6, %43)
                %reinterpret_cast_7 = memref.reinterpret_cast %arg0 to offset: [%44], sizes: [32, 128, 32], strides: [524288, 4096, 1] : memref<32x128x4096xf32> to memref<32x128x32xf32, strided<[524288, 4096, 1], offset: ?>>
                %45 = loom.copy_to_tensor %reinterpret_cast_7, %16 on @L1, interconnect : [], broadcast : [1, 1] : memref<32x128x32xf32, strided<[524288, 4096, 1], offset: ?>>, memref<32x128x32xf32> -> tensor<32x128x32xf32>
                %46 = linalg.fill ins(%cst_1 : f32) outs(%15 : tensor<32x32x32xf32>) -> tensor<32x32x32xf32>
                %47 = linalg.batch_matmul ins(%36, %45 : tensor<32x32x128xf32>, tensor<32x128x32xf32>) outs(%46 : tensor<32x32x32xf32>) -> tensor<32x32x32xf32>
                %48 = linalg.fill ins(%cst_3 : f32) outs(%22 : tensor<32x32xf32>) -> tensor<32x32xf32>
                %49 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%47 : tensor<32x32x32xf32>) outs(%48 : tensor<32x32xf32>) {
                ^bb0(%in: f32, %out: f32):
                  %62 = arith.maximumf %in, %out : f32
                  linalg.yield %62 : f32
                } -> tensor<32x32xf32>
                %50 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%arg9, %49 : tensor<32x32xf32>, tensor<32x32xf32>) outs(%22 : tensor<32x32xf32>) {
                ^bb0(%in: f32, %in_10: f32, %out: f32):
                  %62 = arith.truncf %cst_0 : f64 to f32
                  %63 = arith.mulf %in_10, %62 : f32
                  %64 = arith.cmpf ogt, %in, %63 : f32
                  %65 = arith.select %64, %in, %63 : f32
                  linalg.yield %65 : f32
                } -> tensor<32x32xf32>
                %51 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%47, %50 : tensor<32x32x32xf32>, tensor<32x32xf32>) outs(%15 : tensor<32x32x32xf32>) {
                ^bb0(%in: f32, %in_10: f32, %out: f32):
                  %62 = arith.truncf %cst_0 : f64 to f32
                  %63 = arith.mulf %in, %62 : f32
                  %64 = arith.subf %63, %in_10 : f32
                  %65 = math.powf %cst, %64 : f32
                  linalg.yield %65 : f32
                } -> tensor<32x32x32xf32>
                %52 = linalg.fill ins(%cst_1 : f32) outs(%24 : tensor<32x32xf32>) -> tensor<32x32xf32>
                %53 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%51 : tensor<32x32x32xf32>) outs(%52 : tensor<32x32xf32>) {
                ^bb0(%in: f32, %out: f32):
                  %62 = arith.addf %in, %out : f32
                  linalg.yield %62 : f32
                } -> tensor<32x32xf32>
                %54 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%arg9, %50 : tensor<32x32xf32>, tensor<32x32xf32>) outs(%26 : tensor<32x32xf32>) {
                ^bb0(%in: f32, %in_10: f32, %out: f32):
                  %62 = arith.subf %in, %in_10 : f32
                  %63 = math.powf %cst, %62 : f32
                  linalg.yield %63 : f32
                } -> tensor<32x32xf32>
                %55 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%arg10, %54, %53 : tensor<32x32xf32>, tensor<32x32xf32>, tensor<32x32xf32>) outs(%arg10 : tensor<32x32xf32>) {
                ^bb0(%in: f32, %in_10: f32, %in_11: f32, %out: f32):
                  %62 = arith.mulf %in, %in_10 : f32
                  %63 = arith.addf %62, %in_11 : f32
                  linalg.yield %63 : f32
                } -> tensor<32x32xf32>
                %c0_8 = arith.constant 0 : index
                %56 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 524288 + d1 * 128 + d2)>(%33, %43, %c0_8)
                %reinterpret_cast_9 = memref.reinterpret_cast %arg1 to offset: [%56], sizes: [32, 32, 128], strides: [524288, 128, 1] : memref<32x4096x128xf32> to memref<32x32x128xf32, strided<[524288, 128, 1], offset: ?>>
                %57 = loom.copy_to_tensor %reinterpret_cast_9, %13 on @L1, interconnect : [@vertical_links], broadcast : [8, 1] : memref<32x32x128xf32, strided<[524288, 128, 1], offset: ?>>, memref<32x32x128xf32> -> tensor<32x32x128xf32>
                %58 = linalg.fill ins(%cst_1 : f32) outs(%32 : tensor<32x32x128xf32>) -> tensor<32x32x128xf32>
                %59 = linalg.batch_matmul ins(%51, %57 : tensor<32x32x32xf32>, tensor<32x32x128xf32>) outs(%58 : tensor<32x32x128xf32>) -> tensor<32x32x128xf32>
                %60 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%59, %arg11, %54 : tensor<32x32x128xf32>, tensor<32x32x128xf32>, tensor<32x32xf32>) outs(%arg11 : tensor<32x32x128xf32>) {
                ^bb0(%in: f32, %in_10: f32, %in_11: f32, %out: f32):
                  %62 = arith.mulf %in_10, %in_11 : f32
                  %63 = arith.addf %in, %62 : f32
                  linalg.yield %63 : f32
                } -> tensor<32x32x128xf32>
                %61 = linalg.copy ins(%50 : tensor<32x32xf32>) outs(%arg9 : tensor<32x32xf32>) -> tensor<32x32xf32>
                affine.yield %61, %55, %60 : tensor<32x32xf32>, tensor<32x32xf32>, tensor<32x32x128xf32>
              }
              %41 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%40#2, %40#1 : tensor<32x32x128xf32>, tensor<32x32xf32>) outs(%28 : tensor<32x32x128xf32>) {
              ^bb0(%in: f32, %in_6: f32, %out: f32):
                %43 = arith.divf %in, %in_6 : f32
                linalg.yield %43 : f32
              } -> tensor<32x32x128xf32>
              %c0_4 = arith.constant 0 : index
              %42 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 524288 + d1 * 128 + d2)>(%33, %34, %c0_4)
              %reinterpret_cast_5 = memref.reinterpret_cast %arg3 to offset: [%42], sizes: [32, 32, 128], strides: [524288, 128, 1] : memref<32x4096x128xf32> to memref<32x32x128xf32, strided<[524288, 128, 1], offset: ?>>
              loom.copy_from_tensor %41, %reinterpret_cast_5 on @DRAM : tensor<32x32x128xf32>, memref<32x32x128xf32, strided<[524288, 128, 1], offset: ?>>
            } {loom.iter_type = #loom.iter_type<temporal>}
          } {loom.iter_type = #loom.iter_type<temporal>}
        } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @y}
      } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @x}
      return
    }
  }
  module attributes {loom.pass_name = "Materialize"} {
    func.func @attention__d0i1_d1i1__f01__d_a_d__BB32__BM32__BN32(%arg0: memref<32x128x4096xf32>, %arg1: memref<32x4096x128xf32>, %arg2: memref<32x4096x128xf32>, %arg3: memref<32x4096x128xf32>) {
      %cst = arith.constant 2.000000e+00 : f32
      %cst_0 = arith.constant 0.12751743074602467 : f64
      %cst_1 = arith.constant 0.000000e+00 : f32
      %cst_2 = arith.constant 1.000000e+00 : f32
      %cst_3 = arith.constant 0xFF800000 : f32
      %c32 = arith.constant 32 : index
      affine.parallel (%arg4) = (0) to (8) {
        affine.parallel (%arg5) = (0) to (8) {
          affine.for %arg6 = 0 to 1 {
            affine.for %arg7 = 0 to 2 {
              %12 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 8 + d1 + d2 * 64)>(%arg4, %arg5, %arg7)
              %13 = loom.alloc [32, 32, 128] on @L1 : memref<32x32x128xf32>
              %14 = loom.alloc [32, 32, 32] on @L1 : memref<32x32x32xf32>
              %15 = loom.init_tensor %14[32, 32, 32] : memref<32x32x32xf32> -> tensor<32x32x32xf32>
              %16 = loom.alloc [32, 128, 32] on @L1 : memref<32x128x32xf32>
              %17 = loom.alloc [32, 32] on @L1 : memref<32x32xf32>
              %18 = loom.init_tensor %17[32, 32] : memref<32x32xf32> -> tensor<32x32xf32>
              %19 = loom.alloc [32, 32] on @L1 : memref<32x32xf32>
              %20 = loom.init_tensor %19[32, 32] : memref<32x32xf32> -> tensor<32x32xf32>
              %21 = loom.alloc [32, 32] on @L1 : memref<32x32xf32>
              %22 = loom.init_tensor %21[32, 32] : memref<32x32xf32> -> tensor<32x32xf32>
              %23 = loom.alloc [32, 32] on @L1 : memref<32x32xf32>
              %24 = loom.init_tensor %23[32, 32] : memref<32x32xf32> -> tensor<32x32xf32>
              %25 = loom.alloc [32, 32] on @L1 : memref<32x32xf32>
              %26 = loom.init_tensor %25[32, 32] : memref<32x32xf32> -> tensor<32x32xf32>
              %27 = loom.alloc [32, 32, 128] on @L1 : memref<32x32x128xf32>
              %28 = loom.init_tensor %27[32, 32, 128] : memref<32x32x128xf32> -> tensor<32x32x128xf32>
              %29 = loom.alloc [32, 32, 128] on @L1 : memref<32x32x128xf32>
              %30 = loom.init_tensor %29[32, 32, 128] : memref<32x32x128xf32> -> tensor<32x32x128xf32>
              %31 = loom.alloc [32, 32, 128] on @L1 : memref<32x32x128xf32>
              %32 = loom.init_tensor %31[32, 32, 128] : memref<32x32x128xf32> -> tensor<32x32x128xf32>
              %33 = arith.muli %arg6, %c32 : index
              %34 = arith.muli %12, %c32 : index
              %c0 = arith.constant 0 : index
              %35 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 524288 + d1 * 128 + d2)>(%33, %34, %c0)
              %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%35], sizes: [32, 32, 128], strides: [524288, 128, 1] : memref<32x4096x128xf32> to memref<32x32x128xf32, strided<[524288, 128, 1], offset: ?>>
              %36 = loom.copy_to_tensor %reinterpret_cast, %27 on @L1, interconnect : [], broadcast : [1, 1] : memref<32x32x128xf32, strided<[524288, 128, 1], offset: ?>>, memref<32x32x128xf32> -> tensor<32x32x128xf32>
              %37 = linalg.fill ins(%cst_1 : f32) outs(%30 : tensor<32x32x128xf32>) -> tensor<32x32x128xf32>
              %38 = linalg.fill ins(%cst_2 : f32) outs(%18 : tensor<32x32xf32>) -> tensor<32x32xf32>
              %39 = linalg.fill ins(%cst_3 : f32) outs(%20 : tensor<32x32xf32>) -> tensor<32x32xf32>
              %40:3 = affine.for %arg8 = 0 to 128 iter_args(%arg9 = %39, %arg10 = %38, %arg11 = %37) -> (tensor<32x32xf32>, tensor<32x32xf32>, tensor<32x32x128xf32>) {
                %43 = arith.muli %arg8, %c32 : index
                %c0_6 = arith.constant 0 : index
                %44 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 524288 + d1 * 4096 + d2)>(%33, %c0_6, %43)
                %reinterpret_cast_7 = memref.reinterpret_cast %arg0 to offset: [%44], sizes: [32, 128, 32], strides: [524288, 4096, 1] : memref<32x128x4096xf32> to memref<32x128x32xf32, strided<[524288, 4096, 1], offset: ?>>
                %45 = loom.copy_to_tensor %reinterpret_cast_7, %16 on @L1, interconnect : [@horizontal_links, @vertical_links], broadcast : [8, 8] : memref<32x128x32xf32, strided<[524288, 4096, 1], offset: ?>>, memref<32x128x32xf32> -> tensor<32x128x32xf32>
                %46 = linalg.fill ins(%cst_1 : f32) outs(%15 : tensor<32x32x32xf32>) -> tensor<32x32x32xf32>
                %47 = linalg.batch_matmul ins(%36, %45 : tensor<32x32x128xf32>, tensor<32x128x32xf32>) outs(%46 : tensor<32x32x32xf32>) -> tensor<32x32x32xf32>
                %48 = linalg.fill ins(%cst_3 : f32) outs(%22 : tensor<32x32xf32>) -> tensor<32x32xf32>
                %49 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%47 : tensor<32x32x32xf32>) outs(%48 : tensor<32x32xf32>) {
                ^bb0(%in: f32, %out: f32):
                  %62 = arith.maximumf %in, %out : f32
                  linalg.yield %62 : f32
                } -> tensor<32x32xf32>
                %50 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%arg9, %49 : tensor<32x32xf32>, tensor<32x32xf32>) outs(%22 : tensor<32x32xf32>) {
                ^bb0(%in: f32, %in_10: f32, %out: f32):
                  %62 = arith.truncf %cst_0 : f64 to f32
                  %63 = arith.mulf %in_10, %62 : f32
                  %64 = arith.cmpf ogt, %in, %63 : f32
                  %65 = arith.select %64, %in, %63 : f32
                  linalg.yield %65 : f32
                } -> tensor<32x32xf32>
                %51 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%47, %50 : tensor<32x32x32xf32>, tensor<32x32xf32>) outs(%15 : tensor<32x32x32xf32>) {
                ^bb0(%in: f32, %in_10: f32, %out: f32):
                  %62 = arith.truncf %cst_0 : f64 to f32
                  %63 = arith.mulf %in, %62 : f32
                  %64 = arith.subf %63, %in_10 : f32
                  %65 = math.powf %cst, %64 : f32
                  linalg.yield %65 : f32
                } -> tensor<32x32x32xf32>
                %52 = linalg.fill ins(%cst_1 : f32) outs(%24 : tensor<32x32xf32>) -> tensor<32x32xf32>
                %53 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%51 : tensor<32x32x32xf32>) outs(%52 : tensor<32x32xf32>) {
                ^bb0(%in: f32, %out: f32):
                  %62 = arith.addf %in, %out : f32
                  linalg.yield %62 : f32
                } -> tensor<32x32xf32>
                %54 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%arg9, %50 : tensor<32x32xf32>, tensor<32x32xf32>) outs(%26 : tensor<32x32xf32>) {
                ^bb0(%in: f32, %in_10: f32, %out: f32):
                  %62 = arith.subf %in, %in_10 : f32
                  %63 = math.powf %cst, %62 : f32
                  linalg.yield %63 : f32
                } -> tensor<32x32xf32>
                %55 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%arg10, %54, %53 : tensor<32x32xf32>, tensor<32x32xf32>, tensor<32x32xf32>) outs(%arg10 : tensor<32x32xf32>) {
                ^bb0(%in: f32, %in_10: f32, %in_11: f32, %out: f32):
                  %62 = arith.mulf %in, %in_10 : f32
                  %63 = arith.addf %62, %in_11 : f32
                  linalg.yield %63 : f32
                } -> tensor<32x32xf32>
                %c0_8 = arith.constant 0 : index
                %56 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 524288 + d1 * 128 + d2)>(%33, %43, %c0_8)
                %reinterpret_cast_9 = memref.reinterpret_cast %arg1 to offset: [%56], sizes: [32, 32, 128], strides: [524288, 128, 1] : memref<32x4096x128xf32> to memref<32x32x128xf32, strided<[524288, 128, 1], offset: ?>>
                %57 = loom.copy_to_tensor %reinterpret_cast_9, %13 on @L1, interconnect : [], broadcast : [1, 1] : memref<32x32x128xf32, strided<[524288, 128, 1], offset: ?>>, memref<32x32x128xf32> -> tensor<32x32x128xf32>
                %58 = linalg.fill ins(%cst_1 : f32) outs(%32 : tensor<32x32x128xf32>) -> tensor<32x32x128xf32>
                %59 = linalg.batch_matmul ins(%51, %57 : tensor<32x32x32xf32>, tensor<32x32x128xf32>) outs(%58 : tensor<32x32x128xf32>) -> tensor<32x32x128xf32>
                %60 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%59, %arg11, %54 : tensor<32x32x128xf32>, tensor<32x32x128xf32>, tensor<32x32xf32>) outs(%arg11 : tensor<32x32x128xf32>) {
                ^bb0(%in: f32, %in_10: f32, %in_11: f32, %out: f32):
                  %62 = arith.mulf %in_10, %in_11 : f32
                  %63 = arith.addf %in, %62 : f32
                  linalg.yield %63 : f32
                } -> tensor<32x32x128xf32>
                %61 = linalg.copy ins(%50 : tensor<32x32xf32>) outs(%arg9 : tensor<32x32xf32>) -> tensor<32x32xf32>
                affine.yield %61, %55, %60 : tensor<32x32xf32>, tensor<32x32xf32>, tensor<32x32x128xf32>
              }
              %41 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%40#2, %40#1 : tensor<32x32x128xf32>, tensor<32x32xf32>) outs(%28 : tensor<32x32x128xf32>) {
              ^bb0(%in: f32, %in_6: f32, %out: f32):
                %43 = arith.divf %in, %in_6 : f32
                linalg.yield %43 : f32
              } -> tensor<32x32x128xf32>
              %c0_4 = arith.constant 0 : index
              %42 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 524288 + d1 * 128 + d2)>(%33, %34, %c0_4)
              %reinterpret_cast_5 = memref.reinterpret_cast %arg3 to offset: [%42], sizes: [32, 32, 128], strides: [524288, 128, 1] : memref<32x4096x128xf32> to memref<32x32x128xf32, strided<[524288, 128, 1], offset: ?>>
              loom.copy_from_tensor %41, %reinterpret_cast_5 on @DRAM : tensor<32x32x128xf32>, memref<32x32x128xf32, strided<[524288, 128, 1], offset: ?>>
            } {loom.iter_type = #loom.iter_type<temporal>}
          } {loom.iter_type = #loom.iter_type<temporal>}
        } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @y}
      } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @x}
      return
    }
  }
  module attributes {loom.pass_name = "Materialize"} {
    func.func @attention__d0i1_d1i1__f01__d_a_a__BB32__BM32__BN32(%arg0: memref<32x128x4096xf32>, %arg1: memref<32x4096x128xf32>, %arg2: memref<32x4096x128xf32>, %arg3: memref<32x4096x128xf32>) {
      %cst = arith.constant 2.000000e+00 : f32
      %cst_0 = arith.constant 0.12751743074602467 : f64
      %cst_1 = arith.constant 0.000000e+00 : f32
      %cst_2 = arith.constant 1.000000e+00 : f32
      %cst_3 = arith.constant 0xFF800000 : f32
      %c32 = arith.constant 32 : index
      affine.parallel (%arg4) = (0) to (8) {
        affine.parallel (%arg5) = (0) to (8) {
          affine.for %arg6 = 0 to 1 {
            affine.for %arg7 = 0 to 2 {
              %12 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 8 + d1 + d2 * 64)>(%arg4, %arg5, %arg7)
              %13 = loom.alloc [32, 32, 128] on @L1 : memref<32x32x128xf32>
              %14 = loom.alloc [32, 32, 32] on @L1 : memref<32x32x32xf32>
              %15 = loom.init_tensor %14[32, 32, 32] : memref<32x32x32xf32> -> tensor<32x32x32xf32>
              %16 = loom.alloc [32, 128, 32] on @L1 : memref<32x128x32xf32>
              %17 = loom.alloc [32, 32] on @L1 : memref<32x32xf32>
              %18 = loom.init_tensor %17[32, 32] : memref<32x32xf32> -> tensor<32x32xf32>
              %19 = loom.alloc [32, 32] on @L1 : memref<32x32xf32>
              %20 = loom.init_tensor %19[32, 32] : memref<32x32xf32> -> tensor<32x32xf32>
              %21 = loom.alloc [32, 32] on @L1 : memref<32x32xf32>
              %22 = loom.init_tensor %21[32, 32] : memref<32x32xf32> -> tensor<32x32xf32>
              %23 = loom.alloc [32, 32] on @L1 : memref<32x32xf32>
              %24 = loom.init_tensor %23[32, 32] : memref<32x32xf32> -> tensor<32x32xf32>
              %25 = loom.alloc [32, 32] on @L1 : memref<32x32xf32>
              %26 = loom.init_tensor %25[32, 32] : memref<32x32xf32> -> tensor<32x32xf32>
              %27 = loom.alloc [32, 32, 128] on @L1 : memref<32x32x128xf32>
              %28 = loom.init_tensor %27[32, 32, 128] : memref<32x32x128xf32> -> tensor<32x32x128xf32>
              %29 = loom.alloc [32, 32, 128] on @L1 : memref<32x32x128xf32>
              %30 = loom.init_tensor %29[32, 32, 128] : memref<32x32x128xf32> -> tensor<32x32x128xf32>
              %31 = loom.alloc [32, 32, 128] on @L1 : memref<32x32x128xf32>
              %32 = loom.init_tensor %31[32, 32, 128] : memref<32x32x128xf32> -> tensor<32x32x128xf32>
              %33 = arith.muli %arg6, %c32 : index
              %34 = arith.muli %12, %c32 : index
              %c0 = arith.constant 0 : index
              %35 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 524288 + d1 * 128 + d2)>(%33, %34, %c0)
              %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%35], sizes: [32, 32, 128], strides: [524288, 128, 1] : memref<32x4096x128xf32> to memref<32x32x128xf32, strided<[524288, 128, 1], offset: ?>>
              %36 = loom.copy_to_tensor %reinterpret_cast, %27 on @L1, interconnect : [], broadcast : [1, 1] : memref<32x32x128xf32, strided<[524288, 128, 1], offset: ?>>, memref<32x32x128xf32> -> tensor<32x32x128xf32>
              %37 = linalg.fill ins(%cst_1 : f32) outs(%30 : tensor<32x32x128xf32>) -> tensor<32x32x128xf32>
              %38 = linalg.fill ins(%cst_2 : f32) outs(%18 : tensor<32x32xf32>) -> tensor<32x32xf32>
              %39 = linalg.fill ins(%cst_3 : f32) outs(%20 : tensor<32x32xf32>) -> tensor<32x32xf32>
              %40:3 = affine.for %arg8 = 0 to 128 iter_args(%arg9 = %39, %arg10 = %38, %arg11 = %37) -> (tensor<32x32xf32>, tensor<32x32xf32>, tensor<32x32x128xf32>) {
                %43 = arith.muli %arg8, %c32 : index
                %c0_6 = arith.constant 0 : index
                %44 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 524288 + d1 * 4096 + d2)>(%33, %c0_6, %43)
                %reinterpret_cast_7 = memref.reinterpret_cast %arg0 to offset: [%44], sizes: [32, 128, 32], strides: [524288, 4096, 1] : memref<32x128x4096xf32> to memref<32x128x32xf32, strided<[524288, 4096, 1], offset: ?>>
                %45 = loom.copy_to_tensor %reinterpret_cast_7, %16 on @L1, interconnect : [@horizontal_links, @vertical_links], broadcast : [8, 8] : memref<32x128x32xf32, strided<[524288, 4096, 1], offset: ?>>, memref<32x128x32xf32> -> tensor<32x128x32xf32>
                %46 = linalg.fill ins(%cst_1 : f32) outs(%15 : tensor<32x32x32xf32>) -> tensor<32x32x32xf32>
                %47 = linalg.batch_matmul ins(%36, %45 : tensor<32x32x128xf32>, tensor<32x128x32xf32>) outs(%46 : tensor<32x32x32xf32>) -> tensor<32x32x32xf32>
                %48 = linalg.fill ins(%cst_3 : f32) outs(%22 : tensor<32x32xf32>) -> tensor<32x32xf32>
                %49 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%47 : tensor<32x32x32xf32>) outs(%48 : tensor<32x32xf32>) {
                ^bb0(%in: f32, %out: f32):
                  %62 = arith.maximumf %in, %out : f32
                  linalg.yield %62 : f32
                } -> tensor<32x32xf32>
                %50 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%arg9, %49 : tensor<32x32xf32>, tensor<32x32xf32>) outs(%22 : tensor<32x32xf32>) {
                ^bb0(%in: f32, %in_10: f32, %out: f32):
                  %62 = arith.truncf %cst_0 : f64 to f32
                  %63 = arith.mulf %in_10, %62 : f32
                  %64 = arith.cmpf ogt, %in, %63 : f32
                  %65 = arith.select %64, %in, %63 : f32
                  linalg.yield %65 : f32
                } -> tensor<32x32xf32>
                %51 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%47, %50 : tensor<32x32x32xf32>, tensor<32x32xf32>) outs(%15 : tensor<32x32x32xf32>) {
                ^bb0(%in: f32, %in_10: f32, %out: f32):
                  %62 = arith.truncf %cst_0 : f64 to f32
                  %63 = arith.mulf %in, %62 : f32
                  %64 = arith.subf %63, %in_10 : f32
                  %65 = math.powf %cst, %64 : f32
                  linalg.yield %65 : f32
                } -> tensor<32x32x32xf32>
                %52 = linalg.fill ins(%cst_1 : f32) outs(%24 : tensor<32x32xf32>) -> tensor<32x32xf32>
                %53 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%51 : tensor<32x32x32xf32>) outs(%52 : tensor<32x32xf32>) {
                ^bb0(%in: f32, %out: f32):
                  %62 = arith.addf %in, %out : f32
                  linalg.yield %62 : f32
                } -> tensor<32x32xf32>
                %54 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%arg9, %50 : tensor<32x32xf32>, tensor<32x32xf32>) outs(%26 : tensor<32x32xf32>) {
                ^bb0(%in: f32, %in_10: f32, %out: f32):
                  %62 = arith.subf %in, %in_10 : f32
                  %63 = math.powf %cst, %62 : f32
                  linalg.yield %63 : f32
                } -> tensor<32x32xf32>
                %55 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%arg10, %54, %53 : tensor<32x32xf32>, tensor<32x32xf32>, tensor<32x32xf32>) outs(%arg10 : tensor<32x32xf32>) {
                ^bb0(%in: f32, %in_10: f32, %in_11: f32, %out: f32):
                  %62 = arith.mulf %in, %in_10 : f32
                  %63 = arith.addf %62, %in_11 : f32
                  linalg.yield %63 : f32
                } -> tensor<32x32xf32>
                %c0_8 = arith.constant 0 : index
                %56 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 524288 + d1 * 128 + d2)>(%33, %43, %c0_8)
                %reinterpret_cast_9 = memref.reinterpret_cast %arg1 to offset: [%56], sizes: [32, 32, 128], strides: [524288, 128, 1] : memref<32x4096x128xf32> to memref<32x32x128xf32, strided<[524288, 128, 1], offset: ?>>
                %57 = loom.copy_to_tensor %reinterpret_cast_9, %13 on @L1, interconnect : [@horizontal_links, @vertical_links], broadcast : [8, 8] : memref<32x32x128xf32, strided<[524288, 128, 1], offset: ?>>, memref<32x32x128xf32> -> tensor<32x32x128xf32>
                %58 = linalg.fill ins(%cst_1 : f32) outs(%32 : tensor<32x32x128xf32>) -> tensor<32x32x128xf32>
                %59 = linalg.batch_matmul ins(%51, %57 : tensor<32x32x32xf32>, tensor<32x32x128xf32>) outs(%58 : tensor<32x32x128xf32>) -> tensor<32x32x128xf32>
                %60 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%59, %arg11, %54 : tensor<32x32x128xf32>, tensor<32x32x128xf32>, tensor<32x32xf32>) outs(%arg11 : tensor<32x32x128xf32>) {
                ^bb0(%in: f32, %in_10: f32, %in_11: f32, %out: f32):
                  %62 = arith.mulf %in_10, %in_11 : f32
                  %63 = arith.addf %in, %62 : f32
                  linalg.yield %63 : f32
                } -> tensor<32x32x128xf32>
                %61 = linalg.copy ins(%50 : tensor<32x32xf32>) outs(%arg9 : tensor<32x32xf32>) -> tensor<32x32xf32>
                affine.yield %61, %55, %60 : tensor<32x32xf32>, tensor<32x32xf32>, tensor<32x32x128xf32>
              }
              %41 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%40#2, %40#1 : tensor<32x32x128xf32>, tensor<32x32xf32>) outs(%28 : tensor<32x32x128xf32>) {
              ^bb0(%in: f32, %in_6: f32, %out: f32):
                %43 = arith.divf %in, %in_6 : f32
                linalg.yield %43 : f32
              } -> tensor<32x32x128xf32>
              %c0_4 = arith.constant 0 : index
              %42 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 524288 + d1 * 128 + d2)>(%33, %34, %c0_4)
              %reinterpret_cast_5 = memref.reinterpret_cast %arg3 to offset: [%42], sizes: [32, 32, 128], strides: [524288, 128, 1] : memref<32x4096x128xf32> to memref<32x32x128xf32, strided<[524288, 128, 1], offset: ?>>
              loom.copy_from_tensor %41, %reinterpret_cast_5 on @DRAM : tensor<32x32x128xf32>, memref<32x32x128xf32, strided<[524288, 128, 1], offset: ?>>
            } {loom.iter_type = #loom.iter_type<temporal>}
          } {loom.iter_type = #loom.iter_type<temporal>}
        } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @y}
      } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @x}
      return
    }
  }
  module attributes {loom.pass_name = "Materialize"} {
    func.func @attention__d0i1_d1i1__f01__d_a_h__BB32__BM32__BN32(%arg0: memref<32x128x4096xf32>, %arg1: memref<32x4096x128xf32>, %arg2: memref<32x4096x128xf32>, %arg3: memref<32x4096x128xf32>) {
      %cst = arith.constant 2.000000e+00 : f32
      %cst_0 = arith.constant 0.12751743074602467 : f64
      %cst_1 = arith.constant 0.000000e+00 : f32
      %cst_2 = arith.constant 1.000000e+00 : f32
      %cst_3 = arith.constant 0xFF800000 : f32
      %c32 = arith.constant 32 : index
      affine.parallel (%arg4) = (0) to (8) {
        affine.parallel (%arg5) = (0) to (8) {
          affine.for %arg6 = 0 to 1 {
            affine.for %arg7 = 0 to 2 {
              %12 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 8 + d1 + d2 * 64)>(%arg4, %arg5, %arg7)
              %13 = loom.alloc [32, 32, 128] on @L1 : memref<32x32x128xf32>
              %14 = loom.alloc [32, 32, 32] on @L1 : memref<32x32x32xf32>
              %15 = loom.init_tensor %14[32, 32, 32] : memref<32x32x32xf32> -> tensor<32x32x32xf32>
              %16 = loom.alloc [32, 128, 32] on @L1 : memref<32x128x32xf32>
              %17 = loom.alloc [32, 32] on @L1 : memref<32x32xf32>
              %18 = loom.init_tensor %17[32, 32] : memref<32x32xf32> -> tensor<32x32xf32>
              %19 = loom.alloc [32, 32] on @L1 : memref<32x32xf32>
              %20 = loom.init_tensor %19[32, 32] : memref<32x32xf32> -> tensor<32x32xf32>
              %21 = loom.alloc [32, 32] on @L1 : memref<32x32xf32>
              %22 = loom.init_tensor %21[32, 32] : memref<32x32xf32> -> tensor<32x32xf32>
              %23 = loom.alloc [32, 32] on @L1 : memref<32x32xf32>
              %24 = loom.init_tensor %23[32, 32] : memref<32x32xf32> -> tensor<32x32xf32>
              %25 = loom.alloc [32, 32] on @L1 : memref<32x32xf32>
              %26 = loom.init_tensor %25[32, 32] : memref<32x32xf32> -> tensor<32x32xf32>
              %27 = loom.alloc [32, 32, 128] on @L1 : memref<32x32x128xf32>
              %28 = loom.init_tensor %27[32, 32, 128] : memref<32x32x128xf32> -> tensor<32x32x128xf32>
              %29 = loom.alloc [32, 32, 128] on @L1 : memref<32x32x128xf32>
              %30 = loom.init_tensor %29[32, 32, 128] : memref<32x32x128xf32> -> tensor<32x32x128xf32>
              %31 = loom.alloc [32, 32, 128] on @L1 : memref<32x32x128xf32>
              %32 = loom.init_tensor %31[32, 32, 128] : memref<32x32x128xf32> -> tensor<32x32x128xf32>
              %33 = arith.muli %arg6, %c32 : index
              %34 = arith.muli %12, %c32 : index
              %c0 = arith.constant 0 : index
              %35 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 524288 + d1 * 128 + d2)>(%33, %34, %c0)
              %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%35], sizes: [32, 32, 128], strides: [524288, 128, 1] : memref<32x4096x128xf32> to memref<32x32x128xf32, strided<[524288, 128, 1], offset: ?>>
              %36 = loom.copy_to_tensor %reinterpret_cast, %27 on @L1, interconnect : [], broadcast : [1, 1] : memref<32x32x128xf32, strided<[524288, 128, 1], offset: ?>>, memref<32x32x128xf32> -> tensor<32x32x128xf32>
              %37 = linalg.fill ins(%cst_1 : f32) outs(%30 : tensor<32x32x128xf32>) -> tensor<32x32x128xf32>
              %38 = linalg.fill ins(%cst_2 : f32) outs(%18 : tensor<32x32xf32>) -> tensor<32x32xf32>
              %39 = linalg.fill ins(%cst_3 : f32) outs(%20 : tensor<32x32xf32>) -> tensor<32x32xf32>
              %40:3 = affine.for %arg8 = 0 to 128 iter_args(%arg9 = %39, %arg10 = %38, %arg11 = %37) -> (tensor<32x32xf32>, tensor<32x32xf32>, tensor<32x32x128xf32>) {
                %43 = arith.muli %arg8, %c32 : index
                %c0_6 = arith.constant 0 : index
                %44 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 524288 + d1 * 4096 + d2)>(%33, %c0_6, %43)
                %reinterpret_cast_7 = memref.reinterpret_cast %arg0 to offset: [%44], sizes: [32, 128, 32], strides: [524288, 4096, 1] : memref<32x128x4096xf32> to memref<32x128x32xf32, strided<[524288, 4096, 1], offset: ?>>
                %45 = loom.copy_to_tensor %reinterpret_cast_7, %16 on @L1, interconnect : [@horizontal_links, @vertical_links], broadcast : [8, 8] : memref<32x128x32xf32, strided<[524288, 4096, 1], offset: ?>>, memref<32x128x32xf32> -> tensor<32x128x32xf32>
                %46 = linalg.fill ins(%cst_1 : f32) outs(%15 : tensor<32x32x32xf32>) -> tensor<32x32x32xf32>
                %47 = linalg.batch_matmul ins(%36, %45 : tensor<32x32x128xf32>, tensor<32x128x32xf32>) outs(%46 : tensor<32x32x32xf32>) -> tensor<32x32x32xf32>
                %48 = linalg.fill ins(%cst_3 : f32) outs(%22 : tensor<32x32xf32>) -> tensor<32x32xf32>
                %49 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%47 : tensor<32x32x32xf32>) outs(%48 : tensor<32x32xf32>) {
                ^bb0(%in: f32, %out: f32):
                  %62 = arith.maximumf %in, %out : f32
                  linalg.yield %62 : f32
                } -> tensor<32x32xf32>
                %50 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%arg9, %49 : tensor<32x32xf32>, tensor<32x32xf32>) outs(%22 : tensor<32x32xf32>) {
                ^bb0(%in: f32, %in_10: f32, %out: f32):
                  %62 = arith.truncf %cst_0 : f64 to f32
                  %63 = arith.mulf %in_10, %62 : f32
                  %64 = arith.cmpf ogt, %in, %63 : f32
                  %65 = arith.select %64, %in, %63 : f32
                  linalg.yield %65 : f32
                } -> tensor<32x32xf32>
                %51 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%47, %50 : tensor<32x32x32xf32>, tensor<32x32xf32>) outs(%15 : tensor<32x32x32xf32>) {
                ^bb0(%in: f32, %in_10: f32, %out: f32):
                  %62 = arith.truncf %cst_0 : f64 to f32
                  %63 = arith.mulf %in, %62 : f32
                  %64 = arith.subf %63, %in_10 : f32
                  %65 = math.powf %cst, %64 : f32
                  linalg.yield %65 : f32
                } -> tensor<32x32x32xf32>
                %52 = linalg.fill ins(%cst_1 : f32) outs(%24 : tensor<32x32xf32>) -> tensor<32x32xf32>
                %53 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%51 : tensor<32x32x32xf32>) outs(%52 : tensor<32x32xf32>) {
                ^bb0(%in: f32, %out: f32):
                  %62 = arith.addf %in, %out : f32
                  linalg.yield %62 : f32
                } -> tensor<32x32xf32>
                %54 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%arg9, %50 : tensor<32x32xf32>, tensor<32x32xf32>) outs(%26 : tensor<32x32xf32>) {
                ^bb0(%in: f32, %in_10: f32, %out: f32):
                  %62 = arith.subf %in, %in_10 : f32
                  %63 = math.powf %cst, %62 : f32
                  linalg.yield %63 : f32
                } -> tensor<32x32xf32>
                %55 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%arg10, %54, %53 : tensor<32x32xf32>, tensor<32x32xf32>, tensor<32x32xf32>) outs(%arg10 : tensor<32x32xf32>) {
                ^bb0(%in: f32, %in_10: f32, %in_11: f32, %out: f32):
                  %62 = arith.mulf %in, %in_10 : f32
                  %63 = arith.addf %62, %in_11 : f32
                  linalg.yield %63 : f32
                } -> tensor<32x32xf32>
                %c0_8 = arith.constant 0 : index
                %56 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 524288 + d1 * 128 + d2)>(%33, %43, %c0_8)
                %reinterpret_cast_9 = memref.reinterpret_cast %arg1 to offset: [%56], sizes: [32, 32, 128], strides: [524288, 128, 1] : memref<32x4096x128xf32> to memref<32x32x128xf32, strided<[524288, 128, 1], offset: ?>>
                %57 = loom.copy_to_tensor %reinterpret_cast_9, %13 on @L1, interconnect : [@horizontal_links], broadcast : [1, 8] : memref<32x32x128xf32, strided<[524288, 128, 1], offset: ?>>, memref<32x32x128xf32> -> tensor<32x32x128xf32>
                %58 = linalg.fill ins(%cst_1 : f32) outs(%32 : tensor<32x32x128xf32>) -> tensor<32x32x128xf32>
                %59 = linalg.batch_matmul ins(%51, %57 : tensor<32x32x32xf32>, tensor<32x32x128xf32>) outs(%58 : tensor<32x32x128xf32>) -> tensor<32x32x128xf32>
                %60 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%59, %arg11, %54 : tensor<32x32x128xf32>, tensor<32x32x128xf32>, tensor<32x32xf32>) outs(%arg11 : tensor<32x32x128xf32>) {
                ^bb0(%in: f32, %in_10: f32, %in_11: f32, %out: f32):
                  %62 = arith.mulf %in_10, %in_11 : f32
                  %63 = arith.addf %in, %62 : f32
                  linalg.yield %63 : f32
                } -> tensor<32x32x128xf32>
                %61 = linalg.copy ins(%50 : tensor<32x32xf32>) outs(%arg9 : tensor<32x32xf32>) -> tensor<32x32xf32>
                affine.yield %61, %55, %60 : tensor<32x32xf32>, tensor<32x32xf32>, tensor<32x32x128xf32>
              }
              %41 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%40#2, %40#1 : tensor<32x32x128xf32>, tensor<32x32xf32>) outs(%28 : tensor<32x32x128xf32>) {
              ^bb0(%in: f32, %in_6: f32, %out: f32):
                %43 = arith.divf %in, %in_6 : f32
                linalg.yield %43 : f32
              } -> tensor<32x32x128xf32>
              %c0_4 = arith.constant 0 : index
              %42 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 524288 + d1 * 128 + d2)>(%33, %34, %c0_4)
              %reinterpret_cast_5 = memref.reinterpret_cast %arg3 to offset: [%42], sizes: [32, 32, 128], strides: [524288, 128, 1] : memref<32x4096x128xf32> to memref<32x32x128xf32, strided<[524288, 128, 1], offset: ?>>
              loom.copy_from_tensor %41, %reinterpret_cast_5 on @DRAM : tensor<32x32x128xf32>, memref<32x32x128xf32, strided<[524288, 128, 1], offset: ?>>
            } {loom.iter_type = #loom.iter_type<temporal>}
          } {loom.iter_type = #loom.iter_type<temporal>}
        } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @y}
      } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @x}
      return
    }
  }
  module attributes {loom.pass_name = "Materialize"} {
    func.func @attention__d0i1_d1i1__f01__d_a_v__BB32__BM32__BN32(%arg0: memref<32x128x4096xf32>, %arg1: memref<32x4096x128xf32>, %arg2: memref<32x4096x128xf32>, %arg3: memref<32x4096x128xf32>) {
      %cst = arith.constant 2.000000e+00 : f32
      %cst_0 = arith.constant 0.12751743074602467 : f64
      %cst_1 = arith.constant 0.000000e+00 : f32
      %cst_2 = arith.constant 1.000000e+00 : f32
      %cst_3 = arith.constant 0xFF800000 : f32
      %c32 = arith.constant 32 : index
      affine.parallel (%arg4) = (0) to (8) {
        affine.parallel (%arg5) = (0) to (8) {
          affine.for %arg6 = 0 to 1 {
            affine.for %arg7 = 0 to 2 {
              %12 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 8 + d1 + d2 * 64)>(%arg4, %arg5, %arg7)
              %13 = loom.alloc [32, 32, 128] on @L1 : memref<32x32x128xf32>
              %14 = loom.alloc [32, 32, 32] on @L1 : memref<32x32x32xf32>
              %15 = loom.init_tensor %14[32, 32, 32] : memref<32x32x32xf32> -> tensor<32x32x32xf32>
              %16 = loom.alloc [32, 128, 32] on @L1 : memref<32x128x32xf32>
              %17 = loom.alloc [32, 32] on @L1 : memref<32x32xf32>
              %18 = loom.init_tensor %17[32, 32] : memref<32x32xf32> -> tensor<32x32xf32>
              %19 = loom.alloc [32, 32] on @L1 : memref<32x32xf32>
              %20 = loom.init_tensor %19[32, 32] : memref<32x32xf32> -> tensor<32x32xf32>
              %21 = loom.alloc [32, 32] on @L1 : memref<32x32xf32>
              %22 = loom.init_tensor %21[32, 32] : memref<32x32xf32> -> tensor<32x32xf32>
              %23 = loom.alloc [32, 32] on @L1 : memref<32x32xf32>
              %24 = loom.init_tensor %23[32, 32] : memref<32x32xf32> -> tensor<32x32xf32>
              %25 = loom.alloc [32, 32] on @L1 : memref<32x32xf32>
              %26 = loom.init_tensor %25[32, 32] : memref<32x32xf32> -> tensor<32x32xf32>
              %27 = loom.alloc [32, 32, 128] on @L1 : memref<32x32x128xf32>
              %28 = loom.init_tensor %27[32, 32, 128] : memref<32x32x128xf32> -> tensor<32x32x128xf32>
              %29 = loom.alloc [32, 32, 128] on @L1 : memref<32x32x128xf32>
              %30 = loom.init_tensor %29[32, 32, 128] : memref<32x32x128xf32> -> tensor<32x32x128xf32>
              %31 = loom.alloc [32, 32, 128] on @L1 : memref<32x32x128xf32>
              %32 = loom.init_tensor %31[32, 32, 128] : memref<32x32x128xf32> -> tensor<32x32x128xf32>
              %33 = arith.muli %arg6, %c32 : index
              %34 = arith.muli %12, %c32 : index
              %c0 = arith.constant 0 : index
              %35 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 524288 + d1 * 128 + d2)>(%33, %34, %c0)
              %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%35], sizes: [32, 32, 128], strides: [524288, 128, 1] : memref<32x4096x128xf32> to memref<32x32x128xf32, strided<[524288, 128, 1], offset: ?>>
              %36 = loom.copy_to_tensor %reinterpret_cast, %27 on @L1, interconnect : [], broadcast : [1, 1] : memref<32x32x128xf32, strided<[524288, 128, 1], offset: ?>>, memref<32x32x128xf32> -> tensor<32x32x128xf32>
              %37 = linalg.fill ins(%cst_1 : f32) outs(%30 : tensor<32x32x128xf32>) -> tensor<32x32x128xf32>
              %38 = linalg.fill ins(%cst_2 : f32) outs(%18 : tensor<32x32xf32>) -> tensor<32x32xf32>
              %39 = linalg.fill ins(%cst_3 : f32) outs(%20 : tensor<32x32xf32>) -> tensor<32x32xf32>
              %40:3 = affine.for %arg8 = 0 to 128 iter_args(%arg9 = %39, %arg10 = %38, %arg11 = %37) -> (tensor<32x32xf32>, tensor<32x32xf32>, tensor<32x32x128xf32>) {
                %43 = arith.muli %arg8, %c32 : index
                %c0_6 = arith.constant 0 : index
                %44 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 524288 + d1 * 4096 + d2)>(%33, %c0_6, %43)
                %reinterpret_cast_7 = memref.reinterpret_cast %arg0 to offset: [%44], sizes: [32, 128, 32], strides: [524288, 4096, 1] : memref<32x128x4096xf32> to memref<32x128x32xf32, strided<[524288, 4096, 1], offset: ?>>
                %45 = loom.copy_to_tensor %reinterpret_cast_7, %16 on @L1, interconnect : [@horizontal_links, @vertical_links], broadcast : [8, 8] : memref<32x128x32xf32, strided<[524288, 4096, 1], offset: ?>>, memref<32x128x32xf32> -> tensor<32x128x32xf32>
                %46 = linalg.fill ins(%cst_1 : f32) outs(%15 : tensor<32x32x32xf32>) -> tensor<32x32x32xf32>
                %47 = linalg.batch_matmul ins(%36, %45 : tensor<32x32x128xf32>, tensor<32x128x32xf32>) outs(%46 : tensor<32x32x32xf32>) -> tensor<32x32x32xf32>
                %48 = linalg.fill ins(%cst_3 : f32) outs(%22 : tensor<32x32xf32>) -> tensor<32x32xf32>
                %49 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%47 : tensor<32x32x32xf32>) outs(%48 : tensor<32x32xf32>) {
                ^bb0(%in: f32, %out: f32):
                  %62 = arith.maximumf %in, %out : f32
                  linalg.yield %62 : f32
                } -> tensor<32x32xf32>
                %50 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%arg9, %49 : tensor<32x32xf32>, tensor<32x32xf32>) outs(%22 : tensor<32x32xf32>) {
                ^bb0(%in: f32, %in_10: f32, %out: f32):
                  %62 = arith.truncf %cst_0 : f64 to f32
                  %63 = arith.mulf %in_10, %62 : f32
                  %64 = arith.cmpf ogt, %in, %63 : f32
                  %65 = arith.select %64, %in, %63 : f32
                  linalg.yield %65 : f32
                } -> tensor<32x32xf32>
                %51 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%47, %50 : tensor<32x32x32xf32>, tensor<32x32xf32>) outs(%15 : tensor<32x32x32xf32>) {
                ^bb0(%in: f32, %in_10: f32, %out: f32):
                  %62 = arith.truncf %cst_0 : f64 to f32
                  %63 = arith.mulf %in, %62 : f32
                  %64 = arith.subf %63, %in_10 : f32
                  %65 = math.powf %cst, %64 : f32
                  linalg.yield %65 : f32
                } -> tensor<32x32x32xf32>
                %52 = linalg.fill ins(%cst_1 : f32) outs(%24 : tensor<32x32xf32>) -> tensor<32x32xf32>
                %53 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%51 : tensor<32x32x32xf32>) outs(%52 : tensor<32x32xf32>) {
                ^bb0(%in: f32, %out: f32):
                  %62 = arith.addf %in, %out : f32
                  linalg.yield %62 : f32
                } -> tensor<32x32xf32>
                %54 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%arg9, %50 : tensor<32x32xf32>, tensor<32x32xf32>) outs(%26 : tensor<32x32xf32>) {
                ^bb0(%in: f32, %in_10: f32, %out: f32):
                  %62 = arith.subf %in, %in_10 : f32
                  %63 = math.powf %cst, %62 : f32
                  linalg.yield %63 : f32
                } -> tensor<32x32xf32>
                %55 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%arg10, %54, %53 : tensor<32x32xf32>, tensor<32x32xf32>, tensor<32x32xf32>) outs(%arg10 : tensor<32x32xf32>) {
                ^bb0(%in: f32, %in_10: f32, %in_11: f32, %out: f32):
                  %62 = arith.mulf %in, %in_10 : f32
                  %63 = arith.addf %62, %in_11 : f32
                  linalg.yield %63 : f32
                } -> tensor<32x32xf32>
                %c0_8 = arith.constant 0 : index
                %56 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 524288 + d1 * 128 + d2)>(%33, %43, %c0_8)
                %reinterpret_cast_9 = memref.reinterpret_cast %arg1 to offset: [%56], sizes: [32, 32, 128], strides: [524288, 128, 1] : memref<32x4096x128xf32> to memref<32x32x128xf32, strided<[524288, 128, 1], offset: ?>>
                %57 = loom.copy_to_tensor %reinterpret_cast_9, %13 on @L1, interconnect : [@vertical_links], broadcast : [8, 1] : memref<32x32x128xf32, strided<[524288, 128, 1], offset: ?>>, memref<32x32x128xf32> -> tensor<32x32x128xf32>
                %58 = linalg.fill ins(%cst_1 : f32) outs(%32 : tensor<32x32x128xf32>) -> tensor<32x32x128xf32>
                %59 = linalg.batch_matmul ins(%51, %57 : tensor<32x32x32xf32>, tensor<32x32x128xf32>) outs(%58 : tensor<32x32x128xf32>) -> tensor<32x32x128xf32>
                %60 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%59, %arg11, %54 : tensor<32x32x128xf32>, tensor<32x32x128xf32>, tensor<32x32xf32>) outs(%arg11 : tensor<32x32x128xf32>) {
                ^bb0(%in: f32, %in_10: f32, %in_11: f32, %out: f32):
                  %62 = arith.mulf %in_10, %in_11 : f32
                  %63 = arith.addf %in, %62 : f32
                  linalg.yield %63 : f32
                } -> tensor<32x32x128xf32>
                %61 = linalg.copy ins(%50 : tensor<32x32xf32>) outs(%arg9 : tensor<32x32xf32>) -> tensor<32x32xf32>
                affine.yield %61, %55, %60 : tensor<32x32xf32>, tensor<32x32xf32>, tensor<32x32x128xf32>
              }
              %41 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%40#2, %40#1 : tensor<32x32x128xf32>, tensor<32x32xf32>) outs(%28 : tensor<32x32x128xf32>) {
              ^bb0(%in: f32, %in_6: f32, %out: f32):
                %43 = arith.divf %in, %in_6 : f32
                linalg.yield %43 : f32
              } -> tensor<32x32x128xf32>
              %c0_4 = arith.constant 0 : index
              %42 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 524288 + d1 * 128 + d2)>(%33, %34, %c0_4)
              %reinterpret_cast_5 = memref.reinterpret_cast %arg3 to offset: [%42], sizes: [32, 32, 128], strides: [524288, 128, 1] : memref<32x4096x128xf32> to memref<32x32x128xf32, strided<[524288, 128, 1], offset: ?>>
              loom.copy_from_tensor %41, %reinterpret_cast_5 on @DRAM : tensor<32x32x128xf32>, memref<32x32x128xf32, strided<[524288, 128, 1], offset: ?>>
            } {loom.iter_type = #loom.iter_type<temporal>}
          } {loom.iter_type = #loom.iter_type<temporal>}
        } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @y}
      } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @x}
      return
    }
  }
  module attributes {loom.pass_name = "Materialize"} {
    func.func @attention__d0i1_d1i1__f01__d_h_d__BB32__BM32__BN32(%arg0: memref<32x128x4096xf32>, %arg1: memref<32x4096x128xf32>, %arg2: memref<32x4096x128xf32>, %arg3: memref<32x4096x128xf32>) {
      %cst = arith.constant 2.000000e+00 : f32
      %cst_0 = arith.constant 0.12751743074602467 : f64
      %cst_1 = arith.constant 0.000000e+00 : f32
      %cst_2 = arith.constant 1.000000e+00 : f32
      %cst_3 = arith.constant 0xFF800000 : f32
      %c32 = arith.constant 32 : index
      affine.parallel (%arg4) = (0) to (8) {
        affine.parallel (%arg5) = (0) to (8) {
          affine.for %arg6 = 0 to 1 {
            affine.for %arg7 = 0 to 2 {
              %12 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 8 + d1 + d2 * 64)>(%arg4, %arg5, %arg7)
              %13 = loom.alloc [32, 32, 128] on @L1 : memref<32x32x128xf32>
              %14 = loom.alloc [32, 32, 32] on @L1 : memref<32x32x32xf32>
              %15 = loom.init_tensor %14[32, 32, 32] : memref<32x32x32xf32> -> tensor<32x32x32xf32>
              %16 = loom.alloc [32, 128, 32] on @L1 : memref<32x128x32xf32>
              %17 = loom.alloc [32, 32] on @L1 : memref<32x32xf32>
              %18 = loom.init_tensor %17[32, 32] : memref<32x32xf32> -> tensor<32x32xf32>
              %19 = loom.alloc [32, 32] on @L1 : memref<32x32xf32>
              %20 = loom.init_tensor %19[32, 32] : memref<32x32xf32> -> tensor<32x32xf32>
              %21 = loom.alloc [32, 32] on @L1 : memref<32x32xf32>
              %22 = loom.init_tensor %21[32, 32] : memref<32x32xf32> -> tensor<32x32xf32>
              %23 = loom.alloc [32, 32] on @L1 : memref<32x32xf32>
              %24 = loom.init_tensor %23[32, 32] : memref<32x32xf32> -> tensor<32x32xf32>
              %25 = loom.alloc [32, 32] on @L1 : memref<32x32xf32>
              %26 = loom.init_tensor %25[32, 32] : memref<32x32xf32> -> tensor<32x32xf32>
              %27 = loom.alloc [32, 32, 128] on @L1 : memref<32x32x128xf32>
              %28 = loom.init_tensor %27[32, 32, 128] : memref<32x32x128xf32> -> tensor<32x32x128xf32>
              %29 = loom.alloc [32, 32, 128] on @L1 : memref<32x32x128xf32>
              %30 = loom.init_tensor %29[32, 32, 128] : memref<32x32x128xf32> -> tensor<32x32x128xf32>
              %31 = loom.alloc [32, 32, 128] on @L1 : memref<32x32x128xf32>
              %32 = loom.init_tensor %31[32, 32, 128] : memref<32x32x128xf32> -> tensor<32x32x128xf32>
              %33 = arith.muli %arg6, %c32 : index
              %34 = arith.muli %12, %c32 : index
              %c0 = arith.constant 0 : index
              %35 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 524288 + d1 * 128 + d2)>(%33, %34, %c0)
              %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%35], sizes: [32, 32, 128], strides: [524288, 128, 1] : memref<32x4096x128xf32> to memref<32x32x128xf32, strided<[524288, 128, 1], offset: ?>>
              %36 = loom.copy_to_tensor %reinterpret_cast, %27 on @L1, interconnect : [], broadcast : [1, 1] : memref<32x32x128xf32, strided<[524288, 128, 1], offset: ?>>, memref<32x32x128xf32> -> tensor<32x32x128xf32>
              %37 = linalg.fill ins(%cst_1 : f32) outs(%30 : tensor<32x32x128xf32>) -> tensor<32x32x128xf32>
              %38 = linalg.fill ins(%cst_2 : f32) outs(%18 : tensor<32x32xf32>) -> tensor<32x32xf32>
              %39 = linalg.fill ins(%cst_3 : f32) outs(%20 : tensor<32x32xf32>) -> tensor<32x32xf32>
              %40:3 = affine.for %arg8 = 0 to 128 iter_args(%arg9 = %39, %arg10 = %38, %arg11 = %37) -> (tensor<32x32xf32>, tensor<32x32xf32>, tensor<32x32x128xf32>) {
                %43 = arith.muli %arg8, %c32 : index
                %c0_6 = arith.constant 0 : index
                %44 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 524288 + d1 * 4096 + d2)>(%33, %c0_6, %43)
                %reinterpret_cast_7 = memref.reinterpret_cast %arg0 to offset: [%44], sizes: [32, 128, 32], strides: [524288, 4096, 1] : memref<32x128x4096xf32> to memref<32x128x32xf32, strided<[524288, 4096, 1], offset: ?>>
                %45 = loom.copy_to_tensor %reinterpret_cast_7, %16 on @L1, interconnect : [@horizontal_links], broadcast : [1, 8] : memref<32x128x32xf32, strided<[524288, 4096, 1], offset: ?>>, memref<32x128x32xf32> -> tensor<32x128x32xf32>
                %46 = linalg.fill ins(%cst_1 : f32) outs(%15 : tensor<32x32x32xf32>) -> tensor<32x32x32xf32>
                %47 = linalg.batch_matmul ins(%36, %45 : tensor<32x32x128xf32>, tensor<32x128x32xf32>) outs(%46 : tensor<32x32x32xf32>) -> tensor<32x32x32xf32>
                %48 = linalg.fill ins(%cst_3 : f32) outs(%22 : tensor<32x32xf32>) -> tensor<32x32xf32>
                %49 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%47 : tensor<32x32x32xf32>) outs(%48 : tensor<32x32xf32>) {
                ^bb0(%in: f32, %out: f32):
                  %62 = arith.maximumf %in, %out : f32
                  linalg.yield %62 : f32
                } -> tensor<32x32xf32>
                %50 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%arg9, %49 : tensor<32x32xf32>, tensor<32x32xf32>) outs(%22 : tensor<32x32xf32>) {
                ^bb0(%in: f32, %in_10: f32, %out: f32):
                  %62 = arith.truncf %cst_0 : f64 to f32
                  %63 = arith.mulf %in_10, %62 : f32
                  %64 = arith.cmpf ogt, %in, %63 : f32
                  %65 = arith.select %64, %in, %63 : f32
                  linalg.yield %65 : f32
                } -> tensor<32x32xf32>
                %51 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%47, %50 : tensor<32x32x32xf32>, tensor<32x32xf32>) outs(%15 : tensor<32x32x32xf32>) {
                ^bb0(%in: f32, %in_10: f32, %out: f32):
                  %62 = arith.truncf %cst_0 : f64 to f32
                  %63 = arith.mulf %in, %62 : f32
                  %64 = arith.subf %63, %in_10 : f32
                  %65 = math.powf %cst, %64 : f32
                  linalg.yield %65 : f32
                } -> tensor<32x32x32xf32>
                %52 = linalg.fill ins(%cst_1 : f32) outs(%24 : tensor<32x32xf32>) -> tensor<32x32xf32>
                %53 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%51 : tensor<32x32x32xf32>) outs(%52 : tensor<32x32xf32>) {
                ^bb0(%in: f32, %out: f32):
                  %62 = arith.addf %in, %out : f32
                  linalg.yield %62 : f32
                } -> tensor<32x32xf32>
                %54 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%arg9, %50 : tensor<32x32xf32>, tensor<32x32xf32>) outs(%26 : tensor<32x32xf32>) {
                ^bb0(%in: f32, %in_10: f32, %out: f32):
                  %62 = arith.subf %in, %in_10 : f32
                  %63 = math.powf %cst, %62 : f32
                  linalg.yield %63 : f32
                } -> tensor<32x32xf32>
                %55 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%arg10, %54, %53 : tensor<32x32xf32>, tensor<32x32xf32>, tensor<32x32xf32>) outs(%arg10 : tensor<32x32xf32>) {
                ^bb0(%in: f32, %in_10: f32, %in_11: f32, %out: f32):
                  %62 = arith.mulf %in, %in_10 : f32
                  %63 = arith.addf %62, %in_11 : f32
                  linalg.yield %63 : f32
                } -> tensor<32x32xf32>
                %c0_8 = arith.constant 0 : index
                %56 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 524288 + d1 * 128 + d2)>(%33, %43, %c0_8)
                %reinterpret_cast_9 = memref.reinterpret_cast %arg1 to offset: [%56], sizes: [32, 32, 128], strides: [524288, 128, 1] : memref<32x4096x128xf32> to memref<32x32x128xf32, strided<[524288, 128, 1], offset: ?>>
                %57 = loom.copy_to_tensor %reinterpret_cast_9, %13 on @L1, interconnect : [], broadcast : [1, 1] : memref<32x32x128xf32, strided<[524288, 128, 1], offset: ?>>, memref<32x32x128xf32> -> tensor<32x32x128xf32>
                %58 = linalg.fill ins(%cst_1 : f32) outs(%32 : tensor<32x32x128xf32>) -> tensor<32x32x128xf32>
                %59 = linalg.batch_matmul ins(%51, %57 : tensor<32x32x32xf32>, tensor<32x32x128xf32>) outs(%58 : tensor<32x32x128xf32>) -> tensor<32x32x128xf32>
                %60 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%59, %arg11, %54 : tensor<32x32x128xf32>, tensor<32x32x128xf32>, tensor<32x32xf32>) outs(%arg11 : tensor<32x32x128xf32>) {
                ^bb0(%in: f32, %in_10: f32, %in_11: f32, %out: f32):
                  %62 = arith.mulf %in_10, %in_11 : f32
                  %63 = arith.addf %in, %62 : f32
                  linalg.yield %63 : f32
                } -> tensor<32x32x128xf32>
                %61 = linalg.copy ins(%50 : tensor<32x32xf32>) outs(%arg9 : tensor<32x32xf32>) -> tensor<32x32xf32>
                affine.yield %61, %55, %60 : tensor<32x32xf32>, tensor<32x32xf32>, tensor<32x32x128xf32>
              }
              %41 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%40#2, %40#1 : tensor<32x32x128xf32>, tensor<32x32xf32>) outs(%28 : tensor<32x32x128xf32>) {
              ^bb0(%in: f32, %in_6: f32, %out: f32):
                %43 = arith.divf %in, %in_6 : f32
                linalg.yield %43 : f32
              } -> tensor<32x32x128xf32>
              %c0_4 = arith.constant 0 : index
              %42 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 524288 + d1 * 128 + d2)>(%33, %34, %c0_4)
              %reinterpret_cast_5 = memref.reinterpret_cast %arg3 to offset: [%42], sizes: [32, 32, 128], strides: [524288, 128, 1] : memref<32x4096x128xf32> to memref<32x32x128xf32, strided<[524288, 128, 1], offset: ?>>
              loom.copy_from_tensor %41, %reinterpret_cast_5 on @DRAM : tensor<32x32x128xf32>, memref<32x32x128xf32, strided<[524288, 128, 1], offset: ?>>
            } {loom.iter_type = #loom.iter_type<temporal>}
          } {loom.iter_type = #loom.iter_type<temporal>}
        } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @y}
      } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @x}
      return
    }
  }
  module attributes {loom.pass_name = "Materialize"} {
    func.func @attention__d0i1_d1i1__f01__d_h_a__BB32__BM32__BN32(%arg0: memref<32x128x4096xf32>, %arg1: memref<32x4096x128xf32>, %arg2: memref<32x4096x128xf32>, %arg3: memref<32x4096x128xf32>) {
      %cst = arith.constant 2.000000e+00 : f32
      %cst_0 = arith.constant 0.12751743074602467 : f64
      %cst_1 = arith.constant 0.000000e+00 : f32
      %cst_2 = arith.constant 1.000000e+00 : f32
      %cst_3 = arith.constant 0xFF800000 : f32
      %c32 = arith.constant 32 : index
      affine.parallel (%arg4) = (0) to (8) {
        affine.parallel (%arg5) = (0) to (8) {
          affine.for %arg6 = 0 to 1 {
            affine.for %arg7 = 0 to 2 {
              %12 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 8 + d1 + d2 * 64)>(%arg4, %arg5, %arg7)
              %13 = loom.alloc [32, 32, 128] on @L1 : memref<32x32x128xf32>
              %14 = loom.alloc [32, 32, 32] on @L1 : memref<32x32x32xf32>
              %15 = loom.init_tensor %14[32, 32, 32] : memref<32x32x32xf32> -> tensor<32x32x32xf32>
              %16 = loom.alloc [32, 128, 32] on @L1 : memref<32x128x32xf32>
              %17 = loom.alloc [32, 32] on @L1 : memref<32x32xf32>
              %18 = loom.init_tensor %17[32, 32] : memref<32x32xf32> -> tensor<32x32xf32>
              %19 = loom.alloc [32, 32] on @L1 : memref<32x32xf32>
              %20 = loom.init_tensor %19[32, 32] : memref<32x32xf32> -> tensor<32x32xf32>
              %21 = loom.alloc [32, 32] on @L1 : memref<32x32xf32>
              %22 = loom.init_tensor %21[32, 32] : memref<32x32xf32> -> tensor<32x32xf32>
              %23 = loom.alloc [32, 32] on @L1 : memref<32x32xf32>
              %24 = loom.init_tensor %23[32, 32] : memref<32x32xf32> -> tensor<32x32xf32>
              %25 = loom.alloc [32, 32] on @L1 : memref<32x32xf32>
              %26 = loom.init_tensor %25[32, 32] : memref<32x32xf32> -> tensor<32x32xf32>
              %27 = loom.alloc [32, 32, 128] on @L1 : memref<32x32x128xf32>
              %28 = loom.init_tensor %27[32, 32, 128] : memref<32x32x128xf32> -> tensor<32x32x128xf32>
              %29 = loom.alloc [32, 32, 128] on @L1 : memref<32x32x128xf32>
              %30 = loom.init_tensor %29[32, 32, 128] : memref<32x32x128xf32> -> tensor<32x32x128xf32>
              %31 = loom.alloc [32, 32, 128] on @L1 : memref<32x32x128xf32>
              %32 = loom.init_tensor %31[32, 32, 128] : memref<32x32x128xf32> -> tensor<32x32x128xf32>
              %33 = arith.muli %arg6, %c32 : index
              %34 = arith.muli %12, %c32 : index
              %c0 = arith.constant 0 : index
              %35 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 524288 + d1 * 128 + d2)>(%33, %34, %c0)
              %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%35], sizes: [32, 32, 128], strides: [524288, 128, 1] : memref<32x4096x128xf32> to memref<32x32x128xf32, strided<[524288, 128, 1], offset: ?>>
              %36 = loom.copy_to_tensor %reinterpret_cast, %27 on @L1, interconnect : [], broadcast : [1, 1] : memref<32x32x128xf32, strided<[524288, 128, 1], offset: ?>>, memref<32x32x128xf32> -> tensor<32x32x128xf32>
              %37 = linalg.fill ins(%cst_1 : f32) outs(%30 : tensor<32x32x128xf32>) -> tensor<32x32x128xf32>
              %38 = linalg.fill ins(%cst_2 : f32) outs(%18 : tensor<32x32xf32>) -> tensor<32x32xf32>
              %39 = linalg.fill ins(%cst_3 : f32) outs(%20 : tensor<32x32xf32>) -> tensor<32x32xf32>
              %40:3 = affine.for %arg8 = 0 to 128 iter_args(%arg9 = %39, %arg10 = %38, %arg11 = %37) -> (tensor<32x32xf32>, tensor<32x32xf32>, tensor<32x32x128xf32>) {
                %43 = arith.muli %arg8, %c32 : index
                %c0_6 = arith.constant 0 : index
                %44 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 524288 + d1 * 4096 + d2)>(%33, %c0_6, %43)
                %reinterpret_cast_7 = memref.reinterpret_cast %arg0 to offset: [%44], sizes: [32, 128, 32], strides: [524288, 4096, 1] : memref<32x128x4096xf32> to memref<32x128x32xf32, strided<[524288, 4096, 1], offset: ?>>
                %45 = loom.copy_to_tensor %reinterpret_cast_7, %16 on @L1, interconnect : [@horizontal_links], broadcast : [1, 8] : memref<32x128x32xf32, strided<[524288, 4096, 1], offset: ?>>, memref<32x128x32xf32> -> tensor<32x128x32xf32>
                %46 = linalg.fill ins(%cst_1 : f32) outs(%15 : tensor<32x32x32xf32>) -> tensor<32x32x32xf32>
                %47 = linalg.batch_matmul ins(%36, %45 : tensor<32x32x128xf32>, tensor<32x128x32xf32>) outs(%46 : tensor<32x32x32xf32>) -> tensor<32x32x32xf32>
                %48 = linalg.fill ins(%cst_3 : f32) outs(%22 : tensor<32x32xf32>) -> tensor<32x32xf32>
                %49 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%47 : tensor<32x32x32xf32>) outs(%48 : tensor<32x32xf32>) {
                ^bb0(%in: f32, %out: f32):
                  %62 = arith.maximumf %in, %out : f32
                  linalg.yield %62 : f32
                } -> tensor<32x32xf32>
                %50 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%arg9, %49 : tensor<32x32xf32>, tensor<32x32xf32>) outs(%22 : tensor<32x32xf32>) {
                ^bb0(%in: f32, %in_10: f32, %out: f32):
                  %62 = arith.truncf %cst_0 : f64 to f32
                  %63 = arith.mulf %in_10, %62 : f32
                  %64 = arith.cmpf ogt, %in, %63 : f32
                  %65 = arith.select %64, %in, %63 : f32
                  linalg.yield %65 : f32
                } -> tensor<32x32xf32>
                %51 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%47, %50 : tensor<32x32x32xf32>, tensor<32x32xf32>) outs(%15 : tensor<32x32x32xf32>) {
                ^bb0(%in: f32, %in_10: f32, %out: f32):
                  %62 = arith.truncf %cst_0 : f64 to f32
                  %63 = arith.mulf %in, %62 : f32
                  %64 = arith.subf %63, %in_10 : f32
                  %65 = math.powf %cst, %64 : f32
                  linalg.yield %65 : f32
                } -> tensor<32x32x32xf32>
                %52 = linalg.fill ins(%cst_1 : f32) outs(%24 : tensor<32x32xf32>) -> tensor<32x32xf32>
                %53 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%51 : tensor<32x32x32xf32>) outs(%52 : tensor<32x32xf32>) {
                ^bb0(%in: f32, %out: f32):
                  %62 = arith.addf %in, %out : f32
                  linalg.yield %62 : f32
                } -> tensor<32x32xf32>
                %54 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%arg9, %50 : tensor<32x32xf32>, tensor<32x32xf32>) outs(%26 : tensor<32x32xf32>) {
                ^bb0(%in: f32, %in_10: f32, %out: f32):
                  %62 = arith.subf %in, %in_10 : f32
                  %63 = math.powf %cst, %62 : f32
                  linalg.yield %63 : f32
                } -> tensor<32x32xf32>
                %55 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%arg10, %54, %53 : tensor<32x32xf32>, tensor<32x32xf32>, tensor<32x32xf32>) outs(%arg10 : tensor<32x32xf32>) {
                ^bb0(%in: f32, %in_10: f32, %in_11: f32, %out: f32):
                  %62 = arith.mulf %in, %in_10 : f32
                  %63 = arith.addf %62, %in_11 : f32
                  linalg.yield %63 : f32
                } -> tensor<32x32xf32>
                %c0_8 = arith.constant 0 : index
                %56 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 524288 + d1 * 128 + d2)>(%33, %43, %c0_8)
                %reinterpret_cast_9 = memref.reinterpret_cast %arg1 to offset: [%56], sizes: [32, 32, 128], strides: [524288, 128, 1] : memref<32x4096x128xf32> to memref<32x32x128xf32, strided<[524288, 128, 1], offset: ?>>
                %57 = loom.copy_to_tensor %reinterpret_cast_9, %13 on @L1, interconnect : [@horizontal_links, @vertical_links], broadcast : [8, 8] : memref<32x32x128xf32, strided<[524288, 128, 1], offset: ?>>, memref<32x32x128xf32> -> tensor<32x32x128xf32>
                %58 = linalg.fill ins(%cst_1 : f32) outs(%32 : tensor<32x32x128xf32>) -> tensor<32x32x128xf32>
                %59 = linalg.batch_matmul ins(%51, %57 : tensor<32x32x32xf32>, tensor<32x32x128xf32>) outs(%58 : tensor<32x32x128xf32>) -> tensor<32x32x128xf32>
                %60 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%59, %arg11, %54 : tensor<32x32x128xf32>, tensor<32x32x128xf32>, tensor<32x32xf32>) outs(%arg11 : tensor<32x32x128xf32>) {
                ^bb0(%in: f32, %in_10: f32, %in_11: f32, %out: f32):
                  %62 = arith.mulf %in_10, %in_11 : f32
                  %63 = arith.addf %in, %62 : f32
                  linalg.yield %63 : f32
                } -> tensor<32x32x128xf32>
                %61 = linalg.copy ins(%50 : tensor<32x32xf32>) outs(%arg9 : tensor<32x32xf32>) -> tensor<32x32xf32>
                affine.yield %61, %55, %60 : tensor<32x32xf32>, tensor<32x32xf32>, tensor<32x32x128xf32>
              }
              %41 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%40#2, %40#1 : tensor<32x32x128xf32>, tensor<32x32xf32>) outs(%28 : tensor<32x32x128xf32>) {
              ^bb0(%in: f32, %in_6: f32, %out: f32):
                %43 = arith.divf %in, %in_6 : f32
                linalg.yield %43 : f32
              } -> tensor<32x32x128xf32>
              %c0_4 = arith.constant 0 : index
              %42 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 524288 + d1 * 128 + d2)>(%33, %34, %c0_4)
              %reinterpret_cast_5 = memref.reinterpret_cast %arg3 to offset: [%42], sizes: [32, 32, 128], strides: [524288, 128, 1] : memref<32x4096x128xf32> to memref<32x32x128xf32, strided<[524288, 128, 1], offset: ?>>
              loom.copy_from_tensor %41, %reinterpret_cast_5 on @DRAM : tensor<32x32x128xf32>, memref<32x32x128xf32, strided<[524288, 128, 1], offset: ?>>
            } {loom.iter_type = #loom.iter_type<temporal>}
          } {loom.iter_type = #loom.iter_type<temporal>}
        } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @y}
      } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @x}
      return
    }
  }
  module attributes {loom.pass_name = "Materialize"} {
    func.func @attention__d0i1_d1i1__f01__d_h_h__BB32__BM32__BN32(%arg0: memref<32x128x4096xf32>, %arg1: memref<32x4096x128xf32>, %arg2: memref<32x4096x128xf32>, %arg3: memref<32x4096x128xf32>) {
      %cst = arith.constant 2.000000e+00 : f32
      %cst_0 = arith.constant 0.12751743074602467 : f64
      %cst_1 = arith.constant 0.000000e+00 : f32
      %cst_2 = arith.constant 1.000000e+00 : f32
      %cst_3 = arith.constant 0xFF800000 : f32
      %c32 = arith.constant 32 : index
      affine.parallel (%arg4) = (0) to (8) {
        affine.parallel (%arg5) = (0) to (8) {
          affine.for %arg6 = 0 to 1 {
            affine.for %arg7 = 0 to 2 {
              %12 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 8 + d1 + d2 * 64)>(%arg4, %arg5, %arg7)
              %13 = loom.alloc [32, 32, 128] on @L1 : memref<32x32x128xf32>
              %14 = loom.alloc [32, 32, 32] on @L1 : memref<32x32x32xf32>
              %15 = loom.init_tensor %14[32, 32, 32] : memref<32x32x32xf32> -> tensor<32x32x32xf32>
              %16 = loom.alloc [32, 128, 32] on @L1 : memref<32x128x32xf32>
              %17 = loom.alloc [32, 32] on @L1 : memref<32x32xf32>
              %18 = loom.init_tensor %17[32, 32] : memref<32x32xf32> -> tensor<32x32xf32>
              %19 = loom.alloc [32, 32] on @L1 : memref<32x32xf32>
              %20 = loom.init_tensor %19[32, 32] : memref<32x32xf32> -> tensor<32x32xf32>
              %21 = loom.alloc [32, 32] on @L1 : memref<32x32xf32>
              %22 = loom.init_tensor %21[32, 32] : memref<32x32xf32> -> tensor<32x32xf32>
              %23 = loom.alloc [32, 32] on @L1 : memref<32x32xf32>
              %24 = loom.init_tensor %23[32, 32] : memref<32x32xf32> -> tensor<32x32xf32>
              %25 = loom.alloc [32, 32] on @L1 : memref<32x32xf32>
              %26 = loom.init_tensor %25[32, 32] : memref<32x32xf32> -> tensor<32x32xf32>
              %27 = loom.alloc [32, 32, 128] on @L1 : memref<32x32x128xf32>
              %28 = loom.init_tensor %27[32, 32, 128] : memref<32x32x128xf32> -> tensor<32x32x128xf32>
              %29 = loom.alloc [32, 32, 128] on @L1 : memref<32x32x128xf32>
              %30 = loom.init_tensor %29[32, 32, 128] : memref<32x32x128xf32> -> tensor<32x32x128xf32>
              %31 = loom.alloc [32, 32, 128] on @L1 : memref<32x32x128xf32>
              %32 = loom.init_tensor %31[32, 32, 128] : memref<32x32x128xf32> -> tensor<32x32x128xf32>
              %33 = arith.muli %arg6, %c32 : index
              %34 = arith.muli %12, %c32 : index
              %c0 = arith.constant 0 : index
              %35 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 524288 + d1 * 128 + d2)>(%33, %34, %c0)
              %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%35], sizes: [32, 32, 128], strides: [524288, 128, 1] : memref<32x4096x128xf32> to memref<32x32x128xf32, strided<[524288, 128, 1], offset: ?>>
              %36 = loom.copy_to_tensor %reinterpret_cast, %27 on @L1, interconnect : [], broadcast : [1, 1] : memref<32x32x128xf32, strided<[524288, 128, 1], offset: ?>>, memref<32x32x128xf32> -> tensor<32x32x128xf32>
              %37 = linalg.fill ins(%cst_1 : f32) outs(%30 : tensor<32x32x128xf32>) -> tensor<32x32x128xf32>
              %38 = linalg.fill ins(%cst_2 : f32) outs(%18 : tensor<32x32xf32>) -> tensor<32x32xf32>
              %39 = linalg.fill ins(%cst_3 : f32) outs(%20 : tensor<32x32xf32>) -> tensor<32x32xf32>
              %40:3 = affine.for %arg8 = 0 to 128 iter_args(%arg9 = %39, %arg10 = %38, %arg11 = %37) -> (tensor<32x32xf32>, tensor<32x32xf32>, tensor<32x32x128xf32>) {
                %43 = arith.muli %arg8, %c32 : index
                %c0_6 = arith.constant 0 : index
                %44 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 524288 + d1 * 4096 + d2)>(%33, %c0_6, %43)
                %reinterpret_cast_7 = memref.reinterpret_cast %arg0 to offset: [%44], sizes: [32, 128, 32], strides: [524288, 4096, 1] : memref<32x128x4096xf32> to memref<32x128x32xf32, strided<[524288, 4096, 1], offset: ?>>
                %45 = loom.copy_to_tensor %reinterpret_cast_7, %16 on @L1, interconnect : [@horizontal_links], broadcast : [1, 8] : memref<32x128x32xf32, strided<[524288, 4096, 1], offset: ?>>, memref<32x128x32xf32> -> tensor<32x128x32xf32>
                %46 = linalg.fill ins(%cst_1 : f32) outs(%15 : tensor<32x32x32xf32>) -> tensor<32x32x32xf32>
                %47 = linalg.batch_matmul ins(%36, %45 : tensor<32x32x128xf32>, tensor<32x128x32xf32>) outs(%46 : tensor<32x32x32xf32>) -> tensor<32x32x32xf32>
                %48 = linalg.fill ins(%cst_3 : f32) outs(%22 : tensor<32x32xf32>) -> tensor<32x32xf32>
                %49 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%47 : tensor<32x32x32xf32>) outs(%48 : tensor<32x32xf32>) {
                ^bb0(%in: f32, %out: f32):
                  %62 = arith.maximumf %in, %out : f32
                  linalg.yield %62 : f32
                } -> tensor<32x32xf32>
                %50 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%arg9, %49 : tensor<32x32xf32>, tensor<32x32xf32>) outs(%22 : tensor<32x32xf32>) {
                ^bb0(%in: f32, %in_10: f32, %out: f32):
                  %62 = arith.truncf %cst_0 : f64 to f32
                  %63 = arith.mulf %in_10, %62 : f32
                  %64 = arith.cmpf ogt, %in, %63 : f32
                  %65 = arith.select %64, %in, %63 : f32
                  linalg.yield %65 : f32
                } -> tensor<32x32xf32>
                %51 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%47, %50 : tensor<32x32x32xf32>, tensor<32x32xf32>) outs(%15 : tensor<32x32x32xf32>) {
                ^bb0(%in: f32, %in_10: f32, %out: f32):
                  %62 = arith.truncf %cst_0 : f64 to f32
                  %63 = arith.mulf %in, %62 : f32
                  %64 = arith.subf %63, %in_10 : f32
                  %65 = math.powf %cst, %64 : f32
                  linalg.yield %65 : f32
                } -> tensor<32x32x32xf32>
                %52 = linalg.fill ins(%cst_1 : f32) outs(%24 : tensor<32x32xf32>) -> tensor<32x32xf32>
                %53 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%51 : tensor<32x32x32xf32>) outs(%52 : tensor<32x32xf32>) {
                ^bb0(%in: f32, %out: f32):
                  %62 = arith.addf %in, %out : f32
                  linalg.yield %62 : f32
                } -> tensor<32x32xf32>
                %54 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%arg9, %50 : tensor<32x32xf32>, tensor<32x32xf32>) outs(%26 : tensor<32x32xf32>) {
                ^bb0(%in: f32, %in_10: f32, %out: f32):
                  %62 = arith.subf %in, %in_10 : f32
                  %63 = math.powf %cst, %62 : f32
                  linalg.yield %63 : f32
                } -> tensor<32x32xf32>
                %55 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%arg10, %54, %53 : tensor<32x32xf32>, tensor<32x32xf32>, tensor<32x32xf32>) outs(%arg10 : tensor<32x32xf32>) {
                ^bb0(%in: f32, %in_10: f32, %in_11: f32, %out: f32):
                  %62 = arith.mulf %in, %in_10 : f32
                  %63 = arith.addf %62, %in_11 : f32
                  linalg.yield %63 : f32
                } -> tensor<32x32xf32>
                %c0_8 = arith.constant 0 : index
                %56 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 524288 + d1 * 128 + d2)>(%33, %43, %c0_8)
                %reinterpret_cast_9 = memref.reinterpret_cast %arg1 to offset: [%56], sizes: [32, 32, 128], strides: [524288, 128, 1] : memref<32x4096x128xf32> to memref<32x32x128xf32, strided<[524288, 128, 1], offset: ?>>
                %57 = loom.copy_to_tensor %reinterpret_cast_9, %13 on @L1, interconnect : [@horizontal_links], broadcast : [1, 8] : memref<32x32x128xf32, strided<[524288, 128, 1], offset: ?>>, memref<32x32x128xf32> -> tensor<32x32x128xf32>
                %58 = linalg.fill ins(%cst_1 : f32) outs(%32 : tensor<32x32x128xf32>) -> tensor<32x32x128xf32>
                %59 = linalg.batch_matmul ins(%51, %57 : tensor<32x32x32xf32>, tensor<32x32x128xf32>) outs(%58 : tensor<32x32x128xf32>) -> tensor<32x32x128xf32>
                %60 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%59, %arg11, %54 : tensor<32x32x128xf32>, tensor<32x32x128xf32>, tensor<32x32xf32>) outs(%arg11 : tensor<32x32x128xf32>) {
                ^bb0(%in: f32, %in_10: f32, %in_11: f32, %out: f32):
                  %62 = arith.mulf %in_10, %in_11 : f32
                  %63 = arith.addf %in, %62 : f32
                  linalg.yield %63 : f32
                } -> tensor<32x32x128xf32>
                %61 = linalg.copy ins(%50 : tensor<32x32xf32>) outs(%arg9 : tensor<32x32xf32>) -> tensor<32x32xf32>
                affine.yield %61, %55, %60 : tensor<32x32xf32>, tensor<32x32xf32>, tensor<32x32x128xf32>
              }
              %41 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%40#2, %40#1 : tensor<32x32x128xf32>, tensor<32x32xf32>) outs(%28 : tensor<32x32x128xf32>) {
              ^bb0(%in: f32, %in_6: f32, %out: f32):
                %43 = arith.divf %in, %in_6 : f32
                linalg.yield %43 : f32
              } -> tensor<32x32x128xf32>
              %c0_4 = arith.constant 0 : index
              %42 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 524288 + d1 * 128 + d2)>(%33, %34, %c0_4)
              %reinterpret_cast_5 = memref.reinterpret_cast %arg3 to offset: [%42], sizes: [32, 32, 128], strides: [524288, 128, 1] : memref<32x4096x128xf32> to memref<32x32x128xf32, strided<[524288, 128, 1], offset: ?>>
              loom.copy_from_tensor %41, %reinterpret_cast_5 on @DRAM : tensor<32x32x128xf32>, memref<32x32x128xf32, strided<[524288, 128, 1], offset: ?>>
            } {loom.iter_type = #loom.iter_type<temporal>}
          } {loom.iter_type = #loom.iter_type<temporal>}
        } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @y}
      } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @x}
      return
    }
  }
  module attributes {loom.pass_name = "Materialize"} {
    func.func @attention__d0i1_d1i1__f01__d_h_v__BB32__BM32__BN32(%arg0: memref<32x128x4096xf32>, %arg1: memref<32x4096x128xf32>, %arg2: memref<32x4096x128xf32>, %arg3: memref<32x4096x128xf32>) {
      %cst = arith.constant 2.000000e+00 : f32
      %cst_0 = arith.constant 0.12751743074602467 : f64
      %cst_1 = arith.constant 0.000000e+00 : f32
      %cst_2 = arith.constant 1.000000e+00 : f32
      %cst_3 = arith.constant 0xFF800000 : f32
      %c32 = arith.constant 32 : index
      affine.parallel (%arg4) = (0) to (8) {
        affine.parallel (%arg5) = (0) to (8) {
          affine.for %arg6 = 0 to 1 {
            affine.for %arg7 = 0 to 2 {
              %12 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 8 + d1 + d2 * 64)>(%arg4, %arg5, %arg7)
              %13 = loom.alloc [32, 32, 128] on @L1 : memref<32x32x128xf32>
              %14 = loom.alloc [32, 32, 32] on @L1 : memref<32x32x32xf32>
              %15 = loom.init_tensor %14[32, 32, 32] : memref<32x32x32xf32> -> tensor<32x32x32xf32>
              %16 = loom.alloc [32, 128, 32] on @L1 : memref<32x128x32xf32>
              %17 = loom.alloc [32, 32] on @L1 : memref<32x32xf32>
              %18 = loom.init_tensor %17[32, 32] : memref<32x32xf32> -> tensor<32x32xf32>
              %19 = loom.alloc [32, 32] on @L1 : memref<32x32xf32>
              %20 = loom.init_tensor %19[32, 32] : memref<32x32xf32> -> tensor<32x32xf32>
              %21 = loom.alloc [32, 32] on @L1 : memref<32x32xf32>
              %22 = loom.init_tensor %21[32, 32] : memref<32x32xf32> -> tensor<32x32xf32>
              %23 = loom.alloc [32, 32] on @L1 : memref<32x32xf32>
              %24 = loom.init_tensor %23[32, 32] : memref<32x32xf32> -> tensor<32x32xf32>
              %25 = loom.alloc [32, 32] on @L1 : memref<32x32xf32>
              %26 = loom.init_tensor %25[32, 32] : memref<32x32xf32> -> tensor<32x32xf32>
              %27 = loom.alloc [32, 32, 128] on @L1 : memref<32x32x128xf32>
              %28 = loom.init_tensor %27[32, 32, 128] : memref<32x32x128xf32> -> tensor<32x32x128xf32>
              %29 = loom.alloc [32, 32, 128] on @L1 : memref<32x32x128xf32>
              %30 = loom.init_tensor %29[32, 32, 128] : memref<32x32x128xf32> -> tensor<32x32x128xf32>
              %31 = loom.alloc [32, 32, 128] on @L1 : memref<32x32x128xf32>
              %32 = loom.init_tensor %31[32, 32, 128] : memref<32x32x128xf32> -> tensor<32x32x128xf32>
              %33 = arith.muli %arg6, %c32 : index
              %34 = arith.muli %12, %c32 : index
              %c0 = arith.constant 0 : index
              %35 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 524288 + d1 * 128 + d2)>(%33, %34, %c0)
              %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%35], sizes: [32, 32, 128], strides: [524288, 128, 1] : memref<32x4096x128xf32> to memref<32x32x128xf32, strided<[524288, 128, 1], offset: ?>>
              %36 = loom.copy_to_tensor %reinterpret_cast, %27 on @L1, interconnect : [], broadcast : [1, 1] : memref<32x32x128xf32, strided<[524288, 128, 1], offset: ?>>, memref<32x32x128xf32> -> tensor<32x32x128xf32>
              %37 = linalg.fill ins(%cst_1 : f32) outs(%30 : tensor<32x32x128xf32>) -> tensor<32x32x128xf32>
              %38 = linalg.fill ins(%cst_2 : f32) outs(%18 : tensor<32x32xf32>) -> tensor<32x32xf32>
              %39 = linalg.fill ins(%cst_3 : f32) outs(%20 : tensor<32x32xf32>) -> tensor<32x32xf32>
              %40:3 = affine.for %arg8 = 0 to 128 iter_args(%arg9 = %39, %arg10 = %38, %arg11 = %37) -> (tensor<32x32xf32>, tensor<32x32xf32>, tensor<32x32x128xf32>) {
                %43 = arith.muli %arg8, %c32 : index
                %c0_6 = arith.constant 0 : index
                %44 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 524288 + d1 * 4096 + d2)>(%33, %c0_6, %43)
                %reinterpret_cast_7 = memref.reinterpret_cast %arg0 to offset: [%44], sizes: [32, 128, 32], strides: [524288, 4096, 1] : memref<32x128x4096xf32> to memref<32x128x32xf32, strided<[524288, 4096, 1], offset: ?>>
                %45 = loom.copy_to_tensor %reinterpret_cast_7, %16 on @L1, interconnect : [@horizontal_links], broadcast : [1, 8] : memref<32x128x32xf32, strided<[524288, 4096, 1], offset: ?>>, memref<32x128x32xf32> -> tensor<32x128x32xf32>
                %46 = linalg.fill ins(%cst_1 : f32) outs(%15 : tensor<32x32x32xf32>) -> tensor<32x32x32xf32>
                %47 = linalg.batch_matmul ins(%36, %45 : tensor<32x32x128xf32>, tensor<32x128x32xf32>) outs(%46 : tensor<32x32x32xf32>) -> tensor<32x32x32xf32>
                %48 = linalg.fill ins(%cst_3 : f32) outs(%22 : tensor<32x32xf32>) -> tensor<32x32xf32>
                %49 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%47 : tensor<32x32x32xf32>) outs(%48 : tensor<32x32xf32>) {
                ^bb0(%in: f32, %out: f32):
                  %62 = arith.maximumf %in, %out : f32
                  linalg.yield %62 : f32
                } -> tensor<32x32xf32>
                %50 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%arg9, %49 : tensor<32x32xf32>, tensor<32x32xf32>) outs(%22 : tensor<32x32xf32>) {
                ^bb0(%in: f32, %in_10: f32, %out: f32):
                  %62 = arith.truncf %cst_0 : f64 to f32
                  %63 = arith.mulf %in_10, %62 : f32
                  %64 = arith.cmpf ogt, %in, %63 : f32
                  %65 = arith.select %64, %in, %63 : f32
                  linalg.yield %65 : f32
                } -> tensor<32x32xf32>
                %51 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%47, %50 : tensor<32x32x32xf32>, tensor<32x32xf32>) outs(%15 : tensor<32x32x32xf32>) {
                ^bb0(%in: f32, %in_10: f32, %out: f32):
                  %62 = arith.truncf %cst_0 : f64 to f32
                  %63 = arith.mulf %in, %62 : f32
                  %64 = arith.subf %63, %in_10 : f32
                  %65 = math.powf %cst, %64 : f32
                  linalg.yield %65 : f32
                } -> tensor<32x32x32xf32>
                %52 = linalg.fill ins(%cst_1 : f32) outs(%24 : tensor<32x32xf32>) -> tensor<32x32xf32>
                %53 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%51 : tensor<32x32x32xf32>) outs(%52 : tensor<32x32xf32>) {
                ^bb0(%in: f32, %out: f32):
                  %62 = arith.addf %in, %out : f32
                  linalg.yield %62 : f32
                } -> tensor<32x32xf32>
                %54 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%arg9, %50 : tensor<32x32xf32>, tensor<32x32xf32>) outs(%26 : tensor<32x32xf32>) {
                ^bb0(%in: f32, %in_10: f32, %out: f32):
                  %62 = arith.subf %in, %in_10 : f32
                  %63 = math.powf %cst, %62 : f32
                  linalg.yield %63 : f32
                } -> tensor<32x32xf32>
                %55 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%arg10, %54, %53 : tensor<32x32xf32>, tensor<32x32xf32>, tensor<32x32xf32>) outs(%arg10 : tensor<32x32xf32>) {
                ^bb0(%in: f32, %in_10: f32, %in_11: f32, %out: f32):
                  %62 = arith.mulf %in, %in_10 : f32
                  %63 = arith.addf %62, %in_11 : f32
                  linalg.yield %63 : f32
                } -> tensor<32x32xf32>
                %c0_8 = arith.constant 0 : index
                %56 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 524288 + d1 * 128 + d2)>(%33, %43, %c0_8)
                %reinterpret_cast_9 = memref.reinterpret_cast %arg1 to offset: [%56], sizes: [32, 32, 128], strides: [524288, 128, 1] : memref<32x4096x128xf32> to memref<32x32x128xf32, strided<[524288, 128, 1], offset: ?>>
                %57 = loom.copy_to_tensor %reinterpret_cast_9, %13 on @L1, interconnect : [@vertical_links], broadcast : [8, 1] : memref<32x32x128xf32, strided<[524288, 128, 1], offset: ?>>, memref<32x32x128xf32> -> tensor<32x32x128xf32>
                %58 = linalg.fill ins(%cst_1 : f32) outs(%32 : tensor<32x32x128xf32>) -> tensor<32x32x128xf32>
                %59 = linalg.batch_matmul ins(%51, %57 : tensor<32x32x32xf32>, tensor<32x32x128xf32>) outs(%58 : tensor<32x32x128xf32>) -> tensor<32x32x128xf32>
                %60 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%59, %arg11, %54 : tensor<32x32x128xf32>, tensor<32x32x128xf32>, tensor<32x32xf32>) outs(%arg11 : tensor<32x32x128xf32>) {
                ^bb0(%in: f32, %in_10: f32, %in_11: f32, %out: f32):
                  %62 = arith.mulf %in_10, %in_11 : f32
                  %63 = arith.addf %in, %62 : f32
                  linalg.yield %63 : f32
                } -> tensor<32x32x128xf32>
                %61 = linalg.copy ins(%50 : tensor<32x32xf32>) outs(%arg9 : tensor<32x32xf32>) -> tensor<32x32xf32>
                affine.yield %61, %55, %60 : tensor<32x32xf32>, tensor<32x32xf32>, tensor<32x32x128xf32>
              }
              %41 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%40#2, %40#1 : tensor<32x32x128xf32>, tensor<32x32xf32>) outs(%28 : tensor<32x32x128xf32>) {
              ^bb0(%in: f32, %in_6: f32, %out: f32):
                %43 = arith.divf %in, %in_6 : f32
                linalg.yield %43 : f32
              } -> tensor<32x32x128xf32>
              %c0_4 = arith.constant 0 : index
              %42 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 524288 + d1 * 128 + d2)>(%33, %34, %c0_4)
              %reinterpret_cast_5 = memref.reinterpret_cast %arg3 to offset: [%42], sizes: [32, 32, 128], strides: [524288, 128, 1] : memref<32x4096x128xf32> to memref<32x32x128xf32, strided<[524288, 128, 1], offset: ?>>
              loom.copy_from_tensor %41, %reinterpret_cast_5 on @DRAM : tensor<32x32x128xf32>, memref<32x32x128xf32, strided<[524288, 128, 1], offset: ?>>
            } {loom.iter_type = #loom.iter_type<temporal>}
          } {loom.iter_type = #loom.iter_type<temporal>}
        } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @y}
      } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @x}
      return
    }
  }
  module attributes {loom.pass_name = "Materialize"} {
    func.func @attention__d0i1_d1i1__f01__d_v_d__BB32__BM32__BN32(%arg0: memref<32x128x4096xf32>, %arg1: memref<32x4096x128xf32>, %arg2: memref<32x4096x128xf32>, %arg3: memref<32x4096x128xf32>) {
      %cst = arith.constant 2.000000e+00 : f32
      %cst_0 = arith.constant 0.12751743074602467 : f64
      %cst_1 = arith.constant 0.000000e+00 : f32
      %cst_2 = arith.constant 1.000000e+00 : f32
      %cst_3 = arith.constant 0xFF800000 : f32
      %c32 = arith.constant 32 : index
      affine.parallel (%arg4) = (0) to (8) {
        affine.parallel (%arg5) = (0) to (8) {
          affine.for %arg6 = 0 to 1 {
            affine.for %arg7 = 0 to 2 {
              %12 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 8 + d1 + d2 * 64)>(%arg4, %arg5, %arg7)
              %13 = loom.alloc [32, 32, 128] on @L1 : memref<32x32x128xf32>
              %14 = loom.alloc [32, 32, 32] on @L1 : memref<32x32x32xf32>
              %15 = loom.init_tensor %14[32, 32, 32] : memref<32x32x32xf32> -> tensor<32x32x32xf32>
              %16 = loom.alloc [32, 128, 32] on @L1 : memref<32x128x32xf32>
              %17 = loom.alloc [32, 32] on @L1 : memref<32x32xf32>
              %18 = loom.init_tensor %17[32, 32] : memref<32x32xf32> -> tensor<32x32xf32>
              %19 = loom.alloc [32, 32] on @L1 : memref<32x32xf32>
              %20 = loom.init_tensor %19[32, 32] : memref<32x32xf32> -> tensor<32x32xf32>
              %21 = loom.alloc [32, 32] on @L1 : memref<32x32xf32>
              %22 = loom.init_tensor %21[32, 32] : memref<32x32xf32> -> tensor<32x32xf32>
              %23 = loom.alloc [32, 32] on @L1 : memref<32x32xf32>
              %24 = loom.init_tensor %23[32, 32] : memref<32x32xf32> -> tensor<32x32xf32>
              %25 = loom.alloc [32, 32] on @L1 : memref<32x32xf32>
              %26 = loom.init_tensor %25[32, 32] : memref<32x32xf32> -> tensor<32x32xf32>
              %27 = loom.alloc [32, 32, 128] on @L1 : memref<32x32x128xf32>
              %28 = loom.init_tensor %27[32, 32, 128] : memref<32x32x128xf32> -> tensor<32x32x128xf32>
              %29 = loom.alloc [32, 32, 128] on @L1 : memref<32x32x128xf32>
              %30 = loom.init_tensor %29[32, 32, 128] : memref<32x32x128xf32> -> tensor<32x32x128xf32>
              %31 = loom.alloc [32, 32, 128] on @L1 : memref<32x32x128xf32>
              %32 = loom.init_tensor %31[32, 32, 128] : memref<32x32x128xf32> -> tensor<32x32x128xf32>
              %33 = arith.muli %arg6, %c32 : index
              %34 = arith.muli %12, %c32 : index
              %c0 = arith.constant 0 : index
              %35 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 524288 + d1 * 128 + d2)>(%33, %34, %c0)
              %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%35], sizes: [32, 32, 128], strides: [524288, 128, 1] : memref<32x4096x128xf32> to memref<32x32x128xf32, strided<[524288, 128, 1], offset: ?>>
              %36 = loom.copy_to_tensor %reinterpret_cast, %27 on @L1, interconnect : [], broadcast : [1, 1] : memref<32x32x128xf32, strided<[524288, 128, 1], offset: ?>>, memref<32x32x128xf32> -> tensor<32x32x128xf32>
              %37 = linalg.fill ins(%cst_1 : f32) outs(%30 : tensor<32x32x128xf32>) -> tensor<32x32x128xf32>
              %38 = linalg.fill ins(%cst_2 : f32) outs(%18 : tensor<32x32xf32>) -> tensor<32x32xf32>
              %39 = linalg.fill ins(%cst_3 : f32) outs(%20 : tensor<32x32xf32>) -> tensor<32x32xf32>
              %40:3 = affine.for %arg8 = 0 to 128 iter_args(%arg9 = %39, %arg10 = %38, %arg11 = %37) -> (tensor<32x32xf32>, tensor<32x32xf32>, tensor<32x32x128xf32>) {
                %43 = arith.muli %arg8, %c32 : index
                %c0_6 = arith.constant 0 : index
                %44 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 524288 + d1 * 4096 + d2)>(%33, %c0_6, %43)
                %reinterpret_cast_7 = memref.reinterpret_cast %arg0 to offset: [%44], sizes: [32, 128, 32], strides: [524288, 4096, 1] : memref<32x128x4096xf32> to memref<32x128x32xf32, strided<[524288, 4096, 1], offset: ?>>
                %45 = loom.copy_to_tensor %reinterpret_cast_7, %16 on @L1, interconnect : [@vertical_links], broadcast : [8, 1] : memref<32x128x32xf32, strided<[524288, 4096, 1], offset: ?>>, memref<32x128x32xf32> -> tensor<32x128x32xf32>
                %46 = linalg.fill ins(%cst_1 : f32) outs(%15 : tensor<32x32x32xf32>) -> tensor<32x32x32xf32>
                %47 = linalg.batch_matmul ins(%36, %45 : tensor<32x32x128xf32>, tensor<32x128x32xf32>) outs(%46 : tensor<32x32x32xf32>) -> tensor<32x32x32xf32>
                %48 = linalg.fill ins(%cst_3 : f32) outs(%22 : tensor<32x32xf32>) -> tensor<32x32xf32>
                %49 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%47 : tensor<32x32x32xf32>) outs(%48 : tensor<32x32xf32>) {
                ^bb0(%in: f32, %out: f32):
                  %62 = arith.maximumf %in, %out : f32
                  linalg.yield %62 : f32
                } -> tensor<32x32xf32>
                %50 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%arg9, %49 : tensor<32x32xf32>, tensor<32x32xf32>) outs(%22 : tensor<32x32xf32>) {
                ^bb0(%in: f32, %in_10: f32, %out: f32):
                  %62 = arith.truncf %cst_0 : f64 to f32
                  %63 = arith.mulf %in_10, %62 : f32
                  %64 = arith.cmpf ogt, %in, %63 : f32
                  %65 = arith.select %64, %in, %63 : f32
                  linalg.yield %65 : f32
                } -> tensor<32x32xf32>
                %51 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%47, %50 : tensor<32x32x32xf32>, tensor<32x32xf32>) outs(%15 : tensor<32x32x32xf32>) {
                ^bb0(%in: f32, %in_10: f32, %out: f32):
                  %62 = arith.truncf %cst_0 : f64 to f32
                  %63 = arith.mulf %in, %62 : f32
                  %64 = arith.subf %63, %in_10 : f32
                  %65 = math.powf %cst, %64 : f32
                  linalg.yield %65 : f32
                } -> tensor<32x32x32xf32>
                %52 = linalg.fill ins(%cst_1 : f32) outs(%24 : tensor<32x32xf32>) -> tensor<32x32xf32>
                %53 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%51 : tensor<32x32x32xf32>) outs(%52 : tensor<32x32xf32>) {
                ^bb0(%in: f32, %out: f32):
                  %62 = arith.addf %in, %out : f32
                  linalg.yield %62 : f32
                } -> tensor<32x32xf32>
                %54 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%arg9, %50 : tensor<32x32xf32>, tensor<32x32xf32>) outs(%26 : tensor<32x32xf32>) {
                ^bb0(%in: f32, %in_10: f32, %out: f32):
                  %62 = arith.subf %in, %in_10 : f32
                  %63 = math.powf %cst, %62 : f32
                  linalg.yield %63 : f32
                } -> tensor<32x32xf32>
                %55 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%arg10, %54, %53 : tensor<32x32xf32>, tensor<32x32xf32>, tensor<32x32xf32>) outs(%arg10 : tensor<32x32xf32>) {
                ^bb0(%in: f32, %in_10: f32, %in_11: f32, %out: f32):
                  %62 = arith.mulf %in, %in_10 : f32
                  %63 = arith.addf %62, %in_11 : f32
                  linalg.yield %63 : f32
                } -> tensor<32x32xf32>
                %c0_8 = arith.constant 0 : index
                %56 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 524288 + d1 * 128 + d2)>(%33, %43, %c0_8)
                %reinterpret_cast_9 = memref.reinterpret_cast %arg1 to offset: [%56], sizes: [32, 32, 128], strides: [524288, 128, 1] : memref<32x4096x128xf32> to memref<32x32x128xf32, strided<[524288, 128, 1], offset: ?>>
                %57 = loom.copy_to_tensor %reinterpret_cast_9, %13 on @L1, interconnect : [], broadcast : [1, 1] : memref<32x32x128xf32, strided<[524288, 128, 1], offset: ?>>, memref<32x32x128xf32> -> tensor<32x32x128xf32>
                %58 = linalg.fill ins(%cst_1 : f32) outs(%32 : tensor<32x32x128xf32>) -> tensor<32x32x128xf32>
                %59 = linalg.batch_matmul ins(%51, %57 : tensor<32x32x32xf32>, tensor<32x32x128xf32>) outs(%58 : tensor<32x32x128xf32>) -> tensor<32x32x128xf32>
                %60 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%59, %arg11, %54 : tensor<32x32x128xf32>, tensor<32x32x128xf32>, tensor<32x32xf32>) outs(%arg11 : tensor<32x32x128xf32>) {
                ^bb0(%in: f32, %in_10: f32, %in_11: f32, %out: f32):
                  %62 = arith.mulf %in_10, %in_11 : f32
                  %63 = arith.addf %in, %62 : f32
                  linalg.yield %63 : f32
                } -> tensor<32x32x128xf32>
                %61 = linalg.copy ins(%50 : tensor<32x32xf32>) outs(%arg9 : tensor<32x32xf32>) -> tensor<32x32xf32>
                affine.yield %61, %55, %60 : tensor<32x32xf32>, tensor<32x32xf32>, tensor<32x32x128xf32>
              }
              %41 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%40#2, %40#1 : tensor<32x32x128xf32>, tensor<32x32xf32>) outs(%28 : tensor<32x32x128xf32>) {
              ^bb0(%in: f32, %in_6: f32, %out: f32):
                %43 = arith.divf %in, %in_6 : f32
                linalg.yield %43 : f32
              } -> tensor<32x32x128xf32>
              %c0_4 = arith.constant 0 : index
              %42 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 524288 + d1 * 128 + d2)>(%33, %34, %c0_4)
              %reinterpret_cast_5 = memref.reinterpret_cast %arg3 to offset: [%42], sizes: [32, 32, 128], strides: [524288, 128, 1] : memref<32x4096x128xf32> to memref<32x32x128xf32, strided<[524288, 128, 1], offset: ?>>
              loom.copy_from_tensor %41, %reinterpret_cast_5 on @DRAM : tensor<32x32x128xf32>, memref<32x32x128xf32, strided<[524288, 128, 1], offset: ?>>
            } {loom.iter_type = #loom.iter_type<temporal>}
          } {loom.iter_type = #loom.iter_type<temporal>}
        } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @y}
      } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @x}
      return
    }
  }
  module attributes {loom.pass_name = "Materialize"} {
    func.func @attention__d0i1_d1i1__f01__d_v_a__BB32__BM32__BN32(%arg0: memref<32x128x4096xf32>, %arg1: memref<32x4096x128xf32>, %arg2: memref<32x4096x128xf32>, %arg3: memref<32x4096x128xf32>) {
      %cst = arith.constant 2.000000e+00 : f32
      %cst_0 = arith.constant 0.12751743074602467 : f64
      %cst_1 = arith.constant 0.000000e+00 : f32
      %cst_2 = arith.constant 1.000000e+00 : f32
      %cst_3 = arith.constant 0xFF800000 : f32
      %c32 = arith.constant 32 : index
      affine.parallel (%arg4) = (0) to (8) {
        affine.parallel (%arg5) = (0) to (8) {
          affine.for %arg6 = 0 to 1 {
            affine.for %arg7 = 0 to 2 {
              %12 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 8 + d1 + d2 * 64)>(%arg4, %arg5, %arg7)
              %13 = loom.alloc [32, 32, 128] on @L1 : memref<32x32x128xf32>
              %14 = loom.alloc [32, 32, 32] on @L1 : memref<32x32x32xf32>
              %15 = loom.init_tensor %14[32, 32, 32] : memref<32x32x32xf32> -> tensor<32x32x32xf32>
              %16 = loom.alloc [32, 128, 32] on @L1 : memref<32x128x32xf32>
              %17 = loom.alloc [32, 32] on @L1 : memref<32x32xf32>
              %18 = loom.init_tensor %17[32, 32] : memref<32x32xf32> -> tensor<32x32xf32>
              %19 = loom.alloc [32, 32] on @L1 : memref<32x32xf32>
              %20 = loom.init_tensor %19[32, 32] : memref<32x32xf32> -> tensor<32x32xf32>
              %21 = loom.alloc [32, 32] on @L1 : memref<32x32xf32>
              %22 = loom.init_tensor %21[32, 32] : memref<32x32xf32> -> tensor<32x32xf32>
              %23 = loom.alloc [32, 32] on @L1 : memref<32x32xf32>
              %24 = loom.init_tensor %23[32, 32] : memref<32x32xf32> -> tensor<32x32xf32>
              %25 = loom.alloc [32, 32] on @L1 : memref<32x32xf32>
              %26 = loom.init_tensor %25[32, 32] : memref<32x32xf32> -> tensor<32x32xf32>
              %27 = loom.alloc [32, 32, 128] on @L1 : memref<32x32x128xf32>
              %28 = loom.init_tensor %27[32, 32, 128] : memref<32x32x128xf32> -> tensor<32x32x128xf32>
              %29 = loom.alloc [32, 32, 128] on @L1 : memref<32x32x128xf32>
              %30 = loom.init_tensor %29[32, 32, 128] : memref<32x32x128xf32> -> tensor<32x32x128xf32>
              %31 = loom.alloc [32, 32, 128] on @L1 : memref<32x32x128xf32>
              %32 = loom.init_tensor %31[32, 32, 128] : memref<32x32x128xf32> -> tensor<32x32x128xf32>
              %33 = arith.muli %arg6, %c32 : index
              %34 = arith.muli %12, %c32 : index
              %c0 = arith.constant 0 : index
              %35 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 524288 + d1 * 128 + d2)>(%33, %34, %c0)
              %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%35], sizes: [32, 32, 128], strides: [524288, 128, 1] : memref<32x4096x128xf32> to memref<32x32x128xf32, strided<[524288, 128, 1], offset: ?>>
              %36 = loom.copy_to_tensor %reinterpret_cast, %27 on @L1, interconnect : [], broadcast : [1, 1] : memref<32x32x128xf32, strided<[524288, 128, 1], offset: ?>>, memref<32x32x128xf32> -> tensor<32x32x128xf32>
              %37 = linalg.fill ins(%cst_1 : f32) outs(%30 : tensor<32x32x128xf32>) -> tensor<32x32x128xf32>
              %38 = linalg.fill ins(%cst_2 : f32) outs(%18 : tensor<32x32xf32>) -> tensor<32x32xf32>
              %39 = linalg.fill ins(%cst_3 : f32) outs(%20 : tensor<32x32xf32>) -> tensor<32x32xf32>
              %40:3 = affine.for %arg8 = 0 to 128 iter_args(%arg9 = %39, %arg10 = %38, %arg11 = %37) -> (tensor<32x32xf32>, tensor<32x32xf32>, tensor<32x32x128xf32>) {
                %43 = arith.muli %arg8, %c32 : index
                %c0_6 = arith.constant 0 : index
                %44 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 524288 + d1 * 4096 + d2)>(%33, %c0_6, %43)
                %reinterpret_cast_7 = memref.reinterpret_cast %arg0 to offset: [%44], sizes: [32, 128, 32], strides: [524288, 4096, 1] : memref<32x128x4096xf32> to memref<32x128x32xf32, strided<[524288, 4096, 1], offset: ?>>
                %45 = loom.copy_to_tensor %reinterpret_cast_7, %16 on @L1, interconnect : [@vertical_links], broadcast : [8, 1] : memref<32x128x32xf32, strided<[524288, 4096, 1], offset: ?>>, memref<32x128x32xf32> -> tensor<32x128x32xf32>
                %46 = linalg.fill ins(%cst_1 : f32) outs(%15 : tensor<32x32x32xf32>) -> tensor<32x32x32xf32>
                %47 = linalg.batch_matmul ins(%36, %45 : tensor<32x32x128xf32>, tensor<32x128x32xf32>) outs(%46 : tensor<32x32x32xf32>) -> tensor<32x32x32xf32>
                %48 = linalg.fill ins(%cst_3 : f32) outs(%22 : tensor<32x32xf32>) -> tensor<32x32xf32>
                %49 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%47 : tensor<32x32x32xf32>) outs(%48 : tensor<32x32xf32>) {
                ^bb0(%in: f32, %out: f32):
                  %62 = arith.maximumf %in, %out : f32
                  linalg.yield %62 : f32
                } -> tensor<32x32xf32>
                %50 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%arg9, %49 : tensor<32x32xf32>, tensor<32x32xf32>) outs(%22 : tensor<32x32xf32>) {
                ^bb0(%in: f32, %in_10: f32, %out: f32):
                  %62 = arith.truncf %cst_0 : f64 to f32
                  %63 = arith.mulf %in_10, %62 : f32
                  %64 = arith.cmpf ogt, %in, %63 : f32
                  %65 = arith.select %64, %in, %63 : f32
                  linalg.yield %65 : f32
                } -> tensor<32x32xf32>
                %51 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%47, %50 : tensor<32x32x32xf32>, tensor<32x32xf32>) outs(%15 : tensor<32x32x32xf32>) {
                ^bb0(%in: f32, %in_10: f32, %out: f32):
                  %62 = arith.truncf %cst_0 : f64 to f32
                  %63 = arith.mulf %in, %62 : f32
                  %64 = arith.subf %63, %in_10 : f32
                  %65 = math.powf %cst, %64 : f32
                  linalg.yield %65 : f32
                } -> tensor<32x32x32xf32>
                %52 = linalg.fill ins(%cst_1 : f32) outs(%24 : tensor<32x32xf32>) -> tensor<32x32xf32>
                %53 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%51 : tensor<32x32x32xf32>) outs(%52 : tensor<32x32xf32>) {
                ^bb0(%in: f32, %out: f32):
                  %62 = arith.addf %in, %out : f32
                  linalg.yield %62 : f32
                } -> tensor<32x32xf32>
                %54 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%arg9, %50 : tensor<32x32xf32>, tensor<32x32xf32>) outs(%26 : tensor<32x32xf32>) {
                ^bb0(%in: f32, %in_10: f32, %out: f32):
                  %62 = arith.subf %in, %in_10 : f32
                  %63 = math.powf %cst, %62 : f32
                  linalg.yield %63 : f32
                } -> tensor<32x32xf32>
                %55 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%arg10, %54, %53 : tensor<32x32xf32>, tensor<32x32xf32>, tensor<32x32xf32>) outs(%arg10 : tensor<32x32xf32>) {
                ^bb0(%in: f32, %in_10: f32, %in_11: f32, %out: f32):
                  %62 = arith.mulf %in, %in_10 : f32
                  %63 = arith.addf %62, %in_11 : f32
                  linalg.yield %63 : f32
                } -> tensor<32x32xf32>
                %c0_8 = arith.constant 0 : index
                %56 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 524288 + d1 * 128 + d2)>(%33, %43, %c0_8)
                %reinterpret_cast_9 = memref.reinterpret_cast %arg1 to offset: [%56], sizes: [32, 32, 128], strides: [524288, 128, 1] : memref<32x4096x128xf32> to memref<32x32x128xf32, strided<[524288, 128, 1], offset: ?>>
                %57 = loom.copy_to_tensor %reinterpret_cast_9, %13 on @L1, interconnect : [@horizontal_links, @vertical_links], broadcast : [8, 8] : memref<32x32x128xf32, strided<[524288, 128, 1], offset: ?>>, memref<32x32x128xf32> -> tensor<32x32x128xf32>
                %58 = linalg.fill ins(%cst_1 : f32) outs(%32 : tensor<32x32x128xf32>) -> tensor<32x32x128xf32>
                %59 = linalg.batch_matmul ins(%51, %57 : tensor<32x32x32xf32>, tensor<32x32x128xf32>) outs(%58 : tensor<32x32x128xf32>) -> tensor<32x32x128xf32>
                %60 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%59, %arg11, %54 : tensor<32x32x128xf32>, tensor<32x32x128xf32>, tensor<32x32xf32>) outs(%arg11 : tensor<32x32x128xf32>) {
                ^bb0(%in: f32, %in_10: f32, %in_11: f32, %out: f32):
                  %62 = arith.mulf %in_10, %in_11 : f32
                  %63 = arith.addf %in, %62 : f32
                  linalg.yield %63 : f32
                } -> tensor<32x32x128xf32>
                %61 = linalg.copy ins(%50 : tensor<32x32xf32>) outs(%arg9 : tensor<32x32xf32>) -> tensor<32x32xf32>
                affine.yield %61, %55, %60 : tensor<32x32xf32>, tensor<32x32xf32>, tensor<32x32x128xf32>
              }
              %41 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%40#2, %40#1 : tensor<32x32x128xf32>, tensor<32x32xf32>) outs(%28 : tensor<32x32x128xf32>) {
              ^bb0(%in: f32, %in_6: f32, %out: f32):
                %43 = arith.divf %in, %in_6 : f32
                linalg.yield %43 : f32
              } -> tensor<32x32x128xf32>
              %c0_4 = arith.constant 0 : index
              %42 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 524288 + d1 * 128 + d2)>(%33, %34, %c0_4)
              %reinterpret_cast_5 = memref.reinterpret_cast %arg3 to offset: [%42], sizes: [32, 32, 128], strides: [524288, 128, 1] : memref<32x4096x128xf32> to memref<32x32x128xf32, strided<[524288, 128, 1], offset: ?>>
              loom.copy_from_tensor %41, %reinterpret_cast_5 on @DRAM : tensor<32x32x128xf32>, memref<32x32x128xf32, strided<[524288, 128, 1], offset: ?>>
            } {loom.iter_type = #loom.iter_type<temporal>}
          } {loom.iter_type = #loom.iter_type<temporal>}
        } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @y}
      } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @x}
      return
    }
  }
  module attributes {loom.pass_name = "Materialize"} {
    func.func @attention__d0i1_d1i1__f01__d_v_h__BB32__BM32__BN32(%arg0: memref<32x128x4096xf32>, %arg1: memref<32x4096x128xf32>, %arg2: memref<32x4096x128xf32>, %arg3: memref<32x4096x128xf32>) {
      %cst = arith.constant 2.000000e+00 : f32
      %cst_0 = arith.constant 0.12751743074602467 : f64
      %cst_1 = arith.constant 0.000000e+00 : f32
      %cst_2 = arith.constant 1.000000e+00 : f32
      %cst_3 = arith.constant 0xFF800000 : f32
      %c32 = arith.constant 32 : index
      affine.parallel (%arg4) = (0) to (8) {
        affine.parallel (%arg5) = (0) to (8) {
          affine.for %arg6 = 0 to 1 {
            affine.for %arg7 = 0 to 2 {
              %12 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 8 + d1 + d2 * 64)>(%arg4, %arg5, %arg7)
              %13 = loom.alloc [32, 32, 128] on @L1 : memref<32x32x128xf32>
              %14 = loom.alloc [32, 32, 32] on @L1 : memref<32x32x32xf32>
              %15 = loom.init_tensor %14[32, 32, 32] : memref<32x32x32xf32> -> tensor<32x32x32xf32>
              %16 = loom.alloc [32, 128, 32] on @L1 : memref<32x128x32xf32>
              %17 = loom.alloc [32, 32] on @L1 : memref<32x32xf32>
              %18 = loom.init_tensor %17[32, 32] : memref<32x32xf32> -> tensor<32x32xf32>
              %19 = loom.alloc [32, 32] on @L1 : memref<32x32xf32>
              %20 = loom.init_tensor %19[32, 32] : memref<32x32xf32> -> tensor<32x32xf32>
              %21 = loom.alloc [32, 32] on @L1 : memref<32x32xf32>
              %22 = loom.init_tensor %21[32, 32] : memref<32x32xf32> -> tensor<32x32xf32>
              %23 = loom.alloc [32, 32] on @L1 : memref<32x32xf32>
              %24 = loom.init_tensor %23[32, 32] : memref<32x32xf32> -> tensor<32x32xf32>
              %25 = loom.alloc [32, 32] on @L1 : memref<32x32xf32>
              %26 = loom.init_tensor %25[32, 32] : memref<32x32xf32> -> tensor<32x32xf32>
              %27 = loom.alloc [32, 32, 128] on @L1 : memref<32x32x128xf32>
              %28 = loom.init_tensor %27[32, 32, 128] : memref<32x32x128xf32> -> tensor<32x32x128xf32>
              %29 = loom.alloc [32, 32, 128] on @L1 : memref<32x32x128xf32>
              %30 = loom.init_tensor %29[32, 32, 128] : memref<32x32x128xf32> -> tensor<32x32x128xf32>
              %31 = loom.alloc [32, 32, 128] on @L1 : memref<32x32x128xf32>
              %32 = loom.init_tensor %31[32, 32, 128] : memref<32x32x128xf32> -> tensor<32x32x128xf32>
              %33 = arith.muli %arg6, %c32 : index
              %34 = arith.muli %12, %c32 : index
              %c0 = arith.constant 0 : index
              %35 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 524288 + d1 * 128 + d2)>(%33, %34, %c0)
              %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%35], sizes: [32, 32, 128], strides: [524288, 128, 1] : memref<32x4096x128xf32> to memref<32x32x128xf32, strided<[524288, 128, 1], offset: ?>>
              %36 = loom.copy_to_tensor %reinterpret_cast, %27 on @L1, interconnect : [], broadcast : [1, 1] : memref<32x32x128xf32, strided<[524288, 128, 1], offset: ?>>, memref<32x32x128xf32> -> tensor<32x32x128xf32>
              %37 = linalg.fill ins(%cst_1 : f32) outs(%30 : tensor<32x32x128xf32>) -> tensor<32x32x128xf32>
              %38 = linalg.fill ins(%cst_2 : f32) outs(%18 : tensor<32x32xf32>) -> tensor<32x32xf32>
              %39 = linalg.fill ins(%cst_3 : f32) outs(%20 : tensor<32x32xf32>) -> tensor<32x32xf32>
              %40:3 = affine.for %arg8 = 0 to 128 iter_args(%arg9 = %39, %arg10 = %38, %arg11 = %37) -> (tensor<32x32xf32>, tensor<32x32xf32>, tensor<32x32x128xf32>) {
                %43 = arith.muli %arg8, %c32 : index
                %c0_6 = arith.constant 0 : index
                %44 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 524288 + d1 * 4096 + d2)>(%33, %c0_6, %43)
                %reinterpret_cast_7 = memref.reinterpret_cast %arg0 to offset: [%44], sizes: [32, 128, 32], strides: [524288, 4096, 1] : memref<32x128x4096xf32> to memref<32x128x32xf32, strided<[524288, 4096, 1], offset: ?>>
                %45 = loom.copy_to_tensor %reinterpret_cast_7, %16 on @L1, interconnect : [@vertical_links], broadcast : [8, 1] : memref<32x128x32xf32, strided<[524288, 4096, 1], offset: ?>>, memref<32x128x32xf32> -> tensor<32x128x32xf32>
                %46 = linalg.fill ins(%cst_1 : f32) outs(%15 : tensor<32x32x32xf32>) -> tensor<32x32x32xf32>
                %47 = linalg.batch_matmul ins(%36, %45 : tensor<32x32x128xf32>, tensor<32x128x32xf32>) outs(%46 : tensor<32x32x32xf32>) -> tensor<32x32x32xf32>
                %48 = linalg.fill ins(%cst_3 : f32) outs(%22 : tensor<32x32xf32>) -> tensor<32x32xf32>
                %49 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%47 : tensor<32x32x32xf32>) outs(%48 : tensor<32x32xf32>) {
                ^bb0(%in: f32, %out: f32):
                  %62 = arith.maximumf %in, %out : f32
                  linalg.yield %62 : f32
                } -> tensor<32x32xf32>
                %50 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%arg9, %49 : tensor<32x32xf32>, tensor<32x32xf32>) outs(%22 : tensor<32x32xf32>) {
                ^bb0(%in: f32, %in_10: f32, %out: f32):
                  %62 = arith.truncf %cst_0 : f64 to f32
                  %63 = arith.mulf %in_10, %62 : f32
                  %64 = arith.cmpf ogt, %in, %63 : f32
                  %65 = arith.select %64, %in, %63 : f32
                  linalg.yield %65 : f32
                } -> tensor<32x32xf32>
                %51 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%47, %50 : tensor<32x32x32xf32>, tensor<32x32xf32>) outs(%15 : tensor<32x32x32xf32>) {
                ^bb0(%in: f32, %in_10: f32, %out: f32):
                  %62 = arith.truncf %cst_0 : f64 to f32
                  %63 = arith.mulf %in, %62 : f32
                  %64 = arith.subf %63, %in_10 : f32
                  %65 = math.powf %cst, %64 : f32
                  linalg.yield %65 : f32
                } -> tensor<32x32x32xf32>
                %52 = linalg.fill ins(%cst_1 : f32) outs(%24 : tensor<32x32xf32>) -> tensor<32x32xf32>
                %53 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%51 : tensor<32x32x32xf32>) outs(%52 : tensor<32x32xf32>) {
                ^bb0(%in: f32, %out: f32):
                  %62 = arith.addf %in, %out : f32
                  linalg.yield %62 : f32
                } -> tensor<32x32xf32>
                %54 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%arg9, %50 : tensor<32x32xf32>, tensor<32x32xf32>) outs(%26 : tensor<32x32xf32>) {
                ^bb0(%in: f32, %in_10: f32, %out: f32):
                  %62 = arith.subf %in, %in_10 : f32
                  %63 = math.powf %cst, %62 : f32
                  linalg.yield %63 : f32
                } -> tensor<32x32xf32>
                %55 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%arg10, %54, %53 : tensor<32x32xf32>, tensor<32x32xf32>, tensor<32x32xf32>) outs(%arg10 : tensor<32x32xf32>) {
                ^bb0(%in: f32, %in_10: f32, %in_11: f32, %out: f32):
                  %62 = arith.mulf %in, %in_10 : f32
                  %63 = arith.addf %62, %in_11 : f32
                  linalg.yield %63 : f32
                } -> tensor<32x32xf32>
                %c0_8 = arith.constant 0 : index
                %56 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 524288 + d1 * 128 + d2)>(%33, %43, %c0_8)
                %reinterpret_cast_9 = memref.reinterpret_cast %arg1 to offset: [%56], sizes: [32, 32, 128], strides: [524288, 128, 1] : memref<32x4096x128xf32> to memref<32x32x128xf32, strided<[524288, 128, 1], offset: ?>>
                %57 = loom.copy_to_tensor %reinterpret_cast_9, %13 on @L1, interconnect : [@horizontal_links], broadcast : [1, 8] : memref<32x32x128xf32, strided<[524288, 128, 1], offset: ?>>, memref<32x32x128xf32> -> tensor<32x32x128xf32>
                %58 = linalg.fill ins(%cst_1 : f32) outs(%32 : tensor<32x32x128xf32>) -> tensor<32x32x128xf32>
                %59 = linalg.batch_matmul ins(%51, %57 : tensor<32x32x32xf32>, tensor<32x32x128xf32>) outs(%58 : tensor<32x32x128xf32>) -> tensor<32x32x128xf32>
                %60 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%59, %arg11, %54 : tensor<32x32x128xf32>, tensor<32x32x128xf32>, tensor<32x32xf32>) outs(%arg11 : tensor<32x32x128xf32>) {
                ^bb0(%in: f32, %in_10: f32, %in_11: f32, %out: f32):
                  %62 = arith.mulf %in_10, %in_11 : f32
                  %63 = arith.addf %in, %62 : f32
                  linalg.yield %63 : f32
                } -> tensor<32x32x128xf32>
                %61 = linalg.copy ins(%50 : tensor<32x32xf32>) outs(%arg9 : tensor<32x32xf32>) -> tensor<32x32xf32>
                affine.yield %61, %55, %60 : tensor<32x32xf32>, tensor<32x32xf32>, tensor<32x32x128xf32>
              }
              %41 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%40#2, %40#1 : tensor<32x32x128xf32>, tensor<32x32xf32>) outs(%28 : tensor<32x32x128xf32>) {
              ^bb0(%in: f32, %in_6: f32, %out: f32):
                %43 = arith.divf %in, %in_6 : f32
                linalg.yield %43 : f32
              } -> tensor<32x32x128xf32>
              %c0_4 = arith.constant 0 : index
              %42 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 524288 + d1 * 128 + d2)>(%33, %34, %c0_4)
              %reinterpret_cast_5 = memref.reinterpret_cast %arg3 to offset: [%42], sizes: [32, 32, 128], strides: [524288, 128, 1] : memref<32x4096x128xf32> to memref<32x32x128xf32, strided<[524288, 128, 1], offset: ?>>
              loom.copy_from_tensor %41, %reinterpret_cast_5 on @DRAM : tensor<32x32x128xf32>, memref<32x32x128xf32, strided<[524288, 128, 1], offset: ?>>
            } {loom.iter_type = #loom.iter_type<temporal>}
          } {loom.iter_type = #loom.iter_type<temporal>}
        } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @y}
      } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @x}
      return
    }
  }
  module attributes {loom.pass_name = "Materialize"} {
    func.func @attention__d0i1_d1i1__f01__d_v_v__BB32__BM32__BN32(%arg0: memref<32x128x4096xf32>, %arg1: memref<32x4096x128xf32>, %arg2: memref<32x4096x128xf32>, %arg3: memref<32x4096x128xf32>) {
      %cst = arith.constant 2.000000e+00 : f32
      %cst_0 = arith.constant 0.12751743074602467 : f64
      %cst_1 = arith.constant 0.000000e+00 : f32
      %cst_2 = arith.constant 1.000000e+00 : f32
      %cst_3 = arith.constant 0xFF800000 : f32
      %c32 = arith.constant 32 : index
      affine.parallel (%arg4) = (0) to (8) {
        affine.parallel (%arg5) = (0) to (8) {
          affine.for %arg6 = 0 to 1 {
            affine.for %arg7 = 0 to 2 {
              %12 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 8 + d1 + d2 * 64)>(%arg4, %arg5, %arg7)
              %13 = loom.alloc [32, 32, 128] on @L1 : memref<32x32x128xf32>
              %14 = loom.alloc [32, 32, 32] on @L1 : memref<32x32x32xf32>
              %15 = loom.init_tensor %14[32, 32, 32] : memref<32x32x32xf32> -> tensor<32x32x32xf32>
              %16 = loom.alloc [32, 128, 32] on @L1 : memref<32x128x32xf32>
              %17 = loom.alloc [32, 32] on @L1 : memref<32x32xf32>
              %18 = loom.init_tensor %17[32, 32] : memref<32x32xf32> -> tensor<32x32xf32>
              %19 = loom.alloc [32, 32] on @L1 : memref<32x32xf32>
              %20 = loom.init_tensor %19[32, 32] : memref<32x32xf32> -> tensor<32x32xf32>
              %21 = loom.alloc [32, 32] on @L1 : memref<32x32xf32>
              %22 = loom.init_tensor %21[32, 32] : memref<32x32xf32> -> tensor<32x32xf32>
              %23 = loom.alloc [32, 32] on @L1 : memref<32x32xf32>
              %24 = loom.init_tensor %23[32, 32] : memref<32x32xf32> -> tensor<32x32xf32>
              %25 = loom.alloc [32, 32] on @L1 : memref<32x32xf32>
              %26 = loom.init_tensor %25[32, 32] : memref<32x32xf32> -> tensor<32x32xf32>
              %27 = loom.alloc [32, 32, 128] on @L1 : memref<32x32x128xf32>
              %28 = loom.init_tensor %27[32, 32, 128] : memref<32x32x128xf32> -> tensor<32x32x128xf32>
              %29 = loom.alloc [32, 32, 128] on @L1 : memref<32x32x128xf32>
              %30 = loom.init_tensor %29[32, 32, 128] : memref<32x32x128xf32> -> tensor<32x32x128xf32>
              %31 = loom.alloc [32, 32, 128] on @L1 : memref<32x32x128xf32>
              %32 = loom.init_tensor %31[32, 32, 128] : memref<32x32x128xf32> -> tensor<32x32x128xf32>
              %33 = arith.muli %arg6, %c32 : index
              %34 = arith.muli %12, %c32 : index
              %c0 = arith.constant 0 : index
              %35 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 524288 + d1 * 128 + d2)>(%33, %34, %c0)
              %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%35], sizes: [32, 32, 128], strides: [524288, 128, 1] : memref<32x4096x128xf32> to memref<32x32x128xf32, strided<[524288, 128, 1], offset: ?>>
              %36 = loom.copy_to_tensor %reinterpret_cast, %27 on @L1, interconnect : [], broadcast : [1, 1] : memref<32x32x128xf32, strided<[524288, 128, 1], offset: ?>>, memref<32x32x128xf32> -> tensor<32x32x128xf32>
              %37 = linalg.fill ins(%cst_1 : f32) outs(%30 : tensor<32x32x128xf32>) -> tensor<32x32x128xf32>
              %38 = linalg.fill ins(%cst_2 : f32) outs(%18 : tensor<32x32xf32>) -> tensor<32x32xf32>
              %39 = linalg.fill ins(%cst_3 : f32) outs(%20 : tensor<32x32xf32>) -> tensor<32x32xf32>
              %40:3 = affine.for %arg8 = 0 to 128 iter_args(%arg9 = %39, %arg10 = %38, %arg11 = %37) -> (tensor<32x32xf32>, tensor<32x32xf32>, tensor<32x32x128xf32>) {
                %43 = arith.muli %arg8, %c32 : index
                %c0_6 = arith.constant 0 : index
                %44 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 524288 + d1 * 4096 + d2)>(%33, %c0_6, %43)
                %reinterpret_cast_7 = memref.reinterpret_cast %arg0 to offset: [%44], sizes: [32, 128, 32], strides: [524288, 4096, 1] : memref<32x128x4096xf32> to memref<32x128x32xf32, strided<[524288, 4096, 1], offset: ?>>
                %45 = loom.copy_to_tensor %reinterpret_cast_7, %16 on @L1, interconnect : [@vertical_links], broadcast : [8, 1] : memref<32x128x32xf32, strided<[524288, 4096, 1], offset: ?>>, memref<32x128x32xf32> -> tensor<32x128x32xf32>
                %46 = linalg.fill ins(%cst_1 : f32) outs(%15 : tensor<32x32x32xf32>) -> tensor<32x32x32xf32>
                %47 = linalg.batch_matmul ins(%36, %45 : tensor<32x32x128xf32>, tensor<32x128x32xf32>) outs(%46 : tensor<32x32x32xf32>) -> tensor<32x32x32xf32>
                %48 = linalg.fill ins(%cst_3 : f32) outs(%22 : tensor<32x32xf32>) -> tensor<32x32xf32>
                %49 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%47 : tensor<32x32x32xf32>) outs(%48 : tensor<32x32xf32>) {
                ^bb0(%in: f32, %out: f32):
                  %62 = arith.maximumf %in, %out : f32
                  linalg.yield %62 : f32
                } -> tensor<32x32xf32>
                %50 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%arg9, %49 : tensor<32x32xf32>, tensor<32x32xf32>) outs(%22 : tensor<32x32xf32>) {
                ^bb0(%in: f32, %in_10: f32, %out: f32):
                  %62 = arith.truncf %cst_0 : f64 to f32
                  %63 = arith.mulf %in_10, %62 : f32
                  %64 = arith.cmpf ogt, %in, %63 : f32
                  %65 = arith.select %64, %in, %63 : f32
                  linalg.yield %65 : f32
                } -> tensor<32x32xf32>
                %51 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%47, %50 : tensor<32x32x32xf32>, tensor<32x32xf32>) outs(%15 : tensor<32x32x32xf32>) {
                ^bb0(%in: f32, %in_10: f32, %out: f32):
                  %62 = arith.truncf %cst_0 : f64 to f32
                  %63 = arith.mulf %in, %62 : f32
                  %64 = arith.subf %63, %in_10 : f32
                  %65 = math.powf %cst, %64 : f32
                  linalg.yield %65 : f32
                } -> tensor<32x32x32xf32>
                %52 = linalg.fill ins(%cst_1 : f32) outs(%24 : tensor<32x32xf32>) -> tensor<32x32xf32>
                %53 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%51 : tensor<32x32x32xf32>) outs(%52 : tensor<32x32xf32>) {
                ^bb0(%in: f32, %out: f32):
                  %62 = arith.addf %in, %out : f32
                  linalg.yield %62 : f32
                } -> tensor<32x32xf32>
                %54 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%arg9, %50 : tensor<32x32xf32>, tensor<32x32xf32>) outs(%26 : tensor<32x32xf32>) {
                ^bb0(%in: f32, %in_10: f32, %out: f32):
                  %62 = arith.subf %in, %in_10 : f32
                  %63 = math.powf %cst, %62 : f32
                  linalg.yield %63 : f32
                } -> tensor<32x32xf32>
                %55 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%arg10, %54, %53 : tensor<32x32xf32>, tensor<32x32xf32>, tensor<32x32xf32>) outs(%arg10 : tensor<32x32xf32>) {
                ^bb0(%in: f32, %in_10: f32, %in_11: f32, %out: f32):
                  %62 = arith.mulf %in, %in_10 : f32
                  %63 = arith.addf %62, %in_11 : f32
                  linalg.yield %63 : f32
                } -> tensor<32x32xf32>
                %c0_8 = arith.constant 0 : index
                %56 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 524288 + d1 * 128 + d2)>(%33, %43, %c0_8)
                %reinterpret_cast_9 = memref.reinterpret_cast %arg1 to offset: [%56], sizes: [32, 32, 128], strides: [524288, 128, 1] : memref<32x4096x128xf32> to memref<32x32x128xf32, strided<[524288, 128, 1], offset: ?>>
                %57 = loom.copy_to_tensor %reinterpret_cast_9, %13 on @L1, interconnect : [@vertical_links], broadcast : [8, 1] : memref<32x32x128xf32, strided<[524288, 128, 1], offset: ?>>, memref<32x32x128xf32> -> tensor<32x32x128xf32>
                %58 = linalg.fill ins(%cst_1 : f32) outs(%32 : tensor<32x32x128xf32>) -> tensor<32x32x128xf32>
                %59 = linalg.batch_matmul ins(%51, %57 : tensor<32x32x32xf32>, tensor<32x32x128xf32>) outs(%58 : tensor<32x32x128xf32>) -> tensor<32x32x128xf32>
                %60 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%59, %arg11, %54 : tensor<32x32x128xf32>, tensor<32x32x128xf32>, tensor<32x32xf32>) outs(%arg11 : tensor<32x32x128xf32>) {
                ^bb0(%in: f32, %in_10: f32, %in_11: f32, %out: f32):
                  %62 = arith.mulf %in_10, %in_11 : f32
                  %63 = arith.addf %in, %62 : f32
                  linalg.yield %63 : f32
                } -> tensor<32x32x128xf32>
                %61 = linalg.copy ins(%50 : tensor<32x32xf32>) outs(%arg9 : tensor<32x32xf32>) -> tensor<32x32xf32>
                affine.yield %61, %55, %60 : tensor<32x32xf32>, tensor<32x32xf32>, tensor<32x32x128xf32>
              }
              %41 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%40#2, %40#1 : tensor<32x32x128xf32>, tensor<32x32xf32>) outs(%28 : tensor<32x32x128xf32>) {
              ^bb0(%in: f32, %in_6: f32, %out: f32):
                %43 = arith.divf %in, %in_6 : f32
                linalg.yield %43 : f32
              } -> tensor<32x32x128xf32>
              %c0_4 = arith.constant 0 : index
              %42 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 524288 + d1 * 128 + d2)>(%33, %34, %c0_4)
              %reinterpret_cast_5 = memref.reinterpret_cast %arg3 to offset: [%42], sizes: [32, 32, 128], strides: [524288, 128, 1] : memref<32x4096x128xf32> to memref<32x32x128xf32, strided<[524288, 128, 1], offset: ?>>
              loom.copy_from_tensor %41, %reinterpret_cast_5 on @DRAM : tensor<32x32x128xf32>, memref<32x32x128xf32, strided<[524288, 128, 1], offset: ?>>
            } {loom.iter_type = #loom.iter_type<temporal>}
          } {loom.iter_type = #loom.iter_type<temporal>}
        } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @y}
      } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @x}
      return
    }
  }
  module attributes {loom.pass_name = "Materialize"} {
    func.func @attention__d0i1_d1i1__f10__d_d_d__BB32__BM32__BN32(%arg0: memref<32x128x4096xf32>, %arg1: memref<32x4096x128xf32>, %arg2: memref<32x4096x128xf32>, %arg3: memref<32x4096x128xf32>) {
      %cst = arith.constant 2.000000e+00 : f32
      %cst_0 = arith.constant 0.12751743074602467 : f64
      %cst_1 = arith.constant 0.000000e+00 : f32
      %cst_2 = arith.constant 1.000000e+00 : f32
      %cst_3 = arith.constant 0xFF800000 : f32
      %c32 = arith.constant 32 : index
      affine.parallel (%arg4) = (0) to (8) {
        affine.parallel (%arg5) = (0) to (8) {
          affine.for %arg6 = 0 to 2 {
            affine.for %arg7 = 0 to 1 {
              %12 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 8 + d1 + d2 * 64)>(%arg4, %arg5, %arg6)
              %13 = loom.alloc [32, 32, 128] on @L1 : memref<32x32x128xf32>
              %14 = loom.alloc [32, 32, 32] on @L1 : memref<32x32x32xf32>
              %15 = loom.init_tensor %14[32, 32, 32] : memref<32x32x32xf32> -> tensor<32x32x32xf32>
              %16 = loom.alloc [32, 128, 32] on @L1 : memref<32x128x32xf32>
              %17 = loom.alloc [32, 32] on @L1 : memref<32x32xf32>
              %18 = loom.init_tensor %17[32, 32] : memref<32x32xf32> -> tensor<32x32xf32>
              %19 = loom.alloc [32, 32] on @L1 : memref<32x32xf32>
              %20 = loom.init_tensor %19[32, 32] : memref<32x32xf32> -> tensor<32x32xf32>
              %21 = loom.alloc [32, 32] on @L1 : memref<32x32xf32>
              %22 = loom.init_tensor %21[32, 32] : memref<32x32xf32> -> tensor<32x32xf32>
              %23 = loom.alloc [32, 32] on @L1 : memref<32x32xf32>
              %24 = loom.init_tensor %23[32, 32] : memref<32x32xf32> -> tensor<32x32xf32>
              %25 = loom.alloc [32, 32] on @L1 : memref<32x32xf32>
              %26 = loom.init_tensor %25[32, 32] : memref<32x32xf32> -> tensor<32x32xf32>
              %27 = loom.alloc [32, 32, 128] on @L1 : memref<32x32x128xf32>
              %28 = loom.init_tensor %27[32, 32, 128] : memref<32x32x128xf32> -> tensor<32x32x128xf32>
              %29 = loom.alloc [32, 32, 128] on @L1 : memref<32x32x128xf32>
              %30 = loom.init_tensor %29[32, 32, 128] : memref<32x32x128xf32> -> tensor<32x32x128xf32>
              %31 = loom.alloc [32, 32, 128] on @L1 : memref<32x32x128xf32>
              %32 = loom.init_tensor %31[32, 32, 128] : memref<32x32x128xf32> -> tensor<32x32x128xf32>
              %33 = arith.muli %arg7, %c32 : index
              %34 = arith.muli %12, %c32 : index
              %c0 = arith.constant 0 : index
              %35 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 524288 + d1 * 128 + d2)>(%33, %34, %c0)
              %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%35], sizes: [32, 32, 128], strides: [524288, 128, 1] : memref<32x4096x128xf32> to memref<32x32x128xf32, strided<[524288, 128, 1], offset: ?>>
              %36 = loom.copy_to_tensor %reinterpret_cast, %27 on @L1, interconnect : [], broadcast : [1, 1] : memref<32x32x128xf32, strided<[524288, 128, 1], offset: ?>>, memref<32x32x128xf32> -> tensor<32x32x128xf32>
              %37 = linalg.fill ins(%cst_1 : f32) outs(%30 : tensor<32x32x128xf32>) -> tensor<32x32x128xf32>
              %38 = linalg.fill ins(%cst_2 : f32) outs(%18 : tensor<32x32xf32>) -> tensor<32x32xf32>
              %39 = linalg.fill ins(%cst_3 : f32) outs(%20 : tensor<32x32xf32>) -> tensor<32x32xf32>
              %40:3 = affine.for %arg8 = 0 to 128 iter_args(%arg9 = %39, %arg10 = %38, %arg11 = %37) -> (tensor<32x32xf32>, tensor<32x32xf32>, tensor<32x32x128xf32>) {
                %43 = arith.muli %arg8, %c32 : index
                %c0_6 = arith.constant 0 : index
                %44 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 524288 + d1 * 4096 + d2)>(%33, %c0_6, %43)
                %reinterpret_cast_7 = memref.reinterpret_cast %arg0 to offset: [%44], sizes: [32, 128, 32], strides: [524288, 4096, 1] : memref<32x128x4096xf32> to memref<32x128x32xf32, strided<[524288, 4096, 1], offset: ?>>
                %45 = loom.copy_to_tensor %reinterpret_cast_7, %16 on @L1, interconnect : [], broadcast : [1, 1] : memref<32x128x32xf32, strided<[524288, 4096, 1], offset: ?>>, memref<32x128x32xf32> -> tensor<32x128x32xf32>
                %46 = linalg.fill ins(%cst_1 : f32) outs(%15 : tensor<32x32x32xf32>) -> tensor<32x32x32xf32>
                %47 = linalg.batch_matmul ins(%36, %45 : tensor<32x32x128xf32>, tensor<32x128x32xf32>) outs(%46 : tensor<32x32x32xf32>) -> tensor<32x32x32xf32>
                %48 = linalg.fill ins(%cst_3 : f32) outs(%22 : tensor<32x32xf32>) -> tensor<32x32xf32>
                %49 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%47 : tensor<32x32x32xf32>) outs(%48 : tensor<32x32xf32>) {
                ^bb0(%in: f32, %out: f32):
                  %62 = arith.maximumf %in, %out : f32
                  linalg.yield %62 : f32
                } -> tensor<32x32xf32>
                %50 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%arg9, %49 : tensor<32x32xf32>, tensor<32x32xf32>) outs(%22 : tensor<32x32xf32>) {
                ^bb0(%in: f32, %in_10: f32, %out: f32):
                  %62 = arith.truncf %cst_0 : f64 to f32
                  %63 = arith.mulf %in_10, %62 : f32
                  %64 = arith.cmpf ogt, %in, %63 : f32
                  %65 = arith.select %64, %in, %63 : f32
                  linalg.yield %65 : f32
                } -> tensor<32x32xf32>
                %51 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%47, %50 : tensor<32x32x32xf32>, tensor<32x32xf32>) outs(%15 : tensor<32x32x32xf32>) {
                ^bb0(%in: f32, %in_10: f32, %out: f32):
                  %62 = arith.truncf %cst_0 : f64 to f32
                  %63 = arith.mulf %in, %62 : f32
                  %64 = arith.subf %63, %in_10 : f32
                  %65 = math.powf %cst, %64 : f32
                  linalg.yield %65 : f32
                } -> tensor<32x32x32xf32>
                %52 = linalg.fill ins(%cst_1 : f32) outs(%24 : tensor<32x32xf32>) -> tensor<32x32xf32>
                %53 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%51 : tensor<32x32x32xf32>) outs(%52 : tensor<32x32xf32>) {
                ^bb0(%in: f32, %out: f32):
                  %62 = arith.addf %in, %out : f32
                  linalg.yield %62 : f32
                } -> tensor<32x32xf32>
                %54 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%arg9, %50 : tensor<32x32xf32>, tensor<32x32xf32>) outs(%26 : tensor<32x32xf32>) {
                ^bb0(%in: f32, %in_10: f32, %out: f32):
                  %62 = arith.subf %in, %in_10 : f32
                  %63 = math.powf %cst, %62 : f32
                  linalg.yield %63 : f32
                } -> tensor<32x32xf32>
                %55 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%arg10, %54, %53 : tensor<32x32xf32>, tensor<32x32xf32>, tensor<32x32xf32>) outs(%arg10 : tensor<32x32xf32>) {
                ^bb0(%in: f32, %in_10: f32, %in_11: f32, %out: f32):
                  %62 = arith.mulf %in, %in_10 : f32
                  %63 = arith.addf %62, %in_11 : f32
                  linalg.yield %63 : f32
                } -> tensor<32x32xf32>
                %c0_8 = arith.constant 0 : index
                %56 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 524288 + d1 * 128 + d2)>(%33, %43, %c0_8)
                %reinterpret_cast_9 = memref.reinterpret_cast %arg1 to offset: [%56], sizes: [32, 32, 128], strides: [524288, 128, 1] : memref<32x4096x128xf32> to memref<32x32x128xf32, strided<[524288, 128, 1], offset: ?>>
                %57 = loom.copy_to_tensor %reinterpret_cast_9, %13 on @L1, interconnect : [], broadcast : [1, 1] : memref<32x32x128xf32, strided<[524288, 128, 1], offset: ?>>, memref<32x32x128xf32> -> tensor<32x32x128xf32>
                %58 = linalg.fill ins(%cst_1 : f32) outs(%32 : tensor<32x32x128xf32>) -> tensor<32x32x128xf32>
                %59 = linalg.batch_matmul ins(%51, %57 : tensor<32x32x32xf32>, tensor<32x32x128xf32>) outs(%58 : tensor<32x32x128xf32>) -> tensor<32x32x128xf32>
                %60 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%59, %arg11, %54 : tensor<32x32x128xf32>, tensor<32x32x128xf32>, tensor<32x32xf32>) outs(%arg11 : tensor<32x32x128xf32>) {
                ^bb0(%in: f32, %in_10: f32, %in_11: f32, %out: f32):
                  %62 = arith.mulf %in_10, %in_11 : f32
                  %63 = arith.addf %in, %62 : f32
                  linalg.yield %63 : f32
                } -> tensor<32x32x128xf32>
                %61 = linalg.copy ins(%50 : tensor<32x32xf32>) outs(%arg9 : tensor<32x32xf32>) -> tensor<32x32xf32>
                affine.yield %61, %55, %60 : tensor<32x32xf32>, tensor<32x32xf32>, tensor<32x32x128xf32>
              }
              %41 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%40#2, %40#1 : tensor<32x32x128xf32>, tensor<32x32xf32>) outs(%28 : tensor<32x32x128xf32>) {
              ^bb0(%in: f32, %in_6: f32, %out: f32):
                %43 = arith.divf %in, %in_6 : f32
                linalg.yield %43 : f32
              } -> tensor<32x32x128xf32>
              %c0_4 = arith.constant 0 : index
              %42 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 524288 + d1 * 128 + d2)>(%33, %34, %c0_4)
              %reinterpret_cast_5 = memref.reinterpret_cast %arg3 to offset: [%42], sizes: [32, 32, 128], strides: [524288, 128, 1] : memref<32x4096x128xf32> to memref<32x32x128xf32, strided<[524288, 128, 1], offset: ?>>
              loom.copy_from_tensor %41, %reinterpret_cast_5 on @DRAM : tensor<32x32x128xf32>, memref<32x32x128xf32, strided<[524288, 128, 1], offset: ?>>
            } {loom.iter_type = #loom.iter_type<temporal>}
          } {loom.iter_type = #loom.iter_type<temporal>}
        } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @y}
      } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @x}
      return
    }
  }
  module attributes {loom.pass_name = "Materialize"} {
    func.func @attention__d0i1_d1i1__f10__d_d_a__BB32__BM32__BN32(%arg0: memref<32x128x4096xf32>, %arg1: memref<32x4096x128xf32>, %arg2: memref<32x4096x128xf32>, %arg3: memref<32x4096x128xf32>) {
      %cst = arith.constant 2.000000e+00 : f32
      %cst_0 = arith.constant 0.12751743074602467 : f64
      %cst_1 = arith.constant 0.000000e+00 : f32
      %cst_2 = arith.constant 1.000000e+00 : f32
      %cst_3 = arith.constant 0xFF800000 : f32
      %c32 = arith.constant 32 : index
      affine.parallel (%arg4) = (0) to (8) {
        affine.parallel (%arg5) = (0) to (8) {
          affine.for %arg6 = 0 to 2 {
            affine.for %arg7 = 0 to 1 {
              %12 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 8 + d1 + d2 * 64)>(%arg4, %arg5, %arg6)
              %13 = loom.alloc [32, 32, 128] on @L1 : memref<32x32x128xf32>
              %14 = loom.alloc [32, 32, 32] on @L1 : memref<32x32x32xf32>
              %15 = loom.init_tensor %14[32, 32, 32] : memref<32x32x32xf32> -> tensor<32x32x32xf32>
              %16 = loom.alloc [32, 128, 32] on @L1 : memref<32x128x32xf32>
              %17 = loom.alloc [32, 32] on @L1 : memref<32x32xf32>
              %18 = loom.init_tensor %17[32, 32] : memref<32x32xf32> -> tensor<32x32xf32>
              %19 = loom.alloc [32, 32] on @L1 : memref<32x32xf32>
              %20 = loom.init_tensor %19[32, 32] : memref<32x32xf32> -> tensor<32x32xf32>
              %21 = loom.alloc [32, 32] on @L1 : memref<32x32xf32>
              %22 = loom.init_tensor %21[32, 32] : memref<32x32xf32> -> tensor<32x32xf32>
              %23 = loom.alloc [32, 32] on @L1 : memref<32x32xf32>
              %24 = loom.init_tensor %23[32, 32] : memref<32x32xf32> -> tensor<32x32xf32>
              %25 = loom.alloc [32, 32] on @L1 : memref<32x32xf32>
              %26 = loom.init_tensor %25[32, 32] : memref<32x32xf32> -> tensor<32x32xf32>
              %27 = loom.alloc [32, 32, 128] on @L1 : memref<32x32x128xf32>
              %28 = loom.init_tensor %27[32, 32, 128] : memref<32x32x128xf32> -> tensor<32x32x128xf32>
              %29 = loom.alloc [32, 32, 128] on @L1 : memref<32x32x128xf32>
              %30 = loom.init_tensor %29[32, 32, 128] : memref<32x32x128xf32> -> tensor<32x32x128xf32>
              %31 = loom.alloc [32, 32, 128] on @L1 : memref<32x32x128xf32>
              %32 = loom.init_tensor %31[32, 32, 128] : memref<32x32x128xf32> -> tensor<32x32x128xf32>
              %33 = arith.muli %arg7, %c32 : index
              %34 = arith.muli %12, %c32 : index
              %c0 = arith.constant 0 : index
              %35 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 524288 + d1 * 128 + d2)>(%33, %34, %c0)
              %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%35], sizes: [32, 32, 128], strides: [524288, 128, 1] : memref<32x4096x128xf32> to memref<32x32x128xf32, strided<[524288, 128, 1], offset: ?>>
              %36 = loom.copy_to_tensor %reinterpret_cast, %27 on @L1, interconnect : [], broadcast : [1, 1] : memref<32x32x128xf32, strided<[524288, 128, 1], offset: ?>>, memref<32x32x128xf32> -> tensor<32x32x128xf32>
              %37 = linalg.fill ins(%cst_1 : f32) outs(%30 : tensor<32x32x128xf32>) -> tensor<32x32x128xf32>
              %38 = linalg.fill ins(%cst_2 : f32) outs(%18 : tensor<32x32xf32>) -> tensor<32x32xf32>
              %39 = linalg.fill ins(%cst_3 : f32) outs(%20 : tensor<32x32xf32>) -> tensor<32x32xf32>
              %40:3 = affine.for %arg8 = 0 to 128 iter_args(%arg9 = %39, %arg10 = %38, %arg11 = %37) -> (tensor<32x32xf32>, tensor<32x32xf32>, tensor<32x32x128xf32>) {
                %43 = arith.muli %arg8, %c32 : index
                %c0_6 = arith.constant 0 : index
                %44 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 524288 + d1 * 4096 + d2)>(%33, %c0_6, %43)
                %reinterpret_cast_7 = memref.reinterpret_cast %arg0 to offset: [%44], sizes: [32, 128, 32], strides: [524288, 4096, 1] : memref<32x128x4096xf32> to memref<32x128x32xf32, strided<[524288, 4096, 1], offset: ?>>
                %45 = loom.copy_to_tensor %reinterpret_cast_7, %16 on @L1, interconnect : [], broadcast : [1, 1] : memref<32x128x32xf32, strided<[524288, 4096, 1], offset: ?>>, memref<32x128x32xf32> -> tensor<32x128x32xf32>
                %46 = linalg.fill ins(%cst_1 : f32) outs(%15 : tensor<32x32x32xf32>) -> tensor<32x32x32xf32>
                %47 = linalg.batch_matmul ins(%36, %45 : tensor<32x32x128xf32>, tensor<32x128x32xf32>) outs(%46 : tensor<32x32x32xf32>) -> tensor<32x32x32xf32>
                %48 = linalg.fill ins(%cst_3 : f32) outs(%22 : tensor<32x32xf32>) -> tensor<32x32xf32>
                %49 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%47 : tensor<32x32x32xf32>) outs(%48 : tensor<32x32xf32>) {
                ^bb0(%in: f32, %out: f32):
                  %62 = arith.maximumf %in, %out : f32
                  linalg.yield %62 : f32
                } -> tensor<32x32xf32>
                %50 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%arg9, %49 : tensor<32x32xf32>, tensor<32x32xf32>) outs(%22 : tensor<32x32xf32>) {
                ^bb0(%in: f32, %in_10: f32, %out: f32):
                  %62 = arith.truncf %cst_0 : f64 to f32
                  %63 = arith.mulf %in_10, %62 : f32
                  %64 = arith.cmpf ogt, %in, %63 : f32
                  %65 = arith.select %64, %in, %63 : f32
                  linalg.yield %65 : f32
                } -> tensor<32x32xf32>
                %51 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%47, %50 : tensor<32x32x32xf32>, tensor<32x32xf32>) outs(%15 : tensor<32x32x32xf32>) {
                ^bb0(%in: f32, %in_10: f32, %out: f32):
                  %62 = arith.truncf %cst_0 : f64 to f32
                  %63 = arith.mulf %in, %62 : f32
                  %64 = arith.subf %63, %in_10 : f32
                  %65 = math.powf %cst, %64 : f32
                  linalg.yield %65 : f32
                } -> tensor<32x32x32xf32>
                %52 = linalg.fill ins(%cst_1 : f32) outs(%24 : tensor<32x32xf32>) -> tensor<32x32xf32>
                %53 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%51 : tensor<32x32x32xf32>) outs(%52 : tensor<32x32xf32>) {
                ^bb0(%in: f32, %out: f32):
                  %62 = arith.addf %in, %out : f32
                  linalg.yield %62 : f32
                } -> tensor<32x32xf32>
                %54 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%arg9, %50 : tensor<32x32xf32>, tensor<32x32xf32>) outs(%26 : tensor<32x32xf32>) {
                ^bb0(%in: f32, %in_10: f32, %out: f32):
                  %62 = arith.subf %in, %in_10 : f32
                  %63 = math.powf %cst, %62 : f32
                  linalg.yield %63 : f32
                } -> tensor<32x32xf32>
                %55 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%arg10, %54, %53 : tensor<32x32xf32>, tensor<32x32xf32>, tensor<32x32xf32>) outs(%arg10 : tensor<32x32xf32>) {
                ^bb0(%in: f32, %in_10: f32, %in_11: f32, %out: f32):
                  %62 = arith.mulf %in, %in_10 : f32
                  %63 = arith.addf %62, %in_11 : f32
                  linalg.yield %63 : f32
                } -> tensor<32x32xf32>
                %c0_8 = arith.constant 0 : index
                %56 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 524288 + d1 * 128 + d2)>(%33, %43, %c0_8)
                %reinterpret_cast_9 = memref.reinterpret_cast %arg1 to offset: [%56], sizes: [32, 32, 128], strides: [524288, 128, 1] : memref<32x4096x128xf32> to memref<32x32x128xf32, strided<[524288, 128, 1], offset: ?>>
                %57 = loom.copy_to_tensor %reinterpret_cast_9, %13 on @L1, interconnect : [@horizontal_links, @vertical_links], broadcast : [8, 8] : memref<32x32x128xf32, strided<[524288, 128, 1], offset: ?>>, memref<32x32x128xf32> -> tensor<32x32x128xf32>
                %58 = linalg.fill ins(%cst_1 : f32) outs(%32 : tensor<32x32x128xf32>) -> tensor<32x32x128xf32>
                %59 = linalg.batch_matmul ins(%51, %57 : tensor<32x32x32xf32>, tensor<32x32x128xf32>) outs(%58 : tensor<32x32x128xf32>) -> tensor<32x32x128xf32>
                %60 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%59, %arg11, %54 : tensor<32x32x128xf32>, tensor<32x32x128xf32>, tensor<32x32xf32>) outs(%arg11 : tensor<32x32x128xf32>) {
                ^bb0(%in: f32, %in_10: f32, %in_11: f32, %out: f32):
                  %62 = arith.mulf %in_10, %in_11 : f32
                  %63 = arith.addf %in, %62 : f32
                  linalg.yield %63 : f32
                } -> tensor<32x32x128xf32>
                %61 = linalg.copy ins(%50 : tensor<32x32xf32>) outs(%arg9 : tensor<32x32xf32>) -> tensor<32x32xf32>
                affine.yield %61, %55, %60 : tensor<32x32xf32>, tensor<32x32xf32>, tensor<32x32x128xf32>
              }
              %41 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%40#2, %40#1 : tensor<32x32x128xf32>, tensor<32x32xf32>) outs(%28 : tensor<32x32x128xf32>) {
              ^bb0(%in: f32, %in_6: f32, %out: f32):
                %43 = arith.divf %in, %in_6 : f32
                linalg.yield %43 : f32
              } -> tensor<32x32x128xf32>
              %c0_4 = arith.constant 0 : index
              %42 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 524288 + d1 * 128 + d2)>(%33, %34, %c0_4)
              %reinterpret_cast_5 = memref.reinterpret_cast %arg3 to offset: [%42], sizes: [32, 32, 128], strides: [524288, 128, 1] : memref<32x4096x128xf32> to memref<32x32x128xf32, strided<[524288, 128, 1], offset: ?>>
              loom.copy_from_tensor %41, %reinterpret_cast_5 on @DRAM : tensor<32x32x128xf32>, memref<32x32x128xf32, strided<[524288, 128, 1], offset: ?>>
            } {loom.iter_type = #loom.iter_type<temporal>}
          } {loom.iter_type = #loom.iter_type<temporal>}
        } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @y}
      } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @x}
      return
    }
  }
  module attributes {loom.pass_name = "Materialize"} {
    func.func @attention__d0i1_d1i1__f10__d_d_h__BB32__BM32__BN32(%arg0: memref<32x128x4096xf32>, %arg1: memref<32x4096x128xf32>, %arg2: memref<32x4096x128xf32>, %arg3: memref<32x4096x128xf32>) {
      %cst = arith.constant 2.000000e+00 : f32
      %cst_0 = arith.constant 0.12751743074602467 : f64
      %cst_1 = arith.constant 0.000000e+00 : f32
      %cst_2 = arith.constant 1.000000e+00 : f32
      %cst_3 = arith.constant 0xFF800000 : f32
      %c32 = arith.constant 32 : index
      affine.parallel (%arg4) = (0) to (8) {
        affine.parallel (%arg5) = (0) to (8) {
          affine.for %arg6 = 0 to 2 {
            affine.for %arg7 = 0 to 1 {
              %12 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 8 + d1 + d2 * 64)>(%arg4, %arg5, %arg6)
              %13 = loom.alloc [32, 32, 128] on @L1 : memref<32x32x128xf32>
              %14 = loom.alloc [32, 32, 32] on @L1 : memref<32x32x32xf32>
              %15 = loom.init_tensor %14[32, 32, 32] : memref<32x32x32xf32> -> tensor<32x32x32xf32>
              %16 = loom.alloc [32, 128, 32] on @L1 : memref<32x128x32xf32>
              %17 = loom.alloc [32, 32] on @L1 : memref<32x32xf32>
              %18 = loom.init_tensor %17[32, 32] : memref<32x32xf32> -> tensor<32x32xf32>
              %19 = loom.alloc [32, 32] on @L1 : memref<32x32xf32>
              %20 = loom.init_tensor %19[32, 32] : memref<32x32xf32> -> tensor<32x32xf32>
              %21 = loom.alloc [32, 32] on @L1 : memref<32x32xf32>
              %22 = loom.init_tensor %21[32, 32] : memref<32x32xf32> -> tensor<32x32xf32>
              %23 = loom.alloc [32, 32] on @L1 : memref<32x32xf32>
              %24 = loom.init_tensor %23[32, 32] : memref<32x32xf32> -> tensor<32x32xf32>
              %25 = loom.alloc [32, 32] on @L1 : memref<32x32xf32>
              %26 = loom.init_tensor %25[32, 32] : memref<32x32xf32> -> tensor<32x32xf32>
              %27 = loom.alloc [32, 32, 128] on @L1 : memref<32x32x128xf32>
              %28 = loom.init_tensor %27[32, 32, 128] : memref<32x32x128xf32> -> tensor<32x32x128xf32>
              %29 = loom.alloc [32, 32, 128] on @L1 : memref<32x32x128xf32>
              %30 = loom.init_tensor %29[32, 32, 128] : memref<32x32x128xf32> -> tensor<32x32x128xf32>
              %31 = loom.alloc [32, 32, 128] on @L1 : memref<32x32x128xf32>
              %32 = loom.init_tensor %31[32, 32, 128] : memref<32x32x128xf32> -> tensor<32x32x128xf32>
              %33 = arith.muli %arg7, %c32 : index
              %34 = arith.muli %12, %c32 : index
              %c0 = arith.constant 0 : index
              %35 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 524288 + d1 * 128 + d2)>(%33, %34, %c0)
              %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%35], sizes: [32, 32, 128], strides: [524288, 128, 1] : memref<32x4096x128xf32> to memref<32x32x128xf32, strided<[524288, 128, 1], offset: ?>>
              %36 = loom.copy_to_tensor %reinterpret_cast, %27 on @L1, interconnect : [], broadcast : [1, 1] : memref<32x32x128xf32, strided<[524288, 128, 1], offset: ?>>, memref<32x32x128xf32> -> tensor<32x32x128xf32>
              %37 = linalg.fill ins(%cst_1 : f32) outs(%30 : tensor<32x32x128xf32>) -> tensor<32x32x128xf32>
              %38 = linalg.fill ins(%cst_2 : f32) outs(%18 : tensor<32x32xf32>) -> tensor<32x32xf32>
              %39 = linalg.fill ins(%cst_3 : f32) outs(%20 : tensor<32x32xf32>) -> tensor<32x32xf32>
              %40:3 = affine.for %arg8 = 0 to 128 iter_args(%arg9 = %39, %arg10 = %38, %arg11 = %37) -> (tensor<32x32xf32>, tensor<32x32xf32>, tensor<32x32x128xf32>) {
                %43 = arith.muli %arg8, %c32 : index
                %c0_6 = arith.constant 0 : index
                %44 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 524288 + d1 * 4096 + d2)>(%33, %c0_6, %43)
                %reinterpret_cast_7 = memref.reinterpret_cast %arg0 to offset: [%44], sizes: [32, 128, 32], strides: [524288, 4096, 1] : memref<32x128x4096xf32> to memref<32x128x32xf32, strided<[524288, 4096, 1], offset: ?>>
                %45 = loom.copy_to_tensor %reinterpret_cast_7, %16 on @L1, interconnect : [], broadcast : [1, 1] : memref<32x128x32xf32, strided<[524288, 4096, 1], offset: ?>>, memref<32x128x32xf32> -> tensor<32x128x32xf32>
                %46 = linalg.fill ins(%cst_1 : f32) outs(%15 : tensor<32x32x32xf32>) -> tensor<32x32x32xf32>
                %47 = linalg.batch_matmul ins(%36, %45 : tensor<32x32x128xf32>, tensor<32x128x32xf32>) outs(%46 : tensor<32x32x32xf32>) -> tensor<32x32x32xf32>
                %48 = linalg.fill ins(%cst_3 : f32) outs(%22 : tensor<32x32xf32>) -> tensor<32x32xf32>
                %49 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%47 : tensor<32x32x32xf32>) outs(%48 : tensor<32x32xf32>) {
                ^bb0(%in: f32, %out: f32):
                  %62 = arith.maximumf %in, %out : f32
                  linalg.yield %62 : f32
                } -> tensor<32x32xf32>
                %50 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%arg9, %49 : tensor<32x32xf32>, tensor<32x32xf32>) outs(%22 : tensor<32x32xf32>) {
                ^bb0(%in: f32, %in_10: f32, %out: f32):
                  %62 = arith.truncf %cst_0 : f64 to f32
                  %63 = arith.mulf %in_10, %62 : f32
                  %64 = arith.cmpf ogt, %in, %63 : f32
                  %65 = arith.select %64, %in, %63 : f32
                  linalg.yield %65 : f32
                } -> tensor<32x32xf32>
                %51 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%47, %50 : tensor<32x32x32xf32>, tensor<32x32xf32>) outs(%15 : tensor<32x32x32xf32>) {
                ^bb0(%in: f32, %in_10: f32, %out: f32):
                  %62 = arith.truncf %cst_0 : f64 to f32
                  %63 = arith.mulf %in, %62 : f32
                  %64 = arith.subf %63, %in_10 : f32
                  %65 = math.powf %cst, %64 : f32
                  linalg.yield %65 : f32
                } -> tensor<32x32x32xf32>
                %52 = linalg.fill ins(%cst_1 : f32) outs(%24 : tensor<32x32xf32>) -> tensor<32x32xf32>
                %53 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%51 : tensor<32x32x32xf32>) outs(%52 : tensor<32x32xf32>) {
                ^bb0(%in: f32, %out: f32):
                  %62 = arith.addf %in, %out : f32
                  linalg.yield %62 : f32
                } -> tensor<32x32xf32>
                %54 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%arg9, %50 : tensor<32x32xf32>, tensor<32x32xf32>) outs(%26 : tensor<32x32xf32>) {
                ^bb0(%in: f32, %in_10: f32, %out: f32):
                  %62 = arith.subf %in, %in_10 : f32
                  %63 = math.powf %cst, %62 : f32
                  linalg.yield %63 : f32
                } -> tensor<32x32xf32>
                %55 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%arg10, %54, %53 : tensor<32x32xf32>, tensor<32x32xf32>, tensor<32x32xf32>) outs(%arg10 : tensor<32x32xf32>) {
                ^bb0(%in: f32, %in_10: f32, %in_11: f32, %out: f32):
                  %62 = arith.mulf %in, %in_10 : f32
                  %63 = arith.addf %62, %in_11 : f32
                  linalg.yield %63 : f32
                } -> tensor<32x32xf32>
                %c0_8 = arith.constant 0 : index
                %56 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 524288 + d1 * 128 + d2)>(%33, %43, %c0_8)
                %reinterpret_cast_9 = memref.reinterpret_cast %arg1 to offset: [%56], sizes: [32, 32, 128], strides: [524288, 128, 1] : memref<32x4096x128xf32> to memref<32x32x128xf32, strided<[524288, 128, 1], offset: ?>>
                %57 = loom.copy_to_tensor %reinterpret_cast_9, %13 on @L1, interconnect : [@horizontal_links], broadcast : [1, 8] : memref<32x32x128xf32, strided<[524288, 128, 1], offset: ?>>, memref<32x32x128xf32> -> tensor<32x32x128xf32>
                %58 = linalg.fill ins(%cst_1 : f32) outs(%32 : tensor<32x32x128xf32>) -> tensor<32x32x128xf32>
                %59 = linalg.batch_matmul ins(%51, %57 : tensor<32x32x32xf32>, tensor<32x32x128xf32>) outs(%58 : tensor<32x32x128xf32>) -> tensor<32x32x128xf32>
                %60 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%59, %arg11, %54 : tensor<32x32x128xf32>, tensor<32x32x128xf32>, tensor<32x32xf32>) outs(%arg11 : tensor<32x32x128xf32>) {
                ^bb0(%in: f32, %in_10: f32, %in_11: f32, %out: f32):
                  %62 = arith.mulf %in_10, %in_11 : f32
                  %63 = arith.addf %in, %62 : f32
                  linalg.yield %63 : f32
                } -> tensor<32x32x128xf32>
                %61 = linalg.copy ins(%50 : tensor<32x32xf32>) outs(%arg9 : tensor<32x32xf32>) -> tensor<32x32xf32>
                affine.yield %61, %55, %60 : tensor<32x32xf32>, tensor<32x32xf32>, tensor<32x32x128xf32>
              }
              %41 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%40#2, %40#1 : tensor<32x32x128xf32>, tensor<32x32xf32>) outs(%28 : tensor<32x32x128xf32>) {
              ^bb0(%in: f32, %in_6: f32, %out: f32):
                %43 = arith.divf %in, %in_6 : f32
                linalg.yield %43 : f32
              } -> tensor<32x32x128xf32>
              %c0_4 = arith.constant 0 : index
              %42 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 524288 + d1 * 128 + d2)>(%33, %34, %c0_4)
              %reinterpret_cast_5 = memref.reinterpret_cast %arg3 to offset: [%42], sizes: [32, 32, 128], strides: [524288, 128, 1] : memref<32x4096x128xf32> to memref<32x32x128xf32, strided<[524288, 128, 1], offset: ?>>
              loom.copy_from_tensor %41, %reinterpret_cast_5 on @DRAM : tensor<32x32x128xf32>, memref<32x32x128xf32, strided<[524288, 128, 1], offset: ?>>
            } {loom.iter_type = #loom.iter_type<temporal>}
          } {loom.iter_type = #loom.iter_type<temporal>}
        } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @y}
      } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @x}
      return
    }
  }
  module attributes {loom.pass_name = "Materialize"} {
    func.func @attention__d0i1_d1i1__f10__d_d_v__BB32__BM32__BN32(%arg0: memref<32x128x4096xf32>, %arg1: memref<32x4096x128xf32>, %arg2: memref<32x4096x128xf32>, %arg3: memref<32x4096x128xf32>) {
      %cst = arith.constant 2.000000e+00 : f32
      %cst_0 = arith.constant 0.12751743074602467 : f64
      %cst_1 = arith.constant 0.000000e+00 : f32
      %cst_2 = arith.constant 1.000000e+00 : f32
      %cst_3 = arith.constant 0xFF800000 : f32
      %c32 = arith.constant 32 : index
      affine.parallel (%arg4) = (0) to (8) {
        affine.parallel (%arg5) = (0) to (8) {
          affine.for %arg6 = 0 to 2 {
            affine.for %arg7 = 0 to 1 {
              %12 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 8 + d1 + d2 * 64)>(%arg4, %arg5, %arg6)
              %13 = loom.alloc [32, 32, 128] on @L1 : memref<32x32x128xf32>
              %14 = loom.alloc [32, 32, 32] on @L1 : memref<32x32x32xf32>
              %15 = loom.init_tensor %14[32, 32, 32] : memref<32x32x32xf32> -> tensor<32x32x32xf32>
              %16 = loom.alloc [32, 128, 32] on @L1 : memref<32x128x32xf32>
              %17 = loom.alloc [32, 32] on @L1 : memref<32x32xf32>
              %18 = loom.init_tensor %17[32, 32] : memref<32x32xf32> -> tensor<32x32xf32>
              %19 = loom.alloc [32, 32] on @L1 : memref<32x32xf32>
              %20 = loom.init_tensor %19[32, 32] : memref<32x32xf32> -> tensor<32x32xf32>
              %21 = loom.alloc [32, 32] on @L1 : memref<32x32xf32>
              %22 = loom.init_tensor %21[32, 32] : memref<32x32xf32> -> tensor<32x32xf32>
              %23 = loom.alloc [32, 32] on @L1 : memref<32x32xf32>
              %24 = loom.init_tensor %23[32, 32] : memref<32x32xf32> -> tensor<32x32xf32>
              %25 = loom.alloc [32, 32] on @L1 : memref<32x32xf32>
              %26 = loom.init_tensor %25[32, 32] : memref<32x32xf32> -> tensor<32x32xf32>
              %27 = loom.alloc [32, 32, 128] on @L1 : memref<32x32x128xf32>
              %28 = loom.init_tensor %27[32, 32, 128] : memref<32x32x128xf32> -> tensor<32x32x128xf32>
              %29 = loom.alloc [32, 32, 128] on @L1 : memref<32x32x128xf32>
              %30 = loom.init_tensor %29[32, 32, 128] : memref<32x32x128xf32> -> tensor<32x32x128xf32>
              %31 = loom.alloc [32, 32, 128] on @L1 : memref<32x32x128xf32>
              %32 = loom.init_tensor %31[32, 32, 128] : memref<32x32x128xf32> -> tensor<32x32x128xf32>
              %33 = arith.muli %arg7, %c32 : index
              %34 = arith.muli %12, %c32 : index
              %c0 = arith.constant 0 : index
              %35 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 524288 + d1 * 128 + d2)>(%33, %34, %c0)
              %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%35], sizes: [32, 32, 128], strides: [524288, 128, 1] : memref<32x4096x128xf32> to memref<32x32x128xf32, strided<[524288, 128, 1], offset: ?>>
              %36 = loom.copy_to_tensor %reinterpret_cast, %27 on @L1, interconnect : [], broadcast : [1, 1] : memref<32x32x128xf32, strided<[524288, 128, 1], offset: ?>>, memref<32x32x128xf32> -> tensor<32x32x128xf32>
              %37 = linalg.fill ins(%cst_1 : f32) outs(%30 : tensor<32x32x128xf32>) -> tensor<32x32x128xf32>
              %38 = linalg.fill ins(%cst_2 : f32) outs(%18 : tensor<32x32xf32>) -> tensor<32x32xf32>
              %39 = linalg.fill ins(%cst_3 : f32) outs(%20 : tensor<32x32xf32>) -> tensor<32x32xf32>
              %40:3 = affine.for %arg8 = 0 to 128 iter_args(%arg9 = %39, %arg10 = %38, %arg11 = %37) -> (tensor<32x32xf32>, tensor<32x32xf32>, tensor<32x32x128xf32>) {
                %43 = arith.muli %arg8, %c32 : index
                %c0_6 = arith.constant 0 : index
                %44 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 524288 + d1 * 4096 + d2)>(%33, %c0_6, %43)
                %reinterpret_cast_7 = memref.reinterpret_cast %arg0 to offset: [%44], sizes: [32, 128, 32], strides: [524288, 4096, 1] : memref<32x128x4096xf32> to memref<32x128x32xf32, strided<[524288, 4096, 1], offset: ?>>
                %45 = loom.copy_to_tensor %reinterpret_cast_7, %16 on @L1, interconnect : [], broadcast : [1, 1] : memref<32x128x32xf32, strided<[524288, 4096, 1], offset: ?>>, memref<32x128x32xf32> -> tensor<32x128x32xf32>
                %46 = linalg.fill ins(%cst_1 : f32) outs(%15 : tensor<32x32x32xf32>) -> tensor<32x32x32xf32>
                %47 = linalg.batch_matmul ins(%36, %45 : tensor<32x32x128xf32>, tensor<32x128x32xf32>) outs(%46 : tensor<32x32x32xf32>) -> tensor<32x32x32xf32>
                %48 = linalg.fill ins(%cst_3 : f32) outs(%22 : tensor<32x32xf32>) -> tensor<32x32xf32>
                %49 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%47 : tensor<32x32x32xf32>) outs(%48 : tensor<32x32xf32>) {
                ^bb0(%in: f32, %out: f32):
                  %62 = arith.maximumf %in, %out : f32
                  linalg.yield %62 : f32
                } -> tensor<32x32xf32>
                %50 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%arg9, %49 : tensor<32x32xf32>, tensor<32x32xf32>) outs(%22 : tensor<32x32xf32>) {
                ^bb0(%in: f32, %in_10: f32, %out: f32):
                  %62 = arith.truncf %cst_0 : f64 to f32
                  %63 = arith.mulf %in_10, %62 : f32
                  %64 = arith.cmpf ogt, %in, %63 : f32
                  %65 = arith.select %64, %in, %63 : f32
                  linalg.yield %65 : f32
                } -> tensor<32x32xf32>
                %51 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%47, %50 : tensor<32x32x32xf32>, tensor<32x32xf32>) outs(%15 : tensor<32x32x32xf32>) {
                ^bb0(%in: f32, %in_10: f32, %out: f32):
                  %62 = arith.truncf %cst_0 : f64 to f32
                  %63 = arith.mulf %in, %62 : f32
                  %64 = arith.subf %63, %in_10 : f32
                  %65 = math.powf %cst, %64 : f32
                  linalg.yield %65 : f32
                } -> tensor<32x32x32xf32>
                %52 = linalg.fill ins(%cst_1 : f32) outs(%24 : tensor<32x32xf32>) -> tensor<32x32xf32>
                %53 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%51 : tensor<32x32x32xf32>) outs(%52 : tensor<32x32xf32>) {
                ^bb0(%in: f32, %out: f32):
                  %62 = arith.addf %in, %out : f32
                  linalg.yield %62 : f32
                } -> tensor<32x32xf32>
                %54 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%arg9, %50 : tensor<32x32xf32>, tensor<32x32xf32>) outs(%26 : tensor<32x32xf32>) {
                ^bb0(%in: f32, %in_10: f32, %out: f32):
                  %62 = arith.subf %in, %in_10 : f32
                  %63 = math.powf %cst, %62 : f32
                  linalg.yield %63 : f32
                } -> tensor<32x32xf32>
                %55 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%arg10, %54, %53 : tensor<32x32xf32>, tensor<32x32xf32>, tensor<32x32xf32>) outs(%arg10 : tensor<32x32xf32>) {
                ^bb0(%in: f32, %in_10: f32, %in_11: f32, %out: f32):
                  %62 = arith.mulf %in, %in_10 : f32
                  %63 = arith.addf %62, %in_11 : f32
                  linalg.yield %63 : f32
                } -> tensor<32x32xf32>
                %c0_8 = arith.constant 0 : index
                %56 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 524288 + d1 * 128 + d2)>(%33, %43, %c0_8)
                %reinterpret_cast_9 = memref.reinterpret_cast %arg1 to offset: [%56], sizes: [32, 32, 128], strides: [524288, 128, 1] : memref<32x4096x128xf32> to memref<32x32x128xf32, strided<[524288, 128, 1], offset: ?>>
                %57 = loom.copy_to_tensor %reinterpret_cast_9, %13 on @L1, interconnect : [@vertical_links], broadcast : [8, 1] : memref<32x32x128xf32, strided<[524288, 128, 1], offset: ?>>, memref<32x32x128xf32> -> tensor<32x32x128xf32>
                %58 = linalg.fill ins(%cst_1 : f32) outs(%32 : tensor<32x32x128xf32>) -> tensor<32x32x128xf32>
                %59 = linalg.batch_matmul ins(%51, %57 : tensor<32x32x32xf32>, tensor<32x32x128xf32>) outs(%58 : tensor<32x32x128xf32>) -> tensor<32x32x128xf32>
                %60 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%59, %arg11, %54 : tensor<32x32x128xf32>, tensor<32x32x128xf32>, tensor<32x32xf32>) outs(%arg11 : tensor<32x32x128xf32>) {
                ^bb0(%in: f32, %in_10: f32, %in_11: f32, %out: f32):
                  %62 = arith.mulf %in_10, %in_11 : f32
                  %63 = arith.addf %in, %62 : f32
                  linalg.yield %63 : f32
                } -> tensor<32x32x128xf32>
                %61 = linalg.copy ins(%50 : tensor<32x32xf32>) outs(%arg9 : tensor<32x32xf32>) -> tensor<32x32xf32>
                affine.yield %61, %55, %60 : tensor<32x32xf32>, tensor<32x32xf32>, tensor<32x32x128xf32>
              }
              %41 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%40#2, %40#1 : tensor<32x32x128xf32>, tensor<32x32xf32>) outs(%28 : tensor<32x32x128xf32>) {
              ^bb0(%in: f32, %in_6: f32, %out: f32):
                %43 = arith.divf %in, %in_6 : f32
                linalg.yield %43 : f32
              } -> tensor<32x32x128xf32>
              %c0_4 = arith.constant 0 : index
              %42 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 524288 + d1 * 128 + d2)>(%33, %34, %c0_4)
              %reinterpret_cast_5 = memref.reinterpret_cast %arg3 to offset: [%42], sizes: [32, 32, 128], strides: [524288, 128, 1] : memref<32x4096x128xf32> to memref<32x32x128xf32, strided<[524288, 128, 1], offset: ?>>
              loom.copy_from_tensor %41, %reinterpret_cast_5 on @DRAM : tensor<32x32x128xf32>, memref<32x32x128xf32, strided<[524288, 128, 1], offset: ?>>
            } {loom.iter_type = #loom.iter_type<temporal>}
          } {loom.iter_type = #loom.iter_type<temporal>}
        } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @y}
      } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @x}
      return
    }
  }
  module attributes {loom.pass_name = "Materialize"} {
    func.func @attention__d0i1_d1i1__f10__d_a_d__BB32__BM32__BN32(%arg0: memref<32x128x4096xf32>, %arg1: memref<32x4096x128xf32>, %arg2: memref<32x4096x128xf32>, %arg3: memref<32x4096x128xf32>) {
      %cst = arith.constant 2.000000e+00 : f32
      %cst_0 = arith.constant 0.12751743074602467 : f64
      %cst_1 = arith.constant 0.000000e+00 : f32
      %cst_2 = arith.constant 1.000000e+00 : f32
      %cst_3 = arith.constant 0xFF800000 : f32
      %c32 = arith.constant 32 : index
      affine.parallel (%arg4) = (0) to (8) {
        affine.parallel (%arg5) = (0) to (8) {
          affine.for %arg6 = 0 to 2 {
            affine.for %arg7 = 0 to 1 {
              %12 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 8 + d1 + d2 * 64)>(%arg4, %arg5, %arg6)
              %13 = loom.alloc [32, 32, 128] on @L1 : memref<32x32x128xf32>
              %14 = loom.alloc [32, 32, 32] on @L1 : memref<32x32x32xf32>
              %15 = loom.init_tensor %14[32, 32, 32] : memref<32x32x32xf32> -> tensor<32x32x32xf32>
              %16 = loom.alloc [32, 128, 32] on @L1 : memref<32x128x32xf32>
              %17 = loom.alloc [32, 32] on @L1 : memref<32x32xf32>
              %18 = loom.init_tensor %17[32, 32] : memref<32x32xf32> -> tensor<32x32xf32>
              %19 = loom.alloc [32, 32] on @L1 : memref<32x32xf32>
              %20 = loom.init_tensor %19[32, 32] : memref<32x32xf32> -> tensor<32x32xf32>
              %21 = loom.alloc [32, 32] on @L1 : memref<32x32xf32>
              %22 = loom.init_tensor %21[32, 32] : memref<32x32xf32> -> tensor<32x32xf32>
              %23 = loom.alloc [32, 32] on @L1 : memref<32x32xf32>
              %24 = loom.init_tensor %23[32, 32] : memref<32x32xf32> -> tensor<32x32xf32>
              %25 = loom.alloc [32, 32] on @L1 : memref<32x32xf32>
              %26 = loom.init_tensor %25[32, 32] : memref<32x32xf32> -> tensor<32x32xf32>
              %27 = loom.alloc [32, 32, 128] on @L1 : memref<32x32x128xf32>
              %28 = loom.init_tensor %27[32, 32, 128] : memref<32x32x128xf32> -> tensor<32x32x128xf32>
              %29 = loom.alloc [32, 32, 128] on @L1 : memref<32x32x128xf32>
              %30 = loom.init_tensor %29[32, 32, 128] : memref<32x32x128xf32> -> tensor<32x32x128xf32>
              %31 = loom.alloc [32, 32, 128] on @L1 : memref<32x32x128xf32>
              %32 = loom.init_tensor %31[32, 32, 128] : memref<32x32x128xf32> -> tensor<32x32x128xf32>
              %33 = arith.muli %arg7, %c32 : index
              %34 = arith.muli %12, %c32 : index
              %c0 = arith.constant 0 : index
              %35 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 524288 + d1 * 128 + d2)>(%33, %34, %c0)
              %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%35], sizes: [32, 32, 128], strides: [524288, 128, 1] : memref<32x4096x128xf32> to memref<32x32x128xf32, strided<[524288, 128, 1], offset: ?>>
              %36 = loom.copy_to_tensor %reinterpret_cast, %27 on @L1, interconnect : [], broadcast : [1, 1] : memref<32x32x128xf32, strided<[524288, 128, 1], offset: ?>>, memref<32x32x128xf32> -> tensor<32x32x128xf32>
              %37 = linalg.fill ins(%cst_1 : f32) outs(%30 : tensor<32x32x128xf32>) -> tensor<32x32x128xf32>
              %38 = linalg.fill ins(%cst_2 : f32) outs(%18 : tensor<32x32xf32>) -> tensor<32x32xf32>
              %39 = linalg.fill ins(%cst_3 : f32) outs(%20 : tensor<32x32xf32>) -> tensor<32x32xf32>
              %40:3 = affine.for %arg8 = 0 to 128 iter_args(%arg9 = %39, %arg10 = %38, %arg11 = %37) -> (tensor<32x32xf32>, tensor<32x32xf32>, tensor<32x32x128xf32>) {
                %43 = arith.muli %arg8, %c32 : index
                %c0_6 = arith.constant 0 : index
                %44 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 524288 + d1 * 4096 + d2)>(%33, %c0_6, %43)
                %reinterpret_cast_7 = memref.reinterpret_cast %arg0 to offset: [%44], sizes: [32, 128, 32], strides: [524288, 4096, 1] : memref<32x128x4096xf32> to memref<32x128x32xf32, strided<[524288, 4096, 1], offset: ?>>
                %45 = loom.copy_to_tensor %reinterpret_cast_7, %16 on @L1, interconnect : [@horizontal_links, @vertical_links], broadcast : [8, 8] : memref<32x128x32xf32, strided<[524288, 4096, 1], offset: ?>>, memref<32x128x32xf32> -> tensor<32x128x32xf32>
                %46 = linalg.fill ins(%cst_1 : f32) outs(%15 : tensor<32x32x32xf32>) -> tensor<32x32x32xf32>
                %47 = linalg.batch_matmul ins(%36, %45 : tensor<32x32x128xf32>, tensor<32x128x32xf32>) outs(%46 : tensor<32x32x32xf32>) -> tensor<32x32x32xf32>
                %48 = linalg.fill ins(%cst_3 : f32) outs(%22 : tensor<32x32xf32>) -> tensor<32x32xf32>
                %49 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%47 : tensor<32x32x32xf32>) outs(%48 : tensor<32x32xf32>) {
                ^bb0(%in: f32, %out: f32):
                  %62 = arith.maximumf %in, %out : f32
                  linalg.yield %62 : f32
                } -> tensor<32x32xf32>
                %50 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%arg9, %49 : tensor<32x32xf32>, tensor<32x32xf32>) outs(%22 : tensor<32x32xf32>) {
                ^bb0(%in: f32, %in_10: f32, %out: f32):
                  %62 = arith.truncf %cst_0 : f64 to f32
                  %63 = arith.mulf %in_10, %62 : f32
                  %64 = arith.cmpf ogt, %in, %63 : f32
                  %65 = arith.select %64, %in, %63 : f32
                  linalg.yield %65 : f32
                } -> tensor<32x32xf32>
                %51 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%47, %50 : tensor<32x32x32xf32>, tensor<32x32xf32>) outs(%15 : tensor<32x32x32xf32>) {
                ^bb0(%in: f32, %in_10: f32, %out: f32):
                  %62 = arith.truncf %cst_0 : f64 to f32
                  %63 = arith.mulf %in, %62 : f32
                  %64 = arith.subf %63, %in_10 : f32
                  %65 = math.powf %cst, %64 : f32
                  linalg.yield %65 : f32
                } -> tensor<32x32x32xf32>
                %52 = linalg.fill ins(%cst_1 : f32) outs(%24 : tensor<32x32xf32>) -> tensor<32x32xf32>
                %53 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%51 : tensor<32x32x32xf32>) outs(%52 : tensor<32x32xf32>) {
                ^bb0(%in: f32, %out: f32):
                  %62 = arith.addf %in, %out : f32
                  linalg.yield %62 : f32
                } -> tensor<32x32xf32>
                %54 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%arg9, %50 : tensor<32x32xf32>, tensor<32x32xf32>) outs(%26 : tensor<32x32xf32>) {
                ^bb0(%in: f32, %in_10: f32, %out: f32):
                  %62 = arith.subf %in, %in_10 : f32
                  %63 = math.powf %cst, %62 : f32
                  linalg.yield %63 : f32
                } -> tensor<32x32xf32>
                %55 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%arg10, %54, %53 : tensor<32x32xf32>, tensor<32x32xf32>, tensor<32x32xf32>) outs(%arg10 : tensor<32x32xf32>) {
                ^bb0(%in: f32, %in_10: f32, %in_11: f32, %out: f32):
                  %62 = arith.mulf %in, %in_10 : f32
                  %63 = arith.addf %62, %in_11 : f32
                  linalg.yield %63 : f32
                } -> tensor<32x32xf32>
                %c0_8 = arith.constant 0 : index
                %56 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 524288 + d1 * 128 + d2)>(%33, %43, %c0_8)
                %reinterpret_cast_9 = memref.reinterpret_cast %arg1 to offset: [%56], sizes: [32, 32, 128], strides: [524288, 128, 1] : memref<32x4096x128xf32> to memref<32x32x128xf32, strided<[524288, 128, 1], offset: ?>>
                %57 = loom.copy_to_tensor %reinterpret_cast_9, %13 on @L1, interconnect : [], broadcast : [1, 1] : memref<32x32x128xf32, strided<[524288, 128, 1], offset: ?>>, memref<32x32x128xf32> -> tensor<32x32x128xf32>
                %58 = linalg.fill ins(%cst_1 : f32) outs(%32 : tensor<32x32x128xf32>) -> tensor<32x32x128xf32>
                %59 = linalg.batch_matmul ins(%51, %57 : tensor<32x32x32xf32>, tensor<32x32x128xf32>) outs(%58 : tensor<32x32x128xf32>) -> tensor<32x32x128xf32>
                %60 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%59, %arg11, %54 : tensor<32x32x128xf32>, tensor<32x32x128xf32>, tensor<32x32xf32>) outs(%arg11 : tensor<32x32x128xf32>) {
                ^bb0(%in: f32, %in_10: f32, %in_11: f32, %out: f32):
                  %62 = arith.mulf %in_10, %in_11 : f32
                  %63 = arith.addf %in, %62 : f32
                  linalg.yield %63 : f32
                } -> tensor<32x32x128xf32>
                %61 = linalg.copy ins(%50 : tensor<32x32xf32>) outs(%arg9 : tensor<32x32xf32>) -> tensor<32x32xf32>
                affine.yield %61, %55, %60 : tensor<32x32xf32>, tensor<32x32xf32>, tensor<32x32x128xf32>
              }
              %41 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%40#2, %40#1 : tensor<32x32x128xf32>, tensor<32x32xf32>) outs(%28 : tensor<32x32x128xf32>) {
              ^bb0(%in: f32, %in_6: f32, %out: f32):
                %43 = arith.divf %in, %in_6 : f32
                linalg.yield %43 : f32
              } -> tensor<32x32x128xf32>
              %c0_4 = arith.constant 0 : index
              %42 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 524288 + d1 * 128 + d2)>(%33, %34, %c0_4)
              %reinterpret_cast_5 = memref.reinterpret_cast %arg3 to offset: [%42], sizes: [32, 32, 128], strides: [524288, 128, 1] : memref<32x4096x128xf32> to memref<32x32x128xf32, strided<[524288, 128, 1], offset: ?>>
              loom.copy_from_tensor %41, %reinterpret_cast_5 on @DRAM : tensor<32x32x128xf32>, memref<32x32x128xf32, strided<[524288, 128, 1], offset: ?>>
            } {loom.iter_type = #loom.iter_type<temporal>}
          } {loom.iter_type = #loom.iter_type<temporal>}
        } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @y}
      } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @x}
      return
    }
  }
  module attributes {loom.pass_name = "Materialize"} {
    func.func @attention__d0i1_d1i1__f10__d_a_a__BB32__BM32__BN32(%arg0: memref<32x128x4096xf32>, %arg1: memref<32x4096x128xf32>, %arg2: memref<32x4096x128xf32>, %arg3: memref<32x4096x128xf32>) {
      %cst = arith.constant 2.000000e+00 : f32
      %cst_0 = arith.constant 0.12751743074602467 : f64
      %cst_1 = arith.constant 0.000000e+00 : f32
      %cst_2 = arith.constant 1.000000e+00 : f32
      %cst_3 = arith.constant 0xFF800000 : f32
      %c32 = arith.constant 32 : index
      affine.parallel (%arg4) = (0) to (8) {
        affine.parallel (%arg5) = (0) to (8) {
          affine.for %arg6 = 0 to 2 {
            affine.for %arg7 = 0 to 1 {
              %12 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 8 + d1 + d2 * 64)>(%arg4, %arg5, %arg6)
              %13 = loom.alloc [32, 32, 128] on @L1 : memref<32x32x128xf32>
              %14 = loom.alloc [32, 32, 32] on @L1 : memref<32x32x32xf32>
              %15 = loom.init_tensor %14[32, 32, 32] : memref<32x32x32xf32> -> tensor<32x32x32xf32>
              %16 = loom.alloc [32, 128, 32] on @L1 : memref<32x128x32xf32>
              %17 = loom.alloc [32, 32] on @L1 : memref<32x32xf32>
              %18 = loom.init_tensor %17[32, 32] : memref<32x32xf32> -> tensor<32x32xf32>
              %19 = loom.alloc [32, 32] on @L1 : memref<32x32xf32>
              %20 = loom.init_tensor %19[32, 32] : memref<32x32xf32> -> tensor<32x32xf32>
              %21 = loom.alloc [32, 32] on @L1 : memref<32x32xf32>
              %22 = loom.init_tensor %21[32, 32] : memref<32x32xf32> -> tensor<32x32xf32>
              %23 = loom.alloc [32, 32] on @L1 : memref<32x32xf32>
              %24 = loom.init_tensor %23[32, 32] : memref<32x32xf32> -> tensor<32x32xf32>
              %25 = loom.alloc [32, 32] on @L1 : memref<32x32xf32>
              %26 = loom.init_tensor %25[32, 32] : memref<32x32xf32> -> tensor<32x32xf32>
              %27 = loom.alloc [32, 32, 128] on @L1 : memref<32x32x128xf32>
              %28 = loom.init_tensor %27[32, 32, 128] : memref<32x32x128xf32> -> tensor<32x32x128xf32>
              %29 = loom.alloc [32, 32, 128] on @L1 : memref<32x32x128xf32>
              %30 = loom.init_tensor %29[32, 32, 128] : memref<32x32x128xf32> -> tensor<32x32x128xf32>
              %31 = loom.alloc [32, 32, 128] on @L1 : memref<32x32x128xf32>
              %32 = loom.init_tensor %31[32, 32, 128] : memref<32x32x128xf32> -> tensor<32x32x128xf32>
              %33 = arith.muli %arg7, %c32 : index
              %34 = arith.muli %12, %c32 : index
              %c0 = arith.constant 0 : index
              %35 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 524288 + d1 * 128 + d2)>(%33, %34, %c0)
              %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%35], sizes: [32, 32, 128], strides: [524288, 128, 1] : memref<32x4096x128xf32> to memref<32x32x128xf32, strided<[524288, 128, 1], offset: ?>>
              %36 = loom.copy_to_tensor %reinterpret_cast, %27 on @L1, interconnect : [], broadcast : [1, 1] : memref<32x32x128xf32, strided<[524288, 128, 1], offset: ?>>, memref<32x32x128xf32> -> tensor<32x32x128xf32>
              %37 = linalg.fill ins(%cst_1 : f32) outs(%30 : tensor<32x32x128xf32>) -> tensor<32x32x128xf32>
              %38 = linalg.fill ins(%cst_2 : f32) outs(%18 : tensor<32x32xf32>) -> tensor<32x32xf32>
              %39 = linalg.fill ins(%cst_3 : f32) outs(%20 : tensor<32x32xf32>) -> tensor<32x32xf32>
              %40:3 = affine.for %arg8 = 0 to 128 iter_args(%arg9 = %39, %arg10 = %38, %arg11 = %37) -> (tensor<32x32xf32>, tensor<32x32xf32>, tensor<32x32x128xf32>) {
                %43 = arith.muli %arg8, %c32 : index
                %c0_6 = arith.constant 0 : index
                %44 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 524288 + d1 * 4096 + d2)>(%33, %c0_6, %43)
                %reinterpret_cast_7 = memref.reinterpret_cast %arg0 to offset: [%44], sizes: [32, 128, 32], strides: [524288, 4096, 1] : memref<32x128x4096xf32> to memref<32x128x32xf32, strided<[524288, 4096, 1], offset: ?>>
                %45 = loom.copy_to_tensor %reinterpret_cast_7, %16 on @L1, interconnect : [@horizontal_links, @vertical_links], broadcast : [8, 8] : memref<32x128x32xf32, strided<[524288, 4096, 1], offset: ?>>, memref<32x128x32xf32> -> tensor<32x128x32xf32>
                %46 = linalg.fill ins(%cst_1 : f32) outs(%15 : tensor<32x32x32xf32>) -> tensor<32x32x32xf32>
                %47 = linalg.batch_matmul ins(%36, %45 : tensor<32x32x128xf32>, tensor<32x128x32xf32>) outs(%46 : tensor<32x32x32xf32>) -> tensor<32x32x32xf32>
                %48 = linalg.fill ins(%cst_3 : f32) outs(%22 : tensor<32x32xf32>) -> tensor<32x32xf32>
                %49 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%47 : tensor<32x32x32xf32>) outs(%48 : tensor<32x32xf32>) {
                ^bb0(%in: f32, %out: f32):
                  %62 = arith.maximumf %in, %out : f32
                  linalg.yield %62 : f32
                } -> tensor<32x32xf32>
                %50 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%arg9, %49 : tensor<32x32xf32>, tensor<32x32xf32>) outs(%22 : tensor<32x32xf32>) {
                ^bb0(%in: f32, %in_10: f32, %out: f32):
                  %62 = arith.truncf %cst_0 : f64 to f32
                  %63 = arith.mulf %in_10, %62 : f32
                  %64 = arith.cmpf ogt, %in, %63 : f32
                  %65 = arith.select %64, %in, %63 : f32
                  linalg.yield %65 : f32
                } -> tensor<32x32xf32>
                %51 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%47, %50 : tensor<32x32x32xf32>, tensor<32x32xf32>) outs(%15 : tensor<32x32x32xf32>) {
                ^bb0(%in: f32, %in_10: f32, %out: f32):
                  %62 = arith.truncf %cst_0 : f64 to f32
                  %63 = arith.mulf %in, %62 : f32
                  %64 = arith.subf %63, %in_10 : f32
                  %65 = math.powf %cst, %64 : f32
                  linalg.yield %65 : f32
                } -> tensor<32x32x32xf32>
                %52 = linalg.fill ins(%cst_1 : f32) outs(%24 : tensor<32x32xf32>) -> tensor<32x32xf32>
                %53 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%51 : tensor<32x32x32xf32>) outs(%52 : tensor<32x32xf32>) {
                ^bb0(%in: f32, %out: f32):
                  %62 = arith.addf %in, %out : f32
                  linalg.yield %62 : f32
                } -> tensor<32x32xf32>
                %54 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%arg9, %50 : tensor<32x32xf32>, tensor<32x32xf32>) outs(%26 : tensor<32x32xf32>) {
                ^bb0(%in: f32, %in_10: f32, %out: f32):
                  %62 = arith.subf %in, %in_10 : f32
                  %63 = math.powf %cst, %62 : f32
                  linalg.yield %63 : f32
                } -> tensor<32x32xf32>
                %55 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%arg10, %54, %53 : tensor<32x32xf32>, tensor<32x32xf32>, tensor<32x32xf32>) outs(%arg10 : tensor<32x32xf32>) {
                ^bb0(%in: f32, %in_10: f32, %in_11: f32, %out: f32):
                  %62 = arith.mulf %in, %in_10 : f32
                  %63 = arith.addf %62, %in_11 : f32
                  linalg.yield %63 : f32
                } -> tensor<32x32xf32>
                %c0_8 = arith.constant 0 : index
                %56 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 524288 + d1 * 128 + d2)>(%33, %43, %c0_8)
                %reinterpret_cast_9 = memref.reinterpret_cast %arg1 to offset: [%56], sizes: [32, 32, 128], strides: [524288, 128, 1] : memref<32x4096x128xf32> to memref<32x32x128xf32, strided<[524288, 128, 1], offset: ?>>
                %57 = loom.copy_to_tensor %reinterpret_cast_9, %13 on @L1, interconnect : [@horizontal_links, @vertical_links], broadcast : [8, 8] : memref<32x32x128xf32, strided<[524288, 128, 1], offset: ?>>, memref<32x32x128xf32> -> tensor<32x32x128xf32>
                %58 = linalg.fill ins(%cst_1 : f32) outs(%32 : tensor<32x32x128xf32>) -> tensor<32x32x128xf32>
                %59 = linalg.batch_matmul ins(%51, %57 : tensor<32x32x32xf32>, tensor<32x32x128xf32>) outs(%58 : tensor<32x32x128xf32>) -> tensor<32x32x128xf32>
                %60 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%59, %arg11, %54 : tensor<32x32x128xf32>, tensor<32x32x128xf32>, tensor<32x32xf32>) outs(%arg11 : tensor<32x32x128xf32>) {
                ^bb0(%in: f32, %in_10: f32, %in_11: f32, %out: f32):
                  %62 = arith.mulf %in_10, %in_11 : f32
                  %63 = arith.addf %in, %62 : f32
                  linalg.yield %63 : f32
                } -> tensor<32x32x128xf32>
                %61 = linalg.copy ins(%50 : tensor<32x32xf32>) outs(%arg9 : tensor<32x32xf32>) -> tensor<32x32xf32>
                affine.yield %61, %55, %60 : tensor<32x32xf32>, tensor<32x32xf32>, tensor<32x32x128xf32>
              }
              %41 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%40#2, %40#1 : tensor<32x32x128xf32>, tensor<32x32xf32>) outs(%28 : tensor<32x32x128xf32>) {
              ^bb0(%in: f32, %in_6: f32, %out: f32):
                %43 = arith.divf %in, %in_6 : f32
                linalg.yield %43 : f32
              } -> tensor<32x32x128xf32>
              %c0_4 = arith.constant 0 : index
              %42 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 524288 + d1 * 128 + d2)>(%33, %34, %c0_4)
              %reinterpret_cast_5 = memref.reinterpret_cast %arg3 to offset: [%42], sizes: [32, 32, 128], strides: [524288, 128, 1] : memref<32x4096x128xf32> to memref<32x32x128xf32, strided<[524288, 128, 1], offset: ?>>
              loom.copy_from_tensor %41, %reinterpret_cast_5 on @DRAM : tensor<32x32x128xf32>, memref<32x32x128xf32, strided<[524288, 128, 1], offset: ?>>
            } {loom.iter_type = #loom.iter_type<temporal>}
          } {loom.iter_type = #loom.iter_type<temporal>}
        } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @y}
      } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @x}
      return
    }
  }
  module attributes {loom.pass_name = "Materialize"} {
    func.func @attention__d0i1_d1i1__f10__d_a_h__BB32__BM32__BN32(%arg0: memref<32x128x4096xf32>, %arg1: memref<32x4096x128xf32>, %arg2: memref<32x4096x128xf32>, %arg3: memref<32x4096x128xf32>) {
      %cst = arith.constant 2.000000e+00 : f32
      %cst_0 = arith.constant 0.12751743074602467 : f64
      %cst_1 = arith.constant 0.000000e+00 : f32
      %cst_2 = arith.constant 1.000000e+00 : f32
      %cst_3 = arith.constant 0xFF800000 : f32
      %c32 = arith.constant 32 : index
      affine.parallel (%arg4) = (0) to (8) {
        affine.parallel (%arg5) = (0) to (8) {
          affine.for %arg6 = 0 to 2 {
            affine.for %arg7 = 0 to 1 {
              %12 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 8 + d1 + d2 * 64)>(%arg4, %arg5, %arg6)
              %13 = loom.alloc [32, 32, 128] on @L1 : memref<32x32x128xf32>
              %14 = loom.alloc [32, 32, 32] on @L1 : memref<32x32x32xf32>
              %15 = loom.init_tensor %14[32, 32, 32] : memref<32x32x32xf32> -> tensor<32x32x32xf32>
              %16 = loom.alloc [32, 128, 32] on @L1 : memref<32x128x32xf32>
              %17 = loom.alloc [32, 32] on @L1 : memref<32x32xf32>
              %18 = loom.init_tensor %17[32, 32] : memref<32x32xf32> -> tensor<32x32xf32>
              %19 = loom.alloc [32, 32] on @L1 : memref<32x32xf32>
              %20 = loom.init_tensor %19[32, 32] : memref<32x32xf32> -> tensor<32x32xf32>
              %21 = loom.alloc [32, 32] on @L1 : memref<32x32xf32>
              %22 = loom.init_tensor %21[32, 32] : memref<32x32xf32> -> tensor<32x32xf32>
              %23 = loom.alloc [32, 32] on @L1 : memref<32x32xf32>
              %24 = loom.init_tensor %23[32, 32] : memref<32x32xf32> -> tensor<32x32xf32>
              %25 = loom.alloc [32, 32] on @L1 : memref<32x32xf32>
              %26 = loom.init_tensor %25[32, 32] : memref<32x32xf32> -> tensor<32x32xf32>
              %27 = loom.alloc [32, 32, 128] on @L1 : memref<32x32x128xf32>
              %28 = loom.init_tensor %27[32, 32, 128] : memref<32x32x128xf32> -> tensor<32x32x128xf32>
              %29 = loom.alloc [32, 32, 128] on @L1 : memref<32x32x128xf32>
              %30 = loom.init_tensor %29[32, 32, 128] : memref<32x32x128xf32> -> tensor<32x32x128xf32>
              %31 = loom.alloc [32, 32, 128] on @L1 : memref<32x32x128xf32>
              %32 = loom.init_tensor %31[32, 32, 128] : memref<32x32x128xf32> -> tensor<32x32x128xf32>
              %33 = arith.muli %arg7, %c32 : index
              %34 = arith.muli %12, %c32 : index
              %c0 = arith.constant 0 : index
              %35 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 524288 + d1 * 128 + d2)>(%33, %34, %c0)
              %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%35], sizes: [32, 32, 128], strides: [524288, 128, 1] : memref<32x4096x128xf32> to memref<32x32x128xf32, strided<[524288, 128, 1], offset: ?>>
              %36 = loom.copy_to_tensor %reinterpret_cast, %27 on @L1, interconnect : [], broadcast : [1, 1] : memref<32x32x128xf32, strided<[524288, 128, 1], offset: ?>>, memref<32x32x128xf32> -> tensor<32x32x128xf32>
              %37 = linalg.fill ins(%cst_1 : f32) outs(%30 : tensor<32x32x128xf32>) -> tensor<32x32x128xf32>
              %38 = linalg.fill ins(%cst_2 : f32) outs(%18 : tensor<32x32xf32>) -> tensor<32x32xf32>
              %39 = linalg.fill ins(%cst_3 : f32) outs(%20 : tensor<32x32xf32>) -> tensor<32x32xf32>
              %40:3 = affine.for %arg8 = 0 to 128 iter_args(%arg9 = %39, %arg10 = %38, %arg11 = %37) -> (tensor<32x32xf32>, tensor<32x32xf32>, tensor<32x32x128xf32>) {
                %43 = arith.muli %arg8, %c32 : index
                %c0_6 = arith.constant 0 : index
                %44 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 524288 + d1 * 4096 + d2)>(%33, %c0_6, %43)
                %reinterpret_cast_7 = memref.reinterpret_cast %arg0 to offset: [%44], sizes: [32, 128, 32], strides: [524288, 4096, 1] : memref<32x128x4096xf32> to memref<32x128x32xf32, strided<[524288, 4096, 1], offset: ?>>
                %45 = loom.copy_to_tensor %reinterpret_cast_7, %16 on @L1, interconnect : [@horizontal_links, @vertical_links], broadcast : [8, 8] : memref<32x128x32xf32, strided<[524288, 4096, 1], offset: ?>>, memref<32x128x32xf32> -> tensor<32x128x32xf32>
                %46 = linalg.fill ins(%cst_1 : f32) outs(%15 : tensor<32x32x32xf32>) -> tensor<32x32x32xf32>
                %47 = linalg.batch_matmul ins(%36, %45 : tensor<32x32x128xf32>, tensor<32x128x32xf32>) outs(%46 : tensor<32x32x32xf32>) -> tensor<32x32x32xf32>
                %48 = linalg.fill ins(%cst_3 : f32) outs(%22 : tensor<32x32xf32>) -> tensor<32x32xf32>
                %49 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%47 : tensor<32x32x32xf32>) outs(%48 : tensor<32x32xf32>) {
                ^bb0(%in: f32, %out: f32):
                  %62 = arith.maximumf %in, %out : f32
                  linalg.yield %62 : f32
                } -> tensor<32x32xf32>
                %50 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%arg9, %49 : tensor<32x32xf32>, tensor<32x32xf32>) outs(%22 : tensor<32x32xf32>) {
                ^bb0(%in: f32, %in_10: f32, %out: f32):
                  %62 = arith.truncf %cst_0 : f64 to f32
                  %63 = arith.mulf %in_10, %62 : f32
                  %64 = arith.cmpf ogt, %in, %63 : f32
                  %65 = arith.select %64, %in, %63 : f32
                  linalg.yield %65 : f32
                } -> tensor<32x32xf32>
                %51 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%47, %50 : tensor<32x32x32xf32>, tensor<32x32xf32>) outs(%15 : tensor<32x32x32xf32>) {
                ^bb0(%in: f32, %in_10: f32, %out: f32):
                  %62 = arith.truncf %cst_0 : f64 to f32
                  %63 = arith.mulf %in, %62 : f32
                  %64 = arith.subf %63, %in_10 : f32
                  %65 = math.powf %cst, %64 : f32
                  linalg.yield %65 : f32
                } -> tensor<32x32x32xf32>
                %52 = linalg.fill ins(%cst_1 : f32) outs(%24 : tensor<32x32xf32>) -> tensor<32x32xf32>
                %53 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%51 : tensor<32x32x32xf32>) outs(%52 : tensor<32x32xf32>) {
                ^bb0(%in: f32, %out: f32):
                  %62 = arith.addf %in, %out : f32
                  linalg.yield %62 : f32
                } -> tensor<32x32xf32>
                %54 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%arg9, %50 : tensor<32x32xf32>, tensor<32x32xf32>) outs(%26 : tensor<32x32xf32>) {
                ^bb0(%in: f32, %in_10: f32, %out: f32):
                  %62 = arith.subf %in, %in_10 : f32
                  %63 = math.powf %cst, %62 : f32
                  linalg.yield %63 : f32
                } -> tensor<32x32xf32>
                %55 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%arg10, %54, %53 : tensor<32x32xf32>, tensor<32x32xf32>, tensor<32x32xf32>) outs(%arg10 : tensor<32x32xf32>) {
                ^bb0(%in: f32, %in_10: f32, %in_11: f32, %out: f32):
                  %62 = arith.mulf %in, %in_10 : f32
                  %63 = arith.addf %62, %in_11 : f32
                  linalg.yield %63 : f32
                } -> tensor<32x32xf32>
                %c0_8 = arith.constant 0 : index
                %56 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 524288 + d1 * 128 + d2)>(%33, %43, %c0_8)
                %reinterpret_cast_9 = memref.reinterpret_cast %arg1 to offset: [%56], sizes: [32, 32, 128], strides: [524288, 128, 1] : memref<32x4096x128xf32> to memref<32x32x128xf32, strided<[524288, 128, 1], offset: ?>>
                %57 = loom.copy_to_tensor %reinterpret_cast_9, %13 on @L1, interconnect : [@horizontal_links], broadcast : [1, 8] : memref<32x32x128xf32, strided<[524288, 128, 1], offset: ?>>, memref<32x32x128xf32> -> tensor<32x32x128xf32>
                %58 = linalg.fill ins(%cst_1 : f32) outs(%32 : tensor<32x32x128xf32>) -> tensor<32x32x128xf32>
                %59 = linalg.batch_matmul ins(%51, %57 : tensor<32x32x32xf32>, tensor<32x32x128xf32>) outs(%58 : tensor<32x32x128xf32>) -> tensor<32x32x128xf32>
                %60 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%59, %arg11, %54 : tensor<32x32x128xf32>, tensor<32x32x128xf32>, tensor<32x32xf32>) outs(%arg11 : tensor<32x32x128xf32>) {
                ^bb0(%in: f32, %in_10: f32, %in_11: f32, %out: f32):
                  %62 = arith.mulf %in_10, %in_11 : f32
                  %63 = arith.addf %in, %62 : f32
                  linalg.yield %63 : f32
                } -> tensor<32x32x128xf32>
                %61 = linalg.copy ins(%50 : tensor<32x32xf32>) outs(%arg9 : tensor<32x32xf32>) -> tensor<32x32xf32>
                affine.yield %61, %55, %60 : tensor<32x32xf32>, tensor<32x32xf32>, tensor<32x32x128xf32>
              }
              %41 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%40#2, %40#1 : tensor<32x32x128xf32>, tensor<32x32xf32>) outs(%28 : tensor<32x32x128xf32>) {
              ^bb0(%in: f32, %in_6: f32, %out: f32):
                %43 = arith.divf %in, %in_6 : f32
                linalg.yield %43 : f32
              } -> tensor<32x32x128xf32>
              %c0_4 = arith.constant 0 : index
              %42 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 524288 + d1 * 128 + d2)>(%33, %34, %c0_4)
              %reinterpret_cast_5 = memref.reinterpret_cast %arg3 to offset: [%42], sizes: [32, 32, 128], strides: [524288, 128, 1] : memref<32x4096x128xf32> to memref<32x32x128xf32, strided<[524288, 128, 1], offset: ?>>
              loom.copy_from_tensor %41, %reinterpret_cast_5 on @DRAM : tensor<32x32x128xf32>, memref<32x32x128xf32, strided<[524288, 128, 1], offset: ?>>
            } {loom.iter_type = #loom.iter_type<temporal>}
          } {loom.iter_type = #loom.iter_type<temporal>}
        } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @y}
      } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @x}
      return
    }
  }
  module attributes {loom.pass_name = "Materialize"} {
    func.func @attention__d0i1_d1i1__f10__d_a_v__BB32__BM32__BN32(%arg0: memref<32x128x4096xf32>, %arg1: memref<32x4096x128xf32>, %arg2: memref<32x4096x128xf32>, %arg3: memref<32x4096x128xf32>) {
      %cst = arith.constant 2.000000e+00 : f32
      %cst_0 = arith.constant 0.12751743074602467 : f64
      %cst_1 = arith.constant 0.000000e+00 : f32
      %cst_2 = arith.constant 1.000000e+00 : f32
      %cst_3 = arith.constant 0xFF800000 : f32
      %c32 = arith.constant 32 : index
      affine.parallel (%arg4) = (0) to (8) {
        affine.parallel (%arg5) = (0) to (8) {
          affine.for %arg6 = 0 to 2 {
            affine.for %arg7 = 0 to 1 {
              %12 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 8 + d1 + d2 * 64)>(%arg4, %arg5, %arg6)
              %13 = loom.alloc [32, 32, 128] on @L1 : memref<32x32x128xf32>
              %14 = loom.alloc [32, 32, 32] on @L1 : memref<32x32x32xf32>
              %15 = loom.init_tensor %14[32, 32, 32] : memref<32x32x32xf32> -> tensor<32x32x32xf32>
              %16 = loom.alloc [32, 128, 32] on @L1 : memref<32x128x32xf32>
              %17 = loom.alloc [32, 32] on @L1 : memref<32x32xf32>
              %18 = loom.init_tensor %17[32, 32] : memref<32x32xf32> -> tensor<32x32xf32>
              %19 = loom.alloc [32, 32] on @L1 : memref<32x32xf32>
              %20 = loom.init_tensor %19[32, 32] : memref<32x32xf32> -> tensor<32x32xf32>
              %21 = loom.alloc [32, 32] on @L1 : memref<32x32xf32>
              %22 = loom.init_tensor %21[32, 32] : memref<32x32xf32> -> tensor<32x32xf32>
              %23 = loom.alloc [32, 32] on @L1 : memref<32x32xf32>
              %24 = loom.init_tensor %23[32, 32] : memref<32x32xf32> -> tensor<32x32xf32>
              %25 = loom.alloc [32, 32] on @L1 : memref<32x32xf32>
              %26 = loom.init_tensor %25[32, 32] : memref<32x32xf32> -> tensor<32x32xf32>
              %27 = loom.alloc [32, 32, 128] on @L1 : memref<32x32x128xf32>
              %28 = loom.init_tensor %27[32, 32, 128] : memref<32x32x128xf32> -> tensor<32x32x128xf32>
              %29 = loom.alloc [32, 32, 128] on @L1 : memref<32x32x128xf32>
              %30 = loom.init_tensor %29[32, 32, 128] : memref<32x32x128xf32> -> tensor<32x32x128xf32>
              %31 = loom.alloc [32, 32, 128] on @L1 : memref<32x32x128xf32>
              %32 = loom.init_tensor %31[32, 32, 128] : memref<32x32x128xf32> -> tensor<32x32x128xf32>
              %33 = arith.muli %arg7, %c32 : index
              %34 = arith.muli %12, %c32 : index
              %c0 = arith.constant 0 : index
              %35 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 524288 + d1 * 128 + d2)>(%33, %34, %c0)
              %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%35], sizes: [32, 32, 128], strides: [524288, 128, 1] : memref<32x4096x128xf32> to memref<32x32x128xf32, strided<[524288, 128, 1], offset: ?>>
              %36 = loom.copy_to_tensor %reinterpret_cast, %27 on @L1, interconnect : [], broadcast : [1, 1] : memref<32x32x128xf32, strided<[524288, 128, 1], offset: ?>>, memref<32x32x128xf32> -> tensor<32x32x128xf32>
              %37 = linalg.fill ins(%cst_1 : f32) outs(%30 : tensor<32x32x128xf32>) -> tensor<32x32x128xf32>
              %38 = linalg.fill ins(%cst_2 : f32) outs(%18 : tensor<32x32xf32>) -> tensor<32x32xf32>
              %39 = linalg.fill ins(%cst_3 : f32) outs(%20 : tensor<32x32xf32>) -> tensor<32x32xf32>
              %40:3 = affine.for %arg8 = 0 to 128 iter_args(%arg9 = %39, %arg10 = %38, %arg11 = %37) -> (tensor<32x32xf32>, tensor<32x32xf32>, tensor<32x32x128xf32>) {
                %43 = arith.muli %arg8, %c32 : index
                %c0_6 = arith.constant 0 : index
                %44 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 524288 + d1 * 4096 + d2)>(%33, %c0_6, %43)
                %reinterpret_cast_7 = memref.reinterpret_cast %arg0 to offset: [%44], sizes: [32, 128, 32], strides: [524288, 4096, 1] : memref<32x128x4096xf32> to memref<32x128x32xf32, strided<[524288, 4096, 1], offset: ?>>
                %45 = loom.copy_to_tensor %reinterpret_cast_7, %16 on @L1, interconnect : [@horizontal_links, @vertical_links], broadcast : [8, 8] : memref<32x128x32xf32, strided<[524288, 4096, 1], offset: ?>>, memref<32x128x32xf32> -> tensor<32x128x32xf32>
                %46 = linalg.fill ins(%cst_1 : f32) outs(%15 : tensor<32x32x32xf32>) -> tensor<32x32x32xf32>
                %47 = linalg.batch_matmul ins(%36, %45 : tensor<32x32x128xf32>, tensor<32x128x32xf32>) outs(%46 : tensor<32x32x32xf32>) -> tensor<32x32x32xf32>
                %48 = linalg.fill ins(%cst_3 : f32) outs(%22 : tensor<32x32xf32>) -> tensor<32x32xf32>
                %49 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%47 : tensor<32x32x32xf32>) outs(%48 : tensor<32x32xf32>) {
                ^bb0(%in: f32, %out: f32):
                  %62 = arith.maximumf %in, %out : f32
                  linalg.yield %62 : f32
                } -> tensor<32x32xf32>
                %50 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%arg9, %49 : tensor<32x32xf32>, tensor<32x32xf32>) outs(%22 : tensor<32x32xf32>) {
                ^bb0(%in: f32, %in_10: f32, %out: f32):
                  %62 = arith.truncf %cst_0 : f64 to f32
                  %63 = arith.mulf %in_10, %62 : f32
                  %64 = arith.cmpf ogt, %in, %63 : f32
                  %65 = arith.select %64, %in, %63 : f32
                  linalg.yield %65 : f32
                } -> tensor<32x32xf32>
                %51 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%47, %50 : tensor<32x32x32xf32>, tensor<32x32xf32>) outs(%15 : tensor<32x32x32xf32>) {
                ^bb0(%in: f32, %in_10: f32, %out: f32):
                  %62 = arith.truncf %cst_0 : f64 to f32
                  %63 = arith.mulf %in, %62 : f32
                  %64 = arith.subf %63, %in_10 : f32
                  %65 = math.powf %cst, %64 : f32
                  linalg.yield %65 : f32
                } -> tensor<32x32x32xf32>
                %52 = linalg.fill ins(%cst_1 : f32) outs(%24 : tensor<32x32xf32>) -> tensor<32x32xf32>
                %53 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%51 : tensor<32x32x32xf32>) outs(%52 : tensor<32x32xf32>) {
                ^bb0(%in: f32, %out: f32):
                  %62 = arith.addf %in, %out : f32
                  linalg.yield %62 : f32
                } -> tensor<32x32xf32>
                %54 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%arg9, %50 : tensor<32x32xf32>, tensor<32x32xf32>) outs(%26 : tensor<32x32xf32>) {
                ^bb0(%in: f32, %in_10: f32, %out: f32):
                  %62 = arith.subf %in, %in_10 : f32
                  %63 = math.powf %cst, %62 : f32
                  linalg.yield %63 : f32
                } -> tensor<32x32xf32>
                %55 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%arg10, %54, %53 : tensor<32x32xf32>, tensor<32x32xf32>, tensor<32x32xf32>) outs(%arg10 : tensor<32x32xf32>) {
                ^bb0(%in: f32, %in_10: f32, %in_11: f32, %out: f32):
                  %62 = arith.mulf %in, %in_10 : f32
                  %63 = arith.addf %62, %in_11 : f32
                  linalg.yield %63 : f32
                } -> tensor<32x32xf32>
                %c0_8 = arith.constant 0 : index
                %56 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 524288 + d1 * 128 + d2)>(%33, %43, %c0_8)
                %reinterpret_cast_9 = memref.reinterpret_cast %arg1 to offset: [%56], sizes: [32, 32, 128], strides: [524288, 128, 1] : memref<32x4096x128xf32> to memref<32x32x128xf32, strided<[524288, 128, 1], offset: ?>>
                %57 = loom.copy_to_tensor %reinterpret_cast_9, %13 on @L1, interconnect : [@vertical_links], broadcast : [8, 1] : memref<32x32x128xf32, strided<[524288, 128, 1], offset: ?>>, memref<32x32x128xf32> -> tensor<32x32x128xf32>
                %58 = linalg.fill ins(%cst_1 : f32) outs(%32 : tensor<32x32x128xf32>) -> tensor<32x32x128xf32>
                %59 = linalg.batch_matmul ins(%51, %57 : tensor<32x32x32xf32>, tensor<32x32x128xf32>) outs(%58 : tensor<32x32x128xf32>) -> tensor<32x32x128xf32>
                %60 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%59, %arg11, %54 : tensor<32x32x128xf32>, tensor<32x32x128xf32>, tensor<32x32xf32>) outs(%arg11 : tensor<32x32x128xf32>) {
                ^bb0(%in: f32, %in_10: f32, %in_11: f32, %out: f32):
                  %62 = arith.mulf %in_10, %in_11 : f32
                  %63 = arith.addf %in, %62 : f32
                  linalg.yield %63 : f32
                } -> tensor<32x32x128xf32>
                %61 = linalg.copy ins(%50 : tensor<32x32xf32>) outs(%arg9 : tensor<32x32xf32>) -> tensor<32x32xf32>
                affine.yield %61, %55, %60 : tensor<32x32xf32>, tensor<32x32xf32>, tensor<32x32x128xf32>
              }
              %41 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%40#2, %40#1 : tensor<32x32x128xf32>, tensor<32x32xf32>) outs(%28 : tensor<32x32x128xf32>) {
              ^bb0(%in: f32, %in_6: f32, %out: f32):
                %43 = arith.divf %in, %in_6 : f32
                linalg.yield %43 : f32
              } -> tensor<32x32x128xf32>
              %c0_4 = arith.constant 0 : index
              %42 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 524288 + d1 * 128 + d2)>(%33, %34, %c0_4)
              %reinterpret_cast_5 = memref.reinterpret_cast %arg3 to offset: [%42], sizes: [32, 32, 128], strides: [524288, 128, 1] : memref<32x4096x128xf32> to memref<32x32x128xf32, strided<[524288, 128, 1], offset: ?>>
              loom.copy_from_tensor %41, %reinterpret_cast_5 on @DRAM : tensor<32x32x128xf32>, memref<32x32x128xf32, strided<[524288, 128, 1], offset: ?>>
            } {loom.iter_type = #loom.iter_type<temporal>}
          } {loom.iter_type = #loom.iter_type<temporal>}
        } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @y}
      } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @x}
      return
    }
  }
  module attributes {loom.pass_name = "Materialize"} {
    func.func @attention__d0i1_d1i1__f10__d_h_d__BB32__BM32__BN32(%arg0: memref<32x128x4096xf32>, %arg1: memref<32x4096x128xf32>, %arg2: memref<32x4096x128xf32>, %arg3: memref<32x4096x128xf32>) {
      %cst = arith.constant 2.000000e+00 : f32
      %cst_0 = arith.constant 0.12751743074602467 : f64
      %cst_1 = arith.constant 0.000000e+00 : f32
      %cst_2 = arith.constant 1.000000e+00 : f32
      %cst_3 = arith.constant 0xFF800000 : f32
      %c32 = arith.constant 32 : index
      affine.parallel (%arg4) = (0) to (8) {
        affine.parallel (%arg5) = (0) to (8) {
          affine.for %arg6 = 0 to 2 {
            affine.for %arg7 = 0 to 1 {
              %12 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 8 + d1 + d2 * 64)>(%arg4, %arg5, %arg6)
              %13 = loom.alloc [32, 32, 128] on @L1 : memref<32x32x128xf32>
              %14 = loom.alloc [32, 32, 32] on @L1 : memref<32x32x32xf32>
              %15 = loom.init_tensor %14[32, 32, 32] : memref<32x32x32xf32> -> tensor<32x32x32xf32>
              %16 = loom.alloc [32, 128, 32] on @L1 : memref<32x128x32xf32>
              %17 = loom.alloc [32, 32] on @L1 : memref<32x32xf32>
              %18 = loom.init_tensor %17[32, 32] : memref<32x32xf32> -> tensor<32x32xf32>
              %19 = loom.alloc [32, 32] on @L1 : memref<32x32xf32>
              %20 = loom.init_tensor %19[32, 32] : memref<32x32xf32> -> tensor<32x32xf32>
              %21 = loom.alloc [32, 32] on @L1 : memref<32x32xf32>
              %22 = loom.init_tensor %21[32, 32] : memref<32x32xf32> -> tensor<32x32xf32>
              %23 = loom.alloc [32, 32] on @L1 : memref<32x32xf32>
              %24 = loom.init_tensor %23[32, 32] : memref<32x32xf32> -> tensor<32x32xf32>
              %25 = loom.alloc [32, 32] on @L1 : memref<32x32xf32>
              %26 = loom.init_tensor %25[32, 32] : memref<32x32xf32> -> tensor<32x32xf32>
              %27 = loom.alloc [32, 32, 128] on @L1 : memref<32x32x128xf32>
              %28 = loom.init_tensor %27[32, 32, 128] : memref<32x32x128xf32> -> tensor<32x32x128xf32>
              %29 = loom.alloc [32, 32, 128] on @L1 : memref<32x32x128xf32>
              %30 = loom.init_tensor %29[32, 32, 128] : memref<32x32x128xf32> -> tensor<32x32x128xf32>
              %31 = loom.alloc [32, 32, 128] on @L1 : memref<32x32x128xf32>
              %32 = loom.init_tensor %31[32, 32, 128] : memref<32x32x128xf32> -> tensor<32x32x128xf32>
              %33 = arith.muli %arg7, %c32 : index
              %34 = arith.muli %12, %c32 : index
              %c0 = arith.constant 0 : index
              %35 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 524288 + d1 * 128 + d2)>(%33, %34, %c0)
              %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%35], sizes: [32, 32, 128], strides: [524288, 128, 1] : memref<32x4096x128xf32> to memref<32x32x128xf32, strided<[524288, 128, 1], offset: ?>>
              %36 = loom.copy_to_tensor %reinterpret_cast, %27 on @L1, interconnect : [], broadcast : [1, 1] : memref<32x32x128xf32, strided<[524288, 128, 1], offset: ?>>, memref<32x32x128xf32> -> tensor<32x32x128xf32>
              %37 = linalg.fill ins(%cst_1 : f32) outs(%30 : tensor<32x32x128xf32>) -> tensor<32x32x128xf32>
              %38 = linalg.fill ins(%cst_2 : f32) outs(%18 : tensor<32x32xf32>) -> tensor<32x32xf32>
              %39 = linalg.fill ins(%cst_3 : f32) outs(%20 : tensor<32x32xf32>) -> tensor<32x32xf32>
              %40:3 = affine.for %arg8 = 0 to 128 iter_args(%arg9 = %39, %arg10 = %38, %arg11 = %37) -> (tensor<32x32xf32>, tensor<32x32xf32>, tensor<32x32x128xf32>) {
                %43 = arith.muli %arg8, %c32 : index
                %c0_6 = arith.constant 0 : index
                %44 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 524288 + d1 * 4096 + d2)>(%33, %c0_6, %43)
                %reinterpret_cast_7 = memref.reinterpret_cast %arg0 to offset: [%44], sizes: [32, 128, 32], strides: [524288, 4096, 1] : memref<32x128x4096xf32> to memref<32x128x32xf32, strided<[524288, 4096, 1], offset: ?>>
                %45 = loom.copy_to_tensor %reinterpret_cast_7, %16 on @L1, interconnect : [@horizontal_links], broadcast : [1, 8] : memref<32x128x32xf32, strided<[524288, 4096, 1], offset: ?>>, memref<32x128x32xf32> -> tensor<32x128x32xf32>
                %46 = linalg.fill ins(%cst_1 : f32) outs(%15 : tensor<32x32x32xf32>) -> tensor<32x32x32xf32>
                %47 = linalg.batch_matmul ins(%36, %45 : tensor<32x32x128xf32>, tensor<32x128x32xf32>) outs(%46 : tensor<32x32x32xf32>) -> tensor<32x32x32xf32>
                %48 = linalg.fill ins(%cst_3 : f32) outs(%22 : tensor<32x32xf32>) -> tensor<32x32xf32>
                %49 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%47 : tensor<32x32x32xf32>) outs(%48 : tensor<32x32xf32>) {
                ^bb0(%in: f32, %out: f32):
                  %62 = arith.maximumf %in, %out : f32
                  linalg.yield %62 : f32
                } -> tensor<32x32xf32>
                %50 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%arg9, %49 : tensor<32x32xf32>, tensor<32x32xf32>) outs(%22 : tensor<32x32xf32>) {
                ^bb0(%in: f32, %in_10: f32, %out: f32):
                  %62 = arith.truncf %cst_0 : f64 to f32
                  %63 = arith.mulf %in_10, %62 : f32
                  %64 = arith.cmpf ogt, %in, %63 : f32
                  %65 = arith.select %64, %in, %63 : f32
                  linalg.yield %65 : f32
                } -> tensor<32x32xf32>
                %51 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%47, %50 : tensor<32x32x32xf32>, tensor<32x32xf32>) outs(%15 : tensor<32x32x32xf32>) {
                ^bb0(%in: f32, %in_10: f32, %out: f32):
                  %62 = arith.truncf %cst_0 : f64 to f32
                  %63 = arith.mulf %in, %62 : f32
                  %64 = arith.subf %63, %in_10 : f32
                  %65 = math.powf %cst, %64 : f32
                  linalg.yield %65 : f32
                } -> tensor<32x32x32xf32>
                %52 = linalg.fill ins(%cst_1 : f32) outs(%24 : tensor<32x32xf32>) -> tensor<32x32xf32>
                %53 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%51 : tensor<32x32x32xf32>) outs(%52 : tensor<32x32xf32>) {
                ^bb0(%in: f32, %out: f32):
                  %62 = arith.addf %in, %out : f32
                  linalg.yield %62 : f32
                } -> tensor<32x32xf32>
                %54 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%arg9, %50 : tensor<32x32xf32>, tensor<32x32xf32>) outs(%26 : tensor<32x32xf32>) {
                ^bb0(%in: f32, %in_10: f32, %out: f32):
                  %62 = arith.subf %in, %in_10 : f32
                  %63 = math.powf %cst, %62 : f32
                  linalg.yield %63 : f32
                } -> tensor<32x32xf32>
                %55 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%arg10, %54, %53 : tensor<32x32xf32>, tensor<32x32xf32>, tensor<32x32xf32>) outs(%arg10 : tensor<32x32xf32>) {
                ^bb0(%in: f32, %in_10: f32, %in_11: f32, %out: f32):
                  %62 = arith.mulf %in, %in_10 : f32
                  %63 = arith.addf %62, %in_11 : f32
                  linalg.yield %63 : f32
                } -> tensor<32x32xf32>
                %c0_8 = arith.constant 0 : index
                %56 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 524288 + d1 * 128 + d2)>(%33, %43, %c0_8)
                %reinterpret_cast_9 = memref.reinterpret_cast %arg1 to offset: [%56], sizes: [32, 32, 128], strides: [524288, 128, 1] : memref<32x4096x128xf32> to memref<32x32x128xf32, strided<[524288, 128, 1], offset: ?>>
                %57 = loom.copy_to_tensor %reinterpret_cast_9, %13 on @L1, interconnect : [], broadcast : [1, 1] : memref<32x32x128xf32, strided<[524288, 128, 1], offset: ?>>, memref<32x32x128xf32> -> tensor<32x32x128xf32>
                %58 = linalg.fill ins(%cst_1 : f32) outs(%32 : tensor<32x32x128xf32>) -> tensor<32x32x128xf32>
                %59 = linalg.batch_matmul ins(%51, %57 : tensor<32x32x32xf32>, tensor<32x32x128xf32>) outs(%58 : tensor<32x32x128xf32>) -> tensor<32x32x128xf32>
                %60 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%59, %arg11, %54 : tensor<32x32x128xf32>, tensor<32x32x128xf32>, tensor<32x32xf32>) outs(%arg11 : tensor<32x32x128xf32>) {
                ^bb0(%in: f32, %in_10: f32, %in_11: f32, %out: f32):
                  %62 = arith.mulf %in_10, %in_11 : f32
                  %63 = arith.addf %in, %62 : f32
                  linalg.yield %63 : f32
                } -> tensor<32x32x128xf32>
                %61 = linalg.copy ins(%50 : tensor<32x32xf32>) outs(%arg9 : tensor<32x32xf32>) -> tensor<32x32xf32>
                affine.yield %61, %55, %60 : tensor<32x32xf32>, tensor<32x32xf32>, tensor<32x32x128xf32>
              }
              %41 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%40#2, %40#1 : tensor<32x32x128xf32>, tensor<32x32xf32>) outs(%28 : tensor<32x32x128xf32>) {
              ^bb0(%in: f32, %in_6: f32, %out: f32):
                %43 = arith.divf %in, %in_6 : f32
                linalg.yield %43 : f32
              } -> tensor<32x32x128xf32>
              %c0_4 = arith.constant 0 : index
              %42 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 524288 + d1 * 128 + d2)>(%33, %34, %c0_4)
              %reinterpret_cast_5 = memref.reinterpret_cast %arg3 to offset: [%42], sizes: [32, 32, 128], strides: [524288, 128, 1] : memref<32x4096x128xf32> to memref<32x32x128xf32, strided<[524288, 128, 1], offset: ?>>
              loom.copy_from_tensor %41, %reinterpret_cast_5 on @DRAM : tensor<32x32x128xf32>, memref<32x32x128xf32, strided<[524288, 128, 1], offset: ?>>
            } {loom.iter_type = #loom.iter_type<temporal>}
          } {loom.iter_type = #loom.iter_type<temporal>}
        } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @y}
      } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @x}
      return
    }
  }
  module attributes {loom.pass_name = "Materialize"} {
    func.func @attention__d0i1_d1i1__f10__d_h_a__BB32__BM32__BN32(%arg0: memref<32x128x4096xf32>, %arg1: memref<32x4096x128xf32>, %arg2: memref<32x4096x128xf32>, %arg3: memref<32x4096x128xf32>) {
      %cst = arith.constant 2.000000e+00 : f32
      %cst_0 = arith.constant 0.12751743074602467 : f64
      %cst_1 = arith.constant 0.000000e+00 : f32
      %cst_2 = arith.constant 1.000000e+00 : f32
      %cst_3 = arith.constant 0xFF800000 : f32
      %c32 = arith.constant 32 : index
      affine.parallel (%arg4) = (0) to (8) {
        affine.parallel (%arg5) = (0) to (8) {
          affine.for %arg6 = 0 to 2 {
            affine.for %arg7 = 0 to 1 {
              %12 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 8 + d1 + d2 * 64)>(%arg4, %arg5, %arg6)
              %13 = loom.alloc [32, 32, 128] on @L1 : memref<32x32x128xf32>
              %14 = loom.alloc [32, 32, 32] on @L1 : memref<32x32x32xf32>
              %15 = loom.init_tensor %14[32, 32, 32] : memref<32x32x32xf32> -> tensor<32x32x32xf32>
              %16 = loom.alloc [32, 128, 32] on @L1 : memref<32x128x32xf32>
              %17 = loom.alloc [32, 32] on @L1 : memref<32x32xf32>
              %18 = loom.init_tensor %17[32, 32] : memref<32x32xf32> -> tensor<32x32xf32>
              %19 = loom.alloc [32, 32] on @L1 : memref<32x32xf32>
              %20 = loom.init_tensor %19[32, 32] : memref<32x32xf32> -> tensor<32x32xf32>
              %21 = loom.alloc [32, 32] on @L1 : memref<32x32xf32>
              %22 = loom.init_tensor %21[32, 32] : memref<32x32xf32> -> tensor<32x32xf32>
              %23 = loom.alloc [32, 32] on @L1 : memref<32x32xf32>
              %24 = loom.init_tensor %23[32, 32] : memref<32x32xf32> -> tensor<32x32xf32>
              %25 = loom.alloc [32, 32] on @L1 : memref<32x32xf32>
              %26 = loom.init_tensor %25[32, 32] : memref<32x32xf32> -> tensor<32x32xf32>
              %27 = loom.alloc [32, 32, 128] on @L1 : memref<32x32x128xf32>
              %28 = loom.init_tensor %27[32, 32, 128] : memref<32x32x128xf32> -> tensor<32x32x128xf32>
              %29 = loom.alloc [32, 32, 128] on @L1 : memref<32x32x128xf32>
              %30 = loom.init_tensor %29[32, 32, 128] : memref<32x32x128xf32> -> tensor<32x32x128xf32>
              %31 = loom.alloc [32, 32, 128] on @L1 : memref<32x32x128xf32>
              %32 = loom.init_tensor %31[32, 32, 128] : memref<32x32x128xf32> -> tensor<32x32x128xf32>
              %33 = arith.muli %arg7, %c32 : index
              %34 = arith.muli %12, %c32 : index
              %c0 = arith.constant 0 : index
              %35 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 524288 + d1 * 128 + d2)>(%33, %34, %c0)
              %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%35], sizes: [32, 32, 128], strides: [524288, 128, 1] : memref<32x4096x128xf32> to memref<32x32x128xf32, strided<[524288, 128, 1], offset: ?>>
              %36 = loom.copy_to_tensor %reinterpret_cast, %27 on @L1, interconnect : [], broadcast : [1, 1] : memref<32x32x128xf32, strided<[524288, 128, 1], offset: ?>>, memref<32x32x128xf32> -> tensor<32x32x128xf32>
              %37 = linalg.fill ins(%cst_1 : f32) outs(%30 : tensor<32x32x128xf32>) -> tensor<32x32x128xf32>
              %38 = linalg.fill ins(%cst_2 : f32) outs(%18 : tensor<32x32xf32>) -> tensor<32x32xf32>
              %39 = linalg.fill ins(%cst_3 : f32) outs(%20 : tensor<32x32xf32>) -> tensor<32x32xf32>
              %40:3 = affine.for %arg8 = 0 to 128 iter_args(%arg9 = %39, %arg10 = %38, %arg11 = %37) -> (tensor<32x32xf32>, tensor<32x32xf32>, tensor<32x32x128xf32>) {
                %43 = arith.muli %arg8, %c32 : index
                %c0_6 = arith.constant 0 : index
                %44 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 524288 + d1 * 4096 + d2)>(%33, %c0_6, %43)
                %reinterpret_cast_7 = memref.reinterpret_cast %arg0 to offset: [%44], sizes: [32, 128, 32], strides: [524288, 4096, 1] : memref<32x128x4096xf32> to memref<32x128x32xf32, strided<[524288, 4096, 1], offset: ?>>
                %45 = loom.copy_to_tensor %reinterpret_cast_7, %16 on @L1, interconnect : [@horizontal_links], broadcast : [1, 8] : memref<32x128x32xf32, strided<[524288, 4096, 1], offset: ?>>, memref<32x128x32xf32> -> tensor<32x128x32xf32>
                %46 = linalg.fill ins(%cst_1 : f32) outs(%15 : tensor<32x32x32xf32>) -> tensor<32x32x32xf32>
                %47 = linalg.batch_matmul ins(%36, %45 : tensor<32x32x128xf32>, tensor<32x128x32xf32>) outs(%46 : tensor<32x32x32xf32>) -> tensor<32x32x32xf32>
                %48 = linalg.fill ins(%cst_3 : f32) outs(%22 : tensor<32x32xf32>) -> tensor<32x32xf32>
                %49 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%47 : tensor<32x32x32xf32>) outs(%48 : tensor<32x32xf32>) {
                ^bb0(%in: f32, %out: f32):
                  %62 = arith.maximumf %in, %out : f32
                  linalg.yield %62 : f32
                } -> tensor<32x32xf32>
                %50 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%arg9, %49 : tensor<32x32xf32>, tensor<32x32xf32>) outs(%22 : tensor<32x32xf32>) {
                ^bb0(%in: f32, %in_10: f32, %out: f32):
                  %62 = arith.truncf %cst_0 : f64 to f32
                  %63 = arith.mulf %in_10, %62 : f32
                  %64 = arith.cmpf ogt, %in, %63 : f32
                  %65 = arith.select %64, %in, %63 : f32
                  linalg.yield %65 : f32
                } -> tensor<32x32xf32>
                %51 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%47, %50 : tensor<32x32x32xf32>, tensor<32x32xf32>) outs(%15 : tensor<32x32x32xf32>) {
                ^bb0(%in: f32, %in_10: f32, %out: f32):
                  %62 = arith.truncf %cst_0 : f64 to f32
                  %63 = arith.mulf %in, %62 : f32
                  %64 = arith.subf %63, %in_10 : f32
                  %65 = math.powf %cst, %64 : f32
                  linalg.yield %65 : f32
                } -> tensor<32x32x32xf32>
                %52 = linalg.fill ins(%cst_1 : f32) outs(%24 : tensor<32x32xf32>) -> tensor<32x32xf32>
                %53 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%51 : tensor<32x32x32xf32>) outs(%52 : tensor<32x32xf32>) {
                ^bb0(%in: f32, %out: f32):
                  %62 = arith.addf %in, %out : f32
                  linalg.yield %62 : f32
                } -> tensor<32x32xf32>
                %54 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%arg9, %50 : tensor<32x32xf32>, tensor<32x32xf32>) outs(%26 : tensor<32x32xf32>) {
                ^bb0(%in: f32, %in_10: f32, %out: f32):
                  %62 = arith.subf %in, %in_10 : f32
                  %63 = math.powf %cst, %62 : f32
                  linalg.yield %63 : f32
                } -> tensor<32x32xf32>
                %55 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%arg10, %54, %53 : tensor<32x32xf32>, tensor<32x32xf32>, tensor<32x32xf32>) outs(%arg10 : tensor<32x32xf32>) {
                ^bb0(%in: f32, %in_10: f32, %in_11: f32, %out: f32):
                  %62 = arith.mulf %in, %in_10 : f32
                  %63 = arith.addf %62, %in_11 : f32
                  linalg.yield %63 : f32
                } -> tensor<32x32xf32>
                %c0_8 = arith.constant 0 : index
                %56 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 524288 + d1 * 128 + d2)>(%33, %43, %c0_8)
                %reinterpret_cast_9 = memref.reinterpret_cast %arg1 to offset: [%56], sizes: [32, 32, 128], strides: [524288, 128, 1] : memref<32x4096x128xf32> to memref<32x32x128xf32, strided<[524288, 128, 1], offset: ?>>
                %57 = loom.copy_to_tensor %reinterpret_cast_9, %13 on @L1, interconnect : [@horizontal_links, @vertical_links], broadcast : [8, 8] : memref<32x32x128xf32, strided<[524288, 128, 1], offset: ?>>, memref<32x32x128xf32> -> tensor<32x32x128xf32>
                %58 = linalg.fill ins(%cst_1 : f32) outs(%32 : tensor<32x32x128xf32>) -> tensor<32x32x128xf32>
                %59 = linalg.batch_matmul ins(%51, %57 : tensor<32x32x32xf32>, tensor<32x32x128xf32>) outs(%58 : tensor<32x32x128xf32>) -> tensor<32x32x128xf32>
                %60 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%59, %arg11, %54 : tensor<32x32x128xf32>, tensor<32x32x128xf32>, tensor<32x32xf32>) outs(%arg11 : tensor<32x32x128xf32>) {
                ^bb0(%in: f32, %in_10: f32, %in_11: f32, %out: f32):
                  %62 = arith.mulf %in_10, %in_11 : f32
                  %63 = arith.addf %in, %62 : f32
                  linalg.yield %63 : f32
                } -> tensor<32x32x128xf32>
                %61 = linalg.copy ins(%50 : tensor<32x32xf32>) outs(%arg9 : tensor<32x32xf32>) -> tensor<32x32xf32>
                affine.yield %61, %55, %60 : tensor<32x32xf32>, tensor<32x32xf32>, tensor<32x32x128xf32>
              }
              %41 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%40#2, %40#1 : tensor<32x32x128xf32>, tensor<32x32xf32>) outs(%28 : tensor<32x32x128xf32>) {
              ^bb0(%in: f32, %in_6: f32, %out: f32):
                %43 = arith.divf %in, %in_6 : f32
                linalg.yield %43 : f32
              } -> tensor<32x32x128xf32>
              %c0_4 = arith.constant 0 : index
              %42 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 524288 + d1 * 128 + d2)>(%33, %34, %c0_4)
              %reinterpret_cast_5 = memref.reinterpret_cast %arg3 to offset: [%42], sizes: [32, 32, 128], strides: [524288, 128, 1] : memref<32x4096x128xf32> to memref<32x32x128xf32, strided<[524288, 128, 1], offset: ?>>
              loom.copy_from_tensor %41, %reinterpret_cast_5 on @DRAM : tensor<32x32x128xf32>, memref<32x32x128xf32, strided<[524288, 128, 1], offset: ?>>
            } {loom.iter_type = #loom.iter_type<temporal>}
          } {loom.iter_type = #loom.iter_type<temporal>}
        } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @y}
      } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @x}
      return
    }
  }
  module attributes {loom.pass_name = "Materialize"} {
    func.func @attention__d0i1_d1i1__f10__d_h_h__BB32__BM32__BN32(%arg0: memref<32x128x4096xf32>, %arg1: memref<32x4096x128xf32>, %arg2: memref<32x4096x128xf32>, %arg3: memref<32x4096x128xf32>) {
      %cst = arith.constant 2.000000e+00 : f32
      %cst_0 = arith.constant 0.12751743074602467 : f64
      %cst_1 = arith.constant 0.000000e+00 : f32
      %cst_2 = arith.constant 1.000000e+00 : f32
      %cst_3 = arith.constant 0xFF800000 : f32
      %c32 = arith.constant 32 : index
      affine.parallel (%arg4) = (0) to (8) {
        affine.parallel (%arg5) = (0) to (8) {
          affine.for %arg6 = 0 to 2 {
            affine.for %arg7 = 0 to 1 {
              %12 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 8 + d1 + d2 * 64)>(%arg4, %arg5, %arg6)
              %13 = loom.alloc [32, 32, 128] on @L1 : memref<32x32x128xf32>
              %14 = loom.alloc [32, 32, 32] on @L1 : memref<32x32x32xf32>
              %15 = loom.init_tensor %14[32, 32, 32] : memref<32x32x32xf32> -> tensor<32x32x32xf32>
              %16 = loom.alloc [32, 128, 32] on @L1 : memref<32x128x32xf32>
              %17 = loom.alloc [32, 32] on @L1 : memref<32x32xf32>
              %18 = loom.init_tensor %17[32, 32] : memref<32x32xf32> -> tensor<32x32xf32>
              %19 = loom.alloc [32, 32] on @L1 : memref<32x32xf32>
              %20 = loom.init_tensor %19[32, 32] : memref<32x32xf32> -> tensor<32x32xf32>
              %21 = loom.alloc [32, 32] on @L1 : memref<32x32xf32>
              %22 = loom.init_tensor %21[32, 32] : memref<32x32xf32> -> tensor<32x32xf32>
              %23 = loom.alloc [32, 32] on @L1 : memref<32x32xf32>
              %24 = loom.init_tensor %23[32, 32] : memref<32x32xf32> -> tensor<32x32xf32>
              %25 = loom.alloc [32, 32] on @L1 : memref<32x32xf32>
              %26 = loom.init_tensor %25[32, 32] : memref<32x32xf32> -> tensor<32x32xf32>
              %27 = loom.alloc [32, 32, 128] on @L1 : memref<32x32x128xf32>
              %28 = loom.init_tensor %27[32, 32, 128] : memref<32x32x128xf32> -> tensor<32x32x128xf32>
              %29 = loom.alloc [32, 32, 128] on @L1 : memref<32x32x128xf32>
              %30 = loom.init_tensor %29[32, 32, 128] : memref<32x32x128xf32> -> tensor<32x32x128xf32>
              %31 = loom.alloc [32, 32, 128] on @L1 : memref<32x32x128xf32>
              %32 = loom.init_tensor %31[32, 32, 128] : memref<32x32x128xf32> -> tensor<32x32x128xf32>
              %33 = arith.muli %arg7, %c32 : index
              %34 = arith.muli %12, %c32 : index
              %c0 = arith.constant 0 : index
              %35 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 524288 + d1 * 128 + d2)>(%33, %34, %c0)
              %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%35], sizes: [32, 32, 128], strides: [524288, 128, 1] : memref<32x4096x128xf32> to memref<32x32x128xf32, strided<[524288, 128, 1], offset: ?>>
              %36 = loom.copy_to_tensor %reinterpret_cast, %27 on @L1, interconnect : [], broadcast : [1, 1] : memref<32x32x128xf32, strided<[524288, 128, 1], offset: ?>>, memref<32x32x128xf32> -> tensor<32x32x128xf32>
              %37 = linalg.fill ins(%cst_1 : f32) outs(%30 : tensor<32x32x128xf32>) -> tensor<32x32x128xf32>
              %38 = linalg.fill ins(%cst_2 : f32) outs(%18 : tensor<32x32xf32>) -> tensor<32x32xf32>
              %39 = linalg.fill ins(%cst_3 : f32) outs(%20 : tensor<32x32xf32>) -> tensor<32x32xf32>
              %40:3 = affine.for %arg8 = 0 to 128 iter_args(%arg9 = %39, %arg10 = %38, %arg11 = %37) -> (tensor<32x32xf32>, tensor<32x32xf32>, tensor<32x32x128xf32>) {
                %43 = arith.muli %arg8, %c32 : index
                %c0_6 = arith.constant 0 : index
                %44 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 524288 + d1 * 4096 + d2)>(%33, %c0_6, %43)
                %reinterpret_cast_7 = memref.reinterpret_cast %arg0 to offset: [%44], sizes: [32, 128, 32], strides: [524288, 4096, 1] : memref<32x128x4096xf32> to memref<32x128x32xf32, strided<[524288, 4096, 1], offset: ?>>
                %45 = loom.copy_to_tensor %reinterpret_cast_7, %16 on @L1, interconnect : [@horizontal_links], broadcast : [1, 8] : memref<32x128x32xf32, strided<[524288, 4096, 1], offset: ?>>, memref<32x128x32xf32> -> tensor<32x128x32xf32>
                %46 = linalg.fill ins(%cst_1 : f32) outs(%15 : tensor<32x32x32xf32>) -> tensor<32x32x32xf32>
                %47 = linalg.batch_matmul ins(%36, %45 : tensor<32x32x128xf32>, tensor<32x128x32xf32>) outs(%46 : tensor<32x32x32xf32>) -> tensor<32x32x32xf32>
                %48 = linalg.fill ins(%cst_3 : f32) outs(%22 : tensor<32x32xf32>) -> tensor<32x32xf32>
                %49 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%47 : tensor<32x32x32xf32>) outs(%48 : tensor<32x32xf32>) {
                ^bb0(%in: f32, %out: f32):
                  %62 = arith.maximumf %in, %out : f32
                  linalg.yield %62 : f32
                } -> tensor<32x32xf32>
                %50 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%arg9, %49 : tensor<32x32xf32>, tensor<32x32xf32>) outs(%22 : tensor<32x32xf32>) {
                ^bb0(%in: f32, %in_10: f32, %out: f32):
                  %62 = arith.truncf %cst_0 : f64 to f32
                  %63 = arith.mulf %in_10, %62 : f32
                  %64 = arith.cmpf ogt, %in, %63 : f32
                  %65 = arith.select %64, %in, %63 : f32
                  linalg.yield %65 : f32
                } -> tensor<32x32xf32>
                %51 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%47, %50 : tensor<32x32x32xf32>, tensor<32x32xf32>) outs(%15 : tensor<32x32x32xf32>) {
                ^bb0(%in: f32, %in_10: f32, %out: f32):
                  %62 = arith.truncf %cst_0 : f64 to f32
                  %63 = arith.mulf %in, %62 : f32
                  %64 = arith.subf %63, %in_10 : f32
                  %65 = math.powf %cst, %64 : f32
                  linalg.yield %65 : f32
                } -> tensor<32x32x32xf32>
                %52 = linalg.fill ins(%cst_1 : f32) outs(%24 : tensor<32x32xf32>) -> tensor<32x32xf32>
                %53 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%51 : tensor<32x32x32xf32>) outs(%52 : tensor<32x32xf32>) {
                ^bb0(%in: f32, %out: f32):
                  %62 = arith.addf %in, %out : f32
                  linalg.yield %62 : f32
                } -> tensor<32x32xf32>
                %54 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%arg9, %50 : tensor<32x32xf32>, tensor<32x32xf32>) outs(%26 : tensor<32x32xf32>) {
                ^bb0(%in: f32, %in_10: f32, %out: f32):
                  %62 = arith.subf %in, %in_10 : f32
                  %63 = math.powf %cst, %62 : f32
                  linalg.yield %63 : f32
                } -> tensor<32x32xf32>
                %55 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%arg10, %54, %53 : tensor<32x32xf32>, tensor<32x32xf32>, tensor<32x32xf32>) outs(%arg10 : tensor<32x32xf32>) {
                ^bb0(%in: f32, %in_10: f32, %in_11: f32, %out: f32):
                  %62 = arith.mulf %in, %in_10 : f32
                  %63 = arith.addf %62, %in_11 : f32
                  linalg.yield %63 : f32
                } -> tensor<32x32xf32>
                %c0_8 = arith.constant 0 : index
                %56 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 524288 + d1 * 128 + d2)>(%33, %43, %c0_8)
                %reinterpret_cast_9 = memref.reinterpret_cast %arg1 to offset: [%56], sizes: [32, 32, 128], strides: [524288, 128, 1] : memref<32x4096x128xf32> to memref<32x32x128xf32, strided<[524288, 128, 1], offset: ?>>
                %57 = loom.copy_to_tensor %reinterpret_cast_9, %13 on @L1, interconnect : [@horizontal_links], broadcast : [1, 8] : memref<32x32x128xf32, strided<[524288, 128, 1], offset: ?>>, memref<32x32x128xf32> -> tensor<32x32x128xf32>
                %58 = linalg.fill ins(%cst_1 : f32) outs(%32 : tensor<32x32x128xf32>) -> tensor<32x32x128xf32>
                %59 = linalg.batch_matmul ins(%51, %57 : tensor<32x32x32xf32>, tensor<32x32x128xf32>) outs(%58 : tensor<32x32x128xf32>) -> tensor<32x32x128xf32>
                %60 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%59, %arg11, %54 : tensor<32x32x128xf32>, tensor<32x32x128xf32>, tensor<32x32xf32>) outs(%arg11 : tensor<32x32x128xf32>) {
                ^bb0(%in: f32, %in_10: f32, %in_11: f32, %out: f32):
                  %62 = arith.mulf %in_10, %in_11 : f32
                  %63 = arith.addf %in, %62 : f32
                  linalg.yield %63 : f32
                } -> tensor<32x32x128xf32>
                %61 = linalg.copy ins(%50 : tensor<32x32xf32>) outs(%arg9 : tensor<32x32xf32>) -> tensor<32x32xf32>
                affine.yield %61, %55, %60 : tensor<32x32xf32>, tensor<32x32xf32>, tensor<32x32x128xf32>
              }
              %41 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%40#2, %40#1 : tensor<32x32x128xf32>, tensor<32x32xf32>) outs(%28 : tensor<32x32x128xf32>) {
              ^bb0(%in: f32, %in_6: f32, %out: f32):
                %43 = arith.divf %in, %in_6 : f32
                linalg.yield %43 : f32
              } -> tensor<32x32x128xf32>
              %c0_4 = arith.constant 0 : index
              %42 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 524288 + d1 * 128 + d2)>(%33, %34, %c0_4)
              %reinterpret_cast_5 = memref.reinterpret_cast %arg3 to offset: [%42], sizes: [32, 32, 128], strides: [524288, 128, 1] : memref<32x4096x128xf32> to memref<32x32x128xf32, strided<[524288, 128, 1], offset: ?>>
              loom.copy_from_tensor %41, %reinterpret_cast_5 on @DRAM : tensor<32x32x128xf32>, memref<32x32x128xf32, strided<[524288, 128, 1], offset: ?>>
            } {loom.iter_type = #loom.iter_type<temporal>}
          } {loom.iter_type = #loom.iter_type<temporal>}
        } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @y}
      } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @x}
      return
    }
  }
  module attributes {loom.pass_name = "Materialize"} {
    func.func @attention__d0i1_d1i1__f10__d_h_v__BB32__BM32__BN32(%arg0: memref<32x128x4096xf32>, %arg1: memref<32x4096x128xf32>, %arg2: memref<32x4096x128xf32>, %arg3: memref<32x4096x128xf32>) {
      %cst = arith.constant 2.000000e+00 : f32
      %cst_0 = arith.constant 0.12751743074602467 : f64
      %cst_1 = arith.constant 0.000000e+00 : f32
      %cst_2 = arith.constant 1.000000e+00 : f32
      %cst_3 = arith.constant 0xFF800000 : f32
      %c32 = arith.constant 32 : index
      affine.parallel (%arg4) = (0) to (8) {
        affine.parallel (%arg5) = (0) to (8) {
          affine.for %arg6 = 0 to 2 {
            affine.for %arg7 = 0 to 1 {
              %12 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 8 + d1 + d2 * 64)>(%arg4, %arg5, %arg6)
              %13 = loom.alloc [32, 32, 128] on @L1 : memref<32x32x128xf32>
              %14 = loom.alloc [32, 32, 32] on @L1 : memref<32x32x32xf32>
              %15 = loom.init_tensor %14[32, 32, 32] : memref<32x32x32xf32> -> tensor<32x32x32xf32>
              %16 = loom.alloc [32, 128, 32] on @L1 : memref<32x128x32xf32>
              %17 = loom.alloc [32, 32] on @L1 : memref<32x32xf32>
              %18 = loom.init_tensor %17[32, 32] : memref<32x32xf32> -> tensor<32x32xf32>
              %19 = loom.alloc [32, 32] on @L1 : memref<32x32xf32>
              %20 = loom.init_tensor %19[32, 32] : memref<32x32xf32> -> tensor<32x32xf32>
              %21 = loom.alloc [32, 32] on @L1 : memref<32x32xf32>
              %22 = loom.init_tensor %21[32, 32] : memref<32x32xf32> -> tensor<32x32xf32>
              %23 = loom.alloc [32, 32] on @L1 : memref<32x32xf32>
              %24 = loom.init_tensor %23[32, 32] : memref<32x32xf32> -> tensor<32x32xf32>
              %25 = loom.alloc [32, 32] on @L1 : memref<32x32xf32>
              %26 = loom.init_tensor %25[32, 32] : memref<32x32xf32> -> tensor<32x32xf32>
              %27 = loom.alloc [32, 32, 128] on @L1 : memref<32x32x128xf32>
              %28 = loom.init_tensor %27[32, 32, 128] : memref<32x32x128xf32> -> tensor<32x32x128xf32>
              %29 = loom.alloc [32, 32, 128] on @L1 : memref<32x32x128xf32>
              %30 = loom.init_tensor %29[32, 32, 128] : memref<32x32x128xf32> -> tensor<32x32x128xf32>
              %31 = loom.alloc [32, 32, 128] on @L1 : memref<32x32x128xf32>
              %32 = loom.init_tensor %31[32, 32, 128] : memref<32x32x128xf32> -> tensor<32x32x128xf32>
              %33 = arith.muli %arg7, %c32 : index
              %34 = arith.muli %12, %c32 : index
              %c0 = arith.constant 0 : index
              %35 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 524288 + d1 * 128 + d2)>(%33, %34, %c0)
              %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%35], sizes: [32, 32, 128], strides: [524288, 128, 1] : memref<32x4096x128xf32> to memref<32x32x128xf32, strided<[524288, 128, 1], offset: ?>>
              %36 = loom.copy_to_tensor %reinterpret_cast, %27 on @L1, interconnect : [], broadcast : [1, 1] : memref<32x32x128xf32, strided<[524288, 128, 1], offset: ?>>, memref<32x32x128xf32> -> tensor<32x32x128xf32>
              %37 = linalg.fill ins(%cst_1 : f32) outs(%30 : tensor<32x32x128xf32>) -> tensor<32x32x128xf32>
              %38 = linalg.fill ins(%cst_2 : f32) outs(%18 : tensor<32x32xf32>) -> tensor<32x32xf32>
              %39 = linalg.fill ins(%cst_3 : f32) outs(%20 : tensor<32x32xf32>) -> tensor<32x32xf32>
              %40:3 = affine.for %arg8 = 0 to 128 iter_args(%arg9 = %39, %arg10 = %38, %arg11 = %37) -> (tensor<32x32xf32>, tensor<32x32xf32>, tensor<32x32x128xf32>) {
                %43 = arith.muli %arg8, %c32 : index
                %c0_6 = arith.constant 0 : index
                %44 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 524288 + d1 * 4096 + d2)>(%33, %c0_6, %43)
                %reinterpret_cast_7 = memref.reinterpret_cast %arg0 to offset: [%44], sizes: [32, 128, 32], strides: [524288, 4096, 1] : memref<32x128x4096xf32> to memref<32x128x32xf32, strided<[524288, 4096, 1], offset: ?>>
                %45 = loom.copy_to_tensor %reinterpret_cast_7, %16 on @L1, interconnect : [@horizontal_links], broadcast : [1, 8] : memref<32x128x32xf32, strided<[524288, 4096, 1], offset: ?>>, memref<32x128x32xf32> -> tensor<32x128x32xf32>
                %46 = linalg.fill ins(%cst_1 : f32) outs(%15 : tensor<32x32x32xf32>) -> tensor<32x32x32xf32>
                %47 = linalg.batch_matmul ins(%36, %45 : tensor<32x32x128xf32>, tensor<32x128x32xf32>) outs(%46 : tensor<32x32x32xf32>) -> tensor<32x32x32xf32>
                %48 = linalg.fill ins(%cst_3 : f32) outs(%22 : tensor<32x32xf32>) -> tensor<32x32xf32>
                %49 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%47 : tensor<32x32x32xf32>) outs(%48 : tensor<32x32xf32>) {
                ^bb0(%in: f32, %out: f32):
                  %62 = arith.maximumf %in, %out : f32
                  linalg.yield %62 : f32
                } -> tensor<32x32xf32>
                %50 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%arg9, %49 : tensor<32x32xf32>, tensor<32x32xf32>) outs(%22 : tensor<32x32xf32>) {
                ^bb0(%in: f32, %in_10: f32, %out: f32):
                  %62 = arith.truncf %cst_0 : f64 to f32
                  %63 = arith.mulf %in_10, %62 : f32
                  %64 = arith.cmpf ogt, %in, %63 : f32
                  %65 = arith.select %64, %in, %63 : f32
                  linalg.yield %65 : f32
                } -> tensor<32x32xf32>
                %51 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%47, %50 : tensor<32x32x32xf32>, tensor<32x32xf32>) outs(%15 : tensor<32x32x32xf32>) {
                ^bb0(%in: f32, %in_10: f32, %out: f32):
                  %62 = arith.truncf %cst_0 : f64 to f32
                  %63 = arith.mulf %in, %62 : f32
                  %64 = arith.subf %63, %in_10 : f32
                  %65 = math.powf %cst, %64 : f32
                  linalg.yield %65 : f32
                } -> tensor<32x32x32xf32>
                %52 = linalg.fill ins(%cst_1 : f32) outs(%24 : tensor<32x32xf32>) -> tensor<32x32xf32>
                %53 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%51 : tensor<32x32x32xf32>) outs(%52 : tensor<32x32xf32>) {
                ^bb0(%in: f32, %out: f32):
                  %62 = arith.addf %in, %out : f32
                  linalg.yield %62 : f32
                } -> tensor<32x32xf32>
                %54 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%arg9, %50 : tensor<32x32xf32>, tensor<32x32xf32>) outs(%26 : tensor<32x32xf32>) {
                ^bb0(%in: f32, %in_10: f32, %out: f32):
                  %62 = arith.subf %in, %in_10 : f32
                  %63 = math.powf %cst, %62 : f32
                  linalg.yield %63 : f32
                } -> tensor<32x32xf32>
                %55 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%arg10, %54, %53 : tensor<32x32xf32>, tensor<32x32xf32>, tensor<32x32xf32>) outs(%arg10 : tensor<32x32xf32>) {
                ^bb0(%in: f32, %in_10: f32, %in_11: f32, %out: f32):
                  %62 = arith.mulf %in, %in_10 : f32
                  %63 = arith.addf %62, %in_11 : f32
                  linalg.yield %63 : f32
                } -> tensor<32x32xf32>
                %c0_8 = arith.constant 0 : index
                %56 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 524288 + d1 * 128 + d2)>(%33, %43, %c0_8)
                %reinterpret_cast_9 = memref.reinterpret_cast %arg1 to offset: [%56], sizes: [32, 32, 128], strides: [524288, 128, 1] : memref<32x4096x128xf32> to memref<32x32x128xf32, strided<[524288, 128, 1], offset: ?>>
                %57 = loom.copy_to_tensor %reinterpret_cast_9, %13 on @L1, interconnect : [@vertical_links], broadcast : [8, 1] : memref<32x32x128xf32, strided<[524288, 128, 1], offset: ?>>, memref<32x32x128xf32> -> tensor<32x32x128xf32>
                %58 = linalg.fill ins(%cst_1 : f32) outs(%32 : tensor<32x32x128xf32>) -> tensor<32x32x128xf32>
                %59 = linalg.batch_matmul ins(%51, %57 : tensor<32x32x32xf32>, tensor<32x32x128xf32>) outs(%58 : tensor<32x32x128xf32>) -> tensor<32x32x128xf32>
                %60 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%59, %arg11, %54 : tensor<32x32x128xf32>, tensor<32x32x128xf32>, tensor<32x32xf32>) outs(%arg11 : tensor<32x32x128xf32>) {
                ^bb0(%in: f32, %in_10: f32, %in_11: f32, %out: f32):
                  %62 = arith.mulf %in_10, %in_11 : f32
                  %63 = arith.addf %in, %62 : f32
                  linalg.yield %63 : f32
                } -> tensor<32x32x128xf32>
                %61 = linalg.copy ins(%50 : tensor<32x32xf32>) outs(%arg9 : tensor<32x32xf32>) -> tensor<32x32xf32>
                affine.yield %61, %55, %60 : tensor<32x32xf32>, tensor<32x32xf32>, tensor<32x32x128xf32>
              }
              %41 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%40#2, %40#1 : tensor<32x32x128xf32>, tensor<32x32xf32>) outs(%28 : tensor<32x32x128xf32>) {
              ^bb0(%in: f32, %in_6: f32, %out: f32):
                %43 = arith.divf %in, %in_6 : f32
                linalg.yield %43 : f32
              } -> tensor<32x32x128xf32>
              %c0_4 = arith.constant 0 : index
              %42 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 524288 + d1 * 128 + d2)>(%33, %34, %c0_4)
              %reinterpret_cast_5 = memref.reinterpret_cast %arg3 to offset: [%42], sizes: [32, 32, 128], strides: [524288, 128, 1] : memref<32x4096x128xf32> to memref<32x32x128xf32, strided<[524288, 128, 1], offset: ?>>
              loom.copy_from_tensor %41, %reinterpret_cast_5 on @DRAM : tensor<32x32x128xf32>, memref<32x32x128xf32, strided<[524288, 128, 1], offset: ?>>
            } {loom.iter_type = #loom.iter_type<temporal>}
          } {loom.iter_type = #loom.iter_type<temporal>}
        } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @y}
      } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @x}
      return
    }
  }
  module attributes {loom.pass_name = "Materialize"} {
    func.func @attention__d0i1_d1i1__f10__d_v_d__BB32__BM32__BN32(%arg0: memref<32x128x4096xf32>, %arg1: memref<32x4096x128xf32>, %arg2: memref<32x4096x128xf32>, %arg3: memref<32x4096x128xf32>) {
      %cst = arith.constant 2.000000e+00 : f32
      %cst_0 = arith.constant 0.12751743074602467 : f64
      %cst_1 = arith.constant 0.000000e+00 : f32
      %cst_2 = arith.constant 1.000000e+00 : f32
      %cst_3 = arith.constant 0xFF800000 : f32
      %c32 = arith.constant 32 : index
      affine.parallel (%arg4) = (0) to (8) {
        affine.parallel (%arg5) = (0) to (8) {
          affine.for %arg6 = 0 to 2 {
            affine.for %arg7 = 0 to 1 {
              %12 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 8 + d1 + d2 * 64)>(%arg4, %arg5, %arg6)
              %13 = loom.alloc [32, 32, 128] on @L1 : memref<32x32x128xf32>
              %14 = loom.alloc [32, 32, 32] on @L1 : memref<32x32x32xf32>
              %15 = loom.init_tensor %14[32, 32, 32] : memref<32x32x32xf32> -> tensor<32x32x32xf32>
              %16 = loom.alloc [32, 128, 32] on @L1 : memref<32x128x32xf32>
              %17 = loom.alloc [32, 32] on @L1 : memref<32x32xf32>
              %18 = loom.init_tensor %17[32, 32] : memref<32x32xf32> -> tensor<32x32xf32>
              %19 = loom.alloc [32, 32] on @L1 : memref<32x32xf32>
              %20 = loom.init_tensor %19[32, 32] : memref<32x32xf32> -> tensor<32x32xf32>
              %21 = loom.alloc [32, 32] on @L1 : memref<32x32xf32>
              %22 = loom.init_tensor %21[32, 32] : memref<32x32xf32> -> tensor<32x32xf32>
              %23 = loom.alloc [32, 32] on @L1 : memref<32x32xf32>
              %24 = loom.init_tensor %23[32, 32] : memref<32x32xf32> -> tensor<32x32xf32>
              %25 = loom.alloc [32, 32] on @L1 : memref<32x32xf32>
              %26 = loom.init_tensor %25[32, 32] : memref<32x32xf32> -> tensor<32x32xf32>
              %27 = loom.alloc [32, 32, 128] on @L1 : memref<32x32x128xf32>
              %28 = loom.init_tensor %27[32, 32, 128] : memref<32x32x128xf32> -> tensor<32x32x128xf32>
              %29 = loom.alloc [32, 32, 128] on @L1 : memref<32x32x128xf32>
              %30 = loom.init_tensor %29[32, 32, 128] : memref<32x32x128xf32> -> tensor<32x32x128xf32>
              %31 = loom.alloc [32, 32, 128] on @L1 : memref<32x32x128xf32>
              %32 = loom.init_tensor %31[32, 32, 128] : memref<32x32x128xf32> -> tensor<32x32x128xf32>
              %33 = arith.muli %arg7, %c32 : index
              %34 = arith.muli %12, %c32 : index
              %c0 = arith.constant 0 : index
              %35 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 524288 + d1 * 128 + d2)>(%33, %34, %c0)
              %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%35], sizes: [32, 32, 128], strides: [524288, 128, 1] : memref<32x4096x128xf32> to memref<32x32x128xf32, strided<[524288, 128, 1], offset: ?>>
              %36 = loom.copy_to_tensor %reinterpret_cast, %27 on @L1, interconnect : [], broadcast : [1, 1] : memref<32x32x128xf32, strided<[524288, 128, 1], offset: ?>>, memref<32x32x128xf32> -> tensor<32x32x128xf32>
              %37 = linalg.fill ins(%cst_1 : f32) outs(%30 : tensor<32x32x128xf32>) -> tensor<32x32x128xf32>
              %38 = linalg.fill ins(%cst_2 : f32) outs(%18 : tensor<32x32xf32>) -> tensor<32x32xf32>
              %39 = linalg.fill ins(%cst_3 : f32) outs(%20 : tensor<32x32xf32>) -> tensor<32x32xf32>
              %40:3 = affine.for %arg8 = 0 to 128 iter_args(%arg9 = %39, %arg10 = %38, %arg11 = %37) -> (tensor<32x32xf32>, tensor<32x32xf32>, tensor<32x32x128xf32>) {
                %43 = arith.muli %arg8, %c32 : index
                %c0_6 = arith.constant 0 : index
                %44 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 524288 + d1 * 4096 + d2)>(%33, %c0_6, %43)
                %reinterpret_cast_7 = memref.reinterpret_cast %arg0 to offset: [%44], sizes: [32, 128, 32], strides: [524288, 4096, 1] : memref<32x128x4096xf32> to memref<32x128x32xf32, strided<[524288, 4096, 1], offset: ?>>
                %45 = loom.copy_to_tensor %reinterpret_cast_7, %16 on @L1, interconnect : [@vertical_links], broadcast : [8, 1] : memref<32x128x32xf32, strided<[524288, 4096, 1], offset: ?>>, memref<32x128x32xf32> -> tensor<32x128x32xf32>
                %46 = linalg.fill ins(%cst_1 : f32) outs(%15 : tensor<32x32x32xf32>) -> tensor<32x32x32xf32>
                %47 = linalg.batch_matmul ins(%36, %45 : tensor<32x32x128xf32>, tensor<32x128x32xf32>) outs(%46 : tensor<32x32x32xf32>) -> tensor<32x32x32xf32>
                %48 = linalg.fill ins(%cst_3 : f32) outs(%22 : tensor<32x32xf32>) -> tensor<32x32xf32>
                %49 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%47 : tensor<32x32x32xf32>) outs(%48 : tensor<32x32xf32>) {
                ^bb0(%in: f32, %out: f32):
                  %62 = arith.maximumf %in, %out : f32
                  linalg.yield %62 : f32
                } -> tensor<32x32xf32>
                %50 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%arg9, %49 : tensor<32x32xf32>, tensor<32x32xf32>) outs(%22 : tensor<32x32xf32>) {
                ^bb0(%in: f32, %in_10: f32, %out: f32):
                  %62 = arith.truncf %cst_0 : f64 to f32
                  %63 = arith.mulf %in_10, %62 : f32
                  %64 = arith.cmpf ogt, %in, %63 : f32
                  %65 = arith.select %64, %in, %63 : f32
                  linalg.yield %65 : f32
                } -> tensor<32x32xf32>
                %51 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%47, %50 : tensor<32x32x32xf32>, tensor<32x32xf32>) outs(%15 : tensor<32x32x32xf32>) {
                ^bb0(%in: f32, %in_10: f32, %out: f32):
                  %62 = arith.truncf %cst_0 : f64 to f32
                  %63 = arith.mulf %in, %62 : f32
                  %64 = arith.subf %63, %in_10 : f32
                  %65 = math.powf %cst, %64 : f32
                  linalg.yield %65 : f32
                } -> tensor<32x32x32xf32>
                %52 = linalg.fill ins(%cst_1 : f32) outs(%24 : tensor<32x32xf32>) -> tensor<32x32xf32>
                %53 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%51 : tensor<32x32x32xf32>) outs(%52 : tensor<32x32xf32>) {
                ^bb0(%in: f32, %out: f32):
                  %62 = arith.addf %in, %out : f32
                  linalg.yield %62 : f32
                } -> tensor<32x32xf32>
                %54 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%arg9, %50 : tensor<32x32xf32>, tensor<32x32xf32>) outs(%26 : tensor<32x32xf32>) {
                ^bb0(%in: f32, %in_10: f32, %out: f32):
                  %62 = arith.subf %in, %in_10 : f32
                  %63 = math.powf %cst, %62 : f32
                  linalg.yield %63 : f32
                } -> tensor<32x32xf32>
                %55 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%arg10, %54, %53 : tensor<32x32xf32>, tensor<32x32xf32>, tensor<32x32xf32>) outs(%arg10 : tensor<32x32xf32>) {
                ^bb0(%in: f32, %in_10: f32, %in_11: f32, %out: f32):
                  %62 = arith.mulf %in, %in_10 : f32
                  %63 = arith.addf %62, %in_11 : f32
                  linalg.yield %63 : f32
                } -> tensor<32x32xf32>
                %c0_8 = arith.constant 0 : index
                %56 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 524288 + d1 * 128 + d2)>(%33, %43, %c0_8)
                %reinterpret_cast_9 = memref.reinterpret_cast %arg1 to offset: [%56], sizes: [32, 32, 128], strides: [524288, 128, 1] : memref<32x4096x128xf32> to memref<32x32x128xf32, strided<[524288, 128, 1], offset: ?>>
                %57 = loom.copy_to_tensor %reinterpret_cast_9, %13 on @L1, interconnect : [], broadcast : [1, 1] : memref<32x32x128xf32, strided<[524288, 128, 1], offset: ?>>, memref<32x32x128xf32> -> tensor<32x32x128xf32>
                %58 = linalg.fill ins(%cst_1 : f32) outs(%32 : tensor<32x32x128xf32>) -> tensor<32x32x128xf32>
                %59 = linalg.batch_matmul ins(%51, %57 : tensor<32x32x32xf32>, tensor<32x32x128xf32>) outs(%58 : tensor<32x32x128xf32>) -> tensor<32x32x128xf32>
                %60 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%59, %arg11, %54 : tensor<32x32x128xf32>, tensor<32x32x128xf32>, tensor<32x32xf32>) outs(%arg11 : tensor<32x32x128xf32>) {
                ^bb0(%in: f32, %in_10: f32, %in_11: f32, %out: f32):
                  %62 = arith.mulf %in_10, %in_11 : f32
                  %63 = arith.addf %in, %62 : f32
                  linalg.yield %63 : f32
                } -> tensor<32x32x128xf32>
                %61 = linalg.copy ins(%50 : tensor<32x32xf32>) outs(%arg9 : tensor<32x32xf32>) -> tensor<32x32xf32>
                affine.yield %61, %55, %60 : tensor<32x32xf32>, tensor<32x32xf32>, tensor<32x32x128xf32>
              }
              %41 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%40#2, %40#1 : tensor<32x32x128xf32>, tensor<32x32xf32>) outs(%28 : tensor<32x32x128xf32>) {
              ^bb0(%in: f32, %in_6: f32, %out: f32):
                %43 = arith.divf %in, %in_6 : f32
                linalg.yield %43 : f32
              } -> tensor<32x32x128xf32>
              %c0_4 = arith.constant 0 : index
              %42 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 524288 + d1 * 128 + d2)>(%33, %34, %c0_4)
              %reinterpret_cast_5 = memref.reinterpret_cast %arg3 to offset: [%42], sizes: [32, 32, 128], strides: [524288, 128, 1] : memref<32x4096x128xf32> to memref<32x32x128xf32, strided<[524288, 128, 1], offset: ?>>
              loom.copy_from_tensor %41, %reinterpret_cast_5 on @DRAM : tensor<32x32x128xf32>, memref<32x32x128xf32, strided<[524288, 128, 1], offset: ?>>
            } {loom.iter_type = #loom.iter_type<temporal>}
          } {loom.iter_type = #loom.iter_type<temporal>}
        } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @y}
      } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @x}
      return
    }
  }
  module attributes {loom.pass_name = "Materialize"} {
    func.func @attention__d0i1_d1i1__f10__d_v_a__BB32__BM32__BN32(%arg0: memref<32x128x4096xf32>, %arg1: memref<32x4096x128xf32>, %arg2: memref<32x4096x128xf32>, %arg3: memref<32x4096x128xf32>) {
      %cst = arith.constant 2.000000e+00 : f32
      %cst_0 = arith.constant 0.12751743074602467 : f64
      %cst_1 = arith.constant 0.000000e+00 : f32
      %cst_2 = arith.constant 1.000000e+00 : f32
      %cst_3 = arith.constant 0xFF800000 : f32
      %c32 = arith.constant 32 : index
      affine.parallel (%arg4) = (0) to (8) {
        affine.parallel (%arg5) = (0) to (8) {
          affine.for %arg6 = 0 to 2 {
            affine.for %arg7 = 0 to 1 {
              %12 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 8 + d1 + d2 * 64)>(%arg4, %arg5, %arg6)
              %13 = loom.alloc [32, 32, 128] on @L1 : memref<32x32x128xf32>
              %14 = loom.alloc [32, 32, 32] on @L1 : memref<32x32x32xf32>
              %15 = loom.init_tensor %14[32, 32, 32] : memref<32x32x32xf32> -> tensor<32x32x32xf32>
              %16 = loom.alloc [32, 128, 32] on @L1 : memref<32x128x32xf32>
              %17 = loom.alloc [32, 32] on @L1 : memref<32x32xf32>
              %18 = loom.init_tensor %17[32, 32] : memref<32x32xf32> -> tensor<32x32xf32>
              %19 = loom.alloc [32, 32] on @L1 : memref<32x32xf32>
              %20 = loom.init_tensor %19[32, 32] : memref<32x32xf32> -> tensor<32x32xf32>
              %21 = loom.alloc [32, 32] on @L1 : memref<32x32xf32>
              %22 = loom.init_tensor %21[32, 32] : memref<32x32xf32> -> tensor<32x32xf32>
              %23 = loom.alloc [32, 32] on @L1 : memref<32x32xf32>
              %24 = loom.init_tensor %23[32, 32] : memref<32x32xf32> -> tensor<32x32xf32>
              %25 = loom.alloc [32, 32] on @L1 : memref<32x32xf32>
              %26 = loom.init_tensor %25[32, 32] : memref<32x32xf32> -> tensor<32x32xf32>
              %27 = loom.alloc [32, 32, 128] on @L1 : memref<32x32x128xf32>
              %28 = loom.init_tensor %27[32, 32, 128] : memref<32x32x128xf32> -> tensor<32x32x128xf32>
              %29 = loom.alloc [32, 32, 128] on @L1 : memref<32x32x128xf32>
              %30 = loom.init_tensor %29[32, 32, 128] : memref<32x32x128xf32> -> tensor<32x32x128xf32>
              %31 = loom.alloc [32, 32, 128] on @L1 : memref<32x32x128xf32>
              %32 = loom.init_tensor %31[32, 32, 128] : memref<32x32x128xf32> -> tensor<32x32x128xf32>
              %33 = arith.muli %arg7, %c32 : index
              %34 = arith.muli %12, %c32 : index
              %c0 = arith.constant 0 : index
              %35 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 524288 + d1 * 128 + d2)>(%33, %34, %c0)
              %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%35], sizes: [32, 32, 128], strides: [524288, 128, 1] : memref<32x4096x128xf32> to memref<32x32x128xf32, strided<[524288, 128, 1], offset: ?>>
              %36 = loom.copy_to_tensor %reinterpret_cast, %27 on @L1, interconnect : [], broadcast : [1, 1] : memref<32x32x128xf32, strided<[524288, 128, 1], offset: ?>>, memref<32x32x128xf32> -> tensor<32x32x128xf32>
              %37 = linalg.fill ins(%cst_1 : f32) outs(%30 : tensor<32x32x128xf32>) -> tensor<32x32x128xf32>
              %38 = linalg.fill ins(%cst_2 : f32) outs(%18 : tensor<32x32xf32>) -> tensor<32x32xf32>
              %39 = linalg.fill ins(%cst_3 : f32) outs(%20 : tensor<32x32xf32>) -> tensor<32x32xf32>
              %40:3 = affine.for %arg8 = 0 to 128 iter_args(%arg9 = %39, %arg10 = %38, %arg11 = %37) -> (tensor<32x32xf32>, tensor<32x32xf32>, tensor<32x32x128xf32>) {
                %43 = arith.muli %arg8, %c32 : index
                %c0_6 = arith.constant 0 : index
                %44 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 524288 + d1 * 4096 + d2)>(%33, %c0_6, %43)
                %reinterpret_cast_7 = memref.reinterpret_cast %arg0 to offset: [%44], sizes: [32, 128, 32], strides: [524288, 4096, 1] : memref<32x128x4096xf32> to memref<32x128x32xf32, strided<[524288, 4096, 1], offset: ?>>
                %45 = loom.copy_to_tensor %reinterpret_cast_7, %16 on @L1, interconnect : [@vertical_links], broadcast : [8, 1] : memref<32x128x32xf32, strided<[524288, 4096, 1], offset: ?>>, memref<32x128x32xf32> -> tensor<32x128x32xf32>
                %46 = linalg.fill ins(%cst_1 : f32) outs(%15 : tensor<32x32x32xf32>) -> tensor<32x32x32xf32>
                %47 = linalg.batch_matmul ins(%36, %45 : tensor<32x32x128xf32>, tensor<32x128x32xf32>) outs(%46 : tensor<32x32x32xf32>) -> tensor<32x32x32xf32>
                %48 = linalg.fill ins(%cst_3 : f32) outs(%22 : tensor<32x32xf32>) -> tensor<32x32xf32>
                %49 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%47 : tensor<32x32x32xf32>) outs(%48 : tensor<32x32xf32>) {
                ^bb0(%in: f32, %out: f32):
                  %62 = arith.maximumf %in, %out : f32
                  linalg.yield %62 : f32
                } -> tensor<32x32xf32>
                %50 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%arg9, %49 : tensor<32x32xf32>, tensor<32x32xf32>) outs(%22 : tensor<32x32xf32>) {
                ^bb0(%in: f32, %in_10: f32, %out: f32):
                  %62 = arith.truncf %cst_0 : f64 to f32
                  %63 = arith.mulf %in_10, %62 : f32
                  %64 = arith.cmpf ogt, %in, %63 : f32
                  %65 = arith.select %64, %in, %63 : f32
                  linalg.yield %65 : f32
                } -> tensor<32x32xf32>
                %51 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%47, %50 : tensor<32x32x32xf32>, tensor<32x32xf32>) outs(%15 : tensor<32x32x32xf32>) {
                ^bb0(%in: f32, %in_10: f32, %out: f32):
                  %62 = arith.truncf %cst_0 : f64 to f32
                  %63 = arith.mulf %in, %62 : f32
                  %64 = arith.subf %63, %in_10 : f32
                  %65 = math.powf %cst, %64 : f32
                  linalg.yield %65 : f32
                } -> tensor<32x32x32xf32>
                %52 = linalg.fill ins(%cst_1 : f32) outs(%24 : tensor<32x32xf32>) -> tensor<32x32xf32>
                %53 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%51 : tensor<32x32x32xf32>) outs(%52 : tensor<32x32xf32>) {
                ^bb0(%in: f32, %out: f32):
                  %62 = arith.addf %in, %out : f32
                  linalg.yield %62 : f32
                } -> tensor<32x32xf32>
                %54 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%arg9, %50 : tensor<32x32xf32>, tensor<32x32xf32>) outs(%26 : tensor<32x32xf32>) {
                ^bb0(%in: f32, %in_10: f32, %out: f32):
                  %62 = arith.subf %in, %in_10 : f32
                  %63 = math.powf %cst, %62 : f32
                  linalg.yield %63 : f32
                } -> tensor<32x32xf32>
                %55 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%arg10, %54, %53 : tensor<32x32xf32>, tensor<32x32xf32>, tensor<32x32xf32>) outs(%arg10 : tensor<32x32xf32>) {
                ^bb0(%in: f32, %in_10: f32, %in_11: f32, %out: f32):
                  %62 = arith.mulf %in, %in_10 : f32
                  %63 = arith.addf %62, %in_11 : f32
                  linalg.yield %63 : f32
                } -> tensor<32x32xf32>
                %c0_8 = arith.constant 0 : index
                %56 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 524288 + d1 * 128 + d2)>(%33, %43, %c0_8)
                %reinterpret_cast_9 = memref.reinterpret_cast %arg1 to offset: [%56], sizes: [32, 32, 128], strides: [524288, 128, 1] : memref<32x4096x128xf32> to memref<32x32x128xf32, strided<[524288, 128, 1], offset: ?>>
                %57 = loom.copy_to_tensor %reinterpret_cast_9, %13 on @L1, interconnect : [@horizontal_links, @vertical_links], broadcast : [8, 8] : memref<32x32x128xf32, strided<[524288, 128, 1], offset: ?>>, memref<32x32x128xf32> -> tensor<32x32x128xf32>
                %58 = linalg.fill ins(%cst_1 : f32) outs(%32 : tensor<32x32x128xf32>) -> tensor<32x32x128xf32>
                %59 = linalg.batch_matmul ins(%51, %57 : tensor<32x32x32xf32>, tensor<32x32x128xf32>) outs(%58 : tensor<32x32x128xf32>) -> tensor<32x32x128xf32>
                %60 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%59, %arg11, %54 : tensor<32x32x128xf32>, tensor<32x32x128xf32>, tensor<32x32xf32>) outs(%arg11 : tensor<32x32x128xf32>) {
                ^bb0(%in: f32, %in_10: f32, %in_11: f32, %out: f32):
                  %62 = arith.mulf %in_10, %in_11 : f32
                  %63 = arith.addf %in, %62 : f32
                  linalg.yield %63 : f32
                } -> tensor<32x32x128xf32>
                %61 = linalg.copy ins(%50 : tensor<32x32xf32>) outs(%arg9 : tensor<32x32xf32>) -> tensor<32x32xf32>
                affine.yield %61, %55, %60 : tensor<32x32xf32>, tensor<32x32xf32>, tensor<32x32x128xf32>
              }
              %41 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%40#2, %40#1 : tensor<32x32x128xf32>, tensor<32x32xf32>) outs(%28 : tensor<32x32x128xf32>) {
              ^bb0(%in: f32, %in_6: f32, %out: f32):
                %43 = arith.divf %in, %in_6 : f32
                linalg.yield %43 : f32
              } -> tensor<32x32x128xf32>
              %c0_4 = arith.constant 0 : index
              %42 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 524288 + d1 * 128 + d2)>(%33, %34, %c0_4)
              %reinterpret_cast_5 = memref.reinterpret_cast %arg3 to offset: [%42], sizes: [32, 32, 128], strides: [524288, 128, 1] : memref<32x4096x128xf32> to memref<32x32x128xf32, strided<[524288, 128, 1], offset: ?>>
              loom.copy_from_tensor %41, %reinterpret_cast_5 on @DRAM : tensor<32x32x128xf32>, memref<32x32x128xf32, strided<[524288, 128, 1], offset: ?>>
            } {loom.iter_type = #loom.iter_type<temporal>}
          } {loom.iter_type = #loom.iter_type<temporal>}
        } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @y}
      } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @x}
      return
    }
  }
  module attributes {loom.pass_name = "Materialize"} {
    func.func @attention__d0i1_d1i1__f10__d_v_h__BB32__BM32__BN32(%arg0: memref<32x128x4096xf32>, %arg1: memref<32x4096x128xf32>, %arg2: memref<32x4096x128xf32>, %arg3: memref<32x4096x128xf32>) {
      %cst = arith.constant 2.000000e+00 : f32
      %cst_0 = arith.constant 0.12751743074602467 : f64
      %cst_1 = arith.constant 0.000000e+00 : f32
      %cst_2 = arith.constant 1.000000e+00 : f32
      %cst_3 = arith.constant 0xFF800000 : f32
      %c32 = arith.constant 32 : index
      affine.parallel (%arg4) = (0) to (8) {
        affine.parallel (%arg5) = (0) to (8) {
          affine.for %arg6 = 0 to 2 {
            affine.for %arg7 = 0 to 1 {
              %12 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 8 + d1 + d2 * 64)>(%arg4, %arg5, %arg6)
              %13 = loom.alloc [32, 32, 128] on @L1 : memref<32x32x128xf32>
              %14 = loom.alloc [32, 32, 32] on @L1 : memref<32x32x32xf32>
              %15 = loom.init_tensor %14[32, 32, 32] : memref<32x32x32xf32> -> tensor<32x32x32xf32>
              %16 = loom.alloc [32, 128, 32] on @L1 : memref<32x128x32xf32>
              %17 = loom.alloc [32, 32] on @L1 : memref<32x32xf32>
              %18 = loom.init_tensor %17[32, 32] : memref<32x32xf32> -> tensor<32x32xf32>
              %19 = loom.alloc [32, 32] on @L1 : memref<32x32xf32>
              %20 = loom.init_tensor %19[32, 32] : memref<32x32xf32> -> tensor<32x32xf32>
              %21 = loom.alloc [32, 32] on @L1 : memref<32x32xf32>
              %22 = loom.init_tensor %21[32, 32] : memref<32x32xf32> -> tensor<32x32xf32>
              %23 = loom.alloc [32, 32] on @L1 : memref<32x32xf32>
              %24 = loom.init_tensor %23[32, 32] : memref<32x32xf32> -> tensor<32x32xf32>
              %25 = loom.alloc [32, 32] on @L1 : memref<32x32xf32>
              %26 = loom.init_tensor %25[32, 32] : memref<32x32xf32> -> tensor<32x32xf32>
              %27 = loom.alloc [32, 32, 128] on @L1 : memref<32x32x128xf32>
              %28 = loom.init_tensor %27[32, 32, 128] : memref<32x32x128xf32> -> tensor<32x32x128xf32>
              %29 = loom.alloc [32, 32, 128] on @L1 : memref<32x32x128xf32>
              %30 = loom.init_tensor %29[32, 32, 128] : memref<32x32x128xf32> -> tensor<32x32x128xf32>
              %31 = loom.alloc [32, 32, 128] on @L1 : memref<32x32x128xf32>
              %32 = loom.init_tensor %31[32, 32, 128] : memref<32x32x128xf32> -> tensor<32x32x128xf32>
              %33 = arith.muli %arg7, %c32 : index
              %34 = arith.muli %12, %c32 : index
              %c0 = arith.constant 0 : index
              %35 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 524288 + d1 * 128 + d2)>(%33, %34, %c0)
              %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%35], sizes: [32, 32, 128], strides: [524288, 128, 1] : memref<32x4096x128xf32> to memref<32x32x128xf32, strided<[524288, 128, 1], offset: ?>>
              %36 = loom.copy_to_tensor %reinterpret_cast, %27 on @L1, interconnect : [], broadcast : [1, 1] : memref<32x32x128xf32, strided<[524288, 128, 1], offset: ?>>, memref<32x32x128xf32> -> tensor<32x32x128xf32>
              %37 = linalg.fill ins(%cst_1 : f32) outs(%30 : tensor<32x32x128xf32>) -> tensor<32x32x128xf32>
              %38 = linalg.fill ins(%cst_2 : f32) outs(%18 : tensor<32x32xf32>) -> tensor<32x32xf32>
              %39 = linalg.fill ins(%cst_3 : f32) outs(%20 : tensor<32x32xf32>) -> tensor<32x32xf32>
              %40:3 = affine.for %arg8 = 0 to 128 iter_args(%arg9 = %39, %arg10 = %38, %arg11 = %37) -> (tensor<32x32xf32>, tensor<32x32xf32>, tensor<32x32x128xf32>) {
                %43 = arith.muli %arg8, %c32 : index
                %c0_6 = arith.constant 0 : index
                %44 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 524288 + d1 * 4096 + d2)>(%33, %c0_6, %43)
                %reinterpret_cast_7 = memref.reinterpret_cast %arg0 to offset: [%44], sizes: [32, 128, 32], strides: [524288, 4096, 1] : memref<32x128x4096xf32> to memref<32x128x32xf32, strided<[524288, 4096, 1], offset: ?>>
                %45 = loom.copy_to_tensor %reinterpret_cast_7, %16 on @L1, interconnect : [@vertical_links], broadcast : [8, 1] : memref<32x128x32xf32, strided<[524288, 4096, 1], offset: ?>>, memref<32x128x32xf32> -> tensor<32x128x32xf32>
                %46 = linalg.fill ins(%cst_1 : f32) outs(%15 : tensor<32x32x32xf32>) -> tensor<32x32x32xf32>
                %47 = linalg.batch_matmul ins(%36, %45 : tensor<32x32x128xf32>, tensor<32x128x32xf32>) outs(%46 : tensor<32x32x32xf32>) -> tensor<32x32x32xf32>
                %48 = linalg.fill ins(%cst_3 : f32) outs(%22 : tensor<32x32xf32>) -> tensor<32x32xf32>
                %49 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%47 : tensor<32x32x32xf32>) outs(%48 : tensor<32x32xf32>) {
                ^bb0(%in: f32, %out: f32):
                  %62 = arith.maximumf %in, %out : f32
                  linalg.yield %62 : f32
                } -> tensor<32x32xf32>
                %50 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%arg9, %49 : tensor<32x32xf32>, tensor<32x32xf32>) outs(%22 : tensor<32x32xf32>) {
                ^bb0(%in: f32, %in_10: f32, %out: f32):
                  %62 = arith.truncf %cst_0 : f64 to f32
                  %63 = arith.mulf %in_10, %62 : f32
                  %64 = arith.cmpf ogt, %in, %63 : f32
                  %65 = arith.select %64, %in, %63 : f32
                  linalg.yield %65 : f32
                } -> tensor<32x32xf32>
                %51 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%47, %50 : tensor<32x32x32xf32>, tensor<32x32xf32>) outs(%15 : tensor<32x32x32xf32>) {
                ^bb0(%in: f32, %in_10: f32, %out: f32):
                  %62 = arith.truncf %cst_0 : f64 to f32
                  %63 = arith.mulf %in, %62 : f32
                  %64 = arith.subf %63, %in_10 : f32
                  %65 = math.powf %cst, %64 : f32
                  linalg.yield %65 : f32
                } -> tensor<32x32x32xf32>
                %52 = linalg.fill ins(%cst_1 : f32) outs(%24 : tensor<32x32xf32>) -> tensor<32x32xf32>
                %53 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%51 : tensor<32x32x32xf32>) outs(%52 : tensor<32x32xf32>) {
                ^bb0(%in: f32, %out: f32):
                  %62 = arith.addf %in, %out : f32
                  linalg.yield %62 : f32
                } -> tensor<32x32xf32>
                %54 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%arg9, %50 : tensor<32x32xf32>, tensor<32x32xf32>) outs(%26 : tensor<32x32xf32>) {
                ^bb0(%in: f32, %in_10: f32, %out: f32):
                  %62 = arith.subf %in, %in_10 : f32
                  %63 = math.powf %cst, %62 : f32
                  linalg.yield %63 : f32
                } -> tensor<32x32xf32>
                %55 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%arg10, %54, %53 : tensor<32x32xf32>, tensor<32x32xf32>, tensor<32x32xf32>) outs(%arg10 : tensor<32x32xf32>) {
                ^bb0(%in: f32, %in_10: f32, %in_11: f32, %out: f32):
                  %62 = arith.mulf %in, %in_10 : f32
                  %63 = arith.addf %62, %in_11 : f32
                  linalg.yield %63 : f32
                } -> tensor<32x32xf32>
                %c0_8 = arith.constant 0 : index
                %56 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 524288 + d1 * 128 + d2)>(%33, %43, %c0_8)
                %reinterpret_cast_9 = memref.reinterpret_cast %arg1 to offset: [%56], sizes: [32, 32, 128], strides: [524288, 128, 1] : memref<32x4096x128xf32> to memref<32x32x128xf32, strided<[524288, 128, 1], offset: ?>>
                %57 = loom.copy_to_tensor %reinterpret_cast_9, %13 on @L1, interconnect : [@horizontal_links], broadcast : [1, 8] : memref<32x32x128xf32, strided<[524288, 128, 1], offset: ?>>, memref<32x32x128xf32> -> tensor<32x32x128xf32>
                %58 = linalg.fill ins(%cst_1 : f32) outs(%32 : tensor<32x32x128xf32>) -> tensor<32x32x128xf32>
                %59 = linalg.batch_matmul ins(%51, %57 : tensor<32x32x32xf32>, tensor<32x32x128xf32>) outs(%58 : tensor<32x32x128xf32>) -> tensor<32x32x128xf32>
                %60 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%59, %arg11, %54 : tensor<32x32x128xf32>, tensor<32x32x128xf32>, tensor<32x32xf32>) outs(%arg11 : tensor<32x32x128xf32>) {
                ^bb0(%in: f32, %in_10: f32, %in_11: f32, %out: f32):
                  %62 = arith.mulf %in_10, %in_11 : f32
                  %63 = arith.addf %in, %62 : f32
                  linalg.yield %63 : f32
                } -> tensor<32x32x128xf32>
                %61 = linalg.copy ins(%50 : tensor<32x32xf32>) outs(%arg9 : tensor<32x32xf32>) -> tensor<32x32xf32>
                affine.yield %61, %55, %60 : tensor<32x32xf32>, tensor<32x32xf32>, tensor<32x32x128xf32>
              }
              %41 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%40#2, %40#1 : tensor<32x32x128xf32>, tensor<32x32xf32>) outs(%28 : tensor<32x32x128xf32>) {
              ^bb0(%in: f32, %in_6: f32, %out: f32):
                %43 = arith.divf %in, %in_6 : f32
                linalg.yield %43 : f32
              } -> tensor<32x32x128xf32>
              %c0_4 = arith.constant 0 : index
              %42 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 524288 + d1 * 128 + d2)>(%33, %34, %c0_4)
              %reinterpret_cast_5 = memref.reinterpret_cast %arg3 to offset: [%42], sizes: [32, 32, 128], strides: [524288, 128, 1] : memref<32x4096x128xf32> to memref<32x32x128xf32, strided<[524288, 128, 1], offset: ?>>
              loom.copy_from_tensor %41, %reinterpret_cast_5 on @DRAM : tensor<32x32x128xf32>, memref<32x32x128xf32, strided<[524288, 128, 1], offset: ?>>
            } {loom.iter_type = #loom.iter_type<temporal>}
          } {loom.iter_type = #loom.iter_type<temporal>}
        } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @y}
      } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @x}
      return
    }
  }
  module attributes {loom.pass_name = "Materialize"} {
    func.func @attention__d0i1_d1i1__f10__d_v_v__BB32__BM32__BN32(%arg0: memref<32x128x4096xf32>, %arg1: memref<32x4096x128xf32>, %arg2: memref<32x4096x128xf32>, %arg3: memref<32x4096x128xf32>) {
      %cst = arith.constant 2.000000e+00 : f32
      %cst_0 = arith.constant 0.12751743074602467 : f64
      %cst_1 = arith.constant 0.000000e+00 : f32
      %cst_2 = arith.constant 1.000000e+00 : f32
      %cst_3 = arith.constant 0xFF800000 : f32
      %c32 = arith.constant 32 : index
      affine.parallel (%arg4) = (0) to (8) {
        affine.parallel (%arg5) = (0) to (8) {
          affine.for %arg6 = 0 to 2 {
            affine.for %arg7 = 0 to 1 {
              %12 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 8 + d1 + d2 * 64)>(%arg4, %arg5, %arg6)
              %13 = loom.alloc [32, 32, 128] on @L1 : memref<32x32x128xf32>
              %14 = loom.alloc [32, 32, 32] on @L1 : memref<32x32x32xf32>
              %15 = loom.init_tensor %14[32, 32, 32] : memref<32x32x32xf32> -> tensor<32x32x32xf32>
              %16 = loom.alloc [32, 128, 32] on @L1 : memref<32x128x32xf32>
              %17 = loom.alloc [32, 32] on @L1 : memref<32x32xf32>
              %18 = loom.init_tensor %17[32, 32] : memref<32x32xf32> -> tensor<32x32xf32>
              %19 = loom.alloc [32, 32] on @L1 : memref<32x32xf32>
              %20 = loom.init_tensor %19[32, 32] : memref<32x32xf32> -> tensor<32x32xf32>
              %21 = loom.alloc [32, 32] on @L1 : memref<32x32xf32>
              %22 = loom.init_tensor %21[32, 32] : memref<32x32xf32> -> tensor<32x32xf32>
              %23 = loom.alloc [32, 32] on @L1 : memref<32x32xf32>
              %24 = loom.init_tensor %23[32, 32] : memref<32x32xf32> -> tensor<32x32xf32>
              %25 = loom.alloc [32, 32] on @L1 : memref<32x32xf32>
              %26 = loom.init_tensor %25[32, 32] : memref<32x32xf32> -> tensor<32x32xf32>
              %27 = loom.alloc [32, 32, 128] on @L1 : memref<32x32x128xf32>
              %28 = loom.init_tensor %27[32, 32, 128] : memref<32x32x128xf32> -> tensor<32x32x128xf32>
              %29 = loom.alloc [32, 32, 128] on @L1 : memref<32x32x128xf32>
              %30 = loom.init_tensor %29[32, 32, 128] : memref<32x32x128xf32> -> tensor<32x32x128xf32>
              %31 = loom.alloc [32, 32, 128] on @L1 : memref<32x32x128xf32>
              %32 = loom.init_tensor %31[32, 32, 128] : memref<32x32x128xf32> -> tensor<32x32x128xf32>
              %33 = arith.muli %arg7, %c32 : index
              %34 = arith.muli %12, %c32 : index
              %c0 = arith.constant 0 : index
              %35 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 524288 + d1 * 128 + d2)>(%33, %34, %c0)
              %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%35], sizes: [32, 32, 128], strides: [524288, 128, 1] : memref<32x4096x128xf32> to memref<32x32x128xf32, strided<[524288, 128, 1], offset: ?>>
              %36 = loom.copy_to_tensor %reinterpret_cast, %27 on @L1, interconnect : [], broadcast : [1, 1] : memref<32x32x128xf32, strided<[524288, 128, 1], offset: ?>>, memref<32x32x128xf32> -> tensor<32x32x128xf32>
              %37 = linalg.fill ins(%cst_1 : f32) outs(%30 : tensor<32x32x128xf32>) -> tensor<32x32x128xf32>
              %38 = linalg.fill ins(%cst_2 : f32) outs(%18 : tensor<32x32xf32>) -> tensor<32x32xf32>
              %39 = linalg.fill ins(%cst_3 : f32) outs(%20 : tensor<32x32xf32>) -> tensor<32x32xf32>
              %40:3 = affine.for %arg8 = 0 to 128 iter_args(%arg9 = %39, %arg10 = %38, %arg11 = %37) -> (tensor<32x32xf32>, tensor<32x32xf32>, tensor<32x32x128xf32>) {
                %43 = arith.muli %arg8, %c32 : index
                %c0_6 = arith.constant 0 : index
                %44 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 524288 + d1 * 4096 + d2)>(%33, %c0_6, %43)
                %reinterpret_cast_7 = memref.reinterpret_cast %arg0 to offset: [%44], sizes: [32, 128, 32], strides: [524288, 4096, 1] : memref<32x128x4096xf32> to memref<32x128x32xf32, strided<[524288, 4096, 1], offset: ?>>
                %45 = loom.copy_to_tensor %reinterpret_cast_7, %16 on @L1, interconnect : [@vertical_links], broadcast : [8, 1] : memref<32x128x32xf32, strided<[524288, 4096, 1], offset: ?>>, memref<32x128x32xf32> -> tensor<32x128x32xf32>
                %46 = linalg.fill ins(%cst_1 : f32) outs(%15 : tensor<32x32x32xf32>) -> tensor<32x32x32xf32>
                %47 = linalg.batch_matmul ins(%36, %45 : tensor<32x32x128xf32>, tensor<32x128x32xf32>) outs(%46 : tensor<32x32x32xf32>) -> tensor<32x32x32xf32>
                %48 = linalg.fill ins(%cst_3 : f32) outs(%22 : tensor<32x32xf32>) -> tensor<32x32xf32>
                %49 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%47 : tensor<32x32x32xf32>) outs(%48 : tensor<32x32xf32>) {
                ^bb0(%in: f32, %out: f32):
                  %62 = arith.maximumf %in, %out : f32
                  linalg.yield %62 : f32
                } -> tensor<32x32xf32>
                %50 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%arg9, %49 : tensor<32x32xf32>, tensor<32x32xf32>) outs(%22 : tensor<32x32xf32>) {
                ^bb0(%in: f32, %in_10: f32, %out: f32):
                  %62 = arith.truncf %cst_0 : f64 to f32
                  %63 = arith.mulf %in_10, %62 : f32
                  %64 = arith.cmpf ogt, %in, %63 : f32
                  %65 = arith.select %64, %in, %63 : f32
                  linalg.yield %65 : f32
                } -> tensor<32x32xf32>
                %51 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%47, %50 : tensor<32x32x32xf32>, tensor<32x32xf32>) outs(%15 : tensor<32x32x32xf32>) {
                ^bb0(%in: f32, %in_10: f32, %out: f32):
                  %62 = arith.truncf %cst_0 : f64 to f32
                  %63 = arith.mulf %in, %62 : f32
                  %64 = arith.subf %63, %in_10 : f32
                  %65 = math.powf %cst, %64 : f32
                  linalg.yield %65 : f32
                } -> tensor<32x32x32xf32>
                %52 = linalg.fill ins(%cst_1 : f32) outs(%24 : tensor<32x32xf32>) -> tensor<32x32xf32>
                %53 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%51 : tensor<32x32x32xf32>) outs(%52 : tensor<32x32xf32>) {
                ^bb0(%in: f32, %out: f32):
                  %62 = arith.addf %in, %out : f32
                  linalg.yield %62 : f32
                } -> tensor<32x32xf32>
                %54 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%arg9, %50 : tensor<32x32xf32>, tensor<32x32xf32>) outs(%26 : tensor<32x32xf32>) {
                ^bb0(%in: f32, %in_10: f32, %out: f32):
                  %62 = arith.subf %in, %in_10 : f32
                  %63 = math.powf %cst, %62 : f32
                  linalg.yield %63 : f32
                } -> tensor<32x32xf32>
                %55 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%arg10, %54, %53 : tensor<32x32xf32>, tensor<32x32xf32>, tensor<32x32xf32>) outs(%arg10 : tensor<32x32xf32>) {
                ^bb0(%in: f32, %in_10: f32, %in_11: f32, %out: f32):
                  %62 = arith.mulf %in, %in_10 : f32
                  %63 = arith.addf %62, %in_11 : f32
                  linalg.yield %63 : f32
                } -> tensor<32x32xf32>
                %c0_8 = arith.constant 0 : index
                %56 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 524288 + d1 * 128 + d2)>(%33, %43, %c0_8)
                %reinterpret_cast_9 = memref.reinterpret_cast %arg1 to offset: [%56], sizes: [32, 32, 128], strides: [524288, 128, 1] : memref<32x4096x128xf32> to memref<32x32x128xf32, strided<[524288, 128, 1], offset: ?>>
                %57 = loom.copy_to_tensor %reinterpret_cast_9, %13 on @L1, interconnect : [@vertical_links], broadcast : [8, 1] : memref<32x32x128xf32, strided<[524288, 128, 1], offset: ?>>, memref<32x32x128xf32> -> tensor<32x32x128xf32>
                %58 = linalg.fill ins(%cst_1 : f32) outs(%32 : tensor<32x32x128xf32>) -> tensor<32x32x128xf32>
                %59 = linalg.batch_matmul ins(%51, %57 : tensor<32x32x32xf32>, tensor<32x32x128xf32>) outs(%58 : tensor<32x32x128xf32>) -> tensor<32x32x128xf32>
                %60 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%59, %arg11, %54 : tensor<32x32x128xf32>, tensor<32x32x128xf32>, tensor<32x32xf32>) outs(%arg11 : tensor<32x32x128xf32>) {
                ^bb0(%in: f32, %in_10: f32, %in_11: f32, %out: f32):
                  %62 = arith.mulf %in_10, %in_11 : f32
                  %63 = arith.addf %in, %62 : f32
                  linalg.yield %63 : f32
                } -> tensor<32x32x128xf32>
                %61 = linalg.copy ins(%50 : tensor<32x32xf32>) outs(%arg9 : tensor<32x32xf32>) -> tensor<32x32xf32>
                affine.yield %61, %55, %60 : tensor<32x32xf32>, tensor<32x32xf32>, tensor<32x32x128xf32>
              }
              %41 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%40#2, %40#1 : tensor<32x32x128xf32>, tensor<32x32xf32>) outs(%28 : tensor<32x32x128xf32>) {
              ^bb0(%in: f32, %in_6: f32, %out: f32):
                %43 = arith.divf %in, %in_6 : f32
                linalg.yield %43 : f32
              } -> tensor<32x32x128xf32>
              %c0_4 = arith.constant 0 : index
              %42 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 524288 + d1 * 128 + d2)>(%33, %34, %c0_4)
              %reinterpret_cast_5 = memref.reinterpret_cast %arg3 to offset: [%42], sizes: [32, 32, 128], strides: [524288, 128, 1] : memref<32x4096x128xf32> to memref<32x32x128xf32, strided<[524288, 128, 1], offset: ?>>
              loom.copy_from_tensor %41, %reinterpret_cast_5 on @DRAM : tensor<32x32x128xf32>, memref<32x32x128xf32, strided<[524288, 128, 1], offset: ?>>
            } {loom.iter_type = #loom.iter_type<temporal>}
          } {loom.iter_type = #loom.iter_type<temporal>}
        } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @y}
      } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @x}
      return
    }
  }
}
