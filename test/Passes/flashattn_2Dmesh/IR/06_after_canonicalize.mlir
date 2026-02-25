module {
  module attributes {loom.pass_name = "Materialize"} {
    func.func @attention__BB64__BM64__BN64(%arg0: memref<2x4096x4096xf32>, %arg1: memref<2x4096x4096xf32>, %arg2: memref<2x4096x4096xf32>, %arg3: memref<2x4096x4096xf32>) {
      %cst = arith.constant 2.000000e+00 : f32
      %cst_0 = arith.constant 0.022542110000000001 : f64
      %cst_1 = arith.constant 0.000000e+00 : f32
      %cst_2 = arith.constant 1.000000e+00 : f32
      %cst_3 = arith.constant 0xFF800000 : f32
      %c64 = arith.constant 64 : index
      affine.parallel (%arg4, %arg5) = (0, 0) to (1, 64) {
        %0 = loom.alloc [64, 64, 4096] on @L1 : memref<64x64x4096xf32>
        %1 = loom.alloc [64, 64, 64] on @L1 : memref<64x64x64xf32>
        %2 = loom.init_tensor %1[64, 64, 64] : memref<64x64x64xf32> -> tensor<64x64x64xf32>
        %3 = loom.alloc [64, 4096, 64] on @L1 : memref<64x4096x64xf32>
        %4 = loom.alloc [64, 64] on @L1 : memref<64x64xf32>
        %5 = loom.init_tensor %4[64, 64] : memref<64x64xf32> -> tensor<64x64xf32>
        %6 = loom.alloc [64, 64] on @L1 : memref<64x64xf32>
        %7 = loom.init_tensor %6[64, 64] : memref<64x64xf32> -> tensor<64x64xf32>
        %8 = loom.alloc [64, 64] on @L1 : memref<64x64xf32>
        %9 = loom.init_tensor %8[64, 64] : memref<64x64xf32> -> tensor<64x64xf32>
        %10 = loom.alloc [64, 64] on @L1 : memref<64x64xf32>
        %11 = loom.init_tensor %10[64, 64] : memref<64x64xf32> -> tensor<64x64xf32>
        %12 = loom.alloc [64, 64] on @L1 : memref<64x64xf32>
        %13 = loom.init_tensor %12[64, 64] : memref<64x64xf32> -> tensor<64x64xf32>
        %14 = loom.alloc [64, 64, 4096] on @L1 : memref<64x64x4096xf32>
        %15 = loom.init_tensor %14[64, 64, 4096] : memref<64x64x4096xf32> -> tensor<64x64x4096xf32>
        %16 = loom.alloc [64, 64, 4096] on @L1 : memref<64x64x4096xf32>
        %17 = loom.init_tensor %16[64, 64, 4096] : memref<64x64x4096xf32> -> tensor<64x64x4096xf32>
        %18 = loom.alloc [64, 64, 4096] on @L1 : memref<64x64x4096xf32>
        %19 = loom.init_tensor %18[64, 64, 4096] : memref<64x64x4096xf32> -> tensor<64x64x4096xf32>
        %20 = arith.muli %arg4, %c64 : index
        %21 = arith.muli %arg5, %c64 : index
        %c0 = arith.constant 0 : index
        %22 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 16777216 + d1 * 4096 + d2)>(%20, %21, %c0)
        %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%22], sizes: [64, 64, 4096], strides: [16777216, 4096, 1] : memref<2x4096x4096xf32> to memref<64x64x4096xf32, strided<[16777216, 4096, 1], offset: ?>>
        %23 = loom.copy_to_tensor %reinterpret_cast, %14, interconnect : [], broadcast : [1, 1] : memref<64x64x4096xf32, strided<[16777216, 4096, 1], offset: ?>>, memref<64x64x4096xf32> -> tensor<64x64x4096xf32>
        %24 = linalg.fill ins(%cst_1 : f32) outs(%17 : tensor<64x64x4096xf32>) -> tensor<64x64x4096xf32>
        %25 = linalg.fill ins(%cst_2 : f32) outs(%5 : tensor<64x64xf32>) -> tensor<64x64xf32>
        %26 = linalg.fill ins(%cst_3 : f32) outs(%7 : tensor<64x64xf32>) -> tensor<64x64xf32>
        %27:3 = affine.for %arg6 = 0 to 64 iter_args(%arg7 = %26, %arg8 = %25, %arg9 = %24) -> (tensor<64x64xf32>, tensor<64x64xf32>, tensor<64x64x4096xf32>) {
          %30 = arith.muli %arg6, %c64 : index
          %c0_6 = arith.constant 0 : index
          %31 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 16777216 + d1 * 4096 + d2)>(%20, %c0_6, %30)
          %reinterpret_cast_7 = memref.reinterpret_cast %arg0 to offset: [%31], sizes: [64, 4096, 64], strides: [16777216, 4096, 1] : memref<2x4096x4096xf32> to memref<64x4096x64xf32, strided<[16777216, 4096, 1], offset: ?>>
          %32 = loom.copy_to_tensor %reinterpret_cast_7, %3, interconnect : [], broadcast : [1, 1] : memref<64x4096x64xf32, strided<[16777216, 4096, 1], offset: ?>>, memref<64x4096x64xf32> -> tensor<64x4096x64xf32>
          %33 = linalg.fill ins(%cst_1 : f32) outs(%2 : tensor<64x64x64xf32>) -> tensor<64x64x64xf32>
          %34 = linalg.batch_matmul ins(%23, %32 : tensor<64x64x4096xf32>, tensor<64x4096x64xf32>) outs(%33 : tensor<64x64x64xf32>) -> tensor<64x64x64xf32>
          %35 = linalg.fill ins(%cst_3 : f32) outs(%9 : tensor<64x64xf32>) -> tensor<64x64xf32>
          %36 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%34 : tensor<64x64x64xf32>) outs(%35 : tensor<64x64xf32>) {
          ^bb0(%in: f32, %out: f32):
            %49 = arith.maximumf %in, %out : f32
            linalg.yield %49 : f32
          } -> tensor<64x64xf32>
          %37 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%arg7, %36 : tensor<64x64xf32>, tensor<64x64xf32>) outs(%9 : tensor<64x64xf32>) {
          ^bb0(%in: f32, %in_10: f32, %out: f32):
            %49 = arith.truncf %cst_0 : f64 to f32
            %50 = arith.mulf %in_10, %49 : f32
            %51 = arith.cmpf ogt, %in, %50 : f32
            %52 = arith.select %51, %in, %50 : f32
            linalg.yield %52 : f32
          } -> tensor<64x64xf32>
          %38 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%34, %37 : tensor<64x64x64xf32>, tensor<64x64xf32>) outs(%2 : tensor<64x64x64xf32>) {
          ^bb0(%in: f32, %in_10: f32, %out: f32):
            %49 = arith.truncf %cst_0 : f64 to f32
            %50 = arith.mulf %in, %49 : f32
            %51 = arith.subf %50, %in_10 : f32
            %52 = math.powf %cst, %51 : f32
            linalg.yield %52 : f32
          } -> tensor<64x64x64xf32>
          %39 = linalg.fill ins(%cst_1 : f32) outs(%11 : tensor<64x64xf32>) -> tensor<64x64xf32>
          %40 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%38 : tensor<64x64x64xf32>) outs(%39 : tensor<64x64xf32>) {
          ^bb0(%in: f32, %out: f32):
            %49 = arith.addf %in, %out : f32
            linalg.yield %49 : f32
          } -> tensor<64x64xf32>
          %41 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%arg7, %37 : tensor<64x64xf32>, tensor<64x64xf32>) outs(%13 : tensor<64x64xf32>) {
          ^bb0(%in: f32, %in_10: f32, %out: f32):
            %49 = arith.subf %in, %in_10 : f32
            %50 = math.powf %cst, %49 : f32
            linalg.yield %50 : f32
          } -> tensor<64x64xf32>
          %42 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%arg8, %41, %40 : tensor<64x64xf32>, tensor<64x64xf32>, tensor<64x64xf32>) outs(%arg8 : tensor<64x64xf32>) {
          ^bb0(%in: f32, %in_10: f32, %in_11: f32, %out: f32):
            %49 = arith.mulf %in, %in_10 : f32
            %50 = arith.addf %49, %in_11 : f32
            linalg.yield %50 : f32
          } -> tensor<64x64xf32>
          %c0_8 = arith.constant 0 : index
          %43 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 16777216 + d1 * 4096 + d2)>(%20, %30, %c0_8)
          %reinterpret_cast_9 = memref.reinterpret_cast %arg1 to offset: [%43], sizes: [64, 64, 4096], strides: [16777216, 4096, 1] : memref<2x4096x4096xf32> to memref<64x64x4096xf32, strided<[16777216, 4096, 1], offset: ?>>
          %44 = loom.copy_to_tensor %reinterpret_cast_9, %0, interconnect : [], broadcast : [1, 1] : memref<64x64x4096xf32, strided<[16777216, 4096, 1], offset: ?>>, memref<64x64x4096xf32> -> tensor<64x64x4096xf32>
          %45 = linalg.fill ins(%cst_1 : f32) outs(%19 : tensor<64x64x4096xf32>) -> tensor<64x64x4096xf32>
          %46 = linalg.batch_matmul ins(%38, %44 : tensor<64x64x64xf32>, tensor<64x64x4096xf32>) outs(%45 : tensor<64x64x4096xf32>) -> tensor<64x64x4096xf32>
          %47 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%46, %arg9, %41 : tensor<64x64x4096xf32>, tensor<64x64x4096xf32>, tensor<64x64xf32>) outs(%arg9 : tensor<64x64x4096xf32>) {
          ^bb0(%in: f32, %in_10: f32, %in_11: f32, %out: f32):
            %49 = arith.mulf %in_10, %in_11 : f32
            %50 = arith.addf %in, %49 : f32
            linalg.yield %50 : f32
          } -> tensor<64x64x4096xf32>
          %48 = linalg.copy ins(%37 : tensor<64x64xf32>) outs(%arg7 : tensor<64x64xf32>) -> tensor<64x64xf32>
          affine.yield %48, %42, %47 : tensor<64x64xf32>, tensor<64x64xf32>, tensor<64x64x4096xf32>
        }
        %28 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%27#2, %27#1 : tensor<64x64x4096xf32>, tensor<64x64xf32>) outs(%15 : tensor<64x64x4096xf32>) {
        ^bb0(%in: f32, %in_6: f32, %out: f32):
          %30 = arith.divf %in, %in_6 : f32
          linalg.yield %30 : f32
        } -> tensor<64x64x4096xf32>
        %c0_4 = arith.constant 0 : index
        %29 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 16777216 + d1 * 4096 + d2)>(%20, %21, %c0_4)
        %reinterpret_cast_5 = memref.reinterpret_cast %arg3 to offset: [%29], sizes: [64, 64, 4096], strides: [16777216, 4096, 1] : memref<2x4096x4096xf32> to memref<64x64x4096xf32, strided<[16777216, 4096, 1], offset: ?>>
        loom.copy_from_tensor %28, %reinterpret_cast_5 : tensor<64x64x4096xf32>, memref<64x64x4096xf32, strided<[16777216, 4096, 1], offset: ?>>
      }
      return
    }
    func.func @attention__BB32__BM32__BN256(%arg0: memref<2x4096x4096xf32>, %arg1: memref<2x4096x4096xf32>, %arg2: memref<2x4096x4096xf32>, %arg3: memref<2x4096x4096xf32>) {
      %cst = arith.constant 2.000000e+00 : f32
      %cst_0 = arith.constant 0.022542110000000001 : f64
      %cst_1 = arith.constant 0.000000e+00 : f32
      %cst_2 = arith.constant 1.000000e+00 : f32
      %cst_3 = arith.constant 0xFF800000 : f32
      %c32 = arith.constant 32 : index
      %c256 = arith.constant 256 : index
      affine.parallel (%arg4, %arg5) = (0, 0) to (1, 128) {
        %0 = loom.alloc [32, 256, 4096] on @L1 : memref<32x256x4096xf32>
        %1 = loom.alloc [32, 32, 256] on @L1 : memref<32x32x256xf32>
        %2 = loom.init_tensor %1[32, 32, 256] : memref<32x32x256xf32> -> tensor<32x32x256xf32>
        %3 = loom.alloc [32, 4096, 256] on @L1 : memref<32x4096x256xf32>
        %4 = loom.alloc [32, 32] on @L1 : memref<32x32xf32>
        %5 = loom.init_tensor %4[32, 32] : memref<32x32xf32> -> tensor<32x32xf32>
        %6 = loom.alloc [32, 32] on @L1 : memref<32x32xf32>
        %7 = loom.init_tensor %6[32, 32] : memref<32x32xf32> -> tensor<32x32xf32>
        %8 = loom.alloc [32, 32] on @L1 : memref<32x32xf32>
        %9 = loom.init_tensor %8[32, 32] : memref<32x32xf32> -> tensor<32x32xf32>
        %10 = loom.alloc [32, 32] on @L1 : memref<32x32xf32>
        %11 = loom.init_tensor %10[32, 32] : memref<32x32xf32> -> tensor<32x32xf32>
        %12 = loom.alloc [32, 32] on @L1 : memref<32x32xf32>
        %13 = loom.init_tensor %12[32, 32] : memref<32x32xf32> -> tensor<32x32xf32>
        %14 = loom.alloc [32, 32, 4096] on @L1 : memref<32x32x4096xf32>
        %15 = loom.init_tensor %14[32, 32, 4096] : memref<32x32x4096xf32> -> tensor<32x32x4096xf32>
        %16 = loom.alloc [32, 32, 4096] on @L1 : memref<32x32x4096xf32>
        %17 = loom.init_tensor %16[32, 32, 4096] : memref<32x32x4096xf32> -> tensor<32x32x4096xf32>
        %18 = loom.alloc [32, 32, 4096] on @L1 : memref<32x32x4096xf32>
        %19 = loom.init_tensor %18[32, 32, 4096] : memref<32x32x4096xf32> -> tensor<32x32x4096xf32>
        %20 = arith.muli %arg4, %c32 : index
        %21 = arith.muli %arg5, %c32 : index
        %c0 = arith.constant 0 : index
        %22 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 16777216 + d1 * 4096 + d2)>(%20, %21, %c0)
        %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%22], sizes: [32, 32, 4096], strides: [16777216, 4096, 1] : memref<2x4096x4096xf32> to memref<32x32x4096xf32, strided<[16777216, 4096, 1], offset: ?>>
        %23 = loom.copy_to_tensor %reinterpret_cast, %14, interconnect : [], broadcast : [1, 1] : memref<32x32x4096xf32, strided<[16777216, 4096, 1], offset: ?>>, memref<32x32x4096xf32> -> tensor<32x32x4096xf32>
        %24 = linalg.fill ins(%cst_1 : f32) outs(%17 : tensor<32x32x4096xf32>) -> tensor<32x32x4096xf32>
        %25 = linalg.fill ins(%cst_2 : f32) outs(%5 : tensor<32x32xf32>) -> tensor<32x32xf32>
        %26 = linalg.fill ins(%cst_3 : f32) outs(%7 : tensor<32x32xf32>) -> tensor<32x32xf32>
        %27:3 = affine.for %arg6 = 0 to 16 iter_args(%arg7 = %26, %arg8 = %25, %arg9 = %24) -> (tensor<32x32xf32>, tensor<32x32xf32>, tensor<32x32x4096xf32>) {
          %30 = arith.muli %arg6, %c256 : index
          %c0_6 = arith.constant 0 : index
          %31 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 16777216 + d1 * 4096 + d2)>(%20, %c0_6, %30)
          %reinterpret_cast_7 = memref.reinterpret_cast %arg0 to offset: [%31], sizes: [32, 4096, 256], strides: [16777216, 4096, 1] : memref<2x4096x4096xf32> to memref<32x4096x256xf32, strided<[16777216, 4096, 1], offset: ?>>
          %32 = loom.copy_to_tensor %reinterpret_cast_7, %3, interconnect : [], broadcast : [1, 1] : memref<32x4096x256xf32, strided<[16777216, 4096, 1], offset: ?>>, memref<32x4096x256xf32> -> tensor<32x4096x256xf32>
          %33 = linalg.fill ins(%cst_1 : f32) outs(%2 : tensor<32x32x256xf32>) -> tensor<32x32x256xf32>
          %34 = linalg.batch_matmul ins(%23, %32 : tensor<32x32x4096xf32>, tensor<32x4096x256xf32>) outs(%33 : tensor<32x32x256xf32>) -> tensor<32x32x256xf32>
          %35 = linalg.fill ins(%cst_3 : f32) outs(%9 : tensor<32x32xf32>) -> tensor<32x32xf32>
          %36 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%34 : tensor<32x32x256xf32>) outs(%35 : tensor<32x32xf32>) {
          ^bb0(%in: f32, %out: f32):
            %49 = arith.maximumf %in, %out : f32
            linalg.yield %49 : f32
          } -> tensor<32x32xf32>
          %37 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%arg7, %36 : tensor<32x32xf32>, tensor<32x32xf32>) outs(%9 : tensor<32x32xf32>) {
          ^bb0(%in: f32, %in_10: f32, %out: f32):
            %49 = arith.truncf %cst_0 : f64 to f32
            %50 = arith.mulf %in_10, %49 : f32
            %51 = arith.cmpf ogt, %in, %50 : f32
            %52 = arith.select %51, %in, %50 : f32
            linalg.yield %52 : f32
          } -> tensor<32x32xf32>
          %38 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%34, %37 : tensor<32x32x256xf32>, tensor<32x32xf32>) outs(%2 : tensor<32x32x256xf32>) {
          ^bb0(%in: f32, %in_10: f32, %out: f32):
            %49 = arith.truncf %cst_0 : f64 to f32
            %50 = arith.mulf %in, %49 : f32
            %51 = arith.subf %50, %in_10 : f32
            %52 = math.powf %cst, %51 : f32
            linalg.yield %52 : f32
          } -> tensor<32x32x256xf32>
          %39 = linalg.fill ins(%cst_1 : f32) outs(%11 : tensor<32x32xf32>) -> tensor<32x32xf32>
          %40 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%38 : tensor<32x32x256xf32>) outs(%39 : tensor<32x32xf32>) {
          ^bb0(%in: f32, %out: f32):
            %49 = arith.addf %in, %out : f32
            linalg.yield %49 : f32
          } -> tensor<32x32xf32>
          %41 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%arg7, %37 : tensor<32x32xf32>, tensor<32x32xf32>) outs(%13 : tensor<32x32xf32>) {
          ^bb0(%in: f32, %in_10: f32, %out: f32):
            %49 = arith.subf %in, %in_10 : f32
            %50 = math.powf %cst, %49 : f32
            linalg.yield %50 : f32
          } -> tensor<32x32xf32>
          %42 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%arg8, %41, %40 : tensor<32x32xf32>, tensor<32x32xf32>, tensor<32x32xf32>) outs(%arg8 : tensor<32x32xf32>) {
          ^bb0(%in: f32, %in_10: f32, %in_11: f32, %out: f32):
            %49 = arith.mulf %in, %in_10 : f32
            %50 = arith.addf %49, %in_11 : f32
            linalg.yield %50 : f32
          } -> tensor<32x32xf32>
          %c0_8 = arith.constant 0 : index
          %43 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 16777216 + d1 * 4096 + d2)>(%20, %30, %c0_8)
          %reinterpret_cast_9 = memref.reinterpret_cast %arg1 to offset: [%43], sizes: [32, 256, 4096], strides: [16777216, 4096, 1] : memref<2x4096x4096xf32> to memref<32x256x4096xf32, strided<[16777216, 4096, 1], offset: ?>>
          %44 = loom.copy_to_tensor %reinterpret_cast_9, %0, interconnect : [], broadcast : [1, 1] : memref<32x256x4096xf32, strided<[16777216, 4096, 1], offset: ?>>, memref<32x256x4096xf32> -> tensor<32x256x4096xf32>
          %45 = linalg.fill ins(%cst_1 : f32) outs(%19 : tensor<32x32x4096xf32>) -> tensor<32x32x4096xf32>
          %46 = linalg.batch_matmul ins(%38, %44 : tensor<32x32x256xf32>, tensor<32x256x4096xf32>) outs(%45 : tensor<32x32x4096xf32>) -> tensor<32x32x4096xf32>
          %47 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%46, %arg9, %41 : tensor<32x32x4096xf32>, tensor<32x32x4096xf32>, tensor<32x32xf32>) outs(%arg9 : tensor<32x32x4096xf32>) {
          ^bb0(%in: f32, %in_10: f32, %in_11: f32, %out: f32):
            %49 = arith.mulf %in_10, %in_11 : f32
            %50 = arith.addf %in, %49 : f32
            linalg.yield %50 : f32
          } -> tensor<32x32x4096xf32>
          %48 = linalg.copy ins(%37 : tensor<32x32xf32>) outs(%arg7 : tensor<32x32xf32>) -> tensor<32x32xf32>
          affine.yield %48, %42, %47 : tensor<32x32xf32>, tensor<32x32xf32>, tensor<32x32x4096xf32>
        }
        %28 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%27#2, %27#1 : tensor<32x32x4096xf32>, tensor<32x32xf32>) outs(%15 : tensor<32x32x4096xf32>) {
        ^bb0(%in: f32, %in_6: f32, %out: f32):
          %30 = arith.divf %in, %in_6 : f32
          linalg.yield %30 : f32
        } -> tensor<32x32x4096xf32>
        %c0_4 = arith.constant 0 : index
        %29 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 16777216 + d1 * 4096 + d2)>(%20, %21, %c0_4)
        %reinterpret_cast_5 = memref.reinterpret_cast %arg3 to offset: [%29], sizes: [32, 32, 4096], strides: [16777216, 4096, 1] : memref<2x4096x4096xf32> to memref<32x32x4096xf32, strided<[16777216, 4096, 1], offset: ?>>
        loom.copy_from_tensor %28, %reinterpret_cast_5 : tensor<32x32x4096xf32>, memref<32x32x4096xf32, strided<[16777216, 4096, 1], offset: ?>>
      }
      return
    }
  }
}
