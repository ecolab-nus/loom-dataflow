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
        %4 = loom.alloc [64, 64, 4096] on @L1 : memref<64x64x4096xf32>
        %5 = loom.init_tensor %4[64, 64, 4096] : memref<64x64x4096xf32> -> tensor<64x64x4096xf32>
        %6 = loom.alloc [64, 64, 4096] on @L1 : memref<64x64x4096xf32>
        %7 = loom.alloc [64, 64, 4096] on @L1 : memref<64x64x4096xf32>
        %8 = loom.init_tensor %7[64, 64, 4096] : memref<64x64x4096xf32> -> tensor<64x64x4096xf32>
        %9 = loom.alloc [64, 64, 4096] on @L1 : memref<64x64x4096xf32>
        %10 = loom.alloc [64, 64] on @L1 : memref<64x64xf32>
        %11 = loom.init_tensor %10[64, 64] : memref<64x64xf32> -> tensor<64x64xf32>
        %12 = loom.alloc [64, 64] on @L1 : memref<64x64xf32>
        %13 = loom.init_tensor %12[64, 64] : memref<64x64xf32> -> tensor<64x64xf32>
        %14 = loom.alloc [64, 64] on @L1 : memref<64x64xf32>
        %15 = loom.init_tensor %14[64, 64] : memref<64x64xf32> -> tensor<64x64xf32>
        %16 = loom.alloc [64, 64] on @L1 : memref<64x64xf32>
        %17 = loom.init_tensor %16[64, 64] : memref<64x64xf32> -> tensor<64x64xf32>
        %18 = loom.alloc [64, 64] on @L1 : memref<64x64xf32>
        %19 = loom.init_tensor %18[64, 64] : memref<64x64xf32> -> tensor<64x64xf32>
        %20 = linalg.fill ins(%cst_3 : f32) outs(%11 : tensor<64x64xf32>) -> tensor<64x64xf32>
        %21 = linalg.fill ins(%cst_3 : f32) outs(%15 : tensor<64x64xf32>) -> tensor<64x64xf32>
        %22 = linalg.fill ins(%cst_2 : f32) outs(%13 : tensor<64x64xf32>) -> tensor<64x64xf32>
        %23 = linalg.fill ins(%cst_1 : f32) outs(%5 : tensor<64x64x4096xf32>) -> tensor<64x64x4096xf32>
        %24 = linalg.fill ins(%cst_1 : f32) outs(%8 : tensor<64x64x4096xf32>) -> tensor<64x64x4096xf32>
        %25 = arith.muli %arg4, %c64 : index
        %26 = arith.muli %arg5, %c64 : index
        %c0 = arith.constant 0 : index
        %27 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 16777216 + d1 * 4096 + d2)>(%25, %26, %c0)
        %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%27], sizes: [64, 64, 4096], strides: [16777216, 4096, 1] : memref<2x4096x4096xf32> to memref<64x64x4096xf32, strided<[16777216, 4096, 1], offset: ?>>
        %28 = loom.copy_to_tensor %reinterpret_cast, %6, interconnect : [], broadcast : [1, 1] : memref<64x64x4096xf32, strided<[16777216, 4096, 1], offset: ?>>, memref<64x64x4096xf32> -> tensor<64x64x4096xf32>
        %29:3 = affine.for %arg6 = 0 to 64 iter_args(%arg7 = %21, %arg8 = %22, %arg9 = %24) -> (tensor<64x64xf32>, tensor<64x64xf32>, tensor<64x64x4096xf32>) {
          %36 = arith.muli %arg6, %c64 : index
          %c0_6 = arith.constant 0 : index
          %37 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 16777216 + d1 * 4096 + d2)>(%25, %c0_6, %36)
          %reinterpret_cast_7 = memref.reinterpret_cast %arg0 to offset: [%37], sizes: [64, 4096, 64], strides: [16777216, 4096, 1] : memref<2x4096x4096xf32> to memref<64x4096x64xf32, strided<[16777216, 4096, 1], offset: ?>>
          %38 = loom.copy_to_tensor %reinterpret_cast_7, %3, interconnect : [], broadcast : [1, 1] : memref<64x4096x64xf32, strided<[16777216, 4096, 1], offset: ?>>, memref<64x4096x64xf32> -> tensor<64x4096x64xf32>
          %39 = linalg.fill ins(%cst_1 : f32) outs(%2 : tensor<64x64x64xf32>) -> tensor<64x64x64xf32>
          %40 = linalg.batch_matmul ins(%28, %38 : tensor<64x64x4096xf32>, tensor<64x4096x64xf32>) outs(%39 : tensor<64x64x64xf32>) -> tensor<64x64x64xf32>
          %41 = loom.pb_anchor %40, %1 : tensor<64x64x64xf32>, memref<64x64x64xf32> -> tensor<64x64x64xf32>
          %42 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%41 : tensor<64x64x64xf32>) outs(%20 : tensor<64x64xf32>) {
          ^bb0(%in: f32, %out: f32):
            %58 = arith.maximumf %in, %out : f32
            linalg.yield %58 : f32
          } -> tensor<64x64xf32>
          %43 = loom.pb_anchor %42, %16 : tensor<64x64xf32>, memref<64x64xf32> -> tensor<64x64xf32>
          %44 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%arg7, %43 : tensor<64x64xf32>, tensor<64x64xf32>) outs(%15 : tensor<64x64xf32>) {
          ^bb0(%in: f32, %in_10: f32, %out: f32):
            %58 = arith.truncf %cst_0 : f64 to f32
            %59 = arith.mulf %in_10, %58 : f32
            %60 = arith.cmpf ogt, %in, %59 : f32
            %61 = arith.select %60, %in, %59 : f32
            linalg.yield %61 : f32
          } -> tensor<64x64xf32>
          %45 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%41, %44 : tensor<64x64x64xf32>, tensor<64x64xf32>) outs(%2 : tensor<64x64x64xf32>) {
          ^bb0(%in: f32, %in_10: f32, %out: f32):
            %58 = arith.truncf %cst_0 : f64 to f32
            %59 = arith.mulf %in, %58 : f32
            %60 = arith.subf %59, %in_10 : f32
            %61 = math.powf %cst, %60 : f32
            linalg.yield %61 : f32
          } -> tensor<64x64x64xf32>
          %46 = loom.pb_anchor %45, %1 : tensor<64x64x64xf32>, memref<64x64x64xf32> -> tensor<64x64x64xf32>
          %47 = linalg.fill ins(%cst_1 : f32) outs(%17 : tensor<64x64xf32>) -> tensor<64x64xf32>
          %48 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%46 : tensor<64x64x64xf32>) outs(%47 : tensor<64x64xf32>) {
          ^bb0(%in: f32, %out: f32):
            %58 = arith.addf %in, %out : f32
            linalg.yield %58 : f32
          } -> tensor<64x64xf32>
          %49 = loom.pb_anchor %48, %16 : tensor<64x64xf32>, memref<64x64xf32> -> tensor<64x64xf32>
          %50 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%arg7, %44 : tensor<64x64xf32>, tensor<64x64xf32>) outs(%19 : tensor<64x64xf32>) {
          ^bb0(%in: f32, %in_10: f32, %out: f32):
            %58 = arith.subf %in, %in_10 : f32
            %59 = math.powf %cst, %58 : f32
            linalg.yield %59 : f32
          } -> tensor<64x64xf32>
          %51 = loom.pb_anchor %50, %18 : tensor<64x64xf32>, memref<64x64xf32> -> tensor<64x64xf32>
          %52 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%arg8, %51, %49 : tensor<64x64xf32>, tensor<64x64xf32>, tensor<64x64xf32>) outs(%13 : tensor<64x64xf32>) {
          ^bb0(%in: f32, %in_10: f32, %in_11: f32, %out: f32):
            %58 = arith.mulf %in, %in_10 : f32
            %59 = arith.addf %58, %in_11 : f32
            linalg.yield %59 : f32
          } -> tensor<64x64xf32>
          %c0_8 = arith.constant 0 : index
          %53 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 16777216 + d1 * 4096 + d2)>(%25, %36, %c0_8)
          %reinterpret_cast_9 = memref.reinterpret_cast %arg1 to offset: [%53], sizes: [64, 64, 4096], strides: [16777216, 4096, 1] : memref<2x4096x4096xf32> to memref<64x64x4096xf32, strided<[16777216, 4096, 1], offset: ?>>
          %54 = loom.copy_to_tensor %reinterpret_cast_9, %0, interconnect : [], broadcast : [1, 1] : memref<64x64x4096xf32, strided<[16777216, 4096, 1], offset: ?>>, memref<64x64x4096xf32> -> tensor<64x64x4096xf32>
          %55 = linalg.batch_matmul ins(%46, %54 : tensor<64x64x64xf32>, tensor<64x64x4096xf32>) outs(%23 : tensor<64x64x4096xf32>) -> tensor<64x64x4096xf32>
          %56 = loom.pb_anchor %55, %9 : tensor<64x64x4096xf32>, memref<64x64x4096xf32> -> tensor<64x64x4096xf32>
          %57 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%56, %arg9, %51 : tensor<64x64x4096xf32>, tensor<64x64x4096xf32>, tensor<64x64xf32>) outs(%8 : tensor<64x64x4096xf32>) {
          ^bb0(%in: f32, %in_10: f32, %in_11: f32, %out: f32):
            %58 = arith.mulf %in_10, %in_11 : f32
            %59 = arith.addf %in, %58 : f32
            linalg.yield %59 : f32
          } -> tensor<64x64x4096xf32>
          affine.yield %44, %52, %57 : tensor<64x64xf32>, tensor<64x64xf32>, tensor<64x64x4096xf32>
        }
        %30 = loom.pb_anchor %29#0, %14 : tensor<64x64xf32>, memref<64x64xf32> -> tensor<64x64xf32>
        %31 = loom.pb_anchor %29#1, %12 : tensor<64x64xf32>, memref<64x64xf32> -> tensor<64x64xf32>
        %32 = loom.pb_anchor %29#2, %7 : tensor<64x64x4096xf32>, memref<64x64x4096xf32> -> tensor<64x64x4096xf32>
        %33 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%32, %31 : tensor<64x64x4096xf32>, tensor<64x64xf32>) outs(%5 : tensor<64x64x4096xf32>) {
        ^bb0(%in: f32, %in_6: f32, %out: f32):
          %36 = arith.divf %in, %in_6 : f32
          linalg.yield %36 : f32
        } -> tensor<64x64x4096xf32>
        %34 = loom.pb_anchor %33, %4 : tensor<64x64x4096xf32>, memref<64x64x4096xf32> -> tensor<64x64x4096xf32>
        %c0_4 = arith.constant 0 : index
        %35 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 16777216 + d1 * 4096 + d2)>(%25, %26, %c0_4)
        %reinterpret_cast_5 = memref.reinterpret_cast %arg3 to offset: [%35], sizes: [64, 64, 4096], strides: [16777216, 4096, 1] : memref<2x4096x4096xf32> to memref<64x64x4096xf32, strided<[16777216, 4096, 1], offset: ?>>
        loom.copy_from_tensor %34, %reinterpret_cast_5 : tensor<64x64x4096xf32>, memref<64x64x4096xf32, strided<[16777216, 4096, 1], offset: ?>>
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
        %4 = loom.alloc [32, 32, 4096] on @L1 : memref<32x32x4096xf32>
        %5 = loom.init_tensor %4[32, 32, 4096] : memref<32x32x4096xf32> -> tensor<32x32x4096xf32>
        %6 = loom.alloc [32, 32, 4096] on @L1 : memref<32x32x4096xf32>
        %7 = loom.alloc [32, 32, 4096] on @L1 : memref<32x32x4096xf32>
        %8 = loom.init_tensor %7[32, 32, 4096] : memref<32x32x4096xf32> -> tensor<32x32x4096xf32>
        %9 = loom.alloc [32, 32, 4096] on @L1 : memref<32x32x4096xf32>
        %10 = loom.alloc [32, 32] on @L1 : memref<32x32xf32>
        %11 = loom.init_tensor %10[32, 32] : memref<32x32xf32> -> tensor<32x32xf32>
        %12 = loom.alloc [32, 32] on @L1 : memref<32x32xf32>
        %13 = loom.init_tensor %12[32, 32] : memref<32x32xf32> -> tensor<32x32xf32>
        %14 = loom.alloc [32, 32] on @L1 : memref<32x32xf32>
        %15 = loom.init_tensor %14[32, 32] : memref<32x32xf32> -> tensor<32x32xf32>
        %16 = loom.alloc [32, 32] on @L1 : memref<32x32xf32>
        %17 = loom.init_tensor %16[32, 32] : memref<32x32xf32> -> tensor<32x32xf32>
        %18 = loom.alloc [32, 32] on @L1 : memref<32x32xf32>
        %19 = loom.init_tensor %18[32, 32] : memref<32x32xf32> -> tensor<32x32xf32>
        %20 = linalg.fill ins(%cst_3 : f32) outs(%11 : tensor<32x32xf32>) -> tensor<32x32xf32>
        %21 = linalg.fill ins(%cst_3 : f32) outs(%15 : tensor<32x32xf32>) -> tensor<32x32xf32>
        %22 = linalg.fill ins(%cst_2 : f32) outs(%13 : tensor<32x32xf32>) -> tensor<32x32xf32>
        %23 = linalg.fill ins(%cst_1 : f32) outs(%5 : tensor<32x32x4096xf32>) -> tensor<32x32x4096xf32>
        %24 = linalg.fill ins(%cst_1 : f32) outs(%8 : tensor<32x32x4096xf32>) -> tensor<32x32x4096xf32>
        %25 = arith.muli %arg4, %c32 : index
        %26 = arith.muli %arg5, %c32 : index
        %c0 = arith.constant 0 : index
        %27 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 16777216 + d1 * 4096 + d2)>(%25, %26, %c0)
        %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%27], sizes: [32, 32, 4096], strides: [16777216, 4096, 1] : memref<2x4096x4096xf32> to memref<32x32x4096xf32, strided<[16777216, 4096, 1], offset: ?>>
        %28 = loom.copy_to_tensor %reinterpret_cast, %6, interconnect : [], broadcast : [1, 1] : memref<32x32x4096xf32, strided<[16777216, 4096, 1], offset: ?>>, memref<32x32x4096xf32> -> tensor<32x32x4096xf32>
        %29:3 = affine.for %arg6 = 0 to 16 iter_args(%arg7 = %21, %arg8 = %22, %arg9 = %24) -> (tensor<32x32xf32>, tensor<32x32xf32>, tensor<32x32x4096xf32>) {
          %36 = arith.muli %arg6, %c256 : index
          %c0_6 = arith.constant 0 : index
          %37 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 16777216 + d1 * 4096 + d2)>(%25, %c0_6, %36)
          %reinterpret_cast_7 = memref.reinterpret_cast %arg0 to offset: [%37], sizes: [32, 4096, 256], strides: [16777216, 4096, 1] : memref<2x4096x4096xf32> to memref<32x4096x256xf32, strided<[16777216, 4096, 1], offset: ?>>
          %38 = loom.copy_to_tensor %reinterpret_cast_7, %3, interconnect : [], broadcast : [1, 1] : memref<32x4096x256xf32, strided<[16777216, 4096, 1], offset: ?>>, memref<32x4096x256xf32> -> tensor<32x4096x256xf32>
          %39 = linalg.fill ins(%cst_1 : f32) outs(%2 : tensor<32x32x256xf32>) -> tensor<32x32x256xf32>
          %40 = linalg.batch_matmul ins(%28, %38 : tensor<32x32x4096xf32>, tensor<32x4096x256xf32>) outs(%39 : tensor<32x32x256xf32>) -> tensor<32x32x256xf32>
          %41 = loom.pb_anchor %40, %1 : tensor<32x32x256xf32>, memref<32x32x256xf32> -> tensor<32x32x256xf32>
          %42 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%41 : tensor<32x32x256xf32>) outs(%20 : tensor<32x32xf32>) {
          ^bb0(%in: f32, %out: f32):
            %58 = arith.maximumf %in, %out : f32
            linalg.yield %58 : f32
          } -> tensor<32x32xf32>
          %43 = loom.pb_anchor %42, %16 : tensor<32x32xf32>, memref<32x32xf32> -> tensor<32x32xf32>
          %44 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%arg7, %43 : tensor<32x32xf32>, tensor<32x32xf32>) outs(%15 : tensor<32x32xf32>) {
          ^bb0(%in: f32, %in_10: f32, %out: f32):
            %58 = arith.truncf %cst_0 : f64 to f32
            %59 = arith.mulf %in_10, %58 : f32
            %60 = arith.cmpf ogt, %in, %59 : f32
            %61 = arith.select %60, %in, %59 : f32
            linalg.yield %61 : f32
          } -> tensor<32x32xf32>
          %45 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%41, %44 : tensor<32x32x256xf32>, tensor<32x32xf32>) outs(%2 : tensor<32x32x256xf32>) {
          ^bb0(%in: f32, %in_10: f32, %out: f32):
            %58 = arith.truncf %cst_0 : f64 to f32
            %59 = arith.mulf %in, %58 : f32
            %60 = arith.subf %59, %in_10 : f32
            %61 = math.powf %cst, %60 : f32
            linalg.yield %61 : f32
          } -> tensor<32x32x256xf32>
          %46 = loom.pb_anchor %45, %1 : tensor<32x32x256xf32>, memref<32x32x256xf32> -> tensor<32x32x256xf32>
          %47 = linalg.fill ins(%cst_1 : f32) outs(%17 : tensor<32x32xf32>) -> tensor<32x32xf32>
          %48 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%46 : tensor<32x32x256xf32>) outs(%47 : tensor<32x32xf32>) {
          ^bb0(%in: f32, %out: f32):
            %58 = arith.addf %in, %out : f32
            linalg.yield %58 : f32
          } -> tensor<32x32xf32>
          %49 = loom.pb_anchor %48, %16 : tensor<32x32xf32>, memref<32x32xf32> -> tensor<32x32xf32>
          %50 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%arg7, %44 : tensor<32x32xf32>, tensor<32x32xf32>) outs(%19 : tensor<32x32xf32>) {
          ^bb0(%in: f32, %in_10: f32, %out: f32):
            %58 = arith.subf %in, %in_10 : f32
            %59 = math.powf %cst, %58 : f32
            linalg.yield %59 : f32
          } -> tensor<32x32xf32>
          %51 = loom.pb_anchor %50, %18 : tensor<32x32xf32>, memref<32x32xf32> -> tensor<32x32xf32>
          %52 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%arg8, %51, %49 : tensor<32x32xf32>, tensor<32x32xf32>, tensor<32x32xf32>) outs(%13 : tensor<32x32xf32>) {
          ^bb0(%in: f32, %in_10: f32, %in_11: f32, %out: f32):
            %58 = arith.mulf %in, %in_10 : f32
            %59 = arith.addf %58, %in_11 : f32
            linalg.yield %59 : f32
          } -> tensor<32x32xf32>
          %c0_8 = arith.constant 0 : index
          %53 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 16777216 + d1 * 4096 + d2)>(%25, %36, %c0_8)
          %reinterpret_cast_9 = memref.reinterpret_cast %arg1 to offset: [%53], sizes: [32, 256, 4096], strides: [16777216, 4096, 1] : memref<2x4096x4096xf32> to memref<32x256x4096xf32, strided<[16777216, 4096, 1], offset: ?>>
          %54 = loom.copy_to_tensor %reinterpret_cast_9, %0, interconnect : [], broadcast : [1, 1] : memref<32x256x4096xf32, strided<[16777216, 4096, 1], offset: ?>>, memref<32x256x4096xf32> -> tensor<32x256x4096xf32>
          %55 = linalg.batch_matmul ins(%46, %54 : tensor<32x32x256xf32>, tensor<32x256x4096xf32>) outs(%23 : tensor<32x32x4096xf32>) -> tensor<32x32x4096xf32>
          %56 = loom.pb_anchor %55, %9 : tensor<32x32x4096xf32>, memref<32x32x4096xf32> -> tensor<32x32x4096xf32>
          %57 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%56, %arg9, %51 : tensor<32x32x4096xf32>, tensor<32x32x4096xf32>, tensor<32x32xf32>) outs(%8 : tensor<32x32x4096xf32>) {
          ^bb0(%in: f32, %in_10: f32, %in_11: f32, %out: f32):
            %58 = arith.mulf %in_10, %in_11 : f32
            %59 = arith.addf %in, %58 : f32
            linalg.yield %59 : f32
          } -> tensor<32x32x4096xf32>
          affine.yield %44, %52, %57 : tensor<32x32xf32>, tensor<32x32xf32>, tensor<32x32x4096xf32>
        }
        %30 = loom.pb_anchor %29#0, %14 : tensor<32x32xf32>, memref<32x32xf32> -> tensor<32x32xf32>
        %31 = loom.pb_anchor %29#1, %12 : tensor<32x32xf32>, memref<32x32xf32> -> tensor<32x32xf32>
        %32 = loom.pb_anchor %29#2, %7 : tensor<32x32x4096xf32>, memref<32x32x4096xf32> -> tensor<32x32x4096xf32>
        %33 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%32, %31 : tensor<32x32x4096xf32>, tensor<32x32xf32>) outs(%5 : tensor<32x32x4096xf32>) {
        ^bb0(%in: f32, %in_6: f32, %out: f32):
          %36 = arith.divf %in, %in_6 : f32
          linalg.yield %36 : f32
        } -> tensor<32x32x4096xf32>
        %34 = loom.pb_anchor %33, %4 : tensor<32x32x4096xf32>, memref<32x32x4096xf32> -> tensor<32x32x4096xf32>
        %c0_4 = arith.constant 0 : index
        %35 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 16777216 + d1 * 4096 + d2)>(%25, %26, %c0_4)
        %reinterpret_cast_5 = memref.reinterpret_cast %arg3 to offset: [%35], sizes: [32, 32, 4096], strides: [16777216, 4096, 1] : memref<2x4096x4096xf32> to memref<32x32x4096xf32, strided<[16777216, 4096, 1], offset: ?>>
        loom.copy_from_tensor %34, %reinterpret_cast_5 : tensor<32x32x4096xf32>, memref<32x32x4096xf32, strided<[16777216, 4096, 1], offset: ?>>
      }
      return
    }
  }
}
