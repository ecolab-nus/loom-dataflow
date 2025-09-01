#map = affine_map<(d0, d1) -> (d0 + 1, d1)>
#map1 = affine_map<(d0, d1) -> (d0, d1 + 1)>
#map2 = affine_map<(d0, d1) -> (d0 * 8 + d1)>
#map3 = affine_map<(d0, d1) -> (d0, d1)>
module {
  %0 = df.spatial_dim "x", 8
  %1 = df.spatial_dim "y", 8
  %2 = "df.interconnects"(%0, %1) <{map = #map}> : (index, index) -> !df.interconnect
  %3 = "df.interconnects"(%0, %1) <{map = #map1}> : (index, index) -> !df.interconnect
  func.func @matmul_kernel__g0sd0d1_g1unused_g2unused(%arg0: memref<*xf32> {tt.divisibility = 16 : i32}, %arg1: memref<*xf32> {tt.divisibility = 16 : i32}, %arg2: memref<*xf32> {tt.divisibility = 16 : i32}, %arg3: i32 {tt.divisibility = 16 : i32}, %arg4: i32 {tt.divisibility = 16 : i32}, %arg5: i32 {tt.divisibility = 16 : i32}, %arg6: i32 {tt.divisibility = 16 : i32}, %arg7: i32 {tt.divisibility = 16 : i32}, %arg8: i32 {tt.divisibility = 16 : i32}, %arg9: i32, %arg10: i32) attributes {tmd.grid_to_spatial = [[0, 1], [], []], tmd.grid_used = [true, false, false], tmd.spatial_dim_names = ["x", "y"], tmd.spatial_dim_sizes = [8, 8]} {
    %4 = arith.index_cast %arg9 : i32 to index
    %5 = arith.index_cast %arg10 : i32 to index
    %c0_i32 = arith.constant 0 : i32
    %c64_i32 = arith.constant 64 : i32
    %6 = affine.apply #map2(%4, %5)
    %7 = arith.index_cast %6 : index to i32
    %c1_i32 = arith.constant 1 : i32
    %c1_i32_0 = arith.constant 1 : i32
    %cst = arith.constant 0.000000e+00 : f32
    %c1_i32_1 = arith.constant 1 : i32
    %c63_i32 = arith.constant 63 : i32
    %c31_i32 = arith.constant 31 : i32
    %c64 = arith.constant 64 : index
    %c0_i32_2 = arith.constant 0 : i32
    %c32_i32 = arith.constant 32 : i32
    %c64_i32_3 = arith.constant 64 : i32
    %c8_i32 = arith.constant 8 : i32
    %8 = tensor.empty() : tensor<64x64xf32>
    %9 = linalg.fill ins(%cst : f32) outs(%8 : tensor<64x64xf32>) -> tensor<64x64xf32>
    %10 = arith.addi %arg4, %c63_i32 : i32
    %11 = arith.divsi %10, %c64_i32_3 : i32
    %12 = arith.muli %11, %c8_i32 : i32
    %13 = arith.divsi %7, %12 : i32
    %14 = arith.muli %13, %c8_i32 : i32
    %15 = arith.remsi %7, %c8_i32 : i32
    %16 = arith.addi %14, %15 : i32
    %17 = arith.divsi %7, %c8_i32 : i32
    %18 = arith.remsi %17, %11 : i32
    %19 = arith.muli %16, %c64_i32_3 : i32
    %20 = arith.muli %18, %c64_i32_3 : i32
    %21 = arith.addi %arg5, %c31_i32 : i32
    %22 = arith.divsi %21, %c32_i32 : i32
    %23 = arith.index_cast %arg6 : i32 to index
    %24 = arith.index_cast %19 : i32 to index
    %25 = arith.index_cast %arg7 : i32 to index
    %26 = arith.index_cast %20 : i32 to index
    %27 = scf.for %arg11 = %c0_i32_2 to %22 step %c1_i32_1 iter_args(%arg12 = %9) -> (tensor<64x64xf32>)  : i32 {
      %45 = arith.muli %arg11, %c32_i32 : i32
      %46 = arith.index_cast %45 : i32 to index
      %47 = arith.muli %24, %23 : index
      %48 = arith.addi %47, %46 : index
      %reinterpret_cast_4 = memref.reinterpret_cast %arg0 to offset: [%48], sizes: [64, 32], strides: [%23, 1] : memref<*xf32> to memref<64x32xf32, strided<[?, 1], offset: ?>>
      %alloc = memref.alloc() : memref<64x32xf32>
      memref.copy %reinterpret_cast_4, %alloc : memref<64x32xf32, strided<[?, 1], offset: ?>> to memref<64x32xf32>
      %49 = bufferization.to_tensor %alloc restrict writable : memref<64x32xf32> to tensor<64x32xf32>
      %50 = arith.muli %46, %25 : index
      %51 = arith.addi %50, %26 : index
      %reinterpret_cast_5 = memref.reinterpret_cast %arg1 to offset: [%51], sizes: [32, 64], strides: [%25, 1] : memref<*xf32> to memref<32x64xf32, strided<[?, 1], offset: ?>>
      %alloc_6 = memref.alloc() : memref<32x64xf32>
      memref.copy %reinterpret_cast_5, %alloc_6 : memref<32x64xf32, strided<[?, 1], offset: ?>> to memref<32x64xf32>
      %52 = bufferization.to_tensor %alloc_6 restrict writable : memref<32x64xf32> to tensor<32x64xf32>
      %53 = linalg.matmul ins(%49, %52 : tensor<64x32xf32>, tensor<32x64xf32>) outs(%9 : tensor<64x64xf32>) -> tensor<64x64xf32>
      %54 = linalg.generic {indexing_maps = [#map3, #map3, #map3], iterator_types = ["parallel", "parallel"]} ins(%arg12, %53 : tensor<64x64xf32>, tensor<64x64xf32>) outs(%arg12 : tensor<64x64xf32>) {
      ^bb0(%in: f32, %in_7: f32, %out: f32):
        %55 = arith.addf %in, %in_7 : f32
        linalg.yield %55 : f32
      } -> tensor<64x64xf32>
      scf.yield %54 : tensor<64x64xf32>
    }
    %28 = arith.index_cast %arg8 : i32 to index
    %29 = arith.muli %24, %28 : index
    %30 = arith.addi %29, %26 : index
    %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%30], sizes: [64, 64], strides: [%28, 1] : memref<*xf32> to memref<64x64xf32, strided<[?, 1], offset: ?>>
    %31 = arith.addi %24, %c64 : index
    %32 = arith.index_cast %arg3 : i32 to index
    %33 = arith.minsi %31, %32 : index
    %34 = arith.maxsi %33, %24 : index
    %35 = arith.subi %34, %24 : index
    %36 = arith.minsi %35, %c64 : index
    %37 = arith.addi %26, %c64 : index
    %38 = arith.index_cast %arg4 : i32 to index
    %39 = arith.minsi %37, %38 : index
    %40 = arith.maxsi %39, %26 : index
    %41 = arith.subi %40, %26 : index
    %42 = arith.minsi %41, %c64 : index
    %43 = arith.minsi %36, %c64 : index
    %44 = arith.minsi %42, %c64 : index
    %extracted_slice = tensor.extract_slice %27[0, 0] [%43, %44] [1, 1] : tensor<64x64xf32> to tensor<?x?xf32>
    %subview = memref.subview %reinterpret_cast[0, 0] [%43, %44] [1, 1] : memref<64x64xf32, strided<[?, 1], offset: ?>> to memref<?x?xf32, strided<[?, 1], offset: ?>>
    bufferization.materialize_in_destination %extracted_slice in writable %subview : (tensor<?x?xf32>, memref<?x?xf32, strided<[?, 1], offset: ?>>) -> ()
    return
  }
  func.func @matmul_kernel__g0sd1d0_g1unused_g2unused(%arg0: memref<*xf32> {tt.divisibility = 16 : i32}, %arg1: memref<*xf32> {tt.divisibility = 16 : i32}, %arg2: memref<*xf32> {tt.divisibility = 16 : i32}, %arg3: i32 {tt.divisibility = 16 : i32}, %arg4: i32 {tt.divisibility = 16 : i32}, %arg5: i32 {tt.divisibility = 16 : i32}, %arg6: i32 {tt.divisibility = 16 : i32}, %arg7: i32 {tt.divisibility = 16 : i32}, %arg8: i32 {tt.divisibility = 16 : i32}, %arg9: i32, %arg10: i32) attributes {tmd.grid_to_spatial = [[1, 0], [], []], tmd.grid_used = [true, false, false], tmd.spatial_dim_names = ["x", "y"], tmd.spatial_dim_sizes = [8, 8]} {
    %4 = arith.index_cast %arg9 : i32 to index
    %5 = arith.index_cast %arg10 : i32 to index
    %c0_i32 = arith.constant 0 : i32
    %c64_i32 = arith.constant 64 : i32
    %6 = affine.apply #map2(%5, %4)
    %7 = arith.index_cast %6 : index to i32
    %c1_i32 = arith.constant 1 : i32
    %c1_i32_0 = arith.constant 1 : i32
    %cst = arith.constant 0.000000e+00 : f32
    %c1_i32_1 = arith.constant 1 : i32
    %c63_i32 = arith.constant 63 : i32
    %c31_i32 = arith.constant 31 : i32
    %c64 = arith.constant 64 : index
    %c0_i32_2 = arith.constant 0 : i32
    %c32_i32 = arith.constant 32 : i32
    %c64_i32_3 = arith.constant 64 : i32
    %c8_i32 = arith.constant 8 : i32
    %8 = tensor.empty() : tensor<64x64xf32>
    %9 = linalg.fill ins(%cst : f32) outs(%8 : tensor<64x64xf32>) -> tensor<64x64xf32>
    %10 = arith.addi %arg4, %c63_i32 : i32
    %11 = arith.divsi %10, %c64_i32_3 : i32
    %12 = arith.muli %11, %c8_i32 : i32
    %13 = arith.divsi %7, %12 : i32
    %14 = arith.muli %13, %c8_i32 : i32
    %15 = arith.remsi %7, %c8_i32 : i32
    %16 = arith.addi %14, %15 : i32
    %17 = arith.divsi %7, %c8_i32 : i32
    %18 = arith.remsi %17, %11 : i32
    %19 = arith.muli %16, %c64_i32_3 : i32
    %20 = arith.muli %18, %c64_i32_3 : i32
    %21 = arith.addi %arg5, %c31_i32 : i32
    %22 = arith.divsi %21, %c32_i32 : i32
    %23 = arith.index_cast %arg6 : i32 to index
    %24 = arith.index_cast %19 : i32 to index
    %25 = arith.index_cast %arg7 : i32 to index
    %26 = arith.index_cast %20 : i32 to index
    %27 = scf.for %arg11 = %c0_i32_2 to %22 step %c1_i32_1 iter_args(%arg12 = %9) -> (tensor<64x64xf32>)  : i32 {
      %45 = arith.muli %arg11, %c32_i32 : i32
      %46 = arith.index_cast %45 : i32 to index
      %47 = arith.muli %24, %23 : index
      %48 = arith.addi %47, %46 : index
      %reinterpret_cast_4 = memref.reinterpret_cast %arg0 to offset: [%48], sizes: [64, 32], strides: [%23, 1] : memref<*xf32> to memref<64x32xf32, strided<[?, 1], offset: ?>>
      %alloc = memref.alloc() : memref<64x32xf32>
      memref.copy %reinterpret_cast_4, %alloc : memref<64x32xf32, strided<[?, 1], offset: ?>> to memref<64x32xf32>
      %49 = bufferization.to_tensor %alloc restrict writable : memref<64x32xf32> to tensor<64x32xf32>
      %50 = arith.muli %46, %25 : index
      %51 = arith.addi %50, %26 : index
      %reinterpret_cast_5 = memref.reinterpret_cast %arg1 to offset: [%51], sizes: [32, 64], strides: [%25, 1] : memref<*xf32> to memref<32x64xf32, strided<[?, 1], offset: ?>>
      %alloc_6 = memref.alloc() : memref<32x64xf32>
      memref.copy %reinterpret_cast_5, %alloc_6 : memref<32x64xf32, strided<[?, 1], offset: ?>> to memref<32x64xf32>
      %52 = bufferization.to_tensor %alloc_6 restrict writable : memref<32x64xf32> to tensor<32x64xf32>
      %53 = linalg.matmul ins(%49, %52 : tensor<64x32xf32>, tensor<32x64xf32>) outs(%9 : tensor<64x64xf32>) -> tensor<64x64xf32>
      %54 = linalg.generic {indexing_maps = [#map3, #map3, #map3], iterator_types = ["parallel", "parallel"]} ins(%arg12, %53 : tensor<64x64xf32>, tensor<64x64xf32>) outs(%arg12 : tensor<64x64xf32>) {
      ^bb0(%in: f32, %in_7: f32, %out: f32):
        %55 = arith.addf %in, %in_7 : f32
        linalg.yield %55 : f32
      } -> tensor<64x64xf32>
      scf.yield %54 : tensor<64x64xf32>
    }
    %28 = arith.index_cast %arg8 : i32 to index
    %29 = arith.muli %24, %28 : index
    %30 = arith.addi %29, %26 : index
    %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%30], sizes: [64, 64], strides: [%28, 1] : memref<*xf32> to memref<64x64xf32, strided<[?, 1], offset: ?>>
    %31 = arith.addi %24, %c64 : index
    %32 = arith.index_cast %arg3 : i32 to index
    %33 = arith.minsi %31, %32 : index
    %34 = arith.maxsi %33, %24 : index
    %35 = arith.subi %34, %24 : index
    %36 = arith.minsi %35, %c64 : index
    %37 = arith.addi %26, %c64 : index
    %38 = arith.index_cast %arg4 : i32 to index
    %39 = arith.minsi %37, %38 : index
    %40 = arith.maxsi %39, %26 : index
    %41 = arith.subi %40, %26 : index
    %42 = arith.minsi %41, %c64 : index
    %43 = arith.minsi %36, %c64 : index
    %44 = arith.minsi %42, %c64 : index
    %extracted_slice = tensor.extract_slice %27[0, 0] [%43, %44] [1, 1] : tensor<64x64xf32> to tensor<?x?xf32>
    %subview = memref.subview %reinterpret_cast[0, 0] [%43, %44] [1, 1] : memref<64x64xf32, strided<[?, 1], offset: ?>> to memref<?x?xf32, strided<[?, 1], offset: ?>>
    bufferization.materialize_in_destination %extracted_slice in writable %subview : (tensor<?x?xf32>, memref<?x?xf32, strided<[?, 1], offset: ?>>) -> ()
    return
  }
}

