#map = affine_map<(d0, d1) -> (d0, d1)>
module {
  func.func @matmul_kernel(%arg0: memref<*xf32> {tt.divisibility = 16 : i32}, %arg1: memref<*xf32> {tt.divisibility = 16 : i32}, %arg2: memref<*xf32> {tt.divisibility = 16 : i32}, %arg3: i32 {tt.divisibility = 16 : i32}, %arg4: i32 {tt.divisibility = 16 : i32}, %arg5: i32 {tt.divisibility = 16 : i32}, %arg6: i32, %arg7: i32, %arg8: i32, %arg9: i32, %arg10: i32, %arg11: i32) {
    %cst = arith.constant 0.000000e+00 : f32
    %c1_i32 = arith.constant 1 : i32
    %c31_i32 = arith.constant 31 : i32
    %c512 = arith.constant 512 : index
    %c0_i32 = arith.constant 0 : i32
    %c32_i32 = arith.constant 32 : i32
    %alloc = memref.alloc() {alignment = 64 : i64} : memref<32x32xf32>
    linalg.fill ins(%cst : f32) outs(%alloc : memref<32x32xf32>)
    %0 = arith.muli %arg9, %c32_i32 : i32
    %1 = arith.muli %arg10, %c32_i32 : i32
    %2 = arith.addi %arg5, %c31_i32 : i32
    %3 = arith.divsi %2, %c32_i32 : i32
    %4 = arith.index_cast %0 : i32 to index
    %5 = arith.index_cast %1 : i32 to index
    %alloc_0 = memref.alloc() {alignment = 64 : i64} : memref<32x32xf32>
    memref.copy %alloc, %alloc_0 : memref<32x32xf32> to memref<32x32xf32>
    %6 = scf.for %arg12 = %c0_i32 to %3 step %c1_i32 iter_args(%arg13 = %alloc_0) -> (memref<32x32xf32>)  : i32 {
      %9 = arith.muli %arg12, %c32_i32 : i32
      %10 = arith.index_cast %9 : i32 to index
      %11 = arith.muli %4, %c512 : index
      %12 = arith.addi %11, %10 : index
      %reinterpret_cast_1 = memref.reinterpret_cast %arg0 to offset: [%12], sizes: [32, 32], strides: [512, 1] : memref<*xf32> to memref<32x32xf32, strided<[512, 1], offset: ?>>
      %alloc_2 = memref.alloc() : memref<32x32xf32>
      memref.copy %reinterpret_cast_1, %alloc_2 : memref<32x32xf32, strided<[512, 1], offset: ?>> to memref<32x32xf32>
      %13 = arith.muli %10, %c512 : index
      %14 = arith.addi %13, %5 : index
      %reinterpret_cast_3 = memref.reinterpret_cast %arg1 to offset: [%14], sizes: [32, 32], strides: [512, 1] : memref<*xf32> to memref<32x32xf32, strided<[512, 1], offset: ?>>
      %alloc_4 = memref.alloc() : memref<32x32xf32>
      memref.copy %reinterpret_cast_3, %alloc_4 : memref<32x32xf32, strided<[512, 1], offset: ?>> to memref<32x32xf32>
      %alloc_5 = memref.alloc() {alignment = 64 : i64} : memref<32x32xf32>
      memref.copy %alloc, %alloc_5 : memref<32x32xf32> to memref<32x32xf32>
      linalg.matmul ins(%alloc_2, %alloc_4 : memref<32x32xf32>, memref<32x32xf32>) outs(%alloc_5 : memref<32x32xf32>)
      linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel", "parallel"]} ins(%arg13, %alloc_5 : memref<32x32xf32>, memref<32x32xf32>) outs(%arg13 : memref<32x32xf32>) {
      ^bb0(%in: f32, %in_6: f32, %out: f32):
        %15 = arith.addf %in, %in_6 : f32
        linalg.yield %15 : f32
      }
      scf.yield %arg13 : memref<32x32xf32>
    }
    %7 = arith.muli %4, %c512 : index
    %8 = arith.addi %7, %5 : index
    %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%8], sizes: [32, 32], strides: [512, 1] : memref<*xf32> to memref<32x32xf32, strided<[512, 1], offset: ?>>
    memref.copy %6, %reinterpret_cast : memref<32x32xf32> to memref<32x32xf32, strided<[512, 1], offset: ?>>
    return
  }
}

