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
    func.func @attention__d0i0_d1i0__f01__d_d_d__BB1__BM64__BN128(%arg0: memref<32x128x4096xf32>, %arg1: memref<32x4096x128xf32>, %arg2: memref<32x4096x128xf32>, %arg3: memref<32x4096x128xf32>) {
      %c16384 = arith.constant 16384 : index
      %c8192 = arith.constant 8192 : index
      %c32 = arith.constant 32 : index
      %c524288 = arith.constant 524288 : index
      %cst = arith.constant 2.000000e+00 : f32
      %cst_0 = arith.constant 0.12751743074602467 : f64
      %cst_1 = arith.constant 0.000000e+00 : f32
      %cst_2 = arith.constant 1.000000e+00 : f32
      %cst_3 = arith.constant 0xFF800000 : f32
      %c64 = arith.constant 64 : index
      %c128 = arith.constant 128 : index
      %c0 = arith.constant 0 : index
      %c8 = arith.constant 8 : index
      %c1 = arith.constant 1 : index
      scf.parallel (%arg4, %arg5) = (%c0, %c0) to (%c8, %c8) step (%c1, %c1) {
        scf.for %arg6 = %c0 to %c64 step %c1 {
          %12 = arith.muli %arg4, %c8 overflow<nsw> : index
          %13 = arith.addi %12, %arg5 : index
          %14 = loom.alloc [1, 128, 128] on @L1 : memref<1x128x128xf32>
          %15 = loom.semaphore_take %14 : memref<1x128x128xf32> -> memref<1x128x128xf32>
          %16 = loom.alloc [1, 64, 128] on @L1 : memref<1x64x128xf32>
          %17 = loom.semaphore_take %16 : memref<1x64x128xf32> -> memref<1x64x128xf32>
          %18 = loom.alloc [1, 128, 128] on @L1 : memref<1x128x128xf32>
          %19 = loom.semaphore_take %18 : memref<1x128x128xf32> -> memref<1x128x128xf32>
          %20 = loom.alloc [1, 64] on @L1 : memref<1x64xf32>
          %21 = loom.semaphore_take %20 : memref<1x64xf32> -> memref<1x64xf32>
          %22 = loom.alloc [1, 64] on @L1 : memref<1x64xf32>
          %23 = loom.semaphore_take %22 : memref<1x64xf32> -> memref<1x64xf32>
          %24 = loom.alloc [1, 64] on @L1 : memref<1x64xf32>
          %25 = loom.semaphore_take %24 : memref<1x64xf32> -> memref<1x64xf32>
          %26 = loom.alloc [1, 64] on @L1 : memref<1x64xf32>
          %27 = loom.semaphore_take %26 : memref<1x64xf32> -> memref<1x64xf32>
          %28 = loom.alloc [1, 64] on @L1 : memref<1x64xf32>
          %29 = loom.semaphore_take %28 : memref<1x64xf32> -> memref<1x64xf32>
          %30 = loom.alloc [1, 64, 128] on @L1 : memref<1x64x128xf32>
          %31 = loom.semaphore_take %30 : memref<1x64x128xf32> -> memref<1x64x128xf32>
          %32 = loom.semaphore_take %30 : memref<1x64x128xf32> -> memref<1x64x128xf32>
          %33 = loom.alloc [1, 64, 128] on @L1 : memref<1x64x128xf32>
          %34 = loom.semaphore_take %33 : memref<1x64x128xf32> -> memref<1x64x128xf32>
          %35 = loom.alloc [1, 64, 128] on @L1 : memref<1x64x128xf32>
          %36 = loom.semaphore_take %35 : memref<1x64x128xf32> -> memref<1x64x128xf32>
          %37 = arith.muli %13, %c524288 overflow<nsw> : index
          %38 = arith.muli %arg6, %c8192 : index
          %39 = arith.addi %37, %38 : index
          %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%39], sizes: [1, 64, 128], strides: [524288, 128, 1] : memref<32x4096x128xf32> to memref<1x64x128xf32, strided<[524288, 128, 1], offset: ?>>
          loom.copy %reinterpret_cast, %32 src_mem_space @DRAM dst_mem_space @L1, interconnect : [], broadcast : [1, 1] : memref<1x64x128xf32, strided<[524288, 128, 1], offset: ?>>, memref<1x64x128xf32>
          linalg.fill ins(%cst_1 : f32) outs(%34 : memref<1x64x128xf32>)
          linalg.fill ins(%cst_2 : f32) outs(%21 : memref<1x64xf32>)
          linalg.fill ins(%cst_3 : f32) outs(%23 : memref<1x64xf32>)
          scf.for %arg7 = %c0 to %c32 step %c1 {
            %40 = arith.muli %arg7, %c128 : index
            %41 = arith.addi %37, %40 : index
            %reinterpret_cast_5 = memref.reinterpret_cast %arg0 to offset: [%41], sizes: [1, 128, 128], strides: [524288, 4096, 1] : memref<32x128x4096xf32> to memref<1x128x128xf32, strided<[524288, 4096, 1], offset: ?>>
            loom.copy %reinterpret_cast_5, %19 src_mem_space @DRAM dst_mem_space @L1, interconnect : [], broadcast : [1, 1] : memref<1x128x128xf32, strided<[524288, 4096, 1], offset: ?>>, memref<1x128x128xf32>
            linalg.fill ins(%cst_1 : f32) outs(%17 : memref<1x64x128xf32>)
            linalg.batch_matmul ins(%32, %19 : memref<1x64x128xf32>, memref<1x128x128xf32>) outs(%17 : memref<1x64x128xf32>)
            loom.semaphore_give %19 : memref<1x128x128xf32>
            linalg.fill ins(%cst_3 : f32) outs(%25 : memref<1x64xf32>)
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%17 : memref<1x64x128xf32>) outs(%25 : memref<1x64xf32>) {
            ^bb0(%in: f32, %out: f32):
              %44 = arith.maximumf %in, %out : f32
              linalg.yield %44 : f32
            }
            linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%23, %25 : memref<1x64xf32>, memref<1x64xf32>) outs(%25 : memref<1x64xf32>) {
            ^bb0(%in: f32, %in_7: f32, %out: f32):
              %44 = arith.truncf %cst_0 : f64 to f32
              %45 = arith.mulf %in_7, %44 : f32
              %46 = arith.cmpf ogt, %in, %45 : f32
              %47 = arith.select %46, %in, %45 : f32
              linalg.yield %47 : f32
            }
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%17, %25 : memref<1x64x128xf32>, memref<1x64xf32>) outs(%17 : memref<1x64x128xf32>) {
            ^bb0(%in: f32, %in_7: f32, %out: f32):
              %44 = arith.truncf %cst_0 : f64 to f32
              %45 = arith.mulf %in, %44 : f32
              %46 = arith.subf %45, %in_7 : f32
              %47 = math.powf %cst, %46 : f32
              linalg.yield %47 : f32
            }
            linalg.fill ins(%cst_1 : f32) outs(%27 : memref<1x64xf32>)
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%17 : memref<1x64x128xf32>) outs(%27 : memref<1x64xf32>) {
            ^bb0(%in: f32, %out: f32):
              %44 = arith.addf %in, %out : f32
              linalg.yield %44 : f32
            }
            linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%23, %25 : memref<1x64xf32>, memref<1x64xf32>) outs(%29 : memref<1x64xf32>) {
            ^bb0(%in: f32, %in_7: f32, %out: f32):
              %44 = arith.subf %in, %in_7 : f32
              %45 = math.powf %cst, %44 : f32
              linalg.yield %45 : f32
            }
            linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%21, %29, %27 : memref<1x64xf32>, memref<1x64xf32>, memref<1x64xf32>) outs(%21 : memref<1x64xf32>) {
            ^bb0(%in: f32, %in_7: f32, %in_8: f32, %out: f32):
              %44 = arith.mulf %in, %in_7 : f32
              %45 = arith.addf %44, %in_8 : f32
              linalg.yield %45 : f32
            }
            loom.semaphore_give %27 : memref<1x64xf32>
            %42 = arith.muli %arg7, %c16384 : index
            %43 = arith.addi %37, %42 : index
            %reinterpret_cast_6 = memref.reinterpret_cast %arg1 to offset: [%43], sizes: [1, 128, 128], strides: [524288, 128, 1] : memref<32x4096x128xf32> to memref<1x128x128xf32, strided<[524288, 128, 1], offset: ?>>
            loom.copy %reinterpret_cast_6, %15 src_mem_space @DRAM dst_mem_space @L1, interconnect : [], broadcast : [1, 1] : memref<1x128x128xf32, strided<[524288, 128, 1], offset: ?>>, memref<1x128x128xf32>
            linalg.fill ins(%cst_1 : f32) outs(%36 : memref<1x64x128xf32>)
            linalg.batch_matmul ins(%17, %15 : memref<1x64x128xf32>, memref<1x128x128xf32>) outs(%36 : memref<1x64x128xf32>)
            loom.semaphore_give %15 : memref<1x128x128xf32>
            loom.semaphore_give %17 : memref<1x64x128xf32>
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%36, %34, %29 : memref<1x64x128xf32>, memref<1x64x128xf32>, memref<1x64xf32>) outs(%34 : memref<1x64x128xf32>) {
            ^bb0(%in: f32, %in_7: f32, %in_8: f32, %out: f32):
              %44 = arith.mulf %in_7, %in_8 : f32
              %45 = arith.addf %in, %44 : f32
              linalg.yield %45 : f32
            }
            loom.semaphore_give %29 : memref<1x64xf32>
            loom.semaphore_give %36 : memref<1x64x128xf32>
            linalg.copy ins(%25 : memref<1x64xf32>) outs(%23 : memref<1x64xf32>)
            loom.semaphore_give %25 : memref<1x64xf32>
          }
          loom.semaphore_give %23 : memref<1x64xf32>
          loom.semaphore_give %32 : memref<1x64x128xf32>
          linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%34, %21 : memref<1x64x128xf32>, memref<1x64xf32>) outs(%31 : memref<1x64x128xf32>) {
          ^bb0(%in: f32, %in_5: f32, %out: f32):
            %40 = arith.divf %in, %in_5 : f32
            linalg.yield %40 : f32
          }
          loom.semaphore_give %21 : memref<1x64xf32>
          loom.semaphore_give %34 : memref<1x64x128xf32>
          %reinterpret_cast_4 = memref.reinterpret_cast %arg3 to offset: [%39], sizes: [1, 64, 128], strides: [524288, 128, 1] : memref<32x4096x128xf32> to memref<1x64x128xf32, strided<[524288, 128, 1], offset: ?>>
          loom.copy %31, %reinterpret_cast_4 src_mem_space @L1 dst_mem_space @DRAM, interconnect : [], broadcast : [1, 1] : memref<1x64x128xf32>, memref<1x64x128xf32, strided<[524288, 128, 1], offset: ?>>
          loom.semaphore_give %31 : memref<1x64x128xf32>
        }
        scf.reduce 
      } {loom.iter_types = [#loom.iter_type<spatial>, #loom.iter_type<spatial>], loom.mapped_to_dims = [@x, @y]}
      return
    }
  }
  module attributes {loom.pass_name = "Materialize"} {
    func.func @attention__d0i0_d1i0__f10__d_d_d__BB1__BM64__BN128(%arg0: memref<32x128x4096xf32>, %arg1: memref<32x4096x128xf32>, %arg2: memref<32x4096x128xf32>, %arg3: memref<32x4096x128xf32>) {
      %c16384 = arith.constant 16384 : index
      %c8192 = arith.constant 8192 : index
      %c32 = arith.constant 32 : index
      %c524288 = arith.constant 524288 : index
      %cst = arith.constant 2.000000e+00 : f32
      %cst_0 = arith.constant 0.12751743074602467 : f64
      %cst_1 = arith.constant 0.000000e+00 : f32
      %cst_2 = arith.constant 1.000000e+00 : f32
      %cst_3 = arith.constant 0xFF800000 : f32
      %c64 = arith.constant 64 : index
      %c128 = arith.constant 128 : index
      %c0 = arith.constant 0 : index
      %c8 = arith.constant 8 : index
      %c1 = arith.constant 1 : index
      scf.parallel (%arg4, %arg5) = (%c0, %c0) to (%c8, %c8) step (%c1, %c1) {
        scf.for %arg6 = %c0 to %c64 step %c1 {
          %12 = arith.muli %arg4, %c8 overflow<nsw> : index
          %13 = arith.addi %12, %arg5 : index
          %14 = loom.alloc [1, 128, 128] on @L1 : memref<1x128x128xf32>
          %15 = loom.semaphore_take %14 : memref<1x128x128xf32> -> memref<1x128x128xf32>
          %16 = loom.alloc [1, 64, 128] on @L1 : memref<1x64x128xf32>
          %17 = loom.semaphore_take %16 : memref<1x64x128xf32> -> memref<1x64x128xf32>
          %18 = loom.alloc [1, 128, 128] on @L1 : memref<1x128x128xf32>
          %19 = loom.semaphore_take %18 : memref<1x128x128xf32> -> memref<1x128x128xf32>
          %20 = loom.alloc [1, 64] on @L1 : memref<1x64xf32>
          %21 = loom.semaphore_take %20 : memref<1x64xf32> -> memref<1x64xf32>
          %22 = loom.alloc [1, 64] on @L1 : memref<1x64xf32>
          %23 = loom.semaphore_take %22 : memref<1x64xf32> -> memref<1x64xf32>
          %24 = loom.alloc [1, 64] on @L1 : memref<1x64xf32>
          %25 = loom.semaphore_take %24 : memref<1x64xf32> -> memref<1x64xf32>
          %26 = loom.alloc [1, 64] on @L1 : memref<1x64xf32>
          %27 = loom.semaphore_take %26 : memref<1x64xf32> -> memref<1x64xf32>
          %28 = loom.alloc [1, 64] on @L1 : memref<1x64xf32>
          %29 = loom.semaphore_take %28 : memref<1x64xf32> -> memref<1x64xf32>
          %30 = loom.alloc [1, 64, 128] on @L1 : memref<1x64x128xf32>
          %31 = loom.semaphore_take %30 : memref<1x64x128xf32> -> memref<1x64x128xf32>
          %32 = loom.semaphore_take %30 : memref<1x64x128xf32> -> memref<1x64x128xf32>
          %33 = loom.alloc [1, 64, 128] on @L1 : memref<1x64x128xf32>
          %34 = loom.semaphore_take %33 : memref<1x64x128xf32> -> memref<1x64x128xf32>
          %35 = loom.alloc [1, 64, 128] on @L1 : memref<1x64x128xf32>
          %36 = loom.semaphore_take %35 : memref<1x64x128xf32> -> memref<1x64x128xf32>
          %37 = arith.muli %13, %c524288 overflow<nsw> : index
          %38 = arith.muli %arg6, %c8192 : index
          %39 = arith.addi %37, %38 : index
          %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%39], sizes: [1, 64, 128], strides: [524288, 128, 1] : memref<32x4096x128xf32> to memref<1x64x128xf32, strided<[524288, 128, 1], offset: ?>>
          loom.copy %reinterpret_cast, %32 src_mem_space @DRAM dst_mem_space @L1, interconnect : [], broadcast : [1, 1] : memref<1x64x128xf32, strided<[524288, 128, 1], offset: ?>>, memref<1x64x128xf32>
          linalg.fill ins(%cst_1 : f32) outs(%34 : memref<1x64x128xf32>)
          linalg.fill ins(%cst_2 : f32) outs(%21 : memref<1x64xf32>)
          linalg.fill ins(%cst_3 : f32) outs(%23 : memref<1x64xf32>)
          scf.for %arg7 = %c0 to %c32 step %c1 {
            %40 = arith.muli %arg7, %c128 : index
            %41 = arith.addi %37, %40 : index
            %reinterpret_cast_5 = memref.reinterpret_cast %arg0 to offset: [%41], sizes: [1, 128, 128], strides: [524288, 4096, 1] : memref<32x128x4096xf32> to memref<1x128x128xf32, strided<[524288, 4096, 1], offset: ?>>
            loom.copy %reinterpret_cast_5, %19 src_mem_space @DRAM dst_mem_space @L1, interconnect : [], broadcast : [1, 1] : memref<1x128x128xf32, strided<[524288, 4096, 1], offset: ?>>, memref<1x128x128xf32>
            linalg.fill ins(%cst_1 : f32) outs(%17 : memref<1x64x128xf32>)
            linalg.batch_matmul ins(%32, %19 : memref<1x64x128xf32>, memref<1x128x128xf32>) outs(%17 : memref<1x64x128xf32>)
            loom.semaphore_give %19 : memref<1x128x128xf32>
            linalg.fill ins(%cst_3 : f32) outs(%25 : memref<1x64xf32>)
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%17 : memref<1x64x128xf32>) outs(%25 : memref<1x64xf32>) {
            ^bb0(%in: f32, %out: f32):
              %44 = arith.maximumf %in, %out : f32
              linalg.yield %44 : f32
            }
            linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%23, %25 : memref<1x64xf32>, memref<1x64xf32>) outs(%25 : memref<1x64xf32>) {
            ^bb0(%in: f32, %in_7: f32, %out: f32):
              %44 = arith.truncf %cst_0 : f64 to f32
              %45 = arith.mulf %in_7, %44 : f32
              %46 = arith.cmpf ogt, %in, %45 : f32
              %47 = arith.select %46, %in, %45 : f32
              linalg.yield %47 : f32
            }
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%17, %25 : memref<1x64x128xf32>, memref<1x64xf32>) outs(%17 : memref<1x64x128xf32>) {
            ^bb0(%in: f32, %in_7: f32, %out: f32):
              %44 = arith.truncf %cst_0 : f64 to f32
              %45 = arith.mulf %in, %44 : f32
              %46 = arith.subf %45, %in_7 : f32
              %47 = math.powf %cst, %46 : f32
              linalg.yield %47 : f32
            }
            linalg.fill ins(%cst_1 : f32) outs(%27 : memref<1x64xf32>)
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%17 : memref<1x64x128xf32>) outs(%27 : memref<1x64xf32>) {
            ^bb0(%in: f32, %out: f32):
              %44 = arith.addf %in, %out : f32
              linalg.yield %44 : f32
            }
            linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%23, %25 : memref<1x64xf32>, memref<1x64xf32>) outs(%29 : memref<1x64xf32>) {
            ^bb0(%in: f32, %in_7: f32, %out: f32):
              %44 = arith.subf %in, %in_7 : f32
              %45 = math.powf %cst, %44 : f32
              linalg.yield %45 : f32
            }
            linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%21, %29, %27 : memref<1x64xf32>, memref<1x64xf32>, memref<1x64xf32>) outs(%21 : memref<1x64xf32>) {
            ^bb0(%in: f32, %in_7: f32, %in_8: f32, %out: f32):
              %44 = arith.mulf %in, %in_7 : f32
              %45 = arith.addf %44, %in_8 : f32
              linalg.yield %45 : f32
            }
            loom.semaphore_give %27 : memref<1x64xf32>
            %42 = arith.muli %arg7, %c16384 : index
            %43 = arith.addi %37, %42 : index
            %reinterpret_cast_6 = memref.reinterpret_cast %arg1 to offset: [%43], sizes: [1, 128, 128], strides: [524288, 128, 1] : memref<32x4096x128xf32> to memref<1x128x128xf32, strided<[524288, 128, 1], offset: ?>>
            loom.copy %reinterpret_cast_6, %15 src_mem_space @DRAM dst_mem_space @L1, interconnect : [], broadcast : [1, 1] : memref<1x128x128xf32, strided<[524288, 128, 1], offset: ?>>, memref<1x128x128xf32>
            linalg.fill ins(%cst_1 : f32) outs(%36 : memref<1x64x128xf32>)
            linalg.batch_matmul ins(%17, %15 : memref<1x64x128xf32>, memref<1x128x128xf32>) outs(%36 : memref<1x64x128xf32>)
            loom.semaphore_give %15 : memref<1x128x128xf32>
            loom.semaphore_give %17 : memref<1x64x128xf32>
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%36, %34, %29 : memref<1x64x128xf32>, memref<1x64x128xf32>, memref<1x64xf32>) outs(%34 : memref<1x64x128xf32>) {
            ^bb0(%in: f32, %in_7: f32, %in_8: f32, %out: f32):
              %44 = arith.mulf %in_7, %in_8 : f32
              %45 = arith.addf %in, %44 : f32
              linalg.yield %45 : f32
            }
            loom.semaphore_give %29 : memref<1x64xf32>
            loom.semaphore_give %36 : memref<1x64x128xf32>
            linalg.copy ins(%25 : memref<1x64xf32>) outs(%23 : memref<1x64xf32>)
            loom.semaphore_give %25 : memref<1x64xf32>
          }
          loom.semaphore_give %23 : memref<1x64xf32>
          loom.semaphore_give %32 : memref<1x64x128xf32>
          linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%34, %21 : memref<1x64x128xf32>, memref<1x64xf32>) outs(%31 : memref<1x64x128xf32>) {
          ^bb0(%in: f32, %in_5: f32, %out: f32):
            %40 = arith.divf %in, %in_5 : f32
            linalg.yield %40 : f32
          }
          loom.semaphore_give %21 : memref<1x64xf32>
          loom.semaphore_give %34 : memref<1x64x128xf32>
          %reinterpret_cast_4 = memref.reinterpret_cast %arg3 to offset: [%39], sizes: [1, 64, 128], strides: [524288, 128, 1] : memref<32x4096x128xf32> to memref<1x64x128xf32, strided<[524288, 128, 1], offset: ?>>
          loom.copy %31, %reinterpret_cast_4 src_mem_space @L1 dst_mem_space @DRAM, interconnect : [], broadcast : [1, 1] : memref<1x64x128xf32>, memref<1x64x128xf32, strided<[524288, 128, 1], offset: ?>>
          loom.semaphore_give %31 : memref<1x64x128xf32>
        }
        scf.reduce 
      } {loom.iter_types = [#loom.iter_type<spatial>, #loom.iter_type<spatial>], loom.mapped_to_dims = [@x, @y]}
      return
    }
  }
  module attributes {loom.pass_name = "Materialize"} {
    func.func @attention__d1i0_d0i0__f01__d_d_d__BB1__BM64__BN128(%arg0: memref<32x128x4096xf32>, %arg1: memref<32x4096x128xf32>, %arg2: memref<32x4096x128xf32>, %arg3: memref<32x4096x128xf32>) {
      %c16384 = arith.constant 16384 : index
      %c8192 = arith.constant 8192 : index
      %c32 = arith.constant 32 : index
      %c524288 = arith.constant 524288 : index
      %cst = arith.constant 2.000000e+00 : f32
      %cst_0 = arith.constant 0.12751743074602467 : f64
      %cst_1 = arith.constant 0.000000e+00 : f32
      %cst_2 = arith.constant 1.000000e+00 : f32
      %cst_3 = arith.constant 0xFF800000 : f32
      %c64 = arith.constant 64 : index
      %c128 = arith.constant 128 : index
      %c0 = arith.constant 0 : index
      %c8 = arith.constant 8 : index
      %c1 = arith.constant 1 : index
      scf.parallel (%arg4, %arg5) = (%c0, %c0) to (%c8, %c8) step (%c1, %c1) {
        scf.for %arg6 = %c0 to %c64 step %c1 {
          %12 = arith.muli %arg4, %c8 overflow<nsw> : index
          %13 = arith.addi %12, %arg5 : index
          %14 = loom.alloc [1, 128, 128] on @L1 : memref<1x128x128xf32>
          %15 = loom.semaphore_take %14 : memref<1x128x128xf32> -> memref<1x128x128xf32>
          %16 = loom.alloc [1, 64, 128] on @L1 : memref<1x64x128xf32>
          %17 = loom.semaphore_take %16 : memref<1x64x128xf32> -> memref<1x64x128xf32>
          %18 = loom.alloc [1, 128, 128] on @L1 : memref<1x128x128xf32>
          %19 = loom.semaphore_take %18 : memref<1x128x128xf32> -> memref<1x128x128xf32>
          %20 = loom.alloc [1, 64] on @L1 : memref<1x64xf32>
          %21 = loom.semaphore_take %20 : memref<1x64xf32> -> memref<1x64xf32>
          %22 = loom.alloc [1, 64] on @L1 : memref<1x64xf32>
          %23 = loom.semaphore_take %22 : memref<1x64xf32> -> memref<1x64xf32>
          %24 = loom.alloc [1, 64] on @L1 : memref<1x64xf32>
          %25 = loom.semaphore_take %24 : memref<1x64xf32> -> memref<1x64xf32>
          %26 = loom.alloc [1, 64] on @L1 : memref<1x64xf32>
          %27 = loom.semaphore_take %26 : memref<1x64xf32> -> memref<1x64xf32>
          %28 = loom.alloc [1, 64] on @L1 : memref<1x64xf32>
          %29 = loom.semaphore_take %28 : memref<1x64xf32> -> memref<1x64xf32>
          %30 = loom.alloc [1, 64, 128] on @L1 : memref<1x64x128xf32>
          %31 = loom.semaphore_take %30 : memref<1x64x128xf32> -> memref<1x64x128xf32>
          %32 = loom.semaphore_take %30 : memref<1x64x128xf32> -> memref<1x64x128xf32>
          %33 = loom.alloc [1, 64, 128] on @L1 : memref<1x64x128xf32>
          %34 = loom.semaphore_take %33 : memref<1x64x128xf32> -> memref<1x64x128xf32>
          %35 = loom.alloc [1, 64, 128] on @L1 : memref<1x64x128xf32>
          %36 = loom.semaphore_take %35 : memref<1x64x128xf32> -> memref<1x64x128xf32>
          %37 = arith.muli %13, %c524288 overflow<nsw> : index
          %38 = arith.muli %arg6, %c8192 : index
          %39 = arith.addi %37, %38 : index
          %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%39], sizes: [1, 64, 128], strides: [524288, 128, 1] : memref<32x4096x128xf32> to memref<1x64x128xf32, strided<[524288, 128, 1], offset: ?>>
          loom.copy %reinterpret_cast, %32 src_mem_space @DRAM dst_mem_space @L1, interconnect : [], broadcast : [1, 1] : memref<1x64x128xf32, strided<[524288, 128, 1], offset: ?>>, memref<1x64x128xf32>
          linalg.fill ins(%cst_1 : f32) outs(%34 : memref<1x64x128xf32>)
          linalg.fill ins(%cst_2 : f32) outs(%21 : memref<1x64xf32>)
          linalg.fill ins(%cst_3 : f32) outs(%23 : memref<1x64xf32>)
          scf.for %arg7 = %c0 to %c32 step %c1 {
            %40 = arith.muli %arg7, %c128 : index
            %41 = arith.addi %37, %40 : index
            %reinterpret_cast_5 = memref.reinterpret_cast %arg0 to offset: [%41], sizes: [1, 128, 128], strides: [524288, 4096, 1] : memref<32x128x4096xf32> to memref<1x128x128xf32, strided<[524288, 4096, 1], offset: ?>>
            loom.copy %reinterpret_cast_5, %19 src_mem_space @DRAM dst_mem_space @L1, interconnect : [], broadcast : [1, 1] : memref<1x128x128xf32, strided<[524288, 4096, 1], offset: ?>>, memref<1x128x128xf32>
            linalg.fill ins(%cst_1 : f32) outs(%17 : memref<1x64x128xf32>)
            linalg.batch_matmul ins(%32, %19 : memref<1x64x128xf32>, memref<1x128x128xf32>) outs(%17 : memref<1x64x128xf32>)
            loom.semaphore_give %19 : memref<1x128x128xf32>
            linalg.fill ins(%cst_3 : f32) outs(%25 : memref<1x64xf32>)
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%17 : memref<1x64x128xf32>) outs(%25 : memref<1x64xf32>) {
            ^bb0(%in: f32, %out: f32):
              %44 = arith.maximumf %in, %out : f32
              linalg.yield %44 : f32
            }
            linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%23, %25 : memref<1x64xf32>, memref<1x64xf32>) outs(%25 : memref<1x64xf32>) {
            ^bb0(%in: f32, %in_7: f32, %out: f32):
              %44 = arith.truncf %cst_0 : f64 to f32
              %45 = arith.mulf %in_7, %44 : f32
              %46 = arith.cmpf ogt, %in, %45 : f32
              %47 = arith.select %46, %in, %45 : f32
              linalg.yield %47 : f32
            }
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%17, %25 : memref<1x64x128xf32>, memref<1x64xf32>) outs(%17 : memref<1x64x128xf32>) {
            ^bb0(%in: f32, %in_7: f32, %out: f32):
              %44 = arith.truncf %cst_0 : f64 to f32
              %45 = arith.mulf %in, %44 : f32
              %46 = arith.subf %45, %in_7 : f32
              %47 = math.powf %cst, %46 : f32
              linalg.yield %47 : f32
            }
            linalg.fill ins(%cst_1 : f32) outs(%27 : memref<1x64xf32>)
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%17 : memref<1x64x128xf32>) outs(%27 : memref<1x64xf32>) {
            ^bb0(%in: f32, %out: f32):
              %44 = arith.addf %in, %out : f32
              linalg.yield %44 : f32
            }
            linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%23, %25 : memref<1x64xf32>, memref<1x64xf32>) outs(%29 : memref<1x64xf32>) {
            ^bb0(%in: f32, %in_7: f32, %out: f32):
              %44 = arith.subf %in, %in_7 : f32
              %45 = math.powf %cst, %44 : f32
              linalg.yield %45 : f32
            }
            linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%21, %29, %27 : memref<1x64xf32>, memref<1x64xf32>, memref<1x64xf32>) outs(%21 : memref<1x64xf32>) {
            ^bb0(%in: f32, %in_7: f32, %in_8: f32, %out: f32):
              %44 = arith.mulf %in, %in_7 : f32
              %45 = arith.addf %44, %in_8 : f32
              linalg.yield %45 : f32
            }
            loom.semaphore_give %27 : memref<1x64xf32>
            %42 = arith.muli %arg7, %c16384 : index
            %43 = arith.addi %37, %42 : index
            %reinterpret_cast_6 = memref.reinterpret_cast %arg1 to offset: [%43], sizes: [1, 128, 128], strides: [524288, 128, 1] : memref<32x4096x128xf32> to memref<1x128x128xf32, strided<[524288, 128, 1], offset: ?>>
            loom.copy %reinterpret_cast_6, %15 src_mem_space @DRAM dst_mem_space @L1, interconnect : [], broadcast : [1, 1] : memref<1x128x128xf32, strided<[524288, 128, 1], offset: ?>>, memref<1x128x128xf32>
            linalg.fill ins(%cst_1 : f32) outs(%36 : memref<1x64x128xf32>)
            linalg.batch_matmul ins(%17, %15 : memref<1x64x128xf32>, memref<1x128x128xf32>) outs(%36 : memref<1x64x128xf32>)
            loom.semaphore_give %15 : memref<1x128x128xf32>
            loom.semaphore_give %17 : memref<1x64x128xf32>
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%36, %34, %29 : memref<1x64x128xf32>, memref<1x64x128xf32>, memref<1x64xf32>) outs(%34 : memref<1x64x128xf32>) {
            ^bb0(%in: f32, %in_7: f32, %in_8: f32, %out: f32):
              %44 = arith.mulf %in_7, %in_8 : f32
              %45 = arith.addf %in, %44 : f32
              linalg.yield %45 : f32
            }
            loom.semaphore_give %29 : memref<1x64xf32>
            loom.semaphore_give %36 : memref<1x64x128xf32>
            linalg.copy ins(%25 : memref<1x64xf32>) outs(%23 : memref<1x64xf32>)
            loom.semaphore_give %25 : memref<1x64xf32>
          }
          loom.semaphore_give %23 : memref<1x64xf32>
          loom.semaphore_give %32 : memref<1x64x128xf32>
          linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%34, %21 : memref<1x64x128xf32>, memref<1x64xf32>) outs(%31 : memref<1x64x128xf32>) {
          ^bb0(%in: f32, %in_5: f32, %out: f32):
            %40 = arith.divf %in, %in_5 : f32
            linalg.yield %40 : f32
          }
          loom.semaphore_give %21 : memref<1x64xf32>
          loom.semaphore_give %34 : memref<1x64x128xf32>
          %reinterpret_cast_4 = memref.reinterpret_cast %arg3 to offset: [%39], sizes: [1, 64, 128], strides: [524288, 128, 1] : memref<32x4096x128xf32> to memref<1x64x128xf32, strided<[524288, 128, 1], offset: ?>>
          loom.copy %31, %reinterpret_cast_4 src_mem_space @L1 dst_mem_space @DRAM, interconnect : [], broadcast : [1, 1] : memref<1x64x128xf32>, memref<1x64x128xf32, strided<[524288, 128, 1], offset: ?>>
          loom.semaphore_give %31 : memref<1x64x128xf32>
        }
        scf.reduce 
      } {loom.iter_types = [#loom.iter_type<spatial>, #loom.iter_type<spatial>], loom.mapped_to_dims = [@y, @x]}
      return
    }
  }
  module attributes {loom.pass_name = "Materialize"} {
    func.func @attention__d1i0_d0i0__f10__d_d_d__BB1__BM64__BN128(%arg0: memref<32x128x4096xf32>, %arg1: memref<32x4096x128xf32>, %arg2: memref<32x4096x128xf32>, %arg3: memref<32x4096x128xf32>) {
      %c16384 = arith.constant 16384 : index
      %c8192 = arith.constant 8192 : index
      %c32 = arith.constant 32 : index
      %c524288 = arith.constant 524288 : index
      %cst = arith.constant 2.000000e+00 : f32
      %cst_0 = arith.constant 0.12751743074602467 : f64
      %cst_1 = arith.constant 0.000000e+00 : f32
      %cst_2 = arith.constant 1.000000e+00 : f32
      %cst_3 = arith.constant 0xFF800000 : f32
      %c64 = arith.constant 64 : index
      %c128 = arith.constant 128 : index
      %c0 = arith.constant 0 : index
      %c8 = arith.constant 8 : index
      %c1 = arith.constant 1 : index
      scf.parallel (%arg4, %arg5) = (%c0, %c0) to (%c8, %c8) step (%c1, %c1) {
        scf.for %arg6 = %c0 to %c64 step %c1 {
          %12 = arith.muli %arg4, %c8 overflow<nsw> : index
          %13 = arith.addi %12, %arg5 : index
          %14 = loom.alloc [1, 128, 128] on @L1 : memref<1x128x128xf32>
          %15 = loom.semaphore_take %14 : memref<1x128x128xf32> -> memref<1x128x128xf32>
          %16 = loom.alloc [1, 64, 128] on @L1 : memref<1x64x128xf32>
          %17 = loom.semaphore_take %16 : memref<1x64x128xf32> -> memref<1x64x128xf32>
          %18 = loom.alloc [1, 128, 128] on @L1 : memref<1x128x128xf32>
          %19 = loom.semaphore_take %18 : memref<1x128x128xf32> -> memref<1x128x128xf32>
          %20 = loom.alloc [1, 64] on @L1 : memref<1x64xf32>
          %21 = loom.semaphore_take %20 : memref<1x64xf32> -> memref<1x64xf32>
          %22 = loom.alloc [1, 64] on @L1 : memref<1x64xf32>
          %23 = loom.semaphore_take %22 : memref<1x64xf32> -> memref<1x64xf32>
          %24 = loom.alloc [1, 64] on @L1 : memref<1x64xf32>
          %25 = loom.semaphore_take %24 : memref<1x64xf32> -> memref<1x64xf32>
          %26 = loom.alloc [1, 64] on @L1 : memref<1x64xf32>
          %27 = loom.semaphore_take %26 : memref<1x64xf32> -> memref<1x64xf32>
          %28 = loom.alloc [1, 64] on @L1 : memref<1x64xf32>
          %29 = loom.semaphore_take %28 : memref<1x64xf32> -> memref<1x64xf32>
          %30 = loom.alloc [1, 64, 128] on @L1 : memref<1x64x128xf32>
          %31 = loom.semaphore_take %30 : memref<1x64x128xf32> -> memref<1x64x128xf32>
          %32 = loom.semaphore_take %30 : memref<1x64x128xf32> -> memref<1x64x128xf32>
          %33 = loom.alloc [1, 64, 128] on @L1 : memref<1x64x128xf32>
          %34 = loom.semaphore_take %33 : memref<1x64x128xf32> -> memref<1x64x128xf32>
          %35 = loom.alloc [1, 64, 128] on @L1 : memref<1x64x128xf32>
          %36 = loom.semaphore_take %35 : memref<1x64x128xf32> -> memref<1x64x128xf32>
          %37 = arith.muli %13, %c524288 overflow<nsw> : index
          %38 = arith.muli %arg6, %c8192 : index
          %39 = arith.addi %37, %38 : index
          %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%39], sizes: [1, 64, 128], strides: [524288, 128, 1] : memref<32x4096x128xf32> to memref<1x64x128xf32, strided<[524288, 128, 1], offset: ?>>
          loom.copy %reinterpret_cast, %32 src_mem_space @DRAM dst_mem_space @L1, interconnect : [], broadcast : [1, 1] : memref<1x64x128xf32, strided<[524288, 128, 1], offset: ?>>, memref<1x64x128xf32>
          linalg.fill ins(%cst_1 : f32) outs(%34 : memref<1x64x128xf32>)
          linalg.fill ins(%cst_2 : f32) outs(%21 : memref<1x64xf32>)
          linalg.fill ins(%cst_3 : f32) outs(%23 : memref<1x64xf32>)
          scf.for %arg7 = %c0 to %c32 step %c1 {
            %40 = arith.muli %arg7, %c128 : index
            %41 = arith.addi %37, %40 : index
            %reinterpret_cast_5 = memref.reinterpret_cast %arg0 to offset: [%41], sizes: [1, 128, 128], strides: [524288, 4096, 1] : memref<32x128x4096xf32> to memref<1x128x128xf32, strided<[524288, 4096, 1], offset: ?>>
            loom.copy %reinterpret_cast_5, %19 src_mem_space @DRAM dst_mem_space @L1, interconnect : [], broadcast : [1, 1] : memref<1x128x128xf32, strided<[524288, 4096, 1], offset: ?>>, memref<1x128x128xf32>
            linalg.fill ins(%cst_1 : f32) outs(%17 : memref<1x64x128xf32>)
            linalg.batch_matmul ins(%32, %19 : memref<1x64x128xf32>, memref<1x128x128xf32>) outs(%17 : memref<1x64x128xf32>)
            loom.semaphore_give %19 : memref<1x128x128xf32>
            linalg.fill ins(%cst_3 : f32) outs(%25 : memref<1x64xf32>)
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%17 : memref<1x64x128xf32>) outs(%25 : memref<1x64xf32>) {
            ^bb0(%in: f32, %out: f32):
              %44 = arith.maximumf %in, %out : f32
              linalg.yield %44 : f32
            }
            linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%23, %25 : memref<1x64xf32>, memref<1x64xf32>) outs(%25 : memref<1x64xf32>) {
            ^bb0(%in: f32, %in_7: f32, %out: f32):
              %44 = arith.truncf %cst_0 : f64 to f32
              %45 = arith.mulf %in_7, %44 : f32
              %46 = arith.cmpf ogt, %in, %45 : f32
              %47 = arith.select %46, %in, %45 : f32
              linalg.yield %47 : f32
            }
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%17, %25 : memref<1x64x128xf32>, memref<1x64xf32>) outs(%17 : memref<1x64x128xf32>) {
            ^bb0(%in: f32, %in_7: f32, %out: f32):
              %44 = arith.truncf %cst_0 : f64 to f32
              %45 = arith.mulf %in, %44 : f32
              %46 = arith.subf %45, %in_7 : f32
              %47 = math.powf %cst, %46 : f32
              linalg.yield %47 : f32
            }
            linalg.fill ins(%cst_1 : f32) outs(%27 : memref<1x64xf32>)
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%17 : memref<1x64x128xf32>) outs(%27 : memref<1x64xf32>) {
            ^bb0(%in: f32, %out: f32):
              %44 = arith.addf %in, %out : f32
              linalg.yield %44 : f32
            }
            linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%23, %25 : memref<1x64xf32>, memref<1x64xf32>) outs(%29 : memref<1x64xf32>) {
            ^bb0(%in: f32, %in_7: f32, %out: f32):
              %44 = arith.subf %in, %in_7 : f32
              %45 = math.powf %cst, %44 : f32
              linalg.yield %45 : f32
            }
            linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%21, %29, %27 : memref<1x64xf32>, memref<1x64xf32>, memref<1x64xf32>) outs(%21 : memref<1x64xf32>) {
            ^bb0(%in: f32, %in_7: f32, %in_8: f32, %out: f32):
              %44 = arith.mulf %in, %in_7 : f32
              %45 = arith.addf %44, %in_8 : f32
              linalg.yield %45 : f32
            }
            loom.semaphore_give %27 : memref<1x64xf32>
            %42 = arith.muli %arg7, %c16384 : index
            %43 = arith.addi %37, %42 : index
            %reinterpret_cast_6 = memref.reinterpret_cast %arg1 to offset: [%43], sizes: [1, 128, 128], strides: [524288, 128, 1] : memref<32x4096x128xf32> to memref<1x128x128xf32, strided<[524288, 128, 1], offset: ?>>
            loom.copy %reinterpret_cast_6, %15 src_mem_space @DRAM dst_mem_space @L1, interconnect : [], broadcast : [1, 1] : memref<1x128x128xf32, strided<[524288, 128, 1], offset: ?>>, memref<1x128x128xf32>
            linalg.fill ins(%cst_1 : f32) outs(%36 : memref<1x64x128xf32>)
            linalg.batch_matmul ins(%17, %15 : memref<1x64x128xf32>, memref<1x128x128xf32>) outs(%36 : memref<1x64x128xf32>)
            loom.semaphore_give %15 : memref<1x128x128xf32>
            loom.semaphore_give %17 : memref<1x64x128xf32>
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%36, %34, %29 : memref<1x64x128xf32>, memref<1x64x128xf32>, memref<1x64xf32>) outs(%34 : memref<1x64x128xf32>) {
            ^bb0(%in: f32, %in_7: f32, %in_8: f32, %out: f32):
              %44 = arith.mulf %in_7, %in_8 : f32
              %45 = arith.addf %in, %44 : f32
              linalg.yield %45 : f32
            }
            loom.semaphore_give %29 : memref<1x64xf32>
            loom.semaphore_give %36 : memref<1x64x128xf32>
            linalg.copy ins(%25 : memref<1x64xf32>) outs(%23 : memref<1x64xf32>)
            loom.semaphore_give %25 : memref<1x64xf32>
          }
          loom.semaphore_give %23 : memref<1x64xf32>
          loom.semaphore_give %32 : memref<1x64x128xf32>
          linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%34, %21 : memref<1x64x128xf32>, memref<1x64xf32>) outs(%31 : memref<1x64x128xf32>) {
          ^bb0(%in: f32, %in_5: f32, %out: f32):
            %40 = arith.divf %in, %in_5 : f32
            linalg.yield %40 : f32
          }
          loom.semaphore_give %21 : memref<1x64xf32>
          loom.semaphore_give %34 : memref<1x64x128xf32>
          %reinterpret_cast_4 = memref.reinterpret_cast %arg3 to offset: [%39], sizes: [1, 64, 128], strides: [524288, 128, 1] : memref<32x4096x128xf32> to memref<1x64x128xf32, strided<[524288, 128, 1], offset: ?>>
          loom.copy %31, %reinterpret_cast_4 src_mem_space @L1 dst_mem_space @DRAM, interconnect : [], broadcast : [1, 1] : memref<1x64x128xf32>, memref<1x64x128xf32, strided<[524288, 128, 1], offset: ?>>
          loom.semaphore_give %31 : memref<1x64x128xf32>
        }
        scf.reduce 
      } {loom.iter_types = [#loom.iter_type<spatial>, #loom.iter_type<spatial>], loom.mapped_to_dims = [@y, @x]}
      return
    }
  }
  module attributes {loom.pass_name = "Materialize"} {
    func.func @attention__d0i0_d1i1__f01__d_d_d__BB1__BM64__BN128(%arg0: memref<32x128x4096xf32>, %arg1: memref<32x4096x128xf32>, %arg2: memref<32x4096x128xf32>, %arg3: memref<32x4096x128xf32>) {
      %c16384 = arith.constant 16384 : index
      %c8192 = arith.constant 8192 : index
      %c32 = arith.constant 32 : index
      %c524288 = arith.constant 524288 : index
      %c4 = arith.constant 4 : index
      %cst = arith.constant 2.000000e+00 : f32
      %cst_0 = arith.constant 0.12751743074602467 : f64
      %cst_1 = arith.constant 0.000000e+00 : f32
      %cst_2 = arith.constant 1.000000e+00 : f32
      %cst_3 = arith.constant 0xFF800000 : f32
      %c128 = arith.constant 128 : index
      %c0 = arith.constant 0 : index
      %c8 = arith.constant 8 : index
      %c1 = arith.constant 1 : index
      scf.parallel (%arg4, %arg5) = (%c0, %c0) to (%c8, %c8) step (%c1, %c1) {
        scf.for %arg6 = %c0 to %c4 step %c1 {
          scf.for %arg7 = %c0 to %c8 step %c1 {
            %12 = arith.muli %arg6, %c8 overflow<nsw> : index
            %13 = arith.addi %arg4, %12 : index
            %14 = arith.muli %arg7, %c8 overflow<nsw> : index
            %15 = arith.addi %arg5, %14 : index
            %16 = loom.alloc [1, 128, 128] on @L1 : memref<1x128x128xf32>
            %17 = loom.semaphore_take %16 : memref<1x128x128xf32> -> memref<1x128x128xf32>
            %18 = loom.alloc [1, 64, 128] on @L1 : memref<1x64x128xf32>
            %19 = loom.semaphore_take %18 : memref<1x64x128xf32> -> memref<1x64x128xf32>
            %20 = loom.alloc [1, 128, 128] on @L1 : memref<1x128x128xf32>
            %21 = loom.semaphore_take %20 : memref<1x128x128xf32> -> memref<1x128x128xf32>
            %22 = loom.alloc [1, 64] on @L1 : memref<1x64xf32>
            %23 = loom.semaphore_take %22 : memref<1x64xf32> -> memref<1x64xf32>
            %24 = loom.alloc [1, 64] on @L1 : memref<1x64xf32>
            %25 = loom.semaphore_take %24 : memref<1x64xf32> -> memref<1x64xf32>
            %26 = loom.alloc [1, 64] on @L1 : memref<1x64xf32>
            %27 = loom.semaphore_take %26 : memref<1x64xf32> -> memref<1x64xf32>
            %28 = loom.alloc [1, 64] on @L1 : memref<1x64xf32>
            %29 = loom.semaphore_take %28 : memref<1x64xf32> -> memref<1x64xf32>
            %30 = loom.alloc [1, 64] on @L1 : memref<1x64xf32>
            %31 = loom.semaphore_take %30 : memref<1x64xf32> -> memref<1x64xf32>
            %32 = loom.alloc [1, 64, 128] on @L1 : memref<1x64x128xf32>
            %33 = loom.semaphore_take %32 : memref<1x64x128xf32> -> memref<1x64x128xf32>
            %34 = loom.semaphore_take %32 : memref<1x64x128xf32> -> memref<1x64x128xf32>
            %35 = loom.alloc [1, 64, 128] on @L1 : memref<1x64x128xf32>
            %36 = loom.semaphore_take %35 : memref<1x64x128xf32> -> memref<1x64x128xf32>
            %37 = loom.alloc [1, 64, 128] on @L1 : memref<1x64x128xf32>
            %38 = loom.semaphore_take %37 : memref<1x64x128xf32> -> memref<1x64x128xf32>
            %39 = arith.muli %13, %c524288 overflow<nsw> : index
            %40 = arith.muli %15, %c8192 : index
            %41 = arith.addi %39, %40 : index
            %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%41], sizes: [1, 64, 128], strides: [524288, 128, 1] : memref<32x4096x128xf32> to memref<1x64x128xf32, strided<[524288, 128, 1], offset: ?>>
            loom.copy %reinterpret_cast, %34 src_mem_space @DRAM dst_mem_space @L1, interconnect : [], broadcast : [1, 1] : memref<1x64x128xf32, strided<[524288, 128, 1], offset: ?>>, memref<1x64x128xf32>
            linalg.fill ins(%cst_1 : f32) outs(%36 : memref<1x64x128xf32>)
            linalg.fill ins(%cst_2 : f32) outs(%23 : memref<1x64xf32>)
            linalg.fill ins(%cst_3 : f32) outs(%25 : memref<1x64xf32>)
            scf.for %arg8 = %c0 to %c32 step %c1 {
              %42 = arith.muli %arg8, %c128 : index
              %43 = arith.addi %39, %42 : index
              %reinterpret_cast_5 = memref.reinterpret_cast %arg0 to offset: [%43], sizes: [1, 128, 128], strides: [524288, 4096, 1] : memref<32x128x4096xf32> to memref<1x128x128xf32, strided<[524288, 4096, 1], offset: ?>>
              loom.copy %reinterpret_cast_5, %21 src_mem_space @DRAM dst_mem_space @L1, interconnect : [], broadcast : [1, 1] : memref<1x128x128xf32, strided<[524288, 4096, 1], offset: ?>>, memref<1x128x128xf32>
              linalg.fill ins(%cst_1 : f32) outs(%19 : memref<1x64x128xf32>)
              linalg.batch_matmul ins(%34, %21 : memref<1x64x128xf32>, memref<1x128x128xf32>) outs(%19 : memref<1x64x128xf32>)
              loom.semaphore_give %21 : memref<1x128x128xf32>
              linalg.fill ins(%cst_3 : f32) outs(%27 : memref<1x64xf32>)
              linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%19 : memref<1x64x128xf32>) outs(%27 : memref<1x64xf32>) {
              ^bb0(%in: f32, %out: f32):
                %46 = arith.maximumf %in, %out : f32
                linalg.yield %46 : f32
              }
              linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%25, %27 : memref<1x64xf32>, memref<1x64xf32>) outs(%27 : memref<1x64xf32>) {
              ^bb0(%in: f32, %in_7: f32, %out: f32):
                %46 = arith.truncf %cst_0 : f64 to f32
                %47 = arith.mulf %in_7, %46 : f32
                %48 = arith.cmpf ogt, %in, %47 : f32
                %49 = arith.select %48, %in, %47 : f32
                linalg.yield %49 : f32
              }
              linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%19, %27 : memref<1x64x128xf32>, memref<1x64xf32>) outs(%19 : memref<1x64x128xf32>) {
              ^bb0(%in: f32, %in_7: f32, %out: f32):
                %46 = arith.truncf %cst_0 : f64 to f32
                %47 = arith.mulf %in, %46 : f32
                %48 = arith.subf %47, %in_7 : f32
                %49 = math.powf %cst, %48 : f32
                linalg.yield %49 : f32
              }
              linalg.fill ins(%cst_1 : f32) outs(%29 : memref<1x64xf32>)
              linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%19 : memref<1x64x128xf32>) outs(%29 : memref<1x64xf32>) {
              ^bb0(%in: f32, %out: f32):
                %46 = arith.addf %in, %out : f32
                linalg.yield %46 : f32
              }
              linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%25, %27 : memref<1x64xf32>, memref<1x64xf32>) outs(%31 : memref<1x64xf32>) {
              ^bb0(%in: f32, %in_7: f32, %out: f32):
                %46 = arith.subf %in, %in_7 : f32
                %47 = math.powf %cst, %46 : f32
                linalg.yield %47 : f32
              }
              linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%23, %31, %29 : memref<1x64xf32>, memref<1x64xf32>, memref<1x64xf32>) outs(%23 : memref<1x64xf32>) {
              ^bb0(%in: f32, %in_7: f32, %in_8: f32, %out: f32):
                %46 = arith.mulf %in, %in_7 : f32
                %47 = arith.addf %46, %in_8 : f32
                linalg.yield %47 : f32
              }
              loom.semaphore_give %29 : memref<1x64xf32>
              %44 = arith.muli %arg8, %c16384 : index
              %45 = arith.addi %39, %44 : index
              %reinterpret_cast_6 = memref.reinterpret_cast %arg1 to offset: [%45], sizes: [1, 128, 128], strides: [524288, 128, 1] : memref<32x4096x128xf32> to memref<1x128x128xf32, strided<[524288, 128, 1], offset: ?>>
              loom.copy %reinterpret_cast_6, %17 src_mem_space @DRAM dst_mem_space @L1, interconnect : [], broadcast : [1, 1] : memref<1x128x128xf32, strided<[524288, 128, 1], offset: ?>>, memref<1x128x128xf32>
              linalg.fill ins(%cst_1 : f32) outs(%38 : memref<1x64x128xf32>)
              linalg.batch_matmul ins(%19, %17 : memref<1x64x128xf32>, memref<1x128x128xf32>) outs(%38 : memref<1x64x128xf32>)
              loom.semaphore_give %17 : memref<1x128x128xf32>
              loom.semaphore_give %19 : memref<1x64x128xf32>
              linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%38, %36, %31 : memref<1x64x128xf32>, memref<1x64x128xf32>, memref<1x64xf32>) outs(%36 : memref<1x64x128xf32>) {
              ^bb0(%in: f32, %in_7: f32, %in_8: f32, %out: f32):
                %46 = arith.mulf %in_7, %in_8 : f32
                %47 = arith.addf %in, %46 : f32
                linalg.yield %47 : f32
              }
              loom.semaphore_give %31 : memref<1x64xf32>
              loom.semaphore_give %38 : memref<1x64x128xf32>
              linalg.copy ins(%27 : memref<1x64xf32>) outs(%25 : memref<1x64xf32>)
              loom.semaphore_give %27 : memref<1x64xf32>
            }
            loom.semaphore_give %25 : memref<1x64xf32>
            loom.semaphore_give %34 : memref<1x64x128xf32>
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%36, %23 : memref<1x64x128xf32>, memref<1x64xf32>) outs(%33 : memref<1x64x128xf32>) {
            ^bb0(%in: f32, %in_5: f32, %out: f32):
              %42 = arith.divf %in, %in_5 : f32
              linalg.yield %42 : f32
            }
            loom.semaphore_give %23 : memref<1x64xf32>
            loom.semaphore_give %36 : memref<1x64x128xf32>
            %reinterpret_cast_4 = memref.reinterpret_cast %arg3 to offset: [%41], sizes: [1, 64, 128], strides: [524288, 128, 1] : memref<32x4096x128xf32> to memref<1x64x128xf32, strided<[524288, 128, 1], offset: ?>>
            loom.copy %33, %reinterpret_cast_4 src_mem_space @L1 dst_mem_space @DRAM, interconnect : [], broadcast : [1, 1] : memref<1x64x128xf32>, memref<1x64x128xf32, strided<[524288, 128, 1], offset: ?>>
            loom.semaphore_give %33 : memref<1x64x128xf32>
          }
        }
        scf.reduce 
      } {loom.iter_types = [#loom.iter_type<spatial>, #loom.iter_type<spatial>], loom.mapped_to_dims = [@x, @y]}
      return
    }
  }
  module attributes {loom.pass_name = "Materialize"} {
    func.func @attention__d0i0_d1i1__f01__d_d_v__BB1__BM64__BN128(%arg0: memref<32x128x4096xf32>, %arg1: memref<32x4096x128xf32>, %arg2: memref<32x4096x128xf32>, %arg3: memref<32x4096x128xf32>) {
      %c16384 = arith.constant 16384 : index
      %c8192 = arith.constant 8192 : index
      %c32 = arith.constant 32 : index
      %c524288 = arith.constant 524288 : index
      %c4 = arith.constant 4 : index
      %cst = arith.constant 2.000000e+00 : f32
      %cst_0 = arith.constant 0.12751743074602467 : f64
      %cst_1 = arith.constant 0.000000e+00 : f32
      %cst_2 = arith.constant 1.000000e+00 : f32
      %cst_3 = arith.constant 0xFF800000 : f32
      %c128 = arith.constant 128 : index
      %c0 = arith.constant 0 : index
      %c8 = arith.constant 8 : index
      %c1 = arith.constant 1 : index
      scf.parallel (%arg4, %arg5) = (%c0, %c0) to (%c8, %c8) step (%c1, %c1) {
        scf.for %arg6 = %c0 to %c4 step %c1 {
          scf.for %arg7 = %c0 to %c8 step %c1 {
            %12 = arith.muli %arg6, %c8 overflow<nsw> : index
            %13 = arith.addi %arg4, %12 : index
            %14 = arith.muli %arg7, %c8 overflow<nsw> : index
            %15 = arith.addi %arg5, %14 : index
            %16 = loom.alloc [1, 128, 128] on @L1 : memref<1x128x128xf32>
            %17 = loom.semaphore_take %16 : memref<1x128x128xf32> -> memref<1x128x128xf32>
            %18 = loom.alloc [1, 64, 128] on @L1 : memref<1x64x128xf32>
            %19 = loom.semaphore_take %18 : memref<1x64x128xf32> -> memref<1x64x128xf32>
            %20 = loom.alloc [1, 128, 128] on @L1 : memref<1x128x128xf32>
            %21 = loom.semaphore_take %20 : memref<1x128x128xf32> -> memref<1x128x128xf32>
            %22 = loom.alloc [1, 64] on @L1 : memref<1x64xf32>
            %23 = loom.semaphore_take %22 : memref<1x64xf32> -> memref<1x64xf32>
            %24 = loom.alloc [1, 64] on @L1 : memref<1x64xf32>
            %25 = loom.semaphore_take %24 : memref<1x64xf32> -> memref<1x64xf32>
            %26 = loom.alloc [1, 64] on @L1 : memref<1x64xf32>
            %27 = loom.semaphore_take %26 : memref<1x64xf32> -> memref<1x64xf32>
            %28 = loom.alloc [1, 64] on @L1 : memref<1x64xf32>
            %29 = loom.semaphore_take %28 : memref<1x64xf32> -> memref<1x64xf32>
            %30 = loom.alloc [1, 64] on @L1 : memref<1x64xf32>
            %31 = loom.semaphore_take %30 : memref<1x64xf32> -> memref<1x64xf32>
            %32 = loom.alloc [1, 64, 128] on @L1 : memref<1x64x128xf32>
            %33 = loom.semaphore_take %32 : memref<1x64x128xf32> -> memref<1x64x128xf32>
            %34 = loom.semaphore_take %32 : memref<1x64x128xf32> -> memref<1x64x128xf32>
            %35 = loom.alloc [1, 64, 128] on @L1 : memref<1x64x128xf32>
            %36 = loom.semaphore_take %35 : memref<1x64x128xf32> -> memref<1x64x128xf32>
            %37 = loom.alloc [1, 64, 128] on @L1 : memref<1x64x128xf32>
            %38 = loom.semaphore_take %37 : memref<1x64x128xf32> -> memref<1x64x128xf32>
            %39 = arith.muli %13, %c524288 overflow<nsw> : index
            %40 = arith.muli %15, %c8192 : index
            %41 = arith.addi %39, %40 : index
            %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%41], sizes: [1, 64, 128], strides: [524288, 128, 1] : memref<32x4096x128xf32> to memref<1x64x128xf32, strided<[524288, 128, 1], offset: ?>>
            loom.copy %reinterpret_cast, %34 src_mem_space @DRAM dst_mem_space @L1, interconnect : [], broadcast : [1, 1] : memref<1x64x128xf32, strided<[524288, 128, 1], offset: ?>>, memref<1x64x128xf32>
            linalg.fill ins(%cst_1 : f32) outs(%36 : memref<1x64x128xf32>)
            linalg.fill ins(%cst_2 : f32) outs(%23 : memref<1x64xf32>)
            linalg.fill ins(%cst_3 : f32) outs(%25 : memref<1x64xf32>)
            scf.for %arg8 = %c0 to %c32 step %c1 {
              %42 = arith.muli %arg8, %c128 : index
              %43 = arith.addi %39, %42 : index
              %reinterpret_cast_5 = memref.reinterpret_cast %arg0 to offset: [%43], sizes: [1, 128, 128], strides: [524288, 4096, 1] : memref<32x128x4096xf32> to memref<1x128x128xf32, strided<[524288, 4096, 1], offset: ?>>
              loom.copy %reinterpret_cast_5, %21 src_mem_space @DRAM dst_mem_space @L1, interconnect : [], broadcast : [1, 1] : memref<1x128x128xf32, strided<[524288, 4096, 1], offset: ?>>, memref<1x128x128xf32>
              linalg.fill ins(%cst_1 : f32) outs(%19 : memref<1x64x128xf32>)
              linalg.batch_matmul ins(%34, %21 : memref<1x64x128xf32>, memref<1x128x128xf32>) outs(%19 : memref<1x64x128xf32>)
              loom.semaphore_give %21 : memref<1x128x128xf32>
              linalg.fill ins(%cst_3 : f32) outs(%27 : memref<1x64xf32>)
              linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%19 : memref<1x64x128xf32>) outs(%27 : memref<1x64xf32>) {
              ^bb0(%in: f32, %out: f32):
                %46 = arith.maximumf %in, %out : f32
                linalg.yield %46 : f32
              }
              linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%25, %27 : memref<1x64xf32>, memref<1x64xf32>) outs(%27 : memref<1x64xf32>) {
              ^bb0(%in: f32, %in_7: f32, %out: f32):
                %46 = arith.truncf %cst_0 : f64 to f32
                %47 = arith.mulf %in_7, %46 : f32
                %48 = arith.cmpf ogt, %in, %47 : f32
                %49 = arith.select %48, %in, %47 : f32
                linalg.yield %49 : f32
              }
              linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%19, %27 : memref<1x64x128xf32>, memref<1x64xf32>) outs(%19 : memref<1x64x128xf32>) {
              ^bb0(%in: f32, %in_7: f32, %out: f32):
                %46 = arith.truncf %cst_0 : f64 to f32
                %47 = arith.mulf %in, %46 : f32
                %48 = arith.subf %47, %in_7 : f32
                %49 = math.powf %cst, %48 : f32
                linalg.yield %49 : f32
              }
              linalg.fill ins(%cst_1 : f32) outs(%29 : memref<1x64xf32>)
              linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%19 : memref<1x64x128xf32>) outs(%29 : memref<1x64xf32>) {
              ^bb0(%in: f32, %out: f32):
                %46 = arith.addf %in, %out : f32
                linalg.yield %46 : f32
              }
              linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%25, %27 : memref<1x64xf32>, memref<1x64xf32>) outs(%31 : memref<1x64xf32>) {
              ^bb0(%in: f32, %in_7: f32, %out: f32):
                %46 = arith.subf %in, %in_7 : f32
                %47 = math.powf %cst, %46 : f32
                linalg.yield %47 : f32
              }
              linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%23, %31, %29 : memref<1x64xf32>, memref<1x64xf32>, memref<1x64xf32>) outs(%23 : memref<1x64xf32>) {
              ^bb0(%in: f32, %in_7: f32, %in_8: f32, %out: f32):
                %46 = arith.mulf %in, %in_7 : f32
                %47 = arith.addf %46, %in_8 : f32
                linalg.yield %47 : f32
              }
              loom.semaphore_give %29 : memref<1x64xf32>
              %44 = arith.muli %arg8, %c16384 : index
              %45 = arith.addi %39, %44 : index
              %reinterpret_cast_6 = memref.reinterpret_cast %arg1 to offset: [%45], sizes: [1, 128, 128], strides: [524288, 128, 1] : memref<32x4096x128xf32> to memref<1x128x128xf32, strided<[524288, 128, 1], offset: ?>>
              loom.copy %reinterpret_cast_6, %17 src_mem_space @DRAM dst_mem_space @L1, interconnect : [@vertical_links], broadcast : [8, 1] : memref<1x128x128xf32, strided<[524288, 128, 1], offset: ?>>, memref<1x128x128xf32>
              linalg.fill ins(%cst_1 : f32) outs(%38 : memref<1x64x128xf32>)
              linalg.batch_matmul ins(%19, %17 : memref<1x64x128xf32>, memref<1x128x128xf32>) outs(%38 : memref<1x64x128xf32>)
              loom.semaphore_give %17 : memref<1x128x128xf32>
              loom.semaphore_give %19 : memref<1x64x128xf32>
              linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%38, %36, %31 : memref<1x64x128xf32>, memref<1x64x128xf32>, memref<1x64xf32>) outs(%36 : memref<1x64x128xf32>) {
              ^bb0(%in: f32, %in_7: f32, %in_8: f32, %out: f32):
                %46 = arith.mulf %in_7, %in_8 : f32
                %47 = arith.addf %in, %46 : f32
                linalg.yield %47 : f32
              }
              loom.semaphore_give %31 : memref<1x64xf32>
              loom.semaphore_give %38 : memref<1x64x128xf32>
              linalg.copy ins(%27 : memref<1x64xf32>) outs(%25 : memref<1x64xf32>)
              loom.semaphore_give %27 : memref<1x64xf32>
            }
            loom.semaphore_give %25 : memref<1x64xf32>
            loom.semaphore_give %34 : memref<1x64x128xf32>
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%36, %23 : memref<1x64x128xf32>, memref<1x64xf32>) outs(%33 : memref<1x64x128xf32>) {
            ^bb0(%in: f32, %in_5: f32, %out: f32):
              %42 = arith.divf %in, %in_5 : f32
              linalg.yield %42 : f32
            }
            loom.semaphore_give %23 : memref<1x64xf32>
            loom.semaphore_give %36 : memref<1x64x128xf32>
            %reinterpret_cast_4 = memref.reinterpret_cast %arg3 to offset: [%41], sizes: [1, 64, 128], strides: [524288, 128, 1] : memref<32x4096x128xf32> to memref<1x64x128xf32, strided<[524288, 128, 1], offset: ?>>
            loom.copy %33, %reinterpret_cast_4 src_mem_space @L1 dst_mem_space @DRAM, interconnect : [], broadcast : [1, 1] : memref<1x64x128xf32>, memref<1x64x128xf32, strided<[524288, 128, 1], offset: ?>>
            loom.semaphore_give %33 : memref<1x64x128xf32>
          }
        }
        scf.reduce 
      } {loom.iter_types = [#loom.iter_type<spatial>, #loom.iter_type<spatial>], loom.mapped_to_dims = [@x, @y]}
      return
    }
  }
  module attributes {loom.pass_name = "Materialize"} {
    func.func @attention__d0i0_d1i1__f01__d_v_d__BB1__BM64__BN128(%arg0: memref<32x128x4096xf32>, %arg1: memref<32x4096x128xf32>, %arg2: memref<32x4096x128xf32>, %arg3: memref<32x4096x128xf32>) {
      %c16384 = arith.constant 16384 : index
      %c8192 = arith.constant 8192 : index
      %c32 = arith.constant 32 : index
      %c524288 = arith.constant 524288 : index
      %c4 = arith.constant 4 : index
      %cst = arith.constant 2.000000e+00 : f32
      %cst_0 = arith.constant 0.12751743074602467 : f64
      %cst_1 = arith.constant 0.000000e+00 : f32
      %cst_2 = arith.constant 1.000000e+00 : f32
      %cst_3 = arith.constant 0xFF800000 : f32
      %c128 = arith.constant 128 : index
      %c0 = arith.constant 0 : index
      %c8 = arith.constant 8 : index
      %c1 = arith.constant 1 : index
      scf.parallel (%arg4, %arg5) = (%c0, %c0) to (%c8, %c8) step (%c1, %c1) {
        scf.for %arg6 = %c0 to %c4 step %c1 {
          scf.for %arg7 = %c0 to %c8 step %c1 {
            %12 = arith.muli %arg6, %c8 overflow<nsw> : index
            %13 = arith.addi %arg4, %12 : index
            %14 = arith.muli %arg7, %c8 overflow<nsw> : index
            %15 = arith.addi %arg5, %14 : index
            %16 = loom.alloc [1, 128, 128] on @L1 : memref<1x128x128xf32>
            %17 = loom.semaphore_take %16 : memref<1x128x128xf32> -> memref<1x128x128xf32>
            %18 = loom.alloc [1, 64, 128] on @L1 : memref<1x64x128xf32>
            %19 = loom.semaphore_take %18 : memref<1x64x128xf32> -> memref<1x64x128xf32>
            %20 = loom.alloc [1, 128, 128] on @L1 : memref<1x128x128xf32>
            %21 = loom.semaphore_take %20 : memref<1x128x128xf32> -> memref<1x128x128xf32>
            %22 = loom.alloc [1, 64] on @L1 : memref<1x64xf32>
            %23 = loom.semaphore_take %22 : memref<1x64xf32> -> memref<1x64xf32>
            %24 = loom.alloc [1, 64] on @L1 : memref<1x64xf32>
            %25 = loom.semaphore_take %24 : memref<1x64xf32> -> memref<1x64xf32>
            %26 = loom.alloc [1, 64] on @L1 : memref<1x64xf32>
            %27 = loom.semaphore_take %26 : memref<1x64xf32> -> memref<1x64xf32>
            %28 = loom.alloc [1, 64] on @L1 : memref<1x64xf32>
            %29 = loom.semaphore_take %28 : memref<1x64xf32> -> memref<1x64xf32>
            %30 = loom.alloc [1, 64] on @L1 : memref<1x64xf32>
            %31 = loom.semaphore_take %30 : memref<1x64xf32> -> memref<1x64xf32>
            %32 = loom.alloc [1, 64, 128] on @L1 : memref<1x64x128xf32>
            %33 = loom.semaphore_take %32 : memref<1x64x128xf32> -> memref<1x64x128xf32>
            %34 = loom.semaphore_take %32 : memref<1x64x128xf32> -> memref<1x64x128xf32>
            %35 = loom.alloc [1, 64, 128] on @L1 : memref<1x64x128xf32>
            %36 = loom.semaphore_take %35 : memref<1x64x128xf32> -> memref<1x64x128xf32>
            %37 = loom.alloc [1, 64, 128] on @L1 : memref<1x64x128xf32>
            %38 = loom.semaphore_take %37 : memref<1x64x128xf32> -> memref<1x64x128xf32>
            %39 = arith.muli %13, %c524288 overflow<nsw> : index
            %40 = arith.muli %15, %c8192 : index
            %41 = arith.addi %39, %40 : index
            %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%41], sizes: [1, 64, 128], strides: [524288, 128, 1] : memref<32x4096x128xf32> to memref<1x64x128xf32, strided<[524288, 128, 1], offset: ?>>
            loom.copy %reinterpret_cast, %34 src_mem_space @DRAM dst_mem_space @L1, interconnect : [], broadcast : [1, 1] : memref<1x64x128xf32, strided<[524288, 128, 1], offset: ?>>, memref<1x64x128xf32>
            linalg.fill ins(%cst_1 : f32) outs(%36 : memref<1x64x128xf32>)
            linalg.fill ins(%cst_2 : f32) outs(%23 : memref<1x64xf32>)
            linalg.fill ins(%cst_3 : f32) outs(%25 : memref<1x64xf32>)
            scf.for %arg8 = %c0 to %c32 step %c1 {
              %42 = arith.muli %arg8, %c128 : index
              %43 = arith.addi %39, %42 : index
              %reinterpret_cast_5 = memref.reinterpret_cast %arg0 to offset: [%43], sizes: [1, 128, 128], strides: [524288, 4096, 1] : memref<32x128x4096xf32> to memref<1x128x128xf32, strided<[524288, 4096, 1], offset: ?>>
              loom.copy %reinterpret_cast_5, %21 src_mem_space @DRAM dst_mem_space @L1, interconnect : [@vertical_links], broadcast : [8, 1] : memref<1x128x128xf32, strided<[524288, 4096, 1], offset: ?>>, memref<1x128x128xf32>
              linalg.fill ins(%cst_1 : f32) outs(%19 : memref<1x64x128xf32>)
              linalg.batch_matmul ins(%34, %21 : memref<1x64x128xf32>, memref<1x128x128xf32>) outs(%19 : memref<1x64x128xf32>)
              loom.semaphore_give %21 : memref<1x128x128xf32>
              linalg.fill ins(%cst_3 : f32) outs(%27 : memref<1x64xf32>)
              linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%19 : memref<1x64x128xf32>) outs(%27 : memref<1x64xf32>) {
              ^bb0(%in: f32, %out: f32):
                %46 = arith.maximumf %in, %out : f32
                linalg.yield %46 : f32
              }
              linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%25, %27 : memref<1x64xf32>, memref<1x64xf32>) outs(%27 : memref<1x64xf32>) {
              ^bb0(%in: f32, %in_7: f32, %out: f32):
                %46 = arith.truncf %cst_0 : f64 to f32
                %47 = arith.mulf %in_7, %46 : f32
                %48 = arith.cmpf ogt, %in, %47 : f32
                %49 = arith.select %48, %in, %47 : f32
                linalg.yield %49 : f32
              }
              linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%19, %27 : memref<1x64x128xf32>, memref<1x64xf32>) outs(%19 : memref<1x64x128xf32>) {
              ^bb0(%in: f32, %in_7: f32, %out: f32):
                %46 = arith.truncf %cst_0 : f64 to f32
                %47 = arith.mulf %in, %46 : f32
                %48 = arith.subf %47, %in_7 : f32
                %49 = math.powf %cst, %48 : f32
                linalg.yield %49 : f32
              }
              linalg.fill ins(%cst_1 : f32) outs(%29 : memref<1x64xf32>)
              linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%19 : memref<1x64x128xf32>) outs(%29 : memref<1x64xf32>) {
              ^bb0(%in: f32, %out: f32):
                %46 = arith.addf %in, %out : f32
                linalg.yield %46 : f32
              }
              linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%25, %27 : memref<1x64xf32>, memref<1x64xf32>) outs(%31 : memref<1x64xf32>) {
              ^bb0(%in: f32, %in_7: f32, %out: f32):
                %46 = arith.subf %in, %in_7 : f32
                %47 = math.powf %cst, %46 : f32
                linalg.yield %47 : f32
              }
              linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%23, %31, %29 : memref<1x64xf32>, memref<1x64xf32>, memref<1x64xf32>) outs(%23 : memref<1x64xf32>) {
              ^bb0(%in: f32, %in_7: f32, %in_8: f32, %out: f32):
                %46 = arith.mulf %in, %in_7 : f32
                %47 = arith.addf %46, %in_8 : f32
                linalg.yield %47 : f32
              }
              loom.semaphore_give %29 : memref<1x64xf32>
              %44 = arith.muli %arg8, %c16384 : index
              %45 = arith.addi %39, %44 : index
              %reinterpret_cast_6 = memref.reinterpret_cast %arg1 to offset: [%45], sizes: [1, 128, 128], strides: [524288, 128, 1] : memref<32x4096x128xf32> to memref<1x128x128xf32, strided<[524288, 128, 1], offset: ?>>
              loom.copy %reinterpret_cast_6, %17 src_mem_space @DRAM dst_mem_space @L1, interconnect : [], broadcast : [1, 1] : memref<1x128x128xf32, strided<[524288, 128, 1], offset: ?>>, memref<1x128x128xf32>
              linalg.fill ins(%cst_1 : f32) outs(%38 : memref<1x64x128xf32>)
              linalg.batch_matmul ins(%19, %17 : memref<1x64x128xf32>, memref<1x128x128xf32>) outs(%38 : memref<1x64x128xf32>)
              loom.semaphore_give %17 : memref<1x128x128xf32>
              loom.semaphore_give %19 : memref<1x64x128xf32>
              linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%38, %36, %31 : memref<1x64x128xf32>, memref<1x64x128xf32>, memref<1x64xf32>) outs(%36 : memref<1x64x128xf32>) {
              ^bb0(%in: f32, %in_7: f32, %in_8: f32, %out: f32):
                %46 = arith.mulf %in_7, %in_8 : f32
                %47 = arith.addf %in, %46 : f32
                linalg.yield %47 : f32
              }
              loom.semaphore_give %31 : memref<1x64xf32>
              loom.semaphore_give %38 : memref<1x64x128xf32>
              linalg.copy ins(%27 : memref<1x64xf32>) outs(%25 : memref<1x64xf32>)
              loom.semaphore_give %27 : memref<1x64xf32>
            }
            loom.semaphore_give %25 : memref<1x64xf32>
            loom.semaphore_give %34 : memref<1x64x128xf32>
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%36, %23 : memref<1x64x128xf32>, memref<1x64xf32>) outs(%33 : memref<1x64x128xf32>) {
            ^bb0(%in: f32, %in_5: f32, %out: f32):
              %42 = arith.divf %in, %in_5 : f32
              linalg.yield %42 : f32
            }
            loom.semaphore_give %23 : memref<1x64xf32>
            loom.semaphore_give %36 : memref<1x64x128xf32>
            %reinterpret_cast_4 = memref.reinterpret_cast %arg3 to offset: [%41], sizes: [1, 64, 128], strides: [524288, 128, 1] : memref<32x4096x128xf32> to memref<1x64x128xf32, strided<[524288, 128, 1], offset: ?>>
            loom.copy %33, %reinterpret_cast_4 src_mem_space @L1 dst_mem_space @DRAM, interconnect : [], broadcast : [1, 1] : memref<1x64x128xf32>, memref<1x64x128xf32, strided<[524288, 128, 1], offset: ?>>
            loom.semaphore_give %33 : memref<1x64x128xf32>
          }
        }
        scf.reduce 
      } {loom.iter_types = [#loom.iter_type<spatial>, #loom.iter_type<spatial>], loom.mapped_to_dims = [@x, @y]}
      return
    }
  }
  module attributes {loom.pass_name = "Materialize"} {
    func.func @attention__d0i0_d1i1__f01__d_v_v__BB1__BM64__BN128(%arg0: memref<32x128x4096xf32>, %arg1: memref<32x4096x128xf32>, %arg2: memref<32x4096x128xf32>, %arg3: memref<32x4096x128xf32>) {
      %c16384 = arith.constant 16384 : index
      %c8192 = arith.constant 8192 : index
      %c32 = arith.constant 32 : index
      %c524288 = arith.constant 524288 : index
      %c4 = arith.constant 4 : index
      %cst = arith.constant 2.000000e+00 : f32
      %cst_0 = arith.constant 0.12751743074602467 : f64
      %cst_1 = arith.constant 0.000000e+00 : f32
      %cst_2 = arith.constant 1.000000e+00 : f32
      %cst_3 = arith.constant 0xFF800000 : f32
      %c128 = arith.constant 128 : index
      %c0 = arith.constant 0 : index
      %c8 = arith.constant 8 : index
      %c1 = arith.constant 1 : index
      scf.parallel (%arg4, %arg5) = (%c0, %c0) to (%c8, %c8) step (%c1, %c1) {
        scf.for %arg6 = %c0 to %c4 step %c1 {
          scf.for %arg7 = %c0 to %c8 step %c1 {
            %12 = arith.muli %arg6, %c8 overflow<nsw> : index
            %13 = arith.addi %arg4, %12 : index
            %14 = arith.muli %arg7, %c8 overflow<nsw> : index
            %15 = arith.addi %arg5, %14 : index
            %16 = loom.alloc [1, 128, 128] on @L1 : memref<1x128x128xf32>
            %17 = loom.semaphore_take %16 : memref<1x128x128xf32> -> memref<1x128x128xf32>
            %18 = loom.alloc [1, 64, 128] on @L1 : memref<1x64x128xf32>
            %19 = loom.semaphore_take %18 : memref<1x64x128xf32> -> memref<1x64x128xf32>
            %20 = loom.alloc [1, 128, 128] on @L1 : memref<1x128x128xf32>
            %21 = loom.semaphore_take %20 : memref<1x128x128xf32> -> memref<1x128x128xf32>
            %22 = loom.alloc [1, 64] on @L1 : memref<1x64xf32>
            %23 = loom.semaphore_take %22 : memref<1x64xf32> -> memref<1x64xf32>
            %24 = loom.alloc [1, 64] on @L1 : memref<1x64xf32>
            %25 = loom.semaphore_take %24 : memref<1x64xf32> -> memref<1x64xf32>
            %26 = loom.alloc [1, 64] on @L1 : memref<1x64xf32>
            %27 = loom.semaphore_take %26 : memref<1x64xf32> -> memref<1x64xf32>
            %28 = loom.alloc [1, 64] on @L1 : memref<1x64xf32>
            %29 = loom.semaphore_take %28 : memref<1x64xf32> -> memref<1x64xf32>
            %30 = loom.alloc [1, 64] on @L1 : memref<1x64xf32>
            %31 = loom.semaphore_take %30 : memref<1x64xf32> -> memref<1x64xf32>
            %32 = loom.alloc [1, 64, 128] on @L1 : memref<1x64x128xf32>
            %33 = loom.semaphore_take %32 : memref<1x64x128xf32> -> memref<1x64x128xf32>
            %34 = loom.semaphore_take %32 : memref<1x64x128xf32> -> memref<1x64x128xf32>
            %35 = loom.alloc [1, 64, 128] on @L1 : memref<1x64x128xf32>
            %36 = loom.semaphore_take %35 : memref<1x64x128xf32> -> memref<1x64x128xf32>
            %37 = loom.alloc [1, 64, 128] on @L1 : memref<1x64x128xf32>
            %38 = loom.semaphore_take %37 : memref<1x64x128xf32> -> memref<1x64x128xf32>
            %39 = arith.muli %13, %c524288 overflow<nsw> : index
            %40 = arith.muli %15, %c8192 : index
            %41 = arith.addi %39, %40 : index
            %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%41], sizes: [1, 64, 128], strides: [524288, 128, 1] : memref<32x4096x128xf32> to memref<1x64x128xf32, strided<[524288, 128, 1], offset: ?>>
            loom.copy %reinterpret_cast, %34 src_mem_space @DRAM dst_mem_space @L1, interconnect : [], broadcast : [1, 1] : memref<1x64x128xf32, strided<[524288, 128, 1], offset: ?>>, memref<1x64x128xf32>
            linalg.fill ins(%cst_1 : f32) outs(%36 : memref<1x64x128xf32>)
            linalg.fill ins(%cst_2 : f32) outs(%23 : memref<1x64xf32>)
            linalg.fill ins(%cst_3 : f32) outs(%25 : memref<1x64xf32>)
            scf.for %arg8 = %c0 to %c32 step %c1 {
              %42 = arith.muli %arg8, %c128 : index
              %43 = arith.addi %39, %42 : index
              %reinterpret_cast_5 = memref.reinterpret_cast %arg0 to offset: [%43], sizes: [1, 128, 128], strides: [524288, 4096, 1] : memref<32x128x4096xf32> to memref<1x128x128xf32, strided<[524288, 4096, 1], offset: ?>>
              loom.copy %reinterpret_cast_5, %21 src_mem_space @DRAM dst_mem_space @L1, interconnect : [@vertical_links], broadcast : [8, 1] : memref<1x128x128xf32, strided<[524288, 4096, 1], offset: ?>>, memref<1x128x128xf32>
              linalg.fill ins(%cst_1 : f32) outs(%19 : memref<1x64x128xf32>)
              linalg.batch_matmul ins(%34, %21 : memref<1x64x128xf32>, memref<1x128x128xf32>) outs(%19 : memref<1x64x128xf32>)
              loom.semaphore_give %21 : memref<1x128x128xf32>
              linalg.fill ins(%cst_3 : f32) outs(%27 : memref<1x64xf32>)
              linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%19 : memref<1x64x128xf32>) outs(%27 : memref<1x64xf32>) {
              ^bb0(%in: f32, %out: f32):
                %46 = arith.maximumf %in, %out : f32
                linalg.yield %46 : f32
              }
              linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%25, %27 : memref<1x64xf32>, memref<1x64xf32>) outs(%27 : memref<1x64xf32>) {
              ^bb0(%in: f32, %in_7: f32, %out: f32):
                %46 = arith.truncf %cst_0 : f64 to f32
                %47 = arith.mulf %in_7, %46 : f32
                %48 = arith.cmpf ogt, %in, %47 : f32
                %49 = arith.select %48, %in, %47 : f32
                linalg.yield %49 : f32
              }
              linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%19, %27 : memref<1x64x128xf32>, memref<1x64xf32>) outs(%19 : memref<1x64x128xf32>) {
              ^bb0(%in: f32, %in_7: f32, %out: f32):
                %46 = arith.truncf %cst_0 : f64 to f32
                %47 = arith.mulf %in, %46 : f32
                %48 = arith.subf %47, %in_7 : f32
                %49 = math.powf %cst, %48 : f32
                linalg.yield %49 : f32
              }
              linalg.fill ins(%cst_1 : f32) outs(%29 : memref<1x64xf32>)
              linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%19 : memref<1x64x128xf32>) outs(%29 : memref<1x64xf32>) {
              ^bb0(%in: f32, %out: f32):
                %46 = arith.addf %in, %out : f32
                linalg.yield %46 : f32
              }
              linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%25, %27 : memref<1x64xf32>, memref<1x64xf32>) outs(%31 : memref<1x64xf32>) {
              ^bb0(%in: f32, %in_7: f32, %out: f32):
                %46 = arith.subf %in, %in_7 : f32
                %47 = math.powf %cst, %46 : f32
                linalg.yield %47 : f32
              }
              linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%23, %31, %29 : memref<1x64xf32>, memref<1x64xf32>, memref<1x64xf32>) outs(%23 : memref<1x64xf32>) {
              ^bb0(%in: f32, %in_7: f32, %in_8: f32, %out: f32):
                %46 = arith.mulf %in, %in_7 : f32
                %47 = arith.addf %46, %in_8 : f32
                linalg.yield %47 : f32
              }
              loom.semaphore_give %29 : memref<1x64xf32>
              %44 = arith.muli %arg8, %c16384 : index
              %45 = arith.addi %39, %44 : index
              %reinterpret_cast_6 = memref.reinterpret_cast %arg1 to offset: [%45], sizes: [1, 128, 128], strides: [524288, 128, 1] : memref<32x4096x128xf32> to memref<1x128x128xf32, strided<[524288, 128, 1], offset: ?>>
              loom.copy %reinterpret_cast_6, %17 src_mem_space @DRAM dst_mem_space @L1, interconnect : [@vertical_links], broadcast : [8, 1] : memref<1x128x128xf32, strided<[524288, 128, 1], offset: ?>>, memref<1x128x128xf32>
              linalg.fill ins(%cst_1 : f32) outs(%38 : memref<1x64x128xf32>)
              linalg.batch_matmul ins(%19, %17 : memref<1x64x128xf32>, memref<1x128x128xf32>) outs(%38 : memref<1x64x128xf32>)
              loom.semaphore_give %17 : memref<1x128x128xf32>
              loom.semaphore_give %19 : memref<1x64x128xf32>
              linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%38, %36, %31 : memref<1x64x128xf32>, memref<1x64x128xf32>, memref<1x64xf32>) outs(%36 : memref<1x64x128xf32>) {
              ^bb0(%in: f32, %in_7: f32, %in_8: f32, %out: f32):
                %46 = arith.mulf %in_7, %in_8 : f32
                %47 = arith.addf %in, %46 : f32
                linalg.yield %47 : f32
              }
              loom.semaphore_give %31 : memref<1x64xf32>
              loom.semaphore_give %38 : memref<1x64x128xf32>
              linalg.copy ins(%27 : memref<1x64xf32>) outs(%25 : memref<1x64xf32>)
              loom.semaphore_give %27 : memref<1x64xf32>
            }
            loom.semaphore_give %25 : memref<1x64xf32>
            loom.semaphore_give %34 : memref<1x64x128xf32>
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%36, %23 : memref<1x64x128xf32>, memref<1x64xf32>) outs(%33 : memref<1x64x128xf32>) {
            ^bb0(%in: f32, %in_5: f32, %out: f32):
              %42 = arith.divf %in, %in_5 : f32
              linalg.yield %42 : f32
            }
            loom.semaphore_give %23 : memref<1x64xf32>
            loom.semaphore_give %36 : memref<1x64x128xf32>
            %reinterpret_cast_4 = memref.reinterpret_cast %arg3 to offset: [%41], sizes: [1, 64, 128], strides: [524288, 128, 1] : memref<32x4096x128xf32> to memref<1x64x128xf32, strided<[524288, 128, 1], offset: ?>>
            loom.copy %33, %reinterpret_cast_4 src_mem_space @L1 dst_mem_space @DRAM, interconnect : [], broadcast : [1, 1] : memref<1x64x128xf32>, memref<1x64x128xf32, strided<[524288, 128, 1], offset: ?>>
            loom.semaphore_give %33 : memref<1x64x128xf32>
          }
        }
        scf.reduce 
      } {loom.iter_types = [#loom.iter_type<spatial>, #loom.iter_type<spatial>], loom.mapped_to_dims = [@x, @y]}
      return
    }
  }
  module attributes {loom.pass_name = "Materialize"} {
    func.func @attention__d0i0_d1i1__f10__d_d_d__BB1__BM64__BN128(%arg0: memref<32x128x4096xf32>, %arg1: memref<32x4096x128xf32>, %arg2: memref<32x4096x128xf32>, %arg3: memref<32x4096x128xf32>) {
      %c16384 = arith.constant 16384 : index
      %c8192 = arith.constant 8192 : index
      %c32 = arith.constant 32 : index
      %c524288 = arith.constant 524288 : index
      %c4 = arith.constant 4 : index
      %cst = arith.constant 2.000000e+00 : f32
      %cst_0 = arith.constant 0.12751743074602467 : f64
      %cst_1 = arith.constant 0.000000e+00 : f32
      %cst_2 = arith.constant 1.000000e+00 : f32
      %cst_3 = arith.constant 0xFF800000 : f32
      %c128 = arith.constant 128 : index
      %c0 = arith.constant 0 : index
      %c8 = arith.constant 8 : index
      %c1 = arith.constant 1 : index
      scf.parallel (%arg4, %arg5) = (%c0, %c0) to (%c8, %c8) step (%c1, %c1) {
        scf.for %arg6 = %c0 to %c8 step %c1 {
          scf.for %arg7 = %c0 to %c4 step %c1 {
            %12 = arith.muli %arg7, %c8 overflow<nsw> : index
            %13 = arith.addi %arg4, %12 : index
            %14 = arith.muli %arg6, %c8 overflow<nsw> : index
            %15 = arith.addi %arg5, %14 : index
            %16 = loom.alloc [1, 128, 128] on @L1 : memref<1x128x128xf32>
            %17 = loom.semaphore_take %16 : memref<1x128x128xf32> -> memref<1x128x128xf32>
            %18 = loom.alloc [1, 64, 128] on @L1 : memref<1x64x128xf32>
            %19 = loom.semaphore_take %18 : memref<1x64x128xf32> -> memref<1x64x128xf32>
            %20 = loom.alloc [1, 128, 128] on @L1 : memref<1x128x128xf32>
            %21 = loom.semaphore_take %20 : memref<1x128x128xf32> -> memref<1x128x128xf32>
            %22 = loom.alloc [1, 64] on @L1 : memref<1x64xf32>
            %23 = loom.semaphore_take %22 : memref<1x64xf32> -> memref<1x64xf32>
            %24 = loom.alloc [1, 64] on @L1 : memref<1x64xf32>
            %25 = loom.semaphore_take %24 : memref<1x64xf32> -> memref<1x64xf32>
            %26 = loom.alloc [1, 64] on @L1 : memref<1x64xf32>
            %27 = loom.semaphore_take %26 : memref<1x64xf32> -> memref<1x64xf32>
            %28 = loom.alloc [1, 64] on @L1 : memref<1x64xf32>
            %29 = loom.semaphore_take %28 : memref<1x64xf32> -> memref<1x64xf32>
            %30 = loom.alloc [1, 64] on @L1 : memref<1x64xf32>
            %31 = loom.semaphore_take %30 : memref<1x64xf32> -> memref<1x64xf32>
            %32 = loom.alloc [1, 64, 128] on @L1 : memref<1x64x128xf32>
            %33 = loom.semaphore_take %32 : memref<1x64x128xf32> -> memref<1x64x128xf32>
            %34 = loom.semaphore_take %32 : memref<1x64x128xf32> -> memref<1x64x128xf32>
            %35 = loom.alloc [1, 64, 128] on @L1 : memref<1x64x128xf32>
            %36 = loom.semaphore_take %35 : memref<1x64x128xf32> -> memref<1x64x128xf32>
            %37 = loom.alloc [1, 64, 128] on @L1 : memref<1x64x128xf32>
            %38 = loom.semaphore_take %37 : memref<1x64x128xf32> -> memref<1x64x128xf32>
            %39 = arith.muli %13, %c524288 overflow<nsw> : index
            %40 = arith.muli %15, %c8192 : index
            %41 = arith.addi %39, %40 : index
            %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%41], sizes: [1, 64, 128], strides: [524288, 128, 1] : memref<32x4096x128xf32> to memref<1x64x128xf32, strided<[524288, 128, 1], offset: ?>>
            loom.copy %reinterpret_cast, %34 src_mem_space @DRAM dst_mem_space @L1, interconnect : [], broadcast : [1, 1] : memref<1x64x128xf32, strided<[524288, 128, 1], offset: ?>>, memref<1x64x128xf32>
            linalg.fill ins(%cst_1 : f32) outs(%36 : memref<1x64x128xf32>)
            linalg.fill ins(%cst_2 : f32) outs(%23 : memref<1x64xf32>)
            linalg.fill ins(%cst_3 : f32) outs(%25 : memref<1x64xf32>)
            scf.for %arg8 = %c0 to %c32 step %c1 {
              %42 = arith.muli %arg8, %c128 : index
              %43 = arith.addi %39, %42 : index
              %reinterpret_cast_5 = memref.reinterpret_cast %arg0 to offset: [%43], sizes: [1, 128, 128], strides: [524288, 4096, 1] : memref<32x128x4096xf32> to memref<1x128x128xf32, strided<[524288, 4096, 1], offset: ?>>
              loom.copy %reinterpret_cast_5, %21 src_mem_space @DRAM dst_mem_space @L1, interconnect : [], broadcast : [1, 1] : memref<1x128x128xf32, strided<[524288, 4096, 1], offset: ?>>, memref<1x128x128xf32>
              linalg.fill ins(%cst_1 : f32) outs(%19 : memref<1x64x128xf32>)
              linalg.batch_matmul ins(%34, %21 : memref<1x64x128xf32>, memref<1x128x128xf32>) outs(%19 : memref<1x64x128xf32>)
              loom.semaphore_give %21 : memref<1x128x128xf32>
              linalg.fill ins(%cst_3 : f32) outs(%27 : memref<1x64xf32>)
              linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%19 : memref<1x64x128xf32>) outs(%27 : memref<1x64xf32>) {
              ^bb0(%in: f32, %out: f32):
                %46 = arith.maximumf %in, %out : f32
                linalg.yield %46 : f32
              }
              linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%25, %27 : memref<1x64xf32>, memref<1x64xf32>) outs(%27 : memref<1x64xf32>) {
              ^bb0(%in: f32, %in_7: f32, %out: f32):
                %46 = arith.truncf %cst_0 : f64 to f32
                %47 = arith.mulf %in_7, %46 : f32
                %48 = arith.cmpf ogt, %in, %47 : f32
                %49 = arith.select %48, %in, %47 : f32
                linalg.yield %49 : f32
              }
              linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%19, %27 : memref<1x64x128xf32>, memref<1x64xf32>) outs(%19 : memref<1x64x128xf32>) {
              ^bb0(%in: f32, %in_7: f32, %out: f32):
                %46 = arith.truncf %cst_0 : f64 to f32
                %47 = arith.mulf %in, %46 : f32
                %48 = arith.subf %47, %in_7 : f32
                %49 = math.powf %cst, %48 : f32
                linalg.yield %49 : f32
              }
              linalg.fill ins(%cst_1 : f32) outs(%29 : memref<1x64xf32>)
              linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%19 : memref<1x64x128xf32>) outs(%29 : memref<1x64xf32>) {
              ^bb0(%in: f32, %out: f32):
                %46 = arith.addf %in, %out : f32
                linalg.yield %46 : f32
              }
              linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%25, %27 : memref<1x64xf32>, memref<1x64xf32>) outs(%31 : memref<1x64xf32>) {
              ^bb0(%in: f32, %in_7: f32, %out: f32):
                %46 = arith.subf %in, %in_7 : f32
                %47 = math.powf %cst, %46 : f32
                linalg.yield %47 : f32
              }
              linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%23, %31, %29 : memref<1x64xf32>, memref<1x64xf32>, memref<1x64xf32>) outs(%23 : memref<1x64xf32>) {
              ^bb0(%in: f32, %in_7: f32, %in_8: f32, %out: f32):
                %46 = arith.mulf %in, %in_7 : f32
                %47 = arith.addf %46, %in_8 : f32
                linalg.yield %47 : f32
              }
              loom.semaphore_give %29 : memref<1x64xf32>
              %44 = arith.muli %arg8, %c16384 : index
              %45 = arith.addi %39, %44 : index
              %reinterpret_cast_6 = memref.reinterpret_cast %arg1 to offset: [%45], sizes: [1, 128, 128], strides: [524288, 128, 1] : memref<32x4096x128xf32> to memref<1x128x128xf32, strided<[524288, 128, 1], offset: ?>>
              loom.copy %reinterpret_cast_6, %17 src_mem_space @DRAM dst_mem_space @L1, interconnect : [], broadcast : [1, 1] : memref<1x128x128xf32, strided<[524288, 128, 1], offset: ?>>, memref<1x128x128xf32>
              linalg.fill ins(%cst_1 : f32) outs(%38 : memref<1x64x128xf32>)
              linalg.batch_matmul ins(%19, %17 : memref<1x64x128xf32>, memref<1x128x128xf32>) outs(%38 : memref<1x64x128xf32>)
              loom.semaphore_give %17 : memref<1x128x128xf32>
              loom.semaphore_give %19 : memref<1x64x128xf32>
              linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%38, %36, %31 : memref<1x64x128xf32>, memref<1x64x128xf32>, memref<1x64xf32>) outs(%36 : memref<1x64x128xf32>) {
              ^bb0(%in: f32, %in_7: f32, %in_8: f32, %out: f32):
                %46 = arith.mulf %in_7, %in_8 : f32
                %47 = arith.addf %in, %46 : f32
                linalg.yield %47 : f32
              }
              loom.semaphore_give %31 : memref<1x64xf32>
              loom.semaphore_give %38 : memref<1x64x128xf32>
              linalg.copy ins(%27 : memref<1x64xf32>) outs(%25 : memref<1x64xf32>)
              loom.semaphore_give %27 : memref<1x64xf32>
            }
            loom.semaphore_give %25 : memref<1x64xf32>
            loom.semaphore_give %34 : memref<1x64x128xf32>
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%36, %23 : memref<1x64x128xf32>, memref<1x64xf32>) outs(%33 : memref<1x64x128xf32>) {
            ^bb0(%in: f32, %in_5: f32, %out: f32):
              %42 = arith.divf %in, %in_5 : f32
              linalg.yield %42 : f32
            }
            loom.semaphore_give %23 : memref<1x64xf32>
            loom.semaphore_give %36 : memref<1x64x128xf32>
            %reinterpret_cast_4 = memref.reinterpret_cast %arg3 to offset: [%41], sizes: [1, 64, 128], strides: [524288, 128, 1] : memref<32x4096x128xf32> to memref<1x64x128xf32, strided<[524288, 128, 1], offset: ?>>
            loom.copy %33, %reinterpret_cast_4 src_mem_space @L1 dst_mem_space @DRAM, interconnect : [], broadcast : [1, 1] : memref<1x64x128xf32>, memref<1x64x128xf32, strided<[524288, 128, 1], offset: ?>>
            loom.semaphore_give %33 : memref<1x64x128xf32>
          }
        }
        scf.reduce 
      } {loom.iter_types = [#loom.iter_type<spatial>, #loom.iter_type<spatial>], loom.mapped_to_dims = [@x, @y]}
      return
    }
  }
  module attributes {loom.pass_name = "Materialize"} {
    func.func @attention__d0i0_d1i1__f10__d_d_v__BB1__BM64__BN128(%arg0: memref<32x128x4096xf32>, %arg1: memref<32x4096x128xf32>, %arg2: memref<32x4096x128xf32>, %arg3: memref<32x4096x128xf32>) {
      %c16384 = arith.constant 16384 : index
      %c8192 = arith.constant 8192 : index
      %c32 = arith.constant 32 : index
      %c524288 = arith.constant 524288 : index
      %c4 = arith.constant 4 : index
      %cst = arith.constant 2.000000e+00 : f32
      %cst_0 = arith.constant 0.12751743074602467 : f64
      %cst_1 = arith.constant 0.000000e+00 : f32
      %cst_2 = arith.constant 1.000000e+00 : f32
      %cst_3 = arith.constant 0xFF800000 : f32
      %c128 = arith.constant 128 : index
      %c0 = arith.constant 0 : index
      %c8 = arith.constant 8 : index
      %c1 = arith.constant 1 : index
      scf.parallel (%arg4, %arg5) = (%c0, %c0) to (%c8, %c8) step (%c1, %c1) {
        scf.for %arg6 = %c0 to %c8 step %c1 {
          scf.for %arg7 = %c0 to %c4 step %c1 {
            %12 = arith.muli %arg7, %c8 overflow<nsw> : index
            %13 = arith.addi %arg4, %12 : index
            %14 = arith.muli %arg6, %c8 overflow<nsw> : index
            %15 = arith.addi %arg5, %14 : index
            %16 = loom.alloc [1, 128, 128] on @L1 : memref<1x128x128xf32>
            %17 = loom.semaphore_take %16 : memref<1x128x128xf32> -> memref<1x128x128xf32>
            %18 = loom.alloc [1, 64, 128] on @L1 : memref<1x64x128xf32>
            %19 = loom.semaphore_take %18 : memref<1x64x128xf32> -> memref<1x64x128xf32>
            %20 = loom.alloc [1, 128, 128] on @L1 : memref<1x128x128xf32>
            %21 = loom.semaphore_take %20 : memref<1x128x128xf32> -> memref<1x128x128xf32>
            %22 = loom.alloc [1, 64] on @L1 : memref<1x64xf32>
            %23 = loom.semaphore_take %22 : memref<1x64xf32> -> memref<1x64xf32>
            %24 = loom.alloc [1, 64] on @L1 : memref<1x64xf32>
            %25 = loom.semaphore_take %24 : memref<1x64xf32> -> memref<1x64xf32>
            %26 = loom.alloc [1, 64] on @L1 : memref<1x64xf32>
            %27 = loom.semaphore_take %26 : memref<1x64xf32> -> memref<1x64xf32>
            %28 = loom.alloc [1, 64] on @L1 : memref<1x64xf32>
            %29 = loom.semaphore_take %28 : memref<1x64xf32> -> memref<1x64xf32>
            %30 = loom.alloc [1, 64] on @L1 : memref<1x64xf32>
            %31 = loom.semaphore_take %30 : memref<1x64xf32> -> memref<1x64xf32>
            %32 = loom.alloc [1, 64, 128] on @L1 : memref<1x64x128xf32>
            %33 = loom.semaphore_take %32 : memref<1x64x128xf32> -> memref<1x64x128xf32>
            %34 = loom.semaphore_take %32 : memref<1x64x128xf32> -> memref<1x64x128xf32>
            %35 = loom.alloc [1, 64, 128] on @L1 : memref<1x64x128xf32>
            %36 = loom.semaphore_take %35 : memref<1x64x128xf32> -> memref<1x64x128xf32>
            %37 = loom.alloc [1, 64, 128] on @L1 : memref<1x64x128xf32>
            %38 = loom.semaphore_take %37 : memref<1x64x128xf32> -> memref<1x64x128xf32>
            %39 = arith.muli %13, %c524288 overflow<nsw> : index
            %40 = arith.muli %15, %c8192 : index
            %41 = arith.addi %39, %40 : index
            %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%41], sizes: [1, 64, 128], strides: [524288, 128, 1] : memref<32x4096x128xf32> to memref<1x64x128xf32, strided<[524288, 128, 1], offset: ?>>
            loom.copy %reinterpret_cast, %34 src_mem_space @DRAM dst_mem_space @L1, interconnect : [], broadcast : [1, 1] : memref<1x64x128xf32, strided<[524288, 128, 1], offset: ?>>, memref<1x64x128xf32>
            linalg.fill ins(%cst_1 : f32) outs(%36 : memref<1x64x128xf32>)
            linalg.fill ins(%cst_2 : f32) outs(%23 : memref<1x64xf32>)
            linalg.fill ins(%cst_3 : f32) outs(%25 : memref<1x64xf32>)
            scf.for %arg8 = %c0 to %c32 step %c1 {
              %42 = arith.muli %arg8, %c128 : index
              %43 = arith.addi %39, %42 : index
              %reinterpret_cast_5 = memref.reinterpret_cast %arg0 to offset: [%43], sizes: [1, 128, 128], strides: [524288, 4096, 1] : memref<32x128x4096xf32> to memref<1x128x128xf32, strided<[524288, 4096, 1], offset: ?>>
              loom.copy %reinterpret_cast_5, %21 src_mem_space @DRAM dst_mem_space @L1, interconnect : [], broadcast : [1, 1] : memref<1x128x128xf32, strided<[524288, 4096, 1], offset: ?>>, memref<1x128x128xf32>
              linalg.fill ins(%cst_1 : f32) outs(%19 : memref<1x64x128xf32>)
              linalg.batch_matmul ins(%34, %21 : memref<1x64x128xf32>, memref<1x128x128xf32>) outs(%19 : memref<1x64x128xf32>)
              loom.semaphore_give %21 : memref<1x128x128xf32>
              linalg.fill ins(%cst_3 : f32) outs(%27 : memref<1x64xf32>)
              linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%19 : memref<1x64x128xf32>) outs(%27 : memref<1x64xf32>) {
              ^bb0(%in: f32, %out: f32):
                %46 = arith.maximumf %in, %out : f32
                linalg.yield %46 : f32
              }
              linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%25, %27 : memref<1x64xf32>, memref<1x64xf32>) outs(%27 : memref<1x64xf32>) {
              ^bb0(%in: f32, %in_7: f32, %out: f32):
                %46 = arith.truncf %cst_0 : f64 to f32
                %47 = arith.mulf %in_7, %46 : f32
                %48 = arith.cmpf ogt, %in, %47 : f32
                %49 = arith.select %48, %in, %47 : f32
                linalg.yield %49 : f32
              }
              linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%19, %27 : memref<1x64x128xf32>, memref<1x64xf32>) outs(%19 : memref<1x64x128xf32>) {
              ^bb0(%in: f32, %in_7: f32, %out: f32):
                %46 = arith.truncf %cst_0 : f64 to f32
                %47 = arith.mulf %in, %46 : f32
                %48 = arith.subf %47, %in_7 : f32
                %49 = math.powf %cst, %48 : f32
                linalg.yield %49 : f32
              }
              linalg.fill ins(%cst_1 : f32) outs(%29 : memref<1x64xf32>)
              linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%19 : memref<1x64x128xf32>) outs(%29 : memref<1x64xf32>) {
              ^bb0(%in: f32, %out: f32):
                %46 = arith.addf %in, %out : f32
                linalg.yield %46 : f32
              }
              linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%25, %27 : memref<1x64xf32>, memref<1x64xf32>) outs(%31 : memref<1x64xf32>) {
              ^bb0(%in: f32, %in_7: f32, %out: f32):
                %46 = arith.subf %in, %in_7 : f32
                %47 = math.powf %cst, %46 : f32
                linalg.yield %47 : f32
              }
              linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%23, %31, %29 : memref<1x64xf32>, memref<1x64xf32>, memref<1x64xf32>) outs(%23 : memref<1x64xf32>) {
              ^bb0(%in: f32, %in_7: f32, %in_8: f32, %out: f32):
                %46 = arith.mulf %in, %in_7 : f32
                %47 = arith.addf %46, %in_8 : f32
                linalg.yield %47 : f32
              }
              loom.semaphore_give %29 : memref<1x64xf32>
              %44 = arith.muli %arg8, %c16384 : index
              %45 = arith.addi %39, %44 : index
              %reinterpret_cast_6 = memref.reinterpret_cast %arg1 to offset: [%45], sizes: [1, 128, 128], strides: [524288, 128, 1] : memref<32x4096x128xf32> to memref<1x128x128xf32, strided<[524288, 128, 1], offset: ?>>
              loom.copy %reinterpret_cast_6, %17 src_mem_space @DRAM dst_mem_space @L1, interconnect : [@vertical_links], broadcast : [8, 1] : memref<1x128x128xf32, strided<[524288, 128, 1], offset: ?>>, memref<1x128x128xf32>
              linalg.fill ins(%cst_1 : f32) outs(%38 : memref<1x64x128xf32>)
              linalg.batch_matmul ins(%19, %17 : memref<1x64x128xf32>, memref<1x128x128xf32>) outs(%38 : memref<1x64x128xf32>)
              loom.semaphore_give %17 : memref<1x128x128xf32>
              loom.semaphore_give %19 : memref<1x64x128xf32>
              linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%38, %36, %31 : memref<1x64x128xf32>, memref<1x64x128xf32>, memref<1x64xf32>) outs(%36 : memref<1x64x128xf32>) {
              ^bb0(%in: f32, %in_7: f32, %in_8: f32, %out: f32):
                %46 = arith.mulf %in_7, %in_8 : f32
                %47 = arith.addf %in, %46 : f32
                linalg.yield %47 : f32
              }
              loom.semaphore_give %31 : memref<1x64xf32>
              loom.semaphore_give %38 : memref<1x64x128xf32>
              linalg.copy ins(%27 : memref<1x64xf32>) outs(%25 : memref<1x64xf32>)
              loom.semaphore_give %27 : memref<1x64xf32>
            }
            loom.semaphore_give %25 : memref<1x64xf32>
            loom.semaphore_give %34 : memref<1x64x128xf32>
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%36, %23 : memref<1x64x128xf32>, memref<1x64xf32>) outs(%33 : memref<1x64x128xf32>) {
            ^bb0(%in: f32, %in_5: f32, %out: f32):
              %42 = arith.divf %in, %in_5 : f32
              linalg.yield %42 : f32
            }
            loom.semaphore_give %23 : memref<1x64xf32>
            loom.semaphore_give %36 : memref<1x64x128xf32>
            %reinterpret_cast_4 = memref.reinterpret_cast %arg3 to offset: [%41], sizes: [1, 64, 128], strides: [524288, 128, 1] : memref<32x4096x128xf32> to memref<1x64x128xf32, strided<[524288, 128, 1], offset: ?>>
            loom.copy %33, %reinterpret_cast_4 src_mem_space @L1 dst_mem_space @DRAM, interconnect : [], broadcast : [1, 1] : memref<1x64x128xf32>, memref<1x64x128xf32, strided<[524288, 128, 1], offset: ?>>
            loom.semaphore_give %33 : memref<1x64x128xf32>
          }
        }
        scf.reduce 
      } {loom.iter_types = [#loom.iter_type<spatial>, #loom.iter_type<spatial>], loom.mapped_to_dims = [@x, @y]}
      return
    }
  }
  module attributes {loom.pass_name = "Materialize"} {
    func.func @attention__d0i0_d1i1__f10__d_v_d__BB1__BM64__BN128(%arg0: memref<32x128x4096xf32>, %arg1: memref<32x4096x128xf32>, %arg2: memref<32x4096x128xf32>, %arg3: memref<32x4096x128xf32>) {
      %c16384 = arith.constant 16384 : index
      %c8192 = arith.constant 8192 : index
      %c32 = arith.constant 32 : index
      %c524288 = arith.constant 524288 : index
      %c4 = arith.constant 4 : index
      %cst = arith.constant 2.000000e+00 : f32
      %cst_0 = arith.constant 0.12751743074602467 : f64
      %cst_1 = arith.constant 0.000000e+00 : f32
      %cst_2 = arith.constant 1.000000e+00 : f32
      %cst_3 = arith.constant 0xFF800000 : f32
      %c128 = arith.constant 128 : index
      %c0 = arith.constant 0 : index
      %c8 = arith.constant 8 : index
      %c1 = arith.constant 1 : index
      scf.parallel (%arg4, %arg5) = (%c0, %c0) to (%c8, %c8) step (%c1, %c1) {
        scf.for %arg6 = %c0 to %c8 step %c1 {
          scf.for %arg7 = %c0 to %c4 step %c1 {
            %12 = arith.muli %arg7, %c8 overflow<nsw> : index
            %13 = arith.addi %arg4, %12 : index
            %14 = arith.muli %arg6, %c8 overflow<nsw> : index
            %15 = arith.addi %arg5, %14 : index
            %16 = loom.alloc [1, 128, 128] on @L1 : memref<1x128x128xf32>
            %17 = loom.semaphore_take %16 : memref<1x128x128xf32> -> memref<1x128x128xf32>
            %18 = loom.alloc [1, 64, 128] on @L1 : memref<1x64x128xf32>
            %19 = loom.semaphore_take %18 : memref<1x64x128xf32> -> memref<1x64x128xf32>
            %20 = loom.alloc [1, 128, 128] on @L1 : memref<1x128x128xf32>
            %21 = loom.semaphore_take %20 : memref<1x128x128xf32> -> memref<1x128x128xf32>
            %22 = loom.alloc [1, 64] on @L1 : memref<1x64xf32>
            %23 = loom.semaphore_take %22 : memref<1x64xf32> -> memref<1x64xf32>
            %24 = loom.alloc [1, 64] on @L1 : memref<1x64xf32>
            %25 = loom.semaphore_take %24 : memref<1x64xf32> -> memref<1x64xf32>
            %26 = loom.alloc [1, 64] on @L1 : memref<1x64xf32>
            %27 = loom.semaphore_take %26 : memref<1x64xf32> -> memref<1x64xf32>
            %28 = loom.alloc [1, 64] on @L1 : memref<1x64xf32>
            %29 = loom.semaphore_take %28 : memref<1x64xf32> -> memref<1x64xf32>
            %30 = loom.alloc [1, 64] on @L1 : memref<1x64xf32>
            %31 = loom.semaphore_take %30 : memref<1x64xf32> -> memref<1x64xf32>
            %32 = loom.alloc [1, 64, 128] on @L1 : memref<1x64x128xf32>
            %33 = loom.semaphore_take %32 : memref<1x64x128xf32> -> memref<1x64x128xf32>
            %34 = loom.semaphore_take %32 : memref<1x64x128xf32> -> memref<1x64x128xf32>
            %35 = loom.alloc [1, 64, 128] on @L1 : memref<1x64x128xf32>
            %36 = loom.semaphore_take %35 : memref<1x64x128xf32> -> memref<1x64x128xf32>
            %37 = loom.alloc [1, 64, 128] on @L1 : memref<1x64x128xf32>
            %38 = loom.semaphore_take %37 : memref<1x64x128xf32> -> memref<1x64x128xf32>
            %39 = arith.muli %13, %c524288 overflow<nsw> : index
            %40 = arith.muli %15, %c8192 : index
            %41 = arith.addi %39, %40 : index
            %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%41], sizes: [1, 64, 128], strides: [524288, 128, 1] : memref<32x4096x128xf32> to memref<1x64x128xf32, strided<[524288, 128, 1], offset: ?>>
            loom.copy %reinterpret_cast, %34 src_mem_space @DRAM dst_mem_space @L1, interconnect : [], broadcast : [1, 1] : memref<1x64x128xf32, strided<[524288, 128, 1], offset: ?>>, memref<1x64x128xf32>
            linalg.fill ins(%cst_1 : f32) outs(%36 : memref<1x64x128xf32>)
            linalg.fill ins(%cst_2 : f32) outs(%23 : memref<1x64xf32>)
            linalg.fill ins(%cst_3 : f32) outs(%25 : memref<1x64xf32>)
            scf.for %arg8 = %c0 to %c32 step %c1 {
              %42 = arith.muli %arg8, %c128 : index
              %43 = arith.addi %39, %42 : index
              %reinterpret_cast_5 = memref.reinterpret_cast %arg0 to offset: [%43], sizes: [1, 128, 128], strides: [524288, 4096, 1] : memref<32x128x4096xf32> to memref<1x128x128xf32, strided<[524288, 4096, 1], offset: ?>>
              loom.copy %reinterpret_cast_5, %21 src_mem_space @DRAM dst_mem_space @L1, interconnect : [@vertical_links], broadcast : [8, 1] : memref<1x128x128xf32, strided<[524288, 4096, 1], offset: ?>>, memref<1x128x128xf32>
              linalg.fill ins(%cst_1 : f32) outs(%19 : memref<1x64x128xf32>)
              linalg.batch_matmul ins(%34, %21 : memref<1x64x128xf32>, memref<1x128x128xf32>) outs(%19 : memref<1x64x128xf32>)
              loom.semaphore_give %21 : memref<1x128x128xf32>
              linalg.fill ins(%cst_3 : f32) outs(%27 : memref<1x64xf32>)
              linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%19 : memref<1x64x128xf32>) outs(%27 : memref<1x64xf32>) {
              ^bb0(%in: f32, %out: f32):
                %46 = arith.maximumf %in, %out : f32
                linalg.yield %46 : f32
              }
              linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%25, %27 : memref<1x64xf32>, memref<1x64xf32>) outs(%27 : memref<1x64xf32>) {
              ^bb0(%in: f32, %in_7: f32, %out: f32):
                %46 = arith.truncf %cst_0 : f64 to f32
                %47 = arith.mulf %in_7, %46 : f32
                %48 = arith.cmpf ogt, %in, %47 : f32
                %49 = arith.select %48, %in, %47 : f32
                linalg.yield %49 : f32
              }
              linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%19, %27 : memref<1x64x128xf32>, memref<1x64xf32>) outs(%19 : memref<1x64x128xf32>) {
              ^bb0(%in: f32, %in_7: f32, %out: f32):
                %46 = arith.truncf %cst_0 : f64 to f32
                %47 = arith.mulf %in, %46 : f32
                %48 = arith.subf %47, %in_7 : f32
                %49 = math.powf %cst, %48 : f32
                linalg.yield %49 : f32
              }
              linalg.fill ins(%cst_1 : f32) outs(%29 : memref<1x64xf32>)
              linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%19 : memref<1x64x128xf32>) outs(%29 : memref<1x64xf32>) {
              ^bb0(%in: f32, %out: f32):
                %46 = arith.addf %in, %out : f32
                linalg.yield %46 : f32
              }
              linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%25, %27 : memref<1x64xf32>, memref<1x64xf32>) outs(%31 : memref<1x64xf32>) {
              ^bb0(%in: f32, %in_7: f32, %out: f32):
                %46 = arith.subf %in, %in_7 : f32
                %47 = math.powf %cst, %46 : f32
                linalg.yield %47 : f32
              }
              linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%23, %31, %29 : memref<1x64xf32>, memref<1x64xf32>, memref<1x64xf32>) outs(%23 : memref<1x64xf32>) {
              ^bb0(%in: f32, %in_7: f32, %in_8: f32, %out: f32):
                %46 = arith.mulf %in, %in_7 : f32
                %47 = arith.addf %46, %in_8 : f32
                linalg.yield %47 : f32
              }
              loom.semaphore_give %29 : memref<1x64xf32>
              %44 = arith.muli %arg8, %c16384 : index
              %45 = arith.addi %39, %44 : index
              %reinterpret_cast_6 = memref.reinterpret_cast %arg1 to offset: [%45], sizes: [1, 128, 128], strides: [524288, 128, 1] : memref<32x4096x128xf32> to memref<1x128x128xf32, strided<[524288, 128, 1], offset: ?>>
              loom.copy %reinterpret_cast_6, %17 src_mem_space @DRAM dst_mem_space @L1, interconnect : [], broadcast : [1, 1] : memref<1x128x128xf32, strided<[524288, 128, 1], offset: ?>>, memref<1x128x128xf32>
              linalg.fill ins(%cst_1 : f32) outs(%38 : memref<1x64x128xf32>)
              linalg.batch_matmul ins(%19, %17 : memref<1x64x128xf32>, memref<1x128x128xf32>) outs(%38 : memref<1x64x128xf32>)
              loom.semaphore_give %17 : memref<1x128x128xf32>
              loom.semaphore_give %19 : memref<1x64x128xf32>
              linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%38, %36, %31 : memref<1x64x128xf32>, memref<1x64x128xf32>, memref<1x64xf32>) outs(%36 : memref<1x64x128xf32>) {
              ^bb0(%in: f32, %in_7: f32, %in_8: f32, %out: f32):
                %46 = arith.mulf %in_7, %in_8 : f32
                %47 = arith.addf %in, %46 : f32
                linalg.yield %47 : f32
              }
              loom.semaphore_give %31 : memref<1x64xf32>
              loom.semaphore_give %38 : memref<1x64x128xf32>
              linalg.copy ins(%27 : memref<1x64xf32>) outs(%25 : memref<1x64xf32>)
              loom.semaphore_give %27 : memref<1x64xf32>
            }
            loom.semaphore_give %25 : memref<1x64xf32>
            loom.semaphore_give %34 : memref<1x64x128xf32>
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%36, %23 : memref<1x64x128xf32>, memref<1x64xf32>) outs(%33 : memref<1x64x128xf32>) {
            ^bb0(%in: f32, %in_5: f32, %out: f32):
              %42 = arith.divf %in, %in_5 : f32
              linalg.yield %42 : f32
            }
            loom.semaphore_give %23 : memref<1x64xf32>
            loom.semaphore_give %36 : memref<1x64x128xf32>
            %reinterpret_cast_4 = memref.reinterpret_cast %arg3 to offset: [%41], sizes: [1, 64, 128], strides: [524288, 128, 1] : memref<32x4096x128xf32> to memref<1x64x128xf32, strided<[524288, 128, 1], offset: ?>>
            loom.copy %33, %reinterpret_cast_4 src_mem_space @L1 dst_mem_space @DRAM, interconnect : [], broadcast : [1, 1] : memref<1x64x128xf32>, memref<1x64x128xf32, strided<[524288, 128, 1], offset: ?>>
            loom.semaphore_give %33 : memref<1x64x128xf32>
          }
        }
        scf.reduce 
      } {loom.iter_types = [#loom.iter_type<spatial>, #loom.iter_type<spatial>], loom.mapped_to_dims = [@x, @y]}
      return
    }
  }
  module attributes {loom.pass_name = "Materialize"} {
    func.func @attention__d0i0_d1i1__f10__d_v_v__BB1__BM64__BN128(%arg0: memref<32x128x4096xf32>, %arg1: memref<32x4096x128xf32>, %arg2: memref<32x4096x128xf32>, %arg3: memref<32x4096x128xf32>) {
      %c16384 = arith.constant 16384 : index
      %c8192 = arith.constant 8192 : index
      %c32 = arith.constant 32 : index
      %c524288 = arith.constant 524288 : index
      %c4 = arith.constant 4 : index
      %cst = arith.constant 2.000000e+00 : f32
      %cst_0 = arith.constant 0.12751743074602467 : f64
      %cst_1 = arith.constant 0.000000e+00 : f32
      %cst_2 = arith.constant 1.000000e+00 : f32
      %cst_3 = arith.constant 0xFF800000 : f32
      %c128 = arith.constant 128 : index
      %c0 = arith.constant 0 : index
      %c8 = arith.constant 8 : index
      %c1 = arith.constant 1 : index
      scf.parallel (%arg4, %arg5) = (%c0, %c0) to (%c8, %c8) step (%c1, %c1) {
        scf.for %arg6 = %c0 to %c8 step %c1 {
          scf.for %arg7 = %c0 to %c4 step %c1 {
            %12 = arith.muli %arg7, %c8 overflow<nsw> : index
            %13 = arith.addi %arg4, %12 : index
            %14 = arith.muli %arg6, %c8 overflow<nsw> : index
            %15 = arith.addi %arg5, %14 : index
            %16 = loom.alloc [1, 128, 128] on @L1 : memref<1x128x128xf32>
            %17 = loom.semaphore_take %16 : memref<1x128x128xf32> -> memref<1x128x128xf32>
            %18 = loom.alloc [1, 64, 128] on @L1 : memref<1x64x128xf32>
            %19 = loom.semaphore_take %18 : memref<1x64x128xf32> -> memref<1x64x128xf32>
            %20 = loom.alloc [1, 128, 128] on @L1 : memref<1x128x128xf32>
            %21 = loom.semaphore_take %20 : memref<1x128x128xf32> -> memref<1x128x128xf32>
            %22 = loom.alloc [1, 64] on @L1 : memref<1x64xf32>
            %23 = loom.semaphore_take %22 : memref<1x64xf32> -> memref<1x64xf32>
            %24 = loom.alloc [1, 64] on @L1 : memref<1x64xf32>
            %25 = loom.semaphore_take %24 : memref<1x64xf32> -> memref<1x64xf32>
            %26 = loom.alloc [1, 64] on @L1 : memref<1x64xf32>
            %27 = loom.semaphore_take %26 : memref<1x64xf32> -> memref<1x64xf32>
            %28 = loom.alloc [1, 64] on @L1 : memref<1x64xf32>
            %29 = loom.semaphore_take %28 : memref<1x64xf32> -> memref<1x64xf32>
            %30 = loom.alloc [1, 64] on @L1 : memref<1x64xf32>
            %31 = loom.semaphore_take %30 : memref<1x64xf32> -> memref<1x64xf32>
            %32 = loom.alloc [1, 64, 128] on @L1 : memref<1x64x128xf32>
            %33 = loom.semaphore_take %32 : memref<1x64x128xf32> -> memref<1x64x128xf32>
            %34 = loom.semaphore_take %32 : memref<1x64x128xf32> -> memref<1x64x128xf32>
            %35 = loom.alloc [1, 64, 128] on @L1 : memref<1x64x128xf32>
            %36 = loom.semaphore_take %35 : memref<1x64x128xf32> -> memref<1x64x128xf32>
            %37 = loom.alloc [1, 64, 128] on @L1 : memref<1x64x128xf32>
            %38 = loom.semaphore_take %37 : memref<1x64x128xf32> -> memref<1x64x128xf32>
            %39 = arith.muli %13, %c524288 overflow<nsw> : index
            %40 = arith.muli %15, %c8192 : index
            %41 = arith.addi %39, %40 : index
            %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%41], sizes: [1, 64, 128], strides: [524288, 128, 1] : memref<32x4096x128xf32> to memref<1x64x128xf32, strided<[524288, 128, 1], offset: ?>>
            loom.copy %reinterpret_cast, %34 src_mem_space @DRAM dst_mem_space @L1, interconnect : [], broadcast : [1, 1] : memref<1x64x128xf32, strided<[524288, 128, 1], offset: ?>>, memref<1x64x128xf32>
            linalg.fill ins(%cst_1 : f32) outs(%36 : memref<1x64x128xf32>)
            linalg.fill ins(%cst_2 : f32) outs(%23 : memref<1x64xf32>)
            linalg.fill ins(%cst_3 : f32) outs(%25 : memref<1x64xf32>)
            scf.for %arg8 = %c0 to %c32 step %c1 {
              %42 = arith.muli %arg8, %c128 : index
              %43 = arith.addi %39, %42 : index
              %reinterpret_cast_5 = memref.reinterpret_cast %arg0 to offset: [%43], sizes: [1, 128, 128], strides: [524288, 4096, 1] : memref<32x128x4096xf32> to memref<1x128x128xf32, strided<[524288, 4096, 1], offset: ?>>
              loom.copy %reinterpret_cast_5, %21 src_mem_space @DRAM dst_mem_space @L1, interconnect : [@vertical_links], broadcast : [8, 1] : memref<1x128x128xf32, strided<[524288, 4096, 1], offset: ?>>, memref<1x128x128xf32>
              linalg.fill ins(%cst_1 : f32) outs(%19 : memref<1x64x128xf32>)
              linalg.batch_matmul ins(%34, %21 : memref<1x64x128xf32>, memref<1x128x128xf32>) outs(%19 : memref<1x64x128xf32>)
              loom.semaphore_give %21 : memref<1x128x128xf32>
              linalg.fill ins(%cst_3 : f32) outs(%27 : memref<1x64xf32>)
              linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%19 : memref<1x64x128xf32>) outs(%27 : memref<1x64xf32>) {
              ^bb0(%in: f32, %out: f32):
                %46 = arith.maximumf %in, %out : f32
                linalg.yield %46 : f32
              }
              linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%25, %27 : memref<1x64xf32>, memref<1x64xf32>) outs(%27 : memref<1x64xf32>) {
              ^bb0(%in: f32, %in_7: f32, %out: f32):
                %46 = arith.truncf %cst_0 : f64 to f32
                %47 = arith.mulf %in_7, %46 : f32
                %48 = arith.cmpf ogt, %in, %47 : f32
                %49 = arith.select %48, %in, %47 : f32
                linalg.yield %49 : f32
              }
              linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%19, %27 : memref<1x64x128xf32>, memref<1x64xf32>) outs(%19 : memref<1x64x128xf32>) {
              ^bb0(%in: f32, %in_7: f32, %out: f32):
                %46 = arith.truncf %cst_0 : f64 to f32
                %47 = arith.mulf %in, %46 : f32
                %48 = arith.subf %47, %in_7 : f32
                %49 = math.powf %cst, %48 : f32
                linalg.yield %49 : f32
              }
              linalg.fill ins(%cst_1 : f32) outs(%29 : memref<1x64xf32>)
              linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%19 : memref<1x64x128xf32>) outs(%29 : memref<1x64xf32>) {
              ^bb0(%in: f32, %out: f32):
                %46 = arith.addf %in, %out : f32
                linalg.yield %46 : f32
              }
              linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%25, %27 : memref<1x64xf32>, memref<1x64xf32>) outs(%31 : memref<1x64xf32>) {
              ^bb0(%in: f32, %in_7: f32, %out: f32):
                %46 = arith.subf %in, %in_7 : f32
                %47 = math.powf %cst, %46 : f32
                linalg.yield %47 : f32
              }
              linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%23, %31, %29 : memref<1x64xf32>, memref<1x64xf32>, memref<1x64xf32>) outs(%23 : memref<1x64xf32>) {
              ^bb0(%in: f32, %in_7: f32, %in_8: f32, %out: f32):
                %46 = arith.mulf %in, %in_7 : f32
                %47 = arith.addf %46, %in_8 : f32
                linalg.yield %47 : f32
              }
              loom.semaphore_give %29 : memref<1x64xf32>
              %44 = arith.muli %arg8, %c16384 : index
              %45 = arith.addi %39, %44 : index
              %reinterpret_cast_6 = memref.reinterpret_cast %arg1 to offset: [%45], sizes: [1, 128, 128], strides: [524288, 128, 1] : memref<32x4096x128xf32> to memref<1x128x128xf32, strided<[524288, 128, 1], offset: ?>>
              loom.copy %reinterpret_cast_6, %17 src_mem_space @DRAM dst_mem_space @L1, interconnect : [@vertical_links], broadcast : [8, 1] : memref<1x128x128xf32, strided<[524288, 128, 1], offset: ?>>, memref<1x128x128xf32>
              linalg.fill ins(%cst_1 : f32) outs(%38 : memref<1x64x128xf32>)
              linalg.batch_matmul ins(%19, %17 : memref<1x64x128xf32>, memref<1x128x128xf32>) outs(%38 : memref<1x64x128xf32>)
              loom.semaphore_give %17 : memref<1x128x128xf32>
              loom.semaphore_give %19 : memref<1x64x128xf32>
              linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%38, %36, %31 : memref<1x64x128xf32>, memref<1x64x128xf32>, memref<1x64xf32>) outs(%36 : memref<1x64x128xf32>) {
              ^bb0(%in: f32, %in_7: f32, %in_8: f32, %out: f32):
                %46 = arith.mulf %in_7, %in_8 : f32
                %47 = arith.addf %in, %46 : f32
                linalg.yield %47 : f32
              }
              loom.semaphore_give %31 : memref<1x64xf32>
              loom.semaphore_give %38 : memref<1x64x128xf32>
              linalg.copy ins(%27 : memref<1x64xf32>) outs(%25 : memref<1x64xf32>)
              loom.semaphore_give %27 : memref<1x64xf32>
            }
            loom.semaphore_give %25 : memref<1x64xf32>
            loom.semaphore_give %34 : memref<1x64x128xf32>
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%36, %23 : memref<1x64x128xf32>, memref<1x64xf32>) outs(%33 : memref<1x64x128xf32>) {
            ^bb0(%in: f32, %in_5: f32, %out: f32):
              %42 = arith.divf %in, %in_5 : f32
              linalg.yield %42 : f32
            }
            loom.semaphore_give %23 : memref<1x64xf32>
            loom.semaphore_give %36 : memref<1x64x128xf32>
            %reinterpret_cast_4 = memref.reinterpret_cast %arg3 to offset: [%41], sizes: [1, 64, 128], strides: [524288, 128, 1] : memref<32x4096x128xf32> to memref<1x64x128xf32, strided<[524288, 128, 1], offset: ?>>
            loom.copy %33, %reinterpret_cast_4 src_mem_space @L1 dst_mem_space @DRAM, interconnect : [], broadcast : [1, 1] : memref<1x64x128xf32>, memref<1x64x128xf32, strided<[524288, 128, 1], offset: ?>>
            loom.semaphore_give %33 : memref<1x64x128xf32>
          }
        }
        scf.reduce 
      } {loom.iter_types = [#loom.iter_type<spatial>, #loom.iter_type<spatial>], loom.mapped_to_dims = [@x, @y]}
      return
    }
  }
  module attributes {loom.pass_name = "Materialize"} {
    func.func @attention__d1i0_d0i1__f01__d_d_d__BB1__BM64__BN128(%arg0: memref<32x128x4096xf32>, %arg1: memref<32x4096x128xf32>, %arg2: memref<32x4096x128xf32>, %arg3: memref<32x4096x128xf32>) {
      %c16384 = arith.constant 16384 : index
      %c8192 = arith.constant 8192 : index
      %c32 = arith.constant 32 : index
      %c524288 = arith.constant 524288 : index
      %c4 = arith.constant 4 : index
      %cst = arith.constant 2.000000e+00 : f32
      %cst_0 = arith.constant 0.12751743074602467 : f64
      %cst_1 = arith.constant 0.000000e+00 : f32
      %cst_2 = arith.constant 1.000000e+00 : f32
      %cst_3 = arith.constant 0xFF800000 : f32
      %c128 = arith.constant 128 : index
      %c0 = arith.constant 0 : index
      %c8 = arith.constant 8 : index
      %c1 = arith.constant 1 : index
      scf.parallel (%arg4, %arg5) = (%c0, %c0) to (%c8, %c8) step (%c1, %c1) {
        scf.for %arg6 = %c0 to %c4 step %c1 {
          scf.for %arg7 = %c0 to %c8 step %c1 {
            %12 = arith.muli %arg6, %c8 overflow<nsw> : index
            %13 = arith.addi %arg4, %12 : index
            %14 = arith.muli %arg7, %c8 overflow<nsw> : index
            %15 = arith.addi %arg5, %14 : index
            %16 = loom.alloc [1, 128, 128] on @L1 : memref<1x128x128xf32>
            %17 = loom.semaphore_take %16 : memref<1x128x128xf32> -> memref<1x128x128xf32>
            %18 = loom.alloc [1, 64, 128] on @L1 : memref<1x64x128xf32>
            %19 = loom.semaphore_take %18 : memref<1x64x128xf32> -> memref<1x64x128xf32>
            %20 = loom.alloc [1, 128, 128] on @L1 : memref<1x128x128xf32>
            %21 = loom.semaphore_take %20 : memref<1x128x128xf32> -> memref<1x128x128xf32>
            %22 = loom.alloc [1, 64] on @L1 : memref<1x64xf32>
            %23 = loom.semaphore_take %22 : memref<1x64xf32> -> memref<1x64xf32>
            %24 = loom.alloc [1, 64] on @L1 : memref<1x64xf32>
            %25 = loom.semaphore_take %24 : memref<1x64xf32> -> memref<1x64xf32>
            %26 = loom.alloc [1, 64] on @L1 : memref<1x64xf32>
            %27 = loom.semaphore_take %26 : memref<1x64xf32> -> memref<1x64xf32>
            %28 = loom.alloc [1, 64] on @L1 : memref<1x64xf32>
            %29 = loom.semaphore_take %28 : memref<1x64xf32> -> memref<1x64xf32>
            %30 = loom.alloc [1, 64] on @L1 : memref<1x64xf32>
            %31 = loom.semaphore_take %30 : memref<1x64xf32> -> memref<1x64xf32>
            %32 = loom.alloc [1, 64, 128] on @L1 : memref<1x64x128xf32>
            %33 = loom.semaphore_take %32 : memref<1x64x128xf32> -> memref<1x64x128xf32>
            %34 = loom.semaphore_take %32 : memref<1x64x128xf32> -> memref<1x64x128xf32>
            %35 = loom.alloc [1, 64, 128] on @L1 : memref<1x64x128xf32>
            %36 = loom.semaphore_take %35 : memref<1x64x128xf32> -> memref<1x64x128xf32>
            %37 = loom.alloc [1, 64, 128] on @L1 : memref<1x64x128xf32>
            %38 = loom.semaphore_take %37 : memref<1x64x128xf32> -> memref<1x64x128xf32>
            %39 = arith.muli %13, %c524288 overflow<nsw> : index
            %40 = arith.muli %15, %c8192 : index
            %41 = arith.addi %39, %40 : index
            %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%41], sizes: [1, 64, 128], strides: [524288, 128, 1] : memref<32x4096x128xf32> to memref<1x64x128xf32, strided<[524288, 128, 1], offset: ?>>
            loom.copy %reinterpret_cast, %34 src_mem_space @DRAM dst_mem_space @L1, interconnect : [], broadcast : [1, 1] : memref<1x64x128xf32, strided<[524288, 128, 1], offset: ?>>, memref<1x64x128xf32>
            linalg.fill ins(%cst_1 : f32) outs(%36 : memref<1x64x128xf32>)
            linalg.fill ins(%cst_2 : f32) outs(%23 : memref<1x64xf32>)
            linalg.fill ins(%cst_3 : f32) outs(%25 : memref<1x64xf32>)
            scf.for %arg8 = %c0 to %c32 step %c1 {
              %42 = arith.muli %arg8, %c128 : index
              %43 = arith.addi %39, %42 : index
              %reinterpret_cast_5 = memref.reinterpret_cast %arg0 to offset: [%43], sizes: [1, 128, 128], strides: [524288, 4096, 1] : memref<32x128x4096xf32> to memref<1x128x128xf32, strided<[524288, 4096, 1], offset: ?>>
              loom.copy %reinterpret_cast_5, %21 src_mem_space @DRAM dst_mem_space @L1, interconnect : [], broadcast : [1, 1] : memref<1x128x128xf32, strided<[524288, 4096, 1], offset: ?>>, memref<1x128x128xf32>
              linalg.fill ins(%cst_1 : f32) outs(%19 : memref<1x64x128xf32>)
              linalg.batch_matmul ins(%34, %21 : memref<1x64x128xf32>, memref<1x128x128xf32>) outs(%19 : memref<1x64x128xf32>)
              loom.semaphore_give %21 : memref<1x128x128xf32>
              linalg.fill ins(%cst_3 : f32) outs(%27 : memref<1x64xf32>)
              linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%19 : memref<1x64x128xf32>) outs(%27 : memref<1x64xf32>) {
              ^bb0(%in: f32, %out: f32):
                %46 = arith.maximumf %in, %out : f32
                linalg.yield %46 : f32
              }
              linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%25, %27 : memref<1x64xf32>, memref<1x64xf32>) outs(%27 : memref<1x64xf32>) {
              ^bb0(%in: f32, %in_7: f32, %out: f32):
                %46 = arith.truncf %cst_0 : f64 to f32
                %47 = arith.mulf %in_7, %46 : f32
                %48 = arith.cmpf ogt, %in, %47 : f32
                %49 = arith.select %48, %in, %47 : f32
                linalg.yield %49 : f32
              }
              linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%19, %27 : memref<1x64x128xf32>, memref<1x64xf32>) outs(%19 : memref<1x64x128xf32>) {
              ^bb0(%in: f32, %in_7: f32, %out: f32):
                %46 = arith.truncf %cst_0 : f64 to f32
                %47 = arith.mulf %in, %46 : f32
                %48 = arith.subf %47, %in_7 : f32
                %49 = math.powf %cst, %48 : f32
                linalg.yield %49 : f32
              }
              linalg.fill ins(%cst_1 : f32) outs(%29 : memref<1x64xf32>)
              linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%19 : memref<1x64x128xf32>) outs(%29 : memref<1x64xf32>) {
              ^bb0(%in: f32, %out: f32):
                %46 = arith.addf %in, %out : f32
                linalg.yield %46 : f32
              }
              linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%25, %27 : memref<1x64xf32>, memref<1x64xf32>) outs(%31 : memref<1x64xf32>) {
              ^bb0(%in: f32, %in_7: f32, %out: f32):
                %46 = arith.subf %in, %in_7 : f32
                %47 = math.powf %cst, %46 : f32
                linalg.yield %47 : f32
              }
              linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%23, %31, %29 : memref<1x64xf32>, memref<1x64xf32>, memref<1x64xf32>) outs(%23 : memref<1x64xf32>) {
              ^bb0(%in: f32, %in_7: f32, %in_8: f32, %out: f32):
                %46 = arith.mulf %in, %in_7 : f32
                %47 = arith.addf %46, %in_8 : f32
                linalg.yield %47 : f32
              }
              loom.semaphore_give %29 : memref<1x64xf32>
              %44 = arith.muli %arg8, %c16384 : index
              %45 = arith.addi %39, %44 : index
              %reinterpret_cast_6 = memref.reinterpret_cast %arg1 to offset: [%45], sizes: [1, 128, 128], strides: [524288, 128, 1] : memref<32x4096x128xf32> to memref<1x128x128xf32, strided<[524288, 128, 1], offset: ?>>
              loom.copy %reinterpret_cast_6, %17 src_mem_space @DRAM dst_mem_space @L1, interconnect : [], broadcast : [1, 1] : memref<1x128x128xf32, strided<[524288, 128, 1], offset: ?>>, memref<1x128x128xf32>
              linalg.fill ins(%cst_1 : f32) outs(%38 : memref<1x64x128xf32>)
              linalg.batch_matmul ins(%19, %17 : memref<1x64x128xf32>, memref<1x128x128xf32>) outs(%38 : memref<1x64x128xf32>)
              loom.semaphore_give %17 : memref<1x128x128xf32>
              loom.semaphore_give %19 : memref<1x64x128xf32>
              linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%38, %36, %31 : memref<1x64x128xf32>, memref<1x64x128xf32>, memref<1x64xf32>) outs(%36 : memref<1x64x128xf32>) {
              ^bb0(%in: f32, %in_7: f32, %in_8: f32, %out: f32):
                %46 = arith.mulf %in_7, %in_8 : f32
                %47 = arith.addf %in, %46 : f32
                linalg.yield %47 : f32
              }
              loom.semaphore_give %31 : memref<1x64xf32>
              loom.semaphore_give %38 : memref<1x64x128xf32>
              linalg.copy ins(%27 : memref<1x64xf32>) outs(%25 : memref<1x64xf32>)
              loom.semaphore_give %27 : memref<1x64xf32>
            }
            loom.semaphore_give %25 : memref<1x64xf32>
            loom.semaphore_give %34 : memref<1x64x128xf32>
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%36, %23 : memref<1x64x128xf32>, memref<1x64xf32>) outs(%33 : memref<1x64x128xf32>) {
            ^bb0(%in: f32, %in_5: f32, %out: f32):
              %42 = arith.divf %in, %in_5 : f32
              linalg.yield %42 : f32
            }
            loom.semaphore_give %23 : memref<1x64xf32>
            loom.semaphore_give %36 : memref<1x64x128xf32>
            %reinterpret_cast_4 = memref.reinterpret_cast %arg3 to offset: [%41], sizes: [1, 64, 128], strides: [524288, 128, 1] : memref<32x4096x128xf32> to memref<1x64x128xf32, strided<[524288, 128, 1], offset: ?>>
            loom.copy %33, %reinterpret_cast_4 src_mem_space @L1 dst_mem_space @DRAM, interconnect : [], broadcast : [1, 1] : memref<1x64x128xf32>, memref<1x64x128xf32, strided<[524288, 128, 1], offset: ?>>
            loom.semaphore_give %33 : memref<1x64x128xf32>
          }
        }
        scf.reduce 
      } {loom.iter_types = [#loom.iter_type<spatial>, #loom.iter_type<spatial>], loom.mapped_to_dims = [@y, @x]}
      return
    }
  }
  module attributes {loom.pass_name = "Materialize"} {
    func.func @attention__d1i0_d0i1__f01__d_d_h__BB1__BM64__BN128(%arg0: memref<32x128x4096xf32>, %arg1: memref<32x4096x128xf32>, %arg2: memref<32x4096x128xf32>, %arg3: memref<32x4096x128xf32>) {
      %c16384 = arith.constant 16384 : index
      %c8192 = arith.constant 8192 : index
      %c32 = arith.constant 32 : index
      %c524288 = arith.constant 524288 : index
      %c4 = arith.constant 4 : index
      %cst = arith.constant 2.000000e+00 : f32
      %cst_0 = arith.constant 0.12751743074602467 : f64
      %cst_1 = arith.constant 0.000000e+00 : f32
      %cst_2 = arith.constant 1.000000e+00 : f32
      %cst_3 = arith.constant 0xFF800000 : f32
      %c128 = arith.constant 128 : index
      %c0 = arith.constant 0 : index
      %c8 = arith.constant 8 : index
      %c1 = arith.constant 1 : index
      scf.parallel (%arg4, %arg5) = (%c0, %c0) to (%c8, %c8) step (%c1, %c1) {
        scf.for %arg6 = %c0 to %c4 step %c1 {
          scf.for %arg7 = %c0 to %c8 step %c1 {
            %12 = arith.muli %arg6, %c8 overflow<nsw> : index
            %13 = arith.addi %arg4, %12 : index
            %14 = arith.muli %arg7, %c8 overflow<nsw> : index
            %15 = arith.addi %arg5, %14 : index
            %16 = loom.alloc [1, 128, 128] on @L1 : memref<1x128x128xf32>
            %17 = loom.semaphore_take %16 : memref<1x128x128xf32> -> memref<1x128x128xf32>
            %18 = loom.alloc [1, 64, 128] on @L1 : memref<1x64x128xf32>
            %19 = loom.semaphore_take %18 : memref<1x64x128xf32> -> memref<1x64x128xf32>
            %20 = loom.alloc [1, 128, 128] on @L1 : memref<1x128x128xf32>
            %21 = loom.semaphore_take %20 : memref<1x128x128xf32> -> memref<1x128x128xf32>
            %22 = loom.alloc [1, 64] on @L1 : memref<1x64xf32>
            %23 = loom.semaphore_take %22 : memref<1x64xf32> -> memref<1x64xf32>
            %24 = loom.alloc [1, 64] on @L1 : memref<1x64xf32>
            %25 = loom.semaphore_take %24 : memref<1x64xf32> -> memref<1x64xf32>
            %26 = loom.alloc [1, 64] on @L1 : memref<1x64xf32>
            %27 = loom.semaphore_take %26 : memref<1x64xf32> -> memref<1x64xf32>
            %28 = loom.alloc [1, 64] on @L1 : memref<1x64xf32>
            %29 = loom.semaphore_take %28 : memref<1x64xf32> -> memref<1x64xf32>
            %30 = loom.alloc [1, 64] on @L1 : memref<1x64xf32>
            %31 = loom.semaphore_take %30 : memref<1x64xf32> -> memref<1x64xf32>
            %32 = loom.alloc [1, 64, 128] on @L1 : memref<1x64x128xf32>
            %33 = loom.semaphore_take %32 : memref<1x64x128xf32> -> memref<1x64x128xf32>
            %34 = loom.semaphore_take %32 : memref<1x64x128xf32> -> memref<1x64x128xf32>
            %35 = loom.alloc [1, 64, 128] on @L1 : memref<1x64x128xf32>
            %36 = loom.semaphore_take %35 : memref<1x64x128xf32> -> memref<1x64x128xf32>
            %37 = loom.alloc [1, 64, 128] on @L1 : memref<1x64x128xf32>
            %38 = loom.semaphore_take %37 : memref<1x64x128xf32> -> memref<1x64x128xf32>
            %39 = arith.muli %13, %c524288 overflow<nsw> : index
            %40 = arith.muli %15, %c8192 : index
            %41 = arith.addi %39, %40 : index
            %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%41], sizes: [1, 64, 128], strides: [524288, 128, 1] : memref<32x4096x128xf32> to memref<1x64x128xf32, strided<[524288, 128, 1], offset: ?>>
            loom.copy %reinterpret_cast, %34 src_mem_space @DRAM dst_mem_space @L1, interconnect : [], broadcast : [1, 1] : memref<1x64x128xf32, strided<[524288, 128, 1], offset: ?>>, memref<1x64x128xf32>
            linalg.fill ins(%cst_1 : f32) outs(%36 : memref<1x64x128xf32>)
            linalg.fill ins(%cst_2 : f32) outs(%23 : memref<1x64xf32>)
            linalg.fill ins(%cst_3 : f32) outs(%25 : memref<1x64xf32>)
            scf.for %arg8 = %c0 to %c32 step %c1 {
              %42 = arith.muli %arg8, %c128 : index
              %43 = arith.addi %39, %42 : index
              %reinterpret_cast_5 = memref.reinterpret_cast %arg0 to offset: [%43], sizes: [1, 128, 128], strides: [524288, 4096, 1] : memref<32x128x4096xf32> to memref<1x128x128xf32, strided<[524288, 4096, 1], offset: ?>>
              loom.copy %reinterpret_cast_5, %21 src_mem_space @DRAM dst_mem_space @L1, interconnect : [], broadcast : [1, 1] : memref<1x128x128xf32, strided<[524288, 4096, 1], offset: ?>>, memref<1x128x128xf32>
              linalg.fill ins(%cst_1 : f32) outs(%19 : memref<1x64x128xf32>)
              linalg.batch_matmul ins(%34, %21 : memref<1x64x128xf32>, memref<1x128x128xf32>) outs(%19 : memref<1x64x128xf32>)
              loom.semaphore_give %21 : memref<1x128x128xf32>
              linalg.fill ins(%cst_3 : f32) outs(%27 : memref<1x64xf32>)
              linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%19 : memref<1x64x128xf32>) outs(%27 : memref<1x64xf32>) {
              ^bb0(%in: f32, %out: f32):
                %46 = arith.maximumf %in, %out : f32
                linalg.yield %46 : f32
              }
              linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%25, %27 : memref<1x64xf32>, memref<1x64xf32>) outs(%27 : memref<1x64xf32>) {
              ^bb0(%in: f32, %in_7: f32, %out: f32):
                %46 = arith.truncf %cst_0 : f64 to f32
                %47 = arith.mulf %in_7, %46 : f32
                %48 = arith.cmpf ogt, %in, %47 : f32
                %49 = arith.select %48, %in, %47 : f32
                linalg.yield %49 : f32
              }
              linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%19, %27 : memref<1x64x128xf32>, memref<1x64xf32>) outs(%19 : memref<1x64x128xf32>) {
              ^bb0(%in: f32, %in_7: f32, %out: f32):
                %46 = arith.truncf %cst_0 : f64 to f32
                %47 = arith.mulf %in, %46 : f32
                %48 = arith.subf %47, %in_7 : f32
                %49 = math.powf %cst, %48 : f32
                linalg.yield %49 : f32
              }
              linalg.fill ins(%cst_1 : f32) outs(%29 : memref<1x64xf32>)
              linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%19 : memref<1x64x128xf32>) outs(%29 : memref<1x64xf32>) {
              ^bb0(%in: f32, %out: f32):
                %46 = arith.addf %in, %out : f32
                linalg.yield %46 : f32
              }
              linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%25, %27 : memref<1x64xf32>, memref<1x64xf32>) outs(%31 : memref<1x64xf32>) {
              ^bb0(%in: f32, %in_7: f32, %out: f32):
                %46 = arith.subf %in, %in_7 : f32
                %47 = math.powf %cst, %46 : f32
                linalg.yield %47 : f32
              }
              linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%23, %31, %29 : memref<1x64xf32>, memref<1x64xf32>, memref<1x64xf32>) outs(%23 : memref<1x64xf32>) {
              ^bb0(%in: f32, %in_7: f32, %in_8: f32, %out: f32):
                %46 = arith.mulf %in, %in_7 : f32
                %47 = arith.addf %46, %in_8 : f32
                linalg.yield %47 : f32
              }
              loom.semaphore_give %29 : memref<1x64xf32>
              %44 = arith.muli %arg8, %c16384 : index
              %45 = arith.addi %39, %44 : index
              %reinterpret_cast_6 = memref.reinterpret_cast %arg1 to offset: [%45], sizes: [1, 128, 128], strides: [524288, 128, 1] : memref<32x4096x128xf32> to memref<1x128x128xf32, strided<[524288, 128, 1], offset: ?>>
              loom.copy %reinterpret_cast_6, %17 src_mem_space @DRAM dst_mem_space @L1, interconnect : [@horizontal_links], broadcast : [1, 8] : memref<1x128x128xf32, strided<[524288, 128, 1], offset: ?>>, memref<1x128x128xf32>
              linalg.fill ins(%cst_1 : f32) outs(%38 : memref<1x64x128xf32>)
              linalg.batch_matmul ins(%19, %17 : memref<1x64x128xf32>, memref<1x128x128xf32>) outs(%38 : memref<1x64x128xf32>)
              loom.semaphore_give %17 : memref<1x128x128xf32>
              loom.semaphore_give %19 : memref<1x64x128xf32>
              linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%38, %36, %31 : memref<1x64x128xf32>, memref<1x64x128xf32>, memref<1x64xf32>) outs(%36 : memref<1x64x128xf32>) {
              ^bb0(%in: f32, %in_7: f32, %in_8: f32, %out: f32):
                %46 = arith.mulf %in_7, %in_8 : f32
                %47 = arith.addf %in, %46 : f32
                linalg.yield %47 : f32
              }
              loom.semaphore_give %31 : memref<1x64xf32>
              loom.semaphore_give %38 : memref<1x64x128xf32>
              linalg.copy ins(%27 : memref<1x64xf32>) outs(%25 : memref<1x64xf32>)
              loom.semaphore_give %27 : memref<1x64xf32>
            }
            loom.semaphore_give %25 : memref<1x64xf32>
            loom.semaphore_give %34 : memref<1x64x128xf32>
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%36, %23 : memref<1x64x128xf32>, memref<1x64xf32>) outs(%33 : memref<1x64x128xf32>) {
            ^bb0(%in: f32, %in_5: f32, %out: f32):
              %42 = arith.divf %in, %in_5 : f32
              linalg.yield %42 : f32
            }
            loom.semaphore_give %23 : memref<1x64xf32>
            loom.semaphore_give %36 : memref<1x64x128xf32>
            %reinterpret_cast_4 = memref.reinterpret_cast %arg3 to offset: [%41], sizes: [1, 64, 128], strides: [524288, 128, 1] : memref<32x4096x128xf32> to memref<1x64x128xf32, strided<[524288, 128, 1], offset: ?>>
            loom.copy %33, %reinterpret_cast_4 src_mem_space @L1 dst_mem_space @DRAM, interconnect : [], broadcast : [1, 1] : memref<1x64x128xf32>, memref<1x64x128xf32, strided<[524288, 128, 1], offset: ?>>
            loom.semaphore_give %33 : memref<1x64x128xf32>
          }
        }
        scf.reduce 
      } {loom.iter_types = [#loom.iter_type<spatial>, #loom.iter_type<spatial>], loom.mapped_to_dims = [@y, @x]}
      return
    }
  }
  module attributes {loom.pass_name = "Materialize"} {
    func.func @attention__d1i0_d0i1__f01__d_h_d__BB1__BM64__BN128(%arg0: memref<32x128x4096xf32>, %arg1: memref<32x4096x128xf32>, %arg2: memref<32x4096x128xf32>, %arg3: memref<32x4096x128xf32>) {
      %c16384 = arith.constant 16384 : index
      %c8192 = arith.constant 8192 : index
      %c32 = arith.constant 32 : index
      %c524288 = arith.constant 524288 : index
      %c4 = arith.constant 4 : index
      %cst = arith.constant 2.000000e+00 : f32
      %cst_0 = arith.constant 0.12751743074602467 : f64
      %cst_1 = arith.constant 0.000000e+00 : f32
      %cst_2 = arith.constant 1.000000e+00 : f32
      %cst_3 = arith.constant 0xFF800000 : f32
      %c128 = arith.constant 128 : index
      %c0 = arith.constant 0 : index
      %c8 = arith.constant 8 : index
      %c1 = arith.constant 1 : index
      scf.parallel (%arg4, %arg5) = (%c0, %c0) to (%c8, %c8) step (%c1, %c1) {
        scf.for %arg6 = %c0 to %c4 step %c1 {
          scf.for %arg7 = %c0 to %c8 step %c1 {
            %12 = arith.muli %arg6, %c8 overflow<nsw> : index
            %13 = arith.addi %arg4, %12 : index
            %14 = arith.muli %arg7, %c8 overflow<nsw> : index
            %15 = arith.addi %arg5, %14 : index
            %16 = loom.alloc [1, 128, 128] on @L1 : memref<1x128x128xf32>
            %17 = loom.semaphore_take %16 : memref<1x128x128xf32> -> memref<1x128x128xf32>
            %18 = loom.alloc [1, 64, 128] on @L1 : memref<1x64x128xf32>
            %19 = loom.semaphore_take %18 : memref<1x64x128xf32> -> memref<1x64x128xf32>
            %20 = loom.alloc [1, 128, 128] on @L1 : memref<1x128x128xf32>
            %21 = loom.semaphore_take %20 : memref<1x128x128xf32> -> memref<1x128x128xf32>
            %22 = loom.alloc [1, 64] on @L1 : memref<1x64xf32>
            %23 = loom.semaphore_take %22 : memref<1x64xf32> -> memref<1x64xf32>
            %24 = loom.alloc [1, 64] on @L1 : memref<1x64xf32>
            %25 = loom.semaphore_take %24 : memref<1x64xf32> -> memref<1x64xf32>
            %26 = loom.alloc [1, 64] on @L1 : memref<1x64xf32>
            %27 = loom.semaphore_take %26 : memref<1x64xf32> -> memref<1x64xf32>
            %28 = loom.alloc [1, 64] on @L1 : memref<1x64xf32>
            %29 = loom.semaphore_take %28 : memref<1x64xf32> -> memref<1x64xf32>
            %30 = loom.alloc [1, 64] on @L1 : memref<1x64xf32>
            %31 = loom.semaphore_take %30 : memref<1x64xf32> -> memref<1x64xf32>
            %32 = loom.alloc [1, 64, 128] on @L1 : memref<1x64x128xf32>
            %33 = loom.semaphore_take %32 : memref<1x64x128xf32> -> memref<1x64x128xf32>
            %34 = loom.semaphore_take %32 : memref<1x64x128xf32> -> memref<1x64x128xf32>
            %35 = loom.alloc [1, 64, 128] on @L1 : memref<1x64x128xf32>
            %36 = loom.semaphore_take %35 : memref<1x64x128xf32> -> memref<1x64x128xf32>
            %37 = loom.alloc [1, 64, 128] on @L1 : memref<1x64x128xf32>
            %38 = loom.semaphore_take %37 : memref<1x64x128xf32> -> memref<1x64x128xf32>
            %39 = arith.muli %13, %c524288 overflow<nsw> : index
            %40 = arith.muli %15, %c8192 : index
            %41 = arith.addi %39, %40 : index
            %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%41], sizes: [1, 64, 128], strides: [524288, 128, 1] : memref<32x4096x128xf32> to memref<1x64x128xf32, strided<[524288, 128, 1], offset: ?>>
            loom.copy %reinterpret_cast, %34 src_mem_space @DRAM dst_mem_space @L1, interconnect : [], broadcast : [1, 1] : memref<1x64x128xf32, strided<[524288, 128, 1], offset: ?>>, memref<1x64x128xf32>
            linalg.fill ins(%cst_1 : f32) outs(%36 : memref<1x64x128xf32>)
            linalg.fill ins(%cst_2 : f32) outs(%23 : memref<1x64xf32>)
            linalg.fill ins(%cst_3 : f32) outs(%25 : memref<1x64xf32>)
            scf.for %arg8 = %c0 to %c32 step %c1 {
              %42 = arith.muli %arg8, %c128 : index
              %43 = arith.addi %39, %42 : index
              %reinterpret_cast_5 = memref.reinterpret_cast %arg0 to offset: [%43], sizes: [1, 128, 128], strides: [524288, 4096, 1] : memref<32x128x4096xf32> to memref<1x128x128xf32, strided<[524288, 4096, 1], offset: ?>>
              loom.copy %reinterpret_cast_5, %21 src_mem_space @DRAM dst_mem_space @L1, interconnect : [@horizontal_links], broadcast : [1, 8] : memref<1x128x128xf32, strided<[524288, 4096, 1], offset: ?>>, memref<1x128x128xf32>
              linalg.fill ins(%cst_1 : f32) outs(%19 : memref<1x64x128xf32>)
              linalg.batch_matmul ins(%34, %21 : memref<1x64x128xf32>, memref<1x128x128xf32>) outs(%19 : memref<1x64x128xf32>)
              loom.semaphore_give %21 : memref<1x128x128xf32>
              linalg.fill ins(%cst_3 : f32) outs(%27 : memref<1x64xf32>)
              linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%19 : memref<1x64x128xf32>) outs(%27 : memref<1x64xf32>) {
              ^bb0(%in: f32, %out: f32):
                %46 = arith.maximumf %in, %out : f32
                linalg.yield %46 : f32
              }
              linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%25, %27 : memref<1x64xf32>, memref<1x64xf32>) outs(%27 : memref<1x64xf32>) {
              ^bb0(%in: f32, %in_7: f32, %out: f32):
                %46 = arith.truncf %cst_0 : f64 to f32
                %47 = arith.mulf %in_7, %46 : f32
                %48 = arith.cmpf ogt, %in, %47 : f32
                %49 = arith.select %48, %in, %47 : f32
                linalg.yield %49 : f32
              }
              linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%19, %27 : memref<1x64x128xf32>, memref<1x64xf32>) outs(%19 : memref<1x64x128xf32>) {
              ^bb0(%in: f32, %in_7: f32, %out: f32):
                %46 = arith.truncf %cst_0 : f64 to f32
                %47 = arith.mulf %in, %46 : f32
                %48 = arith.subf %47, %in_7 : f32
                %49 = math.powf %cst, %48 : f32
                linalg.yield %49 : f32
              }
              linalg.fill ins(%cst_1 : f32) outs(%29 : memref<1x64xf32>)
              linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%19 : memref<1x64x128xf32>) outs(%29 : memref<1x64xf32>) {
              ^bb0(%in: f32, %out: f32):
                %46 = arith.addf %in, %out : f32
                linalg.yield %46 : f32
              }
              linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%25, %27 : memref<1x64xf32>, memref<1x64xf32>) outs(%31 : memref<1x64xf32>) {
              ^bb0(%in: f32, %in_7: f32, %out: f32):
                %46 = arith.subf %in, %in_7 : f32
                %47 = math.powf %cst, %46 : f32
                linalg.yield %47 : f32
              }
              linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%23, %31, %29 : memref<1x64xf32>, memref<1x64xf32>, memref<1x64xf32>) outs(%23 : memref<1x64xf32>) {
              ^bb0(%in: f32, %in_7: f32, %in_8: f32, %out: f32):
                %46 = arith.mulf %in, %in_7 : f32
                %47 = arith.addf %46, %in_8 : f32
                linalg.yield %47 : f32
              }
              loom.semaphore_give %29 : memref<1x64xf32>
              %44 = arith.muli %arg8, %c16384 : index
              %45 = arith.addi %39, %44 : index
              %reinterpret_cast_6 = memref.reinterpret_cast %arg1 to offset: [%45], sizes: [1, 128, 128], strides: [524288, 128, 1] : memref<32x4096x128xf32> to memref<1x128x128xf32, strided<[524288, 128, 1], offset: ?>>
              loom.copy %reinterpret_cast_6, %17 src_mem_space @DRAM dst_mem_space @L1, interconnect : [], broadcast : [1, 1] : memref<1x128x128xf32, strided<[524288, 128, 1], offset: ?>>, memref<1x128x128xf32>
              linalg.fill ins(%cst_1 : f32) outs(%38 : memref<1x64x128xf32>)
              linalg.batch_matmul ins(%19, %17 : memref<1x64x128xf32>, memref<1x128x128xf32>) outs(%38 : memref<1x64x128xf32>)
              loom.semaphore_give %17 : memref<1x128x128xf32>
              loom.semaphore_give %19 : memref<1x64x128xf32>
              linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%38, %36, %31 : memref<1x64x128xf32>, memref<1x64x128xf32>, memref<1x64xf32>) outs(%36 : memref<1x64x128xf32>) {
              ^bb0(%in: f32, %in_7: f32, %in_8: f32, %out: f32):
                %46 = arith.mulf %in_7, %in_8 : f32
                %47 = arith.addf %in, %46 : f32
                linalg.yield %47 : f32
              }
              loom.semaphore_give %31 : memref<1x64xf32>
              loom.semaphore_give %38 : memref<1x64x128xf32>
              linalg.copy ins(%27 : memref<1x64xf32>) outs(%25 : memref<1x64xf32>)
              loom.semaphore_give %27 : memref<1x64xf32>
            }
            loom.semaphore_give %25 : memref<1x64xf32>
            loom.semaphore_give %34 : memref<1x64x128xf32>
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%36, %23 : memref<1x64x128xf32>, memref<1x64xf32>) outs(%33 : memref<1x64x128xf32>) {
            ^bb0(%in: f32, %in_5: f32, %out: f32):
              %42 = arith.divf %in, %in_5 : f32
              linalg.yield %42 : f32
            }
            loom.semaphore_give %23 : memref<1x64xf32>
            loom.semaphore_give %36 : memref<1x64x128xf32>
            %reinterpret_cast_4 = memref.reinterpret_cast %arg3 to offset: [%41], sizes: [1, 64, 128], strides: [524288, 128, 1] : memref<32x4096x128xf32> to memref<1x64x128xf32, strided<[524288, 128, 1], offset: ?>>
            loom.copy %33, %reinterpret_cast_4 src_mem_space @L1 dst_mem_space @DRAM, interconnect : [], broadcast : [1, 1] : memref<1x64x128xf32>, memref<1x64x128xf32, strided<[524288, 128, 1], offset: ?>>
            loom.semaphore_give %33 : memref<1x64x128xf32>
          }
        }
        scf.reduce 
      } {loom.iter_types = [#loom.iter_type<spatial>, #loom.iter_type<spatial>], loom.mapped_to_dims = [@y, @x]}
      return
    }
  }
  module attributes {loom.pass_name = "Materialize"} {
    func.func @attention__d1i0_d0i1__f01__d_h_h__BB1__BM64__BN128(%arg0: memref<32x128x4096xf32>, %arg1: memref<32x4096x128xf32>, %arg2: memref<32x4096x128xf32>, %arg3: memref<32x4096x128xf32>) {
      %c16384 = arith.constant 16384 : index
      %c8192 = arith.constant 8192 : index
      %c32 = arith.constant 32 : index
      %c524288 = arith.constant 524288 : index
      %c4 = arith.constant 4 : index
      %cst = arith.constant 2.000000e+00 : f32
      %cst_0 = arith.constant 0.12751743074602467 : f64
      %cst_1 = arith.constant 0.000000e+00 : f32
      %cst_2 = arith.constant 1.000000e+00 : f32
      %cst_3 = arith.constant 0xFF800000 : f32
      %c128 = arith.constant 128 : index
      %c0 = arith.constant 0 : index
      %c8 = arith.constant 8 : index
      %c1 = arith.constant 1 : index
      scf.parallel (%arg4, %arg5) = (%c0, %c0) to (%c8, %c8) step (%c1, %c1) {
        scf.for %arg6 = %c0 to %c4 step %c1 {
          scf.for %arg7 = %c0 to %c8 step %c1 {
            %12 = arith.muli %arg6, %c8 overflow<nsw> : index
            %13 = arith.addi %arg4, %12 : index
            %14 = arith.muli %arg7, %c8 overflow<nsw> : index
            %15 = arith.addi %arg5, %14 : index
            %16 = loom.alloc [1, 128, 128] on @L1 : memref<1x128x128xf32>
            %17 = loom.semaphore_take %16 : memref<1x128x128xf32> -> memref<1x128x128xf32>
            %18 = loom.alloc [1, 64, 128] on @L1 : memref<1x64x128xf32>
            %19 = loom.semaphore_take %18 : memref<1x64x128xf32> -> memref<1x64x128xf32>
            %20 = loom.alloc [1, 128, 128] on @L1 : memref<1x128x128xf32>
            %21 = loom.semaphore_take %20 : memref<1x128x128xf32> -> memref<1x128x128xf32>
            %22 = loom.alloc [1, 64] on @L1 : memref<1x64xf32>
            %23 = loom.semaphore_take %22 : memref<1x64xf32> -> memref<1x64xf32>
            %24 = loom.alloc [1, 64] on @L1 : memref<1x64xf32>
            %25 = loom.semaphore_take %24 : memref<1x64xf32> -> memref<1x64xf32>
            %26 = loom.alloc [1, 64] on @L1 : memref<1x64xf32>
            %27 = loom.semaphore_take %26 : memref<1x64xf32> -> memref<1x64xf32>
            %28 = loom.alloc [1, 64] on @L1 : memref<1x64xf32>
            %29 = loom.semaphore_take %28 : memref<1x64xf32> -> memref<1x64xf32>
            %30 = loom.alloc [1, 64] on @L1 : memref<1x64xf32>
            %31 = loom.semaphore_take %30 : memref<1x64xf32> -> memref<1x64xf32>
            %32 = loom.alloc [1, 64, 128] on @L1 : memref<1x64x128xf32>
            %33 = loom.semaphore_take %32 : memref<1x64x128xf32> -> memref<1x64x128xf32>
            %34 = loom.semaphore_take %32 : memref<1x64x128xf32> -> memref<1x64x128xf32>
            %35 = loom.alloc [1, 64, 128] on @L1 : memref<1x64x128xf32>
            %36 = loom.semaphore_take %35 : memref<1x64x128xf32> -> memref<1x64x128xf32>
            %37 = loom.alloc [1, 64, 128] on @L1 : memref<1x64x128xf32>
            %38 = loom.semaphore_take %37 : memref<1x64x128xf32> -> memref<1x64x128xf32>
            %39 = arith.muli %13, %c524288 overflow<nsw> : index
            %40 = arith.muli %15, %c8192 : index
            %41 = arith.addi %39, %40 : index
            %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%41], sizes: [1, 64, 128], strides: [524288, 128, 1] : memref<32x4096x128xf32> to memref<1x64x128xf32, strided<[524288, 128, 1], offset: ?>>
            loom.copy %reinterpret_cast, %34 src_mem_space @DRAM dst_mem_space @L1, interconnect : [], broadcast : [1, 1] : memref<1x64x128xf32, strided<[524288, 128, 1], offset: ?>>, memref<1x64x128xf32>
            linalg.fill ins(%cst_1 : f32) outs(%36 : memref<1x64x128xf32>)
            linalg.fill ins(%cst_2 : f32) outs(%23 : memref<1x64xf32>)
            linalg.fill ins(%cst_3 : f32) outs(%25 : memref<1x64xf32>)
            scf.for %arg8 = %c0 to %c32 step %c1 {
              %42 = arith.muli %arg8, %c128 : index
              %43 = arith.addi %39, %42 : index
              %reinterpret_cast_5 = memref.reinterpret_cast %arg0 to offset: [%43], sizes: [1, 128, 128], strides: [524288, 4096, 1] : memref<32x128x4096xf32> to memref<1x128x128xf32, strided<[524288, 4096, 1], offset: ?>>
              loom.copy %reinterpret_cast_5, %21 src_mem_space @DRAM dst_mem_space @L1, interconnect : [@horizontal_links], broadcast : [1, 8] : memref<1x128x128xf32, strided<[524288, 4096, 1], offset: ?>>, memref<1x128x128xf32>
              linalg.fill ins(%cst_1 : f32) outs(%19 : memref<1x64x128xf32>)
              linalg.batch_matmul ins(%34, %21 : memref<1x64x128xf32>, memref<1x128x128xf32>) outs(%19 : memref<1x64x128xf32>)
              loom.semaphore_give %21 : memref<1x128x128xf32>
              linalg.fill ins(%cst_3 : f32) outs(%27 : memref<1x64xf32>)
              linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%19 : memref<1x64x128xf32>) outs(%27 : memref<1x64xf32>) {
              ^bb0(%in: f32, %out: f32):
                %46 = arith.maximumf %in, %out : f32
                linalg.yield %46 : f32
              }
              linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%25, %27 : memref<1x64xf32>, memref<1x64xf32>) outs(%27 : memref<1x64xf32>) {
              ^bb0(%in: f32, %in_7: f32, %out: f32):
                %46 = arith.truncf %cst_0 : f64 to f32
                %47 = arith.mulf %in_7, %46 : f32
                %48 = arith.cmpf ogt, %in, %47 : f32
                %49 = arith.select %48, %in, %47 : f32
                linalg.yield %49 : f32
              }
              linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%19, %27 : memref<1x64x128xf32>, memref<1x64xf32>) outs(%19 : memref<1x64x128xf32>) {
              ^bb0(%in: f32, %in_7: f32, %out: f32):
                %46 = arith.truncf %cst_0 : f64 to f32
                %47 = arith.mulf %in, %46 : f32
                %48 = arith.subf %47, %in_7 : f32
                %49 = math.powf %cst, %48 : f32
                linalg.yield %49 : f32
              }
              linalg.fill ins(%cst_1 : f32) outs(%29 : memref<1x64xf32>)
              linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%19 : memref<1x64x128xf32>) outs(%29 : memref<1x64xf32>) {
              ^bb0(%in: f32, %out: f32):
                %46 = arith.addf %in, %out : f32
                linalg.yield %46 : f32
              }
              linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%25, %27 : memref<1x64xf32>, memref<1x64xf32>) outs(%31 : memref<1x64xf32>) {
              ^bb0(%in: f32, %in_7: f32, %out: f32):
                %46 = arith.subf %in, %in_7 : f32
                %47 = math.powf %cst, %46 : f32
                linalg.yield %47 : f32
              }
              linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%23, %31, %29 : memref<1x64xf32>, memref<1x64xf32>, memref<1x64xf32>) outs(%23 : memref<1x64xf32>) {
              ^bb0(%in: f32, %in_7: f32, %in_8: f32, %out: f32):
                %46 = arith.mulf %in, %in_7 : f32
                %47 = arith.addf %46, %in_8 : f32
                linalg.yield %47 : f32
              }
              loom.semaphore_give %29 : memref<1x64xf32>
              %44 = arith.muli %arg8, %c16384 : index
              %45 = arith.addi %39, %44 : index
              %reinterpret_cast_6 = memref.reinterpret_cast %arg1 to offset: [%45], sizes: [1, 128, 128], strides: [524288, 128, 1] : memref<32x4096x128xf32> to memref<1x128x128xf32, strided<[524288, 128, 1], offset: ?>>
              loom.copy %reinterpret_cast_6, %17 src_mem_space @DRAM dst_mem_space @L1, interconnect : [@horizontal_links], broadcast : [1, 8] : memref<1x128x128xf32, strided<[524288, 128, 1], offset: ?>>, memref<1x128x128xf32>
              linalg.fill ins(%cst_1 : f32) outs(%38 : memref<1x64x128xf32>)
              linalg.batch_matmul ins(%19, %17 : memref<1x64x128xf32>, memref<1x128x128xf32>) outs(%38 : memref<1x64x128xf32>)
              loom.semaphore_give %17 : memref<1x128x128xf32>
              loom.semaphore_give %19 : memref<1x64x128xf32>
              linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%38, %36, %31 : memref<1x64x128xf32>, memref<1x64x128xf32>, memref<1x64xf32>) outs(%36 : memref<1x64x128xf32>) {
              ^bb0(%in: f32, %in_7: f32, %in_8: f32, %out: f32):
                %46 = arith.mulf %in_7, %in_8 : f32
                %47 = arith.addf %in, %46 : f32
                linalg.yield %47 : f32
              }
              loom.semaphore_give %31 : memref<1x64xf32>
              loom.semaphore_give %38 : memref<1x64x128xf32>
              linalg.copy ins(%27 : memref<1x64xf32>) outs(%25 : memref<1x64xf32>)
              loom.semaphore_give %27 : memref<1x64xf32>
            }
            loom.semaphore_give %25 : memref<1x64xf32>
            loom.semaphore_give %34 : memref<1x64x128xf32>
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%36, %23 : memref<1x64x128xf32>, memref<1x64xf32>) outs(%33 : memref<1x64x128xf32>) {
            ^bb0(%in: f32, %in_5: f32, %out: f32):
              %42 = arith.divf %in, %in_5 : f32
              linalg.yield %42 : f32
            }
            loom.semaphore_give %23 : memref<1x64xf32>
            loom.semaphore_give %36 : memref<1x64x128xf32>
            %reinterpret_cast_4 = memref.reinterpret_cast %arg3 to offset: [%41], sizes: [1, 64, 128], strides: [524288, 128, 1] : memref<32x4096x128xf32> to memref<1x64x128xf32, strided<[524288, 128, 1], offset: ?>>
            loom.copy %33, %reinterpret_cast_4 src_mem_space @L1 dst_mem_space @DRAM, interconnect : [], broadcast : [1, 1] : memref<1x64x128xf32>, memref<1x64x128xf32, strided<[524288, 128, 1], offset: ?>>
            loom.semaphore_give %33 : memref<1x64x128xf32>
          }
        }
        scf.reduce 
      } {loom.iter_types = [#loom.iter_type<spatial>, #loom.iter_type<spatial>], loom.mapped_to_dims = [@y, @x]}
      return
    }
  }
  module attributes {loom.pass_name = "Materialize"} {
    func.func @attention__d1i0_d0i1__f10__d_d_d__BB1__BM64__BN128(%arg0: memref<32x128x4096xf32>, %arg1: memref<32x4096x128xf32>, %arg2: memref<32x4096x128xf32>, %arg3: memref<32x4096x128xf32>) {
      %c16384 = arith.constant 16384 : index
      %c8192 = arith.constant 8192 : index
      %c32 = arith.constant 32 : index
      %c524288 = arith.constant 524288 : index
      %c4 = arith.constant 4 : index
      %cst = arith.constant 2.000000e+00 : f32
      %cst_0 = arith.constant 0.12751743074602467 : f64
      %cst_1 = arith.constant 0.000000e+00 : f32
      %cst_2 = arith.constant 1.000000e+00 : f32
      %cst_3 = arith.constant 0xFF800000 : f32
      %c128 = arith.constant 128 : index
      %c0 = arith.constant 0 : index
      %c8 = arith.constant 8 : index
      %c1 = arith.constant 1 : index
      scf.parallel (%arg4, %arg5) = (%c0, %c0) to (%c8, %c8) step (%c1, %c1) {
        scf.for %arg6 = %c0 to %c8 step %c1 {
          scf.for %arg7 = %c0 to %c4 step %c1 {
            %12 = arith.muli %arg7, %c8 overflow<nsw> : index
            %13 = arith.addi %arg4, %12 : index
            %14 = arith.muli %arg6, %c8 overflow<nsw> : index
            %15 = arith.addi %arg5, %14 : index
            %16 = loom.alloc [1, 128, 128] on @L1 : memref<1x128x128xf32>
            %17 = loom.semaphore_take %16 : memref<1x128x128xf32> -> memref<1x128x128xf32>
            %18 = loom.alloc [1, 64, 128] on @L1 : memref<1x64x128xf32>
            %19 = loom.semaphore_take %18 : memref<1x64x128xf32> -> memref<1x64x128xf32>
            %20 = loom.alloc [1, 128, 128] on @L1 : memref<1x128x128xf32>
            %21 = loom.semaphore_take %20 : memref<1x128x128xf32> -> memref<1x128x128xf32>
            %22 = loom.alloc [1, 64] on @L1 : memref<1x64xf32>
            %23 = loom.semaphore_take %22 : memref<1x64xf32> -> memref<1x64xf32>
            %24 = loom.alloc [1, 64] on @L1 : memref<1x64xf32>
            %25 = loom.semaphore_take %24 : memref<1x64xf32> -> memref<1x64xf32>
            %26 = loom.alloc [1, 64] on @L1 : memref<1x64xf32>
            %27 = loom.semaphore_take %26 : memref<1x64xf32> -> memref<1x64xf32>
            %28 = loom.alloc [1, 64] on @L1 : memref<1x64xf32>
            %29 = loom.semaphore_take %28 : memref<1x64xf32> -> memref<1x64xf32>
            %30 = loom.alloc [1, 64] on @L1 : memref<1x64xf32>
            %31 = loom.semaphore_take %30 : memref<1x64xf32> -> memref<1x64xf32>
            %32 = loom.alloc [1, 64, 128] on @L1 : memref<1x64x128xf32>
            %33 = loom.semaphore_take %32 : memref<1x64x128xf32> -> memref<1x64x128xf32>
            %34 = loom.semaphore_take %32 : memref<1x64x128xf32> -> memref<1x64x128xf32>
            %35 = loom.alloc [1, 64, 128] on @L1 : memref<1x64x128xf32>
            %36 = loom.semaphore_take %35 : memref<1x64x128xf32> -> memref<1x64x128xf32>
            %37 = loom.alloc [1, 64, 128] on @L1 : memref<1x64x128xf32>
            %38 = loom.semaphore_take %37 : memref<1x64x128xf32> -> memref<1x64x128xf32>
            %39 = arith.muli %13, %c524288 overflow<nsw> : index
            %40 = arith.muli %15, %c8192 : index
            %41 = arith.addi %39, %40 : index
            %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%41], sizes: [1, 64, 128], strides: [524288, 128, 1] : memref<32x4096x128xf32> to memref<1x64x128xf32, strided<[524288, 128, 1], offset: ?>>
            loom.copy %reinterpret_cast, %34 src_mem_space @DRAM dst_mem_space @L1, interconnect : [], broadcast : [1, 1] : memref<1x64x128xf32, strided<[524288, 128, 1], offset: ?>>, memref<1x64x128xf32>
            linalg.fill ins(%cst_1 : f32) outs(%36 : memref<1x64x128xf32>)
            linalg.fill ins(%cst_2 : f32) outs(%23 : memref<1x64xf32>)
            linalg.fill ins(%cst_3 : f32) outs(%25 : memref<1x64xf32>)
            scf.for %arg8 = %c0 to %c32 step %c1 {
              %42 = arith.muli %arg8, %c128 : index
              %43 = arith.addi %39, %42 : index
              %reinterpret_cast_5 = memref.reinterpret_cast %arg0 to offset: [%43], sizes: [1, 128, 128], strides: [524288, 4096, 1] : memref<32x128x4096xf32> to memref<1x128x128xf32, strided<[524288, 4096, 1], offset: ?>>
              loom.copy %reinterpret_cast_5, %21 src_mem_space @DRAM dst_mem_space @L1, interconnect : [], broadcast : [1, 1] : memref<1x128x128xf32, strided<[524288, 4096, 1], offset: ?>>, memref<1x128x128xf32>
              linalg.fill ins(%cst_1 : f32) outs(%19 : memref<1x64x128xf32>)
              linalg.batch_matmul ins(%34, %21 : memref<1x64x128xf32>, memref<1x128x128xf32>) outs(%19 : memref<1x64x128xf32>)
              loom.semaphore_give %21 : memref<1x128x128xf32>
              linalg.fill ins(%cst_3 : f32) outs(%27 : memref<1x64xf32>)
              linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%19 : memref<1x64x128xf32>) outs(%27 : memref<1x64xf32>) {
              ^bb0(%in: f32, %out: f32):
                %46 = arith.maximumf %in, %out : f32
                linalg.yield %46 : f32
              }
              linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%25, %27 : memref<1x64xf32>, memref<1x64xf32>) outs(%27 : memref<1x64xf32>) {
              ^bb0(%in: f32, %in_7: f32, %out: f32):
                %46 = arith.truncf %cst_0 : f64 to f32
                %47 = arith.mulf %in_7, %46 : f32
                %48 = arith.cmpf ogt, %in, %47 : f32
                %49 = arith.select %48, %in, %47 : f32
                linalg.yield %49 : f32
              }
              linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%19, %27 : memref<1x64x128xf32>, memref<1x64xf32>) outs(%19 : memref<1x64x128xf32>) {
              ^bb0(%in: f32, %in_7: f32, %out: f32):
                %46 = arith.truncf %cst_0 : f64 to f32
                %47 = arith.mulf %in, %46 : f32
                %48 = arith.subf %47, %in_7 : f32
                %49 = math.powf %cst, %48 : f32
                linalg.yield %49 : f32
              }
              linalg.fill ins(%cst_1 : f32) outs(%29 : memref<1x64xf32>)
              linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%19 : memref<1x64x128xf32>) outs(%29 : memref<1x64xf32>) {
              ^bb0(%in: f32, %out: f32):
                %46 = arith.addf %in, %out : f32
                linalg.yield %46 : f32
              }
              linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%25, %27 : memref<1x64xf32>, memref<1x64xf32>) outs(%31 : memref<1x64xf32>) {
              ^bb0(%in: f32, %in_7: f32, %out: f32):
                %46 = arith.subf %in, %in_7 : f32
                %47 = math.powf %cst, %46 : f32
                linalg.yield %47 : f32
              }
              linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%23, %31, %29 : memref<1x64xf32>, memref<1x64xf32>, memref<1x64xf32>) outs(%23 : memref<1x64xf32>) {
              ^bb0(%in: f32, %in_7: f32, %in_8: f32, %out: f32):
                %46 = arith.mulf %in, %in_7 : f32
                %47 = arith.addf %46, %in_8 : f32
                linalg.yield %47 : f32
              }
              loom.semaphore_give %29 : memref<1x64xf32>
              %44 = arith.muli %arg8, %c16384 : index
              %45 = arith.addi %39, %44 : index
              %reinterpret_cast_6 = memref.reinterpret_cast %arg1 to offset: [%45], sizes: [1, 128, 128], strides: [524288, 128, 1] : memref<32x4096x128xf32> to memref<1x128x128xf32, strided<[524288, 128, 1], offset: ?>>
              loom.copy %reinterpret_cast_6, %17 src_mem_space @DRAM dst_mem_space @L1, interconnect : [], broadcast : [1, 1] : memref<1x128x128xf32, strided<[524288, 128, 1], offset: ?>>, memref<1x128x128xf32>
              linalg.fill ins(%cst_1 : f32) outs(%38 : memref<1x64x128xf32>)
              linalg.batch_matmul ins(%19, %17 : memref<1x64x128xf32>, memref<1x128x128xf32>) outs(%38 : memref<1x64x128xf32>)
              loom.semaphore_give %17 : memref<1x128x128xf32>
              loom.semaphore_give %19 : memref<1x64x128xf32>
              linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%38, %36, %31 : memref<1x64x128xf32>, memref<1x64x128xf32>, memref<1x64xf32>) outs(%36 : memref<1x64x128xf32>) {
              ^bb0(%in: f32, %in_7: f32, %in_8: f32, %out: f32):
                %46 = arith.mulf %in_7, %in_8 : f32
                %47 = arith.addf %in, %46 : f32
                linalg.yield %47 : f32
              }
              loom.semaphore_give %31 : memref<1x64xf32>
              loom.semaphore_give %38 : memref<1x64x128xf32>
              linalg.copy ins(%27 : memref<1x64xf32>) outs(%25 : memref<1x64xf32>)
              loom.semaphore_give %27 : memref<1x64xf32>
            }
            loom.semaphore_give %25 : memref<1x64xf32>
            loom.semaphore_give %34 : memref<1x64x128xf32>
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%36, %23 : memref<1x64x128xf32>, memref<1x64xf32>) outs(%33 : memref<1x64x128xf32>) {
            ^bb0(%in: f32, %in_5: f32, %out: f32):
              %42 = arith.divf %in, %in_5 : f32
              linalg.yield %42 : f32
            }
            loom.semaphore_give %23 : memref<1x64xf32>
            loom.semaphore_give %36 : memref<1x64x128xf32>
            %reinterpret_cast_4 = memref.reinterpret_cast %arg3 to offset: [%41], sizes: [1, 64, 128], strides: [524288, 128, 1] : memref<32x4096x128xf32> to memref<1x64x128xf32, strided<[524288, 128, 1], offset: ?>>
            loom.copy %33, %reinterpret_cast_4 src_mem_space @L1 dst_mem_space @DRAM, interconnect : [], broadcast : [1, 1] : memref<1x64x128xf32>, memref<1x64x128xf32, strided<[524288, 128, 1], offset: ?>>
            loom.semaphore_give %33 : memref<1x64x128xf32>
          }
        }
        scf.reduce 
      } {loom.iter_types = [#loom.iter_type<spatial>, #loom.iter_type<spatial>], loom.mapped_to_dims = [@y, @x]}
      return
    }
  }
  module attributes {loom.pass_name = "Materialize"} {
    func.func @attention__d1i0_d0i1__f10__d_d_h__BB1__BM64__BN128(%arg0: memref<32x128x4096xf32>, %arg1: memref<32x4096x128xf32>, %arg2: memref<32x4096x128xf32>, %arg3: memref<32x4096x128xf32>) {
      %c16384 = arith.constant 16384 : index
      %c8192 = arith.constant 8192 : index
      %c32 = arith.constant 32 : index
      %c524288 = arith.constant 524288 : index
      %c4 = arith.constant 4 : index
      %cst = arith.constant 2.000000e+00 : f32
      %cst_0 = arith.constant 0.12751743074602467 : f64
      %cst_1 = arith.constant 0.000000e+00 : f32
      %cst_2 = arith.constant 1.000000e+00 : f32
      %cst_3 = arith.constant 0xFF800000 : f32
      %c128 = arith.constant 128 : index
      %c0 = arith.constant 0 : index
      %c8 = arith.constant 8 : index
      %c1 = arith.constant 1 : index
      scf.parallel (%arg4, %arg5) = (%c0, %c0) to (%c8, %c8) step (%c1, %c1) {
        scf.for %arg6 = %c0 to %c8 step %c1 {
          scf.for %arg7 = %c0 to %c4 step %c1 {
            %12 = arith.muli %arg7, %c8 overflow<nsw> : index
            %13 = arith.addi %arg4, %12 : index
            %14 = arith.muli %arg6, %c8 overflow<nsw> : index
            %15 = arith.addi %arg5, %14 : index
            %16 = loom.alloc [1, 128, 128] on @L1 : memref<1x128x128xf32>
            %17 = loom.semaphore_take %16 : memref<1x128x128xf32> -> memref<1x128x128xf32>
            %18 = loom.alloc [1, 64, 128] on @L1 : memref<1x64x128xf32>
            %19 = loom.semaphore_take %18 : memref<1x64x128xf32> -> memref<1x64x128xf32>
            %20 = loom.alloc [1, 128, 128] on @L1 : memref<1x128x128xf32>
            %21 = loom.semaphore_take %20 : memref<1x128x128xf32> -> memref<1x128x128xf32>
            %22 = loom.alloc [1, 64] on @L1 : memref<1x64xf32>
            %23 = loom.semaphore_take %22 : memref<1x64xf32> -> memref<1x64xf32>
            %24 = loom.alloc [1, 64] on @L1 : memref<1x64xf32>
            %25 = loom.semaphore_take %24 : memref<1x64xf32> -> memref<1x64xf32>
            %26 = loom.alloc [1, 64] on @L1 : memref<1x64xf32>
            %27 = loom.semaphore_take %26 : memref<1x64xf32> -> memref<1x64xf32>
            %28 = loom.alloc [1, 64] on @L1 : memref<1x64xf32>
            %29 = loom.semaphore_take %28 : memref<1x64xf32> -> memref<1x64xf32>
            %30 = loom.alloc [1, 64] on @L1 : memref<1x64xf32>
            %31 = loom.semaphore_take %30 : memref<1x64xf32> -> memref<1x64xf32>
            %32 = loom.alloc [1, 64, 128] on @L1 : memref<1x64x128xf32>
            %33 = loom.semaphore_take %32 : memref<1x64x128xf32> -> memref<1x64x128xf32>
            %34 = loom.semaphore_take %32 : memref<1x64x128xf32> -> memref<1x64x128xf32>
            %35 = loom.alloc [1, 64, 128] on @L1 : memref<1x64x128xf32>
            %36 = loom.semaphore_take %35 : memref<1x64x128xf32> -> memref<1x64x128xf32>
            %37 = loom.alloc [1, 64, 128] on @L1 : memref<1x64x128xf32>
            %38 = loom.semaphore_take %37 : memref<1x64x128xf32> -> memref<1x64x128xf32>
            %39 = arith.muli %13, %c524288 overflow<nsw> : index
            %40 = arith.muli %15, %c8192 : index
            %41 = arith.addi %39, %40 : index
            %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%41], sizes: [1, 64, 128], strides: [524288, 128, 1] : memref<32x4096x128xf32> to memref<1x64x128xf32, strided<[524288, 128, 1], offset: ?>>
            loom.copy %reinterpret_cast, %34 src_mem_space @DRAM dst_mem_space @L1, interconnect : [], broadcast : [1, 1] : memref<1x64x128xf32, strided<[524288, 128, 1], offset: ?>>, memref<1x64x128xf32>
            linalg.fill ins(%cst_1 : f32) outs(%36 : memref<1x64x128xf32>)
            linalg.fill ins(%cst_2 : f32) outs(%23 : memref<1x64xf32>)
            linalg.fill ins(%cst_3 : f32) outs(%25 : memref<1x64xf32>)
            scf.for %arg8 = %c0 to %c32 step %c1 {
              %42 = arith.muli %arg8, %c128 : index
              %43 = arith.addi %39, %42 : index
              %reinterpret_cast_5 = memref.reinterpret_cast %arg0 to offset: [%43], sizes: [1, 128, 128], strides: [524288, 4096, 1] : memref<32x128x4096xf32> to memref<1x128x128xf32, strided<[524288, 4096, 1], offset: ?>>
              loom.copy %reinterpret_cast_5, %21 src_mem_space @DRAM dst_mem_space @L1, interconnect : [], broadcast : [1, 1] : memref<1x128x128xf32, strided<[524288, 4096, 1], offset: ?>>, memref<1x128x128xf32>
              linalg.fill ins(%cst_1 : f32) outs(%19 : memref<1x64x128xf32>)
              linalg.batch_matmul ins(%34, %21 : memref<1x64x128xf32>, memref<1x128x128xf32>) outs(%19 : memref<1x64x128xf32>)
              loom.semaphore_give %21 : memref<1x128x128xf32>
              linalg.fill ins(%cst_3 : f32) outs(%27 : memref<1x64xf32>)
              linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%19 : memref<1x64x128xf32>) outs(%27 : memref<1x64xf32>) {
              ^bb0(%in: f32, %out: f32):
                %46 = arith.maximumf %in, %out : f32
                linalg.yield %46 : f32
              }
              linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%25, %27 : memref<1x64xf32>, memref<1x64xf32>) outs(%27 : memref<1x64xf32>) {
              ^bb0(%in: f32, %in_7: f32, %out: f32):
                %46 = arith.truncf %cst_0 : f64 to f32
                %47 = arith.mulf %in_7, %46 : f32
                %48 = arith.cmpf ogt, %in, %47 : f32
                %49 = arith.select %48, %in, %47 : f32
                linalg.yield %49 : f32
              }
              linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%19, %27 : memref<1x64x128xf32>, memref<1x64xf32>) outs(%19 : memref<1x64x128xf32>) {
              ^bb0(%in: f32, %in_7: f32, %out: f32):
                %46 = arith.truncf %cst_0 : f64 to f32
                %47 = arith.mulf %in, %46 : f32
                %48 = arith.subf %47, %in_7 : f32
                %49 = math.powf %cst, %48 : f32
                linalg.yield %49 : f32
              }
              linalg.fill ins(%cst_1 : f32) outs(%29 : memref<1x64xf32>)
              linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%19 : memref<1x64x128xf32>) outs(%29 : memref<1x64xf32>) {
              ^bb0(%in: f32, %out: f32):
                %46 = arith.addf %in, %out : f32
                linalg.yield %46 : f32
              }
              linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%25, %27 : memref<1x64xf32>, memref<1x64xf32>) outs(%31 : memref<1x64xf32>) {
              ^bb0(%in: f32, %in_7: f32, %out: f32):
                %46 = arith.subf %in, %in_7 : f32
                %47 = math.powf %cst, %46 : f32
                linalg.yield %47 : f32
              }
              linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%23, %31, %29 : memref<1x64xf32>, memref<1x64xf32>, memref<1x64xf32>) outs(%23 : memref<1x64xf32>) {
              ^bb0(%in: f32, %in_7: f32, %in_8: f32, %out: f32):
                %46 = arith.mulf %in, %in_7 : f32
                %47 = arith.addf %46, %in_8 : f32
                linalg.yield %47 : f32
              }
              loom.semaphore_give %29 : memref<1x64xf32>
              %44 = arith.muli %arg8, %c16384 : index
              %45 = arith.addi %39, %44 : index
              %reinterpret_cast_6 = memref.reinterpret_cast %arg1 to offset: [%45], sizes: [1, 128, 128], strides: [524288, 128, 1] : memref<32x4096x128xf32> to memref<1x128x128xf32, strided<[524288, 128, 1], offset: ?>>
              loom.copy %reinterpret_cast_6, %17 src_mem_space @DRAM dst_mem_space @L1, interconnect : [@horizontal_links], broadcast : [1, 8] : memref<1x128x128xf32, strided<[524288, 128, 1], offset: ?>>, memref<1x128x128xf32>
              linalg.fill ins(%cst_1 : f32) outs(%38 : memref<1x64x128xf32>)
              linalg.batch_matmul ins(%19, %17 : memref<1x64x128xf32>, memref<1x128x128xf32>) outs(%38 : memref<1x64x128xf32>)
              loom.semaphore_give %17 : memref<1x128x128xf32>
              loom.semaphore_give %19 : memref<1x64x128xf32>
              linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%38, %36, %31 : memref<1x64x128xf32>, memref<1x64x128xf32>, memref<1x64xf32>) outs(%36 : memref<1x64x128xf32>) {
              ^bb0(%in: f32, %in_7: f32, %in_8: f32, %out: f32):
                %46 = arith.mulf %in_7, %in_8 : f32
                %47 = arith.addf %in, %46 : f32
                linalg.yield %47 : f32
              }
              loom.semaphore_give %31 : memref<1x64xf32>
              loom.semaphore_give %38 : memref<1x64x128xf32>
              linalg.copy ins(%27 : memref<1x64xf32>) outs(%25 : memref<1x64xf32>)
              loom.semaphore_give %27 : memref<1x64xf32>
            }
            loom.semaphore_give %25 : memref<1x64xf32>
            loom.semaphore_give %34 : memref<1x64x128xf32>
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%36, %23 : memref<1x64x128xf32>, memref<1x64xf32>) outs(%33 : memref<1x64x128xf32>) {
            ^bb0(%in: f32, %in_5: f32, %out: f32):
              %42 = arith.divf %in, %in_5 : f32
              linalg.yield %42 : f32
            }
            loom.semaphore_give %23 : memref<1x64xf32>
            loom.semaphore_give %36 : memref<1x64x128xf32>
            %reinterpret_cast_4 = memref.reinterpret_cast %arg3 to offset: [%41], sizes: [1, 64, 128], strides: [524288, 128, 1] : memref<32x4096x128xf32> to memref<1x64x128xf32, strided<[524288, 128, 1], offset: ?>>
            loom.copy %33, %reinterpret_cast_4 src_mem_space @L1 dst_mem_space @DRAM, interconnect : [], broadcast : [1, 1] : memref<1x64x128xf32>, memref<1x64x128xf32, strided<[524288, 128, 1], offset: ?>>
            loom.semaphore_give %33 : memref<1x64x128xf32>
          }
        }
        scf.reduce 
      } {loom.iter_types = [#loom.iter_type<spatial>, #loom.iter_type<spatial>], loom.mapped_to_dims = [@y, @x]}
      return
    }
  }
  module attributes {loom.pass_name = "Materialize"} {
    func.func @attention__d1i0_d0i1__f10__d_h_d__BB1__BM64__BN128(%arg0: memref<32x128x4096xf32>, %arg1: memref<32x4096x128xf32>, %arg2: memref<32x4096x128xf32>, %arg3: memref<32x4096x128xf32>) {
      %c16384 = arith.constant 16384 : index
      %c8192 = arith.constant 8192 : index
      %c32 = arith.constant 32 : index
      %c524288 = arith.constant 524288 : index
      %c4 = arith.constant 4 : index
      %cst = arith.constant 2.000000e+00 : f32
      %cst_0 = arith.constant 0.12751743074602467 : f64
      %cst_1 = arith.constant 0.000000e+00 : f32
      %cst_2 = arith.constant 1.000000e+00 : f32
      %cst_3 = arith.constant 0xFF800000 : f32
      %c128 = arith.constant 128 : index
      %c0 = arith.constant 0 : index
      %c8 = arith.constant 8 : index
      %c1 = arith.constant 1 : index
      scf.parallel (%arg4, %arg5) = (%c0, %c0) to (%c8, %c8) step (%c1, %c1) {
        scf.for %arg6 = %c0 to %c8 step %c1 {
          scf.for %arg7 = %c0 to %c4 step %c1 {
            %12 = arith.muli %arg7, %c8 overflow<nsw> : index
            %13 = arith.addi %arg4, %12 : index
            %14 = arith.muli %arg6, %c8 overflow<nsw> : index
            %15 = arith.addi %arg5, %14 : index
            %16 = loom.alloc [1, 128, 128] on @L1 : memref<1x128x128xf32>
            %17 = loom.semaphore_take %16 : memref<1x128x128xf32> -> memref<1x128x128xf32>
            %18 = loom.alloc [1, 64, 128] on @L1 : memref<1x64x128xf32>
            %19 = loom.semaphore_take %18 : memref<1x64x128xf32> -> memref<1x64x128xf32>
            %20 = loom.alloc [1, 128, 128] on @L1 : memref<1x128x128xf32>
            %21 = loom.semaphore_take %20 : memref<1x128x128xf32> -> memref<1x128x128xf32>
            %22 = loom.alloc [1, 64] on @L1 : memref<1x64xf32>
            %23 = loom.semaphore_take %22 : memref<1x64xf32> -> memref<1x64xf32>
            %24 = loom.alloc [1, 64] on @L1 : memref<1x64xf32>
            %25 = loom.semaphore_take %24 : memref<1x64xf32> -> memref<1x64xf32>
            %26 = loom.alloc [1, 64] on @L1 : memref<1x64xf32>
            %27 = loom.semaphore_take %26 : memref<1x64xf32> -> memref<1x64xf32>
            %28 = loom.alloc [1, 64] on @L1 : memref<1x64xf32>
            %29 = loom.semaphore_take %28 : memref<1x64xf32> -> memref<1x64xf32>
            %30 = loom.alloc [1, 64] on @L1 : memref<1x64xf32>
            %31 = loom.semaphore_take %30 : memref<1x64xf32> -> memref<1x64xf32>
            %32 = loom.alloc [1, 64, 128] on @L1 : memref<1x64x128xf32>
            %33 = loom.semaphore_take %32 : memref<1x64x128xf32> -> memref<1x64x128xf32>
            %34 = loom.semaphore_take %32 : memref<1x64x128xf32> -> memref<1x64x128xf32>
            %35 = loom.alloc [1, 64, 128] on @L1 : memref<1x64x128xf32>
            %36 = loom.semaphore_take %35 : memref<1x64x128xf32> -> memref<1x64x128xf32>
            %37 = loom.alloc [1, 64, 128] on @L1 : memref<1x64x128xf32>
            %38 = loom.semaphore_take %37 : memref<1x64x128xf32> -> memref<1x64x128xf32>
            %39 = arith.muli %13, %c524288 overflow<nsw> : index
            %40 = arith.muli %15, %c8192 : index
            %41 = arith.addi %39, %40 : index
            %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%41], sizes: [1, 64, 128], strides: [524288, 128, 1] : memref<32x4096x128xf32> to memref<1x64x128xf32, strided<[524288, 128, 1], offset: ?>>
            loom.copy %reinterpret_cast, %34 src_mem_space @DRAM dst_mem_space @L1, interconnect : [], broadcast : [1, 1] : memref<1x64x128xf32, strided<[524288, 128, 1], offset: ?>>, memref<1x64x128xf32>
            linalg.fill ins(%cst_1 : f32) outs(%36 : memref<1x64x128xf32>)
            linalg.fill ins(%cst_2 : f32) outs(%23 : memref<1x64xf32>)
            linalg.fill ins(%cst_3 : f32) outs(%25 : memref<1x64xf32>)
            scf.for %arg8 = %c0 to %c32 step %c1 {
              %42 = arith.muli %arg8, %c128 : index
              %43 = arith.addi %39, %42 : index
              %reinterpret_cast_5 = memref.reinterpret_cast %arg0 to offset: [%43], sizes: [1, 128, 128], strides: [524288, 4096, 1] : memref<32x128x4096xf32> to memref<1x128x128xf32, strided<[524288, 4096, 1], offset: ?>>
              loom.copy %reinterpret_cast_5, %21 src_mem_space @DRAM dst_mem_space @L1, interconnect : [@horizontal_links], broadcast : [1, 8] : memref<1x128x128xf32, strided<[524288, 4096, 1], offset: ?>>, memref<1x128x128xf32>
              linalg.fill ins(%cst_1 : f32) outs(%19 : memref<1x64x128xf32>)
              linalg.batch_matmul ins(%34, %21 : memref<1x64x128xf32>, memref<1x128x128xf32>) outs(%19 : memref<1x64x128xf32>)
              loom.semaphore_give %21 : memref<1x128x128xf32>
              linalg.fill ins(%cst_3 : f32) outs(%27 : memref<1x64xf32>)
              linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%19 : memref<1x64x128xf32>) outs(%27 : memref<1x64xf32>) {
              ^bb0(%in: f32, %out: f32):
                %46 = arith.maximumf %in, %out : f32
                linalg.yield %46 : f32
              }
              linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%25, %27 : memref<1x64xf32>, memref<1x64xf32>) outs(%27 : memref<1x64xf32>) {
              ^bb0(%in: f32, %in_7: f32, %out: f32):
                %46 = arith.truncf %cst_0 : f64 to f32
                %47 = arith.mulf %in_7, %46 : f32
                %48 = arith.cmpf ogt, %in, %47 : f32
                %49 = arith.select %48, %in, %47 : f32
                linalg.yield %49 : f32
              }
              linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%19, %27 : memref<1x64x128xf32>, memref<1x64xf32>) outs(%19 : memref<1x64x128xf32>) {
              ^bb0(%in: f32, %in_7: f32, %out: f32):
                %46 = arith.truncf %cst_0 : f64 to f32
                %47 = arith.mulf %in, %46 : f32
                %48 = arith.subf %47, %in_7 : f32
                %49 = math.powf %cst, %48 : f32
                linalg.yield %49 : f32
              }
              linalg.fill ins(%cst_1 : f32) outs(%29 : memref<1x64xf32>)
              linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%19 : memref<1x64x128xf32>) outs(%29 : memref<1x64xf32>) {
              ^bb0(%in: f32, %out: f32):
                %46 = arith.addf %in, %out : f32
                linalg.yield %46 : f32
              }
              linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%25, %27 : memref<1x64xf32>, memref<1x64xf32>) outs(%31 : memref<1x64xf32>) {
              ^bb0(%in: f32, %in_7: f32, %out: f32):
                %46 = arith.subf %in, %in_7 : f32
                %47 = math.powf %cst, %46 : f32
                linalg.yield %47 : f32
              }
              linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%23, %31, %29 : memref<1x64xf32>, memref<1x64xf32>, memref<1x64xf32>) outs(%23 : memref<1x64xf32>) {
              ^bb0(%in: f32, %in_7: f32, %in_8: f32, %out: f32):
                %46 = arith.mulf %in, %in_7 : f32
                %47 = arith.addf %46, %in_8 : f32
                linalg.yield %47 : f32
              }
              loom.semaphore_give %29 : memref<1x64xf32>
              %44 = arith.muli %arg8, %c16384 : index
              %45 = arith.addi %39, %44 : index
              %reinterpret_cast_6 = memref.reinterpret_cast %arg1 to offset: [%45], sizes: [1, 128, 128], strides: [524288, 128, 1] : memref<32x4096x128xf32> to memref<1x128x128xf32, strided<[524288, 128, 1], offset: ?>>
              loom.copy %reinterpret_cast_6, %17 src_mem_space @DRAM dst_mem_space @L1, interconnect : [], broadcast : [1, 1] : memref<1x128x128xf32, strided<[524288, 128, 1], offset: ?>>, memref<1x128x128xf32>
              linalg.fill ins(%cst_1 : f32) outs(%38 : memref<1x64x128xf32>)
              linalg.batch_matmul ins(%19, %17 : memref<1x64x128xf32>, memref<1x128x128xf32>) outs(%38 : memref<1x64x128xf32>)
              loom.semaphore_give %17 : memref<1x128x128xf32>
              loom.semaphore_give %19 : memref<1x64x128xf32>
              linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%38, %36, %31 : memref<1x64x128xf32>, memref<1x64x128xf32>, memref<1x64xf32>) outs(%36 : memref<1x64x128xf32>) {
              ^bb0(%in: f32, %in_7: f32, %in_8: f32, %out: f32):
                %46 = arith.mulf %in_7, %in_8 : f32
                %47 = arith.addf %in, %46 : f32
                linalg.yield %47 : f32
              }
              loom.semaphore_give %31 : memref<1x64xf32>
              loom.semaphore_give %38 : memref<1x64x128xf32>
              linalg.copy ins(%27 : memref<1x64xf32>) outs(%25 : memref<1x64xf32>)
              loom.semaphore_give %27 : memref<1x64xf32>
            }
            loom.semaphore_give %25 : memref<1x64xf32>
            loom.semaphore_give %34 : memref<1x64x128xf32>
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%36, %23 : memref<1x64x128xf32>, memref<1x64xf32>) outs(%33 : memref<1x64x128xf32>) {
            ^bb0(%in: f32, %in_5: f32, %out: f32):
              %42 = arith.divf %in, %in_5 : f32
              linalg.yield %42 : f32
            }
            loom.semaphore_give %23 : memref<1x64xf32>
            loom.semaphore_give %36 : memref<1x64x128xf32>
            %reinterpret_cast_4 = memref.reinterpret_cast %arg3 to offset: [%41], sizes: [1, 64, 128], strides: [524288, 128, 1] : memref<32x4096x128xf32> to memref<1x64x128xf32, strided<[524288, 128, 1], offset: ?>>
            loom.copy %33, %reinterpret_cast_4 src_mem_space @L1 dst_mem_space @DRAM, interconnect : [], broadcast : [1, 1] : memref<1x64x128xf32>, memref<1x64x128xf32, strided<[524288, 128, 1], offset: ?>>
            loom.semaphore_give %33 : memref<1x64x128xf32>
          }
        }
        scf.reduce 
      } {loom.iter_types = [#loom.iter_type<spatial>, #loom.iter_type<spatial>], loom.mapped_to_dims = [@y, @x]}
      return
    }
  }
  module attributes {loom.pass_name = "Materialize"} {
    func.func @attention__d1i0_d0i1__f10__d_h_h__BB1__BM64__BN128(%arg0: memref<32x128x4096xf32>, %arg1: memref<32x4096x128xf32>, %arg2: memref<32x4096x128xf32>, %arg3: memref<32x4096x128xf32>) {
      %c16384 = arith.constant 16384 : index
      %c8192 = arith.constant 8192 : index
      %c32 = arith.constant 32 : index
      %c524288 = arith.constant 524288 : index
      %c4 = arith.constant 4 : index
      %cst = arith.constant 2.000000e+00 : f32
      %cst_0 = arith.constant 0.12751743074602467 : f64
      %cst_1 = arith.constant 0.000000e+00 : f32
      %cst_2 = arith.constant 1.000000e+00 : f32
      %cst_3 = arith.constant 0xFF800000 : f32
      %c128 = arith.constant 128 : index
      %c0 = arith.constant 0 : index
      %c8 = arith.constant 8 : index
      %c1 = arith.constant 1 : index
      scf.parallel (%arg4, %arg5) = (%c0, %c0) to (%c8, %c8) step (%c1, %c1) {
        scf.for %arg6 = %c0 to %c8 step %c1 {
          scf.for %arg7 = %c0 to %c4 step %c1 {
            %12 = arith.muli %arg7, %c8 overflow<nsw> : index
            %13 = arith.addi %arg4, %12 : index
            %14 = arith.muli %arg6, %c8 overflow<nsw> : index
            %15 = arith.addi %arg5, %14 : index
            %16 = loom.alloc [1, 128, 128] on @L1 : memref<1x128x128xf32>
            %17 = loom.semaphore_take %16 : memref<1x128x128xf32> -> memref<1x128x128xf32>
            %18 = loom.alloc [1, 64, 128] on @L1 : memref<1x64x128xf32>
            %19 = loom.semaphore_take %18 : memref<1x64x128xf32> -> memref<1x64x128xf32>
            %20 = loom.alloc [1, 128, 128] on @L1 : memref<1x128x128xf32>
            %21 = loom.semaphore_take %20 : memref<1x128x128xf32> -> memref<1x128x128xf32>
            %22 = loom.alloc [1, 64] on @L1 : memref<1x64xf32>
            %23 = loom.semaphore_take %22 : memref<1x64xf32> -> memref<1x64xf32>
            %24 = loom.alloc [1, 64] on @L1 : memref<1x64xf32>
            %25 = loom.semaphore_take %24 : memref<1x64xf32> -> memref<1x64xf32>
            %26 = loom.alloc [1, 64] on @L1 : memref<1x64xf32>
            %27 = loom.semaphore_take %26 : memref<1x64xf32> -> memref<1x64xf32>
            %28 = loom.alloc [1, 64] on @L1 : memref<1x64xf32>
            %29 = loom.semaphore_take %28 : memref<1x64xf32> -> memref<1x64xf32>
            %30 = loom.alloc [1, 64] on @L1 : memref<1x64xf32>
            %31 = loom.semaphore_take %30 : memref<1x64xf32> -> memref<1x64xf32>
            %32 = loom.alloc [1, 64, 128] on @L1 : memref<1x64x128xf32>
            %33 = loom.semaphore_take %32 : memref<1x64x128xf32> -> memref<1x64x128xf32>
            %34 = loom.semaphore_take %32 : memref<1x64x128xf32> -> memref<1x64x128xf32>
            %35 = loom.alloc [1, 64, 128] on @L1 : memref<1x64x128xf32>
            %36 = loom.semaphore_take %35 : memref<1x64x128xf32> -> memref<1x64x128xf32>
            %37 = loom.alloc [1, 64, 128] on @L1 : memref<1x64x128xf32>
            %38 = loom.semaphore_take %37 : memref<1x64x128xf32> -> memref<1x64x128xf32>
            %39 = arith.muli %13, %c524288 overflow<nsw> : index
            %40 = arith.muli %15, %c8192 : index
            %41 = arith.addi %39, %40 : index
            %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%41], sizes: [1, 64, 128], strides: [524288, 128, 1] : memref<32x4096x128xf32> to memref<1x64x128xf32, strided<[524288, 128, 1], offset: ?>>
            loom.copy %reinterpret_cast, %34 src_mem_space @DRAM dst_mem_space @L1, interconnect : [], broadcast : [1, 1] : memref<1x64x128xf32, strided<[524288, 128, 1], offset: ?>>, memref<1x64x128xf32>
            linalg.fill ins(%cst_1 : f32) outs(%36 : memref<1x64x128xf32>)
            linalg.fill ins(%cst_2 : f32) outs(%23 : memref<1x64xf32>)
            linalg.fill ins(%cst_3 : f32) outs(%25 : memref<1x64xf32>)
            scf.for %arg8 = %c0 to %c32 step %c1 {
              %42 = arith.muli %arg8, %c128 : index
              %43 = arith.addi %39, %42 : index
              %reinterpret_cast_5 = memref.reinterpret_cast %arg0 to offset: [%43], sizes: [1, 128, 128], strides: [524288, 4096, 1] : memref<32x128x4096xf32> to memref<1x128x128xf32, strided<[524288, 4096, 1], offset: ?>>
              loom.copy %reinterpret_cast_5, %21 src_mem_space @DRAM dst_mem_space @L1, interconnect : [@horizontal_links], broadcast : [1, 8] : memref<1x128x128xf32, strided<[524288, 4096, 1], offset: ?>>, memref<1x128x128xf32>
              linalg.fill ins(%cst_1 : f32) outs(%19 : memref<1x64x128xf32>)
              linalg.batch_matmul ins(%34, %21 : memref<1x64x128xf32>, memref<1x128x128xf32>) outs(%19 : memref<1x64x128xf32>)
              loom.semaphore_give %21 : memref<1x128x128xf32>
              linalg.fill ins(%cst_3 : f32) outs(%27 : memref<1x64xf32>)
              linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%19 : memref<1x64x128xf32>) outs(%27 : memref<1x64xf32>) {
              ^bb0(%in: f32, %out: f32):
                %46 = arith.maximumf %in, %out : f32
                linalg.yield %46 : f32
              }
              linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%25, %27 : memref<1x64xf32>, memref<1x64xf32>) outs(%27 : memref<1x64xf32>) {
              ^bb0(%in: f32, %in_7: f32, %out: f32):
                %46 = arith.truncf %cst_0 : f64 to f32
                %47 = arith.mulf %in_7, %46 : f32
                %48 = arith.cmpf ogt, %in, %47 : f32
                %49 = arith.select %48, %in, %47 : f32
                linalg.yield %49 : f32
              }
              linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%19, %27 : memref<1x64x128xf32>, memref<1x64xf32>) outs(%19 : memref<1x64x128xf32>) {
              ^bb0(%in: f32, %in_7: f32, %out: f32):
                %46 = arith.truncf %cst_0 : f64 to f32
                %47 = arith.mulf %in, %46 : f32
                %48 = arith.subf %47, %in_7 : f32
                %49 = math.powf %cst, %48 : f32
                linalg.yield %49 : f32
              }
              linalg.fill ins(%cst_1 : f32) outs(%29 : memref<1x64xf32>)
              linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%19 : memref<1x64x128xf32>) outs(%29 : memref<1x64xf32>) {
              ^bb0(%in: f32, %out: f32):
                %46 = arith.addf %in, %out : f32
                linalg.yield %46 : f32
              }
              linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%25, %27 : memref<1x64xf32>, memref<1x64xf32>) outs(%31 : memref<1x64xf32>) {
              ^bb0(%in: f32, %in_7: f32, %out: f32):
                %46 = arith.subf %in, %in_7 : f32
                %47 = math.powf %cst, %46 : f32
                linalg.yield %47 : f32
              }
              linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%23, %31, %29 : memref<1x64xf32>, memref<1x64xf32>, memref<1x64xf32>) outs(%23 : memref<1x64xf32>) {
              ^bb0(%in: f32, %in_7: f32, %in_8: f32, %out: f32):
                %46 = arith.mulf %in, %in_7 : f32
                %47 = arith.addf %46, %in_8 : f32
                linalg.yield %47 : f32
              }
              loom.semaphore_give %29 : memref<1x64xf32>
              %44 = arith.muli %arg8, %c16384 : index
              %45 = arith.addi %39, %44 : index
              %reinterpret_cast_6 = memref.reinterpret_cast %arg1 to offset: [%45], sizes: [1, 128, 128], strides: [524288, 128, 1] : memref<32x4096x128xf32> to memref<1x128x128xf32, strided<[524288, 128, 1], offset: ?>>
              loom.copy %reinterpret_cast_6, %17 src_mem_space @DRAM dst_mem_space @L1, interconnect : [@horizontal_links], broadcast : [1, 8] : memref<1x128x128xf32, strided<[524288, 128, 1], offset: ?>>, memref<1x128x128xf32>
              linalg.fill ins(%cst_1 : f32) outs(%38 : memref<1x64x128xf32>)
              linalg.batch_matmul ins(%19, %17 : memref<1x64x128xf32>, memref<1x128x128xf32>) outs(%38 : memref<1x64x128xf32>)
              loom.semaphore_give %17 : memref<1x128x128xf32>
              loom.semaphore_give %19 : memref<1x64x128xf32>
              linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%38, %36, %31 : memref<1x64x128xf32>, memref<1x64x128xf32>, memref<1x64xf32>) outs(%36 : memref<1x64x128xf32>) {
              ^bb0(%in: f32, %in_7: f32, %in_8: f32, %out: f32):
                %46 = arith.mulf %in_7, %in_8 : f32
                %47 = arith.addf %in, %46 : f32
                linalg.yield %47 : f32
              }
              loom.semaphore_give %31 : memref<1x64xf32>
              loom.semaphore_give %38 : memref<1x64x128xf32>
              linalg.copy ins(%27 : memref<1x64xf32>) outs(%25 : memref<1x64xf32>)
              loom.semaphore_give %27 : memref<1x64xf32>
            }
            loom.semaphore_give %25 : memref<1x64xf32>
            loom.semaphore_give %34 : memref<1x64x128xf32>
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%36, %23 : memref<1x64x128xf32>, memref<1x64xf32>) outs(%33 : memref<1x64x128xf32>) {
            ^bb0(%in: f32, %in_5: f32, %out: f32):
              %42 = arith.divf %in, %in_5 : f32
              linalg.yield %42 : f32
            }
            loom.semaphore_give %23 : memref<1x64xf32>
            loom.semaphore_give %36 : memref<1x64x128xf32>
            %reinterpret_cast_4 = memref.reinterpret_cast %arg3 to offset: [%41], sizes: [1, 64, 128], strides: [524288, 128, 1] : memref<32x4096x128xf32> to memref<1x64x128xf32, strided<[524288, 128, 1], offset: ?>>
            loom.copy %33, %reinterpret_cast_4 src_mem_space @L1 dst_mem_space @DRAM, interconnect : [], broadcast : [1, 1] : memref<1x64x128xf32>, memref<1x64x128xf32, strided<[524288, 128, 1], offset: ?>>
            loom.semaphore_give %33 : memref<1x64x128xf32>
          }
        }
        scf.reduce 
      } {loom.iter_types = [#loom.iter_type<spatial>, #loom.iter_type<spatial>], loom.mapped_to_dims = [@y, @x]}
      return
    }
  }
  module attributes {loom.pass_name = "Materialize"} {
    func.func @attention__d0i1_d1i1__f01__d_d_d__BB1__BM64__BN128(%arg0: memref<32x128x4096xf32>, %arg1: memref<32x4096x128xf32>, %arg2: memref<32x4096x128xf32>, %arg3: memref<32x4096x128xf32>) {
      %c16384 = arith.constant 16384 : index
      %c8192 = arith.constant 8192 : index
      %c524288 = arith.constant 524288 : index
      %c32 = arith.constant 32 : index
      %cst = arith.constant 2.000000e+00 : f32
      %cst_0 = arith.constant 0.12751743074602467 : f64
      %cst_1 = arith.constant 0.000000e+00 : f32
      %cst_2 = arith.constant 1.000000e+00 : f32
      %cst_3 = arith.constant 0xFF800000 : f32
      %c128 = arith.constant 128 : index
      %c0 = arith.constant 0 : index
      %c8 = arith.constant 8 : index
      %c1 = arith.constant 1 : index
      scf.parallel (%arg4, %arg5) = (%c0, %c0) to (%c8, %c8) step (%c1, %c1) {
        scf.for %arg6 = %c0 to %c32 step %c1 {
          %12 = arith.muli %arg4, %c8 overflow<nsw> : index
          %13 = arith.addi %12, %arg5 : index
          %14 = loom.alloc [1, 128, 128] on @L1 : memref<1x128x128xf32>
          %15 = loom.semaphore_take %14 : memref<1x128x128xf32> -> memref<1x128x128xf32>
          %16 = loom.alloc [1, 64, 128] on @L1 : memref<1x64x128xf32>
          %17 = loom.semaphore_take %16 : memref<1x64x128xf32> -> memref<1x64x128xf32>
          %18 = loom.alloc [1, 128, 128] on @L1 : memref<1x128x128xf32>
          %19 = loom.semaphore_take %18 : memref<1x128x128xf32> -> memref<1x128x128xf32>
          %20 = loom.alloc [1, 64] on @L1 : memref<1x64xf32>
          %21 = loom.semaphore_take %20 : memref<1x64xf32> -> memref<1x64xf32>
          %22 = loom.alloc [1, 64] on @L1 : memref<1x64xf32>
          %23 = loom.semaphore_take %22 : memref<1x64xf32> -> memref<1x64xf32>
          %24 = loom.alloc [1, 64] on @L1 : memref<1x64xf32>
          %25 = loom.semaphore_take %24 : memref<1x64xf32> -> memref<1x64xf32>
          %26 = loom.alloc [1, 64] on @L1 : memref<1x64xf32>
          %27 = loom.semaphore_take %26 : memref<1x64xf32> -> memref<1x64xf32>
          %28 = loom.alloc [1, 64] on @L1 : memref<1x64xf32>
          %29 = loom.semaphore_take %28 : memref<1x64xf32> -> memref<1x64xf32>
          %30 = loom.alloc [1, 64, 128] on @L1 : memref<1x64x128xf32>
          %31 = loom.semaphore_take %30 : memref<1x64x128xf32> -> memref<1x64x128xf32>
          %32 = loom.semaphore_take %30 : memref<1x64x128xf32> -> memref<1x64x128xf32>
          %33 = loom.alloc [1, 64, 128] on @L1 : memref<1x64x128xf32>
          %34 = loom.semaphore_take %33 : memref<1x64x128xf32> -> memref<1x64x128xf32>
          %35 = loom.alloc [1, 64, 128] on @L1 : memref<1x64x128xf32>
          %36 = loom.semaphore_take %35 : memref<1x64x128xf32> -> memref<1x64x128xf32>
          %37 = arith.muli %arg6, %c524288 overflow<nsw> : index
          %38 = arith.muli %13, %c8192 : index
          %39 = arith.addi %37, %38 : index
          %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%39], sizes: [1, 64, 128], strides: [524288, 128, 1] : memref<32x4096x128xf32> to memref<1x64x128xf32, strided<[524288, 128, 1], offset: ?>>
          loom.copy %reinterpret_cast, %32 src_mem_space @DRAM dst_mem_space @L1, interconnect : [], broadcast : [1, 1] : memref<1x64x128xf32, strided<[524288, 128, 1], offset: ?>>, memref<1x64x128xf32>
          linalg.fill ins(%cst_1 : f32) outs(%34 : memref<1x64x128xf32>)
          linalg.fill ins(%cst_2 : f32) outs(%21 : memref<1x64xf32>)
          linalg.fill ins(%cst_3 : f32) outs(%23 : memref<1x64xf32>)
          scf.for %arg7 = %c0 to %c32 step %c1 {
            %40 = arith.muli %arg7, %c128 : index
            %41 = arith.addi %37, %40 : index
            %reinterpret_cast_5 = memref.reinterpret_cast %arg0 to offset: [%41], sizes: [1, 128, 128], strides: [524288, 4096, 1] : memref<32x128x4096xf32> to memref<1x128x128xf32, strided<[524288, 4096, 1], offset: ?>>
            loom.copy %reinterpret_cast_5, %19 src_mem_space @DRAM dst_mem_space @L1, interconnect : [], broadcast : [1, 1] : memref<1x128x128xf32, strided<[524288, 4096, 1], offset: ?>>, memref<1x128x128xf32>
            linalg.fill ins(%cst_1 : f32) outs(%17 : memref<1x64x128xf32>)
            linalg.batch_matmul ins(%32, %19 : memref<1x64x128xf32>, memref<1x128x128xf32>) outs(%17 : memref<1x64x128xf32>)
            loom.semaphore_give %19 : memref<1x128x128xf32>
            linalg.fill ins(%cst_3 : f32) outs(%25 : memref<1x64xf32>)
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%17 : memref<1x64x128xf32>) outs(%25 : memref<1x64xf32>) {
            ^bb0(%in: f32, %out: f32):
              %44 = arith.maximumf %in, %out : f32
              linalg.yield %44 : f32
            }
            linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%23, %25 : memref<1x64xf32>, memref<1x64xf32>) outs(%25 : memref<1x64xf32>) {
            ^bb0(%in: f32, %in_7: f32, %out: f32):
              %44 = arith.truncf %cst_0 : f64 to f32
              %45 = arith.mulf %in_7, %44 : f32
              %46 = arith.cmpf ogt, %in, %45 : f32
              %47 = arith.select %46, %in, %45 : f32
              linalg.yield %47 : f32
            }
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%17, %25 : memref<1x64x128xf32>, memref<1x64xf32>) outs(%17 : memref<1x64x128xf32>) {
            ^bb0(%in: f32, %in_7: f32, %out: f32):
              %44 = arith.truncf %cst_0 : f64 to f32
              %45 = arith.mulf %in, %44 : f32
              %46 = arith.subf %45, %in_7 : f32
              %47 = math.powf %cst, %46 : f32
              linalg.yield %47 : f32
            }
            linalg.fill ins(%cst_1 : f32) outs(%27 : memref<1x64xf32>)
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%17 : memref<1x64x128xf32>) outs(%27 : memref<1x64xf32>) {
            ^bb0(%in: f32, %out: f32):
              %44 = arith.addf %in, %out : f32
              linalg.yield %44 : f32
            }
            linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%23, %25 : memref<1x64xf32>, memref<1x64xf32>) outs(%29 : memref<1x64xf32>) {
            ^bb0(%in: f32, %in_7: f32, %out: f32):
              %44 = arith.subf %in, %in_7 : f32
              %45 = math.powf %cst, %44 : f32
              linalg.yield %45 : f32
            }
            linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%21, %29, %27 : memref<1x64xf32>, memref<1x64xf32>, memref<1x64xf32>) outs(%21 : memref<1x64xf32>) {
            ^bb0(%in: f32, %in_7: f32, %in_8: f32, %out: f32):
              %44 = arith.mulf %in, %in_7 : f32
              %45 = arith.addf %44, %in_8 : f32
              linalg.yield %45 : f32
            }
            loom.semaphore_give %27 : memref<1x64xf32>
            %42 = arith.muli %arg7, %c16384 : index
            %43 = arith.addi %37, %42 : index
            %reinterpret_cast_6 = memref.reinterpret_cast %arg1 to offset: [%43], sizes: [1, 128, 128], strides: [524288, 128, 1] : memref<32x4096x128xf32> to memref<1x128x128xf32, strided<[524288, 128, 1], offset: ?>>
            loom.copy %reinterpret_cast_6, %15 src_mem_space @DRAM dst_mem_space @L1, interconnect : [], broadcast : [1, 1] : memref<1x128x128xf32, strided<[524288, 128, 1], offset: ?>>, memref<1x128x128xf32>
            linalg.fill ins(%cst_1 : f32) outs(%36 : memref<1x64x128xf32>)
            linalg.batch_matmul ins(%17, %15 : memref<1x64x128xf32>, memref<1x128x128xf32>) outs(%36 : memref<1x64x128xf32>)
            loom.semaphore_give %15 : memref<1x128x128xf32>
            loom.semaphore_give %17 : memref<1x64x128xf32>
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%36, %34, %29 : memref<1x64x128xf32>, memref<1x64x128xf32>, memref<1x64xf32>) outs(%34 : memref<1x64x128xf32>) {
            ^bb0(%in: f32, %in_7: f32, %in_8: f32, %out: f32):
              %44 = arith.mulf %in_7, %in_8 : f32
              %45 = arith.addf %in, %44 : f32
              linalg.yield %45 : f32
            }
            loom.semaphore_give %29 : memref<1x64xf32>
            loom.semaphore_give %36 : memref<1x64x128xf32>
            linalg.copy ins(%25 : memref<1x64xf32>) outs(%23 : memref<1x64xf32>)
            loom.semaphore_give %25 : memref<1x64xf32>
          }
          loom.semaphore_give %23 : memref<1x64xf32>
          loom.semaphore_give %32 : memref<1x64x128xf32>
          linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%34, %21 : memref<1x64x128xf32>, memref<1x64xf32>) outs(%31 : memref<1x64x128xf32>) {
          ^bb0(%in: f32, %in_5: f32, %out: f32):
            %40 = arith.divf %in, %in_5 : f32
            linalg.yield %40 : f32
          }
          loom.semaphore_give %21 : memref<1x64xf32>
          loom.semaphore_give %34 : memref<1x64x128xf32>
          %reinterpret_cast_4 = memref.reinterpret_cast %arg3 to offset: [%39], sizes: [1, 64, 128], strides: [524288, 128, 1] : memref<32x4096x128xf32> to memref<1x64x128xf32, strided<[524288, 128, 1], offset: ?>>
          loom.copy %31, %reinterpret_cast_4 src_mem_space @L1 dst_mem_space @DRAM, interconnect : [], broadcast : [1, 1] : memref<1x64x128xf32>, memref<1x64x128xf32, strided<[524288, 128, 1], offset: ?>>
          loom.semaphore_give %31 : memref<1x64x128xf32>
        }
        scf.reduce 
      } {loom.iter_types = [#loom.iter_type<spatial>, #loom.iter_type<spatial>], loom.mapped_to_dims = [@x, @y]}
      return
    }
  }
  module attributes {loom.pass_name = "Materialize"} {
    func.func @attention__d0i1_d1i1__f01__d_d_a__BB1__BM64__BN128(%arg0: memref<32x128x4096xf32>, %arg1: memref<32x4096x128xf32>, %arg2: memref<32x4096x128xf32>, %arg3: memref<32x4096x128xf32>) {
      %c16384 = arith.constant 16384 : index
      %c8192 = arith.constant 8192 : index
      %c524288 = arith.constant 524288 : index
      %c32 = arith.constant 32 : index
      %cst = arith.constant 2.000000e+00 : f32
      %cst_0 = arith.constant 0.12751743074602467 : f64
      %cst_1 = arith.constant 0.000000e+00 : f32
      %cst_2 = arith.constant 1.000000e+00 : f32
      %cst_3 = arith.constant 0xFF800000 : f32
      %c128 = arith.constant 128 : index
      %c0 = arith.constant 0 : index
      %c8 = arith.constant 8 : index
      %c1 = arith.constant 1 : index
      scf.parallel (%arg4, %arg5) = (%c0, %c0) to (%c8, %c8) step (%c1, %c1) {
        scf.for %arg6 = %c0 to %c32 step %c1 {
          %12 = arith.muli %arg4, %c8 overflow<nsw> : index
          %13 = arith.addi %12, %arg5 : index
          %14 = loom.alloc [1, 128, 128] on @L1 : memref<1x128x128xf32>
          %15 = loom.semaphore_take %14 : memref<1x128x128xf32> -> memref<1x128x128xf32>
          %16 = loom.alloc [1, 64, 128] on @L1 : memref<1x64x128xf32>
          %17 = loom.semaphore_take %16 : memref<1x64x128xf32> -> memref<1x64x128xf32>
          %18 = loom.alloc [1, 128, 128] on @L1 : memref<1x128x128xf32>
          %19 = loom.semaphore_take %18 : memref<1x128x128xf32> -> memref<1x128x128xf32>
          %20 = loom.alloc [1, 64] on @L1 : memref<1x64xf32>
          %21 = loom.semaphore_take %20 : memref<1x64xf32> -> memref<1x64xf32>
          %22 = loom.alloc [1, 64] on @L1 : memref<1x64xf32>
          %23 = loom.semaphore_take %22 : memref<1x64xf32> -> memref<1x64xf32>
          %24 = loom.alloc [1, 64] on @L1 : memref<1x64xf32>
          %25 = loom.semaphore_take %24 : memref<1x64xf32> -> memref<1x64xf32>
          %26 = loom.alloc [1, 64] on @L1 : memref<1x64xf32>
          %27 = loom.semaphore_take %26 : memref<1x64xf32> -> memref<1x64xf32>
          %28 = loom.alloc [1, 64] on @L1 : memref<1x64xf32>
          %29 = loom.semaphore_take %28 : memref<1x64xf32> -> memref<1x64xf32>
          %30 = loom.alloc [1, 64, 128] on @L1 : memref<1x64x128xf32>
          %31 = loom.semaphore_take %30 : memref<1x64x128xf32> -> memref<1x64x128xf32>
          %32 = loom.semaphore_take %30 : memref<1x64x128xf32> -> memref<1x64x128xf32>
          %33 = loom.alloc [1, 64, 128] on @L1 : memref<1x64x128xf32>
          %34 = loom.semaphore_take %33 : memref<1x64x128xf32> -> memref<1x64x128xf32>
          %35 = loom.alloc [1, 64, 128] on @L1 : memref<1x64x128xf32>
          %36 = loom.semaphore_take %35 : memref<1x64x128xf32> -> memref<1x64x128xf32>
          %37 = arith.muli %arg6, %c524288 overflow<nsw> : index
          %38 = arith.muli %13, %c8192 : index
          %39 = arith.addi %37, %38 : index
          %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%39], sizes: [1, 64, 128], strides: [524288, 128, 1] : memref<32x4096x128xf32> to memref<1x64x128xf32, strided<[524288, 128, 1], offset: ?>>
          loom.copy %reinterpret_cast, %32 src_mem_space @DRAM dst_mem_space @L1, interconnect : [], broadcast : [1, 1] : memref<1x64x128xf32, strided<[524288, 128, 1], offset: ?>>, memref<1x64x128xf32>
          linalg.fill ins(%cst_1 : f32) outs(%34 : memref<1x64x128xf32>)
          linalg.fill ins(%cst_2 : f32) outs(%21 : memref<1x64xf32>)
          linalg.fill ins(%cst_3 : f32) outs(%23 : memref<1x64xf32>)
          scf.for %arg7 = %c0 to %c32 step %c1 {
            %40 = arith.muli %arg7, %c128 : index
            %41 = arith.addi %37, %40 : index
            %reinterpret_cast_5 = memref.reinterpret_cast %arg0 to offset: [%41], sizes: [1, 128, 128], strides: [524288, 4096, 1] : memref<32x128x4096xf32> to memref<1x128x128xf32, strided<[524288, 4096, 1], offset: ?>>
            loom.copy %reinterpret_cast_5, %19 src_mem_space @DRAM dst_mem_space @L1, interconnect : [], broadcast : [1, 1] : memref<1x128x128xf32, strided<[524288, 4096, 1], offset: ?>>, memref<1x128x128xf32>
            linalg.fill ins(%cst_1 : f32) outs(%17 : memref<1x64x128xf32>)
            linalg.batch_matmul ins(%32, %19 : memref<1x64x128xf32>, memref<1x128x128xf32>) outs(%17 : memref<1x64x128xf32>)
            loom.semaphore_give %19 : memref<1x128x128xf32>
            linalg.fill ins(%cst_3 : f32) outs(%25 : memref<1x64xf32>)
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%17 : memref<1x64x128xf32>) outs(%25 : memref<1x64xf32>) {
            ^bb0(%in: f32, %out: f32):
              %44 = arith.maximumf %in, %out : f32
              linalg.yield %44 : f32
            }
            linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%23, %25 : memref<1x64xf32>, memref<1x64xf32>) outs(%25 : memref<1x64xf32>) {
            ^bb0(%in: f32, %in_7: f32, %out: f32):
              %44 = arith.truncf %cst_0 : f64 to f32
              %45 = arith.mulf %in_7, %44 : f32
              %46 = arith.cmpf ogt, %in, %45 : f32
              %47 = arith.select %46, %in, %45 : f32
              linalg.yield %47 : f32
            }
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%17, %25 : memref<1x64x128xf32>, memref<1x64xf32>) outs(%17 : memref<1x64x128xf32>) {
            ^bb0(%in: f32, %in_7: f32, %out: f32):
              %44 = arith.truncf %cst_0 : f64 to f32
              %45 = arith.mulf %in, %44 : f32
              %46 = arith.subf %45, %in_7 : f32
              %47 = math.powf %cst, %46 : f32
              linalg.yield %47 : f32
            }
            linalg.fill ins(%cst_1 : f32) outs(%27 : memref<1x64xf32>)
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%17 : memref<1x64x128xf32>) outs(%27 : memref<1x64xf32>) {
            ^bb0(%in: f32, %out: f32):
              %44 = arith.addf %in, %out : f32
              linalg.yield %44 : f32
            }
            linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%23, %25 : memref<1x64xf32>, memref<1x64xf32>) outs(%29 : memref<1x64xf32>) {
            ^bb0(%in: f32, %in_7: f32, %out: f32):
              %44 = arith.subf %in, %in_7 : f32
              %45 = math.powf %cst, %44 : f32
              linalg.yield %45 : f32
            }
            linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%21, %29, %27 : memref<1x64xf32>, memref<1x64xf32>, memref<1x64xf32>) outs(%21 : memref<1x64xf32>) {
            ^bb0(%in: f32, %in_7: f32, %in_8: f32, %out: f32):
              %44 = arith.mulf %in, %in_7 : f32
              %45 = arith.addf %44, %in_8 : f32
              linalg.yield %45 : f32
            }
            loom.semaphore_give %27 : memref<1x64xf32>
            %42 = arith.muli %arg7, %c16384 : index
            %43 = arith.addi %37, %42 : index
            %reinterpret_cast_6 = memref.reinterpret_cast %arg1 to offset: [%43], sizes: [1, 128, 128], strides: [524288, 128, 1] : memref<32x4096x128xf32> to memref<1x128x128xf32, strided<[524288, 128, 1], offset: ?>>
            loom.copy %reinterpret_cast_6, %15 src_mem_space @DRAM dst_mem_space @L1, interconnect : [@horizontal_links, @vertical_links], broadcast : [8, 8] : memref<1x128x128xf32, strided<[524288, 128, 1], offset: ?>>, memref<1x128x128xf32>
            linalg.fill ins(%cst_1 : f32) outs(%36 : memref<1x64x128xf32>)
            linalg.batch_matmul ins(%17, %15 : memref<1x64x128xf32>, memref<1x128x128xf32>) outs(%36 : memref<1x64x128xf32>)
            loom.semaphore_give %15 : memref<1x128x128xf32>
            loom.semaphore_give %17 : memref<1x64x128xf32>
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%36, %34, %29 : memref<1x64x128xf32>, memref<1x64x128xf32>, memref<1x64xf32>) outs(%34 : memref<1x64x128xf32>) {
            ^bb0(%in: f32, %in_7: f32, %in_8: f32, %out: f32):
              %44 = arith.mulf %in_7, %in_8 : f32
              %45 = arith.addf %in, %44 : f32
              linalg.yield %45 : f32
            }
            loom.semaphore_give %29 : memref<1x64xf32>
            loom.semaphore_give %36 : memref<1x64x128xf32>
            linalg.copy ins(%25 : memref<1x64xf32>) outs(%23 : memref<1x64xf32>)
            loom.semaphore_give %25 : memref<1x64xf32>
          }
          loom.semaphore_give %23 : memref<1x64xf32>
          loom.semaphore_give %32 : memref<1x64x128xf32>
          linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%34, %21 : memref<1x64x128xf32>, memref<1x64xf32>) outs(%31 : memref<1x64x128xf32>) {
          ^bb0(%in: f32, %in_5: f32, %out: f32):
            %40 = arith.divf %in, %in_5 : f32
            linalg.yield %40 : f32
          }
          loom.semaphore_give %21 : memref<1x64xf32>
          loom.semaphore_give %34 : memref<1x64x128xf32>
          %reinterpret_cast_4 = memref.reinterpret_cast %arg3 to offset: [%39], sizes: [1, 64, 128], strides: [524288, 128, 1] : memref<32x4096x128xf32> to memref<1x64x128xf32, strided<[524288, 128, 1], offset: ?>>
          loom.copy %31, %reinterpret_cast_4 src_mem_space @L1 dst_mem_space @DRAM, interconnect : [], broadcast : [1, 1] : memref<1x64x128xf32>, memref<1x64x128xf32, strided<[524288, 128, 1], offset: ?>>
          loom.semaphore_give %31 : memref<1x64x128xf32>
        }
        scf.reduce 
      } {loom.iter_types = [#loom.iter_type<spatial>, #loom.iter_type<spatial>], loom.mapped_to_dims = [@x, @y]}
      return
    }
  }
  module attributes {loom.pass_name = "Materialize"} {
    func.func @attention__d0i1_d1i1__f01__d_d_h__BB1__BM64__BN128(%arg0: memref<32x128x4096xf32>, %arg1: memref<32x4096x128xf32>, %arg2: memref<32x4096x128xf32>, %arg3: memref<32x4096x128xf32>) {
      %c16384 = arith.constant 16384 : index
      %c8192 = arith.constant 8192 : index
      %c524288 = arith.constant 524288 : index
      %c32 = arith.constant 32 : index
      %cst = arith.constant 2.000000e+00 : f32
      %cst_0 = arith.constant 0.12751743074602467 : f64
      %cst_1 = arith.constant 0.000000e+00 : f32
      %cst_2 = arith.constant 1.000000e+00 : f32
      %cst_3 = arith.constant 0xFF800000 : f32
      %c128 = arith.constant 128 : index
      %c0 = arith.constant 0 : index
      %c8 = arith.constant 8 : index
      %c1 = arith.constant 1 : index
      scf.parallel (%arg4, %arg5) = (%c0, %c0) to (%c8, %c8) step (%c1, %c1) {
        scf.for %arg6 = %c0 to %c32 step %c1 {
          %12 = arith.muli %arg4, %c8 overflow<nsw> : index
          %13 = arith.addi %12, %arg5 : index
          %14 = loom.alloc [1, 128, 128] on @L1 : memref<1x128x128xf32>
          %15 = loom.semaphore_take %14 : memref<1x128x128xf32> -> memref<1x128x128xf32>
          %16 = loom.alloc [1, 64, 128] on @L1 : memref<1x64x128xf32>
          %17 = loom.semaphore_take %16 : memref<1x64x128xf32> -> memref<1x64x128xf32>
          %18 = loom.alloc [1, 128, 128] on @L1 : memref<1x128x128xf32>
          %19 = loom.semaphore_take %18 : memref<1x128x128xf32> -> memref<1x128x128xf32>
          %20 = loom.alloc [1, 64] on @L1 : memref<1x64xf32>
          %21 = loom.semaphore_take %20 : memref<1x64xf32> -> memref<1x64xf32>
          %22 = loom.alloc [1, 64] on @L1 : memref<1x64xf32>
          %23 = loom.semaphore_take %22 : memref<1x64xf32> -> memref<1x64xf32>
          %24 = loom.alloc [1, 64] on @L1 : memref<1x64xf32>
          %25 = loom.semaphore_take %24 : memref<1x64xf32> -> memref<1x64xf32>
          %26 = loom.alloc [1, 64] on @L1 : memref<1x64xf32>
          %27 = loom.semaphore_take %26 : memref<1x64xf32> -> memref<1x64xf32>
          %28 = loom.alloc [1, 64] on @L1 : memref<1x64xf32>
          %29 = loom.semaphore_take %28 : memref<1x64xf32> -> memref<1x64xf32>
          %30 = loom.alloc [1, 64, 128] on @L1 : memref<1x64x128xf32>
          %31 = loom.semaphore_take %30 : memref<1x64x128xf32> -> memref<1x64x128xf32>
          %32 = loom.semaphore_take %30 : memref<1x64x128xf32> -> memref<1x64x128xf32>
          %33 = loom.alloc [1, 64, 128] on @L1 : memref<1x64x128xf32>
          %34 = loom.semaphore_take %33 : memref<1x64x128xf32> -> memref<1x64x128xf32>
          %35 = loom.alloc [1, 64, 128] on @L1 : memref<1x64x128xf32>
          %36 = loom.semaphore_take %35 : memref<1x64x128xf32> -> memref<1x64x128xf32>
          %37 = arith.muli %arg6, %c524288 overflow<nsw> : index
          %38 = arith.muli %13, %c8192 : index
          %39 = arith.addi %37, %38 : index
          %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%39], sizes: [1, 64, 128], strides: [524288, 128, 1] : memref<32x4096x128xf32> to memref<1x64x128xf32, strided<[524288, 128, 1], offset: ?>>
          loom.copy %reinterpret_cast, %32 src_mem_space @DRAM dst_mem_space @L1, interconnect : [], broadcast : [1, 1] : memref<1x64x128xf32, strided<[524288, 128, 1], offset: ?>>, memref<1x64x128xf32>
          linalg.fill ins(%cst_1 : f32) outs(%34 : memref<1x64x128xf32>)
          linalg.fill ins(%cst_2 : f32) outs(%21 : memref<1x64xf32>)
          linalg.fill ins(%cst_3 : f32) outs(%23 : memref<1x64xf32>)
          scf.for %arg7 = %c0 to %c32 step %c1 {
            %40 = arith.muli %arg7, %c128 : index
            %41 = arith.addi %37, %40 : index
            %reinterpret_cast_5 = memref.reinterpret_cast %arg0 to offset: [%41], sizes: [1, 128, 128], strides: [524288, 4096, 1] : memref<32x128x4096xf32> to memref<1x128x128xf32, strided<[524288, 4096, 1], offset: ?>>
            loom.copy %reinterpret_cast_5, %19 src_mem_space @DRAM dst_mem_space @L1, interconnect : [], broadcast : [1, 1] : memref<1x128x128xf32, strided<[524288, 4096, 1], offset: ?>>, memref<1x128x128xf32>
            linalg.fill ins(%cst_1 : f32) outs(%17 : memref<1x64x128xf32>)
            linalg.batch_matmul ins(%32, %19 : memref<1x64x128xf32>, memref<1x128x128xf32>) outs(%17 : memref<1x64x128xf32>)
            loom.semaphore_give %19 : memref<1x128x128xf32>
            linalg.fill ins(%cst_3 : f32) outs(%25 : memref<1x64xf32>)
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%17 : memref<1x64x128xf32>) outs(%25 : memref<1x64xf32>) {
            ^bb0(%in: f32, %out: f32):
              %44 = arith.maximumf %in, %out : f32
              linalg.yield %44 : f32
            }
            linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%23, %25 : memref<1x64xf32>, memref<1x64xf32>) outs(%25 : memref<1x64xf32>) {
            ^bb0(%in: f32, %in_7: f32, %out: f32):
              %44 = arith.truncf %cst_0 : f64 to f32
              %45 = arith.mulf %in_7, %44 : f32
              %46 = arith.cmpf ogt, %in, %45 : f32
              %47 = arith.select %46, %in, %45 : f32
              linalg.yield %47 : f32
            }
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%17, %25 : memref<1x64x128xf32>, memref<1x64xf32>) outs(%17 : memref<1x64x128xf32>) {
            ^bb0(%in: f32, %in_7: f32, %out: f32):
              %44 = arith.truncf %cst_0 : f64 to f32
              %45 = arith.mulf %in, %44 : f32
              %46 = arith.subf %45, %in_7 : f32
              %47 = math.powf %cst, %46 : f32
              linalg.yield %47 : f32
            }
            linalg.fill ins(%cst_1 : f32) outs(%27 : memref<1x64xf32>)
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%17 : memref<1x64x128xf32>) outs(%27 : memref<1x64xf32>) {
            ^bb0(%in: f32, %out: f32):
              %44 = arith.addf %in, %out : f32
              linalg.yield %44 : f32
            }
            linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%23, %25 : memref<1x64xf32>, memref<1x64xf32>) outs(%29 : memref<1x64xf32>) {
            ^bb0(%in: f32, %in_7: f32, %out: f32):
              %44 = arith.subf %in, %in_7 : f32
              %45 = math.powf %cst, %44 : f32
              linalg.yield %45 : f32
            }
            linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%21, %29, %27 : memref<1x64xf32>, memref<1x64xf32>, memref<1x64xf32>) outs(%21 : memref<1x64xf32>) {
            ^bb0(%in: f32, %in_7: f32, %in_8: f32, %out: f32):
              %44 = arith.mulf %in, %in_7 : f32
              %45 = arith.addf %44, %in_8 : f32
              linalg.yield %45 : f32
            }
            loom.semaphore_give %27 : memref<1x64xf32>
            %42 = arith.muli %arg7, %c16384 : index
            %43 = arith.addi %37, %42 : index
            %reinterpret_cast_6 = memref.reinterpret_cast %arg1 to offset: [%43], sizes: [1, 128, 128], strides: [524288, 128, 1] : memref<32x4096x128xf32> to memref<1x128x128xf32, strided<[524288, 128, 1], offset: ?>>
            loom.copy %reinterpret_cast_6, %15 src_mem_space @DRAM dst_mem_space @L1, interconnect : [@horizontal_links], broadcast : [1, 8] : memref<1x128x128xf32, strided<[524288, 128, 1], offset: ?>>, memref<1x128x128xf32>
            linalg.fill ins(%cst_1 : f32) outs(%36 : memref<1x64x128xf32>)
            linalg.batch_matmul ins(%17, %15 : memref<1x64x128xf32>, memref<1x128x128xf32>) outs(%36 : memref<1x64x128xf32>)
            loom.semaphore_give %15 : memref<1x128x128xf32>
            loom.semaphore_give %17 : memref<1x64x128xf32>
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%36, %34, %29 : memref<1x64x128xf32>, memref<1x64x128xf32>, memref<1x64xf32>) outs(%34 : memref<1x64x128xf32>) {
            ^bb0(%in: f32, %in_7: f32, %in_8: f32, %out: f32):
              %44 = arith.mulf %in_7, %in_8 : f32
              %45 = arith.addf %in, %44 : f32
              linalg.yield %45 : f32
            }
            loom.semaphore_give %29 : memref<1x64xf32>
            loom.semaphore_give %36 : memref<1x64x128xf32>
            linalg.copy ins(%25 : memref<1x64xf32>) outs(%23 : memref<1x64xf32>)
            loom.semaphore_give %25 : memref<1x64xf32>
          }
          loom.semaphore_give %23 : memref<1x64xf32>
          loom.semaphore_give %32 : memref<1x64x128xf32>
          linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%34, %21 : memref<1x64x128xf32>, memref<1x64xf32>) outs(%31 : memref<1x64x128xf32>) {
          ^bb0(%in: f32, %in_5: f32, %out: f32):
            %40 = arith.divf %in, %in_5 : f32
            linalg.yield %40 : f32
          }
          loom.semaphore_give %21 : memref<1x64xf32>
          loom.semaphore_give %34 : memref<1x64x128xf32>
          %reinterpret_cast_4 = memref.reinterpret_cast %arg3 to offset: [%39], sizes: [1, 64, 128], strides: [524288, 128, 1] : memref<32x4096x128xf32> to memref<1x64x128xf32, strided<[524288, 128, 1], offset: ?>>
          loom.copy %31, %reinterpret_cast_4 src_mem_space @L1 dst_mem_space @DRAM, interconnect : [], broadcast : [1, 1] : memref<1x64x128xf32>, memref<1x64x128xf32, strided<[524288, 128, 1], offset: ?>>
          loom.semaphore_give %31 : memref<1x64x128xf32>
        }
        scf.reduce 
      } {loom.iter_types = [#loom.iter_type<spatial>, #loom.iter_type<spatial>], loom.mapped_to_dims = [@x, @y]}
      return
    }
  }
  module attributes {loom.pass_name = "Materialize"} {
    func.func @attention__d0i1_d1i1__f01__d_d_v__BB1__BM64__BN128(%arg0: memref<32x128x4096xf32>, %arg1: memref<32x4096x128xf32>, %arg2: memref<32x4096x128xf32>, %arg3: memref<32x4096x128xf32>) {
      %c16384 = arith.constant 16384 : index
      %c8192 = arith.constant 8192 : index
      %c524288 = arith.constant 524288 : index
      %c32 = arith.constant 32 : index
      %cst = arith.constant 2.000000e+00 : f32
      %cst_0 = arith.constant 0.12751743074602467 : f64
      %cst_1 = arith.constant 0.000000e+00 : f32
      %cst_2 = arith.constant 1.000000e+00 : f32
      %cst_3 = arith.constant 0xFF800000 : f32
      %c128 = arith.constant 128 : index
      %c0 = arith.constant 0 : index
      %c8 = arith.constant 8 : index
      %c1 = arith.constant 1 : index
      scf.parallel (%arg4, %arg5) = (%c0, %c0) to (%c8, %c8) step (%c1, %c1) {
        scf.for %arg6 = %c0 to %c32 step %c1 {
          %12 = arith.muli %arg4, %c8 overflow<nsw> : index
          %13 = arith.addi %12, %arg5 : index
          %14 = loom.alloc [1, 128, 128] on @L1 : memref<1x128x128xf32>
          %15 = loom.semaphore_take %14 : memref<1x128x128xf32> -> memref<1x128x128xf32>
          %16 = loom.alloc [1, 64, 128] on @L1 : memref<1x64x128xf32>
          %17 = loom.semaphore_take %16 : memref<1x64x128xf32> -> memref<1x64x128xf32>
          %18 = loom.alloc [1, 128, 128] on @L1 : memref<1x128x128xf32>
          %19 = loom.semaphore_take %18 : memref<1x128x128xf32> -> memref<1x128x128xf32>
          %20 = loom.alloc [1, 64] on @L1 : memref<1x64xf32>
          %21 = loom.semaphore_take %20 : memref<1x64xf32> -> memref<1x64xf32>
          %22 = loom.alloc [1, 64] on @L1 : memref<1x64xf32>
          %23 = loom.semaphore_take %22 : memref<1x64xf32> -> memref<1x64xf32>
          %24 = loom.alloc [1, 64] on @L1 : memref<1x64xf32>
          %25 = loom.semaphore_take %24 : memref<1x64xf32> -> memref<1x64xf32>
          %26 = loom.alloc [1, 64] on @L1 : memref<1x64xf32>
          %27 = loom.semaphore_take %26 : memref<1x64xf32> -> memref<1x64xf32>
          %28 = loom.alloc [1, 64] on @L1 : memref<1x64xf32>
          %29 = loom.semaphore_take %28 : memref<1x64xf32> -> memref<1x64xf32>
          %30 = loom.alloc [1, 64, 128] on @L1 : memref<1x64x128xf32>
          %31 = loom.semaphore_take %30 : memref<1x64x128xf32> -> memref<1x64x128xf32>
          %32 = loom.semaphore_take %30 : memref<1x64x128xf32> -> memref<1x64x128xf32>
          %33 = loom.alloc [1, 64, 128] on @L1 : memref<1x64x128xf32>
          %34 = loom.semaphore_take %33 : memref<1x64x128xf32> -> memref<1x64x128xf32>
          %35 = loom.alloc [1, 64, 128] on @L1 : memref<1x64x128xf32>
          %36 = loom.semaphore_take %35 : memref<1x64x128xf32> -> memref<1x64x128xf32>
          %37 = arith.muli %arg6, %c524288 overflow<nsw> : index
          %38 = arith.muli %13, %c8192 : index
          %39 = arith.addi %37, %38 : index
          %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%39], sizes: [1, 64, 128], strides: [524288, 128, 1] : memref<32x4096x128xf32> to memref<1x64x128xf32, strided<[524288, 128, 1], offset: ?>>
          loom.copy %reinterpret_cast, %32 src_mem_space @DRAM dst_mem_space @L1, interconnect : [], broadcast : [1, 1] : memref<1x64x128xf32, strided<[524288, 128, 1], offset: ?>>, memref<1x64x128xf32>
          linalg.fill ins(%cst_1 : f32) outs(%34 : memref<1x64x128xf32>)
          linalg.fill ins(%cst_2 : f32) outs(%21 : memref<1x64xf32>)
          linalg.fill ins(%cst_3 : f32) outs(%23 : memref<1x64xf32>)
          scf.for %arg7 = %c0 to %c32 step %c1 {
            %40 = arith.muli %arg7, %c128 : index
            %41 = arith.addi %37, %40 : index
            %reinterpret_cast_5 = memref.reinterpret_cast %arg0 to offset: [%41], sizes: [1, 128, 128], strides: [524288, 4096, 1] : memref<32x128x4096xf32> to memref<1x128x128xf32, strided<[524288, 4096, 1], offset: ?>>
            loom.copy %reinterpret_cast_5, %19 src_mem_space @DRAM dst_mem_space @L1, interconnect : [], broadcast : [1, 1] : memref<1x128x128xf32, strided<[524288, 4096, 1], offset: ?>>, memref<1x128x128xf32>
            linalg.fill ins(%cst_1 : f32) outs(%17 : memref<1x64x128xf32>)
            linalg.batch_matmul ins(%32, %19 : memref<1x64x128xf32>, memref<1x128x128xf32>) outs(%17 : memref<1x64x128xf32>)
            loom.semaphore_give %19 : memref<1x128x128xf32>
            linalg.fill ins(%cst_3 : f32) outs(%25 : memref<1x64xf32>)
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%17 : memref<1x64x128xf32>) outs(%25 : memref<1x64xf32>) {
            ^bb0(%in: f32, %out: f32):
              %44 = arith.maximumf %in, %out : f32
              linalg.yield %44 : f32
            }
            linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%23, %25 : memref<1x64xf32>, memref<1x64xf32>) outs(%25 : memref<1x64xf32>) {
            ^bb0(%in: f32, %in_7: f32, %out: f32):
              %44 = arith.truncf %cst_0 : f64 to f32
              %45 = arith.mulf %in_7, %44 : f32
              %46 = arith.cmpf ogt, %in, %45 : f32
              %47 = arith.select %46, %in, %45 : f32
              linalg.yield %47 : f32
            }
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%17, %25 : memref<1x64x128xf32>, memref<1x64xf32>) outs(%17 : memref<1x64x128xf32>) {
            ^bb0(%in: f32, %in_7: f32, %out: f32):
              %44 = arith.truncf %cst_0 : f64 to f32
              %45 = arith.mulf %in, %44 : f32
              %46 = arith.subf %45, %in_7 : f32
              %47 = math.powf %cst, %46 : f32
              linalg.yield %47 : f32
            }
            linalg.fill ins(%cst_1 : f32) outs(%27 : memref<1x64xf32>)
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%17 : memref<1x64x128xf32>) outs(%27 : memref<1x64xf32>) {
            ^bb0(%in: f32, %out: f32):
              %44 = arith.addf %in, %out : f32
              linalg.yield %44 : f32
            }
            linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%23, %25 : memref<1x64xf32>, memref<1x64xf32>) outs(%29 : memref<1x64xf32>) {
            ^bb0(%in: f32, %in_7: f32, %out: f32):
              %44 = arith.subf %in, %in_7 : f32
              %45 = math.powf %cst, %44 : f32
              linalg.yield %45 : f32
            }
            linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%21, %29, %27 : memref<1x64xf32>, memref<1x64xf32>, memref<1x64xf32>) outs(%21 : memref<1x64xf32>) {
            ^bb0(%in: f32, %in_7: f32, %in_8: f32, %out: f32):
              %44 = arith.mulf %in, %in_7 : f32
              %45 = arith.addf %44, %in_8 : f32
              linalg.yield %45 : f32
            }
            loom.semaphore_give %27 : memref<1x64xf32>
            %42 = arith.muli %arg7, %c16384 : index
            %43 = arith.addi %37, %42 : index
            %reinterpret_cast_6 = memref.reinterpret_cast %arg1 to offset: [%43], sizes: [1, 128, 128], strides: [524288, 128, 1] : memref<32x4096x128xf32> to memref<1x128x128xf32, strided<[524288, 128, 1], offset: ?>>
            loom.copy %reinterpret_cast_6, %15 src_mem_space @DRAM dst_mem_space @L1, interconnect : [@vertical_links], broadcast : [8, 1] : memref<1x128x128xf32, strided<[524288, 128, 1], offset: ?>>, memref<1x128x128xf32>
            linalg.fill ins(%cst_1 : f32) outs(%36 : memref<1x64x128xf32>)
            linalg.batch_matmul ins(%17, %15 : memref<1x64x128xf32>, memref<1x128x128xf32>) outs(%36 : memref<1x64x128xf32>)
            loom.semaphore_give %15 : memref<1x128x128xf32>
            loom.semaphore_give %17 : memref<1x64x128xf32>
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%36, %34, %29 : memref<1x64x128xf32>, memref<1x64x128xf32>, memref<1x64xf32>) outs(%34 : memref<1x64x128xf32>) {
            ^bb0(%in: f32, %in_7: f32, %in_8: f32, %out: f32):
              %44 = arith.mulf %in_7, %in_8 : f32
              %45 = arith.addf %in, %44 : f32
              linalg.yield %45 : f32
            }
            loom.semaphore_give %29 : memref<1x64xf32>
            loom.semaphore_give %36 : memref<1x64x128xf32>
            linalg.copy ins(%25 : memref<1x64xf32>) outs(%23 : memref<1x64xf32>)
            loom.semaphore_give %25 : memref<1x64xf32>
          }
          loom.semaphore_give %23 : memref<1x64xf32>
          loom.semaphore_give %32 : memref<1x64x128xf32>
          linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%34, %21 : memref<1x64x128xf32>, memref<1x64xf32>) outs(%31 : memref<1x64x128xf32>) {
          ^bb0(%in: f32, %in_5: f32, %out: f32):
            %40 = arith.divf %in, %in_5 : f32
            linalg.yield %40 : f32
          }
          loom.semaphore_give %21 : memref<1x64xf32>
          loom.semaphore_give %34 : memref<1x64x128xf32>
          %reinterpret_cast_4 = memref.reinterpret_cast %arg3 to offset: [%39], sizes: [1, 64, 128], strides: [524288, 128, 1] : memref<32x4096x128xf32> to memref<1x64x128xf32, strided<[524288, 128, 1], offset: ?>>
          loom.copy %31, %reinterpret_cast_4 src_mem_space @L1 dst_mem_space @DRAM, interconnect : [], broadcast : [1, 1] : memref<1x64x128xf32>, memref<1x64x128xf32, strided<[524288, 128, 1], offset: ?>>
          loom.semaphore_give %31 : memref<1x64x128xf32>
        }
        scf.reduce 
      } {loom.iter_types = [#loom.iter_type<spatial>, #loom.iter_type<spatial>], loom.mapped_to_dims = [@x, @y]}
      return
    }
  }
  module attributes {loom.pass_name = "Materialize"} {
    func.func @attention__d0i1_d1i1__f01__d_a_d__BB1__BM64__BN128(%arg0: memref<32x128x4096xf32>, %arg1: memref<32x4096x128xf32>, %arg2: memref<32x4096x128xf32>, %arg3: memref<32x4096x128xf32>) {
      %c16384 = arith.constant 16384 : index
      %c8192 = arith.constant 8192 : index
      %c524288 = arith.constant 524288 : index
      %c32 = arith.constant 32 : index
      %cst = arith.constant 2.000000e+00 : f32
      %cst_0 = arith.constant 0.12751743074602467 : f64
      %cst_1 = arith.constant 0.000000e+00 : f32
      %cst_2 = arith.constant 1.000000e+00 : f32
      %cst_3 = arith.constant 0xFF800000 : f32
      %c128 = arith.constant 128 : index
      %c0 = arith.constant 0 : index
      %c8 = arith.constant 8 : index
      %c1 = arith.constant 1 : index
      scf.parallel (%arg4, %arg5) = (%c0, %c0) to (%c8, %c8) step (%c1, %c1) {
        scf.for %arg6 = %c0 to %c32 step %c1 {
          %12 = arith.muli %arg4, %c8 overflow<nsw> : index
          %13 = arith.addi %12, %arg5 : index
          %14 = loom.alloc [1, 128, 128] on @L1 : memref<1x128x128xf32>
          %15 = loom.semaphore_take %14 : memref<1x128x128xf32> -> memref<1x128x128xf32>
          %16 = loom.alloc [1, 64, 128] on @L1 : memref<1x64x128xf32>
          %17 = loom.semaphore_take %16 : memref<1x64x128xf32> -> memref<1x64x128xf32>
          %18 = loom.alloc [1, 128, 128] on @L1 : memref<1x128x128xf32>
          %19 = loom.semaphore_take %18 : memref<1x128x128xf32> -> memref<1x128x128xf32>
          %20 = loom.alloc [1, 64] on @L1 : memref<1x64xf32>
          %21 = loom.semaphore_take %20 : memref<1x64xf32> -> memref<1x64xf32>
          %22 = loom.alloc [1, 64] on @L1 : memref<1x64xf32>
          %23 = loom.semaphore_take %22 : memref<1x64xf32> -> memref<1x64xf32>
          %24 = loom.alloc [1, 64] on @L1 : memref<1x64xf32>
          %25 = loom.semaphore_take %24 : memref<1x64xf32> -> memref<1x64xf32>
          %26 = loom.alloc [1, 64] on @L1 : memref<1x64xf32>
          %27 = loom.semaphore_take %26 : memref<1x64xf32> -> memref<1x64xf32>
          %28 = loom.alloc [1, 64] on @L1 : memref<1x64xf32>
          %29 = loom.semaphore_take %28 : memref<1x64xf32> -> memref<1x64xf32>
          %30 = loom.alloc [1, 64, 128] on @L1 : memref<1x64x128xf32>
          %31 = loom.semaphore_take %30 : memref<1x64x128xf32> -> memref<1x64x128xf32>
          %32 = loom.semaphore_take %30 : memref<1x64x128xf32> -> memref<1x64x128xf32>
          %33 = loom.alloc [1, 64, 128] on @L1 : memref<1x64x128xf32>
          %34 = loom.semaphore_take %33 : memref<1x64x128xf32> -> memref<1x64x128xf32>
          %35 = loom.alloc [1, 64, 128] on @L1 : memref<1x64x128xf32>
          %36 = loom.semaphore_take %35 : memref<1x64x128xf32> -> memref<1x64x128xf32>
          %37 = arith.muli %arg6, %c524288 overflow<nsw> : index
          %38 = arith.muli %13, %c8192 : index
          %39 = arith.addi %37, %38 : index
          %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%39], sizes: [1, 64, 128], strides: [524288, 128, 1] : memref<32x4096x128xf32> to memref<1x64x128xf32, strided<[524288, 128, 1], offset: ?>>
          loom.copy %reinterpret_cast, %32 src_mem_space @DRAM dst_mem_space @L1, interconnect : [], broadcast : [1, 1] : memref<1x64x128xf32, strided<[524288, 128, 1], offset: ?>>, memref<1x64x128xf32>
          linalg.fill ins(%cst_1 : f32) outs(%34 : memref<1x64x128xf32>)
          linalg.fill ins(%cst_2 : f32) outs(%21 : memref<1x64xf32>)
          linalg.fill ins(%cst_3 : f32) outs(%23 : memref<1x64xf32>)
          scf.for %arg7 = %c0 to %c32 step %c1 {
            %40 = arith.muli %arg7, %c128 : index
            %41 = arith.addi %37, %40 : index
            %reinterpret_cast_5 = memref.reinterpret_cast %arg0 to offset: [%41], sizes: [1, 128, 128], strides: [524288, 4096, 1] : memref<32x128x4096xf32> to memref<1x128x128xf32, strided<[524288, 4096, 1], offset: ?>>
            loom.copy %reinterpret_cast_5, %19 src_mem_space @DRAM dst_mem_space @L1, interconnect : [@horizontal_links, @vertical_links], broadcast : [8, 8] : memref<1x128x128xf32, strided<[524288, 4096, 1], offset: ?>>, memref<1x128x128xf32>
            linalg.fill ins(%cst_1 : f32) outs(%17 : memref<1x64x128xf32>)
            linalg.batch_matmul ins(%32, %19 : memref<1x64x128xf32>, memref<1x128x128xf32>) outs(%17 : memref<1x64x128xf32>)
            loom.semaphore_give %19 : memref<1x128x128xf32>
            linalg.fill ins(%cst_3 : f32) outs(%25 : memref<1x64xf32>)
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%17 : memref<1x64x128xf32>) outs(%25 : memref<1x64xf32>) {
            ^bb0(%in: f32, %out: f32):
              %44 = arith.maximumf %in, %out : f32
              linalg.yield %44 : f32
            }
            linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%23, %25 : memref<1x64xf32>, memref<1x64xf32>) outs(%25 : memref<1x64xf32>) {
            ^bb0(%in: f32, %in_7: f32, %out: f32):
              %44 = arith.truncf %cst_0 : f64 to f32
              %45 = arith.mulf %in_7, %44 : f32
              %46 = arith.cmpf ogt, %in, %45 : f32
              %47 = arith.select %46, %in, %45 : f32
              linalg.yield %47 : f32
            }
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%17, %25 : memref<1x64x128xf32>, memref<1x64xf32>) outs(%17 : memref<1x64x128xf32>) {
            ^bb0(%in: f32, %in_7: f32, %out: f32):
              %44 = arith.truncf %cst_0 : f64 to f32
              %45 = arith.mulf %in, %44 : f32
              %46 = arith.subf %45, %in_7 : f32
              %47 = math.powf %cst, %46 : f32
              linalg.yield %47 : f32
            }
            linalg.fill ins(%cst_1 : f32) outs(%27 : memref<1x64xf32>)
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%17 : memref<1x64x128xf32>) outs(%27 : memref<1x64xf32>) {
            ^bb0(%in: f32, %out: f32):
              %44 = arith.addf %in, %out : f32
              linalg.yield %44 : f32
            }
            linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%23, %25 : memref<1x64xf32>, memref<1x64xf32>) outs(%29 : memref<1x64xf32>) {
            ^bb0(%in: f32, %in_7: f32, %out: f32):
              %44 = arith.subf %in, %in_7 : f32
              %45 = math.powf %cst, %44 : f32
              linalg.yield %45 : f32
            }
            linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%21, %29, %27 : memref<1x64xf32>, memref<1x64xf32>, memref<1x64xf32>) outs(%21 : memref<1x64xf32>) {
            ^bb0(%in: f32, %in_7: f32, %in_8: f32, %out: f32):
              %44 = arith.mulf %in, %in_7 : f32
              %45 = arith.addf %44, %in_8 : f32
              linalg.yield %45 : f32
            }
            loom.semaphore_give %27 : memref<1x64xf32>
            %42 = arith.muli %arg7, %c16384 : index
            %43 = arith.addi %37, %42 : index
            %reinterpret_cast_6 = memref.reinterpret_cast %arg1 to offset: [%43], sizes: [1, 128, 128], strides: [524288, 128, 1] : memref<32x4096x128xf32> to memref<1x128x128xf32, strided<[524288, 128, 1], offset: ?>>
            loom.copy %reinterpret_cast_6, %15 src_mem_space @DRAM dst_mem_space @L1, interconnect : [], broadcast : [1, 1] : memref<1x128x128xf32, strided<[524288, 128, 1], offset: ?>>, memref<1x128x128xf32>
            linalg.fill ins(%cst_1 : f32) outs(%36 : memref<1x64x128xf32>)
            linalg.batch_matmul ins(%17, %15 : memref<1x64x128xf32>, memref<1x128x128xf32>) outs(%36 : memref<1x64x128xf32>)
            loom.semaphore_give %15 : memref<1x128x128xf32>
            loom.semaphore_give %17 : memref<1x64x128xf32>
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%36, %34, %29 : memref<1x64x128xf32>, memref<1x64x128xf32>, memref<1x64xf32>) outs(%34 : memref<1x64x128xf32>) {
            ^bb0(%in: f32, %in_7: f32, %in_8: f32, %out: f32):
              %44 = arith.mulf %in_7, %in_8 : f32
              %45 = arith.addf %in, %44 : f32
              linalg.yield %45 : f32
            }
            loom.semaphore_give %29 : memref<1x64xf32>
            loom.semaphore_give %36 : memref<1x64x128xf32>
            linalg.copy ins(%25 : memref<1x64xf32>) outs(%23 : memref<1x64xf32>)
            loom.semaphore_give %25 : memref<1x64xf32>
          }
          loom.semaphore_give %23 : memref<1x64xf32>
          loom.semaphore_give %32 : memref<1x64x128xf32>
          linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%34, %21 : memref<1x64x128xf32>, memref<1x64xf32>) outs(%31 : memref<1x64x128xf32>) {
          ^bb0(%in: f32, %in_5: f32, %out: f32):
            %40 = arith.divf %in, %in_5 : f32
            linalg.yield %40 : f32
          }
          loom.semaphore_give %21 : memref<1x64xf32>
          loom.semaphore_give %34 : memref<1x64x128xf32>
          %reinterpret_cast_4 = memref.reinterpret_cast %arg3 to offset: [%39], sizes: [1, 64, 128], strides: [524288, 128, 1] : memref<32x4096x128xf32> to memref<1x64x128xf32, strided<[524288, 128, 1], offset: ?>>
          loom.copy %31, %reinterpret_cast_4 src_mem_space @L1 dst_mem_space @DRAM, interconnect : [], broadcast : [1, 1] : memref<1x64x128xf32>, memref<1x64x128xf32, strided<[524288, 128, 1], offset: ?>>
          loom.semaphore_give %31 : memref<1x64x128xf32>
        }
        scf.reduce 
      } {loom.iter_types = [#loom.iter_type<spatial>, #loom.iter_type<spatial>], loom.mapped_to_dims = [@x, @y]}
      return
    }
  }
  module attributes {loom.pass_name = "Materialize"} {
    func.func @attention__d0i1_d1i1__f01__d_a_a__BB1__BM64__BN128(%arg0: memref<32x128x4096xf32>, %arg1: memref<32x4096x128xf32>, %arg2: memref<32x4096x128xf32>, %arg3: memref<32x4096x128xf32>) {
      %c16384 = arith.constant 16384 : index
      %c8192 = arith.constant 8192 : index
      %c524288 = arith.constant 524288 : index
      %c32 = arith.constant 32 : index
      %cst = arith.constant 2.000000e+00 : f32
      %cst_0 = arith.constant 0.12751743074602467 : f64
      %cst_1 = arith.constant 0.000000e+00 : f32
      %cst_2 = arith.constant 1.000000e+00 : f32
      %cst_3 = arith.constant 0xFF800000 : f32
      %c128 = arith.constant 128 : index
      %c0 = arith.constant 0 : index
      %c8 = arith.constant 8 : index
      %c1 = arith.constant 1 : index
      scf.parallel (%arg4, %arg5) = (%c0, %c0) to (%c8, %c8) step (%c1, %c1) {
        scf.for %arg6 = %c0 to %c32 step %c1 {
          %12 = arith.muli %arg4, %c8 overflow<nsw> : index
          %13 = arith.addi %12, %arg5 : index
          %14 = loom.alloc [1, 128, 128] on @L1 : memref<1x128x128xf32>
          %15 = loom.semaphore_take %14 : memref<1x128x128xf32> -> memref<1x128x128xf32>
          %16 = loom.alloc [1, 64, 128] on @L1 : memref<1x64x128xf32>
          %17 = loom.semaphore_take %16 : memref<1x64x128xf32> -> memref<1x64x128xf32>
          %18 = loom.alloc [1, 128, 128] on @L1 : memref<1x128x128xf32>
          %19 = loom.semaphore_take %18 : memref<1x128x128xf32> -> memref<1x128x128xf32>
          %20 = loom.alloc [1, 64] on @L1 : memref<1x64xf32>
          %21 = loom.semaphore_take %20 : memref<1x64xf32> -> memref<1x64xf32>
          %22 = loom.alloc [1, 64] on @L1 : memref<1x64xf32>
          %23 = loom.semaphore_take %22 : memref<1x64xf32> -> memref<1x64xf32>
          %24 = loom.alloc [1, 64] on @L1 : memref<1x64xf32>
          %25 = loom.semaphore_take %24 : memref<1x64xf32> -> memref<1x64xf32>
          %26 = loom.alloc [1, 64] on @L1 : memref<1x64xf32>
          %27 = loom.semaphore_take %26 : memref<1x64xf32> -> memref<1x64xf32>
          %28 = loom.alloc [1, 64] on @L1 : memref<1x64xf32>
          %29 = loom.semaphore_take %28 : memref<1x64xf32> -> memref<1x64xf32>
          %30 = loom.alloc [1, 64, 128] on @L1 : memref<1x64x128xf32>
          %31 = loom.semaphore_take %30 : memref<1x64x128xf32> -> memref<1x64x128xf32>
          %32 = loom.semaphore_take %30 : memref<1x64x128xf32> -> memref<1x64x128xf32>
          %33 = loom.alloc [1, 64, 128] on @L1 : memref<1x64x128xf32>
          %34 = loom.semaphore_take %33 : memref<1x64x128xf32> -> memref<1x64x128xf32>
          %35 = loom.alloc [1, 64, 128] on @L1 : memref<1x64x128xf32>
          %36 = loom.semaphore_take %35 : memref<1x64x128xf32> -> memref<1x64x128xf32>
          %37 = arith.muli %arg6, %c524288 overflow<nsw> : index
          %38 = arith.muli %13, %c8192 : index
          %39 = arith.addi %37, %38 : index
          %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%39], sizes: [1, 64, 128], strides: [524288, 128, 1] : memref<32x4096x128xf32> to memref<1x64x128xf32, strided<[524288, 128, 1], offset: ?>>
          loom.copy %reinterpret_cast, %32 src_mem_space @DRAM dst_mem_space @L1, interconnect : [], broadcast : [1, 1] : memref<1x64x128xf32, strided<[524288, 128, 1], offset: ?>>, memref<1x64x128xf32>
          linalg.fill ins(%cst_1 : f32) outs(%34 : memref<1x64x128xf32>)
          linalg.fill ins(%cst_2 : f32) outs(%21 : memref<1x64xf32>)
          linalg.fill ins(%cst_3 : f32) outs(%23 : memref<1x64xf32>)
          scf.for %arg7 = %c0 to %c32 step %c1 {
            %40 = arith.muli %arg7, %c128 : index
            %41 = arith.addi %37, %40 : index
            %reinterpret_cast_5 = memref.reinterpret_cast %arg0 to offset: [%41], sizes: [1, 128, 128], strides: [524288, 4096, 1] : memref<32x128x4096xf32> to memref<1x128x128xf32, strided<[524288, 4096, 1], offset: ?>>
            loom.copy %reinterpret_cast_5, %19 src_mem_space @DRAM dst_mem_space @L1, interconnect : [@horizontal_links, @vertical_links], broadcast : [8, 8] : memref<1x128x128xf32, strided<[524288, 4096, 1], offset: ?>>, memref<1x128x128xf32>
            linalg.fill ins(%cst_1 : f32) outs(%17 : memref<1x64x128xf32>)
            linalg.batch_matmul ins(%32, %19 : memref<1x64x128xf32>, memref<1x128x128xf32>) outs(%17 : memref<1x64x128xf32>)
            loom.semaphore_give %19 : memref<1x128x128xf32>
            linalg.fill ins(%cst_3 : f32) outs(%25 : memref<1x64xf32>)
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%17 : memref<1x64x128xf32>) outs(%25 : memref<1x64xf32>) {
            ^bb0(%in: f32, %out: f32):
              %44 = arith.maximumf %in, %out : f32
              linalg.yield %44 : f32
            }
            linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%23, %25 : memref<1x64xf32>, memref<1x64xf32>) outs(%25 : memref<1x64xf32>) {
            ^bb0(%in: f32, %in_7: f32, %out: f32):
              %44 = arith.truncf %cst_0 : f64 to f32
              %45 = arith.mulf %in_7, %44 : f32
              %46 = arith.cmpf ogt, %in, %45 : f32
              %47 = arith.select %46, %in, %45 : f32
              linalg.yield %47 : f32
            }
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%17, %25 : memref<1x64x128xf32>, memref<1x64xf32>) outs(%17 : memref<1x64x128xf32>) {
            ^bb0(%in: f32, %in_7: f32, %out: f32):
              %44 = arith.truncf %cst_0 : f64 to f32
              %45 = arith.mulf %in, %44 : f32
              %46 = arith.subf %45, %in_7 : f32
              %47 = math.powf %cst, %46 : f32
              linalg.yield %47 : f32
            }
            linalg.fill ins(%cst_1 : f32) outs(%27 : memref<1x64xf32>)
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%17 : memref<1x64x128xf32>) outs(%27 : memref<1x64xf32>) {
            ^bb0(%in: f32, %out: f32):
              %44 = arith.addf %in, %out : f32
              linalg.yield %44 : f32
            }
            linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%23, %25 : memref<1x64xf32>, memref<1x64xf32>) outs(%29 : memref<1x64xf32>) {
            ^bb0(%in: f32, %in_7: f32, %out: f32):
              %44 = arith.subf %in, %in_7 : f32
              %45 = math.powf %cst, %44 : f32
              linalg.yield %45 : f32
            }
            linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%21, %29, %27 : memref<1x64xf32>, memref<1x64xf32>, memref<1x64xf32>) outs(%21 : memref<1x64xf32>) {
            ^bb0(%in: f32, %in_7: f32, %in_8: f32, %out: f32):
              %44 = arith.mulf %in, %in_7 : f32
              %45 = arith.addf %44, %in_8 : f32
              linalg.yield %45 : f32
            }
            loom.semaphore_give %27 : memref<1x64xf32>
            %42 = arith.muli %arg7, %c16384 : index
            %43 = arith.addi %37, %42 : index
            %reinterpret_cast_6 = memref.reinterpret_cast %arg1 to offset: [%43], sizes: [1, 128, 128], strides: [524288, 128, 1] : memref<32x4096x128xf32> to memref<1x128x128xf32, strided<[524288, 128, 1], offset: ?>>
            loom.copy %reinterpret_cast_6, %15 src_mem_space @DRAM dst_mem_space @L1, interconnect : [@horizontal_links, @vertical_links], broadcast : [8, 8] : memref<1x128x128xf32, strided<[524288, 128, 1], offset: ?>>, memref<1x128x128xf32>
            linalg.fill ins(%cst_1 : f32) outs(%36 : memref<1x64x128xf32>)
            linalg.batch_matmul ins(%17, %15 : memref<1x64x128xf32>, memref<1x128x128xf32>) outs(%36 : memref<1x64x128xf32>)
            loom.semaphore_give %15 : memref<1x128x128xf32>
            loom.semaphore_give %17 : memref<1x64x128xf32>
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%36, %34, %29 : memref<1x64x128xf32>, memref<1x64x128xf32>, memref<1x64xf32>) outs(%34 : memref<1x64x128xf32>) {
            ^bb0(%in: f32, %in_7: f32, %in_8: f32, %out: f32):
              %44 = arith.mulf %in_7, %in_8 : f32
              %45 = arith.addf %in, %44 : f32
              linalg.yield %45 : f32
            }
            loom.semaphore_give %29 : memref<1x64xf32>
            loom.semaphore_give %36 : memref<1x64x128xf32>
            linalg.copy ins(%25 : memref<1x64xf32>) outs(%23 : memref<1x64xf32>)
            loom.semaphore_give %25 : memref<1x64xf32>
          }
          loom.semaphore_give %23 : memref<1x64xf32>
          loom.semaphore_give %32 : memref<1x64x128xf32>
          linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%34, %21 : memref<1x64x128xf32>, memref<1x64xf32>) outs(%31 : memref<1x64x128xf32>) {
          ^bb0(%in: f32, %in_5: f32, %out: f32):
            %40 = arith.divf %in, %in_5 : f32
            linalg.yield %40 : f32
          }
          loom.semaphore_give %21 : memref<1x64xf32>
          loom.semaphore_give %34 : memref<1x64x128xf32>
          %reinterpret_cast_4 = memref.reinterpret_cast %arg3 to offset: [%39], sizes: [1, 64, 128], strides: [524288, 128, 1] : memref<32x4096x128xf32> to memref<1x64x128xf32, strided<[524288, 128, 1], offset: ?>>
          loom.copy %31, %reinterpret_cast_4 src_mem_space @L1 dst_mem_space @DRAM, interconnect : [], broadcast : [1, 1] : memref<1x64x128xf32>, memref<1x64x128xf32, strided<[524288, 128, 1], offset: ?>>
          loom.semaphore_give %31 : memref<1x64x128xf32>
        }
        scf.reduce 
      } {loom.iter_types = [#loom.iter_type<spatial>, #loom.iter_type<spatial>], loom.mapped_to_dims = [@x, @y]}
      return
    }
  }
  module attributes {loom.pass_name = "Materialize"} {
    func.func @attention__d0i1_d1i1__f01__d_a_h__BB1__BM64__BN128(%arg0: memref<32x128x4096xf32>, %arg1: memref<32x4096x128xf32>, %arg2: memref<32x4096x128xf32>, %arg3: memref<32x4096x128xf32>) {
      %c16384 = arith.constant 16384 : index
      %c8192 = arith.constant 8192 : index
      %c524288 = arith.constant 524288 : index
      %c32 = arith.constant 32 : index
      %cst = arith.constant 2.000000e+00 : f32
      %cst_0 = arith.constant 0.12751743074602467 : f64
      %cst_1 = arith.constant 0.000000e+00 : f32
      %cst_2 = arith.constant 1.000000e+00 : f32
      %cst_3 = arith.constant 0xFF800000 : f32
      %c128 = arith.constant 128 : index
      %c0 = arith.constant 0 : index
      %c8 = arith.constant 8 : index
      %c1 = arith.constant 1 : index
      scf.parallel (%arg4, %arg5) = (%c0, %c0) to (%c8, %c8) step (%c1, %c1) {
        scf.for %arg6 = %c0 to %c32 step %c1 {
          %12 = arith.muli %arg4, %c8 overflow<nsw> : index
          %13 = arith.addi %12, %arg5 : index
          %14 = loom.alloc [1, 128, 128] on @L1 : memref<1x128x128xf32>
          %15 = loom.semaphore_take %14 : memref<1x128x128xf32> -> memref<1x128x128xf32>
          %16 = loom.alloc [1, 64, 128] on @L1 : memref<1x64x128xf32>
          %17 = loom.semaphore_take %16 : memref<1x64x128xf32> -> memref<1x64x128xf32>
          %18 = loom.alloc [1, 128, 128] on @L1 : memref<1x128x128xf32>
          %19 = loom.semaphore_take %18 : memref<1x128x128xf32> -> memref<1x128x128xf32>
          %20 = loom.alloc [1, 64] on @L1 : memref<1x64xf32>
          %21 = loom.semaphore_take %20 : memref<1x64xf32> -> memref<1x64xf32>
          %22 = loom.alloc [1, 64] on @L1 : memref<1x64xf32>
          %23 = loom.semaphore_take %22 : memref<1x64xf32> -> memref<1x64xf32>
          %24 = loom.alloc [1, 64] on @L1 : memref<1x64xf32>
          %25 = loom.semaphore_take %24 : memref<1x64xf32> -> memref<1x64xf32>
          %26 = loom.alloc [1, 64] on @L1 : memref<1x64xf32>
          %27 = loom.semaphore_take %26 : memref<1x64xf32> -> memref<1x64xf32>
          %28 = loom.alloc [1, 64] on @L1 : memref<1x64xf32>
          %29 = loom.semaphore_take %28 : memref<1x64xf32> -> memref<1x64xf32>
          %30 = loom.alloc [1, 64, 128] on @L1 : memref<1x64x128xf32>
          %31 = loom.semaphore_take %30 : memref<1x64x128xf32> -> memref<1x64x128xf32>
          %32 = loom.semaphore_take %30 : memref<1x64x128xf32> -> memref<1x64x128xf32>
          %33 = loom.alloc [1, 64, 128] on @L1 : memref<1x64x128xf32>
          %34 = loom.semaphore_take %33 : memref<1x64x128xf32> -> memref<1x64x128xf32>
          %35 = loom.alloc [1, 64, 128] on @L1 : memref<1x64x128xf32>
          %36 = loom.semaphore_take %35 : memref<1x64x128xf32> -> memref<1x64x128xf32>
          %37 = arith.muli %arg6, %c524288 overflow<nsw> : index
          %38 = arith.muli %13, %c8192 : index
          %39 = arith.addi %37, %38 : index
          %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%39], sizes: [1, 64, 128], strides: [524288, 128, 1] : memref<32x4096x128xf32> to memref<1x64x128xf32, strided<[524288, 128, 1], offset: ?>>
          loom.copy %reinterpret_cast, %32 src_mem_space @DRAM dst_mem_space @L1, interconnect : [], broadcast : [1, 1] : memref<1x64x128xf32, strided<[524288, 128, 1], offset: ?>>, memref<1x64x128xf32>
          linalg.fill ins(%cst_1 : f32) outs(%34 : memref<1x64x128xf32>)
          linalg.fill ins(%cst_2 : f32) outs(%21 : memref<1x64xf32>)
          linalg.fill ins(%cst_3 : f32) outs(%23 : memref<1x64xf32>)
          scf.for %arg7 = %c0 to %c32 step %c1 {
            %40 = arith.muli %arg7, %c128 : index
            %41 = arith.addi %37, %40 : index
            %reinterpret_cast_5 = memref.reinterpret_cast %arg0 to offset: [%41], sizes: [1, 128, 128], strides: [524288, 4096, 1] : memref<32x128x4096xf32> to memref<1x128x128xf32, strided<[524288, 4096, 1], offset: ?>>
            loom.copy %reinterpret_cast_5, %19 src_mem_space @DRAM dst_mem_space @L1, interconnect : [@horizontal_links, @vertical_links], broadcast : [8, 8] : memref<1x128x128xf32, strided<[524288, 4096, 1], offset: ?>>, memref<1x128x128xf32>
            linalg.fill ins(%cst_1 : f32) outs(%17 : memref<1x64x128xf32>)
            linalg.batch_matmul ins(%32, %19 : memref<1x64x128xf32>, memref<1x128x128xf32>) outs(%17 : memref<1x64x128xf32>)
            loom.semaphore_give %19 : memref<1x128x128xf32>
            linalg.fill ins(%cst_3 : f32) outs(%25 : memref<1x64xf32>)
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%17 : memref<1x64x128xf32>) outs(%25 : memref<1x64xf32>) {
            ^bb0(%in: f32, %out: f32):
              %44 = arith.maximumf %in, %out : f32
              linalg.yield %44 : f32
            }
            linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%23, %25 : memref<1x64xf32>, memref<1x64xf32>) outs(%25 : memref<1x64xf32>) {
            ^bb0(%in: f32, %in_7: f32, %out: f32):
              %44 = arith.truncf %cst_0 : f64 to f32
              %45 = arith.mulf %in_7, %44 : f32
              %46 = arith.cmpf ogt, %in, %45 : f32
              %47 = arith.select %46, %in, %45 : f32
              linalg.yield %47 : f32
            }
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%17, %25 : memref<1x64x128xf32>, memref<1x64xf32>) outs(%17 : memref<1x64x128xf32>) {
            ^bb0(%in: f32, %in_7: f32, %out: f32):
              %44 = arith.truncf %cst_0 : f64 to f32
              %45 = arith.mulf %in, %44 : f32
              %46 = arith.subf %45, %in_7 : f32
              %47 = math.powf %cst, %46 : f32
              linalg.yield %47 : f32
            }
            linalg.fill ins(%cst_1 : f32) outs(%27 : memref<1x64xf32>)
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%17 : memref<1x64x128xf32>) outs(%27 : memref<1x64xf32>) {
            ^bb0(%in: f32, %out: f32):
              %44 = arith.addf %in, %out : f32
              linalg.yield %44 : f32
            }
            linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%23, %25 : memref<1x64xf32>, memref<1x64xf32>) outs(%29 : memref<1x64xf32>) {
            ^bb0(%in: f32, %in_7: f32, %out: f32):
              %44 = arith.subf %in, %in_7 : f32
              %45 = math.powf %cst, %44 : f32
              linalg.yield %45 : f32
            }
            linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%21, %29, %27 : memref<1x64xf32>, memref<1x64xf32>, memref<1x64xf32>) outs(%21 : memref<1x64xf32>) {
            ^bb0(%in: f32, %in_7: f32, %in_8: f32, %out: f32):
              %44 = arith.mulf %in, %in_7 : f32
              %45 = arith.addf %44, %in_8 : f32
              linalg.yield %45 : f32
            }
            loom.semaphore_give %27 : memref<1x64xf32>
            %42 = arith.muli %arg7, %c16384 : index
            %43 = arith.addi %37, %42 : index
            %reinterpret_cast_6 = memref.reinterpret_cast %arg1 to offset: [%43], sizes: [1, 128, 128], strides: [524288, 128, 1] : memref<32x4096x128xf32> to memref<1x128x128xf32, strided<[524288, 128, 1], offset: ?>>
            loom.copy %reinterpret_cast_6, %15 src_mem_space @DRAM dst_mem_space @L1, interconnect : [@horizontal_links], broadcast : [1, 8] : memref<1x128x128xf32, strided<[524288, 128, 1], offset: ?>>, memref<1x128x128xf32>
            linalg.fill ins(%cst_1 : f32) outs(%36 : memref<1x64x128xf32>)
            linalg.batch_matmul ins(%17, %15 : memref<1x64x128xf32>, memref<1x128x128xf32>) outs(%36 : memref<1x64x128xf32>)
            loom.semaphore_give %15 : memref<1x128x128xf32>
            loom.semaphore_give %17 : memref<1x64x128xf32>
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%36, %34, %29 : memref<1x64x128xf32>, memref<1x64x128xf32>, memref<1x64xf32>) outs(%34 : memref<1x64x128xf32>) {
            ^bb0(%in: f32, %in_7: f32, %in_8: f32, %out: f32):
              %44 = arith.mulf %in_7, %in_8 : f32
              %45 = arith.addf %in, %44 : f32
              linalg.yield %45 : f32
            }
            loom.semaphore_give %29 : memref<1x64xf32>
            loom.semaphore_give %36 : memref<1x64x128xf32>
            linalg.copy ins(%25 : memref<1x64xf32>) outs(%23 : memref<1x64xf32>)
            loom.semaphore_give %25 : memref<1x64xf32>
          }
          loom.semaphore_give %23 : memref<1x64xf32>
          loom.semaphore_give %32 : memref<1x64x128xf32>
          linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%34, %21 : memref<1x64x128xf32>, memref<1x64xf32>) outs(%31 : memref<1x64x128xf32>) {
          ^bb0(%in: f32, %in_5: f32, %out: f32):
            %40 = arith.divf %in, %in_5 : f32
            linalg.yield %40 : f32
          }
          loom.semaphore_give %21 : memref<1x64xf32>
          loom.semaphore_give %34 : memref<1x64x128xf32>
          %reinterpret_cast_4 = memref.reinterpret_cast %arg3 to offset: [%39], sizes: [1, 64, 128], strides: [524288, 128, 1] : memref<32x4096x128xf32> to memref<1x64x128xf32, strided<[524288, 128, 1], offset: ?>>
          loom.copy %31, %reinterpret_cast_4 src_mem_space @L1 dst_mem_space @DRAM, interconnect : [], broadcast : [1, 1] : memref<1x64x128xf32>, memref<1x64x128xf32, strided<[524288, 128, 1], offset: ?>>
          loom.semaphore_give %31 : memref<1x64x128xf32>
        }
        scf.reduce 
      } {loom.iter_types = [#loom.iter_type<spatial>, #loom.iter_type<spatial>], loom.mapped_to_dims = [@x, @y]}
      return
    }
  }
  module attributes {loom.pass_name = "Materialize"} {
    func.func @attention__d0i1_d1i1__f01__d_a_v__BB1__BM64__BN128(%arg0: memref<32x128x4096xf32>, %arg1: memref<32x4096x128xf32>, %arg2: memref<32x4096x128xf32>, %arg3: memref<32x4096x128xf32>) {
      %c16384 = arith.constant 16384 : index
      %c8192 = arith.constant 8192 : index
      %c524288 = arith.constant 524288 : index
      %c32 = arith.constant 32 : index
      %cst = arith.constant 2.000000e+00 : f32
      %cst_0 = arith.constant 0.12751743074602467 : f64
      %cst_1 = arith.constant 0.000000e+00 : f32
      %cst_2 = arith.constant 1.000000e+00 : f32
      %cst_3 = arith.constant 0xFF800000 : f32
      %c128 = arith.constant 128 : index
      %c0 = arith.constant 0 : index
      %c8 = arith.constant 8 : index
      %c1 = arith.constant 1 : index
      scf.parallel (%arg4, %arg5) = (%c0, %c0) to (%c8, %c8) step (%c1, %c1) {
        scf.for %arg6 = %c0 to %c32 step %c1 {
          %12 = arith.muli %arg4, %c8 overflow<nsw> : index
          %13 = arith.addi %12, %arg5 : index
          %14 = loom.alloc [1, 128, 128] on @L1 : memref<1x128x128xf32>
          %15 = loom.semaphore_take %14 : memref<1x128x128xf32> -> memref<1x128x128xf32>
          %16 = loom.alloc [1, 64, 128] on @L1 : memref<1x64x128xf32>
          %17 = loom.semaphore_take %16 : memref<1x64x128xf32> -> memref<1x64x128xf32>
          %18 = loom.alloc [1, 128, 128] on @L1 : memref<1x128x128xf32>
          %19 = loom.semaphore_take %18 : memref<1x128x128xf32> -> memref<1x128x128xf32>
          %20 = loom.alloc [1, 64] on @L1 : memref<1x64xf32>
          %21 = loom.semaphore_take %20 : memref<1x64xf32> -> memref<1x64xf32>
          %22 = loom.alloc [1, 64] on @L1 : memref<1x64xf32>
          %23 = loom.semaphore_take %22 : memref<1x64xf32> -> memref<1x64xf32>
          %24 = loom.alloc [1, 64] on @L1 : memref<1x64xf32>
          %25 = loom.semaphore_take %24 : memref<1x64xf32> -> memref<1x64xf32>
          %26 = loom.alloc [1, 64] on @L1 : memref<1x64xf32>
          %27 = loom.semaphore_take %26 : memref<1x64xf32> -> memref<1x64xf32>
          %28 = loom.alloc [1, 64] on @L1 : memref<1x64xf32>
          %29 = loom.semaphore_take %28 : memref<1x64xf32> -> memref<1x64xf32>
          %30 = loom.alloc [1, 64, 128] on @L1 : memref<1x64x128xf32>
          %31 = loom.semaphore_take %30 : memref<1x64x128xf32> -> memref<1x64x128xf32>
          %32 = loom.semaphore_take %30 : memref<1x64x128xf32> -> memref<1x64x128xf32>
          %33 = loom.alloc [1, 64, 128] on @L1 : memref<1x64x128xf32>
          %34 = loom.semaphore_take %33 : memref<1x64x128xf32> -> memref<1x64x128xf32>
          %35 = loom.alloc [1, 64, 128] on @L1 : memref<1x64x128xf32>
          %36 = loom.semaphore_take %35 : memref<1x64x128xf32> -> memref<1x64x128xf32>
          %37 = arith.muli %arg6, %c524288 overflow<nsw> : index
          %38 = arith.muli %13, %c8192 : index
          %39 = arith.addi %37, %38 : index
          %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%39], sizes: [1, 64, 128], strides: [524288, 128, 1] : memref<32x4096x128xf32> to memref<1x64x128xf32, strided<[524288, 128, 1], offset: ?>>
          loom.copy %reinterpret_cast, %32 src_mem_space @DRAM dst_mem_space @L1, interconnect : [], broadcast : [1, 1] : memref<1x64x128xf32, strided<[524288, 128, 1], offset: ?>>, memref<1x64x128xf32>
          linalg.fill ins(%cst_1 : f32) outs(%34 : memref<1x64x128xf32>)
          linalg.fill ins(%cst_2 : f32) outs(%21 : memref<1x64xf32>)
          linalg.fill ins(%cst_3 : f32) outs(%23 : memref<1x64xf32>)
          scf.for %arg7 = %c0 to %c32 step %c1 {
            %40 = arith.muli %arg7, %c128 : index
            %41 = arith.addi %37, %40 : index
            %reinterpret_cast_5 = memref.reinterpret_cast %arg0 to offset: [%41], sizes: [1, 128, 128], strides: [524288, 4096, 1] : memref<32x128x4096xf32> to memref<1x128x128xf32, strided<[524288, 4096, 1], offset: ?>>
            loom.copy %reinterpret_cast_5, %19 src_mem_space @DRAM dst_mem_space @L1, interconnect : [@horizontal_links, @vertical_links], broadcast : [8, 8] : memref<1x128x128xf32, strided<[524288, 4096, 1], offset: ?>>, memref<1x128x128xf32>
            linalg.fill ins(%cst_1 : f32) outs(%17 : memref<1x64x128xf32>)
            linalg.batch_matmul ins(%32, %19 : memref<1x64x128xf32>, memref<1x128x128xf32>) outs(%17 : memref<1x64x128xf32>)
            loom.semaphore_give %19 : memref<1x128x128xf32>
            linalg.fill ins(%cst_3 : f32) outs(%25 : memref<1x64xf32>)
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%17 : memref<1x64x128xf32>) outs(%25 : memref<1x64xf32>) {
            ^bb0(%in: f32, %out: f32):
              %44 = arith.maximumf %in, %out : f32
              linalg.yield %44 : f32
            }
            linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%23, %25 : memref<1x64xf32>, memref<1x64xf32>) outs(%25 : memref<1x64xf32>) {
            ^bb0(%in: f32, %in_7: f32, %out: f32):
              %44 = arith.truncf %cst_0 : f64 to f32
              %45 = arith.mulf %in_7, %44 : f32
              %46 = arith.cmpf ogt, %in, %45 : f32
              %47 = arith.select %46, %in, %45 : f32
              linalg.yield %47 : f32
            }
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%17, %25 : memref<1x64x128xf32>, memref<1x64xf32>) outs(%17 : memref<1x64x128xf32>) {
            ^bb0(%in: f32, %in_7: f32, %out: f32):
              %44 = arith.truncf %cst_0 : f64 to f32
              %45 = arith.mulf %in, %44 : f32
              %46 = arith.subf %45, %in_7 : f32
              %47 = math.powf %cst, %46 : f32
              linalg.yield %47 : f32
            }
            linalg.fill ins(%cst_1 : f32) outs(%27 : memref<1x64xf32>)
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%17 : memref<1x64x128xf32>) outs(%27 : memref<1x64xf32>) {
            ^bb0(%in: f32, %out: f32):
              %44 = arith.addf %in, %out : f32
              linalg.yield %44 : f32
            }
            linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%23, %25 : memref<1x64xf32>, memref<1x64xf32>) outs(%29 : memref<1x64xf32>) {
            ^bb0(%in: f32, %in_7: f32, %out: f32):
              %44 = arith.subf %in, %in_7 : f32
              %45 = math.powf %cst, %44 : f32
              linalg.yield %45 : f32
            }
            linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%21, %29, %27 : memref<1x64xf32>, memref<1x64xf32>, memref<1x64xf32>) outs(%21 : memref<1x64xf32>) {
            ^bb0(%in: f32, %in_7: f32, %in_8: f32, %out: f32):
              %44 = arith.mulf %in, %in_7 : f32
              %45 = arith.addf %44, %in_8 : f32
              linalg.yield %45 : f32
            }
            loom.semaphore_give %27 : memref<1x64xf32>
            %42 = arith.muli %arg7, %c16384 : index
            %43 = arith.addi %37, %42 : index
            %reinterpret_cast_6 = memref.reinterpret_cast %arg1 to offset: [%43], sizes: [1, 128, 128], strides: [524288, 128, 1] : memref<32x4096x128xf32> to memref<1x128x128xf32, strided<[524288, 128, 1], offset: ?>>
            loom.copy %reinterpret_cast_6, %15 src_mem_space @DRAM dst_mem_space @L1, interconnect : [@vertical_links], broadcast : [8, 1] : memref<1x128x128xf32, strided<[524288, 128, 1], offset: ?>>, memref<1x128x128xf32>
            linalg.fill ins(%cst_1 : f32) outs(%36 : memref<1x64x128xf32>)
            linalg.batch_matmul ins(%17, %15 : memref<1x64x128xf32>, memref<1x128x128xf32>) outs(%36 : memref<1x64x128xf32>)
            loom.semaphore_give %15 : memref<1x128x128xf32>
            loom.semaphore_give %17 : memref<1x64x128xf32>
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%36, %34, %29 : memref<1x64x128xf32>, memref<1x64x128xf32>, memref<1x64xf32>) outs(%34 : memref<1x64x128xf32>) {
            ^bb0(%in: f32, %in_7: f32, %in_8: f32, %out: f32):
              %44 = arith.mulf %in_7, %in_8 : f32
              %45 = arith.addf %in, %44 : f32
              linalg.yield %45 : f32
            }
            loom.semaphore_give %29 : memref<1x64xf32>
            loom.semaphore_give %36 : memref<1x64x128xf32>
            linalg.copy ins(%25 : memref<1x64xf32>) outs(%23 : memref<1x64xf32>)
            loom.semaphore_give %25 : memref<1x64xf32>
          }
          loom.semaphore_give %23 : memref<1x64xf32>
          loom.semaphore_give %32 : memref<1x64x128xf32>
          linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%34, %21 : memref<1x64x128xf32>, memref<1x64xf32>) outs(%31 : memref<1x64x128xf32>) {
          ^bb0(%in: f32, %in_5: f32, %out: f32):
            %40 = arith.divf %in, %in_5 : f32
            linalg.yield %40 : f32
          }
          loom.semaphore_give %21 : memref<1x64xf32>
          loom.semaphore_give %34 : memref<1x64x128xf32>
          %reinterpret_cast_4 = memref.reinterpret_cast %arg3 to offset: [%39], sizes: [1, 64, 128], strides: [524288, 128, 1] : memref<32x4096x128xf32> to memref<1x64x128xf32, strided<[524288, 128, 1], offset: ?>>
          loom.copy %31, %reinterpret_cast_4 src_mem_space @L1 dst_mem_space @DRAM, interconnect : [], broadcast : [1, 1] : memref<1x64x128xf32>, memref<1x64x128xf32, strided<[524288, 128, 1], offset: ?>>
          loom.semaphore_give %31 : memref<1x64x128xf32>
        }
        scf.reduce 
      } {loom.iter_types = [#loom.iter_type<spatial>, #loom.iter_type<spatial>], loom.mapped_to_dims = [@x, @y]}
      return
    }
  }
  module attributes {loom.pass_name = "Materialize"} {
    func.func @attention__d0i1_d1i1__f01__d_h_d__BB1__BM64__BN128(%arg0: memref<32x128x4096xf32>, %arg1: memref<32x4096x128xf32>, %arg2: memref<32x4096x128xf32>, %arg3: memref<32x4096x128xf32>) {
      %c16384 = arith.constant 16384 : index
      %c8192 = arith.constant 8192 : index
      %c524288 = arith.constant 524288 : index
      %c32 = arith.constant 32 : index
      %cst = arith.constant 2.000000e+00 : f32
      %cst_0 = arith.constant 0.12751743074602467 : f64
      %cst_1 = arith.constant 0.000000e+00 : f32
      %cst_2 = arith.constant 1.000000e+00 : f32
      %cst_3 = arith.constant 0xFF800000 : f32
      %c128 = arith.constant 128 : index
      %c0 = arith.constant 0 : index
      %c8 = arith.constant 8 : index
      %c1 = arith.constant 1 : index
      scf.parallel (%arg4, %arg5) = (%c0, %c0) to (%c8, %c8) step (%c1, %c1) {
        scf.for %arg6 = %c0 to %c32 step %c1 {
          %12 = arith.muli %arg4, %c8 overflow<nsw> : index
          %13 = arith.addi %12, %arg5 : index
          %14 = loom.alloc [1, 128, 128] on @L1 : memref<1x128x128xf32>
          %15 = loom.semaphore_take %14 : memref<1x128x128xf32> -> memref<1x128x128xf32>
          %16 = loom.alloc [1, 64, 128] on @L1 : memref<1x64x128xf32>
          %17 = loom.semaphore_take %16 : memref<1x64x128xf32> -> memref<1x64x128xf32>
          %18 = loom.alloc [1, 128, 128] on @L1 : memref<1x128x128xf32>
          %19 = loom.semaphore_take %18 : memref<1x128x128xf32> -> memref<1x128x128xf32>
          %20 = loom.alloc [1, 64] on @L1 : memref<1x64xf32>
          %21 = loom.semaphore_take %20 : memref<1x64xf32> -> memref<1x64xf32>
          %22 = loom.alloc [1, 64] on @L1 : memref<1x64xf32>
          %23 = loom.semaphore_take %22 : memref<1x64xf32> -> memref<1x64xf32>
          %24 = loom.alloc [1, 64] on @L1 : memref<1x64xf32>
          %25 = loom.semaphore_take %24 : memref<1x64xf32> -> memref<1x64xf32>
          %26 = loom.alloc [1, 64] on @L1 : memref<1x64xf32>
          %27 = loom.semaphore_take %26 : memref<1x64xf32> -> memref<1x64xf32>
          %28 = loom.alloc [1, 64] on @L1 : memref<1x64xf32>
          %29 = loom.semaphore_take %28 : memref<1x64xf32> -> memref<1x64xf32>
          %30 = loom.alloc [1, 64, 128] on @L1 : memref<1x64x128xf32>
          %31 = loom.semaphore_take %30 : memref<1x64x128xf32> -> memref<1x64x128xf32>
          %32 = loom.semaphore_take %30 : memref<1x64x128xf32> -> memref<1x64x128xf32>
          %33 = loom.alloc [1, 64, 128] on @L1 : memref<1x64x128xf32>
          %34 = loom.semaphore_take %33 : memref<1x64x128xf32> -> memref<1x64x128xf32>
          %35 = loom.alloc [1, 64, 128] on @L1 : memref<1x64x128xf32>
          %36 = loom.semaphore_take %35 : memref<1x64x128xf32> -> memref<1x64x128xf32>
          %37 = arith.muli %arg6, %c524288 overflow<nsw> : index
          %38 = arith.muli %13, %c8192 : index
          %39 = arith.addi %37, %38 : index
          %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%39], sizes: [1, 64, 128], strides: [524288, 128, 1] : memref<32x4096x128xf32> to memref<1x64x128xf32, strided<[524288, 128, 1], offset: ?>>
          loom.copy %reinterpret_cast, %32 src_mem_space @DRAM dst_mem_space @L1, interconnect : [], broadcast : [1, 1] : memref<1x64x128xf32, strided<[524288, 128, 1], offset: ?>>, memref<1x64x128xf32>
          linalg.fill ins(%cst_1 : f32) outs(%34 : memref<1x64x128xf32>)
          linalg.fill ins(%cst_2 : f32) outs(%21 : memref<1x64xf32>)
          linalg.fill ins(%cst_3 : f32) outs(%23 : memref<1x64xf32>)
          scf.for %arg7 = %c0 to %c32 step %c1 {
            %40 = arith.muli %arg7, %c128 : index
            %41 = arith.addi %37, %40 : index
            %reinterpret_cast_5 = memref.reinterpret_cast %arg0 to offset: [%41], sizes: [1, 128, 128], strides: [524288, 4096, 1] : memref<32x128x4096xf32> to memref<1x128x128xf32, strided<[524288, 4096, 1], offset: ?>>
            loom.copy %reinterpret_cast_5, %19 src_mem_space @DRAM dst_mem_space @L1, interconnect : [@horizontal_links], broadcast : [1, 8] : memref<1x128x128xf32, strided<[524288, 4096, 1], offset: ?>>, memref<1x128x128xf32>
            linalg.fill ins(%cst_1 : f32) outs(%17 : memref<1x64x128xf32>)
            linalg.batch_matmul ins(%32, %19 : memref<1x64x128xf32>, memref<1x128x128xf32>) outs(%17 : memref<1x64x128xf32>)
            loom.semaphore_give %19 : memref<1x128x128xf32>
            linalg.fill ins(%cst_3 : f32) outs(%25 : memref<1x64xf32>)
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%17 : memref<1x64x128xf32>) outs(%25 : memref<1x64xf32>) {
            ^bb0(%in: f32, %out: f32):
              %44 = arith.maximumf %in, %out : f32
              linalg.yield %44 : f32
            }
            linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%23, %25 : memref<1x64xf32>, memref<1x64xf32>) outs(%25 : memref<1x64xf32>) {
            ^bb0(%in: f32, %in_7: f32, %out: f32):
              %44 = arith.truncf %cst_0 : f64 to f32
              %45 = arith.mulf %in_7, %44 : f32
              %46 = arith.cmpf ogt, %in, %45 : f32
              %47 = arith.select %46, %in, %45 : f32
              linalg.yield %47 : f32
            }
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%17, %25 : memref<1x64x128xf32>, memref<1x64xf32>) outs(%17 : memref<1x64x128xf32>) {
            ^bb0(%in: f32, %in_7: f32, %out: f32):
              %44 = arith.truncf %cst_0 : f64 to f32
              %45 = arith.mulf %in, %44 : f32
              %46 = arith.subf %45, %in_7 : f32
              %47 = math.powf %cst, %46 : f32
              linalg.yield %47 : f32
            }
            linalg.fill ins(%cst_1 : f32) outs(%27 : memref<1x64xf32>)
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%17 : memref<1x64x128xf32>) outs(%27 : memref<1x64xf32>) {
            ^bb0(%in: f32, %out: f32):
              %44 = arith.addf %in, %out : f32
              linalg.yield %44 : f32
            }
            linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%23, %25 : memref<1x64xf32>, memref<1x64xf32>) outs(%29 : memref<1x64xf32>) {
            ^bb0(%in: f32, %in_7: f32, %out: f32):
              %44 = arith.subf %in, %in_7 : f32
              %45 = math.powf %cst, %44 : f32
              linalg.yield %45 : f32
            }
            linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%21, %29, %27 : memref<1x64xf32>, memref<1x64xf32>, memref<1x64xf32>) outs(%21 : memref<1x64xf32>) {
            ^bb0(%in: f32, %in_7: f32, %in_8: f32, %out: f32):
              %44 = arith.mulf %in, %in_7 : f32
              %45 = arith.addf %44, %in_8 : f32
              linalg.yield %45 : f32
            }
            loom.semaphore_give %27 : memref<1x64xf32>
            %42 = arith.muli %arg7, %c16384 : index
            %43 = arith.addi %37, %42 : index
            %reinterpret_cast_6 = memref.reinterpret_cast %arg1 to offset: [%43], sizes: [1, 128, 128], strides: [524288, 128, 1] : memref<32x4096x128xf32> to memref<1x128x128xf32, strided<[524288, 128, 1], offset: ?>>
            loom.copy %reinterpret_cast_6, %15 src_mem_space @DRAM dst_mem_space @L1, interconnect : [], broadcast : [1, 1] : memref<1x128x128xf32, strided<[524288, 128, 1], offset: ?>>, memref<1x128x128xf32>
            linalg.fill ins(%cst_1 : f32) outs(%36 : memref<1x64x128xf32>)
            linalg.batch_matmul ins(%17, %15 : memref<1x64x128xf32>, memref<1x128x128xf32>) outs(%36 : memref<1x64x128xf32>)
            loom.semaphore_give %15 : memref<1x128x128xf32>
            loom.semaphore_give %17 : memref<1x64x128xf32>
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%36, %34, %29 : memref<1x64x128xf32>, memref<1x64x128xf32>, memref<1x64xf32>) outs(%34 : memref<1x64x128xf32>) {
            ^bb0(%in: f32, %in_7: f32, %in_8: f32, %out: f32):
              %44 = arith.mulf %in_7, %in_8 : f32
              %45 = arith.addf %in, %44 : f32
              linalg.yield %45 : f32
            }
            loom.semaphore_give %29 : memref<1x64xf32>
            loom.semaphore_give %36 : memref<1x64x128xf32>
            linalg.copy ins(%25 : memref<1x64xf32>) outs(%23 : memref<1x64xf32>)
            loom.semaphore_give %25 : memref<1x64xf32>
          }
          loom.semaphore_give %23 : memref<1x64xf32>
          loom.semaphore_give %32 : memref<1x64x128xf32>
          linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%34, %21 : memref<1x64x128xf32>, memref<1x64xf32>) outs(%31 : memref<1x64x128xf32>) {
          ^bb0(%in: f32, %in_5: f32, %out: f32):
            %40 = arith.divf %in, %in_5 : f32
            linalg.yield %40 : f32
          }
          loom.semaphore_give %21 : memref<1x64xf32>
          loom.semaphore_give %34 : memref<1x64x128xf32>
          %reinterpret_cast_4 = memref.reinterpret_cast %arg3 to offset: [%39], sizes: [1, 64, 128], strides: [524288, 128, 1] : memref<32x4096x128xf32> to memref<1x64x128xf32, strided<[524288, 128, 1], offset: ?>>
          loom.copy %31, %reinterpret_cast_4 src_mem_space @L1 dst_mem_space @DRAM, interconnect : [], broadcast : [1, 1] : memref<1x64x128xf32>, memref<1x64x128xf32, strided<[524288, 128, 1], offset: ?>>
          loom.semaphore_give %31 : memref<1x64x128xf32>
        }
        scf.reduce 
      } {loom.iter_types = [#loom.iter_type<spatial>, #loom.iter_type<spatial>], loom.mapped_to_dims = [@x, @y]}
      return
    }
  }
  module attributes {loom.pass_name = "Materialize"} {
    func.func @attention__d0i1_d1i1__f01__d_h_a__BB1__BM64__BN128(%arg0: memref<32x128x4096xf32>, %arg1: memref<32x4096x128xf32>, %arg2: memref<32x4096x128xf32>, %arg3: memref<32x4096x128xf32>) {
      %c16384 = arith.constant 16384 : index
      %c8192 = arith.constant 8192 : index
      %c524288 = arith.constant 524288 : index
      %c32 = arith.constant 32 : index
      %cst = arith.constant 2.000000e+00 : f32
      %cst_0 = arith.constant 0.12751743074602467 : f64
      %cst_1 = arith.constant 0.000000e+00 : f32
      %cst_2 = arith.constant 1.000000e+00 : f32
      %cst_3 = arith.constant 0xFF800000 : f32
      %c128 = arith.constant 128 : index
      %c0 = arith.constant 0 : index
      %c8 = arith.constant 8 : index
      %c1 = arith.constant 1 : index
      scf.parallel (%arg4, %arg5) = (%c0, %c0) to (%c8, %c8) step (%c1, %c1) {
        scf.for %arg6 = %c0 to %c32 step %c1 {
          %12 = arith.muli %arg4, %c8 overflow<nsw> : index
          %13 = arith.addi %12, %arg5 : index
          %14 = loom.alloc [1, 128, 128] on @L1 : memref<1x128x128xf32>
          %15 = loom.semaphore_take %14 : memref<1x128x128xf32> -> memref<1x128x128xf32>
          %16 = loom.alloc [1, 64, 128] on @L1 : memref<1x64x128xf32>
          %17 = loom.semaphore_take %16 : memref<1x64x128xf32> -> memref<1x64x128xf32>
          %18 = loom.alloc [1, 128, 128] on @L1 : memref<1x128x128xf32>
          %19 = loom.semaphore_take %18 : memref<1x128x128xf32> -> memref<1x128x128xf32>
          %20 = loom.alloc [1, 64] on @L1 : memref<1x64xf32>
          %21 = loom.semaphore_take %20 : memref<1x64xf32> -> memref<1x64xf32>
          %22 = loom.alloc [1, 64] on @L1 : memref<1x64xf32>
          %23 = loom.semaphore_take %22 : memref<1x64xf32> -> memref<1x64xf32>
          %24 = loom.alloc [1, 64] on @L1 : memref<1x64xf32>
          %25 = loom.semaphore_take %24 : memref<1x64xf32> -> memref<1x64xf32>
          %26 = loom.alloc [1, 64] on @L1 : memref<1x64xf32>
          %27 = loom.semaphore_take %26 : memref<1x64xf32> -> memref<1x64xf32>
          %28 = loom.alloc [1, 64] on @L1 : memref<1x64xf32>
          %29 = loom.semaphore_take %28 : memref<1x64xf32> -> memref<1x64xf32>
          %30 = loom.alloc [1, 64, 128] on @L1 : memref<1x64x128xf32>
          %31 = loom.semaphore_take %30 : memref<1x64x128xf32> -> memref<1x64x128xf32>
          %32 = loom.semaphore_take %30 : memref<1x64x128xf32> -> memref<1x64x128xf32>
          %33 = loom.alloc [1, 64, 128] on @L1 : memref<1x64x128xf32>
          %34 = loom.semaphore_take %33 : memref<1x64x128xf32> -> memref<1x64x128xf32>
          %35 = loom.alloc [1, 64, 128] on @L1 : memref<1x64x128xf32>
          %36 = loom.semaphore_take %35 : memref<1x64x128xf32> -> memref<1x64x128xf32>
          %37 = arith.muli %arg6, %c524288 overflow<nsw> : index
          %38 = arith.muli %13, %c8192 : index
          %39 = arith.addi %37, %38 : index
          %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%39], sizes: [1, 64, 128], strides: [524288, 128, 1] : memref<32x4096x128xf32> to memref<1x64x128xf32, strided<[524288, 128, 1], offset: ?>>
          loom.copy %reinterpret_cast, %32 src_mem_space @DRAM dst_mem_space @L1, interconnect : [], broadcast : [1, 1] : memref<1x64x128xf32, strided<[524288, 128, 1], offset: ?>>, memref<1x64x128xf32>
          linalg.fill ins(%cst_1 : f32) outs(%34 : memref<1x64x128xf32>)
          linalg.fill ins(%cst_2 : f32) outs(%21 : memref<1x64xf32>)
          linalg.fill ins(%cst_3 : f32) outs(%23 : memref<1x64xf32>)
          scf.for %arg7 = %c0 to %c32 step %c1 {
            %40 = arith.muli %arg7, %c128 : index
            %41 = arith.addi %37, %40 : index
            %reinterpret_cast_5 = memref.reinterpret_cast %arg0 to offset: [%41], sizes: [1, 128, 128], strides: [524288, 4096, 1] : memref<32x128x4096xf32> to memref<1x128x128xf32, strided<[524288, 4096, 1], offset: ?>>
            loom.copy %reinterpret_cast_5, %19 src_mem_space @DRAM dst_mem_space @L1, interconnect : [@horizontal_links], broadcast : [1, 8] : memref<1x128x128xf32, strided<[524288, 4096, 1], offset: ?>>, memref<1x128x128xf32>
            linalg.fill ins(%cst_1 : f32) outs(%17 : memref<1x64x128xf32>)
            linalg.batch_matmul ins(%32, %19 : memref<1x64x128xf32>, memref<1x128x128xf32>) outs(%17 : memref<1x64x128xf32>)
            loom.semaphore_give %19 : memref<1x128x128xf32>
            linalg.fill ins(%cst_3 : f32) outs(%25 : memref<1x64xf32>)
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%17 : memref<1x64x128xf32>) outs(%25 : memref<1x64xf32>) {
            ^bb0(%in: f32, %out: f32):
              %44 = arith.maximumf %in, %out : f32
              linalg.yield %44 : f32
            }
            linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%23, %25 : memref<1x64xf32>, memref<1x64xf32>) outs(%25 : memref<1x64xf32>) {
            ^bb0(%in: f32, %in_7: f32, %out: f32):
              %44 = arith.truncf %cst_0 : f64 to f32
              %45 = arith.mulf %in_7, %44 : f32
              %46 = arith.cmpf ogt, %in, %45 : f32
              %47 = arith.select %46, %in, %45 : f32
              linalg.yield %47 : f32
            }
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%17, %25 : memref<1x64x128xf32>, memref<1x64xf32>) outs(%17 : memref<1x64x128xf32>) {
            ^bb0(%in: f32, %in_7: f32, %out: f32):
              %44 = arith.truncf %cst_0 : f64 to f32
              %45 = arith.mulf %in, %44 : f32
              %46 = arith.subf %45, %in_7 : f32
              %47 = math.powf %cst, %46 : f32
              linalg.yield %47 : f32
            }
            linalg.fill ins(%cst_1 : f32) outs(%27 : memref<1x64xf32>)
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%17 : memref<1x64x128xf32>) outs(%27 : memref<1x64xf32>) {
            ^bb0(%in: f32, %out: f32):
              %44 = arith.addf %in, %out : f32
              linalg.yield %44 : f32
            }
            linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%23, %25 : memref<1x64xf32>, memref<1x64xf32>) outs(%29 : memref<1x64xf32>) {
            ^bb0(%in: f32, %in_7: f32, %out: f32):
              %44 = arith.subf %in, %in_7 : f32
              %45 = math.powf %cst, %44 : f32
              linalg.yield %45 : f32
            }
            linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%21, %29, %27 : memref<1x64xf32>, memref<1x64xf32>, memref<1x64xf32>) outs(%21 : memref<1x64xf32>) {
            ^bb0(%in: f32, %in_7: f32, %in_8: f32, %out: f32):
              %44 = arith.mulf %in, %in_7 : f32
              %45 = arith.addf %44, %in_8 : f32
              linalg.yield %45 : f32
            }
            loom.semaphore_give %27 : memref<1x64xf32>
            %42 = arith.muli %arg7, %c16384 : index
            %43 = arith.addi %37, %42 : index
            %reinterpret_cast_6 = memref.reinterpret_cast %arg1 to offset: [%43], sizes: [1, 128, 128], strides: [524288, 128, 1] : memref<32x4096x128xf32> to memref<1x128x128xf32, strided<[524288, 128, 1], offset: ?>>
            loom.copy %reinterpret_cast_6, %15 src_mem_space @DRAM dst_mem_space @L1, interconnect : [@horizontal_links, @vertical_links], broadcast : [8, 8] : memref<1x128x128xf32, strided<[524288, 128, 1], offset: ?>>, memref<1x128x128xf32>
            linalg.fill ins(%cst_1 : f32) outs(%36 : memref<1x64x128xf32>)
            linalg.batch_matmul ins(%17, %15 : memref<1x64x128xf32>, memref<1x128x128xf32>) outs(%36 : memref<1x64x128xf32>)
            loom.semaphore_give %15 : memref<1x128x128xf32>
            loom.semaphore_give %17 : memref<1x64x128xf32>
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%36, %34, %29 : memref<1x64x128xf32>, memref<1x64x128xf32>, memref<1x64xf32>) outs(%34 : memref<1x64x128xf32>) {
            ^bb0(%in: f32, %in_7: f32, %in_8: f32, %out: f32):
              %44 = arith.mulf %in_7, %in_8 : f32
              %45 = arith.addf %in, %44 : f32
              linalg.yield %45 : f32
            }
            loom.semaphore_give %29 : memref<1x64xf32>
            loom.semaphore_give %36 : memref<1x64x128xf32>
            linalg.copy ins(%25 : memref<1x64xf32>) outs(%23 : memref<1x64xf32>)
            loom.semaphore_give %25 : memref<1x64xf32>
          }
          loom.semaphore_give %23 : memref<1x64xf32>
          loom.semaphore_give %32 : memref<1x64x128xf32>
          linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%34, %21 : memref<1x64x128xf32>, memref<1x64xf32>) outs(%31 : memref<1x64x128xf32>) {
          ^bb0(%in: f32, %in_5: f32, %out: f32):
            %40 = arith.divf %in, %in_5 : f32
            linalg.yield %40 : f32
          }
          loom.semaphore_give %21 : memref<1x64xf32>
          loom.semaphore_give %34 : memref<1x64x128xf32>
          %reinterpret_cast_4 = memref.reinterpret_cast %arg3 to offset: [%39], sizes: [1, 64, 128], strides: [524288, 128, 1] : memref<32x4096x128xf32> to memref<1x64x128xf32, strided<[524288, 128, 1], offset: ?>>
          loom.copy %31, %reinterpret_cast_4 src_mem_space @L1 dst_mem_space @DRAM, interconnect : [], broadcast : [1, 1] : memref<1x64x128xf32>, memref<1x64x128xf32, strided<[524288, 128, 1], offset: ?>>
          loom.semaphore_give %31 : memref<1x64x128xf32>
        }
        scf.reduce 
      } {loom.iter_types = [#loom.iter_type<spatial>, #loom.iter_type<spatial>], loom.mapped_to_dims = [@x, @y]}
      return
    }
  }
  module attributes {loom.pass_name = "Materialize"} {
    func.func @attention__d0i1_d1i1__f01__d_h_h__BB1__BM64__BN128(%arg0: memref<32x128x4096xf32>, %arg1: memref<32x4096x128xf32>, %arg2: memref<32x4096x128xf32>, %arg3: memref<32x4096x128xf32>) {
      %c16384 = arith.constant 16384 : index
      %c8192 = arith.constant 8192 : index
      %c524288 = arith.constant 524288 : index
      %c32 = arith.constant 32 : index
      %cst = arith.constant 2.000000e+00 : f32
      %cst_0 = arith.constant 0.12751743074602467 : f64
      %cst_1 = arith.constant 0.000000e+00 : f32
      %cst_2 = arith.constant 1.000000e+00 : f32
      %cst_3 = arith.constant 0xFF800000 : f32
      %c128 = arith.constant 128 : index
      %c0 = arith.constant 0 : index
      %c8 = arith.constant 8 : index
      %c1 = arith.constant 1 : index
      scf.parallel (%arg4, %arg5) = (%c0, %c0) to (%c8, %c8) step (%c1, %c1) {
        scf.for %arg6 = %c0 to %c32 step %c1 {
          %12 = arith.muli %arg4, %c8 overflow<nsw> : index
          %13 = arith.addi %12, %arg5 : index
          %14 = loom.alloc [1, 128, 128] on @L1 : memref<1x128x128xf32>
          %15 = loom.semaphore_take %14 : memref<1x128x128xf32> -> memref<1x128x128xf32>
          %16 = loom.alloc [1, 64, 128] on @L1 : memref<1x64x128xf32>
          %17 = loom.semaphore_take %16 : memref<1x64x128xf32> -> memref<1x64x128xf32>
          %18 = loom.alloc [1, 128, 128] on @L1 : memref<1x128x128xf32>
          %19 = loom.semaphore_take %18 : memref<1x128x128xf32> -> memref<1x128x128xf32>
          %20 = loom.alloc [1, 64] on @L1 : memref<1x64xf32>
          %21 = loom.semaphore_take %20 : memref<1x64xf32> -> memref<1x64xf32>
          %22 = loom.alloc [1, 64] on @L1 : memref<1x64xf32>
          %23 = loom.semaphore_take %22 : memref<1x64xf32> -> memref<1x64xf32>
          %24 = loom.alloc [1, 64] on @L1 : memref<1x64xf32>
          %25 = loom.semaphore_take %24 : memref<1x64xf32> -> memref<1x64xf32>
          %26 = loom.alloc [1, 64] on @L1 : memref<1x64xf32>
          %27 = loom.semaphore_take %26 : memref<1x64xf32> -> memref<1x64xf32>
          %28 = loom.alloc [1, 64] on @L1 : memref<1x64xf32>
          %29 = loom.semaphore_take %28 : memref<1x64xf32> -> memref<1x64xf32>
          %30 = loom.alloc [1, 64, 128] on @L1 : memref<1x64x128xf32>
          %31 = loom.semaphore_take %30 : memref<1x64x128xf32> -> memref<1x64x128xf32>
          %32 = loom.semaphore_take %30 : memref<1x64x128xf32> -> memref<1x64x128xf32>
          %33 = loom.alloc [1, 64, 128] on @L1 : memref<1x64x128xf32>
          %34 = loom.semaphore_take %33 : memref<1x64x128xf32> -> memref<1x64x128xf32>
          %35 = loom.alloc [1, 64, 128] on @L1 : memref<1x64x128xf32>
          %36 = loom.semaphore_take %35 : memref<1x64x128xf32> -> memref<1x64x128xf32>
          %37 = arith.muli %arg6, %c524288 overflow<nsw> : index
          %38 = arith.muli %13, %c8192 : index
          %39 = arith.addi %37, %38 : index
          %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%39], sizes: [1, 64, 128], strides: [524288, 128, 1] : memref<32x4096x128xf32> to memref<1x64x128xf32, strided<[524288, 128, 1], offset: ?>>
          loom.copy %reinterpret_cast, %32 src_mem_space @DRAM dst_mem_space @L1, interconnect : [], broadcast : [1, 1] : memref<1x64x128xf32, strided<[524288, 128, 1], offset: ?>>, memref<1x64x128xf32>
          linalg.fill ins(%cst_1 : f32) outs(%34 : memref<1x64x128xf32>)
          linalg.fill ins(%cst_2 : f32) outs(%21 : memref<1x64xf32>)
          linalg.fill ins(%cst_3 : f32) outs(%23 : memref<1x64xf32>)
          scf.for %arg7 = %c0 to %c32 step %c1 {
            %40 = arith.muli %arg7, %c128 : index
            %41 = arith.addi %37, %40 : index
            %reinterpret_cast_5 = memref.reinterpret_cast %arg0 to offset: [%41], sizes: [1, 128, 128], strides: [524288, 4096, 1] : memref<32x128x4096xf32> to memref<1x128x128xf32, strided<[524288, 4096, 1], offset: ?>>
            loom.copy %reinterpret_cast_5, %19 src_mem_space @DRAM dst_mem_space @L1, interconnect : [@horizontal_links], broadcast : [1, 8] : memref<1x128x128xf32, strided<[524288, 4096, 1], offset: ?>>, memref<1x128x128xf32>
            linalg.fill ins(%cst_1 : f32) outs(%17 : memref<1x64x128xf32>)
            linalg.batch_matmul ins(%32, %19 : memref<1x64x128xf32>, memref<1x128x128xf32>) outs(%17 : memref<1x64x128xf32>)
            loom.semaphore_give %19 : memref<1x128x128xf32>
            linalg.fill ins(%cst_3 : f32) outs(%25 : memref<1x64xf32>)
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%17 : memref<1x64x128xf32>) outs(%25 : memref<1x64xf32>) {
            ^bb0(%in: f32, %out: f32):
              %44 = arith.maximumf %in, %out : f32
              linalg.yield %44 : f32
            }
            linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%23, %25 : memref<1x64xf32>, memref<1x64xf32>) outs(%25 : memref<1x64xf32>) {
            ^bb0(%in: f32, %in_7: f32, %out: f32):
              %44 = arith.truncf %cst_0 : f64 to f32
              %45 = arith.mulf %in_7, %44 : f32
              %46 = arith.cmpf ogt, %in, %45 : f32
              %47 = arith.select %46, %in, %45 : f32
              linalg.yield %47 : f32
            }
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%17, %25 : memref<1x64x128xf32>, memref<1x64xf32>) outs(%17 : memref<1x64x128xf32>) {
            ^bb0(%in: f32, %in_7: f32, %out: f32):
              %44 = arith.truncf %cst_0 : f64 to f32
              %45 = arith.mulf %in, %44 : f32
              %46 = arith.subf %45, %in_7 : f32
              %47 = math.powf %cst, %46 : f32
              linalg.yield %47 : f32
            }
            linalg.fill ins(%cst_1 : f32) outs(%27 : memref<1x64xf32>)
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%17 : memref<1x64x128xf32>) outs(%27 : memref<1x64xf32>) {
            ^bb0(%in: f32, %out: f32):
              %44 = arith.addf %in, %out : f32
              linalg.yield %44 : f32
            }
            linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%23, %25 : memref<1x64xf32>, memref<1x64xf32>) outs(%29 : memref<1x64xf32>) {
            ^bb0(%in: f32, %in_7: f32, %out: f32):
              %44 = arith.subf %in, %in_7 : f32
              %45 = math.powf %cst, %44 : f32
              linalg.yield %45 : f32
            }
            linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%21, %29, %27 : memref<1x64xf32>, memref<1x64xf32>, memref<1x64xf32>) outs(%21 : memref<1x64xf32>) {
            ^bb0(%in: f32, %in_7: f32, %in_8: f32, %out: f32):
              %44 = arith.mulf %in, %in_7 : f32
              %45 = arith.addf %44, %in_8 : f32
              linalg.yield %45 : f32
            }
            loom.semaphore_give %27 : memref<1x64xf32>
            %42 = arith.muli %arg7, %c16384 : index
            %43 = arith.addi %37, %42 : index
            %reinterpret_cast_6 = memref.reinterpret_cast %arg1 to offset: [%43], sizes: [1, 128, 128], strides: [524288, 128, 1] : memref<32x4096x128xf32> to memref<1x128x128xf32, strided<[524288, 128, 1], offset: ?>>
            loom.copy %reinterpret_cast_6, %15 src_mem_space @DRAM dst_mem_space @L1, interconnect : [@horizontal_links], broadcast : [1, 8] : memref<1x128x128xf32, strided<[524288, 128, 1], offset: ?>>, memref<1x128x128xf32>
            linalg.fill ins(%cst_1 : f32) outs(%36 : memref<1x64x128xf32>)
            linalg.batch_matmul ins(%17, %15 : memref<1x64x128xf32>, memref<1x128x128xf32>) outs(%36 : memref<1x64x128xf32>)
            loom.semaphore_give %15 : memref<1x128x128xf32>
            loom.semaphore_give %17 : memref<1x64x128xf32>
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%36, %34, %29 : memref<1x64x128xf32>, memref<1x64x128xf32>, memref<1x64xf32>) outs(%34 : memref<1x64x128xf32>) {
            ^bb0(%in: f32, %in_7: f32, %in_8: f32, %out: f32):
              %44 = arith.mulf %in_7, %in_8 : f32
              %45 = arith.addf %in, %44 : f32
              linalg.yield %45 : f32
            }
            loom.semaphore_give %29 : memref<1x64xf32>
            loom.semaphore_give %36 : memref<1x64x128xf32>
            linalg.copy ins(%25 : memref<1x64xf32>) outs(%23 : memref<1x64xf32>)
            loom.semaphore_give %25 : memref<1x64xf32>
          }
          loom.semaphore_give %23 : memref<1x64xf32>
          loom.semaphore_give %32 : memref<1x64x128xf32>
          linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%34, %21 : memref<1x64x128xf32>, memref<1x64xf32>) outs(%31 : memref<1x64x128xf32>) {
          ^bb0(%in: f32, %in_5: f32, %out: f32):
            %40 = arith.divf %in, %in_5 : f32
            linalg.yield %40 : f32
          }
          loom.semaphore_give %21 : memref<1x64xf32>
          loom.semaphore_give %34 : memref<1x64x128xf32>
          %reinterpret_cast_4 = memref.reinterpret_cast %arg3 to offset: [%39], sizes: [1, 64, 128], strides: [524288, 128, 1] : memref<32x4096x128xf32> to memref<1x64x128xf32, strided<[524288, 128, 1], offset: ?>>
          loom.copy %31, %reinterpret_cast_4 src_mem_space @L1 dst_mem_space @DRAM, interconnect : [], broadcast : [1, 1] : memref<1x64x128xf32>, memref<1x64x128xf32, strided<[524288, 128, 1], offset: ?>>
          loom.semaphore_give %31 : memref<1x64x128xf32>
        }
        scf.reduce 
      } {loom.iter_types = [#loom.iter_type<spatial>, #loom.iter_type<spatial>], loom.mapped_to_dims = [@x, @y]}
      return
    }
  }
  module attributes {loom.pass_name = "Materialize"} {
    func.func @attention__d0i1_d1i1__f01__d_h_v__BB1__BM64__BN128(%arg0: memref<32x128x4096xf32>, %arg1: memref<32x4096x128xf32>, %arg2: memref<32x4096x128xf32>, %arg3: memref<32x4096x128xf32>) {
      %c16384 = arith.constant 16384 : index
      %c8192 = arith.constant 8192 : index
      %c524288 = arith.constant 524288 : index
      %c32 = arith.constant 32 : index
      %cst = arith.constant 2.000000e+00 : f32
      %cst_0 = arith.constant 0.12751743074602467 : f64
      %cst_1 = arith.constant 0.000000e+00 : f32
      %cst_2 = arith.constant 1.000000e+00 : f32
      %cst_3 = arith.constant 0xFF800000 : f32
      %c128 = arith.constant 128 : index
      %c0 = arith.constant 0 : index
      %c8 = arith.constant 8 : index
      %c1 = arith.constant 1 : index
      scf.parallel (%arg4, %arg5) = (%c0, %c0) to (%c8, %c8) step (%c1, %c1) {
        scf.for %arg6 = %c0 to %c32 step %c1 {
          %12 = arith.muli %arg4, %c8 overflow<nsw> : index
          %13 = arith.addi %12, %arg5 : index
          %14 = loom.alloc [1, 128, 128] on @L1 : memref<1x128x128xf32>
          %15 = loom.semaphore_take %14 : memref<1x128x128xf32> -> memref<1x128x128xf32>
          %16 = loom.alloc [1, 64, 128] on @L1 : memref<1x64x128xf32>
          %17 = loom.semaphore_take %16 : memref<1x64x128xf32> -> memref<1x64x128xf32>
          %18 = loom.alloc [1, 128, 128] on @L1 : memref<1x128x128xf32>
          %19 = loom.semaphore_take %18 : memref<1x128x128xf32> -> memref<1x128x128xf32>
          %20 = loom.alloc [1, 64] on @L1 : memref<1x64xf32>
          %21 = loom.semaphore_take %20 : memref<1x64xf32> -> memref<1x64xf32>
          %22 = loom.alloc [1, 64] on @L1 : memref<1x64xf32>
          %23 = loom.semaphore_take %22 : memref<1x64xf32> -> memref<1x64xf32>
          %24 = loom.alloc [1, 64] on @L1 : memref<1x64xf32>
          %25 = loom.semaphore_take %24 : memref<1x64xf32> -> memref<1x64xf32>
          %26 = loom.alloc [1, 64] on @L1 : memref<1x64xf32>
          %27 = loom.semaphore_take %26 : memref<1x64xf32> -> memref<1x64xf32>
          %28 = loom.alloc [1, 64] on @L1 : memref<1x64xf32>
          %29 = loom.semaphore_take %28 : memref<1x64xf32> -> memref<1x64xf32>
          %30 = loom.alloc [1, 64, 128] on @L1 : memref<1x64x128xf32>
          %31 = loom.semaphore_take %30 : memref<1x64x128xf32> -> memref<1x64x128xf32>
          %32 = loom.semaphore_take %30 : memref<1x64x128xf32> -> memref<1x64x128xf32>
          %33 = loom.alloc [1, 64, 128] on @L1 : memref<1x64x128xf32>
          %34 = loom.semaphore_take %33 : memref<1x64x128xf32> -> memref<1x64x128xf32>
          %35 = loom.alloc [1, 64, 128] on @L1 : memref<1x64x128xf32>
          %36 = loom.semaphore_take %35 : memref<1x64x128xf32> -> memref<1x64x128xf32>
          %37 = arith.muli %arg6, %c524288 overflow<nsw> : index
          %38 = arith.muli %13, %c8192 : index
          %39 = arith.addi %37, %38 : index
          %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%39], sizes: [1, 64, 128], strides: [524288, 128, 1] : memref<32x4096x128xf32> to memref<1x64x128xf32, strided<[524288, 128, 1], offset: ?>>
          loom.copy %reinterpret_cast, %32 src_mem_space @DRAM dst_mem_space @L1, interconnect : [], broadcast : [1, 1] : memref<1x64x128xf32, strided<[524288, 128, 1], offset: ?>>, memref<1x64x128xf32>
          linalg.fill ins(%cst_1 : f32) outs(%34 : memref<1x64x128xf32>)
          linalg.fill ins(%cst_2 : f32) outs(%21 : memref<1x64xf32>)
          linalg.fill ins(%cst_3 : f32) outs(%23 : memref<1x64xf32>)
          scf.for %arg7 = %c0 to %c32 step %c1 {
            %40 = arith.muli %arg7, %c128 : index
            %41 = arith.addi %37, %40 : index
            %reinterpret_cast_5 = memref.reinterpret_cast %arg0 to offset: [%41], sizes: [1, 128, 128], strides: [524288, 4096, 1] : memref<32x128x4096xf32> to memref<1x128x128xf32, strided<[524288, 4096, 1], offset: ?>>
            loom.copy %reinterpret_cast_5, %19 src_mem_space @DRAM dst_mem_space @L1, interconnect : [@horizontal_links], broadcast : [1, 8] : memref<1x128x128xf32, strided<[524288, 4096, 1], offset: ?>>, memref<1x128x128xf32>
            linalg.fill ins(%cst_1 : f32) outs(%17 : memref<1x64x128xf32>)
            linalg.batch_matmul ins(%32, %19 : memref<1x64x128xf32>, memref<1x128x128xf32>) outs(%17 : memref<1x64x128xf32>)
            loom.semaphore_give %19 : memref<1x128x128xf32>
            linalg.fill ins(%cst_3 : f32) outs(%25 : memref<1x64xf32>)
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%17 : memref<1x64x128xf32>) outs(%25 : memref<1x64xf32>) {
            ^bb0(%in: f32, %out: f32):
              %44 = arith.maximumf %in, %out : f32
              linalg.yield %44 : f32
            }
            linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%23, %25 : memref<1x64xf32>, memref<1x64xf32>) outs(%25 : memref<1x64xf32>) {
            ^bb0(%in: f32, %in_7: f32, %out: f32):
              %44 = arith.truncf %cst_0 : f64 to f32
              %45 = arith.mulf %in_7, %44 : f32
              %46 = arith.cmpf ogt, %in, %45 : f32
              %47 = arith.select %46, %in, %45 : f32
              linalg.yield %47 : f32
            }
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%17, %25 : memref<1x64x128xf32>, memref<1x64xf32>) outs(%17 : memref<1x64x128xf32>) {
            ^bb0(%in: f32, %in_7: f32, %out: f32):
              %44 = arith.truncf %cst_0 : f64 to f32
              %45 = arith.mulf %in, %44 : f32
              %46 = arith.subf %45, %in_7 : f32
              %47 = math.powf %cst, %46 : f32
              linalg.yield %47 : f32
            }
            linalg.fill ins(%cst_1 : f32) outs(%27 : memref<1x64xf32>)
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%17 : memref<1x64x128xf32>) outs(%27 : memref<1x64xf32>) {
            ^bb0(%in: f32, %out: f32):
              %44 = arith.addf %in, %out : f32
              linalg.yield %44 : f32
            }
            linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%23, %25 : memref<1x64xf32>, memref<1x64xf32>) outs(%29 : memref<1x64xf32>) {
            ^bb0(%in: f32, %in_7: f32, %out: f32):
              %44 = arith.subf %in, %in_7 : f32
              %45 = math.powf %cst, %44 : f32
              linalg.yield %45 : f32
            }
            linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%21, %29, %27 : memref<1x64xf32>, memref<1x64xf32>, memref<1x64xf32>) outs(%21 : memref<1x64xf32>) {
            ^bb0(%in: f32, %in_7: f32, %in_8: f32, %out: f32):
              %44 = arith.mulf %in, %in_7 : f32
              %45 = arith.addf %44, %in_8 : f32
              linalg.yield %45 : f32
            }
            loom.semaphore_give %27 : memref<1x64xf32>
            %42 = arith.muli %arg7, %c16384 : index
            %43 = arith.addi %37, %42 : index
            %reinterpret_cast_6 = memref.reinterpret_cast %arg1 to offset: [%43], sizes: [1, 128, 128], strides: [524288, 128, 1] : memref<32x4096x128xf32> to memref<1x128x128xf32, strided<[524288, 128, 1], offset: ?>>
            loom.copy %reinterpret_cast_6, %15 src_mem_space @DRAM dst_mem_space @L1, interconnect : [@vertical_links], broadcast : [8, 1] : memref<1x128x128xf32, strided<[524288, 128, 1], offset: ?>>, memref<1x128x128xf32>
            linalg.fill ins(%cst_1 : f32) outs(%36 : memref<1x64x128xf32>)
            linalg.batch_matmul ins(%17, %15 : memref<1x64x128xf32>, memref<1x128x128xf32>) outs(%36 : memref<1x64x128xf32>)
            loom.semaphore_give %15 : memref<1x128x128xf32>
            loom.semaphore_give %17 : memref<1x64x128xf32>
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%36, %34, %29 : memref<1x64x128xf32>, memref<1x64x128xf32>, memref<1x64xf32>) outs(%34 : memref<1x64x128xf32>) {
            ^bb0(%in: f32, %in_7: f32, %in_8: f32, %out: f32):
              %44 = arith.mulf %in_7, %in_8 : f32
              %45 = arith.addf %in, %44 : f32
              linalg.yield %45 : f32
            }
            loom.semaphore_give %29 : memref<1x64xf32>
            loom.semaphore_give %36 : memref<1x64x128xf32>
            linalg.copy ins(%25 : memref<1x64xf32>) outs(%23 : memref<1x64xf32>)
            loom.semaphore_give %25 : memref<1x64xf32>
          }
          loom.semaphore_give %23 : memref<1x64xf32>
          loom.semaphore_give %32 : memref<1x64x128xf32>
          linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%34, %21 : memref<1x64x128xf32>, memref<1x64xf32>) outs(%31 : memref<1x64x128xf32>) {
          ^bb0(%in: f32, %in_5: f32, %out: f32):
            %40 = arith.divf %in, %in_5 : f32
            linalg.yield %40 : f32
          }
          loom.semaphore_give %21 : memref<1x64xf32>
          loom.semaphore_give %34 : memref<1x64x128xf32>
          %reinterpret_cast_4 = memref.reinterpret_cast %arg3 to offset: [%39], sizes: [1, 64, 128], strides: [524288, 128, 1] : memref<32x4096x128xf32> to memref<1x64x128xf32, strided<[524288, 128, 1], offset: ?>>
          loom.copy %31, %reinterpret_cast_4 src_mem_space @L1 dst_mem_space @DRAM, interconnect : [], broadcast : [1, 1] : memref<1x64x128xf32>, memref<1x64x128xf32, strided<[524288, 128, 1], offset: ?>>
          loom.semaphore_give %31 : memref<1x64x128xf32>
        }
        scf.reduce 
      } {loom.iter_types = [#loom.iter_type<spatial>, #loom.iter_type<spatial>], loom.mapped_to_dims = [@x, @y]}
      return
    }
  }
  module attributes {loom.pass_name = "Materialize"} {
    func.func @attention__d0i1_d1i1__f01__d_v_d__BB1__BM64__BN128(%arg0: memref<32x128x4096xf32>, %arg1: memref<32x4096x128xf32>, %arg2: memref<32x4096x128xf32>, %arg3: memref<32x4096x128xf32>) {
      %c16384 = arith.constant 16384 : index
      %c8192 = arith.constant 8192 : index
      %c524288 = arith.constant 524288 : index
      %c32 = arith.constant 32 : index
      %cst = arith.constant 2.000000e+00 : f32
      %cst_0 = arith.constant 0.12751743074602467 : f64
      %cst_1 = arith.constant 0.000000e+00 : f32
      %cst_2 = arith.constant 1.000000e+00 : f32
      %cst_3 = arith.constant 0xFF800000 : f32
      %c128 = arith.constant 128 : index
      %c0 = arith.constant 0 : index
      %c8 = arith.constant 8 : index
      %c1 = arith.constant 1 : index
      scf.parallel (%arg4, %arg5) = (%c0, %c0) to (%c8, %c8) step (%c1, %c1) {
        scf.for %arg6 = %c0 to %c32 step %c1 {
          %12 = arith.muli %arg4, %c8 overflow<nsw> : index
          %13 = arith.addi %12, %arg5 : index
          %14 = loom.alloc [1, 128, 128] on @L1 : memref<1x128x128xf32>
          %15 = loom.semaphore_take %14 : memref<1x128x128xf32> -> memref<1x128x128xf32>
          %16 = loom.alloc [1, 64, 128] on @L1 : memref<1x64x128xf32>
          %17 = loom.semaphore_take %16 : memref<1x64x128xf32> -> memref<1x64x128xf32>
          %18 = loom.alloc [1, 128, 128] on @L1 : memref<1x128x128xf32>
          %19 = loom.semaphore_take %18 : memref<1x128x128xf32> -> memref<1x128x128xf32>
          %20 = loom.alloc [1, 64] on @L1 : memref<1x64xf32>
          %21 = loom.semaphore_take %20 : memref<1x64xf32> -> memref<1x64xf32>
          %22 = loom.alloc [1, 64] on @L1 : memref<1x64xf32>
          %23 = loom.semaphore_take %22 : memref<1x64xf32> -> memref<1x64xf32>
          %24 = loom.alloc [1, 64] on @L1 : memref<1x64xf32>
          %25 = loom.semaphore_take %24 : memref<1x64xf32> -> memref<1x64xf32>
          %26 = loom.alloc [1, 64] on @L1 : memref<1x64xf32>
          %27 = loom.semaphore_take %26 : memref<1x64xf32> -> memref<1x64xf32>
          %28 = loom.alloc [1, 64] on @L1 : memref<1x64xf32>
          %29 = loom.semaphore_take %28 : memref<1x64xf32> -> memref<1x64xf32>
          %30 = loom.alloc [1, 64, 128] on @L1 : memref<1x64x128xf32>
          %31 = loom.semaphore_take %30 : memref<1x64x128xf32> -> memref<1x64x128xf32>
          %32 = loom.semaphore_take %30 : memref<1x64x128xf32> -> memref<1x64x128xf32>
          %33 = loom.alloc [1, 64, 128] on @L1 : memref<1x64x128xf32>
          %34 = loom.semaphore_take %33 : memref<1x64x128xf32> -> memref<1x64x128xf32>
          %35 = loom.alloc [1, 64, 128] on @L1 : memref<1x64x128xf32>
          %36 = loom.semaphore_take %35 : memref<1x64x128xf32> -> memref<1x64x128xf32>
          %37 = arith.muli %arg6, %c524288 overflow<nsw> : index
          %38 = arith.muli %13, %c8192 : index
          %39 = arith.addi %37, %38 : index
          %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%39], sizes: [1, 64, 128], strides: [524288, 128, 1] : memref<32x4096x128xf32> to memref<1x64x128xf32, strided<[524288, 128, 1], offset: ?>>
          loom.copy %reinterpret_cast, %32 src_mem_space @DRAM dst_mem_space @L1, interconnect : [], broadcast : [1, 1] : memref<1x64x128xf32, strided<[524288, 128, 1], offset: ?>>, memref<1x64x128xf32>
          linalg.fill ins(%cst_1 : f32) outs(%34 : memref<1x64x128xf32>)
          linalg.fill ins(%cst_2 : f32) outs(%21 : memref<1x64xf32>)
          linalg.fill ins(%cst_3 : f32) outs(%23 : memref<1x64xf32>)
          scf.for %arg7 = %c0 to %c32 step %c1 {
            %40 = arith.muli %arg7, %c128 : index
            %41 = arith.addi %37, %40 : index
            %reinterpret_cast_5 = memref.reinterpret_cast %arg0 to offset: [%41], sizes: [1, 128, 128], strides: [524288, 4096, 1] : memref<32x128x4096xf32> to memref<1x128x128xf32, strided<[524288, 4096, 1], offset: ?>>
            loom.copy %reinterpret_cast_5, %19 src_mem_space @DRAM dst_mem_space @L1, interconnect : [@vertical_links], broadcast : [8, 1] : memref<1x128x128xf32, strided<[524288, 4096, 1], offset: ?>>, memref<1x128x128xf32>
            linalg.fill ins(%cst_1 : f32) outs(%17 : memref<1x64x128xf32>)
            linalg.batch_matmul ins(%32, %19 : memref<1x64x128xf32>, memref<1x128x128xf32>) outs(%17 : memref<1x64x128xf32>)
            loom.semaphore_give %19 : memref<1x128x128xf32>
            linalg.fill ins(%cst_3 : f32) outs(%25 : memref<1x64xf32>)
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%17 : memref<1x64x128xf32>) outs(%25 : memref<1x64xf32>) {
            ^bb0(%in: f32, %out: f32):
              %44 = arith.maximumf %in, %out : f32
              linalg.yield %44 : f32
            }
            linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%23, %25 : memref<1x64xf32>, memref<1x64xf32>) outs(%25 : memref<1x64xf32>) {
            ^bb0(%in: f32, %in_7: f32, %out: f32):
              %44 = arith.truncf %cst_0 : f64 to f32
              %45 = arith.mulf %in_7, %44 : f32
              %46 = arith.cmpf ogt, %in, %45 : f32
              %47 = arith.select %46, %in, %45 : f32
              linalg.yield %47 : f32
            }
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%17, %25 : memref<1x64x128xf32>, memref<1x64xf32>) outs(%17 : memref<1x64x128xf32>) {
            ^bb0(%in: f32, %in_7: f32, %out: f32):
              %44 = arith.truncf %cst_0 : f64 to f32
              %45 = arith.mulf %in, %44 : f32
              %46 = arith.subf %45, %in_7 : f32
              %47 = math.powf %cst, %46 : f32
              linalg.yield %47 : f32
            }
            linalg.fill ins(%cst_1 : f32) outs(%27 : memref<1x64xf32>)
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%17 : memref<1x64x128xf32>) outs(%27 : memref<1x64xf32>) {
            ^bb0(%in: f32, %out: f32):
              %44 = arith.addf %in, %out : f32
              linalg.yield %44 : f32
            }
            linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%23, %25 : memref<1x64xf32>, memref<1x64xf32>) outs(%29 : memref<1x64xf32>) {
            ^bb0(%in: f32, %in_7: f32, %out: f32):
              %44 = arith.subf %in, %in_7 : f32
              %45 = math.powf %cst, %44 : f32
              linalg.yield %45 : f32
            }
            linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%21, %29, %27 : memref<1x64xf32>, memref<1x64xf32>, memref<1x64xf32>) outs(%21 : memref<1x64xf32>) {
            ^bb0(%in: f32, %in_7: f32, %in_8: f32, %out: f32):
              %44 = arith.mulf %in, %in_7 : f32
              %45 = arith.addf %44, %in_8 : f32
              linalg.yield %45 : f32
            }
            loom.semaphore_give %27 : memref<1x64xf32>
            %42 = arith.muli %arg7, %c16384 : index
            %43 = arith.addi %37, %42 : index
            %reinterpret_cast_6 = memref.reinterpret_cast %arg1 to offset: [%43], sizes: [1, 128, 128], strides: [524288, 128, 1] : memref<32x4096x128xf32> to memref<1x128x128xf32, strided<[524288, 128, 1], offset: ?>>
            loom.copy %reinterpret_cast_6, %15 src_mem_space @DRAM dst_mem_space @L1, interconnect : [], broadcast : [1, 1] : memref<1x128x128xf32, strided<[524288, 128, 1], offset: ?>>, memref<1x128x128xf32>
            linalg.fill ins(%cst_1 : f32) outs(%36 : memref<1x64x128xf32>)
            linalg.batch_matmul ins(%17, %15 : memref<1x64x128xf32>, memref<1x128x128xf32>) outs(%36 : memref<1x64x128xf32>)
            loom.semaphore_give %15 : memref<1x128x128xf32>
            loom.semaphore_give %17 : memref<1x64x128xf32>
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%36, %34, %29 : memref<1x64x128xf32>, memref<1x64x128xf32>, memref<1x64xf32>) outs(%34 : memref<1x64x128xf32>) {
            ^bb0(%in: f32, %in_7: f32, %in_8: f32, %out: f32):
              %44 = arith.mulf %in_7, %in_8 : f32
              %45 = arith.addf %in, %44 : f32
              linalg.yield %45 : f32
            }
            loom.semaphore_give %29 : memref<1x64xf32>
            loom.semaphore_give %36 : memref<1x64x128xf32>
            linalg.copy ins(%25 : memref<1x64xf32>) outs(%23 : memref<1x64xf32>)
            loom.semaphore_give %25 : memref<1x64xf32>
          }
          loom.semaphore_give %23 : memref<1x64xf32>
          loom.semaphore_give %32 : memref<1x64x128xf32>
          linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%34, %21 : memref<1x64x128xf32>, memref<1x64xf32>) outs(%31 : memref<1x64x128xf32>) {
          ^bb0(%in: f32, %in_5: f32, %out: f32):
            %40 = arith.divf %in, %in_5 : f32
            linalg.yield %40 : f32
          }
          loom.semaphore_give %21 : memref<1x64xf32>
          loom.semaphore_give %34 : memref<1x64x128xf32>
          %reinterpret_cast_4 = memref.reinterpret_cast %arg3 to offset: [%39], sizes: [1, 64, 128], strides: [524288, 128, 1] : memref<32x4096x128xf32> to memref<1x64x128xf32, strided<[524288, 128, 1], offset: ?>>
          loom.copy %31, %reinterpret_cast_4 src_mem_space @L1 dst_mem_space @DRAM, interconnect : [], broadcast : [1, 1] : memref<1x64x128xf32>, memref<1x64x128xf32, strided<[524288, 128, 1], offset: ?>>
          loom.semaphore_give %31 : memref<1x64x128xf32>
        }
        scf.reduce 
      } {loom.iter_types = [#loom.iter_type<spatial>, #loom.iter_type<spatial>], loom.mapped_to_dims = [@x, @y]}
      return
    }
  }
  module attributes {loom.pass_name = "Materialize"} {
    func.func @attention__d0i1_d1i1__f01__d_v_a__BB1__BM64__BN128(%arg0: memref<32x128x4096xf32>, %arg1: memref<32x4096x128xf32>, %arg2: memref<32x4096x128xf32>, %arg3: memref<32x4096x128xf32>) {
      %c16384 = arith.constant 16384 : index
      %c8192 = arith.constant 8192 : index
      %c524288 = arith.constant 524288 : index
      %c32 = arith.constant 32 : index
      %cst = arith.constant 2.000000e+00 : f32
      %cst_0 = arith.constant 0.12751743074602467 : f64
      %cst_1 = arith.constant 0.000000e+00 : f32
      %cst_2 = arith.constant 1.000000e+00 : f32
      %cst_3 = arith.constant 0xFF800000 : f32
      %c128 = arith.constant 128 : index
      %c0 = arith.constant 0 : index
      %c8 = arith.constant 8 : index
      %c1 = arith.constant 1 : index
      scf.parallel (%arg4, %arg5) = (%c0, %c0) to (%c8, %c8) step (%c1, %c1) {
        scf.for %arg6 = %c0 to %c32 step %c1 {
          %12 = arith.muli %arg4, %c8 overflow<nsw> : index
          %13 = arith.addi %12, %arg5 : index
          %14 = loom.alloc [1, 128, 128] on @L1 : memref<1x128x128xf32>
          %15 = loom.semaphore_take %14 : memref<1x128x128xf32> -> memref<1x128x128xf32>
          %16 = loom.alloc [1, 64, 128] on @L1 : memref<1x64x128xf32>
          %17 = loom.semaphore_take %16 : memref<1x64x128xf32> -> memref<1x64x128xf32>
          %18 = loom.alloc [1, 128, 128] on @L1 : memref<1x128x128xf32>
          %19 = loom.semaphore_take %18 : memref<1x128x128xf32> -> memref<1x128x128xf32>
          %20 = loom.alloc [1, 64] on @L1 : memref<1x64xf32>
          %21 = loom.semaphore_take %20 : memref<1x64xf32> -> memref<1x64xf32>
          %22 = loom.alloc [1, 64] on @L1 : memref<1x64xf32>
          %23 = loom.semaphore_take %22 : memref<1x64xf32> -> memref<1x64xf32>
          %24 = loom.alloc [1, 64] on @L1 : memref<1x64xf32>
          %25 = loom.semaphore_take %24 : memref<1x64xf32> -> memref<1x64xf32>
          %26 = loom.alloc [1, 64] on @L1 : memref<1x64xf32>
          %27 = loom.semaphore_take %26 : memref<1x64xf32> -> memref<1x64xf32>
          %28 = loom.alloc [1, 64] on @L1 : memref<1x64xf32>
          %29 = loom.semaphore_take %28 : memref<1x64xf32> -> memref<1x64xf32>
          %30 = loom.alloc [1, 64, 128] on @L1 : memref<1x64x128xf32>
          %31 = loom.semaphore_take %30 : memref<1x64x128xf32> -> memref<1x64x128xf32>
          %32 = loom.semaphore_take %30 : memref<1x64x128xf32> -> memref<1x64x128xf32>
          %33 = loom.alloc [1, 64, 128] on @L1 : memref<1x64x128xf32>
          %34 = loom.semaphore_take %33 : memref<1x64x128xf32> -> memref<1x64x128xf32>
          %35 = loom.alloc [1, 64, 128] on @L1 : memref<1x64x128xf32>
          %36 = loom.semaphore_take %35 : memref<1x64x128xf32> -> memref<1x64x128xf32>
          %37 = arith.muli %arg6, %c524288 overflow<nsw> : index
          %38 = arith.muli %13, %c8192 : index
          %39 = arith.addi %37, %38 : index
          %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%39], sizes: [1, 64, 128], strides: [524288, 128, 1] : memref<32x4096x128xf32> to memref<1x64x128xf32, strided<[524288, 128, 1], offset: ?>>
          loom.copy %reinterpret_cast, %32 src_mem_space @DRAM dst_mem_space @L1, interconnect : [], broadcast : [1, 1] : memref<1x64x128xf32, strided<[524288, 128, 1], offset: ?>>, memref<1x64x128xf32>
          linalg.fill ins(%cst_1 : f32) outs(%34 : memref<1x64x128xf32>)
          linalg.fill ins(%cst_2 : f32) outs(%21 : memref<1x64xf32>)
          linalg.fill ins(%cst_3 : f32) outs(%23 : memref<1x64xf32>)
          scf.for %arg7 = %c0 to %c32 step %c1 {
            %40 = arith.muli %arg7, %c128 : index
            %41 = arith.addi %37, %40 : index
            %reinterpret_cast_5 = memref.reinterpret_cast %arg0 to offset: [%41], sizes: [1, 128, 128], strides: [524288, 4096, 1] : memref<32x128x4096xf32> to memref<1x128x128xf32, strided<[524288, 4096, 1], offset: ?>>
            loom.copy %reinterpret_cast_5, %19 src_mem_space @DRAM dst_mem_space @L1, interconnect : [@vertical_links], broadcast : [8, 1] : memref<1x128x128xf32, strided<[524288, 4096, 1], offset: ?>>, memref<1x128x128xf32>
            linalg.fill ins(%cst_1 : f32) outs(%17 : memref<1x64x128xf32>)
            linalg.batch_matmul ins(%32, %19 : memref<1x64x128xf32>, memref<1x128x128xf32>) outs(%17 : memref<1x64x128xf32>)
            loom.semaphore_give %19 : memref<1x128x128xf32>
            linalg.fill ins(%cst_3 : f32) outs(%25 : memref<1x64xf32>)
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%17 : memref<1x64x128xf32>) outs(%25 : memref<1x64xf32>) {
            ^bb0(%in: f32, %out: f32):
              %44 = arith.maximumf %in, %out : f32
              linalg.yield %44 : f32
            }
            linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%23, %25 : memref<1x64xf32>, memref<1x64xf32>) outs(%25 : memref<1x64xf32>) {
            ^bb0(%in: f32, %in_7: f32, %out: f32):
              %44 = arith.truncf %cst_0 : f64 to f32
              %45 = arith.mulf %in_7, %44 : f32
              %46 = arith.cmpf ogt, %in, %45 : f32
              %47 = arith.select %46, %in, %45 : f32
              linalg.yield %47 : f32
            }
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%17, %25 : memref<1x64x128xf32>, memref<1x64xf32>) outs(%17 : memref<1x64x128xf32>) {
            ^bb0(%in: f32, %in_7: f32, %out: f32):
              %44 = arith.truncf %cst_0 : f64 to f32
              %45 = arith.mulf %in, %44 : f32
              %46 = arith.subf %45, %in_7 : f32
              %47 = math.powf %cst, %46 : f32
              linalg.yield %47 : f32
            }
            linalg.fill ins(%cst_1 : f32) outs(%27 : memref<1x64xf32>)
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%17 : memref<1x64x128xf32>) outs(%27 : memref<1x64xf32>) {
            ^bb0(%in: f32, %out: f32):
              %44 = arith.addf %in, %out : f32
              linalg.yield %44 : f32
            }
            linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%23, %25 : memref<1x64xf32>, memref<1x64xf32>) outs(%29 : memref<1x64xf32>) {
            ^bb0(%in: f32, %in_7: f32, %out: f32):
              %44 = arith.subf %in, %in_7 : f32
              %45 = math.powf %cst, %44 : f32
              linalg.yield %45 : f32
            }
            linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%21, %29, %27 : memref<1x64xf32>, memref<1x64xf32>, memref<1x64xf32>) outs(%21 : memref<1x64xf32>) {
            ^bb0(%in: f32, %in_7: f32, %in_8: f32, %out: f32):
              %44 = arith.mulf %in, %in_7 : f32
              %45 = arith.addf %44, %in_8 : f32
              linalg.yield %45 : f32
            }
            loom.semaphore_give %27 : memref<1x64xf32>
            %42 = arith.muli %arg7, %c16384 : index
            %43 = arith.addi %37, %42 : index
            %reinterpret_cast_6 = memref.reinterpret_cast %arg1 to offset: [%43], sizes: [1, 128, 128], strides: [524288, 128, 1] : memref<32x4096x128xf32> to memref<1x128x128xf32, strided<[524288, 128, 1], offset: ?>>
            loom.copy %reinterpret_cast_6, %15 src_mem_space @DRAM dst_mem_space @L1, interconnect : [@horizontal_links, @vertical_links], broadcast : [8, 8] : memref<1x128x128xf32, strided<[524288, 128, 1], offset: ?>>, memref<1x128x128xf32>
            linalg.fill ins(%cst_1 : f32) outs(%36 : memref<1x64x128xf32>)
            linalg.batch_matmul ins(%17, %15 : memref<1x64x128xf32>, memref<1x128x128xf32>) outs(%36 : memref<1x64x128xf32>)
            loom.semaphore_give %15 : memref<1x128x128xf32>
            loom.semaphore_give %17 : memref<1x64x128xf32>
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%36, %34, %29 : memref<1x64x128xf32>, memref<1x64x128xf32>, memref<1x64xf32>) outs(%34 : memref<1x64x128xf32>) {
            ^bb0(%in: f32, %in_7: f32, %in_8: f32, %out: f32):
              %44 = arith.mulf %in_7, %in_8 : f32
              %45 = arith.addf %in, %44 : f32
              linalg.yield %45 : f32
            }
            loom.semaphore_give %29 : memref<1x64xf32>
            loom.semaphore_give %36 : memref<1x64x128xf32>
            linalg.copy ins(%25 : memref<1x64xf32>) outs(%23 : memref<1x64xf32>)
            loom.semaphore_give %25 : memref<1x64xf32>
          }
          loom.semaphore_give %23 : memref<1x64xf32>
          loom.semaphore_give %32 : memref<1x64x128xf32>
          linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%34, %21 : memref<1x64x128xf32>, memref<1x64xf32>) outs(%31 : memref<1x64x128xf32>) {
          ^bb0(%in: f32, %in_5: f32, %out: f32):
            %40 = arith.divf %in, %in_5 : f32
            linalg.yield %40 : f32
          }
          loom.semaphore_give %21 : memref<1x64xf32>
          loom.semaphore_give %34 : memref<1x64x128xf32>
          %reinterpret_cast_4 = memref.reinterpret_cast %arg3 to offset: [%39], sizes: [1, 64, 128], strides: [524288, 128, 1] : memref<32x4096x128xf32> to memref<1x64x128xf32, strided<[524288, 128, 1], offset: ?>>
          loom.copy %31, %reinterpret_cast_4 src_mem_space @L1 dst_mem_space @DRAM, interconnect : [], broadcast : [1, 1] : memref<1x64x128xf32>, memref<1x64x128xf32, strided<[524288, 128, 1], offset: ?>>
          loom.semaphore_give %31 : memref<1x64x128xf32>
        }
        scf.reduce 
      } {loom.iter_types = [#loom.iter_type<spatial>, #loom.iter_type<spatial>], loom.mapped_to_dims = [@x, @y]}
      return
    }
  }
  module attributes {loom.pass_name = "Materialize"} {
    func.func @attention__d0i1_d1i1__f01__d_v_h__BB1__BM64__BN128(%arg0: memref<32x128x4096xf32>, %arg1: memref<32x4096x128xf32>, %arg2: memref<32x4096x128xf32>, %arg3: memref<32x4096x128xf32>) {
      %c16384 = arith.constant 16384 : index
      %c8192 = arith.constant 8192 : index
      %c524288 = arith.constant 524288 : index
      %c32 = arith.constant 32 : index
      %cst = arith.constant 2.000000e+00 : f32
      %cst_0 = arith.constant 0.12751743074602467 : f64
      %cst_1 = arith.constant 0.000000e+00 : f32
      %cst_2 = arith.constant 1.000000e+00 : f32
      %cst_3 = arith.constant 0xFF800000 : f32
      %c128 = arith.constant 128 : index
      %c0 = arith.constant 0 : index
      %c8 = arith.constant 8 : index
      %c1 = arith.constant 1 : index
      scf.parallel (%arg4, %arg5) = (%c0, %c0) to (%c8, %c8) step (%c1, %c1) {
        scf.for %arg6 = %c0 to %c32 step %c1 {
          %12 = arith.muli %arg4, %c8 overflow<nsw> : index
          %13 = arith.addi %12, %arg5 : index
          %14 = loom.alloc [1, 128, 128] on @L1 : memref<1x128x128xf32>
          %15 = loom.semaphore_take %14 : memref<1x128x128xf32> -> memref<1x128x128xf32>
          %16 = loom.alloc [1, 64, 128] on @L1 : memref<1x64x128xf32>
          %17 = loom.semaphore_take %16 : memref<1x64x128xf32> -> memref<1x64x128xf32>
          %18 = loom.alloc [1, 128, 128] on @L1 : memref<1x128x128xf32>
          %19 = loom.semaphore_take %18 : memref<1x128x128xf32> -> memref<1x128x128xf32>
          %20 = loom.alloc [1, 64] on @L1 : memref<1x64xf32>
          %21 = loom.semaphore_take %20 : memref<1x64xf32> -> memref<1x64xf32>
          %22 = loom.alloc [1, 64] on @L1 : memref<1x64xf32>
          %23 = loom.semaphore_take %22 : memref<1x64xf32> -> memref<1x64xf32>
          %24 = loom.alloc [1, 64] on @L1 : memref<1x64xf32>
          %25 = loom.semaphore_take %24 : memref<1x64xf32> -> memref<1x64xf32>
          %26 = loom.alloc [1, 64] on @L1 : memref<1x64xf32>
          %27 = loom.semaphore_take %26 : memref<1x64xf32> -> memref<1x64xf32>
          %28 = loom.alloc [1, 64] on @L1 : memref<1x64xf32>
          %29 = loom.semaphore_take %28 : memref<1x64xf32> -> memref<1x64xf32>
          %30 = loom.alloc [1, 64, 128] on @L1 : memref<1x64x128xf32>
          %31 = loom.semaphore_take %30 : memref<1x64x128xf32> -> memref<1x64x128xf32>
          %32 = loom.semaphore_take %30 : memref<1x64x128xf32> -> memref<1x64x128xf32>
          %33 = loom.alloc [1, 64, 128] on @L1 : memref<1x64x128xf32>
          %34 = loom.semaphore_take %33 : memref<1x64x128xf32> -> memref<1x64x128xf32>
          %35 = loom.alloc [1, 64, 128] on @L1 : memref<1x64x128xf32>
          %36 = loom.semaphore_take %35 : memref<1x64x128xf32> -> memref<1x64x128xf32>
          %37 = arith.muli %arg6, %c524288 overflow<nsw> : index
          %38 = arith.muli %13, %c8192 : index
          %39 = arith.addi %37, %38 : index
          %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%39], sizes: [1, 64, 128], strides: [524288, 128, 1] : memref<32x4096x128xf32> to memref<1x64x128xf32, strided<[524288, 128, 1], offset: ?>>
          loom.copy %reinterpret_cast, %32 src_mem_space @DRAM dst_mem_space @L1, interconnect : [], broadcast : [1, 1] : memref<1x64x128xf32, strided<[524288, 128, 1], offset: ?>>, memref<1x64x128xf32>
          linalg.fill ins(%cst_1 : f32) outs(%34 : memref<1x64x128xf32>)
          linalg.fill ins(%cst_2 : f32) outs(%21 : memref<1x64xf32>)
          linalg.fill ins(%cst_3 : f32) outs(%23 : memref<1x64xf32>)
          scf.for %arg7 = %c0 to %c32 step %c1 {
            %40 = arith.muli %arg7, %c128 : index
            %41 = arith.addi %37, %40 : index
            %reinterpret_cast_5 = memref.reinterpret_cast %arg0 to offset: [%41], sizes: [1, 128, 128], strides: [524288, 4096, 1] : memref<32x128x4096xf32> to memref<1x128x128xf32, strided<[524288, 4096, 1], offset: ?>>
            loom.copy %reinterpret_cast_5, %19 src_mem_space @DRAM dst_mem_space @L1, interconnect : [@vertical_links], broadcast : [8, 1] : memref<1x128x128xf32, strided<[524288, 4096, 1], offset: ?>>, memref<1x128x128xf32>
            linalg.fill ins(%cst_1 : f32) outs(%17 : memref<1x64x128xf32>)
            linalg.batch_matmul ins(%32, %19 : memref<1x64x128xf32>, memref<1x128x128xf32>) outs(%17 : memref<1x64x128xf32>)
            loom.semaphore_give %19 : memref<1x128x128xf32>
            linalg.fill ins(%cst_3 : f32) outs(%25 : memref<1x64xf32>)
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%17 : memref<1x64x128xf32>) outs(%25 : memref<1x64xf32>) {
            ^bb0(%in: f32, %out: f32):
              %44 = arith.maximumf %in, %out : f32
              linalg.yield %44 : f32
            }
            linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%23, %25 : memref<1x64xf32>, memref<1x64xf32>) outs(%25 : memref<1x64xf32>) {
            ^bb0(%in: f32, %in_7: f32, %out: f32):
              %44 = arith.truncf %cst_0 : f64 to f32
              %45 = arith.mulf %in_7, %44 : f32
              %46 = arith.cmpf ogt, %in, %45 : f32
              %47 = arith.select %46, %in, %45 : f32
              linalg.yield %47 : f32
            }
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%17, %25 : memref<1x64x128xf32>, memref<1x64xf32>) outs(%17 : memref<1x64x128xf32>) {
            ^bb0(%in: f32, %in_7: f32, %out: f32):
              %44 = arith.truncf %cst_0 : f64 to f32
              %45 = arith.mulf %in, %44 : f32
              %46 = arith.subf %45, %in_7 : f32
              %47 = math.powf %cst, %46 : f32
              linalg.yield %47 : f32
            }
            linalg.fill ins(%cst_1 : f32) outs(%27 : memref<1x64xf32>)
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%17 : memref<1x64x128xf32>) outs(%27 : memref<1x64xf32>) {
            ^bb0(%in: f32, %out: f32):
              %44 = arith.addf %in, %out : f32
              linalg.yield %44 : f32
            }
            linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%23, %25 : memref<1x64xf32>, memref<1x64xf32>) outs(%29 : memref<1x64xf32>) {
            ^bb0(%in: f32, %in_7: f32, %out: f32):
              %44 = arith.subf %in, %in_7 : f32
              %45 = math.powf %cst, %44 : f32
              linalg.yield %45 : f32
            }
            linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%21, %29, %27 : memref<1x64xf32>, memref<1x64xf32>, memref<1x64xf32>) outs(%21 : memref<1x64xf32>) {
            ^bb0(%in: f32, %in_7: f32, %in_8: f32, %out: f32):
              %44 = arith.mulf %in, %in_7 : f32
              %45 = arith.addf %44, %in_8 : f32
              linalg.yield %45 : f32
            }
            loom.semaphore_give %27 : memref<1x64xf32>
            %42 = arith.muli %arg7, %c16384 : index
            %43 = arith.addi %37, %42 : index
            %reinterpret_cast_6 = memref.reinterpret_cast %arg1 to offset: [%43], sizes: [1, 128, 128], strides: [524288, 128, 1] : memref<32x4096x128xf32> to memref<1x128x128xf32, strided<[524288, 128, 1], offset: ?>>
            loom.copy %reinterpret_cast_6, %15 src_mem_space @DRAM dst_mem_space @L1, interconnect : [@horizontal_links], broadcast : [1, 8] : memref<1x128x128xf32, strided<[524288, 128, 1], offset: ?>>, memref<1x128x128xf32>
            linalg.fill ins(%cst_1 : f32) outs(%36 : memref<1x64x128xf32>)
            linalg.batch_matmul ins(%17, %15 : memref<1x64x128xf32>, memref<1x128x128xf32>) outs(%36 : memref<1x64x128xf32>)
            loom.semaphore_give %15 : memref<1x128x128xf32>
            loom.semaphore_give %17 : memref<1x64x128xf32>
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%36, %34, %29 : memref<1x64x128xf32>, memref<1x64x128xf32>, memref<1x64xf32>) outs(%34 : memref<1x64x128xf32>) {
            ^bb0(%in: f32, %in_7: f32, %in_8: f32, %out: f32):
              %44 = arith.mulf %in_7, %in_8 : f32
              %45 = arith.addf %in, %44 : f32
              linalg.yield %45 : f32
            }
            loom.semaphore_give %29 : memref<1x64xf32>
            loom.semaphore_give %36 : memref<1x64x128xf32>
            linalg.copy ins(%25 : memref<1x64xf32>) outs(%23 : memref<1x64xf32>)
            loom.semaphore_give %25 : memref<1x64xf32>
          }
          loom.semaphore_give %23 : memref<1x64xf32>
          loom.semaphore_give %32 : memref<1x64x128xf32>
          linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%34, %21 : memref<1x64x128xf32>, memref<1x64xf32>) outs(%31 : memref<1x64x128xf32>) {
          ^bb0(%in: f32, %in_5: f32, %out: f32):
            %40 = arith.divf %in, %in_5 : f32
            linalg.yield %40 : f32
          }
          loom.semaphore_give %21 : memref<1x64xf32>
          loom.semaphore_give %34 : memref<1x64x128xf32>
          %reinterpret_cast_4 = memref.reinterpret_cast %arg3 to offset: [%39], sizes: [1, 64, 128], strides: [524288, 128, 1] : memref<32x4096x128xf32> to memref<1x64x128xf32, strided<[524288, 128, 1], offset: ?>>
          loom.copy %31, %reinterpret_cast_4 src_mem_space @L1 dst_mem_space @DRAM, interconnect : [], broadcast : [1, 1] : memref<1x64x128xf32>, memref<1x64x128xf32, strided<[524288, 128, 1], offset: ?>>
          loom.semaphore_give %31 : memref<1x64x128xf32>
        }
        scf.reduce 
      } {loom.iter_types = [#loom.iter_type<spatial>, #loom.iter_type<spatial>], loom.mapped_to_dims = [@x, @y]}
      return
    }
  }
  module attributes {loom.pass_name = "Materialize"} {
    func.func @attention__d0i1_d1i1__f01__d_v_v__BB1__BM64__BN128(%arg0: memref<32x128x4096xf32>, %arg1: memref<32x4096x128xf32>, %arg2: memref<32x4096x128xf32>, %arg3: memref<32x4096x128xf32>) {
      %c16384 = arith.constant 16384 : index
      %c8192 = arith.constant 8192 : index
      %c524288 = arith.constant 524288 : index
      %c32 = arith.constant 32 : index
      %cst = arith.constant 2.000000e+00 : f32
      %cst_0 = arith.constant 0.12751743074602467 : f64
      %cst_1 = arith.constant 0.000000e+00 : f32
      %cst_2 = arith.constant 1.000000e+00 : f32
      %cst_3 = arith.constant 0xFF800000 : f32
      %c128 = arith.constant 128 : index
      %c0 = arith.constant 0 : index
      %c8 = arith.constant 8 : index
      %c1 = arith.constant 1 : index
      scf.parallel (%arg4, %arg5) = (%c0, %c0) to (%c8, %c8) step (%c1, %c1) {
        scf.for %arg6 = %c0 to %c32 step %c1 {
          %12 = arith.muli %arg4, %c8 overflow<nsw> : index
          %13 = arith.addi %12, %arg5 : index
          %14 = loom.alloc [1, 128, 128] on @L1 : memref<1x128x128xf32>
          %15 = loom.semaphore_take %14 : memref<1x128x128xf32> -> memref<1x128x128xf32>
          %16 = loom.alloc [1, 64, 128] on @L1 : memref<1x64x128xf32>
          %17 = loom.semaphore_take %16 : memref<1x64x128xf32> -> memref<1x64x128xf32>
          %18 = loom.alloc [1, 128, 128] on @L1 : memref<1x128x128xf32>
          %19 = loom.semaphore_take %18 : memref<1x128x128xf32> -> memref<1x128x128xf32>
          %20 = loom.alloc [1, 64] on @L1 : memref<1x64xf32>
          %21 = loom.semaphore_take %20 : memref<1x64xf32> -> memref<1x64xf32>
          %22 = loom.alloc [1, 64] on @L1 : memref<1x64xf32>
          %23 = loom.semaphore_take %22 : memref<1x64xf32> -> memref<1x64xf32>
          %24 = loom.alloc [1, 64] on @L1 : memref<1x64xf32>
          %25 = loom.semaphore_take %24 : memref<1x64xf32> -> memref<1x64xf32>
          %26 = loom.alloc [1, 64] on @L1 : memref<1x64xf32>
          %27 = loom.semaphore_take %26 : memref<1x64xf32> -> memref<1x64xf32>
          %28 = loom.alloc [1, 64] on @L1 : memref<1x64xf32>
          %29 = loom.semaphore_take %28 : memref<1x64xf32> -> memref<1x64xf32>
          %30 = loom.alloc [1, 64, 128] on @L1 : memref<1x64x128xf32>
          %31 = loom.semaphore_take %30 : memref<1x64x128xf32> -> memref<1x64x128xf32>
          %32 = loom.semaphore_take %30 : memref<1x64x128xf32> -> memref<1x64x128xf32>
          %33 = loom.alloc [1, 64, 128] on @L1 : memref<1x64x128xf32>
          %34 = loom.semaphore_take %33 : memref<1x64x128xf32> -> memref<1x64x128xf32>
          %35 = loom.alloc [1, 64, 128] on @L1 : memref<1x64x128xf32>
          %36 = loom.semaphore_take %35 : memref<1x64x128xf32> -> memref<1x64x128xf32>
          %37 = arith.muli %arg6, %c524288 overflow<nsw> : index
          %38 = arith.muli %13, %c8192 : index
          %39 = arith.addi %37, %38 : index
          %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%39], sizes: [1, 64, 128], strides: [524288, 128, 1] : memref<32x4096x128xf32> to memref<1x64x128xf32, strided<[524288, 128, 1], offset: ?>>
          loom.copy %reinterpret_cast, %32 src_mem_space @DRAM dst_mem_space @L1, interconnect : [], broadcast : [1, 1] : memref<1x64x128xf32, strided<[524288, 128, 1], offset: ?>>, memref<1x64x128xf32>
          linalg.fill ins(%cst_1 : f32) outs(%34 : memref<1x64x128xf32>)
          linalg.fill ins(%cst_2 : f32) outs(%21 : memref<1x64xf32>)
          linalg.fill ins(%cst_3 : f32) outs(%23 : memref<1x64xf32>)
          scf.for %arg7 = %c0 to %c32 step %c1 {
            %40 = arith.muli %arg7, %c128 : index
            %41 = arith.addi %37, %40 : index
            %reinterpret_cast_5 = memref.reinterpret_cast %arg0 to offset: [%41], sizes: [1, 128, 128], strides: [524288, 4096, 1] : memref<32x128x4096xf32> to memref<1x128x128xf32, strided<[524288, 4096, 1], offset: ?>>
            loom.copy %reinterpret_cast_5, %19 src_mem_space @DRAM dst_mem_space @L1, interconnect : [@vertical_links], broadcast : [8, 1] : memref<1x128x128xf32, strided<[524288, 4096, 1], offset: ?>>, memref<1x128x128xf32>
            linalg.fill ins(%cst_1 : f32) outs(%17 : memref<1x64x128xf32>)
            linalg.batch_matmul ins(%32, %19 : memref<1x64x128xf32>, memref<1x128x128xf32>) outs(%17 : memref<1x64x128xf32>)
            loom.semaphore_give %19 : memref<1x128x128xf32>
            linalg.fill ins(%cst_3 : f32) outs(%25 : memref<1x64xf32>)
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%17 : memref<1x64x128xf32>) outs(%25 : memref<1x64xf32>) {
            ^bb0(%in: f32, %out: f32):
              %44 = arith.maximumf %in, %out : f32
              linalg.yield %44 : f32
            }
            linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%23, %25 : memref<1x64xf32>, memref<1x64xf32>) outs(%25 : memref<1x64xf32>) {
            ^bb0(%in: f32, %in_7: f32, %out: f32):
              %44 = arith.truncf %cst_0 : f64 to f32
              %45 = arith.mulf %in_7, %44 : f32
              %46 = arith.cmpf ogt, %in, %45 : f32
              %47 = arith.select %46, %in, %45 : f32
              linalg.yield %47 : f32
            }
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%17, %25 : memref<1x64x128xf32>, memref<1x64xf32>) outs(%17 : memref<1x64x128xf32>) {
            ^bb0(%in: f32, %in_7: f32, %out: f32):
              %44 = arith.truncf %cst_0 : f64 to f32
              %45 = arith.mulf %in, %44 : f32
              %46 = arith.subf %45, %in_7 : f32
              %47 = math.powf %cst, %46 : f32
              linalg.yield %47 : f32
            }
            linalg.fill ins(%cst_1 : f32) outs(%27 : memref<1x64xf32>)
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%17 : memref<1x64x128xf32>) outs(%27 : memref<1x64xf32>) {
            ^bb0(%in: f32, %out: f32):
              %44 = arith.addf %in, %out : f32
              linalg.yield %44 : f32
            }
            linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%23, %25 : memref<1x64xf32>, memref<1x64xf32>) outs(%29 : memref<1x64xf32>) {
            ^bb0(%in: f32, %in_7: f32, %out: f32):
              %44 = arith.subf %in, %in_7 : f32
              %45 = math.powf %cst, %44 : f32
              linalg.yield %45 : f32
            }
            linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%21, %29, %27 : memref<1x64xf32>, memref<1x64xf32>, memref<1x64xf32>) outs(%21 : memref<1x64xf32>) {
            ^bb0(%in: f32, %in_7: f32, %in_8: f32, %out: f32):
              %44 = arith.mulf %in, %in_7 : f32
              %45 = arith.addf %44, %in_8 : f32
              linalg.yield %45 : f32
            }
            loom.semaphore_give %27 : memref<1x64xf32>
            %42 = arith.muli %arg7, %c16384 : index
            %43 = arith.addi %37, %42 : index
            %reinterpret_cast_6 = memref.reinterpret_cast %arg1 to offset: [%43], sizes: [1, 128, 128], strides: [524288, 128, 1] : memref<32x4096x128xf32> to memref<1x128x128xf32, strided<[524288, 128, 1], offset: ?>>
            loom.copy %reinterpret_cast_6, %15 src_mem_space @DRAM dst_mem_space @L1, interconnect : [@vertical_links], broadcast : [8, 1] : memref<1x128x128xf32, strided<[524288, 128, 1], offset: ?>>, memref<1x128x128xf32>
            linalg.fill ins(%cst_1 : f32) outs(%36 : memref<1x64x128xf32>)
            linalg.batch_matmul ins(%17, %15 : memref<1x64x128xf32>, memref<1x128x128xf32>) outs(%36 : memref<1x64x128xf32>)
            loom.semaphore_give %15 : memref<1x128x128xf32>
            loom.semaphore_give %17 : memref<1x64x128xf32>
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%36, %34, %29 : memref<1x64x128xf32>, memref<1x64x128xf32>, memref<1x64xf32>) outs(%34 : memref<1x64x128xf32>) {
            ^bb0(%in: f32, %in_7: f32, %in_8: f32, %out: f32):
              %44 = arith.mulf %in_7, %in_8 : f32
              %45 = arith.addf %in, %44 : f32
              linalg.yield %45 : f32
            }
            loom.semaphore_give %29 : memref<1x64xf32>
            loom.semaphore_give %36 : memref<1x64x128xf32>
            linalg.copy ins(%25 : memref<1x64xf32>) outs(%23 : memref<1x64xf32>)
            loom.semaphore_give %25 : memref<1x64xf32>
          }
          loom.semaphore_give %23 : memref<1x64xf32>
          loom.semaphore_give %32 : memref<1x64x128xf32>
          linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%34, %21 : memref<1x64x128xf32>, memref<1x64xf32>) outs(%31 : memref<1x64x128xf32>) {
          ^bb0(%in: f32, %in_5: f32, %out: f32):
            %40 = arith.divf %in, %in_5 : f32
            linalg.yield %40 : f32
          }
          loom.semaphore_give %21 : memref<1x64xf32>
          loom.semaphore_give %34 : memref<1x64x128xf32>
          %reinterpret_cast_4 = memref.reinterpret_cast %arg3 to offset: [%39], sizes: [1, 64, 128], strides: [524288, 128, 1] : memref<32x4096x128xf32> to memref<1x64x128xf32, strided<[524288, 128, 1], offset: ?>>
          loom.copy %31, %reinterpret_cast_4 src_mem_space @L1 dst_mem_space @DRAM, interconnect : [], broadcast : [1, 1] : memref<1x64x128xf32>, memref<1x64x128xf32, strided<[524288, 128, 1], offset: ?>>
          loom.semaphore_give %31 : memref<1x64x128xf32>
        }
        scf.reduce 
      } {loom.iter_types = [#loom.iter_type<spatial>, #loom.iter_type<spatial>], loom.mapped_to_dims = [@x, @y]}
      return
    }
  }
  module attributes {loom.pass_name = "Materialize"} {
    func.func @attention__d0i1_d1i1__f10__d_d_d__BB1__BM64__BN128(%arg0: memref<32x128x4096xf32>, %arg1: memref<32x4096x128xf32>, %arg2: memref<32x4096x128xf32>, %arg3: memref<32x4096x128xf32>) {
      %c16384 = arith.constant 16384 : index
      %c8192 = arith.constant 8192 : index
      %c524288 = arith.constant 524288 : index
      %c32 = arith.constant 32 : index
      %cst = arith.constant 2.000000e+00 : f32
      %cst_0 = arith.constant 0.12751743074602467 : f64
      %cst_1 = arith.constant 0.000000e+00 : f32
      %cst_2 = arith.constant 1.000000e+00 : f32
      %cst_3 = arith.constant 0xFF800000 : f32
      %c128 = arith.constant 128 : index
      %c0 = arith.constant 0 : index
      %c8 = arith.constant 8 : index
      %c1 = arith.constant 1 : index
      scf.parallel (%arg4, %arg5) = (%c0, %c0) to (%c8, %c8) step (%c1, %c1) {
        scf.for %arg6 = %c0 to %c32 step %c1 {
          %12 = arith.muli %arg4, %c8 overflow<nsw> : index
          %13 = arith.addi %12, %arg5 : index
          %14 = loom.alloc [1, 128, 128] on @L1 : memref<1x128x128xf32>
          %15 = loom.semaphore_take %14 : memref<1x128x128xf32> -> memref<1x128x128xf32>
          %16 = loom.alloc [1, 64, 128] on @L1 : memref<1x64x128xf32>
          %17 = loom.semaphore_take %16 : memref<1x64x128xf32> -> memref<1x64x128xf32>
          %18 = loom.alloc [1, 128, 128] on @L1 : memref<1x128x128xf32>
          %19 = loom.semaphore_take %18 : memref<1x128x128xf32> -> memref<1x128x128xf32>
          %20 = loom.alloc [1, 64] on @L1 : memref<1x64xf32>
          %21 = loom.semaphore_take %20 : memref<1x64xf32> -> memref<1x64xf32>
          %22 = loom.alloc [1, 64] on @L1 : memref<1x64xf32>
          %23 = loom.semaphore_take %22 : memref<1x64xf32> -> memref<1x64xf32>
          %24 = loom.alloc [1, 64] on @L1 : memref<1x64xf32>
          %25 = loom.semaphore_take %24 : memref<1x64xf32> -> memref<1x64xf32>
          %26 = loom.alloc [1, 64] on @L1 : memref<1x64xf32>
          %27 = loom.semaphore_take %26 : memref<1x64xf32> -> memref<1x64xf32>
          %28 = loom.alloc [1, 64] on @L1 : memref<1x64xf32>
          %29 = loom.semaphore_take %28 : memref<1x64xf32> -> memref<1x64xf32>
          %30 = loom.alloc [1, 64, 128] on @L1 : memref<1x64x128xf32>
          %31 = loom.semaphore_take %30 : memref<1x64x128xf32> -> memref<1x64x128xf32>
          %32 = loom.semaphore_take %30 : memref<1x64x128xf32> -> memref<1x64x128xf32>
          %33 = loom.alloc [1, 64, 128] on @L1 : memref<1x64x128xf32>
          %34 = loom.semaphore_take %33 : memref<1x64x128xf32> -> memref<1x64x128xf32>
          %35 = loom.alloc [1, 64, 128] on @L1 : memref<1x64x128xf32>
          %36 = loom.semaphore_take %35 : memref<1x64x128xf32> -> memref<1x64x128xf32>
          %37 = arith.muli %arg6, %c524288 overflow<nsw> : index
          %38 = arith.muli %13, %c8192 : index
          %39 = arith.addi %37, %38 : index
          %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%39], sizes: [1, 64, 128], strides: [524288, 128, 1] : memref<32x4096x128xf32> to memref<1x64x128xf32, strided<[524288, 128, 1], offset: ?>>
          loom.copy %reinterpret_cast, %32 src_mem_space @DRAM dst_mem_space @L1, interconnect : [], broadcast : [1, 1] : memref<1x64x128xf32, strided<[524288, 128, 1], offset: ?>>, memref<1x64x128xf32>
          linalg.fill ins(%cst_1 : f32) outs(%34 : memref<1x64x128xf32>)
          linalg.fill ins(%cst_2 : f32) outs(%21 : memref<1x64xf32>)
          linalg.fill ins(%cst_3 : f32) outs(%23 : memref<1x64xf32>)
          scf.for %arg7 = %c0 to %c32 step %c1 {
            %40 = arith.muli %arg7, %c128 : index
            %41 = arith.addi %37, %40 : index
            %reinterpret_cast_5 = memref.reinterpret_cast %arg0 to offset: [%41], sizes: [1, 128, 128], strides: [524288, 4096, 1] : memref<32x128x4096xf32> to memref<1x128x128xf32, strided<[524288, 4096, 1], offset: ?>>
            loom.copy %reinterpret_cast_5, %19 src_mem_space @DRAM dst_mem_space @L1, interconnect : [], broadcast : [1, 1] : memref<1x128x128xf32, strided<[524288, 4096, 1], offset: ?>>, memref<1x128x128xf32>
            linalg.fill ins(%cst_1 : f32) outs(%17 : memref<1x64x128xf32>)
            linalg.batch_matmul ins(%32, %19 : memref<1x64x128xf32>, memref<1x128x128xf32>) outs(%17 : memref<1x64x128xf32>)
            loom.semaphore_give %19 : memref<1x128x128xf32>
            linalg.fill ins(%cst_3 : f32) outs(%25 : memref<1x64xf32>)
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%17 : memref<1x64x128xf32>) outs(%25 : memref<1x64xf32>) {
            ^bb0(%in: f32, %out: f32):
              %44 = arith.maximumf %in, %out : f32
              linalg.yield %44 : f32
            }
            linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%23, %25 : memref<1x64xf32>, memref<1x64xf32>) outs(%25 : memref<1x64xf32>) {
            ^bb0(%in: f32, %in_7: f32, %out: f32):
              %44 = arith.truncf %cst_0 : f64 to f32
              %45 = arith.mulf %in_7, %44 : f32
              %46 = arith.cmpf ogt, %in, %45 : f32
              %47 = arith.select %46, %in, %45 : f32
              linalg.yield %47 : f32
            }
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%17, %25 : memref<1x64x128xf32>, memref<1x64xf32>) outs(%17 : memref<1x64x128xf32>) {
            ^bb0(%in: f32, %in_7: f32, %out: f32):
              %44 = arith.truncf %cst_0 : f64 to f32
              %45 = arith.mulf %in, %44 : f32
              %46 = arith.subf %45, %in_7 : f32
              %47 = math.powf %cst, %46 : f32
              linalg.yield %47 : f32
            }
            linalg.fill ins(%cst_1 : f32) outs(%27 : memref<1x64xf32>)
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%17 : memref<1x64x128xf32>) outs(%27 : memref<1x64xf32>) {
            ^bb0(%in: f32, %out: f32):
              %44 = arith.addf %in, %out : f32
              linalg.yield %44 : f32
            }
            linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%23, %25 : memref<1x64xf32>, memref<1x64xf32>) outs(%29 : memref<1x64xf32>) {
            ^bb0(%in: f32, %in_7: f32, %out: f32):
              %44 = arith.subf %in, %in_7 : f32
              %45 = math.powf %cst, %44 : f32
              linalg.yield %45 : f32
            }
            linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%21, %29, %27 : memref<1x64xf32>, memref<1x64xf32>, memref<1x64xf32>) outs(%21 : memref<1x64xf32>) {
            ^bb0(%in: f32, %in_7: f32, %in_8: f32, %out: f32):
              %44 = arith.mulf %in, %in_7 : f32
              %45 = arith.addf %44, %in_8 : f32
              linalg.yield %45 : f32
            }
            loom.semaphore_give %27 : memref<1x64xf32>
            %42 = arith.muli %arg7, %c16384 : index
            %43 = arith.addi %37, %42 : index
            %reinterpret_cast_6 = memref.reinterpret_cast %arg1 to offset: [%43], sizes: [1, 128, 128], strides: [524288, 128, 1] : memref<32x4096x128xf32> to memref<1x128x128xf32, strided<[524288, 128, 1], offset: ?>>
            loom.copy %reinterpret_cast_6, %15 src_mem_space @DRAM dst_mem_space @L1, interconnect : [], broadcast : [1, 1] : memref<1x128x128xf32, strided<[524288, 128, 1], offset: ?>>, memref<1x128x128xf32>
            linalg.fill ins(%cst_1 : f32) outs(%36 : memref<1x64x128xf32>)
            linalg.batch_matmul ins(%17, %15 : memref<1x64x128xf32>, memref<1x128x128xf32>) outs(%36 : memref<1x64x128xf32>)
            loom.semaphore_give %15 : memref<1x128x128xf32>
            loom.semaphore_give %17 : memref<1x64x128xf32>
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%36, %34, %29 : memref<1x64x128xf32>, memref<1x64x128xf32>, memref<1x64xf32>) outs(%34 : memref<1x64x128xf32>) {
            ^bb0(%in: f32, %in_7: f32, %in_8: f32, %out: f32):
              %44 = arith.mulf %in_7, %in_8 : f32
              %45 = arith.addf %in, %44 : f32
              linalg.yield %45 : f32
            }
            loom.semaphore_give %29 : memref<1x64xf32>
            loom.semaphore_give %36 : memref<1x64x128xf32>
            linalg.copy ins(%25 : memref<1x64xf32>) outs(%23 : memref<1x64xf32>)
            loom.semaphore_give %25 : memref<1x64xf32>
          }
          loom.semaphore_give %23 : memref<1x64xf32>
          loom.semaphore_give %32 : memref<1x64x128xf32>
          linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%34, %21 : memref<1x64x128xf32>, memref<1x64xf32>) outs(%31 : memref<1x64x128xf32>) {
          ^bb0(%in: f32, %in_5: f32, %out: f32):
            %40 = arith.divf %in, %in_5 : f32
            linalg.yield %40 : f32
          }
          loom.semaphore_give %21 : memref<1x64xf32>
          loom.semaphore_give %34 : memref<1x64x128xf32>
          %reinterpret_cast_4 = memref.reinterpret_cast %arg3 to offset: [%39], sizes: [1, 64, 128], strides: [524288, 128, 1] : memref<32x4096x128xf32> to memref<1x64x128xf32, strided<[524288, 128, 1], offset: ?>>
          loom.copy %31, %reinterpret_cast_4 src_mem_space @L1 dst_mem_space @DRAM, interconnect : [], broadcast : [1, 1] : memref<1x64x128xf32>, memref<1x64x128xf32, strided<[524288, 128, 1], offset: ?>>
          loom.semaphore_give %31 : memref<1x64x128xf32>
        }
        scf.reduce 
      } {loom.iter_types = [#loom.iter_type<spatial>, #loom.iter_type<spatial>], loom.mapped_to_dims = [@x, @y]}
      return
    }
  }
  module attributes {loom.pass_name = "Materialize"} {
    func.func @attention__d0i1_d1i1__f10__d_d_a__BB1__BM64__BN128(%arg0: memref<32x128x4096xf32>, %arg1: memref<32x4096x128xf32>, %arg2: memref<32x4096x128xf32>, %arg3: memref<32x4096x128xf32>) {
      %c16384 = arith.constant 16384 : index
      %c8192 = arith.constant 8192 : index
      %c524288 = arith.constant 524288 : index
      %c32 = arith.constant 32 : index
      %cst = arith.constant 2.000000e+00 : f32
      %cst_0 = arith.constant 0.12751743074602467 : f64
      %cst_1 = arith.constant 0.000000e+00 : f32
      %cst_2 = arith.constant 1.000000e+00 : f32
      %cst_3 = arith.constant 0xFF800000 : f32
      %c128 = arith.constant 128 : index
      %c0 = arith.constant 0 : index
      %c8 = arith.constant 8 : index
      %c1 = arith.constant 1 : index
      scf.parallel (%arg4, %arg5) = (%c0, %c0) to (%c8, %c8) step (%c1, %c1) {
        scf.for %arg6 = %c0 to %c32 step %c1 {
          %12 = arith.muli %arg4, %c8 overflow<nsw> : index
          %13 = arith.addi %12, %arg5 : index
          %14 = loom.alloc [1, 128, 128] on @L1 : memref<1x128x128xf32>
          %15 = loom.semaphore_take %14 : memref<1x128x128xf32> -> memref<1x128x128xf32>
          %16 = loom.alloc [1, 64, 128] on @L1 : memref<1x64x128xf32>
          %17 = loom.semaphore_take %16 : memref<1x64x128xf32> -> memref<1x64x128xf32>
          %18 = loom.alloc [1, 128, 128] on @L1 : memref<1x128x128xf32>
          %19 = loom.semaphore_take %18 : memref<1x128x128xf32> -> memref<1x128x128xf32>
          %20 = loom.alloc [1, 64] on @L1 : memref<1x64xf32>
          %21 = loom.semaphore_take %20 : memref<1x64xf32> -> memref<1x64xf32>
          %22 = loom.alloc [1, 64] on @L1 : memref<1x64xf32>
          %23 = loom.semaphore_take %22 : memref<1x64xf32> -> memref<1x64xf32>
          %24 = loom.alloc [1, 64] on @L1 : memref<1x64xf32>
          %25 = loom.semaphore_take %24 : memref<1x64xf32> -> memref<1x64xf32>
          %26 = loom.alloc [1, 64] on @L1 : memref<1x64xf32>
          %27 = loom.semaphore_take %26 : memref<1x64xf32> -> memref<1x64xf32>
          %28 = loom.alloc [1, 64] on @L1 : memref<1x64xf32>
          %29 = loom.semaphore_take %28 : memref<1x64xf32> -> memref<1x64xf32>
          %30 = loom.alloc [1, 64, 128] on @L1 : memref<1x64x128xf32>
          %31 = loom.semaphore_take %30 : memref<1x64x128xf32> -> memref<1x64x128xf32>
          %32 = loom.semaphore_take %30 : memref<1x64x128xf32> -> memref<1x64x128xf32>
          %33 = loom.alloc [1, 64, 128] on @L1 : memref<1x64x128xf32>
          %34 = loom.semaphore_take %33 : memref<1x64x128xf32> -> memref<1x64x128xf32>
          %35 = loom.alloc [1, 64, 128] on @L1 : memref<1x64x128xf32>
          %36 = loom.semaphore_take %35 : memref<1x64x128xf32> -> memref<1x64x128xf32>
          %37 = arith.muli %arg6, %c524288 overflow<nsw> : index
          %38 = arith.muli %13, %c8192 : index
          %39 = arith.addi %37, %38 : index
          %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%39], sizes: [1, 64, 128], strides: [524288, 128, 1] : memref<32x4096x128xf32> to memref<1x64x128xf32, strided<[524288, 128, 1], offset: ?>>
          loom.copy %reinterpret_cast, %32 src_mem_space @DRAM dst_mem_space @L1, interconnect : [], broadcast : [1, 1] : memref<1x64x128xf32, strided<[524288, 128, 1], offset: ?>>, memref<1x64x128xf32>
          linalg.fill ins(%cst_1 : f32) outs(%34 : memref<1x64x128xf32>)
          linalg.fill ins(%cst_2 : f32) outs(%21 : memref<1x64xf32>)
          linalg.fill ins(%cst_3 : f32) outs(%23 : memref<1x64xf32>)
          scf.for %arg7 = %c0 to %c32 step %c1 {
            %40 = arith.muli %arg7, %c128 : index
            %41 = arith.addi %37, %40 : index
            %reinterpret_cast_5 = memref.reinterpret_cast %arg0 to offset: [%41], sizes: [1, 128, 128], strides: [524288, 4096, 1] : memref<32x128x4096xf32> to memref<1x128x128xf32, strided<[524288, 4096, 1], offset: ?>>
            loom.copy %reinterpret_cast_5, %19 src_mem_space @DRAM dst_mem_space @L1, interconnect : [], broadcast : [1, 1] : memref<1x128x128xf32, strided<[524288, 4096, 1], offset: ?>>, memref<1x128x128xf32>
            linalg.fill ins(%cst_1 : f32) outs(%17 : memref<1x64x128xf32>)
            linalg.batch_matmul ins(%32, %19 : memref<1x64x128xf32>, memref<1x128x128xf32>) outs(%17 : memref<1x64x128xf32>)
            loom.semaphore_give %19 : memref<1x128x128xf32>
            linalg.fill ins(%cst_3 : f32) outs(%25 : memref<1x64xf32>)
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%17 : memref<1x64x128xf32>) outs(%25 : memref<1x64xf32>) {
            ^bb0(%in: f32, %out: f32):
              %44 = arith.maximumf %in, %out : f32
              linalg.yield %44 : f32
            }
            linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%23, %25 : memref<1x64xf32>, memref<1x64xf32>) outs(%25 : memref<1x64xf32>) {
            ^bb0(%in: f32, %in_7: f32, %out: f32):
              %44 = arith.truncf %cst_0 : f64 to f32
              %45 = arith.mulf %in_7, %44 : f32
              %46 = arith.cmpf ogt, %in, %45 : f32
              %47 = arith.select %46, %in, %45 : f32
              linalg.yield %47 : f32
            }
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%17, %25 : memref<1x64x128xf32>, memref<1x64xf32>) outs(%17 : memref<1x64x128xf32>) {
            ^bb0(%in: f32, %in_7: f32, %out: f32):
              %44 = arith.truncf %cst_0 : f64 to f32
              %45 = arith.mulf %in, %44 : f32
              %46 = arith.subf %45, %in_7 : f32
              %47 = math.powf %cst, %46 : f32
              linalg.yield %47 : f32
            }
            linalg.fill ins(%cst_1 : f32) outs(%27 : memref<1x64xf32>)
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%17 : memref<1x64x128xf32>) outs(%27 : memref<1x64xf32>) {
            ^bb0(%in: f32, %out: f32):
              %44 = arith.addf %in, %out : f32
              linalg.yield %44 : f32
            }
            linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%23, %25 : memref<1x64xf32>, memref<1x64xf32>) outs(%29 : memref<1x64xf32>) {
            ^bb0(%in: f32, %in_7: f32, %out: f32):
              %44 = arith.subf %in, %in_7 : f32
              %45 = math.powf %cst, %44 : f32
              linalg.yield %45 : f32
            }
            linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%21, %29, %27 : memref<1x64xf32>, memref<1x64xf32>, memref<1x64xf32>) outs(%21 : memref<1x64xf32>) {
            ^bb0(%in: f32, %in_7: f32, %in_8: f32, %out: f32):
              %44 = arith.mulf %in, %in_7 : f32
              %45 = arith.addf %44, %in_8 : f32
              linalg.yield %45 : f32
            }
            loom.semaphore_give %27 : memref<1x64xf32>
            %42 = arith.muli %arg7, %c16384 : index
            %43 = arith.addi %37, %42 : index
            %reinterpret_cast_6 = memref.reinterpret_cast %arg1 to offset: [%43], sizes: [1, 128, 128], strides: [524288, 128, 1] : memref<32x4096x128xf32> to memref<1x128x128xf32, strided<[524288, 128, 1], offset: ?>>
            loom.copy %reinterpret_cast_6, %15 src_mem_space @DRAM dst_mem_space @L1, interconnect : [@horizontal_links, @vertical_links], broadcast : [8, 8] : memref<1x128x128xf32, strided<[524288, 128, 1], offset: ?>>, memref<1x128x128xf32>
            linalg.fill ins(%cst_1 : f32) outs(%36 : memref<1x64x128xf32>)
            linalg.batch_matmul ins(%17, %15 : memref<1x64x128xf32>, memref<1x128x128xf32>) outs(%36 : memref<1x64x128xf32>)
            loom.semaphore_give %15 : memref<1x128x128xf32>
            loom.semaphore_give %17 : memref<1x64x128xf32>
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%36, %34, %29 : memref<1x64x128xf32>, memref<1x64x128xf32>, memref<1x64xf32>) outs(%34 : memref<1x64x128xf32>) {
            ^bb0(%in: f32, %in_7: f32, %in_8: f32, %out: f32):
              %44 = arith.mulf %in_7, %in_8 : f32
              %45 = arith.addf %in, %44 : f32
              linalg.yield %45 : f32
            }
            loom.semaphore_give %29 : memref<1x64xf32>
            loom.semaphore_give %36 : memref<1x64x128xf32>
            linalg.copy ins(%25 : memref<1x64xf32>) outs(%23 : memref<1x64xf32>)
            loom.semaphore_give %25 : memref<1x64xf32>
          }
          loom.semaphore_give %23 : memref<1x64xf32>
          loom.semaphore_give %32 : memref<1x64x128xf32>
          linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%34, %21 : memref<1x64x128xf32>, memref<1x64xf32>) outs(%31 : memref<1x64x128xf32>) {
          ^bb0(%in: f32, %in_5: f32, %out: f32):
            %40 = arith.divf %in, %in_5 : f32
            linalg.yield %40 : f32
          }
          loom.semaphore_give %21 : memref<1x64xf32>
          loom.semaphore_give %34 : memref<1x64x128xf32>
          %reinterpret_cast_4 = memref.reinterpret_cast %arg3 to offset: [%39], sizes: [1, 64, 128], strides: [524288, 128, 1] : memref<32x4096x128xf32> to memref<1x64x128xf32, strided<[524288, 128, 1], offset: ?>>
          loom.copy %31, %reinterpret_cast_4 src_mem_space @L1 dst_mem_space @DRAM, interconnect : [], broadcast : [1, 1] : memref<1x64x128xf32>, memref<1x64x128xf32, strided<[524288, 128, 1], offset: ?>>
          loom.semaphore_give %31 : memref<1x64x128xf32>
        }
        scf.reduce 
      } {loom.iter_types = [#loom.iter_type<spatial>, #loom.iter_type<spatial>], loom.mapped_to_dims = [@x, @y]}
      return
    }
  }
  module attributes {loom.pass_name = "Materialize"} {
    func.func @attention__d0i1_d1i1__f10__d_d_h__BB1__BM64__BN128(%arg0: memref<32x128x4096xf32>, %arg1: memref<32x4096x128xf32>, %arg2: memref<32x4096x128xf32>, %arg3: memref<32x4096x128xf32>) {
      %c16384 = arith.constant 16384 : index
      %c8192 = arith.constant 8192 : index
      %c524288 = arith.constant 524288 : index
      %c32 = arith.constant 32 : index
      %cst = arith.constant 2.000000e+00 : f32
      %cst_0 = arith.constant 0.12751743074602467 : f64
      %cst_1 = arith.constant 0.000000e+00 : f32
      %cst_2 = arith.constant 1.000000e+00 : f32
      %cst_3 = arith.constant 0xFF800000 : f32
      %c128 = arith.constant 128 : index
      %c0 = arith.constant 0 : index
      %c8 = arith.constant 8 : index
      %c1 = arith.constant 1 : index
      scf.parallel (%arg4, %arg5) = (%c0, %c0) to (%c8, %c8) step (%c1, %c1) {
        scf.for %arg6 = %c0 to %c32 step %c1 {
          %12 = arith.muli %arg4, %c8 overflow<nsw> : index
          %13 = arith.addi %12, %arg5 : index
          %14 = loom.alloc [1, 128, 128] on @L1 : memref<1x128x128xf32>
          %15 = loom.semaphore_take %14 : memref<1x128x128xf32> -> memref<1x128x128xf32>
          %16 = loom.alloc [1, 64, 128] on @L1 : memref<1x64x128xf32>
          %17 = loom.semaphore_take %16 : memref<1x64x128xf32> -> memref<1x64x128xf32>
          %18 = loom.alloc [1, 128, 128] on @L1 : memref<1x128x128xf32>
          %19 = loom.semaphore_take %18 : memref<1x128x128xf32> -> memref<1x128x128xf32>
          %20 = loom.alloc [1, 64] on @L1 : memref<1x64xf32>
          %21 = loom.semaphore_take %20 : memref<1x64xf32> -> memref<1x64xf32>
          %22 = loom.alloc [1, 64] on @L1 : memref<1x64xf32>
          %23 = loom.semaphore_take %22 : memref<1x64xf32> -> memref<1x64xf32>
          %24 = loom.alloc [1, 64] on @L1 : memref<1x64xf32>
          %25 = loom.semaphore_take %24 : memref<1x64xf32> -> memref<1x64xf32>
          %26 = loom.alloc [1, 64] on @L1 : memref<1x64xf32>
          %27 = loom.semaphore_take %26 : memref<1x64xf32> -> memref<1x64xf32>
          %28 = loom.alloc [1, 64] on @L1 : memref<1x64xf32>
          %29 = loom.semaphore_take %28 : memref<1x64xf32> -> memref<1x64xf32>
          %30 = loom.alloc [1, 64, 128] on @L1 : memref<1x64x128xf32>
          %31 = loom.semaphore_take %30 : memref<1x64x128xf32> -> memref<1x64x128xf32>
          %32 = loom.semaphore_take %30 : memref<1x64x128xf32> -> memref<1x64x128xf32>
          %33 = loom.alloc [1, 64, 128] on @L1 : memref<1x64x128xf32>
          %34 = loom.semaphore_take %33 : memref<1x64x128xf32> -> memref<1x64x128xf32>
          %35 = loom.alloc [1, 64, 128] on @L1 : memref<1x64x128xf32>
          %36 = loom.semaphore_take %35 : memref<1x64x128xf32> -> memref<1x64x128xf32>
          %37 = arith.muli %arg6, %c524288 overflow<nsw> : index
          %38 = arith.muli %13, %c8192 : index
          %39 = arith.addi %37, %38 : index
          %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%39], sizes: [1, 64, 128], strides: [524288, 128, 1] : memref<32x4096x128xf32> to memref<1x64x128xf32, strided<[524288, 128, 1], offset: ?>>
          loom.copy %reinterpret_cast, %32 src_mem_space @DRAM dst_mem_space @L1, interconnect : [], broadcast : [1, 1] : memref<1x64x128xf32, strided<[524288, 128, 1], offset: ?>>, memref<1x64x128xf32>
          linalg.fill ins(%cst_1 : f32) outs(%34 : memref<1x64x128xf32>)
          linalg.fill ins(%cst_2 : f32) outs(%21 : memref<1x64xf32>)
          linalg.fill ins(%cst_3 : f32) outs(%23 : memref<1x64xf32>)
          scf.for %arg7 = %c0 to %c32 step %c1 {
            %40 = arith.muli %arg7, %c128 : index
            %41 = arith.addi %37, %40 : index
            %reinterpret_cast_5 = memref.reinterpret_cast %arg0 to offset: [%41], sizes: [1, 128, 128], strides: [524288, 4096, 1] : memref<32x128x4096xf32> to memref<1x128x128xf32, strided<[524288, 4096, 1], offset: ?>>
            loom.copy %reinterpret_cast_5, %19 src_mem_space @DRAM dst_mem_space @L1, interconnect : [], broadcast : [1, 1] : memref<1x128x128xf32, strided<[524288, 4096, 1], offset: ?>>, memref<1x128x128xf32>
            linalg.fill ins(%cst_1 : f32) outs(%17 : memref<1x64x128xf32>)
            linalg.batch_matmul ins(%32, %19 : memref<1x64x128xf32>, memref<1x128x128xf32>) outs(%17 : memref<1x64x128xf32>)
            loom.semaphore_give %19 : memref<1x128x128xf32>
            linalg.fill ins(%cst_3 : f32) outs(%25 : memref<1x64xf32>)
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%17 : memref<1x64x128xf32>) outs(%25 : memref<1x64xf32>) {
            ^bb0(%in: f32, %out: f32):
              %44 = arith.maximumf %in, %out : f32
              linalg.yield %44 : f32
            }
            linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%23, %25 : memref<1x64xf32>, memref<1x64xf32>) outs(%25 : memref<1x64xf32>) {
            ^bb0(%in: f32, %in_7: f32, %out: f32):
              %44 = arith.truncf %cst_0 : f64 to f32
              %45 = arith.mulf %in_7, %44 : f32
              %46 = arith.cmpf ogt, %in, %45 : f32
              %47 = arith.select %46, %in, %45 : f32
              linalg.yield %47 : f32
            }
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%17, %25 : memref<1x64x128xf32>, memref<1x64xf32>) outs(%17 : memref<1x64x128xf32>) {
            ^bb0(%in: f32, %in_7: f32, %out: f32):
              %44 = arith.truncf %cst_0 : f64 to f32
              %45 = arith.mulf %in, %44 : f32
              %46 = arith.subf %45, %in_7 : f32
              %47 = math.powf %cst, %46 : f32
              linalg.yield %47 : f32
            }
            linalg.fill ins(%cst_1 : f32) outs(%27 : memref<1x64xf32>)
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%17 : memref<1x64x128xf32>) outs(%27 : memref<1x64xf32>) {
            ^bb0(%in: f32, %out: f32):
              %44 = arith.addf %in, %out : f32
              linalg.yield %44 : f32
            }
            linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%23, %25 : memref<1x64xf32>, memref<1x64xf32>) outs(%29 : memref<1x64xf32>) {
            ^bb0(%in: f32, %in_7: f32, %out: f32):
              %44 = arith.subf %in, %in_7 : f32
              %45 = math.powf %cst, %44 : f32
              linalg.yield %45 : f32
            }
            linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%21, %29, %27 : memref<1x64xf32>, memref<1x64xf32>, memref<1x64xf32>) outs(%21 : memref<1x64xf32>) {
            ^bb0(%in: f32, %in_7: f32, %in_8: f32, %out: f32):
              %44 = arith.mulf %in, %in_7 : f32
              %45 = arith.addf %44, %in_8 : f32
              linalg.yield %45 : f32
            }
            loom.semaphore_give %27 : memref<1x64xf32>
            %42 = arith.muli %arg7, %c16384 : index
            %43 = arith.addi %37, %42 : index
            %reinterpret_cast_6 = memref.reinterpret_cast %arg1 to offset: [%43], sizes: [1, 128, 128], strides: [524288, 128, 1] : memref<32x4096x128xf32> to memref<1x128x128xf32, strided<[524288, 128, 1], offset: ?>>
            loom.copy %reinterpret_cast_6, %15 src_mem_space @DRAM dst_mem_space @L1, interconnect : [@horizontal_links], broadcast : [1, 8] : memref<1x128x128xf32, strided<[524288, 128, 1], offset: ?>>, memref<1x128x128xf32>
            linalg.fill ins(%cst_1 : f32) outs(%36 : memref<1x64x128xf32>)
            linalg.batch_matmul ins(%17, %15 : memref<1x64x128xf32>, memref<1x128x128xf32>) outs(%36 : memref<1x64x128xf32>)
            loom.semaphore_give %15 : memref<1x128x128xf32>
            loom.semaphore_give %17 : memref<1x64x128xf32>
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%36, %34, %29 : memref<1x64x128xf32>, memref<1x64x128xf32>, memref<1x64xf32>) outs(%34 : memref<1x64x128xf32>) {
            ^bb0(%in: f32, %in_7: f32, %in_8: f32, %out: f32):
              %44 = arith.mulf %in_7, %in_8 : f32
              %45 = arith.addf %in, %44 : f32
              linalg.yield %45 : f32
            }
            loom.semaphore_give %29 : memref<1x64xf32>
            loom.semaphore_give %36 : memref<1x64x128xf32>
            linalg.copy ins(%25 : memref<1x64xf32>) outs(%23 : memref<1x64xf32>)
            loom.semaphore_give %25 : memref<1x64xf32>
          }
          loom.semaphore_give %23 : memref<1x64xf32>
          loom.semaphore_give %32 : memref<1x64x128xf32>
          linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%34, %21 : memref<1x64x128xf32>, memref<1x64xf32>) outs(%31 : memref<1x64x128xf32>) {
          ^bb0(%in: f32, %in_5: f32, %out: f32):
            %40 = arith.divf %in, %in_5 : f32
            linalg.yield %40 : f32
          }
          loom.semaphore_give %21 : memref<1x64xf32>
          loom.semaphore_give %34 : memref<1x64x128xf32>
          %reinterpret_cast_4 = memref.reinterpret_cast %arg3 to offset: [%39], sizes: [1, 64, 128], strides: [524288, 128, 1] : memref<32x4096x128xf32> to memref<1x64x128xf32, strided<[524288, 128, 1], offset: ?>>
          loom.copy %31, %reinterpret_cast_4 src_mem_space @L1 dst_mem_space @DRAM, interconnect : [], broadcast : [1, 1] : memref<1x64x128xf32>, memref<1x64x128xf32, strided<[524288, 128, 1], offset: ?>>
          loom.semaphore_give %31 : memref<1x64x128xf32>
        }
        scf.reduce 
      } {loom.iter_types = [#loom.iter_type<spatial>, #loom.iter_type<spatial>], loom.mapped_to_dims = [@x, @y]}
      return
    }
  }
  module attributes {loom.pass_name = "Materialize"} {
    func.func @attention__d0i1_d1i1__f10__d_d_v__BB1__BM64__BN128(%arg0: memref<32x128x4096xf32>, %arg1: memref<32x4096x128xf32>, %arg2: memref<32x4096x128xf32>, %arg3: memref<32x4096x128xf32>) {
      %c16384 = arith.constant 16384 : index
      %c8192 = arith.constant 8192 : index
      %c524288 = arith.constant 524288 : index
      %c32 = arith.constant 32 : index
      %cst = arith.constant 2.000000e+00 : f32
      %cst_0 = arith.constant 0.12751743074602467 : f64
      %cst_1 = arith.constant 0.000000e+00 : f32
      %cst_2 = arith.constant 1.000000e+00 : f32
      %cst_3 = arith.constant 0xFF800000 : f32
      %c128 = arith.constant 128 : index
      %c0 = arith.constant 0 : index
      %c8 = arith.constant 8 : index
      %c1 = arith.constant 1 : index
      scf.parallel (%arg4, %arg5) = (%c0, %c0) to (%c8, %c8) step (%c1, %c1) {
        scf.for %arg6 = %c0 to %c32 step %c1 {
          %12 = arith.muli %arg4, %c8 overflow<nsw> : index
          %13 = arith.addi %12, %arg5 : index
          %14 = loom.alloc [1, 128, 128] on @L1 : memref<1x128x128xf32>
          %15 = loom.semaphore_take %14 : memref<1x128x128xf32> -> memref<1x128x128xf32>
          %16 = loom.alloc [1, 64, 128] on @L1 : memref<1x64x128xf32>
          %17 = loom.semaphore_take %16 : memref<1x64x128xf32> -> memref<1x64x128xf32>
          %18 = loom.alloc [1, 128, 128] on @L1 : memref<1x128x128xf32>
          %19 = loom.semaphore_take %18 : memref<1x128x128xf32> -> memref<1x128x128xf32>
          %20 = loom.alloc [1, 64] on @L1 : memref<1x64xf32>
          %21 = loom.semaphore_take %20 : memref<1x64xf32> -> memref<1x64xf32>
          %22 = loom.alloc [1, 64] on @L1 : memref<1x64xf32>
          %23 = loom.semaphore_take %22 : memref<1x64xf32> -> memref<1x64xf32>
          %24 = loom.alloc [1, 64] on @L1 : memref<1x64xf32>
          %25 = loom.semaphore_take %24 : memref<1x64xf32> -> memref<1x64xf32>
          %26 = loom.alloc [1, 64] on @L1 : memref<1x64xf32>
          %27 = loom.semaphore_take %26 : memref<1x64xf32> -> memref<1x64xf32>
          %28 = loom.alloc [1, 64] on @L1 : memref<1x64xf32>
          %29 = loom.semaphore_take %28 : memref<1x64xf32> -> memref<1x64xf32>
          %30 = loom.alloc [1, 64, 128] on @L1 : memref<1x64x128xf32>
          %31 = loom.semaphore_take %30 : memref<1x64x128xf32> -> memref<1x64x128xf32>
          %32 = loom.semaphore_take %30 : memref<1x64x128xf32> -> memref<1x64x128xf32>
          %33 = loom.alloc [1, 64, 128] on @L1 : memref<1x64x128xf32>
          %34 = loom.semaphore_take %33 : memref<1x64x128xf32> -> memref<1x64x128xf32>
          %35 = loom.alloc [1, 64, 128] on @L1 : memref<1x64x128xf32>
          %36 = loom.semaphore_take %35 : memref<1x64x128xf32> -> memref<1x64x128xf32>
          %37 = arith.muli %arg6, %c524288 overflow<nsw> : index
          %38 = arith.muli %13, %c8192 : index
          %39 = arith.addi %37, %38 : index
          %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%39], sizes: [1, 64, 128], strides: [524288, 128, 1] : memref<32x4096x128xf32> to memref<1x64x128xf32, strided<[524288, 128, 1], offset: ?>>
          loom.copy %reinterpret_cast, %32 src_mem_space @DRAM dst_mem_space @L1, interconnect : [], broadcast : [1, 1] : memref<1x64x128xf32, strided<[524288, 128, 1], offset: ?>>, memref<1x64x128xf32>
          linalg.fill ins(%cst_1 : f32) outs(%34 : memref<1x64x128xf32>)
          linalg.fill ins(%cst_2 : f32) outs(%21 : memref<1x64xf32>)
          linalg.fill ins(%cst_3 : f32) outs(%23 : memref<1x64xf32>)
          scf.for %arg7 = %c0 to %c32 step %c1 {
            %40 = arith.muli %arg7, %c128 : index
            %41 = arith.addi %37, %40 : index
            %reinterpret_cast_5 = memref.reinterpret_cast %arg0 to offset: [%41], sizes: [1, 128, 128], strides: [524288, 4096, 1] : memref<32x128x4096xf32> to memref<1x128x128xf32, strided<[524288, 4096, 1], offset: ?>>
            loom.copy %reinterpret_cast_5, %19 src_mem_space @DRAM dst_mem_space @L1, interconnect : [], broadcast : [1, 1] : memref<1x128x128xf32, strided<[524288, 4096, 1], offset: ?>>, memref<1x128x128xf32>
            linalg.fill ins(%cst_1 : f32) outs(%17 : memref<1x64x128xf32>)
            linalg.batch_matmul ins(%32, %19 : memref<1x64x128xf32>, memref<1x128x128xf32>) outs(%17 : memref<1x64x128xf32>)
            loom.semaphore_give %19 : memref<1x128x128xf32>
            linalg.fill ins(%cst_3 : f32) outs(%25 : memref<1x64xf32>)
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%17 : memref<1x64x128xf32>) outs(%25 : memref<1x64xf32>) {
            ^bb0(%in: f32, %out: f32):
              %44 = arith.maximumf %in, %out : f32
              linalg.yield %44 : f32
            }
            linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%23, %25 : memref<1x64xf32>, memref<1x64xf32>) outs(%25 : memref<1x64xf32>) {
            ^bb0(%in: f32, %in_7: f32, %out: f32):
              %44 = arith.truncf %cst_0 : f64 to f32
              %45 = arith.mulf %in_7, %44 : f32
              %46 = arith.cmpf ogt, %in, %45 : f32
              %47 = arith.select %46, %in, %45 : f32
              linalg.yield %47 : f32
            }
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%17, %25 : memref<1x64x128xf32>, memref<1x64xf32>) outs(%17 : memref<1x64x128xf32>) {
            ^bb0(%in: f32, %in_7: f32, %out: f32):
              %44 = arith.truncf %cst_0 : f64 to f32
              %45 = arith.mulf %in, %44 : f32
              %46 = arith.subf %45, %in_7 : f32
              %47 = math.powf %cst, %46 : f32
              linalg.yield %47 : f32
            }
            linalg.fill ins(%cst_1 : f32) outs(%27 : memref<1x64xf32>)
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%17 : memref<1x64x128xf32>) outs(%27 : memref<1x64xf32>) {
            ^bb0(%in: f32, %out: f32):
              %44 = arith.addf %in, %out : f32
              linalg.yield %44 : f32
            }
            linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%23, %25 : memref<1x64xf32>, memref<1x64xf32>) outs(%29 : memref<1x64xf32>) {
            ^bb0(%in: f32, %in_7: f32, %out: f32):
              %44 = arith.subf %in, %in_7 : f32
              %45 = math.powf %cst, %44 : f32
              linalg.yield %45 : f32
            }
            linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%21, %29, %27 : memref<1x64xf32>, memref<1x64xf32>, memref<1x64xf32>) outs(%21 : memref<1x64xf32>) {
            ^bb0(%in: f32, %in_7: f32, %in_8: f32, %out: f32):
              %44 = arith.mulf %in, %in_7 : f32
              %45 = arith.addf %44, %in_8 : f32
              linalg.yield %45 : f32
            }
            loom.semaphore_give %27 : memref<1x64xf32>
            %42 = arith.muli %arg7, %c16384 : index
            %43 = arith.addi %37, %42 : index
            %reinterpret_cast_6 = memref.reinterpret_cast %arg1 to offset: [%43], sizes: [1, 128, 128], strides: [524288, 128, 1] : memref<32x4096x128xf32> to memref<1x128x128xf32, strided<[524288, 128, 1], offset: ?>>
            loom.copy %reinterpret_cast_6, %15 src_mem_space @DRAM dst_mem_space @L1, interconnect : [@vertical_links], broadcast : [8, 1] : memref<1x128x128xf32, strided<[524288, 128, 1], offset: ?>>, memref<1x128x128xf32>
            linalg.fill ins(%cst_1 : f32) outs(%36 : memref<1x64x128xf32>)
            linalg.batch_matmul ins(%17, %15 : memref<1x64x128xf32>, memref<1x128x128xf32>) outs(%36 : memref<1x64x128xf32>)
            loom.semaphore_give %15 : memref<1x128x128xf32>
            loom.semaphore_give %17 : memref<1x64x128xf32>
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%36, %34, %29 : memref<1x64x128xf32>, memref<1x64x128xf32>, memref<1x64xf32>) outs(%34 : memref<1x64x128xf32>) {
            ^bb0(%in: f32, %in_7: f32, %in_8: f32, %out: f32):
              %44 = arith.mulf %in_7, %in_8 : f32
              %45 = arith.addf %in, %44 : f32
              linalg.yield %45 : f32
            }
            loom.semaphore_give %29 : memref<1x64xf32>
            loom.semaphore_give %36 : memref<1x64x128xf32>
            linalg.copy ins(%25 : memref<1x64xf32>) outs(%23 : memref<1x64xf32>)
            loom.semaphore_give %25 : memref<1x64xf32>
          }
          loom.semaphore_give %23 : memref<1x64xf32>
          loom.semaphore_give %32 : memref<1x64x128xf32>
          linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%34, %21 : memref<1x64x128xf32>, memref<1x64xf32>) outs(%31 : memref<1x64x128xf32>) {
          ^bb0(%in: f32, %in_5: f32, %out: f32):
            %40 = arith.divf %in, %in_5 : f32
            linalg.yield %40 : f32
          }
          loom.semaphore_give %21 : memref<1x64xf32>
          loom.semaphore_give %34 : memref<1x64x128xf32>
          %reinterpret_cast_4 = memref.reinterpret_cast %arg3 to offset: [%39], sizes: [1, 64, 128], strides: [524288, 128, 1] : memref<32x4096x128xf32> to memref<1x64x128xf32, strided<[524288, 128, 1], offset: ?>>
          loom.copy %31, %reinterpret_cast_4 src_mem_space @L1 dst_mem_space @DRAM, interconnect : [], broadcast : [1, 1] : memref<1x64x128xf32>, memref<1x64x128xf32, strided<[524288, 128, 1], offset: ?>>
          loom.semaphore_give %31 : memref<1x64x128xf32>
        }
        scf.reduce 
      } {loom.iter_types = [#loom.iter_type<spatial>, #loom.iter_type<spatial>], loom.mapped_to_dims = [@x, @y]}
      return
    }
  }
  module attributes {loom.pass_name = "Materialize"} {
    func.func @attention__d0i1_d1i1__f10__d_a_d__BB1__BM64__BN128(%arg0: memref<32x128x4096xf32>, %arg1: memref<32x4096x128xf32>, %arg2: memref<32x4096x128xf32>, %arg3: memref<32x4096x128xf32>) {
      %c16384 = arith.constant 16384 : index
      %c8192 = arith.constant 8192 : index
      %c524288 = arith.constant 524288 : index
      %c32 = arith.constant 32 : index
      %cst = arith.constant 2.000000e+00 : f32
      %cst_0 = arith.constant 0.12751743074602467 : f64
      %cst_1 = arith.constant 0.000000e+00 : f32
      %cst_2 = arith.constant 1.000000e+00 : f32
      %cst_3 = arith.constant 0xFF800000 : f32
      %c128 = arith.constant 128 : index
      %c0 = arith.constant 0 : index
      %c8 = arith.constant 8 : index
      %c1 = arith.constant 1 : index
      scf.parallel (%arg4, %arg5) = (%c0, %c0) to (%c8, %c8) step (%c1, %c1) {
        scf.for %arg6 = %c0 to %c32 step %c1 {
          %12 = arith.muli %arg4, %c8 overflow<nsw> : index
          %13 = arith.addi %12, %arg5 : index
          %14 = loom.alloc [1, 128, 128] on @L1 : memref<1x128x128xf32>
          %15 = loom.semaphore_take %14 : memref<1x128x128xf32> -> memref<1x128x128xf32>
          %16 = loom.alloc [1, 64, 128] on @L1 : memref<1x64x128xf32>
          %17 = loom.semaphore_take %16 : memref<1x64x128xf32> -> memref<1x64x128xf32>
          %18 = loom.alloc [1, 128, 128] on @L1 : memref<1x128x128xf32>
          %19 = loom.semaphore_take %18 : memref<1x128x128xf32> -> memref<1x128x128xf32>
          %20 = loom.alloc [1, 64] on @L1 : memref<1x64xf32>
          %21 = loom.semaphore_take %20 : memref<1x64xf32> -> memref<1x64xf32>
          %22 = loom.alloc [1, 64] on @L1 : memref<1x64xf32>
          %23 = loom.semaphore_take %22 : memref<1x64xf32> -> memref<1x64xf32>
          %24 = loom.alloc [1, 64] on @L1 : memref<1x64xf32>
          %25 = loom.semaphore_take %24 : memref<1x64xf32> -> memref<1x64xf32>
          %26 = loom.alloc [1, 64] on @L1 : memref<1x64xf32>
          %27 = loom.semaphore_take %26 : memref<1x64xf32> -> memref<1x64xf32>
          %28 = loom.alloc [1, 64] on @L1 : memref<1x64xf32>
          %29 = loom.semaphore_take %28 : memref<1x64xf32> -> memref<1x64xf32>
          %30 = loom.alloc [1, 64, 128] on @L1 : memref<1x64x128xf32>
          %31 = loom.semaphore_take %30 : memref<1x64x128xf32> -> memref<1x64x128xf32>
          %32 = loom.semaphore_take %30 : memref<1x64x128xf32> -> memref<1x64x128xf32>
          %33 = loom.alloc [1, 64, 128] on @L1 : memref<1x64x128xf32>
          %34 = loom.semaphore_take %33 : memref<1x64x128xf32> -> memref<1x64x128xf32>
          %35 = loom.alloc [1, 64, 128] on @L1 : memref<1x64x128xf32>
          %36 = loom.semaphore_take %35 : memref<1x64x128xf32> -> memref<1x64x128xf32>
          %37 = arith.muli %arg6, %c524288 overflow<nsw> : index
          %38 = arith.muli %13, %c8192 : index
          %39 = arith.addi %37, %38 : index
          %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%39], sizes: [1, 64, 128], strides: [524288, 128, 1] : memref<32x4096x128xf32> to memref<1x64x128xf32, strided<[524288, 128, 1], offset: ?>>
          loom.copy %reinterpret_cast, %32 src_mem_space @DRAM dst_mem_space @L1, interconnect : [], broadcast : [1, 1] : memref<1x64x128xf32, strided<[524288, 128, 1], offset: ?>>, memref<1x64x128xf32>
          linalg.fill ins(%cst_1 : f32) outs(%34 : memref<1x64x128xf32>)
          linalg.fill ins(%cst_2 : f32) outs(%21 : memref<1x64xf32>)
          linalg.fill ins(%cst_3 : f32) outs(%23 : memref<1x64xf32>)
          scf.for %arg7 = %c0 to %c32 step %c1 {
            %40 = arith.muli %arg7, %c128 : index
            %41 = arith.addi %37, %40 : index
            %reinterpret_cast_5 = memref.reinterpret_cast %arg0 to offset: [%41], sizes: [1, 128, 128], strides: [524288, 4096, 1] : memref<32x128x4096xf32> to memref<1x128x128xf32, strided<[524288, 4096, 1], offset: ?>>
            loom.copy %reinterpret_cast_5, %19 src_mem_space @DRAM dst_mem_space @L1, interconnect : [@horizontal_links, @vertical_links], broadcast : [8, 8] : memref<1x128x128xf32, strided<[524288, 4096, 1], offset: ?>>, memref<1x128x128xf32>
            linalg.fill ins(%cst_1 : f32) outs(%17 : memref<1x64x128xf32>)
            linalg.batch_matmul ins(%32, %19 : memref<1x64x128xf32>, memref<1x128x128xf32>) outs(%17 : memref<1x64x128xf32>)
            loom.semaphore_give %19 : memref<1x128x128xf32>
            linalg.fill ins(%cst_3 : f32) outs(%25 : memref<1x64xf32>)
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%17 : memref<1x64x128xf32>) outs(%25 : memref<1x64xf32>) {
            ^bb0(%in: f32, %out: f32):
              %44 = arith.maximumf %in, %out : f32
              linalg.yield %44 : f32
            }
            linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%23, %25 : memref<1x64xf32>, memref<1x64xf32>) outs(%25 : memref<1x64xf32>) {
            ^bb0(%in: f32, %in_7: f32, %out: f32):
              %44 = arith.truncf %cst_0 : f64 to f32
              %45 = arith.mulf %in_7, %44 : f32
              %46 = arith.cmpf ogt, %in, %45 : f32
              %47 = arith.select %46, %in, %45 : f32
              linalg.yield %47 : f32
            }
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%17, %25 : memref<1x64x128xf32>, memref<1x64xf32>) outs(%17 : memref<1x64x128xf32>) {
            ^bb0(%in: f32, %in_7: f32, %out: f32):
              %44 = arith.truncf %cst_0 : f64 to f32
              %45 = arith.mulf %in, %44 : f32
              %46 = arith.subf %45, %in_7 : f32
              %47 = math.powf %cst, %46 : f32
              linalg.yield %47 : f32
            }
            linalg.fill ins(%cst_1 : f32) outs(%27 : memref<1x64xf32>)
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%17 : memref<1x64x128xf32>) outs(%27 : memref<1x64xf32>) {
            ^bb0(%in: f32, %out: f32):
              %44 = arith.addf %in, %out : f32
              linalg.yield %44 : f32
            }
            linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%23, %25 : memref<1x64xf32>, memref<1x64xf32>) outs(%29 : memref<1x64xf32>) {
            ^bb0(%in: f32, %in_7: f32, %out: f32):
              %44 = arith.subf %in, %in_7 : f32
              %45 = math.powf %cst, %44 : f32
              linalg.yield %45 : f32
            }
            linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%21, %29, %27 : memref<1x64xf32>, memref<1x64xf32>, memref<1x64xf32>) outs(%21 : memref<1x64xf32>) {
            ^bb0(%in: f32, %in_7: f32, %in_8: f32, %out: f32):
              %44 = arith.mulf %in, %in_7 : f32
              %45 = arith.addf %44, %in_8 : f32
              linalg.yield %45 : f32
            }
            loom.semaphore_give %27 : memref<1x64xf32>
            %42 = arith.muli %arg7, %c16384 : index
            %43 = arith.addi %37, %42 : index
            %reinterpret_cast_6 = memref.reinterpret_cast %arg1 to offset: [%43], sizes: [1, 128, 128], strides: [524288, 128, 1] : memref<32x4096x128xf32> to memref<1x128x128xf32, strided<[524288, 128, 1], offset: ?>>
            loom.copy %reinterpret_cast_6, %15 src_mem_space @DRAM dst_mem_space @L1, interconnect : [], broadcast : [1, 1] : memref<1x128x128xf32, strided<[524288, 128, 1], offset: ?>>, memref<1x128x128xf32>
            linalg.fill ins(%cst_1 : f32) outs(%36 : memref<1x64x128xf32>)
            linalg.batch_matmul ins(%17, %15 : memref<1x64x128xf32>, memref<1x128x128xf32>) outs(%36 : memref<1x64x128xf32>)
            loom.semaphore_give %15 : memref<1x128x128xf32>
            loom.semaphore_give %17 : memref<1x64x128xf32>
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%36, %34, %29 : memref<1x64x128xf32>, memref<1x64x128xf32>, memref<1x64xf32>) outs(%34 : memref<1x64x128xf32>) {
            ^bb0(%in: f32, %in_7: f32, %in_8: f32, %out: f32):
              %44 = arith.mulf %in_7, %in_8 : f32
              %45 = arith.addf %in, %44 : f32
              linalg.yield %45 : f32
            }
            loom.semaphore_give %29 : memref<1x64xf32>
            loom.semaphore_give %36 : memref<1x64x128xf32>
            linalg.copy ins(%25 : memref<1x64xf32>) outs(%23 : memref<1x64xf32>)
            loom.semaphore_give %25 : memref<1x64xf32>
          }
          loom.semaphore_give %23 : memref<1x64xf32>
          loom.semaphore_give %32 : memref<1x64x128xf32>
          linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%34, %21 : memref<1x64x128xf32>, memref<1x64xf32>) outs(%31 : memref<1x64x128xf32>) {
          ^bb0(%in: f32, %in_5: f32, %out: f32):
            %40 = arith.divf %in, %in_5 : f32
            linalg.yield %40 : f32
          }
          loom.semaphore_give %21 : memref<1x64xf32>
          loom.semaphore_give %34 : memref<1x64x128xf32>
          %reinterpret_cast_4 = memref.reinterpret_cast %arg3 to offset: [%39], sizes: [1, 64, 128], strides: [524288, 128, 1] : memref<32x4096x128xf32> to memref<1x64x128xf32, strided<[524288, 128, 1], offset: ?>>
          loom.copy %31, %reinterpret_cast_4 src_mem_space @L1 dst_mem_space @DRAM, interconnect : [], broadcast : [1, 1] : memref<1x64x128xf32>, memref<1x64x128xf32, strided<[524288, 128, 1], offset: ?>>
          loom.semaphore_give %31 : memref<1x64x128xf32>
        }
        scf.reduce 
      } {loom.iter_types = [#loom.iter_type<spatial>, #loom.iter_type<spatial>], loom.mapped_to_dims = [@x, @y]}
      return
    }
  }
  module attributes {loom.pass_name = "Materialize"} {
    func.func @attention__d0i1_d1i1__f10__d_a_a__BB1__BM64__BN128(%arg0: memref<32x128x4096xf32>, %arg1: memref<32x4096x128xf32>, %arg2: memref<32x4096x128xf32>, %arg3: memref<32x4096x128xf32>) {
      %c16384 = arith.constant 16384 : index
      %c8192 = arith.constant 8192 : index
      %c524288 = arith.constant 524288 : index
      %c32 = arith.constant 32 : index
      %cst = arith.constant 2.000000e+00 : f32
      %cst_0 = arith.constant 0.12751743074602467 : f64
      %cst_1 = arith.constant 0.000000e+00 : f32
      %cst_2 = arith.constant 1.000000e+00 : f32
      %cst_3 = arith.constant 0xFF800000 : f32
      %c128 = arith.constant 128 : index
      %c0 = arith.constant 0 : index
      %c8 = arith.constant 8 : index
      %c1 = arith.constant 1 : index
      scf.parallel (%arg4, %arg5) = (%c0, %c0) to (%c8, %c8) step (%c1, %c1) {
        scf.for %arg6 = %c0 to %c32 step %c1 {
          %12 = arith.muli %arg4, %c8 overflow<nsw> : index
          %13 = arith.addi %12, %arg5 : index
          %14 = loom.alloc [1, 128, 128] on @L1 : memref<1x128x128xf32>
          %15 = loom.semaphore_take %14 : memref<1x128x128xf32> -> memref<1x128x128xf32>
          %16 = loom.alloc [1, 64, 128] on @L1 : memref<1x64x128xf32>
          %17 = loom.semaphore_take %16 : memref<1x64x128xf32> -> memref<1x64x128xf32>
          %18 = loom.alloc [1, 128, 128] on @L1 : memref<1x128x128xf32>
          %19 = loom.semaphore_take %18 : memref<1x128x128xf32> -> memref<1x128x128xf32>
          %20 = loom.alloc [1, 64] on @L1 : memref<1x64xf32>
          %21 = loom.semaphore_take %20 : memref<1x64xf32> -> memref<1x64xf32>
          %22 = loom.alloc [1, 64] on @L1 : memref<1x64xf32>
          %23 = loom.semaphore_take %22 : memref<1x64xf32> -> memref<1x64xf32>
          %24 = loom.alloc [1, 64] on @L1 : memref<1x64xf32>
          %25 = loom.semaphore_take %24 : memref<1x64xf32> -> memref<1x64xf32>
          %26 = loom.alloc [1, 64] on @L1 : memref<1x64xf32>
          %27 = loom.semaphore_take %26 : memref<1x64xf32> -> memref<1x64xf32>
          %28 = loom.alloc [1, 64] on @L1 : memref<1x64xf32>
          %29 = loom.semaphore_take %28 : memref<1x64xf32> -> memref<1x64xf32>
          %30 = loom.alloc [1, 64, 128] on @L1 : memref<1x64x128xf32>
          %31 = loom.semaphore_take %30 : memref<1x64x128xf32> -> memref<1x64x128xf32>
          %32 = loom.semaphore_take %30 : memref<1x64x128xf32> -> memref<1x64x128xf32>
          %33 = loom.alloc [1, 64, 128] on @L1 : memref<1x64x128xf32>
          %34 = loom.semaphore_take %33 : memref<1x64x128xf32> -> memref<1x64x128xf32>
          %35 = loom.alloc [1, 64, 128] on @L1 : memref<1x64x128xf32>
          %36 = loom.semaphore_take %35 : memref<1x64x128xf32> -> memref<1x64x128xf32>
          %37 = arith.muli %arg6, %c524288 overflow<nsw> : index
          %38 = arith.muli %13, %c8192 : index
          %39 = arith.addi %37, %38 : index
          %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%39], sizes: [1, 64, 128], strides: [524288, 128, 1] : memref<32x4096x128xf32> to memref<1x64x128xf32, strided<[524288, 128, 1], offset: ?>>
          loom.copy %reinterpret_cast, %32 src_mem_space @DRAM dst_mem_space @L1, interconnect : [], broadcast : [1, 1] : memref<1x64x128xf32, strided<[524288, 128, 1], offset: ?>>, memref<1x64x128xf32>
          linalg.fill ins(%cst_1 : f32) outs(%34 : memref<1x64x128xf32>)
          linalg.fill ins(%cst_2 : f32) outs(%21 : memref<1x64xf32>)
          linalg.fill ins(%cst_3 : f32) outs(%23 : memref<1x64xf32>)
          scf.for %arg7 = %c0 to %c32 step %c1 {
            %40 = arith.muli %arg7, %c128 : index
            %41 = arith.addi %37, %40 : index
            %reinterpret_cast_5 = memref.reinterpret_cast %arg0 to offset: [%41], sizes: [1, 128, 128], strides: [524288, 4096, 1] : memref<32x128x4096xf32> to memref<1x128x128xf32, strided<[524288, 4096, 1], offset: ?>>
            loom.copy %reinterpret_cast_5, %19 src_mem_space @DRAM dst_mem_space @L1, interconnect : [@horizontal_links, @vertical_links], broadcast : [8, 8] : memref<1x128x128xf32, strided<[524288, 4096, 1], offset: ?>>, memref<1x128x128xf32>
            linalg.fill ins(%cst_1 : f32) outs(%17 : memref<1x64x128xf32>)
            linalg.batch_matmul ins(%32, %19 : memref<1x64x128xf32>, memref<1x128x128xf32>) outs(%17 : memref<1x64x128xf32>)
            loom.semaphore_give %19 : memref<1x128x128xf32>
            linalg.fill ins(%cst_3 : f32) outs(%25 : memref<1x64xf32>)
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%17 : memref<1x64x128xf32>) outs(%25 : memref<1x64xf32>) {
            ^bb0(%in: f32, %out: f32):
              %44 = arith.maximumf %in, %out : f32
              linalg.yield %44 : f32
            }
            linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%23, %25 : memref<1x64xf32>, memref<1x64xf32>) outs(%25 : memref<1x64xf32>) {
            ^bb0(%in: f32, %in_7: f32, %out: f32):
              %44 = arith.truncf %cst_0 : f64 to f32
              %45 = arith.mulf %in_7, %44 : f32
              %46 = arith.cmpf ogt, %in, %45 : f32
              %47 = arith.select %46, %in, %45 : f32
              linalg.yield %47 : f32
            }
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%17, %25 : memref<1x64x128xf32>, memref<1x64xf32>) outs(%17 : memref<1x64x128xf32>) {
            ^bb0(%in: f32, %in_7: f32, %out: f32):
              %44 = arith.truncf %cst_0 : f64 to f32
              %45 = arith.mulf %in, %44 : f32
              %46 = arith.subf %45, %in_7 : f32
              %47 = math.powf %cst, %46 : f32
              linalg.yield %47 : f32
            }
            linalg.fill ins(%cst_1 : f32) outs(%27 : memref<1x64xf32>)
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%17 : memref<1x64x128xf32>) outs(%27 : memref<1x64xf32>) {
            ^bb0(%in: f32, %out: f32):
              %44 = arith.addf %in, %out : f32
              linalg.yield %44 : f32
            }
            linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%23, %25 : memref<1x64xf32>, memref<1x64xf32>) outs(%29 : memref<1x64xf32>) {
            ^bb0(%in: f32, %in_7: f32, %out: f32):
              %44 = arith.subf %in, %in_7 : f32
              %45 = math.powf %cst, %44 : f32
              linalg.yield %45 : f32
            }
            linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%21, %29, %27 : memref<1x64xf32>, memref<1x64xf32>, memref<1x64xf32>) outs(%21 : memref<1x64xf32>) {
            ^bb0(%in: f32, %in_7: f32, %in_8: f32, %out: f32):
              %44 = arith.mulf %in, %in_7 : f32
              %45 = arith.addf %44, %in_8 : f32
              linalg.yield %45 : f32
            }
            loom.semaphore_give %27 : memref<1x64xf32>
            %42 = arith.muli %arg7, %c16384 : index
            %43 = arith.addi %37, %42 : index
            %reinterpret_cast_6 = memref.reinterpret_cast %arg1 to offset: [%43], sizes: [1, 128, 128], strides: [524288, 128, 1] : memref<32x4096x128xf32> to memref<1x128x128xf32, strided<[524288, 128, 1], offset: ?>>
            loom.copy %reinterpret_cast_6, %15 src_mem_space @DRAM dst_mem_space @L1, interconnect : [@horizontal_links, @vertical_links], broadcast : [8, 8] : memref<1x128x128xf32, strided<[524288, 128, 1], offset: ?>>, memref<1x128x128xf32>
            linalg.fill ins(%cst_1 : f32) outs(%36 : memref<1x64x128xf32>)
            linalg.batch_matmul ins(%17, %15 : memref<1x64x128xf32>, memref<1x128x128xf32>) outs(%36 : memref<1x64x128xf32>)
            loom.semaphore_give %15 : memref<1x128x128xf32>
            loom.semaphore_give %17 : memref<1x64x128xf32>
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%36, %34, %29 : memref<1x64x128xf32>, memref<1x64x128xf32>, memref<1x64xf32>) outs(%34 : memref<1x64x128xf32>) {
            ^bb0(%in: f32, %in_7: f32, %in_8: f32, %out: f32):
              %44 = arith.mulf %in_7, %in_8 : f32
              %45 = arith.addf %in, %44 : f32
              linalg.yield %45 : f32
            }
            loom.semaphore_give %29 : memref<1x64xf32>
            loom.semaphore_give %36 : memref<1x64x128xf32>
            linalg.copy ins(%25 : memref<1x64xf32>) outs(%23 : memref<1x64xf32>)
            loom.semaphore_give %25 : memref<1x64xf32>
          }
          loom.semaphore_give %23 : memref<1x64xf32>
          loom.semaphore_give %32 : memref<1x64x128xf32>
          linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%34, %21 : memref<1x64x128xf32>, memref<1x64xf32>) outs(%31 : memref<1x64x128xf32>) {
          ^bb0(%in: f32, %in_5: f32, %out: f32):
            %40 = arith.divf %in, %in_5 : f32
            linalg.yield %40 : f32
          }
          loom.semaphore_give %21 : memref<1x64xf32>
          loom.semaphore_give %34 : memref<1x64x128xf32>
          %reinterpret_cast_4 = memref.reinterpret_cast %arg3 to offset: [%39], sizes: [1, 64, 128], strides: [524288, 128, 1] : memref<32x4096x128xf32> to memref<1x64x128xf32, strided<[524288, 128, 1], offset: ?>>
          loom.copy %31, %reinterpret_cast_4 src_mem_space @L1 dst_mem_space @DRAM, interconnect : [], broadcast : [1, 1] : memref<1x64x128xf32>, memref<1x64x128xf32, strided<[524288, 128, 1], offset: ?>>
          loom.semaphore_give %31 : memref<1x64x128xf32>
        }
        scf.reduce 
      } {loom.iter_types = [#loom.iter_type<spatial>, #loom.iter_type<spatial>], loom.mapped_to_dims = [@x, @y]}
      return
    }
  }
  module attributes {loom.pass_name = "Materialize"} {
    func.func @attention__d0i1_d1i1__f10__d_a_h__BB1__BM64__BN128(%arg0: memref<32x128x4096xf32>, %arg1: memref<32x4096x128xf32>, %arg2: memref<32x4096x128xf32>, %arg3: memref<32x4096x128xf32>) {
      %c16384 = arith.constant 16384 : index
      %c8192 = arith.constant 8192 : index
      %c524288 = arith.constant 524288 : index
      %c32 = arith.constant 32 : index
      %cst = arith.constant 2.000000e+00 : f32
      %cst_0 = arith.constant 0.12751743074602467 : f64
      %cst_1 = arith.constant 0.000000e+00 : f32
      %cst_2 = arith.constant 1.000000e+00 : f32
      %cst_3 = arith.constant 0xFF800000 : f32
      %c128 = arith.constant 128 : index
      %c0 = arith.constant 0 : index
      %c8 = arith.constant 8 : index
      %c1 = arith.constant 1 : index
      scf.parallel (%arg4, %arg5) = (%c0, %c0) to (%c8, %c8) step (%c1, %c1) {
        scf.for %arg6 = %c0 to %c32 step %c1 {
          %12 = arith.muli %arg4, %c8 overflow<nsw> : index
          %13 = arith.addi %12, %arg5 : index
          %14 = loom.alloc [1, 128, 128] on @L1 : memref<1x128x128xf32>
          %15 = loom.semaphore_take %14 : memref<1x128x128xf32> -> memref<1x128x128xf32>
          %16 = loom.alloc [1, 64, 128] on @L1 : memref<1x64x128xf32>
          %17 = loom.semaphore_take %16 : memref<1x64x128xf32> -> memref<1x64x128xf32>
          %18 = loom.alloc [1, 128, 128] on @L1 : memref<1x128x128xf32>
          %19 = loom.semaphore_take %18 : memref<1x128x128xf32> -> memref<1x128x128xf32>
          %20 = loom.alloc [1, 64] on @L1 : memref<1x64xf32>
          %21 = loom.semaphore_take %20 : memref<1x64xf32> -> memref<1x64xf32>
          %22 = loom.alloc [1, 64] on @L1 : memref<1x64xf32>
          %23 = loom.semaphore_take %22 : memref<1x64xf32> -> memref<1x64xf32>
          %24 = loom.alloc [1, 64] on @L1 : memref<1x64xf32>
          %25 = loom.semaphore_take %24 : memref<1x64xf32> -> memref<1x64xf32>
          %26 = loom.alloc [1, 64] on @L1 : memref<1x64xf32>
          %27 = loom.semaphore_take %26 : memref<1x64xf32> -> memref<1x64xf32>
          %28 = loom.alloc [1, 64] on @L1 : memref<1x64xf32>
          %29 = loom.semaphore_take %28 : memref<1x64xf32> -> memref<1x64xf32>
          %30 = loom.alloc [1, 64, 128] on @L1 : memref<1x64x128xf32>
          %31 = loom.semaphore_take %30 : memref<1x64x128xf32> -> memref<1x64x128xf32>
          %32 = loom.semaphore_take %30 : memref<1x64x128xf32> -> memref<1x64x128xf32>
          %33 = loom.alloc [1, 64, 128] on @L1 : memref<1x64x128xf32>
          %34 = loom.semaphore_take %33 : memref<1x64x128xf32> -> memref<1x64x128xf32>
          %35 = loom.alloc [1, 64, 128] on @L1 : memref<1x64x128xf32>
          %36 = loom.semaphore_take %35 : memref<1x64x128xf32> -> memref<1x64x128xf32>
          %37 = arith.muli %arg6, %c524288 overflow<nsw> : index
          %38 = arith.muli %13, %c8192 : index
          %39 = arith.addi %37, %38 : index
          %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%39], sizes: [1, 64, 128], strides: [524288, 128, 1] : memref<32x4096x128xf32> to memref<1x64x128xf32, strided<[524288, 128, 1], offset: ?>>
          loom.copy %reinterpret_cast, %32 src_mem_space @DRAM dst_mem_space @L1, interconnect : [], broadcast : [1, 1] : memref<1x64x128xf32, strided<[524288, 128, 1], offset: ?>>, memref<1x64x128xf32>
          linalg.fill ins(%cst_1 : f32) outs(%34 : memref<1x64x128xf32>)
          linalg.fill ins(%cst_2 : f32) outs(%21 : memref<1x64xf32>)
          linalg.fill ins(%cst_3 : f32) outs(%23 : memref<1x64xf32>)
          scf.for %arg7 = %c0 to %c32 step %c1 {
            %40 = arith.muli %arg7, %c128 : index
            %41 = arith.addi %37, %40 : index
            %reinterpret_cast_5 = memref.reinterpret_cast %arg0 to offset: [%41], sizes: [1, 128, 128], strides: [524288, 4096, 1] : memref<32x128x4096xf32> to memref<1x128x128xf32, strided<[524288, 4096, 1], offset: ?>>
            loom.copy %reinterpret_cast_5, %19 src_mem_space @DRAM dst_mem_space @L1, interconnect : [@horizontal_links, @vertical_links], broadcast : [8, 8] : memref<1x128x128xf32, strided<[524288, 4096, 1], offset: ?>>, memref<1x128x128xf32>
            linalg.fill ins(%cst_1 : f32) outs(%17 : memref<1x64x128xf32>)
            linalg.batch_matmul ins(%32, %19 : memref<1x64x128xf32>, memref<1x128x128xf32>) outs(%17 : memref<1x64x128xf32>)
            loom.semaphore_give %19 : memref<1x128x128xf32>
            linalg.fill ins(%cst_3 : f32) outs(%25 : memref<1x64xf32>)
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%17 : memref<1x64x128xf32>) outs(%25 : memref<1x64xf32>) {
            ^bb0(%in: f32, %out: f32):
              %44 = arith.maximumf %in, %out : f32
              linalg.yield %44 : f32
            }
            linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%23, %25 : memref<1x64xf32>, memref<1x64xf32>) outs(%25 : memref<1x64xf32>) {
            ^bb0(%in: f32, %in_7: f32, %out: f32):
              %44 = arith.truncf %cst_0 : f64 to f32
              %45 = arith.mulf %in_7, %44 : f32
              %46 = arith.cmpf ogt, %in, %45 : f32
              %47 = arith.select %46, %in, %45 : f32
              linalg.yield %47 : f32
            }
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%17, %25 : memref<1x64x128xf32>, memref<1x64xf32>) outs(%17 : memref<1x64x128xf32>) {
            ^bb0(%in: f32, %in_7: f32, %out: f32):
              %44 = arith.truncf %cst_0 : f64 to f32
              %45 = arith.mulf %in, %44 : f32
              %46 = arith.subf %45, %in_7 : f32
              %47 = math.powf %cst, %46 : f32
              linalg.yield %47 : f32
            }
            linalg.fill ins(%cst_1 : f32) outs(%27 : memref<1x64xf32>)
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%17 : memref<1x64x128xf32>) outs(%27 : memref<1x64xf32>) {
            ^bb0(%in: f32, %out: f32):
              %44 = arith.addf %in, %out : f32
              linalg.yield %44 : f32
            }
            linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%23, %25 : memref<1x64xf32>, memref<1x64xf32>) outs(%29 : memref<1x64xf32>) {
            ^bb0(%in: f32, %in_7: f32, %out: f32):
              %44 = arith.subf %in, %in_7 : f32
              %45 = math.powf %cst, %44 : f32
              linalg.yield %45 : f32
            }
            linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%21, %29, %27 : memref<1x64xf32>, memref<1x64xf32>, memref<1x64xf32>) outs(%21 : memref<1x64xf32>) {
            ^bb0(%in: f32, %in_7: f32, %in_8: f32, %out: f32):
              %44 = arith.mulf %in, %in_7 : f32
              %45 = arith.addf %44, %in_8 : f32
              linalg.yield %45 : f32
            }
            loom.semaphore_give %27 : memref<1x64xf32>
            %42 = arith.muli %arg7, %c16384 : index
            %43 = arith.addi %37, %42 : index
            %reinterpret_cast_6 = memref.reinterpret_cast %arg1 to offset: [%43], sizes: [1, 128, 128], strides: [524288, 128, 1] : memref<32x4096x128xf32> to memref<1x128x128xf32, strided<[524288, 128, 1], offset: ?>>
            loom.copy %reinterpret_cast_6, %15 src_mem_space @DRAM dst_mem_space @L1, interconnect : [@horizontal_links], broadcast : [1, 8] : memref<1x128x128xf32, strided<[524288, 128, 1], offset: ?>>, memref<1x128x128xf32>
            linalg.fill ins(%cst_1 : f32) outs(%36 : memref<1x64x128xf32>)
            linalg.batch_matmul ins(%17, %15 : memref<1x64x128xf32>, memref<1x128x128xf32>) outs(%36 : memref<1x64x128xf32>)
            loom.semaphore_give %15 : memref<1x128x128xf32>
            loom.semaphore_give %17 : memref<1x64x128xf32>
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%36, %34, %29 : memref<1x64x128xf32>, memref<1x64x128xf32>, memref<1x64xf32>) outs(%34 : memref<1x64x128xf32>) {
            ^bb0(%in: f32, %in_7: f32, %in_8: f32, %out: f32):
              %44 = arith.mulf %in_7, %in_8 : f32
              %45 = arith.addf %in, %44 : f32
              linalg.yield %45 : f32
            }
            loom.semaphore_give %29 : memref<1x64xf32>
            loom.semaphore_give %36 : memref<1x64x128xf32>
            linalg.copy ins(%25 : memref<1x64xf32>) outs(%23 : memref<1x64xf32>)
            loom.semaphore_give %25 : memref<1x64xf32>
          }
          loom.semaphore_give %23 : memref<1x64xf32>
          loom.semaphore_give %32 : memref<1x64x128xf32>
          linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%34, %21 : memref<1x64x128xf32>, memref<1x64xf32>) outs(%31 : memref<1x64x128xf32>) {
          ^bb0(%in: f32, %in_5: f32, %out: f32):
            %40 = arith.divf %in, %in_5 : f32
            linalg.yield %40 : f32
          }
          loom.semaphore_give %21 : memref<1x64xf32>
          loom.semaphore_give %34 : memref<1x64x128xf32>
          %reinterpret_cast_4 = memref.reinterpret_cast %arg3 to offset: [%39], sizes: [1, 64, 128], strides: [524288, 128, 1] : memref<32x4096x128xf32> to memref<1x64x128xf32, strided<[524288, 128, 1], offset: ?>>
          loom.copy %31, %reinterpret_cast_4 src_mem_space @L1 dst_mem_space @DRAM, interconnect : [], broadcast : [1, 1] : memref<1x64x128xf32>, memref<1x64x128xf32, strided<[524288, 128, 1], offset: ?>>
          loom.semaphore_give %31 : memref<1x64x128xf32>
        }
        scf.reduce 
      } {loom.iter_types = [#loom.iter_type<spatial>, #loom.iter_type<spatial>], loom.mapped_to_dims = [@x, @y]}
      return
    }
  }
  module attributes {loom.pass_name = "Materialize"} {
    func.func @attention__d0i1_d1i1__f10__d_a_v__BB1__BM64__BN128(%arg0: memref<32x128x4096xf32>, %arg1: memref<32x4096x128xf32>, %arg2: memref<32x4096x128xf32>, %arg3: memref<32x4096x128xf32>) {
      %c16384 = arith.constant 16384 : index
      %c8192 = arith.constant 8192 : index
      %c524288 = arith.constant 524288 : index
      %c32 = arith.constant 32 : index
      %cst = arith.constant 2.000000e+00 : f32
      %cst_0 = arith.constant 0.12751743074602467 : f64
      %cst_1 = arith.constant 0.000000e+00 : f32
      %cst_2 = arith.constant 1.000000e+00 : f32
      %cst_3 = arith.constant 0xFF800000 : f32
      %c128 = arith.constant 128 : index
      %c0 = arith.constant 0 : index
      %c8 = arith.constant 8 : index
      %c1 = arith.constant 1 : index
      scf.parallel (%arg4, %arg5) = (%c0, %c0) to (%c8, %c8) step (%c1, %c1) {
        scf.for %arg6 = %c0 to %c32 step %c1 {
          %12 = arith.muli %arg4, %c8 overflow<nsw> : index
          %13 = arith.addi %12, %arg5 : index
          %14 = loom.alloc [1, 128, 128] on @L1 : memref<1x128x128xf32>
          %15 = loom.semaphore_take %14 : memref<1x128x128xf32> -> memref<1x128x128xf32>
          %16 = loom.alloc [1, 64, 128] on @L1 : memref<1x64x128xf32>
          %17 = loom.semaphore_take %16 : memref<1x64x128xf32> -> memref<1x64x128xf32>
          %18 = loom.alloc [1, 128, 128] on @L1 : memref<1x128x128xf32>
          %19 = loom.semaphore_take %18 : memref<1x128x128xf32> -> memref<1x128x128xf32>
          %20 = loom.alloc [1, 64] on @L1 : memref<1x64xf32>
          %21 = loom.semaphore_take %20 : memref<1x64xf32> -> memref<1x64xf32>
          %22 = loom.alloc [1, 64] on @L1 : memref<1x64xf32>
          %23 = loom.semaphore_take %22 : memref<1x64xf32> -> memref<1x64xf32>
          %24 = loom.alloc [1, 64] on @L1 : memref<1x64xf32>
          %25 = loom.semaphore_take %24 : memref<1x64xf32> -> memref<1x64xf32>
          %26 = loom.alloc [1, 64] on @L1 : memref<1x64xf32>
          %27 = loom.semaphore_take %26 : memref<1x64xf32> -> memref<1x64xf32>
          %28 = loom.alloc [1, 64] on @L1 : memref<1x64xf32>
          %29 = loom.semaphore_take %28 : memref<1x64xf32> -> memref<1x64xf32>
          %30 = loom.alloc [1, 64, 128] on @L1 : memref<1x64x128xf32>
          %31 = loom.semaphore_take %30 : memref<1x64x128xf32> -> memref<1x64x128xf32>
          %32 = loom.semaphore_take %30 : memref<1x64x128xf32> -> memref<1x64x128xf32>
          %33 = loom.alloc [1, 64, 128] on @L1 : memref<1x64x128xf32>
          %34 = loom.semaphore_take %33 : memref<1x64x128xf32> -> memref<1x64x128xf32>
          %35 = loom.alloc [1, 64, 128] on @L1 : memref<1x64x128xf32>
          %36 = loom.semaphore_take %35 : memref<1x64x128xf32> -> memref<1x64x128xf32>
          %37 = arith.muli %arg6, %c524288 overflow<nsw> : index
          %38 = arith.muli %13, %c8192 : index
          %39 = arith.addi %37, %38 : index
          %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%39], sizes: [1, 64, 128], strides: [524288, 128, 1] : memref<32x4096x128xf32> to memref<1x64x128xf32, strided<[524288, 128, 1], offset: ?>>
          loom.copy %reinterpret_cast, %32 src_mem_space @DRAM dst_mem_space @L1, interconnect : [], broadcast : [1, 1] : memref<1x64x128xf32, strided<[524288, 128, 1], offset: ?>>, memref<1x64x128xf32>
          linalg.fill ins(%cst_1 : f32) outs(%34 : memref<1x64x128xf32>)
          linalg.fill ins(%cst_2 : f32) outs(%21 : memref<1x64xf32>)
          linalg.fill ins(%cst_3 : f32) outs(%23 : memref<1x64xf32>)
          scf.for %arg7 = %c0 to %c32 step %c1 {
            %40 = arith.muli %arg7, %c128 : index
            %41 = arith.addi %37, %40 : index
            %reinterpret_cast_5 = memref.reinterpret_cast %arg0 to offset: [%41], sizes: [1, 128, 128], strides: [524288, 4096, 1] : memref<32x128x4096xf32> to memref<1x128x128xf32, strided<[524288, 4096, 1], offset: ?>>
            loom.copy %reinterpret_cast_5, %19 src_mem_space @DRAM dst_mem_space @L1, interconnect : [@horizontal_links, @vertical_links], broadcast : [8, 8] : memref<1x128x128xf32, strided<[524288, 4096, 1], offset: ?>>, memref<1x128x128xf32>
            linalg.fill ins(%cst_1 : f32) outs(%17 : memref<1x64x128xf32>)
            linalg.batch_matmul ins(%32, %19 : memref<1x64x128xf32>, memref<1x128x128xf32>) outs(%17 : memref<1x64x128xf32>)
            loom.semaphore_give %19 : memref<1x128x128xf32>
            linalg.fill ins(%cst_3 : f32) outs(%25 : memref<1x64xf32>)
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%17 : memref<1x64x128xf32>) outs(%25 : memref<1x64xf32>) {
            ^bb0(%in: f32, %out: f32):
              %44 = arith.maximumf %in, %out : f32
              linalg.yield %44 : f32
            }
            linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%23, %25 : memref<1x64xf32>, memref<1x64xf32>) outs(%25 : memref<1x64xf32>) {
            ^bb0(%in: f32, %in_7: f32, %out: f32):
              %44 = arith.truncf %cst_0 : f64 to f32
              %45 = arith.mulf %in_7, %44 : f32
              %46 = arith.cmpf ogt, %in, %45 : f32
              %47 = arith.select %46, %in, %45 : f32
              linalg.yield %47 : f32
            }
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%17, %25 : memref<1x64x128xf32>, memref<1x64xf32>) outs(%17 : memref<1x64x128xf32>) {
            ^bb0(%in: f32, %in_7: f32, %out: f32):
              %44 = arith.truncf %cst_0 : f64 to f32
              %45 = arith.mulf %in, %44 : f32
              %46 = arith.subf %45, %in_7 : f32
              %47 = math.powf %cst, %46 : f32
              linalg.yield %47 : f32
            }
            linalg.fill ins(%cst_1 : f32) outs(%27 : memref<1x64xf32>)
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%17 : memref<1x64x128xf32>) outs(%27 : memref<1x64xf32>) {
            ^bb0(%in: f32, %out: f32):
              %44 = arith.addf %in, %out : f32
              linalg.yield %44 : f32
            }
            linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%23, %25 : memref<1x64xf32>, memref<1x64xf32>) outs(%29 : memref<1x64xf32>) {
            ^bb0(%in: f32, %in_7: f32, %out: f32):
              %44 = arith.subf %in, %in_7 : f32
              %45 = math.powf %cst, %44 : f32
              linalg.yield %45 : f32
            }
            linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%21, %29, %27 : memref<1x64xf32>, memref<1x64xf32>, memref<1x64xf32>) outs(%21 : memref<1x64xf32>) {
            ^bb0(%in: f32, %in_7: f32, %in_8: f32, %out: f32):
              %44 = arith.mulf %in, %in_7 : f32
              %45 = arith.addf %44, %in_8 : f32
              linalg.yield %45 : f32
            }
            loom.semaphore_give %27 : memref<1x64xf32>
            %42 = arith.muli %arg7, %c16384 : index
            %43 = arith.addi %37, %42 : index
            %reinterpret_cast_6 = memref.reinterpret_cast %arg1 to offset: [%43], sizes: [1, 128, 128], strides: [524288, 128, 1] : memref<32x4096x128xf32> to memref<1x128x128xf32, strided<[524288, 128, 1], offset: ?>>
            loom.copy %reinterpret_cast_6, %15 src_mem_space @DRAM dst_mem_space @L1, interconnect : [@vertical_links], broadcast : [8, 1] : memref<1x128x128xf32, strided<[524288, 128, 1], offset: ?>>, memref<1x128x128xf32>
            linalg.fill ins(%cst_1 : f32) outs(%36 : memref<1x64x128xf32>)
            linalg.batch_matmul ins(%17, %15 : memref<1x64x128xf32>, memref<1x128x128xf32>) outs(%36 : memref<1x64x128xf32>)
            loom.semaphore_give %15 : memref<1x128x128xf32>
            loom.semaphore_give %17 : memref<1x64x128xf32>
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%36, %34, %29 : memref<1x64x128xf32>, memref<1x64x128xf32>, memref<1x64xf32>) outs(%34 : memref<1x64x128xf32>) {
            ^bb0(%in: f32, %in_7: f32, %in_8: f32, %out: f32):
              %44 = arith.mulf %in_7, %in_8 : f32
              %45 = arith.addf %in, %44 : f32
              linalg.yield %45 : f32
            }
            loom.semaphore_give %29 : memref<1x64xf32>
            loom.semaphore_give %36 : memref<1x64x128xf32>
            linalg.copy ins(%25 : memref<1x64xf32>) outs(%23 : memref<1x64xf32>)
            loom.semaphore_give %25 : memref<1x64xf32>
          }
          loom.semaphore_give %23 : memref<1x64xf32>
          loom.semaphore_give %32 : memref<1x64x128xf32>
          linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%34, %21 : memref<1x64x128xf32>, memref<1x64xf32>) outs(%31 : memref<1x64x128xf32>) {
          ^bb0(%in: f32, %in_5: f32, %out: f32):
            %40 = arith.divf %in, %in_5 : f32
            linalg.yield %40 : f32
          }
          loom.semaphore_give %21 : memref<1x64xf32>
          loom.semaphore_give %34 : memref<1x64x128xf32>
          %reinterpret_cast_4 = memref.reinterpret_cast %arg3 to offset: [%39], sizes: [1, 64, 128], strides: [524288, 128, 1] : memref<32x4096x128xf32> to memref<1x64x128xf32, strided<[524288, 128, 1], offset: ?>>
          loom.copy %31, %reinterpret_cast_4 src_mem_space @L1 dst_mem_space @DRAM, interconnect : [], broadcast : [1, 1] : memref<1x64x128xf32>, memref<1x64x128xf32, strided<[524288, 128, 1], offset: ?>>
          loom.semaphore_give %31 : memref<1x64x128xf32>
        }
        scf.reduce 
      } {loom.iter_types = [#loom.iter_type<spatial>, #loom.iter_type<spatial>], loom.mapped_to_dims = [@x, @y]}
      return
    }
  }
  module attributes {loom.pass_name = "Materialize"} {
    func.func @attention__d0i1_d1i1__f10__d_h_d__BB1__BM64__BN128(%arg0: memref<32x128x4096xf32>, %arg1: memref<32x4096x128xf32>, %arg2: memref<32x4096x128xf32>, %arg3: memref<32x4096x128xf32>) {
      %c16384 = arith.constant 16384 : index
      %c8192 = arith.constant 8192 : index
      %c524288 = arith.constant 524288 : index
      %c32 = arith.constant 32 : index
      %cst = arith.constant 2.000000e+00 : f32
      %cst_0 = arith.constant 0.12751743074602467 : f64
      %cst_1 = arith.constant 0.000000e+00 : f32
      %cst_2 = arith.constant 1.000000e+00 : f32
      %cst_3 = arith.constant 0xFF800000 : f32
      %c128 = arith.constant 128 : index
      %c0 = arith.constant 0 : index
      %c8 = arith.constant 8 : index
      %c1 = arith.constant 1 : index
      scf.parallel (%arg4, %arg5) = (%c0, %c0) to (%c8, %c8) step (%c1, %c1) {
        scf.for %arg6 = %c0 to %c32 step %c1 {
          %12 = arith.muli %arg4, %c8 overflow<nsw> : index
          %13 = arith.addi %12, %arg5 : index
          %14 = loom.alloc [1, 128, 128] on @L1 : memref<1x128x128xf32>
          %15 = loom.semaphore_take %14 : memref<1x128x128xf32> -> memref<1x128x128xf32>
          %16 = loom.alloc [1, 64, 128] on @L1 : memref<1x64x128xf32>
          %17 = loom.semaphore_take %16 : memref<1x64x128xf32> -> memref<1x64x128xf32>
          %18 = loom.alloc [1, 128, 128] on @L1 : memref<1x128x128xf32>
          %19 = loom.semaphore_take %18 : memref<1x128x128xf32> -> memref<1x128x128xf32>
          %20 = loom.alloc [1, 64] on @L1 : memref<1x64xf32>
          %21 = loom.semaphore_take %20 : memref<1x64xf32> -> memref<1x64xf32>
          %22 = loom.alloc [1, 64] on @L1 : memref<1x64xf32>
          %23 = loom.semaphore_take %22 : memref<1x64xf32> -> memref<1x64xf32>
          %24 = loom.alloc [1, 64] on @L1 : memref<1x64xf32>
          %25 = loom.semaphore_take %24 : memref<1x64xf32> -> memref<1x64xf32>
          %26 = loom.alloc [1, 64] on @L1 : memref<1x64xf32>
          %27 = loom.semaphore_take %26 : memref<1x64xf32> -> memref<1x64xf32>
          %28 = loom.alloc [1, 64] on @L1 : memref<1x64xf32>
          %29 = loom.semaphore_take %28 : memref<1x64xf32> -> memref<1x64xf32>
          %30 = loom.alloc [1, 64, 128] on @L1 : memref<1x64x128xf32>
          %31 = loom.semaphore_take %30 : memref<1x64x128xf32> -> memref<1x64x128xf32>
          %32 = loom.semaphore_take %30 : memref<1x64x128xf32> -> memref<1x64x128xf32>
          %33 = loom.alloc [1, 64, 128] on @L1 : memref<1x64x128xf32>
          %34 = loom.semaphore_take %33 : memref<1x64x128xf32> -> memref<1x64x128xf32>
          %35 = loom.alloc [1, 64, 128] on @L1 : memref<1x64x128xf32>
          %36 = loom.semaphore_take %35 : memref<1x64x128xf32> -> memref<1x64x128xf32>
          %37 = arith.muli %arg6, %c524288 overflow<nsw> : index
          %38 = arith.muli %13, %c8192 : index
          %39 = arith.addi %37, %38 : index
          %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%39], sizes: [1, 64, 128], strides: [524288, 128, 1] : memref<32x4096x128xf32> to memref<1x64x128xf32, strided<[524288, 128, 1], offset: ?>>
          loom.copy %reinterpret_cast, %32 src_mem_space @DRAM dst_mem_space @L1, interconnect : [], broadcast : [1, 1] : memref<1x64x128xf32, strided<[524288, 128, 1], offset: ?>>, memref<1x64x128xf32>
          linalg.fill ins(%cst_1 : f32) outs(%34 : memref<1x64x128xf32>)
          linalg.fill ins(%cst_2 : f32) outs(%21 : memref<1x64xf32>)
          linalg.fill ins(%cst_3 : f32) outs(%23 : memref<1x64xf32>)
          scf.for %arg7 = %c0 to %c32 step %c1 {
            %40 = arith.muli %arg7, %c128 : index
            %41 = arith.addi %37, %40 : index
            %reinterpret_cast_5 = memref.reinterpret_cast %arg0 to offset: [%41], sizes: [1, 128, 128], strides: [524288, 4096, 1] : memref<32x128x4096xf32> to memref<1x128x128xf32, strided<[524288, 4096, 1], offset: ?>>
            loom.copy %reinterpret_cast_5, %19 src_mem_space @DRAM dst_mem_space @L1, interconnect : [@horizontal_links], broadcast : [1, 8] : memref<1x128x128xf32, strided<[524288, 4096, 1], offset: ?>>, memref<1x128x128xf32>
            linalg.fill ins(%cst_1 : f32) outs(%17 : memref<1x64x128xf32>)
            linalg.batch_matmul ins(%32, %19 : memref<1x64x128xf32>, memref<1x128x128xf32>) outs(%17 : memref<1x64x128xf32>)
            loom.semaphore_give %19 : memref<1x128x128xf32>
            linalg.fill ins(%cst_3 : f32) outs(%25 : memref<1x64xf32>)
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%17 : memref<1x64x128xf32>) outs(%25 : memref<1x64xf32>) {
            ^bb0(%in: f32, %out: f32):
              %44 = arith.maximumf %in, %out : f32
              linalg.yield %44 : f32
            }
            linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%23, %25 : memref<1x64xf32>, memref<1x64xf32>) outs(%25 : memref<1x64xf32>) {
            ^bb0(%in: f32, %in_7: f32, %out: f32):
              %44 = arith.truncf %cst_0 : f64 to f32
              %45 = arith.mulf %in_7, %44 : f32
              %46 = arith.cmpf ogt, %in, %45 : f32
              %47 = arith.select %46, %in, %45 : f32
              linalg.yield %47 : f32
            }
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%17, %25 : memref<1x64x128xf32>, memref<1x64xf32>) outs(%17 : memref<1x64x128xf32>) {
            ^bb0(%in: f32, %in_7: f32, %out: f32):
              %44 = arith.truncf %cst_0 : f64 to f32
              %45 = arith.mulf %in, %44 : f32
              %46 = arith.subf %45, %in_7 : f32
              %47 = math.powf %cst, %46 : f32
              linalg.yield %47 : f32
            }
            linalg.fill ins(%cst_1 : f32) outs(%27 : memref<1x64xf32>)
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%17 : memref<1x64x128xf32>) outs(%27 : memref<1x64xf32>) {
            ^bb0(%in: f32, %out: f32):
              %44 = arith.addf %in, %out : f32
              linalg.yield %44 : f32
            }
            linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%23, %25 : memref<1x64xf32>, memref<1x64xf32>) outs(%29 : memref<1x64xf32>) {
            ^bb0(%in: f32, %in_7: f32, %out: f32):
              %44 = arith.subf %in, %in_7 : f32
              %45 = math.powf %cst, %44 : f32
              linalg.yield %45 : f32
            }
            linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%21, %29, %27 : memref<1x64xf32>, memref<1x64xf32>, memref<1x64xf32>) outs(%21 : memref<1x64xf32>) {
            ^bb0(%in: f32, %in_7: f32, %in_8: f32, %out: f32):
              %44 = arith.mulf %in, %in_7 : f32
              %45 = arith.addf %44, %in_8 : f32
              linalg.yield %45 : f32
            }
            loom.semaphore_give %27 : memref<1x64xf32>
            %42 = arith.muli %arg7, %c16384 : index
            %43 = arith.addi %37, %42 : index
            %reinterpret_cast_6 = memref.reinterpret_cast %arg1 to offset: [%43], sizes: [1, 128, 128], strides: [524288, 128, 1] : memref<32x4096x128xf32> to memref<1x128x128xf32, strided<[524288, 128, 1], offset: ?>>
            loom.copy %reinterpret_cast_6, %15 src_mem_space @DRAM dst_mem_space @L1, interconnect : [], broadcast : [1, 1] : memref<1x128x128xf32, strided<[524288, 128, 1], offset: ?>>, memref<1x128x128xf32>
            linalg.fill ins(%cst_1 : f32) outs(%36 : memref<1x64x128xf32>)
            linalg.batch_matmul ins(%17, %15 : memref<1x64x128xf32>, memref<1x128x128xf32>) outs(%36 : memref<1x64x128xf32>)
            loom.semaphore_give %15 : memref<1x128x128xf32>
            loom.semaphore_give %17 : memref<1x64x128xf32>
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%36, %34, %29 : memref<1x64x128xf32>, memref<1x64x128xf32>, memref<1x64xf32>) outs(%34 : memref<1x64x128xf32>) {
            ^bb0(%in: f32, %in_7: f32, %in_8: f32, %out: f32):
              %44 = arith.mulf %in_7, %in_8 : f32
              %45 = arith.addf %in, %44 : f32
              linalg.yield %45 : f32
            }
            loom.semaphore_give %29 : memref<1x64xf32>
            loom.semaphore_give %36 : memref<1x64x128xf32>
            linalg.copy ins(%25 : memref<1x64xf32>) outs(%23 : memref<1x64xf32>)
            loom.semaphore_give %25 : memref<1x64xf32>
          }
          loom.semaphore_give %23 : memref<1x64xf32>
          loom.semaphore_give %32 : memref<1x64x128xf32>
          linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%34, %21 : memref<1x64x128xf32>, memref<1x64xf32>) outs(%31 : memref<1x64x128xf32>) {
          ^bb0(%in: f32, %in_5: f32, %out: f32):
            %40 = arith.divf %in, %in_5 : f32
            linalg.yield %40 : f32
          }
          loom.semaphore_give %21 : memref<1x64xf32>
          loom.semaphore_give %34 : memref<1x64x128xf32>
          %reinterpret_cast_4 = memref.reinterpret_cast %arg3 to offset: [%39], sizes: [1, 64, 128], strides: [524288, 128, 1] : memref<32x4096x128xf32> to memref<1x64x128xf32, strided<[524288, 128, 1], offset: ?>>
          loom.copy %31, %reinterpret_cast_4 src_mem_space @L1 dst_mem_space @DRAM, interconnect : [], broadcast : [1, 1] : memref<1x64x128xf32>, memref<1x64x128xf32, strided<[524288, 128, 1], offset: ?>>
          loom.semaphore_give %31 : memref<1x64x128xf32>
        }
        scf.reduce 
      } {loom.iter_types = [#loom.iter_type<spatial>, #loom.iter_type<spatial>], loom.mapped_to_dims = [@x, @y]}
      return
    }
  }
  module attributes {loom.pass_name = "Materialize"} {
    func.func @attention__d0i1_d1i1__f10__d_h_a__BB1__BM64__BN128(%arg0: memref<32x128x4096xf32>, %arg1: memref<32x4096x128xf32>, %arg2: memref<32x4096x128xf32>, %arg3: memref<32x4096x128xf32>) {
      %c16384 = arith.constant 16384 : index
      %c8192 = arith.constant 8192 : index
      %c524288 = arith.constant 524288 : index
      %c32 = arith.constant 32 : index
      %cst = arith.constant 2.000000e+00 : f32
      %cst_0 = arith.constant 0.12751743074602467 : f64
      %cst_1 = arith.constant 0.000000e+00 : f32
      %cst_2 = arith.constant 1.000000e+00 : f32
      %cst_3 = arith.constant 0xFF800000 : f32
      %c128 = arith.constant 128 : index
      %c0 = arith.constant 0 : index
      %c8 = arith.constant 8 : index
      %c1 = arith.constant 1 : index
      scf.parallel (%arg4, %arg5) = (%c0, %c0) to (%c8, %c8) step (%c1, %c1) {
        scf.for %arg6 = %c0 to %c32 step %c1 {
          %12 = arith.muli %arg4, %c8 overflow<nsw> : index
          %13 = arith.addi %12, %arg5 : index
          %14 = loom.alloc [1, 128, 128] on @L1 : memref<1x128x128xf32>
          %15 = loom.semaphore_take %14 : memref<1x128x128xf32> -> memref<1x128x128xf32>
          %16 = loom.alloc [1, 64, 128] on @L1 : memref<1x64x128xf32>
          %17 = loom.semaphore_take %16 : memref<1x64x128xf32> -> memref<1x64x128xf32>
          %18 = loom.alloc [1, 128, 128] on @L1 : memref<1x128x128xf32>
          %19 = loom.semaphore_take %18 : memref<1x128x128xf32> -> memref<1x128x128xf32>
          %20 = loom.alloc [1, 64] on @L1 : memref<1x64xf32>
          %21 = loom.semaphore_take %20 : memref<1x64xf32> -> memref<1x64xf32>
          %22 = loom.alloc [1, 64] on @L1 : memref<1x64xf32>
          %23 = loom.semaphore_take %22 : memref<1x64xf32> -> memref<1x64xf32>
          %24 = loom.alloc [1, 64] on @L1 : memref<1x64xf32>
          %25 = loom.semaphore_take %24 : memref<1x64xf32> -> memref<1x64xf32>
          %26 = loom.alloc [1, 64] on @L1 : memref<1x64xf32>
          %27 = loom.semaphore_take %26 : memref<1x64xf32> -> memref<1x64xf32>
          %28 = loom.alloc [1, 64] on @L1 : memref<1x64xf32>
          %29 = loom.semaphore_take %28 : memref<1x64xf32> -> memref<1x64xf32>
          %30 = loom.alloc [1, 64, 128] on @L1 : memref<1x64x128xf32>
          %31 = loom.semaphore_take %30 : memref<1x64x128xf32> -> memref<1x64x128xf32>
          %32 = loom.semaphore_take %30 : memref<1x64x128xf32> -> memref<1x64x128xf32>
          %33 = loom.alloc [1, 64, 128] on @L1 : memref<1x64x128xf32>
          %34 = loom.semaphore_take %33 : memref<1x64x128xf32> -> memref<1x64x128xf32>
          %35 = loom.alloc [1, 64, 128] on @L1 : memref<1x64x128xf32>
          %36 = loom.semaphore_take %35 : memref<1x64x128xf32> -> memref<1x64x128xf32>
          %37 = arith.muli %arg6, %c524288 overflow<nsw> : index
          %38 = arith.muli %13, %c8192 : index
          %39 = arith.addi %37, %38 : index
          %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%39], sizes: [1, 64, 128], strides: [524288, 128, 1] : memref<32x4096x128xf32> to memref<1x64x128xf32, strided<[524288, 128, 1], offset: ?>>
          loom.copy %reinterpret_cast, %32 src_mem_space @DRAM dst_mem_space @L1, interconnect : [], broadcast : [1, 1] : memref<1x64x128xf32, strided<[524288, 128, 1], offset: ?>>, memref<1x64x128xf32>
          linalg.fill ins(%cst_1 : f32) outs(%34 : memref<1x64x128xf32>)
          linalg.fill ins(%cst_2 : f32) outs(%21 : memref<1x64xf32>)
          linalg.fill ins(%cst_3 : f32) outs(%23 : memref<1x64xf32>)
          scf.for %arg7 = %c0 to %c32 step %c1 {
            %40 = arith.muli %arg7, %c128 : index
            %41 = arith.addi %37, %40 : index
            %reinterpret_cast_5 = memref.reinterpret_cast %arg0 to offset: [%41], sizes: [1, 128, 128], strides: [524288, 4096, 1] : memref<32x128x4096xf32> to memref<1x128x128xf32, strided<[524288, 4096, 1], offset: ?>>
            loom.copy %reinterpret_cast_5, %19 src_mem_space @DRAM dst_mem_space @L1, interconnect : [@horizontal_links], broadcast : [1, 8] : memref<1x128x128xf32, strided<[524288, 4096, 1], offset: ?>>, memref<1x128x128xf32>
            linalg.fill ins(%cst_1 : f32) outs(%17 : memref<1x64x128xf32>)
            linalg.batch_matmul ins(%32, %19 : memref<1x64x128xf32>, memref<1x128x128xf32>) outs(%17 : memref<1x64x128xf32>)
            loom.semaphore_give %19 : memref<1x128x128xf32>
            linalg.fill ins(%cst_3 : f32) outs(%25 : memref<1x64xf32>)
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%17 : memref<1x64x128xf32>) outs(%25 : memref<1x64xf32>) {
            ^bb0(%in: f32, %out: f32):
              %44 = arith.maximumf %in, %out : f32
              linalg.yield %44 : f32
            }
            linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%23, %25 : memref<1x64xf32>, memref<1x64xf32>) outs(%25 : memref<1x64xf32>) {
            ^bb0(%in: f32, %in_7: f32, %out: f32):
              %44 = arith.truncf %cst_0 : f64 to f32
              %45 = arith.mulf %in_7, %44 : f32
              %46 = arith.cmpf ogt, %in, %45 : f32
              %47 = arith.select %46, %in, %45 : f32
              linalg.yield %47 : f32
            }
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%17, %25 : memref<1x64x128xf32>, memref<1x64xf32>) outs(%17 : memref<1x64x128xf32>) {
            ^bb0(%in: f32, %in_7: f32, %out: f32):
              %44 = arith.truncf %cst_0 : f64 to f32
              %45 = arith.mulf %in, %44 : f32
              %46 = arith.subf %45, %in_7 : f32
              %47 = math.powf %cst, %46 : f32
              linalg.yield %47 : f32
            }
            linalg.fill ins(%cst_1 : f32) outs(%27 : memref<1x64xf32>)
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%17 : memref<1x64x128xf32>) outs(%27 : memref<1x64xf32>) {
            ^bb0(%in: f32, %out: f32):
              %44 = arith.addf %in, %out : f32
              linalg.yield %44 : f32
            }
            linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%23, %25 : memref<1x64xf32>, memref<1x64xf32>) outs(%29 : memref<1x64xf32>) {
            ^bb0(%in: f32, %in_7: f32, %out: f32):
              %44 = arith.subf %in, %in_7 : f32
              %45 = math.powf %cst, %44 : f32
              linalg.yield %45 : f32
            }
            linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%21, %29, %27 : memref<1x64xf32>, memref<1x64xf32>, memref<1x64xf32>) outs(%21 : memref<1x64xf32>) {
            ^bb0(%in: f32, %in_7: f32, %in_8: f32, %out: f32):
              %44 = arith.mulf %in, %in_7 : f32
              %45 = arith.addf %44, %in_8 : f32
              linalg.yield %45 : f32
            }
            loom.semaphore_give %27 : memref<1x64xf32>
            %42 = arith.muli %arg7, %c16384 : index
            %43 = arith.addi %37, %42 : index
            %reinterpret_cast_6 = memref.reinterpret_cast %arg1 to offset: [%43], sizes: [1, 128, 128], strides: [524288, 128, 1] : memref<32x4096x128xf32> to memref<1x128x128xf32, strided<[524288, 128, 1], offset: ?>>
            loom.copy %reinterpret_cast_6, %15 src_mem_space @DRAM dst_mem_space @L1, interconnect : [@horizontal_links, @vertical_links], broadcast : [8, 8] : memref<1x128x128xf32, strided<[524288, 128, 1], offset: ?>>, memref<1x128x128xf32>
            linalg.fill ins(%cst_1 : f32) outs(%36 : memref<1x64x128xf32>)
            linalg.batch_matmul ins(%17, %15 : memref<1x64x128xf32>, memref<1x128x128xf32>) outs(%36 : memref<1x64x128xf32>)
            loom.semaphore_give %15 : memref<1x128x128xf32>
            loom.semaphore_give %17 : memref<1x64x128xf32>
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%36, %34, %29 : memref<1x64x128xf32>, memref<1x64x128xf32>, memref<1x64xf32>) outs(%34 : memref<1x64x128xf32>) {
            ^bb0(%in: f32, %in_7: f32, %in_8: f32, %out: f32):
              %44 = arith.mulf %in_7, %in_8 : f32
              %45 = arith.addf %in, %44 : f32
              linalg.yield %45 : f32
            }
            loom.semaphore_give %29 : memref<1x64xf32>
            loom.semaphore_give %36 : memref<1x64x128xf32>
            linalg.copy ins(%25 : memref<1x64xf32>) outs(%23 : memref<1x64xf32>)
            loom.semaphore_give %25 : memref<1x64xf32>
          }
          loom.semaphore_give %23 : memref<1x64xf32>
          loom.semaphore_give %32 : memref<1x64x128xf32>
          linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%34, %21 : memref<1x64x128xf32>, memref<1x64xf32>) outs(%31 : memref<1x64x128xf32>) {
          ^bb0(%in: f32, %in_5: f32, %out: f32):
            %40 = arith.divf %in, %in_5 : f32
            linalg.yield %40 : f32
          }
          loom.semaphore_give %21 : memref<1x64xf32>
          loom.semaphore_give %34 : memref<1x64x128xf32>
          %reinterpret_cast_4 = memref.reinterpret_cast %arg3 to offset: [%39], sizes: [1, 64, 128], strides: [524288, 128, 1] : memref<32x4096x128xf32> to memref<1x64x128xf32, strided<[524288, 128, 1], offset: ?>>
          loom.copy %31, %reinterpret_cast_4 src_mem_space @L1 dst_mem_space @DRAM, interconnect : [], broadcast : [1, 1] : memref<1x64x128xf32>, memref<1x64x128xf32, strided<[524288, 128, 1], offset: ?>>
          loom.semaphore_give %31 : memref<1x64x128xf32>
        }
        scf.reduce 
      } {loom.iter_types = [#loom.iter_type<spatial>, #loom.iter_type<spatial>], loom.mapped_to_dims = [@x, @y]}
      return
    }
  }
  module attributes {loom.pass_name = "Materialize"} {
    func.func @attention__d0i1_d1i1__f10__d_h_h__BB1__BM64__BN128(%arg0: memref<32x128x4096xf32>, %arg1: memref<32x4096x128xf32>, %arg2: memref<32x4096x128xf32>, %arg3: memref<32x4096x128xf32>) {
      %c16384 = arith.constant 16384 : index
      %c8192 = arith.constant 8192 : index
      %c524288 = arith.constant 524288 : index
      %c32 = arith.constant 32 : index
      %cst = arith.constant 2.000000e+00 : f32
      %cst_0 = arith.constant 0.12751743074602467 : f64
      %cst_1 = arith.constant 0.000000e+00 : f32
      %cst_2 = arith.constant 1.000000e+00 : f32
      %cst_3 = arith.constant 0xFF800000 : f32
      %c128 = arith.constant 128 : index
      %c0 = arith.constant 0 : index
      %c8 = arith.constant 8 : index
      %c1 = arith.constant 1 : index
      scf.parallel (%arg4, %arg5) = (%c0, %c0) to (%c8, %c8) step (%c1, %c1) {
        scf.for %arg6 = %c0 to %c32 step %c1 {
          %12 = arith.muli %arg4, %c8 overflow<nsw> : index
          %13 = arith.addi %12, %arg5 : index
          %14 = loom.alloc [1, 128, 128] on @L1 : memref<1x128x128xf32>
          %15 = loom.semaphore_take %14 : memref<1x128x128xf32> -> memref<1x128x128xf32>
          %16 = loom.alloc [1, 64, 128] on @L1 : memref<1x64x128xf32>
          %17 = loom.semaphore_take %16 : memref<1x64x128xf32> -> memref<1x64x128xf32>
          %18 = loom.alloc [1, 128, 128] on @L1 : memref<1x128x128xf32>
          %19 = loom.semaphore_take %18 : memref<1x128x128xf32> -> memref<1x128x128xf32>
          %20 = loom.alloc [1, 64] on @L1 : memref<1x64xf32>
          %21 = loom.semaphore_take %20 : memref<1x64xf32> -> memref<1x64xf32>
          %22 = loom.alloc [1, 64] on @L1 : memref<1x64xf32>
          %23 = loom.semaphore_take %22 : memref<1x64xf32> -> memref<1x64xf32>
          %24 = loom.alloc [1, 64] on @L1 : memref<1x64xf32>
          %25 = loom.semaphore_take %24 : memref<1x64xf32> -> memref<1x64xf32>
          %26 = loom.alloc [1, 64] on @L1 : memref<1x64xf32>
          %27 = loom.semaphore_take %26 : memref<1x64xf32> -> memref<1x64xf32>
          %28 = loom.alloc [1, 64] on @L1 : memref<1x64xf32>
          %29 = loom.semaphore_take %28 : memref<1x64xf32> -> memref<1x64xf32>
          %30 = loom.alloc [1, 64, 128] on @L1 : memref<1x64x128xf32>
          %31 = loom.semaphore_take %30 : memref<1x64x128xf32> -> memref<1x64x128xf32>
          %32 = loom.semaphore_take %30 : memref<1x64x128xf32> -> memref<1x64x128xf32>
          %33 = loom.alloc [1, 64, 128] on @L1 : memref<1x64x128xf32>
          %34 = loom.semaphore_take %33 : memref<1x64x128xf32> -> memref<1x64x128xf32>
          %35 = loom.alloc [1, 64, 128] on @L1 : memref<1x64x128xf32>
          %36 = loom.semaphore_take %35 : memref<1x64x128xf32> -> memref<1x64x128xf32>
          %37 = arith.muli %arg6, %c524288 overflow<nsw> : index
          %38 = arith.muli %13, %c8192 : index
          %39 = arith.addi %37, %38 : index
          %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%39], sizes: [1, 64, 128], strides: [524288, 128, 1] : memref<32x4096x128xf32> to memref<1x64x128xf32, strided<[524288, 128, 1], offset: ?>>
          loom.copy %reinterpret_cast, %32 src_mem_space @DRAM dst_mem_space @L1, interconnect : [], broadcast : [1, 1] : memref<1x64x128xf32, strided<[524288, 128, 1], offset: ?>>, memref<1x64x128xf32>
          linalg.fill ins(%cst_1 : f32) outs(%34 : memref<1x64x128xf32>)
          linalg.fill ins(%cst_2 : f32) outs(%21 : memref<1x64xf32>)
          linalg.fill ins(%cst_3 : f32) outs(%23 : memref<1x64xf32>)
          scf.for %arg7 = %c0 to %c32 step %c1 {
            %40 = arith.muli %arg7, %c128 : index
            %41 = arith.addi %37, %40 : index
            %reinterpret_cast_5 = memref.reinterpret_cast %arg0 to offset: [%41], sizes: [1, 128, 128], strides: [524288, 4096, 1] : memref<32x128x4096xf32> to memref<1x128x128xf32, strided<[524288, 4096, 1], offset: ?>>
            loom.copy %reinterpret_cast_5, %19 src_mem_space @DRAM dst_mem_space @L1, interconnect : [@horizontal_links], broadcast : [1, 8] : memref<1x128x128xf32, strided<[524288, 4096, 1], offset: ?>>, memref<1x128x128xf32>
            linalg.fill ins(%cst_1 : f32) outs(%17 : memref<1x64x128xf32>)
            linalg.batch_matmul ins(%32, %19 : memref<1x64x128xf32>, memref<1x128x128xf32>) outs(%17 : memref<1x64x128xf32>)
            loom.semaphore_give %19 : memref<1x128x128xf32>
            linalg.fill ins(%cst_3 : f32) outs(%25 : memref<1x64xf32>)
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%17 : memref<1x64x128xf32>) outs(%25 : memref<1x64xf32>) {
            ^bb0(%in: f32, %out: f32):
              %44 = arith.maximumf %in, %out : f32
              linalg.yield %44 : f32
            }
            linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%23, %25 : memref<1x64xf32>, memref<1x64xf32>) outs(%25 : memref<1x64xf32>) {
            ^bb0(%in: f32, %in_7: f32, %out: f32):
              %44 = arith.truncf %cst_0 : f64 to f32
              %45 = arith.mulf %in_7, %44 : f32
              %46 = arith.cmpf ogt, %in, %45 : f32
              %47 = arith.select %46, %in, %45 : f32
              linalg.yield %47 : f32
            }
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%17, %25 : memref<1x64x128xf32>, memref<1x64xf32>) outs(%17 : memref<1x64x128xf32>) {
            ^bb0(%in: f32, %in_7: f32, %out: f32):
              %44 = arith.truncf %cst_0 : f64 to f32
              %45 = arith.mulf %in, %44 : f32
              %46 = arith.subf %45, %in_7 : f32
              %47 = math.powf %cst, %46 : f32
              linalg.yield %47 : f32
            }
            linalg.fill ins(%cst_1 : f32) outs(%27 : memref<1x64xf32>)
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%17 : memref<1x64x128xf32>) outs(%27 : memref<1x64xf32>) {
            ^bb0(%in: f32, %out: f32):
              %44 = arith.addf %in, %out : f32
              linalg.yield %44 : f32
            }
            linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%23, %25 : memref<1x64xf32>, memref<1x64xf32>) outs(%29 : memref<1x64xf32>) {
            ^bb0(%in: f32, %in_7: f32, %out: f32):
              %44 = arith.subf %in, %in_7 : f32
              %45 = math.powf %cst, %44 : f32
              linalg.yield %45 : f32
            }
            linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%21, %29, %27 : memref<1x64xf32>, memref<1x64xf32>, memref<1x64xf32>) outs(%21 : memref<1x64xf32>) {
            ^bb0(%in: f32, %in_7: f32, %in_8: f32, %out: f32):
              %44 = arith.mulf %in, %in_7 : f32
              %45 = arith.addf %44, %in_8 : f32
              linalg.yield %45 : f32
            }
            loom.semaphore_give %27 : memref<1x64xf32>
            %42 = arith.muli %arg7, %c16384 : index
            %43 = arith.addi %37, %42 : index
            %reinterpret_cast_6 = memref.reinterpret_cast %arg1 to offset: [%43], sizes: [1, 128, 128], strides: [524288, 128, 1] : memref<32x4096x128xf32> to memref<1x128x128xf32, strided<[524288, 128, 1], offset: ?>>
            loom.copy %reinterpret_cast_6, %15 src_mem_space @DRAM dst_mem_space @L1, interconnect : [@horizontal_links], broadcast : [1, 8] : memref<1x128x128xf32, strided<[524288, 128, 1], offset: ?>>, memref<1x128x128xf32>
            linalg.fill ins(%cst_1 : f32) outs(%36 : memref<1x64x128xf32>)
            linalg.batch_matmul ins(%17, %15 : memref<1x64x128xf32>, memref<1x128x128xf32>) outs(%36 : memref<1x64x128xf32>)
            loom.semaphore_give %15 : memref<1x128x128xf32>
            loom.semaphore_give %17 : memref<1x64x128xf32>
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%36, %34, %29 : memref<1x64x128xf32>, memref<1x64x128xf32>, memref<1x64xf32>) outs(%34 : memref<1x64x128xf32>) {
            ^bb0(%in: f32, %in_7: f32, %in_8: f32, %out: f32):
              %44 = arith.mulf %in_7, %in_8 : f32
              %45 = arith.addf %in, %44 : f32
              linalg.yield %45 : f32
            }
            loom.semaphore_give %29 : memref<1x64xf32>
            loom.semaphore_give %36 : memref<1x64x128xf32>
            linalg.copy ins(%25 : memref<1x64xf32>) outs(%23 : memref<1x64xf32>)
            loom.semaphore_give %25 : memref<1x64xf32>
          }
          loom.semaphore_give %23 : memref<1x64xf32>
          loom.semaphore_give %32 : memref<1x64x128xf32>
          linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%34, %21 : memref<1x64x128xf32>, memref<1x64xf32>) outs(%31 : memref<1x64x128xf32>) {
          ^bb0(%in: f32, %in_5: f32, %out: f32):
            %40 = arith.divf %in, %in_5 : f32
            linalg.yield %40 : f32
          }
          loom.semaphore_give %21 : memref<1x64xf32>
          loom.semaphore_give %34 : memref<1x64x128xf32>
          %reinterpret_cast_4 = memref.reinterpret_cast %arg3 to offset: [%39], sizes: [1, 64, 128], strides: [524288, 128, 1] : memref<32x4096x128xf32> to memref<1x64x128xf32, strided<[524288, 128, 1], offset: ?>>
          loom.copy %31, %reinterpret_cast_4 src_mem_space @L1 dst_mem_space @DRAM, interconnect : [], broadcast : [1, 1] : memref<1x64x128xf32>, memref<1x64x128xf32, strided<[524288, 128, 1], offset: ?>>
          loom.semaphore_give %31 : memref<1x64x128xf32>
        }
        scf.reduce 
      } {loom.iter_types = [#loom.iter_type<spatial>, #loom.iter_type<spatial>], loom.mapped_to_dims = [@x, @y]}
      return
    }
  }
  module attributes {loom.pass_name = "Materialize"} {
    func.func @attention__d0i1_d1i1__f10__d_h_v__BB1__BM64__BN128(%arg0: memref<32x128x4096xf32>, %arg1: memref<32x4096x128xf32>, %arg2: memref<32x4096x128xf32>, %arg3: memref<32x4096x128xf32>) {
      %c16384 = arith.constant 16384 : index
      %c8192 = arith.constant 8192 : index
      %c524288 = arith.constant 524288 : index
      %c32 = arith.constant 32 : index
      %cst = arith.constant 2.000000e+00 : f32
      %cst_0 = arith.constant 0.12751743074602467 : f64
      %cst_1 = arith.constant 0.000000e+00 : f32
      %cst_2 = arith.constant 1.000000e+00 : f32
      %cst_3 = arith.constant 0xFF800000 : f32
      %c128 = arith.constant 128 : index
      %c0 = arith.constant 0 : index
      %c8 = arith.constant 8 : index
      %c1 = arith.constant 1 : index
      scf.parallel (%arg4, %arg5) = (%c0, %c0) to (%c8, %c8) step (%c1, %c1) {
        scf.for %arg6 = %c0 to %c32 step %c1 {
          %12 = arith.muli %arg4, %c8 overflow<nsw> : index
          %13 = arith.addi %12, %arg5 : index
          %14 = loom.alloc [1, 128, 128] on @L1 : memref<1x128x128xf32>
          %15 = loom.semaphore_take %14 : memref<1x128x128xf32> -> memref<1x128x128xf32>
          %16 = loom.alloc [1, 64, 128] on @L1 : memref<1x64x128xf32>
          %17 = loom.semaphore_take %16 : memref<1x64x128xf32> -> memref<1x64x128xf32>
          %18 = loom.alloc [1, 128, 128] on @L1 : memref<1x128x128xf32>
          %19 = loom.semaphore_take %18 : memref<1x128x128xf32> -> memref<1x128x128xf32>
          %20 = loom.alloc [1, 64] on @L1 : memref<1x64xf32>
          %21 = loom.semaphore_take %20 : memref<1x64xf32> -> memref<1x64xf32>
          %22 = loom.alloc [1, 64] on @L1 : memref<1x64xf32>
          %23 = loom.semaphore_take %22 : memref<1x64xf32> -> memref<1x64xf32>
          %24 = loom.alloc [1, 64] on @L1 : memref<1x64xf32>
          %25 = loom.semaphore_take %24 : memref<1x64xf32> -> memref<1x64xf32>
          %26 = loom.alloc [1, 64] on @L1 : memref<1x64xf32>
          %27 = loom.semaphore_take %26 : memref<1x64xf32> -> memref<1x64xf32>
          %28 = loom.alloc [1, 64] on @L1 : memref<1x64xf32>
          %29 = loom.semaphore_take %28 : memref<1x64xf32> -> memref<1x64xf32>
          %30 = loom.alloc [1, 64, 128] on @L1 : memref<1x64x128xf32>
          %31 = loom.semaphore_take %30 : memref<1x64x128xf32> -> memref<1x64x128xf32>
          %32 = loom.semaphore_take %30 : memref<1x64x128xf32> -> memref<1x64x128xf32>
          %33 = loom.alloc [1, 64, 128] on @L1 : memref<1x64x128xf32>
          %34 = loom.semaphore_take %33 : memref<1x64x128xf32> -> memref<1x64x128xf32>
          %35 = loom.alloc [1, 64, 128] on @L1 : memref<1x64x128xf32>
          %36 = loom.semaphore_take %35 : memref<1x64x128xf32> -> memref<1x64x128xf32>
          %37 = arith.muli %arg6, %c524288 overflow<nsw> : index
          %38 = arith.muli %13, %c8192 : index
          %39 = arith.addi %37, %38 : index
          %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%39], sizes: [1, 64, 128], strides: [524288, 128, 1] : memref<32x4096x128xf32> to memref<1x64x128xf32, strided<[524288, 128, 1], offset: ?>>
          loom.copy %reinterpret_cast, %32 src_mem_space @DRAM dst_mem_space @L1, interconnect : [], broadcast : [1, 1] : memref<1x64x128xf32, strided<[524288, 128, 1], offset: ?>>, memref<1x64x128xf32>
          linalg.fill ins(%cst_1 : f32) outs(%34 : memref<1x64x128xf32>)
          linalg.fill ins(%cst_2 : f32) outs(%21 : memref<1x64xf32>)
          linalg.fill ins(%cst_3 : f32) outs(%23 : memref<1x64xf32>)
          scf.for %arg7 = %c0 to %c32 step %c1 {
            %40 = arith.muli %arg7, %c128 : index
            %41 = arith.addi %37, %40 : index
            %reinterpret_cast_5 = memref.reinterpret_cast %arg0 to offset: [%41], sizes: [1, 128, 128], strides: [524288, 4096, 1] : memref<32x128x4096xf32> to memref<1x128x128xf32, strided<[524288, 4096, 1], offset: ?>>
            loom.copy %reinterpret_cast_5, %19 src_mem_space @DRAM dst_mem_space @L1, interconnect : [@horizontal_links], broadcast : [1, 8] : memref<1x128x128xf32, strided<[524288, 4096, 1], offset: ?>>, memref<1x128x128xf32>
            linalg.fill ins(%cst_1 : f32) outs(%17 : memref<1x64x128xf32>)
            linalg.batch_matmul ins(%32, %19 : memref<1x64x128xf32>, memref<1x128x128xf32>) outs(%17 : memref<1x64x128xf32>)
            loom.semaphore_give %19 : memref<1x128x128xf32>
            linalg.fill ins(%cst_3 : f32) outs(%25 : memref<1x64xf32>)
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%17 : memref<1x64x128xf32>) outs(%25 : memref<1x64xf32>) {
            ^bb0(%in: f32, %out: f32):
              %44 = arith.maximumf %in, %out : f32
              linalg.yield %44 : f32
            }
            linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%23, %25 : memref<1x64xf32>, memref<1x64xf32>) outs(%25 : memref<1x64xf32>) {
            ^bb0(%in: f32, %in_7: f32, %out: f32):
              %44 = arith.truncf %cst_0 : f64 to f32
              %45 = arith.mulf %in_7, %44 : f32
              %46 = arith.cmpf ogt, %in, %45 : f32
              %47 = arith.select %46, %in, %45 : f32
              linalg.yield %47 : f32
            }
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%17, %25 : memref<1x64x128xf32>, memref<1x64xf32>) outs(%17 : memref<1x64x128xf32>) {
            ^bb0(%in: f32, %in_7: f32, %out: f32):
              %44 = arith.truncf %cst_0 : f64 to f32
              %45 = arith.mulf %in, %44 : f32
              %46 = arith.subf %45, %in_7 : f32
              %47 = math.powf %cst, %46 : f32
              linalg.yield %47 : f32
            }
            linalg.fill ins(%cst_1 : f32) outs(%27 : memref<1x64xf32>)
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%17 : memref<1x64x128xf32>) outs(%27 : memref<1x64xf32>) {
            ^bb0(%in: f32, %out: f32):
              %44 = arith.addf %in, %out : f32
              linalg.yield %44 : f32
            }
            linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%23, %25 : memref<1x64xf32>, memref<1x64xf32>) outs(%29 : memref<1x64xf32>) {
            ^bb0(%in: f32, %in_7: f32, %out: f32):
              %44 = arith.subf %in, %in_7 : f32
              %45 = math.powf %cst, %44 : f32
              linalg.yield %45 : f32
            }
            linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%21, %29, %27 : memref<1x64xf32>, memref<1x64xf32>, memref<1x64xf32>) outs(%21 : memref<1x64xf32>) {
            ^bb0(%in: f32, %in_7: f32, %in_8: f32, %out: f32):
              %44 = arith.mulf %in, %in_7 : f32
              %45 = arith.addf %44, %in_8 : f32
              linalg.yield %45 : f32
            }
            loom.semaphore_give %27 : memref<1x64xf32>
            %42 = arith.muli %arg7, %c16384 : index
            %43 = arith.addi %37, %42 : index
            %reinterpret_cast_6 = memref.reinterpret_cast %arg1 to offset: [%43], sizes: [1, 128, 128], strides: [524288, 128, 1] : memref<32x4096x128xf32> to memref<1x128x128xf32, strided<[524288, 128, 1], offset: ?>>
            loom.copy %reinterpret_cast_6, %15 src_mem_space @DRAM dst_mem_space @L1, interconnect : [@vertical_links], broadcast : [8, 1] : memref<1x128x128xf32, strided<[524288, 128, 1], offset: ?>>, memref<1x128x128xf32>
            linalg.fill ins(%cst_1 : f32) outs(%36 : memref<1x64x128xf32>)
            linalg.batch_matmul ins(%17, %15 : memref<1x64x128xf32>, memref<1x128x128xf32>) outs(%36 : memref<1x64x128xf32>)
            loom.semaphore_give %15 : memref<1x128x128xf32>
            loom.semaphore_give %17 : memref<1x64x128xf32>
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%36, %34, %29 : memref<1x64x128xf32>, memref<1x64x128xf32>, memref<1x64xf32>) outs(%34 : memref<1x64x128xf32>) {
            ^bb0(%in: f32, %in_7: f32, %in_8: f32, %out: f32):
              %44 = arith.mulf %in_7, %in_8 : f32
              %45 = arith.addf %in, %44 : f32
              linalg.yield %45 : f32
            }
            loom.semaphore_give %29 : memref<1x64xf32>
            loom.semaphore_give %36 : memref<1x64x128xf32>
            linalg.copy ins(%25 : memref<1x64xf32>) outs(%23 : memref<1x64xf32>)
            loom.semaphore_give %25 : memref<1x64xf32>
          }
          loom.semaphore_give %23 : memref<1x64xf32>
          loom.semaphore_give %32 : memref<1x64x128xf32>
          linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%34, %21 : memref<1x64x128xf32>, memref<1x64xf32>) outs(%31 : memref<1x64x128xf32>) {
          ^bb0(%in: f32, %in_5: f32, %out: f32):
            %40 = arith.divf %in, %in_5 : f32
            linalg.yield %40 : f32
          }
          loom.semaphore_give %21 : memref<1x64xf32>
          loom.semaphore_give %34 : memref<1x64x128xf32>
          %reinterpret_cast_4 = memref.reinterpret_cast %arg3 to offset: [%39], sizes: [1, 64, 128], strides: [524288, 128, 1] : memref<32x4096x128xf32> to memref<1x64x128xf32, strided<[524288, 128, 1], offset: ?>>
          loom.copy %31, %reinterpret_cast_4 src_mem_space @L1 dst_mem_space @DRAM, interconnect : [], broadcast : [1, 1] : memref<1x64x128xf32>, memref<1x64x128xf32, strided<[524288, 128, 1], offset: ?>>
          loom.semaphore_give %31 : memref<1x64x128xf32>
        }
        scf.reduce 
      } {loom.iter_types = [#loom.iter_type<spatial>, #loom.iter_type<spatial>], loom.mapped_to_dims = [@x, @y]}
      return
    }
  }
  module attributes {loom.pass_name = "Materialize"} {
    func.func @attention__d0i1_d1i1__f10__d_v_d__BB1__BM64__BN128(%arg0: memref<32x128x4096xf32>, %arg1: memref<32x4096x128xf32>, %arg2: memref<32x4096x128xf32>, %arg3: memref<32x4096x128xf32>) {
      %c16384 = arith.constant 16384 : index
      %c8192 = arith.constant 8192 : index
      %c524288 = arith.constant 524288 : index
      %c32 = arith.constant 32 : index
      %cst = arith.constant 2.000000e+00 : f32
      %cst_0 = arith.constant 0.12751743074602467 : f64
      %cst_1 = arith.constant 0.000000e+00 : f32
      %cst_2 = arith.constant 1.000000e+00 : f32
      %cst_3 = arith.constant 0xFF800000 : f32
      %c128 = arith.constant 128 : index
      %c0 = arith.constant 0 : index
      %c8 = arith.constant 8 : index
      %c1 = arith.constant 1 : index
      scf.parallel (%arg4, %arg5) = (%c0, %c0) to (%c8, %c8) step (%c1, %c1) {
        scf.for %arg6 = %c0 to %c32 step %c1 {
          %12 = arith.muli %arg4, %c8 overflow<nsw> : index
          %13 = arith.addi %12, %arg5 : index
          %14 = loom.alloc [1, 128, 128] on @L1 : memref<1x128x128xf32>
          %15 = loom.semaphore_take %14 : memref<1x128x128xf32> -> memref<1x128x128xf32>
          %16 = loom.alloc [1, 64, 128] on @L1 : memref<1x64x128xf32>
          %17 = loom.semaphore_take %16 : memref<1x64x128xf32> -> memref<1x64x128xf32>
          %18 = loom.alloc [1, 128, 128] on @L1 : memref<1x128x128xf32>
          %19 = loom.semaphore_take %18 : memref<1x128x128xf32> -> memref<1x128x128xf32>
          %20 = loom.alloc [1, 64] on @L1 : memref<1x64xf32>
          %21 = loom.semaphore_take %20 : memref<1x64xf32> -> memref<1x64xf32>
          %22 = loom.alloc [1, 64] on @L1 : memref<1x64xf32>
          %23 = loom.semaphore_take %22 : memref<1x64xf32> -> memref<1x64xf32>
          %24 = loom.alloc [1, 64] on @L1 : memref<1x64xf32>
          %25 = loom.semaphore_take %24 : memref<1x64xf32> -> memref<1x64xf32>
          %26 = loom.alloc [1, 64] on @L1 : memref<1x64xf32>
          %27 = loom.semaphore_take %26 : memref<1x64xf32> -> memref<1x64xf32>
          %28 = loom.alloc [1, 64] on @L1 : memref<1x64xf32>
          %29 = loom.semaphore_take %28 : memref<1x64xf32> -> memref<1x64xf32>
          %30 = loom.alloc [1, 64, 128] on @L1 : memref<1x64x128xf32>
          %31 = loom.semaphore_take %30 : memref<1x64x128xf32> -> memref<1x64x128xf32>
          %32 = loom.semaphore_take %30 : memref<1x64x128xf32> -> memref<1x64x128xf32>
          %33 = loom.alloc [1, 64, 128] on @L1 : memref<1x64x128xf32>
          %34 = loom.semaphore_take %33 : memref<1x64x128xf32> -> memref<1x64x128xf32>
          %35 = loom.alloc [1, 64, 128] on @L1 : memref<1x64x128xf32>
          %36 = loom.semaphore_take %35 : memref<1x64x128xf32> -> memref<1x64x128xf32>
          %37 = arith.muli %arg6, %c524288 overflow<nsw> : index
          %38 = arith.muli %13, %c8192 : index
          %39 = arith.addi %37, %38 : index
          %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%39], sizes: [1, 64, 128], strides: [524288, 128, 1] : memref<32x4096x128xf32> to memref<1x64x128xf32, strided<[524288, 128, 1], offset: ?>>
          loom.copy %reinterpret_cast, %32 src_mem_space @DRAM dst_mem_space @L1, interconnect : [], broadcast : [1, 1] : memref<1x64x128xf32, strided<[524288, 128, 1], offset: ?>>, memref<1x64x128xf32>
          linalg.fill ins(%cst_1 : f32) outs(%34 : memref<1x64x128xf32>)
          linalg.fill ins(%cst_2 : f32) outs(%21 : memref<1x64xf32>)
          linalg.fill ins(%cst_3 : f32) outs(%23 : memref<1x64xf32>)
          scf.for %arg7 = %c0 to %c32 step %c1 {
            %40 = arith.muli %arg7, %c128 : index
            %41 = arith.addi %37, %40 : index
            %reinterpret_cast_5 = memref.reinterpret_cast %arg0 to offset: [%41], sizes: [1, 128, 128], strides: [524288, 4096, 1] : memref<32x128x4096xf32> to memref<1x128x128xf32, strided<[524288, 4096, 1], offset: ?>>
            loom.copy %reinterpret_cast_5, %19 src_mem_space @DRAM dst_mem_space @L1, interconnect : [@vertical_links], broadcast : [8, 1] : memref<1x128x128xf32, strided<[524288, 4096, 1], offset: ?>>, memref<1x128x128xf32>
            linalg.fill ins(%cst_1 : f32) outs(%17 : memref<1x64x128xf32>)
            linalg.batch_matmul ins(%32, %19 : memref<1x64x128xf32>, memref<1x128x128xf32>) outs(%17 : memref<1x64x128xf32>)
            loom.semaphore_give %19 : memref<1x128x128xf32>
            linalg.fill ins(%cst_3 : f32) outs(%25 : memref<1x64xf32>)
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%17 : memref<1x64x128xf32>) outs(%25 : memref<1x64xf32>) {
            ^bb0(%in: f32, %out: f32):
              %44 = arith.maximumf %in, %out : f32
              linalg.yield %44 : f32
            }
            linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%23, %25 : memref<1x64xf32>, memref<1x64xf32>) outs(%25 : memref<1x64xf32>) {
            ^bb0(%in: f32, %in_7: f32, %out: f32):
              %44 = arith.truncf %cst_0 : f64 to f32
              %45 = arith.mulf %in_7, %44 : f32
              %46 = arith.cmpf ogt, %in, %45 : f32
              %47 = arith.select %46, %in, %45 : f32
              linalg.yield %47 : f32
            }
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%17, %25 : memref<1x64x128xf32>, memref<1x64xf32>) outs(%17 : memref<1x64x128xf32>) {
            ^bb0(%in: f32, %in_7: f32, %out: f32):
              %44 = arith.truncf %cst_0 : f64 to f32
              %45 = arith.mulf %in, %44 : f32
              %46 = arith.subf %45, %in_7 : f32
              %47 = math.powf %cst, %46 : f32
              linalg.yield %47 : f32
            }
            linalg.fill ins(%cst_1 : f32) outs(%27 : memref<1x64xf32>)
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%17 : memref<1x64x128xf32>) outs(%27 : memref<1x64xf32>) {
            ^bb0(%in: f32, %out: f32):
              %44 = arith.addf %in, %out : f32
              linalg.yield %44 : f32
            }
            linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%23, %25 : memref<1x64xf32>, memref<1x64xf32>) outs(%29 : memref<1x64xf32>) {
            ^bb0(%in: f32, %in_7: f32, %out: f32):
              %44 = arith.subf %in, %in_7 : f32
              %45 = math.powf %cst, %44 : f32
              linalg.yield %45 : f32
            }
            linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%21, %29, %27 : memref<1x64xf32>, memref<1x64xf32>, memref<1x64xf32>) outs(%21 : memref<1x64xf32>) {
            ^bb0(%in: f32, %in_7: f32, %in_8: f32, %out: f32):
              %44 = arith.mulf %in, %in_7 : f32
              %45 = arith.addf %44, %in_8 : f32
              linalg.yield %45 : f32
            }
            loom.semaphore_give %27 : memref<1x64xf32>
            %42 = arith.muli %arg7, %c16384 : index
            %43 = arith.addi %37, %42 : index
            %reinterpret_cast_6 = memref.reinterpret_cast %arg1 to offset: [%43], sizes: [1, 128, 128], strides: [524288, 128, 1] : memref<32x4096x128xf32> to memref<1x128x128xf32, strided<[524288, 128, 1], offset: ?>>
            loom.copy %reinterpret_cast_6, %15 src_mem_space @DRAM dst_mem_space @L1, interconnect : [], broadcast : [1, 1] : memref<1x128x128xf32, strided<[524288, 128, 1], offset: ?>>, memref<1x128x128xf32>
            linalg.fill ins(%cst_1 : f32) outs(%36 : memref<1x64x128xf32>)
            linalg.batch_matmul ins(%17, %15 : memref<1x64x128xf32>, memref<1x128x128xf32>) outs(%36 : memref<1x64x128xf32>)
            loom.semaphore_give %15 : memref<1x128x128xf32>
            loom.semaphore_give %17 : memref<1x64x128xf32>
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%36, %34, %29 : memref<1x64x128xf32>, memref<1x64x128xf32>, memref<1x64xf32>) outs(%34 : memref<1x64x128xf32>) {
            ^bb0(%in: f32, %in_7: f32, %in_8: f32, %out: f32):
              %44 = arith.mulf %in_7, %in_8 : f32
              %45 = arith.addf %in, %44 : f32
              linalg.yield %45 : f32
            }
            loom.semaphore_give %29 : memref<1x64xf32>
            loom.semaphore_give %36 : memref<1x64x128xf32>
            linalg.copy ins(%25 : memref<1x64xf32>) outs(%23 : memref<1x64xf32>)
            loom.semaphore_give %25 : memref<1x64xf32>
          }
          loom.semaphore_give %23 : memref<1x64xf32>
          loom.semaphore_give %32 : memref<1x64x128xf32>
          linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%34, %21 : memref<1x64x128xf32>, memref<1x64xf32>) outs(%31 : memref<1x64x128xf32>) {
          ^bb0(%in: f32, %in_5: f32, %out: f32):
            %40 = arith.divf %in, %in_5 : f32
            linalg.yield %40 : f32
          }
          loom.semaphore_give %21 : memref<1x64xf32>
          loom.semaphore_give %34 : memref<1x64x128xf32>
          %reinterpret_cast_4 = memref.reinterpret_cast %arg3 to offset: [%39], sizes: [1, 64, 128], strides: [524288, 128, 1] : memref<32x4096x128xf32> to memref<1x64x128xf32, strided<[524288, 128, 1], offset: ?>>
          loom.copy %31, %reinterpret_cast_4 src_mem_space @L1 dst_mem_space @DRAM, interconnect : [], broadcast : [1, 1] : memref<1x64x128xf32>, memref<1x64x128xf32, strided<[524288, 128, 1], offset: ?>>
          loom.semaphore_give %31 : memref<1x64x128xf32>
        }
        scf.reduce 
      } {loom.iter_types = [#loom.iter_type<spatial>, #loom.iter_type<spatial>], loom.mapped_to_dims = [@x, @y]}
      return
    }
  }
  module attributes {loom.pass_name = "Materialize"} {
    func.func @attention__d0i1_d1i1__f10__d_v_a__BB1__BM64__BN128(%arg0: memref<32x128x4096xf32>, %arg1: memref<32x4096x128xf32>, %arg2: memref<32x4096x128xf32>, %arg3: memref<32x4096x128xf32>) {
      %c16384 = arith.constant 16384 : index
      %c8192 = arith.constant 8192 : index
      %c524288 = arith.constant 524288 : index
      %c32 = arith.constant 32 : index
      %cst = arith.constant 2.000000e+00 : f32
      %cst_0 = arith.constant 0.12751743074602467 : f64
      %cst_1 = arith.constant 0.000000e+00 : f32
      %cst_2 = arith.constant 1.000000e+00 : f32
      %cst_3 = arith.constant 0xFF800000 : f32
      %c128 = arith.constant 128 : index
      %c0 = arith.constant 0 : index
      %c8 = arith.constant 8 : index
      %c1 = arith.constant 1 : index
      scf.parallel (%arg4, %arg5) = (%c0, %c0) to (%c8, %c8) step (%c1, %c1) {
        scf.for %arg6 = %c0 to %c32 step %c1 {
          %12 = arith.muli %arg4, %c8 overflow<nsw> : index
          %13 = arith.addi %12, %arg5 : index
          %14 = loom.alloc [1, 128, 128] on @L1 : memref<1x128x128xf32>
          %15 = loom.semaphore_take %14 : memref<1x128x128xf32> -> memref<1x128x128xf32>
          %16 = loom.alloc [1, 64, 128] on @L1 : memref<1x64x128xf32>
          %17 = loom.semaphore_take %16 : memref<1x64x128xf32> -> memref<1x64x128xf32>
          %18 = loom.alloc [1, 128, 128] on @L1 : memref<1x128x128xf32>
          %19 = loom.semaphore_take %18 : memref<1x128x128xf32> -> memref<1x128x128xf32>
          %20 = loom.alloc [1, 64] on @L1 : memref<1x64xf32>
          %21 = loom.semaphore_take %20 : memref<1x64xf32> -> memref<1x64xf32>
          %22 = loom.alloc [1, 64] on @L1 : memref<1x64xf32>
          %23 = loom.semaphore_take %22 : memref<1x64xf32> -> memref<1x64xf32>
          %24 = loom.alloc [1, 64] on @L1 : memref<1x64xf32>
          %25 = loom.semaphore_take %24 : memref<1x64xf32> -> memref<1x64xf32>
          %26 = loom.alloc [1, 64] on @L1 : memref<1x64xf32>
          %27 = loom.semaphore_take %26 : memref<1x64xf32> -> memref<1x64xf32>
          %28 = loom.alloc [1, 64] on @L1 : memref<1x64xf32>
          %29 = loom.semaphore_take %28 : memref<1x64xf32> -> memref<1x64xf32>
          %30 = loom.alloc [1, 64, 128] on @L1 : memref<1x64x128xf32>
          %31 = loom.semaphore_take %30 : memref<1x64x128xf32> -> memref<1x64x128xf32>
          %32 = loom.semaphore_take %30 : memref<1x64x128xf32> -> memref<1x64x128xf32>
          %33 = loom.alloc [1, 64, 128] on @L1 : memref<1x64x128xf32>
          %34 = loom.semaphore_take %33 : memref<1x64x128xf32> -> memref<1x64x128xf32>
          %35 = loom.alloc [1, 64, 128] on @L1 : memref<1x64x128xf32>
          %36 = loom.semaphore_take %35 : memref<1x64x128xf32> -> memref<1x64x128xf32>
          %37 = arith.muli %arg6, %c524288 overflow<nsw> : index
          %38 = arith.muli %13, %c8192 : index
          %39 = arith.addi %37, %38 : index
          %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%39], sizes: [1, 64, 128], strides: [524288, 128, 1] : memref<32x4096x128xf32> to memref<1x64x128xf32, strided<[524288, 128, 1], offset: ?>>
          loom.copy %reinterpret_cast, %32 src_mem_space @DRAM dst_mem_space @L1, interconnect : [], broadcast : [1, 1] : memref<1x64x128xf32, strided<[524288, 128, 1], offset: ?>>, memref<1x64x128xf32>
          linalg.fill ins(%cst_1 : f32) outs(%34 : memref<1x64x128xf32>)
          linalg.fill ins(%cst_2 : f32) outs(%21 : memref<1x64xf32>)
          linalg.fill ins(%cst_3 : f32) outs(%23 : memref<1x64xf32>)
          scf.for %arg7 = %c0 to %c32 step %c1 {
            %40 = arith.muli %arg7, %c128 : index
            %41 = arith.addi %37, %40 : index
            %reinterpret_cast_5 = memref.reinterpret_cast %arg0 to offset: [%41], sizes: [1, 128, 128], strides: [524288, 4096, 1] : memref<32x128x4096xf32> to memref<1x128x128xf32, strided<[524288, 4096, 1], offset: ?>>
            loom.copy %reinterpret_cast_5, %19 src_mem_space @DRAM dst_mem_space @L1, interconnect : [@vertical_links], broadcast : [8, 1] : memref<1x128x128xf32, strided<[524288, 4096, 1], offset: ?>>, memref<1x128x128xf32>
            linalg.fill ins(%cst_1 : f32) outs(%17 : memref<1x64x128xf32>)
            linalg.batch_matmul ins(%32, %19 : memref<1x64x128xf32>, memref<1x128x128xf32>) outs(%17 : memref<1x64x128xf32>)
            loom.semaphore_give %19 : memref<1x128x128xf32>
            linalg.fill ins(%cst_3 : f32) outs(%25 : memref<1x64xf32>)
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%17 : memref<1x64x128xf32>) outs(%25 : memref<1x64xf32>) {
            ^bb0(%in: f32, %out: f32):
              %44 = arith.maximumf %in, %out : f32
              linalg.yield %44 : f32
            }
            linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%23, %25 : memref<1x64xf32>, memref<1x64xf32>) outs(%25 : memref<1x64xf32>) {
            ^bb0(%in: f32, %in_7: f32, %out: f32):
              %44 = arith.truncf %cst_0 : f64 to f32
              %45 = arith.mulf %in_7, %44 : f32
              %46 = arith.cmpf ogt, %in, %45 : f32
              %47 = arith.select %46, %in, %45 : f32
              linalg.yield %47 : f32
            }
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%17, %25 : memref<1x64x128xf32>, memref<1x64xf32>) outs(%17 : memref<1x64x128xf32>) {
            ^bb0(%in: f32, %in_7: f32, %out: f32):
              %44 = arith.truncf %cst_0 : f64 to f32
              %45 = arith.mulf %in, %44 : f32
              %46 = arith.subf %45, %in_7 : f32
              %47 = math.powf %cst, %46 : f32
              linalg.yield %47 : f32
            }
            linalg.fill ins(%cst_1 : f32) outs(%27 : memref<1x64xf32>)
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%17 : memref<1x64x128xf32>) outs(%27 : memref<1x64xf32>) {
            ^bb0(%in: f32, %out: f32):
              %44 = arith.addf %in, %out : f32
              linalg.yield %44 : f32
            }
            linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%23, %25 : memref<1x64xf32>, memref<1x64xf32>) outs(%29 : memref<1x64xf32>) {
            ^bb0(%in: f32, %in_7: f32, %out: f32):
              %44 = arith.subf %in, %in_7 : f32
              %45 = math.powf %cst, %44 : f32
              linalg.yield %45 : f32
            }
            linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%21, %29, %27 : memref<1x64xf32>, memref<1x64xf32>, memref<1x64xf32>) outs(%21 : memref<1x64xf32>) {
            ^bb0(%in: f32, %in_7: f32, %in_8: f32, %out: f32):
              %44 = arith.mulf %in, %in_7 : f32
              %45 = arith.addf %44, %in_8 : f32
              linalg.yield %45 : f32
            }
            loom.semaphore_give %27 : memref<1x64xf32>
            %42 = arith.muli %arg7, %c16384 : index
            %43 = arith.addi %37, %42 : index
            %reinterpret_cast_6 = memref.reinterpret_cast %arg1 to offset: [%43], sizes: [1, 128, 128], strides: [524288, 128, 1] : memref<32x4096x128xf32> to memref<1x128x128xf32, strided<[524288, 128, 1], offset: ?>>
            loom.copy %reinterpret_cast_6, %15 src_mem_space @DRAM dst_mem_space @L1, interconnect : [@horizontal_links, @vertical_links], broadcast : [8, 8] : memref<1x128x128xf32, strided<[524288, 128, 1], offset: ?>>, memref<1x128x128xf32>
            linalg.fill ins(%cst_1 : f32) outs(%36 : memref<1x64x128xf32>)
            linalg.batch_matmul ins(%17, %15 : memref<1x64x128xf32>, memref<1x128x128xf32>) outs(%36 : memref<1x64x128xf32>)
            loom.semaphore_give %15 : memref<1x128x128xf32>
            loom.semaphore_give %17 : memref<1x64x128xf32>
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%36, %34, %29 : memref<1x64x128xf32>, memref<1x64x128xf32>, memref<1x64xf32>) outs(%34 : memref<1x64x128xf32>) {
            ^bb0(%in: f32, %in_7: f32, %in_8: f32, %out: f32):
              %44 = arith.mulf %in_7, %in_8 : f32
              %45 = arith.addf %in, %44 : f32
              linalg.yield %45 : f32
            }
            loom.semaphore_give %29 : memref<1x64xf32>
            loom.semaphore_give %36 : memref<1x64x128xf32>
            linalg.copy ins(%25 : memref<1x64xf32>) outs(%23 : memref<1x64xf32>)
            loom.semaphore_give %25 : memref<1x64xf32>
          }
          loom.semaphore_give %23 : memref<1x64xf32>
          loom.semaphore_give %32 : memref<1x64x128xf32>
          linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%34, %21 : memref<1x64x128xf32>, memref<1x64xf32>) outs(%31 : memref<1x64x128xf32>) {
          ^bb0(%in: f32, %in_5: f32, %out: f32):
            %40 = arith.divf %in, %in_5 : f32
            linalg.yield %40 : f32
          }
          loom.semaphore_give %21 : memref<1x64xf32>
          loom.semaphore_give %34 : memref<1x64x128xf32>
          %reinterpret_cast_4 = memref.reinterpret_cast %arg3 to offset: [%39], sizes: [1, 64, 128], strides: [524288, 128, 1] : memref<32x4096x128xf32> to memref<1x64x128xf32, strided<[524288, 128, 1], offset: ?>>
          loom.copy %31, %reinterpret_cast_4 src_mem_space @L1 dst_mem_space @DRAM, interconnect : [], broadcast : [1, 1] : memref<1x64x128xf32>, memref<1x64x128xf32, strided<[524288, 128, 1], offset: ?>>
          loom.semaphore_give %31 : memref<1x64x128xf32>
        }
        scf.reduce 
      } {loom.iter_types = [#loom.iter_type<spatial>, #loom.iter_type<spatial>], loom.mapped_to_dims = [@x, @y]}
      return
    }
  }
  module attributes {loom.pass_name = "Materialize"} {
    func.func @attention__d0i1_d1i1__f10__d_v_h__BB1__BM64__BN128(%arg0: memref<32x128x4096xf32>, %arg1: memref<32x4096x128xf32>, %arg2: memref<32x4096x128xf32>, %arg3: memref<32x4096x128xf32>) {
      %c16384 = arith.constant 16384 : index
      %c8192 = arith.constant 8192 : index
      %c524288 = arith.constant 524288 : index
      %c32 = arith.constant 32 : index
      %cst = arith.constant 2.000000e+00 : f32
      %cst_0 = arith.constant 0.12751743074602467 : f64
      %cst_1 = arith.constant 0.000000e+00 : f32
      %cst_2 = arith.constant 1.000000e+00 : f32
      %cst_3 = arith.constant 0xFF800000 : f32
      %c128 = arith.constant 128 : index
      %c0 = arith.constant 0 : index
      %c8 = arith.constant 8 : index
      %c1 = arith.constant 1 : index
      scf.parallel (%arg4, %arg5) = (%c0, %c0) to (%c8, %c8) step (%c1, %c1) {
        scf.for %arg6 = %c0 to %c32 step %c1 {
          %12 = arith.muli %arg4, %c8 overflow<nsw> : index
          %13 = arith.addi %12, %arg5 : index
          %14 = loom.alloc [1, 128, 128] on @L1 : memref<1x128x128xf32>
          %15 = loom.semaphore_take %14 : memref<1x128x128xf32> -> memref<1x128x128xf32>
          %16 = loom.alloc [1, 64, 128] on @L1 : memref<1x64x128xf32>
          %17 = loom.semaphore_take %16 : memref<1x64x128xf32> -> memref<1x64x128xf32>
          %18 = loom.alloc [1, 128, 128] on @L1 : memref<1x128x128xf32>
          %19 = loom.semaphore_take %18 : memref<1x128x128xf32> -> memref<1x128x128xf32>
          %20 = loom.alloc [1, 64] on @L1 : memref<1x64xf32>
          %21 = loom.semaphore_take %20 : memref<1x64xf32> -> memref<1x64xf32>
          %22 = loom.alloc [1, 64] on @L1 : memref<1x64xf32>
          %23 = loom.semaphore_take %22 : memref<1x64xf32> -> memref<1x64xf32>
          %24 = loom.alloc [1, 64] on @L1 : memref<1x64xf32>
          %25 = loom.semaphore_take %24 : memref<1x64xf32> -> memref<1x64xf32>
          %26 = loom.alloc [1, 64] on @L1 : memref<1x64xf32>
          %27 = loom.semaphore_take %26 : memref<1x64xf32> -> memref<1x64xf32>
          %28 = loom.alloc [1, 64] on @L1 : memref<1x64xf32>
          %29 = loom.semaphore_take %28 : memref<1x64xf32> -> memref<1x64xf32>
          %30 = loom.alloc [1, 64, 128] on @L1 : memref<1x64x128xf32>
          %31 = loom.semaphore_take %30 : memref<1x64x128xf32> -> memref<1x64x128xf32>
          %32 = loom.semaphore_take %30 : memref<1x64x128xf32> -> memref<1x64x128xf32>
          %33 = loom.alloc [1, 64, 128] on @L1 : memref<1x64x128xf32>
          %34 = loom.semaphore_take %33 : memref<1x64x128xf32> -> memref<1x64x128xf32>
          %35 = loom.alloc [1, 64, 128] on @L1 : memref<1x64x128xf32>
          %36 = loom.semaphore_take %35 : memref<1x64x128xf32> -> memref<1x64x128xf32>
          %37 = arith.muli %arg6, %c524288 overflow<nsw> : index
          %38 = arith.muli %13, %c8192 : index
          %39 = arith.addi %37, %38 : index
          %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%39], sizes: [1, 64, 128], strides: [524288, 128, 1] : memref<32x4096x128xf32> to memref<1x64x128xf32, strided<[524288, 128, 1], offset: ?>>
          loom.copy %reinterpret_cast, %32 src_mem_space @DRAM dst_mem_space @L1, interconnect : [], broadcast : [1, 1] : memref<1x64x128xf32, strided<[524288, 128, 1], offset: ?>>, memref<1x64x128xf32>
          linalg.fill ins(%cst_1 : f32) outs(%34 : memref<1x64x128xf32>)
          linalg.fill ins(%cst_2 : f32) outs(%21 : memref<1x64xf32>)
          linalg.fill ins(%cst_3 : f32) outs(%23 : memref<1x64xf32>)
          scf.for %arg7 = %c0 to %c32 step %c1 {
            %40 = arith.muli %arg7, %c128 : index
            %41 = arith.addi %37, %40 : index
            %reinterpret_cast_5 = memref.reinterpret_cast %arg0 to offset: [%41], sizes: [1, 128, 128], strides: [524288, 4096, 1] : memref<32x128x4096xf32> to memref<1x128x128xf32, strided<[524288, 4096, 1], offset: ?>>
            loom.copy %reinterpret_cast_5, %19 src_mem_space @DRAM dst_mem_space @L1, interconnect : [@vertical_links], broadcast : [8, 1] : memref<1x128x128xf32, strided<[524288, 4096, 1], offset: ?>>, memref<1x128x128xf32>
            linalg.fill ins(%cst_1 : f32) outs(%17 : memref<1x64x128xf32>)
            linalg.batch_matmul ins(%32, %19 : memref<1x64x128xf32>, memref<1x128x128xf32>) outs(%17 : memref<1x64x128xf32>)
            loom.semaphore_give %19 : memref<1x128x128xf32>
            linalg.fill ins(%cst_3 : f32) outs(%25 : memref<1x64xf32>)
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%17 : memref<1x64x128xf32>) outs(%25 : memref<1x64xf32>) {
            ^bb0(%in: f32, %out: f32):
              %44 = arith.maximumf %in, %out : f32
              linalg.yield %44 : f32
            }
            linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%23, %25 : memref<1x64xf32>, memref<1x64xf32>) outs(%25 : memref<1x64xf32>) {
            ^bb0(%in: f32, %in_7: f32, %out: f32):
              %44 = arith.truncf %cst_0 : f64 to f32
              %45 = arith.mulf %in_7, %44 : f32
              %46 = arith.cmpf ogt, %in, %45 : f32
              %47 = arith.select %46, %in, %45 : f32
              linalg.yield %47 : f32
            }
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%17, %25 : memref<1x64x128xf32>, memref<1x64xf32>) outs(%17 : memref<1x64x128xf32>) {
            ^bb0(%in: f32, %in_7: f32, %out: f32):
              %44 = arith.truncf %cst_0 : f64 to f32
              %45 = arith.mulf %in, %44 : f32
              %46 = arith.subf %45, %in_7 : f32
              %47 = math.powf %cst, %46 : f32
              linalg.yield %47 : f32
            }
            linalg.fill ins(%cst_1 : f32) outs(%27 : memref<1x64xf32>)
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%17 : memref<1x64x128xf32>) outs(%27 : memref<1x64xf32>) {
            ^bb0(%in: f32, %out: f32):
              %44 = arith.addf %in, %out : f32
              linalg.yield %44 : f32
            }
            linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%23, %25 : memref<1x64xf32>, memref<1x64xf32>) outs(%29 : memref<1x64xf32>) {
            ^bb0(%in: f32, %in_7: f32, %out: f32):
              %44 = arith.subf %in, %in_7 : f32
              %45 = math.powf %cst, %44 : f32
              linalg.yield %45 : f32
            }
            linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%21, %29, %27 : memref<1x64xf32>, memref<1x64xf32>, memref<1x64xf32>) outs(%21 : memref<1x64xf32>) {
            ^bb0(%in: f32, %in_7: f32, %in_8: f32, %out: f32):
              %44 = arith.mulf %in, %in_7 : f32
              %45 = arith.addf %44, %in_8 : f32
              linalg.yield %45 : f32
            }
            loom.semaphore_give %27 : memref<1x64xf32>
            %42 = arith.muli %arg7, %c16384 : index
            %43 = arith.addi %37, %42 : index
            %reinterpret_cast_6 = memref.reinterpret_cast %arg1 to offset: [%43], sizes: [1, 128, 128], strides: [524288, 128, 1] : memref<32x4096x128xf32> to memref<1x128x128xf32, strided<[524288, 128, 1], offset: ?>>
            loom.copy %reinterpret_cast_6, %15 src_mem_space @DRAM dst_mem_space @L1, interconnect : [@horizontal_links], broadcast : [1, 8] : memref<1x128x128xf32, strided<[524288, 128, 1], offset: ?>>, memref<1x128x128xf32>
            linalg.fill ins(%cst_1 : f32) outs(%36 : memref<1x64x128xf32>)
            linalg.batch_matmul ins(%17, %15 : memref<1x64x128xf32>, memref<1x128x128xf32>) outs(%36 : memref<1x64x128xf32>)
            loom.semaphore_give %15 : memref<1x128x128xf32>
            loom.semaphore_give %17 : memref<1x64x128xf32>
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%36, %34, %29 : memref<1x64x128xf32>, memref<1x64x128xf32>, memref<1x64xf32>) outs(%34 : memref<1x64x128xf32>) {
            ^bb0(%in: f32, %in_7: f32, %in_8: f32, %out: f32):
              %44 = arith.mulf %in_7, %in_8 : f32
              %45 = arith.addf %in, %44 : f32
              linalg.yield %45 : f32
            }
            loom.semaphore_give %29 : memref<1x64xf32>
            loom.semaphore_give %36 : memref<1x64x128xf32>
            linalg.copy ins(%25 : memref<1x64xf32>) outs(%23 : memref<1x64xf32>)
            loom.semaphore_give %25 : memref<1x64xf32>
          }
          loom.semaphore_give %23 : memref<1x64xf32>
          loom.semaphore_give %32 : memref<1x64x128xf32>
          linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%34, %21 : memref<1x64x128xf32>, memref<1x64xf32>) outs(%31 : memref<1x64x128xf32>) {
          ^bb0(%in: f32, %in_5: f32, %out: f32):
            %40 = arith.divf %in, %in_5 : f32
            linalg.yield %40 : f32
          }
          loom.semaphore_give %21 : memref<1x64xf32>
          loom.semaphore_give %34 : memref<1x64x128xf32>
          %reinterpret_cast_4 = memref.reinterpret_cast %arg3 to offset: [%39], sizes: [1, 64, 128], strides: [524288, 128, 1] : memref<32x4096x128xf32> to memref<1x64x128xf32, strided<[524288, 128, 1], offset: ?>>
          loom.copy %31, %reinterpret_cast_4 src_mem_space @L1 dst_mem_space @DRAM, interconnect : [], broadcast : [1, 1] : memref<1x64x128xf32>, memref<1x64x128xf32, strided<[524288, 128, 1], offset: ?>>
          loom.semaphore_give %31 : memref<1x64x128xf32>
        }
        scf.reduce 
      } {loom.iter_types = [#loom.iter_type<spatial>, #loom.iter_type<spatial>], loom.mapped_to_dims = [@x, @y]}
      return
    }
  }
  module attributes {loom.pass_name = "Materialize"} {
    func.func @attention__d0i1_d1i1__f10__d_v_v__BB1__BM64__BN128(%arg0: memref<32x128x4096xf32>, %arg1: memref<32x4096x128xf32>, %arg2: memref<32x4096x128xf32>, %arg3: memref<32x4096x128xf32>) {
      %c16384 = arith.constant 16384 : index
      %c8192 = arith.constant 8192 : index
      %c524288 = arith.constant 524288 : index
      %c32 = arith.constant 32 : index
      %cst = arith.constant 2.000000e+00 : f32
      %cst_0 = arith.constant 0.12751743074602467 : f64
      %cst_1 = arith.constant 0.000000e+00 : f32
      %cst_2 = arith.constant 1.000000e+00 : f32
      %cst_3 = arith.constant 0xFF800000 : f32
      %c128 = arith.constant 128 : index
      %c0 = arith.constant 0 : index
      %c8 = arith.constant 8 : index
      %c1 = arith.constant 1 : index
      scf.parallel (%arg4, %arg5) = (%c0, %c0) to (%c8, %c8) step (%c1, %c1) {
        scf.for %arg6 = %c0 to %c32 step %c1 {
          %12 = arith.muli %arg4, %c8 overflow<nsw> : index
          %13 = arith.addi %12, %arg5 : index
          %14 = loom.alloc [1, 128, 128] on @L1 : memref<1x128x128xf32>
          %15 = loom.semaphore_take %14 : memref<1x128x128xf32> -> memref<1x128x128xf32>
          %16 = loom.alloc [1, 64, 128] on @L1 : memref<1x64x128xf32>
          %17 = loom.semaphore_take %16 : memref<1x64x128xf32> -> memref<1x64x128xf32>
          %18 = loom.alloc [1, 128, 128] on @L1 : memref<1x128x128xf32>
          %19 = loom.semaphore_take %18 : memref<1x128x128xf32> -> memref<1x128x128xf32>
          %20 = loom.alloc [1, 64] on @L1 : memref<1x64xf32>
          %21 = loom.semaphore_take %20 : memref<1x64xf32> -> memref<1x64xf32>
          %22 = loom.alloc [1, 64] on @L1 : memref<1x64xf32>
          %23 = loom.semaphore_take %22 : memref<1x64xf32> -> memref<1x64xf32>
          %24 = loom.alloc [1, 64] on @L1 : memref<1x64xf32>
          %25 = loom.semaphore_take %24 : memref<1x64xf32> -> memref<1x64xf32>
          %26 = loom.alloc [1, 64] on @L1 : memref<1x64xf32>
          %27 = loom.semaphore_take %26 : memref<1x64xf32> -> memref<1x64xf32>
          %28 = loom.alloc [1, 64] on @L1 : memref<1x64xf32>
          %29 = loom.semaphore_take %28 : memref<1x64xf32> -> memref<1x64xf32>
          %30 = loom.alloc [1, 64, 128] on @L1 : memref<1x64x128xf32>
          %31 = loom.semaphore_take %30 : memref<1x64x128xf32> -> memref<1x64x128xf32>
          %32 = loom.semaphore_take %30 : memref<1x64x128xf32> -> memref<1x64x128xf32>
          %33 = loom.alloc [1, 64, 128] on @L1 : memref<1x64x128xf32>
          %34 = loom.semaphore_take %33 : memref<1x64x128xf32> -> memref<1x64x128xf32>
          %35 = loom.alloc [1, 64, 128] on @L1 : memref<1x64x128xf32>
          %36 = loom.semaphore_take %35 : memref<1x64x128xf32> -> memref<1x64x128xf32>
          %37 = arith.muli %arg6, %c524288 overflow<nsw> : index
          %38 = arith.muli %13, %c8192 : index
          %39 = arith.addi %37, %38 : index
          %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%39], sizes: [1, 64, 128], strides: [524288, 128, 1] : memref<32x4096x128xf32> to memref<1x64x128xf32, strided<[524288, 128, 1], offset: ?>>
          loom.copy %reinterpret_cast, %32 src_mem_space @DRAM dst_mem_space @L1, interconnect : [], broadcast : [1, 1] : memref<1x64x128xf32, strided<[524288, 128, 1], offset: ?>>, memref<1x64x128xf32>
          linalg.fill ins(%cst_1 : f32) outs(%34 : memref<1x64x128xf32>)
          linalg.fill ins(%cst_2 : f32) outs(%21 : memref<1x64xf32>)
          linalg.fill ins(%cst_3 : f32) outs(%23 : memref<1x64xf32>)
          scf.for %arg7 = %c0 to %c32 step %c1 {
            %40 = arith.muli %arg7, %c128 : index
            %41 = arith.addi %37, %40 : index
            %reinterpret_cast_5 = memref.reinterpret_cast %arg0 to offset: [%41], sizes: [1, 128, 128], strides: [524288, 4096, 1] : memref<32x128x4096xf32> to memref<1x128x128xf32, strided<[524288, 4096, 1], offset: ?>>
            loom.copy %reinterpret_cast_5, %19 src_mem_space @DRAM dst_mem_space @L1, interconnect : [@vertical_links], broadcast : [8, 1] : memref<1x128x128xf32, strided<[524288, 4096, 1], offset: ?>>, memref<1x128x128xf32>
            linalg.fill ins(%cst_1 : f32) outs(%17 : memref<1x64x128xf32>)
            linalg.batch_matmul ins(%32, %19 : memref<1x64x128xf32>, memref<1x128x128xf32>) outs(%17 : memref<1x64x128xf32>)
            loom.semaphore_give %19 : memref<1x128x128xf32>
            linalg.fill ins(%cst_3 : f32) outs(%25 : memref<1x64xf32>)
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%17 : memref<1x64x128xf32>) outs(%25 : memref<1x64xf32>) {
            ^bb0(%in: f32, %out: f32):
              %44 = arith.maximumf %in, %out : f32
              linalg.yield %44 : f32
            }
            linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%23, %25 : memref<1x64xf32>, memref<1x64xf32>) outs(%25 : memref<1x64xf32>) {
            ^bb0(%in: f32, %in_7: f32, %out: f32):
              %44 = arith.truncf %cst_0 : f64 to f32
              %45 = arith.mulf %in_7, %44 : f32
              %46 = arith.cmpf ogt, %in, %45 : f32
              %47 = arith.select %46, %in, %45 : f32
              linalg.yield %47 : f32
            }
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%17, %25 : memref<1x64x128xf32>, memref<1x64xf32>) outs(%17 : memref<1x64x128xf32>) {
            ^bb0(%in: f32, %in_7: f32, %out: f32):
              %44 = arith.truncf %cst_0 : f64 to f32
              %45 = arith.mulf %in, %44 : f32
              %46 = arith.subf %45, %in_7 : f32
              %47 = math.powf %cst, %46 : f32
              linalg.yield %47 : f32
            }
            linalg.fill ins(%cst_1 : f32) outs(%27 : memref<1x64xf32>)
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%17 : memref<1x64x128xf32>) outs(%27 : memref<1x64xf32>) {
            ^bb0(%in: f32, %out: f32):
              %44 = arith.addf %in, %out : f32
              linalg.yield %44 : f32
            }
            linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%23, %25 : memref<1x64xf32>, memref<1x64xf32>) outs(%29 : memref<1x64xf32>) {
            ^bb0(%in: f32, %in_7: f32, %out: f32):
              %44 = arith.subf %in, %in_7 : f32
              %45 = math.powf %cst, %44 : f32
              linalg.yield %45 : f32
            }
            linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%21, %29, %27 : memref<1x64xf32>, memref<1x64xf32>, memref<1x64xf32>) outs(%21 : memref<1x64xf32>) {
            ^bb0(%in: f32, %in_7: f32, %in_8: f32, %out: f32):
              %44 = arith.mulf %in, %in_7 : f32
              %45 = arith.addf %44, %in_8 : f32
              linalg.yield %45 : f32
            }
            loom.semaphore_give %27 : memref<1x64xf32>
            %42 = arith.muli %arg7, %c16384 : index
            %43 = arith.addi %37, %42 : index
            %reinterpret_cast_6 = memref.reinterpret_cast %arg1 to offset: [%43], sizes: [1, 128, 128], strides: [524288, 128, 1] : memref<32x4096x128xf32> to memref<1x128x128xf32, strided<[524288, 128, 1], offset: ?>>
            loom.copy %reinterpret_cast_6, %15 src_mem_space @DRAM dst_mem_space @L1, interconnect : [@vertical_links], broadcast : [8, 1] : memref<1x128x128xf32, strided<[524288, 128, 1], offset: ?>>, memref<1x128x128xf32>
            linalg.fill ins(%cst_1 : f32) outs(%36 : memref<1x64x128xf32>)
            linalg.batch_matmul ins(%17, %15 : memref<1x64x128xf32>, memref<1x128x128xf32>) outs(%36 : memref<1x64x128xf32>)
            loom.semaphore_give %15 : memref<1x128x128xf32>
            loom.semaphore_give %17 : memref<1x64x128xf32>
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%36, %34, %29 : memref<1x64x128xf32>, memref<1x64x128xf32>, memref<1x64xf32>) outs(%34 : memref<1x64x128xf32>) {
            ^bb0(%in: f32, %in_7: f32, %in_8: f32, %out: f32):
              %44 = arith.mulf %in_7, %in_8 : f32
              %45 = arith.addf %in, %44 : f32
              linalg.yield %45 : f32
            }
            loom.semaphore_give %29 : memref<1x64xf32>
            loom.semaphore_give %36 : memref<1x64x128xf32>
            linalg.copy ins(%25 : memref<1x64xf32>) outs(%23 : memref<1x64xf32>)
            loom.semaphore_give %25 : memref<1x64xf32>
          }
          loom.semaphore_give %23 : memref<1x64xf32>
          loom.semaphore_give %32 : memref<1x64x128xf32>
          linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%34, %21 : memref<1x64x128xf32>, memref<1x64xf32>) outs(%31 : memref<1x64x128xf32>) {
          ^bb0(%in: f32, %in_5: f32, %out: f32):
            %40 = arith.divf %in, %in_5 : f32
            linalg.yield %40 : f32
          }
          loom.semaphore_give %21 : memref<1x64xf32>
          loom.semaphore_give %34 : memref<1x64x128xf32>
          %reinterpret_cast_4 = memref.reinterpret_cast %arg3 to offset: [%39], sizes: [1, 64, 128], strides: [524288, 128, 1] : memref<32x4096x128xf32> to memref<1x64x128xf32, strided<[524288, 128, 1], offset: ?>>
          loom.copy %31, %reinterpret_cast_4 src_mem_space @L1 dst_mem_space @DRAM, interconnect : [], broadcast : [1, 1] : memref<1x64x128xf32>, memref<1x64x128xf32, strided<[524288, 128, 1], offset: ?>>
          loom.semaphore_give %31 : memref<1x64x128xf32>
        }
        scf.reduce 
      } {loom.iter_types = [#loom.iter_type<spatial>, #loom.iter_type<spatial>], loom.mapped_to_dims = [@x, @y]}
      return
    }
  }
  module attributes {loom.pass_name = "Materialize"} {
    func.func @attention__d1i1_d0i1__f01__d_d_d__BB1__BM64__BN128(%arg0: memref<32x128x4096xf32>, %arg1: memref<32x4096x128xf32>, %arg2: memref<32x4096x128xf32>, %arg3: memref<32x4096x128xf32>) {
      %c16384 = arith.constant 16384 : index
      %c8192 = arith.constant 8192 : index
      %c524288 = arith.constant 524288 : index
      %c32 = arith.constant 32 : index
      %cst = arith.constant 2.000000e+00 : f32
      %cst_0 = arith.constant 0.12751743074602467 : f64
      %cst_1 = arith.constant 0.000000e+00 : f32
      %cst_2 = arith.constant 1.000000e+00 : f32
      %cst_3 = arith.constant 0xFF800000 : f32
      %c128 = arith.constant 128 : index
      %c0 = arith.constant 0 : index
      %c8 = arith.constant 8 : index
      %c1 = arith.constant 1 : index
      scf.parallel (%arg4, %arg5) = (%c0, %c0) to (%c8, %c8) step (%c1, %c1) {
        scf.for %arg6 = %c0 to %c32 step %c1 {
          %12 = arith.muli %arg4, %c8 overflow<nsw> : index
          %13 = arith.addi %12, %arg5 : index
          %14 = loom.alloc [1, 128, 128] on @L1 : memref<1x128x128xf32>
          %15 = loom.semaphore_take %14 : memref<1x128x128xf32> -> memref<1x128x128xf32>
          %16 = loom.alloc [1, 64, 128] on @L1 : memref<1x64x128xf32>
          %17 = loom.semaphore_take %16 : memref<1x64x128xf32> -> memref<1x64x128xf32>
          %18 = loom.alloc [1, 128, 128] on @L1 : memref<1x128x128xf32>
          %19 = loom.semaphore_take %18 : memref<1x128x128xf32> -> memref<1x128x128xf32>
          %20 = loom.alloc [1, 64] on @L1 : memref<1x64xf32>
          %21 = loom.semaphore_take %20 : memref<1x64xf32> -> memref<1x64xf32>
          %22 = loom.alloc [1, 64] on @L1 : memref<1x64xf32>
          %23 = loom.semaphore_take %22 : memref<1x64xf32> -> memref<1x64xf32>
          %24 = loom.alloc [1, 64] on @L1 : memref<1x64xf32>
          %25 = loom.semaphore_take %24 : memref<1x64xf32> -> memref<1x64xf32>
          %26 = loom.alloc [1, 64] on @L1 : memref<1x64xf32>
          %27 = loom.semaphore_take %26 : memref<1x64xf32> -> memref<1x64xf32>
          %28 = loom.alloc [1, 64] on @L1 : memref<1x64xf32>
          %29 = loom.semaphore_take %28 : memref<1x64xf32> -> memref<1x64xf32>
          %30 = loom.alloc [1, 64, 128] on @L1 : memref<1x64x128xf32>
          %31 = loom.semaphore_take %30 : memref<1x64x128xf32> -> memref<1x64x128xf32>
          %32 = loom.semaphore_take %30 : memref<1x64x128xf32> -> memref<1x64x128xf32>
          %33 = loom.alloc [1, 64, 128] on @L1 : memref<1x64x128xf32>
          %34 = loom.semaphore_take %33 : memref<1x64x128xf32> -> memref<1x64x128xf32>
          %35 = loom.alloc [1, 64, 128] on @L1 : memref<1x64x128xf32>
          %36 = loom.semaphore_take %35 : memref<1x64x128xf32> -> memref<1x64x128xf32>
          %37 = arith.muli %arg6, %c524288 overflow<nsw> : index
          %38 = arith.muli %13, %c8192 : index
          %39 = arith.addi %37, %38 : index
          %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%39], sizes: [1, 64, 128], strides: [524288, 128, 1] : memref<32x4096x128xf32> to memref<1x64x128xf32, strided<[524288, 128, 1], offset: ?>>
          loom.copy %reinterpret_cast, %32 src_mem_space @DRAM dst_mem_space @L1, interconnect : [], broadcast : [1, 1] : memref<1x64x128xf32, strided<[524288, 128, 1], offset: ?>>, memref<1x64x128xf32>
          linalg.fill ins(%cst_1 : f32) outs(%34 : memref<1x64x128xf32>)
          linalg.fill ins(%cst_2 : f32) outs(%21 : memref<1x64xf32>)
          linalg.fill ins(%cst_3 : f32) outs(%23 : memref<1x64xf32>)
          scf.for %arg7 = %c0 to %c32 step %c1 {
            %40 = arith.muli %arg7, %c128 : index
            %41 = arith.addi %37, %40 : index
            %reinterpret_cast_5 = memref.reinterpret_cast %arg0 to offset: [%41], sizes: [1, 128, 128], strides: [524288, 4096, 1] : memref<32x128x4096xf32> to memref<1x128x128xf32, strided<[524288, 4096, 1], offset: ?>>
            loom.copy %reinterpret_cast_5, %19 src_mem_space @DRAM dst_mem_space @L1, interconnect : [], broadcast : [1, 1] : memref<1x128x128xf32, strided<[524288, 4096, 1], offset: ?>>, memref<1x128x128xf32>
            linalg.fill ins(%cst_1 : f32) outs(%17 : memref<1x64x128xf32>)
            linalg.batch_matmul ins(%32, %19 : memref<1x64x128xf32>, memref<1x128x128xf32>) outs(%17 : memref<1x64x128xf32>)
            loom.semaphore_give %19 : memref<1x128x128xf32>
            linalg.fill ins(%cst_3 : f32) outs(%25 : memref<1x64xf32>)
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%17 : memref<1x64x128xf32>) outs(%25 : memref<1x64xf32>) {
            ^bb0(%in: f32, %out: f32):
              %44 = arith.maximumf %in, %out : f32
              linalg.yield %44 : f32
            }
            linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%23, %25 : memref<1x64xf32>, memref<1x64xf32>) outs(%25 : memref<1x64xf32>) {
            ^bb0(%in: f32, %in_7: f32, %out: f32):
              %44 = arith.truncf %cst_0 : f64 to f32
              %45 = arith.mulf %in_7, %44 : f32
              %46 = arith.cmpf ogt, %in, %45 : f32
              %47 = arith.select %46, %in, %45 : f32
              linalg.yield %47 : f32
            }
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%17, %25 : memref<1x64x128xf32>, memref<1x64xf32>) outs(%17 : memref<1x64x128xf32>) {
            ^bb0(%in: f32, %in_7: f32, %out: f32):
              %44 = arith.truncf %cst_0 : f64 to f32
              %45 = arith.mulf %in, %44 : f32
              %46 = arith.subf %45, %in_7 : f32
              %47 = math.powf %cst, %46 : f32
              linalg.yield %47 : f32
            }
            linalg.fill ins(%cst_1 : f32) outs(%27 : memref<1x64xf32>)
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%17 : memref<1x64x128xf32>) outs(%27 : memref<1x64xf32>) {
            ^bb0(%in: f32, %out: f32):
              %44 = arith.addf %in, %out : f32
              linalg.yield %44 : f32
            }
            linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%23, %25 : memref<1x64xf32>, memref<1x64xf32>) outs(%29 : memref<1x64xf32>) {
            ^bb0(%in: f32, %in_7: f32, %out: f32):
              %44 = arith.subf %in, %in_7 : f32
              %45 = math.powf %cst, %44 : f32
              linalg.yield %45 : f32
            }
            linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%21, %29, %27 : memref<1x64xf32>, memref<1x64xf32>, memref<1x64xf32>) outs(%21 : memref<1x64xf32>) {
            ^bb0(%in: f32, %in_7: f32, %in_8: f32, %out: f32):
              %44 = arith.mulf %in, %in_7 : f32
              %45 = arith.addf %44, %in_8 : f32
              linalg.yield %45 : f32
            }
            loom.semaphore_give %27 : memref<1x64xf32>
            %42 = arith.muli %arg7, %c16384 : index
            %43 = arith.addi %37, %42 : index
            %reinterpret_cast_6 = memref.reinterpret_cast %arg1 to offset: [%43], sizes: [1, 128, 128], strides: [524288, 128, 1] : memref<32x4096x128xf32> to memref<1x128x128xf32, strided<[524288, 128, 1], offset: ?>>
            loom.copy %reinterpret_cast_6, %15 src_mem_space @DRAM dst_mem_space @L1, interconnect : [], broadcast : [1, 1] : memref<1x128x128xf32, strided<[524288, 128, 1], offset: ?>>, memref<1x128x128xf32>
            linalg.fill ins(%cst_1 : f32) outs(%36 : memref<1x64x128xf32>)
            linalg.batch_matmul ins(%17, %15 : memref<1x64x128xf32>, memref<1x128x128xf32>) outs(%36 : memref<1x64x128xf32>)
            loom.semaphore_give %15 : memref<1x128x128xf32>
            loom.semaphore_give %17 : memref<1x64x128xf32>
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%36, %34, %29 : memref<1x64x128xf32>, memref<1x64x128xf32>, memref<1x64xf32>) outs(%34 : memref<1x64x128xf32>) {
            ^bb0(%in: f32, %in_7: f32, %in_8: f32, %out: f32):
              %44 = arith.mulf %in_7, %in_8 : f32
              %45 = arith.addf %in, %44 : f32
              linalg.yield %45 : f32
            }
            loom.semaphore_give %29 : memref<1x64xf32>
            loom.semaphore_give %36 : memref<1x64x128xf32>
            linalg.copy ins(%25 : memref<1x64xf32>) outs(%23 : memref<1x64xf32>)
            loom.semaphore_give %25 : memref<1x64xf32>
          }
          loom.semaphore_give %23 : memref<1x64xf32>
          loom.semaphore_give %32 : memref<1x64x128xf32>
          linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%34, %21 : memref<1x64x128xf32>, memref<1x64xf32>) outs(%31 : memref<1x64x128xf32>) {
          ^bb0(%in: f32, %in_5: f32, %out: f32):
            %40 = arith.divf %in, %in_5 : f32
            linalg.yield %40 : f32
          }
          loom.semaphore_give %21 : memref<1x64xf32>
          loom.semaphore_give %34 : memref<1x64x128xf32>
          %reinterpret_cast_4 = memref.reinterpret_cast %arg3 to offset: [%39], sizes: [1, 64, 128], strides: [524288, 128, 1] : memref<32x4096x128xf32> to memref<1x64x128xf32, strided<[524288, 128, 1], offset: ?>>
          loom.copy %31, %reinterpret_cast_4 src_mem_space @L1 dst_mem_space @DRAM, interconnect : [], broadcast : [1, 1] : memref<1x64x128xf32>, memref<1x64x128xf32, strided<[524288, 128, 1], offset: ?>>
          loom.semaphore_give %31 : memref<1x64x128xf32>
        }
        scf.reduce 
      } {loom.iter_types = [#loom.iter_type<spatial>, #loom.iter_type<spatial>], loom.mapped_to_dims = [@y, @x]}
      return
    }
  }
  module attributes {loom.pass_name = "Materialize"} {
    func.func @attention__d1i1_d0i1__f01__d_d_a__BB1__BM64__BN128(%arg0: memref<32x128x4096xf32>, %arg1: memref<32x4096x128xf32>, %arg2: memref<32x4096x128xf32>, %arg3: memref<32x4096x128xf32>) {
      %c16384 = arith.constant 16384 : index
      %c8192 = arith.constant 8192 : index
      %c524288 = arith.constant 524288 : index
      %c32 = arith.constant 32 : index
      %cst = arith.constant 2.000000e+00 : f32
      %cst_0 = arith.constant 0.12751743074602467 : f64
      %cst_1 = arith.constant 0.000000e+00 : f32
      %cst_2 = arith.constant 1.000000e+00 : f32
      %cst_3 = arith.constant 0xFF800000 : f32
      %c128 = arith.constant 128 : index
      %c0 = arith.constant 0 : index
      %c8 = arith.constant 8 : index
      %c1 = arith.constant 1 : index
      scf.parallel (%arg4, %arg5) = (%c0, %c0) to (%c8, %c8) step (%c1, %c1) {
        scf.for %arg6 = %c0 to %c32 step %c1 {
          %12 = arith.muli %arg4, %c8 overflow<nsw> : index
          %13 = arith.addi %12, %arg5 : index
          %14 = loom.alloc [1, 128, 128] on @L1 : memref<1x128x128xf32>
          %15 = loom.semaphore_take %14 : memref<1x128x128xf32> -> memref<1x128x128xf32>
          %16 = loom.alloc [1, 64, 128] on @L1 : memref<1x64x128xf32>
          %17 = loom.semaphore_take %16 : memref<1x64x128xf32> -> memref<1x64x128xf32>
          %18 = loom.alloc [1, 128, 128] on @L1 : memref<1x128x128xf32>
          %19 = loom.semaphore_take %18 : memref<1x128x128xf32> -> memref<1x128x128xf32>
          %20 = loom.alloc [1, 64] on @L1 : memref<1x64xf32>
          %21 = loom.semaphore_take %20 : memref<1x64xf32> -> memref<1x64xf32>
          %22 = loom.alloc [1, 64] on @L1 : memref<1x64xf32>
          %23 = loom.semaphore_take %22 : memref<1x64xf32> -> memref<1x64xf32>
          %24 = loom.alloc [1, 64] on @L1 : memref<1x64xf32>
          %25 = loom.semaphore_take %24 : memref<1x64xf32> -> memref<1x64xf32>
          %26 = loom.alloc [1, 64] on @L1 : memref<1x64xf32>
          %27 = loom.semaphore_take %26 : memref<1x64xf32> -> memref<1x64xf32>
          %28 = loom.alloc [1, 64] on @L1 : memref<1x64xf32>
          %29 = loom.semaphore_take %28 : memref<1x64xf32> -> memref<1x64xf32>
          %30 = loom.alloc [1, 64, 128] on @L1 : memref<1x64x128xf32>
          %31 = loom.semaphore_take %30 : memref<1x64x128xf32> -> memref<1x64x128xf32>
          %32 = loom.semaphore_take %30 : memref<1x64x128xf32> -> memref<1x64x128xf32>
          %33 = loom.alloc [1, 64, 128] on @L1 : memref<1x64x128xf32>
          %34 = loom.semaphore_take %33 : memref<1x64x128xf32> -> memref<1x64x128xf32>
          %35 = loom.alloc [1, 64, 128] on @L1 : memref<1x64x128xf32>
          %36 = loom.semaphore_take %35 : memref<1x64x128xf32> -> memref<1x64x128xf32>
          %37 = arith.muli %arg6, %c524288 overflow<nsw> : index
          %38 = arith.muli %13, %c8192 : index
          %39 = arith.addi %37, %38 : index
          %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%39], sizes: [1, 64, 128], strides: [524288, 128, 1] : memref<32x4096x128xf32> to memref<1x64x128xf32, strided<[524288, 128, 1], offset: ?>>
          loom.copy %reinterpret_cast, %32 src_mem_space @DRAM dst_mem_space @L1, interconnect : [], broadcast : [1, 1] : memref<1x64x128xf32, strided<[524288, 128, 1], offset: ?>>, memref<1x64x128xf32>
          linalg.fill ins(%cst_1 : f32) outs(%34 : memref<1x64x128xf32>)
          linalg.fill ins(%cst_2 : f32) outs(%21 : memref<1x64xf32>)
          linalg.fill ins(%cst_3 : f32) outs(%23 : memref<1x64xf32>)
          scf.for %arg7 = %c0 to %c32 step %c1 {
            %40 = arith.muli %arg7, %c128 : index
            %41 = arith.addi %37, %40 : index
            %reinterpret_cast_5 = memref.reinterpret_cast %arg0 to offset: [%41], sizes: [1, 128, 128], strides: [524288, 4096, 1] : memref<32x128x4096xf32> to memref<1x128x128xf32, strided<[524288, 4096, 1], offset: ?>>
            loom.copy %reinterpret_cast_5, %19 src_mem_space @DRAM dst_mem_space @L1, interconnect : [], broadcast : [1, 1] : memref<1x128x128xf32, strided<[524288, 4096, 1], offset: ?>>, memref<1x128x128xf32>
            linalg.fill ins(%cst_1 : f32) outs(%17 : memref<1x64x128xf32>)
            linalg.batch_matmul ins(%32, %19 : memref<1x64x128xf32>, memref<1x128x128xf32>) outs(%17 : memref<1x64x128xf32>)
            loom.semaphore_give %19 : memref<1x128x128xf32>
            linalg.fill ins(%cst_3 : f32) outs(%25 : memref<1x64xf32>)
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%17 : memref<1x64x128xf32>) outs(%25 : memref<1x64xf32>) {
            ^bb0(%in: f32, %out: f32):
              %44 = arith.maximumf %in, %out : f32
              linalg.yield %44 : f32
            }
            linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%23, %25 : memref<1x64xf32>, memref<1x64xf32>) outs(%25 : memref<1x64xf32>) {
            ^bb0(%in: f32, %in_7: f32, %out: f32):
              %44 = arith.truncf %cst_0 : f64 to f32
              %45 = arith.mulf %in_7, %44 : f32
              %46 = arith.cmpf ogt, %in, %45 : f32
              %47 = arith.select %46, %in, %45 : f32
              linalg.yield %47 : f32
            }
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%17, %25 : memref<1x64x128xf32>, memref<1x64xf32>) outs(%17 : memref<1x64x128xf32>) {
            ^bb0(%in: f32, %in_7: f32, %out: f32):
              %44 = arith.truncf %cst_0 : f64 to f32
              %45 = arith.mulf %in, %44 : f32
              %46 = arith.subf %45, %in_7 : f32
              %47 = math.powf %cst, %46 : f32
              linalg.yield %47 : f32
            }
            linalg.fill ins(%cst_1 : f32) outs(%27 : memref<1x64xf32>)
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%17 : memref<1x64x128xf32>) outs(%27 : memref<1x64xf32>) {
            ^bb0(%in: f32, %out: f32):
              %44 = arith.addf %in, %out : f32
              linalg.yield %44 : f32
            }
            linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%23, %25 : memref<1x64xf32>, memref<1x64xf32>) outs(%29 : memref<1x64xf32>) {
            ^bb0(%in: f32, %in_7: f32, %out: f32):
              %44 = arith.subf %in, %in_7 : f32
              %45 = math.powf %cst, %44 : f32
              linalg.yield %45 : f32
            }
            linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%21, %29, %27 : memref<1x64xf32>, memref<1x64xf32>, memref<1x64xf32>) outs(%21 : memref<1x64xf32>) {
            ^bb0(%in: f32, %in_7: f32, %in_8: f32, %out: f32):
              %44 = arith.mulf %in, %in_7 : f32
              %45 = arith.addf %44, %in_8 : f32
              linalg.yield %45 : f32
            }
            loom.semaphore_give %27 : memref<1x64xf32>
            %42 = arith.muli %arg7, %c16384 : index
            %43 = arith.addi %37, %42 : index
            %reinterpret_cast_6 = memref.reinterpret_cast %arg1 to offset: [%43], sizes: [1, 128, 128], strides: [524288, 128, 1] : memref<32x4096x128xf32> to memref<1x128x128xf32, strided<[524288, 128, 1], offset: ?>>
            loom.copy %reinterpret_cast_6, %15 src_mem_space @DRAM dst_mem_space @L1, interconnect : [@horizontal_links, @vertical_links], broadcast : [8, 8] : memref<1x128x128xf32, strided<[524288, 128, 1], offset: ?>>, memref<1x128x128xf32>
            linalg.fill ins(%cst_1 : f32) outs(%36 : memref<1x64x128xf32>)
            linalg.batch_matmul ins(%17, %15 : memref<1x64x128xf32>, memref<1x128x128xf32>) outs(%36 : memref<1x64x128xf32>)
            loom.semaphore_give %15 : memref<1x128x128xf32>
            loom.semaphore_give %17 : memref<1x64x128xf32>
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%36, %34, %29 : memref<1x64x128xf32>, memref<1x64x128xf32>, memref<1x64xf32>) outs(%34 : memref<1x64x128xf32>) {
            ^bb0(%in: f32, %in_7: f32, %in_8: f32, %out: f32):
              %44 = arith.mulf %in_7, %in_8 : f32
              %45 = arith.addf %in, %44 : f32
              linalg.yield %45 : f32
            }
            loom.semaphore_give %29 : memref<1x64xf32>
            loom.semaphore_give %36 : memref<1x64x128xf32>
            linalg.copy ins(%25 : memref<1x64xf32>) outs(%23 : memref<1x64xf32>)
            loom.semaphore_give %25 : memref<1x64xf32>
          }
          loom.semaphore_give %23 : memref<1x64xf32>
          loom.semaphore_give %32 : memref<1x64x128xf32>
          linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%34, %21 : memref<1x64x128xf32>, memref<1x64xf32>) outs(%31 : memref<1x64x128xf32>) {
          ^bb0(%in: f32, %in_5: f32, %out: f32):
            %40 = arith.divf %in, %in_5 : f32
            linalg.yield %40 : f32
          }
          loom.semaphore_give %21 : memref<1x64xf32>
          loom.semaphore_give %34 : memref<1x64x128xf32>
          %reinterpret_cast_4 = memref.reinterpret_cast %arg3 to offset: [%39], sizes: [1, 64, 128], strides: [524288, 128, 1] : memref<32x4096x128xf32> to memref<1x64x128xf32, strided<[524288, 128, 1], offset: ?>>
          loom.copy %31, %reinterpret_cast_4 src_mem_space @L1 dst_mem_space @DRAM, interconnect : [], broadcast : [1, 1] : memref<1x64x128xf32>, memref<1x64x128xf32, strided<[524288, 128, 1], offset: ?>>
          loom.semaphore_give %31 : memref<1x64x128xf32>
        }
        scf.reduce 
      } {loom.iter_types = [#loom.iter_type<spatial>, #loom.iter_type<spatial>], loom.mapped_to_dims = [@y, @x]}
      return
    }
  }
  module attributes {loom.pass_name = "Materialize"} {
    func.func @attention__d1i1_d0i1__f01__d_d_h__BB1__BM64__BN128(%arg0: memref<32x128x4096xf32>, %arg1: memref<32x4096x128xf32>, %arg2: memref<32x4096x128xf32>, %arg3: memref<32x4096x128xf32>) {
      %c16384 = arith.constant 16384 : index
      %c8192 = arith.constant 8192 : index
      %c524288 = arith.constant 524288 : index
      %c32 = arith.constant 32 : index
      %cst = arith.constant 2.000000e+00 : f32
      %cst_0 = arith.constant 0.12751743074602467 : f64
      %cst_1 = arith.constant 0.000000e+00 : f32
      %cst_2 = arith.constant 1.000000e+00 : f32
      %cst_3 = arith.constant 0xFF800000 : f32
      %c128 = arith.constant 128 : index
      %c0 = arith.constant 0 : index
      %c8 = arith.constant 8 : index
      %c1 = arith.constant 1 : index
      scf.parallel (%arg4, %arg5) = (%c0, %c0) to (%c8, %c8) step (%c1, %c1) {
        scf.for %arg6 = %c0 to %c32 step %c1 {
          %12 = arith.muli %arg4, %c8 overflow<nsw> : index
          %13 = arith.addi %12, %arg5 : index
          %14 = loom.alloc [1, 128, 128] on @L1 : memref<1x128x128xf32>
          %15 = loom.semaphore_take %14 : memref<1x128x128xf32> -> memref<1x128x128xf32>
          %16 = loom.alloc [1, 64, 128] on @L1 : memref<1x64x128xf32>
          %17 = loom.semaphore_take %16 : memref<1x64x128xf32> -> memref<1x64x128xf32>
          %18 = loom.alloc [1, 128, 128] on @L1 : memref<1x128x128xf32>
          %19 = loom.semaphore_take %18 : memref<1x128x128xf32> -> memref<1x128x128xf32>
          %20 = loom.alloc [1, 64] on @L1 : memref<1x64xf32>
          %21 = loom.semaphore_take %20 : memref<1x64xf32> -> memref<1x64xf32>
          %22 = loom.alloc [1, 64] on @L1 : memref<1x64xf32>
          %23 = loom.semaphore_take %22 : memref<1x64xf32> -> memref<1x64xf32>
          %24 = loom.alloc [1, 64] on @L1 : memref<1x64xf32>
          %25 = loom.semaphore_take %24 : memref<1x64xf32> -> memref<1x64xf32>
          %26 = loom.alloc [1, 64] on @L1 : memref<1x64xf32>
          %27 = loom.semaphore_take %26 : memref<1x64xf32> -> memref<1x64xf32>
          %28 = loom.alloc [1, 64] on @L1 : memref<1x64xf32>
          %29 = loom.semaphore_take %28 : memref<1x64xf32> -> memref<1x64xf32>
          %30 = loom.alloc [1, 64, 128] on @L1 : memref<1x64x128xf32>
          %31 = loom.semaphore_take %30 : memref<1x64x128xf32> -> memref<1x64x128xf32>
          %32 = loom.semaphore_take %30 : memref<1x64x128xf32> -> memref<1x64x128xf32>
          %33 = loom.alloc [1, 64, 128] on @L1 : memref<1x64x128xf32>
          %34 = loom.semaphore_take %33 : memref<1x64x128xf32> -> memref<1x64x128xf32>
          %35 = loom.alloc [1, 64, 128] on @L1 : memref<1x64x128xf32>
          %36 = loom.semaphore_take %35 : memref<1x64x128xf32> -> memref<1x64x128xf32>
          %37 = arith.muli %arg6, %c524288 overflow<nsw> : index
          %38 = arith.muli %13, %c8192 : index
          %39 = arith.addi %37, %38 : index
          %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%39], sizes: [1, 64, 128], strides: [524288, 128, 1] : memref<32x4096x128xf32> to memref<1x64x128xf32, strided<[524288, 128, 1], offset: ?>>
          loom.copy %reinterpret_cast, %32 src_mem_space @DRAM dst_mem_space @L1, interconnect : [], broadcast : [1, 1] : memref<1x64x128xf32, strided<[524288, 128, 1], offset: ?>>, memref<1x64x128xf32>
          linalg.fill ins(%cst_1 : f32) outs(%34 : memref<1x64x128xf32>)
          linalg.fill ins(%cst_2 : f32) outs(%21 : memref<1x64xf32>)
          linalg.fill ins(%cst_3 : f32) outs(%23 : memref<1x64xf32>)
          scf.for %arg7 = %c0 to %c32 step %c1 {
            %40 = arith.muli %arg7, %c128 : index
            %41 = arith.addi %37, %40 : index
            %reinterpret_cast_5 = memref.reinterpret_cast %arg0 to offset: [%41], sizes: [1, 128, 128], strides: [524288, 4096, 1] : memref<32x128x4096xf32> to memref<1x128x128xf32, strided<[524288, 4096, 1], offset: ?>>
            loom.copy %reinterpret_cast_5, %19 src_mem_space @DRAM dst_mem_space @L1, interconnect : [], broadcast : [1, 1] : memref<1x128x128xf32, strided<[524288, 4096, 1], offset: ?>>, memref<1x128x128xf32>
            linalg.fill ins(%cst_1 : f32) outs(%17 : memref<1x64x128xf32>)
            linalg.batch_matmul ins(%32, %19 : memref<1x64x128xf32>, memref<1x128x128xf32>) outs(%17 : memref<1x64x128xf32>)
            loom.semaphore_give %19 : memref<1x128x128xf32>
            linalg.fill ins(%cst_3 : f32) outs(%25 : memref<1x64xf32>)
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%17 : memref<1x64x128xf32>) outs(%25 : memref<1x64xf32>) {
            ^bb0(%in: f32, %out: f32):
              %44 = arith.maximumf %in, %out : f32
              linalg.yield %44 : f32
            }
            linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%23, %25 : memref<1x64xf32>, memref<1x64xf32>) outs(%25 : memref<1x64xf32>) {
            ^bb0(%in: f32, %in_7: f32, %out: f32):
              %44 = arith.truncf %cst_0 : f64 to f32
              %45 = arith.mulf %in_7, %44 : f32
              %46 = arith.cmpf ogt, %in, %45 : f32
              %47 = arith.select %46, %in, %45 : f32
              linalg.yield %47 : f32
            }
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%17, %25 : memref<1x64x128xf32>, memref<1x64xf32>) outs(%17 : memref<1x64x128xf32>) {
            ^bb0(%in: f32, %in_7: f32, %out: f32):
              %44 = arith.truncf %cst_0 : f64 to f32
              %45 = arith.mulf %in, %44 : f32
              %46 = arith.subf %45, %in_7 : f32
              %47 = math.powf %cst, %46 : f32
              linalg.yield %47 : f32
            }
            linalg.fill ins(%cst_1 : f32) outs(%27 : memref<1x64xf32>)
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%17 : memref<1x64x128xf32>) outs(%27 : memref<1x64xf32>) {
            ^bb0(%in: f32, %out: f32):
              %44 = arith.addf %in, %out : f32
              linalg.yield %44 : f32
            }
            linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%23, %25 : memref<1x64xf32>, memref<1x64xf32>) outs(%29 : memref<1x64xf32>) {
            ^bb0(%in: f32, %in_7: f32, %out: f32):
              %44 = arith.subf %in, %in_7 : f32
              %45 = math.powf %cst, %44 : f32
              linalg.yield %45 : f32
            }
            linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%21, %29, %27 : memref<1x64xf32>, memref<1x64xf32>, memref<1x64xf32>) outs(%21 : memref<1x64xf32>) {
            ^bb0(%in: f32, %in_7: f32, %in_8: f32, %out: f32):
              %44 = arith.mulf %in, %in_7 : f32
              %45 = arith.addf %44, %in_8 : f32
              linalg.yield %45 : f32
            }
            loom.semaphore_give %27 : memref<1x64xf32>
            %42 = arith.muli %arg7, %c16384 : index
            %43 = arith.addi %37, %42 : index
            %reinterpret_cast_6 = memref.reinterpret_cast %arg1 to offset: [%43], sizes: [1, 128, 128], strides: [524288, 128, 1] : memref<32x4096x128xf32> to memref<1x128x128xf32, strided<[524288, 128, 1], offset: ?>>
            loom.copy %reinterpret_cast_6, %15 src_mem_space @DRAM dst_mem_space @L1, interconnect : [@horizontal_links], broadcast : [1, 8] : memref<1x128x128xf32, strided<[524288, 128, 1], offset: ?>>, memref<1x128x128xf32>
            linalg.fill ins(%cst_1 : f32) outs(%36 : memref<1x64x128xf32>)
            linalg.batch_matmul ins(%17, %15 : memref<1x64x128xf32>, memref<1x128x128xf32>) outs(%36 : memref<1x64x128xf32>)
            loom.semaphore_give %15 : memref<1x128x128xf32>
            loom.semaphore_give %17 : memref<1x64x128xf32>
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%36, %34, %29 : memref<1x64x128xf32>, memref<1x64x128xf32>, memref<1x64xf32>) outs(%34 : memref<1x64x128xf32>) {
            ^bb0(%in: f32, %in_7: f32, %in_8: f32, %out: f32):
              %44 = arith.mulf %in_7, %in_8 : f32
              %45 = arith.addf %in, %44 : f32
              linalg.yield %45 : f32
            }
            loom.semaphore_give %29 : memref<1x64xf32>
            loom.semaphore_give %36 : memref<1x64x128xf32>
            linalg.copy ins(%25 : memref<1x64xf32>) outs(%23 : memref<1x64xf32>)
            loom.semaphore_give %25 : memref<1x64xf32>
          }
          loom.semaphore_give %23 : memref<1x64xf32>
          loom.semaphore_give %32 : memref<1x64x128xf32>
          linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%34, %21 : memref<1x64x128xf32>, memref<1x64xf32>) outs(%31 : memref<1x64x128xf32>) {
          ^bb0(%in: f32, %in_5: f32, %out: f32):
            %40 = arith.divf %in, %in_5 : f32
            linalg.yield %40 : f32
          }
          loom.semaphore_give %21 : memref<1x64xf32>
          loom.semaphore_give %34 : memref<1x64x128xf32>
          %reinterpret_cast_4 = memref.reinterpret_cast %arg3 to offset: [%39], sizes: [1, 64, 128], strides: [524288, 128, 1] : memref<32x4096x128xf32> to memref<1x64x128xf32, strided<[524288, 128, 1], offset: ?>>
          loom.copy %31, %reinterpret_cast_4 src_mem_space @L1 dst_mem_space @DRAM, interconnect : [], broadcast : [1, 1] : memref<1x64x128xf32>, memref<1x64x128xf32, strided<[524288, 128, 1], offset: ?>>
          loom.semaphore_give %31 : memref<1x64x128xf32>
        }
        scf.reduce 
      } {loom.iter_types = [#loom.iter_type<spatial>, #loom.iter_type<spatial>], loom.mapped_to_dims = [@y, @x]}
      return
    }
  }
  module attributes {loom.pass_name = "Materialize"} {
    func.func @attention__d1i1_d0i1__f01__d_d_v__BB1__BM64__BN128(%arg0: memref<32x128x4096xf32>, %arg1: memref<32x4096x128xf32>, %arg2: memref<32x4096x128xf32>, %arg3: memref<32x4096x128xf32>) {
      %c16384 = arith.constant 16384 : index
      %c8192 = arith.constant 8192 : index
      %c524288 = arith.constant 524288 : index
      %c32 = arith.constant 32 : index
      %cst = arith.constant 2.000000e+00 : f32
      %cst_0 = arith.constant 0.12751743074602467 : f64
      %cst_1 = arith.constant 0.000000e+00 : f32
      %cst_2 = arith.constant 1.000000e+00 : f32
      %cst_3 = arith.constant 0xFF800000 : f32
      %c128 = arith.constant 128 : index
      %c0 = arith.constant 0 : index
      %c8 = arith.constant 8 : index
      %c1 = arith.constant 1 : index
      scf.parallel (%arg4, %arg5) = (%c0, %c0) to (%c8, %c8) step (%c1, %c1) {
        scf.for %arg6 = %c0 to %c32 step %c1 {
          %12 = arith.muli %arg4, %c8 overflow<nsw> : index
          %13 = arith.addi %12, %arg5 : index
          %14 = loom.alloc [1, 128, 128] on @L1 : memref<1x128x128xf32>
          %15 = loom.semaphore_take %14 : memref<1x128x128xf32> -> memref<1x128x128xf32>
          %16 = loom.alloc [1, 64, 128] on @L1 : memref<1x64x128xf32>
          %17 = loom.semaphore_take %16 : memref<1x64x128xf32> -> memref<1x64x128xf32>
          %18 = loom.alloc [1, 128, 128] on @L1 : memref<1x128x128xf32>
          %19 = loom.semaphore_take %18 : memref<1x128x128xf32> -> memref<1x128x128xf32>
          %20 = loom.alloc [1, 64] on @L1 : memref<1x64xf32>
          %21 = loom.semaphore_take %20 : memref<1x64xf32> -> memref<1x64xf32>
          %22 = loom.alloc [1, 64] on @L1 : memref<1x64xf32>
          %23 = loom.semaphore_take %22 : memref<1x64xf32> -> memref<1x64xf32>
          %24 = loom.alloc [1, 64] on @L1 : memref<1x64xf32>
          %25 = loom.semaphore_take %24 : memref<1x64xf32> -> memref<1x64xf32>
          %26 = loom.alloc [1, 64] on @L1 : memref<1x64xf32>
          %27 = loom.semaphore_take %26 : memref<1x64xf32> -> memref<1x64xf32>
          %28 = loom.alloc [1, 64] on @L1 : memref<1x64xf32>
          %29 = loom.semaphore_take %28 : memref<1x64xf32> -> memref<1x64xf32>
          %30 = loom.alloc [1, 64, 128] on @L1 : memref<1x64x128xf32>
          %31 = loom.semaphore_take %30 : memref<1x64x128xf32> -> memref<1x64x128xf32>
          %32 = loom.semaphore_take %30 : memref<1x64x128xf32> -> memref<1x64x128xf32>
          %33 = loom.alloc [1, 64, 128] on @L1 : memref<1x64x128xf32>
          %34 = loom.semaphore_take %33 : memref<1x64x128xf32> -> memref<1x64x128xf32>
          %35 = loom.alloc [1, 64, 128] on @L1 : memref<1x64x128xf32>
          %36 = loom.semaphore_take %35 : memref<1x64x128xf32> -> memref<1x64x128xf32>
          %37 = arith.muli %arg6, %c524288 overflow<nsw> : index
          %38 = arith.muli %13, %c8192 : index
          %39 = arith.addi %37, %38 : index
          %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%39], sizes: [1, 64, 128], strides: [524288, 128, 1] : memref<32x4096x128xf32> to memref<1x64x128xf32, strided<[524288, 128, 1], offset: ?>>
          loom.copy %reinterpret_cast, %32 src_mem_space @DRAM dst_mem_space @L1, interconnect : [], broadcast : [1, 1] : memref<1x64x128xf32, strided<[524288, 128, 1], offset: ?>>, memref<1x64x128xf32>
          linalg.fill ins(%cst_1 : f32) outs(%34 : memref<1x64x128xf32>)
          linalg.fill ins(%cst_2 : f32) outs(%21 : memref<1x64xf32>)
          linalg.fill ins(%cst_3 : f32) outs(%23 : memref<1x64xf32>)
          scf.for %arg7 = %c0 to %c32 step %c1 {
            %40 = arith.muli %arg7, %c128 : index
            %41 = arith.addi %37, %40 : index
            %reinterpret_cast_5 = memref.reinterpret_cast %arg0 to offset: [%41], sizes: [1, 128, 128], strides: [524288, 4096, 1] : memref<32x128x4096xf32> to memref<1x128x128xf32, strided<[524288, 4096, 1], offset: ?>>
            loom.copy %reinterpret_cast_5, %19 src_mem_space @DRAM dst_mem_space @L1, interconnect : [], broadcast : [1, 1] : memref<1x128x128xf32, strided<[524288, 4096, 1], offset: ?>>, memref<1x128x128xf32>
            linalg.fill ins(%cst_1 : f32) outs(%17 : memref<1x64x128xf32>)
            linalg.batch_matmul ins(%32, %19 : memref<1x64x128xf32>, memref<1x128x128xf32>) outs(%17 : memref<1x64x128xf32>)
            loom.semaphore_give %19 : memref<1x128x128xf32>
            linalg.fill ins(%cst_3 : f32) outs(%25 : memref<1x64xf32>)
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%17 : memref<1x64x128xf32>) outs(%25 : memref<1x64xf32>) {
            ^bb0(%in: f32, %out: f32):
              %44 = arith.maximumf %in, %out : f32
              linalg.yield %44 : f32
            }
            linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%23, %25 : memref<1x64xf32>, memref<1x64xf32>) outs(%25 : memref<1x64xf32>) {
            ^bb0(%in: f32, %in_7: f32, %out: f32):
              %44 = arith.truncf %cst_0 : f64 to f32
              %45 = arith.mulf %in_7, %44 : f32
              %46 = arith.cmpf ogt, %in, %45 : f32
              %47 = arith.select %46, %in, %45 : f32
              linalg.yield %47 : f32
            }
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%17, %25 : memref<1x64x128xf32>, memref<1x64xf32>) outs(%17 : memref<1x64x128xf32>) {
            ^bb0(%in: f32, %in_7: f32, %out: f32):
              %44 = arith.truncf %cst_0 : f64 to f32
              %45 = arith.mulf %in, %44 : f32
              %46 = arith.subf %45, %in_7 : f32
              %47 = math.powf %cst, %46 : f32
              linalg.yield %47 : f32
            }
            linalg.fill ins(%cst_1 : f32) outs(%27 : memref<1x64xf32>)
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%17 : memref<1x64x128xf32>) outs(%27 : memref<1x64xf32>) {
            ^bb0(%in: f32, %out: f32):
              %44 = arith.addf %in, %out : f32
              linalg.yield %44 : f32
            }
            linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%23, %25 : memref<1x64xf32>, memref<1x64xf32>) outs(%29 : memref<1x64xf32>) {
            ^bb0(%in: f32, %in_7: f32, %out: f32):
              %44 = arith.subf %in, %in_7 : f32
              %45 = math.powf %cst, %44 : f32
              linalg.yield %45 : f32
            }
            linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%21, %29, %27 : memref<1x64xf32>, memref<1x64xf32>, memref<1x64xf32>) outs(%21 : memref<1x64xf32>) {
            ^bb0(%in: f32, %in_7: f32, %in_8: f32, %out: f32):
              %44 = arith.mulf %in, %in_7 : f32
              %45 = arith.addf %44, %in_8 : f32
              linalg.yield %45 : f32
            }
            loom.semaphore_give %27 : memref<1x64xf32>
            %42 = arith.muli %arg7, %c16384 : index
            %43 = arith.addi %37, %42 : index
            %reinterpret_cast_6 = memref.reinterpret_cast %arg1 to offset: [%43], sizes: [1, 128, 128], strides: [524288, 128, 1] : memref<32x4096x128xf32> to memref<1x128x128xf32, strided<[524288, 128, 1], offset: ?>>
            loom.copy %reinterpret_cast_6, %15 src_mem_space @DRAM dst_mem_space @L1, interconnect : [@vertical_links], broadcast : [8, 1] : memref<1x128x128xf32, strided<[524288, 128, 1], offset: ?>>, memref<1x128x128xf32>
            linalg.fill ins(%cst_1 : f32) outs(%36 : memref<1x64x128xf32>)
            linalg.batch_matmul ins(%17, %15 : memref<1x64x128xf32>, memref<1x128x128xf32>) outs(%36 : memref<1x64x128xf32>)
            loom.semaphore_give %15 : memref<1x128x128xf32>
            loom.semaphore_give %17 : memref<1x64x128xf32>
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%36, %34, %29 : memref<1x64x128xf32>, memref<1x64x128xf32>, memref<1x64xf32>) outs(%34 : memref<1x64x128xf32>) {
            ^bb0(%in: f32, %in_7: f32, %in_8: f32, %out: f32):
              %44 = arith.mulf %in_7, %in_8 : f32
              %45 = arith.addf %in, %44 : f32
              linalg.yield %45 : f32
            }
            loom.semaphore_give %29 : memref<1x64xf32>
            loom.semaphore_give %36 : memref<1x64x128xf32>
            linalg.copy ins(%25 : memref<1x64xf32>) outs(%23 : memref<1x64xf32>)
            loom.semaphore_give %25 : memref<1x64xf32>
          }
          loom.semaphore_give %23 : memref<1x64xf32>
          loom.semaphore_give %32 : memref<1x64x128xf32>
          linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%34, %21 : memref<1x64x128xf32>, memref<1x64xf32>) outs(%31 : memref<1x64x128xf32>) {
          ^bb0(%in: f32, %in_5: f32, %out: f32):
            %40 = arith.divf %in, %in_5 : f32
            linalg.yield %40 : f32
          }
          loom.semaphore_give %21 : memref<1x64xf32>
          loom.semaphore_give %34 : memref<1x64x128xf32>
          %reinterpret_cast_4 = memref.reinterpret_cast %arg3 to offset: [%39], sizes: [1, 64, 128], strides: [524288, 128, 1] : memref<32x4096x128xf32> to memref<1x64x128xf32, strided<[524288, 128, 1], offset: ?>>
          loom.copy %31, %reinterpret_cast_4 src_mem_space @L1 dst_mem_space @DRAM, interconnect : [], broadcast : [1, 1] : memref<1x64x128xf32>, memref<1x64x128xf32, strided<[524288, 128, 1], offset: ?>>
          loom.semaphore_give %31 : memref<1x64x128xf32>
        }
        scf.reduce 
      } {loom.iter_types = [#loom.iter_type<spatial>, #loom.iter_type<spatial>], loom.mapped_to_dims = [@y, @x]}
      return
    }
  }
  module attributes {loom.pass_name = "Materialize"} {
    func.func @attention__d1i1_d0i1__f01__d_a_d__BB1__BM64__BN128(%arg0: memref<32x128x4096xf32>, %arg1: memref<32x4096x128xf32>, %arg2: memref<32x4096x128xf32>, %arg3: memref<32x4096x128xf32>) {
      %c16384 = arith.constant 16384 : index
      %c8192 = arith.constant 8192 : index
      %c524288 = arith.constant 524288 : index
      %c32 = arith.constant 32 : index
      %cst = arith.constant 2.000000e+00 : f32
      %cst_0 = arith.constant 0.12751743074602467 : f64
      %cst_1 = arith.constant 0.000000e+00 : f32
      %cst_2 = arith.constant 1.000000e+00 : f32
      %cst_3 = arith.constant 0xFF800000 : f32
      %c128 = arith.constant 128 : index
      %c0 = arith.constant 0 : index
      %c8 = arith.constant 8 : index
      %c1 = arith.constant 1 : index
      scf.parallel (%arg4, %arg5) = (%c0, %c0) to (%c8, %c8) step (%c1, %c1) {
        scf.for %arg6 = %c0 to %c32 step %c1 {
          %12 = arith.muli %arg4, %c8 overflow<nsw> : index
          %13 = arith.addi %12, %arg5 : index
          %14 = loom.alloc [1, 128, 128] on @L1 : memref<1x128x128xf32>
          %15 = loom.semaphore_take %14 : memref<1x128x128xf32> -> memref<1x128x128xf32>
          %16 = loom.alloc [1, 64, 128] on @L1 : memref<1x64x128xf32>
          %17 = loom.semaphore_take %16 : memref<1x64x128xf32> -> memref<1x64x128xf32>
          %18 = loom.alloc [1, 128, 128] on @L1 : memref<1x128x128xf32>
          %19 = loom.semaphore_take %18 : memref<1x128x128xf32> -> memref<1x128x128xf32>
          %20 = loom.alloc [1, 64] on @L1 : memref<1x64xf32>
          %21 = loom.semaphore_take %20 : memref<1x64xf32> -> memref<1x64xf32>
          %22 = loom.alloc [1, 64] on @L1 : memref<1x64xf32>
          %23 = loom.semaphore_take %22 : memref<1x64xf32> -> memref<1x64xf32>
          %24 = loom.alloc [1, 64] on @L1 : memref<1x64xf32>
          %25 = loom.semaphore_take %24 : memref<1x64xf32> -> memref<1x64xf32>
          %26 = loom.alloc [1, 64] on @L1 : memref<1x64xf32>
          %27 = loom.semaphore_take %26 : memref<1x64xf32> -> memref<1x64xf32>
          %28 = loom.alloc [1, 64] on @L1 : memref<1x64xf32>
          %29 = loom.semaphore_take %28 : memref<1x64xf32> -> memref<1x64xf32>
          %30 = loom.alloc [1, 64, 128] on @L1 : memref<1x64x128xf32>
          %31 = loom.semaphore_take %30 : memref<1x64x128xf32> -> memref<1x64x128xf32>
          %32 = loom.semaphore_take %30 : memref<1x64x128xf32> -> memref<1x64x128xf32>
          %33 = loom.alloc [1, 64, 128] on @L1 : memref<1x64x128xf32>
          %34 = loom.semaphore_take %33 : memref<1x64x128xf32> -> memref<1x64x128xf32>
          %35 = loom.alloc [1, 64, 128] on @L1 : memref<1x64x128xf32>
          %36 = loom.semaphore_take %35 : memref<1x64x128xf32> -> memref<1x64x128xf32>
          %37 = arith.muli %arg6, %c524288 overflow<nsw> : index
          %38 = arith.muli %13, %c8192 : index
          %39 = arith.addi %37, %38 : index
          %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%39], sizes: [1, 64, 128], strides: [524288, 128, 1] : memref<32x4096x128xf32> to memref<1x64x128xf32, strided<[524288, 128, 1], offset: ?>>
          loom.copy %reinterpret_cast, %32 src_mem_space @DRAM dst_mem_space @L1, interconnect : [], broadcast : [1, 1] : memref<1x64x128xf32, strided<[524288, 128, 1], offset: ?>>, memref<1x64x128xf32>
          linalg.fill ins(%cst_1 : f32) outs(%34 : memref<1x64x128xf32>)
          linalg.fill ins(%cst_2 : f32) outs(%21 : memref<1x64xf32>)
          linalg.fill ins(%cst_3 : f32) outs(%23 : memref<1x64xf32>)
          scf.for %arg7 = %c0 to %c32 step %c1 {
            %40 = arith.muli %arg7, %c128 : index
            %41 = arith.addi %37, %40 : index
            %reinterpret_cast_5 = memref.reinterpret_cast %arg0 to offset: [%41], sizes: [1, 128, 128], strides: [524288, 4096, 1] : memref<32x128x4096xf32> to memref<1x128x128xf32, strided<[524288, 4096, 1], offset: ?>>
            loom.copy %reinterpret_cast_5, %19 src_mem_space @DRAM dst_mem_space @L1, interconnect : [@horizontal_links, @vertical_links], broadcast : [8, 8] : memref<1x128x128xf32, strided<[524288, 4096, 1], offset: ?>>, memref<1x128x128xf32>
            linalg.fill ins(%cst_1 : f32) outs(%17 : memref<1x64x128xf32>)
            linalg.batch_matmul ins(%32, %19 : memref<1x64x128xf32>, memref<1x128x128xf32>) outs(%17 : memref<1x64x128xf32>)
            loom.semaphore_give %19 : memref<1x128x128xf32>
            linalg.fill ins(%cst_3 : f32) outs(%25 : memref<1x64xf32>)
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%17 : memref<1x64x128xf32>) outs(%25 : memref<1x64xf32>) {
            ^bb0(%in: f32, %out: f32):
              %44 = arith.maximumf %in, %out : f32
              linalg.yield %44 : f32
            }
            linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%23, %25 : memref<1x64xf32>, memref<1x64xf32>) outs(%25 : memref<1x64xf32>) {
            ^bb0(%in: f32, %in_7: f32, %out: f32):
              %44 = arith.truncf %cst_0 : f64 to f32
              %45 = arith.mulf %in_7, %44 : f32
              %46 = arith.cmpf ogt, %in, %45 : f32
              %47 = arith.select %46, %in, %45 : f32
              linalg.yield %47 : f32
            }
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%17, %25 : memref<1x64x128xf32>, memref<1x64xf32>) outs(%17 : memref<1x64x128xf32>) {
            ^bb0(%in: f32, %in_7: f32, %out: f32):
              %44 = arith.truncf %cst_0 : f64 to f32
              %45 = arith.mulf %in, %44 : f32
              %46 = arith.subf %45, %in_7 : f32
              %47 = math.powf %cst, %46 : f32
              linalg.yield %47 : f32
            }
            linalg.fill ins(%cst_1 : f32) outs(%27 : memref<1x64xf32>)
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%17 : memref<1x64x128xf32>) outs(%27 : memref<1x64xf32>) {
            ^bb0(%in: f32, %out: f32):
              %44 = arith.addf %in, %out : f32
              linalg.yield %44 : f32
            }
            linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%23, %25 : memref<1x64xf32>, memref<1x64xf32>) outs(%29 : memref<1x64xf32>) {
            ^bb0(%in: f32, %in_7: f32, %out: f32):
              %44 = arith.subf %in, %in_7 : f32
              %45 = math.powf %cst, %44 : f32
              linalg.yield %45 : f32
            }
            linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%21, %29, %27 : memref<1x64xf32>, memref<1x64xf32>, memref<1x64xf32>) outs(%21 : memref<1x64xf32>) {
            ^bb0(%in: f32, %in_7: f32, %in_8: f32, %out: f32):
              %44 = arith.mulf %in, %in_7 : f32
              %45 = arith.addf %44, %in_8 : f32
              linalg.yield %45 : f32
            }
            loom.semaphore_give %27 : memref<1x64xf32>
            %42 = arith.muli %arg7, %c16384 : index
            %43 = arith.addi %37, %42 : index
            %reinterpret_cast_6 = memref.reinterpret_cast %arg1 to offset: [%43], sizes: [1, 128, 128], strides: [524288, 128, 1] : memref<32x4096x128xf32> to memref<1x128x128xf32, strided<[524288, 128, 1], offset: ?>>
            loom.copy %reinterpret_cast_6, %15 src_mem_space @DRAM dst_mem_space @L1, interconnect : [], broadcast : [1, 1] : memref<1x128x128xf32, strided<[524288, 128, 1], offset: ?>>, memref<1x128x128xf32>
            linalg.fill ins(%cst_1 : f32) outs(%36 : memref<1x64x128xf32>)
            linalg.batch_matmul ins(%17, %15 : memref<1x64x128xf32>, memref<1x128x128xf32>) outs(%36 : memref<1x64x128xf32>)
            loom.semaphore_give %15 : memref<1x128x128xf32>
            loom.semaphore_give %17 : memref<1x64x128xf32>
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%36, %34, %29 : memref<1x64x128xf32>, memref<1x64x128xf32>, memref<1x64xf32>) outs(%34 : memref<1x64x128xf32>) {
            ^bb0(%in: f32, %in_7: f32, %in_8: f32, %out: f32):
              %44 = arith.mulf %in_7, %in_8 : f32
              %45 = arith.addf %in, %44 : f32
              linalg.yield %45 : f32
            }
            loom.semaphore_give %29 : memref<1x64xf32>
            loom.semaphore_give %36 : memref<1x64x128xf32>
            linalg.copy ins(%25 : memref<1x64xf32>) outs(%23 : memref<1x64xf32>)
            loom.semaphore_give %25 : memref<1x64xf32>
          }
          loom.semaphore_give %23 : memref<1x64xf32>
          loom.semaphore_give %32 : memref<1x64x128xf32>
          linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%34, %21 : memref<1x64x128xf32>, memref<1x64xf32>) outs(%31 : memref<1x64x128xf32>) {
          ^bb0(%in: f32, %in_5: f32, %out: f32):
            %40 = arith.divf %in, %in_5 : f32
            linalg.yield %40 : f32
          }
          loom.semaphore_give %21 : memref<1x64xf32>
          loom.semaphore_give %34 : memref<1x64x128xf32>
          %reinterpret_cast_4 = memref.reinterpret_cast %arg3 to offset: [%39], sizes: [1, 64, 128], strides: [524288, 128, 1] : memref<32x4096x128xf32> to memref<1x64x128xf32, strided<[524288, 128, 1], offset: ?>>
          loom.copy %31, %reinterpret_cast_4 src_mem_space @L1 dst_mem_space @DRAM, interconnect : [], broadcast : [1, 1] : memref<1x64x128xf32>, memref<1x64x128xf32, strided<[524288, 128, 1], offset: ?>>
          loom.semaphore_give %31 : memref<1x64x128xf32>
        }
        scf.reduce 
      } {loom.iter_types = [#loom.iter_type<spatial>, #loom.iter_type<spatial>], loom.mapped_to_dims = [@y, @x]}
      return
    }
  }
  module attributes {loom.pass_name = "Materialize"} {
    func.func @attention__d1i1_d0i1__f01__d_a_a__BB1__BM64__BN128(%arg0: memref<32x128x4096xf32>, %arg1: memref<32x4096x128xf32>, %arg2: memref<32x4096x128xf32>, %arg3: memref<32x4096x128xf32>) {
      %c16384 = arith.constant 16384 : index
      %c8192 = arith.constant 8192 : index
      %c524288 = arith.constant 524288 : index
      %c32 = arith.constant 32 : index
      %cst = arith.constant 2.000000e+00 : f32
      %cst_0 = arith.constant 0.12751743074602467 : f64
      %cst_1 = arith.constant 0.000000e+00 : f32
      %cst_2 = arith.constant 1.000000e+00 : f32
      %cst_3 = arith.constant 0xFF800000 : f32
      %c128 = arith.constant 128 : index
      %c0 = arith.constant 0 : index
      %c8 = arith.constant 8 : index
      %c1 = arith.constant 1 : index
      scf.parallel (%arg4, %arg5) = (%c0, %c0) to (%c8, %c8) step (%c1, %c1) {
        scf.for %arg6 = %c0 to %c32 step %c1 {
          %12 = arith.muli %arg4, %c8 overflow<nsw> : index
          %13 = arith.addi %12, %arg5 : index
          %14 = loom.alloc [1, 128, 128] on @L1 : memref<1x128x128xf32>
          %15 = loom.semaphore_take %14 : memref<1x128x128xf32> -> memref<1x128x128xf32>
          %16 = loom.alloc [1, 64, 128] on @L1 : memref<1x64x128xf32>
          %17 = loom.semaphore_take %16 : memref<1x64x128xf32> -> memref<1x64x128xf32>
          %18 = loom.alloc [1, 128, 128] on @L1 : memref<1x128x128xf32>
          %19 = loom.semaphore_take %18 : memref<1x128x128xf32> -> memref<1x128x128xf32>
          %20 = loom.alloc [1, 64] on @L1 : memref<1x64xf32>
          %21 = loom.semaphore_take %20 : memref<1x64xf32> -> memref<1x64xf32>
          %22 = loom.alloc [1, 64] on @L1 : memref<1x64xf32>
          %23 = loom.semaphore_take %22 : memref<1x64xf32> -> memref<1x64xf32>
          %24 = loom.alloc [1, 64] on @L1 : memref<1x64xf32>
          %25 = loom.semaphore_take %24 : memref<1x64xf32> -> memref<1x64xf32>
          %26 = loom.alloc [1, 64] on @L1 : memref<1x64xf32>
          %27 = loom.semaphore_take %26 : memref<1x64xf32> -> memref<1x64xf32>
          %28 = loom.alloc [1, 64] on @L1 : memref<1x64xf32>
          %29 = loom.semaphore_take %28 : memref<1x64xf32> -> memref<1x64xf32>
          %30 = loom.alloc [1, 64, 128] on @L1 : memref<1x64x128xf32>
          %31 = loom.semaphore_take %30 : memref<1x64x128xf32> -> memref<1x64x128xf32>
          %32 = loom.semaphore_take %30 : memref<1x64x128xf32> -> memref<1x64x128xf32>
          %33 = loom.alloc [1, 64, 128] on @L1 : memref<1x64x128xf32>
          %34 = loom.semaphore_take %33 : memref<1x64x128xf32> -> memref<1x64x128xf32>
          %35 = loom.alloc [1, 64, 128] on @L1 : memref<1x64x128xf32>
          %36 = loom.semaphore_take %35 : memref<1x64x128xf32> -> memref<1x64x128xf32>
          %37 = arith.muli %arg6, %c524288 overflow<nsw> : index
          %38 = arith.muli %13, %c8192 : index
          %39 = arith.addi %37, %38 : index
          %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%39], sizes: [1, 64, 128], strides: [524288, 128, 1] : memref<32x4096x128xf32> to memref<1x64x128xf32, strided<[524288, 128, 1], offset: ?>>
          loom.copy %reinterpret_cast, %32 src_mem_space @DRAM dst_mem_space @L1, interconnect : [], broadcast : [1, 1] : memref<1x64x128xf32, strided<[524288, 128, 1], offset: ?>>, memref<1x64x128xf32>
          linalg.fill ins(%cst_1 : f32) outs(%34 : memref<1x64x128xf32>)
          linalg.fill ins(%cst_2 : f32) outs(%21 : memref<1x64xf32>)
          linalg.fill ins(%cst_3 : f32) outs(%23 : memref<1x64xf32>)
          scf.for %arg7 = %c0 to %c32 step %c1 {
            %40 = arith.muli %arg7, %c128 : index
            %41 = arith.addi %37, %40 : index
            %reinterpret_cast_5 = memref.reinterpret_cast %arg0 to offset: [%41], sizes: [1, 128, 128], strides: [524288, 4096, 1] : memref<32x128x4096xf32> to memref<1x128x128xf32, strided<[524288, 4096, 1], offset: ?>>
            loom.copy %reinterpret_cast_5, %19 src_mem_space @DRAM dst_mem_space @L1, interconnect : [@horizontal_links, @vertical_links], broadcast : [8, 8] : memref<1x128x128xf32, strided<[524288, 4096, 1], offset: ?>>, memref<1x128x128xf32>
            linalg.fill ins(%cst_1 : f32) outs(%17 : memref<1x64x128xf32>)
            linalg.batch_matmul ins(%32, %19 : memref<1x64x128xf32>, memref<1x128x128xf32>) outs(%17 : memref<1x64x128xf32>)
            loom.semaphore_give %19 : memref<1x128x128xf32>
            linalg.fill ins(%cst_3 : f32) outs(%25 : memref<1x64xf32>)
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%17 : memref<1x64x128xf32>) outs(%25 : memref<1x64xf32>) {
            ^bb0(%in: f32, %out: f32):
              %44 = arith.maximumf %in, %out : f32
              linalg.yield %44 : f32
            }
            linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%23, %25 : memref<1x64xf32>, memref<1x64xf32>) outs(%25 : memref<1x64xf32>) {
            ^bb0(%in: f32, %in_7: f32, %out: f32):
              %44 = arith.truncf %cst_0 : f64 to f32
              %45 = arith.mulf %in_7, %44 : f32
              %46 = arith.cmpf ogt, %in, %45 : f32
              %47 = arith.select %46, %in, %45 : f32
              linalg.yield %47 : f32
            }
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%17, %25 : memref<1x64x128xf32>, memref<1x64xf32>) outs(%17 : memref<1x64x128xf32>) {
            ^bb0(%in: f32, %in_7: f32, %out: f32):
              %44 = arith.truncf %cst_0 : f64 to f32
              %45 = arith.mulf %in, %44 : f32
              %46 = arith.subf %45, %in_7 : f32
              %47 = math.powf %cst, %46 : f32
              linalg.yield %47 : f32
            }
            linalg.fill ins(%cst_1 : f32) outs(%27 : memref<1x64xf32>)
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%17 : memref<1x64x128xf32>) outs(%27 : memref<1x64xf32>) {
            ^bb0(%in: f32, %out: f32):
              %44 = arith.addf %in, %out : f32
              linalg.yield %44 : f32
            }
            linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%23, %25 : memref<1x64xf32>, memref<1x64xf32>) outs(%29 : memref<1x64xf32>) {
            ^bb0(%in: f32, %in_7: f32, %out: f32):
              %44 = arith.subf %in, %in_7 : f32
              %45 = math.powf %cst, %44 : f32
              linalg.yield %45 : f32
            }
            linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%21, %29, %27 : memref<1x64xf32>, memref<1x64xf32>, memref<1x64xf32>) outs(%21 : memref<1x64xf32>) {
            ^bb0(%in: f32, %in_7: f32, %in_8: f32, %out: f32):
              %44 = arith.mulf %in, %in_7 : f32
              %45 = arith.addf %44, %in_8 : f32
              linalg.yield %45 : f32
            }
            loom.semaphore_give %27 : memref<1x64xf32>
            %42 = arith.muli %arg7, %c16384 : index
            %43 = arith.addi %37, %42 : index
            %reinterpret_cast_6 = memref.reinterpret_cast %arg1 to offset: [%43], sizes: [1, 128, 128], strides: [524288, 128, 1] : memref<32x4096x128xf32> to memref<1x128x128xf32, strided<[524288, 128, 1], offset: ?>>
            loom.copy %reinterpret_cast_6, %15 src_mem_space @DRAM dst_mem_space @L1, interconnect : [@horizontal_links, @vertical_links], broadcast : [8, 8] : memref<1x128x128xf32, strided<[524288, 128, 1], offset: ?>>, memref<1x128x128xf32>
            linalg.fill ins(%cst_1 : f32) outs(%36 : memref<1x64x128xf32>)
            linalg.batch_matmul ins(%17, %15 : memref<1x64x128xf32>, memref<1x128x128xf32>) outs(%36 : memref<1x64x128xf32>)
            loom.semaphore_give %15 : memref<1x128x128xf32>
            loom.semaphore_give %17 : memref<1x64x128xf32>
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%36, %34, %29 : memref<1x64x128xf32>, memref<1x64x128xf32>, memref<1x64xf32>) outs(%34 : memref<1x64x128xf32>) {
            ^bb0(%in: f32, %in_7: f32, %in_8: f32, %out: f32):
              %44 = arith.mulf %in_7, %in_8 : f32
              %45 = arith.addf %in, %44 : f32
              linalg.yield %45 : f32
            }
            loom.semaphore_give %29 : memref<1x64xf32>
            loom.semaphore_give %36 : memref<1x64x128xf32>
            linalg.copy ins(%25 : memref<1x64xf32>) outs(%23 : memref<1x64xf32>)
            loom.semaphore_give %25 : memref<1x64xf32>
          }
          loom.semaphore_give %23 : memref<1x64xf32>
          loom.semaphore_give %32 : memref<1x64x128xf32>
          linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%34, %21 : memref<1x64x128xf32>, memref<1x64xf32>) outs(%31 : memref<1x64x128xf32>) {
          ^bb0(%in: f32, %in_5: f32, %out: f32):
            %40 = arith.divf %in, %in_5 : f32
            linalg.yield %40 : f32
          }
          loom.semaphore_give %21 : memref<1x64xf32>
          loom.semaphore_give %34 : memref<1x64x128xf32>
          %reinterpret_cast_4 = memref.reinterpret_cast %arg3 to offset: [%39], sizes: [1, 64, 128], strides: [524288, 128, 1] : memref<32x4096x128xf32> to memref<1x64x128xf32, strided<[524288, 128, 1], offset: ?>>
          loom.copy %31, %reinterpret_cast_4 src_mem_space @L1 dst_mem_space @DRAM, interconnect : [], broadcast : [1, 1] : memref<1x64x128xf32>, memref<1x64x128xf32, strided<[524288, 128, 1], offset: ?>>
          loom.semaphore_give %31 : memref<1x64x128xf32>
        }
        scf.reduce 
      } {loom.iter_types = [#loom.iter_type<spatial>, #loom.iter_type<spatial>], loom.mapped_to_dims = [@y, @x]}
      return
    }
  }
  module attributes {loom.pass_name = "Materialize"} {
    func.func @attention__d1i1_d0i1__f01__d_a_h__BB1__BM64__BN128(%arg0: memref<32x128x4096xf32>, %arg1: memref<32x4096x128xf32>, %arg2: memref<32x4096x128xf32>, %arg3: memref<32x4096x128xf32>) {
      %c16384 = arith.constant 16384 : index
      %c8192 = arith.constant 8192 : index
      %c524288 = arith.constant 524288 : index
      %c32 = arith.constant 32 : index
      %cst = arith.constant 2.000000e+00 : f32
      %cst_0 = arith.constant 0.12751743074602467 : f64
      %cst_1 = arith.constant 0.000000e+00 : f32
      %cst_2 = arith.constant 1.000000e+00 : f32
      %cst_3 = arith.constant 0xFF800000 : f32
      %c128 = arith.constant 128 : index
      %c0 = arith.constant 0 : index
      %c8 = arith.constant 8 : index
      %c1 = arith.constant 1 : index
      scf.parallel (%arg4, %arg5) = (%c0, %c0) to (%c8, %c8) step (%c1, %c1) {
        scf.for %arg6 = %c0 to %c32 step %c1 {
          %12 = arith.muli %arg4, %c8 overflow<nsw> : index
          %13 = arith.addi %12, %arg5 : index
          %14 = loom.alloc [1, 128, 128] on @L1 : memref<1x128x128xf32>
          %15 = loom.semaphore_take %14 : memref<1x128x128xf32> -> memref<1x128x128xf32>
          %16 = loom.alloc [1, 64, 128] on @L1 : memref<1x64x128xf32>
          %17 = loom.semaphore_take %16 : memref<1x64x128xf32> -> memref<1x64x128xf32>
          %18 = loom.alloc [1, 128, 128] on @L1 : memref<1x128x128xf32>
          %19 = loom.semaphore_take %18 : memref<1x128x128xf32> -> memref<1x128x128xf32>
          %20 = loom.alloc [1, 64] on @L1 : memref<1x64xf32>
          %21 = loom.semaphore_take %20 : memref<1x64xf32> -> memref<1x64xf32>
          %22 = loom.alloc [1, 64] on @L1 : memref<1x64xf32>
          %23 = loom.semaphore_take %22 : memref<1x64xf32> -> memref<1x64xf32>
          %24 = loom.alloc [1, 64] on @L1 : memref<1x64xf32>
          %25 = loom.semaphore_take %24 : memref<1x64xf32> -> memref<1x64xf32>
          %26 = loom.alloc [1, 64] on @L1 : memref<1x64xf32>
          %27 = loom.semaphore_take %26 : memref<1x64xf32> -> memref<1x64xf32>
          %28 = loom.alloc [1, 64] on @L1 : memref<1x64xf32>
          %29 = loom.semaphore_take %28 : memref<1x64xf32> -> memref<1x64xf32>
          %30 = loom.alloc [1, 64, 128] on @L1 : memref<1x64x128xf32>
          %31 = loom.semaphore_take %30 : memref<1x64x128xf32> -> memref<1x64x128xf32>
          %32 = loom.semaphore_take %30 : memref<1x64x128xf32> -> memref<1x64x128xf32>
          %33 = loom.alloc [1, 64, 128] on @L1 : memref<1x64x128xf32>
          %34 = loom.semaphore_take %33 : memref<1x64x128xf32> -> memref<1x64x128xf32>
          %35 = loom.alloc [1, 64, 128] on @L1 : memref<1x64x128xf32>
          %36 = loom.semaphore_take %35 : memref<1x64x128xf32> -> memref<1x64x128xf32>
          %37 = arith.muli %arg6, %c524288 overflow<nsw> : index
          %38 = arith.muli %13, %c8192 : index
          %39 = arith.addi %37, %38 : index
          %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%39], sizes: [1, 64, 128], strides: [524288, 128, 1] : memref<32x4096x128xf32> to memref<1x64x128xf32, strided<[524288, 128, 1], offset: ?>>
          loom.copy %reinterpret_cast, %32 src_mem_space @DRAM dst_mem_space @L1, interconnect : [], broadcast : [1, 1] : memref<1x64x128xf32, strided<[524288, 128, 1], offset: ?>>, memref<1x64x128xf32>
          linalg.fill ins(%cst_1 : f32) outs(%34 : memref<1x64x128xf32>)
          linalg.fill ins(%cst_2 : f32) outs(%21 : memref<1x64xf32>)
          linalg.fill ins(%cst_3 : f32) outs(%23 : memref<1x64xf32>)
          scf.for %arg7 = %c0 to %c32 step %c1 {
            %40 = arith.muli %arg7, %c128 : index
            %41 = arith.addi %37, %40 : index
            %reinterpret_cast_5 = memref.reinterpret_cast %arg0 to offset: [%41], sizes: [1, 128, 128], strides: [524288, 4096, 1] : memref<32x128x4096xf32> to memref<1x128x128xf32, strided<[524288, 4096, 1], offset: ?>>
            loom.copy %reinterpret_cast_5, %19 src_mem_space @DRAM dst_mem_space @L1, interconnect : [@horizontal_links, @vertical_links], broadcast : [8, 8] : memref<1x128x128xf32, strided<[524288, 4096, 1], offset: ?>>, memref<1x128x128xf32>
            linalg.fill ins(%cst_1 : f32) outs(%17 : memref<1x64x128xf32>)
            linalg.batch_matmul ins(%32, %19 : memref<1x64x128xf32>, memref<1x128x128xf32>) outs(%17 : memref<1x64x128xf32>)
            loom.semaphore_give %19 : memref<1x128x128xf32>
            linalg.fill ins(%cst_3 : f32) outs(%25 : memref<1x64xf32>)
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%17 : memref<1x64x128xf32>) outs(%25 : memref<1x64xf32>) {
            ^bb0(%in: f32, %out: f32):
              %44 = arith.maximumf %in, %out : f32
              linalg.yield %44 : f32
            }
            linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%23, %25 : memref<1x64xf32>, memref<1x64xf32>) outs(%25 : memref<1x64xf32>) {
            ^bb0(%in: f32, %in_7: f32, %out: f32):
              %44 = arith.truncf %cst_0 : f64 to f32
              %45 = arith.mulf %in_7, %44 : f32
              %46 = arith.cmpf ogt, %in, %45 : f32
              %47 = arith.select %46, %in, %45 : f32
              linalg.yield %47 : f32
            }
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%17, %25 : memref<1x64x128xf32>, memref<1x64xf32>) outs(%17 : memref<1x64x128xf32>) {
            ^bb0(%in: f32, %in_7: f32, %out: f32):
              %44 = arith.truncf %cst_0 : f64 to f32
              %45 = arith.mulf %in, %44 : f32
              %46 = arith.subf %45, %in_7 : f32
              %47 = math.powf %cst, %46 : f32
              linalg.yield %47 : f32
            }
            linalg.fill ins(%cst_1 : f32) outs(%27 : memref<1x64xf32>)
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%17 : memref<1x64x128xf32>) outs(%27 : memref<1x64xf32>) {
            ^bb0(%in: f32, %out: f32):
              %44 = arith.addf %in, %out : f32
              linalg.yield %44 : f32
            }
            linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%23, %25 : memref<1x64xf32>, memref<1x64xf32>) outs(%29 : memref<1x64xf32>) {
            ^bb0(%in: f32, %in_7: f32, %out: f32):
              %44 = arith.subf %in, %in_7 : f32
              %45 = math.powf %cst, %44 : f32
              linalg.yield %45 : f32
            }
            linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%21, %29, %27 : memref<1x64xf32>, memref<1x64xf32>, memref<1x64xf32>) outs(%21 : memref<1x64xf32>) {
            ^bb0(%in: f32, %in_7: f32, %in_8: f32, %out: f32):
              %44 = arith.mulf %in, %in_7 : f32
              %45 = arith.addf %44, %in_8 : f32
              linalg.yield %45 : f32
            }
            loom.semaphore_give %27 : memref<1x64xf32>
            %42 = arith.muli %arg7, %c16384 : index
            %43 = arith.addi %37, %42 : index
            %reinterpret_cast_6 = memref.reinterpret_cast %arg1 to offset: [%43], sizes: [1, 128, 128], strides: [524288, 128, 1] : memref<32x4096x128xf32> to memref<1x128x128xf32, strided<[524288, 128, 1], offset: ?>>
            loom.copy %reinterpret_cast_6, %15 src_mem_space @DRAM dst_mem_space @L1, interconnect : [@horizontal_links], broadcast : [1, 8] : memref<1x128x128xf32, strided<[524288, 128, 1], offset: ?>>, memref<1x128x128xf32>
            linalg.fill ins(%cst_1 : f32) outs(%36 : memref<1x64x128xf32>)
            linalg.batch_matmul ins(%17, %15 : memref<1x64x128xf32>, memref<1x128x128xf32>) outs(%36 : memref<1x64x128xf32>)
            loom.semaphore_give %15 : memref<1x128x128xf32>
            loom.semaphore_give %17 : memref<1x64x128xf32>
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%36, %34, %29 : memref<1x64x128xf32>, memref<1x64x128xf32>, memref<1x64xf32>) outs(%34 : memref<1x64x128xf32>) {
            ^bb0(%in: f32, %in_7: f32, %in_8: f32, %out: f32):
              %44 = arith.mulf %in_7, %in_8 : f32
              %45 = arith.addf %in, %44 : f32
              linalg.yield %45 : f32
            }
            loom.semaphore_give %29 : memref<1x64xf32>
            loom.semaphore_give %36 : memref<1x64x128xf32>
            linalg.copy ins(%25 : memref<1x64xf32>) outs(%23 : memref<1x64xf32>)
            loom.semaphore_give %25 : memref<1x64xf32>
          }
          loom.semaphore_give %23 : memref<1x64xf32>
          loom.semaphore_give %32 : memref<1x64x128xf32>
          linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%34, %21 : memref<1x64x128xf32>, memref<1x64xf32>) outs(%31 : memref<1x64x128xf32>) {
          ^bb0(%in: f32, %in_5: f32, %out: f32):
            %40 = arith.divf %in, %in_5 : f32
            linalg.yield %40 : f32
          }
          loom.semaphore_give %21 : memref<1x64xf32>
          loom.semaphore_give %34 : memref<1x64x128xf32>
          %reinterpret_cast_4 = memref.reinterpret_cast %arg3 to offset: [%39], sizes: [1, 64, 128], strides: [524288, 128, 1] : memref<32x4096x128xf32> to memref<1x64x128xf32, strided<[524288, 128, 1], offset: ?>>
          loom.copy %31, %reinterpret_cast_4 src_mem_space @L1 dst_mem_space @DRAM, interconnect : [], broadcast : [1, 1] : memref<1x64x128xf32>, memref<1x64x128xf32, strided<[524288, 128, 1], offset: ?>>
          loom.semaphore_give %31 : memref<1x64x128xf32>
        }
        scf.reduce 
      } {loom.iter_types = [#loom.iter_type<spatial>, #loom.iter_type<spatial>], loom.mapped_to_dims = [@y, @x]}
      return
    }
  }
  module attributes {loom.pass_name = "Materialize"} {
    func.func @attention__d1i1_d0i1__f01__d_a_v__BB1__BM64__BN128(%arg0: memref<32x128x4096xf32>, %arg1: memref<32x4096x128xf32>, %arg2: memref<32x4096x128xf32>, %arg3: memref<32x4096x128xf32>) {
      %c16384 = arith.constant 16384 : index
      %c8192 = arith.constant 8192 : index
      %c524288 = arith.constant 524288 : index
      %c32 = arith.constant 32 : index
      %cst = arith.constant 2.000000e+00 : f32
      %cst_0 = arith.constant 0.12751743074602467 : f64
      %cst_1 = arith.constant 0.000000e+00 : f32
      %cst_2 = arith.constant 1.000000e+00 : f32
      %cst_3 = arith.constant 0xFF800000 : f32
      %c128 = arith.constant 128 : index
      %c0 = arith.constant 0 : index
      %c8 = arith.constant 8 : index
      %c1 = arith.constant 1 : index
      scf.parallel (%arg4, %arg5) = (%c0, %c0) to (%c8, %c8) step (%c1, %c1) {
        scf.for %arg6 = %c0 to %c32 step %c1 {
          %12 = arith.muli %arg4, %c8 overflow<nsw> : index
          %13 = arith.addi %12, %arg5 : index
          %14 = loom.alloc [1, 128, 128] on @L1 : memref<1x128x128xf32>
          %15 = loom.semaphore_take %14 : memref<1x128x128xf32> -> memref<1x128x128xf32>
          %16 = loom.alloc [1, 64, 128] on @L1 : memref<1x64x128xf32>
          %17 = loom.semaphore_take %16 : memref<1x64x128xf32> -> memref<1x64x128xf32>
          %18 = loom.alloc [1, 128, 128] on @L1 : memref<1x128x128xf32>
          %19 = loom.semaphore_take %18 : memref<1x128x128xf32> -> memref<1x128x128xf32>
          %20 = loom.alloc [1, 64] on @L1 : memref<1x64xf32>
          %21 = loom.semaphore_take %20 : memref<1x64xf32> -> memref<1x64xf32>
          %22 = loom.alloc [1, 64] on @L1 : memref<1x64xf32>
          %23 = loom.semaphore_take %22 : memref<1x64xf32> -> memref<1x64xf32>
          %24 = loom.alloc [1, 64] on @L1 : memref<1x64xf32>
          %25 = loom.semaphore_take %24 : memref<1x64xf32> -> memref<1x64xf32>
          %26 = loom.alloc [1, 64] on @L1 : memref<1x64xf32>
          %27 = loom.semaphore_take %26 : memref<1x64xf32> -> memref<1x64xf32>
          %28 = loom.alloc [1, 64] on @L1 : memref<1x64xf32>
          %29 = loom.semaphore_take %28 : memref<1x64xf32> -> memref<1x64xf32>
          %30 = loom.alloc [1, 64, 128] on @L1 : memref<1x64x128xf32>
          %31 = loom.semaphore_take %30 : memref<1x64x128xf32> -> memref<1x64x128xf32>
          %32 = loom.semaphore_take %30 : memref<1x64x128xf32> -> memref<1x64x128xf32>
          %33 = loom.alloc [1, 64, 128] on @L1 : memref<1x64x128xf32>
          %34 = loom.semaphore_take %33 : memref<1x64x128xf32> -> memref<1x64x128xf32>
          %35 = loom.alloc [1, 64, 128] on @L1 : memref<1x64x128xf32>
          %36 = loom.semaphore_take %35 : memref<1x64x128xf32> -> memref<1x64x128xf32>
          %37 = arith.muli %arg6, %c524288 overflow<nsw> : index
          %38 = arith.muli %13, %c8192 : index
          %39 = arith.addi %37, %38 : index
          %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%39], sizes: [1, 64, 128], strides: [524288, 128, 1] : memref<32x4096x128xf32> to memref<1x64x128xf32, strided<[524288, 128, 1], offset: ?>>
          loom.copy %reinterpret_cast, %32 src_mem_space @DRAM dst_mem_space @L1, interconnect : [], broadcast : [1, 1] : memref<1x64x128xf32, strided<[524288, 128, 1], offset: ?>>, memref<1x64x128xf32>
          linalg.fill ins(%cst_1 : f32) outs(%34 : memref<1x64x128xf32>)
          linalg.fill ins(%cst_2 : f32) outs(%21 : memref<1x64xf32>)
          linalg.fill ins(%cst_3 : f32) outs(%23 : memref<1x64xf32>)
          scf.for %arg7 = %c0 to %c32 step %c1 {
            %40 = arith.muli %arg7, %c128 : index
            %41 = arith.addi %37, %40 : index
            %reinterpret_cast_5 = memref.reinterpret_cast %arg0 to offset: [%41], sizes: [1, 128, 128], strides: [524288, 4096, 1] : memref<32x128x4096xf32> to memref<1x128x128xf32, strided<[524288, 4096, 1], offset: ?>>
            loom.copy %reinterpret_cast_5, %19 src_mem_space @DRAM dst_mem_space @L1, interconnect : [@horizontal_links, @vertical_links], broadcast : [8, 8] : memref<1x128x128xf32, strided<[524288, 4096, 1], offset: ?>>, memref<1x128x128xf32>
            linalg.fill ins(%cst_1 : f32) outs(%17 : memref<1x64x128xf32>)
            linalg.batch_matmul ins(%32, %19 : memref<1x64x128xf32>, memref<1x128x128xf32>) outs(%17 : memref<1x64x128xf32>)
            loom.semaphore_give %19 : memref<1x128x128xf32>
            linalg.fill ins(%cst_3 : f32) outs(%25 : memref<1x64xf32>)
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%17 : memref<1x64x128xf32>) outs(%25 : memref<1x64xf32>) {
            ^bb0(%in: f32, %out: f32):
              %44 = arith.maximumf %in, %out : f32
              linalg.yield %44 : f32
            }
            linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%23, %25 : memref<1x64xf32>, memref<1x64xf32>) outs(%25 : memref<1x64xf32>) {
            ^bb0(%in: f32, %in_7: f32, %out: f32):
              %44 = arith.truncf %cst_0 : f64 to f32
              %45 = arith.mulf %in_7, %44 : f32
              %46 = arith.cmpf ogt, %in, %45 : f32
              %47 = arith.select %46, %in, %45 : f32
              linalg.yield %47 : f32
            }
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%17, %25 : memref<1x64x128xf32>, memref<1x64xf32>) outs(%17 : memref<1x64x128xf32>) {
            ^bb0(%in: f32, %in_7: f32, %out: f32):
              %44 = arith.truncf %cst_0 : f64 to f32
              %45 = arith.mulf %in, %44 : f32
              %46 = arith.subf %45, %in_7 : f32
              %47 = math.powf %cst, %46 : f32
              linalg.yield %47 : f32
            }
            linalg.fill ins(%cst_1 : f32) outs(%27 : memref<1x64xf32>)
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%17 : memref<1x64x128xf32>) outs(%27 : memref<1x64xf32>) {
            ^bb0(%in: f32, %out: f32):
              %44 = arith.addf %in, %out : f32
              linalg.yield %44 : f32
            }
            linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%23, %25 : memref<1x64xf32>, memref<1x64xf32>) outs(%29 : memref<1x64xf32>) {
            ^bb0(%in: f32, %in_7: f32, %out: f32):
              %44 = arith.subf %in, %in_7 : f32
              %45 = math.powf %cst, %44 : f32
              linalg.yield %45 : f32
            }
            linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%21, %29, %27 : memref<1x64xf32>, memref<1x64xf32>, memref<1x64xf32>) outs(%21 : memref<1x64xf32>) {
            ^bb0(%in: f32, %in_7: f32, %in_8: f32, %out: f32):
              %44 = arith.mulf %in, %in_7 : f32
              %45 = arith.addf %44, %in_8 : f32
              linalg.yield %45 : f32
            }
            loom.semaphore_give %27 : memref<1x64xf32>
            %42 = arith.muli %arg7, %c16384 : index
            %43 = arith.addi %37, %42 : index
            %reinterpret_cast_6 = memref.reinterpret_cast %arg1 to offset: [%43], sizes: [1, 128, 128], strides: [524288, 128, 1] : memref<32x4096x128xf32> to memref<1x128x128xf32, strided<[524288, 128, 1], offset: ?>>
            loom.copy %reinterpret_cast_6, %15 src_mem_space @DRAM dst_mem_space @L1, interconnect : [@vertical_links], broadcast : [8, 1] : memref<1x128x128xf32, strided<[524288, 128, 1], offset: ?>>, memref<1x128x128xf32>
            linalg.fill ins(%cst_1 : f32) outs(%36 : memref<1x64x128xf32>)
            linalg.batch_matmul ins(%17, %15 : memref<1x64x128xf32>, memref<1x128x128xf32>) outs(%36 : memref<1x64x128xf32>)
            loom.semaphore_give %15 : memref<1x128x128xf32>
            loom.semaphore_give %17 : memref<1x64x128xf32>
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%36, %34, %29 : memref<1x64x128xf32>, memref<1x64x128xf32>, memref<1x64xf32>) outs(%34 : memref<1x64x128xf32>) {
            ^bb0(%in: f32, %in_7: f32, %in_8: f32, %out: f32):
              %44 = arith.mulf %in_7, %in_8 : f32
              %45 = arith.addf %in, %44 : f32
              linalg.yield %45 : f32
            }
            loom.semaphore_give %29 : memref<1x64xf32>
            loom.semaphore_give %36 : memref<1x64x128xf32>
            linalg.copy ins(%25 : memref<1x64xf32>) outs(%23 : memref<1x64xf32>)
            loom.semaphore_give %25 : memref<1x64xf32>
          }
          loom.semaphore_give %23 : memref<1x64xf32>
          loom.semaphore_give %32 : memref<1x64x128xf32>
          linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%34, %21 : memref<1x64x128xf32>, memref<1x64xf32>) outs(%31 : memref<1x64x128xf32>) {
          ^bb0(%in: f32, %in_5: f32, %out: f32):
            %40 = arith.divf %in, %in_5 : f32
            linalg.yield %40 : f32
          }
          loom.semaphore_give %21 : memref<1x64xf32>
          loom.semaphore_give %34 : memref<1x64x128xf32>
          %reinterpret_cast_4 = memref.reinterpret_cast %arg3 to offset: [%39], sizes: [1, 64, 128], strides: [524288, 128, 1] : memref<32x4096x128xf32> to memref<1x64x128xf32, strided<[524288, 128, 1], offset: ?>>
          loom.copy %31, %reinterpret_cast_4 src_mem_space @L1 dst_mem_space @DRAM, interconnect : [], broadcast : [1, 1] : memref<1x64x128xf32>, memref<1x64x128xf32, strided<[524288, 128, 1], offset: ?>>
          loom.semaphore_give %31 : memref<1x64x128xf32>
        }
        scf.reduce 
      } {loom.iter_types = [#loom.iter_type<spatial>, #loom.iter_type<spatial>], loom.mapped_to_dims = [@y, @x]}
      return
    }
  }
  module attributes {loom.pass_name = "Materialize"} {
    func.func @attention__d1i1_d0i1__f01__d_h_d__BB1__BM64__BN128(%arg0: memref<32x128x4096xf32>, %arg1: memref<32x4096x128xf32>, %arg2: memref<32x4096x128xf32>, %arg3: memref<32x4096x128xf32>) {
      %c16384 = arith.constant 16384 : index
      %c8192 = arith.constant 8192 : index
      %c524288 = arith.constant 524288 : index
      %c32 = arith.constant 32 : index
      %cst = arith.constant 2.000000e+00 : f32
      %cst_0 = arith.constant 0.12751743074602467 : f64
      %cst_1 = arith.constant 0.000000e+00 : f32
      %cst_2 = arith.constant 1.000000e+00 : f32
      %cst_3 = arith.constant 0xFF800000 : f32
      %c128 = arith.constant 128 : index
      %c0 = arith.constant 0 : index
      %c8 = arith.constant 8 : index
      %c1 = arith.constant 1 : index
      scf.parallel (%arg4, %arg5) = (%c0, %c0) to (%c8, %c8) step (%c1, %c1) {
        scf.for %arg6 = %c0 to %c32 step %c1 {
          %12 = arith.muli %arg4, %c8 overflow<nsw> : index
          %13 = arith.addi %12, %arg5 : index
          %14 = loom.alloc [1, 128, 128] on @L1 : memref<1x128x128xf32>
          %15 = loom.semaphore_take %14 : memref<1x128x128xf32> -> memref<1x128x128xf32>
          %16 = loom.alloc [1, 64, 128] on @L1 : memref<1x64x128xf32>
          %17 = loom.semaphore_take %16 : memref<1x64x128xf32> -> memref<1x64x128xf32>
          %18 = loom.alloc [1, 128, 128] on @L1 : memref<1x128x128xf32>
          %19 = loom.semaphore_take %18 : memref<1x128x128xf32> -> memref<1x128x128xf32>
          %20 = loom.alloc [1, 64] on @L1 : memref<1x64xf32>
          %21 = loom.semaphore_take %20 : memref<1x64xf32> -> memref<1x64xf32>
          %22 = loom.alloc [1, 64] on @L1 : memref<1x64xf32>
          %23 = loom.semaphore_take %22 : memref<1x64xf32> -> memref<1x64xf32>
          %24 = loom.alloc [1, 64] on @L1 : memref<1x64xf32>
          %25 = loom.semaphore_take %24 : memref<1x64xf32> -> memref<1x64xf32>
          %26 = loom.alloc [1, 64] on @L1 : memref<1x64xf32>
          %27 = loom.semaphore_take %26 : memref<1x64xf32> -> memref<1x64xf32>
          %28 = loom.alloc [1, 64] on @L1 : memref<1x64xf32>
          %29 = loom.semaphore_take %28 : memref<1x64xf32> -> memref<1x64xf32>
          %30 = loom.alloc [1, 64, 128] on @L1 : memref<1x64x128xf32>
          %31 = loom.semaphore_take %30 : memref<1x64x128xf32> -> memref<1x64x128xf32>
          %32 = loom.semaphore_take %30 : memref<1x64x128xf32> -> memref<1x64x128xf32>
          %33 = loom.alloc [1, 64, 128] on @L1 : memref<1x64x128xf32>
          %34 = loom.semaphore_take %33 : memref<1x64x128xf32> -> memref<1x64x128xf32>
          %35 = loom.alloc [1, 64, 128] on @L1 : memref<1x64x128xf32>
          %36 = loom.semaphore_take %35 : memref<1x64x128xf32> -> memref<1x64x128xf32>
          %37 = arith.muli %arg6, %c524288 overflow<nsw> : index
          %38 = arith.muli %13, %c8192 : index
          %39 = arith.addi %37, %38 : index
          %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%39], sizes: [1, 64, 128], strides: [524288, 128, 1] : memref<32x4096x128xf32> to memref<1x64x128xf32, strided<[524288, 128, 1], offset: ?>>
          loom.copy %reinterpret_cast, %32 src_mem_space @DRAM dst_mem_space @L1, interconnect : [], broadcast : [1, 1] : memref<1x64x128xf32, strided<[524288, 128, 1], offset: ?>>, memref<1x64x128xf32>
          linalg.fill ins(%cst_1 : f32) outs(%34 : memref<1x64x128xf32>)
          linalg.fill ins(%cst_2 : f32) outs(%21 : memref<1x64xf32>)
          linalg.fill ins(%cst_3 : f32) outs(%23 : memref<1x64xf32>)
          scf.for %arg7 = %c0 to %c32 step %c1 {
            %40 = arith.muli %arg7, %c128 : index
            %41 = arith.addi %37, %40 : index
            %reinterpret_cast_5 = memref.reinterpret_cast %arg0 to offset: [%41], sizes: [1, 128, 128], strides: [524288, 4096, 1] : memref<32x128x4096xf32> to memref<1x128x128xf32, strided<[524288, 4096, 1], offset: ?>>
            loom.copy %reinterpret_cast_5, %19 src_mem_space @DRAM dst_mem_space @L1, interconnect : [@horizontal_links], broadcast : [1, 8] : memref<1x128x128xf32, strided<[524288, 4096, 1], offset: ?>>, memref<1x128x128xf32>
            linalg.fill ins(%cst_1 : f32) outs(%17 : memref<1x64x128xf32>)
            linalg.batch_matmul ins(%32, %19 : memref<1x64x128xf32>, memref<1x128x128xf32>) outs(%17 : memref<1x64x128xf32>)
            loom.semaphore_give %19 : memref<1x128x128xf32>
            linalg.fill ins(%cst_3 : f32) outs(%25 : memref<1x64xf32>)
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%17 : memref<1x64x128xf32>) outs(%25 : memref<1x64xf32>) {
            ^bb0(%in: f32, %out: f32):
              %44 = arith.maximumf %in, %out : f32
              linalg.yield %44 : f32
            }
            linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%23, %25 : memref<1x64xf32>, memref<1x64xf32>) outs(%25 : memref<1x64xf32>) {
            ^bb0(%in: f32, %in_7: f32, %out: f32):
              %44 = arith.truncf %cst_0 : f64 to f32
              %45 = arith.mulf %in_7, %44 : f32
              %46 = arith.cmpf ogt, %in, %45 : f32
              %47 = arith.select %46, %in, %45 : f32
              linalg.yield %47 : f32
            }
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%17, %25 : memref<1x64x128xf32>, memref<1x64xf32>) outs(%17 : memref<1x64x128xf32>) {
            ^bb0(%in: f32, %in_7: f32, %out: f32):
              %44 = arith.truncf %cst_0 : f64 to f32
              %45 = arith.mulf %in, %44 : f32
              %46 = arith.subf %45, %in_7 : f32
              %47 = math.powf %cst, %46 : f32
              linalg.yield %47 : f32
            }
            linalg.fill ins(%cst_1 : f32) outs(%27 : memref<1x64xf32>)
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%17 : memref<1x64x128xf32>) outs(%27 : memref<1x64xf32>) {
            ^bb0(%in: f32, %out: f32):
              %44 = arith.addf %in, %out : f32
              linalg.yield %44 : f32
            }
            linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%23, %25 : memref<1x64xf32>, memref<1x64xf32>) outs(%29 : memref<1x64xf32>) {
            ^bb0(%in: f32, %in_7: f32, %out: f32):
              %44 = arith.subf %in, %in_7 : f32
              %45 = math.powf %cst, %44 : f32
              linalg.yield %45 : f32
            }
            linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%21, %29, %27 : memref<1x64xf32>, memref<1x64xf32>, memref<1x64xf32>) outs(%21 : memref<1x64xf32>) {
            ^bb0(%in: f32, %in_7: f32, %in_8: f32, %out: f32):
              %44 = arith.mulf %in, %in_7 : f32
              %45 = arith.addf %44, %in_8 : f32
              linalg.yield %45 : f32
            }
            loom.semaphore_give %27 : memref<1x64xf32>
            %42 = arith.muli %arg7, %c16384 : index
            %43 = arith.addi %37, %42 : index
            %reinterpret_cast_6 = memref.reinterpret_cast %arg1 to offset: [%43], sizes: [1, 128, 128], strides: [524288, 128, 1] : memref<32x4096x128xf32> to memref<1x128x128xf32, strided<[524288, 128, 1], offset: ?>>
            loom.copy %reinterpret_cast_6, %15 src_mem_space @DRAM dst_mem_space @L1, interconnect : [], broadcast : [1, 1] : memref<1x128x128xf32, strided<[524288, 128, 1], offset: ?>>, memref<1x128x128xf32>
            linalg.fill ins(%cst_1 : f32) outs(%36 : memref<1x64x128xf32>)
            linalg.batch_matmul ins(%17, %15 : memref<1x64x128xf32>, memref<1x128x128xf32>) outs(%36 : memref<1x64x128xf32>)
            loom.semaphore_give %15 : memref<1x128x128xf32>
            loom.semaphore_give %17 : memref<1x64x128xf32>
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%36, %34, %29 : memref<1x64x128xf32>, memref<1x64x128xf32>, memref<1x64xf32>) outs(%34 : memref<1x64x128xf32>) {
            ^bb0(%in: f32, %in_7: f32, %in_8: f32, %out: f32):
              %44 = arith.mulf %in_7, %in_8 : f32
              %45 = arith.addf %in, %44 : f32
              linalg.yield %45 : f32
            }
            loom.semaphore_give %29 : memref<1x64xf32>
            loom.semaphore_give %36 : memref<1x64x128xf32>
            linalg.copy ins(%25 : memref<1x64xf32>) outs(%23 : memref<1x64xf32>)
            loom.semaphore_give %25 : memref<1x64xf32>
          }
          loom.semaphore_give %23 : memref<1x64xf32>
          loom.semaphore_give %32 : memref<1x64x128xf32>
          linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%34, %21 : memref<1x64x128xf32>, memref<1x64xf32>) outs(%31 : memref<1x64x128xf32>) {
          ^bb0(%in: f32, %in_5: f32, %out: f32):
            %40 = arith.divf %in, %in_5 : f32
            linalg.yield %40 : f32
          }
          loom.semaphore_give %21 : memref<1x64xf32>
          loom.semaphore_give %34 : memref<1x64x128xf32>
          %reinterpret_cast_4 = memref.reinterpret_cast %arg3 to offset: [%39], sizes: [1, 64, 128], strides: [524288, 128, 1] : memref<32x4096x128xf32> to memref<1x64x128xf32, strided<[524288, 128, 1], offset: ?>>
          loom.copy %31, %reinterpret_cast_4 src_mem_space @L1 dst_mem_space @DRAM, interconnect : [], broadcast : [1, 1] : memref<1x64x128xf32>, memref<1x64x128xf32, strided<[524288, 128, 1], offset: ?>>
          loom.semaphore_give %31 : memref<1x64x128xf32>
        }
        scf.reduce 
      } {loom.iter_types = [#loom.iter_type<spatial>, #loom.iter_type<spatial>], loom.mapped_to_dims = [@y, @x]}
      return
    }
  }
  module attributes {loom.pass_name = "Materialize"} {
    func.func @attention__d1i1_d0i1__f01__d_h_a__BB1__BM64__BN128(%arg0: memref<32x128x4096xf32>, %arg1: memref<32x4096x128xf32>, %arg2: memref<32x4096x128xf32>, %arg3: memref<32x4096x128xf32>) {
      %c16384 = arith.constant 16384 : index
      %c8192 = arith.constant 8192 : index
      %c524288 = arith.constant 524288 : index
      %c32 = arith.constant 32 : index
      %cst = arith.constant 2.000000e+00 : f32
      %cst_0 = arith.constant 0.12751743074602467 : f64
      %cst_1 = arith.constant 0.000000e+00 : f32
      %cst_2 = arith.constant 1.000000e+00 : f32
      %cst_3 = arith.constant 0xFF800000 : f32
      %c128 = arith.constant 128 : index
      %c0 = arith.constant 0 : index
      %c8 = arith.constant 8 : index
      %c1 = arith.constant 1 : index
      scf.parallel (%arg4, %arg5) = (%c0, %c0) to (%c8, %c8) step (%c1, %c1) {
        scf.for %arg6 = %c0 to %c32 step %c1 {
          %12 = arith.muli %arg4, %c8 overflow<nsw> : index
          %13 = arith.addi %12, %arg5 : index
          %14 = loom.alloc [1, 128, 128] on @L1 : memref<1x128x128xf32>
          %15 = loom.semaphore_take %14 : memref<1x128x128xf32> -> memref<1x128x128xf32>
          %16 = loom.alloc [1, 64, 128] on @L1 : memref<1x64x128xf32>
          %17 = loom.semaphore_take %16 : memref<1x64x128xf32> -> memref<1x64x128xf32>
          %18 = loom.alloc [1, 128, 128] on @L1 : memref<1x128x128xf32>
          %19 = loom.semaphore_take %18 : memref<1x128x128xf32> -> memref<1x128x128xf32>
          %20 = loom.alloc [1, 64] on @L1 : memref<1x64xf32>
          %21 = loom.semaphore_take %20 : memref<1x64xf32> -> memref<1x64xf32>
          %22 = loom.alloc [1, 64] on @L1 : memref<1x64xf32>
          %23 = loom.semaphore_take %22 : memref<1x64xf32> -> memref<1x64xf32>
          %24 = loom.alloc [1, 64] on @L1 : memref<1x64xf32>
          %25 = loom.semaphore_take %24 : memref<1x64xf32> -> memref<1x64xf32>
          %26 = loom.alloc [1, 64] on @L1 : memref<1x64xf32>
          %27 = loom.semaphore_take %26 : memref<1x64xf32> -> memref<1x64xf32>
          %28 = loom.alloc [1, 64] on @L1 : memref<1x64xf32>
          %29 = loom.semaphore_take %28 : memref<1x64xf32> -> memref<1x64xf32>
          %30 = loom.alloc [1, 64, 128] on @L1 : memref<1x64x128xf32>
          %31 = loom.semaphore_take %30 : memref<1x64x128xf32> -> memref<1x64x128xf32>
          %32 = loom.semaphore_take %30 : memref<1x64x128xf32> -> memref<1x64x128xf32>
          %33 = loom.alloc [1, 64, 128] on @L1 : memref<1x64x128xf32>
          %34 = loom.semaphore_take %33 : memref<1x64x128xf32> -> memref<1x64x128xf32>
          %35 = loom.alloc [1, 64, 128] on @L1 : memref<1x64x128xf32>
          %36 = loom.semaphore_take %35 : memref<1x64x128xf32> -> memref<1x64x128xf32>
          %37 = arith.muli %arg6, %c524288 overflow<nsw> : index
          %38 = arith.muli %13, %c8192 : index
          %39 = arith.addi %37, %38 : index
          %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%39], sizes: [1, 64, 128], strides: [524288, 128, 1] : memref<32x4096x128xf32> to memref<1x64x128xf32, strided<[524288, 128, 1], offset: ?>>
          loom.copy %reinterpret_cast, %32 src_mem_space @DRAM dst_mem_space @L1, interconnect : [], broadcast : [1, 1] : memref<1x64x128xf32, strided<[524288, 128, 1], offset: ?>>, memref<1x64x128xf32>
          linalg.fill ins(%cst_1 : f32) outs(%34 : memref<1x64x128xf32>)
          linalg.fill ins(%cst_2 : f32) outs(%21 : memref<1x64xf32>)
          linalg.fill ins(%cst_3 : f32) outs(%23 : memref<1x64xf32>)
          scf.for %arg7 = %c0 to %c32 step %c1 {
            %40 = arith.muli %arg7, %c128 : index
            %41 = arith.addi %37, %40 : index
            %reinterpret_cast_5 = memref.reinterpret_cast %arg0 to offset: [%41], sizes: [1, 128, 128], strides: [524288, 4096, 1] : memref<32x128x4096xf32> to memref<1x128x128xf32, strided<[524288, 4096, 1], offset: ?>>
            loom.copy %reinterpret_cast_5, %19 src_mem_space @DRAM dst_mem_space @L1, interconnect : [@horizontal_links], broadcast : [1, 8] : memref<1x128x128xf32, strided<[524288, 4096, 1], offset: ?>>, memref<1x128x128xf32>
            linalg.fill ins(%cst_1 : f32) outs(%17 : memref<1x64x128xf32>)
            linalg.batch_matmul ins(%32, %19 : memref<1x64x128xf32>, memref<1x128x128xf32>) outs(%17 : memref<1x64x128xf32>)
            loom.semaphore_give %19 : memref<1x128x128xf32>
            linalg.fill ins(%cst_3 : f32) outs(%25 : memref<1x64xf32>)
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%17 : memref<1x64x128xf32>) outs(%25 : memref<1x64xf32>) {
            ^bb0(%in: f32, %out: f32):
              %44 = arith.maximumf %in, %out : f32
              linalg.yield %44 : f32
            }
            linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%23, %25 : memref<1x64xf32>, memref<1x64xf32>) outs(%25 : memref<1x64xf32>) {
            ^bb0(%in: f32, %in_7: f32, %out: f32):
              %44 = arith.truncf %cst_0 : f64 to f32
              %45 = arith.mulf %in_7, %44 : f32
              %46 = arith.cmpf ogt, %in, %45 : f32
              %47 = arith.select %46, %in, %45 : f32
              linalg.yield %47 : f32
            }
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%17, %25 : memref<1x64x128xf32>, memref<1x64xf32>) outs(%17 : memref<1x64x128xf32>) {
            ^bb0(%in: f32, %in_7: f32, %out: f32):
              %44 = arith.truncf %cst_0 : f64 to f32
              %45 = arith.mulf %in, %44 : f32
              %46 = arith.subf %45, %in_7 : f32
              %47 = math.powf %cst, %46 : f32
              linalg.yield %47 : f32
            }
            linalg.fill ins(%cst_1 : f32) outs(%27 : memref<1x64xf32>)
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%17 : memref<1x64x128xf32>) outs(%27 : memref<1x64xf32>) {
            ^bb0(%in: f32, %out: f32):
              %44 = arith.addf %in, %out : f32
              linalg.yield %44 : f32
            }
            linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%23, %25 : memref<1x64xf32>, memref<1x64xf32>) outs(%29 : memref<1x64xf32>) {
            ^bb0(%in: f32, %in_7: f32, %out: f32):
              %44 = arith.subf %in, %in_7 : f32
              %45 = math.powf %cst, %44 : f32
              linalg.yield %45 : f32
            }
            linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%21, %29, %27 : memref<1x64xf32>, memref<1x64xf32>, memref<1x64xf32>) outs(%21 : memref<1x64xf32>) {
            ^bb0(%in: f32, %in_7: f32, %in_8: f32, %out: f32):
              %44 = arith.mulf %in, %in_7 : f32
              %45 = arith.addf %44, %in_8 : f32
              linalg.yield %45 : f32
            }
            loom.semaphore_give %27 : memref<1x64xf32>
            %42 = arith.muli %arg7, %c16384 : index
            %43 = arith.addi %37, %42 : index
            %reinterpret_cast_6 = memref.reinterpret_cast %arg1 to offset: [%43], sizes: [1, 128, 128], strides: [524288, 128, 1] : memref<32x4096x128xf32> to memref<1x128x128xf32, strided<[524288, 128, 1], offset: ?>>
            loom.copy %reinterpret_cast_6, %15 src_mem_space @DRAM dst_mem_space @L1, interconnect : [@horizontal_links, @vertical_links], broadcast : [8, 8] : memref<1x128x128xf32, strided<[524288, 128, 1], offset: ?>>, memref<1x128x128xf32>
            linalg.fill ins(%cst_1 : f32) outs(%36 : memref<1x64x128xf32>)
            linalg.batch_matmul ins(%17, %15 : memref<1x64x128xf32>, memref<1x128x128xf32>) outs(%36 : memref<1x64x128xf32>)
            loom.semaphore_give %15 : memref<1x128x128xf32>
            loom.semaphore_give %17 : memref<1x64x128xf32>
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%36, %34, %29 : memref<1x64x128xf32>, memref<1x64x128xf32>, memref<1x64xf32>) outs(%34 : memref<1x64x128xf32>) {
            ^bb0(%in: f32, %in_7: f32, %in_8: f32, %out: f32):
              %44 = arith.mulf %in_7, %in_8 : f32
              %45 = arith.addf %in, %44 : f32
              linalg.yield %45 : f32
            }
            loom.semaphore_give %29 : memref<1x64xf32>
            loom.semaphore_give %36 : memref<1x64x128xf32>
            linalg.copy ins(%25 : memref<1x64xf32>) outs(%23 : memref<1x64xf32>)
            loom.semaphore_give %25 : memref<1x64xf32>
          }
          loom.semaphore_give %23 : memref<1x64xf32>
          loom.semaphore_give %32 : memref<1x64x128xf32>
          linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%34, %21 : memref<1x64x128xf32>, memref<1x64xf32>) outs(%31 : memref<1x64x128xf32>) {
          ^bb0(%in: f32, %in_5: f32, %out: f32):
            %40 = arith.divf %in, %in_5 : f32
            linalg.yield %40 : f32
          }
          loom.semaphore_give %21 : memref<1x64xf32>
          loom.semaphore_give %34 : memref<1x64x128xf32>
          %reinterpret_cast_4 = memref.reinterpret_cast %arg3 to offset: [%39], sizes: [1, 64, 128], strides: [524288, 128, 1] : memref<32x4096x128xf32> to memref<1x64x128xf32, strided<[524288, 128, 1], offset: ?>>
          loom.copy %31, %reinterpret_cast_4 src_mem_space @L1 dst_mem_space @DRAM, interconnect : [], broadcast : [1, 1] : memref<1x64x128xf32>, memref<1x64x128xf32, strided<[524288, 128, 1], offset: ?>>
          loom.semaphore_give %31 : memref<1x64x128xf32>
        }
        scf.reduce 
      } {loom.iter_types = [#loom.iter_type<spatial>, #loom.iter_type<spatial>], loom.mapped_to_dims = [@y, @x]}
      return
    }
  }
  module attributes {loom.pass_name = "Materialize"} {
    func.func @attention__d1i1_d0i1__f01__d_h_h__BB1__BM64__BN128(%arg0: memref<32x128x4096xf32>, %arg1: memref<32x4096x128xf32>, %arg2: memref<32x4096x128xf32>, %arg3: memref<32x4096x128xf32>) {
      %c16384 = arith.constant 16384 : index
      %c8192 = arith.constant 8192 : index
      %c524288 = arith.constant 524288 : index
      %c32 = arith.constant 32 : index
      %cst = arith.constant 2.000000e+00 : f32
      %cst_0 = arith.constant 0.12751743074602467 : f64
      %cst_1 = arith.constant 0.000000e+00 : f32
      %cst_2 = arith.constant 1.000000e+00 : f32
      %cst_3 = arith.constant 0xFF800000 : f32
      %c128 = arith.constant 128 : index
      %c0 = arith.constant 0 : index
      %c8 = arith.constant 8 : index
      %c1 = arith.constant 1 : index
      scf.parallel (%arg4, %arg5) = (%c0, %c0) to (%c8, %c8) step (%c1, %c1) {
        scf.for %arg6 = %c0 to %c32 step %c1 {
          %12 = arith.muli %arg4, %c8 overflow<nsw> : index
          %13 = arith.addi %12, %arg5 : index
          %14 = loom.alloc [1, 128, 128] on @L1 : memref<1x128x128xf32>
          %15 = loom.semaphore_take %14 : memref<1x128x128xf32> -> memref<1x128x128xf32>
          %16 = loom.alloc [1, 64, 128] on @L1 : memref<1x64x128xf32>
          %17 = loom.semaphore_take %16 : memref<1x64x128xf32> -> memref<1x64x128xf32>
          %18 = loom.alloc [1, 128, 128] on @L1 : memref<1x128x128xf32>
          %19 = loom.semaphore_take %18 : memref<1x128x128xf32> -> memref<1x128x128xf32>
          %20 = loom.alloc [1, 64] on @L1 : memref<1x64xf32>
          %21 = loom.semaphore_take %20 : memref<1x64xf32> -> memref<1x64xf32>
          %22 = loom.alloc [1, 64] on @L1 : memref<1x64xf32>
          %23 = loom.semaphore_take %22 : memref<1x64xf32> -> memref<1x64xf32>
          %24 = loom.alloc [1, 64] on @L1 : memref<1x64xf32>
          %25 = loom.semaphore_take %24 : memref<1x64xf32> -> memref<1x64xf32>
          %26 = loom.alloc [1, 64] on @L1 : memref<1x64xf32>
          %27 = loom.semaphore_take %26 : memref<1x64xf32> -> memref<1x64xf32>
          %28 = loom.alloc [1, 64] on @L1 : memref<1x64xf32>
          %29 = loom.semaphore_take %28 : memref<1x64xf32> -> memref<1x64xf32>
          %30 = loom.alloc [1, 64, 128] on @L1 : memref<1x64x128xf32>
          %31 = loom.semaphore_take %30 : memref<1x64x128xf32> -> memref<1x64x128xf32>
          %32 = loom.semaphore_take %30 : memref<1x64x128xf32> -> memref<1x64x128xf32>
          %33 = loom.alloc [1, 64, 128] on @L1 : memref<1x64x128xf32>
          %34 = loom.semaphore_take %33 : memref<1x64x128xf32> -> memref<1x64x128xf32>
          %35 = loom.alloc [1, 64, 128] on @L1 : memref<1x64x128xf32>
          %36 = loom.semaphore_take %35 : memref<1x64x128xf32> -> memref<1x64x128xf32>
          %37 = arith.muli %arg6, %c524288 overflow<nsw> : index
          %38 = arith.muli %13, %c8192 : index
          %39 = arith.addi %37, %38 : index
          %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%39], sizes: [1, 64, 128], strides: [524288, 128, 1] : memref<32x4096x128xf32> to memref<1x64x128xf32, strided<[524288, 128, 1], offset: ?>>
          loom.copy %reinterpret_cast, %32 src_mem_space @DRAM dst_mem_space @L1, interconnect : [], broadcast : [1, 1] : memref<1x64x128xf32, strided<[524288, 128, 1], offset: ?>>, memref<1x64x128xf32>
          linalg.fill ins(%cst_1 : f32) outs(%34 : memref<1x64x128xf32>)
          linalg.fill ins(%cst_2 : f32) outs(%21 : memref<1x64xf32>)
          linalg.fill ins(%cst_3 : f32) outs(%23 : memref<1x64xf32>)
          scf.for %arg7 = %c0 to %c32 step %c1 {
            %40 = arith.muli %arg7, %c128 : index
            %41 = arith.addi %37, %40 : index
            %reinterpret_cast_5 = memref.reinterpret_cast %arg0 to offset: [%41], sizes: [1, 128, 128], strides: [524288, 4096, 1] : memref<32x128x4096xf32> to memref<1x128x128xf32, strided<[524288, 4096, 1], offset: ?>>
            loom.copy %reinterpret_cast_5, %19 src_mem_space @DRAM dst_mem_space @L1, interconnect : [@horizontal_links], broadcast : [1, 8] : memref<1x128x128xf32, strided<[524288, 4096, 1], offset: ?>>, memref<1x128x128xf32>
            linalg.fill ins(%cst_1 : f32) outs(%17 : memref<1x64x128xf32>)
            linalg.batch_matmul ins(%32, %19 : memref<1x64x128xf32>, memref<1x128x128xf32>) outs(%17 : memref<1x64x128xf32>)
            loom.semaphore_give %19 : memref<1x128x128xf32>
            linalg.fill ins(%cst_3 : f32) outs(%25 : memref<1x64xf32>)
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%17 : memref<1x64x128xf32>) outs(%25 : memref<1x64xf32>) {
            ^bb0(%in: f32, %out: f32):
              %44 = arith.maximumf %in, %out : f32
              linalg.yield %44 : f32
            }
            linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%23, %25 : memref<1x64xf32>, memref<1x64xf32>) outs(%25 : memref<1x64xf32>) {
            ^bb0(%in: f32, %in_7: f32, %out: f32):
              %44 = arith.truncf %cst_0 : f64 to f32
              %45 = arith.mulf %in_7, %44 : f32
              %46 = arith.cmpf ogt, %in, %45 : f32
              %47 = arith.select %46, %in, %45 : f32
              linalg.yield %47 : f32
            }
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%17, %25 : memref<1x64x128xf32>, memref<1x64xf32>) outs(%17 : memref<1x64x128xf32>) {
            ^bb0(%in: f32, %in_7: f32, %out: f32):
              %44 = arith.truncf %cst_0 : f64 to f32
              %45 = arith.mulf %in, %44 : f32
              %46 = arith.subf %45, %in_7 : f32
              %47 = math.powf %cst, %46 : f32
              linalg.yield %47 : f32
            }
            linalg.fill ins(%cst_1 : f32) outs(%27 : memref<1x64xf32>)
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%17 : memref<1x64x128xf32>) outs(%27 : memref<1x64xf32>) {
            ^bb0(%in: f32, %out: f32):
              %44 = arith.addf %in, %out : f32
              linalg.yield %44 : f32
            }
            linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%23, %25 : memref<1x64xf32>, memref<1x64xf32>) outs(%29 : memref<1x64xf32>) {
            ^bb0(%in: f32, %in_7: f32, %out: f32):
              %44 = arith.subf %in, %in_7 : f32
              %45 = math.powf %cst, %44 : f32
              linalg.yield %45 : f32
            }
            linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%21, %29, %27 : memref<1x64xf32>, memref<1x64xf32>, memref<1x64xf32>) outs(%21 : memref<1x64xf32>) {
            ^bb0(%in: f32, %in_7: f32, %in_8: f32, %out: f32):
              %44 = arith.mulf %in, %in_7 : f32
              %45 = arith.addf %44, %in_8 : f32
              linalg.yield %45 : f32
            }
            loom.semaphore_give %27 : memref<1x64xf32>
            %42 = arith.muli %arg7, %c16384 : index
            %43 = arith.addi %37, %42 : index
            %reinterpret_cast_6 = memref.reinterpret_cast %arg1 to offset: [%43], sizes: [1, 128, 128], strides: [524288, 128, 1] : memref<32x4096x128xf32> to memref<1x128x128xf32, strided<[524288, 128, 1], offset: ?>>
            loom.copy %reinterpret_cast_6, %15 src_mem_space @DRAM dst_mem_space @L1, interconnect : [@horizontal_links], broadcast : [1, 8] : memref<1x128x128xf32, strided<[524288, 128, 1], offset: ?>>, memref<1x128x128xf32>
            linalg.fill ins(%cst_1 : f32) outs(%36 : memref<1x64x128xf32>)
            linalg.batch_matmul ins(%17, %15 : memref<1x64x128xf32>, memref<1x128x128xf32>) outs(%36 : memref<1x64x128xf32>)
            loom.semaphore_give %15 : memref<1x128x128xf32>
            loom.semaphore_give %17 : memref<1x64x128xf32>
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%36, %34, %29 : memref<1x64x128xf32>, memref<1x64x128xf32>, memref<1x64xf32>) outs(%34 : memref<1x64x128xf32>) {
            ^bb0(%in: f32, %in_7: f32, %in_8: f32, %out: f32):
              %44 = arith.mulf %in_7, %in_8 : f32
              %45 = arith.addf %in, %44 : f32
              linalg.yield %45 : f32
            }
            loom.semaphore_give %29 : memref<1x64xf32>
            loom.semaphore_give %36 : memref<1x64x128xf32>
            linalg.copy ins(%25 : memref<1x64xf32>) outs(%23 : memref<1x64xf32>)
            loom.semaphore_give %25 : memref<1x64xf32>
          }
          loom.semaphore_give %23 : memref<1x64xf32>
          loom.semaphore_give %32 : memref<1x64x128xf32>
          linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%34, %21 : memref<1x64x128xf32>, memref<1x64xf32>) outs(%31 : memref<1x64x128xf32>) {
          ^bb0(%in: f32, %in_5: f32, %out: f32):
            %40 = arith.divf %in, %in_5 : f32
            linalg.yield %40 : f32
          }
          loom.semaphore_give %21 : memref<1x64xf32>
          loom.semaphore_give %34 : memref<1x64x128xf32>
          %reinterpret_cast_4 = memref.reinterpret_cast %arg3 to offset: [%39], sizes: [1, 64, 128], strides: [524288, 128, 1] : memref<32x4096x128xf32> to memref<1x64x128xf32, strided<[524288, 128, 1], offset: ?>>
          loom.copy %31, %reinterpret_cast_4 src_mem_space @L1 dst_mem_space @DRAM, interconnect : [], broadcast : [1, 1] : memref<1x64x128xf32>, memref<1x64x128xf32, strided<[524288, 128, 1], offset: ?>>
          loom.semaphore_give %31 : memref<1x64x128xf32>
        }
        scf.reduce 
      } {loom.iter_types = [#loom.iter_type<spatial>, #loom.iter_type<spatial>], loom.mapped_to_dims = [@y, @x]}
      return
    }
  }
  module attributes {loom.pass_name = "Materialize"} {
    func.func @attention__d1i1_d0i1__f01__d_h_v__BB1__BM64__BN128(%arg0: memref<32x128x4096xf32>, %arg1: memref<32x4096x128xf32>, %arg2: memref<32x4096x128xf32>, %arg3: memref<32x4096x128xf32>) {
      %c16384 = arith.constant 16384 : index
      %c8192 = arith.constant 8192 : index
      %c524288 = arith.constant 524288 : index
      %c32 = arith.constant 32 : index
      %cst = arith.constant 2.000000e+00 : f32
      %cst_0 = arith.constant 0.12751743074602467 : f64
      %cst_1 = arith.constant 0.000000e+00 : f32
      %cst_2 = arith.constant 1.000000e+00 : f32
      %cst_3 = arith.constant 0xFF800000 : f32
      %c128 = arith.constant 128 : index
      %c0 = arith.constant 0 : index
      %c8 = arith.constant 8 : index
      %c1 = arith.constant 1 : index
      scf.parallel (%arg4, %arg5) = (%c0, %c0) to (%c8, %c8) step (%c1, %c1) {
        scf.for %arg6 = %c0 to %c32 step %c1 {
          %12 = arith.muli %arg4, %c8 overflow<nsw> : index
          %13 = arith.addi %12, %arg5 : index
          %14 = loom.alloc [1, 128, 128] on @L1 : memref<1x128x128xf32>
          %15 = loom.semaphore_take %14 : memref<1x128x128xf32> -> memref<1x128x128xf32>
          %16 = loom.alloc [1, 64, 128] on @L1 : memref<1x64x128xf32>
          %17 = loom.semaphore_take %16 : memref<1x64x128xf32> -> memref<1x64x128xf32>
          %18 = loom.alloc [1, 128, 128] on @L1 : memref<1x128x128xf32>
          %19 = loom.semaphore_take %18 : memref<1x128x128xf32> -> memref<1x128x128xf32>
          %20 = loom.alloc [1, 64] on @L1 : memref<1x64xf32>
          %21 = loom.semaphore_take %20 : memref<1x64xf32> -> memref<1x64xf32>
          %22 = loom.alloc [1, 64] on @L1 : memref<1x64xf32>
          %23 = loom.semaphore_take %22 : memref<1x64xf32> -> memref<1x64xf32>
          %24 = loom.alloc [1, 64] on @L1 : memref<1x64xf32>
          %25 = loom.semaphore_take %24 : memref<1x64xf32> -> memref<1x64xf32>
          %26 = loom.alloc [1, 64] on @L1 : memref<1x64xf32>
          %27 = loom.semaphore_take %26 : memref<1x64xf32> -> memref<1x64xf32>
          %28 = loom.alloc [1, 64] on @L1 : memref<1x64xf32>
          %29 = loom.semaphore_take %28 : memref<1x64xf32> -> memref<1x64xf32>
          %30 = loom.alloc [1, 64, 128] on @L1 : memref<1x64x128xf32>
          %31 = loom.semaphore_take %30 : memref<1x64x128xf32> -> memref<1x64x128xf32>
          %32 = loom.semaphore_take %30 : memref<1x64x128xf32> -> memref<1x64x128xf32>
          %33 = loom.alloc [1, 64, 128] on @L1 : memref<1x64x128xf32>
          %34 = loom.semaphore_take %33 : memref<1x64x128xf32> -> memref<1x64x128xf32>
          %35 = loom.alloc [1, 64, 128] on @L1 : memref<1x64x128xf32>
          %36 = loom.semaphore_take %35 : memref<1x64x128xf32> -> memref<1x64x128xf32>
          %37 = arith.muli %arg6, %c524288 overflow<nsw> : index
          %38 = arith.muli %13, %c8192 : index
          %39 = arith.addi %37, %38 : index
          %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%39], sizes: [1, 64, 128], strides: [524288, 128, 1] : memref<32x4096x128xf32> to memref<1x64x128xf32, strided<[524288, 128, 1], offset: ?>>
          loom.copy %reinterpret_cast, %32 src_mem_space @DRAM dst_mem_space @L1, interconnect : [], broadcast : [1, 1] : memref<1x64x128xf32, strided<[524288, 128, 1], offset: ?>>, memref<1x64x128xf32>
          linalg.fill ins(%cst_1 : f32) outs(%34 : memref<1x64x128xf32>)
          linalg.fill ins(%cst_2 : f32) outs(%21 : memref<1x64xf32>)
          linalg.fill ins(%cst_3 : f32) outs(%23 : memref<1x64xf32>)
          scf.for %arg7 = %c0 to %c32 step %c1 {
            %40 = arith.muli %arg7, %c128 : index
            %41 = arith.addi %37, %40 : index
            %reinterpret_cast_5 = memref.reinterpret_cast %arg0 to offset: [%41], sizes: [1, 128, 128], strides: [524288, 4096, 1] : memref<32x128x4096xf32> to memref<1x128x128xf32, strided<[524288, 4096, 1], offset: ?>>
            loom.copy %reinterpret_cast_5, %19 src_mem_space @DRAM dst_mem_space @L1, interconnect : [@horizontal_links], broadcast : [1, 8] : memref<1x128x128xf32, strided<[524288, 4096, 1], offset: ?>>, memref<1x128x128xf32>
            linalg.fill ins(%cst_1 : f32) outs(%17 : memref<1x64x128xf32>)
            linalg.batch_matmul ins(%32, %19 : memref<1x64x128xf32>, memref<1x128x128xf32>) outs(%17 : memref<1x64x128xf32>)
            loom.semaphore_give %19 : memref<1x128x128xf32>
            linalg.fill ins(%cst_3 : f32) outs(%25 : memref<1x64xf32>)
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%17 : memref<1x64x128xf32>) outs(%25 : memref<1x64xf32>) {
            ^bb0(%in: f32, %out: f32):
              %44 = arith.maximumf %in, %out : f32
              linalg.yield %44 : f32
            }
            linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%23, %25 : memref<1x64xf32>, memref<1x64xf32>) outs(%25 : memref<1x64xf32>) {
            ^bb0(%in: f32, %in_7: f32, %out: f32):
              %44 = arith.truncf %cst_0 : f64 to f32
              %45 = arith.mulf %in_7, %44 : f32
              %46 = arith.cmpf ogt, %in, %45 : f32
              %47 = arith.select %46, %in, %45 : f32
              linalg.yield %47 : f32
            }
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%17, %25 : memref<1x64x128xf32>, memref<1x64xf32>) outs(%17 : memref<1x64x128xf32>) {
            ^bb0(%in: f32, %in_7: f32, %out: f32):
              %44 = arith.truncf %cst_0 : f64 to f32
              %45 = arith.mulf %in, %44 : f32
              %46 = arith.subf %45, %in_7 : f32
              %47 = math.powf %cst, %46 : f32
              linalg.yield %47 : f32
            }
            linalg.fill ins(%cst_1 : f32) outs(%27 : memref<1x64xf32>)
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%17 : memref<1x64x128xf32>) outs(%27 : memref<1x64xf32>) {
            ^bb0(%in: f32, %out: f32):
              %44 = arith.addf %in, %out : f32
              linalg.yield %44 : f32
            }
            linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%23, %25 : memref<1x64xf32>, memref<1x64xf32>) outs(%29 : memref<1x64xf32>) {
            ^bb0(%in: f32, %in_7: f32, %out: f32):
              %44 = arith.subf %in, %in_7 : f32
              %45 = math.powf %cst, %44 : f32
              linalg.yield %45 : f32
            }
            linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%21, %29, %27 : memref<1x64xf32>, memref<1x64xf32>, memref<1x64xf32>) outs(%21 : memref<1x64xf32>) {
            ^bb0(%in: f32, %in_7: f32, %in_8: f32, %out: f32):
              %44 = arith.mulf %in, %in_7 : f32
              %45 = arith.addf %44, %in_8 : f32
              linalg.yield %45 : f32
            }
            loom.semaphore_give %27 : memref<1x64xf32>
            %42 = arith.muli %arg7, %c16384 : index
            %43 = arith.addi %37, %42 : index
            %reinterpret_cast_6 = memref.reinterpret_cast %arg1 to offset: [%43], sizes: [1, 128, 128], strides: [524288, 128, 1] : memref<32x4096x128xf32> to memref<1x128x128xf32, strided<[524288, 128, 1], offset: ?>>
            loom.copy %reinterpret_cast_6, %15 src_mem_space @DRAM dst_mem_space @L1, interconnect : [@vertical_links], broadcast : [8, 1] : memref<1x128x128xf32, strided<[524288, 128, 1], offset: ?>>, memref<1x128x128xf32>
            linalg.fill ins(%cst_1 : f32) outs(%36 : memref<1x64x128xf32>)
            linalg.batch_matmul ins(%17, %15 : memref<1x64x128xf32>, memref<1x128x128xf32>) outs(%36 : memref<1x64x128xf32>)
            loom.semaphore_give %15 : memref<1x128x128xf32>
            loom.semaphore_give %17 : memref<1x64x128xf32>
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%36, %34, %29 : memref<1x64x128xf32>, memref<1x64x128xf32>, memref<1x64xf32>) outs(%34 : memref<1x64x128xf32>) {
            ^bb0(%in: f32, %in_7: f32, %in_8: f32, %out: f32):
              %44 = arith.mulf %in_7, %in_8 : f32
              %45 = arith.addf %in, %44 : f32
              linalg.yield %45 : f32
            }
            loom.semaphore_give %29 : memref<1x64xf32>
            loom.semaphore_give %36 : memref<1x64x128xf32>
            linalg.copy ins(%25 : memref<1x64xf32>) outs(%23 : memref<1x64xf32>)
            loom.semaphore_give %25 : memref<1x64xf32>
          }
          loom.semaphore_give %23 : memref<1x64xf32>
          loom.semaphore_give %32 : memref<1x64x128xf32>
          linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%34, %21 : memref<1x64x128xf32>, memref<1x64xf32>) outs(%31 : memref<1x64x128xf32>) {
          ^bb0(%in: f32, %in_5: f32, %out: f32):
            %40 = arith.divf %in, %in_5 : f32
            linalg.yield %40 : f32
          }
          loom.semaphore_give %21 : memref<1x64xf32>
          loom.semaphore_give %34 : memref<1x64x128xf32>
          %reinterpret_cast_4 = memref.reinterpret_cast %arg3 to offset: [%39], sizes: [1, 64, 128], strides: [524288, 128, 1] : memref<32x4096x128xf32> to memref<1x64x128xf32, strided<[524288, 128, 1], offset: ?>>
          loom.copy %31, %reinterpret_cast_4 src_mem_space @L1 dst_mem_space @DRAM, interconnect : [], broadcast : [1, 1] : memref<1x64x128xf32>, memref<1x64x128xf32, strided<[524288, 128, 1], offset: ?>>
          loom.semaphore_give %31 : memref<1x64x128xf32>
        }
        scf.reduce 
      } {loom.iter_types = [#loom.iter_type<spatial>, #loom.iter_type<spatial>], loom.mapped_to_dims = [@y, @x]}
      return
    }
  }
  module attributes {loom.pass_name = "Materialize"} {
    func.func @attention__d1i1_d0i1__f01__d_v_d__BB1__BM64__BN128(%arg0: memref<32x128x4096xf32>, %arg1: memref<32x4096x128xf32>, %arg2: memref<32x4096x128xf32>, %arg3: memref<32x4096x128xf32>) {
      %c16384 = arith.constant 16384 : index
      %c8192 = arith.constant 8192 : index
      %c524288 = arith.constant 524288 : index
      %c32 = arith.constant 32 : index
      %cst = arith.constant 2.000000e+00 : f32
      %cst_0 = arith.constant 0.12751743074602467 : f64
      %cst_1 = arith.constant 0.000000e+00 : f32
      %cst_2 = arith.constant 1.000000e+00 : f32
      %cst_3 = arith.constant 0xFF800000 : f32
      %c128 = arith.constant 128 : index
      %c0 = arith.constant 0 : index
      %c8 = arith.constant 8 : index
      %c1 = arith.constant 1 : index
      scf.parallel (%arg4, %arg5) = (%c0, %c0) to (%c8, %c8) step (%c1, %c1) {
        scf.for %arg6 = %c0 to %c32 step %c1 {
          %12 = arith.muli %arg4, %c8 overflow<nsw> : index
          %13 = arith.addi %12, %arg5 : index
          %14 = loom.alloc [1, 128, 128] on @L1 : memref<1x128x128xf32>
          %15 = loom.semaphore_take %14 : memref<1x128x128xf32> -> memref<1x128x128xf32>
          %16 = loom.alloc [1, 64, 128] on @L1 : memref<1x64x128xf32>
          %17 = loom.semaphore_take %16 : memref<1x64x128xf32> -> memref<1x64x128xf32>
          %18 = loom.alloc [1, 128, 128] on @L1 : memref<1x128x128xf32>
          %19 = loom.semaphore_take %18 : memref<1x128x128xf32> -> memref<1x128x128xf32>
          %20 = loom.alloc [1, 64] on @L1 : memref<1x64xf32>
          %21 = loom.semaphore_take %20 : memref<1x64xf32> -> memref<1x64xf32>
          %22 = loom.alloc [1, 64] on @L1 : memref<1x64xf32>
          %23 = loom.semaphore_take %22 : memref<1x64xf32> -> memref<1x64xf32>
          %24 = loom.alloc [1, 64] on @L1 : memref<1x64xf32>
          %25 = loom.semaphore_take %24 : memref<1x64xf32> -> memref<1x64xf32>
          %26 = loom.alloc [1, 64] on @L1 : memref<1x64xf32>
          %27 = loom.semaphore_take %26 : memref<1x64xf32> -> memref<1x64xf32>
          %28 = loom.alloc [1, 64] on @L1 : memref<1x64xf32>
          %29 = loom.semaphore_take %28 : memref<1x64xf32> -> memref<1x64xf32>
          %30 = loom.alloc [1, 64, 128] on @L1 : memref<1x64x128xf32>
          %31 = loom.semaphore_take %30 : memref<1x64x128xf32> -> memref<1x64x128xf32>
          %32 = loom.semaphore_take %30 : memref<1x64x128xf32> -> memref<1x64x128xf32>
          %33 = loom.alloc [1, 64, 128] on @L1 : memref<1x64x128xf32>
          %34 = loom.semaphore_take %33 : memref<1x64x128xf32> -> memref<1x64x128xf32>
          %35 = loom.alloc [1, 64, 128] on @L1 : memref<1x64x128xf32>
          %36 = loom.semaphore_take %35 : memref<1x64x128xf32> -> memref<1x64x128xf32>
          %37 = arith.muli %arg6, %c524288 overflow<nsw> : index
          %38 = arith.muli %13, %c8192 : index
          %39 = arith.addi %37, %38 : index
          %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%39], sizes: [1, 64, 128], strides: [524288, 128, 1] : memref<32x4096x128xf32> to memref<1x64x128xf32, strided<[524288, 128, 1], offset: ?>>
          loom.copy %reinterpret_cast, %32 src_mem_space @DRAM dst_mem_space @L1, interconnect : [], broadcast : [1, 1] : memref<1x64x128xf32, strided<[524288, 128, 1], offset: ?>>, memref<1x64x128xf32>
          linalg.fill ins(%cst_1 : f32) outs(%34 : memref<1x64x128xf32>)
          linalg.fill ins(%cst_2 : f32) outs(%21 : memref<1x64xf32>)
          linalg.fill ins(%cst_3 : f32) outs(%23 : memref<1x64xf32>)
          scf.for %arg7 = %c0 to %c32 step %c1 {
            %40 = arith.muli %arg7, %c128 : index
            %41 = arith.addi %37, %40 : index
            %reinterpret_cast_5 = memref.reinterpret_cast %arg0 to offset: [%41], sizes: [1, 128, 128], strides: [524288, 4096, 1] : memref<32x128x4096xf32> to memref<1x128x128xf32, strided<[524288, 4096, 1], offset: ?>>
            loom.copy %reinterpret_cast_5, %19 src_mem_space @DRAM dst_mem_space @L1, interconnect : [@vertical_links], broadcast : [8, 1] : memref<1x128x128xf32, strided<[524288, 4096, 1], offset: ?>>, memref<1x128x128xf32>
            linalg.fill ins(%cst_1 : f32) outs(%17 : memref<1x64x128xf32>)
            linalg.batch_matmul ins(%32, %19 : memref<1x64x128xf32>, memref<1x128x128xf32>) outs(%17 : memref<1x64x128xf32>)
            loom.semaphore_give %19 : memref<1x128x128xf32>
            linalg.fill ins(%cst_3 : f32) outs(%25 : memref<1x64xf32>)
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%17 : memref<1x64x128xf32>) outs(%25 : memref<1x64xf32>) {
            ^bb0(%in: f32, %out: f32):
              %44 = arith.maximumf %in, %out : f32
              linalg.yield %44 : f32
            }
            linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%23, %25 : memref<1x64xf32>, memref<1x64xf32>) outs(%25 : memref<1x64xf32>) {
            ^bb0(%in: f32, %in_7: f32, %out: f32):
              %44 = arith.truncf %cst_0 : f64 to f32
              %45 = arith.mulf %in_7, %44 : f32
              %46 = arith.cmpf ogt, %in, %45 : f32
              %47 = arith.select %46, %in, %45 : f32
              linalg.yield %47 : f32
            }
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%17, %25 : memref<1x64x128xf32>, memref<1x64xf32>) outs(%17 : memref<1x64x128xf32>) {
            ^bb0(%in: f32, %in_7: f32, %out: f32):
              %44 = arith.truncf %cst_0 : f64 to f32
              %45 = arith.mulf %in, %44 : f32
              %46 = arith.subf %45, %in_7 : f32
              %47 = math.powf %cst, %46 : f32
              linalg.yield %47 : f32
            }
            linalg.fill ins(%cst_1 : f32) outs(%27 : memref<1x64xf32>)
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%17 : memref<1x64x128xf32>) outs(%27 : memref<1x64xf32>) {
            ^bb0(%in: f32, %out: f32):
              %44 = arith.addf %in, %out : f32
              linalg.yield %44 : f32
            }
            linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%23, %25 : memref<1x64xf32>, memref<1x64xf32>) outs(%29 : memref<1x64xf32>) {
            ^bb0(%in: f32, %in_7: f32, %out: f32):
              %44 = arith.subf %in, %in_7 : f32
              %45 = math.powf %cst, %44 : f32
              linalg.yield %45 : f32
            }
            linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%21, %29, %27 : memref<1x64xf32>, memref<1x64xf32>, memref<1x64xf32>) outs(%21 : memref<1x64xf32>) {
            ^bb0(%in: f32, %in_7: f32, %in_8: f32, %out: f32):
              %44 = arith.mulf %in, %in_7 : f32
              %45 = arith.addf %44, %in_8 : f32
              linalg.yield %45 : f32
            }
            loom.semaphore_give %27 : memref<1x64xf32>
            %42 = arith.muli %arg7, %c16384 : index
            %43 = arith.addi %37, %42 : index
            %reinterpret_cast_6 = memref.reinterpret_cast %arg1 to offset: [%43], sizes: [1, 128, 128], strides: [524288, 128, 1] : memref<32x4096x128xf32> to memref<1x128x128xf32, strided<[524288, 128, 1], offset: ?>>
            loom.copy %reinterpret_cast_6, %15 src_mem_space @DRAM dst_mem_space @L1, interconnect : [], broadcast : [1, 1] : memref<1x128x128xf32, strided<[524288, 128, 1], offset: ?>>, memref<1x128x128xf32>
            linalg.fill ins(%cst_1 : f32) outs(%36 : memref<1x64x128xf32>)
            linalg.batch_matmul ins(%17, %15 : memref<1x64x128xf32>, memref<1x128x128xf32>) outs(%36 : memref<1x64x128xf32>)
            loom.semaphore_give %15 : memref<1x128x128xf32>
            loom.semaphore_give %17 : memref<1x64x128xf32>
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%36, %34, %29 : memref<1x64x128xf32>, memref<1x64x128xf32>, memref<1x64xf32>) outs(%34 : memref<1x64x128xf32>) {
            ^bb0(%in: f32, %in_7: f32, %in_8: f32, %out: f32):
              %44 = arith.mulf %in_7, %in_8 : f32
              %45 = arith.addf %in, %44 : f32
              linalg.yield %45 : f32
            }
            loom.semaphore_give %29 : memref<1x64xf32>
            loom.semaphore_give %36 : memref<1x64x128xf32>
            linalg.copy ins(%25 : memref<1x64xf32>) outs(%23 : memref<1x64xf32>)
            loom.semaphore_give %25 : memref<1x64xf32>
          }
          loom.semaphore_give %23 : memref<1x64xf32>
          loom.semaphore_give %32 : memref<1x64x128xf32>
          linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%34, %21 : memref<1x64x128xf32>, memref<1x64xf32>) outs(%31 : memref<1x64x128xf32>) {
          ^bb0(%in: f32, %in_5: f32, %out: f32):
            %40 = arith.divf %in, %in_5 : f32
            linalg.yield %40 : f32
          }
          loom.semaphore_give %21 : memref<1x64xf32>
          loom.semaphore_give %34 : memref<1x64x128xf32>
          %reinterpret_cast_4 = memref.reinterpret_cast %arg3 to offset: [%39], sizes: [1, 64, 128], strides: [524288, 128, 1] : memref<32x4096x128xf32> to memref<1x64x128xf32, strided<[524288, 128, 1], offset: ?>>
          loom.copy %31, %reinterpret_cast_4 src_mem_space @L1 dst_mem_space @DRAM, interconnect : [], broadcast : [1, 1] : memref<1x64x128xf32>, memref<1x64x128xf32, strided<[524288, 128, 1], offset: ?>>
          loom.semaphore_give %31 : memref<1x64x128xf32>
        }
        scf.reduce 
      } {loom.iter_types = [#loom.iter_type<spatial>, #loom.iter_type<spatial>], loom.mapped_to_dims = [@y, @x]}
      return
    }
  }
  module attributes {loom.pass_name = "Materialize"} {
    func.func @attention__d1i1_d0i1__f01__d_v_a__BB1__BM64__BN128(%arg0: memref<32x128x4096xf32>, %arg1: memref<32x4096x128xf32>, %arg2: memref<32x4096x128xf32>, %arg3: memref<32x4096x128xf32>) {
      %c16384 = arith.constant 16384 : index
      %c8192 = arith.constant 8192 : index
      %c524288 = arith.constant 524288 : index
      %c32 = arith.constant 32 : index
      %cst = arith.constant 2.000000e+00 : f32
      %cst_0 = arith.constant 0.12751743074602467 : f64
      %cst_1 = arith.constant 0.000000e+00 : f32
      %cst_2 = arith.constant 1.000000e+00 : f32
      %cst_3 = arith.constant 0xFF800000 : f32
      %c128 = arith.constant 128 : index
      %c0 = arith.constant 0 : index
      %c8 = arith.constant 8 : index
      %c1 = arith.constant 1 : index
      scf.parallel (%arg4, %arg5) = (%c0, %c0) to (%c8, %c8) step (%c1, %c1) {
        scf.for %arg6 = %c0 to %c32 step %c1 {
          %12 = arith.muli %arg4, %c8 overflow<nsw> : index
          %13 = arith.addi %12, %arg5 : index
          %14 = loom.alloc [1, 128, 128] on @L1 : memref<1x128x128xf32>
          %15 = loom.semaphore_take %14 : memref<1x128x128xf32> -> memref<1x128x128xf32>
          %16 = loom.alloc [1, 64, 128] on @L1 : memref<1x64x128xf32>
          %17 = loom.semaphore_take %16 : memref<1x64x128xf32> -> memref<1x64x128xf32>
          %18 = loom.alloc [1, 128, 128] on @L1 : memref<1x128x128xf32>
          %19 = loom.semaphore_take %18 : memref<1x128x128xf32> -> memref<1x128x128xf32>
          %20 = loom.alloc [1, 64] on @L1 : memref<1x64xf32>
          %21 = loom.semaphore_take %20 : memref<1x64xf32> -> memref<1x64xf32>
          %22 = loom.alloc [1, 64] on @L1 : memref<1x64xf32>
          %23 = loom.semaphore_take %22 : memref<1x64xf32> -> memref<1x64xf32>
          %24 = loom.alloc [1, 64] on @L1 : memref<1x64xf32>
          %25 = loom.semaphore_take %24 : memref<1x64xf32> -> memref<1x64xf32>
          %26 = loom.alloc [1, 64] on @L1 : memref<1x64xf32>
          %27 = loom.semaphore_take %26 : memref<1x64xf32> -> memref<1x64xf32>
          %28 = loom.alloc [1, 64] on @L1 : memref<1x64xf32>
          %29 = loom.semaphore_take %28 : memref<1x64xf32> -> memref<1x64xf32>
          %30 = loom.alloc [1, 64, 128] on @L1 : memref<1x64x128xf32>
          %31 = loom.semaphore_take %30 : memref<1x64x128xf32> -> memref<1x64x128xf32>
          %32 = loom.semaphore_take %30 : memref<1x64x128xf32> -> memref<1x64x128xf32>
          %33 = loom.alloc [1, 64, 128] on @L1 : memref<1x64x128xf32>
          %34 = loom.semaphore_take %33 : memref<1x64x128xf32> -> memref<1x64x128xf32>
          %35 = loom.alloc [1, 64, 128] on @L1 : memref<1x64x128xf32>
          %36 = loom.semaphore_take %35 : memref<1x64x128xf32> -> memref<1x64x128xf32>
          %37 = arith.muli %arg6, %c524288 overflow<nsw> : index
          %38 = arith.muli %13, %c8192 : index
          %39 = arith.addi %37, %38 : index
          %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%39], sizes: [1, 64, 128], strides: [524288, 128, 1] : memref<32x4096x128xf32> to memref<1x64x128xf32, strided<[524288, 128, 1], offset: ?>>
          loom.copy %reinterpret_cast, %32 src_mem_space @DRAM dst_mem_space @L1, interconnect : [], broadcast : [1, 1] : memref<1x64x128xf32, strided<[524288, 128, 1], offset: ?>>, memref<1x64x128xf32>
          linalg.fill ins(%cst_1 : f32) outs(%34 : memref<1x64x128xf32>)
          linalg.fill ins(%cst_2 : f32) outs(%21 : memref<1x64xf32>)
          linalg.fill ins(%cst_3 : f32) outs(%23 : memref<1x64xf32>)
          scf.for %arg7 = %c0 to %c32 step %c1 {
            %40 = arith.muli %arg7, %c128 : index
            %41 = arith.addi %37, %40 : index
            %reinterpret_cast_5 = memref.reinterpret_cast %arg0 to offset: [%41], sizes: [1, 128, 128], strides: [524288, 4096, 1] : memref<32x128x4096xf32> to memref<1x128x128xf32, strided<[524288, 4096, 1], offset: ?>>
            loom.copy %reinterpret_cast_5, %19 src_mem_space @DRAM dst_mem_space @L1, interconnect : [@vertical_links], broadcast : [8, 1] : memref<1x128x128xf32, strided<[524288, 4096, 1], offset: ?>>, memref<1x128x128xf32>
            linalg.fill ins(%cst_1 : f32) outs(%17 : memref<1x64x128xf32>)
            linalg.batch_matmul ins(%32, %19 : memref<1x64x128xf32>, memref<1x128x128xf32>) outs(%17 : memref<1x64x128xf32>)
            loom.semaphore_give %19 : memref<1x128x128xf32>
            linalg.fill ins(%cst_3 : f32) outs(%25 : memref<1x64xf32>)
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%17 : memref<1x64x128xf32>) outs(%25 : memref<1x64xf32>) {
            ^bb0(%in: f32, %out: f32):
              %44 = arith.maximumf %in, %out : f32
              linalg.yield %44 : f32
            }
            linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%23, %25 : memref<1x64xf32>, memref<1x64xf32>) outs(%25 : memref<1x64xf32>) {
            ^bb0(%in: f32, %in_7: f32, %out: f32):
              %44 = arith.truncf %cst_0 : f64 to f32
              %45 = arith.mulf %in_7, %44 : f32
              %46 = arith.cmpf ogt, %in, %45 : f32
              %47 = arith.select %46, %in, %45 : f32
              linalg.yield %47 : f32
            }
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%17, %25 : memref<1x64x128xf32>, memref<1x64xf32>) outs(%17 : memref<1x64x128xf32>) {
            ^bb0(%in: f32, %in_7: f32, %out: f32):
              %44 = arith.truncf %cst_0 : f64 to f32
              %45 = arith.mulf %in, %44 : f32
              %46 = arith.subf %45, %in_7 : f32
              %47 = math.powf %cst, %46 : f32
              linalg.yield %47 : f32
            }
            linalg.fill ins(%cst_1 : f32) outs(%27 : memref<1x64xf32>)
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%17 : memref<1x64x128xf32>) outs(%27 : memref<1x64xf32>) {
            ^bb0(%in: f32, %out: f32):
              %44 = arith.addf %in, %out : f32
              linalg.yield %44 : f32
            }
            linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%23, %25 : memref<1x64xf32>, memref<1x64xf32>) outs(%29 : memref<1x64xf32>) {
            ^bb0(%in: f32, %in_7: f32, %out: f32):
              %44 = arith.subf %in, %in_7 : f32
              %45 = math.powf %cst, %44 : f32
              linalg.yield %45 : f32
            }
            linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%21, %29, %27 : memref<1x64xf32>, memref<1x64xf32>, memref<1x64xf32>) outs(%21 : memref<1x64xf32>) {
            ^bb0(%in: f32, %in_7: f32, %in_8: f32, %out: f32):
              %44 = arith.mulf %in, %in_7 : f32
              %45 = arith.addf %44, %in_8 : f32
              linalg.yield %45 : f32
            }
            loom.semaphore_give %27 : memref<1x64xf32>
            %42 = arith.muli %arg7, %c16384 : index
            %43 = arith.addi %37, %42 : index
            %reinterpret_cast_6 = memref.reinterpret_cast %arg1 to offset: [%43], sizes: [1, 128, 128], strides: [524288, 128, 1] : memref<32x4096x128xf32> to memref<1x128x128xf32, strided<[524288, 128, 1], offset: ?>>
            loom.copy %reinterpret_cast_6, %15 src_mem_space @DRAM dst_mem_space @L1, interconnect : [@horizontal_links, @vertical_links], broadcast : [8, 8] : memref<1x128x128xf32, strided<[524288, 128, 1], offset: ?>>, memref<1x128x128xf32>
            linalg.fill ins(%cst_1 : f32) outs(%36 : memref<1x64x128xf32>)
            linalg.batch_matmul ins(%17, %15 : memref<1x64x128xf32>, memref<1x128x128xf32>) outs(%36 : memref<1x64x128xf32>)
            loom.semaphore_give %15 : memref<1x128x128xf32>
            loom.semaphore_give %17 : memref<1x64x128xf32>
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%36, %34, %29 : memref<1x64x128xf32>, memref<1x64x128xf32>, memref<1x64xf32>) outs(%34 : memref<1x64x128xf32>) {
            ^bb0(%in: f32, %in_7: f32, %in_8: f32, %out: f32):
              %44 = arith.mulf %in_7, %in_8 : f32
              %45 = arith.addf %in, %44 : f32
              linalg.yield %45 : f32
            }
            loom.semaphore_give %29 : memref<1x64xf32>
            loom.semaphore_give %36 : memref<1x64x128xf32>
            linalg.copy ins(%25 : memref<1x64xf32>) outs(%23 : memref<1x64xf32>)
            loom.semaphore_give %25 : memref<1x64xf32>
          }
          loom.semaphore_give %23 : memref<1x64xf32>
          loom.semaphore_give %32 : memref<1x64x128xf32>
          linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%34, %21 : memref<1x64x128xf32>, memref<1x64xf32>) outs(%31 : memref<1x64x128xf32>) {
          ^bb0(%in: f32, %in_5: f32, %out: f32):
            %40 = arith.divf %in, %in_5 : f32
            linalg.yield %40 : f32
          }
          loom.semaphore_give %21 : memref<1x64xf32>
          loom.semaphore_give %34 : memref<1x64x128xf32>
          %reinterpret_cast_4 = memref.reinterpret_cast %arg3 to offset: [%39], sizes: [1, 64, 128], strides: [524288, 128, 1] : memref<32x4096x128xf32> to memref<1x64x128xf32, strided<[524288, 128, 1], offset: ?>>
          loom.copy %31, %reinterpret_cast_4 src_mem_space @L1 dst_mem_space @DRAM, interconnect : [], broadcast : [1, 1] : memref<1x64x128xf32>, memref<1x64x128xf32, strided<[524288, 128, 1], offset: ?>>
          loom.semaphore_give %31 : memref<1x64x128xf32>
        }
        scf.reduce 
      } {loom.iter_types = [#loom.iter_type<spatial>, #loom.iter_type<spatial>], loom.mapped_to_dims = [@y, @x]}
      return
    }
  }
  module attributes {loom.pass_name = "Materialize"} {
    func.func @attention__d1i1_d0i1__f01__d_v_h__BB1__BM64__BN128(%arg0: memref<32x128x4096xf32>, %arg1: memref<32x4096x128xf32>, %arg2: memref<32x4096x128xf32>, %arg3: memref<32x4096x128xf32>) {
      %c16384 = arith.constant 16384 : index
      %c8192 = arith.constant 8192 : index
      %c524288 = arith.constant 524288 : index
      %c32 = arith.constant 32 : index
      %cst = arith.constant 2.000000e+00 : f32
      %cst_0 = arith.constant 0.12751743074602467 : f64
      %cst_1 = arith.constant 0.000000e+00 : f32
      %cst_2 = arith.constant 1.000000e+00 : f32
      %cst_3 = arith.constant 0xFF800000 : f32
      %c128 = arith.constant 128 : index
      %c0 = arith.constant 0 : index
      %c8 = arith.constant 8 : index
      %c1 = arith.constant 1 : index
      scf.parallel (%arg4, %arg5) = (%c0, %c0) to (%c8, %c8) step (%c1, %c1) {
        scf.for %arg6 = %c0 to %c32 step %c1 {
          %12 = arith.muli %arg4, %c8 overflow<nsw> : index
          %13 = arith.addi %12, %arg5 : index
          %14 = loom.alloc [1, 128, 128] on @L1 : memref<1x128x128xf32>
          %15 = loom.semaphore_take %14 : memref<1x128x128xf32> -> memref<1x128x128xf32>
          %16 = loom.alloc [1, 64, 128] on @L1 : memref<1x64x128xf32>
          %17 = loom.semaphore_take %16 : memref<1x64x128xf32> -> memref<1x64x128xf32>
          %18 = loom.alloc [1, 128, 128] on @L1 : memref<1x128x128xf32>
          %19 = loom.semaphore_take %18 : memref<1x128x128xf32> -> memref<1x128x128xf32>
          %20 = loom.alloc [1, 64] on @L1 : memref<1x64xf32>
          %21 = loom.semaphore_take %20 : memref<1x64xf32> -> memref<1x64xf32>
          %22 = loom.alloc [1, 64] on @L1 : memref<1x64xf32>
          %23 = loom.semaphore_take %22 : memref<1x64xf32> -> memref<1x64xf32>
          %24 = loom.alloc [1, 64] on @L1 : memref<1x64xf32>
          %25 = loom.semaphore_take %24 : memref<1x64xf32> -> memref<1x64xf32>
          %26 = loom.alloc [1, 64] on @L1 : memref<1x64xf32>
          %27 = loom.semaphore_take %26 : memref<1x64xf32> -> memref<1x64xf32>
          %28 = loom.alloc [1, 64] on @L1 : memref<1x64xf32>
          %29 = loom.semaphore_take %28 : memref<1x64xf32> -> memref<1x64xf32>
          %30 = loom.alloc [1, 64, 128] on @L1 : memref<1x64x128xf32>
          %31 = loom.semaphore_take %30 : memref<1x64x128xf32> -> memref<1x64x128xf32>
          %32 = loom.semaphore_take %30 : memref<1x64x128xf32> -> memref<1x64x128xf32>
          %33 = loom.alloc [1, 64, 128] on @L1 : memref<1x64x128xf32>
          %34 = loom.semaphore_take %33 : memref<1x64x128xf32> -> memref<1x64x128xf32>
          %35 = loom.alloc [1, 64, 128] on @L1 : memref<1x64x128xf32>
          %36 = loom.semaphore_take %35 : memref<1x64x128xf32> -> memref<1x64x128xf32>
          %37 = arith.muli %arg6, %c524288 overflow<nsw> : index
          %38 = arith.muli %13, %c8192 : index
          %39 = arith.addi %37, %38 : index
          %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%39], sizes: [1, 64, 128], strides: [524288, 128, 1] : memref<32x4096x128xf32> to memref<1x64x128xf32, strided<[524288, 128, 1], offset: ?>>
          loom.copy %reinterpret_cast, %32 src_mem_space @DRAM dst_mem_space @L1, interconnect : [], broadcast : [1, 1] : memref<1x64x128xf32, strided<[524288, 128, 1], offset: ?>>, memref<1x64x128xf32>
          linalg.fill ins(%cst_1 : f32) outs(%34 : memref<1x64x128xf32>)
          linalg.fill ins(%cst_2 : f32) outs(%21 : memref<1x64xf32>)
          linalg.fill ins(%cst_3 : f32) outs(%23 : memref<1x64xf32>)
          scf.for %arg7 = %c0 to %c32 step %c1 {
            %40 = arith.muli %arg7, %c128 : index
            %41 = arith.addi %37, %40 : index
            %reinterpret_cast_5 = memref.reinterpret_cast %arg0 to offset: [%41], sizes: [1, 128, 128], strides: [524288, 4096, 1] : memref<32x128x4096xf32> to memref<1x128x128xf32, strided<[524288, 4096, 1], offset: ?>>
            loom.copy %reinterpret_cast_5, %19 src_mem_space @DRAM dst_mem_space @L1, interconnect : [@vertical_links], broadcast : [8, 1] : memref<1x128x128xf32, strided<[524288, 4096, 1], offset: ?>>, memref<1x128x128xf32>
            linalg.fill ins(%cst_1 : f32) outs(%17 : memref<1x64x128xf32>)
            linalg.batch_matmul ins(%32, %19 : memref<1x64x128xf32>, memref<1x128x128xf32>) outs(%17 : memref<1x64x128xf32>)
            loom.semaphore_give %19 : memref<1x128x128xf32>
            linalg.fill ins(%cst_3 : f32) outs(%25 : memref<1x64xf32>)
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%17 : memref<1x64x128xf32>) outs(%25 : memref<1x64xf32>) {
            ^bb0(%in: f32, %out: f32):
              %44 = arith.maximumf %in, %out : f32
              linalg.yield %44 : f32
            }
            linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%23, %25 : memref<1x64xf32>, memref<1x64xf32>) outs(%25 : memref<1x64xf32>) {
            ^bb0(%in: f32, %in_7: f32, %out: f32):
              %44 = arith.truncf %cst_0 : f64 to f32
              %45 = arith.mulf %in_7, %44 : f32
              %46 = arith.cmpf ogt, %in, %45 : f32
              %47 = arith.select %46, %in, %45 : f32
              linalg.yield %47 : f32
            }
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%17, %25 : memref<1x64x128xf32>, memref<1x64xf32>) outs(%17 : memref<1x64x128xf32>) {
            ^bb0(%in: f32, %in_7: f32, %out: f32):
              %44 = arith.truncf %cst_0 : f64 to f32
              %45 = arith.mulf %in, %44 : f32
              %46 = arith.subf %45, %in_7 : f32
              %47 = math.powf %cst, %46 : f32
              linalg.yield %47 : f32
            }
            linalg.fill ins(%cst_1 : f32) outs(%27 : memref<1x64xf32>)
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%17 : memref<1x64x128xf32>) outs(%27 : memref<1x64xf32>) {
            ^bb0(%in: f32, %out: f32):
              %44 = arith.addf %in, %out : f32
              linalg.yield %44 : f32
            }
            linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%23, %25 : memref<1x64xf32>, memref<1x64xf32>) outs(%29 : memref<1x64xf32>) {
            ^bb0(%in: f32, %in_7: f32, %out: f32):
              %44 = arith.subf %in, %in_7 : f32
              %45 = math.powf %cst, %44 : f32
              linalg.yield %45 : f32
            }
            linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%21, %29, %27 : memref<1x64xf32>, memref<1x64xf32>, memref<1x64xf32>) outs(%21 : memref<1x64xf32>) {
            ^bb0(%in: f32, %in_7: f32, %in_8: f32, %out: f32):
              %44 = arith.mulf %in, %in_7 : f32
              %45 = arith.addf %44, %in_8 : f32
              linalg.yield %45 : f32
            }
            loom.semaphore_give %27 : memref<1x64xf32>
            %42 = arith.muli %arg7, %c16384 : index
            %43 = arith.addi %37, %42 : index
            %reinterpret_cast_6 = memref.reinterpret_cast %arg1 to offset: [%43], sizes: [1, 128, 128], strides: [524288, 128, 1] : memref<32x4096x128xf32> to memref<1x128x128xf32, strided<[524288, 128, 1], offset: ?>>
            loom.copy %reinterpret_cast_6, %15 src_mem_space @DRAM dst_mem_space @L1, interconnect : [@horizontal_links], broadcast : [1, 8] : memref<1x128x128xf32, strided<[524288, 128, 1], offset: ?>>, memref<1x128x128xf32>
            linalg.fill ins(%cst_1 : f32) outs(%36 : memref<1x64x128xf32>)
            linalg.batch_matmul ins(%17, %15 : memref<1x64x128xf32>, memref<1x128x128xf32>) outs(%36 : memref<1x64x128xf32>)
            loom.semaphore_give %15 : memref<1x128x128xf32>
            loom.semaphore_give %17 : memref<1x64x128xf32>
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%36, %34, %29 : memref<1x64x128xf32>, memref<1x64x128xf32>, memref<1x64xf32>) outs(%34 : memref<1x64x128xf32>) {
            ^bb0(%in: f32, %in_7: f32, %in_8: f32, %out: f32):
              %44 = arith.mulf %in_7, %in_8 : f32
              %45 = arith.addf %in, %44 : f32
              linalg.yield %45 : f32
            }
            loom.semaphore_give %29 : memref<1x64xf32>
            loom.semaphore_give %36 : memref<1x64x128xf32>
            linalg.copy ins(%25 : memref<1x64xf32>) outs(%23 : memref<1x64xf32>)
            loom.semaphore_give %25 : memref<1x64xf32>
          }
          loom.semaphore_give %23 : memref<1x64xf32>
          loom.semaphore_give %32 : memref<1x64x128xf32>
          linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%34, %21 : memref<1x64x128xf32>, memref<1x64xf32>) outs(%31 : memref<1x64x128xf32>) {
          ^bb0(%in: f32, %in_5: f32, %out: f32):
            %40 = arith.divf %in, %in_5 : f32
            linalg.yield %40 : f32
          }
          loom.semaphore_give %21 : memref<1x64xf32>
          loom.semaphore_give %34 : memref<1x64x128xf32>
          %reinterpret_cast_4 = memref.reinterpret_cast %arg3 to offset: [%39], sizes: [1, 64, 128], strides: [524288, 128, 1] : memref<32x4096x128xf32> to memref<1x64x128xf32, strided<[524288, 128, 1], offset: ?>>
          loom.copy %31, %reinterpret_cast_4 src_mem_space @L1 dst_mem_space @DRAM, interconnect : [], broadcast : [1, 1] : memref<1x64x128xf32>, memref<1x64x128xf32, strided<[524288, 128, 1], offset: ?>>
          loom.semaphore_give %31 : memref<1x64x128xf32>
        }
        scf.reduce 
      } {loom.iter_types = [#loom.iter_type<spatial>, #loom.iter_type<spatial>], loom.mapped_to_dims = [@y, @x]}
      return
    }
  }
  module attributes {loom.pass_name = "Materialize"} {
    func.func @attention__d1i1_d0i1__f01__d_v_v__BB1__BM64__BN128(%arg0: memref<32x128x4096xf32>, %arg1: memref<32x4096x128xf32>, %arg2: memref<32x4096x128xf32>, %arg3: memref<32x4096x128xf32>) {
      %c16384 = arith.constant 16384 : index
      %c8192 = arith.constant 8192 : index
      %c524288 = arith.constant 524288 : index
      %c32 = arith.constant 32 : index
      %cst = arith.constant 2.000000e+00 : f32
      %cst_0 = arith.constant 0.12751743074602467 : f64
      %cst_1 = arith.constant 0.000000e+00 : f32
      %cst_2 = arith.constant 1.000000e+00 : f32
      %cst_3 = arith.constant 0xFF800000 : f32
      %c128 = arith.constant 128 : index
      %c0 = arith.constant 0 : index
      %c8 = arith.constant 8 : index
      %c1 = arith.constant 1 : index
      scf.parallel (%arg4, %arg5) = (%c0, %c0) to (%c8, %c8) step (%c1, %c1) {
        scf.for %arg6 = %c0 to %c32 step %c1 {
          %12 = arith.muli %arg4, %c8 overflow<nsw> : index
          %13 = arith.addi %12, %arg5 : index
          %14 = loom.alloc [1, 128, 128] on @L1 : memref<1x128x128xf32>
          %15 = loom.semaphore_take %14 : memref<1x128x128xf32> -> memref<1x128x128xf32>
          %16 = loom.alloc [1, 64, 128] on @L1 : memref<1x64x128xf32>
          %17 = loom.semaphore_take %16 : memref<1x64x128xf32> -> memref<1x64x128xf32>
          %18 = loom.alloc [1, 128, 128] on @L1 : memref<1x128x128xf32>
          %19 = loom.semaphore_take %18 : memref<1x128x128xf32> -> memref<1x128x128xf32>
          %20 = loom.alloc [1, 64] on @L1 : memref<1x64xf32>
          %21 = loom.semaphore_take %20 : memref<1x64xf32> -> memref<1x64xf32>
          %22 = loom.alloc [1, 64] on @L1 : memref<1x64xf32>
          %23 = loom.semaphore_take %22 : memref<1x64xf32> -> memref<1x64xf32>
          %24 = loom.alloc [1, 64] on @L1 : memref<1x64xf32>
          %25 = loom.semaphore_take %24 : memref<1x64xf32> -> memref<1x64xf32>
          %26 = loom.alloc [1, 64] on @L1 : memref<1x64xf32>
          %27 = loom.semaphore_take %26 : memref<1x64xf32> -> memref<1x64xf32>
          %28 = loom.alloc [1, 64] on @L1 : memref<1x64xf32>
          %29 = loom.semaphore_take %28 : memref<1x64xf32> -> memref<1x64xf32>
          %30 = loom.alloc [1, 64, 128] on @L1 : memref<1x64x128xf32>
          %31 = loom.semaphore_take %30 : memref<1x64x128xf32> -> memref<1x64x128xf32>
          %32 = loom.semaphore_take %30 : memref<1x64x128xf32> -> memref<1x64x128xf32>
          %33 = loom.alloc [1, 64, 128] on @L1 : memref<1x64x128xf32>
          %34 = loom.semaphore_take %33 : memref<1x64x128xf32> -> memref<1x64x128xf32>
          %35 = loom.alloc [1, 64, 128] on @L1 : memref<1x64x128xf32>
          %36 = loom.semaphore_take %35 : memref<1x64x128xf32> -> memref<1x64x128xf32>
          %37 = arith.muli %arg6, %c524288 overflow<nsw> : index
          %38 = arith.muli %13, %c8192 : index
          %39 = arith.addi %37, %38 : index
          %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%39], sizes: [1, 64, 128], strides: [524288, 128, 1] : memref<32x4096x128xf32> to memref<1x64x128xf32, strided<[524288, 128, 1], offset: ?>>
          loom.copy %reinterpret_cast, %32 src_mem_space @DRAM dst_mem_space @L1, interconnect : [], broadcast : [1, 1] : memref<1x64x128xf32, strided<[524288, 128, 1], offset: ?>>, memref<1x64x128xf32>
          linalg.fill ins(%cst_1 : f32) outs(%34 : memref<1x64x128xf32>)
          linalg.fill ins(%cst_2 : f32) outs(%21 : memref<1x64xf32>)
          linalg.fill ins(%cst_3 : f32) outs(%23 : memref<1x64xf32>)
          scf.for %arg7 = %c0 to %c32 step %c1 {
            %40 = arith.muli %arg7, %c128 : index
            %41 = arith.addi %37, %40 : index
            %reinterpret_cast_5 = memref.reinterpret_cast %arg0 to offset: [%41], sizes: [1, 128, 128], strides: [524288, 4096, 1] : memref<32x128x4096xf32> to memref<1x128x128xf32, strided<[524288, 4096, 1], offset: ?>>
            loom.copy %reinterpret_cast_5, %19 src_mem_space @DRAM dst_mem_space @L1, interconnect : [@vertical_links], broadcast : [8, 1] : memref<1x128x128xf32, strided<[524288, 4096, 1], offset: ?>>, memref<1x128x128xf32>
            linalg.fill ins(%cst_1 : f32) outs(%17 : memref<1x64x128xf32>)
            linalg.batch_matmul ins(%32, %19 : memref<1x64x128xf32>, memref<1x128x128xf32>) outs(%17 : memref<1x64x128xf32>)
            loom.semaphore_give %19 : memref<1x128x128xf32>
            linalg.fill ins(%cst_3 : f32) outs(%25 : memref<1x64xf32>)
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%17 : memref<1x64x128xf32>) outs(%25 : memref<1x64xf32>) {
            ^bb0(%in: f32, %out: f32):
              %44 = arith.maximumf %in, %out : f32
              linalg.yield %44 : f32
            }
            linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%23, %25 : memref<1x64xf32>, memref<1x64xf32>) outs(%25 : memref<1x64xf32>) {
            ^bb0(%in: f32, %in_7: f32, %out: f32):
              %44 = arith.truncf %cst_0 : f64 to f32
              %45 = arith.mulf %in_7, %44 : f32
              %46 = arith.cmpf ogt, %in, %45 : f32
              %47 = arith.select %46, %in, %45 : f32
              linalg.yield %47 : f32
            }
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%17, %25 : memref<1x64x128xf32>, memref<1x64xf32>) outs(%17 : memref<1x64x128xf32>) {
            ^bb0(%in: f32, %in_7: f32, %out: f32):
              %44 = arith.truncf %cst_0 : f64 to f32
              %45 = arith.mulf %in, %44 : f32
              %46 = arith.subf %45, %in_7 : f32
              %47 = math.powf %cst, %46 : f32
              linalg.yield %47 : f32
            }
            linalg.fill ins(%cst_1 : f32) outs(%27 : memref<1x64xf32>)
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%17 : memref<1x64x128xf32>) outs(%27 : memref<1x64xf32>) {
            ^bb0(%in: f32, %out: f32):
              %44 = arith.addf %in, %out : f32
              linalg.yield %44 : f32
            }
            linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%23, %25 : memref<1x64xf32>, memref<1x64xf32>) outs(%29 : memref<1x64xf32>) {
            ^bb0(%in: f32, %in_7: f32, %out: f32):
              %44 = arith.subf %in, %in_7 : f32
              %45 = math.powf %cst, %44 : f32
              linalg.yield %45 : f32
            }
            linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%21, %29, %27 : memref<1x64xf32>, memref<1x64xf32>, memref<1x64xf32>) outs(%21 : memref<1x64xf32>) {
            ^bb0(%in: f32, %in_7: f32, %in_8: f32, %out: f32):
              %44 = arith.mulf %in, %in_7 : f32
              %45 = arith.addf %44, %in_8 : f32
              linalg.yield %45 : f32
            }
            loom.semaphore_give %27 : memref<1x64xf32>
            %42 = arith.muli %arg7, %c16384 : index
            %43 = arith.addi %37, %42 : index
            %reinterpret_cast_6 = memref.reinterpret_cast %arg1 to offset: [%43], sizes: [1, 128, 128], strides: [524288, 128, 1] : memref<32x4096x128xf32> to memref<1x128x128xf32, strided<[524288, 128, 1], offset: ?>>
            loom.copy %reinterpret_cast_6, %15 src_mem_space @DRAM dst_mem_space @L1, interconnect : [@vertical_links], broadcast : [8, 1] : memref<1x128x128xf32, strided<[524288, 128, 1], offset: ?>>, memref<1x128x128xf32>
            linalg.fill ins(%cst_1 : f32) outs(%36 : memref<1x64x128xf32>)
            linalg.batch_matmul ins(%17, %15 : memref<1x64x128xf32>, memref<1x128x128xf32>) outs(%36 : memref<1x64x128xf32>)
            loom.semaphore_give %15 : memref<1x128x128xf32>
            loom.semaphore_give %17 : memref<1x64x128xf32>
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%36, %34, %29 : memref<1x64x128xf32>, memref<1x64x128xf32>, memref<1x64xf32>) outs(%34 : memref<1x64x128xf32>) {
            ^bb0(%in: f32, %in_7: f32, %in_8: f32, %out: f32):
              %44 = arith.mulf %in_7, %in_8 : f32
              %45 = arith.addf %in, %44 : f32
              linalg.yield %45 : f32
            }
            loom.semaphore_give %29 : memref<1x64xf32>
            loom.semaphore_give %36 : memref<1x64x128xf32>
            linalg.copy ins(%25 : memref<1x64xf32>) outs(%23 : memref<1x64xf32>)
            loom.semaphore_give %25 : memref<1x64xf32>
          }
          loom.semaphore_give %23 : memref<1x64xf32>
          loom.semaphore_give %32 : memref<1x64x128xf32>
          linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%34, %21 : memref<1x64x128xf32>, memref<1x64xf32>) outs(%31 : memref<1x64x128xf32>) {
          ^bb0(%in: f32, %in_5: f32, %out: f32):
            %40 = arith.divf %in, %in_5 : f32
            linalg.yield %40 : f32
          }
          loom.semaphore_give %21 : memref<1x64xf32>
          loom.semaphore_give %34 : memref<1x64x128xf32>
          %reinterpret_cast_4 = memref.reinterpret_cast %arg3 to offset: [%39], sizes: [1, 64, 128], strides: [524288, 128, 1] : memref<32x4096x128xf32> to memref<1x64x128xf32, strided<[524288, 128, 1], offset: ?>>
          loom.copy %31, %reinterpret_cast_4 src_mem_space @L1 dst_mem_space @DRAM, interconnect : [], broadcast : [1, 1] : memref<1x64x128xf32>, memref<1x64x128xf32, strided<[524288, 128, 1], offset: ?>>
          loom.semaphore_give %31 : memref<1x64x128xf32>
        }
        scf.reduce 
      } {loom.iter_types = [#loom.iter_type<spatial>, #loom.iter_type<spatial>], loom.mapped_to_dims = [@y, @x]}
      return
    }
  }
  module attributes {loom.pass_name = "Materialize"} {
    func.func @attention__d1i1_d0i1__f10__d_d_d__BB1__BM64__BN128(%arg0: memref<32x128x4096xf32>, %arg1: memref<32x4096x128xf32>, %arg2: memref<32x4096x128xf32>, %arg3: memref<32x4096x128xf32>) {
      %c16384 = arith.constant 16384 : index
      %c8192 = arith.constant 8192 : index
      %c524288 = arith.constant 524288 : index
      %c32 = arith.constant 32 : index
      %cst = arith.constant 2.000000e+00 : f32
      %cst_0 = arith.constant 0.12751743074602467 : f64
      %cst_1 = arith.constant 0.000000e+00 : f32
      %cst_2 = arith.constant 1.000000e+00 : f32
      %cst_3 = arith.constant 0xFF800000 : f32
      %c128 = arith.constant 128 : index
      %c0 = arith.constant 0 : index
      %c8 = arith.constant 8 : index
      %c1 = arith.constant 1 : index
      scf.parallel (%arg4, %arg5) = (%c0, %c0) to (%c8, %c8) step (%c1, %c1) {
        scf.for %arg6 = %c0 to %c32 step %c1 {
          %12 = arith.muli %arg4, %c8 overflow<nsw> : index
          %13 = arith.addi %12, %arg5 : index
          %14 = loom.alloc [1, 128, 128] on @L1 : memref<1x128x128xf32>
          %15 = loom.semaphore_take %14 : memref<1x128x128xf32> -> memref<1x128x128xf32>
          %16 = loom.alloc [1, 64, 128] on @L1 : memref<1x64x128xf32>
          %17 = loom.semaphore_take %16 : memref<1x64x128xf32> -> memref<1x64x128xf32>
          %18 = loom.alloc [1, 128, 128] on @L1 : memref<1x128x128xf32>
          %19 = loom.semaphore_take %18 : memref<1x128x128xf32> -> memref<1x128x128xf32>
          %20 = loom.alloc [1, 64] on @L1 : memref<1x64xf32>
          %21 = loom.semaphore_take %20 : memref<1x64xf32> -> memref<1x64xf32>
          %22 = loom.alloc [1, 64] on @L1 : memref<1x64xf32>
          %23 = loom.semaphore_take %22 : memref<1x64xf32> -> memref<1x64xf32>
          %24 = loom.alloc [1, 64] on @L1 : memref<1x64xf32>
          %25 = loom.semaphore_take %24 : memref<1x64xf32> -> memref<1x64xf32>
          %26 = loom.alloc [1, 64] on @L1 : memref<1x64xf32>
          %27 = loom.semaphore_take %26 : memref<1x64xf32> -> memref<1x64xf32>
          %28 = loom.alloc [1, 64] on @L1 : memref<1x64xf32>
          %29 = loom.semaphore_take %28 : memref<1x64xf32> -> memref<1x64xf32>
          %30 = loom.alloc [1, 64, 128] on @L1 : memref<1x64x128xf32>
          %31 = loom.semaphore_take %30 : memref<1x64x128xf32> -> memref<1x64x128xf32>
          %32 = loom.semaphore_take %30 : memref<1x64x128xf32> -> memref<1x64x128xf32>
          %33 = loom.alloc [1, 64, 128] on @L1 : memref<1x64x128xf32>
          %34 = loom.semaphore_take %33 : memref<1x64x128xf32> -> memref<1x64x128xf32>
          %35 = loom.alloc [1, 64, 128] on @L1 : memref<1x64x128xf32>
          %36 = loom.semaphore_take %35 : memref<1x64x128xf32> -> memref<1x64x128xf32>
          %37 = arith.muli %arg6, %c524288 overflow<nsw> : index
          %38 = arith.muli %13, %c8192 : index
          %39 = arith.addi %37, %38 : index
          %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%39], sizes: [1, 64, 128], strides: [524288, 128, 1] : memref<32x4096x128xf32> to memref<1x64x128xf32, strided<[524288, 128, 1], offset: ?>>
          loom.copy %reinterpret_cast, %32 src_mem_space @DRAM dst_mem_space @L1, interconnect : [], broadcast : [1, 1] : memref<1x64x128xf32, strided<[524288, 128, 1], offset: ?>>, memref<1x64x128xf32>
          linalg.fill ins(%cst_1 : f32) outs(%34 : memref<1x64x128xf32>)
          linalg.fill ins(%cst_2 : f32) outs(%21 : memref<1x64xf32>)
          linalg.fill ins(%cst_3 : f32) outs(%23 : memref<1x64xf32>)
          scf.for %arg7 = %c0 to %c32 step %c1 {
            %40 = arith.muli %arg7, %c128 : index
            %41 = arith.addi %37, %40 : index
            %reinterpret_cast_5 = memref.reinterpret_cast %arg0 to offset: [%41], sizes: [1, 128, 128], strides: [524288, 4096, 1] : memref<32x128x4096xf32> to memref<1x128x128xf32, strided<[524288, 4096, 1], offset: ?>>
            loom.copy %reinterpret_cast_5, %19 src_mem_space @DRAM dst_mem_space @L1, interconnect : [], broadcast : [1, 1] : memref<1x128x128xf32, strided<[524288, 4096, 1], offset: ?>>, memref<1x128x128xf32>
            linalg.fill ins(%cst_1 : f32) outs(%17 : memref<1x64x128xf32>)
            linalg.batch_matmul ins(%32, %19 : memref<1x64x128xf32>, memref<1x128x128xf32>) outs(%17 : memref<1x64x128xf32>)
            loom.semaphore_give %19 : memref<1x128x128xf32>
            linalg.fill ins(%cst_3 : f32) outs(%25 : memref<1x64xf32>)
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%17 : memref<1x64x128xf32>) outs(%25 : memref<1x64xf32>) {
            ^bb0(%in: f32, %out: f32):
              %44 = arith.maximumf %in, %out : f32
              linalg.yield %44 : f32
            }
            linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%23, %25 : memref<1x64xf32>, memref<1x64xf32>) outs(%25 : memref<1x64xf32>) {
            ^bb0(%in: f32, %in_7: f32, %out: f32):
              %44 = arith.truncf %cst_0 : f64 to f32
              %45 = arith.mulf %in_7, %44 : f32
              %46 = arith.cmpf ogt, %in, %45 : f32
              %47 = arith.select %46, %in, %45 : f32
              linalg.yield %47 : f32
            }
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%17, %25 : memref<1x64x128xf32>, memref<1x64xf32>) outs(%17 : memref<1x64x128xf32>) {
            ^bb0(%in: f32, %in_7: f32, %out: f32):
              %44 = arith.truncf %cst_0 : f64 to f32
              %45 = arith.mulf %in, %44 : f32
              %46 = arith.subf %45, %in_7 : f32
              %47 = math.powf %cst, %46 : f32
              linalg.yield %47 : f32
            }
            linalg.fill ins(%cst_1 : f32) outs(%27 : memref<1x64xf32>)
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%17 : memref<1x64x128xf32>) outs(%27 : memref<1x64xf32>) {
            ^bb0(%in: f32, %out: f32):
              %44 = arith.addf %in, %out : f32
              linalg.yield %44 : f32
            }
            linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%23, %25 : memref<1x64xf32>, memref<1x64xf32>) outs(%29 : memref<1x64xf32>) {
            ^bb0(%in: f32, %in_7: f32, %out: f32):
              %44 = arith.subf %in, %in_7 : f32
              %45 = math.powf %cst, %44 : f32
              linalg.yield %45 : f32
            }
            linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%21, %29, %27 : memref<1x64xf32>, memref<1x64xf32>, memref<1x64xf32>) outs(%21 : memref<1x64xf32>) {
            ^bb0(%in: f32, %in_7: f32, %in_8: f32, %out: f32):
              %44 = arith.mulf %in, %in_7 : f32
              %45 = arith.addf %44, %in_8 : f32
              linalg.yield %45 : f32
            }
            loom.semaphore_give %27 : memref<1x64xf32>
            %42 = arith.muli %arg7, %c16384 : index
            %43 = arith.addi %37, %42 : index
            %reinterpret_cast_6 = memref.reinterpret_cast %arg1 to offset: [%43], sizes: [1, 128, 128], strides: [524288, 128, 1] : memref<32x4096x128xf32> to memref<1x128x128xf32, strided<[524288, 128, 1], offset: ?>>
            loom.copy %reinterpret_cast_6, %15 src_mem_space @DRAM dst_mem_space @L1, interconnect : [], broadcast : [1, 1] : memref<1x128x128xf32, strided<[524288, 128, 1], offset: ?>>, memref<1x128x128xf32>
            linalg.fill ins(%cst_1 : f32) outs(%36 : memref<1x64x128xf32>)
            linalg.batch_matmul ins(%17, %15 : memref<1x64x128xf32>, memref<1x128x128xf32>) outs(%36 : memref<1x64x128xf32>)
            loom.semaphore_give %15 : memref<1x128x128xf32>
            loom.semaphore_give %17 : memref<1x64x128xf32>
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%36, %34, %29 : memref<1x64x128xf32>, memref<1x64x128xf32>, memref<1x64xf32>) outs(%34 : memref<1x64x128xf32>) {
            ^bb0(%in: f32, %in_7: f32, %in_8: f32, %out: f32):
              %44 = arith.mulf %in_7, %in_8 : f32
              %45 = arith.addf %in, %44 : f32
              linalg.yield %45 : f32
            }
            loom.semaphore_give %29 : memref<1x64xf32>
            loom.semaphore_give %36 : memref<1x64x128xf32>
            linalg.copy ins(%25 : memref<1x64xf32>) outs(%23 : memref<1x64xf32>)
            loom.semaphore_give %25 : memref<1x64xf32>
          }
          loom.semaphore_give %23 : memref<1x64xf32>
          loom.semaphore_give %32 : memref<1x64x128xf32>
          linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%34, %21 : memref<1x64x128xf32>, memref<1x64xf32>) outs(%31 : memref<1x64x128xf32>) {
          ^bb0(%in: f32, %in_5: f32, %out: f32):
            %40 = arith.divf %in, %in_5 : f32
            linalg.yield %40 : f32
          }
          loom.semaphore_give %21 : memref<1x64xf32>
          loom.semaphore_give %34 : memref<1x64x128xf32>
          %reinterpret_cast_4 = memref.reinterpret_cast %arg3 to offset: [%39], sizes: [1, 64, 128], strides: [524288, 128, 1] : memref<32x4096x128xf32> to memref<1x64x128xf32, strided<[524288, 128, 1], offset: ?>>
          loom.copy %31, %reinterpret_cast_4 src_mem_space @L1 dst_mem_space @DRAM, interconnect : [], broadcast : [1, 1] : memref<1x64x128xf32>, memref<1x64x128xf32, strided<[524288, 128, 1], offset: ?>>
          loom.semaphore_give %31 : memref<1x64x128xf32>
        }
        scf.reduce 
      } {loom.iter_types = [#loom.iter_type<spatial>, #loom.iter_type<spatial>], loom.mapped_to_dims = [@y, @x]}
      return
    }
  }
  module attributes {loom.pass_name = "Materialize"} {
    func.func @attention__d1i1_d0i1__f10__d_d_a__BB1__BM64__BN128(%arg0: memref<32x128x4096xf32>, %arg1: memref<32x4096x128xf32>, %arg2: memref<32x4096x128xf32>, %arg3: memref<32x4096x128xf32>) {
      %c16384 = arith.constant 16384 : index
      %c8192 = arith.constant 8192 : index
      %c524288 = arith.constant 524288 : index
      %c32 = arith.constant 32 : index
      %cst = arith.constant 2.000000e+00 : f32
      %cst_0 = arith.constant 0.12751743074602467 : f64
      %cst_1 = arith.constant 0.000000e+00 : f32
      %cst_2 = arith.constant 1.000000e+00 : f32
      %cst_3 = arith.constant 0xFF800000 : f32
      %c128 = arith.constant 128 : index
      %c0 = arith.constant 0 : index
      %c8 = arith.constant 8 : index
      %c1 = arith.constant 1 : index
      scf.parallel (%arg4, %arg5) = (%c0, %c0) to (%c8, %c8) step (%c1, %c1) {
        scf.for %arg6 = %c0 to %c32 step %c1 {
          %12 = arith.muli %arg4, %c8 overflow<nsw> : index
          %13 = arith.addi %12, %arg5 : index
          %14 = loom.alloc [1, 128, 128] on @L1 : memref<1x128x128xf32>
          %15 = loom.semaphore_take %14 : memref<1x128x128xf32> -> memref<1x128x128xf32>
          %16 = loom.alloc [1, 64, 128] on @L1 : memref<1x64x128xf32>
          %17 = loom.semaphore_take %16 : memref<1x64x128xf32> -> memref<1x64x128xf32>
          %18 = loom.alloc [1, 128, 128] on @L1 : memref<1x128x128xf32>
          %19 = loom.semaphore_take %18 : memref<1x128x128xf32> -> memref<1x128x128xf32>
          %20 = loom.alloc [1, 64] on @L1 : memref<1x64xf32>
          %21 = loom.semaphore_take %20 : memref<1x64xf32> -> memref<1x64xf32>
          %22 = loom.alloc [1, 64] on @L1 : memref<1x64xf32>
          %23 = loom.semaphore_take %22 : memref<1x64xf32> -> memref<1x64xf32>
          %24 = loom.alloc [1, 64] on @L1 : memref<1x64xf32>
          %25 = loom.semaphore_take %24 : memref<1x64xf32> -> memref<1x64xf32>
          %26 = loom.alloc [1, 64] on @L1 : memref<1x64xf32>
          %27 = loom.semaphore_take %26 : memref<1x64xf32> -> memref<1x64xf32>
          %28 = loom.alloc [1, 64] on @L1 : memref<1x64xf32>
          %29 = loom.semaphore_take %28 : memref<1x64xf32> -> memref<1x64xf32>
          %30 = loom.alloc [1, 64, 128] on @L1 : memref<1x64x128xf32>
          %31 = loom.semaphore_take %30 : memref<1x64x128xf32> -> memref<1x64x128xf32>
          %32 = loom.semaphore_take %30 : memref<1x64x128xf32> -> memref<1x64x128xf32>
          %33 = loom.alloc [1, 64, 128] on @L1 : memref<1x64x128xf32>
          %34 = loom.semaphore_take %33 : memref<1x64x128xf32> -> memref<1x64x128xf32>
          %35 = loom.alloc [1, 64, 128] on @L1 : memref<1x64x128xf32>
          %36 = loom.semaphore_take %35 : memref<1x64x128xf32> -> memref<1x64x128xf32>
          %37 = arith.muli %arg6, %c524288 overflow<nsw> : index
          %38 = arith.muli %13, %c8192 : index
          %39 = arith.addi %37, %38 : index
          %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%39], sizes: [1, 64, 128], strides: [524288, 128, 1] : memref<32x4096x128xf32> to memref<1x64x128xf32, strided<[524288, 128, 1], offset: ?>>
          loom.copy %reinterpret_cast, %32 src_mem_space @DRAM dst_mem_space @L1, interconnect : [], broadcast : [1, 1] : memref<1x64x128xf32, strided<[524288, 128, 1], offset: ?>>, memref<1x64x128xf32>
          linalg.fill ins(%cst_1 : f32) outs(%34 : memref<1x64x128xf32>)
          linalg.fill ins(%cst_2 : f32) outs(%21 : memref<1x64xf32>)
          linalg.fill ins(%cst_3 : f32) outs(%23 : memref<1x64xf32>)
          scf.for %arg7 = %c0 to %c32 step %c1 {
            %40 = arith.muli %arg7, %c128 : index
            %41 = arith.addi %37, %40 : index
            %reinterpret_cast_5 = memref.reinterpret_cast %arg0 to offset: [%41], sizes: [1, 128, 128], strides: [524288, 4096, 1] : memref<32x128x4096xf32> to memref<1x128x128xf32, strided<[524288, 4096, 1], offset: ?>>
            loom.copy %reinterpret_cast_5, %19 src_mem_space @DRAM dst_mem_space @L1, interconnect : [], broadcast : [1, 1] : memref<1x128x128xf32, strided<[524288, 4096, 1], offset: ?>>, memref<1x128x128xf32>
            linalg.fill ins(%cst_1 : f32) outs(%17 : memref<1x64x128xf32>)
            linalg.batch_matmul ins(%32, %19 : memref<1x64x128xf32>, memref<1x128x128xf32>) outs(%17 : memref<1x64x128xf32>)
            loom.semaphore_give %19 : memref<1x128x128xf32>
            linalg.fill ins(%cst_3 : f32) outs(%25 : memref<1x64xf32>)
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%17 : memref<1x64x128xf32>) outs(%25 : memref<1x64xf32>) {
            ^bb0(%in: f32, %out: f32):
              %44 = arith.maximumf %in, %out : f32
              linalg.yield %44 : f32
            }
            linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%23, %25 : memref<1x64xf32>, memref<1x64xf32>) outs(%25 : memref<1x64xf32>) {
            ^bb0(%in: f32, %in_7: f32, %out: f32):
              %44 = arith.truncf %cst_0 : f64 to f32
              %45 = arith.mulf %in_7, %44 : f32
              %46 = arith.cmpf ogt, %in, %45 : f32
              %47 = arith.select %46, %in, %45 : f32
              linalg.yield %47 : f32
            }
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%17, %25 : memref<1x64x128xf32>, memref<1x64xf32>) outs(%17 : memref<1x64x128xf32>) {
            ^bb0(%in: f32, %in_7: f32, %out: f32):
              %44 = arith.truncf %cst_0 : f64 to f32
              %45 = arith.mulf %in, %44 : f32
              %46 = arith.subf %45, %in_7 : f32
              %47 = math.powf %cst, %46 : f32
              linalg.yield %47 : f32
            }
            linalg.fill ins(%cst_1 : f32) outs(%27 : memref<1x64xf32>)
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%17 : memref<1x64x128xf32>) outs(%27 : memref<1x64xf32>) {
            ^bb0(%in: f32, %out: f32):
              %44 = arith.addf %in, %out : f32
              linalg.yield %44 : f32
            }
            linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%23, %25 : memref<1x64xf32>, memref<1x64xf32>) outs(%29 : memref<1x64xf32>) {
            ^bb0(%in: f32, %in_7: f32, %out: f32):
              %44 = arith.subf %in, %in_7 : f32
              %45 = math.powf %cst, %44 : f32
              linalg.yield %45 : f32
            }
            linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%21, %29, %27 : memref<1x64xf32>, memref<1x64xf32>, memref<1x64xf32>) outs(%21 : memref<1x64xf32>) {
            ^bb0(%in: f32, %in_7: f32, %in_8: f32, %out: f32):
              %44 = arith.mulf %in, %in_7 : f32
              %45 = arith.addf %44, %in_8 : f32
              linalg.yield %45 : f32
            }
            loom.semaphore_give %27 : memref<1x64xf32>
            %42 = arith.muli %arg7, %c16384 : index
            %43 = arith.addi %37, %42 : index
            %reinterpret_cast_6 = memref.reinterpret_cast %arg1 to offset: [%43], sizes: [1, 128, 128], strides: [524288, 128, 1] : memref<32x4096x128xf32> to memref<1x128x128xf32, strided<[524288, 128, 1], offset: ?>>
            loom.copy %reinterpret_cast_6, %15 src_mem_space @DRAM dst_mem_space @L1, interconnect : [@horizontal_links, @vertical_links], broadcast : [8, 8] : memref<1x128x128xf32, strided<[524288, 128, 1], offset: ?>>, memref<1x128x128xf32>
            linalg.fill ins(%cst_1 : f32) outs(%36 : memref<1x64x128xf32>)
            linalg.batch_matmul ins(%17, %15 : memref<1x64x128xf32>, memref<1x128x128xf32>) outs(%36 : memref<1x64x128xf32>)
            loom.semaphore_give %15 : memref<1x128x128xf32>
            loom.semaphore_give %17 : memref<1x64x128xf32>
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%36, %34, %29 : memref<1x64x128xf32>, memref<1x64x128xf32>, memref<1x64xf32>) outs(%34 : memref<1x64x128xf32>) {
            ^bb0(%in: f32, %in_7: f32, %in_8: f32, %out: f32):
              %44 = arith.mulf %in_7, %in_8 : f32
              %45 = arith.addf %in, %44 : f32
              linalg.yield %45 : f32
            }
            loom.semaphore_give %29 : memref<1x64xf32>
            loom.semaphore_give %36 : memref<1x64x128xf32>
            linalg.copy ins(%25 : memref<1x64xf32>) outs(%23 : memref<1x64xf32>)
            loom.semaphore_give %25 : memref<1x64xf32>
          }
          loom.semaphore_give %23 : memref<1x64xf32>
          loom.semaphore_give %32 : memref<1x64x128xf32>
          linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%34, %21 : memref<1x64x128xf32>, memref<1x64xf32>) outs(%31 : memref<1x64x128xf32>) {
          ^bb0(%in: f32, %in_5: f32, %out: f32):
            %40 = arith.divf %in, %in_5 : f32
            linalg.yield %40 : f32
          }
          loom.semaphore_give %21 : memref<1x64xf32>
          loom.semaphore_give %34 : memref<1x64x128xf32>
          %reinterpret_cast_4 = memref.reinterpret_cast %arg3 to offset: [%39], sizes: [1, 64, 128], strides: [524288, 128, 1] : memref<32x4096x128xf32> to memref<1x64x128xf32, strided<[524288, 128, 1], offset: ?>>
          loom.copy %31, %reinterpret_cast_4 src_mem_space @L1 dst_mem_space @DRAM, interconnect : [], broadcast : [1, 1] : memref<1x64x128xf32>, memref<1x64x128xf32, strided<[524288, 128, 1], offset: ?>>
          loom.semaphore_give %31 : memref<1x64x128xf32>
        }
        scf.reduce 
      } {loom.iter_types = [#loom.iter_type<spatial>, #loom.iter_type<spatial>], loom.mapped_to_dims = [@y, @x]}
      return
    }
  }
  module attributes {loom.pass_name = "Materialize"} {
    func.func @attention__d1i1_d0i1__f10__d_d_h__BB1__BM64__BN128(%arg0: memref<32x128x4096xf32>, %arg1: memref<32x4096x128xf32>, %arg2: memref<32x4096x128xf32>, %arg3: memref<32x4096x128xf32>) {
      %c16384 = arith.constant 16384 : index
      %c8192 = arith.constant 8192 : index
      %c524288 = arith.constant 524288 : index
      %c32 = arith.constant 32 : index
      %cst = arith.constant 2.000000e+00 : f32
      %cst_0 = arith.constant 0.12751743074602467 : f64
      %cst_1 = arith.constant 0.000000e+00 : f32
      %cst_2 = arith.constant 1.000000e+00 : f32
      %cst_3 = arith.constant 0xFF800000 : f32
      %c128 = arith.constant 128 : index
      %c0 = arith.constant 0 : index
      %c8 = arith.constant 8 : index
      %c1 = arith.constant 1 : index
      scf.parallel (%arg4, %arg5) = (%c0, %c0) to (%c8, %c8) step (%c1, %c1) {
        scf.for %arg6 = %c0 to %c32 step %c1 {
          %12 = arith.muli %arg4, %c8 overflow<nsw> : index
          %13 = arith.addi %12, %arg5 : index
          %14 = loom.alloc [1, 128, 128] on @L1 : memref<1x128x128xf32>
          %15 = loom.semaphore_take %14 : memref<1x128x128xf32> -> memref<1x128x128xf32>
          %16 = loom.alloc [1, 64, 128] on @L1 : memref<1x64x128xf32>
          %17 = loom.semaphore_take %16 : memref<1x64x128xf32> -> memref<1x64x128xf32>
          %18 = loom.alloc [1, 128, 128] on @L1 : memref<1x128x128xf32>
          %19 = loom.semaphore_take %18 : memref<1x128x128xf32> -> memref<1x128x128xf32>
          %20 = loom.alloc [1, 64] on @L1 : memref<1x64xf32>
          %21 = loom.semaphore_take %20 : memref<1x64xf32> -> memref<1x64xf32>
          %22 = loom.alloc [1, 64] on @L1 : memref<1x64xf32>
          %23 = loom.semaphore_take %22 : memref<1x64xf32> -> memref<1x64xf32>
          %24 = loom.alloc [1, 64] on @L1 : memref<1x64xf32>
          %25 = loom.semaphore_take %24 : memref<1x64xf32> -> memref<1x64xf32>
          %26 = loom.alloc [1, 64] on @L1 : memref<1x64xf32>
          %27 = loom.semaphore_take %26 : memref<1x64xf32> -> memref<1x64xf32>
          %28 = loom.alloc [1, 64] on @L1 : memref<1x64xf32>
          %29 = loom.semaphore_take %28 : memref<1x64xf32> -> memref<1x64xf32>
          %30 = loom.alloc [1, 64, 128] on @L1 : memref<1x64x128xf32>
          %31 = loom.semaphore_take %30 : memref<1x64x128xf32> -> memref<1x64x128xf32>
          %32 = loom.semaphore_take %30 : memref<1x64x128xf32> -> memref<1x64x128xf32>
          %33 = loom.alloc [1, 64, 128] on @L1 : memref<1x64x128xf32>
          %34 = loom.semaphore_take %33 : memref<1x64x128xf32> -> memref<1x64x128xf32>
          %35 = loom.alloc [1, 64, 128] on @L1 : memref<1x64x128xf32>
          %36 = loom.semaphore_take %35 : memref<1x64x128xf32> -> memref<1x64x128xf32>
          %37 = arith.muli %arg6, %c524288 overflow<nsw> : index
          %38 = arith.muli %13, %c8192 : index
          %39 = arith.addi %37, %38 : index
          %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%39], sizes: [1, 64, 128], strides: [524288, 128, 1] : memref<32x4096x128xf32> to memref<1x64x128xf32, strided<[524288, 128, 1], offset: ?>>
          loom.copy %reinterpret_cast, %32 src_mem_space @DRAM dst_mem_space @L1, interconnect : [], broadcast : [1, 1] : memref<1x64x128xf32, strided<[524288, 128, 1], offset: ?>>, memref<1x64x128xf32>
          linalg.fill ins(%cst_1 : f32) outs(%34 : memref<1x64x128xf32>)
          linalg.fill ins(%cst_2 : f32) outs(%21 : memref<1x64xf32>)
          linalg.fill ins(%cst_3 : f32) outs(%23 : memref<1x64xf32>)
          scf.for %arg7 = %c0 to %c32 step %c1 {
            %40 = arith.muli %arg7, %c128 : index
            %41 = arith.addi %37, %40 : index
            %reinterpret_cast_5 = memref.reinterpret_cast %arg0 to offset: [%41], sizes: [1, 128, 128], strides: [524288, 4096, 1] : memref<32x128x4096xf32> to memref<1x128x128xf32, strided<[524288, 4096, 1], offset: ?>>
            loom.copy %reinterpret_cast_5, %19 src_mem_space @DRAM dst_mem_space @L1, interconnect : [], broadcast : [1, 1] : memref<1x128x128xf32, strided<[524288, 4096, 1], offset: ?>>, memref<1x128x128xf32>
            linalg.fill ins(%cst_1 : f32) outs(%17 : memref<1x64x128xf32>)
            linalg.batch_matmul ins(%32, %19 : memref<1x64x128xf32>, memref<1x128x128xf32>) outs(%17 : memref<1x64x128xf32>)
            loom.semaphore_give %19 : memref<1x128x128xf32>
            linalg.fill ins(%cst_3 : f32) outs(%25 : memref<1x64xf32>)
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%17 : memref<1x64x128xf32>) outs(%25 : memref<1x64xf32>) {
            ^bb0(%in: f32, %out: f32):
              %44 = arith.maximumf %in, %out : f32
              linalg.yield %44 : f32
            }
            linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%23, %25 : memref<1x64xf32>, memref<1x64xf32>) outs(%25 : memref<1x64xf32>) {
            ^bb0(%in: f32, %in_7: f32, %out: f32):
              %44 = arith.truncf %cst_0 : f64 to f32
              %45 = arith.mulf %in_7, %44 : f32
              %46 = arith.cmpf ogt, %in, %45 : f32
              %47 = arith.select %46, %in, %45 : f32
              linalg.yield %47 : f32
            }
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%17, %25 : memref<1x64x128xf32>, memref<1x64xf32>) outs(%17 : memref<1x64x128xf32>) {
            ^bb0(%in: f32, %in_7: f32, %out: f32):
              %44 = arith.truncf %cst_0 : f64 to f32
              %45 = arith.mulf %in, %44 : f32
              %46 = arith.subf %45, %in_7 : f32
              %47 = math.powf %cst, %46 : f32
              linalg.yield %47 : f32
            }
            linalg.fill ins(%cst_1 : f32) outs(%27 : memref<1x64xf32>)
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%17 : memref<1x64x128xf32>) outs(%27 : memref<1x64xf32>) {
            ^bb0(%in: f32, %out: f32):
              %44 = arith.addf %in, %out : f32
              linalg.yield %44 : f32
            }
            linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%23, %25 : memref<1x64xf32>, memref<1x64xf32>) outs(%29 : memref<1x64xf32>) {
            ^bb0(%in: f32, %in_7: f32, %out: f32):
              %44 = arith.subf %in, %in_7 : f32
              %45 = math.powf %cst, %44 : f32
              linalg.yield %45 : f32
            }
            linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%21, %29, %27 : memref<1x64xf32>, memref<1x64xf32>, memref<1x64xf32>) outs(%21 : memref<1x64xf32>) {
            ^bb0(%in: f32, %in_7: f32, %in_8: f32, %out: f32):
              %44 = arith.mulf %in, %in_7 : f32
              %45 = arith.addf %44, %in_8 : f32
              linalg.yield %45 : f32
            }
            loom.semaphore_give %27 : memref<1x64xf32>
            %42 = arith.muli %arg7, %c16384 : index
            %43 = arith.addi %37, %42 : index
            %reinterpret_cast_6 = memref.reinterpret_cast %arg1 to offset: [%43], sizes: [1, 128, 128], strides: [524288, 128, 1] : memref<32x4096x128xf32> to memref<1x128x128xf32, strided<[524288, 128, 1], offset: ?>>
            loom.copy %reinterpret_cast_6, %15 src_mem_space @DRAM dst_mem_space @L1, interconnect : [@horizontal_links], broadcast : [1, 8] : memref<1x128x128xf32, strided<[524288, 128, 1], offset: ?>>, memref<1x128x128xf32>
            linalg.fill ins(%cst_1 : f32) outs(%36 : memref<1x64x128xf32>)
            linalg.batch_matmul ins(%17, %15 : memref<1x64x128xf32>, memref<1x128x128xf32>) outs(%36 : memref<1x64x128xf32>)
            loom.semaphore_give %15 : memref<1x128x128xf32>
            loom.semaphore_give %17 : memref<1x64x128xf32>
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%36, %34, %29 : memref<1x64x128xf32>, memref<1x64x128xf32>, memref<1x64xf32>) outs(%34 : memref<1x64x128xf32>) {
            ^bb0(%in: f32, %in_7: f32, %in_8: f32, %out: f32):
              %44 = arith.mulf %in_7, %in_8 : f32
              %45 = arith.addf %in, %44 : f32
              linalg.yield %45 : f32
            }
            loom.semaphore_give %29 : memref<1x64xf32>
            loom.semaphore_give %36 : memref<1x64x128xf32>
            linalg.copy ins(%25 : memref<1x64xf32>) outs(%23 : memref<1x64xf32>)
            loom.semaphore_give %25 : memref<1x64xf32>
          }
          loom.semaphore_give %23 : memref<1x64xf32>
          loom.semaphore_give %32 : memref<1x64x128xf32>
          linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%34, %21 : memref<1x64x128xf32>, memref<1x64xf32>) outs(%31 : memref<1x64x128xf32>) {
          ^bb0(%in: f32, %in_5: f32, %out: f32):
            %40 = arith.divf %in, %in_5 : f32
            linalg.yield %40 : f32
          }
          loom.semaphore_give %21 : memref<1x64xf32>
          loom.semaphore_give %34 : memref<1x64x128xf32>
          %reinterpret_cast_4 = memref.reinterpret_cast %arg3 to offset: [%39], sizes: [1, 64, 128], strides: [524288, 128, 1] : memref<32x4096x128xf32> to memref<1x64x128xf32, strided<[524288, 128, 1], offset: ?>>
          loom.copy %31, %reinterpret_cast_4 src_mem_space @L1 dst_mem_space @DRAM, interconnect : [], broadcast : [1, 1] : memref<1x64x128xf32>, memref<1x64x128xf32, strided<[524288, 128, 1], offset: ?>>
          loom.semaphore_give %31 : memref<1x64x128xf32>
        }
        scf.reduce 
      } {loom.iter_types = [#loom.iter_type<spatial>, #loom.iter_type<spatial>], loom.mapped_to_dims = [@y, @x]}
      return
    }
  }
  module attributes {loom.pass_name = "Materialize"} {
    func.func @attention__d1i1_d0i1__f10__d_d_v__BB1__BM64__BN128(%arg0: memref<32x128x4096xf32>, %arg1: memref<32x4096x128xf32>, %arg2: memref<32x4096x128xf32>, %arg3: memref<32x4096x128xf32>) {
      %c16384 = arith.constant 16384 : index
      %c8192 = arith.constant 8192 : index
      %c524288 = arith.constant 524288 : index
      %c32 = arith.constant 32 : index
      %cst = arith.constant 2.000000e+00 : f32
      %cst_0 = arith.constant 0.12751743074602467 : f64
      %cst_1 = arith.constant 0.000000e+00 : f32
      %cst_2 = arith.constant 1.000000e+00 : f32
      %cst_3 = arith.constant 0xFF800000 : f32
      %c128 = arith.constant 128 : index
      %c0 = arith.constant 0 : index
      %c8 = arith.constant 8 : index
      %c1 = arith.constant 1 : index
      scf.parallel (%arg4, %arg5) = (%c0, %c0) to (%c8, %c8) step (%c1, %c1) {
        scf.for %arg6 = %c0 to %c32 step %c1 {
          %12 = arith.muli %arg4, %c8 overflow<nsw> : index
          %13 = arith.addi %12, %arg5 : index
          %14 = loom.alloc [1, 128, 128] on @L1 : memref<1x128x128xf32>
          %15 = loom.semaphore_take %14 : memref<1x128x128xf32> -> memref<1x128x128xf32>
          %16 = loom.alloc [1, 64, 128] on @L1 : memref<1x64x128xf32>
          %17 = loom.semaphore_take %16 : memref<1x64x128xf32> -> memref<1x64x128xf32>
          %18 = loom.alloc [1, 128, 128] on @L1 : memref<1x128x128xf32>
          %19 = loom.semaphore_take %18 : memref<1x128x128xf32> -> memref<1x128x128xf32>
          %20 = loom.alloc [1, 64] on @L1 : memref<1x64xf32>
          %21 = loom.semaphore_take %20 : memref<1x64xf32> -> memref<1x64xf32>
          %22 = loom.alloc [1, 64] on @L1 : memref<1x64xf32>
          %23 = loom.semaphore_take %22 : memref<1x64xf32> -> memref<1x64xf32>
          %24 = loom.alloc [1, 64] on @L1 : memref<1x64xf32>
          %25 = loom.semaphore_take %24 : memref<1x64xf32> -> memref<1x64xf32>
          %26 = loom.alloc [1, 64] on @L1 : memref<1x64xf32>
          %27 = loom.semaphore_take %26 : memref<1x64xf32> -> memref<1x64xf32>
          %28 = loom.alloc [1, 64] on @L1 : memref<1x64xf32>
          %29 = loom.semaphore_take %28 : memref<1x64xf32> -> memref<1x64xf32>
          %30 = loom.alloc [1, 64, 128] on @L1 : memref<1x64x128xf32>
          %31 = loom.semaphore_take %30 : memref<1x64x128xf32> -> memref<1x64x128xf32>
          %32 = loom.semaphore_take %30 : memref<1x64x128xf32> -> memref<1x64x128xf32>
          %33 = loom.alloc [1, 64, 128] on @L1 : memref<1x64x128xf32>
          %34 = loom.semaphore_take %33 : memref<1x64x128xf32> -> memref<1x64x128xf32>
          %35 = loom.alloc [1, 64, 128] on @L1 : memref<1x64x128xf32>
          %36 = loom.semaphore_take %35 : memref<1x64x128xf32> -> memref<1x64x128xf32>
          %37 = arith.muli %arg6, %c524288 overflow<nsw> : index
          %38 = arith.muli %13, %c8192 : index
          %39 = arith.addi %37, %38 : index
          %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%39], sizes: [1, 64, 128], strides: [524288, 128, 1] : memref<32x4096x128xf32> to memref<1x64x128xf32, strided<[524288, 128, 1], offset: ?>>
          loom.copy %reinterpret_cast, %32 src_mem_space @DRAM dst_mem_space @L1, interconnect : [], broadcast : [1, 1] : memref<1x64x128xf32, strided<[524288, 128, 1], offset: ?>>, memref<1x64x128xf32>
          linalg.fill ins(%cst_1 : f32) outs(%34 : memref<1x64x128xf32>)
          linalg.fill ins(%cst_2 : f32) outs(%21 : memref<1x64xf32>)
          linalg.fill ins(%cst_3 : f32) outs(%23 : memref<1x64xf32>)
          scf.for %arg7 = %c0 to %c32 step %c1 {
            %40 = arith.muli %arg7, %c128 : index
            %41 = arith.addi %37, %40 : index
            %reinterpret_cast_5 = memref.reinterpret_cast %arg0 to offset: [%41], sizes: [1, 128, 128], strides: [524288, 4096, 1] : memref<32x128x4096xf32> to memref<1x128x128xf32, strided<[524288, 4096, 1], offset: ?>>
            loom.copy %reinterpret_cast_5, %19 src_mem_space @DRAM dst_mem_space @L1, interconnect : [], broadcast : [1, 1] : memref<1x128x128xf32, strided<[524288, 4096, 1], offset: ?>>, memref<1x128x128xf32>
            linalg.fill ins(%cst_1 : f32) outs(%17 : memref<1x64x128xf32>)
            linalg.batch_matmul ins(%32, %19 : memref<1x64x128xf32>, memref<1x128x128xf32>) outs(%17 : memref<1x64x128xf32>)
            loom.semaphore_give %19 : memref<1x128x128xf32>
            linalg.fill ins(%cst_3 : f32) outs(%25 : memref<1x64xf32>)
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%17 : memref<1x64x128xf32>) outs(%25 : memref<1x64xf32>) {
            ^bb0(%in: f32, %out: f32):
              %44 = arith.maximumf %in, %out : f32
              linalg.yield %44 : f32
            }
            linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%23, %25 : memref<1x64xf32>, memref<1x64xf32>) outs(%25 : memref<1x64xf32>) {
            ^bb0(%in: f32, %in_7: f32, %out: f32):
              %44 = arith.truncf %cst_0 : f64 to f32
              %45 = arith.mulf %in_7, %44 : f32
              %46 = arith.cmpf ogt, %in, %45 : f32
              %47 = arith.select %46, %in, %45 : f32
              linalg.yield %47 : f32
            }
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%17, %25 : memref<1x64x128xf32>, memref<1x64xf32>) outs(%17 : memref<1x64x128xf32>) {
            ^bb0(%in: f32, %in_7: f32, %out: f32):
              %44 = arith.truncf %cst_0 : f64 to f32
              %45 = arith.mulf %in, %44 : f32
              %46 = arith.subf %45, %in_7 : f32
              %47 = math.powf %cst, %46 : f32
              linalg.yield %47 : f32
            }
            linalg.fill ins(%cst_1 : f32) outs(%27 : memref<1x64xf32>)
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%17 : memref<1x64x128xf32>) outs(%27 : memref<1x64xf32>) {
            ^bb0(%in: f32, %out: f32):
              %44 = arith.addf %in, %out : f32
              linalg.yield %44 : f32
            }
            linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%23, %25 : memref<1x64xf32>, memref<1x64xf32>) outs(%29 : memref<1x64xf32>) {
            ^bb0(%in: f32, %in_7: f32, %out: f32):
              %44 = arith.subf %in, %in_7 : f32
              %45 = math.powf %cst, %44 : f32
              linalg.yield %45 : f32
            }
            linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%21, %29, %27 : memref<1x64xf32>, memref<1x64xf32>, memref<1x64xf32>) outs(%21 : memref<1x64xf32>) {
            ^bb0(%in: f32, %in_7: f32, %in_8: f32, %out: f32):
              %44 = arith.mulf %in, %in_7 : f32
              %45 = arith.addf %44, %in_8 : f32
              linalg.yield %45 : f32
            }
            loom.semaphore_give %27 : memref<1x64xf32>
            %42 = arith.muli %arg7, %c16384 : index
            %43 = arith.addi %37, %42 : index
            %reinterpret_cast_6 = memref.reinterpret_cast %arg1 to offset: [%43], sizes: [1, 128, 128], strides: [524288, 128, 1] : memref<32x4096x128xf32> to memref<1x128x128xf32, strided<[524288, 128, 1], offset: ?>>
            loom.copy %reinterpret_cast_6, %15 src_mem_space @DRAM dst_mem_space @L1, interconnect : [@vertical_links], broadcast : [8, 1] : memref<1x128x128xf32, strided<[524288, 128, 1], offset: ?>>, memref<1x128x128xf32>
            linalg.fill ins(%cst_1 : f32) outs(%36 : memref<1x64x128xf32>)
            linalg.batch_matmul ins(%17, %15 : memref<1x64x128xf32>, memref<1x128x128xf32>) outs(%36 : memref<1x64x128xf32>)
            loom.semaphore_give %15 : memref<1x128x128xf32>
            loom.semaphore_give %17 : memref<1x64x128xf32>
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%36, %34, %29 : memref<1x64x128xf32>, memref<1x64x128xf32>, memref<1x64xf32>) outs(%34 : memref<1x64x128xf32>) {
            ^bb0(%in: f32, %in_7: f32, %in_8: f32, %out: f32):
              %44 = arith.mulf %in_7, %in_8 : f32
              %45 = arith.addf %in, %44 : f32
              linalg.yield %45 : f32
            }
            loom.semaphore_give %29 : memref<1x64xf32>
            loom.semaphore_give %36 : memref<1x64x128xf32>
            linalg.copy ins(%25 : memref<1x64xf32>) outs(%23 : memref<1x64xf32>)
            loom.semaphore_give %25 : memref<1x64xf32>
          }
          loom.semaphore_give %23 : memref<1x64xf32>
          loom.semaphore_give %32 : memref<1x64x128xf32>
          linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%34, %21 : memref<1x64x128xf32>, memref<1x64xf32>) outs(%31 : memref<1x64x128xf32>) {
          ^bb0(%in: f32, %in_5: f32, %out: f32):
            %40 = arith.divf %in, %in_5 : f32
            linalg.yield %40 : f32
          }
          loom.semaphore_give %21 : memref<1x64xf32>
          loom.semaphore_give %34 : memref<1x64x128xf32>
          %reinterpret_cast_4 = memref.reinterpret_cast %arg3 to offset: [%39], sizes: [1, 64, 128], strides: [524288, 128, 1] : memref<32x4096x128xf32> to memref<1x64x128xf32, strided<[524288, 128, 1], offset: ?>>
          loom.copy %31, %reinterpret_cast_4 src_mem_space @L1 dst_mem_space @DRAM, interconnect : [], broadcast : [1, 1] : memref<1x64x128xf32>, memref<1x64x128xf32, strided<[524288, 128, 1], offset: ?>>
          loom.semaphore_give %31 : memref<1x64x128xf32>
        }
        scf.reduce 
      } {loom.iter_types = [#loom.iter_type<spatial>, #loom.iter_type<spatial>], loom.mapped_to_dims = [@y, @x]}
      return
    }
  }
  module attributes {loom.pass_name = "Materialize"} {
    func.func @attention__d1i1_d0i1__f10__d_a_d__BB1__BM64__BN128(%arg0: memref<32x128x4096xf32>, %arg1: memref<32x4096x128xf32>, %arg2: memref<32x4096x128xf32>, %arg3: memref<32x4096x128xf32>) {
      %c16384 = arith.constant 16384 : index
      %c8192 = arith.constant 8192 : index
      %c524288 = arith.constant 524288 : index
      %c32 = arith.constant 32 : index
      %cst = arith.constant 2.000000e+00 : f32
      %cst_0 = arith.constant 0.12751743074602467 : f64
      %cst_1 = arith.constant 0.000000e+00 : f32
      %cst_2 = arith.constant 1.000000e+00 : f32
      %cst_3 = arith.constant 0xFF800000 : f32
      %c128 = arith.constant 128 : index
      %c0 = arith.constant 0 : index
      %c8 = arith.constant 8 : index
      %c1 = arith.constant 1 : index
      scf.parallel (%arg4, %arg5) = (%c0, %c0) to (%c8, %c8) step (%c1, %c1) {
        scf.for %arg6 = %c0 to %c32 step %c1 {
          %12 = arith.muli %arg4, %c8 overflow<nsw> : index
          %13 = arith.addi %12, %arg5 : index
          %14 = loom.alloc [1, 128, 128] on @L1 : memref<1x128x128xf32>
          %15 = loom.semaphore_take %14 : memref<1x128x128xf32> -> memref<1x128x128xf32>
          %16 = loom.alloc [1, 64, 128] on @L1 : memref<1x64x128xf32>
          %17 = loom.semaphore_take %16 : memref<1x64x128xf32> -> memref<1x64x128xf32>
          %18 = loom.alloc [1, 128, 128] on @L1 : memref<1x128x128xf32>
          %19 = loom.semaphore_take %18 : memref<1x128x128xf32> -> memref<1x128x128xf32>
          %20 = loom.alloc [1, 64] on @L1 : memref<1x64xf32>
          %21 = loom.semaphore_take %20 : memref<1x64xf32> -> memref<1x64xf32>
          %22 = loom.alloc [1, 64] on @L1 : memref<1x64xf32>
          %23 = loom.semaphore_take %22 : memref<1x64xf32> -> memref<1x64xf32>
          %24 = loom.alloc [1, 64] on @L1 : memref<1x64xf32>
          %25 = loom.semaphore_take %24 : memref<1x64xf32> -> memref<1x64xf32>
          %26 = loom.alloc [1, 64] on @L1 : memref<1x64xf32>
          %27 = loom.semaphore_take %26 : memref<1x64xf32> -> memref<1x64xf32>
          %28 = loom.alloc [1, 64] on @L1 : memref<1x64xf32>
          %29 = loom.semaphore_take %28 : memref<1x64xf32> -> memref<1x64xf32>
          %30 = loom.alloc [1, 64, 128] on @L1 : memref<1x64x128xf32>
          %31 = loom.semaphore_take %30 : memref<1x64x128xf32> -> memref<1x64x128xf32>
          %32 = loom.semaphore_take %30 : memref<1x64x128xf32> -> memref<1x64x128xf32>
          %33 = loom.alloc [1, 64, 128] on @L1 : memref<1x64x128xf32>
          %34 = loom.semaphore_take %33 : memref<1x64x128xf32> -> memref<1x64x128xf32>
          %35 = loom.alloc [1, 64, 128] on @L1 : memref<1x64x128xf32>
          %36 = loom.semaphore_take %35 : memref<1x64x128xf32> -> memref<1x64x128xf32>
          %37 = arith.muli %arg6, %c524288 overflow<nsw> : index
          %38 = arith.muli %13, %c8192 : index
          %39 = arith.addi %37, %38 : index
          %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%39], sizes: [1, 64, 128], strides: [524288, 128, 1] : memref<32x4096x128xf32> to memref<1x64x128xf32, strided<[524288, 128, 1], offset: ?>>
          loom.copy %reinterpret_cast, %32 src_mem_space @DRAM dst_mem_space @L1, interconnect : [], broadcast : [1, 1] : memref<1x64x128xf32, strided<[524288, 128, 1], offset: ?>>, memref<1x64x128xf32>
          linalg.fill ins(%cst_1 : f32) outs(%34 : memref<1x64x128xf32>)
          linalg.fill ins(%cst_2 : f32) outs(%21 : memref<1x64xf32>)
          linalg.fill ins(%cst_3 : f32) outs(%23 : memref<1x64xf32>)
          scf.for %arg7 = %c0 to %c32 step %c1 {
            %40 = arith.muli %arg7, %c128 : index
            %41 = arith.addi %37, %40 : index
            %reinterpret_cast_5 = memref.reinterpret_cast %arg0 to offset: [%41], sizes: [1, 128, 128], strides: [524288, 4096, 1] : memref<32x128x4096xf32> to memref<1x128x128xf32, strided<[524288, 4096, 1], offset: ?>>
            loom.copy %reinterpret_cast_5, %19 src_mem_space @DRAM dst_mem_space @L1, interconnect : [@horizontal_links, @vertical_links], broadcast : [8, 8] : memref<1x128x128xf32, strided<[524288, 4096, 1], offset: ?>>, memref<1x128x128xf32>
            linalg.fill ins(%cst_1 : f32) outs(%17 : memref<1x64x128xf32>)
            linalg.batch_matmul ins(%32, %19 : memref<1x64x128xf32>, memref<1x128x128xf32>) outs(%17 : memref<1x64x128xf32>)
            loom.semaphore_give %19 : memref<1x128x128xf32>
            linalg.fill ins(%cst_3 : f32) outs(%25 : memref<1x64xf32>)
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%17 : memref<1x64x128xf32>) outs(%25 : memref<1x64xf32>) {
            ^bb0(%in: f32, %out: f32):
              %44 = arith.maximumf %in, %out : f32
              linalg.yield %44 : f32
            }
            linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%23, %25 : memref<1x64xf32>, memref<1x64xf32>) outs(%25 : memref<1x64xf32>) {
            ^bb0(%in: f32, %in_7: f32, %out: f32):
              %44 = arith.truncf %cst_0 : f64 to f32
              %45 = arith.mulf %in_7, %44 : f32
              %46 = arith.cmpf ogt, %in, %45 : f32
              %47 = arith.select %46, %in, %45 : f32
              linalg.yield %47 : f32
            }
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%17, %25 : memref<1x64x128xf32>, memref<1x64xf32>) outs(%17 : memref<1x64x128xf32>) {
            ^bb0(%in: f32, %in_7: f32, %out: f32):
              %44 = arith.truncf %cst_0 : f64 to f32
              %45 = arith.mulf %in, %44 : f32
              %46 = arith.subf %45, %in_7 : f32
              %47 = math.powf %cst, %46 : f32
              linalg.yield %47 : f32
            }
            linalg.fill ins(%cst_1 : f32) outs(%27 : memref<1x64xf32>)
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%17 : memref<1x64x128xf32>) outs(%27 : memref<1x64xf32>) {
            ^bb0(%in: f32, %out: f32):
              %44 = arith.addf %in, %out : f32
              linalg.yield %44 : f32
            }
            linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%23, %25 : memref<1x64xf32>, memref<1x64xf32>) outs(%29 : memref<1x64xf32>) {
            ^bb0(%in: f32, %in_7: f32, %out: f32):
              %44 = arith.subf %in, %in_7 : f32
              %45 = math.powf %cst, %44 : f32
              linalg.yield %45 : f32
            }
            linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%21, %29, %27 : memref<1x64xf32>, memref<1x64xf32>, memref<1x64xf32>) outs(%21 : memref<1x64xf32>) {
            ^bb0(%in: f32, %in_7: f32, %in_8: f32, %out: f32):
              %44 = arith.mulf %in, %in_7 : f32
              %45 = arith.addf %44, %in_8 : f32
              linalg.yield %45 : f32
            }
            loom.semaphore_give %27 : memref<1x64xf32>
            %42 = arith.muli %arg7, %c16384 : index
            %43 = arith.addi %37, %42 : index
            %reinterpret_cast_6 = memref.reinterpret_cast %arg1 to offset: [%43], sizes: [1, 128, 128], strides: [524288, 128, 1] : memref<32x4096x128xf32> to memref<1x128x128xf32, strided<[524288, 128, 1], offset: ?>>
            loom.copy %reinterpret_cast_6, %15 src_mem_space @DRAM dst_mem_space @L1, interconnect : [], broadcast : [1, 1] : memref<1x128x128xf32, strided<[524288, 128, 1], offset: ?>>, memref<1x128x128xf32>
            linalg.fill ins(%cst_1 : f32) outs(%36 : memref<1x64x128xf32>)
            linalg.batch_matmul ins(%17, %15 : memref<1x64x128xf32>, memref<1x128x128xf32>) outs(%36 : memref<1x64x128xf32>)
            loom.semaphore_give %15 : memref<1x128x128xf32>
            loom.semaphore_give %17 : memref<1x64x128xf32>
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%36, %34, %29 : memref<1x64x128xf32>, memref<1x64x128xf32>, memref<1x64xf32>) outs(%34 : memref<1x64x128xf32>) {
            ^bb0(%in: f32, %in_7: f32, %in_8: f32, %out: f32):
              %44 = arith.mulf %in_7, %in_8 : f32
              %45 = arith.addf %in, %44 : f32
              linalg.yield %45 : f32
            }
            loom.semaphore_give %29 : memref<1x64xf32>
            loom.semaphore_give %36 : memref<1x64x128xf32>
            linalg.copy ins(%25 : memref<1x64xf32>) outs(%23 : memref<1x64xf32>)
            loom.semaphore_give %25 : memref<1x64xf32>
          }
          loom.semaphore_give %23 : memref<1x64xf32>
          loom.semaphore_give %32 : memref<1x64x128xf32>
          linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%34, %21 : memref<1x64x128xf32>, memref<1x64xf32>) outs(%31 : memref<1x64x128xf32>) {
          ^bb0(%in: f32, %in_5: f32, %out: f32):
            %40 = arith.divf %in, %in_5 : f32
            linalg.yield %40 : f32
          }
          loom.semaphore_give %21 : memref<1x64xf32>
          loom.semaphore_give %34 : memref<1x64x128xf32>
          %reinterpret_cast_4 = memref.reinterpret_cast %arg3 to offset: [%39], sizes: [1, 64, 128], strides: [524288, 128, 1] : memref<32x4096x128xf32> to memref<1x64x128xf32, strided<[524288, 128, 1], offset: ?>>
          loom.copy %31, %reinterpret_cast_4 src_mem_space @L1 dst_mem_space @DRAM, interconnect : [], broadcast : [1, 1] : memref<1x64x128xf32>, memref<1x64x128xf32, strided<[524288, 128, 1], offset: ?>>
          loom.semaphore_give %31 : memref<1x64x128xf32>
        }
        scf.reduce 
      } {loom.iter_types = [#loom.iter_type<spatial>, #loom.iter_type<spatial>], loom.mapped_to_dims = [@y, @x]}
      return
    }
  }
  module attributes {loom.pass_name = "Materialize"} {
    func.func @attention__d1i1_d0i1__f10__d_a_a__BB1__BM64__BN128(%arg0: memref<32x128x4096xf32>, %arg1: memref<32x4096x128xf32>, %arg2: memref<32x4096x128xf32>, %arg3: memref<32x4096x128xf32>) {
      %c16384 = arith.constant 16384 : index
      %c8192 = arith.constant 8192 : index
      %c524288 = arith.constant 524288 : index
      %c32 = arith.constant 32 : index
      %cst = arith.constant 2.000000e+00 : f32
      %cst_0 = arith.constant 0.12751743074602467 : f64
      %cst_1 = arith.constant 0.000000e+00 : f32
      %cst_2 = arith.constant 1.000000e+00 : f32
      %cst_3 = arith.constant 0xFF800000 : f32
      %c128 = arith.constant 128 : index
      %c0 = arith.constant 0 : index
      %c8 = arith.constant 8 : index
      %c1 = arith.constant 1 : index
      scf.parallel (%arg4, %arg5) = (%c0, %c0) to (%c8, %c8) step (%c1, %c1) {
        scf.for %arg6 = %c0 to %c32 step %c1 {
          %12 = arith.muli %arg4, %c8 overflow<nsw> : index
          %13 = arith.addi %12, %arg5 : index
          %14 = loom.alloc [1, 128, 128] on @L1 : memref<1x128x128xf32>
          %15 = loom.semaphore_take %14 : memref<1x128x128xf32> -> memref<1x128x128xf32>
          %16 = loom.alloc [1, 64, 128] on @L1 : memref<1x64x128xf32>
          %17 = loom.semaphore_take %16 : memref<1x64x128xf32> -> memref<1x64x128xf32>
          %18 = loom.alloc [1, 128, 128] on @L1 : memref<1x128x128xf32>
          %19 = loom.semaphore_take %18 : memref<1x128x128xf32> -> memref<1x128x128xf32>
          %20 = loom.alloc [1, 64] on @L1 : memref<1x64xf32>
          %21 = loom.semaphore_take %20 : memref<1x64xf32> -> memref<1x64xf32>
          %22 = loom.alloc [1, 64] on @L1 : memref<1x64xf32>
          %23 = loom.semaphore_take %22 : memref<1x64xf32> -> memref<1x64xf32>
          %24 = loom.alloc [1, 64] on @L1 : memref<1x64xf32>
          %25 = loom.semaphore_take %24 : memref<1x64xf32> -> memref<1x64xf32>
          %26 = loom.alloc [1, 64] on @L1 : memref<1x64xf32>
          %27 = loom.semaphore_take %26 : memref<1x64xf32> -> memref<1x64xf32>
          %28 = loom.alloc [1, 64] on @L1 : memref<1x64xf32>
          %29 = loom.semaphore_take %28 : memref<1x64xf32> -> memref<1x64xf32>
          %30 = loom.alloc [1, 64, 128] on @L1 : memref<1x64x128xf32>
          %31 = loom.semaphore_take %30 : memref<1x64x128xf32> -> memref<1x64x128xf32>
          %32 = loom.semaphore_take %30 : memref<1x64x128xf32> -> memref<1x64x128xf32>
          %33 = loom.alloc [1, 64, 128] on @L1 : memref<1x64x128xf32>
          %34 = loom.semaphore_take %33 : memref<1x64x128xf32> -> memref<1x64x128xf32>
          %35 = loom.alloc [1, 64, 128] on @L1 : memref<1x64x128xf32>
          %36 = loom.semaphore_take %35 : memref<1x64x128xf32> -> memref<1x64x128xf32>
          %37 = arith.muli %arg6, %c524288 overflow<nsw> : index
          %38 = arith.muli %13, %c8192 : index
          %39 = arith.addi %37, %38 : index
          %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%39], sizes: [1, 64, 128], strides: [524288, 128, 1] : memref<32x4096x128xf32> to memref<1x64x128xf32, strided<[524288, 128, 1], offset: ?>>
          loom.copy %reinterpret_cast, %32 src_mem_space @DRAM dst_mem_space @L1, interconnect : [], broadcast : [1, 1] : memref<1x64x128xf32, strided<[524288, 128, 1], offset: ?>>, memref<1x64x128xf32>
          linalg.fill ins(%cst_1 : f32) outs(%34 : memref<1x64x128xf32>)
          linalg.fill ins(%cst_2 : f32) outs(%21 : memref<1x64xf32>)
          linalg.fill ins(%cst_3 : f32) outs(%23 : memref<1x64xf32>)
          scf.for %arg7 = %c0 to %c32 step %c1 {
            %40 = arith.muli %arg7, %c128 : index
            %41 = arith.addi %37, %40 : index
            %reinterpret_cast_5 = memref.reinterpret_cast %arg0 to offset: [%41], sizes: [1, 128, 128], strides: [524288, 4096, 1] : memref<32x128x4096xf32> to memref<1x128x128xf32, strided<[524288, 4096, 1], offset: ?>>
            loom.copy %reinterpret_cast_5, %19 src_mem_space @DRAM dst_mem_space @L1, interconnect : [@horizontal_links, @vertical_links], broadcast : [8, 8] : memref<1x128x128xf32, strided<[524288, 4096, 1], offset: ?>>, memref<1x128x128xf32>
            linalg.fill ins(%cst_1 : f32) outs(%17 : memref<1x64x128xf32>)
            linalg.batch_matmul ins(%32, %19 : memref<1x64x128xf32>, memref<1x128x128xf32>) outs(%17 : memref<1x64x128xf32>)
            loom.semaphore_give %19 : memref<1x128x128xf32>
            linalg.fill ins(%cst_3 : f32) outs(%25 : memref<1x64xf32>)
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%17 : memref<1x64x128xf32>) outs(%25 : memref<1x64xf32>) {
            ^bb0(%in: f32, %out: f32):
              %44 = arith.maximumf %in, %out : f32
              linalg.yield %44 : f32
            }
            linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%23, %25 : memref<1x64xf32>, memref<1x64xf32>) outs(%25 : memref<1x64xf32>) {
            ^bb0(%in: f32, %in_7: f32, %out: f32):
              %44 = arith.truncf %cst_0 : f64 to f32
              %45 = arith.mulf %in_7, %44 : f32
              %46 = arith.cmpf ogt, %in, %45 : f32
              %47 = arith.select %46, %in, %45 : f32
              linalg.yield %47 : f32
            }
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%17, %25 : memref<1x64x128xf32>, memref<1x64xf32>) outs(%17 : memref<1x64x128xf32>) {
            ^bb0(%in: f32, %in_7: f32, %out: f32):
              %44 = arith.truncf %cst_0 : f64 to f32
              %45 = arith.mulf %in, %44 : f32
              %46 = arith.subf %45, %in_7 : f32
              %47 = math.powf %cst, %46 : f32
              linalg.yield %47 : f32
            }
            linalg.fill ins(%cst_1 : f32) outs(%27 : memref<1x64xf32>)
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%17 : memref<1x64x128xf32>) outs(%27 : memref<1x64xf32>) {
            ^bb0(%in: f32, %out: f32):
              %44 = arith.addf %in, %out : f32
              linalg.yield %44 : f32
            }
            linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%23, %25 : memref<1x64xf32>, memref<1x64xf32>) outs(%29 : memref<1x64xf32>) {
            ^bb0(%in: f32, %in_7: f32, %out: f32):
              %44 = arith.subf %in, %in_7 : f32
              %45 = math.powf %cst, %44 : f32
              linalg.yield %45 : f32
            }
            linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%21, %29, %27 : memref<1x64xf32>, memref<1x64xf32>, memref<1x64xf32>) outs(%21 : memref<1x64xf32>) {
            ^bb0(%in: f32, %in_7: f32, %in_8: f32, %out: f32):
              %44 = arith.mulf %in, %in_7 : f32
              %45 = arith.addf %44, %in_8 : f32
              linalg.yield %45 : f32
            }
            loom.semaphore_give %27 : memref<1x64xf32>
            %42 = arith.muli %arg7, %c16384 : index
            %43 = arith.addi %37, %42 : index
            %reinterpret_cast_6 = memref.reinterpret_cast %arg1 to offset: [%43], sizes: [1, 128, 128], strides: [524288, 128, 1] : memref<32x4096x128xf32> to memref<1x128x128xf32, strided<[524288, 128, 1], offset: ?>>
            loom.copy %reinterpret_cast_6, %15 src_mem_space @DRAM dst_mem_space @L1, interconnect : [@horizontal_links, @vertical_links], broadcast : [8, 8] : memref<1x128x128xf32, strided<[524288, 128, 1], offset: ?>>, memref<1x128x128xf32>
            linalg.fill ins(%cst_1 : f32) outs(%36 : memref<1x64x128xf32>)
            linalg.batch_matmul ins(%17, %15 : memref<1x64x128xf32>, memref<1x128x128xf32>) outs(%36 : memref<1x64x128xf32>)
            loom.semaphore_give %15 : memref<1x128x128xf32>
            loom.semaphore_give %17 : memref<1x64x128xf32>
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%36, %34, %29 : memref<1x64x128xf32>, memref<1x64x128xf32>, memref<1x64xf32>) outs(%34 : memref<1x64x128xf32>) {
            ^bb0(%in: f32, %in_7: f32, %in_8: f32, %out: f32):
              %44 = arith.mulf %in_7, %in_8 : f32
              %45 = arith.addf %in, %44 : f32
              linalg.yield %45 : f32
            }
            loom.semaphore_give %29 : memref<1x64xf32>
            loom.semaphore_give %36 : memref<1x64x128xf32>
            linalg.copy ins(%25 : memref<1x64xf32>) outs(%23 : memref<1x64xf32>)
            loom.semaphore_give %25 : memref<1x64xf32>
          }
          loom.semaphore_give %23 : memref<1x64xf32>
          loom.semaphore_give %32 : memref<1x64x128xf32>
          linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%34, %21 : memref<1x64x128xf32>, memref<1x64xf32>) outs(%31 : memref<1x64x128xf32>) {
          ^bb0(%in: f32, %in_5: f32, %out: f32):
            %40 = arith.divf %in, %in_5 : f32
            linalg.yield %40 : f32
          }
          loom.semaphore_give %21 : memref<1x64xf32>
          loom.semaphore_give %34 : memref<1x64x128xf32>
          %reinterpret_cast_4 = memref.reinterpret_cast %arg3 to offset: [%39], sizes: [1, 64, 128], strides: [524288, 128, 1] : memref<32x4096x128xf32> to memref<1x64x128xf32, strided<[524288, 128, 1], offset: ?>>
          loom.copy %31, %reinterpret_cast_4 src_mem_space @L1 dst_mem_space @DRAM, interconnect : [], broadcast : [1, 1] : memref<1x64x128xf32>, memref<1x64x128xf32, strided<[524288, 128, 1], offset: ?>>
          loom.semaphore_give %31 : memref<1x64x128xf32>
        }
        scf.reduce 
      } {loom.iter_types = [#loom.iter_type<spatial>, #loom.iter_type<spatial>], loom.mapped_to_dims = [@y, @x]}
      return
    }
  }
  module attributes {loom.pass_name = "Materialize"} {
    func.func @attention__d1i1_d0i1__f10__d_a_h__BB1__BM64__BN128(%arg0: memref<32x128x4096xf32>, %arg1: memref<32x4096x128xf32>, %arg2: memref<32x4096x128xf32>, %arg3: memref<32x4096x128xf32>) {
      %c16384 = arith.constant 16384 : index
      %c8192 = arith.constant 8192 : index
      %c524288 = arith.constant 524288 : index
      %c32 = arith.constant 32 : index
      %cst = arith.constant 2.000000e+00 : f32
      %cst_0 = arith.constant 0.12751743074602467 : f64
      %cst_1 = arith.constant 0.000000e+00 : f32
      %cst_2 = arith.constant 1.000000e+00 : f32
      %cst_3 = arith.constant 0xFF800000 : f32
      %c128 = arith.constant 128 : index
      %c0 = arith.constant 0 : index
      %c8 = arith.constant 8 : index
      %c1 = arith.constant 1 : index
      scf.parallel (%arg4, %arg5) = (%c0, %c0) to (%c8, %c8) step (%c1, %c1) {
        scf.for %arg6 = %c0 to %c32 step %c1 {
          %12 = arith.muli %arg4, %c8 overflow<nsw> : index
          %13 = arith.addi %12, %arg5 : index
          %14 = loom.alloc [1, 128, 128] on @L1 : memref<1x128x128xf32>
          %15 = loom.semaphore_take %14 : memref<1x128x128xf32> -> memref<1x128x128xf32>
          %16 = loom.alloc [1, 64, 128] on @L1 : memref<1x64x128xf32>
          %17 = loom.semaphore_take %16 : memref<1x64x128xf32> -> memref<1x64x128xf32>
          %18 = loom.alloc [1, 128, 128] on @L1 : memref<1x128x128xf32>
          %19 = loom.semaphore_take %18 : memref<1x128x128xf32> -> memref<1x128x128xf32>
          %20 = loom.alloc [1, 64] on @L1 : memref<1x64xf32>
          %21 = loom.semaphore_take %20 : memref<1x64xf32> -> memref<1x64xf32>
          %22 = loom.alloc [1, 64] on @L1 : memref<1x64xf32>
          %23 = loom.semaphore_take %22 : memref<1x64xf32> -> memref<1x64xf32>
          %24 = loom.alloc [1, 64] on @L1 : memref<1x64xf32>
          %25 = loom.semaphore_take %24 : memref<1x64xf32> -> memref<1x64xf32>
          %26 = loom.alloc [1, 64] on @L1 : memref<1x64xf32>
          %27 = loom.semaphore_take %26 : memref<1x64xf32> -> memref<1x64xf32>
          %28 = loom.alloc [1, 64] on @L1 : memref<1x64xf32>
          %29 = loom.semaphore_take %28 : memref<1x64xf32> -> memref<1x64xf32>
          %30 = loom.alloc [1, 64, 128] on @L1 : memref<1x64x128xf32>
          %31 = loom.semaphore_take %30 : memref<1x64x128xf32> -> memref<1x64x128xf32>
          %32 = loom.semaphore_take %30 : memref<1x64x128xf32> -> memref<1x64x128xf32>
          %33 = loom.alloc [1, 64, 128] on @L1 : memref<1x64x128xf32>
          %34 = loom.semaphore_take %33 : memref<1x64x128xf32> -> memref<1x64x128xf32>
          %35 = loom.alloc [1, 64, 128] on @L1 : memref<1x64x128xf32>
          %36 = loom.semaphore_take %35 : memref<1x64x128xf32> -> memref<1x64x128xf32>
          %37 = arith.muli %arg6, %c524288 overflow<nsw> : index
          %38 = arith.muli %13, %c8192 : index
          %39 = arith.addi %37, %38 : index
          %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%39], sizes: [1, 64, 128], strides: [524288, 128, 1] : memref<32x4096x128xf32> to memref<1x64x128xf32, strided<[524288, 128, 1], offset: ?>>
          loom.copy %reinterpret_cast, %32 src_mem_space @DRAM dst_mem_space @L1, interconnect : [], broadcast : [1, 1] : memref<1x64x128xf32, strided<[524288, 128, 1], offset: ?>>, memref<1x64x128xf32>
          linalg.fill ins(%cst_1 : f32) outs(%34 : memref<1x64x128xf32>)
          linalg.fill ins(%cst_2 : f32) outs(%21 : memref<1x64xf32>)
          linalg.fill ins(%cst_3 : f32) outs(%23 : memref<1x64xf32>)
          scf.for %arg7 = %c0 to %c32 step %c1 {
            %40 = arith.muli %arg7, %c128 : index
            %41 = arith.addi %37, %40 : index
            %reinterpret_cast_5 = memref.reinterpret_cast %arg0 to offset: [%41], sizes: [1, 128, 128], strides: [524288, 4096, 1] : memref<32x128x4096xf32> to memref<1x128x128xf32, strided<[524288, 4096, 1], offset: ?>>
            loom.copy %reinterpret_cast_5, %19 src_mem_space @DRAM dst_mem_space @L1, interconnect : [@horizontal_links, @vertical_links], broadcast : [8, 8] : memref<1x128x128xf32, strided<[524288, 4096, 1], offset: ?>>, memref<1x128x128xf32>
            linalg.fill ins(%cst_1 : f32) outs(%17 : memref<1x64x128xf32>)
            linalg.batch_matmul ins(%32, %19 : memref<1x64x128xf32>, memref<1x128x128xf32>) outs(%17 : memref<1x64x128xf32>)
            loom.semaphore_give %19 : memref<1x128x128xf32>
            linalg.fill ins(%cst_3 : f32) outs(%25 : memref<1x64xf32>)
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%17 : memref<1x64x128xf32>) outs(%25 : memref<1x64xf32>) {
            ^bb0(%in: f32, %out: f32):
              %44 = arith.maximumf %in, %out : f32
              linalg.yield %44 : f32
            }
            linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%23, %25 : memref<1x64xf32>, memref<1x64xf32>) outs(%25 : memref<1x64xf32>) {
            ^bb0(%in: f32, %in_7: f32, %out: f32):
              %44 = arith.truncf %cst_0 : f64 to f32
              %45 = arith.mulf %in_7, %44 : f32
              %46 = arith.cmpf ogt, %in, %45 : f32
              %47 = arith.select %46, %in, %45 : f32
              linalg.yield %47 : f32
            }
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%17, %25 : memref<1x64x128xf32>, memref<1x64xf32>) outs(%17 : memref<1x64x128xf32>) {
            ^bb0(%in: f32, %in_7: f32, %out: f32):
              %44 = arith.truncf %cst_0 : f64 to f32
              %45 = arith.mulf %in, %44 : f32
              %46 = arith.subf %45, %in_7 : f32
              %47 = math.powf %cst, %46 : f32
              linalg.yield %47 : f32
            }
            linalg.fill ins(%cst_1 : f32) outs(%27 : memref<1x64xf32>)
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%17 : memref<1x64x128xf32>) outs(%27 : memref<1x64xf32>) {
            ^bb0(%in: f32, %out: f32):
              %44 = arith.addf %in, %out : f32
              linalg.yield %44 : f32
            }
            linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%23, %25 : memref<1x64xf32>, memref<1x64xf32>) outs(%29 : memref<1x64xf32>) {
            ^bb0(%in: f32, %in_7: f32, %out: f32):
              %44 = arith.subf %in, %in_7 : f32
              %45 = math.powf %cst, %44 : f32
              linalg.yield %45 : f32
            }
            linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%21, %29, %27 : memref<1x64xf32>, memref<1x64xf32>, memref<1x64xf32>) outs(%21 : memref<1x64xf32>) {
            ^bb0(%in: f32, %in_7: f32, %in_8: f32, %out: f32):
              %44 = arith.mulf %in, %in_7 : f32
              %45 = arith.addf %44, %in_8 : f32
              linalg.yield %45 : f32
            }
            loom.semaphore_give %27 : memref<1x64xf32>
            %42 = arith.muli %arg7, %c16384 : index
            %43 = arith.addi %37, %42 : index
            %reinterpret_cast_6 = memref.reinterpret_cast %arg1 to offset: [%43], sizes: [1, 128, 128], strides: [524288, 128, 1] : memref<32x4096x128xf32> to memref<1x128x128xf32, strided<[524288, 128, 1], offset: ?>>
            loom.copy %reinterpret_cast_6, %15 src_mem_space @DRAM dst_mem_space @L1, interconnect : [@horizontal_links], broadcast : [1, 8] : memref<1x128x128xf32, strided<[524288, 128, 1], offset: ?>>, memref<1x128x128xf32>
            linalg.fill ins(%cst_1 : f32) outs(%36 : memref<1x64x128xf32>)
            linalg.batch_matmul ins(%17, %15 : memref<1x64x128xf32>, memref<1x128x128xf32>) outs(%36 : memref<1x64x128xf32>)
            loom.semaphore_give %15 : memref<1x128x128xf32>
            loom.semaphore_give %17 : memref<1x64x128xf32>
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%36, %34, %29 : memref<1x64x128xf32>, memref<1x64x128xf32>, memref<1x64xf32>) outs(%34 : memref<1x64x128xf32>) {
            ^bb0(%in: f32, %in_7: f32, %in_8: f32, %out: f32):
              %44 = arith.mulf %in_7, %in_8 : f32
              %45 = arith.addf %in, %44 : f32
              linalg.yield %45 : f32
            }
            loom.semaphore_give %29 : memref<1x64xf32>
            loom.semaphore_give %36 : memref<1x64x128xf32>
            linalg.copy ins(%25 : memref<1x64xf32>) outs(%23 : memref<1x64xf32>)
            loom.semaphore_give %25 : memref<1x64xf32>
          }
          loom.semaphore_give %23 : memref<1x64xf32>
          loom.semaphore_give %32 : memref<1x64x128xf32>
          linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%34, %21 : memref<1x64x128xf32>, memref<1x64xf32>) outs(%31 : memref<1x64x128xf32>) {
          ^bb0(%in: f32, %in_5: f32, %out: f32):
            %40 = arith.divf %in, %in_5 : f32
            linalg.yield %40 : f32
          }
          loom.semaphore_give %21 : memref<1x64xf32>
          loom.semaphore_give %34 : memref<1x64x128xf32>
          %reinterpret_cast_4 = memref.reinterpret_cast %arg3 to offset: [%39], sizes: [1, 64, 128], strides: [524288, 128, 1] : memref<32x4096x128xf32> to memref<1x64x128xf32, strided<[524288, 128, 1], offset: ?>>
          loom.copy %31, %reinterpret_cast_4 src_mem_space @L1 dst_mem_space @DRAM, interconnect : [], broadcast : [1, 1] : memref<1x64x128xf32>, memref<1x64x128xf32, strided<[524288, 128, 1], offset: ?>>
          loom.semaphore_give %31 : memref<1x64x128xf32>
        }
        scf.reduce 
      } {loom.iter_types = [#loom.iter_type<spatial>, #loom.iter_type<spatial>], loom.mapped_to_dims = [@y, @x]}
      return
    }
  }
  module attributes {loom.pass_name = "Materialize"} {
    func.func @attention__d1i1_d0i1__f10__d_a_v__BB1__BM64__BN128(%arg0: memref<32x128x4096xf32>, %arg1: memref<32x4096x128xf32>, %arg2: memref<32x4096x128xf32>, %arg3: memref<32x4096x128xf32>) {
      %c16384 = arith.constant 16384 : index
      %c8192 = arith.constant 8192 : index
      %c524288 = arith.constant 524288 : index
      %c32 = arith.constant 32 : index
      %cst = arith.constant 2.000000e+00 : f32
      %cst_0 = arith.constant 0.12751743074602467 : f64
      %cst_1 = arith.constant 0.000000e+00 : f32
      %cst_2 = arith.constant 1.000000e+00 : f32
      %cst_3 = arith.constant 0xFF800000 : f32
      %c128 = arith.constant 128 : index
      %c0 = arith.constant 0 : index
      %c8 = arith.constant 8 : index
      %c1 = arith.constant 1 : index
      scf.parallel (%arg4, %arg5) = (%c0, %c0) to (%c8, %c8) step (%c1, %c1) {
        scf.for %arg6 = %c0 to %c32 step %c1 {
          %12 = arith.muli %arg4, %c8 overflow<nsw> : index
          %13 = arith.addi %12, %arg5 : index
          %14 = loom.alloc [1, 128, 128] on @L1 : memref<1x128x128xf32>
          %15 = loom.semaphore_take %14 : memref<1x128x128xf32> -> memref<1x128x128xf32>
          %16 = loom.alloc [1, 64, 128] on @L1 : memref<1x64x128xf32>
          %17 = loom.semaphore_take %16 : memref<1x64x128xf32> -> memref<1x64x128xf32>
          %18 = loom.alloc [1, 128, 128] on @L1 : memref<1x128x128xf32>
          %19 = loom.semaphore_take %18 : memref<1x128x128xf32> -> memref<1x128x128xf32>
          %20 = loom.alloc [1, 64] on @L1 : memref<1x64xf32>
          %21 = loom.semaphore_take %20 : memref<1x64xf32> -> memref<1x64xf32>
          %22 = loom.alloc [1, 64] on @L1 : memref<1x64xf32>
          %23 = loom.semaphore_take %22 : memref<1x64xf32> -> memref<1x64xf32>
          %24 = loom.alloc [1, 64] on @L1 : memref<1x64xf32>
          %25 = loom.semaphore_take %24 : memref<1x64xf32> -> memref<1x64xf32>
          %26 = loom.alloc [1, 64] on @L1 : memref<1x64xf32>
          %27 = loom.semaphore_take %26 : memref<1x64xf32> -> memref<1x64xf32>
          %28 = loom.alloc [1, 64] on @L1 : memref<1x64xf32>
          %29 = loom.semaphore_take %28 : memref<1x64xf32> -> memref<1x64xf32>
          %30 = loom.alloc [1, 64, 128] on @L1 : memref<1x64x128xf32>
          %31 = loom.semaphore_take %30 : memref<1x64x128xf32> -> memref<1x64x128xf32>
          %32 = loom.semaphore_take %30 : memref<1x64x128xf32> -> memref<1x64x128xf32>
          %33 = loom.alloc [1, 64, 128] on @L1 : memref<1x64x128xf32>
          %34 = loom.semaphore_take %33 : memref<1x64x128xf32> -> memref<1x64x128xf32>
          %35 = loom.alloc [1, 64, 128] on @L1 : memref<1x64x128xf32>
          %36 = loom.semaphore_take %35 : memref<1x64x128xf32> -> memref<1x64x128xf32>
          %37 = arith.muli %arg6, %c524288 overflow<nsw> : index
          %38 = arith.muli %13, %c8192 : index
          %39 = arith.addi %37, %38 : index
          %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%39], sizes: [1, 64, 128], strides: [524288, 128, 1] : memref<32x4096x128xf32> to memref<1x64x128xf32, strided<[524288, 128, 1], offset: ?>>
          loom.copy %reinterpret_cast, %32 src_mem_space @DRAM dst_mem_space @L1, interconnect : [], broadcast : [1, 1] : memref<1x64x128xf32, strided<[524288, 128, 1], offset: ?>>, memref<1x64x128xf32>
          linalg.fill ins(%cst_1 : f32) outs(%34 : memref<1x64x128xf32>)
          linalg.fill ins(%cst_2 : f32) outs(%21 : memref<1x64xf32>)
          linalg.fill ins(%cst_3 : f32) outs(%23 : memref<1x64xf32>)
          scf.for %arg7 = %c0 to %c32 step %c1 {
            %40 = arith.muli %arg7, %c128 : index
            %41 = arith.addi %37, %40 : index
            %reinterpret_cast_5 = memref.reinterpret_cast %arg0 to offset: [%41], sizes: [1, 128, 128], strides: [524288, 4096, 1] : memref<32x128x4096xf32> to memref<1x128x128xf32, strided<[524288, 4096, 1], offset: ?>>
            loom.copy %reinterpret_cast_5, %19 src_mem_space @DRAM dst_mem_space @L1, interconnect : [@horizontal_links, @vertical_links], broadcast : [8, 8] : memref<1x128x128xf32, strided<[524288, 4096, 1], offset: ?>>, memref<1x128x128xf32>
            linalg.fill ins(%cst_1 : f32) outs(%17 : memref<1x64x128xf32>)
            linalg.batch_matmul ins(%32, %19 : memref<1x64x128xf32>, memref<1x128x128xf32>) outs(%17 : memref<1x64x128xf32>)
            loom.semaphore_give %19 : memref<1x128x128xf32>
            linalg.fill ins(%cst_3 : f32) outs(%25 : memref<1x64xf32>)
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%17 : memref<1x64x128xf32>) outs(%25 : memref<1x64xf32>) {
            ^bb0(%in: f32, %out: f32):
              %44 = arith.maximumf %in, %out : f32
              linalg.yield %44 : f32
            }
            linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%23, %25 : memref<1x64xf32>, memref<1x64xf32>) outs(%25 : memref<1x64xf32>) {
            ^bb0(%in: f32, %in_7: f32, %out: f32):
              %44 = arith.truncf %cst_0 : f64 to f32
              %45 = arith.mulf %in_7, %44 : f32
              %46 = arith.cmpf ogt, %in, %45 : f32
              %47 = arith.select %46, %in, %45 : f32
              linalg.yield %47 : f32
            }
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%17, %25 : memref<1x64x128xf32>, memref<1x64xf32>) outs(%17 : memref<1x64x128xf32>) {
            ^bb0(%in: f32, %in_7: f32, %out: f32):
              %44 = arith.truncf %cst_0 : f64 to f32
              %45 = arith.mulf %in, %44 : f32
              %46 = arith.subf %45, %in_7 : f32
              %47 = math.powf %cst, %46 : f32
              linalg.yield %47 : f32
            }
            linalg.fill ins(%cst_1 : f32) outs(%27 : memref<1x64xf32>)
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%17 : memref<1x64x128xf32>) outs(%27 : memref<1x64xf32>) {
            ^bb0(%in: f32, %out: f32):
              %44 = arith.addf %in, %out : f32
              linalg.yield %44 : f32
            }
            linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%23, %25 : memref<1x64xf32>, memref<1x64xf32>) outs(%29 : memref<1x64xf32>) {
            ^bb0(%in: f32, %in_7: f32, %out: f32):
              %44 = arith.subf %in, %in_7 : f32
              %45 = math.powf %cst, %44 : f32
              linalg.yield %45 : f32
            }
            linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%21, %29, %27 : memref<1x64xf32>, memref<1x64xf32>, memref<1x64xf32>) outs(%21 : memref<1x64xf32>) {
            ^bb0(%in: f32, %in_7: f32, %in_8: f32, %out: f32):
              %44 = arith.mulf %in, %in_7 : f32
              %45 = arith.addf %44, %in_8 : f32
              linalg.yield %45 : f32
            }
            loom.semaphore_give %27 : memref<1x64xf32>
            %42 = arith.muli %arg7, %c16384 : index
            %43 = arith.addi %37, %42 : index
            %reinterpret_cast_6 = memref.reinterpret_cast %arg1 to offset: [%43], sizes: [1, 128, 128], strides: [524288, 128, 1] : memref<32x4096x128xf32> to memref<1x128x128xf32, strided<[524288, 128, 1], offset: ?>>
            loom.copy %reinterpret_cast_6, %15 src_mem_space @DRAM dst_mem_space @L1, interconnect : [@vertical_links], broadcast : [8, 1] : memref<1x128x128xf32, strided<[524288, 128, 1], offset: ?>>, memref<1x128x128xf32>
            linalg.fill ins(%cst_1 : f32) outs(%36 : memref<1x64x128xf32>)
            linalg.batch_matmul ins(%17, %15 : memref<1x64x128xf32>, memref<1x128x128xf32>) outs(%36 : memref<1x64x128xf32>)
            loom.semaphore_give %15 : memref<1x128x128xf32>
            loom.semaphore_give %17 : memref<1x64x128xf32>
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%36, %34, %29 : memref<1x64x128xf32>, memref<1x64x128xf32>, memref<1x64xf32>) outs(%34 : memref<1x64x128xf32>) {
            ^bb0(%in: f32, %in_7: f32, %in_8: f32, %out: f32):
              %44 = arith.mulf %in_7, %in_8 : f32
              %45 = arith.addf %in, %44 : f32
              linalg.yield %45 : f32
            }
            loom.semaphore_give %29 : memref<1x64xf32>
            loom.semaphore_give %36 : memref<1x64x128xf32>
            linalg.copy ins(%25 : memref<1x64xf32>) outs(%23 : memref<1x64xf32>)
            loom.semaphore_give %25 : memref<1x64xf32>
          }
          loom.semaphore_give %23 : memref<1x64xf32>
          loom.semaphore_give %32 : memref<1x64x128xf32>
          linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%34, %21 : memref<1x64x128xf32>, memref<1x64xf32>) outs(%31 : memref<1x64x128xf32>) {
          ^bb0(%in: f32, %in_5: f32, %out: f32):
            %40 = arith.divf %in, %in_5 : f32
            linalg.yield %40 : f32
          }
          loom.semaphore_give %21 : memref<1x64xf32>
          loom.semaphore_give %34 : memref<1x64x128xf32>
          %reinterpret_cast_4 = memref.reinterpret_cast %arg3 to offset: [%39], sizes: [1, 64, 128], strides: [524288, 128, 1] : memref<32x4096x128xf32> to memref<1x64x128xf32, strided<[524288, 128, 1], offset: ?>>
          loom.copy %31, %reinterpret_cast_4 src_mem_space @L1 dst_mem_space @DRAM, interconnect : [], broadcast : [1, 1] : memref<1x64x128xf32>, memref<1x64x128xf32, strided<[524288, 128, 1], offset: ?>>
          loom.semaphore_give %31 : memref<1x64x128xf32>
        }
        scf.reduce 
      } {loom.iter_types = [#loom.iter_type<spatial>, #loom.iter_type<spatial>], loom.mapped_to_dims = [@y, @x]}
      return
    }
  }
  module attributes {loom.pass_name = "Materialize"} {
    func.func @attention__d1i1_d0i1__f10__d_h_d__BB1__BM64__BN128(%arg0: memref<32x128x4096xf32>, %arg1: memref<32x4096x128xf32>, %arg2: memref<32x4096x128xf32>, %arg3: memref<32x4096x128xf32>) {
      %c16384 = arith.constant 16384 : index
      %c8192 = arith.constant 8192 : index
      %c524288 = arith.constant 524288 : index
      %c32 = arith.constant 32 : index
      %cst = arith.constant 2.000000e+00 : f32
      %cst_0 = arith.constant 0.12751743074602467 : f64
      %cst_1 = arith.constant 0.000000e+00 : f32
      %cst_2 = arith.constant 1.000000e+00 : f32
      %cst_3 = arith.constant 0xFF800000 : f32
      %c128 = arith.constant 128 : index
      %c0 = arith.constant 0 : index
      %c8 = arith.constant 8 : index
      %c1 = arith.constant 1 : index
      scf.parallel (%arg4, %arg5) = (%c0, %c0) to (%c8, %c8) step (%c1, %c1) {
        scf.for %arg6 = %c0 to %c32 step %c1 {
          %12 = arith.muli %arg4, %c8 overflow<nsw> : index
          %13 = arith.addi %12, %arg5 : index
          %14 = loom.alloc [1, 128, 128] on @L1 : memref<1x128x128xf32>
          %15 = loom.semaphore_take %14 : memref<1x128x128xf32> -> memref<1x128x128xf32>
          %16 = loom.alloc [1, 64, 128] on @L1 : memref<1x64x128xf32>
          %17 = loom.semaphore_take %16 : memref<1x64x128xf32> -> memref<1x64x128xf32>
          %18 = loom.alloc [1, 128, 128] on @L1 : memref<1x128x128xf32>
          %19 = loom.semaphore_take %18 : memref<1x128x128xf32> -> memref<1x128x128xf32>
          %20 = loom.alloc [1, 64] on @L1 : memref<1x64xf32>
          %21 = loom.semaphore_take %20 : memref<1x64xf32> -> memref<1x64xf32>
          %22 = loom.alloc [1, 64] on @L1 : memref<1x64xf32>
          %23 = loom.semaphore_take %22 : memref<1x64xf32> -> memref<1x64xf32>
          %24 = loom.alloc [1, 64] on @L1 : memref<1x64xf32>
          %25 = loom.semaphore_take %24 : memref<1x64xf32> -> memref<1x64xf32>
          %26 = loom.alloc [1, 64] on @L1 : memref<1x64xf32>
          %27 = loom.semaphore_take %26 : memref<1x64xf32> -> memref<1x64xf32>
          %28 = loom.alloc [1, 64] on @L1 : memref<1x64xf32>
          %29 = loom.semaphore_take %28 : memref<1x64xf32> -> memref<1x64xf32>
          %30 = loom.alloc [1, 64, 128] on @L1 : memref<1x64x128xf32>
          %31 = loom.semaphore_take %30 : memref<1x64x128xf32> -> memref<1x64x128xf32>
          %32 = loom.semaphore_take %30 : memref<1x64x128xf32> -> memref<1x64x128xf32>
          %33 = loom.alloc [1, 64, 128] on @L1 : memref<1x64x128xf32>
          %34 = loom.semaphore_take %33 : memref<1x64x128xf32> -> memref<1x64x128xf32>
          %35 = loom.alloc [1, 64, 128] on @L1 : memref<1x64x128xf32>
          %36 = loom.semaphore_take %35 : memref<1x64x128xf32> -> memref<1x64x128xf32>
          %37 = arith.muli %arg6, %c524288 overflow<nsw> : index
          %38 = arith.muli %13, %c8192 : index
          %39 = arith.addi %37, %38 : index
          %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%39], sizes: [1, 64, 128], strides: [524288, 128, 1] : memref<32x4096x128xf32> to memref<1x64x128xf32, strided<[524288, 128, 1], offset: ?>>
          loom.copy %reinterpret_cast, %32 src_mem_space @DRAM dst_mem_space @L1, interconnect : [], broadcast : [1, 1] : memref<1x64x128xf32, strided<[524288, 128, 1], offset: ?>>, memref<1x64x128xf32>
          linalg.fill ins(%cst_1 : f32) outs(%34 : memref<1x64x128xf32>)
          linalg.fill ins(%cst_2 : f32) outs(%21 : memref<1x64xf32>)
          linalg.fill ins(%cst_3 : f32) outs(%23 : memref<1x64xf32>)
          scf.for %arg7 = %c0 to %c32 step %c1 {
            %40 = arith.muli %arg7, %c128 : index
            %41 = arith.addi %37, %40 : index
            %reinterpret_cast_5 = memref.reinterpret_cast %arg0 to offset: [%41], sizes: [1, 128, 128], strides: [524288, 4096, 1] : memref<32x128x4096xf32> to memref<1x128x128xf32, strided<[524288, 4096, 1], offset: ?>>
            loom.copy %reinterpret_cast_5, %19 src_mem_space @DRAM dst_mem_space @L1, interconnect : [@horizontal_links], broadcast : [1, 8] : memref<1x128x128xf32, strided<[524288, 4096, 1], offset: ?>>, memref<1x128x128xf32>
            linalg.fill ins(%cst_1 : f32) outs(%17 : memref<1x64x128xf32>)
            linalg.batch_matmul ins(%32, %19 : memref<1x64x128xf32>, memref<1x128x128xf32>) outs(%17 : memref<1x64x128xf32>)
            loom.semaphore_give %19 : memref<1x128x128xf32>
            linalg.fill ins(%cst_3 : f32) outs(%25 : memref<1x64xf32>)
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%17 : memref<1x64x128xf32>) outs(%25 : memref<1x64xf32>) {
            ^bb0(%in: f32, %out: f32):
              %44 = arith.maximumf %in, %out : f32
              linalg.yield %44 : f32
            }
            linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%23, %25 : memref<1x64xf32>, memref<1x64xf32>) outs(%25 : memref<1x64xf32>) {
            ^bb0(%in: f32, %in_7: f32, %out: f32):
              %44 = arith.truncf %cst_0 : f64 to f32
              %45 = arith.mulf %in_7, %44 : f32
              %46 = arith.cmpf ogt, %in, %45 : f32
              %47 = arith.select %46, %in, %45 : f32
              linalg.yield %47 : f32
            }
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%17, %25 : memref<1x64x128xf32>, memref<1x64xf32>) outs(%17 : memref<1x64x128xf32>) {
            ^bb0(%in: f32, %in_7: f32, %out: f32):
              %44 = arith.truncf %cst_0 : f64 to f32
              %45 = arith.mulf %in, %44 : f32
              %46 = arith.subf %45, %in_7 : f32
              %47 = math.powf %cst, %46 : f32
              linalg.yield %47 : f32
            }
            linalg.fill ins(%cst_1 : f32) outs(%27 : memref<1x64xf32>)
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%17 : memref<1x64x128xf32>) outs(%27 : memref<1x64xf32>) {
            ^bb0(%in: f32, %out: f32):
              %44 = arith.addf %in, %out : f32
              linalg.yield %44 : f32
            }
            linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%23, %25 : memref<1x64xf32>, memref<1x64xf32>) outs(%29 : memref<1x64xf32>) {
            ^bb0(%in: f32, %in_7: f32, %out: f32):
              %44 = arith.subf %in, %in_7 : f32
              %45 = math.powf %cst, %44 : f32
              linalg.yield %45 : f32
            }
            linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%21, %29, %27 : memref<1x64xf32>, memref<1x64xf32>, memref<1x64xf32>) outs(%21 : memref<1x64xf32>) {
            ^bb0(%in: f32, %in_7: f32, %in_8: f32, %out: f32):
              %44 = arith.mulf %in, %in_7 : f32
              %45 = arith.addf %44, %in_8 : f32
              linalg.yield %45 : f32
            }
            loom.semaphore_give %27 : memref<1x64xf32>
            %42 = arith.muli %arg7, %c16384 : index
            %43 = arith.addi %37, %42 : index
            %reinterpret_cast_6 = memref.reinterpret_cast %arg1 to offset: [%43], sizes: [1, 128, 128], strides: [524288, 128, 1] : memref<32x4096x128xf32> to memref<1x128x128xf32, strided<[524288, 128, 1], offset: ?>>
            loom.copy %reinterpret_cast_6, %15 src_mem_space @DRAM dst_mem_space @L1, interconnect : [], broadcast : [1, 1] : memref<1x128x128xf32, strided<[524288, 128, 1], offset: ?>>, memref<1x128x128xf32>
            linalg.fill ins(%cst_1 : f32) outs(%36 : memref<1x64x128xf32>)
            linalg.batch_matmul ins(%17, %15 : memref<1x64x128xf32>, memref<1x128x128xf32>) outs(%36 : memref<1x64x128xf32>)
            loom.semaphore_give %15 : memref<1x128x128xf32>
            loom.semaphore_give %17 : memref<1x64x128xf32>
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%36, %34, %29 : memref<1x64x128xf32>, memref<1x64x128xf32>, memref<1x64xf32>) outs(%34 : memref<1x64x128xf32>) {
            ^bb0(%in: f32, %in_7: f32, %in_8: f32, %out: f32):
              %44 = arith.mulf %in_7, %in_8 : f32
              %45 = arith.addf %in, %44 : f32
              linalg.yield %45 : f32
            }
            loom.semaphore_give %29 : memref<1x64xf32>
            loom.semaphore_give %36 : memref<1x64x128xf32>
            linalg.copy ins(%25 : memref<1x64xf32>) outs(%23 : memref<1x64xf32>)
            loom.semaphore_give %25 : memref<1x64xf32>
          }
          loom.semaphore_give %23 : memref<1x64xf32>
          loom.semaphore_give %32 : memref<1x64x128xf32>
          linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%34, %21 : memref<1x64x128xf32>, memref<1x64xf32>) outs(%31 : memref<1x64x128xf32>) {
          ^bb0(%in: f32, %in_5: f32, %out: f32):
            %40 = arith.divf %in, %in_5 : f32
            linalg.yield %40 : f32
          }
          loom.semaphore_give %21 : memref<1x64xf32>
          loom.semaphore_give %34 : memref<1x64x128xf32>
          %reinterpret_cast_4 = memref.reinterpret_cast %arg3 to offset: [%39], sizes: [1, 64, 128], strides: [524288, 128, 1] : memref<32x4096x128xf32> to memref<1x64x128xf32, strided<[524288, 128, 1], offset: ?>>
          loom.copy %31, %reinterpret_cast_4 src_mem_space @L1 dst_mem_space @DRAM, interconnect : [], broadcast : [1, 1] : memref<1x64x128xf32>, memref<1x64x128xf32, strided<[524288, 128, 1], offset: ?>>
          loom.semaphore_give %31 : memref<1x64x128xf32>
        }
        scf.reduce 
      } {loom.iter_types = [#loom.iter_type<spatial>, #loom.iter_type<spatial>], loom.mapped_to_dims = [@y, @x]}
      return
    }
  }
  module attributes {loom.pass_name = "Materialize"} {
    func.func @attention__d1i1_d0i1__f10__d_h_a__BB1__BM64__BN128(%arg0: memref<32x128x4096xf32>, %arg1: memref<32x4096x128xf32>, %arg2: memref<32x4096x128xf32>, %arg3: memref<32x4096x128xf32>) {
      %c16384 = arith.constant 16384 : index
      %c8192 = arith.constant 8192 : index
      %c524288 = arith.constant 524288 : index
      %c32 = arith.constant 32 : index
      %cst = arith.constant 2.000000e+00 : f32
      %cst_0 = arith.constant 0.12751743074602467 : f64
      %cst_1 = arith.constant 0.000000e+00 : f32
      %cst_2 = arith.constant 1.000000e+00 : f32
      %cst_3 = arith.constant 0xFF800000 : f32
      %c128 = arith.constant 128 : index
      %c0 = arith.constant 0 : index
      %c8 = arith.constant 8 : index
      %c1 = arith.constant 1 : index
      scf.parallel (%arg4, %arg5) = (%c0, %c0) to (%c8, %c8) step (%c1, %c1) {
        scf.for %arg6 = %c0 to %c32 step %c1 {
          %12 = arith.muli %arg4, %c8 overflow<nsw> : index
          %13 = arith.addi %12, %arg5 : index
          %14 = loom.alloc [1, 128, 128] on @L1 : memref<1x128x128xf32>
          %15 = loom.semaphore_take %14 : memref<1x128x128xf32> -> memref<1x128x128xf32>
          %16 = loom.alloc [1, 64, 128] on @L1 : memref<1x64x128xf32>
          %17 = loom.semaphore_take %16 : memref<1x64x128xf32> -> memref<1x64x128xf32>
          %18 = loom.alloc [1, 128, 128] on @L1 : memref<1x128x128xf32>
          %19 = loom.semaphore_take %18 : memref<1x128x128xf32> -> memref<1x128x128xf32>
          %20 = loom.alloc [1, 64] on @L1 : memref<1x64xf32>
          %21 = loom.semaphore_take %20 : memref<1x64xf32> -> memref<1x64xf32>
          %22 = loom.alloc [1, 64] on @L1 : memref<1x64xf32>
          %23 = loom.semaphore_take %22 : memref<1x64xf32> -> memref<1x64xf32>
          %24 = loom.alloc [1, 64] on @L1 : memref<1x64xf32>
          %25 = loom.semaphore_take %24 : memref<1x64xf32> -> memref<1x64xf32>
          %26 = loom.alloc [1, 64] on @L1 : memref<1x64xf32>
          %27 = loom.semaphore_take %26 : memref<1x64xf32> -> memref<1x64xf32>
          %28 = loom.alloc [1, 64] on @L1 : memref<1x64xf32>
          %29 = loom.semaphore_take %28 : memref<1x64xf32> -> memref<1x64xf32>
          %30 = loom.alloc [1, 64, 128] on @L1 : memref<1x64x128xf32>
          %31 = loom.semaphore_take %30 : memref<1x64x128xf32> -> memref<1x64x128xf32>
          %32 = loom.semaphore_take %30 : memref<1x64x128xf32> -> memref<1x64x128xf32>
          %33 = loom.alloc [1, 64, 128] on @L1 : memref<1x64x128xf32>
          %34 = loom.semaphore_take %33 : memref<1x64x128xf32> -> memref<1x64x128xf32>
          %35 = loom.alloc [1, 64, 128] on @L1 : memref<1x64x128xf32>
          %36 = loom.semaphore_take %35 : memref<1x64x128xf32> -> memref<1x64x128xf32>
          %37 = arith.muli %arg6, %c524288 overflow<nsw> : index
          %38 = arith.muli %13, %c8192 : index
          %39 = arith.addi %37, %38 : index
          %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%39], sizes: [1, 64, 128], strides: [524288, 128, 1] : memref<32x4096x128xf32> to memref<1x64x128xf32, strided<[524288, 128, 1], offset: ?>>
          loom.copy %reinterpret_cast, %32 src_mem_space @DRAM dst_mem_space @L1, interconnect : [], broadcast : [1, 1] : memref<1x64x128xf32, strided<[524288, 128, 1], offset: ?>>, memref<1x64x128xf32>
          linalg.fill ins(%cst_1 : f32) outs(%34 : memref<1x64x128xf32>)
          linalg.fill ins(%cst_2 : f32) outs(%21 : memref<1x64xf32>)
          linalg.fill ins(%cst_3 : f32) outs(%23 : memref<1x64xf32>)
          scf.for %arg7 = %c0 to %c32 step %c1 {
            %40 = arith.muli %arg7, %c128 : index
            %41 = arith.addi %37, %40 : index
            %reinterpret_cast_5 = memref.reinterpret_cast %arg0 to offset: [%41], sizes: [1, 128, 128], strides: [524288, 4096, 1] : memref<32x128x4096xf32> to memref<1x128x128xf32, strided<[524288, 4096, 1], offset: ?>>
            loom.copy %reinterpret_cast_5, %19 src_mem_space @DRAM dst_mem_space @L1, interconnect : [@horizontal_links], broadcast : [1, 8] : memref<1x128x128xf32, strided<[524288, 4096, 1], offset: ?>>, memref<1x128x128xf32>
            linalg.fill ins(%cst_1 : f32) outs(%17 : memref<1x64x128xf32>)
            linalg.batch_matmul ins(%32, %19 : memref<1x64x128xf32>, memref<1x128x128xf32>) outs(%17 : memref<1x64x128xf32>)
            loom.semaphore_give %19 : memref<1x128x128xf32>
            linalg.fill ins(%cst_3 : f32) outs(%25 : memref<1x64xf32>)
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%17 : memref<1x64x128xf32>) outs(%25 : memref<1x64xf32>) {
            ^bb0(%in: f32, %out: f32):
              %44 = arith.maximumf %in, %out : f32
              linalg.yield %44 : f32
            }
            linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%23, %25 : memref<1x64xf32>, memref<1x64xf32>) outs(%25 : memref<1x64xf32>) {
            ^bb0(%in: f32, %in_7: f32, %out: f32):
              %44 = arith.truncf %cst_0 : f64 to f32
              %45 = arith.mulf %in_7, %44 : f32
              %46 = arith.cmpf ogt, %in, %45 : f32
              %47 = arith.select %46, %in, %45 : f32
              linalg.yield %47 : f32
            }
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%17, %25 : memref<1x64x128xf32>, memref<1x64xf32>) outs(%17 : memref<1x64x128xf32>) {
            ^bb0(%in: f32, %in_7: f32, %out: f32):
              %44 = arith.truncf %cst_0 : f64 to f32
              %45 = arith.mulf %in, %44 : f32
              %46 = arith.subf %45, %in_7 : f32
              %47 = math.powf %cst, %46 : f32
              linalg.yield %47 : f32
            }
            linalg.fill ins(%cst_1 : f32) outs(%27 : memref<1x64xf32>)
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%17 : memref<1x64x128xf32>) outs(%27 : memref<1x64xf32>) {
            ^bb0(%in: f32, %out: f32):
              %44 = arith.addf %in, %out : f32
              linalg.yield %44 : f32
            }
            linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%23, %25 : memref<1x64xf32>, memref<1x64xf32>) outs(%29 : memref<1x64xf32>) {
            ^bb0(%in: f32, %in_7: f32, %out: f32):
              %44 = arith.subf %in, %in_7 : f32
              %45 = math.powf %cst, %44 : f32
              linalg.yield %45 : f32
            }
            linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%21, %29, %27 : memref<1x64xf32>, memref<1x64xf32>, memref<1x64xf32>) outs(%21 : memref<1x64xf32>) {
            ^bb0(%in: f32, %in_7: f32, %in_8: f32, %out: f32):
              %44 = arith.mulf %in, %in_7 : f32
              %45 = arith.addf %44, %in_8 : f32
              linalg.yield %45 : f32
            }
            loom.semaphore_give %27 : memref<1x64xf32>
            %42 = arith.muli %arg7, %c16384 : index
            %43 = arith.addi %37, %42 : index
            %reinterpret_cast_6 = memref.reinterpret_cast %arg1 to offset: [%43], sizes: [1, 128, 128], strides: [524288, 128, 1] : memref<32x4096x128xf32> to memref<1x128x128xf32, strided<[524288, 128, 1], offset: ?>>
            loom.copy %reinterpret_cast_6, %15 src_mem_space @DRAM dst_mem_space @L1, interconnect : [@horizontal_links, @vertical_links], broadcast : [8, 8] : memref<1x128x128xf32, strided<[524288, 128, 1], offset: ?>>, memref<1x128x128xf32>
            linalg.fill ins(%cst_1 : f32) outs(%36 : memref<1x64x128xf32>)
            linalg.batch_matmul ins(%17, %15 : memref<1x64x128xf32>, memref<1x128x128xf32>) outs(%36 : memref<1x64x128xf32>)
            loom.semaphore_give %15 : memref<1x128x128xf32>
            loom.semaphore_give %17 : memref<1x64x128xf32>
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%36, %34, %29 : memref<1x64x128xf32>, memref<1x64x128xf32>, memref<1x64xf32>) outs(%34 : memref<1x64x128xf32>) {
            ^bb0(%in: f32, %in_7: f32, %in_8: f32, %out: f32):
              %44 = arith.mulf %in_7, %in_8 : f32
              %45 = arith.addf %in, %44 : f32
              linalg.yield %45 : f32
            }
            loom.semaphore_give %29 : memref<1x64xf32>
            loom.semaphore_give %36 : memref<1x64x128xf32>
            linalg.copy ins(%25 : memref<1x64xf32>) outs(%23 : memref<1x64xf32>)
            loom.semaphore_give %25 : memref<1x64xf32>
          }
          loom.semaphore_give %23 : memref<1x64xf32>
          loom.semaphore_give %32 : memref<1x64x128xf32>
          linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%34, %21 : memref<1x64x128xf32>, memref<1x64xf32>) outs(%31 : memref<1x64x128xf32>) {
          ^bb0(%in: f32, %in_5: f32, %out: f32):
            %40 = arith.divf %in, %in_5 : f32
            linalg.yield %40 : f32
          }
          loom.semaphore_give %21 : memref<1x64xf32>
          loom.semaphore_give %34 : memref<1x64x128xf32>
          %reinterpret_cast_4 = memref.reinterpret_cast %arg3 to offset: [%39], sizes: [1, 64, 128], strides: [524288, 128, 1] : memref<32x4096x128xf32> to memref<1x64x128xf32, strided<[524288, 128, 1], offset: ?>>
          loom.copy %31, %reinterpret_cast_4 src_mem_space @L1 dst_mem_space @DRAM, interconnect : [], broadcast : [1, 1] : memref<1x64x128xf32>, memref<1x64x128xf32, strided<[524288, 128, 1], offset: ?>>
          loom.semaphore_give %31 : memref<1x64x128xf32>
        }
        scf.reduce 
      } {loom.iter_types = [#loom.iter_type<spatial>, #loom.iter_type<spatial>], loom.mapped_to_dims = [@y, @x]}
      return
    }
  }
  module attributes {loom.pass_name = "Materialize"} {
    func.func @attention__d1i1_d0i1__f10__d_h_h__BB1__BM64__BN128(%arg0: memref<32x128x4096xf32>, %arg1: memref<32x4096x128xf32>, %arg2: memref<32x4096x128xf32>, %arg3: memref<32x4096x128xf32>) {
      %c16384 = arith.constant 16384 : index
      %c8192 = arith.constant 8192 : index
      %c524288 = arith.constant 524288 : index
      %c32 = arith.constant 32 : index
      %cst = arith.constant 2.000000e+00 : f32
      %cst_0 = arith.constant 0.12751743074602467 : f64
      %cst_1 = arith.constant 0.000000e+00 : f32
      %cst_2 = arith.constant 1.000000e+00 : f32
      %cst_3 = arith.constant 0xFF800000 : f32
      %c128 = arith.constant 128 : index
      %c0 = arith.constant 0 : index
      %c8 = arith.constant 8 : index
      %c1 = arith.constant 1 : index
      scf.parallel (%arg4, %arg5) = (%c0, %c0) to (%c8, %c8) step (%c1, %c1) {
        scf.for %arg6 = %c0 to %c32 step %c1 {
          %12 = arith.muli %arg4, %c8 overflow<nsw> : index
          %13 = arith.addi %12, %arg5 : index
          %14 = loom.alloc [1, 128, 128] on @L1 : memref<1x128x128xf32>
          %15 = loom.semaphore_take %14 : memref<1x128x128xf32> -> memref<1x128x128xf32>
          %16 = loom.alloc [1, 64, 128] on @L1 : memref<1x64x128xf32>
          %17 = loom.semaphore_take %16 : memref<1x64x128xf32> -> memref<1x64x128xf32>
          %18 = loom.alloc [1, 128, 128] on @L1 : memref<1x128x128xf32>
          %19 = loom.semaphore_take %18 : memref<1x128x128xf32> -> memref<1x128x128xf32>
          %20 = loom.alloc [1, 64] on @L1 : memref<1x64xf32>
          %21 = loom.semaphore_take %20 : memref<1x64xf32> -> memref<1x64xf32>
          %22 = loom.alloc [1, 64] on @L1 : memref<1x64xf32>
          %23 = loom.semaphore_take %22 : memref<1x64xf32> -> memref<1x64xf32>
          %24 = loom.alloc [1, 64] on @L1 : memref<1x64xf32>
          %25 = loom.semaphore_take %24 : memref<1x64xf32> -> memref<1x64xf32>
          %26 = loom.alloc [1, 64] on @L1 : memref<1x64xf32>
          %27 = loom.semaphore_take %26 : memref<1x64xf32> -> memref<1x64xf32>
          %28 = loom.alloc [1, 64] on @L1 : memref<1x64xf32>
          %29 = loom.semaphore_take %28 : memref<1x64xf32> -> memref<1x64xf32>
          %30 = loom.alloc [1, 64, 128] on @L1 : memref<1x64x128xf32>
          %31 = loom.semaphore_take %30 : memref<1x64x128xf32> -> memref<1x64x128xf32>
          %32 = loom.semaphore_take %30 : memref<1x64x128xf32> -> memref<1x64x128xf32>
          %33 = loom.alloc [1, 64, 128] on @L1 : memref<1x64x128xf32>
          %34 = loom.semaphore_take %33 : memref<1x64x128xf32> -> memref<1x64x128xf32>
          %35 = loom.alloc [1, 64, 128] on @L1 : memref<1x64x128xf32>
          %36 = loom.semaphore_take %35 : memref<1x64x128xf32> -> memref<1x64x128xf32>
          %37 = arith.muli %arg6, %c524288 overflow<nsw> : index
          %38 = arith.muli %13, %c8192 : index
          %39 = arith.addi %37, %38 : index
          %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%39], sizes: [1, 64, 128], strides: [524288, 128, 1] : memref<32x4096x128xf32> to memref<1x64x128xf32, strided<[524288, 128, 1], offset: ?>>
          loom.copy %reinterpret_cast, %32 src_mem_space @DRAM dst_mem_space @L1, interconnect : [], broadcast : [1, 1] : memref<1x64x128xf32, strided<[524288, 128, 1], offset: ?>>, memref<1x64x128xf32>
          linalg.fill ins(%cst_1 : f32) outs(%34 : memref<1x64x128xf32>)
          linalg.fill ins(%cst_2 : f32) outs(%21 : memref<1x64xf32>)
          linalg.fill ins(%cst_3 : f32) outs(%23 : memref<1x64xf32>)
          scf.for %arg7 = %c0 to %c32 step %c1 {
            %40 = arith.muli %arg7, %c128 : index
            %41 = arith.addi %37, %40 : index
            %reinterpret_cast_5 = memref.reinterpret_cast %arg0 to offset: [%41], sizes: [1, 128, 128], strides: [524288, 4096, 1] : memref<32x128x4096xf32> to memref<1x128x128xf32, strided<[524288, 4096, 1], offset: ?>>
            loom.copy %reinterpret_cast_5, %19 src_mem_space @DRAM dst_mem_space @L1, interconnect : [@horizontal_links], broadcast : [1, 8] : memref<1x128x128xf32, strided<[524288, 4096, 1], offset: ?>>, memref<1x128x128xf32>
            linalg.fill ins(%cst_1 : f32) outs(%17 : memref<1x64x128xf32>)
            linalg.batch_matmul ins(%32, %19 : memref<1x64x128xf32>, memref<1x128x128xf32>) outs(%17 : memref<1x64x128xf32>)
            loom.semaphore_give %19 : memref<1x128x128xf32>
            linalg.fill ins(%cst_3 : f32) outs(%25 : memref<1x64xf32>)
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%17 : memref<1x64x128xf32>) outs(%25 : memref<1x64xf32>) {
            ^bb0(%in: f32, %out: f32):
              %44 = arith.maximumf %in, %out : f32
              linalg.yield %44 : f32
            }
            linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%23, %25 : memref<1x64xf32>, memref<1x64xf32>) outs(%25 : memref<1x64xf32>) {
            ^bb0(%in: f32, %in_7: f32, %out: f32):
              %44 = arith.truncf %cst_0 : f64 to f32
              %45 = arith.mulf %in_7, %44 : f32
              %46 = arith.cmpf ogt, %in, %45 : f32
              %47 = arith.select %46, %in, %45 : f32
              linalg.yield %47 : f32
            }
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%17, %25 : memref<1x64x128xf32>, memref<1x64xf32>) outs(%17 : memref<1x64x128xf32>) {
            ^bb0(%in: f32, %in_7: f32, %out: f32):
              %44 = arith.truncf %cst_0 : f64 to f32
              %45 = arith.mulf %in, %44 : f32
              %46 = arith.subf %45, %in_7 : f32
              %47 = math.powf %cst, %46 : f32
              linalg.yield %47 : f32
            }
            linalg.fill ins(%cst_1 : f32) outs(%27 : memref<1x64xf32>)
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%17 : memref<1x64x128xf32>) outs(%27 : memref<1x64xf32>) {
            ^bb0(%in: f32, %out: f32):
              %44 = arith.addf %in, %out : f32
              linalg.yield %44 : f32
            }
            linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%23, %25 : memref<1x64xf32>, memref<1x64xf32>) outs(%29 : memref<1x64xf32>) {
            ^bb0(%in: f32, %in_7: f32, %out: f32):
              %44 = arith.subf %in, %in_7 : f32
              %45 = math.powf %cst, %44 : f32
              linalg.yield %45 : f32
            }
            linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%21, %29, %27 : memref<1x64xf32>, memref<1x64xf32>, memref<1x64xf32>) outs(%21 : memref<1x64xf32>) {
            ^bb0(%in: f32, %in_7: f32, %in_8: f32, %out: f32):
              %44 = arith.mulf %in, %in_7 : f32
              %45 = arith.addf %44, %in_8 : f32
              linalg.yield %45 : f32
            }
            loom.semaphore_give %27 : memref<1x64xf32>
            %42 = arith.muli %arg7, %c16384 : index
            %43 = arith.addi %37, %42 : index
            %reinterpret_cast_6 = memref.reinterpret_cast %arg1 to offset: [%43], sizes: [1, 128, 128], strides: [524288, 128, 1] : memref<32x4096x128xf32> to memref<1x128x128xf32, strided<[524288, 128, 1], offset: ?>>
            loom.copy %reinterpret_cast_6, %15 src_mem_space @DRAM dst_mem_space @L1, interconnect : [@horizontal_links], broadcast : [1, 8] : memref<1x128x128xf32, strided<[524288, 128, 1], offset: ?>>, memref<1x128x128xf32>
            linalg.fill ins(%cst_1 : f32) outs(%36 : memref<1x64x128xf32>)
            linalg.batch_matmul ins(%17, %15 : memref<1x64x128xf32>, memref<1x128x128xf32>) outs(%36 : memref<1x64x128xf32>)
            loom.semaphore_give %15 : memref<1x128x128xf32>
            loom.semaphore_give %17 : memref<1x64x128xf32>
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%36, %34, %29 : memref<1x64x128xf32>, memref<1x64x128xf32>, memref<1x64xf32>) outs(%34 : memref<1x64x128xf32>) {
            ^bb0(%in: f32, %in_7: f32, %in_8: f32, %out: f32):
              %44 = arith.mulf %in_7, %in_8 : f32
              %45 = arith.addf %in, %44 : f32
              linalg.yield %45 : f32
            }
            loom.semaphore_give %29 : memref<1x64xf32>
            loom.semaphore_give %36 : memref<1x64x128xf32>
            linalg.copy ins(%25 : memref<1x64xf32>) outs(%23 : memref<1x64xf32>)
            loom.semaphore_give %25 : memref<1x64xf32>
          }
          loom.semaphore_give %23 : memref<1x64xf32>
          loom.semaphore_give %32 : memref<1x64x128xf32>
          linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%34, %21 : memref<1x64x128xf32>, memref<1x64xf32>) outs(%31 : memref<1x64x128xf32>) {
          ^bb0(%in: f32, %in_5: f32, %out: f32):
            %40 = arith.divf %in, %in_5 : f32
            linalg.yield %40 : f32
          }
          loom.semaphore_give %21 : memref<1x64xf32>
          loom.semaphore_give %34 : memref<1x64x128xf32>
          %reinterpret_cast_4 = memref.reinterpret_cast %arg3 to offset: [%39], sizes: [1, 64, 128], strides: [524288, 128, 1] : memref<32x4096x128xf32> to memref<1x64x128xf32, strided<[524288, 128, 1], offset: ?>>
          loom.copy %31, %reinterpret_cast_4 src_mem_space @L1 dst_mem_space @DRAM, interconnect : [], broadcast : [1, 1] : memref<1x64x128xf32>, memref<1x64x128xf32, strided<[524288, 128, 1], offset: ?>>
          loom.semaphore_give %31 : memref<1x64x128xf32>
        }
        scf.reduce 
      } {loom.iter_types = [#loom.iter_type<spatial>, #loom.iter_type<spatial>], loom.mapped_to_dims = [@y, @x]}
      return
    }
  }
  module attributes {loom.pass_name = "Materialize"} {
    func.func @attention__d1i1_d0i1__f10__d_h_v__BB1__BM64__BN128(%arg0: memref<32x128x4096xf32>, %arg1: memref<32x4096x128xf32>, %arg2: memref<32x4096x128xf32>, %arg3: memref<32x4096x128xf32>) {
      %c16384 = arith.constant 16384 : index
      %c8192 = arith.constant 8192 : index
      %c524288 = arith.constant 524288 : index
      %c32 = arith.constant 32 : index
      %cst = arith.constant 2.000000e+00 : f32
      %cst_0 = arith.constant 0.12751743074602467 : f64
      %cst_1 = arith.constant 0.000000e+00 : f32
      %cst_2 = arith.constant 1.000000e+00 : f32
      %cst_3 = arith.constant 0xFF800000 : f32
      %c128 = arith.constant 128 : index
      %c0 = arith.constant 0 : index
      %c8 = arith.constant 8 : index
      %c1 = arith.constant 1 : index
      scf.parallel (%arg4, %arg5) = (%c0, %c0) to (%c8, %c8) step (%c1, %c1) {
        scf.for %arg6 = %c0 to %c32 step %c1 {
          %12 = arith.muli %arg4, %c8 overflow<nsw> : index
          %13 = arith.addi %12, %arg5 : index
          %14 = loom.alloc [1, 128, 128] on @L1 : memref<1x128x128xf32>
          %15 = loom.semaphore_take %14 : memref<1x128x128xf32> -> memref<1x128x128xf32>
          %16 = loom.alloc [1, 64, 128] on @L1 : memref<1x64x128xf32>
          %17 = loom.semaphore_take %16 : memref<1x64x128xf32> -> memref<1x64x128xf32>
          %18 = loom.alloc [1, 128, 128] on @L1 : memref<1x128x128xf32>
          %19 = loom.semaphore_take %18 : memref<1x128x128xf32> -> memref<1x128x128xf32>
          %20 = loom.alloc [1, 64] on @L1 : memref<1x64xf32>
          %21 = loom.semaphore_take %20 : memref<1x64xf32> -> memref<1x64xf32>
          %22 = loom.alloc [1, 64] on @L1 : memref<1x64xf32>
          %23 = loom.semaphore_take %22 : memref<1x64xf32> -> memref<1x64xf32>
          %24 = loom.alloc [1, 64] on @L1 : memref<1x64xf32>
          %25 = loom.semaphore_take %24 : memref<1x64xf32> -> memref<1x64xf32>
          %26 = loom.alloc [1, 64] on @L1 : memref<1x64xf32>
          %27 = loom.semaphore_take %26 : memref<1x64xf32> -> memref<1x64xf32>
          %28 = loom.alloc [1, 64] on @L1 : memref<1x64xf32>
          %29 = loom.semaphore_take %28 : memref<1x64xf32> -> memref<1x64xf32>
          %30 = loom.alloc [1, 64, 128] on @L1 : memref<1x64x128xf32>
          %31 = loom.semaphore_take %30 : memref<1x64x128xf32> -> memref<1x64x128xf32>
          %32 = loom.semaphore_take %30 : memref<1x64x128xf32> -> memref<1x64x128xf32>
          %33 = loom.alloc [1, 64, 128] on @L1 : memref<1x64x128xf32>
          %34 = loom.semaphore_take %33 : memref<1x64x128xf32> -> memref<1x64x128xf32>
          %35 = loom.alloc [1, 64, 128] on @L1 : memref<1x64x128xf32>
          %36 = loom.semaphore_take %35 : memref<1x64x128xf32> -> memref<1x64x128xf32>
          %37 = arith.muli %arg6, %c524288 overflow<nsw> : index
          %38 = arith.muli %13, %c8192 : index
          %39 = arith.addi %37, %38 : index
          %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%39], sizes: [1, 64, 128], strides: [524288, 128, 1] : memref<32x4096x128xf32> to memref<1x64x128xf32, strided<[524288, 128, 1], offset: ?>>
          loom.copy %reinterpret_cast, %32 src_mem_space @DRAM dst_mem_space @L1, interconnect : [], broadcast : [1, 1] : memref<1x64x128xf32, strided<[524288, 128, 1], offset: ?>>, memref<1x64x128xf32>
          linalg.fill ins(%cst_1 : f32) outs(%34 : memref<1x64x128xf32>)
          linalg.fill ins(%cst_2 : f32) outs(%21 : memref<1x64xf32>)
          linalg.fill ins(%cst_3 : f32) outs(%23 : memref<1x64xf32>)
          scf.for %arg7 = %c0 to %c32 step %c1 {
            %40 = arith.muli %arg7, %c128 : index
            %41 = arith.addi %37, %40 : index
            %reinterpret_cast_5 = memref.reinterpret_cast %arg0 to offset: [%41], sizes: [1, 128, 128], strides: [524288, 4096, 1] : memref<32x128x4096xf32> to memref<1x128x128xf32, strided<[524288, 4096, 1], offset: ?>>
            loom.copy %reinterpret_cast_5, %19 src_mem_space @DRAM dst_mem_space @L1, interconnect : [@horizontal_links], broadcast : [1, 8] : memref<1x128x128xf32, strided<[524288, 4096, 1], offset: ?>>, memref<1x128x128xf32>
            linalg.fill ins(%cst_1 : f32) outs(%17 : memref<1x64x128xf32>)
            linalg.batch_matmul ins(%32, %19 : memref<1x64x128xf32>, memref<1x128x128xf32>) outs(%17 : memref<1x64x128xf32>)
            loom.semaphore_give %19 : memref<1x128x128xf32>
            linalg.fill ins(%cst_3 : f32) outs(%25 : memref<1x64xf32>)
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%17 : memref<1x64x128xf32>) outs(%25 : memref<1x64xf32>) {
            ^bb0(%in: f32, %out: f32):
              %44 = arith.maximumf %in, %out : f32
              linalg.yield %44 : f32
            }
            linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%23, %25 : memref<1x64xf32>, memref<1x64xf32>) outs(%25 : memref<1x64xf32>) {
            ^bb0(%in: f32, %in_7: f32, %out: f32):
              %44 = arith.truncf %cst_0 : f64 to f32
              %45 = arith.mulf %in_7, %44 : f32
              %46 = arith.cmpf ogt, %in, %45 : f32
              %47 = arith.select %46, %in, %45 : f32
              linalg.yield %47 : f32
            }
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%17, %25 : memref<1x64x128xf32>, memref<1x64xf32>) outs(%17 : memref<1x64x128xf32>) {
            ^bb0(%in: f32, %in_7: f32, %out: f32):
              %44 = arith.truncf %cst_0 : f64 to f32
              %45 = arith.mulf %in, %44 : f32
              %46 = arith.subf %45, %in_7 : f32
              %47 = math.powf %cst, %46 : f32
              linalg.yield %47 : f32
            }
            linalg.fill ins(%cst_1 : f32) outs(%27 : memref<1x64xf32>)
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%17 : memref<1x64x128xf32>) outs(%27 : memref<1x64xf32>) {
            ^bb0(%in: f32, %out: f32):
              %44 = arith.addf %in, %out : f32
              linalg.yield %44 : f32
            }
            linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%23, %25 : memref<1x64xf32>, memref<1x64xf32>) outs(%29 : memref<1x64xf32>) {
            ^bb0(%in: f32, %in_7: f32, %out: f32):
              %44 = arith.subf %in, %in_7 : f32
              %45 = math.powf %cst, %44 : f32
              linalg.yield %45 : f32
            }
            linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%21, %29, %27 : memref<1x64xf32>, memref<1x64xf32>, memref<1x64xf32>) outs(%21 : memref<1x64xf32>) {
            ^bb0(%in: f32, %in_7: f32, %in_8: f32, %out: f32):
              %44 = arith.mulf %in, %in_7 : f32
              %45 = arith.addf %44, %in_8 : f32
              linalg.yield %45 : f32
            }
            loom.semaphore_give %27 : memref<1x64xf32>
            %42 = arith.muli %arg7, %c16384 : index
            %43 = arith.addi %37, %42 : index
            %reinterpret_cast_6 = memref.reinterpret_cast %arg1 to offset: [%43], sizes: [1, 128, 128], strides: [524288, 128, 1] : memref<32x4096x128xf32> to memref<1x128x128xf32, strided<[524288, 128, 1], offset: ?>>
            loom.copy %reinterpret_cast_6, %15 src_mem_space @DRAM dst_mem_space @L1, interconnect : [@vertical_links], broadcast : [8, 1] : memref<1x128x128xf32, strided<[524288, 128, 1], offset: ?>>, memref<1x128x128xf32>
            linalg.fill ins(%cst_1 : f32) outs(%36 : memref<1x64x128xf32>)
            linalg.batch_matmul ins(%17, %15 : memref<1x64x128xf32>, memref<1x128x128xf32>) outs(%36 : memref<1x64x128xf32>)
            loom.semaphore_give %15 : memref<1x128x128xf32>
            loom.semaphore_give %17 : memref<1x64x128xf32>
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%36, %34, %29 : memref<1x64x128xf32>, memref<1x64x128xf32>, memref<1x64xf32>) outs(%34 : memref<1x64x128xf32>) {
            ^bb0(%in: f32, %in_7: f32, %in_8: f32, %out: f32):
              %44 = arith.mulf %in_7, %in_8 : f32
              %45 = arith.addf %in, %44 : f32
              linalg.yield %45 : f32
            }
            loom.semaphore_give %29 : memref<1x64xf32>
            loom.semaphore_give %36 : memref<1x64x128xf32>
            linalg.copy ins(%25 : memref<1x64xf32>) outs(%23 : memref<1x64xf32>)
            loom.semaphore_give %25 : memref<1x64xf32>
          }
          loom.semaphore_give %23 : memref<1x64xf32>
          loom.semaphore_give %32 : memref<1x64x128xf32>
          linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%34, %21 : memref<1x64x128xf32>, memref<1x64xf32>) outs(%31 : memref<1x64x128xf32>) {
          ^bb0(%in: f32, %in_5: f32, %out: f32):
            %40 = arith.divf %in, %in_5 : f32
            linalg.yield %40 : f32
          }
          loom.semaphore_give %21 : memref<1x64xf32>
          loom.semaphore_give %34 : memref<1x64x128xf32>
          %reinterpret_cast_4 = memref.reinterpret_cast %arg3 to offset: [%39], sizes: [1, 64, 128], strides: [524288, 128, 1] : memref<32x4096x128xf32> to memref<1x64x128xf32, strided<[524288, 128, 1], offset: ?>>
          loom.copy %31, %reinterpret_cast_4 src_mem_space @L1 dst_mem_space @DRAM, interconnect : [], broadcast : [1, 1] : memref<1x64x128xf32>, memref<1x64x128xf32, strided<[524288, 128, 1], offset: ?>>
          loom.semaphore_give %31 : memref<1x64x128xf32>
        }
        scf.reduce 
      } {loom.iter_types = [#loom.iter_type<spatial>, #loom.iter_type<spatial>], loom.mapped_to_dims = [@y, @x]}
      return
    }
  }
  module attributes {loom.pass_name = "Materialize"} {
    func.func @attention__d1i1_d0i1__f10__d_v_d__BB1__BM64__BN128(%arg0: memref<32x128x4096xf32>, %arg1: memref<32x4096x128xf32>, %arg2: memref<32x4096x128xf32>, %arg3: memref<32x4096x128xf32>) {
      %c16384 = arith.constant 16384 : index
      %c8192 = arith.constant 8192 : index
      %c524288 = arith.constant 524288 : index
      %c32 = arith.constant 32 : index
      %cst = arith.constant 2.000000e+00 : f32
      %cst_0 = arith.constant 0.12751743074602467 : f64
      %cst_1 = arith.constant 0.000000e+00 : f32
      %cst_2 = arith.constant 1.000000e+00 : f32
      %cst_3 = arith.constant 0xFF800000 : f32
      %c128 = arith.constant 128 : index
      %c0 = arith.constant 0 : index
      %c8 = arith.constant 8 : index
      %c1 = arith.constant 1 : index
      scf.parallel (%arg4, %arg5) = (%c0, %c0) to (%c8, %c8) step (%c1, %c1) {
        scf.for %arg6 = %c0 to %c32 step %c1 {
          %12 = arith.muli %arg4, %c8 overflow<nsw> : index
          %13 = arith.addi %12, %arg5 : index
          %14 = loom.alloc [1, 128, 128] on @L1 : memref<1x128x128xf32>
          %15 = loom.semaphore_take %14 : memref<1x128x128xf32> -> memref<1x128x128xf32>
          %16 = loom.alloc [1, 64, 128] on @L1 : memref<1x64x128xf32>
          %17 = loom.semaphore_take %16 : memref<1x64x128xf32> -> memref<1x64x128xf32>
          %18 = loom.alloc [1, 128, 128] on @L1 : memref<1x128x128xf32>
          %19 = loom.semaphore_take %18 : memref<1x128x128xf32> -> memref<1x128x128xf32>
          %20 = loom.alloc [1, 64] on @L1 : memref<1x64xf32>
          %21 = loom.semaphore_take %20 : memref<1x64xf32> -> memref<1x64xf32>
          %22 = loom.alloc [1, 64] on @L1 : memref<1x64xf32>
          %23 = loom.semaphore_take %22 : memref<1x64xf32> -> memref<1x64xf32>
          %24 = loom.alloc [1, 64] on @L1 : memref<1x64xf32>
          %25 = loom.semaphore_take %24 : memref<1x64xf32> -> memref<1x64xf32>
          %26 = loom.alloc [1, 64] on @L1 : memref<1x64xf32>
          %27 = loom.semaphore_take %26 : memref<1x64xf32> -> memref<1x64xf32>
          %28 = loom.alloc [1, 64] on @L1 : memref<1x64xf32>
          %29 = loom.semaphore_take %28 : memref<1x64xf32> -> memref<1x64xf32>
          %30 = loom.alloc [1, 64, 128] on @L1 : memref<1x64x128xf32>
          %31 = loom.semaphore_take %30 : memref<1x64x128xf32> -> memref<1x64x128xf32>
          %32 = loom.semaphore_take %30 : memref<1x64x128xf32> -> memref<1x64x128xf32>
          %33 = loom.alloc [1, 64, 128] on @L1 : memref<1x64x128xf32>
          %34 = loom.semaphore_take %33 : memref<1x64x128xf32> -> memref<1x64x128xf32>
          %35 = loom.alloc [1, 64, 128] on @L1 : memref<1x64x128xf32>
          %36 = loom.semaphore_take %35 : memref<1x64x128xf32> -> memref<1x64x128xf32>
          %37 = arith.muli %arg6, %c524288 overflow<nsw> : index
          %38 = arith.muli %13, %c8192 : index
          %39 = arith.addi %37, %38 : index
          %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%39], sizes: [1, 64, 128], strides: [524288, 128, 1] : memref<32x4096x128xf32> to memref<1x64x128xf32, strided<[524288, 128, 1], offset: ?>>
          loom.copy %reinterpret_cast, %32 src_mem_space @DRAM dst_mem_space @L1, interconnect : [], broadcast : [1, 1] : memref<1x64x128xf32, strided<[524288, 128, 1], offset: ?>>, memref<1x64x128xf32>
          linalg.fill ins(%cst_1 : f32) outs(%34 : memref<1x64x128xf32>)
          linalg.fill ins(%cst_2 : f32) outs(%21 : memref<1x64xf32>)
          linalg.fill ins(%cst_3 : f32) outs(%23 : memref<1x64xf32>)
          scf.for %arg7 = %c0 to %c32 step %c1 {
            %40 = arith.muli %arg7, %c128 : index
            %41 = arith.addi %37, %40 : index
            %reinterpret_cast_5 = memref.reinterpret_cast %arg0 to offset: [%41], sizes: [1, 128, 128], strides: [524288, 4096, 1] : memref<32x128x4096xf32> to memref<1x128x128xf32, strided<[524288, 4096, 1], offset: ?>>
            loom.copy %reinterpret_cast_5, %19 src_mem_space @DRAM dst_mem_space @L1, interconnect : [@vertical_links], broadcast : [8, 1] : memref<1x128x128xf32, strided<[524288, 4096, 1], offset: ?>>, memref<1x128x128xf32>
            linalg.fill ins(%cst_1 : f32) outs(%17 : memref<1x64x128xf32>)
            linalg.batch_matmul ins(%32, %19 : memref<1x64x128xf32>, memref<1x128x128xf32>) outs(%17 : memref<1x64x128xf32>)
            loom.semaphore_give %19 : memref<1x128x128xf32>
            linalg.fill ins(%cst_3 : f32) outs(%25 : memref<1x64xf32>)
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%17 : memref<1x64x128xf32>) outs(%25 : memref<1x64xf32>) {
            ^bb0(%in: f32, %out: f32):
              %44 = arith.maximumf %in, %out : f32
              linalg.yield %44 : f32
            }
            linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%23, %25 : memref<1x64xf32>, memref<1x64xf32>) outs(%25 : memref<1x64xf32>) {
            ^bb0(%in: f32, %in_7: f32, %out: f32):
              %44 = arith.truncf %cst_0 : f64 to f32
              %45 = arith.mulf %in_7, %44 : f32
              %46 = arith.cmpf ogt, %in, %45 : f32
              %47 = arith.select %46, %in, %45 : f32
              linalg.yield %47 : f32
            }
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%17, %25 : memref<1x64x128xf32>, memref<1x64xf32>) outs(%17 : memref<1x64x128xf32>) {
            ^bb0(%in: f32, %in_7: f32, %out: f32):
              %44 = arith.truncf %cst_0 : f64 to f32
              %45 = arith.mulf %in, %44 : f32
              %46 = arith.subf %45, %in_7 : f32
              %47 = math.powf %cst, %46 : f32
              linalg.yield %47 : f32
            }
            linalg.fill ins(%cst_1 : f32) outs(%27 : memref<1x64xf32>)
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%17 : memref<1x64x128xf32>) outs(%27 : memref<1x64xf32>) {
            ^bb0(%in: f32, %out: f32):
              %44 = arith.addf %in, %out : f32
              linalg.yield %44 : f32
            }
            linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%23, %25 : memref<1x64xf32>, memref<1x64xf32>) outs(%29 : memref<1x64xf32>) {
            ^bb0(%in: f32, %in_7: f32, %out: f32):
              %44 = arith.subf %in, %in_7 : f32
              %45 = math.powf %cst, %44 : f32
              linalg.yield %45 : f32
            }
            linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%21, %29, %27 : memref<1x64xf32>, memref<1x64xf32>, memref<1x64xf32>) outs(%21 : memref<1x64xf32>) {
            ^bb0(%in: f32, %in_7: f32, %in_8: f32, %out: f32):
              %44 = arith.mulf %in, %in_7 : f32
              %45 = arith.addf %44, %in_8 : f32
              linalg.yield %45 : f32
            }
            loom.semaphore_give %27 : memref<1x64xf32>
            %42 = arith.muli %arg7, %c16384 : index
            %43 = arith.addi %37, %42 : index
            %reinterpret_cast_6 = memref.reinterpret_cast %arg1 to offset: [%43], sizes: [1, 128, 128], strides: [524288, 128, 1] : memref<32x4096x128xf32> to memref<1x128x128xf32, strided<[524288, 128, 1], offset: ?>>
            loom.copy %reinterpret_cast_6, %15 src_mem_space @DRAM dst_mem_space @L1, interconnect : [], broadcast : [1, 1] : memref<1x128x128xf32, strided<[524288, 128, 1], offset: ?>>, memref<1x128x128xf32>
            linalg.fill ins(%cst_1 : f32) outs(%36 : memref<1x64x128xf32>)
            linalg.batch_matmul ins(%17, %15 : memref<1x64x128xf32>, memref<1x128x128xf32>) outs(%36 : memref<1x64x128xf32>)
            loom.semaphore_give %15 : memref<1x128x128xf32>
            loom.semaphore_give %17 : memref<1x64x128xf32>
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%36, %34, %29 : memref<1x64x128xf32>, memref<1x64x128xf32>, memref<1x64xf32>) outs(%34 : memref<1x64x128xf32>) {
            ^bb0(%in: f32, %in_7: f32, %in_8: f32, %out: f32):
              %44 = arith.mulf %in_7, %in_8 : f32
              %45 = arith.addf %in, %44 : f32
              linalg.yield %45 : f32
            }
            loom.semaphore_give %29 : memref<1x64xf32>
            loom.semaphore_give %36 : memref<1x64x128xf32>
            linalg.copy ins(%25 : memref<1x64xf32>) outs(%23 : memref<1x64xf32>)
            loom.semaphore_give %25 : memref<1x64xf32>
          }
          loom.semaphore_give %23 : memref<1x64xf32>
          loom.semaphore_give %32 : memref<1x64x128xf32>
          linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%34, %21 : memref<1x64x128xf32>, memref<1x64xf32>) outs(%31 : memref<1x64x128xf32>) {
          ^bb0(%in: f32, %in_5: f32, %out: f32):
            %40 = arith.divf %in, %in_5 : f32
            linalg.yield %40 : f32
          }
          loom.semaphore_give %21 : memref<1x64xf32>
          loom.semaphore_give %34 : memref<1x64x128xf32>
          %reinterpret_cast_4 = memref.reinterpret_cast %arg3 to offset: [%39], sizes: [1, 64, 128], strides: [524288, 128, 1] : memref<32x4096x128xf32> to memref<1x64x128xf32, strided<[524288, 128, 1], offset: ?>>
          loom.copy %31, %reinterpret_cast_4 src_mem_space @L1 dst_mem_space @DRAM, interconnect : [], broadcast : [1, 1] : memref<1x64x128xf32>, memref<1x64x128xf32, strided<[524288, 128, 1], offset: ?>>
          loom.semaphore_give %31 : memref<1x64x128xf32>
        }
        scf.reduce 
      } {loom.iter_types = [#loom.iter_type<spatial>, #loom.iter_type<spatial>], loom.mapped_to_dims = [@y, @x]}
      return
    }
  }
  module attributes {loom.pass_name = "Materialize"} {
    func.func @attention__d1i1_d0i1__f10__d_v_a__BB1__BM64__BN128(%arg0: memref<32x128x4096xf32>, %arg1: memref<32x4096x128xf32>, %arg2: memref<32x4096x128xf32>, %arg3: memref<32x4096x128xf32>) {
      %c16384 = arith.constant 16384 : index
      %c8192 = arith.constant 8192 : index
      %c524288 = arith.constant 524288 : index
      %c32 = arith.constant 32 : index
      %cst = arith.constant 2.000000e+00 : f32
      %cst_0 = arith.constant 0.12751743074602467 : f64
      %cst_1 = arith.constant 0.000000e+00 : f32
      %cst_2 = arith.constant 1.000000e+00 : f32
      %cst_3 = arith.constant 0xFF800000 : f32
      %c128 = arith.constant 128 : index
      %c0 = arith.constant 0 : index
      %c8 = arith.constant 8 : index
      %c1 = arith.constant 1 : index
      scf.parallel (%arg4, %arg5) = (%c0, %c0) to (%c8, %c8) step (%c1, %c1) {
        scf.for %arg6 = %c0 to %c32 step %c1 {
          %12 = arith.muli %arg4, %c8 overflow<nsw> : index
          %13 = arith.addi %12, %arg5 : index
          %14 = loom.alloc [1, 128, 128] on @L1 : memref<1x128x128xf32>
          %15 = loom.semaphore_take %14 : memref<1x128x128xf32> -> memref<1x128x128xf32>
          %16 = loom.alloc [1, 64, 128] on @L1 : memref<1x64x128xf32>
          %17 = loom.semaphore_take %16 : memref<1x64x128xf32> -> memref<1x64x128xf32>
          %18 = loom.alloc [1, 128, 128] on @L1 : memref<1x128x128xf32>
          %19 = loom.semaphore_take %18 : memref<1x128x128xf32> -> memref<1x128x128xf32>
          %20 = loom.alloc [1, 64] on @L1 : memref<1x64xf32>
          %21 = loom.semaphore_take %20 : memref<1x64xf32> -> memref<1x64xf32>
          %22 = loom.alloc [1, 64] on @L1 : memref<1x64xf32>
          %23 = loom.semaphore_take %22 : memref<1x64xf32> -> memref<1x64xf32>
          %24 = loom.alloc [1, 64] on @L1 : memref<1x64xf32>
          %25 = loom.semaphore_take %24 : memref<1x64xf32> -> memref<1x64xf32>
          %26 = loom.alloc [1, 64] on @L1 : memref<1x64xf32>
          %27 = loom.semaphore_take %26 : memref<1x64xf32> -> memref<1x64xf32>
          %28 = loom.alloc [1, 64] on @L1 : memref<1x64xf32>
          %29 = loom.semaphore_take %28 : memref<1x64xf32> -> memref<1x64xf32>
          %30 = loom.alloc [1, 64, 128] on @L1 : memref<1x64x128xf32>
          %31 = loom.semaphore_take %30 : memref<1x64x128xf32> -> memref<1x64x128xf32>
          %32 = loom.semaphore_take %30 : memref<1x64x128xf32> -> memref<1x64x128xf32>
          %33 = loom.alloc [1, 64, 128] on @L1 : memref<1x64x128xf32>
          %34 = loom.semaphore_take %33 : memref<1x64x128xf32> -> memref<1x64x128xf32>
          %35 = loom.alloc [1, 64, 128] on @L1 : memref<1x64x128xf32>
          %36 = loom.semaphore_take %35 : memref<1x64x128xf32> -> memref<1x64x128xf32>
          %37 = arith.muli %arg6, %c524288 overflow<nsw> : index
          %38 = arith.muli %13, %c8192 : index
          %39 = arith.addi %37, %38 : index
          %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%39], sizes: [1, 64, 128], strides: [524288, 128, 1] : memref<32x4096x128xf32> to memref<1x64x128xf32, strided<[524288, 128, 1], offset: ?>>
          loom.copy %reinterpret_cast, %32 src_mem_space @DRAM dst_mem_space @L1, interconnect : [], broadcast : [1, 1] : memref<1x64x128xf32, strided<[524288, 128, 1], offset: ?>>, memref<1x64x128xf32>
          linalg.fill ins(%cst_1 : f32) outs(%34 : memref<1x64x128xf32>)
          linalg.fill ins(%cst_2 : f32) outs(%21 : memref<1x64xf32>)
          linalg.fill ins(%cst_3 : f32) outs(%23 : memref<1x64xf32>)
          scf.for %arg7 = %c0 to %c32 step %c1 {
            %40 = arith.muli %arg7, %c128 : index
            %41 = arith.addi %37, %40 : index
            %reinterpret_cast_5 = memref.reinterpret_cast %arg0 to offset: [%41], sizes: [1, 128, 128], strides: [524288, 4096, 1] : memref<32x128x4096xf32> to memref<1x128x128xf32, strided<[524288, 4096, 1], offset: ?>>
            loom.copy %reinterpret_cast_5, %19 src_mem_space @DRAM dst_mem_space @L1, interconnect : [@vertical_links], broadcast : [8, 1] : memref<1x128x128xf32, strided<[524288, 4096, 1], offset: ?>>, memref<1x128x128xf32>
            linalg.fill ins(%cst_1 : f32) outs(%17 : memref<1x64x128xf32>)
            linalg.batch_matmul ins(%32, %19 : memref<1x64x128xf32>, memref<1x128x128xf32>) outs(%17 : memref<1x64x128xf32>)
            loom.semaphore_give %19 : memref<1x128x128xf32>
            linalg.fill ins(%cst_3 : f32) outs(%25 : memref<1x64xf32>)
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%17 : memref<1x64x128xf32>) outs(%25 : memref<1x64xf32>) {
            ^bb0(%in: f32, %out: f32):
              %44 = arith.maximumf %in, %out : f32
              linalg.yield %44 : f32
            }
            linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%23, %25 : memref<1x64xf32>, memref<1x64xf32>) outs(%25 : memref<1x64xf32>) {
            ^bb0(%in: f32, %in_7: f32, %out: f32):
              %44 = arith.truncf %cst_0 : f64 to f32
              %45 = arith.mulf %in_7, %44 : f32
              %46 = arith.cmpf ogt, %in, %45 : f32
              %47 = arith.select %46, %in, %45 : f32
              linalg.yield %47 : f32
            }
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%17, %25 : memref<1x64x128xf32>, memref<1x64xf32>) outs(%17 : memref<1x64x128xf32>) {
            ^bb0(%in: f32, %in_7: f32, %out: f32):
              %44 = arith.truncf %cst_0 : f64 to f32
              %45 = arith.mulf %in, %44 : f32
              %46 = arith.subf %45, %in_7 : f32
              %47 = math.powf %cst, %46 : f32
              linalg.yield %47 : f32
            }
            linalg.fill ins(%cst_1 : f32) outs(%27 : memref<1x64xf32>)
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%17 : memref<1x64x128xf32>) outs(%27 : memref<1x64xf32>) {
            ^bb0(%in: f32, %out: f32):
              %44 = arith.addf %in, %out : f32
              linalg.yield %44 : f32
            }
            linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%23, %25 : memref<1x64xf32>, memref<1x64xf32>) outs(%29 : memref<1x64xf32>) {
            ^bb0(%in: f32, %in_7: f32, %out: f32):
              %44 = arith.subf %in, %in_7 : f32
              %45 = math.powf %cst, %44 : f32
              linalg.yield %45 : f32
            }
            linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%21, %29, %27 : memref<1x64xf32>, memref<1x64xf32>, memref<1x64xf32>) outs(%21 : memref<1x64xf32>) {
            ^bb0(%in: f32, %in_7: f32, %in_8: f32, %out: f32):
              %44 = arith.mulf %in, %in_7 : f32
              %45 = arith.addf %44, %in_8 : f32
              linalg.yield %45 : f32
            }
            loom.semaphore_give %27 : memref<1x64xf32>
            %42 = arith.muli %arg7, %c16384 : index
            %43 = arith.addi %37, %42 : index
            %reinterpret_cast_6 = memref.reinterpret_cast %arg1 to offset: [%43], sizes: [1, 128, 128], strides: [524288, 128, 1] : memref<32x4096x128xf32> to memref<1x128x128xf32, strided<[524288, 128, 1], offset: ?>>
            loom.copy %reinterpret_cast_6, %15 src_mem_space @DRAM dst_mem_space @L1, interconnect : [@horizontal_links, @vertical_links], broadcast : [8, 8] : memref<1x128x128xf32, strided<[524288, 128, 1], offset: ?>>, memref<1x128x128xf32>
            linalg.fill ins(%cst_1 : f32) outs(%36 : memref<1x64x128xf32>)
            linalg.batch_matmul ins(%17, %15 : memref<1x64x128xf32>, memref<1x128x128xf32>) outs(%36 : memref<1x64x128xf32>)
            loom.semaphore_give %15 : memref<1x128x128xf32>
            loom.semaphore_give %17 : memref<1x64x128xf32>
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%36, %34, %29 : memref<1x64x128xf32>, memref<1x64x128xf32>, memref<1x64xf32>) outs(%34 : memref<1x64x128xf32>) {
            ^bb0(%in: f32, %in_7: f32, %in_8: f32, %out: f32):
              %44 = arith.mulf %in_7, %in_8 : f32
              %45 = arith.addf %in, %44 : f32
              linalg.yield %45 : f32
            }
            loom.semaphore_give %29 : memref<1x64xf32>
            loom.semaphore_give %36 : memref<1x64x128xf32>
            linalg.copy ins(%25 : memref<1x64xf32>) outs(%23 : memref<1x64xf32>)
            loom.semaphore_give %25 : memref<1x64xf32>
          }
          loom.semaphore_give %23 : memref<1x64xf32>
          loom.semaphore_give %32 : memref<1x64x128xf32>
          linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%34, %21 : memref<1x64x128xf32>, memref<1x64xf32>) outs(%31 : memref<1x64x128xf32>) {
          ^bb0(%in: f32, %in_5: f32, %out: f32):
            %40 = arith.divf %in, %in_5 : f32
            linalg.yield %40 : f32
          }
          loom.semaphore_give %21 : memref<1x64xf32>
          loom.semaphore_give %34 : memref<1x64x128xf32>
          %reinterpret_cast_4 = memref.reinterpret_cast %arg3 to offset: [%39], sizes: [1, 64, 128], strides: [524288, 128, 1] : memref<32x4096x128xf32> to memref<1x64x128xf32, strided<[524288, 128, 1], offset: ?>>
          loom.copy %31, %reinterpret_cast_4 src_mem_space @L1 dst_mem_space @DRAM, interconnect : [], broadcast : [1, 1] : memref<1x64x128xf32>, memref<1x64x128xf32, strided<[524288, 128, 1], offset: ?>>
          loom.semaphore_give %31 : memref<1x64x128xf32>
        }
        scf.reduce 
      } {loom.iter_types = [#loom.iter_type<spatial>, #loom.iter_type<spatial>], loom.mapped_to_dims = [@y, @x]}
      return
    }
  }
  module attributes {loom.pass_name = "Materialize"} {
    func.func @attention__d1i1_d0i1__f10__d_v_h__BB1__BM64__BN128(%arg0: memref<32x128x4096xf32>, %arg1: memref<32x4096x128xf32>, %arg2: memref<32x4096x128xf32>, %arg3: memref<32x4096x128xf32>) {
      %c16384 = arith.constant 16384 : index
      %c8192 = arith.constant 8192 : index
      %c524288 = arith.constant 524288 : index
      %c32 = arith.constant 32 : index
      %cst = arith.constant 2.000000e+00 : f32
      %cst_0 = arith.constant 0.12751743074602467 : f64
      %cst_1 = arith.constant 0.000000e+00 : f32
      %cst_2 = arith.constant 1.000000e+00 : f32
      %cst_3 = arith.constant 0xFF800000 : f32
      %c128 = arith.constant 128 : index
      %c0 = arith.constant 0 : index
      %c8 = arith.constant 8 : index
      %c1 = arith.constant 1 : index
      scf.parallel (%arg4, %arg5) = (%c0, %c0) to (%c8, %c8) step (%c1, %c1) {
        scf.for %arg6 = %c0 to %c32 step %c1 {
          %12 = arith.muli %arg4, %c8 overflow<nsw> : index
          %13 = arith.addi %12, %arg5 : index
          %14 = loom.alloc [1, 128, 128] on @L1 : memref<1x128x128xf32>
          %15 = loom.semaphore_take %14 : memref<1x128x128xf32> -> memref<1x128x128xf32>
          %16 = loom.alloc [1, 64, 128] on @L1 : memref<1x64x128xf32>
          %17 = loom.semaphore_take %16 : memref<1x64x128xf32> -> memref<1x64x128xf32>
          %18 = loom.alloc [1, 128, 128] on @L1 : memref<1x128x128xf32>
          %19 = loom.semaphore_take %18 : memref<1x128x128xf32> -> memref<1x128x128xf32>
          %20 = loom.alloc [1, 64] on @L1 : memref<1x64xf32>
          %21 = loom.semaphore_take %20 : memref<1x64xf32> -> memref<1x64xf32>
          %22 = loom.alloc [1, 64] on @L1 : memref<1x64xf32>
          %23 = loom.semaphore_take %22 : memref<1x64xf32> -> memref<1x64xf32>
          %24 = loom.alloc [1, 64] on @L1 : memref<1x64xf32>
          %25 = loom.semaphore_take %24 : memref<1x64xf32> -> memref<1x64xf32>
          %26 = loom.alloc [1, 64] on @L1 : memref<1x64xf32>
          %27 = loom.semaphore_take %26 : memref<1x64xf32> -> memref<1x64xf32>
          %28 = loom.alloc [1, 64] on @L1 : memref<1x64xf32>
          %29 = loom.semaphore_take %28 : memref<1x64xf32> -> memref<1x64xf32>
          %30 = loom.alloc [1, 64, 128] on @L1 : memref<1x64x128xf32>
          %31 = loom.semaphore_take %30 : memref<1x64x128xf32> -> memref<1x64x128xf32>
          %32 = loom.semaphore_take %30 : memref<1x64x128xf32> -> memref<1x64x128xf32>
          %33 = loom.alloc [1, 64, 128] on @L1 : memref<1x64x128xf32>
          %34 = loom.semaphore_take %33 : memref<1x64x128xf32> -> memref<1x64x128xf32>
          %35 = loom.alloc [1, 64, 128] on @L1 : memref<1x64x128xf32>
          %36 = loom.semaphore_take %35 : memref<1x64x128xf32> -> memref<1x64x128xf32>
          %37 = arith.muli %arg6, %c524288 overflow<nsw> : index
          %38 = arith.muli %13, %c8192 : index
          %39 = arith.addi %37, %38 : index
          %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%39], sizes: [1, 64, 128], strides: [524288, 128, 1] : memref<32x4096x128xf32> to memref<1x64x128xf32, strided<[524288, 128, 1], offset: ?>>
          loom.copy %reinterpret_cast, %32 src_mem_space @DRAM dst_mem_space @L1, interconnect : [], broadcast : [1, 1] : memref<1x64x128xf32, strided<[524288, 128, 1], offset: ?>>, memref<1x64x128xf32>
          linalg.fill ins(%cst_1 : f32) outs(%34 : memref<1x64x128xf32>)
          linalg.fill ins(%cst_2 : f32) outs(%21 : memref<1x64xf32>)
          linalg.fill ins(%cst_3 : f32) outs(%23 : memref<1x64xf32>)
          scf.for %arg7 = %c0 to %c32 step %c1 {
            %40 = arith.muli %arg7, %c128 : index
            %41 = arith.addi %37, %40 : index
            %reinterpret_cast_5 = memref.reinterpret_cast %arg0 to offset: [%41], sizes: [1, 128, 128], strides: [524288, 4096, 1] : memref<32x128x4096xf32> to memref<1x128x128xf32, strided<[524288, 4096, 1], offset: ?>>
            loom.copy %reinterpret_cast_5, %19 src_mem_space @DRAM dst_mem_space @L1, interconnect : [@vertical_links], broadcast : [8, 1] : memref<1x128x128xf32, strided<[524288, 4096, 1], offset: ?>>, memref<1x128x128xf32>
            linalg.fill ins(%cst_1 : f32) outs(%17 : memref<1x64x128xf32>)
            linalg.batch_matmul ins(%32, %19 : memref<1x64x128xf32>, memref<1x128x128xf32>) outs(%17 : memref<1x64x128xf32>)
            loom.semaphore_give %19 : memref<1x128x128xf32>
            linalg.fill ins(%cst_3 : f32) outs(%25 : memref<1x64xf32>)
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%17 : memref<1x64x128xf32>) outs(%25 : memref<1x64xf32>) {
            ^bb0(%in: f32, %out: f32):
              %44 = arith.maximumf %in, %out : f32
              linalg.yield %44 : f32
            }
            linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%23, %25 : memref<1x64xf32>, memref<1x64xf32>) outs(%25 : memref<1x64xf32>) {
            ^bb0(%in: f32, %in_7: f32, %out: f32):
              %44 = arith.truncf %cst_0 : f64 to f32
              %45 = arith.mulf %in_7, %44 : f32
              %46 = arith.cmpf ogt, %in, %45 : f32
              %47 = arith.select %46, %in, %45 : f32
              linalg.yield %47 : f32
            }
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%17, %25 : memref<1x64x128xf32>, memref<1x64xf32>) outs(%17 : memref<1x64x128xf32>) {
            ^bb0(%in: f32, %in_7: f32, %out: f32):
              %44 = arith.truncf %cst_0 : f64 to f32
              %45 = arith.mulf %in, %44 : f32
              %46 = arith.subf %45, %in_7 : f32
              %47 = math.powf %cst, %46 : f32
              linalg.yield %47 : f32
            }
            linalg.fill ins(%cst_1 : f32) outs(%27 : memref<1x64xf32>)
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%17 : memref<1x64x128xf32>) outs(%27 : memref<1x64xf32>) {
            ^bb0(%in: f32, %out: f32):
              %44 = arith.addf %in, %out : f32
              linalg.yield %44 : f32
            }
            linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%23, %25 : memref<1x64xf32>, memref<1x64xf32>) outs(%29 : memref<1x64xf32>) {
            ^bb0(%in: f32, %in_7: f32, %out: f32):
              %44 = arith.subf %in, %in_7 : f32
              %45 = math.powf %cst, %44 : f32
              linalg.yield %45 : f32
            }
            linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%21, %29, %27 : memref<1x64xf32>, memref<1x64xf32>, memref<1x64xf32>) outs(%21 : memref<1x64xf32>) {
            ^bb0(%in: f32, %in_7: f32, %in_8: f32, %out: f32):
              %44 = arith.mulf %in, %in_7 : f32
              %45 = arith.addf %44, %in_8 : f32
              linalg.yield %45 : f32
            }
            loom.semaphore_give %27 : memref<1x64xf32>
            %42 = arith.muli %arg7, %c16384 : index
            %43 = arith.addi %37, %42 : index
            %reinterpret_cast_6 = memref.reinterpret_cast %arg1 to offset: [%43], sizes: [1, 128, 128], strides: [524288, 128, 1] : memref<32x4096x128xf32> to memref<1x128x128xf32, strided<[524288, 128, 1], offset: ?>>
            loom.copy %reinterpret_cast_6, %15 src_mem_space @DRAM dst_mem_space @L1, interconnect : [@horizontal_links], broadcast : [1, 8] : memref<1x128x128xf32, strided<[524288, 128, 1], offset: ?>>, memref<1x128x128xf32>
            linalg.fill ins(%cst_1 : f32) outs(%36 : memref<1x64x128xf32>)
            linalg.batch_matmul ins(%17, %15 : memref<1x64x128xf32>, memref<1x128x128xf32>) outs(%36 : memref<1x64x128xf32>)
            loom.semaphore_give %15 : memref<1x128x128xf32>
            loom.semaphore_give %17 : memref<1x64x128xf32>
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%36, %34, %29 : memref<1x64x128xf32>, memref<1x64x128xf32>, memref<1x64xf32>) outs(%34 : memref<1x64x128xf32>) {
            ^bb0(%in: f32, %in_7: f32, %in_8: f32, %out: f32):
              %44 = arith.mulf %in_7, %in_8 : f32
              %45 = arith.addf %in, %44 : f32
              linalg.yield %45 : f32
            }
            loom.semaphore_give %29 : memref<1x64xf32>
            loom.semaphore_give %36 : memref<1x64x128xf32>
            linalg.copy ins(%25 : memref<1x64xf32>) outs(%23 : memref<1x64xf32>)
            loom.semaphore_give %25 : memref<1x64xf32>
          }
          loom.semaphore_give %23 : memref<1x64xf32>
          loom.semaphore_give %32 : memref<1x64x128xf32>
          linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%34, %21 : memref<1x64x128xf32>, memref<1x64xf32>) outs(%31 : memref<1x64x128xf32>) {
          ^bb0(%in: f32, %in_5: f32, %out: f32):
            %40 = arith.divf %in, %in_5 : f32
            linalg.yield %40 : f32
          }
          loom.semaphore_give %21 : memref<1x64xf32>
          loom.semaphore_give %34 : memref<1x64x128xf32>
          %reinterpret_cast_4 = memref.reinterpret_cast %arg3 to offset: [%39], sizes: [1, 64, 128], strides: [524288, 128, 1] : memref<32x4096x128xf32> to memref<1x64x128xf32, strided<[524288, 128, 1], offset: ?>>
          loom.copy %31, %reinterpret_cast_4 src_mem_space @L1 dst_mem_space @DRAM, interconnect : [], broadcast : [1, 1] : memref<1x64x128xf32>, memref<1x64x128xf32, strided<[524288, 128, 1], offset: ?>>
          loom.semaphore_give %31 : memref<1x64x128xf32>
        }
        scf.reduce 
      } {loom.iter_types = [#loom.iter_type<spatial>, #loom.iter_type<spatial>], loom.mapped_to_dims = [@y, @x]}
      return
    }
  }
  module attributes {loom.pass_name = "Materialize"} {
    func.func @attention__d1i1_d0i1__f10__d_v_v__BB1__BM64__BN128(%arg0: memref<32x128x4096xf32>, %arg1: memref<32x4096x128xf32>, %arg2: memref<32x4096x128xf32>, %arg3: memref<32x4096x128xf32>) {
      %c16384 = arith.constant 16384 : index
      %c8192 = arith.constant 8192 : index
      %c524288 = arith.constant 524288 : index
      %c32 = arith.constant 32 : index
      %cst = arith.constant 2.000000e+00 : f32
      %cst_0 = arith.constant 0.12751743074602467 : f64
      %cst_1 = arith.constant 0.000000e+00 : f32
      %cst_2 = arith.constant 1.000000e+00 : f32
      %cst_3 = arith.constant 0xFF800000 : f32
      %c128 = arith.constant 128 : index
      %c0 = arith.constant 0 : index
      %c8 = arith.constant 8 : index
      %c1 = arith.constant 1 : index
      scf.parallel (%arg4, %arg5) = (%c0, %c0) to (%c8, %c8) step (%c1, %c1) {
        scf.for %arg6 = %c0 to %c32 step %c1 {
          %12 = arith.muli %arg4, %c8 overflow<nsw> : index
          %13 = arith.addi %12, %arg5 : index
          %14 = loom.alloc [1, 128, 128] on @L1 : memref<1x128x128xf32>
          %15 = loom.semaphore_take %14 : memref<1x128x128xf32> -> memref<1x128x128xf32>
          %16 = loom.alloc [1, 64, 128] on @L1 : memref<1x64x128xf32>
          %17 = loom.semaphore_take %16 : memref<1x64x128xf32> -> memref<1x64x128xf32>
          %18 = loom.alloc [1, 128, 128] on @L1 : memref<1x128x128xf32>
          %19 = loom.semaphore_take %18 : memref<1x128x128xf32> -> memref<1x128x128xf32>
          %20 = loom.alloc [1, 64] on @L1 : memref<1x64xf32>
          %21 = loom.semaphore_take %20 : memref<1x64xf32> -> memref<1x64xf32>
          %22 = loom.alloc [1, 64] on @L1 : memref<1x64xf32>
          %23 = loom.semaphore_take %22 : memref<1x64xf32> -> memref<1x64xf32>
          %24 = loom.alloc [1, 64] on @L1 : memref<1x64xf32>
          %25 = loom.semaphore_take %24 : memref<1x64xf32> -> memref<1x64xf32>
          %26 = loom.alloc [1, 64] on @L1 : memref<1x64xf32>
          %27 = loom.semaphore_take %26 : memref<1x64xf32> -> memref<1x64xf32>
          %28 = loom.alloc [1, 64] on @L1 : memref<1x64xf32>
          %29 = loom.semaphore_take %28 : memref<1x64xf32> -> memref<1x64xf32>
          %30 = loom.alloc [1, 64, 128] on @L1 : memref<1x64x128xf32>
          %31 = loom.semaphore_take %30 : memref<1x64x128xf32> -> memref<1x64x128xf32>
          %32 = loom.semaphore_take %30 : memref<1x64x128xf32> -> memref<1x64x128xf32>
          %33 = loom.alloc [1, 64, 128] on @L1 : memref<1x64x128xf32>
          %34 = loom.semaphore_take %33 : memref<1x64x128xf32> -> memref<1x64x128xf32>
          %35 = loom.alloc [1, 64, 128] on @L1 : memref<1x64x128xf32>
          %36 = loom.semaphore_take %35 : memref<1x64x128xf32> -> memref<1x64x128xf32>
          %37 = arith.muli %arg6, %c524288 overflow<nsw> : index
          %38 = arith.muli %13, %c8192 : index
          %39 = arith.addi %37, %38 : index
          %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%39], sizes: [1, 64, 128], strides: [524288, 128, 1] : memref<32x4096x128xf32> to memref<1x64x128xf32, strided<[524288, 128, 1], offset: ?>>
          loom.copy %reinterpret_cast, %32 src_mem_space @DRAM dst_mem_space @L1, interconnect : [], broadcast : [1, 1] : memref<1x64x128xf32, strided<[524288, 128, 1], offset: ?>>, memref<1x64x128xf32>
          linalg.fill ins(%cst_1 : f32) outs(%34 : memref<1x64x128xf32>)
          linalg.fill ins(%cst_2 : f32) outs(%21 : memref<1x64xf32>)
          linalg.fill ins(%cst_3 : f32) outs(%23 : memref<1x64xf32>)
          scf.for %arg7 = %c0 to %c32 step %c1 {
            %40 = arith.muli %arg7, %c128 : index
            %41 = arith.addi %37, %40 : index
            %reinterpret_cast_5 = memref.reinterpret_cast %arg0 to offset: [%41], sizes: [1, 128, 128], strides: [524288, 4096, 1] : memref<32x128x4096xf32> to memref<1x128x128xf32, strided<[524288, 4096, 1], offset: ?>>
            loom.copy %reinterpret_cast_5, %19 src_mem_space @DRAM dst_mem_space @L1, interconnect : [@vertical_links], broadcast : [8, 1] : memref<1x128x128xf32, strided<[524288, 4096, 1], offset: ?>>, memref<1x128x128xf32>
            linalg.fill ins(%cst_1 : f32) outs(%17 : memref<1x64x128xf32>)
            linalg.batch_matmul ins(%32, %19 : memref<1x64x128xf32>, memref<1x128x128xf32>) outs(%17 : memref<1x64x128xf32>)
            loom.semaphore_give %19 : memref<1x128x128xf32>
            linalg.fill ins(%cst_3 : f32) outs(%25 : memref<1x64xf32>)
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%17 : memref<1x64x128xf32>) outs(%25 : memref<1x64xf32>) {
            ^bb0(%in: f32, %out: f32):
              %44 = arith.maximumf %in, %out : f32
              linalg.yield %44 : f32
            }
            linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%23, %25 : memref<1x64xf32>, memref<1x64xf32>) outs(%25 : memref<1x64xf32>) {
            ^bb0(%in: f32, %in_7: f32, %out: f32):
              %44 = arith.truncf %cst_0 : f64 to f32
              %45 = arith.mulf %in_7, %44 : f32
              %46 = arith.cmpf ogt, %in, %45 : f32
              %47 = arith.select %46, %in, %45 : f32
              linalg.yield %47 : f32
            }
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%17, %25 : memref<1x64x128xf32>, memref<1x64xf32>) outs(%17 : memref<1x64x128xf32>) {
            ^bb0(%in: f32, %in_7: f32, %out: f32):
              %44 = arith.truncf %cst_0 : f64 to f32
              %45 = arith.mulf %in, %44 : f32
              %46 = arith.subf %45, %in_7 : f32
              %47 = math.powf %cst, %46 : f32
              linalg.yield %47 : f32
            }
            linalg.fill ins(%cst_1 : f32) outs(%27 : memref<1x64xf32>)
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%17 : memref<1x64x128xf32>) outs(%27 : memref<1x64xf32>) {
            ^bb0(%in: f32, %out: f32):
              %44 = arith.addf %in, %out : f32
              linalg.yield %44 : f32
            }
            linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%23, %25 : memref<1x64xf32>, memref<1x64xf32>) outs(%29 : memref<1x64xf32>) {
            ^bb0(%in: f32, %in_7: f32, %out: f32):
              %44 = arith.subf %in, %in_7 : f32
              %45 = math.powf %cst, %44 : f32
              linalg.yield %45 : f32
            }
            linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%21, %29, %27 : memref<1x64xf32>, memref<1x64xf32>, memref<1x64xf32>) outs(%21 : memref<1x64xf32>) {
            ^bb0(%in: f32, %in_7: f32, %in_8: f32, %out: f32):
              %44 = arith.mulf %in, %in_7 : f32
              %45 = arith.addf %44, %in_8 : f32
              linalg.yield %45 : f32
            }
            loom.semaphore_give %27 : memref<1x64xf32>
            %42 = arith.muli %arg7, %c16384 : index
            %43 = arith.addi %37, %42 : index
            %reinterpret_cast_6 = memref.reinterpret_cast %arg1 to offset: [%43], sizes: [1, 128, 128], strides: [524288, 128, 1] : memref<32x4096x128xf32> to memref<1x128x128xf32, strided<[524288, 128, 1], offset: ?>>
            loom.copy %reinterpret_cast_6, %15 src_mem_space @DRAM dst_mem_space @L1, interconnect : [@vertical_links], broadcast : [8, 1] : memref<1x128x128xf32, strided<[524288, 128, 1], offset: ?>>, memref<1x128x128xf32>
            linalg.fill ins(%cst_1 : f32) outs(%36 : memref<1x64x128xf32>)
            linalg.batch_matmul ins(%17, %15 : memref<1x64x128xf32>, memref<1x128x128xf32>) outs(%36 : memref<1x64x128xf32>)
            loom.semaphore_give %15 : memref<1x128x128xf32>
            loom.semaphore_give %17 : memref<1x64x128xf32>
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%36, %34, %29 : memref<1x64x128xf32>, memref<1x64x128xf32>, memref<1x64xf32>) outs(%34 : memref<1x64x128xf32>) {
            ^bb0(%in: f32, %in_7: f32, %in_8: f32, %out: f32):
              %44 = arith.mulf %in_7, %in_8 : f32
              %45 = arith.addf %in, %44 : f32
              linalg.yield %45 : f32
            }
            loom.semaphore_give %29 : memref<1x64xf32>
            loom.semaphore_give %36 : memref<1x64x128xf32>
            linalg.copy ins(%25 : memref<1x64xf32>) outs(%23 : memref<1x64xf32>)
            loom.semaphore_give %25 : memref<1x64xf32>
          }
          loom.semaphore_give %23 : memref<1x64xf32>
          loom.semaphore_give %32 : memref<1x64x128xf32>
          linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%34, %21 : memref<1x64x128xf32>, memref<1x64xf32>) outs(%31 : memref<1x64x128xf32>) {
          ^bb0(%in: f32, %in_5: f32, %out: f32):
            %40 = arith.divf %in, %in_5 : f32
            linalg.yield %40 : f32
          }
          loom.semaphore_give %21 : memref<1x64xf32>
          loom.semaphore_give %34 : memref<1x64x128xf32>
          %reinterpret_cast_4 = memref.reinterpret_cast %arg3 to offset: [%39], sizes: [1, 64, 128], strides: [524288, 128, 1] : memref<32x4096x128xf32> to memref<1x64x128xf32, strided<[524288, 128, 1], offset: ?>>
          loom.copy %31, %reinterpret_cast_4 src_mem_space @L1 dst_mem_space @DRAM, interconnect : [], broadcast : [1, 1] : memref<1x64x128xf32>, memref<1x64x128xf32, strided<[524288, 128, 1], offset: ?>>
          loom.semaphore_give %31 : memref<1x64x128xf32>
        }
        scf.reduce 
      } {loom.iter_types = [#loom.iter_type<spatial>, #loom.iter_type<spatial>], loom.mapped_to_dims = [@y, @x]}
      return
    }
  }
}
