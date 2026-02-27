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
          %15 = loom.alloc [1, 64, 128] on @L1 : memref<1x64x128xf32>
          %16 = loom.alloc [1, 128, 128] on @L1 : memref<1x128x128xf32>
          %17 = loom.alloc [1, 64] on @L1 : memref<1x64xf32>
          %18 = loom.alloc [1, 64] on @L1 : memref<1x64xf32>
          %19 = loom.alloc [1, 64] on @L1 : memref<1x64xf32>
          %20 = loom.alloc [1, 64] on @L1 : memref<1x64xf32>
          %21 = loom.alloc [1, 64] on @L1 : memref<1x64xf32>
          %22 = loom.alloc [1, 64, 128] on @L1 : memref<1x64x128xf32>
          %23 = loom.alloc [1, 64, 128] on @L1 : memref<1x64x128xf32>
          %24 = loom.alloc [1, 64, 128] on @L1 : memref<1x64x128xf32>
          %25 = arith.muli %13, %c524288 overflow<nsw> : index
          %26 = arith.muli %arg6, %c8192 : index
          %27 = arith.addi %25, %26 : index
          %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%27], sizes: [1, 64, 128], strides: [524288, 128, 1] : memref<32x4096x128xf32> to memref<1x64x128xf32, strided<[524288, 128, 1], offset: ?>>
          loom.copy %reinterpret_cast, %22 src_mem_space @DRAM dst_mem_space @L1, interconnect : [], broadcast : [1, 1] : memref<1x64x128xf32, strided<[524288, 128, 1], offset: ?>>, memref<1x64x128xf32>
          linalg.fill ins(%cst_1 : f32) outs(%23 : memref<1x64x128xf32>)
          linalg.fill ins(%cst_2 : f32) outs(%17 : memref<1x64xf32>)
          linalg.fill ins(%cst_3 : f32) outs(%18 : memref<1x64xf32>)
          scf.for %arg7 = %c0 to %c32 step %c1 {
            %28 = arith.muli %arg7, %c128 : index
            %29 = arith.addi %25, %28 : index
            %reinterpret_cast_5 = memref.reinterpret_cast %arg0 to offset: [%29], sizes: [1, 128, 128], strides: [524288, 4096, 1] : memref<32x128x4096xf32> to memref<1x128x128xf32, strided<[524288, 4096, 1], offset: ?>>
            loom.copy %reinterpret_cast_5, %16 src_mem_space @DRAM dst_mem_space @L1, interconnect : [], broadcast : [1, 1] : memref<1x128x128xf32, strided<[524288, 4096, 1], offset: ?>>, memref<1x128x128xf32>
            linalg.fill ins(%cst_1 : f32) outs(%15 : memref<1x64x128xf32>)
            linalg.batch_matmul ins(%22, %16 : memref<1x64x128xf32>, memref<1x128x128xf32>) outs(%15 : memref<1x64x128xf32>)
            linalg.fill ins(%cst_3 : f32) outs(%19 : memref<1x64xf32>)
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%15 : memref<1x64x128xf32>) outs(%19 : memref<1x64xf32>) {
            ^bb0(%in: f32, %out: f32):
              %32 = arith.maximumf %in, %out : f32
              linalg.yield %32 : f32
            }
            linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%18, %19 : memref<1x64xf32>, memref<1x64xf32>) outs(%19 : memref<1x64xf32>) {
            ^bb0(%in: f32, %in_7: f32, %out: f32):
              %32 = arith.truncf %cst_0 : f64 to f32
              %33 = arith.mulf %in_7, %32 : f32
              %34 = arith.cmpf ogt, %in, %33 : f32
              %35 = arith.select %34, %in, %33 : f32
              linalg.yield %35 : f32
            }
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%15, %19 : memref<1x64x128xf32>, memref<1x64xf32>) outs(%15 : memref<1x64x128xf32>) {
            ^bb0(%in: f32, %in_7: f32, %out: f32):
              %32 = arith.truncf %cst_0 : f64 to f32
              %33 = arith.mulf %in, %32 : f32
              %34 = arith.subf %33, %in_7 : f32
              %35 = math.powf %cst, %34 : f32
              linalg.yield %35 : f32
            }
            linalg.fill ins(%cst_1 : f32) outs(%20 : memref<1x64xf32>)
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%15 : memref<1x64x128xf32>) outs(%20 : memref<1x64xf32>) {
            ^bb0(%in: f32, %out: f32):
              %32 = arith.addf %in, %out : f32
              linalg.yield %32 : f32
            }
            linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%18, %19 : memref<1x64xf32>, memref<1x64xf32>) outs(%21 : memref<1x64xf32>) {
            ^bb0(%in: f32, %in_7: f32, %out: f32):
              %32 = arith.subf %in, %in_7 : f32
              %33 = math.powf %cst, %32 : f32
              linalg.yield %33 : f32
            }
            linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%17, %21, %20 : memref<1x64xf32>, memref<1x64xf32>, memref<1x64xf32>) outs(%17 : memref<1x64xf32>) {
            ^bb0(%in: f32, %in_7: f32, %in_8: f32, %out: f32):
              %32 = arith.mulf %in, %in_7 : f32
              %33 = arith.addf %32, %in_8 : f32
              linalg.yield %33 : f32
            }
            %30 = arith.muli %arg7, %c16384 : index
            %31 = arith.addi %25, %30 : index
            %reinterpret_cast_6 = memref.reinterpret_cast %arg1 to offset: [%31], sizes: [1, 128, 128], strides: [524288, 128, 1] : memref<32x4096x128xf32> to memref<1x128x128xf32, strided<[524288, 128, 1], offset: ?>>
            loom.copy %reinterpret_cast_6, %14 src_mem_space @DRAM dst_mem_space @L1, interconnect : [], broadcast : [1, 1] : memref<1x128x128xf32, strided<[524288, 128, 1], offset: ?>>, memref<1x128x128xf32>
            linalg.fill ins(%cst_1 : f32) outs(%24 : memref<1x64x128xf32>)
            linalg.batch_matmul ins(%15, %14 : memref<1x64x128xf32>, memref<1x128x128xf32>) outs(%24 : memref<1x64x128xf32>)
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%24, %23, %21 : memref<1x64x128xf32>, memref<1x64x128xf32>, memref<1x64xf32>) outs(%23 : memref<1x64x128xf32>) {
            ^bb0(%in: f32, %in_7: f32, %in_8: f32, %out: f32):
              %32 = arith.mulf %in_7, %in_8 : f32
              %33 = arith.addf %in, %32 : f32
              linalg.yield %33 : f32
            }
            linalg.copy ins(%19 : memref<1x64xf32>) outs(%18 : memref<1x64xf32>)
          }
          linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%23, %17 : memref<1x64x128xf32>, memref<1x64xf32>) outs(%22 : memref<1x64x128xf32>) {
          ^bb0(%in: f32, %in_5: f32, %out: f32):
            %28 = arith.divf %in, %in_5 : f32
            linalg.yield %28 : f32
          }
          %reinterpret_cast_4 = memref.reinterpret_cast %arg3 to offset: [%27], sizes: [1, 64, 128], strides: [524288, 128, 1] : memref<32x4096x128xf32> to memref<1x64x128xf32, strided<[524288, 128, 1], offset: ?>>
          loom.copy %22, %reinterpret_cast_4 src_mem_space @L1 dst_mem_space @DRAM, interconnect : [], broadcast : [1, 1] : memref<1x64x128xf32>, memref<1x64x128xf32, strided<[524288, 128, 1], offset: ?>>
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
          %15 = loom.alloc [1, 64, 128] on @L1 : memref<1x64x128xf32>
          %16 = loom.alloc [1, 128, 128] on @L1 : memref<1x128x128xf32>
          %17 = loom.alloc [1, 64] on @L1 : memref<1x64xf32>
          %18 = loom.alloc [1, 64] on @L1 : memref<1x64xf32>
          %19 = loom.alloc [1, 64] on @L1 : memref<1x64xf32>
          %20 = loom.alloc [1, 64] on @L1 : memref<1x64xf32>
          %21 = loom.alloc [1, 64] on @L1 : memref<1x64xf32>
          %22 = loom.alloc [1, 64, 128] on @L1 : memref<1x64x128xf32>
          %23 = loom.alloc [1, 64, 128] on @L1 : memref<1x64x128xf32>
          %24 = loom.alloc [1, 64, 128] on @L1 : memref<1x64x128xf32>
          %25 = arith.muli %13, %c524288 overflow<nsw> : index
          %26 = arith.muli %arg6, %c8192 : index
          %27 = arith.addi %25, %26 : index
          %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%27], sizes: [1, 64, 128], strides: [524288, 128, 1] : memref<32x4096x128xf32> to memref<1x64x128xf32, strided<[524288, 128, 1], offset: ?>>
          loom.copy %reinterpret_cast, %22 src_mem_space @DRAM dst_mem_space @L1, interconnect : [], broadcast : [1, 1] : memref<1x64x128xf32, strided<[524288, 128, 1], offset: ?>>, memref<1x64x128xf32>
          linalg.fill ins(%cst_1 : f32) outs(%23 : memref<1x64x128xf32>)
          linalg.fill ins(%cst_2 : f32) outs(%17 : memref<1x64xf32>)
          linalg.fill ins(%cst_3 : f32) outs(%18 : memref<1x64xf32>)
          scf.for %arg7 = %c0 to %c32 step %c1 {
            %28 = arith.muli %arg7, %c128 : index
            %29 = arith.addi %25, %28 : index
            %reinterpret_cast_5 = memref.reinterpret_cast %arg0 to offset: [%29], sizes: [1, 128, 128], strides: [524288, 4096, 1] : memref<32x128x4096xf32> to memref<1x128x128xf32, strided<[524288, 4096, 1], offset: ?>>
            loom.copy %reinterpret_cast_5, %16 src_mem_space @DRAM dst_mem_space @L1, interconnect : [], broadcast : [1, 1] : memref<1x128x128xf32, strided<[524288, 4096, 1], offset: ?>>, memref<1x128x128xf32>
            linalg.fill ins(%cst_1 : f32) outs(%15 : memref<1x64x128xf32>)
            linalg.batch_matmul ins(%22, %16 : memref<1x64x128xf32>, memref<1x128x128xf32>) outs(%15 : memref<1x64x128xf32>)
            linalg.fill ins(%cst_3 : f32) outs(%19 : memref<1x64xf32>)
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%15 : memref<1x64x128xf32>) outs(%19 : memref<1x64xf32>) {
            ^bb0(%in: f32, %out: f32):
              %32 = arith.maximumf %in, %out : f32
              linalg.yield %32 : f32
            }
            linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%18, %19 : memref<1x64xf32>, memref<1x64xf32>) outs(%19 : memref<1x64xf32>) {
            ^bb0(%in: f32, %in_7: f32, %out: f32):
              %32 = arith.truncf %cst_0 : f64 to f32
              %33 = arith.mulf %in_7, %32 : f32
              %34 = arith.cmpf ogt, %in, %33 : f32
              %35 = arith.select %34, %in, %33 : f32
              linalg.yield %35 : f32
            }
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%15, %19 : memref<1x64x128xf32>, memref<1x64xf32>) outs(%15 : memref<1x64x128xf32>) {
            ^bb0(%in: f32, %in_7: f32, %out: f32):
              %32 = arith.truncf %cst_0 : f64 to f32
              %33 = arith.mulf %in, %32 : f32
              %34 = arith.subf %33, %in_7 : f32
              %35 = math.powf %cst, %34 : f32
              linalg.yield %35 : f32
            }
            linalg.fill ins(%cst_1 : f32) outs(%20 : memref<1x64xf32>)
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%15 : memref<1x64x128xf32>) outs(%20 : memref<1x64xf32>) {
            ^bb0(%in: f32, %out: f32):
              %32 = arith.addf %in, %out : f32
              linalg.yield %32 : f32
            }
            linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%18, %19 : memref<1x64xf32>, memref<1x64xf32>) outs(%21 : memref<1x64xf32>) {
            ^bb0(%in: f32, %in_7: f32, %out: f32):
              %32 = arith.subf %in, %in_7 : f32
              %33 = math.powf %cst, %32 : f32
              linalg.yield %33 : f32
            }
            linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%17, %21, %20 : memref<1x64xf32>, memref<1x64xf32>, memref<1x64xf32>) outs(%17 : memref<1x64xf32>) {
            ^bb0(%in: f32, %in_7: f32, %in_8: f32, %out: f32):
              %32 = arith.mulf %in, %in_7 : f32
              %33 = arith.addf %32, %in_8 : f32
              linalg.yield %33 : f32
            }
            %30 = arith.muli %arg7, %c16384 : index
            %31 = arith.addi %25, %30 : index
            %reinterpret_cast_6 = memref.reinterpret_cast %arg1 to offset: [%31], sizes: [1, 128, 128], strides: [524288, 128, 1] : memref<32x4096x128xf32> to memref<1x128x128xf32, strided<[524288, 128, 1], offset: ?>>
            loom.copy %reinterpret_cast_6, %14 src_mem_space @DRAM dst_mem_space @L1, interconnect : [], broadcast : [1, 1] : memref<1x128x128xf32, strided<[524288, 128, 1], offset: ?>>, memref<1x128x128xf32>
            linalg.fill ins(%cst_1 : f32) outs(%24 : memref<1x64x128xf32>)
            linalg.batch_matmul ins(%15, %14 : memref<1x64x128xf32>, memref<1x128x128xf32>) outs(%24 : memref<1x64x128xf32>)
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%24, %23, %21 : memref<1x64x128xf32>, memref<1x64x128xf32>, memref<1x64xf32>) outs(%23 : memref<1x64x128xf32>) {
            ^bb0(%in: f32, %in_7: f32, %in_8: f32, %out: f32):
              %32 = arith.mulf %in_7, %in_8 : f32
              %33 = arith.addf %in, %32 : f32
              linalg.yield %33 : f32
            }
            linalg.copy ins(%19 : memref<1x64xf32>) outs(%18 : memref<1x64xf32>)
          }
          linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%23, %17 : memref<1x64x128xf32>, memref<1x64xf32>) outs(%22 : memref<1x64x128xf32>) {
          ^bb0(%in: f32, %in_5: f32, %out: f32):
            %28 = arith.divf %in, %in_5 : f32
            linalg.yield %28 : f32
          }
          %reinterpret_cast_4 = memref.reinterpret_cast %arg3 to offset: [%27], sizes: [1, 64, 128], strides: [524288, 128, 1] : memref<32x4096x128xf32> to memref<1x64x128xf32, strided<[524288, 128, 1], offset: ?>>
          loom.copy %22, %reinterpret_cast_4 src_mem_space @L1 dst_mem_space @DRAM, interconnect : [], broadcast : [1, 1] : memref<1x64x128xf32>, memref<1x64x128xf32, strided<[524288, 128, 1], offset: ?>>
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
          %15 = loom.alloc [1, 64, 128] on @L1 : memref<1x64x128xf32>
          %16 = loom.alloc [1, 128, 128] on @L1 : memref<1x128x128xf32>
          %17 = loom.alloc [1, 64] on @L1 : memref<1x64xf32>
          %18 = loom.alloc [1, 64] on @L1 : memref<1x64xf32>
          %19 = loom.alloc [1, 64] on @L1 : memref<1x64xf32>
          %20 = loom.alloc [1, 64] on @L1 : memref<1x64xf32>
          %21 = loom.alloc [1, 64] on @L1 : memref<1x64xf32>
          %22 = loom.alloc [1, 64, 128] on @L1 : memref<1x64x128xf32>
          %23 = loom.alloc [1, 64, 128] on @L1 : memref<1x64x128xf32>
          %24 = loom.alloc [1, 64, 128] on @L1 : memref<1x64x128xf32>
          %25 = arith.muli %13, %c524288 overflow<nsw> : index
          %26 = arith.muli %arg6, %c8192 : index
          %27 = arith.addi %25, %26 : index
          %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%27], sizes: [1, 64, 128], strides: [524288, 128, 1] : memref<32x4096x128xf32> to memref<1x64x128xf32, strided<[524288, 128, 1], offset: ?>>
          loom.copy %reinterpret_cast, %22 src_mem_space @DRAM dst_mem_space @L1, interconnect : [], broadcast : [1, 1] : memref<1x64x128xf32, strided<[524288, 128, 1], offset: ?>>, memref<1x64x128xf32>
          linalg.fill ins(%cst_1 : f32) outs(%23 : memref<1x64x128xf32>)
          linalg.fill ins(%cst_2 : f32) outs(%17 : memref<1x64xf32>)
          linalg.fill ins(%cst_3 : f32) outs(%18 : memref<1x64xf32>)
          scf.for %arg7 = %c0 to %c32 step %c1 {
            %28 = arith.muli %arg7, %c128 : index
            %29 = arith.addi %25, %28 : index
            %reinterpret_cast_5 = memref.reinterpret_cast %arg0 to offset: [%29], sizes: [1, 128, 128], strides: [524288, 4096, 1] : memref<32x128x4096xf32> to memref<1x128x128xf32, strided<[524288, 4096, 1], offset: ?>>
            loom.copy %reinterpret_cast_5, %16 src_mem_space @DRAM dst_mem_space @L1, interconnect : [], broadcast : [1, 1] : memref<1x128x128xf32, strided<[524288, 4096, 1], offset: ?>>, memref<1x128x128xf32>
            linalg.fill ins(%cst_1 : f32) outs(%15 : memref<1x64x128xf32>)
            linalg.batch_matmul ins(%22, %16 : memref<1x64x128xf32>, memref<1x128x128xf32>) outs(%15 : memref<1x64x128xf32>)
            linalg.fill ins(%cst_3 : f32) outs(%19 : memref<1x64xf32>)
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%15 : memref<1x64x128xf32>) outs(%19 : memref<1x64xf32>) {
            ^bb0(%in: f32, %out: f32):
              %32 = arith.maximumf %in, %out : f32
              linalg.yield %32 : f32
            }
            linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%18, %19 : memref<1x64xf32>, memref<1x64xf32>) outs(%19 : memref<1x64xf32>) {
            ^bb0(%in: f32, %in_7: f32, %out: f32):
              %32 = arith.truncf %cst_0 : f64 to f32
              %33 = arith.mulf %in_7, %32 : f32
              %34 = arith.cmpf ogt, %in, %33 : f32
              %35 = arith.select %34, %in, %33 : f32
              linalg.yield %35 : f32
            }
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%15, %19 : memref<1x64x128xf32>, memref<1x64xf32>) outs(%15 : memref<1x64x128xf32>) {
            ^bb0(%in: f32, %in_7: f32, %out: f32):
              %32 = arith.truncf %cst_0 : f64 to f32
              %33 = arith.mulf %in, %32 : f32
              %34 = arith.subf %33, %in_7 : f32
              %35 = math.powf %cst, %34 : f32
              linalg.yield %35 : f32
            }
            linalg.fill ins(%cst_1 : f32) outs(%20 : memref<1x64xf32>)
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%15 : memref<1x64x128xf32>) outs(%20 : memref<1x64xf32>) {
            ^bb0(%in: f32, %out: f32):
              %32 = arith.addf %in, %out : f32
              linalg.yield %32 : f32
            }
            linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%18, %19 : memref<1x64xf32>, memref<1x64xf32>) outs(%21 : memref<1x64xf32>) {
            ^bb0(%in: f32, %in_7: f32, %out: f32):
              %32 = arith.subf %in, %in_7 : f32
              %33 = math.powf %cst, %32 : f32
              linalg.yield %33 : f32
            }
            linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%17, %21, %20 : memref<1x64xf32>, memref<1x64xf32>, memref<1x64xf32>) outs(%17 : memref<1x64xf32>) {
            ^bb0(%in: f32, %in_7: f32, %in_8: f32, %out: f32):
              %32 = arith.mulf %in, %in_7 : f32
              %33 = arith.addf %32, %in_8 : f32
              linalg.yield %33 : f32
            }
            %30 = arith.muli %arg7, %c16384 : index
            %31 = arith.addi %25, %30 : index
            %reinterpret_cast_6 = memref.reinterpret_cast %arg1 to offset: [%31], sizes: [1, 128, 128], strides: [524288, 128, 1] : memref<32x4096x128xf32> to memref<1x128x128xf32, strided<[524288, 128, 1], offset: ?>>
            loom.copy %reinterpret_cast_6, %14 src_mem_space @DRAM dst_mem_space @L1, interconnect : [], broadcast : [1, 1] : memref<1x128x128xf32, strided<[524288, 128, 1], offset: ?>>, memref<1x128x128xf32>
            linalg.fill ins(%cst_1 : f32) outs(%24 : memref<1x64x128xf32>)
            linalg.batch_matmul ins(%15, %14 : memref<1x64x128xf32>, memref<1x128x128xf32>) outs(%24 : memref<1x64x128xf32>)
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%24, %23, %21 : memref<1x64x128xf32>, memref<1x64x128xf32>, memref<1x64xf32>) outs(%23 : memref<1x64x128xf32>) {
            ^bb0(%in: f32, %in_7: f32, %in_8: f32, %out: f32):
              %32 = arith.mulf %in_7, %in_8 : f32
              %33 = arith.addf %in, %32 : f32
              linalg.yield %33 : f32
            }
            linalg.copy ins(%19 : memref<1x64xf32>) outs(%18 : memref<1x64xf32>)
          }
          linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%23, %17 : memref<1x64x128xf32>, memref<1x64xf32>) outs(%22 : memref<1x64x128xf32>) {
          ^bb0(%in: f32, %in_5: f32, %out: f32):
            %28 = arith.divf %in, %in_5 : f32
            linalg.yield %28 : f32
          }
          %reinterpret_cast_4 = memref.reinterpret_cast %arg3 to offset: [%27], sizes: [1, 64, 128], strides: [524288, 128, 1] : memref<32x4096x128xf32> to memref<1x64x128xf32, strided<[524288, 128, 1], offset: ?>>
          loom.copy %22, %reinterpret_cast_4 src_mem_space @L1 dst_mem_space @DRAM, interconnect : [], broadcast : [1, 1] : memref<1x64x128xf32>, memref<1x64x128xf32, strided<[524288, 128, 1], offset: ?>>
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
          %15 = loom.alloc [1, 64, 128] on @L1 : memref<1x64x128xf32>
          %16 = loom.alloc [1, 128, 128] on @L1 : memref<1x128x128xf32>
          %17 = loom.alloc [1, 64] on @L1 : memref<1x64xf32>
          %18 = loom.alloc [1, 64] on @L1 : memref<1x64xf32>
          %19 = loom.alloc [1, 64] on @L1 : memref<1x64xf32>
          %20 = loom.alloc [1, 64] on @L1 : memref<1x64xf32>
          %21 = loom.alloc [1, 64] on @L1 : memref<1x64xf32>
          %22 = loom.alloc [1, 64, 128] on @L1 : memref<1x64x128xf32>
          %23 = loom.alloc [1, 64, 128] on @L1 : memref<1x64x128xf32>
          %24 = loom.alloc [1, 64, 128] on @L1 : memref<1x64x128xf32>
          %25 = arith.muli %13, %c524288 overflow<nsw> : index
          %26 = arith.muli %arg6, %c8192 : index
          %27 = arith.addi %25, %26 : index
          %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%27], sizes: [1, 64, 128], strides: [524288, 128, 1] : memref<32x4096x128xf32> to memref<1x64x128xf32, strided<[524288, 128, 1], offset: ?>>
          loom.copy %reinterpret_cast, %22 src_mem_space @DRAM dst_mem_space @L1, interconnect : [], broadcast : [1, 1] : memref<1x64x128xf32, strided<[524288, 128, 1], offset: ?>>, memref<1x64x128xf32>
          linalg.fill ins(%cst_1 : f32) outs(%23 : memref<1x64x128xf32>)
          linalg.fill ins(%cst_2 : f32) outs(%17 : memref<1x64xf32>)
          linalg.fill ins(%cst_3 : f32) outs(%18 : memref<1x64xf32>)
          scf.for %arg7 = %c0 to %c32 step %c1 {
            %28 = arith.muli %arg7, %c128 : index
            %29 = arith.addi %25, %28 : index
            %reinterpret_cast_5 = memref.reinterpret_cast %arg0 to offset: [%29], sizes: [1, 128, 128], strides: [524288, 4096, 1] : memref<32x128x4096xf32> to memref<1x128x128xf32, strided<[524288, 4096, 1], offset: ?>>
            loom.copy %reinterpret_cast_5, %16 src_mem_space @DRAM dst_mem_space @L1, interconnect : [], broadcast : [1, 1] : memref<1x128x128xf32, strided<[524288, 4096, 1], offset: ?>>, memref<1x128x128xf32>
            linalg.fill ins(%cst_1 : f32) outs(%15 : memref<1x64x128xf32>)
            linalg.batch_matmul ins(%22, %16 : memref<1x64x128xf32>, memref<1x128x128xf32>) outs(%15 : memref<1x64x128xf32>)
            linalg.fill ins(%cst_3 : f32) outs(%19 : memref<1x64xf32>)
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%15 : memref<1x64x128xf32>) outs(%19 : memref<1x64xf32>) {
            ^bb0(%in: f32, %out: f32):
              %32 = arith.maximumf %in, %out : f32
              linalg.yield %32 : f32
            }
            linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%18, %19 : memref<1x64xf32>, memref<1x64xf32>) outs(%19 : memref<1x64xf32>) {
            ^bb0(%in: f32, %in_7: f32, %out: f32):
              %32 = arith.truncf %cst_0 : f64 to f32
              %33 = arith.mulf %in_7, %32 : f32
              %34 = arith.cmpf ogt, %in, %33 : f32
              %35 = arith.select %34, %in, %33 : f32
              linalg.yield %35 : f32
            }
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%15, %19 : memref<1x64x128xf32>, memref<1x64xf32>) outs(%15 : memref<1x64x128xf32>) {
            ^bb0(%in: f32, %in_7: f32, %out: f32):
              %32 = arith.truncf %cst_0 : f64 to f32
              %33 = arith.mulf %in, %32 : f32
              %34 = arith.subf %33, %in_7 : f32
              %35 = math.powf %cst, %34 : f32
              linalg.yield %35 : f32
            }
            linalg.fill ins(%cst_1 : f32) outs(%20 : memref<1x64xf32>)
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%15 : memref<1x64x128xf32>) outs(%20 : memref<1x64xf32>) {
            ^bb0(%in: f32, %out: f32):
              %32 = arith.addf %in, %out : f32
              linalg.yield %32 : f32
            }
            linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%18, %19 : memref<1x64xf32>, memref<1x64xf32>) outs(%21 : memref<1x64xf32>) {
            ^bb0(%in: f32, %in_7: f32, %out: f32):
              %32 = arith.subf %in, %in_7 : f32
              %33 = math.powf %cst, %32 : f32
              linalg.yield %33 : f32
            }
            linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%17, %21, %20 : memref<1x64xf32>, memref<1x64xf32>, memref<1x64xf32>) outs(%17 : memref<1x64xf32>) {
            ^bb0(%in: f32, %in_7: f32, %in_8: f32, %out: f32):
              %32 = arith.mulf %in, %in_7 : f32
              %33 = arith.addf %32, %in_8 : f32
              linalg.yield %33 : f32
            }
            %30 = arith.muli %arg7, %c16384 : index
            %31 = arith.addi %25, %30 : index
            %reinterpret_cast_6 = memref.reinterpret_cast %arg1 to offset: [%31], sizes: [1, 128, 128], strides: [524288, 128, 1] : memref<32x4096x128xf32> to memref<1x128x128xf32, strided<[524288, 128, 1], offset: ?>>
            loom.copy %reinterpret_cast_6, %14 src_mem_space @DRAM dst_mem_space @L1, interconnect : [], broadcast : [1, 1] : memref<1x128x128xf32, strided<[524288, 128, 1], offset: ?>>, memref<1x128x128xf32>
            linalg.fill ins(%cst_1 : f32) outs(%24 : memref<1x64x128xf32>)
            linalg.batch_matmul ins(%15, %14 : memref<1x64x128xf32>, memref<1x128x128xf32>) outs(%24 : memref<1x64x128xf32>)
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%24, %23, %21 : memref<1x64x128xf32>, memref<1x64x128xf32>, memref<1x64xf32>) outs(%23 : memref<1x64x128xf32>) {
            ^bb0(%in: f32, %in_7: f32, %in_8: f32, %out: f32):
              %32 = arith.mulf %in_7, %in_8 : f32
              %33 = arith.addf %in, %32 : f32
              linalg.yield %33 : f32
            }
            linalg.copy ins(%19 : memref<1x64xf32>) outs(%18 : memref<1x64xf32>)
          }
          linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%23, %17 : memref<1x64x128xf32>, memref<1x64xf32>) outs(%22 : memref<1x64x128xf32>) {
          ^bb0(%in: f32, %in_5: f32, %out: f32):
            %28 = arith.divf %in, %in_5 : f32
            linalg.yield %28 : f32
          }
          %reinterpret_cast_4 = memref.reinterpret_cast %arg3 to offset: [%27], sizes: [1, 64, 128], strides: [524288, 128, 1] : memref<32x4096x128xf32> to memref<1x64x128xf32, strided<[524288, 128, 1], offset: ?>>
          loom.copy %22, %reinterpret_cast_4 src_mem_space @L1 dst_mem_space @DRAM, interconnect : [], broadcast : [1, 1] : memref<1x64x128xf32>, memref<1x64x128xf32, strided<[524288, 128, 1], offset: ?>>
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
            %17 = loom.alloc [1, 64, 128] on @L1 : memref<1x64x128xf32>
            %18 = loom.alloc [1, 128, 128] on @L1 : memref<1x128x128xf32>
            %19 = loom.alloc [1, 64] on @L1 : memref<1x64xf32>
            %20 = loom.alloc [1, 64] on @L1 : memref<1x64xf32>
            %21 = loom.alloc [1, 64] on @L1 : memref<1x64xf32>
            %22 = loom.alloc [1, 64] on @L1 : memref<1x64xf32>
            %23 = loom.alloc [1, 64] on @L1 : memref<1x64xf32>
            %24 = loom.alloc [1, 64, 128] on @L1 : memref<1x64x128xf32>
            %25 = loom.alloc [1, 64, 128] on @L1 : memref<1x64x128xf32>
            %26 = loom.alloc [1, 64, 128] on @L1 : memref<1x64x128xf32>
            %27 = arith.muli %13, %c524288 overflow<nsw> : index
            %28 = arith.muli %15, %c8192 : index
            %29 = arith.addi %27, %28 : index
            %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%29], sizes: [1, 64, 128], strides: [524288, 128, 1] : memref<32x4096x128xf32> to memref<1x64x128xf32, strided<[524288, 128, 1], offset: ?>>
            loom.copy %reinterpret_cast, %24 src_mem_space @DRAM dst_mem_space @L1, interconnect : [], broadcast : [1, 1] : memref<1x64x128xf32, strided<[524288, 128, 1], offset: ?>>, memref<1x64x128xf32>
            linalg.fill ins(%cst_1 : f32) outs(%25 : memref<1x64x128xf32>)
            linalg.fill ins(%cst_2 : f32) outs(%19 : memref<1x64xf32>)
            linalg.fill ins(%cst_3 : f32) outs(%20 : memref<1x64xf32>)
            scf.for %arg8 = %c0 to %c32 step %c1 {
              %30 = arith.muli %arg8, %c128 : index
              %31 = arith.addi %27, %30 : index
              %reinterpret_cast_5 = memref.reinterpret_cast %arg0 to offset: [%31], sizes: [1, 128, 128], strides: [524288, 4096, 1] : memref<32x128x4096xf32> to memref<1x128x128xf32, strided<[524288, 4096, 1], offset: ?>>
              loom.copy %reinterpret_cast_5, %18 src_mem_space @DRAM dst_mem_space @L1, interconnect : [], broadcast : [1, 1] : memref<1x128x128xf32, strided<[524288, 4096, 1], offset: ?>>, memref<1x128x128xf32>
              linalg.fill ins(%cst_1 : f32) outs(%17 : memref<1x64x128xf32>)
              linalg.batch_matmul ins(%24, %18 : memref<1x64x128xf32>, memref<1x128x128xf32>) outs(%17 : memref<1x64x128xf32>)
              linalg.fill ins(%cst_3 : f32) outs(%21 : memref<1x64xf32>)
              linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%17 : memref<1x64x128xf32>) outs(%21 : memref<1x64xf32>) {
              ^bb0(%in: f32, %out: f32):
                %34 = arith.maximumf %in, %out : f32
                linalg.yield %34 : f32
              }
              linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%20, %21 : memref<1x64xf32>, memref<1x64xf32>) outs(%21 : memref<1x64xf32>) {
              ^bb0(%in: f32, %in_7: f32, %out: f32):
                %34 = arith.truncf %cst_0 : f64 to f32
                %35 = arith.mulf %in_7, %34 : f32
                %36 = arith.cmpf ogt, %in, %35 : f32
                %37 = arith.select %36, %in, %35 : f32
                linalg.yield %37 : f32
              }
              linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%17, %21 : memref<1x64x128xf32>, memref<1x64xf32>) outs(%17 : memref<1x64x128xf32>) {
              ^bb0(%in: f32, %in_7: f32, %out: f32):
                %34 = arith.truncf %cst_0 : f64 to f32
                %35 = arith.mulf %in, %34 : f32
                %36 = arith.subf %35, %in_7 : f32
                %37 = math.powf %cst, %36 : f32
                linalg.yield %37 : f32
              }
              linalg.fill ins(%cst_1 : f32) outs(%22 : memref<1x64xf32>)
              linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%17 : memref<1x64x128xf32>) outs(%22 : memref<1x64xf32>) {
              ^bb0(%in: f32, %out: f32):
                %34 = arith.addf %in, %out : f32
                linalg.yield %34 : f32
              }
              linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%20, %21 : memref<1x64xf32>, memref<1x64xf32>) outs(%23 : memref<1x64xf32>) {
              ^bb0(%in: f32, %in_7: f32, %out: f32):
                %34 = arith.subf %in, %in_7 : f32
                %35 = math.powf %cst, %34 : f32
                linalg.yield %35 : f32
              }
              linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%19, %23, %22 : memref<1x64xf32>, memref<1x64xf32>, memref<1x64xf32>) outs(%19 : memref<1x64xf32>) {
              ^bb0(%in: f32, %in_7: f32, %in_8: f32, %out: f32):
                %34 = arith.mulf %in, %in_7 : f32
                %35 = arith.addf %34, %in_8 : f32
                linalg.yield %35 : f32
              }
              %32 = arith.muli %arg8, %c16384 : index
              %33 = arith.addi %27, %32 : index
              %reinterpret_cast_6 = memref.reinterpret_cast %arg1 to offset: [%33], sizes: [1, 128, 128], strides: [524288, 128, 1] : memref<32x4096x128xf32> to memref<1x128x128xf32, strided<[524288, 128, 1], offset: ?>>
              loom.copy %reinterpret_cast_6, %16 src_mem_space @DRAM dst_mem_space @L1, interconnect : [], broadcast : [1, 1] : memref<1x128x128xf32, strided<[524288, 128, 1], offset: ?>>, memref<1x128x128xf32>
              linalg.fill ins(%cst_1 : f32) outs(%26 : memref<1x64x128xf32>)
              linalg.batch_matmul ins(%17, %16 : memref<1x64x128xf32>, memref<1x128x128xf32>) outs(%26 : memref<1x64x128xf32>)
              linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%26, %25, %23 : memref<1x64x128xf32>, memref<1x64x128xf32>, memref<1x64xf32>) outs(%25 : memref<1x64x128xf32>) {
              ^bb0(%in: f32, %in_7: f32, %in_8: f32, %out: f32):
                %34 = arith.mulf %in_7, %in_8 : f32
                %35 = arith.addf %in, %34 : f32
                linalg.yield %35 : f32
              }
              linalg.copy ins(%21 : memref<1x64xf32>) outs(%20 : memref<1x64xf32>)
            }
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%25, %19 : memref<1x64x128xf32>, memref<1x64xf32>) outs(%24 : memref<1x64x128xf32>) {
            ^bb0(%in: f32, %in_5: f32, %out: f32):
              %30 = arith.divf %in, %in_5 : f32
              linalg.yield %30 : f32
            }
            %reinterpret_cast_4 = memref.reinterpret_cast %arg3 to offset: [%29], sizes: [1, 64, 128], strides: [524288, 128, 1] : memref<32x4096x128xf32> to memref<1x64x128xf32, strided<[524288, 128, 1], offset: ?>>
            loom.copy %24, %reinterpret_cast_4 src_mem_space @L1 dst_mem_space @DRAM, interconnect : [], broadcast : [1, 1] : memref<1x64x128xf32>, memref<1x64x128xf32, strided<[524288, 128, 1], offset: ?>>
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
            %17 = loom.alloc [1, 64, 128] on @L1 : memref<1x64x128xf32>
            %18 = loom.alloc [1, 128, 128] on @L1 : memref<1x128x128xf32>
            %19 = loom.alloc [1, 64] on @L1 : memref<1x64xf32>
            %20 = loom.alloc [1, 64] on @L1 : memref<1x64xf32>
            %21 = loom.alloc [1, 64] on @L1 : memref<1x64xf32>
            %22 = loom.alloc [1, 64] on @L1 : memref<1x64xf32>
            %23 = loom.alloc [1, 64] on @L1 : memref<1x64xf32>
            %24 = loom.alloc [1, 64, 128] on @L1 : memref<1x64x128xf32>
            %25 = loom.alloc [1, 64, 128] on @L1 : memref<1x64x128xf32>
            %26 = loom.alloc [1, 64, 128] on @L1 : memref<1x64x128xf32>
            %27 = arith.muli %13, %c524288 overflow<nsw> : index
            %28 = arith.muli %15, %c8192 : index
            %29 = arith.addi %27, %28 : index
            %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%29], sizes: [1, 64, 128], strides: [524288, 128, 1] : memref<32x4096x128xf32> to memref<1x64x128xf32, strided<[524288, 128, 1], offset: ?>>
            loom.copy %reinterpret_cast, %24 src_mem_space @DRAM dst_mem_space @L1, interconnect : [], broadcast : [1, 1] : memref<1x64x128xf32, strided<[524288, 128, 1], offset: ?>>, memref<1x64x128xf32>
            linalg.fill ins(%cst_1 : f32) outs(%25 : memref<1x64x128xf32>)
            linalg.fill ins(%cst_2 : f32) outs(%19 : memref<1x64xf32>)
            linalg.fill ins(%cst_3 : f32) outs(%20 : memref<1x64xf32>)
            scf.for %arg8 = %c0 to %c32 step %c1 {
              %30 = arith.muli %arg8, %c128 : index
              %31 = arith.addi %27, %30 : index
              %reinterpret_cast_5 = memref.reinterpret_cast %arg0 to offset: [%31], sizes: [1, 128, 128], strides: [524288, 4096, 1] : memref<32x128x4096xf32> to memref<1x128x128xf32, strided<[524288, 4096, 1], offset: ?>>
              loom.copy %reinterpret_cast_5, %18 src_mem_space @DRAM dst_mem_space @L1, interconnect : [], broadcast : [1, 1] : memref<1x128x128xf32, strided<[524288, 4096, 1], offset: ?>>, memref<1x128x128xf32>
              linalg.fill ins(%cst_1 : f32) outs(%17 : memref<1x64x128xf32>)
              linalg.batch_matmul ins(%24, %18 : memref<1x64x128xf32>, memref<1x128x128xf32>) outs(%17 : memref<1x64x128xf32>)
              linalg.fill ins(%cst_3 : f32) outs(%21 : memref<1x64xf32>)
              linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%17 : memref<1x64x128xf32>) outs(%21 : memref<1x64xf32>) {
              ^bb0(%in: f32, %out: f32):
                %34 = arith.maximumf %in, %out : f32
                linalg.yield %34 : f32
              }
              linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%20, %21 : memref<1x64xf32>, memref<1x64xf32>) outs(%21 : memref<1x64xf32>) {
              ^bb0(%in: f32, %in_7: f32, %out: f32):
                %34 = arith.truncf %cst_0 : f64 to f32
                %35 = arith.mulf %in_7, %34 : f32
                %36 = arith.cmpf ogt, %in, %35 : f32
                %37 = arith.select %36, %in, %35 : f32
                linalg.yield %37 : f32
              }
              linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%17, %21 : memref<1x64x128xf32>, memref<1x64xf32>) outs(%17 : memref<1x64x128xf32>) {
              ^bb0(%in: f32, %in_7: f32, %out: f32):
                %34 = arith.truncf %cst_0 : f64 to f32
                %35 = arith.mulf %in, %34 : f32
                %36 = arith.subf %35, %in_7 : f32
                %37 = math.powf %cst, %36 : f32
                linalg.yield %37 : f32
              }
              linalg.fill ins(%cst_1 : f32) outs(%22 : memref<1x64xf32>)
              linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%17 : memref<1x64x128xf32>) outs(%22 : memref<1x64xf32>) {
              ^bb0(%in: f32, %out: f32):
                %34 = arith.addf %in, %out : f32
                linalg.yield %34 : f32
              }
              linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%20, %21 : memref<1x64xf32>, memref<1x64xf32>) outs(%23 : memref<1x64xf32>) {
              ^bb0(%in: f32, %in_7: f32, %out: f32):
                %34 = arith.subf %in, %in_7 : f32
                %35 = math.powf %cst, %34 : f32
                linalg.yield %35 : f32
              }
              linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%19, %23, %22 : memref<1x64xf32>, memref<1x64xf32>, memref<1x64xf32>) outs(%19 : memref<1x64xf32>) {
              ^bb0(%in: f32, %in_7: f32, %in_8: f32, %out: f32):
                %34 = arith.mulf %in, %in_7 : f32
                %35 = arith.addf %34, %in_8 : f32
                linalg.yield %35 : f32
              }
              %32 = arith.muli %arg8, %c16384 : index
              %33 = arith.addi %27, %32 : index
              %reinterpret_cast_6 = memref.reinterpret_cast %arg1 to offset: [%33], sizes: [1, 128, 128], strides: [524288, 128, 1] : memref<32x4096x128xf32> to memref<1x128x128xf32, strided<[524288, 128, 1], offset: ?>>
              loom.copy %reinterpret_cast_6, %16 src_mem_space @DRAM dst_mem_space @L1, interconnect : [@vertical_links], broadcast : [8, 1] : memref<1x128x128xf32, strided<[524288, 128, 1], offset: ?>>, memref<1x128x128xf32>
              linalg.fill ins(%cst_1 : f32) outs(%26 : memref<1x64x128xf32>)
              linalg.batch_matmul ins(%17, %16 : memref<1x64x128xf32>, memref<1x128x128xf32>) outs(%26 : memref<1x64x128xf32>)
              linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%26, %25, %23 : memref<1x64x128xf32>, memref<1x64x128xf32>, memref<1x64xf32>) outs(%25 : memref<1x64x128xf32>) {
              ^bb0(%in: f32, %in_7: f32, %in_8: f32, %out: f32):
                %34 = arith.mulf %in_7, %in_8 : f32
                %35 = arith.addf %in, %34 : f32
                linalg.yield %35 : f32
              }
              linalg.copy ins(%21 : memref<1x64xf32>) outs(%20 : memref<1x64xf32>)
            }
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%25, %19 : memref<1x64x128xf32>, memref<1x64xf32>) outs(%24 : memref<1x64x128xf32>) {
            ^bb0(%in: f32, %in_5: f32, %out: f32):
              %30 = arith.divf %in, %in_5 : f32
              linalg.yield %30 : f32
            }
            %reinterpret_cast_4 = memref.reinterpret_cast %arg3 to offset: [%29], sizes: [1, 64, 128], strides: [524288, 128, 1] : memref<32x4096x128xf32> to memref<1x64x128xf32, strided<[524288, 128, 1], offset: ?>>
            loom.copy %24, %reinterpret_cast_4 src_mem_space @L1 dst_mem_space @DRAM, interconnect : [], broadcast : [1, 1] : memref<1x64x128xf32>, memref<1x64x128xf32, strided<[524288, 128, 1], offset: ?>>
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
            %17 = loom.alloc [1, 64, 128] on @L1 : memref<1x64x128xf32>
            %18 = loom.alloc [1, 128, 128] on @L1 : memref<1x128x128xf32>
            %19 = loom.alloc [1, 64] on @L1 : memref<1x64xf32>
            %20 = loom.alloc [1, 64] on @L1 : memref<1x64xf32>
            %21 = loom.alloc [1, 64] on @L1 : memref<1x64xf32>
            %22 = loom.alloc [1, 64] on @L1 : memref<1x64xf32>
            %23 = loom.alloc [1, 64] on @L1 : memref<1x64xf32>
            %24 = loom.alloc [1, 64, 128] on @L1 : memref<1x64x128xf32>
            %25 = loom.alloc [1, 64, 128] on @L1 : memref<1x64x128xf32>
            %26 = loom.alloc [1, 64, 128] on @L1 : memref<1x64x128xf32>
            %27 = arith.muli %13, %c524288 overflow<nsw> : index
            %28 = arith.muli %15, %c8192 : index
            %29 = arith.addi %27, %28 : index
            %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%29], sizes: [1, 64, 128], strides: [524288, 128, 1] : memref<32x4096x128xf32> to memref<1x64x128xf32, strided<[524288, 128, 1], offset: ?>>
            loom.copy %reinterpret_cast, %24 src_mem_space @DRAM dst_mem_space @L1, interconnect : [], broadcast : [1, 1] : memref<1x64x128xf32, strided<[524288, 128, 1], offset: ?>>, memref<1x64x128xf32>
            linalg.fill ins(%cst_1 : f32) outs(%25 : memref<1x64x128xf32>)
            linalg.fill ins(%cst_2 : f32) outs(%19 : memref<1x64xf32>)
            linalg.fill ins(%cst_3 : f32) outs(%20 : memref<1x64xf32>)
            scf.for %arg8 = %c0 to %c32 step %c1 {
              %30 = arith.muli %arg8, %c128 : index
              %31 = arith.addi %27, %30 : index
              %reinterpret_cast_5 = memref.reinterpret_cast %arg0 to offset: [%31], sizes: [1, 128, 128], strides: [524288, 4096, 1] : memref<32x128x4096xf32> to memref<1x128x128xf32, strided<[524288, 4096, 1], offset: ?>>
              loom.copy %reinterpret_cast_5, %18 src_mem_space @DRAM dst_mem_space @L1, interconnect : [@vertical_links], broadcast : [8, 1] : memref<1x128x128xf32, strided<[524288, 4096, 1], offset: ?>>, memref<1x128x128xf32>
              linalg.fill ins(%cst_1 : f32) outs(%17 : memref<1x64x128xf32>)
              linalg.batch_matmul ins(%24, %18 : memref<1x64x128xf32>, memref<1x128x128xf32>) outs(%17 : memref<1x64x128xf32>)
              linalg.fill ins(%cst_3 : f32) outs(%21 : memref<1x64xf32>)
              linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%17 : memref<1x64x128xf32>) outs(%21 : memref<1x64xf32>) {
              ^bb0(%in: f32, %out: f32):
                %34 = arith.maximumf %in, %out : f32
                linalg.yield %34 : f32
              }
              linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%20, %21 : memref<1x64xf32>, memref<1x64xf32>) outs(%21 : memref<1x64xf32>) {
              ^bb0(%in: f32, %in_7: f32, %out: f32):
                %34 = arith.truncf %cst_0 : f64 to f32
                %35 = arith.mulf %in_7, %34 : f32
                %36 = arith.cmpf ogt, %in, %35 : f32
                %37 = arith.select %36, %in, %35 : f32
                linalg.yield %37 : f32
              }
              linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%17, %21 : memref<1x64x128xf32>, memref<1x64xf32>) outs(%17 : memref<1x64x128xf32>) {
              ^bb0(%in: f32, %in_7: f32, %out: f32):
                %34 = arith.truncf %cst_0 : f64 to f32
                %35 = arith.mulf %in, %34 : f32
                %36 = arith.subf %35, %in_7 : f32
                %37 = math.powf %cst, %36 : f32
                linalg.yield %37 : f32
              }
              linalg.fill ins(%cst_1 : f32) outs(%22 : memref<1x64xf32>)
              linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%17 : memref<1x64x128xf32>) outs(%22 : memref<1x64xf32>) {
              ^bb0(%in: f32, %out: f32):
                %34 = arith.addf %in, %out : f32
                linalg.yield %34 : f32
              }
              linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%20, %21 : memref<1x64xf32>, memref<1x64xf32>) outs(%23 : memref<1x64xf32>) {
              ^bb0(%in: f32, %in_7: f32, %out: f32):
                %34 = arith.subf %in, %in_7 : f32
                %35 = math.powf %cst, %34 : f32
                linalg.yield %35 : f32
              }
              linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%19, %23, %22 : memref<1x64xf32>, memref<1x64xf32>, memref<1x64xf32>) outs(%19 : memref<1x64xf32>) {
              ^bb0(%in: f32, %in_7: f32, %in_8: f32, %out: f32):
                %34 = arith.mulf %in, %in_7 : f32
                %35 = arith.addf %34, %in_8 : f32
                linalg.yield %35 : f32
              }
              %32 = arith.muli %arg8, %c16384 : index
              %33 = arith.addi %27, %32 : index
              %reinterpret_cast_6 = memref.reinterpret_cast %arg1 to offset: [%33], sizes: [1, 128, 128], strides: [524288, 128, 1] : memref<32x4096x128xf32> to memref<1x128x128xf32, strided<[524288, 128, 1], offset: ?>>
              loom.copy %reinterpret_cast_6, %16 src_mem_space @DRAM dst_mem_space @L1, interconnect : [], broadcast : [1, 1] : memref<1x128x128xf32, strided<[524288, 128, 1], offset: ?>>, memref<1x128x128xf32>
              linalg.fill ins(%cst_1 : f32) outs(%26 : memref<1x64x128xf32>)
              linalg.batch_matmul ins(%17, %16 : memref<1x64x128xf32>, memref<1x128x128xf32>) outs(%26 : memref<1x64x128xf32>)
              linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%26, %25, %23 : memref<1x64x128xf32>, memref<1x64x128xf32>, memref<1x64xf32>) outs(%25 : memref<1x64x128xf32>) {
              ^bb0(%in: f32, %in_7: f32, %in_8: f32, %out: f32):
                %34 = arith.mulf %in_7, %in_8 : f32
                %35 = arith.addf %in, %34 : f32
                linalg.yield %35 : f32
              }
              linalg.copy ins(%21 : memref<1x64xf32>) outs(%20 : memref<1x64xf32>)
            }
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%25, %19 : memref<1x64x128xf32>, memref<1x64xf32>) outs(%24 : memref<1x64x128xf32>) {
            ^bb0(%in: f32, %in_5: f32, %out: f32):
              %30 = arith.divf %in, %in_5 : f32
              linalg.yield %30 : f32
            }
            %reinterpret_cast_4 = memref.reinterpret_cast %arg3 to offset: [%29], sizes: [1, 64, 128], strides: [524288, 128, 1] : memref<32x4096x128xf32> to memref<1x64x128xf32, strided<[524288, 128, 1], offset: ?>>
            loom.copy %24, %reinterpret_cast_4 src_mem_space @L1 dst_mem_space @DRAM, interconnect : [], broadcast : [1, 1] : memref<1x64x128xf32>, memref<1x64x128xf32, strided<[524288, 128, 1], offset: ?>>
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
            %17 = loom.alloc [1, 64, 128] on @L1 : memref<1x64x128xf32>
            %18 = loom.alloc [1, 128, 128] on @L1 : memref<1x128x128xf32>
            %19 = loom.alloc [1, 64] on @L1 : memref<1x64xf32>
            %20 = loom.alloc [1, 64] on @L1 : memref<1x64xf32>
            %21 = loom.alloc [1, 64] on @L1 : memref<1x64xf32>
            %22 = loom.alloc [1, 64] on @L1 : memref<1x64xf32>
            %23 = loom.alloc [1, 64] on @L1 : memref<1x64xf32>
            %24 = loom.alloc [1, 64, 128] on @L1 : memref<1x64x128xf32>
            %25 = loom.alloc [1, 64, 128] on @L1 : memref<1x64x128xf32>
            %26 = loom.alloc [1, 64, 128] on @L1 : memref<1x64x128xf32>
            %27 = arith.muli %13, %c524288 overflow<nsw> : index
            %28 = arith.muli %15, %c8192 : index
            %29 = arith.addi %27, %28 : index
            %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%29], sizes: [1, 64, 128], strides: [524288, 128, 1] : memref<32x4096x128xf32> to memref<1x64x128xf32, strided<[524288, 128, 1], offset: ?>>
            loom.copy %reinterpret_cast, %24 src_mem_space @DRAM dst_mem_space @L1, interconnect : [], broadcast : [1, 1] : memref<1x64x128xf32, strided<[524288, 128, 1], offset: ?>>, memref<1x64x128xf32>
            linalg.fill ins(%cst_1 : f32) outs(%25 : memref<1x64x128xf32>)
            linalg.fill ins(%cst_2 : f32) outs(%19 : memref<1x64xf32>)
            linalg.fill ins(%cst_3 : f32) outs(%20 : memref<1x64xf32>)
            scf.for %arg8 = %c0 to %c32 step %c1 {
              %30 = arith.muli %arg8, %c128 : index
              %31 = arith.addi %27, %30 : index
              %reinterpret_cast_5 = memref.reinterpret_cast %arg0 to offset: [%31], sizes: [1, 128, 128], strides: [524288, 4096, 1] : memref<32x128x4096xf32> to memref<1x128x128xf32, strided<[524288, 4096, 1], offset: ?>>
              loom.copy %reinterpret_cast_5, %18 src_mem_space @DRAM dst_mem_space @L1, interconnect : [@vertical_links], broadcast : [8, 1] : memref<1x128x128xf32, strided<[524288, 4096, 1], offset: ?>>, memref<1x128x128xf32>
              linalg.fill ins(%cst_1 : f32) outs(%17 : memref<1x64x128xf32>)
              linalg.batch_matmul ins(%24, %18 : memref<1x64x128xf32>, memref<1x128x128xf32>) outs(%17 : memref<1x64x128xf32>)
              linalg.fill ins(%cst_3 : f32) outs(%21 : memref<1x64xf32>)
              linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%17 : memref<1x64x128xf32>) outs(%21 : memref<1x64xf32>) {
              ^bb0(%in: f32, %out: f32):
                %34 = arith.maximumf %in, %out : f32
                linalg.yield %34 : f32
              }
              linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%20, %21 : memref<1x64xf32>, memref<1x64xf32>) outs(%21 : memref<1x64xf32>) {
              ^bb0(%in: f32, %in_7: f32, %out: f32):
                %34 = arith.truncf %cst_0 : f64 to f32
                %35 = arith.mulf %in_7, %34 : f32
                %36 = arith.cmpf ogt, %in, %35 : f32
                %37 = arith.select %36, %in, %35 : f32
                linalg.yield %37 : f32
              }
              linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%17, %21 : memref<1x64x128xf32>, memref<1x64xf32>) outs(%17 : memref<1x64x128xf32>) {
              ^bb0(%in: f32, %in_7: f32, %out: f32):
                %34 = arith.truncf %cst_0 : f64 to f32
                %35 = arith.mulf %in, %34 : f32
                %36 = arith.subf %35, %in_7 : f32
                %37 = math.powf %cst, %36 : f32
                linalg.yield %37 : f32
              }
              linalg.fill ins(%cst_1 : f32) outs(%22 : memref<1x64xf32>)
              linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%17 : memref<1x64x128xf32>) outs(%22 : memref<1x64xf32>) {
              ^bb0(%in: f32, %out: f32):
                %34 = arith.addf %in, %out : f32
                linalg.yield %34 : f32
              }
              linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%20, %21 : memref<1x64xf32>, memref<1x64xf32>) outs(%23 : memref<1x64xf32>) {
              ^bb0(%in: f32, %in_7: f32, %out: f32):
                %34 = arith.subf %in, %in_7 : f32
                %35 = math.powf %cst, %34 : f32
                linalg.yield %35 : f32
              }
              linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%19, %23, %22 : memref<1x64xf32>, memref<1x64xf32>, memref<1x64xf32>) outs(%19 : memref<1x64xf32>) {
              ^bb0(%in: f32, %in_7: f32, %in_8: f32, %out: f32):
                %34 = arith.mulf %in, %in_7 : f32
                %35 = arith.addf %34, %in_8 : f32
                linalg.yield %35 : f32
              }
              %32 = arith.muli %arg8, %c16384 : index
              %33 = arith.addi %27, %32 : index
              %reinterpret_cast_6 = memref.reinterpret_cast %arg1 to offset: [%33], sizes: [1, 128, 128], strides: [524288, 128, 1] : memref<32x4096x128xf32> to memref<1x128x128xf32, strided<[524288, 128, 1], offset: ?>>
              loom.copy %reinterpret_cast_6, %16 src_mem_space @DRAM dst_mem_space @L1, interconnect : [@vertical_links], broadcast : [8, 1] : memref<1x128x128xf32, strided<[524288, 128, 1], offset: ?>>, memref<1x128x128xf32>
              linalg.fill ins(%cst_1 : f32) outs(%26 : memref<1x64x128xf32>)
              linalg.batch_matmul ins(%17, %16 : memref<1x64x128xf32>, memref<1x128x128xf32>) outs(%26 : memref<1x64x128xf32>)
              linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%26, %25, %23 : memref<1x64x128xf32>, memref<1x64x128xf32>, memref<1x64xf32>) outs(%25 : memref<1x64x128xf32>) {
              ^bb0(%in: f32, %in_7: f32, %in_8: f32, %out: f32):
                %34 = arith.mulf %in_7, %in_8 : f32
                %35 = arith.addf %in, %34 : f32
                linalg.yield %35 : f32
              }
              linalg.copy ins(%21 : memref<1x64xf32>) outs(%20 : memref<1x64xf32>)
            }
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%25, %19 : memref<1x64x128xf32>, memref<1x64xf32>) outs(%24 : memref<1x64x128xf32>) {
            ^bb0(%in: f32, %in_5: f32, %out: f32):
              %30 = arith.divf %in, %in_5 : f32
              linalg.yield %30 : f32
            }
            %reinterpret_cast_4 = memref.reinterpret_cast %arg3 to offset: [%29], sizes: [1, 64, 128], strides: [524288, 128, 1] : memref<32x4096x128xf32> to memref<1x64x128xf32, strided<[524288, 128, 1], offset: ?>>
            loom.copy %24, %reinterpret_cast_4 src_mem_space @L1 dst_mem_space @DRAM, interconnect : [], broadcast : [1, 1] : memref<1x64x128xf32>, memref<1x64x128xf32, strided<[524288, 128, 1], offset: ?>>
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
            %17 = loom.alloc [1, 64, 128] on @L1 : memref<1x64x128xf32>
            %18 = loom.alloc [1, 128, 128] on @L1 : memref<1x128x128xf32>
            %19 = loom.alloc [1, 64] on @L1 : memref<1x64xf32>
            %20 = loom.alloc [1, 64] on @L1 : memref<1x64xf32>
            %21 = loom.alloc [1, 64] on @L1 : memref<1x64xf32>
            %22 = loom.alloc [1, 64] on @L1 : memref<1x64xf32>
            %23 = loom.alloc [1, 64] on @L1 : memref<1x64xf32>
            %24 = loom.alloc [1, 64, 128] on @L1 : memref<1x64x128xf32>
            %25 = loom.alloc [1, 64, 128] on @L1 : memref<1x64x128xf32>
            %26 = loom.alloc [1, 64, 128] on @L1 : memref<1x64x128xf32>
            %27 = arith.muli %13, %c524288 overflow<nsw> : index
            %28 = arith.muli %15, %c8192 : index
            %29 = arith.addi %27, %28 : index
            %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%29], sizes: [1, 64, 128], strides: [524288, 128, 1] : memref<32x4096x128xf32> to memref<1x64x128xf32, strided<[524288, 128, 1], offset: ?>>
            loom.copy %reinterpret_cast, %24 src_mem_space @DRAM dst_mem_space @L1, interconnect : [], broadcast : [1, 1] : memref<1x64x128xf32, strided<[524288, 128, 1], offset: ?>>, memref<1x64x128xf32>
            linalg.fill ins(%cst_1 : f32) outs(%25 : memref<1x64x128xf32>)
            linalg.fill ins(%cst_2 : f32) outs(%19 : memref<1x64xf32>)
            linalg.fill ins(%cst_3 : f32) outs(%20 : memref<1x64xf32>)
            scf.for %arg8 = %c0 to %c32 step %c1 {
              %30 = arith.muli %arg8, %c128 : index
              %31 = arith.addi %27, %30 : index
              %reinterpret_cast_5 = memref.reinterpret_cast %arg0 to offset: [%31], sizes: [1, 128, 128], strides: [524288, 4096, 1] : memref<32x128x4096xf32> to memref<1x128x128xf32, strided<[524288, 4096, 1], offset: ?>>
              loom.copy %reinterpret_cast_5, %18 src_mem_space @DRAM dst_mem_space @L1, interconnect : [], broadcast : [1, 1] : memref<1x128x128xf32, strided<[524288, 4096, 1], offset: ?>>, memref<1x128x128xf32>
              linalg.fill ins(%cst_1 : f32) outs(%17 : memref<1x64x128xf32>)
              linalg.batch_matmul ins(%24, %18 : memref<1x64x128xf32>, memref<1x128x128xf32>) outs(%17 : memref<1x64x128xf32>)
              linalg.fill ins(%cst_3 : f32) outs(%21 : memref<1x64xf32>)
              linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%17 : memref<1x64x128xf32>) outs(%21 : memref<1x64xf32>) {
              ^bb0(%in: f32, %out: f32):
                %34 = arith.maximumf %in, %out : f32
                linalg.yield %34 : f32
              }
              linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%20, %21 : memref<1x64xf32>, memref<1x64xf32>) outs(%21 : memref<1x64xf32>) {
              ^bb0(%in: f32, %in_7: f32, %out: f32):
                %34 = arith.truncf %cst_0 : f64 to f32
                %35 = arith.mulf %in_7, %34 : f32
                %36 = arith.cmpf ogt, %in, %35 : f32
                %37 = arith.select %36, %in, %35 : f32
                linalg.yield %37 : f32
              }
              linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%17, %21 : memref<1x64x128xf32>, memref<1x64xf32>) outs(%17 : memref<1x64x128xf32>) {
              ^bb0(%in: f32, %in_7: f32, %out: f32):
                %34 = arith.truncf %cst_0 : f64 to f32
                %35 = arith.mulf %in, %34 : f32
                %36 = arith.subf %35, %in_7 : f32
                %37 = math.powf %cst, %36 : f32
                linalg.yield %37 : f32
              }
              linalg.fill ins(%cst_1 : f32) outs(%22 : memref<1x64xf32>)
              linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%17 : memref<1x64x128xf32>) outs(%22 : memref<1x64xf32>) {
              ^bb0(%in: f32, %out: f32):
                %34 = arith.addf %in, %out : f32
                linalg.yield %34 : f32
              }
              linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%20, %21 : memref<1x64xf32>, memref<1x64xf32>) outs(%23 : memref<1x64xf32>) {
              ^bb0(%in: f32, %in_7: f32, %out: f32):
                %34 = arith.subf %in, %in_7 : f32
                %35 = math.powf %cst, %34 : f32
                linalg.yield %35 : f32
              }
              linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%19, %23, %22 : memref<1x64xf32>, memref<1x64xf32>, memref<1x64xf32>) outs(%19 : memref<1x64xf32>) {
              ^bb0(%in: f32, %in_7: f32, %in_8: f32, %out: f32):
                %34 = arith.mulf %in, %in_7 : f32
                %35 = arith.addf %34, %in_8 : f32
                linalg.yield %35 : f32
              }
              %32 = arith.muli %arg8, %c16384 : index
              %33 = arith.addi %27, %32 : index
              %reinterpret_cast_6 = memref.reinterpret_cast %arg1 to offset: [%33], sizes: [1, 128, 128], strides: [524288, 128, 1] : memref<32x4096x128xf32> to memref<1x128x128xf32, strided<[524288, 128, 1], offset: ?>>
              loom.copy %reinterpret_cast_6, %16 src_mem_space @DRAM dst_mem_space @L1, interconnect : [], broadcast : [1, 1] : memref<1x128x128xf32, strided<[524288, 128, 1], offset: ?>>, memref<1x128x128xf32>
              linalg.fill ins(%cst_1 : f32) outs(%26 : memref<1x64x128xf32>)
              linalg.batch_matmul ins(%17, %16 : memref<1x64x128xf32>, memref<1x128x128xf32>) outs(%26 : memref<1x64x128xf32>)
              linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%26, %25, %23 : memref<1x64x128xf32>, memref<1x64x128xf32>, memref<1x64xf32>) outs(%25 : memref<1x64x128xf32>) {
              ^bb0(%in: f32, %in_7: f32, %in_8: f32, %out: f32):
                %34 = arith.mulf %in_7, %in_8 : f32
                %35 = arith.addf %in, %34 : f32
                linalg.yield %35 : f32
              }
              linalg.copy ins(%21 : memref<1x64xf32>) outs(%20 : memref<1x64xf32>)
            }
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%25, %19 : memref<1x64x128xf32>, memref<1x64xf32>) outs(%24 : memref<1x64x128xf32>) {
            ^bb0(%in: f32, %in_5: f32, %out: f32):
              %30 = arith.divf %in, %in_5 : f32
              linalg.yield %30 : f32
            }
            %reinterpret_cast_4 = memref.reinterpret_cast %arg3 to offset: [%29], sizes: [1, 64, 128], strides: [524288, 128, 1] : memref<32x4096x128xf32> to memref<1x64x128xf32, strided<[524288, 128, 1], offset: ?>>
            loom.copy %24, %reinterpret_cast_4 src_mem_space @L1 dst_mem_space @DRAM, interconnect : [], broadcast : [1, 1] : memref<1x64x128xf32>, memref<1x64x128xf32, strided<[524288, 128, 1], offset: ?>>
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
            %17 = loom.alloc [1, 64, 128] on @L1 : memref<1x64x128xf32>
            %18 = loom.alloc [1, 128, 128] on @L1 : memref<1x128x128xf32>
            %19 = loom.alloc [1, 64] on @L1 : memref<1x64xf32>
            %20 = loom.alloc [1, 64] on @L1 : memref<1x64xf32>
            %21 = loom.alloc [1, 64] on @L1 : memref<1x64xf32>
            %22 = loom.alloc [1, 64] on @L1 : memref<1x64xf32>
            %23 = loom.alloc [1, 64] on @L1 : memref<1x64xf32>
            %24 = loom.alloc [1, 64, 128] on @L1 : memref<1x64x128xf32>
            %25 = loom.alloc [1, 64, 128] on @L1 : memref<1x64x128xf32>
            %26 = loom.alloc [1, 64, 128] on @L1 : memref<1x64x128xf32>
            %27 = arith.muli %13, %c524288 overflow<nsw> : index
            %28 = arith.muli %15, %c8192 : index
            %29 = arith.addi %27, %28 : index
            %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%29], sizes: [1, 64, 128], strides: [524288, 128, 1] : memref<32x4096x128xf32> to memref<1x64x128xf32, strided<[524288, 128, 1], offset: ?>>
            loom.copy %reinterpret_cast, %24 src_mem_space @DRAM dst_mem_space @L1, interconnect : [], broadcast : [1, 1] : memref<1x64x128xf32, strided<[524288, 128, 1], offset: ?>>, memref<1x64x128xf32>
            linalg.fill ins(%cst_1 : f32) outs(%25 : memref<1x64x128xf32>)
            linalg.fill ins(%cst_2 : f32) outs(%19 : memref<1x64xf32>)
            linalg.fill ins(%cst_3 : f32) outs(%20 : memref<1x64xf32>)
            scf.for %arg8 = %c0 to %c32 step %c1 {
              %30 = arith.muli %arg8, %c128 : index
              %31 = arith.addi %27, %30 : index
              %reinterpret_cast_5 = memref.reinterpret_cast %arg0 to offset: [%31], sizes: [1, 128, 128], strides: [524288, 4096, 1] : memref<32x128x4096xf32> to memref<1x128x128xf32, strided<[524288, 4096, 1], offset: ?>>
              loom.copy %reinterpret_cast_5, %18 src_mem_space @DRAM dst_mem_space @L1, interconnect : [], broadcast : [1, 1] : memref<1x128x128xf32, strided<[524288, 4096, 1], offset: ?>>, memref<1x128x128xf32>
              linalg.fill ins(%cst_1 : f32) outs(%17 : memref<1x64x128xf32>)
              linalg.batch_matmul ins(%24, %18 : memref<1x64x128xf32>, memref<1x128x128xf32>) outs(%17 : memref<1x64x128xf32>)
              linalg.fill ins(%cst_3 : f32) outs(%21 : memref<1x64xf32>)
              linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%17 : memref<1x64x128xf32>) outs(%21 : memref<1x64xf32>) {
              ^bb0(%in: f32, %out: f32):
                %34 = arith.maximumf %in, %out : f32
                linalg.yield %34 : f32
              }
              linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%20, %21 : memref<1x64xf32>, memref<1x64xf32>) outs(%21 : memref<1x64xf32>) {
              ^bb0(%in: f32, %in_7: f32, %out: f32):
                %34 = arith.truncf %cst_0 : f64 to f32
                %35 = arith.mulf %in_7, %34 : f32
                %36 = arith.cmpf ogt, %in, %35 : f32
                %37 = arith.select %36, %in, %35 : f32
                linalg.yield %37 : f32
              }
              linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%17, %21 : memref<1x64x128xf32>, memref<1x64xf32>) outs(%17 : memref<1x64x128xf32>) {
              ^bb0(%in: f32, %in_7: f32, %out: f32):
                %34 = arith.truncf %cst_0 : f64 to f32
                %35 = arith.mulf %in, %34 : f32
                %36 = arith.subf %35, %in_7 : f32
                %37 = math.powf %cst, %36 : f32
                linalg.yield %37 : f32
              }
              linalg.fill ins(%cst_1 : f32) outs(%22 : memref<1x64xf32>)
              linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%17 : memref<1x64x128xf32>) outs(%22 : memref<1x64xf32>) {
              ^bb0(%in: f32, %out: f32):
                %34 = arith.addf %in, %out : f32
                linalg.yield %34 : f32
              }
              linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%20, %21 : memref<1x64xf32>, memref<1x64xf32>) outs(%23 : memref<1x64xf32>) {
              ^bb0(%in: f32, %in_7: f32, %out: f32):
                %34 = arith.subf %in, %in_7 : f32
                %35 = math.powf %cst, %34 : f32
                linalg.yield %35 : f32
              }
              linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%19, %23, %22 : memref<1x64xf32>, memref<1x64xf32>, memref<1x64xf32>) outs(%19 : memref<1x64xf32>) {
              ^bb0(%in: f32, %in_7: f32, %in_8: f32, %out: f32):
                %34 = arith.mulf %in, %in_7 : f32
                %35 = arith.addf %34, %in_8 : f32
                linalg.yield %35 : f32
              }
              %32 = arith.muli %arg8, %c16384 : index
              %33 = arith.addi %27, %32 : index
              %reinterpret_cast_6 = memref.reinterpret_cast %arg1 to offset: [%33], sizes: [1, 128, 128], strides: [524288, 128, 1] : memref<32x4096x128xf32> to memref<1x128x128xf32, strided<[524288, 128, 1], offset: ?>>
              loom.copy %reinterpret_cast_6, %16 src_mem_space @DRAM dst_mem_space @L1, interconnect : [@vertical_links], broadcast : [8, 1] : memref<1x128x128xf32, strided<[524288, 128, 1], offset: ?>>, memref<1x128x128xf32>
              linalg.fill ins(%cst_1 : f32) outs(%26 : memref<1x64x128xf32>)
              linalg.batch_matmul ins(%17, %16 : memref<1x64x128xf32>, memref<1x128x128xf32>) outs(%26 : memref<1x64x128xf32>)
              linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%26, %25, %23 : memref<1x64x128xf32>, memref<1x64x128xf32>, memref<1x64xf32>) outs(%25 : memref<1x64x128xf32>) {
              ^bb0(%in: f32, %in_7: f32, %in_8: f32, %out: f32):
                %34 = arith.mulf %in_7, %in_8 : f32
                %35 = arith.addf %in, %34 : f32
                linalg.yield %35 : f32
              }
              linalg.copy ins(%21 : memref<1x64xf32>) outs(%20 : memref<1x64xf32>)
            }
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%25, %19 : memref<1x64x128xf32>, memref<1x64xf32>) outs(%24 : memref<1x64x128xf32>) {
            ^bb0(%in: f32, %in_5: f32, %out: f32):
              %30 = arith.divf %in, %in_5 : f32
              linalg.yield %30 : f32
            }
            %reinterpret_cast_4 = memref.reinterpret_cast %arg3 to offset: [%29], sizes: [1, 64, 128], strides: [524288, 128, 1] : memref<32x4096x128xf32> to memref<1x64x128xf32, strided<[524288, 128, 1], offset: ?>>
            loom.copy %24, %reinterpret_cast_4 src_mem_space @L1 dst_mem_space @DRAM, interconnect : [], broadcast : [1, 1] : memref<1x64x128xf32>, memref<1x64x128xf32, strided<[524288, 128, 1], offset: ?>>
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
            %17 = loom.alloc [1, 64, 128] on @L1 : memref<1x64x128xf32>
            %18 = loom.alloc [1, 128, 128] on @L1 : memref<1x128x128xf32>
            %19 = loom.alloc [1, 64] on @L1 : memref<1x64xf32>
            %20 = loom.alloc [1, 64] on @L1 : memref<1x64xf32>
            %21 = loom.alloc [1, 64] on @L1 : memref<1x64xf32>
            %22 = loom.alloc [1, 64] on @L1 : memref<1x64xf32>
            %23 = loom.alloc [1, 64] on @L1 : memref<1x64xf32>
            %24 = loom.alloc [1, 64, 128] on @L1 : memref<1x64x128xf32>
            %25 = loom.alloc [1, 64, 128] on @L1 : memref<1x64x128xf32>
            %26 = loom.alloc [1, 64, 128] on @L1 : memref<1x64x128xf32>
            %27 = arith.muli %13, %c524288 overflow<nsw> : index
            %28 = arith.muli %15, %c8192 : index
            %29 = arith.addi %27, %28 : index
            %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%29], sizes: [1, 64, 128], strides: [524288, 128, 1] : memref<32x4096x128xf32> to memref<1x64x128xf32, strided<[524288, 128, 1], offset: ?>>
            loom.copy %reinterpret_cast, %24 src_mem_space @DRAM dst_mem_space @L1, interconnect : [], broadcast : [1, 1] : memref<1x64x128xf32, strided<[524288, 128, 1], offset: ?>>, memref<1x64x128xf32>
            linalg.fill ins(%cst_1 : f32) outs(%25 : memref<1x64x128xf32>)
            linalg.fill ins(%cst_2 : f32) outs(%19 : memref<1x64xf32>)
            linalg.fill ins(%cst_3 : f32) outs(%20 : memref<1x64xf32>)
            scf.for %arg8 = %c0 to %c32 step %c1 {
              %30 = arith.muli %arg8, %c128 : index
              %31 = arith.addi %27, %30 : index
              %reinterpret_cast_5 = memref.reinterpret_cast %arg0 to offset: [%31], sizes: [1, 128, 128], strides: [524288, 4096, 1] : memref<32x128x4096xf32> to memref<1x128x128xf32, strided<[524288, 4096, 1], offset: ?>>
              loom.copy %reinterpret_cast_5, %18 src_mem_space @DRAM dst_mem_space @L1, interconnect : [@vertical_links], broadcast : [8, 1] : memref<1x128x128xf32, strided<[524288, 4096, 1], offset: ?>>, memref<1x128x128xf32>
              linalg.fill ins(%cst_1 : f32) outs(%17 : memref<1x64x128xf32>)
              linalg.batch_matmul ins(%24, %18 : memref<1x64x128xf32>, memref<1x128x128xf32>) outs(%17 : memref<1x64x128xf32>)
              linalg.fill ins(%cst_3 : f32) outs(%21 : memref<1x64xf32>)
              linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%17 : memref<1x64x128xf32>) outs(%21 : memref<1x64xf32>) {
              ^bb0(%in: f32, %out: f32):
                %34 = arith.maximumf %in, %out : f32
                linalg.yield %34 : f32
              }
              linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%20, %21 : memref<1x64xf32>, memref<1x64xf32>) outs(%21 : memref<1x64xf32>) {
              ^bb0(%in: f32, %in_7: f32, %out: f32):
                %34 = arith.truncf %cst_0 : f64 to f32
                %35 = arith.mulf %in_7, %34 : f32
                %36 = arith.cmpf ogt, %in, %35 : f32
                %37 = arith.select %36, %in, %35 : f32
                linalg.yield %37 : f32
              }
              linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%17, %21 : memref<1x64x128xf32>, memref<1x64xf32>) outs(%17 : memref<1x64x128xf32>) {
              ^bb0(%in: f32, %in_7: f32, %out: f32):
                %34 = arith.truncf %cst_0 : f64 to f32
                %35 = arith.mulf %in, %34 : f32
                %36 = arith.subf %35, %in_7 : f32
                %37 = math.powf %cst, %36 : f32
                linalg.yield %37 : f32
              }
              linalg.fill ins(%cst_1 : f32) outs(%22 : memref<1x64xf32>)
              linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%17 : memref<1x64x128xf32>) outs(%22 : memref<1x64xf32>) {
              ^bb0(%in: f32, %out: f32):
                %34 = arith.addf %in, %out : f32
                linalg.yield %34 : f32
              }
              linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%20, %21 : memref<1x64xf32>, memref<1x64xf32>) outs(%23 : memref<1x64xf32>) {
              ^bb0(%in: f32, %in_7: f32, %out: f32):
                %34 = arith.subf %in, %in_7 : f32
                %35 = math.powf %cst, %34 : f32
                linalg.yield %35 : f32
              }
              linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%19, %23, %22 : memref<1x64xf32>, memref<1x64xf32>, memref<1x64xf32>) outs(%19 : memref<1x64xf32>) {
              ^bb0(%in: f32, %in_7: f32, %in_8: f32, %out: f32):
                %34 = arith.mulf %in, %in_7 : f32
                %35 = arith.addf %34, %in_8 : f32
                linalg.yield %35 : f32
              }
              %32 = arith.muli %arg8, %c16384 : index
              %33 = arith.addi %27, %32 : index
              %reinterpret_cast_6 = memref.reinterpret_cast %arg1 to offset: [%33], sizes: [1, 128, 128], strides: [524288, 128, 1] : memref<32x4096x128xf32> to memref<1x128x128xf32, strided<[524288, 128, 1], offset: ?>>
              loom.copy %reinterpret_cast_6, %16 src_mem_space @DRAM dst_mem_space @L1, interconnect : [], broadcast : [1, 1] : memref<1x128x128xf32, strided<[524288, 128, 1], offset: ?>>, memref<1x128x128xf32>
              linalg.fill ins(%cst_1 : f32) outs(%26 : memref<1x64x128xf32>)
              linalg.batch_matmul ins(%17, %16 : memref<1x64x128xf32>, memref<1x128x128xf32>) outs(%26 : memref<1x64x128xf32>)
              linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%26, %25, %23 : memref<1x64x128xf32>, memref<1x64x128xf32>, memref<1x64xf32>) outs(%25 : memref<1x64x128xf32>) {
              ^bb0(%in: f32, %in_7: f32, %in_8: f32, %out: f32):
                %34 = arith.mulf %in_7, %in_8 : f32
                %35 = arith.addf %in, %34 : f32
                linalg.yield %35 : f32
              }
              linalg.copy ins(%21 : memref<1x64xf32>) outs(%20 : memref<1x64xf32>)
            }
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%25, %19 : memref<1x64x128xf32>, memref<1x64xf32>) outs(%24 : memref<1x64x128xf32>) {
            ^bb0(%in: f32, %in_5: f32, %out: f32):
              %30 = arith.divf %in, %in_5 : f32
              linalg.yield %30 : f32
            }
            %reinterpret_cast_4 = memref.reinterpret_cast %arg3 to offset: [%29], sizes: [1, 64, 128], strides: [524288, 128, 1] : memref<32x4096x128xf32> to memref<1x64x128xf32, strided<[524288, 128, 1], offset: ?>>
            loom.copy %24, %reinterpret_cast_4 src_mem_space @L1 dst_mem_space @DRAM, interconnect : [], broadcast : [1, 1] : memref<1x64x128xf32>, memref<1x64x128xf32, strided<[524288, 128, 1], offset: ?>>
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
            %17 = loom.alloc [1, 64, 128] on @L1 : memref<1x64x128xf32>
            %18 = loom.alloc [1, 128, 128] on @L1 : memref<1x128x128xf32>
            %19 = loom.alloc [1, 64] on @L1 : memref<1x64xf32>
            %20 = loom.alloc [1, 64] on @L1 : memref<1x64xf32>
            %21 = loom.alloc [1, 64] on @L1 : memref<1x64xf32>
            %22 = loom.alloc [1, 64] on @L1 : memref<1x64xf32>
            %23 = loom.alloc [1, 64] on @L1 : memref<1x64xf32>
            %24 = loom.alloc [1, 64, 128] on @L1 : memref<1x64x128xf32>
            %25 = loom.alloc [1, 64, 128] on @L1 : memref<1x64x128xf32>
            %26 = loom.alloc [1, 64, 128] on @L1 : memref<1x64x128xf32>
            %27 = arith.muli %13, %c524288 overflow<nsw> : index
            %28 = arith.muli %15, %c8192 : index
            %29 = arith.addi %27, %28 : index
            %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%29], sizes: [1, 64, 128], strides: [524288, 128, 1] : memref<32x4096x128xf32> to memref<1x64x128xf32, strided<[524288, 128, 1], offset: ?>>
            loom.copy %reinterpret_cast, %24 src_mem_space @DRAM dst_mem_space @L1, interconnect : [], broadcast : [1, 1] : memref<1x64x128xf32, strided<[524288, 128, 1], offset: ?>>, memref<1x64x128xf32>
            linalg.fill ins(%cst_1 : f32) outs(%25 : memref<1x64x128xf32>)
            linalg.fill ins(%cst_2 : f32) outs(%19 : memref<1x64xf32>)
            linalg.fill ins(%cst_3 : f32) outs(%20 : memref<1x64xf32>)
            scf.for %arg8 = %c0 to %c32 step %c1 {
              %30 = arith.muli %arg8, %c128 : index
              %31 = arith.addi %27, %30 : index
              %reinterpret_cast_5 = memref.reinterpret_cast %arg0 to offset: [%31], sizes: [1, 128, 128], strides: [524288, 4096, 1] : memref<32x128x4096xf32> to memref<1x128x128xf32, strided<[524288, 4096, 1], offset: ?>>
              loom.copy %reinterpret_cast_5, %18 src_mem_space @DRAM dst_mem_space @L1, interconnect : [@vertical_links], broadcast : [8, 1] : memref<1x128x128xf32, strided<[524288, 4096, 1], offset: ?>>, memref<1x128x128xf32>
              linalg.fill ins(%cst_1 : f32) outs(%17 : memref<1x64x128xf32>)
              linalg.batch_matmul ins(%24, %18 : memref<1x64x128xf32>, memref<1x128x128xf32>) outs(%17 : memref<1x64x128xf32>)
              linalg.fill ins(%cst_3 : f32) outs(%21 : memref<1x64xf32>)
              linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%17 : memref<1x64x128xf32>) outs(%21 : memref<1x64xf32>) {
              ^bb0(%in: f32, %out: f32):
                %34 = arith.maximumf %in, %out : f32
                linalg.yield %34 : f32
              }
              linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%20, %21 : memref<1x64xf32>, memref<1x64xf32>) outs(%21 : memref<1x64xf32>) {
              ^bb0(%in: f32, %in_7: f32, %out: f32):
                %34 = arith.truncf %cst_0 : f64 to f32
                %35 = arith.mulf %in_7, %34 : f32
                %36 = arith.cmpf ogt, %in, %35 : f32
                %37 = arith.select %36, %in, %35 : f32
                linalg.yield %37 : f32
              }
              linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%17, %21 : memref<1x64x128xf32>, memref<1x64xf32>) outs(%17 : memref<1x64x128xf32>) {
              ^bb0(%in: f32, %in_7: f32, %out: f32):
                %34 = arith.truncf %cst_0 : f64 to f32
                %35 = arith.mulf %in, %34 : f32
                %36 = arith.subf %35, %in_7 : f32
                %37 = math.powf %cst, %36 : f32
                linalg.yield %37 : f32
              }
              linalg.fill ins(%cst_1 : f32) outs(%22 : memref<1x64xf32>)
              linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%17 : memref<1x64x128xf32>) outs(%22 : memref<1x64xf32>) {
              ^bb0(%in: f32, %out: f32):
                %34 = arith.addf %in, %out : f32
                linalg.yield %34 : f32
              }
              linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%20, %21 : memref<1x64xf32>, memref<1x64xf32>) outs(%23 : memref<1x64xf32>) {
              ^bb0(%in: f32, %in_7: f32, %out: f32):
                %34 = arith.subf %in, %in_7 : f32
                %35 = math.powf %cst, %34 : f32
                linalg.yield %35 : f32
              }
              linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%19, %23, %22 : memref<1x64xf32>, memref<1x64xf32>, memref<1x64xf32>) outs(%19 : memref<1x64xf32>) {
              ^bb0(%in: f32, %in_7: f32, %in_8: f32, %out: f32):
                %34 = arith.mulf %in, %in_7 : f32
                %35 = arith.addf %34, %in_8 : f32
                linalg.yield %35 : f32
              }
              %32 = arith.muli %arg8, %c16384 : index
              %33 = arith.addi %27, %32 : index
              %reinterpret_cast_6 = memref.reinterpret_cast %arg1 to offset: [%33], sizes: [1, 128, 128], strides: [524288, 128, 1] : memref<32x4096x128xf32> to memref<1x128x128xf32, strided<[524288, 128, 1], offset: ?>>
              loom.copy %reinterpret_cast_6, %16 src_mem_space @DRAM dst_mem_space @L1, interconnect : [@vertical_links], broadcast : [8, 1] : memref<1x128x128xf32, strided<[524288, 128, 1], offset: ?>>, memref<1x128x128xf32>
              linalg.fill ins(%cst_1 : f32) outs(%26 : memref<1x64x128xf32>)
              linalg.batch_matmul ins(%17, %16 : memref<1x64x128xf32>, memref<1x128x128xf32>) outs(%26 : memref<1x64x128xf32>)
              linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%26, %25, %23 : memref<1x64x128xf32>, memref<1x64x128xf32>, memref<1x64xf32>) outs(%25 : memref<1x64x128xf32>) {
              ^bb0(%in: f32, %in_7: f32, %in_8: f32, %out: f32):
                %34 = arith.mulf %in_7, %in_8 : f32
                %35 = arith.addf %in, %34 : f32
                linalg.yield %35 : f32
              }
              linalg.copy ins(%21 : memref<1x64xf32>) outs(%20 : memref<1x64xf32>)
            }
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%25, %19 : memref<1x64x128xf32>, memref<1x64xf32>) outs(%24 : memref<1x64x128xf32>) {
            ^bb0(%in: f32, %in_5: f32, %out: f32):
              %30 = arith.divf %in, %in_5 : f32
              linalg.yield %30 : f32
            }
            %reinterpret_cast_4 = memref.reinterpret_cast %arg3 to offset: [%29], sizes: [1, 64, 128], strides: [524288, 128, 1] : memref<32x4096x128xf32> to memref<1x64x128xf32, strided<[524288, 128, 1], offset: ?>>
            loom.copy %24, %reinterpret_cast_4 src_mem_space @L1 dst_mem_space @DRAM, interconnect : [], broadcast : [1, 1] : memref<1x64x128xf32>, memref<1x64x128xf32, strided<[524288, 128, 1], offset: ?>>
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
            %17 = loom.alloc [1, 64, 128] on @L1 : memref<1x64x128xf32>
            %18 = loom.alloc [1, 128, 128] on @L1 : memref<1x128x128xf32>
            %19 = loom.alloc [1, 64] on @L1 : memref<1x64xf32>
            %20 = loom.alloc [1, 64] on @L1 : memref<1x64xf32>
            %21 = loom.alloc [1, 64] on @L1 : memref<1x64xf32>
            %22 = loom.alloc [1, 64] on @L1 : memref<1x64xf32>
            %23 = loom.alloc [1, 64] on @L1 : memref<1x64xf32>
            %24 = loom.alloc [1, 64, 128] on @L1 : memref<1x64x128xf32>
            %25 = loom.alloc [1, 64, 128] on @L1 : memref<1x64x128xf32>
            %26 = loom.alloc [1, 64, 128] on @L1 : memref<1x64x128xf32>
            %27 = arith.muli %13, %c524288 overflow<nsw> : index
            %28 = arith.muli %15, %c8192 : index
            %29 = arith.addi %27, %28 : index
            %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%29], sizes: [1, 64, 128], strides: [524288, 128, 1] : memref<32x4096x128xf32> to memref<1x64x128xf32, strided<[524288, 128, 1], offset: ?>>
            loom.copy %reinterpret_cast, %24 src_mem_space @DRAM dst_mem_space @L1, interconnect : [], broadcast : [1, 1] : memref<1x64x128xf32, strided<[524288, 128, 1], offset: ?>>, memref<1x64x128xf32>
            linalg.fill ins(%cst_1 : f32) outs(%25 : memref<1x64x128xf32>)
            linalg.fill ins(%cst_2 : f32) outs(%19 : memref<1x64xf32>)
            linalg.fill ins(%cst_3 : f32) outs(%20 : memref<1x64xf32>)
            scf.for %arg8 = %c0 to %c32 step %c1 {
              %30 = arith.muli %arg8, %c128 : index
              %31 = arith.addi %27, %30 : index
              %reinterpret_cast_5 = memref.reinterpret_cast %arg0 to offset: [%31], sizes: [1, 128, 128], strides: [524288, 4096, 1] : memref<32x128x4096xf32> to memref<1x128x128xf32, strided<[524288, 4096, 1], offset: ?>>
              loom.copy %reinterpret_cast_5, %18 src_mem_space @DRAM dst_mem_space @L1, interconnect : [], broadcast : [1, 1] : memref<1x128x128xf32, strided<[524288, 4096, 1], offset: ?>>, memref<1x128x128xf32>
              linalg.fill ins(%cst_1 : f32) outs(%17 : memref<1x64x128xf32>)
              linalg.batch_matmul ins(%24, %18 : memref<1x64x128xf32>, memref<1x128x128xf32>) outs(%17 : memref<1x64x128xf32>)
              linalg.fill ins(%cst_3 : f32) outs(%21 : memref<1x64xf32>)
              linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%17 : memref<1x64x128xf32>) outs(%21 : memref<1x64xf32>) {
              ^bb0(%in: f32, %out: f32):
                %34 = arith.maximumf %in, %out : f32
                linalg.yield %34 : f32
              }
              linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%20, %21 : memref<1x64xf32>, memref<1x64xf32>) outs(%21 : memref<1x64xf32>) {
              ^bb0(%in: f32, %in_7: f32, %out: f32):
                %34 = arith.truncf %cst_0 : f64 to f32
                %35 = arith.mulf %in_7, %34 : f32
                %36 = arith.cmpf ogt, %in, %35 : f32
                %37 = arith.select %36, %in, %35 : f32
                linalg.yield %37 : f32
              }
              linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%17, %21 : memref<1x64x128xf32>, memref<1x64xf32>) outs(%17 : memref<1x64x128xf32>) {
              ^bb0(%in: f32, %in_7: f32, %out: f32):
                %34 = arith.truncf %cst_0 : f64 to f32
                %35 = arith.mulf %in, %34 : f32
                %36 = arith.subf %35, %in_7 : f32
                %37 = math.powf %cst, %36 : f32
                linalg.yield %37 : f32
              }
              linalg.fill ins(%cst_1 : f32) outs(%22 : memref<1x64xf32>)
              linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%17 : memref<1x64x128xf32>) outs(%22 : memref<1x64xf32>) {
              ^bb0(%in: f32, %out: f32):
                %34 = arith.addf %in, %out : f32
                linalg.yield %34 : f32
              }
              linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%20, %21 : memref<1x64xf32>, memref<1x64xf32>) outs(%23 : memref<1x64xf32>) {
              ^bb0(%in: f32, %in_7: f32, %out: f32):
                %34 = arith.subf %in, %in_7 : f32
                %35 = math.powf %cst, %34 : f32
                linalg.yield %35 : f32
              }
              linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%19, %23, %22 : memref<1x64xf32>, memref<1x64xf32>, memref<1x64xf32>) outs(%19 : memref<1x64xf32>) {
              ^bb0(%in: f32, %in_7: f32, %in_8: f32, %out: f32):
                %34 = arith.mulf %in, %in_7 : f32
                %35 = arith.addf %34, %in_8 : f32
                linalg.yield %35 : f32
              }
              %32 = arith.muli %arg8, %c16384 : index
              %33 = arith.addi %27, %32 : index
              %reinterpret_cast_6 = memref.reinterpret_cast %arg1 to offset: [%33], sizes: [1, 128, 128], strides: [524288, 128, 1] : memref<32x4096x128xf32> to memref<1x128x128xf32, strided<[524288, 128, 1], offset: ?>>
              loom.copy %reinterpret_cast_6, %16 src_mem_space @DRAM dst_mem_space @L1, interconnect : [], broadcast : [1, 1] : memref<1x128x128xf32, strided<[524288, 128, 1], offset: ?>>, memref<1x128x128xf32>
              linalg.fill ins(%cst_1 : f32) outs(%26 : memref<1x64x128xf32>)
              linalg.batch_matmul ins(%17, %16 : memref<1x64x128xf32>, memref<1x128x128xf32>) outs(%26 : memref<1x64x128xf32>)
              linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%26, %25, %23 : memref<1x64x128xf32>, memref<1x64x128xf32>, memref<1x64xf32>) outs(%25 : memref<1x64x128xf32>) {
              ^bb0(%in: f32, %in_7: f32, %in_8: f32, %out: f32):
                %34 = arith.mulf %in_7, %in_8 : f32
                %35 = arith.addf %in, %34 : f32
                linalg.yield %35 : f32
              }
              linalg.copy ins(%21 : memref<1x64xf32>) outs(%20 : memref<1x64xf32>)
            }
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%25, %19 : memref<1x64x128xf32>, memref<1x64xf32>) outs(%24 : memref<1x64x128xf32>) {
            ^bb0(%in: f32, %in_5: f32, %out: f32):
              %30 = arith.divf %in, %in_5 : f32
              linalg.yield %30 : f32
            }
            %reinterpret_cast_4 = memref.reinterpret_cast %arg3 to offset: [%29], sizes: [1, 64, 128], strides: [524288, 128, 1] : memref<32x4096x128xf32> to memref<1x64x128xf32, strided<[524288, 128, 1], offset: ?>>
            loom.copy %24, %reinterpret_cast_4 src_mem_space @L1 dst_mem_space @DRAM, interconnect : [], broadcast : [1, 1] : memref<1x64x128xf32>, memref<1x64x128xf32, strided<[524288, 128, 1], offset: ?>>
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
            %17 = loom.alloc [1, 64, 128] on @L1 : memref<1x64x128xf32>
            %18 = loom.alloc [1, 128, 128] on @L1 : memref<1x128x128xf32>
            %19 = loom.alloc [1, 64] on @L1 : memref<1x64xf32>
            %20 = loom.alloc [1, 64] on @L1 : memref<1x64xf32>
            %21 = loom.alloc [1, 64] on @L1 : memref<1x64xf32>
            %22 = loom.alloc [1, 64] on @L1 : memref<1x64xf32>
            %23 = loom.alloc [1, 64] on @L1 : memref<1x64xf32>
            %24 = loom.alloc [1, 64, 128] on @L1 : memref<1x64x128xf32>
            %25 = loom.alloc [1, 64, 128] on @L1 : memref<1x64x128xf32>
            %26 = loom.alloc [1, 64, 128] on @L1 : memref<1x64x128xf32>
            %27 = arith.muli %13, %c524288 overflow<nsw> : index
            %28 = arith.muli %15, %c8192 : index
            %29 = arith.addi %27, %28 : index
            %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%29], sizes: [1, 64, 128], strides: [524288, 128, 1] : memref<32x4096x128xf32> to memref<1x64x128xf32, strided<[524288, 128, 1], offset: ?>>
            loom.copy %reinterpret_cast, %24 src_mem_space @DRAM dst_mem_space @L1, interconnect : [], broadcast : [1, 1] : memref<1x64x128xf32, strided<[524288, 128, 1], offset: ?>>, memref<1x64x128xf32>
            linalg.fill ins(%cst_1 : f32) outs(%25 : memref<1x64x128xf32>)
            linalg.fill ins(%cst_2 : f32) outs(%19 : memref<1x64xf32>)
            linalg.fill ins(%cst_3 : f32) outs(%20 : memref<1x64xf32>)
            scf.for %arg8 = %c0 to %c32 step %c1 {
              %30 = arith.muli %arg8, %c128 : index
              %31 = arith.addi %27, %30 : index
              %reinterpret_cast_5 = memref.reinterpret_cast %arg0 to offset: [%31], sizes: [1, 128, 128], strides: [524288, 4096, 1] : memref<32x128x4096xf32> to memref<1x128x128xf32, strided<[524288, 4096, 1], offset: ?>>
              loom.copy %reinterpret_cast_5, %18 src_mem_space @DRAM dst_mem_space @L1, interconnect : [], broadcast : [1, 1] : memref<1x128x128xf32, strided<[524288, 4096, 1], offset: ?>>, memref<1x128x128xf32>
              linalg.fill ins(%cst_1 : f32) outs(%17 : memref<1x64x128xf32>)
              linalg.batch_matmul ins(%24, %18 : memref<1x64x128xf32>, memref<1x128x128xf32>) outs(%17 : memref<1x64x128xf32>)
              linalg.fill ins(%cst_3 : f32) outs(%21 : memref<1x64xf32>)
              linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%17 : memref<1x64x128xf32>) outs(%21 : memref<1x64xf32>) {
              ^bb0(%in: f32, %out: f32):
                %34 = arith.maximumf %in, %out : f32
                linalg.yield %34 : f32
              }
              linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%20, %21 : memref<1x64xf32>, memref<1x64xf32>) outs(%21 : memref<1x64xf32>) {
              ^bb0(%in: f32, %in_7: f32, %out: f32):
                %34 = arith.truncf %cst_0 : f64 to f32
                %35 = arith.mulf %in_7, %34 : f32
                %36 = arith.cmpf ogt, %in, %35 : f32
                %37 = arith.select %36, %in, %35 : f32
                linalg.yield %37 : f32
              }
              linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%17, %21 : memref<1x64x128xf32>, memref<1x64xf32>) outs(%17 : memref<1x64x128xf32>) {
              ^bb0(%in: f32, %in_7: f32, %out: f32):
                %34 = arith.truncf %cst_0 : f64 to f32
                %35 = arith.mulf %in, %34 : f32
                %36 = arith.subf %35, %in_7 : f32
                %37 = math.powf %cst, %36 : f32
                linalg.yield %37 : f32
              }
              linalg.fill ins(%cst_1 : f32) outs(%22 : memref<1x64xf32>)
              linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%17 : memref<1x64x128xf32>) outs(%22 : memref<1x64xf32>) {
              ^bb0(%in: f32, %out: f32):
                %34 = arith.addf %in, %out : f32
                linalg.yield %34 : f32
              }
              linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%20, %21 : memref<1x64xf32>, memref<1x64xf32>) outs(%23 : memref<1x64xf32>) {
              ^bb0(%in: f32, %in_7: f32, %out: f32):
                %34 = arith.subf %in, %in_7 : f32
                %35 = math.powf %cst, %34 : f32
                linalg.yield %35 : f32
              }
              linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%19, %23, %22 : memref<1x64xf32>, memref<1x64xf32>, memref<1x64xf32>) outs(%19 : memref<1x64xf32>) {
              ^bb0(%in: f32, %in_7: f32, %in_8: f32, %out: f32):
                %34 = arith.mulf %in, %in_7 : f32
                %35 = arith.addf %34, %in_8 : f32
                linalg.yield %35 : f32
              }
              %32 = arith.muli %arg8, %c16384 : index
              %33 = arith.addi %27, %32 : index
              %reinterpret_cast_6 = memref.reinterpret_cast %arg1 to offset: [%33], sizes: [1, 128, 128], strides: [524288, 128, 1] : memref<32x4096x128xf32> to memref<1x128x128xf32, strided<[524288, 128, 1], offset: ?>>
              loom.copy %reinterpret_cast_6, %16 src_mem_space @DRAM dst_mem_space @L1, interconnect : [@horizontal_links], broadcast : [1, 8] : memref<1x128x128xf32, strided<[524288, 128, 1], offset: ?>>, memref<1x128x128xf32>
              linalg.fill ins(%cst_1 : f32) outs(%26 : memref<1x64x128xf32>)
              linalg.batch_matmul ins(%17, %16 : memref<1x64x128xf32>, memref<1x128x128xf32>) outs(%26 : memref<1x64x128xf32>)
              linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%26, %25, %23 : memref<1x64x128xf32>, memref<1x64x128xf32>, memref<1x64xf32>) outs(%25 : memref<1x64x128xf32>) {
              ^bb0(%in: f32, %in_7: f32, %in_8: f32, %out: f32):
                %34 = arith.mulf %in_7, %in_8 : f32
                %35 = arith.addf %in, %34 : f32
                linalg.yield %35 : f32
              }
              linalg.copy ins(%21 : memref<1x64xf32>) outs(%20 : memref<1x64xf32>)
            }
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%25, %19 : memref<1x64x128xf32>, memref<1x64xf32>) outs(%24 : memref<1x64x128xf32>) {
            ^bb0(%in: f32, %in_5: f32, %out: f32):
              %30 = arith.divf %in, %in_5 : f32
              linalg.yield %30 : f32
            }
            %reinterpret_cast_4 = memref.reinterpret_cast %arg3 to offset: [%29], sizes: [1, 64, 128], strides: [524288, 128, 1] : memref<32x4096x128xf32> to memref<1x64x128xf32, strided<[524288, 128, 1], offset: ?>>
            loom.copy %24, %reinterpret_cast_4 src_mem_space @L1 dst_mem_space @DRAM, interconnect : [], broadcast : [1, 1] : memref<1x64x128xf32>, memref<1x64x128xf32, strided<[524288, 128, 1], offset: ?>>
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
            %17 = loom.alloc [1, 64, 128] on @L1 : memref<1x64x128xf32>
            %18 = loom.alloc [1, 128, 128] on @L1 : memref<1x128x128xf32>
            %19 = loom.alloc [1, 64] on @L1 : memref<1x64xf32>
            %20 = loom.alloc [1, 64] on @L1 : memref<1x64xf32>
            %21 = loom.alloc [1, 64] on @L1 : memref<1x64xf32>
            %22 = loom.alloc [1, 64] on @L1 : memref<1x64xf32>
            %23 = loom.alloc [1, 64] on @L1 : memref<1x64xf32>
            %24 = loom.alloc [1, 64, 128] on @L1 : memref<1x64x128xf32>
            %25 = loom.alloc [1, 64, 128] on @L1 : memref<1x64x128xf32>
            %26 = loom.alloc [1, 64, 128] on @L1 : memref<1x64x128xf32>
            %27 = arith.muli %13, %c524288 overflow<nsw> : index
            %28 = arith.muli %15, %c8192 : index
            %29 = arith.addi %27, %28 : index
            %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%29], sizes: [1, 64, 128], strides: [524288, 128, 1] : memref<32x4096x128xf32> to memref<1x64x128xf32, strided<[524288, 128, 1], offset: ?>>
            loom.copy %reinterpret_cast, %24 src_mem_space @DRAM dst_mem_space @L1, interconnect : [], broadcast : [1, 1] : memref<1x64x128xf32, strided<[524288, 128, 1], offset: ?>>, memref<1x64x128xf32>
            linalg.fill ins(%cst_1 : f32) outs(%25 : memref<1x64x128xf32>)
            linalg.fill ins(%cst_2 : f32) outs(%19 : memref<1x64xf32>)
            linalg.fill ins(%cst_3 : f32) outs(%20 : memref<1x64xf32>)
            scf.for %arg8 = %c0 to %c32 step %c1 {
              %30 = arith.muli %arg8, %c128 : index
              %31 = arith.addi %27, %30 : index
              %reinterpret_cast_5 = memref.reinterpret_cast %arg0 to offset: [%31], sizes: [1, 128, 128], strides: [524288, 4096, 1] : memref<32x128x4096xf32> to memref<1x128x128xf32, strided<[524288, 4096, 1], offset: ?>>
              loom.copy %reinterpret_cast_5, %18 src_mem_space @DRAM dst_mem_space @L1, interconnect : [@horizontal_links], broadcast : [1, 8] : memref<1x128x128xf32, strided<[524288, 4096, 1], offset: ?>>, memref<1x128x128xf32>
              linalg.fill ins(%cst_1 : f32) outs(%17 : memref<1x64x128xf32>)
              linalg.batch_matmul ins(%24, %18 : memref<1x64x128xf32>, memref<1x128x128xf32>) outs(%17 : memref<1x64x128xf32>)
              linalg.fill ins(%cst_3 : f32) outs(%21 : memref<1x64xf32>)
              linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%17 : memref<1x64x128xf32>) outs(%21 : memref<1x64xf32>) {
              ^bb0(%in: f32, %out: f32):
                %34 = arith.maximumf %in, %out : f32
                linalg.yield %34 : f32
              }
              linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%20, %21 : memref<1x64xf32>, memref<1x64xf32>) outs(%21 : memref<1x64xf32>) {
              ^bb0(%in: f32, %in_7: f32, %out: f32):
                %34 = arith.truncf %cst_0 : f64 to f32
                %35 = arith.mulf %in_7, %34 : f32
                %36 = arith.cmpf ogt, %in, %35 : f32
                %37 = arith.select %36, %in, %35 : f32
                linalg.yield %37 : f32
              }
              linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%17, %21 : memref<1x64x128xf32>, memref<1x64xf32>) outs(%17 : memref<1x64x128xf32>) {
              ^bb0(%in: f32, %in_7: f32, %out: f32):
                %34 = arith.truncf %cst_0 : f64 to f32
                %35 = arith.mulf %in, %34 : f32
                %36 = arith.subf %35, %in_7 : f32
                %37 = math.powf %cst, %36 : f32
                linalg.yield %37 : f32
              }
              linalg.fill ins(%cst_1 : f32) outs(%22 : memref<1x64xf32>)
              linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%17 : memref<1x64x128xf32>) outs(%22 : memref<1x64xf32>) {
              ^bb0(%in: f32, %out: f32):
                %34 = arith.addf %in, %out : f32
                linalg.yield %34 : f32
              }
              linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%20, %21 : memref<1x64xf32>, memref<1x64xf32>) outs(%23 : memref<1x64xf32>) {
              ^bb0(%in: f32, %in_7: f32, %out: f32):
                %34 = arith.subf %in, %in_7 : f32
                %35 = math.powf %cst, %34 : f32
                linalg.yield %35 : f32
              }
              linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%19, %23, %22 : memref<1x64xf32>, memref<1x64xf32>, memref<1x64xf32>) outs(%19 : memref<1x64xf32>) {
              ^bb0(%in: f32, %in_7: f32, %in_8: f32, %out: f32):
                %34 = arith.mulf %in, %in_7 : f32
                %35 = arith.addf %34, %in_8 : f32
                linalg.yield %35 : f32
              }
              %32 = arith.muli %arg8, %c16384 : index
              %33 = arith.addi %27, %32 : index
              %reinterpret_cast_6 = memref.reinterpret_cast %arg1 to offset: [%33], sizes: [1, 128, 128], strides: [524288, 128, 1] : memref<32x4096x128xf32> to memref<1x128x128xf32, strided<[524288, 128, 1], offset: ?>>
              loom.copy %reinterpret_cast_6, %16 src_mem_space @DRAM dst_mem_space @L1, interconnect : [], broadcast : [1, 1] : memref<1x128x128xf32, strided<[524288, 128, 1], offset: ?>>, memref<1x128x128xf32>
              linalg.fill ins(%cst_1 : f32) outs(%26 : memref<1x64x128xf32>)
              linalg.batch_matmul ins(%17, %16 : memref<1x64x128xf32>, memref<1x128x128xf32>) outs(%26 : memref<1x64x128xf32>)
              linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%26, %25, %23 : memref<1x64x128xf32>, memref<1x64x128xf32>, memref<1x64xf32>) outs(%25 : memref<1x64x128xf32>) {
              ^bb0(%in: f32, %in_7: f32, %in_8: f32, %out: f32):
                %34 = arith.mulf %in_7, %in_8 : f32
                %35 = arith.addf %in, %34 : f32
                linalg.yield %35 : f32
              }
              linalg.copy ins(%21 : memref<1x64xf32>) outs(%20 : memref<1x64xf32>)
            }
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%25, %19 : memref<1x64x128xf32>, memref<1x64xf32>) outs(%24 : memref<1x64x128xf32>) {
            ^bb0(%in: f32, %in_5: f32, %out: f32):
              %30 = arith.divf %in, %in_5 : f32
              linalg.yield %30 : f32
            }
            %reinterpret_cast_4 = memref.reinterpret_cast %arg3 to offset: [%29], sizes: [1, 64, 128], strides: [524288, 128, 1] : memref<32x4096x128xf32> to memref<1x64x128xf32, strided<[524288, 128, 1], offset: ?>>
            loom.copy %24, %reinterpret_cast_4 src_mem_space @L1 dst_mem_space @DRAM, interconnect : [], broadcast : [1, 1] : memref<1x64x128xf32>, memref<1x64x128xf32, strided<[524288, 128, 1], offset: ?>>
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
            %17 = loom.alloc [1, 64, 128] on @L1 : memref<1x64x128xf32>
            %18 = loom.alloc [1, 128, 128] on @L1 : memref<1x128x128xf32>
            %19 = loom.alloc [1, 64] on @L1 : memref<1x64xf32>
            %20 = loom.alloc [1, 64] on @L1 : memref<1x64xf32>
            %21 = loom.alloc [1, 64] on @L1 : memref<1x64xf32>
            %22 = loom.alloc [1, 64] on @L1 : memref<1x64xf32>
            %23 = loom.alloc [1, 64] on @L1 : memref<1x64xf32>
            %24 = loom.alloc [1, 64, 128] on @L1 : memref<1x64x128xf32>
            %25 = loom.alloc [1, 64, 128] on @L1 : memref<1x64x128xf32>
            %26 = loom.alloc [1, 64, 128] on @L1 : memref<1x64x128xf32>
            %27 = arith.muli %13, %c524288 overflow<nsw> : index
            %28 = arith.muli %15, %c8192 : index
            %29 = arith.addi %27, %28 : index
            %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%29], sizes: [1, 64, 128], strides: [524288, 128, 1] : memref<32x4096x128xf32> to memref<1x64x128xf32, strided<[524288, 128, 1], offset: ?>>
            loom.copy %reinterpret_cast, %24 src_mem_space @DRAM dst_mem_space @L1, interconnect : [], broadcast : [1, 1] : memref<1x64x128xf32, strided<[524288, 128, 1], offset: ?>>, memref<1x64x128xf32>
            linalg.fill ins(%cst_1 : f32) outs(%25 : memref<1x64x128xf32>)
            linalg.fill ins(%cst_2 : f32) outs(%19 : memref<1x64xf32>)
            linalg.fill ins(%cst_3 : f32) outs(%20 : memref<1x64xf32>)
            scf.for %arg8 = %c0 to %c32 step %c1 {
              %30 = arith.muli %arg8, %c128 : index
              %31 = arith.addi %27, %30 : index
              %reinterpret_cast_5 = memref.reinterpret_cast %arg0 to offset: [%31], sizes: [1, 128, 128], strides: [524288, 4096, 1] : memref<32x128x4096xf32> to memref<1x128x128xf32, strided<[524288, 4096, 1], offset: ?>>
              loom.copy %reinterpret_cast_5, %18 src_mem_space @DRAM dst_mem_space @L1, interconnect : [@horizontal_links], broadcast : [1, 8] : memref<1x128x128xf32, strided<[524288, 4096, 1], offset: ?>>, memref<1x128x128xf32>
              linalg.fill ins(%cst_1 : f32) outs(%17 : memref<1x64x128xf32>)
              linalg.batch_matmul ins(%24, %18 : memref<1x64x128xf32>, memref<1x128x128xf32>) outs(%17 : memref<1x64x128xf32>)
              linalg.fill ins(%cst_3 : f32) outs(%21 : memref<1x64xf32>)
              linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%17 : memref<1x64x128xf32>) outs(%21 : memref<1x64xf32>) {
              ^bb0(%in: f32, %out: f32):
                %34 = arith.maximumf %in, %out : f32
                linalg.yield %34 : f32
              }
              linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%20, %21 : memref<1x64xf32>, memref<1x64xf32>) outs(%21 : memref<1x64xf32>) {
              ^bb0(%in: f32, %in_7: f32, %out: f32):
                %34 = arith.truncf %cst_0 : f64 to f32
                %35 = arith.mulf %in_7, %34 : f32
                %36 = arith.cmpf ogt, %in, %35 : f32
                %37 = arith.select %36, %in, %35 : f32
                linalg.yield %37 : f32
              }
              linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%17, %21 : memref<1x64x128xf32>, memref<1x64xf32>) outs(%17 : memref<1x64x128xf32>) {
              ^bb0(%in: f32, %in_7: f32, %out: f32):
                %34 = arith.truncf %cst_0 : f64 to f32
                %35 = arith.mulf %in, %34 : f32
                %36 = arith.subf %35, %in_7 : f32
                %37 = math.powf %cst, %36 : f32
                linalg.yield %37 : f32
              }
              linalg.fill ins(%cst_1 : f32) outs(%22 : memref<1x64xf32>)
              linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%17 : memref<1x64x128xf32>) outs(%22 : memref<1x64xf32>) {
              ^bb0(%in: f32, %out: f32):
                %34 = arith.addf %in, %out : f32
                linalg.yield %34 : f32
              }
              linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%20, %21 : memref<1x64xf32>, memref<1x64xf32>) outs(%23 : memref<1x64xf32>) {
              ^bb0(%in: f32, %in_7: f32, %out: f32):
                %34 = arith.subf %in, %in_7 : f32
                %35 = math.powf %cst, %34 : f32
                linalg.yield %35 : f32
              }
              linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%19, %23, %22 : memref<1x64xf32>, memref<1x64xf32>, memref<1x64xf32>) outs(%19 : memref<1x64xf32>) {
              ^bb0(%in: f32, %in_7: f32, %in_8: f32, %out: f32):
                %34 = arith.mulf %in, %in_7 : f32
                %35 = arith.addf %34, %in_8 : f32
                linalg.yield %35 : f32
              }
              %32 = arith.muli %arg8, %c16384 : index
              %33 = arith.addi %27, %32 : index
              %reinterpret_cast_6 = memref.reinterpret_cast %arg1 to offset: [%33], sizes: [1, 128, 128], strides: [524288, 128, 1] : memref<32x4096x128xf32> to memref<1x128x128xf32, strided<[524288, 128, 1], offset: ?>>
              loom.copy %reinterpret_cast_6, %16 src_mem_space @DRAM dst_mem_space @L1, interconnect : [@horizontal_links], broadcast : [1, 8] : memref<1x128x128xf32, strided<[524288, 128, 1], offset: ?>>, memref<1x128x128xf32>
              linalg.fill ins(%cst_1 : f32) outs(%26 : memref<1x64x128xf32>)
              linalg.batch_matmul ins(%17, %16 : memref<1x64x128xf32>, memref<1x128x128xf32>) outs(%26 : memref<1x64x128xf32>)
              linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%26, %25, %23 : memref<1x64x128xf32>, memref<1x64x128xf32>, memref<1x64xf32>) outs(%25 : memref<1x64x128xf32>) {
              ^bb0(%in: f32, %in_7: f32, %in_8: f32, %out: f32):
                %34 = arith.mulf %in_7, %in_8 : f32
                %35 = arith.addf %in, %34 : f32
                linalg.yield %35 : f32
              }
              linalg.copy ins(%21 : memref<1x64xf32>) outs(%20 : memref<1x64xf32>)
            }
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%25, %19 : memref<1x64x128xf32>, memref<1x64xf32>) outs(%24 : memref<1x64x128xf32>) {
            ^bb0(%in: f32, %in_5: f32, %out: f32):
              %30 = arith.divf %in, %in_5 : f32
              linalg.yield %30 : f32
            }
            %reinterpret_cast_4 = memref.reinterpret_cast %arg3 to offset: [%29], sizes: [1, 64, 128], strides: [524288, 128, 1] : memref<32x4096x128xf32> to memref<1x64x128xf32, strided<[524288, 128, 1], offset: ?>>
            loom.copy %24, %reinterpret_cast_4 src_mem_space @L1 dst_mem_space @DRAM, interconnect : [], broadcast : [1, 1] : memref<1x64x128xf32>, memref<1x64x128xf32, strided<[524288, 128, 1], offset: ?>>
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
            %17 = loom.alloc [1, 64, 128] on @L1 : memref<1x64x128xf32>
            %18 = loom.alloc [1, 128, 128] on @L1 : memref<1x128x128xf32>
            %19 = loom.alloc [1, 64] on @L1 : memref<1x64xf32>
            %20 = loom.alloc [1, 64] on @L1 : memref<1x64xf32>
            %21 = loom.alloc [1, 64] on @L1 : memref<1x64xf32>
            %22 = loom.alloc [1, 64] on @L1 : memref<1x64xf32>
            %23 = loom.alloc [1, 64] on @L1 : memref<1x64xf32>
            %24 = loom.alloc [1, 64, 128] on @L1 : memref<1x64x128xf32>
            %25 = loom.alloc [1, 64, 128] on @L1 : memref<1x64x128xf32>
            %26 = loom.alloc [1, 64, 128] on @L1 : memref<1x64x128xf32>
            %27 = arith.muli %13, %c524288 overflow<nsw> : index
            %28 = arith.muli %15, %c8192 : index
            %29 = arith.addi %27, %28 : index
            %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%29], sizes: [1, 64, 128], strides: [524288, 128, 1] : memref<32x4096x128xf32> to memref<1x64x128xf32, strided<[524288, 128, 1], offset: ?>>
            loom.copy %reinterpret_cast, %24 src_mem_space @DRAM dst_mem_space @L1, interconnect : [], broadcast : [1, 1] : memref<1x64x128xf32, strided<[524288, 128, 1], offset: ?>>, memref<1x64x128xf32>
            linalg.fill ins(%cst_1 : f32) outs(%25 : memref<1x64x128xf32>)
            linalg.fill ins(%cst_2 : f32) outs(%19 : memref<1x64xf32>)
            linalg.fill ins(%cst_3 : f32) outs(%20 : memref<1x64xf32>)
            scf.for %arg8 = %c0 to %c32 step %c1 {
              %30 = arith.muli %arg8, %c128 : index
              %31 = arith.addi %27, %30 : index
              %reinterpret_cast_5 = memref.reinterpret_cast %arg0 to offset: [%31], sizes: [1, 128, 128], strides: [524288, 4096, 1] : memref<32x128x4096xf32> to memref<1x128x128xf32, strided<[524288, 4096, 1], offset: ?>>
              loom.copy %reinterpret_cast_5, %18 src_mem_space @DRAM dst_mem_space @L1, interconnect : [], broadcast : [1, 1] : memref<1x128x128xf32, strided<[524288, 4096, 1], offset: ?>>, memref<1x128x128xf32>
              linalg.fill ins(%cst_1 : f32) outs(%17 : memref<1x64x128xf32>)
              linalg.batch_matmul ins(%24, %18 : memref<1x64x128xf32>, memref<1x128x128xf32>) outs(%17 : memref<1x64x128xf32>)
              linalg.fill ins(%cst_3 : f32) outs(%21 : memref<1x64xf32>)
              linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%17 : memref<1x64x128xf32>) outs(%21 : memref<1x64xf32>) {
              ^bb0(%in: f32, %out: f32):
                %34 = arith.maximumf %in, %out : f32
                linalg.yield %34 : f32
              }
              linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%20, %21 : memref<1x64xf32>, memref<1x64xf32>) outs(%21 : memref<1x64xf32>) {
              ^bb0(%in: f32, %in_7: f32, %out: f32):
                %34 = arith.truncf %cst_0 : f64 to f32
                %35 = arith.mulf %in_7, %34 : f32
                %36 = arith.cmpf ogt, %in, %35 : f32
                %37 = arith.select %36, %in, %35 : f32
                linalg.yield %37 : f32
              }
              linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%17, %21 : memref<1x64x128xf32>, memref<1x64xf32>) outs(%17 : memref<1x64x128xf32>) {
              ^bb0(%in: f32, %in_7: f32, %out: f32):
                %34 = arith.truncf %cst_0 : f64 to f32
                %35 = arith.mulf %in, %34 : f32
                %36 = arith.subf %35, %in_7 : f32
                %37 = math.powf %cst, %36 : f32
                linalg.yield %37 : f32
              }
              linalg.fill ins(%cst_1 : f32) outs(%22 : memref<1x64xf32>)
              linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%17 : memref<1x64x128xf32>) outs(%22 : memref<1x64xf32>) {
              ^bb0(%in: f32, %out: f32):
                %34 = arith.addf %in, %out : f32
                linalg.yield %34 : f32
              }
              linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%20, %21 : memref<1x64xf32>, memref<1x64xf32>) outs(%23 : memref<1x64xf32>) {
              ^bb0(%in: f32, %in_7: f32, %out: f32):
                %34 = arith.subf %in, %in_7 : f32
                %35 = math.powf %cst, %34 : f32
                linalg.yield %35 : f32
              }
              linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%19, %23, %22 : memref<1x64xf32>, memref<1x64xf32>, memref<1x64xf32>) outs(%19 : memref<1x64xf32>) {
              ^bb0(%in: f32, %in_7: f32, %in_8: f32, %out: f32):
                %34 = arith.mulf %in, %in_7 : f32
                %35 = arith.addf %34, %in_8 : f32
                linalg.yield %35 : f32
              }
              %32 = arith.muli %arg8, %c16384 : index
              %33 = arith.addi %27, %32 : index
              %reinterpret_cast_6 = memref.reinterpret_cast %arg1 to offset: [%33], sizes: [1, 128, 128], strides: [524288, 128, 1] : memref<32x4096x128xf32> to memref<1x128x128xf32, strided<[524288, 128, 1], offset: ?>>
              loom.copy %reinterpret_cast_6, %16 src_mem_space @DRAM dst_mem_space @L1, interconnect : [], broadcast : [1, 1] : memref<1x128x128xf32, strided<[524288, 128, 1], offset: ?>>, memref<1x128x128xf32>
              linalg.fill ins(%cst_1 : f32) outs(%26 : memref<1x64x128xf32>)
              linalg.batch_matmul ins(%17, %16 : memref<1x64x128xf32>, memref<1x128x128xf32>) outs(%26 : memref<1x64x128xf32>)
              linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%26, %25, %23 : memref<1x64x128xf32>, memref<1x64x128xf32>, memref<1x64xf32>) outs(%25 : memref<1x64x128xf32>) {
              ^bb0(%in: f32, %in_7: f32, %in_8: f32, %out: f32):
                %34 = arith.mulf %in_7, %in_8 : f32
                %35 = arith.addf %in, %34 : f32
                linalg.yield %35 : f32
              }
              linalg.copy ins(%21 : memref<1x64xf32>) outs(%20 : memref<1x64xf32>)
            }
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%25, %19 : memref<1x64x128xf32>, memref<1x64xf32>) outs(%24 : memref<1x64x128xf32>) {
            ^bb0(%in: f32, %in_5: f32, %out: f32):
              %30 = arith.divf %in, %in_5 : f32
              linalg.yield %30 : f32
            }
            %reinterpret_cast_4 = memref.reinterpret_cast %arg3 to offset: [%29], sizes: [1, 64, 128], strides: [524288, 128, 1] : memref<32x4096x128xf32> to memref<1x64x128xf32, strided<[524288, 128, 1], offset: ?>>
            loom.copy %24, %reinterpret_cast_4 src_mem_space @L1 dst_mem_space @DRAM, interconnect : [], broadcast : [1, 1] : memref<1x64x128xf32>, memref<1x64x128xf32, strided<[524288, 128, 1], offset: ?>>
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
            %17 = loom.alloc [1, 64, 128] on @L1 : memref<1x64x128xf32>
            %18 = loom.alloc [1, 128, 128] on @L1 : memref<1x128x128xf32>
            %19 = loom.alloc [1, 64] on @L1 : memref<1x64xf32>
            %20 = loom.alloc [1, 64] on @L1 : memref<1x64xf32>
            %21 = loom.alloc [1, 64] on @L1 : memref<1x64xf32>
            %22 = loom.alloc [1, 64] on @L1 : memref<1x64xf32>
            %23 = loom.alloc [1, 64] on @L1 : memref<1x64xf32>
            %24 = loom.alloc [1, 64, 128] on @L1 : memref<1x64x128xf32>
            %25 = loom.alloc [1, 64, 128] on @L1 : memref<1x64x128xf32>
            %26 = loom.alloc [1, 64, 128] on @L1 : memref<1x64x128xf32>
            %27 = arith.muli %13, %c524288 overflow<nsw> : index
            %28 = arith.muli %15, %c8192 : index
            %29 = arith.addi %27, %28 : index
            %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%29], sizes: [1, 64, 128], strides: [524288, 128, 1] : memref<32x4096x128xf32> to memref<1x64x128xf32, strided<[524288, 128, 1], offset: ?>>
            loom.copy %reinterpret_cast, %24 src_mem_space @DRAM dst_mem_space @L1, interconnect : [], broadcast : [1, 1] : memref<1x64x128xf32, strided<[524288, 128, 1], offset: ?>>, memref<1x64x128xf32>
            linalg.fill ins(%cst_1 : f32) outs(%25 : memref<1x64x128xf32>)
            linalg.fill ins(%cst_2 : f32) outs(%19 : memref<1x64xf32>)
            linalg.fill ins(%cst_3 : f32) outs(%20 : memref<1x64xf32>)
            scf.for %arg8 = %c0 to %c32 step %c1 {
              %30 = arith.muli %arg8, %c128 : index
              %31 = arith.addi %27, %30 : index
              %reinterpret_cast_5 = memref.reinterpret_cast %arg0 to offset: [%31], sizes: [1, 128, 128], strides: [524288, 4096, 1] : memref<32x128x4096xf32> to memref<1x128x128xf32, strided<[524288, 4096, 1], offset: ?>>
              loom.copy %reinterpret_cast_5, %18 src_mem_space @DRAM dst_mem_space @L1, interconnect : [], broadcast : [1, 1] : memref<1x128x128xf32, strided<[524288, 4096, 1], offset: ?>>, memref<1x128x128xf32>
              linalg.fill ins(%cst_1 : f32) outs(%17 : memref<1x64x128xf32>)
              linalg.batch_matmul ins(%24, %18 : memref<1x64x128xf32>, memref<1x128x128xf32>) outs(%17 : memref<1x64x128xf32>)
              linalg.fill ins(%cst_3 : f32) outs(%21 : memref<1x64xf32>)
              linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%17 : memref<1x64x128xf32>) outs(%21 : memref<1x64xf32>) {
              ^bb0(%in: f32, %out: f32):
                %34 = arith.maximumf %in, %out : f32
                linalg.yield %34 : f32
              }
              linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%20, %21 : memref<1x64xf32>, memref<1x64xf32>) outs(%21 : memref<1x64xf32>) {
              ^bb0(%in: f32, %in_7: f32, %out: f32):
                %34 = arith.truncf %cst_0 : f64 to f32
                %35 = arith.mulf %in_7, %34 : f32
                %36 = arith.cmpf ogt, %in, %35 : f32
                %37 = arith.select %36, %in, %35 : f32
                linalg.yield %37 : f32
              }
              linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%17, %21 : memref<1x64x128xf32>, memref<1x64xf32>) outs(%17 : memref<1x64x128xf32>) {
              ^bb0(%in: f32, %in_7: f32, %out: f32):
                %34 = arith.truncf %cst_0 : f64 to f32
                %35 = arith.mulf %in, %34 : f32
                %36 = arith.subf %35, %in_7 : f32
                %37 = math.powf %cst, %36 : f32
                linalg.yield %37 : f32
              }
              linalg.fill ins(%cst_1 : f32) outs(%22 : memref<1x64xf32>)
              linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%17 : memref<1x64x128xf32>) outs(%22 : memref<1x64xf32>) {
              ^bb0(%in: f32, %out: f32):
                %34 = arith.addf %in, %out : f32
                linalg.yield %34 : f32
              }
              linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%20, %21 : memref<1x64xf32>, memref<1x64xf32>) outs(%23 : memref<1x64xf32>) {
              ^bb0(%in: f32, %in_7: f32, %out: f32):
                %34 = arith.subf %in, %in_7 : f32
                %35 = math.powf %cst, %34 : f32
                linalg.yield %35 : f32
              }
              linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%19, %23, %22 : memref<1x64xf32>, memref<1x64xf32>, memref<1x64xf32>) outs(%19 : memref<1x64xf32>) {
              ^bb0(%in: f32, %in_7: f32, %in_8: f32, %out: f32):
                %34 = arith.mulf %in, %in_7 : f32
                %35 = arith.addf %34, %in_8 : f32
                linalg.yield %35 : f32
              }
              %32 = arith.muli %arg8, %c16384 : index
              %33 = arith.addi %27, %32 : index
              %reinterpret_cast_6 = memref.reinterpret_cast %arg1 to offset: [%33], sizes: [1, 128, 128], strides: [524288, 128, 1] : memref<32x4096x128xf32> to memref<1x128x128xf32, strided<[524288, 128, 1], offset: ?>>
              loom.copy %reinterpret_cast_6, %16 src_mem_space @DRAM dst_mem_space @L1, interconnect : [@horizontal_links], broadcast : [1, 8] : memref<1x128x128xf32, strided<[524288, 128, 1], offset: ?>>, memref<1x128x128xf32>
              linalg.fill ins(%cst_1 : f32) outs(%26 : memref<1x64x128xf32>)
              linalg.batch_matmul ins(%17, %16 : memref<1x64x128xf32>, memref<1x128x128xf32>) outs(%26 : memref<1x64x128xf32>)
              linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%26, %25, %23 : memref<1x64x128xf32>, memref<1x64x128xf32>, memref<1x64xf32>) outs(%25 : memref<1x64x128xf32>) {
              ^bb0(%in: f32, %in_7: f32, %in_8: f32, %out: f32):
                %34 = arith.mulf %in_7, %in_8 : f32
                %35 = arith.addf %in, %34 : f32
                linalg.yield %35 : f32
              }
              linalg.copy ins(%21 : memref<1x64xf32>) outs(%20 : memref<1x64xf32>)
            }
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%25, %19 : memref<1x64x128xf32>, memref<1x64xf32>) outs(%24 : memref<1x64x128xf32>) {
            ^bb0(%in: f32, %in_5: f32, %out: f32):
              %30 = arith.divf %in, %in_5 : f32
              linalg.yield %30 : f32
            }
            %reinterpret_cast_4 = memref.reinterpret_cast %arg3 to offset: [%29], sizes: [1, 64, 128], strides: [524288, 128, 1] : memref<32x4096x128xf32> to memref<1x64x128xf32, strided<[524288, 128, 1], offset: ?>>
            loom.copy %24, %reinterpret_cast_4 src_mem_space @L1 dst_mem_space @DRAM, interconnect : [], broadcast : [1, 1] : memref<1x64x128xf32>, memref<1x64x128xf32, strided<[524288, 128, 1], offset: ?>>
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
            %17 = loom.alloc [1, 64, 128] on @L1 : memref<1x64x128xf32>
            %18 = loom.alloc [1, 128, 128] on @L1 : memref<1x128x128xf32>
            %19 = loom.alloc [1, 64] on @L1 : memref<1x64xf32>
            %20 = loom.alloc [1, 64] on @L1 : memref<1x64xf32>
            %21 = loom.alloc [1, 64] on @L1 : memref<1x64xf32>
            %22 = loom.alloc [1, 64] on @L1 : memref<1x64xf32>
            %23 = loom.alloc [1, 64] on @L1 : memref<1x64xf32>
            %24 = loom.alloc [1, 64, 128] on @L1 : memref<1x64x128xf32>
            %25 = loom.alloc [1, 64, 128] on @L1 : memref<1x64x128xf32>
            %26 = loom.alloc [1, 64, 128] on @L1 : memref<1x64x128xf32>
            %27 = arith.muli %13, %c524288 overflow<nsw> : index
            %28 = arith.muli %15, %c8192 : index
            %29 = arith.addi %27, %28 : index
            %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%29], sizes: [1, 64, 128], strides: [524288, 128, 1] : memref<32x4096x128xf32> to memref<1x64x128xf32, strided<[524288, 128, 1], offset: ?>>
            loom.copy %reinterpret_cast, %24 src_mem_space @DRAM dst_mem_space @L1, interconnect : [], broadcast : [1, 1] : memref<1x64x128xf32, strided<[524288, 128, 1], offset: ?>>, memref<1x64x128xf32>
            linalg.fill ins(%cst_1 : f32) outs(%25 : memref<1x64x128xf32>)
            linalg.fill ins(%cst_2 : f32) outs(%19 : memref<1x64xf32>)
            linalg.fill ins(%cst_3 : f32) outs(%20 : memref<1x64xf32>)
            scf.for %arg8 = %c0 to %c32 step %c1 {
              %30 = arith.muli %arg8, %c128 : index
              %31 = arith.addi %27, %30 : index
              %reinterpret_cast_5 = memref.reinterpret_cast %arg0 to offset: [%31], sizes: [1, 128, 128], strides: [524288, 4096, 1] : memref<32x128x4096xf32> to memref<1x128x128xf32, strided<[524288, 4096, 1], offset: ?>>
              loom.copy %reinterpret_cast_5, %18 src_mem_space @DRAM dst_mem_space @L1, interconnect : [@horizontal_links], broadcast : [1, 8] : memref<1x128x128xf32, strided<[524288, 4096, 1], offset: ?>>, memref<1x128x128xf32>
              linalg.fill ins(%cst_1 : f32) outs(%17 : memref<1x64x128xf32>)
              linalg.batch_matmul ins(%24, %18 : memref<1x64x128xf32>, memref<1x128x128xf32>) outs(%17 : memref<1x64x128xf32>)
              linalg.fill ins(%cst_3 : f32) outs(%21 : memref<1x64xf32>)
              linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%17 : memref<1x64x128xf32>) outs(%21 : memref<1x64xf32>) {
              ^bb0(%in: f32, %out: f32):
                %34 = arith.maximumf %in, %out : f32
                linalg.yield %34 : f32
              }
              linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%20, %21 : memref<1x64xf32>, memref<1x64xf32>) outs(%21 : memref<1x64xf32>) {
              ^bb0(%in: f32, %in_7: f32, %out: f32):
                %34 = arith.truncf %cst_0 : f64 to f32
                %35 = arith.mulf %in_7, %34 : f32
                %36 = arith.cmpf ogt, %in, %35 : f32
                %37 = arith.select %36, %in, %35 : f32
                linalg.yield %37 : f32
              }
              linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%17, %21 : memref<1x64x128xf32>, memref<1x64xf32>) outs(%17 : memref<1x64x128xf32>) {
              ^bb0(%in: f32, %in_7: f32, %out: f32):
                %34 = arith.truncf %cst_0 : f64 to f32
                %35 = arith.mulf %in, %34 : f32
                %36 = arith.subf %35, %in_7 : f32
                %37 = math.powf %cst, %36 : f32
                linalg.yield %37 : f32
              }
              linalg.fill ins(%cst_1 : f32) outs(%22 : memref<1x64xf32>)
              linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%17 : memref<1x64x128xf32>) outs(%22 : memref<1x64xf32>) {
              ^bb0(%in: f32, %out: f32):
                %34 = arith.addf %in, %out : f32
                linalg.yield %34 : f32
              }
              linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%20, %21 : memref<1x64xf32>, memref<1x64xf32>) outs(%23 : memref<1x64xf32>) {
              ^bb0(%in: f32, %in_7: f32, %out: f32):
                %34 = arith.subf %in, %in_7 : f32
                %35 = math.powf %cst, %34 : f32
                linalg.yield %35 : f32
              }
              linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%19, %23, %22 : memref<1x64xf32>, memref<1x64xf32>, memref<1x64xf32>) outs(%19 : memref<1x64xf32>) {
              ^bb0(%in: f32, %in_7: f32, %in_8: f32, %out: f32):
                %34 = arith.mulf %in, %in_7 : f32
                %35 = arith.addf %34, %in_8 : f32
                linalg.yield %35 : f32
              }
              %32 = arith.muli %arg8, %c16384 : index
              %33 = arith.addi %27, %32 : index
              %reinterpret_cast_6 = memref.reinterpret_cast %arg1 to offset: [%33], sizes: [1, 128, 128], strides: [524288, 128, 1] : memref<32x4096x128xf32> to memref<1x128x128xf32, strided<[524288, 128, 1], offset: ?>>
              loom.copy %reinterpret_cast_6, %16 src_mem_space @DRAM dst_mem_space @L1, interconnect : [], broadcast : [1, 1] : memref<1x128x128xf32, strided<[524288, 128, 1], offset: ?>>, memref<1x128x128xf32>
              linalg.fill ins(%cst_1 : f32) outs(%26 : memref<1x64x128xf32>)
              linalg.batch_matmul ins(%17, %16 : memref<1x64x128xf32>, memref<1x128x128xf32>) outs(%26 : memref<1x64x128xf32>)
              linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%26, %25, %23 : memref<1x64x128xf32>, memref<1x64x128xf32>, memref<1x64xf32>) outs(%25 : memref<1x64x128xf32>) {
              ^bb0(%in: f32, %in_7: f32, %in_8: f32, %out: f32):
                %34 = arith.mulf %in_7, %in_8 : f32
                %35 = arith.addf %in, %34 : f32
                linalg.yield %35 : f32
              }
              linalg.copy ins(%21 : memref<1x64xf32>) outs(%20 : memref<1x64xf32>)
            }
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%25, %19 : memref<1x64x128xf32>, memref<1x64xf32>) outs(%24 : memref<1x64x128xf32>) {
            ^bb0(%in: f32, %in_5: f32, %out: f32):
              %30 = arith.divf %in, %in_5 : f32
              linalg.yield %30 : f32
            }
            %reinterpret_cast_4 = memref.reinterpret_cast %arg3 to offset: [%29], sizes: [1, 64, 128], strides: [524288, 128, 1] : memref<32x4096x128xf32> to memref<1x64x128xf32, strided<[524288, 128, 1], offset: ?>>
            loom.copy %24, %reinterpret_cast_4 src_mem_space @L1 dst_mem_space @DRAM, interconnect : [], broadcast : [1, 1] : memref<1x64x128xf32>, memref<1x64x128xf32, strided<[524288, 128, 1], offset: ?>>
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
            %17 = loom.alloc [1, 64, 128] on @L1 : memref<1x64x128xf32>
            %18 = loom.alloc [1, 128, 128] on @L1 : memref<1x128x128xf32>
            %19 = loom.alloc [1, 64] on @L1 : memref<1x64xf32>
            %20 = loom.alloc [1, 64] on @L1 : memref<1x64xf32>
            %21 = loom.alloc [1, 64] on @L1 : memref<1x64xf32>
            %22 = loom.alloc [1, 64] on @L1 : memref<1x64xf32>
            %23 = loom.alloc [1, 64] on @L1 : memref<1x64xf32>
            %24 = loom.alloc [1, 64, 128] on @L1 : memref<1x64x128xf32>
            %25 = loom.alloc [1, 64, 128] on @L1 : memref<1x64x128xf32>
            %26 = loom.alloc [1, 64, 128] on @L1 : memref<1x64x128xf32>
            %27 = arith.muli %13, %c524288 overflow<nsw> : index
            %28 = arith.muli %15, %c8192 : index
            %29 = arith.addi %27, %28 : index
            %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%29], sizes: [1, 64, 128], strides: [524288, 128, 1] : memref<32x4096x128xf32> to memref<1x64x128xf32, strided<[524288, 128, 1], offset: ?>>
            loom.copy %reinterpret_cast, %24 src_mem_space @DRAM dst_mem_space @L1, interconnect : [], broadcast : [1, 1] : memref<1x64x128xf32, strided<[524288, 128, 1], offset: ?>>, memref<1x64x128xf32>
            linalg.fill ins(%cst_1 : f32) outs(%25 : memref<1x64x128xf32>)
            linalg.fill ins(%cst_2 : f32) outs(%19 : memref<1x64xf32>)
            linalg.fill ins(%cst_3 : f32) outs(%20 : memref<1x64xf32>)
            scf.for %arg8 = %c0 to %c32 step %c1 {
              %30 = arith.muli %arg8, %c128 : index
              %31 = arith.addi %27, %30 : index
              %reinterpret_cast_5 = memref.reinterpret_cast %arg0 to offset: [%31], sizes: [1, 128, 128], strides: [524288, 4096, 1] : memref<32x128x4096xf32> to memref<1x128x128xf32, strided<[524288, 4096, 1], offset: ?>>
              loom.copy %reinterpret_cast_5, %18 src_mem_space @DRAM dst_mem_space @L1, interconnect : [@horizontal_links], broadcast : [1, 8] : memref<1x128x128xf32, strided<[524288, 4096, 1], offset: ?>>, memref<1x128x128xf32>
              linalg.fill ins(%cst_1 : f32) outs(%17 : memref<1x64x128xf32>)
              linalg.batch_matmul ins(%24, %18 : memref<1x64x128xf32>, memref<1x128x128xf32>) outs(%17 : memref<1x64x128xf32>)
              linalg.fill ins(%cst_3 : f32) outs(%21 : memref<1x64xf32>)
              linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%17 : memref<1x64x128xf32>) outs(%21 : memref<1x64xf32>) {
              ^bb0(%in: f32, %out: f32):
                %34 = arith.maximumf %in, %out : f32
                linalg.yield %34 : f32
              }
              linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%20, %21 : memref<1x64xf32>, memref<1x64xf32>) outs(%21 : memref<1x64xf32>) {
              ^bb0(%in: f32, %in_7: f32, %out: f32):
                %34 = arith.truncf %cst_0 : f64 to f32
                %35 = arith.mulf %in_7, %34 : f32
                %36 = arith.cmpf ogt, %in, %35 : f32
                %37 = arith.select %36, %in, %35 : f32
                linalg.yield %37 : f32
              }
              linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%17, %21 : memref<1x64x128xf32>, memref<1x64xf32>) outs(%17 : memref<1x64x128xf32>) {
              ^bb0(%in: f32, %in_7: f32, %out: f32):
                %34 = arith.truncf %cst_0 : f64 to f32
                %35 = arith.mulf %in, %34 : f32
                %36 = arith.subf %35, %in_7 : f32
                %37 = math.powf %cst, %36 : f32
                linalg.yield %37 : f32
              }
              linalg.fill ins(%cst_1 : f32) outs(%22 : memref<1x64xf32>)
              linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%17 : memref<1x64x128xf32>) outs(%22 : memref<1x64xf32>) {
              ^bb0(%in: f32, %out: f32):
                %34 = arith.addf %in, %out : f32
                linalg.yield %34 : f32
              }
              linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%20, %21 : memref<1x64xf32>, memref<1x64xf32>) outs(%23 : memref<1x64xf32>) {
              ^bb0(%in: f32, %in_7: f32, %out: f32):
                %34 = arith.subf %in, %in_7 : f32
                %35 = math.powf %cst, %34 : f32
                linalg.yield %35 : f32
              }
              linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%19, %23, %22 : memref<1x64xf32>, memref<1x64xf32>, memref<1x64xf32>) outs(%19 : memref<1x64xf32>) {
              ^bb0(%in: f32, %in_7: f32, %in_8: f32, %out: f32):
                %34 = arith.mulf %in, %in_7 : f32
                %35 = arith.addf %34, %in_8 : f32
                linalg.yield %35 : f32
              }
              %32 = arith.muli %arg8, %c16384 : index
              %33 = arith.addi %27, %32 : index
              %reinterpret_cast_6 = memref.reinterpret_cast %arg1 to offset: [%33], sizes: [1, 128, 128], strides: [524288, 128, 1] : memref<32x4096x128xf32> to memref<1x128x128xf32, strided<[524288, 128, 1], offset: ?>>
              loom.copy %reinterpret_cast_6, %16 src_mem_space @DRAM dst_mem_space @L1, interconnect : [@horizontal_links], broadcast : [1, 8] : memref<1x128x128xf32, strided<[524288, 128, 1], offset: ?>>, memref<1x128x128xf32>
              linalg.fill ins(%cst_1 : f32) outs(%26 : memref<1x64x128xf32>)
              linalg.batch_matmul ins(%17, %16 : memref<1x64x128xf32>, memref<1x128x128xf32>) outs(%26 : memref<1x64x128xf32>)
              linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%26, %25, %23 : memref<1x64x128xf32>, memref<1x64x128xf32>, memref<1x64xf32>) outs(%25 : memref<1x64x128xf32>) {
              ^bb0(%in: f32, %in_7: f32, %in_8: f32, %out: f32):
                %34 = arith.mulf %in_7, %in_8 : f32
                %35 = arith.addf %in, %34 : f32
                linalg.yield %35 : f32
              }
              linalg.copy ins(%21 : memref<1x64xf32>) outs(%20 : memref<1x64xf32>)
            }
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%25, %19 : memref<1x64x128xf32>, memref<1x64xf32>) outs(%24 : memref<1x64x128xf32>) {
            ^bb0(%in: f32, %in_5: f32, %out: f32):
              %30 = arith.divf %in, %in_5 : f32
              linalg.yield %30 : f32
            }
            %reinterpret_cast_4 = memref.reinterpret_cast %arg3 to offset: [%29], sizes: [1, 64, 128], strides: [524288, 128, 1] : memref<32x4096x128xf32> to memref<1x64x128xf32, strided<[524288, 128, 1], offset: ?>>
            loom.copy %24, %reinterpret_cast_4 src_mem_space @L1 dst_mem_space @DRAM, interconnect : [], broadcast : [1, 1] : memref<1x64x128xf32>, memref<1x64x128xf32, strided<[524288, 128, 1], offset: ?>>
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
          %15 = loom.alloc [1, 64, 128] on @L1 : memref<1x64x128xf32>
          %16 = loom.alloc [1, 128, 128] on @L1 : memref<1x128x128xf32>
          %17 = loom.alloc [1, 64] on @L1 : memref<1x64xf32>
          %18 = loom.alloc [1, 64] on @L1 : memref<1x64xf32>
          %19 = loom.alloc [1, 64] on @L1 : memref<1x64xf32>
          %20 = loom.alloc [1, 64] on @L1 : memref<1x64xf32>
          %21 = loom.alloc [1, 64] on @L1 : memref<1x64xf32>
          %22 = loom.alloc [1, 64, 128] on @L1 : memref<1x64x128xf32>
          %23 = loom.alloc [1, 64, 128] on @L1 : memref<1x64x128xf32>
          %24 = loom.alloc [1, 64, 128] on @L1 : memref<1x64x128xf32>
          %25 = arith.muli %arg6, %c524288 overflow<nsw> : index
          %26 = arith.muli %13, %c8192 : index
          %27 = arith.addi %25, %26 : index
          %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%27], sizes: [1, 64, 128], strides: [524288, 128, 1] : memref<32x4096x128xf32> to memref<1x64x128xf32, strided<[524288, 128, 1], offset: ?>>
          loom.copy %reinterpret_cast, %22 src_mem_space @DRAM dst_mem_space @L1, interconnect : [], broadcast : [1, 1] : memref<1x64x128xf32, strided<[524288, 128, 1], offset: ?>>, memref<1x64x128xf32>
          linalg.fill ins(%cst_1 : f32) outs(%23 : memref<1x64x128xf32>)
          linalg.fill ins(%cst_2 : f32) outs(%17 : memref<1x64xf32>)
          linalg.fill ins(%cst_3 : f32) outs(%18 : memref<1x64xf32>)
          scf.for %arg7 = %c0 to %c32 step %c1 {
            %28 = arith.muli %arg7, %c128 : index
            %29 = arith.addi %25, %28 : index
            %reinterpret_cast_5 = memref.reinterpret_cast %arg0 to offset: [%29], sizes: [1, 128, 128], strides: [524288, 4096, 1] : memref<32x128x4096xf32> to memref<1x128x128xf32, strided<[524288, 4096, 1], offset: ?>>
            loom.copy %reinterpret_cast_5, %16 src_mem_space @DRAM dst_mem_space @L1, interconnect : [], broadcast : [1, 1] : memref<1x128x128xf32, strided<[524288, 4096, 1], offset: ?>>, memref<1x128x128xf32>
            linalg.fill ins(%cst_1 : f32) outs(%15 : memref<1x64x128xf32>)
            linalg.batch_matmul ins(%22, %16 : memref<1x64x128xf32>, memref<1x128x128xf32>) outs(%15 : memref<1x64x128xf32>)
            linalg.fill ins(%cst_3 : f32) outs(%19 : memref<1x64xf32>)
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%15 : memref<1x64x128xf32>) outs(%19 : memref<1x64xf32>) {
            ^bb0(%in: f32, %out: f32):
              %32 = arith.maximumf %in, %out : f32
              linalg.yield %32 : f32
            }
            linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%18, %19 : memref<1x64xf32>, memref<1x64xf32>) outs(%19 : memref<1x64xf32>) {
            ^bb0(%in: f32, %in_7: f32, %out: f32):
              %32 = arith.truncf %cst_0 : f64 to f32
              %33 = arith.mulf %in_7, %32 : f32
              %34 = arith.cmpf ogt, %in, %33 : f32
              %35 = arith.select %34, %in, %33 : f32
              linalg.yield %35 : f32
            }
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%15, %19 : memref<1x64x128xf32>, memref<1x64xf32>) outs(%15 : memref<1x64x128xf32>) {
            ^bb0(%in: f32, %in_7: f32, %out: f32):
              %32 = arith.truncf %cst_0 : f64 to f32
              %33 = arith.mulf %in, %32 : f32
              %34 = arith.subf %33, %in_7 : f32
              %35 = math.powf %cst, %34 : f32
              linalg.yield %35 : f32
            }
            linalg.fill ins(%cst_1 : f32) outs(%20 : memref<1x64xf32>)
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%15 : memref<1x64x128xf32>) outs(%20 : memref<1x64xf32>) {
            ^bb0(%in: f32, %out: f32):
              %32 = arith.addf %in, %out : f32
              linalg.yield %32 : f32
            }
            linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%18, %19 : memref<1x64xf32>, memref<1x64xf32>) outs(%21 : memref<1x64xf32>) {
            ^bb0(%in: f32, %in_7: f32, %out: f32):
              %32 = arith.subf %in, %in_7 : f32
              %33 = math.powf %cst, %32 : f32
              linalg.yield %33 : f32
            }
            linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%17, %21, %20 : memref<1x64xf32>, memref<1x64xf32>, memref<1x64xf32>) outs(%17 : memref<1x64xf32>) {
            ^bb0(%in: f32, %in_7: f32, %in_8: f32, %out: f32):
              %32 = arith.mulf %in, %in_7 : f32
              %33 = arith.addf %32, %in_8 : f32
              linalg.yield %33 : f32
            }
            %30 = arith.muli %arg7, %c16384 : index
            %31 = arith.addi %25, %30 : index
            %reinterpret_cast_6 = memref.reinterpret_cast %arg1 to offset: [%31], sizes: [1, 128, 128], strides: [524288, 128, 1] : memref<32x4096x128xf32> to memref<1x128x128xf32, strided<[524288, 128, 1], offset: ?>>
            loom.copy %reinterpret_cast_6, %14 src_mem_space @DRAM dst_mem_space @L1, interconnect : [], broadcast : [1, 1] : memref<1x128x128xf32, strided<[524288, 128, 1], offset: ?>>, memref<1x128x128xf32>
            linalg.fill ins(%cst_1 : f32) outs(%24 : memref<1x64x128xf32>)
            linalg.batch_matmul ins(%15, %14 : memref<1x64x128xf32>, memref<1x128x128xf32>) outs(%24 : memref<1x64x128xf32>)
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%24, %23, %21 : memref<1x64x128xf32>, memref<1x64x128xf32>, memref<1x64xf32>) outs(%23 : memref<1x64x128xf32>) {
            ^bb0(%in: f32, %in_7: f32, %in_8: f32, %out: f32):
              %32 = arith.mulf %in_7, %in_8 : f32
              %33 = arith.addf %in, %32 : f32
              linalg.yield %33 : f32
            }
            linalg.copy ins(%19 : memref<1x64xf32>) outs(%18 : memref<1x64xf32>)
          }
          linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%23, %17 : memref<1x64x128xf32>, memref<1x64xf32>) outs(%22 : memref<1x64x128xf32>) {
          ^bb0(%in: f32, %in_5: f32, %out: f32):
            %28 = arith.divf %in, %in_5 : f32
            linalg.yield %28 : f32
          }
          %reinterpret_cast_4 = memref.reinterpret_cast %arg3 to offset: [%27], sizes: [1, 64, 128], strides: [524288, 128, 1] : memref<32x4096x128xf32> to memref<1x64x128xf32, strided<[524288, 128, 1], offset: ?>>
          loom.copy %22, %reinterpret_cast_4 src_mem_space @L1 dst_mem_space @DRAM, interconnect : [], broadcast : [1, 1] : memref<1x64x128xf32>, memref<1x64x128xf32, strided<[524288, 128, 1], offset: ?>>
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
          %15 = loom.alloc [1, 64, 128] on @L1 : memref<1x64x128xf32>
          %16 = loom.alloc [1, 128, 128] on @L1 : memref<1x128x128xf32>
          %17 = loom.alloc [1, 64] on @L1 : memref<1x64xf32>
          %18 = loom.alloc [1, 64] on @L1 : memref<1x64xf32>
          %19 = loom.alloc [1, 64] on @L1 : memref<1x64xf32>
          %20 = loom.alloc [1, 64] on @L1 : memref<1x64xf32>
          %21 = loom.alloc [1, 64] on @L1 : memref<1x64xf32>
          %22 = loom.alloc [1, 64, 128] on @L1 : memref<1x64x128xf32>
          %23 = loom.alloc [1, 64, 128] on @L1 : memref<1x64x128xf32>
          %24 = loom.alloc [1, 64, 128] on @L1 : memref<1x64x128xf32>
          %25 = arith.muli %arg6, %c524288 overflow<nsw> : index
          %26 = arith.muli %13, %c8192 : index
          %27 = arith.addi %25, %26 : index
          %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%27], sizes: [1, 64, 128], strides: [524288, 128, 1] : memref<32x4096x128xf32> to memref<1x64x128xf32, strided<[524288, 128, 1], offset: ?>>
          loom.copy %reinterpret_cast, %22 src_mem_space @DRAM dst_mem_space @L1, interconnect : [], broadcast : [1, 1] : memref<1x64x128xf32, strided<[524288, 128, 1], offset: ?>>, memref<1x64x128xf32>
          linalg.fill ins(%cst_1 : f32) outs(%23 : memref<1x64x128xf32>)
          linalg.fill ins(%cst_2 : f32) outs(%17 : memref<1x64xf32>)
          linalg.fill ins(%cst_3 : f32) outs(%18 : memref<1x64xf32>)
          scf.for %arg7 = %c0 to %c32 step %c1 {
            %28 = arith.muli %arg7, %c128 : index
            %29 = arith.addi %25, %28 : index
            %reinterpret_cast_5 = memref.reinterpret_cast %arg0 to offset: [%29], sizes: [1, 128, 128], strides: [524288, 4096, 1] : memref<32x128x4096xf32> to memref<1x128x128xf32, strided<[524288, 4096, 1], offset: ?>>
            loom.copy %reinterpret_cast_5, %16 src_mem_space @DRAM dst_mem_space @L1, interconnect : [], broadcast : [1, 1] : memref<1x128x128xf32, strided<[524288, 4096, 1], offset: ?>>, memref<1x128x128xf32>
            linalg.fill ins(%cst_1 : f32) outs(%15 : memref<1x64x128xf32>)
            linalg.batch_matmul ins(%22, %16 : memref<1x64x128xf32>, memref<1x128x128xf32>) outs(%15 : memref<1x64x128xf32>)
            linalg.fill ins(%cst_3 : f32) outs(%19 : memref<1x64xf32>)
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%15 : memref<1x64x128xf32>) outs(%19 : memref<1x64xf32>) {
            ^bb0(%in: f32, %out: f32):
              %32 = arith.maximumf %in, %out : f32
              linalg.yield %32 : f32
            }
            linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%18, %19 : memref<1x64xf32>, memref<1x64xf32>) outs(%19 : memref<1x64xf32>) {
            ^bb0(%in: f32, %in_7: f32, %out: f32):
              %32 = arith.truncf %cst_0 : f64 to f32
              %33 = arith.mulf %in_7, %32 : f32
              %34 = arith.cmpf ogt, %in, %33 : f32
              %35 = arith.select %34, %in, %33 : f32
              linalg.yield %35 : f32
            }
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%15, %19 : memref<1x64x128xf32>, memref<1x64xf32>) outs(%15 : memref<1x64x128xf32>) {
            ^bb0(%in: f32, %in_7: f32, %out: f32):
              %32 = arith.truncf %cst_0 : f64 to f32
              %33 = arith.mulf %in, %32 : f32
              %34 = arith.subf %33, %in_7 : f32
              %35 = math.powf %cst, %34 : f32
              linalg.yield %35 : f32
            }
            linalg.fill ins(%cst_1 : f32) outs(%20 : memref<1x64xf32>)
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%15 : memref<1x64x128xf32>) outs(%20 : memref<1x64xf32>) {
            ^bb0(%in: f32, %out: f32):
              %32 = arith.addf %in, %out : f32
              linalg.yield %32 : f32
            }
            linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%18, %19 : memref<1x64xf32>, memref<1x64xf32>) outs(%21 : memref<1x64xf32>) {
            ^bb0(%in: f32, %in_7: f32, %out: f32):
              %32 = arith.subf %in, %in_7 : f32
              %33 = math.powf %cst, %32 : f32
              linalg.yield %33 : f32
            }
            linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%17, %21, %20 : memref<1x64xf32>, memref<1x64xf32>, memref<1x64xf32>) outs(%17 : memref<1x64xf32>) {
            ^bb0(%in: f32, %in_7: f32, %in_8: f32, %out: f32):
              %32 = arith.mulf %in, %in_7 : f32
              %33 = arith.addf %32, %in_8 : f32
              linalg.yield %33 : f32
            }
            %30 = arith.muli %arg7, %c16384 : index
            %31 = arith.addi %25, %30 : index
            %reinterpret_cast_6 = memref.reinterpret_cast %arg1 to offset: [%31], sizes: [1, 128, 128], strides: [524288, 128, 1] : memref<32x4096x128xf32> to memref<1x128x128xf32, strided<[524288, 128, 1], offset: ?>>
            loom.copy %reinterpret_cast_6, %14 src_mem_space @DRAM dst_mem_space @L1, interconnect : [@horizontal_links, @vertical_links], broadcast : [8, 8] : memref<1x128x128xf32, strided<[524288, 128, 1], offset: ?>>, memref<1x128x128xf32>
            linalg.fill ins(%cst_1 : f32) outs(%24 : memref<1x64x128xf32>)
            linalg.batch_matmul ins(%15, %14 : memref<1x64x128xf32>, memref<1x128x128xf32>) outs(%24 : memref<1x64x128xf32>)
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%24, %23, %21 : memref<1x64x128xf32>, memref<1x64x128xf32>, memref<1x64xf32>) outs(%23 : memref<1x64x128xf32>) {
            ^bb0(%in: f32, %in_7: f32, %in_8: f32, %out: f32):
              %32 = arith.mulf %in_7, %in_8 : f32
              %33 = arith.addf %in, %32 : f32
              linalg.yield %33 : f32
            }
            linalg.copy ins(%19 : memref<1x64xf32>) outs(%18 : memref<1x64xf32>)
          }
          linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%23, %17 : memref<1x64x128xf32>, memref<1x64xf32>) outs(%22 : memref<1x64x128xf32>) {
          ^bb0(%in: f32, %in_5: f32, %out: f32):
            %28 = arith.divf %in, %in_5 : f32
            linalg.yield %28 : f32
          }
          %reinterpret_cast_4 = memref.reinterpret_cast %arg3 to offset: [%27], sizes: [1, 64, 128], strides: [524288, 128, 1] : memref<32x4096x128xf32> to memref<1x64x128xf32, strided<[524288, 128, 1], offset: ?>>
          loom.copy %22, %reinterpret_cast_4 src_mem_space @L1 dst_mem_space @DRAM, interconnect : [], broadcast : [1, 1] : memref<1x64x128xf32>, memref<1x64x128xf32, strided<[524288, 128, 1], offset: ?>>
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
          %15 = loom.alloc [1, 64, 128] on @L1 : memref<1x64x128xf32>
          %16 = loom.alloc [1, 128, 128] on @L1 : memref<1x128x128xf32>
          %17 = loom.alloc [1, 64] on @L1 : memref<1x64xf32>
          %18 = loom.alloc [1, 64] on @L1 : memref<1x64xf32>
          %19 = loom.alloc [1, 64] on @L1 : memref<1x64xf32>
          %20 = loom.alloc [1, 64] on @L1 : memref<1x64xf32>
          %21 = loom.alloc [1, 64] on @L1 : memref<1x64xf32>
          %22 = loom.alloc [1, 64, 128] on @L1 : memref<1x64x128xf32>
          %23 = loom.alloc [1, 64, 128] on @L1 : memref<1x64x128xf32>
          %24 = loom.alloc [1, 64, 128] on @L1 : memref<1x64x128xf32>
          %25 = arith.muli %arg6, %c524288 overflow<nsw> : index
          %26 = arith.muli %13, %c8192 : index
          %27 = arith.addi %25, %26 : index
          %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%27], sizes: [1, 64, 128], strides: [524288, 128, 1] : memref<32x4096x128xf32> to memref<1x64x128xf32, strided<[524288, 128, 1], offset: ?>>
          loom.copy %reinterpret_cast, %22 src_mem_space @DRAM dst_mem_space @L1, interconnect : [], broadcast : [1, 1] : memref<1x64x128xf32, strided<[524288, 128, 1], offset: ?>>, memref<1x64x128xf32>
          linalg.fill ins(%cst_1 : f32) outs(%23 : memref<1x64x128xf32>)
          linalg.fill ins(%cst_2 : f32) outs(%17 : memref<1x64xf32>)
          linalg.fill ins(%cst_3 : f32) outs(%18 : memref<1x64xf32>)
          scf.for %arg7 = %c0 to %c32 step %c1 {
            %28 = arith.muli %arg7, %c128 : index
            %29 = arith.addi %25, %28 : index
            %reinterpret_cast_5 = memref.reinterpret_cast %arg0 to offset: [%29], sizes: [1, 128, 128], strides: [524288, 4096, 1] : memref<32x128x4096xf32> to memref<1x128x128xf32, strided<[524288, 4096, 1], offset: ?>>
            loom.copy %reinterpret_cast_5, %16 src_mem_space @DRAM dst_mem_space @L1, interconnect : [], broadcast : [1, 1] : memref<1x128x128xf32, strided<[524288, 4096, 1], offset: ?>>, memref<1x128x128xf32>
            linalg.fill ins(%cst_1 : f32) outs(%15 : memref<1x64x128xf32>)
            linalg.batch_matmul ins(%22, %16 : memref<1x64x128xf32>, memref<1x128x128xf32>) outs(%15 : memref<1x64x128xf32>)
            linalg.fill ins(%cst_3 : f32) outs(%19 : memref<1x64xf32>)
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%15 : memref<1x64x128xf32>) outs(%19 : memref<1x64xf32>) {
            ^bb0(%in: f32, %out: f32):
              %32 = arith.maximumf %in, %out : f32
              linalg.yield %32 : f32
            }
            linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%18, %19 : memref<1x64xf32>, memref<1x64xf32>) outs(%19 : memref<1x64xf32>) {
            ^bb0(%in: f32, %in_7: f32, %out: f32):
              %32 = arith.truncf %cst_0 : f64 to f32
              %33 = arith.mulf %in_7, %32 : f32
              %34 = arith.cmpf ogt, %in, %33 : f32
              %35 = arith.select %34, %in, %33 : f32
              linalg.yield %35 : f32
            }
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%15, %19 : memref<1x64x128xf32>, memref<1x64xf32>) outs(%15 : memref<1x64x128xf32>) {
            ^bb0(%in: f32, %in_7: f32, %out: f32):
              %32 = arith.truncf %cst_0 : f64 to f32
              %33 = arith.mulf %in, %32 : f32
              %34 = arith.subf %33, %in_7 : f32
              %35 = math.powf %cst, %34 : f32
              linalg.yield %35 : f32
            }
            linalg.fill ins(%cst_1 : f32) outs(%20 : memref<1x64xf32>)
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%15 : memref<1x64x128xf32>) outs(%20 : memref<1x64xf32>) {
            ^bb0(%in: f32, %out: f32):
              %32 = arith.addf %in, %out : f32
              linalg.yield %32 : f32
            }
            linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%18, %19 : memref<1x64xf32>, memref<1x64xf32>) outs(%21 : memref<1x64xf32>) {
            ^bb0(%in: f32, %in_7: f32, %out: f32):
              %32 = arith.subf %in, %in_7 : f32
              %33 = math.powf %cst, %32 : f32
              linalg.yield %33 : f32
            }
            linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%17, %21, %20 : memref<1x64xf32>, memref<1x64xf32>, memref<1x64xf32>) outs(%17 : memref<1x64xf32>) {
            ^bb0(%in: f32, %in_7: f32, %in_8: f32, %out: f32):
              %32 = arith.mulf %in, %in_7 : f32
              %33 = arith.addf %32, %in_8 : f32
              linalg.yield %33 : f32
            }
            %30 = arith.muli %arg7, %c16384 : index
            %31 = arith.addi %25, %30 : index
            %reinterpret_cast_6 = memref.reinterpret_cast %arg1 to offset: [%31], sizes: [1, 128, 128], strides: [524288, 128, 1] : memref<32x4096x128xf32> to memref<1x128x128xf32, strided<[524288, 128, 1], offset: ?>>
            loom.copy %reinterpret_cast_6, %14 src_mem_space @DRAM dst_mem_space @L1, interconnect : [@horizontal_links], broadcast : [1, 8] : memref<1x128x128xf32, strided<[524288, 128, 1], offset: ?>>, memref<1x128x128xf32>
            linalg.fill ins(%cst_1 : f32) outs(%24 : memref<1x64x128xf32>)
            linalg.batch_matmul ins(%15, %14 : memref<1x64x128xf32>, memref<1x128x128xf32>) outs(%24 : memref<1x64x128xf32>)
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%24, %23, %21 : memref<1x64x128xf32>, memref<1x64x128xf32>, memref<1x64xf32>) outs(%23 : memref<1x64x128xf32>) {
            ^bb0(%in: f32, %in_7: f32, %in_8: f32, %out: f32):
              %32 = arith.mulf %in_7, %in_8 : f32
              %33 = arith.addf %in, %32 : f32
              linalg.yield %33 : f32
            }
            linalg.copy ins(%19 : memref<1x64xf32>) outs(%18 : memref<1x64xf32>)
          }
          linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%23, %17 : memref<1x64x128xf32>, memref<1x64xf32>) outs(%22 : memref<1x64x128xf32>) {
          ^bb0(%in: f32, %in_5: f32, %out: f32):
            %28 = arith.divf %in, %in_5 : f32
            linalg.yield %28 : f32
          }
          %reinterpret_cast_4 = memref.reinterpret_cast %arg3 to offset: [%27], sizes: [1, 64, 128], strides: [524288, 128, 1] : memref<32x4096x128xf32> to memref<1x64x128xf32, strided<[524288, 128, 1], offset: ?>>
          loom.copy %22, %reinterpret_cast_4 src_mem_space @L1 dst_mem_space @DRAM, interconnect : [], broadcast : [1, 1] : memref<1x64x128xf32>, memref<1x64x128xf32, strided<[524288, 128, 1], offset: ?>>
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
          %15 = loom.alloc [1, 64, 128] on @L1 : memref<1x64x128xf32>
          %16 = loom.alloc [1, 128, 128] on @L1 : memref<1x128x128xf32>
          %17 = loom.alloc [1, 64] on @L1 : memref<1x64xf32>
          %18 = loom.alloc [1, 64] on @L1 : memref<1x64xf32>
          %19 = loom.alloc [1, 64] on @L1 : memref<1x64xf32>
          %20 = loom.alloc [1, 64] on @L1 : memref<1x64xf32>
          %21 = loom.alloc [1, 64] on @L1 : memref<1x64xf32>
          %22 = loom.alloc [1, 64, 128] on @L1 : memref<1x64x128xf32>
          %23 = loom.alloc [1, 64, 128] on @L1 : memref<1x64x128xf32>
          %24 = loom.alloc [1, 64, 128] on @L1 : memref<1x64x128xf32>
          %25 = arith.muli %arg6, %c524288 overflow<nsw> : index
          %26 = arith.muli %13, %c8192 : index
          %27 = arith.addi %25, %26 : index
          %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%27], sizes: [1, 64, 128], strides: [524288, 128, 1] : memref<32x4096x128xf32> to memref<1x64x128xf32, strided<[524288, 128, 1], offset: ?>>
          loom.copy %reinterpret_cast, %22 src_mem_space @DRAM dst_mem_space @L1, interconnect : [], broadcast : [1, 1] : memref<1x64x128xf32, strided<[524288, 128, 1], offset: ?>>, memref<1x64x128xf32>
          linalg.fill ins(%cst_1 : f32) outs(%23 : memref<1x64x128xf32>)
          linalg.fill ins(%cst_2 : f32) outs(%17 : memref<1x64xf32>)
          linalg.fill ins(%cst_3 : f32) outs(%18 : memref<1x64xf32>)
          scf.for %arg7 = %c0 to %c32 step %c1 {
            %28 = arith.muli %arg7, %c128 : index
            %29 = arith.addi %25, %28 : index
            %reinterpret_cast_5 = memref.reinterpret_cast %arg0 to offset: [%29], sizes: [1, 128, 128], strides: [524288, 4096, 1] : memref<32x128x4096xf32> to memref<1x128x128xf32, strided<[524288, 4096, 1], offset: ?>>
            loom.copy %reinterpret_cast_5, %16 src_mem_space @DRAM dst_mem_space @L1, interconnect : [], broadcast : [1, 1] : memref<1x128x128xf32, strided<[524288, 4096, 1], offset: ?>>, memref<1x128x128xf32>
            linalg.fill ins(%cst_1 : f32) outs(%15 : memref<1x64x128xf32>)
            linalg.batch_matmul ins(%22, %16 : memref<1x64x128xf32>, memref<1x128x128xf32>) outs(%15 : memref<1x64x128xf32>)
            linalg.fill ins(%cst_3 : f32) outs(%19 : memref<1x64xf32>)
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%15 : memref<1x64x128xf32>) outs(%19 : memref<1x64xf32>) {
            ^bb0(%in: f32, %out: f32):
              %32 = arith.maximumf %in, %out : f32
              linalg.yield %32 : f32
            }
            linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%18, %19 : memref<1x64xf32>, memref<1x64xf32>) outs(%19 : memref<1x64xf32>) {
            ^bb0(%in: f32, %in_7: f32, %out: f32):
              %32 = arith.truncf %cst_0 : f64 to f32
              %33 = arith.mulf %in_7, %32 : f32
              %34 = arith.cmpf ogt, %in, %33 : f32
              %35 = arith.select %34, %in, %33 : f32
              linalg.yield %35 : f32
            }
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%15, %19 : memref<1x64x128xf32>, memref<1x64xf32>) outs(%15 : memref<1x64x128xf32>) {
            ^bb0(%in: f32, %in_7: f32, %out: f32):
              %32 = arith.truncf %cst_0 : f64 to f32
              %33 = arith.mulf %in, %32 : f32
              %34 = arith.subf %33, %in_7 : f32
              %35 = math.powf %cst, %34 : f32
              linalg.yield %35 : f32
            }
            linalg.fill ins(%cst_1 : f32) outs(%20 : memref<1x64xf32>)
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%15 : memref<1x64x128xf32>) outs(%20 : memref<1x64xf32>) {
            ^bb0(%in: f32, %out: f32):
              %32 = arith.addf %in, %out : f32
              linalg.yield %32 : f32
            }
            linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%18, %19 : memref<1x64xf32>, memref<1x64xf32>) outs(%21 : memref<1x64xf32>) {
            ^bb0(%in: f32, %in_7: f32, %out: f32):
              %32 = arith.subf %in, %in_7 : f32
              %33 = math.powf %cst, %32 : f32
              linalg.yield %33 : f32
            }
            linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%17, %21, %20 : memref<1x64xf32>, memref<1x64xf32>, memref<1x64xf32>) outs(%17 : memref<1x64xf32>) {
            ^bb0(%in: f32, %in_7: f32, %in_8: f32, %out: f32):
              %32 = arith.mulf %in, %in_7 : f32
              %33 = arith.addf %32, %in_8 : f32
              linalg.yield %33 : f32
            }
            %30 = arith.muli %arg7, %c16384 : index
            %31 = arith.addi %25, %30 : index
            %reinterpret_cast_6 = memref.reinterpret_cast %arg1 to offset: [%31], sizes: [1, 128, 128], strides: [524288, 128, 1] : memref<32x4096x128xf32> to memref<1x128x128xf32, strided<[524288, 128, 1], offset: ?>>
            loom.copy %reinterpret_cast_6, %14 src_mem_space @DRAM dst_mem_space @L1, interconnect : [@vertical_links], broadcast : [8, 1] : memref<1x128x128xf32, strided<[524288, 128, 1], offset: ?>>, memref<1x128x128xf32>
            linalg.fill ins(%cst_1 : f32) outs(%24 : memref<1x64x128xf32>)
            linalg.batch_matmul ins(%15, %14 : memref<1x64x128xf32>, memref<1x128x128xf32>) outs(%24 : memref<1x64x128xf32>)
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%24, %23, %21 : memref<1x64x128xf32>, memref<1x64x128xf32>, memref<1x64xf32>) outs(%23 : memref<1x64x128xf32>) {
            ^bb0(%in: f32, %in_7: f32, %in_8: f32, %out: f32):
              %32 = arith.mulf %in_7, %in_8 : f32
              %33 = arith.addf %in, %32 : f32
              linalg.yield %33 : f32
            }
            linalg.copy ins(%19 : memref<1x64xf32>) outs(%18 : memref<1x64xf32>)
          }
          linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%23, %17 : memref<1x64x128xf32>, memref<1x64xf32>) outs(%22 : memref<1x64x128xf32>) {
          ^bb0(%in: f32, %in_5: f32, %out: f32):
            %28 = arith.divf %in, %in_5 : f32
            linalg.yield %28 : f32
          }
          %reinterpret_cast_4 = memref.reinterpret_cast %arg3 to offset: [%27], sizes: [1, 64, 128], strides: [524288, 128, 1] : memref<32x4096x128xf32> to memref<1x64x128xf32, strided<[524288, 128, 1], offset: ?>>
          loom.copy %22, %reinterpret_cast_4 src_mem_space @L1 dst_mem_space @DRAM, interconnect : [], broadcast : [1, 1] : memref<1x64x128xf32>, memref<1x64x128xf32, strided<[524288, 128, 1], offset: ?>>
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
          %15 = loom.alloc [1, 64, 128] on @L1 : memref<1x64x128xf32>
          %16 = loom.alloc [1, 128, 128] on @L1 : memref<1x128x128xf32>
          %17 = loom.alloc [1, 64] on @L1 : memref<1x64xf32>
          %18 = loom.alloc [1, 64] on @L1 : memref<1x64xf32>
          %19 = loom.alloc [1, 64] on @L1 : memref<1x64xf32>
          %20 = loom.alloc [1, 64] on @L1 : memref<1x64xf32>
          %21 = loom.alloc [1, 64] on @L1 : memref<1x64xf32>
          %22 = loom.alloc [1, 64, 128] on @L1 : memref<1x64x128xf32>
          %23 = loom.alloc [1, 64, 128] on @L1 : memref<1x64x128xf32>
          %24 = loom.alloc [1, 64, 128] on @L1 : memref<1x64x128xf32>
          %25 = arith.muli %arg6, %c524288 overflow<nsw> : index
          %26 = arith.muli %13, %c8192 : index
          %27 = arith.addi %25, %26 : index
          %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%27], sizes: [1, 64, 128], strides: [524288, 128, 1] : memref<32x4096x128xf32> to memref<1x64x128xf32, strided<[524288, 128, 1], offset: ?>>
          loom.copy %reinterpret_cast, %22 src_mem_space @DRAM dst_mem_space @L1, interconnect : [], broadcast : [1, 1] : memref<1x64x128xf32, strided<[524288, 128, 1], offset: ?>>, memref<1x64x128xf32>
          linalg.fill ins(%cst_1 : f32) outs(%23 : memref<1x64x128xf32>)
          linalg.fill ins(%cst_2 : f32) outs(%17 : memref<1x64xf32>)
          linalg.fill ins(%cst_3 : f32) outs(%18 : memref<1x64xf32>)
          scf.for %arg7 = %c0 to %c32 step %c1 {
            %28 = arith.muli %arg7, %c128 : index
            %29 = arith.addi %25, %28 : index
            %reinterpret_cast_5 = memref.reinterpret_cast %arg0 to offset: [%29], sizes: [1, 128, 128], strides: [524288, 4096, 1] : memref<32x128x4096xf32> to memref<1x128x128xf32, strided<[524288, 4096, 1], offset: ?>>
            loom.copy %reinterpret_cast_5, %16 src_mem_space @DRAM dst_mem_space @L1, interconnect : [@horizontal_links, @vertical_links], broadcast : [8, 8] : memref<1x128x128xf32, strided<[524288, 4096, 1], offset: ?>>, memref<1x128x128xf32>
            linalg.fill ins(%cst_1 : f32) outs(%15 : memref<1x64x128xf32>)
            linalg.batch_matmul ins(%22, %16 : memref<1x64x128xf32>, memref<1x128x128xf32>) outs(%15 : memref<1x64x128xf32>)
            linalg.fill ins(%cst_3 : f32) outs(%19 : memref<1x64xf32>)
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%15 : memref<1x64x128xf32>) outs(%19 : memref<1x64xf32>) {
            ^bb0(%in: f32, %out: f32):
              %32 = arith.maximumf %in, %out : f32
              linalg.yield %32 : f32
            }
            linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%18, %19 : memref<1x64xf32>, memref<1x64xf32>) outs(%19 : memref<1x64xf32>) {
            ^bb0(%in: f32, %in_7: f32, %out: f32):
              %32 = arith.truncf %cst_0 : f64 to f32
              %33 = arith.mulf %in_7, %32 : f32
              %34 = arith.cmpf ogt, %in, %33 : f32
              %35 = arith.select %34, %in, %33 : f32
              linalg.yield %35 : f32
            }
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%15, %19 : memref<1x64x128xf32>, memref<1x64xf32>) outs(%15 : memref<1x64x128xf32>) {
            ^bb0(%in: f32, %in_7: f32, %out: f32):
              %32 = arith.truncf %cst_0 : f64 to f32
              %33 = arith.mulf %in, %32 : f32
              %34 = arith.subf %33, %in_7 : f32
              %35 = math.powf %cst, %34 : f32
              linalg.yield %35 : f32
            }
            linalg.fill ins(%cst_1 : f32) outs(%20 : memref<1x64xf32>)
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%15 : memref<1x64x128xf32>) outs(%20 : memref<1x64xf32>) {
            ^bb0(%in: f32, %out: f32):
              %32 = arith.addf %in, %out : f32
              linalg.yield %32 : f32
            }
            linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%18, %19 : memref<1x64xf32>, memref<1x64xf32>) outs(%21 : memref<1x64xf32>) {
            ^bb0(%in: f32, %in_7: f32, %out: f32):
              %32 = arith.subf %in, %in_7 : f32
              %33 = math.powf %cst, %32 : f32
              linalg.yield %33 : f32
            }
            linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%17, %21, %20 : memref<1x64xf32>, memref<1x64xf32>, memref<1x64xf32>) outs(%17 : memref<1x64xf32>) {
            ^bb0(%in: f32, %in_7: f32, %in_8: f32, %out: f32):
              %32 = arith.mulf %in, %in_7 : f32
              %33 = arith.addf %32, %in_8 : f32
              linalg.yield %33 : f32
            }
            %30 = arith.muli %arg7, %c16384 : index
            %31 = arith.addi %25, %30 : index
            %reinterpret_cast_6 = memref.reinterpret_cast %arg1 to offset: [%31], sizes: [1, 128, 128], strides: [524288, 128, 1] : memref<32x4096x128xf32> to memref<1x128x128xf32, strided<[524288, 128, 1], offset: ?>>
            loom.copy %reinterpret_cast_6, %14 src_mem_space @DRAM dst_mem_space @L1, interconnect : [], broadcast : [1, 1] : memref<1x128x128xf32, strided<[524288, 128, 1], offset: ?>>, memref<1x128x128xf32>
            linalg.fill ins(%cst_1 : f32) outs(%24 : memref<1x64x128xf32>)
            linalg.batch_matmul ins(%15, %14 : memref<1x64x128xf32>, memref<1x128x128xf32>) outs(%24 : memref<1x64x128xf32>)
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%24, %23, %21 : memref<1x64x128xf32>, memref<1x64x128xf32>, memref<1x64xf32>) outs(%23 : memref<1x64x128xf32>) {
            ^bb0(%in: f32, %in_7: f32, %in_8: f32, %out: f32):
              %32 = arith.mulf %in_7, %in_8 : f32
              %33 = arith.addf %in, %32 : f32
              linalg.yield %33 : f32
            }
            linalg.copy ins(%19 : memref<1x64xf32>) outs(%18 : memref<1x64xf32>)
          }
          linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%23, %17 : memref<1x64x128xf32>, memref<1x64xf32>) outs(%22 : memref<1x64x128xf32>) {
          ^bb0(%in: f32, %in_5: f32, %out: f32):
            %28 = arith.divf %in, %in_5 : f32
            linalg.yield %28 : f32
          }
          %reinterpret_cast_4 = memref.reinterpret_cast %arg3 to offset: [%27], sizes: [1, 64, 128], strides: [524288, 128, 1] : memref<32x4096x128xf32> to memref<1x64x128xf32, strided<[524288, 128, 1], offset: ?>>
          loom.copy %22, %reinterpret_cast_4 src_mem_space @L1 dst_mem_space @DRAM, interconnect : [], broadcast : [1, 1] : memref<1x64x128xf32>, memref<1x64x128xf32, strided<[524288, 128, 1], offset: ?>>
        }
        scf.reduce 
      } {loom.iter_types = [#loom.iter_type<spatial>, #loom.iter_type<spatial>], loom.mapped_to_dims = [@x, @y]}
      return
    }
  }
  // =====================================================================================
  // Variant: attention__d0i1_d1i1__f01__d_a_a__BB1__BM64__BN128
  //
  //   Naming breakdown:
  //     d0i1_d1i1 : spatial dim 0 (x) → interconnect idx 1 (horizontal_links),
  //                 spatial dim 1 (y) → interconnect idx 1 (vertical_links)
  //     f01       : function arg ordering — arg0=K, arg1=V, arg2=Q (original order)
  //     d_a_a     : copy strategy per operand — K^T=direct(d), V=all-broadcast(a), Q=... (see below)
  //                 Q copy uses broadcast:[1,1] (direct, no interconnect).
  //                 K^T copy uses [@horizontal_links, @vertical_links], broadcast:[8,8] (all-broadcast).
  //                 V copy uses [@horizontal_links, @vertical_links], broadcast:[8,8] (all-broadcast).
  //     BB=1, BM=64, BN=128 : tile sizes
  //
  //   Global tensor shapes (after per-core tiling by 8x8 mesh):
  //     K  (arg0): memref<32x128x4096xf32>  — [batch_total, head_dim, N_total]
  //     V  (arg1): memref<32x4096x128xf32>  — [batch_total, N_total, head_dim]
  //     Q  (arg2): memref<32x4096x128xf32>  — [batch_total, M_total, head_dim]
  //     O  (arg3): memref<32x4096x128xf32>  — [batch_total, M_total, head_dim]
  //   where BB=1 (batch tile is 1), BM=64, BN=128.
  //         M_total=4096, N_total=4096, head_dim=128.
  //
  //   Loop structure:
  //     Outer: scf.parallel (%arg4, %arg5) in [0,8) x [0,8) — 8x8 spatial cores
  //     Middle: scf.for %arg6 in [0,32) — 32 iterations over batch items
  //       Each iteration processes one batch item (BB=1).
  //       The 64 spatial cores perfectly tile the M_total=4096 dimension: 64 cores * 64 (BM) = 4096.
  //       Core index %13 = %arg4*8 + %arg5 defines the M-tile index [0, 63].
  //     Inner: scf.for %arg7 in [0,32) — 32 iterations over N-dimension tiles
  //       (N_total / BN = 4096 / 128 = 32 iterations)
  //
  //   Interconnect strategy:
  //     Q copy:    direct (no broadcast), each core loads its own Q tile from DRAM
  //     K^T copy:  all-broadcast via horizontal_links + vertical_links, broadcast [8,8]
  //                → one core loads K^T tile, broadcasts to all 64 cores in mesh
  //     V copy:    all-broadcast via horizontal_links + vertical_links, broadcast [8,8]
  //                → one core loads V tile, broadcasts to all 64 cores in mesh
  //
  //   Same structural improvements as other variants: 11 allocs (down from 13),
  //   fills sunk to consumers, Q tile buffer reused for post-loop output.
  // =====================================================================================
  module attributes {loom.pass_name = "Materialize"} {
    func.func @attention__d0i1_d1i1__f01__d_a_a__BB1__BM64__BN128(%arg0: memref<32x128x4096xf32>, %arg1: memref<32x4096x128xf32>, %arg2: memref<32x4096x128xf32>, %arg3: memref<32x4096x128xf32>) {
      // arg0 = K (Key matrix, transposed view), shape [batch_total=32, head_dim=128, N_total=4096]
      // arg1 = V (Value matrix),               shape [batch_total=32, N_total=4096, head_dim=128]
      // arg2 = Q (Query matrix),               shape [batch_total=32, M_total=4096, head_dim=128]
      // arg3 = O (Output matrix),              shape [batch_total=32, M_total=4096, head_dim=128]
      %c16384 = arith.constant 16384 : index     // BN * head_dim = 128 * 128
      %c8192 = arith.constant 8192 : index       // BM * head_dim = 64 * 128 = 8192
      %c524288 = arith.constant 524288 : index   // M_total * head_dim = 4096 * 128 = 524288
      %c32 = arith.constant 32 : index           // batch size = 32, also inner loop bound (4096 / 128 = 32)
      %cst = arith.constant 2.000000e+00 : f32   // base for exp2
      %cst_0 = arith.constant 0.12751743074602467 : f64 // qk_scale = 1/sqrt(head_dim) ≈ 1/sqrt(128)
      %cst_1 = arith.constant 0.000000e+00 : f32 // zero (for fill)
      %cst_2 = arith.constant 1.000000e+00 : f32 // one (for l_i init)
      %cst_3 = arith.constant 0xFF800000 : f32   // -inf (for m_i init and amax init)
      %c128 = arith.constant 128 : index         // BN = 128
      %c0 = arith.constant 0 : index
      %c8 = arith.constant 8 : index             // spatial mesh dimension (8x8)
      %c1 = arith.constant 1 : index
      // =================== Spatial Parallel: 8x8 core mesh ===================
      // %arg4 = x-dim core index [0,8), %arg5 = y-dim core index [0,8)
      scf.parallel (%arg4, %arg5) = (%c0, %c0) to (%c8, %c8) step (%c1, %c1) {
        // =================== Middle Loop: iterate over batch items ===================
        // %arg6 ∈ [0, 32): each core processes all 32 batch items
        scf.for %arg6 = %c0 to %c32 step %c1 {
          // ---- Compute linearized core index (acts as M-tile index) ----
          %12 = arith.muli %arg4, %c8 overflow<nsw> : index                  // (core_x contributes row offset)
          // Since M_total / BM = 4096 / 64 = 64, the 64 cores perfectly map to the 64 M-tiles.
          %13 = arith.addi %12, %arg5 : index          // (linearized core index within 8x8 mesh, range [0, 63])
          // =================== L1 Memory Allocations (11 buffers) ===================
          // Same alloc count as other variants after SinkFillOps optimization.
          //
          // --- 3D buffers ---
          %14 = loom.alloc [1, 128, 128] on @L1 : memref<1x128x128xf32>    // V tile [BB, BN, head_dim] (loaded from DRAM each inner iter)
          %15 = loom.alloc [1, 64, 128] on @L1 : memref<1x64x128xf32>      // qk = Q @ K^T [BB, BM, BN], later reused in-place for p
          %16 = loom.alloc [1, 128, 128] on @L1 : memref<1x128x128xf32>    // K^T tile [BB, head_dim, BN] (loaded from DRAM each inner iter)
          // --- 2D buffers (BB x BM = 1 x 64) ---
          %17 = loom.alloc [1, 64] on @L1 : memref<1x64xf32>               // l_i: running softmax denominator (in-place across iters)
          %18 = loom.alloc [1, 64] on @L1 : memref<1x64xf32>               // m_i: running row-wise max (updated via copy at end of iter)
          %19 = loom.alloc [1, 64] on @L1 : memref<1x64xf32>               // m_ij: new row-wise max for current iter (also holds intermediate amax)
          %20 = loom.alloc [1, 64] on @L1 : memref<1x64xf32>               // l_ij: row-wise sum of p for current iter
          %21 = loom.alloc [1, 64] on @L1 : memref<1x64xf32>               // alpha = 2^(m_i - m_ij), rescaling factor
          // --- 3D buffers (reused / accumulator) ---
          %22 = loom.alloc [1, 64, 128] on @L1 : memref<1x64x128xf32>      // Q tile [BB, BM, head_dim] (loaded once); REUSED post-loop for output = acc / l_i
          %23 = loom.alloc [1, 64, 128] on @L1 : memref<1x64x128xf32>      // acc (accumulator, updated in-place across iters)
          %24 = loom.alloc [1, 64, 128] on @L1 : memref<1x64x128xf32>      // p @ V intermediate result (per-iter scratch)
          // =================== Load Q tile from DRAM to L1 ===================
          // Q_tile = Q[batch_idx:batch_idx+1, m_start:m_end, :]
          // where batch_idx = %arg6, m_start = %13 * BM
          //
          // Address calculation for Q (arg2, shape [32, 4096, 128]):
          //   %25 = %arg6 * 524288              (batch offset = batch_idx * M_total * head_dim)
          //   %26 = %13 * 8192                  (M-tile offset = m_tile_idx * BM * head_dim = %13 * 64 * 128)
          //   %27 = %25 + %26                   (Total offset for Q tile)
          //   The view: offset=%27, sizes=[1, 64, 128], strides=[524288, 128, 1]
          %25 = arith.muli %arg6, %c524288 overflow<nsw> : index
          %26 = arith.muli %13, %c8192 : index
          %27 = arith.addi %25, %26 : index
          %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%27], sizes: [1, 64, 128], strides: [524288, 128, 1] : memref<32x4096x128xf32> to memref<1x64x128xf32, strided<[524288, 128, 1], offset: ?>>
          // DMA: DRAM → L1, load Q tile into %22 (direct, no broadcast)
          loom.copy %reinterpret_cast, %22 src_mem_space @DRAM dst_mem_space @L1, interconnect : [], broadcast : [1, 1] : memref<1x64x128xf32, strided<[524288, 128, 1], offset: ?>>, memref<1x64x128xf32>
          // =================== Loop Initialization ===================
          // acc = zeros([BB, BM, head_dim])
          // Memory: writes to %23, out-of-place (fresh init)
          linalg.fill ins(%cst_1 : f32) outs(%23 : memref<1x64x128xf32>)
          // l_i = full([BB, BM], 1.0)
          // Memory: writes to %17, out-of-place (fresh init)
          linalg.fill ins(%cst_2 : f32) outs(%17 : memref<1x64xf32>)
          // m_i = full([BB, BM], -inf)
          // Memory: writes to %18, out-of-place (fresh init)
          linalg.fill ins(%cst_3 : f32) outs(%18 : memref<1x64xf32>)
          // =================== Inner Loop: iterate over K/V tiles (32 iters) ===================
          // for n_iter in range(N_total / BN):  (4096 / 128 = 32 iterations)
          scf.for %arg7 = %c0 to %c32 step %c1 {
            // ---- Load K^T tile from DRAM ----
            // K_tile = K[batch_idx:batch_idx+1, :, n_start:n_end]  (transposed view for matmul)
            // where batch_idx = %arg6, n_start = %arg7 * BN
            //
            // Address calculation for K^T (arg0, shape [32, 128, 4096]):
            //   %28 = %arg7 * 128                 (N-tile offset = %arg7 * BN)
            //   %29 = %25 + %28                   (Total offset = batch_offset + N-tile offset)
            //   The view: offset=%29, sizes=[1, 128, 128], strides=[524288, 4096, 1]
            %28 = arith.muli %arg7, %c128 : index
            %29 = arith.addi %25, %28 : index
            %reinterpret_cast_5 = memref.reinterpret_cast %arg0 to offset: [%29], sizes: [1, 128, 128], strides: [524288, 4096, 1] : memref<32x128x4096xf32> to memref<1x128x128xf32, strided<[524288, 4096, 1], offset: ?>>
            // DMA: DRAM → L1, load K^T tile into %16
            // Interconnect: ALL-BROADCAST via horizontal_links + vertical_links, broadcast [8,8]
            // One core loads and the data is replicated to all 8*8=64 cores.
            loom.copy %reinterpret_cast_5, %16 src_mem_space @DRAM dst_mem_space @L1, interconnect : [@horizontal_links, @vertical_links], broadcast : [8, 8] : memref<1x128x128xf32, strided<[524288, 4096, 1], offset: ?>>, memref<1x128x128xf32>
            // ---- qk = Q @ K^T ----
            // Zero-init qk buffer, then compute batch matmul
            // Memory: %15 is zeroed then written by batch_matmul (out-of-place, accumulation pattern)
            // NOTE: fill sunk from pre-loop to here by SinkFillOps pass
            linalg.fill ins(%cst_1 : f32) outs(%15 : memref<1x64x128xf32>)
            // qk = Q_tile @ K_tile^T, result in %15
            // shapes: [1, 64, 128] @ [1, 128, 128] → [1, 64, 128]
            linalg.batch_matmul ins(%22, %16 : memref<1x64x128xf32>, memref<1x128x128xf32>) outs(%15 : memref<1x64x128xf32>)
            // ---- amax(qk, -1): row-wise max of qk ----
            // Memory: %19 is first filled with -inf, then reduced over (out-of-place init + in-place reduction)
            // NOTE: fill sunk from pre-loop to here by SinkFillOps pass
            linalg.fill ins(%cst_3 : f32) outs(%19 : memref<1x64xf32>)
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%15 : memref<1x64x128xf32>) outs(%19 : memref<1x64xf32>) {
            ^bb0(%in: f32, %out: f32):
              %32 = arith.maximumf %in, %out : f32
              linalg.yield %32 : f32
            }
            // After this: %19 = amax(qk, dim=-1), shape [1, 64]

            // ---- m_ij = max(m_i, amax(qk) * qk_scale) ----
            // Reads m_i from %18, reads amax from %19, writes result back to %19
            // Memory: IN-PLACE on %19 — the amax result is overwritten with m_ij.
            //         This is safe because amax is only needed to compute m_ij itself.
            linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%18, %19 : memref<1x64xf32>, memref<1x64xf32>) outs(%19 : memref<1x64xf32>) {
            ^bb0(%in: f32, %in_7: f32, %out: f32):
              %32 = arith.truncf %cst_0 : f64 to f32
              %33 = arith.mulf %in_7, %32 : f32
              %34 = arith.cmpf ogt, %in, %33 : f32
              %35 = arith.select %34, %in, %33 : f32
              linalg.yield %35 : f32
            }
            // After this: %19 = m_ij = max(m_i, amax(qk) * qk_scale)

            // ---- p = exp2(qk * qk_scale - m_ij) ----
            // Reads qk from %15, reads m_ij from %19, writes p back to %15
            // Memory: IN-PLACE on %15 — qk is overwritten with p.
            //         qk is no longer needed after this point (consumed by amax above).
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%15, %19 : memref<1x64x128xf32>, memref<1x64xf32>) outs(%15 : memref<1x64x128xf32>) {
            ^bb0(%in: f32, %in_7: f32, %out: f32):
              %32 = arith.truncf %cst_0 : f64 to f32
              %33 = arith.mulf %in, %32 : f32
              %34 = arith.subf %33, %in_7 : f32
              %35 = math.powf %cst, %34 : f32
              linalg.yield %35 : f32
            }
            // After this: %15 = p (attention weights), shape [1, 64, 128]

            // ---- l_ij = sum(p, dim=-1) ----
            // Memory: %20 is zeroed then reduced in-place
            // NOTE: fill sunk from pre-loop to here by SinkFillOps pass
            linalg.fill ins(%cst_1 : f32) outs(%20 : memref<1x64xf32>)
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%15 : memref<1x64x128xf32>) outs(%20 : memref<1x64xf32>) {
            ^bb0(%in: f32, %out: f32):
              %32 = arith.addf %in, %out : f32
              linalg.yield %32 : f32
            }
            // After this: %20 = l_ij = sum(p, dim=-1), shape [1, 64]

            // ---- alpha = 2^(m_i - m_ij) ----
            // Reads m_i from %18, reads m_ij from %19, writes alpha to %21
            // Memory: OUT-OF-PLACE, result in separate buffer %21
            linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%18, %19 : memref<1x64xf32>, memref<1x64xf32>) outs(%21 : memref<1x64xf32>) {
            ^bb0(%in: f32, %in_7: f32, %out: f32):
              %32 = arith.subf %in, %in_7 : f32
              %33 = math.powf %cst, %32 : f32
              linalg.yield %33 : f32
            }
            // After this: %21 = alpha = 2^(m_i - m_ij), shape [1, 64]

            // ---- l_i = alpha * l_i + l_ij ----
            // Reads l_i from %17, alpha from %21, l_ij from %20, writes back to %17
            // Memory: IN-PLACE on %17 — the running l_i is updated in its own buffer.
            //         This is the key in-place update for the online softmax denominator.
            linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%17, %21, %20 : memref<1x64xf32>, memref<1x64xf32>, memref<1x64xf32>) outs(%17 : memref<1x64xf32>) {
            ^bb0(%in: f32, %in_7: f32, %in_8: f32, %out: f32):
              %32 = arith.mulf %in, %in_7 : f32
              %33 = arith.addf %32, %in_8 : f32
              linalg.yield %33 : f32
            }
            // After this: %17 = updated l_i

            // ---- Load V tile from DRAM ----
            // V_tile = V[batch_idx:batch_idx+1, n_start:n_end, :]
            // where batch_idx = %arg6, n_start = %arg7 * BN
            //
            // Address calculation for V (arg1, shape [32, 4096, 128]):
            //   %30 = %arg7 * 16384                 (N-tile offset = %arg7 * BN * head_dim = %arg7 * 128 * 128)
            //   %31 = %25 + %30                   (Total offset = batch_offset + n_tile_offset)
            //   The view: offset=%31, sizes=[1, 128, 128], strides=[524288, 128, 1]
            %30 = arith.muli %arg7, %c16384 : index
            %31 = arith.addi %25, %30 : index
            %reinterpret_cast_6 = memref.reinterpret_cast %arg1 to offset: [%31], sizes: [1, 128, 128], strides: [524288, 128, 1] : memref<32x4096x128xf32> to memref<1x128x128xf32, strided<[524288, 128, 1], offset: ?>>
            // DMA: DRAM → L1, load V tile into %14
            // Interconnect: ALL-BROADCAST via horizontal_links + vertical_links, broadcast [8,8]
            // One core loads and the data is replicated to all 8*8=64 cores.
            loom.copy %reinterpret_cast_6, %14 src_mem_space @DRAM dst_mem_space @L1, interconnect : [@horizontal_links, @vertical_links], broadcast : [8, 8] : memref<1x128x128xf32, strided<[524288, 128, 1], offset: ?>>, memref<1x128x128xf32>
            // ---- pv = p @ V ----
            // Memory: %24 is zeroed then used as accumulator for batch_matmul (out-of-place)
            // NOTE: fill sunk from pre-loop to here by SinkFillOps pass
            linalg.fill ins(%cst_1 : f32) outs(%24 : memref<1x64x128xf32>)
            // p @ V: [1, 64, 128] @ [1, 128, 128] → [1, 64, 128], result in %24
            linalg.batch_matmul ins(%15, %14 : memref<1x64x128xf32>, memref<1x128x128xf32>) outs(%24 : memref<1x64x128xf32>)
            // ---- acc = pv + acc * alpha ----
            // Reads pv from %24, reads previous acc from %23, reads alpha from %21, writes to %23
            // Memory: IN-PLACE on %23 — the running accumulator is updated.
            //         This is the core FlashAttention rescaling: acc = (p @ V) + acc * alpha
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%24, %23, %21 : memref<1x64x128xf32>, memref<1x64x128xf32>, memref<1x64xf32>) outs(%23 : memref<1x64x128xf32>) {
            ^bb0(%in: f32, %in_7: f32, %in_8: f32, %out: f32):
              %32 = arith.mulf %in_7, %in_8 : f32
              %33 = arith.addf %in, %32 : f32
              linalg.yield %33 : f32
            }
            // After this: %23 = updated acc

            // ---- m_i = m_ij (carry forward the max for next iteration) ----
            // Memory: explicit copy from %19 (m_ij) to %18 (m_i).
            // This is needed because OSB cannot alias m_i and m_ij (they are live simultaneously
            // during the alpha computation), so a separate copy is required at iteration end.
            linalg.copy ins(%19 : memref<1x64xf32>) outs(%18 : memref<1x64xf32>)
          }
          // =================== Post-loop: normalize by l_i ===================
          // output = acc / l_i
          // Reads final acc from %23, final l_i from %17, writes to %22
          // Memory: OUT-OF-PLACE — result goes to %22 (which previously held Q tile).
          //         Q is no longer needed after the inner loop, so %22 is safely reused here.
          //         This is an improvement: the old version needed a separate buffer for this.
          linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%23, %17 : memref<1x64x128xf32>, memref<1x64xf32>) outs(%22 : memref<1x64x128xf32>) {
          ^bb0(%in: f32, %in_5: f32, %out: f32):
            %28 = arith.divf %in, %in_5 : f32
            linalg.yield %28 : f32
          }
          // =================== Store output tile to DRAM ===================
          // O[batch_idx:batch_idx+1, m_start:m_end, :] = output
          //
          // Address calculation for O (arg3, shape [32, 4096, 128]):
          //   Reuses %27 = batch_offset + m_tile_offset (same offset as Q load, since Q and O share the same tiling)
          //   The view: offset=%27, sizes=[1, 64, 128], strides=[524288, 128, 1]
          %reinterpret_cast_4 = memref.reinterpret_cast %arg3 to offset: [%27], sizes: [1, 64, 128], strides: [524288, 128, 1] : memref<32x4096x128xf32> to memref<1x64x128xf32, strided<[524288, 128, 1], offset: ?>>
          // DMA: L1 → DRAM, store result from %22 to output (direct, no broadcast)
          loom.copy %22, %reinterpret_cast_4 src_mem_space @L1 dst_mem_space @DRAM, interconnect : [], broadcast : [1, 1] : memref<1x64x128xf32>, memref<1x64x128xf32, strided<[524288, 128, 1], offset: ?>>
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
          %15 = loom.alloc [1, 64, 128] on @L1 : memref<1x64x128xf32>
          %16 = loom.alloc [1, 128, 128] on @L1 : memref<1x128x128xf32>
          %17 = loom.alloc [1, 64] on @L1 : memref<1x64xf32>
          %18 = loom.alloc [1, 64] on @L1 : memref<1x64xf32>
          %19 = loom.alloc [1, 64] on @L1 : memref<1x64xf32>
          %20 = loom.alloc [1, 64] on @L1 : memref<1x64xf32>
          %21 = loom.alloc [1, 64] on @L1 : memref<1x64xf32>
          %22 = loom.alloc [1, 64, 128] on @L1 : memref<1x64x128xf32>
          %23 = loom.alloc [1, 64, 128] on @L1 : memref<1x64x128xf32>
          %24 = loom.alloc [1, 64, 128] on @L1 : memref<1x64x128xf32>
          %25 = arith.muli %arg6, %c524288 overflow<nsw> : index
          %26 = arith.muli %13, %c8192 : index
          %27 = arith.addi %25, %26 : index
          %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%27], sizes: [1, 64, 128], strides: [524288, 128, 1] : memref<32x4096x128xf32> to memref<1x64x128xf32, strided<[524288, 128, 1], offset: ?>>
          loom.copy %reinterpret_cast, %22 src_mem_space @DRAM dst_mem_space @L1, interconnect : [], broadcast : [1, 1] : memref<1x64x128xf32, strided<[524288, 128, 1], offset: ?>>, memref<1x64x128xf32>
          linalg.fill ins(%cst_1 : f32) outs(%23 : memref<1x64x128xf32>)
          linalg.fill ins(%cst_2 : f32) outs(%17 : memref<1x64xf32>)
          linalg.fill ins(%cst_3 : f32) outs(%18 : memref<1x64xf32>)
          scf.for %arg7 = %c0 to %c32 step %c1 {
            %28 = arith.muli %arg7, %c128 : index
            %29 = arith.addi %25, %28 : index
            %reinterpret_cast_5 = memref.reinterpret_cast %arg0 to offset: [%29], sizes: [1, 128, 128], strides: [524288, 4096, 1] : memref<32x128x4096xf32> to memref<1x128x128xf32, strided<[524288, 4096, 1], offset: ?>>
            loom.copy %reinterpret_cast_5, %16 src_mem_space @DRAM dst_mem_space @L1, interconnect : [@horizontal_links, @vertical_links], broadcast : [8, 8] : memref<1x128x128xf32, strided<[524288, 4096, 1], offset: ?>>, memref<1x128x128xf32>
            linalg.fill ins(%cst_1 : f32) outs(%15 : memref<1x64x128xf32>)
            linalg.batch_matmul ins(%22, %16 : memref<1x64x128xf32>, memref<1x128x128xf32>) outs(%15 : memref<1x64x128xf32>)
            linalg.fill ins(%cst_3 : f32) outs(%19 : memref<1x64xf32>)
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%15 : memref<1x64x128xf32>) outs(%19 : memref<1x64xf32>) {
            ^bb0(%in: f32, %out: f32):
              %32 = arith.maximumf %in, %out : f32
              linalg.yield %32 : f32
            }
            linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%18, %19 : memref<1x64xf32>, memref<1x64xf32>) outs(%19 : memref<1x64xf32>) {
            ^bb0(%in: f32, %in_7: f32, %out: f32):
              %32 = arith.truncf %cst_0 : f64 to f32
              %33 = arith.mulf %in_7, %32 : f32
              %34 = arith.cmpf ogt, %in, %33 : f32
              %35 = arith.select %34, %in, %33 : f32
              linalg.yield %35 : f32
            }
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%15, %19 : memref<1x64x128xf32>, memref<1x64xf32>) outs(%15 : memref<1x64x128xf32>) {
            ^bb0(%in: f32, %in_7: f32, %out: f32):
              %32 = arith.truncf %cst_0 : f64 to f32
              %33 = arith.mulf %in, %32 : f32
              %34 = arith.subf %33, %in_7 : f32
              %35 = math.powf %cst, %34 : f32
              linalg.yield %35 : f32
            }
            linalg.fill ins(%cst_1 : f32) outs(%20 : memref<1x64xf32>)
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%15 : memref<1x64x128xf32>) outs(%20 : memref<1x64xf32>) {
            ^bb0(%in: f32, %out: f32):
              %32 = arith.addf %in, %out : f32
              linalg.yield %32 : f32
            }
            linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%18, %19 : memref<1x64xf32>, memref<1x64xf32>) outs(%21 : memref<1x64xf32>) {
            ^bb0(%in: f32, %in_7: f32, %out: f32):
              %32 = arith.subf %in, %in_7 : f32
              %33 = math.powf %cst, %32 : f32
              linalg.yield %33 : f32
            }
            linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%17, %21, %20 : memref<1x64xf32>, memref<1x64xf32>, memref<1x64xf32>) outs(%17 : memref<1x64xf32>) {
            ^bb0(%in: f32, %in_7: f32, %in_8: f32, %out: f32):
              %32 = arith.mulf %in, %in_7 : f32
              %33 = arith.addf %32, %in_8 : f32
              linalg.yield %33 : f32
            }
            %30 = arith.muli %arg7, %c16384 : index
            %31 = arith.addi %25, %30 : index
            %reinterpret_cast_6 = memref.reinterpret_cast %arg1 to offset: [%31], sizes: [1, 128, 128], strides: [524288, 128, 1] : memref<32x4096x128xf32> to memref<1x128x128xf32, strided<[524288, 128, 1], offset: ?>>
            loom.copy %reinterpret_cast_6, %14 src_mem_space @DRAM dst_mem_space @L1, interconnect : [@horizontal_links], broadcast : [1, 8] : memref<1x128x128xf32, strided<[524288, 128, 1], offset: ?>>, memref<1x128x128xf32>
            linalg.fill ins(%cst_1 : f32) outs(%24 : memref<1x64x128xf32>)
            linalg.batch_matmul ins(%15, %14 : memref<1x64x128xf32>, memref<1x128x128xf32>) outs(%24 : memref<1x64x128xf32>)
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%24, %23, %21 : memref<1x64x128xf32>, memref<1x64x128xf32>, memref<1x64xf32>) outs(%23 : memref<1x64x128xf32>) {
            ^bb0(%in: f32, %in_7: f32, %in_8: f32, %out: f32):
              %32 = arith.mulf %in_7, %in_8 : f32
              %33 = arith.addf %in, %32 : f32
              linalg.yield %33 : f32
            }
            linalg.copy ins(%19 : memref<1x64xf32>) outs(%18 : memref<1x64xf32>)
          }
          linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%23, %17 : memref<1x64x128xf32>, memref<1x64xf32>) outs(%22 : memref<1x64x128xf32>) {
          ^bb0(%in: f32, %in_5: f32, %out: f32):
            %28 = arith.divf %in, %in_5 : f32
            linalg.yield %28 : f32
          }
          %reinterpret_cast_4 = memref.reinterpret_cast %arg3 to offset: [%27], sizes: [1, 64, 128], strides: [524288, 128, 1] : memref<32x4096x128xf32> to memref<1x64x128xf32, strided<[524288, 128, 1], offset: ?>>
          loom.copy %22, %reinterpret_cast_4 src_mem_space @L1 dst_mem_space @DRAM, interconnect : [], broadcast : [1, 1] : memref<1x64x128xf32>, memref<1x64x128xf32, strided<[524288, 128, 1], offset: ?>>
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
          %15 = loom.alloc [1, 64, 128] on @L1 : memref<1x64x128xf32>
          %16 = loom.alloc [1, 128, 128] on @L1 : memref<1x128x128xf32>
          %17 = loom.alloc [1, 64] on @L1 : memref<1x64xf32>
          %18 = loom.alloc [1, 64] on @L1 : memref<1x64xf32>
          %19 = loom.alloc [1, 64] on @L1 : memref<1x64xf32>
          %20 = loom.alloc [1, 64] on @L1 : memref<1x64xf32>
          %21 = loom.alloc [1, 64] on @L1 : memref<1x64xf32>
          %22 = loom.alloc [1, 64, 128] on @L1 : memref<1x64x128xf32>
          %23 = loom.alloc [1, 64, 128] on @L1 : memref<1x64x128xf32>
          %24 = loom.alloc [1, 64, 128] on @L1 : memref<1x64x128xf32>
          %25 = arith.muli %arg6, %c524288 overflow<nsw> : index
          %26 = arith.muli %13, %c8192 : index
          %27 = arith.addi %25, %26 : index
          %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%27], sizes: [1, 64, 128], strides: [524288, 128, 1] : memref<32x4096x128xf32> to memref<1x64x128xf32, strided<[524288, 128, 1], offset: ?>>
          loom.copy %reinterpret_cast, %22 src_mem_space @DRAM dst_mem_space @L1, interconnect : [], broadcast : [1, 1] : memref<1x64x128xf32, strided<[524288, 128, 1], offset: ?>>, memref<1x64x128xf32>
          linalg.fill ins(%cst_1 : f32) outs(%23 : memref<1x64x128xf32>)
          linalg.fill ins(%cst_2 : f32) outs(%17 : memref<1x64xf32>)
          linalg.fill ins(%cst_3 : f32) outs(%18 : memref<1x64xf32>)
          scf.for %arg7 = %c0 to %c32 step %c1 {
            %28 = arith.muli %arg7, %c128 : index
            %29 = arith.addi %25, %28 : index
            %reinterpret_cast_5 = memref.reinterpret_cast %arg0 to offset: [%29], sizes: [1, 128, 128], strides: [524288, 4096, 1] : memref<32x128x4096xf32> to memref<1x128x128xf32, strided<[524288, 4096, 1], offset: ?>>
            loom.copy %reinterpret_cast_5, %16 src_mem_space @DRAM dst_mem_space @L1, interconnect : [@horizontal_links, @vertical_links], broadcast : [8, 8] : memref<1x128x128xf32, strided<[524288, 4096, 1], offset: ?>>, memref<1x128x128xf32>
            linalg.fill ins(%cst_1 : f32) outs(%15 : memref<1x64x128xf32>)
            linalg.batch_matmul ins(%22, %16 : memref<1x64x128xf32>, memref<1x128x128xf32>) outs(%15 : memref<1x64x128xf32>)
            linalg.fill ins(%cst_3 : f32) outs(%19 : memref<1x64xf32>)
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%15 : memref<1x64x128xf32>) outs(%19 : memref<1x64xf32>) {
            ^bb0(%in: f32, %out: f32):
              %32 = arith.maximumf %in, %out : f32
              linalg.yield %32 : f32
            }
            linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%18, %19 : memref<1x64xf32>, memref<1x64xf32>) outs(%19 : memref<1x64xf32>) {
            ^bb0(%in: f32, %in_7: f32, %out: f32):
              %32 = arith.truncf %cst_0 : f64 to f32
              %33 = arith.mulf %in_7, %32 : f32
              %34 = arith.cmpf ogt, %in, %33 : f32
              %35 = arith.select %34, %in, %33 : f32
              linalg.yield %35 : f32
            }
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%15, %19 : memref<1x64x128xf32>, memref<1x64xf32>) outs(%15 : memref<1x64x128xf32>) {
            ^bb0(%in: f32, %in_7: f32, %out: f32):
              %32 = arith.truncf %cst_0 : f64 to f32
              %33 = arith.mulf %in, %32 : f32
              %34 = arith.subf %33, %in_7 : f32
              %35 = math.powf %cst, %34 : f32
              linalg.yield %35 : f32
            }
            linalg.fill ins(%cst_1 : f32) outs(%20 : memref<1x64xf32>)
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%15 : memref<1x64x128xf32>) outs(%20 : memref<1x64xf32>) {
            ^bb0(%in: f32, %out: f32):
              %32 = arith.addf %in, %out : f32
              linalg.yield %32 : f32
            }
            linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%18, %19 : memref<1x64xf32>, memref<1x64xf32>) outs(%21 : memref<1x64xf32>) {
            ^bb0(%in: f32, %in_7: f32, %out: f32):
              %32 = arith.subf %in, %in_7 : f32
              %33 = math.powf %cst, %32 : f32
              linalg.yield %33 : f32
            }
            linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%17, %21, %20 : memref<1x64xf32>, memref<1x64xf32>, memref<1x64xf32>) outs(%17 : memref<1x64xf32>) {
            ^bb0(%in: f32, %in_7: f32, %in_8: f32, %out: f32):
              %32 = arith.mulf %in, %in_7 : f32
              %33 = arith.addf %32, %in_8 : f32
              linalg.yield %33 : f32
            }
            %30 = arith.muli %arg7, %c16384 : index
            %31 = arith.addi %25, %30 : index
            %reinterpret_cast_6 = memref.reinterpret_cast %arg1 to offset: [%31], sizes: [1, 128, 128], strides: [524288, 128, 1] : memref<32x4096x128xf32> to memref<1x128x128xf32, strided<[524288, 128, 1], offset: ?>>
            loom.copy %reinterpret_cast_6, %14 src_mem_space @DRAM dst_mem_space @L1, interconnect : [@vertical_links], broadcast : [8, 1] : memref<1x128x128xf32, strided<[524288, 128, 1], offset: ?>>, memref<1x128x128xf32>
            linalg.fill ins(%cst_1 : f32) outs(%24 : memref<1x64x128xf32>)
            linalg.batch_matmul ins(%15, %14 : memref<1x64x128xf32>, memref<1x128x128xf32>) outs(%24 : memref<1x64x128xf32>)
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%24, %23, %21 : memref<1x64x128xf32>, memref<1x64x128xf32>, memref<1x64xf32>) outs(%23 : memref<1x64x128xf32>) {
            ^bb0(%in: f32, %in_7: f32, %in_8: f32, %out: f32):
              %32 = arith.mulf %in_7, %in_8 : f32
              %33 = arith.addf %in, %32 : f32
              linalg.yield %33 : f32
            }
            linalg.copy ins(%19 : memref<1x64xf32>) outs(%18 : memref<1x64xf32>)
          }
          linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%23, %17 : memref<1x64x128xf32>, memref<1x64xf32>) outs(%22 : memref<1x64x128xf32>) {
          ^bb0(%in: f32, %in_5: f32, %out: f32):
            %28 = arith.divf %in, %in_5 : f32
            linalg.yield %28 : f32
          }
          %reinterpret_cast_4 = memref.reinterpret_cast %arg3 to offset: [%27], sizes: [1, 64, 128], strides: [524288, 128, 1] : memref<32x4096x128xf32> to memref<1x64x128xf32, strided<[524288, 128, 1], offset: ?>>
          loom.copy %22, %reinterpret_cast_4 src_mem_space @L1 dst_mem_space @DRAM, interconnect : [], broadcast : [1, 1] : memref<1x64x128xf32>, memref<1x64x128xf32, strided<[524288, 128, 1], offset: ?>>
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
          %15 = loom.alloc [1, 64, 128] on @L1 : memref<1x64x128xf32>
          %16 = loom.alloc [1, 128, 128] on @L1 : memref<1x128x128xf32>
          %17 = loom.alloc [1, 64] on @L1 : memref<1x64xf32>
          %18 = loom.alloc [1, 64] on @L1 : memref<1x64xf32>
          %19 = loom.alloc [1, 64] on @L1 : memref<1x64xf32>
          %20 = loom.alloc [1, 64] on @L1 : memref<1x64xf32>
          %21 = loom.alloc [1, 64] on @L1 : memref<1x64xf32>
          %22 = loom.alloc [1, 64, 128] on @L1 : memref<1x64x128xf32>
          %23 = loom.alloc [1, 64, 128] on @L1 : memref<1x64x128xf32>
          %24 = loom.alloc [1, 64, 128] on @L1 : memref<1x64x128xf32>
          %25 = arith.muli %arg6, %c524288 overflow<nsw> : index
          %26 = arith.muli %13, %c8192 : index
          %27 = arith.addi %25, %26 : index
          %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%27], sizes: [1, 64, 128], strides: [524288, 128, 1] : memref<32x4096x128xf32> to memref<1x64x128xf32, strided<[524288, 128, 1], offset: ?>>
          loom.copy %reinterpret_cast, %22 src_mem_space @DRAM dst_mem_space @L1, interconnect : [], broadcast : [1, 1] : memref<1x64x128xf32, strided<[524288, 128, 1], offset: ?>>, memref<1x64x128xf32>
          linalg.fill ins(%cst_1 : f32) outs(%23 : memref<1x64x128xf32>)
          linalg.fill ins(%cst_2 : f32) outs(%17 : memref<1x64xf32>)
          linalg.fill ins(%cst_3 : f32) outs(%18 : memref<1x64xf32>)
          scf.for %arg7 = %c0 to %c32 step %c1 {
            %28 = arith.muli %arg7, %c128 : index
            %29 = arith.addi %25, %28 : index
            %reinterpret_cast_5 = memref.reinterpret_cast %arg0 to offset: [%29], sizes: [1, 128, 128], strides: [524288, 4096, 1] : memref<32x128x4096xf32> to memref<1x128x128xf32, strided<[524288, 4096, 1], offset: ?>>
            loom.copy %reinterpret_cast_5, %16 src_mem_space @DRAM dst_mem_space @L1, interconnect : [@horizontal_links], broadcast : [1, 8] : memref<1x128x128xf32, strided<[524288, 4096, 1], offset: ?>>, memref<1x128x128xf32>
            linalg.fill ins(%cst_1 : f32) outs(%15 : memref<1x64x128xf32>)
            linalg.batch_matmul ins(%22, %16 : memref<1x64x128xf32>, memref<1x128x128xf32>) outs(%15 : memref<1x64x128xf32>)
            linalg.fill ins(%cst_3 : f32) outs(%19 : memref<1x64xf32>)
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%15 : memref<1x64x128xf32>) outs(%19 : memref<1x64xf32>) {
            ^bb0(%in: f32, %out: f32):
              %32 = arith.maximumf %in, %out : f32
              linalg.yield %32 : f32
            }
            linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%18, %19 : memref<1x64xf32>, memref<1x64xf32>) outs(%19 : memref<1x64xf32>) {
            ^bb0(%in: f32, %in_7: f32, %out: f32):
              %32 = arith.truncf %cst_0 : f64 to f32
              %33 = arith.mulf %in_7, %32 : f32
              %34 = arith.cmpf ogt, %in, %33 : f32
              %35 = arith.select %34, %in, %33 : f32
              linalg.yield %35 : f32
            }
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%15, %19 : memref<1x64x128xf32>, memref<1x64xf32>) outs(%15 : memref<1x64x128xf32>) {
            ^bb0(%in: f32, %in_7: f32, %out: f32):
              %32 = arith.truncf %cst_0 : f64 to f32
              %33 = arith.mulf %in, %32 : f32
              %34 = arith.subf %33, %in_7 : f32
              %35 = math.powf %cst, %34 : f32
              linalg.yield %35 : f32
            }
            linalg.fill ins(%cst_1 : f32) outs(%20 : memref<1x64xf32>)
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%15 : memref<1x64x128xf32>) outs(%20 : memref<1x64xf32>) {
            ^bb0(%in: f32, %out: f32):
              %32 = arith.addf %in, %out : f32
              linalg.yield %32 : f32
            }
            linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%18, %19 : memref<1x64xf32>, memref<1x64xf32>) outs(%21 : memref<1x64xf32>) {
            ^bb0(%in: f32, %in_7: f32, %out: f32):
              %32 = arith.subf %in, %in_7 : f32
              %33 = math.powf %cst, %32 : f32
              linalg.yield %33 : f32
            }
            linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%17, %21, %20 : memref<1x64xf32>, memref<1x64xf32>, memref<1x64xf32>) outs(%17 : memref<1x64xf32>) {
            ^bb0(%in: f32, %in_7: f32, %in_8: f32, %out: f32):
              %32 = arith.mulf %in, %in_7 : f32
              %33 = arith.addf %32, %in_8 : f32
              linalg.yield %33 : f32
            }
            %30 = arith.muli %arg7, %c16384 : index
            %31 = arith.addi %25, %30 : index
            %reinterpret_cast_6 = memref.reinterpret_cast %arg1 to offset: [%31], sizes: [1, 128, 128], strides: [524288, 128, 1] : memref<32x4096x128xf32> to memref<1x128x128xf32, strided<[524288, 128, 1], offset: ?>>
            loom.copy %reinterpret_cast_6, %14 src_mem_space @DRAM dst_mem_space @L1, interconnect : [], broadcast : [1, 1] : memref<1x128x128xf32, strided<[524288, 128, 1], offset: ?>>, memref<1x128x128xf32>
            linalg.fill ins(%cst_1 : f32) outs(%24 : memref<1x64x128xf32>)
            linalg.batch_matmul ins(%15, %14 : memref<1x64x128xf32>, memref<1x128x128xf32>) outs(%24 : memref<1x64x128xf32>)
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%24, %23, %21 : memref<1x64x128xf32>, memref<1x64x128xf32>, memref<1x64xf32>) outs(%23 : memref<1x64x128xf32>) {
            ^bb0(%in: f32, %in_7: f32, %in_8: f32, %out: f32):
              %32 = arith.mulf %in_7, %in_8 : f32
              %33 = arith.addf %in, %32 : f32
              linalg.yield %33 : f32
            }
            linalg.copy ins(%19 : memref<1x64xf32>) outs(%18 : memref<1x64xf32>)
          }
          linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%23, %17 : memref<1x64x128xf32>, memref<1x64xf32>) outs(%22 : memref<1x64x128xf32>) {
          ^bb0(%in: f32, %in_5: f32, %out: f32):
            %28 = arith.divf %in, %in_5 : f32
            linalg.yield %28 : f32
          }
          %reinterpret_cast_4 = memref.reinterpret_cast %arg3 to offset: [%27], sizes: [1, 64, 128], strides: [524288, 128, 1] : memref<32x4096x128xf32> to memref<1x64x128xf32, strided<[524288, 128, 1], offset: ?>>
          loom.copy %22, %reinterpret_cast_4 src_mem_space @L1 dst_mem_space @DRAM, interconnect : [], broadcast : [1, 1] : memref<1x64x128xf32>, memref<1x64x128xf32, strided<[524288, 128, 1], offset: ?>>
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
          %15 = loom.alloc [1, 64, 128] on @L1 : memref<1x64x128xf32>
          %16 = loom.alloc [1, 128, 128] on @L1 : memref<1x128x128xf32>
          %17 = loom.alloc [1, 64] on @L1 : memref<1x64xf32>
          %18 = loom.alloc [1, 64] on @L1 : memref<1x64xf32>
          %19 = loom.alloc [1, 64] on @L1 : memref<1x64xf32>
          %20 = loom.alloc [1, 64] on @L1 : memref<1x64xf32>
          %21 = loom.alloc [1, 64] on @L1 : memref<1x64xf32>
          %22 = loom.alloc [1, 64, 128] on @L1 : memref<1x64x128xf32>
          %23 = loom.alloc [1, 64, 128] on @L1 : memref<1x64x128xf32>
          %24 = loom.alloc [1, 64, 128] on @L1 : memref<1x64x128xf32>
          %25 = arith.muli %arg6, %c524288 overflow<nsw> : index
          %26 = arith.muli %13, %c8192 : index
          %27 = arith.addi %25, %26 : index
          %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%27], sizes: [1, 64, 128], strides: [524288, 128, 1] : memref<32x4096x128xf32> to memref<1x64x128xf32, strided<[524288, 128, 1], offset: ?>>
          loom.copy %reinterpret_cast, %22 src_mem_space @DRAM dst_mem_space @L1, interconnect : [], broadcast : [1, 1] : memref<1x64x128xf32, strided<[524288, 128, 1], offset: ?>>, memref<1x64x128xf32>
          linalg.fill ins(%cst_1 : f32) outs(%23 : memref<1x64x128xf32>)
          linalg.fill ins(%cst_2 : f32) outs(%17 : memref<1x64xf32>)
          linalg.fill ins(%cst_3 : f32) outs(%18 : memref<1x64xf32>)
          scf.for %arg7 = %c0 to %c32 step %c1 {
            %28 = arith.muli %arg7, %c128 : index
            %29 = arith.addi %25, %28 : index
            %reinterpret_cast_5 = memref.reinterpret_cast %arg0 to offset: [%29], sizes: [1, 128, 128], strides: [524288, 4096, 1] : memref<32x128x4096xf32> to memref<1x128x128xf32, strided<[524288, 4096, 1], offset: ?>>
            loom.copy %reinterpret_cast_5, %16 src_mem_space @DRAM dst_mem_space @L1, interconnect : [@horizontal_links], broadcast : [1, 8] : memref<1x128x128xf32, strided<[524288, 4096, 1], offset: ?>>, memref<1x128x128xf32>
            linalg.fill ins(%cst_1 : f32) outs(%15 : memref<1x64x128xf32>)
            linalg.batch_matmul ins(%22, %16 : memref<1x64x128xf32>, memref<1x128x128xf32>) outs(%15 : memref<1x64x128xf32>)
            linalg.fill ins(%cst_3 : f32) outs(%19 : memref<1x64xf32>)
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%15 : memref<1x64x128xf32>) outs(%19 : memref<1x64xf32>) {
            ^bb0(%in: f32, %out: f32):
              %32 = arith.maximumf %in, %out : f32
              linalg.yield %32 : f32
            }
            linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%18, %19 : memref<1x64xf32>, memref<1x64xf32>) outs(%19 : memref<1x64xf32>) {
            ^bb0(%in: f32, %in_7: f32, %out: f32):
              %32 = arith.truncf %cst_0 : f64 to f32
              %33 = arith.mulf %in_7, %32 : f32
              %34 = arith.cmpf ogt, %in, %33 : f32
              %35 = arith.select %34, %in, %33 : f32
              linalg.yield %35 : f32
            }
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%15, %19 : memref<1x64x128xf32>, memref<1x64xf32>) outs(%15 : memref<1x64x128xf32>) {
            ^bb0(%in: f32, %in_7: f32, %out: f32):
              %32 = arith.truncf %cst_0 : f64 to f32
              %33 = arith.mulf %in, %32 : f32
              %34 = arith.subf %33, %in_7 : f32
              %35 = math.powf %cst, %34 : f32
              linalg.yield %35 : f32
            }
            linalg.fill ins(%cst_1 : f32) outs(%20 : memref<1x64xf32>)
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%15 : memref<1x64x128xf32>) outs(%20 : memref<1x64xf32>) {
            ^bb0(%in: f32, %out: f32):
              %32 = arith.addf %in, %out : f32
              linalg.yield %32 : f32
            }
            linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%18, %19 : memref<1x64xf32>, memref<1x64xf32>) outs(%21 : memref<1x64xf32>) {
            ^bb0(%in: f32, %in_7: f32, %out: f32):
              %32 = arith.subf %in, %in_7 : f32
              %33 = math.powf %cst, %32 : f32
              linalg.yield %33 : f32
            }
            linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%17, %21, %20 : memref<1x64xf32>, memref<1x64xf32>, memref<1x64xf32>) outs(%17 : memref<1x64xf32>) {
            ^bb0(%in: f32, %in_7: f32, %in_8: f32, %out: f32):
              %32 = arith.mulf %in, %in_7 : f32
              %33 = arith.addf %32, %in_8 : f32
              linalg.yield %33 : f32
            }
            %30 = arith.muli %arg7, %c16384 : index
            %31 = arith.addi %25, %30 : index
            %reinterpret_cast_6 = memref.reinterpret_cast %arg1 to offset: [%31], sizes: [1, 128, 128], strides: [524288, 128, 1] : memref<32x4096x128xf32> to memref<1x128x128xf32, strided<[524288, 128, 1], offset: ?>>
            loom.copy %reinterpret_cast_6, %14 src_mem_space @DRAM dst_mem_space @L1, interconnect : [@horizontal_links, @vertical_links], broadcast : [8, 8] : memref<1x128x128xf32, strided<[524288, 128, 1], offset: ?>>, memref<1x128x128xf32>
            linalg.fill ins(%cst_1 : f32) outs(%24 : memref<1x64x128xf32>)
            linalg.batch_matmul ins(%15, %14 : memref<1x64x128xf32>, memref<1x128x128xf32>) outs(%24 : memref<1x64x128xf32>)
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%24, %23, %21 : memref<1x64x128xf32>, memref<1x64x128xf32>, memref<1x64xf32>) outs(%23 : memref<1x64x128xf32>) {
            ^bb0(%in: f32, %in_7: f32, %in_8: f32, %out: f32):
              %32 = arith.mulf %in_7, %in_8 : f32
              %33 = arith.addf %in, %32 : f32
              linalg.yield %33 : f32
            }
            linalg.copy ins(%19 : memref<1x64xf32>) outs(%18 : memref<1x64xf32>)
          }
          linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%23, %17 : memref<1x64x128xf32>, memref<1x64xf32>) outs(%22 : memref<1x64x128xf32>) {
          ^bb0(%in: f32, %in_5: f32, %out: f32):
            %28 = arith.divf %in, %in_5 : f32
            linalg.yield %28 : f32
          }
          %reinterpret_cast_4 = memref.reinterpret_cast %arg3 to offset: [%27], sizes: [1, 64, 128], strides: [524288, 128, 1] : memref<32x4096x128xf32> to memref<1x64x128xf32, strided<[524288, 128, 1], offset: ?>>
          loom.copy %22, %reinterpret_cast_4 src_mem_space @L1 dst_mem_space @DRAM, interconnect : [], broadcast : [1, 1] : memref<1x64x128xf32>, memref<1x64x128xf32, strided<[524288, 128, 1], offset: ?>>
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
          %15 = loom.alloc [1, 64, 128] on @L1 : memref<1x64x128xf32>
          %16 = loom.alloc [1, 128, 128] on @L1 : memref<1x128x128xf32>
          %17 = loom.alloc [1, 64] on @L1 : memref<1x64xf32>
          %18 = loom.alloc [1, 64] on @L1 : memref<1x64xf32>
          %19 = loom.alloc [1, 64] on @L1 : memref<1x64xf32>
          %20 = loom.alloc [1, 64] on @L1 : memref<1x64xf32>
          %21 = loom.alloc [1, 64] on @L1 : memref<1x64xf32>
          %22 = loom.alloc [1, 64, 128] on @L1 : memref<1x64x128xf32>
          %23 = loom.alloc [1, 64, 128] on @L1 : memref<1x64x128xf32>
          %24 = loom.alloc [1, 64, 128] on @L1 : memref<1x64x128xf32>
          %25 = arith.muli %arg6, %c524288 overflow<nsw> : index
          %26 = arith.muli %13, %c8192 : index
          %27 = arith.addi %25, %26 : index
          %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%27], sizes: [1, 64, 128], strides: [524288, 128, 1] : memref<32x4096x128xf32> to memref<1x64x128xf32, strided<[524288, 128, 1], offset: ?>>
          loom.copy %reinterpret_cast, %22 src_mem_space @DRAM dst_mem_space @L1, interconnect : [], broadcast : [1, 1] : memref<1x64x128xf32, strided<[524288, 128, 1], offset: ?>>, memref<1x64x128xf32>
          linalg.fill ins(%cst_1 : f32) outs(%23 : memref<1x64x128xf32>)
          linalg.fill ins(%cst_2 : f32) outs(%17 : memref<1x64xf32>)
          linalg.fill ins(%cst_3 : f32) outs(%18 : memref<1x64xf32>)
          scf.for %arg7 = %c0 to %c32 step %c1 {
            %28 = arith.muli %arg7, %c128 : index
            %29 = arith.addi %25, %28 : index
            %reinterpret_cast_5 = memref.reinterpret_cast %arg0 to offset: [%29], sizes: [1, 128, 128], strides: [524288, 4096, 1] : memref<32x128x4096xf32> to memref<1x128x128xf32, strided<[524288, 4096, 1], offset: ?>>
            loom.copy %reinterpret_cast_5, %16 src_mem_space @DRAM dst_mem_space @L1, interconnect : [@horizontal_links], broadcast : [1, 8] : memref<1x128x128xf32, strided<[524288, 4096, 1], offset: ?>>, memref<1x128x128xf32>
            linalg.fill ins(%cst_1 : f32) outs(%15 : memref<1x64x128xf32>)
            linalg.batch_matmul ins(%22, %16 : memref<1x64x128xf32>, memref<1x128x128xf32>) outs(%15 : memref<1x64x128xf32>)
            linalg.fill ins(%cst_3 : f32) outs(%19 : memref<1x64xf32>)
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%15 : memref<1x64x128xf32>) outs(%19 : memref<1x64xf32>) {
            ^bb0(%in: f32, %out: f32):
              %32 = arith.maximumf %in, %out : f32
              linalg.yield %32 : f32
            }
            linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%18, %19 : memref<1x64xf32>, memref<1x64xf32>) outs(%19 : memref<1x64xf32>) {
            ^bb0(%in: f32, %in_7: f32, %out: f32):
              %32 = arith.truncf %cst_0 : f64 to f32
              %33 = arith.mulf %in_7, %32 : f32
              %34 = arith.cmpf ogt, %in, %33 : f32
              %35 = arith.select %34, %in, %33 : f32
              linalg.yield %35 : f32
            }
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%15, %19 : memref<1x64x128xf32>, memref<1x64xf32>) outs(%15 : memref<1x64x128xf32>) {
            ^bb0(%in: f32, %in_7: f32, %out: f32):
              %32 = arith.truncf %cst_0 : f64 to f32
              %33 = arith.mulf %in, %32 : f32
              %34 = arith.subf %33, %in_7 : f32
              %35 = math.powf %cst, %34 : f32
              linalg.yield %35 : f32
            }
            linalg.fill ins(%cst_1 : f32) outs(%20 : memref<1x64xf32>)
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%15 : memref<1x64x128xf32>) outs(%20 : memref<1x64xf32>) {
            ^bb0(%in: f32, %out: f32):
              %32 = arith.addf %in, %out : f32
              linalg.yield %32 : f32
            }
            linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%18, %19 : memref<1x64xf32>, memref<1x64xf32>) outs(%21 : memref<1x64xf32>) {
            ^bb0(%in: f32, %in_7: f32, %out: f32):
              %32 = arith.subf %in, %in_7 : f32
              %33 = math.powf %cst, %32 : f32
              linalg.yield %33 : f32
            }
            linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%17, %21, %20 : memref<1x64xf32>, memref<1x64xf32>, memref<1x64xf32>) outs(%17 : memref<1x64xf32>) {
            ^bb0(%in: f32, %in_7: f32, %in_8: f32, %out: f32):
              %32 = arith.mulf %in, %in_7 : f32
              %33 = arith.addf %32, %in_8 : f32
              linalg.yield %33 : f32
            }
            %30 = arith.muli %arg7, %c16384 : index
            %31 = arith.addi %25, %30 : index
            %reinterpret_cast_6 = memref.reinterpret_cast %arg1 to offset: [%31], sizes: [1, 128, 128], strides: [524288, 128, 1] : memref<32x4096x128xf32> to memref<1x128x128xf32, strided<[524288, 128, 1], offset: ?>>
            loom.copy %reinterpret_cast_6, %14 src_mem_space @DRAM dst_mem_space @L1, interconnect : [@horizontal_links], broadcast : [1, 8] : memref<1x128x128xf32, strided<[524288, 128, 1], offset: ?>>, memref<1x128x128xf32>
            linalg.fill ins(%cst_1 : f32) outs(%24 : memref<1x64x128xf32>)
            linalg.batch_matmul ins(%15, %14 : memref<1x64x128xf32>, memref<1x128x128xf32>) outs(%24 : memref<1x64x128xf32>)
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%24, %23, %21 : memref<1x64x128xf32>, memref<1x64x128xf32>, memref<1x64xf32>) outs(%23 : memref<1x64x128xf32>) {
            ^bb0(%in: f32, %in_7: f32, %in_8: f32, %out: f32):
              %32 = arith.mulf %in_7, %in_8 : f32
              %33 = arith.addf %in, %32 : f32
              linalg.yield %33 : f32
            }
            linalg.copy ins(%19 : memref<1x64xf32>) outs(%18 : memref<1x64xf32>)
          }
          linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%23, %17 : memref<1x64x128xf32>, memref<1x64xf32>) outs(%22 : memref<1x64x128xf32>) {
          ^bb0(%in: f32, %in_5: f32, %out: f32):
            %28 = arith.divf %in, %in_5 : f32
            linalg.yield %28 : f32
          }
          %reinterpret_cast_4 = memref.reinterpret_cast %arg3 to offset: [%27], sizes: [1, 64, 128], strides: [524288, 128, 1] : memref<32x4096x128xf32> to memref<1x64x128xf32, strided<[524288, 128, 1], offset: ?>>
          loom.copy %22, %reinterpret_cast_4 src_mem_space @L1 dst_mem_space @DRAM, interconnect : [], broadcast : [1, 1] : memref<1x64x128xf32>, memref<1x64x128xf32, strided<[524288, 128, 1], offset: ?>>
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
          %15 = loom.alloc [1, 64, 128] on @L1 : memref<1x64x128xf32>
          %16 = loom.alloc [1, 128, 128] on @L1 : memref<1x128x128xf32>
          %17 = loom.alloc [1, 64] on @L1 : memref<1x64xf32>
          %18 = loom.alloc [1, 64] on @L1 : memref<1x64xf32>
          %19 = loom.alloc [1, 64] on @L1 : memref<1x64xf32>
          %20 = loom.alloc [1, 64] on @L1 : memref<1x64xf32>
          %21 = loom.alloc [1, 64] on @L1 : memref<1x64xf32>
          %22 = loom.alloc [1, 64, 128] on @L1 : memref<1x64x128xf32>
          %23 = loom.alloc [1, 64, 128] on @L1 : memref<1x64x128xf32>
          %24 = loom.alloc [1, 64, 128] on @L1 : memref<1x64x128xf32>
          %25 = arith.muli %arg6, %c524288 overflow<nsw> : index
          %26 = arith.muli %13, %c8192 : index
          %27 = arith.addi %25, %26 : index
          %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%27], sizes: [1, 64, 128], strides: [524288, 128, 1] : memref<32x4096x128xf32> to memref<1x64x128xf32, strided<[524288, 128, 1], offset: ?>>
          loom.copy %reinterpret_cast, %22 src_mem_space @DRAM dst_mem_space @L1, interconnect : [], broadcast : [1, 1] : memref<1x64x128xf32, strided<[524288, 128, 1], offset: ?>>, memref<1x64x128xf32>
          linalg.fill ins(%cst_1 : f32) outs(%23 : memref<1x64x128xf32>)
          linalg.fill ins(%cst_2 : f32) outs(%17 : memref<1x64xf32>)
          linalg.fill ins(%cst_3 : f32) outs(%18 : memref<1x64xf32>)
          scf.for %arg7 = %c0 to %c32 step %c1 {
            %28 = arith.muli %arg7, %c128 : index
            %29 = arith.addi %25, %28 : index
            %reinterpret_cast_5 = memref.reinterpret_cast %arg0 to offset: [%29], sizes: [1, 128, 128], strides: [524288, 4096, 1] : memref<32x128x4096xf32> to memref<1x128x128xf32, strided<[524288, 4096, 1], offset: ?>>
            loom.copy %reinterpret_cast_5, %16 src_mem_space @DRAM dst_mem_space @L1, interconnect : [@horizontal_links], broadcast : [1, 8] : memref<1x128x128xf32, strided<[524288, 4096, 1], offset: ?>>, memref<1x128x128xf32>
            linalg.fill ins(%cst_1 : f32) outs(%15 : memref<1x64x128xf32>)
            linalg.batch_matmul ins(%22, %16 : memref<1x64x128xf32>, memref<1x128x128xf32>) outs(%15 : memref<1x64x128xf32>)
            linalg.fill ins(%cst_3 : f32) outs(%19 : memref<1x64xf32>)
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%15 : memref<1x64x128xf32>) outs(%19 : memref<1x64xf32>) {
            ^bb0(%in: f32, %out: f32):
              %32 = arith.maximumf %in, %out : f32
              linalg.yield %32 : f32
            }
            linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%18, %19 : memref<1x64xf32>, memref<1x64xf32>) outs(%19 : memref<1x64xf32>) {
            ^bb0(%in: f32, %in_7: f32, %out: f32):
              %32 = arith.truncf %cst_0 : f64 to f32
              %33 = arith.mulf %in_7, %32 : f32
              %34 = arith.cmpf ogt, %in, %33 : f32
              %35 = arith.select %34, %in, %33 : f32
              linalg.yield %35 : f32
            }
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%15, %19 : memref<1x64x128xf32>, memref<1x64xf32>) outs(%15 : memref<1x64x128xf32>) {
            ^bb0(%in: f32, %in_7: f32, %out: f32):
              %32 = arith.truncf %cst_0 : f64 to f32
              %33 = arith.mulf %in, %32 : f32
              %34 = arith.subf %33, %in_7 : f32
              %35 = math.powf %cst, %34 : f32
              linalg.yield %35 : f32
            }
            linalg.fill ins(%cst_1 : f32) outs(%20 : memref<1x64xf32>)
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%15 : memref<1x64x128xf32>) outs(%20 : memref<1x64xf32>) {
            ^bb0(%in: f32, %out: f32):
              %32 = arith.addf %in, %out : f32
              linalg.yield %32 : f32
            }
            linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%18, %19 : memref<1x64xf32>, memref<1x64xf32>) outs(%21 : memref<1x64xf32>) {
            ^bb0(%in: f32, %in_7: f32, %out: f32):
              %32 = arith.subf %in, %in_7 : f32
              %33 = math.powf %cst, %32 : f32
              linalg.yield %33 : f32
            }
            linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%17, %21, %20 : memref<1x64xf32>, memref<1x64xf32>, memref<1x64xf32>) outs(%17 : memref<1x64xf32>) {
            ^bb0(%in: f32, %in_7: f32, %in_8: f32, %out: f32):
              %32 = arith.mulf %in, %in_7 : f32
              %33 = arith.addf %32, %in_8 : f32
              linalg.yield %33 : f32
            }
            %30 = arith.muli %arg7, %c16384 : index
            %31 = arith.addi %25, %30 : index
            %reinterpret_cast_6 = memref.reinterpret_cast %arg1 to offset: [%31], sizes: [1, 128, 128], strides: [524288, 128, 1] : memref<32x4096x128xf32> to memref<1x128x128xf32, strided<[524288, 128, 1], offset: ?>>
            loom.copy %reinterpret_cast_6, %14 src_mem_space @DRAM dst_mem_space @L1, interconnect : [@vertical_links], broadcast : [8, 1] : memref<1x128x128xf32, strided<[524288, 128, 1], offset: ?>>, memref<1x128x128xf32>
            linalg.fill ins(%cst_1 : f32) outs(%24 : memref<1x64x128xf32>)
            linalg.batch_matmul ins(%15, %14 : memref<1x64x128xf32>, memref<1x128x128xf32>) outs(%24 : memref<1x64x128xf32>)
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%24, %23, %21 : memref<1x64x128xf32>, memref<1x64x128xf32>, memref<1x64xf32>) outs(%23 : memref<1x64x128xf32>) {
            ^bb0(%in: f32, %in_7: f32, %in_8: f32, %out: f32):
              %32 = arith.mulf %in_7, %in_8 : f32
              %33 = arith.addf %in, %32 : f32
              linalg.yield %33 : f32
            }
            linalg.copy ins(%19 : memref<1x64xf32>) outs(%18 : memref<1x64xf32>)
          }
          linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%23, %17 : memref<1x64x128xf32>, memref<1x64xf32>) outs(%22 : memref<1x64x128xf32>) {
          ^bb0(%in: f32, %in_5: f32, %out: f32):
            %28 = arith.divf %in, %in_5 : f32
            linalg.yield %28 : f32
          }
          %reinterpret_cast_4 = memref.reinterpret_cast %arg3 to offset: [%27], sizes: [1, 64, 128], strides: [524288, 128, 1] : memref<32x4096x128xf32> to memref<1x64x128xf32, strided<[524288, 128, 1], offset: ?>>
          loom.copy %22, %reinterpret_cast_4 src_mem_space @L1 dst_mem_space @DRAM, interconnect : [], broadcast : [1, 1] : memref<1x64x128xf32>, memref<1x64x128xf32, strided<[524288, 128, 1], offset: ?>>
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
          %15 = loom.alloc [1, 64, 128] on @L1 : memref<1x64x128xf32>
          %16 = loom.alloc [1, 128, 128] on @L1 : memref<1x128x128xf32>
          %17 = loom.alloc [1, 64] on @L1 : memref<1x64xf32>
          %18 = loom.alloc [1, 64] on @L1 : memref<1x64xf32>
          %19 = loom.alloc [1, 64] on @L1 : memref<1x64xf32>
          %20 = loom.alloc [1, 64] on @L1 : memref<1x64xf32>
          %21 = loom.alloc [1, 64] on @L1 : memref<1x64xf32>
          %22 = loom.alloc [1, 64, 128] on @L1 : memref<1x64x128xf32>
          %23 = loom.alloc [1, 64, 128] on @L1 : memref<1x64x128xf32>
          %24 = loom.alloc [1, 64, 128] on @L1 : memref<1x64x128xf32>
          %25 = arith.muli %arg6, %c524288 overflow<nsw> : index
          %26 = arith.muli %13, %c8192 : index
          %27 = arith.addi %25, %26 : index
          %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%27], sizes: [1, 64, 128], strides: [524288, 128, 1] : memref<32x4096x128xf32> to memref<1x64x128xf32, strided<[524288, 128, 1], offset: ?>>
          loom.copy %reinterpret_cast, %22 src_mem_space @DRAM dst_mem_space @L1, interconnect : [], broadcast : [1, 1] : memref<1x64x128xf32, strided<[524288, 128, 1], offset: ?>>, memref<1x64x128xf32>
          linalg.fill ins(%cst_1 : f32) outs(%23 : memref<1x64x128xf32>)
          linalg.fill ins(%cst_2 : f32) outs(%17 : memref<1x64xf32>)
          linalg.fill ins(%cst_3 : f32) outs(%18 : memref<1x64xf32>)
          scf.for %arg7 = %c0 to %c32 step %c1 {
            %28 = arith.muli %arg7, %c128 : index
            %29 = arith.addi %25, %28 : index
            %reinterpret_cast_5 = memref.reinterpret_cast %arg0 to offset: [%29], sizes: [1, 128, 128], strides: [524288, 4096, 1] : memref<32x128x4096xf32> to memref<1x128x128xf32, strided<[524288, 4096, 1], offset: ?>>
            loom.copy %reinterpret_cast_5, %16 src_mem_space @DRAM dst_mem_space @L1, interconnect : [@vertical_links], broadcast : [8, 1] : memref<1x128x128xf32, strided<[524288, 4096, 1], offset: ?>>, memref<1x128x128xf32>
            linalg.fill ins(%cst_1 : f32) outs(%15 : memref<1x64x128xf32>)
            linalg.batch_matmul ins(%22, %16 : memref<1x64x128xf32>, memref<1x128x128xf32>) outs(%15 : memref<1x64x128xf32>)
            linalg.fill ins(%cst_3 : f32) outs(%19 : memref<1x64xf32>)
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%15 : memref<1x64x128xf32>) outs(%19 : memref<1x64xf32>) {
            ^bb0(%in: f32, %out: f32):
              %32 = arith.maximumf %in, %out : f32
              linalg.yield %32 : f32
            }
            linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%18, %19 : memref<1x64xf32>, memref<1x64xf32>) outs(%19 : memref<1x64xf32>) {
            ^bb0(%in: f32, %in_7: f32, %out: f32):
              %32 = arith.truncf %cst_0 : f64 to f32
              %33 = arith.mulf %in_7, %32 : f32
              %34 = arith.cmpf ogt, %in, %33 : f32
              %35 = arith.select %34, %in, %33 : f32
              linalg.yield %35 : f32
            }
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%15, %19 : memref<1x64x128xf32>, memref<1x64xf32>) outs(%15 : memref<1x64x128xf32>) {
            ^bb0(%in: f32, %in_7: f32, %out: f32):
              %32 = arith.truncf %cst_0 : f64 to f32
              %33 = arith.mulf %in, %32 : f32
              %34 = arith.subf %33, %in_7 : f32
              %35 = math.powf %cst, %34 : f32
              linalg.yield %35 : f32
            }
            linalg.fill ins(%cst_1 : f32) outs(%20 : memref<1x64xf32>)
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%15 : memref<1x64x128xf32>) outs(%20 : memref<1x64xf32>) {
            ^bb0(%in: f32, %out: f32):
              %32 = arith.addf %in, %out : f32
              linalg.yield %32 : f32
            }
            linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%18, %19 : memref<1x64xf32>, memref<1x64xf32>) outs(%21 : memref<1x64xf32>) {
            ^bb0(%in: f32, %in_7: f32, %out: f32):
              %32 = arith.subf %in, %in_7 : f32
              %33 = math.powf %cst, %32 : f32
              linalg.yield %33 : f32
            }
            linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%17, %21, %20 : memref<1x64xf32>, memref<1x64xf32>, memref<1x64xf32>) outs(%17 : memref<1x64xf32>) {
            ^bb0(%in: f32, %in_7: f32, %in_8: f32, %out: f32):
              %32 = arith.mulf %in, %in_7 : f32
              %33 = arith.addf %32, %in_8 : f32
              linalg.yield %33 : f32
            }
            %30 = arith.muli %arg7, %c16384 : index
            %31 = arith.addi %25, %30 : index
            %reinterpret_cast_6 = memref.reinterpret_cast %arg1 to offset: [%31], sizes: [1, 128, 128], strides: [524288, 128, 1] : memref<32x4096x128xf32> to memref<1x128x128xf32, strided<[524288, 128, 1], offset: ?>>
            loom.copy %reinterpret_cast_6, %14 src_mem_space @DRAM dst_mem_space @L1, interconnect : [], broadcast : [1, 1] : memref<1x128x128xf32, strided<[524288, 128, 1], offset: ?>>, memref<1x128x128xf32>
            linalg.fill ins(%cst_1 : f32) outs(%24 : memref<1x64x128xf32>)
            linalg.batch_matmul ins(%15, %14 : memref<1x64x128xf32>, memref<1x128x128xf32>) outs(%24 : memref<1x64x128xf32>)
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%24, %23, %21 : memref<1x64x128xf32>, memref<1x64x128xf32>, memref<1x64xf32>) outs(%23 : memref<1x64x128xf32>) {
            ^bb0(%in: f32, %in_7: f32, %in_8: f32, %out: f32):
              %32 = arith.mulf %in_7, %in_8 : f32
              %33 = arith.addf %in, %32 : f32
              linalg.yield %33 : f32
            }
            linalg.copy ins(%19 : memref<1x64xf32>) outs(%18 : memref<1x64xf32>)
          }
          linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%23, %17 : memref<1x64x128xf32>, memref<1x64xf32>) outs(%22 : memref<1x64x128xf32>) {
          ^bb0(%in: f32, %in_5: f32, %out: f32):
            %28 = arith.divf %in, %in_5 : f32
            linalg.yield %28 : f32
          }
          %reinterpret_cast_4 = memref.reinterpret_cast %arg3 to offset: [%27], sizes: [1, 64, 128], strides: [524288, 128, 1] : memref<32x4096x128xf32> to memref<1x64x128xf32, strided<[524288, 128, 1], offset: ?>>
          loom.copy %22, %reinterpret_cast_4 src_mem_space @L1 dst_mem_space @DRAM, interconnect : [], broadcast : [1, 1] : memref<1x64x128xf32>, memref<1x64x128xf32, strided<[524288, 128, 1], offset: ?>>
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
          %15 = loom.alloc [1, 64, 128] on @L1 : memref<1x64x128xf32>
          %16 = loom.alloc [1, 128, 128] on @L1 : memref<1x128x128xf32>
          %17 = loom.alloc [1, 64] on @L1 : memref<1x64xf32>
          %18 = loom.alloc [1, 64] on @L1 : memref<1x64xf32>
          %19 = loom.alloc [1, 64] on @L1 : memref<1x64xf32>
          %20 = loom.alloc [1, 64] on @L1 : memref<1x64xf32>
          %21 = loom.alloc [1, 64] on @L1 : memref<1x64xf32>
          %22 = loom.alloc [1, 64, 128] on @L1 : memref<1x64x128xf32>
          %23 = loom.alloc [1, 64, 128] on @L1 : memref<1x64x128xf32>
          %24 = loom.alloc [1, 64, 128] on @L1 : memref<1x64x128xf32>
          %25 = arith.muli %arg6, %c524288 overflow<nsw> : index
          %26 = arith.muli %13, %c8192 : index
          %27 = arith.addi %25, %26 : index
          %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%27], sizes: [1, 64, 128], strides: [524288, 128, 1] : memref<32x4096x128xf32> to memref<1x64x128xf32, strided<[524288, 128, 1], offset: ?>>
          loom.copy %reinterpret_cast, %22 src_mem_space @DRAM dst_mem_space @L1, interconnect : [], broadcast : [1, 1] : memref<1x64x128xf32, strided<[524288, 128, 1], offset: ?>>, memref<1x64x128xf32>
          linalg.fill ins(%cst_1 : f32) outs(%23 : memref<1x64x128xf32>)
          linalg.fill ins(%cst_2 : f32) outs(%17 : memref<1x64xf32>)
          linalg.fill ins(%cst_3 : f32) outs(%18 : memref<1x64xf32>)
          scf.for %arg7 = %c0 to %c32 step %c1 {
            %28 = arith.muli %arg7, %c128 : index
            %29 = arith.addi %25, %28 : index
            %reinterpret_cast_5 = memref.reinterpret_cast %arg0 to offset: [%29], sizes: [1, 128, 128], strides: [524288, 4096, 1] : memref<32x128x4096xf32> to memref<1x128x128xf32, strided<[524288, 4096, 1], offset: ?>>
            loom.copy %reinterpret_cast_5, %16 src_mem_space @DRAM dst_mem_space @L1, interconnect : [@vertical_links], broadcast : [8, 1] : memref<1x128x128xf32, strided<[524288, 4096, 1], offset: ?>>, memref<1x128x128xf32>
            linalg.fill ins(%cst_1 : f32) outs(%15 : memref<1x64x128xf32>)
            linalg.batch_matmul ins(%22, %16 : memref<1x64x128xf32>, memref<1x128x128xf32>) outs(%15 : memref<1x64x128xf32>)
            linalg.fill ins(%cst_3 : f32) outs(%19 : memref<1x64xf32>)
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%15 : memref<1x64x128xf32>) outs(%19 : memref<1x64xf32>) {
            ^bb0(%in: f32, %out: f32):
              %32 = arith.maximumf %in, %out : f32
              linalg.yield %32 : f32
            }
            linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%18, %19 : memref<1x64xf32>, memref<1x64xf32>) outs(%19 : memref<1x64xf32>) {
            ^bb0(%in: f32, %in_7: f32, %out: f32):
              %32 = arith.truncf %cst_0 : f64 to f32
              %33 = arith.mulf %in_7, %32 : f32
              %34 = arith.cmpf ogt, %in, %33 : f32
              %35 = arith.select %34, %in, %33 : f32
              linalg.yield %35 : f32
            }
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%15, %19 : memref<1x64x128xf32>, memref<1x64xf32>) outs(%15 : memref<1x64x128xf32>) {
            ^bb0(%in: f32, %in_7: f32, %out: f32):
              %32 = arith.truncf %cst_0 : f64 to f32
              %33 = arith.mulf %in, %32 : f32
              %34 = arith.subf %33, %in_7 : f32
              %35 = math.powf %cst, %34 : f32
              linalg.yield %35 : f32
            }
            linalg.fill ins(%cst_1 : f32) outs(%20 : memref<1x64xf32>)
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%15 : memref<1x64x128xf32>) outs(%20 : memref<1x64xf32>) {
            ^bb0(%in: f32, %out: f32):
              %32 = arith.addf %in, %out : f32
              linalg.yield %32 : f32
            }
            linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%18, %19 : memref<1x64xf32>, memref<1x64xf32>) outs(%21 : memref<1x64xf32>) {
            ^bb0(%in: f32, %in_7: f32, %out: f32):
              %32 = arith.subf %in, %in_7 : f32
              %33 = math.powf %cst, %32 : f32
              linalg.yield %33 : f32
            }
            linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%17, %21, %20 : memref<1x64xf32>, memref<1x64xf32>, memref<1x64xf32>) outs(%17 : memref<1x64xf32>) {
            ^bb0(%in: f32, %in_7: f32, %in_8: f32, %out: f32):
              %32 = arith.mulf %in, %in_7 : f32
              %33 = arith.addf %32, %in_8 : f32
              linalg.yield %33 : f32
            }
            %30 = arith.muli %arg7, %c16384 : index
            %31 = arith.addi %25, %30 : index
            %reinterpret_cast_6 = memref.reinterpret_cast %arg1 to offset: [%31], sizes: [1, 128, 128], strides: [524288, 128, 1] : memref<32x4096x128xf32> to memref<1x128x128xf32, strided<[524288, 128, 1], offset: ?>>
            loom.copy %reinterpret_cast_6, %14 src_mem_space @DRAM dst_mem_space @L1, interconnect : [@horizontal_links, @vertical_links], broadcast : [8, 8] : memref<1x128x128xf32, strided<[524288, 128, 1], offset: ?>>, memref<1x128x128xf32>
            linalg.fill ins(%cst_1 : f32) outs(%24 : memref<1x64x128xf32>)
            linalg.batch_matmul ins(%15, %14 : memref<1x64x128xf32>, memref<1x128x128xf32>) outs(%24 : memref<1x64x128xf32>)
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%24, %23, %21 : memref<1x64x128xf32>, memref<1x64x128xf32>, memref<1x64xf32>) outs(%23 : memref<1x64x128xf32>) {
            ^bb0(%in: f32, %in_7: f32, %in_8: f32, %out: f32):
              %32 = arith.mulf %in_7, %in_8 : f32
              %33 = arith.addf %in, %32 : f32
              linalg.yield %33 : f32
            }
            linalg.copy ins(%19 : memref<1x64xf32>) outs(%18 : memref<1x64xf32>)
          }
          linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%23, %17 : memref<1x64x128xf32>, memref<1x64xf32>) outs(%22 : memref<1x64x128xf32>) {
          ^bb0(%in: f32, %in_5: f32, %out: f32):
            %28 = arith.divf %in, %in_5 : f32
            linalg.yield %28 : f32
          }
          %reinterpret_cast_4 = memref.reinterpret_cast %arg3 to offset: [%27], sizes: [1, 64, 128], strides: [524288, 128, 1] : memref<32x4096x128xf32> to memref<1x64x128xf32, strided<[524288, 128, 1], offset: ?>>
          loom.copy %22, %reinterpret_cast_4 src_mem_space @L1 dst_mem_space @DRAM, interconnect : [], broadcast : [1, 1] : memref<1x64x128xf32>, memref<1x64x128xf32, strided<[524288, 128, 1], offset: ?>>
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
          %15 = loom.alloc [1, 64, 128] on @L1 : memref<1x64x128xf32>
          %16 = loom.alloc [1, 128, 128] on @L1 : memref<1x128x128xf32>
          %17 = loom.alloc [1, 64] on @L1 : memref<1x64xf32>
          %18 = loom.alloc [1, 64] on @L1 : memref<1x64xf32>
          %19 = loom.alloc [1, 64] on @L1 : memref<1x64xf32>
          %20 = loom.alloc [1, 64] on @L1 : memref<1x64xf32>
          %21 = loom.alloc [1, 64] on @L1 : memref<1x64xf32>
          %22 = loom.alloc [1, 64, 128] on @L1 : memref<1x64x128xf32>
          %23 = loom.alloc [1, 64, 128] on @L1 : memref<1x64x128xf32>
          %24 = loom.alloc [1, 64, 128] on @L1 : memref<1x64x128xf32>
          %25 = arith.muli %arg6, %c524288 overflow<nsw> : index
          %26 = arith.muli %13, %c8192 : index
          %27 = arith.addi %25, %26 : index
          %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%27], sizes: [1, 64, 128], strides: [524288, 128, 1] : memref<32x4096x128xf32> to memref<1x64x128xf32, strided<[524288, 128, 1], offset: ?>>
          loom.copy %reinterpret_cast, %22 src_mem_space @DRAM dst_mem_space @L1, interconnect : [], broadcast : [1, 1] : memref<1x64x128xf32, strided<[524288, 128, 1], offset: ?>>, memref<1x64x128xf32>
          linalg.fill ins(%cst_1 : f32) outs(%23 : memref<1x64x128xf32>)
          linalg.fill ins(%cst_2 : f32) outs(%17 : memref<1x64xf32>)
          linalg.fill ins(%cst_3 : f32) outs(%18 : memref<1x64xf32>)
          scf.for %arg7 = %c0 to %c32 step %c1 {
            %28 = arith.muli %arg7, %c128 : index
            %29 = arith.addi %25, %28 : index
            %reinterpret_cast_5 = memref.reinterpret_cast %arg0 to offset: [%29], sizes: [1, 128, 128], strides: [524288, 4096, 1] : memref<32x128x4096xf32> to memref<1x128x128xf32, strided<[524288, 4096, 1], offset: ?>>
            loom.copy %reinterpret_cast_5, %16 src_mem_space @DRAM dst_mem_space @L1, interconnect : [@vertical_links], broadcast : [8, 1] : memref<1x128x128xf32, strided<[524288, 4096, 1], offset: ?>>, memref<1x128x128xf32>
            linalg.fill ins(%cst_1 : f32) outs(%15 : memref<1x64x128xf32>)
            linalg.batch_matmul ins(%22, %16 : memref<1x64x128xf32>, memref<1x128x128xf32>) outs(%15 : memref<1x64x128xf32>)
            linalg.fill ins(%cst_3 : f32) outs(%19 : memref<1x64xf32>)
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%15 : memref<1x64x128xf32>) outs(%19 : memref<1x64xf32>) {
            ^bb0(%in: f32, %out: f32):
              %32 = arith.maximumf %in, %out : f32
              linalg.yield %32 : f32
            }
            linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%18, %19 : memref<1x64xf32>, memref<1x64xf32>) outs(%19 : memref<1x64xf32>) {
            ^bb0(%in: f32, %in_7: f32, %out: f32):
              %32 = arith.truncf %cst_0 : f64 to f32
              %33 = arith.mulf %in_7, %32 : f32
              %34 = arith.cmpf ogt, %in, %33 : f32
              %35 = arith.select %34, %in, %33 : f32
              linalg.yield %35 : f32
            }
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%15, %19 : memref<1x64x128xf32>, memref<1x64xf32>) outs(%15 : memref<1x64x128xf32>) {
            ^bb0(%in: f32, %in_7: f32, %out: f32):
              %32 = arith.truncf %cst_0 : f64 to f32
              %33 = arith.mulf %in, %32 : f32
              %34 = arith.subf %33, %in_7 : f32
              %35 = math.powf %cst, %34 : f32
              linalg.yield %35 : f32
            }
            linalg.fill ins(%cst_1 : f32) outs(%20 : memref<1x64xf32>)
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%15 : memref<1x64x128xf32>) outs(%20 : memref<1x64xf32>) {
            ^bb0(%in: f32, %out: f32):
              %32 = arith.addf %in, %out : f32
              linalg.yield %32 : f32
            }
            linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%18, %19 : memref<1x64xf32>, memref<1x64xf32>) outs(%21 : memref<1x64xf32>) {
            ^bb0(%in: f32, %in_7: f32, %out: f32):
              %32 = arith.subf %in, %in_7 : f32
              %33 = math.powf %cst, %32 : f32
              linalg.yield %33 : f32
            }
            linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%17, %21, %20 : memref<1x64xf32>, memref<1x64xf32>, memref<1x64xf32>) outs(%17 : memref<1x64xf32>) {
            ^bb0(%in: f32, %in_7: f32, %in_8: f32, %out: f32):
              %32 = arith.mulf %in, %in_7 : f32
              %33 = arith.addf %32, %in_8 : f32
              linalg.yield %33 : f32
            }
            %30 = arith.muli %arg7, %c16384 : index
            %31 = arith.addi %25, %30 : index
            %reinterpret_cast_6 = memref.reinterpret_cast %arg1 to offset: [%31], sizes: [1, 128, 128], strides: [524288, 128, 1] : memref<32x4096x128xf32> to memref<1x128x128xf32, strided<[524288, 128, 1], offset: ?>>
            loom.copy %reinterpret_cast_6, %14 src_mem_space @DRAM dst_mem_space @L1, interconnect : [@horizontal_links], broadcast : [1, 8] : memref<1x128x128xf32, strided<[524288, 128, 1], offset: ?>>, memref<1x128x128xf32>
            linalg.fill ins(%cst_1 : f32) outs(%24 : memref<1x64x128xf32>)
            linalg.batch_matmul ins(%15, %14 : memref<1x64x128xf32>, memref<1x128x128xf32>) outs(%24 : memref<1x64x128xf32>)
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%24, %23, %21 : memref<1x64x128xf32>, memref<1x64x128xf32>, memref<1x64xf32>) outs(%23 : memref<1x64x128xf32>) {
            ^bb0(%in: f32, %in_7: f32, %in_8: f32, %out: f32):
              %32 = arith.mulf %in_7, %in_8 : f32
              %33 = arith.addf %in, %32 : f32
              linalg.yield %33 : f32
            }
            linalg.copy ins(%19 : memref<1x64xf32>) outs(%18 : memref<1x64xf32>)
          }
          linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%23, %17 : memref<1x64x128xf32>, memref<1x64xf32>) outs(%22 : memref<1x64x128xf32>) {
          ^bb0(%in: f32, %in_5: f32, %out: f32):
            %28 = arith.divf %in, %in_5 : f32
            linalg.yield %28 : f32
          }
          %reinterpret_cast_4 = memref.reinterpret_cast %arg3 to offset: [%27], sizes: [1, 64, 128], strides: [524288, 128, 1] : memref<32x4096x128xf32> to memref<1x64x128xf32, strided<[524288, 128, 1], offset: ?>>
          loom.copy %22, %reinterpret_cast_4 src_mem_space @L1 dst_mem_space @DRAM, interconnect : [], broadcast : [1, 1] : memref<1x64x128xf32>, memref<1x64x128xf32, strided<[524288, 128, 1], offset: ?>>
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
          %15 = loom.alloc [1, 64, 128] on @L1 : memref<1x64x128xf32>
          %16 = loom.alloc [1, 128, 128] on @L1 : memref<1x128x128xf32>
          %17 = loom.alloc [1, 64] on @L1 : memref<1x64xf32>
          %18 = loom.alloc [1, 64] on @L1 : memref<1x64xf32>
          %19 = loom.alloc [1, 64] on @L1 : memref<1x64xf32>
          %20 = loom.alloc [1, 64] on @L1 : memref<1x64xf32>
          %21 = loom.alloc [1, 64] on @L1 : memref<1x64xf32>
          %22 = loom.alloc [1, 64, 128] on @L1 : memref<1x64x128xf32>
          %23 = loom.alloc [1, 64, 128] on @L1 : memref<1x64x128xf32>
          %24 = loom.alloc [1, 64, 128] on @L1 : memref<1x64x128xf32>
          %25 = arith.muli %arg6, %c524288 overflow<nsw> : index
          %26 = arith.muli %13, %c8192 : index
          %27 = arith.addi %25, %26 : index
          %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%27], sizes: [1, 64, 128], strides: [524288, 128, 1] : memref<32x4096x128xf32> to memref<1x64x128xf32, strided<[524288, 128, 1], offset: ?>>
          loom.copy %reinterpret_cast, %22 src_mem_space @DRAM dst_mem_space @L1, interconnect : [], broadcast : [1, 1] : memref<1x64x128xf32, strided<[524288, 128, 1], offset: ?>>, memref<1x64x128xf32>
          linalg.fill ins(%cst_1 : f32) outs(%23 : memref<1x64x128xf32>)
          linalg.fill ins(%cst_2 : f32) outs(%17 : memref<1x64xf32>)
          linalg.fill ins(%cst_3 : f32) outs(%18 : memref<1x64xf32>)
          scf.for %arg7 = %c0 to %c32 step %c1 {
            %28 = arith.muli %arg7, %c128 : index
            %29 = arith.addi %25, %28 : index
            %reinterpret_cast_5 = memref.reinterpret_cast %arg0 to offset: [%29], sizes: [1, 128, 128], strides: [524288, 4096, 1] : memref<32x128x4096xf32> to memref<1x128x128xf32, strided<[524288, 4096, 1], offset: ?>>
            loom.copy %reinterpret_cast_5, %16 src_mem_space @DRAM dst_mem_space @L1, interconnect : [@vertical_links], broadcast : [8, 1] : memref<1x128x128xf32, strided<[524288, 4096, 1], offset: ?>>, memref<1x128x128xf32>
            linalg.fill ins(%cst_1 : f32) outs(%15 : memref<1x64x128xf32>)
            linalg.batch_matmul ins(%22, %16 : memref<1x64x128xf32>, memref<1x128x128xf32>) outs(%15 : memref<1x64x128xf32>)
            linalg.fill ins(%cst_3 : f32) outs(%19 : memref<1x64xf32>)
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%15 : memref<1x64x128xf32>) outs(%19 : memref<1x64xf32>) {
            ^bb0(%in: f32, %out: f32):
              %32 = arith.maximumf %in, %out : f32
              linalg.yield %32 : f32
            }
            linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%18, %19 : memref<1x64xf32>, memref<1x64xf32>) outs(%19 : memref<1x64xf32>) {
            ^bb0(%in: f32, %in_7: f32, %out: f32):
              %32 = arith.truncf %cst_0 : f64 to f32
              %33 = arith.mulf %in_7, %32 : f32
              %34 = arith.cmpf ogt, %in, %33 : f32
              %35 = arith.select %34, %in, %33 : f32
              linalg.yield %35 : f32
            }
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%15, %19 : memref<1x64x128xf32>, memref<1x64xf32>) outs(%15 : memref<1x64x128xf32>) {
            ^bb0(%in: f32, %in_7: f32, %out: f32):
              %32 = arith.truncf %cst_0 : f64 to f32
              %33 = arith.mulf %in, %32 : f32
              %34 = arith.subf %33, %in_7 : f32
              %35 = math.powf %cst, %34 : f32
              linalg.yield %35 : f32
            }
            linalg.fill ins(%cst_1 : f32) outs(%20 : memref<1x64xf32>)
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%15 : memref<1x64x128xf32>) outs(%20 : memref<1x64xf32>) {
            ^bb0(%in: f32, %out: f32):
              %32 = arith.addf %in, %out : f32
              linalg.yield %32 : f32
            }
            linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%18, %19 : memref<1x64xf32>, memref<1x64xf32>) outs(%21 : memref<1x64xf32>) {
            ^bb0(%in: f32, %in_7: f32, %out: f32):
              %32 = arith.subf %in, %in_7 : f32
              %33 = math.powf %cst, %32 : f32
              linalg.yield %33 : f32
            }
            linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%17, %21, %20 : memref<1x64xf32>, memref<1x64xf32>, memref<1x64xf32>) outs(%17 : memref<1x64xf32>) {
            ^bb0(%in: f32, %in_7: f32, %in_8: f32, %out: f32):
              %32 = arith.mulf %in, %in_7 : f32
              %33 = arith.addf %32, %in_8 : f32
              linalg.yield %33 : f32
            }
            %30 = arith.muli %arg7, %c16384 : index
            %31 = arith.addi %25, %30 : index
            %reinterpret_cast_6 = memref.reinterpret_cast %arg1 to offset: [%31], sizes: [1, 128, 128], strides: [524288, 128, 1] : memref<32x4096x128xf32> to memref<1x128x128xf32, strided<[524288, 128, 1], offset: ?>>
            loom.copy %reinterpret_cast_6, %14 src_mem_space @DRAM dst_mem_space @L1, interconnect : [@vertical_links], broadcast : [8, 1] : memref<1x128x128xf32, strided<[524288, 128, 1], offset: ?>>, memref<1x128x128xf32>
            linalg.fill ins(%cst_1 : f32) outs(%24 : memref<1x64x128xf32>)
            linalg.batch_matmul ins(%15, %14 : memref<1x64x128xf32>, memref<1x128x128xf32>) outs(%24 : memref<1x64x128xf32>)
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%24, %23, %21 : memref<1x64x128xf32>, memref<1x64x128xf32>, memref<1x64xf32>) outs(%23 : memref<1x64x128xf32>) {
            ^bb0(%in: f32, %in_7: f32, %in_8: f32, %out: f32):
              %32 = arith.mulf %in_7, %in_8 : f32
              %33 = arith.addf %in, %32 : f32
              linalg.yield %33 : f32
            }
            linalg.copy ins(%19 : memref<1x64xf32>) outs(%18 : memref<1x64xf32>)
          }
          linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%23, %17 : memref<1x64x128xf32>, memref<1x64xf32>) outs(%22 : memref<1x64x128xf32>) {
          ^bb0(%in: f32, %in_5: f32, %out: f32):
            %28 = arith.divf %in, %in_5 : f32
            linalg.yield %28 : f32
          }
          %reinterpret_cast_4 = memref.reinterpret_cast %arg3 to offset: [%27], sizes: [1, 64, 128], strides: [524288, 128, 1] : memref<32x4096x128xf32> to memref<1x64x128xf32, strided<[524288, 128, 1], offset: ?>>
          loom.copy %22, %reinterpret_cast_4 src_mem_space @L1 dst_mem_space @DRAM, interconnect : [], broadcast : [1, 1] : memref<1x64x128xf32>, memref<1x64x128xf32, strided<[524288, 128, 1], offset: ?>>
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
          %15 = loom.alloc [1, 64, 128] on @L1 : memref<1x64x128xf32>
          %16 = loom.alloc [1, 128, 128] on @L1 : memref<1x128x128xf32>
          %17 = loom.alloc [1, 64] on @L1 : memref<1x64xf32>
          %18 = loom.alloc [1, 64] on @L1 : memref<1x64xf32>
          %19 = loom.alloc [1, 64] on @L1 : memref<1x64xf32>
          %20 = loom.alloc [1, 64] on @L1 : memref<1x64xf32>
          %21 = loom.alloc [1, 64] on @L1 : memref<1x64xf32>
          %22 = loom.alloc [1, 64, 128] on @L1 : memref<1x64x128xf32>
          %23 = loom.alloc [1, 64, 128] on @L1 : memref<1x64x128xf32>
          %24 = loom.alloc [1, 64, 128] on @L1 : memref<1x64x128xf32>
          %25 = arith.muli %arg6, %c524288 overflow<nsw> : index
          %26 = arith.muli %13, %c8192 : index
          %27 = arith.addi %25, %26 : index
          %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%27], sizes: [1, 64, 128], strides: [524288, 128, 1] : memref<32x4096x128xf32> to memref<1x64x128xf32, strided<[524288, 128, 1], offset: ?>>
          loom.copy %reinterpret_cast, %22 src_mem_space @DRAM dst_mem_space @L1, interconnect : [], broadcast : [1, 1] : memref<1x64x128xf32, strided<[524288, 128, 1], offset: ?>>, memref<1x64x128xf32>
          linalg.fill ins(%cst_1 : f32) outs(%23 : memref<1x64x128xf32>)
          linalg.fill ins(%cst_2 : f32) outs(%17 : memref<1x64xf32>)
          linalg.fill ins(%cst_3 : f32) outs(%18 : memref<1x64xf32>)
          scf.for %arg7 = %c0 to %c32 step %c1 {
            %28 = arith.muli %arg7, %c128 : index
            %29 = arith.addi %25, %28 : index
            %reinterpret_cast_5 = memref.reinterpret_cast %arg0 to offset: [%29], sizes: [1, 128, 128], strides: [524288, 4096, 1] : memref<32x128x4096xf32> to memref<1x128x128xf32, strided<[524288, 4096, 1], offset: ?>>
            loom.copy %reinterpret_cast_5, %16 src_mem_space @DRAM dst_mem_space @L1, interconnect : [], broadcast : [1, 1] : memref<1x128x128xf32, strided<[524288, 4096, 1], offset: ?>>, memref<1x128x128xf32>
            linalg.fill ins(%cst_1 : f32) outs(%15 : memref<1x64x128xf32>)
            linalg.batch_matmul ins(%22, %16 : memref<1x64x128xf32>, memref<1x128x128xf32>) outs(%15 : memref<1x64x128xf32>)
            linalg.fill ins(%cst_3 : f32) outs(%19 : memref<1x64xf32>)
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%15 : memref<1x64x128xf32>) outs(%19 : memref<1x64xf32>) {
            ^bb0(%in: f32, %out: f32):
              %32 = arith.maximumf %in, %out : f32
              linalg.yield %32 : f32
            }
            linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%18, %19 : memref<1x64xf32>, memref<1x64xf32>) outs(%19 : memref<1x64xf32>) {
            ^bb0(%in: f32, %in_7: f32, %out: f32):
              %32 = arith.truncf %cst_0 : f64 to f32
              %33 = arith.mulf %in_7, %32 : f32
              %34 = arith.cmpf ogt, %in, %33 : f32
              %35 = arith.select %34, %in, %33 : f32
              linalg.yield %35 : f32
            }
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%15, %19 : memref<1x64x128xf32>, memref<1x64xf32>) outs(%15 : memref<1x64x128xf32>) {
            ^bb0(%in: f32, %in_7: f32, %out: f32):
              %32 = arith.truncf %cst_0 : f64 to f32
              %33 = arith.mulf %in, %32 : f32
              %34 = arith.subf %33, %in_7 : f32
              %35 = math.powf %cst, %34 : f32
              linalg.yield %35 : f32
            }
            linalg.fill ins(%cst_1 : f32) outs(%20 : memref<1x64xf32>)
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%15 : memref<1x64x128xf32>) outs(%20 : memref<1x64xf32>) {
            ^bb0(%in: f32, %out: f32):
              %32 = arith.addf %in, %out : f32
              linalg.yield %32 : f32
            }
            linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%18, %19 : memref<1x64xf32>, memref<1x64xf32>) outs(%21 : memref<1x64xf32>) {
            ^bb0(%in: f32, %in_7: f32, %out: f32):
              %32 = arith.subf %in, %in_7 : f32
              %33 = math.powf %cst, %32 : f32
              linalg.yield %33 : f32
            }
            linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%17, %21, %20 : memref<1x64xf32>, memref<1x64xf32>, memref<1x64xf32>) outs(%17 : memref<1x64xf32>) {
            ^bb0(%in: f32, %in_7: f32, %in_8: f32, %out: f32):
              %32 = arith.mulf %in, %in_7 : f32
              %33 = arith.addf %32, %in_8 : f32
              linalg.yield %33 : f32
            }
            %30 = arith.muli %arg7, %c16384 : index
            %31 = arith.addi %25, %30 : index
            %reinterpret_cast_6 = memref.reinterpret_cast %arg1 to offset: [%31], sizes: [1, 128, 128], strides: [524288, 128, 1] : memref<32x4096x128xf32> to memref<1x128x128xf32, strided<[524288, 128, 1], offset: ?>>
            loom.copy %reinterpret_cast_6, %14 src_mem_space @DRAM dst_mem_space @L1, interconnect : [], broadcast : [1, 1] : memref<1x128x128xf32, strided<[524288, 128, 1], offset: ?>>, memref<1x128x128xf32>
            linalg.fill ins(%cst_1 : f32) outs(%24 : memref<1x64x128xf32>)
            linalg.batch_matmul ins(%15, %14 : memref<1x64x128xf32>, memref<1x128x128xf32>) outs(%24 : memref<1x64x128xf32>)
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%24, %23, %21 : memref<1x64x128xf32>, memref<1x64x128xf32>, memref<1x64xf32>) outs(%23 : memref<1x64x128xf32>) {
            ^bb0(%in: f32, %in_7: f32, %in_8: f32, %out: f32):
              %32 = arith.mulf %in_7, %in_8 : f32
              %33 = arith.addf %in, %32 : f32
              linalg.yield %33 : f32
            }
            linalg.copy ins(%19 : memref<1x64xf32>) outs(%18 : memref<1x64xf32>)
          }
          linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%23, %17 : memref<1x64x128xf32>, memref<1x64xf32>) outs(%22 : memref<1x64x128xf32>) {
          ^bb0(%in: f32, %in_5: f32, %out: f32):
            %28 = arith.divf %in, %in_5 : f32
            linalg.yield %28 : f32
          }
          %reinterpret_cast_4 = memref.reinterpret_cast %arg3 to offset: [%27], sizes: [1, 64, 128], strides: [524288, 128, 1] : memref<32x4096x128xf32> to memref<1x64x128xf32, strided<[524288, 128, 1], offset: ?>>
          loom.copy %22, %reinterpret_cast_4 src_mem_space @L1 dst_mem_space @DRAM, interconnect : [], broadcast : [1, 1] : memref<1x64x128xf32>, memref<1x64x128xf32, strided<[524288, 128, 1], offset: ?>>
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
          %15 = loom.alloc [1, 64, 128] on @L1 : memref<1x64x128xf32>
          %16 = loom.alloc [1, 128, 128] on @L1 : memref<1x128x128xf32>
          %17 = loom.alloc [1, 64] on @L1 : memref<1x64xf32>
          %18 = loom.alloc [1, 64] on @L1 : memref<1x64xf32>
          %19 = loom.alloc [1, 64] on @L1 : memref<1x64xf32>
          %20 = loom.alloc [1, 64] on @L1 : memref<1x64xf32>
          %21 = loom.alloc [1, 64] on @L1 : memref<1x64xf32>
          %22 = loom.alloc [1, 64, 128] on @L1 : memref<1x64x128xf32>
          %23 = loom.alloc [1, 64, 128] on @L1 : memref<1x64x128xf32>
          %24 = loom.alloc [1, 64, 128] on @L1 : memref<1x64x128xf32>
          %25 = arith.muli %arg6, %c524288 overflow<nsw> : index
          %26 = arith.muli %13, %c8192 : index
          %27 = arith.addi %25, %26 : index
          %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%27], sizes: [1, 64, 128], strides: [524288, 128, 1] : memref<32x4096x128xf32> to memref<1x64x128xf32, strided<[524288, 128, 1], offset: ?>>
          loom.copy %reinterpret_cast, %22 src_mem_space @DRAM dst_mem_space @L1, interconnect : [], broadcast : [1, 1] : memref<1x64x128xf32, strided<[524288, 128, 1], offset: ?>>, memref<1x64x128xf32>
          linalg.fill ins(%cst_1 : f32) outs(%23 : memref<1x64x128xf32>)
          linalg.fill ins(%cst_2 : f32) outs(%17 : memref<1x64xf32>)
          linalg.fill ins(%cst_3 : f32) outs(%18 : memref<1x64xf32>)
          scf.for %arg7 = %c0 to %c32 step %c1 {
            %28 = arith.muli %arg7, %c128 : index
            %29 = arith.addi %25, %28 : index
            %reinterpret_cast_5 = memref.reinterpret_cast %arg0 to offset: [%29], sizes: [1, 128, 128], strides: [524288, 4096, 1] : memref<32x128x4096xf32> to memref<1x128x128xf32, strided<[524288, 4096, 1], offset: ?>>
            loom.copy %reinterpret_cast_5, %16 src_mem_space @DRAM dst_mem_space @L1, interconnect : [], broadcast : [1, 1] : memref<1x128x128xf32, strided<[524288, 4096, 1], offset: ?>>, memref<1x128x128xf32>
            linalg.fill ins(%cst_1 : f32) outs(%15 : memref<1x64x128xf32>)
            linalg.batch_matmul ins(%22, %16 : memref<1x64x128xf32>, memref<1x128x128xf32>) outs(%15 : memref<1x64x128xf32>)
            linalg.fill ins(%cst_3 : f32) outs(%19 : memref<1x64xf32>)
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%15 : memref<1x64x128xf32>) outs(%19 : memref<1x64xf32>) {
            ^bb0(%in: f32, %out: f32):
              %32 = arith.maximumf %in, %out : f32
              linalg.yield %32 : f32
            }
            linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%18, %19 : memref<1x64xf32>, memref<1x64xf32>) outs(%19 : memref<1x64xf32>) {
            ^bb0(%in: f32, %in_7: f32, %out: f32):
              %32 = arith.truncf %cst_0 : f64 to f32
              %33 = arith.mulf %in_7, %32 : f32
              %34 = arith.cmpf ogt, %in, %33 : f32
              %35 = arith.select %34, %in, %33 : f32
              linalg.yield %35 : f32
            }
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%15, %19 : memref<1x64x128xf32>, memref<1x64xf32>) outs(%15 : memref<1x64x128xf32>) {
            ^bb0(%in: f32, %in_7: f32, %out: f32):
              %32 = arith.truncf %cst_0 : f64 to f32
              %33 = arith.mulf %in, %32 : f32
              %34 = arith.subf %33, %in_7 : f32
              %35 = math.powf %cst, %34 : f32
              linalg.yield %35 : f32
            }
            linalg.fill ins(%cst_1 : f32) outs(%20 : memref<1x64xf32>)
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%15 : memref<1x64x128xf32>) outs(%20 : memref<1x64xf32>) {
            ^bb0(%in: f32, %out: f32):
              %32 = arith.addf %in, %out : f32
              linalg.yield %32 : f32
            }
            linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%18, %19 : memref<1x64xf32>, memref<1x64xf32>) outs(%21 : memref<1x64xf32>) {
            ^bb0(%in: f32, %in_7: f32, %out: f32):
              %32 = arith.subf %in, %in_7 : f32
              %33 = math.powf %cst, %32 : f32
              linalg.yield %33 : f32
            }
            linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%17, %21, %20 : memref<1x64xf32>, memref<1x64xf32>, memref<1x64xf32>) outs(%17 : memref<1x64xf32>) {
            ^bb0(%in: f32, %in_7: f32, %in_8: f32, %out: f32):
              %32 = arith.mulf %in, %in_7 : f32
              %33 = arith.addf %32, %in_8 : f32
              linalg.yield %33 : f32
            }
            %30 = arith.muli %arg7, %c16384 : index
            %31 = arith.addi %25, %30 : index
            %reinterpret_cast_6 = memref.reinterpret_cast %arg1 to offset: [%31], sizes: [1, 128, 128], strides: [524288, 128, 1] : memref<32x4096x128xf32> to memref<1x128x128xf32, strided<[524288, 128, 1], offset: ?>>
            loom.copy %reinterpret_cast_6, %14 src_mem_space @DRAM dst_mem_space @L1, interconnect : [@horizontal_links, @vertical_links], broadcast : [8, 8] : memref<1x128x128xf32, strided<[524288, 128, 1], offset: ?>>, memref<1x128x128xf32>
            linalg.fill ins(%cst_1 : f32) outs(%24 : memref<1x64x128xf32>)
            linalg.batch_matmul ins(%15, %14 : memref<1x64x128xf32>, memref<1x128x128xf32>) outs(%24 : memref<1x64x128xf32>)
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%24, %23, %21 : memref<1x64x128xf32>, memref<1x64x128xf32>, memref<1x64xf32>) outs(%23 : memref<1x64x128xf32>) {
            ^bb0(%in: f32, %in_7: f32, %in_8: f32, %out: f32):
              %32 = arith.mulf %in_7, %in_8 : f32
              %33 = arith.addf %in, %32 : f32
              linalg.yield %33 : f32
            }
            linalg.copy ins(%19 : memref<1x64xf32>) outs(%18 : memref<1x64xf32>)
          }
          linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%23, %17 : memref<1x64x128xf32>, memref<1x64xf32>) outs(%22 : memref<1x64x128xf32>) {
          ^bb0(%in: f32, %in_5: f32, %out: f32):
            %28 = arith.divf %in, %in_5 : f32
            linalg.yield %28 : f32
          }
          %reinterpret_cast_4 = memref.reinterpret_cast %arg3 to offset: [%27], sizes: [1, 64, 128], strides: [524288, 128, 1] : memref<32x4096x128xf32> to memref<1x64x128xf32, strided<[524288, 128, 1], offset: ?>>
          loom.copy %22, %reinterpret_cast_4 src_mem_space @L1 dst_mem_space @DRAM, interconnect : [], broadcast : [1, 1] : memref<1x64x128xf32>, memref<1x64x128xf32, strided<[524288, 128, 1], offset: ?>>
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
          %15 = loom.alloc [1, 64, 128] on @L1 : memref<1x64x128xf32>
          %16 = loom.alloc [1, 128, 128] on @L1 : memref<1x128x128xf32>
          %17 = loom.alloc [1, 64] on @L1 : memref<1x64xf32>
          %18 = loom.alloc [1, 64] on @L1 : memref<1x64xf32>
          %19 = loom.alloc [1, 64] on @L1 : memref<1x64xf32>
          %20 = loom.alloc [1, 64] on @L1 : memref<1x64xf32>
          %21 = loom.alloc [1, 64] on @L1 : memref<1x64xf32>
          %22 = loom.alloc [1, 64, 128] on @L1 : memref<1x64x128xf32>
          %23 = loom.alloc [1, 64, 128] on @L1 : memref<1x64x128xf32>
          %24 = loom.alloc [1, 64, 128] on @L1 : memref<1x64x128xf32>
          %25 = arith.muli %arg6, %c524288 overflow<nsw> : index
          %26 = arith.muli %13, %c8192 : index
          %27 = arith.addi %25, %26 : index
          %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%27], sizes: [1, 64, 128], strides: [524288, 128, 1] : memref<32x4096x128xf32> to memref<1x64x128xf32, strided<[524288, 128, 1], offset: ?>>
          loom.copy %reinterpret_cast, %22 src_mem_space @DRAM dst_mem_space @L1, interconnect : [], broadcast : [1, 1] : memref<1x64x128xf32, strided<[524288, 128, 1], offset: ?>>, memref<1x64x128xf32>
          linalg.fill ins(%cst_1 : f32) outs(%23 : memref<1x64x128xf32>)
          linalg.fill ins(%cst_2 : f32) outs(%17 : memref<1x64xf32>)
          linalg.fill ins(%cst_3 : f32) outs(%18 : memref<1x64xf32>)
          scf.for %arg7 = %c0 to %c32 step %c1 {
            %28 = arith.muli %arg7, %c128 : index
            %29 = arith.addi %25, %28 : index
            %reinterpret_cast_5 = memref.reinterpret_cast %arg0 to offset: [%29], sizes: [1, 128, 128], strides: [524288, 4096, 1] : memref<32x128x4096xf32> to memref<1x128x128xf32, strided<[524288, 4096, 1], offset: ?>>
            loom.copy %reinterpret_cast_5, %16 src_mem_space @DRAM dst_mem_space @L1, interconnect : [], broadcast : [1, 1] : memref<1x128x128xf32, strided<[524288, 4096, 1], offset: ?>>, memref<1x128x128xf32>
            linalg.fill ins(%cst_1 : f32) outs(%15 : memref<1x64x128xf32>)
            linalg.batch_matmul ins(%22, %16 : memref<1x64x128xf32>, memref<1x128x128xf32>) outs(%15 : memref<1x64x128xf32>)
            linalg.fill ins(%cst_3 : f32) outs(%19 : memref<1x64xf32>)
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%15 : memref<1x64x128xf32>) outs(%19 : memref<1x64xf32>) {
            ^bb0(%in: f32, %out: f32):
              %32 = arith.maximumf %in, %out : f32
              linalg.yield %32 : f32
            }
            linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%18, %19 : memref<1x64xf32>, memref<1x64xf32>) outs(%19 : memref<1x64xf32>) {
            ^bb0(%in: f32, %in_7: f32, %out: f32):
              %32 = arith.truncf %cst_0 : f64 to f32
              %33 = arith.mulf %in_7, %32 : f32
              %34 = arith.cmpf ogt, %in, %33 : f32
              %35 = arith.select %34, %in, %33 : f32
              linalg.yield %35 : f32
            }
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%15, %19 : memref<1x64x128xf32>, memref<1x64xf32>) outs(%15 : memref<1x64x128xf32>) {
            ^bb0(%in: f32, %in_7: f32, %out: f32):
              %32 = arith.truncf %cst_0 : f64 to f32
              %33 = arith.mulf %in, %32 : f32
              %34 = arith.subf %33, %in_7 : f32
              %35 = math.powf %cst, %34 : f32
              linalg.yield %35 : f32
            }
            linalg.fill ins(%cst_1 : f32) outs(%20 : memref<1x64xf32>)
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%15 : memref<1x64x128xf32>) outs(%20 : memref<1x64xf32>) {
            ^bb0(%in: f32, %out: f32):
              %32 = arith.addf %in, %out : f32
              linalg.yield %32 : f32
            }
            linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%18, %19 : memref<1x64xf32>, memref<1x64xf32>) outs(%21 : memref<1x64xf32>) {
            ^bb0(%in: f32, %in_7: f32, %out: f32):
              %32 = arith.subf %in, %in_7 : f32
              %33 = math.powf %cst, %32 : f32
              linalg.yield %33 : f32
            }
            linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%17, %21, %20 : memref<1x64xf32>, memref<1x64xf32>, memref<1x64xf32>) outs(%17 : memref<1x64xf32>) {
            ^bb0(%in: f32, %in_7: f32, %in_8: f32, %out: f32):
              %32 = arith.mulf %in, %in_7 : f32
              %33 = arith.addf %32, %in_8 : f32
              linalg.yield %33 : f32
            }
            %30 = arith.muli %arg7, %c16384 : index
            %31 = arith.addi %25, %30 : index
            %reinterpret_cast_6 = memref.reinterpret_cast %arg1 to offset: [%31], sizes: [1, 128, 128], strides: [524288, 128, 1] : memref<32x4096x128xf32> to memref<1x128x128xf32, strided<[524288, 128, 1], offset: ?>>
            loom.copy %reinterpret_cast_6, %14 src_mem_space @DRAM dst_mem_space @L1, interconnect : [@horizontal_links], broadcast : [1, 8] : memref<1x128x128xf32, strided<[524288, 128, 1], offset: ?>>, memref<1x128x128xf32>
            linalg.fill ins(%cst_1 : f32) outs(%24 : memref<1x64x128xf32>)
            linalg.batch_matmul ins(%15, %14 : memref<1x64x128xf32>, memref<1x128x128xf32>) outs(%24 : memref<1x64x128xf32>)
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%24, %23, %21 : memref<1x64x128xf32>, memref<1x64x128xf32>, memref<1x64xf32>) outs(%23 : memref<1x64x128xf32>) {
            ^bb0(%in: f32, %in_7: f32, %in_8: f32, %out: f32):
              %32 = arith.mulf %in_7, %in_8 : f32
              %33 = arith.addf %in, %32 : f32
              linalg.yield %33 : f32
            }
            linalg.copy ins(%19 : memref<1x64xf32>) outs(%18 : memref<1x64xf32>)
          }
          linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%23, %17 : memref<1x64x128xf32>, memref<1x64xf32>) outs(%22 : memref<1x64x128xf32>) {
          ^bb0(%in: f32, %in_5: f32, %out: f32):
            %28 = arith.divf %in, %in_5 : f32
            linalg.yield %28 : f32
          }
          %reinterpret_cast_4 = memref.reinterpret_cast %arg3 to offset: [%27], sizes: [1, 64, 128], strides: [524288, 128, 1] : memref<32x4096x128xf32> to memref<1x64x128xf32, strided<[524288, 128, 1], offset: ?>>
          loom.copy %22, %reinterpret_cast_4 src_mem_space @L1 dst_mem_space @DRAM, interconnect : [], broadcast : [1, 1] : memref<1x64x128xf32>, memref<1x64x128xf32, strided<[524288, 128, 1], offset: ?>>
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
          %15 = loom.alloc [1, 64, 128] on @L1 : memref<1x64x128xf32>
          %16 = loom.alloc [1, 128, 128] on @L1 : memref<1x128x128xf32>
          %17 = loom.alloc [1, 64] on @L1 : memref<1x64xf32>
          %18 = loom.alloc [1, 64] on @L1 : memref<1x64xf32>
          %19 = loom.alloc [1, 64] on @L1 : memref<1x64xf32>
          %20 = loom.alloc [1, 64] on @L1 : memref<1x64xf32>
          %21 = loom.alloc [1, 64] on @L1 : memref<1x64xf32>
          %22 = loom.alloc [1, 64, 128] on @L1 : memref<1x64x128xf32>
          %23 = loom.alloc [1, 64, 128] on @L1 : memref<1x64x128xf32>
          %24 = loom.alloc [1, 64, 128] on @L1 : memref<1x64x128xf32>
          %25 = arith.muli %arg6, %c524288 overflow<nsw> : index
          %26 = arith.muli %13, %c8192 : index
          %27 = arith.addi %25, %26 : index
          %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%27], sizes: [1, 64, 128], strides: [524288, 128, 1] : memref<32x4096x128xf32> to memref<1x64x128xf32, strided<[524288, 128, 1], offset: ?>>
          loom.copy %reinterpret_cast, %22 src_mem_space @DRAM dst_mem_space @L1, interconnect : [], broadcast : [1, 1] : memref<1x64x128xf32, strided<[524288, 128, 1], offset: ?>>, memref<1x64x128xf32>
          linalg.fill ins(%cst_1 : f32) outs(%23 : memref<1x64x128xf32>)
          linalg.fill ins(%cst_2 : f32) outs(%17 : memref<1x64xf32>)
          linalg.fill ins(%cst_3 : f32) outs(%18 : memref<1x64xf32>)
          scf.for %arg7 = %c0 to %c32 step %c1 {
            %28 = arith.muli %arg7, %c128 : index
            %29 = arith.addi %25, %28 : index
            %reinterpret_cast_5 = memref.reinterpret_cast %arg0 to offset: [%29], sizes: [1, 128, 128], strides: [524288, 4096, 1] : memref<32x128x4096xf32> to memref<1x128x128xf32, strided<[524288, 4096, 1], offset: ?>>
            loom.copy %reinterpret_cast_5, %16 src_mem_space @DRAM dst_mem_space @L1, interconnect : [], broadcast : [1, 1] : memref<1x128x128xf32, strided<[524288, 4096, 1], offset: ?>>, memref<1x128x128xf32>
            linalg.fill ins(%cst_1 : f32) outs(%15 : memref<1x64x128xf32>)
            linalg.batch_matmul ins(%22, %16 : memref<1x64x128xf32>, memref<1x128x128xf32>) outs(%15 : memref<1x64x128xf32>)
            linalg.fill ins(%cst_3 : f32) outs(%19 : memref<1x64xf32>)
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%15 : memref<1x64x128xf32>) outs(%19 : memref<1x64xf32>) {
            ^bb0(%in: f32, %out: f32):
              %32 = arith.maximumf %in, %out : f32
              linalg.yield %32 : f32
            }
            linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%18, %19 : memref<1x64xf32>, memref<1x64xf32>) outs(%19 : memref<1x64xf32>) {
            ^bb0(%in: f32, %in_7: f32, %out: f32):
              %32 = arith.truncf %cst_0 : f64 to f32
              %33 = arith.mulf %in_7, %32 : f32
              %34 = arith.cmpf ogt, %in, %33 : f32
              %35 = arith.select %34, %in, %33 : f32
              linalg.yield %35 : f32
            }
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%15, %19 : memref<1x64x128xf32>, memref<1x64xf32>) outs(%15 : memref<1x64x128xf32>) {
            ^bb0(%in: f32, %in_7: f32, %out: f32):
              %32 = arith.truncf %cst_0 : f64 to f32
              %33 = arith.mulf %in, %32 : f32
              %34 = arith.subf %33, %in_7 : f32
              %35 = math.powf %cst, %34 : f32
              linalg.yield %35 : f32
            }
            linalg.fill ins(%cst_1 : f32) outs(%20 : memref<1x64xf32>)
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%15 : memref<1x64x128xf32>) outs(%20 : memref<1x64xf32>) {
            ^bb0(%in: f32, %out: f32):
              %32 = arith.addf %in, %out : f32
              linalg.yield %32 : f32
            }
            linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%18, %19 : memref<1x64xf32>, memref<1x64xf32>) outs(%21 : memref<1x64xf32>) {
            ^bb0(%in: f32, %in_7: f32, %out: f32):
              %32 = arith.subf %in, %in_7 : f32
              %33 = math.powf %cst, %32 : f32
              linalg.yield %33 : f32
            }
            linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%17, %21, %20 : memref<1x64xf32>, memref<1x64xf32>, memref<1x64xf32>) outs(%17 : memref<1x64xf32>) {
            ^bb0(%in: f32, %in_7: f32, %in_8: f32, %out: f32):
              %32 = arith.mulf %in, %in_7 : f32
              %33 = arith.addf %32, %in_8 : f32
              linalg.yield %33 : f32
            }
            %30 = arith.muli %arg7, %c16384 : index
            %31 = arith.addi %25, %30 : index
            %reinterpret_cast_6 = memref.reinterpret_cast %arg1 to offset: [%31], sizes: [1, 128, 128], strides: [524288, 128, 1] : memref<32x4096x128xf32> to memref<1x128x128xf32, strided<[524288, 128, 1], offset: ?>>
            loom.copy %reinterpret_cast_6, %14 src_mem_space @DRAM dst_mem_space @L1, interconnect : [@vertical_links], broadcast : [8, 1] : memref<1x128x128xf32, strided<[524288, 128, 1], offset: ?>>, memref<1x128x128xf32>
            linalg.fill ins(%cst_1 : f32) outs(%24 : memref<1x64x128xf32>)
            linalg.batch_matmul ins(%15, %14 : memref<1x64x128xf32>, memref<1x128x128xf32>) outs(%24 : memref<1x64x128xf32>)
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%24, %23, %21 : memref<1x64x128xf32>, memref<1x64x128xf32>, memref<1x64xf32>) outs(%23 : memref<1x64x128xf32>) {
            ^bb0(%in: f32, %in_7: f32, %in_8: f32, %out: f32):
              %32 = arith.mulf %in_7, %in_8 : f32
              %33 = arith.addf %in, %32 : f32
              linalg.yield %33 : f32
            }
            linalg.copy ins(%19 : memref<1x64xf32>) outs(%18 : memref<1x64xf32>)
          }
          linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%23, %17 : memref<1x64x128xf32>, memref<1x64xf32>) outs(%22 : memref<1x64x128xf32>) {
          ^bb0(%in: f32, %in_5: f32, %out: f32):
            %28 = arith.divf %in, %in_5 : f32
            linalg.yield %28 : f32
          }
          %reinterpret_cast_4 = memref.reinterpret_cast %arg3 to offset: [%27], sizes: [1, 64, 128], strides: [524288, 128, 1] : memref<32x4096x128xf32> to memref<1x64x128xf32, strided<[524288, 128, 1], offset: ?>>
          loom.copy %22, %reinterpret_cast_4 src_mem_space @L1 dst_mem_space @DRAM, interconnect : [], broadcast : [1, 1] : memref<1x64x128xf32>, memref<1x64x128xf32, strided<[524288, 128, 1], offset: ?>>
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
          %15 = loom.alloc [1, 64, 128] on @L1 : memref<1x64x128xf32>
          %16 = loom.alloc [1, 128, 128] on @L1 : memref<1x128x128xf32>
          %17 = loom.alloc [1, 64] on @L1 : memref<1x64xf32>
          %18 = loom.alloc [1, 64] on @L1 : memref<1x64xf32>
          %19 = loom.alloc [1, 64] on @L1 : memref<1x64xf32>
          %20 = loom.alloc [1, 64] on @L1 : memref<1x64xf32>
          %21 = loom.alloc [1, 64] on @L1 : memref<1x64xf32>
          %22 = loom.alloc [1, 64, 128] on @L1 : memref<1x64x128xf32>
          %23 = loom.alloc [1, 64, 128] on @L1 : memref<1x64x128xf32>
          %24 = loom.alloc [1, 64, 128] on @L1 : memref<1x64x128xf32>
          %25 = arith.muli %arg6, %c524288 overflow<nsw> : index
          %26 = arith.muli %13, %c8192 : index
          %27 = arith.addi %25, %26 : index
          %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%27], sizes: [1, 64, 128], strides: [524288, 128, 1] : memref<32x4096x128xf32> to memref<1x64x128xf32, strided<[524288, 128, 1], offset: ?>>
          loom.copy %reinterpret_cast, %22 src_mem_space @DRAM dst_mem_space @L1, interconnect : [], broadcast : [1, 1] : memref<1x64x128xf32, strided<[524288, 128, 1], offset: ?>>, memref<1x64x128xf32>
          linalg.fill ins(%cst_1 : f32) outs(%23 : memref<1x64x128xf32>)
          linalg.fill ins(%cst_2 : f32) outs(%17 : memref<1x64xf32>)
          linalg.fill ins(%cst_3 : f32) outs(%18 : memref<1x64xf32>)
          scf.for %arg7 = %c0 to %c32 step %c1 {
            %28 = arith.muli %arg7, %c128 : index
            %29 = arith.addi %25, %28 : index
            %reinterpret_cast_5 = memref.reinterpret_cast %arg0 to offset: [%29], sizes: [1, 128, 128], strides: [524288, 4096, 1] : memref<32x128x4096xf32> to memref<1x128x128xf32, strided<[524288, 4096, 1], offset: ?>>
            loom.copy %reinterpret_cast_5, %16 src_mem_space @DRAM dst_mem_space @L1, interconnect : [@horizontal_links, @vertical_links], broadcast : [8, 8] : memref<1x128x128xf32, strided<[524288, 4096, 1], offset: ?>>, memref<1x128x128xf32>
            linalg.fill ins(%cst_1 : f32) outs(%15 : memref<1x64x128xf32>)
            linalg.batch_matmul ins(%22, %16 : memref<1x64x128xf32>, memref<1x128x128xf32>) outs(%15 : memref<1x64x128xf32>)
            linalg.fill ins(%cst_3 : f32) outs(%19 : memref<1x64xf32>)
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%15 : memref<1x64x128xf32>) outs(%19 : memref<1x64xf32>) {
            ^bb0(%in: f32, %out: f32):
              %32 = arith.maximumf %in, %out : f32
              linalg.yield %32 : f32
            }
            linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%18, %19 : memref<1x64xf32>, memref<1x64xf32>) outs(%19 : memref<1x64xf32>) {
            ^bb0(%in: f32, %in_7: f32, %out: f32):
              %32 = arith.truncf %cst_0 : f64 to f32
              %33 = arith.mulf %in_7, %32 : f32
              %34 = arith.cmpf ogt, %in, %33 : f32
              %35 = arith.select %34, %in, %33 : f32
              linalg.yield %35 : f32
            }
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%15, %19 : memref<1x64x128xf32>, memref<1x64xf32>) outs(%15 : memref<1x64x128xf32>) {
            ^bb0(%in: f32, %in_7: f32, %out: f32):
              %32 = arith.truncf %cst_0 : f64 to f32
              %33 = arith.mulf %in, %32 : f32
              %34 = arith.subf %33, %in_7 : f32
              %35 = math.powf %cst, %34 : f32
              linalg.yield %35 : f32
            }
            linalg.fill ins(%cst_1 : f32) outs(%20 : memref<1x64xf32>)
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%15 : memref<1x64x128xf32>) outs(%20 : memref<1x64xf32>) {
            ^bb0(%in: f32, %out: f32):
              %32 = arith.addf %in, %out : f32
              linalg.yield %32 : f32
            }
            linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%18, %19 : memref<1x64xf32>, memref<1x64xf32>) outs(%21 : memref<1x64xf32>) {
            ^bb0(%in: f32, %in_7: f32, %out: f32):
              %32 = arith.subf %in, %in_7 : f32
              %33 = math.powf %cst, %32 : f32
              linalg.yield %33 : f32
            }
            linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%17, %21, %20 : memref<1x64xf32>, memref<1x64xf32>, memref<1x64xf32>) outs(%17 : memref<1x64xf32>) {
            ^bb0(%in: f32, %in_7: f32, %in_8: f32, %out: f32):
              %32 = arith.mulf %in, %in_7 : f32
              %33 = arith.addf %32, %in_8 : f32
              linalg.yield %33 : f32
            }
            %30 = arith.muli %arg7, %c16384 : index
            %31 = arith.addi %25, %30 : index
            %reinterpret_cast_6 = memref.reinterpret_cast %arg1 to offset: [%31], sizes: [1, 128, 128], strides: [524288, 128, 1] : memref<32x4096x128xf32> to memref<1x128x128xf32, strided<[524288, 128, 1], offset: ?>>
            loom.copy %reinterpret_cast_6, %14 src_mem_space @DRAM dst_mem_space @L1, interconnect : [], broadcast : [1, 1] : memref<1x128x128xf32, strided<[524288, 128, 1], offset: ?>>, memref<1x128x128xf32>
            linalg.fill ins(%cst_1 : f32) outs(%24 : memref<1x64x128xf32>)
            linalg.batch_matmul ins(%15, %14 : memref<1x64x128xf32>, memref<1x128x128xf32>) outs(%24 : memref<1x64x128xf32>)
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%24, %23, %21 : memref<1x64x128xf32>, memref<1x64x128xf32>, memref<1x64xf32>) outs(%23 : memref<1x64x128xf32>) {
            ^bb0(%in: f32, %in_7: f32, %in_8: f32, %out: f32):
              %32 = arith.mulf %in_7, %in_8 : f32
              %33 = arith.addf %in, %32 : f32
              linalg.yield %33 : f32
            }
            linalg.copy ins(%19 : memref<1x64xf32>) outs(%18 : memref<1x64xf32>)
          }
          linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%23, %17 : memref<1x64x128xf32>, memref<1x64xf32>) outs(%22 : memref<1x64x128xf32>) {
          ^bb0(%in: f32, %in_5: f32, %out: f32):
            %28 = arith.divf %in, %in_5 : f32
            linalg.yield %28 : f32
          }
          %reinterpret_cast_4 = memref.reinterpret_cast %arg3 to offset: [%27], sizes: [1, 64, 128], strides: [524288, 128, 1] : memref<32x4096x128xf32> to memref<1x64x128xf32, strided<[524288, 128, 1], offset: ?>>
          loom.copy %22, %reinterpret_cast_4 src_mem_space @L1 dst_mem_space @DRAM, interconnect : [], broadcast : [1, 1] : memref<1x64x128xf32>, memref<1x64x128xf32, strided<[524288, 128, 1], offset: ?>>
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
          %15 = loom.alloc [1, 64, 128] on @L1 : memref<1x64x128xf32>
          %16 = loom.alloc [1, 128, 128] on @L1 : memref<1x128x128xf32>
          %17 = loom.alloc [1, 64] on @L1 : memref<1x64xf32>
          %18 = loom.alloc [1, 64] on @L1 : memref<1x64xf32>
          %19 = loom.alloc [1, 64] on @L1 : memref<1x64xf32>
          %20 = loom.alloc [1, 64] on @L1 : memref<1x64xf32>
          %21 = loom.alloc [1, 64] on @L1 : memref<1x64xf32>
          %22 = loom.alloc [1, 64, 128] on @L1 : memref<1x64x128xf32>
          %23 = loom.alloc [1, 64, 128] on @L1 : memref<1x64x128xf32>
          %24 = loom.alloc [1, 64, 128] on @L1 : memref<1x64x128xf32>
          %25 = arith.muli %arg6, %c524288 overflow<nsw> : index
          %26 = arith.muli %13, %c8192 : index
          %27 = arith.addi %25, %26 : index
          %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%27], sizes: [1, 64, 128], strides: [524288, 128, 1] : memref<32x4096x128xf32> to memref<1x64x128xf32, strided<[524288, 128, 1], offset: ?>>
          loom.copy %reinterpret_cast, %22 src_mem_space @DRAM dst_mem_space @L1, interconnect : [], broadcast : [1, 1] : memref<1x64x128xf32, strided<[524288, 128, 1], offset: ?>>, memref<1x64x128xf32>
          linalg.fill ins(%cst_1 : f32) outs(%23 : memref<1x64x128xf32>)
          linalg.fill ins(%cst_2 : f32) outs(%17 : memref<1x64xf32>)
          linalg.fill ins(%cst_3 : f32) outs(%18 : memref<1x64xf32>)
          scf.for %arg7 = %c0 to %c32 step %c1 {
            %28 = arith.muli %arg7, %c128 : index
            %29 = arith.addi %25, %28 : index
            %reinterpret_cast_5 = memref.reinterpret_cast %arg0 to offset: [%29], sizes: [1, 128, 128], strides: [524288, 4096, 1] : memref<32x128x4096xf32> to memref<1x128x128xf32, strided<[524288, 4096, 1], offset: ?>>
            loom.copy %reinterpret_cast_5, %16 src_mem_space @DRAM dst_mem_space @L1, interconnect : [@horizontal_links, @vertical_links], broadcast : [8, 8] : memref<1x128x128xf32, strided<[524288, 4096, 1], offset: ?>>, memref<1x128x128xf32>
            linalg.fill ins(%cst_1 : f32) outs(%15 : memref<1x64x128xf32>)
            linalg.batch_matmul ins(%22, %16 : memref<1x64x128xf32>, memref<1x128x128xf32>) outs(%15 : memref<1x64x128xf32>)
            linalg.fill ins(%cst_3 : f32) outs(%19 : memref<1x64xf32>)
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%15 : memref<1x64x128xf32>) outs(%19 : memref<1x64xf32>) {
            ^bb0(%in: f32, %out: f32):
              %32 = arith.maximumf %in, %out : f32
              linalg.yield %32 : f32
            }
            linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%18, %19 : memref<1x64xf32>, memref<1x64xf32>) outs(%19 : memref<1x64xf32>) {
            ^bb0(%in: f32, %in_7: f32, %out: f32):
              %32 = arith.truncf %cst_0 : f64 to f32
              %33 = arith.mulf %in_7, %32 : f32
              %34 = arith.cmpf ogt, %in, %33 : f32
              %35 = arith.select %34, %in, %33 : f32
              linalg.yield %35 : f32
            }
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%15, %19 : memref<1x64x128xf32>, memref<1x64xf32>) outs(%15 : memref<1x64x128xf32>) {
            ^bb0(%in: f32, %in_7: f32, %out: f32):
              %32 = arith.truncf %cst_0 : f64 to f32
              %33 = arith.mulf %in, %32 : f32
              %34 = arith.subf %33, %in_7 : f32
              %35 = math.powf %cst, %34 : f32
              linalg.yield %35 : f32
            }
            linalg.fill ins(%cst_1 : f32) outs(%20 : memref<1x64xf32>)
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%15 : memref<1x64x128xf32>) outs(%20 : memref<1x64xf32>) {
            ^bb0(%in: f32, %out: f32):
              %32 = arith.addf %in, %out : f32
              linalg.yield %32 : f32
            }
            linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%18, %19 : memref<1x64xf32>, memref<1x64xf32>) outs(%21 : memref<1x64xf32>) {
            ^bb0(%in: f32, %in_7: f32, %out: f32):
              %32 = arith.subf %in, %in_7 : f32
              %33 = math.powf %cst, %32 : f32
              linalg.yield %33 : f32
            }
            linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%17, %21, %20 : memref<1x64xf32>, memref<1x64xf32>, memref<1x64xf32>) outs(%17 : memref<1x64xf32>) {
            ^bb0(%in: f32, %in_7: f32, %in_8: f32, %out: f32):
              %32 = arith.mulf %in, %in_7 : f32
              %33 = arith.addf %32, %in_8 : f32
              linalg.yield %33 : f32
            }
            %30 = arith.muli %arg7, %c16384 : index
            %31 = arith.addi %25, %30 : index
            %reinterpret_cast_6 = memref.reinterpret_cast %arg1 to offset: [%31], sizes: [1, 128, 128], strides: [524288, 128, 1] : memref<32x4096x128xf32> to memref<1x128x128xf32, strided<[524288, 128, 1], offset: ?>>
            loom.copy %reinterpret_cast_6, %14 src_mem_space @DRAM dst_mem_space @L1, interconnect : [@horizontal_links, @vertical_links], broadcast : [8, 8] : memref<1x128x128xf32, strided<[524288, 128, 1], offset: ?>>, memref<1x128x128xf32>
            linalg.fill ins(%cst_1 : f32) outs(%24 : memref<1x64x128xf32>)
            linalg.batch_matmul ins(%15, %14 : memref<1x64x128xf32>, memref<1x128x128xf32>) outs(%24 : memref<1x64x128xf32>)
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%24, %23, %21 : memref<1x64x128xf32>, memref<1x64x128xf32>, memref<1x64xf32>) outs(%23 : memref<1x64x128xf32>) {
            ^bb0(%in: f32, %in_7: f32, %in_8: f32, %out: f32):
              %32 = arith.mulf %in_7, %in_8 : f32
              %33 = arith.addf %in, %32 : f32
              linalg.yield %33 : f32
            }
            linalg.copy ins(%19 : memref<1x64xf32>) outs(%18 : memref<1x64xf32>)
          }
          linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%23, %17 : memref<1x64x128xf32>, memref<1x64xf32>) outs(%22 : memref<1x64x128xf32>) {
          ^bb0(%in: f32, %in_5: f32, %out: f32):
            %28 = arith.divf %in, %in_5 : f32
            linalg.yield %28 : f32
          }
          %reinterpret_cast_4 = memref.reinterpret_cast %arg3 to offset: [%27], sizes: [1, 64, 128], strides: [524288, 128, 1] : memref<32x4096x128xf32> to memref<1x64x128xf32, strided<[524288, 128, 1], offset: ?>>
          loom.copy %22, %reinterpret_cast_4 src_mem_space @L1 dst_mem_space @DRAM, interconnect : [], broadcast : [1, 1] : memref<1x64x128xf32>, memref<1x64x128xf32, strided<[524288, 128, 1], offset: ?>>
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
          %15 = loom.alloc [1, 64, 128] on @L1 : memref<1x64x128xf32>
          %16 = loom.alloc [1, 128, 128] on @L1 : memref<1x128x128xf32>
          %17 = loom.alloc [1, 64] on @L1 : memref<1x64xf32>
          %18 = loom.alloc [1, 64] on @L1 : memref<1x64xf32>
          %19 = loom.alloc [1, 64] on @L1 : memref<1x64xf32>
          %20 = loom.alloc [1, 64] on @L1 : memref<1x64xf32>
          %21 = loom.alloc [1, 64] on @L1 : memref<1x64xf32>
          %22 = loom.alloc [1, 64, 128] on @L1 : memref<1x64x128xf32>
          %23 = loom.alloc [1, 64, 128] on @L1 : memref<1x64x128xf32>
          %24 = loom.alloc [1, 64, 128] on @L1 : memref<1x64x128xf32>
          %25 = arith.muli %arg6, %c524288 overflow<nsw> : index
          %26 = arith.muli %13, %c8192 : index
          %27 = arith.addi %25, %26 : index
          %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%27], sizes: [1, 64, 128], strides: [524288, 128, 1] : memref<32x4096x128xf32> to memref<1x64x128xf32, strided<[524288, 128, 1], offset: ?>>
          loom.copy %reinterpret_cast, %22 src_mem_space @DRAM dst_mem_space @L1, interconnect : [], broadcast : [1, 1] : memref<1x64x128xf32, strided<[524288, 128, 1], offset: ?>>, memref<1x64x128xf32>
          linalg.fill ins(%cst_1 : f32) outs(%23 : memref<1x64x128xf32>)
          linalg.fill ins(%cst_2 : f32) outs(%17 : memref<1x64xf32>)
          linalg.fill ins(%cst_3 : f32) outs(%18 : memref<1x64xf32>)
          scf.for %arg7 = %c0 to %c32 step %c1 {
            %28 = arith.muli %arg7, %c128 : index
            %29 = arith.addi %25, %28 : index
            %reinterpret_cast_5 = memref.reinterpret_cast %arg0 to offset: [%29], sizes: [1, 128, 128], strides: [524288, 4096, 1] : memref<32x128x4096xf32> to memref<1x128x128xf32, strided<[524288, 4096, 1], offset: ?>>
            loom.copy %reinterpret_cast_5, %16 src_mem_space @DRAM dst_mem_space @L1, interconnect : [@horizontal_links, @vertical_links], broadcast : [8, 8] : memref<1x128x128xf32, strided<[524288, 4096, 1], offset: ?>>, memref<1x128x128xf32>
            linalg.fill ins(%cst_1 : f32) outs(%15 : memref<1x64x128xf32>)
            linalg.batch_matmul ins(%22, %16 : memref<1x64x128xf32>, memref<1x128x128xf32>) outs(%15 : memref<1x64x128xf32>)
            linalg.fill ins(%cst_3 : f32) outs(%19 : memref<1x64xf32>)
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%15 : memref<1x64x128xf32>) outs(%19 : memref<1x64xf32>) {
            ^bb0(%in: f32, %out: f32):
              %32 = arith.maximumf %in, %out : f32
              linalg.yield %32 : f32
            }
            linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%18, %19 : memref<1x64xf32>, memref<1x64xf32>) outs(%19 : memref<1x64xf32>) {
            ^bb0(%in: f32, %in_7: f32, %out: f32):
              %32 = arith.truncf %cst_0 : f64 to f32
              %33 = arith.mulf %in_7, %32 : f32
              %34 = arith.cmpf ogt, %in, %33 : f32
              %35 = arith.select %34, %in, %33 : f32
              linalg.yield %35 : f32
            }
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%15, %19 : memref<1x64x128xf32>, memref<1x64xf32>) outs(%15 : memref<1x64x128xf32>) {
            ^bb0(%in: f32, %in_7: f32, %out: f32):
              %32 = arith.truncf %cst_0 : f64 to f32
              %33 = arith.mulf %in, %32 : f32
              %34 = arith.subf %33, %in_7 : f32
              %35 = math.powf %cst, %34 : f32
              linalg.yield %35 : f32
            }
            linalg.fill ins(%cst_1 : f32) outs(%20 : memref<1x64xf32>)
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%15 : memref<1x64x128xf32>) outs(%20 : memref<1x64xf32>) {
            ^bb0(%in: f32, %out: f32):
              %32 = arith.addf %in, %out : f32
              linalg.yield %32 : f32
            }
            linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%18, %19 : memref<1x64xf32>, memref<1x64xf32>) outs(%21 : memref<1x64xf32>) {
            ^bb0(%in: f32, %in_7: f32, %out: f32):
              %32 = arith.subf %in, %in_7 : f32
              %33 = math.powf %cst, %32 : f32
              linalg.yield %33 : f32
            }
            linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%17, %21, %20 : memref<1x64xf32>, memref<1x64xf32>, memref<1x64xf32>) outs(%17 : memref<1x64xf32>) {
            ^bb0(%in: f32, %in_7: f32, %in_8: f32, %out: f32):
              %32 = arith.mulf %in, %in_7 : f32
              %33 = arith.addf %32, %in_8 : f32
              linalg.yield %33 : f32
            }
            %30 = arith.muli %arg7, %c16384 : index
            %31 = arith.addi %25, %30 : index
            %reinterpret_cast_6 = memref.reinterpret_cast %arg1 to offset: [%31], sizes: [1, 128, 128], strides: [524288, 128, 1] : memref<32x4096x128xf32> to memref<1x128x128xf32, strided<[524288, 128, 1], offset: ?>>
            loom.copy %reinterpret_cast_6, %14 src_mem_space @DRAM dst_mem_space @L1, interconnect : [@horizontal_links], broadcast : [1, 8] : memref<1x128x128xf32, strided<[524288, 128, 1], offset: ?>>, memref<1x128x128xf32>
            linalg.fill ins(%cst_1 : f32) outs(%24 : memref<1x64x128xf32>)
            linalg.batch_matmul ins(%15, %14 : memref<1x64x128xf32>, memref<1x128x128xf32>) outs(%24 : memref<1x64x128xf32>)
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%24, %23, %21 : memref<1x64x128xf32>, memref<1x64x128xf32>, memref<1x64xf32>) outs(%23 : memref<1x64x128xf32>) {
            ^bb0(%in: f32, %in_7: f32, %in_8: f32, %out: f32):
              %32 = arith.mulf %in_7, %in_8 : f32
              %33 = arith.addf %in, %32 : f32
              linalg.yield %33 : f32
            }
            linalg.copy ins(%19 : memref<1x64xf32>) outs(%18 : memref<1x64xf32>)
          }
          linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%23, %17 : memref<1x64x128xf32>, memref<1x64xf32>) outs(%22 : memref<1x64x128xf32>) {
          ^bb0(%in: f32, %in_5: f32, %out: f32):
            %28 = arith.divf %in, %in_5 : f32
            linalg.yield %28 : f32
          }
          %reinterpret_cast_4 = memref.reinterpret_cast %arg3 to offset: [%27], sizes: [1, 64, 128], strides: [524288, 128, 1] : memref<32x4096x128xf32> to memref<1x64x128xf32, strided<[524288, 128, 1], offset: ?>>
          loom.copy %22, %reinterpret_cast_4 src_mem_space @L1 dst_mem_space @DRAM, interconnect : [], broadcast : [1, 1] : memref<1x64x128xf32>, memref<1x64x128xf32, strided<[524288, 128, 1], offset: ?>>
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
          %15 = loom.alloc [1, 64, 128] on @L1 : memref<1x64x128xf32>
          %16 = loom.alloc [1, 128, 128] on @L1 : memref<1x128x128xf32>
          %17 = loom.alloc [1, 64] on @L1 : memref<1x64xf32>
          %18 = loom.alloc [1, 64] on @L1 : memref<1x64xf32>
          %19 = loom.alloc [1, 64] on @L1 : memref<1x64xf32>
          %20 = loom.alloc [1, 64] on @L1 : memref<1x64xf32>
          %21 = loom.alloc [1, 64] on @L1 : memref<1x64xf32>
          %22 = loom.alloc [1, 64, 128] on @L1 : memref<1x64x128xf32>
          %23 = loom.alloc [1, 64, 128] on @L1 : memref<1x64x128xf32>
          %24 = loom.alloc [1, 64, 128] on @L1 : memref<1x64x128xf32>
          %25 = arith.muli %arg6, %c524288 overflow<nsw> : index
          %26 = arith.muli %13, %c8192 : index
          %27 = arith.addi %25, %26 : index
          %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%27], sizes: [1, 64, 128], strides: [524288, 128, 1] : memref<32x4096x128xf32> to memref<1x64x128xf32, strided<[524288, 128, 1], offset: ?>>
          loom.copy %reinterpret_cast, %22 src_mem_space @DRAM dst_mem_space @L1, interconnect : [], broadcast : [1, 1] : memref<1x64x128xf32, strided<[524288, 128, 1], offset: ?>>, memref<1x64x128xf32>
          linalg.fill ins(%cst_1 : f32) outs(%23 : memref<1x64x128xf32>)
          linalg.fill ins(%cst_2 : f32) outs(%17 : memref<1x64xf32>)
          linalg.fill ins(%cst_3 : f32) outs(%18 : memref<1x64xf32>)
          scf.for %arg7 = %c0 to %c32 step %c1 {
            %28 = arith.muli %arg7, %c128 : index
            %29 = arith.addi %25, %28 : index
            %reinterpret_cast_5 = memref.reinterpret_cast %arg0 to offset: [%29], sizes: [1, 128, 128], strides: [524288, 4096, 1] : memref<32x128x4096xf32> to memref<1x128x128xf32, strided<[524288, 4096, 1], offset: ?>>
            loom.copy %reinterpret_cast_5, %16 src_mem_space @DRAM dst_mem_space @L1, interconnect : [@horizontal_links, @vertical_links], broadcast : [8, 8] : memref<1x128x128xf32, strided<[524288, 4096, 1], offset: ?>>, memref<1x128x128xf32>
            linalg.fill ins(%cst_1 : f32) outs(%15 : memref<1x64x128xf32>)
            linalg.batch_matmul ins(%22, %16 : memref<1x64x128xf32>, memref<1x128x128xf32>) outs(%15 : memref<1x64x128xf32>)
            linalg.fill ins(%cst_3 : f32) outs(%19 : memref<1x64xf32>)
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%15 : memref<1x64x128xf32>) outs(%19 : memref<1x64xf32>) {
            ^bb0(%in: f32, %out: f32):
              %32 = arith.maximumf %in, %out : f32
              linalg.yield %32 : f32
            }
            linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%18, %19 : memref<1x64xf32>, memref<1x64xf32>) outs(%19 : memref<1x64xf32>) {
            ^bb0(%in: f32, %in_7: f32, %out: f32):
              %32 = arith.truncf %cst_0 : f64 to f32
              %33 = arith.mulf %in_7, %32 : f32
              %34 = arith.cmpf ogt, %in, %33 : f32
              %35 = arith.select %34, %in, %33 : f32
              linalg.yield %35 : f32
            }
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%15, %19 : memref<1x64x128xf32>, memref<1x64xf32>) outs(%15 : memref<1x64x128xf32>) {
            ^bb0(%in: f32, %in_7: f32, %out: f32):
              %32 = arith.truncf %cst_0 : f64 to f32
              %33 = arith.mulf %in, %32 : f32
              %34 = arith.subf %33, %in_7 : f32
              %35 = math.powf %cst, %34 : f32
              linalg.yield %35 : f32
            }
            linalg.fill ins(%cst_1 : f32) outs(%20 : memref<1x64xf32>)
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%15 : memref<1x64x128xf32>) outs(%20 : memref<1x64xf32>) {
            ^bb0(%in: f32, %out: f32):
              %32 = arith.addf %in, %out : f32
              linalg.yield %32 : f32
            }
            linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%18, %19 : memref<1x64xf32>, memref<1x64xf32>) outs(%21 : memref<1x64xf32>) {
            ^bb0(%in: f32, %in_7: f32, %out: f32):
              %32 = arith.subf %in, %in_7 : f32
              %33 = math.powf %cst, %32 : f32
              linalg.yield %33 : f32
            }
            linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%17, %21, %20 : memref<1x64xf32>, memref<1x64xf32>, memref<1x64xf32>) outs(%17 : memref<1x64xf32>) {
            ^bb0(%in: f32, %in_7: f32, %in_8: f32, %out: f32):
              %32 = arith.mulf %in, %in_7 : f32
              %33 = arith.addf %32, %in_8 : f32
              linalg.yield %33 : f32
            }
            %30 = arith.muli %arg7, %c16384 : index
            %31 = arith.addi %25, %30 : index
            %reinterpret_cast_6 = memref.reinterpret_cast %arg1 to offset: [%31], sizes: [1, 128, 128], strides: [524288, 128, 1] : memref<32x4096x128xf32> to memref<1x128x128xf32, strided<[524288, 128, 1], offset: ?>>
            loom.copy %reinterpret_cast_6, %14 src_mem_space @DRAM dst_mem_space @L1, interconnect : [@vertical_links], broadcast : [8, 1] : memref<1x128x128xf32, strided<[524288, 128, 1], offset: ?>>, memref<1x128x128xf32>
            linalg.fill ins(%cst_1 : f32) outs(%24 : memref<1x64x128xf32>)
            linalg.batch_matmul ins(%15, %14 : memref<1x64x128xf32>, memref<1x128x128xf32>) outs(%24 : memref<1x64x128xf32>)
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%24, %23, %21 : memref<1x64x128xf32>, memref<1x64x128xf32>, memref<1x64xf32>) outs(%23 : memref<1x64x128xf32>) {
            ^bb0(%in: f32, %in_7: f32, %in_8: f32, %out: f32):
              %32 = arith.mulf %in_7, %in_8 : f32
              %33 = arith.addf %in, %32 : f32
              linalg.yield %33 : f32
            }
            linalg.copy ins(%19 : memref<1x64xf32>) outs(%18 : memref<1x64xf32>)
          }
          linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%23, %17 : memref<1x64x128xf32>, memref<1x64xf32>) outs(%22 : memref<1x64x128xf32>) {
          ^bb0(%in: f32, %in_5: f32, %out: f32):
            %28 = arith.divf %in, %in_5 : f32
            linalg.yield %28 : f32
          }
          %reinterpret_cast_4 = memref.reinterpret_cast %arg3 to offset: [%27], sizes: [1, 64, 128], strides: [524288, 128, 1] : memref<32x4096x128xf32> to memref<1x64x128xf32, strided<[524288, 128, 1], offset: ?>>
          loom.copy %22, %reinterpret_cast_4 src_mem_space @L1 dst_mem_space @DRAM, interconnect : [], broadcast : [1, 1] : memref<1x64x128xf32>, memref<1x64x128xf32, strided<[524288, 128, 1], offset: ?>>
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
          %15 = loom.alloc [1, 64, 128] on @L1 : memref<1x64x128xf32>
          %16 = loom.alloc [1, 128, 128] on @L1 : memref<1x128x128xf32>
          %17 = loom.alloc [1, 64] on @L1 : memref<1x64xf32>
          %18 = loom.alloc [1, 64] on @L1 : memref<1x64xf32>
          %19 = loom.alloc [1, 64] on @L1 : memref<1x64xf32>
          %20 = loom.alloc [1, 64] on @L1 : memref<1x64xf32>
          %21 = loom.alloc [1, 64] on @L1 : memref<1x64xf32>
          %22 = loom.alloc [1, 64, 128] on @L1 : memref<1x64x128xf32>
          %23 = loom.alloc [1, 64, 128] on @L1 : memref<1x64x128xf32>
          %24 = loom.alloc [1, 64, 128] on @L1 : memref<1x64x128xf32>
          %25 = arith.muli %arg6, %c524288 overflow<nsw> : index
          %26 = arith.muli %13, %c8192 : index
          %27 = arith.addi %25, %26 : index
          %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%27], sizes: [1, 64, 128], strides: [524288, 128, 1] : memref<32x4096x128xf32> to memref<1x64x128xf32, strided<[524288, 128, 1], offset: ?>>
          loom.copy %reinterpret_cast, %22 src_mem_space @DRAM dst_mem_space @L1, interconnect : [], broadcast : [1, 1] : memref<1x64x128xf32, strided<[524288, 128, 1], offset: ?>>, memref<1x64x128xf32>
          linalg.fill ins(%cst_1 : f32) outs(%23 : memref<1x64x128xf32>)
          linalg.fill ins(%cst_2 : f32) outs(%17 : memref<1x64xf32>)
          linalg.fill ins(%cst_3 : f32) outs(%18 : memref<1x64xf32>)
          scf.for %arg7 = %c0 to %c32 step %c1 {
            %28 = arith.muli %arg7, %c128 : index
            %29 = arith.addi %25, %28 : index
            %reinterpret_cast_5 = memref.reinterpret_cast %arg0 to offset: [%29], sizes: [1, 128, 128], strides: [524288, 4096, 1] : memref<32x128x4096xf32> to memref<1x128x128xf32, strided<[524288, 4096, 1], offset: ?>>
            loom.copy %reinterpret_cast_5, %16 src_mem_space @DRAM dst_mem_space @L1, interconnect : [@horizontal_links], broadcast : [1, 8] : memref<1x128x128xf32, strided<[524288, 4096, 1], offset: ?>>, memref<1x128x128xf32>
            linalg.fill ins(%cst_1 : f32) outs(%15 : memref<1x64x128xf32>)
            linalg.batch_matmul ins(%22, %16 : memref<1x64x128xf32>, memref<1x128x128xf32>) outs(%15 : memref<1x64x128xf32>)
            linalg.fill ins(%cst_3 : f32) outs(%19 : memref<1x64xf32>)
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%15 : memref<1x64x128xf32>) outs(%19 : memref<1x64xf32>) {
            ^bb0(%in: f32, %out: f32):
              %32 = arith.maximumf %in, %out : f32
              linalg.yield %32 : f32
            }
            linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%18, %19 : memref<1x64xf32>, memref<1x64xf32>) outs(%19 : memref<1x64xf32>) {
            ^bb0(%in: f32, %in_7: f32, %out: f32):
              %32 = arith.truncf %cst_0 : f64 to f32
              %33 = arith.mulf %in_7, %32 : f32
              %34 = arith.cmpf ogt, %in, %33 : f32
              %35 = arith.select %34, %in, %33 : f32
              linalg.yield %35 : f32
            }
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%15, %19 : memref<1x64x128xf32>, memref<1x64xf32>) outs(%15 : memref<1x64x128xf32>) {
            ^bb0(%in: f32, %in_7: f32, %out: f32):
              %32 = arith.truncf %cst_0 : f64 to f32
              %33 = arith.mulf %in, %32 : f32
              %34 = arith.subf %33, %in_7 : f32
              %35 = math.powf %cst, %34 : f32
              linalg.yield %35 : f32
            }
            linalg.fill ins(%cst_1 : f32) outs(%20 : memref<1x64xf32>)
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%15 : memref<1x64x128xf32>) outs(%20 : memref<1x64xf32>) {
            ^bb0(%in: f32, %out: f32):
              %32 = arith.addf %in, %out : f32
              linalg.yield %32 : f32
            }
            linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%18, %19 : memref<1x64xf32>, memref<1x64xf32>) outs(%21 : memref<1x64xf32>) {
            ^bb0(%in: f32, %in_7: f32, %out: f32):
              %32 = arith.subf %in, %in_7 : f32
              %33 = math.powf %cst, %32 : f32
              linalg.yield %33 : f32
            }
            linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%17, %21, %20 : memref<1x64xf32>, memref<1x64xf32>, memref<1x64xf32>) outs(%17 : memref<1x64xf32>) {
            ^bb0(%in: f32, %in_7: f32, %in_8: f32, %out: f32):
              %32 = arith.mulf %in, %in_7 : f32
              %33 = arith.addf %32, %in_8 : f32
              linalg.yield %33 : f32
            }
            %30 = arith.muli %arg7, %c16384 : index
            %31 = arith.addi %25, %30 : index
            %reinterpret_cast_6 = memref.reinterpret_cast %arg1 to offset: [%31], sizes: [1, 128, 128], strides: [524288, 128, 1] : memref<32x4096x128xf32> to memref<1x128x128xf32, strided<[524288, 128, 1], offset: ?>>
            loom.copy %reinterpret_cast_6, %14 src_mem_space @DRAM dst_mem_space @L1, interconnect : [], broadcast : [1, 1] : memref<1x128x128xf32, strided<[524288, 128, 1], offset: ?>>, memref<1x128x128xf32>
            linalg.fill ins(%cst_1 : f32) outs(%24 : memref<1x64x128xf32>)
            linalg.batch_matmul ins(%15, %14 : memref<1x64x128xf32>, memref<1x128x128xf32>) outs(%24 : memref<1x64x128xf32>)
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%24, %23, %21 : memref<1x64x128xf32>, memref<1x64x128xf32>, memref<1x64xf32>) outs(%23 : memref<1x64x128xf32>) {
            ^bb0(%in: f32, %in_7: f32, %in_8: f32, %out: f32):
              %32 = arith.mulf %in_7, %in_8 : f32
              %33 = arith.addf %in, %32 : f32
              linalg.yield %33 : f32
            }
            linalg.copy ins(%19 : memref<1x64xf32>) outs(%18 : memref<1x64xf32>)
          }
          linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%23, %17 : memref<1x64x128xf32>, memref<1x64xf32>) outs(%22 : memref<1x64x128xf32>) {
          ^bb0(%in: f32, %in_5: f32, %out: f32):
            %28 = arith.divf %in, %in_5 : f32
            linalg.yield %28 : f32
          }
          %reinterpret_cast_4 = memref.reinterpret_cast %arg3 to offset: [%27], sizes: [1, 64, 128], strides: [524288, 128, 1] : memref<32x4096x128xf32> to memref<1x64x128xf32, strided<[524288, 128, 1], offset: ?>>
          loom.copy %22, %reinterpret_cast_4 src_mem_space @L1 dst_mem_space @DRAM, interconnect : [], broadcast : [1, 1] : memref<1x64x128xf32>, memref<1x64x128xf32, strided<[524288, 128, 1], offset: ?>>
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
          %15 = loom.alloc [1, 64, 128] on @L1 : memref<1x64x128xf32>
          %16 = loom.alloc [1, 128, 128] on @L1 : memref<1x128x128xf32>
          %17 = loom.alloc [1, 64] on @L1 : memref<1x64xf32>
          %18 = loom.alloc [1, 64] on @L1 : memref<1x64xf32>
          %19 = loom.alloc [1, 64] on @L1 : memref<1x64xf32>
          %20 = loom.alloc [1, 64] on @L1 : memref<1x64xf32>
          %21 = loom.alloc [1, 64] on @L1 : memref<1x64xf32>
          %22 = loom.alloc [1, 64, 128] on @L1 : memref<1x64x128xf32>
          %23 = loom.alloc [1, 64, 128] on @L1 : memref<1x64x128xf32>
          %24 = loom.alloc [1, 64, 128] on @L1 : memref<1x64x128xf32>
          %25 = arith.muli %arg6, %c524288 overflow<nsw> : index
          %26 = arith.muli %13, %c8192 : index
          %27 = arith.addi %25, %26 : index
          %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%27], sizes: [1, 64, 128], strides: [524288, 128, 1] : memref<32x4096x128xf32> to memref<1x64x128xf32, strided<[524288, 128, 1], offset: ?>>
          loom.copy %reinterpret_cast, %22 src_mem_space @DRAM dst_mem_space @L1, interconnect : [], broadcast : [1, 1] : memref<1x64x128xf32, strided<[524288, 128, 1], offset: ?>>, memref<1x64x128xf32>
          linalg.fill ins(%cst_1 : f32) outs(%23 : memref<1x64x128xf32>)
          linalg.fill ins(%cst_2 : f32) outs(%17 : memref<1x64xf32>)
          linalg.fill ins(%cst_3 : f32) outs(%18 : memref<1x64xf32>)
          scf.for %arg7 = %c0 to %c32 step %c1 {
            %28 = arith.muli %arg7, %c128 : index
            %29 = arith.addi %25, %28 : index
            %reinterpret_cast_5 = memref.reinterpret_cast %arg0 to offset: [%29], sizes: [1, 128, 128], strides: [524288, 4096, 1] : memref<32x128x4096xf32> to memref<1x128x128xf32, strided<[524288, 4096, 1], offset: ?>>
            loom.copy %reinterpret_cast_5, %16 src_mem_space @DRAM dst_mem_space @L1, interconnect : [@horizontal_links], broadcast : [1, 8] : memref<1x128x128xf32, strided<[524288, 4096, 1], offset: ?>>, memref<1x128x128xf32>
            linalg.fill ins(%cst_1 : f32) outs(%15 : memref<1x64x128xf32>)
            linalg.batch_matmul ins(%22, %16 : memref<1x64x128xf32>, memref<1x128x128xf32>) outs(%15 : memref<1x64x128xf32>)
            linalg.fill ins(%cst_3 : f32) outs(%19 : memref<1x64xf32>)
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%15 : memref<1x64x128xf32>) outs(%19 : memref<1x64xf32>) {
            ^bb0(%in: f32, %out: f32):
              %32 = arith.maximumf %in, %out : f32
              linalg.yield %32 : f32
            }
            linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%18, %19 : memref<1x64xf32>, memref<1x64xf32>) outs(%19 : memref<1x64xf32>) {
            ^bb0(%in: f32, %in_7: f32, %out: f32):
              %32 = arith.truncf %cst_0 : f64 to f32
              %33 = arith.mulf %in_7, %32 : f32
              %34 = arith.cmpf ogt, %in, %33 : f32
              %35 = arith.select %34, %in, %33 : f32
              linalg.yield %35 : f32
            }
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%15, %19 : memref<1x64x128xf32>, memref<1x64xf32>) outs(%15 : memref<1x64x128xf32>) {
            ^bb0(%in: f32, %in_7: f32, %out: f32):
              %32 = arith.truncf %cst_0 : f64 to f32
              %33 = arith.mulf %in, %32 : f32
              %34 = arith.subf %33, %in_7 : f32
              %35 = math.powf %cst, %34 : f32
              linalg.yield %35 : f32
            }
            linalg.fill ins(%cst_1 : f32) outs(%20 : memref<1x64xf32>)
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%15 : memref<1x64x128xf32>) outs(%20 : memref<1x64xf32>) {
            ^bb0(%in: f32, %out: f32):
              %32 = arith.addf %in, %out : f32
              linalg.yield %32 : f32
            }
            linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%18, %19 : memref<1x64xf32>, memref<1x64xf32>) outs(%21 : memref<1x64xf32>) {
            ^bb0(%in: f32, %in_7: f32, %out: f32):
              %32 = arith.subf %in, %in_7 : f32
              %33 = math.powf %cst, %32 : f32
              linalg.yield %33 : f32
            }
            linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%17, %21, %20 : memref<1x64xf32>, memref<1x64xf32>, memref<1x64xf32>) outs(%17 : memref<1x64xf32>) {
            ^bb0(%in: f32, %in_7: f32, %in_8: f32, %out: f32):
              %32 = arith.mulf %in, %in_7 : f32
              %33 = arith.addf %32, %in_8 : f32
              linalg.yield %33 : f32
            }
            %30 = arith.muli %arg7, %c16384 : index
            %31 = arith.addi %25, %30 : index
            %reinterpret_cast_6 = memref.reinterpret_cast %arg1 to offset: [%31], sizes: [1, 128, 128], strides: [524288, 128, 1] : memref<32x4096x128xf32> to memref<1x128x128xf32, strided<[524288, 128, 1], offset: ?>>
            loom.copy %reinterpret_cast_6, %14 src_mem_space @DRAM dst_mem_space @L1, interconnect : [@horizontal_links, @vertical_links], broadcast : [8, 8] : memref<1x128x128xf32, strided<[524288, 128, 1], offset: ?>>, memref<1x128x128xf32>
            linalg.fill ins(%cst_1 : f32) outs(%24 : memref<1x64x128xf32>)
            linalg.batch_matmul ins(%15, %14 : memref<1x64x128xf32>, memref<1x128x128xf32>) outs(%24 : memref<1x64x128xf32>)
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%24, %23, %21 : memref<1x64x128xf32>, memref<1x64x128xf32>, memref<1x64xf32>) outs(%23 : memref<1x64x128xf32>) {
            ^bb0(%in: f32, %in_7: f32, %in_8: f32, %out: f32):
              %32 = arith.mulf %in_7, %in_8 : f32
              %33 = arith.addf %in, %32 : f32
              linalg.yield %33 : f32
            }
            linalg.copy ins(%19 : memref<1x64xf32>) outs(%18 : memref<1x64xf32>)
          }
          linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%23, %17 : memref<1x64x128xf32>, memref<1x64xf32>) outs(%22 : memref<1x64x128xf32>) {
          ^bb0(%in: f32, %in_5: f32, %out: f32):
            %28 = arith.divf %in, %in_5 : f32
            linalg.yield %28 : f32
          }
          %reinterpret_cast_4 = memref.reinterpret_cast %arg3 to offset: [%27], sizes: [1, 64, 128], strides: [524288, 128, 1] : memref<32x4096x128xf32> to memref<1x64x128xf32, strided<[524288, 128, 1], offset: ?>>
          loom.copy %22, %reinterpret_cast_4 src_mem_space @L1 dst_mem_space @DRAM, interconnect : [], broadcast : [1, 1] : memref<1x64x128xf32>, memref<1x64x128xf32, strided<[524288, 128, 1], offset: ?>>
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
          %15 = loom.alloc [1, 64, 128] on @L1 : memref<1x64x128xf32>
          %16 = loom.alloc [1, 128, 128] on @L1 : memref<1x128x128xf32>
          %17 = loom.alloc [1, 64] on @L1 : memref<1x64xf32>
          %18 = loom.alloc [1, 64] on @L1 : memref<1x64xf32>
          %19 = loom.alloc [1, 64] on @L1 : memref<1x64xf32>
          %20 = loom.alloc [1, 64] on @L1 : memref<1x64xf32>
          %21 = loom.alloc [1, 64] on @L1 : memref<1x64xf32>
          %22 = loom.alloc [1, 64, 128] on @L1 : memref<1x64x128xf32>
          %23 = loom.alloc [1, 64, 128] on @L1 : memref<1x64x128xf32>
          %24 = loom.alloc [1, 64, 128] on @L1 : memref<1x64x128xf32>
          %25 = arith.muli %arg6, %c524288 overflow<nsw> : index
          %26 = arith.muli %13, %c8192 : index
          %27 = arith.addi %25, %26 : index
          %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%27], sizes: [1, 64, 128], strides: [524288, 128, 1] : memref<32x4096x128xf32> to memref<1x64x128xf32, strided<[524288, 128, 1], offset: ?>>
          loom.copy %reinterpret_cast, %22 src_mem_space @DRAM dst_mem_space @L1, interconnect : [], broadcast : [1, 1] : memref<1x64x128xf32, strided<[524288, 128, 1], offset: ?>>, memref<1x64x128xf32>
          linalg.fill ins(%cst_1 : f32) outs(%23 : memref<1x64x128xf32>)
          linalg.fill ins(%cst_2 : f32) outs(%17 : memref<1x64xf32>)
          linalg.fill ins(%cst_3 : f32) outs(%18 : memref<1x64xf32>)
          scf.for %arg7 = %c0 to %c32 step %c1 {
            %28 = arith.muli %arg7, %c128 : index
            %29 = arith.addi %25, %28 : index
            %reinterpret_cast_5 = memref.reinterpret_cast %arg0 to offset: [%29], sizes: [1, 128, 128], strides: [524288, 4096, 1] : memref<32x128x4096xf32> to memref<1x128x128xf32, strided<[524288, 4096, 1], offset: ?>>
            loom.copy %reinterpret_cast_5, %16 src_mem_space @DRAM dst_mem_space @L1, interconnect : [@horizontal_links], broadcast : [1, 8] : memref<1x128x128xf32, strided<[524288, 4096, 1], offset: ?>>, memref<1x128x128xf32>
            linalg.fill ins(%cst_1 : f32) outs(%15 : memref<1x64x128xf32>)
            linalg.batch_matmul ins(%22, %16 : memref<1x64x128xf32>, memref<1x128x128xf32>) outs(%15 : memref<1x64x128xf32>)
            linalg.fill ins(%cst_3 : f32) outs(%19 : memref<1x64xf32>)
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%15 : memref<1x64x128xf32>) outs(%19 : memref<1x64xf32>) {
            ^bb0(%in: f32, %out: f32):
              %32 = arith.maximumf %in, %out : f32
              linalg.yield %32 : f32
            }
            linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%18, %19 : memref<1x64xf32>, memref<1x64xf32>) outs(%19 : memref<1x64xf32>) {
            ^bb0(%in: f32, %in_7: f32, %out: f32):
              %32 = arith.truncf %cst_0 : f64 to f32
              %33 = arith.mulf %in_7, %32 : f32
              %34 = arith.cmpf ogt, %in, %33 : f32
              %35 = arith.select %34, %in, %33 : f32
              linalg.yield %35 : f32
            }
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%15, %19 : memref<1x64x128xf32>, memref<1x64xf32>) outs(%15 : memref<1x64x128xf32>) {
            ^bb0(%in: f32, %in_7: f32, %out: f32):
              %32 = arith.truncf %cst_0 : f64 to f32
              %33 = arith.mulf %in, %32 : f32
              %34 = arith.subf %33, %in_7 : f32
              %35 = math.powf %cst, %34 : f32
              linalg.yield %35 : f32
            }
            linalg.fill ins(%cst_1 : f32) outs(%20 : memref<1x64xf32>)
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%15 : memref<1x64x128xf32>) outs(%20 : memref<1x64xf32>) {
            ^bb0(%in: f32, %out: f32):
              %32 = arith.addf %in, %out : f32
              linalg.yield %32 : f32
            }
            linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%18, %19 : memref<1x64xf32>, memref<1x64xf32>) outs(%21 : memref<1x64xf32>) {
            ^bb0(%in: f32, %in_7: f32, %out: f32):
              %32 = arith.subf %in, %in_7 : f32
              %33 = math.powf %cst, %32 : f32
              linalg.yield %33 : f32
            }
            linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%17, %21, %20 : memref<1x64xf32>, memref<1x64xf32>, memref<1x64xf32>) outs(%17 : memref<1x64xf32>) {
            ^bb0(%in: f32, %in_7: f32, %in_8: f32, %out: f32):
              %32 = arith.mulf %in, %in_7 : f32
              %33 = arith.addf %32, %in_8 : f32
              linalg.yield %33 : f32
            }
            %30 = arith.muli %arg7, %c16384 : index
            %31 = arith.addi %25, %30 : index
            %reinterpret_cast_6 = memref.reinterpret_cast %arg1 to offset: [%31], sizes: [1, 128, 128], strides: [524288, 128, 1] : memref<32x4096x128xf32> to memref<1x128x128xf32, strided<[524288, 128, 1], offset: ?>>
            loom.copy %reinterpret_cast_6, %14 src_mem_space @DRAM dst_mem_space @L1, interconnect : [@horizontal_links], broadcast : [1, 8] : memref<1x128x128xf32, strided<[524288, 128, 1], offset: ?>>, memref<1x128x128xf32>
            linalg.fill ins(%cst_1 : f32) outs(%24 : memref<1x64x128xf32>)
            linalg.batch_matmul ins(%15, %14 : memref<1x64x128xf32>, memref<1x128x128xf32>) outs(%24 : memref<1x64x128xf32>)
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%24, %23, %21 : memref<1x64x128xf32>, memref<1x64x128xf32>, memref<1x64xf32>) outs(%23 : memref<1x64x128xf32>) {
            ^bb0(%in: f32, %in_7: f32, %in_8: f32, %out: f32):
              %32 = arith.mulf %in_7, %in_8 : f32
              %33 = arith.addf %in, %32 : f32
              linalg.yield %33 : f32
            }
            linalg.copy ins(%19 : memref<1x64xf32>) outs(%18 : memref<1x64xf32>)
          }
          linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%23, %17 : memref<1x64x128xf32>, memref<1x64xf32>) outs(%22 : memref<1x64x128xf32>) {
          ^bb0(%in: f32, %in_5: f32, %out: f32):
            %28 = arith.divf %in, %in_5 : f32
            linalg.yield %28 : f32
          }
          %reinterpret_cast_4 = memref.reinterpret_cast %arg3 to offset: [%27], sizes: [1, 64, 128], strides: [524288, 128, 1] : memref<32x4096x128xf32> to memref<1x64x128xf32, strided<[524288, 128, 1], offset: ?>>
          loom.copy %22, %reinterpret_cast_4 src_mem_space @L1 dst_mem_space @DRAM, interconnect : [], broadcast : [1, 1] : memref<1x64x128xf32>, memref<1x64x128xf32, strided<[524288, 128, 1], offset: ?>>
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
          %15 = loom.alloc [1, 64, 128] on @L1 : memref<1x64x128xf32>
          %16 = loom.alloc [1, 128, 128] on @L1 : memref<1x128x128xf32>
          %17 = loom.alloc [1, 64] on @L1 : memref<1x64xf32>
          %18 = loom.alloc [1, 64] on @L1 : memref<1x64xf32>
          %19 = loom.alloc [1, 64] on @L1 : memref<1x64xf32>
          %20 = loom.alloc [1, 64] on @L1 : memref<1x64xf32>
          %21 = loom.alloc [1, 64] on @L1 : memref<1x64xf32>
          %22 = loom.alloc [1, 64, 128] on @L1 : memref<1x64x128xf32>
          %23 = loom.alloc [1, 64, 128] on @L1 : memref<1x64x128xf32>
          %24 = loom.alloc [1, 64, 128] on @L1 : memref<1x64x128xf32>
          %25 = arith.muli %arg6, %c524288 overflow<nsw> : index
          %26 = arith.muli %13, %c8192 : index
          %27 = arith.addi %25, %26 : index
          %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%27], sizes: [1, 64, 128], strides: [524288, 128, 1] : memref<32x4096x128xf32> to memref<1x64x128xf32, strided<[524288, 128, 1], offset: ?>>
          loom.copy %reinterpret_cast, %22 src_mem_space @DRAM dst_mem_space @L1, interconnect : [], broadcast : [1, 1] : memref<1x64x128xf32, strided<[524288, 128, 1], offset: ?>>, memref<1x64x128xf32>
          linalg.fill ins(%cst_1 : f32) outs(%23 : memref<1x64x128xf32>)
          linalg.fill ins(%cst_2 : f32) outs(%17 : memref<1x64xf32>)
          linalg.fill ins(%cst_3 : f32) outs(%18 : memref<1x64xf32>)
          scf.for %arg7 = %c0 to %c32 step %c1 {
            %28 = arith.muli %arg7, %c128 : index
            %29 = arith.addi %25, %28 : index
            %reinterpret_cast_5 = memref.reinterpret_cast %arg0 to offset: [%29], sizes: [1, 128, 128], strides: [524288, 4096, 1] : memref<32x128x4096xf32> to memref<1x128x128xf32, strided<[524288, 4096, 1], offset: ?>>
            loom.copy %reinterpret_cast_5, %16 src_mem_space @DRAM dst_mem_space @L1, interconnect : [@horizontal_links], broadcast : [1, 8] : memref<1x128x128xf32, strided<[524288, 4096, 1], offset: ?>>, memref<1x128x128xf32>
            linalg.fill ins(%cst_1 : f32) outs(%15 : memref<1x64x128xf32>)
            linalg.batch_matmul ins(%22, %16 : memref<1x64x128xf32>, memref<1x128x128xf32>) outs(%15 : memref<1x64x128xf32>)
            linalg.fill ins(%cst_3 : f32) outs(%19 : memref<1x64xf32>)
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%15 : memref<1x64x128xf32>) outs(%19 : memref<1x64xf32>) {
            ^bb0(%in: f32, %out: f32):
              %32 = arith.maximumf %in, %out : f32
              linalg.yield %32 : f32
            }
            linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%18, %19 : memref<1x64xf32>, memref<1x64xf32>) outs(%19 : memref<1x64xf32>) {
            ^bb0(%in: f32, %in_7: f32, %out: f32):
              %32 = arith.truncf %cst_0 : f64 to f32
              %33 = arith.mulf %in_7, %32 : f32
              %34 = arith.cmpf ogt, %in, %33 : f32
              %35 = arith.select %34, %in, %33 : f32
              linalg.yield %35 : f32
            }
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%15, %19 : memref<1x64x128xf32>, memref<1x64xf32>) outs(%15 : memref<1x64x128xf32>) {
            ^bb0(%in: f32, %in_7: f32, %out: f32):
              %32 = arith.truncf %cst_0 : f64 to f32
              %33 = arith.mulf %in, %32 : f32
              %34 = arith.subf %33, %in_7 : f32
              %35 = math.powf %cst, %34 : f32
              linalg.yield %35 : f32
            }
            linalg.fill ins(%cst_1 : f32) outs(%20 : memref<1x64xf32>)
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%15 : memref<1x64x128xf32>) outs(%20 : memref<1x64xf32>) {
            ^bb0(%in: f32, %out: f32):
              %32 = arith.addf %in, %out : f32
              linalg.yield %32 : f32
            }
            linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%18, %19 : memref<1x64xf32>, memref<1x64xf32>) outs(%21 : memref<1x64xf32>) {
            ^bb0(%in: f32, %in_7: f32, %out: f32):
              %32 = arith.subf %in, %in_7 : f32
              %33 = math.powf %cst, %32 : f32
              linalg.yield %33 : f32
            }
            linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%17, %21, %20 : memref<1x64xf32>, memref<1x64xf32>, memref<1x64xf32>) outs(%17 : memref<1x64xf32>) {
            ^bb0(%in: f32, %in_7: f32, %in_8: f32, %out: f32):
              %32 = arith.mulf %in, %in_7 : f32
              %33 = arith.addf %32, %in_8 : f32
              linalg.yield %33 : f32
            }
            %30 = arith.muli %arg7, %c16384 : index
            %31 = arith.addi %25, %30 : index
            %reinterpret_cast_6 = memref.reinterpret_cast %arg1 to offset: [%31], sizes: [1, 128, 128], strides: [524288, 128, 1] : memref<32x4096x128xf32> to memref<1x128x128xf32, strided<[524288, 128, 1], offset: ?>>
            loom.copy %reinterpret_cast_6, %14 src_mem_space @DRAM dst_mem_space @L1, interconnect : [@vertical_links], broadcast : [8, 1] : memref<1x128x128xf32, strided<[524288, 128, 1], offset: ?>>, memref<1x128x128xf32>
            linalg.fill ins(%cst_1 : f32) outs(%24 : memref<1x64x128xf32>)
            linalg.batch_matmul ins(%15, %14 : memref<1x64x128xf32>, memref<1x128x128xf32>) outs(%24 : memref<1x64x128xf32>)
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%24, %23, %21 : memref<1x64x128xf32>, memref<1x64x128xf32>, memref<1x64xf32>) outs(%23 : memref<1x64x128xf32>) {
            ^bb0(%in: f32, %in_7: f32, %in_8: f32, %out: f32):
              %32 = arith.mulf %in_7, %in_8 : f32
              %33 = arith.addf %in, %32 : f32
              linalg.yield %33 : f32
            }
            linalg.copy ins(%19 : memref<1x64xf32>) outs(%18 : memref<1x64xf32>)
          }
          linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%23, %17 : memref<1x64x128xf32>, memref<1x64xf32>) outs(%22 : memref<1x64x128xf32>) {
          ^bb0(%in: f32, %in_5: f32, %out: f32):
            %28 = arith.divf %in, %in_5 : f32
            linalg.yield %28 : f32
          }
          %reinterpret_cast_4 = memref.reinterpret_cast %arg3 to offset: [%27], sizes: [1, 64, 128], strides: [524288, 128, 1] : memref<32x4096x128xf32> to memref<1x64x128xf32, strided<[524288, 128, 1], offset: ?>>
          loom.copy %22, %reinterpret_cast_4 src_mem_space @L1 dst_mem_space @DRAM, interconnect : [], broadcast : [1, 1] : memref<1x64x128xf32>, memref<1x64x128xf32, strided<[524288, 128, 1], offset: ?>>
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
          %15 = loom.alloc [1, 64, 128] on @L1 : memref<1x64x128xf32>
          %16 = loom.alloc [1, 128, 128] on @L1 : memref<1x128x128xf32>
          %17 = loom.alloc [1, 64] on @L1 : memref<1x64xf32>
          %18 = loom.alloc [1, 64] on @L1 : memref<1x64xf32>
          %19 = loom.alloc [1, 64] on @L1 : memref<1x64xf32>
          %20 = loom.alloc [1, 64] on @L1 : memref<1x64xf32>
          %21 = loom.alloc [1, 64] on @L1 : memref<1x64xf32>
          %22 = loom.alloc [1, 64, 128] on @L1 : memref<1x64x128xf32>
          %23 = loom.alloc [1, 64, 128] on @L1 : memref<1x64x128xf32>
          %24 = loom.alloc [1, 64, 128] on @L1 : memref<1x64x128xf32>
          %25 = arith.muli %arg6, %c524288 overflow<nsw> : index
          %26 = arith.muli %13, %c8192 : index
          %27 = arith.addi %25, %26 : index
          %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%27], sizes: [1, 64, 128], strides: [524288, 128, 1] : memref<32x4096x128xf32> to memref<1x64x128xf32, strided<[524288, 128, 1], offset: ?>>
          loom.copy %reinterpret_cast, %22 src_mem_space @DRAM dst_mem_space @L1, interconnect : [], broadcast : [1, 1] : memref<1x64x128xf32, strided<[524288, 128, 1], offset: ?>>, memref<1x64x128xf32>
          linalg.fill ins(%cst_1 : f32) outs(%23 : memref<1x64x128xf32>)
          linalg.fill ins(%cst_2 : f32) outs(%17 : memref<1x64xf32>)
          linalg.fill ins(%cst_3 : f32) outs(%18 : memref<1x64xf32>)
          scf.for %arg7 = %c0 to %c32 step %c1 {
            %28 = arith.muli %arg7, %c128 : index
            %29 = arith.addi %25, %28 : index
            %reinterpret_cast_5 = memref.reinterpret_cast %arg0 to offset: [%29], sizes: [1, 128, 128], strides: [524288, 4096, 1] : memref<32x128x4096xf32> to memref<1x128x128xf32, strided<[524288, 4096, 1], offset: ?>>
            loom.copy %reinterpret_cast_5, %16 src_mem_space @DRAM dst_mem_space @L1, interconnect : [@vertical_links], broadcast : [8, 1] : memref<1x128x128xf32, strided<[524288, 4096, 1], offset: ?>>, memref<1x128x128xf32>
            linalg.fill ins(%cst_1 : f32) outs(%15 : memref<1x64x128xf32>)
            linalg.batch_matmul ins(%22, %16 : memref<1x64x128xf32>, memref<1x128x128xf32>) outs(%15 : memref<1x64x128xf32>)
            linalg.fill ins(%cst_3 : f32) outs(%19 : memref<1x64xf32>)
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%15 : memref<1x64x128xf32>) outs(%19 : memref<1x64xf32>) {
            ^bb0(%in: f32, %out: f32):
              %32 = arith.maximumf %in, %out : f32
              linalg.yield %32 : f32
            }
            linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%18, %19 : memref<1x64xf32>, memref<1x64xf32>) outs(%19 : memref<1x64xf32>) {
            ^bb0(%in: f32, %in_7: f32, %out: f32):
              %32 = arith.truncf %cst_0 : f64 to f32
              %33 = arith.mulf %in_7, %32 : f32
              %34 = arith.cmpf ogt, %in, %33 : f32
              %35 = arith.select %34, %in, %33 : f32
              linalg.yield %35 : f32
            }
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%15, %19 : memref<1x64x128xf32>, memref<1x64xf32>) outs(%15 : memref<1x64x128xf32>) {
            ^bb0(%in: f32, %in_7: f32, %out: f32):
              %32 = arith.truncf %cst_0 : f64 to f32
              %33 = arith.mulf %in, %32 : f32
              %34 = arith.subf %33, %in_7 : f32
              %35 = math.powf %cst, %34 : f32
              linalg.yield %35 : f32
            }
            linalg.fill ins(%cst_1 : f32) outs(%20 : memref<1x64xf32>)
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%15 : memref<1x64x128xf32>) outs(%20 : memref<1x64xf32>) {
            ^bb0(%in: f32, %out: f32):
              %32 = arith.addf %in, %out : f32
              linalg.yield %32 : f32
            }
            linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%18, %19 : memref<1x64xf32>, memref<1x64xf32>) outs(%21 : memref<1x64xf32>) {
            ^bb0(%in: f32, %in_7: f32, %out: f32):
              %32 = arith.subf %in, %in_7 : f32
              %33 = math.powf %cst, %32 : f32
              linalg.yield %33 : f32
            }
            linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%17, %21, %20 : memref<1x64xf32>, memref<1x64xf32>, memref<1x64xf32>) outs(%17 : memref<1x64xf32>) {
            ^bb0(%in: f32, %in_7: f32, %in_8: f32, %out: f32):
              %32 = arith.mulf %in, %in_7 : f32
              %33 = arith.addf %32, %in_8 : f32
              linalg.yield %33 : f32
            }
            %30 = arith.muli %arg7, %c16384 : index
            %31 = arith.addi %25, %30 : index
            %reinterpret_cast_6 = memref.reinterpret_cast %arg1 to offset: [%31], sizes: [1, 128, 128], strides: [524288, 128, 1] : memref<32x4096x128xf32> to memref<1x128x128xf32, strided<[524288, 128, 1], offset: ?>>
            loom.copy %reinterpret_cast_6, %14 src_mem_space @DRAM dst_mem_space @L1, interconnect : [], broadcast : [1, 1] : memref<1x128x128xf32, strided<[524288, 128, 1], offset: ?>>, memref<1x128x128xf32>
            linalg.fill ins(%cst_1 : f32) outs(%24 : memref<1x64x128xf32>)
            linalg.batch_matmul ins(%15, %14 : memref<1x64x128xf32>, memref<1x128x128xf32>) outs(%24 : memref<1x64x128xf32>)
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%24, %23, %21 : memref<1x64x128xf32>, memref<1x64x128xf32>, memref<1x64xf32>) outs(%23 : memref<1x64x128xf32>) {
            ^bb0(%in: f32, %in_7: f32, %in_8: f32, %out: f32):
              %32 = arith.mulf %in_7, %in_8 : f32
              %33 = arith.addf %in, %32 : f32
              linalg.yield %33 : f32
            }
            linalg.copy ins(%19 : memref<1x64xf32>) outs(%18 : memref<1x64xf32>)
          }
          linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%23, %17 : memref<1x64x128xf32>, memref<1x64xf32>) outs(%22 : memref<1x64x128xf32>) {
          ^bb0(%in: f32, %in_5: f32, %out: f32):
            %28 = arith.divf %in, %in_5 : f32
            linalg.yield %28 : f32
          }
          %reinterpret_cast_4 = memref.reinterpret_cast %arg3 to offset: [%27], sizes: [1, 64, 128], strides: [524288, 128, 1] : memref<32x4096x128xf32> to memref<1x64x128xf32, strided<[524288, 128, 1], offset: ?>>
          loom.copy %22, %reinterpret_cast_4 src_mem_space @L1 dst_mem_space @DRAM, interconnect : [], broadcast : [1, 1] : memref<1x64x128xf32>, memref<1x64x128xf32, strided<[524288, 128, 1], offset: ?>>
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
          %15 = loom.alloc [1, 64, 128] on @L1 : memref<1x64x128xf32>
          %16 = loom.alloc [1, 128, 128] on @L1 : memref<1x128x128xf32>
          %17 = loom.alloc [1, 64] on @L1 : memref<1x64xf32>
          %18 = loom.alloc [1, 64] on @L1 : memref<1x64xf32>
          %19 = loom.alloc [1, 64] on @L1 : memref<1x64xf32>
          %20 = loom.alloc [1, 64] on @L1 : memref<1x64xf32>
          %21 = loom.alloc [1, 64] on @L1 : memref<1x64xf32>
          %22 = loom.alloc [1, 64, 128] on @L1 : memref<1x64x128xf32>
          %23 = loom.alloc [1, 64, 128] on @L1 : memref<1x64x128xf32>
          %24 = loom.alloc [1, 64, 128] on @L1 : memref<1x64x128xf32>
          %25 = arith.muli %arg6, %c524288 overflow<nsw> : index
          %26 = arith.muli %13, %c8192 : index
          %27 = arith.addi %25, %26 : index
          %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%27], sizes: [1, 64, 128], strides: [524288, 128, 1] : memref<32x4096x128xf32> to memref<1x64x128xf32, strided<[524288, 128, 1], offset: ?>>
          loom.copy %reinterpret_cast, %22 src_mem_space @DRAM dst_mem_space @L1, interconnect : [], broadcast : [1, 1] : memref<1x64x128xf32, strided<[524288, 128, 1], offset: ?>>, memref<1x64x128xf32>
          linalg.fill ins(%cst_1 : f32) outs(%23 : memref<1x64x128xf32>)
          linalg.fill ins(%cst_2 : f32) outs(%17 : memref<1x64xf32>)
          linalg.fill ins(%cst_3 : f32) outs(%18 : memref<1x64xf32>)
          scf.for %arg7 = %c0 to %c32 step %c1 {
            %28 = arith.muli %arg7, %c128 : index
            %29 = arith.addi %25, %28 : index
            %reinterpret_cast_5 = memref.reinterpret_cast %arg0 to offset: [%29], sizes: [1, 128, 128], strides: [524288, 4096, 1] : memref<32x128x4096xf32> to memref<1x128x128xf32, strided<[524288, 4096, 1], offset: ?>>
            loom.copy %reinterpret_cast_5, %16 src_mem_space @DRAM dst_mem_space @L1, interconnect : [@vertical_links], broadcast : [8, 1] : memref<1x128x128xf32, strided<[524288, 4096, 1], offset: ?>>, memref<1x128x128xf32>
            linalg.fill ins(%cst_1 : f32) outs(%15 : memref<1x64x128xf32>)
            linalg.batch_matmul ins(%22, %16 : memref<1x64x128xf32>, memref<1x128x128xf32>) outs(%15 : memref<1x64x128xf32>)
            linalg.fill ins(%cst_3 : f32) outs(%19 : memref<1x64xf32>)
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%15 : memref<1x64x128xf32>) outs(%19 : memref<1x64xf32>) {
            ^bb0(%in: f32, %out: f32):
              %32 = arith.maximumf %in, %out : f32
              linalg.yield %32 : f32
            }
            linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%18, %19 : memref<1x64xf32>, memref<1x64xf32>) outs(%19 : memref<1x64xf32>) {
            ^bb0(%in: f32, %in_7: f32, %out: f32):
              %32 = arith.truncf %cst_0 : f64 to f32
              %33 = arith.mulf %in_7, %32 : f32
              %34 = arith.cmpf ogt, %in, %33 : f32
              %35 = arith.select %34, %in, %33 : f32
              linalg.yield %35 : f32
            }
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%15, %19 : memref<1x64x128xf32>, memref<1x64xf32>) outs(%15 : memref<1x64x128xf32>) {
            ^bb0(%in: f32, %in_7: f32, %out: f32):
              %32 = arith.truncf %cst_0 : f64 to f32
              %33 = arith.mulf %in, %32 : f32
              %34 = arith.subf %33, %in_7 : f32
              %35 = math.powf %cst, %34 : f32
              linalg.yield %35 : f32
            }
            linalg.fill ins(%cst_1 : f32) outs(%20 : memref<1x64xf32>)
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%15 : memref<1x64x128xf32>) outs(%20 : memref<1x64xf32>) {
            ^bb0(%in: f32, %out: f32):
              %32 = arith.addf %in, %out : f32
              linalg.yield %32 : f32
            }
            linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%18, %19 : memref<1x64xf32>, memref<1x64xf32>) outs(%21 : memref<1x64xf32>) {
            ^bb0(%in: f32, %in_7: f32, %out: f32):
              %32 = arith.subf %in, %in_7 : f32
              %33 = math.powf %cst, %32 : f32
              linalg.yield %33 : f32
            }
            linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%17, %21, %20 : memref<1x64xf32>, memref<1x64xf32>, memref<1x64xf32>) outs(%17 : memref<1x64xf32>) {
            ^bb0(%in: f32, %in_7: f32, %in_8: f32, %out: f32):
              %32 = arith.mulf %in, %in_7 : f32
              %33 = arith.addf %32, %in_8 : f32
              linalg.yield %33 : f32
            }
            %30 = arith.muli %arg7, %c16384 : index
            %31 = arith.addi %25, %30 : index
            %reinterpret_cast_6 = memref.reinterpret_cast %arg1 to offset: [%31], sizes: [1, 128, 128], strides: [524288, 128, 1] : memref<32x4096x128xf32> to memref<1x128x128xf32, strided<[524288, 128, 1], offset: ?>>
            loom.copy %reinterpret_cast_6, %14 src_mem_space @DRAM dst_mem_space @L1, interconnect : [@horizontal_links, @vertical_links], broadcast : [8, 8] : memref<1x128x128xf32, strided<[524288, 128, 1], offset: ?>>, memref<1x128x128xf32>
            linalg.fill ins(%cst_1 : f32) outs(%24 : memref<1x64x128xf32>)
            linalg.batch_matmul ins(%15, %14 : memref<1x64x128xf32>, memref<1x128x128xf32>) outs(%24 : memref<1x64x128xf32>)
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%24, %23, %21 : memref<1x64x128xf32>, memref<1x64x128xf32>, memref<1x64xf32>) outs(%23 : memref<1x64x128xf32>) {
            ^bb0(%in: f32, %in_7: f32, %in_8: f32, %out: f32):
              %32 = arith.mulf %in_7, %in_8 : f32
              %33 = arith.addf %in, %32 : f32
              linalg.yield %33 : f32
            }
            linalg.copy ins(%19 : memref<1x64xf32>) outs(%18 : memref<1x64xf32>)
          }
          linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%23, %17 : memref<1x64x128xf32>, memref<1x64xf32>) outs(%22 : memref<1x64x128xf32>) {
          ^bb0(%in: f32, %in_5: f32, %out: f32):
            %28 = arith.divf %in, %in_5 : f32
            linalg.yield %28 : f32
          }
          %reinterpret_cast_4 = memref.reinterpret_cast %arg3 to offset: [%27], sizes: [1, 64, 128], strides: [524288, 128, 1] : memref<32x4096x128xf32> to memref<1x64x128xf32, strided<[524288, 128, 1], offset: ?>>
          loom.copy %22, %reinterpret_cast_4 src_mem_space @L1 dst_mem_space @DRAM, interconnect : [], broadcast : [1, 1] : memref<1x64x128xf32>, memref<1x64x128xf32, strided<[524288, 128, 1], offset: ?>>
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
          %15 = loom.alloc [1, 64, 128] on @L1 : memref<1x64x128xf32>
          %16 = loom.alloc [1, 128, 128] on @L1 : memref<1x128x128xf32>
          %17 = loom.alloc [1, 64] on @L1 : memref<1x64xf32>
          %18 = loom.alloc [1, 64] on @L1 : memref<1x64xf32>
          %19 = loom.alloc [1, 64] on @L1 : memref<1x64xf32>
          %20 = loom.alloc [1, 64] on @L1 : memref<1x64xf32>
          %21 = loom.alloc [1, 64] on @L1 : memref<1x64xf32>
          %22 = loom.alloc [1, 64, 128] on @L1 : memref<1x64x128xf32>
          %23 = loom.alloc [1, 64, 128] on @L1 : memref<1x64x128xf32>
          %24 = loom.alloc [1, 64, 128] on @L1 : memref<1x64x128xf32>
          %25 = arith.muli %arg6, %c524288 overflow<nsw> : index
          %26 = arith.muli %13, %c8192 : index
          %27 = arith.addi %25, %26 : index
          %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%27], sizes: [1, 64, 128], strides: [524288, 128, 1] : memref<32x4096x128xf32> to memref<1x64x128xf32, strided<[524288, 128, 1], offset: ?>>
          loom.copy %reinterpret_cast, %22 src_mem_space @DRAM dst_mem_space @L1, interconnect : [], broadcast : [1, 1] : memref<1x64x128xf32, strided<[524288, 128, 1], offset: ?>>, memref<1x64x128xf32>
          linalg.fill ins(%cst_1 : f32) outs(%23 : memref<1x64x128xf32>)
          linalg.fill ins(%cst_2 : f32) outs(%17 : memref<1x64xf32>)
          linalg.fill ins(%cst_3 : f32) outs(%18 : memref<1x64xf32>)
          scf.for %arg7 = %c0 to %c32 step %c1 {
            %28 = arith.muli %arg7, %c128 : index
            %29 = arith.addi %25, %28 : index
            %reinterpret_cast_5 = memref.reinterpret_cast %arg0 to offset: [%29], sizes: [1, 128, 128], strides: [524288, 4096, 1] : memref<32x128x4096xf32> to memref<1x128x128xf32, strided<[524288, 4096, 1], offset: ?>>
            loom.copy %reinterpret_cast_5, %16 src_mem_space @DRAM dst_mem_space @L1, interconnect : [@vertical_links], broadcast : [8, 1] : memref<1x128x128xf32, strided<[524288, 4096, 1], offset: ?>>, memref<1x128x128xf32>
            linalg.fill ins(%cst_1 : f32) outs(%15 : memref<1x64x128xf32>)
            linalg.batch_matmul ins(%22, %16 : memref<1x64x128xf32>, memref<1x128x128xf32>) outs(%15 : memref<1x64x128xf32>)
            linalg.fill ins(%cst_3 : f32) outs(%19 : memref<1x64xf32>)
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%15 : memref<1x64x128xf32>) outs(%19 : memref<1x64xf32>) {
            ^bb0(%in: f32, %out: f32):
              %32 = arith.maximumf %in, %out : f32
              linalg.yield %32 : f32
            }
            linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%18, %19 : memref<1x64xf32>, memref<1x64xf32>) outs(%19 : memref<1x64xf32>) {
            ^bb0(%in: f32, %in_7: f32, %out: f32):
              %32 = arith.truncf %cst_0 : f64 to f32
              %33 = arith.mulf %in_7, %32 : f32
              %34 = arith.cmpf ogt, %in, %33 : f32
              %35 = arith.select %34, %in, %33 : f32
              linalg.yield %35 : f32
            }
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%15, %19 : memref<1x64x128xf32>, memref<1x64xf32>) outs(%15 : memref<1x64x128xf32>) {
            ^bb0(%in: f32, %in_7: f32, %out: f32):
              %32 = arith.truncf %cst_0 : f64 to f32
              %33 = arith.mulf %in, %32 : f32
              %34 = arith.subf %33, %in_7 : f32
              %35 = math.powf %cst, %34 : f32
              linalg.yield %35 : f32
            }
            linalg.fill ins(%cst_1 : f32) outs(%20 : memref<1x64xf32>)
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%15 : memref<1x64x128xf32>) outs(%20 : memref<1x64xf32>) {
            ^bb0(%in: f32, %out: f32):
              %32 = arith.addf %in, %out : f32
              linalg.yield %32 : f32
            }
            linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%18, %19 : memref<1x64xf32>, memref<1x64xf32>) outs(%21 : memref<1x64xf32>) {
            ^bb0(%in: f32, %in_7: f32, %out: f32):
              %32 = arith.subf %in, %in_7 : f32
              %33 = math.powf %cst, %32 : f32
              linalg.yield %33 : f32
            }
            linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%17, %21, %20 : memref<1x64xf32>, memref<1x64xf32>, memref<1x64xf32>) outs(%17 : memref<1x64xf32>) {
            ^bb0(%in: f32, %in_7: f32, %in_8: f32, %out: f32):
              %32 = arith.mulf %in, %in_7 : f32
              %33 = arith.addf %32, %in_8 : f32
              linalg.yield %33 : f32
            }
            %30 = arith.muli %arg7, %c16384 : index
            %31 = arith.addi %25, %30 : index
            %reinterpret_cast_6 = memref.reinterpret_cast %arg1 to offset: [%31], sizes: [1, 128, 128], strides: [524288, 128, 1] : memref<32x4096x128xf32> to memref<1x128x128xf32, strided<[524288, 128, 1], offset: ?>>
            loom.copy %reinterpret_cast_6, %14 src_mem_space @DRAM dst_mem_space @L1, interconnect : [@horizontal_links], broadcast : [1, 8] : memref<1x128x128xf32, strided<[524288, 128, 1], offset: ?>>, memref<1x128x128xf32>
            linalg.fill ins(%cst_1 : f32) outs(%24 : memref<1x64x128xf32>)
            linalg.batch_matmul ins(%15, %14 : memref<1x64x128xf32>, memref<1x128x128xf32>) outs(%24 : memref<1x64x128xf32>)
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%24, %23, %21 : memref<1x64x128xf32>, memref<1x64x128xf32>, memref<1x64xf32>) outs(%23 : memref<1x64x128xf32>) {
            ^bb0(%in: f32, %in_7: f32, %in_8: f32, %out: f32):
              %32 = arith.mulf %in_7, %in_8 : f32
              %33 = arith.addf %in, %32 : f32
              linalg.yield %33 : f32
            }
            linalg.copy ins(%19 : memref<1x64xf32>) outs(%18 : memref<1x64xf32>)
          }
          linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%23, %17 : memref<1x64x128xf32>, memref<1x64xf32>) outs(%22 : memref<1x64x128xf32>) {
          ^bb0(%in: f32, %in_5: f32, %out: f32):
            %28 = arith.divf %in, %in_5 : f32
            linalg.yield %28 : f32
          }
          %reinterpret_cast_4 = memref.reinterpret_cast %arg3 to offset: [%27], sizes: [1, 64, 128], strides: [524288, 128, 1] : memref<32x4096x128xf32> to memref<1x64x128xf32, strided<[524288, 128, 1], offset: ?>>
          loom.copy %22, %reinterpret_cast_4 src_mem_space @L1 dst_mem_space @DRAM, interconnect : [], broadcast : [1, 1] : memref<1x64x128xf32>, memref<1x64x128xf32, strided<[524288, 128, 1], offset: ?>>
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
          %15 = loom.alloc [1, 64, 128] on @L1 : memref<1x64x128xf32>
          %16 = loom.alloc [1, 128, 128] on @L1 : memref<1x128x128xf32>
          %17 = loom.alloc [1, 64] on @L1 : memref<1x64xf32>
          %18 = loom.alloc [1, 64] on @L1 : memref<1x64xf32>
          %19 = loom.alloc [1, 64] on @L1 : memref<1x64xf32>
          %20 = loom.alloc [1, 64] on @L1 : memref<1x64xf32>
          %21 = loom.alloc [1, 64] on @L1 : memref<1x64xf32>
          %22 = loom.alloc [1, 64, 128] on @L1 : memref<1x64x128xf32>
          %23 = loom.alloc [1, 64, 128] on @L1 : memref<1x64x128xf32>
          %24 = loom.alloc [1, 64, 128] on @L1 : memref<1x64x128xf32>
          %25 = arith.muli %arg6, %c524288 overflow<nsw> : index
          %26 = arith.muli %13, %c8192 : index
          %27 = arith.addi %25, %26 : index
          %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%27], sizes: [1, 64, 128], strides: [524288, 128, 1] : memref<32x4096x128xf32> to memref<1x64x128xf32, strided<[524288, 128, 1], offset: ?>>
          loom.copy %reinterpret_cast, %22 src_mem_space @DRAM dst_mem_space @L1, interconnect : [], broadcast : [1, 1] : memref<1x64x128xf32, strided<[524288, 128, 1], offset: ?>>, memref<1x64x128xf32>
          linalg.fill ins(%cst_1 : f32) outs(%23 : memref<1x64x128xf32>)
          linalg.fill ins(%cst_2 : f32) outs(%17 : memref<1x64xf32>)
          linalg.fill ins(%cst_3 : f32) outs(%18 : memref<1x64xf32>)
          scf.for %arg7 = %c0 to %c32 step %c1 {
            %28 = arith.muli %arg7, %c128 : index
            %29 = arith.addi %25, %28 : index
            %reinterpret_cast_5 = memref.reinterpret_cast %arg0 to offset: [%29], sizes: [1, 128, 128], strides: [524288, 4096, 1] : memref<32x128x4096xf32> to memref<1x128x128xf32, strided<[524288, 4096, 1], offset: ?>>
            loom.copy %reinterpret_cast_5, %16 src_mem_space @DRAM dst_mem_space @L1, interconnect : [@vertical_links], broadcast : [8, 1] : memref<1x128x128xf32, strided<[524288, 4096, 1], offset: ?>>, memref<1x128x128xf32>
            linalg.fill ins(%cst_1 : f32) outs(%15 : memref<1x64x128xf32>)
            linalg.batch_matmul ins(%22, %16 : memref<1x64x128xf32>, memref<1x128x128xf32>) outs(%15 : memref<1x64x128xf32>)
            linalg.fill ins(%cst_3 : f32) outs(%19 : memref<1x64xf32>)
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%15 : memref<1x64x128xf32>) outs(%19 : memref<1x64xf32>) {
            ^bb0(%in: f32, %out: f32):
              %32 = arith.maximumf %in, %out : f32
              linalg.yield %32 : f32
            }
            linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%18, %19 : memref<1x64xf32>, memref<1x64xf32>) outs(%19 : memref<1x64xf32>) {
            ^bb0(%in: f32, %in_7: f32, %out: f32):
              %32 = arith.truncf %cst_0 : f64 to f32
              %33 = arith.mulf %in_7, %32 : f32
              %34 = arith.cmpf ogt, %in, %33 : f32
              %35 = arith.select %34, %in, %33 : f32
              linalg.yield %35 : f32
            }
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%15, %19 : memref<1x64x128xf32>, memref<1x64xf32>) outs(%15 : memref<1x64x128xf32>) {
            ^bb0(%in: f32, %in_7: f32, %out: f32):
              %32 = arith.truncf %cst_0 : f64 to f32
              %33 = arith.mulf %in, %32 : f32
              %34 = arith.subf %33, %in_7 : f32
              %35 = math.powf %cst, %34 : f32
              linalg.yield %35 : f32
            }
            linalg.fill ins(%cst_1 : f32) outs(%20 : memref<1x64xf32>)
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%15 : memref<1x64x128xf32>) outs(%20 : memref<1x64xf32>) {
            ^bb0(%in: f32, %out: f32):
              %32 = arith.addf %in, %out : f32
              linalg.yield %32 : f32
            }
            linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%18, %19 : memref<1x64xf32>, memref<1x64xf32>) outs(%21 : memref<1x64xf32>) {
            ^bb0(%in: f32, %in_7: f32, %out: f32):
              %32 = arith.subf %in, %in_7 : f32
              %33 = math.powf %cst, %32 : f32
              linalg.yield %33 : f32
            }
            linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%17, %21, %20 : memref<1x64xf32>, memref<1x64xf32>, memref<1x64xf32>) outs(%17 : memref<1x64xf32>) {
            ^bb0(%in: f32, %in_7: f32, %in_8: f32, %out: f32):
              %32 = arith.mulf %in, %in_7 : f32
              %33 = arith.addf %32, %in_8 : f32
              linalg.yield %33 : f32
            }
            %30 = arith.muli %arg7, %c16384 : index
            %31 = arith.addi %25, %30 : index
            %reinterpret_cast_6 = memref.reinterpret_cast %arg1 to offset: [%31], sizes: [1, 128, 128], strides: [524288, 128, 1] : memref<32x4096x128xf32> to memref<1x128x128xf32, strided<[524288, 128, 1], offset: ?>>
            loom.copy %reinterpret_cast_6, %14 src_mem_space @DRAM dst_mem_space @L1, interconnect : [@vertical_links], broadcast : [8, 1] : memref<1x128x128xf32, strided<[524288, 128, 1], offset: ?>>, memref<1x128x128xf32>
            linalg.fill ins(%cst_1 : f32) outs(%24 : memref<1x64x128xf32>)
            linalg.batch_matmul ins(%15, %14 : memref<1x64x128xf32>, memref<1x128x128xf32>) outs(%24 : memref<1x64x128xf32>)
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%24, %23, %21 : memref<1x64x128xf32>, memref<1x64x128xf32>, memref<1x64xf32>) outs(%23 : memref<1x64x128xf32>) {
            ^bb0(%in: f32, %in_7: f32, %in_8: f32, %out: f32):
              %32 = arith.mulf %in_7, %in_8 : f32
              %33 = arith.addf %in, %32 : f32
              linalg.yield %33 : f32
            }
            linalg.copy ins(%19 : memref<1x64xf32>) outs(%18 : memref<1x64xf32>)
          }
          linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%23, %17 : memref<1x64x128xf32>, memref<1x64xf32>) outs(%22 : memref<1x64x128xf32>) {
          ^bb0(%in: f32, %in_5: f32, %out: f32):
            %28 = arith.divf %in, %in_5 : f32
            linalg.yield %28 : f32
          }
          %reinterpret_cast_4 = memref.reinterpret_cast %arg3 to offset: [%27], sizes: [1, 64, 128], strides: [524288, 128, 1] : memref<32x4096x128xf32> to memref<1x64x128xf32, strided<[524288, 128, 1], offset: ?>>
          loom.copy %22, %reinterpret_cast_4 src_mem_space @L1 dst_mem_space @DRAM, interconnect : [], broadcast : [1, 1] : memref<1x64x128xf32>, memref<1x64x128xf32, strided<[524288, 128, 1], offset: ?>>
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
          %15 = loom.alloc [1, 64, 128] on @L1 : memref<1x64x128xf32>
          %16 = loom.alloc [1, 128, 128] on @L1 : memref<1x128x128xf32>
          %17 = loom.alloc [1, 64] on @L1 : memref<1x64xf32>
          %18 = loom.alloc [1, 64] on @L1 : memref<1x64xf32>
          %19 = loom.alloc [1, 64] on @L1 : memref<1x64xf32>
          %20 = loom.alloc [1, 64] on @L1 : memref<1x64xf32>
          %21 = loom.alloc [1, 64] on @L1 : memref<1x64xf32>
          %22 = loom.alloc [1, 64, 128] on @L1 : memref<1x64x128xf32>
          %23 = loom.alloc [1, 64, 128] on @L1 : memref<1x64x128xf32>
          %24 = loom.alloc [1, 64, 128] on @L1 : memref<1x64x128xf32>
          %25 = arith.muli %arg6, %c524288 overflow<nsw> : index
          %26 = arith.muli %13, %c8192 : index
          %27 = arith.addi %25, %26 : index
          %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%27], sizes: [1, 64, 128], strides: [524288, 128, 1] : memref<32x4096x128xf32> to memref<1x64x128xf32, strided<[524288, 128, 1], offset: ?>>
          loom.copy %reinterpret_cast, %22 src_mem_space @DRAM dst_mem_space @L1, interconnect : [], broadcast : [1, 1] : memref<1x64x128xf32, strided<[524288, 128, 1], offset: ?>>, memref<1x64x128xf32>
          linalg.fill ins(%cst_1 : f32) outs(%23 : memref<1x64x128xf32>)
          linalg.fill ins(%cst_2 : f32) outs(%17 : memref<1x64xf32>)
          linalg.fill ins(%cst_3 : f32) outs(%18 : memref<1x64xf32>)
          scf.for %arg7 = %c0 to %c32 step %c1 {
            %28 = arith.muli %arg7, %c128 : index
            %29 = arith.addi %25, %28 : index
            %reinterpret_cast_5 = memref.reinterpret_cast %arg0 to offset: [%29], sizes: [1, 128, 128], strides: [524288, 4096, 1] : memref<32x128x4096xf32> to memref<1x128x128xf32, strided<[524288, 4096, 1], offset: ?>>
            loom.copy %reinterpret_cast_5, %16 src_mem_space @DRAM dst_mem_space @L1, interconnect : [], broadcast : [1, 1] : memref<1x128x128xf32, strided<[524288, 4096, 1], offset: ?>>, memref<1x128x128xf32>
            linalg.fill ins(%cst_1 : f32) outs(%15 : memref<1x64x128xf32>)
            linalg.batch_matmul ins(%22, %16 : memref<1x64x128xf32>, memref<1x128x128xf32>) outs(%15 : memref<1x64x128xf32>)
            linalg.fill ins(%cst_3 : f32) outs(%19 : memref<1x64xf32>)
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%15 : memref<1x64x128xf32>) outs(%19 : memref<1x64xf32>) {
            ^bb0(%in: f32, %out: f32):
              %32 = arith.maximumf %in, %out : f32
              linalg.yield %32 : f32
            }
            linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%18, %19 : memref<1x64xf32>, memref<1x64xf32>) outs(%19 : memref<1x64xf32>) {
            ^bb0(%in: f32, %in_7: f32, %out: f32):
              %32 = arith.truncf %cst_0 : f64 to f32
              %33 = arith.mulf %in_7, %32 : f32
              %34 = arith.cmpf ogt, %in, %33 : f32
              %35 = arith.select %34, %in, %33 : f32
              linalg.yield %35 : f32
            }
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%15, %19 : memref<1x64x128xf32>, memref<1x64xf32>) outs(%15 : memref<1x64x128xf32>) {
            ^bb0(%in: f32, %in_7: f32, %out: f32):
              %32 = arith.truncf %cst_0 : f64 to f32
              %33 = arith.mulf %in, %32 : f32
              %34 = arith.subf %33, %in_7 : f32
              %35 = math.powf %cst, %34 : f32
              linalg.yield %35 : f32
            }
            linalg.fill ins(%cst_1 : f32) outs(%20 : memref<1x64xf32>)
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%15 : memref<1x64x128xf32>) outs(%20 : memref<1x64xf32>) {
            ^bb0(%in: f32, %out: f32):
              %32 = arith.addf %in, %out : f32
              linalg.yield %32 : f32
            }
            linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%18, %19 : memref<1x64xf32>, memref<1x64xf32>) outs(%21 : memref<1x64xf32>) {
            ^bb0(%in: f32, %in_7: f32, %out: f32):
              %32 = arith.subf %in, %in_7 : f32
              %33 = math.powf %cst, %32 : f32
              linalg.yield %33 : f32
            }
            linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%17, %21, %20 : memref<1x64xf32>, memref<1x64xf32>, memref<1x64xf32>) outs(%17 : memref<1x64xf32>) {
            ^bb0(%in: f32, %in_7: f32, %in_8: f32, %out: f32):
              %32 = arith.mulf %in, %in_7 : f32
              %33 = arith.addf %32, %in_8 : f32
              linalg.yield %33 : f32
            }
            %30 = arith.muli %arg7, %c16384 : index
            %31 = arith.addi %25, %30 : index
            %reinterpret_cast_6 = memref.reinterpret_cast %arg1 to offset: [%31], sizes: [1, 128, 128], strides: [524288, 128, 1] : memref<32x4096x128xf32> to memref<1x128x128xf32, strided<[524288, 128, 1], offset: ?>>
            loom.copy %reinterpret_cast_6, %14 src_mem_space @DRAM dst_mem_space @L1, interconnect : [], broadcast : [1, 1] : memref<1x128x128xf32, strided<[524288, 128, 1], offset: ?>>, memref<1x128x128xf32>
            linalg.fill ins(%cst_1 : f32) outs(%24 : memref<1x64x128xf32>)
            linalg.batch_matmul ins(%15, %14 : memref<1x64x128xf32>, memref<1x128x128xf32>) outs(%24 : memref<1x64x128xf32>)
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%24, %23, %21 : memref<1x64x128xf32>, memref<1x64x128xf32>, memref<1x64xf32>) outs(%23 : memref<1x64x128xf32>) {
            ^bb0(%in: f32, %in_7: f32, %in_8: f32, %out: f32):
              %32 = arith.mulf %in_7, %in_8 : f32
              %33 = arith.addf %in, %32 : f32
              linalg.yield %33 : f32
            }
            linalg.copy ins(%19 : memref<1x64xf32>) outs(%18 : memref<1x64xf32>)
          }
          linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%23, %17 : memref<1x64x128xf32>, memref<1x64xf32>) outs(%22 : memref<1x64x128xf32>) {
          ^bb0(%in: f32, %in_5: f32, %out: f32):
            %28 = arith.divf %in, %in_5 : f32
            linalg.yield %28 : f32
          }
          %reinterpret_cast_4 = memref.reinterpret_cast %arg3 to offset: [%27], sizes: [1, 64, 128], strides: [524288, 128, 1] : memref<32x4096x128xf32> to memref<1x64x128xf32, strided<[524288, 128, 1], offset: ?>>
          loom.copy %22, %reinterpret_cast_4 src_mem_space @L1 dst_mem_space @DRAM, interconnect : [], broadcast : [1, 1] : memref<1x64x128xf32>, memref<1x64x128xf32, strided<[524288, 128, 1], offset: ?>>
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
          %15 = loom.alloc [1, 64, 128] on @L1 : memref<1x64x128xf32>
          %16 = loom.alloc [1, 128, 128] on @L1 : memref<1x128x128xf32>
          %17 = loom.alloc [1, 64] on @L1 : memref<1x64xf32>
          %18 = loom.alloc [1, 64] on @L1 : memref<1x64xf32>
          %19 = loom.alloc [1, 64] on @L1 : memref<1x64xf32>
          %20 = loom.alloc [1, 64] on @L1 : memref<1x64xf32>
          %21 = loom.alloc [1, 64] on @L1 : memref<1x64xf32>
          %22 = loom.alloc [1, 64, 128] on @L1 : memref<1x64x128xf32>
          %23 = loom.alloc [1, 64, 128] on @L1 : memref<1x64x128xf32>
          %24 = loom.alloc [1, 64, 128] on @L1 : memref<1x64x128xf32>
          %25 = arith.muli %arg6, %c524288 overflow<nsw> : index
          %26 = arith.muli %13, %c8192 : index
          %27 = arith.addi %25, %26 : index
          %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%27], sizes: [1, 64, 128], strides: [524288, 128, 1] : memref<32x4096x128xf32> to memref<1x64x128xf32, strided<[524288, 128, 1], offset: ?>>
          loom.copy %reinterpret_cast, %22 src_mem_space @DRAM dst_mem_space @L1, interconnect : [], broadcast : [1, 1] : memref<1x64x128xf32, strided<[524288, 128, 1], offset: ?>>, memref<1x64x128xf32>
          linalg.fill ins(%cst_1 : f32) outs(%23 : memref<1x64x128xf32>)
          linalg.fill ins(%cst_2 : f32) outs(%17 : memref<1x64xf32>)
          linalg.fill ins(%cst_3 : f32) outs(%18 : memref<1x64xf32>)
          scf.for %arg7 = %c0 to %c32 step %c1 {
            %28 = arith.muli %arg7, %c128 : index
            %29 = arith.addi %25, %28 : index
            %reinterpret_cast_5 = memref.reinterpret_cast %arg0 to offset: [%29], sizes: [1, 128, 128], strides: [524288, 4096, 1] : memref<32x128x4096xf32> to memref<1x128x128xf32, strided<[524288, 4096, 1], offset: ?>>
            loom.copy %reinterpret_cast_5, %16 src_mem_space @DRAM dst_mem_space @L1, interconnect : [], broadcast : [1, 1] : memref<1x128x128xf32, strided<[524288, 4096, 1], offset: ?>>, memref<1x128x128xf32>
            linalg.fill ins(%cst_1 : f32) outs(%15 : memref<1x64x128xf32>)
            linalg.batch_matmul ins(%22, %16 : memref<1x64x128xf32>, memref<1x128x128xf32>) outs(%15 : memref<1x64x128xf32>)
            linalg.fill ins(%cst_3 : f32) outs(%19 : memref<1x64xf32>)
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%15 : memref<1x64x128xf32>) outs(%19 : memref<1x64xf32>) {
            ^bb0(%in: f32, %out: f32):
              %32 = arith.maximumf %in, %out : f32
              linalg.yield %32 : f32
            }
            linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%18, %19 : memref<1x64xf32>, memref<1x64xf32>) outs(%19 : memref<1x64xf32>) {
            ^bb0(%in: f32, %in_7: f32, %out: f32):
              %32 = arith.truncf %cst_0 : f64 to f32
              %33 = arith.mulf %in_7, %32 : f32
              %34 = arith.cmpf ogt, %in, %33 : f32
              %35 = arith.select %34, %in, %33 : f32
              linalg.yield %35 : f32
            }
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%15, %19 : memref<1x64x128xf32>, memref<1x64xf32>) outs(%15 : memref<1x64x128xf32>) {
            ^bb0(%in: f32, %in_7: f32, %out: f32):
              %32 = arith.truncf %cst_0 : f64 to f32
              %33 = arith.mulf %in, %32 : f32
              %34 = arith.subf %33, %in_7 : f32
              %35 = math.powf %cst, %34 : f32
              linalg.yield %35 : f32
            }
            linalg.fill ins(%cst_1 : f32) outs(%20 : memref<1x64xf32>)
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%15 : memref<1x64x128xf32>) outs(%20 : memref<1x64xf32>) {
            ^bb0(%in: f32, %out: f32):
              %32 = arith.addf %in, %out : f32
              linalg.yield %32 : f32
            }
            linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%18, %19 : memref<1x64xf32>, memref<1x64xf32>) outs(%21 : memref<1x64xf32>) {
            ^bb0(%in: f32, %in_7: f32, %out: f32):
              %32 = arith.subf %in, %in_7 : f32
              %33 = math.powf %cst, %32 : f32
              linalg.yield %33 : f32
            }
            linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%17, %21, %20 : memref<1x64xf32>, memref<1x64xf32>, memref<1x64xf32>) outs(%17 : memref<1x64xf32>) {
            ^bb0(%in: f32, %in_7: f32, %in_8: f32, %out: f32):
              %32 = arith.mulf %in, %in_7 : f32
              %33 = arith.addf %32, %in_8 : f32
              linalg.yield %33 : f32
            }
            %30 = arith.muli %arg7, %c16384 : index
            %31 = arith.addi %25, %30 : index
            %reinterpret_cast_6 = memref.reinterpret_cast %arg1 to offset: [%31], sizes: [1, 128, 128], strides: [524288, 128, 1] : memref<32x4096x128xf32> to memref<1x128x128xf32, strided<[524288, 128, 1], offset: ?>>
            loom.copy %reinterpret_cast_6, %14 src_mem_space @DRAM dst_mem_space @L1, interconnect : [@horizontal_links, @vertical_links], broadcast : [8, 8] : memref<1x128x128xf32, strided<[524288, 128, 1], offset: ?>>, memref<1x128x128xf32>
            linalg.fill ins(%cst_1 : f32) outs(%24 : memref<1x64x128xf32>)
            linalg.batch_matmul ins(%15, %14 : memref<1x64x128xf32>, memref<1x128x128xf32>) outs(%24 : memref<1x64x128xf32>)
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%24, %23, %21 : memref<1x64x128xf32>, memref<1x64x128xf32>, memref<1x64xf32>) outs(%23 : memref<1x64x128xf32>) {
            ^bb0(%in: f32, %in_7: f32, %in_8: f32, %out: f32):
              %32 = arith.mulf %in_7, %in_8 : f32
              %33 = arith.addf %in, %32 : f32
              linalg.yield %33 : f32
            }
            linalg.copy ins(%19 : memref<1x64xf32>) outs(%18 : memref<1x64xf32>)
          }
          linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%23, %17 : memref<1x64x128xf32>, memref<1x64xf32>) outs(%22 : memref<1x64x128xf32>) {
          ^bb0(%in: f32, %in_5: f32, %out: f32):
            %28 = arith.divf %in, %in_5 : f32
            linalg.yield %28 : f32
          }
          %reinterpret_cast_4 = memref.reinterpret_cast %arg3 to offset: [%27], sizes: [1, 64, 128], strides: [524288, 128, 1] : memref<32x4096x128xf32> to memref<1x64x128xf32, strided<[524288, 128, 1], offset: ?>>
          loom.copy %22, %reinterpret_cast_4 src_mem_space @L1 dst_mem_space @DRAM, interconnect : [], broadcast : [1, 1] : memref<1x64x128xf32>, memref<1x64x128xf32, strided<[524288, 128, 1], offset: ?>>
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
          %15 = loom.alloc [1, 64, 128] on @L1 : memref<1x64x128xf32>
          %16 = loom.alloc [1, 128, 128] on @L1 : memref<1x128x128xf32>
          %17 = loom.alloc [1, 64] on @L1 : memref<1x64xf32>
          %18 = loom.alloc [1, 64] on @L1 : memref<1x64xf32>
          %19 = loom.alloc [1, 64] on @L1 : memref<1x64xf32>
          %20 = loom.alloc [1, 64] on @L1 : memref<1x64xf32>
          %21 = loom.alloc [1, 64] on @L1 : memref<1x64xf32>
          %22 = loom.alloc [1, 64, 128] on @L1 : memref<1x64x128xf32>
          %23 = loom.alloc [1, 64, 128] on @L1 : memref<1x64x128xf32>
          %24 = loom.alloc [1, 64, 128] on @L1 : memref<1x64x128xf32>
          %25 = arith.muli %arg6, %c524288 overflow<nsw> : index
          %26 = arith.muli %13, %c8192 : index
          %27 = arith.addi %25, %26 : index
          %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%27], sizes: [1, 64, 128], strides: [524288, 128, 1] : memref<32x4096x128xf32> to memref<1x64x128xf32, strided<[524288, 128, 1], offset: ?>>
          loom.copy %reinterpret_cast, %22 src_mem_space @DRAM dst_mem_space @L1, interconnect : [], broadcast : [1, 1] : memref<1x64x128xf32, strided<[524288, 128, 1], offset: ?>>, memref<1x64x128xf32>
          linalg.fill ins(%cst_1 : f32) outs(%23 : memref<1x64x128xf32>)
          linalg.fill ins(%cst_2 : f32) outs(%17 : memref<1x64xf32>)
          linalg.fill ins(%cst_3 : f32) outs(%18 : memref<1x64xf32>)
          scf.for %arg7 = %c0 to %c32 step %c1 {
            %28 = arith.muli %arg7, %c128 : index
            %29 = arith.addi %25, %28 : index
            %reinterpret_cast_5 = memref.reinterpret_cast %arg0 to offset: [%29], sizes: [1, 128, 128], strides: [524288, 4096, 1] : memref<32x128x4096xf32> to memref<1x128x128xf32, strided<[524288, 4096, 1], offset: ?>>
            loom.copy %reinterpret_cast_5, %16 src_mem_space @DRAM dst_mem_space @L1, interconnect : [], broadcast : [1, 1] : memref<1x128x128xf32, strided<[524288, 4096, 1], offset: ?>>, memref<1x128x128xf32>
            linalg.fill ins(%cst_1 : f32) outs(%15 : memref<1x64x128xf32>)
            linalg.batch_matmul ins(%22, %16 : memref<1x64x128xf32>, memref<1x128x128xf32>) outs(%15 : memref<1x64x128xf32>)
            linalg.fill ins(%cst_3 : f32) outs(%19 : memref<1x64xf32>)
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%15 : memref<1x64x128xf32>) outs(%19 : memref<1x64xf32>) {
            ^bb0(%in: f32, %out: f32):
              %32 = arith.maximumf %in, %out : f32
              linalg.yield %32 : f32
            }
            linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%18, %19 : memref<1x64xf32>, memref<1x64xf32>) outs(%19 : memref<1x64xf32>) {
            ^bb0(%in: f32, %in_7: f32, %out: f32):
              %32 = arith.truncf %cst_0 : f64 to f32
              %33 = arith.mulf %in_7, %32 : f32
              %34 = arith.cmpf ogt, %in, %33 : f32
              %35 = arith.select %34, %in, %33 : f32
              linalg.yield %35 : f32
            }
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%15, %19 : memref<1x64x128xf32>, memref<1x64xf32>) outs(%15 : memref<1x64x128xf32>) {
            ^bb0(%in: f32, %in_7: f32, %out: f32):
              %32 = arith.truncf %cst_0 : f64 to f32
              %33 = arith.mulf %in, %32 : f32
              %34 = arith.subf %33, %in_7 : f32
              %35 = math.powf %cst, %34 : f32
              linalg.yield %35 : f32
            }
            linalg.fill ins(%cst_1 : f32) outs(%20 : memref<1x64xf32>)
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%15 : memref<1x64x128xf32>) outs(%20 : memref<1x64xf32>) {
            ^bb0(%in: f32, %out: f32):
              %32 = arith.addf %in, %out : f32
              linalg.yield %32 : f32
            }
            linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%18, %19 : memref<1x64xf32>, memref<1x64xf32>) outs(%21 : memref<1x64xf32>) {
            ^bb0(%in: f32, %in_7: f32, %out: f32):
              %32 = arith.subf %in, %in_7 : f32
              %33 = math.powf %cst, %32 : f32
              linalg.yield %33 : f32
            }
            linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%17, %21, %20 : memref<1x64xf32>, memref<1x64xf32>, memref<1x64xf32>) outs(%17 : memref<1x64xf32>) {
            ^bb0(%in: f32, %in_7: f32, %in_8: f32, %out: f32):
              %32 = arith.mulf %in, %in_7 : f32
              %33 = arith.addf %32, %in_8 : f32
              linalg.yield %33 : f32
            }
            %30 = arith.muli %arg7, %c16384 : index
            %31 = arith.addi %25, %30 : index
            %reinterpret_cast_6 = memref.reinterpret_cast %arg1 to offset: [%31], sizes: [1, 128, 128], strides: [524288, 128, 1] : memref<32x4096x128xf32> to memref<1x128x128xf32, strided<[524288, 128, 1], offset: ?>>
            loom.copy %reinterpret_cast_6, %14 src_mem_space @DRAM dst_mem_space @L1, interconnect : [@horizontal_links], broadcast : [1, 8] : memref<1x128x128xf32, strided<[524288, 128, 1], offset: ?>>, memref<1x128x128xf32>
            linalg.fill ins(%cst_1 : f32) outs(%24 : memref<1x64x128xf32>)
            linalg.batch_matmul ins(%15, %14 : memref<1x64x128xf32>, memref<1x128x128xf32>) outs(%24 : memref<1x64x128xf32>)
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%24, %23, %21 : memref<1x64x128xf32>, memref<1x64x128xf32>, memref<1x64xf32>) outs(%23 : memref<1x64x128xf32>) {
            ^bb0(%in: f32, %in_7: f32, %in_8: f32, %out: f32):
              %32 = arith.mulf %in_7, %in_8 : f32
              %33 = arith.addf %in, %32 : f32
              linalg.yield %33 : f32
            }
            linalg.copy ins(%19 : memref<1x64xf32>) outs(%18 : memref<1x64xf32>)
          }
          linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%23, %17 : memref<1x64x128xf32>, memref<1x64xf32>) outs(%22 : memref<1x64x128xf32>) {
          ^bb0(%in: f32, %in_5: f32, %out: f32):
            %28 = arith.divf %in, %in_5 : f32
            linalg.yield %28 : f32
          }
          %reinterpret_cast_4 = memref.reinterpret_cast %arg3 to offset: [%27], sizes: [1, 64, 128], strides: [524288, 128, 1] : memref<32x4096x128xf32> to memref<1x64x128xf32, strided<[524288, 128, 1], offset: ?>>
          loom.copy %22, %reinterpret_cast_4 src_mem_space @L1 dst_mem_space @DRAM, interconnect : [], broadcast : [1, 1] : memref<1x64x128xf32>, memref<1x64x128xf32, strided<[524288, 128, 1], offset: ?>>
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
          %15 = loom.alloc [1, 64, 128] on @L1 : memref<1x64x128xf32>
          %16 = loom.alloc [1, 128, 128] on @L1 : memref<1x128x128xf32>
          %17 = loom.alloc [1, 64] on @L1 : memref<1x64xf32>
          %18 = loom.alloc [1, 64] on @L1 : memref<1x64xf32>
          %19 = loom.alloc [1, 64] on @L1 : memref<1x64xf32>
          %20 = loom.alloc [1, 64] on @L1 : memref<1x64xf32>
          %21 = loom.alloc [1, 64] on @L1 : memref<1x64xf32>
          %22 = loom.alloc [1, 64, 128] on @L1 : memref<1x64x128xf32>
          %23 = loom.alloc [1, 64, 128] on @L1 : memref<1x64x128xf32>
          %24 = loom.alloc [1, 64, 128] on @L1 : memref<1x64x128xf32>
          %25 = arith.muli %arg6, %c524288 overflow<nsw> : index
          %26 = arith.muli %13, %c8192 : index
          %27 = arith.addi %25, %26 : index
          %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%27], sizes: [1, 64, 128], strides: [524288, 128, 1] : memref<32x4096x128xf32> to memref<1x64x128xf32, strided<[524288, 128, 1], offset: ?>>
          loom.copy %reinterpret_cast, %22 src_mem_space @DRAM dst_mem_space @L1, interconnect : [], broadcast : [1, 1] : memref<1x64x128xf32, strided<[524288, 128, 1], offset: ?>>, memref<1x64x128xf32>
          linalg.fill ins(%cst_1 : f32) outs(%23 : memref<1x64x128xf32>)
          linalg.fill ins(%cst_2 : f32) outs(%17 : memref<1x64xf32>)
          linalg.fill ins(%cst_3 : f32) outs(%18 : memref<1x64xf32>)
          scf.for %arg7 = %c0 to %c32 step %c1 {
            %28 = arith.muli %arg7, %c128 : index
            %29 = arith.addi %25, %28 : index
            %reinterpret_cast_5 = memref.reinterpret_cast %arg0 to offset: [%29], sizes: [1, 128, 128], strides: [524288, 4096, 1] : memref<32x128x4096xf32> to memref<1x128x128xf32, strided<[524288, 4096, 1], offset: ?>>
            loom.copy %reinterpret_cast_5, %16 src_mem_space @DRAM dst_mem_space @L1, interconnect : [], broadcast : [1, 1] : memref<1x128x128xf32, strided<[524288, 4096, 1], offset: ?>>, memref<1x128x128xf32>
            linalg.fill ins(%cst_1 : f32) outs(%15 : memref<1x64x128xf32>)
            linalg.batch_matmul ins(%22, %16 : memref<1x64x128xf32>, memref<1x128x128xf32>) outs(%15 : memref<1x64x128xf32>)
            linalg.fill ins(%cst_3 : f32) outs(%19 : memref<1x64xf32>)
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%15 : memref<1x64x128xf32>) outs(%19 : memref<1x64xf32>) {
            ^bb0(%in: f32, %out: f32):
              %32 = arith.maximumf %in, %out : f32
              linalg.yield %32 : f32
            }
            linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%18, %19 : memref<1x64xf32>, memref<1x64xf32>) outs(%19 : memref<1x64xf32>) {
            ^bb0(%in: f32, %in_7: f32, %out: f32):
              %32 = arith.truncf %cst_0 : f64 to f32
              %33 = arith.mulf %in_7, %32 : f32
              %34 = arith.cmpf ogt, %in, %33 : f32
              %35 = arith.select %34, %in, %33 : f32
              linalg.yield %35 : f32
            }
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%15, %19 : memref<1x64x128xf32>, memref<1x64xf32>) outs(%15 : memref<1x64x128xf32>) {
            ^bb0(%in: f32, %in_7: f32, %out: f32):
              %32 = arith.truncf %cst_0 : f64 to f32
              %33 = arith.mulf %in, %32 : f32
              %34 = arith.subf %33, %in_7 : f32
              %35 = math.powf %cst, %34 : f32
              linalg.yield %35 : f32
            }
            linalg.fill ins(%cst_1 : f32) outs(%20 : memref<1x64xf32>)
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%15 : memref<1x64x128xf32>) outs(%20 : memref<1x64xf32>) {
            ^bb0(%in: f32, %out: f32):
              %32 = arith.addf %in, %out : f32
              linalg.yield %32 : f32
            }
            linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%18, %19 : memref<1x64xf32>, memref<1x64xf32>) outs(%21 : memref<1x64xf32>) {
            ^bb0(%in: f32, %in_7: f32, %out: f32):
              %32 = arith.subf %in, %in_7 : f32
              %33 = math.powf %cst, %32 : f32
              linalg.yield %33 : f32
            }
            linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%17, %21, %20 : memref<1x64xf32>, memref<1x64xf32>, memref<1x64xf32>) outs(%17 : memref<1x64xf32>) {
            ^bb0(%in: f32, %in_7: f32, %in_8: f32, %out: f32):
              %32 = arith.mulf %in, %in_7 : f32
              %33 = arith.addf %32, %in_8 : f32
              linalg.yield %33 : f32
            }
            %30 = arith.muli %arg7, %c16384 : index
            %31 = arith.addi %25, %30 : index
            %reinterpret_cast_6 = memref.reinterpret_cast %arg1 to offset: [%31], sizes: [1, 128, 128], strides: [524288, 128, 1] : memref<32x4096x128xf32> to memref<1x128x128xf32, strided<[524288, 128, 1], offset: ?>>
            loom.copy %reinterpret_cast_6, %14 src_mem_space @DRAM dst_mem_space @L1, interconnect : [@vertical_links], broadcast : [8, 1] : memref<1x128x128xf32, strided<[524288, 128, 1], offset: ?>>, memref<1x128x128xf32>
            linalg.fill ins(%cst_1 : f32) outs(%24 : memref<1x64x128xf32>)
            linalg.batch_matmul ins(%15, %14 : memref<1x64x128xf32>, memref<1x128x128xf32>) outs(%24 : memref<1x64x128xf32>)
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%24, %23, %21 : memref<1x64x128xf32>, memref<1x64x128xf32>, memref<1x64xf32>) outs(%23 : memref<1x64x128xf32>) {
            ^bb0(%in: f32, %in_7: f32, %in_8: f32, %out: f32):
              %32 = arith.mulf %in_7, %in_8 : f32
              %33 = arith.addf %in, %32 : f32
              linalg.yield %33 : f32
            }
            linalg.copy ins(%19 : memref<1x64xf32>) outs(%18 : memref<1x64xf32>)
          }
          linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%23, %17 : memref<1x64x128xf32>, memref<1x64xf32>) outs(%22 : memref<1x64x128xf32>) {
          ^bb0(%in: f32, %in_5: f32, %out: f32):
            %28 = arith.divf %in, %in_5 : f32
            linalg.yield %28 : f32
          }
          %reinterpret_cast_4 = memref.reinterpret_cast %arg3 to offset: [%27], sizes: [1, 64, 128], strides: [524288, 128, 1] : memref<32x4096x128xf32> to memref<1x64x128xf32, strided<[524288, 128, 1], offset: ?>>
          loom.copy %22, %reinterpret_cast_4 src_mem_space @L1 dst_mem_space @DRAM, interconnect : [], broadcast : [1, 1] : memref<1x64x128xf32>, memref<1x64x128xf32, strided<[524288, 128, 1], offset: ?>>
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
          %15 = loom.alloc [1, 64, 128] on @L1 : memref<1x64x128xf32>
          %16 = loom.alloc [1, 128, 128] on @L1 : memref<1x128x128xf32>
          %17 = loom.alloc [1, 64] on @L1 : memref<1x64xf32>
          %18 = loom.alloc [1, 64] on @L1 : memref<1x64xf32>
          %19 = loom.alloc [1, 64] on @L1 : memref<1x64xf32>
          %20 = loom.alloc [1, 64] on @L1 : memref<1x64xf32>
          %21 = loom.alloc [1, 64] on @L1 : memref<1x64xf32>
          %22 = loom.alloc [1, 64, 128] on @L1 : memref<1x64x128xf32>
          %23 = loom.alloc [1, 64, 128] on @L1 : memref<1x64x128xf32>
          %24 = loom.alloc [1, 64, 128] on @L1 : memref<1x64x128xf32>
          %25 = arith.muli %arg6, %c524288 overflow<nsw> : index
          %26 = arith.muli %13, %c8192 : index
          %27 = arith.addi %25, %26 : index
          %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%27], sizes: [1, 64, 128], strides: [524288, 128, 1] : memref<32x4096x128xf32> to memref<1x64x128xf32, strided<[524288, 128, 1], offset: ?>>
          loom.copy %reinterpret_cast, %22 src_mem_space @DRAM dst_mem_space @L1, interconnect : [], broadcast : [1, 1] : memref<1x64x128xf32, strided<[524288, 128, 1], offset: ?>>, memref<1x64x128xf32>
          linalg.fill ins(%cst_1 : f32) outs(%23 : memref<1x64x128xf32>)
          linalg.fill ins(%cst_2 : f32) outs(%17 : memref<1x64xf32>)
          linalg.fill ins(%cst_3 : f32) outs(%18 : memref<1x64xf32>)
          scf.for %arg7 = %c0 to %c32 step %c1 {
            %28 = arith.muli %arg7, %c128 : index
            %29 = arith.addi %25, %28 : index
            %reinterpret_cast_5 = memref.reinterpret_cast %arg0 to offset: [%29], sizes: [1, 128, 128], strides: [524288, 4096, 1] : memref<32x128x4096xf32> to memref<1x128x128xf32, strided<[524288, 4096, 1], offset: ?>>
            loom.copy %reinterpret_cast_5, %16 src_mem_space @DRAM dst_mem_space @L1, interconnect : [@horizontal_links, @vertical_links], broadcast : [8, 8] : memref<1x128x128xf32, strided<[524288, 4096, 1], offset: ?>>, memref<1x128x128xf32>
            linalg.fill ins(%cst_1 : f32) outs(%15 : memref<1x64x128xf32>)
            linalg.batch_matmul ins(%22, %16 : memref<1x64x128xf32>, memref<1x128x128xf32>) outs(%15 : memref<1x64x128xf32>)
            linalg.fill ins(%cst_3 : f32) outs(%19 : memref<1x64xf32>)
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%15 : memref<1x64x128xf32>) outs(%19 : memref<1x64xf32>) {
            ^bb0(%in: f32, %out: f32):
              %32 = arith.maximumf %in, %out : f32
              linalg.yield %32 : f32
            }
            linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%18, %19 : memref<1x64xf32>, memref<1x64xf32>) outs(%19 : memref<1x64xf32>) {
            ^bb0(%in: f32, %in_7: f32, %out: f32):
              %32 = arith.truncf %cst_0 : f64 to f32
              %33 = arith.mulf %in_7, %32 : f32
              %34 = arith.cmpf ogt, %in, %33 : f32
              %35 = arith.select %34, %in, %33 : f32
              linalg.yield %35 : f32
            }
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%15, %19 : memref<1x64x128xf32>, memref<1x64xf32>) outs(%15 : memref<1x64x128xf32>) {
            ^bb0(%in: f32, %in_7: f32, %out: f32):
              %32 = arith.truncf %cst_0 : f64 to f32
              %33 = arith.mulf %in, %32 : f32
              %34 = arith.subf %33, %in_7 : f32
              %35 = math.powf %cst, %34 : f32
              linalg.yield %35 : f32
            }
            linalg.fill ins(%cst_1 : f32) outs(%20 : memref<1x64xf32>)
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%15 : memref<1x64x128xf32>) outs(%20 : memref<1x64xf32>) {
            ^bb0(%in: f32, %out: f32):
              %32 = arith.addf %in, %out : f32
              linalg.yield %32 : f32
            }
            linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%18, %19 : memref<1x64xf32>, memref<1x64xf32>) outs(%21 : memref<1x64xf32>) {
            ^bb0(%in: f32, %in_7: f32, %out: f32):
              %32 = arith.subf %in, %in_7 : f32
              %33 = math.powf %cst, %32 : f32
              linalg.yield %33 : f32
            }
            linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%17, %21, %20 : memref<1x64xf32>, memref<1x64xf32>, memref<1x64xf32>) outs(%17 : memref<1x64xf32>) {
            ^bb0(%in: f32, %in_7: f32, %in_8: f32, %out: f32):
              %32 = arith.mulf %in, %in_7 : f32
              %33 = arith.addf %32, %in_8 : f32
              linalg.yield %33 : f32
            }
            %30 = arith.muli %arg7, %c16384 : index
            %31 = arith.addi %25, %30 : index
            %reinterpret_cast_6 = memref.reinterpret_cast %arg1 to offset: [%31], sizes: [1, 128, 128], strides: [524288, 128, 1] : memref<32x4096x128xf32> to memref<1x128x128xf32, strided<[524288, 128, 1], offset: ?>>
            loom.copy %reinterpret_cast_6, %14 src_mem_space @DRAM dst_mem_space @L1, interconnect : [], broadcast : [1, 1] : memref<1x128x128xf32, strided<[524288, 128, 1], offset: ?>>, memref<1x128x128xf32>
            linalg.fill ins(%cst_1 : f32) outs(%24 : memref<1x64x128xf32>)
            linalg.batch_matmul ins(%15, %14 : memref<1x64x128xf32>, memref<1x128x128xf32>) outs(%24 : memref<1x64x128xf32>)
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%24, %23, %21 : memref<1x64x128xf32>, memref<1x64x128xf32>, memref<1x64xf32>) outs(%23 : memref<1x64x128xf32>) {
            ^bb0(%in: f32, %in_7: f32, %in_8: f32, %out: f32):
              %32 = arith.mulf %in_7, %in_8 : f32
              %33 = arith.addf %in, %32 : f32
              linalg.yield %33 : f32
            }
            linalg.copy ins(%19 : memref<1x64xf32>) outs(%18 : memref<1x64xf32>)
          }
          linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%23, %17 : memref<1x64x128xf32>, memref<1x64xf32>) outs(%22 : memref<1x64x128xf32>) {
          ^bb0(%in: f32, %in_5: f32, %out: f32):
            %28 = arith.divf %in, %in_5 : f32
            linalg.yield %28 : f32
          }
          %reinterpret_cast_4 = memref.reinterpret_cast %arg3 to offset: [%27], sizes: [1, 64, 128], strides: [524288, 128, 1] : memref<32x4096x128xf32> to memref<1x64x128xf32, strided<[524288, 128, 1], offset: ?>>
          loom.copy %22, %reinterpret_cast_4 src_mem_space @L1 dst_mem_space @DRAM, interconnect : [], broadcast : [1, 1] : memref<1x64x128xf32>, memref<1x64x128xf32, strided<[524288, 128, 1], offset: ?>>
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
          %15 = loom.alloc [1, 64, 128] on @L1 : memref<1x64x128xf32>
          %16 = loom.alloc [1, 128, 128] on @L1 : memref<1x128x128xf32>
          %17 = loom.alloc [1, 64] on @L1 : memref<1x64xf32>
          %18 = loom.alloc [1, 64] on @L1 : memref<1x64xf32>
          %19 = loom.alloc [1, 64] on @L1 : memref<1x64xf32>
          %20 = loom.alloc [1, 64] on @L1 : memref<1x64xf32>
          %21 = loom.alloc [1, 64] on @L1 : memref<1x64xf32>
          %22 = loom.alloc [1, 64, 128] on @L1 : memref<1x64x128xf32>
          %23 = loom.alloc [1, 64, 128] on @L1 : memref<1x64x128xf32>
          %24 = loom.alloc [1, 64, 128] on @L1 : memref<1x64x128xf32>
          %25 = arith.muli %arg6, %c524288 overflow<nsw> : index
          %26 = arith.muli %13, %c8192 : index
          %27 = arith.addi %25, %26 : index
          %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%27], sizes: [1, 64, 128], strides: [524288, 128, 1] : memref<32x4096x128xf32> to memref<1x64x128xf32, strided<[524288, 128, 1], offset: ?>>
          loom.copy %reinterpret_cast, %22 src_mem_space @DRAM dst_mem_space @L1, interconnect : [], broadcast : [1, 1] : memref<1x64x128xf32, strided<[524288, 128, 1], offset: ?>>, memref<1x64x128xf32>
          linalg.fill ins(%cst_1 : f32) outs(%23 : memref<1x64x128xf32>)
          linalg.fill ins(%cst_2 : f32) outs(%17 : memref<1x64xf32>)
          linalg.fill ins(%cst_3 : f32) outs(%18 : memref<1x64xf32>)
          scf.for %arg7 = %c0 to %c32 step %c1 {
            %28 = arith.muli %arg7, %c128 : index
            %29 = arith.addi %25, %28 : index
            %reinterpret_cast_5 = memref.reinterpret_cast %arg0 to offset: [%29], sizes: [1, 128, 128], strides: [524288, 4096, 1] : memref<32x128x4096xf32> to memref<1x128x128xf32, strided<[524288, 4096, 1], offset: ?>>
            loom.copy %reinterpret_cast_5, %16 src_mem_space @DRAM dst_mem_space @L1, interconnect : [@horizontal_links, @vertical_links], broadcast : [8, 8] : memref<1x128x128xf32, strided<[524288, 4096, 1], offset: ?>>, memref<1x128x128xf32>
            linalg.fill ins(%cst_1 : f32) outs(%15 : memref<1x64x128xf32>)
            linalg.batch_matmul ins(%22, %16 : memref<1x64x128xf32>, memref<1x128x128xf32>) outs(%15 : memref<1x64x128xf32>)
            linalg.fill ins(%cst_3 : f32) outs(%19 : memref<1x64xf32>)
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%15 : memref<1x64x128xf32>) outs(%19 : memref<1x64xf32>) {
            ^bb0(%in: f32, %out: f32):
              %32 = arith.maximumf %in, %out : f32
              linalg.yield %32 : f32
            }
            linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%18, %19 : memref<1x64xf32>, memref<1x64xf32>) outs(%19 : memref<1x64xf32>) {
            ^bb0(%in: f32, %in_7: f32, %out: f32):
              %32 = arith.truncf %cst_0 : f64 to f32
              %33 = arith.mulf %in_7, %32 : f32
              %34 = arith.cmpf ogt, %in, %33 : f32
              %35 = arith.select %34, %in, %33 : f32
              linalg.yield %35 : f32
            }
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%15, %19 : memref<1x64x128xf32>, memref<1x64xf32>) outs(%15 : memref<1x64x128xf32>) {
            ^bb0(%in: f32, %in_7: f32, %out: f32):
              %32 = arith.truncf %cst_0 : f64 to f32
              %33 = arith.mulf %in, %32 : f32
              %34 = arith.subf %33, %in_7 : f32
              %35 = math.powf %cst, %34 : f32
              linalg.yield %35 : f32
            }
            linalg.fill ins(%cst_1 : f32) outs(%20 : memref<1x64xf32>)
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%15 : memref<1x64x128xf32>) outs(%20 : memref<1x64xf32>) {
            ^bb0(%in: f32, %out: f32):
              %32 = arith.addf %in, %out : f32
              linalg.yield %32 : f32
            }
            linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%18, %19 : memref<1x64xf32>, memref<1x64xf32>) outs(%21 : memref<1x64xf32>) {
            ^bb0(%in: f32, %in_7: f32, %out: f32):
              %32 = arith.subf %in, %in_7 : f32
              %33 = math.powf %cst, %32 : f32
              linalg.yield %33 : f32
            }
            linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%17, %21, %20 : memref<1x64xf32>, memref<1x64xf32>, memref<1x64xf32>) outs(%17 : memref<1x64xf32>) {
            ^bb0(%in: f32, %in_7: f32, %in_8: f32, %out: f32):
              %32 = arith.mulf %in, %in_7 : f32
              %33 = arith.addf %32, %in_8 : f32
              linalg.yield %33 : f32
            }
            %30 = arith.muli %arg7, %c16384 : index
            %31 = arith.addi %25, %30 : index
            %reinterpret_cast_6 = memref.reinterpret_cast %arg1 to offset: [%31], sizes: [1, 128, 128], strides: [524288, 128, 1] : memref<32x4096x128xf32> to memref<1x128x128xf32, strided<[524288, 128, 1], offset: ?>>
            loom.copy %reinterpret_cast_6, %14 src_mem_space @DRAM dst_mem_space @L1, interconnect : [@horizontal_links, @vertical_links], broadcast : [8, 8] : memref<1x128x128xf32, strided<[524288, 128, 1], offset: ?>>, memref<1x128x128xf32>
            linalg.fill ins(%cst_1 : f32) outs(%24 : memref<1x64x128xf32>)
            linalg.batch_matmul ins(%15, %14 : memref<1x64x128xf32>, memref<1x128x128xf32>) outs(%24 : memref<1x64x128xf32>)
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%24, %23, %21 : memref<1x64x128xf32>, memref<1x64x128xf32>, memref<1x64xf32>) outs(%23 : memref<1x64x128xf32>) {
            ^bb0(%in: f32, %in_7: f32, %in_8: f32, %out: f32):
              %32 = arith.mulf %in_7, %in_8 : f32
              %33 = arith.addf %in, %32 : f32
              linalg.yield %33 : f32
            }
            linalg.copy ins(%19 : memref<1x64xf32>) outs(%18 : memref<1x64xf32>)
          }
          linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%23, %17 : memref<1x64x128xf32>, memref<1x64xf32>) outs(%22 : memref<1x64x128xf32>) {
          ^bb0(%in: f32, %in_5: f32, %out: f32):
            %28 = arith.divf %in, %in_5 : f32
            linalg.yield %28 : f32
          }
          %reinterpret_cast_4 = memref.reinterpret_cast %arg3 to offset: [%27], sizes: [1, 64, 128], strides: [524288, 128, 1] : memref<32x4096x128xf32> to memref<1x64x128xf32, strided<[524288, 128, 1], offset: ?>>
          loom.copy %22, %reinterpret_cast_4 src_mem_space @L1 dst_mem_space @DRAM, interconnect : [], broadcast : [1, 1] : memref<1x64x128xf32>, memref<1x64x128xf32, strided<[524288, 128, 1], offset: ?>>
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
          %15 = loom.alloc [1, 64, 128] on @L1 : memref<1x64x128xf32>
          %16 = loom.alloc [1, 128, 128] on @L1 : memref<1x128x128xf32>
          %17 = loom.alloc [1, 64] on @L1 : memref<1x64xf32>
          %18 = loom.alloc [1, 64] on @L1 : memref<1x64xf32>
          %19 = loom.alloc [1, 64] on @L1 : memref<1x64xf32>
          %20 = loom.alloc [1, 64] on @L1 : memref<1x64xf32>
          %21 = loom.alloc [1, 64] on @L1 : memref<1x64xf32>
          %22 = loom.alloc [1, 64, 128] on @L1 : memref<1x64x128xf32>
          %23 = loom.alloc [1, 64, 128] on @L1 : memref<1x64x128xf32>
          %24 = loom.alloc [1, 64, 128] on @L1 : memref<1x64x128xf32>
          %25 = arith.muli %arg6, %c524288 overflow<nsw> : index
          %26 = arith.muli %13, %c8192 : index
          %27 = arith.addi %25, %26 : index
          %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%27], sizes: [1, 64, 128], strides: [524288, 128, 1] : memref<32x4096x128xf32> to memref<1x64x128xf32, strided<[524288, 128, 1], offset: ?>>
          loom.copy %reinterpret_cast, %22 src_mem_space @DRAM dst_mem_space @L1, interconnect : [], broadcast : [1, 1] : memref<1x64x128xf32, strided<[524288, 128, 1], offset: ?>>, memref<1x64x128xf32>
          linalg.fill ins(%cst_1 : f32) outs(%23 : memref<1x64x128xf32>)
          linalg.fill ins(%cst_2 : f32) outs(%17 : memref<1x64xf32>)
          linalg.fill ins(%cst_3 : f32) outs(%18 : memref<1x64xf32>)
          scf.for %arg7 = %c0 to %c32 step %c1 {
            %28 = arith.muli %arg7, %c128 : index
            %29 = arith.addi %25, %28 : index
            %reinterpret_cast_5 = memref.reinterpret_cast %arg0 to offset: [%29], sizes: [1, 128, 128], strides: [524288, 4096, 1] : memref<32x128x4096xf32> to memref<1x128x128xf32, strided<[524288, 4096, 1], offset: ?>>
            loom.copy %reinterpret_cast_5, %16 src_mem_space @DRAM dst_mem_space @L1, interconnect : [@horizontal_links, @vertical_links], broadcast : [8, 8] : memref<1x128x128xf32, strided<[524288, 4096, 1], offset: ?>>, memref<1x128x128xf32>
            linalg.fill ins(%cst_1 : f32) outs(%15 : memref<1x64x128xf32>)
            linalg.batch_matmul ins(%22, %16 : memref<1x64x128xf32>, memref<1x128x128xf32>) outs(%15 : memref<1x64x128xf32>)
            linalg.fill ins(%cst_3 : f32) outs(%19 : memref<1x64xf32>)
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%15 : memref<1x64x128xf32>) outs(%19 : memref<1x64xf32>) {
            ^bb0(%in: f32, %out: f32):
              %32 = arith.maximumf %in, %out : f32
              linalg.yield %32 : f32
            }
            linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%18, %19 : memref<1x64xf32>, memref<1x64xf32>) outs(%19 : memref<1x64xf32>) {
            ^bb0(%in: f32, %in_7: f32, %out: f32):
              %32 = arith.truncf %cst_0 : f64 to f32
              %33 = arith.mulf %in_7, %32 : f32
              %34 = arith.cmpf ogt, %in, %33 : f32
              %35 = arith.select %34, %in, %33 : f32
              linalg.yield %35 : f32
            }
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%15, %19 : memref<1x64x128xf32>, memref<1x64xf32>) outs(%15 : memref<1x64x128xf32>) {
            ^bb0(%in: f32, %in_7: f32, %out: f32):
              %32 = arith.truncf %cst_0 : f64 to f32
              %33 = arith.mulf %in, %32 : f32
              %34 = arith.subf %33, %in_7 : f32
              %35 = math.powf %cst, %34 : f32
              linalg.yield %35 : f32
            }
            linalg.fill ins(%cst_1 : f32) outs(%20 : memref<1x64xf32>)
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%15 : memref<1x64x128xf32>) outs(%20 : memref<1x64xf32>) {
            ^bb0(%in: f32, %out: f32):
              %32 = arith.addf %in, %out : f32
              linalg.yield %32 : f32
            }
            linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%18, %19 : memref<1x64xf32>, memref<1x64xf32>) outs(%21 : memref<1x64xf32>) {
            ^bb0(%in: f32, %in_7: f32, %out: f32):
              %32 = arith.subf %in, %in_7 : f32
              %33 = math.powf %cst, %32 : f32
              linalg.yield %33 : f32
            }
            linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%17, %21, %20 : memref<1x64xf32>, memref<1x64xf32>, memref<1x64xf32>) outs(%17 : memref<1x64xf32>) {
            ^bb0(%in: f32, %in_7: f32, %in_8: f32, %out: f32):
              %32 = arith.mulf %in, %in_7 : f32
              %33 = arith.addf %32, %in_8 : f32
              linalg.yield %33 : f32
            }
            %30 = arith.muli %arg7, %c16384 : index
            %31 = arith.addi %25, %30 : index
            %reinterpret_cast_6 = memref.reinterpret_cast %arg1 to offset: [%31], sizes: [1, 128, 128], strides: [524288, 128, 1] : memref<32x4096x128xf32> to memref<1x128x128xf32, strided<[524288, 128, 1], offset: ?>>
            loom.copy %reinterpret_cast_6, %14 src_mem_space @DRAM dst_mem_space @L1, interconnect : [@horizontal_links], broadcast : [1, 8] : memref<1x128x128xf32, strided<[524288, 128, 1], offset: ?>>, memref<1x128x128xf32>
            linalg.fill ins(%cst_1 : f32) outs(%24 : memref<1x64x128xf32>)
            linalg.batch_matmul ins(%15, %14 : memref<1x64x128xf32>, memref<1x128x128xf32>) outs(%24 : memref<1x64x128xf32>)
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%24, %23, %21 : memref<1x64x128xf32>, memref<1x64x128xf32>, memref<1x64xf32>) outs(%23 : memref<1x64x128xf32>) {
            ^bb0(%in: f32, %in_7: f32, %in_8: f32, %out: f32):
              %32 = arith.mulf %in_7, %in_8 : f32
              %33 = arith.addf %in, %32 : f32
              linalg.yield %33 : f32
            }
            linalg.copy ins(%19 : memref<1x64xf32>) outs(%18 : memref<1x64xf32>)
          }
          linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%23, %17 : memref<1x64x128xf32>, memref<1x64xf32>) outs(%22 : memref<1x64x128xf32>) {
          ^bb0(%in: f32, %in_5: f32, %out: f32):
            %28 = arith.divf %in, %in_5 : f32
            linalg.yield %28 : f32
          }
          %reinterpret_cast_4 = memref.reinterpret_cast %arg3 to offset: [%27], sizes: [1, 64, 128], strides: [524288, 128, 1] : memref<32x4096x128xf32> to memref<1x64x128xf32, strided<[524288, 128, 1], offset: ?>>
          loom.copy %22, %reinterpret_cast_4 src_mem_space @L1 dst_mem_space @DRAM, interconnect : [], broadcast : [1, 1] : memref<1x64x128xf32>, memref<1x64x128xf32, strided<[524288, 128, 1], offset: ?>>
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
          %15 = loom.alloc [1, 64, 128] on @L1 : memref<1x64x128xf32>
          %16 = loom.alloc [1, 128, 128] on @L1 : memref<1x128x128xf32>
          %17 = loom.alloc [1, 64] on @L1 : memref<1x64xf32>
          %18 = loom.alloc [1, 64] on @L1 : memref<1x64xf32>
          %19 = loom.alloc [1, 64] on @L1 : memref<1x64xf32>
          %20 = loom.alloc [1, 64] on @L1 : memref<1x64xf32>
          %21 = loom.alloc [1, 64] on @L1 : memref<1x64xf32>
          %22 = loom.alloc [1, 64, 128] on @L1 : memref<1x64x128xf32>
          %23 = loom.alloc [1, 64, 128] on @L1 : memref<1x64x128xf32>
          %24 = loom.alloc [1, 64, 128] on @L1 : memref<1x64x128xf32>
          %25 = arith.muli %arg6, %c524288 overflow<nsw> : index
          %26 = arith.muli %13, %c8192 : index
          %27 = arith.addi %25, %26 : index
          %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%27], sizes: [1, 64, 128], strides: [524288, 128, 1] : memref<32x4096x128xf32> to memref<1x64x128xf32, strided<[524288, 128, 1], offset: ?>>
          loom.copy %reinterpret_cast, %22 src_mem_space @DRAM dst_mem_space @L1, interconnect : [], broadcast : [1, 1] : memref<1x64x128xf32, strided<[524288, 128, 1], offset: ?>>, memref<1x64x128xf32>
          linalg.fill ins(%cst_1 : f32) outs(%23 : memref<1x64x128xf32>)
          linalg.fill ins(%cst_2 : f32) outs(%17 : memref<1x64xf32>)
          linalg.fill ins(%cst_3 : f32) outs(%18 : memref<1x64xf32>)
          scf.for %arg7 = %c0 to %c32 step %c1 {
            %28 = arith.muli %arg7, %c128 : index
            %29 = arith.addi %25, %28 : index
            %reinterpret_cast_5 = memref.reinterpret_cast %arg0 to offset: [%29], sizes: [1, 128, 128], strides: [524288, 4096, 1] : memref<32x128x4096xf32> to memref<1x128x128xf32, strided<[524288, 4096, 1], offset: ?>>
            loom.copy %reinterpret_cast_5, %16 src_mem_space @DRAM dst_mem_space @L1, interconnect : [@horizontal_links, @vertical_links], broadcast : [8, 8] : memref<1x128x128xf32, strided<[524288, 4096, 1], offset: ?>>, memref<1x128x128xf32>
            linalg.fill ins(%cst_1 : f32) outs(%15 : memref<1x64x128xf32>)
            linalg.batch_matmul ins(%22, %16 : memref<1x64x128xf32>, memref<1x128x128xf32>) outs(%15 : memref<1x64x128xf32>)
            linalg.fill ins(%cst_3 : f32) outs(%19 : memref<1x64xf32>)
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%15 : memref<1x64x128xf32>) outs(%19 : memref<1x64xf32>) {
            ^bb0(%in: f32, %out: f32):
              %32 = arith.maximumf %in, %out : f32
              linalg.yield %32 : f32
            }
            linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%18, %19 : memref<1x64xf32>, memref<1x64xf32>) outs(%19 : memref<1x64xf32>) {
            ^bb0(%in: f32, %in_7: f32, %out: f32):
              %32 = arith.truncf %cst_0 : f64 to f32
              %33 = arith.mulf %in_7, %32 : f32
              %34 = arith.cmpf ogt, %in, %33 : f32
              %35 = arith.select %34, %in, %33 : f32
              linalg.yield %35 : f32
            }
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%15, %19 : memref<1x64x128xf32>, memref<1x64xf32>) outs(%15 : memref<1x64x128xf32>) {
            ^bb0(%in: f32, %in_7: f32, %out: f32):
              %32 = arith.truncf %cst_0 : f64 to f32
              %33 = arith.mulf %in, %32 : f32
              %34 = arith.subf %33, %in_7 : f32
              %35 = math.powf %cst, %34 : f32
              linalg.yield %35 : f32
            }
            linalg.fill ins(%cst_1 : f32) outs(%20 : memref<1x64xf32>)
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%15 : memref<1x64x128xf32>) outs(%20 : memref<1x64xf32>) {
            ^bb0(%in: f32, %out: f32):
              %32 = arith.addf %in, %out : f32
              linalg.yield %32 : f32
            }
            linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%18, %19 : memref<1x64xf32>, memref<1x64xf32>) outs(%21 : memref<1x64xf32>) {
            ^bb0(%in: f32, %in_7: f32, %out: f32):
              %32 = arith.subf %in, %in_7 : f32
              %33 = math.powf %cst, %32 : f32
              linalg.yield %33 : f32
            }
            linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%17, %21, %20 : memref<1x64xf32>, memref<1x64xf32>, memref<1x64xf32>) outs(%17 : memref<1x64xf32>) {
            ^bb0(%in: f32, %in_7: f32, %in_8: f32, %out: f32):
              %32 = arith.mulf %in, %in_7 : f32
              %33 = arith.addf %32, %in_8 : f32
              linalg.yield %33 : f32
            }
            %30 = arith.muli %arg7, %c16384 : index
            %31 = arith.addi %25, %30 : index
            %reinterpret_cast_6 = memref.reinterpret_cast %arg1 to offset: [%31], sizes: [1, 128, 128], strides: [524288, 128, 1] : memref<32x4096x128xf32> to memref<1x128x128xf32, strided<[524288, 128, 1], offset: ?>>
            loom.copy %reinterpret_cast_6, %14 src_mem_space @DRAM dst_mem_space @L1, interconnect : [@vertical_links], broadcast : [8, 1] : memref<1x128x128xf32, strided<[524288, 128, 1], offset: ?>>, memref<1x128x128xf32>
            linalg.fill ins(%cst_1 : f32) outs(%24 : memref<1x64x128xf32>)
            linalg.batch_matmul ins(%15, %14 : memref<1x64x128xf32>, memref<1x128x128xf32>) outs(%24 : memref<1x64x128xf32>)
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%24, %23, %21 : memref<1x64x128xf32>, memref<1x64x128xf32>, memref<1x64xf32>) outs(%23 : memref<1x64x128xf32>) {
            ^bb0(%in: f32, %in_7: f32, %in_8: f32, %out: f32):
              %32 = arith.mulf %in_7, %in_8 : f32
              %33 = arith.addf %in, %32 : f32
              linalg.yield %33 : f32
            }
            linalg.copy ins(%19 : memref<1x64xf32>) outs(%18 : memref<1x64xf32>)
          }
          linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%23, %17 : memref<1x64x128xf32>, memref<1x64xf32>) outs(%22 : memref<1x64x128xf32>) {
          ^bb0(%in: f32, %in_5: f32, %out: f32):
            %28 = arith.divf %in, %in_5 : f32
            linalg.yield %28 : f32
          }
          %reinterpret_cast_4 = memref.reinterpret_cast %arg3 to offset: [%27], sizes: [1, 64, 128], strides: [524288, 128, 1] : memref<32x4096x128xf32> to memref<1x64x128xf32, strided<[524288, 128, 1], offset: ?>>
          loom.copy %22, %reinterpret_cast_4 src_mem_space @L1 dst_mem_space @DRAM, interconnect : [], broadcast : [1, 1] : memref<1x64x128xf32>, memref<1x64x128xf32, strided<[524288, 128, 1], offset: ?>>
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
          %15 = loom.alloc [1, 64, 128] on @L1 : memref<1x64x128xf32>
          %16 = loom.alloc [1, 128, 128] on @L1 : memref<1x128x128xf32>
          %17 = loom.alloc [1, 64] on @L1 : memref<1x64xf32>
          %18 = loom.alloc [1, 64] on @L1 : memref<1x64xf32>
          %19 = loom.alloc [1, 64] on @L1 : memref<1x64xf32>
          %20 = loom.alloc [1, 64] on @L1 : memref<1x64xf32>
          %21 = loom.alloc [1, 64] on @L1 : memref<1x64xf32>
          %22 = loom.alloc [1, 64, 128] on @L1 : memref<1x64x128xf32>
          %23 = loom.alloc [1, 64, 128] on @L1 : memref<1x64x128xf32>
          %24 = loom.alloc [1, 64, 128] on @L1 : memref<1x64x128xf32>
          %25 = arith.muli %arg6, %c524288 overflow<nsw> : index
          %26 = arith.muli %13, %c8192 : index
          %27 = arith.addi %25, %26 : index
          %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%27], sizes: [1, 64, 128], strides: [524288, 128, 1] : memref<32x4096x128xf32> to memref<1x64x128xf32, strided<[524288, 128, 1], offset: ?>>
          loom.copy %reinterpret_cast, %22 src_mem_space @DRAM dst_mem_space @L1, interconnect : [], broadcast : [1, 1] : memref<1x64x128xf32, strided<[524288, 128, 1], offset: ?>>, memref<1x64x128xf32>
          linalg.fill ins(%cst_1 : f32) outs(%23 : memref<1x64x128xf32>)
          linalg.fill ins(%cst_2 : f32) outs(%17 : memref<1x64xf32>)
          linalg.fill ins(%cst_3 : f32) outs(%18 : memref<1x64xf32>)
          scf.for %arg7 = %c0 to %c32 step %c1 {
            %28 = arith.muli %arg7, %c128 : index
            %29 = arith.addi %25, %28 : index
            %reinterpret_cast_5 = memref.reinterpret_cast %arg0 to offset: [%29], sizes: [1, 128, 128], strides: [524288, 4096, 1] : memref<32x128x4096xf32> to memref<1x128x128xf32, strided<[524288, 4096, 1], offset: ?>>
            loom.copy %reinterpret_cast_5, %16 src_mem_space @DRAM dst_mem_space @L1, interconnect : [@horizontal_links], broadcast : [1, 8] : memref<1x128x128xf32, strided<[524288, 4096, 1], offset: ?>>, memref<1x128x128xf32>
            linalg.fill ins(%cst_1 : f32) outs(%15 : memref<1x64x128xf32>)
            linalg.batch_matmul ins(%22, %16 : memref<1x64x128xf32>, memref<1x128x128xf32>) outs(%15 : memref<1x64x128xf32>)
            linalg.fill ins(%cst_3 : f32) outs(%19 : memref<1x64xf32>)
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%15 : memref<1x64x128xf32>) outs(%19 : memref<1x64xf32>) {
            ^bb0(%in: f32, %out: f32):
              %32 = arith.maximumf %in, %out : f32
              linalg.yield %32 : f32
            }
            linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%18, %19 : memref<1x64xf32>, memref<1x64xf32>) outs(%19 : memref<1x64xf32>) {
            ^bb0(%in: f32, %in_7: f32, %out: f32):
              %32 = arith.truncf %cst_0 : f64 to f32
              %33 = arith.mulf %in_7, %32 : f32
              %34 = arith.cmpf ogt, %in, %33 : f32
              %35 = arith.select %34, %in, %33 : f32
              linalg.yield %35 : f32
            }
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%15, %19 : memref<1x64x128xf32>, memref<1x64xf32>) outs(%15 : memref<1x64x128xf32>) {
            ^bb0(%in: f32, %in_7: f32, %out: f32):
              %32 = arith.truncf %cst_0 : f64 to f32
              %33 = arith.mulf %in, %32 : f32
              %34 = arith.subf %33, %in_7 : f32
              %35 = math.powf %cst, %34 : f32
              linalg.yield %35 : f32
            }
            linalg.fill ins(%cst_1 : f32) outs(%20 : memref<1x64xf32>)
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%15 : memref<1x64x128xf32>) outs(%20 : memref<1x64xf32>) {
            ^bb0(%in: f32, %out: f32):
              %32 = arith.addf %in, %out : f32
              linalg.yield %32 : f32
            }
            linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%18, %19 : memref<1x64xf32>, memref<1x64xf32>) outs(%21 : memref<1x64xf32>) {
            ^bb0(%in: f32, %in_7: f32, %out: f32):
              %32 = arith.subf %in, %in_7 : f32
              %33 = math.powf %cst, %32 : f32
              linalg.yield %33 : f32
            }
            linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%17, %21, %20 : memref<1x64xf32>, memref<1x64xf32>, memref<1x64xf32>) outs(%17 : memref<1x64xf32>) {
            ^bb0(%in: f32, %in_7: f32, %in_8: f32, %out: f32):
              %32 = arith.mulf %in, %in_7 : f32
              %33 = arith.addf %32, %in_8 : f32
              linalg.yield %33 : f32
            }
            %30 = arith.muli %arg7, %c16384 : index
            %31 = arith.addi %25, %30 : index
            %reinterpret_cast_6 = memref.reinterpret_cast %arg1 to offset: [%31], sizes: [1, 128, 128], strides: [524288, 128, 1] : memref<32x4096x128xf32> to memref<1x128x128xf32, strided<[524288, 128, 1], offset: ?>>
            loom.copy %reinterpret_cast_6, %14 src_mem_space @DRAM dst_mem_space @L1, interconnect : [], broadcast : [1, 1] : memref<1x128x128xf32, strided<[524288, 128, 1], offset: ?>>, memref<1x128x128xf32>
            linalg.fill ins(%cst_1 : f32) outs(%24 : memref<1x64x128xf32>)
            linalg.batch_matmul ins(%15, %14 : memref<1x64x128xf32>, memref<1x128x128xf32>) outs(%24 : memref<1x64x128xf32>)
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%24, %23, %21 : memref<1x64x128xf32>, memref<1x64x128xf32>, memref<1x64xf32>) outs(%23 : memref<1x64x128xf32>) {
            ^bb0(%in: f32, %in_7: f32, %in_8: f32, %out: f32):
              %32 = arith.mulf %in_7, %in_8 : f32
              %33 = arith.addf %in, %32 : f32
              linalg.yield %33 : f32
            }
            linalg.copy ins(%19 : memref<1x64xf32>) outs(%18 : memref<1x64xf32>)
          }
          linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%23, %17 : memref<1x64x128xf32>, memref<1x64xf32>) outs(%22 : memref<1x64x128xf32>) {
          ^bb0(%in: f32, %in_5: f32, %out: f32):
            %28 = arith.divf %in, %in_5 : f32
            linalg.yield %28 : f32
          }
          %reinterpret_cast_4 = memref.reinterpret_cast %arg3 to offset: [%27], sizes: [1, 64, 128], strides: [524288, 128, 1] : memref<32x4096x128xf32> to memref<1x64x128xf32, strided<[524288, 128, 1], offset: ?>>
          loom.copy %22, %reinterpret_cast_4 src_mem_space @L1 dst_mem_space @DRAM, interconnect : [], broadcast : [1, 1] : memref<1x64x128xf32>, memref<1x64x128xf32, strided<[524288, 128, 1], offset: ?>>
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
          %15 = loom.alloc [1, 64, 128] on @L1 : memref<1x64x128xf32>
          %16 = loom.alloc [1, 128, 128] on @L1 : memref<1x128x128xf32>
          %17 = loom.alloc [1, 64] on @L1 : memref<1x64xf32>
          %18 = loom.alloc [1, 64] on @L1 : memref<1x64xf32>
          %19 = loom.alloc [1, 64] on @L1 : memref<1x64xf32>
          %20 = loom.alloc [1, 64] on @L1 : memref<1x64xf32>
          %21 = loom.alloc [1, 64] on @L1 : memref<1x64xf32>
          %22 = loom.alloc [1, 64, 128] on @L1 : memref<1x64x128xf32>
          %23 = loom.alloc [1, 64, 128] on @L1 : memref<1x64x128xf32>
          %24 = loom.alloc [1, 64, 128] on @L1 : memref<1x64x128xf32>
          %25 = arith.muli %arg6, %c524288 overflow<nsw> : index
          %26 = arith.muli %13, %c8192 : index
          %27 = arith.addi %25, %26 : index
          %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%27], sizes: [1, 64, 128], strides: [524288, 128, 1] : memref<32x4096x128xf32> to memref<1x64x128xf32, strided<[524288, 128, 1], offset: ?>>
          loom.copy %reinterpret_cast, %22 src_mem_space @DRAM dst_mem_space @L1, interconnect : [], broadcast : [1, 1] : memref<1x64x128xf32, strided<[524288, 128, 1], offset: ?>>, memref<1x64x128xf32>
          linalg.fill ins(%cst_1 : f32) outs(%23 : memref<1x64x128xf32>)
          linalg.fill ins(%cst_2 : f32) outs(%17 : memref<1x64xf32>)
          linalg.fill ins(%cst_3 : f32) outs(%18 : memref<1x64xf32>)
          scf.for %arg7 = %c0 to %c32 step %c1 {
            %28 = arith.muli %arg7, %c128 : index
            %29 = arith.addi %25, %28 : index
            %reinterpret_cast_5 = memref.reinterpret_cast %arg0 to offset: [%29], sizes: [1, 128, 128], strides: [524288, 4096, 1] : memref<32x128x4096xf32> to memref<1x128x128xf32, strided<[524288, 4096, 1], offset: ?>>
            loom.copy %reinterpret_cast_5, %16 src_mem_space @DRAM dst_mem_space @L1, interconnect : [@horizontal_links], broadcast : [1, 8] : memref<1x128x128xf32, strided<[524288, 4096, 1], offset: ?>>, memref<1x128x128xf32>
            linalg.fill ins(%cst_1 : f32) outs(%15 : memref<1x64x128xf32>)
            linalg.batch_matmul ins(%22, %16 : memref<1x64x128xf32>, memref<1x128x128xf32>) outs(%15 : memref<1x64x128xf32>)
            linalg.fill ins(%cst_3 : f32) outs(%19 : memref<1x64xf32>)
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%15 : memref<1x64x128xf32>) outs(%19 : memref<1x64xf32>) {
            ^bb0(%in: f32, %out: f32):
              %32 = arith.maximumf %in, %out : f32
              linalg.yield %32 : f32
            }
            linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%18, %19 : memref<1x64xf32>, memref<1x64xf32>) outs(%19 : memref<1x64xf32>) {
            ^bb0(%in: f32, %in_7: f32, %out: f32):
              %32 = arith.truncf %cst_0 : f64 to f32
              %33 = arith.mulf %in_7, %32 : f32
              %34 = arith.cmpf ogt, %in, %33 : f32
              %35 = arith.select %34, %in, %33 : f32
              linalg.yield %35 : f32
            }
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%15, %19 : memref<1x64x128xf32>, memref<1x64xf32>) outs(%15 : memref<1x64x128xf32>) {
            ^bb0(%in: f32, %in_7: f32, %out: f32):
              %32 = arith.truncf %cst_0 : f64 to f32
              %33 = arith.mulf %in, %32 : f32
              %34 = arith.subf %33, %in_7 : f32
              %35 = math.powf %cst, %34 : f32
              linalg.yield %35 : f32
            }
            linalg.fill ins(%cst_1 : f32) outs(%20 : memref<1x64xf32>)
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%15 : memref<1x64x128xf32>) outs(%20 : memref<1x64xf32>) {
            ^bb0(%in: f32, %out: f32):
              %32 = arith.addf %in, %out : f32
              linalg.yield %32 : f32
            }
            linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%18, %19 : memref<1x64xf32>, memref<1x64xf32>) outs(%21 : memref<1x64xf32>) {
            ^bb0(%in: f32, %in_7: f32, %out: f32):
              %32 = arith.subf %in, %in_7 : f32
              %33 = math.powf %cst, %32 : f32
              linalg.yield %33 : f32
            }
            linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%17, %21, %20 : memref<1x64xf32>, memref<1x64xf32>, memref<1x64xf32>) outs(%17 : memref<1x64xf32>) {
            ^bb0(%in: f32, %in_7: f32, %in_8: f32, %out: f32):
              %32 = arith.mulf %in, %in_7 : f32
              %33 = arith.addf %32, %in_8 : f32
              linalg.yield %33 : f32
            }
            %30 = arith.muli %arg7, %c16384 : index
            %31 = arith.addi %25, %30 : index
            %reinterpret_cast_6 = memref.reinterpret_cast %arg1 to offset: [%31], sizes: [1, 128, 128], strides: [524288, 128, 1] : memref<32x4096x128xf32> to memref<1x128x128xf32, strided<[524288, 128, 1], offset: ?>>
            loom.copy %reinterpret_cast_6, %14 src_mem_space @DRAM dst_mem_space @L1, interconnect : [@horizontal_links, @vertical_links], broadcast : [8, 8] : memref<1x128x128xf32, strided<[524288, 128, 1], offset: ?>>, memref<1x128x128xf32>
            linalg.fill ins(%cst_1 : f32) outs(%24 : memref<1x64x128xf32>)
            linalg.batch_matmul ins(%15, %14 : memref<1x64x128xf32>, memref<1x128x128xf32>) outs(%24 : memref<1x64x128xf32>)
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%24, %23, %21 : memref<1x64x128xf32>, memref<1x64x128xf32>, memref<1x64xf32>) outs(%23 : memref<1x64x128xf32>) {
            ^bb0(%in: f32, %in_7: f32, %in_8: f32, %out: f32):
              %32 = arith.mulf %in_7, %in_8 : f32
              %33 = arith.addf %in, %32 : f32
              linalg.yield %33 : f32
            }
            linalg.copy ins(%19 : memref<1x64xf32>) outs(%18 : memref<1x64xf32>)
          }
          linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%23, %17 : memref<1x64x128xf32>, memref<1x64xf32>) outs(%22 : memref<1x64x128xf32>) {
          ^bb0(%in: f32, %in_5: f32, %out: f32):
            %28 = arith.divf %in, %in_5 : f32
            linalg.yield %28 : f32
          }
          %reinterpret_cast_4 = memref.reinterpret_cast %arg3 to offset: [%27], sizes: [1, 64, 128], strides: [524288, 128, 1] : memref<32x4096x128xf32> to memref<1x64x128xf32, strided<[524288, 128, 1], offset: ?>>
          loom.copy %22, %reinterpret_cast_4 src_mem_space @L1 dst_mem_space @DRAM, interconnect : [], broadcast : [1, 1] : memref<1x64x128xf32>, memref<1x64x128xf32, strided<[524288, 128, 1], offset: ?>>
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
          %15 = loom.alloc [1, 64, 128] on @L1 : memref<1x64x128xf32>
          %16 = loom.alloc [1, 128, 128] on @L1 : memref<1x128x128xf32>
          %17 = loom.alloc [1, 64] on @L1 : memref<1x64xf32>
          %18 = loom.alloc [1, 64] on @L1 : memref<1x64xf32>
          %19 = loom.alloc [1, 64] on @L1 : memref<1x64xf32>
          %20 = loom.alloc [1, 64] on @L1 : memref<1x64xf32>
          %21 = loom.alloc [1, 64] on @L1 : memref<1x64xf32>
          %22 = loom.alloc [1, 64, 128] on @L1 : memref<1x64x128xf32>
          %23 = loom.alloc [1, 64, 128] on @L1 : memref<1x64x128xf32>
          %24 = loom.alloc [1, 64, 128] on @L1 : memref<1x64x128xf32>
          %25 = arith.muli %arg6, %c524288 overflow<nsw> : index
          %26 = arith.muli %13, %c8192 : index
          %27 = arith.addi %25, %26 : index
          %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%27], sizes: [1, 64, 128], strides: [524288, 128, 1] : memref<32x4096x128xf32> to memref<1x64x128xf32, strided<[524288, 128, 1], offset: ?>>
          loom.copy %reinterpret_cast, %22 src_mem_space @DRAM dst_mem_space @L1, interconnect : [], broadcast : [1, 1] : memref<1x64x128xf32, strided<[524288, 128, 1], offset: ?>>, memref<1x64x128xf32>
          linalg.fill ins(%cst_1 : f32) outs(%23 : memref<1x64x128xf32>)
          linalg.fill ins(%cst_2 : f32) outs(%17 : memref<1x64xf32>)
          linalg.fill ins(%cst_3 : f32) outs(%18 : memref<1x64xf32>)
          scf.for %arg7 = %c0 to %c32 step %c1 {
            %28 = arith.muli %arg7, %c128 : index
            %29 = arith.addi %25, %28 : index
            %reinterpret_cast_5 = memref.reinterpret_cast %arg0 to offset: [%29], sizes: [1, 128, 128], strides: [524288, 4096, 1] : memref<32x128x4096xf32> to memref<1x128x128xf32, strided<[524288, 4096, 1], offset: ?>>
            loom.copy %reinterpret_cast_5, %16 src_mem_space @DRAM dst_mem_space @L1, interconnect : [@horizontal_links], broadcast : [1, 8] : memref<1x128x128xf32, strided<[524288, 4096, 1], offset: ?>>, memref<1x128x128xf32>
            linalg.fill ins(%cst_1 : f32) outs(%15 : memref<1x64x128xf32>)
            linalg.batch_matmul ins(%22, %16 : memref<1x64x128xf32>, memref<1x128x128xf32>) outs(%15 : memref<1x64x128xf32>)
            linalg.fill ins(%cst_3 : f32) outs(%19 : memref<1x64xf32>)
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%15 : memref<1x64x128xf32>) outs(%19 : memref<1x64xf32>) {
            ^bb0(%in: f32, %out: f32):
              %32 = arith.maximumf %in, %out : f32
              linalg.yield %32 : f32
            }
            linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%18, %19 : memref<1x64xf32>, memref<1x64xf32>) outs(%19 : memref<1x64xf32>) {
            ^bb0(%in: f32, %in_7: f32, %out: f32):
              %32 = arith.truncf %cst_0 : f64 to f32
              %33 = arith.mulf %in_7, %32 : f32
              %34 = arith.cmpf ogt, %in, %33 : f32
              %35 = arith.select %34, %in, %33 : f32
              linalg.yield %35 : f32
            }
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%15, %19 : memref<1x64x128xf32>, memref<1x64xf32>) outs(%15 : memref<1x64x128xf32>) {
            ^bb0(%in: f32, %in_7: f32, %out: f32):
              %32 = arith.truncf %cst_0 : f64 to f32
              %33 = arith.mulf %in, %32 : f32
              %34 = arith.subf %33, %in_7 : f32
              %35 = math.powf %cst, %34 : f32
              linalg.yield %35 : f32
            }
            linalg.fill ins(%cst_1 : f32) outs(%20 : memref<1x64xf32>)
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%15 : memref<1x64x128xf32>) outs(%20 : memref<1x64xf32>) {
            ^bb0(%in: f32, %out: f32):
              %32 = arith.addf %in, %out : f32
              linalg.yield %32 : f32
            }
            linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%18, %19 : memref<1x64xf32>, memref<1x64xf32>) outs(%21 : memref<1x64xf32>) {
            ^bb0(%in: f32, %in_7: f32, %out: f32):
              %32 = arith.subf %in, %in_7 : f32
              %33 = math.powf %cst, %32 : f32
              linalg.yield %33 : f32
            }
            linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%17, %21, %20 : memref<1x64xf32>, memref<1x64xf32>, memref<1x64xf32>) outs(%17 : memref<1x64xf32>) {
            ^bb0(%in: f32, %in_7: f32, %in_8: f32, %out: f32):
              %32 = arith.mulf %in, %in_7 : f32
              %33 = arith.addf %32, %in_8 : f32
              linalg.yield %33 : f32
            }
            %30 = arith.muli %arg7, %c16384 : index
            %31 = arith.addi %25, %30 : index
            %reinterpret_cast_6 = memref.reinterpret_cast %arg1 to offset: [%31], sizes: [1, 128, 128], strides: [524288, 128, 1] : memref<32x4096x128xf32> to memref<1x128x128xf32, strided<[524288, 128, 1], offset: ?>>
            loom.copy %reinterpret_cast_6, %14 src_mem_space @DRAM dst_mem_space @L1, interconnect : [@horizontal_links], broadcast : [1, 8] : memref<1x128x128xf32, strided<[524288, 128, 1], offset: ?>>, memref<1x128x128xf32>
            linalg.fill ins(%cst_1 : f32) outs(%24 : memref<1x64x128xf32>)
            linalg.batch_matmul ins(%15, %14 : memref<1x64x128xf32>, memref<1x128x128xf32>) outs(%24 : memref<1x64x128xf32>)
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%24, %23, %21 : memref<1x64x128xf32>, memref<1x64x128xf32>, memref<1x64xf32>) outs(%23 : memref<1x64x128xf32>) {
            ^bb0(%in: f32, %in_7: f32, %in_8: f32, %out: f32):
              %32 = arith.mulf %in_7, %in_8 : f32
              %33 = arith.addf %in, %32 : f32
              linalg.yield %33 : f32
            }
            linalg.copy ins(%19 : memref<1x64xf32>) outs(%18 : memref<1x64xf32>)
          }
          linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%23, %17 : memref<1x64x128xf32>, memref<1x64xf32>) outs(%22 : memref<1x64x128xf32>) {
          ^bb0(%in: f32, %in_5: f32, %out: f32):
            %28 = arith.divf %in, %in_5 : f32
            linalg.yield %28 : f32
          }
          %reinterpret_cast_4 = memref.reinterpret_cast %arg3 to offset: [%27], sizes: [1, 64, 128], strides: [524288, 128, 1] : memref<32x4096x128xf32> to memref<1x64x128xf32, strided<[524288, 128, 1], offset: ?>>
          loom.copy %22, %reinterpret_cast_4 src_mem_space @L1 dst_mem_space @DRAM, interconnect : [], broadcast : [1, 1] : memref<1x64x128xf32>, memref<1x64x128xf32, strided<[524288, 128, 1], offset: ?>>
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
          %15 = loom.alloc [1, 64, 128] on @L1 : memref<1x64x128xf32>
          %16 = loom.alloc [1, 128, 128] on @L1 : memref<1x128x128xf32>
          %17 = loom.alloc [1, 64] on @L1 : memref<1x64xf32>
          %18 = loom.alloc [1, 64] on @L1 : memref<1x64xf32>
          %19 = loom.alloc [1, 64] on @L1 : memref<1x64xf32>
          %20 = loom.alloc [1, 64] on @L1 : memref<1x64xf32>
          %21 = loom.alloc [1, 64] on @L1 : memref<1x64xf32>
          %22 = loom.alloc [1, 64, 128] on @L1 : memref<1x64x128xf32>
          %23 = loom.alloc [1, 64, 128] on @L1 : memref<1x64x128xf32>
          %24 = loom.alloc [1, 64, 128] on @L1 : memref<1x64x128xf32>
          %25 = arith.muli %arg6, %c524288 overflow<nsw> : index
          %26 = arith.muli %13, %c8192 : index
          %27 = arith.addi %25, %26 : index
          %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%27], sizes: [1, 64, 128], strides: [524288, 128, 1] : memref<32x4096x128xf32> to memref<1x64x128xf32, strided<[524288, 128, 1], offset: ?>>
          loom.copy %reinterpret_cast, %22 src_mem_space @DRAM dst_mem_space @L1, interconnect : [], broadcast : [1, 1] : memref<1x64x128xf32, strided<[524288, 128, 1], offset: ?>>, memref<1x64x128xf32>
          linalg.fill ins(%cst_1 : f32) outs(%23 : memref<1x64x128xf32>)
          linalg.fill ins(%cst_2 : f32) outs(%17 : memref<1x64xf32>)
          linalg.fill ins(%cst_3 : f32) outs(%18 : memref<1x64xf32>)
          scf.for %arg7 = %c0 to %c32 step %c1 {
            %28 = arith.muli %arg7, %c128 : index
            %29 = arith.addi %25, %28 : index
            %reinterpret_cast_5 = memref.reinterpret_cast %arg0 to offset: [%29], sizes: [1, 128, 128], strides: [524288, 4096, 1] : memref<32x128x4096xf32> to memref<1x128x128xf32, strided<[524288, 4096, 1], offset: ?>>
            loom.copy %reinterpret_cast_5, %16 src_mem_space @DRAM dst_mem_space @L1, interconnect : [@horizontal_links], broadcast : [1, 8] : memref<1x128x128xf32, strided<[524288, 4096, 1], offset: ?>>, memref<1x128x128xf32>
            linalg.fill ins(%cst_1 : f32) outs(%15 : memref<1x64x128xf32>)
            linalg.batch_matmul ins(%22, %16 : memref<1x64x128xf32>, memref<1x128x128xf32>) outs(%15 : memref<1x64x128xf32>)
            linalg.fill ins(%cst_3 : f32) outs(%19 : memref<1x64xf32>)
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%15 : memref<1x64x128xf32>) outs(%19 : memref<1x64xf32>) {
            ^bb0(%in: f32, %out: f32):
              %32 = arith.maximumf %in, %out : f32
              linalg.yield %32 : f32
            }
            linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%18, %19 : memref<1x64xf32>, memref<1x64xf32>) outs(%19 : memref<1x64xf32>) {
            ^bb0(%in: f32, %in_7: f32, %out: f32):
              %32 = arith.truncf %cst_0 : f64 to f32
              %33 = arith.mulf %in_7, %32 : f32
              %34 = arith.cmpf ogt, %in, %33 : f32
              %35 = arith.select %34, %in, %33 : f32
              linalg.yield %35 : f32
            }
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%15, %19 : memref<1x64x128xf32>, memref<1x64xf32>) outs(%15 : memref<1x64x128xf32>) {
            ^bb0(%in: f32, %in_7: f32, %out: f32):
              %32 = arith.truncf %cst_0 : f64 to f32
              %33 = arith.mulf %in, %32 : f32
              %34 = arith.subf %33, %in_7 : f32
              %35 = math.powf %cst, %34 : f32
              linalg.yield %35 : f32
            }
            linalg.fill ins(%cst_1 : f32) outs(%20 : memref<1x64xf32>)
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%15 : memref<1x64x128xf32>) outs(%20 : memref<1x64xf32>) {
            ^bb0(%in: f32, %out: f32):
              %32 = arith.addf %in, %out : f32
              linalg.yield %32 : f32
            }
            linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%18, %19 : memref<1x64xf32>, memref<1x64xf32>) outs(%21 : memref<1x64xf32>) {
            ^bb0(%in: f32, %in_7: f32, %out: f32):
              %32 = arith.subf %in, %in_7 : f32
              %33 = math.powf %cst, %32 : f32
              linalg.yield %33 : f32
            }
            linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%17, %21, %20 : memref<1x64xf32>, memref<1x64xf32>, memref<1x64xf32>) outs(%17 : memref<1x64xf32>) {
            ^bb0(%in: f32, %in_7: f32, %in_8: f32, %out: f32):
              %32 = arith.mulf %in, %in_7 : f32
              %33 = arith.addf %32, %in_8 : f32
              linalg.yield %33 : f32
            }
            %30 = arith.muli %arg7, %c16384 : index
            %31 = arith.addi %25, %30 : index
            %reinterpret_cast_6 = memref.reinterpret_cast %arg1 to offset: [%31], sizes: [1, 128, 128], strides: [524288, 128, 1] : memref<32x4096x128xf32> to memref<1x128x128xf32, strided<[524288, 128, 1], offset: ?>>
            loom.copy %reinterpret_cast_6, %14 src_mem_space @DRAM dst_mem_space @L1, interconnect : [@vertical_links], broadcast : [8, 1] : memref<1x128x128xf32, strided<[524288, 128, 1], offset: ?>>, memref<1x128x128xf32>
            linalg.fill ins(%cst_1 : f32) outs(%24 : memref<1x64x128xf32>)
            linalg.batch_matmul ins(%15, %14 : memref<1x64x128xf32>, memref<1x128x128xf32>) outs(%24 : memref<1x64x128xf32>)
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%24, %23, %21 : memref<1x64x128xf32>, memref<1x64x128xf32>, memref<1x64xf32>) outs(%23 : memref<1x64x128xf32>) {
            ^bb0(%in: f32, %in_7: f32, %in_8: f32, %out: f32):
              %32 = arith.mulf %in_7, %in_8 : f32
              %33 = arith.addf %in, %32 : f32
              linalg.yield %33 : f32
            }
            linalg.copy ins(%19 : memref<1x64xf32>) outs(%18 : memref<1x64xf32>)
          }
          linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%23, %17 : memref<1x64x128xf32>, memref<1x64xf32>) outs(%22 : memref<1x64x128xf32>) {
          ^bb0(%in: f32, %in_5: f32, %out: f32):
            %28 = arith.divf %in, %in_5 : f32
            linalg.yield %28 : f32
          }
          %reinterpret_cast_4 = memref.reinterpret_cast %arg3 to offset: [%27], sizes: [1, 64, 128], strides: [524288, 128, 1] : memref<32x4096x128xf32> to memref<1x64x128xf32, strided<[524288, 128, 1], offset: ?>>
          loom.copy %22, %reinterpret_cast_4 src_mem_space @L1 dst_mem_space @DRAM, interconnect : [], broadcast : [1, 1] : memref<1x64x128xf32>, memref<1x64x128xf32, strided<[524288, 128, 1], offset: ?>>
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
          %15 = loom.alloc [1, 64, 128] on @L1 : memref<1x64x128xf32>
          %16 = loom.alloc [1, 128, 128] on @L1 : memref<1x128x128xf32>
          %17 = loom.alloc [1, 64] on @L1 : memref<1x64xf32>
          %18 = loom.alloc [1, 64] on @L1 : memref<1x64xf32>
          %19 = loom.alloc [1, 64] on @L1 : memref<1x64xf32>
          %20 = loom.alloc [1, 64] on @L1 : memref<1x64xf32>
          %21 = loom.alloc [1, 64] on @L1 : memref<1x64xf32>
          %22 = loom.alloc [1, 64, 128] on @L1 : memref<1x64x128xf32>
          %23 = loom.alloc [1, 64, 128] on @L1 : memref<1x64x128xf32>
          %24 = loom.alloc [1, 64, 128] on @L1 : memref<1x64x128xf32>
          %25 = arith.muli %arg6, %c524288 overflow<nsw> : index
          %26 = arith.muli %13, %c8192 : index
          %27 = arith.addi %25, %26 : index
          %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%27], sizes: [1, 64, 128], strides: [524288, 128, 1] : memref<32x4096x128xf32> to memref<1x64x128xf32, strided<[524288, 128, 1], offset: ?>>
          loom.copy %reinterpret_cast, %22 src_mem_space @DRAM dst_mem_space @L1, interconnect : [], broadcast : [1, 1] : memref<1x64x128xf32, strided<[524288, 128, 1], offset: ?>>, memref<1x64x128xf32>
          linalg.fill ins(%cst_1 : f32) outs(%23 : memref<1x64x128xf32>)
          linalg.fill ins(%cst_2 : f32) outs(%17 : memref<1x64xf32>)
          linalg.fill ins(%cst_3 : f32) outs(%18 : memref<1x64xf32>)
          scf.for %arg7 = %c0 to %c32 step %c1 {
            %28 = arith.muli %arg7, %c128 : index
            %29 = arith.addi %25, %28 : index
            %reinterpret_cast_5 = memref.reinterpret_cast %arg0 to offset: [%29], sizes: [1, 128, 128], strides: [524288, 4096, 1] : memref<32x128x4096xf32> to memref<1x128x128xf32, strided<[524288, 4096, 1], offset: ?>>
            loom.copy %reinterpret_cast_5, %16 src_mem_space @DRAM dst_mem_space @L1, interconnect : [@vertical_links], broadcast : [8, 1] : memref<1x128x128xf32, strided<[524288, 4096, 1], offset: ?>>, memref<1x128x128xf32>
            linalg.fill ins(%cst_1 : f32) outs(%15 : memref<1x64x128xf32>)
            linalg.batch_matmul ins(%22, %16 : memref<1x64x128xf32>, memref<1x128x128xf32>) outs(%15 : memref<1x64x128xf32>)
            linalg.fill ins(%cst_3 : f32) outs(%19 : memref<1x64xf32>)
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%15 : memref<1x64x128xf32>) outs(%19 : memref<1x64xf32>) {
            ^bb0(%in: f32, %out: f32):
              %32 = arith.maximumf %in, %out : f32
              linalg.yield %32 : f32
            }
            linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%18, %19 : memref<1x64xf32>, memref<1x64xf32>) outs(%19 : memref<1x64xf32>) {
            ^bb0(%in: f32, %in_7: f32, %out: f32):
              %32 = arith.truncf %cst_0 : f64 to f32
              %33 = arith.mulf %in_7, %32 : f32
              %34 = arith.cmpf ogt, %in, %33 : f32
              %35 = arith.select %34, %in, %33 : f32
              linalg.yield %35 : f32
            }
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%15, %19 : memref<1x64x128xf32>, memref<1x64xf32>) outs(%15 : memref<1x64x128xf32>) {
            ^bb0(%in: f32, %in_7: f32, %out: f32):
              %32 = arith.truncf %cst_0 : f64 to f32
              %33 = arith.mulf %in, %32 : f32
              %34 = arith.subf %33, %in_7 : f32
              %35 = math.powf %cst, %34 : f32
              linalg.yield %35 : f32
            }
            linalg.fill ins(%cst_1 : f32) outs(%20 : memref<1x64xf32>)
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%15 : memref<1x64x128xf32>) outs(%20 : memref<1x64xf32>) {
            ^bb0(%in: f32, %out: f32):
              %32 = arith.addf %in, %out : f32
              linalg.yield %32 : f32
            }
            linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%18, %19 : memref<1x64xf32>, memref<1x64xf32>) outs(%21 : memref<1x64xf32>) {
            ^bb0(%in: f32, %in_7: f32, %out: f32):
              %32 = arith.subf %in, %in_7 : f32
              %33 = math.powf %cst, %32 : f32
              linalg.yield %33 : f32
            }
            linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%17, %21, %20 : memref<1x64xf32>, memref<1x64xf32>, memref<1x64xf32>) outs(%17 : memref<1x64xf32>) {
            ^bb0(%in: f32, %in_7: f32, %in_8: f32, %out: f32):
              %32 = arith.mulf %in, %in_7 : f32
              %33 = arith.addf %32, %in_8 : f32
              linalg.yield %33 : f32
            }
            %30 = arith.muli %arg7, %c16384 : index
            %31 = arith.addi %25, %30 : index
            %reinterpret_cast_6 = memref.reinterpret_cast %arg1 to offset: [%31], sizes: [1, 128, 128], strides: [524288, 128, 1] : memref<32x4096x128xf32> to memref<1x128x128xf32, strided<[524288, 128, 1], offset: ?>>
            loom.copy %reinterpret_cast_6, %14 src_mem_space @DRAM dst_mem_space @L1, interconnect : [], broadcast : [1, 1] : memref<1x128x128xf32, strided<[524288, 128, 1], offset: ?>>, memref<1x128x128xf32>
            linalg.fill ins(%cst_1 : f32) outs(%24 : memref<1x64x128xf32>)
            linalg.batch_matmul ins(%15, %14 : memref<1x64x128xf32>, memref<1x128x128xf32>) outs(%24 : memref<1x64x128xf32>)
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%24, %23, %21 : memref<1x64x128xf32>, memref<1x64x128xf32>, memref<1x64xf32>) outs(%23 : memref<1x64x128xf32>) {
            ^bb0(%in: f32, %in_7: f32, %in_8: f32, %out: f32):
              %32 = arith.mulf %in_7, %in_8 : f32
              %33 = arith.addf %in, %32 : f32
              linalg.yield %33 : f32
            }
            linalg.copy ins(%19 : memref<1x64xf32>) outs(%18 : memref<1x64xf32>)
          }
          linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%23, %17 : memref<1x64x128xf32>, memref<1x64xf32>) outs(%22 : memref<1x64x128xf32>) {
          ^bb0(%in: f32, %in_5: f32, %out: f32):
            %28 = arith.divf %in, %in_5 : f32
            linalg.yield %28 : f32
          }
          %reinterpret_cast_4 = memref.reinterpret_cast %arg3 to offset: [%27], sizes: [1, 64, 128], strides: [524288, 128, 1] : memref<32x4096x128xf32> to memref<1x64x128xf32, strided<[524288, 128, 1], offset: ?>>
          loom.copy %22, %reinterpret_cast_4 src_mem_space @L1 dst_mem_space @DRAM, interconnect : [], broadcast : [1, 1] : memref<1x64x128xf32>, memref<1x64x128xf32, strided<[524288, 128, 1], offset: ?>>
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
          %15 = loom.alloc [1, 64, 128] on @L1 : memref<1x64x128xf32>
          %16 = loom.alloc [1, 128, 128] on @L1 : memref<1x128x128xf32>
          %17 = loom.alloc [1, 64] on @L1 : memref<1x64xf32>
          %18 = loom.alloc [1, 64] on @L1 : memref<1x64xf32>
          %19 = loom.alloc [1, 64] on @L1 : memref<1x64xf32>
          %20 = loom.alloc [1, 64] on @L1 : memref<1x64xf32>
          %21 = loom.alloc [1, 64] on @L1 : memref<1x64xf32>
          %22 = loom.alloc [1, 64, 128] on @L1 : memref<1x64x128xf32>
          %23 = loom.alloc [1, 64, 128] on @L1 : memref<1x64x128xf32>
          %24 = loom.alloc [1, 64, 128] on @L1 : memref<1x64x128xf32>
          %25 = arith.muli %arg6, %c524288 overflow<nsw> : index
          %26 = arith.muli %13, %c8192 : index
          %27 = arith.addi %25, %26 : index
          %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%27], sizes: [1, 64, 128], strides: [524288, 128, 1] : memref<32x4096x128xf32> to memref<1x64x128xf32, strided<[524288, 128, 1], offset: ?>>
          loom.copy %reinterpret_cast, %22 src_mem_space @DRAM dst_mem_space @L1, interconnect : [], broadcast : [1, 1] : memref<1x64x128xf32, strided<[524288, 128, 1], offset: ?>>, memref<1x64x128xf32>
          linalg.fill ins(%cst_1 : f32) outs(%23 : memref<1x64x128xf32>)
          linalg.fill ins(%cst_2 : f32) outs(%17 : memref<1x64xf32>)
          linalg.fill ins(%cst_3 : f32) outs(%18 : memref<1x64xf32>)
          scf.for %arg7 = %c0 to %c32 step %c1 {
            %28 = arith.muli %arg7, %c128 : index
            %29 = arith.addi %25, %28 : index
            %reinterpret_cast_5 = memref.reinterpret_cast %arg0 to offset: [%29], sizes: [1, 128, 128], strides: [524288, 4096, 1] : memref<32x128x4096xf32> to memref<1x128x128xf32, strided<[524288, 4096, 1], offset: ?>>
            loom.copy %reinterpret_cast_5, %16 src_mem_space @DRAM dst_mem_space @L1, interconnect : [@vertical_links], broadcast : [8, 1] : memref<1x128x128xf32, strided<[524288, 4096, 1], offset: ?>>, memref<1x128x128xf32>
            linalg.fill ins(%cst_1 : f32) outs(%15 : memref<1x64x128xf32>)
            linalg.batch_matmul ins(%22, %16 : memref<1x64x128xf32>, memref<1x128x128xf32>) outs(%15 : memref<1x64x128xf32>)
            linalg.fill ins(%cst_3 : f32) outs(%19 : memref<1x64xf32>)
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%15 : memref<1x64x128xf32>) outs(%19 : memref<1x64xf32>) {
            ^bb0(%in: f32, %out: f32):
              %32 = arith.maximumf %in, %out : f32
              linalg.yield %32 : f32
            }
            linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%18, %19 : memref<1x64xf32>, memref<1x64xf32>) outs(%19 : memref<1x64xf32>) {
            ^bb0(%in: f32, %in_7: f32, %out: f32):
              %32 = arith.truncf %cst_0 : f64 to f32
              %33 = arith.mulf %in_7, %32 : f32
              %34 = arith.cmpf ogt, %in, %33 : f32
              %35 = arith.select %34, %in, %33 : f32
              linalg.yield %35 : f32
            }
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%15, %19 : memref<1x64x128xf32>, memref<1x64xf32>) outs(%15 : memref<1x64x128xf32>) {
            ^bb0(%in: f32, %in_7: f32, %out: f32):
              %32 = arith.truncf %cst_0 : f64 to f32
              %33 = arith.mulf %in, %32 : f32
              %34 = arith.subf %33, %in_7 : f32
              %35 = math.powf %cst, %34 : f32
              linalg.yield %35 : f32
            }
            linalg.fill ins(%cst_1 : f32) outs(%20 : memref<1x64xf32>)
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%15 : memref<1x64x128xf32>) outs(%20 : memref<1x64xf32>) {
            ^bb0(%in: f32, %out: f32):
              %32 = arith.addf %in, %out : f32
              linalg.yield %32 : f32
            }
            linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%18, %19 : memref<1x64xf32>, memref<1x64xf32>) outs(%21 : memref<1x64xf32>) {
            ^bb0(%in: f32, %in_7: f32, %out: f32):
              %32 = arith.subf %in, %in_7 : f32
              %33 = math.powf %cst, %32 : f32
              linalg.yield %33 : f32
            }
            linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%17, %21, %20 : memref<1x64xf32>, memref<1x64xf32>, memref<1x64xf32>) outs(%17 : memref<1x64xf32>) {
            ^bb0(%in: f32, %in_7: f32, %in_8: f32, %out: f32):
              %32 = arith.mulf %in, %in_7 : f32
              %33 = arith.addf %32, %in_8 : f32
              linalg.yield %33 : f32
            }
            %30 = arith.muli %arg7, %c16384 : index
            %31 = arith.addi %25, %30 : index
            %reinterpret_cast_6 = memref.reinterpret_cast %arg1 to offset: [%31], sizes: [1, 128, 128], strides: [524288, 128, 1] : memref<32x4096x128xf32> to memref<1x128x128xf32, strided<[524288, 128, 1], offset: ?>>
            loom.copy %reinterpret_cast_6, %14 src_mem_space @DRAM dst_mem_space @L1, interconnect : [@horizontal_links, @vertical_links], broadcast : [8, 8] : memref<1x128x128xf32, strided<[524288, 128, 1], offset: ?>>, memref<1x128x128xf32>
            linalg.fill ins(%cst_1 : f32) outs(%24 : memref<1x64x128xf32>)
            linalg.batch_matmul ins(%15, %14 : memref<1x64x128xf32>, memref<1x128x128xf32>) outs(%24 : memref<1x64x128xf32>)
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%24, %23, %21 : memref<1x64x128xf32>, memref<1x64x128xf32>, memref<1x64xf32>) outs(%23 : memref<1x64x128xf32>) {
            ^bb0(%in: f32, %in_7: f32, %in_8: f32, %out: f32):
              %32 = arith.mulf %in_7, %in_8 : f32
              %33 = arith.addf %in, %32 : f32
              linalg.yield %33 : f32
            }
            linalg.copy ins(%19 : memref<1x64xf32>) outs(%18 : memref<1x64xf32>)
          }
          linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%23, %17 : memref<1x64x128xf32>, memref<1x64xf32>) outs(%22 : memref<1x64x128xf32>) {
          ^bb0(%in: f32, %in_5: f32, %out: f32):
            %28 = arith.divf %in, %in_5 : f32
            linalg.yield %28 : f32
          }
          %reinterpret_cast_4 = memref.reinterpret_cast %arg3 to offset: [%27], sizes: [1, 64, 128], strides: [524288, 128, 1] : memref<32x4096x128xf32> to memref<1x64x128xf32, strided<[524288, 128, 1], offset: ?>>
          loom.copy %22, %reinterpret_cast_4 src_mem_space @L1 dst_mem_space @DRAM, interconnect : [], broadcast : [1, 1] : memref<1x64x128xf32>, memref<1x64x128xf32, strided<[524288, 128, 1], offset: ?>>
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
          %15 = loom.alloc [1, 64, 128] on @L1 : memref<1x64x128xf32>
          %16 = loom.alloc [1, 128, 128] on @L1 : memref<1x128x128xf32>
          %17 = loom.alloc [1, 64] on @L1 : memref<1x64xf32>
          %18 = loom.alloc [1, 64] on @L1 : memref<1x64xf32>
          %19 = loom.alloc [1, 64] on @L1 : memref<1x64xf32>
          %20 = loom.alloc [1, 64] on @L1 : memref<1x64xf32>
          %21 = loom.alloc [1, 64] on @L1 : memref<1x64xf32>
          %22 = loom.alloc [1, 64, 128] on @L1 : memref<1x64x128xf32>
          %23 = loom.alloc [1, 64, 128] on @L1 : memref<1x64x128xf32>
          %24 = loom.alloc [1, 64, 128] on @L1 : memref<1x64x128xf32>
          %25 = arith.muli %arg6, %c524288 overflow<nsw> : index
          %26 = arith.muli %13, %c8192 : index
          %27 = arith.addi %25, %26 : index
          %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%27], sizes: [1, 64, 128], strides: [524288, 128, 1] : memref<32x4096x128xf32> to memref<1x64x128xf32, strided<[524288, 128, 1], offset: ?>>
          loom.copy %reinterpret_cast, %22 src_mem_space @DRAM dst_mem_space @L1, interconnect : [], broadcast : [1, 1] : memref<1x64x128xf32, strided<[524288, 128, 1], offset: ?>>, memref<1x64x128xf32>
          linalg.fill ins(%cst_1 : f32) outs(%23 : memref<1x64x128xf32>)
          linalg.fill ins(%cst_2 : f32) outs(%17 : memref<1x64xf32>)
          linalg.fill ins(%cst_3 : f32) outs(%18 : memref<1x64xf32>)
          scf.for %arg7 = %c0 to %c32 step %c1 {
            %28 = arith.muli %arg7, %c128 : index
            %29 = arith.addi %25, %28 : index
            %reinterpret_cast_5 = memref.reinterpret_cast %arg0 to offset: [%29], sizes: [1, 128, 128], strides: [524288, 4096, 1] : memref<32x128x4096xf32> to memref<1x128x128xf32, strided<[524288, 4096, 1], offset: ?>>
            loom.copy %reinterpret_cast_5, %16 src_mem_space @DRAM dst_mem_space @L1, interconnect : [@vertical_links], broadcast : [8, 1] : memref<1x128x128xf32, strided<[524288, 4096, 1], offset: ?>>, memref<1x128x128xf32>
            linalg.fill ins(%cst_1 : f32) outs(%15 : memref<1x64x128xf32>)
            linalg.batch_matmul ins(%22, %16 : memref<1x64x128xf32>, memref<1x128x128xf32>) outs(%15 : memref<1x64x128xf32>)
            linalg.fill ins(%cst_3 : f32) outs(%19 : memref<1x64xf32>)
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%15 : memref<1x64x128xf32>) outs(%19 : memref<1x64xf32>) {
            ^bb0(%in: f32, %out: f32):
              %32 = arith.maximumf %in, %out : f32
              linalg.yield %32 : f32
            }
            linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%18, %19 : memref<1x64xf32>, memref<1x64xf32>) outs(%19 : memref<1x64xf32>) {
            ^bb0(%in: f32, %in_7: f32, %out: f32):
              %32 = arith.truncf %cst_0 : f64 to f32
              %33 = arith.mulf %in_7, %32 : f32
              %34 = arith.cmpf ogt, %in, %33 : f32
              %35 = arith.select %34, %in, %33 : f32
              linalg.yield %35 : f32
            }
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%15, %19 : memref<1x64x128xf32>, memref<1x64xf32>) outs(%15 : memref<1x64x128xf32>) {
            ^bb0(%in: f32, %in_7: f32, %out: f32):
              %32 = arith.truncf %cst_0 : f64 to f32
              %33 = arith.mulf %in, %32 : f32
              %34 = arith.subf %33, %in_7 : f32
              %35 = math.powf %cst, %34 : f32
              linalg.yield %35 : f32
            }
            linalg.fill ins(%cst_1 : f32) outs(%20 : memref<1x64xf32>)
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%15 : memref<1x64x128xf32>) outs(%20 : memref<1x64xf32>) {
            ^bb0(%in: f32, %out: f32):
              %32 = arith.addf %in, %out : f32
              linalg.yield %32 : f32
            }
            linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%18, %19 : memref<1x64xf32>, memref<1x64xf32>) outs(%21 : memref<1x64xf32>) {
            ^bb0(%in: f32, %in_7: f32, %out: f32):
              %32 = arith.subf %in, %in_7 : f32
              %33 = math.powf %cst, %32 : f32
              linalg.yield %33 : f32
            }
            linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%17, %21, %20 : memref<1x64xf32>, memref<1x64xf32>, memref<1x64xf32>) outs(%17 : memref<1x64xf32>) {
            ^bb0(%in: f32, %in_7: f32, %in_8: f32, %out: f32):
              %32 = arith.mulf %in, %in_7 : f32
              %33 = arith.addf %32, %in_8 : f32
              linalg.yield %33 : f32
            }
            %30 = arith.muli %arg7, %c16384 : index
            %31 = arith.addi %25, %30 : index
            %reinterpret_cast_6 = memref.reinterpret_cast %arg1 to offset: [%31], sizes: [1, 128, 128], strides: [524288, 128, 1] : memref<32x4096x128xf32> to memref<1x128x128xf32, strided<[524288, 128, 1], offset: ?>>
            loom.copy %reinterpret_cast_6, %14 src_mem_space @DRAM dst_mem_space @L1, interconnect : [@horizontal_links], broadcast : [1, 8] : memref<1x128x128xf32, strided<[524288, 128, 1], offset: ?>>, memref<1x128x128xf32>
            linalg.fill ins(%cst_1 : f32) outs(%24 : memref<1x64x128xf32>)
            linalg.batch_matmul ins(%15, %14 : memref<1x64x128xf32>, memref<1x128x128xf32>) outs(%24 : memref<1x64x128xf32>)
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%24, %23, %21 : memref<1x64x128xf32>, memref<1x64x128xf32>, memref<1x64xf32>) outs(%23 : memref<1x64x128xf32>) {
            ^bb0(%in: f32, %in_7: f32, %in_8: f32, %out: f32):
              %32 = arith.mulf %in_7, %in_8 : f32
              %33 = arith.addf %in, %32 : f32
              linalg.yield %33 : f32
            }
            linalg.copy ins(%19 : memref<1x64xf32>) outs(%18 : memref<1x64xf32>)
          }
          linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%23, %17 : memref<1x64x128xf32>, memref<1x64xf32>) outs(%22 : memref<1x64x128xf32>) {
          ^bb0(%in: f32, %in_5: f32, %out: f32):
            %28 = arith.divf %in, %in_5 : f32
            linalg.yield %28 : f32
          }
          %reinterpret_cast_4 = memref.reinterpret_cast %arg3 to offset: [%27], sizes: [1, 64, 128], strides: [524288, 128, 1] : memref<32x4096x128xf32> to memref<1x64x128xf32, strided<[524288, 128, 1], offset: ?>>
          loom.copy %22, %reinterpret_cast_4 src_mem_space @L1 dst_mem_space @DRAM, interconnect : [], broadcast : [1, 1] : memref<1x64x128xf32>, memref<1x64x128xf32, strided<[524288, 128, 1], offset: ?>>
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
          %15 = loom.alloc [1, 64, 128] on @L1 : memref<1x64x128xf32>
          %16 = loom.alloc [1, 128, 128] on @L1 : memref<1x128x128xf32>
          %17 = loom.alloc [1, 64] on @L1 : memref<1x64xf32>
          %18 = loom.alloc [1, 64] on @L1 : memref<1x64xf32>
          %19 = loom.alloc [1, 64] on @L1 : memref<1x64xf32>
          %20 = loom.alloc [1, 64] on @L1 : memref<1x64xf32>
          %21 = loom.alloc [1, 64] on @L1 : memref<1x64xf32>
          %22 = loom.alloc [1, 64, 128] on @L1 : memref<1x64x128xf32>
          %23 = loom.alloc [1, 64, 128] on @L1 : memref<1x64x128xf32>
          %24 = loom.alloc [1, 64, 128] on @L1 : memref<1x64x128xf32>
          %25 = arith.muli %arg6, %c524288 overflow<nsw> : index
          %26 = arith.muli %13, %c8192 : index
          %27 = arith.addi %25, %26 : index
          %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%27], sizes: [1, 64, 128], strides: [524288, 128, 1] : memref<32x4096x128xf32> to memref<1x64x128xf32, strided<[524288, 128, 1], offset: ?>>
          loom.copy %reinterpret_cast, %22 src_mem_space @DRAM dst_mem_space @L1, interconnect : [], broadcast : [1, 1] : memref<1x64x128xf32, strided<[524288, 128, 1], offset: ?>>, memref<1x64x128xf32>
          linalg.fill ins(%cst_1 : f32) outs(%23 : memref<1x64x128xf32>)
          linalg.fill ins(%cst_2 : f32) outs(%17 : memref<1x64xf32>)
          linalg.fill ins(%cst_3 : f32) outs(%18 : memref<1x64xf32>)
          scf.for %arg7 = %c0 to %c32 step %c1 {
            %28 = arith.muli %arg7, %c128 : index
            %29 = arith.addi %25, %28 : index
            %reinterpret_cast_5 = memref.reinterpret_cast %arg0 to offset: [%29], sizes: [1, 128, 128], strides: [524288, 4096, 1] : memref<32x128x4096xf32> to memref<1x128x128xf32, strided<[524288, 4096, 1], offset: ?>>
            loom.copy %reinterpret_cast_5, %16 src_mem_space @DRAM dst_mem_space @L1, interconnect : [@vertical_links], broadcast : [8, 1] : memref<1x128x128xf32, strided<[524288, 4096, 1], offset: ?>>, memref<1x128x128xf32>
            linalg.fill ins(%cst_1 : f32) outs(%15 : memref<1x64x128xf32>)
            linalg.batch_matmul ins(%22, %16 : memref<1x64x128xf32>, memref<1x128x128xf32>) outs(%15 : memref<1x64x128xf32>)
            linalg.fill ins(%cst_3 : f32) outs(%19 : memref<1x64xf32>)
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%15 : memref<1x64x128xf32>) outs(%19 : memref<1x64xf32>) {
            ^bb0(%in: f32, %out: f32):
              %32 = arith.maximumf %in, %out : f32
              linalg.yield %32 : f32
            }
            linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%18, %19 : memref<1x64xf32>, memref<1x64xf32>) outs(%19 : memref<1x64xf32>) {
            ^bb0(%in: f32, %in_7: f32, %out: f32):
              %32 = arith.truncf %cst_0 : f64 to f32
              %33 = arith.mulf %in_7, %32 : f32
              %34 = arith.cmpf ogt, %in, %33 : f32
              %35 = arith.select %34, %in, %33 : f32
              linalg.yield %35 : f32
            }
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%15, %19 : memref<1x64x128xf32>, memref<1x64xf32>) outs(%15 : memref<1x64x128xf32>) {
            ^bb0(%in: f32, %in_7: f32, %out: f32):
              %32 = arith.truncf %cst_0 : f64 to f32
              %33 = arith.mulf %in, %32 : f32
              %34 = arith.subf %33, %in_7 : f32
              %35 = math.powf %cst, %34 : f32
              linalg.yield %35 : f32
            }
            linalg.fill ins(%cst_1 : f32) outs(%20 : memref<1x64xf32>)
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%15 : memref<1x64x128xf32>) outs(%20 : memref<1x64xf32>) {
            ^bb0(%in: f32, %out: f32):
              %32 = arith.addf %in, %out : f32
              linalg.yield %32 : f32
            }
            linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%18, %19 : memref<1x64xf32>, memref<1x64xf32>) outs(%21 : memref<1x64xf32>) {
            ^bb0(%in: f32, %in_7: f32, %out: f32):
              %32 = arith.subf %in, %in_7 : f32
              %33 = math.powf %cst, %32 : f32
              linalg.yield %33 : f32
            }
            linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%17, %21, %20 : memref<1x64xf32>, memref<1x64xf32>, memref<1x64xf32>) outs(%17 : memref<1x64xf32>) {
            ^bb0(%in: f32, %in_7: f32, %in_8: f32, %out: f32):
              %32 = arith.mulf %in, %in_7 : f32
              %33 = arith.addf %32, %in_8 : f32
              linalg.yield %33 : f32
            }
            %30 = arith.muli %arg7, %c16384 : index
            %31 = arith.addi %25, %30 : index
            %reinterpret_cast_6 = memref.reinterpret_cast %arg1 to offset: [%31], sizes: [1, 128, 128], strides: [524288, 128, 1] : memref<32x4096x128xf32> to memref<1x128x128xf32, strided<[524288, 128, 1], offset: ?>>
            loom.copy %reinterpret_cast_6, %14 src_mem_space @DRAM dst_mem_space @L1, interconnect : [@vertical_links], broadcast : [8, 1] : memref<1x128x128xf32, strided<[524288, 128, 1], offset: ?>>, memref<1x128x128xf32>
            linalg.fill ins(%cst_1 : f32) outs(%24 : memref<1x64x128xf32>)
            linalg.batch_matmul ins(%15, %14 : memref<1x64x128xf32>, memref<1x128x128xf32>) outs(%24 : memref<1x64x128xf32>)
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%24, %23, %21 : memref<1x64x128xf32>, memref<1x64x128xf32>, memref<1x64xf32>) outs(%23 : memref<1x64x128xf32>) {
            ^bb0(%in: f32, %in_7: f32, %in_8: f32, %out: f32):
              %32 = arith.mulf %in_7, %in_8 : f32
              %33 = arith.addf %in, %32 : f32
              linalg.yield %33 : f32
            }
            linalg.copy ins(%19 : memref<1x64xf32>) outs(%18 : memref<1x64xf32>)
          }
          linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%23, %17 : memref<1x64x128xf32>, memref<1x64xf32>) outs(%22 : memref<1x64x128xf32>) {
          ^bb0(%in: f32, %in_5: f32, %out: f32):
            %28 = arith.divf %in, %in_5 : f32
            linalg.yield %28 : f32
          }
          %reinterpret_cast_4 = memref.reinterpret_cast %arg3 to offset: [%27], sizes: [1, 64, 128], strides: [524288, 128, 1] : memref<32x4096x128xf32> to memref<1x64x128xf32, strided<[524288, 128, 1], offset: ?>>
          loom.copy %22, %reinterpret_cast_4 src_mem_space @L1 dst_mem_space @DRAM, interconnect : [], broadcast : [1, 1] : memref<1x64x128xf32>, memref<1x64x128xf32, strided<[524288, 128, 1], offset: ?>>
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
          %15 = loom.alloc [1, 64, 128] on @L1 : memref<1x64x128xf32>
          %16 = loom.alloc [1, 128, 128] on @L1 : memref<1x128x128xf32>
          %17 = loom.alloc [1, 64] on @L1 : memref<1x64xf32>
          %18 = loom.alloc [1, 64] on @L1 : memref<1x64xf32>
          %19 = loom.alloc [1, 64] on @L1 : memref<1x64xf32>
          %20 = loom.alloc [1, 64] on @L1 : memref<1x64xf32>
          %21 = loom.alloc [1, 64] on @L1 : memref<1x64xf32>
          %22 = loom.alloc [1, 64, 128] on @L1 : memref<1x64x128xf32>
          %23 = loom.alloc [1, 64, 128] on @L1 : memref<1x64x128xf32>
          %24 = loom.alloc [1, 64, 128] on @L1 : memref<1x64x128xf32>
          %25 = arith.muli %arg6, %c524288 overflow<nsw> : index
          %26 = arith.muli %13, %c8192 : index
          %27 = arith.addi %25, %26 : index
          %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%27], sizes: [1, 64, 128], strides: [524288, 128, 1] : memref<32x4096x128xf32> to memref<1x64x128xf32, strided<[524288, 128, 1], offset: ?>>
          loom.copy %reinterpret_cast, %22 src_mem_space @DRAM dst_mem_space @L1, interconnect : [], broadcast : [1, 1] : memref<1x64x128xf32, strided<[524288, 128, 1], offset: ?>>, memref<1x64x128xf32>
          linalg.fill ins(%cst_1 : f32) outs(%23 : memref<1x64x128xf32>)
          linalg.fill ins(%cst_2 : f32) outs(%17 : memref<1x64xf32>)
          linalg.fill ins(%cst_3 : f32) outs(%18 : memref<1x64xf32>)
          scf.for %arg7 = %c0 to %c32 step %c1 {
            %28 = arith.muli %arg7, %c128 : index
            %29 = arith.addi %25, %28 : index
            %reinterpret_cast_5 = memref.reinterpret_cast %arg0 to offset: [%29], sizes: [1, 128, 128], strides: [524288, 4096, 1] : memref<32x128x4096xf32> to memref<1x128x128xf32, strided<[524288, 4096, 1], offset: ?>>
            loom.copy %reinterpret_cast_5, %16 src_mem_space @DRAM dst_mem_space @L1, interconnect : [], broadcast : [1, 1] : memref<1x128x128xf32, strided<[524288, 4096, 1], offset: ?>>, memref<1x128x128xf32>
            linalg.fill ins(%cst_1 : f32) outs(%15 : memref<1x64x128xf32>)
            linalg.batch_matmul ins(%22, %16 : memref<1x64x128xf32>, memref<1x128x128xf32>) outs(%15 : memref<1x64x128xf32>)
            linalg.fill ins(%cst_3 : f32) outs(%19 : memref<1x64xf32>)
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%15 : memref<1x64x128xf32>) outs(%19 : memref<1x64xf32>) {
            ^bb0(%in: f32, %out: f32):
              %32 = arith.maximumf %in, %out : f32
              linalg.yield %32 : f32
            }
            linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%18, %19 : memref<1x64xf32>, memref<1x64xf32>) outs(%19 : memref<1x64xf32>) {
            ^bb0(%in: f32, %in_7: f32, %out: f32):
              %32 = arith.truncf %cst_0 : f64 to f32
              %33 = arith.mulf %in_7, %32 : f32
              %34 = arith.cmpf ogt, %in, %33 : f32
              %35 = arith.select %34, %in, %33 : f32
              linalg.yield %35 : f32
            }
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%15, %19 : memref<1x64x128xf32>, memref<1x64xf32>) outs(%15 : memref<1x64x128xf32>) {
            ^bb0(%in: f32, %in_7: f32, %out: f32):
              %32 = arith.truncf %cst_0 : f64 to f32
              %33 = arith.mulf %in, %32 : f32
              %34 = arith.subf %33, %in_7 : f32
              %35 = math.powf %cst, %34 : f32
              linalg.yield %35 : f32
            }
            linalg.fill ins(%cst_1 : f32) outs(%20 : memref<1x64xf32>)
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%15 : memref<1x64x128xf32>) outs(%20 : memref<1x64xf32>) {
            ^bb0(%in: f32, %out: f32):
              %32 = arith.addf %in, %out : f32
              linalg.yield %32 : f32
            }
            linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%18, %19 : memref<1x64xf32>, memref<1x64xf32>) outs(%21 : memref<1x64xf32>) {
            ^bb0(%in: f32, %in_7: f32, %out: f32):
              %32 = arith.subf %in, %in_7 : f32
              %33 = math.powf %cst, %32 : f32
              linalg.yield %33 : f32
            }
            linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%17, %21, %20 : memref<1x64xf32>, memref<1x64xf32>, memref<1x64xf32>) outs(%17 : memref<1x64xf32>) {
            ^bb0(%in: f32, %in_7: f32, %in_8: f32, %out: f32):
              %32 = arith.mulf %in, %in_7 : f32
              %33 = arith.addf %32, %in_8 : f32
              linalg.yield %33 : f32
            }
            %30 = arith.muli %arg7, %c16384 : index
            %31 = arith.addi %25, %30 : index
            %reinterpret_cast_6 = memref.reinterpret_cast %arg1 to offset: [%31], sizes: [1, 128, 128], strides: [524288, 128, 1] : memref<32x4096x128xf32> to memref<1x128x128xf32, strided<[524288, 128, 1], offset: ?>>
            loom.copy %reinterpret_cast_6, %14 src_mem_space @DRAM dst_mem_space @L1, interconnect : [], broadcast : [1, 1] : memref<1x128x128xf32, strided<[524288, 128, 1], offset: ?>>, memref<1x128x128xf32>
            linalg.fill ins(%cst_1 : f32) outs(%24 : memref<1x64x128xf32>)
            linalg.batch_matmul ins(%15, %14 : memref<1x64x128xf32>, memref<1x128x128xf32>) outs(%24 : memref<1x64x128xf32>)
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%24, %23, %21 : memref<1x64x128xf32>, memref<1x64x128xf32>, memref<1x64xf32>) outs(%23 : memref<1x64x128xf32>) {
            ^bb0(%in: f32, %in_7: f32, %in_8: f32, %out: f32):
              %32 = arith.mulf %in_7, %in_8 : f32
              %33 = arith.addf %in, %32 : f32
              linalg.yield %33 : f32
            }
            linalg.copy ins(%19 : memref<1x64xf32>) outs(%18 : memref<1x64xf32>)
          }
          linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%23, %17 : memref<1x64x128xf32>, memref<1x64xf32>) outs(%22 : memref<1x64x128xf32>) {
          ^bb0(%in: f32, %in_5: f32, %out: f32):
            %28 = arith.divf %in, %in_5 : f32
            linalg.yield %28 : f32
          }
          %reinterpret_cast_4 = memref.reinterpret_cast %arg3 to offset: [%27], sizes: [1, 64, 128], strides: [524288, 128, 1] : memref<32x4096x128xf32> to memref<1x64x128xf32, strided<[524288, 128, 1], offset: ?>>
          loom.copy %22, %reinterpret_cast_4 src_mem_space @L1 dst_mem_space @DRAM, interconnect : [], broadcast : [1, 1] : memref<1x64x128xf32>, memref<1x64x128xf32, strided<[524288, 128, 1], offset: ?>>
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
          %15 = loom.alloc [1, 64, 128] on @L1 : memref<1x64x128xf32>
          %16 = loom.alloc [1, 128, 128] on @L1 : memref<1x128x128xf32>
          %17 = loom.alloc [1, 64] on @L1 : memref<1x64xf32>
          %18 = loom.alloc [1, 64] on @L1 : memref<1x64xf32>
          %19 = loom.alloc [1, 64] on @L1 : memref<1x64xf32>
          %20 = loom.alloc [1, 64] on @L1 : memref<1x64xf32>
          %21 = loom.alloc [1, 64] on @L1 : memref<1x64xf32>
          %22 = loom.alloc [1, 64, 128] on @L1 : memref<1x64x128xf32>
          %23 = loom.alloc [1, 64, 128] on @L1 : memref<1x64x128xf32>
          %24 = loom.alloc [1, 64, 128] on @L1 : memref<1x64x128xf32>
          %25 = arith.muli %arg6, %c524288 overflow<nsw> : index
          %26 = arith.muli %13, %c8192 : index
          %27 = arith.addi %25, %26 : index
          %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%27], sizes: [1, 64, 128], strides: [524288, 128, 1] : memref<32x4096x128xf32> to memref<1x64x128xf32, strided<[524288, 128, 1], offset: ?>>
          loom.copy %reinterpret_cast, %22 src_mem_space @DRAM dst_mem_space @L1, interconnect : [], broadcast : [1, 1] : memref<1x64x128xf32, strided<[524288, 128, 1], offset: ?>>, memref<1x64x128xf32>
          linalg.fill ins(%cst_1 : f32) outs(%23 : memref<1x64x128xf32>)
          linalg.fill ins(%cst_2 : f32) outs(%17 : memref<1x64xf32>)
          linalg.fill ins(%cst_3 : f32) outs(%18 : memref<1x64xf32>)
          scf.for %arg7 = %c0 to %c32 step %c1 {
            %28 = arith.muli %arg7, %c128 : index
            %29 = arith.addi %25, %28 : index
            %reinterpret_cast_5 = memref.reinterpret_cast %arg0 to offset: [%29], sizes: [1, 128, 128], strides: [524288, 4096, 1] : memref<32x128x4096xf32> to memref<1x128x128xf32, strided<[524288, 4096, 1], offset: ?>>
            loom.copy %reinterpret_cast_5, %16 src_mem_space @DRAM dst_mem_space @L1, interconnect : [], broadcast : [1, 1] : memref<1x128x128xf32, strided<[524288, 4096, 1], offset: ?>>, memref<1x128x128xf32>
            linalg.fill ins(%cst_1 : f32) outs(%15 : memref<1x64x128xf32>)
            linalg.batch_matmul ins(%22, %16 : memref<1x64x128xf32>, memref<1x128x128xf32>) outs(%15 : memref<1x64x128xf32>)
            linalg.fill ins(%cst_3 : f32) outs(%19 : memref<1x64xf32>)
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%15 : memref<1x64x128xf32>) outs(%19 : memref<1x64xf32>) {
            ^bb0(%in: f32, %out: f32):
              %32 = arith.maximumf %in, %out : f32
              linalg.yield %32 : f32
            }
            linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%18, %19 : memref<1x64xf32>, memref<1x64xf32>) outs(%19 : memref<1x64xf32>) {
            ^bb0(%in: f32, %in_7: f32, %out: f32):
              %32 = arith.truncf %cst_0 : f64 to f32
              %33 = arith.mulf %in_7, %32 : f32
              %34 = arith.cmpf ogt, %in, %33 : f32
              %35 = arith.select %34, %in, %33 : f32
              linalg.yield %35 : f32
            }
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%15, %19 : memref<1x64x128xf32>, memref<1x64xf32>) outs(%15 : memref<1x64x128xf32>) {
            ^bb0(%in: f32, %in_7: f32, %out: f32):
              %32 = arith.truncf %cst_0 : f64 to f32
              %33 = arith.mulf %in, %32 : f32
              %34 = arith.subf %33, %in_7 : f32
              %35 = math.powf %cst, %34 : f32
              linalg.yield %35 : f32
            }
            linalg.fill ins(%cst_1 : f32) outs(%20 : memref<1x64xf32>)
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%15 : memref<1x64x128xf32>) outs(%20 : memref<1x64xf32>) {
            ^bb0(%in: f32, %out: f32):
              %32 = arith.addf %in, %out : f32
              linalg.yield %32 : f32
            }
            linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%18, %19 : memref<1x64xf32>, memref<1x64xf32>) outs(%21 : memref<1x64xf32>) {
            ^bb0(%in: f32, %in_7: f32, %out: f32):
              %32 = arith.subf %in, %in_7 : f32
              %33 = math.powf %cst, %32 : f32
              linalg.yield %33 : f32
            }
            linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%17, %21, %20 : memref<1x64xf32>, memref<1x64xf32>, memref<1x64xf32>) outs(%17 : memref<1x64xf32>) {
            ^bb0(%in: f32, %in_7: f32, %in_8: f32, %out: f32):
              %32 = arith.mulf %in, %in_7 : f32
              %33 = arith.addf %32, %in_8 : f32
              linalg.yield %33 : f32
            }
            %30 = arith.muli %arg7, %c16384 : index
            %31 = arith.addi %25, %30 : index
            %reinterpret_cast_6 = memref.reinterpret_cast %arg1 to offset: [%31], sizes: [1, 128, 128], strides: [524288, 128, 1] : memref<32x4096x128xf32> to memref<1x128x128xf32, strided<[524288, 128, 1], offset: ?>>
            loom.copy %reinterpret_cast_6, %14 src_mem_space @DRAM dst_mem_space @L1, interconnect : [@horizontal_links, @vertical_links], broadcast : [8, 8] : memref<1x128x128xf32, strided<[524288, 128, 1], offset: ?>>, memref<1x128x128xf32>
            linalg.fill ins(%cst_1 : f32) outs(%24 : memref<1x64x128xf32>)
            linalg.batch_matmul ins(%15, %14 : memref<1x64x128xf32>, memref<1x128x128xf32>) outs(%24 : memref<1x64x128xf32>)
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%24, %23, %21 : memref<1x64x128xf32>, memref<1x64x128xf32>, memref<1x64xf32>) outs(%23 : memref<1x64x128xf32>) {
            ^bb0(%in: f32, %in_7: f32, %in_8: f32, %out: f32):
              %32 = arith.mulf %in_7, %in_8 : f32
              %33 = arith.addf %in, %32 : f32
              linalg.yield %33 : f32
            }
            linalg.copy ins(%19 : memref<1x64xf32>) outs(%18 : memref<1x64xf32>)
          }
          linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%23, %17 : memref<1x64x128xf32>, memref<1x64xf32>) outs(%22 : memref<1x64x128xf32>) {
          ^bb0(%in: f32, %in_5: f32, %out: f32):
            %28 = arith.divf %in, %in_5 : f32
            linalg.yield %28 : f32
          }
          %reinterpret_cast_4 = memref.reinterpret_cast %arg3 to offset: [%27], sizes: [1, 64, 128], strides: [524288, 128, 1] : memref<32x4096x128xf32> to memref<1x64x128xf32, strided<[524288, 128, 1], offset: ?>>
          loom.copy %22, %reinterpret_cast_4 src_mem_space @L1 dst_mem_space @DRAM, interconnect : [], broadcast : [1, 1] : memref<1x64x128xf32>, memref<1x64x128xf32, strided<[524288, 128, 1], offset: ?>>
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
          %15 = loom.alloc [1, 64, 128] on @L1 : memref<1x64x128xf32>
          %16 = loom.alloc [1, 128, 128] on @L1 : memref<1x128x128xf32>
          %17 = loom.alloc [1, 64] on @L1 : memref<1x64xf32>
          %18 = loom.alloc [1, 64] on @L1 : memref<1x64xf32>
          %19 = loom.alloc [1, 64] on @L1 : memref<1x64xf32>
          %20 = loom.alloc [1, 64] on @L1 : memref<1x64xf32>
          %21 = loom.alloc [1, 64] on @L1 : memref<1x64xf32>
          %22 = loom.alloc [1, 64, 128] on @L1 : memref<1x64x128xf32>
          %23 = loom.alloc [1, 64, 128] on @L1 : memref<1x64x128xf32>
          %24 = loom.alloc [1, 64, 128] on @L1 : memref<1x64x128xf32>
          %25 = arith.muli %arg6, %c524288 overflow<nsw> : index
          %26 = arith.muli %13, %c8192 : index
          %27 = arith.addi %25, %26 : index
          %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%27], sizes: [1, 64, 128], strides: [524288, 128, 1] : memref<32x4096x128xf32> to memref<1x64x128xf32, strided<[524288, 128, 1], offset: ?>>
          loom.copy %reinterpret_cast, %22 src_mem_space @DRAM dst_mem_space @L1, interconnect : [], broadcast : [1, 1] : memref<1x64x128xf32, strided<[524288, 128, 1], offset: ?>>, memref<1x64x128xf32>
          linalg.fill ins(%cst_1 : f32) outs(%23 : memref<1x64x128xf32>)
          linalg.fill ins(%cst_2 : f32) outs(%17 : memref<1x64xf32>)
          linalg.fill ins(%cst_3 : f32) outs(%18 : memref<1x64xf32>)
          scf.for %arg7 = %c0 to %c32 step %c1 {
            %28 = arith.muli %arg7, %c128 : index
            %29 = arith.addi %25, %28 : index
            %reinterpret_cast_5 = memref.reinterpret_cast %arg0 to offset: [%29], sizes: [1, 128, 128], strides: [524288, 4096, 1] : memref<32x128x4096xf32> to memref<1x128x128xf32, strided<[524288, 4096, 1], offset: ?>>
            loom.copy %reinterpret_cast_5, %16 src_mem_space @DRAM dst_mem_space @L1, interconnect : [], broadcast : [1, 1] : memref<1x128x128xf32, strided<[524288, 4096, 1], offset: ?>>, memref<1x128x128xf32>
            linalg.fill ins(%cst_1 : f32) outs(%15 : memref<1x64x128xf32>)
            linalg.batch_matmul ins(%22, %16 : memref<1x64x128xf32>, memref<1x128x128xf32>) outs(%15 : memref<1x64x128xf32>)
            linalg.fill ins(%cst_3 : f32) outs(%19 : memref<1x64xf32>)
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%15 : memref<1x64x128xf32>) outs(%19 : memref<1x64xf32>) {
            ^bb0(%in: f32, %out: f32):
              %32 = arith.maximumf %in, %out : f32
              linalg.yield %32 : f32
            }
            linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%18, %19 : memref<1x64xf32>, memref<1x64xf32>) outs(%19 : memref<1x64xf32>) {
            ^bb0(%in: f32, %in_7: f32, %out: f32):
              %32 = arith.truncf %cst_0 : f64 to f32
              %33 = arith.mulf %in_7, %32 : f32
              %34 = arith.cmpf ogt, %in, %33 : f32
              %35 = arith.select %34, %in, %33 : f32
              linalg.yield %35 : f32
            }
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%15, %19 : memref<1x64x128xf32>, memref<1x64xf32>) outs(%15 : memref<1x64x128xf32>) {
            ^bb0(%in: f32, %in_7: f32, %out: f32):
              %32 = arith.truncf %cst_0 : f64 to f32
              %33 = arith.mulf %in, %32 : f32
              %34 = arith.subf %33, %in_7 : f32
              %35 = math.powf %cst, %34 : f32
              linalg.yield %35 : f32
            }
            linalg.fill ins(%cst_1 : f32) outs(%20 : memref<1x64xf32>)
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%15 : memref<1x64x128xf32>) outs(%20 : memref<1x64xf32>) {
            ^bb0(%in: f32, %out: f32):
              %32 = arith.addf %in, %out : f32
              linalg.yield %32 : f32
            }
            linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%18, %19 : memref<1x64xf32>, memref<1x64xf32>) outs(%21 : memref<1x64xf32>) {
            ^bb0(%in: f32, %in_7: f32, %out: f32):
              %32 = arith.subf %in, %in_7 : f32
              %33 = math.powf %cst, %32 : f32
              linalg.yield %33 : f32
            }
            linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%17, %21, %20 : memref<1x64xf32>, memref<1x64xf32>, memref<1x64xf32>) outs(%17 : memref<1x64xf32>) {
            ^bb0(%in: f32, %in_7: f32, %in_8: f32, %out: f32):
              %32 = arith.mulf %in, %in_7 : f32
              %33 = arith.addf %32, %in_8 : f32
              linalg.yield %33 : f32
            }
            %30 = arith.muli %arg7, %c16384 : index
            %31 = arith.addi %25, %30 : index
            %reinterpret_cast_6 = memref.reinterpret_cast %arg1 to offset: [%31], sizes: [1, 128, 128], strides: [524288, 128, 1] : memref<32x4096x128xf32> to memref<1x128x128xf32, strided<[524288, 128, 1], offset: ?>>
            loom.copy %reinterpret_cast_6, %14 src_mem_space @DRAM dst_mem_space @L1, interconnect : [@horizontal_links], broadcast : [1, 8] : memref<1x128x128xf32, strided<[524288, 128, 1], offset: ?>>, memref<1x128x128xf32>
            linalg.fill ins(%cst_1 : f32) outs(%24 : memref<1x64x128xf32>)
            linalg.batch_matmul ins(%15, %14 : memref<1x64x128xf32>, memref<1x128x128xf32>) outs(%24 : memref<1x64x128xf32>)
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%24, %23, %21 : memref<1x64x128xf32>, memref<1x64x128xf32>, memref<1x64xf32>) outs(%23 : memref<1x64x128xf32>) {
            ^bb0(%in: f32, %in_7: f32, %in_8: f32, %out: f32):
              %32 = arith.mulf %in_7, %in_8 : f32
              %33 = arith.addf %in, %32 : f32
              linalg.yield %33 : f32
            }
            linalg.copy ins(%19 : memref<1x64xf32>) outs(%18 : memref<1x64xf32>)
          }
          linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%23, %17 : memref<1x64x128xf32>, memref<1x64xf32>) outs(%22 : memref<1x64x128xf32>) {
          ^bb0(%in: f32, %in_5: f32, %out: f32):
            %28 = arith.divf %in, %in_5 : f32
            linalg.yield %28 : f32
          }
          %reinterpret_cast_4 = memref.reinterpret_cast %arg3 to offset: [%27], sizes: [1, 64, 128], strides: [524288, 128, 1] : memref<32x4096x128xf32> to memref<1x64x128xf32, strided<[524288, 128, 1], offset: ?>>
          loom.copy %22, %reinterpret_cast_4 src_mem_space @L1 dst_mem_space @DRAM, interconnect : [], broadcast : [1, 1] : memref<1x64x128xf32>, memref<1x64x128xf32, strided<[524288, 128, 1], offset: ?>>
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
          %15 = loom.alloc [1, 64, 128] on @L1 : memref<1x64x128xf32>
          %16 = loom.alloc [1, 128, 128] on @L1 : memref<1x128x128xf32>
          %17 = loom.alloc [1, 64] on @L1 : memref<1x64xf32>
          %18 = loom.alloc [1, 64] on @L1 : memref<1x64xf32>
          %19 = loom.alloc [1, 64] on @L1 : memref<1x64xf32>
          %20 = loom.alloc [1, 64] on @L1 : memref<1x64xf32>
          %21 = loom.alloc [1, 64] on @L1 : memref<1x64xf32>
          %22 = loom.alloc [1, 64, 128] on @L1 : memref<1x64x128xf32>
          %23 = loom.alloc [1, 64, 128] on @L1 : memref<1x64x128xf32>
          %24 = loom.alloc [1, 64, 128] on @L1 : memref<1x64x128xf32>
          %25 = arith.muli %arg6, %c524288 overflow<nsw> : index
          %26 = arith.muli %13, %c8192 : index
          %27 = arith.addi %25, %26 : index
          %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%27], sizes: [1, 64, 128], strides: [524288, 128, 1] : memref<32x4096x128xf32> to memref<1x64x128xf32, strided<[524288, 128, 1], offset: ?>>
          loom.copy %reinterpret_cast, %22 src_mem_space @DRAM dst_mem_space @L1, interconnect : [], broadcast : [1, 1] : memref<1x64x128xf32, strided<[524288, 128, 1], offset: ?>>, memref<1x64x128xf32>
          linalg.fill ins(%cst_1 : f32) outs(%23 : memref<1x64x128xf32>)
          linalg.fill ins(%cst_2 : f32) outs(%17 : memref<1x64xf32>)
          linalg.fill ins(%cst_3 : f32) outs(%18 : memref<1x64xf32>)
          scf.for %arg7 = %c0 to %c32 step %c1 {
            %28 = arith.muli %arg7, %c128 : index
            %29 = arith.addi %25, %28 : index
            %reinterpret_cast_5 = memref.reinterpret_cast %arg0 to offset: [%29], sizes: [1, 128, 128], strides: [524288, 4096, 1] : memref<32x128x4096xf32> to memref<1x128x128xf32, strided<[524288, 4096, 1], offset: ?>>
            loom.copy %reinterpret_cast_5, %16 src_mem_space @DRAM dst_mem_space @L1, interconnect : [], broadcast : [1, 1] : memref<1x128x128xf32, strided<[524288, 4096, 1], offset: ?>>, memref<1x128x128xf32>
            linalg.fill ins(%cst_1 : f32) outs(%15 : memref<1x64x128xf32>)
            linalg.batch_matmul ins(%22, %16 : memref<1x64x128xf32>, memref<1x128x128xf32>) outs(%15 : memref<1x64x128xf32>)
            linalg.fill ins(%cst_3 : f32) outs(%19 : memref<1x64xf32>)
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%15 : memref<1x64x128xf32>) outs(%19 : memref<1x64xf32>) {
            ^bb0(%in: f32, %out: f32):
              %32 = arith.maximumf %in, %out : f32
              linalg.yield %32 : f32
            }
            linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%18, %19 : memref<1x64xf32>, memref<1x64xf32>) outs(%19 : memref<1x64xf32>) {
            ^bb0(%in: f32, %in_7: f32, %out: f32):
              %32 = arith.truncf %cst_0 : f64 to f32
              %33 = arith.mulf %in_7, %32 : f32
              %34 = arith.cmpf ogt, %in, %33 : f32
              %35 = arith.select %34, %in, %33 : f32
              linalg.yield %35 : f32
            }
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%15, %19 : memref<1x64x128xf32>, memref<1x64xf32>) outs(%15 : memref<1x64x128xf32>) {
            ^bb0(%in: f32, %in_7: f32, %out: f32):
              %32 = arith.truncf %cst_0 : f64 to f32
              %33 = arith.mulf %in, %32 : f32
              %34 = arith.subf %33, %in_7 : f32
              %35 = math.powf %cst, %34 : f32
              linalg.yield %35 : f32
            }
            linalg.fill ins(%cst_1 : f32) outs(%20 : memref<1x64xf32>)
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%15 : memref<1x64x128xf32>) outs(%20 : memref<1x64xf32>) {
            ^bb0(%in: f32, %out: f32):
              %32 = arith.addf %in, %out : f32
              linalg.yield %32 : f32
            }
            linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%18, %19 : memref<1x64xf32>, memref<1x64xf32>) outs(%21 : memref<1x64xf32>) {
            ^bb0(%in: f32, %in_7: f32, %out: f32):
              %32 = arith.subf %in, %in_7 : f32
              %33 = math.powf %cst, %32 : f32
              linalg.yield %33 : f32
            }
            linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%17, %21, %20 : memref<1x64xf32>, memref<1x64xf32>, memref<1x64xf32>) outs(%17 : memref<1x64xf32>) {
            ^bb0(%in: f32, %in_7: f32, %in_8: f32, %out: f32):
              %32 = arith.mulf %in, %in_7 : f32
              %33 = arith.addf %32, %in_8 : f32
              linalg.yield %33 : f32
            }
            %30 = arith.muli %arg7, %c16384 : index
            %31 = arith.addi %25, %30 : index
            %reinterpret_cast_6 = memref.reinterpret_cast %arg1 to offset: [%31], sizes: [1, 128, 128], strides: [524288, 128, 1] : memref<32x4096x128xf32> to memref<1x128x128xf32, strided<[524288, 128, 1], offset: ?>>
            loom.copy %reinterpret_cast_6, %14 src_mem_space @DRAM dst_mem_space @L1, interconnect : [@vertical_links], broadcast : [8, 1] : memref<1x128x128xf32, strided<[524288, 128, 1], offset: ?>>, memref<1x128x128xf32>
            linalg.fill ins(%cst_1 : f32) outs(%24 : memref<1x64x128xf32>)
            linalg.batch_matmul ins(%15, %14 : memref<1x64x128xf32>, memref<1x128x128xf32>) outs(%24 : memref<1x64x128xf32>)
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%24, %23, %21 : memref<1x64x128xf32>, memref<1x64x128xf32>, memref<1x64xf32>) outs(%23 : memref<1x64x128xf32>) {
            ^bb0(%in: f32, %in_7: f32, %in_8: f32, %out: f32):
              %32 = arith.mulf %in_7, %in_8 : f32
              %33 = arith.addf %in, %32 : f32
              linalg.yield %33 : f32
            }
            linalg.copy ins(%19 : memref<1x64xf32>) outs(%18 : memref<1x64xf32>)
          }
          linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%23, %17 : memref<1x64x128xf32>, memref<1x64xf32>) outs(%22 : memref<1x64x128xf32>) {
          ^bb0(%in: f32, %in_5: f32, %out: f32):
            %28 = arith.divf %in, %in_5 : f32
            linalg.yield %28 : f32
          }
          %reinterpret_cast_4 = memref.reinterpret_cast %arg3 to offset: [%27], sizes: [1, 64, 128], strides: [524288, 128, 1] : memref<32x4096x128xf32> to memref<1x64x128xf32, strided<[524288, 128, 1], offset: ?>>
          loom.copy %22, %reinterpret_cast_4 src_mem_space @L1 dst_mem_space @DRAM, interconnect : [], broadcast : [1, 1] : memref<1x64x128xf32>, memref<1x64x128xf32, strided<[524288, 128, 1], offset: ?>>
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
          %15 = loom.alloc [1, 64, 128] on @L1 : memref<1x64x128xf32>
          %16 = loom.alloc [1, 128, 128] on @L1 : memref<1x128x128xf32>
          %17 = loom.alloc [1, 64] on @L1 : memref<1x64xf32>
          %18 = loom.alloc [1, 64] on @L1 : memref<1x64xf32>
          %19 = loom.alloc [1, 64] on @L1 : memref<1x64xf32>
          %20 = loom.alloc [1, 64] on @L1 : memref<1x64xf32>
          %21 = loom.alloc [1, 64] on @L1 : memref<1x64xf32>
          %22 = loom.alloc [1, 64, 128] on @L1 : memref<1x64x128xf32>
          %23 = loom.alloc [1, 64, 128] on @L1 : memref<1x64x128xf32>
          %24 = loom.alloc [1, 64, 128] on @L1 : memref<1x64x128xf32>
          %25 = arith.muli %arg6, %c524288 overflow<nsw> : index
          %26 = arith.muli %13, %c8192 : index
          %27 = arith.addi %25, %26 : index
          %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%27], sizes: [1, 64, 128], strides: [524288, 128, 1] : memref<32x4096x128xf32> to memref<1x64x128xf32, strided<[524288, 128, 1], offset: ?>>
          loom.copy %reinterpret_cast, %22 src_mem_space @DRAM dst_mem_space @L1, interconnect : [], broadcast : [1, 1] : memref<1x64x128xf32, strided<[524288, 128, 1], offset: ?>>, memref<1x64x128xf32>
          linalg.fill ins(%cst_1 : f32) outs(%23 : memref<1x64x128xf32>)
          linalg.fill ins(%cst_2 : f32) outs(%17 : memref<1x64xf32>)
          linalg.fill ins(%cst_3 : f32) outs(%18 : memref<1x64xf32>)
          scf.for %arg7 = %c0 to %c32 step %c1 {
            %28 = arith.muli %arg7, %c128 : index
            %29 = arith.addi %25, %28 : index
            %reinterpret_cast_5 = memref.reinterpret_cast %arg0 to offset: [%29], sizes: [1, 128, 128], strides: [524288, 4096, 1] : memref<32x128x4096xf32> to memref<1x128x128xf32, strided<[524288, 4096, 1], offset: ?>>
            loom.copy %reinterpret_cast_5, %16 src_mem_space @DRAM dst_mem_space @L1, interconnect : [@horizontal_links, @vertical_links], broadcast : [8, 8] : memref<1x128x128xf32, strided<[524288, 4096, 1], offset: ?>>, memref<1x128x128xf32>
            linalg.fill ins(%cst_1 : f32) outs(%15 : memref<1x64x128xf32>)
            linalg.batch_matmul ins(%22, %16 : memref<1x64x128xf32>, memref<1x128x128xf32>) outs(%15 : memref<1x64x128xf32>)
            linalg.fill ins(%cst_3 : f32) outs(%19 : memref<1x64xf32>)
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%15 : memref<1x64x128xf32>) outs(%19 : memref<1x64xf32>) {
            ^bb0(%in: f32, %out: f32):
              %32 = arith.maximumf %in, %out : f32
              linalg.yield %32 : f32
            }
            linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%18, %19 : memref<1x64xf32>, memref<1x64xf32>) outs(%19 : memref<1x64xf32>) {
            ^bb0(%in: f32, %in_7: f32, %out: f32):
              %32 = arith.truncf %cst_0 : f64 to f32
              %33 = arith.mulf %in_7, %32 : f32
              %34 = arith.cmpf ogt, %in, %33 : f32
              %35 = arith.select %34, %in, %33 : f32
              linalg.yield %35 : f32
            }
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%15, %19 : memref<1x64x128xf32>, memref<1x64xf32>) outs(%15 : memref<1x64x128xf32>) {
            ^bb0(%in: f32, %in_7: f32, %out: f32):
              %32 = arith.truncf %cst_0 : f64 to f32
              %33 = arith.mulf %in, %32 : f32
              %34 = arith.subf %33, %in_7 : f32
              %35 = math.powf %cst, %34 : f32
              linalg.yield %35 : f32
            }
            linalg.fill ins(%cst_1 : f32) outs(%20 : memref<1x64xf32>)
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%15 : memref<1x64x128xf32>) outs(%20 : memref<1x64xf32>) {
            ^bb0(%in: f32, %out: f32):
              %32 = arith.addf %in, %out : f32
              linalg.yield %32 : f32
            }
            linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%18, %19 : memref<1x64xf32>, memref<1x64xf32>) outs(%21 : memref<1x64xf32>) {
            ^bb0(%in: f32, %in_7: f32, %out: f32):
              %32 = arith.subf %in, %in_7 : f32
              %33 = math.powf %cst, %32 : f32
              linalg.yield %33 : f32
            }
            linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%17, %21, %20 : memref<1x64xf32>, memref<1x64xf32>, memref<1x64xf32>) outs(%17 : memref<1x64xf32>) {
            ^bb0(%in: f32, %in_7: f32, %in_8: f32, %out: f32):
              %32 = arith.mulf %in, %in_7 : f32
              %33 = arith.addf %32, %in_8 : f32
              linalg.yield %33 : f32
            }
            %30 = arith.muli %arg7, %c16384 : index
            %31 = arith.addi %25, %30 : index
            %reinterpret_cast_6 = memref.reinterpret_cast %arg1 to offset: [%31], sizes: [1, 128, 128], strides: [524288, 128, 1] : memref<32x4096x128xf32> to memref<1x128x128xf32, strided<[524288, 128, 1], offset: ?>>
            loom.copy %reinterpret_cast_6, %14 src_mem_space @DRAM dst_mem_space @L1, interconnect : [], broadcast : [1, 1] : memref<1x128x128xf32, strided<[524288, 128, 1], offset: ?>>, memref<1x128x128xf32>
            linalg.fill ins(%cst_1 : f32) outs(%24 : memref<1x64x128xf32>)
            linalg.batch_matmul ins(%15, %14 : memref<1x64x128xf32>, memref<1x128x128xf32>) outs(%24 : memref<1x64x128xf32>)
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%24, %23, %21 : memref<1x64x128xf32>, memref<1x64x128xf32>, memref<1x64xf32>) outs(%23 : memref<1x64x128xf32>) {
            ^bb0(%in: f32, %in_7: f32, %in_8: f32, %out: f32):
              %32 = arith.mulf %in_7, %in_8 : f32
              %33 = arith.addf %in, %32 : f32
              linalg.yield %33 : f32
            }
            linalg.copy ins(%19 : memref<1x64xf32>) outs(%18 : memref<1x64xf32>)
          }
          linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%23, %17 : memref<1x64x128xf32>, memref<1x64xf32>) outs(%22 : memref<1x64x128xf32>) {
          ^bb0(%in: f32, %in_5: f32, %out: f32):
            %28 = arith.divf %in, %in_5 : f32
            linalg.yield %28 : f32
          }
          %reinterpret_cast_4 = memref.reinterpret_cast %arg3 to offset: [%27], sizes: [1, 64, 128], strides: [524288, 128, 1] : memref<32x4096x128xf32> to memref<1x64x128xf32, strided<[524288, 128, 1], offset: ?>>
          loom.copy %22, %reinterpret_cast_4 src_mem_space @L1 dst_mem_space @DRAM, interconnect : [], broadcast : [1, 1] : memref<1x64x128xf32>, memref<1x64x128xf32, strided<[524288, 128, 1], offset: ?>>
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
          %15 = loom.alloc [1, 64, 128] on @L1 : memref<1x64x128xf32>
          %16 = loom.alloc [1, 128, 128] on @L1 : memref<1x128x128xf32>
          %17 = loom.alloc [1, 64] on @L1 : memref<1x64xf32>
          %18 = loom.alloc [1, 64] on @L1 : memref<1x64xf32>
          %19 = loom.alloc [1, 64] on @L1 : memref<1x64xf32>
          %20 = loom.alloc [1, 64] on @L1 : memref<1x64xf32>
          %21 = loom.alloc [1, 64] on @L1 : memref<1x64xf32>
          %22 = loom.alloc [1, 64, 128] on @L1 : memref<1x64x128xf32>
          %23 = loom.alloc [1, 64, 128] on @L1 : memref<1x64x128xf32>
          %24 = loom.alloc [1, 64, 128] on @L1 : memref<1x64x128xf32>
          %25 = arith.muli %arg6, %c524288 overflow<nsw> : index
          %26 = arith.muli %13, %c8192 : index
          %27 = arith.addi %25, %26 : index
          %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%27], sizes: [1, 64, 128], strides: [524288, 128, 1] : memref<32x4096x128xf32> to memref<1x64x128xf32, strided<[524288, 128, 1], offset: ?>>
          loom.copy %reinterpret_cast, %22 src_mem_space @DRAM dst_mem_space @L1, interconnect : [], broadcast : [1, 1] : memref<1x64x128xf32, strided<[524288, 128, 1], offset: ?>>, memref<1x64x128xf32>
          linalg.fill ins(%cst_1 : f32) outs(%23 : memref<1x64x128xf32>)
          linalg.fill ins(%cst_2 : f32) outs(%17 : memref<1x64xf32>)
          linalg.fill ins(%cst_3 : f32) outs(%18 : memref<1x64xf32>)
          scf.for %arg7 = %c0 to %c32 step %c1 {
            %28 = arith.muli %arg7, %c128 : index
            %29 = arith.addi %25, %28 : index
            %reinterpret_cast_5 = memref.reinterpret_cast %arg0 to offset: [%29], sizes: [1, 128, 128], strides: [524288, 4096, 1] : memref<32x128x4096xf32> to memref<1x128x128xf32, strided<[524288, 4096, 1], offset: ?>>
            loom.copy %reinterpret_cast_5, %16 src_mem_space @DRAM dst_mem_space @L1, interconnect : [@horizontal_links, @vertical_links], broadcast : [8, 8] : memref<1x128x128xf32, strided<[524288, 4096, 1], offset: ?>>, memref<1x128x128xf32>
            linalg.fill ins(%cst_1 : f32) outs(%15 : memref<1x64x128xf32>)
            linalg.batch_matmul ins(%22, %16 : memref<1x64x128xf32>, memref<1x128x128xf32>) outs(%15 : memref<1x64x128xf32>)
            linalg.fill ins(%cst_3 : f32) outs(%19 : memref<1x64xf32>)
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%15 : memref<1x64x128xf32>) outs(%19 : memref<1x64xf32>) {
            ^bb0(%in: f32, %out: f32):
              %32 = arith.maximumf %in, %out : f32
              linalg.yield %32 : f32
            }
            linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%18, %19 : memref<1x64xf32>, memref<1x64xf32>) outs(%19 : memref<1x64xf32>) {
            ^bb0(%in: f32, %in_7: f32, %out: f32):
              %32 = arith.truncf %cst_0 : f64 to f32
              %33 = arith.mulf %in_7, %32 : f32
              %34 = arith.cmpf ogt, %in, %33 : f32
              %35 = arith.select %34, %in, %33 : f32
              linalg.yield %35 : f32
            }
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%15, %19 : memref<1x64x128xf32>, memref<1x64xf32>) outs(%15 : memref<1x64x128xf32>) {
            ^bb0(%in: f32, %in_7: f32, %out: f32):
              %32 = arith.truncf %cst_0 : f64 to f32
              %33 = arith.mulf %in, %32 : f32
              %34 = arith.subf %33, %in_7 : f32
              %35 = math.powf %cst, %34 : f32
              linalg.yield %35 : f32
            }
            linalg.fill ins(%cst_1 : f32) outs(%20 : memref<1x64xf32>)
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%15 : memref<1x64x128xf32>) outs(%20 : memref<1x64xf32>) {
            ^bb0(%in: f32, %out: f32):
              %32 = arith.addf %in, %out : f32
              linalg.yield %32 : f32
            }
            linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%18, %19 : memref<1x64xf32>, memref<1x64xf32>) outs(%21 : memref<1x64xf32>) {
            ^bb0(%in: f32, %in_7: f32, %out: f32):
              %32 = arith.subf %in, %in_7 : f32
              %33 = math.powf %cst, %32 : f32
              linalg.yield %33 : f32
            }
            linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%17, %21, %20 : memref<1x64xf32>, memref<1x64xf32>, memref<1x64xf32>) outs(%17 : memref<1x64xf32>) {
            ^bb0(%in: f32, %in_7: f32, %in_8: f32, %out: f32):
              %32 = arith.mulf %in, %in_7 : f32
              %33 = arith.addf %32, %in_8 : f32
              linalg.yield %33 : f32
            }
            %30 = arith.muli %arg7, %c16384 : index
            %31 = arith.addi %25, %30 : index
            %reinterpret_cast_6 = memref.reinterpret_cast %arg1 to offset: [%31], sizes: [1, 128, 128], strides: [524288, 128, 1] : memref<32x4096x128xf32> to memref<1x128x128xf32, strided<[524288, 128, 1], offset: ?>>
            loom.copy %reinterpret_cast_6, %14 src_mem_space @DRAM dst_mem_space @L1, interconnect : [@horizontal_links, @vertical_links], broadcast : [8, 8] : memref<1x128x128xf32, strided<[524288, 128, 1], offset: ?>>, memref<1x128x128xf32>
            linalg.fill ins(%cst_1 : f32) outs(%24 : memref<1x64x128xf32>)
            linalg.batch_matmul ins(%15, %14 : memref<1x64x128xf32>, memref<1x128x128xf32>) outs(%24 : memref<1x64x128xf32>)
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%24, %23, %21 : memref<1x64x128xf32>, memref<1x64x128xf32>, memref<1x64xf32>) outs(%23 : memref<1x64x128xf32>) {
            ^bb0(%in: f32, %in_7: f32, %in_8: f32, %out: f32):
              %32 = arith.mulf %in_7, %in_8 : f32
              %33 = arith.addf %in, %32 : f32
              linalg.yield %33 : f32
            }
            linalg.copy ins(%19 : memref<1x64xf32>) outs(%18 : memref<1x64xf32>)
          }
          linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%23, %17 : memref<1x64x128xf32>, memref<1x64xf32>) outs(%22 : memref<1x64x128xf32>) {
          ^bb0(%in: f32, %in_5: f32, %out: f32):
            %28 = arith.divf %in, %in_5 : f32
            linalg.yield %28 : f32
          }
          %reinterpret_cast_4 = memref.reinterpret_cast %arg3 to offset: [%27], sizes: [1, 64, 128], strides: [524288, 128, 1] : memref<32x4096x128xf32> to memref<1x64x128xf32, strided<[524288, 128, 1], offset: ?>>
          loom.copy %22, %reinterpret_cast_4 src_mem_space @L1 dst_mem_space @DRAM, interconnect : [], broadcast : [1, 1] : memref<1x64x128xf32>, memref<1x64x128xf32, strided<[524288, 128, 1], offset: ?>>
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
          %15 = loom.alloc [1, 64, 128] on @L1 : memref<1x64x128xf32>
          %16 = loom.alloc [1, 128, 128] on @L1 : memref<1x128x128xf32>
          %17 = loom.alloc [1, 64] on @L1 : memref<1x64xf32>
          %18 = loom.alloc [1, 64] on @L1 : memref<1x64xf32>
          %19 = loom.alloc [1, 64] on @L1 : memref<1x64xf32>
          %20 = loom.alloc [1, 64] on @L1 : memref<1x64xf32>
          %21 = loom.alloc [1, 64] on @L1 : memref<1x64xf32>
          %22 = loom.alloc [1, 64, 128] on @L1 : memref<1x64x128xf32>
          %23 = loom.alloc [1, 64, 128] on @L1 : memref<1x64x128xf32>
          %24 = loom.alloc [1, 64, 128] on @L1 : memref<1x64x128xf32>
          %25 = arith.muli %arg6, %c524288 overflow<nsw> : index
          %26 = arith.muli %13, %c8192 : index
          %27 = arith.addi %25, %26 : index
          %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%27], sizes: [1, 64, 128], strides: [524288, 128, 1] : memref<32x4096x128xf32> to memref<1x64x128xf32, strided<[524288, 128, 1], offset: ?>>
          loom.copy %reinterpret_cast, %22 src_mem_space @DRAM dst_mem_space @L1, interconnect : [], broadcast : [1, 1] : memref<1x64x128xf32, strided<[524288, 128, 1], offset: ?>>, memref<1x64x128xf32>
          linalg.fill ins(%cst_1 : f32) outs(%23 : memref<1x64x128xf32>)
          linalg.fill ins(%cst_2 : f32) outs(%17 : memref<1x64xf32>)
          linalg.fill ins(%cst_3 : f32) outs(%18 : memref<1x64xf32>)
          scf.for %arg7 = %c0 to %c32 step %c1 {
            %28 = arith.muli %arg7, %c128 : index
            %29 = arith.addi %25, %28 : index
            %reinterpret_cast_5 = memref.reinterpret_cast %arg0 to offset: [%29], sizes: [1, 128, 128], strides: [524288, 4096, 1] : memref<32x128x4096xf32> to memref<1x128x128xf32, strided<[524288, 4096, 1], offset: ?>>
            loom.copy %reinterpret_cast_5, %16 src_mem_space @DRAM dst_mem_space @L1, interconnect : [@horizontal_links, @vertical_links], broadcast : [8, 8] : memref<1x128x128xf32, strided<[524288, 4096, 1], offset: ?>>, memref<1x128x128xf32>
            linalg.fill ins(%cst_1 : f32) outs(%15 : memref<1x64x128xf32>)
            linalg.batch_matmul ins(%22, %16 : memref<1x64x128xf32>, memref<1x128x128xf32>) outs(%15 : memref<1x64x128xf32>)
            linalg.fill ins(%cst_3 : f32) outs(%19 : memref<1x64xf32>)
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%15 : memref<1x64x128xf32>) outs(%19 : memref<1x64xf32>) {
            ^bb0(%in: f32, %out: f32):
              %32 = arith.maximumf %in, %out : f32
              linalg.yield %32 : f32
            }
            linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%18, %19 : memref<1x64xf32>, memref<1x64xf32>) outs(%19 : memref<1x64xf32>) {
            ^bb0(%in: f32, %in_7: f32, %out: f32):
              %32 = arith.truncf %cst_0 : f64 to f32
              %33 = arith.mulf %in_7, %32 : f32
              %34 = arith.cmpf ogt, %in, %33 : f32
              %35 = arith.select %34, %in, %33 : f32
              linalg.yield %35 : f32
            }
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%15, %19 : memref<1x64x128xf32>, memref<1x64xf32>) outs(%15 : memref<1x64x128xf32>) {
            ^bb0(%in: f32, %in_7: f32, %out: f32):
              %32 = arith.truncf %cst_0 : f64 to f32
              %33 = arith.mulf %in, %32 : f32
              %34 = arith.subf %33, %in_7 : f32
              %35 = math.powf %cst, %34 : f32
              linalg.yield %35 : f32
            }
            linalg.fill ins(%cst_1 : f32) outs(%20 : memref<1x64xf32>)
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%15 : memref<1x64x128xf32>) outs(%20 : memref<1x64xf32>) {
            ^bb0(%in: f32, %out: f32):
              %32 = arith.addf %in, %out : f32
              linalg.yield %32 : f32
            }
            linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%18, %19 : memref<1x64xf32>, memref<1x64xf32>) outs(%21 : memref<1x64xf32>) {
            ^bb0(%in: f32, %in_7: f32, %out: f32):
              %32 = arith.subf %in, %in_7 : f32
              %33 = math.powf %cst, %32 : f32
              linalg.yield %33 : f32
            }
            linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%17, %21, %20 : memref<1x64xf32>, memref<1x64xf32>, memref<1x64xf32>) outs(%17 : memref<1x64xf32>) {
            ^bb0(%in: f32, %in_7: f32, %in_8: f32, %out: f32):
              %32 = arith.mulf %in, %in_7 : f32
              %33 = arith.addf %32, %in_8 : f32
              linalg.yield %33 : f32
            }
            %30 = arith.muli %arg7, %c16384 : index
            %31 = arith.addi %25, %30 : index
            %reinterpret_cast_6 = memref.reinterpret_cast %arg1 to offset: [%31], sizes: [1, 128, 128], strides: [524288, 128, 1] : memref<32x4096x128xf32> to memref<1x128x128xf32, strided<[524288, 128, 1], offset: ?>>
            loom.copy %reinterpret_cast_6, %14 src_mem_space @DRAM dst_mem_space @L1, interconnect : [@horizontal_links], broadcast : [1, 8] : memref<1x128x128xf32, strided<[524288, 128, 1], offset: ?>>, memref<1x128x128xf32>
            linalg.fill ins(%cst_1 : f32) outs(%24 : memref<1x64x128xf32>)
            linalg.batch_matmul ins(%15, %14 : memref<1x64x128xf32>, memref<1x128x128xf32>) outs(%24 : memref<1x64x128xf32>)
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%24, %23, %21 : memref<1x64x128xf32>, memref<1x64x128xf32>, memref<1x64xf32>) outs(%23 : memref<1x64x128xf32>) {
            ^bb0(%in: f32, %in_7: f32, %in_8: f32, %out: f32):
              %32 = arith.mulf %in_7, %in_8 : f32
              %33 = arith.addf %in, %32 : f32
              linalg.yield %33 : f32
            }
            linalg.copy ins(%19 : memref<1x64xf32>) outs(%18 : memref<1x64xf32>)
          }
          linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%23, %17 : memref<1x64x128xf32>, memref<1x64xf32>) outs(%22 : memref<1x64x128xf32>) {
          ^bb0(%in: f32, %in_5: f32, %out: f32):
            %28 = arith.divf %in, %in_5 : f32
            linalg.yield %28 : f32
          }
          %reinterpret_cast_4 = memref.reinterpret_cast %arg3 to offset: [%27], sizes: [1, 64, 128], strides: [524288, 128, 1] : memref<32x4096x128xf32> to memref<1x64x128xf32, strided<[524288, 128, 1], offset: ?>>
          loom.copy %22, %reinterpret_cast_4 src_mem_space @L1 dst_mem_space @DRAM, interconnect : [], broadcast : [1, 1] : memref<1x64x128xf32>, memref<1x64x128xf32, strided<[524288, 128, 1], offset: ?>>
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
          %15 = loom.alloc [1, 64, 128] on @L1 : memref<1x64x128xf32>
          %16 = loom.alloc [1, 128, 128] on @L1 : memref<1x128x128xf32>
          %17 = loom.alloc [1, 64] on @L1 : memref<1x64xf32>
          %18 = loom.alloc [1, 64] on @L1 : memref<1x64xf32>
          %19 = loom.alloc [1, 64] on @L1 : memref<1x64xf32>
          %20 = loom.alloc [1, 64] on @L1 : memref<1x64xf32>
          %21 = loom.alloc [1, 64] on @L1 : memref<1x64xf32>
          %22 = loom.alloc [1, 64, 128] on @L1 : memref<1x64x128xf32>
          %23 = loom.alloc [1, 64, 128] on @L1 : memref<1x64x128xf32>
          %24 = loom.alloc [1, 64, 128] on @L1 : memref<1x64x128xf32>
          %25 = arith.muli %arg6, %c524288 overflow<nsw> : index
          %26 = arith.muli %13, %c8192 : index
          %27 = arith.addi %25, %26 : index
          %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%27], sizes: [1, 64, 128], strides: [524288, 128, 1] : memref<32x4096x128xf32> to memref<1x64x128xf32, strided<[524288, 128, 1], offset: ?>>
          loom.copy %reinterpret_cast, %22 src_mem_space @DRAM dst_mem_space @L1, interconnect : [], broadcast : [1, 1] : memref<1x64x128xf32, strided<[524288, 128, 1], offset: ?>>, memref<1x64x128xf32>
          linalg.fill ins(%cst_1 : f32) outs(%23 : memref<1x64x128xf32>)
          linalg.fill ins(%cst_2 : f32) outs(%17 : memref<1x64xf32>)
          linalg.fill ins(%cst_3 : f32) outs(%18 : memref<1x64xf32>)
          scf.for %arg7 = %c0 to %c32 step %c1 {
            %28 = arith.muli %arg7, %c128 : index
            %29 = arith.addi %25, %28 : index
            %reinterpret_cast_5 = memref.reinterpret_cast %arg0 to offset: [%29], sizes: [1, 128, 128], strides: [524288, 4096, 1] : memref<32x128x4096xf32> to memref<1x128x128xf32, strided<[524288, 4096, 1], offset: ?>>
            loom.copy %reinterpret_cast_5, %16 src_mem_space @DRAM dst_mem_space @L1, interconnect : [@horizontal_links, @vertical_links], broadcast : [8, 8] : memref<1x128x128xf32, strided<[524288, 4096, 1], offset: ?>>, memref<1x128x128xf32>
            linalg.fill ins(%cst_1 : f32) outs(%15 : memref<1x64x128xf32>)
            linalg.batch_matmul ins(%22, %16 : memref<1x64x128xf32>, memref<1x128x128xf32>) outs(%15 : memref<1x64x128xf32>)
            linalg.fill ins(%cst_3 : f32) outs(%19 : memref<1x64xf32>)
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%15 : memref<1x64x128xf32>) outs(%19 : memref<1x64xf32>) {
            ^bb0(%in: f32, %out: f32):
              %32 = arith.maximumf %in, %out : f32
              linalg.yield %32 : f32
            }
            linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%18, %19 : memref<1x64xf32>, memref<1x64xf32>) outs(%19 : memref<1x64xf32>) {
            ^bb0(%in: f32, %in_7: f32, %out: f32):
              %32 = arith.truncf %cst_0 : f64 to f32
              %33 = arith.mulf %in_7, %32 : f32
              %34 = arith.cmpf ogt, %in, %33 : f32
              %35 = arith.select %34, %in, %33 : f32
              linalg.yield %35 : f32
            }
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%15, %19 : memref<1x64x128xf32>, memref<1x64xf32>) outs(%15 : memref<1x64x128xf32>) {
            ^bb0(%in: f32, %in_7: f32, %out: f32):
              %32 = arith.truncf %cst_0 : f64 to f32
              %33 = arith.mulf %in, %32 : f32
              %34 = arith.subf %33, %in_7 : f32
              %35 = math.powf %cst, %34 : f32
              linalg.yield %35 : f32
            }
            linalg.fill ins(%cst_1 : f32) outs(%20 : memref<1x64xf32>)
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%15 : memref<1x64x128xf32>) outs(%20 : memref<1x64xf32>) {
            ^bb0(%in: f32, %out: f32):
              %32 = arith.addf %in, %out : f32
              linalg.yield %32 : f32
            }
            linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%18, %19 : memref<1x64xf32>, memref<1x64xf32>) outs(%21 : memref<1x64xf32>) {
            ^bb0(%in: f32, %in_7: f32, %out: f32):
              %32 = arith.subf %in, %in_7 : f32
              %33 = math.powf %cst, %32 : f32
              linalg.yield %33 : f32
            }
            linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%17, %21, %20 : memref<1x64xf32>, memref<1x64xf32>, memref<1x64xf32>) outs(%17 : memref<1x64xf32>) {
            ^bb0(%in: f32, %in_7: f32, %in_8: f32, %out: f32):
              %32 = arith.mulf %in, %in_7 : f32
              %33 = arith.addf %32, %in_8 : f32
              linalg.yield %33 : f32
            }
            %30 = arith.muli %arg7, %c16384 : index
            %31 = arith.addi %25, %30 : index
            %reinterpret_cast_6 = memref.reinterpret_cast %arg1 to offset: [%31], sizes: [1, 128, 128], strides: [524288, 128, 1] : memref<32x4096x128xf32> to memref<1x128x128xf32, strided<[524288, 128, 1], offset: ?>>
            loom.copy %reinterpret_cast_6, %14 src_mem_space @DRAM dst_mem_space @L1, interconnect : [@vertical_links], broadcast : [8, 1] : memref<1x128x128xf32, strided<[524288, 128, 1], offset: ?>>, memref<1x128x128xf32>
            linalg.fill ins(%cst_1 : f32) outs(%24 : memref<1x64x128xf32>)
            linalg.batch_matmul ins(%15, %14 : memref<1x64x128xf32>, memref<1x128x128xf32>) outs(%24 : memref<1x64x128xf32>)
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%24, %23, %21 : memref<1x64x128xf32>, memref<1x64x128xf32>, memref<1x64xf32>) outs(%23 : memref<1x64x128xf32>) {
            ^bb0(%in: f32, %in_7: f32, %in_8: f32, %out: f32):
              %32 = arith.mulf %in_7, %in_8 : f32
              %33 = arith.addf %in, %32 : f32
              linalg.yield %33 : f32
            }
            linalg.copy ins(%19 : memref<1x64xf32>) outs(%18 : memref<1x64xf32>)
          }
          linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%23, %17 : memref<1x64x128xf32>, memref<1x64xf32>) outs(%22 : memref<1x64x128xf32>) {
          ^bb0(%in: f32, %in_5: f32, %out: f32):
            %28 = arith.divf %in, %in_5 : f32
            linalg.yield %28 : f32
          }
          %reinterpret_cast_4 = memref.reinterpret_cast %arg3 to offset: [%27], sizes: [1, 64, 128], strides: [524288, 128, 1] : memref<32x4096x128xf32> to memref<1x64x128xf32, strided<[524288, 128, 1], offset: ?>>
          loom.copy %22, %reinterpret_cast_4 src_mem_space @L1 dst_mem_space @DRAM, interconnect : [], broadcast : [1, 1] : memref<1x64x128xf32>, memref<1x64x128xf32, strided<[524288, 128, 1], offset: ?>>
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
          %15 = loom.alloc [1, 64, 128] on @L1 : memref<1x64x128xf32>
          %16 = loom.alloc [1, 128, 128] on @L1 : memref<1x128x128xf32>
          %17 = loom.alloc [1, 64] on @L1 : memref<1x64xf32>
          %18 = loom.alloc [1, 64] on @L1 : memref<1x64xf32>
          %19 = loom.alloc [1, 64] on @L1 : memref<1x64xf32>
          %20 = loom.alloc [1, 64] on @L1 : memref<1x64xf32>
          %21 = loom.alloc [1, 64] on @L1 : memref<1x64xf32>
          %22 = loom.alloc [1, 64, 128] on @L1 : memref<1x64x128xf32>
          %23 = loom.alloc [1, 64, 128] on @L1 : memref<1x64x128xf32>
          %24 = loom.alloc [1, 64, 128] on @L1 : memref<1x64x128xf32>
          %25 = arith.muli %arg6, %c524288 overflow<nsw> : index
          %26 = arith.muli %13, %c8192 : index
          %27 = arith.addi %25, %26 : index
          %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%27], sizes: [1, 64, 128], strides: [524288, 128, 1] : memref<32x4096x128xf32> to memref<1x64x128xf32, strided<[524288, 128, 1], offset: ?>>
          loom.copy %reinterpret_cast, %22 src_mem_space @DRAM dst_mem_space @L1, interconnect : [], broadcast : [1, 1] : memref<1x64x128xf32, strided<[524288, 128, 1], offset: ?>>, memref<1x64x128xf32>
          linalg.fill ins(%cst_1 : f32) outs(%23 : memref<1x64x128xf32>)
          linalg.fill ins(%cst_2 : f32) outs(%17 : memref<1x64xf32>)
          linalg.fill ins(%cst_3 : f32) outs(%18 : memref<1x64xf32>)
          scf.for %arg7 = %c0 to %c32 step %c1 {
            %28 = arith.muli %arg7, %c128 : index
            %29 = arith.addi %25, %28 : index
            %reinterpret_cast_5 = memref.reinterpret_cast %arg0 to offset: [%29], sizes: [1, 128, 128], strides: [524288, 4096, 1] : memref<32x128x4096xf32> to memref<1x128x128xf32, strided<[524288, 4096, 1], offset: ?>>
            loom.copy %reinterpret_cast_5, %16 src_mem_space @DRAM dst_mem_space @L1, interconnect : [@horizontal_links], broadcast : [1, 8] : memref<1x128x128xf32, strided<[524288, 4096, 1], offset: ?>>, memref<1x128x128xf32>
            linalg.fill ins(%cst_1 : f32) outs(%15 : memref<1x64x128xf32>)
            linalg.batch_matmul ins(%22, %16 : memref<1x64x128xf32>, memref<1x128x128xf32>) outs(%15 : memref<1x64x128xf32>)
            linalg.fill ins(%cst_3 : f32) outs(%19 : memref<1x64xf32>)
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%15 : memref<1x64x128xf32>) outs(%19 : memref<1x64xf32>) {
            ^bb0(%in: f32, %out: f32):
              %32 = arith.maximumf %in, %out : f32
              linalg.yield %32 : f32
            }
            linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%18, %19 : memref<1x64xf32>, memref<1x64xf32>) outs(%19 : memref<1x64xf32>) {
            ^bb0(%in: f32, %in_7: f32, %out: f32):
              %32 = arith.truncf %cst_0 : f64 to f32
              %33 = arith.mulf %in_7, %32 : f32
              %34 = arith.cmpf ogt, %in, %33 : f32
              %35 = arith.select %34, %in, %33 : f32
              linalg.yield %35 : f32
            }
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%15, %19 : memref<1x64x128xf32>, memref<1x64xf32>) outs(%15 : memref<1x64x128xf32>) {
            ^bb0(%in: f32, %in_7: f32, %out: f32):
              %32 = arith.truncf %cst_0 : f64 to f32
              %33 = arith.mulf %in, %32 : f32
              %34 = arith.subf %33, %in_7 : f32
              %35 = math.powf %cst, %34 : f32
              linalg.yield %35 : f32
            }
            linalg.fill ins(%cst_1 : f32) outs(%20 : memref<1x64xf32>)
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%15 : memref<1x64x128xf32>) outs(%20 : memref<1x64xf32>) {
            ^bb0(%in: f32, %out: f32):
              %32 = arith.addf %in, %out : f32
              linalg.yield %32 : f32
            }
            linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%18, %19 : memref<1x64xf32>, memref<1x64xf32>) outs(%21 : memref<1x64xf32>) {
            ^bb0(%in: f32, %in_7: f32, %out: f32):
              %32 = arith.subf %in, %in_7 : f32
              %33 = math.powf %cst, %32 : f32
              linalg.yield %33 : f32
            }
            linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%17, %21, %20 : memref<1x64xf32>, memref<1x64xf32>, memref<1x64xf32>) outs(%17 : memref<1x64xf32>) {
            ^bb0(%in: f32, %in_7: f32, %in_8: f32, %out: f32):
              %32 = arith.mulf %in, %in_7 : f32
              %33 = arith.addf %32, %in_8 : f32
              linalg.yield %33 : f32
            }
            %30 = arith.muli %arg7, %c16384 : index
            %31 = arith.addi %25, %30 : index
            %reinterpret_cast_6 = memref.reinterpret_cast %arg1 to offset: [%31], sizes: [1, 128, 128], strides: [524288, 128, 1] : memref<32x4096x128xf32> to memref<1x128x128xf32, strided<[524288, 128, 1], offset: ?>>
            loom.copy %reinterpret_cast_6, %14 src_mem_space @DRAM dst_mem_space @L1, interconnect : [], broadcast : [1, 1] : memref<1x128x128xf32, strided<[524288, 128, 1], offset: ?>>, memref<1x128x128xf32>
            linalg.fill ins(%cst_1 : f32) outs(%24 : memref<1x64x128xf32>)
            linalg.batch_matmul ins(%15, %14 : memref<1x64x128xf32>, memref<1x128x128xf32>) outs(%24 : memref<1x64x128xf32>)
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%24, %23, %21 : memref<1x64x128xf32>, memref<1x64x128xf32>, memref<1x64xf32>) outs(%23 : memref<1x64x128xf32>) {
            ^bb0(%in: f32, %in_7: f32, %in_8: f32, %out: f32):
              %32 = arith.mulf %in_7, %in_8 : f32
              %33 = arith.addf %in, %32 : f32
              linalg.yield %33 : f32
            }
            linalg.copy ins(%19 : memref<1x64xf32>) outs(%18 : memref<1x64xf32>)
          }
          linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%23, %17 : memref<1x64x128xf32>, memref<1x64xf32>) outs(%22 : memref<1x64x128xf32>) {
          ^bb0(%in: f32, %in_5: f32, %out: f32):
            %28 = arith.divf %in, %in_5 : f32
            linalg.yield %28 : f32
          }
          %reinterpret_cast_4 = memref.reinterpret_cast %arg3 to offset: [%27], sizes: [1, 64, 128], strides: [524288, 128, 1] : memref<32x4096x128xf32> to memref<1x64x128xf32, strided<[524288, 128, 1], offset: ?>>
          loom.copy %22, %reinterpret_cast_4 src_mem_space @L1 dst_mem_space @DRAM, interconnect : [], broadcast : [1, 1] : memref<1x64x128xf32>, memref<1x64x128xf32, strided<[524288, 128, 1], offset: ?>>
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
          %15 = loom.alloc [1, 64, 128] on @L1 : memref<1x64x128xf32>
          %16 = loom.alloc [1, 128, 128] on @L1 : memref<1x128x128xf32>
          %17 = loom.alloc [1, 64] on @L1 : memref<1x64xf32>
          %18 = loom.alloc [1, 64] on @L1 : memref<1x64xf32>
          %19 = loom.alloc [1, 64] on @L1 : memref<1x64xf32>
          %20 = loom.alloc [1, 64] on @L1 : memref<1x64xf32>
          %21 = loom.alloc [1, 64] on @L1 : memref<1x64xf32>
          %22 = loom.alloc [1, 64, 128] on @L1 : memref<1x64x128xf32>
          %23 = loom.alloc [1, 64, 128] on @L1 : memref<1x64x128xf32>
          %24 = loom.alloc [1, 64, 128] on @L1 : memref<1x64x128xf32>
          %25 = arith.muli %arg6, %c524288 overflow<nsw> : index
          %26 = arith.muli %13, %c8192 : index
          %27 = arith.addi %25, %26 : index
          %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%27], sizes: [1, 64, 128], strides: [524288, 128, 1] : memref<32x4096x128xf32> to memref<1x64x128xf32, strided<[524288, 128, 1], offset: ?>>
          loom.copy %reinterpret_cast, %22 src_mem_space @DRAM dst_mem_space @L1, interconnect : [], broadcast : [1, 1] : memref<1x64x128xf32, strided<[524288, 128, 1], offset: ?>>, memref<1x64x128xf32>
          linalg.fill ins(%cst_1 : f32) outs(%23 : memref<1x64x128xf32>)
          linalg.fill ins(%cst_2 : f32) outs(%17 : memref<1x64xf32>)
          linalg.fill ins(%cst_3 : f32) outs(%18 : memref<1x64xf32>)
          scf.for %arg7 = %c0 to %c32 step %c1 {
            %28 = arith.muli %arg7, %c128 : index
            %29 = arith.addi %25, %28 : index
            %reinterpret_cast_5 = memref.reinterpret_cast %arg0 to offset: [%29], sizes: [1, 128, 128], strides: [524288, 4096, 1] : memref<32x128x4096xf32> to memref<1x128x128xf32, strided<[524288, 4096, 1], offset: ?>>
            loom.copy %reinterpret_cast_5, %16 src_mem_space @DRAM dst_mem_space @L1, interconnect : [@horizontal_links], broadcast : [1, 8] : memref<1x128x128xf32, strided<[524288, 4096, 1], offset: ?>>, memref<1x128x128xf32>
            linalg.fill ins(%cst_1 : f32) outs(%15 : memref<1x64x128xf32>)
            linalg.batch_matmul ins(%22, %16 : memref<1x64x128xf32>, memref<1x128x128xf32>) outs(%15 : memref<1x64x128xf32>)
            linalg.fill ins(%cst_3 : f32) outs(%19 : memref<1x64xf32>)
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%15 : memref<1x64x128xf32>) outs(%19 : memref<1x64xf32>) {
            ^bb0(%in: f32, %out: f32):
              %32 = arith.maximumf %in, %out : f32
              linalg.yield %32 : f32
            }
            linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%18, %19 : memref<1x64xf32>, memref<1x64xf32>) outs(%19 : memref<1x64xf32>) {
            ^bb0(%in: f32, %in_7: f32, %out: f32):
              %32 = arith.truncf %cst_0 : f64 to f32
              %33 = arith.mulf %in_7, %32 : f32
              %34 = arith.cmpf ogt, %in, %33 : f32
              %35 = arith.select %34, %in, %33 : f32
              linalg.yield %35 : f32
            }
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%15, %19 : memref<1x64x128xf32>, memref<1x64xf32>) outs(%15 : memref<1x64x128xf32>) {
            ^bb0(%in: f32, %in_7: f32, %out: f32):
              %32 = arith.truncf %cst_0 : f64 to f32
              %33 = arith.mulf %in, %32 : f32
              %34 = arith.subf %33, %in_7 : f32
              %35 = math.powf %cst, %34 : f32
              linalg.yield %35 : f32
            }
            linalg.fill ins(%cst_1 : f32) outs(%20 : memref<1x64xf32>)
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%15 : memref<1x64x128xf32>) outs(%20 : memref<1x64xf32>) {
            ^bb0(%in: f32, %out: f32):
              %32 = arith.addf %in, %out : f32
              linalg.yield %32 : f32
            }
            linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%18, %19 : memref<1x64xf32>, memref<1x64xf32>) outs(%21 : memref<1x64xf32>) {
            ^bb0(%in: f32, %in_7: f32, %out: f32):
              %32 = arith.subf %in, %in_7 : f32
              %33 = math.powf %cst, %32 : f32
              linalg.yield %33 : f32
            }
            linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%17, %21, %20 : memref<1x64xf32>, memref<1x64xf32>, memref<1x64xf32>) outs(%17 : memref<1x64xf32>) {
            ^bb0(%in: f32, %in_7: f32, %in_8: f32, %out: f32):
              %32 = arith.mulf %in, %in_7 : f32
              %33 = arith.addf %32, %in_8 : f32
              linalg.yield %33 : f32
            }
            %30 = arith.muli %arg7, %c16384 : index
            %31 = arith.addi %25, %30 : index
            %reinterpret_cast_6 = memref.reinterpret_cast %arg1 to offset: [%31], sizes: [1, 128, 128], strides: [524288, 128, 1] : memref<32x4096x128xf32> to memref<1x128x128xf32, strided<[524288, 128, 1], offset: ?>>
            loom.copy %reinterpret_cast_6, %14 src_mem_space @DRAM dst_mem_space @L1, interconnect : [@horizontal_links, @vertical_links], broadcast : [8, 8] : memref<1x128x128xf32, strided<[524288, 128, 1], offset: ?>>, memref<1x128x128xf32>
            linalg.fill ins(%cst_1 : f32) outs(%24 : memref<1x64x128xf32>)
            linalg.batch_matmul ins(%15, %14 : memref<1x64x128xf32>, memref<1x128x128xf32>) outs(%24 : memref<1x64x128xf32>)
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%24, %23, %21 : memref<1x64x128xf32>, memref<1x64x128xf32>, memref<1x64xf32>) outs(%23 : memref<1x64x128xf32>) {
            ^bb0(%in: f32, %in_7: f32, %in_8: f32, %out: f32):
              %32 = arith.mulf %in_7, %in_8 : f32
              %33 = arith.addf %in, %32 : f32
              linalg.yield %33 : f32
            }
            linalg.copy ins(%19 : memref<1x64xf32>) outs(%18 : memref<1x64xf32>)
          }
          linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%23, %17 : memref<1x64x128xf32>, memref<1x64xf32>) outs(%22 : memref<1x64x128xf32>) {
          ^bb0(%in: f32, %in_5: f32, %out: f32):
            %28 = arith.divf %in, %in_5 : f32
            linalg.yield %28 : f32
          }
          %reinterpret_cast_4 = memref.reinterpret_cast %arg3 to offset: [%27], sizes: [1, 64, 128], strides: [524288, 128, 1] : memref<32x4096x128xf32> to memref<1x64x128xf32, strided<[524288, 128, 1], offset: ?>>
          loom.copy %22, %reinterpret_cast_4 src_mem_space @L1 dst_mem_space @DRAM, interconnect : [], broadcast : [1, 1] : memref<1x64x128xf32>, memref<1x64x128xf32, strided<[524288, 128, 1], offset: ?>>
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
          %15 = loom.alloc [1, 64, 128] on @L1 : memref<1x64x128xf32>
          %16 = loom.alloc [1, 128, 128] on @L1 : memref<1x128x128xf32>
          %17 = loom.alloc [1, 64] on @L1 : memref<1x64xf32>
          %18 = loom.alloc [1, 64] on @L1 : memref<1x64xf32>
          %19 = loom.alloc [1, 64] on @L1 : memref<1x64xf32>
          %20 = loom.alloc [1, 64] on @L1 : memref<1x64xf32>
          %21 = loom.alloc [1, 64] on @L1 : memref<1x64xf32>
          %22 = loom.alloc [1, 64, 128] on @L1 : memref<1x64x128xf32>
          %23 = loom.alloc [1, 64, 128] on @L1 : memref<1x64x128xf32>
          %24 = loom.alloc [1, 64, 128] on @L1 : memref<1x64x128xf32>
          %25 = arith.muli %arg6, %c524288 overflow<nsw> : index
          %26 = arith.muli %13, %c8192 : index
          %27 = arith.addi %25, %26 : index
          %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%27], sizes: [1, 64, 128], strides: [524288, 128, 1] : memref<32x4096x128xf32> to memref<1x64x128xf32, strided<[524288, 128, 1], offset: ?>>
          loom.copy %reinterpret_cast, %22 src_mem_space @DRAM dst_mem_space @L1, interconnect : [], broadcast : [1, 1] : memref<1x64x128xf32, strided<[524288, 128, 1], offset: ?>>, memref<1x64x128xf32>
          linalg.fill ins(%cst_1 : f32) outs(%23 : memref<1x64x128xf32>)
          linalg.fill ins(%cst_2 : f32) outs(%17 : memref<1x64xf32>)
          linalg.fill ins(%cst_3 : f32) outs(%18 : memref<1x64xf32>)
          scf.for %arg7 = %c0 to %c32 step %c1 {
            %28 = arith.muli %arg7, %c128 : index
            %29 = arith.addi %25, %28 : index
            %reinterpret_cast_5 = memref.reinterpret_cast %arg0 to offset: [%29], sizes: [1, 128, 128], strides: [524288, 4096, 1] : memref<32x128x4096xf32> to memref<1x128x128xf32, strided<[524288, 4096, 1], offset: ?>>
            loom.copy %reinterpret_cast_5, %16 src_mem_space @DRAM dst_mem_space @L1, interconnect : [@horizontal_links], broadcast : [1, 8] : memref<1x128x128xf32, strided<[524288, 4096, 1], offset: ?>>, memref<1x128x128xf32>
            linalg.fill ins(%cst_1 : f32) outs(%15 : memref<1x64x128xf32>)
            linalg.batch_matmul ins(%22, %16 : memref<1x64x128xf32>, memref<1x128x128xf32>) outs(%15 : memref<1x64x128xf32>)
            linalg.fill ins(%cst_3 : f32) outs(%19 : memref<1x64xf32>)
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%15 : memref<1x64x128xf32>) outs(%19 : memref<1x64xf32>) {
            ^bb0(%in: f32, %out: f32):
              %32 = arith.maximumf %in, %out : f32
              linalg.yield %32 : f32
            }
            linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%18, %19 : memref<1x64xf32>, memref<1x64xf32>) outs(%19 : memref<1x64xf32>) {
            ^bb0(%in: f32, %in_7: f32, %out: f32):
              %32 = arith.truncf %cst_0 : f64 to f32
              %33 = arith.mulf %in_7, %32 : f32
              %34 = arith.cmpf ogt, %in, %33 : f32
              %35 = arith.select %34, %in, %33 : f32
              linalg.yield %35 : f32
            }
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%15, %19 : memref<1x64x128xf32>, memref<1x64xf32>) outs(%15 : memref<1x64x128xf32>) {
            ^bb0(%in: f32, %in_7: f32, %out: f32):
              %32 = arith.truncf %cst_0 : f64 to f32
              %33 = arith.mulf %in, %32 : f32
              %34 = arith.subf %33, %in_7 : f32
              %35 = math.powf %cst, %34 : f32
              linalg.yield %35 : f32
            }
            linalg.fill ins(%cst_1 : f32) outs(%20 : memref<1x64xf32>)
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%15 : memref<1x64x128xf32>) outs(%20 : memref<1x64xf32>) {
            ^bb0(%in: f32, %out: f32):
              %32 = arith.addf %in, %out : f32
              linalg.yield %32 : f32
            }
            linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%18, %19 : memref<1x64xf32>, memref<1x64xf32>) outs(%21 : memref<1x64xf32>) {
            ^bb0(%in: f32, %in_7: f32, %out: f32):
              %32 = arith.subf %in, %in_7 : f32
              %33 = math.powf %cst, %32 : f32
              linalg.yield %33 : f32
            }
            linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%17, %21, %20 : memref<1x64xf32>, memref<1x64xf32>, memref<1x64xf32>) outs(%17 : memref<1x64xf32>) {
            ^bb0(%in: f32, %in_7: f32, %in_8: f32, %out: f32):
              %32 = arith.mulf %in, %in_7 : f32
              %33 = arith.addf %32, %in_8 : f32
              linalg.yield %33 : f32
            }
            %30 = arith.muli %arg7, %c16384 : index
            %31 = arith.addi %25, %30 : index
            %reinterpret_cast_6 = memref.reinterpret_cast %arg1 to offset: [%31], sizes: [1, 128, 128], strides: [524288, 128, 1] : memref<32x4096x128xf32> to memref<1x128x128xf32, strided<[524288, 128, 1], offset: ?>>
            loom.copy %reinterpret_cast_6, %14 src_mem_space @DRAM dst_mem_space @L1, interconnect : [@horizontal_links], broadcast : [1, 8] : memref<1x128x128xf32, strided<[524288, 128, 1], offset: ?>>, memref<1x128x128xf32>
            linalg.fill ins(%cst_1 : f32) outs(%24 : memref<1x64x128xf32>)
            linalg.batch_matmul ins(%15, %14 : memref<1x64x128xf32>, memref<1x128x128xf32>) outs(%24 : memref<1x64x128xf32>)
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%24, %23, %21 : memref<1x64x128xf32>, memref<1x64x128xf32>, memref<1x64xf32>) outs(%23 : memref<1x64x128xf32>) {
            ^bb0(%in: f32, %in_7: f32, %in_8: f32, %out: f32):
              %32 = arith.mulf %in_7, %in_8 : f32
              %33 = arith.addf %in, %32 : f32
              linalg.yield %33 : f32
            }
            linalg.copy ins(%19 : memref<1x64xf32>) outs(%18 : memref<1x64xf32>)
          }
          linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%23, %17 : memref<1x64x128xf32>, memref<1x64xf32>) outs(%22 : memref<1x64x128xf32>) {
          ^bb0(%in: f32, %in_5: f32, %out: f32):
            %28 = arith.divf %in, %in_5 : f32
            linalg.yield %28 : f32
          }
          %reinterpret_cast_4 = memref.reinterpret_cast %arg3 to offset: [%27], sizes: [1, 64, 128], strides: [524288, 128, 1] : memref<32x4096x128xf32> to memref<1x64x128xf32, strided<[524288, 128, 1], offset: ?>>
          loom.copy %22, %reinterpret_cast_4 src_mem_space @L1 dst_mem_space @DRAM, interconnect : [], broadcast : [1, 1] : memref<1x64x128xf32>, memref<1x64x128xf32, strided<[524288, 128, 1], offset: ?>>
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
          %15 = loom.alloc [1, 64, 128] on @L1 : memref<1x64x128xf32>
          %16 = loom.alloc [1, 128, 128] on @L1 : memref<1x128x128xf32>
          %17 = loom.alloc [1, 64] on @L1 : memref<1x64xf32>
          %18 = loom.alloc [1, 64] on @L1 : memref<1x64xf32>
          %19 = loom.alloc [1, 64] on @L1 : memref<1x64xf32>
          %20 = loom.alloc [1, 64] on @L1 : memref<1x64xf32>
          %21 = loom.alloc [1, 64] on @L1 : memref<1x64xf32>
          %22 = loom.alloc [1, 64, 128] on @L1 : memref<1x64x128xf32>
          %23 = loom.alloc [1, 64, 128] on @L1 : memref<1x64x128xf32>
          %24 = loom.alloc [1, 64, 128] on @L1 : memref<1x64x128xf32>
          %25 = arith.muli %arg6, %c524288 overflow<nsw> : index
          %26 = arith.muli %13, %c8192 : index
          %27 = arith.addi %25, %26 : index
          %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%27], sizes: [1, 64, 128], strides: [524288, 128, 1] : memref<32x4096x128xf32> to memref<1x64x128xf32, strided<[524288, 128, 1], offset: ?>>
          loom.copy %reinterpret_cast, %22 src_mem_space @DRAM dst_mem_space @L1, interconnect : [], broadcast : [1, 1] : memref<1x64x128xf32, strided<[524288, 128, 1], offset: ?>>, memref<1x64x128xf32>
          linalg.fill ins(%cst_1 : f32) outs(%23 : memref<1x64x128xf32>)
          linalg.fill ins(%cst_2 : f32) outs(%17 : memref<1x64xf32>)
          linalg.fill ins(%cst_3 : f32) outs(%18 : memref<1x64xf32>)
          scf.for %arg7 = %c0 to %c32 step %c1 {
            %28 = arith.muli %arg7, %c128 : index
            %29 = arith.addi %25, %28 : index
            %reinterpret_cast_5 = memref.reinterpret_cast %arg0 to offset: [%29], sizes: [1, 128, 128], strides: [524288, 4096, 1] : memref<32x128x4096xf32> to memref<1x128x128xf32, strided<[524288, 4096, 1], offset: ?>>
            loom.copy %reinterpret_cast_5, %16 src_mem_space @DRAM dst_mem_space @L1, interconnect : [@horizontal_links], broadcast : [1, 8] : memref<1x128x128xf32, strided<[524288, 4096, 1], offset: ?>>, memref<1x128x128xf32>
            linalg.fill ins(%cst_1 : f32) outs(%15 : memref<1x64x128xf32>)
            linalg.batch_matmul ins(%22, %16 : memref<1x64x128xf32>, memref<1x128x128xf32>) outs(%15 : memref<1x64x128xf32>)
            linalg.fill ins(%cst_3 : f32) outs(%19 : memref<1x64xf32>)
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%15 : memref<1x64x128xf32>) outs(%19 : memref<1x64xf32>) {
            ^bb0(%in: f32, %out: f32):
              %32 = arith.maximumf %in, %out : f32
              linalg.yield %32 : f32
            }
            linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%18, %19 : memref<1x64xf32>, memref<1x64xf32>) outs(%19 : memref<1x64xf32>) {
            ^bb0(%in: f32, %in_7: f32, %out: f32):
              %32 = arith.truncf %cst_0 : f64 to f32
              %33 = arith.mulf %in_7, %32 : f32
              %34 = arith.cmpf ogt, %in, %33 : f32
              %35 = arith.select %34, %in, %33 : f32
              linalg.yield %35 : f32
            }
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%15, %19 : memref<1x64x128xf32>, memref<1x64xf32>) outs(%15 : memref<1x64x128xf32>) {
            ^bb0(%in: f32, %in_7: f32, %out: f32):
              %32 = arith.truncf %cst_0 : f64 to f32
              %33 = arith.mulf %in, %32 : f32
              %34 = arith.subf %33, %in_7 : f32
              %35 = math.powf %cst, %34 : f32
              linalg.yield %35 : f32
            }
            linalg.fill ins(%cst_1 : f32) outs(%20 : memref<1x64xf32>)
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%15 : memref<1x64x128xf32>) outs(%20 : memref<1x64xf32>) {
            ^bb0(%in: f32, %out: f32):
              %32 = arith.addf %in, %out : f32
              linalg.yield %32 : f32
            }
            linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%18, %19 : memref<1x64xf32>, memref<1x64xf32>) outs(%21 : memref<1x64xf32>) {
            ^bb0(%in: f32, %in_7: f32, %out: f32):
              %32 = arith.subf %in, %in_7 : f32
              %33 = math.powf %cst, %32 : f32
              linalg.yield %33 : f32
            }
            linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%17, %21, %20 : memref<1x64xf32>, memref<1x64xf32>, memref<1x64xf32>) outs(%17 : memref<1x64xf32>) {
            ^bb0(%in: f32, %in_7: f32, %in_8: f32, %out: f32):
              %32 = arith.mulf %in, %in_7 : f32
              %33 = arith.addf %32, %in_8 : f32
              linalg.yield %33 : f32
            }
            %30 = arith.muli %arg7, %c16384 : index
            %31 = arith.addi %25, %30 : index
            %reinterpret_cast_6 = memref.reinterpret_cast %arg1 to offset: [%31], sizes: [1, 128, 128], strides: [524288, 128, 1] : memref<32x4096x128xf32> to memref<1x128x128xf32, strided<[524288, 128, 1], offset: ?>>
            loom.copy %reinterpret_cast_6, %14 src_mem_space @DRAM dst_mem_space @L1, interconnect : [@vertical_links], broadcast : [8, 1] : memref<1x128x128xf32, strided<[524288, 128, 1], offset: ?>>, memref<1x128x128xf32>
            linalg.fill ins(%cst_1 : f32) outs(%24 : memref<1x64x128xf32>)
            linalg.batch_matmul ins(%15, %14 : memref<1x64x128xf32>, memref<1x128x128xf32>) outs(%24 : memref<1x64x128xf32>)
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%24, %23, %21 : memref<1x64x128xf32>, memref<1x64x128xf32>, memref<1x64xf32>) outs(%23 : memref<1x64x128xf32>) {
            ^bb0(%in: f32, %in_7: f32, %in_8: f32, %out: f32):
              %32 = arith.mulf %in_7, %in_8 : f32
              %33 = arith.addf %in, %32 : f32
              linalg.yield %33 : f32
            }
            linalg.copy ins(%19 : memref<1x64xf32>) outs(%18 : memref<1x64xf32>)
          }
          linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%23, %17 : memref<1x64x128xf32>, memref<1x64xf32>) outs(%22 : memref<1x64x128xf32>) {
          ^bb0(%in: f32, %in_5: f32, %out: f32):
            %28 = arith.divf %in, %in_5 : f32
            linalg.yield %28 : f32
          }
          %reinterpret_cast_4 = memref.reinterpret_cast %arg3 to offset: [%27], sizes: [1, 64, 128], strides: [524288, 128, 1] : memref<32x4096x128xf32> to memref<1x64x128xf32, strided<[524288, 128, 1], offset: ?>>
          loom.copy %22, %reinterpret_cast_4 src_mem_space @L1 dst_mem_space @DRAM, interconnect : [], broadcast : [1, 1] : memref<1x64x128xf32>, memref<1x64x128xf32, strided<[524288, 128, 1], offset: ?>>
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
          %15 = loom.alloc [1, 64, 128] on @L1 : memref<1x64x128xf32>
          %16 = loom.alloc [1, 128, 128] on @L1 : memref<1x128x128xf32>
          %17 = loom.alloc [1, 64] on @L1 : memref<1x64xf32>
          %18 = loom.alloc [1, 64] on @L1 : memref<1x64xf32>
          %19 = loom.alloc [1, 64] on @L1 : memref<1x64xf32>
          %20 = loom.alloc [1, 64] on @L1 : memref<1x64xf32>
          %21 = loom.alloc [1, 64] on @L1 : memref<1x64xf32>
          %22 = loom.alloc [1, 64, 128] on @L1 : memref<1x64x128xf32>
          %23 = loom.alloc [1, 64, 128] on @L1 : memref<1x64x128xf32>
          %24 = loom.alloc [1, 64, 128] on @L1 : memref<1x64x128xf32>
          %25 = arith.muli %arg6, %c524288 overflow<nsw> : index
          %26 = arith.muli %13, %c8192 : index
          %27 = arith.addi %25, %26 : index
          %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%27], sizes: [1, 64, 128], strides: [524288, 128, 1] : memref<32x4096x128xf32> to memref<1x64x128xf32, strided<[524288, 128, 1], offset: ?>>
          loom.copy %reinterpret_cast, %22 src_mem_space @DRAM dst_mem_space @L1, interconnect : [], broadcast : [1, 1] : memref<1x64x128xf32, strided<[524288, 128, 1], offset: ?>>, memref<1x64x128xf32>
          linalg.fill ins(%cst_1 : f32) outs(%23 : memref<1x64x128xf32>)
          linalg.fill ins(%cst_2 : f32) outs(%17 : memref<1x64xf32>)
          linalg.fill ins(%cst_3 : f32) outs(%18 : memref<1x64xf32>)
          scf.for %arg7 = %c0 to %c32 step %c1 {
            %28 = arith.muli %arg7, %c128 : index
            %29 = arith.addi %25, %28 : index
            %reinterpret_cast_5 = memref.reinterpret_cast %arg0 to offset: [%29], sizes: [1, 128, 128], strides: [524288, 4096, 1] : memref<32x128x4096xf32> to memref<1x128x128xf32, strided<[524288, 4096, 1], offset: ?>>
            loom.copy %reinterpret_cast_5, %16 src_mem_space @DRAM dst_mem_space @L1, interconnect : [@vertical_links], broadcast : [8, 1] : memref<1x128x128xf32, strided<[524288, 4096, 1], offset: ?>>, memref<1x128x128xf32>
            linalg.fill ins(%cst_1 : f32) outs(%15 : memref<1x64x128xf32>)
            linalg.batch_matmul ins(%22, %16 : memref<1x64x128xf32>, memref<1x128x128xf32>) outs(%15 : memref<1x64x128xf32>)
            linalg.fill ins(%cst_3 : f32) outs(%19 : memref<1x64xf32>)
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%15 : memref<1x64x128xf32>) outs(%19 : memref<1x64xf32>) {
            ^bb0(%in: f32, %out: f32):
              %32 = arith.maximumf %in, %out : f32
              linalg.yield %32 : f32
            }
            linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%18, %19 : memref<1x64xf32>, memref<1x64xf32>) outs(%19 : memref<1x64xf32>) {
            ^bb0(%in: f32, %in_7: f32, %out: f32):
              %32 = arith.truncf %cst_0 : f64 to f32
              %33 = arith.mulf %in_7, %32 : f32
              %34 = arith.cmpf ogt, %in, %33 : f32
              %35 = arith.select %34, %in, %33 : f32
              linalg.yield %35 : f32
            }
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%15, %19 : memref<1x64x128xf32>, memref<1x64xf32>) outs(%15 : memref<1x64x128xf32>) {
            ^bb0(%in: f32, %in_7: f32, %out: f32):
              %32 = arith.truncf %cst_0 : f64 to f32
              %33 = arith.mulf %in, %32 : f32
              %34 = arith.subf %33, %in_7 : f32
              %35 = math.powf %cst, %34 : f32
              linalg.yield %35 : f32
            }
            linalg.fill ins(%cst_1 : f32) outs(%20 : memref<1x64xf32>)
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%15 : memref<1x64x128xf32>) outs(%20 : memref<1x64xf32>) {
            ^bb0(%in: f32, %out: f32):
              %32 = arith.addf %in, %out : f32
              linalg.yield %32 : f32
            }
            linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%18, %19 : memref<1x64xf32>, memref<1x64xf32>) outs(%21 : memref<1x64xf32>) {
            ^bb0(%in: f32, %in_7: f32, %out: f32):
              %32 = arith.subf %in, %in_7 : f32
              %33 = math.powf %cst, %32 : f32
              linalg.yield %33 : f32
            }
            linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%17, %21, %20 : memref<1x64xf32>, memref<1x64xf32>, memref<1x64xf32>) outs(%17 : memref<1x64xf32>) {
            ^bb0(%in: f32, %in_7: f32, %in_8: f32, %out: f32):
              %32 = arith.mulf %in, %in_7 : f32
              %33 = arith.addf %32, %in_8 : f32
              linalg.yield %33 : f32
            }
            %30 = arith.muli %arg7, %c16384 : index
            %31 = arith.addi %25, %30 : index
            %reinterpret_cast_6 = memref.reinterpret_cast %arg1 to offset: [%31], sizes: [1, 128, 128], strides: [524288, 128, 1] : memref<32x4096x128xf32> to memref<1x128x128xf32, strided<[524288, 128, 1], offset: ?>>
            loom.copy %reinterpret_cast_6, %14 src_mem_space @DRAM dst_mem_space @L1, interconnect : [], broadcast : [1, 1] : memref<1x128x128xf32, strided<[524288, 128, 1], offset: ?>>, memref<1x128x128xf32>
            linalg.fill ins(%cst_1 : f32) outs(%24 : memref<1x64x128xf32>)
            linalg.batch_matmul ins(%15, %14 : memref<1x64x128xf32>, memref<1x128x128xf32>) outs(%24 : memref<1x64x128xf32>)
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%24, %23, %21 : memref<1x64x128xf32>, memref<1x64x128xf32>, memref<1x64xf32>) outs(%23 : memref<1x64x128xf32>) {
            ^bb0(%in: f32, %in_7: f32, %in_8: f32, %out: f32):
              %32 = arith.mulf %in_7, %in_8 : f32
              %33 = arith.addf %in, %32 : f32
              linalg.yield %33 : f32
            }
            linalg.copy ins(%19 : memref<1x64xf32>) outs(%18 : memref<1x64xf32>)
          }
          linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%23, %17 : memref<1x64x128xf32>, memref<1x64xf32>) outs(%22 : memref<1x64x128xf32>) {
          ^bb0(%in: f32, %in_5: f32, %out: f32):
            %28 = arith.divf %in, %in_5 : f32
            linalg.yield %28 : f32
          }
          %reinterpret_cast_4 = memref.reinterpret_cast %arg3 to offset: [%27], sizes: [1, 64, 128], strides: [524288, 128, 1] : memref<32x4096x128xf32> to memref<1x64x128xf32, strided<[524288, 128, 1], offset: ?>>
          loom.copy %22, %reinterpret_cast_4 src_mem_space @L1 dst_mem_space @DRAM, interconnect : [], broadcast : [1, 1] : memref<1x64x128xf32>, memref<1x64x128xf32, strided<[524288, 128, 1], offset: ?>>
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
          %15 = loom.alloc [1, 64, 128] on @L1 : memref<1x64x128xf32>
          %16 = loom.alloc [1, 128, 128] on @L1 : memref<1x128x128xf32>
          %17 = loom.alloc [1, 64] on @L1 : memref<1x64xf32>
          %18 = loom.alloc [1, 64] on @L1 : memref<1x64xf32>
          %19 = loom.alloc [1, 64] on @L1 : memref<1x64xf32>
          %20 = loom.alloc [1, 64] on @L1 : memref<1x64xf32>
          %21 = loom.alloc [1, 64] on @L1 : memref<1x64xf32>
          %22 = loom.alloc [1, 64, 128] on @L1 : memref<1x64x128xf32>
          %23 = loom.alloc [1, 64, 128] on @L1 : memref<1x64x128xf32>
          %24 = loom.alloc [1, 64, 128] on @L1 : memref<1x64x128xf32>
          %25 = arith.muli %arg6, %c524288 overflow<nsw> : index
          %26 = arith.muli %13, %c8192 : index
          %27 = arith.addi %25, %26 : index
          %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%27], sizes: [1, 64, 128], strides: [524288, 128, 1] : memref<32x4096x128xf32> to memref<1x64x128xf32, strided<[524288, 128, 1], offset: ?>>
          loom.copy %reinterpret_cast, %22 src_mem_space @DRAM dst_mem_space @L1, interconnect : [], broadcast : [1, 1] : memref<1x64x128xf32, strided<[524288, 128, 1], offset: ?>>, memref<1x64x128xf32>
          linalg.fill ins(%cst_1 : f32) outs(%23 : memref<1x64x128xf32>)
          linalg.fill ins(%cst_2 : f32) outs(%17 : memref<1x64xf32>)
          linalg.fill ins(%cst_3 : f32) outs(%18 : memref<1x64xf32>)
          scf.for %arg7 = %c0 to %c32 step %c1 {
            %28 = arith.muli %arg7, %c128 : index
            %29 = arith.addi %25, %28 : index
            %reinterpret_cast_5 = memref.reinterpret_cast %arg0 to offset: [%29], sizes: [1, 128, 128], strides: [524288, 4096, 1] : memref<32x128x4096xf32> to memref<1x128x128xf32, strided<[524288, 4096, 1], offset: ?>>
            loom.copy %reinterpret_cast_5, %16 src_mem_space @DRAM dst_mem_space @L1, interconnect : [@vertical_links], broadcast : [8, 1] : memref<1x128x128xf32, strided<[524288, 4096, 1], offset: ?>>, memref<1x128x128xf32>
            linalg.fill ins(%cst_1 : f32) outs(%15 : memref<1x64x128xf32>)
            linalg.batch_matmul ins(%22, %16 : memref<1x64x128xf32>, memref<1x128x128xf32>) outs(%15 : memref<1x64x128xf32>)
            linalg.fill ins(%cst_3 : f32) outs(%19 : memref<1x64xf32>)
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%15 : memref<1x64x128xf32>) outs(%19 : memref<1x64xf32>) {
            ^bb0(%in: f32, %out: f32):
              %32 = arith.maximumf %in, %out : f32
              linalg.yield %32 : f32
            }
            linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%18, %19 : memref<1x64xf32>, memref<1x64xf32>) outs(%19 : memref<1x64xf32>) {
            ^bb0(%in: f32, %in_7: f32, %out: f32):
              %32 = arith.truncf %cst_0 : f64 to f32
              %33 = arith.mulf %in_7, %32 : f32
              %34 = arith.cmpf ogt, %in, %33 : f32
              %35 = arith.select %34, %in, %33 : f32
              linalg.yield %35 : f32
            }
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%15, %19 : memref<1x64x128xf32>, memref<1x64xf32>) outs(%15 : memref<1x64x128xf32>) {
            ^bb0(%in: f32, %in_7: f32, %out: f32):
              %32 = arith.truncf %cst_0 : f64 to f32
              %33 = arith.mulf %in, %32 : f32
              %34 = arith.subf %33, %in_7 : f32
              %35 = math.powf %cst, %34 : f32
              linalg.yield %35 : f32
            }
            linalg.fill ins(%cst_1 : f32) outs(%20 : memref<1x64xf32>)
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%15 : memref<1x64x128xf32>) outs(%20 : memref<1x64xf32>) {
            ^bb0(%in: f32, %out: f32):
              %32 = arith.addf %in, %out : f32
              linalg.yield %32 : f32
            }
            linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%18, %19 : memref<1x64xf32>, memref<1x64xf32>) outs(%21 : memref<1x64xf32>) {
            ^bb0(%in: f32, %in_7: f32, %out: f32):
              %32 = arith.subf %in, %in_7 : f32
              %33 = math.powf %cst, %32 : f32
              linalg.yield %33 : f32
            }
            linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%17, %21, %20 : memref<1x64xf32>, memref<1x64xf32>, memref<1x64xf32>) outs(%17 : memref<1x64xf32>) {
            ^bb0(%in: f32, %in_7: f32, %in_8: f32, %out: f32):
              %32 = arith.mulf %in, %in_7 : f32
              %33 = arith.addf %32, %in_8 : f32
              linalg.yield %33 : f32
            }
            %30 = arith.muli %arg7, %c16384 : index
            %31 = arith.addi %25, %30 : index
            %reinterpret_cast_6 = memref.reinterpret_cast %arg1 to offset: [%31], sizes: [1, 128, 128], strides: [524288, 128, 1] : memref<32x4096x128xf32> to memref<1x128x128xf32, strided<[524288, 128, 1], offset: ?>>
            loom.copy %reinterpret_cast_6, %14 src_mem_space @DRAM dst_mem_space @L1, interconnect : [@horizontal_links, @vertical_links], broadcast : [8, 8] : memref<1x128x128xf32, strided<[524288, 128, 1], offset: ?>>, memref<1x128x128xf32>
            linalg.fill ins(%cst_1 : f32) outs(%24 : memref<1x64x128xf32>)
            linalg.batch_matmul ins(%15, %14 : memref<1x64x128xf32>, memref<1x128x128xf32>) outs(%24 : memref<1x64x128xf32>)
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%24, %23, %21 : memref<1x64x128xf32>, memref<1x64x128xf32>, memref<1x64xf32>) outs(%23 : memref<1x64x128xf32>) {
            ^bb0(%in: f32, %in_7: f32, %in_8: f32, %out: f32):
              %32 = arith.mulf %in_7, %in_8 : f32
              %33 = arith.addf %in, %32 : f32
              linalg.yield %33 : f32
            }
            linalg.copy ins(%19 : memref<1x64xf32>) outs(%18 : memref<1x64xf32>)
          }
          linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%23, %17 : memref<1x64x128xf32>, memref<1x64xf32>) outs(%22 : memref<1x64x128xf32>) {
          ^bb0(%in: f32, %in_5: f32, %out: f32):
            %28 = arith.divf %in, %in_5 : f32
            linalg.yield %28 : f32
          }
          %reinterpret_cast_4 = memref.reinterpret_cast %arg3 to offset: [%27], sizes: [1, 64, 128], strides: [524288, 128, 1] : memref<32x4096x128xf32> to memref<1x64x128xf32, strided<[524288, 128, 1], offset: ?>>
          loom.copy %22, %reinterpret_cast_4 src_mem_space @L1 dst_mem_space @DRAM, interconnect : [], broadcast : [1, 1] : memref<1x64x128xf32>, memref<1x64x128xf32, strided<[524288, 128, 1], offset: ?>>
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
          %15 = loom.alloc [1, 64, 128] on @L1 : memref<1x64x128xf32>
          %16 = loom.alloc [1, 128, 128] on @L1 : memref<1x128x128xf32>
          %17 = loom.alloc [1, 64] on @L1 : memref<1x64xf32>
          %18 = loom.alloc [1, 64] on @L1 : memref<1x64xf32>
          %19 = loom.alloc [1, 64] on @L1 : memref<1x64xf32>
          %20 = loom.alloc [1, 64] on @L1 : memref<1x64xf32>
          %21 = loom.alloc [1, 64] on @L1 : memref<1x64xf32>
          %22 = loom.alloc [1, 64, 128] on @L1 : memref<1x64x128xf32>
          %23 = loom.alloc [1, 64, 128] on @L1 : memref<1x64x128xf32>
          %24 = loom.alloc [1, 64, 128] on @L1 : memref<1x64x128xf32>
          %25 = arith.muli %arg6, %c524288 overflow<nsw> : index
          %26 = arith.muli %13, %c8192 : index
          %27 = arith.addi %25, %26 : index
          %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%27], sizes: [1, 64, 128], strides: [524288, 128, 1] : memref<32x4096x128xf32> to memref<1x64x128xf32, strided<[524288, 128, 1], offset: ?>>
          loom.copy %reinterpret_cast, %22 src_mem_space @DRAM dst_mem_space @L1, interconnect : [], broadcast : [1, 1] : memref<1x64x128xf32, strided<[524288, 128, 1], offset: ?>>, memref<1x64x128xf32>
          linalg.fill ins(%cst_1 : f32) outs(%23 : memref<1x64x128xf32>)
          linalg.fill ins(%cst_2 : f32) outs(%17 : memref<1x64xf32>)
          linalg.fill ins(%cst_3 : f32) outs(%18 : memref<1x64xf32>)
          scf.for %arg7 = %c0 to %c32 step %c1 {
            %28 = arith.muli %arg7, %c128 : index
            %29 = arith.addi %25, %28 : index
            %reinterpret_cast_5 = memref.reinterpret_cast %arg0 to offset: [%29], sizes: [1, 128, 128], strides: [524288, 4096, 1] : memref<32x128x4096xf32> to memref<1x128x128xf32, strided<[524288, 4096, 1], offset: ?>>
            loom.copy %reinterpret_cast_5, %16 src_mem_space @DRAM dst_mem_space @L1, interconnect : [@vertical_links], broadcast : [8, 1] : memref<1x128x128xf32, strided<[524288, 4096, 1], offset: ?>>, memref<1x128x128xf32>
            linalg.fill ins(%cst_1 : f32) outs(%15 : memref<1x64x128xf32>)
            linalg.batch_matmul ins(%22, %16 : memref<1x64x128xf32>, memref<1x128x128xf32>) outs(%15 : memref<1x64x128xf32>)
            linalg.fill ins(%cst_3 : f32) outs(%19 : memref<1x64xf32>)
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%15 : memref<1x64x128xf32>) outs(%19 : memref<1x64xf32>) {
            ^bb0(%in: f32, %out: f32):
              %32 = arith.maximumf %in, %out : f32
              linalg.yield %32 : f32
            }
            linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%18, %19 : memref<1x64xf32>, memref<1x64xf32>) outs(%19 : memref<1x64xf32>) {
            ^bb0(%in: f32, %in_7: f32, %out: f32):
              %32 = arith.truncf %cst_0 : f64 to f32
              %33 = arith.mulf %in_7, %32 : f32
              %34 = arith.cmpf ogt, %in, %33 : f32
              %35 = arith.select %34, %in, %33 : f32
              linalg.yield %35 : f32
            }
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%15, %19 : memref<1x64x128xf32>, memref<1x64xf32>) outs(%15 : memref<1x64x128xf32>) {
            ^bb0(%in: f32, %in_7: f32, %out: f32):
              %32 = arith.truncf %cst_0 : f64 to f32
              %33 = arith.mulf %in, %32 : f32
              %34 = arith.subf %33, %in_7 : f32
              %35 = math.powf %cst, %34 : f32
              linalg.yield %35 : f32
            }
            linalg.fill ins(%cst_1 : f32) outs(%20 : memref<1x64xf32>)
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%15 : memref<1x64x128xf32>) outs(%20 : memref<1x64xf32>) {
            ^bb0(%in: f32, %out: f32):
              %32 = arith.addf %in, %out : f32
              linalg.yield %32 : f32
            }
            linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%18, %19 : memref<1x64xf32>, memref<1x64xf32>) outs(%21 : memref<1x64xf32>) {
            ^bb0(%in: f32, %in_7: f32, %out: f32):
              %32 = arith.subf %in, %in_7 : f32
              %33 = math.powf %cst, %32 : f32
              linalg.yield %33 : f32
            }
            linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%17, %21, %20 : memref<1x64xf32>, memref<1x64xf32>, memref<1x64xf32>) outs(%17 : memref<1x64xf32>) {
            ^bb0(%in: f32, %in_7: f32, %in_8: f32, %out: f32):
              %32 = arith.mulf %in, %in_7 : f32
              %33 = arith.addf %32, %in_8 : f32
              linalg.yield %33 : f32
            }
            %30 = arith.muli %arg7, %c16384 : index
            %31 = arith.addi %25, %30 : index
            %reinterpret_cast_6 = memref.reinterpret_cast %arg1 to offset: [%31], sizes: [1, 128, 128], strides: [524288, 128, 1] : memref<32x4096x128xf32> to memref<1x128x128xf32, strided<[524288, 128, 1], offset: ?>>
            loom.copy %reinterpret_cast_6, %14 src_mem_space @DRAM dst_mem_space @L1, interconnect : [@horizontal_links], broadcast : [1, 8] : memref<1x128x128xf32, strided<[524288, 128, 1], offset: ?>>, memref<1x128x128xf32>
            linalg.fill ins(%cst_1 : f32) outs(%24 : memref<1x64x128xf32>)
            linalg.batch_matmul ins(%15, %14 : memref<1x64x128xf32>, memref<1x128x128xf32>) outs(%24 : memref<1x64x128xf32>)
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%24, %23, %21 : memref<1x64x128xf32>, memref<1x64x128xf32>, memref<1x64xf32>) outs(%23 : memref<1x64x128xf32>) {
            ^bb0(%in: f32, %in_7: f32, %in_8: f32, %out: f32):
              %32 = arith.mulf %in_7, %in_8 : f32
              %33 = arith.addf %in, %32 : f32
              linalg.yield %33 : f32
            }
            linalg.copy ins(%19 : memref<1x64xf32>) outs(%18 : memref<1x64xf32>)
          }
          linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%23, %17 : memref<1x64x128xf32>, memref<1x64xf32>) outs(%22 : memref<1x64x128xf32>) {
          ^bb0(%in: f32, %in_5: f32, %out: f32):
            %28 = arith.divf %in, %in_5 : f32
            linalg.yield %28 : f32
          }
          %reinterpret_cast_4 = memref.reinterpret_cast %arg3 to offset: [%27], sizes: [1, 64, 128], strides: [524288, 128, 1] : memref<32x4096x128xf32> to memref<1x64x128xf32, strided<[524288, 128, 1], offset: ?>>
          loom.copy %22, %reinterpret_cast_4 src_mem_space @L1 dst_mem_space @DRAM, interconnect : [], broadcast : [1, 1] : memref<1x64x128xf32>, memref<1x64x128xf32, strided<[524288, 128, 1], offset: ?>>
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
          %15 = loom.alloc [1, 64, 128] on @L1 : memref<1x64x128xf32>
          %16 = loom.alloc [1, 128, 128] on @L1 : memref<1x128x128xf32>
          %17 = loom.alloc [1, 64] on @L1 : memref<1x64xf32>
          %18 = loom.alloc [1, 64] on @L1 : memref<1x64xf32>
          %19 = loom.alloc [1, 64] on @L1 : memref<1x64xf32>
          %20 = loom.alloc [1, 64] on @L1 : memref<1x64xf32>
          %21 = loom.alloc [1, 64] on @L1 : memref<1x64xf32>
          %22 = loom.alloc [1, 64, 128] on @L1 : memref<1x64x128xf32>
          %23 = loom.alloc [1, 64, 128] on @L1 : memref<1x64x128xf32>
          %24 = loom.alloc [1, 64, 128] on @L1 : memref<1x64x128xf32>
          %25 = arith.muli %arg6, %c524288 overflow<nsw> : index
          %26 = arith.muli %13, %c8192 : index
          %27 = arith.addi %25, %26 : index
          %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%27], sizes: [1, 64, 128], strides: [524288, 128, 1] : memref<32x4096x128xf32> to memref<1x64x128xf32, strided<[524288, 128, 1], offset: ?>>
          loom.copy %reinterpret_cast, %22 src_mem_space @DRAM dst_mem_space @L1, interconnect : [], broadcast : [1, 1] : memref<1x64x128xf32, strided<[524288, 128, 1], offset: ?>>, memref<1x64x128xf32>
          linalg.fill ins(%cst_1 : f32) outs(%23 : memref<1x64x128xf32>)
          linalg.fill ins(%cst_2 : f32) outs(%17 : memref<1x64xf32>)
          linalg.fill ins(%cst_3 : f32) outs(%18 : memref<1x64xf32>)
          scf.for %arg7 = %c0 to %c32 step %c1 {
            %28 = arith.muli %arg7, %c128 : index
            %29 = arith.addi %25, %28 : index
            %reinterpret_cast_5 = memref.reinterpret_cast %arg0 to offset: [%29], sizes: [1, 128, 128], strides: [524288, 4096, 1] : memref<32x128x4096xf32> to memref<1x128x128xf32, strided<[524288, 4096, 1], offset: ?>>
            loom.copy %reinterpret_cast_5, %16 src_mem_space @DRAM dst_mem_space @L1, interconnect : [@vertical_links], broadcast : [8, 1] : memref<1x128x128xf32, strided<[524288, 4096, 1], offset: ?>>, memref<1x128x128xf32>
            linalg.fill ins(%cst_1 : f32) outs(%15 : memref<1x64x128xf32>)
            linalg.batch_matmul ins(%22, %16 : memref<1x64x128xf32>, memref<1x128x128xf32>) outs(%15 : memref<1x64x128xf32>)
            linalg.fill ins(%cst_3 : f32) outs(%19 : memref<1x64xf32>)
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%15 : memref<1x64x128xf32>) outs(%19 : memref<1x64xf32>) {
            ^bb0(%in: f32, %out: f32):
              %32 = arith.maximumf %in, %out : f32
              linalg.yield %32 : f32
            }
            linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%18, %19 : memref<1x64xf32>, memref<1x64xf32>) outs(%19 : memref<1x64xf32>) {
            ^bb0(%in: f32, %in_7: f32, %out: f32):
              %32 = arith.truncf %cst_0 : f64 to f32
              %33 = arith.mulf %in_7, %32 : f32
              %34 = arith.cmpf ogt, %in, %33 : f32
              %35 = arith.select %34, %in, %33 : f32
              linalg.yield %35 : f32
            }
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%15, %19 : memref<1x64x128xf32>, memref<1x64xf32>) outs(%15 : memref<1x64x128xf32>) {
            ^bb0(%in: f32, %in_7: f32, %out: f32):
              %32 = arith.truncf %cst_0 : f64 to f32
              %33 = arith.mulf %in, %32 : f32
              %34 = arith.subf %33, %in_7 : f32
              %35 = math.powf %cst, %34 : f32
              linalg.yield %35 : f32
            }
            linalg.fill ins(%cst_1 : f32) outs(%20 : memref<1x64xf32>)
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%15 : memref<1x64x128xf32>) outs(%20 : memref<1x64xf32>) {
            ^bb0(%in: f32, %out: f32):
              %32 = arith.addf %in, %out : f32
              linalg.yield %32 : f32
            }
            linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%18, %19 : memref<1x64xf32>, memref<1x64xf32>) outs(%21 : memref<1x64xf32>) {
            ^bb0(%in: f32, %in_7: f32, %out: f32):
              %32 = arith.subf %in, %in_7 : f32
              %33 = math.powf %cst, %32 : f32
              linalg.yield %33 : f32
            }
            linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%17, %21, %20 : memref<1x64xf32>, memref<1x64xf32>, memref<1x64xf32>) outs(%17 : memref<1x64xf32>) {
            ^bb0(%in: f32, %in_7: f32, %in_8: f32, %out: f32):
              %32 = arith.mulf %in, %in_7 : f32
              %33 = arith.addf %32, %in_8 : f32
              linalg.yield %33 : f32
            }
            %30 = arith.muli %arg7, %c16384 : index
            %31 = arith.addi %25, %30 : index
            %reinterpret_cast_6 = memref.reinterpret_cast %arg1 to offset: [%31], sizes: [1, 128, 128], strides: [524288, 128, 1] : memref<32x4096x128xf32> to memref<1x128x128xf32, strided<[524288, 128, 1], offset: ?>>
            loom.copy %reinterpret_cast_6, %14 src_mem_space @DRAM dst_mem_space @L1, interconnect : [@vertical_links], broadcast : [8, 1] : memref<1x128x128xf32, strided<[524288, 128, 1], offset: ?>>, memref<1x128x128xf32>
            linalg.fill ins(%cst_1 : f32) outs(%24 : memref<1x64x128xf32>)
            linalg.batch_matmul ins(%15, %14 : memref<1x64x128xf32>, memref<1x128x128xf32>) outs(%24 : memref<1x64x128xf32>)
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%24, %23, %21 : memref<1x64x128xf32>, memref<1x64x128xf32>, memref<1x64xf32>) outs(%23 : memref<1x64x128xf32>) {
            ^bb0(%in: f32, %in_7: f32, %in_8: f32, %out: f32):
              %32 = arith.mulf %in_7, %in_8 : f32
              %33 = arith.addf %in, %32 : f32
              linalg.yield %33 : f32
            }
            linalg.copy ins(%19 : memref<1x64xf32>) outs(%18 : memref<1x64xf32>)
          }
          linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%23, %17 : memref<1x64x128xf32>, memref<1x64xf32>) outs(%22 : memref<1x64x128xf32>) {
          ^bb0(%in: f32, %in_5: f32, %out: f32):
            %28 = arith.divf %in, %in_5 : f32
            linalg.yield %28 : f32
          }
          %reinterpret_cast_4 = memref.reinterpret_cast %arg3 to offset: [%27], sizes: [1, 64, 128], strides: [524288, 128, 1] : memref<32x4096x128xf32> to memref<1x64x128xf32, strided<[524288, 128, 1], offset: ?>>
          loom.copy %22, %reinterpret_cast_4 src_mem_space @L1 dst_mem_space @DRAM, interconnect : [], broadcast : [1, 1] : memref<1x64x128xf32>, memref<1x64x128xf32, strided<[524288, 128, 1], offset: ?>>
        }
        scf.reduce 
      } {loom.iter_types = [#loom.iter_type<spatial>, #loom.iter_type<spatial>], loom.mapped_to_dims = [@y, @x]}
      return
    }
  }
}
