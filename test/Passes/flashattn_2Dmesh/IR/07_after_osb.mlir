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
      %c4096 = arith.constant 4096 : index
      %c16777216 = arith.constant 16777216 : index
      %c128 = arith.constant 128 : index
      %cst = arith.constant 2.000000e+00 : f32
      %cst_0 = arith.constant 0.12751743074602467 : f64
      %cst_1 = arith.constant 0.000000e+00 : f32
      %cst_2 = arith.constant 1.000000e+00 : f32
      %cst_3 = arith.constant 0xFF800000 : f32
      %c32 = arith.constant 32 : index
      %c0 = arith.constant 0 : index
      %c8 = arith.constant 8 : index
      %c1 = arith.constant 1 : index
      scf.parallel (%arg4, %arg5) = (%c0, %c0) to (%c8, %c8) step (%c1, %c1) {
        scf.for %arg6 = %c0 to %c128 step %c1 {
          %12 = arith.muli %arg4, %c8 overflow<nsw> : index
          %13 = arith.addi %12, %arg5 : index
          %14 = loom.alloc [32, 32, 128] on @L1 : memref<32x32x128xf32>
          %15 = loom.alloc [32, 32, 32] on @L1 : memref<32x32x32xf32>
          %16 = loom.alloc [32, 128, 32] on @L1 : memref<32x128x32xf32>
          %17 = loom.alloc [32, 32] on @L1 : memref<32x32xf32>
          %18 = loom.alloc [32, 32] on @L1 : memref<32x32xf32>
          %19 = loom.alloc [32, 32] on @L1 : memref<32x32xf32>
          %20 = loom.alloc [32, 32] on @L1 : memref<32x32xf32>
          %21 = loom.alloc [32, 32] on @L1 : memref<32x32xf32>
          %22 = loom.alloc [32, 32, 128] on @L1 : memref<32x32x128xf32>
          %23 = loom.alloc [32, 32, 128] on @L1 : memref<32x32x128xf32>
          %24 = loom.alloc [32, 32, 128] on @L1 : memref<32x32x128xf32>
          %25 = arith.muli %13, %c16777216 : index
          %26 = arith.muli %arg6, %c4096 : index
          %27 = arith.addi %25, %26 : index
          %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%27], sizes: [32, 32, 128], strides: [524288, 128, 1] : memref<32x4096x128xf32> to memref<32x32x128xf32, strided<[524288, 128, 1], offset: ?>>
          loom.copy %reinterpret_cast, %22 src_mem_space @DRAM dst_mem_space @L1, interconnect : [], broadcast : [1, 1] : memref<32x32x128xf32, strided<[524288, 128, 1], offset: ?>>, memref<32x32x128xf32>
          linalg.fill ins(%cst_1 : f32) outs(%23 : memref<32x32x128xf32>)
          linalg.fill ins(%cst_2 : f32) outs(%17 : memref<32x32xf32>)
          linalg.fill ins(%cst_3 : f32) outs(%18 : memref<32x32xf32>)
          scf.for %arg7 = %c0 to %c128 step %c1 {
            %28 = arith.muli %arg7, %c32 : index
            %29 = arith.addi %25, %28 : index
            %reinterpret_cast_5 = memref.reinterpret_cast %arg0 to offset: [%29], sizes: [32, 128, 32], strides: [524288, 4096, 1] : memref<32x128x4096xf32> to memref<32x128x32xf32, strided<[524288, 4096, 1], offset: ?>>
            loom.copy %reinterpret_cast_5, %16 src_mem_space @DRAM dst_mem_space @L1, interconnect : [], broadcast : [1, 1] : memref<32x128x32xf32, strided<[524288, 4096, 1], offset: ?>>, memref<32x128x32xf32>
            linalg.fill ins(%cst_1 : f32) outs(%15 : memref<32x32x32xf32>)
            linalg.batch_matmul ins(%22, %16 : memref<32x32x128xf32>, memref<32x128x32xf32>) outs(%15 : memref<32x32x32xf32>)
            linalg.fill ins(%cst_3 : f32) outs(%19 : memref<32x32xf32>)
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%15 : memref<32x32x32xf32>) outs(%19 : memref<32x32xf32>) {
            ^bb0(%in: f32, %out: f32):
              %32 = arith.maximumf %in, %out : f32
              linalg.yield %32 : f32
            }
            linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%18, %19 : memref<32x32xf32>, memref<32x32xf32>) outs(%19 : memref<32x32xf32>) {
            ^bb0(%in: f32, %in_7: f32, %out: f32):
              %32 = arith.truncf %cst_0 : f64 to f32
              %33 = arith.mulf %in_7, %32 : f32
              %34 = arith.cmpf ogt, %in, %33 : f32
              %35 = arith.select %34, %in, %33 : f32
              linalg.yield %35 : f32
            }
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%15, %19 : memref<32x32x32xf32>, memref<32x32xf32>) outs(%15 : memref<32x32x32xf32>) {
            ^bb0(%in: f32, %in_7: f32, %out: f32):
              %32 = arith.truncf %cst_0 : f64 to f32
              %33 = arith.mulf %in, %32 : f32
              %34 = arith.subf %33, %in_7 : f32
              %35 = math.powf %cst, %34 : f32
              linalg.yield %35 : f32
            }
            linalg.fill ins(%cst_1 : f32) outs(%20 : memref<32x32xf32>)
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%15 : memref<32x32x32xf32>) outs(%20 : memref<32x32xf32>) {
            ^bb0(%in: f32, %out: f32):
              %32 = arith.addf %in, %out : f32
              linalg.yield %32 : f32
            }
            linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%18, %19 : memref<32x32xf32>, memref<32x32xf32>) outs(%21 : memref<32x32xf32>) {
            ^bb0(%in: f32, %in_7: f32, %out: f32):
              %32 = arith.subf %in, %in_7 : f32
              %33 = math.powf %cst, %32 : f32
              linalg.yield %33 : f32
            }
            linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%17, %21, %20 : memref<32x32xf32>, memref<32x32xf32>, memref<32x32xf32>) outs(%17 : memref<32x32xf32>) {
            ^bb0(%in: f32, %in_7: f32, %in_8: f32, %out: f32):
              %32 = arith.mulf %in, %in_7 : f32
              %33 = arith.addf %32, %in_8 : f32
              linalg.yield %33 : f32
            }
            %30 = arith.muli %arg7, %c4096 : index
            %31 = arith.addi %25, %30 : index
            %reinterpret_cast_6 = memref.reinterpret_cast %arg1 to offset: [%31], sizes: [32, 32, 128], strides: [524288, 128, 1] : memref<32x4096x128xf32> to memref<32x32x128xf32, strided<[524288, 128, 1], offset: ?>>
            loom.copy %reinterpret_cast_6, %14 src_mem_space @DRAM dst_mem_space @L1, interconnect : [], broadcast : [1, 1] : memref<32x32x128xf32, strided<[524288, 128, 1], offset: ?>>, memref<32x32x128xf32>
            linalg.fill ins(%cst_1 : f32) outs(%24 : memref<32x32x128xf32>)
            linalg.batch_matmul ins(%15, %14 : memref<32x32x32xf32>, memref<32x32x128xf32>) outs(%24 : memref<32x32x128xf32>)
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%24, %23, %21 : memref<32x32x128xf32>, memref<32x32x128xf32>, memref<32x32xf32>) outs(%23 : memref<32x32x128xf32>) {
            ^bb0(%in: f32, %in_7: f32, %in_8: f32, %out: f32):
              %32 = arith.mulf %in_7, %in_8 : f32
              %33 = arith.addf %in, %32 : f32
              linalg.yield %33 : f32
            }
            linalg.copy ins(%19 : memref<32x32xf32>) outs(%18 : memref<32x32xf32>)
          }
          linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%23, %17 : memref<32x32x128xf32>, memref<32x32xf32>) outs(%22 : memref<32x32x128xf32>) {
          ^bb0(%in: f32, %in_5: f32, %out: f32):
            %28 = arith.divf %in, %in_5 : f32
            linalg.yield %28 : f32
          }
          %reinterpret_cast_4 = memref.reinterpret_cast %arg3 to offset: [%27], sizes: [32, 32, 128], strides: [524288, 128, 1] : memref<32x4096x128xf32> to memref<32x32x128xf32, strided<[524288, 128, 1], offset: ?>>
          loom.copy %22, %reinterpret_cast_4 src_mem_space @L1 dst_mem_space @DRAM, interconnect : [], broadcast : [1, 1] : memref<32x32x128xf32>, memref<32x32x128xf32, strided<[524288, 128, 1], offset: ?>>
        }
        scf.reduce 
      } {loom.iter_types = [#loom.iter_type<spatial>, #loom.iter_type<spatial>], loom.mapped_to_dims = [@x, @y]}
      return
    }
  }
  module attributes {loom.pass_name = "Materialize"} {
    func.func @attention__d0i0_d1i0__f10__d_d_d__BB32__BM32__BN32(%arg0: memref<32x128x4096xf32>, %arg1: memref<32x4096x128xf32>, %arg2: memref<32x4096x128xf32>, %arg3: memref<32x4096x128xf32>) {
      %c4096 = arith.constant 4096 : index
      %c16777216 = arith.constant 16777216 : index
      %c128 = arith.constant 128 : index
      %cst = arith.constant 2.000000e+00 : f32
      %cst_0 = arith.constant 0.12751743074602467 : f64
      %cst_1 = arith.constant 0.000000e+00 : f32
      %cst_2 = arith.constant 1.000000e+00 : f32
      %cst_3 = arith.constant 0xFF800000 : f32
      %c32 = arith.constant 32 : index
      %c0 = arith.constant 0 : index
      %c8 = arith.constant 8 : index
      %c1 = arith.constant 1 : index
      scf.parallel (%arg4, %arg5) = (%c0, %c0) to (%c8, %c8) step (%c1, %c1) {
        scf.for %arg6 = %c0 to %c128 step %c1 {
          %12 = arith.muli %arg4, %c8 overflow<nsw> : index
          %13 = arith.addi %12, %arg5 : index
          %14 = loom.alloc [32, 32, 128] on @L1 : memref<32x32x128xf32>
          %15 = loom.alloc [32, 32, 32] on @L1 : memref<32x32x32xf32>
          %16 = loom.alloc [32, 128, 32] on @L1 : memref<32x128x32xf32>
          %17 = loom.alloc [32, 32] on @L1 : memref<32x32xf32>
          %18 = loom.alloc [32, 32] on @L1 : memref<32x32xf32>
          %19 = loom.alloc [32, 32] on @L1 : memref<32x32xf32>
          %20 = loom.alloc [32, 32] on @L1 : memref<32x32xf32>
          %21 = loom.alloc [32, 32] on @L1 : memref<32x32xf32>
          %22 = loom.alloc [32, 32, 128] on @L1 : memref<32x32x128xf32>
          %23 = loom.alloc [32, 32, 128] on @L1 : memref<32x32x128xf32>
          %24 = loom.alloc [32, 32, 128] on @L1 : memref<32x32x128xf32>
          %25 = arith.muli %13, %c16777216 : index
          %26 = arith.muli %arg6, %c4096 : index
          %27 = arith.addi %25, %26 : index
          %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%27], sizes: [32, 32, 128], strides: [524288, 128, 1] : memref<32x4096x128xf32> to memref<32x32x128xf32, strided<[524288, 128, 1], offset: ?>>
          loom.copy %reinterpret_cast, %22 src_mem_space @DRAM dst_mem_space @L1, interconnect : [], broadcast : [1, 1] : memref<32x32x128xf32, strided<[524288, 128, 1], offset: ?>>, memref<32x32x128xf32>
          linalg.fill ins(%cst_1 : f32) outs(%23 : memref<32x32x128xf32>)
          linalg.fill ins(%cst_2 : f32) outs(%17 : memref<32x32xf32>)
          linalg.fill ins(%cst_3 : f32) outs(%18 : memref<32x32xf32>)
          scf.for %arg7 = %c0 to %c128 step %c1 {
            %28 = arith.muli %arg7, %c32 : index
            %29 = arith.addi %25, %28 : index
            %reinterpret_cast_5 = memref.reinterpret_cast %arg0 to offset: [%29], sizes: [32, 128, 32], strides: [524288, 4096, 1] : memref<32x128x4096xf32> to memref<32x128x32xf32, strided<[524288, 4096, 1], offset: ?>>
            loom.copy %reinterpret_cast_5, %16 src_mem_space @DRAM dst_mem_space @L1, interconnect : [], broadcast : [1, 1] : memref<32x128x32xf32, strided<[524288, 4096, 1], offset: ?>>, memref<32x128x32xf32>
            linalg.fill ins(%cst_1 : f32) outs(%15 : memref<32x32x32xf32>)
            linalg.batch_matmul ins(%22, %16 : memref<32x32x128xf32>, memref<32x128x32xf32>) outs(%15 : memref<32x32x32xf32>)
            linalg.fill ins(%cst_3 : f32) outs(%19 : memref<32x32xf32>)
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%15 : memref<32x32x32xf32>) outs(%19 : memref<32x32xf32>) {
            ^bb0(%in: f32, %out: f32):
              %32 = arith.maximumf %in, %out : f32
              linalg.yield %32 : f32
            }
            linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%18, %19 : memref<32x32xf32>, memref<32x32xf32>) outs(%19 : memref<32x32xf32>) {
            ^bb0(%in: f32, %in_7: f32, %out: f32):
              %32 = arith.truncf %cst_0 : f64 to f32
              %33 = arith.mulf %in_7, %32 : f32
              %34 = arith.cmpf ogt, %in, %33 : f32
              %35 = arith.select %34, %in, %33 : f32
              linalg.yield %35 : f32
            }
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%15, %19 : memref<32x32x32xf32>, memref<32x32xf32>) outs(%15 : memref<32x32x32xf32>) {
            ^bb0(%in: f32, %in_7: f32, %out: f32):
              %32 = arith.truncf %cst_0 : f64 to f32
              %33 = arith.mulf %in, %32 : f32
              %34 = arith.subf %33, %in_7 : f32
              %35 = math.powf %cst, %34 : f32
              linalg.yield %35 : f32
            }
            linalg.fill ins(%cst_1 : f32) outs(%20 : memref<32x32xf32>)
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%15 : memref<32x32x32xf32>) outs(%20 : memref<32x32xf32>) {
            ^bb0(%in: f32, %out: f32):
              %32 = arith.addf %in, %out : f32
              linalg.yield %32 : f32
            }
            linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%18, %19 : memref<32x32xf32>, memref<32x32xf32>) outs(%21 : memref<32x32xf32>) {
            ^bb0(%in: f32, %in_7: f32, %out: f32):
              %32 = arith.subf %in, %in_7 : f32
              %33 = math.powf %cst, %32 : f32
              linalg.yield %33 : f32
            }
            linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%17, %21, %20 : memref<32x32xf32>, memref<32x32xf32>, memref<32x32xf32>) outs(%17 : memref<32x32xf32>) {
            ^bb0(%in: f32, %in_7: f32, %in_8: f32, %out: f32):
              %32 = arith.mulf %in, %in_7 : f32
              %33 = arith.addf %32, %in_8 : f32
              linalg.yield %33 : f32
            }
            %30 = arith.muli %arg7, %c4096 : index
            %31 = arith.addi %25, %30 : index
            %reinterpret_cast_6 = memref.reinterpret_cast %arg1 to offset: [%31], sizes: [32, 32, 128], strides: [524288, 128, 1] : memref<32x4096x128xf32> to memref<32x32x128xf32, strided<[524288, 128, 1], offset: ?>>
            loom.copy %reinterpret_cast_6, %14 src_mem_space @DRAM dst_mem_space @L1, interconnect : [], broadcast : [1, 1] : memref<32x32x128xf32, strided<[524288, 128, 1], offset: ?>>, memref<32x32x128xf32>
            linalg.fill ins(%cst_1 : f32) outs(%24 : memref<32x32x128xf32>)
            linalg.batch_matmul ins(%15, %14 : memref<32x32x32xf32>, memref<32x32x128xf32>) outs(%24 : memref<32x32x128xf32>)
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%24, %23, %21 : memref<32x32x128xf32>, memref<32x32x128xf32>, memref<32x32xf32>) outs(%23 : memref<32x32x128xf32>) {
            ^bb0(%in: f32, %in_7: f32, %in_8: f32, %out: f32):
              %32 = arith.mulf %in_7, %in_8 : f32
              %33 = arith.addf %in, %32 : f32
              linalg.yield %33 : f32
            }
            linalg.copy ins(%19 : memref<32x32xf32>) outs(%18 : memref<32x32xf32>)
          }
          linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%23, %17 : memref<32x32x128xf32>, memref<32x32xf32>) outs(%22 : memref<32x32x128xf32>) {
          ^bb0(%in: f32, %in_5: f32, %out: f32):
            %28 = arith.divf %in, %in_5 : f32
            linalg.yield %28 : f32
          }
          %reinterpret_cast_4 = memref.reinterpret_cast %arg3 to offset: [%27], sizes: [32, 32, 128], strides: [524288, 128, 1] : memref<32x4096x128xf32> to memref<32x32x128xf32, strided<[524288, 128, 1], offset: ?>>
          loom.copy %22, %reinterpret_cast_4 src_mem_space @L1 dst_mem_space @DRAM, interconnect : [], broadcast : [1, 1] : memref<32x32x128xf32>, memref<32x32x128xf32, strided<[524288, 128, 1], offset: ?>>
        }
        scf.reduce 
      } {loom.iter_types = [#loom.iter_type<spatial>, #loom.iter_type<spatial>], loom.mapped_to_dims = [@x, @y]}
      return
    }
  }
  module attributes {loom.pass_name = "Materialize"} {
    func.func @attention__d0i0_d1i1__f01__d_d_d__BB32__BM32__BN32(%arg0: memref<32x128x4096xf32>, %arg1: memref<32x4096x128xf32>, %arg2: memref<32x4096x128xf32>, %arg3: memref<32x4096x128xf32>) {
      %c4096 = arith.constant 4096 : index
      %c16777216 = arith.constant 16777216 : index
      %c128 = arith.constant 128 : index
      %c16 = arith.constant 16 : index
      %cst = arith.constant 2.000000e+00 : f32
      %cst_0 = arith.constant 0.12751743074602467 : f64
      %cst_1 = arith.constant 0.000000e+00 : f32
      %cst_2 = arith.constant 1.000000e+00 : f32
      %cst_3 = arith.constant 0xFF800000 : f32
      %c32 = arith.constant 32 : index
      %c0 = arith.constant 0 : index
      %c8 = arith.constant 8 : index
      %c1 = arith.constant 1 : index
      scf.parallel (%arg4, %arg5) = (%c0, %c0) to (%c8, %c8) step (%c1, %c1) {
        scf.for %arg6 = %c0 to %c16 step %c1 {
          %12 = arith.muli %arg6, %c8 overflow<nsw> : index
          %13 = arith.addi %arg5, %12 : index
          %14 = loom.alloc [32, 32, 128] on @L1 : memref<32x32x128xf32>
          %15 = loom.alloc [32, 32, 32] on @L1 : memref<32x32x32xf32>
          %16 = loom.alloc [32, 128, 32] on @L1 : memref<32x128x32xf32>
          %17 = loom.alloc [32, 32] on @L1 : memref<32x32xf32>
          %18 = loom.alloc [32, 32] on @L1 : memref<32x32xf32>
          %19 = loom.alloc [32, 32] on @L1 : memref<32x32xf32>
          %20 = loom.alloc [32, 32] on @L1 : memref<32x32xf32>
          %21 = loom.alloc [32, 32] on @L1 : memref<32x32xf32>
          %22 = loom.alloc [32, 32, 128] on @L1 : memref<32x32x128xf32>
          %23 = loom.alloc [32, 32, 128] on @L1 : memref<32x32x128xf32>
          %24 = loom.alloc [32, 32, 128] on @L1 : memref<32x32x128xf32>
          %25 = arith.muli %arg4, %c16777216 : index
          %26 = arith.muli %13, %c4096 : index
          %27 = arith.addi %25, %26 : index
          %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%27], sizes: [32, 32, 128], strides: [524288, 128, 1] : memref<32x4096x128xf32> to memref<32x32x128xf32, strided<[524288, 128, 1], offset: ?>>
          loom.copy %reinterpret_cast, %22 src_mem_space @DRAM dst_mem_space @L1, interconnect : [], broadcast : [1, 1] : memref<32x32x128xf32, strided<[524288, 128, 1], offset: ?>>, memref<32x32x128xf32>
          linalg.fill ins(%cst_1 : f32) outs(%23 : memref<32x32x128xf32>)
          linalg.fill ins(%cst_2 : f32) outs(%17 : memref<32x32xf32>)
          linalg.fill ins(%cst_3 : f32) outs(%18 : memref<32x32xf32>)
          scf.for %arg7 = %c0 to %c128 step %c1 {
            %28 = arith.muli %arg7, %c32 : index
            %29 = arith.addi %25, %28 : index
            %reinterpret_cast_5 = memref.reinterpret_cast %arg0 to offset: [%29], sizes: [32, 128, 32], strides: [524288, 4096, 1] : memref<32x128x4096xf32> to memref<32x128x32xf32, strided<[524288, 4096, 1], offset: ?>>
            loom.copy %reinterpret_cast_5, %16 src_mem_space @DRAM dst_mem_space @L1, interconnect : [], broadcast : [1, 1] : memref<32x128x32xf32, strided<[524288, 4096, 1], offset: ?>>, memref<32x128x32xf32>
            linalg.fill ins(%cst_1 : f32) outs(%15 : memref<32x32x32xf32>)
            linalg.batch_matmul ins(%22, %16 : memref<32x32x128xf32>, memref<32x128x32xf32>) outs(%15 : memref<32x32x32xf32>)
            linalg.fill ins(%cst_3 : f32) outs(%19 : memref<32x32xf32>)
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%15 : memref<32x32x32xf32>) outs(%19 : memref<32x32xf32>) {
            ^bb0(%in: f32, %out: f32):
              %32 = arith.maximumf %in, %out : f32
              linalg.yield %32 : f32
            }
            linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%18, %19 : memref<32x32xf32>, memref<32x32xf32>) outs(%19 : memref<32x32xf32>) {
            ^bb0(%in: f32, %in_7: f32, %out: f32):
              %32 = arith.truncf %cst_0 : f64 to f32
              %33 = arith.mulf %in_7, %32 : f32
              %34 = arith.cmpf ogt, %in, %33 : f32
              %35 = arith.select %34, %in, %33 : f32
              linalg.yield %35 : f32
            }
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%15, %19 : memref<32x32x32xf32>, memref<32x32xf32>) outs(%15 : memref<32x32x32xf32>) {
            ^bb0(%in: f32, %in_7: f32, %out: f32):
              %32 = arith.truncf %cst_0 : f64 to f32
              %33 = arith.mulf %in, %32 : f32
              %34 = arith.subf %33, %in_7 : f32
              %35 = math.powf %cst, %34 : f32
              linalg.yield %35 : f32
            }
            linalg.fill ins(%cst_1 : f32) outs(%20 : memref<32x32xf32>)
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%15 : memref<32x32x32xf32>) outs(%20 : memref<32x32xf32>) {
            ^bb0(%in: f32, %out: f32):
              %32 = arith.addf %in, %out : f32
              linalg.yield %32 : f32
            }
            linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%18, %19 : memref<32x32xf32>, memref<32x32xf32>) outs(%21 : memref<32x32xf32>) {
            ^bb0(%in: f32, %in_7: f32, %out: f32):
              %32 = arith.subf %in, %in_7 : f32
              %33 = math.powf %cst, %32 : f32
              linalg.yield %33 : f32
            }
            linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%17, %21, %20 : memref<32x32xf32>, memref<32x32xf32>, memref<32x32xf32>) outs(%17 : memref<32x32xf32>) {
            ^bb0(%in: f32, %in_7: f32, %in_8: f32, %out: f32):
              %32 = arith.mulf %in, %in_7 : f32
              %33 = arith.addf %32, %in_8 : f32
              linalg.yield %33 : f32
            }
            %30 = arith.muli %arg7, %c4096 : index
            %31 = arith.addi %25, %30 : index
            %reinterpret_cast_6 = memref.reinterpret_cast %arg1 to offset: [%31], sizes: [32, 32, 128], strides: [524288, 128, 1] : memref<32x4096x128xf32> to memref<32x32x128xf32, strided<[524288, 128, 1], offset: ?>>
            loom.copy %reinterpret_cast_6, %14 src_mem_space @DRAM dst_mem_space @L1, interconnect : [], broadcast : [1, 1] : memref<32x32x128xf32, strided<[524288, 128, 1], offset: ?>>, memref<32x32x128xf32>
            linalg.fill ins(%cst_1 : f32) outs(%24 : memref<32x32x128xf32>)
            linalg.batch_matmul ins(%15, %14 : memref<32x32x32xf32>, memref<32x32x128xf32>) outs(%24 : memref<32x32x128xf32>)
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%24, %23, %21 : memref<32x32x128xf32>, memref<32x32x128xf32>, memref<32x32xf32>) outs(%23 : memref<32x32x128xf32>) {
            ^bb0(%in: f32, %in_7: f32, %in_8: f32, %out: f32):
              %32 = arith.mulf %in_7, %in_8 : f32
              %33 = arith.addf %in, %32 : f32
              linalg.yield %33 : f32
            }
            linalg.copy ins(%19 : memref<32x32xf32>) outs(%18 : memref<32x32xf32>)
          }
          linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%23, %17 : memref<32x32x128xf32>, memref<32x32xf32>) outs(%22 : memref<32x32x128xf32>) {
          ^bb0(%in: f32, %in_5: f32, %out: f32):
            %28 = arith.divf %in, %in_5 : f32
            linalg.yield %28 : f32
          }
          %reinterpret_cast_4 = memref.reinterpret_cast %arg3 to offset: [%27], sizes: [32, 32, 128], strides: [524288, 128, 1] : memref<32x4096x128xf32> to memref<32x32x128xf32, strided<[524288, 128, 1], offset: ?>>
          loom.copy %22, %reinterpret_cast_4 src_mem_space @L1 dst_mem_space @DRAM, interconnect : [], broadcast : [1, 1] : memref<32x32x128xf32>, memref<32x32x128xf32, strided<[524288, 128, 1], offset: ?>>
        }
        scf.reduce 
      } {loom.iter_types = [#loom.iter_type<spatial>, #loom.iter_type<spatial>], loom.mapped_to_dims = [@x, @y]}
      return
    }
  }
  module attributes {loom.pass_name = "Materialize"} {
    func.func @attention__d0i0_d1i1__f01__d_d_v__BB32__BM32__BN32(%arg0: memref<32x128x4096xf32>, %arg1: memref<32x4096x128xf32>, %arg2: memref<32x4096x128xf32>, %arg3: memref<32x4096x128xf32>) {
      %c4096 = arith.constant 4096 : index
      %c16777216 = arith.constant 16777216 : index
      %c128 = arith.constant 128 : index
      %c16 = arith.constant 16 : index
      %cst = arith.constant 2.000000e+00 : f32
      %cst_0 = arith.constant 0.12751743074602467 : f64
      %cst_1 = arith.constant 0.000000e+00 : f32
      %cst_2 = arith.constant 1.000000e+00 : f32
      %cst_3 = arith.constant 0xFF800000 : f32
      %c32 = arith.constant 32 : index
      %c0 = arith.constant 0 : index
      %c8 = arith.constant 8 : index
      %c1 = arith.constant 1 : index
      scf.parallel (%arg4, %arg5) = (%c0, %c0) to (%c8, %c8) step (%c1, %c1) {
        scf.for %arg6 = %c0 to %c16 step %c1 {
          %12 = arith.muli %arg6, %c8 overflow<nsw> : index
          %13 = arith.addi %arg5, %12 : index
          %14 = loom.alloc [32, 32, 128] on @L1 : memref<32x32x128xf32>
          %15 = loom.alloc [32, 32, 32] on @L1 : memref<32x32x32xf32>
          %16 = loom.alloc [32, 128, 32] on @L1 : memref<32x128x32xf32>
          %17 = loom.alloc [32, 32] on @L1 : memref<32x32xf32>
          %18 = loom.alloc [32, 32] on @L1 : memref<32x32xf32>
          %19 = loom.alloc [32, 32] on @L1 : memref<32x32xf32>
          %20 = loom.alloc [32, 32] on @L1 : memref<32x32xf32>
          %21 = loom.alloc [32, 32] on @L1 : memref<32x32xf32>
          %22 = loom.alloc [32, 32, 128] on @L1 : memref<32x32x128xf32>
          %23 = loom.alloc [32, 32, 128] on @L1 : memref<32x32x128xf32>
          %24 = loom.alloc [32, 32, 128] on @L1 : memref<32x32x128xf32>
          %25 = arith.muli %arg4, %c16777216 : index
          %26 = arith.muli %13, %c4096 : index
          %27 = arith.addi %25, %26 : index
          %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%27], sizes: [32, 32, 128], strides: [524288, 128, 1] : memref<32x4096x128xf32> to memref<32x32x128xf32, strided<[524288, 128, 1], offset: ?>>
          loom.copy %reinterpret_cast, %22 src_mem_space @DRAM dst_mem_space @L1, interconnect : [], broadcast : [1, 1] : memref<32x32x128xf32, strided<[524288, 128, 1], offset: ?>>, memref<32x32x128xf32>
          linalg.fill ins(%cst_1 : f32) outs(%23 : memref<32x32x128xf32>)
          linalg.fill ins(%cst_2 : f32) outs(%17 : memref<32x32xf32>)
          linalg.fill ins(%cst_3 : f32) outs(%18 : memref<32x32xf32>)
          scf.for %arg7 = %c0 to %c128 step %c1 {
            %28 = arith.muli %arg7, %c32 : index
            %29 = arith.addi %25, %28 : index
            %reinterpret_cast_5 = memref.reinterpret_cast %arg0 to offset: [%29], sizes: [32, 128, 32], strides: [524288, 4096, 1] : memref<32x128x4096xf32> to memref<32x128x32xf32, strided<[524288, 4096, 1], offset: ?>>
            loom.copy %reinterpret_cast_5, %16 src_mem_space @DRAM dst_mem_space @L1, interconnect : [], broadcast : [1, 1] : memref<32x128x32xf32, strided<[524288, 4096, 1], offset: ?>>, memref<32x128x32xf32>
            linalg.fill ins(%cst_1 : f32) outs(%15 : memref<32x32x32xf32>)
            linalg.batch_matmul ins(%22, %16 : memref<32x32x128xf32>, memref<32x128x32xf32>) outs(%15 : memref<32x32x32xf32>)
            linalg.fill ins(%cst_3 : f32) outs(%19 : memref<32x32xf32>)
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%15 : memref<32x32x32xf32>) outs(%19 : memref<32x32xf32>) {
            ^bb0(%in: f32, %out: f32):
              %32 = arith.maximumf %in, %out : f32
              linalg.yield %32 : f32
            }
            linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%18, %19 : memref<32x32xf32>, memref<32x32xf32>) outs(%19 : memref<32x32xf32>) {
            ^bb0(%in: f32, %in_7: f32, %out: f32):
              %32 = arith.truncf %cst_0 : f64 to f32
              %33 = arith.mulf %in_7, %32 : f32
              %34 = arith.cmpf ogt, %in, %33 : f32
              %35 = arith.select %34, %in, %33 : f32
              linalg.yield %35 : f32
            }
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%15, %19 : memref<32x32x32xf32>, memref<32x32xf32>) outs(%15 : memref<32x32x32xf32>) {
            ^bb0(%in: f32, %in_7: f32, %out: f32):
              %32 = arith.truncf %cst_0 : f64 to f32
              %33 = arith.mulf %in, %32 : f32
              %34 = arith.subf %33, %in_7 : f32
              %35 = math.powf %cst, %34 : f32
              linalg.yield %35 : f32
            }
            linalg.fill ins(%cst_1 : f32) outs(%20 : memref<32x32xf32>)
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%15 : memref<32x32x32xf32>) outs(%20 : memref<32x32xf32>) {
            ^bb0(%in: f32, %out: f32):
              %32 = arith.addf %in, %out : f32
              linalg.yield %32 : f32
            }
            linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%18, %19 : memref<32x32xf32>, memref<32x32xf32>) outs(%21 : memref<32x32xf32>) {
            ^bb0(%in: f32, %in_7: f32, %out: f32):
              %32 = arith.subf %in, %in_7 : f32
              %33 = math.powf %cst, %32 : f32
              linalg.yield %33 : f32
            }
            linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%17, %21, %20 : memref<32x32xf32>, memref<32x32xf32>, memref<32x32xf32>) outs(%17 : memref<32x32xf32>) {
            ^bb0(%in: f32, %in_7: f32, %in_8: f32, %out: f32):
              %32 = arith.mulf %in, %in_7 : f32
              %33 = arith.addf %32, %in_8 : f32
              linalg.yield %33 : f32
            }
            %30 = arith.muli %arg7, %c4096 : index
            %31 = arith.addi %25, %30 : index
            %reinterpret_cast_6 = memref.reinterpret_cast %arg1 to offset: [%31], sizes: [32, 32, 128], strides: [524288, 128, 1] : memref<32x4096x128xf32> to memref<32x32x128xf32, strided<[524288, 128, 1], offset: ?>>
            loom.copy %reinterpret_cast_6, %14 src_mem_space @DRAM dst_mem_space @L1, interconnect : [@vertical_links], broadcast : [8, 1] : memref<32x32x128xf32, strided<[524288, 128, 1], offset: ?>>, memref<32x32x128xf32>
            linalg.fill ins(%cst_1 : f32) outs(%24 : memref<32x32x128xf32>)
            linalg.batch_matmul ins(%15, %14 : memref<32x32x32xf32>, memref<32x32x128xf32>) outs(%24 : memref<32x32x128xf32>)
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%24, %23, %21 : memref<32x32x128xf32>, memref<32x32x128xf32>, memref<32x32xf32>) outs(%23 : memref<32x32x128xf32>) {
            ^bb0(%in: f32, %in_7: f32, %in_8: f32, %out: f32):
              %32 = arith.mulf %in_7, %in_8 : f32
              %33 = arith.addf %in, %32 : f32
              linalg.yield %33 : f32
            }
            linalg.copy ins(%19 : memref<32x32xf32>) outs(%18 : memref<32x32xf32>)
          }
          linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%23, %17 : memref<32x32x128xf32>, memref<32x32xf32>) outs(%22 : memref<32x32x128xf32>) {
          ^bb0(%in: f32, %in_5: f32, %out: f32):
            %28 = arith.divf %in, %in_5 : f32
            linalg.yield %28 : f32
          }
          %reinterpret_cast_4 = memref.reinterpret_cast %arg3 to offset: [%27], sizes: [32, 32, 128], strides: [524288, 128, 1] : memref<32x4096x128xf32> to memref<32x32x128xf32, strided<[524288, 128, 1], offset: ?>>
          loom.copy %22, %reinterpret_cast_4 src_mem_space @L1 dst_mem_space @DRAM, interconnect : [], broadcast : [1, 1] : memref<32x32x128xf32>, memref<32x32x128xf32, strided<[524288, 128, 1], offset: ?>>
        }
        scf.reduce 
      } {loom.iter_types = [#loom.iter_type<spatial>, #loom.iter_type<spatial>], loom.mapped_to_dims = [@x, @y]}
      return
    }
  }
  module attributes {loom.pass_name = "Materialize"} {
    func.func @attention__d0i0_d1i1__f01__d_v_d__BB32__BM32__BN32(%arg0: memref<32x128x4096xf32>, %arg1: memref<32x4096x128xf32>, %arg2: memref<32x4096x128xf32>, %arg3: memref<32x4096x128xf32>) {
      %c4096 = arith.constant 4096 : index
      %c16777216 = arith.constant 16777216 : index
      %c128 = arith.constant 128 : index
      %c16 = arith.constant 16 : index
      %cst = arith.constant 2.000000e+00 : f32
      %cst_0 = arith.constant 0.12751743074602467 : f64
      %cst_1 = arith.constant 0.000000e+00 : f32
      %cst_2 = arith.constant 1.000000e+00 : f32
      %cst_3 = arith.constant 0xFF800000 : f32
      %c32 = arith.constant 32 : index
      %c0 = arith.constant 0 : index
      %c8 = arith.constant 8 : index
      %c1 = arith.constant 1 : index
      scf.parallel (%arg4, %arg5) = (%c0, %c0) to (%c8, %c8) step (%c1, %c1) {
        scf.for %arg6 = %c0 to %c16 step %c1 {
          %12 = arith.muli %arg6, %c8 overflow<nsw> : index
          %13 = arith.addi %arg5, %12 : index
          %14 = loom.alloc [32, 32, 128] on @L1 : memref<32x32x128xf32>
          %15 = loom.alloc [32, 32, 32] on @L1 : memref<32x32x32xf32>
          %16 = loom.alloc [32, 128, 32] on @L1 : memref<32x128x32xf32>
          %17 = loom.alloc [32, 32] on @L1 : memref<32x32xf32>
          %18 = loom.alloc [32, 32] on @L1 : memref<32x32xf32>
          %19 = loom.alloc [32, 32] on @L1 : memref<32x32xf32>
          %20 = loom.alloc [32, 32] on @L1 : memref<32x32xf32>
          %21 = loom.alloc [32, 32] on @L1 : memref<32x32xf32>
          %22 = loom.alloc [32, 32, 128] on @L1 : memref<32x32x128xf32>
          %23 = loom.alloc [32, 32, 128] on @L1 : memref<32x32x128xf32>
          %24 = loom.alloc [32, 32, 128] on @L1 : memref<32x32x128xf32>
          %25 = arith.muli %arg4, %c16777216 : index
          %26 = arith.muli %13, %c4096 : index
          %27 = arith.addi %25, %26 : index
          %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%27], sizes: [32, 32, 128], strides: [524288, 128, 1] : memref<32x4096x128xf32> to memref<32x32x128xf32, strided<[524288, 128, 1], offset: ?>>
          loom.copy %reinterpret_cast, %22 src_mem_space @DRAM dst_mem_space @L1, interconnect : [], broadcast : [1, 1] : memref<32x32x128xf32, strided<[524288, 128, 1], offset: ?>>, memref<32x32x128xf32>
          linalg.fill ins(%cst_1 : f32) outs(%23 : memref<32x32x128xf32>)
          linalg.fill ins(%cst_2 : f32) outs(%17 : memref<32x32xf32>)
          linalg.fill ins(%cst_3 : f32) outs(%18 : memref<32x32xf32>)
          scf.for %arg7 = %c0 to %c128 step %c1 {
            %28 = arith.muli %arg7, %c32 : index
            %29 = arith.addi %25, %28 : index
            %reinterpret_cast_5 = memref.reinterpret_cast %arg0 to offset: [%29], sizes: [32, 128, 32], strides: [524288, 4096, 1] : memref<32x128x4096xf32> to memref<32x128x32xf32, strided<[524288, 4096, 1], offset: ?>>
            loom.copy %reinterpret_cast_5, %16 src_mem_space @DRAM dst_mem_space @L1, interconnect : [@vertical_links], broadcast : [8, 1] : memref<32x128x32xf32, strided<[524288, 4096, 1], offset: ?>>, memref<32x128x32xf32>
            linalg.fill ins(%cst_1 : f32) outs(%15 : memref<32x32x32xf32>)
            linalg.batch_matmul ins(%22, %16 : memref<32x32x128xf32>, memref<32x128x32xf32>) outs(%15 : memref<32x32x32xf32>)
            linalg.fill ins(%cst_3 : f32) outs(%19 : memref<32x32xf32>)
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%15 : memref<32x32x32xf32>) outs(%19 : memref<32x32xf32>) {
            ^bb0(%in: f32, %out: f32):
              %32 = arith.maximumf %in, %out : f32
              linalg.yield %32 : f32
            }
            linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%18, %19 : memref<32x32xf32>, memref<32x32xf32>) outs(%19 : memref<32x32xf32>) {
            ^bb0(%in: f32, %in_7: f32, %out: f32):
              %32 = arith.truncf %cst_0 : f64 to f32
              %33 = arith.mulf %in_7, %32 : f32
              %34 = arith.cmpf ogt, %in, %33 : f32
              %35 = arith.select %34, %in, %33 : f32
              linalg.yield %35 : f32
            }
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%15, %19 : memref<32x32x32xf32>, memref<32x32xf32>) outs(%15 : memref<32x32x32xf32>) {
            ^bb0(%in: f32, %in_7: f32, %out: f32):
              %32 = arith.truncf %cst_0 : f64 to f32
              %33 = arith.mulf %in, %32 : f32
              %34 = arith.subf %33, %in_7 : f32
              %35 = math.powf %cst, %34 : f32
              linalg.yield %35 : f32
            }
            linalg.fill ins(%cst_1 : f32) outs(%20 : memref<32x32xf32>)
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%15 : memref<32x32x32xf32>) outs(%20 : memref<32x32xf32>) {
            ^bb0(%in: f32, %out: f32):
              %32 = arith.addf %in, %out : f32
              linalg.yield %32 : f32
            }
            linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%18, %19 : memref<32x32xf32>, memref<32x32xf32>) outs(%21 : memref<32x32xf32>) {
            ^bb0(%in: f32, %in_7: f32, %out: f32):
              %32 = arith.subf %in, %in_7 : f32
              %33 = math.powf %cst, %32 : f32
              linalg.yield %33 : f32
            }
            linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%17, %21, %20 : memref<32x32xf32>, memref<32x32xf32>, memref<32x32xf32>) outs(%17 : memref<32x32xf32>) {
            ^bb0(%in: f32, %in_7: f32, %in_8: f32, %out: f32):
              %32 = arith.mulf %in, %in_7 : f32
              %33 = arith.addf %32, %in_8 : f32
              linalg.yield %33 : f32
            }
            %30 = arith.muli %arg7, %c4096 : index
            %31 = arith.addi %25, %30 : index
            %reinterpret_cast_6 = memref.reinterpret_cast %arg1 to offset: [%31], sizes: [32, 32, 128], strides: [524288, 128, 1] : memref<32x4096x128xf32> to memref<32x32x128xf32, strided<[524288, 128, 1], offset: ?>>
            loom.copy %reinterpret_cast_6, %14 src_mem_space @DRAM dst_mem_space @L1, interconnect : [], broadcast : [1, 1] : memref<32x32x128xf32, strided<[524288, 128, 1], offset: ?>>, memref<32x32x128xf32>
            linalg.fill ins(%cst_1 : f32) outs(%24 : memref<32x32x128xf32>)
            linalg.batch_matmul ins(%15, %14 : memref<32x32x32xf32>, memref<32x32x128xf32>) outs(%24 : memref<32x32x128xf32>)
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%24, %23, %21 : memref<32x32x128xf32>, memref<32x32x128xf32>, memref<32x32xf32>) outs(%23 : memref<32x32x128xf32>) {
            ^bb0(%in: f32, %in_7: f32, %in_8: f32, %out: f32):
              %32 = arith.mulf %in_7, %in_8 : f32
              %33 = arith.addf %in, %32 : f32
              linalg.yield %33 : f32
            }
            linalg.copy ins(%19 : memref<32x32xf32>) outs(%18 : memref<32x32xf32>)
          }
          linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%23, %17 : memref<32x32x128xf32>, memref<32x32xf32>) outs(%22 : memref<32x32x128xf32>) {
          ^bb0(%in: f32, %in_5: f32, %out: f32):
            %28 = arith.divf %in, %in_5 : f32
            linalg.yield %28 : f32
          }
          %reinterpret_cast_4 = memref.reinterpret_cast %arg3 to offset: [%27], sizes: [32, 32, 128], strides: [524288, 128, 1] : memref<32x4096x128xf32> to memref<32x32x128xf32, strided<[524288, 128, 1], offset: ?>>
          loom.copy %22, %reinterpret_cast_4 src_mem_space @L1 dst_mem_space @DRAM, interconnect : [], broadcast : [1, 1] : memref<32x32x128xf32>, memref<32x32x128xf32, strided<[524288, 128, 1], offset: ?>>
        }
        scf.reduce 
      } {loom.iter_types = [#loom.iter_type<spatial>, #loom.iter_type<spatial>], loom.mapped_to_dims = [@x, @y]}
      return
    }
  }
  module attributes {loom.pass_name = "Materialize"} {
    func.func @attention__d0i0_d1i1__f01__d_v_v__BB32__BM32__BN32(%arg0: memref<32x128x4096xf32>, %arg1: memref<32x4096x128xf32>, %arg2: memref<32x4096x128xf32>, %arg3: memref<32x4096x128xf32>) {
      %c4096 = arith.constant 4096 : index
      %c16777216 = arith.constant 16777216 : index
      %c128 = arith.constant 128 : index
      %c16 = arith.constant 16 : index
      %cst = arith.constant 2.000000e+00 : f32
      %cst_0 = arith.constant 0.12751743074602467 : f64
      %cst_1 = arith.constant 0.000000e+00 : f32
      %cst_2 = arith.constant 1.000000e+00 : f32
      %cst_3 = arith.constant 0xFF800000 : f32
      %c32 = arith.constant 32 : index
      %c0 = arith.constant 0 : index
      %c8 = arith.constant 8 : index
      %c1 = arith.constant 1 : index
      scf.parallel (%arg4, %arg5) = (%c0, %c0) to (%c8, %c8) step (%c1, %c1) {
        scf.for %arg6 = %c0 to %c16 step %c1 {
          %12 = arith.muli %arg6, %c8 overflow<nsw> : index
          %13 = arith.addi %arg5, %12 : index
          %14 = loom.alloc [32, 32, 128] on @L1 : memref<32x32x128xf32>
          %15 = loom.alloc [32, 32, 32] on @L1 : memref<32x32x32xf32>
          %16 = loom.alloc [32, 128, 32] on @L1 : memref<32x128x32xf32>
          %17 = loom.alloc [32, 32] on @L1 : memref<32x32xf32>
          %18 = loom.alloc [32, 32] on @L1 : memref<32x32xf32>
          %19 = loom.alloc [32, 32] on @L1 : memref<32x32xf32>
          %20 = loom.alloc [32, 32] on @L1 : memref<32x32xf32>
          %21 = loom.alloc [32, 32] on @L1 : memref<32x32xf32>
          %22 = loom.alloc [32, 32, 128] on @L1 : memref<32x32x128xf32>
          %23 = loom.alloc [32, 32, 128] on @L1 : memref<32x32x128xf32>
          %24 = loom.alloc [32, 32, 128] on @L1 : memref<32x32x128xf32>
          %25 = arith.muli %arg4, %c16777216 : index
          %26 = arith.muli %13, %c4096 : index
          %27 = arith.addi %25, %26 : index
          %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%27], sizes: [32, 32, 128], strides: [524288, 128, 1] : memref<32x4096x128xf32> to memref<32x32x128xf32, strided<[524288, 128, 1], offset: ?>>
          loom.copy %reinterpret_cast, %22 src_mem_space @DRAM dst_mem_space @L1, interconnect : [], broadcast : [1, 1] : memref<32x32x128xf32, strided<[524288, 128, 1], offset: ?>>, memref<32x32x128xf32>
          linalg.fill ins(%cst_1 : f32) outs(%23 : memref<32x32x128xf32>)
          linalg.fill ins(%cst_2 : f32) outs(%17 : memref<32x32xf32>)
          linalg.fill ins(%cst_3 : f32) outs(%18 : memref<32x32xf32>)
          scf.for %arg7 = %c0 to %c128 step %c1 {
            %28 = arith.muli %arg7, %c32 : index
            %29 = arith.addi %25, %28 : index
            %reinterpret_cast_5 = memref.reinterpret_cast %arg0 to offset: [%29], sizes: [32, 128, 32], strides: [524288, 4096, 1] : memref<32x128x4096xf32> to memref<32x128x32xf32, strided<[524288, 4096, 1], offset: ?>>
            loom.copy %reinterpret_cast_5, %16 src_mem_space @DRAM dst_mem_space @L1, interconnect : [@vertical_links], broadcast : [8, 1] : memref<32x128x32xf32, strided<[524288, 4096, 1], offset: ?>>, memref<32x128x32xf32>
            linalg.fill ins(%cst_1 : f32) outs(%15 : memref<32x32x32xf32>)
            linalg.batch_matmul ins(%22, %16 : memref<32x32x128xf32>, memref<32x128x32xf32>) outs(%15 : memref<32x32x32xf32>)
            linalg.fill ins(%cst_3 : f32) outs(%19 : memref<32x32xf32>)
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%15 : memref<32x32x32xf32>) outs(%19 : memref<32x32xf32>) {
            ^bb0(%in: f32, %out: f32):
              %32 = arith.maximumf %in, %out : f32
              linalg.yield %32 : f32
            }
            linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%18, %19 : memref<32x32xf32>, memref<32x32xf32>) outs(%19 : memref<32x32xf32>) {
            ^bb0(%in: f32, %in_7: f32, %out: f32):
              %32 = arith.truncf %cst_0 : f64 to f32
              %33 = arith.mulf %in_7, %32 : f32
              %34 = arith.cmpf ogt, %in, %33 : f32
              %35 = arith.select %34, %in, %33 : f32
              linalg.yield %35 : f32
            }
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%15, %19 : memref<32x32x32xf32>, memref<32x32xf32>) outs(%15 : memref<32x32x32xf32>) {
            ^bb0(%in: f32, %in_7: f32, %out: f32):
              %32 = arith.truncf %cst_0 : f64 to f32
              %33 = arith.mulf %in, %32 : f32
              %34 = arith.subf %33, %in_7 : f32
              %35 = math.powf %cst, %34 : f32
              linalg.yield %35 : f32
            }
            linalg.fill ins(%cst_1 : f32) outs(%20 : memref<32x32xf32>)
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%15 : memref<32x32x32xf32>) outs(%20 : memref<32x32xf32>) {
            ^bb0(%in: f32, %out: f32):
              %32 = arith.addf %in, %out : f32
              linalg.yield %32 : f32
            }
            linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%18, %19 : memref<32x32xf32>, memref<32x32xf32>) outs(%21 : memref<32x32xf32>) {
            ^bb0(%in: f32, %in_7: f32, %out: f32):
              %32 = arith.subf %in, %in_7 : f32
              %33 = math.powf %cst, %32 : f32
              linalg.yield %33 : f32
            }
            linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%17, %21, %20 : memref<32x32xf32>, memref<32x32xf32>, memref<32x32xf32>) outs(%17 : memref<32x32xf32>) {
            ^bb0(%in: f32, %in_7: f32, %in_8: f32, %out: f32):
              %32 = arith.mulf %in, %in_7 : f32
              %33 = arith.addf %32, %in_8 : f32
              linalg.yield %33 : f32
            }
            %30 = arith.muli %arg7, %c4096 : index
            %31 = arith.addi %25, %30 : index
            %reinterpret_cast_6 = memref.reinterpret_cast %arg1 to offset: [%31], sizes: [32, 32, 128], strides: [524288, 128, 1] : memref<32x4096x128xf32> to memref<32x32x128xf32, strided<[524288, 128, 1], offset: ?>>
            loom.copy %reinterpret_cast_6, %14 src_mem_space @DRAM dst_mem_space @L1, interconnect : [@vertical_links], broadcast : [8, 1] : memref<32x32x128xf32, strided<[524288, 128, 1], offset: ?>>, memref<32x32x128xf32>
            linalg.fill ins(%cst_1 : f32) outs(%24 : memref<32x32x128xf32>)
            linalg.batch_matmul ins(%15, %14 : memref<32x32x32xf32>, memref<32x32x128xf32>) outs(%24 : memref<32x32x128xf32>)
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%24, %23, %21 : memref<32x32x128xf32>, memref<32x32x128xf32>, memref<32x32xf32>) outs(%23 : memref<32x32x128xf32>) {
            ^bb0(%in: f32, %in_7: f32, %in_8: f32, %out: f32):
              %32 = arith.mulf %in_7, %in_8 : f32
              %33 = arith.addf %in, %32 : f32
              linalg.yield %33 : f32
            }
            linalg.copy ins(%19 : memref<32x32xf32>) outs(%18 : memref<32x32xf32>)
          }
          linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%23, %17 : memref<32x32x128xf32>, memref<32x32xf32>) outs(%22 : memref<32x32x128xf32>) {
          ^bb0(%in: f32, %in_5: f32, %out: f32):
            %28 = arith.divf %in, %in_5 : f32
            linalg.yield %28 : f32
          }
          %reinterpret_cast_4 = memref.reinterpret_cast %arg3 to offset: [%27], sizes: [32, 32, 128], strides: [524288, 128, 1] : memref<32x4096x128xf32> to memref<32x32x128xf32, strided<[524288, 128, 1], offset: ?>>
          loom.copy %22, %reinterpret_cast_4 src_mem_space @L1 dst_mem_space @DRAM, interconnect : [], broadcast : [1, 1] : memref<32x32x128xf32>, memref<32x32x128xf32, strided<[524288, 128, 1], offset: ?>>
        }
        scf.reduce 
      } {loom.iter_types = [#loom.iter_type<spatial>, #loom.iter_type<spatial>], loom.mapped_to_dims = [@x, @y]}
      return
    }
  }
  module attributes {loom.pass_name = "Materialize"} {
    func.func @attention__d0i0_d1i1__f10__d_d_d__BB32__BM32__BN32(%arg0: memref<32x128x4096xf32>, %arg1: memref<32x4096x128xf32>, %arg2: memref<32x4096x128xf32>, %arg3: memref<32x4096x128xf32>) {
      %c4096 = arith.constant 4096 : index
      %c16777216 = arith.constant 16777216 : index
      %c128 = arith.constant 128 : index
      %c16 = arith.constant 16 : index
      %cst = arith.constant 2.000000e+00 : f32
      %cst_0 = arith.constant 0.12751743074602467 : f64
      %cst_1 = arith.constant 0.000000e+00 : f32
      %cst_2 = arith.constant 1.000000e+00 : f32
      %cst_3 = arith.constant 0xFF800000 : f32
      %c32 = arith.constant 32 : index
      %c0 = arith.constant 0 : index
      %c8 = arith.constant 8 : index
      %c1 = arith.constant 1 : index
      scf.parallel (%arg4, %arg5) = (%c0, %c0) to (%c8, %c8) step (%c1, %c1) {
        scf.for %arg6 = %c0 to %c16 step %c1 {
          %12 = arith.muli %arg6, %c8 overflow<nsw> : index
          %13 = arith.addi %arg5, %12 : index
          %14 = loom.alloc [32, 32, 128] on @L1 : memref<32x32x128xf32>
          %15 = loom.alloc [32, 32, 32] on @L1 : memref<32x32x32xf32>
          %16 = loom.alloc [32, 128, 32] on @L1 : memref<32x128x32xf32>
          %17 = loom.alloc [32, 32] on @L1 : memref<32x32xf32>
          %18 = loom.alloc [32, 32] on @L1 : memref<32x32xf32>
          %19 = loom.alloc [32, 32] on @L1 : memref<32x32xf32>
          %20 = loom.alloc [32, 32] on @L1 : memref<32x32xf32>
          %21 = loom.alloc [32, 32] on @L1 : memref<32x32xf32>
          %22 = loom.alloc [32, 32, 128] on @L1 : memref<32x32x128xf32>
          %23 = loom.alloc [32, 32, 128] on @L1 : memref<32x32x128xf32>
          %24 = loom.alloc [32, 32, 128] on @L1 : memref<32x32x128xf32>
          %25 = arith.muli %arg4, %c16777216 : index
          %26 = arith.muli %13, %c4096 : index
          %27 = arith.addi %25, %26 : index
          %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%27], sizes: [32, 32, 128], strides: [524288, 128, 1] : memref<32x4096x128xf32> to memref<32x32x128xf32, strided<[524288, 128, 1], offset: ?>>
          loom.copy %reinterpret_cast, %22 src_mem_space @DRAM dst_mem_space @L1, interconnect : [], broadcast : [1, 1] : memref<32x32x128xf32, strided<[524288, 128, 1], offset: ?>>, memref<32x32x128xf32>
          linalg.fill ins(%cst_1 : f32) outs(%23 : memref<32x32x128xf32>)
          linalg.fill ins(%cst_2 : f32) outs(%17 : memref<32x32xf32>)
          linalg.fill ins(%cst_3 : f32) outs(%18 : memref<32x32xf32>)
          scf.for %arg7 = %c0 to %c128 step %c1 {
            %28 = arith.muli %arg7, %c32 : index
            %29 = arith.addi %25, %28 : index
            %reinterpret_cast_5 = memref.reinterpret_cast %arg0 to offset: [%29], sizes: [32, 128, 32], strides: [524288, 4096, 1] : memref<32x128x4096xf32> to memref<32x128x32xf32, strided<[524288, 4096, 1], offset: ?>>
            loom.copy %reinterpret_cast_5, %16 src_mem_space @DRAM dst_mem_space @L1, interconnect : [], broadcast : [1, 1] : memref<32x128x32xf32, strided<[524288, 4096, 1], offset: ?>>, memref<32x128x32xf32>
            linalg.fill ins(%cst_1 : f32) outs(%15 : memref<32x32x32xf32>)
            linalg.batch_matmul ins(%22, %16 : memref<32x32x128xf32>, memref<32x128x32xf32>) outs(%15 : memref<32x32x32xf32>)
            linalg.fill ins(%cst_3 : f32) outs(%19 : memref<32x32xf32>)
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%15 : memref<32x32x32xf32>) outs(%19 : memref<32x32xf32>) {
            ^bb0(%in: f32, %out: f32):
              %32 = arith.maximumf %in, %out : f32
              linalg.yield %32 : f32
            }
            linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%18, %19 : memref<32x32xf32>, memref<32x32xf32>) outs(%19 : memref<32x32xf32>) {
            ^bb0(%in: f32, %in_7: f32, %out: f32):
              %32 = arith.truncf %cst_0 : f64 to f32
              %33 = arith.mulf %in_7, %32 : f32
              %34 = arith.cmpf ogt, %in, %33 : f32
              %35 = arith.select %34, %in, %33 : f32
              linalg.yield %35 : f32
            }
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%15, %19 : memref<32x32x32xf32>, memref<32x32xf32>) outs(%15 : memref<32x32x32xf32>) {
            ^bb0(%in: f32, %in_7: f32, %out: f32):
              %32 = arith.truncf %cst_0 : f64 to f32
              %33 = arith.mulf %in, %32 : f32
              %34 = arith.subf %33, %in_7 : f32
              %35 = math.powf %cst, %34 : f32
              linalg.yield %35 : f32
            }
            linalg.fill ins(%cst_1 : f32) outs(%20 : memref<32x32xf32>)
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%15 : memref<32x32x32xf32>) outs(%20 : memref<32x32xf32>) {
            ^bb0(%in: f32, %out: f32):
              %32 = arith.addf %in, %out : f32
              linalg.yield %32 : f32
            }
            linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%18, %19 : memref<32x32xf32>, memref<32x32xf32>) outs(%21 : memref<32x32xf32>) {
            ^bb0(%in: f32, %in_7: f32, %out: f32):
              %32 = arith.subf %in, %in_7 : f32
              %33 = math.powf %cst, %32 : f32
              linalg.yield %33 : f32
            }
            linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%17, %21, %20 : memref<32x32xf32>, memref<32x32xf32>, memref<32x32xf32>) outs(%17 : memref<32x32xf32>) {
            ^bb0(%in: f32, %in_7: f32, %in_8: f32, %out: f32):
              %32 = arith.mulf %in, %in_7 : f32
              %33 = arith.addf %32, %in_8 : f32
              linalg.yield %33 : f32
            }
            %30 = arith.muli %arg7, %c4096 : index
            %31 = arith.addi %25, %30 : index
            %reinterpret_cast_6 = memref.reinterpret_cast %arg1 to offset: [%31], sizes: [32, 32, 128], strides: [524288, 128, 1] : memref<32x4096x128xf32> to memref<32x32x128xf32, strided<[524288, 128, 1], offset: ?>>
            loom.copy %reinterpret_cast_6, %14 src_mem_space @DRAM dst_mem_space @L1, interconnect : [], broadcast : [1, 1] : memref<32x32x128xf32, strided<[524288, 128, 1], offset: ?>>, memref<32x32x128xf32>
            linalg.fill ins(%cst_1 : f32) outs(%24 : memref<32x32x128xf32>)
            linalg.batch_matmul ins(%15, %14 : memref<32x32x32xf32>, memref<32x32x128xf32>) outs(%24 : memref<32x32x128xf32>)
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%24, %23, %21 : memref<32x32x128xf32>, memref<32x32x128xf32>, memref<32x32xf32>) outs(%23 : memref<32x32x128xf32>) {
            ^bb0(%in: f32, %in_7: f32, %in_8: f32, %out: f32):
              %32 = arith.mulf %in_7, %in_8 : f32
              %33 = arith.addf %in, %32 : f32
              linalg.yield %33 : f32
            }
            linalg.copy ins(%19 : memref<32x32xf32>) outs(%18 : memref<32x32xf32>)
          }
          linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%23, %17 : memref<32x32x128xf32>, memref<32x32xf32>) outs(%22 : memref<32x32x128xf32>) {
          ^bb0(%in: f32, %in_5: f32, %out: f32):
            %28 = arith.divf %in, %in_5 : f32
            linalg.yield %28 : f32
          }
          %reinterpret_cast_4 = memref.reinterpret_cast %arg3 to offset: [%27], sizes: [32, 32, 128], strides: [524288, 128, 1] : memref<32x4096x128xf32> to memref<32x32x128xf32, strided<[524288, 128, 1], offset: ?>>
          loom.copy %22, %reinterpret_cast_4 src_mem_space @L1 dst_mem_space @DRAM, interconnect : [], broadcast : [1, 1] : memref<32x32x128xf32>, memref<32x32x128xf32, strided<[524288, 128, 1], offset: ?>>
        }
        scf.reduce 
      } {loom.iter_types = [#loom.iter_type<spatial>, #loom.iter_type<spatial>], loom.mapped_to_dims = [@x, @y]}
      return
    }
  }
  module attributes {loom.pass_name = "Materialize"} {
    func.func @attention__d0i0_d1i1__f10__d_d_v__BB32__BM32__BN32(%arg0: memref<32x128x4096xf32>, %arg1: memref<32x4096x128xf32>, %arg2: memref<32x4096x128xf32>, %arg3: memref<32x4096x128xf32>) {
      %c4096 = arith.constant 4096 : index
      %c16777216 = arith.constant 16777216 : index
      %c128 = arith.constant 128 : index
      %c16 = arith.constant 16 : index
      %cst = arith.constant 2.000000e+00 : f32
      %cst_0 = arith.constant 0.12751743074602467 : f64
      %cst_1 = arith.constant 0.000000e+00 : f32
      %cst_2 = arith.constant 1.000000e+00 : f32
      %cst_3 = arith.constant 0xFF800000 : f32
      %c32 = arith.constant 32 : index
      %c0 = arith.constant 0 : index
      %c8 = arith.constant 8 : index
      %c1 = arith.constant 1 : index
      scf.parallel (%arg4, %arg5) = (%c0, %c0) to (%c8, %c8) step (%c1, %c1) {
        scf.for %arg6 = %c0 to %c16 step %c1 {
          %12 = arith.muli %arg6, %c8 overflow<nsw> : index
          %13 = arith.addi %arg5, %12 : index
          %14 = loom.alloc [32, 32, 128] on @L1 : memref<32x32x128xf32>
          %15 = loom.alloc [32, 32, 32] on @L1 : memref<32x32x32xf32>
          %16 = loom.alloc [32, 128, 32] on @L1 : memref<32x128x32xf32>
          %17 = loom.alloc [32, 32] on @L1 : memref<32x32xf32>
          %18 = loom.alloc [32, 32] on @L1 : memref<32x32xf32>
          %19 = loom.alloc [32, 32] on @L1 : memref<32x32xf32>
          %20 = loom.alloc [32, 32] on @L1 : memref<32x32xf32>
          %21 = loom.alloc [32, 32] on @L1 : memref<32x32xf32>
          %22 = loom.alloc [32, 32, 128] on @L1 : memref<32x32x128xf32>
          %23 = loom.alloc [32, 32, 128] on @L1 : memref<32x32x128xf32>
          %24 = loom.alloc [32, 32, 128] on @L1 : memref<32x32x128xf32>
          %25 = arith.muli %arg4, %c16777216 : index
          %26 = arith.muli %13, %c4096 : index
          %27 = arith.addi %25, %26 : index
          %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%27], sizes: [32, 32, 128], strides: [524288, 128, 1] : memref<32x4096x128xf32> to memref<32x32x128xf32, strided<[524288, 128, 1], offset: ?>>
          loom.copy %reinterpret_cast, %22 src_mem_space @DRAM dst_mem_space @L1, interconnect : [], broadcast : [1, 1] : memref<32x32x128xf32, strided<[524288, 128, 1], offset: ?>>, memref<32x32x128xf32>
          linalg.fill ins(%cst_1 : f32) outs(%23 : memref<32x32x128xf32>)
          linalg.fill ins(%cst_2 : f32) outs(%17 : memref<32x32xf32>)
          linalg.fill ins(%cst_3 : f32) outs(%18 : memref<32x32xf32>)
          scf.for %arg7 = %c0 to %c128 step %c1 {
            %28 = arith.muli %arg7, %c32 : index
            %29 = arith.addi %25, %28 : index
            %reinterpret_cast_5 = memref.reinterpret_cast %arg0 to offset: [%29], sizes: [32, 128, 32], strides: [524288, 4096, 1] : memref<32x128x4096xf32> to memref<32x128x32xf32, strided<[524288, 4096, 1], offset: ?>>
            loom.copy %reinterpret_cast_5, %16 src_mem_space @DRAM dst_mem_space @L1, interconnect : [], broadcast : [1, 1] : memref<32x128x32xf32, strided<[524288, 4096, 1], offset: ?>>, memref<32x128x32xf32>
            linalg.fill ins(%cst_1 : f32) outs(%15 : memref<32x32x32xf32>)
            linalg.batch_matmul ins(%22, %16 : memref<32x32x128xf32>, memref<32x128x32xf32>) outs(%15 : memref<32x32x32xf32>)
            linalg.fill ins(%cst_3 : f32) outs(%19 : memref<32x32xf32>)
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%15 : memref<32x32x32xf32>) outs(%19 : memref<32x32xf32>) {
            ^bb0(%in: f32, %out: f32):
              %32 = arith.maximumf %in, %out : f32
              linalg.yield %32 : f32
            }
            linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%18, %19 : memref<32x32xf32>, memref<32x32xf32>) outs(%19 : memref<32x32xf32>) {
            ^bb0(%in: f32, %in_7: f32, %out: f32):
              %32 = arith.truncf %cst_0 : f64 to f32
              %33 = arith.mulf %in_7, %32 : f32
              %34 = arith.cmpf ogt, %in, %33 : f32
              %35 = arith.select %34, %in, %33 : f32
              linalg.yield %35 : f32
            }
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%15, %19 : memref<32x32x32xf32>, memref<32x32xf32>) outs(%15 : memref<32x32x32xf32>) {
            ^bb0(%in: f32, %in_7: f32, %out: f32):
              %32 = arith.truncf %cst_0 : f64 to f32
              %33 = arith.mulf %in, %32 : f32
              %34 = arith.subf %33, %in_7 : f32
              %35 = math.powf %cst, %34 : f32
              linalg.yield %35 : f32
            }
            linalg.fill ins(%cst_1 : f32) outs(%20 : memref<32x32xf32>)
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%15 : memref<32x32x32xf32>) outs(%20 : memref<32x32xf32>) {
            ^bb0(%in: f32, %out: f32):
              %32 = arith.addf %in, %out : f32
              linalg.yield %32 : f32
            }
            linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%18, %19 : memref<32x32xf32>, memref<32x32xf32>) outs(%21 : memref<32x32xf32>) {
            ^bb0(%in: f32, %in_7: f32, %out: f32):
              %32 = arith.subf %in, %in_7 : f32
              %33 = math.powf %cst, %32 : f32
              linalg.yield %33 : f32
            }
            linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%17, %21, %20 : memref<32x32xf32>, memref<32x32xf32>, memref<32x32xf32>) outs(%17 : memref<32x32xf32>) {
            ^bb0(%in: f32, %in_7: f32, %in_8: f32, %out: f32):
              %32 = arith.mulf %in, %in_7 : f32
              %33 = arith.addf %32, %in_8 : f32
              linalg.yield %33 : f32
            }
            %30 = arith.muli %arg7, %c4096 : index
            %31 = arith.addi %25, %30 : index
            %reinterpret_cast_6 = memref.reinterpret_cast %arg1 to offset: [%31], sizes: [32, 32, 128], strides: [524288, 128, 1] : memref<32x4096x128xf32> to memref<32x32x128xf32, strided<[524288, 128, 1], offset: ?>>
            loom.copy %reinterpret_cast_6, %14 src_mem_space @DRAM dst_mem_space @L1, interconnect : [@vertical_links], broadcast : [8, 1] : memref<32x32x128xf32, strided<[524288, 128, 1], offset: ?>>, memref<32x32x128xf32>
            linalg.fill ins(%cst_1 : f32) outs(%24 : memref<32x32x128xf32>)
            linalg.batch_matmul ins(%15, %14 : memref<32x32x32xf32>, memref<32x32x128xf32>) outs(%24 : memref<32x32x128xf32>)
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%24, %23, %21 : memref<32x32x128xf32>, memref<32x32x128xf32>, memref<32x32xf32>) outs(%23 : memref<32x32x128xf32>) {
            ^bb0(%in: f32, %in_7: f32, %in_8: f32, %out: f32):
              %32 = arith.mulf %in_7, %in_8 : f32
              %33 = arith.addf %in, %32 : f32
              linalg.yield %33 : f32
            }
            linalg.copy ins(%19 : memref<32x32xf32>) outs(%18 : memref<32x32xf32>)
          }
          linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%23, %17 : memref<32x32x128xf32>, memref<32x32xf32>) outs(%22 : memref<32x32x128xf32>) {
          ^bb0(%in: f32, %in_5: f32, %out: f32):
            %28 = arith.divf %in, %in_5 : f32
            linalg.yield %28 : f32
          }
          %reinterpret_cast_4 = memref.reinterpret_cast %arg3 to offset: [%27], sizes: [32, 32, 128], strides: [524288, 128, 1] : memref<32x4096x128xf32> to memref<32x32x128xf32, strided<[524288, 128, 1], offset: ?>>
          loom.copy %22, %reinterpret_cast_4 src_mem_space @L1 dst_mem_space @DRAM, interconnect : [], broadcast : [1, 1] : memref<32x32x128xf32>, memref<32x32x128xf32, strided<[524288, 128, 1], offset: ?>>
        }
        scf.reduce 
      } {loom.iter_types = [#loom.iter_type<spatial>, #loom.iter_type<spatial>], loom.mapped_to_dims = [@x, @y]}
      return
    }
  }
  module attributes {loom.pass_name = "Materialize"} {
    func.func @attention__d0i0_d1i1__f10__d_v_d__BB32__BM32__BN32(%arg0: memref<32x128x4096xf32>, %arg1: memref<32x4096x128xf32>, %arg2: memref<32x4096x128xf32>, %arg3: memref<32x4096x128xf32>) {
      %c4096 = arith.constant 4096 : index
      %c16777216 = arith.constant 16777216 : index
      %c128 = arith.constant 128 : index
      %c16 = arith.constant 16 : index
      %cst = arith.constant 2.000000e+00 : f32
      %cst_0 = arith.constant 0.12751743074602467 : f64
      %cst_1 = arith.constant 0.000000e+00 : f32
      %cst_2 = arith.constant 1.000000e+00 : f32
      %cst_3 = arith.constant 0xFF800000 : f32
      %c32 = arith.constant 32 : index
      %c0 = arith.constant 0 : index
      %c8 = arith.constant 8 : index
      %c1 = arith.constant 1 : index
      scf.parallel (%arg4, %arg5) = (%c0, %c0) to (%c8, %c8) step (%c1, %c1) {
        scf.for %arg6 = %c0 to %c16 step %c1 {
          %12 = arith.muli %arg6, %c8 overflow<nsw> : index
          %13 = arith.addi %arg5, %12 : index
          %14 = loom.alloc [32, 32, 128] on @L1 : memref<32x32x128xf32>
          %15 = loom.alloc [32, 32, 32] on @L1 : memref<32x32x32xf32>
          %16 = loom.alloc [32, 128, 32] on @L1 : memref<32x128x32xf32>
          %17 = loom.alloc [32, 32] on @L1 : memref<32x32xf32>
          %18 = loom.alloc [32, 32] on @L1 : memref<32x32xf32>
          %19 = loom.alloc [32, 32] on @L1 : memref<32x32xf32>
          %20 = loom.alloc [32, 32] on @L1 : memref<32x32xf32>
          %21 = loom.alloc [32, 32] on @L1 : memref<32x32xf32>
          %22 = loom.alloc [32, 32, 128] on @L1 : memref<32x32x128xf32>
          %23 = loom.alloc [32, 32, 128] on @L1 : memref<32x32x128xf32>
          %24 = loom.alloc [32, 32, 128] on @L1 : memref<32x32x128xf32>
          %25 = arith.muli %arg4, %c16777216 : index
          %26 = arith.muli %13, %c4096 : index
          %27 = arith.addi %25, %26 : index
          %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%27], sizes: [32, 32, 128], strides: [524288, 128, 1] : memref<32x4096x128xf32> to memref<32x32x128xf32, strided<[524288, 128, 1], offset: ?>>
          loom.copy %reinterpret_cast, %22 src_mem_space @DRAM dst_mem_space @L1, interconnect : [], broadcast : [1, 1] : memref<32x32x128xf32, strided<[524288, 128, 1], offset: ?>>, memref<32x32x128xf32>
          linalg.fill ins(%cst_1 : f32) outs(%23 : memref<32x32x128xf32>)
          linalg.fill ins(%cst_2 : f32) outs(%17 : memref<32x32xf32>)
          linalg.fill ins(%cst_3 : f32) outs(%18 : memref<32x32xf32>)
          scf.for %arg7 = %c0 to %c128 step %c1 {
            %28 = arith.muli %arg7, %c32 : index
            %29 = arith.addi %25, %28 : index
            %reinterpret_cast_5 = memref.reinterpret_cast %arg0 to offset: [%29], sizes: [32, 128, 32], strides: [524288, 4096, 1] : memref<32x128x4096xf32> to memref<32x128x32xf32, strided<[524288, 4096, 1], offset: ?>>
            loom.copy %reinterpret_cast_5, %16 src_mem_space @DRAM dst_mem_space @L1, interconnect : [@vertical_links], broadcast : [8, 1] : memref<32x128x32xf32, strided<[524288, 4096, 1], offset: ?>>, memref<32x128x32xf32>
            linalg.fill ins(%cst_1 : f32) outs(%15 : memref<32x32x32xf32>)
            linalg.batch_matmul ins(%22, %16 : memref<32x32x128xf32>, memref<32x128x32xf32>) outs(%15 : memref<32x32x32xf32>)
            linalg.fill ins(%cst_3 : f32) outs(%19 : memref<32x32xf32>)
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%15 : memref<32x32x32xf32>) outs(%19 : memref<32x32xf32>) {
            ^bb0(%in: f32, %out: f32):
              %32 = arith.maximumf %in, %out : f32
              linalg.yield %32 : f32
            }
            linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%18, %19 : memref<32x32xf32>, memref<32x32xf32>) outs(%19 : memref<32x32xf32>) {
            ^bb0(%in: f32, %in_7: f32, %out: f32):
              %32 = arith.truncf %cst_0 : f64 to f32
              %33 = arith.mulf %in_7, %32 : f32
              %34 = arith.cmpf ogt, %in, %33 : f32
              %35 = arith.select %34, %in, %33 : f32
              linalg.yield %35 : f32
            }
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%15, %19 : memref<32x32x32xf32>, memref<32x32xf32>) outs(%15 : memref<32x32x32xf32>) {
            ^bb0(%in: f32, %in_7: f32, %out: f32):
              %32 = arith.truncf %cst_0 : f64 to f32
              %33 = arith.mulf %in, %32 : f32
              %34 = arith.subf %33, %in_7 : f32
              %35 = math.powf %cst, %34 : f32
              linalg.yield %35 : f32
            }
            linalg.fill ins(%cst_1 : f32) outs(%20 : memref<32x32xf32>)
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%15 : memref<32x32x32xf32>) outs(%20 : memref<32x32xf32>) {
            ^bb0(%in: f32, %out: f32):
              %32 = arith.addf %in, %out : f32
              linalg.yield %32 : f32
            }
            linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%18, %19 : memref<32x32xf32>, memref<32x32xf32>) outs(%21 : memref<32x32xf32>) {
            ^bb0(%in: f32, %in_7: f32, %out: f32):
              %32 = arith.subf %in, %in_7 : f32
              %33 = math.powf %cst, %32 : f32
              linalg.yield %33 : f32
            }
            linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%17, %21, %20 : memref<32x32xf32>, memref<32x32xf32>, memref<32x32xf32>) outs(%17 : memref<32x32xf32>) {
            ^bb0(%in: f32, %in_7: f32, %in_8: f32, %out: f32):
              %32 = arith.mulf %in, %in_7 : f32
              %33 = arith.addf %32, %in_8 : f32
              linalg.yield %33 : f32
            }
            %30 = arith.muli %arg7, %c4096 : index
            %31 = arith.addi %25, %30 : index
            %reinterpret_cast_6 = memref.reinterpret_cast %arg1 to offset: [%31], sizes: [32, 32, 128], strides: [524288, 128, 1] : memref<32x4096x128xf32> to memref<32x32x128xf32, strided<[524288, 128, 1], offset: ?>>
            loom.copy %reinterpret_cast_6, %14 src_mem_space @DRAM dst_mem_space @L1, interconnect : [], broadcast : [1, 1] : memref<32x32x128xf32, strided<[524288, 128, 1], offset: ?>>, memref<32x32x128xf32>
            linalg.fill ins(%cst_1 : f32) outs(%24 : memref<32x32x128xf32>)
            linalg.batch_matmul ins(%15, %14 : memref<32x32x32xf32>, memref<32x32x128xf32>) outs(%24 : memref<32x32x128xf32>)
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%24, %23, %21 : memref<32x32x128xf32>, memref<32x32x128xf32>, memref<32x32xf32>) outs(%23 : memref<32x32x128xf32>) {
            ^bb0(%in: f32, %in_7: f32, %in_8: f32, %out: f32):
              %32 = arith.mulf %in_7, %in_8 : f32
              %33 = arith.addf %in, %32 : f32
              linalg.yield %33 : f32
            }
            linalg.copy ins(%19 : memref<32x32xf32>) outs(%18 : memref<32x32xf32>)
          }
          linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%23, %17 : memref<32x32x128xf32>, memref<32x32xf32>) outs(%22 : memref<32x32x128xf32>) {
          ^bb0(%in: f32, %in_5: f32, %out: f32):
            %28 = arith.divf %in, %in_5 : f32
            linalg.yield %28 : f32
          }
          %reinterpret_cast_4 = memref.reinterpret_cast %arg3 to offset: [%27], sizes: [32, 32, 128], strides: [524288, 128, 1] : memref<32x4096x128xf32> to memref<32x32x128xf32, strided<[524288, 128, 1], offset: ?>>
          loom.copy %22, %reinterpret_cast_4 src_mem_space @L1 dst_mem_space @DRAM, interconnect : [], broadcast : [1, 1] : memref<32x32x128xf32>, memref<32x32x128xf32, strided<[524288, 128, 1], offset: ?>>
        }
        scf.reduce 
      } {loom.iter_types = [#loom.iter_type<spatial>, #loom.iter_type<spatial>], loom.mapped_to_dims = [@x, @y]}
      return
    }
  }
  module attributes {loom.pass_name = "Materialize"} {
    func.func @attention__d0i0_d1i1__f10__d_v_v__BB32__BM32__BN32(%arg0: memref<32x128x4096xf32>, %arg1: memref<32x4096x128xf32>, %arg2: memref<32x4096x128xf32>, %arg3: memref<32x4096x128xf32>) {
      %c4096 = arith.constant 4096 : index
      %c16777216 = arith.constant 16777216 : index
      %c128 = arith.constant 128 : index
      %c16 = arith.constant 16 : index
      %cst = arith.constant 2.000000e+00 : f32
      %cst_0 = arith.constant 0.12751743074602467 : f64
      %cst_1 = arith.constant 0.000000e+00 : f32
      %cst_2 = arith.constant 1.000000e+00 : f32
      %cst_3 = arith.constant 0xFF800000 : f32
      %c32 = arith.constant 32 : index
      %c0 = arith.constant 0 : index
      %c8 = arith.constant 8 : index
      %c1 = arith.constant 1 : index
      scf.parallel (%arg4, %arg5) = (%c0, %c0) to (%c8, %c8) step (%c1, %c1) {
        scf.for %arg6 = %c0 to %c16 step %c1 {
          %12 = arith.muli %arg6, %c8 overflow<nsw> : index
          %13 = arith.addi %arg5, %12 : index
          %14 = loom.alloc [32, 32, 128] on @L1 : memref<32x32x128xf32>
          %15 = loom.alloc [32, 32, 32] on @L1 : memref<32x32x32xf32>
          %16 = loom.alloc [32, 128, 32] on @L1 : memref<32x128x32xf32>
          %17 = loom.alloc [32, 32] on @L1 : memref<32x32xf32>
          %18 = loom.alloc [32, 32] on @L1 : memref<32x32xf32>
          %19 = loom.alloc [32, 32] on @L1 : memref<32x32xf32>
          %20 = loom.alloc [32, 32] on @L1 : memref<32x32xf32>
          %21 = loom.alloc [32, 32] on @L1 : memref<32x32xf32>
          %22 = loom.alloc [32, 32, 128] on @L1 : memref<32x32x128xf32>
          %23 = loom.alloc [32, 32, 128] on @L1 : memref<32x32x128xf32>
          %24 = loom.alloc [32, 32, 128] on @L1 : memref<32x32x128xf32>
          %25 = arith.muli %arg4, %c16777216 : index
          %26 = arith.muli %13, %c4096 : index
          %27 = arith.addi %25, %26 : index
          %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%27], sizes: [32, 32, 128], strides: [524288, 128, 1] : memref<32x4096x128xf32> to memref<32x32x128xf32, strided<[524288, 128, 1], offset: ?>>
          loom.copy %reinterpret_cast, %22 src_mem_space @DRAM dst_mem_space @L1, interconnect : [], broadcast : [1, 1] : memref<32x32x128xf32, strided<[524288, 128, 1], offset: ?>>, memref<32x32x128xf32>
          linalg.fill ins(%cst_1 : f32) outs(%23 : memref<32x32x128xf32>)
          linalg.fill ins(%cst_2 : f32) outs(%17 : memref<32x32xf32>)
          linalg.fill ins(%cst_3 : f32) outs(%18 : memref<32x32xf32>)
          scf.for %arg7 = %c0 to %c128 step %c1 {
            %28 = arith.muli %arg7, %c32 : index
            %29 = arith.addi %25, %28 : index
            %reinterpret_cast_5 = memref.reinterpret_cast %arg0 to offset: [%29], sizes: [32, 128, 32], strides: [524288, 4096, 1] : memref<32x128x4096xf32> to memref<32x128x32xf32, strided<[524288, 4096, 1], offset: ?>>
            loom.copy %reinterpret_cast_5, %16 src_mem_space @DRAM dst_mem_space @L1, interconnect : [@vertical_links], broadcast : [8, 1] : memref<32x128x32xf32, strided<[524288, 4096, 1], offset: ?>>, memref<32x128x32xf32>
            linalg.fill ins(%cst_1 : f32) outs(%15 : memref<32x32x32xf32>)
            linalg.batch_matmul ins(%22, %16 : memref<32x32x128xf32>, memref<32x128x32xf32>) outs(%15 : memref<32x32x32xf32>)
            linalg.fill ins(%cst_3 : f32) outs(%19 : memref<32x32xf32>)
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%15 : memref<32x32x32xf32>) outs(%19 : memref<32x32xf32>) {
            ^bb0(%in: f32, %out: f32):
              %32 = arith.maximumf %in, %out : f32
              linalg.yield %32 : f32
            }
            linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%18, %19 : memref<32x32xf32>, memref<32x32xf32>) outs(%19 : memref<32x32xf32>) {
            ^bb0(%in: f32, %in_7: f32, %out: f32):
              %32 = arith.truncf %cst_0 : f64 to f32
              %33 = arith.mulf %in_7, %32 : f32
              %34 = arith.cmpf ogt, %in, %33 : f32
              %35 = arith.select %34, %in, %33 : f32
              linalg.yield %35 : f32
            }
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%15, %19 : memref<32x32x32xf32>, memref<32x32xf32>) outs(%15 : memref<32x32x32xf32>) {
            ^bb0(%in: f32, %in_7: f32, %out: f32):
              %32 = arith.truncf %cst_0 : f64 to f32
              %33 = arith.mulf %in, %32 : f32
              %34 = arith.subf %33, %in_7 : f32
              %35 = math.powf %cst, %34 : f32
              linalg.yield %35 : f32
            }
            linalg.fill ins(%cst_1 : f32) outs(%20 : memref<32x32xf32>)
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%15 : memref<32x32x32xf32>) outs(%20 : memref<32x32xf32>) {
            ^bb0(%in: f32, %out: f32):
              %32 = arith.addf %in, %out : f32
              linalg.yield %32 : f32
            }
            linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%18, %19 : memref<32x32xf32>, memref<32x32xf32>) outs(%21 : memref<32x32xf32>) {
            ^bb0(%in: f32, %in_7: f32, %out: f32):
              %32 = arith.subf %in, %in_7 : f32
              %33 = math.powf %cst, %32 : f32
              linalg.yield %33 : f32
            }
            linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%17, %21, %20 : memref<32x32xf32>, memref<32x32xf32>, memref<32x32xf32>) outs(%17 : memref<32x32xf32>) {
            ^bb0(%in: f32, %in_7: f32, %in_8: f32, %out: f32):
              %32 = arith.mulf %in, %in_7 : f32
              %33 = arith.addf %32, %in_8 : f32
              linalg.yield %33 : f32
            }
            %30 = arith.muli %arg7, %c4096 : index
            %31 = arith.addi %25, %30 : index
            %reinterpret_cast_6 = memref.reinterpret_cast %arg1 to offset: [%31], sizes: [32, 32, 128], strides: [524288, 128, 1] : memref<32x4096x128xf32> to memref<32x32x128xf32, strided<[524288, 128, 1], offset: ?>>
            loom.copy %reinterpret_cast_6, %14 src_mem_space @DRAM dst_mem_space @L1, interconnect : [@vertical_links], broadcast : [8, 1] : memref<32x32x128xf32, strided<[524288, 128, 1], offset: ?>>, memref<32x32x128xf32>
            linalg.fill ins(%cst_1 : f32) outs(%24 : memref<32x32x128xf32>)
            linalg.batch_matmul ins(%15, %14 : memref<32x32x32xf32>, memref<32x32x128xf32>) outs(%24 : memref<32x32x128xf32>)
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%24, %23, %21 : memref<32x32x128xf32>, memref<32x32x128xf32>, memref<32x32xf32>) outs(%23 : memref<32x32x128xf32>) {
            ^bb0(%in: f32, %in_7: f32, %in_8: f32, %out: f32):
              %32 = arith.mulf %in_7, %in_8 : f32
              %33 = arith.addf %in, %32 : f32
              linalg.yield %33 : f32
            }
            linalg.copy ins(%19 : memref<32x32xf32>) outs(%18 : memref<32x32xf32>)
          }
          linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%23, %17 : memref<32x32x128xf32>, memref<32x32xf32>) outs(%22 : memref<32x32x128xf32>) {
          ^bb0(%in: f32, %in_5: f32, %out: f32):
            %28 = arith.divf %in, %in_5 : f32
            linalg.yield %28 : f32
          }
          %reinterpret_cast_4 = memref.reinterpret_cast %arg3 to offset: [%27], sizes: [32, 32, 128], strides: [524288, 128, 1] : memref<32x4096x128xf32> to memref<32x32x128xf32, strided<[524288, 128, 1], offset: ?>>
          loom.copy %22, %reinterpret_cast_4 src_mem_space @L1 dst_mem_space @DRAM, interconnect : [], broadcast : [1, 1] : memref<32x32x128xf32>, memref<32x32x128xf32, strided<[524288, 128, 1], offset: ?>>
        }
        scf.reduce 
      } {loom.iter_types = [#loom.iter_type<spatial>, #loom.iter_type<spatial>], loom.mapped_to_dims = [@x, @y]}
      return
    }
  }
  module attributes {loom.pass_name = "Materialize"} {
    func.func @attention__d1i0_d0i1__f01__d_d_d__BB32__BM32__BN32(%arg0: memref<32x128x4096xf32>, %arg1: memref<32x4096x128xf32>, %arg2: memref<32x4096x128xf32>, %arg3: memref<32x4096x128xf32>) {
      %c4096 = arith.constant 4096 : index
      %c16777216 = arith.constant 16777216 : index
      %c128 = arith.constant 128 : index
      %c16 = arith.constant 16 : index
      %cst = arith.constant 2.000000e+00 : f32
      %cst_0 = arith.constant 0.12751743074602467 : f64
      %cst_1 = arith.constant 0.000000e+00 : f32
      %cst_2 = arith.constant 1.000000e+00 : f32
      %cst_3 = arith.constant 0xFF800000 : f32
      %c32 = arith.constant 32 : index
      %c0 = arith.constant 0 : index
      %c8 = arith.constant 8 : index
      %c1 = arith.constant 1 : index
      scf.parallel (%arg4, %arg5) = (%c0, %c0) to (%c8, %c8) step (%c1, %c1) {
        scf.for %arg6 = %c0 to %c16 step %c1 {
          %12 = arith.muli %arg6, %c8 overflow<nsw> : index
          %13 = arith.addi %arg5, %12 : index
          %14 = loom.alloc [32, 32, 128] on @L1 : memref<32x32x128xf32>
          %15 = loom.alloc [32, 32, 32] on @L1 : memref<32x32x32xf32>
          %16 = loom.alloc [32, 128, 32] on @L1 : memref<32x128x32xf32>
          %17 = loom.alloc [32, 32] on @L1 : memref<32x32xf32>
          %18 = loom.alloc [32, 32] on @L1 : memref<32x32xf32>
          %19 = loom.alloc [32, 32] on @L1 : memref<32x32xf32>
          %20 = loom.alloc [32, 32] on @L1 : memref<32x32xf32>
          %21 = loom.alloc [32, 32] on @L1 : memref<32x32xf32>
          %22 = loom.alloc [32, 32, 128] on @L1 : memref<32x32x128xf32>
          %23 = loom.alloc [32, 32, 128] on @L1 : memref<32x32x128xf32>
          %24 = loom.alloc [32, 32, 128] on @L1 : memref<32x32x128xf32>
          %25 = arith.muli %arg4, %c16777216 : index
          %26 = arith.muli %13, %c4096 : index
          %27 = arith.addi %25, %26 : index
          %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%27], sizes: [32, 32, 128], strides: [524288, 128, 1] : memref<32x4096x128xf32> to memref<32x32x128xf32, strided<[524288, 128, 1], offset: ?>>
          loom.copy %reinterpret_cast, %22 src_mem_space @DRAM dst_mem_space @L1, interconnect : [], broadcast : [1, 1] : memref<32x32x128xf32, strided<[524288, 128, 1], offset: ?>>, memref<32x32x128xf32>
          linalg.fill ins(%cst_1 : f32) outs(%23 : memref<32x32x128xf32>)
          linalg.fill ins(%cst_2 : f32) outs(%17 : memref<32x32xf32>)
          linalg.fill ins(%cst_3 : f32) outs(%18 : memref<32x32xf32>)
          scf.for %arg7 = %c0 to %c128 step %c1 {
            %28 = arith.muli %arg7, %c32 : index
            %29 = arith.addi %25, %28 : index
            %reinterpret_cast_5 = memref.reinterpret_cast %arg0 to offset: [%29], sizes: [32, 128, 32], strides: [524288, 4096, 1] : memref<32x128x4096xf32> to memref<32x128x32xf32, strided<[524288, 4096, 1], offset: ?>>
            loom.copy %reinterpret_cast_5, %16 src_mem_space @DRAM dst_mem_space @L1, interconnect : [], broadcast : [1, 1] : memref<32x128x32xf32, strided<[524288, 4096, 1], offset: ?>>, memref<32x128x32xf32>
            linalg.fill ins(%cst_1 : f32) outs(%15 : memref<32x32x32xf32>)
            linalg.batch_matmul ins(%22, %16 : memref<32x32x128xf32>, memref<32x128x32xf32>) outs(%15 : memref<32x32x32xf32>)
            linalg.fill ins(%cst_3 : f32) outs(%19 : memref<32x32xf32>)
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%15 : memref<32x32x32xf32>) outs(%19 : memref<32x32xf32>) {
            ^bb0(%in: f32, %out: f32):
              %32 = arith.maximumf %in, %out : f32
              linalg.yield %32 : f32
            }
            linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%18, %19 : memref<32x32xf32>, memref<32x32xf32>) outs(%19 : memref<32x32xf32>) {
            ^bb0(%in: f32, %in_7: f32, %out: f32):
              %32 = arith.truncf %cst_0 : f64 to f32
              %33 = arith.mulf %in_7, %32 : f32
              %34 = arith.cmpf ogt, %in, %33 : f32
              %35 = arith.select %34, %in, %33 : f32
              linalg.yield %35 : f32
            }
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%15, %19 : memref<32x32x32xf32>, memref<32x32xf32>) outs(%15 : memref<32x32x32xf32>) {
            ^bb0(%in: f32, %in_7: f32, %out: f32):
              %32 = arith.truncf %cst_0 : f64 to f32
              %33 = arith.mulf %in, %32 : f32
              %34 = arith.subf %33, %in_7 : f32
              %35 = math.powf %cst, %34 : f32
              linalg.yield %35 : f32
            }
            linalg.fill ins(%cst_1 : f32) outs(%20 : memref<32x32xf32>)
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%15 : memref<32x32x32xf32>) outs(%20 : memref<32x32xf32>) {
            ^bb0(%in: f32, %out: f32):
              %32 = arith.addf %in, %out : f32
              linalg.yield %32 : f32
            }
            linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%18, %19 : memref<32x32xf32>, memref<32x32xf32>) outs(%21 : memref<32x32xf32>) {
            ^bb0(%in: f32, %in_7: f32, %out: f32):
              %32 = arith.subf %in, %in_7 : f32
              %33 = math.powf %cst, %32 : f32
              linalg.yield %33 : f32
            }
            linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%17, %21, %20 : memref<32x32xf32>, memref<32x32xf32>, memref<32x32xf32>) outs(%17 : memref<32x32xf32>) {
            ^bb0(%in: f32, %in_7: f32, %in_8: f32, %out: f32):
              %32 = arith.mulf %in, %in_7 : f32
              %33 = arith.addf %32, %in_8 : f32
              linalg.yield %33 : f32
            }
            %30 = arith.muli %arg7, %c4096 : index
            %31 = arith.addi %25, %30 : index
            %reinterpret_cast_6 = memref.reinterpret_cast %arg1 to offset: [%31], sizes: [32, 32, 128], strides: [524288, 128, 1] : memref<32x4096x128xf32> to memref<32x32x128xf32, strided<[524288, 128, 1], offset: ?>>
            loom.copy %reinterpret_cast_6, %14 src_mem_space @DRAM dst_mem_space @L1, interconnect : [], broadcast : [1, 1] : memref<32x32x128xf32, strided<[524288, 128, 1], offset: ?>>, memref<32x32x128xf32>
            linalg.fill ins(%cst_1 : f32) outs(%24 : memref<32x32x128xf32>)
            linalg.batch_matmul ins(%15, %14 : memref<32x32x32xf32>, memref<32x32x128xf32>) outs(%24 : memref<32x32x128xf32>)
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%24, %23, %21 : memref<32x32x128xf32>, memref<32x32x128xf32>, memref<32x32xf32>) outs(%23 : memref<32x32x128xf32>) {
            ^bb0(%in: f32, %in_7: f32, %in_8: f32, %out: f32):
              %32 = arith.mulf %in_7, %in_8 : f32
              %33 = arith.addf %in, %32 : f32
              linalg.yield %33 : f32
            }
            linalg.copy ins(%19 : memref<32x32xf32>) outs(%18 : memref<32x32xf32>)
          }
          linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%23, %17 : memref<32x32x128xf32>, memref<32x32xf32>) outs(%22 : memref<32x32x128xf32>) {
          ^bb0(%in: f32, %in_5: f32, %out: f32):
            %28 = arith.divf %in, %in_5 : f32
            linalg.yield %28 : f32
          }
          %reinterpret_cast_4 = memref.reinterpret_cast %arg3 to offset: [%27], sizes: [32, 32, 128], strides: [524288, 128, 1] : memref<32x4096x128xf32> to memref<32x32x128xf32, strided<[524288, 128, 1], offset: ?>>
          loom.copy %22, %reinterpret_cast_4 src_mem_space @L1 dst_mem_space @DRAM, interconnect : [], broadcast : [1, 1] : memref<32x32x128xf32>, memref<32x32x128xf32, strided<[524288, 128, 1], offset: ?>>
        }
        scf.reduce 
      } {loom.iter_types = [#loom.iter_type<spatial>, #loom.iter_type<spatial>], loom.mapped_to_dims = [@y, @x]}
      return
    }
  }
  module attributes {loom.pass_name = "Materialize"} {
    func.func @attention__d1i0_d0i1__f01__d_d_h__BB32__BM32__BN32(%arg0: memref<32x128x4096xf32>, %arg1: memref<32x4096x128xf32>, %arg2: memref<32x4096x128xf32>, %arg3: memref<32x4096x128xf32>) {
      %c4096 = arith.constant 4096 : index
      %c16777216 = arith.constant 16777216 : index
      %c128 = arith.constant 128 : index
      %c16 = arith.constant 16 : index
      %cst = arith.constant 2.000000e+00 : f32
      %cst_0 = arith.constant 0.12751743074602467 : f64
      %cst_1 = arith.constant 0.000000e+00 : f32
      %cst_2 = arith.constant 1.000000e+00 : f32
      %cst_3 = arith.constant 0xFF800000 : f32
      %c32 = arith.constant 32 : index
      %c0 = arith.constant 0 : index
      %c8 = arith.constant 8 : index
      %c1 = arith.constant 1 : index
      scf.parallel (%arg4, %arg5) = (%c0, %c0) to (%c8, %c8) step (%c1, %c1) {
        scf.for %arg6 = %c0 to %c16 step %c1 {
          %12 = arith.muli %arg6, %c8 overflow<nsw> : index
          %13 = arith.addi %arg5, %12 : index
          %14 = loom.alloc [32, 32, 128] on @L1 : memref<32x32x128xf32>
          %15 = loom.alloc [32, 32, 32] on @L1 : memref<32x32x32xf32>
          %16 = loom.alloc [32, 128, 32] on @L1 : memref<32x128x32xf32>
          %17 = loom.alloc [32, 32] on @L1 : memref<32x32xf32>
          %18 = loom.alloc [32, 32] on @L1 : memref<32x32xf32>
          %19 = loom.alloc [32, 32] on @L1 : memref<32x32xf32>
          %20 = loom.alloc [32, 32] on @L1 : memref<32x32xf32>
          %21 = loom.alloc [32, 32] on @L1 : memref<32x32xf32>
          %22 = loom.alloc [32, 32, 128] on @L1 : memref<32x32x128xf32>
          %23 = loom.alloc [32, 32, 128] on @L1 : memref<32x32x128xf32>
          %24 = loom.alloc [32, 32, 128] on @L1 : memref<32x32x128xf32>
          %25 = arith.muli %arg4, %c16777216 : index
          %26 = arith.muli %13, %c4096 : index
          %27 = arith.addi %25, %26 : index
          %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%27], sizes: [32, 32, 128], strides: [524288, 128, 1] : memref<32x4096x128xf32> to memref<32x32x128xf32, strided<[524288, 128, 1], offset: ?>>
          loom.copy %reinterpret_cast, %22 src_mem_space @DRAM dst_mem_space @L1, interconnect : [], broadcast : [1, 1] : memref<32x32x128xf32, strided<[524288, 128, 1], offset: ?>>, memref<32x32x128xf32>
          linalg.fill ins(%cst_1 : f32) outs(%23 : memref<32x32x128xf32>)
          linalg.fill ins(%cst_2 : f32) outs(%17 : memref<32x32xf32>)
          linalg.fill ins(%cst_3 : f32) outs(%18 : memref<32x32xf32>)
          scf.for %arg7 = %c0 to %c128 step %c1 {
            %28 = arith.muli %arg7, %c32 : index
            %29 = arith.addi %25, %28 : index
            %reinterpret_cast_5 = memref.reinterpret_cast %arg0 to offset: [%29], sizes: [32, 128, 32], strides: [524288, 4096, 1] : memref<32x128x4096xf32> to memref<32x128x32xf32, strided<[524288, 4096, 1], offset: ?>>
            loom.copy %reinterpret_cast_5, %16 src_mem_space @DRAM dst_mem_space @L1, interconnect : [], broadcast : [1, 1] : memref<32x128x32xf32, strided<[524288, 4096, 1], offset: ?>>, memref<32x128x32xf32>
            linalg.fill ins(%cst_1 : f32) outs(%15 : memref<32x32x32xf32>)
            linalg.batch_matmul ins(%22, %16 : memref<32x32x128xf32>, memref<32x128x32xf32>) outs(%15 : memref<32x32x32xf32>)
            linalg.fill ins(%cst_3 : f32) outs(%19 : memref<32x32xf32>)
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%15 : memref<32x32x32xf32>) outs(%19 : memref<32x32xf32>) {
            ^bb0(%in: f32, %out: f32):
              %32 = arith.maximumf %in, %out : f32
              linalg.yield %32 : f32
            }
            linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%18, %19 : memref<32x32xf32>, memref<32x32xf32>) outs(%19 : memref<32x32xf32>) {
            ^bb0(%in: f32, %in_7: f32, %out: f32):
              %32 = arith.truncf %cst_0 : f64 to f32
              %33 = arith.mulf %in_7, %32 : f32
              %34 = arith.cmpf ogt, %in, %33 : f32
              %35 = arith.select %34, %in, %33 : f32
              linalg.yield %35 : f32
            }
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%15, %19 : memref<32x32x32xf32>, memref<32x32xf32>) outs(%15 : memref<32x32x32xf32>) {
            ^bb0(%in: f32, %in_7: f32, %out: f32):
              %32 = arith.truncf %cst_0 : f64 to f32
              %33 = arith.mulf %in, %32 : f32
              %34 = arith.subf %33, %in_7 : f32
              %35 = math.powf %cst, %34 : f32
              linalg.yield %35 : f32
            }
            linalg.fill ins(%cst_1 : f32) outs(%20 : memref<32x32xf32>)
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%15 : memref<32x32x32xf32>) outs(%20 : memref<32x32xf32>) {
            ^bb0(%in: f32, %out: f32):
              %32 = arith.addf %in, %out : f32
              linalg.yield %32 : f32
            }
            linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%18, %19 : memref<32x32xf32>, memref<32x32xf32>) outs(%21 : memref<32x32xf32>) {
            ^bb0(%in: f32, %in_7: f32, %out: f32):
              %32 = arith.subf %in, %in_7 : f32
              %33 = math.powf %cst, %32 : f32
              linalg.yield %33 : f32
            }
            linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%17, %21, %20 : memref<32x32xf32>, memref<32x32xf32>, memref<32x32xf32>) outs(%17 : memref<32x32xf32>) {
            ^bb0(%in: f32, %in_7: f32, %in_8: f32, %out: f32):
              %32 = arith.mulf %in, %in_7 : f32
              %33 = arith.addf %32, %in_8 : f32
              linalg.yield %33 : f32
            }
            %30 = arith.muli %arg7, %c4096 : index
            %31 = arith.addi %25, %30 : index
            %reinterpret_cast_6 = memref.reinterpret_cast %arg1 to offset: [%31], sizes: [32, 32, 128], strides: [524288, 128, 1] : memref<32x4096x128xf32> to memref<32x32x128xf32, strided<[524288, 128, 1], offset: ?>>
            loom.copy %reinterpret_cast_6, %14 src_mem_space @DRAM dst_mem_space @L1, interconnect : [@horizontal_links], broadcast : [1, 8] : memref<32x32x128xf32, strided<[524288, 128, 1], offset: ?>>, memref<32x32x128xf32>
            linalg.fill ins(%cst_1 : f32) outs(%24 : memref<32x32x128xf32>)
            linalg.batch_matmul ins(%15, %14 : memref<32x32x32xf32>, memref<32x32x128xf32>) outs(%24 : memref<32x32x128xf32>)
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%24, %23, %21 : memref<32x32x128xf32>, memref<32x32x128xf32>, memref<32x32xf32>) outs(%23 : memref<32x32x128xf32>) {
            ^bb0(%in: f32, %in_7: f32, %in_8: f32, %out: f32):
              %32 = arith.mulf %in_7, %in_8 : f32
              %33 = arith.addf %in, %32 : f32
              linalg.yield %33 : f32
            }
            linalg.copy ins(%19 : memref<32x32xf32>) outs(%18 : memref<32x32xf32>)
          }
          linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%23, %17 : memref<32x32x128xf32>, memref<32x32xf32>) outs(%22 : memref<32x32x128xf32>) {
          ^bb0(%in: f32, %in_5: f32, %out: f32):
            %28 = arith.divf %in, %in_5 : f32
            linalg.yield %28 : f32
          }
          %reinterpret_cast_4 = memref.reinterpret_cast %arg3 to offset: [%27], sizes: [32, 32, 128], strides: [524288, 128, 1] : memref<32x4096x128xf32> to memref<32x32x128xf32, strided<[524288, 128, 1], offset: ?>>
          loom.copy %22, %reinterpret_cast_4 src_mem_space @L1 dst_mem_space @DRAM, interconnect : [], broadcast : [1, 1] : memref<32x32x128xf32>, memref<32x32x128xf32, strided<[524288, 128, 1], offset: ?>>
        }
        scf.reduce 
      } {loom.iter_types = [#loom.iter_type<spatial>, #loom.iter_type<spatial>], loom.mapped_to_dims = [@y, @x]}
      return
    }
  }
  module attributes {loom.pass_name = "Materialize"} {
    func.func @attention__d1i0_d0i1__f01__d_h_d__BB32__BM32__BN32(%arg0: memref<32x128x4096xf32>, %arg1: memref<32x4096x128xf32>, %arg2: memref<32x4096x128xf32>, %arg3: memref<32x4096x128xf32>) {
      %c4096 = arith.constant 4096 : index
      %c16777216 = arith.constant 16777216 : index
      %c128 = arith.constant 128 : index
      %c16 = arith.constant 16 : index
      %cst = arith.constant 2.000000e+00 : f32
      %cst_0 = arith.constant 0.12751743074602467 : f64
      %cst_1 = arith.constant 0.000000e+00 : f32
      %cst_2 = arith.constant 1.000000e+00 : f32
      %cst_3 = arith.constant 0xFF800000 : f32
      %c32 = arith.constant 32 : index
      %c0 = arith.constant 0 : index
      %c8 = arith.constant 8 : index
      %c1 = arith.constant 1 : index
      scf.parallel (%arg4, %arg5) = (%c0, %c0) to (%c8, %c8) step (%c1, %c1) {
        scf.for %arg6 = %c0 to %c16 step %c1 {
          %12 = arith.muli %arg6, %c8 overflow<nsw> : index
          %13 = arith.addi %arg5, %12 : index
          %14 = loom.alloc [32, 32, 128] on @L1 : memref<32x32x128xf32>
          %15 = loom.alloc [32, 32, 32] on @L1 : memref<32x32x32xf32>
          %16 = loom.alloc [32, 128, 32] on @L1 : memref<32x128x32xf32>
          %17 = loom.alloc [32, 32] on @L1 : memref<32x32xf32>
          %18 = loom.alloc [32, 32] on @L1 : memref<32x32xf32>
          %19 = loom.alloc [32, 32] on @L1 : memref<32x32xf32>
          %20 = loom.alloc [32, 32] on @L1 : memref<32x32xf32>
          %21 = loom.alloc [32, 32] on @L1 : memref<32x32xf32>
          %22 = loom.alloc [32, 32, 128] on @L1 : memref<32x32x128xf32>
          %23 = loom.alloc [32, 32, 128] on @L1 : memref<32x32x128xf32>
          %24 = loom.alloc [32, 32, 128] on @L1 : memref<32x32x128xf32>
          %25 = arith.muli %arg4, %c16777216 : index
          %26 = arith.muli %13, %c4096 : index
          %27 = arith.addi %25, %26 : index
          %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%27], sizes: [32, 32, 128], strides: [524288, 128, 1] : memref<32x4096x128xf32> to memref<32x32x128xf32, strided<[524288, 128, 1], offset: ?>>
          loom.copy %reinterpret_cast, %22 src_mem_space @DRAM dst_mem_space @L1, interconnect : [], broadcast : [1, 1] : memref<32x32x128xf32, strided<[524288, 128, 1], offset: ?>>, memref<32x32x128xf32>
          linalg.fill ins(%cst_1 : f32) outs(%23 : memref<32x32x128xf32>)
          linalg.fill ins(%cst_2 : f32) outs(%17 : memref<32x32xf32>)
          linalg.fill ins(%cst_3 : f32) outs(%18 : memref<32x32xf32>)
          scf.for %arg7 = %c0 to %c128 step %c1 {
            %28 = arith.muli %arg7, %c32 : index
            %29 = arith.addi %25, %28 : index
            %reinterpret_cast_5 = memref.reinterpret_cast %arg0 to offset: [%29], sizes: [32, 128, 32], strides: [524288, 4096, 1] : memref<32x128x4096xf32> to memref<32x128x32xf32, strided<[524288, 4096, 1], offset: ?>>
            loom.copy %reinterpret_cast_5, %16 src_mem_space @DRAM dst_mem_space @L1, interconnect : [@horizontal_links], broadcast : [1, 8] : memref<32x128x32xf32, strided<[524288, 4096, 1], offset: ?>>, memref<32x128x32xf32>
            linalg.fill ins(%cst_1 : f32) outs(%15 : memref<32x32x32xf32>)
            linalg.batch_matmul ins(%22, %16 : memref<32x32x128xf32>, memref<32x128x32xf32>) outs(%15 : memref<32x32x32xf32>)
            linalg.fill ins(%cst_3 : f32) outs(%19 : memref<32x32xf32>)
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%15 : memref<32x32x32xf32>) outs(%19 : memref<32x32xf32>) {
            ^bb0(%in: f32, %out: f32):
              %32 = arith.maximumf %in, %out : f32
              linalg.yield %32 : f32
            }
            linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%18, %19 : memref<32x32xf32>, memref<32x32xf32>) outs(%19 : memref<32x32xf32>) {
            ^bb0(%in: f32, %in_7: f32, %out: f32):
              %32 = arith.truncf %cst_0 : f64 to f32
              %33 = arith.mulf %in_7, %32 : f32
              %34 = arith.cmpf ogt, %in, %33 : f32
              %35 = arith.select %34, %in, %33 : f32
              linalg.yield %35 : f32
            }
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%15, %19 : memref<32x32x32xf32>, memref<32x32xf32>) outs(%15 : memref<32x32x32xf32>) {
            ^bb0(%in: f32, %in_7: f32, %out: f32):
              %32 = arith.truncf %cst_0 : f64 to f32
              %33 = arith.mulf %in, %32 : f32
              %34 = arith.subf %33, %in_7 : f32
              %35 = math.powf %cst, %34 : f32
              linalg.yield %35 : f32
            }
            linalg.fill ins(%cst_1 : f32) outs(%20 : memref<32x32xf32>)
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%15 : memref<32x32x32xf32>) outs(%20 : memref<32x32xf32>) {
            ^bb0(%in: f32, %out: f32):
              %32 = arith.addf %in, %out : f32
              linalg.yield %32 : f32
            }
            linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%18, %19 : memref<32x32xf32>, memref<32x32xf32>) outs(%21 : memref<32x32xf32>) {
            ^bb0(%in: f32, %in_7: f32, %out: f32):
              %32 = arith.subf %in, %in_7 : f32
              %33 = math.powf %cst, %32 : f32
              linalg.yield %33 : f32
            }
            linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%17, %21, %20 : memref<32x32xf32>, memref<32x32xf32>, memref<32x32xf32>) outs(%17 : memref<32x32xf32>) {
            ^bb0(%in: f32, %in_7: f32, %in_8: f32, %out: f32):
              %32 = arith.mulf %in, %in_7 : f32
              %33 = arith.addf %32, %in_8 : f32
              linalg.yield %33 : f32
            }
            %30 = arith.muli %arg7, %c4096 : index
            %31 = arith.addi %25, %30 : index
            %reinterpret_cast_6 = memref.reinterpret_cast %arg1 to offset: [%31], sizes: [32, 32, 128], strides: [524288, 128, 1] : memref<32x4096x128xf32> to memref<32x32x128xf32, strided<[524288, 128, 1], offset: ?>>
            loom.copy %reinterpret_cast_6, %14 src_mem_space @DRAM dst_mem_space @L1, interconnect : [], broadcast : [1, 1] : memref<32x32x128xf32, strided<[524288, 128, 1], offset: ?>>, memref<32x32x128xf32>
            linalg.fill ins(%cst_1 : f32) outs(%24 : memref<32x32x128xf32>)
            linalg.batch_matmul ins(%15, %14 : memref<32x32x32xf32>, memref<32x32x128xf32>) outs(%24 : memref<32x32x128xf32>)
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%24, %23, %21 : memref<32x32x128xf32>, memref<32x32x128xf32>, memref<32x32xf32>) outs(%23 : memref<32x32x128xf32>) {
            ^bb0(%in: f32, %in_7: f32, %in_8: f32, %out: f32):
              %32 = arith.mulf %in_7, %in_8 : f32
              %33 = arith.addf %in, %32 : f32
              linalg.yield %33 : f32
            }
            linalg.copy ins(%19 : memref<32x32xf32>) outs(%18 : memref<32x32xf32>)
          }
          linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%23, %17 : memref<32x32x128xf32>, memref<32x32xf32>) outs(%22 : memref<32x32x128xf32>) {
          ^bb0(%in: f32, %in_5: f32, %out: f32):
            %28 = arith.divf %in, %in_5 : f32
            linalg.yield %28 : f32
          }
          %reinterpret_cast_4 = memref.reinterpret_cast %arg3 to offset: [%27], sizes: [32, 32, 128], strides: [524288, 128, 1] : memref<32x4096x128xf32> to memref<32x32x128xf32, strided<[524288, 128, 1], offset: ?>>
          loom.copy %22, %reinterpret_cast_4 src_mem_space @L1 dst_mem_space @DRAM, interconnect : [], broadcast : [1, 1] : memref<32x32x128xf32>, memref<32x32x128xf32, strided<[524288, 128, 1], offset: ?>>
        }
        scf.reduce 
      } {loom.iter_types = [#loom.iter_type<spatial>, #loom.iter_type<spatial>], loom.mapped_to_dims = [@y, @x]}
      return
    }
  }
  module attributes {loom.pass_name = "Materialize"} {
    func.func @attention__d1i0_d0i1__f01__d_h_h__BB32__BM32__BN32(%arg0: memref<32x128x4096xf32>, %arg1: memref<32x4096x128xf32>, %arg2: memref<32x4096x128xf32>, %arg3: memref<32x4096x128xf32>) {
      %c4096 = arith.constant 4096 : index
      %c16777216 = arith.constant 16777216 : index
      %c128 = arith.constant 128 : index
      %c16 = arith.constant 16 : index
      %cst = arith.constant 2.000000e+00 : f32
      %cst_0 = arith.constant 0.12751743074602467 : f64
      %cst_1 = arith.constant 0.000000e+00 : f32
      %cst_2 = arith.constant 1.000000e+00 : f32
      %cst_3 = arith.constant 0xFF800000 : f32
      %c32 = arith.constant 32 : index
      %c0 = arith.constant 0 : index
      %c8 = arith.constant 8 : index
      %c1 = arith.constant 1 : index
      scf.parallel (%arg4, %arg5) = (%c0, %c0) to (%c8, %c8) step (%c1, %c1) {
        scf.for %arg6 = %c0 to %c16 step %c1 {
          %12 = arith.muli %arg6, %c8 overflow<nsw> : index
          %13 = arith.addi %arg5, %12 : index
          %14 = loom.alloc [32, 32, 128] on @L1 : memref<32x32x128xf32>
          %15 = loom.alloc [32, 32, 32] on @L1 : memref<32x32x32xf32>
          %16 = loom.alloc [32, 128, 32] on @L1 : memref<32x128x32xf32>
          %17 = loom.alloc [32, 32] on @L1 : memref<32x32xf32>
          %18 = loom.alloc [32, 32] on @L1 : memref<32x32xf32>
          %19 = loom.alloc [32, 32] on @L1 : memref<32x32xf32>
          %20 = loom.alloc [32, 32] on @L1 : memref<32x32xf32>
          %21 = loom.alloc [32, 32] on @L1 : memref<32x32xf32>
          %22 = loom.alloc [32, 32, 128] on @L1 : memref<32x32x128xf32>
          %23 = loom.alloc [32, 32, 128] on @L1 : memref<32x32x128xf32>
          %24 = loom.alloc [32, 32, 128] on @L1 : memref<32x32x128xf32>
          %25 = arith.muli %arg4, %c16777216 : index
          %26 = arith.muli %13, %c4096 : index
          %27 = arith.addi %25, %26 : index
          %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%27], sizes: [32, 32, 128], strides: [524288, 128, 1] : memref<32x4096x128xf32> to memref<32x32x128xf32, strided<[524288, 128, 1], offset: ?>>
          loom.copy %reinterpret_cast, %22 src_mem_space @DRAM dst_mem_space @L1, interconnect : [], broadcast : [1, 1] : memref<32x32x128xf32, strided<[524288, 128, 1], offset: ?>>, memref<32x32x128xf32>
          linalg.fill ins(%cst_1 : f32) outs(%23 : memref<32x32x128xf32>)
          linalg.fill ins(%cst_2 : f32) outs(%17 : memref<32x32xf32>)
          linalg.fill ins(%cst_3 : f32) outs(%18 : memref<32x32xf32>)
          scf.for %arg7 = %c0 to %c128 step %c1 {
            %28 = arith.muli %arg7, %c32 : index
            %29 = arith.addi %25, %28 : index
            %reinterpret_cast_5 = memref.reinterpret_cast %arg0 to offset: [%29], sizes: [32, 128, 32], strides: [524288, 4096, 1] : memref<32x128x4096xf32> to memref<32x128x32xf32, strided<[524288, 4096, 1], offset: ?>>
            loom.copy %reinterpret_cast_5, %16 src_mem_space @DRAM dst_mem_space @L1, interconnect : [@horizontal_links], broadcast : [1, 8] : memref<32x128x32xf32, strided<[524288, 4096, 1], offset: ?>>, memref<32x128x32xf32>
            linalg.fill ins(%cst_1 : f32) outs(%15 : memref<32x32x32xf32>)
            linalg.batch_matmul ins(%22, %16 : memref<32x32x128xf32>, memref<32x128x32xf32>) outs(%15 : memref<32x32x32xf32>)
            linalg.fill ins(%cst_3 : f32) outs(%19 : memref<32x32xf32>)
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%15 : memref<32x32x32xf32>) outs(%19 : memref<32x32xf32>) {
            ^bb0(%in: f32, %out: f32):
              %32 = arith.maximumf %in, %out : f32
              linalg.yield %32 : f32
            }
            linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%18, %19 : memref<32x32xf32>, memref<32x32xf32>) outs(%19 : memref<32x32xf32>) {
            ^bb0(%in: f32, %in_7: f32, %out: f32):
              %32 = arith.truncf %cst_0 : f64 to f32
              %33 = arith.mulf %in_7, %32 : f32
              %34 = arith.cmpf ogt, %in, %33 : f32
              %35 = arith.select %34, %in, %33 : f32
              linalg.yield %35 : f32
            }
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%15, %19 : memref<32x32x32xf32>, memref<32x32xf32>) outs(%15 : memref<32x32x32xf32>) {
            ^bb0(%in: f32, %in_7: f32, %out: f32):
              %32 = arith.truncf %cst_0 : f64 to f32
              %33 = arith.mulf %in, %32 : f32
              %34 = arith.subf %33, %in_7 : f32
              %35 = math.powf %cst, %34 : f32
              linalg.yield %35 : f32
            }
            linalg.fill ins(%cst_1 : f32) outs(%20 : memref<32x32xf32>)
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%15 : memref<32x32x32xf32>) outs(%20 : memref<32x32xf32>) {
            ^bb0(%in: f32, %out: f32):
              %32 = arith.addf %in, %out : f32
              linalg.yield %32 : f32
            }
            linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%18, %19 : memref<32x32xf32>, memref<32x32xf32>) outs(%21 : memref<32x32xf32>) {
            ^bb0(%in: f32, %in_7: f32, %out: f32):
              %32 = arith.subf %in, %in_7 : f32
              %33 = math.powf %cst, %32 : f32
              linalg.yield %33 : f32
            }
            linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%17, %21, %20 : memref<32x32xf32>, memref<32x32xf32>, memref<32x32xf32>) outs(%17 : memref<32x32xf32>) {
            ^bb0(%in: f32, %in_7: f32, %in_8: f32, %out: f32):
              %32 = arith.mulf %in, %in_7 : f32
              %33 = arith.addf %32, %in_8 : f32
              linalg.yield %33 : f32
            }
            %30 = arith.muli %arg7, %c4096 : index
            %31 = arith.addi %25, %30 : index
            %reinterpret_cast_6 = memref.reinterpret_cast %arg1 to offset: [%31], sizes: [32, 32, 128], strides: [524288, 128, 1] : memref<32x4096x128xf32> to memref<32x32x128xf32, strided<[524288, 128, 1], offset: ?>>
            loom.copy %reinterpret_cast_6, %14 src_mem_space @DRAM dst_mem_space @L1, interconnect : [@horizontal_links], broadcast : [1, 8] : memref<32x32x128xf32, strided<[524288, 128, 1], offset: ?>>, memref<32x32x128xf32>
            linalg.fill ins(%cst_1 : f32) outs(%24 : memref<32x32x128xf32>)
            linalg.batch_matmul ins(%15, %14 : memref<32x32x32xf32>, memref<32x32x128xf32>) outs(%24 : memref<32x32x128xf32>)
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%24, %23, %21 : memref<32x32x128xf32>, memref<32x32x128xf32>, memref<32x32xf32>) outs(%23 : memref<32x32x128xf32>) {
            ^bb0(%in: f32, %in_7: f32, %in_8: f32, %out: f32):
              %32 = arith.mulf %in_7, %in_8 : f32
              %33 = arith.addf %in, %32 : f32
              linalg.yield %33 : f32
            }
            linalg.copy ins(%19 : memref<32x32xf32>) outs(%18 : memref<32x32xf32>)
          }
          linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%23, %17 : memref<32x32x128xf32>, memref<32x32xf32>) outs(%22 : memref<32x32x128xf32>) {
          ^bb0(%in: f32, %in_5: f32, %out: f32):
            %28 = arith.divf %in, %in_5 : f32
            linalg.yield %28 : f32
          }
          %reinterpret_cast_4 = memref.reinterpret_cast %arg3 to offset: [%27], sizes: [32, 32, 128], strides: [524288, 128, 1] : memref<32x4096x128xf32> to memref<32x32x128xf32, strided<[524288, 128, 1], offset: ?>>
          loom.copy %22, %reinterpret_cast_4 src_mem_space @L1 dst_mem_space @DRAM, interconnect : [], broadcast : [1, 1] : memref<32x32x128xf32>, memref<32x32x128xf32, strided<[524288, 128, 1], offset: ?>>
        }
        scf.reduce 
      } {loom.iter_types = [#loom.iter_type<spatial>, #loom.iter_type<spatial>], loom.mapped_to_dims = [@y, @x]}
      return
    }
  }
  module attributes {loom.pass_name = "Materialize"} {
    func.func @attention__d1i0_d0i1__f10__d_d_d__BB32__BM32__BN32(%arg0: memref<32x128x4096xf32>, %arg1: memref<32x4096x128xf32>, %arg2: memref<32x4096x128xf32>, %arg3: memref<32x4096x128xf32>) {
      %c4096 = arith.constant 4096 : index
      %c16777216 = arith.constant 16777216 : index
      %c128 = arith.constant 128 : index
      %c16 = arith.constant 16 : index
      %cst = arith.constant 2.000000e+00 : f32
      %cst_0 = arith.constant 0.12751743074602467 : f64
      %cst_1 = arith.constant 0.000000e+00 : f32
      %cst_2 = arith.constant 1.000000e+00 : f32
      %cst_3 = arith.constant 0xFF800000 : f32
      %c32 = arith.constant 32 : index
      %c0 = arith.constant 0 : index
      %c8 = arith.constant 8 : index
      %c1 = arith.constant 1 : index
      scf.parallel (%arg4, %arg5) = (%c0, %c0) to (%c8, %c8) step (%c1, %c1) {
        scf.for %arg6 = %c0 to %c16 step %c1 {
          %12 = arith.muli %arg6, %c8 overflow<nsw> : index
          %13 = arith.addi %arg5, %12 : index
          %14 = loom.alloc [32, 32, 128] on @L1 : memref<32x32x128xf32>
          %15 = loom.alloc [32, 32, 32] on @L1 : memref<32x32x32xf32>
          %16 = loom.alloc [32, 128, 32] on @L1 : memref<32x128x32xf32>
          %17 = loom.alloc [32, 32] on @L1 : memref<32x32xf32>
          %18 = loom.alloc [32, 32] on @L1 : memref<32x32xf32>
          %19 = loom.alloc [32, 32] on @L1 : memref<32x32xf32>
          %20 = loom.alloc [32, 32] on @L1 : memref<32x32xf32>
          %21 = loom.alloc [32, 32] on @L1 : memref<32x32xf32>
          %22 = loom.alloc [32, 32, 128] on @L1 : memref<32x32x128xf32>
          %23 = loom.alloc [32, 32, 128] on @L1 : memref<32x32x128xf32>
          %24 = loom.alloc [32, 32, 128] on @L1 : memref<32x32x128xf32>
          %25 = arith.muli %arg4, %c16777216 : index
          %26 = arith.muli %13, %c4096 : index
          %27 = arith.addi %25, %26 : index
          %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%27], sizes: [32, 32, 128], strides: [524288, 128, 1] : memref<32x4096x128xf32> to memref<32x32x128xf32, strided<[524288, 128, 1], offset: ?>>
          loom.copy %reinterpret_cast, %22 src_mem_space @DRAM dst_mem_space @L1, interconnect : [], broadcast : [1, 1] : memref<32x32x128xf32, strided<[524288, 128, 1], offset: ?>>, memref<32x32x128xf32>
          linalg.fill ins(%cst_1 : f32) outs(%23 : memref<32x32x128xf32>)
          linalg.fill ins(%cst_2 : f32) outs(%17 : memref<32x32xf32>)
          linalg.fill ins(%cst_3 : f32) outs(%18 : memref<32x32xf32>)
          scf.for %arg7 = %c0 to %c128 step %c1 {
            %28 = arith.muli %arg7, %c32 : index
            %29 = arith.addi %25, %28 : index
            %reinterpret_cast_5 = memref.reinterpret_cast %arg0 to offset: [%29], sizes: [32, 128, 32], strides: [524288, 4096, 1] : memref<32x128x4096xf32> to memref<32x128x32xf32, strided<[524288, 4096, 1], offset: ?>>
            loom.copy %reinterpret_cast_5, %16 src_mem_space @DRAM dst_mem_space @L1, interconnect : [], broadcast : [1, 1] : memref<32x128x32xf32, strided<[524288, 4096, 1], offset: ?>>, memref<32x128x32xf32>
            linalg.fill ins(%cst_1 : f32) outs(%15 : memref<32x32x32xf32>)
            linalg.batch_matmul ins(%22, %16 : memref<32x32x128xf32>, memref<32x128x32xf32>) outs(%15 : memref<32x32x32xf32>)
            linalg.fill ins(%cst_3 : f32) outs(%19 : memref<32x32xf32>)
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%15 : memref<32x32x32xf32>) outs(%19 : memref<32x32xf32>) {
            ^bb0(%in: f32, %out: f32):
              %32 = arith.maximumf %in, %out : f32
              linalg.yield %32 : f32
            }
            linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%18, %19 : memref<32x32xf32>, memref<32x32xf32>) outs(%19 : memref<32x32xf32>) {
            ^bb0(%in: f32, %in_7: f32, %out: f32):
              %32 = arith.truncf %cst_0 : f64 to f32
              %33 = arith.mulf %in_7, %32 : f32
              %34 = arith.cmpf ogt, %in, %33 : f32
              %35 = arith.select %34, %in, %33 : f32
              linalg.yield %35 : f32
            }
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%15, %19 : memref<32x32x32xf32>, memref<32x32xf32>) outs(%15 : memref<32x32x32xf32>) {
            ^bb0(%in: f32, %in_7: f32, %out: f32):
              %32 = arith.truncf %cst_0 : f64 to f32
              %33 = arith.mulf %in, %32 : f32
              %34 = arith.subf %33, %in_7 : f32
              %35 = math.powf %cst, %34 : f32
              linalg.yield %35 : f32
            }
            linalg.fill ins(%cst_1 : f32) outs(%20 : memref<32x32xf32>)
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%15 : memref<32x32x32xf32>) outs(%20 : memref<32x32xf32>) {
            ^bb0(%in: f32, %out: f32):
              %32 = arith.addf %in, %out : f32
              linalg.yield %32 : f32
            }
            linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%18, %19 : memref<32x32xf32>, memref<32x32xf32>) outs(%21 : memref<32x32xf32>) {
            ^bb0(%in: f32, %in_7: f32, %out: f32):
              %32 = arith.subf %in, %in_7 : f32
              %33 = math.powf %cst, %32 : f32
              linalg.yield %33 : f32
            }
            linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%17, %21, %20 : memref<32x32xf32>, memref<32x32xf32>, memref<32x32xf32>) outs(%17 : memref<32x32xf32>) {
            ^bb0(%in: f32, %in_7: f32, %in_8: f32, %out: f32):
              %32 = arith.mulf %in, %in_7 : f32
              %33 = arith.addf %32, %in_8 : f32
              linalg.yield %33 : f32
            }
            %30 = arith.muli %arg7, %c4096 : index
            %31 = arith.addi %25, %30 : index
            %reinterpret_cast_6 = memref.reinterpret_cast %arg1 to offset: [%31], sizes: [32, 32, 128], strides: [524288, 128, 1] : memref<32x4096x128xf32> to memref<32x32x128xf32, strided<[524288, 128, 1], offset: ?>>
            loom.copy %reinterpret_cast_6, %14 src_mem_space @DRAM dst_mem_space @L1, interconnect : [], broadcast : [1, 1] : memref<32x32x128xf32, strided<[524288, 128, 1], offset: ?>>, memref<32x32x128xf32>
            linalg.fill ins(%cst_1 : f32) outs(%24 : memref<32x32x128xf32>)
            linalg.batch_matmul ins(%15, %14 : memref<32x32x32xf32>, memref<32x32x128xf32>) outs(%24 : memref<32x32x128xf32>)
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%24, %23, %21 : memref<32x32x128xf32>, memref<32x32x128xf32>, memref<32x32xf32>) outs(%23 : memref<32x32x128xf32>) {
            ^bb0(%in: f32, %in_7: f32, %in_8: f32, %out: f32):
              %32 = arith.mulf %in_7, %in_8 : f32
              %33 = arith.addf %in, %32 : f32
              linalg.yield %33 : f32
            }
            linalg.copy ins(%19 : memref<32x32xf32>) outs(%18 : memref<32x32xf32>)
          }
          linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%23, %17 : memref<32x32x128xf32>, memref<32x32xf32>) outs(%22 : memref<32x32x128xf32>) {
          ^bb0(%in: f32, %in_5: f32, %out: f32):
            %28 = arith.divf %in, %in_5 : f32
            linalg.yield %28 : f32
          }
          %reinterpret_cast_4 = memref.reinterpret_cast %arg3 to offset: [%27], sizes: [32, 32, 128], strides: [524288, 128, 1] : memref<32x4096x128xf32> to memref<32x32x128xf32, strided<[524288, 128, 1], offset: ?>>
          loom.copy %22, %reinterpret_cast_4 src_mem_space @L1 dst_mem_space @DRAM, interconnect : [], broadcast : [1, 1] : memref<32x32x128xf32>, memref<32x32x128xf32, strided<[524288, 128, 1], offset: ?>>
        }
        scf.reduce 
      } {loom.iter_types = [#loom.iter_type<spatial>, #loom.iter_type<spatial>], loom.mapped_to_dims = [@y, @x]}
      return
    }
  }
  module attributes {loom.pass_name = "Materialize"} {
    func.func @attention__d1i0_d0i1__f10__d_d_h__BB32__BM32__BN32(%arg0: memref<32x128x4096xf32>, %arg1: memref<32x4096x128xf32>, %arg2: memref<32x4096x128xf32>, %arg3: memref<32x4096x128xf32>) {
      %c4096 = arith.constant 4096 : index
      %c16777216 = arith.constant 16777216 : index
      %c128 = arith.constant 128 : index
      %c16 = arith.constant 16 : index
      %cst = arith.constant 2.000000e+00 : f32
      %cst_0 = arith.constant 0.12751743074602467 : f64
      %cst_1 = arith.constant 0.000000e+00 : f32
      %cst_2 = arith.constant 1.000000e+00 : f32
      %cst_3 = arith.constant 0xFF800000 : f32
      %c32 = arith.constant 32 : index
      %c0 = arith.constant 0 : index
      %c8 = arith.constant 8 : index
      %c1 = arith.constant 1 : index
      scf.parallel (%arg4, %arg5) = (%c0, %c0) to (%c8, %c8) step (%c1, %c1) {
        scf.for %arg6 = %c0 to %c16 step %c1 {
          %12 = arith.muli %arg6, %c8 overflow<nsw> : index
          %13 = arith.addi %arg5, %12 : index
          %14 = loom.alloc [32, 32, 128] on @L1 : memref<32x32x128xf32>
          %15 = loom.alloc [32, 32, 32] on @L1 : memref<32x32x32xf32>
          %16 = loom.alloc [32, 128, 32] on @L1 : memref<32x128x32xf32>
          %17 = loom.alloc [32, 32] on @L1 : memref<32x32xf32>
          %18 = loom.alloc [32, 32] on @L1 : memref<32x32xf32>
          %19 = loom.alloc [32, 32] on @L1 : memref<32x32xf32>
          %20 = loom.alloc [32, 32] on @L1 : memref<32x32xf32>
          %21 = loom.alloc [32, 32] on @L1 : memref<32x32xf32>
          %22 = loom.alloc [32, 32, 128] on @L1 : memref<32x32x128xf32>
          %23 = loom.alloc [32, 32, 128] on @L1 : memref<32x32x128xf32>
          %24 = loom.alloc [32, 32, 128] on @L1 : memref<32x32x128xf32>
          %25 = arith.muli %arg4, %c16777216 : index
          %26 = arith.muli %13, %c4096 : index
          %27 = arith.addi %25, %26 : index
          %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%27], sizes: [32, 32, 128], strides: [524288, 128, 1] : memref<32x4096x128xf32> to memref<32x32x128xf32, strided<[524288, 128, 1], offset: ?>>
          loom.copy %reinterpret_cast, %22 src_mem_space @DRAM dst_mem_space @L1, interconnect : [], broadcast : [1, 1] : memref<32x32x128xf32, strided<[524288, 128, 1], offset: ?>>, memref<32x32x128xf32>
          linalg.fill ins(%cst_1 : f32) outs(%23 : memref<32x32x128xf32>)
          linalg.fill ins(%cst_2 : f32) outs(%17 : memref<32x32xf32>)
          linalg.fill ins(%cst_3 : f32) outs(%18 : memref<32x32xf32>)
          scf.for %arg7 = %c0 to %c128 step %c1 {
            %28 = arith.muli %arg7, %c32 : index
            %29 = arith.addi %25, %28 : index
            %reinterpret_cast_5 = memref.reinterpret_cast %arg0 to offset: [%29], sizes: [32, 128, 32], strides: [524288, 4096, 1] : memref<32x128x4096xf32> to memref<32x128x32xf32, strided<[524288, 4096, 1], offset: ?>>
            loom.copy %reinterpret_cast_5, %16 src_mem_space @DRAM dst_mem_space @L1, interconnect : [], broadcast : [1, 1] : memref<32x128x32xf32, strided<[524288, 4096, 1], offset: ?>>, memref<32x128x32xf32>
            linalg.fill ins(%cst_1 : f32) outs(%15 : memref<32x32x32xf32>)
            linalg.batch_matmul ins(%22, %16 : memref<32x32x128xf32>, memref<32x128x32xf32>) outs(%15 : memref<32x32x32xf32>)
            linalg.fill ins(%cst_3 : f32) outs(%19 : memref<32x32xf32>)
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%15 : memref<32x32x32xf32>) outs(%19 : memref<32x32xf32>) {
            ^bb0(%in: f32, %out: f32):
              %32 = arith.maximumf %in, %out : f32
              linalg.yield %32 : f32
            }
            linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%18, %19 : memref<32x32xf32>, memref<32x32xf32>) outs(%19 : memref<32x32xf32>) {
            ^bb0(%in: f32, %in_7: f32, %out: f32):
              %32 = arith.truncf %cst_0 : f64 to f32
              %33 = arith.mulf %in_7, %32 : f32
              %34 = arith.cmpf ogt, %in, %33 : f32
              %35 = arith.select %34, %in, %33 : f32
              linalg.yield %35 : f32
            }
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%15, %19 : memref<32x32x32xf32>, memref<32x32xf32>) outs(%15 : memref<32x32x32xf32>) {
            ^bb0(%in: f32, %in_7: f32, %out: f32):
              %32 = arith.truncf %cst_0 : f64 to f32
              %33 = arith.mulf %in, %32 : f32
              %34 = arith.subf %33, %in_7 : f32
              %35 = math.powf %cst, %34 : f32
              linalg.yield %35 : f32
            }
            linalg.fill ins(%cst_1 : f32) outs(%20 : memref<32x32xf32>)
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%15 : memref<32x32x32xf32>) outs(%20 : memref<32x32xf32>) {
            ^bb0(%in: f32, %out: f32):
              %32 = arith.addf %in, %out : f32
              linalg.yield %32 : f32
            }
            linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%18, %19 : memref<32x32xf32>, memref<32x32xf32>) outs(%21 : memref<32x32xf32>) {
            ^bb0(%in: f32, %in_7: f32, %out: f32):
              %32 = arith.subf %in, %in_7 : f32
              %33 = math.powf %cst, %32 : f32
              linalg.yield %33 : f32
            }
            linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%17, %21, %20 : memref<32x32xf32>, memref<32x32xf32>, memref<32x32xf32>) outs(%17 : memref<32x32xf32>) {
            ^bb0(%in: f32, %in_7: f32, %in_8: f32, %out: f32):
              %32 = arith.mulf %in, %in_7 : f32
              %33 = arith.addf %32, %in_8 : f32
              linalg.yield %33 : f32
            }
            %30 = arith.muli %arg7, %c4096 : index
            %31 = arith.addi %25, %30 : index
            %reinterpret_cast_6 = memref.reinterpret_cast %arg1 to offset: [%31], sizes: [32, 32, 128], strides: [524288, 128, 1] : memref<32x4096x128xf32> to memref<32x32x128xf32, strided<[524288, 128, 1], offset: ?>>
            loom.copy %reinterpret_cast_6, %14 src_mem_space @DRAM dst_mem_space @L1, interconnect : [@horizontal_links], broadcast : [1, 8] : memref<32x32x128xf32, strided<[524288, 128, 1], offset: ?>>, memref<32x32x128xf32>
            linalg.fill ins(%cst_1 : f32) outs(%24 : memref<32x32x128xf32>)
            linalg.batch_matmul ins(%15, %14 : memref<32x32x32xf32>, memref<32x32x128xf32>) outs(%24 : memref<32x32x128xf32>)
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%24, %23, %21 : memref<32x32x128xf32>, memref<32x32x128xf32>, memref<32x32xf32>) outs(%23 : memref<32x32x128xf32>) {
            ^bb0(%in: f32, %in_7: f32, %in_8: f32, %out: f32):
              %32 = arith.mulf %in_7, %in_8 : f32
              %33 = arith.addf %in, %32 : f32
              linalg.yield %33 : f32
            }
            linalg.copy ins(%19 : memref<32x32xf32>) outs(%18 : memref<32x32xf32>)
          }
          linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%23, %17 : memref<32x32x128xf32>, memref<32x32xf32>) outs(%22 : memref<32x32x128xf32>) {
          ^bb0(%in: f32, %in_5: f32, %out: f32):
            %28 = arith.divf %in, %in_5 : f32
            linalg.yield %28 : f32
          }
          %reinterpret_cast_4 = memref.reinterpret_cast %arg3 to offset: [%27], sizes: [32, 32, 128], strides: [524288, 128, 1] : memref<32x4096x128xf32> to memref<32x32x128xf32, strided<[524288, 128, 1], offset: ?>>
          loom.copy %22, %reinterpret_cast_4 src_mem_space @L1 dst_mem_space @DRAM, interconnect : [], broadcast : [1, 1] : memref<32x32x128xf32>, memref<32x32x128xf32, strided<[524288, 128, 1], offset: ?>>
        }
        scf.reduce 
      } {loom.iter_types = [#loom.iter_type<spatial>, #loom.iter_type<spatial>], loom.mapped_to_dims = [@y, @x]}
      return
    }
  }
  module attributes {loom.pass_name = "Materialize"} {
    func.func @attention__d1i0_d0i1__f10__d_h_d__BB32__BM32__BN32(%arg0: memref<32x128x4096xf32>, %arg1: memref<32x4096x128xf32>, %arg2: memref<32x4096x128xf32>, %arg3: memref<32x4096x128xf32>) {
      %c4096 = arith.constant 4096 : index
      %c16777216 = arith.constant 16777216 : index
      %c128 = arith.constant 128 : index
      %c16 = arith.constant 16 : index
      %cst = arith.constant 2.000000e+00 : f32
      %cst_0 = arith.constant 0.12751743074602467 : f64
      %cst_1 = arith.constant 0.000000e+00 : f32
      %cst_2 = arith.constant 1.000000e+00 : f32
      %cst_3 = arith.constant 0xFF800000 : f32
      %c32 = arith.constant 32 : index
      %c0 = arith.constant 0 : index
      %c8 = arith.constant 8 : index
      %c1 = arith.constant 1 : index
      scf.parallel (%arg4, %arg5) = (%c0, %c0) to (%c8, %c8) step (%c1, %c1) {
        scf.for %arg6 = %c0 to %c16 step %c1 {
          %12 = arith.muli %arg6, %c8 overflow<nsw> : index
          %13 = arith.addi %arg5, %12 : index
          %14 = loom.alloc [32, 32, 128] on @L1 : memref<32x32x128xf32>
          %15 = loom.alloc [32, 32, 32] on @L1 : memref<32x32x32xf32>
          %16 = loom.alloc [32, 128, 32] on @L1 : memref<32x128x32xf32>
          %17 = loom.alloc [32, 32] on @L1 : memref<32x32xf32>
          %18 = loom.alloc [32, 32] on @L1 : memref<32x32xf32>
          %19 = loom.alloc [32, 32] on @L1 : memref<32x32xf32>
          %20 = loom.alloc [32, 32] on @L1 : memref<32x32xf32>
          %21 = loom.alloc [32, 32] on @L1 : memref<32x32xf32>
          %22 = loom.alloc [32, 32, 128] on @L1 : memref<32x32x128xf32>
          %23 = loom.alloc [32, 32, 128] on @L1 : memref<32x32x128xf32>
          %24 = loom.alloc [32, 32, 128] on @L1 : memref<32x32x128xf32>
          %25 = arith.muli %arg4, %c16777216 : index
          %26 = arith.muli %13, %c4096 : index
          %27 = arith.addi %25, %26 : index
          %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%27], sizes: [32, 32, 128], strides: [524288, 128, 1] : memref<32x4096x128xf32> to memref<32x32x128xf32, strided<[524288, 128, 1], offset: ?>>
          loom.copy %reinterpret_cast, %22 src_mem_space @DRAM dst_mem_space @L1, interconnect : [], broadcast : [1, 1] : memref<32x32x128xf32, strided<[524288, 128, 1], offset: ?>>, memref<32x32x128xf32>
          linalg.fill ins(%cst_1 : f32) outs(%23 : memref<32x32x128xf32>)
          linalg.fill ins(%cst_2 : f32) outs(%17 : memref<32x32xf32>)
          linalg.fill ins(%cst_3 : f32) outs(%18 : memref<32x32xf32>)
          scf.for %arg7 = %c0 to %c128 step %c1 {
            %28 = arith.muli %arg7, %c32 : index
            %29 = arith.addi %25, %28 : index
            %reinterpret_cast_5 = memref.reinterpret_cast %arg0 to offset: [%29], sizes: [32, 128, 32], strides: [524288, 4096, 1] : memref<32x128x4096xf32> to memref<32x128x32xf32, strided<[524288, 4096, 1], offset: ?>>
            loom.copy %reinterpret_cast_5, %16 src_mem_space @DRAM dst_mem_space @L1, interconnect : [@horizontal_links], broadcast : [1, 8] : memref<32x128x32xf32, strided<[524288, 4096, 1], offset: ?>>, memref<32x128x32xf32>
            linalg.fill ins(%cst_1 : f32) outs(%15 : memref<32x32x32xf32>)
            linalg.batch_matmul ins(%22, %16 : memref<32x32x128xf32>, memref<32x128x32xf32>) outs(%15 : memref<32x32x32xf32>)
            linalg.fill ins(%cst_3 : f32) outs(%19 : memref<32x32xf32>)
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%15 : memref<32x32x32xf32>) outs(%19 : memref<32x32xf32>) {
            ^bb0(%in: f32, %out: f32):
              %32 = arith.maximumf %in, %out : f32
              linalg.yield %32 : f32
            }
            linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%18, %19 : memref<32x32xf32>, memref<32x32xf32>) outs(%19 : memref<32x32xf32>) {
            ^bb0(%in: f32, %in_7: f32, %out: f32):
              %32 = arith.truncf %cst_0 : f64 to f32
              %33 = arith.mulf %in_7, %32 : f32
              %34 = arith.cmpf ogt, %in, %33 : f32
              %35 = arith.select %34, %in, %33 : f32
              linalg.yield %35 : f32
            }
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%15, %19 : memref<32x32x32xf32>, memref<32x32xf32>) outs(%15 : memref<32x32x32xf32>) {
            ^bb0(%in: f32, %in_7: f32, %out: f32):
              %32 = arith.truncf %cst_0 : f64 to f32
              %33 = arith.mulf %in, %32 : f32
              %34 = arith.subf %33, %in_7 : f32
              %35 = math.powf %cst, %34 : f32
              linalg.yield %35 : f32
            }
            linalg.fill ins(%cst_1 : f32) outs(%20 : memref<32x32xf32>)
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%15 : memref<32x32x32xf32>) outs(%20 : memref<32x32xf32>) {
            ^bb0(%in: f32, %out: f32):
              %32 = arith.addf %in, %out : f32
              linalg.yield %32 : f32
            }
            linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%18, %19 : memref<32x32xf32>, memref<32x32xf32>) outs(%21 : memref<32x32xf32>) {
            ^bb0(%in: f32, %in_7: f32, %out: f32):
              %32 = arith.subf %in, %in_7 : f32
              %33 = math.powf %cst, %32 : f32
              linalg.yield %33 : f32
            }
            linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%17, %21, %20 : memref<32x32xf32>, memref<32x32xf32>, memref<32x32xf32>) outs(%17 : memref<32x32xf32>) {
            ^bb0(%in: f32, %in_7: f32, %in_8: f32, %out: f32):
              %32 = arith.mulf %in, %in_7 : f32
              %33 = arith.addf %32, %in_8 : f32
              linalg.yield %33 : f32
            }
            %30 = arith.muli %arg7, %c4096 : index
            %31 = arith.addi %25, %30 : index
            %reinterpret_cast_6 = memref.reinterpret_cast %arg1 to offset: [%31], sizes: [32, 32, 128], strides: [524288, 128, 1] : memref<32x4096x128xf32> to memref<32x32x128xf32, strided<[524288, 128, 1], offset: ?>>
            loom.copy %reinterpret_cast_6, %14 src_mem_space @DRAM dst_mem_space @L1, interconnect : [], broadcast : [1, 1] : memref<32x32x128xf32, strided<[524288, 128, 1], offset: ?>>, memref<32x32x128xf32>
            linalg.fill ins(%cst_1 : f32) outs(%24 : memref<32x32x128xf32>)
            linalg.batch_matmul ins(%15, %14 : memref<32x32x32xf32>, memref<32x32x128xf32>) outs(%24 : memref<32x32x128xf32>)
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%24, %23, %21 : memref<32x32x128xf32>, memref<32x32x128xf32>, memref<32x32xf32>) outs(%23 : memref<32x32x128xf32>) {
            ^bb0(%in: f32, %in_7: f32, %in_8: f32, %out: f32):
              %32 = arith.mulf %in_7, %in_8 : f32
              %33 = arith.addf %in, %32 : f32
              linalg.yield %33 : f32
            }
            linalg.copy ins(%19 : memref<32x32xf32>) outs(%18 : memref<32x32xf32>)
          }
          linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%23, %17 : memref<32x32x128xf32>, memref<32x32xf32>) outs(%22 : memref<32x32x128xf32>) {
          ^bb0(%in: f32, %in_5: f32, %out: f32):
            %28 = arith.divf %in, %in_5 : f32
            linalg.yield %28 : f32
          }
          %reinterpret_cast_4 = memref.reinterpret_cast %arg3 to offset: [%27], sizes: [32, 32, 128], strides: [524288, 128, 1] : memref<32x4096x128xf32> to memref<32x32x128xf32, strided<[524288, 128, 1], offset: ?>>
          loom.copy %22, %reinterpret_cast_4 src_mem_space @L1 dst_mem_space @DRAM, interconnect : [], broadcast : [1, 1] : memref<32x32x128xf32>, memref<32x32x128xf32, strided<[524288, 128, 1], offset: ?>>
        }
        scf.reduce 
      } {loom.iter_types = [#loom.iter_type<spatial>, #loom.iter_type<spatial>], loom.mapped_to_dims = [@y, @x]}
      return
    }
  }
  module attributes {loom.pass_name = "Materialize"} {
    func.func @attention__d1i0_d0i1__f10__d_h_h__BB32__BM32__BN32(%arg0: memref<32x128x4096xf32>, %arg1: memref<32x4096x128xf32>, %arg2: memref<32x4096x128xf32>, %arg3: memref<32x4096x128xf32>) {
      %c4096 = arith.constant 4096 : index
      %c16777216 = arith.constant 16777216 : index
      %c128 = arith.constant 128 : index
      %c16 = arith.constant 16 : index
      %cst = arith.constant 2.000000e+00 : f32
      %cst_0 = arith.constant 0.12751743074602467 : f64
      %cst_1 = arith.constant 0.000000e+00 : f32
      %cst_2 = arith.constant 1.000000e+00 : f32
      %cst_3 = arith.constant 0xFF800000 : f32
      %c32 = arith.constant 32 : index
      %c0 = arith.constant 0 : index
      %c8 = arith.constant 8 : index
      %c1 = arith.constant 1 : index
      scf.parallel (%arg4, %arg5) = (%c0, %c0) to (%c8, %c8) step (%c1, %c1) {
        scf.for %arg6 = %c0 to %c16 step %c1 {
          %12 = arith.muli %arg6, %c8 overflow<nsw> : index
          %13 = arith.addi %arg5, %12 : index
          %14 = loom.alloc [32, 32, 128] on @L1 : memref<32x32x128xf32>
          %15 = loom.alloc [32, 32, 32] on @L1 : memref<32x32x32xf32>
          %16 = loom.alloc [32, 128, 32] on @L1 : memref<32x128x32xf32>
          %17 = loom.alloc [32, 32] on @L1 : memref<32x32xf32>
          %18 = loom.alloc [32, 32] on @L1 : memref<32x32xf32>
          %19 = loom.alloc [32, 32] on @L1 : memref<32x32xf32>
          %20 = loom.alloc [32, 32] on @L1 : memref<32x32xf32>
          %21 = loom.alloc [32, 32] on @L1 : memref<32x32xf32>
          %22 = loom.alloc [32, 32, 128] on @L1 : memref<32x32x128xf32>
          %23 = loom.alloc [32, 32, 128] on @L1 : memref<32x32x128xf32>
          %24 = loom.alloc [32, 32, 128] on @L1 : memref<32x32x128xf32>
          %25 = arith.muli %arg4, %c16777216 : index
          %26 = arith.muli %13, %c4096 : index
          %27 = arith.addi %25, %26 : index
          %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%27], sizes: [32, 32, 128], strides: [524288, 128, 1] : memref<32x4096x128xf32> to memref<32x32x128xf32, strided<[524288, 128, 1], offset: ?>>
          loom.copy %reinterpret_cast, %22 src_mem_space @DRAM dst_mem_space @L1, interconnect : [], broadcast : [1, 1] : memref<32x32x128xf32, strided<[524288, 128, 1], offset: ?>>, memref<32x32x128xf32>
          linalg.fill ins(%cst_1 : f32) outs(%23 : memref<32x32x128xf32>)
          linalg.fill ins(%cst_2 : f32) outs(%17 : memref<32x32xf32>)
          linalg.fill ins(%cst_3 : f32) outs(%18 : memref<32x32xf32>)
          scf.for %arg7 = %c0 to %c128 step %c1 {
            %28 = arith.muli %arg7, %c32 : index
            %29 = arith.addi %25, %28 : index
            %reinterpret_cast_5 = memref.reinterpret_cast %arg0 to offset: [%29], sizes: [32, 128, 32], strides: [524288, 4096, 1] : memref<32x128x4096xf32> to memref<32x128x32xf32, strided<[524288, 4096, 1], offset: ?>>
            loom.copy %reinterpret_cast_5, %16 src_mem_space @DRAM dst_mem_space @L1, interconnect : [@horizontal_links], broadcast : [1, 8] : memref<32x128x32xf32, strided<[524288, 4096, 1], offset: ?>>, memref<32x128x32xf32>
            linalg.fill ins(%cst_1 : f32) outs(%15 : memref<32x32x32xf32>)
            linalg.batch_matmul ins(%22, %16 : memref<32x32x128xf32>, memref<32x128x32xf32>) outs(%15 : memref<32x32x32xf32>)
            linalg.fill ins(%cst_3 : f32) outs(%19 : memref<32x32xf32>)
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%15 : memref<32x32x32xf32>) outs(%19 : memref<32x32xf32>) {
            ^bb0(%in: f32, %out: f32):
              %32 = arith.maximumf %in, %out : f32
              linalg.yield %32 : f32
            }
            linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%18, %19 : memref<32x32xf32>, memref<32x32xf32>) outs(%19 : memref<32x32xf32>) {
            ^bb0(%in: f32, %in_7: f32, %out: f32):
              %32 = arith.truncf %cst_0 : f64 to f32
              %33 = arith.mulf %in_7, %32 : f32
              %34 = arith.cmpf ogt, %in, %33 : f32
              %35 = arith.select %34, %in, %33 : f32
              linalg.yield %35 : f32
            }
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%15, %19 : memref<32x32x32xf32>, memref<32x32xf32>) outs(%15 : memref<32x32x32xf32>) {
            ^bb0(%in: f32, %in_7: f32, %out: f32):
              %32 = arith.truncf %cst_0 : f64 to f32
              %33 = arith.mulf %in, %32 : f32
              %34 = arith.subf %33, %in_7 : f32
              %35 = math.powf %cst, %34 : f32
              linalg.yield %35 : f32
            }
            linalg.fill ins(%cst_1 : f32) outs(%20 : memref<32x32xf32>)
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%15 : memref<32x32x32xf32>) outs(%20 : memref<32x32xf32>) {
            ^bb0(%in: f32, %out: f32):
              %32 = arith.addf %in, %out : f32
              linalg.yield %32 : f32
            }
            linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%18, %19 : memref<32x32xf32>, memref<32x32xf32>) outs(%21 : memref<32x32xf32>) {
            ^bb0(%in: f32, %in_7: f32, %out: f32):
              %32 = arith.subf %in, %in_7 : f32
              %33 = math.powf %cst, %32 : f32
              linalg.yield %33 : f32
            }
            linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%17, %21, %20 : memref<32x32xf32>, memref<32x32xf32>, memref<32x32xf32>) outs(%17 : memref<32x32xf32>) {
            ^bb0(%in: f32, %in_7: f32, %in_8: f32, %out: f32):
              %32 = arith.mulf %in, %in_7 : f32
              %33 = arith.addf %32, %in_8 : f32
              linalg.yield %33 : f32
            }
            %30 = arith.muli %arg7, %c4096 : index
            %31 = arith.addi %25, %30 : index
            %reinterpret_cast_6 = memref.reinterpret_cast %arg1 to offset: [%31], sizes: [32, 32, 128], strides: [524288, 128, 1] : memref<32x4096x128xf32> to memref<32x32x128xf32, strided<[524288, 128, 1], offset: ?>>
            loom.copy %reinterpret_cast_6, %14 src_mem_space @DRAM dst_mem_space @L1, interconnect : [@horizontal_links], broadcast : [1, 8] : memref<32x32x128xf32, strided<[524288, 128, 1], offset: ?>>, memref<32x32x128xf32>
            linalg.fill ins(%cst_1 : f32) outs(%24 : memref<32x32x128xf32>)
            linalg.batch_matmul ins(%15, %14 : memref<32x32x32xf32>, memref<32x32x128xf32>) outs(%24 : memref<32x32x128xf32>)
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%24, %23, %21 : memref<32x32x128xf32>, memref<32x32x128xf32>, memref<32x32xf32>) outs(%23 : memref<32x32x128xf32>) {
            ^bb0(%in: f32, %in_7: f32, %in_8: f32, %out: f32):
              %32 = arith.mulf %in_7, %in_8 : f32
              %33 = arith.addf %in, %32 : f32
              linalg.yield %33 : f32
            }
            linalg.copy ins(%19 : memref<32x32xf32>) outs(%18 : memref<32x32xf32>)
          }
          linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%23, %17 : memref<32x32x128xf32>, memref<32x32xf32>) outs(%22 : memref<32x32x128xf32>) {
          ^bb0(%in: f32, %in_5: f32, %out: f32):
            %28 = arith.divf %in, %in_5 : f32
            linalg.yield %28 : f32
          }
          %reinterpret_cast_4 = memref.reinterpret_cast %arg3 to offset: [%27], sizes: [32, 32, 128], strides: [524288, 128, 1] : memref<32x4096x128xf32> to memref<32x32x128xf32, strided<[524288, 128, 1], offset: ?>>
          loom.copy %22, %reinterpret_cast_4 src_mem_space @L1 dst_mem_space @DRAM, interconnect : [], broadcast : [1, 1] : memref<32x32x128xf32>, memref<32x32x128xf32, strided<[524288, 128, 1], offset: ?>>
        }
        scf.reduce 
      } {loom.iter_types = [#loom.iter_type<spatial>, #loom.iter_type<spatial>], loom.mapped_to_dims = [@y, @x]}
      return
    }
  }
  module attributes {loom.pass_name = "Materialize"} {
    func.func @attention__d0i1_d1i1__f01__d_d_d__BB32__BM32__BN32(%arg0: memref<32x128x4096xf32>, %arg1: memref<32x4096x128xf32>, %arg2: memref<32x4096x128xf32>, %arg3: memref<32x4096x128xf32>) {
      %c4096 = arith.constant 4096 : index
      %c128 = arith.constant 128 : index
      %c64 = arith.constant 64 : index
      %c2 = arith.constant 2 : index
      %cst = arith.constant 2.000000e+00 : f32
      %cst_0 = arith.constant 0.12751743074602467 : f64
      %cst_1 = arith.constant 0.000000e+00 : f32
      %cst_2 = arith.constant 1.000000e+00 : f32
      %cst_3 = arith.constant 0xFF800000 : f32
      %c32 = arith.constant 32 : index
      %c0 = arith.constant 0 : index
      %c8 = arith.constant 8 : index
      %c1 = arith.constant 1 : index
      scf.parallel (%arg4, %arg5) = (%c0, %c0) to (%c8, %c8) step (%c1, %c1) {
        scf.for %arg6 = %c0 to %c2 step %c1 {
          %12 = arith.muli %arg4, %c8 overflow<nsw> : index
          %13 = arith.addi %12, %arg5 : index
          %14 = arith.muli %arg6, %c64 overflow<nsw> : index
          %15 = arith.addi %13, %14 : index
          %16 = loom.alloc [32, 32, 128] on @L1 : memref<32x32x128xf32>
          %17 = loom.alloc [32, 32, 32] on @L1 : memref<32x32x32xf32>
          %18 = loom.alloc [32, 128, 32] on @L1 : memref<32x128x32xf32>
          %19 = loom.alloc [32, 32] on @L1 : memref<32x32xf32>
          %20 = loom.alloc [32, 32] on @L1 : memref<32x32xf32>
          %21 = loom.alloc [32, 32] on @L1 : memref<32x32xf32>
          %22 = loom.alloc [32, 32] on @L1 : memref<32x32xf32>
          %23 = loom.alloc [32, 32] on @L1 : memref<32x32xf32>
          %24 = loom.alloc [32, 32, 128] on @L1 : memref<32x32x128xf32>
          %25 = loom.alloc [32, 32, 128] on @L1 : memref<32x32x128xf32>
          %26 = loom.alloc [32, 32, 128] on @L1 : memref<32x32x128xf32>
          %27 = arith.muli %15, %c4096 : index
          %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%27], sizes: [32, 32, 128], strides: [524288, 128, 1] : memref<32x4096x128xf32> to memref<32x32x128xf32, strided<[524288, 128, 1], offset: ?>>
          loom.copy %reinterpret_cast, %24 src_mem_space @DRAM dst_mem_space @L1, interconnect : [], broadcast : [1, 1] : memref<32x32x128xf32, strided<[524288, 128, 1], offset: ?>>, memref<32x32x128xf32>
          linalg.fill ins(%cst_1 : f32) outs(%25 : memref<32x32x128xf32>)
          linalg.fill ins(%cst_2 : f32) outs(%19 : memref<32x32xf32>)
          linalg.fill ins(%cst_3 : f32) outs(%20 : memref<32x32xf32>)
          scf.for %arg7 = %c0 to %c128 step %c1 {
            %28 = arith.muli %arg7, %c32 : index
            %reinterpret_cast_5 = memref.reinterpret_cast %arg0 to offset: [%28], sizes: [32, 128, 32], strides: [524288, 4096, 1] : memref<32x128x4096xf32> to memref<32x128x32xf32, strided<[524288, 4096, 1], offset: ?>>
            loom.copy %reinterpret_cast_5, %18 src_mem_space @DRAM dst_mem_space @L1, interconnect : [], broadcast : [1, 1] : memref<32x128x32xf32, strided<[524288, 4096, 1], offset: ?>>, memref<32x128x32xf32>
            linalg.fill ins(%cst_1 : f32) outs(%17 : memref<32x32x32xf32>)
            linalg.batch_matmul ins(%24, %18 : memref<32x32x128xf32>, memref<32x128x32xf32>) outs(%17 : memref<32x32x32xf32>)
            linalg.fill ins(%cst_3 : f32) outs(%21 : memref<32x32xf32>)
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%17 : memref<32x32x32xf32>) outs(%21 : memref<32x32xf32>) {
            ^bb0(%in: f32, %out: f32):
              %30 = arith.maximumf %in, %out : f32
              linalg.yield %30 : f32
            }
            linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%20, %21 : memref<32x32xf32>, memref<32x32xf32>) outs(%21 : memref<32x32xf32>) {
            ^bb0(%in: f32, %in_7: f32, %out: f32):
              %30 = arith.truncf %cst_0 : f64 to f32
              %31 = arith.mulf %in_7, %30 : f32
              %32 = arith.cmpf ogt, %in, %31 : f32
              %33 = arith.select %32, %in, %31 : f32
              linalg.yield %33 : f32
            }
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%17, %21 : memref<32x32x32xf32>, memref<32x32xf32>) outs(%17 : memref<32x32x32xf32>) {
            ^bb0(%in: f32, %in_7: f32, %out: f32):
              %30 = arith.truncf %cst_0 : f64 to f32
              %31 = arith.mulf %in, %30 : f32
              %32 = arith.subf %31, %in_7 : f32
              %33 = math.powf %cst, %32 : f32
              linalg.yield %33 : f32
            }
            linalg.fill ins(%cst_1 : f32) outs(%22 : memref<32x32xf32>)
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%17 : memref<32x32x32xf32>) outs(%22 : memref<32x32xf32>) {
            ^bb0(%in: f32, %out: f32):
              %30 = arith.addf %in, %out : f32
              linalg.yield %30 : f32
            }
            linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%20, %21 : memref<32x32xf32>, memref<32x32xf32>) outs(%23 : memref<32x32xf32>) {
            ^bb0(%in: f32, %in_7: f32, %out: f32):
              %30 = arith.subf %in, %in_7 : f32
              %31 = math.powf %cst, %30 : f32
              linalg.yield %31 : f32
            }
            linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%19, %23, %22 : memref<32x32xf32>, memref<32x32xf32>, memref<32x32xf32>) outs(%19 : memref<32x32xf32>) {
            ^bb0(%in: f32, %in_7: f32, %in_8: f32, %out: f32):
              %30 = arith.mulf %in, %in_7 : f32
              %31 = arith.addf %30, %in_8 : f32
              linalg.yield %31 : f32
            }
            %29 = arith.muli %arg7, %c4096 : index
            %reinterpret_cast_6 = memref.reinterpret_cast %arg1 to offset: [%29], sizes: [32, 32, 128], strides: [524288, 128, 1] : memref<32x4096x128xf32> to memref<32x32x128xf32, strided<[524288, 128, 1], offset: ?>>
            loom.copy %reinterpret_cast_6, %16 src_mem_space @DRAM dst_mem_space @L1, interconnect : [], broadcast : [1, 1] : memref<32x32x128xf32, strided<[524288, 128, 1], offset: ?>>, memref<32x32x128xf32>
            linalg.fill ins(%cst_1 : f32) outs(%26 : memref<32x32x128xf32>)
            linalg.batch_matmul ins(%17, %16 : memref<32x32x32xf32>, memref<32x32x128xf32>) outs(%26 : memref<32x32x128xf32>)
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%26, %25, %23 : memref<32x32x128xf32>, memref<32x32x128xf32>, memref<32x32xf32>) outs(%25 : memref<32x32x128xf32>) {
            ^bb0(%in: f32, %in_7: f32, %in_8: f32, %out: f32):
              %30 = arith.mulf %in_7, %in_8 : f32
              %31 = arith.addf %in, %30 : f32
              linalg.yield %31 : f32
            }
            linalg.copy ins(%21 : memref<32x32xf32>) outs(%20 : memref<32x32xf32>)
          }
          linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%25, %19 : memref<32x32x128xf32>, memref<32x32xf32>) outs(%24 : memref<32x32x128xf32>) {
          ^bb0(%in: f32, %in_5: f32, %out: f32):
            %28 = arith.divf %in, %in_5 : f32
            linalg.yield %28 : f32
          }
          %reinterpret_cast_4 = memref.reinterpret_cast %arg3 to offset: [%27], sizes: [32, 32, 128], strides: [524288, 128, 1] : memref<32x4096x128xf32> to memref<32x32x128xf32, strided<[524288, 128, 1], offset: ?>>
          loom.copy %24, %reinterpret_cast_4 src_mem_space @L1 dst_mem_space @DRAM, interconnect : [], broadcast : [1, 1] : memref<32x32x128xf32>, memref<32x32x128xf32, strided<[524288, 128, 1], offset: ?>>
        }
        scf.reduce 
      } {loom.iter_types = [#loom.iter_type<spatial>, #loom.iter_type<spatial>], loom.mapped_to_dims = [@x, @y]}
      return
    }
  }
  module attributes {loom.pass_name = "Materialize"} {
    func.func @attention__d0i1_d1i1__f01__d_d_a__BB32__BM32__BN32(%arg0: memref<32x128x4096xf32>, %arg1: memref<32x4096x128xf32>, %arg2: memref<32x4096x128xf32>, %arg3: memref<32x4096x128xf32>) {
      %c4096 = arith.constant 4096 : index
      %c128 = arith.constant 128 : index
      %c64 = arith.constant 64 : index
      %c2 = arith.constant 2 : index
      %cst = arith.constant 2.000000e+00 : f32
      %cst_0 = arith.constant 0.12751743074602467 : f64
      %cst_1 = arith.constant 0.000000e+00 : f32
      %cst_2 = arith.constant 1.000000e+00 : f32
      %cst_3 = arith.constant 0xFF800000 : f32
      %c32 = arith.constant 32 : index
      %c0 = arith.constant 0 : index
      %c8 = arith.constant 8 : index
      %c1 = arith.constant 1 : index
      scf.parallel (%arg4, %arg5) = (%c0, %c0) to (%c8, %c8) step (%c1, %c1) {
        scf.for %arg6 = %c0 to %c2 step %c1 {
          %12 = arith.muli %arg4, %c8 overflow<nsw> : index
          %13 = arith.addi %12, %arg5 : index
          %14 = arith.muli %arg6, %c64 overflow<nsw> : index
          %15 = arith.addi %13, %14 : index
          %16 = loom.alloc [32, 32, 128] on @L1 : memref<32x32x128xf32>
          %17 = loom.alloc [32, 32, 32] on @L1 : memref<32x32x32xf32>
          %18 = loom.alloc [32, 128, 32] on @L1 : memref<32x128x32xf32>
          %19 = loom.alloc [32, 32] on @L1 : memref<32x32xf32>
          %20 = loom.alloc [32, 32] on @L1 : memref<32x32xf32>
          %21 = loom.alloc [32, 32] on @L1 : memref<32x32xf32>
          %22 = loom.alloc [32, 32] on @L1 : memref<32x32xf32>
          %23 = loom.alloc [32, 32] on @L1 : memref<32x32xf32>
          %24 = loom.alloc [32, 32, 128] on @L1 : memref<32x32x128xf32>
          %25 = loom.alloc [32, 32, 128] on @L1 : memref<32x32x128xf32>
          %26 = loom.alloc [32, 32, 128] on @L1 : memref<32x32x128xf32>
          %27 = arith.muli %15, %c4096 : index
          %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%27], sizes: [32, 32, 128], strides: [524288, 128, 1] : memref<32x4096x128xf32> to memref<32x32x128xf32, strided<[524288, 128, 1], offset: ?>>
          loom.copy %reinterpret_cast, %24 src_mem_space @DRAM dst_mem_space @L1, interconnect : [], broadcast : [1, 1] : memref<32x32x128xf32, strided<[524288, 128, 1], offset: ?>>, memref<32x32x128xf32>
          linalg.fill ins(%cst_1 : f32) outs(%25 : memref<32x32x128xf32>)
          linalg.fill ins(%cst_2 : f32) outs(%19 : memref<32x32xf32>)
          linalg.fill ins(%cst_3 : f32) outs(%20 : memref<32x32xf32>)
          scf.for %arg7 = %c0 to %c128 step %c1 {
            %28 = arith.muli %arg7, %c32 : index
            %reinterpret_cast_5 = memref.reinterpret_cast %arg0 to offset: [%28], sizes: [32, 128, 32], strides: [524288, 4096, 1] : memref<32x128x4096xf32> to memref<32x128x32xf32, strided<[524288, 4096, 1], offset: ?>>
            loom.copy %reinterpret_cast_5, %18 src_mem_space @DRAM dst_mem_space @L1, interconnect : [], broadcast : [1, 1] : memref<32x128x32xf32, strided<[524288, 4096, 1], offset: ?>>, memref<32x128x32xf32>
            linalg.fill ins(%cst_1 : f32) outs(%17 : memref<32x32x32xf32>)
            linalg.batch_matmul ins(%24, %18 : memref<32x32x128xf32>, memref<32x128x32xf32>) outs(%17 : memref<32x32x32xf32>)
            linalg.fill ins(%cst_3 : f32) outs(%21 : memref<32x32xf32>)
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%17 : memref<32x32x32xf32>) outs(%21 : memref<32x32xf32>) {
            ^bb0(%in: f32, %out: f32):
              %30 = arith.maximumf %in, %out : f32
              linalg.yield %30 : f32
            }
            linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%20, %21 : memref<32x32xf32>, memref<32x32xf32>) outs(%21 : memref<32x32xf32>) {
            ^bb0(%in: f32, %in_7: f32, %out: f32):
              %30 = arith.truncf %cst_0 : f64 to f32
              %31 = arith.mulf %in_7, %30 : f32
              %32 = arith.cmpf ogt, %in, %31 : f32
              %33 = arith.select %32, %in, %31 : f32
              linalg.yield %33 : f32
            }
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%17, %21 : memref<32x32x32xf32>, memref<32x32xf32>) outs(%17 : memref<32x32x32xf32>) {
            ^bb0(%in: f32, %in_7: f32, %out: f32):
              %30 = arith.truncf %cst_0 : f64 to f32
              %31 = arith.mulf %in, %30 : f32
              %32 = arith.subf %31, %in_7 : f32
              %33 = math.powf %cst, %32 : f32
              linalg.yield %33 : f32
            }
            linalg.fill ins(%cst_1 : f32) outs(%22 : memref<32x32xf32>)
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%17 : memref<32x32x32xf32>) outs(%22 : memref<32x32xf32>) {
            ^bb0(%in: f32, %out: f32):
              %30 = arith.addf %in, %out : f32
              linalg.yield %30 : f32
            }
            linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%20, %21 : memref<32x32xf32>, memref<32x32xf32>) outs(%23 : memref<32x32xf32>) {
            ^bb0(%in: f32, %in_7: f32, %out: f32):
              %30 = arith.subf %in, %in_7 : f32
              %31 = math.powf %cst, %30 : f32
              linalg.yield %31 : f32
            }
            linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%19, %23, %22 : memref<32x32xf32>, memref<32x32xf32>, memref<32x32xf32>) outs(%19 : memref<32x32xf32>) {
            ^bb0(%in: f32, %in_7: f32, %in_8: f32, %out: f32):
              %30 = arith.mulf %in, %in_7 : f32
              %31 = arith.addf %30, %in_8 : f32
              linalg.yield %31 : f32
            }
            %29 = arith.muli %arg7, %c4096 : index
            %reinterpret_cast_6 = memref.reinterpret_cast %arg1 to offset: [%29], sizes: [32, 32, 128], strides: [524288, 128, 1] : memref<32x4096x128xf32> to memref<32x32x128xf32, strided<[524288, 128, 1], offset: ?>>
            loom.copy %reinterpret_cast_6, %16 src_mem_space @DRAM dst_mem_space @L1, interconnect : [@horizontal_links, @vertical_links], broadcast : [8, 8] : memref<32x32x128xf32, strided<[524288, 128, 1], offset: ?>>, memref<32x32x128xf32>
            linalg.fill ins(%cst_1 : f32) outs(%26 : memref<32x32x128xf32>)
            linalg.batch_matmul ins(%17, %16 : memref<32x32x32xf32>, memref<32x32x128xf32>) outs(%26 : memref<32x32x128xf32>)
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%26, %25, %23 : memref<32x32x128xf32>, memref<32x32x128xf32>, memref<32x32xf32>) outs(%25 : memref<32x32x128xf32>) {
            ^bb0(%in: f32, %in_7: f32, %in_8: f32, %out: f32):
              %30 = arith.mulf %in_7, %in_8 : f32
              %31 = arith.addf %in, %30 : f32
              linalg.yield %31 : f32
            }
            linalg.copy ins(%21 : memref<32x32xf32>) outs(%20 : memref<32x32xf32>)
          }
          linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%25, %19 : memref<32x32x128xf32>, memref<32x32xf32>) outs(%24 : memref<32x32x128xf32>) {
          ^bb0(%in: f32, %in_5: f32, %out: f32):
            %28 = arith.divf %in, %in_5 : f32
            linalg.yield %28 : f32
          }
          %reinterpret_cast_4 = memref.reinterpret_cast %arg3 to offset: [%27], sizes: [32, 32, 128], strides: [524288, 128, 1] : memref<32x4096x128xf32> to memref<32x32x128xf32, strided<[524288, 128, 1], offset: ?>>
          loom.copy %24, %reinterpret_cast_4 src_mem_space @L1 dst_mem_space @DRAM, interconnect : [], broadcast : [1, 1] : memref<32x32x128xf32>, memref<32x32x128xf32, strided<[524288, 128, 1], offset: ?>>
        }
        scf.reduce 
      } {loom.iter_types = [#loom.iter_type<spatial>, #loom.iter_type<spatial>], loom.mapped_to_dims = [@x, @y]}
      return
    }
  }
  module attributes {loom.pass_name = "Materialize"} {
    func.func @attention__d0i1_d1i1__f01__d_d_h__BB32__BM32__BN32(%arg0: memref<32x128x4096xf32>, %arg1: memref<32x4096x128xf32>, %arg2: memref<32x4096x128xf32>, %arg3: memref<32x4096x128xf32>) {
      %c4096 = arith.constant 4096 : index
      %c128 = arith.constant 128 : index
      %c64 = arith.constant 64 : index
      %c2 = arith.constant 2 : index
      %cst = arith.constant 2.000000e+00 : f32
      %cst_0 = arith.constant 0.12751743074602467 : f64
      %cst_1 = arith.constant 0.000000e+00 : f32
      %cst_2 = arith.constant 1.000000e+00 : f32
      %cst_3 = arith.constant 0xFF800000 : f32
      %c32 = arith.constant 32 : index
      %c0 = arith.constant 0 : index
      %c8 = arith.constant 8 : index
      %c1 = arith.constant 1 : index
      scf.parallel (%arg4, %arg5) = (%c0, %c0) to (%c8, %c8) step (%c1, %c1) {
        scf.for %arg6 = %c0 to %c2 step %c1 {
          %12 = arith.muli %arg4, %c8 overflow<nsw> : index
          %13 = arith.addi %12, %arg5 : index
          %14 = arith.muli %arg6, %c64 overflow<nsw> : index
          %15 = arith.addi %13, %14 : index
          %16 = loom.alloc [32, 32, 128] on @L1 : memref<32x32x128xf32>
          %17 = loom.alloc [32, 32, 32] on @L1 : memref<32x32x32xf32>
          %18 = loom.alloc [32, 128, 32] on @L1 : memref<32x128x32xf32>
          %19 = loom.alloc [32, 32] on @L1 : memref<32x32xf32>
          %20 = loom.alloc [32, 32] on @L1 : memref<32x32xf32>
          %21 = loom.alloc [32, 32] on @L1 : memref<32x32xf32>
          %22 = loom.alloc [32, 32] on @L1 : memref<32x32xf32>
          %23 = loom.alloc [32, 32] on @L1 : memref<32x32xf32>
          %24 = loom.alloc [32, 32, 128] on @L1 : memref<32x32x128xf32>
          %25 = loom.alloc [32, 32, 128] on @L1 : memref<32x32x128xf32>
          %26 = loom.alloc [32, 32, 128] on @L1 : memref<32x32x128xf32>
          %27 = arith.muli %15, %c4096 : index
          %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%27], sizes: [32, 32, 128], strides: [524288, 128, 1] : memref<32x4096x128xf32> to memref<32x32x128xf32, strided<[524288, 128, 1], offset: ?>>
          loom.copy %reinterpret_cast, %24 src_mem_space @DRAM dst_mem_space @L1, interconnect : [], broadcast : [1, 1] : memref<32x32x128xf32, strided<[524288, 128, 1], offset: ?>>, memref<32x32x128xf32>
          linalg.fill ins(%cst_1 : f32) outs(%25 : memref<32x32x128xf32>)
          linalg.fill ins(%cst_2 : f32) outs(%19 : memref<32x32xf32>)
          linalg.fill ins(%cst_3 : f32) outs(%20 : memref<32x32xf32>)
          scf.for %arg7 = %c0 to %c128 step %c1 {
            %28 = arith.muli %arg7, %c32 : index
            %reinterpret_cast_5 = memref.reinterpret_cast %arg0 to offset: [%28], sizes: [32, 128, 32], strides: [524288, 4096, 1] : memref<32x128x4096xf32> to memref<32x128x32xf32, strided<[524288, 4096, 1], offset: ?>>
            loom.copy %reinterpret_cast_5, %18 src_mem_space @DRAM dst_mem_space @L1, interconnect : [], broadcast : [1, 1] : memref<32x128x32xf32, strided<[524288, 4096, 1], offset: ?>>, memref<32x128x32xf32>
            linalg.fill ins(%cst_1 : f32) outs(%17 : memref<32x32x32xf32>)
            linalg.batch_matmul ins(%24, %18 : memref<32x32x128xf32>, memref<32x128x32xf32>) outs(%17 : memref<32x32x32xf32>)
            linalg.fill ins(%cst_3 : f32) outs(%21 : memref<32x32xf32>)
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%17 : memref<32x32x32xf32>) outs(%21 : memref<32x32xf32>) {
            ^bb0(%in: f32, %out: f32):
              %30 = arith.maximumf %in, %out : f32
              linalg.yield %30 : f32
            }
            linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%20, %21 : memref<32x32xf32>, memref<32x32xf32>) outs(%21 : memref<32x32xf32>) {
            ^bb0(%in: f32, %in_7: f32, %out: f32):
              %30 = arith.truncf %cst_0 : f64 to f32
              %31 = arith.mulf %in_7, %30 : f32
              %32 = arith.cmpf ogt, %in, %31 : f32
              %33 = arith.select %32, %in, %31 : f32
              linalg.yield %33 : f32
            }
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%17, %21 : memref<32x32x32xf32>, memref<32x32xf32>) outs(%17 : memref<32x32x32xf32>) {
            ^bb0(%in: f32, %in_7: f32, %out: f32):
              %30 = arith.truncf %cst_0 : f64 to f32
              %31 = arith.mulf %in, %30 : f32
              %32 = arith.subf %31, %in_7 : f32
              %33 = math.powf %cst, %32 : f32
              linalg.yield %33 : f32
            }
            linalg.fill ins(%cst_1 : f32) outs(%22 : memref<32x32xf32>)
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%17 : memref<32x32x32xf32>) outs(%22 : memref<32x32xf32>) {
            ^bb0(%in: f32, %out: f32):
              %30 = arith.addf %in, %out : f32
              linalg.yield %30 : f32
            }
            linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%20, %21 : memref<32x32xf32>, memref<32x32xf32>) outs(%23 : memref<32x32xf32>) {
            ^bb0(%in: f32, %in_7: f32, %out: f32):
              %30 = arith.subf %in, %in_7 : f32
              %31 = math.powf %cst, %30 : f32
              linalg.yield %31 : f32
            }
            linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%19, %23, %22 : memref<32x32xf32>, memref<32x32xf32>, memref<32x32xf32>) outs(%19 : memref<32x32xf32>) {
            ^bb0(%in: f32, %in_7: f32, %in_8: f32, %out: f32):
              %30 = arith.mulf %in, %in_7 : f32
              %31 = arith.addf %30, %in_8 : f32
              linalg.yield %31 : f32
            }
            %29 = arith.muli %arg7, %c4096 : index
            %reinterpret_cast_6 = memref.reinterpret_cast %arg1 to offset: [%29], sizes: [32, 32, 128], strides: [524288, 128, 1] : memref<32x4096x128xf32> to memref<32x32x128xf32, strided<[524288, 128, 1], offset: ?>>
            loom.copy %reinterpret_cast_6, %16 src_mem_space @DRAM dst_mem_space @L1, interconnect : [@horizontal_links], broadcast : [1, 8] : memref<32x32x128xf32, strided<[524288, 128, 1], offset: ?>>, memref<32x32x128xf32>
            linalg.fill ins(%cst_1 : f32) outs(%26 : memref<32x32x128xf32>)
            linalg.batch_matmul ins(%17, %16 : memref<32x32x32xf32>, memref<32x32x128xf32>) outs(%26 : memref<32x32x128xf32>)
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%26, %25, %23 : memref<32x32x128xf32>, memref<32x32x128xf32>, memref<32x32xf32>) outs(%25 : memref<32x32x128xf32>) {
            ^bb0(%in: f32, %in_7: f32, %in_8: f32, %out: f32):
              %30 = arith.mulf %in_7, %in_8 : f32
              %31 = arith.addf %in, %30 : f32
              linalg.yield %31 : f32
            }
            linalg.copy ins(%21 : memref<32x32xf32>) outs(%20 : memref<32x32xf32>)
          }
          linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%25, %19 : memref<32x32x128xf32>, memref<32x32xf32>) outs(%24 : memref<32x32x128xf32>) {
          ^bb0(%in: f32, %in_5: f32, %out: f32):
            %28 = arith.divf %in, %in_5 : f32
            linalg.yield %28 : f32
          }
          %reinterpret_cast_4 = memref.reinterpret_cast %arg3 to offset: [%27], sizes: [32, 32, 128], strides: [524288, 128, 1] : memref<32x4096x128xf32> to memref<32x32x128xf32, strided<[524288, 128, 1], offset: ?>>
          loom.copy %24, %reinterpret_cast_4 src_mem_space @L1 dst_mem_space @DRAM, interconnect : [], broadcast : [1, 1] : memref<32x32x128xf32>, memref<32x32x128xf32, strided<[524288, 128, 1], offset: ?>>
        }
        scf.reduce 
      } {loom.iter_types = [#loom.iter_type<spatial>, #loom.iter_type<spatial>], loom.mapped_to_dims = [@x, @y]}
      return
    }
  }
  module attributes {loom.pass_name = "Materialize"} {
    func.func @attention__d0i1_d1i1__f01__d_d_v__BB32__BM32__BN32(%arg0: memref<32x128x4096xf32>, %arg1: memref<32x4096x128xf32>, %arg2: memref<32x4096x128xf32>, %arg3: memref<32x4096x128xf32>) {
      %c4096 = arith.constant 4096 : index
      %c128 = arith.constant 128 : index
      %c64 = arith.constant 64 : index
      %c2 = arith.constant 2 : index
      %cst = arith.constant 2.000000e+00 : f32
      %cst_0 = arith.constant 0.12751743074602467 : f64
      %cst_1 = arith.constant 0.000000e+00 : f32
      %cst_2 = arith.constant 1.000000e+00 : f32
      %cst_3 = arith.constant 0xFF800000 : f32
      %c32 = arith.constant 32 : index
      %c0 = arith.constant 0 : index
      %c8 = arith.constant 8 : index
      %c1 = arith.constant 1 : index
      scf.parallel (%arg4, %arg5) = (%c0, %c0) to (%c8, %c8) step (%c1, %c1) {
        scf.for %arg6 = %c0 to %c2 step %c1 {
          %12 = arith.muli %arg4, %c8 overflow<nsw> : index
          %13 = arith.addi %12, %arg5 : index
          %14 = arith.muli %arg6, %c64 overflow<nsw> : index
          %15 = arith.addi %13, %14 : index
          %16 = loom.alloc [32, 32, 128] on @L1 : memref<32x32x128xf32>
          %17 = loom.alloc [32, 32, 32] on @L1 : memref<32x32x32xf32>
          %18 = loom.alloc [32, 128, 32] on @L1 : memref<32x128x32xf32>
          %19 = loom.alloc [32, 32] on @L1 : memref<32x32xf32>
          %20 = loom.alloc [32, 32] on @L1 : memref<32x32xf32>
          %21 = loom.alloc [32, 32] on @L1 : memref<32x32xf32>
          %22 = loom.alloc [32, 32] on @L1 : memref<32x32xf32>
          %23 = loom.alloc [32, 32] on @L1 : memref<32x32xf32>
          %24 = loom.alloc [32, 32, 128] on @L1 : memref<32x32x128xf32>
          %25 = loom.alloc [32, 32, 128] on @L1 : memref<32x32x128xf32>
          %26 = loom.alloc [32, 32, 128] on @L1 : memref<32x32x128xf32>
          %27 = arith.muli %15, %c4096 : index
          %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%27], sizes: [32, 32, 128], strides: [524288, 128, 1] : memref<32x4096x128xf32> to memref<32x32x128xf32, strided<[524288, 128, 1], offset: ?>>
          loom.copy %reinterpret_cast, %24 src_mem_space @DRAM dst_mem_space @L1, interconnect : [], broadcast : [1, 1] : memref<32x32x128xf32, strided<[524288, 128, 1], offset: ?>>, memref<32x32x128xf32>
          linalg.fill ins(%cst_1 : f32) outs(%25 : memref<32x32x128xf32>)
          linalg.fill ins(%cst_2 : f32) outs(%19 : memref<32x32xf32>)
          linalg.fill ins(%cst_3 : f32) outs(%20 : memref<32x32xf32>)
          scf.for %arg7 = %c0 to %c128 step %c1 {
            %28 = arith.muli %arg7, %c32 : index
            %reinterpret_cast_5 = memref.reinterpret_cast %arg0 to offset: [%28], sizes: [32, 128, 32], strides: [524288, 4096, 1] : memref<32x128x4096xf32> to memref<32x128x32xf32, strided<[524288, 4096, 1], offset: ?>>
            loom.copy %reinterpret_cast_5, %18 src_mem_space @DRAM dst_mem_space @L1, interconnect : [], broadcast : [1, 1] : memref<32x128x32xf32, strided<[524288, 4096, 1], offset: ?>>, memref<32x128x32xf32>
            linalg.fill ins(%cst_1 : f32) outs(%17 : memref<32x32x32xf32>)
            linalg.batch_matmul ins(%24, %18 : memref<32x32x128xf32>, memref<32x128x32xf32>) outs(%17 : memref<32x32x32xf32>)
            linalg.fill ins(%cst_3 : f32) outs(%21 : memref<32x32xf32>)
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%17 : memref<32x32x32xf32>) outs(%21 : memref<32x32xf32>) {
            ^bb0(%in: f32, %out: f32):
              %30 = arith.maximumf %in, %out : f32
              linalg.yield %30 : f32
            }
            linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%20, %21 : memref<32x32xf32>, memref<32x32xf32>) outs(%21 : memref<32x32xf32>) {
            ^bb0(%in: f32, %in_7: f32, %out: f32):
              %30 = arith.truncf %cst_0 : f64 to f32
              %31 = arith.mulf %in_7, %30 : f32
              %32 = arith.cmpf ogt, %in, %31 : f32
              %33 = arith.select %32, %in, %31 : f32
              linalg.yield %33 : f32
            }
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%17, %21 : memref<32x32x32xf32>, memref<32x32xf32>) outs(%17 : memref<32x32x32xf32>) {
            ^bb0(%in: f32, %in_7: f32, %out: f32):
              %30 = arith.truncf %cst_0 : f64 to f32
              %31 = arith.mulf %in, %30 : f32
              %32 = arith.subf %31, %in_7 : f32
              %33 = math.powf %cst, %32 : f32
              linalg.yield %33 : f32
            }
            linalg.fill ins(%cst_1 : f32) outs(%22 : memref<32x32xf32>)
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%17 : memref<32x32x32xf32>) outs(%22 : memref<32x32xf32>) {
            ^bb0(%in: f32, %out: f32):
              %30 = arith.addf %in, %out : f32
              linalg.yield %30 : f32
            }
            linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%20, %21 : memref<32x32xf32>, memref<32x32xf32>) outs(%23 : memref<32x32xf32>) {
            ^bb0(%in: f32, %in_7: f32, %out: f32):
              %30 = arith.subf %in, %in_7 : f32
              %31 = math.powf %cst, %30 : f32
              linalg.yield %31 : f32
            }
            linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%19, %23, %22 : memref<32x32xf32>, memref<32x32xf32>, memref<32x32xf32>) outs(%19 : memref<32x32xf32>) {
            ^bb0(%in: f32, %in_7: f32, %in_8: f32, %out: f32):
              %30 = arith.mulf %in, %in_7 : f32
              %31 = arith.addf %30, %in_8 : f32
              linalg.yield %31 : f32
            }
            %29 = arith.muli %arg7, %c4096 : index
            %reinterpret_cast_6 = memref.reinterpret_cast %arg1 to offset: [%29], sizes: [32, 32, 128], strides: [524288, 128, 1] : memref<32x4096x128xf32> to memref<32x32x128xf32, strided<[524288, 128, 1], offset: ?>>
            loom.copy %reinterpret_cast_6, %16 src_mem_space @DRAM dst_mem_space @L1, interconnect : [@vertical_links], broadcast : [8, 1] : memref<32x32x128xf32, strided<[524288, 128, 1], offset: ?>>, memref<32x32x128xf32>
            linalg.fill ins(%cst_1 : f32) outs(%26 : memref<32x32x128xf32>)
            linalg.batch_matmul ins(%17, %16 : memref<32x32x32xf32>, memref<32x32x128xf32>) outs(%26 : memref<32x32x128xf32>)
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%26, %25, %23 : memref<32x32x128xf32>, memref<32x32x128xf32>, memref<32x32xf32>) outs(%25 : memref<32x32x128xf32>) {
            ^bb0(%in: f32, %in_7: f32, %in_8: f32, %out: f32):
              %30 = arith.mulf %in_7, %in_8 : f32
              %31 = arith.addf %in, %30 : f32
              linalg.yield %31 : f32
            }
            linalg.copy ins(%21 : memref<32x32xf32>) outs(%20 : memref<32x32xf32>)
          }
          linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%25, %19 : memref<32x32x128xf32>, memref<32x32xf32>) outs(%24 : memref<32x32x128xf32>) {
          ^bb0(%in: f32, %in_5: f32, %out: f32):
            %28 = arith.divf %in, %in_5 : f32
            linalg.yield %28 : f32
          }
          %reinterpret_cast_4 = memref.reinterpret_cast %arg3 to offset: [%27], sizes: [32, 32, 128], strides: [524288, 128, 1] : memref<32x4096x128xf32> to memref<32x32x128xf32, strided<[524288, 128, 1], offset: ?>>
          loom.copy %24, %reinterpret_cast_4 src_mem_space @L1 dst_mem_space @DRAM, interconnect : [], broadcast : [1, 1] : memref<32x32x128xf32>, memref<32x32x128xf32, strided<[524288, 128, 1], offset: ?>>
        }
        scf.reduce 
      } {loom.iter_types = [#loom.iter_type<spatial>, #loom.iter_type<spatial>], loom.mapped_to_dims = [@x, @y]}
      return
    }
  }
  module attributes {loom.pass_name = "Materialize"} {
    func.func @attention__d0i1_d1i1__f01__d_a_d__BB32__BM32__BN32(%arg0: memref<32x128x4096xf32>, %arg1: memref<32x4096x128xf32>, %arg2: memref<32x4096x128xf32>, %arg3: memref<32x4096x128xf32>) {
      %c4096 = arith.constant 4096 : index
      %c128 = arith.constant 128 : index
      %c64 = arith.constant 64 : index
      %c2 = arith.constant 2 : index
      %cst = arith.constant 2.000000e+00 : f32
      %cst_0 = arith.constant 0.12751743074602467 : f64
      %cst_1 = arith.constant 0.000000e+00 : f32
      %cst_2 = arith.constant 1.000000e+00 : f32
      %cst_3 = arith.constant 0xFF800000 : f32
      %c32 = arith.constant 32 : index
      %c0 = arith.constant 0 : index
      %c8 = arith.constant 8 : index
      %c1 = arith.constant 1 : index
      scf.parallel (%arg4, %arg5) = (%c0, %c0) to (%c8, %c8) step (%c1, %c1) {
        scf.for %arg6 = %c0 to %c2 step %c1 {
          %12 = arith.muli %arg4, %c8 overflow<nsw> : index
          %13 = arith.addi %12, %arg5 : index
          %14 = arith.muli %arg6, %c64 overflow<nsw> : index
          %15 = arith.addi %13, %14 : index
          %16 = loom.alloc [32, 32, 128] on @L1 : memref<32x32x128xf32>
          %17 = loom.alloc [32, 32, 32] on @L1 : memref<32x32x32xf32>
          %18 = loom.alloc [32, 128, 32] on @L1 : memref<32x128x32xf32>
          %19 = loom.alloc [32, 32] on @L1 : memref<32x32xf32>
          %20 = loom.alloc [32, 32] on @L1 : memref<32x32xf32>
          %21 = loom.alloc [32, 32] on @L1 : memref<32x32xf32>
          %22 = loom.alloc [32, 32] on @L1 : memref<32x32xf32>
          %23 = loom.alloc [32, 32] on @L1 : memref<32x32xf32>
          %24 = loom.alloc [32, 32, 128] on @L1 : memref<32x32x128xf32>
          %25 = loom.alloc [32, 32, 128] on @L1 : memref<32x32x128xf32>
          %26 = loom.alloc [32, 32, 128] on @L1 : memref<32x32x128xf32>
          %27 = arith.muli %15, %c4096 : index
          %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%27], sizes: [32, 32, 128], strides: [524288, 128, 1] : memref<32x4096x128xf32> to memref<32x32x128xf32, strided<[524288, 128, 1], offset: ?>>
          loom.copy %reinterpret_cast, %24 src_mem_space @DRAM dst_mem_space @L1, interconnect : [], broadcast : [1, 1] : memref<32x32x128xf32, strided<[524288, 128, 1], offset: ?>>, memref<32x32x128xf32>
          linalg.fill ins(%cst_1 : f32) outs(%25 : memref<32x32x128xf32>)
          linalg.fill ins(%cst_2 : f32) outs(%19 : memref<32x32xf32>)
          linalg.fill ins(%cst_3 : f32) outs(%20 : memref<32x32xf32>)
          scf.for %arg7 = %c0 to %c128 step %c1 {
            %28 = arith.muli %arg7, %c32 : index
            %reinterpret_cast_5 = memref.reinterpret_cast %arg0 to offset: [%28], sizes: [32, 128, 32], strides: [524288, 4096, 1] : memref<32x128x4096xf32> to memref<32x128x32xf32, strided<[524288, 4096, 1], offset: ?>>
            loom.copy %reinterpret_cast_5, %18 src_mem_space @DRAM dst_mem_space @L1, interconnect : [@horizontal_links, @vertical_links], broadcast : [8, 8] : memref<32x128x32xf32, strided<[524288, 4096, 1], offset: ?>>, memref<32x128x32xf32>
            linalg.fill ins(%cst_1 : f32) outs(%17 : memref<32x32x32xf32>)
            linalg.batch_matmul ins(%24, %18 : memref<32x32x128xf32>, memref<32x128x32xf32>) outs(%17 : memref<32x32x32xf32>)
            linalg.fill ins(%cst_3 : f32) outs(%21 : memref<32x32xf32>)
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%17 : memref<32x32x32xf32>) outs(%21 : memref<32x32xf32>) {
            ^bb0(%in: f32, %out: f32):
              %30 = arith.maximumf %in, %out : f32
              linalg.yield %30 : f32
            }
            linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%20, %21 : memref<32x32xf32>, memref<32x32xf32>) outs(%21 : memref<32x32xf32>) {
            ^bb0(%in: f32, %in_7: f32, %out: f32):
              %30 = arith.truncf %cst_0 : f64 to f32
              %31 = arith.mulf %in_7, %30 : f32
              %32 = arith.cmpf ogt, %in, %31 : f32
              %33 = arith.select %32, %in, %31 : f32
              linalg.yield %33 : f32
            }
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%17, %21 : memref<32x32x32xf32>, memref<32x32xf32>) outs(%17 : memref<32x32x32xf32>) {
            ^bb0(%in: f32, %in_7: f32, %out: f32):
              %30 = arith.truncf %cst_0 : f64 to f32
              %31 = arith.mulf %in, %30 : f32
              %32 = arith.subf %31, %in_7 : f32
              %33 = math.powf %cst, %32 : f32
              linalg.yield %33 : f32
            }
            linalg.fill ins(%cst_1 : f32) outs(%22 : memref<32x32xf32>)
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%17 : memref<32x32x32xf32>) outs(%22 : memref<32x32xf32>) {
            ^bb0(%in: f32, %out: f32):
              %30 = arith.addf %in, %out : f32
              linalg.yield %30 : f32
            }
            linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%20, %21 : memref<32x32xf32>, memref<32x32xf32>) outs(%23 : memref<32x32xf32>) {
            ^bb0(%in: f32, %in_7: f32, %out: f32):
              %30 = arith.subf %in, %in_7 : f32
              %31 = math.powf %cst, %30 : f32
              linalg.yield %31 : f32
            }
            linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%19, %23, %22 : memref<32x32xf32>, memref<32x32xf32>, memref<32x32xf32>) outs(%19 : memref<32x32xf32>) {
            ^bb0(%in: f32, %in_7: f32, %in_8: f32, %out: f32):
              %30 = arith.mulf %in, %in_7 : f32
              %31 = arith.addf %30, %in_8 : f32
              linalg.yield %31 : f32
            }
            %29 = arith.muli %arg7, %c4096 : index
            %reinterpret_cast_6 = memref.reinterpret_cast %arg1 to offset: [%29], sizes: [32, 32, 128], strides: [524288, 128, 1] : memref<32x4096x128xf32> to memref<32x32x128xf32, strided<[524288, 128, 1], offset: ?>>
            loom.copy %reinterpret_cast_6, %16 src_mem_space @DRAM dst_mem_space @L1, interconnect : [], broadcast : [1, 1] : memref<32x32x128xf32, strided<[524288, 128, 1], offset: ?>>, memref<32x32x128xf32>
            linalg.fill ins(%cst_1 : f32) outs(%26 : memref<32x32x128xf32>)
            linalg.batch_matmul ins(%17, %16 : memref<32x32x32xf32>, memref<32x32x128xf32>) outs(%26 : memref<32x32x128xf32>)
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%26, %25, %23 : memref<32x32x128xf32>, memref<32x32x128xf32>, memref<32x32xf32>) outs(%25 : memref<32x32x128xf32>) {
            ^bb0(%in: f32, %in_7: f32, %in_8: f32, %out: f32):
              %30 = arith.mulf %in_7, %in_8 : f32
              %31 = arith.addf %in, %30 : f32
              linalg.yield %31 : f32
            }
            linalg.copy ins(%21 : memref<32x32xf32>) outs(%20 : memref<32x32xf32>)
          }
          linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%25, %19 : memref<32x32x128xf32>, memref<32x32xf32>) outs(%24 : memref<32x32x128xf32>) {
          ^bb0(%in: f32, %in_5: f32, %out: f32):
            %28 = arith.divf %in, %in_5 : f32
            linalg.yield %28 : f32
          }
          %reinterpret_cast_4 = memref.reinterpret_cast %arg3 to offset: [%27], sizes: [32, 32, 128], strides: [524288, 128, 1] : memref<32x4096x128xf32> to memref<32x32x128xf32, strided<[524288, 128, 1], offset: ?>>
          loom.copy %24, %reinterpret_cast_4 src_mem_space @L1 dst_mem_space @DRAM, interconnect : [], broadcast : [1, 1] : memref<32x32x128xf32>, memref<32x32x128xf32, strided<[524288, 128, 1], offset: ?>>
        }
        scf.reduce 
      } {loom.iter_types = [#loom.iter_type<spatial>, #loom.iter_type<spatial>], loom.mapped_to_dims = [@x, @y]}
      return
    }
  }
  // =====================================================================================
  // Variant: attention__d0i1_d1i1__f01__d_a_a__BB32__BM32__BN32
  //
  //   Naming breakdown:
  //     d0i1_d1i1 : spatial dim 0 (x) → interconnect idx 1 (horizontal_links),
  //                 spatial dim 1 (y) → interconnect idx 1 (vertical_links)
  //     f01       : function arg ordering — arg0=K, arg1=V, arg2=Q (original order)
  //     d_a_a     : copy strategy per operand — K^T=direct(d), V=all-broadcast(a), Q=... (see below)
  //                 Q copy uses broadcast:[1,1] (direct, no interconnect).
  //                 K^T copy uses [@horizontal_links, @vertical_links], broadcast:[8,8] (all-broadcast).
  //                 V copy uses [@horizontal_links, @vertical_links], broadcast:[8,8] (all-broadcast).
  //     BB=32, BM=32, BN=32 : tile sizes
  //
  //   Global tensor shapes (after per-core tiling by 8x8 mesh):
  //     K  (arg0): memref<32x128x4096xf32>  — [BB, head_dim, N_total]
  //     V  (arg1): memref<32x4096x128xf32>  — [BB, N_total, head_dim]
  //     Q  (arg2): memref<32x4096x128xf32>  — [BB, M_total, head_dim]
  //     O  (arg3): memref<32x4096x128xf32>  — [BB, M_total, head_dim]
  //   where BB=32 (batch after tiling: ceildiv(2,32)=1 batch tile of 32 batches padded),
  //         M_total=4096, N_total=4096, head_dim=128.
  //
  //   Loop structure:
  //     Outer: scf.parallel (%arg4, %arg5) in [0,8) x [0,8) — 8x8 spatial cores
  //     Middle: scf.for %arg6 in [0,2) — 2 iterations over linearized (batch, M) tiles
  //       Each iteration processes one tile: %15 = %arg4*8 + %arg5 + %arg6*64
  //       This gives tile_idx ∈ [0, 127]. Since M_total/BM = 4096/32 = 128 tiles,
  //       and BB=32 so ceildiv(ceildiv(2,32) * ceildiv(4096,32), 64) = ceildiv(128, 64) = 2.
  //     Inner: scf.for %arg7 in [0,128) — 128 iterations over N-dimension tiles
  //       (ceildiv(4096, BN) = ceildiv(4096, 32) = 128 iterations)
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
    func.func @attention__d0i1_d1i1__f01__d_a_a__BB32__BM32__BN32(%arg0: memref<32x128x4096xf32>, %arg1: memref<32x4096x128xf32>, %arg2: memref<32x4096x128xf32>, %arg3: memref<32x4096x128xf32>) {
      // arg0 = K (Key matrix, transposed view), shape [BB=32, head_dim=128, N_total=4096]
      // arg1 = V (Value matrix),               shape [BB=32, N_total=4096, head_dim=128]
      // arg2 = Q (Query matrix),               shape [BB=32, M_total=4096, head_dim=128]
      // arg3 = O (Output matrix),              shape [BB=32, M_total=4096, head_dim=128]
      %c4096 = arith.constant 4096 : index       // N_total (also M_total)
      %c128 = arith.constant 128 : index          // head_dim (also inner loop bound = N_total/BN = 4096/32)
      %c64 = arith.constant 64 : index            // number of (batch,M) tiles per core group (8*8=64)
      %c2 = arith.constant 2 : index              // outer tile loop bound = ceildiv(128, 64) = 2
      %cst = arith.constant 2.000000e+00 : f32    // base for exp2
      %cst_0 = arith.constant 0.12751743074602467 : f64 // qk_scale = 1/sqrt(head_dim) ≈ 1/sqrt(128)
      %cst_1 = arith.constant 0.000000e+00 : f32  // zero (for fill)
      %cst_2 = arith.constant 1.000000e+00 : f32  // one (for l_i init)
      %cst_3 = arith.constant 0xFF800000 : f32    // -inf (for m_i init and amax init)
      %c32 = arith.constant 32 : index            // BM = BN = BB = 32
      %c0 = arith.constant 0 : index
      %c8 = arith.constant 8 : index              // spatial mesh dimension (8x8)
      %c1 = arith.constant 1 : index
      // =================== Spatial Parallel: 8x8 core mesh ===================
      // %arg4 = x-dim core index [0,8), %arg5 = y-dim core index [0,8)
      scf.parallel (%arg4, %arg5) = (%c0, %c0) to (%c8, %c8) step (%c1, %c1) {
        // =================== Middle Loop: iterate over (batch, M) tile groups ===================
        // %arg6 ∈ [0, 2): each core processes 2 tile groups
        scf.for %arg6 = %c0 to %c2 step %c1 {
          // ---- Compute linearized tile index ----
          // %12 = %arg4 * 8                  (core_x contributes row offset)
          %12 = arith.muli %arg4, %c8 overflow<nsw> : index
          // %13 = %arg4 * 8 + %arg5          (linearized core index within 8x8 mesh, range [0,63])
          %13 = arith.addi %12, %arg5 : index
          // %14 = %arg6 * 64                 (tile group offset: 0 or 64)
          %14 = arith.muli %arg6, %c64 overflow<nsw> : index
          // %15 = core_linear + group_offset = (%arg4*8 + %arg5) + %arg6*64
          //     range: [0, 127] — linearized (batch, M) tile index
          //     Since BB=32, ceildiv(2,32)=1, so batch_tile=0 always.
          //     Effective M tile index = %15, selects rows [%15*BM, %15*BM+BM) = [%15*32, %15*32+32)
          %15 = arith.addi %13, %14 : index

          // =================== L1 Memory Allocations (11 buffers) ===================
          // Same alloc count as other variants after SinkFillOps optimization.
          //
          // --- 3D buffers ---
          %16 = loom.alloc [32, 32, 128] on @L1 : memref<32x32x128xf32>    // V tile [BB, BN, head_dim] (loaded from DRAM each inner iter)
          %17 = loom.alloc [32, 32, 32] on @L1 : memref<32x32x32xf32>      // qk = Q @ K^T [BB, BM, BN], later reused in-place for p
          %18 = loom.alloc [32, 128, 32] on @L1 : memref<32x128x32xf32>    // K^T tile [BB, head_dim, BN] (loaded from DRAM each inner iter)
          // --- 2D buffers (BB x BM = 32 x 32) ---
          %19 = loom.alloc [32, 32] on @L1 : memref<32x32xf32>             // l_i: running softmax denominator (in-place across iters)
          %20 = loom.alloc [32, 32] on @L1 : memref<32x32xf32>             // m_i: running row-wise max (updated via copy at end of iter)
          %21 = loom.alloc [32, 32] on @L1 : memref<32x32xf32>             // m_ij: new row-wise max for current iter (also holds intermediate amax)
          %22 = loom.alloc [32, 32] on @L1 : memref<32x32xf32>             // l_ij: row-wise sum of p for current iter
          %23 = loom.alloc [32, 32] on @L1 : memref<32x32xf32>             // alpha = 2^(m_i - m_ij), rescaling factor
          // --- 3D buffers (reused / accumulator) ---
          %24 = loom.alloc [32, 32, 128] on @L1 : memref<32x32x128xf32>    // Q tile [BB, BM, head_dim] (loaded once); REUSED post-loop for output = acc / l_i
          %25 = loom.alloc [32, 32, 128] on @L1 : memref<32x32x128xf32>    // acc (accumulator, updated in-place across iters)
          %26 = loom.alloc [32, 32, 128] on @L1 : memref<32x32x128xf32>    // p @ V intermediate result (per-iter scratch)

          // =================== Load Q tile from DRAM to L1 ===================
          // Q_tile = Q[batch_all, m_start:m_end, :]
          // where m_start = %15 * BM, m_end = %15 * BM + BM
          //
          // Address calculation for Q (arg2, shape [32, 4096, 128]):
          //   %27 = %15 * 4096
          //   Logical index: Q[b, m, h] at flat offset = b*4096*128 + m*128 + h
          //                                            = b*524288 + m*128 + h
          //   We want to access rows [%15*32 .. %15*32+31] across all batches.
          //   Since the view covers all BB=32 batches, the offset only encodes the M-tile start:
          //     offset = %15 * head_dim = %15 * 128?
          //   Actually: offset = %15 * 4096.
          //     Verification: For %15 = tile_idx, this represents the M-row tile index.
          //     But strides are [524288, 128, 1], with sizes [32, 32, 128].
          //     A flat element at [b, m_local, h] is at: offset + b*524288 + m_local*128 + h
          //       = %15*4096 + b*524288 + m_local*128 + h
          //     This should equal: b * (4096*128) + (%15*32 + m_local) * 128 + h
          //       = b*524288 + %15*32*128 + m_local*128 + h
          //       = b*524288 + %15*4096 + m_local*128 + h  ✓
          //     So %27 = %15 * 4096 = %15 * BM * head_dim = %15 * 32 * 128 ✓
          %27 = arith.muli %15, %c4096 : index
          %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%27], sizes: [32, 32, 128], strides: [524288, 128, 1] : memref<32x4096x128xf32> to memref<32x32x128xf32, strided<[524288, 128, 1], offset: ?>>
          // DMA: DRAM → L1, load Q tile into %24 (direct, no broadcast)
          loom.copy %reinterpret_cast, %24 src_mem_space @DRAM dst_mem_space @L1, interconnect : [], broadcast : [1, 1] : memref<32x32x128xf32, strided<[524288, 128, 1], offset: ?>>, memref<32x32x128xf32>

          // =================== Loop Initialization ===================
          // acc = zeros([BB, BM, head_dim])
          // Memory: writes to %25, out-of-place (fresh init)
          linalg.fill ins(%cst_1 : f32) outs(%25 : memref<32x32x128xf32>)
          // l_i = full([BB, BM], 1.0)
          // Memory: writes to %19, out-of-place (fresh init)
          linalg.fill ins(%cst_2 : f32) outs(%19 : memref<32x32xf32>)
          // m_i = full([BB, BM], -inf)
          // Memory: writes to %20, out-of-place (fresh init)
          linalg.fill ins(%cst_3 : f32) outs(%20 : memref<32x32xf32>)

          // =================== Inner Loop: iterate over K/V tiles (128 iters) ===================
          // for n_iter in range(ceildiv(4096, BN)):  (ceildiv(4096,32) = 128 iterations)
          scf.for %arg7 = %c0 to %c128 step %c1 {
            // ---- Load K^T tile from DRAM ----
            // K_tile = K[batch_all, :, n_start:n_end]  (transposed view for matmul)
            // where n_start = %arg7 * BN, n_end = n_start + BN
            //
            // Address calculation for K^T (arg0, shape [32, 128, 4096]):
            //   %28 = %arg7 * 32 = %arg7 * BN
            //   Logical index: K[b, h, n] at flat offset = b*128*4096 + h*4096 + n
            //                                            = b*524288 + h*4096 + n
            //   The view: offset=%28, sizes=[32,128,32], strides=[524288, 4096, 1]
            //   Element [b, h_local, n_local] is at: %28 + b*524288 + h_local*4096 + n_local
            //     = %arg7*32 + b*524288 + h_local*4096 + n_local
            //   Should equal: b*524288 + h_local*4096 + (%arg7*32 + n_local)
            //     = b*524288 + h_local*4096 + %arg7*32 + n_local  ✓
            //   So %28 = %arg7 * BN correctly selects the N-tile starting column ✓
            %28 = arith.muli %arg7, %c32 : index
            %reinterpret_cast_5 = memref.reinterpret_cast %arg0 to offset: [%28], sizes: [32, 128, 32], strides: [524288, 4096, 1] : memref<32x128x4096xf32> to memref<32x128x32xf32, strided<[524288, 4096, 1], offset: ?>>
            // DMA: DRAM → L1, load K^T tile into %18
            // Interconnect: ALL-BROADCAST via horizontal_links + vertical_links, broadcast [8,8]
            // One core loads and the data is replicated to all 8*8=64 cores.
            loom.copy %reinterpret_cast_5, %18 src_mem_space @DRAM dst_mem_space @L1, interconnect : [@horizontal_links, @vertical_links], broadcast : [8, 8] : memref<32x128x32xf32, strided<[524288, 4096, 1], offset: ?>>, memref<32x128x32xf32>

            // ---- qk = Q @ K^T ----
            // Zero-init qk buffer, then compute batch matmul
            // Memory: %17 is zeroed then written by batch_matmul (out-of-place, accumulation pattern)
            // NOTE: fill sunk from pre-loop to here by SinkFillOps pass
            linalg.fill ins(%cst_1 : f32) outs(%17 : memref<32x32x32xf32>)
            // qk = Q_tile @ K_tile^T, result in %17
            // shapes: [32,32,128] @ [32,128,32] → [32,32,32]
            linalg.batch_matmul ins(%24, %18 : memref<32x32x128xf32>, memref<32x128x32xf32>) outs(%17 : memref<32x32x32xf32>)

            // ---- amax(qk, -1): row-wise max of qk ----
            // Memory: %21 is first filled with -inf, then reduced over (out-of-place init + in-place reduction)
            // NOTE: fill sunk from pre-loop to here by SinkFillOps pass
            linalg.fill ins(%cst_3 : f32) outs(%21 : memref<32x32xf32>)
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%17 : memref<32x32x32xf32>) outs(%21 : memref<32x32xf32>) {
            ^bb0(%in: f32, %out: f32):
              %30 = arith.maximumf %in, %out : f32
              linalg.yield %30 : f32
            }
            // After this: %21 = amax(qk, dim=-1), shape [32, 32]

            // ---- m_ij = max(m_i, amax(qk) * qk_scale) ----
            // Reads m_i from %20, reads amax from %21, writes result back to %21
            // Memory: IN-PLACE on %21 — the amax result is overwritten with m_ij.
            //         This is safe because amax is only needed to compute m_ij itself.
            linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%20, %21 : memref<32x32xf32>, memref<32x32xf32>) outs(%21 : memref<32x32xf32>) {
            ^bb0(%in: f32, %in_7: f32, %out: f32):
              %30 = arith.truncf %cst_0 : f64 to f32
              %31 = arith.mulf %in_7, %30 : f32
              %32 = arith.cmpf ogt, %in, %31 : f32
              %33 = arith.select %32, %in, %31 : f32
              linalg.yield %33 : f32
            }
            // After this: %21 = m_ij = max(m_i, amax(qk) * qk_scale)

            // ---- p = exp2(qk * qk_scale - m_ij) ----
            // Reads qk from %17, reads m_ij from %21, writes p back to %17
            // Memory: IN-PLACE on %17 — qk is overwritten with p.
            //         qk is no longer needed after this point (consumed by amax above).
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%17, %21 : memref<32x32x32xf32>, memref<32x32xf32>) outs(%17 : memref<32x32x32xf32>) {
            ^bb0(%in: f32, %in_7: f32, %out: f32):
              %30 = arith.truncf %cst_0 : f64 to f32
              %31 = arith.mulf %in, %30 : f32
              %32 = arith.subf %31, %in_7 : f32
              %33 = math.powf %cst, %32 : f32
              linalg.yield %33 : f32
            }
            // After this: %17 = p (attention weights), shape [32, 32, 32]

            // ---- l_ij = sum(p, dim=-1) ----
            // Memory: %22 is zeroed then reduced in-place
            // NOTE: fill sunk from pre-loop to here by SinkFillOps pass
            linalg.fill ins(%cst_1 : f32) outs(%22 : memref<32x32xf32>)
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%17 : memref<32x32x32xf32>) outs(%22 : memref<32x32xf32>) {
            ^bb0(%in: f32, %out: f32):
              %30 = arith.addf %in, %out : f32
              linalg.yield %30 : f32
            }
            // After this: %22 = l_ij = sum(p, dim=-1), shape [32, 32]

            // ---- alpha = 2^(m_i - m_ij) ----
            // Reads m_i from %20, reads m_ij from %21, writes alpha to %23
            // Memory: OUT-OF-PLACE, result in separate buffer %23
            linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%20, %21 : memref<32x32xf32>, memref<32x32xf32>) outs(%23 : memref<32x32xf32>) {
            ^bb0(%in: f32, %in_7: f32, %out: f32):
              %30 = arith.subf %in, %in_7 : f32
              %31 = math.powf %cst, %30 : f32
              linalg.yield %31 : f32
            }
            // After this: %23 = alpha = 2^(m_i - m_ij), shape [32, 32]

            // ---- l_i = alpha * l_i + l_ij ----
            // Reads l_i from %19, alpha from %23, l_ij from %22, writes back to %19
            // Memory: IN-PLACE on %19 — the running l_i is updated in its own buffer.
            //         This is the key in-place update for the online softmax denominator.
            linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%19, %23, %22 : memref<32x32xf32>, memref<32x32xf32>, memref<32x32xf32>) outs(%19 : memref<32x32xf32>) {
            ^bb0(%in: f32, %in_7: f32, %in_8: f32, %out: f32):
              %30 = arith.mulf %in, %in_7 : f32
              %31 = arith.addf %30, %in_8 : f32
              linalg.yield %31 : f32
            }
            // After this: %19 = updated l_i

            // ---- Load V tile from DRAM ----
            // V_tile = V[batch_all, n_start:n_end, :]
            // where n_start = %arg7 * BN, n_end = n_start + BN
            //
            // Address calculation for V (arg1, shape [32, 4096, 128]):
            //   %29 = %arg7 * 4096
            //   Logical index: V[b, n, h] at flat offset = b*4096*128 + n*128 + h
            //                                            = b*524288 + n*128 + h
            //   The view: offset=%29, sizes=[32, 32, 128], strides=[524288, 128, 1]
            //   Element [b, n_local, h] is at: %29 + b*524288 + n_local*128 + h
            //     = %arg7*4096 + b*524288 + n_local*128 + h
            //   Should equal: b*524288 + (%arg7*32 + n_local)*128 + h
            //     = b*524288 + %arg7*32*128 + n_local*128 + h
            //     = b*524288 + %arg7*4096 + n_local*128 + h  ✓
            //   So %29 = %arg7 * 4096 = %arg7 * BN * head_dim = %arg7 * 32 * 128 ✓
            %29 = arith.muli %arg7, %c4096 : index
            %reinterpret_cast_6 = memref.reinterpret_cast %arg1 to offset: [%29], sizes: [32, 32, 128], strides: [524288, 128, 1] : memref<32x4096x128xf32> to memref<32x32x128xf32, strided<[524288, 128, 1], offset: ?>>
            // DMA: DRAM → L1, load V tile into %16
            // Interconnect: ALL-BROADCAST via horizontal_links + vertical_links, broadcast [8,8]
            // One core loads and the data is replicated to all 8*8=64 cores.
            loom.copy %reinterpret_cast_6, %16 src_mem_space @DRAM dst_mem_space @L1, interconnect : [@horizontal_links, @vertical_links], broadcast : [8, 8] : memref<32x32x128xf32, strided<[524288, 128, 1], offset: ?>>, memref<32x32x128xf32>

            // ---- pv = p @ V ----
            // Memory: %26 is zeroed then used as accumulator for batch_matmul (out-of-place)
            // NOTE: fill sunk from pre-loop to here by SinkFillOps pass
            linalg.fill ins(%cst_1 : f32) outs(%26 : memref<32x32x128xf32>)
            // p @ V: [32,32,32] @ [32,32,128] → [32,32,128], result in %26
            linalg.batch_matmul ins(%17, %16 : memref<32x32x32xf32>, memref<32x32x128xf32>) outs(%26 : memref<32x32x128xf32>)

            // ---- acc = pv + acc * alpha ----
            // Reads pv from %26, reads previous acc from %25, reads alpha from %23, writes to %25
            // Memory: IN-PLACE on %25 — the running accumulator is updated.
            //         This is the core FlashAttention rescaling: acc = (p @ V) + acc * alpha
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%26, %25, %23 : memref<32x32x128xf32>, memref<32x32x128xf32>, memref<32x32xf32>) outs(%25 : memref<32x32x128xf32>) {
            ^bb0(%in: f32, %in_7: f32, %in_8: f32, %out: f32):
              %30 = arith.mulf %in_7, %in_8 : f32
              %31 = arith.addf %in, %30 : f32
              linalg.yield %31 : f32
            }
            // After this: %25 = updated acc

            // ---- m_i = m_ij (carry forward the max for next iteration) ----
            // Memory: explicit copy from %21 (m_ij) to %20 (m_i).
            // This is needed because OSB cannot alias m_i and m_ij (they are live simultaneously
            // during the alpha computation), so a separate copy is required at iteration end.
            linalg.copy ins(%21 : memref<32x32xf32>) outs(%20 : memref<32x32xf32>)
          }
          // =================== Post-loop: normalize by l_i ===================
          // output = acc / l_i
          // Reads final acc from %25, final l_i from %19, writes to %24
          // Memory: OUT-OF-PLACE — result goes to %24 (which previously held Q tile).
          //         Q is no longer needed after the inner loop, so %24 is safely reused here.
          //         This is an improvement: the old version needed a separate buffer for this.
          linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%25, %19 : memref<32x32x128xf32>, memref<32x32xf32>) outs(%24 : memref<32x32x128xf32>) {
          ^bb0(%in: f32, %in_5: f32, %out: f32):
            %28 = arith.divf %in, %in_5 : f32
            linalg.yield %28 : f32
          }

          // =================== Store output tile to DRAM ===================
          // O[batch_all, m_start:m_end, :] = output
          //
          // Address calculation for O (arg3, shape [32, 4096, 128]):
          //   Reuses %27 = %15 * 4096 (same offset as Q load, since Q and O share the same tiling)
          //   The view: offset=%27, sizes=[32, 32, 128], strides=[524288, 128, 1]
          //   Element [b, m_local, h] is at: %27 + b*524288 + m_local*128 + h
          //     = %15*4096 + b*524288 + m_local*128 + h
          //   Should equal: b*524288 + (%15*32 + m_local)*128 + h
          //     = b*524288 + %15*4096 + m_local*128 + h  ✓
          //   So output is written to the correct M-tile slice of O ✓
          %reinterpret_cast_4 = memref.reinterpret_cast %arg3 to offset: [%27], sizes: [32, 32, 128], strides: [524288, 128, 1] : memref<32x4096x128xf32> to memref<32x32x128xf32, strided<[524288, 128, 1], offset: ?>>
          // DMA: L1 → DRAM, store result from %24 to output (direct, no broadcast)
          loom.copy %24, %reinterpret_cast_4 src_mem_space @L1 dst_mem_space @DRAM, interconnect : [], broadcast : [1, 1] : memref<32x32x128xf32>, memref<32x32x128xf32, strided<[524288, 128, 1], offset: ?>>
        }
        scf.reduce 
      } {loom.iter_types = [#loom.iter_type<spatial>, #loom.iter_type<spatial>], loom.mapped_to_dims = [@x, @y]}
      return
    }
  }
  module attributes {loom.pass_name = "Materialize"} {
    func.func @attention__d0i1_d1i1__f01__d_a_h__BB32__BM32__BN32(%arg0: memref<32x128x4096xf32>, %arg1: memref<32x4096x128xf32>, %arg2: memref<32x4096x128xf32>, %arg3: memref<32x4096x128xf32>) {
      %c4096 = arith.constant 4096 : index
      %c128 = arith.constant 128 : index
      %c64 = arith.constant 64 : index
      %c2 = arith.constant 2 : index
      %cst = arith.constant 2.000000e+00 : f32
      %cst_0 = arith.constant 0.12751743074602467 : f64
      %cst_1 = arith.constant 0.000000e+00 : f32
      %cst_2 = arith.constant 1.000000e+00 : f32
      %cst_3 = arith.constant 0xFF800000 : f32
      %c32 = arith.constant 32 : index
      %c0 = arith.constant 0 : index
      %c8 = arith.constant 8 : index
      %c1 = arith.constant 1 : index
      scf.parallel (%arg4, %arg5) = (%c0, %c0) to (%c8, %c8) step (%c1, %c1) {
        scf.for %arg6 = %c0 to %c2 step %c1 {
          %12 = arith.muli %arg4, %c8 overflow<nsw> : index
          %13 = arith.addi %12, %arg5 : index
          %14 = arith.muli %arg6, %c64 overflow<nsw> : index
          %15 = arith.addi %13, %14 : index
          %16 = loom.alloc [32, 32, 128] on @L1 : memref<32x32x128xf32>
          %17 = loom.alloc [32, 32, 32] on @L1 : memref<32x32x32xf32>
          %18 = loom.alloc [32, 128, 32] on @L1 : memref<32x128x32xf32>
          %19 = loom.alloc [32, 32] on @L1 : memref<32x32xf32>
          %20 = loom.alloc [32, 32] on @L1 : memref<32x32xf32>
          %21 = loom.alloc [32, 32] on @L1 : memref<32x32xf32>
          %22 = loom.alloc [32, 32] on @L1 : memref<32x32xf32>
          %23 = loom.alloc [32, 32] on @L1 : memref<32x32xf32>
          %24 = loom.alloc [32, 32, 128] on @L1 : memref<32x32x128xf32>
          %25 = loom.alloc [32, 32, 128] on @L1 : memref<32x32x128xf32>
          %26 = loom.alloc [32, 32, 128] on @L1 : memref<32x32x128xf32>
          %27 = arith.muli %15, %c4096 : index
          %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%27], sizes: [32, 32, 128], strides: [524288, 128, 1] : memref<32x4096x128xf32> to memref<32x32x128xf32, strided<[524288, 128, 1], offset: ?>>
          loom.copy %reinterpret_cast, %24 src_mem_space @DRAM dst_mem_space @L1, interconnect : [], broadcast : [1, 1] : memref<32x32x128xf32, strided<[524288, 128, 1], offset: ?>>, memref<32x32x128xf32>
          linalg.fill ins(%cst_1 : f32) outs(%25 : memref<32x32x128xf32>)
          linalg.fill ins(%cst_2 : f32) outs(%19 : memref<32x32xf32>)
          linalg.fill ins(%cst_3 : f32) outs(%20 : memref<32x32xf32>)
          scf.for %arg7 = %c0 to %c128 step %c1 {
            %28 = arith.muli %arg7, %c32 : index
            %reinterpret_cast_5 = memref.reinterpret_cast %arg0 to offset: [%28], sizes: [32, 128, 32], strides: [524288, 4096, 1] : memref<32x128x4096xf32> to memref<32x128x32xf32, strided<[524288, 4096, 1], offset: ?>>
            loom.copy %reinterpret_cast_5, %18 src_mem_space @DRAM dst_mem_space @L1, interconnect : [@horizontal_links, @vertical_links], broadcast : [8, 8] : memref<32x128x32xf32, strided<[524288, 4096, 1], offset: ?>>, memref<32x128x32xf32>
            linalg.fill ins(%cst_1 : f32) outs(%17 : memref<32x32x32xf32>)
            linalg.batch_matmul ins(%24, %18 : memref<32x32x128xf32>, memref<32x128x32xf32>) outs(%17 : memref<32x32x32xf32>)
            linalg.fill ins(%cst_3 : f32) outs(%21 : memref<32x32xf32>)
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%17 : memref<32x32x32xf32>) outs(%21 : memref<32x32xf32>) {
            ^bb0(%in: f32, %out: f32):
              %30 = arith.maximumf %in, %out : f32
              linalg.yield %30 : f32
            }
            linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%20, %21 : memref<32x32xf32>, memref<32x32xf32>) outs(%21 : memref<32x32xf32>) {
            ^bb0(%in: f32, %in_7: f32, %out: f32):
              %30 = arith.truncf %cst_0 : f64 to f32
              %31 = arith.mulf %in_7, %30 : f32
              %32 = arith.cmpf ogt, %in, %31 : f32
              %33 = arith.select %32, %in, %31 : f32
              linalg.yield %33 : f32
            }
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%17, %21 : memref<32x32x32xf32>, memref<32x32xf32>) outs(%17 : memref<32x32x32xf32>) {
            ^bb0(%in: f32, %in_7: f32, %out: f32):
              %30 = arith.truncf %cst_0 : f64 to f32
              %31 = arith.mulf %in, %30 : f32
              %32 = arith.subf %31, %in_7 : f32
              %33 = math.powf %cst, %32 : f32
              linalg.yield %33 : f32
            }
            linalg.fill ins(%cst_1 : f32) outs(%22 : memref<32x32xf32>)
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%17 : memref<32x32x32xf32>) outs(%22 : memref<32x32xf32>) {
            ^bb0(%in: f32, %out: f32):
              %30 = arith.addf %in, %out : f32
              linalg.yield %30 : f32
            }
            linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%20, %21 : memref<32x32xf32>, memref<32x32xf32>) outs(%23 : memref<32x32xf32>) {
            ^bb0(%in: f32, %in_7: f32, %out: f32):
              %30 = arith.subf %in, %in_7 : f32
              %31 = math.powf %cst, %30 : f32
              linalg.yield %31 : f32
            }
            linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%19, %23, %22 : memref<32x32xf32>, memref<32x32xf32>, memref<32x32xf32>) outs(%19 : memref<32x32xf32>) {
            ^bb0(%in: f32, %in_7: f32, %in_8: f32, %out: f32):
              %30 = arith.mulf %in, %in_7 : f32
              %31 = arith.addf %30, %in_8 : f32
              linalg.yield %31 : f32
            }
            %29 = arith.muli %arg7, %c4096 : index
            %reinterpret_cast_6 = memref.reinterpret_cast %arg1 to offset: [%29], sizes: [32, 32, 128], strides: [524288, 128, 1] : memref<32x4096x128xf32> to memref<32x32x128xf32, strided<[524288, 128, 1], offset: ?>>
            loom.copy %reinterpret_cast_6, %16 src_mem_space @DRAM dst_mem_space @L1, interconnect : [@horizontal_links], broadcast : [1, 8] : memref<32x32x128xf32, strided<[524288, 128, 1], offset: ?>>, memref<32x32x128xf32>
            linalg.fill ins(%cst_1 : f32) outs(%26 : memref<32x32x128xf32>)
            linalg.batch_matmul ins(%17, %16 : memref<32x32x32xf32>, memref<32x32x128xf32>) outs(%26 : memref<32x32x128xf32>)
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%26, %25, %23 : memref<32x32x128xf32>, memref<32x32x128xf32>, memref<32x32xf32>) outs(%25 : memref<32x32x128xf32>) {
            ^bb0(%in: f32, %in_7: f32, %in_8: f32, %out: f32):
              %30 = arith.mulf %in_7, %in_8 : f32
              %31 = arith.addf %in, %30 : f32
              linalg.yield %31 : f32
            }
            linalg.copy ins(%21 : memref<32x32xf32>) outs(%20 : memref<32x32xf32>)
          }
          linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%25, %19 : memref<32x32x128xf32>, memref<32x32xf32>) outs(%24 : memref<32x32x128xf32>) {
          ^bb0(%in: f32, %in_5: f32, %out: f32):
            %28 = arith.divf %in, %in_5 : f32
            linalg.yield %28 : f32
          }
          %reinterpret_cast_4 = memref.reinterpret_cast %arg3 to offset: [%27], sizes: [32, 32, 128], strides: [524288, 128, 1] : memref<32x4096x128xf32> to memref<32x32x128xf32, strided<[524288, 128, 1], offset: ?>>
          loom.copy %24, %reinterpret_cast_4 src_mem_space @L1 dst_mem_space @DRAM, interconnect : [], broadcast : [1, 1] : memref<32x32x128xf32>, memref<32x32x128xf32, strided<[524288, 128, 1], offset: ?>>
        }
        scf.reduce 
      } {loom.iter_types = [#loom.iter_type<spatial>, #loom.iter_type<spatial>], loom.mapped_to_dims = [@x, @y]}
      return
    }
  }
  module attributes {loom.pass_name = "Materialize"} {
    func.func @attention__d0i1_d1i1__f01__d_a_v__BB32__BM32__BN32(%arg0: memref<32x128x4096xf32>, %arg1: memref<32x4096x128xf32>, %arg2: memref<32x4096x128xf32>, %arg3: memref<32x4096x128xf32>) {
      %c4096 = arith.constant 4096 : index
      %c128 = arith.constant 128 : index
      %c64 = arith.constant 64 : index
      %c2 = arith.constant 2 : index
      %cst = arith.constant 2.000000e+00 : f32
      %cst_0 = arith.constant 0.12751743074602467 : f64
      %cst_1 = arith.constant 0.000000e+00 : f32
      %cst_2 = arith.constant 1.000000e+00 : f32
      %cst_3 = arith.constant 0xFF800000 : f32
      %c32 = arith.constant 32 : index
      %c0 = arith.constant 0 : index
      %c8 = arith.constant 8 : index
      %c1 = arith.constant 1 : index
      scf.parallel (%arg4, %arg5) = (%c0, %c0) to (%c8, %c8) step (%c1, %c1) {
        scf.for %arg6 = %c0 to %c2 step %c1 {
          %12 = arith.muli %arg4, %c8 overflow<nsw> : index
          %13 = arith.addi %12, %arg5 : index
          %14 = arith.muli %arg6, %c64 overflow<nsw> : index
          %15 = arith.addi %13, %14 : index
          %16 = loom.alloc [32, 32, 128] on @L1 : memref<32x32x128xf32>
          %17 = loom.alloc [32, 32, 32] on @L1 : memref<32x32x32xf32>
          %18 = loom.alloc [32, 128, 32] on @L1 : memref<32x128x32xf32>
          %19 = loom.alloc [32, 32] on @L1 : memref<32x32xf32>
          %20 = loom.alloc [32, 32] on @L1 : memref<32x32xf32>
          %21 = loom.alloc [32, 32] on @L1 : memref<32x32xf32>
          %22 = loom.alloc [32, 32] on @L1 : memref<32x32xf32>
          %23 = loom.alloc [32, 32] on @L1 : memref<32x32xf32>
          %24 = loom.alloc [32, 32, 128] on @L1 : memref<32x32x128xf32>
          %25 = loom.alloc [32, 32, 128] on @L1 : memref<32x32x128xf32>
          %26 = loom.alloc [32, 32, 128] on @L1 : memref<32x32x128xf32>
          %27 = arith.muli %15, %c4096 : index
          %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%27], sizes: [32, 32, 128], strides: [524288, 128, 1] : memref<32x4096x128xf32> to memref<32x32x128xf32, strided<[524288, 128, 1], offset: ?>>
          loom.copy %reinterpret_cast, %24 src_mem_space @DRAM dst_mem_space @L1, interconnect : [], broadcast : [1, 1] : memref<32x32x128xf32, strided<[524288, 128, 1], offset: ?>>, memref<32x32x128xf32>
          linalg.fill ins(%cst_1 : f32) outs(%25 : memref<32x32x128xf32>)
          linalg.fill ins(%cst_2 : f32) outs(%19 : memref<32x32xf32>)
          linalg.fill ins(%cst_3 : f32) outs(%20 : memref<32x32xf32>)
          scf.for %arg7 = %c0 to %c128 step %c1 {
            %28 = arith.muli %arg7, %c32 : index
            %reinterpret_cast_5 = memref.reinterpret_cast %arg0 to offset: [%28], sizes: [32, 128, 32], strides: [524288, 4096, 1] : memref<32x128x4096xf32> to memref<32x128x32xf32, strided<[524288, 4096, 1], offset: ?>>
            loom.copy %reinterpret_cast_5, %18 src_mem_space @DRAM dst_mem_space @L1, interconnect : [@horizontal_links, @vertical_links], broadcast : [8, 8] : memref<32x128x32xf32, strided<[524288, 4096, 1], offset: ?>>, memref<32x128x32xf32>
            linalg.fill ins(%cst_1 : f32) outs(%17 : memref<32x32x32xf32>)
            linalg.batch_matmul ins(%24, %18 : memref<32x32x128xf32>, memref<32x128x32xf32>) outs(%17 : memref<32x32x32xf32>)
            linalg.fill ins(%cst_3 : f32) outs(%21 : memref<32x32xf32>)
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%17 : memref<32x32x32xf32>) outs(%21 : memref<32x32xf32>) {
            ^bb0(%in: f32, %out: f32):
              %30 = arith.maximumf %in, %out : f32
              linalg.yield %30 : f32
            }
            linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%20, %21 : memref<32x32xf32>, memref<32x32xf32>) outs(%21 : memref<32x32xf32>) {
            ^bb0(%in: f32, %in_7: f32, %out: f32):
              %30 = arith.truncf %cst_0 : f64 to f32
              %31 = arith.mulf %in_7, %30 : f32
              %32 = arith.cmpf ogt, %in, %31 : f32
              %33 = arith.select %32, %in, %31 : f32
              linalg.yield %33 : f32
            }
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%17, %21 : memref<32x32x32xf32>, memref<32x32xf32>) outs(%17 : memref<32x32x32xf32>) {
            ^bb0(%in: f32, %in_7: f32, %out: f32):
              %30 = arith.truncf %cst_0 : f64 to f32
              %31 = arith.mulf %in, %30 : f32
              %32 = arith.subf %31, %in_7 : f32
              %33 = math.powf %cst, %32 : f32
              linalg.yield %33 : f32
            }
            linalg.fill ins(%cst_1 : f32) outs(%22 : memref<32x32xf32>)
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%17 : memref<32x32x32xf32>) outs(%22 : memref<32x32xf32>) {
            ^bb0(%in: f32, %out: f32):
              %30 = arith.addf %in, %out : f32
              linalg.yield %30 : f32
            }
            linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%20, %21 : memref<32x32xf32>, memref<32x32xf32>) outs(%23 : memref<32x32xf32>) {
            ^bb0(%in: f32, %in_7: f32, %out: f32):
              %30 = arith.subf %in, %in_7 : f32
              %31 = math.powf %cst, %30 : f32
              linalg.yield %31 : f32
            }
            linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%19, %23, %22 : memref<32x32xf32>, memref<32x32xf32>, memref<32x32xf32>) outs(%19 : memref<32x32xf32>) {
            ^bb0(%in: f32, %in_7: f32, %in_8: f32, %out: f32):
              %30 = arith.mulf %in, %in_7 : f32
              %31 = arith.addf %30, %in_8 : f32
              linalg.yield %31 : f32
            }
            %29 = arith.muli %arg7, %c4096 : index
            %reinterpret_cast_6 = memref.reinterpret_cast %arg1 to offset: [%29], sizes: [32, 32, 128], strides: [524288, 128, 1] : memref<32x4096x128xf32> to memref<32x32x128xf32, strided<[524288, 128, 1], offset: ?>>
            loom.copy %reinterpret_cast_6, %16 src_mem_space @DRAM dst_mem_space @L1, interconnect : [@vertical_links], broadcast : [8, 1] : memref<32x32x128xf32, strided<[524288, 128, 1], offset: ?>>, memref<32x32x128xf32>
            linalg.fill ins(%cst_1 : f32) outs(%26 : memref<32x32x128xf32>)
            linalg.batch_matmul ins(%17, %16 : memref<32x32x32xf32>, memref<32x32x128xf32>) outs(%26 : memref<32x32x128xf32>)
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%26, %25, %23 : memref<32x32x128xf32>, memref<32x32x128xf32>, memref<32x32xf32>) outs(%25 : memref<32x32x128xf32>) {
            ^bb0(%in: f32, %in_7: f32, %in_8: f32, %out: f32):
              %30 = arith.mulf %in_7, %in_8 : f32
              %31 = arith.addf %in, %30 : f32
              linalg.yield %31 : f32
            }
            linalg.copy ins(%21 : memref<32x32xf32>) outs(%20 : memref<32x32xf32>)
          }
          linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%25, %19 : memref<32x32x128xf32>, memref<32x32xf32>) outs(%24 : memref<32x32x128xf32>) {
          ^bb0(%in: f32, %in_5: f32, %out: f32):
            %28 = arith.divf %in, %in_5 : f32
            linalg.yield %28 : f32
          }
          %reinterpret_cast_4 = memref.reinterpret_cast %arg3 to offset: [%27], sizes: [32, 32, 128], strides: [524288, 128, 1] : memref<32x4096x128xf32> to memref<32x32x128xf32, strided<[524288, 128, 1], offset: ?>>
          loom.copy %24, %reinterpret_cast_4 src_mem_space @L1 dst_mem_space @DRAM, interconnect : [], broadcast : [1, 1] : memref<32x32x128xf32>, memref<32x32x128xf32, strided<[524288, 128, 1], offset: ?>>
        }
        scf.reduce 
      } {loom.iter_types = [#loom.iter_type<spatial>, #loom.iter_type<spatial>], loom.mapped_to_dims = [@x, @y]}
      return
    }
  }
  module attributes {loom.pass_name = "Materialize"} {
    func.func @attention__d0i1_d1i1__f01__d_h_d__BB32__BM32__BN32(%arg0: memref<32x128x4096xf32>, %arg1: memref<32x4096x128xf32>, %arg2: memref<32x4096x128xf32>, %arg3: memref<32x4096x128xf32>) {
      %c4096 = arith.constant 4096 : index
      %c128 = arith.constant 128 : index
      %c64 = arith.constant 64 : index
      %c2 = arith.constant 2 : index
      %cst = arith.constant 2.000000e+00 : f32
      %cst_0 = arith.constant 0.12751743074602467 : f64
      %cst_1 = arith.constant 0.000000e+00 : f32
      %cst_2 = arith.constant 1.000000e+00 : f32
      %cst_3 = arith.constant 0xFF800000 : f32
      %c32 = arith.constant 32 : index
      %c0 = arith.constant 0 : index
      %c8 = arith.constant 8 : index
      %c1 = arith.constant 1 : index
      scf.parallel (%arg4, %arg5) = (%c0, %c0) to (%c8, %c8) step (%c1, %c1) {
        scf.for %arg6 = %c0 to %c2 step %c1 {
          %12 = arith.muli %arg4, %c8 overflow<nsw> : index
          %13 = arith.addi %12, %arg5 : index
          %14 = arith.muli %arg6, %c64 overflow<nsw> : index
          %15 = arith.addi %13, %14 : index
          %16 = loom.alloc [32, 32, 128] on @L1 : memref<32x32x128xf32>
          %17 = loom.alloc [32, 32, 32] on @L1 : memref<32x32x32xf32>
          %18 = loom.alloc [32, 128, 32] on @L1 : memref<32x128x32xf32>
          %19 = loom.alloc [32, 32] on @L1 : memref<32x32xf32>
          %20 = loom.alloc [32, 32] on @L1 : memref<32x32xf32>
          %21 = loom.alloc [32, 32] on @L1 : memref<32x32xf32>
          %22 = loom.alloc [32, 32] on @L1 : memref<32x32xf32>
          %23 = loom.alloc [32, 32] on @L1 : memref<32x32xf32>
          %24 = loom.alloc [32, 32, 128] on @L1 : memref<32x32x128xf32>
          %25 = loom.alloc [32, 32, 128] on @L1 : memref<32x32x128xf32>
          %26 = loom.alloc [32, 32, 128] on @L1 : memref<32x32x128xf32>
          %27 = arith.muli %15, %c4096 : index
          %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%27], sizes: [32, 32, 128], strides: [524288, 128, 1] : memref<32x4096x128xf32> to memref<32x32x128xf32, strided<[524288, 128, 1], offset: ?>>
          loom.copy %reinterpret_cast, %24 src_mem_space @DRAM dst_mem_space @L1, interconnect : [], broadcast : [1, 1] : memref<32x32x128xf32, strided<[524288, 128, 1], offset: ?>>, memref<32x32x128xf32>
          linalg.fill ins(%cst_1 : f32) outs(%25 : memref<32x32x128xf32>)
          linalg.fill ins(%cst_2 : f32) outs(%19 : memref<32x32xf32>)
          linalg.fill ins(%cst_3 : f32) outs(%20 : memref<32x32xf32>)
          scf.for %arg7 = %c0 to %c128 step %c1 {
            %28 = arith.muli %arg7, %c32 : index
            %reinterpret_cast_5 = memref.reinterpret_cast %arg0 to offset: [%28], sizes: [32, 128, 32], strides: [524288, 4096, 1] : memref<32x128x4096xf32> to memref<32x128x32xf32, strided<[524288, 4096, 1], offset: ?>>
            loom.copy %reinterpret_cast_5, %18 src_mem_space @DRAM dst_mem_space @L1, interconnect : [@horizontal_links], broadcast : [1, 8] : memref<32x128x32xf32, strided<[524288, 4096, 1], offset: ?>>, memref<32x128x32xf32>
            linalg.fill ins(%cst_1 : f32) outs(%17 : memref<32x32x32xf32>)
            linalg.batch_matmul ins(%24, %18 : memref<32x32x128xf32>, memref<32x128x32xf32>) outs(%17 : memref<32x32x32xf32>)
            linalg.fill ins(%cst_3 : f32) outs(%21 : memref<32x32xf32>)
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%17 : memref<32x32x32xf32>) outs(%21 : memref<32x32xf32>) {
            ^bb0(%in: f32, %out: f32):
              %30 = arith.maximumf %in, %out : f32
              linalg.yield %30 : f32
            }
            linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%20, %21 : memref<32x32xf32>, memref<32x32xf32>) outs(%21 : memref<32x32xf32>) {
            ^bb0(%in: f32, %in_7: f32, %out: f32):
              %30 = arith.truncf %cst_0 : f64 to f32
              %31 = arith.mulf %in_7, %30 : f32
              %32 = arith.cmpf ogt, %in, %31 : f32
              %33 = arith.select %32, %in, %31 : f32
              linalg.yield %33 : f32
            }
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%17, %21 : memref<32x32x32xf32>, memref<32x32xf32>) outs(%17 : memref<32x32x32xf32>) {
            ^bb0(%in: f32, %in_7: f32, %out: f32):
              %30 = arith.truncf %cst_0 : f64 to f32
              %31 = arith.mulf %in, %30 : f32
              %32 = arith.subf %31, %in_7 : f32
              %33 = math.powf %cst, %32 : f32
              linalg.yield %33 : f32
            }
            linalg.fill ins(%cst_1 : f32) outs(%22 : memref<32x32xf32>)
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%17 : memref<32x32x32xf32>) outs(%22 : memref<32x32xf32>) {
            ^bb0(%in: f32, %out: f32):
              %30 = arith.addf %in, %out : f32
              linalg.yield %30 : f32
            }
            linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%20, %21 : memref<32x32xf32>, memref<32x32xf32>) outs(%23 : memref<32x32xf32>) {
            ^bb0(%in: f32, %in_7: f32, %out: f32):
              %30 = arith.subf %in, %in_7 : f32
              %31 = math.powf %cst, %30 : f32
              linalg.yield %31 : f32
            }
            linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%19, %23, %22 : memref<32x32xf32>, memref<32x32xf32>, memref<32x32xf32>) outs(%19 : memref<32x32xf32>) {
            ^bb0(%in: f32, %in_7: f32, %in_8: f32, %out: f32):
              %30 = arith.mulf %in, %in_7 : f32
              %31 = arith.addf %30, %in_8 : f32
              linalg.yield %31 : f32
            }
            %29 = arith.muli %arg7, %c4096 : index
            %reinterpret_cast_6 = memref.reinterpret_cast %arg1 to offset: [%29], sizes: [32, 32, 128], strides: [524288, 128, 1] : memref<32x4096x128xf32> to memref<32x32x128xf32, strided<[524288, 128, 1], offset: ?>>
            loom.copy %reinterpret_cast_6, %16 src_mem_space @DRAM dst_mem_space @L1, interconnect : [], broadcast : [1, 1] : memref<32x32x128xf32, strided<[524288, 128, 1], offset: ?>>, memref<32x32x128xf32>
            linalg.fill ins(%cst_1 : f32) outs(%26 : memref<32x32x128xf32>)
            linalg.batch_matmul ins(%17, %16 : memref<32x32x32xf32>, memref<32x32x128xf32>) outs(%26 : memref<32x32x128xf32>)
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%26, %25, %23 : memref<32x32x128xf32>, memref<32x32x128xf32>, memref<32x32xf32>) outs(%25 : memref<32x32x128xf32>) {
            ^bb0(%in: f32, %in_7: f32, %in_8: f32, %out: f32):
              %30 = arith.mulf %in_7, %in_8 : f32
              %31 = arith.addf %in, %30 : f32
              linalg.yield %31 : f32
            }
            linalg.copy ins(%21 : memref<32x32xf32>) outs(%20 : memref<32x32xf32>)
          }
          linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%25, %19 : memref<32x32x128xf32>, memref<32x32xf32>) outs(%24 : memref<32x32x128xf32>) {
          ^bb0(%in: f32, %in_5: f32, %out: f32):
            %28 = arith.divf %in, %in_5 : f32
            linalg.yield %28 : f32
          }
          %reinterpret_cast_4 = memref.reinterpret_cast %arg3 to offset: [%27], sizes: [32, 32, 128], strides: [524288, 128, 1] : memref<32x4096x128xf32> to memref<32x32x128xf32, strided<[524288, 128, 1], offset: ?>>
          loom.copy %24, %reinterpret_cast_4 src_mem_space @L1 dst_mem_space @DRAM, interconnect : [], broadcast : [1, 1] : memref<32x32x128xf32>, memref<32x32x128xf32, strided<[524288, 128, 1], offset: ?>>
        }
        scf.reduce 
      } {loom.iter_types = [#loom.iter_type<spatial>, #loom.iter_type<spatial>], loom.mapped_to_dims = [@x, @y]}
      return
    }
  }
  module attributes {loom.pass_name = "Materialize"} {
    func.func @attention__d0i1_d1i1__f01__d_h_a__BB32__BM32__BN32(%arg0: memref<32x128x4096xf32>, %arg1: memref<32x4096x128xf32>, %arg2: memref<32x4096x128xf32>, %arg3: memref<32x4096x128xf32>) {
      %c4096 = arith.constant 4096 : index
      %c128 = arith.constant 128 : index
      %c64 = arith.constant 64 : index
      %c2 = arith.constant 2 : index
      %cst = arith.constant 2.000000e+00 : f32
      %cst_0 = arith.constant 0.12751743074602467 : f64
      %cst_1 = arith.constant 0.000000e+00 : f32
      %cst_2 = arith.constant 1.000000e+00 : f32
      %cst_3 = arith.constant 0xFF800000 : f32
      %c32 = arith.constant 32 : index
      %c0 = arith.constant 0 : index
      %c8 = arith.constant 8 : index
      %c1 = arith.constant 1 : index
      scf.parallel (%arg4, %arg5) = (%c0, %c0) to (%c8, %c8) step (%c1, %c1) {
        scf.for %arg6 = %c0 to %c2 step %c1 {
          %12 = arith.muli %arg4, %c8 overflow<nsw> : index
          %13 = arith.addi %12, %arg5 : index
          %14 = arith.muli %arg6, %c64 overflow<nsw> : index
          %15 = arith.addi %13, %14 : index
          %16 = loom.alloc [32, 32, 128] on @L1 : memref<32x32x128xf32>
          %17 = loom.alloc [32, 32, 32] on @L1 : memref<32x32x32xf32>
          %18 = loom.alloc [32, 128, 32] on @L1 : memref<32x128x32xf32>
          %19 = loom.alloc [32, 32] on @L1 : memref<32x32xf32>
          %20 = loom.alloc [32, 32] on @L1 : memref<32x32xf32>
          %21 = loom.alloc [32, 32] on @L1 : memref<32x32xf32>
          %22 = loom.alloc [32, 32] on @L1 : memref<32x32xf32>
          %23 = loom.alloc [32, 32] on @L1 : memref<32x32xf32>
          %24 = loom.alloc [32, 32, 128] on @L1 : memref<32x32x128xf32>
          %25 = loom.alloc [32, 32, 128] on @L1 : memref<32x32x128xf32>
          %26 = loom.alloc [32, 32, 128] on @L1 : memref<32x32x128xf32>
          %27 = arith.muli %15, %c4096 : index
          %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%27], sizes: [32, 32, 128], strides: [524288, 128, 1] : memref<32x4096x128xf32> to memref<32x32x128xf32, strided<[524288, 128, 1], offset: ?>>
          loom.copy %reinterpret_cast, %24 src_mem_space @DRAM dst_mem_space @L1, interconnect : [], broadcast : [1, 1] : memref<32x32x128xf32, strided<[524288, 128, 1], offset: ?>>, memref<32x32x128xf32>
          linalg.fill ins(%cst_1 : f32) outs(%25 : memref<32x32x128xf32>)
          linalg.fill ins(%cst_2 : f32) outs(%19 : memref<32x32xf32>)
          linalg.fill ins(%cst_3 : f32) outs(%20 : memref<32x32xf32>)
          scf.for %arg7 = %c0 to %c128 step %c1 {
            %28 = arith.muli %arg7, %c32 : index
            %reinterpret_cast_5 = memref.reinterpret_cast %arg0 to offset: [%28], sizes: [32, 128, 32], strides: [524288, 4096, 1] : memref<32x128x4096xf32> to memref<32x128x32xf32, strided<[524288, 4096, 1], offset: ?>>
            loom.copy %reinterpret_cast_5, %18 src_mem_space @DRAM dst_mem_space @L1, interconnect : [@horizontal_links], broadcast : [1, 8] : memref<32x128x32xf32, strided<[524288, 4096, 1], offset: ?>>, memref<32x128x32xf32>
            linalg.fill ins(%cst_1 : f32) outs(%17 : memref<32x32x32xf32>)
            linalg.batch_matmul ins(%24, %18 : memref<32x32x128xf32>, memref<32x128x32xf32>) outs(%17 : memref<32x32x32xf32>)
            linalg.fill ins(%cst_3 : f32) outs(%21 : memref<32x32xf32>)
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%17 : memref<32x32x32xf32>) outs(%21 : memref<32x32xf32>) {
            ^bb0(%in: f32, %out: f32):
              %30 = arith.maximumf %in, %out : f32
              linalg.yield %30 : f32
            }
            linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%20, %21 : memref<32x32xf32>, memref<32x32xf32>) outs(%21 : memref<32x32xf32>) {
            ^bb0(%in: f32, %in_7: f32, %out: f32):
              %30 = arith.truncf %cst_0 : f64 to f32
              %31 = arith.mulf %in_7, %30 : f32
              %32 = arith.cmpf ogt, %in, %31 : f32
              %33 = arith.select %32, %in, %31 : f32
              linalg.yield %33 : f32
            }
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%17, %21 : memref<32x32x32xf32>, memref<32x32xf32>) outs(%17 : memref<32x32x32xf32>) {
            ^bb0(%in: f32, %in_7: f32, %out: f32):
              %30 = arith.truncf %cst_0 : f64 to f32
              %31 = arith.mulf %in, %30 : f32
              %32 = arith.subf %31, %in_7 : f32
              %33 = math.powf %cst, %32 : f32
              linalg.yield %33 : f32
            }
            linalg.fill ins(%cst_1 : f32) outs(%22 : memref<32x32xf32>)
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%17 : memref<32x32x32xf32>) outs(%22 : memref<32x32xf32>) {
            ^bb0(%in: f32, %out: f32):
              %30 = arith.addf %in, %out : f32
              linalg.yield %30 : f32
            }
            linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%20, %21 : memref<32x32xf32>, memref<32x32xf32>) outs(%23 : memref<32x32xf32>) {
            ^bb0(%in: f32, %in_7: f32, %out: f32):
              %30 = arith.subf %in, %in_7 : f32
              %31 = math.powf %cst, %30 : f32
              linalg.yield %31 : f32
            }
            linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%19, %23, %22 : memref<32x32xf32>, memref<32x32xf32>, memref<32x32xf32>) outs(%19 : memref<32x32xf32>) {
            ^bb0(%in: f32, %in_7: f32, %in_8: f32, %out: f32):
              %30 = arith.mulf %in, %in_7 : f32
              %31 = arith.addf %30, %in_8 : f32
              linalg.yield %31 : f32
            }
            %29 = arith.muli %arg7, %c4096 : index
            %reinterpret_cast_6 = memref.reinterpret_cast %arg1 to offset: [%29], sizes: [32, 32, 128], strides: [524288, 128, 1] : memref<32x4096x128xf32> to memref<32x32x128xf32, strided<[524288, 128, 1], offset: ?>>
            loom.copy %reinterpret_cast_6, %16 src_mem_space @DRAM dst_mem_space @L1, interconnect : [@horizontal_links, @vertical_links], broadcast : [8, 8] : memref<32x32x128xf32, strided<[524288, 128, 1], offset: ?>>, memref<32x32x128xf32>
            linalg.fill ins(%cst_1 : f32) outs(%26 : memref<32x32x128xf32>)
            linalg.batch_matmul ins(%17, %16 : memref<32x32x32xf32>, memref<32x32x128xf32>) outs(%26 : memref<32x32x128xf32>)
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%26, %25, %23 : memref<32x32x128xf32>, memref<32x32x128xf32>, memref<32x32xf32>) outs(%25 : memref<32x32x128xf32>) {
            ^bb0(%in: f32, %in_7: f32, %in_8: f32, %out: f32):
              %30 = arith.mulf %in_7, %in_8 : f32
              %31 = arith.addf %in, %30 : f32
              linalg.yield %31 : f32
            }
            linalg.copy ins(%21 : memref<32x32xf32>) outs(%20 : memref<32x32xf32>)
          }
          linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%25, %19 : memref<32x32x128xf32>, memref<32x32xf32>) outs(%24 : memref<32x32x128xf32>) {
          ^bb0(%in: f32, %in_5: f32, %out: f32):
            %28 = arith.divf %in, %in_5 : f32
            linalg.yield %28 : f32
          }
          %reinterpret_cast_4 = memref.reinterpret_cast %arg3 to offset: [%27], sizes: [32, 32, 128], strides: [524288, 128, 1] : memref<32x4096x128xf32> to memref<32x32x128xf32, strided<[524288, 128, 1], offset: ?>>
          loom.copy %24, %reinterpret_cast_4 src_mem_space @L1 dst_mem_space @DRAM, interconnect : [], broadcast : [1, 1] : memref<32x32x128xf32>, memref<32x32x128xf32, strided<[524288, 128, 1], offset: ?>>
        }
        scf.reduce 
      } {loom.iter_types = [#loom.iter_type<spatial>, #loom.iter_type<spatial>], loom.mapped_to_dims = [@x, @y]}
      return
    }
  }
  module attributes {loom.pass_name = "Materialize"} {
    func.func @attention__d0i1_d1i1__f01__d_h_h__BB32__BM32__BN32(%arg0: memref<32x128x4096xf32>, %arg1: memref<32x4096x128xf32>, %arg2: memref<32x4096x128xf32>, %arg3: memref<32x4096x128xf32>) {
      %c4096 = arith.constant 4096 : index
      %c128 = arith.constant 128 : index
      %c64 = arith.constant 64 : index
      %c2 = arith.constant 2 : index
      %cst = arith.constant 2.000000e+00 : f32
      %cst_0 = arith.constant 0.12751743074602467 : f64
      %cst_1 = arith.constant 0.000000e+00 : f32
      %cst_2 = arith.constant 1.000000e+00 : f32
      %cst_3 = arith.constant 0xFF800000 : f32
      %c32 = arith.constant 32 : index
      %c0 = arith.constant 0 : index
      %c8 = arith.constant 8 : index
      %c1 = arith.constant 1 : index
      scf.parallel (%arg4, %arg5) = (%c0, %c0) to (%c8, %c8) step (%c1, %c1) {
        scf.for %arg6 = %c0 to %c2 step %c1 {
          %12 = arith.muli %arg4, %c8 overflow<nsw> : index
          %13 = arith.addi %12, %arg5 : index
          %14 = arith.muli %arg6, %c64 overflow<nsw> : index
          %15 = arith.addi %13, %14 : index
          %16 = loom.alloc [32, 32, 128] on @L1 : memref<32x32x128xf32>
          %17 = loom.alloc [32, 32, 32] on @L1 : memref<32x32x32xf32>
          %18 = loom.alloc [32, 128, 32] on @L1 : memref<32x128x32xf32>
          %19 = loom.alloc [32, 32] on @L1 : memref<32x32xf32>
          %20 = loom.alloc [32, 32] on @L1 : memref<32x32xf32>
          %21 = loom.alloc [32, 32] on @L1 : memref<32x32xf32>
          %22 = loom.alloc [32, 32] on @L1 : memref<32x32xf32>
          %23 = loom.alloc [32, 32] on @L1 : memref<32x32xf32>
          %24 = loom.alloc [32, 32, 128] on @L1 : memref<32x32x128xf32>
          %25 = loom.alloc [32, 32, 128] on @L1 : memref<32x32x128xf32>
          %26 = loom.alloc [32, 32, 128] on @L1 : memref<32x32x128xf32>
          %27 = arith.muli %15, %c4096 : index
          %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%27], sizes: [32, 32, 128], strides: [524288, 128, 1] : memref<32x4096x128xf32> to memref<32x32x128xf32, strided<[524288, 128, 1], offset: ?>>
          loom.copy %reinterpret_cast, %24 src_mem_space @DRAM dst_mem_space @L1, interconnect : [], broadcast : [1, 1] : memref<32x32x128xf32, strided<[524288, 128, 1], offset: ?>>, memref<32x32x128xf32>
          linalg.fill ins(%cst_1 : f32) outs(%25 : memref<32x32x128xf32>)
          linalg.fill ins(%cst_2 : f32) outs(%19 : memref<32x32xf32>)
          linalg.fill ins(%cst_3 : f32) outs(%20 : memref<32x32xf32>)
          scf.for %arg7 = %c0 to %c128 step %c1 {
            %28 = arith.muli %arg7, %c32 : index
            %reinterpret_cast_5 = memref.reinterpret_cast %arg0 to offset: [%28], sizes: [32, 128, 32], strides: [524288, 4096, 1] : memref<32x128x4096xf32> to memref<32x128x32xf32, strided<[524288, 4096, 1], offset: ?>>
            loom.copy %reinterpret_cast_5, %18 src_mem_space @DRAM dst_mem_space @L1, interconnect : [@horizontal_links], broadcast : [1, 8] : memref<32x128x32xf32, strided<[524288, 4096, 1], offset: ?>>, memref<32x128x32xf32>
            linalg.fill ins(%cst_1 : f32) outs(%17 : memref<32x32x32xf32>)
            linalg.batch_matmul ins(%24, %18 : memref<32x32x128xf32>, memref<32x128x32xf32>) outs(%17 : memref<32x32x32xf32>)
            linalg.fill ins(%cst_3 : f32) outs(%21 : memref<32x32xf32>)
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%17 : memref<32x32x32xf32>) outs(%21 : memref<32x32xf32>) {
            ^bb0(%in: f32, %out: f32):
              %30 = arith.maximumf %in, %out : f32
              linalg.yield %30 : f32
            }
            linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%20, %21 : memref<32x32xf32>, memref<32x32xf32>) outs(%21 : memref<32x32xf32>) {
            ^bb0(%in: f32, %in_7: f32, %out: f32):
              %30 = arith.truncf %cst_0 : f64 to f32
              %31 = arith.mulf %in_7, %30 : f32
              %32 = arith.cmpf ogt, %in, %31 : f32
              %33 = arith.select %32, %in, %31 : f32
              linalg.yield %33 : f32
            }
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%17, %21 : memref<32x32x32xf32>, memref<32x32xf32>) outs(%17 : memref<32x32x32xf32>) {
            ^bb0(%in: f32, %in_7: f32, %out: f32):
              %30 = arith.truncf %cst_0 : f64 to f32
              %31 = arith.mulf %in, %30 : f32
              %32 = arith.subf %31, %in_7 : f32
              %33 = math.powf %cst, %32 : f32
              linalg.yield %33 : f32
            }
            linalg.fill ins(%cst_1 : f32) outs(%22 : memref<32x32xf32>)
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%17 : memref<32x32x32xf32>) outs(%22 : memref<32x32xf32>) {
            ^bb0(%in: f32, %out: f32):
              %30 = arith.addf %in, %out : f32
              linalg.yield %30 : f32
            }
            linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%20, %21 : memref<32x32xf32>, memref<32x32xf32>) outs(%23 : memref<32x32xf32>) {
            ^bb0(%in: f32, %in_7: f32, %out: f32):
              %30 = arith.subf %in, %in_7 : f32
              %31 = math.powf %cst, %30 : f32
              linalg.yield %31 : f32
            }
            linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%19, %23, %22 : memref<32x32xf32>, memref<32x32xf32>, memref<32x32xf32>) outs(%19 : memref<32x32xf32>) {
            ^bb0(%in: f32, %in_7: f32, %in_8: f32, %out: f32):
              %30 = arith.mulf %in, %in_7 : f32
              %31 = arith.addf %30, %in_8 : f32
              linalg.yield %31 : f32
            }
            %29 = arith.muli %arg7, %c4096 : index
            %reinterpret_cast_6 = memref.reinterpret_cast %arg1 to offset: [%29], sizes: [32, 32, 128], strides: [524288, 128, 1] : memref<32x4096x128xf32> to memref<32x32x128xf32, strided<[524288, 128, 1], offset: ?>>
            loom.copy %reinterpret_cast_6, %16 src_mem_space @DRAM dst_mem_space @L1, interconnect : [@horizontal_links], broadcast : [1, 8] : memref<32x32x128xf32, strided<[524288, 128, 1], offset: ?>>, memref<32x32x128xf32>
            linalg.fill ins(%cst_1 : f32) outs(%26 : memref<32x32x128xf32>)
            linalg.batch_matmul ins(%17, %16 : memref<32x32x32xf32>, memref<32x32x128xf32>) outs(%26 : memref<32x32x128xf32>)
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%26, %25, %23 : memref<32x32x128xf32>, memref<32x32x128xf32>, memref<32x32xf32>) outs(%25 : memref<32x32x128xf32>) {
            ^bb0(%in: f32, %in_7: f32, %in_8: f32, %out: f32):
              %30 = arith.mulf %in_7, %in_8 : f32
              %31 = arith.addf %in, %30 : f32
              linalg.yield %31 : f32
            }
            linalg.copy ins(%21 : memref<32x32xf32>) outs(%20 : memref<32x32xf32>)
          }
          linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%25, %19 : memref<32x32x128xf32>, memref<32x32xf32>) outs(%24 : memref<32x32x128xf32>) {
          ^bb0(%in: f32, %in_5: f32, %out: f32):
            %28 = arith.divf %in, %in_5 : f32
            linalg.yield %28 : f32
          }
          %reinterpret_cast_4 = memref.reinterpret_cast %arg3 to offset: [%27], sizes: [32, 32, 128], strides: [524288, 128, 1] : memref<32x4096x128xf32> to memref<32x32x128xf32, strided<[524288, 128, 1], offset: ?>>
          loom.copy %24, %reinterpret_cast_4 src_mem_space @L1 dst_mem_space @DRAM, interconnect : [], broadcast : [1, 1] : memref<32x32x128xf32>, memref<32x32x128xf32, strided<[524288, 128, 1], offset: ?>>
        }
        scf.reduce 
      } {loom.iter_types = [#loom.iter_type<spatial>, #loom.iter_type<spatial>], loom.mapped_to_dims = [@x, @y]}
      return
    }
  }
  module attributes {loom.pass_name = "Materialize"} {
    func.func @attention__d0i1_d1i1__f01__d_h_v__BB32__BM32__BN32(%arg0: memref<32x128x4096xf32>, %arg1: memref<32x4096x128xf32>, %arg2: memref<32x4096x128xf32>, %arg3: memref<32x4096x128xf32>) {
      %c4096 = arith.constant 4096 : index
      %c128 = arith.constant 128 : index
      %c64 = arith.constant 64 : index
      %c2 = arith.constant 2 : index
      %cst = arith.constant 2.000000e+00 : f32
      %cst_0 = arith.constant 0.12751743074602467 : f64
      %cst_1 = arith.constant 0.000000e+00 : f32
      %cst_2 = arith.constant 1.000000e+00 : f32
      %cst_3 = arith.constant 0xFF800000 : f32
      %c32 = arith.constant 32 : index
      %c0 = arith.constant 0 : index
      %c8 = arith.constant 8 : index
      %c1 = arith.constant 1 : index
      scf.parallel (%arg4, %arg5) = (%c0, %c0) to (%c8, %c8) step (%c1, %c1) {
        scf.for %arg6 = %c0 to %c2 step %c1 {
          %12 = arith.muli %arg4, %c8 overflow<nsw> : index
          %13 = arith.addi %12, %arg5 : index
          %14 = arith.muli %arg6, %c64 overflow<nsw> : index
          %15 = arith.addi %13, %14 : index
          %16 = loom.alloc [32, 32, 128] on @L1 : memref<32x32x128xf32>
          %17 = loom.alloc [32, 32, 32] on @L1 : memref<32x32x32xf32>
          %18 = loom.alloc [32, 128, 32] on @L1 : memref<32x128x32xf32>
          %19 = loom.alloc [32, 32] on @L1 : memref<32x32xf32>
          %20 = loom.alloc [32, 32] on @L1 : memref<32x32xf32>
          %21 = loom.alloc [32, 32] on @L1 : memref<32x32xf32>
          %22 = loom.alloc [32, 32] on @L1 : memref<32x32xf32>
          %23 = loom.alloc [32, 32] on @L1 : memref<32x32xf32>
          %24 = loom.alloc [32, 32, 128] on @L1 : memref<32x32x128xf32>
          %25 = loom.alloc [32, 32, 128] on @L1 : memref<32x32x128xf32>
          %26 = loom.alloc [32, 32, 128] on @L1 : memref<32x32x128xf32>
          %27 = arith.muli %15, %c4096 : index
          %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%27], sizes: [32, 32, 128], strides: [524288, 128, 1] : memref<32x4096x128xf32> to memref<32x32x128xf32, strided<[524288, 128, 1], offset: ?>>
          loom.copy %reinterpret_cast, %24 src_mem_space @DRAM dst_mem_space @L1, interconnect : [], broadcast : [1, 1] : memref<32x32x128xf32, strided<[524288, 128, 1], offset: ?>>, memref<32x32x128xf32>
          linalg.fill ins(%cst_1 : f32) outs(%25 : memref<32x32x128xf32>)
          linalg.fill ins(%cst_2 : f32) outs(%19 : memref<32x32xf32>)
          linalg.fill ins(%cst_3 : f32) outs(%20 : memref<32x32xf32>)
          scf.for %arg7 = %c0 to %c128 step %c1 {
            %28 = arith.muli %arg7, %c32 : index
            %reinterpret_cast_5 = memref.reinterpret_cast %arg0 to offset: [%28], sizes: [32, 128, 32], strides: [524288, 4096, 1] : memref<32x128x4096xf32> to memref<32x128x32xf32, strided<[524288, 4096, 1], offset: ?>>
            loom.copy %reinterpret_cast_5, %18 src_mem_space @DRAM dst_mem_space @L1, interconnect : [@horizontal_links], broadcast : [1, 8] : memref<32x128x32xf32, strided<[524288, 4096, 1], offset: ?>>, memref<32x128x32xf32>
            linalg.fill ins(%cst_1 : f32) outs(%17 : memref<32x32x32xf32>)
            linalg.batch_matmul ins(%24, %18 : memref<32x32x128xf32>, memref<32x128x32xf32>) outs(%17 : memref<32x32x32xf32>)
            linalg.fill ins(%cst_3 : f32) outs(%21 : memref<32x32xf32>)
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%17 : memref<32x32x32xf32>) outs(%21 : memref<32x32xf32>) {
            ^bb0(%in: f32, %out: f32):
              %30 = arith.maximumf %in, %out : f32
              linalg.yield %30 : f32
            }
            linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%20, %21 : memref<32x32xf32>, memref<32x32xf32>) outs(%21 : memref<32x32xf32>) {
            ^bb0(%in: f32, %in_7: f32, %out: f32):
              %30 = arith.truncf %cst_0 : f64 to f32
              %31 = arith.mulf %in_7, %30 : f32
              %32 = arith.cmpf ogt, %in, %31 : f32
              %33 = arith.select %32, %in, %31 : f32
              linalg.yield %33 : f32
            }
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%17, %21 : memref<32x32x32xf32>, memref<32x32xf32>) outs(%17 : memref<32x32x32xf32>) {
            ^bb0(%in: f32, %in_7: f32, %out: f32):
              %30 = arith.truncf %cst_0 : f64 to f32
              %31 = arith.mulf %in, %30 : f32
              %32 = arith.subf %31, %in_7 : f32
              %33 = math.powf %cst, %32 : f32
              linalg.yield %33 : f32
            }
            linalg.fill ins(%cst_1 : f32) outs(%22 : memref<32x32xf32>)
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%17 : memref<32x32x32xf32>) outs(%22 : memref<32x32xf32>) {
            ^bb0(%in: f32, %out: f32):
              %30 = arith.addf %in, %out : f32
              linalg.yield %30 : f32
            }
            linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%20, %21 : memref<32x32xf32>, memref<32x32xf32>) outs(%23 : memref<32x32xf32>) {
            ^bb0(%in: f32, %in_7: f32, %out: f32):
              %30 = arith.subf %in, %in_7 : f32
              %31 = math.powf %cst, %30 : f32
              linalg.yield %31 : f32
            }
            linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%19, %23, %22 : memref<32x32xf32>, memref<32x32xf32>, memref<32x32xf32>) outs(%19 : memref<32x32xf32>) {
            ^bb0(%in: f32, %in_7: f32, %in_8: f32, %out: f32):
              %30 = arith.mulf %in, %in_7 : f32
              %31 = arith.addf %30, %in_8 : f32
              linalg.yield %31 : f32
            }
            %29 = arith.muli %arg7, %c4096 : index
            %reinterpret_cast_6 = memref.reinterpret_cast %arg1 to offset: [%29], sizes: [32, 32, 128], strides: [524288, 128, 1] : memref<32x4096x128xf32> to memref<32x32x128xf32, strided<[524288, 128, 1], offset: ?>>
            loom.copy %reinterpret_cast_6, %16 src_mem_space @DRAM dst_mem_space @L1, interconnect : [@vertical_links], broadcast : [8, 1] : memref<32x32x128xf32, strided<[524288, 128, 1], offset: ?>>, memref<32x32x128xf32>
            linalg.fill ins(%cst_1 : f32) outs(%26 : memref<32x32x128xf32>)
            linalg.batch_matmul ins(%17, %16 : memref<32x32x32xf32>, memref<32x32x128xf32>) outs(%26 : memref<32x32x128xf32>)
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%26, %25, %23 : memref<32x32x128xf32>, memref<32x32x128xf32>, memref<32x32xf32>) outs(%25 : memref<32x32x128xf32>) {
            ^bb0(%in: f32, %in_7: f32, %in_8: f32, %out: f32):
              %30 = arith.mulf %in_7, %in_8 : f32
              %31 = arith.addf %in, %30 : f32
              linalg.yield %31 : f32
            }
            linalg.copy ins(%21 : memref<32x32xf32>) outs(%20 : memref<32x32xf32>)
          }
          linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%25, %19 : memref<32x32x128xf32>, memref<32x32xf32>) outs(%24 : memref<32x32x128xf32>) {
          ^bb0(%in: f32, %in_5: f32, %out: f32):
            %28 = arith.divf %in, %in_5 : f32
            linalg.yield %28 : f32
          }
          %reinterpret_cast_4 = memref.reinterpret_cast %arg3 to offset: [%27], sizes: [32, 32, 128], strides: [524288, 128, 1] : memref<32x4096x128xf32> to memref<32x32x128xf32, strided<[524288, 128, 1], offset: ?>>
          loom.copy %24, %reinterpret_cast_4 src_mem_space @L1 dst_mem_space @DRAM, interconnect : [], broadcast : [1, 1] : memref<32x32x128xf32>, memref<32x32x128xf32, strided<[524288, 128, 1], offset: ?>>
        }
        scf.reduce 
      } {loom.iter_types = [#loom.iter_type<spatial>, #loom.iter_type<spatial>], loom.mapped_to_dims = [@x, @y]}
      return
    }
  }
  module attributes {loom.pass_name = "Materialize"} {
    func.func @attention__d0i1_d1i1__f01__d_v_d__BB32__BM32__BN32(%arg0: memref<32x128x4096xf32>, %arg1: memref<32x4096x128xf32>, %arg2: memref<32x4096x128xf32>, %arg3: memref<32x4096x128xf32>) {
      %c4096 = arith.constant 4096 : index
      %c128 = arith.constant 128 : index
      %c64 = arith.constant 64 : index
      %c2 = arith.constant 2 : index
      %cst = arith.constant 2.000000e+00 : f32
      %cst_0 = arith.constant 0.12751743074602467 : f64
      %cst_1 = arith.constant 0.000000e+00 : f32
      %cst_2 = arith.constant 1.000000e+00 : f32
      %cst_3 = arith.constant 0xFF800000 : f32
      %c32 = arith.constant 32 : index
      %c0 = arith.constant 0 : index
      %c8 = arith.constant 8 : index
      %c1 = arith.constant 1 : index
      scf.parallel (%arg4, %arg5) = (%c0, %c0) to (%c8, %c8) step (%c1, %c1) {
        scf.for %arg6 = %c0 to %c2 step %c1 {
          %12 = arith.muli %arg4, %c8 overflow<nsw> : index
          %13 = arith.addi %12, %arg5 : index
          %14 = arith.muli %arg6, %c64 overflow<nsw> : index
          %15 = arith.addi %13, %14 : index
          %16 = loom.alloc [32, 32, 128] on @L1 : memref<32x32x128xf32>
          %17 = loom.alloc [32, 32, 32] on @L1 : memref<32x32x32xf32>
          %18 = loom.alloc [32, 128, 32] on @L1 : memref<32x128x32xf32>
          %19 = loom.alloc [32, 32] on @L1 : memref<32x32xf32>
          %20 = loom.alloc [32, 32] on @L1 : memref<32x32xf32>
          %21 = loom.alloc [32, 32] on @L1 : memref<32x32xf32>
          %22 = loom.alloc [32, 32] on @L1 : memref<32x32xf32>
          %23 = loom.alloc [32, 32] on @L1 : memref<32x32xf32>
          %24 = loom.alloc [32, 32, 128] on @L1 : memref<32x32x128xf32>
          %25 = loom.alloc [32, 32, 128] on @L1 : memref<32x32x128xf32>
          %26 = loom.alloc [32, 32, 128] on @L1 : memref<32x32x128xf32>
          %27 = arith.muli %15, %c4096 : index
          %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%27], sizes: [32, 32, 128], strides: [524288, 128, 1] : memref<32x4096x128xf32> to memref<32x32x128xf32, strided<[524288, 128, 1], offset: ?>>
          loom.copy %reinterpret_cast, %24 src_mem_space @DRAM dst_mem_space @L1, interconnect : [], broadcast : [1, 1] : memref<32x32x128xf32, strided<[524288, 128, 1], offset: ?>>, memref<32x32x128xf32>
          linalg.fill ins(%cst_1 : f32) outs(%25 : memref<32x32x128xf32>)
          linalg.fill ins(%cst_2 : f32) outs(%19 : memref<32x32xf32>)
          linalg.fill ins(%cst_3 : f32) outs(%20 : memref<32x32xf32>)
          scf.for %arg7 = %c0 to %c128 step %c1 {
            %28 = arith.muli %arg7, %c32 : index
            %reinterpret_cast_5 = memref.reinterpret_cast %arg0 to offset: [%28], sizes: [32, 128, 32], strides: [524288, 4096, 1] : memref<32x128x4096xf32> to memref<32x128x32xf32, strided<[524288, 4096, 1], offset: ?>>
            loom.copy %reinterpret_cast_5, %18 src_mem_space @DRAM dst_mem_space @L1, interconnect : [@vertical_links], broadcast : [8, 1] : memref<32x128x32xf32, strided<[524288, 4096, 1], offset: ?>>, memref<32x128x32xf32>
            linalg.fill ins(%cst_1 : f32) outs(%17 : memref<32x32x32xf32>)
            linalg.batch_matmul ins(%24, %18 : memref<32x32x128xf32>, memref<32x128x32xf32>) outs(%17 : memref<32x32x32xf32>)
            linalg.fill ins(%cst_3 : f32) outs(%21 : memref<32x32xf32>)
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%17 : memref<32x32x32xf32>) outs(%21 : memref<32x32xf32>) {
            ^bb0(%in: f32, %out: f32):
              %30 = arith.maximumf %in, %out : f32
              linalg.yield %30 : f32
            }
            linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%20, %21 : memref<32x32xf32>, memref<32x32xf32>) outs(%21 : memref<32x32xf32>) {
            ^bb0(%in: f32, %in_7: f32, %out: f32):
              %30 = arith.truncf %cst_0 : f64 to f32
              %31 = arith.mulf %in_7, %30 : f32
              %32 = arith.cmpf ogt, %in, %31 : f32
              %33 = arith.select %32, %in, %31 : f32
              linalg.yield %33 : f32
            }
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%17, %21 : memref<32x32x32xf32>, memref<32x32xf32>) outs(%17 : memref<32x32x32xf32>) {
            ^bb0(%in: f32, %in_7: f32, %out: f32):
              %30 = arith.truncf %cst_0 : f64 to f32
              %31 = arith.mulf %in, %30 : f32
              %32 = arith.subf %31, %in_7 : f32
              %33 = math.powf %cst, %32 : f32
              linalg.yield %33 : f32
            }
            linalg.fill ins(%cst_1 : f32) outs(%22 : memref<32x32xf32>)
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%17 : memref<32x32x32xf32>) outs(%22 : memref<32x32xf32>) {
            ^bb0(%in: f32, %out: f32):
              %30 = arith.addf %in, %out : f32
              linalg.yield %30 : f32
            }
            linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%20, %21 : memref<32x32xf32>, memref<32x32xf32>) outs(%23 : memref<32x32xf32>) {
            ^bb0(%in: f32, %in_7: f32, %out: f32):
              %30 = arith.subf %in, %in_7 : f32
              %31 = math.powf %cst, %30 : f32
              linalg.yield %31 : f32
            }
            linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%19, %23, %22 : memref<32x32xf32>, memref<32x32xf32>, memref<32x32xf32>) outs(%19 : memref<32x32xf32>) {
            ^bb0(%in: f32, %in_7: f32, %in_8: f32, %out: f32):
              %30 = arith.mulf %in, %in_7 : f32
              %31 = arith.addf %30, %in_8 : f32
              linalg.yield %31 : f32
            }
            %29 = arith.muli %arg7, %c4096 : index
            %reinterpret_cast_6 = memref.reinterpret_cast %arg1 to offset: [%29], sizes: [32, 32, 128], strides: [524288, 128, 1] : memref<32x4096x128xf32> to memref<32x32x128xf32, strided<[524288, 128, 1], offset: ?>>
            loom.copy %reinterpret_cast_6, %16 src_mem_space @DRAM dst_mem_space @L1, interconnect : [], broadcast : [1, 1] : memref<32x32x128xf32, strided<[524288, 128, 1], offset: ?>>, memref<32x32x128xf32>
            linalg.fill ins(%cst_1 : f32) outs(%26 : memref<32x32x128xf32>)
            linalg.batch_matmul ins(%17, %16 : memref<32x32x32xf32>, memref<32x32x128xf32>) outs(%26 : memref<32x32x128xf32>)
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%26, %25, %23 : memref<32x32x128xf32>, memref<32x32x128xf32>, memref<32x32xf32>) outs(%25 : memref<32x32x128xf32>) {
            ^bb0(%in: f32, %in_7: f32, %in_8: f32, %out: f32):
              %30 = arith.mulf %in_7, %in_8 : f32
              %31 = arith.addf %in, %30 : f32
              linalg.yield %31 : f32
            }
            linalg.copy ins(%21 : memref<32x32xf32>) outs(%20 : memref<32x32xf32>)
          }
          linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%25, %19 : memref<32x32x128xf32>, memref<32x32xf32>) outs(%24 : memref<32x32x128xf32>) {
          ^bb0(%in: f32, %in_5: f32, %out: f32):
            %28 = arith.divf %in, %in_5 : f32
            linalg.yield %28 : f32
          }
          %reinterpret_cast_4 = memref.reinterpret_cast %arg3 to offset: [%27], sizes: [32, 32, 128], strides: [524288, 128, 1] : memref<32x4096x128xf32> to memref<32x32x128xf32, strided<[524288, 128, 1], offset: ?>>
          loom.copy %24, %reinterpret_cast_4 src_mem_space @L1 dst_mem_space @DRAM, interconnect : [], broadcast : [1, 1] : memref<32x32x128xf32>, memref<32x32x128xf32, strided<[524288, 128, 1], offset: ?>>
        }
        scf.reduce 
      } {loom.iter_types = [#loom.iter_type<spatial>, #loom.iter_type<spatial>], loom.mapped_to_dims = [@x, @y]}
      return
    }
  }
  module attributes {loom.pass_name = "Materialize"} {
    func.func @attention__d0i1_d1i1__f01__d_v_a__BB32__BM32__BN32(%arg0: memref<32x128x4096xf32>, %arg1: memref<32x4096x128xf32>, %arg2: memref<32x4096x128xf32>, %arg3: memref<32x4096x128xf32>) {
      %c4096 = arith.constant 4096 : index
      %c128 = arith.constant 128 : index
      %c64 = arith.constant 64 : index
      %c2 = arith.constant 2 : index
      %cst = arith.constant 2.000000e+00 : f32
      %cst_0 = arith.constant 0.12751743074602467 : f64
      %cst_1 = arith.constant 0.000000e+00 : f32
      %cst_2 = arith.constant 1.000000e+00 : f32
      %cst_3 = arith.constant 0xFF800000 : f32
      %c32 = arith.constant 32 : index
      %c0 = arith.constant 0 : index
      %c8 = arith.constant 8 : index
      %c1 = arith.constant 1 : index
      scf.parallel (%arg4, %arg5) = (%c0, %c0) to (%c8, %c8) step (%c1, %c1) {
        scf.for %arg6 = %c0 to %c2 step %c1 {
          %12 = arith.muli %arg4, %c8 overflow<nsw> : index
          %13 = arith.addi %12, %arg5 : index
          %14 = arith.muli %arg6, %c64 overflow<nsw> : index
          %15 = arith.addi %13, %14 : index
          %16 = loom.alloc [32, 32, 128] on @L1 : memref<32x32x128xf32>
          %17 = loom.alloc [32, 32, 32] on @L1 : memref<32x32x32xf32>
          %18 = loom.alloc [32, 128, 32] on @L1 : memref<32x128x32xf32>
          %19 = loom.alloc [32, 32] on @L1 : memref<32x32xf32>
          %20 = loom.alloc [32, 32] on @L1 : memref<32x32xf32>
          %21 = loom.alloc [32, 32] on @L1 : memref<32x32xf32>
          %22 = loom.alloc [32, 32] on @L1 : memref<32x32xf32>
          %23 = loom.alloc [32, 32] on @L1 : memref<32x32xf32>
          %24 = loom.alloc [32, 32, 128] on @L1 : memref<32x32x128xf32>
          %25 = loom.alloc [32, 32, 128] on @L1 : memref<32x32x128xf32>
          %26 = loom.alloc [32, 32, 128] on @L1 : memref<32x32x128xf32>
          %27 = arith.muli %15, %c4096 : index
          %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%27], sizes: [32, 32, 128], strides: [524288, 128, 1] : memref<32x4096x128xf32> to memref<32x32x128xf32, strided<[524288, 128, 1], offset: ?>>
          loom.copy %reinterpret_cast, %24 src_mem_space @DRAM dst_mem_space @L1, interconnect : [], broadcast : [1, 1] : memref<32x32x128xf32, strided<[524288, 128, 1], offset: ?>>, memref<32x32x128xf32>
          linalg.fill ins(%cst_1 : f32) outs(%25 : memref<32x32x128xf32>)
          linalg.fill ins(%cst_2 : f32) outs(%19 : memref<32x32xf32>)
          linalg.fill ins(%cst_3 : f32) outs(%20 : memref<32x32xf32>)
          scf.for %arg7 = %c0 to %c128 step %c1 {
            %28 = arith.muli %arg7, %c32 : index
            %reinterpret_cast_5 = memref.reinterpret_cast %arg0 to offset: [%28], sizes: [32, 128, 32], strides: [524288, 4096, 1] : memref<32x128x4096xf32> to memref<32x128x32xf32, strided<[524288, 4096, 1], offset: ?>>
            loom.copy %reinterpret_cast_5, %18 src_mem_space @DRAM dst_mem_space @L1, interconnect : [@vertical_links], broadcast : [8, 1] : memref<32x128x32xf32, strided<[524288, 4096, 1], offset: ?>>, memref<32x128x32xf32>
            linalg.fill ins(%cst_1 : f32) outs(%17 : memref<32x32x32xf32>)
            linalg.batch_matmul ins(%24, %18 : memref<32x32x128xf32>, memref<32x128x32xf32>) outs(%17 : memref<32x32x32xf32>)
            linalg.fill ins(%cst_3 : f32) outs(%21 : memref<32x32xf32>)
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%17 : memref<32x32x32xf32>) outs(%21 : memref<32x32xf32>) {
            ^bb0(%in: f32, %out: f32):
              %30 = arith.maximumf %in, %out : f32
              linalg.yield %30 : f32
            }
            linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%20, %21 : memref<32x32xf32>, memref<32x32xf32>) outs(%21 : memref<32x32xf32>) {
            ^bb0(%in: f32, %in_7: f32, %out: f32):
              %30 = arith.truncf %cst_0 : f64 to f32
              %31 = arith.mulf %in_7, %30 : f32
              %32 = arith.cmpf ogt, %in, %31 : f32
              %33 = arith.select %32, %in, %31 : f32
              linalg.yield %33 : f32
            }
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%17, %21 : memref<32x32x32xf32>, memref<32x32xf32>) outs(%17 : memref<32x32x32xf32>) {
            ^bb0(%in: f32, %in_7: f32, %out: f32):
              %30 = arith.truncf %cst_0 : f64 to f32
              %31 = arith.mulf %in, %30 : f32
              %32 = arith.subf %31, %in_7 : f32
              %33 = math.powf %cst, %32 : f32
              linalg.yield %33 : f32
            }
            linalg.fill ins(%cst_1 : f32) outs(%22 : memref<32x32xf32>)
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%17 : memref<32x32x32xf32>) outs(%22 : memref<32x32xf32>) {
            ^bb0(%in: f32, %out: f32):
              %30 = arith.addf %in, %out : f32
              linalg.yield %30 : f32
            }
            linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%20, %21 : memref<32x32xf32>, memref<32x32xf32>) outs(%23 : memref<32x32xf32>) {
            ^bb0(%in: f32, %in_7: f32, %out: f32):
              %30 = arith.subf %in, %in_7 : f32
              %31 = math.powf %cst, %30 : f32
              linalg.yield %31 : f32
            }
            linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%19, %23, %22 : memref<32x32xf32>, memref<32x32xf32>, memref<32x32xf32>) outs(%19 : memref<32x32xf32>) {
            ^bb0(%in: f32, %in_7: f32, %in_8: f32, %out: f32):
              %30 = arith.mulf %in, %in_7 : f32
              %31 = arith.addf %30, %in_8 : f32
              linalg.yield %31 : f32
            }
            %29 = arith.muli %arg7, %c4096 : index
            %reinterpret_cast_6 = memref.reinterpret_cast %arg1 to offset: [%29], sizes: [32, 32, 128], strides: [524288, 128, 1] : memref<32x4096x128xf32> to memref<32x32x128xf32, strided<[524288, 128, 1], offset: ?>>
            loom.copy %reinterpret_cast_6, %16 src_mem_space @DRAM dst_mem_space @L1, interconnect : [@horizontal_links, @vertical_links], broadcast : [8, 8] : memref<32x32x128xf32, strided<[524288, 128, 1], offset: ?>>, memref<32x32x128xf32>
            linalg.fill ins(%cst_1 : f32) outs(%26 : memref<32x32x128xf32>)
            linalg.batch_matmul ins(%17, %16 : memref<32x32x32xf32>, memref<32x32x128xf32>) outs(%26 : memref<32x32x128xf32>)
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%26, %25, %23 : memref<32x32x128xf32>, memref<32x32x128xf32>, memref<32x32xf32>) outs(%25 : memref<32x32x128xf32>) {
            ^bb0(%in: f32, %in_7: f32, %in_8: f32, %out: f32):
              %30 = arith.mulf %in_7, %in_8 : f32
              %31 = arith.addf %in, %30 : f32
              linalg.yield %31 : f32
            }
            linalg.copy ins(%21 : memref<32x32xf32>) outs(%20 : memref<32x32xf32>)
          }
          linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%25, %19 : memref<32x32x128xf32>, memref<32x32xf32>) outs(%24 : memref<32x32x128xf32>) {
          ^bb0(%in: f32, %in_5: f32, %out: f32):
            %28 = arith.divf %in, %in_5 : f32
            linalg.yield %28 : f32
          }
          %reinterpret_cast_4 = memref.reinterpret_cast %arg3 to offset: [%27], sizes: [32, 32, 128], strides: [524288, 128, 1] : memref<32x4096x128xf32> to memref<32x32x128xf32, strided<[524288, 128, 1], offset: ?>>
          loom.copy %24, %reinterpret_cast_4 src_mem_space @L1 dst_mem_space @DRAM, interconnect : [], broadcast : [1, 1] : memref<32x32x128xf32>, memref<32x32x128xf32, strided<[524288, 128, 1], offset: ?>>
        }
        scf.reduce 
      } {loom.iter_types = [#loom.iter_type<spatial>, #loom.iter_type<spatial>], loom.mapped_to_dims = [@x, @y]}
      return
    }
  }
  module attributes {loom.pass_name = "Materialize"} {
    func.func @attention__d0i1_d1i1__f01__d_v_h__BB32__BM32__BN32(%arg0: memref<32x128x4096xf32>, %arg1: memref<32x4096x128xf32>, %arg2: memref<32x4096x128xf32>, %arg3: memref<32x4096x128xf32>) {
      %c4096 = arith.constant 4096 : index
      %c128 = arith.constant 128 : index
      %c64 = arith.constant 64 : index
      %c2 = arith.constant 2 : index
      %cst = arith.constant 2.000000e+00 : f32
      %cst_0 = arith.constant 0.12751743074602467 : f64
      %cst_1 = arith.constant 0.000000e+00 : f32
      %cst_2 = arith.constant 1.000000e+00 : f32
      %cst_3 = arith.constant 0xFF800000 : f32
      %c32 = arith.constant 32 : index
      %c0 = arith.constant 0 : index
      %c8 = arith.constant 8 : index
      %c1 = arith.constant 1 : index
      scf.parallel (%arg4, %arg5) = (%c0, %c0) to (%c8, %c8) step (%c1, %c1) {
        scf.for %arg6 = %c0 to %c2 step %c1 {
          %12 = arith.muli %arg4, %c8 overflow<nsw> : index
          %13 = arith.addi %12, %arg5 : index
          %14 = arith.muli %arg6, %c64 overflow<nsw> : index
          %15 = arith.addi %13, %14 : index
          %16 = loom.alloc [32, 32, 128] on @L1 : memref<32x32x128xf32>
          %17 = loom.alloc [32, 32, 32] on @L1 : memref<32x32x32xf32>
          %18 = loom.alloc [32, 128, 32] on @L1 : memref<32x128x32xf32>
          %19 = loom.alloc [32, 32] on @L1 : memref<32x32xf32>
          %20 = loom.alloc [32, 32] on @L1 : memref<32x32xf32>
          %21 = loom.alloc [32, 32] on @L1 : memref<32x32xf32>
          %22 = loom.alloc [32, 32] on @L1 : memref<32x32xf32>
          %23 = loom.alloc [32, 32] on @L1 : memref<32x32xf32>
          %24 = loom.alloc [32, 32, 128] on @L1 : memref<32x32x128xf32>
          %25 = loom.alloc [32, 32, 128] on @L1 : memref<32x32x128xf32>
          %26 = loom.alloc [32, 32, 128] on @L1 : memref<32x32x128xf32>
          %27 = arith.muli %15, %c4096 : index
          %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%27], sizes: [32, 32, 128], strides: [524288, 128, 1] : memref<32x4096x128xf32> to memref<32x32x128xf32, strided<[524288, 128, 1], offset: ?>>
          loom.copy %reinterpret_cast, %24 src_mem_space @DRAM dst_mem_space @L1, interconnect : [], broadcast : [1, 1] : memref<32x32x128xf32, strided<[524288, 128, 1], offset: ?>>, memref<32x32x128xf32>
          linalg.fill ins(%cst_1 : f32) outs(%25 : memref<32x32x128xf32>)
          linalg.fill ins(%cst_2 : f32) outs(%19 : memref<32x32xf32>)
          linalg.fill ins(%cst_3 : f32) outs(%20 : memref<32x32xf32>)
          scf.for %arg7 = %c0 to %c128 step %c1 {
            %28 = arith.muli %arg7, %c32 : index
            %reinterpret_cast_5 = memref.reinterpret_cast %arg0 to offset: [%28], sizes: [32, 128, 32], strides: [524288, 4096, 1] : memref<32x128x4096xf32> to memref<32x128x32xf32, strided<[524288, 4096, 1], offset: ?>>
            loom.copy %reinterpret_cast_5, %18 src_mem_space @DRAM dst_mem_space @L1, interconnect : [@vertical_links], broadcast : [8, 1] : memref<32x128x32xf32, strided<[524288, 4096, 1], offset: ?>>, memref<32x128x32xf32>
            linalg.fill ins(%cst_1 : f32) outs(%17 : memref<32x32x32xf32>)
            linalg.batch_matmul ins(%24, %18 : memref<32x32x128xf32>, memref<32x128x32xf32>) outs(%17 : memref<32x32x32xf32>)
            linalg.fill ins(%cst_3 : f32) outs(%21 : memref<32x32xf32>)
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%17 : memref<32x32x32xf32>) outs(%21 : memref<32x32xf32>) {
            ^bb0(%in: f32, %out: f32):
              %30 = arith.maximumf %in, %out : f32
              linalg.yield %30 : f32
            }
            linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%20, %21 : memref<32x32xf32>, memref<32x32xf32>) outs(%21 : memref<32x32xf32>) {
            ^bb0(%in: f32, %in_7: f32, %out: f32):
              %30 = arith.truncf %cst_0 : f64 to f32
              %31 = arith.mulf %in_7, %30 : f32
              %32 = arith.cmpf ogt, %in, %31 : f32
              %33 = arith.select %32, %in, %31 : f32
              linalg.yield %33 : f32
            }
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%17, %21 : memref<32x32x32xf32>, memref<32x32xf32>) outs(%17 : memref<32x32x32xf32>) {
            ^bb0(%in: f32, %in_7: f32, %out: f32):
              %30 = arith.truncf %cst_0 : f64 to f32
              %31 = arith.mulf %in, %30 : f32
              %32 = arith.subf %31, %in_7 : f32
              %33 = math.powf %cst, %32 : f32
              linalg.yield %33 : f32
            }
            linalg.fill ins(%cst_1 : f32) outs(%22 : memref<32x32xf32>)
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%17 : memref<32x32x32xf32>) outs(%22 : memref<32x32xf32>) {
            ^bb0(%in: f32, %out: f32):
              %30 = arith.addf %in, %out : f32
              linalg.yield %30 : f32
            }
            linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%20, %21 : memref<32x32xf32>, memref<32x32xf32>) outs(%23 : memref<32x32xf32>) {
            ^bb0(%in: f32, %in_7: f32, %out: f32):
              %30 = arith.subf %in, %in_7 : f32
              %31 = math.powf %cst, %30 : f32
              linalg.yield %31 : f32
            }
            linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%19, %23, %22 : memref<32x32xf32>, memref<32x32xf32>, memref<32x32xf32>) outs(%19 : memref<32x32xf32>) {
            ^bb0(%in: f32, %in_7: f32, %in_8: f32, %out: f32):
              %30 = arith.mulf %in, %in_7 : f32
              %31 = arith.addf %30, %in_8 : f32
              linalg.yield %31 : f32
            }
            %29 = arith.muli %arg7, %c4096 : index
            %reinterpret_cast_6 = memref.reinterpret_cast %arg1 to offset: [%29], sizes: [32, 32, 128], strides: [524288, 128, 1] : memref<32x4096x128xf32> to memref<32x32x128xf32, strided<[524288, 128, 1], offset: ?>>
            loom.copy %reinterpret_cast_6, %16 src_mem_space @DRAM dst_mem_space @L1, interconnect : [@horizontal_links], broadcast : [1, 8] : memref<32x32x128xf32, strided<[524288, 128, 1], offset: ?>>, memref<32x32x128xf32>
            linalg.fill ins(%cst_1 : f32) outs(%26 : memref<32x32x128xf32>)
            linalg.batch_matmul ins(%17, %16 : memref<32x32x32xf32>, memref<32x32x128xf32>) outs(%26 : memref<32x32x128xf32>)
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%26, %25, %23 : memref<32x32x128xf32>, memref<32x32x128xf32>, memref<32x32xf32>) outs(%25 : memref<32x32x128xf32>) {
            ^bb0(%in: f32, %in_7: f32, %in_8: f32, %out: f32):
              %30 = arith.mulf %in_7, %in_8 : f32
              %31 = arith.addf %in, %30 : f32
              linalg.yield %31 : f32
            }
            linalg.copy ins(%21 : memref<32x32xf32>) outs(%20 : memref<32x32xf32>)
          }
          linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%25, %19 : memref<32x32x128xf32>, memref<32x32xf32>) outs(%24 : memref<32x32x128xf32>) {
          ^bb0(%in: f32, %in_5: f32, %out: f32):
            %28 = arith.divf %in, %in_5 : f32
            linalg.yield %28 : f32
          }
          %reinterpret_cast_4 = memref.reinterpret_cast %arg3 to offset: [%27], sizes: [32, 32, 128], strides: [524288, 128, 1] : memref<32x4096x128xf32> to memref<32x32x128xf32, strided<[524288, 128, 1], offset: ?>>
          loom.copy %24, %reinterpret_cast_4 src_mem_space @L1 dst_mem_space @DRAM, interconnect : [], broadcast : [1, 1] : memref<32x32x128xf32>, memref<32x32x128xf32, strided<[524288, 128, 1], offset: ?>>
        }
        scf.reduce 
      } {loom.iter_types = [#loom.iter_type<spatial>, #loom.iter_type<spatial>], loom.mapped_to_dims = [@x, @y]}
      return
    }
  }
  module attributes {loom.pass_name = "Materialize"} {
    func.func @attention__d0i1_d1i1__f01__d_v_v__BB32__BM32__BN32(%arg0: memref<32x128x4096xf32>, %arg1: memref<32x4096x128xf32>, %arg2: memref<32x4096x128xf32>, %arg3: memref<32x4096x128xf32>) {
      %c4096 = arith.constant 4096 : index
      %c128 = arith.constant 128 : index
      %c64 = arith.constant 64 : index
      %c2 = arith.constant 2 : index
      %cst = arith.constant 2.000000e+00 : f32
      %cst_0 = arith.constant 0.12751743074602467 : f64
      %cst_1 = arith.constant 0.000000e+00 : f32
      %cst_2 = arith.constant 1.000000e+00 : f32
      %cst_3 = arith.constant 0xFF800000 : f32
      %c32 = arith.constant 32 : index
      %c0 = arith.constant 0 : index
      %c8 = arith.constant 8 : index
      %c1 = arith.constant 1 : index
      scf.parallel (%arg4, %arg5) = (%c0, %c0) to (%c8, %c8) step (%c1, %c1) {
        scf.for %arg6 = %c0 to %c2 step %c1 {
          %12 = arith.muli %arg4, %c8 overflow<nsw> : index
          %13 = arith.addi %12, %arg5 : index
          %14 = arith.muli %arg6, %c64 overflow<nsw> : index
          %15 = arith.addi %13, %14 : index
          %16 = loom.alloc [32, 32, 128] on @L1 : memref<32x32x128xf32>
          %17 = loom.alloc [32, 32, 32] on @L1 : memref<32x32x32xf32>
          %18 = loom.alloc [32, 128, 32] on @L1 : memref<32x128x32xf32>
          %19 = loom.alloc [32, 32] on @L1 : memref<32x32xf32>
          %20 = loom.alloc [32, 32] on @L1 : memref<32x32xf32>
          %21 = loom.alloc [32, 32] on @L1 : memref<32x32xf32>
          %22 = loom.alloc [32, 32] on @L1 : memref<32x32xf32>
          %23 = loom.alloc [32, 32] on @L1 : memref<32x32xf32>
          %24 = loom.alloc [32, 32, 128] on @L1 : memref<32x32x128xf32>
          %25 = loom.alloc [32, 32, 128] on @L1 : memref<32x32x128xf32>
          %26 = loom.alloc [32, 32, 128] on @L1 : memref<32x32x128xf32>
          %27 = arith.muli %15, %c4096 : index
          %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%27], sizes: [32, 32, 128], strides: [524288, 128, 1] : memref<32x4096x128xf32> to memref<32x32x128xf32, strided<[524288, 128, 1], offset: ?>>
          loom.copy %reinterpret_cast, %24 src_mem_space @DRAM dst_mem_space @L1, interconnect : [], broadcast : [1, 1] : memref<32x32x128xf32, strided<[524288, 128, 1], offset: ?>>, memref<32x32x128xf32>
          linalg.fill ins(%cst_1 : f32) outs(%25 : memref<32x32x128xf32>)
          linalg.fill ins(%cst_2 : f32) outs(%19 : memref<32x32xf32>)
          linalg.fill ins(%cst_3 : f32) outs(%20 : memref<32x32xf32>)
          scf.for %arg7 = %c0 to %c128 step %c1 {
            %28 = arith.muli %arg7, %c32 : index
            %reinterpret_cast_5 = memref.reinterpret_cast %arg0 to offset: [%28], sizes: [32, 128, 32], strides: [524288, 4096, 1] : memref<32x128x4096xf32> to memref<32x128x32xf32, strided<[524288, 4096, 1], offset: ?>>
            loom.copy %reinterpret_cast_5, %18 src_mem_space @DRAM dst_mem_space @L1, interconnect : [@vertical_links], broadcast : [8, 1] : memref<32x128x32xf32, strided<[524288, 4096, 1], offset: ?>>, memref<32x128x32xf32>
            linalg.fill ins(%cst_1 : f32) outs(%17 : memref<32x32x32xf32>)
            linalg.batch_matmul ins(%24, %18 : memref<32x32x128xf32>, memref<32x128x32xf32>) outs(%17 : memref<32x32x32xf32>)
            linalg.fill ins(%cst_3 : f32) outs(%21 : memref<32x32xf32>)
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%17 : memref<32x32x32xf32>) outs(%21 : memref<32x32xf32>) {
            ^bb0(%in: f32, %out: f32):
              %30 = arith.maximumf %in, %out : f32
              linalg.yield %30 : f32
            }
            linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%20, %21 : memref<32x32xf32>, memref<32x32xf32>) outs(%21 : memref<32x32xf32>) {
            ^bb0(%in: f32, %in_7: f32, %out: f32):
              %30 = arith.truncf %cst_0 : f64 to f32
              %31 = arith.mulf %in_7, %30 : f32
              %32 = arith.cmpf ogt, %in, %31 : f32
              %33 = arith.select %32, %in, %31 : f32
              linalg.yield %33 : f32
            }
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%17, %21 : memref<32x32x32xf32>, memref<32x32xf32>) outs(%17 : memref<32x32x32xf32>) {
            ^bb0(%in: f32, %in_7: f32, %out: f32):
              %30 = arith.truncf %cst_0 : f64 to f32
              %31 = arith.mulf %in, %30 : f32
              %32 = arith.subf %31, %in_7 : f32
              %33 = math.powf %cst, %32 : f32
              linalg.yield %33 : f32
            }
            linalg.fill ins(%cst_1 : f32) outs(%22 : memref<32x32xf32>)
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%17 : memref<32x32x32xf32>) outs(%22 : memref<32x32xf32>) {
            ^bb0(%in: f32, %out: f32):
              %30 = arith.addf %in, %out : f32
              linalg.yield %30 : f32
            }
            linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%20, %21 : memref<32x32xf32>, memref<32x32xf32>) outs(%23 : memref<32x32xf32>) {
            ^bb0(%in: f32, %in_7: f32, %out: f32):
              %30 = arith.subf %in, %in_7 : f32
              %31 = math.powf %cst, %30 : f32
              linalg.yield %31 : f32
            }
            linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%19, %23, %22 : memref<32x32xf32>, memref<32x32xf32>, memref<32x32xf32>) outs(%19 : memref<32x32xf32>) {
            ^bb0(%in: f32, %in_7: f32, %in_8: f32, %out: f32):
              %30 = arith.mulf %in, %in_7 : f32
              %31 = arith.addf %30, %in_8 : f32
              linalg.yield %31 : f32
            }
            %29 = arith.muli %arg7, %c4096 : index
            %reinterpret_cast_6 = memref.reinterpret_cast %arg1 to offset: [%29], sizes: [32, 32, 128], strides: [524288, 128, 1] : memref<32x4096x128xf32> to memref<32x32x128xf32, strided<[524288, 128, 1], offset: ?>>
            loom.copy %reinterpret_cast_6, %16 src_mem_space @DRAM dst_mem_space @L1, interconnect : [@vertical_links], broadcast : [8, 1] : memref<32x32x128xf32, strided<[524288, 128, 1], offset: ?>>, memref<32x32x128xf32>
            linalg.fill ins(%cst_1 : f32) outs(%26 : memref<32x32x128xf32>)
            linalg.batch_matmul ins(%17, %16 : memref<32x32x32xf32>, memref<32x32x128xf32>) outs(%26 : memref<32x32x128xf32>)
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%26, %25, %23 : memref<32x32x128xf32>, memref<32x32x128xf32>, memref<32x32xf32>) outs(%25 : memref<32x32x128xf32>) {
            ^bb0(%in: f32, %in_7: f32, %in_8: f32, %out: f32):
              %30 = arith.mulf %in_7, %in_8 : f32
              %31 = arith.addf %in, %30 : f32
              linalg.yield %31 : f32
            }
            linalg.copy ins(%21 : memref<32x32xf32>) outs(%20 : memref<32x32xf32>)
          }
          linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%25, %19 : memref<32x32x128xf32>, memref<32x32xf32>) outs(%24 : memref<32x32x128xf32>) {
          ^bb0(%in: f32, %in_5: f32, %out: f32):
            %28 = arith.divf %in, %in_5 : f32
            linalg.yield %28 : f32
          }
          %reinterpret_cast_4 = memref.reinterpret_cast %arg3 to offset: [%27], sizes: [32, 32, 128], strides: [524288, 128, 1] : memref<32x4096x128xf32> to memref<32x32x128xf32, strided<[524288, 128, 1], offset: ?>>
          loom.copy %24, %reinterpret_cast_4 src_mem_space @L1 dst_mem_space @DRAM, interconnect : [], broadcast : [1, 1] : memref<32x32x128xf32>, memref<32x32x128xf32, strided<[524288, 128, 1], offset: ?>>
        }
        scf.reduce 
      } {loom.iter_types = [#loom.iter_type<spatial>, #loom.iter_type<spatial>], loom.mapped_to_dims = [@x, @y]}
      return
    }
  }
  module attributes {loom.pass_name = "Materialize"} {
    func.func @attention__d0i1_d1i1__f10__d_d_d__BB32__BM32__BN32(%arg0: memref<32x128x4096xf32>, %arg1: memref<32x4096x128xf32>, %arg2: memref<32x4096x128xf32>, %arg3: memref<32x4096x128xf32>) {
      %c4096 = arith.constant 4096 : index
      %c128 = arith.constant 128 : index
      %c64 = arith.constant 64 : index
      %c2 = arith.constant 2 : index
      %cst = arith.constant 2.000000e+00 : f32
      %cst_0 = arith.constant 0.12751743074602467 : f64
      %cst_1 = arith.constant 0.000000e+00 : f32
      %cst_2 = arith.constant 1.000000e+00 : f32
      %cst_3 = arith.constant 0xFF800000 : f32
      %c32 = arith.constant 32 : index
      %c0 = arith.constant 0 : index
      %c8 = arith.constant 8 : index
      %c1 = arith.constant 1 : index
      scf.parallel (%arg4, %arg5) = (%c0, %c0) to (%c8, %c8) step (%c1, %c1) {
        scf.for %arg6 = %c0 to %c2 step %c1 {
          %12 = arith.muli %arg4, %c8 overflow<nsw> : index
          %13 = arith.addi %12, %arg5 : index
          %14 = arith.muli %arg6, %c64 overflow<nsw> : index
          %15 = arith.addi %13, %14 : index
          %16 = loom.alloc [32, 32, 128] on @L1 : memref<32x32x128xf32>
          %17 = loom.alloc [32, 32, 32] on @L1 : memref<32x32x32xf32>
          %18 = loom.alloc [32, 128, 32] on @L1 : memref<32x128x32xf32>
          %19 = loom.alloc [32, 32] on @L1 : memref<32x32xf32>
          %20 = loom.alloc [32, 32] on @L1 : memref<32x32xf32>
          %21 = loom.alloc [32, 32] on @L1 : memref<32x32xf32>
          %22 = loom.alloc [32, 32] on @L1 : memref<32x32xf32>
          %23 = loom.alloc [32, 32] on @L1 : memref<32x32xf32>
          %24 = loom.alloc [32, 32, 128] on @L1 : memref<32x32x128xf32>
          %25 = loom.alloc [32, 32, 128] on @L1 : memref<32x32x128xf32>
          %26 = loom.alloc [32, 32, 128] on @L1 : memref<32x32x128xf32>
          %27 = arith.muli %15, %c4096 : index
          %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%27], sizes: [32, 32, 128], strides: [524288, 128, 1] : memref<32x4096x128xf32> to memref<32x32x128xf32, strided<[524288, 128, 1], offset: ?>>
          loom.copy %reinterpret_cast, %24 src_mem_space @DRAM dst_mem_space @L1, interconnect : [], broadcast : [1, 1] : memref<32x32x128xf32, strided<[524288, 128, 1], offset: ?>>, memref<32x32x128xf32>
          linalg.fill ins(%cst_1 : f32) outs(%25 : memref<32x32x128xf32>)
          linalg.fill ins(%cst_2 : f32) outs(%19 : memref<32x32xf32>)
          linalg.fill ins(%cst_3 : f32) outs(%20 : memref<32x32xf32>)
          scf.for %arg7 = %c0 to %c128 step %c1 {
            %28 = arith.muli %arg7, %c32 : index
            %reinterpret_cast_5 = memref.reinterpret_cast %arg0 to offset: [%28], sizes: [32, 128, 32], strides: [524288, 4096, 1] : memref<32x128x4096xf32> to memref<32x128x32xf32, strided<[524288, 4096, 1], offset: ?>>
            loom.copy %reinterpret_cast_5, %18 src_mem_space @DRAM dst_mem_space @L1, interconnect : [], broadcast : [1, 1] : memref<32x128x32xf32, strided<[524288, 4096, 1], offset: ?>>, memref<32x128x32xf32>
            linalg.fill ins(%cst_1 : f32) outs(%17 : memref<32x32x32xf32>)
            linalg.batch_matmul ins(%24, %18 : memref<32x32x128xf32>, memref<32x128x32xf32>) outs(%17 : memref<32x32x32xf32>)
            linalg.fill ins(%cst_3 : f32) outs(%21 : memref<32x32xf32>)
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%17 : memref<32x32x32xf32>) outs(%21 : memref<32x32xf32>) {
            ^bb0(%in: f32, %out: f32):
              %30 = arith.maximumf %in, %out : f32
              linalg.yield %30 : f32
            }
            linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%20, %21 : memref<32x32xf32>, memref<32x32xf32>) outs(%21 : memref<32x32xf32>) {
            ^bb0(%in: f32, %in_7: f32, %out: f32):
              %30 = arith.truncf %cst_0 : f64 to f32
              %31 = arith.mulf %in_7, %30 : f32
              %32 = arith.cmpf ogt, %in, %31 : f32
              %33 = arith.select %32, %in, %31 : f32
              linalg.yield %33 : f32
            }
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%17, %21 : memref<32x32x32xf32>, memref<32x32xf32>) outs(%17 : memref<32x32x32xf32>) {
            ^bb0(%in: f32, %in_7: f32, %out: f32):
              %30 = arith.truncf %cst_0 : f64 to f32
              %31 = arith.mulf %in, %30 : f32
              %32 = arith.subf %31, %in_7 : f32
              %33 = math.powf %cst, %32 : f32
              linalg.yield %33 : f32
            }
            linalg.fill ins(%cst_1 : f32) outs(%22 : memref<32x32xf32>)
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%17 : memref<32x32x32xf32>) outs(%22 : memref<32x32xf32>) {
            ^bb0(%in: f32, %out: f32):
              %30 = arith.addf %in, %out : f32
              linalg.yield %30 : f32
            }
            linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%20, %21 : memref<32x32xf32>, memref<32x32xf32>) outs(%23 : memref<32x32xf32>) {
            ^bb0(%in: f32, %in_7: f32, %out: f32):
              %30 = arith.subf %in, %in_7 : f32
              %31 = math.powf %cst, %30 : f32
              linalg.yield %31 : f32
            }
            linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%19, %23, %22 : memref<32x32xf32>, memref<32x32xf32>, memref<32x32xf32>) outs(%19 : memref<32x32xf32>) {
            ^bb0(%in: f32, %in_7: f32, %in_8: f32, %out: f32):
              %30 = arith.mulf %in, %in_7 : f32
              %31 = arith.addf %30, %in_8 : f32
              linalg.yield %31 : f32
            }
            %29 = arith.muli %arg7, %c4096 : index
            %reinterpret_cast_6 = memref.reinterpret_cast %arg1 to offset: [%29], sizes: [32, 32, 128], strides: [524288, 128, 1] : memref<32x4096x128xf32> to memref<32x32x128xf32, strided<[524288, 128, 1], offset: ?>>
            loom.copy %reinterpret_cast_6, %16 src_mem_space @DRAM dst_mem_space @L1, interconnect : [], broadcast : [1, 1] : memref<32x32x128xf32, strided<[524288, 128, 1], offset: ?>>, memref<32x32x128xf32>
            linalg.fill ins(%cst_1 : f32) outs(%26 : memref<32x32x128xf32>)
            linalg.batch_matmul ins(%17, %16 : memref<32x32x32xf32>, memref<32x32x128xf32>) outs(%26 : memref<32x32x128xf32>)
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%26, %25, %23 : memref<32x32x128xf32>, memref<32x32x128xf32>, memref<32x32xf32>) outs(%25 : memref<32x32x128xf32>) {
            ^bb0(%in: f32, %in_7: f32, %in_8: f32, %out: f32):
              %30 = arith.mulf %in_7, %in_8 : f32
              %31 = arith.addf %in, %30 : f32
              linalg.yield %31 : f32
            }
            linalg.copy ins(%21 : memref<32x32xf32>) outs(%20 : memref<32x32xf32>)
          }
          linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%25, %19 : memref<32x32x128xf32>, memref<32x32xf32>) outs(%24 : memref<32x32x128xf32>) {
          ^bb0(%in: f32, %in_5: f32, %out: f32):
            %28 = arith.divf %in, %in_5 : f32
            linalg.yield %28 : f32
          }
          %reinterpret_cast_4 = memref.reinterpret_cast %arg3 to offset: [%27], sizes: [32, 32, 128], strides: [524288, 128, 1] : memref<32x4096x128xf32> to memref<32x32x128xf32, strided<[524288, 128, 1], offset: ?>>
          loom.copy %24, %reinterpret_cast_4 src_mem_space @L1 dst_mem_space @DRAM, interconnect : [], broadcast : [1, 1] : memref<32x32x128xf32>, memref<32x32x128xf32, strided<[524288, 128, 1], offset: ?>>
        }
        scf.reduce 
      } {loom.iter_types = [#loom.iter_type<spatial>, #loom.iter_type<spatial>], loom.mapped_to_dims = [@x, @y]}
      return
    }
  }
  module attributes {loom.pass_name = "Materialize"} {
    func.func @attention__d0i1_d1i1__f10__d_d_a__BB32__BM32__BN32(%arg0: memref<32x128x4096xf32>, %arg1: memref<32x4096x128xf32>, %arg2: memref<32x4096x128xf32>, %arg3: memref<32x4096x128xf32>) {
      %c4096 = arith.constant 4096 : index
      %c128 = arith.constant 128 : index
      %c64 = arith.constant 64 : index
      %c2 = arith.constant 2 : index
      %cst = arith.constant 2.000000e+00 : f32
      %cst_0 = arith.constant 0.12751743074602467 : f64
      %cst_1 = arith.constant 0.000000e+00 : f32
      %cst_2 = arith.constant 1.000000e+00 : f32
      %cst_3 = arith.constant 0xFF800000 : f32
      %c32 = arith.constant 32 : index
      %c0 = arith.constant 0 : index
      %c8 = arith.constant 8 : index
      %c1 = arith.constant 1 : index
      scf.parallel (%arg4, %arg5) = (%c0, %c0) to (%c8, %c8) step (%c1, %c1) {
        scf.for %arg6 = %c0 to %c2 step %c1 {
          %12 = arith.muli %arg4, %c8 overflow<nsw> : index
          %13 = arith.addi %12, %arg5 : index
          %14 = arith.muli %arg6, %c64 overflow<nsw> : index
          %15 = arith.addi %13, %14 : index
          %16 = loom.alloc [32, 32, 128] on @L1 : memref<32x32x128xf32>
          %17 = loom.alloc [32, 32, 32] on @L1 : memref<32x32x32xf32>
          %18 = loom.alloc [32, 128, 32] on @L1 : memref<32x128x32xf32>
          %19 = loom.alloc [32, 32] on @L1 : memref<32x32xf32>
          %20 = loom.alloc [32, 32] on @L1 : memref<32x32xf32>
          %21 = loom.alloc [32, 32] on @L1 : memref<32x32xf32>
          %22 = loom.alloc [32, 32] on @L1 : memref<32x32xf32>
          %23 = loom.alloc [32, 32] on @L1 : memref<32x32xf32>
          %24 = loom.alloc [32, 32, 128] on @L1 : memref<32x32x128xf32>
          %25 = loom.alloc [32, 32, 128] on @L1 : memref<32x32x128xf32>
          %26 = loom.alloc [32, 32, 128] on @L1 : memref<32x32x128xf32>
          %27 = arith.muli %15, %c4096 : index
          %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%27], sizes: [32, 32, 128], strides: [524288, 128, 1] : memref<32x4096x128xf32> to memref<32x32x128xf32, strided<[524288, 128, 1], offset: ?>>
          loom.copy %reinterpret_cast, %24 src_mem_space @DRAM dst_mem_space @L1, interconnect : [], broadcast : [1, 1] : memref<32x32x128xf32, strided<[524288, 128, 1], offset: ?>>, memref<32x32x128xf32>
          linalg.fill ins(%cst_1 : f32) outs(%25 : memref<32x32x128xf32>)
          linalg.fill ins(%cst_2 : f32) outs(%19 : memref<32x32xf32>)
          linalg.fill ins(%cst_3 : f32) outs(%20 : memref<32x32xf32>)
          scf.for %arg7 = %c0 to %c128 step %c1 {
            %28 = arith.muli %arg7, %c32 : index
            %reinterpret_cast_5 = memref.reinterpret_cast %arg0 to offset: [%28], sizes: [32, 128, 32], strides: [524288, 4096, 1] : memref<32x128x4096xf32> to memref<32x128x32xf32, strided<[524288, 4096, 1], offset: ?>>
            loom.copy %reinterpret_cast_5, %18 src_mem_space @DRAM dst_mem_space @L1, interconnect : [], broadcast : [1, 1] : memref<32x128x32xf32, strided<[524288, 4096, 1], offset: ?>>, memref<32x128x32xf32>
            linalg.fill ins(%cst_1 : f32) outs(%17 : memref<32x32x32xf32>)
            linalg.batch_matmul ins(%24, %18 : memref<32x32x128xf32>, memref<32x128x32xf32>) outs(%17 : memref<32x32x32xf32>)
            linalg.fill ins(%cst_3 : f32) outs(%21 : memref<32x32xf32>)
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%17 : memref<32x32x32xf32>) outs(%21 : memref<32x32xf32>) {
            ^bb0(%in: f32, %out: f32):
              %30 = arith.maximumf %in, %out : f32
              linalg.yield %30 : f32
            }
            linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%20, %21 : memref<32x32xf32>, memref<32x32xf32>) outs(%21 : memref<32x32xf32>) {
            ^bb0(%in: f32, %in_7: f32, %out: f32):
              %30 = arith.truncf %cst_0 : f64 to f32
              %31 = arith.mulf %in_7, %30 : f32
              %32 = arith.cmpf ogt, %in, %31 : f32
              %33 = arith.select %32, %in, %31 : f32
              linalg.yield %33 : f32
            }
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%17, %21 : memref<32x32x32xf32>, memref<32x32xf32>) outs(%17 : memref<32x32x32xf32>) {
            ^bb0(%in: f32, %in_7: f32, %out: f32):
              %30 = arith.truncf %cst_0 : f64 to f32
              %31 = arith.mulf %in, %30 : f32
              %32 = arith.subf %31, %in_7 : f32
              %33 = math.powf %cst, %32 : f32
              linalg.yield %33 : f32
            }
            linalg.fill ins(%cst_1 : f32) outs(%22 : memref<32x32xf32>)
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%17 : memref<32x32x32xf32>) outs(%22 : memref<32x32xf32>) {
            ^bb0(%in: f32, %out: f32):
              %30 = arith.addf %in, %out : f32
              linalg.yield %30 : f32
            }
            linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%20, %21 : memref<32x32xf32>, memref<32x32xf32>) outs(%23 : memref<32x32xf32>) {
            ^bb0(%in: f32, %in_7: f32, %out: f32):
              %30 = arith.subf %in, %in_7 : f32
              %31 = math.powf %cst, %30 : f32
              linalg.yield %31 : f32
            }
            linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%19, %23, %22 : memref<32x32xf32>, memref<32x32xf32>, memref<32x32xf32>) outs(%19 : memref<32x32xf32>) {
            ^bb0(%in: f32, %in_7: f32, %in_8: f32, %out: f32):
              %30 = arith.mulf %in, %in_7 : f32
              %31 = arith.addf %30, %in_8 : f32
              linalg.yield %31 : f32
            }
            %29 = arith.muli %arg7, %c4096 : index
            %reinterpret_cast_6 = memref.reinterpret_cast %arg1 to offset: [%29], sizes: [32, 32, 128], strides: [524288, 128, 1] : memref<32x4096x128xf32> to memref<32x32x128xf32, strided<[524288, 128, 1], offset: ?>>
            loom.copy %reinterpret_cast_6, %16 src_mem_space @DRAM dst_mem_space @L1, interconnect : [@horizontal_links, @vertical_links], broadcast : [8, 8] : memref<32x32x128xf32, strided<[524288, 128, 1], offset: ?>>, memref<32x32x128xf32>
            linalg.fill ins(%cst_1 : f32) outs(%26 : memref<32x32x128xf32>)
            linalg.batch_matmul ins(%17, %16 : memref<32x32x32xf32>, memref<32x32x128xf32>) outs(%26 : memref<32x32x128xf32>)
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%26, %25, %23 : memref<32x32x128xf32>, memref<32x32x128xf32>, memref<32x32xf32>) outs(%25 : memref<32x32x128xf32>) {
            ^bb0(%in: f32, %in_7: f32, %in_8: f32, %out: f32):
              %30 = arith.mulf %in_7, %in_8 : f32
              %31 = arith.addf %in, %30 : f32
              linalg.yield %31 : f32
            }
            linalg.copy ins(%21 : memref<32x32xf32>) outs(%20 : memref<32x32xf32>)
          }
          linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%25, %19 : memref<32x32x128xf32>, memref<32x32xf32>) outs(%24 : memref<32x32x128xf32>) {
          ^bb0(%in: f32, %in_5: f32, %out: f32):
            %28 = arith.divf %in, %in_5 : f32
            linalg.yield %28 : f32
          }
          %reinterpret_cast_4 = memref.reinterpret_cast %arg3 to offset: [%27], sizes: [32, 32, 128], strides: [524288, 128, 1] : memref<32x4096x128xf32> to memref<32x32x128xf32, strided<[524288, 128, 1], offset: ?>>
          loom.copy %24, %reinterpret_cast_4 src_mem_space @L1 dst_mem_space @DRAM, interconnect : [], broadcast : [1, 1] : memref<32x32x128xf32>, memref<32x32x128xf32, strided<[524288, 128, 1], offset: ?>>
        }
        scf.reduce 
      } {loom.iter_types = [#loom.iter_type<spatial>, #loom.iter_type<spatial>], loom.mapped_to_dims = [@x, @y]}
      return
    }
  }
  module attributes {loom.pass_name = "Materialize"} {
    func.func @attention__d0i1_d1i1__f10__d_d_h__BB32__BM32__BN32(%arg0: memref<32x128x4096xf32>, %arg1: memref<32x4096x128xf32>, %arg2: memref<32x4096x128xf32>, %arg3: memref<32x4096x128xf32>) {
      %c4096 = arith.constant 4096 : index
      %c128 = arith.constant 128 : index
      %c64 = arith.constant 64 : index
      %c2 = arith.constant 2 : index
      %cst = arith.constant 2.000000e+00 : f32
      %cst_0 = arith.constant 0.12751743074602467 : f64
      %cst_1 = arith.constant 0.000000e+00 : f32
      %cst_2 = arith.constant 1.000000e+00 : f32
      %cst_3 = arith.constant 0xFF800000 : f32
      %c32 = arith.constant 32 : index
      %c0 = arith.constant 0 : index
      %c8 = arith.constant 8 : index
      %c1 = arith.constant 1 : index
      scf.parallel (%arg4, %arg5) = (%c0, %c0) to (%c8, %c8) step (%c1, %c1) {
        scf.for %arg6 = %c0 to %c2 step %c1 {
          %12 = arith.muli %arg4, %c8 overflow<nsw> : index
          %13 = arith.addi %12, %arg5 : index
          %14 = arith.muli %arg6, %c64 overflow<nsw> : index
          %15 = arith.addi %13, %14 : index
          %16 = loom.alloc [32, 32, 128] on @L1 : memref<32x32x128xf32>
          %17 = loom.alloc [32, 32, 32] on @L1 : memref<32x32x32xf32>
          %18 = loom.alloc [32, 128, 32] on @L1 : memref<32x128x32xf32>
          %19 = loom.alloc [32, 32] on @L1 : memref<32x32xf32>
          %20 = loom.alloc [32, 32] on @L1 : memref<32x32xf32>
          %21 = loom.alloc [32, 32] on @L1 : memref<32x32xf32>
          %22 = loom.alloc [32, 32] on @L1 : memref<32x32xf32>
          %23 = loom.alloc [32, 32] on @L1 : memref<32x32xf32>
          %24 = loom.alloc [32, 32, 128] on @L1 : memref<32x32x128xf32>
          %25 = loom.alloc [32, 32, 128] on @L1 : memref<32x32x128xf32>
          %26 = loom.alloc [32, 32, 128] on @L1 : memref<32x32x128xf32>
          %27 = arith.muli %15, %c4096 : index
          %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%27], sizes: [32, 32, 128], strides: [524288, 128, 1] : memref<32x4096x128xf32> to memref<32x32x128xf32, strided<[524288, 128, 1], offset: ?>>
          loom.copy %reinterpret_cast, %24 src_mem_space @DRAM dst_mem_space @L1, interconnect : [], broadcast : [1, 1] : memref<32x32x128xf32, strided<[524288, 128, 1], offset: ?>>, memref<32x32x128xf32>
          linalg.fill ins(%cst_1 : f32) outs(%25 : memref<32x32x128xf32>)
          linalg.fill ins(%cst_2 : f32) outs(%19 : memref<32x32xf32>)
          linalg.fill ins(%cst_3 : f32) outs(%20 : memref<32x32xf32>)
          scf.for %arg7 = %c0 to %c128 step %c1 {
            %28 = arith.muli %arg7, %c32 : index
            %reinterpret_cast_5 = memref.reinterpret_cast %arg0 to offset: [%28], sizes: [32, 128, 32], strides: [524288, 4096, 1] : memref<32x128x4096xf32> to memref<32x128x32xf32, strided<[524288, 4096, 1], offset: ?>>
            loom.copy %reinterpret_cast_5, %18 src_mem_space @DRAM dst_mem_space @L1, interconnect : [], broadcast : [1, 1] : memref<32x128x32xf32, strided<[524288, 4096, 1], offset: ?>>, memref<32x128x32xf32>
            linalg.fill ins(%cst_1 : f32) outs(%17 : memref<32x32x32xf32>)
            linalg.batch_matmul ins(%24, %18 : memref<32x32x128xf32>, memref<32x128x32xf32>) outs(%17 : memref<32x32x32xf32>)
            linalg.fill ins(%cst_3 : f32) outs(%21 : memref<32x32xf32>)
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%17 : memref<32x32x32xf32>) outs(%21 : memref<32x32xf32>) {
            ^bb0(%in: f32, %out: f32):
              %30 = arith.maximumf %in, %out : f32
              linalg.yield %30 : f32
            }
            linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%20, %21 : memref<32x32xf32>, memref<32x32xf32>) outs(%21 : memref<32x32xf32>) {
            ^bb0(%in: f32, %in_7: f32, %out: f32):
              %30 = arith.truncf %cst_0 : f64 to f32
              %31 = arith.mulf %in_7, %30 : f32
              %32 = arith.cmpf ogt, %in, %31 : f32
              %33 = arith.select %32, %in, %31 : f32
              linalg.yield %33 : f32
            }
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%17, %21 : memref<32x32x32xf32>, memref<32x32xf32>) outs(%17 : memref<32x32x32xf32>) {
            ^bb0(%in: f32, %in_7: f32, %out: f32):
              %30 = arith.truncf %cst_0 : f64 to f32
              %31 = arith.mulf %in, %30 : f32
              %32 = arith.subf %31, %in_7 : f32
              %33 = math.powf %cst, %32 : f32
              linalg.yield %33 : f32
            }
            linalg.fill ins(%cst_1 : f32) outs(%22 : memref<32x32xf32>)
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%17 : memref<32x32x32xf32>) outs(%22 : memref<32x32xf32>) {
            ^bb0(%in: f32, %out: f32):
              %30 = arith.addf %in, %out : f32
              linalg.yield %30 : f32
            }
            linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%20, %21 : memref<32x32xf32>, memref<32x32xf32>) outs(%23 : memref<32x32xf32>) {
            ^bb0(%in: f32, %in_7: f32, %out: f32):
              %30 = arith.subf %in, %in_7 : f32
              %31 = math.powf %cst, %30 : f32
              linalg.yield %31 : f32
            }
            linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%19, %23, %22 : memref<32x32xf32>, memref<32x32xf32>, memref<32x32xf32>) outs(%19 : memref<32x32xf32>) {
            ^bb0(%in: f32, %in_7: f32, %in_8: f32, %out: f32):
              %30 = arith.mulf %in, %in_7 : f32
              %31 = arith.addf %30, %in_8 : f32
              linalg.yield %31 : f32
            }
            %29 = arith.muli %arg7, %c4096 : index
            %reinterpret_cast_6 = memref.reinterpret_cast %arg1 to offset: [%29], sizes: [32, 32, 128], strides: [524288, 128, 1] : memref<32x4096x128xf32> to memref<32x32x128xf32, strided<[524288, 128, 1], offset: ?>>
            loom.copy %reinterpret_cast_6, %16 src_mem_space @DRAM dst_mem_space @L1, interconnect : [@horizontal_links], broadcast : [1, 8] : memref<32x32x128xf32, strided<[524288, 128, 1], offset: ?>>, memref<32x32x128xf32>
            linalg.fill ins(%cst_1 : f32) outs(%26 : memref<32x32x128xf32>)
            linalg.batch_matmul ins(%17, %16 : memref<32x32x32xf32>, memref<32x32x128xf32>) outs(%26 : memref<32x32x128xf32>)
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%26, %25, %23 : memref<32x32x128xf32>, memref<32x32x128xf32>, memref<32x32xf32>) outs(%25 : memref<32x32x128xf32>) {
            ^bb0(%in: f32, %in_7: f32, %in_8: f32, %out: f32):
              %30 = arith.mulf %in_7, %in_8 : f32
              %31 = arith.addf %in, %30 : f32
              linalg.yield %31 : f32
            }
            linalg.copy ins(%21 : memref<32x32xf32>) outs(%20 : memref<32x32xf32>)
          }
          linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%25, %19 : memref<32x32x128xf32>, memref<32x32xf32>) outs(%24 : memref<32x32x128xf32>) {
          ^bb0(%in: f32, %in_5: f32, %out: f32):
            %28 = arith.divf %in, %in_5 : f32
            linalg.yield %28 : f32
          }
          %reinterpret_cast_4 = memref.reinterpret_cast %arg3 to offset: [%27], sizes: [32, 32, 128], strides: [524288, 128, 1] : memref<32x4096x128xf32> to memref<32x32x128xf32, strided<[524288, 128, 1], offset: ?>>
          loom.copy %24, %reinterpret_cast_4 src_mem_space @L1 dst_mem_space @DRAM, interconnect : [], broadcast : [1, 1] : memref<32x32x128xf32>, memref<32x32x128xf32, strided<[524288, 128, 1], offset: ?>>
        }
        scf.reduce 
      } {loom.iter_types = [#loom.iter_type<spatial>, #loom.iter_type<spatial>], loom.mapped_to_dims = [@x, @y]}
      return
    }
  }
  module attributes {loom.pass_name = "Materialize"} {
    func.func @attention__d0i1_d1i1__f10__d_d_v__BB32__BM32__BN32(%arg0: memref<32x128x4096xf32>, %arg1: memref<32x4096x128xf32>, %arg2: memref<32x4096x128xf32>, %arg3: memref<32x4096x128xf32>) {
      %c4096 = arith.constant 4096 : index
      %c128 = arith.constant 128 : index
      %c64 = arith.constant 64 : index
      %c2 = arith.constant 2 : index
      %cst = arith.constant 2.000000e+00 : f32
      %cst_0 = arith.constant 0.12751743074602467 : f64
      %cst_1 = arith.constant 0.000000e+00 : f32
      %cst_2 = arith.constant 1.000000e+00 : f32
      %cst_3 = arith.constant 0xFF800000 : f32
      %c32 = arith.constant 32 : index
      %c0 = arith.constant 0 : index
      %c8 = arith.constant 8 : index
      %c1 = arith.constant 1 : index
      scf.parallel (%arg4, %arg5) = (%c0, %c0) to (%c8, %c8) step (%c1, %c1) {
        scf.for %arg6 = %c0 to %c2 step %c1 {
          %12 = arith.muli %arg4, %c8 overflow<nsw> : index
          %13 = arith.addi %12, %arg5 : index
          %14 = arith.muli %arg6, %c64 overflow<nsw> : index
          %15 = arith.addi %13, %14 : index
          %16 = loom.alloc [32, 32, 128] on @L1 : memref<32x32x128xf32>
          %17 = loom.alloc [32, 32, 32] on @L1 : memref<32x32x32xf32>
          %18 = loom.alloc [32, 128, 32] on @L1 : memref<32x128x32xf32>
          %19 = loom.alloc [32, 32] on @L1 : memref<32x32xf32>
          %20 = loom.alloc [32, 32] on @L1 : memref<32x32xf32>
          %21 = loom.alloc [32, 32] on @L1 : memref<32x32xf32>
          %22 = loom.alloc [32, 32] on @L1 : memref<32x32xf32>
          %23 = loom.alloc [32, 32] on @L1 : memref<32x32xf32>
          %24 = loom.alloc [32, 32, 128] on @L1 : memref<32x32x128xf32>
          %25 = loom.alloc [32, 32, 128] on @L1 : memref<32x32x128xf32>
          %26 = loom.alloc [32, 32, 128] on @L1 : memref<32x32x128xf32>
          %27 = arith.muli %15, %c4096 : index
          %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%27], sizes: [32, 32, 128], strides: [524288, 128, 1] : memref<32x4096x128xf32> to memref<32x32x128xf32, strided<[524288, 128, 1], offset: ?>>
          loom.copy %reinterpret_cast, %24 src_mem_space @DRAM dst_mem_space @L1, interconnect : [], broadcast : [1, 1] : memref<32x32x128xf32, strided<[524288, 128, 1], offset: ?>>, memref<32x32x128xf32>
          linalg.fill ins(%cst_1 : f32) outs(%25 : memref<32x32x128xf32>)
          linalg.fill ins(%cst_2 : f32) outs(%19 : memref<32x32xf32>)
          linalg.fill ins(%cst_3 : f32) outs(%20 : memref<32x32xf32>)
          scf.for %arg7 = %c0 to %c128 step %c1 {
            %28 = arith.muli %arg7, %c32 : index
            %reinterpret_cast_5 = memref.reinterpret_cast %arg0 to offset: [%28], sizes: [32, 128, 32], strides: [524288, 4096, 1] : memref<32x128x4096xf32> to memref<32x128x32xf32, strided<[524288, 4096, 1], offset: ?>>
            loom.copy %reinterpret_cast_5, %18 src_mem_space @DRAM dst_mem_space @L1, interconnect : [], broadcast : [1, 1] : memref<32x128x32xf32, strided<[524288, 4096, 1], offset: ?>>, memref<32x128x32xf32>
            linalg.fill ins(%cst_1 : f32) outs(%17 : memref<32x32x32xf32>)
            linalg.batch_matmul ins(%24, %18 : memref<32x32x128xf32>, memref<32x128x32xf32>) outs(%17 : memref<32x32x32xf32>)
            linalg.fill ins(%cst_3 : f32) outs(%21 : memref<32x32xf32>)
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%17 : memref<32x32x32xf32>) outs(%21 : memref<32x32xf32>) {
            ^bb0(%in: f32, %out: f32):
              %30 = arith.maximumf %in, %out : f32
              linalg.yield %30 : f32
            }
            linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%20, %21 : memref<32x32xf32>, memref<32x32xf32>) outs(%21 : memref<32x32xf32>) {
            ^bb0(%in: f32, %in_7: f32, %out: f32):
              %30 = arith.truncf %cst_0 : f64 to f32
              %31 = arith.mulf %in_7, %30 : f32
              %32 = arith.cmpf ogt, %in, %31 : f32
              %33 = arith.select %32, %in, %31 : f32
              linalg.yield %33 : f32
            }
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%17, %21 : memref<32x32x32xf32>, memref<32x32xf32>) outs(%17 : memref<32x32x32xf32>) {
            ^bb0(%in: f32, %in_7: f32, %out: f32):
              %30 = arith.truncf %cst_0 : f64 to f32
              %31 = arith.mulf %in, %30 : f32
              %32 = arith.subf %31, %in_7 : f32
              %33 = math.powf %cst, %32 : f32
              linalg.yield %33 : f32
            }
            linalg.fill ins(%cst_1 : f32) outs(%22 : memref<32x32xf32>)
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%17 : memref<32x32x32xf32>) outs(%22 : memref<32x32xf32>) {
            ^bb0(%in: f32, %out: f32):
              %30 = arith.addf %in, %out : f32
              linalg.yield %30 : f32
            }
            linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%20, %21 : memref<32x32xf32>, memref<32x32xf32>) outs(%23 : memref<32x32xf32>) {
            ^bb0(%in: f32, %in_7: f32, %out: f32):
              %30 = arith.subf %in, %in_7 : f32
              %31 = math.powf %cst, %30 : f32
              linalg.yield %31 : f32
            }
            linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%19, %23, %22 : memref<32x32xf32>, memref<32x32xf32>, memref<32x32xf32>) outs(%19 : memref<32x32xf32>) {
            ^bb0(%in: f32, %in_7: f32, %in_8: f32, %out: f32):
              %30 = arith.mulf %in, %in_7 : f32
              %31 = arith.addf %30, %in_8 : f32
              linalg.yield %31 : f32
            }
            %29 = arith.muli %arg7, %c4096 : index
            %reinterpret_cast_6 = memref.reinterpret_cast %arg1 to offset: [%29], sizes: [32, 32, 128], strides: [524288, 128, 1] : memref<32x4096x128xf32> to memref<32x32x128xf32, strided<[524288, 128, 1], offset: ?>>
            loom.copy %reinterpret_cast_6, %16 src_mem_space @DRAM dst_mem_space @L1, interconnect : [@vertical_links], broadcast : [8, 1] : memref<32x32x128xf32, strided<[524288, 128, 1], offset: ?>>, memref<32x32x128xf32>
            linalg.fill ins(%cst_1 : f32) outs(%26 : memref<32x32x128xf32>)
            linalg.batch_matmul ins(%17, %16 : memref<32x32x32xf32>, memref<32x32x128xf32>) outs(%26 : memref<32x32x128xf32>)
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%26, %25, %23 : memref<32x32x128xf32>, memref<32x32x128xf32>, memref<32x32xf32>) outs(%25 : memref<32x32x128xf32>) {
            ^bb0(%in: f32, %in_7: f32, %in_8: f32, %out: f32):
              %30 = arith.mulf %in_7, %in_8 : f32
              %31 = arith.addf %in, %30 : f32
              linalg.yield %31 : f32
            }
            linalg.copy ins(%21 : memref<32x32xf32>) outs(%20 : memref<32x32xf32>)
          }
          linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%25, %19 : memref<32x32x128xf32>, memref<32x32xf32>) outs(%24 : memref<32x32x128xf32>) {
          ^bb0(%in: f32, %in_5: f32, %out: f32):
            %28 = arith.divf %in, %in_5 : f32
            linalg.yield %28 : f32
          }
          %reinterpret_cast_4 = memref.reinterpret_cast %arg3 to offset: [%27], sizes: [32, 32, 128], strides: [524288, 128, 1] : memref<32x4096x128xf32> to memref<32x32x128xf32, strided<[524288, 128, 1], offset: ?>>
          loom.copy %24, %reinterpret_cast_4 src_mem_space @L1 dst_mem_space @DRAM, interconnect : [], broadcast : [1, 1] : memref<32x32x128xf32>, memref<32x32x128xf32, strided<[524288, 128, 1], offset: ?>>
        }
        scf.reduce 
      } {loom.iter_types = [#loom.iter_type<spatial>, #loom.iter_type<spatial>], loom.mapped_to_dims = [@x, @y]}
      return
    }
  }
  module attributes {loom.pass_name = "Materialize"} {
    func.func @attention__d0i1_d1i1__f10__d_a_d__BB32__BM32__BN32(%arg0: memref<32x128x4096xf32>, %arg1: memref<32x4096x128xf32>, %arg2: memref<32x4096x128xf32>, %arg3: memref<32x4096x128xf32>) {
      %c4096 = arith.constant 4096 : index
      %c128 = arith.constant 128 : index
      %c64 = arith.constant 64 : index
      %c2 = arith.constant 2 : index
      %cst = arith.constant 2.000000e+00 : f32
      %cst_0 = arith.constant 0.12751743074602467 : f64
      %cst_1 = arith.constant 0.000000e+00 : f32
      %cst_2 = arith.constant 1.000000e+00 : f32
      %cst_3 = arith.constant 0xFF800000 : f32
      %c32 = arith.constant 32 : index
      %c0 = arith.constant 0 : index
      %c8 = arith.constant 8 : index
      %c1 = arith.constant 1 : index
      scf.parallel (%arg4, %arg5) = (%c0, %c0) to (%c8, %c8) step (%c1, %c1) {
        scf.for %arg6 = %c0 to %c2 step %c1 {
          %12 = arith.muli %arg4, %c8 overflow<nsw> : index
          %13 = arith.addi %12, %arg5 : index
          %14 = arith.muli %arg6, %c64 overflow<nsw> : index
          %15 = arith.addi %13, %14 : index
          %16 = loom.alloc [32, 32, 128] on @L1 : memref<32x32x128xf32>
          %17 = loom.alloc [32, 32, 32] on @L1 : memref<32x32x32xf32>
          %18 = loom.alloc [32, 128, 32] on @L1 : memref<32x128x32xf32>
          %19 = loom.alloc [32, 32] on @L1 : memref<32x32xf32>
          %20 = loom.alloc [32, 32] on @L1 : memref<32x32xf32>
          %21 = loom.alloc [32, 32] on @L1 : memref<32x32xf32>
          %22 = loom.alloc [32, 32] on @L1 : memref<32x32xf32>
          %23 = loom.alloc [32, 32] on @L1 : memref<32x32xf32>
          %24 = loom.alloc [32, 32, 128] on @L1 : memref<32x32x128xf32>
          %25 = loom.alloc [32, 32, 128] on @L1 : memref<32x32x128xf32>
          %26 = loom.alloc [32, 32, 128] on @L1 : memref<32x32x128xf32>
          %27 = arith.muli %15, %c4096 : index
          %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%27], sizes: [32, 32, 128], strides: [524288, 128, 1] : memref<32x4096x128xf32> to memref<32x32x128xf32, strided<[524288, 128, 1], offset: ?>>
          loom.copy %reinterpret_cast, %24 src_mem_space @DRAM dst_mem_space @L1, interconnect : [], broadcast : [1, 1] : memref<32x32x128xf32, strided<[524288, 128, 1], offset: ?>>, memref<32x32x128xf32>
          linalg.fill ins(%cst_1 : f32) outs(%25 : memref<32x32x128xf32>)
          linalg.fill ins(%cst_2 : f32) outs(%19 : memref<32x32xf32>)
          linalg.fill ins(%cst_3 : f32) outs(%20 : memref<32x32xf32>)
          scf.for %arg7 = %c0 to %c128 step %c1 {
            %28 = arith.muli %arg7, %c32 : index
            %reinterpret_cast_5 = memref.reinterpret_cast %arg0 to offset: [%28], sizes: [32, 128, 32], strides: [524288, 4096, 1] : memref<32x128x4096xf32> to memref<32x128x32xf32, strided<[524288, 4096, 1], offset: ?>>
            loom.copy %reinterpret_cast_5, %18 src_mem_space @DRAM dst_mem_space @L1, interconnect : [@horizontal_links, @vertical_links], broadcast : [8, 8] : memref<32x128x32xf32, strided<[524288, 4096, 1], offset: ?>>, memref<32x128x32xf32>
            linalg.fill ins(%cst_1 : f32) outs(%17 : memref<32x32x32xf32>)
            linalg.batch_matmul ins(%24, %18 : memref<32x32x128xf32>, memref<32x128x32xf32>) outs(%17 : memref<32x32x32xf32>)
            linalg.fill ins(%cst_3 : f32) outs(%21 : memref<32x32xf32>)
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%17 : memref<32x32x32xf32>) outs(%21 : memref<32x32xf32>) {
            ^bb0(%in: f32, %out: f32):
              %30 = arith.maximumf %in, %out : f32
              linalg.yield %30 : f32
            }
            linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%20, %21 : memref<32x32xf32>, memref<32x32xf32>) outs(%21 : memref<32x32xf32>) {
            ^bb0(%in: f32, %in_7: f32, %out: f32):
              %30 = arith.truncf %cst_0 : f64 to f32
              %31 = arith.mulf %in_7, %30 : f32
              %32 = arith.cmpf ogt, %in, %31 : f32
              %33 = arith.select %32, %in, %31 : f32
              linalg.yield %33 : f32
            }
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%17, %21 : memref<32x32x32xf32>, memref<32x32xf32>) outs(%17 : memref<32x32x32xf32>) {
            ^bb0(%in: f32, %in_7: f32, %out: f32):
              %30 = arith.truncf %cst_0 : f64 to f32
              %31 = arith.mulf %in, %30 : f32
              %32 = arith.subf %31, %in_7 : f32
              %33 = math.powf %cst, %32 : f32
              linalg.yield %33 : f32
            }
            linalg.fill ins(%cst_1 : f32) outs(%22 : memref<32x32xf32>)
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%17 : memref<32x32x32xf32>) outs(%22 : memref<32x32xf32>) {
            ^bb0(%in: f32, %out: f32):
              %30 = arith.addf %in, %out : f32
              linalg.yield %30 : f32
            }
            linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%20, %21 : memref<32x32xf32>, memref<32x32xf32>) outs(%23 : memref<32x32xf32>) {
            ^bb0(%in: f32, %in_7: f32, %out: f32):
              %30 = arith.subf %in, %in_7 : f32
              %31 = math.powf %cst, %30 : f32
              linalg.yield %31 : f32
            }
            linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%19, %23, %22 : memref<32x32xf32>, memref<32x32xf32>, memref<32x32xf32>) outs(%19 : memref<32x32xf32>) {
            ^bb0(%in: f32, %in_7: f32, %in_8: f32, %out: f32):
              %30 = arith.mulf %in, %in_7 : f32
              %31 = arith.addf %30, %in_8 : f32
              linalg.yield %31 : f32
            }
            %29 = arith.muli %arg7, %c4096 : index
            %reinterpret_cast_6 = memref.reinterpret_cast %arg1 to offset: [%29], sizes: [32, 32, 128], strides: [524288, 128, 1] : memref<32x4096x128xf32> to memref<32x32x128xf32, strided<[524288, 128, 1], offset: ?>>
            loom.copy %reinterpret_cast_6, %16 src_mem_space @DRAM dst_mem_space @L1, interconnect : [], broadcast : [1, 1] : memref<32x32x128xf32, strided<[524288, 128, 1], offset: ?>>, memref<32x32x128xf32>
            linalg.fill ins(%cst_1 : f32) outs(%26 : memref<32x32x128xf32>)
            linalg.batch_matmul ins(%17, %16 : memref<32x32x32xf32>, memref<32x32x128xf32>) outs(%26 : memref<32x32x128xf32>)
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%26, %25, %23 : memref<32x32x128xf32>, memref<32x32x128xf32>, memref<32x32xf32>) outs(%25 : memref<32x32x128xf32>) {
            ^bb0(%in: f32, %in_7: f32, %in_8: f32, %out: f32):
              %30 = arith.mulf %in_7, %in_8 : f32
              %31 = arith.addf %in, %30 : f32
              linalg.yield %31 : f32
            }
            linalg.copy ins(%21 : memref<32x32xf32>) outs(%20 : memref<32x32xf32>)
          }
          linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%25, %19 : memref<32x32x128xf32>, memref<32x32xf32>) outs(%24 : memref<32x32x128xf32>) {
          ^bb0(%in: f32, %in_5: f32, %out: f32):
            %28 = arith.divf %in, %in_5 : f32
            linalg.yield %28 : f32
          }
          %reinterpret_cast_4 = memref.reinterpret_cast %arg3 to offset: [%27], sizes: [32, 32, 128], strides: [524288, 128, 1] : memref<32x4096x128xf32> to memref<32x32x128xf32, strided<[524288, 128, 1], offset: ?>>
          loom.copy %24, %reinterpret_cast_4 src_mem_space @L1 dst_mem_space @DRAM, interconnect : [], broadcast : [1, 1] : memref<32x32x128xf32>, memref<32x32x128xf32, strided<[524288, 128, 1], offset: ?>>
        }
        scf.reduce 
      } {loom.iter_types = [#loom.iter_type<spatial>, #loom.iter_type<spatial>], loom.mapped_to_dims = [@x, @y]}
      return
    }
  }
  module attributes {loom.pass_name = "Materialize"} {
    func.func @attention__d0i1_d1i1__f10__d_a_a__BB32__BM32__BN32(%arg0: memref<32x128x4096xf32>, %arg1: memref<32x4096x128xf32>, %arg2: memref<32x4096x128xf32>, %arg3: memref<32x4096x128xf32>) {
      %c4096 = arith.constant 4096 : index
      %c128 = arith.constant 128 : index
      %c64 = arith.constant 64 : index
      %c2 = arith.constant 2 : index
      %cst = arith.constant 2.000000e+00 : f32
      %cst_0 = arith.constant 0.12751743074602467 : f64
      %cst_1 = arith.constant 0.000000e+00 : f32
      %cst_2 = arith.constant 1.000000e+00 : f32
      %cst_3 = arith.constant 0xFF800000 : f32
      %c32 = arith.constant 32 : index
      %c0 = arith.constant 0 : index
      %c8 = arith.constant 8 : index
      %c1 = arith.constant 1 : index
      scf.parallel (%arg4, %arg5) = (%c0, %c0) to (%c8, %c8) step (%c1, %c1) {
        scf.for %arg6 = %c0 to %c2 step %c1 {
          %12 = arith.muli %arg4, %c8 overflow<nsw> : index
          %13 = arith.addi %12, %arg5 : index
          %14 = arith.muli %arg6, %c64 overflow<nsw> : index
          %15 = arith.addi %13, %14 : index
          %16 = loom.alloc [32, 32, 128] on @L1 : memref<32x32x128xf32>
          %17 = loom.alloc [32, 32, 32] on @L1 : memref<32x32x32xf32>
          %18 = loom.alloc [32, 128, 32] on @L1 : memref<32x128x32xf32>
          %19 = loom.alloc [32, 32] on @L1 : memref<32x32xf32>
          %20 = loom.alloc [32, 32] on @L1 : memref<32x32xf32>
          %21 = loom.alloc [32, 32] on @L1 : memref<32x32xf32>
          %22 = loom.alloc [32, 32] on @L1 : memref<32x32xf32>
          %23 = loom.alloc [32, 32] on @L1 : memref<32x32xf32>
          %24 = loom.alloc [32, 32, 128] on @L1 : memref<32x32x128xf32>
          %25 = loom.alloc [32, 32, 128] on @L1 : memref<32x32x128xf32>
          %26 = loom.alloc [32, 32, 128] on @L1 : memref<32x32x128xf32>
          %27 = arith.muli %15, %c4096 : index
          %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%27], sizes: [32, 32, 128], strides: [524288, 128, 1] : memref<32x4096x128xf32> to memref<32x32x128xf32, strided<[524288, 128, 1], offset: ?>>
          loom.copy %reinterpret_cast, %24 src_mem_space @DRAM dst_mem_space @L1, interconnect : [], broadcast : [1, 1] : memref<32x32x128xf32, strided<[524288, 128, 1], offset: ?>>, memref<32x32x128xf32>
          linalg.fill ins(%cst_1 : f32) outs(%25 : memref<32x32x128xf32>)
          linalg.fill ins(%cst_2 : f32) outs(%19 : memref<32x32xf32>)
          linalg.fill ins(%cst_3 : f32) outs(%20 : memref<32x32xf32>)
          scf.for %arg7 = %c0 to %c128 step %c1 {
            %28 = arith.muli %arg7, %c32 : index
            %reinterpret_cast_5 = memref.reinterpret_cast %arg0 to offset: [%28], sizes: [32, 128, 32], strides: [524288, 4096, 1] : memref<32x128x4096xf32> to memref<32x128x32xf32, strided<[524288, 4096, 1], offset: ?>>
            loom.copy %reinterpret_cast_5, %18 src_mem_space @DRAM dst_mem_space @L1, interconnect : [@horizontal_links, @vertical_links], broadcast : [8, 8] : memref<32x128x32xf32, strided<[524288, 4096, 1], offset: ?>>, memref<32x128x32xf32>
            linalg.fill ins(%cst_1 : f32) outs(%17 : memref<32x32x32xf32>)
            linalg.batch_matmul ins(%24, %18 : memref<32x32x128xf32>, memref<32x128x32xf32>) outs(%17 : memref<32x32x32xf32>)
            linalg.fill ins(%cst_3 : f32) outs(%21 : memref<32x32xf32>)
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%17 : memref<32x32x32xf32>) outs(%21 : memref<32x32xf32>) {
            ^bb0(%in: f32, %out: f32):
              %30 = arith.maximumf %in, %out : f32
              linalg.yield %30 : f32
            }
            linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%20, %21 : memref<32x32xf32>, memref<32x32xf32>) outs(%21 : memref<32x32xf32>) {
            ^bb0(%in: f32, %in_7: f32, %out: f32):
              %30 = arith.truncf %cst_0 : f64 to f32
              %31 = arith.mulf %in_7, %30 : f32
              %32 = arith.cmpf ogt, %in, %31 : f32
              %33 = arith.select %32, %in, %31 : f32
              linalg.yield %33 : f32
            }
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%17, %21 : memref<32x32x32xf32>, memref<32x32xf32>) outs(%17 : memref<32x32x32xf32>) {
            ^bb0(%in: f32, %in_7: f32, %out: f32):
              %30 = arith.truncf %cst_0 : f64 to f32
              %31 = arith.mulf %in, %30 : f32
              %32 = arith.subf %31, %in_7 : f32
              %33 = math.powf %cst, %32 : f32
              linalg.yield %33 : f32
            }
            linalg.fill ins(%cst_1 : f32) outs(%22 : memref<32x32xf32>)
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%17 : memref<32x32x32xf32>) outs(%22 : memref<32x32xf32>) {
            ^bb0(%in: f32, %out: f32):
              %30 = arith.addf %in, %out : f32
              linalg.yield %30 : f32
            }
            linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%20, %21 : memref<32x32xf32>, memref<32x32xf32>) outs(%23 : memref<32x32xf32>) {
            ^bb0(%in: f32, %in_7: f32, %out: f32):
              %30 = arith.subf %in, %in_7 : f32
              %31 = math.powf %cst, %30 : f32
              linalg.yield %31 : f32
            }
            linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%19, %23, %22 : memref<32x32xf32>, memref<32x32xf32>, memref<32x32xf32>) outs(%19 : memref<32x32xf32>) {
            ^bb0(%in: f32, %in_7: f32, %in_8: f32, %out: f32):
              %30 = arith.mulf %in, %in_7 : f32
              %31 = arith.addf %30, %in_8 : f32
              linalg.yield %31 : f32
            }
            %29 = arith.muli %arg7, %c4096 : index
            %reinterpret_cast_6 = memref.reinterpret_cast %arg1 to offset: [%29], sizes: [32, 32, 128], strides: [524288, 128, 1] : memref<32x4096x128xf32> to memref<32x32x128xf32, strided<[524288, 128, 1], offset: ?>>
            loom.copy %reinterpret_cast_6, %16 src_mem_space @DRAM dst_mem_space @L1, interconnect : [@horizontal_links, @vertical_links], broadcast : [8, 8] : memref<32x32x128xf32, strided<[524288, 128, 1], offset: ?>>, memref<32x32x128xf32>
            linalg.fill ins(%cst_1 : f32) outs(%26 : memref<32x32x128xf32>)
            linalg.batch_matmul ins(%17, %16 : memref<32x32x32xf32>, memref<32x32x128xf32>) outs(%26 : memref<32x32x128xf32>)
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%26, %25, %23 : memref<32x32x128xf32>, memref<32x32x128xf32>, memref<32x32xf32>) outs(%25 : memref<32x32x128xf32>) {
            ^bb0(%in: f32, %in_7: f32, %in_8: f32, %out: f32):
              %30 = arith.mulf %in_7, %in_8 : f32
              %31 = arith.addf %in, %30 : f32
              linalg.yield %31 : f32
            }
            linalg.copy ins(%21 : memref<32x32xf32>) outs(%20 : memref<32x32xf32>)
          }
          linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%25, %19 : memref<32x32x128xf32>, memref<32x32xf32>) outs(%24 : memref<32x32x128xf32>) {
          ^bb0(%in: f32, %in_5: f32, %out: f32):
            %28 = arith.divf %in, %in_5 : f32
            linalg.yield %28 : f32
          }
          %reinterpret_cast_4 = memref.reinterpret_cast %arg3 to offset: [%27], sizes: [32, 32, 128], strides: [524288, 128, 1] : memref<32x4096x128xf32> to memref<32x32x128xf32, strided<[524288, 128, 1], offset: ?>>
          loom.copy %24, %reinterpret_cast_4 src_mem_space @L1 dst_mem_space @DRAM, interconnect : [], broadcast : [1, 1] : memref<32x32x128xf32>, memref<32x32x128xf32, strided<[524288, 128, 1], offset: ?>>
        }
        scf.reduce 
      } {loom.iter_types = [#loom.iter_type<spatial>, #loom.iter_type<spatial>], loom.mapped_to_dims = [@x, @y]}
      return
    }
  }
  module attributes {loom.pass_name = "Materialize"} {
    func.func @attention__d0i1_d1i1__f10__d_a_h__BB32__BM32__BN32(%arg0: memref<32x128x4096xf32>, %arg1: memref<32x4096x128xf32>, %arg2: memref<32x4096x128xf32>, %arg3: memref<32x4096x128xf32>) {
      %c4096 = arith.constant 4096 : index
      %c128 = arith.constant 128 : index
      %c64 = arith.constant 64 : index
      %c2 = arith.constant 2 : index
      %cst = arith.constant 2.000000e+00 : f32
      %cst_0 = arith.constant 0.12751743074602467 : f64
      %cst_1 = arith.constant 0.000000e+00 : f32
      %cst_2 = arith.constant 1.000000e+00 : f32
      %cst_3 = arith.constant 0xFF800000 : f32
      %c32 = arith.constant 32 : index
      %c0 = arith.constant 0 : index
      %c8 = arith.constant 8 : index
      %c1 = arith.constant 1 : index
      scf.parallel (%arg4, %arg5) = (%c0, %c0) to (%c8, %c8) step (%c1, %c1) {
        scf.for %arg6 = %c0 to %c2 step %c1 {
          %12 = arith.muli %arg4, %c8 overflow<nsw> : index
          %13 = arith.addi %12, %arg5 : index
          %14 = arith.muli %arg6, %c64 overflow<nsw> : index
          %15 = arith.addi %13, %14 : index
          %16 = loom.alloc [32, 32, 128] on @L1 : memref<32x32x128xf32>
          %17 = loom.alloc [32, 32, 32] on @L1 : memref<32x32x32xf32>
          %18 = loom.alloc [32, 128, 32] on @L1 : memref<32x128x32xf32>
          %19 = loom.alloc [32, 32] on @L1 : memref<32x32xf32>
          %20 = loom.alloc [32, 32] on @L1 : memref<32x32xf32>
          %21 = loom.alloc [32, 32] on @L1 : memref<32x32xf32>
          %22 = loom.alloc [32, 32] on @L1 : memref<32x32xf32>
          %23 = loom.alloc [32, 32] on @L1 : memref<32x32xf32>
          %24 = loom.alloc [32, 32, 128] on @L1 : memref<32x32x128xf32>
          %25 = loom.alloc [32, 32, 128] on @L1 : memref<32x32x128xf32>
          %26 = loom.alloc [32, 32, 128] on @L1 : memref<32x32x128xf32>
          %27 = arith.muli %15, %c4096 : index
          %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%27], sizes: [32, 32, 128], strides: [524288, 128, 1] : memref<32x4096x128xf32> to memref<32x32x128xf32, strided<[524288, 128, 1], offset: ?>>
          loom.copy %reinterpret_cast, %24 src_mem_space @DRAM dst_mem_space @L1, interconnect : [], broadcast : [1, 1] : memref<32x32x128xf32, strided<[524288, 128, 1], offset: ?>>, memref<32x32x128xf32>
          linalg.fill ins(%cst_1 : f32) outs(%25 : memref<32x32x128xf32>)
          linalg.fill ins(%cst_2 : f32) outs(%19 : memref<32x32xf32>)
          linalg.fill ins(%cst_3 : f32) outs(%20 : memref<32x32xf32>)
          scf.for %arg7 = %c0 to %c128 step %c1 {
            %28 = arith.muli %arg7, %c32 : index
            %reinterpret_cast_5 = memref.reinterpret_cast %arg0 to offset: [%28], sizes: [32, 128, 32], strides: [524288, 4096, 1] : memref<32x128x4096xf32> to memref<32x128x32xf32, strided<[524288, 4096, 1], offset: ?>>
            loom.copy %reinterpret_cast_5, %18 src_mem_space @DRAM dst_mem_space @L1, interconnect : [@horizontal_links, @vertical_links], broadcast : [8, 8] : memref<32x128x32xf32, strided<[524288, 4096, 1], offset: ?>>, memref<32x128x32xf32>
            linalg.fill ins(%cst_1 : f32) outs(%17 : memref<32x32x32xf32>)
            linalg.batch_matmul ins(%24, %18 : memref<32x32x128xf32>, memref<32x128x32xf32>) outs(%17 : memref<32x32x32xf32>)
            linalg.fill ins(%cst_3 : f32) outs(%21 : memref<32x32xf32>)
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%17 : memref<32x32x32xf32>) outs(%21 : memref<32x32xf32>) {
            ^bb0(%in: f32, %out: f32):
              %30 = arith.maximumf %in, %out : f32
              linalg.yield %30 : f32
            }
            linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%20, %21 : memref<32x32xf32>, memref<32x32xf32>) outs(%21 : memref<32x32xf32>) {
            ^bb0(%in: f32, %in_7: f32, %out: f32):
              %30 = arith.truncf %cst_0 : f64 to f32
              %31 = arith.mulf %in_7, %30 : f32
              %32 = arith.cmpf ogt, %in, %31 : f32
              %33 = arith.select %32, %in, %31 : f32
              linalg.yield %33 : f32
            }
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%17, %21 : memref<32x32x32xf32>, memref<32x32xf32>) outs(%17 : memref<32x32x32xf32>) {
            ^bb0(%in: f32, %in_7: f32, %out: f32):
              %30 = arith.truncf %cst_0 : f64 to f32
              %31 = arith.mulf %in, %30 : f32
              %32 = arith.subf %31, %in_7 : f32
              %33 = math.powf %cst, %32 : f32
              linalg.yield %33 : f32
            }
            linalg.fill ins(%cst_1 : f32) outs(%22 : memref<32x32xf32>)
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%17 : memref<32x32x32xf32>) outs(%22 : memref<32x32xf32>) {
            ^bb0(%in: f32, %out: f32):
              %30 = arith.addf %in, %out : f32
              linalg.yield %30 : f32
            }
            linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%20, %21 : memref<32x32xf32>, memref<32x32xf32>) outs(%23 : memref<32x32xf32>) {
            ^bb0(%in: f32, %in_7: f32, %out: f32):
              %30 = arith.subf %in, %in_7 : f32
              %31 = math.powf %cst, %30 : f32
              linalg.yield %31 : f32
            }
            linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%19, %23, %22 : memref<32x32xf32>, memref<32x32xf32>, memref<32x32xf32>) outs(%19 : memref<32x32xf32>) {
            ^bb0(%in: f32, %in_7: f32, %in_8: f32, %out: f32):
              %30 = arith.mulf %in, %in_7 : f32
              %31 = arith.addf %30, %in_8 : f32
              linalg.yield %31 : f32
            }
            %29 = arith.muli %arg7, %c4096 : index
            %reinterpret_cast_6 = memref.reinterpret_cast %arg1 to offset: [%29], sizes: [32, 32, 128], strides: [524288, 128, 1] : memref<32x4096x128xf32> to memref<32x32x128xf32, strided<[524288, 128, 1], offset: ?>>
            loom.copy %reinterpret_cast_6, %16 src_mem_space @DRAM dst_mem_space @L1, interconnect : [@horizontal_links], broadcast : [1, 8] : memref<32x32x128xf32, strided<[524288, 128, 1], offset: ?>>, memref<32x32x128xf32>
            linalg.fill ins(%cst_1 : f32) outs(%26 : memref<32x32x128xf32>)
            linalg.batch_matmul ins(%17, %16 : memref<32x32x32xf32>, memref<32x32x128xf32>) outs(%26 : memref<32x32x128xf32>)
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%26, %25, %23 : memref<32x32x128xf32>, memref<32x32x128xf32>, memref<32x32xf32>) outs(%25 : memref<32x32x128xf32>) {
            ^bb0(%in: f32, %in_7: f32, %in_8: f32, %out: f32):
              %30 = arith.mulf %in_7, %in_8 : f32
              %31 = arith.addf %in, %30 : f32
              linalg.yield %31 : f32
            }
            linalg.copy ins(%21 : memref<32x32xf32>) outs(%20 : memref<32x32xf32>)
          }
          linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%25, %19 : memref<32x32x128xf32>, memref<32x32xf32>) outs(%24 : memref<32x32x128xf32>) {
          ^bb0(%in: f32, %in_5: f32, %out: f32):
            %28 = arith.divf %in, %in_5 : f32
            linalg.yield %28 : f32
          }
          %reinterpret_cast_4 = memref.reinterpret_cast %arg3 to offset: [%27], sizes: [32, 32, 128], strides: [524288, 128, 1] : memref<32x4096x128xf32> to memref<32x32x128xf32, strided<[524288, 128, 1], offset: ?>>
          loom.copy %24, %reinterpret_cast_4 src_mem_space @L1 dst_mem_space @DRAM, interconnect : [], broadcast : [1, 1] : memref<32x32x128xf32>, memref<32x32x128xf32, strided<[524288, 128, 1], offset: ?>>
        }
        scf.reduce 
      } {loom.iter_types = [#loom.iter_type<spatial>, #loom.iter_type<spatial>], loom.mapped_to_dims = [@x, @y]}
      return
    }
  }
  module attributes {loom.pass_name = "Materialize"} {
    func.func @attention__d0i1_d1i1__f10__d_a_v__BB32__BM32__BN32(%arg0: memref<32x128x4096xf32>, %arg1: memref<32x4096x128xf32>, %arg2: memref<32x4096x128xf32>, %arg3: memref<32x4096x128xf32>) {
      %c4096 = arith.constant 4096 : index
      %c128 = arith.constant 128 : index
      %c64 = arith.constant 64 : index
      %c2 = arith.constant 2 : index
      %cst = arith.constant 2.000000e+00 : f32
      %cst_0 = arith.constant 0.12751743074602467 : f64
      %cst_1 = arith.constant 0.000000e+00 : f32
      %cst_2 = arith.constant 1.000000e+00 : f32
      %cst_3 = arith.constant 0xFF800000 : f32
      %c32 = arith.constant 32 : index
      %c0 = arith.constant 0 : index
      %c8 = arith.constant 8 : index
      %c1 = arith.constant 1 : index
      scf.parallel (%arg4, %arg5) = (%c0, %c0) to (%c8, %c8) step (%c1, %c1) {
        scf.for %arg6 = %c0 to %c2 step %c1 {
          %12 = arith.muli %arg4, %c8 overflow<nsw> : index
          %13 = arith.addi %12, %arg5 : index
          %14 = arith.muli %arg6, %c64 overflow<nsw> : index
          %15 = arith.addi %13, %14 : index
          %16 = loom.alloc [32, 32, 128] on @L1 : memref<32x32x128xf32>
          %17 = loom.alloc [32, 32, 32] on @L1 : memref<32x32x32xf32>
          %18 = loom.alloc [32, 128, 32] on @L1 : memref<32x128x32xf32>
          %19 = loom.alloc [32, 32] on @L1 : memref<32x32xf32>
          %20 = loom.alloc [32, 32] on @L1 : memref<32x32xf32>
          %21 = loom.alloc [32, 32] on @L1 : memref<32x32xf32>
          %22 = loom.alloc [32, 32] on @L1 : memref<32x32xf32>
          %23 = loom.alloc [32, 32] on @L1 : memref<32x32xf32>
          %24 = loom.alloc [32, 32, 128] on @L1 : memref<32x32x128xf32>
          %25 = loom.alloc [32, 32, 128] on @L1 : memref<32x32x128xf32>
          %26 = loom.alloc [32, 32, 128] on @L1 : memref<32x32x128xf32>
          %27 = arith.muli %15, %c4096 : index
          %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%27], sizes: [32, 32, 128], strides: [524288, 128, 1] : memref<32x4096x128xf32> to memref<32x32x128xf32, strided<[524288, 128, 1], offset: ?>>
          loom.copy %reinterpret_cast, %24 src_mem_space @DRAM dst_mem_space @L1, interconnect : [], broadcast : [1, 1] : memref<32x32x128xf32, strided<[524288, 128, 1], offset: ?>>, memref<32x32x128xf32>
          linalg.fill ins(%cst_1 : f32) outs(%25 : memref<32x32x128xf32>)
          linalg.fill ins(%cst_2 : f32) outs(%19 : memref<32x32xf32>)
          linalg.fill ins(%cst_3 : f32) outs(%20 : memref<32x32xf32>)
          scf.for %arg7 = %c0 to %c128 step %c1 {
            %28 = arith.muli %arg7, %c32 : index
            %reinterpret_cast_5 = memref.reinterpret_cast %arg0 to offset: [%28], sizes: [32, 128, 32], strides: [524288, 4096, 1] : memref<32x128x4096xf32> to memref<32x128x32xf32, strided<[524288, 4096, 1], offset: ?>>
            loom.copy %reinterpret_cast_5, %18 src_mem_space @DRAM dst_mem_space @L1, interconnect : [@horizontal_links, @vertical_links], broadcast : [8, 8] : memref<32x128x32xf32, strided<[524288, 4096, 1], offset: ?>>, memref<32x128x32xf32>
            linalg.fill ins(%cst_1 : f32) outs(%17 : memref<32x32x32xf32>)
            linalg.batch_matmul ins(%24, %18 : memref<32x32x128xf32>, memref<32x128x32xf32>) outs(%17 : memref<32x32x32xf32>)
            linalg.fill ins(%cst_3 : f32) outs(%21 : memref<32x32xf32>)
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%17 : memref<32x32x32xf32>) outs(%21 : memref<32x32xf32>) {
            ^bb0(%in: f32, %out: f32):
              %30 = arith.maximumf %in, %out : f32
              linalg.yield %30 : f32
            }
            linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%20, %21 : memref<32x32xf32>, memref<32x32xf32>) outs(%21 : memref<32x32xf32>) {
            ^bb0(%in: f32, %in_7: f32, %out: f32):
              %30 = arith.truncf %cst_0 : f64 to f32
              %31 = arith.mulf %in_7, %30 : f32
              %32 = arith.cmpf ogt, %in, %31 : f32
              %33 = arith.select %32, %in, %31 : f32
              linalg.yield %33 : f32
            }
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%17, %21 : memref<32x32x32xf32>, memref<32x32xf32>) outs(%17 : memref<32x32x32xf32>) {
            ^bb0(%in: f32, %in_7: f32, %out: f32):
              %30 = arith.truncf %cst_0 : f64 to f32
              %31 = arith.mulf %in, %30 : f32
              %32 = arith.subf %31, %in_7 : f32
              %33 = math.powf %cst, %32 : f32
              linalg.yield %33 : f32
            }
            linalg.fill ins(%cst_1 : f32) outs(%22 : memref<32x32xf32>)
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%17 : memref<32x32x32xf32>) outs(%22 : memref<32x32xf32>) {
            ^bb0(%in: f32, %out: f32):
              %30 = arith.addf %in, %out : f32
              linalg.yield %30 : f32
            }
            linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%20, %21 : memref<32x32xf32>, memref<32x32xf32>) outs(%23 : memref<32x32xf32>) {
            ^bb0(%in: f32, %in_7: f32, %out: f32):
              %30 = arith.subf %in, %in_7 : f32
              %31 = math.powf %cst, %30 : f32
              linalg.yield %31 : f32
            }
            linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%19, %23, %22 : memref<32x32xf32>, memref<32x32xf32>, memref<32x32xf32>) outs(%19 : memref<32x32xf32>) {
            ^bb0(%in: f32, %in_7: f32, %in_8: f32, %out: f32):
              %30 = arith.mulf %in, %in_7 : f32
              %31 = arith.addf %30, %in_8 : f32
              linalg.yield %31 : f32
            }
            %29 = arith.muli %arg7, %c4096 : index
            %reinterpret_cast_6 = memref.reinterpret_cast %arg1 to offset: [%29], sizes: [32, 32, 128], strides: [524288, 128, 1] : memref<32x4096x128xf32> to memref<32x32x128xf32, strided<[524288, 128, 1], offset: ?>>
            loom.copy %reinterpret_cast_6, %16 src_mem_space @DRAM dst_mem_space @L1, interconnect : [@vertical_links], broadcast : [8, 1] : memref<32x32x128xf32, strided<[524288, 128, 1], offset: ?>>, memref<32x32x128xf32>
            linalg.fill ins(%cst_1 : f32) outs(%26 : memref<32x32x128xf32>)
            linalg.batch_matmul ins(%17, %16 : memref<32x32x32xf32>, memref<32x32x128xf32>) outs(%26 : memref<32x32x128xf32>)
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%26, %25, %23 : memref<32x32x128xf32>, memref<32x32x128xf32>, memref<32x32xf32>) outs(%25 : memref<32x32x128xf32>) {
            ^bb0(%in: f32, %in_7: f32, %in_8: f32, %out: f32):
              %30 = arith.mulf %in_7, %in_8 : f32
              %31 = arith.addf %in, %30 : f32
              linalg.yield %31 : f32
            }
            linalg.copy ins(%21 : memref<32x32xf32>) outs(%20 : memref<32x32xf32>)
          }
          linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%25, %19 : memref<32x32x128xf32>, memref<32x32xf32>) outs(%24 : memref<32x32x128xf32>) {
          ^bb0(%in: f32, %in_5: f32, %out: f32):
            %28 = arith.divf %in, %in_5 : f32
            linalg.yield %28 : f32
          }
          %reinterpret_cast_4 = memref.reinterpret_cast %arg3 to offset: [%27], sizes: [32, 32, 128], strides: [524288, 128, 1] : memref<32x4096x128xf32> to memref<32x32x128xf32, strided<[524288, 128, 1], offset: ?>>
          loom.copy %24, %reinterpret_cast_4 src_mem_space @L1 dst_mem_space @DRAM, interconnect : [], broadcast : [1, 1] : memref<32x32x128xf32>, memref<32x32x128xf32, strided<[524288, 128, 1], offset: ?>>
        }
        scf.reduce 
      } {loom.iter_types = [#loom.iter_type<spatial>, #loom.iter_type<spatial>], loom.mapped_to_dims = [@x, @y]}
      return
    }
  }
  module attributes {loom.pass_name = "Materialize"} {
    func.func @attention__d0i1_d1i1__f10__d_h_d__BB32__BM32__BN32(%arg0: memref<32x128x4096xf32>, %arg1: memref<32x4096x128xf32>, %arg2: memref<32x4096x128xf32>, %arg3: memref<32x4096x128xf32>) {
      %c4096 = arith.constant 4096 : index
      %c128 = arith.constant 128 : index
      %c64 = arith.constant 64 : index
      %c2 = arith.constant 2 : index
      %cst = arith.constant 2.000000e+00 : f32
      %cst_0 = arith.constant 0.12751743074602467 : f64
      %cst_1 = arith.constant 0.000000e+00 : f32
      %cst_2 = arith.constant 1.000000e+00 : f32
      %cst_3 = arith.constant 0xFF800000 : f32
      %c32 = arith.constant 32 : index
      %c0 = arith.constant 0 : index
      %c8 = arith.constant 8 : index
      %c1 = arith.constant 1 : index
      scf.parallel (%arg4, %arg5) = (%c0, %c0) to (%c8, %c8) step (%c1, %c1) {
        scf.for %arg6 = %c0 to %c2 step %c1 {
          %12 = arith.muli %arg4, %c8 overflow<nsw> : index
          %13 = arith.addi %12, %arg5 : index
          %14 = arith.muli %arg6, %c64 overflow<nsw> : index
          %15 = arith.addi %13, %14 : index
          %16 = loom.alloc [32, 32, 128] on @L1 : memref<32x32x128xf32>
          %17 = loom.alloc [32, 32, 32] on @L1 : memref<32x32x32xf32>
          %18 = loom.alloc [32, 128, 32] on @L1 : memref<32x128x32xf32>
          %19 = loom.alloc [32, 32] on @L1 : memref<32x32xf32>
          %20 = loom.alloc [32, 32] on @L1 : memref<32x32xf32>
          %21 = loom.alloc [32, 32] on @L1 : memref<32x32xf32>
          %22 = loom.alloc [32, 32] on @L1 : memref<32x32xf32>
          %23 = loom.alloc [32, 32] on @L1 : memref<32x32xf32>
          %24 = loom.alloc [32, 32, 128] on @L1 : memref<32x32x128xf32>
          %25 = loom.alloc [32, 32, 128] on @L1 : memref<32x32x128xf32>
          %26 = loom.alloc [32, 32, 128] on @L1 : memref<32x32x128xf32>
          %27 = arith.muli %15, %c4096 : index
          %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%27], sizes: [32, 32, 128], strides: [524288, 128, 1] : memref<32x4096x128xf32> to memref<32x32x128xf32, strided<[524288, 128, 1], offset: ?>>
          loom.copy %reinterpret_cast, %24 src_mem_space @DRAM dst_mem_space @L1, interconnect : [], broadcast : [1, 1] : memref<32x32x128xf32, strided<[524288, 128, 1], offset: ?>>, memref<32x32x128xf32>
          linalg.fill ins(%cst_1 : f32) outs(%25 : memref<32x32x128xf32>)
          linalg.fill ins(%cst_2 : f32) outs(%19 : memref<32x32xf32>)
          linalg.fill ins(%cst_3 : f32) outs(%20 : memref<32x32xf32>)
          scf.for %arg7 = %c0 to %c128 step %c1 {
            %28 = arith.muli %arg7, %c32 : index
            %reinterpret_cast_5 = memref.reinterpret_cast %arg0 to offset: [%28], sizes: [32, 128, 32], strides: [524288, 4096, 1] : memref<32x128x4096xf32> to memref<32x128x32xf32, strided<[524288, 4096, 1], offset: ?>>
            loom.copy %reinterpret_cast_5, %18 src_mem_space @DRAM dst_mem_space @L1, interconnect : [@horizontal_links], broadcast : [1, 8] : memref<32x128x32xf32, strided<[524288, 4096, 1], offset: ?>>, memref<32x128x32xf32>
            linalg.fill ins(%cst_1 : f32) outs(%17 : memref<32x32x32xf32>)
            linalg.batch_matmul ins(%24, %18 : memref<32x32x128xf32>, memref<32x128x32xf32>) outs(%17 : memref<32x32x32xf32>)
            linalg.fill ins(%cst_3 : f32) outs(%21 : memref<32x32xf32>)
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%17 : memref<32x32x32xf32>) outs(%21 : memref<32x32xf32>) {
            ^bb0(%in: f32, %out: f32):
              %30 = arith.maximumf %in, %out : f32
              linalg.yield %30 : f32
            }
            linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%20, %21 : memref<32x32xf32>, memref<32x32xf32>) outs(%21 : memref<32x32xf32>) {
            ^bb0(%in: f32, %in_7: f32, %out: f32):
              %30 = arith.truncf %cst_0 : f64 to f32
              %31 = arith.mulf %in_7, %30 : f32
              %32 = arith.cmpf ogt, %in, %31 : f32
              %33 = arith.select %32, %in, %31 : f32
              linalg.yield %33 : f32
            }
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%17, %21 : memref<32x32x32xf32>, memref<32x32xf32>) outs(%17 : memref<32x32x32xf32>) {
            ^bb0(%in: f32, %in_7: f32, %out: f32):
              %30 = arith.truncf %cst_0 : f64 to f32
              %31 = arith.mulf %in, %30 : f32
              %32 = arith.subf %31, %in_7 : f32
              %33 = math.powf %cst, %32 : f32
              linalg.yield %33 : f32
            }
            linalg.fill ins(%cst_1 : f32) outs(%22 : memref<32x32xf32>)
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%17 : memref<32x32x32xf32>) outs(%22 : memref<32x32xf32>) {
            ^bb0(%in: f32, %out: f32):
              %30 = arith.addf %in, %out : f32
              linalg.yield %30 : f32
            }
            linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%20, %21 : memref<32x32xf32>, memref<32x32xf32>) outs(%23 : memref<32x32xf32>) {
            ^bb0(%in: f32, %in_7: f32, %out: f32):
              %30 = arith.subf %in, %in_7 : f32
              %31 = math.powf %cst, %30 : f32
              linalg.yield %31 : f32
            }
            linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%19, %23, %22 : memref<32x32xf32>, memref<32x32xf32>, memref<32x32xf32>) outs(%19 : memref<32x32xf32>) {
            ^bb0(%in: f32, %in_7: f32, %in_8: f32, %out: f32):
              %30 = arith.mulf %in, %in_7 : f32
              %31 = arith.addf %30, %in_8 : f32
              linalg.yield %31 : f32
            }
            %29 = arith.muli %arg7, %c4096 : index
            %reinterpret_cast_6 = memref.reinterpret_cast %arg1 to offset: [%29], sizes: [32, 32, 128], strides: [524288, 128, 1] : memref<32x4096x128xf32> to memref<32x32x128xf32, strided<[524288, 128, 1], offset: ?>>
            loom.copy %reinterpret_cast_6, %16 src_mem_space @DRAM dst_mem_space @L1, interconnect : [], broadcast : [1, 1] : memref<32x32x128xf32, strided<[524288, 128, 1], offset: ?>>, memref<32x32x128xf32>
            linalg.fill ins(%cst_1 : f32) outs(%26 : memref<32x32x128xf32>)
            linalg.batch_matmul ins(%17, %16 : memref<32x32x32xf32>, memref<32x32x128xf32>) outs(%26 : memref<32x32x128xf32>)
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%26, %25, %23 : memref<32x32x128xf32>, memref<32x32x128xf32>, memref<32x32xf32>) outs(%25 : memref<32x32x128xf32>) {
            ^bb0(%in: f32, %in_7: f32, %in_8: f32, %out: f32):
              %30 = arith.mulf %in_7, %in_8 : f32
              %31 = arith.addf %in, %30 : f32
              linalg.yield %31 : f32
            }
            linalg.copy ins(%21 : memref<32x32xf32>) outs(%20 : memref<32x32xf32>)
          }
          linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%25, %19 : memref<32x32x128xf32>, memref<32x32xf32>) outs(%24 : memref<32x32x128xf32>) {
          ^bb0(%in: f32, %in_5: f32, %out: f32):
            %28 = arith.divf %in, %in_5 : f32
            linalg.yield %28 : f32
          }
          %reinterpret_cast_4 = memref.reinterpret_cast %arg3 to offset: [%27], sizes: [32, 32, 128], strides: [524288, 128, 1] : memref<32x4096x128xf32> to memref<32x32x128xf32, strided<[524288, 128, 1], offset: ?>>
          loom.copy %24, %reinterpret_cast_4 src_mem_space @L1 dst_mem_space @DRAM, interconnect : [], broadcast : [1, 1] : memref<32x32x128xf32>, memref<32x32x128xf32, strided<[524288, 128, 1], offset: ?>>
        }
        scf.reduce 
      } {loom.iter_types = [#loom.iter_type<spatial>, #loom.iter_type<spatial>], loom.mapped_to_dims = [@x, @y]}
      return
    }
  }
  module attributes {loom.pass_name = "Materialize"} {
    func.func @attention__d0i1_d1i1__f10__d_h_a__BB32__BM32__BN32(%arg0: memref<32x128x4096xf32>, %arg1: memref<32x4096x128xf32>, %arg2: memref<32x4096x128xf32>, %arg3: memref<32x4096x128xf32>) {
      %c4096 = arith.constant 4096 : index
      %c128 = arith.constant 128 : index
      %c64 = arith.constant 64 : index
      %c2 = arith.constant 2 : index
      %cst = arith.constant 2.000000e+00 : f32
      %cst_0 = arith.constant 0.12751743074602467 : f64
      %cst_1 = arith.constant 0.000000e+00 : f32
      %cst_2 = arith.constant 1.000000e+00 : f32
      %cst_3 = arith.constant 0xFF800000 : f32
      %c32 = arith.constant 32 : index
      %c0 = arith.constant 0 : index
      %c8 = arith.constant 8 : index
      %c1 = arith.constant 1 : index
      scf.parallel (%arg4, %arg5) = (%c0, %c0) to (%c8, %c8) step (%c1, %c1) {
        scf.for %arg6 = %c0 to %c2 step %c1 {
          %12 = arith.muli %arg4, %c8 overflow<nsw> : index
          %13 = arith.addi %12, %arg5 : index
          %14 = arith.muli %arg6, %c64 overflow<nsw> : index
          %15 = arith.addi %13, %14 : index
          %16 = loom.alloc [32, 32, 128] on @L1 : memref<32x32x128xf32>
          %17 = loom.alloc [32, 32, 32] on @L1 : memref<32x32x32xf32>
          %18 = loom.alloc [32, 128, 32] on @L1 : memref<32x128x32xf32>
          %19 = loom.alloc [32, 32] on @L1 : memref<32x32xf32>
          %20 = loom.alloc [32, 32] on @L1 : memref<32x32xf32>
          %21 = loom.alloc [32, 32] on @L1 : memref<32x32xf32>
          %22 = loom.alloc [32, 32] on @L1 : memref<32x32xf32>
          %23 = loom.alloc [32, 32] on @L1 : memref<32x32xf32>
          %24 = loom.alloc [32, 32, 128] on @L1 : memref<32x32x128xf32>
          %25 = loom.alloc [32, 32, 128] on @L1 : memref<32x32x128xf32>
          %26 = loom.alloc [32, 32, 128] on @L1 : memref<32x32x128xf32>
          %27 = arith.muli %15, %c4096 : index
          %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%27], sizes: [32, 32, 128], strides: [524288, 128, 1] : memref<32x4096x128xf32> to memref<32x32x128xf32, strided<[524288, 128, 1], offset: ?>>
          loom.copy %reinterpret_cast, %24 src_mem_space @DRAM dst_mem_space @L1, interconnect : [], broadcast : [1, 1] : memref<32x32x128xf32, strided<[524288, 128, 1], offset: ?>>, memref<32x32x128xf32>
          linalg.fill ins(%cst_1 : f32) outs(%25 : memref<32x32x128xf32>)
          linalg.fill ins(%cst_2 : f32) outs(%19 : memref<32x32xf32>)
          linalg.fill ins(%cst_3 : f32) outs(%20 : memref<32x32xf32>)
          scf.for %arg7 = %c0 to %c128 step %c1 {
            %28 = arith.muli %arg7, %c32 : index
            %reinterpret_cast_5 = memref.reinterpret_cast %arg0 to offset: [%28], sizes: [32, 128, 32], strides: [524288, 4096, 1] : memref<32x128x4096xf32> to memref<32x128x32xf32, strided<[524288, 4096, 1], offset: ?>>
            loom.copy %reinterpret_cast_5, %18 src_mem_space @DRAM dst_mem_space @L1, interconnect : [@horizontal_links], broadcast : [1, 8] : memref<32x128x32xf32, strided<[524288, 4096, 1], offset: ?>>, memref<32x128x32xf32>
            linalg.fill ins(%cst_1 : f32) outs(%17 : memref<32x32x32xf32>)
            linalg.batch_matmul ins(%24, %18 : memref<32x32x128xf32>, memref<32x128x32xf32>) outs(%17 : memref<32x32x32xf32>)
            linalg.fill ins(%cst_3 : f32) outs(%21 : memref<32x32xf32>)
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%17 : memref<32x32x32xf32>) outs(%21 : memref<32x32xf32>) {
            ^bb0(%in: f32, %out: f32):
              %30 = arith.maximumf %in, %out : f32
              linalg.yield %30 : f32
            }
            linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%20, %21 : memref<32x32xf32>, memref<32x32xf32>) outs(%21 : memref<32x32xf32>) {
            ^bb0(%in: f32, %in_7: f32, %out: f32):
              %30 = arith.truncf %cst_0 : f64 to f32
              %31 = arith.mulf %in_7, %30 : f32
              %32 = arith.cmpf ogt, %in, %31 : f32
              %33 = arith.select %32, %in, %31 : f32
              linalg.yield %33 : f32
            }
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%17, %21 : memref<32x32x32xf32>, memref<32x32xf32>) outs(%17 : memref<32x32x32xf32>) {
            ^bb0(%in: f32, %in_7: f32, %out: f32):
              %30 = arith.truncf %cst_0 : f64 to f32
              %31 = arith.mulf %in, %30 : f32
              %32 = arith.subf %31, %in_7 : f32
              %33 = math.powf %cst, %32 : f32
              linalg.yield %33 : f32
            }
            linalg.fill ins(%cst_1 : f32) outs(%22 : memref<32x32xf32>)
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%17 : memref<32x32x32xf32>) outs(%22 : memref<32x32xf32>) {
            ^bb0(%in: f32, %out: f32):
              %30 = arith.addf %in, %out : f32
              linalg.yield %30 : f32
            }
            linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%20, %21 : memref<32x32xf32>, memref<32x32xf32>) outs(%23 : memref<32x32xf32>) {
            ^bb0(%in: f32, %in_7: f32, %out: f32):
              %30 = arith.subf %in, %in_7 : f32
              %31 = math.powf %cst, %30 : f32
              linalg.yield %31 : f32
            }
            linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%19, %23, %22 : memref<32x32xf32>, memref<32x32xf32>, memref<32x32xf32>) outs(%19 : memref<32x32xf32>) {
            ^bb0(%in: f32, %in_7: f32, %in_8: f32, %out: f32):
              %30 = arith.mulf %in, %in_7 : f32
              %31 = arith.addf %30, %in_8 : f32
              linalg.yield %31 : f32
            }
            %29 = arith.muli %arg7, %c4096 : index
            %reinterpret_cast_6 = memref.reinterpret_cast %arg1 to offset: [%29], sizes: [32, 32, 128], strides: [524288, 128, 1] : memref<32x4096x128xf32> to memref<32x32x128xf32, strided<[524288, 128, 1], offset: ?>>
            loom.copy %reinterpret_cast_6, %16 src_mem_space @DRAM dst_mem_space @L1, interconnect : [@horizontal_links, @vertical_links], broadcast : [8, 8] : memref<32x32x128xf32, strided<[524288, 128, 1], offset: ?>>, memref<32x32x128xf32>
            linalg.fill ins(%cst_1 : f32) outs(%26 : memref<32x32x128xf32>)
            linalg.batch_matmul ins(%17, %16 : memref<32x32x32xf32>, memref<32x32x128xf32>) outs(%26 : memref<32x32x128xf32>)
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%26, %25, %23 : memref<32x32x128xf32>, memref<32x32x128xf32>, memref<32x32xf32>) outs(%25 : memref<32x32x128xf32>) {
            ^bb0(%in: f32, %in_7: f32, %in_8: f32, %out: f32):
              %30 = arith.mulf %in_7, %in_8 : f32
              %31 = arith.addf %in, %30 : f32
              linalg.yield %31 : f32
            }
            linalg.copy ins(%21 : memref<32x32xf32>) outs(%20 : memref<32x32xf32>)
          }
          linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%25, %19 : memref<32x32x128xf32>, memref<32x32xf32>) outs(%24 : memref<32x32x128xf32>) {
          ^bb0(%in: f32, %in_5: f32, %out: f32):
            %28 = arith.divf %in, %in_5 : f32
            linalg.yield %28 : f32
          }
          %reinterpret_cast_4 = memref.reinterpret_cast %arg3 to offset: [%27], sizes: [32, 32, 128], strides: [524288, 128, 1] : memref<32x4096x128xf32> to memref<32x32x128xf32, strided<[524288, 128, 1], offset: ?>>
          loom.copy %24, %reinterpret_cast_4 src_mem_space @L1 dst_mem_space @DRAM, interconnect : [], broadcast : [1, 1] : memref<32x32x128xf32>, memref<32x32x128xf32, strided<[524288, 128, 1], offset: ?>>
        }
        scf.reduce 
      } {loom.iter_types = [#loom.iter_type<spatial>, #loom.iter_type<spatial>], loom.mapped_to_dims = [@x, @y]}
      return
    }
  }
  module attributes {loom.pass_name = "Materialize"} {
    func.func @attention__d0i1_d1i1__f10__d_h_h__BB32__BM32__BN32(%arg0: memref<32x128x4096xf32>, %arg1: memref<32x4096x128xf32>, %arg2: memref<32x4096x128xf32>, %arg3: memref<32x4096x128xf32>) {
      %c4096 = arith.constant 4096 : index
      %c128 = arith.constant 128 : index
      %c64 = arith.constant 64 : index
      %c2 = arith.constant 2 : index
      %cst = arith.constant 2.000000e+00 : f32
      %cst_0 = arith.constant 0.12751743074602467 : f64
      %cst_1 = arith.constant 0.000000e+00 : f32
      %cst_2 = arith.constant 1.000000e+00 : f32
      %cst_3 = arith.constant 0xFF800000 : f32
      %c32 = arith.constant 32 : index
      %c0 = arith.constant 0 : index
      %c8 = arith.constant 8 : index
      %c1 = arith.constant 1 : index
      scf.parallel (%arg4, %arg5) = (%c0, %c0) to (%c8, %c8) step (%c1, %c1) {
        scf.for %arg6 = %c0 to %c2 step %c1 {
          %12 = arith.muli %arg4, %c8 overflow<nsw> : index
          %13 = arith.addi %12, %arg5 : index
          %14 = arith.muli %arg6, %c64 overflow<nsw> : index
          %15 = arith.addi %13, %14 : index
          %16 = loom.alloc [32, 32, 128] on @L1 : memref<32x32x128xf32>
          %17 = loom.alloc [32, 32, 32] on @L1 : memref<32x32x32xf32>
          %18 = loom.alloc [32, 128, 32] on @L1 : memref<32x128x32xf32>
          %19 = loom.alloc [32, 32] on @L1 : memref<32x32xf32>
          %20 = loom.alloc [32, 32] on @L1 : memref<32x32xf32>
          %21 = loom.alloc [32, 32] on @L1 : memref<32x32xf32>
          %22 = loom.alloc [32, 32] on @L1 : memref<32x32xf32>
          %23 = loom.alloc [32, 32] on @L1 : memref<32x32xf32>
          %24 = loom.alloc [32, 32, 128] on @L1 : memref<32x32x128xf32>
          %25 = loom.alloc [32, 32, 128] on @L1 : memref<32x32x128xf32>
          %26 = loom.alloc [32, 32, 128] on @L1 : memref<32x32x128xf32>
          %27 = arith.muli %15, %c4096 : index
          %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%27], sizes: [32, 32, 128], strides: [524288, 128, 1] : memref<32x4096x128xf32> to memref<32x32x128xf32, strided<[524288, 128, 1], offset: ?>>
          loom.copy %reinterpret_cast, %24 src_mem_space @DRAM dst_mem_space @L1, interconnect : [], broadcast : [1, 1] : memref<32x32x128xf32, strided<[524288, 128, 1], offset: ?>>, memref<32x32x128xf32>
          linalg.fill ins(%cst_1 : f32) outs(%25 : memref<32x32x128xf32>)
          linalg.fill ins(%cst_2 : f32) outs(%19 : memref<32x32xf32>)
          linalg.fill ins(%cst_3 : f32) outs(%20 : memref<32x32xf32>)
          scf.for %arg7 = %c0 to %c128 step %c1 {
            %28 = arith.muli %arg7, %c32 : index
            %reinterpret_cast_5 = memref.reinterpret_cast %arg0 to offset: [%28], sizes: [32, 128, 32], strides: [524288, 4096, 1] : memref<32x128x4096xf32> to memref<32x128x32xf32, strided<[524288, 4096, 1], offset: ?>>
            loom.copy %reinterpret_cast_5, %18 src_mem_space @DRAM dst_mem_space @L1, interconnect : [@horizontal_links], broadcast : [1, 8] : memref<32x128x32xf32, strided<[524288, 4096, 1], offset: ?>>, memref<32x128x32xf32>
            linalg.fill ins(%cst_1 : f32) outs(%17 : memref<32x32x32xf32>)
            linalg.batch_matmul ins(%24, %18 : memref<32x32x128xf32>, memref<32x128x32xf32>) outs(%17 : memref<32x32x32xf32>)
            linalg.fill ins(%cst_3 : f32) outs(%21 : memref<32x32xf32>)
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%17 : memref<32x32x32xf32>) outs(%21 : memref<32x32xf32>) {
            ^bb0(%in: f32, %out: f32):
              %30 = arith.maximumf %in, %out : f32
              linalg.yield %30 : f32
            }
            linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%20, %21 : memref<32x32xf32>, memref<32x32xf32>) outs(%21 : memref<32x32xf32>) {
            ^bb0(%in: f32, %in_7: f32, %out: f32):
              %30 = arith.truncf %cst_0 : f64 to f32
              %31 = arith.mulf %in_7, %30 : f32
              %32 = arith.cmpf ogt, %in, %31 : f32
              %33 = arith.select %32, %in, %31 : f32
              linalg.yield %33 : f32
            }
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%17, %21 : memref<32x32x32xf32>, memref<32x32xf32>) outs(%17 : memref<32x32x32xf32>) {
            ^bb0(%in: f32, %in_7: f32, %out: f32):
              %30 = arith.truncf %cst_0 : f64 to f32
              %31 = arith.mulf %in, %30 : f32
              %32 = arith.subf %31, %in_7 : f32
              %33 = math.powf %cst, %32 : f32
              linalg.yield %33 : f32
            }
            linalg.fill ins(%cst_1 : f32) outs(%22 : memref<32x32xf32>)
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%17 : memref<32x32x32xf32>) outs(%22 : memref<32x32xf32>) {
            ^bb0(%in: f32, %out: f32):
              %30 = arith.addf %in, %out : f32
              linalg.yield %30 : f32
            }
            linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%20, %21 : memref<32x32xf32>, memref<32x32xf32>) outs(%23 : memref<32x32xf32>) {
            ^bb0(%in: f32, %in_7: f32, %out: f32):
              %30 = arith.subf %in, %in_7 : f32
              %31 = math.powf %cst, %30 : f32
              linalg.yield %31 : f32
            }
            linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%19, %23, %22 : memref<32x32xf32>, memref<32x32xf32>, memref<32x32xf32>) outs(%19 : memref<32x32xf32>) {
            ^bb0(%in: f32, %in_7: f32, %in_8: f32, %out: f32):
              %30 = arith.mulf %in, %in_7 : f32
              %31 = arith.addf %30, %in_8 : f32
              linalg.yield %31 : f32
            }
            %29 = arith.muli %arg7, %c4096 : index
            %reinterpret_cast_6 = memref.reinterpret_cast %arg1 to offset: [%29], sizes: [32, 32, 128], strides: [524288, 128, 1] : memref<32x4096x128xf32> to memref<32x32x128xf32, strided<[524288, 128, 1], offset: ?>>
            loom.copy %reinterpret_cast_6, %16 src_mem_space @DRAM dst_mem_space @L1, interconnect : [@horizontal_links], broadcast : [1, 8] : memref<32x32x128xf32, strided<[524288, 128, 1], offset: ?>>, memref<32x32x128xf32>
            linalg.fill ins(%cst_1 : f32) outs(%26 : memref<32x32x128xf32>)
            linalg.batch_matmul ins(%17, %16 : memref<32x32x32xf32>, memref<32x32x128xf32>) outs(%26 : memref<32x32x128xf32>)
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%26, %25, %23 : memref<32x32x128xf32>, memref<32x32x128xf32>, memref<32x32xf32>) outs(%25 : memref<32x32x128xf32>) {
            ^bb0(%in: f32, %in_7: f32, %in_8: f32, %out: f32):
              %30 = arith.mulf %in_7, %in_8 : f32
              %31 = arith.addf %in, %30 : f32
              linalg.yield %31 : f32
            }
            linalg.copy ins(%21 : memref<32x32xf32>) outs(%20 : memref<32x32xf32>)
          }
          linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%25, %19 : memref<32x32x128xf32>, memref<32x32xf32>) outs(%24 : memref<32x32x128xf32>) {
          ^bb0(%in: f32, %in_5: f32, %out: f32):
            %28 = arith.divf %in, %in_5 : f32
            linalg.yield %28 : f32
          }
          %reinterpret_cast_4 = memref.reinterpret_cast %arg3 to offset: [%27], sizes: [32, 32, 128], strides: [524288, 128, 1] : memref<32x4096x128xf32> to memref<32x32x128xf32, strided<[524288, 128, 1], offset: ?>>
          loom.copy %24, %reinterpret_cast_4 src_mem_space @L1 dst_mem_space @DRAM, interconnect : [], broadcast : [1, 1] : memref<32x32x128xf32>, memref<32x32x128xf32, strided<[524288, 128, 1], offset: ?>>
        }
        scf.reduce 
      } {loom.iter_types = [#loom.iter_type<spatial>, #loom.iter_type<spatial>], loom.mapped_to_dims = [@x, @y]}
      return
    }
  }
  module attributes {loom.pass_name = "Materialize"} {
    func.func @attention__d0i1_d1i1__f10__d_h_v__BB32__BM32__BN32(%arg0: memref<32x128x4096xf32>, %arg1: memref<32x4096x128xf32>, %arg2: memref<32x4096x128xf32>, %arg3: memref<32x4096x128xf32>) {
      %c4096 = arith.constant 4096 : index
      %c128 = arith.constant 128 : index
      %c64 = arith.constant 64 : index
      %c2 = arith.constant 2 : index
      %cst = arith.constant 2.000000e+00 : f32
      %cst_0 = arith.constant 0.12751743074602467 : f64
      %cst_1 = arith.constant 0.000000e+00 : f32
      %cst_2 = arith.constant 1.000000e+00 : f32
      %cst_3 = arith.constant 0xFF800000 : f32
      %c32 = arith.constant 32 : index
      %c0 = arith.constant 0 : index
      %c8 = arith.constant 8 : index
      %c1 = arith.constant 1 : index
      scf.parallel (%arg4, %arg5) = (%c0, %c0) to (%c8, %c8) step (%c1, %c1) {
        scf.for %arg6 = %c0 to %c2 step %c1 {
          %12 = arith.muli %arg4, %c8 overflow<nsw> : index
          %13 = arith.addi %12, %arg5 : index
          %14 = arith.muli %arg6, %c64 overflow<nsw> : index
          %15 = arith.addi %13, %14 : index
          %16 = loom.alloc [32, 32, 128] on @L1 : memref<32x32x128xf32>
          %17 = loom.alloc [32, 32, 32] on @L1 : memref<32x32x32xf32>
          %18 = loom.alloc [32, 128, 32] on @L1 : memref<32x128x32xf32>
          %19 = loom.alloc [32, 32] on @L1 : memref<32x32xf32>
          %20 = loom.alloc [32, 32] on @L1 : memref<32x32xf32>
          %21 = loom.alloc [32, 32] on @L1 : memref<32x32xf32>
          %22 = loom.alloc [32, 32] on @L1 : memref<32x32xf32>
          %23 = loom.alloc [32, 32] on @L1 : memref<32x32xf32>
          %24 = loom.alloc [32, 32, 128] on @L1 : memref<32x32x128xf32>
          %25 = loom.alloc [32, 32, 128] on @L1 : memref<32x32x128xf32>
          %26 = loom.alloc [32, 32, 128] on @L1 : memref<32x32x128xf32>
          %27 = arith.muli %15, %c4096 : index
          %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%27], sizes: [32, 32, 128], strides: [524288, 128, 1] : memref<32x4096x128xf32> to memref<32x32x128xf32, strided<[524288, 128, 1], offset: ?>>
          loom.copy %reinterpret_cast, %24 src_mem_space @DRAM dst_mem_space @L1, interconnect : [], broadcast : [1, 1] : memref<32x32x128xf32, strided<[524288, 128, 1], offset: ?>>, memref<32x32x128xf32>
          linalg.fill ins(%cst_1 : f32) outs(%25 : memref<32x32x128xf32>)
          linalg.fill ins(%cst_2 : f32) outs(%19 : memref<32x32xf32>)
          linalg.fill ins(%cst_3 : f32) outs(%20 : memref<32x32xf32>)
          scf.for %arg7 = %c0 to %c128 step %c1 {
            %28 = arith.muli %arg7, %c32 : index
            %reinterpret_cast_5 = memref.reinterpret_cast %arg0 to offset: [%28], sizes: [32, 128, 32], strides: [524288, 4096, 1] : memref<32x128x4096xf32> to memref<32x128x32xf32, strided<[524288, 4096, 1], offset: ?>>
            loom.copy %reinterpret_cast_5, %18 src_mem_space @DRAM dst_mem_space @L1, interconnect : [@horizontal_links], broadcast : [1, 8] : memref<32x128x32xf32, strided<[524288, 4096, 1], offset: ?>>, memref<32x128x32xf32>
            linalg.fill ins(%cst_1 : f32) outs(%17 : memref<32x32x32xf32>)
            linalg.batch_matmul ins(%24, %18 : memref<32x32x128xf32>, memref<32x128x32xf32>) outs(%17 : memref<32x32x32xf32>)
            linalg.fill ins(%cst_3 : f32) outs(%21 : memref<32x32xf32>)
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%17 : memref<32x32x32xf32>) outs(%21 : memref<32x32xf32>) {
            ^bb0(%in: f32, %out: f32):
              %30 = arith.maximumf %in, %out : f32
              linalg.yield %30 : f32
            }
            linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%20, %21 : memref<32x32xf32>, memref<32x32xf32>) outs(%21 : memref<32x32xf32>) {
            ^bb0(%in: f32, %in_7: f32, %out: f32):
              %30 = arith.truncf %cst_0 : f64 to f32
              %31 = arith.mulf %in_7, %30 : f32
              %32 = arith.cmpf ogt, %in, %31 : f32
              %33 = arith.select %32, %in, %31 : f32
              linalg.yield %33 : f32
            }
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%17, %21 : memref<32x32x32xf32>, memref<32x32xf32>) outs(%17 : memref<32x32x32xf32>) {
            ^bb0(%in: f32, %in_7: f32, %out: f32):
              %30 = arith.truncf %cst_0 : f64 to f32
              %31 = arith.mulf %in, %30 : f32
              %32 = arith.subf %31, %in_7 : f32
              %33 = math.powf %cst, %32 : f32
              linalg.yield %33 : f32
            }
            linalg.fill ins(%cst_1 : f32) outs(%22 : memref<32x32xf32>)
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%17 : memref<32x32x32xf32>) outs(%22 : memref<32x32xf32>) {
            ^bb0(%in: f32, %out: f32):
              %30 = arith.addf %in, %out : f32
              linalg.yield %30 : f32
            }
            linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%20, %21 : memref<32x32xf32>, memref<32x32xf32>) outs(%23 : memref<32x32xf32>) {
            ^bb0(%in: f32, %in_7: f32, %out: f32):
              %30 = arith.subf %in, %in_7 : f32
              %31 = math.powf %cst, %30 : f32
              linalg.yield %31 : f32
            }
            linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%19, %23, %22 : memref<32x32xf32>, memref<32x32xf32>, memref<32x32xf32>) outs(%19 : memref<32x32xf32>) {
            ^bb0(%in: f32, %in_7: f32, %in_8: f32, %out: f32):
              %30 = arith.mulf %in, %in_7 : f32
              %31 = arith.addf %30, %in_8 : f32
              linalg.yield %31 : f32
            }
            %29 = arith.muli %arg7, %c4096 : index
            %reinterpret_cast_6 = memref.reinterpret_cast %arg1 to offset: [%29], sizes: [32, 32, 128], strides: [524288, 128, 1] : memref<32x4096x128xf32> to memref<32x32x128xf32, strided<[524288, 128, 1], offset: ?>>
            loom.copy %reinterpret_cast_6, %16 src_mem_space @DRAM dst_mem_space @L1, interconnect : [@vertical_links], broadcast : [8, 1] : memref<32x32x128xf32, strided<[524288, 128, 1], offset: ?>>, memref<32x32x128xf32>
            linalg.fill ins(%cst_1 : f32) outs(%26 : memref<32x32x128xf32>)
            linalg.batch_matmul ins(%17, %16 : memref<32x32x32xf32>, memref<32x32x128xf32>) outs(%26 : memref<32x32x128xf32>)
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%26, %25, %23 : memref<32x32x128xf32>, memref<32x32x128xf32>, memref<32x32xf32>) outs(%25 : memref<32x32x128xf32>) {
            ^bb0(%in: f32, %in_7: f32, %in_8: f32, %out: f32):
              %30 = arith.mulf %in_7, %in_8 : f32
              %31 = arith.addf %in, %30 : f32
              linalg.yield %31 : f32
            }
            linalg.copy ins(%21 : memref<32x32xf32>) outs(%20 : memref<32x32xf32>)
          }
          linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%25, %19 : memref<32x32x128xf32>, memref<32x32xf32>) outs(%24 : memref<32x32x128xf32>) {
          ^bb0(%in: f32, %in_5: f32, %out: f32):
            %28 = arith.divf %in, %in_5 : f32
            linalg.yield %28 : f32
          }
          %reinterpret_cast_4 = memref.reinterpret_cast %arg3 to offset: [%27], sizes: [32, 32, 128], strides: [524288, 128, 1] : memref<32x4096x128xf32> to memref<32x32x128xf32, strided<[524288, 128, 1], offset: ?>>
          loom.copy %24, %reinterpret_cast_4 src_mem_space @L1 dst_mem_space @DRAM, interconnect : [], broadcast : [1, 1] : memref<32x32x128xf32>, memref<32x32x128xf32, strided<[524288, 128, 1], offset: ?>>
        }
        scf.reduce 
      } {loom.iter_types = [#loom.iter_type<spatial>, #loom.iter_type<spatial>], loom.mapped_to_dims = [@x, @y]}
      return
    }
  }
  module attributes {loom.pass_name = "Materialize"} {
    func.func @attention__d0i1_d1i1__f10__d_v_d__BB32__BM32__BN32(%arg0: memref<32x128x4096xf32>, %arg1: memref<32x4096x128xf32>, %arg2: memref<32x4096x128xf32>, %arg3: memref<32x4096x128xf32>) {
      %c4096 = arith.constant 4096 : index
      %c128 = arith.constant 128 : index
      %c64 = arith.constant 64 : index
      %c2 = arith.constant 2 : index
      %cst = arith.constant 2.000000e+00 : f32
      %cst_0 = arith.constant 0.12751743074602467 : f64
      %cst_1 = arith.constant 0.000000e+00 : f32
      %cst_2 = arith.constant 1.000000e+00 : f32
      %cst_3 = arith.constant 0xFF800000 : f32
      %c32 = arith.constant 32 : index
      %c0 = arith.constant 0 : index
      %c8 = arith.constant 8 : index
      %c1 = arith.constant 1 : index
      scf.parallel (%arg4, %arg5) = (%c0, %c0) to (%c8, %c8) step (%c1, %c1) {
        scf.for %arg6 = %c0 to %c2 step %c1 {
          %12 = arith.muli %arg4, %c8 overflow<nsw> : index
          %13 = arith.addi %12, %arg5 : index
          %14 = arith.muli %arg6, %c64 overflow<nsw> : index
          %15 = arith.addi %13, %14 : index
          %16 = loom.alloc [32, 32, 128] on @L1 : memref<32x32x128xf32>
          %17 = loom.alloc [32, 32, 32] on @L1 : memref<32x32x32xf32>
          %18 = loom.alloc [32, 128, 32] on @L1 : memref<32x128x32xf32>
          %19 = loom.alloc [32, 32] on @L1 : memref<32x32xf32>
          %20 = loom.alloc [32, 32] on @L1 : memref<32x32xf32>
          %21 = loom.alloc [32, 32] on @L1 : memref<32x32xf32>
          %22 = loom.alloc [32, 32] on @L1 : memref<32x32xf32>
          %23 = loom.alloc [32, 32] on @L1 : memref<32x32xf32>
          %24 = loom.alloc [32, 32, 128] on @L1 : memref<32x32x128xf32>
          %25 = loom.alloc [32, 32, 128] on @L1 : memref<32x32x128xf32>
          %26 = loom.alloc [32, 32, 128] on @L1 : memref<32x32x128xf32>
          %27 = arith.muli %15, %c4096 : index
          %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%27], sizes: [32, 32, 128], strides: [524288, 128, 1] : memref<32x4096x128xf32> to memref<32x32x128xf32, strided<[524288, 128, 1], offset: ?>>
          loom.copy %reinterpret_cast, %24 src_mem_space @DRAM dst_mem_space @L1, interconnect : [], broadcast : [1, 1] : memref<32x32x128xf32, strided<[524288, 128, 1], offset: ?>>, memref<32x32x128xf32>
          linalg.fill ins(%cst_1 : f32) outs(%25 : memref<32x32x128xf32>)
          linalg.fill ins(%cst_2 : f32) outs(%19 : memref<32x32xf32>)
          linalg.fill ins(%cst_3 : f32) outs(%20 : memref<32x32xf32>)
          scf.for %arg7 = %c0 to %c128 step %c1 {
            %28 = arith.muli %arg7, %c32 : index
            %reinterpret_cast_5 = memref.reinterpret_cast %arg0 to offset: [%28], sizes: [32, 128, 32], strides: [524288, 4096, 1] : memref<32x128x4096xf32> to memref<32x128x32xf32, strided<[524288, 4096, 1], offset: ?>>
            loom.copy %reinterpret_cast_5, %18 src_mem_space @DRAM dst_mem_space @L1, interconnect : [@vertical_links], broadcast : [8, 1] : memref<32x128x32xf32, strided<[524288, 4096, 1], offset: ?>>, memref<32x128x32xf32>
            linalg.fill ins(%cst_1 : f32) outs(%17 : memref<32x32x32xf32>)
            linalg.batch_matmul ins(%24, %18 : memref<32x32x128xf32>, memref<32x128x32xf32>) outs(%17 : memref<32x32x32xf32>)
            linalg.fill ins(%cst_3 : f32) outs(%21 : memref<32x32xf32>)
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%17 : memref<32x32x32xf32>) outs(%21 : memref<32x32xf32>) {
            ^bb0(%in: f32, %out: f32):
              %30 = arith.maximumf %in, %out : f32
              linalg.yield %30 : f32
            }
            linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%20, %21 : memref<32x32xf32>, memref<32x32xf32>) outs(%21 : memref<32x32xf32>) {
            ^bb0(%in: f32, %in_7: f32, %out: f32):
              %30 = arith.truncf %cst_0 : f64 to f32
              %31 = arith.mulf %in_7, %30 : f32
              %32 = arith.cmpf ogt, %in, %31 : f32
              %33 = arith.select %32, %in, %31 : f32
              linalg.yield %33 : f32
            }
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%17, %21 : memref<32x32x32xf32>, memref<32x32xf32>) outs(%17 : memref<32x32x32xf32>) {
            ^bb0(%in: f32, %in_7: f32, %out: f32):
              %30 = arith.truncf %cst_0 : f64 to f32
              %31 = arith.mulf %in, %30 : f32
              %32 = arith.subf %31, %in_7 : f32
              %33 = math.powf %cst, %32 : f32
              linalg.yield %33 : f32
            }
            linalg.fill ins(%cst_1 : f32) outs(%22 : memref<32x32xf32>)
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%17 : memref<32x32x32xf32>) outs(%22 : memref<32x32xf32>) {
            ^bb0(%in: f32, %out: f32):
              %30 = arith.addf %in, %out : f32
              linalg.yield %30 : f32
            }
            linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%20, %21 : memref<32x32xf32>, memref<32x32xf32>) outs(%23 : memref<32x32xf32>) {
            ^bb0(%in: f32, %in_7: f32, %out: f32):
              %30 = arith.subf %in, %in_7 : f32
              %31 = math.powf %cst, %30 : f32
              linalg.yield %31 : f32
            }
            linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%19, %23, %22 : memref<32x32xf32>, memref<32x32xf32>, memref<32x32xf32>) outs(%19 : memref<32x32xf32>) {
            ^bb0(%in: f32, %in_7: f32, %in_8: f32, %out: f32):
              %30 = arith.mulf %in, %in_7 : f32
              %31 = arith.addf %30, %in_8 : f32
              linalg.yield %31 : f32
            }
            %29 = arith.muli %arg7, %c4096 : index
            %reinterpret_cast_6 = memref.reinterpret_cast %arg1 to offset: [%29], sizes: [32, 32, 128], strides: [524288, 128, 1] : memref<32x4096x128xf32> to memref<32x32x128xf32, strided<[524288, 128, 1], offset: ?>>
            loom.copy %reinterpret_cast_6, %16 src_mem_space @DRAM dst_mem_space @L1, interconnect : [], broadcast : [1, 1] : memref<32x32x128xf32, strided<[524288, 128, 1], offset: ?>>, memref<32x32x128xf32>
            linalg.fill ins(%cst_1 : f32) outs(%26 : memref<32x32x128xf32>)
            linalg.batch_matmul ins(%17, %16 : memref<32x32x32xf32>, memref<32x32x128xf32>) outs(%26 : memref<32x32x128xf32>)
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%26, %25, %23 : memref<32x32x128xf32>, memref<32x32x128xf32>, memref<32x32xf32>) outs(%25 : memref<32x32x128xf32>) {
            ^bb0(%in: f32, %in_7: f32, %in_8: f32, %out: f32):
              %30 = arith.mulf %in_7, %in_8 : f32
              %31 = arith.addf %in, %30 : f32
              linalg.yield %31 : f32
            }
            linalg.copy ins(%21 : memref<32x32xf32>) outs(%20 : memref<32x32xf32>)
          }
          linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%25, %19 : memref<32x32x128xf32>, memref<32x32xf32>) outs(%24 : memref<32x32x128xf32>) {
          ^bb0(%in: f32, %in_5: f32, %out: f32):
            %28 = arith.divf %in, %in_5 : f32
            linalg.yield %28 : f32
          }
          %reinterpret_cast_4 = memref.reinterpret_cast %arg3 to offset: [%27], sizes: [32, 32, 128], strides: [524288, 128, 1] : memref<32x4096x128xf32> to memref<32x32x128xf32, strided<[524288, 128, 1], offset: ?>>
          loom.copy %24, %reinterpret_cast_4 src_mem_space @L1 dst_mem_space @DRAM, interconnect : [], broadcast : [1, 1] : memref<32x32x128xf32>, memref<32x32x128xf32, strided<[524288, 128, 1], offset: ?>>
        }
        scf.reduce 
      } {loom.iter_types = [#loom.iter_type<spatial>, #loom.iter_type<spatial>], loom.mapped_to_dims = [@x, @y]}
      return
    }
  }
  module attributes {loom.pass_name = "Materialize"} {
    func.func @attention__d0i1_d1i1__f10__d_v_a__BB32__BM32__BN32(%arg0: memref<32x128x4096xf32>, %arg1: memref<32x4096x128xf32>, %arg2: memref<32x4096x128xf32>, %arg3: memref<32x4096x128xf32>) {
      %c4096 = arith.constant 4096 : index
      %c128 = arith.constant 128 : index
      %c64 = arith.constant 64 : index
      %c2 = arith.constant 2 : index
      %cst = arith.constant 2.000000e+00 : f32
      %cst_0 = arith.constant 0.12751743074602467 : f64
      %cst_1 = arith.constant 0.000000e+00 : f32
      %cst_2 = arith.constant 1.000000e+00 : f32
      %cst_3 = arith.constant 0xFF800000 : f32
      %c32 = arith.constant 32 : index
      %c0 = arith.constant 0 : index
      %c8 = arith.constant 8 : index
      %c1 = arith.constant 1 : index
      scf.parallel (%arg4, %arg5) = (%c0, %c0) to (%c8, %c8) step (%c1, %c1) {
        scf.for %arg6 = %c0 to %c2 step %c1 {
          %12 = arith.muli %arg4, %c8 overflow<nsw> : index
          %13 = arith.addi %12, %arg5 : index
          %14 = arith.muli %arg6, %c64 overflow<nsw> : index
          %15 = arith.addi %13, %14 : index
          %16 = loom.alloc [32, 32, 128] on @L1 : memref<32x32x128xf32>
          %17 = loom.alloc [32, 32, 32] on @L1 : memref<32x32x32xf32>
          %18 = loom.alloc [32, 128, 32] on @L1 : memref<32x128x32xf32>
          %19 = loom.alloc [32, 32] on @L1 : memref<32x32xf32>
          %20 = loom.alloc [32, 32] on @L1 : memref<32x32xf32>
          %21 = loom.alloc [32, 32] on @L1 : memref<32x32xf32>
          %22 = loom.alloc [32, 32] on @L1 : memref<32x32xf32>
          %23 = loom.alloc [32, 32] on @L1 : memref<32x32xf32>
          %24 = loom.alloc [32, 32, 128] on @L1 : memref<32x32x128xf32>
          %25 = loom.alloc [32, 32, 128] on @L1 : memref<32x32x128xf32>
          %26 = loom.alloc [32, 32, 128] on @L1 : memref<32x32x128xf32>
          %27 = arith.muli %15, %c4096 : index
          %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%27], sizes: [32, 32, 128], strides: [524288, 128, 1] : memref<32x4096x128xf32> to memref<32x32x128xf32, strided<[524288, 128, 1], offset: ?>>
          loom.copy %reinterpret_cast, %24 src_mem_space @DRAM dst_mem_space @L1, interconnect : [], broadcast : [1, 1] : memref<32x32x128xf32, strided<[524288, 128, 1], offset: ?>>, memref<32x32x128xf32>
          linalg.fill ins(%cst_1 : f32) outs(%25 : memref<32x32x128xf32>)
          linalg.fill ins(%cst_2 : f32) outs(%19 : memref<32x32xf32>)
          linalg.fill ins(%cst_3 : f32) outs(%20 : memref<32x32xf32>)
          scf.for %arg7 = %c0 to %c128 step %c1 {
            %28 = arith.muli %arg7, %c32 : index
            %reinterpret_cast_5 = memref.reinterpret_cast %arg0 to offset: [%28], sizes: [32, 128, 32], strides: [524288, 4096, 1] : memref<32x128x4096xf32> to memref<32x128x32xf32, strided<[524288, 4096, 1], offset: ?>>
            loom.copy %reinterpret_cast_5, %18 src_mem_space @DRAM dst_mem_space @L1, interconnect : [@vertical_links], broadcast : [8, 1] : memref<32x128x32xf32, strided<[524288, 4096, 1], offset: ?>>, memref<32x128x32xf32>
            linalg.fill ins(%cst_1 : f32) outs(%17 : memref<32x32x32xf32>)
            linalg.batch_matmul ins(%24, %18 : memref<32x32x128xf32>, memref<32x128x32xf32>) outs(%17 : memref<32x32x32xf32>)
            linalg.fill ins(%cst_3 : f32) outs(%21 : memref<32x32xf32>)
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%17 : memref<32x32x32xf32>) outs(%21 : memref<32x32xf32>) {
            ^bb0(%in: f32, %out: f32):
              %30 = arith.maximumf %in, %out : f32
              linalg.yield %30 : f32
            }
            linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%20, %21 : memref<32x32xf32>, memref<32x32xf32>) outs(%21 : memref<32x32xf32>) {
            ^bb0(%in: f32, %in_7: f32, %out: f32):
              %30 = arith.truncf %cst_0 : f64 to f32
              %31 = arith.mulf %in_7, %30 : f32
              %32 = arith.cmpf ogt, %in, %31 : f32
              %33 = arith.select %32, %in, %31 : f32
              linalg.yield %33 : f32
            }
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%17, %21 : memref<32x32x32xf32>, memref<32x32xf32>) outs(%17 : memref<32x32x32xf32>) {
            ^bb0(%in: f32, %in_7: f32, %out: f32):
              %30 = arith.truncf %cst_0 : f64 to f32
              %31 = arith.mulf %in, %30 : f32
              %32 = arith.subf %31, %in_7 : f32
              %33 = math.powf %cst, %32 : f32
              linalg.yield %33 : f32
            }
            linalg.fill ins(%cst_1 : f32) outs(%22 : memref<32x32xf32>)
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%17 : memref<32x32x32xf32>) outs(%22 : memref<32x32xf32>) {
            ^bb0(%in: f32, %out: f32):
              %30 = arith.addf %in, %out : f32
              linalg.yield %30 : f32
            }
            linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%20, %21 : memref<32x32xf32>, memref<32x32xf32>) outs(%23 : memref<32x32xf32>) {
            ^bb0(%in: f32, %in_7: f32, %out: f32):
              %30 = arith.subf %in, %in_7 : f32
              %31 = math.powf %cst, %30 : f32
              linalg.yield %31 : f32
            }
            linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%19, %23, %22 : memref<32x32xf32>, memref<32x32xf32>, memref<32x32xf32>) outs(%19 : memref<32x32xf32>) {
            ^bb0(%in: f32, %in_7: f32, %in_8: f32, %out: f32):
              %30 = arith.mulf %in, %in_7 : f32
              %31 = arith.addf %30, %in_8 : f32
              linalg.yield %31 : f32
            }
            %29 = arith.muli %arg7, %c4096 : index
            %reinterpret_cast_6 = memref.reinterpret_cast %arg1 to offset: [%29], sizes: [32, 32, 128], strides: [524288, 128, 1] : memref<32x4096x128xf32> to memref<32x32x128xf32, strided<[524288, 128, 1], offset: ?>>
            loom.copy %reinterpret_cast_6, %16 src_mem_space @DRAM dst_mem_space @L1, interconnect : [@horizontal_links, @vertical_links], broadcast : [8, 8] : memref<32x32x128xf32, strided<[524288, 128, 1], offset: ?>>, memref<32x32x128xf32>
            linalg.fill ins(%cst_1 : f32) outs(%26 : memref<32x32x128xf32>)
            linalg.batch_matmul ins(%17, %16 : memref<32x32x32xf32>, memref<32x32x128xf32>) outs(%26 : memref<32x32x128xf32>)
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%26, %25, %23 : memref<32x32x128xf32>, memref<32x32x128xf32>, memref<32x32xf32>) outs(%25 : memref<32x32x128xf32>) {
            ^bb0(%in: f32, %in_7: f32, %in_8: f32, %out: f32):
              %30 = arith.mulf %in_7, %in_8 : f32
              %31 = arith.addf %in, %30 : f32
              linalg.yield %31 : f32
            }
            linalg.copy ins(%21 : memref<32x32xf32>) outs(%20 : memref<32x32xf32>)
          }
          linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%25, %19 : memref<32x32x128xf32>, memref<32x32xf32>) outs(%24 : memref<32x32x128xf32>) {
          ^bb0(%in: f32, %in_5: f32, %out: f32):
            %28 = arith.divf %in, %in_5 : f32
            linalg.yield %28 : f32
          }
          %reinterpret_cast_4 = memref.reinterpret_cast %arg3 to offset: [%27], sizes: [32, 32, 128], strides: [524288, 128, 1] : memref<32x4096x128xf32> to memref<32x32x128xf32, strided<[524288, 128, 1], offset: ?>>
          loom.copy %24, %reinterpret_cast_4 src_mem_space @L1 dst_mem_space @DRAM, interconnect : [], broadcast : [1, 1] : memref<32x32x128xf32>, memref<32x32x128xf32, strided<[524288, 128, 1], offset: ?>>
        }
        scf.reduce 
      } {loom.iter_types = [#loom.iter_type<spatial>, #loom.iter_type<spatial>], loom.mapped_to_dims = [@x, @y]}
      return
    }
  }
  module attributes {loom.pass_name = "Materialize"} {
    func.func @attention__d0i1_d1i1__f10__d_v_h__BB32__BM32__BN32(%arg0: memref<32x128x4096xf32>, %arg1: memref<32x4096x128xf32>, %arg2: memref<32x4096x128xf32>, %arg3: memref<32x4096x128xf32>) {
      %c4096 = arith.constant 4096 : index
      %c128 = arith.constant 128 : index
      %c64 = arith.constant 64 : index
      %c2 = arith.constant 2 : index
      %cst = arith.constant 2.000000e+00 : f32
      %cst_0 = arith.constant 0.12751743074602467 : f64
      %cst_1 = arith.constant 0.000000e+00 : f32
      %cst_2 = arith.constant 1.000000e+00 : f32
      %cst_3 = arith.constant 0xFF800000 : f32
      %c32 = arith.constant 32 : index
      %c0 = arith.constant 0 : index
      %c8 = arith.constant 8 : index
      %c1 = arith.constant 1 : index
      scf.parallel (%arg4, %arg5) = (%c0, %c0) to (%c8, %c8) step (%c1, %c1) {
        scf.for %arg6 = %c0 to %c2 step %c1 {
          %12 = arith.muli %arg4, %c8 overflow<nsw> : index
          %13 = arith.addi %12, %arg5 : index
          %14 = arith.muli %arg6, %c64 overflow<nsw> : index
          %15 = arith.addi %13, %14 : index
          %16 = loom.alloc [32, 32, 128] on @L1 : memref<32x32x128xf32>
          %17 = loom.alloc [32, 32, 32] on @L1 : memref<32x32x32xf32>
          %18 = loom.alloc [32, 128, 32] on @L1 : memref<32x128x32xf32>
          %19 = loom.alloc [32, 32] on @L1 : memref<32x32xf32>
          %20 = loom.alloc [32, 32] on @L1 : memref<32x32xf32>
          %21 = loom.alloc [32, 32] on @L1 : memref<32x32xf32>
          %22 = loom.alloc [32, 32] on @L1 : memref<32x32xf32>
          %23 = loom.alloc [32, 32] on @L1 : memref<32x32xf32>
          %24 = loom.alloc [32, 32, 128] on @L1 : memref<32x32x128xf32>
          %25 = loom.alloc [32, 32, 128] on @L1 : memref<32x32x128xf32>
          %26 = loom.alloc [32, 32, 128] on @L1 : memref<32x32x128xf32>
          %27 = arith.muli %15, %c4096 : index
          %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%27], sizes: [32, 32, 128], strides: [524288, 128, 1] : memref<32x4096x128xf32> to memref<32x32x128xf32, strided<[524288, 128, 1], offset: ?>>
          loom.copy %reinterpret_cast, %24 src_mem_space @DRAM dst_mem_space @L1, interconnect : [], broadcast : [1, 1] : memref<32x32x128xf32, strided<[524288, 128, 1], offset: ?>>, memref<32x32x128xf32>
          linalg.fill ins(%cst_1 : f32) outs(%25 : memref<32x32x128xf32>)
          linalg.fill ins(%cst_2 : f32) outs(%19 : memref<32x32xf32>)
          linalg.fill ins(%cst_3 : f32) outs(%20 : memref<32x32xf32>)
          scf.for %arg7 = %c0 to %c128 step %c1 {
            %28 = arith.muli %arg7, %c32 : index
            %reinterpret_cast_5 = memref.reinterpret_cast %arg0 to offset: [%28], sizes: [32, 128, 32], strides: [524288, 4096, 1] : memref<32x128x4096xf32> to memref<32x128x32xf32, strided<[524288, 4096, 1], offset: ?>>
            loom.copy %reinterpret_cast_5, %18 src_mem_space @DRAM dst_mem_space @L1, interconnect : [@vertical_links], broadcast : [8, 1] : memref<32x128x32xf32, strided<[524288, 4096, 1], offset: ?>>, memref<32x128x32xf32>
            linalg.fill ins(%cst_1 : f32) outs(%17 : memref<32x32x32xf32>)
            linalg.batch_matmul ins(%24, %18 : memref<32x32x128xf32>, memref<32x128x32xf32>) outs(%17 : memref<32x32x32xf32>)
            linalg.fill ins(%cst_3 : f32) outs(%21 : memref<32x32xf32>)
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%17 : memref<32x32x32xf32>) outs(%21 : memref<32x32xf32>) {
            ^bb0(%in: f32, %out: f32):
              %30 = arith.maximumf %in, %out : f32
              linalg.yield %30 : f32
            }
            linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%20, %21 : memref<32x32xf32>, memref<32x32xf32>) outs(%21 : memref<32x32xf32>) {
            ^bb0(%in: f32, %in_7: f32, %out: f32):
              %30 = arith.truncf %cst_0 : f64 to f32
              %31 = arith.mulf %in_7, %30 : f32
              %32 = arith.cmpf ogt, %in, %31 : f32
              %33 = arith.select %32, %in, %31 : f32
              linalg.yield %33 : f32
            }
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%17, %21 : memref<32x32x32xf32>, memref<32x32xf32>) outs(%17 : memref<32x32x32xf32>) {
            ^bb0(%in: f32, %in_7: f32, %out: f32):
              %30 = arith.truncf %cst_0 : f64 to f32
              %31 = arith.mulf %in, %30 : f32
              %32 = arith.subf %31, %in_7 : f32
              %33 = math.powf %cst, %32 : f32
              linalg.yield %33 : f32
            }
            linalg.fill ins(%cst_1 : f32) outs(%22 : memref<32x32xf32>)
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%17 : memref<32x32x32xf32>) outs(%22 : memref<32x32xf32>) {
            ^bb0(%in: f32, %out: f32):
              %30 = arith.addf %in, %out : f32
              linalg.yield %30 : f32
            }
            linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%20, %21 : memref<32x32xf32>, memref<32x32xf32>) outs(%23 : memref<32x32xf32>) {
            ^bb0(%in: f32, %in_7: f32, %out: f32):
              %30 = arith.subf %in, %in_7 : f32
              %31 = math.powf %cst, %30 : f32
              linalg.yield %31 : f32
            }
            linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%19, %23, %22 : memref<32x32xf32>, memref<32x32xf32>, memref<32x32xf32>) outs(%19 : memref<32x32xf32>) {
            ^bb0(%in: f32, %in_7: f32, %in_8: f32, %out: f32):
              %30 = arith.mulf %in, %in_7 : f32
              %31 = arith.addf %30, %in_8 : f32
              linalg.yield %31 : f32
            }
            %29 = arith.muli %arg7, %c4096 : index
            %reinterpret_cast_6 = memref.reinterpret_cast %arg1 to offset: [%29], sizes: [32, 32, 128], strides: [524288, 128, 1] : memref<32x4096x128xf32> to memref<32x32x128xf32, strided<[524288, 128, 1], offset: ?>>
            loom.copy %reinterpret_cast_6, %16 src_mem_space @DRAM dst_mem_space @L1, interconnect : [@horizontal_links], broadcast : [1, 8] : memref<32x32x128xf32, strided<[524288, 128, 1], offset: ?>>, memref<32x32x128xf32>
            linalg.fill ins(%cst_1 : f32) outs(%26 : memref<32x32x128xf32>)
            linalg.batch_matmul ins(%17, %16 : memref<32x32x32xf32>, memref<32x32x128xf32>) outs(%26 : memref<32x32x128xf32>)
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%26, %25, %23 : memref<32x32x128xf32>, memref<32x32x128xf32>, memref<32x32xf32>) outs(%25 : memref<32x32x128xf32>) {
            ^bb0(%in: f32, %in_7: f32, %in_8: f32, %out: f32):
              %30 = arith.mulf %in_7, %in_8 : f32
              %31 = arith.addf %in, %30 : f32
              linalg.yield %31 : f32
            }
            linalg.copy ins(%21 : memref<32x32xf32>) outs(%20 : memref<32x32xf32>)
          }
          linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%25, %19 : memref<32x32x128xf32>, memref<32x32xf32>) outs(%24 : memref<32x32x128xf32>) {
          ^bb0(%in: f32, %in_5: f32, %out: f32):
            %28 = arith.divf %in, %in_5 : f32
            linalg.yield %28 : f32
          }
          %reinterpret_cast_4 = memref.reinterpret_cast %arg3 to offset: [%27], sizes: [32, 32, 128], strides: [524288, 128, 1] : memref<32x4096x128xf32> to memref<32x32x128xf32, strided<[524288, 128, 1], offset: ?>>
          loom.copy %24, %reinterpret_cast_4 src_mem_space @L1 dst_mem_space @DRAM, interconnect : [], broadcast : [1, 1] : memref<32x32x128xf32>, memref<32x32x128xf32, strided<[524288, 128, 1], offset: ?>>
        }
        scf.reduce 
      } {loom.iter_types = [#loom.iter_type<spatial>, #loom.iter_type<spatial>], loom.mapped_to_dims = [@x, @y]}
      return
    }
  }
  module attributes {loom.pass_name = "Materialize"} {
    func.func @attention__d0i1_d1i1__f10__d_v_v__BB32__BM32__BN32(%arg0: memref<32x128x4096xf32>, %arg1: memref<32x4096x128xf32>, %arg2: memref<32x4096x128xf32>, %arg3: memref<32x4096x128xf32>) {
      %c4096 = arith.constant 4096 : index
      %c128 = arith.constant 128 : index
      %c64 = arith.constant 64 : index
      %c2 = arith.constant 2 : index
      %cst = arith.constant 2.000000e+00 : f32
      %cst_0 = arith.constant 0.12751743074602467 : f64
      %cst_1 = arith.constant 0.000000e+00 : f32
      %cst_2 = arith.constant 1.000000e+00 : f32
      %cst_3 = arith.constant 0xFF800000 : f32
      %c32 = arith.constant 32 : index
      %c0 = arith.constant 0 : index
      %c8 = arith.constant 8 : index
      %c1 = arith.constant 1 : index
      scf.parallel (%arg4, %arg5) = (%c0, %c0) to (%c8, %c8) step (%c1, %c1) {
        scf.for %arg6 = %c0 to %c2 step %c1 {
          %12 = arith.muli %arg4, %c8 overflow<nsw> : index
          %13 = arith.addi %12, %arg5 : index
          %14 = arith.muli %arg6, %c64 overflow<nsw> : index
          %15 = arith.addi %13, %14 : index
          %16 = loom.alloc [32, 32, 128] on @L1 : memref<32x32x128xf32>
          %17 = loom.alloc [32, 32, 32] on @L1 : memref<32x32x32xf32>
          %18 = loom.alloc [32, 128, 32] on @L1 : memref<32x128x32xf32>
          %19 = loom.alloc [32, 32] on @L1 : memref<32x32xf32>
          %20 = loom.alloc [32, 32] on @L1 : memref<32x32xf32>
          %21 = loom.alloc [32, 32] on @L1 : memref<32x32xf32>
          %22 = loom.alloc [32, 32] on @L1 : memref<32x32xf32>
          %23 = loom.alloc [32, 32] on @L1 : memref<32x32xf32>
          %24 = loom.alloc [32, 32, 128] on @L1 : memref<32x32x128xf32>
          %25 = loom.alloc [32, 32, 128] on @L1 : memref<32x32x128xf32>
          %26 = loom.alloc [32, 32, 128] on @L1 : memref<32x32x128xf32>
          %27 = arith.muli %15, %c4096 : index
          %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%27], sizes: [32, 32, 128], strides: [524288, 128, 1] : memref<32x4096x128xf32> to memref<32x32x128xf32, strided<[524288, 128, 1], offset: ?>>
          loom.copy %reinterpret_cast, %24 src_mem_space @DRAM dst_mem_space @L1, interconnect : [], broadcast : [1, 1] : memref<32x32x128xf32, strided<[524288, 128, 1], offset: ?>>, memref<32x32x128xf32>
          linalg.fill ins(%cst_1 : f32) outs(%25 : memref<32x32x128xf32>)
          linalg.fill ins(%cst_2 : f32) outs(%19 : memref<32x32xf32>)
          linalg.fill ins(%cst_3 : f32) outs(%20 : memref<32x32xf32>)
          scf.for %arg7 = %c0 to %c128 step %c1 {
            %28 = arith.muli %arg7, %c32 : index
            %reinterpret_cast_5 = memref.reinterpret_cast %arg0 to offset: [%28], sizes: [32, 128, 32], strides: [524288, 4096, 1] : memref<32x128x4096xf32> to memref<32x128x32xf32, strided<[524288, 4096, 1], offset: ?>>
            loom.copy %reinterpret_cast_5, %18 src_mem_space @DRAM dst_mem_space @L1, interconnect : [@vertical_links], broadcast : [8, 1] : memref<32x128x32xf32, strided<[524288, 4096, 1], offset: ?>>, memref<32x128x32xf32>
            linalg.fill ins(%cst_1 : f32) outs(%17 : memref<32x32x32xf32>)
            linalg.batch_matmul ins(%24, %18 : memref<32x32x128xf32>, memref<32x128x32xf32>) outs(%17 : memref<32x32x32xf32>)
            linalg.fill ins(%cst_3 : f32) outs(%21 : memref<32x32xf32>)
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%17 : memref<32x32x32xf32>) outs(%21 : memref<32x32xf32>) {
            ^bb0(%in: f32, %out: f32):
              %30 = arith.maximumf %in, %out : f32
              linalg.yield %30 : f32
            }
            linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%20, %21 : memref<32x32xf32>, memref<32x32xf32>) outs(%21 : memref<32x32xf32>) {
            ^bb0(%in: f32, %in_7: f32, %out: f32):
              %30 = arith.truncf %cst_0 : f64 to f32
              %31 = arith.mulf %in_7, %30 : f32
              %32 = arith.cmpf ogt, %in, %31 : f32
              %33 = arith.select %32, %in, %31 : f32
              linalg.yield %33 : f32
            }
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%17, %21 : memref<32x32x32xf32>, memref<32x32xf32>) outs(%17 : memref<32x32x32xf32>) {
            ^bb0(%in: f32, %in_7: f32, %out: f32):
              %30 = arith.truncf %cst_0 : f64 to f32
              %31 = arith.mulf %in, %30 : f32
              %32 = arith.subf %31, %in_7 : f32
              %33 = math.powf %cst, %32 : f32
              linalg.yield %33 : f32
            }
            linalg.fill ins(%cst_1 : f32) outs(%22 : memref<32x32xf32>)
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%17 : memref<32x32x32xf32>) outs(%22 : memref<32x32xf32>) {
            ^bb0(%in: f32, %out: f32):
              %30 = arith.addf %in, %out : f32
              linalg.yield %30 : f32
            }
            linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%20, %21 : memref<32x32xf32>, memref<32x32xf32>) outs(%23 : memref<32x32xf32>) {
            ^bb0(%in: f32, %in_7: f32, %out: f32):
              %30 = arith.subf %in, %in_7 : f32
              %31 = math.powf %cst, %30 : f32
              linalg.yield %31 : f32
            }
            linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%19, %23, %22 : memref<32x32xf32>, memref<32x32xf32>, memref<32x32xf32>) outs(%19 : memref<32x32xf32>) {
            ^bb0(%in: f32, %in_7: f32, %in_8: f32, %out: f32):
              %30 = arith.mulf %in, %in_7 : f32
              %31 = arith.addf %30, %in_8 : f32
              linalg.yield %31 : f32
            }
            %29 = arith.muli %arg7, %c4096 : index
            %reinterpret_cast_6 = memref.reinterpret_cast %arg1 to offset: [%29], sizes: [32, 32, 128], strides: [524288, 128, 1] : memref<32x4096x128xf32> to memref<32x32x128xf32, strided<[524288, 128, 1], offset: ?>>
            loom.copy %reinterpret_cast_6, %16 src_mem_space @DRAM dst_mem_space @L1, interconnect : [@vertical_links], broadcast : [8, 1] : memref<32x32x128xf32, strided<[524288, 128, 1], offset: ?>>, memref<32x32x128xf32>
            linalg.fill ins(%cst_1 : f32) outs(%26 : memref<32x32x128xf32>)
            linalg.batch_matmul ins(%17, %16 : memref<32x32x32xf32>, memref<32x32x128xf32>) outs(%26 : memref<32x32x128xf32>)
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%26, %25, %23 : memref<32x32x128xf32>, memref<32x32x128xf32>, memref<32x32xf32>) outs(%25 : memref<32x32x128xf32>) {
            ^bb0(%in: f32, %in_7: f32, %in_8: f32, %out: f32):
              %30 = arith.mulf %in_7, %in_8 : f32
              %31 = arith.addf %in, %30 : f32
              linalg.yield %31 : f32
            }
            linalg.copy ins(%21 : memref<32x32xf32>) outs(%20 : memref<32x32xf32>)
          }
          linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%25, %19 : memref<32x32x128xf32>, memref<32x32xf32>) outs(%24 : memref<32x32x128xf32>) {
          ^bb0(%in: f32, %in_5: f32, %out: f32):
            %28 = arith.divf %in, %in_5 : f32
            linalg.yield %28 : f32
          }
          %reinterpret_cast_4 = memref.reinterpret_cast %arg3 to offset: [%27], sizes: [32, 32, 128], strides: [524288, 128, 1] : memref<32x4096x128xf32> to memref<32x32x128xf32, strided<[524288, 128, 1], offset: ?>>
          loom.copy %24, %reinterpret_cast_4 src_mem_space @L1 dst_mem_space @DRAM, interconnect : [], broadcast : [1, 1] : memref<32x32x128xf32>, memref<32x32x128xf32, strided<[524288, 128, 1], offset: ?>>
        }
        scf.reduce 
      } {loom.iter_types = [#loom.iter_type<spatial>, #loom.iter_type<spatial>], loom.mapped_to_dims = [@x, @y]}
      return
    }
  }
}
