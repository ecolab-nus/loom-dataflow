module {
  module attributes {loom.pass_name = "Materialize"} {
    func.func @attention__BB64__BM64__BN64(%arg0: memref<2x4096x4096xf32>, %arg1: memref<2x4096x4096xf32>, %arg2: memref<2x4096x4096xf32>, %arg3: memref<2x4096x4096xf32>) {
      %c262144 = arith.constant 262144 : index
      %cst = arith.constant 2.000000e+00 : f32
      %cst_0 = arith.constant 0.022542110000000001 : f64
      %cst_1 = arith.constant 0.000000e+00 : f32
      %cst_2 = arith.constant 1.000000e+00 : f32
      %cst_3 = arith.constant 0xFF800000 : f32
      %c64 = arith.constant 64 : index
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      scf.parallel (%arg4) = (%c0) to (%c64) step (%c1) {
        %0 = loom.alloc [64, 64, 4096] on @L1 : memref<64x64x4096xf32>
        %1 = loom.alloc [64, 64, 64] on @L1 : memref<64x64x64xf32>
        %2 = loom.alloc [64, 4096, 64] on @L1 : memref<64x4096x64xf32>
        %3 = loom.alloc [64, 64, 4096] on @L1 : memref<64x64x4096xf32>
        %4 = loom.alloc [64, 64, 4096] on @L1 : memref<64x64x4096xf32>
        %5 = loom.alloc [64, 64, 4096] on @L1 : memref<64x64x4096xf32>
        %6 = loom.alloc [64, 64, 4096] on @L1 : memref<64x64x4096xf32>
        %7 = loom.alloc [64, 64] on @L1 : memref<64x64xf32>
        %8 = loom.alloc [64, 64] on @L1 : memref<64x64xf32>
        %9 = loom.alloc [64, 64] on @L1 : memref<64x64xf32>
        %10 = loom.alloc [64, 64] on @L1 : memref<64x64xf32>
        %11 = loom.alloc [64, 64] on @L1 : memref<64x64xf32>
        %12 = loom.alloc [64, 64] on @L1 : memref<64x64xf32>
        linalg.fill ins(%cst_3 : f32) outs(%9 : memref<64x64xf32>)
        linalg.fill ins(%cst_2 : f32) outs(%8 : memref<64x64xf32>)
        linalg.fill ins(%cst_1 : f32) outs(%5 : memref<64x64x4096xf32>)
        %13 = arith.muli %arg4, %c262144 : index
        %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%13], sizes: [64, 64, 4096], strides: [16777216, 4096, 1] : memref<2x4096x4096xf32> to memref<64x64x4096xf32, strided<[16777216, 4096, 1], offset: ?>>
        loom.copy %reinterpret_cast, %4 src_mem_space @DRAM dst_mem_space @L1, interconnect : [], broadcast : [1, 1] : memref<64x64x4096xf32, strided<[16777216, 4096, 1], offset: ?>>, memref<64x64x4096xf32>
        scf.for %arg5 = %c0 to %c64 step %c1 {
          %14 = arith.muli %arg5, %c64 : index
          %reinterpret_cast_5 = memref.reinterpret_cast %arg0 to offset: [%14], sizes: [64, 4096, 64], strides: [16777216, 4096, 1] : memref<2x4096x4096xf32> to memref<64x4096x64xf32, strided<[16777216, 4096, 1], offset: ?>>
          loom.copy %reinterpret_cast_5, %2 src_mem_space @DRAM dst_mem_space @L1, interconnect : [], broadcast : [1, 1] : memref<64x4096x64xf32, strided<[16777216, 4096, 1], offset: ?>>, memref<64x4096x64xf32>
          linalg.fill ins(%cst_1 : f32) outs(%1 : memref<64x64x64xf32>)
          linalg.batch_matmul ins(%4, %2 : memref<64x64x4096xf32>, memref<64x4096x64xf32>) outs(%1 : memref<64x64x64xf32>)
          linalg.fill ins(%cst_3 : f32) outs(%10 : memref<64x64xf32>)
          linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%1 : memref<64x64x64xf32>) outs(%10 : memref<64x64xf32>) {
          ^bb0(%in: f32, %out: f32):
            %16 = arith.maximumf %in, %out : f32
            linalg.yield %16 : f32
          }
          linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%9, %10 : memref<64x64xf32>, memref<64x64xf32>) outs(%10 : memref<64x64xf32>) {
          ^bb0(%in: f32, %in_7: f32, %out: f32):
            %16 = arith.truncf %cst_0 : f64 to f32
            %17 = arith.mulf %in_7, %16 : f32
            %18 = arith.cmpf ogt, %in, %17 : f32
            %19 = arith.select %18, %in, %17 : f32
            linalg.yield %19 : f32
          }
          linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1, %10 : memref<64x64x64xf32>, memref<64x64xf32>) outs(%1 : memref<64x64x64xf32>) {
          ^bb0(%in: f32, %in_7: f32, %out: f32):
            %16 = arith.truncf %cst_0 : f64 to f32
            %17 = arith.mulf %in, %16 : f32
            %18 = arith.subf %17, %in_7 : f32
            %19 = math.powf %cst, %18 : f32
            linalg.yield %19 : f32
          }
          linalg.fill ins(%cst_1 : f32) outs(%11 : memref<64x64xf32>)
          linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%1 : memref<64x64x64xf32>) outs(%11 : memref<64x64xf32>) {
          ^bb0(%in: f32, %out: f32):
            %16 = arith.addf %in, %out : f32
            linalg.yield %16 : f32
          }
          linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%9, %10 : memref<64x64xf32>, memref<64x64xf32>) outs(%12 : memref<64x64xf32>) {
          ^bb0(%in: f32, %in_7: f32, %out: f32):
            %16 = arith.subf %in, %in_7 : f32
            %17 = math.powf %cst, %16 : f32
            linalg.yield %17 : f32
          }
          linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%8, %12, %11 : memref<64x64xf32>, memref<64x64xf32>, memref<64x64xf32>) outs(%8 : memref<64x64xf32>) {
          ^bb0(%in: f32, %in_7: f32, %in_8: f32, %out: f32):
            %16 = arith.mulf %in, %in_7 : f32
            %17 = arith.addf %16, %in_8 : f32
            linalg.yield %17 : f32
          }
          %15 = arith.muli %arg5, %c262144 : index
          %reinterpret_cast_6 = memref.reinterpret_cast %arg1 to offset: [%15], sizes: [64, 64, 4096], strides: [16777216, 4096, 1] : memref<2x4096x4096xf32> to memref<64x64x4096xf32, strided<[16777216, 4096, 1], offset: ?>>
          loom.copy %reinterpret_cast_6, %0 src_mem_space @DRAM dst_mem_space @L1, interconnect : [], broadcast : [1, 1] : memref<64x64x4096xf32, strided<[16777216, 4096, 1], offset: ?>>, memref<64x64x4096xf32>
          linalg.fill ins(%cst_1 : f32) outs(%6 : memref<64x64x4096xf32>)
          linalg.batch_matmul ins(%1, %0 : memref<64x64x64xf32>, memref<64x64x4096xf32>) outs(%6 : memref<64x64x4096xf32>)
          linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%6, %5, %12 : memref<64x64x4096xf32>, memref<64x64x4096xf32>, memref<64x64xf32>) outs(%5 : memref<64x64x4096xf32>) {
          ^bb0(%in: f32, %in_7: f32, %in_8: f32, %out: f32):
            %16 = arith.mulf %in_7, %in_8 : f32
            %17 = arith.addf %in, %16 : f32
            linalg.yield %17 : f32
          }
          linalg.copy ins(%10 : memref<64x64xf32>) outs(%9 : memref<64x64xf32>)
        }
        linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%5, %8 : memref<64x64x4096xf32>, memref<64x64xf32>) outs(%3 : memref<64x64x4096xf32>) {
        ^bb0(%in: f32, %in_5: f32, %out: f32):
          %14 = arith.divf %in, %in_5 : f32
          linalg.yield %14 : f32
        }
        %reinterpret_cast_4 = memref.reinterpret_cast %arg3 to offset: [%13], sizes: [64, 64, 4096], strides: [16777216, 4096, 1] : memref<2x4096x4096xf32> to memref<64x64x4096xf32, strided<[16777216, 4096, 1], offset: ?>>
        loom.copy %3, %reinterpret_cast_4 src_mem_space @L1 dst_mem_space @DRAM, interconnect : [], broadcast : [1, 1] : memref<64x64x4096xf32>, memref<64x64x4096xf32, strided<[16777216, 4096, 1], offset: ?>>
        scf.reduce 
      }
      return
    }
    func.func @attention__BB32__BM32__BN256(%arg0: memref<2x4096x4096xf32>, %arg1: memref<2x4096x4096xf32>, %arg2: memref<2x4096x4096xf32>, %arg3: memref<2x4096x4096xf32>) {
      %c1048576 = arith.constant 1048576 : index
      %c131072 = arith.constant 131072 : index
      %c16 = arith.constant 16 : index
      %cst = arith.constant 2.000000e+00 : f32
      %cst_0 = arith.constant 0.022542110000000001 : f64
      %cst_1 = arith.constant 0.000000e+00 : f32
      %cst_2 = arith.constant 1.000000e+00 : f32
      %cst_3 = arith.constant 0xFF800000 : f32
      %c256 = arith.constant 256 : index
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c128 = arith.constant 128 : index
      scf.parallel (%arg4) = (%c0) to (%c128) step (%c1) {
        %0 = loom.alloc [32, 256, 4096] on @L1 : memref<32x256x4096xf32>
        %1 = loom.alloc [32, 32, 256] on @L1 : memref<32x32x256xf32>
        %2 = loom.alloc [32, 4096, 256] on @L1 : memref<32x4096x256xf32>
        %3 = loom.alloc [32, 32, 4096] on @L1 : memref<32x32x4096xf32>
        %4 = loom.alloc [32, 32, 4096] on @L1 : memref<32x32x4096xf32>
        %5 = loom.alloc [32, 32, 4096] on @L1 : memref<32x32x4096xf32>
        %6 = loom.alloc [32, 32, 4096] on @L1 : memref<32x32x4096xf32>
        %7 = loom.alloc [32, 32] on @L1 : memref<32x32xf32>
        %8 = loom.alloc [32, 32] on @L1 : memref<32x32xf32>
        %9 = loom.alloc [32, 32] on @L1 : memref<32x32xf32>
        %10 = loom.alloc [32, 32] on @L1 : memref<32x32xf32>
        %11 = loom.alloc [32, 32] on @L1 : memref<32x32xf32>
        %12 = loom.alloc [32, 32] on @L1 : memref<32x32xf32>
        linalg.fill ins(%cst_3 : f32) outs(%9 : memref<32x32xf32>)
        linalg.fill ins(%cst_2 : f32) outs(%8 : memref<32x32xf32>)
        linalg.fill ins(%cst_1 : f32) outs(%5 : memref<32x32x4096xf32>)
        %13 = arith.muli %arg4, %c131072 : index
        %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%13], sizes: [32, 32, 4096], strides: [16777216, 4096, 1] : memref<2x4096x4096xf32> to memref<32x32x4096xf32, strided<[16777216, 4096, 1], offset: ?>>
        loom.copy %reinterpret_cast, %4 src_mem_space @DRAM dst_mem_space @L1, interconnect : [], broadcast : [1, 1] : memref<32x32x4096xf32, strided<[16777216, 4096, 1], offset: ?>>, memref<32x32x4096xf32>
        scf.for %arg5 = %c0 to %c16 step %c1 {
          %14 = arith.muli %arg5, %c256 : index
          %reinterpret_cast_5 = memref.reinterpret_cast %arg0 to offset: [%14], sizes: [32, 4096, 256], strides: [16777216, 4096, 1] : memref<2x4096x4096xf32> to memref<32x4096x256xf32, strided<[16777216, 4096, 1], offset: ?>>
          loom.copy %reinterpret_cast_5, %2 src_mem_space @DRAM dst_mem_space @L1, interconnect : [], broadcast : [1, 1] : memref<32x4096x256xf32, strided<[16777216, 4096, 1], offset: ?>>, memref<32x4096x256xf32>
          linalg.fill ins(%cst_1 : f32) outs(%1 : memref<32x32x256xf32>)
          linalg.batch_matmul ins(%4, %2 : memref<32x32x4096xf32>, memref<32x4096x256xf32>) outs(%1 : memref<32x32x256xf32>)
          linalg.fill ins(%cst_3 : f32) outs(%10 : memref<32x32xf32>)
          linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%1 : memref<32x32x256xf32>) outs(%10 : memref<32x32xf32>) {
          ^bb0(%in: f32, %out: f32):
            %16 = arith.maximumf %in, %out : f32
            linalg.yield %16 : f32
          }
          linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%9, %10 : memref<32x32xf32>, memref<32x32xf32>) outs(%10 : memref<32x32xf32>) {
          ^bb0(%in: f32, %in_7: f32, %out: f32):
            %16 = arith.truncf %cst_0 : f64 to f32
            %17 = arith.mulf %in_7, %16 : f32
            %18 = arith.cmpf ogt, %in, %17 : f32
            %19 = arith.select %18, %in, %17 : f32
            linalg.yield %19 : f32
          }
          linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1, %10 : memref<32x32x256xf32>, memref<32x32xf32>) outs(%1 : memref<32x32x256xf32>) {
          ^bb0(%in: f32, %in_7: f32, %out: f32):
            %16 = arith.truncf %cst_0 : f64 to f32
            %17 = arith.mulf %in, %16 : f32
            %18 = arith.subf %17, %in_7 : f32
            %19 = math.powf %cst, %18 : f32
            linalg.yield %19 : f32
          }
          linalg.fill ins(%cst_1 : f32) outs(%11 : memref<32x32xf32>)
          linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%1 : memref<32x32x256xf32>) outs(%11 : memref<32x32xf32>) {
          ^bb0(%in: f32, %out: f32):
            %16 = arith.addf %in, %out : f32
            linalg.yield %16 : f32
          }
          linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%9, %10 : memref<32x32xf32>, memref<32x32xf32>) outs(%12 : memref<32x32xf32>) {
          ^bb0(%in: f32, %in_7: f32, %out: f32):
            %16 = arith.subf %in, %in_7 : f32
            %17 = math.powf %cst, %16 : f32
            linalg.yield %17 : f32
          }
          linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%8, %12, %11 : memref<32x32xf32>, memref<32x32xf32>, memref<32x32xf32>) outs(%8 : memref<32x32xf32>) {
          ^bb0(%in: f32, %in_7: f32, %in_8: f32, %out: f32):
            %16 = arith.mulf %in, %in_7 : f32
            %17 = arith.addf %16, %in_8 : f32
            linalg.yield %17 : f32
          }
          %15 = arith.muli %arg5, %c1048576 : index
          %reinterpret_cast_6 = memref.reinterpret_cast %arg1 to offset: [%15], sizes: [32, 256, 4096], strides: [16777216, 4096, 1] : memref<2x4096x4096xf32> to memref<32x256x4096xf32, strided<[16777216, 4096, 1], offset: ?>>
          loom.copy %reinterpret_cast_6, %0 src_mem_space @DRAM dst_mem_space @L1, interconnect : [], broadcast : [1, 1] : memref<32x256x4096xf32, strided<[16777216, 4096, 1], offset: ?>>, memref<32x256x4096xf32>
          linalg.fill ins(%cst_1 : f32) outs(%6 : memref<32x32x4096xf32>)
          linalg.batch_matmul ins(%1, %0 : memref<32x32x256xf32>, memref<32x256x4096xf32>) outs(%6 : memref<32x32x4096xf32>)
          linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%6, %5, %12 : memref<32x32x4096xf32>, memref<32x32x4096xf32>, memref<32x32xf32>) outs(%5 : memref<32x32x4096xf32>) {
          ^bb0(%in: f32, %in_7: f32, %in_8: f32, %out: f32):
            %16 = arith.mulf %in_7, %in_8 : f32
            %17 = arith.addf %in, %16 : f32
            linalg.yield %17 : f32
          }
          linalg.copy ins(%10 : memref<32x32xf32>) outs(%9 : memref<32x32xf32>)
        }
        linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%5, %8 : memref<32x32x4096xf32>, memref<32x32xf32>) outs(%3 : memref<32x32x4096xf32>) {
        ^bb0(%in: f32, %in_5: f32, %out: f32):
          %14 = arith.divf %in, %in_5 : f32
          linalg.yield %14 : f32
        }
        %reinterpret_cast_4 = memref.reinterpret_cast %arg3 to offset: [%13], sizes: [32, 32, 4096], strides: [16777216, 4096, 1] : memref<2x4096x4096xf32> to memref<32x32x4096xf32, strided<[16777216, 4096, 1], offset: ?>>
        loom.copy %3, %reinterpret_cast_4 src_mem_space @L1 dst_mem_space @DRAM, interconnect : [], broadcast : [1, 1] : memref<32x32x4096xf32>, memref<32x32x4096xf32, strided<[16777216, 4096, 1], offset: ?>>
        scf.reduce 
      }
      return
    }
  }
}
