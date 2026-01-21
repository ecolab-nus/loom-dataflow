#loc = loc("/path/to/loom/test/Triton/mm_fixed_strides/mm.py":17:0)
#loc2 = loc("/path/to/loom/test/Triton/mm_fixed_strides/mm.py":75:25)
#loc12 = loc("/tmp/tmpupax5k0l/tt.mlir":29:69)
#map = affine_map<(d0, d1) -> (d0, d1)>
module {
  module {
    loom.constraint_space @constraints {
      %bm = loom.symbolic_var "BM" : index
      %bn = loom.symbolic_var "BN" : index
      %bk = loom.symbolic_var "BK" : index

      loom.range %bm [0, 512]
      loom.range %bn [0, 512]
      loom.range %bk [0, 512]
    }
    func.func @matmul_kernel (
      %arg0: memref<*xf32> {tt.divisibility = 16 : i32}, 
      %arg1: memref<*xf32> {tt.divisibility = 16 : i32}, 
      %arg2: memref<*xf32> {tt.divisibility = 16 : i32}, 
      %arg3: index, %arg4: index, %arg5: index, %arg6: index, %arg7: index, %arg8: index) {
      %M_idx = arith.constant 512 : index
      %N_idx = arith.constant 512 : index
      %K_idx = arith.constant 512 : index
      %c1_idx = arith.constant 1 : index
      %block_m = loom.get_symbolic_block_size @constraints::@BM : index
      %block_n = loom.get_symbolic_block_size @constraints::@BN : index
      %block_k = loom.get_symbolic_block_size @constraints::@BK : index
      %Kblk_idx = arith.ceildivsi %K_idx, %block_k : index
      %cst = arith.constant 0.000000e+00 : f32 loc(#loc1)
      %c8_idx = arith.constant 8 : index loc(#loc1)
      %c512 = arith.constant 512 : index loc(#loc1)
      %c0_idx = arith.constant 0 : index loc(#loc1)
      %0 = tensor.empty(%block_m, %block_n) : tensor<?x?xf32> loc(#loc2)
      %1 = linalg.fill ins(%cst : f32) outs(%0 : tensor<?x?xf32>) -> tensor<?x?xf32> loc(#loc2)
      %2 = arith.muli %arg6, %block_m : index loc(#loc3)
      %3 = arith.muli %arg7, %block_n : index loc(#loc4)
      %6 = scf.for %arg9 = %c0_idx to %Kblk_idx step %c1_idx iter_args(%arg10 = %1) -> (tensor<?x?xf32>) {
        %9 = arith.muli %arg9, %block_k : index loc(#loc9)
        %11 = arith.muli %2, %c512 : index loc(#loc11)
        %12 = arith.addi %11, %9 : index loc(#loc11)
        %reinterpret_cast_0 = loom.reinterpret_cast %arg0 to offset: [%12], sizes: [%block_m, %block_k], strides: [%K_idx, %c1_idx], reuse: [seq = false, spat = false, temp =false] : memref<*xf32> to memref<?x?xf32, strided<[?, ?], offset: ?>> loc(#loc11)
        %alloc = loom.alloc(%block_m, %block_k) on @L1 : memref<?x?xf32> loc(#loc11)
        loom.copy %reinterpret_cast_0, %alloc, interconnect: [], broadcast: [1, 1] : memref<?x?xf32, strided<[?, ?], offset: ?>> to memref<?x?xf32> loc(#loc11)
        %13 = bufferization.to_tensor %alloc restrict writable : memref<?x?xf32> to tensor<?x?xf32> loc(#loc11)
        %14 = arith.muli %9, %c512 : index loc(#loc10)
        %15 = arith.addi %14, %3 : index loc(#loc10)
        %reinterpret_cast_1 = loom.reinterpret_cast %arg1 to offset: [%15], sizes: [%block_k, %block_n], strides: [%N_idx, %c1_idx], reuse: [seq = false, spat = false, temp =false] : memref<*xf32> to memref<?x?xf32, strided<[?, ?], offset: ?>> loc(#loc10)
        %alloc_2 = loom.alloc(%block_k, %block_n) on @L1 : memref<?x?xf32> loc(#loc10)
        loom.copy %reinterpret_cast_1, %alloc_2, interconnect: [], broadcast: [1, 1] : memref<?x?xf32, strided<[?, ?], offset: ?>> to memref<?x?xf32> loc(#loc10)
        %16 = bufferization.to_tensor %alloc_2 restrict writable : memref<?x?xf32> to tensor<?x?xf32> loc(#loc10)
        %17 = linalg.matmul ins(%13, %16 : tensor<?x?xf32>, tensor<?x?xf32>) outs(%1 : tensor<?x?xf32>) -> tensor<?x?xf32> loc(#loc2)
        %18 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel", "parallel"]} ins(%arg10, %17 : tensor<?x?xf32>, tensor<?x?xf32>) outs(%arg10 : tensor<?x?xf32>) {
        ^bb0(%in: f32 loc("/tmp/tmpupax5k0l/tt.mlir":29:69), %in_3: f32 loc("/path/to/loom/test/Triton/mm_fixed_strides/mm.py":75:25), %out: f32 loc("/tmp/tmpupax5k0l/tt.mlir":29:69)):
          %19 = arith.addf %in, %in_3 : f32 loc(#loc2)
          linalg.yield %19 : f32 loc(#loc2)
        } -> tensor<?x?xf32> loc(#loc2)
        scf.yield %18 : tensor<?x?xf32> loc(#loc13)
      } loc(#loc8)
      %7 = arith.muli %2, %c512 : index loc(#loc11)
      %8 = arith.addi %7, %3 : index loc(#loc5)
      %reinterpret_cast = loom.reinterpret_cast %arg2 to offset: [%8], sizes: [%block_m, %block_n], strides: [%N_idx, %c1_idx], reuse: [seq = false, spat = false, temp =false] : memref<*xf32> to memref<?x?xf32, strided<[?, ?], offset: ?>> loc(#loc5)
      bufferization.materialize_in_destination %6 in writable %reinterpret_cast : (tensor<?x?xf32>, memref<?x?xf32, strided<[?, ?], offset: ?>>) -> () loc(#loc5)
      return loc(#loc)
    } loc(#loc)
  }
} loc(#loc)
#loc1 = loc(unknown)
#loc3 = loc("/path/to/loom/test/Triton/mm_fixed_strides/mm.py":44:21)
#loc4 = loc("/path/to/loom/test/Triton/mm_fixed_strides/mm.py":45:21)
#loc5 = loc("/path/to/loom/test/Triton/mm_fixed_strides/mm.py":85:22)
#loc6 = loc("/path/to/loom/test/Triton/mm_fixed_strides/mm.py":58:12)
#loc7 = loc("/path/to/loom/test/Triton/mm_fixed_strides/mm.py":67:12)
#loc8 = loc("/path/to/loom/test/Triton/mm_fixed_strides/mm.py":49:23)
#loc9 = loc("/path/to/loom/test/Triton/mm_fixed_strides/mm.py":50:22)
#loc10 = loc("/path/to/loom/test/Triton/mm_fixed_strides/mm.py":74:20)
#loc11 = loc("/path/to/loom/test/Triton/mm_fixed_strides/mm.py":73:20)
#loc13 = loc("/path/to/loom/test/Triton/mm_fixed_strides/mm.py":75:8)
#loc14 = loc(fused[#loc5, #loc6])
#loc15 = loc(fused[#loc5, #loc7])
#loc16 = loc(fused[#loc10, #loc6])

