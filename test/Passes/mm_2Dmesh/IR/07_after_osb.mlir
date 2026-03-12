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
    func.func @matmul__d0i0_d1i0__f01__d_d__BK512__BM64__BN4096(%arg0: memref<4096x512xf16>, %arg1: memref<512x4096xf16>, %arg2: memref<4096x4096xf16>) {
      %c262144 = arith.constant 262144 : index
      %c32768 = arith.constant 32768 : index
      %cst = arith.constant 0.000000e+00 : f16
      %c0 = arith.constant 0 : index
      %c8 = arith.constant 8 : index
      %c1 = arith.constant 1 : index
      scf.parallel (%arg3, %arg4) = (%c0, %c0) to (%c8, %c8) step (%c1, %c1) {
        %12 = arith.muli %arg3, %c8 overflow<nsw> : index
        %13 = arith.addi %12, %arg4 : index
        %14 = loom.alloc [512, 4096] on @L1 : memref<512x4096xf16>
        %15 = loom.semaphore_take %14 : memref<512x4096xf16> -> memref<512x4096xf16>
        %16 = loom.alloc [64, 512] on @L1 : memref<64x512xf16>
        %17 = loom.semaphore_take %16 : memref<64x512xf16> -> memref<64x512xf16>
        %18 = loom.alloc [64, 4096] on @L1 : memref<64x4096xf16>
        %19 = loom.semaphore_take %18 : memref<64x4096xf16> -> memref<64x4096xf16>
        linalg.fill ins(%cst : f16) outs(%19 : memref<64x4096xf16>)
        %20 = arith.muli %13, %c32768 : index
        %reinterpret_cast = memref.reinterpret_cast %arg0 to offset: [%20], sizes: [64, 512], strides: [512, 1] : memref<4096x512xf16> to memref<64x512xf16, strided<[512, 1], offset: ?>>
        loom.copy %reinterpret_cast, %17 src_mem_space @DRAM dst_mem_space @L1, interconnect : [], broadcast : [1, 1] : memref<64x512xf16, strided<[512, 1], offset: ?>>, memref<64x512xf16>
        %reinterpret_cast_0 = memref.reinterpret_cast %arg1 to offset: [0], sizes: [512, 4096], strides: [4096, 1] : memref<512x4096xf16> to memref<512x4096xf16, strided<[4096, 1]>>
        %cast = memref.cast %reinterpret_cast_0 : memref<512x4096xf16, strided<[4096, 1]>> to memref<512x4096xf16, strided<[4096, 1], offset: ?>>
        loom.copy %cast, %15 src_mem_space @DRAM dst_mem_space @L1, interconnect : [], broadcast : [1, 1] : memref<512x4096xf16, strided<[4096, 1], offset: ?>>, memref<512x4096xf16>
        linalg.matmul ins(%17, %15 : memref<64x512xf16>, memref<512x4096xf16>) outs(%19 : memref<64x4096xf16>)
        loom.semaphore_give %15 : memref<512x4096xf16>
        loom.semaphore_give %17 : memref<64x512xf16>
        %21 = arith.muli %13, %c262144 : index
        %reinterpret_cast_1 = memref.reinterpret_cast %arg2 to offset: [%21], sizes: [64, 4096], strides: [4096, 1] : memref<4096x4096xf16> to memref<64x4096xf16, strided<[4096, 1], offset: ?>>
        loom.copy %19, %reinterpret_cast_1 src_mem_space @L1 dst_mem_space @DRAM, interconnect : [], broadcast : [1, 1] : memref<64x4096xf16>, memref<64x4096xf16, strided<[4096, 1], offset: ?>>
        loom.semaphore_give %19 : memref<64x4096xf16>
        scf.reduce 
      } {loom.iter_types = [#loom.iter_type<spatial>, #loom.iter_type<spatial>], loom.mapped_to_dims = [@x, @y]}
      return
    }
  }
  module attributes {loom.pass_name = "Materialize"} {
    func.func @matmul__d0i0_d1i0__f01__d_a__BK512__BM64__BN4096(%arg0: memref<4096x512xf16>, %arg1: memref<512x4096xf16>, %arg2: memref<4096x4096xf16>) {
      %c262144 = arith.constant 262144 : index
      %c32768 = arith.constant 32768 : index
      %cst = arith.constant 0.000000e+00 : f16
      %c0 = arith.constant 0 : index
      %c8 = arith.constant 8 : index
      %c1 = arith.constant 1 : index
      scf.parallel (%arg3, %arg4) = (%c0, %c0) to (%c8, %c8) step (%c1, %c1) {
        %12 = arith.muli %arg3, %c8 overflow<nsw> : index
        %13 = arith.addi %12, %arg4 : index
        %14 = loom.alloc [512, 4096] on @L1 : memref<512x4096xf16>
        %15 = loom.semaphore_take %14 : memref<512x4096xf16> -> memref<512x4096xf16>
        %16 = loom.alloc [64, 512] on @L1 : memref<64x512xf16>
        %17 = loom.semaphore_take %16 : memref<64x512xf16> -> memref<64x512xf16>
        %18 = loom.alloc [64, 4096] on @L1 : memref<64x4096xf16>
        %19 = loom.semaphore_take %18 : memref<64x4096xf16> -> memref<64x4096xf16>
        linalg.fill ins(%cst : f16) outs(%19 : memref<64x4096xf16>)
        %20 = arith.muli %13, %c32768 : index
        %reinterpret_cast = memref.reinterpret_cast %arg0 to offset: [%20], sizes: [64, 512], strides: [512, 1] : memref<4096x512xf16> to memref<64x512xf16, strided<[512, 1], offset: ?>>
        loom.copy %reinterpret_cast, %17 src_mem_space @DRAM dst_mem_space @L1, interconnect : [], broadcast : [1, 1] : memref<64x512xf16, strided<[512, 1], offset: ?>>, memref<64x512xf16>
        %reinterpret_cast_0 = memref.reinterpret_cast %arg1 to offset: [0], sizes: [512, 4096], strides: [4096, 1] : memref<512x4096xf16> to memref<512x4096xf16, strided<[4096, 1]>>
        %cast = memref.cast %reinterpret_cast_0 : memref<512x4096xf16, strided<[4096, 1]>> to memref<512x4096xf16, strided<[4096, 1], offset: ?>>
        loom.copy %cast, %15 src_mem_space @DRAM dst_mem_space @L1, interconnect : [@horizontal_links, @vertical_links], broadcast : [8, 8] : memref<512x4096xf16, strided<[4096, 1], offset: ?>>, memref<512x4096xf16>
        linalg.matmul ins(%17, %15 : memref<64x512xf16>, memref<512x4096xf16>) outs(%19 : memref<64x4096xf16>)
        loom.semaphore_give %15 : memref<512x4096xf16>
        loom.semaphore_give %17 : memref<64x512xf16>
        %21 = arith.muli %13, %c262144 : index
        %reinterpret_cast_1 = memref.reinterpret_cast %arg2 to offset: [%21], sizes: [64, 4096], strides: [4096, 1] : memref<4096x4096xf16> to memref<64x4096xf16, strided<[4096, 1], offset: ?>>
        loom.copy %19, %reinterpret_cast_1 src_mem_space @L1 dst_mem_space @DRAM, interconnect : [], broadcast : [1, 1] : memref<64x4096xf16>, memref<64x4096xf16, strided<[4096, 1], offset: ?>>
        loom.semaphore_give %19 : memref<64x4096xf16>
        scf.reduce 
      } {loom.iter_types = [#loom.iter_type<spatial>, #loom.iter_type<spatial>], loom.mapped_to_dims = [@x, @y]}
      return
    }
  }
  module attributes {loom.pass_name = "Materialize"} {
    func.func @matmul__d0i0_d1i0__f01__d_h__BK64__BM64__BN4096(%arg0: memref<4096x512xf16>, %arg1: memref<512x4096xf16>, %arg2: memref<4096x4096xf16>) {
      %c262144 = arith.constant 262144 : index
      %c32768 = arith.constant 32768 : index
      %cst = arith.constant 0.000000e+00 : f16
      %c64 = arith.constant 64 : index
      %c0 = arith.constant 0 : index
      %c8 = arith.constant 8 : index
      %c1 = arith.constant 1 : index
      scf.parallel (%arg3, %arg4) = (%c0, %c0) to (%c8, %c8) step (%c1, %c1) {
        %12 = arith.muli %arg3, %c8 overflow<nsw> : index
        %13 = arith.addi %12, %arg4 : index
        %14 = loom.alloc [64, 4096] on @L1 : memref<64x4096xf16>
        %15 = loom.semaphore_take %14 : memref<64x4096xf16> -> memref<64x4096xf16>
        %16 = loom.alloc [64, 64] on @L1 : memref<64x64xf16>
        %17 = loom.semaphore_take %16 : memref<64x64xf16> -> memref<64x64xf16>
        %18 = loom.alloc [64, 4096] on @L1 : memref<64x4096xf16>
        %19 = loom.semaphore_take %18 : memref<64x4096xf16> -> memref<64x4096xf16>
        linalg.fill ins(%cst : f16) outs(%19 : memref<64x4096xf16>)
        scf.for %arg5 = %c0 to %c8 step %c1 {
          %21 = arith.muli %arg5, %c64 : index
          %22 = arith.muli %13, %c32768 : index
          %23 = arith.addi %22, %21 : index
          %reinterpret_cast_0 = memref.reinterpret_cast %arg0 to offset: [%23], sizes: [64, 64], strides: [512, 1] : memref<4096x512xf16> to memref<64x64xf16, strided<[512, 1], offset: ?>>
          loom.copy %reinterpret_cast_0, %17 src_mem_space @DRAM dst_mem_space @L1, interconnect : [], broadcast : [1, 1] : memref<64x64xf16, strided<[512, 1], offset: ?>>, memref<64x64xf16>
          %24 = arith.muli %arg5, %c262144 : index
          %reinterpret_cast_1 = memref.reinterpret_cast %arg1 to offset: [%24], sizes: [64, 4096], strides: [4096, 1] : memref<512x4096xf16> to memref<64x4096xf16, strided<[4096, 1], offset: ?>>
          loom.copy %reinterpret_cast_1, %15 src_mem_space @DRAM dst_mem_space @L1, interconnect : [@horizontal_links], broadcast : [1, 8] : memref<64x4096xf16, strided<[4096, 1], offset: ?>>, memref<64x4096xf16>
          linalg.matmul ins(%17, %15 : memref<64x64xf16>, memref<64x4096xf16>) outs(%19 : memref<64x4096xf16>)
          loom.semaphore_give %15 : memref<64x4096xf16>
          loom.semaphore_give %17 : memref<64x64xf16>
        }
        %20 = arith.muli %13, %c262144 : index
        %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%20], sizes: [64, 4096], strides: [4096, 1] : memref<4096x4096xf16> to memref<64x4096xf16, strided<[4096, 1], offset: ?>>
        loom.copy %19, %reinterpret_cast src_mem_space @L1 dst_mem_space @DRAM, interconnect : [], broadcast : [1, 1] : memref<64x4096xf16>, memref<64x4096xf16, strided<[4096, 1], offset: ?>>
        loom.semaphore_give %19 : memref<64x4096xf16>
        scf.reduce 
      } {loom.iter_types = [#loom.iter_type<spatial>, #loom.iter_type<spatial>], loom.mapped_to_dims = [@x, @y]}
      return
    }
  }
  module attributes {loom.pass_name = "Materialize"} {
    func.func @matmul__d0i0_d1i0__f01__d_v__BK256__BM64__BN1024(%arg0: memref<4096x512xf16>, %arg1: memref<512x4096xf16>, %arg2: memref<4096x4096xf16>) {
      %c262144 = arith.constant 262144 : index
      %c1048576 = arith.constant 1048576 : index
      %c32768 = arith.constant 32768 : index
      %c2 = arith.constant 2 : index
      %c4 = arith.constant 4 : index
      %cst = arith.constant 0.000000e+00 : f16
      %c1024 = arith.constant 1024 : index
      %c256 = arith.constant 256 : index
      %c0 = arith.constant 0 : index
      %c8 = arith.constant 8 : index
      %c1 = arith.constant 1 : index
      scf.parallel (%arg3, %arg4) = (%c0, %c0) to (%c8, %c8) step (%c1, %c1) {
        scf.for %arg5 = %c0 to %c4 step %c1 {
          %12 = arith.muli %arg3, %c8 overflow<nsw> : index
          %13 = arith.addi %12, %arg4 : index
          %14 = loom.alloc [256, 1024] on @L1 : memref<256x1024xf16>
          %15 = loom.semaphore_take %14 : memref<256x1024xf16> -> memref<256x1024xf16>
          %16 = loom.alloc [64, 256] on @L1 : memref<64x256xf16>
          %17 = loom.semaphore_take %16 : memref<64x256xf16> -> memref<64x256xf16>
          %18 = loom.alloc [64, 1024] on @L1 : memref<64x1024xf16>
          %19 = loom.semaphore_take %18 : memref<64x1024xf16> -> memref<64x1024xf16>
          linalg.fill ins(%cst : f16) outs(%19 : memref<64x1024xf16>)
          scf.for %arg6 = %c0 to %c2 step %c1 {
            %23 = arith.muli %arg6, %c256 : index
            %24 = arith.muli %13, %c32768 : index
            %25 = arith.addi %24, %23 : index
            %reinterpret_cast_0 = memref.reinterpret_cast %arg0 to offset: [%25], sizes: [64, 256], strides: [512, 1] : memref<4096x512xf16> to memref<64x256xf16, strided<[512, 1], offset: ?>>
            loom.copy %reinterpret_cast_0, %17 src_mem_space @DRAM dst_mem_space @L1, interconnect : [], broadcast : [1, 1] : memref<64x256xf16, strided<[512, 1], offset: ?>>, memref<64x256xf16>
            %26 = arith.muli %arg5, %c1024 : index
            %27 = arith.muli %arg6, %c1048576 : index
            %28 = arith.addi %27, %26 : index
            %reinterpret_cast_1 = memref.reinterpret_cast %arg1 to offset: [%28], sizes: [256, 1024], strides: [4096, 1] : memref<512x4096xf16> to memref<256x1024xf16, strided<[4096, 1], offset: ?>>
            loom.copy %reinterpret_cast_1, %15 src_mem_space @DRAM dst_mem_space @L1, interconnect : [@vertical_links], broadcast : [8, 1] : memref<256x1024xf16, strided<[4096, 1], offset: ?>>, memref<256x1024xf16>
            linalg.matmul ins(%17, %15 : memref<64x256xf16>, memref<256x1024xf16>) outs(%19 : memref<64x1024xf16>)
            loom.semaphore_give %15 : memref<256x1024xf16>
            loom.semaphore_give %17 : memref<64x256xf16>
          }
          %20 = arith.muli %arg5, %c1024 : index
          %21 = arith.muli %13, %c262144 : index
          %22 = arith.addi %21, %20 : index
          %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%22], sizes: [64, 1024], strides: [4096, 1] : memref<4096x4096xf16> to memref<64x1024xf16, strided<[4096, 1], offset: ?>>
          loom.copy %19, %reinterpret_cast src_mem_space @L1 dst_mem_space @DRAM, interconnect : [], broadcast : [1, 1] : memref<64x1024xf16>, memref<64x1024xf16, strided<[4096, 1], offset: ?>>
          loom.semaphore_give %19 : memref<64x1024xf16>
        }
        scf.reduce 
      } {loom.iter_types = [#loom.iter_type<spatial>, #loom.iter_type<spatial>], loom.mapped_to_dims = [@x, @y]}
      return
    }
  }
  module attributes {loom.pass_name = "Materialize"} {
    func.func @matmul__d0i0_d1i0__f10__d_d__BK512__BM64__BN4096(%arg0: memref<4096x512xf16>, %arg1: memref<512x4096xf16>, %arg2: memref<4096x4096xf16>) {
      %c262144 = arith.constant 262144 : index
      %c32768 = arith.constant 32768 : index
      %cst = arith.constant 0.000000e+00 : f16
      %c0 = arith.constant 0 : index
      %c8 = arith.constant 8 : index
      %c1 = arith.constant 1 : index
      scf.parallel (%arg3, %arg4) = (%c0, %c0) to (%c8, %c8) step (%c1, %c1) {
        %12 = arith.muli %arg3, %c8 overflow<nsw> : index
        %13 = arith.addi %12, %arg4 : index
        %14 = loom.alloc [512, 4096] on @L1 : memref<512x4096xf16>
        %15 = loom.semaphore_take %14 : memref<512x4096xf16> -> memref<512x4096xf16>
        %16 = loom.alloc [64, 512] on @L1 : memref<64x512xf16>
        %17 = loom.semaphore_take %16 : memref<64x512xf16> -> memref<64x512xf16>
        %18 = loom.alloc [64, 4096] on @L1 : memref<64x4096xf16>
        %19 = loom.semaphore_take %18 : memref<64x4096xf16> -> memref<64x4096xf16>
        linalg.fill ins(%cst : f16) outs(%19 : memref<64x4096xf16>)
        %20 = arith.muli %13, %c32768 : index
        %reinterpret_cast = memref.reinterpret_cast %arg0 to offset: [%20], sizes: [64, 512], strides: [512, 1] : memref<4096x512xf16> to memref<64x512xf16, strided<[512, 1], offset: ?>>
        loom.copy %reinterpret_cast, %17 src_mem_space @DRAM dst_mem_space @L1, interconnect : [], broadcast : [1, 1] : memref<64x512xf16, strided<[512, 1], offset: ?>>, memref<64x512xf16>
        %reinterpret_cast_0 = memref.reinterpret_cast %arg1 to offset: [0], sizes: [512, 4096], strides: [4096, 1] : memref<512x4096xf16> to memref<512x4096xf16, strided<[4096, 1]>>
        %cast = memref.cast %reinterpret_cast_0 : memref<512x4096xf16, strided<[4096, 1]>> to memref<512x4096xf16, strided<[4096, 1], offset: ?>>
        loom.copy %cast, %15 src_mem_space @DRAM dst_mem_space @L1, interconnect : [], broadcast : [1, 1] : memref<512x4096xf16, strided<[4096, 1], offset: ?>>, memref<512x4096xf16>
        linalg.matmul ins(%17, %15 : memref<64x512xf16>, memref<512x4096xf16>) outs(%19 : memref<64x4096xf16>)
        loom.semaphore_give %15 : memref<512x4096xf16>
        loom.semaphore_give %17 : memref<64x512xf16>
        %21 = arith.muli %13, %c262144 : index
        %reinterpret_cast_1 = memref.reinterpret_cast %arg2 to offset: [%21], sizes: [64, 4096], strides: [4096, 1] : memref<4096x4096xf16> to memref<64x4096xf16, strided<[4096, 1], offset: ?>>
        loom.copy %19, %reinterpret_cast_1 src_mem_space @L1 dst_mem_space @DRAM, interconnect : [], broadcast : [1, 1] : memref<64x4096xf16>, memref<64x4096xf16, strided<[4096, 1], offset: ?>>
        loom.semaphore_give %19 : memref<64x4096xf16>
        scf.reduce 
      } {loom.iter_types = [#loom.iter_type<spatial>, #loom.iter_type<spatial>], loom.mapped_to_dims = [@x, @y]}
      return
    }
  }
  module attributes {loom.pass_name = "Materialize"} {
    func.func @matmul__d0i0_d1i0__f10__d_a__BK512__BM64__BN4096(%arg0: memref<4096x512xf16>, %arg1: memref<512x4096xf16>, %arg2: memref<4096x4096xf16>) {
      %c262144 = arith.constant 262144 : index
      %c32768 = arith.constant 32768 : index
      %cst = arith.constant 0.000000e+00 : f16
      %c0 = arith.constant 0 : index
      %c8 = arith.constant 8 : index
      %c1 = arith.constant 1 : index
      scf.parallel (%arg3, %arg4) = (%c0, %c0) to (%c8, %c8) step (%c1, %c1) {
        %12 = arith.muli %arg3, %c8 overflow<nsw> : index
        %13 = arith.addi %12, %arg4 : index
        %14 = loom.alloc [512, 4096] on @L1 : memref<512x4096xf16>
        %15 = loom.semaphore_take %14 : memref<512x4096xf16> -> memref<512x4096xf16>
        %16 = loom.alloc [64, 512] on @L1 : memref<64x512xf16>
        %17 = loom.semaphore_take %16 : memref<64x512xf16> -> memref<64x512xf16>
        %18 = loom.alloc [64, 4096] on @L1 : memref<64x4096xf16>
        %19 = loom.semaphore_take %18 : memref<64x4096xf16> -> memref<64x4096xf16>
        linalg.fill ins(%cst : f16) outs(%19 : memref<64x4096xf16>)
        %20 = arith.muli %13, %c32768 : index
        %reinterpret_cast = memref.reinterpret_cast %arg0 to offset: [%20], sizes: [64, 512], strides: [512, 1] : memref<4096x512xf16> to memref<64x512xf16, strided<[512, 1], offset: ?>>
        loom.copy %reinterpret_cast, %17 src_mem_space @DRAM dst_mem_space @L1, interconnect : [], broadcast : [1, 1] : memref<64x512xf16, strided<[512, 1], offset: ?>>, memref<64x512xf16>
        %reinterpret_cast_0 = memref.reinterpret_cast %arg1 to offset: [0], sizes: [512, 4096], strides: [4096, 1] : memref<512x4096xf16> to memref<512x4096xf16, strided<[4096, 1]>>
        %cast = memref.cast %reinterpret_cast_0 : memref<512x4096xf16, strided<[4096, 1]>> to memref<512x4096xf16, strided<[4096, 1], offset: ?>>
        loom.copy %cast, %15 src_mem_space @DRAM dst_mem_space @L1, interconnect : [@horizontal_links, @vertical_links], broadcast : [8, 8] : memref<512x4096xf16, strided<[4096, 1], offset: ?>>, memref<512x4096xf16>
        linalg.matmul ins(%17, %15 : memref<64x512xf16>, memref<512x4096xf16>) outs(%19 : memref<64x4096xf16>)
        loom.semaphore_give %15 : memref<512x4096xf16>
        loom.semaphore_give %17 : memref<64x512xf16>
        %21 = arith.muli %13, %c262144 : index
        %reinterpret_cast_1 = memref.reinterpret_cast %arg2 to offset: [%21], sizes: [64, 4096], strides: [4096, 1] : memref<4096x4096xf16> to memref<64x4096xf16, strided<[4096, 1], offset: ?>>
        loom.copy %19, %reinterpret_cast_1 src_mem_space @L1 dst_mem_space @DRAM, interconnect : [], broadcast : [1, 1] : memref<64x4096xf16>, memref<64x4096xf16, strided<[4096, 1], offset: ?>>
        loom.semaphore_give %19 : memref<64x4096xf16>
        scf.reduce 
      } {loom.iter_types = [#loom.iter_type<spatial>, #loom.iter_type<spatial>], loom.mapped_to_dims = [@x, @y]}
      return
    }
  }
  module attributes {loom.pass_name = "Materialize"} {
    func.func @matmul__d0i0_d1i0__f10__d_h__BK32__BM64__BN4096(%arg0: memref<4096x512xf16>, %arg1: memref<512x4096xf16>, %arg2: memref<4096x4096xf16>) {
      %c262144 = arith.constant 262144 : index
      %c131072 = arith.constant 131072 : index
      %c32768 = arith.constant 32768 : index
      %c16 = arith.constant 16 : index
      %cst = arith.constant 0.000000e+00 : f16
      %c32 = arith.constant 32 : index
      %c0 = arith.constant 0 : index
      %c8 = arith.constant 8 : index
      %c1 = arith.constant 1 : index
      scf.parallel (%arg3, %arg4) = (%c0, %c0) to (%c8, %c8) step (%c1, %c1) {
        %12 = arith.muli %arg3, %c8 overflow<nsw> : index
        %13 = arith.addi %12, %arg4 : index
        %14 = loom.alloc [32, 4096] on @L1 : memref<32x4096xf16>
        %15 = loom.semaphore_take %14 : memref<32x4096xf16> -> memref<32x4096xf16>
        %16 = loom.alloc [64, 32] on @L1 : memref<64x32xf16>
        %17 = loom.semaphore_take %16 : memref<64x32xf16> -> memref<64x32xf16>
        %18 = loom.alloc [64, 4096] on @L1 : memref<64x4096xf16>
        %19 = loom.semaphore_take %18 : memref<64x4096xf16> -> memref<64x4096xf16>
        linalg.fill ins(%cst : f16) outs(%19 : memref<64x4096xf16>)
        scf.for %arg5 = %c0 to %c16 step %c1 {
          %21 = arith.muli %arg5, %c32 : index
          %22 = arith.muli %13, %c32768 : index
          %23 = arith.addi %22, %21 : index
          %reinterpret_cast_0 = memref.reinterpret_cast %arg0 to offset: [%23], sizes: [64, 32], strides: [512, 1] : memref<4096x512xf16> to memref<64x32xf16, strided<[512, 1], offset: ?>>
          loom.copy %reinterpret_cast_0, %17 src_mem_space @DRAM dst_mem_space @L1, interconnect : [], broadcast : [1, 1] : memref<64x32xf16, strided<[512, 1], offset: ?>>, memref<64x32xf16>
          %24 = arith.muli %arg5, %c131072 : index
          %reinterpret_cast_1 = memref.reinterpret_cast %arg1 to offset: [%24], sizes: [32, 4096], strides: [4096, 1] : memref<512x4096xf16> to memref<32x4096xf16, strided<[4096, 1], offset: ?>>
          loom.copy %reinterpret_cast_1, %15 src_mem_space @DRAM dst_mem_space @L1, interconnect : [@horizontal_links], broadcast : [1, 8] : memref<32x4096xf16, strided<[4096, 1], offset: ?>>, memref<32x4096xf16>
          linalg.matmul ins(%17, %15 : memref<64x32xf16>, memref<32x4096xf16>) outs(%19 : memref<64x4096xf16>)
          loom.semaphore_give %15 : memref<32x4096xf16>
          loom.semaphore_give %17 : memref<64x32xf16>
        }
        %20 = arith.muli %13, %c262144 : index
        %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%20], sizes: [64, 4096], strides: [4096, 1] : memref<4096x4096xf16> to memref<64x4096xf16, strided<[4096, 1], offset: ?>>
        loom.copy %19, %reinterpret_cast src_mem_space @L1 dst_mem_space @DRAM, interconnect : [], broadcast : [1, 1] : memref<64x4096xf16>, memref<64x4096xf16, strided<[4096, 1], offset: ?>>
        loom.semaphore_give %19 : memref<64x4096xf16>
        scf.reduce 
      } {loom.iter_types = [#loom.iter_type<spatial>, #loom.iter_type<spatial>], loom.mapped_to_dims = [@x, @y]}
      return
    }
  }
  module attributes {loom.pass_name = "Materialize"} {
    func.func @matmul__d0i0_d1i0__f10__d_v__BK128__BM64__BN4096(%arg0: memref<4096x512xf16>, %arg1: memref<512x4096xf16>, %arg2: memref<4096x4096xf16>) {
      %c262144 = arith.constant 262144 : index
      %c524288 = arith.constant 524288 : index
      %c32768 = arith.constant 32768 : index
      %c4 = arith.constant 4 : index
      %cst = arith.constant 0.000000e+00 : f16
      %c128 = arith.constant 128 : index
      %c0 = arith.constant 0 : index
      %c8 = arith.constant 8 : index
      %c1 = arith.constant 1 : index
      scf.parallel (%arg3, %arg4) = (%c0, %c0) to (%c8, %c8) step (%c1, %c1) {
        %12 = arith.muli %arg3, %c8 overflow<nsw> : index
        %13 = arith.addi %12, %arg4 : index
        %14 = loom.alloc [128, 4096] on @L1 : memref<128x4096xf16>
        %15 = loom.semaphore_take %14 : memref<128x4096xf16> -> memref<128x4096xf16>
        %16 = loom.alloc [64, 128] on @L1 : memref<64x128xf16>
        %17 = loom.semaphore_take %16 : memref<64x128xf16> -> memref<64x128xf16>
        %18 = loom.alloc [64, 4096] on @L1 : memref<64x4096xf16>
        %19 = loom.semaphore_take %18 : memref<64x4096xf16> -> memref<64x4096xf16>
        linalg.fill ins(%cst : f16) outs(%19 : memref<64x4096xf16>)
        scf.for %arg5 = %c0 to %c4 step %c1 {
          %21 = arith.muli %arg5, %c128 : index
          %22 = arith.muli %13, %c32768 : index
          %23 = arith.addi %22, %21 : index
          %reinterpret_cast_0 = memref.reinterpret_cast %arg0 to offset: [%23], sizes: [64, 128], strides: [512, 1] : memref<4096x512xf16> to memref<64x128xf16, strided<[512, 1], offset: ?>>
          loom.copy %reinterpret_cast_0, %17 src_mem_space @DRAM dst_mem_space @L1, interconnect : [], broadcast : [1, 1] : memref<64x128xf16, strided<[512, 1], offset: ?>>, memref<64x128xf16>
          %24 = arith.muli %arg5, %c524288 : index
          %reinterpret_cast_1 = memref.reinterpret_cast %arg1 to offset: [%24], sizes: [128, 4096], strides: [4096, 1] : memref<512x4096xf16> to memref<128x4096xf16, strided<[4096, 1], offset: ?>>
          loom.copy %reinterpret_cast_1, %15 src_mem_space @DRAM dst_mem_space @L1, interconnect : [@vertical_links], broadcast : [8, 1] : memref<128x4096xf16, strided<[4096, 1], offset: ?>>, memref<128x4096xf16>
          linalg.matmul ins(%17, %15 : memref<64x128xf16>, memref<128x4096xf16>) outs(%19 : memref<64x4096xf16>)
          loom.semaphore_give %15 : memref<128x4096xf16>
          loom.semaphore_give %17 : memref<64x128xf16>
        }
        %20 = arith.muli %13, %c262144 : index
        %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%20], sizes: [64, 4096], strides: [4096, 1] : memref<4096x4096xf16> to memref<64x4096xf16, strided<[4096, 1], offset: ?>>
        loom.copy %19, %reinterpret_cast src_mem_space @L1 dst_mem_space @DRAM, interconnect : [], broadcast : [1, 1] : memref<64x4096xf16>, memref<64x4096xf16, strided<[4096, 1], offset: ?>>
        loom.semaphore_give %19 : memref<64x4096xf16>
        scf.reduce 
      } {loom.iter_types = [#loom.iter_type<spatial>, #loom.iter_type<spatial>], loom.mapped_to_dims = [@x, @y]}
      return
    }
  }
  module attributes {loom.pass_name = "Materialize"} {
    func.func @matmul__d1i0_d0i0__f01__d_d__BK512__BM64__BN4096(%arg0: memref<4096x512xf16>, %arg1: memref<512x4096xf16>, %arg2: memref<4096x4096xf16>) {
      %c262144 = arith.constant 262144 : index
      %c32768 = arith.constant 32768 : index
      %cst = arith.constant 0.000000e+00 : f16
      %c0 = arith.constant 0 : index
      %c8 = arith.constant 8 : index
      %c1 = arith.constant 1 : index
      scf.parallel (%arg3, %arg4) = (%c0, %c0) to (%c8, %c8) step (%c1, %c1) {
        %12 = arith.muli %arg3, %c8 overflow<nsw> : index
        %13 = arith.addi %12, %arg4 : index
        %14 = loom.alloc [512, 4096] on @L1 : memref<512x4096xf16>
        %15 = loom.semaphore_take %14 : memref<512x4096xf16> -> memref<512x4096xf16>
        %16 = loom.alloc [64, 512] on @L1 : memref<64x512xf16>
        %17 = loom.semaphore_take %16 : memref<64x512xf16> -> memref<64x512xf16>
        %18 = loom.alloc [64, 4096] on @L1 : memref<64x4096xf16>
        %19 = loom.semaphore_take %18 : memref<64x4096xf16> -> memref<64x4096xf16>
        linalg.fill ins(%cst : f16) outs(%19 : memref<64x4096xf16>)
        %20 = arith.muli %13, %c32768 : index
        %reinterpret_cast = memref.reinterpret_cast %arg0 to offset: [%20], sizes: [64, 512], strides: [512, 1] : memref<4096x512xf16> to memref<64x512xf16, strided<[512, 1], offset: ?>>
        loom.copy %reinterpret_cast, %17 src_mem_space @DRAM dst_mem_space @L1, interconnect : [], broadcast : [1, 1] : memref<64x512xf16, strided<[512, 1], offset: ?>>, memref<64x512xf16>
        %reinterpret_cast_0 = memref.reinterpret_cast %arg1 to offset: [0], sizes: [512, 4096], strides: [4096, 1] : memref<512x4096xf16> to memref<512x4096xf16, strided<[4096, 1]>>
        %cast = memref.cast %reinterpret_cast_0 : memref<512x4096xf16, strided<[4096, 1]>> to memref<512x4096xf16, strided<[4096, 1], offset: ?>>
        loom.copy %cast, %15 src_mem_space @DRAM dst_mem_space @L1, interconnect : [], broadcast : [1, 1] : memref<512x4096xf16, strided<[4096, 1], offset: ?>>, memref<512x4096xf16>
        linalg.matmul ins(%17, %15 : memref<64x512xf16>, memref<512x4096xf16>) outs(%19 : memref<64x4096xf16>)
        loom.semaphore_give %15 : memref<512x4096xf16>
        loom.semaphore_give %17 : memref<64x512xf16>
        %21 = arith.muli %13, %c262144 : index
        %reinterpret_cast_1 = memref.reinterpret_cast %arg2 to offset: [%21], sizes: [64, 4096], strides: [4096, 1] : memref<4096x4096xf16> to memref<64x4096xf16, strided<[4096, 1], offset: ?>>
        loom.copy %19, %reinterpret_cast_1 src_mem_space @L1 dst_mem_space @DRAM, interconnect : [], broadcast : [1, 1] : memref<64x4096xf16>, memref<64x4096xf16, strided<[4096, 1], offset: ?>>
        loom.semaphore_give %19 : memref<64x4096xf16>
        scf.reduce 
      } {loom.iter_types = [#loom.iter_type<spatial>, #loom.iter_type<spatial>], loom.mapped_to_dims = [@y, @x]}
      return
    }
  }
  module attributes {loom.pass_name = "Materialize"} {
    func.func @matmul__d1i0_d0i0__f01__d_a__BK512__BM64__BN4096(%arg0: memref<4096x512xf16>, %arg1: memref<512x4096xf16>, %arg2: memref<4096x4096xf16>) {
      %c262144 = arith.constant 262144 : index
      %c32768 = arith.constant 32768 : index
      %cst = arith.constant 0.000000e+00 : f16
      %c0 = arith.constant 0 : index
      %c8 = arith.constant 8 : index
      %c1 = arith.constant 1 : index
      scf.parallel (%arg3, %arg4) = (%c0, %c0) to (%c8, %c8) step (%c1, %c1) {
        %12 = arith.muli %arg3, %c8 overflow<nsw> : index
        %13 = arith.addi %12, %arg4 : index
        %14 = loom.alloc [512, 4096] on @L1 : memref<512x4096xf16>
        %15 = loom.semaphore_take %14 : memref<512x4096xf16> -> memref<512x4096xf16>
        %16 = loom.alloc [64, 512] on @L1 : memref<64x512xf16>
        %17 = loom.semaphore_take %16 : memref<64x512xf16> -> memref<64x512xf16>
        %18 = loom.alloc [64, 4096] on @L1 : memref<64x4096xf16>
        %19 = loom.semaphore_take %18 : memref<64x4096xf16> -> memref<64x4096xf16>
        linalg.fill ins(%cst : f16) outs(%19 : memref<64x4096xf16>)
        %20 = arith.muli %13, %c32768 : index
        %reinterpret_cast = memref.reinterpret_cast %arg0 to offset: [%20], sizes: [64, 512], strides: [512, 1] : memref<4096x512xf16> to memref<64x512xf16, strided<[512, 1], offset: ?>>
        loom.copy %reinterpret_cast, %17 src_mem_space @DRAM dst_mem_space @L1, interconnect : [], broadcast : [1, 1] : memref<64x512xf16, strided<[512, 1], offset: ?>>, memref<64x512xf16>
        %reinterpret_cast_0 = memref.reinterpret_cast %arg1 to offset: [0], sizes: [512, 4096], strides: [4096, 1] : memref<512x4096xf16> to memref<512x4096xf16, strided<[4096, 1]>>
        %cast = memref.cast %reinterpret_cast_0 : memref<512x4096xf16, strided<[4096, 1]>> to memref<512x4096xf16, strided<[4096, 1], offset: ?>>
        loom.copy %cast, %15 src_mem_space @DRAM dst_mem_space @L1, interconnect : [@horizontal_links, @vertical_links], broadcast : [8, 8] : memref<512x4096xf16, strided<[4096, 1], offset: ?>>, memref<512x4096xf16>
        linalg.matmul ins(%17, %15 : memref<64x512xf16>, memref<512x4096xf16>) outs(%19 : memref<64x4096xf16>)
        loom.semaphore_give %15 : memref<512x4096xf16>
        loom.semaphore_give %17 : memref<64x512xf16>
        %21 = arith.muli %13, %c262144 : index
        %reinterpret_cast_1 = memref.reinterpret_cast %arg2 to offset: [%21], sizes: [64, 4096], strides: [4096, 1] : memref<4096x4096xf16> to memref<64x4096xf16, strided<[4096, 1], offset: ?>>
        loom.copy %19, %reinterpret_cast_1 src_mem_space @L1 dst_mem_space @DRAM, interconnect : [], broadcast : [1, 1] : memref<64x4096xf16>, memref<64x4096xf16, strided<[4096, 1], offset: ?>>
        loom.semaphore_give %19 : memref<64x4096xf16>
        scf.reduce 
      } {loom.iter_types = [#loom.iter_type<spatial>, #loom.iter_type<spatial>], loom.mapped_to_dims = [@y, @x]}
      return
    }
  }
  module attributes {loom.pass_name = "Materialize"} {
    func.func @matmul__d1i0_d0i0__f01__d_h__BK64__BM64__BN4096(%arg0: memref<4096x512xf16>, %arg1: memref<512x4096xf16>, %arg2: memref<4096x4096xf16>) {
      %c262144 = arith.constant 262144 : index
      %c32768 = arith.constant 32768 : index
      %cst = arith.constant 0.000000e+00 : f16
      %c64 = arith.constant 64 : index
      %c0 = arith.constant 0 : index
      %c8 = arith.constant 8 : index
      %c1 = arith.constant 1 : index
      scf.parallel (%arg3, %arg4) = (%c0, %c0) to (%c8, %c8) step (%c1, %c1) {
        %12 = arith.muli %arg3, %c8 overflow<nsw> : index
        %13 = arith.addi %12, %arg4 : index
        %14 = loom.alloc [64, 4096] on @L1 : memref<64x4096xf16>
        %15 = loom.semaphore_take %14 : memref<64x4096xf16> -> memref<64x4096xf16>
        %16 = loom.alloc [64, 64] on @L1 : memref<64x64xf16>
        %17 = loom.semaphore_take %16 : memref<64x64xf16> -> memref<64x64xf16>
        %18 = loom.alloc [64, 4096] on @L1 : memref<64x4096xf16>
        %19 = loom.semaphore_take %18 : memref<64x4096xf16> -> memref<64x4096xf16>
        linalg.fill ins(%cst : f16) outs(%19 : memref<64x4096xf16>)
        scf.for %arg5 = %c0 to %c8 step %c1 {
          %21 = arith.muli %arg5, %c64 : index
          %22 = arith.muli %13, %c32768 : index
          %23 = arith.addi %22, %21 : index
          %reinterpret_cast_0 = memref.reinterpret_cast %arg0 to offset: [%23], sizes: [64, 64], strides: [512, 1] : memref<4096x512xf16> to memref<64x64xf16, strided<[512, 1], offset: ?>>
          loom.copy %reinterpret_cast_0, %17 src_mem_space @DRAM dst_mem_space @L1, interconnect : [], broadcast : [1, 1] : memref<64x64xf16, strided<[512, 1], offset: ?>>, memref<64x64xf16>
          %24 = arith.muli %arg5, %c262144 : index
          %reinterpret_cast_1 = memref.reinterpret_cast %arg1 to offset: [%24], sizes: [64, 4096], strides: [4096, 1] : memref<512x4096xf16> to memref<64x4096xf16, strided<[4096, 1], offset: ?>>
          loom.copy %reinterpret_cast_1, %15 src_mem_space @DRAM dst_mem_space @L1, interconnect : [@horizontal_links], broadcast : [1, 8] : memref<64x4096xf16, strided<[4096, 1], offset: ?>>, memref<64x4096xf16>
          linalg.matmul ins(%17, %15 : memref<64x64xf16>, memref<64x4096xf16>) outs(%19 : memref<64x4096xf16>)
          loom.semaphore_give %15 : memref<64x4096xf16>
          loom.semaphore_give %17 : memref<64x64xf16>
        }
        %20 = arith.muli %13, %c262144 : index
        %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%20], sizes: [64, 4096], strides: [4096, 1] : memref<4096x4096xf16> to memref<64x4096xf16, strided<[4096, 1], offset: ?>>
        loom.copy %19, %reinterpret_cast src_mem_space @L1 dst_mem_space @DRAM, interconnect : [], broadcast : [1, 1] : memref<64x4096xf16>, memref<64x4096xf16, strided<[4096, 1], offset: ?>>
        loom.semaphore_give %19 : memref<64x4096xf16>
        scf.reduce 
      } {loom.iter_types = [#loom.iter_type<spatial>, #loom.iter_type<spatial>], loom.mapped_to_dims = [@y, @x]}
      return
    }
  }
  module attributes {loom.pass_name = "Materialize"} {
    func.func @matmul__d1i0_d0i0__f01__d_v__BK256__BM64__BN1024(%arg0: memref<4096x512xf16>, %arg1: memref<512x4096xf16>, %arg2: memref<4096x4096xf16>) {
      %c262144 = arith.constant 262144 : index
      %c1048576 = arith.constant 1048576 : index
      %c32768 = arith.constant 32768 : index
      %c2 = arith.constant 2 : index
      %c4 = arith.constant 4 : index
      %cst = arith.constant 0.000000e+00 : f16
      %c1024 = arith.constant 1024 : index
      %c256 = arith.constant 256 : index
      %c0 = arith.constant 0 : index
      %c8 = arith.constant 8 : index
      %c1 = arith.constant 1 : index
      scf.parallel (%arg3, %arg4) = (%c0, %c0) to (%c8, %c8) step (%c1, %c1) {
        scf.for %arg5 = %c0 to %c4 step %c1 {
          %12 = arith.muli %arg3, %c8 overflow<nsw> : index
          %13 = arith.addi %12, %arg4 : index
          %14 = loom.alloc [256, 1024] on @L1 : memref<256x1024xf16>
          %15 = loom.semaphore_take %14 : memref<256x1024xf16> -> memref<256x1024xf16>
          %16 = loom.alloc [64, 256] on @L1 : memref<64x256xf16>
          %17 = loom.semaphore_take %16 : memref<64x256xf16> -> memref<64x256xf16>
          %18 = loom.alloc [64, 1024] on @L1 : memref<64x1024xf16>
          %19 = loom.semaphore_take %18 : memref<64x1024xf16> -> memref<64x1024xf16>
          linalg.fill ins(%cst : f16) outs(%19 : memref<64x1024xf16>)
          scf.for %arg6 = %c0 to %c2 step %c1 {
            %23 = arith.muli %arg6, %c256 : index
            %24 = arith.muli %13, %c32768 : index
            %25 = arith.addi %24, %23 : index
            %reinterpret_cast_0 = memref.reinterpret_cast %arg0 to offset: [%25], sizes: [64, 256], strides: [512, 1] : memref<4096x512xf16> to memref<64x256xf16, strided<[512, 1], offset: ?>>
            loom.copy %reinterpret_cast_0, %17 src_mem_space @DRAM dst_mem_space @L1, interconnect : [], broadcast : [1, 1] : memref<64x256xf16, strided<[512, 1], offset: ?>>, memref<64x256xf16>
            %26 = arith.muli %arg5, %c1024 : index
            %27 = arith.muli %arg6, %c1048576 : index
            %28 = arith.addi %27, %26 : index
            %reinterpret_cast_1 = memref.reinterpret_cast %arg1 to offset: [%28], sizes: [256, 1024], strides: [4096, 1] : memref<512x4096xf16> to memref<256x1024xf16, strided<[4096, 1], offset: ?>>
            loom.copy %reinterpret_cast_1, %15 src_mem_space @DRAM dst_mem_space @L1, interconnect : [@vertical_links], broadcast : [8, 1] : memref<256x1024xf16, strided<[4096, 1], offset: ?>>, memref<256x1024xf16>
            linalg.matmul ins(%17, %15 : memref<64x256xf16>, memref<256x1024xf16>) outs(%19 : memref<64x1024xf16>)
            loom.semaphore_give %15 : memref<256x1024xf16>
            loom.semaphore_give %17 : memref<64x256xf16>
          }
          %20 = arith.muli %arg5, %c1024 : index
          %21 = arith.muli %13, %c262144 : index
          %22 = arith.addi %21, %20 : index
          %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%22], sizes: [64, 1024], strides: [4096, 1] : memref<4096x4096xf16> to memref<64x1024xf16, strided<[4096, 1], offset: ?>>
          loom.copy %19, %reinterpret_cast src_mem_space @L1 dst_mem_space @DRAM, interconnect : [], broadcast : [1, 1] : memref<64x1024xf16>, memref<64x1024xf16, strided<[4096, 1], offset: ?>>
          loom.semaphore_give %19 : memref<64x1024xf16>
        }
        scf.reduce 
      } {loom.iter_types = [#loom.iter_type<spatial>, #loom.iter_type<spatial>], loom.mapped_to_dims = [@y, @x]}
      return
    }
  }
  module attributes {loom.pass_name = "Materialize"} {
    func.func @matmul__d1i0_d0i0__f10__d_d__BK512__BM64__BN4096(%arg0: memref<4096x512xf16>, %arg1: memref<512x4096xf16>, %arg2: memref<4096x4096xf16>) {
      %c262144 = arith.constant 262144 : index
      %c32768 = arith.constant 32768 : index
      %cst = arith.constant 0.000000e+00 : f16
      %c0 = arith.constant 0 : index
      %c8 = arith.constant 8 : index
      %c1 = arith.constant 1 : index
      scf.parallel (%arg3, %arg4) = (%c0, %c0) to (%c8, %c8) step (%c1, %c1) {
        %12 = arith.muli %arg3, %c8 overflow<nsw> : index
        %13 = arith.addi %12, %arg4 : index
        %14 = loom.alloc [512, 4096] on @L1 : memref<512x4096xf16>
        %15 = loom.semaphore_take %14 : memref<512x4096xf16> -> memref<512x4096xf16>
        %16 = loom.alloc [64, 512] on @L1 : memref<64x512xf16>
        %17 = loom.semaphore_take %16 : memref<64x512xf16> -> memref<64x512xf16>
        %18 = loom.alloc [64, 4096] on @L1 : memref<64x4096xf16>
        %19 = loom.semaphore_take %18 : memref<64x4096xf16> -> memref<64x4096xf16>
        linalg.fill ins(%cst : f16) outs(%19 : memref<64x4096xf16>)
        %20 = arith.muli %13, %c32768 : index
        %reinterpret_cast = memref.reinterpret_cast %arg0 to offset: [%20], sizes: [64, 512], strides: [512, 1] : memref<4096x512xf16> to memref<64x512xf16, strided<[512, 1], offset: ?>>
        loom.copy %reinterpret_cast, %17 src_mem_space @DRAM dst_mem_space @L1, interconnect : [], broadcast : [1, 1] : memref<64x512xf16, strided<[512, 1], offset: ?>>, memref<64x512xf16>
        %reinterpret_cast_0 = memref.reinterpret_cast %arg1 to offset: [0], sizes: [512, 4096], strides: [4096, 1] : memref<512x4096xf16> to memref<512x4096xf16, strided<[4096, 1]>>
        %cast = memref.cast %reinterpret_cast_0 : memref<512x4096xf16, strided<[4096, 1]>> to memref<512x4096xf16, strided<[4096, 1], offset: ?>>
        loom.copy %cast, %15 src_mem_space @DRAM dst_mem_space @L1, interconnect : [], broadcast : [1, 1] : memref<512x4096xf16, strided<[4096, 1], offset: ?>>, memref<512x4096xf16>
        linalg.matmul ins(%17, %15 : memref<64x512xf16>, memref<512x4096xf16>) outs(%19 : memref<64x4096xf16>)
        loom.semaphore_give %15 : memref<512x4096xf16>
        loom.semaphore_give %17 : memref<64x512xf16>
        %21 = arith.muli %13, %c262144 : index
        %reinterpret_cast_1 = memref.reinterpret_cast %arg2 to offset: [%21], sizes: [64, 4096], strides: [4096, 1] : memref<4096x4096xf16> to memref<64x4096xf16, strided<[4096, 1], offset: ?>>
        loom.copy %19, %reinterpret_cast_1 src_mem_space @L1 dst_mem_space @DRAM, interconnect : [], broadcast : [1, 1] : memref<64x4096xf16>, memref<64x4096xf16, strided<[4096, 1], offset: ?>>
        loom.semaphore_give %19 : memref<64x4096xf16>
        scf.reduce 
      } {loom.iter_types = [#loom.iter_type<spatial>, #loom.iter_type<spatial>], loom.mapped_to_dims = [@y, @x]}
      return
    }
  }
  module attributes {loom.pass_name = "Materialize"} {
    func.func @matmul__d1i0_d0i0__f10__d_a__BK512__BM64__BN4096(%arg0: memref<4096x512xf16>, %arg1: memref<512x4096xf16>, %arg2: memref<4096x4096xf16>) {
      %c262144 = arith.constant 262144 : index
      %c32768 = arith.constant 32768 : index
      %cst = arith.constant 0.000000e+00 : f16
      %c0 = arith.constant 0 : index
      %c8 = arith.constant 8 : index
      %c1 = arith.constant 1 : index
      scf.parallel (%arg3, %arg4) = (%c0, %c0) to (%c8, %c8) step (%c1, %c1) {
        %12 = arith.muli %arg3, %c8 overflow<nsw> : index
        %13 = arith.addi %12, %arg4 : index
        %14 = loom.alloc [512, 4096] on @L1 : memref<512x4096xf16>
        %15 = loom.semaphore_take %14 : memref<512x4096xf16> -> memref<512x4096xf16>
        %16 = loom.alloc [64, 512] on @L1 : memref<64x512xf16>
        %17 = loom.semaphore_take %16 : memref<64x512xf16> -> memref<64x512xf16>
        %18 = loom.alloc [64, 4096] on @L1 : memref<64x4096xf16>
        %19 = loom.semaphore_take %18 : memref<64x4096xf16> -> memref<64x4096xf16>
        linalg.fill ins(%cst : f16) outs(%19 : memref<64x4096xf16>)
        %20 = arith.muli %13, %c32768 : index
        %reinterpret_cast = memref.reinterpret_cast %arg0 to offset: [%20], sizes: [64, 512], strides: [512, 1] : memref<4096x512xf16> to memref<64x512xf16, strided<[512, 1], offset: ?>>
        loom.copy %reinterpret_cast, %17 src_mem_space @DRAM dst_mem_space @L1, interconnect : [], broadcast : [1, 1] : memref<64x512xf16, strided<[512, 1], offset: ?>>, memref<64x512xf16>
        %reinterpret_cast_0 = memref.reinterpret_cast %arg1 to offset: [0], sizes: [512, 4096], strides: [4096, 1] : memref<512x4096xf16> to memref<512x4096xf16, strided<[4096, 1]>>
        %cast = memref.cast %reinterpret_cast_0 : memref<512x4096xf16, strided<[4096, 1]>> to memref<512x4096xf16, strided<[4096, 1], offset: ?>>
        loom.copy %cast, %15 src_mem_space @DRAM dst_mem_space @L1, interconnect : [@horizontal_links, @vertical_links], broadcast : [8, 8] : memref<512x4096xf16, strided<[4096, 1], offset: ?>>, memref<512x4096xf16>
        linalg.matmul ins(%17, %15 : memref<64x512xf16>, memref<512x4096xf16>) outs(%19 : memref<64x4096xf16>)
        loom.semaphore_give %15 : memref<512x4096xf16>
        loom.semaphore_give %17 : memref<64x512xf16>
        %21 = arith.muli %13, %c262144 : index
        %reinterpret_cast_1 = memref.reinterpret_cast %arg2 to offset: [%21], sizes: [64, 4096], strides: [4096, 1] : memref<4096x4096xf16> to memref<64x4096xf16, strided<[4096, 1], offset: ?>>
        loom.copy %19, %reinterpret_cast_1 src_mem_space @L1 dst_mem_space @DRAM, interconnect : [], broadcast : [1, 1] : memref<64x4096xf16>, memref<64x4096xf16, strided<[4096, 1], offset: ?>>
        loom.semaphore_give %19 : memref<64x4096xf16>
        scf.reduce 
      } {loom.iter_types = [#loom.iter_type<spatial>, #loom.iter_type<spatial>], loom.mapped_to_dims = [@y, @x]}
      return
    }
  }
  module attributes {loom.pass_name = "Materialize"} {
    func.func @matmul__d1i0_d0i0__f10__d_h__BK32__BM64__BN4096(%arg0: memref<4096x512xf16>, %arg1: memref<512x4096xf16>, %arg2: memref<4096x4096xf16>) {
      %c262144 = arith.constant 262144 : index
      %c131072 = arith.constant 131072 : index
      %c32768 = arith.constant 32768 : index
      %c16 = arith.constant 16 : index
      %cst = arith.constant 0.000000e+00 : f16
      %c32 = arith.constant 32 : index
      %c0 = arith.constant 0 : index
      %c8 = arith.constant 8 : index
      %c1 = arith.constant 1 : index
      scf.parallel (%arg3, %arg4) = (%c0, %c0) to (%c8, %c8) step (%c1, %c1) {
        %12 = arith.muli %arg3, %c8 overflow<nsw> : index
        %13 = arith.addi %12, %arg4 : index
        %14 = loom.alloc [32, 4096] on @L1 : memref<32x4096xf16>
        %15 = loom.semaphore_take %14 : memref<32x4096xf16> -> memref<32x4096xf16>
        %16 = loom.alloc [64, 32] on @L1 : memref<64x32xf16>
        %17 = loom.semaphore_take %16 : memref<64x32xf16> -> memref<64x32xf16>
        %18 = loom.alloc [64, 4096] on @L1 : memref<64x4096xf16>
        %19 = loom.semaphore_take %18 : memref<64x4096xf16> -> memref<64x4096xf16>
        linalg.fill ins(%cst : f16) outs(%19 : memref<64x4096xf16>)
        scf.for %arg5 = %c0 to %c16 step %c1 {
          %21 = arith.muli %arg5, %c32 : index
          %22 = arith.muli %13, %c32768 : index
          %23 = arith.addi %22, %21 : index
          %reinterpret_cast_0 = memref.reinterpret_cast %arg0 to offset: [%23], sizes: [64, 32], strides: [512, 1] : memref<4096x512xf16> to memref<64x32xf16, strided<[512, 1], offset: ?>>
          loom.copy %reinterpret_cast_0, %17 src_mem_space @DRAM dst_mem_space @L1, interconnect : [], broadcast : [1, 1] : memref<64x32xf16, strided<[512, 1], offset: ?>>, memref<64x32xf16>
          %24 = arith.muli %arg5, %c131072 : index
          %reinterpret_cast_1 = memref.reinterpret_cast %arg1 to offset: [%24], sizes: [32, 4096], strides: [4096, 1] : memref<512x4096xf16> to memref<32x4096xf16, strided<[4096, 1], offset: ?>>
          loom.copy %reinterpret_cast_1, %15 src_mem_space @DRAM dst_mem_space @L1, interconnect : [@horizontal_links], broadcast : [1, 8] : memref<32x4096xf16, strided<[4096, 1], offset: ?>>, memref<32x4096xf16>
          linalg.matmul ins(%17, %15 : memref<64x32xf16>, memref<32x4096xf16>) outs(%19 : memref<64x4096xf16>)
          loom.semaphore_give %15 : memref<32x4096xf16>
          loom.semaphore_give %17 : memref<64x32xf16>
        }
        %20 = arith.muli %13, %c262144 : index
        %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%20], sizes: [64, 4096], strides: [4096, 1] : memref<4096x4096xf16> to memref<64x4096xf16, strided<[4096, 1], offset: ?>>
        loom.copy %19, %reinterpret_cast src_mem_space @L1 dst_mem_space @DRAM, interconnect : [], broadcast : [1, 1] : memref<64x4096xf16>, memref<64x4096xf16, strided<[4096, 1], offset: ?>>
        loom.semaphore_give %19 : memref<64x4096xf16>
        scf.reduce 
      } {loom.iter_types = [#loom.iter_type<spatial>, #loom.iter_type<spatial>], loom.mapped_to_dims = [@y, @x]}
      return
    }
  }
  module attributes {loom.pass_name = "Materialize"} {
    func.func @matmul__d1i0_d0i0__f10__d_v__BK128__BM64__BN4096(%arg0: memref<4096x512xf16>, %arg1: memref<512x4096xf16>, %arg2: memref<4096x4096xf16>) {
      %c262144 = arith.constant 262144 : index
      %c524288 = arith.constant 524288 : index
      %c32768 = arith.constant 32768 : index
      %c4 = arith.constant 4 : index
      %cst = arith.constant 0.000000e+00 : f16
      %c128 = arith.constant 128 : index
      %c0 = arith.constant 0 : index
      %c8 = arith.constant 8 : index
      %c1 = arith.constant 1 : index
      scf.parallel (%arg3, %arg4) = (%c0, %c0) to (%c8, %c8) step (%c1, %c1) {
        %12 = arith.muli %arg3, %c8 overflow<nsw> : index
        %13 = arith.addi %12, %arg4 : index
        %14 = loom.alloc [128, 4096] on @L1 : memref<128x4096xf16>
        %15 = loom.semaphore_take %14 : memref<128x4096xf16> -> memref<128x4096xf16>
        %16 = loom.alloc [64, 128] on @L1 : memref<64x128xf16>
        %17 = loom.semaphore_take %16 : memref<64x128xf16> -> memref<64x128xf16>
        %18 = loom.alloc [64, 4096] on @L1 : memref<64x4096xf16>
        %19 = loom.semaphore_take %18 : memref<64x4096xf16> -> memref<64x4096xf16>
        linalg.fill ins(%cst : f16) outs(%19 : memref<64x4096xf16>)
        scf.for %arg5 = %c0 to %c4 step %c1 {
          %21 = arith.muli %arg5, %c128 : index
          %22 = arith.muli %13, %c32768 : index
          %23 = arith.addi %22, %21 : index
          %reinterpret_cast_0 = memref.reinterpret_cast %arg0 to offset: [%23], sizes: [64, 128], strides: [512, 1] : memref<4096x512xf16> to memref<64x128xf16, strided<[512, 1], offset: ?>>
          loom.copy %reinterpret_cast_0, %17 src_mem_space @DRAM dst_mem_space @L1, interconnect : [], broadcast : [1, 1] : memref<64x128xf16, strided<[512, 1], offset: ?>>, memref<64x128xf16>
          %24 = arith.muli %arg5, %c524288 : index
          %reinterpret_cast_1 = memref.reinterpret_cast %arg1 to offset: [%24], sizes: [128, 4096], strides: [4096, 1] : memref<512x4096xf16> to memref<128x4096xf16, strided<[4096, 1], offset: ?>>
          loom.copy %reinterpret_cast_1, %15 src_mem_space @DRAM dst_mem_space @L1, interconnect : [@vertical_links], broadcast : [8, 1] : memref<128x4096xf16, strided<[4096, 1], offset: ?>>, memref<128x4096xf16>
          linalg.matmul ins(%17, %15 : memref<64x128xf16>, memref<128x4096xf16>) outs(%19 : memref<64x4096xf16>)
          loom.semaphore_give %15 : memref<128x4096xf16>
          loom.semaphore_give %17 : memref<64x128xf16>
        }
        %20 = arith.muli %13, %c262144 : index
        %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%20], sizes: [64, 4096], strides: [4096, 1] : memref<4096x4096xf16> to memref<64x4096xf16, strided<[4096, 1], offset: ?>>
        loom.copy %19, %reinterpret_cast src_mem_space @L1 dst_mem_space @DRAM, interconnect : [], broadcast : [1, 1] : memref<64x4096xf16>, memref<64x4096xf16, strided<[4096, 1], offset: ?>>
        loom.semaphore_give %19 : memref<64x4096xf16>
        scf.reduce 
      } {loom.iter_types = [#loom.iter_type<spatial>, #loom.iter_type<spatial>], loom.mapped_to_dims = [@y, @x]}
      return
    }
  }
  module attributes {loom.pass_name = "Materialize"} {
    func.func @matmul__d0i0_d1i1__f01__d_d__BK32__BM512__BN256(%arg0: memref<4096x512xf16>, %arg1: memref<512x4096xf16>, %arg2: memref<4096x4096xf16>) {
      %c2097152 = arith.constant 2097152 : index
      %c131072 = arith.constant 131072 : index
      %c262144 = arith.constant 262144 : index
      %c16 = arith.constant 16 : index
      %c2 = arith.constant 2 : index
      %cst = arith.constant 0.000000e+00 : f16
      %c256 = arith.constant 256 : index
      %c32 = arith.constant 32 : index
      %c0 = arith.constant 0 : index
      %c8 = arith.constant 8 : index
      %c1 = arith.constant 1 : index
      scf.parallel (%arg3, %arg4) = (%c0, %c0) to (%c8, %c8) step (%c1, %c1) {
        scf.for %arg5 = %c0 to %c2 step %c1 {
          %12 = arith.muli %arg5, %c8 overflow<nsw> : index
          %13 = arith.addi %arg4, %12 : index
          %14 = loom.alloc [32, 256] on @L1 : memref<32x256xf16>
          %15 = loom.semaphore_take %14 : memref<32x256xf16> -> memref<32x256xf16>
          %16 = loom.alloc [512, 32] on @L1 : memref<512x32xf16>
          %17 = loom.semaphore_take %16 : memref<512x32xf16> -> memref<512x32xf16>
          %18 = loom.alloc [512, 256] on @L1 : memref<512x256xf16>
          %19 = loom.semaphore_take %18 : memref<512x256xf16> -> memref<512x256xf16>
          linalg.fill ins(%cst : f16) outs(%19 : memref<512x256xf16>)
          scf.for %arg6 = %c0 to %c16 step %c1 {
            %23 = arith.muli %arg6, %c32 : index
            %24 = arith.muli %arg3, %c262144 : index
            %25 = arith.addi %24, %23 : index
            %reinterpret_cast_0 = memref.reinterpret_cast %arg0 to offset: [%25], sizes: [512, 32], strides: [512, 1] : memref<4096x512xf16> to memref<512x32xf16, strided<[512, 1], offset: ?>>
            loom.copy %reinterpret_cast_0, %17 src_mem_space @DRAM dst_mem_space @L1, interconnect : [], broadcast : [1, 1] : memref<512x32xf16, strided<[512, 1], offset: ?>>, memref<512x32xf16>
            %26 = arith.muli %13, %c256 : index
            %27 = arith.muli %arg6, %c131072 : index
            %28 = arith.addi %27, %26 : index
            %reinterpret_cast_1 = memref.reinterpret_cast %arg1 to offset: [%28], sizes: [32, 256], strides: [4096, 1] : memref<512x4096xf16> to memref<32x256xf16, strided<[4096, 1], offset: ?>>
            loom.copy %reinterpret_cast_1, %15 src_mem_space @DRAM dst_mem_space @L1, interconnect : [], broadcast : [1, 1] : memref<32x256xf16, strided<[4096, 1], offset: ?>>, memref<32x256xf16>
            linalg.matmul ins(%17, %15 : memref<512x32xf16>, memref<32x256xf16>) outs(%19 : memref<512x256xf16>)
            loom.semaphore_give %15 : memref<32x256xf16>
            loom.semaphore_give %17 : memref<512x32xf16>
          }
          %20 = arith.muli %13, %c256 : index
          %21 = arith.muli %arg3, %c2097152 : index
          %22 = arith.addi %21, %20 : index
          %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%22], sizes: [512, 256], strides: [4096, 1] : memref<4096x4096xf16> to memref<512x256xf16, strided<[4096, 1], offset: ?>>
          loom.copy %19, %reinterpret_cast src_mem_space @L1 dst_mem_space @DRAM, interconnect : [], broadcast : [1, 1] : memref<512x256xf16>, memref<512x256xf16, strided<[4096, 1], offset: ?>>
          loom.semaphore_give %19 : memref<512x256xf16>
        }
        scf.reduce 
      } {loom.iter_types = [#loom.iter_type<spatial>, #loom.iter_type<spatial>], loom.mapped_to_dims = [@x, @y]}
      return
    }
  }
  module attributes {loom.pass_name = "Materialize"} {
    func.func @matmul__d0i0_d1i1__f01__d_h__BK64__BM512__BN128(%arg0: memref<4096x512xf16>, %arg1: memref<512x4096xf16>, %arg2: memref<4096x4096xf16>) {
      %c2097152 = arith.constant 2097152 : index
      %c262144 = arith.constant 262144 : index
      %c4 = arith.constant 4 : index
      %cst = arith.constant 0.000000e+00 : f16
      %c128 = arith.constant 128 : index
      %c64 = arith.constant 64 : index
      %c0 = arith.constant 0 : index
      %c8 = arith.constant 8 : index
      %c1 = arith.constant 1 : index
      scf.parallel (%arg3, %arg4) = (%c0, %c0) to (%c8, %c8) step (%c1, %c1) {
        scf.for %arg5 = %c0 to %c4 step %c1 {
          %12 = arith.muli %arg5, %c8 overflow<nsw> : index
          %13 = arith.addi %arg4, %12 : index
          %14 = loom.alloc [64, 128] on @L1 : memref<64x128xf16>
          %15 = loom.semaphore_take %14 : memref<64x128xf16> -> memref<64x128xf16>
          %16 = loom.alloc [512, 64] on @L1 : memref<512x64xf16>
          %17 = loom.semaphore_take %16 : memref<512x64xf16> -> memref<512x64xf16>
          %18 = loom.alloc [512, 128] on @L1 : memref<512x128xf16>
          %19 = loom.semaphore_take %18 : memref<512x128xf16> -> memref<512x128xf16>
          linalg.fill ins(%cst : f16) outs(%19 : memref<512x128xf16>)
          scf.for %arg6 = %c0 to %c8 step %c1 {
            %23 = arith.muli %arg6, %c64 : index
            %24 = arith.muli %arg3, %c262144 : index
            %25 = arith.addi %24, %23 : index
            %reinterpret_cast_0 = memref.reinterpret_cast %arg0 to offset: [%25], sizes: [512, 64], strides: [512, 1] : memref<4096x512xf16> to memref<512x64xf16, strided<[512, 1], offset: ?>>
            loom.copy %reinterpret_cast_0, %17 src_mem_space @DRAM dst_mem_space @L1, interconnect : [], broadcast : [1, 1] : memref<512x64xf16, strided<[512, 1], offset: ?>>, memref<512x64xf16>
            %26 = arith.muli %13, %c128 : index
            %27 = arith.muli %arg6, %c262144 : index
            %28 = arith.addi %27, %26 : index
            %reinterpret_cast_1 = memref.reinterpret_cast %arg1 to offset: [%28], sizes: [64, 128], strides: [4096, 1] : memref<512x4096xf16> to memref<64x128xf16, strided<[4096, 1], offset: ?>>
            loom.copy %reinterpret_cast_1, %15 src_mem_space @DRAM dst_mem_space @L1, interconnect : [@horizontal_links], broadcast : [1, 8] : memref<64x128xf16, strided<[4096, 1], offset: ?>>, memref<64x128xf16>
            linalg.matmul ins(%17, %15 : memref<512x64xf16>, memref<64x128xf16>) outs(%19 : memref<512x128xf16>)
            loom.semaphore_give %15 : memref<64x128xf16>
            loom.semaphore_give %17 : memref<512x64xf16>
          }
          %20 = arith.muli %13, %c128 : index
          %21 = arith.muli %arg3, %c2097152 : index
          %22 = arith.addi %21, %20 : index
          %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%22], sizes: [512, 128], strides: [4096, 1] : memref<4096x4096xf16> to memref<512x128xf16, strided<[4096, 1], offset: ?>>
          loom.copy %19, %reinterpret_cast src_mem_space @L1 dst_mem_space @DRAM, interconnect : [], broadcast : [1, 1] : memref<512x128xf16>, memref<512x128xf16, strided<[4096, 1], offset: ?>>
          loom.semaphore_give %19 : memref<512x128xf16>
        }
        scf.reduce 
      } {loom.iter_types = [#loom.iter_type<spatial>, #loom.iter_type<spatial>], loom.mapped_to_dims = [@x, @y]}
      return
    }
  }
  module attributes {loom.pass_name = "Materialize"} {
    func.func @matmul__d0i0_d1i1__f01__v_d__BK512__BM256__BN512(%arg0: memref<4096x512xf16>, %arg1: memref<512x4096xf16>, %arg2: memref<4096x4096xf16>) {
      %c1048576 = arith.constant 1048576 : index
      %c131072 = arith.constant 131072 : index
      %c2 = arith.constant 2 : index
      %cst = arith.constant 0.000000e+00 : f16
      %c512 = arith.constant 512 : index
      %c0 = arith.constant 0 : index
      %c8 = arith.constant 8 : index
      %c1 = arith.constant 1 : index
      scf.parallel (%arg3, %arg4) = (%c0, %c0) to (%c8, %c8) step (%c1, %c1) {
        scf.for %arg5 = %c0 to %c2 step %c1 {
          %12 = arith.muli %arg5, %c8 overflow<nsw> : index
          %13 = arith.addi %arg3, %12 : index
          %14 = loom.alloc [512, 512] on @L1 : memref<512x512xf16>
          %15 = loom.semaphore_take %14 : memref<512x512xf16> -> memref<512x512xf16>
          %16 = loom.alloc [256, 512] on @L1 : memref<256x512xf16>
          %17 = loom.semaphore_take %16 : memref<256x512xf16> -> memref<256x512xf16>
          %18 = loom.alloc [256, 512] on @L1 : memref<256x512xf16>
          %19 = loom.semaphore_take %18 : memref<256x512xf16> -> memref<256x512xf16>
          linalg.fill ins(%cst : f16) outs(%19 : memref<256x512xf16>)
          %20 = arith.muli %13, %c131072 : index
          %reinterpret_cast = memref.reinterpret_cast %arg0 to offset: [%20], sizes: [256, 512], strides: [512, 1] : memref<4096x512xf16> to memref<256x512xf16, strided<[512, 1], offset: ?>>
          loom.copy %reinterpret_cast, %17 src_mem_space @DRAM dst_mem_space @L1, interconnect : [@vertical_links], broadcast : [8, 1] : memref<256x512xf16, strided<[512, 1], offset: ?>>, memref<256x512xf16>
          %21 = arith.muli %arg4, %c512 : index
          %reinterpret_cast_0 = memref.reinterpret_cast %arg1 to offset: [%21], sizes: [512, 512], strides: [4096, 1] : memref<512x4096xf16> to memref<512x512xf16, strided<[4096, 1], offset: ?>>
          loom.copy %reinterpret_cast_0, %15 src_mem_space @DRAM dst_mem_space @L1, interconnect : [], broadcast : [1, 1] : memref<512x512xf16, strided<[4096, 1], offset: ?>>, memref<512x512xf16>
          linalg.matmul ins(%17, %15 : memref<256x512xf16>, memref<512x512xf16>) outs(%19 : memref<256x512xf16>)
          loom.semaphore_give %15 : memref<512x512xf16>
          loom.semaphore_give %17 : memref<256x512xf16>
          %22 = arith.muli %13, %c1048576 : index
          %23 = arith.addi %22, %21 : index
          %reinterpret_cast_1 = memref.reinterpret_cast %arg2 to offset: [%23], sizes: [256, 512], strides: [4096, 1] : memref<4096x4096xf16> to memref<256x512xf16, strided<[4096, 1], offset: ?>>
          loom.copy %19, %reinterpret_cast_1 src_mem_space @L1 dst_mem_space @DRAM, interconnect : [], broadcast : [1, 1] : memref<256x512xf16>, memref<256x512xf16, strided<[4096, 1], offset: ?>>
          loom.semaphore_give %19 : memref<256x512xf16>
        }
        scf.reduce 
      } {loom.iter_types = [#loom.iter_type<spatial>, #loom.iter_type<spatial>], loom.mapped_to_dims = [@x, @y]}
      return
    }
  }
  module attributes {loom.pass_name = "Materialize"} {
    func.func @matmul__d0i0_d1i1__f01__v_h__BK256__BM128__BN512(%arg0: memref<4096x512xf16>, %arg1: memref<512x4096xf16>, %arg2: memref<4096x4096xf16>) {
      %c524288 = arith.constant 524288 : index
      %c1048576 = arith.constant 1048576 : index
      %c65536 = arith.constant 65536 : index
      %c2 = arith.constant 2 : index
      %c4 = arith.constant 4 : index
      %cst = arith.constant 0.000000e+00 : f16
      %c512 = arith.constant 512 : index
      %c256 = arith.constant 256 : index
      %c0 = arith.constant 0 : index
      %c8 = arith.constant 8 : index
      %c1 = arith.constant 1 : index
      scf.parallel (%arg3, %arg4) = (%c0, %c0) to (%c8, %c8) step (%c1, %c1) {
        scf.for %arg5 = %c0 to %c4 step %c1 {
          %12 = arith.muli %arg5, %c8 overflow<nsw> : index
          %13 = arith.addi %arg3, %12 : index
          %14 = loom.alloc [256, 512] on @L1 : memref<256x512xf16>
          %15 = loom.semaphore_take %14 : memref<256x512xf16> -> memref<256x512xf16>
          %16 = loom.alloc [128, 256] on @L1 : memref<128x256xf16>
          %17 = loom.semaphore_take %16 : memref<128x256xf16> -> memref<128x256xf16>
          %18 = loom.alloc [128, 512] on @L1 : memref<128x512xf16>
          %19 = loom.semaphore_take %18 : memref<128x512xf16> -> memref<128x512xf16>
          linalg.fill ins(%cst : f16) outs(%19 : memref<128x512xf16>)
          scf.for %arg6 = %c0 to %c2 step %c1 {
            %23 = arith.muli %arg6, %c256 : index
            %24 = arith.muli %13, %c65536 : index
            %25 = arith.addi %24, %23 : index
            %reinterpret_cast_0 = memref.reinterpret_cast %arg0 to offset: [%25], sizes: [128, 256], strides: [512, 1] : memref<4096x512xf16> to memref<128x256xf16, strided<[512, 1], offset: ?>>
            loom.copy %reinterpret_cast_0, %17 src_mem_space @DRAM dst_mem_space @L1, interconnect : [@vertical_links], broadcast : [8, 1] : memref<128x256xf16, strided<[512, 1], offset: ?>>, memref<128x256xf16>
            %26 = arith.muli %arg4, %c512 : index
            %27 = arith.muli %arg6, %c1048576 : index
            %28 = arith.addi %27, %26 : index
            %reinterpret_cast_1 = memref.reinterpret_cast %arg1 to offset: [%28], sizes: [256, 512], strides: [4096, 1] : memref<512x4096xf16> to memref<256x512xf16, strided<[4096, 1], offset: ?>>
            loom.copy %reinterpret_cast_1, %15 src_mem_space @DRAM dst_mem_space @L1, interconnect : [@horizontal_links], broadcast : [1, 8] : memref<256x512xf16, strided<[4096, 1], offset: ?>>, memref<256x512xf16>
            linalg.matmul ins(%17, %15 : memref<128x256xf16>, memref<256x512xf16>) outs(%19 : memref<128x512xf16>)
            loom.semaphore_give %15 : memref<256x512xf16>
            loom.semaphore_give %17 : memref<128x256xf16>
          }
          %20 = arith.muli %arg4, %c512 : index
          %21 = arith.muli %13, %c524288 : index
          %22 = arith.addi %21, %20 : index
          %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%22], sizes: [128, 512], strides: [4096, 1] : memref<4096x4096xf16> to memref<128x512xf16, strided<[4096, 1], offset: ?>>
          loom.copy %19, %reinterpret_cast src_mem_space @L1 dst_mem_space @DRAM, interconnect : [], broadcast : [1, 1] : memref<128x512xf16>, memref<128x512xf16, strided<[4096, 1], offset: ?>>
          loom.semaphore_give %19 : memref<128x512xf16>
        }
        scf.reduce 
      } {loom.iter_types = [#loom.iter_type<spatial>, #loom.iter_type<spatial>], loom.mapped_to_dims = [@x, @y]}
      return
    }
  }
  module attributes {loom.pass_name = "Materialize"} {
    func.func @matmul__d0i0_d1i1__f10__d_d__BK32__BM512__BN256(%arg0: memref<4096x512xf16>, %arg1: memref<512x4096xf16>, %arg2: memref<4096x4096xf16>) {
      %c2097152 = arith.constant 2097152 : index
      %c131072 = arith.constant 131072 : index
      %c262144 = arith.constant 262144 : index
      %c16 = arith.constant 16 : index
      %c2 = arith.constant 2 : index
      %cst = arith.constant 0.000000e+00 : f16
      %c256 = arith.constant 256 : index
      %c32 = arith.constant 32 : index
      %c0 = arith.constant 0 : index
      %c8 = arith.constant 8 : index
      %c1 = arith.constant 1 : index
      scf.parallel (%arg3, %arg4) = (%c0, %c0) to (%c8, %c8) step (%c1, %c1) {
        scf.for %arg5 = %c0 to %c2 step %c1 {
          %12 = arith.muli %arg5, %c8 overflow<nsw> : index
          %13 = arith.addi %arg4, %12 : index
          %14 = loom.alloc [32, 256] on @L1 : memref<32x256xf16>
          %15 = loom.semaphore_take %14 : memref<32x256xf16> -> memref<32x256xf16>
          %16 = loom.alloc [512, 32] on @L1 : memref<512x32xf16>
          %17 = loom.semaphore_take %16 : memref<512x32xf16> -> memref<512x32xf16>
          %18 = loom.alloc [512, 256] on @L1 : memref<512x256xf16>
          %19 = loom.semaphore_take %18 : memref<512x256xf16> -> memref<512x256xf16>
          linalg.fill ins(%cst : f16) outs(%19 : memref<512x256xf16>)
          scf.for %arg6 = %c0 to %c16 step %c1 {
            %23 = arith.muli %arg6, %c32 : index
            %24 = arith.muli %arg3, %c262144 : index
            %25 = arith.addi %24, %23 : index
            %reinterpret_cast_0 = memref.reinterpret_cast %arg0 to offset: [%25], sizes: [512, 32], strides: [512, 1] : memref<4096x512xf16> to memref<512x32xf16, strided<[512, 1], offset: ?>>
            loom.copy %reinterpret_cast_0, %17 src_mem_space @DRAM dst_mem_space @L1, interconnect : [], broadcast : [1, 1] : memref<512x32xf16, strided<[512, 1], offset: ?>>, memref<512x32xf16>
            %26 = arith.muli %13, %c256 : index
            %27 = arith.muli %arg6, %c131072 : index
            %28 = arith.addi %27, %26 : index
            %reinterpret_cast_1 = memref.reinterpret_cast %arg1 to offset: [%28], sizes: [32, 256], strides: [4096, 1] : memref<512x4096xf16> to memref<32x256xf16, strided<[4096, 1], offset: ?>>
            loom.copy %reinterpret_cast_1, %15 src_mem_space @DRAM dst_mem_space @L1, interconnect : [], broadcast : [1, 1] : memref<32x256xf16, strided<[4096, 1], offset: ?>>, memref<32x256xf16>
            linalg.matmul ins(%17, %15 : memref<512x32xf16>, memref<32x256xf16>) outs(%19 : memref<512x256xf16>)
            loom.semaphore_give %15 : memref<32x256xf16>
            loom.semaphore_give %17 : memref<512x32xf16>
          }
          %20 = arith.muli %13, %c256 : index
          %21 = arith.muli %arg3, %c2097152 : index
          %22 = arith.addi %21, %20 : index
          %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%22], sizes: [512, 256], strides: [4096, 1] : memref<4096x4096xf16> to memref<512x256xf16, strided<[4096, 1], offset: ?>>
          loom.copy %19, %reinterpret_cast src_mem_space @L1 dst_mem_space @DRAM, interconnect : [], broadcast : [1, 1] : memref<512x256xf16>, memref<512x256xf16, strided<[4096, 1], offset: ?>>
          loom.semaphore_give %19 : memref<512x256xf16>
        }
        scf.reduce 
      } {loom.iter_types = [#loom.iter_type<spatial>, #loom.iter_type<spatial>], loom.mapped_to_dims = [@x, @y]}
      return
    }
  }
  module attributes {loom.pass_name = "Materialize"} {
    func.func @matmul__d0i0_d1i1__f10__d_h__BK64__BM128__BN512(%arg0: memref<4096x512xf16>, %arg1: memref<512x4096xf16>, %arg2: memref<4096x4096xf16>) {
      %c524288 = arith.constant 524288 : index
      %c262144 = arith.constant 262144 : index
      %c65536 = arith.constant 65536 : index
      %c4 = arith.constant 4 : index
      %cst = arith.constant 0.000000e+00 : f16
      %c512 = arith.constant 512 : index
      %c64 = arith.constant 64 : index
      %c0 = arith.constant 0 : index
      %c8 = arith.constant 8 : index
      %c1 = arith.constant 1 : index
      scf.parallel (%arg3, %arg4) = (%c0, %c0) to (%c8, %c8) step (%c1, %c1) {
        scf.for %arg5 = %c0 to %c4 step %c1 {
          %12 = arith.muli %arg5, %c8 overflow<nsw> : index
          %13 = arith.addi %arg3, %12 : index
          %14 = loom.alloc [64, 512] on @L1 : memref<64x512xf16>
          %15 = loom.semaphore_take %14 : memref<64x512xf16> -> memref<64x512xf16>
          %16 = loom.alloc [128, 64] on @L1 : memref<128x64xf16>
          %17 = loom.semaphore_take %16 : memref<128x64xf16> -> memref<128x64xf16>
          %18 = loom.alloc [128, 512] on @L1 : memref<128x512xf16>
          %19 = loom.semaphore_take %18 : memref<128x512xf16> -> memref<128x512xf16>
          linalg.fill ins(%cst : f16) outs(%19 : memref<128x512xf16>)
          scf.for %arg6 = %c0 to %c8 step %c1 {
            %23 = arith.muli %arg6, %c64 : index
            %24 = arith.muli %13, %c65536 : index
            %25 = arith.addi %24, %23 : index
            %reinterpret_cast_0 = memref.reinterpret_cast %arg0 to offset: [%25], sizes: [128, 64], strides: [512, 1] : memref<4096x512xf16> to memref<128x64xf16, strided<[512, 1], offset: ?>>
            loom.copy %reinterpret_cast_0, %17 src_mem_space @DRAM dst_mem_space @L1, interconnect : [], broadcast : [1, 1] : memref<128x64xf16, strided<[512, 1], offset: ?>>, memref<128x64xf16>
            %26 = arith.muli %arg4, %c512 : index
            %27 = arith.muli %arg6, %c262144 : index
            %28 = arith.addi %27, %26 : index
            %reinterpret_cast_1 = memref.reinterpret_cast %arg1 to offset: [%28], sizes: [64, 512], strides: [4096, 1] : memref<512x4096xf16> to memref<64x512xf16, strided<[4096, 1], offset: ?>>
            loom.copy %reinterpret_cast_1, %15 src_mem_space @DRAM dst_mem_space @L1, interconnect : [@horizontal_links], broadcast : [1, 8] : memref<64x512xf16, strided<[4096, 1], offset: ?>>, memref<64x512xf16>
            linalg.matmul ins(%17, %15 : memref<128x64xf16>, memref<64x512xf16>) outs(%19 : memref<128x512xf16>)
            loom.semaphore_give %15 : memref<64x512xf16>
            loom.semaphore_give %17 : memref<128x64xf16>
          }
          %20 = arith.muli %arg4, %c512 : index
          %21 = arith.muli %13, %c524288 : index
          %22 = arith.addi %21, %20 : index
          %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%22], sizes: [128, 512], strides: [4096, 1] : memref<4096x4096xf16> to memref<128x512xf16, strided<[4096, 1], offset: ?>>
          loom.copy %19, %reinterpret_cast src_mem_space @L1 dst_mem_space @DRAM, interconnect : [], broadcast : [1, 1] : memref<128x512xf16>, memref<128x512xf16, strided<[4096, 1], offset: ?>>
          loom.semaphore_give %19 : memref<128x512xf16>
        }
        scf.reduce 
      } {loom.iter_types = [#loom.iter_type<spatial>, #loom.iter_type<spatial>], loom.mapped_to_dims = [@x, @y]}
      return
    }
  }
  module attributes {loom.pass_name = "Materialize"} {
    func.func @matmul__d0i0_d1i1__f10__v_d__BK512__BM256__BN512(%arg0: memref<4096x512xf16>, %arg1: memref<512x4096xf16>, %arg2: memref<4096x4096xf16>) {
      %c1048576 = arith.constant 1048576 : index
      %c131072 = arith.constant 131072 : index
      %c2 = arith.constant 2 : index
      %cst = arith.constant 0.000000e+00 : f16
      %c512 = arith.constant 512 : index
      %c0 = arith.constant 0 : index
      %c8 = arith.constant 8 : index
      %c1 = arith.constant 1 : index
      scf.parallel (%arg3, %arg4) = (%c0, %c0) to (%c8, %c8) step (%c1, %c1) {
        scf.for %arg5 = %c0 to %c2 step %c1 {
          %12 = arith.muli %arg5, %c8 overflow<nsw> : index
          %13 = arith.addi %arg3, %12 : index
          %14 = loom.alloc [512, 512] on @L1 : memref<512x512xf16>
          %15 = loom.semaphore_take %14 : memref<512x512xf16> -> memref<512x512xf16>
          %16 = loom.alloc [256, 512] on @L1 : memref<256x512xf16>
          %17 = loom.semaphore_take %16 : memref<256x512xf16> -> memref<256x512xf16>
          %18 = loom.alloc [256, 512] on @L1 : memref<256x512xf16>
          %19 = loom.semaphore_take %18 : memref<256x512xf16> -> memref<256x512xf16>
          linalg.fill ins(%cst : f16) outs(%19 : memref<256x512xf16>)
          %20 = arith.muli %13, %c131072 : index
          %reinterpret_cast = memref.reinterpret_cast %arg0 to offset: [%20], sizes: [256, 512], strides: [512, 1] : memref<4096x512xf16> to memref<256x512xf16, strided<[512, 1], offset: ?>>
          loom.copy %reinterpret_cast, %17 src_mem_space @DRAM dst_mem_space @L1, interconnect : [@vertical_links], broadcast : [8, 1] : memref<256x512xf16, strided<[512, 1], offset: ?>>, memref<256x512xf16>
          %21 = arith.muli %arg4, %c512 : index
          %reinterpret_cast_0 = memref.reinterpret_cast %arg1 to offset: [%21], sizes: [512, 512], strides: [4096, 1] : memref<512x4096xf16> to memref<512x512xf16, strided<[4096, 1], offset: ?>>
          loom.copy %reinterpret_cast_0, %15 src_mem_space @DRAM dst_mem_space @L1, interconnect : [], broadcast : [1, 1] : memref<512x512xf16, strided<[4096, 1], offset: ?>>, memref<512x512xf16>
          linalg.matmul ins(%17, %15 : memref<256x512xf16>, memref<512x512xf16>) outs(%19 : memref<256x512xf16>)
          loom.semaphore_give %15 : memref<512x512xf16>
          loom.semaphore_give %17 : memref<256x512xf16>
          %22 = arith.muli %13, %c1048576 : index
          %23 = arith.addi %22, %21 : index
          %reinterpret_cast_1 = memref.reinterpret_cast %arg2 to offset: [%23], sizes: [256, 512], strides: [4096, 1] : memref<4096x4096xf16> to memref<256x512xf16, strided<[4096, 1], offset: ?>>
          loom.copy %19, %reinterpret_cast_1 src_mem_space @L1 dst_mem_space @DRAM, interconnect : [], broadcast : [1, 1] : memref<256x512xf16>, memref<256x512xf16, strided<[4096, 1], offset: ?>>
          loom.semaphore_give %19 : memref<256x512xf16>
        }
        scf.reduce 
      } {loom.iter_types = [#loom.iter_type<spatial>, #loom.iter_type<spatial>], loom.mapped_to_dims = [@x, @y]}
      return
    }
  }
  module attributes {loom.pass_name = "Materialize"} {
    func.func @matmul__d0i0_d1i1__f10__v_h__BK256__BM128__BN64(%arg0: memref<4096x512xf16>, %arg1: memref<512x4096xf16>, %arg2: memref<4096x4096xf16>) {
      %c524288 = arith.constant 524288 : index
      %c1048576 = arith.constant 1048576 : index
      %c65536 = arith.constant 65536 : index
      %c2 = arith.constant 2 : index
      %c4 = arith.constant 4 : index
      %cst = arith.constant 0.000000e+00 : f16
      %c64 = arith.constant 64 : index
      %c256 = arith.constant 256 : index
      %c0 = arith.constant 0 : index
      %c8 = arith.constant 8 : index
      %c1 = arith.constant 1 : index
      scf.parallel (%arg3, %arg4) = (%c0, %c0) to (%c8, %c8) step (%c1, %c1) {
        scf.for %arg5 = %c0 to %c8 step %c1 {
          scf.for %arg6 = %c0 to %c4 step %c1 {
            %12 = arith.muli %arg6, %c8 overflow<nsw> : index
            %13 = arith.addi %arg3, %12 : index
            %14 = arith.muli %arg5, %c8 overflow<nsw> : index
            %15 = arith.addi %arg4, %14 : index
            %16 = loom.alloc [256, 64] on @L1 : memref<256x64xf16>
            %17 = loom.semaphore_take %16 : memref<256x64xf16> -> memref<256x64xf16>
            %18 = loom.alloc [128, 256] on @L1 : memref<128x256xf16>
            %19 = loom.semaphore_take %18 : memref<128x256xf16> -> memref<128x256xf16>
            %20 = loom.alloc [128, 64] on @L1 : memref<128x64xf16>
            %21 = loom.semaphore_take %20 : memref<128x64xf16> -> memref<128x64xf16>
            linalg.fill ins(%cst : f16) outs(%21 : memref<128x64xf16>)
            scf.for %arg7 = %c0 to %c2 step %c1 {
              %25 = arith.muli %arg7, %c256 : index
              %26 = arith.muli %13, %c65536 : index
              %27 = arith.addi %26, %25 : index
              %reinterpret_cast_0 = memref.reinterpret_cast %arg0 to offset: [%27], sizes: [128, 256], strides: [512, 1] : memref<4096x512xf16> to memref<128x256xf16, strided<[512, 1], offset: ?>>
              loom.copy %reinterpret_cast_0, %19 src_mem_space @DRAM dst_mem_space @L1, interconnect : [@vertical_links], broadcast : [8, 1] : memref<128x256xf16, strided<[512, 1], offset: ?>>, memref<128x256xf16>
              %28 = arith.muli %15, %c64 : index
              %29 = arith.muli %arg7, %c1048576 : index
              %30 = arith.addi %29, %28 : index
              %reinterpret_cast_1 = memref.reinterpret_cast %arg1 to offset: [%30], sizes: [256, 64], strides: [4096, 1] : memref<512x4096xf16> to memref<256x64xf16, strided<[4096, 1], offset: ?>>
              loom.copy %reinterpret_cast_1, %17 src_mem_space @DRAM dst_mem_space @L1, interconnect : [@horizontal_links], broadcast : [1, 8] : memref<256x64xf16, strided<[4096, 1], offset: ?>>, memref<256x64xf16>
              linalg.matmul ins(%19, %17 : memref<128x256xf16>, memref<256x64xf16>) outs(%21 : memref<128x64xf16>)
              loom.semaphore_give %17 : memref<256x64xf16>
              loom.semaphore_give %19 : memref<128x256xf16>
            }
            %22 = arith.muli %15, %c64 : index
            %23 = arith.muli %13, %c524288 : index
            %24 = arith.addi %23, %22 : index
            %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%24], sizes: [128, 64], strides: [4096, 1] : memref<4096x4096xf16> to memref<128x64xf16, strided<[4096, 1], offset: ?>>
            loom.copy %21, %reinterpret_cast src_mem_space @L1 dst_mem_space @DRAM, interconnect : [], broadcast : [1, 1] : memref<128x64xf16>, memref<128x64xf16, strided<[4096, 1], offset: ?>>
            loom.semaphore_give %21 : memref<128x64xf16>
          }
        }
        scf.reduce 
      } {loom.iter_types = [#loom.iter_type<spatial>, #loom.iter_type<spatial>], loom.mapped_to_dims = [@x, @y]}
      return
    }
  }
  module attributes {loom.pass_name = "Materialize"} {
    func.func @matmul__d1i0_d0i1__f01__d_d__BK32__BM512__BN256(%arg0: memref<4096x512xf16>, %arg1: memref<512x4096xf16>, %arg2: memref<4096x4096xf16>) {
      %c2097152 = arith.constant 2097152 : index
      %c131072 = arith.constant 131072 : index
      %c262144 = arith.constant 262144 : index
      %c16 = arith.constant 16 : index
      %c2 = arith.constant 2 : index
      %cst = arith.constant 0.000000e+00 : f16
      %c256 = arith.constant 256 : index
      %c32 = arith.constant 32 : index
      %c0 = arith.constant 0 : index
      %c8 = arith.constant 8 : index
      %c1 = arith.constant 1 : index
      scf.parallel (%arg3, %arg4) = (%c0, %c0) to (%c8, %c8) step (%c1, %c1) {
        scf.for %arg5 = %c0 to %c2 step %c1 {
          %12 = arith.muli %arg5, %c8 overflow<nsw> : index
          %13 = arith.addi %arg4, %12 : index
          %14 = loom.alloc [32, 256] on @L1 : memref<32x256xf16>
          %15 = loom.semaphore_take %14 : memref<32x256xf16> -> memref<32x256xf16>
          %16 = loom.alloc [512, 32] on @L1 : memref<512x32xf16>
          %17 = loom.semaphore_take %16 : memref<512x32xf16> -> memref<512x32xf16>
          %18 = loom.alloc [512, 256] on @L1 : memref<512x256xf16>
          %19 = loom.semaphore_take %18 : memref<512x256xf16> -> memref<512x256xf16>
          linalg.fill ins(%cst : f16) outs(%19 : memref<512x256xf16>)
          scf.for %arg6 = %c0 to %c16 step %c1 {
            %23 = arith.muli %arg6, %c32 : index
            %24 = arith.muli %arg3, %c262144 : index
            %25 = arith.addi %24, %23 : index
            %reinterpret_cast_0 = memref.reinterpret_cast %arg0 to offset: [%25], sizes: [512, 32], strides: [512, 1] : memref<4096x512xf16> to memref<512x32xf16, strided<[512, 1], offset: ?>>
            loom.copy %reinterpret_cast_0, %17 src_mem_space @DRAM dst_mem_space @L1, interconnect : [], broadcast : [1, 1] : memref<512x32xf16, strided<[512, 1], offset: ?>>, memref<512x32xf16>
            %26 = arith.muli %13, %c256 : index
            %27 = arith.muli %arg6, %c131072 : index
            %28 = arith.addi %27, %26 : index
            %reinterpret_cast_1 = memref.reinterpret_cast %arg1 to offset: [%28], sizes: [32, 256], strides: [4096, 1] : memref<512x4096xf16> to memref<32x256xf16, strided<[4096, 1], offset: ?>>
            loom.copy %reinterpret_cast_1, %15 src_mem_space @DRAM dst_mem_space @L1, interconnect : [], broadcast : [1, 1] : memref<32x256xf16, strided<[4096, 1], offset: ?>>, memref<32x256xf16>
            linalg.matmul ins(%17, %15 : memref<512x32xf16>, memref<32x256xf16>) outs(%19 : memref<512x256xf16>)
            loom.semaphore_give %15 : memref<32x256xf16>
            loom.semaphore_give %17 : memref<512x32xf16>
          }
          %20 = arith.muli %13, %c256 : index
          %21 = arith.muli %arg3, %c2097152 : index
          %22 = arith.addi %21, %20 : index
          %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%22], sizes: [512, 256], strides: [4096, 1] : memref<4096x4096xf16> to memref<512x256xf16, strided<[4096, 1], offset: ?>>
          loom.copy %19, %reinterpret_cast src_mem_space @L1 dst_mem_space @DRAM, interconnect : [], broadcast : [1, 1] : memref<512x256xf16>, memref<512x256xf16, strided<[4096, 1], offset: ?>>
          loom.semaphore_give %19 : memref<512x256xf16>
        }
        scf.reduce 
      } {loom.iter_types = [#loom.iter_type<spatial>, #loom.iter_type<spatial>], loom.mapped_to_dims = [@y, @x]}
      return
    }
  }
  module attributes {loom.pass_name = "Materialize"} {
    func.func @matmul__d1i0_d0i1__f01__d_v__BK256__BM256__BN512(%arg0: memref<4096x512xf16>, %arg1: memref<512x4096xf16>, %arg2: memref<4096x4096xf16>) {
      %c1048576 = arith.constant 1048576 : index
      %c131072 = arith.constant 131072 : index
      %c2 = arith.constant 2 : index
      %cst = arith.constant 0.000000e+00 : f16
      %c256 = arith.constant 256 : index
      %c512 = arith.constant 512 : index
      %c0 = arith.constant 0 : index
      %c8 = arith.constant 8 : index
      %c1 = arith.constant 1 : index
      scf.parallel (%arg3, %arg4) = (%c0, %c0) to (%c8, %c8) step (%c1, %c1) {
        scf.for %arg5 = %c0 to %c2 step %c1 {
          %12 = arith.muli %arg5, %c8 overflow<nsw> : index
          %13 = arith.addi %arg3, %12 : index
          %14 = loom.alloc [256, 512] on @L1 : memref<256x512xf16>
          %15 = loom.semaphore_take %14 : memref<256x512xf16> -> memref<256x512xf16>
          %16 = loom.alloc [256, 256] on @L1 : memref<256x256xf16>
          %17 = loom.semaphore_take %16 : memref<256x256xf16> -> memref<256x256xf16>
          %18 = loom.alloc [256, 512] on @L1 : memref<256x512xf16>
          %19 = loom.semaphore_take %18 : memref<256x512xf16> -> memref<256x512xf16>
          linalg.fill ins(%cst : f16) outs(%19 : memref<256x512xf16>)
          scf.for %arg6 = %c0 to %c2 step %c1 {
            %23 = arith.muli %arg6, %c256 : index
            %24 = arith.muli %13, %c131072 : index
            %25 = arith.addi %24, %23 : index
            %reinterpret_cast_0 = memref.reinterpret_cast %arg0 to offset: [%25], sizes: [256, 256], strides: [512, 1] : memref<4096x512xf16> to memref<256x256xf16, strided<[512, 1], offset: ?>>
            loom.copy %reinterpret_cast_0, %17 src_mem_space @DRAM dst_mem_space @L1, interconnect : [], broadcast : [1, 1] : memref<256x256xf16, strided<[512, 1], offset: ?>>, memref<256x256xf16>
            %26 = arith.muli %arg4, %c512 : index
            %27 = arith.muli %arg6, %c1048576 : index
            %28 = arith.addi %27, %26 : index
            %reinterpret_cast_1 = memref.reinterpret_cast %arg1 to offset: [%28], sizes: [256, 512], strides: [4096, 1] : memref<512x4096xf16> to memref<256x512xf16, strided<[4096, 1], offset: ?>>
            loom.copy %reinterpret_cast_1, %15 src_mem_space @DRAM dst_mem_space @L1, interconnect : [@vertical_links], broadcast : [8, 1] : memref<256x512xf16, strided<[4096, 1], offset: ?>>, memref<256x512xf16>
            linalg.matmul ins(%17, %15 : memref<256x256xf16>, memref<256x512xf16>) outs(%19 : memref<256x512xf16>)
            loom.semaphore_give %15 : memref<256x512xf16>
            loom.semaphore_give %17 : memref<256x256xf16>
          }
          %20 = arith.muli %arg4, %c512 : index
          %21 = arith.muli %13, %c1048576 : index
          %22 = arith.addi %21, %20 : index
          %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%22], sizes: [256, 512], strides: [4096, 1] : memref<4096x4096xf16> to memref<256x512xf16, strided<[4096, 1], offset: ?>>
          loom.copy %19, %reinterpret_cast src_mem_space @L1 dst_mem_space @DRAM, interconnect : [], broadcast : [1, 1] : memref<256x512xf16>, memref<256x512xf16, strided<[4096, 1], offset: ?>>
          loom.semaphore_give %19 : memref<256x512xf16>
        }
        scf.reduce 
      } {loom.iter_types = [#loom.iter_type<spatial>, #loom.iter_type<spatial>], loom.mapped_to_dims = [@y, @x]}
      return
    }
  }
  module attributes {loom.pass_name = "Materialize"} {
    func.func @matmul__d1i0_d0i1__f01__h_d__BK128__BM128__BN512(%arg0: memref<4096x512xf16>, %arg1: memref<512x4096xf16>, %arg2: memref<4096x4096xf16>) {
      %c524288 = arith.constant 524288 : index
      %c65536 = arith.constant 65536 : index
      %c4 = arith.constant 4 : index
      %cst = arith.constant 0.000000e+00 : f16
      %c128 = arith.constant 128 : index
      %c512 = arith.constant 512 : index
      %c0 = arith.constant 0 : index
      %c8 = arith.constant 8 : index
      %c1 = arith.constant 1 : index
      scf.parallel (%arg3, %arg4) = (%c0, %c0) to (%c8, %c8) step (%c1, %c1) {
        scf.for %arg5 = %c0 to %c4 step %c1 {
          %12 = arith.muli %arg5, %c8 overflow<nsw> : index
          %13 = arith.addi %arg3, %12 : index
          %14 = loom.alloc [128, 512] on @L1 : memref<128x512xf16>
          %15 = loom.semaphore_take %14 : memref<128x512xf16> -> memref<128x512xf16>
          %16 = loom.alloc [128, 128] on @L1 : memref<128x128xf16>
          %17 = loom.semaphore_take %16 : memref<128x128xf16> -> memref<128x128xf16>
          %18 = loom.alloc [128, 512] on @L1 : memref<128x512xf16>
          %19 = loom.semaphore_take %18 : memref<128x512xf16> -> memref<128x512xf16>
          linalg.fill ins(%cst : f16) outs(%19 : memref<128x512xf16>)
          scf.for %arg6 = %c0 to %c4 step %c1 {
            %23 = arith.muli %arg6, %c128 : index
            %24 = arith.muli %13, %c65536 : index
            %25 = arith.addi %24, %23 : index
            %reinterpret_cast_0 = memref.reinterpret_cast %arg0 to offset: [%25], sizes: [128, 128], strides: [512, 1] : memref<4096x512xf16> to memref<128x128xf16, strided<[512, 1], offset: ?>>
            loom.copy %reinterpret_cast_0, %17 src_mem_space @DRAM dst_mem_space @L1, interconnect : [@horizontal_links], broadcast : [1, 8] : memref<128x128xf16, strided<[512, 1], offset: ?>>, memref<128x128xf16>
            %26 = arith.muli %arg4, %c512 : index
            %27 = arith.muli %arg6, %c524288 : index
            %28 = arith.addi %27, %26 : index
            %reinterpret_cast_1 = memref.reinterpret_cast %arg1 to offset: [%28], sizes: [128, 512], strides: [4096, 1] : memref<512x4096xf16> to memref<128x512xf16, strided<[4096, 1], offset: ?>>
            loom.copy %reinterpret_cast_1, %15 src_mem_space @DRAM dst_mem_space @L1, interconnect : [], broadcast : [1, 1] : memref<128x512xf16, strided<[4096, 1], offset: ?>>, memref<128x512xf16>
            linalg.matmul ins(%17, %15 : memref<128x128xf16>, memref<128x512xf16>) outs(%19 : memref<128x512xf16>)
            loom.semaphore_give %15 : memref<128x512xf16>
            loom.semaphore_give %17 : memref<128x128xf16>
          }
          %20 = arith.muli %arg4, %c512 : index
          %21 = arith.muli %13, %c524288 : index
          %22 = arith.addi %21, %20 : index
          %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%22], sizes: [128, 512], strides: [4096, 1] : memref<4096x4096xf16> to memref<128x512xf16, strided<[4096, 1], offset: ?>>
          loom.copy %19, %reinterpret_cast src_mem_space @L1 dst_mem_space @DRAM, interconnect : [], broadcast : [1, 1] : memref<128x512xf16>, memref<128x512xf16, strided<[4096, 1], offset: ?>>
          loom.semaphore_give %19 : memref<128x512xf16>
        }
        scf.reduce 
      } {loom.iter_types = [#loom.iter_type<spatial>, #loom.iter_type<spatial>], loom.mapped_to_dims = [@y, @x]}
      return
    }
  }
  module attributes {loom.pass_name = "Materialize"} {
    func.func @matmul__d1i0_d0i1__f01__h_v__BK64__BM64__BN512(%arg0: memref<4096x512xf16>, %arg1: memref<512x4096xf16>, %arg2: memref<4096x4096xf16>) {
      %c262144 = arith.constant 262144 : index
      %c32768 = arith.constant 32768 : index
      %cst = arith.constant 0.000000e+00 : f16
      %c64 = arith.constant 64 : index
      %c512 = arith.constant 512 : index
      %c0 = arith.constant 0 : index
      %c8 = arith.constant 8 : index
      %c1 = arith.constant 1 : index
      scf.parallel (%arg3, %arg4) = (%c0, %c0) to (%c8, %c8) step (%c1, %c1) {
        scf.for %arg5 = %c0 to %c8 step %c1 {
          %12 = arith.muli %arg5, %c8 overflow<nsw> : index
          %13 = arith.addi %arg3, %12 : index
          %14 = loom.alloc [64, 512] on @L1 : memref<64x512xf16>
          %15 = loom.semaphore_take %14 : memref<64x512xf16> -> memref<64x512xf16>
          %16 = loom.alloc [64, 64] on @L1 : memref<64x64xf16>
          %17 = loom.semaphore_take %16 : memref<64x64xf16> -> memref<64x64xf16>
          %18 = loom.alloc [64, 512] on @L1 : memref<64x512xf16>
          %19 = loom.semaphore_take %18 : memref<64x512xf16> -> memref<64x512xf16>
          linalg.fill ins(%cst : f16) outs(%19 : memref<64x512xf16>)
          scf.for %arg6 = %c0 to %c8 step %c1 {
            %23 = arith.muli %arg6, %c64 : index
            %24 = arith.muli %13, %c32768 : index
            %25 = arith.addi %24, %23 : index
            %reinterpret_cast_0 = memref.reinterpret_cast %arg0 to offset: [%25], sizes: [64, 64], strides: [512, 1] : memref<4096x512xf16> to memref<64x64xf16, strided<[512, 1], offset: ?>>
            loom.copy %reinterpret_cast_0, %17 src_mem_space @DRAM dst_mem_space @L1, interconnect : [@horizontal_links], broadcast : [1, 8] : memref<64x64xf16, strided<[512, 1], offset: ?>>, memref<64x64xf16>
            %26 = arith.muli %arg4, %c512 : index
            %27 = arith.muli %arg6, %c262144 : index
            %28 = arith.addi %27, %26 : index
            %reinterpret_cast_1 = memref.reinterpret_cast %arg1 to offset: [%28], sizes: [64, 512], strides: [4096, 1] : memref<512x4096xf16> to memref<64x512xf16, strided<[4096, 1], offset: ?>>
            loom.copy %reinterpret_cast_1, %15 src_mem_space @DRAM dst_mem_space @L1, interconnect : [@vertical_links], broadcast : [8, 1] : memref<64x512xf16, strided<[4096, 1], offset: ?>>, memref<64x512xf16>
            linalg.matmul ins(%17, %15 : memref<64x64xf16>, memref<64x512xf16>) outs(%19 : memref<64x512xf16>)
            loom.semaphore_give %15 : memref<64x512xf16>
            loom.semaphore_give %17 : memref<64x64xf16>
          }
          %20 = arith.muli %arg4, %c512 : index
          %21 = arith.muli %13, %c262144 : index
          %22 = arith.addi %21, %20 : index
          %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%22], sizes: [64, 512], strides: [4096, 1] : memref<4096x4096xf16> to memref<64x512xf16, strided<[4096, 1], offset: ?>>
          loom.copy %19, %reinterpret_cast src_mem_space @L1 dst_mem_space @DRAM, interconnect : [], broadcast : [1, 1] : memref<64x512xf16>, memref<64x512xf16, strided<[4096, 1], offset: ?>>
          loom.semaphore_give %19 : memref<64x512xf16>
        }
        scf.reduce 
      } {loom.iter_types = [#loom.iter_type<spatial>, #loom.iter_type<spatial>], loom.mapped_to_dims = [@y, @x]}
      return
    }
  }
  module attributes {loom.pass_name = "Materialize"} {
    func.func @matmul__d1i0_d0i1__f10__d_d__BK32__BM512__BN256(%arg0: memref<4096x512xf16>, %arg1: memref<512x4096xf16>, %arg2: memref<4096x4096xf16>) {
      %c2097152 = arith.constant 2097152 : index
      %c131072 = arith.constant 131072 : index
      %c262144 = arith.constant 262144 : index
      %c16 = arith.constant 16 : index
      %c2 = arith.constant 2 : index
      %cst = arith.constant 0.000000e+00 : f16
      %c256 = arith.constant 256 : index
      %c32 = arith.constant 32 : index
      %c0 = arith.constant 0 : index
      %c8 = arith.constant 8 : index
      %c1 = arith.constant 1 : index
      scf.parallel (%arg3, %arg4) = (%c0, %c0) to (%c8, %c8) step (%c1, %c1) {
        scf.for %arg5 = %c0 to %c2 step %c1 {
          %12 = arith.muli %arg5, %c8 overflow<nsw> : index
          %13 = arith.addi %arg4, %12 : index
          %14 = loom.alloc [32, 256] on @L1 : memref<32x256xf16>
          %15 = loom.semaphore_take %14 : memref<32x256xf16> -> memref<32x256xf16>
          %16 = loom.alloc [512, 32] on @L1 : memref<512x32xf16>
          %17 = loom.semaphore_take %16 : memref<512x32xf16> -> memref<512x32xf16>
          %18 = loom.alloc [512, 256] on @L1 : memref<512x256xf16>
          %19 = loom.semaphore_take %18 : memref<512x256xf16> -> memref<512x256xf16>
          linalg.fill ins(%cst : f16) outs(%19 : memref<512x256xf16>)
          scf.for %arg6 = %c0 to %c16 step %c1 {
            %23 = arith.muli %arg6, %c32 : index
            %24 = arith.muli %arg3, %c262144 : index
            %25 = arith.addi %24, %23 : index
            %reinterpret_cast_0 = memref.reinterpret_cast %arg0 to offset: [%25], sizes: [512, 32], strides: [512, 1] : memref<4096x512xf16> to memref<512x32xf16, strided<[512, 1], offset: ?>>
            loom.copy %reinterpret_cast_0, %17 src_mem_space @DRAM dst_mem_space @L1, interconnect : [], broadcast : [1, 1] : memref<512x32xf16, strided<[512, 1], offset: ?>>, memref<512x32xf16>
            %26 = arith.muli %13, %c256 : index
            %27 = arith.muli %arg6, %c131072 : index
            %28 = arith.addi %27, %26 : index
            %reinterpret_cast_1 = memref.reinterpret_cast %arg1 to offset: [%28], sizes: [32, 256], strides: [4096, 1] : memref<512x4096xf16> to memref<32x256xf16, strided<[4096, 1], offset: ?>>
            loom.copy %reinterpret_cast_1, %15 src_mem_space @DRAM dst_mem_space @L1, interconnect : [], broadcast : [1, 1] : memref<32x256xf16, strided<[4096, 1], offset: ?>>, memref<32x256xf16>
            linalg.matmul ins(%17, %15 : memref<512x32xf16>, memref<32x256xf16>) outs(%19 : memref<512x256xf16>)
            loom.semaphore_give %15 : memref<32x256xf16>
            loom.semaphore_give %17 : memref<512x32xf16>
          }
          %20 = arith.muli %13, %c256 : index
          %21 = arith.muli %arg3, %c2097152 : index
          %22 = arith.addi %21, %20 : index
          %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%22], sizes: [512, 256], strides: [4096, 1] : memref<4096x4096xf16> to memref<512x256xf16, strided<[4096, 1], offset: ?>>
          loom.copy %19, %reinterpret_cast src_mem_space @L1 dst_mem_space @DRAM, interconnect : [], broadcast : [1, 1] : memref<512x256xf16>, memref<512x256xf16, strided<[4096, 1], offset: ?>>
          loom.semaphore_give %19 : memref<512x256xf16>
        }
        scf.reduce 
      } {loom.iter_types = [#loom.iter_type<spatial>, #loom.iter_type<spatial>], loom.mapped_to_dims = [@y, @x]}
      return
    }
  }
  module attributes {loom.pass_name = "Materialize"} {
    func.func @matmul__d1i0_d0i1__f10__d_v__BK256__BM512__BN128(%arg0: memref<4096x512xf16>, %arg1: memref<512x4096xf16>, %arg2: memref<4096x4096xf16>) {
      %c2097152 = arith.constant 2097152 : index
      %c1048576 = arith.constant 1048576 : index
      %c262144 = arith.constant 262144 : index
      %c2 = arith.constant 2 : index
      %c4 = arith.constant 4 : index
      %cst = arith.constant 0.000000e+00 : f16
      %c128 = arith.constant 128 : index
      %c256 = arith.constant 256 : index
      %c0 = arith.constant 0 : index
      %c8 = arith.constant 8 : index
      %c1 = arith.constant 1 : index
      scf.parallel (%arg3, %arg4) = (%c0, %c0) to (%c8, %c8) step (%c1, %c1) {
        scf.for %arg5 = %c0 to %c4 step %c1 {
          %12 = arith.muli %arg5, %c8 overflow<nsw> : index
          %13 = arith.addi %arg4, %12 : index
          %14 = loom.alloc [256, 128] on @L1 : memref<256x128xf16>
          %15 = loom.semaphore_take %14 : memref<256x128xf16> -> memref<256x128xf16>
          %16 = loom.alloc [512, 256] on @L1 : memref<512x256xf16>
          %17 = loom.semaphore_take %16 : memref<512x256xf16> -> memref<512x256xf16>
          %18 = loom.alloc [512, 128] on @L1 : memref<512x128xf16>
          %19 = loom.semaphore_take %18 : memref<512x128xf16> -> memref<512x128xf16>
          linalg.fill ins(%cst : f16) outs(%19 : memref<512x128xf16>)
          scf.for %arg6 = %c0 to %c2 step %c1 {
            %23 = arith.muli %arg6, %c256 : index
            %24 = arith.muli %arg3, %c262144 : index
            %25 = arith.addi %24, %23 : index
            %reinterpret_cast_0 = memref.reinterpret_cast %arg0 to offset: [%25], sizes: [512, 256], strides: [512, 1] : memref<4096x512xf16> to memref<512x256xf16, strided<[512, 1], offset: ?>>
            loom.copy %reinterpret_cast_0, %17 src_mem_space @DRAM dst_mem_space @L1, interconnect : [], broadcast : [1, 1] : memref<512x256xf16, strided<[512, 1], offset: ?>>, memref<512x256xf16>
            %26 = arith.muli %13, %c128 : index
            %27 = arith.muli %arg6, %c1048576 : index
            %28 = arith.addi %27, %26 : index
            %reinterpret_cast_1 = memref.reinterpret_cast %arg1 to offset: [%28], sizes: [256, 128], strides: [4096, 1] : memref<512x4096xf16> to memref<256x128xf16, strided<[4096, 1], offset: ?>>
            loom.copy %reinterpret_cast_1, %15 src_mem_space @DRAM dst_mem_space @L1, interconnect : [@vertical_links], broadcast : [8, 1] : memref<256x128xf16, strided<[4096, 1], offset: ?>>, memref<256x128xf16>
            linalg.matmul ins(%17, %15 : memref<512x256xf16>, memref<256x128xf16>) outs(%19 : memref<512x128xf16>)
            loom.semaphore_give %15 : memref<256x128xf16>
            loom.semaphore_give %17 : memref<512x256xf16>
          }
          %20 = arith.muli %13, %c128 : index
          %21 = arith.muli %arg3, %c2097152 : index
          %22 = arith.addi %21, %20 : index
          %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%22], sizes: [512, 128], strides: [4096, 1] : memref<4096x4096xf16> to memref<512x128xf16, strided<[4096, 1], offset: ?>>
          loom.copy %19, %reinterpret_cast src_mem_space @L1 dst_mem_space @DRAM, interconnect : [], broadcast : [1, 1] : memref<512x128xf16>, memref<512x128xf16, strided<[4096, 1], offset: ?>>
          loom.semaphore_give %19 : memref<512x128xf16>
        }
        scf.reduce 
      } {loom.iter_types = [#loom.iter_type<spatial>, #loom.iter_type<spatial>], loom.mapped_to_dims = [@y, @x]}
      return
    }
  }
  module attributes {loom.pass_name = "Materialize"} {
    func.func @matmul__d1i0_d0i1__f10__h_d__BK256__BM256__BN128(%arg0: memref<4096x512xf16>, %arg1: memref<512x4096xf16>, %arg2: memref<4096x4096xf16>) {
      %c1048576 = arith.constant 1048576 : index
      %c131072 = arith.constant 131072 : index
      %c2 = arith.constant 2 : index
      %c4 = arith.constant 4 : index
      %cst = arith.constant 0.000000e+00 : f16
      %c256 = arith.constant 256 : index
      %c128 = arith.constant 128 : index
      %c0 = arith.constant 0 : index
      %c8 = arith.constant 8 : index
      %c1 = arith.constant 1 : index
      scf.parallel (%arg3, %arg4) = (%c0, %c0) to (%c8, %c8) step (%c1, %c1) {
        scf.for %arg5 = %c0 to %c4 step %c1 {
          scf.for %arg6 = %c0 to %c2 step %c1 {
            %12 = arith.muli %arg6, %c8 overflow<nsw> : index
            %13 = arith.addi %arg3, %12 : index
            %14 = arith.muli %arg5, %c8 overflow<nsw> : index
            %15 = arith.addi %arg4, %14 : index
            %16 = loom.alloc [256, 128] on @L1 : memref<256x128xf16>
            %17 = loom.semaphore_take %16 : memref<256x128xf16> -> memref<256x128xf16>
            %18 = loom.alloc [256, 256] on @L1 : memref<256x256xf16>
            %19 = loom.semaphore_take %18 : memref<256x256xf16> -> memref<256x256xf16>
            %20 = loom.alloc [256, 128] on @L1 : memref<256x128xf16>
            %21 = loom.semaphore_take %20 : memref<256x128xf16> -> memref<256x128xf16>
            linalg.fill ins(%cst : f16) outs(%21 : memref<256x128xf16>)
            scf.for %arg7 = %c0 to %c2 step %c1 {
              %25 = arith.muli %arg7, %c256 : index
              %26 = arith.muli %13, %c131072 : index
              %27 = arith.addi %26, %25 : index
              %reinterpret_cast_0 = memref.reinterpret_cast %arg0 to offset: [%27], sizes: [256, 256], strides: [512, 1] : memref<4096x512xf16> to memref<256x256xf16, strided<[512, 1], offset: ?>>
              loom.copy %reinterpret_cast_0, %19 src_mem_space @DRAM dst_mem_space @L1, interconnect : [@horizontal_links], broadcast : [1, 8] : memref<256x256xf16, strided<[512, 1], offset: ?>>, memref<256x256xf16>
              %28 = arith.muli %15, %c128 : index
              %29 = arith.muli %arg7, %c1048576 : index
              %30 = arith.addi %29, %28 : index
              %reinterpret_cast_1 = memref.reinterpret_cast %arg1 to offset: [%30], sizes: [256, 128], strides: [4096, 1] : memref<512x4096xf16> to memref<256x128xf16, strided<[4096, 1], offset: ?>>
              loom.copy %reinterpret_cast_1, %17 src_mem_space @DRAM dst_mem_space @L1, interconnect : [], broadcast : [1, 1] : memref<256x128xf16, strided<[4096, 1], offset: ?>>, memref<256x128xf16>
              linalg.matmul ins(%19, %17 : memref<256x256xf16>, memref<256x128xf16>) outs(%21 : memref<256x128xf16>)
              loom.semaphore_give %17 : memref<256x128xf16>
              loom.semaphore_give %19 : memref<256x256xf16>
            }
            %22 = arith.muli %15, %c128 : index
            %23 = arith.muli %13, %c1048576 : index
            %24 = arith.addi %23, %22 : index
            %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%24], sizes: [256, 128], strides: [4096, 1] : memref<4096x4096xf16> to memref<256x128xf16, strided<[4096, 1], offset: ?>>
            loom.copy %21, %reinterpret_cast src_mem_space @L1 dst_mem_space @DRAM, interconnect : [], broadcast : [1, 1] : memref<256x128xf16>, memref<256x128xf16, strided<[4096, 1], offset: ?>>
            loom.semaphore_give %21 : memref<256x128xf16>
          }
        }
        scf.reduce 
      } {loom.iter_types = [#loom.iter_type<spatial>, #loom.iter_type<spatial>], loom.mapped_to_dims = [@y, @x]}
      return
    }
  }
  module attributes {loom.pass_name = "Materialize"} {
    func.func @matmul__d1i0_d0i1__f10__h_v__BK64__BM128__BN512(%arg0: memref<4096x512xf16>, %arg1: memref<512x4096xf16>, %arg2: memref<4096x4096xf16>) {
      %c524288 = arith.constant 524288 : index
      %c262144 = arith.constant 262144 : index
      %c65536 = arith.constant 65536 : index
      %c4 = arith.constant 4 : index
      %cst = arith.constant 0.000000e+00 : f16
      %c512 = arith.constant 512 : index
      %c64 = arith.constant 64 : index
      %c0 = arith.constant 0 : index
      %c8 = arith.constant 8 : index
      %c1 = arith.constant 1 : index
      scf.parallel (%arg3, %arg4) = (%c0, %c0) to (%c8, %c8) step (%c1, %c1) {
        scf.for %arg5 = %c0 to %c4 step %c1 {
          %12 = arith.muli %arg5, %c8 overflow<nsw> : index
          %13 = arith.addi %arg3, %12 : index
          %14 = loom.alloc [64, 512] on @L1 : memref<64x512xf16>
          %15 = loom.semaphore_take %14 : memref<64x512xf16> -> memref<64x512xf16>
          %16 = loom.alloc [128, 64] on @L1 : memref<128x64xf16>
          %17 = loom.semaphore_take %16 : memref<128x64xf16> -> memref<128x64xf16>
          %18 = loom.alloc [128, 512] on @L1 : memref<128x512xf16>
          %19 = loom.semaphore_take %18 : memref<128x512xf16> -> memref<128x512xf16>
          linalg.fill ins(%cst : f16) outs(%19 : memref<128x512xf16>)
          scf.for %arg6 = %c0 to %c8 step %c1 {
            %23 = arith.muli %arg6, %c64 : index
            %24 = arith.muli %13, %c65536 : index
            %25 = arith.addi %24, %23 : index
            %reinterpret_cast_0 = memref.reinterpret_cast %arg0 to offset: [%25], sizes: [128, 64], strides: [512, 1] : memref<4096x512xf16> to memref<128x64xf16, strided<[512, 1], offset: ?>>
            loom.copy %reinterpret_cast_0, %17 src_mem_space @DRAM dst_mem_space @L1, interconnect : [@horizontal_links], broadcast : [1, 8] : memref<128x64xf16, strided<[512, 1], offset: ?>>, memref<128x64xf16>
            %26 = arith.muli %arg4, %c512 : index
            %27 = arith.muli %arg6, %c262144 : index
            %28 = arith.addi %27, %26 : index
            %reinterpret_cast_1 = memref.reinterpret_cast %arg1 to offset: [%28], sizes: [64, 512], strides: [4096, 1] : memref<512x4096xf16> to memref<64x512xf16, strided<[4096, 1], offset: ?>>
            loom.copy %reinterpret_cast_1, %15 src_mem_space @DRAM dst_mem_space @L1, interconnect : [@vertical_links], broadcast : [8, 1] : memref<64x512xf16, strided<[4096, 1], offset: ?>>, memref<64x512xf16>
            linalg.matmul ins(%17, %15 : memref<128x64xf16>, memref<64x512xf16>) outs(%19 : memref<128x512xf16>)
            loom.semaphore_give %15 : memref<64x512xf16>
            loom.semaphore_give %17 : memref<128x64xf16>
          }
          %20 = arith.muli %arg4, %c512 : index
          %21 = arith.muli %13, %c524288 : index
          %22 = arith.addi %21, %20 : index
          %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%22], sizes: [128, 512], strides: [4096, 1] : memref<4096x4096xf16> to memref<128x512xf16, strided<[4096, 1], offset: ?>>
          loom.copy %19, %reinterpret_cast src_mem_space @L1 dst_mem_space @DRAM, interconnect : [], broadcast : [1, 1] : memref<128x512xf16>, memref<128x512xf16, strided<[4096, 1], offset: ?>>
          loom.semaphore_give %19 : memref<128x512xf16>
        }
        scf.reduce 
      } {loom.iter_types = [#loom.iter_type<spatial>, #loom.iter_type<spatial>], loom.mapped_to_dims = [@y, @x]}
      return
    }
  }
  module attributes {loom.pass_name = "Materialize"} {
    func.func @matmul__d0i1_d1i1__f01__d_d__BK512__BM4096__BN64(%arg0: memref<4096x512xf16>, %arg1: memref<512x4096xf16>, %arg2: memref<4096x4096xf16>) {
      %cst = arith.constant 0.000000e+00 : f16
      %c64 = arith.constant 64 : index
      %c0 = arith.constant 0 : index
      %c8 = arith.constant 8 : index
      %c1 = arith.constant 1 : index
      scf.parallel (%arg3, %arg4) = (%c0, %c0) to (%c8, %c8) step (%c1, %c1) {
        %12 = arith.muli %arg3, %c8 overflow<nsw> : index
        %13 = arith.addi %12, %arg4 : index
        %14 = loom.alloc [512, 64] on @L1 : memref<512x64xf16>
        %15 = loom.semaphore_take %14 : memref<512x64xf16> -> memref<512x64xf16>
        %16 = loom.alloc [4096, 512] on @L1 : memref<4096x512xf16>
        %17 = loom.semaphore_take %16 : memref<4096x512xf16> -> memref<4096x512xf16>
        %18 = loom.alloc [4096, 64] on @L1 : memref<4096x64xf16>
        %19 = loom.semaphore_take %18 : memref<4096x64xf16> -> memref<4096x64xf16>
        linalg.fill ins(%cst : f16) outs(%19 : memref<4096x64xf16>)
        %reinterpret_cast = memref.reinterpret_cast %arg0 to offset: [0], sizes: [4096, 512], strides: [512, 1] : memref<4096x512xf16> to memref<4096x512xf16, strided<[512, 1]>>
        %cast = memref.cast %reinterpret_cast : memref<4096x512xf16, strided<[512, 1]>> to memref<4096x512xf16, strided<[512, 1], offset: ?>>
        loom.copy %cast, %17 src_mem_space @DRAM dst_mem_space @L1, interconnect : [], broadcast : [1, 1] : memref<4096x512xf16, strided<[512, 1], offset: ?>>, memref<4096x512xf16>
        %20 = arith.muli %13, %c64 : index
        %reinterpret_cast_0 = memref.reinterpret_cast %arg1 to offset: [%20], sizes: [512, 64], strides: [4096, 1] : memref<512x4096xf16> to memref<512x64xf16, strided<[4096, 1], offset: ?>>
        loom.copy %reinterpret_cast_0, %15 src_mem_space @DRAM dst_mem_space @L1, interconnect : [], broadcast : [1, 1] : memref<512x64xf16, strided<[4096, 1], offset: ?>>, memref<512x64xf16>
        linalg.matmul ins(%17, %15 : memref<4096x512xf16>, memref<512x64xf16>) outs(%19 : memref<4096x64xf16>)
        loom.semaphore_give %15 : memref<512x64xf16>
        loom.semaphore_give %17 : memref<4096x512xf16>
        %reinterpret_cast_1 = memref.reinterpret_cast %arg2 to offset: [%20], sizes: [4096, 64], strides: [4096, 1] : memref<4096x4096xf16> to memref<4096x64xf16, strided<[4096, 1], offset: ?>>
        loom.copy %19, %reinterpret_cast_1 src_mem_space @L1 dst_mem_space @DRAM, interconnect : [], broadcast : [1, 1] : memref<4096x64xf16>, memref<4096x64xf16, strided<[4096, 1], offset: ?>>
        loom.semaphore_give %19 : memref<4096x64xf16>
        scf.reduce 
      } {loom.iter_types = [#loom.iter_type<spatial>, #loom.iter_type<spatial>], loom.mapped_to_dims = [@x, @y]}
      return
    }
  }
  module attributes {loom.pass_name = "Materialize"} {
    func.func @matmul__d0i1_d1i1__f01__a_d__BK512__BM4096__BN64(%arg0: memref<4096x512xf16>, %arg1: memref<512x4096xf16>, %arg2: memref<4096x4096xf16>) {
      %cst = arith.constant 0.000000e+00 : f16
      %c64 = arith.constant 64 : index
      %c0 = arith.constant 0 : index
      %c8 = arith.constant 8 : index
      %c1 = arith.constant 1 : index
      scf.parallel (%arg3, %arg4) = (%c0, %c0) to (%c8, %c8) step (%c1, %c1) {
        %12 = arith.muli %arg3, %c8 overflow<nsw> : index
        %13 = arith.addi %12, %arg4 : index
        %14 = loom.alloc [512, 64] on @L1 : memref<512x64xf16>
        %15 = loom.semaphore_take %14 : memref<512x64xf16> -> memref<512x64xf16>
        %16 = loom.alloc [4096, 512] on @L1 : memref<4096x512xf16>
        %17 = loom.semaphore_take %16 : memref<4096x512xf16> -> memref<4096x512xf16>
        %18 = loom.alloc [4096, 64] on @L1 : memref<4096x64xf16>
        %19 = loom.semaphore_take %18 : memref<4096x64xf16> -> memref<4096x64xf16>
        linalg.fill ins(%cst : f16) outs(%19 : memref<4096x64xf16>)
        %reinterpret_cast = memref.reinterpret_cast %arg0 to offset: [0], sizes: [4096, 512], strides: [512, 1] : memref<4096x512xf16> to memref<4096x512xf16, strided<[512, 1]>>
        %cast = memref.cast %reinterpret_cast : memref<4096x512xf16, strided<[512, 1]>> to memref<4096x512xf16, strided<[512, 1], offset: ?>>
        loom.copy %cast, %17 src_mem_space @DRAM dst_mem_space @L1, interconnect : [@horizontal_links, @vertical_links], broadcast : [8, 8] : memref<4096x512xf16, strided<[512, 1], offset: ?>>, memref<4096x512xf16>
        %20 = arith.muli %13, %c64 : index
        %reinterpret_cast_0 = memref.reinterpret_cast %arg1 to offset: [%20], sizes: [512, 64], strides: [4096, 1] : memref<512x4096xf16> to memref<512x64xf16, strided<[4096, 1], offset: ?>>
        loom.copy %reinterpret_cast_0, %15 src_mem_space @DRAM dst_mem_space @L1, interconnect : [], broadcast : [1, 1] : memref<512x64xf16, strided<[4096, 1], offset: ?>>, memref<512x64xf16>
        linalg.matmul ins(%17, %15 : memref<4096x512xf16>, memref<512x64xf16>) outs(%19 : memref<4096x64xf16>)
        loom.semaphore_give %15 : memref<512x64xf16>
        loom.semaphore_give %17 : memref<4096x512xf16>
        %reinterpret_cast_1 = memref.reinterpret_cast %arg2 to offset: [%20], sizes: [4096, 64], strides: [4096, 1] : memref<4096x4096xf16> to memref<4096x64xf16, strided<[4096, 1], offset: ?>>
        loom.copy %19, %reinterpret_cast_1 src_mem_space @L1 dst_mem_space @DRAM, interconnect : [], broadcast : [1, 1] : memref<4096x64xf16>, memref<4096x64xf16, strided<[4096, 1], offset: ?>>
        loom.semaphore_give %19 : memref<4096x64xf16>
        scf.reduce 
      } {loom.iter_types = [#loom.iter_type<spatial>, #loom.iter_type<spatial>], loom.mapped_to_dims = [@x, @y]}
      return
    }
  }
  module attributes {loom.pass_name = "Materialize"} {
    func.func @matmul__d0i1_d1i1__f01__h_d__BK128__BM1024__BN64(%arg0: memref<4096x512xf16>, %arg1: memref<512x4096xf16>, %arg2: memref<4096x4096xf16>) {
      %c4194304 = arith.constant 4194304 : index
      %c524288 = arith.constant 524288 : index
      %c4 = arith.constant 4 : index
      %cst = arith.constant 0.000000e+00 : f16
      %c64 = arith.constant 64 : index
      %c128 = arith.constant 128 : index
      %c0 = arith.constant 0 : index
      %c8 = arith.constant 8 : index
      %c1 = arith.constant 1 : index
      scf.parallel (%arg3, %arg4) = (%c0, %c0) to (%c8, %c8) step (%c1, %c1) {
        scf.for %arg5 = %c0 to %c4 step %c1 {
          %12 = arith.muli %arg3, %c8 overflow<nsw> : index
          %13 = arith.addi %12, %arg4 : index
          %14 = loom.alloc [128, 64] on @L1 : memref<128x64xf16>
          %15 = loom.semaphore_take %14 : memref<128x64xf16> -> memref<128x64xf16>
          %16 = loom.alloc [1024, 128] on @L1 : memref<1024x128xf16>
          %17 = loom.semaphore_take %16 : memref<1024x128xf16> -> memref<1024x128xf16>
          %18 = loom.alloc [1024, 64] on @L1 : memref<1024x64xf16>
          %19 = loom.semaphore_take %18 : memref<1024x64xf16> -> memref<1024x64xf16>
          linalg.fill ins(%cst : f16) outs(%19 : memref<1024x64xf16>)
          scf.for %arg6 = %c0 to %c4 step %c1 {
            %23 = arith.muli %arg6, %c128 : index
            %24 = arith.muli %arg5, %c524288 : index
            %25 = arith.addi %24, %23 : index
            %reinterpret_cast_0 = memref.reinterpret_cast %arg0 to offset: [%25], sizes: [1024, 128], strides: [512, 1] : memref<4096x512xf16> to memref<1024x128xf16, strided<[512, 1], offset: ?>>
            loom.copy %reinterpret_cast_0, %17 src_mem_space @DRAM dst_mem_space @L1, interconnect : [@horizontal_links], broadcast : [1, 8] : memref<1024x128xf16, strided<[512, 1], offset: ?>>, memref<1024x128xf16>
            %26 = arith.muli %13, %c64 : index
            %27 = arith.muli %arg6, %c524288 : index
            %28 = arith.addi %27, %26 : index
            %reinterpret_cast_1 = memref.reinterpret_cast %arg1 to offset: [%28], sizes: [128, 64], strides: [4096, 1] : memref<512x4096xf16> to memref<128x64xf16, strided<[4096, 1], offset: ?>>
            loom.copy %reinterpret_cast_1, %15 src_mem_space @DRAM dst_mem_space @L1, interconnect : [], broadcast : [1, 1] : memref<128x64xf16, strided<[4096, 1], offset: ?>>, memref<128x64xf16>
            linalg.matmul ins(%17, %15 : memref<1024x128xf16>, memref<128x64xf16>) outs(%19 : memref<1024x64xf16>)
            loom.semaphore_give %15 : memref<128x64xf16>
            loom.semaphore_give %17 : memref<1024x128xf16>
          }
          %20 = arith.muli %13, %c64 : index
          %21 = arith.muli %arg5, %c4194304 : index
          %22 = arith.addi %21, %20 : index
          %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%22], sizes: [1024, 64], strides: [4096, 1] : memref<4096x4096xf16> to memref<1024x64xf16, strided<[4096, 1], offset: ?>>
          loom.copy %19, %reinterpret_cast src_mem_space @L1 dst_mem_space @DRAM, interconnect : [], broadcast : [1, 1] : memref<1024x64xf16>, memref<1024x64xf16, strided<[4096, 1], offset: ?>>
          loom.semaphore_give %19 : memref<1024x64xf16>
        }
        scf.reduce 
      } {loom.iter_types = [#loom.iter_type<spatial>, #loom.iter_type<spatial>], loom.mapped_to_dims = [@x, @y]}
      return
    }
  }
  module attributes {loom.pass_name = "Materialize"} {
    func.func @matmul__d0i1_d1i1__f01__v_d__BK64__BM2048__BN64(%arg0: memref<4096x512xf16>, %arg1: memref<512x4096xf16>, %arg2: memref<4096x4096xf16>) {
      %c8388608 = arith.constant 8388608 : index
      %c262144 = arith.constant 262144 : index
      %c1048576 = arith.constant 1048576 : index
      %c2 = arith.constant 2 : index
      %cst = arith.constant 0.000000e+00 : f16
      %c64 = arith.constant 64 : index
      %c0 = arith.constant 0 : index
      %c8 = arith.constant 8 : index
      %c1 = arith.constant 1 : index
      scf.parallel (%arg3, %arg4) = (%c0, %c0) to (%c8, %c8) step (%c1, %c1) {
        scf.for %arg5 = %c0 to %c2 step %c1 {
          %12 = arith.muli %arg3, %c8 overflow<nsw> : index
          %13 = arith.addi %12, %arg4 : index
          %14 = loom.alloc [64, 64] on @L1 : memref<64x64xf16>
          %15 = loom.semaphore_take %14 : memref<64x64xf16> -> memref<64x64xf16>
          %16 = loom.alloc [2048, 64] on @L1 : memref<2048x64xf16>
          %17 = loom.semaphore_take %16 : memref<2048x64xf16> -> memref<2048x64xf16>
          %18 = loom.alloc [2048, 64] on @L1 : memref<2048x64xf16>
          %19 = loom.semaphore_take %18 : memref<2048x64xf16> -> memref<2048x64xf16>
          linalg.fill ins(%cst : f16) outs(%19 : memref<2048x64xf16>)
          scf.for %arg6 = %c0 to %c8 step %c1 {
            %23 = arith.muli %arg6, %c64 : index
            %24 = arith.muli %arg5, %c1048576 : index
            %25 = arith.addi %24, %23 : index
            %reinterpret_cast_0 = memref.reinterpret_cast %arg0 to offset: [%25], sizes: [2048, 64], strides: [512, 1] : memref<4096x512xf16> to memref<2048x64xf16, strided<[512, 1], offset: ?>>
            loom.copy %reinterpret_cast_0, %17 src_mem_space @DRAM dst_mem_space @L1, interconnect : [@vertical_links], broadcast : [8, 1] : memref<2048x64xf16, strided<[512, 1], offset: ?>>, memref<2048x64xf16>
            %26 = arith.muli %13, %c64 : index
            %27 = arith.muli %arg6, %c262144 : index
            %28 = arith.addi %27, %26 : index
            %reinterpret_cast_1 = memref.reinterpret_cast %arg1 to offset: [%28], sizes: [64, 64], strides: [4096, 1] : memref<512x4096xf16> to memref<64x64xf16, strided<[4096, 1], offset: ?>>
            loom.copy %reinterpret_cast_1, %15 src_mem_space @DRAM dst_mem_space @L1, interconnect : [], broadcast : [1, 1] : memref<64x64xf16, strided<[4096, 1], offset: ?>>, memref<64x64xf16>
            linalg.matmul ins(%17, %15 : memref<2048x64xf16>, memref<64x64xf16>) outs(%19 : memref<2048x64xf16>)
            loom.semaphore_give %15 : memref<64x64xf16>
            loom.semaphore_give %17 : memref<2048x64xf16>
          }
          %20 = arith.muli %13, %c64 : index
          %21 = arith.muli %arg5, %c8388608 : index
          %22 = arith.addi %21, %20 : index
          %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%22], sizes: [2048, 64], strides: [4096, 1] : memref<4096x4096xf16> to memref<2048x64xf16, strided<[4096, 1], offset: ?>>
          loom.copy %19, %reinterpret_cast src_mem_space @L1 dst_mem_space @DRAM, interconnect : [], broadcast : [1, 1] : memref<2048x64xf16>, memref<2048x64xf16, strided<[4096, 1], offset: ?>>
          loom.semaphore_give %19 : memref<2048x64xf16>
        }
        scf.reduce 
      } {loom.iter_types = [#loom.iter_type<spatial>, #loom.iter_type<spatial>], loom.mapped_to_dims = [@x, @y]}
      return
    }
  }
  module attributes {loom.pass_name = "Materialize"} {
    func.func @matmul__d0i1_d1i1__f10__d_d__BK512__BM4096__BN64(%arg0: memref<4096x512xf16>, %arg1: memref<512x4096xf16>, %arg2: memref<4096x4096xf16>) {
      %cst = arith.constant 0.000000e+00 : f16
      %c64 = arith.constant 64 : index
      %c0 = arith.constant 0 : index
      %c8 = arith.constant 8 : index
      %c1 = arith.constant 1 : index
      scf.parallel (%arg3, %arg4) = (%c0, %c0) to (%c8, %c8) step (%c1, %c1) {
        %12 = arith.muli %arg3, %c8 overflow<nsw> : index
        %13 = arith.addi %12, %arg4 : index
        %14 = loom.alloc [512, 64] on @L1 : memref<512x64xf16>
        %15 = loom.semaphore_take %14 : memref<512x64xf16> -> memref<512x64xf16>
        %16 = loom.alloc [4096, 512] on @L1 : memref<4096x512xf16>
        %17 = loom.semaphore_take %16 : memref<4096x512xf16> -> memref<4096x512xf16>
        %18 = loom.alloc [4096, 64] on @L1 : memref<4096x64xf16>
        %19 = loom.semaphore_take %18 : memref<4096x64xf16> -> memref<4096x64xf16>
        linalg.fill ins(%cst : f16) outs(%19 : memref<4096x64xf16>)
        %reinterpret_cast = memref.reinterpret_cast %arg0 to offset: [0], sizes: [4096, 512], strides: [512, 1] : memref<4096x512xf16> to memref<4096x512xf16, strided<[512, 1]>>
        %cast = memref.cast %reinterpret_cast : memref<4096x512xf16, strided<[512, 1]>> to memref<4096x512xf16, strided<[512, 1], offset: ?>>
        loom.copy %cast, %17 src_mem_space @DRAM dst_mem_space @L1, interconnect : [], broadcast : [1, 1] : memref<4096x512xf16, strided<[512, 1], offset: ?>>, memref<4096x512xf16>
        %20 = arith.muli %13, %c64 : index
        %reinterpret_cast_0 = memref.reinterpret_cast %arg1 to offset: [%20], sizes: [512, 64], strides: [4096, 1] : memref<512x4096xf16> to memref<512x64xf16, strided<[4096, 1], offset: ?>>
        loom.copy %reinterpret_cast_0, %15 src_mem_space @DRAM dst_mem_space @L1, interconnect : [], broadcast : [1, 1] : memref<512x64xf16, strided<[4096, 1], offset: ?>>, memref<512x64xf16>
        linalg.matmul ins(%17, %15 : memref<4096x512xf16>, memref<512x64xf16>) outs(%19 : memref<4096x64xf16>)
        loom.semaphore_give %15 : memref<512x64xf16>
        loom.semaphore_give %17 : memref<4096x512xf16>
        %reinterpret_cast_1 = memref.reinterpret_cast %arg2 to offset: [%20], sizes: [4096, 64], strides: [4096, 1] : memref<4096x4096xf16> to memref<4096x64xf16, strided<[4096, 1], offset: ?>>
        loom.copy %19, %reinterpret_cast_1 src_mem_space @L1 dst_mem_space @DRAM, interconnect : [], broadcast : [1, 1] : memref<4096x64xf16>, memref<4096x64xf16, strided<[4096, 1], offset: ?>>
        loom.semaphore_give %19 : memref<4096x64xf16>
        scf.reduce 
      } {loom.iter_types = [#loom.iter_type<spatial>, #loom.iter_type<spatial>], loom.mapped_to_dims = [@x, @y]}
      return
    }
  }
  module attributes {loom.pass_name = "Materialize"} {
    func.func @matmul__d0i1_d1i1__f10__a_d__BK512__BM4096__BN64(%arg0: memref<4096x512xf16>, %arg1: memref<512x4096xf16>, %arg2: memref<4096x4096xf16>) {
      %cst = arith.constant 0.000000e+00 : f16
      %c64 = arith.constant 64 : index
      %c0 = arith.constant 0 : index
      %c8 = arith.constant 8 : index
      %c1 = arith.constant 1 : index
      scf.parallel (%arg3, %arg4) = (%c0, %c0) to (%c8, %c8) step (%c1, %c1) {
        %12 = arith.muli %arg3, %c8 overflow<nsw> : index
        %13 = arith.addi %12, %arg4 : index
        %14 = loom.alloc [512, 64] on @L1 : memref<512x64xf16>
        %15 = loom.semaphore_take %14 : memref<512x64xf16> -> memref<512x64xf16>
        %16 = loom.alloc [4096, 512] on @L1 : memref<4096x512xf16>
        %17 = loom.semaphore_take %16 : memref<4096x512xf16> -> memref<4096x512xf16>
        %18 = loom.alloc [4096, 64] on @L1 : memref<4096x64xf16>
        %19 = loom.semaphore_take %18 : memref<4096x64xf16> -> memref<4096x64xf16>
        linalg.fill ins(%cst : f16) outs(%19 : memref<4096x64xf16>)
        %reinterpret_cast = memref.reinterpret_cast %arg0 to offset: [0], sizes: [4096, 512], strides: [512, 1] : memref<4096x512xf16> to memref<4096x512xf16, strided<[512, 1]>>
        %cast = memref.cast %reinterpret_cast : memref<4096x512xf16, strided<[512, 1]>> to memref<4096x512xf16, strided<[512, 1], offset: ?>>
        loom.copy %cast, %17 src_mem_space @DRAM dst_mem_space @L1, interconnect : [@horizontal_links, @vertical_links], broadcast : [8, 8] : memref<4096x512xf16, strided<[512, 1], offset: ?>>, memref<4096x512xf16>
        %20 = arith.muli %13, %c64 : index
        %reinterpret_cast_0 = memref.reinterpret_cast %arg1 to offset: [%20], sizes: [512, 64], strides: [4096, 1] : memref<512x4096xf16> to memref<512x64xf16, strided<[4096, 1], offset: ?>>
        loom.copy %reinterpret_cast_0, %15 src_mem_space @DRAM dst_mem_space @L1, interconnect : [], broadcast : [1, 1] : memref<512x64xf16, strided<[4096, 1], offset: ?>>, memref<512x64xf16>
        linalg.matmul ins(%17, %15 : memref<4096x512xf16>, memref<512x64xf16>) outs(%19 : memref<4096x64xf16>)
        loom.semaphore_give %15 : memref<512x64xf16>
        loom.semaphore_give %17 : memref<4096x512xf16>
        %reinterpret_cast_1 = memref.reinterpret_cast %arg2 to offset: [%20], sizes: [4096, 64], strides: [4096, 1] : memref<4096x4096xf16> to memref<4096x64xf16, strided<[4096, 1], offset: ?>>
        loom.copy %19, %reinterpret_cast_1 src_mem_space @L1 dst_mem_space @DRAM, interconnect : [], broadcast : [1, 1] : memref<4096x64xf16>, memref<4096x64xf16, strided<[4096, 1], offset: ?>>
        loom.semaphore_give %19 : memref<4096x64xf16>
        scf.reduce 
      } {loom.iter_types = [#loom.iter_type<spatial>, #loom.iter_type<spatial>], loom.mapped_to_dims = [@x, @y]}
      return
    }
  }
  module attributes {loom.pass_name = "Materialize"} {
    func.func @matmul__d0i1_d1i1__f10__h_d__BK32__BM4096__BN64(%arg0: memref<4096x512xf16>, %arg1: memref<512x4096xf16>, %arg2: memref<4096x4096xf16>) {
      %c131072 = arith.constant 131072 : index
      %c16 = arith.constant 16 : index
      %cst = arith.constant 0.000000e+00 : f16
      %c64 = arith.constant 64 : index
      %c32 = arith.constant 32 : index
      %c0 = arith.constant 0 : index
      %c8 = arith.constant 8 : index
      %c1 = arith.constant 1 : index
      scf.parallel (%arg3, %arg4) = (%c0, %c0) to (%c8, %c8) step (%c1, %c1) {
        %12 = arith.muli %arg3, %c8 overflow<nsw> : index
        %13 = arith.addi %12, %arg4 : index
        %14 = loom.alloc [32, 64] on @L1 : memref<32x64xf16>
        %15 = loom.semaphore_take %14 : memref<32x64xf16> -> memref<32x64xf16>
        %16 = loom.alloc [4096, 32] on @L1 : memref<4096x32xf16>
        %17 = loom.semaphore_take %16 : memref<4096x32xf16> -> memref<4096x32xf16>
        %18 = loom.alloc [4096, 64] on @L1 : memref<4096x64xf16>
        %19 = loom.semaphore_take %18 : memref<4096x64xf16> -> memref<4096x64xf16>
        linalg.fill ins(%cst : f16) outs(%19 : memref<4096x64xf16>)
        scf.for %arg5 = %c0 to %c16 step %c1 {
          %21 = arith.muli %arg5, %c32 : index
          %reinterpret_cast_0 = memref.reinterpret_cast %arg0 to offset: [%21], sizes: [4096, 32], strides: [512, 1] : memref<4096x512xf16> to memref<4096x32xf16, strided<[512, 1], offset: ?>>
          loom.copy %reinterpret_cast_0, %17 src_mem_space @DRAM dst_mem_space @L1, interconnect : [@horizontal_links], broadcast : [1, 8] : memref<4096x32xf16, strided<[512, 1], offset: ?>>, memref<4096x32xf16>
          %22 = arith.muli %13, %c64 : index
          %23 = arith.muli %arg5, %c131072 : index
          %24 = arith.addi %23, %22 : index
          %reinterpret_cast_1 = memref.reinterpret_cast %arg1 to offset: [%24], sizes: [32, 64], strides: [4096, 1] : memref<512x4096xf16> to memref<32x64xf16, strided<[4096, 1], offset: ?>>
          loom.copy %reinterpret_cast_1, %15 src_mem_space @DRAM dst_mem_space @L1, interconnect : [], broadcast : [1, 1] : memref<32x64xf16, strided<[4096, 1], offset: ?>>, memref<32x64xf16>
          linalg.matmul ins(%17, %15 : memref<4096x32xf16>, memref<32x64xf16>) outs(%19 : memref<4096x64xf16>)
          loom.semaphore_give %15 : memref<32x64xf16>
          loom.semaphore_give %17 : memref<4096x32xf16>
        }
        %20 = arith.muli %13, %c64 : index
        %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%20], sizes: [4096, 64], strides: [4096, 1] : memref<4096x4096xf16> to memref<4096x64xf16, strided<[4096, 1], offset: ?>>
        loom.copy %19, %reinterpret_cast src_mem_space @L1 dst_mem_space @DRAM, interconnect : [], broadcast : [1, 1] : memref<4096x64xf16>, memref<4096x64xf16, strided<[4096, 1], offset: ?>>
        loom.semaphore_give %19 : memref<4096x64xf16>
        scf.reduce 
      } {loom.iter_types = [#loom.iter_type<spatial>, #loom.iter_type<spatial>], loom.mapped_to_dims = [@x, @y]}
      return
    }
  }
  module attributes {loom.pass_name = "Materialize"} {
    func.func @matmul__d0i1_d1i1__f10__v_d__BK128__BM1024__BN64(%arg0: memref<4096x512xf16>, %arg1: memref<512x4096xf16>, %arg2: memref<4096x4096xf16>) {
      %c4194304 = arith.constant 4194304 : index
      %c524288 = arith.constant 524288 : index
      %c4 = arith.constant 4 : index
      %cst = arith.constant 0.000000e+00 : f16
      %c64 = arith.constant 64 : index
      %c128 = arith.constant 128 : index
      %c0 = arith.constant 0 : index
      %c8 = arith.constant 8 : index
      %c1 = arith.constant 1 : index
      scf.parallel (%arg3, %arg4) = (%c0, %c0) to (%c8, %c8) step (%c1, %c1) {
        scf.for %arg5 = %c0 to %c4 step %c1 {
          %12 = arith.muli %arg3, %c8 overflow<nsw> : index
          %13 = arith.addi %12, %arg4 : index
          %14 = loom.alloc [128, 64] on @L1 : memref<128x64xf16>
          %15 = loom.semaphore_take %14 : memref<128x64xf16> -> memref<128x64xf16>
          %16 = loom.alloc [1024, 128] on @L1 : memref<1024x128xf16>
          %17 = loom.semaphore_take %16 : memref<1024x128xf16> -> memref<1024x128xf16>
          %18 = loom.alloc [1024, 64] on @L1 : memref<1024x64xf16>
          %19 = loom.semaphore_take %18 : memref<1024x64xf16> -> memref<1024x64xf16>
          linalg.fill ins(%cst : f16) outs(%19 : memref<1024x64xf16>)
          scf.for %arg6 = %c0 to %c4 step %c1 {
            %23 = arith.muli %arg6, %c128 : index
            %24 = arith.muli %arg5, %c524288 : index
            %25 = arith.addi %24, %23 : index
            %reinterpret_cast_0 = memref.reinterpret_cast %arg0 to offset: [%25], sizes: [1024, 128], strides: [512, 1] : memref<4096x512xf16> to memref<1024x128xf16, strided<[512, 1], offset: ?>>
            loom.copy %reinterpret_cast_0, %17 src_mem_space @DRAM dst_mem_space @L1, interconnect : [@vertical_links], broadcast : [8, 1] : memref<1024x128xf16, strided<[512, 1], offset: ?>>, memref<1024x128xf16>
            %26 = arith.muli %13, %c64 : index
            %27 = arith.muli %arg6, %c524288 : index
            %28 = arith.addi %27, %26 : index
            %reinterpret_cast_1 = memref.reinterpret_cast %arg1 to offset: [%28], sizes: [128, 64], strides: [4096, 1] : memref<512x4096xf16> to memref<128x64xf16, strided<[4096, 1], offset: ?>>
            loom.copy %reinterpret_cast_1, %15 src_mem_space @DRAM dst_mem_space @L1, interconnect : [], broadcast : [1, 1] : memref<128x64xf16, strided<[4096, 1], offset: ?>>, memref<128x64xf16>
            linalg.matmul ins(%17, %15 : memref<1024x128xf16>, memref<128x64xf16>) outs(%19 : memref<1024x64xf16>)
            loom.semaphore_give %15 : memref<128x64xf16>
            loom.semaphore_give %17 : memref<1024x128xf16>
          }
          %20 = arith.muli %13, %c64 : index
          %21 = arith.muli %arg5, %c4194304 : index
          %22 = arith.addi %21, %20 : index
          %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%22], sizes: [1024, 64], strides: [4096, 1] : memref<4096x4096xf16> to memref<1024x64xf16, strided<[4096, 1], offset: ?>>
          loom.copy %19, %reinterpret_cast src_mem_space @L1 dst_mem_space @DRAM, interconnect : [], broadcast : [1, 1] : memref<1024x64xf16>, memref<1024x64xf16, strided<[4096, 1], offset: ?>>
          loom.semaphore_give %19 : memref<1024x64xf16>
        }
        scf.reduce 
      } {loom.iter_types = [#loom.iter_type<spatial>, #loom.iter_type<spatial>], loom.mapped_to_dims = [@x, @y]}
      return
    }
  }
  module attributes {loom.pass_name = "Materialize"} {
    func.func @matmul__d1i1_d0i1__f01__d_d__BK512__BM4096__BN64(%arg0: memref<4096x512xf16>, %arg1: memref<512x4096xf16>, %arg2: memref<4096x4096xf16>) {
      %cst = arith.constant 0.000000e+00 : f16
      %c64 = arith.constant 64 : index
      %c0 = arith.constant 0 : index
      %c8 = arith.constant 8 : index
      %c1 = arith.constant 1 : index
      scf.parallel (%arg3, %arg4) = (%c0, %c0) to (%c8, %c8) step (%c1, %c1) {
        %12 = arith.muli %arg3, %c8 overflow<nsw> : index
        %13 = arith.addi %12, %arg4 : index
        %14 = loom.alloc [512, 64] on @L1 : memref<512x64xf16>
        %15 = loom.semaphore_take %14 : memref<512x64xf16> -> memref<512x64xf16>
        %16 = loom.alloc [4096, 512] on @L1 : memref<4096x512xf16>
        %17 = loom.semaphore_take %16 : memref<4096x512xf16> -> memref<4096x512xf16>
        %18 = loom.alloc [4096, 64] on @L1 : memref<4096x64xf16>
        %19 = loom.semaphore_take %18 : memref<4096x64xf16> -> memref<4096x64xf16>
        linalg.fill ins(%cst : f16) outs(%19 : memref<4096x64xf16>)
        %reinterpret_cast = memref.reinterpret_cast %arg0 to offset: [0], sizes: [4096, 512], strides: [512, 1] : memref<4096x512xf16> to memref<4096x512xf16, strided<[512, 1]>>
        %cast = memref.cast %reinterpret_cast : memref<4096x512xf16, strided<[512, 1]>> to memref<4096x512xf16, strided<[512, 1], offset: ?>>
        loom.copy %cast, %17 src_mem_space @DRAM dst_mem_space @L1, interconnect : [], broadcast : [1, 1] : memref<4096x512xf16, strided<[512, 1], offset: ?>>, memref<4096x512xf16>
        %20 = arith.muli %13, %c64 : index
        %reinterpret_cast_0 = memref.reinterpret_cast %arg1 to offset: [%20], sizes: [512, 64], strides: [4096, 1] : memref<512x4096xf16> to memref<512x64xf16, strided<[4096, 1], offset: ?>>
        loom.copy %reinterpret_cast_0, %15 src_mem_space @DRAM dst_mem_space @L1, interconnect : [], broadcast : [1, 1] : memref<512x64xf16, strided<[4096, 1], offset: ?>>, memref<512x64xf16>
        linalg.matmul ins(%17, %15 : memref<4096x512xf16>, memref<512x64xf16>) outs(%19 : memref<4096x64xf16>)
        loom.semaphore_give %15 : memref<512x64xf16>
        loom.semaphore_give %17 : memref<4096x512xf16>
        %reinterpret_cast_1 = memref.reinterpret_cast %arg2 to offset: [%20], sizes: [4096, 64], strides: [4096, 1] : memref<4096x4096xf16> to memref<4096x64xf16, strided<[4096, 1], offset: ?>>
        loom.copy %19, %reinterpret_cast_1 src_mem_space @L1 dst_mem_space @DRAM, interconnect : [], broadcast : [1, 1] : memref<4096x64xf16>, memref<4096x64xf16, strided<[4096, 1], offset: ?>>
        loom.semaphore_give %19 : memref<4096x64xf16>
        scf.reduce 
      } {loom.iter_types = [#loom.iter_type<spatial>, #loom.iter_type<spatial>], loom.mapped_to_dims = [@y, @x]}
      return
    }
  }
  module attributes {loom.pass_name = "Materialize"} {
    func.func @matmul__d1i1_d0i1__f01__a_d__BK512__BM4096__BN64(%arg0: memref<4096x512xf16>, %arg1: memref<512x4096xf16>, %arg2: memref<4096x4096xf16>) {
      %cst = arith.constant 0.000000e+00 : f16
      %c64 = arith.constant 64 : index
      %c0 = arith.constant 0 : index
      %c8 = arith.constant 8 : index
      %c1 = arith.constant 1 : index
      scf.parallel (%arg3, %arg4) = (%c0, %c0) to (%c8, %c8) step (%c1, %c1) {
        %12 = arith.muli %arg3, %c8 overflow<nsw> : index
        %13 = arith.addi %12, %arg4 : index
        %14 = loom.alloc [512, 64] on @L1 : memref<512x64xf16>
        %15 = loom.semaphore_take %14 : memref<512x64xf16> -> memref<512x64xf16>
        %16 = loom.alloc [4096, 512] on @L1 : memref<4096x512xf16>
        %17 = loom.semaphore_take %16 : memref<4096x512xf16> -> memref<4096x512xf16>
        %18 = loom.alloc [4096, 64] on @L1 : memref<4096x64xf16>
        %19 = loom.semaphore_take %18 : memref<4096x64xf16> -> memref<4096x64xf16>
        linalg.fill ins(%cst : f16) outs(%19 : memref<4096x64xf16>)
        %reinterpret_cast = memref.reinterpret_cast %arg0 to offset: [0], sizes: [4096, 512], strides: [512, 1] : memref<4096x512xf16> to memref<4096x512xf16, strided<[512, 1]>>
        %cast = memref.cast %reinterpret_cast : memref<4096x512xf16, strided<[512, 1]>> to memref<4096x512xf16, strided<[512, 1], offset: ?>>
        loom.copy %cast, %17 src_mem_space @DRAM dst_mem_space @L1, interconnect : [@horizontal_links, @vertical_links], broadcast : [8, 8] : memref<4096x512xf16, strided<[512, 1], offset: ?>>, memref<4096x512xf16>
        %20 = arith.muli %13, %c64 : index
        %reinterpret_cast_0 = memref.reinterpret_cast %arg1 to offset: [%20], sizes: [512, 64], strides: [4096, 1] : memref<512x4096xf16> to memref<512x64xf16, strided<[4096, 1], offset: ?>>
        loom.copy %reinterpret_cast_0, %15 src_mem_space @DRAM dst_mem_space @L1, interconnect : [], broadcast : [1, 1] : memref<512x64xf16, strided<[4096, 1], offset: ?>>, memref<512x64xf16>
        linalg.matmul ins(%17, %15 : memref<4096x512xf16>, memref<512x64xf16>) outs(%19 : memref<4096x64xf16>)
        loom.semaphore_give %15 : memref<512x64xf16>
        loom.semaphore_give %17 : memref<4096x512xf16>
        %reinterpret_cast_1 = memref.reinterpret_cast %arg2 to offset: [%20], sizes: [4096, 64], strides: [4096, 1] : memref<4096x4096xf16> to memref<4096x64xf16, strided<[4096, 1], offset: ?>>
        loom.copy %19, %reinterpret_cast_1 src_mem_space @L1 dst_mem_space @DRAM, interconnect : [], broadcast : [1, 1] : memref<4096x64xf16>, memref<4096x64xf16, strided<[4096, 1], offset: ?>>
        loom.semaphore_give %19 : memref<4096x64xf16>
        scf.reduce 
      } {loom.iter_types = [#loom.iter_type<spatial>, #loom.iter_type<spatial>], loom.mapped_to_dims = [@y, @x]}
      return
    }
  }
  module attributes {loom.pass_name = "Materialize"} {
    func.func @matmul__d1i1_d0i1__f01__h_d__BK128__BM1024__BN64(%arg0: memref<4096x512xf16>, %arg1: memref<512x4096xf16>, %arg2: memref<4096x4096xf16>) {
      %c4194304 = arith.constant 4194304 : index
      %c524288 = arith.constant 524288 : index
      %c4 = arith.constant 4 : index
      %cst = arith.constant 0.000000e+00 : f16
      %c64 = arith.constant 64 : index
      %c128 = arith.constant 128 : index
      %c0 = arith.constant 0 : index
      %c8 = arith.constant 8 : index
      %c1 = arith.constant 1 : index
      scf.parallel (%arg3, %arg4) = (%c0, %c0) to (%c8, %c8) step (%c1, %c1) {
        scf.for %arg5 = %c0 to %c4 step %c1 {
          %12 = arith.muli %arg3, %c8 overflow<nsw> : index
          %13 = arith.addi %12, %arg4 : index
          %14 = loom.alloc [128, 64] on @L1 : memref<128x64xf16>
          %15 = loom.semaphore_take %14 : memref<128x64xf16> -> memref<128x64xf16>
          %16 = loom.alloc [1024, 128] on @L1 : memref<1024x128xf16>
          %17 = loom.semaphore_take %16 : memref<1024x128xf16> -> memref<1024x128xf16>
          %18 = loom.alloc [1024, 64] on @L1 : memref<1024x64xf16>
          %19 = loom.semaphore_take %18 : memref<1024x64xf16> -> memref<1024x64xf16>
          linalg.fill ins(%cst : f16) outs(%19 : memref<1024x64xf16>)
          scf.for %arg6 = %c0 to %c4 step %c1 {
            %23 = arith.muli %arg6, %c128 : index
            %24 = arith.muli %arg5, %c524288 : index
            %25 = arith.addi %24, %23 : index
            %reinterpret_cast_0 = memref.reinterpret_cast %arg0 to offset: [%25], sizes: [1024, 128], strides: [512, 1] : memref<4096x512xf16> to memref<1024x128xf16, strided<[512, 1], offset: ?>>
            loom.copy %reinterpret_cast_0, %17 src_mem_space @DRAM dst_mem_space @L1, interconnect : [@horizontal_links], broadcast : [1, 8] : memref<1024x128xf16, strided<[512, 1], offset: ?>>, memref<1024x128xf16>
            %26 = arith.muli %13, %c64 : index
            %27 = arith.muli %arg6, %c524288 : index
            %28 = arith.addi %27, %26 : index
            %reinterpret_cast_1 = memref.reinterpret_cast %arg1 to offset: [%28], sizes: [128, 64], strides: [4096, 1] : memref<512x4096xf16> to memref<128x64xf16, strided<[4096, 1], offset: ?>>
            loom.copy %reinterpret_cast_1, %15 src_mem_space @DRAM dst_mem_space @L1, interconnect : [], broadcast : [1, 1] : memref<128x64xf16, strided<[4096, 1], offset: ?>>, memref<128x64xf16>
            linalg.matmul ins(%17, %15 : memref<1024x128xf16>, memref<128x64xf16>) outs(%19 : memref<1024x64xf16>)
            loom.semaphore_give %15 : memref<128x64xf16>
            loom.semaphore_give %17 : memref<1024x128xf16>
          }
          %20 = arith.muli %13, %c64 : index
          %21 = arith.muli %arg5, %c4194304 : index
          %22 = arith.addi %21, %20 : index
          %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%22], sizes: [1024, 64], strides: [4096, 1] : memref<4096x4096xf16> to memref<1024x64xf16, strided<[4096, 1], offset: ?>>
          loom.copy %19, %reinterpret_cast src_mem_space @L1 dst_mem_space @DRAM, interconnect : [], broadcast : [1, 1] : memref<1024x64xf16>, memref<1024x64xf16, strided<[4096, 1], offset: ?>>
          loom.semaphore_give %19 : memref<1024x64xf16>
        }
        scf.reduce 
      } {loom.iter_types = [#loom.iter_type<spatial>, #loom.iter_type<spatial>], loom.mapped_to_dims = [@y, @x]}
      return
    }
  }
  module attributes {loom.pass_name = "Materialize"} {
    func.func @matmul__d1i1_d0i1__f01__v_d__BK64__BM2048__BN64(%arg0: memref<4096x512xf16>, %arg1: memref<512x4096xf16>, %arg2: memref<4096x4096xf16>) {
      %c8388608 = arith.constant 8388608 : index
      %c262144 = arith.constant 262144 : index
      %c1048576 = arith.constant 1048576 : index
      %c2 = arith.constant 2 : index
      %cst = arith.constant 0.000000e+00 : f16
      %c64 = arith.constant 64 : index
      %c0 = arith.constant 0 : index
      %c8 = arith.constant 8 : index
      %c1 = arith.constant 1 : index
      scf.parallel (%arg3, %arg4) = (%c0, %c0) to (%c8, %c8) step (%c1, %c1) {
        scf.for %arg5 = %c0 to %c2 step %c1 {
          %12 = arith.muli %arg3, %c8 overflow<nsw> : index
          %13 = arith.addi %12, %arg4 : index
          %14 = loom.alloc [64, 64] on @L1 : memref<64x64xf16>
          %15 = loom.semaphore_take %14 : memref<64x64xf16> -> memref<64x64xf16>
          %16 = loom.alloc [2048, 64] on @L1 : memref<2048x64xf16>
          %17 = loom.semaphore_take %16 : memref<2048x64xf16> -> memref<2048x64xf16>
          %18 = loom.alloc [2048, 64] on @L1 : memref<2048x64xf16>
          %19 = loom.semaphore_take %18 : memref<2048x64xf16> -> memref<2048x64xf16>
          linalg.fill ins(%cst : f16) outs(%19 : memref<2048x64xf16>)
          scf.for %arg6 = %c0 to %c8 step %c1 {
            %23 = arith.muli %arg6, %c64 : index
            %24 = arith.muli %arg5, %c1048576 : index
            %25 = arith.addi %24, %23 : index
            %reinterpret_cast_0 = memref.reinterpret_cast %arg0 to offset: [%25], sizes: [2048, 64], strides: [512, 1] : memref<4096x512xf16> to memref<2048x64xf16, strided<[512, 1], offset: ?>>
            loom.copy %reinterpret_cast_0, %17 src_mem_space @DRAM dst_mem_space @L1, interconnect : [@vertical_links], broadcast : [8, 1] : memref<2048x64xf16, strided<[512, 1], offset: ?>>, memref<2048x64xf16>
            %26 = arith.muli %13, %c64 : index
            %27 = arith.muli %arg6, %c262144 : index
            %28 = arith.addi %27, %26 : index
            %reinterpret_cast_1 = memref.reinterpret_cast %arg1 to offset: [%28], sizes: [64, 64], strides: [4096, 1] : memref<512x4096xf16> to memref<64x64xf16, strided<[4096, 1], offset: ?>>
            loom.copy %reinterpret_cast_1, %15 src_mem_space @DRAM dst_mem_space @L1, interconnect : [], broadcast : [1, 1] : memref<64x64xf16, strided<[4096, 1], offset: ?>>, memref<64x64xf16>
            linalg.matmul ins(%17, %15 : memref<2048x64xf16>, memref<64x64xf16>) outs(%19 : memref<2048x64xf16>)
            loom.semaphore_give %15 : memref<64x64xf16>
            loom.semaphore_give %17 : memref<2048x64xf16>
          }
          %20 = arith.muli %13, %c64 : index
          %21 = arith.muli %arg5, %c8388608 : index
          %22 = arith.addi %21, %20 : index
          %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%22], sizes: [2048, 64], strides: [4096, 1] : memref<4096x4096xf16> to memref<2048x64xf16, strided<[4096, 1], offset: ?>>
          loom.copy %19, %reinterpret_cast src_mem_space @L1 dst_mem_space @DRAM, interconnect : [], broadcast : [1, 1] : memref<2048x64xf16>, memref<2048x64xf16, strided<[4096, 1], offset: ?>>
          loom.semaphore_give %19 : memref<2048x64xf16>
        }
        scf.reduce 
      } {loom.iter_types = [#loom.iter_type<spatial>, #loom.iter_type<spatial>], loom.mapped_to_dims = [@y, @x]}
      return
    }
  }
  module attributes {loom.pass_name = "Materialize"} {
    func.func @matmul__d1i1_d0i1__f10__d_d__BK512__BM4096__BN64(%arg0: memref<4096x512xf16>, %arg1: memref<512x4096xf16>, %arg2: memref<4096x4096xf16>) {
      %cst = arith.constant 0.000000e+00 : f16
      %c64 = arith.constant 64 : index
      %c0 = arith.constant 0 : index
      %c8 = arith.constant 8 : index
      %c1 = arith.constant 1 : index
      scf.parallel (%arg3, %arg4) = (%c0, %c0) to (%c8, %c8) step (%c1, %c1) {
        %12 = arith.muli %arg3, %c8 overflow<nsw> : index
        %13 = arith.addi %12, %arg4 : index
        %14 = loom.alloc [512, 64] on @L1 : memref<512x64xf16>
        %15 = loom.semaphore_take %14 : memref<512x64xf16> -> memref<512x64xf16>
        %16 = loom.alloc [4096, 512] on @L1 : memref<4096x512xf16>
        %17 = loom.semaphore_take %16 : memref<4096x512xf16> -> memref<4096x512xf16>
        %18 = loom.alloc [4096, 64] on @L1 : memref<4096x64xf16>
        %19 = loom.semaphore_take %18 : memref<4096x64xf16> -> memref<4096x64xf16>
        linalg.fill ins(%cst : f16) outs(%19 : memref<4096x64xf16>)
        %reinterpret_cast = memref.reinterpret_cast %arg0 to offset: [0], sizes: [4096, 512], strides: [512, 1] : memref<4096x512xf16> to memref<4096x512xf16, strided<[512, 1]>>
        %cast = memref.cast %reinterpret_cast : memref<4096x512xf16, strided<[512, 1]>> to memref<4096x512xf16, strided<[512, 1], offset: ?>>
        loom.copy %cast, %17 src_mem_space @DRAM dst_mem_space @L1, interconnect : [], broadcast : [1, 1] : memref<4096x512xf16, strided<[512, 1], offset: ?>>, memref<4096x512xf16>
        %20 = arith.muli %13, %c64 : index
        %reinterpret_cast_0 = memref.reinterpret_cast %arg1 to offset: [%20], sizes: [512, 64], strides: [4096, 1] : memref<512x4096xf16> to memref<512x64xf16, strided<[4096, 1], offset: ?>>
        loom.copy %reinterpret_cast_0, %15 src_mem_space @DRAM dst_mem_space @L1, interconnect : [], broadcast : [1, 1] : memref<512x64xf16, strided<[4096, 1], offset: ?>>, memref<512x64xf16>
        linalg.matmul ins(%17, %15 : memref<4096x512xf16>, memref<512x64xf16>) outs(%19 : memref<4096x64xf16>)
        loom.semaphore_give %15 : memref<512x64xf16>
        loom.semaphore_give %17 : memref<4096x512xf16>
        %reinterpret_cast_1 = memref.reinterpret_cast %arg2 to offset: [%20], sizes: [4096, 64], strides: [4096, 1] : memref<4096x4096xf16> to memref<4096x64xf16, strided<[4096, 1], offset: ?>>
        loom.copy %19, %reinterpret_cast_1 src_mem_space @L1 dst_mem_space @DRAM, interconnect : [], broadcast : [1, 1] : memref<4096x64xf16>, memref<4096x64xf16, strided<[4096, 1], offset: ?>>
        loom.semaphore_give %19 : memref<4096x64xf16>
        scf.reduce 
      } {loom.iter_types = [#loom.iter_type<spatial>, #loom.iter_type<spatial>], loom.mapped_to_dims = [@y, @x]}
      return
    }
  }
  module attributes {loom.pass_name = "Materialize"} {
    func.func @matmul__d1i1_d0i1__f10__a_d__BK512__BM4096__BN64(%arg0: memref<4096x512xf16>, %arg1: memref<512x4096xf16>, %arg2: memref<4096x4096xf16>) {
      %cst = arith.constant 0.000000e+00 : f16
      %c64 = arith.constant 64 : index
      %c0 = arith.constant 0 : index
      %c8 = arith.constant 8 : index
      %c1 = arith.constant 1 : index
      scf.parallel (%arg3, %arg4) = (%c0, %c0) to (%c8, %c8) step (%c1, %c1) {
        %12 = arith.muli %arg3, %c8 overflow<nsw> : index
        %13 = arith.addi %12, %arg4 : index
        %14 = loom.alloc [512, 64] on @L1 : memref<512x64xf16>
        %15 = loom.semaphore_take %14 : memref<512x64xf16> -> memref<512x64xf16>
        %16 = loom.alloc [4096, 512] on @L1 : memref<4096x512xf16>
        %17 = loom.semaphore_take %16 : memref<4096x512xf16> -> memref<4096x512xf16>
        %18 = loom.alloc [4096, 64] on @L1 : memref<4096x64xf16>
        %19 = loom.semaphore_take %18 : memref<4096x64xf16> -> memref<4096x64xf16>
        linalg.fill ins(%cst : f16) outs(%19 : memref<4096x64xf16>)
        %reinterpret_cast = memref.reinterpret_cast %arg0 to offset: [0], sizes: [4096, 512], strides: [512, 1] : memref<4096x512xf16> to memref<4096x512xf16, strided<[512, 1]>>
        %cast = memref.cast %reinterpret_cast : memref<4096x512xf16, strided<[512, 1]>> to memref<4096x512xf16, strided<[512, 1], offset: ?>>
        loom.copy %cast, %17 src_mem_space @DRAM dst_mem_space @L1, interconnect : [@horizontal_links, @vertical_links], broadcast : [8, 8] : memref<4096x512xf16, strided<[512, 1], offset: ?>>, memref<4096x512xf16>
        %20 = arith.muli %13, %c64 : index
        %reinterpret_cast_0 = memref.reinterpret_cast %arg1 to offset: [%20], sizes: [512, 64], strides: [4096, 1] : memref<512x4096xf16> to memref<512x64xf16, strided<[4096, 1], offset: ?>>
        loom.copy %reinterpret_cast_0, %15 src_mem_space @DRAM dst_mem_space @L1, interconnect : [], broadcast : [1, 1] : memref<512x64xf16, strided<[4096, 1], offset: ?>>, memref<512x64xf16>
        linalg.matmul ins(%17, %15 : memref<4096x512xf16>, memref<512x64xf16>) outs(%19 : memref<4096x64xf16>)
        loom.semaphore_give %15 : memref<512x64xf16>
        loom.semaphore_give %17 : memref<4096x512xf16>
        %reinterpret_cast_1 = memref.reinterpret_cast %arg2 to offset: [%20], sizes: [4096, 64], strides: [4096, 1] : memref<4096x4096xf16> to memref<4096x64xf16, strided<[4096, 1], offset: ?>>
        loom.copy %19, %reinterpret_cast_1 src_mem_space @L1 dst_mem_space @DRAM, interconnect : [], broadcast : [1, 1] : memref<4096x64xf16>, memref<4096x64xf16, strided<[4096, 1], offset: ?>>
        loom.semaphore_give %19 : memref<4096x64xf16>
        scf.reduce 
      } {loom.iter_types = [#loom.iter_type<spatial>, #loom.iter_type<spatial>], loom.mapped_to_dims = [@y, @x]}
      return
    }
  }
  module attributes {loom.pass_name = "Materialize"} {
    func.func @matmul__d1i1_d0i1__f10__h_d__BK32__BM4096__BN64(%arg0: memref<4096x512xf16>, %arg1: memref<512x4096xf16>, %arg2: memref<4096x4096xf16>) {
      %c131072 = arith.constant 131072 : index
      %c16 = arith.constant 16 : index
      %cst = arith.constant 0.000000e+00 : f16
      %c64 = arith.constant 64 : index
      %c32 = arith.constant 32 : index
      %c0 = arith.constant 0 : index
      %c8 = arith.constant 8 : index
      %c1 = arith.constant 1 : index
      scf.parallel (%arg3, %arg4) = (%c0, %c0) to (%c8, %c8) step (%c1, %c1) {
        %12 = arith.muli %arg3, %c8 overflow<nsw> : index
        %13 = arith.addi %12, %arg4 : index
        %14 = loom.alloc [32, 64] on @L1 : memref<32x64xf16>
        %15 = loom.semaphore_take %14 : memref<32x64xf16> -> memref<32x64xf16>
        %16 = loom.alloc [4096, 32] on @L1 : memref<4096x32xf16>
        %17 = loom.semaphore_take %16 : memref<4096x32xf16> -> memref<4096x32xf16>
        %18 = loom.alloc [4096, 64] on @L1 : memref<4096x64xf16>
        %19 = loom.semaphore_take %18 : memref<4096x64xf16> -> memref<4096x64xf16>
        linalg.fill ins(%cst : f16) outs(%19 : memref<4096x64xf16>)
        scf.for %arg5 = %c0 to %c16 step %c1 {
          %21 = arith.muli %arg5, %c32 : index
          %reinterpret_cast_0 = memref.reinterpret_cast %arg0 to offset: [%21], sizes: [4096, 32], strides: [512, 1] : memref<4096x512xf16> to memref<4096x32xf16, strided<[512, 1], offset: ?>>
          loom.copy %reinterpret_cast_0, %17 src_mem_space @DRAM dst_mem_space @L1, interconnect : [@horizontal_links], broadcast : [1, 8] : memref<4096x32xf16, strided<[512, 1], offset: ?>>, memref<4096x32xf16>
          %22 = arith.muli %13, %c64 : index
          %23 = arith.muli %arg5, %c131072 : index
          %24 = arith.addi %23, %22 : index
          %reinterpret_cast_1 = memref.reinterpret_cast %arg1 to offset: [%24], sizes: [32, 64], strides: [4096, 1] : memref<512x4096xf16> to memref<32x64xf16, strided<[4096, 1], offset: ?>>
          loom.copy %reinterpret_cast_1, %15 src_mem_space @DRAM dst_mem_space @L1, interconnect : [], broadcast : [1, 1] : memref<32x64xf16, strided<[4096, 1], offset: ?>>, memref<32x64xf16>
          linalg.matmul ins(%17, %15 : memref<4096x32xf16>, memref<32x64xf16>) outs(%19 : memref<4096x64xf16>)
          loom.semaphore_give %15 : memref<32x64xf16>
          loom.semaphore_give %17 : memref<4096x32xf16>
        }
        %20 = arith.muli %13, %c64 : index
        %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%20], sizes: [4096, 64], strides: [4096, 1] : memref<4096x4096xf16> to memref<4096x64xf16, strided<[4096, 1], offset: ?>>
        loom.copy %19, %reinterpret_cast src_mem_space @L1 dst_mem_space @DRAM, interconnect : [], broadcast : [1, 1] : memref<4096x64xf16>, memref<4096x64xf16, strided<[4096, 1], offset: ?>>
        loom.semaphore_give %19 : memref<4096x64xf16>
        scf.reduce 
      } {loom.iter_types = [#loom.iter_type<spatial>, #loom.iter_type<spatial>], loom.mapped_to_dims = [@y, @x]}
      return
    }
  }
  module attributes {loom.pass_name = "Materialize"} {
    func.func @matmul__d1i1_d0i1__f10__v_d__BK128__BM1024__BN64(%arg0: memref<4096x512xf16>, %arg1: memref<512x4096xf16>, %arg2: memref<4096x4096xf16>) {
      %c4194304 = arith.constant 4194304 : index
      %c524288 = arith.constant 524288 : index
      %c4 = arith.constant 4 : index
      %cst = arith.constant 0.000000e+00 : f16
      %c64 = arith.constant 64 : index
      %c128 = arith.constant 128 : index
      %c0 = arith.constant 0 : index
      %c8 = arith.constant 8 : index
      %c1 = arith.constant 1 : index
      scf.parallel (%arg3, %arg4) = (%c0, %c0) to (%c8, %c8) step (%c1, %c1) {
        scf.for %arg5 = %c0 to %c4 step %c1 {
          %12 = arith.muli %arg3, %c8 overflow<nsw> : index
          %13 = arith.addi %12, %arg4 : index
          %14 = loom.alloc [128, 64] on @L1 : memref<128x64xf16>
          %15 = loom.semaphore_take %14 : memref<128x64xf16> -> memref<128x64xf16>
          %16 = loom.alloc [1024, 128] on @L1 : memref<1024x128xf16>
          %17 = loom.semaphore_take %16 : memref<1024x128xf16> -> memref<1024x128xf16>
          %18 = loom.alloc [1024, 64] on @L1 : memref<1024x64xf16>
          %19 = loom.semaphore_take %18 : memref<1024x64xf16> -> memref<1024x64xf16>
          linalg.fill ins(%cst : f16) outs(%19 : memref<1024x64xf16>)
          scf.for %arg6 = %c0 to %c4 step %c1 {
            %23 = arith.muli %arg6, %c128 : index
            %24 = arith.muli %arg5, %c524288 : index
            %25 = arith.addi %24, %23 : index
            %reinterpret_cast_0 = memref.reinterpret_cast %arg0 to offset: [%25], sizes: [1024, 128], strides: [512, 1] : memref<4096x512xf16> to memref<1024x128xf16, strided<[512, 1], offset: ?>>
            loom.copy %reinterpret_cast_0, %17 src_mem_space @DRAM dst_mem_space @L1, interconnect : [@vertical_links], broadcast : [8, 1] : memref<1024x128xf16, strided<[512, 1], offset: ?>>, memref<1024x128xf16>
            %26 = arith.muli %13, %c64 : index
            %27 = arith.muli %arg6, %c524288 : index
            %28 = arith.addi %27, %26 : index
            %reinterpret_cast_1 = memref.reinterpret_cast %arg1 to offset: [%28], sizes: [128, 64], strides: [4096, 1] : memref<512x4096xf16> to memref<128x64xf16, strided<[4096, 1], offset: ?>>
            loom.copy %reinterpret_cast_1, %15 src_mem_space @DRAM dst_mem_space @L1, interconnect : [], broadcast : [1, 1] : memref<128x64xf16, strided<[4096, 1], offset: ?>>, memref<128x64xf16>
            linalg.matmul ins(%17, %15 : memref<1024x128xf16>, memref<128x64xf16>) outs(%19 : memref<1024x64xf16>)
            loom.semaphore_give %15 : memref<128x64xf16>
            loom.semaphore_give %17 : memref<1024x128xf16>
          }
          %20 = arith.muli %13, %c64 : index
          %21 = arith.muli %arg5, %c4194304 : index
          %22 = arith.addi %21, %20 : index
          %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%22], sizes: [1024, 64], strides: [4096, 1] : memref<4096x4096xf16> to memref<1024x64xf16, strided<[4096, 1], offset: ?>>
          loom.copy %19, %reinterpret_cast src_mem_space @L1 dst_mem_space @DRAM, interconnect : [], broadcast : [1, 1] : memref<1024x64xf16>, memref<1024x64xf16, strided<[4096, 1], offset: ?>>
          loom.semaphore_give %19 : memref<1024x64xf16>
        }
        scf.reduce 
      } {loom.iter_types = [#loom.iter_type<spatial>, #loom.iter_type<spatial>], loom.mapped_to_dims = [@y, @x]}
      return
    }
  }
}
