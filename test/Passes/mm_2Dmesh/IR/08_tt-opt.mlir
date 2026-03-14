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
    func.func @matmul__d0i0_d1i0__f01__d_d__BK64__BM64__BN64(%arg0: memref<4096x512xf16>, %arg1: memref<512x4096xf16>, %arg2: memref<4096x4096xf16>) {
      %c262144 = arith.constant 262144 : index
      %c32768 = arith.constant 32768 : index
      %c64 = arith.constant 64 : index
      %c0 = arith.constant 0 : index
      %c8 = arith.constant 8 : index
      %c1 = arith.constant 1 : index
      scf.parallel (%arg3, %arg4) = (%c0, %c0) to (%c8, %c8) step (%c1, %c1) {
        scf.for %arg5 = %c0 to %c64 step %c1 {
          %12 = arith.muli %arg3, %c8 overflow<nsw> : index
          %13 = arith.addi %12, %arg4 : index
          %14 = loom.alloc [64, 64] on @L1 : memref<64x64xf16>
          %15 = loom.semaphore_take %14 : memref<64x64xf16> -> memref<64x64xf16>
          %16 = loom.alloc [64, 64] on @L1 : memref<64x64xf16>
          %17 = loom.semaphore_take %16 : memref<64x64xf16> -> memref<64x64xf16>
          %18 = loom.alloc [64, 64] on @L1 : memref<64x64xf16>
          %19 = loom.semaphore_take %18 : memref<64x64xf16> -> memref<64x64xf16>
          scf.for %arg6 = %c0 to %c8 step %c1 {
            %23 = arith.muli %arg6, %c64 : index
            %24 = arith.muli %13, %c32768 : index
            %25 = arith.addi %24, %23 : index
            %reinterpret_cast_0 = memref.reinterpret_cast %arg0 to offset: [%25], sizes: [64, 64], strides: [512, 1] : memref<4096x512xf16> to memref<64x64xf16, strided<[512, 1], offset: ?>>
            loom.copy %reinterpret_cast_0, %17 src_mem_space @DRAM dst_mem_space @L1, interconnect : [], broadcast : [1, 1] : memref<64x64xf16, strided<[512, 1], offset: ?>>, memref<64x64xf16>
            %26 = arith.muli %arg5, %c64 : index
            %27 = arith.muli %arg6, %c262144 : index
            %28 = arith.addi %27, %26 : index
            %reinterpret_cast_1 = memref.reinterpret_cast %arg1 to offset: [%28], sizes: [64, 64], strides: [4096, 1] : memref<512x4096xf16> to memref<64x64xf16, strided<[4096, 1], offset: ?>>
            loom.copy %reinterpret_cast_1, %15 src_mem_space @DRAM dst_mem_space @L1, interconnect : [], broadcast : [1, 1] : memref<64x64xf16, strided<[4096, 1], offset: ?>>, memref<64x64xf16>
            loom.matmul ins(%17, %15 : memref<64x64xf16>, memref<64x64xf16>) outs(%19 : memref<64x64xf16>)
            loom.semaphore_give %15 : memref<64x64xf16>
            loom.semaphore_give %17 : memref<64x64xf16>
          }
          %20 = arith.muli %arg5, %c64 : index
          %21 = arith.muli %13, %c262144 : index
          %22 = arith.addi %21, %20 : index
          %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%22], sizes: [64, 64], strides: [4096, 1] : memref<4096x4096xf16> to memref<64x64xf16, strided<[4096, 1], offset: ?>>
          loom.copy %19, %reinterpret_cast src_mem_space @L1 dst_mem_space @DRAM, interconnect : [], broadcast : [1, 1] : memref<64x64xf16>, memref<64x64xf16, strided<[4096, 1], offset: ?>>
          loom.semaphore_give %19 : memref<64x64xf16>
        }
        scf.reduce 
      } {loom.iter_types = [#loom.iter_type<spatial>, #loom.iter_type<spatial>], loom.mapped_to_dims = [@x, @y]}
      return
    }
  }
  module attributes {loom.pass_name = "Materialize"} {
    func.func @matmul__d0i0_d1i0__f01__d_a__BK64__BM64__BN64(%arg0: memref<4096x512xf16>, %arg1: memref<512x4096xf16>, %arg2: memref<4096x4096xf16>) {
      %c262144 = arith.constant 262144 : index
      %c32768 = arith.constant 32768 : index
      %c64 = arith.constant 64 : index
      %c0 = arith.constant 0 : index
      %c8 = arith.constant 8 : index
      %c1 = arith.constant 1 : index
      scf.parallel (%arg3, %arg4) = (%c0, %c0) to (%c8, %c8) step (%c1, %c1) {
        scf.for %arg5 = %c0 to %c64 step %c1 {
          %12 = arith.muli %arg3, %c8 overflow<nsw> : index
          %13 = arith.addi %12, %arg4 : index
          %14 = loom.alloc [64, 64] on @L1 : memref<64x64xf16>
          %15 = loom.semaphore_take %14 : memref<64x64xf16> -> memref<64x64xf16>
          %16 = loom.alloc [64, 64] on @L1 : memref<64x64xf16>
          %17 = loom.semaphore_take %16 : memref<64x64xf16> -> memref<64x64xf16>
          %18 = loom.alloc [64, 64] on @L1 : memref<64x64xf16>
          %19 = loom.semaphore_take %18 : memref<64x64xf16> -> memref<64x64xf16>
          scf.for %arg6 = %c0 to %c8 step %c1 {
            %23 = arith.muli %arg6, %c64 : index
            %24 = arith.muli %13, %c32768 : index
            %25 = arith.addi %24, %23 : index
            %reinterpret_cast_0 = memref.reinterpret_cast %arg0 to offset: [%25], sizes: [64, 64], strides: [512, 1] : memref<4096x512xf16> to memref<64x64xf16, strided<[512, 1], offset: ?>>
            loom.copy %reinterpret_cast_0, %17 src_mem_space @DRAM dst_mem_space @L1, interconnect : [], broadcast : [1, 1] : memref<64x64xf16, strided<[512, 1], offset: ?>>, memref<64x64xf16>
            %26 = arith.muli %arg5, %c64 : index
            %27 = arith.muli %arg6, %c262144 : index
            %28 = arith.addi %27, %26 : index
            %reinterpret_cast_1 = memref.reinterpret_cast %arg1 to offset: [%28], sizes: [64, 64], strides: [4096, 1] : memref<512x4096xf16> to memref<64x64xf16, strided<[4096, 1], offset: ?>>
            loom.copy %reinterpret_cast_1, %15 src_mem_space @DRAM dst_mem_space @L1, interconnect : [@horizontal_links, @vertical_links], broadcast : [8, 8] : memref<64x64xf16, strided<[4096, 1], offset: ?>>, memref<64x64xf16>
            loom.matmul ins(%17, %15 : memref<64x64xf16>, memref<64x64xf16>) outs(%19 : memref<64x64xf16>)
            loom.semaphore_give %15 : memref<64x64xf16>
            loom.semaphore_give %17 : memref<64x64xf16>
          }
          %20 = arith.muli %arg5, %c64 : index
          %21 = arith.muli %13, %c262144 : index
          %22 = arith.addi %21, %20 : index
          %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%22], sizes: [64, 64], strides: [4096, 1] : memref<4096x4096xf16> to memref<64x64xf16, strided<[4096, 1], offset: ?>>
          loom.copy %19, %reinterpret_cast src_mem_space @L1 dst_mem_space @DRAM, interconnect : [], broadcast : [1, 1] : memref<64x64xf16>, memref<64x64xf16, strided<[4096, 1], offset: ?>>
          loom.semaphore_give %19 : memref<64x64xf16>
        }
        scf.reduce 
      } {loom.iter_types = [#loom.iter_type<spatial>, #loom.iter_type<spatial>], loom.mapped_to_dims = [@x, @y]}
      return
    }
  }
  module attributes {loom.pass_name = "Materialize"} {
    func.func @matmul__d0i0_d1i0__f01__d_h__BK64__BM64__BN64(%arg0: memref<4096x512xf16>, %arg1: memref<512x4096xf16>, %arg2: memref<4096x4096xf16>) {
      %c262144 = arith.constant 262144 : index
      %c32768 = arith.constant 32768 : index
      %c64 = arith.constant 64 : index
      %c0 = arith.constant 0 : index
      %c8 = arith.constant 8 : index
      %c1 = arith.constant 1 : index
      scf.parallel (%arg3, %arg4) = (%c0, %c0) to (%c8, %c8) step (%c1, %c1) {
        scf.for %arg5 = %c0 to %c64 step %c1 {
          %12 = arith.muli %arg3, %c8 overflow<nsw> : index
          %13 = arith.addi %12, %arg4 : index
          %14 = loom.alloc [64, 64] on @L1 : memref<64x64xf16>
          %15 = loom.semaphore_take %14 : memref<64x64xf16> -> memref<64x64xf16>
          %16 = loom.alloc [64, 64] on @L1 : memref<64x64xf16>
          %17 = loom.semaphore_take %16 : memref<64x64xf16> -> memref<64x64xf16>
          %18 = loom.alloc [64, 64] on @L1 : memref<64x64xf16>
          %19 = loom.semaphore_take %18 : memref<64x64xf16> -> memref<64x64xf16>
          scf.for %arg6 = %c0 to %c8 step %c1 {
            %23 = arith.muli %arg6, %c64 : index
            %24 = arith.muli %13, %c32768 : index
            %25 = arith.addi %24, %23 : index
            %reinterpret_cast_0 = memref.reinterpret_cast %arg0 to offset: [%25], sizes: [64, 64], strides: [512, 1] : memref<4096x512xf16> to memref<64x64xf16, strided<[512, 1], offset: ?>>
            loom.copy %reinterpret_cast_0, %17 src_mem_space @DRAM dst_mem_space @L1, interconnect : [], broadcast : [1, 1] : memref<64x64xf16, strided<[512, 1], offset: ?>>, memref<64x64xf16>
            %26 = arith.muli %arg5, %c64 : index
            %27 = arith.muli %arg6, %c262144 : index
            %28 = arith.addi %27, %26 : index
            %reinterpret_cast_1 = memref.reinterpret_cast %arg1 to offset: [%28], sizes: [64, 64], strides: [4096, 1] : memref<512x4096xf16> to memref<64x64xf16, strided<[4096, 1], offset: ?>>
            loom.copy %reinterpret_cast_1, %15 src_mem_space @DRAM dst_mem_space @L1, interconnect : [@horizontal_links], broadcast : [1, 8] : memref<64x64xf16, strided<[4096, 1], offset: ?>>, memref<64x64xf16>
            loom.matmul ins(%17, %15 : memref<64x64xf16>, memref<64x64xf16>) outs(%19 : memref<64x64xf16>)
            loom.semaphore_give %15 : memref<64x64xf16>
            loom.semaphore_give %17 : memref<64x64xf16>
          }
          %20 = arith.muli %arg5, %c64 : index
          %21 = arith.muli %13, %c262144 : index
          %22 = arith.addi %21, %20 : index
          %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%22], sizes: [64, 64], strides: [4096, 1] : memref<4096x4096xf16> to memref<64x64xf16, strided<[4096, 1], offset: ?>>
          loom.copy %19, %reinterpret_cast src_mem_space @L1 dst_mem_space @DRAM, interconnect : [], broadcast : [1, 1] : memref<64x64xf16>, memref<64x64xf16, strided<[4096, 1], offset: ?>>
          loom.semaphore_give %19 : memref<64x64xf16>
        }
        scf.reduce 
      } {loom.iter_types = [#loom.iter_type<spatial>, #loom.iter_type<spatial>], loom.mapped_to_dims = [@x, @y]}
      return
    }
  }
  module attributes {loom.pass_name = "Materialize"} {
    func.func @matmul__d0i0_d1i0__f01__d_v__BK64__BM64__BN64(%arg0: memref<4096x512xf16>, %arg1: memref<512x4096xf16>, %arg2: memref<4096x4096xf16>) {
      %c262144 = arith.constant 262144 : index
      %c32768 = arith.constant 32768 : index
      %c64 = arith.constant 64 : index
      %c0 = arith.constant 0 : index
      %c8 = arith.constant 8 : index
      %c1 = arith.constant 1 : index
      scf.parallel (%arg3, %arg4) = (%c0, %c0) to (%c8, %c8) step (%c1, %c1) {
        scf.for %arg5 = %c0 to %c64 step %c1 {
          %12 = arith.muli %arg3, %c8 overflow<nsw> : index
          %13 = arith.addi %12, %arg4 : index
          %14 = loom.alloc [64, 64] on @L1 : memref<64x64xf16>
          %15 = loom.semaphore_take %14 : memref<64x64xf16> -> memref<64x64xf16>
          %16 = loom.alloc [64, 64] on @L1 : memref<64x64xf16>
          %17 = loom.semaphore_take %16 : memref<64x64xf16> -> memref<64x64xf16>
          %18 = loom.alloc [64, 64] on @L1 : memref<64x64xf16>
          %19 = loom.semaphore_take %18 : memref<64x64xf16> -> memref<64x64xf16>
          scf.for %arg6 = %c0 to %c8 step %c1 {
            %23 = arith.muli %arg6, %c64 : index
            %24 = arith.muli %13, %c32768 : index
            %25 = arith.addi %24, %23 : index
            %reinterpret_cast_0 = memref.reinterpret_cast %arg0 to offset: [%25], sizes: [64, 64], strides: [512, 1] : memref<4096x512xf16> to memref<64x64xf16, strided<[512, 1], offset: ?>>
            loom.copy %reinterpret_cast_0, %17 src_mem_space @DRAM dst_mem_space @L1, interconnect : [], broadcast : [1, 1] : memref<64x64xf16, strided<[512, 1], offset: ?>>, memref<64x64xf16>
            %26 = arith.muli %arg5, %c64 : index
            %27 = arith.muli %arg6, %c262144 : index
            %28 = arith.addi %27, %26 : index
            %reinterpret_cast_1 = memref.reinterpret_cast %arg1 to offset: [%28], sizes: [64, 64], strides: [4096, 1] : memref<512x4096xf16> to memref<64x64xf16, strided<[4096, 1], offset: ?>>
            loom.copy %reinterpret_cast_1, %15 src_mem_space @DRAM dst_mem_space @L1, interconnect : [@vertical_links], broadcast : [8, 1] : memref<64x64xf16, strided<[4096, 1], offset: ?>>, memref<64x64xf16>
            loom.matmul ins(%17, %15 : memref<64x64xf16>, memref<64x64xf16>) outs(%19 : memref<64x64xf16>)
            loom.semaphore_give %15 : memref<64x64xf16>
            loom.semaphore_give %17 : memref<64x64xf16>
          }
          %20 = arith.muli %arg5, %c64 : index
          %21 = arith.muli %13, %c262144 : index
          %22 = arith.addi %21, %20 : index
          %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%22], sizes: [64, 64], strides: [4096, 1] : memref<4096x4096xf16> to memref<64x64xf16, strided<[4096, 1], offset: ?>>
          loom.copy %19, %reinterpret_cast src_mem_space @L1 dst_mem_space @DRAM, interconnect : [], broadcast : [1, 1] : memref<64x64xf16>, memref<64x64xf16, strided<[4096, 1], offset: ?>>
          loom.semaphore_give %19 : memref<64x64xf16>
        }
        scf.reduce 
      } {loom.iter_types = [#loom.iter_type<spatial>, #loom.iter_type<spatial>], loom.mapped_to_dims = [@x, @y]}
      return
    }
  }
  module attributes {loom.pass_name = "Materialize"} {
    func.func @matmul__d0i0_d1i0__f10__d_d__BK64__BM64__BN64(%arg0: memref<4096x512xf16>, %arg1: memref<512x4096xf16>, %arg2: memref<4096x4096xf16>) {
      %c262144 = arith.constant 262144 : index
      %c32768 = arith.constant 32768 : index
      %c64 = arith.constant 64 : index
      %c0 = arith.constant 0 : index
      %c8 = arith.constant 8 : index
      %c1 = arith.constant 1 : index
      scf.parallel (%arg3, %arg4) = (%c0, %c0) to (%c8, %c8) step (%c1, %c1) {
        scf.for %arg5 = %c0 to %c64 step %c1 {
          %12 = arith.muli %arg3, %c8 overflow<nsw> : index
          %13 = arith.addi %12, %arg4 : index
          %14 = loom.alloc [64, 64] on @L1 : memref<64x64xf16>
          %15 = loom.semaphore_take %14 : memref<64x64xf16> -> memref<64x64xf16>
          %16 = loom.alloc [64, 64] on @L1 : memref<64x64xf16>
          %17 = loom.semaphore_take %16 : memref<64x64xf16> -> memref<64x64xf16>
          %18 = loom.alloc [64, 64] on @L1 : memref<64x64xf16>
          %19 = loom.semaphore_take %18 : memref<64x64xf16> -> memref<64x64xf16>
          scf.for %arg6 = %c0 to %c8 step %c1 {
            %23 = arith.muli %arg6, %c64 : index
            %24 = arith.muli %13, %c32768 : index
            %25 = arith.addi %24, %23 : index
            %reinterpret_cast_0 = memref.reinterpret_cast %arg0 to offset: [%25], sizes: [64, 64], strides: [512, 1] : memref<4096x512xf16> to memref<64x64xf16, strided<[512, 1], offset: ?>>
            loom.copy %reinterpret_cast_0, %17 src_mem_space @DRAM dst_mem_space @L1, interconnect : [], broadcast : [1, 1] : memref<64x64xf16, strided<[512, 1], offset: ?>>, memref<64x64xf16>
            %26 = arith.muli %arg5, %c64 : index
            %27 = arith.muli %arg6, %c262144 : index
            %28 = arith.addi %27, %26 : index
            %reinterpret_cast_1 = memref.reinterpret_cast %arg1 to offset: [%28], sizes: [64, 64], strides: [4096, 1] : memref<512x4096xf16> to memref<64x64xf16, strided<[4096, 1], offset: ?>>
            loom.copy %reinterpret_cast_1, %15 src_mem_space @DRAM dst_mem_space @L1, interconnect : [], broadcast : [1, 1] : memref<64x64xf16, strided<[4096, 1], offset: ?>>, memref<64x64xf16>
            loom.matmul ins(%17, %15 : memref<64x64xf16>, memref<64x64xf16>) outs(%19 : memref<64x64xf16>)
            loom.semaphore_give %15 : memref<64x64xf16>
            loom.semaphore_give %17 : memref<64x64xf16>
          }
          %20 = arith.muli %arg5, %c64 : index
          %21 = arith.muli %13, %c262144 : index
          %22 = arith.addi %21, %20 : index
          %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%22], sizes: [64, 64], strides: [4096, 1] : memref<4096x4096xf16> to memref<64x64xf16, strided<[4096, 1], offset: ?>>
          loom.copy %19, %reinterpret_cast src_mem_space @L1 dst_mem_space @DRAM, interconnect : [], broadcast : [1, 1] : memref<64x64xf16>, memref<64x64xf16, strided<[4096, 1], offset: ?>>
          loom.semaphore_give %19 : memref<64x64xf16>
        }
        scf.reduce 
      } {loom.iter_types = [#loom.iter_type<spatial>, #loom.iter_type<spatial>], loom.mapped_to_dims = [@x, @y]}
      return
    }
  }
  module attributes {loom.pass_name = "Materialize"} {
    func.func @matmul__d0i0_d1i0__f10__d_a__BK64__BM64__BN64(%arg0: memref<4096x512xf16>, %arg1: memref<512x4096xf16>, %arg2: memref<4096x4096xf16>) {
      %c262144 = arith.constant 262144 : index
      %c32768 = arith.constant 32768 : index
      %c64 = arith.constant 64 : index
      %c0 = arith.constant 0 : index
      %c8 = arith.constant 8 : index
      %c1 = arith.constant 1 : index
      scf.parallel (%arg3, %arg4) = (%c0, %c0) to (%c8, %c8) step (%c1, %c1) {
        scf.for %arg5 = %c0 to %c64 step %c1 {
          %12 = arith.muli %arg3, %c8 overflow<nsw> : index
          %13 = arith.addi %12, %arg4 : index
          %14 = loom.alloc [64, 64] on @L1 : memref<64x64xf16>
          %15 = loom.semaphore_take %14 : memref<64x64xf16> -> memref<64x64xf16>
          %16 = loom.alloc [64, 64] on @L1 : memref<64x64xf16>
          %17 = loom.semaphore_take %16 : memref<64x64xf16> -> memref<64x64xf16>
          %18 = loom.alloc [64, 64] on @L1 : memref<64x64xf16>
          %19 = loom.semaphore_take %18 : memref<64x64xf16> -> memref<64x64xf16>
          scf.for %arg6 = %c0 to %c8 step %c1 {
            %23 = arith.muli %arg6, %c64 : index
            %24 = arith.muli %13, %c32768 : index
            %25 = arith.addi %24, %23 : index
            %reinterpret_cast_0 = memref.reinterpret_cast %arg0 to offset: [%25], sizes: [64, 64], strides: [512, 1] : memref<4096x512xf16> to memref<64x64xf16, strided<[512, 1], offset: ?>>
            loom.copy %reinterpret_cast_0, %17 src_mem_space @DRAM dst_mem_space @L1, interconnect : [], broadcast : [1, 1] : memref<64x64xf16, strided<[512, 1], offset: ?>>, memref<64x64xf16>
            %26 = arith.muli %arg5, %c64 : index
            %27 = arith.muli %arg6, %c262144 : index
            %28 = arith.addi %27, %26 : index
            %reinterpret_cast_1 = memref.reinterpret_cast %arg1 to offset: [%28], sizes: [64, 64], strides: [4096, 1] : memref<512x4096xf16> to memref<64x64xf16, strided<[4096, 1], offset: ?>>
            loom.copy %reinterpret_cast_1, %15 src_mem_space @DRAM dst_mem_space @L1, interconnect : [@horizontal_links, @vertical_links], broadcast : [8, 8] : memref<64x64xf16, strided<[4096, 1], offset: ?>>, memref<64x64xf16>
            loom.matmul ins(%17, %15 : memref<64x64xf16>, memref<64x64xf16>) outs(%19 : memref<64x64xf16>)
            loom.semaphore_give %15 : memref<64x64xf16>
            loom.semaphore_give %17 : memref<64x64xf16>
          }
          %20 = arith.muli %arg5, %c64 : index
          %21 = arith.muli %13, %c262144 : index
          %22 = arith.addi %21, %20 : index
          %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%22], sizes: [64, 64], strides: [4096, 1] : memref<4096x4096xf16> to memref<64x64xf16, strided<[4096, 1], offset: ?>>
          loom.copy %19, %reinterpret_cast src_mem_space @L1 dst_mem_space @DRAM, interconnect : [], broadcast : [1, 1] : memref<64x64xf16>, memref<64x64xf16, strided<[4096, 1], offset: ?>>
          loom.semaphore_give %19 : memref<64x64xf16>
        }
        scf.reduce 
      } {loom.iter_types = [#loom.iter_type<spatial>, #loom.iter_type<spatial>], loom.mapped_to_dims = [@x, @y]}
      return
    }
  }
  module attributes {loom.pass_name = "Materialize"} {
    func.func @matmul__d0i0_d1i0__f10__d_h__BK64__BM64__BN64(%arg0: memref<4096x512xf16>, %arg1: memref<512x4096xf16>, %arg2: memref<4096x4096xf16>) {
      %c262144 = arith.constant 262144 : index
      %c32768 = arith.constant 32768 : index
      %c64 = arith.constant 64 : index
      %c0 = arith.constant 0 : index
      %c8 = arith.constant 8 : index
      %c1 = arith.constant 1 : index
      scf.parallel (%arg3, %arg4) = (%c0, %c0) to (%c8, %c8) step (%c1, %c1) {
        scf.for %arg5 = %c0 to %c64 step %c1 {
          %12 = arith.muli %arg3, %c8 overflow<nsw> : index
          %13 = arith.addi %12, %arg4 : index
          %14 = loom.alloc [64, 64] on @L1 : memref<64x64xf16>
          %15 = loom.semaphore_take %14 : memref<64x64xf16> -> memref<64x64xf16>
          %16 = loom.alloc [64, 64] on @L1 : memref<64x64xf16>
          %17 = loom.semaphore_take %16 : memref<64x64xf16> -> memref<64x64xf16>
          %18 = loom.alloc [64, 64] on @L1 : memref<64x64xf16>
          %19 = loom.semaphore_take %18 : memref<64x64xf16> -> memref<64x64xf16>
          scf.for %arg6 = %c0 to %c8 step %c1 {
            %23 = arith.muli %arg6, %c64 : index
            %24 = arith.muli %13, %c32768 : index
            %25 = arith.addi %24, %23 : index
            %reinterpret_cast_0 = memref.reinterpret_cast %arg0 to offset: [%25], sizes: [64, 64], strides: [512, 1] : memref<4096x512xf16> to memref<64x64xf16, strided<[512, 1], offset: ?>>
            loom.copy %reinterpret_cast_0, %17 src_mem_space @DRAM dst_mem_space @L1, interconnect : [], broadcast : [1, 1] : memref<64x64xf16, strided<[512, 1], offset: ?>>, memref<64x64xf16>
            %26 = arith.muli %arg5, %c64 : index
            %27 = arith.muli %arg6, %c262144 : index
            %28 = arith.addi %27, %26 : index
            %reinterpret_cast_1 = memref.reinterpret_cast %arg1 to offset: [%28], sizes: [64, 64], strides: [4096, 1] : memref<512x4096xf16> to memref<64x64xf16, strided<[4096, 1], offset: ?>>
            loom.copy %reinterpret_cast_1, %15 src_mem_space @DRAM dst_mem_space @L1, interconnect : [@horizontal_links], broadcast : [1, 8] : memref<64x64xf16, strided<[4096, 1], offset: ?>>, memref<64x64xf16>
            loom.matmul ins(%17, %15 : memref<64x64xf16>, memref<64x64xf16>) outs(%19 : memref<64x64xf16>)
            loom.semaphore_give %15 : memref<64x64xf16>
            loom.semaphore_give %17 : memref<64x64xf16>
          }
          %20 = arith.muli %arg5, %c64 : index
          %21 = arith.muli %13, %c262144 : index
          %22 = arith.addi %21, %20 : index
          %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%22], sizes: [64, 64], strides: [4096, 1] : memref<4096x4096xf16> to memref<64x64xf16, strided<[4096, 1], offset: ?>>
          loom.copy %19, %reinterpret_cast src_mem_space @L1 dst_mem_space @DRAM, interconnect : [], broadcast : [1, 1] : memref<64x64xf16>, memref<64x64xf16, strided<[4096, 1], offset: ?>>
          loom.semaphore_give %19 : memref<64x64xf16>
        }
        scf.reduce 
      } {loom.iter_types = [#loom.iter_type<spatial>, #loom.iter_type<spatial>], loom.mapped_to_dims = [@x, @y]}
      return
    }
  }
  module attributes {loom.pass_name = "Materialize"} {
    func.func @matmul__d0i0_d1i0__f10__d_v__BK64__BM64__BN64(%arg0: memref<4096x512xf16>, %arg1: memref<512x4096xf16>, %arg2: memref<4096x4096xf16>) {
      %c262144 = arith.constant 262144 : index
      %c32768 = arith.constant 32768 : index
      %c64 = arith.constant 64 : index
      %c0 = arith.constant 0 : index
      %c8 = arith.constant 8 : index
      %c1 = arith.constant 1 : index
      scf.parallel (%arg3, %arg4) = (%c0, %c0) to (%c8, %c8) step (%c1, %c1) {
        scf.for %arg5 = %c0 to %c64 step %c1 {
          %12 = arith.muli %arg3, %c8 overflow<nsw> : index
          %13 = arith.addi %12, %arg4 : index
          %14 = loom.alloc [64, 64] on @L1 : memref<64x64xf16>
          %15 = loom.semaphore_take %14 : memref<64x64xf16> -> memref<64x64xf16>
          %16 = loom.alloc [64, 64] on @L1 : memref<64x64xf16>
          %17 = loom.semaphore_take %16 : memref<64x64xf16> -> memref<64x64xf16>
          %18 = loom.alloc [64, 64] on @L1 : memref<64x64xf16>
          %19 = loom.semaphore_take %18 : memref<64x64xf16> -> memref<64x64xf16>
          scf.for %arg6 = %c0 to %c8 step %c1 {
            %23 = arith.muli %arg6, %c64 : index
            %24 = arith.muli %13, %c32768 : index
            %25 = arith.addi %24, %23 : index
            %reinterpret_cast_0 = memref.reinterpret_cast %arg0 to offset: [%25], sizes: [64, 64], strides: [512, 1] : memref<4096x512xf16> to memref<64x64xf16, strided<[512, 1], offset: ?>>
            loom.copy %reinterpret_cast_0, %17 src_mem_space @DRAM dst_mem_space @L1, interconnect : [], broadcast : [1, 1] : memref<64x64xf16, strided<[512, 1], offset: ?>>, memref<64x64xf16>
            %26 = arith.muli %arg5, %c64 : index
            %27 = arith.muli %arg6, %c262144 : index
            %28 = arith.addi %27, %26 : index
            %reinterpret_cast_1 = memref.reinterpret_cast %arg1 to offset: [%28], sizes: [64, 64], strides: [4096, 1] : memref<512x4096xf16> to memref<64x64xf16, strided<[4096, 1], offset: ?>>
            loom.copy %reinterpret_cast_1, %15 src_mem_space @DRAM dst_mem_space @L1, interconnect : [@vertical_links], broadcast : [8, 1] : memref<64x64xf16, strided<[4096, 1], offset: ?>>, memref<64x64xf16>
            loom.matmul ins(%17, %15 : memref<64x64xf16>, memref<64x64xf16>) outs(%19 : memref<64x64xf16>)
            loom.semaphore_give %15 : memref<64x64xf16>
            loom.semaphore_give %17 : memref<64x64xf16>
          }
          %20 = arith.muli %arg5, %c64 : index
          %21 = arith.muli %13, %c262144 : index
          %22 = arith.addi %21, %20 : index
          %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%22], sizes: [64, 64], strides: [4096, 1] : memref<4096x4096xf16> to memref<64x64xf16, strided<[4096, 1], offset: ?>>
          loom.copy %19, %reinterpret_cast src_mem_space @L1 dst_mem_space @DRAM, interconnect : [], broadcast : [1, 1] : memref<64x64xf16>, memref<64x64xf16, strided<[4096, 1], offset: ?>>
          loom.semaphore_give %19 : memref<64x64xf16>
        }
        scf.reduce 
      } {loom.iter_types = [#loom.iter_type<spatial>, #loom.iter_type<spatial>], loom.mapped_to_dims = [@x, @y]}
      return
    }
  }
  module attributes {loom.pass_name = "Materialize"} {
    func.func @matmul__d1i0_d0i0__f01__d_d__BK64__BM64__BN64(%arg0: memref<4096x512xf16>, %arg1: memref<512x4096xf16>, %arg2: memref<4096x4096xf16>) {
      %c262144 = arith.constant 262144 : index
      %c32768 = arith.constant 32768 : index
      %c64 = arith.constant 64 : index
      %c0 = arith.constant 0 : index
      %c8 = arith.constant 8 : index
      %c1 = arith.constant 1 : index
      scf.parallel (%arg3, %arg4) = (%c0, %c0) to (%c8, %c8) step (%c1, %c1) {
        scf.for %arg5 = %c0 to %c64 step %c1 {
          %12 = arith.muli %arg3, %c8 overflow<nsw> : index
          %13 = arith.addi %12, %arg4 : index
          %14 = loom.alloc [64, 64] on @L1 : memref<64x64xf16>
          %15 = loom.semaphore_take %14 : memref<64x64xf16> -> memref<64x64xf16>
          %16 = loom.alloc [64, 64] on @L1 : memref<64x64xf16>
          %17 = loom.semaphore_take %16 : memref<64x64xf16> -> memref<64x64xf16>
          %18 = loom.alloc [64, 64] on @L1 : memref<64x64xf16>
          %19 = loom.semaphore_take %18 : memref<64x64xf16> -> memref<64x64xf16>
          scf.for %arg6 = %c0 to %c8 step %c1 {
            %23 = arith.muli %arg6, %c64 : index
            %24 = arith.muli %13, %c32768 : index
            %25 = arith.addi %24, %23 : index
            %reinterpret_cast_0 = memref.reinterpret_cast %arg0 to offset: [%25], sizes: [64, 64], strides: [512, 1] : memref<4096x512xf16> to memref<64x64xf16, strided<[512, 1], offset: ?>>
            loom.copy %reinterpret_cast_0, %17 src_mem_space @DRAM dst_mem_space @L1, interconnect : [], broadcast : [1, 1] : memref<64x64xf16, strided<[512, 1], offset: ?>>, memref<64x64xf16>
            %26 = arith.muli %arg5, %c64 : index
            %27 = arith.muli %arg6, %c262144 : index
            %28 = arith.addi %27, %26 : index
            %reinterpret_cast_1 = memref.reinterpret_cast %arg1 to offset: [%28], sizes: [64, 64], strides: [4096, 1] : memref<512x4096xf16> to memref<64x64xf16, strided<[4096, 1], offset: ?>>
            loom.copy %reinterpret_cast_1, %15 src_mem_space @DRAM dst_mem_space @L1, interconnect : [], broadcast : [1, 1] : memref<64x64xf16, strided<[4096, 1], offset: ?>>, memref<64x64xf16>
            loom.matmul ins(%17, %15 : memref<64x64xf16>, memref<64x64xf16>) outs(%19 : memref<64x64xf16>)
            loom.semaphore_give %15 : memref<64x64xf16>
            loom.semaphore_give %17 : memref<64x64xf16>
          }
          %20 = arith.muli %arg5, %c64 : index
          %21 = arith.muli %13, %c262144 : index
          %22 = arith.addi %21, %20 : index
          %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%22], sizes: [64, 64], strides: [4096, 1] : memref<4096x4096xf16> to memref<64x64xf16, strided<[4096, 1], offset: ?>>
          loom.copy %19, %reinterpret_cast src_mem_space @L1 dst_mem_space @DRAM, interconnect : [], broadcast : [1, 1] : memref<64x64xf16>, memref<64x64xf16, strided<[4096, 1], offset: ?>>
          loom.semaphore_give %19 : memref<64x64xf16>
        }
        scf.reduce 
      } {loom.iter_types = [#loom.iter_type<spatial>, #loom.iter_type<spatial>], loom.mapped_to_dims = [@y, @x]}
      return
    }
  }
  module attributes {loom.pass_name = "Materialize"} {
    func.func @matmul__d1i0_d0i0__f01__d_a__BK64__BM64__BN64(%arg0: memref<4096x512xf16>, %arg1: memref<512x4096xf16>, %arg2: memref<4096x4096xf16>) {
      %c262144 = arith.constant 262144 : index
      %c32768 = arith.constant 32768 : index
      %c64 = arith.constant 64 : index
      %c0 = arith.constant 0 : index
      %c8 = arith.constant 8 : index
      %c1 = arith.constant 1 : index
      scf.parallel (%arg3, %arg4) = (%c0, %c0) to (%c8, %c8) step (%c1, %c1) {
        scf.for %arg5 = %c0 to %c64 step %c1 {
          %12 = arith.muli %arg3, %c8 overflow<nsw> : index
          %13 = arith.addi %12, %arg4 : index
          %14 = loom.alloc [64, 64] on @L1 : memref<64x64xf16>
          %15 = loom.semaphore_take %14 : memref<64x64xf16> -> memref<64x64xf16>
          %16 = loom.alloc [64, 64] on @L1 : memref<64x64xf16>
          %17 = loom.semaphore_take %16 : memref<64x64xf16> -> memref<64x64xf16>
          %18 = loom.alloc [64, 64] on @L1 : memref<64x64xf16>
          %19 = loom.semaphore_take %18 : memref<64x64xf16> -> memref<64x64xf16>
          scf.for %arg6 = %c0 to %c8 step %c1 {
            %23 = arith.muli %arg6, %c64 : index
            %24 = arith.muli %13, %c32768 : index
            %25 = arith.addi %24, %23 : index
            %reinterpret_cast_0 = memref.reinterpret_cast %arg0 to offset: [%25], sizes: [64, 64], strides: [512, 1] : memref<4096x512xf16> to memref<64x64xf16, strided<[512, 1], offset: ?>>
            loom.copy %reinterpret_cast_0, %17 src_mem_space @DRAM dst_mem_space @L1, interconnect : [], broadcast : [1, 1] : memref<64x64xf16, strided<[512, 1], offset: ?>>, memref<64x64xf16>
            %26 = arith.muli %arg5, %c64 : index
            %27 = arith.muli %arg6, %c262144 : index
            %28 = arith.addi %27, %26 : index
            %reinterpret_cast_1 = memref.reinterpret_cast %arg1 to offset: [%28], sizes: [64, 64], strides: [4096, 1] : memref<512x4096xf16> to memref<64x64xf16, strided<[4096, 1], offset: ?>>
            loom.copy %reinterpret_cast_1, %15 src_mem_space @DRAM dst_mem_space @L1, interconnect : [@horizontal_links, @vertical_links], broadcast : [8, 8] : memref<64x64xf16, strided<[4096, 1], offset: ?>>, memref<64x64xf16>
            loom.matmul ins(%17, %15 : memref<64x64xf16>, memref<64x64xf16>) outs(%19 : memref<64x64xf16>)
            loom.semaphore_give %15 : memref<64x64xf16>
            loom.semaphore_give %17 : memref<64x64xf16>
          }
          %20 = arith.muli %arg5, %c64 : index
          %21 = arith.muli %13, %c262144 : index
          %22 = arith.addi %21, %20 : index
          %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%22], sizes: [64, 64], strides: [4096, 1] : memref<4096x4096xf16> to memref<64x64xf16, strided<[4096, 1], offset: ?>>
          loom.copy %19, %reinterpret_cast src_mem_space @L1 dst_mem_space @DRAM, interconnect : [], broadcast : [1, 1] : memref<64x64xf16>, memref<64x64xf16, strided<[4096, 1], offset: ?>>
          loom.semaphore_give %19 : memref<64x64xf16>
        }
        scf.reduce 
      } {loom.iter_types = [#loom.iter_type<spatial>, #loom.iter_type<spatial>], loom.mapped_to_dims = [@y, @x]}
      return
    }
  }
  module attributes {loom.pass_name = "Materialize"} {
    func.func @matmul__d1i0_d0i0__f01__d_h__BK64__BM64__BN64(%arg0: memref<4096x512xf16>, %arg1: memref<512x4096xf16>, %arg2: memref<4096x4096xf16>) {
      %c262144 = arith.constant 262144 : index
      %c32768 = arith.constant 32768 : index
      %c64 = arith.constant 64 : index
      %c0 = arith.constant 0 : index
      %c8 = arith.constant 8 : index
      %c1 = arith.constant 1 : index
      scf.parallel (%arg3, %arg4) = (%c0, %c0) to (%c8, %c8) step (%c1, %c1) {
        scf.for %arg5 = %c0 to %c64 step %c1 {
          %12 = arith.muli %arg3, %c8 overflow<nsw> : index
          %13 = arith.addi %12, %arg4 : index
          %14 = loom.alloc [64, 64] on @L1 : memref<64x64xf16>
          %15 = loom.semaphore_take %14 : memref<64x64xf16> -> memref<64x64xf16>
          %16 = loom.alloc [64, 64] on @L1 : memref<64x64xf16>
          %17 = loom.semaphore_take %16 : memref<64x64xf16> -> memref<64x64xf16>
          %18 = loom.alloc [64, 64] on @L1 : memref<64x64xf16>
          %19 = loom.semaphore_take %18 : memref<64x64xf16> -> memref<64x64xf16>
          scf.for %arg6 = %c0 to %c8 step %c1 {
            %23 = arith.muli %arg6, %c64 : index
            %24 = arith.muli %13, %c32768 : index
            %25 = arith.addi %24, %23 : index
            %reinterpret_cast_0 = memref.reinterpret_cast %arg0 to offset: [%25], sizes: [64, 64], strides: [512, 1] : memref<4096x512xf16> to memref<64x64xf16, strided<[512, 1], offset: ?>>
            loom.copy %reinterpret_cast_0, %17 src_mem_space @DRAM dst_mem_space @L1, interconnect : [], broadcast : [1, 1] : memref<64x64xf16, strided<[512, 1], offset: ?>>, memref<64x64xf16>
            %26 = arith.muli %arg5, %c64 : index
            %27 = arith.muli %arg6, %c262144 : index
            %28 = arith.addi %27, %26 : index
            %reinterpret_cast_1 = memref.reinterpret_cast %arg1 to offset: [%28], sizes: [64, 64], strides: [4096, 1] : memref<512x4096xf16> to memref<64x64xf16, strided<[4096, 1], offset: ?>>
            loom.copy %reinterpret_cast_1, %15 src_mem_space @DRAM dst_mem_space @L1, interconnect : [@horizontal_links], broadcast : [1, 8] : memref<64x64xf16, strided<[4096, 1], offset: ?>>, memref<64x64xf16>
            loom.matmul ins(%17, %15 : memref<64x64xf16>, memref<64x64xf16>) outs(%19 : memref<64x64xf16>)
            loom.semaphore_give %15 : memref<64x64xf16>
            loom.semaphore_give %17 : memref<64x64xf16>
          }
          %20 = arith.muli %arg5, %c64 : index
          %21 = arith.muli %13, %c262144 : index
          %22 = arith.addi %21, %20 : index
          %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%22], sizes: [64, 64], strides: [4096, 1] : memref<4096x4096xf16> to memref<64x64xf16, strided<[4096, 1], offset: ?>>
          loom.copy %19, %reinterpret_cast src_mem_space @L1 dst_mem_space @DRAM, interconnect : [], broadcast : [1, 1] : memref<64x64xf16>, memref<64x64xf16, strided<[4096, 1], offset: ?>>
          loom.semaphore_give %19 : memref<64x64xf16>
        }
        scf.reduce 
      } {loom.iter_types = [#loom.iter_type<spatial>, #loom.iter_type<spatial>], loom.mapped_to_dims = [@y, @x]}
      return
    }
  }
  module attributes {loom.pass_name = "Materialize"} {
    func.func @matmul__d1i0_d0i0__f01__d_v__BK64__BM64__BN64(%arg0: memref<4096x512xf16>, %arg1: memref<512x4096xf16>, %arg2: memref<4096x4096xf16>) {
      %c262144 = arith.constant 262144 : index
      %c32768 = arith.constant 32768 : index
      %c64 = arith.constant 64 : index
      %c0 = arith.constant 0 : index
      %c8 = arith.constant 8 : index
      %c1 = arith.constant 1 : index
      scf.parallel (%arg3, %arg4) = (%c0, %c0) to (%c8, %c8) step (%c1, %c1) {
        scf.for %arg5 = %c0 to %c64 step %c1 {
          %12 = arith.muli %arg3, %c8 overflow<nsw> : index
          %13 = arith.addi %12, %arg4 : index
          %14 = loom.alloc [64, 64] on @L1 : memref<64x64xf16>
          %15 = loom.semaphore_take %14 : memref<64x64xf16> -> memref<64x64xf16>
          %16 = loom.alloc [64, 64] on @L1 : memref<64x64xf16>
          %17 = loom.semaphore_take %16 : memref<64x64xf16> -> memref<64x64xf16>
          %18 = loom.alloc [64, 64] on @L1 : memref<64x64xf16>
          %19 = loom.semaphore_take %18 : memref<64x64xf16> -> memref<64x64xf16>
          scf.for %arg6 = %c0 to %c8 step %c1 {
            %23 = arith.muli %arg6, %c64 : index
            %24 = arith.muli %13, %c32768 : index
            %25 = arith.addi %24, %23 : index
            %reinterpret_cast_0 = memref.reinterpret_cast %arg0 to offset: [%25], sizes: [64, 64], strides: [512, 1] : memref<4096x512xf16> to memref<64x64xf16, strided<[512, 1], offset: ?>>
            loom.copy %reinterpret_cast_0, %17 src_mem_space @DRAM dst_mem_space @L1, interconnect : [], broadcast : [1, 1] : memref<64x64xf16, strided<[512, 1], offset: ?>>, memref<64x64xf16>
            %26 = arith.muli %arg5, %c64 : index
            %27 = arith.muli %arg6, %c262144 : index
            %28 = arith.addi %27, %26 : index
            %reinterpret_cast_1 = memref.reinterpret_cast %arg1 to offset: [%28], sizes: [64, 64], strides: [4096, 1] : memref<512x4096xf16> to memref<64x64xf16, strided<[4096, 1], offset: ?>>
            loom.copy %reinterpret_cast_1, %15 src_mem_space @DRAM dst_mem_space @L1, interconnect : [@vertical_links], broadcast : [8, 1] : memref<64x64xf16, strided<[4096, 1], offset: ?>>, memref<64x64xf16>
            loom.matmul ins(%17, %15 : memref<64x64xf16>, memref<64x64xf16>) outs(%19 : memref<64x64xf16>)
            loom.semaphore_give %15 : memref<64x64xf16>
            loom.semaphore_give %17 : memref<64x64xf16>
          }
          %20 = arith.muli %arg5, %c64 : index
          %21 = arith.muli %13, %c262144 : index
          %22 = arith.addi %21, %20 : index
          %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%22], sizes: [64, 64], strides: [4096, 1] : memref<4096x4096xf16> to memref<64x64xf16, strided<[4096, 1], offset: ?>>
          loom.copy %19, %reinterpret_cast src_mem_space @L1 dst_mem_space @DRAM, interconnect : [], broadcast : [1, 1] : memref<64x64xf16>, memref<64x64xf16, strided<[4096, 1], offset: ?>>
          loom.semaphore_give %19 : memref<64x64xf16>
        }
        scf.reduce 
      } {loom.iter_types = [#loom.iter_type<spatial>, #loom.iter_type<spatial>], loom.mapped_to_dims = [@y, @x]}
      return
    }
  }
  module attributes {loom.pass_name = "Materialize"} {
    func.func @matmul__d1i0_d0i0__f10__d_d__BK64__BM64__BN64(%arg0: memref<4096x512xf16>, %arg1: memref<512x4096xf16>, %arg2: memref<4096x4096xf16>) {
      %c262144 = arith.constant 262144 : index
      %c32768 = arith.constant 32768 : index
      %c64 = arith.constant 64 : index
      %c0 = arith.constant 0 : index
      %c8 = arith.constant 8 : index
      %c1 = arith.constant 1 : index
      scf.parallel (%arg3, %arg4) = (%c0, %c0) to (%c8, %c8) step (%c1, %c1) {
        scf.for %arg5 = %c0 to %c64 step %c1 {
          %12 = arith.muli %arg3, %c8 overflow<nsw> : index
          %13 = arith.addi %12, %arg4 : index
          %14 = loom.alloc [64, 64] on @L1 : memref<64x64xf16>
          %15 = loom.semaphore_take %14 : memref<64x64xf16> -> memref<64x64xf16>
          %16 = loom.alloc [64, 64] on @L1 : memref<64x64xf16>
          %17 = loom.semaphore_take %16 : memref<64x64xf16> -> memref<64x64xf16>
          %18 = loom.alloc [64, 64] on @L1 : memref<64x64xf16>
          %19 = loom.semaphore_take %18 : memref<64x64xf16> -> memref<64x64xf16>
          scf.for %arg6 = %c0 to %c8 step %c1 {
            %23 = arith.muli %arg6, %c64 : index
            %24 = arith.muli %13, %c32768 : index
            %25 = arith.addi %24, %23 : index
            %reinterpret_cast_0 = memref.reinterpret_cast %arg0 to offset: [%25], sizes: [64, 64], strides: [512, 1] : memref<4096x512xf16> to memref<64x64xf16, strided<[512, 1], offset: ?>>
            loom.copy %reinterpret_cast_0, %17 src_mem_space @DRAM dst_mem_space @L1, interconnect : [], broadcast : [1, 1] : memref<64x64xf16, strided<[512, 1], offset: ?>>, memref<64x64xf16>
            %26 = arith.muli %arg5, %c64 : index
            %27 = arith.muli %arg6, %c262144 : index
            %28 = arith.addi %27, %26 : index
            %reinterpret_cast_1 = memref.reinterpret_cast %arg1 to offset: [%28], sizes: [64, 64], strides: [4096, 1] : memref<512x4096xf16> to memref<64x64xf16, strided<[4096, 1], offset: ?>>
            loom.copy %reinterpret_cast_1, %15 src_mem_space @DRAM dst_mem_space @L1, interconnect : [], broadcast : [1, 1] : memref<64x64xf16, strided<[4096, 1], offset: ?>>, memref<64x64xf16>
            loom.matmul ins(%17, %15 : memref<64x64xf16>, memref<64x64xf16>) outs(%19 : memref<64x64xf16>)
            loom.semaphore_give %15 : memref<64x64xf16>
            loom.semaphore_give %17 : memref<64x64xf16>
          }
          %20 = arith.muli %arg5, %c64 : index
          %21 = arith.muli %13, %c262144 : index
          %22 = arith.addi %21, %20 : index
          %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%22], sizes: [64, 64], strides: [4096, 1] : memref<4096x4096xf16> to memref<64x64xf16, strided<[4096, 1], offset: ?>>
          loom.copy %19, %reinterpret_cast src_mem_space @L1 dst_mem_space @DRAM, interconnect : [], broadcast : [1, 1] : memref<64x64xf16>, memref<64x64xf16, strided<[4096, 1], offset: ?>>
          loom.semaphore_give %19 : memref<64x64xf16>
        }
        scf.reduce 
      } {loom.iter_types = [#loom.iter_type<spatial>, #loom.iter_type<spatial>], loom.mapped_to_dims = [@y, @x]}
      return
    }
  }
  module attributes {loom.pass_name = "Materialize"} {
    func.func @matmul__d1i0_d0i0__f10__d_a__BK64__BM64__BN64(%arg0: memref<4096x512xf16>, %arg1: memref<512x4096xf16>, %arg2: memref<4096x4096xf16>) {
      %c262144 = arith.constant 262144 : index
      %c32768 = arith.constant 32768 : index
      %c64 = arith.constant 64 : index
      %c0 = arith.constant 0 : index
      %c8 = arith.constant 8 : index
      %c1 = arith.constant 1 : index
      scf.parallel (%arg3, %arg4) = (%c0, %c0) to (%c8, %c8) step (%c1, %c1) {
        scf.for %arg5 = %c0 to %c64 step %c1 {
          %12 = arith.muli %arg3, %c8 overflow<nsw> : index
          %13 = arith.addi %12, %arg4 : index
          %14 = loom.alloc [64, 64] on @L1 : memref<64x64xf16>
          %15 = loom.semaphore_take %14 : memref<64x64xf16> -> memref<64x64xf16>
          %16 = loom.alloc [64, 64] on @L1 : memref<64x64xf16>
          %17 = loom.semaphore_take %16 : memref<64x64xf16> -> memref<64x64xf16>
          %18 = loom.alloc [64, 64] on @L1 : memref<64x64xf16>
          %19 = loom.semaphore_take %18 : memref<64x64xf16> -> memref<64x64xf16>
          scf.for %arg6 = %c0 to %c8 step %c1 {
            %23 = arith.muli %arg6, %c64 : index
            %24 = arith.muli %13, %c32768 : index
            %25 = arith.addi %24, %23 : index
            %reinterpret_cast_0 = memref.reinterpret_cast %arg0 to offset: [%25], sizes: [64, 64], strides: [512, 1] : memref<4096x512xf16> to memref<64x64xf16, strided<[512, 1], offset: ?>>
            loom.copy %reinterpret_cast_0, %17 src_mem_space @DRAM dst_mem_space @L1, interconnect : [], broadcast : [1, 1] : memref<64x64xf16, strided<[512, 1], offset: ?>>, memref<64x64xf16>
            %26 = arith.muli %arg5, %c64 : index
            %27 = arith.muli %arg6, %c262144 : index
            %28 = arith.addi %27, %26 : index
            %reinterpret_cast_1 = memref.reinterpret_cast %arg1 to offset: [%28], sizes: [64, 64], strides: [4096, 1] : memref<512x4096xf16> to memref<64x64xf16, strided<[4096, 1], offset: ?>>
            loom.copy %reinterpret_cast_1, %15 src_mem_space @DRAM dst_mem_space @L1, interconnect : [@horizontal_links, @vertical_links], broadcast : [8, 8] : memref<64x64xf16, strided<[4096, 1], offset: ?>>, memref<64x64xf16>
            loom.matmul ins(%17, %15 : memref<64x64xf16>, memref<64x64xf16>) outs(%19 : memref<64x64xf16>)
            loom.semaphore_give %15 : memref<64x64xf16>
            loom.semaphore_give %17 : memref<64x64xf16>
          }
          %20 = arith.muli %arg5, %c64 : index
          %21 = arith.muli %13, %c262144 : index
          %22 = arith.addi %21, %20 : index
          %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%22], sizes: [64, 64], strides: [4096, 1] : memref<4096x4096xf16> to memref<64x64xf16, strided<[4096, 1], offset: ?>>
          loom.copy %19, %reinterpret_cast src_mem_space @L1 dst_mem_space @DRAM, interconnect : [], broadcast : [1, 1] : memref<64x64xf16>, memref<64x64xf16, strided<[4096, 1], offset: ?>>
          loom.semaphore_give %19 : memref<64x64xf16>
        }
        scf.reduce 
      } {loom.iter_types = [#loom.iter_type<spatial>, #loom.iter_type<spatial>], loom.mapped_to_dims = [@y, @x]}
      return
    }
  }
  module attributes {loom.pass_name = "Materialize"} {
    func.func @matmul__d1i0_d0i0__f10__d_h__BK64__BM64__BN64(%arg0: memref<4096x512xf16>, %arg1: memref<512x4096xf16>, %arg2: memref<4096x4096xf16>) {
      %c262144 = arith.constant 262144 : index
      %c32768 = arith.constant 32768 : index
      %c64 = arith.constant 64 : index
      %c0 = arith.constant 0 : index
      %c8 = arith.constant 8 : index
      %c1 = arith.constant 1 : index
      scf.parallel (%arg3, %arg4) = (%c0, %c0) to (%c8, %c8) step (%c1, %c1) {
        scf.for %arg5 = %c0 to %c64 step %c1 {
          %12 = arith.muli %arg3, %c8 overflow<nsw> : index
          %13 = arith.addi %12, %arg4 : index
          %14 = loom.alloc [64, 64] on @L1 : memref<64x64xf16>
          %15 = loom.semaphore_take %14 : memref<64x64xf16> -> memref<64x64xf16>
          %16 = loom.alloc [64, 64] on @L1 : memref<64x64xf16>
          %17 = loom.semaphore_take %16 : memref<64x64xf16> -> memref<64x64xf16>
          %18 = loom.alloc [64, 64] on @L1 : memref<64x64xf16>
          %19 = loom.semaphore_take %18 : memref<64x64xf16> -> memref<64x64xf16>
          scf.for %arg6 = %c0 to %c8 step %c1 {
            %23 = arith.muli %arg6, %c64 : index
            %24 = arith.muli %13, %c32768 : index
            %25 = arith.addi %24, %23 : index
            %reinterpret_cast_0 = memref.reinterpret_cast %arg0 to offset: [%25], sizes: [64, 64], strides: [512, 1] : memref<4096x512xf16> to memref<64x64xf16, strided<[512, 1], offset: ?>>
            loom.copy %reinterpret_cast_0, %17 src_mem_space @DRAM dst_mem_space @L1, interconnect : [], broadcast : [1, 1] : memref<64x64xf16, strided<[512, 1], offset: ?>>, memref<64x64xf16>
            %26 = arith.muli %arg5, %c64 : index
            %27 = arith.muli %arg6, %c262144 : index
            %28 = arith.addi %27, %26 : index
            %reinterpret_cast_1 = memref.reinterpret_cast %arg1 to offset: [%28], sizes: [64, 64], strides: [4096, 1] : memref<512x4096xf16> to memref<64x64xf16, strided<[4096, 1], offset: ?>>
            loom.copy %reinterpret_cast_1, %15 src_mem_space @DRAM dst_mem_space @L1, interconnect : [@horizontal_links], broadcast : [1, 8] : memref<64x64xf16, strided<[4096, 1], offset: ?>>, memref<64x64xf16>
            loom.matmul ins(%17, %15 : memref<64x64xf16>, memref<64x64xf16>) outs(%19 : memref<64x64xf16>)
            loom.semaphore_give %15 : memref<64x64xf16>
            loom.semaphore_give %17 : memref<64x64xf16>
          }
          %20 = arith.muli %arg5, %c64 : index
          %21 = arith.muli %13, %c262144 : index
          %22 = arith.addi %21, %20 : index
          %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%22], sizes: [64, 64], strides: [4096, 1] : memref<4096x4096xf16> to memref<64x64xf16, strided<[4096, 1], offset: ?>>
          loom.copy %19, %reinterpret_cast src_mem_space @L1 dst_mem_space @DRAM, interconnect : [], broadcast : [1, 1] : memref<64x64xf16>, memref<64x64xf16, strided<[4096, 1], offset: ?>>
          loom.semaphore_give %19 : memref<64x64xf16>
        }
        scf.reduce 
      } {loom.iter_types = [#loom.iter_type<spatial>, #loom.iter_type<spatial>], loom.mapped_to_dims = [@y, @x]}
      return
    }
  }
  module attributes {loom.pass_name = "Materialize"} {
    func.func @matmul__d1i0_d0i0__f10__d_v__BK64__BM64__BN64(%arg0: memref<4096x512xf16>, %arg1: memref<512x4096xf16>, %arg2: memref<4096x4096xf16>) {
      %c262144 = arith.constant 262144 : index
      %c32768 = arith.constant 32768 : index
      %c64 = arith.constant 64 : index
      %c0 = arith.constant 0 : index
      %c8 = arith.constant 8 : index
      %c1 = arith.constant 1 : index
      scf.parallel (%arg3, %arg4) = (%c0, %c0) to (%c8, %c8) step (%c1, %c1) {
        scf.for %arg5 = %c0 to %c64 step %c1 {
          %12 = arith.muli %arg3, %c8 overflow<nsw> : index
          %13 = arith.addi %12, %arg4 : index
          %14 = loom.alloc [64, 64] on @L1 : memref<64x64xf16>
          %15 = loom.semaphore_take %14 : memref<64x64xf16> -> memref<64x64xf16>
          %16 = loom.alloc [64, 64] on @L1 : memref<64x64xf16>
          %17 = loom.semaphore_take %16 : memref<64x64xf16> -> memref<64x64xf16>
          %18 = loom.alloc [64, 64] on @L1 : memref<64x64xf16>
          %19 = loom.semaphore_take %18 : memref<64x64xf16> -> memref<64x64xf16>
          scf.for %arg6 = %c0 to %c8 step %c1 {
            %23 = arith.muli %arg6, %c64 : index
            %24 = arith.muli %13, %c32768 : index
            %25 = arith.addi %24, %23 : index
            %reinterpret_cast_0 = memref.reinterpret_cast %arg0 to offset: [%25], sizes: [64, 64], strides: [512, 1] : memref<4096x512xf16> to memref<64x64xf16, strided<[512, 1], offset: ?>>
            loom.copy %reinterpret_cast_0, %17 src_mem_space @DRAM dst_mem_space @L1, interconnect : [], broadcast : [1, 1] : memref<64x64xf16, strided<[512, 1], offset: ?>>, memref<64x64xf16>
            %26 = arith.muli %arg5, %c64 : index
            %27 = arith.muli %arg6, %c262144 : index
            %28 = arith.addi %27, %26 : index
            %reinterpret_cast_1 = memref.reinterpret_cast %arg1 to offset: [%28], sizes: [64, 64], strides: [4096, 1] : memref<512x4096xf16> to memref<64x64xf16, strided<[4096, 1], offset: ?>>
            loom.copy %reinterpret_cast_1, %15 src_mem_space @DRAM dst_mem_space @L1, interconnect : [@vertical_links], broadcast : [8, 1] : memref<64x64xf16, strided<[4096, 1], offset: ?>>, memref<64x64xf16>
            loom.matmul ins(%17, %15 : memref<64x64xf16>, memref<64x64xf16>) outs(%19 : memref<64x64xf16>)
            loom.semaphore_give %15 : memref<64x64xf16>
            loom.semaphore_give %17 : memref<64x64xf16>
          }
          %20 = arith.muli %arg5, %c64 : index
          %21 = arith.muli %13, %c262144 : index
          %22 = arith.addi %21, %20 : index
          %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%22], sizes: [64, 64], strides: [4096, 1] : memref<4096x4096xf16> to memref<64x64xf16, strided<[4096, 1], offset: ?>>
          loom.copy %19, %reinterpret_cast src_mem_space @L1 dst_mem_space @DRAM, interconnect : [], broadcast : [1, 1] : memref<64x64xf16>, memref<64x64xf16, strided<[4096, 1], offset: ?>>
          loom.semaphore_give %19 : memref<64x64xf16>
        }
        scf.reduce 
      } {loom.iter_types = [#loom.iter_type<spatial>, #loom.iter_type<spatial>], loom.mapped_to_dims = [@y, @x]}
      return
    }
  }
  module attributes {loom.pass_name = "Materialize"} {
    func.func @matmul__d0i0_d1i1__f01__d_d__BK64__BM64__BN64(%arg0: memref<4096x512xf16>, %arg1: memref<512x4096xf16>, %arg2: memref<4096x4096xf16>) {
      %c262144 = arith.constant 262144 : index
      %c32768 = arith.constant 32768 : index
      %c64 = arith.constant 64 : index
      %c0 = arith.constant 0 : index
      %c8 = arith.constant 8 : index
      %c1 = arith.constant 1 : index
      scf.parallel (%arg3, %arg4) = (%c0, %c0) to (%c8, %c8) step (%c1, %c1) {
        scf.for %arg5 = %c0 to %c8 step %c1 {
          scf.for %arg6 = %c0 to %c8 step %c1 {
            %12 = arith.muli %arg5, %c8 overflow<nsw> : index
            %13 = arith.addi %arg3, %12 : index
            %14 = arith.muli %arg6, %c8 overflow<nsw> : index
            %15 = arith.addi %arg4, %14 : index
            %16 = loom.alloc [64, 64] on @L1 : memref<64x64xf16>
            %17 = loom.semaphore_take %16 : memref<64x64xf16> -> memref<64x64xf16>
            %18 = loom.alloc [64, 64] on @L1 : memref<64x64xf16>
            %19 = loom.semaphore_take %18 : memref<64x64xf16> -> memref<64x64xf16>
            %20 = loom.alloc [64, 64] on @L1 : memref<64x64xf16>
            %21 = loom.semaphore_take %20 : memref<64x64xf16> -> memref<64x64xf16>
            scf.for %arg7 = %c0 to %c8 step %c1 {
              %25 = arith.muli %arg7, %c64 : index
              %26 = arith.muli %13, %c32768 : index
              %27 = arith.addi %26, %25 : index
              %reinterpret_cast_0 = memref.reinterpret_cast %arg0 to offset: [%27], sizes: [64, 64], strides: [512, 1] : memref<4096x512xf16> to memref<64x64xf16, strided<[512, 1], offset: ?>>
              loom.copy %reinterpret_cast_0, %19 src_mem_space @DRAM dst_mem_space @L1, interconnect : [], broadcast : [1, 1] : memref<64x64xf16, strided<[512, 1], offset: ?>>, memref<64x64xf16>
              %28 = arith.muli %15, %c64 : index
              %29 = arith.muli %arg7, %c262144 : index
              %30 = arith.addi %29, %28 : index
              %reinterpret_cast_1 = memref.reinterpret_cast %arg1 to offset: [%30], sizes: [64, 64], strides: [4096, 1] : memref<512x4096xf16> to memref<64x64xf16, strided<[4096, 1], offset: ?>>
              loom.copy %reinterpret_cast_1, %17 src_mem_space @DRAM dst_mem_space @L1, interconnect : [], broadcast : [1, 1] : memref<64x64xf16, strided<[4096, 1], offset: ?>>, memref<64x64xf16>
              loom.matmul ins(%19, %17 : memref<64x64xf16>, memref<64x64xf16>) outs(%21 : memref<64x64xf16>)
              loom.semaphore_give %17 : memref<64x64xf16>
              loom.semaphore_give %19 : memref<64x64xf16>
            }
            %22 = arith.muli %15, %c64 : index
            %23 = arith.muli %13, %c262144 : index
            %24 = arith.addi %23, %22 : index
            %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%24], sizes: [64, 64], strides: [4096, 1] : memref<4096x4096xf16> to memref<64x64xf16, strided<[4096, 1], offset: ?>>
            loom.copy %21, %reinterpret_cast src_mem_space @L1 dst_mem_space @DRAM, interconnect : [], broadcast : [1, 1] : memref<64x64xf16>, memref<64x64xf16, strided<[4096, 1], offset: ?>>
            loom.semaphore_give %21 : memref<64x64xf16>
          }
        }
        scf.reduce 
      } {loom.iter_types = [#loom.iter_type<spatial>, #loom.iter_type<spatial>], loom.mapped_to_dims = [@x, @y]}
      return
    }
  }
  module attributes {loom.pass_name = "Materialize"} {
    func.func @matmul__d0i0_d1i1__f01__d_h__BK64__BM64__BN64(%arg0: memref<4096x512xf16>, %arg1: memref<512x4096xf16>, %arg2: memref<4096x4096xf16>) {
      %c262144 = arith.constant 262144 : index
      %c32768 = arith.constant 32768 : index
      %c64 = arith.constant 64 : index
      %c0 = arith.constant 0 : index
      %c8 = arith.constant 8 : index
      %c1 = arith.constant 1 : index
      scf.parallel (%arg3, %arg4) = (%c0, %c0) to (%c8, %c8) step (%c1, %c1) {
        scf.for %arg5 = %c0 to %c8 step %c1 {
          scf.for %arg6 = %c0 to %c8 step %c1 {
            %12 = arith.muli %arg5, %c8 overflow<nsw> : index
            %13 = arith.addi %arg3, %12 : index
            %14 = arith.muli %arg6, %c8 overflow<nsw> : index
            %15 = arith.addi %arg4, %14 : index
            %16 = loom.alloc [64, 64] on @L1 : memref<64x64xf16>
            %17 = loom.semaphore_take %16 : memref<64x64xf16> -> memref<64x64xf16>
            %18 = loom.alloc [64, 64] on @L1 : memref<64x64xf16>
            %19 = loom.semaphore_take %18 : memref<64x64xf16> -> memref<64x64xf16>
            %20 = loom.alloc [64, 64] on @L1 : memref<64x64xf16>
            %21 = loom.semaphore_take %20 : memref<64x64xf16> -> memref<64x64xf16>
            scf.for %arg7 = %c0 to %c8 step %c1 {
              %25 = arith.muli %arg7, %c64 : index
              %26 = arith.muli %13, %c32768 : index
              %27 = arith.addi %26, %25 : index
              %reinterpret_cast_0 = memref.reinterpret_cast %arg0 to offset: [%27], sizes: [64, 64], strides: [512, 1] : memref<4096x512xf16> to memref<64x64xf16, strided<[512, 1], offset: ?>>
              loom.copy %reinterpret_cast_0, %19 src_mem_space @DRAM dst_mem_space @L1, interconnect : [], broadcast : [1, 1] : memref<64x64xf16, strided<[512, 1], offset: ?>>, memref<64x64xf16>
              %28 = arith.muli %15, %c64 : index
              %29 = arith.muli %arg7, %c262144 : index
              %30 = arith.addi %29, %28 : index
              %reinterpret_cast_1 = memref.reinterpret_cast %arg1 to offset: [%30], sizes: [64, 64], strides: [4096, 1] : memref<512x4096xf16> to memref<64x64xf16, strided<[4096, 1], offset: ?>>
              loom.copy %reinterpret_cast_1, %17 src_mem_space @DRAM dst_mem_space @L1, interconnect : [@horizontal_links], broadcast : [1, 8] : memref<64x64xf16, strided<[4096, 1], offset: ?>>, memref<64x64xf16>
              loom.matmul ins(%19, %17 : memref<64x64xf16>, memref<64x64xf16>) outs(%21 : memref<64x64xf16>)
              loom.semaphore_give %17 : memref<64x64xf16>
              loom.semaphore_give %19 : memref<64x64xf16>
            }
            %22 = arith.muli %15, %c64 : index
            %23 = arith.muli %13, %c262144 : index
            %24 = arith.addi %23, %22 : index
            %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%24], sizes: [64, 64], strides: [4096, 1] : memref<4096x4096xf16> to memref<64x64xf16, strided<[4096, 1], offset: ?>>
            loom.copy %21, %reinterpret_cast src_mem_space @L1 dst_mem_space @DRAM, interconnect : [], broadcast : [1, 1] : memref<64x64xf16>, memref<64x64xf16, strided<[4096, 1], offset: ?>>
            loom.semaphore_give %21 : memref<64x64xf16>
          }
        }
        scf.reduce 
      } {loom.iter_types = [#loom.iter_type<spatial>, #loom.iter_type<spatial>], loom.mapped_to_dims = [@x, @y]}
      return
    }
  }
  module attributes {loom.pass_name = "Materialize"} {
    func.func @matmul__d0i0_d1i1__f01__v_d__BK64__BM64__BN64(%arg0: memref<4096x512xf16>, %arg1: memref<512x4096xf16>, %arg2: memref<4096x4096xf16>) {
      %c262144 = arith.constant 262144 : index
      %c32768 = arith.constant 32768 : index
      %c64 = arith.constant 64 : index
      %c0 = arith.constant 0 : index
      %c8 = arith.constant 8 : index
      %c1 = arith.constant 1 : index
      scf.parallel (%arg3, %arg4) = (%c0, %c0) to (%c8, %c8) step (%c1, %c1) {
        scf.for %arg5 = %c0 to %c8 step %c1 {
          scf.for %arg6 = %c0 to %c8 step %c1 {
            %12 = arith.muli %arg5, %c8 overflow<nsw> : index
            %13 = arith.addi %arg3, %12 : index
            %14 = arith.muli %arg6, %c8 overflow<nsw> : index
            %15 = arith.addi %arg4, %14 : index
            %16 = loom.alloc [64, 64] on @L1 : memref<64x64xf16>
            %17 = loom.semaphore_take %16 : memref<64x64xf16> -> memref<64x64xf16>
            %18 = loom.alloc [64, 64] on @L1 : memref<64x64xf16>
            %19 = loom.semaphore_take %18 : memref<64x64xf16> -> memref<64x64xf16>
            %20 = loom.alloc [64, 64] on @L1 : memref<64x64xf16>
            %21 = loom.semaphore_take %20 : memref<64x64xf16> -> memref<64x64xf16>
            scf.for %arg7 = %c0 to %c8 step %c1 {
              %25 = arith.muli %arg7, %c64 : index
              %26 = arith.muli %13, %c32768 : index
              %27 = arith.addi %26, %25 : index
              %reinterpret_cast_0 = memref.reinterpret_cast %arg0 to offset: [%27], sizes: [64, 64], strides: [512, 1] : memref<4096x512xf16> to memref<64x64xf16, strided<[512, 1], offset: ?>>
              loom.copy %reinterpret_cast_0, %19 src_mem_space @DRAM dst_mem_space @L1, interconnect : [@vertical_links], broadcast : [8, 1] : memref<64x64xf16, strided<[512, 1], offset: ?>>, memref<64x64xf16>
              %28 = arith.muli %15, %c64 : index
              %29 = arith.muli %arg7, %c262144 : index
              %30 = arith.addi %29, %28 : index
              %reinterpret_cast_1 = memref.reinterpret_cast %arg1 to offset: [%30], sizes: [64, 64], strides: [4096, 1] : memref<512x4096xf16> to memref<64x64xf16, strided<[4096, 1], offset: ?>>
              loom.copy %reinterpret_cast_1, %17 src_mem_space @DRAM dst_mem_space @L1, interconnect : [], broadcast : [1, 1] : memref<64x64xf16, strided<[4096, 1], offset: ?>>, memref<64x64xf16>
              loom.matmul ins(%19, %17 : memref<64x64xf16>, memref<64x64xf16>) outs(%21 : memref<64x64xf16>)
              loom.semaphore_give %17 : memref<64x64xf16>
              loom.semaphore_give %19 : memref<64x64xf16>
            }
            %22 = arith.muli %15, %c64 : index
            %23 = arith.muli %13, %c262144 : index
            %24 = arith.addi %23, %22 : index
            %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%24], sizes: [64, 64], strides: [4096, 1] : memref<4096x4096xf16> to memref<64x64xf16, strided<[4096, 1], offset: ?>>
            loom.copy %21, %reinterpret_cast src_mem_space @L1 dst_mem_space @DRAM, interconnect : [], broadcast : [1, 1] : memref<64x64xf16>, memref<64x64xf16, strided<[4096, 1], offset: ?>>
            loom.semaphore_give %21 : memref<64x64xf16>
          }
        }
        scf.reduce 
      } {loom.iter_types = [#loom.iter_type<spatial>, #loom.iter_type<spatial>], loom.mapped_to_dims = [@x, @y]}
      return
    }
  }
  module attributes {loom.pass_name = "Materialize"} {
    func.func @matmul__d0i0_d1i1__f01__v_h__BK64__BM64__BN64(%arg0: memref<4096x512xf16>, %arg1: memref<512x4096xf16>, %arg2: memref<4096x4096xf16>) {
      %c262144 = arith.constant 262144 : index
      %c32768 = arith.constant 32768 : index
      %c64 = arith.constant 64 : index
      %c0 = arith.constant 0 : index
      %c8 = arith.constant 8 : index
      %c1 = arith.constant 1 : index
      scf.parallel (%arg3, %arg4) = (%c0, %c0) to (%c8, %c8) step (%c1, %c1) {
        scf.for %arg5 = %c0 to %c8 step %c1 {
          scf.for %arg6 = %c0 to %c8 step %c1 {
            %12 = arith.muli %arg5, %c8 overflow<nsw> : index
            %13 = arith.addi %arg3, %12 : index
            %14 = arith.muli %arg6, %c8 overflow<nsw> : index
            %15 = arith.addi %arg4, %14 : index
            %16 = loom.alloc [64, 64] on @L1 : memref<64x64xf16>
            %17 = loom.semaphore_take %16 : memref<64x64xf16> -> memref<64x64xf16>
            %18 = loom.alloc [64, 64] on @L1 : memref<64x64xf16>
            %19 = loom.semaphore_take %18 : memref<64x64xf16> -> memref<64x64xf16>
            %20 = loom.alloc [64, 64] on @L1 : memref<64x64xf16>
            %21 = loom.semaphore_take %20 : memref<64x64xf16> -> memref<64x64xf16>
            scf.for %arg7 = %c0 to %c8 step %c1 {
              %25 = arith.muli %arg7, %c64 : index
              %26 = arith.muli %13, %c32768 : index
              %27 = arith.addi %26, %25 : index
              %reinterpret_cast_0 = memref.reinterpret_cast %arg0 to offset: [%27], sizes: [64, 64], strides: [512, 1] : memref<4096x512xf16> to memref<64x64xf16, strided<[512, 1], offset: ?>>
              loom.copy %reinterpret_cast_0, %19 src_mem_space @DRAM dst_mem_space @L1, interconnect : [@vertical_links], broadcast : [8, 1] : memref<64x64xf16, strided<[512, 1], offset: ?>>, memref<64x64xf16>
              %28 = arith.muli %15, %c64 : index
              %29 = arith.muli %arg7, %c262144 : index
              %30 = arith.addi %29, %28 : index
              %reinterpret_cast_1 = memref.reinterpret_cast %arg1 to offset: [%30], sizes: [64, 64], strides: [4096, 1] : memref<512x4096xf16> to memref<64x64xf16, strided<[4096, 1], offset: ?>>
              loom.copy %reinterpret_cast_1, %17 src_mem_space @DRAM dst_mem_space @L1, interconnect : [@horizontal_links], broadcast : [1, 8] : memref<64x64xf16, strided<[4096, 1], offset: ?>>, memref<64x64xf16>
              loom.matmul ins(%19, %17 : memref<64x64xf16>, memref<64x64xf16>) outs(%21 : memref<64x64xf16>)
              loom.semaphore_give %17 : memref<64x64xf16>
              loom.semaphore_give %19 : memref<64x64xf16>
            }
            %22 = arith.muli %15, %c64 : index
            %23 = arith.muli %13, %c262144 : index
            %24 = arith.addi %23, %22 : index
            %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%24], sizes: [64, 64], strides: [4096, 1] : memref<4096x4096xf16> to memref<64x64xf16, strided<[4096, 1], offset: ?>>
            loom.copy %21, %reinterpret_cast src_mem_space @L1 dst_mem_space @DRAM, interconnect : [], broadcast : [1, 1] : memref<64x64xf16>, memref<64x64xf16, strided<[4096, 1], offset: ?>>
            loom.semaphore_give %21 : memref<64x64xf16>
          }
        }
        scf.reduce 
      } {loom.iter_types = [#loom.iter_type<spatial>, #loom.iter_type<spatial>], loom.mapped_to_dims = [@x, @y]}
      return
    }
  }
  module attributes {loom.pass_name = "Materialize"} {
    func.func @matmul__d0i0_d1i1__f10__d_d__BK64__BM64__BN64(%arg0: memref<4096x512xf16>, %arg1: memref<512x4096xf16>, %arg2: memref<4096x4096xf16>) {
      %c262144 = arith.constant 262144 : index
      %c32768 = arith.constant 32768 : index
      %c64 = arith.constant 64 : index
      %c0 = arith.constant 0 : index
      %c8 = arith.constant 8 : index
      %c1 = arith.constant 1 : index
      scf.parallel (%arg3, %arg4) = (%c0, %c0) to (%c8, %c8) step (%c1, %c1) {
        scf.for %arg5 = %c0 to %c8 step %c1 {
          scf.for %arg6 = %c0 to %c8 step %c1 {
            %12 = arith.muli %arg6, %c8 overflow<nsw> : index
            %13 = arith.addi %arg3, %12 : index
            %14 = arith.muli %arg5, %c8 overflow<nsw> : index
            %15 = arith.addi %arg4, %14 : index
            %16 = loom.alloc [64, 64] on @L1 : memref<64x64xf16>
            %17 = loom.semaphore_take %16 : memref<64x64xf16> -> memref<64x64xf16>
            %18 = loom.alloc [64, 64] on @L1 : memref<64x64xf16>
            %19 = loom.semaphore_take %18 : memref<64x64xf16> -> memref<64x64xf16>
            %20 = loom.alloc [64, 64] on @L1 : memref<64x64xf16>
            %21 = loom.semaphore_take %20 : memref<64x64xf16> -> memref<64x64xf16>
            scf.for %arg7 = %c0 to %c8 step %c1 {
              %25 = arith.muli %arg7, %c64 : index
              %26 = arith.muli %13, %c32768 : index
              %27 = arith.addi %26, %25 : index
              %reinterpret_cast_0 = memref.reinterpret_cast %arg0 to offset: [%27], sizes: [64, 64], strides: [512, 1] : memref<4096x512xf16> to memref<64x64xf16, strided<[512, 1], offset: ?>>
              loom.copy %reinterpret_cast_0, %19 src_mem_space @DRAM dst_mem_space @L1, interconnect : [], broadcast : [1, 1] : memref<64x64xf16, strided<[512, 1], offset: ?>>, memref<64x64xf16>
              %28 = arith.muli %15, %c64 : index
              %29 = arith.muli %arg7, %c262144 : index
              %30 = arith.addi %29, %28 : index
              %reinterpret_cast_1 = memref.reinterpret_cast %arg1 to offset: [%30], sizes: [64, 64], strides: [4096, 1] : memref<512x4096xf16> to memref<64x64xf16, strided<[4096, 1], offset: ?>>
              loom.copy %reinterpret_cast_1, %17 src_mem_space @DRAM dst_mem_space @L1, interconnect : [], broadcast : [1, 1] : memref<64x64xf16, strided<[4096, 1], offset: ?>>, memref<64x64xf16>
              loom.matmul ins(%19, %17 : memref<64x64xf16>, memref<64x64xf16>) outs(%21 : memref<64x64xf16>)
              loom.semaphore_give %17 : memref<64x64xf16>
              loom.semaphore_give %19 : memref<64x64xf16>
            }
            %22 = arith.muli %15, %c64 : index
            %23 = arith.muli %13, %c262144 : index
            %24 = arith.addi %23, %22 : index
            %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%24], sizes: [64, 64], strides: [4096, 1] : memref<4096x4096xf16> to memref<64x64xf16, strided<[4096, 1], offset: ?>>
            loom.copy %21, %reinterpret_cast src_mem_space @L1 dst_mem_space @DRAM, interconnect : [], broadcast : [1, 1] : memref<64x64xf16>, memref<64x64xf16, strided<[4096, 1], offset: ?>>
            loom.semaphore_give %21 : memref<64x64xf16>
          }
        }
        scf.reduce 
      } {loom.iter_types = [#loom.iter_type<spatial>, #loom.iter_type<spatial>], loom.mapped_to_dims = [@x, @y]}
      return
    }
  }
  module attributes {loom.pass_name = "Materialize"} {
    func.func @matmul__d0i0_d1i1__f10__d_h__BK64__BM64__BN64(%arg0: memref<4096x512xf16>, %arg1: memref<512x4096xf16>, %arg2: memref<4096x4096xf16>) {
      %c262144 = arith.constant 262144 : index
      %c32768 = arith.constant 32768 : index
      %c64 = arith.constant 64 : index
      %c0 = arith.constant 0 : index
      %c8 = arith.constant 8 : index
      %c1 = arith.constant 1 : index
      scf.parallel (%arg3, %arg4) = (%c0, %c0) to (%c8, %c8) step (%c1, %c1) {
        scf.for %arg5 = %c0 to %c8 step %c1 {
          scf.for %arg6 = %c0 to %c8 step %c1 {
            %12 = arith.muli %arg6, %c8 overflow<nsw> : index
            %13 = arith.addi %arg3, %12 : index
            %14 = arith.muli %arg5, %c8 overflow<nsw> : index
            %15 = arith.addi %arg4, %14 : index
            %16 = loom.alloc [64, 64] on @L1 : memref<64x64xf16>
            %17 = loom.semaphore_take %16 : memref<64x64xf16> -> memref<64x64xf16>
            %18 = loom.alloc [64, 64] on @L1 : memref<64x64xf16>
            %19 = loom.semaphore_take %18 : memref<64x64xf16> -> memref<64x64xf16>
            %20 = loom.alloc [64, 64] on @L1 : memref<64x64xf16>
            %21 = loom.semaphore_take %20 : memref<64x64xf16> -> memref<64x64xf16>
            scf.for %arg7 = %c0 to %c8 step %c1 {
              %25 = arith.muli %arg7, %c64 : index
              %26 = arith.muli %13, %c32768 : index
              %27 = arith.addi %26, %25 : index
              %reinterpret_cast_0 = memref.reinterpret_cast %arg0 to offset: [%27], sizes: [64, 64], strides: [512, 1] : memref<4096x512xf16> to memref<64x64xf16, strided<[512, 1], offset: ?>>
              loom.copy %reinterpret_cast_0, %19 src_mem_space @DRAM dst_mem_space @L1, interconnect : [], broadcast : [1, 1] : memref<64x64xf16, strided<[512, 1], offset: ?>>, memref<64x64xf16>
              %28 = arith.muli %15, %c64 : index
              %29 = arith.muli %arg7, %c262144 : index
              %30 = arith.addi %29, %28 : index
              %reinterpret_cast_1 = memref.reinterpret_cast %arg1 to offset: [%30], sizes: [64, 64], strides: [4096, 1] : memref<512x4096xf16> to memref<64x64xf16, strided<[4096, 1], offset: ?>>
              loom.copy %reinterpret_cast_1, %17 src_mem_space @DRAM dst_mem_space @L1, interconnect : [@horizontal_links], broadcast : [1, 8] : memref<64x64xf16, strided<[4096, 1], offset: ?>>, memref<64x64xf16>
              loom.matmul ins(%19, %17 : memref<64x64xf16>, memref<64x64xf16>) outs(%21 : memref<64x64xf16>)
              loom.semaphore_give %17 : memref<64x64xf16>
              loom.semaphore_give %19 : memref<64x64xf16>
            }
            %22 = arith.muli %15, %c64 : index
            %23 = arith.muli %13, %c262144 : index
            %24 = arith.addi %23, %22 : index
            %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%24], sizes: [64, 64], strides: [4096, 1] : memref<4096x4096xf16> to memref<64x64xf16, strided<[4096, 1], offset: ?>>
            loom.copy %21, %reinterpret_cast src_mem_space @L1 dst_mem_space @DRAM, interconnect : [], broadcast : [1, 1] : memref<64x64xf16>, memref<64x64xf16, strided<[4096, 1], offset: ?>>
            loom.semaphore_give %21 : memref<64x64xf16>
          }
        }
        scf.reduce 
      } {loom.iter_types = [#loom.iter_type<spatial>, #loom.iter_type<spatial>], loom.mapped_to_dims = [@x, @y]}
      return
    }
  }
  module attributes {loom.pass_name = "Materialize"} {
    func.func @matmul__d0i0_d1i1__f10__v_d__BK64__BM64__BN64(%arg0: memref<4096x512xf16>, %arg1: memref<512x4096xf16>, %arg2: memref<4096x4096xf16>) {
      %c262144 = arith.constant 262144 : index
      %c32768 = arith.constant 32768 : index
      %c64 = arith.constant 64 : index
      %c0 = arith.constant 0 : index
      %c8 = arith.constant 8 : index
      %c1 = arith.constant 1 : index
      scf.parallel (%arg3, %arg4) = (%c0, %c0) to (%c8, %c8) step (%c1, %c1) {
        scf.for %arg5 = %c0 to %c8 step %c1 {
          scf.for %arg6 = %c0 to %c8 step %c1 {
            %12 = arith.muli %arg6, %c8 overflow<nsw> : index
            %13 = arith.addi %arg3, %12 : index
            %14 = arith.muli %arg5, %c8 overflow<nsw> : index
            %15 = arith.addi %arg4, %14 : index
            %16 = loom.alloc [64, 64] on @L1 : memref<64x64xf16>
            %17 = loom.semaphore_take %16 : memref<64x64xf16> -> memref<64x64xf16>
            %18 = loom.alloc [64, 64] on @L1 : memref<64x64xf16>
            %19 = loom.semaphore_take %18 : memref<64x64xf16> -> memref<64x64xf16>
            %20 = loom.alloc [64, 64] on @L1 : memref<64x64xf16>
            %21 = loom.semaphore_take %20 : memref<64x64xf16> -> memref<64x64xf16>
            scf.for %arg7 = %c0 to %c8 step %c1 {
              %25 = arith.muli %arg7, %c64 : index
              %26 = arith.muli %13, %c32768 : index
              %27 = arith.addi %26, %25 : index
              %reinterpret_cast_0 = memref.reinterpret_cast %arg0 to offset: [%27], sizes: [64, 64], strides: [512, 1] : memref<4096x512xf16> to memref<64x64xf16, strided<[512, 1], offset: ?>>
              loom.copy %reinterpret_cast_0, %19 src_mem_space @DRAM dst_mem_space @L1, interconnect : [@vertical_links], broadcast : [8, 1] : memref<64x64xf16, strided<[512, 1], offset: ?>>, memref<64x64xf16>
              %28 = arith.muli %15, %c64 : index
              %29 = arith.muli %arg7, %c262144 : index
              %30 = arith.addi %29, %28 : index
              %reinterpret_cast_1 = memref.reinterpret_cast %arg1 to offset: [%30], sizes: [64, 64], strides: [4096, 1] : memref<512x4096xf16> to memref<64x64xf16, strided<[4096, 1], offset: ?>>
              loom.copy %reinterpret_cast_1, %17 src_mem_space @DRAM dst_mem_space @L1, interconnect : [], broadcast : [1, 1] : memref<64x64xf16, strided<[4096, 1], offset: ?>>, memref<64x64xf16>
              loom.matmul ins(%19, %17 : memref<64x64xf16>, memref<64x64xf16>) outs(%21 : memref<64x64xf16>)
              loom.semaphore_give %17 : memref<64x64xf16>
              loom.semaphore_give %19 : memref<64x64xf16>
            }
            %22 = arith.muli %15, %c64 : index
            %23 = arith.muli %13, %c262144 : index
            %24 = arith.addi %23, %22 : index
            %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%24], sizes: [64, 64], strides: [4096, 1] : memref<4096x4096xf16> to memref<64x64xf16, strided<[4096, 1], offset: ?>>
            loom.copy %21, %reinterpret_cast src_mem_space @L1 dst_mem_space @DRAM, interconnect : [], broadcast : [1, 1] : memref<64x64xf16>, memref<64x64xf16, strided<[4096, 1], offset: ?>>
            loom.semaphore_give %21 : memref<64x64xf16>
          }
        }
        scf.reduce 
      } {loom.iter_types = [#loom.iter_type<spatial>, #loom.iter_type<spatial>], loom.mapped_to_dims = [@x, @y]}
      return
    }
  }
  module attributes {loom.pass_name = "Materialize"} {
    func.func @matmul__d0i0_d1i1__f10__v_h__BK64__BM64__BN64(%arg0: memref<4096x512xf16>, %arg1: memref<512x4096xf16>, %arg2: memref<4096x4096xf16>) {
      %c262144 = arith.constant 262144 : index
      %c32768 = arith.constant 32768 : index
      %c64 = arith.constant 64 : index
      %c0 = arith.constant 0 : index
      %c8 = arith.constant 8 : index
      %c1 = arith.constant 1 : index
      scf.parallel (%arg3, %arg4) = (%c0, %c0) to (%c8, %c8) step (%c1, %c1) {
        scf.for %arg5 = %c0 to %c8 step %c1 {
          scf.for %arg6 = %c0 to %c8 step %c1 {
            %12 = arith.muli %arg6, %c8 overflow<nsw> : index
            %13 = arith.addi %arg3, %12 : index
            %14 = arith.muli %arg5, %c8 overflow<nsw> : index
            %15 = arith.addi %arg4, %14 : index
            %16 = loom.alloc [64, 64] on @L1 : memref<64x64xf16>
            %17 = loom.semaphore_take %16 : memref<64x64xf16> -> memref<64x64xf16>
            %18 = loom.alloc [64, 64] on @L1 : memref<64x64xf16>
            %19 = loom.semaphore_take %18 : memref<64x64xf16> -> memref<64x64xf16>
            %20 = loom.alloc [64, 64] on @L1 : memref<64x64xf16>
            %21 = loom.semaphore_take %20 : memref<64x64xf16> -> memref<64x64xf16>
            scf.for %arg7 = %c0 to %c8 step %c1 {
              %25 = arith.muli %arg7, %c64 : index
              %26 = arith.muli %13, %c32768 : index
              %27 = arith.addi %26, %25 : index
              %reinterpret_cast_0 = memref.reinterpret_cast %arg0 to offset: [%27], sizes: [64, 64], strides: [512, 1] : memref<4096x512xf16> to memref<64x64xf16, strided<[512, 1], offset: ?>>
              loom.copy %reinterpret_cast_0, %19 src_mem_space @DRAM dst_mem_space @L1, interconnect : [@vertical_links], broadcast : [8, 1] : memref<64x64xf16, strided<[512, 1], offset: ?>>, memref<64x64xf16>
              %28 = arith.muli %15, %c64 : index
              %29 = arith.muli %arg7, %c262144 : index
              %30 = arith.addi %29, %28 : index
              %reinterpret_cast_1 = memref.reinterpret_cast %arg1 to offset: [%30], sizes: [64, 64], strides: [4096, 1] : memref<512x4096xf16> to memref<64x64xf16, strided<[4096, 1], offset: ?>>
              loom.copy %reinterpret_cast_1, %17 src_mem_space @DRAM dst_mem_space @L1, interconnect : [@horizontal_links], broadcast : [1, 8] : memref<64x64xf16, strided<[4096, 1], offset: ?>>, memref<64x64xf16>
              loom.matmul ins(%19, %17 : memref<64x64xf16>, memref<64x64xf16>) outs(%21 : memref<64x64xf16>)
              loom.semaphore_give %17 : memref<64x64xf16>
              loom.semaphore_give %19 : memref<64x64xf16>
            }
            %22 = arith.muli %15, %c64 : index
            %23 = arith.muli %13, %c262144 : index
            %24 = arith.addi %23, %22 : index
            %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%24], sizes: [64, 64], strides: [4096, 1] : memref<4096x4096xf16> to memref<64x64xf16, strided<[4096, 1], offset: ?>>
            loom.copy %21, %reinterpret_cast src_mem_space @L1 dst_mem_space @DRAM, interconnect : [], broadcast : [1, 1] : memref<64x64xf16>, memref<64x64xf16, strided<[4096, 1], offset: ?>>
            loom.semaphore_give %21 : memref<64x64xf16>
          }
        }
        scf.reduce 
      } {loom.iter_types = [#loom.iter_type<spatial>, #loom.iter_type<spatial>], loom.mapped_to_dims = [@x, @y]}
      return
    }
  }
  module attributes {loom.pass_name = "Materialize"} {
    func.func @matmul__d1i0_d0i1__f01__d_d__BK64__BM64__BN64(%arg0: memref<4096x512xf16>, %arg1: memref<512x4096xf16>, %arg2: memref<4096x4096xf16>) {
      %c262144 = arith.constant 262144 : index
      %c32768 = arith.constant 32768 : index
      %c64 = arith.constant 64 : index
      %c0 = arith.constant 0 : index
      %c8 = arith.constant 8 : index
      %c1 = arith.constant 1 : index
      scf.parallel (%arg3, %arg4) = (%c0, %c0) to (%c8, %c8) step (%c1, %c1) {
        scf.for %arg5 = %c0 to %c8 step %c1 {
          scf.for %arg6 = %c0 to %c8 step %c1 {
            %12 = arith.muli %arg5, %c8 overflow<nsw> : index
            %13 = arith.addi %arg3, %12 : index
            %14 = arith.muli %arg6, %c8 overflow<nsw> : index
            %15 = arith.addi %arg4, %14 : index
            %16 = loom.alloc [64, 64] on @L1 : memref<64x64xf16>
            %17 = loom.semaphore_take %16 : memref<64x64xf16> -> memref<64x64xf16>
            %18 = loom.alloc [64, 64] on @L1 : memref<64x64xf16>
            %19 = loom.semaphore_take %18 : memref<64x64xf16> -> memref<64x64xf16>
            %20 = loom.alloc [64, 64] on @L1 : memref<64x64xf16>
            %21 = loom.semaphore_take %20 : memref<64x64xf16> -> memref<64x64xf16>
            scf.for %arg7 = %c0 to %c8 step %c1 {
              %25 = arith.muli %arg7, %c64 : index
              %26 = arith.muli %13, %c32768 : index
              %27 = arith.addi %26, %25 : index
              %reinterpret_cast_0 = memref.reinterpret_cast %arg0 to offset: [%27], sizes: [64, 64], strides: [512, 1] : memref<4096x512xf16> to memref<64x64xf16, strided<[512, 1], offset: ?>>
              loom.copy %reinterpret_cast_0, %19 src_mem_space @DRAM dst_mem_space @L1, interconnect : [], broadcast : [1, 1] : memref<64x64xf16, strided<[512, 1], offset: ?>>, memref<64x64xf16>
              %28 = arith.muli %15, %c64 : index
              %29 = arith.muli %arg7, %c262144 : index
              %30 = arith.addi %29, %28 : index
              %reinterpret_cast_1 = memref.reinterpret_cast %arg1 to offset: [%30], sizes: [64, 64], strides: [4096, 1] : memref<512x4096xf16> to memref<64x64xf16, strided<[4096, 1], offset: ?>>
              loom.copy %reinterpret_cast_1, %17 src_mem_space @DRAM dst_mem_space @L1, interconnect : [], broadcast : [1, 1] : memref<64x64xf16, strided<[4096, 1], offset: ?>>, memref<64x64xf16>
              loom.matmul ins(%19, %17 : memref<64x64xf16>, memref<64x64xf16>) outs(%21 : memref<64x64xf16>)
              loom.semaphore_give %17 : memref<64x64xf16>
              loom.semaphore_give %19 : memref<64x64xf16>
            }
            %22 = arith.muli %15, %c64 : index
            %23 = arith.muli %13, %c262144 : index
            %24 = arith.addi %23, %22 : index
            %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%24], sizes: [64, 64], strides: [4096, 1] : memref<4096x4096xf16> to memref<64x64xf16, strided<[4096, 1], offset: ?>>
            loom.copy %21, %reinterpret_cast src_mem_space @L1 dst_mem_space @DRAM, interconnect : [], broadcast : [1, 1] : memref<64x64xf16>, memref<64x64xf16, strided<[4096, 1], offset: ?>>
            loom.semaphore_give %21 : memref<64x64xf16>
          }
        }
        scf.reduce 
      } {loom.iter_types = [#loom.iter_type<spatial>, #loom.iter_type<spatial>], loom.mapped_to_dims = [@y, @x]}
      return
    }
  }
  module attributes {loom.pass_name = "Materialize"} {
    func.func @matmul__d1i0_d0i1__f01__d_v__BK64__BM64__BN64(%arg0: memref<4096x512xf16>, %arg1: memref<512x4096xf16>, %arg2: memref<4096x4096xf16>) {
      %c262144 = arith.constant 262144 : index
      %c32768 = arith.constant 32768 : index
      %c64 = arith.constant 64 : index
      %c0 = arith.constant 0 : index
      %c8 = arith.constant 8 : index
      %c1 = arith.constant 1 : index
      scf.parallel (%arg3, %arg4) = (%c0, %c0) to (%c8, %c8) step (%c1, %c1) {
        scf.for %arg5 = %c0 to %c8 step %c1 {
          scf.for %arg6 = %c0 to %c8 step %c1 {
            %12 = arith.muli %arg5, %c8 overflow<nsw> : index
            %13 = arith.addi %arg3, %12 : index
            %14 = arith.muli %arg6, %c8 overflow<nsw> : index
            %15 = arith.addi %arg4, %14 : index
            %16 = loom.alloc [64, 64] on @L1 : memref<64x64xf16>
            %17 = loom.semaphore_take %16 : memref<64x64xf16> -> memref<64x64xf16>
            %18 = loom.alloc [64, 64] on @L1 : memref<64x64xf16>
            %19 = loom.semaphore_take %18 : memref<64x64xf16> -> memref<64x64xf16>
            %20 = loom.alloc [64, 64] on @L1 : memref<64x64xf16>
            %21 = loom.semaphore_take %20 : memref<64x64xf16> -> memref<64x64xf16>
            scf.for %arg7 = %c0 to %c8 step %c1 {
              %25 = arith.muli %arg7, %c64 : index
              %26 = arith.muli %13, %c32768 : index
              %27 = arith.addi %26, %25 : index
              %reinterpret_cast_0 = memref.reinterpret_cast %arg0 to offset: [%27], sizes: [64, 64], strides: [512, 1] : memref<4096x512xf16> to memref<64x64xf16, strided<[512, 1], offset: ?>>
              loom.copy %reinterpret_cast_0, %19 src_mem_space @DRAM dst_mem_space @L1, interconnect : [], broadcast : [1, 1] : memref<64x64xf16, strided<[512, 1], offset: ?>>, memref<64x64xf16>
              %28 = arith.muli %15, %c64 : index
              %29 = arith.muli %arg7, %c262144 : index
              %30 = arith.addi %29, %28 : index
              %reinterpret_cast_1 = memref.reinterpret_cast %arg1 to offset: [%30], sizes: [64, 64], strides: [4096, 1] : memref<512x4096xf16> to memref<64x64xf16, strided<[4096, 1], offset: ?>>
              loom.copy %reinterpret_cast_1, %17 src_mem_space @DRAM dst_mem_space @L1, interconnect : [@vertical_links], broadcast : [8, 1] : memref<64x64xf16, strided<[4096, 1], offset: ?>>, memref<64x64xf16>
              loom.matmul ins(%19, %17 : memref<64x64xf16>, memref<64x64xf16>) outs(%21 : memref<64x64xf16>)
              loom.semaphore_give %17 : memref<64x64xf16>
              loom.semaphore_give %19 : memref<64x64xf16>
            }
            %22 = arith.muli %15, %c64 : index
            %23 = arith.muli %13, %c262144 : index
            %24 = arith.addi %23, %22 : index
            %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%24], sizes: [64, 64], strides: [4096, 1] : memref<4096x4096xf16> to memref<64x64xf16, strided<[4096, 1], offset: ?>>
            loom.copy %21, %reinterpret_cast src_mem_space @L1 dst_mem_space @DRAM, interconnect : [], broadcast : [1, 1] : memref<64x64xf16>, memref<64x64xf16, strided<[4096, 1], offset: ?>>
            loom.semaphore_give %21 : memref<64x64xf16>
          }
        }
        scf.reduce 
      } {loom.iter_types = [#loom.iter_type<spatial>, #loom.iter_type<spatial>], loom.mapped_to_dims = [@y, @x]}
      return
    }
  }
  module attributes {loom.pass_name = "Materialize"} {
    func.func @matmul__d1i0_d0i1__f01__h_d__BK64__BM64__BN64(%arg0: memref<4096x512xf16>, %arg1: memref<512x4096xf16>, %arg2: memref<4096x4096xf16>) {
      %c262144 = arith.constant 262144 : index
      %c32768 = arith.constant 32768 : index
      %c64 = arith.constant 64 : index
      %c0 = arith.constant 0 : index
      %c8 = arith.constant 8 : index
      %c1 = arith.constant 1 : index
      scf.parallel (%arg3, %arg4) = (%c0, %c0) to (%c8, %c8) step (%c1, %c1) {
        scf.for %arg5 = %c0 to %c8 step %c1 {
          scf.for %arg6 = %c0 to %c8 step %c1 {
            %12 = arith.muli %arg5, %c8 overflow<nsw> : index
            %13 = arith.addi %arg3, %12 : index
            %14 = arith.muli %arg6, %c8 overflow<nsw> : index
            %15 = arith.addi %arg4, %14 : index
            %16 = loom.alloc [64, 64] on @L1 : memref<64x64xf16>
            %17 = loom.semaphore_take %16 : memref<64x64xf16> -> memref<64x64xf16>
            %18 = loom.alloc [64, 64] on @L1 : memref<64x64xf16>
            %19 = loom.semaphore_take %18 : memref<64x64xf16> -> memref<64x64xf16>
            %20 = loom.alloc [64, 64] on @L1 : memref<64x64xf16>
            %21 = loom.semaphore_take %20 : memref<64x64xf16> -> memref<64x64xf16>
            scf.for %arg7 = %c0 to %c8 step %c1 {
              %25 = arith.muli %arg7, %c64 : index
              %26 = arith.muli %13, %c32768 : index
              %27 = arith.addi %26, %25 : index
              %reinterpret_cast_0 = memref.reinterpret_cast %arg0 to offset: [%27], sizes: [64, 64], strides: [512, 1] : memref<4096x512xf16> to memref<64x64xf16, strided<[512, 1], offset: ?>>
              loom.copy %reinterpret_cast_0, %19 src_mem_space @DRAM dst_mem_space @L1, interconnect : [@horizontal_links], broadcast : [1, 8] : memref<64x64xf16, strided<[512, 1], offset: ?>>, memref<64x64xf16>
              %28 = arith.muli %15, %c64 : index
              %29 = arith.muli %arg7, %c262144 : index
              %30 = arith.addi %29, %28 : index
              %reinterpret_cast_1 = memref.reinterpret_cast %arg1 to offset: [%30], sizes: [64, 64], strides: [4096, 1] : memref<512x4096xf16> to memref<64x64xf16, strided<[4096, 1], offset: ?>>
              loom.copy %reinterpret_cast_1, %17 src_mem_space @DRAM dst_mem_space @L1, interconnect : [], broadcast : [1, 1] : memref<64x64xf16, strided<[4096, 1], offset: ?>>, memref<64x64xf16>
              loom.matmul ins(%19, %17 : memref<64x64xf16>, memref<64x64xf16>) outs(%21 : memref<64x64xf16>)
              loom.semaphore_give %17 : memref<64x64xf16>
              loom.semaphore_give %19 : memref<64x64xf16>
            }
            %22 = arith.muli %15, %c64 : index
            %23 = arith.muli %13, %c262144 : index
            %24 = arith.addi %23, %22 : index
            %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%24], sizes: [64, 64], strides: [4096, 1] : memref<4096x4096xf16> to memref<64x64xf16, strided<[4096, 1], offset: ?>>
            loom.copy %21, %reinterpret_cast src_mem_space @L1 dst_mem_space @DRAM, interconnect : [], broadcast : [1, 1] : memref<64x64xf16>, memref<64x64xf16, strided<[4096, 1], offset: ?>>
            loom.semaphore_give %21 : memref<64x64xf16>
          }
        }
        scf.reduce 
      } {loom.iter_types = [#loom.iter_type<spatial>, #loom.iter_type<spatial>], loom.mapped_to_dims = [@y, @x]}
      return
    }
  }
  module attributes {loom.pass_name = "Materialize"} {
    func.func @matmul__d1i0_d0i1__f01__h_v__BK64__BM64__BN64(%arg0: memref<4096x512xf16>, %arg1: memref<512x4096xf16>, %arg2: memref<4096x4096xf16>) {
      %c262144 = arith.constant 262144 : index
      %c32768 = arith.constant 32768 : index
      %c64 = arith.constant 64 : index
      %c0 = arith.constant 0 : index
      %c8 = arith.constant 8 : index
      %c1 = arith.constant 1 : index
      scf.parallel (%arg3, %arg4) = (%c0, %c0) to (%c8, %c8) step (%c1, %c1) {
        scf.for %arg5 = %c0 to %c8 step %c1 {
          scf.for %arg6 = %c0 to %c8 step %c1 {
            %12 = arith.muli %arg5, %c8 overflow<nsw> : index
            %13 = arith.addi %arg3, %12 : index
            %14 = arith.muli %arg6, %c8 overflow<nsw> : index
            %15 = arith.addi %arg4, %14 : index
            %16 = loom.alloc [64, 64] on @L1 : memref<64x64xf16>
            %17 = loom.semaphore_take %16 : memref<64x64xf16> -> memref<64x64xf16>
            %18 = loom.alloc [64, 64] on @L1 : memref<64x64xf16>
            %19 = loom.semaphore_take %18 : memref<64x64xf16> -> memref<64x64xf16>
            %20 = loom.alloc [64, 64] on @L1 : memref<64x64xf16>
            %21 = loom.semaphore_take %20 : memref<64x64xf16> -> memref<64x64xf16>
            scf.for %arg7 = %c0 to %c8 step %c1 {
              %25 = arith.muli %arg7, %c64 : index
              %26 = arith.muli %13, %c32768 : index
              %27 = arith.addi %26, %25 : index
              %reinterpret_cast_0 = memref.reinterpret_cast %arg0 to offset: [%27], sizes: [64, 64], strides: [512, 1] : memref<4096x512xf16> to memref<64x64xf16, strided<[512, 1], offset: ?>>
              loom.copy %reinterpret_cast_0, %19 src_mem_space @DRAM dst_mem_space @L1, interconnect : [@horizontal_links], broadcast : [1, 8] : memref<64x64xf16, strided<[512, 1], offset: ?>>, memref<64x64xf16>
              %28 = arith.muli %15, %c64 : index
              %29 = arith.muli %arg7, %c262144 : index
              %30 = arith.addi %29, %28 : index
              %reinterpret_cast_1 = memref.reinterpret_cast %arg1 to offset: [%30], sizes: [64, 64], strides: [4096, 1] : memref<512x4096xf16> to memref<64x64xf16, strided<[4096, 1], offset: ?>>
              loom.copy %reinterpret_cast_1, %17 src_mem_space @DRAM dst_mem_space @L1, interconnect : [@vertical_links], broadcast : [8, 1] : memref<64x64xf16, strided<[4096, 1], offset: ?>>, memref<64x64xf16>
              loom.matmul ins(%19, %17 : memref<64x64xf16>, memref<64x64xf16>) outs(%21 : memref<64x64xf16>)
              loom.semaphore_give %17 : memref<64x64xf16>
              loom.semaphore_give %19 : memref<64x64xf16>
            }
            %22 = arith.muli %15, %c64 : index
            %23 = arith.muli %13, %c262144 : index
            %24 = arith.addi %23, %22 : index
            %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%24], sizes: [64, 64], strides: [4096, 1] : memref<4096x4096xf16> to memref<64x64xf16, strided<[4096, 1], offset: ?>>
            loom.copy %21, %reinterpret_cast src_mem_space @L1 dst_mem_space @DRAM, interconnect : [], broadcast : [1, 1] : memref<64x64xf16>, memref<64x64xf16, strided<[4096, 1], offset: ?>>
            loom.semaphore_give %21 : memref<64x64xf16>
          }
        }
        scf.reduce 
      } {loom.iter_types = [#loom.iter_type<spatial>, #loom.iter_type<spatial>], loom.mapped_to_dims = [@y, @x]}
      return
    }
  }
  module attributes {loom.pass_name = "Materialize"} {
    func.func @matmul__d1i0_d0i1__f10__d_d__BK64__BM64__BN64(%arg0: memref<4096x512xf16>, %arg1: memref<512x4096xf16>, %arg2: memref<4096x4096xf16>) {
      %c262144 = arith.constant 262144 : index
      %c32768 = arith.constant 32768 : index
      %c64 = arith.constant 64 : index
      %c0 = arith.constant 0 : index
      %c8 = arith.constant 8 : index
      %c1 = arith.constant 1 : index
      scf.parallel (%arg3, %arg4) = (%c0, %c0) to (%c8, %c8) step (%c1, %c1) {
        scf.for %arg5 = %c0 to %c8 step %c1 {
          scf.for %arg6 = %c0 to %c8 step %c1 {
            %12 = arith.muli %arg6, %c8 overflow<nsw> : index
            %13 = arith.addi %arg3, %12 : index
            %14 = arith.muli %arg5, %c8 overflow<nsw> : index
            %15 = arith.addi %arg4, %14 : index
            %16 = loom.alloc [64, 64] on @L1 : memref<64x64xf16>
            %17 = loom.semaphore_take %16 : memref<64x64xf16> -> memref<64x64xf16>
            %18 = loom.alloc [64, 64] on @L1 : memref<64x64xf16>
            %19 = loom.semaphore_take %18 : memref<64x64xf16> -> memref<64x64xf16>
            %20 = loom.alloc [64, 64] on @L1 : memref<64x64xf16>
            %21 = loom.semaphore_take %20 : memref<64x64xf16> -> memref<64x64xf16>
            scf.for %arg7 = %c0 to %c8 step %c1 {
              %25 = arith.muli %arg7, %c64 : index
              %26 = arith.muli %13, %c32768 : index
              %27 = arith.addi %26, %25 : index
              %reinterpret_cast_0 = memref.reinterpret_cast %arg0 to offset: [%27], sizes: [64, 64], strides: [512, 1] : memref<4096x512xf16> to memref<64x64xf16, strided<[512, 1], offset: ?>>
              loom.copy %reinterpret_cast_0, %19 src_mem_space @DRAM dst_mem_space @L1, interconnect : [], broadcast : [1, 1] : memref<64x64xf16, strided<[512, 1], offset: ?>>, memref<64x64xf16>
              %28 = arith.muli %15, %c64 : index
              %29 = arith.muli %arg7, %c262144 : index
              %30 = arith.addi %29, %28 : index
              %reinterpret_cast_1 = memref.reinterpret_cast %arg1 to offset: [%30], sizes: [64, 64], strides: [4096, 1] : memref<512x4096xf16> to memref<64x64xf16, strided<[4096, 1], offset: ?>>
              loom.copy %reinterpret_cast_1, %17 src_mem_space @DRAM dst_mem_space @L1, interconnect : [], broadcast : [1, 1] : memref<64x64xf16, strided<[4096, 1], offset: ?>>, memref<64x64xf16>
              loom.matmul ins(%19, %17 : memref<64x64xf16>, memref<64x64xf16>) outs(%21 : memref<64x64xf16>)
              loom.semaphore_give %17 : memref<64x64xf16>
              loom.semaphore_give %19 : memref<64x64xf16>
            }
            %22 = arith.muli %15, %c64 : index
            %23 = arith.muli %13, %c262144 : index
            %24 = arith.addi %23, %22 : index
            %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%24], sizes: [64, 64], strides: [4096, 1] : memref<4096x4096xf16> to memref<64x64xf16, strided<[4096, 1], offset: ?>>
            loom.copy %21, %reinterpret_cast src_mem_space @L1 dst_mem_space @DRAM, interconnect : [], broadcast : [1, 1] : memref<64x64xf16>, memref<64x64xf16, strided<[4096, 1], offset: ?>>
            loom.semaphore_give %21 : memref<64x64xf16>
          }
        }
        scf.reduce 
      } {loom.iter_types = [#loom.iter_type<spatial>, #loom.iter_type<spatial>], loom.mapped_to_dims = [@y, @x]}
      return
    }
  }
  module attributes {loom.pass_name = "Materialize"} {
    func.func @matmul__d1i0_d0i1__f10__d_v__BK64__BM64__BN64(%arg0: memref<4096x512xf16>, %arg1: memref<512x4096xf16>, %arg2: memref<4096x4096xf16>) {
      %c262144 = arith.constant 262144 : index
      %c32768 = arith.constant 32768 : index
      %c64 = arith.constant 64 : index
      %c0 = arith.constant 0 : index
      %c8 = arith.constant 8 : index
      %c1 = arith.constant 1 : index
      scf.parallel (%arg3, %arg4) = (%c0, %c0) to (%c8, %c8) step (%c1, %c1) {
        scf.for %arg5 = %c0 to %c8 step %c1 {
          scf.for %arg6 = %c0 to %c8 step %c1 {
            %12 = arith.muli %arg6, %c8 overflow<nsw> : index
            %13 = arith.addi %arg3, %12 : index
            %14 = arith.muli %arg5, %c8 overflow<nsw> : index
            %15 = arith.addi %arg4, %14 : index
            %16 = loom.alloc [64, 64] on @L1 : memref<64x64xf16>
            %17 = loom.semaphore_take %16 : memref<64x64xf16> -> memref<64x64xf16>
            %18 = loom.alloc [64, 64] on @L1 : memref<64x64xf16>
            %19 = loom.semaphore_take %18 : memref<64x64xf16> -> memref<64x64xf16>
            %20 = loom.alloc [64, 64] on @L1 : memref<64x64xf16>
            %21 = loom.semaphore_take %20 : memref<64x64xf16> -> memref<64x64xf16>
            scf.for %arg7 = %c0 to %c8 step %c1 {
              %25 = arith.muli %arg7, %c64 : index
              %26 = arith.muli %13, %c32768 : index
              %27 = arith.addi %26, %25 : index
              %reinterpret_cast_0 = memref.reinterpret_cast %arg0 to offset: [%27], sizes: [64, 64], strides: [512, 1] : memref<4096x512xf16> to memref<64x64xf16, strided<[512, 1], offset: ?>>
              loom.copy %reinterpret_cast_0, %19 src_mem_space @DRAM dst_mem_space @L1, interconnect : [], broadcast : [1, 1] : memref<64x64xf16, strided<[512, 1], offset: ?>>, memref<64x64xf16>
              %28 = arith.muli %15, %c64 : index
              %29 = arith.muli %arg7, %c262144 : index
              %30 = arith.addi %29, %28 : index
              %reinterpret_cast_1 = memref.reinterpret_cast %arg1 to offset: [%30], sizes: [64, 64], strides: [4096, 1] : memref<512x4096xf16> to memref<64x64xf16, strided<[4096, 1], offset: ?>>
              loom.copy %reinterpret_cast_1, %17 src_mem_space @DRAM dst_mem_space @L1, interconnect : [@vertical_links], broadcast : [8, 1] : memref<64x64xf16, strided<[4096, 1], offset: ?>>, memref<64x64xf16>
              loom.matmul ins(%19, %17 : memref<64x64xf16>, memref<64x64xf16>) outs(%21 : memref<64x64xf16>)
              loom.semaphore_give %17 : memref<64x64xf16>
              loom.semaphore_give %19 : memref<64x64xf16>
            }
            %22 = arith.muli %15, %c64 : index
            %23 = arith.muli %13, %c262144 : index
            %24 = arith.addi %23, %22 : index
            %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%24], sizes: [64, 64], strides: [4096, 1] : memref<4096x4096xf16> to memref<64x64xf16, strided<[4096, 1], offset: ?>>
            loom.copy %21, %reinterpret_cast src_mem_space @L1 dst_mem_space @DRAM, interconnect : [], broadcast : [1, 1] : memref<64x64xf16>, memref<64x64xf16, strided<[4096, 1], offset: ?>>
            loom.semaphore_give %21 : memref<64x64xf16>
          }
        }
        scf.reduce 
      } {loom.iter_types = [#loom.iter_type<spatial>, #loom.iter_type<spatial>], loom.mapped_to_dims = [@y, @x]}
      return
    }
  }
  module attributes {loom.pass_name = "Materialize"} {
    func.func @matmul__d1i0_d0i1__f10__h_d__BK64__BM64__BN64(%arg0: memref<4096x512xf16>, %arg1: memref<512x4096xf16>, %arg2: memref<4096x4096xf16>) {
      %c262144 = arith.constant 262144 : index
      %c32768 = arith.constant 32768 : index
      %c64 = arith.constant 64 : index
      %c0 = arith.constant 0 : index
      %c8 = arith.constant 8 : index
      %c1 = arith.constant 1 : index
      scf.parallel (%arg3, %arg4) = (%c0, %c0) to (%c8, %c8) step (%c1, %c1) {
        scf.for %arg5 = %c0 to %c8 step %c1 {
          scf.for %arg6 = %c0 to %c8 step %c1 {
            %12 = arith.muli %arg6, %c8 overflow<nsw> : index
            %13 = arith.addi %arg3, %12 : index
            %14 = arith.muli %arg5, %c8 overflow<nsw> : index
            %15 = arith.addi %arg4, %14 : index
            %16 = loom.alloc [64, 64] on @L1 : memref<64x64xf16>
            %17 = loom.semaphore_take %16 : memref<64x64xf16> -> memref<64x64xf16>
            %18 = loom.alloc [64, 64] on @L1 : memref<64x64xf16>
            %19 = loom.semaphore_take %18 : memref<64x64xf16> -> memref<64x64xf16>
            %20 = loom.alloc [64, 64] on @L1 : memref<64x64xf16>
            %21 = loom.semaphore_take %20 : memref<64x64xf16> -> memref<64x64xf16>
            scf.for %arg7 = %c0 to %c8 step %c1 {
              %25 = arith.muli %arg7, %c64 : index
              %26 = arith.muli %13, %c32768 : index
              %27 = arith.addi %26, %25 : index
              %reinterpret_cast_0 = memref.reinterpret_cast %arg0 to offset: [%27], sizes: [64, 64], strides: [512, 1] : memref<4096x512xf16> to memref<64x64xf16, strided<[512, 1], offset: ?>>
              loom.copy %reinterpret_cast_0, %19 src_mem_space @DRAM dst_mem_space @L1, interconnect : [@horizontal_links], broadcast : [1, 8] : memref<64x64xf16, strided<[512, 1], offset: ?>>, memref<64x64xf16>
              %28 = arith.muli %15, %c64 : index
              %29 = arith.muli %arg7, %c262144 : index
              %30 = arith.addi %29, %28 : index
              %reinterpret_cast_1 = memref.reinterpret_cast %arg1 to offset: [%30], sizes: [64, 64], strides: [4096, 1] : memref<512x4096xf16> to memref<64x64xf16, strided<[4096, 1], offset: ?>>
              loom.copy %reinterpret_cast_1, %17 src_mem_space @DRAM dst_mem_space @L1, interconnect : [], broadcast : [1, 1] : memref<64x64xf16, strided<[4096, 1], offset: ?>>, memref<64x64xf16>
              loom.matmul ins(%19, %17 : memref<64x64xf16>, memref<64x64xf16>) outs(%21 : memref<64x64xf16>)
              loom.semaphore_give %17 : memref<64x64xf16>
              loom.semaphore_give %19 : memref<64x64xf16>
            }
            %22 = arith.muli %15, %c64 : index
            %23 = arith.muli %13, %c262144 : index
            %24 = arith.addi %23, %22 : index
            %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%24], sizes: [64, 64], strides: [4096, 1] : memref<4096x4096xf16> to memref<64x64xf16, strided<[4096, 1], offset: ?>>
            loom.copy %21, %reinterpret_cast src_mem_space @L1 dst_mem_space @DRAM, interconnect : [], broadcast : [1, 1] : memref<64x64xf16>, memref<64x64xf16, strided<[4096, 1], offset: ?>>
            loom.semaphore_give %21 : memref<64x64xf16>
          }
        }
        scf.reduce 
      } {loom.iter_types = [#loom.iter_type<spatial>, #loom.iter_type<spatial>], loom.mapped_to_dims = [@y, @x]}
      return
    }
  }
  module attributes {loom.pass_name = "Materialize"} {
    func.func @matmul__d1i0_d0i1__f10__h_v__BK64__BM64__BN64(%arg0: memref<4096x512xf16>, %arg1: memref<512x4096xf16>, %arg2: memref<4096x4096xf16>) {
      %c262144 = arith.constant 262144 : index
      %c32768 = arith.constant 32768 : index
      %c64 = arith.constant 64 : index
      %c0 = arith.constant 0 : index
      %c8 = arith.constant 8 : index
      %c1 = arith.constant 1 : index
      scf.parallel (%arg3, %arg4) = (%c0, %c0) to (%c8, %c8) step (%c1, %c1) {
        scf.for %arg5 = %c0 to %c8 step %c1 {
          scf.for %arg6 = %c0 to %c8 step %c1 {
            %12 = arith.muli %arg6, %c8 overflow<nsw> : index
            %13 = arith.addi %arg3, %12 : index
            %14 = arith.muli %arg5, %c8 overflow<nsw> : index
            %15 = arith.addi %arg4, %14 : index
            %16 = loom.alloc [64, 64] on @L1 : memref<64x64xf16>
            %17 = loom.semaphore_take %16 : memref<64x64xf16> -> memref<64x64xf16>
            %18 = loom.alloc [64, 64] on @L1 : memref<64x64xf16>
            %19 = loom.semaphore_take %18 : memref<64x64xf16> -> memref<64x64xf16>
            %20 = loom.alloc [64, 64] on @L1 : memref<64x64xf16>
            %21 = loom.semaphore_take %20 : memref<64x64xf16> -> memref<64x64xf16>
            scf.for %arg7 = %c0 to %c8 step %c1 {
              %25 = arith.muli %arg7, %c64 : index
              %26 = arith.muli %13, %c32768 : index
              %27 = arith.addi %26, %25 : index
              %reinterpret_cast_0 = memref.reinterpret_cast %arg0 to offset: [%27], sizes: [64, 64], strides: [512, 1] : memref<4096x512xf16> to memref<64x64xf16, strided<[512, 1], offset: ?>>
              loom.copy %reinterpret_cast_0, %19 src_mem_space @DRAM dst_mem_space @L1, interconnect : [@horizontal_links], broadcast : [1, 8] : memref<64x64xf16, strided<[512, 1], offset: ?>>, memref<64x64xf16>
              %28 = arith.muli %15, %c64 : index
              %29 = arith.muli %arg7, %c262144 : index
              %30 = arith.addi %29, %28 : index
              %reinterpret_cast_1 = memref.reinterpret_cast %arg1 to offset: [%30], sizes: [64, 64], strides: [4096, 1] : memref<512x4096xf16> to memref<64x64xf16, strided<[4096, 1], offset: ?>>
              loom.copy %reinterpret_cast_1, %17 src_mem_space @DRAM dst_mem_space @L1, interconnect : [@vertical_links], broadcast : [8, 1] : memref<64x64xf16, strided<[4096, 1], offset: ?>>, memref<64x64xf16>
              loom.matmul ins(%19, %17 : memref<64x64xf16>, memref<64x64xf16>) outs(%21 : memref<64x64xf16>)
              loom.semaphore_give %17 : memref<64x64xf16>
              loom.semaphore_give %19 : memref<64x64xf16>
            }
            %22 = arith.muli %15, %c64 : index
            %23 = arith.muli %13, %c262144 : index
            %24 = arith.addi %23, %22 : index
            %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%24], sizes: [64, 64], strides: [4096, 1] : memref<4096x4096xf16> to memref<64x64xf16, strided<[4096, 1], offset: ?>>
            loom.copy %21, %reinterpret_cast src_mem_space @L1 dst_mem_space @DRAM, interconnect : [], broadcast : [1, 1] : memref<64x64xf16>, memref<64x64xf16, strided<[4096, 1], offset: ?>>
            loom.semaphore_give %21 : memref<64x64xf16>
          }
        }
        scf.reduce 
      } {loom.iter_types = [#loom.iter_type<spatial>, #loom.iter_type<spatial>], loom.mapped_to_dims = [@y, @x]}
      return
    }
  }
  module attributes {loom.pass_name = "Materialize"} {
    func.func @matmul__d0i1_d1i1__f01__d_d__BK64__BM64__BN64(%arg0: memref<4096x512xf16>, %arg1: memref<512x4096xf16>, %arg2: memref<4096x4096xf16>) {
      %c262144 = arith.constant 262144 : index
      %c32768 = arith.constant 32768 : index
      %c64 = arith.constant 64 : index
      %c0 = arith.constant 0 : index
      %c8 = arith.constant 8 : index
      %c1 = arith.constant 1 : index
      scf.parallel (%arg3, %arg4) = (%c0, %c0) to (%c8, %c8) step (%c1, %c1) {
        scf.for %arg5 = %c0 to %c64 step %c1 {
          %12 = arith.muli %arg3, %c8 overflow<nsw> : index
          %13 = arith.addi %12, %arg4 : index
          %14 = loom.alloc [64, 64] on @L1 : memref<64x64xf16>
          %15 = loom.semaphore_take %14 : memref<64x64xf16> -> memref<64x64xf16>
          %16 = loom.alloc [64, 64] on @L1 : memref<64x64xf16>
          %17 = loom.semaphore_take %16 : memref<64x64xf16> -> memref<64x64xf16>
          %18 = loom.alloc [64, 64] on @L1 : memref<64x64xf16>
          %19 = loom.semaphore_take %18 : memref<64x64xf16> -> memref<64x64xf16>
          scf.for %arg6 = %c0 to %c8 step %c1 {
            %23 = arith.muli %arg6, %c64 : index
            %24 = arith.muli %arg5, %c32768 : index
            %25 = arith.addi %24, %23 : index
            %reinterpret_cast_0 = memref.reinterpret_cast %arg0 to offset: [%25], sizes: [64, 64], strides: [512, 1] : memref<4096x512xf16> to memref<64x64xf16, strided<[512, 1], offset: ?>>
            loom.copy %reinterpret_cast_0, %17 src_mem_space @DRAM dst_mem_space @L1, interconnect : [], broadcast : [1, 1] : memref<64x64xf16, strided<[512, 1], offset: ?>>, memref<64x64xf16>
            %26 = arith.muli %13, %c64 : index
            %27 = arith.muli %arg6, %c262144 : index
            %28 = arith.addi %27, %26 : index
            %reinterpret_cast_1 = memref.reinterpret_cast %arg1 to offset: [%28], sizes: [64, 64], strides: [4096, 1] : memref<512x4096xf16> to memref<64x64xf16, strided<[4096, 1], offset: ?>>
            loom.copy %reinterpret_cast_1, %15 src_mem_space @DRAM dst_mem_space @L1, interconnect : [], broadcast : [1, 1] : memref<64x64xf16, strided<[4096, 1], offset: ?>>, memref<64x64xf16>
            loom.matmul ins(%17, %15 : memref<64x64xf16>, memref<64x64xf16>) outs(%19 : memref<64x64xf16>)
            loom.semaphore_give %15 : memref<64x64xf16>
            loom.semaphore_give %17 : memref<64x64xf16>
          }
          %20 = arith.muli %13, %c64 : index
          %21 = arith.muli %arg5, %c262144 : index
          %22 = arith.addi %21, %20 : index
          %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%22], sizes: [64, 64], strides: [4096, 1] : memref<4096x4096xf16> to memref<64x64xf16, strided<[4096, 1], offset: ?>>
          loom.copy %19, %reinterpret_cast src_mem_space @L1 dst_mem_space @DRAM, interconnect : [], broadcast : [1, 1] : memref<64x64xf16>, memref<64x64xf16, strided<[4096, 1], offset: ?>>
          loom.semaphore_give %19 : memref<64x64xf16>
        }
        scf.reduce 
      } {loom.iter_types = [#loom.iter_type<spatial>, #loom.iter_type<spatial>], loom.mapped_to_dims = [@x, @y]}
      return
    }
  }
  module attributes {loom.pass_name = "Materialize"} {
    func.func @matmul__d0i1_d1i1__f01__a_d__BK64__BM64__BN64(%arg0: memref<4096x512xf16>, %arg1: memref<512x4096xf16>, %arg2: memref<4096x4096xf16>) {
      %c262144 = arith.constant 262144 : index
      %c32768 = arith.constant 32768 : index
      %c64 = arith.constant 64 : index
      %c0 = arith.constant 0 : index
      %c8 = arith.constant 8 : index
      %c1 = arith.constant 1 : index
      scf.parallel (%arg3, %arg4) = (%c0, %c0) to (%c8, %c8) step (%c1, %c1) {
        scf.for %arg5 = %c0 to %c64 step %c1 {
          %12 = arith.muli %arg3, %c8 overflow<nsw> : index
          %13 = arith.addi %12, %arg4 : index
          %14 = loom.alloc [64, 64] on @L1 : memref<64x64xf16>
          %15 = loom.semaphore_take %14 : memref<64x64xf16> -> memref<64x64xf16>
          %16 = loom.alloc [64, 64] on @L1 : memref<64x64xf16>
          %17 = loom.semaphore_take %16 : memref<64x64xf16> -> memref<64x64xf16>
          %18 = loom.alloc [64, 64] on @L1 : memref<64x64xf16>
          %19 = loom.semaphore_take %18 : memref<64x64xf16> -> memref<64x64xf16>
          scf.for %arg6 = %c0 to %c8 step %c1 {
            %23 = arith.muli %arg6, %c64 : index
            %24 = arith.muli %arg5, %c32768 : index
            %25 = arith.addi %24, %23 : index
            %reinterpret_cast_0 = memref.reinterpret_cast %arg0 to offset: [%25], sizes: [64, 64], strides: [512, 1] : memref<4096x512xf16> to memref<64x64xf16, strided<[512, 1], offset: ?>>
            loom.copy %reinterpret_cast_0, %17 src_mem_space @DRAM dst_mem_space @L1, interconnect : [@horizontal_links, @vertical_links], broadcast : [8, 8] : memref<64x64xf16, strided<[512, 1], offset: ?>>, memref<64x64xf16>
            %26 = arith.muli %13, %c64 : index
            %27 = arith.muli %arg6, %c262144 : index
            %28 = arith.addi %27, %26 : index
            %reinterpret_cast_1 = memref.reinterpret_cast %arg1 to offset: [%28], sizes: [64, 64], strides: [4096, 1] : memref<512x4096xf16> to memref<64x64xf16, strided<[4096, 1], offset: ?>>
            loom.copy %reinterpret_cast_1, %15 src_mem_space @DRAM dst_mem_space @L1, interconnect : [], broadcast : [1, 1] : memref<64x64xf16, strided<[4096, 1], offset: ?>>, memref<64x64xf16>
            loom.matmul ins(%17, %15 : memref<64x64xf16>, memref<64x64xf16>) outs(%19 : memref<64x64xf16>)
            loom.semaphore_give %15 : memref<64x64xf16>
            loom.semaphore_give %17 : memref<64x64xf16>
          }
          %20 = arith.muli %13, %c64 : index
          %21 = arith.muli %arg5, %c262144 : index
          %22 = arith.addi %21, %20 : index
          %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%22], sizes: [64, 64], strides: [4096, 1] : memref<4096x4096xf16> to memref<64x64xf16, strided<[4096, 1], offset: ?>>
          loom.copy %19, %reinterpret_cast src_mem_space @L1 dst_mem_space @DRAM, interconnect : [], broadcast : [1, 1] : memref<64x64xf16>, memref<64x64xf16, strided<[4096, 1], offset: ?>>
          loom.semaphore_give %19 : memref<64x64xf16>
        }
        scf.reduce 
      } {loom.iter_types = [#loom.iter_type<spatial>, #loom.iter_type<spatial>], loom.mapped_to_dims = [@x, @y]}
      return
    }
  }
  module attributes {loom.pass_name = "Materialize"} {
    func.func @matmul__d0i1_d1i1__f01__h_d__BK64__BM64__BN64(%arg0: memref<4096x512xf16>, %arg1: memref<512x4096xf16>, %arg2: memref<4096x4096xf16>) {
      %c262144 = arith.constant 262144 : index
      %c32768 = arith.constant 32768 : index
      %c64 = arith.constant 64 : index
      %c0 = arith.constant 0 : index
      %c8 = arith.constant 8 : index
      %c1 = arith.constant 1 : index
      scf.parallel (%arg3, %arg4) = (%c0, %c0) to (%c8, %c8) step (%c1, %c1) {
        scf.for %arg5 = %c0 to %c64 step %c1 {
          %12 = arith.muli %arg3, %c8 overflow<nsw> : index
          %13 = arith.addi %12, %arg4 : index
          %14 = loom.alloc [64, 64] on @L1 : memref<64x64xf16>
          %15 = loom.semaphore_take %14 : memref<64x64xf16> -> memref<64x64xf16>
          %16 = loom.alloc [64, 64] on @L1 : memref<64x64xf16>
          %17 = loom.semaphore_take %16 : memref<64x64xf16> -> memref<64x64xf16>
          %18 = loom.alloc [64, 64] on @L1 : memref<64x64xf16>
          %19 = loom.semaphore_take %18 : memref<64x64xf16> -> memref<64x64xf16>
          scf.for %arg6 = %c0 to %c8 step %c1 {
            %23 = arith.muli %arg6, %c64 : index
            %24 = arith.muli %arg5, %c32768 : index
            %25 = arith.addi %24, %23 : index
            %reinterpret_cast_0 = memref.reinterpret_cast %arg0 to offset: [%25], sizes: [64, 64], strides: [512, 1] : memref<4096x512xf16> to memref<64x64xf16, strided<[512, 1], offset: ?>>
            loom.copy %reinterpret_cast_0, %17 src_mem_space @DRAM dst_mem_space @L1, interconnect : [@horizontal_links], broadcast : [1, 8] : memref<64x64xf16, strided<[512, 1], offset: ?>>, memref<64x64xf16>
            %26 = arith.muli %13, %c64 : index
            %27 = arith.muli %arg6, %c262144 : index
            %28 = arith.addi %27, %26 : index
            %reinterpret_cast_1 = memref.reinterpret_cast %arg1 to offset: [%28], sizes: [64, 64], strides: [4096, 1] : memref<512x4096xf16> to memref<64x64xf16, strided<[4096, 1], offset: ?>>
            loom.copy %reinterpret_cast_1, %15 src_mem_space @DRAM dst_mem_space @L1, interconnect : [], broadcast : [1, 1] : memref<64x64xf16, strided<[4096, 1], offset: ?>>, memref<64x64xf16>
            loom.matmul ins(%17, %15 : memref<64x64xf16>, memref<64x64xf16>) outs(%19 : memref<64x64xf16>)
            loom.semaphore_give %15 : memref<64x64xf16>
            loom.semaphore_give %17 : memref<64x64xf16>
          }
          %20 = arith.muli %13, %c64 : index
          %21 = arith.muli %arg5, %c262144 : index
          %22 = arith.addi %21, %20 : index
          %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%22], sizes: [64, 64], strides: [4096, 1] : memref<4096x4096xf16> to memref<64x64xf16, strided<[4096, 1], offset: ?>>
          loom.copy %19, %reinterpret_cast src_mem_space @L1 dst_mem_space @DRAM, interconnect : [], broadcast : [1, 1] : memref<64x64xf16>, memref<64x64xf16, strided<[4096, 1], offset: ?>>
          loom.semaphore_give %19 : memref<64x64xf16>
        }
        scf.reduce 
      } {loom.iter_types = [#loom.iter_type<spatial>, #loom.iter_type<spatial>], loom.mapped_to_dims = [@x, @y]}
      return
    }
  }
  module attributes {loom.pass_name = "Materialize"} {
    func.func @matmul__d0i1_d1i1__f01__v_d__BK64__BM64__BN64(%arg0: memref<4096x512xf16>, %arg1: memref<512x4096xf16>, %arg2: memref<4096x4096xf16>) {
      %c262144 = arith.constant 262144 : index
      %c32768 = arith.constant 32768 : index
      %c64 = arith.constant 64 : index
      %c0 = arith.constant 0 : index
      %c8 = arith.constant 8 : index
      %c1 = arith.constant 1 : index
      scf.parallel (%arg3, %arg4) = (%c0, %c0) to (%c8, %c8) step (%c1, %c1) {
        scf.for %arg5 = %c0 to %c64 step %c1 {
          %12 = arith.muli %arg3, %c8 overflow<nsw> : index
          %13 = arith.addi %12, %arg4 : index
          %14 = loom.alloc [64, 64] on @L1 : memref<64x64xf16>
          %15 = loom.semaphore_take %14 : memref<64x64xf16> -> memref<64x64xf16>
          %16 = loom.alloc [64, 64] on @L1 : memref<64x64xf16>
          %17 = loom.semaphore_take %16 : memref<64x64xf16> -> memref<64x64xf16>
          %18 = loom.alloc [64, 64] on @L1 : memref<64x64xf16>
          %19 = loom.semaphore_take %18 : memref<64x64xf16> -> memref<64x64xf16>
          scf.for %arg6 = %c0 to %c8 step %c1 {
            %23 = arith.muli %arg6, %c64 : index
            %24 = arith.muli %arg5, %c32768 : index
            %25 = arith.addi %24, %23 : index
            %reinterpret_cast_0 = memref.reinterpret_cast %arg0 to offset: [%25], sizes: [64, 64], strides: [512, 1] : memref<4096x512xf16> to memref<64x64xf16, strided<[512, 1], offset: ?>>
            loom.copy %reinterpret_cast_0, %17 src_mem_space @DRAM dst_mem_space @L1, interconnect : [@vertical_links], broadcast : [8, 1] : memref<64x64xf16, strided<[512, 1], offset: ?>>, memref<64x64xf16>
            %26 = arith.muli %13, %c64 : index
            %27 = arith.muli %arg6, %c262144 : index
            %28 = arith.addi %27, %26 : index
            %reinterpret_cast_1 = memref.reinterpret_cast %arg1 to offset: [%28], sizes: [64, 64], strides: [4096, 1] : memref<512x4096xf16> to memref<64x64xf16, strided<[4096, 1], offset: ?>>
            loom.copy %reinterpret_cast_1, %15 src_mem_space @DRAM dst_mem_space @L1, interconnect : [], broadcast : [1, 1] : memref<64x64xf16, strided<[4096, 1], offset: ?>>, memref<64x64xf16>
            loom.matmul ins(%17, %15 : memref<64x64xf16>, memref<64x64xf16>) outs(%19 : memref<64x64xf16>)
            loom.semaphore_give %15 : memref<64x64xf16>
            loom.semaphore_give %17 : memref<64x64xf16>
          }
          %20 = arith.muli %13, %c64 : index
          %21 = arith.muli %arg5, %c262144 : index
          %22 = arith.addi %21, %20 : index
          %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%22], sizes: [64, 64], strides: [4096, 1] : memref<4096x4096xf16> to memref<64x64xf16, strided<[4096, 1], offset: ?>>
          loom.copy %19, %reinterpret_cast src_mem_space @L1 dst_mem_space @DRAM, interconnect : [], broadcast : [1, 1] : memref<64x64xf16>, memref<64x64xf16, strided<[4096, 1], offset: ?>>
          loom.semaphore_give %19 : memref<64x64xf16>
        }
        scf.reduce 
      } {loom.iter_types = [#loom.iter_type<spatial>, #loom.iter_type<spatial>], loom.mapped_to_dims = [@x, @y]}
      return
    }
  }
  module attributes {loom.pass_name = "Materialize"} {
    func.func @matmul__d0i1_d1i1__f10__d_d__BK64__BM64__BN64(%arg0: memref<4096x512xf16>, %arg1: memref<512x4096xf16>, %arg2: memref<4096x4096xf16>) {
      %c262144 = arith.constant 262144 : index
      %c32768 = arith.constant 32768 : index
      %c64 = arith.constant 64 : index
      %c0 = arith.constant 0 : index
      %c8 = arith.constant 8 : index
      %c1 = arith.constant 1 : index
      scf.parallel (%arg3, %arg4) = (%c0, %c0) to (%c8, %c8) step (%c1, %c1) {
        scf.for %arg5 = %c0 to %c64 step %c1 {
          %12 = arith.muli %arg3, %c8 overflow<nsw> : index
          %13 = arith.addi %12, %arg4 : index
          %14 = loom.alloc [64, 64] on @L1 : memref<64x64xf16>
          %15 = loom.semaphore_take %14 : memref<64x64xf16> -> memref<64x64xf16>
          %16 = loom.alloc [64, 64] on @L1 : memref<64x64xf16>
          %17 = loom.semaphore_take %16 : memref<64x64xf16> -> memref<64x64xf16>
          %18 = loom.alloc [64, 64] on @L1 : memref<64x64xf16>
          %19 = loom.semaphore_take %18 : memref<64x64xf16> -> memref<64x64xf16>
          scf.for %arg6 = %c0 to %c8 step %c1 {
            %23 = arith.muli %arg6, %c64 : index
            %24 = arith.muli %arg5, %c32768 : index
            %25 = arith.addi %24, %23 : index
            %reinterpret_cast_0 = memref.reinterpret_cast %arg0 to offset: [%25], sizes: [64, 64], strides: [512, 1] : memref<4096x512xf16> to memref<64x64xf16, strided<[512, 1], offset: ?>>
            loom.copy %reinterpret_cast_0, %17 src_mem_space @DRAM dst_mem_space @L1, interconnect : [], broadcast : [1, 1] : memref<64x64xf16, strided<[512, 1], offset: ?>>, memref<64x64xf16>
            %26 = arith.muli %13, %c64 : index
            %27 = arith.muli %arg6, %c262144 : index
            %28 = arith.addi %27, %26 : index
            %reinterpret_cast_1 = memref.reinterpret_cast %arg1 to offset: [%28], sizes: [64, 64], strides: [4096, 1] : memref<512x4096xf16> to memref<64x64xf16, strided<[4096, 1], offset: ?>>
            loom.copy %reinterpret_cast_1, %15 src_mem_space @DRAM dst_mem_space @L1, interconnect : [], broadcast : [1, 1] : memref<64x64xf16, strided<[4096, 1], offset: ?>>, memref<64x64xf16>
            loom.matmul ins(%17, %15 : memref<64x64xf16>, memref<64x64xf16>) outs(%19 : memref<64x64xf16>)
            loom.semaphore_give %15 : memref<64x64xf16>
            loom.semaphore_give %17 : memref<64x64xf16>
          }
          %20 = arith.muli %13, %c64 : index
          %21 = arith.muli %arg5, %c262144 : index
          %22 = arith.addi %21, %20 : index
          %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%22], sizes: [64, 64], strides: [4096, 1] : memref<4096x4096xf16> to memref<64x64xf16, strided<[4096, 1], offset: ?>>
          loom.copy %19, %reinterpret_cast src_mem_space @L1 dst_mem_space @DRAM, interconnect : [], broadcast : [1, 1] : memref<64x64xf16>, memref<64x64xf16, strided<[4096, 1], offset: ?>>
          loom.semaphore_give %19 : memref<64x64xf16>
        }
        scf.reduce 
      } {loom.iter_types = [#loom.iter_type<spatial>, #loom.iter_type<spatial>], loom.mapped_to_dims = [@x, @y]}
      return
    }
  }
  module attributes {loom.pass_name = "Materialize"} {
    func.func @matmul__d0i1_d1i1__f10__a_d__BK64__BM64__BN64(%arg0: memref<4096x512xf16>, %arg1: memref<512x4096xf16>, %arg2: memref<4096x4096xf16>) {
      %c262144 = arith.constant 262144 : index
      %c32768 = arith.constant 32768 : index
      %c64 = arith.constant 64 : index
      %c0 = arith.constant 0 : index
      %c8 = arith.constant 8 : index
      %c1 = arith.constant 1 : index
      scf.parallel (%arg3, %arg4) = (%c0, %c0) to (%c8, %c8) step (%c1, %c1) {
        scf.for %arg5 = %c0 to %c64 step %c1 {
          %12 = arith.muli %arg3, %c8 overflow<nsw> : index
          %13 = arith.addi %12, %arg4 : index
          %14 = loom.alloc [64, 64] on @L1 : memref<64x64xf16>
          %15 = loom.semaphore_take %14 : memref<64x64xf16> -> memref<64x64xf16>
          %16 = loom.alloc [64, 64] on @L1 : memref<64x64xf16>
          %17 = loom.semaphore_take %16 : memref<64x64xf16> -> memref<64x64xf16>
          %18 = loom.alloc [64, 64] on @L1 : memref<64x64xf16>
          %19 = loom.semaphore_take %18 : memref<64x64xf16> -> memref<64x64xf16>
          scf.for %arg6 = %c0 to %c8 step %c1 {
            %23 = arith.muli %arg6, %c64 : index
            %24 = arith.muli %arg5, %c32768 : index
            %25 = arith.addi %24, %23 : index
            %reinterpret_cast_0 = memref.reinterpret_cast %arg0 to offset: [%25], sizes: [64, 64], strides: [512, 1] : memref<4096x512xf16> to memref<64x64xf16, strided<[512, 1], offset: ?>>
            loom.copy %reinterpret_cast_0, %17 src_mem_space @DRAM dst_mem_space @L1, interconnect : [@horizontal_links, @vertical_links], broadcast : [8, 8] : memref<64x64xf16, strided<[512, 1], offset: ?>>, memref<64x64xf16>
            %26 = arith.muli %13, %c64 : index
            %27 = arith.muli %arg6, %c262144 : index
            %28 = arith.addi %27, %26 : index
            %reinterpret_cast_1 = memref.reinterpret_cast %arg1 to offset: [%28], sizes: [64, 64], strides: [4096, 1] : memref<512x4096xf16> to memref<64x64xf16, strided<[4096, 1], offset: ?>>
            loom.copy %reinterpret_cast_1, %15 src_mem_space @DRAM dst_mem_space @L1, interconnect : [], broadcast : [1, 1] : memref<64x64xf16, strided<[4096, 1], offset: ?>>, memref<64x64xf16>
            loom.matmul ins(%17, %15 : memref<64x64xf16>, memref<64x64xf16>) outs(%19 : memref<64x64xf16>)
            loom.semaphore_give %15 : memref<64x64xf16>
            loom.semaphore_give %17 : memref<64x64xf16>
          }
          %20 = arith.muli %13, %c64 : index
          %21 = arith.muli %arg5, %c262144 : index
          %22 = arith.addi %21, %20 : index
          %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%22], sizes: [64, 64], strides: [4096, 1] : memref<4096x4096xf16> to memref<64x64xf16, strided<[4096, 1], offset: ?>>
          loom.copy %19, %reinterpret_cast src_mem_space @L1 dst_mem_space @DRAM, interconnect : [], broadcast : [1, 1] : memref<64x64xf16>, memref<64x64xf16, strided<[4096, 1], offset: ?>>
          loom.semaphore_give %19 : memref<64x64xf16>
        }
        scf.reduce 
      } {loom.iter_types = [#loom.iter_type<spatial>, #loom.iter_type<spatial>], loom.mapped_to_dims = [@x, @y]}
      return
    }
  }
  module attributes {loom.pass_name = "Materialize"} {
    func.func @matmul__d0i1_d1i1__f10__h_d__BK64__BM64__BN64(%arg0: memref<4096x512xf16>, %arg1: memref<512x4096xf16>, %arg2: memref<4096x4096xf16>) {
      %c262144 = arith.constant 262144 : index
      %c32768 = arith.constant 32768 : index
      %c64 = arith.constant 64 : index
      %c0 = arith.constant 0 : index
      %c8 = arith.constant 8 : index
      %c1 = arith.constant 1 : index
      scf.parallel (%arg3, %arg4) = (%c0, %c0) to (%c8, %c8) step (%c1, %c1) {
        scf.for %arg5 = %c0 to %c64 step %c1 {
          %12 = arith.muli %arg3, %c8 overflow<nsw> : index
          %13 = arith.addi %12, %arg4 : index
          %14 = loom.alloc [64, 64] on @L1 : memref<64x64xf16>
          %15 = loom.semaphore_take %14 : memref<64x64xf16> -> memref<64x64xf16>
          %16 = loom.alloc [64, 64] on @L1 : memref<64x64xf16>
          %17 = loom.semaphore_take %16 : memref<64x64xf16> -> memref<64x64xf16>
          %18 = loom.alloc [64, 64] on @L1 : memref<64x64xf16>
          %19 = loom.semaphore_take %18 : memref<64x64xf16> -> memref<64x64xf16>
          scf.for %arg6 = %c0 to %c8 step %c1 {
            %23 = arith.muli %arg6, %c64 : index
            %24 = arith.muli %arg5, %c32768 : index
            %25 = arith.addi %24, %23 : index
            %reinterpret_cast_0 = memref.reinterpret_cast %arg0 to offset: [%25], sizes: [64, 64], strides: [512, 1] : memref<4096x512xf16> to memref<64x64xf16, strided<[512, 1], offset: ?>>
            loom.copy %reinterpret_cast_0, %17 src_mem_space @DRAM dst_mem_space @L1, interconnect : [@horizontal_links], broadcast : [1, 8] : memref<64x64xf16, strided<[512, 1], offset: ?>>, memref<64x64xf16>
            %26 = arith.muli %13, %c64 : index
            %27 = arith.muli %arg6, %c262144 : index
            %28 = arith.addi %27, %26 : index
            %reinterpret_cast_1 = memref.reinterpret_cast %arg1 to offset: [%28], sizes: [64, 64], strides: [4096, 1] : memref<512x4096xf16> to memref<64x64xf16, strided<[4096, 1], offset: ?>>
            loom.copy %reinterpret_cast_1, %15 src_mem_space @DRAM dst_mem_space @L1, interconnect : [], broadcast : [1, 1] : memref<64x64xf16, strided<[4096, 1], offset: ?>>, memref<64x64xf16>
            loom.matmul ins(%17, %15 : memref<64x64xf16>, memref<64x64xf16>) outs(%19 : memref<64x64xf16>)
            loom.semaphore_give %15 : memref<64x64xf16>
            loom.semaphore_give %17 : memref<64x64xf16>
          }
          %20 = arith.muli %13, %c64 : index
          %21 = arith.muli %arg5, %c262144 : index
          %22 = arith.addi %21, %20 : index
          %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%22], sizes: [64, 64], strides: [4096, 1] : memref<4096x4096xf16> to memref<64x64xf16, strided<[4096, 1], offset: ?>>
          loom.copy %19, %reinterpret_cast src_mem_space @L1 dst_mem_space @DRAM, interconnect : [], broadcast : [1, 1] : memref<64x64xf16>, memref<64x64xf16, strided<[4096, 1], offset: ?>>
          loom.semaphore_give %19 : memref<64x64xf16>
        }
        scf.reduce 
      } {loom.iter_types = [#loom.iter_type<spatial>, #loom.iter_type<spatial>], loom.mapped_to_dims = [@x, @y]}
      return
    }
  }
  module attributes {loom.pass_name = "Materialize"} {
    func.func @matmul__d0i1_d1i1__f10__v_d__BK64__BM64__BN64(%arg0: memref<4096x512xf16>, %arg1: memref<512x4096xf16>, %arg2: memref<4096x4096xf16>) {
      %c262144 = arith.constant 262144 : index
      %c32768 = arith.constant 32768 : index
      %c64 = arith.constant 64 : index
      %c0 = arith.constant 0 : index
      %c8 = arith.constant 8 : index
      %c1 = arith.constant 1 : index
      scf.parallel (%arg3, %arg4) = (%c0, %c0) to (%c8, %c8) step (%c1, %c1) {
        scf.for %arg5 = %c0 to %c64 step %c1 {
          %12 = arith.muli %arg3, %c8 overflow<nsw> : index
          %13 = arith.addi %12, %arg4 : index
          %14 = loom.alloc [64, 64] on @L1 : memref<64x64xf16>
          %15 = loom.semaphore_take %14 : memref<64x64xf16> -> memref<64x64xf16>
          %16 = loom.alloc [64, 64] on @L1 : memref<64x64xf16>
          %17 = loom.semaphore_take %16 : memref<64x64xf16> -> memref<64x64xf16>
          %18 = loom.alloc [64, 64] on @L1 : memref<64x64xf16>
          %19 = loom.semaphore_take %18 : memref<64x64xf16> -> memref<64x64xf16>
          scf.for %arg6 = %c0 to %c8 step %c1 {
            %23 = arith.muli %arg6, %c64 : index
            %24 = arith.muli %arg5, %c32768 : index
            %25 = arith.addi %24, %23 : index
            %reinterpret_cast_0 = memref.reinterpret_cast %arg0 to offset: [%25], sizes: [64, 64], strides: [512, 1] : memref<4096x512xf16> to memref<64x64xf16, strided<[512, 1], offset: ?>>
            loom.copy %reinterpret_cast_0, %17 src_mem_space @DRAM dst_mem_space @L1, interconnect : [@vertical_links], broadcast : [8, 1] : memref<64x64xf16, strided<[512, 1], offset: ?>>, memref<64x64xf16>
            %26 = arith.muli %13, %c64 : index
            %27 = arith.muli %arg6, %c262144 : index
            %28 = arith.addi %27, %26 : index
            %reinterpret_cast_1 = memref.reinterpret_cast %arg1 to offset: [%28], sizes: [64, 64], strides: [4096, 1] : memref<512x4096xf16> to memref<64x64xf16, strided<[4096, 1], offset: ?>>
            loom.copy %reinterpret_cast_1, %15 src_mem_space @DRAM dst_mem_space @L1, interconnect : [], broadcast : [1, 1] : memref<64x64xf16, strided<[4096, 1], offset: ?>>, memref<64x64xf16>
            loom.matmul ins(%17, %15 : memref<64x64xf16>, memref<64x64xf16>) outs(%19 : memref<64x64xf16>)
            loom.semaphore_give %15 : memref<64x64xf16>
            loom.semaphore_give %17 : memref<64x64xf16>
          }
          %20 = arith.muli %13, %c64 : index
          %21 = arith.muli %arg5, %c262144 : index
          %22 = arith.addi %21, %20 : index
          %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%22], sizes: [64, 64], strides: [4096, 1] : memref<4096x4096xf16> to memref<64x64xf16, strided<[4096, 1], offset: ?>>
          loom.copy %19, %reinterpret_cast src_mem_space @L1 dst_mem_space @DRAM, interconnect : [], broadcast : [1, 1] : memref<64x64xf16>, memref<64x64xf16, strided<[4096, 1], offset: ?>>
          loom.semaphore_give %19 : memref<64x64xf16>
        }
        scf.reduce 
      } {loom.iter_types = [#loom.iter_type<spatial>, #loom.iter_type<spatial>], loom.mapped_to_dims = [@x, @y]}
      return
    }
  }
  module attributes {loom.pass_name = "Materialize"} {
    func.func @matmul__d1i1_d0i1__f01__d_d__BK64__BM64__BN64(%arg0: memref<4096x512xf16>, %arg1: memref<512x4096xf16>, %arg2: memref<4096x4096xf16>) {
      %c262144 = arith.constant 262144 : index
      %c32768 = arith.constant 32768 : index
      %c64 = arith.constant 64 : index
      %c0 = arith.constant 0 : index
      %c8 = arith.constant 8 : index
      %c1 = arith.constant 1 : index
      scf.parallel (%arg3, %arg4) = (%c0, %c0) to (%c8, %c8) step (%c1, %c1) {
        scf.for %arg5 = %c0 to %c64 step %c1 {
          %12 = arith.muli %arg3, %c8 overflow<nsw> : index
          %13 = arith.addi %12, %arg4 : index
          %14 = loom.alloc [64, 64] on @L1 : memref<64x64xf16>
          %15 = loom.semaphore_take %14 : memref<64x64xf16> -> memref<64x64xf16>
          %16 = loom.alloc [64, 64] on @L1 : memref<64x64xf16>
          %17 = loom.semaphore_take %16 : memref<64x64xf16> -> memref<64x64xf16>
          %18 = loom.alloc [64, 64] on @L1 : memref<64x64xf16>
          %19 = loom.semaphore_take %18 : memref<64x64xf16> -> memref<64x64xf16>
          scf.for %arg6 = %c0 to %c8 step %c1 {
            %23 = arith.muli %arg6, %c64 : index
            %24 = arith.muli %arg5, %c32768 : index
            %25 = arith.addi %24, %23 : index
            %reinterpret_cast_0 = memref.reinterpret_cast %arg0 to offset: [%25], sizes: [64, 64], strides: [512, 1] : memref<4096x512xf16> to memref<64x64xf16, strided<[512, 1], offset: ?>>
            loom.copy %reinterpret_cast_0, %17 src_mem_space @DRAM dst_mem_space @L1, interconnect : [], broadcast : [1, 1] : memref<64x64xf16, strided<[512, 1], offset: ?>>, memref<64x64xf16>
            %26 = arith.muli %13, %c64 : index
            %27 = arith.muli %arg6, %c262144 : index
            %28 = arith.addi %27, %26 : index
            %reinterpret_cast_1 = memref.reinterpret_cast %arg1 to offset: [%28], sizes: [64, 64], strides: [4096, 1] : memref<512x4096xf16> to memref<64x64xf16, strided<[4096, 1], offset: ?>>
            loom.copy %reinterpret_cast_1, %15 src_mem_space @DRAM dst_mem_space @L1, interconnect : [], broadcast : [1, 1] : memref<64x64xf16, strided<[4096, 1], offset: ?>>, memref<64x64xf16>
            loom.matmul ins(%17, %15 : memref<64x64xf16>, memref<64x64xf16>) outs(%19 : memref<64x64xf16>)
            loom.semaphore_give %15 : memref<64x64xf16>
            loom.semaphore_give %17 : memref<64x64xf16>
          }
          %20 = arith.muli %13, %c64 : index
          %21 = arith.muli %arg5, %c262144 : index
          %22 = arith.addi %21, %20 : index
          %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%22], sizes: [64, 64], strides: [4096, 1] : memref<4096x4096xf16> to memref<64x64xf16, strided<[4096, 1], offset: ?>>
          loom.copy %19, %reinterpret_cast src_mem_space @L1 dst_mem_space @DRAM, interconnect : [], broadcast : [1, 1] : memref<64x64xf16>, memref<64x64xf16, strided<[4096, 1], offset: ?>>
          loom.semaphore_give %19 : memref<64x64xf16>
        }
        scf.reduce 
      } {loom.iter_types = [#loom.iter_type<spatial>, #loom.iter_type<spatial>], loom.mapped_to_dims = [@y, @x]}
      return
    }
  }
  module attributes {loom.pass_name = "Materialize"} {
    func.func @matmul__d1i1_d0i1__f01__a_d__BK64__BM64__BN64(%arg0: memref<4096x512xf16>, %arg1: memref<512x4096xf16>, %arg2: memref<4096x4096xf16>) {
      %c262144 = arith.constant 262144 : index
      %c32768 = arith.constant 32768 : index
      %c64 = arith.constant 64 : index
      %c0 = arith.constant 0 : index
      %c8 = arith.constant 8 : index
      %c1 = arith.constant 1 : index
      scf.parallel (%arg3, %arg4) = (%c0, %c0) to (%c8, %c8) step (%c1, %c1) {
        scf.for %arg5 = %c0 to %c64 step %c1 {
          %12 = arith.muli %arg3, %c8 overflow<nsw> : index
          %13 = arith.addi %12, %arg4 : index
          %14 = loom.alloc [64, 64] on @L1 : memref<64x64xf16>
          %15 = loom.semaphore_take %14 : memref<64x64xf16> -> memref<64x64xf16>
          %16 = loom.alloc [64, 64] on @L1 : memref<64x64xf16>
          %17 = loom.semaphore_take %16 : memref<64x64xf16> -> memref<64x64xf16>
          %18 = loom.alloc [64, 64] on @L1 : memref<64x64xf16>
          %19 = loom.semaphore_take %18 : memref<64x64xf16> -> memref<64x64xf16>
          scf.for %arg6 = %c0 to %c8 step %c1 {
            %23 = arith.muli %arg6, %c64 : index
            %24 = arith.muli %arg5, %c32768 : index
            %25 = arith.addi %24, %23 : index
            %reinterpret_cast_0 = memref.reinterpret_cast %arg0 to offset: [%25], sizes: [64, 64], strides: [512, 1] : memref<4096x512xf16> to memref<64x64xf16, strided<[512, 1], offset: ?>>
            loom.copy %reinterpret_cast_0, %17 src_mem_space @DRAM dst_mem_space @L1, interconnect : [@horizontal_links, @vertical_links], broadcast : [8, 8] : memref<64x64xf16, strided<[512, 1], offset: ?>>, memref<64x64xf16>
            %26 = arith.muli %13, %c64 : index
            %27 = arith.muli %arg6, %c262144 : index
            %28 = arith.addi %27, %26 : index
            %reinterpret_cast_1 = memref.reinterpret_cast %arg1 to offset: [%28], sizes: [64, 64], strides: [4096, 1] : memref<512x4096xf16> to memref<64x64xf16, strided<[4096, 1], offset: ?>>
            loom.copy %reinterpret_cast_1, %15 src_mem_space @DRAM dst_mem_space @L1, interconnect : [], broadcast : [1, 1] : memref<64x64xf16, strided<[4096, 1], offset: ?>>, memref<64x64xf16>
            loom.matmul ins(%17, %15 : memref<64x64xf16>, memref<64x64xf16>) outs(%19 : memref<64x64xf16>)
            loom.semaphore_give %15 : memref<64x64xf16>
            loom.semaphore_give %17 : memref<64x64xf16>
          }
          %20 = arith.muli %13, %c64 : index
          %21 = arith.muli %arg5, %c262144 : index
          %22 = arith.addi %21, %20 : index
          %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%22], sizes: [64, 64], strides: [4096, 1] : memref<4096x4096xf16> to memref<64x64xf16, strided<[4096, 1], offset: ?>>
          loom.copy %19, %reinterpret_cast src_mem_space @L1 dst_mem_space @DRAM, interconnect : [], broadcast : [1, 1] : memref<64x64xf16>, memref<64x64xf16, strided<[4096, 1], offset: ?>>
          loom.semaphore_give %19 : memref<64x64xf16>
        }
        scf.reduce 
      } {loom.iter_types = [#loom.iter_type<spatial>, #loom.iter_type<spatial>], loom.mapped_to_dims = [@y, @x]}
      return
    }
  }
  module attributes {loom.pass_name = "Materialize"} {
    func.func @matmul__d1i1_d0i1__f01__h_d__BK64__BM64__BN64(%arg0: memref<4096x512xf16>, %arg1: memref<512x4096xf16>, %arg2: memref<4096x4096xf16>) {
      %c262144 = arith.constant 262144 : index
      %c32768 = arith.constant 32768 : index
      %c64 = arith.constant 64 : index
      %c0 = arith.constant 0 : index
      %c8 = arith.constant 8 : index
      %c1 = arith.constant 1 : index
      scf.parallel (%arg3, %arg4) = (%c0, %c0) to (%c8, %c8) step (%c1, %c1) {
        scf.for %arg5 = %c0 to %c64 step %c1 {
          %12 = arith.muli %arg3, %c8 overflow<nsw> : index
          %13 = arith.addi %12, %arg4 : index
          %14 = loom.alloc [64, 64] on @L1 : memref<64x64xf16>
          %15 = loom.semaphore_take %14 : memref<64x64xf16> -> memref<64x64xf16>
          %16 = loom.alloc [64, 64] on @L1 : memref<64x64xf16>
          %17 = loom.semaphore_take %16 : memref<64x64xf16> -> memref<64x64xf16>
          %18 = loom.alloc [64, 64] on @L1 : memref<64x64xf16>
          %19 = loom.semaphore_take %18 : memref<64x64xf16> -> memref<64x64xf16>
          scf.for %arg6 = %c0 to %c8 step %c1 {
            %23 = arith.muli %arg6, %c64 : index
            %24 = arith.muli %arg5, %c32768 : index
            %25 = arith.addi %24, %23 : index
            %reinterpret_cast_0 = memref.reinterpret_cast %arg0 to offset: [%25], sizes: [64, 64], strides: [512, 1] : memref<4096x512xf16> to memref<64x64xf16, strided<[512, 1], offset: ?>>
            loom.copy %reinterpret_cast_0, %17 src_mem_space @DRAM dst_mem_space @L1, interconnect : [@horizontal_links], broadcast : [1, 8] : memref<64x64xf16, strided<[512, 1], offset: ?>>, memref<64x64xf16>
            %26 = arith.muli %13, %c64 : index
            %27 = arith.muli %arg6, %c262144 : index
            %28 = arith.addi %27, %26 : index
            %reinterpret_cast_1 = memref.reinterpret_cast %arg1 to offset: [%28], sizes: [64, 64], strides: [4096, 1] : memref<512x4096xf16> to memref<64x64xf16, strided<[4096, 1], offset: ?>>
            loom.copy %reinterpret_cast_1, %15 src_mem_space @DRAM dst_mem_space @L1, interconnect : [], broadcast : [1, 1] : memref<64x64xf16, strided<[4096, 1], offset: ?>>, memref<64x64xf16>
            loom.matmul ins(%17, %15 : memref<64x64xf16>, memref<64x64xf16>) outs(%19 : memref<64x64xf16>)
            loom.semaphore_give %15 : memref<64x64xf16>
            loom.semaphore_give %17 : memref<64x64xf16>
          }
          %20 = arith.muli %13, %c64 : index
          %21 = arith.muli %arg5, %c262144 : index
          %22 = arith.addi %21, %20 : index
          %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%22], sizes: [64, 64], strides: [4096, 1] : memref<4096x4096xf16> to memref<64x64xf16, strided<[4096, 1], offset: ?>>
          loom.copy %19, %reinterpret_cast src_mem_space @L1 dst_mem_space @DRAM, interconnect : [], broadcast : [1, 1] : memref<64x64xf16>, memref<64x64xf16, strided<[4096, 1], offset: ?>>
          loom.semaphore_give %19 : memref<64x64xf16>
        }
        scf.reduce 
      } {loom.iter_types = [#loom.iter_type<spatial>, #loom.iter_type<spatial>], loom.mapped_to_dims = [@y, @x]}
      return
    }
  }
  module attributes {loom.pass_name = "Materialize"} {
    func.func @matmul__d1i1_d0i1__f01__v_d__BK64__BM64__BN64(%arg0: memref<4096x512xf16>, %arg1: memref<512x4096xf16>, %arg2: memref<4096x4096xf16>) {
      %c262144 = arith.constant 262144 : index
      %c32768 = arith.constant 32768 : index
      %c64 = arith.constant 64 : index
      %c0 = arith.constant 0 : index
      %c8 = arith.constant 8 : index
      %c1 = arith.constant 1 : index
      scf.parallel (%arg3, %arg4) = (%c0, %c0) to (%c8, %c8) step (%c1, %c1) {
        scf.for %arg5 = %c0 to %c64 step %c1 {
          %12 = arith.muli %arg3, %c8 overflow<nsw> : index
          %13 = arith.addi %12, %arg4 : index
          %14 = loom.alloc [64, 64] on @L1 : memref<64x64xf16>
          %15 = loom.semaphore_take %14 : memref<64x64xf16> -> memref<64x64xf16>
          %16 = loom.alloc [64, 64] on @L1 : memref<64x64xf16>
          %17 = loom.semaphore_take %16 : memref<64x64xf16> -> memref<64x64xf16>
          %18 = loom.alloc [64, 64] on @L1 : memref<64x64xf16>
          %19 = loom.semaphore_take %18 : memref<64x64xf16> -> memref<64x64xf16>
          scf.for %arg6 = %c0 to %c8 step %c1 {
            %23 = arith.muli %arg6, %c64 : index
            %24 = arith.muli %arg5, %c32768 : index
            %25 = arith.addi %24, %23 : index
            %reinterpret_cast_0 = memref.reinterpret_cast %arg0 to offset: [%25], sizes: [64, 64], strides: [512, 1] : memref<4096x512xf16> to memref<64x64xf16, strided<[512, 1], offset: ?>>
            loom.copy %reinterpret_cast_0, %17 src_mem_space @DRAM dst_mem_space @L1, interconnect : [@vertical_links], broadcast : [8, 1] : memref<64x64xf16, strided<[512, 1], offset: ?>>, memref<64x64xf16>
            %26 = arith.muli %13, %c64 : index
            %27 = arith.muli %arg6, %c262144 : index
            %28 = arith.addi %27, %26 : index
            %reinterpret_cast_1 = memref.reinterpret_cast %arg1 to offset: [%28], sizes: [64, 64], strides: [4096, 1] : memref<512x4096xf16> to memref<64x64xf16, strided<[4096, 1], offset: ?>>
            loom.copy %reinterpret_cast_1, %15 src_mem_space @DRAM dst_mem_space @L1, interconnect : [], broadcast : [1, 1] : memref<64x64xf16, strided<[4096, 1], offset: ?>>, memref<64x64xf16>
            loom.matmul ins(%17, %15 : memref<64x64xf16>, memref<64x64xf16>) outs(%19 : memref<64x64xf16>)
            loom.semaphore_give %15 : memref<64x64xf16>
            loom.semaphore_give %17 : memref<64x64xf16>
          }
          %20 = arith.muli %13, %c64 : index
          %21 = arith.muli %arg5, %c262144 : index
          %22 = arith.addi %21, %20 : index
          %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%22], sizes: [64, 64], strides: [4096, 1] : memref<4096x4096xf16> to memref<64x64xf16, strided<[4096, 1], offset: ?>>
          loom.copy %19, %reinterpret_cast src_mem_space @L1 dst_mem_space @DRAM, interconnect : [], broadcast : [1, 1] : memref<64x64xf16>, memref<64x64xf16, strided<[4096, 1], offset: ?>>
          loom.semaphore_give %19 : memref<64x64xf16>
        }
        scf.reduce 
      } {loom.iter_types = [#loom.iter_type<spatial>, #loom.iter_type<spatial>], loom.mapped_to_dims = [@y, @x]}
      return
    }
  }
  module attributes {loom.pass_name = "Materialize"} {
    func.func @matmul__d1i1_d0i1__f10__d_d__BK64__BM64__BN64(%arg0: memref<4096x512xf16>, %arg1: memref<512x4096xf16>, %arg2: memref<4096x4096xf16>) {
      %c262144 = arith.constant 262144 : index
      %c32768 = arith.constant 32768 : index
      %c64 = arith.constant 64 : index
      %c0 = arith.constant 0 : index
      %c8 = arith.constant 8 : index
      %c1 = arith.constant 1 : index
      scf.parallel (%arg3, %arg4) = (%c0, %c0) to (%c8, %c8) step (%c1, %c1) {
        scf.for %arg5 = %c0 to %c64 step %c1 {
          %12 = arith.muli %arg3, %c8 overflow<nsw> : index
          %13 = arith.addi %12, %arg4 : index
          %14 = loom.alloc [64, 64] on @L1 : memref<64x64xf16>
          %15 = loom.semaphore_take %14 : memref<64x64xf16> -> memref<64x64xf16>
          %16 = loom.alloc [64, 64] on @L1 : memref<64x64xf16>
          %17 = loom.semaphore_take %16 : memref<64x64xf16> -> memref<64x64xf16>
          %18 = loom.alloc [64, 64] on @L1 : memref<64x64xf16>
          %19 = loom.semaphore_take %18 : memref<64x64xf16> -> memref<64x64xf16>
          scf.for %arg6 = %c0 to %c8 step %c1 {
            %23 = arith.muli %arg6, %c64 : index
            %24 = arith.muli %arg5, %c32768 : index
            %25 = arith.addi %24, %23 : index
            %reinterpret_cast_0 = memref.reinterpret_cast %arg0 to offset: [%25], sizes: [64, 64], strides: [512, 1] : memref<4096x512xf16> to memref<64x64xf16, strided<[512, 1], offset: ?>>
            loom.copy %reinterpret_cast_0, %17 src_mem_space @DRAM dst_mem_space @L1, interconnect : [], broadcast : [1, 1] : memref<64x64xf16, strided<[512, 1], offset: ?>>, memref<64x64xf16>
            %26 = arith.muli %13, %c64 : index
            %27 = arith.muli %arg6, %c262144 : index
            %28 = arith.addi %27, %26 : index
            %reinterpret_cast_1 = memref.reinterpret_cast %arg1 to offset: [%28], sizes: [64, 64], strides: [4096, 1] : memref<512x4096xf16> to memref<64x64xf16, strided<[4096, 1], offset: ?>>
            loom.copy %reinterpret_cast_1, %15 src_mem_space @DRAM dst_mem_space @L1, interconnect : [], broadcast : [1, 1] : memref<64x64xf16, strided<[4096, 1], offset: ?>>, memref<64x64xf16>
            loom.matmul ins(%17, %15 : memref<64x64xf16>, memref<64x64xf16>) outs(%19 : memref<64x64xf16>)
            loom.semaphore_give %15 : memref<64x64xf16>
            loom.semaphore_give %17 : memref<64x64xf16>
          }
          %20 = arith.muli %13, %c64 : index
          %21 = arith.muli %arg5, %c262144 : index
          %22 = arith.addi %21, %20 : index
          %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%22], sizes: [64, 64], strides: [4096, 1] : memref<4096x4096xf16> to memref<64x64xf16, strided<[4096, 1], offset: ?>>
          loom.copy %19, %reinterpret_cast src_mem_space @L1 dst_mem_space @DRAM, interconnect : [], broadcast : [1, 1] : memref<64x64xf16>, memref<64x64xf16, strided<[4096, 1], offset: ?>>
          loom.semaphore_give %19 : memref<64x64xf16>
        }
        scf.reduce 
      } {loom.iter_types = [#loom.iter_type<spatial>, #loom.iter_type<spatial>], loom.mapped_to_dims = [@y, @x]}
      return
    }
  }
  module attributes {loom.pass_name = "Materialize"} {
    func.func @matmul__d1i1_d0i1__f10__a_d__BK64__BM64__BN64(%arg0: memref<4096x512xf16>, %arg1: memref<512x4096xf16>, %arg2: memref<4096x4096xf16>) {
      %c262144 = arith.constant 262144 : index
      %c32768 = arith.constant 32768 : index
      %c64 = arith.constant 64 : index
      %c0 = arith.constant 0 : index
      %c8 = arith.constant 8 : index
      %c1 = arith.constant 1 : index
      scf.parallel (%arg3, %arg4) = (%c0, %c0) to (%c8, %c8) step (%c1, %c1) {
        scf.for %arg5 = %c0 to %c64 step %c1 {
          %12 = arith.muli %arg3, %c8 overflow<nsw> : index
          %13 = arith.addi %12, %arg4 : index
          %14 = loom.alloc [64, 64] on @L1 : memref<64x64xf16>
          %15 = loom.semaphore_take %14 : memref<64x64xf16> -> memref<64x64xf16>
          %16 = loom.alloc [64, 64] on @L1 : memref<64x64xf16>
          %17 = loom.semaphore_take %16 : memref<64x64xf16> -> memref<64x64xf16>
          %18 = loom.alloc [64, 64] on @L1 : memref<64x64xf16>
          %19 = loom.semaphore_take %18 : memref<64x64xf16> -> memref<64x64xf16>
          scf.for %arg6 = %c0 to %c8 step %c1 {
            %23 = arith.muli %arg6, %c64 : index
            %24 = arith.muli %arg5, %c32768 : index
            %25 = arith.addi %24, %23 : index
            %reinterpret_cast_0 = memref.reinterpret_cast %arg0 to offset: [%25], sizes: [64, 64], strides: [512, 1] : memref<4096x512xf16> to memref<64x64xf16, strided<[512, 1], offset: ?>>
            loom.copy %reinterpret_cast_0, %17 src_mem_space @DRAM dst_mem_space @L1, interconnect : [@horizontal_links, @vertical_links], broadcast : [8, 8] : memref<64x64xf16, strided<[512, 1], offset: ?>>, memref<64x64xf16>
            %26 = arith.muli %13, %c64 : index
            %27 = arith.muli %arg6, %c262144 : index
            %28 = arith.addi %27, %26 : index
            %reinterpret_cast_1 = memref.reinterpret_cast %arg1 to offset: [%28], sizes: [64, 64], strides: [4096, 1] : memref<512x4096xf16> to memref<64x64xf16, strided<[4096, 1], offset: ?>>
            loom.copy %reinterpret_cast_1, %15 src_mem_space @DRAM dst_mem_space @L1, interconnect : [], broadcast : [1, 1] : memref<64x64xf16, strided<[4096, 1], offset: ?>>, memref<64x64xf16>
            loom.matmul ins(%17, %15 : memref<64x64xf16>, memref<64x64xf16>) outs(%19 : memref<64x64xf16>)
            loom.semaphore_give %15 : memref<64x64xf16>
            loom.semaphore_give %17 : memref<64x64xf16>
          }
          %20 = arith.muli %13, %c64 : index
          %21 = arith.muli %arg5, %c262144 : index
          %22 = arith.addi %21, %20 : index
          %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%22], sizes: [64, 64], strides: [4096, 1] : memref<4096x4096xf16> to memref<64x64xf16, strided<[4096, 1], offset: ?>>
          loom.copy %19, %reinterpret_cast src_mem_space @L1 dst_mem_space @DRAM, interconnect : [], broadcast : [1, 1] : memref<64x64xf16>, memref<64x64xf16, strided<[4096, 1], offset: ?>>
          loom.semaphore_give %19 : memref<64x64xf16>
        }
        scf.reduce 
      } {loom.iter_types = [#loom.iter_type<spatial>, #loom.iter_type<spatial>], loom.mapped_to_dims = [@y, @x]}
      return
    }
  }
  module attributes {loom.pass_name = "Materialize"} {
    func.func @matmul__d1i1_d0i1__f10__h_d__BK64__BM64__BN64(%arg0: memref<4096x512xf16>, %arg1: memref<512x4096xf16>, %arg2: memref<4096x4096xf16>) {
      %c262144 = arith.constant 262144 : index
      %c32768 = arith.constant 32768 : index
      %c64 = arith.constant 64 : index
      %c0 = arith.constant 0 : index
      %c8 = arith.constant 8 : index
      %c1 = arith.constant 1 : index
      scf.parallel (%arg3, %arg4) = (%c0, %c0) to (%c8, %c8) step (%c1, %c1) {
        scf.for %arg5 = %c0 to %c64 step %c1 {
          %12 = arith.muli %arg3, %c8 overflow<nsw> : index
          %13 = arith.addi %12, %arg4 : index
          %14 = loom.alloc [64, 64] on @L1 : memref<64x64xf16>
          %15 = loom.semaphore_take %14 : memref<64x64xf16> -> memref<64x64xf16>
          %16 = loom.alloc [64, 64] on @L1 : memref<64x64xf16>
          %17 = loom.semaphore_take %16 : memref<64x64xf16> -> memref<64x64xf16>
          %18 = loom.alloc [64, 64] on @L1 : memref<64x64xf16>
          %19 = loom.semaphore_take %18 : memref<64x64xf16> -> memref<64x64xf16>
          scf.for %arg6 = %c0 to %c8 step %c1 {
            %23 = arith.muli %arg6, %c64 : index
            %24 = arith.muli %arg5, %c32768 : index
            %25 = arith.addi %24, %23 : index
            %reinterpret_cast_0 = memref.reinterpret_cast %arg0 to offset: [%25], sizes: [64, 64], strides: [512, 1] : memref<4096x512xf16> to memref<64x64xf16, strided<[512, 1], offset: ?>>
            loom.copy %reinterpret_cast_0, %17 src_mem_space @DRAM dst_mem_space @L1, interconnect : [@horizontal_links], broadcast : [1, 8] : memref<64x64xf16, strided<[512, 1], offset: ?>>, memref<64x64xf16>
            %26 = arith.muli %13, %c64 : index
            %27 = arith.muli %arg6, %c262144 : index
            %28 = arith.addi %27, %26 : index
            %reinterpret_cast_1 = memref.reinterpret_cast %arg1 to offset: [%28], sizes: [64, 64], strides: [4096, 1] : memref<512x4096xf16> to memref<64x64xf16, strided<[4096, 1], offset: ?>>
            loom.copy %reinterpret_cast_1, %15 src_mem_space @DRAM dst_mem_space @L1, interconnect : [], broadcast : [1, 1] : memref<64x64xf16, strided<[4096, 1], offset: ?>>, memref<64x64xf16>
            loom.matmul ins(%17, %15 : memref<64x64xf16>, memref<64x64xf16>) outs(%19 : memref<64x64xf16>)
            loom.semaphore_give %15 : memref<64x64xf16>
            loom.semaphore_give %17 : memref<64x64xf16>
          }
          %20 = arith.muli %13, %c64 : index
          %21 = arith.muli %arg5, %c262144 : index
          %22 = arith.addi %21, %20 : index
          %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%22], sizes: [64, 64], strides: [4096, 1] : memref<4096x4096xf16> to memref<64x64xf16, strided<[4096, 1], offset: ?>>
          loom.copy %19, %reinterpret_cast src_mem_space @L1 dst_mem_space @DRAM, interconnect : [], broadcast : [1, 1] : memref<64x64xf16>, memref<64x64xf16, strided<[4096, 1], offset: ?>>
          loom.semaphore_give %19 : memref<64x64xf16>
        }
        scf.reduce 
      } {loom.iter_types = [#loom.iter_type<spatial>, #loom.iter_type<spatial>], loom.mapped_to_dims = [@y, @x]}
      return
    }
  }
  module attributes {loom.pass_name = "Materialize"} {
    func.func @matmul__d1i1_d0i1__f10__v_d__BK64__BM64__BN64(%arg0: memref<4096x512xf16>, %arg1: memref<512x4096xf16>, %arg2: memref<4096x4096xf16>) {
      %c262144 = arith.constant 262144 : index
      %c32768 = arith.constant 32768 : index
      %c64 = arith.constant 64 : index
      %c0 = arith.constant 0 : index
      %c8 = arith.constant 8 : index
      %c1 = arith.constant 1 : index
      scf.parallel (%arg3, %arg4) = (%c0, %c0) to (%c8, %c8) step (%c1, %c1) {
        scf.for %arg5 = %c0 to %c64 step %c1 {
          %12 = arith.muli %arg3, %c8 overflow<nsw> : index
          %13 = arith.addi %12, %arg4 : index
          %14 = loom.alloc [64, 64] on @L1 : memref<64x64xf16>
          %15 = loom.semaphore_take %14 : memref<64x64xf16> -> memref<64x64xf16>
          %16 = loom.alloc [64, 64] on @L1 : memref<64x64xf16>
          %17 = loom.semaphore_take %16 : memref<64x64xf16> -> memref<64x64xf16>
          %18 = loom.alloc [64, 64] on @L1 : memref<64x64xf16>
          %19 = loom.semaphore_take %18 : memref<64x64xf16> -> memref<64x64xf16>
          scf.for %arg6 = %c0 to %c8 step %c1 {
            %23 = arith.muli %arg6, %c64 : index
            %24 = arith.muli %arg5, %c32768 : index
            %25 = arith.addi %24, %23 : index
            %reinterpret_cast_0 = memref.reinterpret_cast %arg0 to offset: [%25], sizes: [64, 64], strides: [512, 1] : memref<4096x512xf16> to memref<64x64xf16, strided<[512, 1], offset: ?>>
            loom.copy %reinterpret_cast_0, %17 src_mem_space @DRAM dst_mem_space @L1, interconnect : [@vertical_links], broadcast : [8, 1] : memref<64x64xf16, strided<[512, 1], offset: ?>>, memref<64x64xf16>
            %26 = arith.muli %13, %c64 : index
            %27 = arith.muli %arg6, %c262144 : index
            %28 = arith.addi %27, %26 : index
            %reinterpret_cast_1 = memref.reinterpret_cast %arg1 to offset: [%28], sizes: [64, 64], strides: [4096, 1] : memref<512x4096xf16> to memref<64x64xf16, strided<[4096, 1], offset: ?>>
            loom.copy %reinterpret_cast_1, %15 src_mem_space @DRAM dst_mem_space @L1, interconnect : [], broadcast : [1, 1] : memref<64x64xf16, strided<[4096, 1], offset: ?>>, memref<64x64xf16>
            loom.matmul ins(%17, %15 : memref<64x64xf16>, memref<64x64xf16>) outs(%19 : memref<64x64xf16>)
            loom.semaphore_give %15 : memref<64x64xf16>
            loom.semaphore_give %17 : memref<64x64xf16>
          }
          %20 = arith.muli %13, %c64 : index
          %21 = arith.muli %arg5, %c262144 : index
          %22 = arith.addi %21, %20 : index
          %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%22], sizes: [64, 64], strides: [4096, 1] : memref<4096x4096xf16> to memref<64x64xf16, strided<[4096, 1], offset: ?>>
          loom.copy %19, %reinterpret_cast src_mem_space @L1 dst_mem_space @DRAM, interconnect : [], broadcast : [1, 1] : memref<64x64xf16>, memref<64x64xf16, strided<[4096, 1], offset: ?>>
          loom.semaphore_give %19 : memref<64x64xf16>
        }
        scf.reduce 
      } {loom.iter_types = [#loom.iter_type<spatial>, #loom.iter_type<spatial>], loom.mapped_to_dims = [@y, @x]}
      return
    }
  }
}
