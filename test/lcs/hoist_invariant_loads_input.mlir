// Global memory is HBM (not DRAM). Only the load of %a may leave the loop:
// %onchip is not a kernel argument and %out is written inside the nest.
func.func @k(%a: memref<128x64xf16>, %out: memref<128x64xf16>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c4 = arith.constant 4 : index
  %c32 = arith.constant 32 : index
  %onchip = loom.alloc [32, 64] on @SRAM : memref<32x64xf16>
  scf.for %i = %c0 to %c4 step %c1 {
    %ra = memref.reinterpret_cast %a to offset: [0], sizes: [32, 64], strides: [64, 1] : memref<128x64xf16> to memref<32x64xf16, strided<[64, 1]>>
    %ta = loom.alloc [32, 64] on @SRAM : memref<32x64xf16>
    %sa = loom.semaphore_take %ta : memref<32x64xf16> -> memref<32x64xf16>
    loom.copy %ra, %sa src_mem_space @mem_HBM dst_mem_space @mem_SRAM, area : [1, 1] : memref<32x64xf16, strided<[64, 1]>> to memref<32x64xf16>
    %tc = loom.alloc [32, 64] on @SRAM : memref<32x64xf16>
    %sc = loom.semaphore_take %tc : memref<32x64xf16> -> memref<32x64xf16>
    loom.copy %onchip, %sc src_mem_space @mem_SRAM dst_mem_space @mem_SRAM, area : [1, 1] : memref<32x64xf16> to memref<32x64xf16>
    %ro = memref.reinterpret_cast %out to offset: [0], sizes: [32, 64], strides: [64, 1] : memref<128x64xf16> to memref<32x64xf16, strided<[64, 1]>>
    %tr = loom.alloc [32, 64] on @SRAM : memref<32x64xf16>
    %sr = loom.semaphore_take %tr : memref<32x64xf16> -> memref<32x64xf16>
    loom.copy %ro, %sr src_mem_space @mem_HBM dst_mem_space @mem_SRAM, area : [1, 1] : memref<32x64xf16, strided<[64, 1]>> to memref<32x64xf16>
    %to = loom.alloc [32, 64] on @SRAM : memref<32x64xf16>
    %so = loom.semaphore_take %to : memref<32x64xf16> -> memref<32x64xf16>
    linalg.add ins(%sa, %sc : memref<32x64xf16>, memref<32x64xf16>) outs(%so : memref<32x64xf16>)
    linalg.add ins(%so, %sr : memref<32x64xf16>, memref<32x64xf16>) outs(%so : memref<32x64xf16>)
    %off = arith.muli %i, %c32 : index
    %wo = memref.reinterpret_cast %out to offset: [%off], sizes: [32, 64], strides: [64, 1] : memref<128x64xf16> to memref<32x64xf16, strided<[64, 1], offset: ?>>
    loom.copy %so, %wo src_mem_space @mem_SRAM dst_mem_space @mem_HBM, area : [1, 1] : memref<32x64xf16> to memref<32x64xf16, strided<[64, 1], offset: ?>>
    loom.semaphore_give %sa : memref<32x64xf16>
    loom.semaphore_give %sc : memref<32x64xf16>
    loom.semaphore_give %sr : memref<32x64xf16>
    loom.semaphore_give %so : memref<32x64xf16>
  } {loom.independent}
  return
}
