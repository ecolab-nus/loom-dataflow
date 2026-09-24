module {
  func.func @kernel(%src_a: memref<32x64xf16>, %src_b: memref<64x32xf16>, %src_d: memref<32x32xf16>, %src_e: memref<32x32xf16>) {
    %a = loom.alloc [32, 64] on @SRAM : memref<32x64xf16, 1>
    %b = loom.alloc [64, 32] on @SRAM : memref<64x32xf16, 1>
    %c = loom.alloc [32, 32] on @SRAM : memref<32x32xf32, 1>
    %a2 = loom.alloc [32, 64] on @SRAM : memref<32x64xf16, 1>
    %b2 = loom.alloc [64, 32] on @SRAM : memref<64x32xf16, 1>
    %c2 = loom.alloc [32, 32] on @SRAM : memref<32x32xf32, 1>
    %d = loom.alloc [32, 32] on @SRAM : memref<32x32xf16, 1>
    %e = loom.alloc [32, 32] on @SRAM : memref<32x32xf16, 1>
    %f = loom.alloc [32, 32] on @SRAM : memref<32x32xf16, 1>
    loom.copy %src_a, %a src_mem_space @mem_DRAM dst_mem_space @mem_SRAM, area: [1, 1] : memref<32x64xf16> to memref<32x64xf16, 1>
    loom.copy %src_b, %b src_mem_space @mem_DRAM dst_mem_space @mem_SRAM, area: [1, 1] : memref<64x32xf16> to memref<64x32xf16, 1>
    loom.copy %src_a, %a2 src_mem_space @mem_DRAM dst_mem_space @mem_SRAM, area: [1, 1] : memref<32x64xf16> to memref<32x64xf16, 1>
    loom.copy %src_b, %b2 src_mem_space @mem_DRAM dst_mem_space @mem_SRAM, area: [1, 1] : memref<64x32xf16> to memref<64x32xf16, 1>
    loom.copy %src_d, %d src_mem_space @mem_DRAM dst_mem_space @mem_SRAM, area: [1, 1] : memref<32x32xf16> to memref<32x32xf16, 1>
    loom.copy %src_e, %e src_mem_space @mem_DRAM dst_mem_space @mem_SRAM, area: [1, 1] : memref<32x32xf16> to memref<32x32xf16, 1>
    linalg.matmul ins(%a2, %b2 : memref<32x64xf16, 1>, memref<64x32xf16, 1>) outs(%c2 : memref<32x32xf32, 1>)
    linalg.matmul ins(%a, %b : memref<32x64xf16, 1>, memref<64x32xf16, 1>) outs(%c : memref<32x32xf32, 1>)
    linalg.add ins(%d, %e : memref<32x32xf16, 1>, memref<32x32xf16, 1>) outs(%f : memref<32x32xf16, 1>)
    return
  }
}
