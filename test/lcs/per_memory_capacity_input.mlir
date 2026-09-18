module {
  func.func @per_memory_capacity(%src_a: memref<32x64xf16>, %src_b: memref<64x32xf16>) {
    %k = loom.sym @K {upper_bound = 64 : index} : index
    %a = loom.alloc [32, %k] on @SRAM : memref<32x?xf16>
    %b = loom.alloc [%k, 32] on @RRAM : memref<?x32xf16>
    %c = loom.alloc [32, 32] on @SRAM : memref<32x32xf32>
    loom.copy %src_a, %a src_mem_space @mem_DRAM dst_mem_space @mem_SRAM, area: [1, 1] : memref<32x64xf16> to memref<32x?xf16>
    loom.copy %src_b, %b src_mem_space @mem_DRAM dst_mem_space @mem_RRAM, area: [1, 1] : memref<64x32xf16> to memref<?x32xf16>
    return
  }
}
