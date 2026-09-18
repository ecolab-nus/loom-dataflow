module {
  %dram_bank = adl.memory.bank "dram_bank", {bsize = 8192 : i64, nblk = 1024 : i64}
  %sram_bank = adl.memory.bank "sram_bank", {bsize = 8192 : i64, nblk = 1 : i64}
  %rram_bank = adl.memory.bank "rram_bank", {bsize = 4096 : i64, nblk = 1 : i64}
  %instance = adl.spatial_dim "instance", 1
  %dram = adl.memory.array "mem_DRAM", [%instance] of %dram_bank
  %sram = adl.memory.array "mem_SRAM", [%instance] of %sram_bank
  %rram = adl.memory.array "mem_RRAM", [%instance] of %rram_bank
  %dram_sram = adl.processor.dmover @proc_dram_sram, from %dram to %sram
  %dram_rram = adl.processor.dmover @proc_dram_rram, from %dram to %rram

  module @proc_dram_sram {
    func.func @dram_to_sram(%src: memref<?x?xf16>, %dst: memref<?x?xf16>) {
      %m = loom.sym @M : index
      %n = loom.sym @N : index
      loom.bind_shape %src, [%m, %n] : memref<?x?xf16>
      loom.bind_shape %dst, [%m, %n] : memref<?x?xf16>
      loom.bind_mem %src, @mem_DRAM : memref<?x?xf16>
      loom.bind_mem %dst, @mem_SRAM : memref<?x?xf16>
      loom.copy %src, %dst src_mem_space @mem_DRAM dst_mem_space @mem_SRAM, area: [1, 1] : memref<?x?xf16> to memref<?x?xf16>
      return
    }
  }

  module @proc_dram_rram {
    func.func @dram_to_rram(%src: memref<?x?xf16>, %dst: memref<?x?xf16>) {
      %m = loom.sym @M : index
      %n = loom.sym @N : index
      loom.bind_shape %src, [%m, %n] : memref<?x?xf16>
      loom.bind_shape %dst, [%m, %n] : memref<?x?xf16>
      loom.bind_mem %src, @mem_DRAM : memref<?x?xf16>
      loom.bind_mem %dst, @mem_RRAM : memref<?x?xf16>
      loom.copy %src, %dst src_mem_space @mem_DRAM dst_mem_space @mem_RRAM, area: [1, 1] : memref<?x?xf16> to memref<?x?xf16>
      return
    }
  }
}
