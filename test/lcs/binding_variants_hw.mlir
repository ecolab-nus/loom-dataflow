module @arch_system {
  %x = adl.spatial_dim "dim_x", 1
  %y = adl.spatial_dim "dim_y", 1
  %dram_bank = adl.memory.bank "mem_DRAM_instance", {bsize = 8192, nblk = 1024}
  %sram_bank = adl.memory.bank "mem_SRAM_instance", {bsize = 8192, nblk = 1024}
  %rram_bank = adl.memory.bank "mem_RRAM_instance", {bsize = 8192, nblk = 1024}
  %dram = adl.memory.array "mem_DRAM", [%x, %y] of %dram_bank
  %sram = adl.memory.array "mem_SRAM", [%x, %y] of %sram_bank
  %rram = adl.memory.array "mem_RRAM", [%x, %y] of %rram_bank
  %dram_sram = adl.processor.dmover @proc_dram_sram, from %dram to %sram
  %dram_rram = adl.processor.dmover @proc_dram_rram, from %dram to %rram
  %sram_dram = adl.processor.dmover @proc_sram_dram, from %sram to %dram
  %rram_dram = adl.processor.dmover @proc_rram_dram, from %rram to %dram
  %matrix_sram = adl.processor.compute @proc_matrix_sram, from %sram to %sram
  %matrix_rram = adl.processor.compute @proc_matrix_rram, from %rram to %rram
  %vector_sram = adl.processor.compute @proc_vector_sram, from %sram to %sram
  %vector_rram = adl.processor.compute @proc_vector_rram, from %rram to %rram
  %element = adl.arch.compose "arch_element", arch[%dram_sram, %dram_rram, %sram_dram, %rram_dram, %matrix_sram, %matrix_rram, %vector_sram, %vector_rram], mem[%dram, %sram, %rram]
  %root = adl.arch.scale "arch_x_y", [%x, %y] of %element

  module @proc_dram_sram attributes {mlar.processor_array = "dram_sram", mlar.processor_definition = "dram_sram"} {
    func.func @load_sram(%src: memref<?x?xf16>, %dst: memref<?x?xf16, 1>) {
      %m = loom.sym @M : index
      %n = loom.sym @N : index
      loom.bind_shape %src, [%m, %n] : memref<?x?xf16>
      loom.bind_shape %dst, [%m, %n] : memref<?x?xf16, 1>
      loom.bind_mem %src, @mem_DRAM : memref<?x?xf16>
      loom.bind_mem %dst, @mem_SRAM : memref<?x?xf16, 1>
      loom.copy %src, %dst src_mem_space @mem_DRAM dst_mem_space @mem_SRAM, area: [1, 1] : memref<?x?xf16> to memref<?x?xf16, 1>
      return
    }
  }

  module @proc_dram_rram attributes {mlar.processor_array = "dram_rram", mlar.processor_definition = "dram_rram"} {
    func.func @load_rram(%src: memref<?x?xf16>, %dst: memref<?x?xf16>) {
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

  module @proc_sram_dram attributes {mlar.processor_array = "sram_dram", mlar.processor_definition = "sram_dram"} {
    func.func @store_sram_f16(%src: memref<?x?xf16, 1>, %dst: memref<?x?xf16>) {
      %m = loom.sym @M : index
      %n = loom.sym @N : index
      loom.bind_shape %src, [%m, %n] : memref<?x?xf16, 1>
      loom.bind_shape %dst, [%m, %n] : memref<?x?xf16>
      loom.bind_mem %src, @mem_SRAM : memref<?x?xf16, 1>
      loom.bind_mem %dst, @mem_DRAM : memref<?x?xf16>
      loom.copy %src, %dst src_mem_space @mem_SRAM dst_mem_space @mem_DRAM, area: [1, 1] : memref<?x?xf16, 1> to memref<?x?xf16>
      return
    }
    func.func @store_sram_f32(%src: memref<?x?xf32, 1>, %dst: memref<?x?xf32>) {
      %m = loom.sym @M : index
      %n = loom.sym @N : index
      loom.bind_shape %src, [%m, %n] : memref<?x?xf32, 1>
      loom.bind_shape %dst, [%m, %n] : memref<?x?xf32>
      loom.bind_mem %src, @mem_SRAM : memref<?x?xf32, 1>
      loom.bind_mem %dst, @mem_DRAM : memref<?x?xf32>
      loom.copy %src, %dst src_mem_space @mem_SRAM dst_mem_space @mem_DRAM, area: [1, 1] : memref<?x?xf32, 1> to memref<?x?xf32>
      return
    }
  }

  module @proc_rram_dram attributes {mlar.processor_array = "rram_dram", mlar.processor_definition = "rram_dram"} {
    func.func @store_rram_f16(%src: memref<?x?xf16>, %dst: memref<?x?xf16>) {
      %m = loom.sym @M : index
      %n = loom.sym @N : index
      loom.bind_shape %src, [%m, %n] : memref<?x?xf16>
      loom.bind_shape %dst, [%m, %n] : memref<?x?xf16>
      loom.bind_mem %src, @mem_RRAM : memref<?x?xf16>
      loom.bind_mem %dst, @mem_DRAM : memref<?x?xf16>
      loom.copy %src, %dst src_mem_space @mem_RRAM dst_mem_space @mem_DRAM, area: [1, 1] : memref<?x?xf16> to memref<?x?xf16>
      return
    }
    func.func @store_rram_f32(%src: memref<?x?xf32>, %dst: memref<?x?xf32>) {
      %m = loom.sym @M : index
      %n = loom.sym @N : index
      loom.bind_shape %src, [%m, %n] : memref<?x?xf32>
      loom.bind_shape %dst, [%m, %n] : memref<?x?xf32>
      loom.bind_mem %src, @mem_RRAM : memref<?x?xf32>
      loom.bind_mem %dst, @mem_DRAM : memref<?x?xf32>
      loom.copy %src, %dst src_mem_space @mem_RRAM dst_mem_space @mem_DRAM, area: [1, 1] : memref<?x?xf32> to memref<?x?xf32>
      return
    }
  }

  module @proc_matrix_sram attributes {mlar.processor_array = "matrix_sram", mlar.processor_definition = "matrix", mlar.processor_domain = ["x", "y"]} {
    func.func @matmul(%a: memref<?x?xf16, 1>, %b: memref<?x?xf16, 1>, %c: memref<?x?xf32, 1>) {
      %m = loom.sym @M : index
      %n = loom.sym @N : index
      %k = loom.sym @K : index
      loom.bind_shape %a, [%m, %k] : memref<?x?xf16, 1>
      loom.bind_shape %b, [%k, %n] : memref<?x?xf16, 1>
      loom.bind_shape %c, [%m, %n] : memref<?x?xf32, 1>
      loom.bind_mem %a, @mem_SRAM : memref<?x?xf16, 1>
      loom.bind_mem %b, @mem_SRAM : memref<?x?xf16, 1>
      loom.bind_mem %c, @mem_SRAM : memref<?x?xf32, 1>
      linalg.matmul ins(%a, %b : memref<?x?xf16, 1>, memref<?x?xf16, 1>) outs(%c : memref<?x?xf32, 1>)
      return
    }
  }

  module @proc_matrix_rram attributes {mlar.processor_array = "matrix_rram", mlar.processor_definition = "matrix", mlar.processor_domain = ["x", "y"]} {
    func.func @matmul(%a: memref<?x?xf16>, %b: memref<?x?xf16>, %c: memref<?x?xf32>) {
      %m = loom.sym @M : index
      %n = loom.sym @N : index
      %k = loom.sym @K : index
      loom.bind_shape %a, [%m, %k] : memref<?x?xf16>
      loom.bind_shape %b, [%k, %n] : memref<?x?xf16>
      loom.bind_shape %c, [%m, %n] : memref<?x?xf32>
      loom.bind_mem %a, @mem_RRAM : memref<?x?xf16>
      loom.bind_mem %b, @mem_RRAM : memref<?x?xf16>
      loom.bind_mem %c, @mem_RRAM : memref<?x?xf32>
      linalg.matmul ins(%a, %b : memref<?x?xf16>, memref<?x?xf16>) outs(%c : memref<?x?xf32>)
      return
    }
  }

  module @proc_vector_sram attributes {mlar.processor_array = "vector_sram", mlar.processor_definition = "vector", mlar.processor_domain = ["x", "y"]} {
    func.func @add(%a: memref<?x?xf16, 1>, %b: memref<?x?xf16, 1>, %c: memref<?x?xf16, 1>) {
      %m = loom.sym @M : index
      %n = loom.sym @N : index
      loom.bind_shape %a, [%m, %n] : memref<?x?xf16, 1>
      loom.bind_shape %b, [%m, %n] : memref<?x?xf16, 1>
      loom.bind_shape %c, [%m, %n] : memref<?x?xf16, 1>
      loom.bind_mem %a, @mem_SRAM : memref<?x?xf16, 1>
      loom.bind_mem %b, @mem_SRAM : memref<?x?xf16, 1>
      loom.bind_mem %c, @mem_SRAM : memref<?x?xf16, 1>
      linalg.add ins(%a, %b : memref<?x?xf16, 1>, memref<?x?xf16, 1>) outs(%c : memref<?x?xf16, 1>)
      return
    }
    func.func @sub(%a: memref<?xf16, 1>, %b: memref<?xf16, 1>, %c: memref<?xf16, 1>) {
      %n = loom.sym @N : index
      loom.bind_shape %a, [%n] : memref<?xf16, 1>
      loom.bind_shape %b, [%n] : memref<?xf16, 1>
      loom.bind_shape %c, [%n] : memref<?xf16, 1>
      loom.bind_mem %a, @mem_SRAM : memref<?xf16, 1>
      loom.bind_mem %b, @mem_SRAM : memref<?xf16, 1>
      loom.bind_mem %c, @mem_SRAM : memref<?xf16, 1>
      linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%a, %b : memref<?xf16, 1>, memref<?xf16, 1>) outs(%c : memref<?xf16, 1>) {
      ^bb0(%lhs: f16, %rhs: f16, %out: f16):
        %v = arith.subf %lhs, %rhs : f16
        linalg.yield %v : f16
      }
      return
    }
    func.func @exp(%a: memref<?xf16, 1>, %c: memref<?xf16, 1>) {
      %n = loom.sym @N : index
      loom.bind_shape %a, [%n] : memref<?xf16, 1>
      loom.bind_shape %c, [%n] : memref<?xf16, 1>
      loom.bind_mem %a, @mem_SRAM : memref<?xf16, 1>
      loom.bind_mem %c, @mem_SRAM : memref<?xf16, 1>
      linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%a : memref<?xf16, 1>) outs(%c : memref<?xf16, 1>) {
      ^bb0(%input: f16, %out: f16):
        %v = math.exp %input : f16
        linalg.yield %v : f16
      }
      return
    }
  }

  module @proc_vector_rram attributes {mlar.processor_array = "vector_rram", mlar.processor_definition = "vector", mlar.processor_domain = ["x", "y"]} {
    func.func @add(%a: memref<?x?xf16>, %b: memref<?x?xf16>, %c: memref<?x?xf16>) {
      %m = loom.sym @M : index
      %n = loom.sym @N : index
      loom.bind_shape %a, [%m, %n] : memref<?x?xf16>
      loom.bind_shape %b, [%m, %n] : memref<?x?xf16>
      loom.bind_shape %c, [%m, %n] : memref<?x?xf16>
      loom.bind_mem %a, @mem_RRAM : memref<?x?xf16>
      loom.bind_mem %b, @mem_RRAM : memref<?x?xf16>
      loom.bind_mem %c, @mem_RRAM : memref<?x?xf16>
      linalg.add ins(%a, %b : memref<?x?xf16>, memref<?x?xf16>) outs(%c : memref<?x?xf16>)
      return
    }
    func.func @sub(%a: memref<?xf16>, %b: memref<?xf16>, %c: memref<?xf16>) {
      %n = loom.sym @N : index
      loom.bind_shape %a, [%n] : memref<?xf16>
      loom.bind_shape %b, [%n] : memref<?xf16>
      loom.bind_shape %c, [%n] : memref<?xf16>
      loom.bind_mem %a, @mem_RRAM : memref<?xf16>
      loom.bind_mem %b, @mem_RRAM : memref<?xf16>
      loom.bind_mem %c, @mem_RRAM : memref<?xf16>
      linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%a, %b : memref<?xf16>, memref<?xf16>) outs(%c : memref<?xf16>) {
      ^bb0(%lhs: f16, %rhs: f16, %out: f16):
        %v = arith.subf %lhs, %rhs : f16
        linalg.yield %v : f16
      }
      return
    }
    func.func @exp(%a: memref<?xf16>, %c: memref<?xf16>) {
      %n = loom.sym @N : index
      loom.bind_shape %a, [%n] : memref<?xf16>
      loom.bind_shape %c, [%n] : memref<?xf16>
      loom.bind_mem %a, @mem_RRAM : memref<?xf16>
      loom.bind_mem %c, @mem_RRAM : memref<?xf16>
      linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%a : memref<?xf16>) outs(%c : memref<?xf16>) {
      ^bb0(%input: f16, %out: f16):
        %v = math.exp %input : f16
        linalg.yield %v : f16
      }
      return
    }
  }
}
