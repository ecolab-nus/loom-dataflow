module {
  func.func @compound() {
    %a = loom.alloc [1, 32] on @SRAM {loom.inferred_residency} : memref<1x32xf16>
    %b = loom.alloc [1, 32] on @SRAM {loom.inferred_residency} : memref<1x32xf16>
    %c = loom.alloc [1, 32] on @SRAM {loom.inferred_residency} : memref<1x32xf16>
    linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%a, %b : memref<1x32xf16>, memref<1x32xf16>) outs(%c : memref<1x32xf16>) {
    ^bb0(%lhs: f16, %rhs: f16, %out: f16):
      %sub = arith.subf %lhs, %rhs : f16
      %exp = math.exp %sub : f16
      linalg.yield %exp : f16
    }
    return
  }
}
