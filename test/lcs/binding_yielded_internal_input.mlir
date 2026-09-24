module {
  func.func @yielded_internal() {
    %a = loom.alloc [1, 32] on @SRAM {loom.inferred_residency} : memref<1x32xf16>
    %b = loom.alloc [1, 32] on @SRAM {loom.inferred_residency} : memref<1x32xf16>
    %sub_out = loom.alloc [1, 32] on @SRAM {loom.inferred_residency} : memref<1x32xf16>
    %exp_out = loom.alloc [1, 32] on @SRAM {loom.inferred_residency} : memref<1x32xf16>
    linalg.generic {
      indexing_maps = [
        affine_map<(d0, d1) -> (d0, d1)>,
        affine_map<(d0, d1) -> (d0, d1)>,
        affine_map<(d0, d1) -> (d0, d1)>,
        affine_map<(d0, d1) -> (d0, d1)>
      ],
      iterator_types = ["parallel", "parallel"]
    } ins(%a, %b : memref<1x32xf16>, memref<1x32xf16>)
      outs(%sub_out, %exp_out : memref<1x32xf16>, memref<1x32xf16>) {
    ^bb0(%lhs: f16, %rhs: f16, %unused_sub: f16, %unused_exp: f16):
      %sub = arith.subf %lhs, %rhs : f16
      %exp = math.exp %sub : f16
      linalg.yield %sub, %exp : f16, f16
    }
    return
  }
}
