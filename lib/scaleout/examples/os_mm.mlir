module {
  func.func @mm_analysis_one_output_per_core(
      %A: memref<?x?xf32>,     // M x K
      %B: memref<?x?xf32>,     // K x N
      %C: memref<?x?xf32>,     // M x N
      %M: index, %N: index, %K: index) {

    // Declare 8x8 mesh using chains
    %x = df.spatial_dim 8
    %y = df.spatial_dim 8
    %horizontal_chains = "df.interconnects"(%x, %y) {map = affine_map<(d0, d1) -> (d0 + 1, d1)>} : (index, index) -> !df.interconnect
    %vertical_chains = "df.interconnects"(%x, %y) {map = affine_map<(d0, d1) -> (d0, d1 + 1)>} : (index, index) -> !df.interconnect

    // Hardware-agnostic: this "grid" is the output index space itself.
    // (Mapping to real hardware happens later in your compiler.)
    affine.parallel (%m, %n) = (0, 0) to (%M, %N) {
      affine.for %k = 0 to %K {
        %a = df.chained_load %A[%m, %k] : memref<?x?xf32>, over %horizontal_chains
        %b = df.chained_load %B[%k, %n] : memref<?x?xf32>, over %vertical_chains
        %c = affine.load %C[%m, %n] : memref<?x?xf32>
        affine.store %c, %C[%m, %n] : memref<?x?xf32>
      }
    }
    return
  }
}