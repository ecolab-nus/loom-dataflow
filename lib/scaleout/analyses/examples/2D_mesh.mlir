module {
    // Declare 8x8 mesh using chains
    %x = df.spatial_dim 8
    %y = df.spatial_dim 8
    %horizontal_chains = "df.interconnects"(%x, %y) {map = affine_map<(d0, d1) -> (d0 + 1, d1)>} : (index, index) -> !df.interconnect
    %vertical_chains = "df.interconnects"(%x, %y) {map = affine_map<(d0, d1) -> (d0, d1 + 1)>} : (index, index) -> !df.interconnect
}