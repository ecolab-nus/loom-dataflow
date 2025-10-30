module {
    // Declare 8x8 mesh using chains
    %x = df.spatial_dim "x", 8
    %y = df.spatial_dim "y", 8
    %cores = df.compute "cores", %x, %y {map = affine_map<(d0, d1) -> (d0, d1)>}
    %memories = df.memory "L1", %x, %y {map = affine_map<(d0, d1) -> (d0, d1)>, size = 32768 : i64, bandwidth = 64 : i64}
    %core_to_mem = df.mux %cores: !df.compute, %memories: !df.memory, %x, %y {map = affine_map<(d0, d1) -> (d0, d1)>}
    %horizontal = df.interconnects "horizontal_links" %memories : !df.memory, %memories: !df.memory, %x, %y {map = affine_map<(d0, d1) -> (d0 + 1, d1)>} : !df.interconnect
    %vertical = df.interconnects "vertical_links" %memories : !df.memory, %memories: !df.memory, %x, %y {map = affine_map<(d0, d1) -> (d0, d1 + 1)>} : !df.interconnect
}