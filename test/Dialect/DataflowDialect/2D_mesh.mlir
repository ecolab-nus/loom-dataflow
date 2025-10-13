module {
    // Declare 8x8 mesh using chains
    %x = df.spatial_dim "x", 8
    %y = df.spatial_dim "y", 8
    %cores = df.compute "cluster_cores", %x, %y {map = affine_map<(d0, d1) -> (d0, d1)>}
    %memories = df.memory "cluster_mem", %x, %y {map = affine_map<(d0, d1) -> (d0, d1)>}
    %core_to_mem = df.mux %cores: !df.compute, %memories: !df.memory, %x, %y {map = affine_map<(d0, d1) -> (d0, d1)>}
    %cores_to_cores_vertical = df.interconnects %cores : !df.compute, %cores: !df.compute, %x, %y {map = affine_map<(d0, d1) -> (d0 + 1, d1)>} : !df.interconnect
    %cores_to_cores_horizontal = df.interconnects %cores : !df.compute, %cores: !df.compute, %x, %y {map = affine_map<(d0, d1) -> (d0, d1 + 1)>} : !df.interconnect
}