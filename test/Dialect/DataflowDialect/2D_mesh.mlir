module {
    // Declare 8x8 mesh using chains
    %x = df.spatial_dim "x", 8
    %y = df.spatial_dim "y", 8
    %cores = "df.compute"(%x, %y) {label = "cluster_cores", map = affine_map<(d0, d1) -> (d0, d1)>} : (index, index) -> !df.compute
    %memories = "df.memory"(%x) {label = "cluster_mem", map = affine_map<(d0) -> (d0)>} : (index) -> !df.memory
    %core_to_mem = "df.mux"(%cores, %memories, %x, %y)
        {map = affine_map<(d0, d1) -> (d0)>}
        : (!df.compute, !df.memory, index, index) -> !df.mux
    %cores_to_cores = "df.interconnects"(%cores, %cores, %x, %y)
        {map = affine_map<(d0, d1) -> (d0 + 1, d1)>}
        : (!df.compute, !df.compute, index, index) -> !df.interconnect
    %mems_to_mems = "df.interconnects"(%memories, %memories, %x)
        {map = affine_map<(d0) -> (d0 + 1)>}
        : (!df.memory, !df.memory, index) -> !df.interconnect
}