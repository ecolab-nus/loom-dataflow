module {
    // functional units
    %mat_unit = df.mat "PT" {shape = [128, 128, 128]}
    %vec_unit = df.vec "SFP" {shape = [32]}
    // scale-out
    %x = df.spatial_dim "x", 32
    %y = df.spatial_dim "y", 2
    %cores = df.core "cores" {scaleout=(%x, %y) , scalein=(%mat_unit, %vec_unit, [1,1])}
    %memories = df.memory "L1" {scaleout=(%x) , size = 2097152, bandwidth = 128}
    %core_to_mem = df.mux %cores, %memories, {map = affine_map<(d0, d1) -> (d0)>}
    %small_rings = df.interconnects "small_rings" %cores : !df.compute, %cores: !df.compute, {map = affine_map<(d0, d1) -> ((d0 + 1) mod 8, d1)>, bandwidth = 32}
    %big_ring = df.interconnects "big_ring" %memories : !df.memory, %memories: !df.memory, {map = affine_map<(d0) -> ((d0 + 1) mod 8)>, bandwidth = 258}
    // Global (DRAM/L2)
    %d = df.spatial_dim "d", 2
    %dram = df.memory "DRAM" {scaleout=(%d) , size = 34359738368, bandwidth = 512}
    %to_dram = df.interconnects %dram : !df.memory, %memories : !df.memory, {map = affine_map<(d0) -> (d0*31)>}
}