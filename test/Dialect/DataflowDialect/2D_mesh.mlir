module {
    // functional untis description
    %mat_unit = df.mat "FPU" {shape = [32, 32, 32], throughput = 128}
    %vec_unit = df.vec "SFPU" {shape = [32]}
    // scale-out description
    %x = df.spatial_dim "x", 8
    %y = df.spatial_dim "y", 8
    %cores = df.core "core" {scaleout=(%x, %y) , scalein=(%mat_unit, %vec_unit, [8,1])}
    %L1 = df.memory "L1" {scaleout=(%x, %y) , size = 1499136, bandwidth = 15}
    %core_to_mem = df.mux %cores, %L1, {map = affine_map<(d0, d1) -> (d0, d1)>}
    %noc_h = df.interconnects "horizontal_links" %L1 : !df.memory, %L1 : !df.memory, {map = affine_map<(d0, d1) -> ((d0 + 1) mod 8, d1)>, bandwidth = 128, spatial_dims = [@x]} : !df.interconnect
    %noc_v = df.interconnects "vertical_links" %L1 : !df.memory, %L1 : !df.memory, {map = affine_map<(d0, d1) -> (d0, (d1 + 1) mod 8)>, bandwidth = 128, spatial_dims = [@y]} : !df.interconnect 
    // dram
    %dram_idx = df.spatial_dim "d", 4
    %drams = df.memory "DRAM" {scaleout=(%dram_idx) , size = 34359738368, bandwidth = 288}
    %to_dram = df.interconnects "NoC" %L1: !df.memory, %drams : !df.memory, {map = affine_map<(d0, d1) -> (d0 ceildiv 4 + 2 * (d1 ceildiv 4))>}
}