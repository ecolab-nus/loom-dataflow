module {
    // functional untis description
    %mat_unit = df.mat "FPU" {shape = [32, 32, 32]}
    %vec_unit = df.vec "SFPU" {shape = [32]}
    // scale-out description
    %x = df.spatial_dim "x", 8
    %y = df.spatial_dim "y", 8
    %cores = df.core "core" {scaleout=(%x, %y) , scalein=(mat_unit, vec_unit, [8,1])}
    %memories = df.memory "L1" {scaleout=(%x, %y) , size = 32768, bandwidth = 64}
    %core_to_mem = df.mux %cores, %memories, {map = affine_map<(d0, d1) -> (d0, d1)>}
    %noc_h = df.interconnects "horizontal_links" %memories, %memories, {map = affine_map<(d0, d1) -> ((d0 + 1) mod 8, d1)>} 
    %noc_v = df.interconnects "vertical_links" %memories, %memories, {map = affine_map<(d0, d1) -> (d0, (d1 + 1) mod 8)>} 
    // dram
    %dram_idx = df.spatial_dim "d", 4
    %drams = df.memory "DRAM"{scaleout=(%dram_idx) , size = 34359738368, bandwidth = 512}
    %to_dram = df.interconnects %memories: !df.memory, %drams : !df.memory, {map = affine_map<(d0, d1) -> (d0 ceildiv 4 + 2 * (d1 ceildiv 4))>}
}