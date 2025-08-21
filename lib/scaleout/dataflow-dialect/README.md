# The Dataflow (df) dialect

## Hardware Scaleout Declaration
When lowering the code to the hardware, you need to describe how your hardware scales out.
First, you need to declare the dimensions of scaleout. 
You can do that with `%dim_name = df.spatial_dim constant` to declare a dimension of size constant, i.e. from `0` to `constant`.

Then you need to declare the interconnects among the dimensions.
Currently we only support `affine_chains`.
You can declare affine chains with an affine expression: `%chain_name = df.chains[multi-dim-affine-map-of-ssa-ids]`, where the ssa-ids in the affine expression must be the dimensions declared with `df.spatial_dim`. The `multi-dim-affine-map-of-ssa-ids` is the same syntax as when you use `affine.load[multi-dim-affine-map-of-ssa-ids]` in the affine dialect. It assumes that the input dimensions are all dimensions declared with `df.spatial_dim`, so any missing dimension is considered as a repetition.

### Example
For a 8x8 2D-mesh array, you would declare
```
%x = df.spatial_dim 8
%y = df.spatial_dim 8
%horizontal_chains = "df.chains"(%x, %y) {map = affine_map<(d0, d1) -> (d0 + 1, d1)>} : (index, index) -> !df.chain
%vertical_chains = "df.chains"(%x, %y) {map = affine_map<(d0, d1) -> (d0, d1 + 1)>} : (index, index) -> !df.chain
```
Note that the vertical_chains represent all chains for each row of cores, because `%y` is missing.
As a result, this means each `core` at `%x,%y` can send data to `%x+1,%y` and can send data to `%x, %y+1`.


## Lowereing Memory Operations with Scaleout hardware

You can lower memory operations such as affine.load to concrete load with dataflow facilities.
Currently we only support `df.chained_load`.
You can do
      `%0 = df.chained_load %arg0[%arg1 + 3, %arg2 + 7] : memref<100x100xf32>, over %vertical_chains`