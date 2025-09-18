# `lib/`

The `lib` directory hosts the C++ side of TMD: hardware resource models, higher-level modules, the custom MLIR `df` dialect, and compiler pass implementations. The `scaleout` library defined in `lib/CMakeLists.txt` aggregates these components and links against MLIR.

## Layout
- `resources/` – primitive building blocks such as SRAM capacities/ports, ring and chain interconnects, and a global `ResourceManager` for lifecycle tracking.
- `modules/` – compositions of resources that capture hardware topologies (1-D chains, 2-D meshes, and torus networks). Modules maintain affine placement information through `mlir::AffineMap` objects.
- `dataflow-dialect/` – TableGen specifications and the C++ dialect library for the `df` dialect that describes spatial dimensions and interconnect declarations.
- `passes/` – MLIR transformations/analyses targeting affine IR and Triton-shared kernels. Shared utilities (e.g., spatial mapping enumeration and `common/input_sharing_analysis.cpp`) live here as well.

## How the pieces fit together
1. **Resources** provide the atomic concepts (capacity, ports, interconnect availability).
2. **Modules** build on resources to express reusable hardware patterns and expose convenience methods such as `Torus::acquireForBroadcast` or `Mesh2D::acquire`.
3. **dataflow-dialect** mirrors those structures on the MLIR side by offering ops like `df.spatial_dim`, `df.interconnects`, and `df.chained_load`.
4. **Passes** bridge high-level kernels to the dialect and modules by normalizing IR (affinization, tiling) and enumerating mappings described by DF programs.

Each subdirectory carries additional documentation in its local README.
