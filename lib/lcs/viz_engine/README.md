# LOOM Constraint Visualization Engine

This package provides tools to visualize polynomial and linear constraints on the $M, N, K$ search space for tiling optimization.

## Components

- **Parser (`parser.py`)**: Loads JSON constraint spaces and compiles polynomial/linear terms into NumPy-evaluable functions.
- **Sampler (`sampler.py`)**: Generates grids (continuous and discrete) to evaluate constraint satisfaction across the 3D space.
- **Renderer (`renderer.py`)**: Uses Plotly to create high-quality 3D visualizations, including:
  - **Isosurface**: Shows the continuous boundary of physical constraints.
  - **Scatter3d**: Shows a sparse cloud of valid, hardware-aligned points.
  - **K-Slider**: Interactive animation to slice through the $K$ dimension.
- **CLI (`cli.py`)**: Unified entry point for processing and visualizing constraint files.

## Usage

Run the visualization engine using the `loom-dev` environment:

```bash
conda activate loom-dev
python -m lib.lcs.viz_engine.cli --func-name <FUNC_NAME> <JSON_FILE> [--output <OUTPUT_HTML>]
```

### Parameters

- `--func-name`: The function name to extract from the JSON (e.g., `matmul_kernel__d0i0_d1i0__f01__d_d`).
- `json_files`: One or more paths to constraint JSON files.
- `--resolution`: (Optional) Resolution of the continuous surface grid (default: 40).
- `--output`: (Optional) Output HTML file path (default: `constraint_viz.html`).
- `--no-slider`: (Optional) Disable the interactive K-slider animation for potentially better performance in very large spaces.

## Example

```bash
python -m lib.lcs.viz_engine.cli \
  --func-name matmul_kernel__d0i0_d1i0__f01__d_d \
  test/Passes/mm_2Dmesh/raw_constraint_space.json
```
