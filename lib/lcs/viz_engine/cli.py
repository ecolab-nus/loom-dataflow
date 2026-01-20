import argparse
import sys
import numpy as np
from .parser import load_constraint_spaces, find_by_func_name, compile_constraint_function, compile_expression_function, get_ranges, get_alignments
from .sampler import generate_continuous_grid, evaluate_constraints, evaluate_continuous_field, apply_alignment_filter, get_aligned_points
from .renderer import create_isosurface, create_scatter3d, build_figure, add_k_slider

def main():
    parser = argparse.ArgumentParser(description='LOOM Constraint Visualization Engine')
    parser.add_argument('--func-name', required=True, help='Name of the function to visualize')
    parser.add_argument('json_files', nargs='+', help='Path to one or more constraint JSON files')
    parser.add_argument('--resolution', type=int, default=50, help='Grid resolution for continuous surface')
    parser.add_argument('--output', type=str, default='constraint_viz.html', help='Output HTML file name')
    parser.add_argument('--no-slider', action='store_true', help='Disable K-slider animation')

    args = parser.parse_args()

    # 1. Load data
    spaces = load_constraint_spaces(args.json_files)
    space = find_by_func_name(spaces, args.func_name)
    
    if not space:
        print(f"Error: Function '{args.func_name}' not found in provided JSON files.")
        available = [s.func_name for s in spaces]
        print(f"Available functions: {', '.join(available[:10])}{'...' if len(available) > 10 else ''}")
        sys.exit(1)

    print(f"Processing: {space.func_name}")

    # 2. Setup grid and constraints
    var_names = [v.name for v in space.variables if not v.is_intermediate]
    ranges = get_ranges(space)
    alignments = get_alignments(space)
    
    if len(var_names) < 3:
        print(f"Warning: Expected 3 variables (M, N, K), but found {var_names}")

    # Generate dense grid for continuous surface
    grid = generate_continuous_grid(ranges, resolution=args.resolution)
    
    # Compile and evaluate
    expr_fns = [compile_expression_function(c, var_names) for c in space.constraints]
    field = evaluate_continuous_field(grid, expr_fns)
    
    # 3. Handle alignment points (Discrete Search Space)
    # We still need the boolean mask for scatter points
    constraint_fns = [compile_constraint_function(c, var_names) for c in space.constraints]
    aligned_grid_parts = []
    for name in var_names:
        r = ranges.get(name, (0, 512))
        step = alignments.get(name, 1)
        # Generate points: 0, 32, 64... within range
        start = (r[0] + step - 1) // step * step
        stop = r[1] // step * step
        points = np.arange(start, stop + step, step)
        aligned_grid_parts.append(points)
    
    if len(aligned_grid_parts) >= 3:
        # Use only the first 3 variables for 3D visualization if more exist
        a_grid = np.meshgrid(*aligned_grid_parts[:3], indexing='ij')
        # If there are more than 3 vars, we might need a better way to evaluate.
        # But for M,N,K it's perfect.
        a_mask = evaluate_constraints(a_grid, constraint_fns)
        aligned_points = get_aligned_points(a_grid, a_mask)
        print(f"Found {len(aligned_points)} valid aligned points.")
    else:
        aligned_points = np.array([])

    # 4. Render
    # Boundary is where field == 1.0. 
    # Points inside have field <= 1.0.
    iso = create_isosurface(grid, field, isomin=0.0, isomax=1.0)
    scatter = create_scatter3d(aligned_points)
    fig = build_figure([iso], scatter, var_names)
    
    if not args.no_slider and aligned_points.size > 0:
        fig = add_k_slider(fig, aligned_points, var_names)

    # 5. Output
    print(f"Saving visualization to {args.output}")
    fig.write_html(args.output)

if __name__ == '__main__':
    main()
