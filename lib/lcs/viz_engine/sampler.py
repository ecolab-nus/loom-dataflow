import numpy as np
from typing import List, Dict, Tuple, Callable, Any

def generate_continuous_grid(ranges: Dict[str, Tuple[int, int]], resolution: int = 100) -> Tuple[np.ndarray, ...]:
    """
    Generates a voxel grid based on ranges and resolution.
    Returns (grid_M, grid_N, grid_K).
    """
    # Assuming M, N, K are the keys
    m_range = ranges.get('M', (0, 512))
    n_range = ranges.get('N', (0, 512))
    k_range = ranges.get('K', (0, 1500))
    
    m = np.linspace(m_range[0], m_range[1], resolution)
    n = np.linspace(n_range[0], n_range[1], resolution)
    k = np.linspace(k_range[0], k_range[1], resolution)
    
    return np.meshgrid(m, n, k, indexing='ij')

def evaluate_constraints(grid: Tuple[np.ndarray, ...], constraint_fns: List[Callable]) -> np.ndarray:
    """
    Evaluates all constraints on the grid.
    Returns a boolean mask.
    """
    mask = np.ones(grid[0].shape, dtype=bool)
    for fn in constraint_fns:
        mask &= fn(*grid)
    return mask

def evaluate_continuous_field(grid: Tuple[np.ndarray, ...], expr_fns: List[Callable]) -> np.ndarray:
    """
    Evaluates normalized constraint scores for smooth rendering.
    Boundary is at 1.0. We use max(normalized_scores) to combine multiple constraints.
    Normalization ensures that score <= 1.0 iff the constraint is satisfied.
    """
    # Initialize with a very small value to not interfere with max
    combined = np.full(grid[0].shape, -1e9)
    for fn in expr_fns:
        lhs, rhs = fn(*grid)
        # Robust normalization to handle negative RHS and zero RHS
        # score = (LHS - RHS) / max(1, abs(RHS)) + 1.0
        # This ensures score == 1.0 at the boundary, and score <= 1.0 when satisfied.
        denom = np.maximum(1.0, np.abs(rhs))
        score = (lhs - rhs) / denom + 1.0
        combined = np.maximum(combined, score)
    return combined

def evaluate_projected_field(grid: Tuple[np.ndarray, ...], projector: Any) -> np.ndarray:
    """
    Evaluates the feasibility field after projecting out intermediate variables.
    Boundary is at 1.0.
    """
    eval_fn = projector.compile_projected_field_function()
    return eval_fn(*grid)

def apply_alignment_filter(grid: Tuple[np.ndarray, ...], mask: np.ndarray, alignments: Dict[str, int]) -> np.ndarray:
    """
    Applies alignment conditions (e.g., % 32 == 0) to the mask.
    """
    align_mask = np.ones(mask.shape, dtype=bool)
    # Mapping grid indices to M, N, K
    # Order should match var_names used in compile_constraint_function
    # Let's assume M, N, K order for now.
    names = ['M', 'N', 'K']
    for i, name in enumerate(names):
        if name in alignments:
            align_mask &= (grid[i] % alignments[name] == 0)
    
    return mask & align_mask

def get_aligned_points(grid: Tuple[np.ndarray, ...], mask: np.ndarray) -> np.ndarray:
    """
    Returns (N, 3) array of coordinates for all points where mask is True.
    """
    indices = np.where(mask)
    points = np.stack([grid[i][indices] for i in range(len(grid))], axis=-1)
    return points
