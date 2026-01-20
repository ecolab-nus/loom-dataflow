import plotly.graph_objects as go
import numpy as np
from typing import List, Tuple, Optional

def create_isosurface(grid: Tuple[np.ndarray, ...], field: np.ndarray, isomin: float = 0.0, isomax: float = 1.0) -> go.Isosurface:
    """
    Renders the boundary where field == 1.0.
    """
    mm, nn, kk = grid
    return go.Isosurface(
        x=mm.flatten(),
        y=nn.flatten(),
        z=kk.flatten(),
        value=field.flatten(),
        isomin=isomin,
        isomax=isomax,
        surface_count=2, 
        opacity=0.5,
        colorscale='Blues',
        showscale=False,
        name='Boundary (f=1.0)',
        caps=dict(x_show=False, y_show=False, z_show=False)
    )

def create_scatter3d(points: np.ndarray) -> go.Scatter3d:
    """
    Renders discrete valid points as a scatter plot.
    """
    if points.size == 0:
        return go.Scatter3d(x=[], y=[], z=[], mode='markers', name='No Aligned Points Found')
        
    return go.Scatter3d(
        x=points[:, 0],
        y=points[:, 1],
        z=points[:, 2],
        mode='markers',
        marker=dict(
            size=2,
            color=points[:, 2], # Color by K value
            colorscale='Viridis',
            opacity=0.9,
            line=dict(width=0.5, color='rgba(0,0,0,0.3)')
        ),
        name='Aligned Solutions'
    )

def build_figure(isosurfaces: List[go.Isosurface], scatter: go.Scatter3d, var_names: List[str]) -> go.Figure:
    """
    Combines traces and sets layout with a premium feel.
    """
    fig = go.Figure(data=[*isosurfaces, scatter])
    
    # Modern Dark Theme or Sleek Light Theme? Let's go with a sleek light/glassmorphism feel or dark.
    # The prompt mentions "vibrant colors, dark modes". Let's use a dark theme.
    
    fig.update_layout(
        template="plotly_dark",
        scene=dict(
            xaxis=dict(title=var_names[0] if len(var_names) > 0 else 'M', gridcolor='gray'),
            yaxis=dict(title=var_names[1] if len(var_names) > 1 else 'N', gridcolor='gray'),
            zaxis=dict(title=var_names[2] if len(var_names) > 2 else 'K', gridcolor='gray'),
            bgcolor='rgba(0,0,0,0)',
            camera=dict(eye=dict(x=1.5, y=1.5, z=1.5))
        ),
        margin=dict(l=0, r=0, b=0, t=50),
        title=dict(
            text='<b>LOOM Constraint Space Visualization</b>',
            x=0.5,
            y=0.95,
            font=dict(size=24, color='white')
        ),
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01)
    )
    
    return fig

def add_k_slider(fig: go.Figure, points: np.ndarray, var_names: List[str]) -> go.Figure:
    """
    Adds a slider to filter the scatter points by K.
    We'll use frames for this.
    """
    if points.size == 0:
        return fig
        
    k_vals = points[:, 2]
    unique_ks = np.unique(k_vals)
    unique_ks.sort()
    
    # Get the original scatter trace and its style to avoid overriding user settings
    iso_trace = fig.data[0]
    original_scatter = fig.data[1]
    base_marker = original_scatter.marker.to_plotly_json()
    
    # Prepend a 'Show All' frame
    frames = [go.Frame(
        data=[iso_trace, original_scatter],
        name="All"
    )]
    
    # Add per-K frames
    for k in unique_ks:
        mask = (k_vals == k)
        filtered_points = points[mask]
        
        # Create a marker style for the slice (maybe highlight in orange but keep user size)
        slice_marker = base_marker.copy()
        slice_marker.update(color='orange', opacity=1.0)
        
        frames.append(go.Frame(
            data=[
                iso_trace,
                go.Scatter3d(
                    x=filtered_points[:, 0],
                    y=filtered_points[:, 1],
                    z=filtered_points[:, 2],
                    mode='markers',
                    marker=slice_marker,
                    name=f'Slice K={k}'
                )
            ],
            name=str(int(k))
        ))
    
    fig.frames = frames
    
    # Configure slider
    sliders = [dict(
        steps=[dict(
            method='animate',
            args=[[f.name], dict(mode='immediate', frame=dict(duration=0, redraw=True), transition=dict(duration=0))],
            label=f.name
        ) for f in frames],
        active=0, # Start with 'All'
        transition=dict(duration=0),
        x=0.1, y=0, len=0.9,
        currentvalue=dict(font=dict(size=14), prefix="View: ", visible=True, xanchor="right")
    )]
    
    fig.update_layout(sliders=sliders)
    
    return fig
