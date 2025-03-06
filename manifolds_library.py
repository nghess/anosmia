import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots





"""
Plotting
"""


# Create a custom 2D color mapping function
def position_to_color(pos, colormode='corner', x_range=None, y_range=None):
    """
    Convert 2D position to RGB color values.
    
    Parameters:
    -----------
    pos : numpy array
        2D positions with shape (n_samples, 2)
    colormode : str
        'corner', 'hsv', or 'rgb' color mapping mode
    x_range : tuple
        (min_x, max_x) range of x values, if None will use min/max of the data
    y_range : tuple
        (min_y, max_y) range of y values, if None will use min/max of the data
    """
    x, y = pos[:, 0], pos[:, 1]
    
    # Define the position ranges (used for normalization)
    if x_range is None:
        x_min, x_max = np.min(x), np.max(x)
    else:
        x_min, x_max = x_range
        
    if y_range is None:
        y_min, y_max = np.min(y), np.max(y)
    else:
        y_min, y_max = y_range
    
    # Normalize positions to 0-1 range for coloring
    x_norm = (x - x_min) / (x_max - x_min)
    y_norm = (y - y_min) / (y_max - y_min)
    
    if colormode == 'corner':

        
        # Define corner colors (RGB values between 0-1)
        bottom_left = np.array([1, 0, 0]) 
        bottom_right = np.array([1, 0.7, 0])
        top_right = np.array([0, 0.7, 1])
        top_left = np.array([0, 0, 1])
        
        # Bilinear interpolation between corners
        # First interpolate along bottom edge (blue to green)
        bottom = bottom_left * (1 - x_norm)[:, np.newaxis] + bottom_right * x_norm[:, np.newaxis]
        
        # Then interpolate along top edge (red to yellow)
        top = top_left * (1 - x_norm)[:, np.newaxis] + top_right * x_norm[:, np.newaxis]
        
        # Finally interpolate between bottom and top
        rgb_values = bottom * (1 - y_norm)[:, np.newaxis] + top * y_norm[:, np.newaxis]

        # Ensure all values are within 0-1 range (prevent any potential negative values)
        rgb_values = np.clip(rgb_values, 0, 1)
        
        # Format for Plotly
        r, g, b = rgb_values[:, 0], rgb_values[:, 1], rgb_values[:, 2]
    
    elif colormode == 'hsv':
        # HSV-based coloring: angular position maps to hue, radial position to saturation
        # Convert x,y to angle (0-360) and radius (0-1)
        # Shift x,y to be centered at 0.5, 0.5
        x_centered = x_norm - 0.5
        y_centered = y_norm - 0.5
        
        # Calculate angle (0 to 2π) and normalize radius
        angles = (np.arctan2(y_centered, x_centered) + np.pi) / (2 * np.pi)  # 0 to 1
        radii = np.minimum(np.sqrt(x_centered**2 + y_centered**2) * 2, 1)  # 0 to 1, capped at 1
        
        # Convert to RGB (simplified HSV to RGB conversion)
        h = angles * 6  # 0 to 6
        i = np.floor(h)
        f = h - i
        s = radii
        v = np.ones_like(radii)  # Set value to 1
        
        # HSV to RGB conversion
        r = np.zeros_like(h)
        g = np.zeros_like(h)
        b = np.zeros_like(h)
        
        # Cases based on hue segment
        mask = (i % 6 == 0)
        r[mask], g[mask], b[mask] = v[mask], v[mask] * (1 - s[mask] * (1 - f[mask])), v[mask] * (1 - s[mask])
        
        mask = (i % 6 == 1)
        r[mask], g[mask], b[mask] = v[mask] * (1 - s[mask] * f[mask]), v[mask], v[mask] * (1 - s[mask])
        
        mask = (i % 6 == 2)
        r[mask], g[mask], b[mask] = v[mask] * (1 - s[mask]), v[mask], v[mask] * (1 - s[mask] * (1 - f[mask]))
        
        mask = (i % 6 == 3)
        r[mask], g[mask], b[mask] = v[mask] * (1 - s[mask]), v[mask] * (1 - s[mask] * f[mask]), v[mask]
        
        mask = (i % 6 == 4)
        r[mask], g[mask], b[mask] = v[mask] * (1 - s[mask] * (1 - f[mask])), v[mask] * (1 - s[mask]), v[mask]
        
        mask = (i % 6 == 5)
        r[mask], g[mask], b[mask] = v[mask], v[mask] * (1 - s[mask]), v[mask] * (1 - s[mask] * f[mask])
        
    elif colormode == 'rgb':
        # Simple RGB mapping: x maps to red, y maps to green, their product to blue
        r = x_norm
        g = y_norm
        b = (x_norm + y_norm) / 2  # Creates variation in blue based on both coordinates
    
    # Format for plotly
    colors = [f'rgb({int(r[i]*255)}, {int(g[i]*255)}, {int(b[i]*255)})' for i in range(len(r))]
    return colors, np.column_stack((r, g, b))


def plot_2d_embedding_with_position_colors(embedding, positions, position_x_range, position_y_range, region="Brain Region", method="Embedding", 
                               dark_mode=True, global_font="Arial", global_font_size=14,
                               save_path=None, show=True):

    # Get colors for the positions
    colors, rgb_values = position_to_color(
        positions, 
        colormode='corner',
        x_range=position_x_range,
        y_range=position_y_range
    )

    # Create a figure with 2 subplots: UMAP with 2D colors and a small Position Color Map
    fig = make_subplots(
        rows=1, cols=2,
        specs=[[{"type": "scatter"}, {"type": "heatmap"}]],
        column_widths=[0.85, 0.15],  # Make the color map smaller
        horizontal_spacing=0.005
    )

    # Style settings
    global_font = "Arial"
    global_font_size = 20
    font_color = "white" if dark_mode else "black"
    background_color = "rgba(0,0,0,0)"  # Transparent
    axis_color = "white" if dark_mode else "black"

    # UMAP scatter with custom colors
    fig.add_trace(
        go.Scatter(
            x=embedding[:, 0],
            y=embedding[:, 1],
            mode='markers',
            marker=dict(
                size=3,  # Slightly smaller markers
                color=colors,
                opacity=0.9,
                symbol='circle',
                line=dict(width=0)
            ),
            customdata=np.stack((positions[:, 0], positions[:, 1]), axis=-1),
            hovertemplate='<b>UMAP 1</b>: %{x:.2f}<br>' +
                        '<b>UMAP 2</b>: %{y:.2f}<br>' +
                        '<b>X Position</b>: %{customdata[0]:.2f}<br>' +
                        '<b>Y Position</b>: %{customdata[1]:.2f}',
            name='Population activity',
            showlegend=False
        ),
        row=1, col=1
    )

    # Create a heatmap as a 2D color legend
    # Use normalized grid for the color calculation
    x_grid_norm = np.linspace(0, 1, 80) 
    y_grid_norm = np.linspace(0, 1, 80)
    X_norm, Y_norm = np.meshgrid(x_grid_norm, y_grid_norm)

    # Scale to actual environment dimensions for display
    X_scaled = X_norm * (position_x_range[1] - position_x_range[0]) + position_x_range[0]
    Y_scaled = Y_norm * (position_y_range[1] - position_y_range[0]) + position_y_range[0]

    positions_grid = np.column_stack((X_scaled.flatten(), Y_scaled.flatten()))
    colors_grid, _ = position_to_color(positions_grid, colormode='corner', 
                                    x_range=position_x_range, 
                                    y_range=position_y_range)

    # Create a 2D array for the heatmap
    z_grid = np.arange(len(colors_grid)).reshape(len(y_grid_norm), len(x_grid_norm))

    # Add color legend as a heatmap with scaled axes
    fig.add_trace(
        go.Heatmap(
            z=z_grid,
            x=X_scaled[0, :],  # Use the first row for x-axis values
            y=Y_scaled[:, 0],  # Use the first column for y-axis values
            colorscale=[[i/(len(colors_grid)-1), colors_grid[i]] for i in range(len(colors_grid))],
            showscale=False
        ),
        row=1, col=2
    )

    # Add small L-shaped axis on the main plot
    x_min, x_max = embedding[:, 0].min(), embedding[:, 0].max()
    y_min, y_max = embedding[:, 1].min(), embedding[:, 1].max()

    axis_length_x = (x_max - x_min) * 0.05  # 5% of plot width
    axis_length_y = (y_max - y_min) * 0.10  # 10% of plot height

    axis_x_start = x_min + axis_length_x * 0.2
    axis_y_start = y_min + axis_length_y * 0.2

    fig.add_trace(
        go.Scatter(
            x=[axis_x_start, axis_x_start + axis_length_x, None, axis_x_start, axis_x_start],
            y=[axis_y_start, axis_y_start, None, axis_y_start, axis_y_start + axis_length_y],
            mode='lines',
            line=dict(color=axis_color, width=3),
            showlegend=False
        ),
        row=1, col=1
    )

    # Add axis labels near the small L-shaped axis
    fig.add_annotation(
        x=axis_x_start + axis_length_x/2,
        y=axis_y_start - 0.01 * (y_max - y_min),
        text="UMAP 1", 
        showarrow=False,
        font=dict(size=global_font_size, color=font_color, family=global_font),
        xanchor="center", yanchor="top",
        row=1, col=1
    )

    fig.add_annotation(
        x=axis_x_start - 0.01 * (x_max - x_min), 
        y=axis_y_start + axis_length_y/2,
        text="UMAP 2", 
        showarrow=False,
        font=dict(size=global_font_size, color=font_color, family=global_font),
        xanchor="right", yanchor="middle",
        textangle=-90,
        row=1, col=1
    )

    # Update layout for both plots
    fig.update_layout(
        title=f"{region} spike rate embedding colored by position",
        title_font=dict(color=font_color, family=global_font, size=global_font_size + 6),
        paper_bgcolor=background_color,
        plot_bgcolor=background_color,
        font=dict(color=font_color, family=global_font, size=global_font_size),
        width=1600, height=800
    )

    # Remove grid, axis ticks, and labels from the main UMAP plot
    fig.update_xaxes(
        showticklabels=False, showgrid=False, zeroline=False, title=None,
        row=1, col=1
    )
    fig.update_yaxes(
        showticklabels=False, showgrid=False, zeroline=False, title=None,
        row=1, col=1
    )

    # Format the color map axes
    fig.update_xaxes(
        title_text="X Position (cm)", 
        showline=True, linewidth=1, linecolor=axis_color, mirror=True,
        title_font=dict(family=global_font, size=global_font_size, color=font_color),
        row=1, col=2
    )
    fig.update_yaxes(
        title_text="Y Position (cm)",
        showline=True, linewidth=1, linecolor=axis_color, mirror=True, 
        title_font=dict(family=global_font, size=global_font_size, color=font_color),
        row=1, col=2
    )

    if dark_mode:
        fig.update_xaxes(showgrid=False, zeroline=False)
        fig.update_yaxes(showgrid=False, zeroline=False)
    # Update the color map axis ticks


    if show:
        # Show the figure
        fig.show()

    # Save the figure
    if save_path:
        if dark_mode:
            # For saving, use solid background color
            fig.update_layout(
                paper_bgcolor="black",
                plot_bgcolor="black"
            )
        else:
            fig.update_layout(
                paper_bgcolor="white",
                plot_bgcolor="white"
            )
        fig.write_html(save_path + '.html')
        fig.write_image(save_path + '.png', width=1600, height=800, scale=5)
    
    return fig


def plot_3d_embedding_with_position_colors(embedding, positions, position_x_range, position_y_range, region="Brain Region", method="Embedding", 
                               dark_mode=True, global_font="Arial", global_font_size=14,
                               save_path=None, show=True):
    """
    Plots a 3D embedding using Plotly with position-based colors.
    
    Parameters:
    -----------
    embedding : numpy array
        3D embedding with shape (n_samples, 3)
    positions : numpy array
        2D positions with shape (n_samples, 2) used for coloring
    region : str
        Name of the brain region
    method : str
        Name of the embedding method (e.g., "UMAP", "t-SNE")
    dark_mode : bool
        Whether to use dark mode
    global_font : str
        Font family to use
    global_font_size : int
        Base font size
    save_path : str
        Path to save the figure, if None, the figure is not saved
    show : bool
        Whether to show the figure
    """
    # Determine colors based on dark mode
    font_color = "white" if dark_mode else "black"
    background_color = "rgba(0,0,0,0)"  # Transparent
    
    
    # Get colors for the positions
    colors, rgb_values = position_to_color(
        positions, 
        colormode='corner',
        x_range=position_x_range,
        y_range=position_y_range
    )
    
    # Create a figure with 2 subplots: 3D embedding and a 2D color map
    fig = make_subplots(
        rows=1, cols=2,
        specs=[[{"type": "scene"}, {"type": "heatmap"}]],
        column_widths=[0.85, 0.15],
        horizontal_spacing=0.005
    )
    
    # Add 3D scatter trace
    fig.add_trace(
        go.Scatter3d(
            x=embedding[:, 0],
            y=embedding[:, 1],
            z=embedding[:, 2],
            mode='markers',
            marker=dict(
                size=1.3,  # Small markers for 3D plot
                color=colors,
                opacity=1,
                symbol='circle',
                line=dict(width=0)  # No outline
            ),
            customdata=np.stack((positions[:, 0], positions[:, 1]), axis=-1),
            hovertemplate='<b>Dim 1</b>: %{x:.2f}<br>' +
                          '<b>Dim 2</b>: %{y:.2f}<br>' +
                          '<b>Dim 3</b>: %{z:.2f}<br>' +
                          '<b>X Position</b>: %{customdata[0]:.2f}<br>' +
                          '<b>Y Position</b>: %{customdata[1]:.2f}',
            name='Population activity',
            showlegend=False
        ),
        row=1, col=1
    )
    
    # Create a heatmap as a 2D color legend
    # Use normalized grid for the color calculation
    x_grid_norm = np.linspace(0, 1, 120) 
    y_grid_norm = np.linspace(0, 1, 120)
    X_norm, Y_norm = np.meshgrid(x_grid_norm, y_grid_norm)
    
    # Scale to actual environment dimensions for display
    X_scaled = X_norm * (position_x_range[1] - position_x_range[0]) + position_x_range[0]
    Y_scaled = Y_norm * (position_y_range[1] - position_y_range[0]) + position_y_range[0]
    
    positions_grid = np.column_stack((X_scaled.flatten(), Y_scaled.flatten()))
    colors_grid, _ = position_to_color(positions_grid, colormode='corner', 
                                  x_range=position_x_range, 
                                  y_range=position_y_range)
    
    # Create a 2D array for the heatmap
    z_grid = np.arange(len(colors_grid)).reshape(len(y_grid_norm), len(x_grid_norm))
    
    # Add color legend as a heatmap with scaled axes
    fig.add_trace(
        go.Heatmap(
            z=z_grid,
            x=X_scaled[0, :],  # Use the first row for x-axis values
            y=Y_scaled[:, 0],  # Use the first column for y-axis values
            colorscale=[[i/(len(colors_grid)-1), colors_grid[i]] for i in range(len(colors_grid))],
            showscale=False
        ),
        row=1, col=2
    )

    # High-quality 3D scene configuration
    scene_config = dict(
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
        zaxis=dict(visible=False),
    )

    scene_config.update(
            aspectmode='data',  # Preserve data aspect ratio
            camera=dict(
                eye=dict(x=1.8, y=1.8, z=1.8),  # Adjust camera distance
                projection=dict(type='orthographic')
            ),
            dragmode='turntable'  # Smoother rotation mode
        )
    fig.update_scenes(**scene_config)
    
    # Hide axis planes and grid for 3D plot
    fig.update_scenes(
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
        zaxis=dict(visible=False)
    )
    
    # Format the color map axes
    fig.update_xaxes(
        title_text="X Position (cm)", 
        showline=True, linewidth=1, linecolor=font_color, mirror=True,
        title_font=dict(family=global_font, size=global_font_size, color=font_color),
        row=1, col=2
    )
    fig.update_yaxes(
        title_text="Y Position (cm)",
        showline=True, linewidth=1, linecolor=font_color, mirror=True, 
        title_font=dict(family=global_font, size=global_font_size, color=font_color),
        row=1, col=2
    )
    
    # Set background, font, and sizing
    fig.update_layout(
        title=f"{region} spike rate embedding colored by position",
        paper_bgcolor=background_color,
        plot_bgcolor=background_color,
        title_font=dict(color=font_color, family=global_font, size=global_font_size + 6),
        font=dict(color=font_color, family=global_font, size=global_font_size),
        width=1600, height=800
    )
    
    if show:
        # Show the figure
        fig.show()

    # Save the figure
    if save_path:
        if dark_mode:
            # For saving, use solid background color
            fig.update_layout(
                paper_bgcolor="black",
                plot_bgcolor="black"
            )
        else:
            fig.update_layout(
                paper_bgcolor="white",
                plot_bgcolor="white"
            )
        fig.write_html(save_path + '.html')
    
    return fig



def plot_embedding_2d(embedding_OB: np.ndarray, label: np.ndarray, region: str, method: str, 
                      dark_mode: bool = True, global_font: str = "Arial", global_font_size: int = 14,
                      save_path: str = None, show: bool = True, colorbar_title = None, colorbar_range = None, colormap = 'plasma'):
    """
    Plots a 2D embedding using Plotly with color-coded sniff frequency.
    """
    
    # Determine colors based on dark mode
    font_color = "white" if dark_mode else "black"
    background_color = "black" if dark_mode else "white"
    axis_color = "white" if dark_mode else "black"

    if method == 'PCA':
        dim = 'PC'
    else:
        dim = method

    # Create scatter plot
    fig = px.scatter(
        x=embedding_OB[:, 0], 
        y=embedding_OB[:, 1], 
        color=label, range_color=colorbar_range,
        color_continuous_scale=colormap, 
        labels={'color': colorbar_title},
        title=f"{method} embedding of {region} spike rates"
    )
    
    # Hide axis ticks
    fig.update_xaxes(title=None, showticklabels=False, showgrid=False, zeroline=False)
    fig.update_yaxes(title=None, showticklabels=False, showgrid=False, zeroline=False)

    # Set background, font, and sizing
    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        title_font=dict(color=font_color, family=global_font, size=global_font_size + 6),
        font=dict(color=font_color, family=global_font, size=global_font_size),
        width=1600, height=800
    )
    
   
    
    
    
    # Add small L-shaped axis
    x_min, x_max = embedding_OB[:, 0].min(), embedding_OB[:, 0].max()
    y_min, y_max = embedding_OB[:, 1].min(), embedding_OB[:, 1].max()
    
    axis_length_x = (x_max - x_min) * 0.05  # 5% of plot width
    axis_length_y = (y_max - y_min) * 0.10  # 10% of plot height
    
    axis_x_start = x_min + axis_length_x * 0.2
    axis_y_start = y_min + axis_length_y * 0.2
    
    fig.add_trace(go.Scatter(
        x=[axis_x_start, axis_x_start + axis_length_x, None, axis_x_start, axis_x_start],
        y=[axis_y_start, axis_y_start, None, axis_y_start, axis_y_start + axis_length_y],
        mode='lines',
        line=dict(color=axis_color, width=3),
        showlegend=False
    ))
    
    # Add axis labels near the small L-shaped axis
    fig.add_annotation(
        x=axis_x_start,
        y=axis_y_start - 0.01 * (y_max - y_min),
        text=f"{dim} 1", 
        showarrow=False,
        font=dict(size=global_font_size, color=font_color),
        xanchor="left", yanchor="top"
    )
    
    fig.add_annotation(
        x=axis_x_start - 0.01 * (x_max - x_min), 
        y=axis_y_start,
        text=f"{dim} 2", 
        showarrow=False,
        font=dict(size=global_font_size, color=font_color),
        xanchor="right", yanchor="bottom",
        textangle=-90
    )
    
    if show:
        fig.show()

    # Save the figure
    if save_path:
        fig.update_layout(
            paper_bgcolor=background_color,
            plot_bgcolor=background_color
        )

        fig.write_html(save_path + '.html')
        fig.write_image(save_path + '.png', scale = 3)


def plot_embedding_3d(embedding: np.ndarray, labels: np.ndarray, region: str, method: str, 
                      dark_mode: bool = True, global_font: str = "Arial", global_font_size: int = 14,
                      save_path: str = None, show: bool = True, colorbar_title = None, colorbar_range = None, colormap = 'plasma'):
    """
    Plots a 3D embedding using Plotly with transparent background and an orthogonal axis.
    """
    
    # Determine colors based on dark mode
    font_color = "white" if dark_mode else "black"
    background_color = "black" if dark_mode else "white"
    
    # Create scatter plot
    fig = px.scatter_3d(
        x=embedding[:, 0], y=embedding[:, 1], z=embedding[:, 2], range_color=colorbar_range,
        color=labels, title=f"{method} embedding of {region} spike rates", labels={'color': colorbar_title}, 
        color_continuous_scale=colormap
    )
    
    # Hide axis planes and grid
    fig.update_scenes(
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
        zaxis=dict(visible=False)
    )
    
    # Set background, font, and sizing
    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        title_font=dict(color=font_color, family=global_font, size=global_font_size + 6),
        font=dict(color=font_color, family=global_font, size=global_font_size),
        width=1600, height=800
    )

    # Adjust marker size for better visualization
    fig.update_traces(marker=dict(size=2))
    
    if show:
        # Show the figure
        fig.show()

    # Save the figure
    if save_path:
        fig.update_layout(
            paper_bgcolor=background_color,
            plot_bgcolor=background_color
        )

        fig.write_html(save_path + '.html')


