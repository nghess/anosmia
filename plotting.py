
import numpy as np
import pandas as pd
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.colors as mcolors
from scipy.io import loadmat
import os
import seaborn as sns



def load_colormap(cmap = 'smoothjazz', type: str = 'matplotlib'):
    # Load the .mat file
    if cmap == 'smoothjazz':
        mat_file = "E:\\Sid_LFP\\solojazz.mat"
        data = loadmat(mat_file)["pmc"]
    elif cmap == 'burningchrome':
        csv_file = "E:\\Sid_LFP\\burning_chrome.csv"
        data = pd.read_csv(csv_file, header=None).values
        
    
    # Create a custom colormap
    if type == 'matplotlib':
        colormap = mcolors.ListedColormap(data, name="solojazz")
    elif type == 'plotly':
        colormap = []
        for i in range(data.shape[0]):
            r, g, b = data[i]
            colormap.append(f'rgba({int(r * 255)}, {int(g * 255)}, {int(b * 255)}, 1)')
        colormap = [colormap[i] for i in range(len(colormap) - 1, -1, -1)]

    return colormap



def plot_spike_rates(time_bins: np.ndarray, rates_OB: np.ndarray, rates_HC: np.ndarray, 
                     ob_units: np.ndarray, hc_units: np.ndarray, dark_mode: bool = True, 
                     global_font: str = "Arial", global_font_size: int = 14, 
                     show: bool = True, save_path: str = None, normalized: bool = False, cmap = 'Magma', step: int = 1, title = "Spike rates in olfactory bulb and hippocampus units", log_scale = False):
    """
    Plots spike rates for OB and HC units using Plotly as two separate heatmaps stacked vertically with a transparent background.
    Ensures both heatmaps share the same color scale and only one colorbar is displayed.
    """

    # Determine font and axis color
    font_color = "white" if dark_mode else "black"


       # Handle log transformation if requested
    if log_scale:
        # Add small epsilon to avoid log(0)
        epsilon = 1e-10
        rates_OB_plot = np.log(rates_OB + epsilon)
        rates_HC_plot = np.log(rates_HC + epsilon)
        
        if normalized == True:
            colorbar_title = "Log Spike rate (z)"
        elif normalized == 'latency':
            colorbar_title = "Log Spike latency (s)"
            vmin, vmax = 0, 0.1
        else:
            colorbar_title = "Log Spike rate (Hz)"
            
        vmin = -7
        vmax = 0
    else:
        rates_OB_plot = rates_OB
        rates_HC_plot = rates_HC
        
        if normalized == True:
            colorbar_title = "Spike rate (z)"
            vmin, vmax = -2, 3
        elif normalized == 'latency':
            colorbar_title = "Spike latency (s)"
            vmin, vmax = 0, 0.1
        else:
            colorbar_title = "Spike rate (Hz)"
            vmin = np.min(np.concatenate((rates_OB, rates_HC)))
            vmax = np.max(np.concatenate((rates_OB, rates_HC)))




    # Create subplot layout
    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True, 
        subplot_titles=["OB Units", "HC Units"], 
        vertical_spacing=0.1
    )

    # Create OB heatmap
    heatmap_OB = go.Heatmap(
        z=rates_OB_plot[:, ::step],
        x=time_bins[::step],
        y=np.arange(len(ob_units)),
        colorscale=cmap,
        zmin=vmin, zmax=vmax,  
        colorbar=dict(title=dict(text=colorbar_title, font=dict(family=global_font, size=global_font_size))),
        name='OB Units',
    )

    # Create HC heatmap without colorbar
    heatmap_HC = go.Heatmap(
        z=rates_HC_plot[:, ::step],
        x=time_bins[::step],
        y=np.arange(len(hc_units)),
        colorscale=cmap,
        zmin=vmin, zmax=vmax,  
        showscale=False,  
        name='HC Units',
    )

    # Add traces to the figure
    fig.add_trace(heatmap_OB, row=1, col=1)
    fig.add_trace(heatmap_HC, row=2, col=1)

    # Update layout for transparency and styling
    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        title_font=dict(color=font_color, family=global_font, size=global_font_size + 10),
        font=dict(color=font_color, family=global_font, size=global_font_size),
        xaxis=dict(
            title=None, showgrid=False, zeroline=False, 
            tickfont=dict(family=global_font, size=global_font_size)
        ),
        yaxis=dict(
            title='OB Neuron ID', showgrid=False, zeroline=False, 
            tickmode='array', tickvals=np.linspace(0, len(ob_units) - 1, num=5, dtype=int),
            tickfont=dict(family=global_font, size=global_font_size)
        ),
        xaxis2=dict(
            title='Time (min)', showgrid=False, zeroline=False, 
            tickfont=dict(family=global_font, size=global_font_size)
        ),
        yaxis2=dict(
            title='HC Neuron ID', showgrid=False, zeroline=False, 
            tickmode='array', tickvals=np.linspace(0, len(hc_units) - 1, num=5, dtype=int),
            tickfont=dict(family=global_font, size=global_font_size)
        ),
        width=1400,
        height=600,
    )

    # Update title
    fig.update_layout(
        title=dict(
            text=title,
            font=dict(family=global_font, size=global_font_size + 10, color=font_color), 
            xanchor="left",
            yanchor="top",
            y=0.95,
            x=0.02
        )
    )

    # converting x-axis to minutes
    fig.update_xaxes(tickvals=np.linspace(0, time_bins[-1], num=7), ticktext=[f"{int(t/60)}" for t in np.linspace(0, time_bins[-1], num=7)])

    # Show the figure
    if show:
        fig.show()

    # Save the figure
    if save_path:
        # Save as html and png with dark mode
        fig.update_layout(paper_bgcolor='black', plot_bgcolor='black')
        fig.write_image(save_path + '.png', scale=3)
        fig.write_html(save_path + '.html')



def plot_raw_signals(lfp, theta, mua, sniff,channel, seg, mouse, session, save_path, time_axis, length, fs = 30000, colors = {'LFP': '#1f77b4', 'MUA': '#ff7f0e', 'Sniff': '#2ca02c'}):

    fig, ax = plt.subplots(3, 1, figsize=(20, 10), sharex=True)
    signals = {'LFP': lfp, 'MUA': mua, 'Sniff': sniff}
    for i, (label, data) in enumerate(signals.items()):
        if label in ['LFP', 'MUA']:
            ax[i].plot(time_axis, data[channel, seg:seg + length * fs], color=colors[label], linewidth=0.5, alpha =0.7)
        else:
            ax[i].plot(time_axis, data[seg:seg + length * fs], color=colors[label], linewidth=0.8)

        
        # Y-axis labels
        if i in [0, 1]:
            ax[i].set_ylabel(f'{label} (uV)')
        elif i in [2,3]:
            ax[i].set_ylabel(f'{label} (A.U.)')


    ax[-1].set_xlabel('Time (s)')

    # Setting x-ticks and labels
    xticks= np.arange(0, length + 0.5, 0.5)
    start_time = seg / fs
    xtick_labels = [f"{start_time + x:.1f}" for x in xticks]
    ax[-1].set_xticks(xticks)
    ax[-1].set_xticklabels(xtick_labels)
    sns.despine()
    plt.savefig(save_path, dpi=300)
    plt.close()


def plot_PSD(ob_data, hc_data, save_dir, dark_mode=True):
        
    
    # Apply dark background styling if enabled
    if dark_mode:
        plt.style.use('dark_background')
    
    sns.set_context('paper', font_scale=3)
    plt.rcParams['font.family'] = 'Arial'

    # Define colors for OB and HC
    if dark_mode:
        colors = {
            'ob': (0.128, 0.334, 0.517, 1.0),  # Teal Blue (OB)
            'hc': (0.666, 0.227, 0.430, 1.0)   # Warm Pink (HC)
        }
        bandwidth_colors = {
            'theta': (0.204, 0.591, 0.663, 1.0),  # Deep Blue-Teal
            'beta': (0.797, 0.105, 0.311, 1.0),   # Muted Red-Orange
            'gamma': (0.267, 0.749, 0.441, 1.0)   # Greenish-Yellow
        }
    else:
        colors = {
            'ob': [0.00959, 0.81097, 1],  # Light Blue (OB)
            'hc': [1, 0.19862, 0.00959]   # Red-Orange (HC)
        }
        bandwidth_colors = {
            'theta': [0.7647, 0.0392, 0.200],  # Deep Red
            'beta': [0.7961, 0.4980, 0.0275],  # Orange
            'gamma': [0.3961, 0.2588, 0.1725]  # Brownish
        }
    

    bandwidths = {'theta': [2, 12], 'beta': [18, 30], 'gamma': [65, 100]}
    freq_range = [1, 300]



    if dark_mode:
        plt.figure(figsize=(20, 10))
    else:
        plt.figure(figsize=(10,10))
    for condition, data, color in zip(['hc', 'ob'], [hc_data, ob_data], [colors['hc'], colors['ob']]):
        grand_mean_psd_list = []
        for mouse in data['Mouse'].unique():
            mouse_data = data[(data['Mouse'] == mouse)]
            mouse_data = mouse_data[(mouse_data['Frequency'] >= freq_range[0]) & (mouse_data['Frequency'] <= freq_range[1])]
            mouse_data = mouse_data.groupby(['Mouse', 'Frequency']).psd.mean().reset_index()
            grand_mean_psd_list.append(mouse_data['psd'].values)
            plt.plot(mouse_data['Frequency'], mouse_data['psd'], color=color, alpha=0.5, linewidth=1, label=f'{condition} Mouse {mouse}' if mouse == data['Mouse'].unique()[0] else "")

        grand_mean_psd = np.mean(grand_mean_psd_list, axis=0)
        grand_mse_psd = np.std(grand_mean_psd_list, axis=0) / np.sqrt(len(grand_mean_psd_list))

        plt.plot(mouse_data['Frequency'], grand_mean_psd, color=color, linewidth=3, label=f'{condition.capitalize()} Grand Mean')
        plt.fill_between(mouse_data['Frequency'], grand_mean_psd - grand_mse_psd, grand_mean_psd + grand_mse_psd, color=color, alpha=0.2)
    plt.yscale('log')
    plt.xscale('log')
    sns.despine()

    # setting y scale
    if dark_mode:
        plt.ylim(1, 10**6)

    # shading the bandwidths of interest and adding Greek letters
    greek_letters = {'theta': '$\\theta$', 'beta': '$\\beta$', 'gamma': '$\\gamma$'}
    y_max = plt.gca().get_ylim()[1]
    
    for band, bandcolor in bandwidth_colors.items():
        plt.axvspan(bandwidths[band][0], bandwidths[band][1], color=bandcolor, alpha=0.2)
        # Calculate middle of the band for text placement
        x_pos = np.sqrt(bandwidths[band][0] * bandwidths[band][1])  # Geometric mean for log scale
        plt.text(x_pos, y_max*0.7, greek_letters[band], 
                horizontalalignment='center', 
                verticalalignment='center',
                color=bandcolor,
                fontsize=24)
    
    plt.xticks([1, 2, 12, 18, 30, 65, 100, 300], [1, 2, 12, 18, 30, 65, 100, 300])

    plt.savefig(os.path.join(save_dir, f'Combined_psd_logx_letters_ticks.png'), format='png', dpi=600)
    plt.savefig(os.path.join(save_dir, f'Combined_psd_logx_letters_ticks.svg'), format='svg')
    plt.close()


def plot_sniff_frequencies(time_bins: np.ndarray, mean_freqs: np.ndarray, 
                           dark_mode: bool = True, global_font: str = "Arial", 
                           global_font_size: int = 14, log_y: str = None, 
                           show: bool = True, save_path: str = None, step: int = 1):
    """
    Plots sniff frequency over time using Plotly with a transparent background.
    Supports log-scaled y-axis (log2, log10, ln) while keeping tick labels in Hz.
    """

    # Determine font and axis color
    font_color = "white" if dark_mode else "black"

    # Define y-axis scale type and tick values
    if log_y == "log2":
        yaxis_type = "log"
        y_label = "Sniffs per second (log₂ scale)"
        tickvals = [2**i for i in range(0, 5)]  # Log2 scale ticks (2, 4, 8, 16)
    elif log_y == "log10":
        yaxis_type = "log"
        y_label = "Sniffs per second (log₁₀ scale)"
        tickvals = [10**i for i in range(0, 2)]  # Log10 scale ticks (1, 10)
    elif log_y == "ln":
        yaxis_type = "log"
        y_label = "Sniffs per second (ln scale)"
        tickvals = np.exp(np.arange(0, 3)).tolist()  # Natural log scale ticks (1, e≈2.71, e²≈7.39)
    else:
        yaxis_type = None  # Linear scale
        y_label = "Sniffs per second"
        tickvals = [2, 4, 6, 8, 10, 12]  # ✅ Exact tick values for linear scale

    # Ensure tick values are within data range
    tickvals = [t for t in tickvals if np.min(mean_freqs) <= t <= np.max(mean_freqs)] if tickvals else None

    # Create scatter plot
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=time_bins[::step] / 60,  # Convert time to minutes
        y=mean_freqs[::step],
        mode='markers',
        marker=dict(size=4, color='dodgerblue'),
        name="Mean Sniff Frequency"
    ))

    # Update layout for transparency and styling
    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        width=1400, height=600,
        title=dict(
            text="Sniffing behavior over time",
            font=dict(family=global_font, size=global_font_size + 10, color=font_color),
            x=0.02, y=0.95, xanchor="left", yanchor="top"
        ),
        font=dict(color=font_color, family=global_font, size=global_font_size),
        xaxis=dict(
            title="Time (min)", 
            showgrid=False, zeroline=True, 
            tickfont=dict(family=global_font, size=global_font_size),
            range=[0, np.max(time_bins) / 60]
        ),
        yaxis=dict(
            title=y_label, 
            type=yaxis_type,  # Set log scale if needed
            tickmode="array" if tickvals else "auto",
            tickvals=tickvals if tickvals else None,  
            ticktext=[f"{v}" for v in tickvals] if tickvals else None,  
            showgrid=False, zeroline=True, 
            showline=True,  # ✅ Ensures y-axis is always visible
            tickfont=dict(family=global_font, size=global_font_size),
        ),
    )

    # Special handling for linear scale tick labels (forces correct display)
    if log_y is None:
        fig.update_yaxes(
            tickmode="array",
            tickvals=[2, 4, 6, 8, 10, 12, 14],  # ✅ Ensures only these values appear
            ticktext=[str(v) for v in [2, 4, 6, 8, 10, 12, 14]],
            showgrid = False
        )

    # Show the figure
    if show:
        fig.show()

    # Save the figure
    if save_path:
        # Save as SVG
        fig.write_image(save_path + '.svg')

        # Save as PNG and HTML with dark mode adjustments
        fig.update_layout(paper_bgcolor='black', plot_bgcolor='black')
        fig.write_image(save_path + '.png', scale=3)
        fig.write_html(save_path + '.html')


def plot_position_trajectories(data, save_path=None):
    # Create segments for the line
    points = np.array([data['x'], data['y']]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)

    # Define colormap
    cmap = plt.get_cmap("rainbow")
    norm = Normalize(vmin=data['time'].min(), vmax=data['time'].max())
    colors = cmap(norm(data['time'].values))

    # Plot
    fig, ax = plt.subplots(figsize=(20, 20))
    lc = LineCollection(segments, cmap=cmap, norm=norm, linewidth=1)
    lc.set_array(data['time'])
    ax.add_collection(lc)

    # Adjust limits
    ax.set_xlim(data['x'].min(), data['x'].max())
    ax.set_ylim(data['y'].min(), data['y'].max())

    # Labels and title
    ax.set_title("Positions over time")
    ax.set_xlabel("X Position (cm)")
    ax.set_ylabel("Y Position (cm)")

    # Colorbar
    cbar = plt.colorbar(lc, ax=ax)
    cbar.set_label("Time (s)", rotation=270)

    plt.tight_layout()


    if save_path:
        plt.savefig(save_path + '\\position.png', dpi=300)
        plt.close(fig)
    else:
        plt.show()