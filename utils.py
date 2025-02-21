import os
import re
import numpy as np
import pandas as pd
from kilosort import run_kilosort
from kilosort.io import save_preprocessing, load_ops
from pathlib import Path
from scipy.io import loadmat
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import signal
import matplotlib.colors as mcolors
from concurrent.futures import ProcessPoolExecutor
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from scipy.ndimage import gaussian_filter1d


"""
Run kilosort4. Use settings dictionary to change kilosort settings for the run.
"""
def kilosort(data_path: str, results_path: str, probe_path: str = 'probe_maps/8_tetrode_2_region_20um.mat', num_channels: int = 32, save_preprocessed: bool = True, clean_outliers: bool = True):
    # Initialize paths
    data_path = Path(data_path)
    results_path = Path(results_path)
    
    # Multiplier for min/max thresholds. Values outside these ranges will be set to zero. Only applies if clean_outliers = True.
    clip_mult = 3
    
    # Handle numpy files by temporarily converting to .bin format   
    if data_path.suffix == '.npy':
        # Load .npy file and save as binary
        data = np.load(data_path)
        print(f"{data_path.parent}")
        print(f"Data import shape:{data.shape}")
        data_min = data.min()
        data_max = data.max()
        data_std = data.std()
        
        # Apply outlier clipping
        if clean_outliers:
            data = clip_outliers_with_window(data, clip_mult)
        
        data = data.reshape(-1, order = 'F')
        temp_bin_path = data_path.parent / 'temp.bin'
        data.tofile(temp_bin_path)
        print(f"Created temporary binary file: {temp_bin_path}")

        # Create temporary binary file in data parent directory
        data_path = data_path.parent / 'temp.bin'
      
    else:
        data = np.load(data_path)
        if clean_outliers:
            data = clip_outliers_with_window(data, clip_mult)

    # Create results directory if it doesn't exist
    results_path.mkdir(parents=True, exist_ok=True)

    # Define Kilosort4 settings for the current run
    settings = {'data_dir': data_path.parent, 'n_chan_bin': num_channels, 'Th_universal': 10, 'Th_learned': 9, 'nblocks': 0, 'drift_smoothing': [0, 0, 0], 'dminx': 20, 'artifact_threshold': np.inf, 'batch_size': 60000}

    # Run Kilosort 4
    try:
        ops, st, clu, tF, Wall, similar_templates, is_ref, est_contam_rate, kept_spikes = \
            run_kilosort(
                settings=settings, 
                probe_name=Path.cwd() / probe_path,
                save_preprocessed_copy=save_preprocessed,
                do_CAR= False,
                results_dir=results_path
                )
        
        # Delete temporary binary file from drive if it exists
        temp_bin_path = data_path.parent / 'temp.bin'
        if temp_bin_path.exists():
            temp_bin_path.unlink()

        # Write to 'good' units summary
        unit_summary(data_path, results_path, data_min, data_max, data_std, clip_mult, error=False)
        
        # Return results
        return ops, st, clu, tF, Wall, similar_templates, is_ref, est_contam_rate, kept_spikes
    
    except:
        # Write error to log
        unit_summary(data_path, results_path, data_min, data_max, data_std, clip_mult, error=True)
        return None
    
"""
Helper Functions Below
"""

# Get the name of the last two directories in data path
def get_savedirs(path):
    path = str(path)
    parts = path.split(os.path.sep)
    return os.path.sep.join(parts[-3:-1])

# Get all files matching extension and keyword in a directory and its subdirectories
def get_file_paths(directory: str, extension: str, keyword: str, print_paths=False) -> list:
    paths = [f for f in Path(directory).glob(f'**/*.{extension}') if keyword in f.name]
    print(f'Found {len(paths)} {keyword}.{extension} files')
    if print_paths:
        show_paths(paths)
    return paths

# Print collected paths and their indices
def show_paths(data_paths):
    for ii, path in enumerate(data_paths):
        print(f"{ii} {path}")

def clip_outliers_with_window(data: np.ndarray, clip_mult: float = 2, window_size: int = 30000, overlap: int = 10000) -> np.ndarray:
    """
    Clips outlier values in neural data with a ±1 sample window around detected outliers.
    
    Args:
        data:  Data array of shape (n_channels, n_samples)
        clip_mult: Multiplier for min/max thresholds (default: 2)
        window_size: Size of sliding window for min/max calculation (default: 30000)
        overlap: Overlap between windows (default: 10000)
    
    Returns:
        Processed data array with outliers set to zero
    """
    # Calculate number of windows
    num_windows = (data.shape[1] - window_size) // (window_size - overlap) + 1
    min_vals = np.zeros((data.shape[0], num_windows))
    max_vals = np.zeros((data.shape[0], num_windows))
    
    # Process each channel separately to get min/max values
    for ch in range(data.shape[0]):
        for i in range(num_windows):
            start = i * (window_size - overlap)
            end = start + window_size
            min_vals[ch,i] = np.min(data[ch,start:end])
            max_vals[ch,i] = np.max(data[ch,start:end])
    
    # Get mean of min and max values per channel
    avg_min_vals = np.mean(min_vals, axis=1)
    avg_max_vals = np.mean(max_vals, axis=1)
    
    # Apply clipping thresholds per channel
    for ch in range(data.shape[0]):
        # Create boolean masks for outlier points
        upper_outliers = data[ch,:] > clip_mult*avg_max_vals[ch]
        lower_outliers = data[ch,:] < clip_mult*avg_min_vals[ch]
        
        # Combine outlier masks
        outliers = upper_outliers | lower_outliers
        
        # Create shifted masks for ±1 window
        outliers_shifted_left = np.roll(outliers, 1)
        outliers_shifted_right = np.roll(outliers, -1)
        
        # Combine all masks to include the window
        final_mask = outliers | outliers_shifted_left | outliers_shifted_right
        
        # Set values to zero where mask is True
        data[ch, final_mask] = 0
    
    return data

# Grab the number of single units found from kilosort.log and append them to a summary txt file
def unit_summary(data_path, results_path, data_min, data_max, data_std, clip_mult, error=False):

    mouse_session = get_savedirs(data_path)
    savedir = results_path.parents[1]
    
    log_file = savedir / mouse_session / "kilosort4.log"
    output_file = savedir / "good_units.txt"

    with open(log_file, 'r') as file:
        content = file.read()
    
    # Use regex to find the number before "units"
    pattern = r'(\d{1,3}) units found with good refractory periods'
    match = re.search(pattern, content)

    if match and not error:
        # Extract the number from the first capture group
        num_units = match.group(1)
        
        # Append the number to the output file
        with open(output_file, 'a') as outfile:
            outfile.write(f"{mouse_session} - {num_units} units - min: {data_min} max: {data_max} std: {round(data_std, 3)}, clip_mult: {clip_mult}\n")
    elif error:
        with open(output_file, 'a') as outfile:
            outfile.write(f"{mouse_session} - Kilosort failed - min: {data_min} max: {data_max} std: {round(data_std, 3)}, clip_mult: {clip_mult}\n")
    else:
        with open(output_file, 'a') as outfile:
            outfile.write(f"{mouse_session} - No matching pattern found in the log file\n")

    print(f"Summary written to {output_file}")


"""
Functions for camera TTL
"""

def ttl_bool(data_path: str, results_path: str, sample_hz=30000, resample_hz=1000, save=True):
    data = np.load(data_path)

    #Resample data to 1000 Hz
    ttl_resample = data[::sample_hz//resample_hz]

    # Normalize to 0-1 range
    normalized = (ttl_resample - np.min(ttl_resample)) / (np.max(ttl_resample) - np.min(ttl_resample))
    
    # Rebuild ttl signal as boolean
    ttl_bool = ttl_resample > -30000

    if save:
        np.save(results_path, ttl_bool)
    return ttl_bool


def clean_camera_ttl(signal, threshold=-30000, min_frame_duration=20, min_frame_spacing=20):
    # Initial threshold
    binary = (signal < threshold).astype(int)
    print("Number of samples below threshold:", np.sum(binary))
    
    # Find potential frame boundaries
    transitions = np.diff(binary)
    starts = np.where(transitions == 1)[0]
    ends = np.where(transitions == -1)[0]
    print("Number of starts found:", len(starts))
    print("Number of ends found:", len(ends))
    if len(starts) > 0:
        print("First few start indices:", starts[:5])
    if len(ends) > 0:
        print("First few end indices:", ends[:5])
    
    # Ensure we have matching starts and ends
    if len(starts) == 0 or len(ends) == 0:
        print("No valid transitions found")
        return np.zeros_like(signal, dtype=int)
    
    if ends[0] < starts[0]:
        ends = ends[1:]
    if len(starts) > len(ends):
        starts = starts[:-1]
    
    #print("After matching, number of potential frames:", len(starts))
    
    # Filter based on duration and spacing
    valid_frames = []
    last_valid_end = -min_frame_spacing
    
    for start, end in zip(starts, ends):
        duration = end - start
        spacing = start - last_valid_end
        #print(f"Frame: start={start}, end={end}, duration={duration}, spacing={spacing}")
        
        if duration >= min_frame_duration and spacing >= min_frame_spacing:
            valid_frames.append((start, end))
            last_valid_end = end
    
    #print("Number of valid frames found:", len(valid_frames))
    
    # Create cleaned signal
    cleaned = np.zeros_like(signal, dtype=int)
    for start, end in valid_frames:
        cleaned[start:end] = 1
        
    return cleaned


def analyze_ttl_timing(signal, threshold=-25000):
    binary = (signal < threshold).astype(int)
    transitions = np.diff(binary)
    starts = np.where(transitions == 1)[0]
    ends = np.where(transitions == -1)[0]
    
    if len(starts) > 0 and len(ends) > 0:
        # Make sure we have matching pairs
        if ends[0] < starts[0]:
            ends = ends[1:]
        if len(starts) > len(ends):
            starts = starts[:-1]
            
        # Now calculate durations and spacings
        durations = ends - starts  # How long the signal is "on"
        spacings = starts[1:] - ends[:-1]  # Time between pulses
        
        print(f"Average pulse duration: {np.mean(durations):.2f} samples ({np.mean(durations)/1000*1000:.2f} ms)")
        print(f"Average spacing between pulses: {np.mean(spacings):.2f} samples ({np.mean(spacings)/1000*1000:.2f} ms)")
        print(f"Frame rate: {1000/np.mean(starts[1:] - starts[:-1]):.2f} fps")
        
        # Additional diagnostic info
        print(f"\nNumber of pulses analyzed: {len(durations)}")
        print(f"Duration range: {np.min(durations):.2f} to {np.max(durations):.2f} samples")
        print(f"Spacing range: {np.min(spacings):.2f} to {np.max(spacings):.2f} samples")



"""
Preprocessing and loading functions
"""


def load_ephys(ephys_file: str, num_samples: int = -1, start = 0, stop = 0, nchannels: int = 16, dtype = np.uint32, order = 'F') -> np.array:
    '''
    Load and reshape binary electrophysiology data into a NumPy array.

    This function is designed to read binary files containing electrophysiology 
    (ephys) data. It loads the specified number of samples from the file and 
    reshapes them into a 2D NumPy array, where each row represents a channel.

    Parameters:
    ephys_file (str): Path to the binary file containing electrophysiology data.
    num_samples (int): Number of samples to read from the file.
    nchannels (int, optional): Number of channels in the ephys data. Defaults to 16.

    Returns:
    np.array: A 2D NumPy array of the electrophysiology data, reshaped into
              (nchannels, number_of_samples_per_channel).
    '''
  
    # reading in binary data
    num_samples = num_samples * nchannels
    ephys_bin = np.fromfile(ephys_file, dtype=dtype, count = num_samples)
    
    # ensuring equal samples from each channel
    num_complete_sets = len(ephys_bin) // nchannels
    ephys_bin = ephys_bin[:num_complete_sets * nchannels]

    # reshape 1d array into nchannels x num_samples NumPy array
    ephys_data = np.reshape(ephys_bin, (nchannels, -1), order=order)

    # removing start seconds from beggining, and stop from end of signal; default is 0
    start = start * 30000
    stop = stop * 30000

    if stop == 0:
        ephys = ephys_data[:, start:]
    else:
        ephys = ephys_data[:, start: -stop]

    return ephys


def load_sniff_MATLAB(file: str) -> np.array:
    '''
    Loads a MATLAB file containing sniff data and returns a numpy array
    '''

    mat = loadmat(file)
    sniff_params = mat['sniff_params']

    # loading sniff parameters
    inhalation_times = sniff_params[:, 0]
    inhalation_voltage = sniff_params[:, 1]
    exhalation_times = sniff_params[:, 2]
    exhalation_voltage = sniff_params[:, 3]

    # bad sniffs are indicated by 0 value in exhalation_times
    bad_indices = np.where(exhalation_times == 0)


    # removing bad sniffs
    inhalation_times = np.delete(inhalation_times, bad_indices)
    inhalation_voltage = np.delete(inhalation_voltage, bad_indices)
    exhalation_times = np.delete(exhalation_times, bad_indices)
    exhalation_voltage = np.delete(exhalation_voltage, bad_indices)

    return inhalation_times.astype(np.int32), inhalation_voltage, exhalation_times.astype(np.int32), exhalation_voltage


def compute_spike_rates_sliding_window_by_region_smooth(kilosort_dir: str, sampling_rate: int, window_size: float = 1.0, step_size: float = 0.5, use_units: str = 'all', sigma: float = 2.5, zscore: bool = True):
    """
    Computes the spike rate for OB and HC units in sliding windows and applies Gaussian smoothing.

    Parameters:
    - kilosort_dir (str): Path to the Kilosort4 output directory.
    - sampling_rate (float): Sampling rate of the recording in Hz.
    - window_size (float): Size of the sliding window in seconds.
    - step_size (float): Step size between consecutive windows in seconds.
    - use_units (str): Specify which units to include ('all', 'good', 'mua', 'good/mua').
    - sigma (float): Standard deviation for Gaussian kernel smoothing.

    Returns:
    - spike_rate_matrix_OB (np.ndarray): Smoothed 2D array (OB units x windows) of spike rates (Hz).
    - spike_rate_matrix_HC (np.ndarray): Smoothed 2D array (HC units x windows) of spike rates (Hz).
    - time_bins (np.ndarray): Time points (seconds) corresponding to each window.
    - unit_ids_OB (np.ndarray): OB Unit IDs.
    - unit_ids_HC (np.ndarray): HC Unit IDs.
    """

    # Load spike times and cluster assignments
    spike_times_path = os.path.join(kilosort_dir, "spike_times.npy")
    spike_clusters_path = os.path.join(kilosort_dir, "spike_clusters.npy")  # Cluster assignments from Phy2 manual curation
    templates_path = os.path.join(kilosort_dir, "templates.npy")
    templates_ind_path = os.path.join(kilosort_dir, "templates_ind.npy")
    cluster_groups_path = os.path.join(kilosort_dir, "cluster_group.tsv")

    # Ensure all required files exist
    if not all(os.path.exists(p) for p in [spike_times_path, spike_clusters_path, templates_path, templates_ind_path, cluster_groups_path]):
        raise FileNotFoundError("Missing required Kilosort output files.")

    # Loading the data
    templates = np.load(templates_path)  # Shape: (nTemplates, nTimePoints, nChannels)
    templates_ind = np.load(templates_ind_path)  # Shape: (nTemplates, nChannels)
    spike_times = np.load(spike_times_path) / sampling_rate  # Convert to seconds
    spike_clusters = np.load(spike_clusters_path)
    cluster_groups = np.loadtxt(cluster_groups_path, dtype=str, skiprows=1, usecols=[1])

    # Find peak amplitude channel for each template and assign to unit
    peak_channels = np.argmax(np.max(np.abs(templates), axis=1), axis=1)
    unit_best_channels = {unit: templates_ind[unit, peak_channels[unit]] for unit in range(len(peak_channels))}
    
    # Filter units based on use_units parameter
    if use_units == 'all':
        unit_best_channels = unit_best_channels
    elif use_units == 'good':
        unit_indices = np.where(cluster_groups == 'good')[0]
        unit_best_channels = {unit: unit_best_channels[unit] for unit in unit_indices}
    elif use_units == 'mua':
        unit_indices = np.where(cluster_groups == 'mua')[0]
        unit_best_channels = {unit: unit_best_channels[unit] for unit in unit_indices}
    elif use_units == 'good/mua':
        unit_indices = np.where(np.isin(cluster_groups, ['good', 'mua']))[0]
        unit_best_channels = {unit: unit_best_channels[unit] for unit in unit_indices}
    elif use_units == 'noise':
        unit_indices = np.where(cluster_groups == 'noise')[0]
        unit_best_channels = {unit: unit_best_channels[unit] for unit in unit_indices}

    # Get total duration of the recording
    recording_duration = np.max(spike_times)

    # Define time windows
    time_bins = np.arange(0, recording_duration - window_size, step_size)
    num_windows = len(time_bins)

    # Separate OB and HC units
    hc_units = np.array([unit for unit, ch in unit_best_channels.items() if ch in range(0, 16)])
    ob_units = np.array([unit for unit, ch in unit_best_channels.items() if ch in range(16, 32)])
    num_ob_units = len(ob_units)
    num_hc_units = len(hc_units)

    # Initialize spike rate matrices
    spike_rate_matrix_OB = np.zeros((num_ob_units, num_windows))
    spike_rate_matrix_HC = np.zeros((num_hc_units, num_windows))

    # Compute spike counts in each window
    for i, t_start in tqdm(enumerate(time_bins), total=num_windows, desc="Computing spike rates"):
        t_end = t_start + window_size

        # Find spikes in this window
        in_window = (spike_times >= t_start) & (spike_times < t_end)
        spike_clusters_in_window = spike_clusters[in_window]

        # Compute spike rates for OB
        for j, unit in enumerate(ob_units):
            spike_rate_matrix_OB[j, i] = np.sum(spike_clusters_in_window == unit) / window_size  # Hz

        # Compute spike rates for HC
        for j, unit in enumerate(hc_units):
            spike_rate_matrix_HC[j, i] = np.sum(spike_clusters_in_window == unit) / window_size  # Hz

    # Apply Gaussian smoothing
    for j in range(num_ob_units):
        spike_rate_matrix_OB[j, :] = gaussian_filter1d(spike_rate_matrix_OB[j, :], sigma=sigma)

    for j in range(num_hc_units):
        spike_rate_matrix_HC[j, :] = gaussian_filter1d(spike_rate_matrix_HC[j, :], sigma=sigma)

    # Apply Z-scoring (optional)
    if zscore:
        def z_score(matrix):
            mean_firing = np.mean(matrix, axis=1, keepdims=True)
            std_firing = np.std(matrix, axis=1, keepdims=True)
            std_firing[std_firing == 0] = 1  # Prevent division by zero
            return (matrix - mean_firing) / std_firing

        spike_rate_matrix_OB = z_score(spike_rate_matrix_OB)
        spike_rate_matrix_HC = z_score(spike_rate_matrix_HC)

    return spike_rate_matrix_OB, spike_rate_matrix_HC, time_bins, ob_units, hc_units



"""
LFP Analysis
"""

def sniff_lock_lfp(locs: np.array, ephys: np.array, window_size = 1000, nsniffs = 512, beg = 3000, method = 'zscore') -> np.array:
    '''
    Aligns local field potential (LFP) signals with sniff inhalation times and constructs a 3D array of z-scored LFP activity.

    This function identifies segments of LFP signals corresponding to inhalation times (specified by 'locs') and 
    standardizes these segments across channels. The output is a 3D array where each 'slice' corresponds to the LFP 
    activity surrounding a single sniff event, with data from all channels.

    Parameters:
    locs (np.array): Array of sniff inhalation times (indices).
    ephys (np.array): 2D array of electrophysiological data with shape (nchannels, number_of_samples).
    nchannels (int, optional): Number of channels in the ephys data. Defaults to 16.
    window_size (int, optional): The size of the window around each sniff event to consider for LFP activity. Defaults to 1000.
    nsniffs (int, optional): Number of sniff events to process. Defaults to 512.
    beg (int, optional): Starting index to begin looking for sniff events. Defaults to 3000.

    Returns:
    sniff_activity (np.array): A 3D NumPy array with shape (nsniffs, window_size, nchannels). Each 'slice' of this array 
              represents the z-scored LFP activity around a single sniff event for all channels.
    loc_set (np.array): An array of indices where inhalation peaks are located.

    Raises:
    ValueError: If the 'locs' array does not contain enough data after the specified 'beg' index for the required number of sniffs.
    '''


    # finding number of channels
    nchannels = ephys.shape[0]


    # finding the set of inhalation times to use
    if nsniffs == 'all':
        loc_set = locs[5:-5]
        nsniffs = len(loc_set)
    elif isinstance(nsniffs, int):
        first_loc = np.argmax(locs >= beg)
        loc_set = locs[first_loc: first_loc + nsniffs]
    else:
        raise ValueError("nsniffs must be either 'all' or an integer.")

    # checking if locs array has enough data for the specified range
    if isinstance(nsniffs, int):
        if len(loc_set) < nsniffs:
            raise ValueError("locs array does not have enough data for the specified range.")
        
    # propogates an nx2 array containing times half the window size in both directions from inhalation times
    windows = np.zeros((nsniffs, 2), dtype=int)
    for ii in range(nsniffs):
        win_beg = loc_set[ii] - round(window_size/2)
        win_end = loc_set[ii] + round(window_size/2)
        windows[ii] = [win_beg, win_end]

    if method == 'zscore':
        # finds and saves zscored ephys data from each channel for each inhalaion locked time window
        sniff_activity = np.zeros((nchannels, nsniffs, window_size))
        for ii in range(nsniffs):
            for ch in range(nchannels):
                win_beg, win_end = windows[ii]
                data = ephys[ch, win_beg:win_end]
                if len(data) == 0:
                    zscore_data = np.zeros(window_size)
                elif len(data) < window_size:
                    data = np.pad(data, (0, window_size - len(data)), mode = 'constant', constant_values = 0)
                    data_mean = np.mean(data)
                    data_std = np.std(data)
                    zscore_data = (data - data_mean) / data_std
                else:
                    data_mean = np.mean(data)
                    data_std = np.std(data)
                    zscore_data = (data - data_mean) / data_std
                sniff_activity[ch,ii,:] = zscore_data

    if method == 'custom':
        # Same as z-score but does not subtract the mean
        sniff_activity = np.zeros((nchannels, nsniffs, window_size))
        for ii in range(nsniffs):
            for ch in range(nchannels):
                win_beg, win_end = windows[ii]
                data = ephys[ch, win_beg:win_end]
                if len(data) == 0:
                    custom_data = np.zeros(window_size)
                elif len(data) < window_size:
                    data = np.pad(data, (0, window_size - len(data)), mode = 'constant', constant_values = 0)
                    data_std = np.std(data)
                    custom_data = (data) / data_std
                else:
                    data_std = np.std(data)
                    custom_data = (data) / data_std
                sniff_activity[ch,ii,:] = custom_data


    elif method == 'none':
        sniff_activity = np.zeros((nchannels, nsniffs, window_size))
        for ii in range(nsniffs):
            for ch in range(nchannels):
                win_beg, win_end = windows[ii]
                data = ephys[ch, win_beg:win_end]
                if len(data) < window_size:
                    data = np.pad(data, (0, window_size - len(data)), mode = 'constant', constant_values = 0)
                    print('!!! padding !!!')
                sniff_activity[ch,ii,:] = data

    return sniff_activity, loc_set



def sort_lfp(sniff_activity, locs):
    '''sorts the sniff locked lfp trace by sniff frequency'''

    # finding data shape
    nchannels = sniff_activity.shape[0]
    nsniffs = sniff_activity.shape[1]
    window_size = sniff_activity.shape[2]
    
    sorted_activity = np.zeros((nchannels, nsniffs-1, window_size))
    
    # finding sniff frequencies by inhalation time differences (we lose the last sniff)
    freqs = np.diff(locs)

    # sorting the ephys data and frequency values according to these times
    sort_indices = np.argsort(freqs)
    sorted_activity[:, :, :] = sniff_activity[:, sort_indices, :]
    sorted_freqs = freqs[sort_indices]
    sorted_freqs = 1 / (sorted_freqs / 1000)
    new_locs = locs[sort_indices]


    return sorted_activity, sorted_freqs, new_locs, sort_indices


def build_raster(lfp: np.array, inh: np.array, exh: np.array, save_path: str, filter: tuple = ('bandpass', [2, 12]), window_size: int = 2000, f: int = 1000, dark_mode: bool = False):

    custom_x=False
    solojazz = load_colormap()

    if dark_mode:
        plt.style.use('dark_background')
    else:
        font_color = 'black'
        plt.style.use('default')

    # plot with arial font
    plt.rcParams['font.family'] = 'Arial'
    sns.set_context('paper', font_scale=4)
    
    # filtering the LFP if filter is not a string
    if not isinstance(filter, str):
        sos = signal.butter(4, filter[1], filter[0], fs = 1000, output = 'sos')
        lfp = signal.sosfiltfilt(sos, lfp)

        if filter[1] == 'bandstop':
            sos = signal.butter(4, 'highpass', 2, fs = 1000, output = 'sos')
            lfp = signal.sosfiltfilt(sos, lfp)

    # setting the parameters for the filter
    if filter == 'gamma':
        cmap = 'Greys'
    else:
        cmap = solojazz
    vmax = 2

    # extract the instantaneous frequency
    freqs = f / np.diff(inh)
    inh = inh[:-1]
    exh = exh[:-1]

    # cleaning frequency data
    bad_indicies = np.where((freqs < 0) | (freqs > 16))
    freqs = np.delete(freqs, bad_indicies)
    inh = np.delete(inh, bad_indicies)

    # aligning the LFPs to sniff times
    print('Sniff locking LFP')
    sniff_activity, loc_set = sniff_lock_lfp(inh, lfp, method = 'custom', window_size=window_size, nsniffs='all')
    if sniff_activity.shape[1] < 10:
        print('Not enough sniffs')
        return
    
    sorted_activity, sorted_freqs, loc_set, sort_indices = sort_lfp(sniff_activity, loc_set)


    # looping through the channels and plotting the data
    for ch in range(sorted_activity.shape[0]):
        if filter == 'gamma':
            vmin = 0
            vmax = 4
        else:
            vmin = -vmax

       
        # keeping only the middle 1000 points of the sorted_activity
        middle_index = window_size // 2
        sorted_activity = sorted_activity[:, :, middle_index - 500: middle_index + 500]


        # plot inhalation time aligned, time-domain LFP activity
        _, ax = plt.subplots(figsize = (10, 8))
        sns.heatmap(sorted_activity[ch, :, :], cmap = cmap, robust = True, vmin = vmin, vmax = vmax, rasterized = True)
        
        # plotting line to indicate the current and the next inhalation
        plt.axvline(500, color = 'black', alpha = 0.8)
        next_inhale = np.zeros(len(sorted_freqs))
        for i in range(len(sorted_freqs)):
            next_inhale[i] = 500 + 1000 // sorted_freqs[i]
        sns.lineplot(x = next_inhale, y = np.arange(len(sorted_freqs)), color = 'black', alpha = 0.8)
        
        #only showing 3 ticks on colorbar, the top middle and bottom
        cbar = ax.collections[0].colorbar
        cbar.set_ticks([vmin, (vmin + vmax) // 2, vmax])
        cbar.set_ticklabels([f'{int(vmin)}', f'{int((vmin + vmax) // 2)}', f'{int(vmax)}'])
        cbar.ax.yaxis.set_tick_params(length=0)

        # adjusting the axis
        if custom_x:
            ax.set_xticks([0, 1000])
            ax.set_xticklabels(['', ''])
        else:
            ax.set_xticks([0, window_size //2, window_size])
            ax.set_xticklabels([-window_size // 2, 0, window_size // 2], rotation = 0)

        # remove y ticks and labels
        ax.set_yticks([])

        if dark_mode:
            # add labels
            plt.xlabel('Time (ms)')
            plt.ylabel('Sniff number')
            plt.title(f'Channel {ch} - {filter[0]} filter')
            cbar.set_label('LFP amplitude (z)', rotation=270, labelpad=20)


        plt.savefig(os.path.join(save_path, f'Channel_{ch}_LFP.png'), dpi = 300)
        plt.close()



def process_filter(args):

    matplotlib.use('Agg')

    ephys, inh_use, exh_use, filter_info, save_path, dark_mode = args
    filter_name, filter_params = filter_info
    
    current_save_path = os.path.join(save_path, filter_name)
    os.makedirs(current_save_path, exist_ok=True)

    # Filter the ephys data
    print(f"Filtering with {filter_name} filter")
    if filter_params[0] is not None:
        sos = signal.butter(4, filter_params[1], filter_params[0], fs=1000, output='sos')
        ephys_filtered = signal.sosfiltfilt(sos, ephys)
    else:
        ephys_filtered = ephys

    if filter_name == 'gamma':
        ephys_filtered = np.abs(signal.hilbert(ephys_filtered))

    # Build raster
    print(f"Building raster")
    build_raster(ephys_filtered, inh_use, exh_use, current_save_path, 
                            filter=filter_name, window_size=1000, f=1000, dark_mode=dark_mode)



def build_sniff_rasters(ephys: np.array, inh: np.array, exh: np.array, save_path: str, max_workers: int = 2, dark_mode: bool = False):
    
    filters = {
        'raw': (None, None),
        'theta': ('bandpass', (2, 12)),
        'beta': ('bandpass', (18, 30)),
        'gamma': ('bandpass', (65, 100))}

    # Create processing tasks
    tasks = []
    for filter_name, filter_params in filters.items():
        tasks.append((ephys, inh, exh, (filter_name, filter_params), save_path, dark_mode))

    # Process in parallel
    with ProcessPoolExecutor(max_workers = max_workers) as executor:
        list(executor.map(process_filter, tasks))



def compute_multichannel_psd(lfp_data, fs=1000, nperseg=4000, noverlap=2000, method="mean_psd"):
    """
    Computes the Power Spectral Density (PSD) across multiple LFP channels.

    Parameters:
    - lfp_data (np.ndarray): 2D array of shape (channels, time) containing LFP data.
    - fs (int): Sampling rate in Hz (default = 1000 Hz).
    - nperseg (int): Segment length for Welch's method (default = 4s segments at 1kHz = 4000 samples).
    - noverlap (int): Overlapping samples between segments (default = 50% overlap = 2000 samples).
    - method (str): Method for combining PSD across channels:
        - "mean_psd": Compute PSD per channel and average across channels (default).
        - "psd_of_mean": Compute the PSD of the averaged LFP signal.
        - "per_channel": Return PSD for each channel separately.

    Returns:
    - freqs (np.ndarray): Frequency values in Hz.
    - psd (np.ndarray): PSD values depending on the chosen method.
    """

    # Ensure input is a NumPy array
    lfp_data = np.asarray(lfp_data)

    # Check if LFP data has 16 channels
    if lfp_data.shape[0] != 16:
        raise ValueError(f"Expected 16 channels, but got {lfp_data.shape[0]} channels.")

    # Initialize storage for per-channel PSD
    psds = []

    # Compute PSD for each channel
    for ch in range(lfp_data.shape[0]):
        freqs, psd = signal.welch(lfp_data[ch], fs=fs, nperseg=nperseg, noverlap=noverlap, window='hann')
        psds.append(psd)

    psds = np.array(psds)  # Shape: (16, frequencies)

    if method == "mean_psd":
        # Compute the average PSD across all channels
        psd_avg = np.mean(psds, axis=0)
        return freqs, psd_avg

    elif method == "psd_of_mean":
        # Compute PSD of the averaged LFP signal
        lfp_mean = np.mean(lfp_data, axis=0)
        freqs, psd_mean = signal.welch(lfp_mean, fs=fs, nperseg=nperseg, noverlap=noverlap, window='hann')
        return freqs, psd_mean

    elif method == "per_channel":
        # Return PSD for each channel separately
        return freqs, psds

    else:
        raise ValueError("Invalid method. Choose 'mean_psd', 'psd_of_mean', or 'per_channel'.")



"""
Plotting
"""

def load_colormap(cmap = 'smoothjazz'):
    # Load the .mat file
    if cmap == 'smoothjazz':
        mat_file = "E:\\Sid_LFP\\solojazz.mat"
        data = loadmat(mat_file)["pmc"]
    elif cmap == 'burningchrome':
        csv_file = "E:\\Sid_LFP\\burning_chrome.csv"
        data = pd.read_csv(csv_file, header=None).values
        
    
    # Create a custom colormap
    colormap = mcolors.ListedColormap(data, name="solojazz")

    return colormap


def plot_spike_rates(time_bins: np.ndarray, rates_OB: np.ndarray, rates_HC: np.ndarray, 
                     ob_units: np.ndarray, hc_units: np.ndarray, dark_mode: bool = True, 
                     global_font: str = "Arial", global_font_size: int = 14, 
                     show: bool = True, save_path: str = None, normalized: bool = False):
    """
    Plots spike rates for OB and HC units using Plotly as two separate heatmaps stacked vertically with a transparent background.
    Ensures both heatmaps share the same color scale and only one colorbar is displayed.
    """

    # Determine font and axis color
    font_color = "white" if dark_mode else "black"

    if normalized:
        colorbar_title = "Normalized spike rate (z)"
    else:
        colorbar_title = "Spike rate (Hz)"

    # Determine shared color scale range
    vmin = min(np.min(rates_OB), np.min(rates_HC))
    vmax = min(max(np.max(rates_OB), np.max(rates_HC)), 20) if normalized else max(np.max(rates_OB), np.max(rates_HC))

    # Create subplot layout
    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True, 
        subplot_titles=["OB Units", "HC Units"], 
        vertical_spacing=0.1
    )

    # Create OB heatmap
    heatmap_OB = go.Heatmap(
        z=rates_OB,
        x=time_bins,
        y=np.arange(len(ob_units)),
        colorscale='Magma',
        zmin=vmin, zmax=vmax,  
        colorbar=dict(title=dict(text=colorbar_title, font=dict(family=global_font, size=global_font_size))),
        name='OB Units',
    )

    # Create HC heatmap without colorbar
    heatmap_HC = go.Heatmap(
        z=rates_HC,
        x=time_bins,
        y=np.arange(len(hc_units)),
        colorscale='Magma',
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
            text="Spike rates in olfactory bulb and hippocampus units",
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
        fig.write_image(save_path + '.png')
        fig.write_html(save_path + '.html')


def plot_sniff_frequencies(time_bins: np.ndarray, mean_freqs: np.ndarray, 
                           dark_mode: bool = True, global_font: str = "Arial", 
                           global_font_size: int = 14, log_y: str = None, 
                           show: bool = True, save_path: str = None):
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
        x=time_bins / 60,  # Convert time to minutes
        y=mean_freqs,
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
        fig.write_image(save_path + '.png')
        fig.write_html(save_path + '.html')


def plot_embedding_2d(embedding_OB: np.ndarray, label: np.ndarray, region: str, method: str, 
                      dark_mode: bool = True, global_font: str = "Arial", global_font_size: int = 14,
                      save_path: str = None, show: bool = True):
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
        color=label, range_color=[2, 12],
        color_continuous_scale='plasma', 
        labels={'color': 'Sniffs per second'},
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
        fig.write_image(save_path + '.png')


def plot_embedding_3d(embedding: np.ndarray, labels: np.ndarray, region: str, method: str, 
                      dark_mode: bool = True, global_font: str = "Arial", global_font_size: int = 14,
                      save_path: str = None, show: bool = True):
    """
    Plots a 3D embedding using Plotly with transparent background and an orthogonal axis.
    """
    
    # Determine colors based on dark mode
    font_color = "white" if dark_mode else "black"
    background_color = "black" if dark_mode else "white"
    
    # Create scatter plot
    fig = px.scatter_3d(
        x=embedding[:, 0], y=embedding[:, 1], z=embedding[:, 2], range_color=[2, 12],
        color=labels, title=f"{method} embedding of {region} spike rates", labels={'color': 'Sniffs per second'}, 
        color_continuous_scale='plasma'
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
    freq_range = [1, 100]



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
    
    plt.xticks([2, 12, 18, 30, 65, 100], [2, 12, 18, 30, 65, 100])

    plt.savefig(os.path.join(save_dir, f'Combined_psd_logx_letters_ticks.png'), format='png', dpi=600)
    plt.savefig(os.path.join(save_dir, f'Combined_psd_logx_letters_ticks.svg'), format='svg')
    plt.close()

