import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d
from scipy.io import loadmat
from concurrent.futures import ProcessPoolExecutor
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.signal as signal
from plotting import load_colormap
import h5py
import os



"""
Most basic data loading and preprocessing functions
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


def compute_sniff_freqs_bins(sniff_params_file:str, time_bins:np.ndarray, window_size:float, sfs:int):
    
    inhalation_times, _, _, _ = load_sniff_MATLAB(sniff_params_file)
    inhalation_times = inhalation_times / sfs  # Convert to seconds

    # Compute sniff frequencies
    freqs = 1 / np.diff(inhalation_times) # Compute frequencies

    # Remove unrealistic frequencies
    bad_indices = np.where((freqs > 16) | (freqs < 0.8))[0]  # Fixed OR condition
    freqs = np.delete(freqs, bad_indices)
    inhalation_times = np.delete(inhalation_times[:-1], bad_indices)  # Fix slicing

    # Compute mean sniff frequency in each time bin
    mean_freqs = np.full(len(time_bins), np.nan)  # Initialize with NaNs
    inhalation_latencies = np.full(len(time_bins), np.nan)  # Initialize with NaNs

    for i, t_start in enumerate(time_bins):
        t_end = t_start + window_size
        middle = t_start + window_size / 2
        in_window = (inhalation_times >= t_start) & (inhalation_times < t_end)
        
        # computing the latency of the middle of the time bin from the last inhalation time
        last_inh_time = inhalation_times[inhalation_times < middle][-1] if np.any(inhalation_times < middle) else np.nan
        inhalation_latencies[i] = middle - last_inh_time

        
        if np.any(in_window):  # Ensure there are valid inhalation times
            mean_freqs[i] = np.nanmean(freqs[in_window])  # Avoid NaN issues

    return mean_freqs, inhalation_latencies


def load_behavior(behavior_file: str, tracking_file: str = None) -> pd.DataFrame:

    """
    Load and preprocess behavioral tracking data from a CSV file.
    
    This function loads movement tracking data, normalizes spatial coordinates by
    centering them around zero, calculates velocity components and overall speed,
    and returns a filtered dataframe with relevant movement metrics.
    
    Parameters
    ----------
    behavior_file : str
        Path to the CSV file containing behavioral tracking data. The file should
        include columns for 'centroid_x', 'centroid_y', and 'timestamps_ms'.
        
    Returns
    -------
    events : pandas.DataFrame
        Processed dataframe containing the following columns:
        - 'time': Original time values
        - 'centroid_x': Zero-centered x coordinates 
        - 'centroid_y': Zero-centered y coordinates
        - 'velocity_x': Rate of change in x position
        - 'velocity_y': Rate of change in y position
        - 'speed': Overall movement speed (Euclidean norm of velocity components)
        - 'timestamps_ms': Timestamps in milliseconds
        
    Notes
    -----
    - Position coordinates are normalized by subtracting the mean to center around zero
    - Velocity is calculated using first-order differences (current - previous position)
    - The first velocity value uses the first position value as the "previous" position
    - Speed is calculated as the Euclidean distance between consecutive positions
    """

    # Load the behavior data
    events = pd.read_csv(os.path.join(behavior_file, 'events.csv'))

    if tracking_file:
        # Load the SLEAP tracking data from the HDF5 file
        f = h5py.File(tracking_file, 'r')
        nose = f['tracks'][:].T[:, 0, :]
        nose = nose[:np.shape(events)[0], :]
        mean_x, mean_y = np.nanmean(nose[:, 0]), np.nanmean(nose[:, 1])
        events['position_x'] = nose[:, 0] - mean_x
        events['position_y'] = nose[:, 1] - mean_y
        
    else:
        # zero-mean normalize the x and y coordinates
        mean_x, mean_y = np.nanmean(events['centroid_x']), np.nanmean(events['centroid_y'])
        events['position_x'] = events['centroid_x'] - mean_x
        events['position_y'] = events['centroid_y'] - mean_y

    # Estimating velocity and speed
    events['velocity_x'] = np.diff(events['position_x'], prepend=events['position_x'].iloc[0])
    events['velocity_y'] = np.diff(events['position_y'], prepend=events['position_y'].iloc[0])
    events['speed'] = np.sqrt(events['velocity_x']**2 + events['velocity_y']**2)



    # keeping only the columns we need
    events = events[['position_x', 'position_y', 'velocity_x', 'velocity_y', 'reward_state', 'speed', 'timestamp_ms']]
    return events




"""
Spike analysis
"""


def compute_spike_rates_sliding_window_by_region_smooth(kilosort_dir: str, sampling_rate: int, window_size: float = 1.0, step_size: float = 0.5, use_units: str = 'all', sigma: float = 2.5, zscore: bool = True):
    
    """
    Compute smoothed spike rates for neural units in OB (olfactory bulb) and HC (hippocampus) regions 
    using a sliding window approach from Kilosort output data.
    
    This function processes spike times and cluster assignments from Kilosort/Phy2, separates units by 
    brain region based on channel mapping, calculates firing rates within sliding time windows, and 
    applies Gaussian smoothing. Optionally, z-scoring can be applied to normalize firing rates.
    
    Parameters
    ----------
    kilosort_dir : str
        Path to the directory containing Kilosort output files.
    sampling_rate : int
        Sampling rate of the recording in Hz.
    window_size : float, optional
        Size of the sliding window in seconds, default is 1.0.
    step_size : float, optional
        Step size for sliding window advancement in seconds, default is 0.5.
    use_units : str, optional
        Filter for unit types to include:
        - 'all': Include all units
        - 'good': Include only good units
        - 'mua': Include only multi-unit activity
        - 'good/mua': Include both good units and multi-unit activity
        - 'noise': Include only noise units
        Default is 'all'.
    sigma : float, optional
        Standard deviation for Gaussian smoothing kernel, default is 2.5.
    zscore : bool, optional
        Whether to z-score the firing rates, default is True.
    
    Returns
    -------
    spike_rate_matrix_OB : ndarray
        Matrix of spike rates for OB units (shape: num_OB_units × num_windows).
    spike_rate_matrix_HC : ndarray
        Matrix of spike rates for HC units (shape: num_HC_units × num_windows).
    time_bins : ndarray
        Array of starting times for each window.
    ob_units : ndarray
        Array of unit IDs for OB region.
    hc_units : ndarray
        Array of unit IDs for HC region.
    
    Notes
    -----
    - OB units are assumed to be on channels 16-31
    - HC units are assumed to be on channels 0-15
    - Firing rates are computed in Hz (spikes per second)
    
    Raises
    ------
    FileNotFoundError
        If any required Kilosort output files are missing.
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
    for i, t_start in enumerate(time_bins):
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


def compute_spike_latency(kilosort_dir: str, sampling_rate: int, use_units: str = 'all'):
    """
    Computes the time since last spike with 1ms fixed bins for neural units.
    
    Parameters:
    -----------
    kilosort_dir : str
        Directory containing Kilosort output files
    sampling_rate : int
        Sampling rate in Hz
    use_units : str, optional
        Which units to include: 'all', 'good', 'mua', 'good/mua', or 'noise'
    
    Returns:
    --------
    spike_latency_matrix_OB : numpy.ndarray
        Matrix of spike latencies for OB units (1ms resolution)
    spike_latency_matrix_HC : numpy.ndarray
        Matrix of spike latencies for HC units (1ms resolution)
    time_bins : numpy.ndarray
        Array of time points (in seconds) corresponding to each 1ms bin
    ob_units : numpy.ndarray
        Array of OB unit IDs
    hc_units : numpy.ndarray
        Array of HC unit IDs
    """
    # Load spike times and cluster assignments
    spike_times_path = os.path.join(kilosort_dir, "spike_times.npy")
    spike_clusters_path = os.path.join(kilosort_dir, "spike_clusters.npy")
    templates_path = os.path.join(kilosort_dir, "templates.npy")
    templates_ind_path = os.path.join(kilosort_dir, "templates_ind.npy")
    cluster_groups_path = os.path.join(kilosort_dir, "cluster_group.tsv")

    # Ensure all required files exist
    if not all(os.path.exists(p) for p in [spike_times_path, spike_clusters_path, templates_path, templates_ind_path, cluster_groups_path]):
        raise FileNotFoundError("Missing required Kilosort output files.")

    # Loading the data
    templates = np.load(templates_path)
    templates_ind = np.load(templates_ind_path)
    spike_times_samples = np.load(spike_times_path)  
    spike_clusters = np.load(spike_clusters_path)
    
    # Convert spike times to milliseconds
    spike_times_ms = spike_times_samples * (1000 / sampling_rate)
    
    # Read cluster groups (good, mua, noise)
    cluster_groups_df = pd.read_csv(cluster_groups_path, sep='\t')
    cluster_ids = cluster_groups_df['cluster_id'].values
    cluster_labels = cluster_groups_df['KSLabel'].values
    cluster_groups = {cluster_id: label for cluster_id, label in zip(cluster_ids, cluster_labels)}

    # Find peak amplitude channel for each template and assign to unit
    peak_channels = np.argmax(np.max(np.abs(templates), axis=1), axis=1)
    unit_best_channels = {unit: templates_ind[unit, peak_channels[unit]] for unit in range(len(peak_channels))}
    
    # Filter units based on use_units parameter
    filtered_units = set()
    if use_units == 'all':
        filtered_units = set(unit_best_channels.keys())
    elif use_units == 'good':
        filtered_units = {unit for unit in unit_best_channels if cluster_groups.get(unit) == 'good'}
    elif use_units == 'mua':
        filtered_units = {unit for unit in unit_best_channels if cluster_groups.get(unit) == 'mua'}
    elif use_units == 'good/mua':
        filtered_units = {unit for unit in unit_best_channels if cluster_groups.get(unit) in ['good', 'mua']}
    elif use_units == 'noise':
        filtered_units = {unit for unit in unit_best_channels if cluster_groups.get(unit) == 'noise'}

    # Keep only filtered units
    unit_best_channels = {unit: ch for unit, ch in unit_best_channels.items() if unit in filtered_units}

    # Get total duration of the recording in milliseconds
    recording_duration_ms = int(np.ceil(np.max(spike_times_ms)))
    
    # Create 1ms bins
    num_bins = recording_duration_ms
    time_bins = np.arange(num_bins) / 1000.0  # Convert to seconds for compatibility
    
    # Separate OB and HC units
    hc_units = np.array([unit for unit, ch in unit_best_channels.items() if ch in range(0, 16)])
    ob_units = np.array([unit for unit, ch in unit_best_channels.items() if ch in range(16, 32)])
    num_ob_units = len(ob_units)
    num_hc_units = len(hc_units)

    # Initialize latency matrices
    spike_latency_matrix_OB = np.full((num_ob_units, num_bins), np.nan)
    spike_latency_matrix_HC = np.full((num_hc_units, num_bins), np.nan)
    
    # For each unit, pre-compute the most recent spike before each millisecond
    for j, unit in enumerate(ob_units):
        # Get spike times for this unit in ms
        unit_spikes_ms = spike_times_ms[spike_clusters == unit]
        
        if len(unit_spikes_ms) > 0:
            # Round spike times to nearest ms
            unit_spikes_ms = np.floor(unit_spikes_ms).astype(int)
            
            # For each 1ms bin, find the most recent spike
            last_spike_time = -1  # Initialize to -1 (no previous spike)
            
            # Create sparse vector of spike times 
            # (1 at spike times, 0 elsewhere)
            spike_bins = np.zeros(num_bins, dtype=bool)
            valid_indices = unit_spikes_ms[(unit_spikes_ms >= 0) & (unit_spikes_ms < num_bins)]
            if len(valid_indices) > 0:
                spike_bins[valid_indices] = True
            
            # Iterate through each bin
            for i in range(num_bins):
                if spike_bins[i]:
                    # If there's a spike in this bin, latency is 0
                    last_spike_time = i
                    spike_latency_matrix_OB[j, i] = 0
                elif last_spike_time >= 0:
                    # If no spike in this bin but there was a previous spike
                    spike_latency_matrix_OB[j, i] = (i - last_spike_time) / 1000.0  # Convert to seconds
    
    # Same for HC units
    for j, unit in enumerate(hc_units):
        # Get spike times for this unit in ms
        unit_spikes_ms = spike_times_ms[spike_clusters == unit]
        
        if len(unit_spikes_ms) > 0:
            # Round spike times to nearest ms
            unit_spikes_ms = np.floor(unit_spikes_ms).astype(int)
            
            # For each 1ms bin, find the most recent spike
            last_spike_time = -1  # Initialize to -1 (no previous spike)
            
            # Create sparse vector of spike times 
            spike_bins = np.zeros(num_bins, dtype=bool)
            valid_indices = unit_spikes_ms[(unit_spikes_ms >= 0) & (unit_spikes_ms < num_bins)]
            if len(valid_indices) > 0:
                spike_bins[valid_indices] = True
            
            # Iterate through each bin
            for i in range(num_bins):
                if spike_bins[i]:
                    # If there's a spike in this bin, latency is 0
                    last_spike_time = i
                    spike_latency_matrix_HC[j, i] = 0
                elif last_spike_time >= 0:
                    # If no spike in this bin but there was a previous spike
                    spike_latency_matrix_HC[j, i] = (i - last_spike_time) / 1000.0  # Convert to seconds
    
    return spike_latency_matrix_OB, spike_latency_matrix_HC, time_bins, ob_units, hc_units


def align_brain_and_behavior(events: pd.DataFrame, spike_rates: np.ndarray, units: np.ndarray, time_bins: np.ndarray, window_size: float = 0.1, speed_threshold: float = 100, interp_method = 'linear', order = None):
    
    """
    Align neural spike rate data with behavioral tracking data using time windows.
    
    This function matches neural activity from spike rates with behavioral metrics (position, velocity, speed)
    by finding the closest behavioral event to the middle of each time bin. It creates a unified dataframe
    containing both neural and behavioral data, removes outliers based on speed threshold, and interpolates
    missing values.
    
    Parameters
    ----------
    events : pd.DataFrame
        Behavioral tracking data containing columns:
        - 'timestamps_ms': Timestamps in milliseconds
        - 'centroid_x', 'centroid_y': Position coordinates
        - 'velocity_x', 'velocity_y': Velocity components
        - 'speed': Overall movement speed
    
    spike_rates : np.ndarray
        Matrix of spike rates with shape (n_units, n_time_bins).
    
    units : np.ndarray
        Array of unit IDs corresponding to rows in spike_rates.
    
    time_bins : np.ndarray
        Array of starting times for each time bin in seconds.
    
    window_size : float, optional
        Size of each time window in seconds, default is 0.1.
    
    speed_threshold : float, optional
        Threshold for removing speed outliers, expressed as multiplier of standard deviation, 
        default is 4.0 (values > 4 × std are treated as outliers).
    
    Returns
    -------
    pd.DataFrame
        Combined dataframe with aligned neural and behavioral data containing:
        - Unit columns: Spike rates for each neural unit
        - 'x', 'y': Position coordinates
        - 'v_x', 'v_y': Velocity components
        - 'speed': Movement speed
        - 'time': Time bin start times
        
    Notes
    -----
    - For each time bin, the behavioral event closest to the middle of the bin is selected
    - Speed outliers are removed using a threshold based on standard deviation
    - Missing values are interpolated using linear interpolation
    - Rows with missing behavioral data (typically at beginning/end of recording) are removed
    """

    # Initialize arrays for holding aligned data
    mean_positions_x = np.full(len(time_bins), np.nan)
    mean_positions_y = np.full(len(time_bins), np.nan)
    mean_velocities_x = np.full(len(time_bins), np.nan)
    mean_velocities_y = np.full(len(time_bins), np.nan)
    mean_speeds = np.full(len(time_bins), np.nan)
    mean_rewards = np.full(len(time_bins), np.nan)

    # getting event times in seconds
    event_times = events['timestamp_ms'].values / 1000

    # Calculate mean behavior in each time bin
    for i, t_start in enumerate(time_bins):
        t_end = t_start + window_size
        middle = t_start + window_size / 2

        if np.any(event_times < middle):
            nearest_event_index = np.argmin(np.abs(event_times - middle))
            mean_positions_x[i] = events['position_x'].iloc[nearest_event_index]
            mean_positions_y[i] = events['position_y'].iloc[nearest_event_index]
            mean_velocities_x[i] = events['velocity_x'].iloc[nearest_event_index]
            mean_velocities_y[i] = events['velocity_y'].iloc[nearest_event_index]
            mean_speeds[i] = events['speed'].iloc[nearest_event_index]
            mean_rewards[i] = events['reward_state'].iloc[nearest_event_index]
        else:
            mean_positions_x[i] = np.nan
            mean_positions_y[i] = np.nan
            mean_velocities_x[i] = np.nan
            mean_velocities_y[i] = np.nan
            mean_speeds[i] = np.nan
            mean_rewards[i] = np.nan


    # converting the spike rate matrix to a DataFrame
    data = pd.DataFrame(spike_rates.T, columns=[f"Unit {i}" for i in units])

    # adding the tracking data to the DataFrame
    conversion = 5.1
    data['x'] = mean_positions_x / conversion # convert to cm
    data['y'] = mean_positions_y / conversion # convert to cm
    data['v_x'] = mean_velocities_x / conversion # convert to cm/s
    data['v_y'] = mean_velocities_y / conversion # convert to cm/s
    data['speed'] = mean_speeds / conversion # convert to cm/s
    data['time'] = time_bins # in seconds
    data['reward_state'] = mean_rewards

    
    data.loc[data['speed'] > speed_threshold, ['x', 'y', 'v_x', 'v_y', 'speed']] = np.nan

    # interpolating the tracking data to fill in NaN values
    data.interpolate(method=interp_method, inplace=True, order = order)

    return data




"""
LFP analysis
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



def compute_multichannel_psd(lfp_data, fs=1000, nperseg=4000, noverlap=2000, nfft = 4096, method="mean_psd", plot_dir=None, region=None, sensitivity = 5):
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
    noisy_channels = []

    if plot_dir:
        sns.set_context('talk')
        plt.style.use('dark_background')
        plt.figure(figsize=(20, 10))

    # Compute PSD for each channel
    for ch in range(lfp_data.shape[0]):
        freqs, psd = signal.welch(lfp_data[ch], fs=fs, nperseg=nperseg, noverlap=noverlap, nfft = nfft, window='hann')
        psds.append(psd)

        # check if there is a peak at 60Hz
        target_freq = 60  # Hz
        idx_60hz = np.argmin(np.abs(freqs - target_freq))  # Closest index to 60Hz
        psd_60hz = psd[idx_60hz]

        # Compute local median around 60Hz (excluding the peak itself)
        nearby_idxs = (freqs > target_freq - 10) & (freqs < target_freq + 10)  # 10Hz window
        nearby_psd = psd[nearby_idxs]
        median_power = np.median(nearby_psd)

        # Check if 60Hz peak is significantly higher than surrounding noise
        if psd_60hz > sensitivity * median_power:
            noisy_channels.append(ch)  # Mark channel as noisy
        



        if plot_dir:
            sns.lineplot(x=freqs, y=psd, label=f'Channel {ch + 1}', alpha = 0.8)

    if plot_dir:

        # plot thin dotted lines at 60Hz and harmonics
        for i in range(1, 7):
            plt.axvline(x=60 * i, color='crimson', linestyle='dotted', alpha = 0.5)
        plt.title(f'{region} power spectral density across channels')
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('Power/Frequency (dB/Hz)')
        #plt.xscale('log')
        plt.yscale('log')
        #plt.xticks(ticks=[1, 10, 100], labels=['1', '10', '100'])
        sns.despine()
        plt.legend(fontsize = 12)
        plt.savefig(os.path.join(plot_dir), dpi=300)
        plt.close()

    psds = np.array(psds)  # Shape: (16, frequencies)

    if method == "mean_psd":
        # Compute the average PSD across all channels
        psd_avg = np.mean(psds, axis=0)
        return freqs, psd_avg, noisy_channels

    elif method == "psd_of_mean":
        # Compute PSD of the averaged LFP signal
        lfp_mean = np.mean(lfp_data, axis=0)
        freqs, psd_mean = signal.welch(lfp_mean, fs=fs, nperseg=nperseg, noverlap=noverlap, window='hann')
        return freqs, psd_mean, noisy_channels

    elif method == "per_channel":
        # Return PSD for each channel separately
        return freqs, psds, noisy_channels

    else:
        raise ValueError("Invalid method. Choose 'mean_psd', 'psd_of_mean', or 'per_channel'.")
    


def compute_multichannel_psd_noiseremoved(
    lfp_data, fs=1000, nperseg=4000, noverlap=2000, nfft=4096, method="mean_psd", 
    plot_dir=None, region=None, sensitivity=5.0
):
    """
    Computes the Power Spectral Density (PSD) across multiple LFP channels and detects 60Hz noise.
    Plots noisy channels in red and good channels in green.

    Parameters:
    - lfp_data (np.ndarray): 2D array of shape (channels, time) containing LFP data.
    - fs (int): Sampling rate in Hz (default = 1000 Hz).
    - nperseg (int): Segment length for Welch's method (default = 4s segments at 1kHz = 4000 samples).
    - noverlap (int): Overlapping samples between segments (default = 50% overlap = 2000 samples).
    - method (str): Method for combining PSD across channels.
    - plot_dir (str, optional): Directory to save PSD plots.
    - region (str, optional): Brain region name.
    - sensitivity (float): Controls peak detection sensitivity (higher = stricter).

    Returns:
    - freqs (np.ndarray): Frequency values in Hz.
    - psd (np.ndarray): PSD values (computed using only good channels).
    - noisy_channels (list): List of channel indices with significant 60Hz noise.
    """

    # Ensure input is a NumPy array
    lfp_data = np.asarray(lfp_data)

    # Number of channels
    num_channels = lfp_data.shape[0]

    # Initialize storage for PSDs
    psds = []
    noisy_channels = []
    good_channels = []

    if plot_dir:
        sns.set_context('talk')
        plt.style.use('dark_background')
        plt.figure(figsize=(20, 10))

    # Compute PSD for each channel and detect 60Hz noise
    for ch in range(num_channels):
        freqs, psd = signal.welch(lfp_data[ch], fs=fs, nperseg=nperseg, noverlap=noverlap, nfft=nfft, window='hann')
        psds.append(psd)

        # Detect 60Hz noise
        target_freq = 60  # Hz
        idx_60hz = np.argmin(np.abs(freqs - target_freq))  # Closest index to 60Hz
        psd_60hz = psd[idx_60hz]

        # Compute local median around 60Hz (excluding the peak itself)
        nearby_idxs = (freqs > target_freq - 10) & (freqs < target_freq + 10)  # 10Hz window
        nearby_psd = psd[nearby_idxs]
        median_power = np.median(nearby_psd)

        # Mark as noisy if 60Hz peak is significantly higher than surrounding noise
        if psd_60hz > sensitivity * median_power:
            noisy_channels.append(ch)  # Mark as noisy
            color = 'red'
        else:
            good_channels.append(ch)  # Mark as good
            color = 'green'

        if plot_dir:
            sns.lineplot(x=freqs, y=psd, color=color, alpha=0.8, label=f'Channel {ch + 1}')

    if plot_dir:
        # Plot vertical dotted lines at 60Hz and harmonics
        plt.axvline(x=60, color='crimson', linestyle='dotted', alpha=0.3)

        plt.title(f'{region} power spectral density across channels')
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('Power/Frequency (dB/Hz)')
        plt.xscale('log')
        plt.yscale('log')
        plt.xticks(ticks=[1, 10, 100], labels=['1', '10', '100'])
        sns.despine()
        plt.legend(fontsize=12)
        plt.savefig(os.path.join(plot_dir), dpi=300)
        plt.close()

    psds = np.array(psds)  # Shape: (channels, frequencies)

    # Use only good channels for PSD computation
    if len(good_channels) > 0:
        psd_final = np.mean(psds[good_channels], axis=0)  # Average PSD across good channels
    else:
        print("Warning: No clean channels detected, returning none.")
        psd_final = None

    return freqs, psd_final, noisy_channels


