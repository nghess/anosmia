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
from scipy.interpolate import interp1d
import matplotlib.colors as mcolors
from concurrent.futures import ProcessPoolExecutor


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



"""
LFP Analysis helper functions
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


def build_raster(lfp: np.array, inh: np.array, exh: np.array, save_path: str, filter: tuple = ('bandpass', [2, 12]), window_size: int = 2000, f: int = 1000):

    custom_x=False
    solojazz = load_colormap()

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

        plt.savefig(os.path.join(save_path, f'Channel_{ch}_LFP.png'), dpi = 300)
        plt.close()



def process_filter(args):

    matplotlib.use('Agg')

    ephys, inh_use, exh_use, filter_info, save_path = args
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
                            filter=filter_name, window_size=1000, f=1000)



def build_sniff_rasters(ephys: np.array, inh: np.array, exh: np.array, save_path: str, max_workers: int = 2):
    
    filters = {
        'raw': (None, None),
        'theta': ('bandpass', (2, 12)),
        'beta': ('bandpass', (18, 30)),
        'gamma': ('bandpass', (65, 100))}

    # Create processing tasks
    tasks = []
    for filter_name, filter_params in filters.items():
        tasks.append((ephys, inh, exh, (filter_name, filter_params), save_path))

    # Process in parallel
    with ProcessPoolExecutor(max_workers = max_workers) as executor:
        list(executor.map(process_filter, tasks))

        

"""
Colormap and plotting functions
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

