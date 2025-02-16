import os
import re
import numpy as np
from kilosort import run_kilosort
from kilosort.io import save_preprocessing, load_ops
from pathlib import Path

"""
Run kilosort4. Use settings dictionary to change kilosort settings for the run.
"""
def kilosort(data_path: str, results_path: str, probe_path: str = '8_tetrode.mat', num_channels: int = 32, save_preprocessed: bool = True):
    # Initialize paths
    data_path = Path(data_path)
    results_path = Path(results_path)

    # Handle numpy files by temporarily converting to .bin format   
    if data_path.suffix == '.npy':
        # Load .npy file and save as binary
        data = np.load(data_path)
        print(f"Data import shape:{data.shape}")
        data_min = data.min()
        data_max = data.max()
        data_mean = data.mean()
        data_std = data.std()
        
        # Apply outlier clipping
        data = clip_outliers_with_window(data)
        
        data = data.reshape(-1, order = 'F')
        temp_bin_path = data_path.parent / 'temp.bin'
        data.tofile(temp_bin_path)
        print(f"Created temporary binary file: {temp_bin_path}")

        # Create temporary binary file in data parent directory
        data_path = data_path.parent / 'temp.bin'
      
    else:
        data = np.load(data_path)

    # Apply per-channel outlier clipping with window
    data = clip_outliers_with_window(data, clip_mult=2)

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
        unit_summary(data_path, results_path, data_min, data_max, data_mean, data_std, data_mean, data_std, 2, error=False)
        
        # Return results
        return ops, st, clu, tF, Wall, similar_templates, is_ref, est_contam_rate, kept_spikes
    
    except:
        # Write error to log
        unit_summary(data_path, results_path, data_min, data_max, data_mean, data_std, data_mean, data_std, 2, error=True)
        return None
    
"""
Helper Functions Below
"""

# Get the name of the last two directories in data path
def get_savedirs(path):
    path = str(path)
    parts = path.split(os.path.sep)
    return os.path.sep.join(parts[-3:-1])

# Get all mua.npy files in a directory and its subdirectories
def get_mua_paths(directory: str, print_paths=False) -> list:
    paths = [f for f in Path(directory).glob('**/*.npy') if 'mua' in f.name]
    print(f'Found {len(paths)} mua.npy files')
    if print_paths:
        show_paths(paths)
    return paths

# Get all continuous.dat files in a directory and its subdirectories
def get_continuous_paths(directory: str, print_paths=False) -> list:
    paths = [f for f in Path(directory).glob('**/*.dat') if 'continuous' in f.name]
    print(f'Found {len(paths)} continuous.dat files')
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
def unit_summary(data_path, results_path, data_min, data_max, data_mean, data_std, avg_min_val, avg_max_val, clip_mult, error=False):

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
            outfile.write(f"{mouse_session} - {num_units} units - min: {data_min} max: {data_max} std: {round(data_std, 3)} avg_min: {round(avg_min_val, 3)}, avg_max: {round(avg_max_val, 3)}, clip_mult: {clip_mult}\n")
    elif error:
        with open(output_file, 'a') as outfile:
            outfile.write(f"{mouse_session} - Kilosort failed - min: {data_min} max: {data_max} std: {round(data_std, 3)} avg_min: {round(avg_min_val, 3)}, avg_max: {round(avg_max_val, 3)}, clip_mult: {clip_mult}\n")
    else:
        with open(output_file, 'a') as outfile:
            outfile.write(f"{mouse_session} - No matching pattern found in the log file\n")

    print(f"Summary written to {output_file}")