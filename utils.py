import os
import re
import numpy as np
from kilosort import run_kilosort
from kilosort.io import save_preprocessing, load_ops
from pathlib import Path

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