
import numpy as np
import pandas as pd
from kilosort import run_kilosort
from kilosort.io import save_preprocessing, load_ops
from pathlib import Path
import matplotlib.pyplot as plt

def kilosort(data_path: str, save_path: str, probe_path: str = '8_tetrode.mat', num_channels: int = 32, save_preprocessed: bool = True):
    # Initialize paths
    data_path = Path(data_path)
    save_path = Path(save_path)

    # Handle numpy files by temporarily converting to .bin format   
    if data_path.suffix == '.npy':
        # Load .npy file and save as binary
        data = np.load(data_path)
        # Ensure data is in (samples, channels) format
        #if data.shape[1] > data.shape[0]:  # If channels > samples, transpose
        print(f"Data import shape:{data.shape}")

        # plt.figure()
        # plt.title(f'tetrode {1}')
        # for j in range(4):
        #     plt.plot(data[j,:30_000 * 10], linewidth = 0.1, alpha = 0.8)
        # plt.show()

        # If data exceeds abs(max), set to 0
        #data = data * .1945
        #data[np.abs(data) > 500] = 0


        print(f"Min:{np.min(data)}")
        print(f"Max:{np.max(data)}")

        # Convert to int16 and save as binary
        data = data.T.ravel().astype(np.int16)  # Transpose and flatten for Kilosort format
        temp_bin_path = data_path.parent / 'temp.bin'
        data.tofile(temp_bin_path)
        # Create temporary binary file in data parent directory
        data_path = data_path.parent / 'temp.bin'

    # Create results directory if it doesn't exist
    save_path.mkdir(parents=True, exist_ok=True)

    # Run Kilosort 4
    settings = {'data_dir': data_path.parent, 'n_chan_bin': num_channels, 'drift_smoothing': [0, 0, 0]}
    ops, st, clu, tF, Wall, similar_templates, is_ref, est_contam_rate, kept_spikes = \
        run_kilosort(
            settings=settings, 
            probe_name=Path.cwd() / probe_path,
            save_preprocessed_copy=save_preprocessed,
            do_CAR= False,
            results_dir=save_path
            )
    
    # Delete temporary binary file from drive if it exists
    temp_bin_path = data_path.parent / 'temp.bin'
    if temp_bin_path.exists():
        temp_bin_path.unlink()
    
    # Return results
    return ops, st, clu, tF, Wall, similar_templates, is_ref, est_contam_rate, kept_spikes


# Get all mua.npy files in a directory and its subdirectories
def get_mua_paths(directory: str) -> list:
    paths = [f for f in Path(directory).glob('**/*.npy') if 'mua' in f.name]
    print(f'Found {len(paths)} mua.npy files')
    return paths

# Get all continuous.dat files in a directory and its subdirectories
def get_continuous_paths(directory: str) -> list:
    paths = [f for f in Path(directory).glob('**/*.dat') if 'continuous' in f.name]
    print(f'Found {len(paths)} continuous.dat files')
    return paths



