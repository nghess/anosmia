import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib
from scipy.stats import ranksums
import concurrent.futures
import torch
from torch.utils.data import DataLoader
import time
import os

from decoding_library import *
from plotting import plot_training, plot_preds, plot_rmse
from core import compute_spike_rates_sliding_window_by_region_smooth, load_behavior, align_brain_and_behavior, compute_sniff_freqs_bins, preprocess_rnp
from plotting import plot_position_trajectories



def process_fold_worker(rank, args):
    """
    Worker function to process a single fold.
    
    Parameters:
        rank: The fold number (process index).
        args: A tuple containing:
            - spike_rates: Full spike rates array.
            - behavior: Normalized behavior array.
            - n_folds: Total number of folds.
            - shift: The current shift value.
            - current_save_path: Where to save outputs.
            - behavior_name: Name of the behavior.
            - behavior_dim: Behavior dimensions.
            - model_params: Model and training parameters.
            - rmse_list: A shared list to store the RMSE for each fold.
    """
    (spike_rates, behavior, n_folds, shift, current_save_path,
     behavior_name, behavior_dim, model_params, rmse_list) = args

    # Here, 'rank' serves as the fold index.
    rmse = process_fold(spike_rates, behavior, rank, shift,
                        current_save_path, behavior_name, behavior_dim, model_params, rank)
    rmse_list[rank] = rmse  # Store the result in the shared list.



def process_fold(spike_rates, behavior, k, shift, current_save_path, behavior_name, params, device):


    np.random.seed(int(time.time()) + k)
    plt.style.use('dark_background')


    k_CV, n_blocks, plot_predictions, sequence_length, hidden_dim, num_layers, dropout, num_epochs, lr, patience, min_delta, factor = params

    behavior_mean = np.mean(behavior, axis=0)
    behavior_std = np.std(behavior, axis=0)
    behavior = (behavior - behavior_mean) / behavior_std
    
    rates_train, rates_test, train_switch_ind, test_switch_ind = cv_split(spike_rates, k, k_CV=k_CV, n_blocks=n_blocks)
    behavior_train, behavior_test, _, _ = cv_split(behavior, k, k_CV=k_CV, n_blocks=n_blocks)


    # Create the model
    lstm_model = LSTMDecoder(input_dim=rates_train.shape[1], hidden_dim=hidden_dim, output_dim=behavior.shape[1], num_layers=num_layers, dropout = dropout).to(device)

    # Prepare the training data for LSTM
    blocks = [(train_switch_ind[i], train_switch_ind[i + 1]) for i in range(len(train_switch_ind) - 1)]
    train_dataset = SequenceDataset(rates_train, behavior_train, blocks, sequence_length)
    train_loader = DataLoader(train_dataset, batch_size=min(16_384, len(train_dataset)), shuffle=False, num_workers=1, pin_memory=True, prefetch_factor=1, persistent_workers=True)

    # Training the LSTM model
    trained_lstm_model, lstm_history = train_LSTM(lstm_model, train_loader, device, lr=lr, epochs=num_epochs, patience=patience, min_delta=min_delta, factor=factor)

    # Free up memory
    del lstm_model, train_dataset, train_loader

    # Plot loss
    if plot_predictions:
        plot_training(lstm_history, current_save_path, shift, k)

    # Prepare the test data for LSTM
    test_blocks = [(test_switch_ind[i], test_switch_ind[i + 1]) for i in range(len(test_switch_ind) - 1)]
    test_dataset = SequenceDataset(rates_test, behavior_test, test_blocks, sequence_length)
    test_loader = DataLoader(test_dataset, batch_size=min(16_384, len(test_dataset)), persistent_workers=True, num_workers=1, pin_memory=True)

    # Predict on the test set
    trained_lstm_model.eval()
    predictions = []
    targets = []
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch = X_batch.to(device)
            preds = trained_lstm_model(X_batch)
            predictions.append(preds.cpu().numpy())
            targets.append(y_batch.cpu().numpy())
    predictions = np.concatenate(predictions, axis=0)
    targets = np.concatenate(targets, axis=0)

    # Clean up
    del trained_lstm_model, test_dataset, test_loader

    # Converting the predictions and true values back to original scale
    predictions = predictions * behavior_std + behavior_mean
    targets = targets * behavior_std + behavior_mean

    # plotting the predicted and true values
    if plot_predictions:
        plot_preds(targets, predictions, test_switch_ind, behavior_name, behavior.shape[1], sequence_length, current_save_path, k, shift)

    # Calculate the RMSE
    rmse = np.sqrt(np.mean((predictions - targets) ** 2))

    # Final clean up
    del predictions, targets
    torch.cuda.empty_cache()

    return rmse



def parallel_process_session_GPU(spike_rates, behavior, n_folds, shift, current_save_path, 
                             behavior_name, behavior_dim, model_params, parallel):
    """
    Run process_fold in parallel for all folds using torch.multiprocessing.
    
    Parameters:
        spike_rates: Full spike rates array.
        behavior: Normalized behavior array.
        n_folds: Number of CV folds.
        shift: The current shift value.
        current_save_path: Where to save outputs for this region/shift.
        behavior_name, behavior_dim: Behavior scaling and labeling.
        model_params: List of model and training parameters (e.g., k_CV, n_blocks, plot_predictions, etc.).
    
    Returns:
        List of RMSE values, one per fold.
    """
    
    # Create a manager to handle a shared list.
    manager = mp.Manager()
    rmse_list = manager.list([None] * n_folds)
    
    # Pack the common arguments to pass to each worker.
    args = (spike_rates, behavior, n_folds, shift, current_save_path, 
            behavior_name, behavior_dim, model_params, rmse_list)
    
    # Launch processes, one per fold.
    # mp.spawn will automatically pass the rank (from 0 to n_folds-1) as the first argument.
    mp.spawn(process_fold_worker, args=(args,), nprocs=n_folds, join=True)
    
    # Convert the shared list to a regular list before returning.
    return list(rmse_list)


def parallel_process_session_CPU(spike_rates, behavior, k_CV, n_shifts, region_save_path, behavior_name, num_workers, model_params):
    """
    Run process_fold in parallel for all folds using torch.multiprocessing.
    
    Parameters:
        spike_rates: Full spike rates array.
        behavior: Normalized behavior array.
        n_folds: Number of CV folds.
        shift: The current shift value.
        current_save_path: Where to save outputs for this region/shift.
        behavior_name, behavior_dim: Behavior scaling and labeling.
        model_params: List of model and training parameters (e.g., k_CV, n_blocks, plot_predictions, etc.).
    
    Returns:
        List of RMSE values, one per fold.
    """
 
    rmse_list = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = []

        # Looping through the circular shifts
        for shift in range(n_shifts):
            if shift == 0:
                current_save_path = os.path.join(region_save_path, 'true')
                behavior_use = behavior
            else:
                roll_value = np.random.randint(100, np.max(behavior.shape) - 100)
                behavior_use = np.roll(behavior, roll_value, axis=0)
                current_save_path = os.path.join(region_save_path, 'null')
                os.makedirs(current_save_path, exist_ok=True)

            # Loop through the cross-validation folds
            for k in range(k_CV):
                futures.append(executor.submit(
                    process_fold,
                    spike_rates, behavior_use, k, shift, current_save_path,
                    behavior_name, model_params, torch.device('cpu')))
        
        for future in concurrent.futures.as_completed(futures):
            rmse_list.append(future.result())

    # Separate the true and null RMSE
    true_rmse = rmse_list[:k_CV * n_shifts]
    null_rmse = rmse_list[k_CV * n_shifts:]
    
    return true_rmse, null_rmse
    


def parallel_process_session(spike_rates, behavior, n_folds, shift, current_save_path, 
                           behavior_name, behavior_dim, model_params, parallel, use_GPU):
    if not parallel:
        return [process_fold(spike_rates, behavior, k, shift, current_save_path, 
                           behavior_name, behavior_dim, model_params, use_GPU) 
                for k in range(n_folds)]
    
    # Calculate number of GPU workers based on available memory
    n_gpus = torch.cuda.device_count()
    gpu_memory = torch.cuda.get_device_properties(0).total_memory
    # Adjust workers based on model size and available memory
    max_workers = min(n_gpus * 2, n_folds)  # Use up to 2 workers per GPU
    
    rmse_list = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = []
        for k in range(n_folds):
            futures.append(executor.submit(
                process_fold,
                spike_rates, behavior, k, shift, current_save_path,
                behavior_name, behavior_dim, model_params, use_GPU
            ))
        
        for future in concurrent.futures.as_completed(futures):
            rmse_list.append(future.result())
    
    return rmse_list


def process_session(params):
    
    """"
    Decode the neural data using LSTM model.

    Parameters
    ----------
    params : list
        List of parameters for the decoding.
    """

 

    # Unpack the parameters
    mouse, session, kilosort_dir, behavior_dir, tracking_dir, sniff_dir, save_dir, window_size, step_size, fs, sigma_smooth, use_units, speed_threshold, k_CV, n_blocks, n_shifts, plot_predictions, target, sequence_length, hidden_dim, num_layers, dropout, num_epochs, lr, patience, min_delta, factor, use_GPU = params
    model_params = [k_CV, n_blocks, plot_predictions, sequence_length, hidden_dim, num_layers, dropout, num_epochs, lr, patience, min_delta, factor]


    # Preprocessing for the clickbait-ephys project
    if behavior_dir:
        # Load the spike rates
        rates_OB, rates_HC, time_bins, units_OB, units_HC = compute_spike_rates_sliding_window_by_region_smooth(kilosort_dir, fs, window_size, step_size, use_units=use_units, sigma=sigma_smooth, zscore=True)
        
        # load the tracking data
        tracking_file = os.path.join(tracking_dir, [f for f in os.listdir(tracking_dir) if f.endswith('.analysis.h5')][0])
        events = load_behavior(behavior_dir, tracking_file)
        
        # Loading the sniffing
        sniff_params_file = os.path.join(sniff_dir, 'sniff_params.mat')
        if os.path.exists(sniff_params_file):
            mean_freqs, _ = compute_sniff_freqs_bins(sniff_params_file, time_bins, window_size, sfs = 1000)
        else:
            mean_freqs = np.zeros(len(time_bins))



    # Looping through the regions for decodinh
    for region in ['HC', 'OB']:

        # Preprocessing for the RNP dataset
        if not behavior_dir:
            if region == 'HC':
                continue
            region_save_path = os.path.join(save_dir, mouse, session, region)
            os.makedirs(region_save_path, exist_ok=True)
            spks, pos_ss, _, _ = preprocess_rnp(kilosort_dir, bin_size=10, crop_hf=True, flip_data=None, make_plots=True, save_path= region_save_path)
            spike_rates = spks
            behavior = pos_ss
            behavior_name = ['x', 'y']
            max_roll = len(behavior) - 100

            

        # Preprocessing for the clickbait-ephys dataset
        else:
            if region == 'OB':
                rates = rates_OB
                units = units_OB
                other_rates = rates_HC
                other_units = units_HC
            else:
                rates = rates_HC
                units = units_HC
                other_rates = rates_OB
                other_units = units_OB

            # Skip if there are too few neurons
            if rates.shape[0] < 2:
                print(f"Skipping {mouse}/{session}/{region} due to insufficient neurons.")
                continue
                
            # Save path for the region
            region_save_path = os.path.join(save_dir, mouse, session, region)
            os.makedirs(region_save_path, exist_ok=True)

            # Aligning the brain and behavior data
            data = align_brain_and_behavior(events, rates, units, time_bins, window_size, speed_threshold = speed_threshold)
            plot_position_trajectories(data, save_path=region_save_path)
            data['sns'] = mean_freqs
            data['sns'] = data['sns'].interpolate(method='linear')
            data.dropna(subset=['x', 'y', 'v_x', 'v_y', 'sns'], inplace=True)
            spike_rates = data.iloc[:, :-8].values


            # Getting the behavior data
            if target == 'position':
                behavior = data[['x', 'y']].values
                behavior_name = ['x', 'y']
            elif target == 'sniffing':
                behavior = data['sns'].values
                behavior_name = ['sniff rate']
                behavior = behavior.reshape(-1, 1)
            elif target == 'velocity':
                behavior = data[['v_x', 'v_y']].values
                behavior_name = ['v_x', 'v_y']
            elif target == 'neural':
                if other_rates.shape[0] < 1:
                    print(f"Skipping {mouse}/{session}/{region} due to insufficient neurons.")
                    continue

                data_other = align_brain_and_behavior(events, other_rates, other_units, time_bins, window_size, speed_threshold = speed_threshold)
                data_other['sns'] = mean_freqs
                data_other['sns'] = data['sns'].interpolate(method='linear')
                data_other.dropna(subset=['x', 'y', 'v_x', 'v_y', 'sns'], inplace=True)
                behavior = data.iloc[:, :-8].values
                behavior_name = [f"$Y_{{{i}}}$" for i in range(behavior.shape[1])]
            max_roll = len(behavior) - 100


        print(f"Processing {mouse}/{session}/{region}")

        # Loop through the shifts
        if use_GPU:
            true_rmse = []
            null_rmse = []
            for shift in range(n_shifts + 1):
                if shift == 0:
                    current_save_path = os.path.join(region_save_path, 'true')
                    os.makedirs(current_save_path, exist_ok=True)
                    behavior_use = behavior
                else:
                    roll_value = np.random.randint(100, max_roll)
                    behavior_use = np.roll(behavior, roll_value, axis=0)
                    current_save_path = os.path.join(region_save_path, 'null')
                    os.makedirs(current_save_path, exist_ok=True)

                # Loop through the cross-validation folds to get decoding errors
                if shift == 0:
                    true_rmse = parallel_process_session_GPU(spike_rates, behavior_use, k_CV, shift, current_save_path, behavior_name, model_params, parallel = True)
                else:
                    null_rmse.append(parallel_process_session_GPU(spike_rates, behavior_use, k_CV, shift, current_save_path, behavior_name, model_params, parallel = True))
            true_rmse = np.array(true_rmse).flatten()
            null_rmse = np.array(null_rmse).flatten()
        
        else:
            true_rmse, null_rmse = parallel_process_session_CPU(spike_rates, behavior, k_CV, n_shifts, region_save_path, behavior_name, 60, model_params)

        # rank sum test
        _ , p_val = ranksums(true_rmse, null_rmse, 'less')
        print(f"Rank sum test p-value for {mouse}/{session}: {p_val}")
        
        # plot histograms of the rmse
        plot_rmse(true_rmse, null_rmse, p_val, region_save_path)


        # create a text file to save the p-value
        with open(os.path.join(region_save_path, 'p_value.txt'), 'w') as f:
            f.write(f"Rank sum test p-value: {p_val}\n")

        # save the mse results
        np.save(os.path.join(region_save_path, f"true_rmse.npy"), true_rmse)
        np.save(os.path.join(region_save_path, f"null_rmse.npy"), null_rmse)




def rnp():


    # Set plotting style and context
    matplotlib.use('Agg')  # Use non-interactive backend to avoid issues in threaded environment
    plt.style.use('dark_background')
    sns.set_context('poster')


    # defining directories
    data_dir = r"E:\place_decoding\data\bulb"
    save_dir_main = r"E:\clickbait-ephys\figures\RNP LSTM (3-20-25)"



    # defining the subset of data to process
    mice = ['4122', '4127', '4131', '4138']
    sessions = ['7', '8', '9', '10', '11', '12', '13', '14', '15', '16', '17', '18', '19', '0', '1', '2', '3', '4', '5', '6', ]


    # defining the data variables
    fs = 30_000  # sampling rate for neural data (mua.npy)


    # Defining the data preprocessing parameters
    window_size = .1  # Window size for spike rate computation (in seconds)
    step_size = .1  # Step size for sliding window (in seconds)
    sigma_smooth = 2.5  # Standard deviation for gaussian smoothing of spike rates
    speed_threshold = 100  # Tracking point with speed above this value will be removed before interpolation


    # Defining the decoding parameters
    n_shifts = 10 # Define number of shifts for circular shifting of behavior data
    k_CV = 10 # Define number of cross-validation folds
    n_blocks = 10 # Define number of blocks for cross-validation
    plot_predictions = True
    sequence_length = 10 # Define the sequence length for LSTM input
    hidden_dim = 64 # Define the hidden dimension for LSTM
    num_layers = 2 # Define the number of layers for LSTM
    dropout = 0.5 # Define the dropout for LSTM
    num_epochs = 500 # Define the number of epochs for LSTM training
    lr = 0.01
    patience = 20
    min_delta = 0.001
    factor = 0.5

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")




    # creating a directory to save the figures
    save_dir = os.path.join(save_dir_main, f"window_size_{window_size}")
    os.makedirs(save_dir, exist_ok=True)

    # saving a textfile with all parameters
    with open(os.path.join(save_dir, 'parameters.txt'), 'w') as f:
        f.write(f"window_size: {window_size}\n")
        f.write(f"step_size: {step_size}\n")
        f.write(f"sigma_smooth: {sigma_smooth}\n")
        f.write(f"speed_threshold: {speed_threshold}\n")
        f.write(f"n_shifts: {n_shifts}\n")
        f.write(f"k_CV: {k_CV}\n")
        f.write(f"n_blocks: {n_blocks}\n")
        f.write(f"sequence_length: {sequence_length}\n")
        f.write(f"hidden_dim: {hidden_dim}\n")
        f.write(f"num_layers: {num_layers}\n")
        f.write(f"dropout: {dropout}\n")
        f.write(f"num_epochs: {num_epochs}\n")
        f.write(f"lr: {lr}\n")
        f.write(f"patience: {patience}\n")
        f.write(f"min_delta: {min_delta}\n")
        f.write(f"factor: {factor}\n")
        f.write(f"save_dir: {save_dir}\n")


        # Looping through the data
        for mouse in mice:
            for session in sessions:
                # Building the task list
                current_data_dir = os.path.join(data_dir, mouse, session)
                if not os.path.exists(current_data_dir):
                    print(f"Skipping {mouse}/{session} due to missing data.")
                    continue
                params = [mouse, session, current_data_dir, None, None, None, save_dir, window_size, step_size, fs, sigma_smooth, None, speed_threshold, k_CV, n_blocks, n_shifts, plot_predictions, None, sequence_length, hidden_dim, num_layers, dropout, num_epochs, lr, patience, min_delta, factor]


                # Run the decoding
                #try:
                process_session(params)
                #except Exception as e:
                    #print(f"Error processing {mouse}/{session}: {e}")



def main(window_size: float = 0.1, step_size: float = 0.1, sigma_smooth: float = 2.5, use_units: str = 'good/mua', speed_threshold: int = 100, n_shifts: int = 1, k_CV: int = 8, n_blocks: int = 10, plot_predictions: bool = True, sequence_length: int = 10, hidden_dim: int = 64, num_layers: int = 2, dropout: float = 0.5, num_epochs: int = 500, lr: float = 0.01, patience: int = 20, min_delta: float = 0.001, factor: float = 0.5, fs: int = 30_000, use_GPU: bool = True):


    """
    Decode the neural data using LSTM model.

    Parameters
    ----------
    window_size : float
        Window size for spike rate computation (in seconds)
    step_size : float
        Step size for sliding window (in seconds)
    sigma_smooth : float
        Standard deviation for gaussian smoothing of spike rates
    use_units : str
        What kilosort cluster labels to use
    speed_threshold : int
        Tracking point with speed above this value will be removed before interpolation
    n_shifts : int
        Define number of shifts for circular shifting of behavior data
    k_CV : int
        Define number of cross-validation folds
    n_blocks : int
        Define number of blocks for cross-validation
    plot_predictions : bool
        Whether to plot the predictions
    sequence_length : int
        Define the sequence length for LSTM input
    hidden_dim : int
        Define the hidden dimension for LSTM
    num_layers : int
        Define the number of layers for LSTM
    dropout : float
        Define the dropout for LSTM
    num_epochs : int
        Define the number of epochs for LSTM training
    lr : float
        Define the learning rate for LSTM training
    patience : int
        Define the patience for early stopping
    min_delta : float
        Define the minimum delta for early stopping
    factor : float
        Define the factor for reducing the learning rate on plateau
    fs : int
        Sampling rate for neural data (mua.npy)
    use_GPU : bool
        Whether to use GPU for parallel processing
    
    """

    # Set plotting style and context
    matplotlib.use('Agg')  # Use non-interactive backend to avoid issues in threaded environment
    plt.style.use('dark_background')
    sns.set_context('poster')


    # defining directories
    spike_dir = "/projects/smearlab/shared/clickbait-ephys(3-20-25)/kilosorted"
    save_dir_main = "/projects/smearlab/shared/clickbait-ephys(3-20-25)/figures/testing"
    events_dir = "/projects/smearlab/shared/clickbait-ephys(3-20-25)/behavior_data"
    SLEAP_dir = "/projects/smearlab/shared/clickbait-ephys(3-20-25)/sleap_predictions"
    sniff_dir = "/projects/smearlab/shared/clickbait-ephys(3-20-25)/sniff events"


    for target in ['position', 'sniffing', 'neural']:

        # creating a directory to save the figures
        save_dir = os.path.join(save_dir_main, f"window_size_{window_size}_target_{target}")
        os.makedirs(save_dir, exist_ok=True)

        # saving a textfile with all parameters
        save_parameters(window_size, step_size, sigma_smooth, use_units, speed_threshold, n_shifts, k_CV, n_blocks, target, sequence_length, hidden_dim, num_layers, dropout, num_epochs, lr, patience, min_delta, factor, use_GPU, save_dir)

        # Looping through the data
        mice = os.listdir(spike_dir)
        mice = [m for m in mice if os.path.isdir(os.path.join(spike_dir, m))]
        for mouse in mice:
            sessions = os.listdir(os.path.join(spike_dir, mouse))
            sessions = [s for s in sessions if os.path.isdir(os.path.join(spike_dir, mouse, s))]
            for session in sessions:

                # Building the task list
                kilosort_dir = os.path.join(spike_dir, mouse, session)
                behavior_dir = os.path.join(events_dir, mouse, session)
                tracking_dir = os.path.join(SLEAP_dir, mouse, session)
                sniff_params_dir = os.path.join(sniff_dir, mouse, session)
                if not os.path.exists(kilosort_dir) or not os.path.exists(behavior_dir) or not os.path.exists(tracking_dir) or not os.path.exists(sniff_dir):
                    print(f"Skipping {mouse}/{session} due to missing data.")
                    continue

                # Define the parameters
                params = [mouse, session, kilosort_dir, behavior_dir, tracking_dir, sniff_params_dir, save_dir, window_size, step_size, fs, sigma_smooth, use_units, speed_threshold, k_CV, n_blocks, n_shifts, plot_predictions, target, sequence_length, hidden_dim, num_layers, dropout, num_epochs, lr, patience, min_delta, factor, use_GPU]

                # Run the decoding
                try:
                    process_session(params)
                except Exception as e:
                    print(f"Error processing {mouse}/{session}: {e}")




if __name__ == "__main__":

    use_GPU = True



    # Set the parameters
    window_size = 0.1
    step_size = 0.1
    sigma_smooth = 2.5
    use_units = 'good/mua'
    speed_threshold = 100
    n_shifts = 5
    k_CV = 10
    n_blocks = 5
    plot_predictions = True
    sequence_length = 10
    hidden_dim = 32
    num_layers = 2
    dropout = 0.1
    num_epochs = 500
    lr = 0.01
    patience = 20
    min_delta = 0.001
    factor = 0.5
    fs = 30_000

    # Run the main function
    #main(window_size, step_size, sigma_smooth, use_units, speed_threshold, n_shifts, k_CV, n_blocks, plot_predictions, sequence_length, hidden_dim, num_layers, dropout, num_epochs, lr, patience, min_delta, factor, fs, use_GPU)
    #print(f"Total time: {time.time() - start_time} seconds")