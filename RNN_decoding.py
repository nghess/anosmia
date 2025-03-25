print("\nFiring up the engines!\n")
import time
START = time.time()
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib
from scipy.stats import ranksums
import torch
from torch.utils.data import DataLoader
import torch.multiprocessing as mp
import os
from decoding_library import *
from plotting import plot_training, plot_preds, plot_rmse
from core import compute_spike_rates_sliding_window_by_region_smooth, load_behavior, align_brain_and_behavior, compute_sniff_freqs_bins, preprocess_rnp
from plotting import plot_position_trajectories
print(f"Engines are up and running ({time.time() - START:.2f}s)\n\n")




def process_fold(rates_train, rates_test, train_switch_ind, test_switch_ind, behavior_train, behavior_test, behavior, behavior_mean, behavior_std, k, shift, current_save_path, behavior_name, params, device):

    plt.style.use('dark_background')

    # Set the device
    if device:
        os.environ['CUDA_VISIBLE_DEVICES'] = device
        device = torch.device('cuda:0')
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


    batch_size = min(8192, len(behavior))

    # Unpack the parameters
    plot_predictions, sequence_length, hidden_dim, num_layers, dropout, num_epochs, lr, patience, min_delta, factor = params

    # Create the model
    lstm_model = LSTMDecoder(input_dim=rates_train.shape[1], hidden_dim=hidden_dim, output_dim=behavior.shape[1], num_layers=num_layers, dropout = dropout).to(device)

    # Prepare the training data for LSTM
    blocks = [(train_switch_ind[i], train_switch_ind[i + 1]) for i in range(len(train_switch_ind) - 1)]
    train_dataset = SequenceDataset(rates_train, behavior_train, blocks, sequence_length)
    train_loader = DataLoader(train_dataset, batch_size=min(batch_size, len(train_dataset)), shuffle=False, num_workers=0, pin_memory=True, prefetch_factor=None)

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
    test_loader = DataLoader(test_dataset, batch_size=min(batch_size, len(test_dataset)), num_workers=0, pin_memory=True)

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




def compile_folds(spike_rates, behavior, behavior_mean, behavior_std, n_folds, shift, current_save_path, 
                           behavior_name, model_params, parallel):
    


    # Check the number of available cuda devices (MIG instances) and collect their names.
    devices = os.environ.get('CUDA_VISIBLE_DEVICES')
    mig_devices = []
    if devices:
        device_list = devices.split(',')
        for i, d in enumerate(device_list):
            mig_devices.append(d)
    else:
        print("No visible CUDA devices found.")

    arg_list = []
    if not parallel:
        mig_devices = [False for _ in range(n_folds)]
    for k in range(n_folds):
        rates_train, rates_test, train_switch_ind, test_switch_ind = cv_split(spike_rates, k, k_CV=k_CV, n_blocks=n_blocks)
        behavior_train, behavior_test, _, _ = cv_split(behavior, k, k_CV=k_CV, n_blocks=n_blocks)
        arg_list.append((rates_train, rates_test, train_switch_ind, test_switch_ind, behavior_train, behavior_test, behavior, behavior_mean, behavior_std, k, shift, current_save_path, behavior_name, model_params, mig_devices[k]))



    # For running the process_fold function sequentially or in parallel
    if not parallel:
        return [process_fold(*arg_list[k]) for k in range(n_folds)]
    else:
        with mp.Pool(n_folds) as pool:
            results = pool.starmap(process_fold, arg_list)
        return results
    




def process_session(params):
    
    """"
    Decode the neural data using LSTM model.

    Parameters
    ----------
    params : list
        List of parameters for the decoding.
    """

    

    # Unpack the parameters
    mouse, session, kilosort_dir, behavior_dir, tracking_dir, sniff_dir, save_dir, window_size, step_size, fs, sigma_smooth, use_units, speed_threshold, k_CV, n_blocks, n_shifts, plot_predictions, target, sequence_length, hidden_dim, num_layers, dropout, num_epochs, lr, patience, min_delta, factor = params
    model_params = [plot_predictions, sequence_length, hidden_dim, num_layers, dropout, num_epochs, lr, patience, min_delta, factor]
    print(f"\n\nProcessing {mouse}/{session}")

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
        start_time = time.time()

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
                behavior = data_other.iloc[:, :-8].values
                behavior_name = [f"$Y_{{{i}}}$" for i in range(behavior.shape[1])]
            max_roll = len(behavior) - 1000


        print(f"Beggining {region}")

        behavior_mean = np.mean(behavior, axis=0)
        behavior_std = np.std(behavior, axis=0)
        behavior = (behavior - behavior_mean) / behavior_std

        # Loop through the shifts
        true_rmse = []
        null_rmse = []
        for shift in range(n_shifts + 1):
            if shift == 0:
                current_save_path = os.path.join(region_save_path, 'true')
                os.makedirs(current_save_path, exist_ok=True)
                behavior_use = behavior
            else:
                roll_value = np.random.randint(1000, max_roll)
                behavior_use = np.roll(behavior, roll_value, axis=0)
                current_save_path = os.path.join(region_save_path, 'null')
                os.makedirs(current_save_path, exist_ok=True)

            # Loop through the cross-validation folds to get decoding errors
            if shift == 0:
                true_rmse = compile_folds(spike_rates, behavior_use, behavior_mean, behavior_std, k_CV, shift, current_save_path, behavior_name, model_params, parallel = False)
            else:
                null_rmse.append(compile_folds(spike_rates, behavior_use,behavior_mean, behavior_std,  k_CV, shift, current_save_path, behavior_name, model_params, parallel = False))


        true_rmse = np.array(true_rmse).flatten()
        null_rmse = np.array(null_rmse).flatten()
        

        # rank sum test
        _ , p_val = ranksums(true_rmse, null_rmse, 'less')
        print(f"Rank sum test p-value: {p_val}")
        
        # plot histograms of the rmse
        plot_rmse(true_rmse, null_rmse, p_val, region_save_path)


        # create a text file to save the p-value
        with open(os.path.join(region_save_path, 'p_value.txt'), 'w') as f:
            f.write(f"Rank sum test p-value: {p_val}\n")

        # save the mse results
        np.save(os.path.join(region_save_path, f"true_rmse.npy"), true_rmse)
        np.save(os.path.join(region_save_path, f"null_rmse.npy"), null_rmse)
        print(f"Finished {region} in {time.time() - start_time:.2f}s")






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
    k_CV = 12 # Define number of cross-validation folds
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






def main(args):

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
    """

    # Access arguments as attributes
    target = args.target
    window_size = args.window_size
    n_shifts = args.n_shifts
    use_units = args.use_units
    plot_predictions = args.plot_predictions
    sequence_length = args.sequence_length
    hidden_dim = args.hidden_dim
    num_layers = args.num_layers
    step_size = args.step_size
    sigma_smooth = args.sigma_smooth
    speed_threshold = args.speed_threshold
    k_CV = args.k_CV
    n_blocks = args.n_blocks
    dropout = args.dropout
    num_epochs = args.num_epochs
    lr = args.lr
    patience = args.patience
    min_delta = args.min_delta
    factor = args.factor
    fs = args.fs


    # Set plotting style and context
    matplotlib.use('Agg')  # Use non-interactive backend to avoid issues in threaded environment
    plt.style.use('dark_background')
    sns.set_context('poster')


    # defining directories
    spike_dir = "/projects/smearlab/shared/clickbait-ephys(3-20-25)/kilosorted"
    save_dir_main = "/projects/smearlab/shared/clickbait-ephys(3-20-25)/figures/LSTM"
    events_dir = "/projects/smearlab/shared/clickbait-ephys(3-20-25)/behavior_data"
    SLEAP_dir = "/projects/smearlab/shared/clickbait-ephys(3-20-25)/sleap_predictions"
    sniff_dir = "/projects/smearlab/shared/clickbait-ephys(3-20-25)/sniff events"


    # creating a directory to save the figures
    save_dir = os.path.join(save_dir_main, f"window_size_{window_size}_target_{target}_sequence_length_{sequence_length}_hidden_dim_{hidden_dim}_num_layers_{num_layers}_shifts_{n_shifts}")
    os.makedirs(save_dir, exist_ok=True)

    # saving a textfile with all parameters
    save_parameters(window_size, step_size, sigma_smooth, use_units, speed_threshold, n_shifts, k_CV, n_blocks, target, sequence_length, hidden_dim, num_layers, dropout, num_epochs, lr, patience, min_delta, factor, save_dir)

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
            params = [mouse, session, kilosort_dir, behavior_dir, tracking_dir, sniff_params_dir, save_dir, window_size, step_size, fs, sigma_smooth, use_units, speed_threshold, k_CV, n_blocks, n_shifts, plot_predictions, target, sequence_length, hidden_dim, num_layers, dropout, num_epochs, lr, patience, min_delta, factor]

            # Run the decoding
            try:
                process_session(params)
            except Exception as e:
                print(f"Error processing {mouse}/{session}: {e}")








if __name__ == "__main__":
    import argparse


    parser = argparse.ArgumentParser()

    parser.add_argument('--target', type=str, choices=['position', 'velocity', 'sniffing', 'neural'], default='position',
                        help='What to decode')
    parser.add_argument('--window_size', type=float, default=0.1,
                        help='size of the window for spike rate computation in seconds (e.g., 0.1, 0.03)')
    parser.add_argument('--step_size', type=float, default=0.1,
                        help='size of the step for sliding window in seconds (Should be equal to window_size)')
    parser.add_argument('--sigma_smooth', type=float, default=2.5,
                        help='Standard deviation for gaussian smoothing of spike rates')
    parser.add_argument('--use_units', type=str, choices=['good', 'good/mua', 'mua'], default='good/mua',
                        help='What kilosort cluster labels to use')
    parser.add_argument('--speed_threshold', type=float, default=100,
                        help='Tracking point with speed above this value will be removed before interpolation')
    parser.add_argument('--n_shifts', type=int, default=1,
                        help='Number of shifts for circular shifting of behavior data')
    parser.add_argument('--k_CV', type=int, default=12,
                        help='Number of cross-validation folds')
    parser.add_argument('--n_blocks', type=int, default=5,
                        help='Number of blocks for cross-validation')
    parser.add_argument('--plot_predictions', type=bool, default=True,
                        help='Whether to plot the predictions')
    parser.add_argument('--sequence_length', type=int, default=10,
                        help='Define the sequence length for LSTM input')
    parser.add_argument('--hidden_dim', type=int, default=64,
                        help='Define the hidden dimension for LSTM')
    parser.add_argument('--num_layers', type=int, default=2,
                        help='Define the number of layers for LSTM')
    parser.add_argument('--dropout', type=float, default=0.5,
                        help='Define the dropout for LSTM')
    parser.add_argument('--num_epochs', type=int, default=500,
                        help='Define the number of epochs for LSTM training')
    parser.add_argument('--lr', type=float, default=0.01,
                        help='Define the learning rate for LSTM training')
    parser.add_argument('--patience', type=int, default=20,
                        help='Define the patience for early stopping')
    parser.add_argument('--min_delta', type=float, default=0.001,
                        help='Define the minimum delta for early stopping')
    parser.add_argument('--factor', type=float, default=0.5,
                        help='Define the factor for reducing the learning rate on plateau')
    parser.add_argument('--fs', type=int, default=30_000,
                        help='Sampling rate for neural data (mua)')
    args = parser.parse_args()

    # Run the main function
    start_time = time.time()
    main(args)
    print(f"\n\nTotal time to run analysis: {time.time() - start_time} seconds")