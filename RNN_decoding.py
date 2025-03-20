from scipy.stats import ranksums
import numpy as np
from decoding_library import *
from core import compute_spike_rates_sliding_window_by_region_smooth, load_behavior, align_brain_and_behavior, compute_sniff_freqs_bins
from plotting import plot_position_trajectories
import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib
import concurrent.futures



class LSTMDecoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers = 1, dropout = 0.1):
        super(LSTMDecoder, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers=num_layers, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        return self.fc(lstm_out[:, -1, :])
    

def train_LSTM(model, train_loader, device, lr=0.01, epochs=1000, patience=50, min_delta=1, factor=0.1):
    """
    Train the LSTM model with early stopping and learning rate scheduling.
    
    Parameters
    ----------
    model : MLPModel
        The model to train
    X : torch.Tensor
        Input features
    y : torch.Tensor
        Target values
    lr : float
        Initial learning rate
    epochs : int
        Maximum number of epochs
    patience : int
        Number of epochs with no improvement after which training will be stopped
    min_delta : float
        Minimum change in loss to qualify as an improvement
    factor : float
        Factor by which the learning rate will be reduced
        
    Returns
    -------
    model : MLPModel
        The trained model
    history : list
        Training loss history
    """
    # Initialize the training parameters
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=factor, patience=patience)
    
    best_loss = float('inf')
    best_model_state = model.state_dict().copy()  # Save a copy of the model state
    counter = 0

    # Training the model
    history = []
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            outputs = model(X_batch.to(device, non_blocking=True))
            loss = criterion(outputs, y_batch.to(device, non_blocking=True))
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * X_batch.size(0)

        # Average loss
        epoch_loss /= len(train_loader.dataset)

        # Evaluation and early stopping
        if epoch_loss < best_loss - min_delta:
            best_loss = epoch_loss
            best_model_state = model.state_dict().copy()  # Save a copy of the model state
            counter = 0
        else:
            counter += 1
            if counter >= patience:
                break

        # Learning rate scheduler and history
        scheduler.step(epoch_loss)
        history.append(epoch_loss)

    # Load the best model
    model.load_state_dict(best_model_state)
    
    return model, history
    

class SequenceDataset(Dataset):
    def __init__(self, rates, behavior, blocks, sequence_length):
        # Pre-convert to torch tensors once
        self.rates = torch.tensor(rates, dtype=torch.float32)
        self.behavior = torch.tensor(behavior, dtype=torch.float32)
        self.sequence_length = sequence_length
        self.indices = []
        for start, end in blocks:
            for i in range(start, end - sequence_length):
                self.indices.append(i)
    
    def __len__(self):
        return len(self.indices)
    
    def __getitem__(self, idx):
        i = self.indices[idx]
        # Now slicing is done on pre-converted tensors
        X = self.rates[i: i + self.sequence_length, :]
        y = self.behavior[i + self.sequence_length, :]
        return X, y


def plot_training(lstm_history, save_path, shift, k):

    optimal_loss = min(lstm_history)
    model_used_index = lstm_history.index(optimal_loss)

    plt.figure(figsize=(10, 6))
    sns.lineplot(x=np.arange(len(lstm_history)), y=lstm_history, linewidth=4, color='blue')
    plt.title(f'LSTM Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss (MSE)')
    plt.yscale('log')
    plt.scatter(model_used_index, optimal_loss, color='red', s=100)
    sns.despine()
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, f'lstm_loss_{k}_shift_{shift}.png'), dpi=300)
    plt.close()


def plot_preds(targets, predictions, test_switch_ind, behavior_name, behavior_dim, sequence_length, save_path, k, shift):
    adjusted_test_switch_ind = [ind - sequence_length * k for k, ind in enumerate(test_switch_ind)]
    _, ax = plt.subplots(behavior_dim, 1, figsize=(20, 10))
    if behavior_dim == 1:
        ax = [ax]
    for i in range(behavior_dim):
        ax[i].plot(targets[:, i], label='True', color = 'crimson')
        ax[i].plot(predictions[:, i], label='Predicted')
        ax[i].set_ylabel(behavior_name[i])
        for ind in adjusted_test_switch_ind:
            ax[i].axvline(ind, color='grey', linestyle = '--', alpha=0.5)
        ax[i].legend()
    sns.despine()
    plt.savefig(os.path.join(save_path, f'lstm_predictions_k_{k}_shift_{shift}.png'), dpi=300)
    plt.close()


def plot_rmse(true_rmse, null_rmse, p_val, region_save_path):
    plt.figure(figsize=(20, 10))
    plt.hist(true_rmse, bins=4, alpha=0.5, label='True error', color='dodgerblue')
    plt.hist(null_rmse, bins=4, alpha=0.5, label='Null error', color='crimson')
    plt.xlabel('RMSE')
    plt.ylabel('Count')
    plt.title(f'RMSE Distribution\np_val: {p_val:.2e}')
    plt.legend()
    plt.yticks([0, 1, 2, 3], ['0', '1', '2', '3'])
    sns.despine()
    plt.savefig(os.path.join(region_save_path, 'rmse_distribution.png'), dpi=300)
    plt.close()



def process_fold(spike_rates, behavior, k, shift, current_save_path, behavior_name, behavior_dim, params):

    plt.style.use('dark_background')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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
    train_loader = DataLoader(train_dataset, batch_size=16_384, shuffle=False, num_workers=8, pin_memory=True, prefetch_factor=8, persistent_workers=True)

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
    test_loader = DataLoader(test_dataset, batch_size=16_384, persistent_workers=True, num_workers=1, pin_memory=True)

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
        plot_preds(targets, predictions, test_switch_ind, behavior_name, behavior_dim, sequence_length, current_save_path, k, shift)

    # Calculate the RMSE
    rmse = np.sqrt(np.mean((predictions - targets) ** 2))

    # Final clean up
    del predictions, targets
    torch.cuda.empty_cache()

    return rmse


def parallel_process_session(spike_rates, behavior, n_folds, shift, current_save_path, 
                             behavior_name, behavior_dim, model_params):
    """
    Run process_fold in parallel for all folds.
    
    Parameters:
        spike_rates: Full spike rates array.
        behavior: Normalized behavior array.
        n_folds: Number of CV folds.
        shift: The current shift value.
        current_save_path: Where to save outputs for this region/shift.
        behavior_mean, behavior_std, behavior_name, behavior_dim: Behavior scaling and labeling.
        device: The CUDA device to use.
        model_params: List of model and training parameters (k_CV, n_blocks, plot_predictions, etc.).
    
    Returns:
        List of RMSE values, one per fold.
    """
    rmse_list = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=4) as executor:
        futures = []
        for k in range(n_folds):
            # Submit each fold as a separate process.
            futures.append(executor.submit(process_fold,
                                             spike_rates, behavior, k, shift, current_save_path,
                                                behavior_name, behavior_dim, model_params))
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
    mouse, session, kilosort_dir, behavior_dir, tracking_dir, sniff_dir, save_dir, window_size, step_size, fs, sigma_smooth, use_units, speed_threshold, k_CV, n_blocks, n_shifts, plot_predictions, target, sequence_length, hidden_dim, num_layers, dropout, num_epochs, lr, patience, min_delta, factor = params
    model_params = [k_CV, n_blocks, plot_predictions, sequence_length, hidden_dim, num_layers, dropout, num_epochs, lr, patience, min_delta, factor]

    # Load the spike rates
    print('Loading Data...\n')
    rates_OB, rates_HC, time_bins, units_OB, units_HC = compute_spike_rates_sliding_window_by_region_smooth(kilosort_dir, fs, window_size, step_size, use_units=use_units, sigma=sigma_smooth, zscore=True)

    # load the tracking data
    files = os.listdir(tracking_dir)
    tracking_file = [f for f in files if f.endswith('.analysis.h5')][0]
    tracking_file = os.path.join(tracking_dir, tracking_file)
    events = load_behavior(behavior_dir, tracking_file)

    # Loading the sniffing
    sniff_params_file = os.path.join(sniff_dir, 'sniff_params.mat')
    mean_freqs, _ = compute_sniff_freqs_bins(sniff_params_file, time_bins, window_size, sfs = 1000)


    # Looping through the regions
    for region in ['HC', 'OB']:
        print(f"Processing {mouse}/{session}/{region}...")
        if region == 'OB':
            rates = rates_OB
            units = units_OB
            other_rates = rates_HC
        else:
            rates = rates_HC
            units = units_HC
            other_rates = rates_OB

        # if fewer than 2 neurons in the region, skip the region
        if rates.shape[1] < 2:
            print(f"Skipping {mouse}/{session}/{region} due to insufficient neurons.")
            continue

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
            behavior_dim = 2
        elif target == 'sniffing':
            behavior = data['sns'].values
            behavior_name = ['sniffing']
            behavior_dim = 1
            behavior = behavior.reshape(-1, 1)
        elif target == 'velocity':
            behavior = data[['v_x', 'v_y']].values
            behavior_name = ['v_x', 'v_y']
            behavior_dim = 2
        elif target == 'neural':
            behavior = other_rates
            behavior_name = [f'neuron_{i}' for i in range(other_rates.shape[1])]
            behavior_dim = other_rates.shape[1]


        # Loop through the shifts
        for shift in range(n_shifts + 1):
            if shift == 0:
                current_save_path = os.path.join(region_save_path, 'true')
                os.makedirs(current_save_path, exist_ok=True)
            else:
                roll_value = np.random.randint(100, len(data) - 100)
                behavior = np.roll(behavior, roll_value, axis=0)
                current_save_path = os.path.join(region_save_path, 'null')
                os.makedirs(current_save_path, exist_ok=True)

            # Loop through the cross-validation folds to get decoding errors
            null_rmse = []
            if shift == 0:
                true_rmse = parallel_process_session(spike_rates, behavior, k_CV, shift, current_save_path, behavior_name, behavior_dim, model_params)
            else:
                null_rmse.append(parallel_process_session(spike_rates, behavior, k_CV, shift, current_save_path, behavior_name, behavior_dim, model_params))
        
        true_rmse = np.array(true_rmse).flatten()
        null_rmse = np.array(null_rmse).flatten()

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







def main():

    # Set plotting style and context
    matplotlib.use('Agg')  # Use non-interactive backend to avoid issues in threaded environment
    plt.style.use('dark_background')
    sns.set_context('poster')


    # defining directories
    spike_dir = r"D:\clickbait-ephys\kilosorted"
    save_dir_main = r"E:\clickbait-ephys\figures\LSTM (3-19-25)"
    events_dir = r"D:\clickbait-ephys\behavior_data"
    SLEAP_dir = r"D:\clickbait-ephys\sleap_predictions"
    sniff_dir = r"D:\clickbait-ephys\sniff events"


    # defining the subset of data to process
    mice = ['6002']
    sessions = ['7', '8', '9', '10', '11', '12', '13', '14', '0', '1', '2', '3', '4', '5', '6', ]


    # defining the data variables
    fs = 30_000  # sampling rate for neural data (mua.npy)


    # Defining the data preprocessing parameters
    window_size = 1  # Window size for spike rate computation (in seconds)
    step_size = 1  # Step size for sliding window (in seconds)
    sigma_smooth = 2.5  # Standard deviation for gaussian smoothing of spike rates
    use_units = 'good/mua' # What kilosort cluster labels to use
    speed_threshold = 100  # Tracking point with speed above this value will be removed before interpolation


    # Defining the decoding parameters
    n_shifts = 1 # Define number of shifts for circular shifting of behavior data
    k_CV = 8 # Define number of cross-validation folds
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


    for target in ['position', 'velocity', 'sniffing', 'neural']:

        # creating a directory to save the figures
        save_dir = os.path.join(save_dir_main, f"window_size_{window_size}_target_{target}")
        os.makedirs(save_dir, exist_ok=True)

        # saving a textfile with all parameters
        with open(os.path.join(save_dir, 'parameters.txt'), 'w') as f:
            f.write(f"window_size: {window_size}\n")
            f.write(f"step_size: {step_size}\n")
            f.write(f"sigma_smooth: {sigma_smooth}\n")
            f.write(f"use_units: {use_units}\n")
            f.write(f"speed_threshold: {speed_threshold}\n")
            f.write(f"n_shifts: {n_shifts}\n")
            f.write(f"k_CV: {k_CV}\n")
            f.write(f"n_blocks: {n_blocks}\n")
            f.write(f'Target: {target}\n')
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
                    kilosort_dir = os.path.join(spike_dir, mouse, session)
                    behavior_dir = os.path.join(events_dir, mouse, session)
                    tracking_dir = os.path.join(SLEAP_dir, mouse, session)
                    sniff_params_dir = os.path.join(sniff_dir, mouse, session)
                    if not os.path.exists(kilosort_dir) or not os.path.exists(behavior_dir) or not os.path.exists(tracking_dir) or not os.path.exists(sniff_dir):
                        print(f"Skipping {mouse}/{session} due to missing data.")
                        continue
                    params = [mouse, session, kilosort_dir, behavior_dir, tracking_dir, sniff_params_dir, save_dir, window_size, step_size, fs, sigma_smooth, use_units, speed_threshold, k_CV, n_blocks, n_shifts, plot_predictions, target, sequence_length, hidden_dim, num_layers, dropout, num_epochs, lr, patience, min_delta, factor]


                    # Run the decoding
                    try:
                        process_session(params)
                    except Exception as e:
                        print(f"Error processing {mouse}/{session}: {e}")










if __name__ == "__main__":
    main()
