from scipy.stats import ranksums
import numpy as np
from decoding_library import *
from core import compute_spike_rates_sliding_window_by_region_smooth, load_behavior, align_brain_and_behavior, compute_sniff_freqs_bins
from plotting import plot_position_trajectories
import os
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import seaborn as sns
from concurrent.futures import ThreadPoolExecutor, as_completed
import matplotlib
from numpy.lib.stride_tricks import sliding_window_view


class LSTMDecoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers = 1, dropout = 0.1):
        super(LSTMDecoder, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers=num_layers, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        return self.fc(lstm_out[:, -1, :])
    

def train_LSTM(model, X, y, lr=0.01, epochs=1000, patience=50, min_delta=1, factor=0.1):
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
        # Training
        model.train()
        optimizer.zero_grad()
        outputs = model(X)
        loss = criterion(outputs, y)
        loss.backward()
        optimizer.step()

        # Evaluation and early stopping
        if loss.item() < best_loss - min_delta:
            best_loss = loss.item()
            best_model_state = model.state_dict().copy()  # Save a copy of the model state
            counter = 0
        else:
            counter += 1
            if counter >= patience:
                break

        # Learning rate scheduler
        scheduler.step(loss)

        history.append(loss.item())

    
    # Load the best model
    model.load_state_dict(best_model_state)
    
    return model, history
    

def create_sequences(data, sequence_length):
    """
    Vectorized creation of overlapping sequences using sliding_window_view.
    
    Parameters
    ----------
    data : np.array, shape (T, n)
        The input time-series data.
    sequence_length : int
        The length of each sequence.
    
    Returns
    -------
    X : np.array, shape (T - sequence_length, sequence_length, n)
        The overlapping sequences.
    y : np.array, shape (T - sequence_length, n)
        The labels (next time step following each sequence).
    """
    T, n = data.shape
    if T <= sequence_length:
        return np.empty((0, sequence_length, n)), np.empty((0, n))
    
    # Create a sliding window view over the time dimension.
    X = sliding_window_view(data, window_shape=(sequence_length, n))
    X = X.reshape(-1, sequence_length, n)
    y = data[sequence_length:]
    return X, y


def prepare_sequences(X, y, sequence_length, switch_ind):

    X_lstm = []
    y_lstm = []
    for block in range(len(switch_ind) - 1):
        start = switch_ind[block]
        end = switch_ind[block + 1]

        # Create sequences for the current block
        X_block = X[start:end, :]
        y_block = y[start:end, :][]
        X_block, _ = create_sequences(X_block, sequence_length)
        y_block = y_block[sequence_length:]
        
        # Append to the training data
        X_train_lstm.append(X_block)
        y_train_lstm.append(y_block)

    # Concatenate all blocks
    X_train_lstm = np.concatenate(X_train_lstm, axis=0)
    y_train_lstm = np.concatenate(y_train_lstm, axis=0)

    # Convert the training data to tensors
    X_train_lstm = torch.tensor(X_train_lstm, dtype=torch.float32)
    y_train_lstm = torch.tensor(y_train_lstm, dtype=torch.float32)


def process_session(params, device):


    # Unpack the parameters
    mouse, session, kilosort_dir, behavior_dir, tracking_dir, sniff_dir, save_dir, window_size, step_size, fs, sigma_smooth, use_units, speed_threshold, k_CV, n_blocks, n_shifts, plot_predictions, target, sequence_length, hidden_dim, num_layers, dropout, num_epochs, lr = params

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


        # Unpack the results
        true_rmse = []
        null_rmse = []

        # Loop through the shifts
        for shift in range(n_shifts + 1):
            if shift == 0:
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
                current_save_path = os.path.join(region_save_path, 'true')
                os.makedirs(current_save_path, exist_ok=True)
            else:
                roll_value = np.random.randint(100, len(data) - 100)
                if target == 'position':
                    behavior = data[['x', 'y']].values.copy()
                elif target == 'sniffing':
                    behavior = data['sns'].values.copy()
                    behavior = behavior.reshape(-1, 1)
                elif target == 'velocity':
                    behavior = data[['v_x', 'v_y']].values.copy()
                elif target == 'neural':
                    behavior = other_rates.copy()

                
                behavior = np.roll(behavior, roll_value, axis=0)
                current_save_path = os.path.join(region_save_path, 'null')
                os.makedirs(current_save_path, exist_ok=True)

            # Noramlize the behavior data
            behavior_mean = np.mean(behavior, axis=0)
            behavior_std = np.std(behavior, axis=0)
            behavior = (behavior - behavior_mean) / behavior_std


            # Loop through the cross-validation folds
            for k in range(k_CV):
                rates_train, rates_test, train_switch_ind, test_switch_ind = cv_split(spike_rates, k, k_CV=k_CV, n_blocks=n_blocks)
                behavior_train, behavior_test, _, _ = cv_split(behavior, k, k_CV=k_CV, n_blocks=n_blocks)

                # Create the model
                lstm_model = LSTMDecoder(input_dim=rates_train.shape[1], hidden_dim=hidden_dim, output_dim=behavior.shape[1], num_layers=num_layers, dropout = dropout).to(device)

                # Prepare the training data for LSTM
                X_train_lstm, y_train_lstm = prepare_sequences(rates_train, behavior_train, sequence_length)

                
                # Training the LSTM model
                trained_lstm_model, lstm_history = train_LSTM(lstm_model, X_train_lstm, y_train_lstm, lr=0.001, epochs=num_epochs, patience=100, min_delta=.001, factor=0.1)
                optimal_loss = min(lstm_history)
                model_used_index = lstm_history.index(optimal_loss)

                # Plot loss
                if plot_predictions:
                    plt.figure(figsize=(10, 6))
                    sns.lineplot(x=np.arange(len(lstm_history)), y=lstm_history, linewidth=4, color='blue')
                    plt.title(f'LSTM Training Loss')
                    plt.xlabel('Epoch')
                    plt.ylabel('Loss (Mean Squared Error)')
                    plt.yscale('log')
                    if target == 'sniffing':
                        plt.ylim(1e-3, 10)
                    else:
                        plt.ylim(1, 1e4)
                    plt.scatter(model_used_index, optimal_loss, color='red', s=100)
                    sns.despine()
                    plt.tight_layout()
                    plt.savefig(os.path.join(current_save_path, f'lstm_loss_{k}_shift_{shift}.png'), dpi=300)
                    plt.close()

                # Prepare the test data for LSTM
                X_test_lstm = []
                y_test_lstm = []
                for block in range(len(test_switch_ind) - 1):
                    start = test_switch_ind[block]
                    end = test_switch_ind[block + 1]

                    # Create sequences for the current block
                    rates_block = rates_test[start:end, :]
                    behavior_block = behavior_test[start:end, :]
                    X_block, _ = create_sequences(rates_block, sequence_length)
                    y_block = behavior_block[sequence_length:]

                    # Append to the test data
                    X_test_lstm.append(X_block)
                    y_test_lstm.append(y_block)

                # Concatenate all blocks
                X_test_lstm = np.concatenate(X_test_lstm, axis=0)
                y_test_lstm = np.concatenate(y_test_lstm, axis=0)

                # Convert the test data to tensors
                X_test_lstm = torch.tensor(X_test_lstm, dtype=torch.float32).to(device)
                y_test_lstm = torch.tensor(y_test_lstm, dtype=torch.float32).to(device)


                # Predict on the test set
                lstm_model.eval()
                with torch.no_grad():
                    predictions = trained_lstm_model(X_test_lstm)
                
                # Converting the predictions and true values back to original scale
                predictions = predictions.cpu().numpy() * behavior_std + behavior_mean
                y_test_lstm = y_test_lstm.cpu().numpy() * behavior_std + behavior_mean

                # plotting the predicted and true values
                if plot_predictions:
                    adjusted_test_switch_ind = [ind - sequence_length * k for k, ind in enumerate(test_switch_ind)]
                    fig, ax = plt.subplots(behavior_dim, 1, figsize=(20, 10))
                    if behavior_dim == 1:
                        ax = [ax]
                    for i in range(behavior_dim):
                        ax[i].plot(y_test_lstm[:, i], label='True', color = 'crimson')
                        ax[i].plot(predictions[:, i], label='Predicted')
                        ax[i].set_ylabel(behavior_name[i])
                        for ind in adjusted_test_switch_ind:
                            ax[i].axvline(ind, color='grey', linestyle = '--', alpha=0.5)
                        ax[i].legend()
                    sns.despine()
                    plt.savefig(os.path.join(current_save_path, f'lstm_predictions_k_{k}_shift_{shift}.png'), dpi=300)
                    plt.close()

                # Calculate Euclidean distance error at each time point
                euclidean_distances = np.sqrt(np.sum((predictions - y_test_lstm)**2, axis=1))

                # Calculate RMSE of these distances
                rmse = np.sqrt(np.mean(euclidean_distances**2))

                # Store the MSE
                if shift == 0:
                    true_rmse.append(rmse)
                else:
                    null_rmse.append(rmse)

        true_rmse = np.array(true_rmse)
        null_rmse = np.array(null_rmse)

        # rank sum test
        _ , p_val = ranksums(true_rmse, null_rmse, 'less')
        print(f"Rank sum test p-value for {mouse}/{session}: {p_val}")
        
        # plot histograms of the rmse
        plt.figure(figsize=(20, 10))
        plt.hist(true_rmse, bins=5, alpha=0.5, label='True MSE', color='dodgerblue')
        plt.hist(null_rmse, bins=5, alpha=0.5, label='Null MSE', color='crimson')
        plt.xlabel('RMSE')
        plt.ylabel('Count')
        plt.title(f'RMSE Distribution for {mouse}/{session}/{region}/p_val: {p_val:.2e}')
        plt.legend()
        plt.savefig(os.path.join(region_save_path, 'rmse_distribution.png'), dpi=300)
        plt.close()

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
    save_dir_main = r"E:\clickbait-ephys\figures\LSTM (3-18-25)"
    events_dir = r"D:\clickbait-ephys\behavior_data"
    SLEAP_dir = r"D:\clickbait-ephys\sleap_predictions"
    sniff_dir = r"D:\clickbait-ephys\sniff events"


    # defining the subset of data to process
    mice = ['6002']
    sessions = ['7', '8', '9', '10', '11', '12', '13', '14', '0', '1', '2', '3', '4', '5', '6', ]


    # defining the data variables
    fs = 30_000  # sampling rate for neural data (mua.npy)


    # Defining the data preprocessing parameters
    window_size = .1  # Window size for spike rate computation (in seconds)
    step_size = .1  # Step size for sliding window (in seconds)
    sigma_smooth = 2.5  # Standard deviation for gaussian smoothing of spike rates
    use_units = 'good' # What kilosort cluster labels to use
    speed_threshold = 100  # Tracking point with speed above this value will be removed before interpolation


    # Defining the decoding parameters
    n_shifts = 2 # Define number of shifts for circular shifting of behavior data
    k_CV = 10 # Define number of cross-validation folds
    n_blocks = 12 # Define number of blocks for cross-validation
    plot_predictions = True
    sequence_length = 10 # Define the sequence length for LSTM input
    hidden_dim = 64 # Define the hidden dimension for LSTM
    num_layers = 2 # Define the number of layers for LSTM
    dropout = 0.1 # Define the dropout for LSTM
    num_epochs = 2000 # Define the number of epochs for LSTM training
    lr = 0.01

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")


    for target in ['sniffing']:

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
            f.write(f"save_dir: {save_dir}\n")


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
                    params = [mouse, session, kilosort_dir, behavior_dir, tracking_dir, sniff_params_dir, save_dir, window_size, step_size, fs, sigma_smooth, use_units, speed_threshold, k_CV, n_blocks, n_shifts, plot_predictions, target, sequence_length, hidden_dim, num_layers, dropout, num_epochs, lr]

                    # decode the data

                    process_session(params, device)










if __name__ == "__main__":
    main()
