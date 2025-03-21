from scipy.stats import ranksums
import numpy as np
from decoding_library import *
from core import compute_spike_rates_sliding_window_by_region_smooth, load_behavior, align_brain_and_behavior, compute_sniff_freqs_bins
from plotting import plot_position_trajectories
import os
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib
import multiprocessing as mp
import concurrent.futures
from scipy.io import loadmat
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import time



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


def nan_helper(y):
    """Helper to handle indices and logical indices of NaNs.

    Input:
        - y, 1d numpy array with possible NaNs
    Output:
        - nans, logical indices of NaNs
        - index, a function, with signature indices= index(logical_indices),
          to convert logical indices of NaNs to 'equivalent' indices
    Example:
        >>> # linear interpolation of NaNs
        >>> nans, x= nan_helper(y)
        >>> y[nans]= np.interp(x(nans), x(~nans), y[~nans])
    """
    return np.isnan(y), lambda z: z.nonzero()[0]


def preprocess_rnp(path, bin_size=20, crop_hf=True, flip_data=None, make_plots=False, save_path= None):
    '''
    Get the data from a given file path and preprocess by binning spikes,
    tracking, and sniffing data.
    
    Some preprocessing is also done to exclude 
    units with a large number of refractory violations or negative spike 
    amplitudes (see the readme file).
    
    Parameters
    --
    path : Path to the directory containing the data.
    
    bin_size : Number of 10ms time steps to include in each bin.
    
    crop_hf : If True, crop out the initial and final periods of the 
        session during which the mouse is head fixed.
        
    make_plots : If True, make some plots of the data, saving them as a
        PDF file in the directory containing the data.
        
    flip_data : If flip_data='pre' and there is a floor flip, return 
        only data from before the flip. If flip_data='post' and there 
        is a floor flip, return only data from after the flip.
        
    Returns
    --
    If there is no tracking data for the session, returns None. If there
    is tracking data, the following are returned:
    
    spks : The binned spiking data. 2D array of shape n_bins-by-n_units.
    
    pos_ss : The average x-y position of the mouse's head in each time bin.
        2D array of shape n_bins-by-2.
        
    speed_ss : The average speed (in arbitrary units) of the mouse in each
        time bin. 1D array of length n_bins.
        
    sniff_freq_ss : The average instantaneous sniff frequency (in Hz) in
        each time bin. 1D array of length n_bins. If sniff data is not
        available for the session, returns None.
    '''


    if 'track.mat' in os.listdir(path):
        track = loadmat(path + '/track.mat')
        head = track['track'][:,3:5]
        frame_times_ds = track['track'][:,0]  # frame times in ms (one frame per ~10ms)
    elif 'mtrack.mat' in os.listdir(path):
        track = loadmat(path + '/mtrack.mat')
        head = track['mtrack'][:,2:4]
        frame_times_ds = track['mtrack'][:,0] # frame times in ms (one frame per ~10ms)
    else:
        print('No tracking data for this session.')
        return None, None, None, None


    cluster_ids = loadmat(path + '/cluster_ids.mat')
    spike_times = loadmat(path + '/spike_times.mat')


    frame_times = frame_times_ds * 30  # convert to 30kHz clock

    spike_key = spike_times.keys()
    spike_key = list(spike_key)[-1]
    spikes = spike_times[spike_key][:,0]  # spike times according to ticks of a 30kHz clock
    clusters = cluster_ids['clusters'][:,0]
  
    

    pos = head
    speed = (pos[1:,:] - pos[:-1,:])/np.outer((frame_times_ds[1:] - frame_times_ds[:-1]), np.ones(2))
    speed = np.vstack((speed, np.zeros(2).T))
    
    # Occasionally the shapes of the following two things differ slightly, so chop one:
    if len(frame_times) != len(frame_times_ds):
        print('frame_times and frame_times_ds have different sizes: ', 
              len(frame_times), len(frame_times_ds))
        min_len = np.min([len(frame_times), len(frame_times_ds)])
        frame_times = frame_times[:min_len]
        frame_times_ds = frame_times_ds[:min_len]
        pos = pos[:min_len]
        speed = speed[:min_len]
    n_frames = len(frame_times_ds)

    # Interpolate nans:
    for i in range(2):
        nans, x = nan_helper(pos[:,i])
        if np.sum(nans) > 0:
            pos[nans,i]= np.interp(x(nans), x(~nans), pos[~nans,i])
        nans, x = nan_helper(speed[:,i])
        if np.sum(nans) > 0:
            speed[nans,i]= np.interp(x(nans), x(~nans), speed[~nans,i])

    # Preprocess the sniff data (if it exists):
    if 'sniff_params.mat' in os.listdir(path): 
        sniff = loadmat(path + '/sniff_params.mat')['sniff_params']  # sniff times in ms
        sniffs = sniff[:,0]
        #bad sniffs are sniffs where the third column is zero
        bad_sniffs = np.where(sniff[:,2] == 0)[0]

        sniffs = np.delete(sniffs, bad_sniffs)

        dsniffs = sniffs[1:] - sniffs[:-1]
        sniffs = sniffs[1:]
        sniff_freq = 1000/dsniffs  # instantaneous sniff frequency (in Hz)
        sniff_freq_binned = np.zeros(n_frames)
        for i,t in enumerate(frame_times_ds):
            sniff_freq_binned[i] = np.mean(sniff_freq[(sniffs>t)*(sniffs<t+10*bin_size)])

        # Interpolate nans (in case some bins didn't have sniffs):
        nans, x = nan_helper(sniff_freq_binned)
        if np.sum(nans) > 0:
            sniff_freq_binned[nans]= np.interp(x(nans), x(~nans), sniff_freq_binned[~nans])
    else:
        print('No sniff data for this session.')
        sniff_freq_binned = None



    if 'events.mat' in os.listdir(path): 
        events = loadmat(path + '/events.mat')['events']

        # Event frames and times:
        frame_fm1, t_fm1 = events[0,0], events[0,2]  # frame/time at which initial HF condition ends
        frame_fm2, t_fm2 = events[0,1], events[0,3]  # frame/time at which FM condition begins
        frame_hf1, t_hf1 = events[1,0], events[1,2]  # frame/time at which FM condition ends
        frame_hf2, t_hf2 = events[1,1], events[1,3]  # frame/time at which final HF condition begins
        frame_flip1, t_flip1 = events[2,0], events[2,2]  # frame/time at which floor flip begins
        frame_flip2, t_flip2 = events[2,1], events[2,3]  # frame/time at which floor flip ends

        # Create a mask to handle head-fixed to freely moving transitions and floor flips:
        mask = np.array([True for ii in range(n_frames)])
        color_mask = np.array([0 for ii in range(n_frames)]) # for plotting purposes. 0 for free movement, 1 for headfixed, and 2 for transitions

        if crop_hf:  # crop out the initial and final HF periods
            if frame_fm1!=0:
                mask[:frame_fm2] = False
                color_mask[:frame_fm2] = 1
                color_mask[frame_fm1:frame_fm2] = 2
            elif frame_fm2!=0:
                mask[:frame_fm2] = False
                color_mask[:frame_fm2] = 1

            if frame_hf1!=0:
                mask[frame_hf1:] = False
                color_mask[frame_hf1:] = 1
                color_mask[frame_hf1:frame_hf2] = 2

        else:  # crop out just the transitions between FM and HF. You probably shouldent be using this....
            if frame_fm1!=0:
                mask[frame_fm1:frame_fm2] = False
                color_mask[frame_fm1:frame_fm2] = 2
            if frame_hf1!=0:
                mask[frame_hf1:frame_hf2] = False
                color_mask[frame_hf1:frame_hf2] = 2

        if frame_flip1!=0:  # keep data only from before or after the flip
            mask[frame_flip1:frame_flip2] = False
            color_mask[frame_flip1:frame_flip2] = 2

            if flip_data=='pre':  
                mask[frame_flip1:] = False
            elif flip_data=='post':
                mask[:frame_flip2] = False
                


            # ensuring the length of the mask is at least 20 minutes
            if np.sum(mask) < 12000:
                print('Not enough data to analyze.')
                return None, None, None, None
            
        # plot the sniff frequencies color coded by the 3 conditions
        if sniff_freq_binned is not None and make_plots:
            plt.figure(figsize=(20,8))
            plt.scatter(frame_times_ds[mask == True] / 1000, sniff_freq_binned[mask == True], s=5, marker='.')
            plt.scatter(frame_times_ds[mask == False] / 1000, sniff_freq_binned[mask == False], s=5, marker='.')

            plt.title('Sniff frequency color coded by condition')
            plt.xlabel('Time (s)')
            plt.ylabel('Sniff frequency (Hz)') 
            plt.legend(['Used data', 'Excluded data'])
            plt.tight_layout()
            plt.savefig(save_path + '/sniff_frequency_color_coded.png')
            plt.close()

        # for sessions without sniffing data, plot just a horizontal bar colored based on the mask conditions
        if make_plots:
            plt.figure(figsize=(20,8))
            plt.scatter(frame_times_ds[mask == True] / 1000, np.zeros(np.sum(mask)), s=5, marker='.')
            plt.scatter(frame_times_ds[mask == False] / 1000, np.zeros(np.sum(~mask)), s=5, marker='.')

            plt.title('No sniff data color coded by condition')
            plt.xlabel('Time (s)')
            plt.ylabel('Sniff frequency (Hz)') 
            plt.legend(['Used data', 'Excluded data'])
            plt.tight_layout()
            plt.savefig(save_path + '/no_sniff_data_color_coded.png')
            plt.close()



        # Keep the data selected by the mask; 
        frame_times_ds = frame_times_ds[mask]
        frame_times = frame_times[mask]
        pos = pos[mask,:]
        speed = speed[mask,:]

        # Chop off the last few points if not divisible by bin_size:
        frame_times_ds = frame_times_ds[:bin_size*(len(frame_times_ds)//bin_size)]
        frame_times = frame_times[:bin_size*(len(frame_times)//bin_size)]
        pos = pos[:bin_size*(len(pos)//bin_size)]
        speed = speed[:bin_size*(len(speed)//bin_size)]

        # Do the same thing for the sniff data if it exists:
        if 'sniff_params' in os.listdir(path): 
            sniff_freq_binned = sniff_freq_binned[mask]
            sniff_freq_binned = sniff_freq_binned[:bin_size*(len(sniff_freq_binned)//bin_size)]
        
            # Average the sniff-frequency data within each bin:
            sniff_freq_ss = np.zeros(len(sniff_freq_binned)//bin_size)
            for i in range(len(sniff_freq_binned)//bin_size):
                sniff_freq_ss[i] = np.mean(sniff_freq_binned[i*bin_size:(i+1)*bin_size], axis=0)
        else:
            sniff_freq_ss = None

    # Average the behavioral data within each bin:
    pos_ss = np.zeros((len(pos)//bin_size, 2))
    speed_ss = np.zeros((len(speed)//bin_size, 2))
    for i in range(len(pos)//bin_size):
        pos_ss[i,:] = np.mean(pos[i*bin_size:(i+1)*bin_size,:], axis=0)
        speed_ss[i,:] = np.mean(speed[i*bin_size:(i+1)*bin_size,:], axis=0)

    # Clip and normalize the position data:
    pos_ss[:,0] = np.clip(pos_ss[:,0], np.percentile(pos_ss[:,0], 0.5), np.percentile(pos_ss[:,0], 99.5))
    pos_ss[:,1] = np.clip(pos_ss[:,1], np.percentile(pos_ss[:,1], 0.5), np.percentile(pos_ss[:,1], 99.5))
    pos_ss[:,0] -= np.min(pos_ss[:,0])
    pos_ss[:,1] -= np.min(pos_ss[:,1])
    #pos_ss[:,0] = pos_ss[:,0]*x_max/np.max(pos_ss[:,0])
    #pos_ss[:,1] = pos_ss[:,1]*y_max/np.max(pos_ss[:,1])
        
    # Bin the spiking data:
    spks = np.zeros((0, len(frame_times)//bin_size))
    for cluster in np.unique(clusters):
        # only keep clusters with firing rate > 0.5 Hz:
        c1 = np.sum(spikes[clusters==cluster])/(1e-3*(frame_times[-1] - frame_times[0])) > 0.5

        # < 5% of spikes may violate the 1.5ms refractory period:
        isi = np.diff(spikes[clusters==cluster])
        c2 = np.sum(isi < 1.5)/(1+len(isi)) < 0.05  



        if c1 and c2:
            bin_edges = np.append(frame_times[::bin_size], frame_times[-1])
            spike_counts, _ = np.histogram(spikes[clusters==cluster], bin_edges, density=False)
            # Normalize so that spike counts are in Hz:
            spike_counts = 3e4*spike_counts/(bin_edges[1:] - bin_edges[:-1])
            spks = np.vstack((spks, spike_counts[:len(spks.T)])) 
    spks = spks.T

    if make_plots:
        times = np.arange(len(pos_ss))*0.01*bin_size
        plt.figure(figsize=(9,9))
        plt.subplot(421)
        plt.plot(times, pos_ss[:,0])
        plt.plot(times, pos_ss[:,1])
        plt.xlim(0, times[-1])
        plt.ylabel('x,y')
        plt.xlabel('Time (s)')
        plt.subplot(422)
        plt.plot(pos_ss[:,0], pos_ss[:,1], lw=0.25)
        plt.subplot(423)
        plt.plot(times, np.linalg.norm(speed_ss, axis=1))
        plt.xlim(0, times[-1])
        plt.xlabel('Time (s)')
        plt.ylabel('Speed (a.u.)')
        plt.subplot(424)
        plt.hist(100/bin_size*np.mean(spks, axis=0), bins=30)
        plt.xlabel('Firing rate (Hz)')
        plt.ylabel('Units')
        if sniff_freq_ss is not None:
            plt.subplot(425)
            plt.plot(times, sniff_freq_ss)
            plt.xlim(0, times[-1])
            plt.xlabel('Time (s)')
            plt.ylabel('Sniff frequency (Hz)')
        plt.subplot(427)
        plt.imshow(np.log(1e-3+(spks/np.max(spks, axis=0)).T), aspect='auto', interpolation='none')
        plt.xticks([0, len(spks)], [0, int(times[-1])])
        plt.yticks([0, len(spks.T)-1], [1, len(spks.T)])
        plt.xlabel('Time (s)')
        plt.ylabel('Unit')
        plt.tight_layout()
        if flip_data is not None:
            plt.savefig(save_path + '/data_FNO_F' + flip_data +'.pdf')
        else:
            plt.savefig(save_path + '/data_FNO_F.pdf')

    return spks, pos_ss, speed_ss, sniff_freq_ss



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

    if behavior_dim > 4:
        # remove the y-axis ticks 
        for a in ax:
            a.set_yticks([])
    plt.xlabel('Time')
    ax[0].legend(loc = 'upper right')

    sns.despine()
    plt.savefig(os.path.join(save_path, f'lstm_predictions_k_{k}_shift_{shift}.png'), dpi=300)
    plt.close()


def plot_rmse(true_rmse, null_rmse, p_val, region_save_path):
    plt.figure(figsize=(20, 10))
    plt.hist(true_rmse, bins=10, alpha=0.5, label='True error', color='dodgerblue', density=True)
    plt.hist(null_rmse, bins=10, alpha=0.5, label='Null error', color='crimson', density=True)
    plt.xlabel('RMSE')
    plt.ylabel('Probability Density')
    plt.title(f'RMSE Distribution\np_val: {p_val:.2e}')
    plt.legend()
    sns.despine()
    plt.savefig(os.path.join(region_save_path, 'rmse_distribution.png'), dpi=300)
    plt.close()



def process_fold(spike_rates, behavior, k, shift, current_save_path, behavior_name, behavior_dim, params):


    np.random.seed(int(time.time()) + k)
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
    with concurrent.futures.ProcessPoolExecutor(max_workers=25) as executor:
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

    # Loading the data from the clickbait-ephys project
    if behavior_dir:
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
        # ensure the file exists
        if os.path.exists(sniff_params_file):
            mean_freqs, _ = compute_sniff_freqs_bins(sniff_params_file, time_bins, window_size, sfs = 1000)
        else:
            mean_freqs = np.zeros(len(time_bins))



    # Looping through the regions
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
            behavior_dim = 2
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

            # if fewer than 2 neurons in the region, skip the region
            if rates.shape[0] < 2:
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
                # Ensure at least 1 neuron is present
                if other_rates.shape[0] < 1:
                    print(f"Skipping {mouse}/{session}/{region} due to insufficient neurons.")
                    continue
                data_other = align_brain_and_behavior(events, other_rates, other_units, time_bins, window_size, speed_threshold = speed_threshold)
                data_other['sns'] = mean_freqs
                data_other['sns'] = data['sns'].interpolate(method='linear')
                data_other.dropna(subset=['x', 'y', 'v_x', 'v_y', 'sns'], inplace=True)
                behavior = data.iloc[:, :-8].values
                behavior_name = [f"$Y_{{{i}}}$" for i in range(behavior.shape[1])]
                behavior_dim = behavior.shape[1]
            max_roll = len(behavior) - 100


        print(f"Processing {mouse}/{session}/{region}")

        # Loop through the shifts
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
                true_rmse = parallel_process_session(spike_rates, behavior_use, k_CV, shift, current_save_path, behavior_name, behavior_dim, model_params)
            else:
                null_rmse.append(parallel_process_session(spike_rates, behavior_use, k_CV, shift, current_save_path, behavior_name, behavior_dim, model_params))
        
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






def main(window_size: float = 0.1, step_size: float = 0.1, sigma_smooth: float = 2.5, use_units: str = 'good/mua', speed_threshold: int = 100, n_shifts: int = 1, k_CV: int = 8, n_blocks: int = 10, plot_predictions: bool = True, sequence_length: int = 10, hidden_dim: int = 64, num_layers: int = 2, dropout: float = 0.5, num_epochs: int = 500, lr: float = 0.01, patience: int = 20, min_delta: float = 0.001, factor: float = 0.5, fs: int = 30_000):


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

    # Set plotting style and context
    matplotlib.use('Agg')  # Use non-interactive backend to avoid issues in threaded environment
    plt.style.use('dark_background')
    sns.set_context('poster')


    # defining directories
    spike_dir = "/projects/smearlab/shared/clickbait-ephys(3-20-25)/kilosorted"
    save_dir_main = "/projects/smearlab/shared/clickbait-ephys(3-20-25)/figures"
    events_dir = "/projects/smearlab/shared/clickbait-ephys(3-20-25)/behavior_data"
    SLEAP_dir = "/projects/smearlab/shared/clickbait-ephys(3-20-25)/sleap_predictions"
    sniff_dir = "/projects/smearlab/shared/clickbait-ephys(3-20-25)/sniff events"


    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")


    for target in ['position', 'sniffing', 'neural']:

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
                    params = [mouse, session, kilosort_dir, behavior_dir, tracking_dir, sniff_params_dir, save_dir, window_size, step_size, fs, sigma_smooth, use_units, speed_threshold, k_CV, n_blocks, n_shifts, plot_predictions, target, sequence_length, hidden_dim, num_layers, dropout, num_epochs, lr, patience, min_delta, factor]

                    # Run the decoding
                    try:
                        process_session(params)
                    except Exception as e:
                        print(f"Error processing {mouse}/{session}: {e}")





if __name__ == "__main__":

    mp.set_start_method('spawn', force=True)



    # Set the parameters
    window_size = 0.1
    step_size = 0.1
    sigma_smooth = 2.5
    use_units = 'good/mua'
    speed_threshold = 100
    n_shifts = 4
    k_CV = 25
    n_blocks = 3
    plot_predictions = True
    sequence_length = 10
    hidden_dim = 64
    num_layers = 2
    dropout = 0.1
    num_epochs = 500
    lr = 0.01
    patience = 20
    min_delta = 0.001
    factor = 0.5
    fs = 30_000

    # Run the main function
    main(window_size, step_size, sigma_smooth, use_units, speed_threshold, n_shifts, k_CV, n_blocks, plot_predictions, sequence_length, hidden_dim, num_layers, dropout, num_epochs, lr, patience, min_delta, factor, fs)
