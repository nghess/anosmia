import numpy as np
import matplotlib.pyplot as plt




"""
preprocessing functions
"""

def prepare_latencies_for_kalman(latency_matrix, method: str = 'log', max_latency=1.0, epsilon=1e-6, tau=0.1):
    """
    Prepare spike latency matrix for Kalman filter decoding
    
    Parameters:
    -----------
    latency_matrix : numpy.ndarray
        Matrix of spike latencies (units × time_steps)
    max_latency : float
        Maximum latency value to use for NaN replacement
    epsilon : float
        Small constant to avoid log(0)
        
    Returns:
    --------
    processed_latencies : numpy.ndarray
        Processed latency matrix ready for Kalman filter
    """
    # Replace NaNs with max_latency
    if method == 'log':
        filled_matrix = np.copy(latency_matrix)
        filled_matrix[np.isnan(filled_matrix)] = max_latency
        
        # Apply log transform
        log_latencies = np.log(filled_matrix + epsilon)
        
        # Z-score normalize each unit
        means = np.mean(log_latencies, axis=1, keepdims=True)
        std = np.std(log_latencies, axis=1, keepdims=True)
        
        normalized_latencies = (log_latencies - means) / std

    elif method == 'exp':
        filled_matrix = np.copy(latency_matrix)
        filled_matrix[np.isnan(filled_matrix)] = max_latency
        
        # Apply exp transform
        exp_latencies = np.exp(-filled_matrix / tau)
        
        # Z-score normalize each unit
        means = np.mean(exp_latencies, axis=1, keepdims=True)
        std = np.std(exp_latencies, axis=1, keepdims=True)
        
        normalized_latencies = (exp_latencies - means) / std
    else:
        raise ValueError("Invalid method. Use 'log' or 'exp'.")
    
    return normalized_latencies


"""
Train test split
"""
def cv_split(data, k, k_CV=10, n_blocks=10):
    '''
    Perform cross-validation split of the data, following the Hardcastle et 
    al paper.
    
    Parameters
    --
    data : An array of data.
    
    k : Which CV subset to hold out as testing data (integer from 0 to k_CV-1).
    
    k_CV : Number of CV splits (integer).
        
    n_blocks : Number of blocks for initially partitioning the data. The testing
        data will consist of a fraction 1/k_CV of the data from each of these
        blocks.
        
    Returns
    --
    data_train, data_test, switch_indices : 
        - Data arrays after performing the train/test split
        - Indices in the train and test data where new blocks begin
    '''

    block_size = len(data)//n_blocks
    mask_test = [False for _ in data]
    
    # Keep track of which indices in the original data are the start of test blocks
    test_block_starts = []
    train_block_starts = []
    
    for block in range(n_blocks):
        i_start = int((block + k/k_CV)*block_size)
        i_stop = int(i_start + block_size//k_CV)
        mask_test[i_start:i_stop] = [True for _ in range(block_size//k_CV)]
        test_block_starts.append(i_start)
        
    mask_train = [not a for a in mask_test]
    data_test = data[mask_test]
    data_train = data[mask_train]

    train_switch_indices = [0]
    test_switch_indices = [0]
    train_count = 0
    test_count = 0
    for i in range(len(data)-1):
        if mask_train[i]:
            train_count += 1
        if mask_test[i]:
            test_count += 1
        if not mask_train[i] and mask_train[i + 1]:
            train_switch_indices.append(train_count)
        if not mask_test[i] and mask_test[i + 1]:
            test_switch_indices.append(test_count)

    train_switch_indices = np.unique(train_switch_indices)
    test_switch_indices = np.unique(test_switch_indices)

    
    return data_train, data_test, train_switch_indices, test_switch_indices


"""
Decoders
"""

class KalmanFilterDecoder:
    """
    A Kalman Filter implementation for decoding behavioral states from neural activity.
    
    This class implements a linear Kalman filter to estimate position and velocity
    states from neural firing rate data. It includes methods for training the model
    parameters and using the filter for decoding.
    """

    def __init__(self, state_dim, lambda_reg=1e-5):
        """
        Initialize the Kalman Filter decoder

        Parameters
        ----------
        state_dim : int, optional
            Dimensionality of the state vector, default is 4 (x, y, vx, vy)
        lambda_reg : float, optional
            Regularization parameter for least squares estimation, default is 1e-5
        """

        self.state_dim = state_dim
        self.lambda_reg = lambda_reg

        # Model parameters to be estimated durring training
        self.A = None
        self.H = None
        self.Q = None
        self.R = None

        # initial state and covariance
        self.x_hat = None
        self.P_hat = None

    def train(self, X, Y):

        """
        Train the Kalman Filter by estimating parameters from training data.
        
        Parameters
        ----------
        X : ndarray
            Training behavioral data of shape (n_samples, state_dim)
        Y : ndarray
            Training neural data of shape (n_samples, n_neurons)
            
        Returns
        -------
        self : KalmanFilterDecoder
            Returns self for method chaining
        """

        # Estimate the transition matrix A using regularized least squares
        I = np.eye(self.state_dim)
        self.A = (X[:-1].T @ X[1:]) @ np.linalg.inv(X[:-1].T @ X[:-1] + self.lambda_reg * I)

        # Estimate the observation matrix H
        self.H = (Y.T @ X) @ np.linalg.inv(X.T @ X + self.lambda_reg * I)

        # Compute the process noise covariance Q
        X_errors = X[1:] - (self.A @ X[:-1].T).T
        self.Q = (X_errors.T @ X_errors) / (X_errors.shape[0] - 1)

        # Compute the observation noise covariance R
        Y_pred = (self.H @ X.T).T
        Y_errors = Y - Y_pred
        self.R = (Y_errors.T @ Y_errors) / (Y.shape[0] - 1)

        # Set initial state and covariance
        self.x_hat = np.zeros(self.state_dim)
        self.P_hat = self.Q.copy()

        return self
    
    def decode(self, Y, switch_indices=None, gain: float = 1.0):
        """
        Decode behavioral states from neural activity using the Kalman filter.
        
        Parameters
        ----------
        Y : ndarray
            Neural activity data to decode, shape (n_samples, n_neurons)
        switch_indices : ndarray or list, optional
            Indices where to reset the filter state, default is None
            
        Returns
        -------
        x_est : ndarray
            Estimated behavioral states, shape (n_samples, state_dim)
        P : ndarray
            Estimated state covariances, shape (n_samples, state_dim, state_dim)
        """

        if self.A is None or self.H is None:
            raise ValueError("Model parameters A and H must be trained before decoding.")
        
        T = len(Y)
        x_est = np.zeros((T, self.state_dim))
        P = np.zeros((T, self.state_dim, self.state_dim))

        # Initialize state and covariance
        x_est[0] = self.x_hat
        P[0] = self.P_hat

        # Run the Kalman filter
        for t in range(1, T):
            # Reset state if this is a switch index
            if switch_indices is not None and t in switch_indices:
                x_est[t] = self.x_hat
                P[t] = self.P_hat
                continue
                
            # Prediction step
            x_pred = self.A @ x_est[t-1]
            P_pred = self.A @ P[t-1] @ self.A.T + self.Q
            
            # Update step with current neural activity
            y = Y[t]  # Current neural observation
            y_pred = self.H @ x_pred  # Predicted neural activity
            y_tilde = y - y_pred  # Innovation
            S = self.H @ P_pred @ self.H.T + self.R  # Innovation covariance
            K = gain * (P_pred @ self.H.T @ np.linalg.inv(S))  # Kalman gain
            
            x_est[t] = x_pred + K @ y_tilde  # Updated state
            P[t] = (np.eye(self.state_dim) - K @ self.H) @ P_pred  # Updated covariance
        
        return x_est, P
    
    def plot_transition_matrix(self, state_labels=None):
        """
        Plot the state transition matrix as a heatmap.
        
        Parameters
        ----------
        state_labels : list, optional
            Labels for the state variables, default is ['x', 'y', 'vx', 'vy']
            
        Returns
        -------
        fig : matplotlib.figure.Figure
            The matplotlib figure object
        """
        if self.A is None:
            raise ValueError("Model not trained. Call train() first.")
            
        if state_labels is None:
            state_labels = ['x', 'y', 'vx', 'vy']
            
        fig = plt.figure(figsize=(8, 6))
        im = plt.imshow(self.A, cmap='coolwarm', vmin=-1, vmax=1)
        plt.colorbar(im, fraction=0.046, pad=0.04)
        plt.title('Transition Matrix A', fontsize=14)
        plt.xlabel('State Variables', fontsize=12)
        plt.ylabel('State Variables', fontsize=12)
        plt.xticks(ticks=np.arange(self.state_dim), labels=state_labels, fontsize=10)
        plt.yticks(ticks=np.arange(self.state_dim), labels=state_labels, fontsize=10)
        
        # Add text annotations with the values in each cell
        for i in range(self.state_dim):
            for j in range(self.state_dim):
                text_color = 'white' if abs(self.A[i, j]) > 0.5 else 'black'
                plt.text(j, i, f'{self.A[i, j]:.2f}', 
                         ha='center', va='center', 
                         color=text_color, fontsize=10)
        
        plt.tight_layout()
        return fig
    
    def plot_results(self, x_est, true_behavior=None, switch_indices=None):
        """
        Plot the decoded behavioral states compared to ground truth.
        
        Parameters
        ----------
        x_est : ndarray
            Estimated behavioral states from decode()
        true_behavior : ndarray, optional
            Ground truth behavioral data, default is None
        switch_indices : list or ndarray, optional
            Indices where filter state was reset, default is None
            
        Returns
        -------
        fig : matplotlib.figure.Figure
            The matplotlib figure object
        """
        pred_x, pred_y, pred_vx, pred_vy = x_est[:, 0], x_est[:, 1], x_est[:, 2], x_est[:, 3]
        
        # Set up plot
        fig, axs = plt.subplots(2, 2, figsize=(20, 12))
        
        # Common styling function
        def style_subplot(ax, title, y_label):
            ax.set_title(title, fontsize=18, fontweight='bold', pad=15)
            ax.set_xlabel('Time Steps', fontsize=14)
            ax.set_ylabel(y_label, fontsize=14)
            ax.tick_params(axis='both', labelsize=12)
            ax.set_facecolor('white')
            
            # Add vertical lines at switch indices
            if switch_indices is not None:
                for i in switch_indices:
                    ax.axvline(x=i, color='#7f7f7f', linestyle='--', linewidth=1.0, alpha=0.5)
            
            # Style the legend
            if true_behavior is not None:
                ax.legend(fontsize=14, frameon=True, framealpha=0.9, 
                          loc='upper right', edgecolor='#CCCCCC')
        
        # Position X
        axs[0, 0].plot(pred_x, color='#d62728', linestyle='--', linewidth=2.5, alpha=0.9, label="Predicted X")
        if true_behavior is not None:
            true_x = true_behavior[:, 0]
            axs[0, 0].plot(true_x, color='#1f77b4', linewidth=2.5, label="True X")
        style_subplot(axs[0, 0], "Position X", "Position (units)")
        
        # Position Y
        axs[0, 1].plot(pred_y, color='#d62728', linestyle='--', linewidth=2.5, alpha=0.9, label="Predicted Y")
        if true_behavior is not None:
            true_y = true_behavior[:, 1]
            axs[0, 1].plot(true_y, color='#1f77b4', linewidth=2.5, label="True Y")
        style_subplot(axs[0, 1], "Position Y", "Position (units)")
        
        # Velocity X
        axs[1, 0].plot(pred_vx, color='#d62728', linestyle='--', linewidth=2.5, alpha=0.9, label="Predicted Vx")
        if true_behavior is not None:
            true_vx = true_behavior[:, 2]
            axs[1, 0].plot(true_vx, color='#1f77b4', linewidth=2.5, label="True Vx")
        style_subplot(axs[1, 0], "Velocity X", "Velocity (units/time)")
        
        # Velocity Y
        axs[1, 1].plot(pred_vy, color='#d62728', linestyle='--', linewidth=2.5, alpha=0.9, label="Predicted Vy")
        if true_behavior is not None:
            true_vy = true_behavior[:, 3]
            axs[1, 1].plot(true_vy, color='#1f77b4', linewidth=2.5, label="True Vy")
        style_subplot(axs[1, 1], "Velocity Y", "Velocity (units/time)")
        
        # Add a super title
        fig.suptitle('State Prediction: True vs Predicted Values', 
                     fontsize=22, fontweight='bold', y=0.98)
        
        # Add text explaining the vertical lines if switch_indices is not None
        if switch_indices is not None:
            fig.text(0.5, 0.005, 'Vertical dashed lines indicate block boundaries', 
                     ha='center', fontsize=14, style='italic')
    
        # Adjust layout
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])

        return fig



"""
Plotting functions
"""


"""Helper functions
"""

