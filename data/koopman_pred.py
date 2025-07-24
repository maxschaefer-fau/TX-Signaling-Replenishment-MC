import joblib
import numpy as np
import matplotlib.pyplot as plt
# from pykoopman import EDMDc

# Load the trained model
model = joblib.load('trained_dmdc_model.pkl')

eigvals = np.linalg.eigvals(model.A)
print("Max abs eigenvalue of A:", np.max(np.abs(eigvals)))
print("B matrix max abs value:", np.max(np.abs(model.B)))


# Load and prepare the data
def load_data(file_path='koopman_state_data_ideal_7s.npy', control=False):
    """
    Load the state data and prepare the DMD input matrices.
    """
    state_data = np.load(file_path)
    pyDMD_data = state_data.T  # Transpose to get (features, samples) for PyDMD
    pyDMD_X = pyDMD_data[:5, :]  # State (X)
    pyDMD_Y = pyDMD_data[:5, 1:]   # Next state (Y)
    control_U = pyDMD_data[5:, :-1]  # Control input (U)
    print("Data shapes:", pyDMD_X.shape, pyDMD_Y.shape, control_U.shape)
    return pyDMD_data, pyDMD_X, pyDMD_Y, control_U

# Function to plot the original vs reconstructed states
def plot_comparison(original_data, reconstructed_data, index=1, title="DMD Approximation", xlabel="Time Step", ylabel="Concentration"):
    """
    Plot comparison of original vs reconstructed data for a given index.
    """
    plt.plot(original_data[index], label='Original Data')
    #plt.plot(reconstructed_data[index], '--', label='Reconstructed Data')
    plt.legend()
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.show()

pyDMD_data, pyDMD_X, pyDMD_Y, control_U = load_data(file_path='koopman_state_data_practical_5s_33switch.npy')
#pyDMD_data_n, pyDMD_X_n, pyDMD_Y_n, control_U_n = load_data(file_path='koopman_state_data_practical_5s_33switch.npy')


# Use the first 4 time steps of the test data (shape: (n_features, 4))
X0 = pyDMD_X[:, 0].T  # Shape: (4, n_features)
#X0_n = pyDMD_X_n[:, 0].T  # Shape: (4, n_features)

# Skip the first 3 control steps to align with delayed input
U = control_U.T    # Shape: (n_steps - 3, control_dim)
#U_n = control_U_n.T    # Shape: (n_steps - 3, control_dim)

# Predict future states
X_pred = model.simulate(X0, n_steps=X0.shape[0]).T
#X_pred_n = model.simulate(X0_n, n_steps=X0_n.shape[0]).T

plot_comparison(pyDMD_X, X_pred.T, index=1, title="DMDc Approximation of Signal (S)  train  data")
#plot_comparison(pyDMD_X_n, X_pred_n.T, index=1, title="DMDc Approximation of Signal (S) test random practical")

