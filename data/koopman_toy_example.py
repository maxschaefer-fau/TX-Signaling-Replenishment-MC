import numpy as np
from pydmd import DMD, DMDc, BOPDMD
from pydmd.plotter import plot_eigs, plot_summary
import matplotlib.pyplot as plt
from pydmd.preprocessing import hankel_preprocessing


# Load and prepare the data
def load_data(file_path='koopman_state_data_ideal_7s.npy', control=False):
    """
    Load the state data and prepare the DMD input matrices.
    """
    state_data = np.load(file_path)
    pyDMD_data = state_data.T  # Transpose to get (features, samples) for PyDMD
    pyDMD_X = pyDMD_data[:5, :-1]  # State (X)
    pyDMD_Y = pyDMD_data[:5, 1:]   # Next state (Y)
    control_U = pyDMD_data[5:, :-1]  # Control input (U)
    print("Data shapes:", pyDMD_X.shape, pyDMD_Y.shape, control_U.shape)
    return pyDMD_data, pyDMD_X, pyDMD_Y, control_U


# Function to fit and reconstruct data using standard DMD
def fit_dmd(X, svd_rank=6):
    """
    Fit and reconstruct data using standard DMD.
    """
    dmd = DMD(svd_rank=svd_rank)
    dmd.fit(X)
    X_dmd = dmd.reconstructed_data.real  # Reconstructed states
    return dmd, X_dmd


# Function to fit and reconstruct data using DMDc (Control Input DMD)
def fit_dmdc(X, control_U, svd_rank=6):
    """
    Fit and reconstruct data using DMDc with control input.
    """
    dmdc = DMDc(svd_rank=svd_rank)
    # Fit DMDc using state data X and control input control_U
    dmdc.fit(X=X, I=control_U)
    print("Eigenvalues of DMDc:", np.abs(dmdc.eigs))
    X_dmdc = dmdc.reconstructed_data(control_U).real  # Reconstructed states
    X_dmdc = np.clip(X_dmdc, -1e2, 1e2)  # Clipping large values for stability
    return dmdc, X_dmdc


# Function to fit and reconstruct data using eDMD (Extended DMD)
def fit_bopdmd(X, t, svd_rank=8):
    """
    Fit and reconstruct data using eDMD (Extended DMD) with time vector.
    """
    bopdmd = BOPDMD(svd_rank=svd_rank)
    bopdmd.fit(X, t)  # Fit with both state data and time vector
    X_bopdmd = bopdmd.reconstructed_data.real
    return bopdmd, X_bopdmd


# Function to plot the original vs reconstructed states
def plot_comparison(original_data, reconstructed_data, index=1, title="DMD Approximation", xlabel="Time Step", ylabel="Concentration"):
    """
    Plot comparison of original vs reconstructed data for a given index.
    """
    plt.plot(original_data[index], label='Original Data')
    plt.plot(reconstructed_data[index], '--', label='Reconstructed Data')
    plt.legend()
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.show()


# Function to visualize state evolution over time
def plot_state_evolution(state_data, labels=None):
    """
    Plot state evolution over time for all variables in the dataset.
    """
    if labels is None:
        labels = ['R', 'S', 'MR', 'ER', 'ES', 'rho']

    for i in range(6):
        plt.plot(state_data[:, i], label=labels[i])

    plt.legend()
    plt.title("State Evolution over Time")
    plt.xlabel("Timestep")
    plt.ylabel("Concentration / Permeability")
    plt.show()


# Main function to run the experiments and visualize results
def main():
    # Load the data
    pyDMD_data, pyDMD_X, pyDMD_Y, control_U = load_data()

    # Define a time vector (for example, 0, 1, 2, ..., N-1)
    t = np.arange(pyDMD_X.shape[1])  # Assuming time steps correspond to columns in X

    # Fit models and get reconstructed data
    print("Fitting DMD...")
    dmd, X_dmd = fit_dmd(pyDMD_data)
    print("Fitting BOPDMD...")
    bopdmd, X_bopdmd = fit_bopdmd(pyDMD_X, t)

    # Plot comparisons of original vs reconstructed data for each method
    plot_comparison(pyDMD_X, X_dmd, index=1, title="DMD Approximation of Signal (S)")
    plot_comparison(pyDMD_X, X_bopdmd, index=1, title="BOPDMD Approximation of Signal (S)")


# Run the main function
if __name__ == "__main__":
    main()
