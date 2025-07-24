import numpy as np
import lightning as L
from pydmd import DMD, DMDc, BOPDMD
import matplotlib.pyplot as plt
import joblib
import os
from sklearn.preprocessing import MaxAbsScaler, StandardScaler
#from pydmd.preprocessing import hankel_preprocessing

from pykoopman import Koopman
import pykoopman as pk
from pykoopman import observables
from pykoopman.regression import DMDc as PKDMDc

os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

class IncrementalSequenceLoss(L.Callback):
    def on_train_epoch_end(self, trainer, pl_module):
        max_look_forward = pl_module.look_forward
        if trainer.callback_metrics["loss"] < 1e-2 and \
                pl_module.masked_loss_metric.max_look_forward < max_look_forward:
            print("increase width size from {} to {}".format(
                pl_module.masked_loss_metric.max_look_forward,
                pl_module.masked_loss_metric.max_look_forward+1) )
            print("")
            pl_module.masked_loss_metric.max_look_forward += 1



# Load and prepare the data
def load_data(file_path='koopman_state_data_ideal_7s.npy', control=False, normalize=False):
    """
    Load the state data and prepare the DMD input matrices.
    """
    state_data = np.load(file_path)
    pyDMD_data = state_data.T  # Transpose to get (features, samples) for PyDMD
    pyDMD_X = pyDMD_data[:5, :]  # State (X)
    pyDMD_Y = pyDMD_data[:5, 1:]   # Next state (Y)
    control_U = pyDMD_data[5:, :-1]  # Control input (U)
    if normalize:
        # Apply standard scaling independently
        scaler_x = StandardScaler()
        scaler_u = StandardScaler()
        X_scaled = scaler_x.fit_transform(pyDMD_X)
        U_scaled = scaler_u.fit_transform(control_U)

        return pyDMD_data, X_scaled, pyDMD_Y, U_scaled
    print("Data shapes:", pyDMD_X.shape, pyDMD_Y.shape, control_U.shape)
    return pyDMD_data, pyDMD_X, pyDMD_Y, control_U

def fit_pykoopman_nndmd(X, Y):
    n_features = X.shape[0]
    look_forward = 1
    print("X min/max:", X.min(), X.max())
    print("Y min/max:", Y.min(), Y.max())
    dlk_regressor = pk.regression.NNDMD(look_forward=look_forward,
                                        config_encoder=dict(input_size=n_features,
                                                            hidden_sizes=[16] * 2,
                                                            output_size=n_features,
                                                            activations="linear"),
                                        config_decoder=dict(input_size=n_features,
                                                            hidden_sizes=[16] * 2,
                                                            output_size=n_features,
                                                            activations="linear"),
                                        batch_size=256, lbfgs=False, \
                                                normalize=True, normalize_mode='max',
                                        trainer_kwargs=dict(max_epochs=1,
                                                            accelerator="cpu",
                                                            devices=1,
                                                            ))
                                                            #callbacks=[
                                                            #    IncrementalSequenceLoss()]))

    model = Koopman(regressor=dlk_regressor)
    model.fit(X.T, Y.T)

    return model


def fit_pykoopman_edmdc(X, U, n_delays=3):
    """
    Fit an EDMDc model using time-delay observables.
    Args:
        X: shape (n_features, n_samples)
        U: shape (n_control_inputs, n_samples)
        n_delays: number of delays for the observable transformation
    Returns:
        Fitted Koopman model with EDMDc.
    """
    # Transpose data
    X = X.T  # (n_samples, n_features)
    U = U.T  # (n_samples, n_inputs)

    # Create time-delay observables
    obs = observables.Polynomial(degree=1)

    # Instantiate DMDc regression model
    regressor = PKDMDc()

    # Create Koopman model with time-delay observables
    model = Koopman(observables=obs, regressor=regressor)
    #U = U[n_delays:, :]

    print(f"x {X.shape}, u {U.shape}")

    # Truncate data to match delayed observables
    model.fit(X, u=U)

    return model


def fit_pykoopman_dmdc(X, U, svd_rank=50):
    """
    Fit DMDc using PyKoopman and return the trained model.
    X: shape (n_features, n_samples)
    U: shape (n_control_inputs, n_samples)
    """
    X_train = X.T     # shape (n_samples, n_features)
    U_train = U.T     # shape (n_samples, n_inputs)


    regressor = PKDMDc(svd_rank=svd_rank, alpha=0.1)
    model = Koopman(regressor=regressor)
    model.fit(X_train, u=U_train)

    return model


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
    dmdc = DMDc(svd_rank=svd_rank, lag=1)
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
    pyDMD_data, pyDMD_X, pyDMD_Y, control_U = load_data(file_path='koopman_state_data_practical_8s_15switch.npy',
                                                        normalize=True)

    # Fit models and get reconstructed data
    #print("Fitting DMD...")
    #dmd, X_dmd = fit_dmd(pyDMD_data)
    #print("Fitting DMDc...")
    #dmdc, X_dmdc = fit_dmdc(pyDMD_X, control_U)
    #A_dmdc, B_dmdc = dmdc.sys_dynamic_matrices()
    print("Fitting PyKoopman DMDc...")

    #model = fit_pykoopman_edmdc(pyDMD_X, control_U)
    model = fit_pykoopman_dmdc(pyDMD_X, pyDMD_Y)
    eigvals = np.linalg.eigvals(model.A)
    print("Max abs eigenvalue of A:", np.max(np.abs(eigvals)))
    print("B matrix max abs value:", np.max(np.abs(model.B)))

    joblib.dump(model, 'trained_dmdc_model.pkl')

    # Transpose pyDMD_X to shape (n_samples, n_features)
    #X_for_sim = pyDMD_X.T[:3]  # Use 10 steps = n_delays + 1
    #U_for_sim = control_U.T[:control_U.shape[1] - 3]  # Make U match in length

    #X_dmdc = model.simulate(X_for_sim, u=U_for_sim, n_steps=U_for_sim.shape[0])

    #X_dmdc = model.simulate(pyDMD_X[:, 0].T, u=control_U.T, n_steps=control_U.shape[1]).T
    #print(f"A {model.A.shape}, B {model.B.shape}, x0 {pyDMD_X_n[:, 0:1].shape}, control_U_n {control_U_n.T.shape}")
    #n_delays = 2
    #X0_n_delays = pyDMD_X_n.T[:n_delays + 1]  # Shape: (10, 5)
    #U_sim_n = control_U_n.T[:control_U_n.shape[1] - n_delays - 1]
    #X_dmdc_n = model.simulate(X0_n_delays, u=U_sim_n, n_steps=U_sim_n.shape[0]).T
    #X_dmdc_n = simulate_with_new_control(model.A, model.B,  pyDMD_X_n[:, 0:1], control_U_n.T)
    #X_dmdc_n = model.simulate(pyDMD_X_n[:, 0].T, u=control_U_n.T, n_steps=control_U_n.shape[1]).T
    #print(f"x dmdc shape: {X_dmdc.shape}, new shape {X_dmdc_n.shape}")

    #X_dmdc_n = simulate_with_dmdc(A_dmdc, B_dmdc, pyDMD_X_n[:5, 0], control_U_n)

    #print("Fitting BOPDMD...")
    #bopdmd, X_bopdmd = fit_bopdmd(pyDMD_data, t)

    '''
    EDMDC pykoopman
    '''



    # Plot comparisons of original vs reconstructed data for each method
    #plot_comparison(pyDMD_X, X_dmd, index=1, title="DMD Approximation of Signal (S)")
    #plot_comparison(pyDMD_X, X_dmdc, index=1, title="DMDc Approximation of Signal (S)  fixed+random  practical")
    #plot_comparison(pyDMD_X, X_bopdmd, index=1, title="BOPDMD Approximation of Signal (S)")


# Run the main function
if __name__ == "__main__":
    main()

