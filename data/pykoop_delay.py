import pykoop
import numpy as np
from sklearn.preprocessing import MaxAbsScaler, StandardScaler
import matplotlib.pyplot as plt

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


# data
pyDMD_data, pyDMD_X, pyDMD_Y, control_U = load_data(file_path='koopman_state_data_practical_5s_33switch.npy',
                                                    normalize=False)

pyDMD_data_n, pyDMD_X_n, pyDMD_Y_n, control_U_n = load_data(file_path='koopman_state_data_practical_5s_33switch.npy',
                                                            normalize=False)
# # Print shapes
# print("Training data shapes:")
# print("  pyDMD_data:  ", pyDMD_data.shape)
# print("  pyDMD_X:     ", pyDMD_X.shape)
# print("  pyDMD_Y:     ", pyDMD_Y.shape)
# print("  control_U:   ", control_U.shape)
#
# print("\nTest data shapes:")
# print("  pyDMD_data_n:", pyDMD_data_n.shape)
# print("  pyDMD_X_n:   ", pyDMD_X_n.shape)
# print("  pyDMD_Y_n:   ", pyDMD_Y_n.shape)
# print("  control_U_n: ", control_U_n.shape)

kp = pykoop.KoopmanPipeline(
    lifting_functions=[
        ('ma', pykoop.SkLearnLiftingFn(MaxAbsScaler())),
       ('delay', pykoop.DelayLiftingFn(n_delays_state=1, n_delays_input=0)),
        ('ss', pykoop.SkLearnLiftingFn(StandardScaler())),
    ],
    regressor=pykoop.Dmdc(),
)
kp.fit(
    pyDMD_data.T,
    n_inputs=1,
    episode_feature=False,
)


print(f'kp.fit finished {0}')

# Predict using the pipeline
X_pred = kp.predict_trajectory(X0_or_X=pyDMD_data[:5, 0:2].T,
                               U=control_U[:, :].T,
                               episode_feature=False)


# print(f'Xpred 4 sum: {X_pred[:, :4].T.shape, pyDMD_X[:, ].shape}')
plot_comparison(pyDMD_X[:, :], X_pred[:, :].T, index=1, title="pykoop Approximation of Signal (S)")


# Score using the pipeline
# score = kp.score(pyDMD_data)
