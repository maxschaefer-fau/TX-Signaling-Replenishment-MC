import numpy as np
import pysindy as ps

# Example
X = np.load("koopman_state_data_practical_8s_15switch.npy")  # shape: (n_samples, 7)
t = np.linspace(0, X.shape[0] * 1e-3, X.shape[0])  # Assuming 1 ms step time

# Strip out only concentrations (ignore episode ID and perm state)
X_conc = X[:, :5]  # R, S, MR, ER, ES
u = X[:, 5:]  # control input, shape (n_samples, 1)
dt = 1e-3  # or your actual timestep

# Choose a differentiation method
model = ps.SINDy(
    optimizer=ps.STLSQ(threshold=3e-3),
    #differentiation_method=ps.FiniteDifference(is_uniform=True),
    differentiation_method=ps.SmoothedFiniteDifference(),
    feature_library=ps.PolynomialLibrary(degree=1, include_interaction=True, include_bias=True)
)

model.fit(X_conc,u=u, t=t, ensemble=True)
model.print()
