"""
Ts: Change this variable to increase or decrease the amount of time
kab: Rate constant for the reaction
"""

import numpy as np
import os
from scipy.signal import fftconvolve
from scipy.constants import Avogadro
import matplotlib.pyplot as plt
from config import p, kab, r_tx, r_rx, vol_in, vol_out, l, D_space, step_time, simulation_end
from config import save_data


Ts = simulation_end  # Time in seconds
t = np.arange(0, Ts, step_time)  # t = 0:T:500-T;

# Calculation of Volumn and Surface Area of Tx
A = 4 * np.pi * r_tx**2

# Initial Concentrations
CinA0 = 0
CinB0 = 0
CoutB0 = 0

NAout = 1e16  # Number of A molecules outside
CoutA0 = NAout/Avogadro/vol_out  # Concentration of A outside

# permeability and reaction rates
rho0 = p*A  # connect to area of the NP surface

# Initialization of solution vectors
CinA = np.zeros(len(t))
CinA[0] = CinA0

CinB = np.zeros(len(t))
CinB[0] = CinB0

CoutB = np.zeros(len(t))
CoutB[0] = CoutB0

# time dependent permeability
rho = np.zeros(len(t))

# Switching Permeabilty | Use Ts = 10 for base case
step = Ts // 5  # Get the number of time blocks needed
for i in range(Ts):
    if i % 2 != 0:
        rho[int(len(t) * (i-1)/step):int(len(t) * i/step)] = rho0

# Eigen value calculation
lAin = -(rho / vol_in + kab)
lBin = -rho / vol_in
lBout = rho / vol_out


for k in range(1, len(t)):
    CinA[k] = np.exp(lAin[k] * step_time) * CinA[k-1] + step_time * rho[k] / vol_in * CoutA0
    CinB[k] = np.exp(lBin[k] * step_time) * CinB[k-1] + step_time * kab * CinA[k]
    CoutB[k] = CoutB[k-1] + lBout[k] * step_time * CinB[k]

# convert to molecules
NinA = CinA * vol_in * Avogadro
NinB = CinB * vol_in * Avogadro
NoutB = CoutB * vol_out * Avogadro

# Received molecules
b1 = ((r_tx + r_rx) * (r_tx + r_rx - 2 * l) + l**2) / (4 * D_space)
b2 = ((r_tx - r_rx) * (r_tx - r_rx + 2 * l) + l**2) / (4 * D_space)
ro = 1 / A
pt = (2 * ro * r_tx * r_rx / l) * np.sqrt(np.pi * D_space / t) * (np.exp(-b1 / t) - np.exp(-b2 / t))
pt = np.nan_to_num(pt)

nr = fftconvolve(CoutB * vol_out * Avogadro, pt, mode='full')
Nrec = nr[:len(t)] * step_time


if save_data:
    # Prepare data for saving
    data = np.column_stack((t, rho, NinA, NinB, NoutB, Nrec))

    # Set up directory and dynamic file naming
    output_folder = "output/idealTx"
    file_name = f"{Ts}s_kab_{kab:.0e}.csv" # Create file name based on Kab and Time in Seconds
    file_path = os.path.join(output_folder, file_name)
    os.makedirs(output_folder, exist_ok=True)  # Create folder if it doesn't exist

    # Save to CSV with headers
    header = "Time,Rho,NinA,NinB,NoutB,Nrec"
    np.savetxt(file_path, data, delimiter=",", header=header)
    print(f"Data saved to: {file_path}")

# Plot Number of Molecues of Type A and B inside the Transmitter(Tx) and Number of Type B Molecules outside
plt.figure(1)
plt.grid(True)
plt.plot(t, NinA)
plt.plot(t, NinB)
plt.plot(t, NoutB)
plt.plot(t, Nrec)
plt.xlabel('Time')
plt.ylabel('# Molecules')
plt.legend()
plt.show()
