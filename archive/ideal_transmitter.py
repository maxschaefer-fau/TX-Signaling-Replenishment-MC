"""
Ts: Change this variable to increase or decrease the amount of time
kab: Rate constant for the reaction
"""

import numpy as np
import os
from scipy.constants import Avogadro
import matplotlib.pyplot as plt
from config import p, kab, r_tx, r_rx, vol_in, vol_out, dist, D_space, step_time, simulation_end
from config import save_data, state_path, end_part, time_array, p_close, release_count, receiver_type
from create_state_values import create_state_array
from models.space import AbsorbingReceiver, TransparentReceiver

# Calculation of Volumn and Surface Area of Tx
A = 4 * np.pi * r_tx**2

# Initial Concentrations
CinA0 = 0
CinB0 = 0
CoutB0 = 0
NAout = 1e16  # Number of A molecules outside
CoutA0 = NAout/Avogadro/vol_out  # Concentration of A outside

# Perm Array
try:
    rho = np.genfromtxt(state_path, delimiter=',')
    rho *= A
except FileNotFoundError:
    rho = create_state_array(state_path, p, p_close, release_count, end_part, time_array)

# Initialization of solution vectors
CinA = np.zeros(len(time_array))
CinA[0] = CinA0

CinB = np.zeros(len(time_array))
CinB[0] = CinB0

CoutB = np.zeros(len(time_array))
CoutB[0] = CoutB0


# Eigen value calculation
lAin = -(rho / vol_in + kab)
lBin = -rho / vol_in
lBout = rho / vol_out

for k in range(1, len(time_array)):
    CinA[k] = np.exp(lAin[k] * step_time) * CinA[k-1] + step_time * rho[k]  / vol_in * CoutA0
    CinB[k] = np.exp(lBin[k] * step_time) * CinB[k-1] + step_time * kab * CinA[k]
    CoutB[k] = CoutB[k-1] + lBout[k] * step_time * CinB[k]

# convert to molecules
NinA = CinA * vol_in * Avogadro
NinB = CinB * vol_in * Avogadro
NoutB = CoutB * vol_out * Avogadro

# Received molecules
rec = AbsorbingReceiver(r_rx)
if receiver_type == 'AbsorbingReceiver':
    pass
elif receiver_type == 'TransparentReceiver':
    rec = TransparentReceiver(r_rx)

nr = rec.average_hits(time_array, NoutB, r_tx, D_space, dist)
Nrec = nr[:len(time_array)] * step_time

if save_data:
    # Prepare data for saving
    data = np.column_stack((time_array, rho, NinA, NinB, NoutB, Nrec))

    # Set up directory and dynamic file naming
    output_folder = "output/idealTx"
    file_name = f"{simulation_end}s_kab_{kab:.0e}.csv" # Create file name based on Kab and Time in Seconds
    file_path = os.path.join(output_folder, file_name)
    os.makedirs(output_folder, exist_ok=True)  # Create folder if it doesn't exist

    # Save to CSV with headers
    header = "Time,Rho,NinA,NinB,NoutB,Nrec"
    np.savetxt(file_path, data, delimiter=",", header=header)
    print(f"Data saved to: {file_path}")

# Plot Number of Molecues of Type A and B inside the Transmitter(Tx) and Number of Type B Molecules outside
plt.figure(1)
plt.grid(True)
plt.plot(time_array, NinA, label='NinA')
plt.plot(time_array, NinB, label='NinB')
plt.plot(time_array, NoutB, label='NoutB')
plt.plot(time_array, Nrec, label='Nrec')
plt.xlabel('Time')
plt.ylabel('# Molecules')
plt.legend()
plt.show()
