import numpy as np
import matplotlib.pyplot as plt
from scipy.constants import Avogadro 
from config import NanomachineConfig
from utils import generate_switching_pattern, get_conc_vol_for_practical, save_to_csv, plot_data
from models.space import AbsorbingReceiver, TransparentReceiver


# Set Config
conf = NanomachineConfig()
vol_in, vol_out, conc_in, conc_out = get_conc_vol_for_practical(conf.r_tx, conf.r_out)

# Set Switching pattern
switching_pattern = [1,0,1,0,1,0,1,0,0,0,0]
Ts = 2
conf.simulation_end = len(switching_pattern) * Ts

conf.dist *= 2

# Time Array
time_array = np.linspace(0,
                         conf.simulation_end,
                         int(conf.simulation_end / conf.step_time),
                         endpoint=False)

def point_transmitter(time_array, switching_pattern, N, config):
    '''
    Simulate the release of N molecules of type B at every Ts interval over the total time_array duration.

    Parameters:
    - time_array: The array of time points for the simulation.
    - Ts: Time interval for releasing molecules.
    - N: Number of molecules released at each interval.
    - config: Configuration containing receiver and simulation parameters.

    Returns:
    - NoutB: Number of B molecules released over time.
    - Nrec: Number of B molecules received over time after convolution with receiver response.
    '''
    
    # Initialize the release pattern
    release_pattern = np.zeros_like(time_array)
    
    segment_length = len(time_array) // len(switching_pattern) 

    release_indices = []
    for i, state in enumerate(switching_pattern):
        start_index = i * segment_length
        if state == 1:
            release_indices.append(start_index)

    release_pattern[release_indices] = N
    
    # Create receiver and perform convolution to simulate reception of released molecules
    rec = AbsorbingReceiver(config.r_rx) if config.receiver_type == 'AbsorbingReceiver' else TransparentReceiver(config.r_rx)
    
    # Simulate average hits at the receiver
    # TODO change hitting probability https://ieeexplore.ieee.org/document/8742793
    nr = rec.average_hits(time_array, release_pattern, config.r_tx, config.D_space, config.dist)
    Nrec = nr[:len(time_array)] * config.step_time

    # Convert release pattern to number of molecules outside
    NoutB = release_pattern * config.D_space * Avogadro * config.r_tx * config.dist



    return {
            'NoutB': NoutB,
            'Nrec': Nrec
            }

# Assuming time_array, Ts, and N are defined along with a Configuration object, config
results = point_transmitter(time_array, switching_pattern, 100, conf)
NoutB = results['NoutB']
Nrec = results['Nrec']

fig, ax1 = plt.subplots()

# Plot number of B molecules released
ax1.plot(time_array, NoutB, 'r', label='NoutB (Released)')
ax1.set_xlabel('Time')
ax1.set_ylabel('# B Molecules Released', color='r')
ax1.tick_params(axis='y', labelcolor='r')
ax1.set_ylim(0, max(Nrec) )  # Adjust for better viewing

# Instantiate a second y-axis that shares the same x-axis
ax2 = ax1.twinx()
ax2.plot(time_array, Nrec, 'k', label='Nrec (Received)')
ax2.set_ylabel('# B Molecules Received', color='k')
ax2.tick_params(axis='y', labelcolor='k')
ax2.set_ylim(0, max(Nrec))  # Adjust for better viewing

plt.title('Number of B Molecules Released and Received Over Time')
plt.tight_layout()  # To prevent label overlap
plt.show()
