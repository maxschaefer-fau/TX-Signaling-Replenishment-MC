import os
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt

from models.space import Space, Particle


def generate_switching_pattern(switching_pattern, time_interval, length, config):
    '''
    Example: switching_pattern -> [1,0,0,1,1], time_interval -> 2s

    each switching pattern for 2s
    '''
    rho_array = np.zeros(length)
    segment_length = length // len(switching_pattern) 

    for i, state in enumerate(switching_pattern):
        start_index = i * segment_length
        end_index = start_index + segment_length
        
        # Assign the state value to the corresponding segment in rho_array
        rho_array[start_index:end_index] = state * config.p

    return rho_array


def save_to_csv(data_dict, exp_type, config, **kwargs):
    '''
    data: list of columns to save
    file_name: name of the output file
    output_folder: folder to save the output file
    '''

    # Prepare data for saving
    data = np.column_stack(list(data_dict.values()))

    # Set up directory and dynamic file naming
    output_folder_with_ts = os.path.join(config.output_folder, datetime.now().strftime("%Y%m%d_%H%M"))
    os.makedirs(output_folder_with_ts, exist_ok=True)  # Create folder if it doesn't exist

    file_path = ''
    if exp_type == 'ideal':
        file_name = f'kab_{config.kab}_Ts_{config.simulation_end}'
        file_path = os.path.join(output_folder_with_ts, file_name)
    elif exp_type == 'practical':
        file_name = f'MR_{kwargs['conc_in'].particles['MR'].count:.2f}_Ts_{config.simulation_end}'
        file_path = os.path.join(output_folder_with_ts, file_name)

    # Save to CSV with headers
    header = ','.join(data_dict.keys())
    np.savetxt(file_path, data, delimiter=",", header=header)


def plot_permeability(time_array, rho, switching_pattern, Ts):
    """
    Function to plot permeability over time.

    Parameters:
    - time_array: Array of time values.
    - rho: Array of permeability values.
    - switching_pattern: The switching pattern used in the simulation.
    - Ts: Time interval for the switching pattern.
    """
    plt.figure()
    plt.grid(True)
    plt.plot(time_array, rho, label='Permeability')
    plt.xlabel('Time (s)')
    plt.ylabel('# P')
    plt.title(f"Switching Pattern: {switching_pattern} with Ts: {Ts}")
    plt.legend()
    plt.show()

def get_conc_vol_for_practical(r_tx, r_out):
    # Molecule counts on the inside
    vol_in = (4*np.pi*r_tx*r_tx*r_tx)/3
    conc_in = Space({
        'R': Particle(0.0, True, volume=vol_in),
        'S': Particle(0.0, True, volume=vol_in),
        'MR': Particle(2.0, False, volume=vol_in),
        'ER': Particle(0.0, False, volume=vol_in),
        'ES': Particle(0.0, False, volume=vol_in)
    }, area=4*np.pi*r_tx*r_tx, volume=vol_in)
    
    # Molecule counts on the outside
    # We are considering Env a sphear including vol_in
    vol_out = (4*np.pi*r_out*r_out*r_out)/3
    conc_out = Space({
        'R': Particle(1e16, True, volume=vol_out),
        'S': Particle(0.0, True, volume=vol_out),
        'MR': Particle(0.0, False, volume=vol_out),
        'ER': Particle(0.0, False, volume=vol_out),
        'ES': Particle(0.0, False, volume=vol_out)
    }, area=4*np.pi*r_tx*r_tx, volume=vol_out)
    
    return vol_in, vol_out, conc_in, conc_out
