import os
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt

from models.space import Space, Particle


def generate_switching_pattern(switching_pattern, time_interval, length, config):
    '''
    Example: switching_pattern -> [1,0,0,1,0], time_interval -> 3s

    each switching pattern for 3s
    '''
    rho_array = np.zeros(length)
    segment_length = length // len(switching_pattern) 

    for i, state in enumerate(switching_pattern):
        start_index = i * segment_length
        end_index = start_index + segment_length
        
        # Assign the state value to the corresponding segment in rho_array
        rho_array[start_index:end_index] = state * config.p

    return rho_array

def generate_dyn_switching_pattern(switching_pattern, time_interval, length, config, peak_duration_ratio=0.2):
    '''
    Generate a permeability pattern that rises to a max, stays, then falls.

    - switching_pattern: list of binary values [1,0,0,1,0]
    - time_interval: Duration of each switching pattern (e.g., 3 seconds)
    - length: Total length of the resulting rho_array
    - config: Configuration object with maximum permeability 'p'
    - peak_duration_ratio: Fraction of time_interval to stay at max before decreasing

    Example Usage:
        - switching_pattern = [1, 0, 0, 1, 0]
    - time_interval = 3
    - length = 15 (must be divisible by len(switching_pattern))
    - config.p = max permeability value
    '''

    rho_array = np.zeros(length)
    segment_length = length // len(switching_pattern) 

    # Determine how long to go up, stay at peak, and go down based on peak_duration_ratio
    peak_duration = int(segment_length * peak_duration_ratio)
    rise_duration = (segment_length - peak_duration) // 2
    fall_duration = segment_length - rise_duration - peak_duration

    for i, state in enumerate(switching_pattern):
        start_index = i * segment_length
        end_index = start_index + segment_length

        if state == 1:
            # Rising segment
            for j in range(rise_duration):
                rho_array[start_index + j] = (j / rise_duration) * config.p

            # Peak segment
            for j in range(peak_duration):
                rho_array[start_index + rise_duration + j] = config.p

            # Falling segment
            for j in range(fall_duration):
                rho_array[start_index + rise_duration + peak_duration + j] = config.p * (1 - j / fall_duration)

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
        file_name = f"MR_{kwargs['conc_in'].particles['MR'].count:.2f}_Ts_{config.simulation_end}"
        file_path = os.path.join(output_folder_with_ts, file_name)
    elif exp_type == 'point':
        file_name = f'N_{config.N}_Ts_{config.simulation_end}'
        file_path = os.path.join(output_folder_with_ts, file_name)

    # Save to CSV with headers
    header = ','.join(data_dict.keys())
    np.savetxt(file_path, data, delimiter=",", header=header)

def plot_data(time_array, rho, data_ideal, data_practical, data_pointTx, switching_pattern, config):
    """
    Plot permeability and molecule counts over time.

    Parameters:
    - time_array: Array representing time intervals.
    - permeability_array: Array representing permeability values.
    - NinA, NinB, NoutB, Nrec: Arrays representing molecule counts.
    - switching_pattern: The switching pattern used in the simulation.
    - config: Configuration object with simulation details.
    """
    # Set up directory and dynamic file naming
    output_folder_with_ts = os.path.join(config.output_folder, datetime.now().strftime("%Y%m%d_%H%M"))
    os.makedirs(output_folder_with_ts, exist_ok=True)  # Create folder if it doesn't exist

    # Plot 1
    NinA, NinB = data_ideal['NinA'], data_ideal['NinB']
    NinR, NinS = data_practical['NinR'], data_practical['NinS']

    # Top plot for molecule counts
    fig1 = plt.figure(figsize=(10, 6))
    ax2 = plt.subplot2grid((5, 1), (0, 0), rowspan=4)
    ax2.plot(time_array, NinA, 'r', label='NinA')
    ax2.plot(time_array, NinB, 'r--', label='NinB')
    ax2.plot(time_array, NinR, 'b', label='NinR')
    ax2.plot(time_array, NinS, 'b--', label='NinS')
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    # Set x-axis and y-axis limits to 0 to remove margin at start
    ax2.set_ylim(0, max(NinA.max(), NinB.max(), NinR.max(), NinS.max()) + 10)  # Or set a specific max value
    ax2.set_ylabel('# Molecules')
    ax2.grid(True)
    ax2.legend(loc='upper right')

    # Bottom plot for permeability
    ax1 = plt.subplot2grid((5, 1), (4, 0), rowspan=1, sharex=ax2)
    ax1.plot(time_array, rho, 'k', label='Permeability')
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    # Set x-axis and y-axis limits to 0 to remove margin at start
    ax1.set_xlim(0, time_array[-1] + 0.1)  # Adjust based on your data range
    ax1.set_ylim(0, rho.max())  # Or set a specific max value
    ax1.set_xlabel('Time (s)')
    ax1.set_ylabel('ρ(t)')
    ax1.legend(loc='upper right')

    plt.tight_layout()

    # Save Plot 1 as an image
    figure1_filepath = os.path.join(output_folder_with_ts, 'molecule_counts_and_permeability.png')
    fig1.savefig(figure1_filepath)

    plt.show()

    # Plot 2
    NoutB, Brec = data_ideal['NoutB'], data_ideal['Nrec']
    NoutS, Srec = data_practical['NoutS'], data_practical['Nrec']
    NoutB_point, Brec_point = data_pointTx['NoutB'], data_pointTx['Nrec']

    fig2 = plt.figure(figsize=(10, 6))

    ax3 = plt.subplot2grid((5, 1), (0, 0), rowspan=4)
    ax3.plot(time_array, NoutB, 'r', label='NoutB')
    ax3.plot(time_array, Brec, 'r--', label='Brec')
    ax3.plot(time_array, NoutS, 'k', label='NoutS')
    ax3.plot(time_array, Srec, 'k--', label='Srec')
    ax3.plot(time_array, NoutB_point, 'b', label='NoutB Point')
    ax3.plot(time_array, Brec_point, 'b--', label='Brec Point')
    ax3.spines['top'].set_visible(False)
    ax3.spines['right'].set_visible(False)
    ax3.set_ylim(0, max(NoutB.max(), Brec.max(), NoutS.max(), Srec.max()) + 10)
    ax3.set_xlim(0, time_array[-1] + 0.1)
    ax3.set_xlabel('Time (s)')
    ax3.set_ylabel('# Molecules / Reception')
    ax3.grid(True)
    ax3.legend(loc='upper right')

    # Bottom plot for permeability
    ax4 = plt.subplot2grid((5, 1), (4, 0), rowspan=1, sharex=ax3)
    ax4.plot(time_array, rho, 'k', label='Permeability')
    ax4.spines['top'].set_visible(False)
    ax4.spines['right'].set_visible(False)
    # Set x-axis and y-axis limits to 0 to remove margin at start
    ax4.set_xlim(0, time_array[-1] + 0.1)  # Adjust based on your data range
    ax4.set_ylim(0, rho.max())  # Or set a specific max value
    ax4.set_xlabel('Time (s)')
    ax4.set_ylabel('ρ(t)')
    ax4.legend(loc='upper right')
    plt.tight_layout()

    # Save Plot 2 as an image
    figure2_filepath = os.path.join(output_folder_with_ts, 'outside_molecules_and_receptions.png')
    fig2.savefig(figure2_filepath)

    plt.show()


def plot_pointTx(time_array, data_pointTx, config):

    NoutB, Nrec = data_pointTx['NoutB'], data_pointTx['Nrec']

    # Set up directory and dynamic file naming
    output_folder_with_ts = os.path.join(config.output_folder, datetime.now().strftime("%Y%m%d_%H%M"))
    os.makedirs(output_folder_with_ts, exist_ok=True)  # Create folder if it doesn't exist

    fig, ax1 = plt.subplots()

    # Plot number of B molecules released
    ax1.plot(time_array, NoutB, 'r', label='NoutB (Released)')
    ax1.set_xlabel('Time')
    ax1.set_ylabel('# B Molecules Released', color='r')
    ax1.tick_params(axis='y', labelcolor='r')
    ax1.set_ylim(0, max(NoutB) )  # Adjust for better viewing
    
    # Instantiate a second y-axis that shares the same x-axis
    ax2 = ax1.twinx()
    ax2.plot(time_array, Nrec, 'k', label='Nrec (Received)')
    ax2.set_ylabel('# B Molecules Received', color='k')
    ax2.tick_params(axis='y', labelcolor='k')
    ax2.set_ylim(0, max(NoutB) )  # Adjust for better viewing

    plt.title('Number of B Molecules Released and Received Over Time')
    plt.tight_layout()  # To prevent label overlap

    # Save Plot 2 as an image
    figure_filepath = os.path.join(output_folder_with_ts, 'point_transmitter.png')
    fig.savefig(figure_filepath)

    plt.show()


def plot_hitting_prob(time_array, hitting_prob):
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
    plt.plot(time_array, hitting_prob, label='Hitting Probability')
    plt.xlabel('Time (s)')
    plt.ylabel('# Prob')
    plt.title(f"Hitting Prob")
    plt.legend()
    plt.show()


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
