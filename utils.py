import os
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from models.space import Space, Particle

def generate_random_switching_pattern(length: int = 10, padding: int = 5) -> list[int]:
    """
    Generates a random binary switching pattern for the transmitter.

    The generated pattern contains random bits followed by a specified number of padding bits
    (which are zeros). The random bits represent active (1) or inactive (0) states.

    Parameters:
        length (int): Number of random bits to generate (default is 10).
        padding (int): Number of zeros to append at the end of the pattern (default is 5).

    Returns:
        list[int]: A list containing the generated switching pattern with random bits followed by zeros.

    Example Usage:
        pattern = generate_switching_pattern(length=15, padding=5)
    """
    if length < 0 or padding < 0:
        raise ValueError("Both length and padding must be non-negative integers.")

    random_bits = np.random.randint(0, 2, size=length)  # Generates random bits (0 or 1)
    padding_bits = np.zeros(padding, dtype=int)  # Create an array of zeros for padding
    return np.concatenate((random_bits, padding_bits)).tolist()


def generate_permeability_pattern(
        mode: str,
        switching_pattern: list[int],
        length: int,
        config,
        peak_duration_ratio: float,
        zero_duration_ratio: float):

    """
    Generate a permeability pattern based on the specified mode (ideal/practical).

    Parameters:
        mode (str): Mode of operation ("ideal" or "practical").
        switching_pattern (list[int]): List of binary values indicating the state [1, 0].
        length (int): Total length of the resulting rho_array (must be equal to length of time array).
        config: Configuration object with a maximum permeability attribute 'p'.
        peak_duration_ratio (float): Fraction of the total duration to stay at maximum permeability.
        zero_duration_ratio (float): Fraction of the total duration to stay at zero permeability.

    Returns:
        np.ndarray: Array representing the permeability pattern over time.

    Example Usage:
        rho_array = generate_switching_pattern('practical', [1, 0, 0, 1, 0], 30, config, 0.2, 0.2)
    """

    # Validate input parameters
    if mode not in ['ideal', 'practical']:
        raise ValueError("Mode must be either 'ideal' or 'practical'.")

    rho_array = np.zeros(length)
    segment_length = length // len(switching_pattern) 

    if mode == 'practical':
        # Determine how long to go up, stay at peak, and go down based on peak_duration_ratio
        peak_duration = int(segment_length * peak_duration_ratio)
        zero_duration = int(segment_length * zero_duration_ratio)
        rise_duration = (segment_length - peak_duration - zero_duration) // 2
        fall_duration = segment_length - rise_duration - peak_duration - zero_duration

        def fill_segments(state, start_index):
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

                # Zero duration segment
                for j in range(zero_duration):
                    rho_array[start_index + rise_duration + peak_duration + fall_duration + j] = 0

    for i, state in enumerate(switching_pattern):
        start_index = i * segment_length
        end_index = start_index + segment_length

        if mode == 'ideal':
            # Assign the state value to the corresponding segment in rho_array
            rho_array[start_index:end_index] = state * config.p
        elif mode == 'practical' and state == 1:
            fill_segments(state, start_index)

    return rho_array

def save_to_csv(data_dict: dict, exp_type: str, config, file_extension: str = '.csv', **kwargs) -> None:
    """
    Save simulation results to a CSV file in a designated output folder.

    Parameters:
        data_dict (dict): A dictionary where keys are column headers and values are lists/arrays of data.
        exp_type (str): Type of experiment ('ideal', 'practical', or 'point').
        config (Config): Configuration object containing parameters used for file naming and output folder.
        file_extension (str): The file extension for the saved file (default is '.csv').
        kwargs: Additional keyword arguments to specify parameters related to saving.
            - for 'practical': cons_in 

    Raises:
        ValueError: If required keys are not found in kwargs.
    """

    # Prepare data for saving
    data = np.column_stack(list(data_dict.values()))

    # Set up directory and dynamic file naming
    output_folder_with_ts = Path(config.output_folder) / datetime.now().strftime("%Y%m%d_%H%M")
    output_folder_with_ts.mkdir(parents=True, exist_ok=True)  # Create folder if it doesn't exist

    # Determine the file name based on experiment type
    if exp_type == 'ideal':
        file_name = f'kab_{config.kab}_Ts_{config.simulation_end}{file_extension}'
    elif exp_type == 'practical':
        if 'conc_in' not in kwargs or 'MR' not in kwargs['conc_in'].particles:
            raise ValueError("Missing 'conc_in' or 'MR' in kwargs for 'practical' experiment.")
        file_name = f"MR_{kwargs['conc_in'].particles['MR'].count:.2f}_Ts_{config.simulation_end}{file_extension}"
    elif exp_type == 'point':
        file_name = f'N_{config.N}_Ts_{config.simulation_end}{file_extension}'
    else:
        raise ValueError(f"Unknown experiment type: {exp_type}")

    # Save to CSV with headers
    header = ','.join(data_dict.keys())
    file_path = output_folder_with_ts / file_name

    # Save the data to a CSV file
    np.savetxt(file_path, data, delimiter=",", header=header, comments='')


def plot_data(time_array, rho, data_ideal, data_practical, data_pointTx, switching_pattern, config):
    """
    Plot permeability and molecule counts over time.

    Parameters:
        time_array: Array representing time intervals.
        permeability_array: Array representing permeability values.
        NinA, NinB, NoutB, Nrec: Arrays representing molecule counts.
        switching_pattern: The switching pattern used in the simulation.
        config: Configuration object with simulation details.
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

    """
    Plots the release and reception of B molecules over time in a point transmitter simulation.

    This function generates a dual-axis plot showing the number of B molecules released 
    (NoutB) and the number of B molecules received (Nrec) by the receiver over the time steps
    defined in `time_array`. The plot will save the output as a PNG image in a designated
    output folder with the current timestamp.

    Parameters:
        time_array (np.ndarray): An array of time points corresponding to the simulation.
        data_pointTx (dict): A dictionary containing:
        NoutB (np.ndarray): The number of B molecules released over time.
        Nrec (np.ndarray): The number of B molecules received over time.
        config: A configuration object that contains settings, including the output folder path.

    Returns:
        None: The function saves the plot and does not return any values.
    """

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
        time_array: Array of time values.
        rho: Array of permeability values.
        switching_pattern: The switching pattern used in the simulation.
        Ts: Time interval for the switching pattern.
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
        time_array: Array of time values.
        rho: Array of permeability values.
        switching_pattern: The switching pattern used in the simulation.
        Ts: Time interval for the switching pattern.
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
    """
    Calculate the volume and initial concentrations of particles inside and outside the transmitter.

    This function computes the internal and external volumes based on the radii of the 
    transmitter and the surrounding environment. It also initializes the concentrations of 
    different particle types (R, S, MR, ER, ES) for both the inner and outer spaces.

    Parameters:
        r_tx (float): Radius of the transmitter (m).
        r_out (float): Radius of the environment (m).

    Returns:
        tuple: A tuple containing:
            - vol_in (float): The volume of the inside of the transmitter (m³).
            - vol_out (float): The volume of the surrounding environment (m³).
            - conc_in (Space): An object representing concentrations of particles inside the transmitter.
            - conc_out (Space): An object representing concentrations of particles outside the transmitter.
    """
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
