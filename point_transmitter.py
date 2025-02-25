import numpy as np
from models.space import AbsorbingReceiver, TransparentReceiver

def point_transmitter(time_array: np.ndarray, switching_pattern: list[int], config) -> dict:
    """
    Simulate the release of N molecules of type B at specified intervals defined by the switching pattern.

    This function models the behavior of a point transmitter that releases a certain number 
    of molecules at each specified time point in the switching pattern. Additionally, it simulates 
    the interaction of these molecules with a specified receiver type, calculating the number 
    of molecules received over time.

    Parameters:
        time_array (np.ndarray): An array of time points for the simulation.
        switching_pattern (list[int]): A list of binary values indicating when to release molecules
                                        (1 for release, 0 for no release).
        config: Configuration object containing receiver and simulation parameters (e.g., r_tx, r_rx, N).

    Returns:
        dict: A dictionary with the following keys:
            - NoutB (np.ndarray): The cumulative number of B molecules released over time.
            - Nrec (np.ndarray): The number of B molecules received over time after convolution with the receiver's response handling.
    """

    # Initialize arrays to track the release pattern and number of B molecules released
    release_pattern = np.zeros_like(time_array)
    NoutB = np.zeros_like(time_array)

    # Calculate segment length based on the time array and switching pattern length
    segment_length = len(time_array) // len(switching_pattern)

    # Collect indices where molecules will be released based on the switching pattern
    release_indices = []
    for i, state in enumerate(switching_pattern):
        start_index = i * segment_length
        if state == 1:  # If the state is '1', molecules should be released
            release_indices.append(start_index)
            NoutB[start_index:] += config.N  # Add number of molecules from this point onwards

    # Assign the number of molecules to the release pattern
    release_pattern[release_indices] = config.N

    # Create the appropriate receiver instance based on the configuration
    rec = AbsorbingReceiver(config.r_rx) if config.receiver_type == 'AbsorbingReceiver' else TransparentReceiver(config.r_rx)

    # Adjust instant released molecules for further processing
    NoutB_instant = np.concatenate(([0], NoutB))  # Concatenate to handle differences
    NoutB_instant = NoutB_instant[1:] - NoutB_instant[:-1]  # Compute instant release amounts

    # Compute received molecules based on the average hits and release amounts
    Nrec_inst = rec.average_hits(
        time_array,
        NoutB_instant,
        config.r_tx,
        config.D_space,
        config.dist,
        exp='point'
    )

    # Limit the number of received molecules to the number of simulation steps
    Nrec = Nrec_inst[:config.step_count]

    # Return the results as a dictionary
    return {
        'NoutB': NoutB,
        'Nrec': Nrec
    }
