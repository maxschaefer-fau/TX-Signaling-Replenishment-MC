import numpy as np
from utils import get_conc_vol_for_practical
from models.space import AbsorbingReceiver, TransparentReceiver

def point_transmitter(time_array, switching_pattern, config):
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

    vol_in, vol_out, conc_in, conc_out = get_conc_vol_for_practical(config.r_tx, config.r_out)

    segment_length = len(time_array) // len(switching_pattern)

    release_indices = []
    for i, state in enumerate(switching_pattern):
        start_index = i * segment_length
        if state == 1:
            release_indices.append(start_index)

    release_pattern[release_indices] = config.N

    # Convert release pattern to number of molecules outside
    NoutB = release_pattern

    # Create receiver and perform convolution to simulate reception of released molecules
    rec = AbsorbingReceiver(config.r_rx) if config.receiver_type == 'AbsorbingReceiver' else TransparentReceiver(config.r_rx)
    #rec = TransparentReceiver(config.r_rx)

    # Simulate average hits at the receiver
    # TODO change hitting probability https://ieeexplore.ieee.org/document/8742793
    # hitting_prob = rec.hitting_prob_point(t=time_array, D=conf.D_space, dist=conf.dist)
    # print(hitting_prob)
    nr = rec.average_hits(time_array,
                          NoutB,
                          config.r_tx,
                          config.D_space,
                          config.dist,
                          exp='point')

    Nrec = nr[:len(time_array)] * config.step_time

    return {
            'NoutB': NoutB,
            'Nrec': Nrec
            }
