import numpy as np
from utils import get_conc_vol_for_practical, plot_hitting_prob
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
    NoutB = np.zeros_like(time_array)

    vol_in, vol_out, conc_in, conc_out = get_conc_vol_for_practical(config.r_tx, config.r_out)


    segment_length = len(time_array) // len(switching_pattern)

    release_indices = []
    for i, state in enumerate(switching_pattern):
        start_index = i * segment_length
        if state == 1:
            release_indices.append(start_index)
            #NoutB[start_index] = config.N
            NoutB[start_index:] += config.N

    release_pattern[release_indices] = config.N


    # Create receiver and perform convolution to simulate reception of released molecules
    rec = AbsorbingReceiver(config.r_rx) if config.receiver_type == 'AbsorbingReceiver' else TransparentReceiver(config.r_rx)
    #rec = TransparentReceiver(config.r_rx)

    # Simulate average hits at the receiver
    # TODO change hitting probability https://ieeexplore.ieee.org/document/8742793
    hitting_prob = rec.hitting_prob_point(t=time_array,
                                    D=config.D_space,
                                    dist=config.dist,
                                    )
    # plot_hitting_prob(time_array, hitting_prob)


    # avg_hits_inst = rec.average_hits(time_array, S_released_instant, config.r_tx, config.D_space, config.dist, config.k_d)
    # avg_hits_inst = avg_hits_inst[:config.step_count] * config.step_time
    # avg_hits = np.cumsum(avg_hits_inst)
    NoutB_instant = np.concatenate(([0], NoutB))
    NoutB_instant = NoutB_instant[1:] - NoutB_instant[:-1]
    
    # print(f"NoutB instant: {np.sum(NoutB_instant)}")

    Nrec_inst = rec.average_hits(time_array,
                         NoutB_instant,
                         config.r_tx,
                         config.D_space, # Diffusion Cofficient of Space
                         config.dist, # Distance between Tx and Rx
                         exp='point')

    # print(f"Sum of Nrec_inst with avg_hits: {np.sum(Nrec_inst)}")

    Nrec = Nrec_inst[:config.step_count] # * config.step_time

    # print(f"Sum of Nrec_inst after *config.step_time: {np.sum(Nrec_inst)}")

    Nrecv = np.cumsum(Nrec)

    return {
            'NoutB': NoutB,
            'Nrec': Nrec
            }
