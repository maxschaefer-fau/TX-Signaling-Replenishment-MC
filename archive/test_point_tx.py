import numpy as np
import matplotlib.pyplot as plt
from utils import get_conc_vol_for_practical, plot_hitting_prob
from models.space import AbsorbingReceiver, TransparentReceiver
import numpy as np
from config import NanomachineConfig
from utils import generate_switching_pattern, get_conc_vol_for_practical, plot_pointTx, save_to_csv, plot_data

# Set Config
conf = NanomachineConfig()
vol_in, vol_out, conc_in, conc_out = get_conc_vol_for_practical(conf.r_tx, conf.r_out)

# Set Switching pattern
switching_pattern = [1,0,0,1,0,1,0,0,0,0,0]

Ts = 10
conf.simulation_end = len(switching_pattern) * Ts

# Config Change
# conf.dist *= .5

# Time Array
time_array = np.linspace(0,
                         conf.simulation_end,
                         int(conf.simulation_end / conf.step_time),
                         endpoint=False)

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
    #hitting_prob = rec.hitting_prob_point(t=time_array,
    #                                D=config.D_space,
    #                                dist=config.dist,
    #                                )
    # plot_hitting_prob(time_array, hitting_prob)


    # avg_hits_inst = rec.average_hits(time_array, S_released_instant, config.r_tx, config.D_space, config.dist, config.k_d)
    # avg_hits_inst = avg_hits_inst[:config.step_count] * config.step_time
    # avg_hits = np.cumsum(avg_hits_inst)
    NoutB_instant = np.concatenate(([0], NoutB))
    NoutB_instant = NoutB_instant[1:] - NoutB_instant[:-1]
    
    print(f"NoutB instant: {np.sum(NoutB_instant)}")

    Nrec_inst = rec.average_hits(time_array,
                         NoutB_instant,
                         config.r_tx,
                         config.D_space, # Diffusion Cofficient of Space
                         config.dist, # Distance between Tx and Rx
                                 exp='point')

    print(f"Sum of Nrec_inst with avg_hits: {np.sum(Nrec_inst)}")

    Nrec = Nrec_inst[:len(time_array)]  * config.step_time

    print(f"Sum of Nrec_inst after *config.step_time: {np.sum(Nrec)}")

    Nrecv = np.cumsum(Nrec)

        # Plotting the results
    plt.figure(figsize=(10, 6))
    plt.plot(time_array, NoutB, 'r-', label='NoutB (Released)')
    plt.plot(time_array, NoutB_instant, 'g--', label='NoutB Instantaneous Release')
    plt.plot(time_array, Nrec_inst[:len(time_array)], 'm:', label='Nrec Instantaneous')
    plt.plot(time_array, Nrec, 'b-', label='Nrec (Received)')
    plt.plot(time_array, Nrecv, 'c-', label='Cumulative Received (Nrecv)')

    plt.xlabel('Time (s)')
    plt.ylabel('Molecules')
    plt.title('Molecular Release and Reception')
    plt.legend(loc='upper right')
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    return {
            'NoutB': NoutB,
            'Nrec': Nrec,
            'NoutB_instant': NoutB_instant,
            'Nrec_inst': Nrec_inst,
            'Nrecv': Nrecv
            }

results = point_transmitter(time_array, switching_pattern, conf)
