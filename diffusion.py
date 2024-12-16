from time import time
from Constants_Diff import *
import numpy as np
import matplotlib.pyplot as plt
import pickle as pk
from Models.Reaction_new import *
from tqdm import tqdm
import os
from pathlib import Path
import scipy.signal as sig

from Models.Space import Receiver
from create_state_values import create_state_array

results_folder = Path(__file__).parent / 'diffusion_res'
os.makedirs(str(results_folder), exist_ok=True)

def step(reactions, molars, flows, t):
    
    reactions[0].step(t)
    reactions[1].update_conc(substrate_conc=reactions[0].product_conc)
    reactions[2].update_conc(product_conc=[reactions[0].substrate_conc[0], reactions[2].product_conc[1]])

    reactions[1].step(t)
    reactions[2].update_conc(substrate_conc=reactions[1].product_conc)
    reactions[0].update_conc(product_conc=reactions[1].substrate_conc)

    reactions[2].step(t)
    reactions[0].update_conc(substrate_conc=[reactions[2].product_conc[0], reactions[0].substrate_conc[1]])
    reactions[1].update_conc(product_conc=reactions[2].substrate_conc)


def main():
    """
    # The simulation state
    state = 0
    end_state = len(state_breaks)
    perm_state = p
    membrane_open = True
    perm_close_steps = int(p_close / step_time)
    if perm_close_steps > 0:
        perm_step_change = p / perm_close_steps
    else:
        perm_step_change = 2 * p
    """

    # The receiver variable
    rec = Receiver(r_rx)

    # Reaction variables
    E_and_R_to_ER = Two2OneReaction(k1=k1, k_1=k_1, substrate_conc=[conc_in.particles['MR'].concentration, conc_in.particles['R'].concentration], product_conc=conc_in.particles['ER'].concentration)
    ER_to_ES = One2OneReaction(k1=k2, k_1=k_2, substrate_conc=conc_in.particles['ER'].concentration, product_conc=conc_in.particles['ES'].concentration)
    ES_to_E_and_S = One2TwoReaction(k1=k3, k_1=k_3, substrate_conc=conc_in.particles['ES'].concentration, product_conc=[conc_in.particles['MR'].concentration, conc_in.particles['S'].concentration])

    # Utility variables
    time_array = np.arange(0, simulation_end, step_time)
    particle_names = ['R', 'S', 'MR', 'ER', 'ES']
    flow = np.zeros((5,))

    # State variables
    try:
        perm_states = np.genfromtxt(state_path, delimiter=',')
    except FileNotFoundError:
        perm_states = create_state_array(state_path, 1.0, p_close, release_count, end_part, time_array)

    # Variables to store results
    conc_in_molars = {}
    conc_out_molars = {}
    conc_in_counts = {}
    conc_out_counts = {}
    for key in conc_in.particles:
        conc_in_molars[key] = np.zeros_like(time_array)
        conc_out_molars[key] = np.zeros_like(time_array)
        conc_in_counts[key] = np.zeros_like(time_array)
        conc_out_counts[key] = np.zeros_like(time_array)
    S_released_count = np.zeros_like(time_array)

    # Verbose variables
    pbar = tqdm(total=time_array.shape[0])

    # Simulation Loop
    for t_step, time in enumerate(time_array):
        perm_state = p * perm_states[t_step]
        # Diffusion Step
        conc_in.diffuse(conc_out, perm_state, step_time)
        # Update Reaction states
        E_and_R_to_ER.update_conc([conc_in.particles['MR'].concentration, conc_in.particles['R'].concentration], conc_in.particles['ER'].concentration)
        ER_to_ES.update_conc(conc_in.particles['ER'].concentration, conc_in.particles['ES'].concentration)
        ES_to_E_and_S.update_conc(conc_in.particles['ES'].concentration, [conc_in.particles['MR'].concentration, conc_in.particles['S'].concentration])
        # Step for the Reaction
        molars = [conc_in.particles[name].concentration for name in particle_names]
        step(reactions=[E_and_R_to_ER, ER_to_ES, ES_to_E_and_S], molars=molars, flows=flow, t=step_time)

        # Update Diffusion States
        conc_in.particles['MR'].set_conc(ES_to_E_and_S.product_conc[0])
        conc_in.particles['R'].set_conc(E_and_R_to_ER.substrate_conc[1])
        conc_in.particles['S'].set_conc(ES_to_E_and_S.product_conc[1])
        conc_in.particles['ER'].set_conc(ER_to_ES.substrate_conc)
        conc_in.particles['ES'].set_conc(ER_to_ES.product_conc)

        # Store the results
        for key in conc_in_molars:
            conc_in_molars[key][t_step] = conc_in.particles[key].concentration
            conc_out_molars[key][t_step] = conc_out.particles[key].concentration
            conc_in_counts[key][t_step] = conc_in.particles[key].count
            conc_out_counts[key][t_step] = conc_out.particles[key].count
        S_released_count[t_step] = conc_out.particles['S'].count

        """
        # Update the states
        if state != end_state and time >= simulation_end * state_breaks[state]:
            state += 1
            # Switch the membrane
            if state % 2 == 0:
                membrane_open = True
            else:
                membrane_open = False

        # Change permeability for partially open membrane
        if not membrane_open and perm_state > 0:
            perm_state -= perm_step_change
            perm_state = max(0, perm_state)
        elif membrane_open and perm_state < p:
            perm_state += perm_step_change
            perm_state = min(p, perm_state)
        """
        
        # Update the progress bar
        pbar.update()
    
    # Calculate the average hits
    # Instantaneous increase in released molecule counts
    S_released_instant = np.concatenate(([0], S_released_count))
    S_released_instant = S_released_instant[1:] - S_released_instant[:-1]
    # Average hit counts
    # print('Calculating average hits')
    hit_probs = rec.hitting_prob(time_array, r_tx, D_space, l, k_d)
    hit_probs *= step_time   # Convert from units to time steps
    # hit_probs = np.diff(hit_probs)
    avg_hits_inst = sig.convolve(S_released_instant, hit_probs, mode='full')
    avg_hits_inst = avg_hits_inst[:step_count]
    # avg_hits_inst = rec.average_hits(time_array+step_time, S_released_instant, r_tx, D, l, k_d)
    avg_hits = np.cumsum(avg_hits_inst)
    # print('Average hits calculated')
    # Convert all data in time steps to units
    avg_hits_inst *= steps_in_a_unit
    hit_probs *= steps_in_a_unit
    S_released_instant *= steps_in_a_unit
    

    # Plot the results
    # Concentration plots
    for key in conc_in_molars:
        plt.figure()
        plt.plot(time_array, conc_in_molars[key], label=key + ' conc. inside')
        plt.plot(time_array, conc_out_molars[key], label=key + ' conc. outside')
        plt.xlabel('Time (s)'), plt.ylabel('Concentration (' + key + ')')
        plt.title('Molar Concentration of (' + key + ')')
        plt.legend()
        plt.savefig(results_folder / (key + '_conc.png'))

    pk.dump(conc_in_molars, open(results_folder / 'conc_in_molars.p', 'wb'))
    pk.dump(conc_out_molars, open(results_folder / 'conc_out_molars.p', 'wb'))
    # Combined Concentration Plots
    plt.figure()
    for key in ['R', 'S']:
        plt.plot(time_array, conc_in_molars[key], label=key + ' conc. inside')
    plt.xlabel('Time (s)'), plt.ylabel('Concentration (Molars)')
    plt.title('Molar Concentration of R and S')
    plt.legend()
    plt.savefig(results_folder / ('R_S_conc.png'))

    plt.figure()
    for key in ['MR', 'ER', 'ES']:
        plt.plot(time_array, conc_in_molars[key], label=key + ' conc. inside')
    plt.xlabel('Time (s)'), plt.ylabel('Concentration (Molars)')
    plt.title('Molar Concentration of MR and Combined Forms')
    plt.legend()
    plt.savefig(results_folder / ('MR_ER_ES_conc.png'))
    # Count Plots
    for key in conc_in_counts:
        plt.figure()
        plt.plot(time_array, conc_in_counts[key], label=key + ' count inside')
        plt.plot(time_array, conc_out_counts[key], label=key + ' count outside')
        plt.xlabel('Time (s)'), plt.ylabel('Number of (' + key + ') molecules')
        plt.title('(' + key + ') Molecule Count')
        plt.legend()
        plt.savefig(results_folder / (key + '_count.png'))

    pk.dump(conc_in_counts, open(results_folder / 'conc_in_counts.p', 'wb'))
    pk.dump(conc_out_counts, open(results_folder / 'conc_out_counts.p', 'wb'))
    # Combined Count Plots
    plt.figure()
    for key in ['R', 'S']:
        plt.plot(time_array, conc_in_counts[key], label=key + ' count inside')
    plt.xlabel('Time (s)'), plt.ylabel('Number of molecules')
    plt.title('R and S Molecule Count')
    plt.legend()
    plt.savefig(results_folder / ('R_S_count.png'))

    plt.figure()
    for key in ['MR', 'ER', 'ES']:
        plt.plot(time_array, conc_in_counts[key], label=key + ' count inside')
    plt.xlabel('Time (s)'), plt.ylabel('Number of molecules')
    plt.title('MR and Combined Forms Molecule Count')
    plt.legend()
    plt.savefig(results_folder / ('MR_ER_ES_count.png'))
    # Released count plot
    plt.figure()
    plt.plot(time_array, S_released_count, label='(S)-Mandelate particles')
    #plt.yscale('symlog', linthresh=1e-15)
    plt.xlabel('Time (s)'), plt.ylabel('Particle Count')
    plt.title('Total # of released (S)-Mandelate molecules')
    plt.legend()
    plt.savefig(results_folder / 'S_released.png')
    pk.dump(S_released_count, open(results_folder / 'S_released_count.p', 'wb'))
    # Instantaneous Released count plot
    plt.figure()
    plt.plot(time_array, S_released_instant, label='(S)-Mandelate particles inst')
    #plt.yscale('symlog', linthresh=1e-15)
    plt.xlabel('Time (s)'), plt.ylabel('Particle Count Per Second')
    plt.title('Instantaneous # of released (S)-Mandelate molecules')
    plt.legend()
    plt.savefig(results_folder / 'S_released_inst.png')
    pk.dump(S_released_instant, open(results_folder / 'S_released_instant.p', 'wb'))
    # Hitting Probability Plot
    plt.figure()
    plt.plot(time_array, hit_probs, label='Hitting Probability')
    #plt.yscale('symlog', linthresh=1e-15)
    plt.xlabel('Time from release (s)'), plt.ylabel('Hitting Probability Distribution')
    plt.title('Hitting Probability from Release Time')
    plt.legend()
    plt.savefig(results_folder / 'hitting_probability.png')
    pk.dump(hit_probs, open(results_folder / 'hit_probs.p', 'wb'))
    # Average hit plot
    plt.figure()
    plt.plot(time_array, avg_hits, label='Average RX hit count')
    plt.xlabel('Time (s)'), plt.ylabel('Expected Received Particle Count')
    plt.title('Expected # of received (S)-Mandelate molecules')
    plt.legend()
    plt.savefig(results_folder / 'S_received.png')
    pk.dump(avg_hits, open(results_folder / 'avg_hits.p', 'wb'))
    # Instantaneous avg hit plot
    plt.figure()
    plt.plot(time_array, avg_hits_inst, label='Average inst. RX hit count')
    plt.xlabel('Time (s)'), plt.ylabel('Expected Received Particle Count Per Second')
    plt.title('Expected # of Instantaneous received (S)-Mandelate molecules from time step t=' + str(step_time) + ' s')
    plt.legend()
    plt.savefig(results_folder / 'S_received_inst.png')
    pk.dump(avg_hits_inst, open(results_folder / 'avg_hits_inst.p', 'wb'))
    # Permeability plot
    plt.figure()
    plt.plot(time_array, perm_states, label='Permeability State over Time')
    plt.xlabel('Time (s)'), plt.ylabel('Permeability (m/s)')
    plt.title('Permeability State over Time')
    plt.legend()
    plt.savefig(results_folder / 'permeability.png')
    pk.dump(perm_states, open(results_folder / 'permeability.p', 'wb'))


    # Time array
    pk.dump(time_array, open(results_folder / 'time_array.p', 'wb'))





if __name__ == '__main__':
    main()