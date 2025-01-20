import numpy as np
from models.reaction_new import *
from tqdm import tqdm
from models.space import AbsorbingReceiver, TransparentReceiver

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


def practical_transmitter(time_array, rho, conc_in, conc_out, config):
    """
    Simulates the behavior of a practical transmitter system over time, including
    diffusion and reaction processes involving various particles.

    Parameters:
    - time_array (numpy.ndarray): An array of time points at which the simulation is evaluated.
    - rho (numpy.ndarray): An array representing the state of permeability at each time step.
    - conc_in (Concentration): An object representing the concentration of molecules inside the transmitter.
    - conc_out (Concentration): An object representing the concentration of molecules outside the transmitter.
    - config (Config): A configuration object holding parameters like reaction rates, permeability coefficients, 
                       step time, etc.

    Returns:
    - List: A list containing the concentration counts of:
        1. Concentration of particle type 'R' inside the transmitter.
        2. Concentration of particle type 'S' inside the transmitter.
        3. Concentration of particle type 'S' outside the transmitter.
        4. Average hits over time corresponding to the released molecules.

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
    rec = AbsorbingReceiver(config.r_rx)
    if config.receiver_type == 'AbsorbingReceiver':
        pass
    elif config.receiver_type == 'TransparentReceiver':
        rec = TransparentReceiver(config.r_rx)

    # Reaction variables
    E_and_R_to_ER = Two2OneReaction(k1=config.k1, k_1=config.k_1, substrate_conc=[conc_in.particles['MR'].concentration, conc_in.particles['R'].concentration], product_conc=conc_in.particles['ER'].concentration)
    ER_to_ES = One2OneReaction(k1=config.k2, k_1=config.k_2, substrate_conc=conc_in.particles['ER'].concentration, product_conc=conc_in.particles['ES'].concentration)
    ES_to_E_and_S = One2TwoReaction(k1=config.k3, k_1=config.k_3, substrate_conc=conc_in.particles['ES'].concentration, product_conc=[conc_in.particles['MR'].concentration, conc_in.particles['S'].concentration])

    # Utility variables
    particle_names = ['R', 'S', 'MR', 'ER', 'ES']
    flow = np.zeros((5,))

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
        perm_state = rho[t_step]
        #perm_state = p * perm_states[t_step]
        # Diffusion Step
        conc_in.diffuse(conc_out, perm_state, config.step_time)
        # Update Reaction states
        E_and_R_to_ER.update_conc([conc_in.particles['MR'].concentration, conc_in.particles['R'].concentration], conc_in.particles['ER'].concentration)
        ER_to_ES.update_conc(conc_in.particles['ER'].concentration, conc_in.particles['ES'].concentration)
        ES_to_E_and_S.update_conc(conc_in.particles['ES'].concentration, [conc_in.particles['MR'].concentration, conc_in.particles['S'].concentration])
        # Step for the Reaction
        molars = [conc_in.particles[name].concentration for name in particle_names]
        step(reactions=[E_and_R_to_ER, ER_to_ES, ES_to_E_and_S], molars=molars, flows=flow, t=config.step_time)

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
    hit_probs = rec.hitting_prob(time_array, config.r_tx, config.D_space, config.dist, config.k_d)
    hit_probs *= config.step_time   # Convert from units to time steps
    # hit_probs = np.diff(hit_probs)
    #avg_hits_inst = sig.convolve(S_released_instant, hit_probs, mode='full')
    #avg_hits_inst = avg_hits_inst[:step_count]
    avg_hits_inst = rec.average_hits(time_array, S_released_instant, config.r_tx, config.D_space, config.dist, config.k_d)
    avg_hits_inst = avg_hits_inst[:config.step_count]
    avg_hits = np.cumsum(avg_hits_inst)
    # print('Average hits calculated')
    # Convert all data in time steps to units
    avg_hits_inst *= config.steps_in_a_unit
    hit_probs *= config.steps_in_a_unit
    S_released_instant *= config.steps_in_a_unit

    # Return the results directly as a dictionary
    return {
        'NinR': conc_in_counts['R'],
        'NinS': conc_in_counts['S'],
        'NoutS': conc_out_counts['S'],
        'Nrec': avg_hits
    }

