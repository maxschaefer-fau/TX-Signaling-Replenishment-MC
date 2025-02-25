import numpy as np
from models.reaction_new import *
from tqdm import tqdm
from models.space import AbsorbingReceiver, TransparentReceiver

def step(reactions, molars, flows, t):
    """
    Advances the chemical reactions by one time step.

    This function performs the calculations necessary to update the state of the
    reactions based on the given time step. It updates the concentrations of substrates
    and products for the specified reactions.

    Parameters:
        reactions (list): A list of reaction objects that define the chemical steps.
        molars (list): A list of current molar concentrations of chemical species.
        flows (np.ndarray): An array representing the flow rates of the substances.
        t (float): The current time step for which the reactants are being updated.
    """

    # Step through the reactions in sequence and update concentration
    reactions[0].step(t)  # First reaction (E + R -> ER)
    reactions[1].update_conc(substrate_conc=reactions[0].product_conc)  # Update second reaction based on the product of the first
    reactions[2].update_conc(product_conc=[reactions[0].substrate_conc[0], reactions[2].product_conc[1]])

    reactions[1].step(t)  # Second reaction (ER -> ES)
    reactions[2].update_conc(substrate_conc=reactions[1].product_conc)  # Update third reaction based on the product of the second
    reactions[0].update_conc(product_conc=reactions[1].substrate_conc)  # Update first reaction's concentrations

    reactions[2].step(t)  # Third reaction (ES -> E + S)
    reactions[0].update_conc(substrate_conc=[reactions[2].product_conc[0], reactions[0].substrate_conc[1]])
    reactions[1].update_conc(product_conc=reactions[2].substrate_conc)

def practical_transmitter(time_array: np.ndarray, rho_array: np.ndarray, conc_in, conc_out, config) -> dict:
    """
    Simulates the behavior of a practical transmitter system over time, including
    diffusion and reaction processes involving various particles.

    Parameters:
        time_array (np.ndarray): An array of time points at which the simulation is evaluated.
        rho_array (np.ndarray): An array representing the state of permeability at each time step.
        conc_in (Concentration): An object representing the concentration of molecules inside the transmitter.
        conc_out (Concentration): An object representing the concentration of molecules outside the transmitter.
        config (Config): A configuration object holding parameters like reaction rates, permeability coefficients, 
                       and step time.

    Returns:
        dict: A dictionary containing the concentration counts of:
            - NinR: Concentration of particle type 'R' inside the transmitter.
            - NinS: Concentration of particle type 'S' inside the transmitter.
            - NoutS: Concentration of particle type 'S' outside the transmitter.
            - Nrec: Average hits over time corresponding to the released molecules.
    """

    # Initialize the appropriate receiver based on configuration
    rec = AbsorbingReceiver(config.r_rx) if config.receiver_type == 'AbsorbingReceiver' else TransparentReceiver(config.r_rx)

    # Setup the reaction objects for the chemical processes
    E_and_R_to_ER = Two2OneReaction(k1=config.k1, k_1=config.k_1,
                                    substrate_conc=[conc_in.particles['MR'].concentration, conc_in.particles['R'].concentration],
                                    product_conc=conc_in.particles['ER'].concentration)
    ER_to_ES = One2OneReaction(k1=config.k2, k_1=config.k_2,
                               substrate_conc=conc_in.particles['ER'].concentration,
                               product_conc=conc_in.particles['ES'].concentration)
    ES_to_E_and_S = One2TwoReaction(k1=config.k3, k_1=config.k_3,
                                    substrate_conc=conc_in.particles['ES'].concentration,
                                    product_conc=[conc_in.particles['MR'].concentration, conc_in.particles['S'].concentration])

    # Initialize results storage
    conc_in_molars = {key: np.zeros_like(time_array) for key in conc_in.particles}
    conc_out_molars = {key: np.zeros_like(time_array) for key in conc_out.particles}
    conc_in_counts = {key: np.zeros_like(time_array) for key in conc_in.particles}
    conc_out_counts = {key: np.zeros_like(time_array) for key in conc_out.particles}
    S_released_count = np.zeros_like(time_array)

    # Progress bar for simulation feedback
    pbar = tqdm(total=len(time_array))

    # Simulation Loop
    for t_step, time in enumerate(time_array):
        perm_state = rho_array[t_step]  # Current permeability state from rho_array

        # Perform diffusion in the transmitter system
        conc_in.diffuse(conc_out, perm_state, config.step_time)

        # Update reaction states
        E_and_R_to_ER.update_conc([conc_in.particles['MR'].concentration, conc_in.particles['R'].concentration],
                                  conc_in.particles['ER'].concentration)
        ER_to_ES.update_conc(conc_in.particles['ER'].concentration, conc_in.particles['ES'].concentration)
        ES_to_E_and_S.update_conc(conc_in.particles['ES'].concentration,
                                  [conc_in.particles['MR'].concentration, conc_in.particles['S'].concentration])

        # Execute step for reactions
        molars = [conc_in.particles[name].concentration for name in ['R', 'S', 'MR', 'ER', 'ES']]
        step(reactions=[E_and_R_to_ER, ER_to_ES, ES_to_E_and_S], molars=molars, flows=np.zeros(5), t=config.step_time)

        # Update concentrations of the molecules in the input
        conc_in.particles['MR'].set_conc(ES_to_E_and_S.product_conc[0])
        conc_in.particles['R'].set_conc(E_and_R_to_ER.substrate_conc[1])
        conc_in.particles['S'].set_conc(ES_to_E_and_S.product_conc[1])
        conc_in.particles['ER'].set_conc(ER_to_ES.substrate_conc)
        conc_in.particles['ES'].set_conc(ER_to_ES.product_conc)

        # Store the results of concentrations and counts over time
        for key in conc_in_molars:
            conc_in_molars[key][t_step] = conc_in.particles[key].concentration
            conc_out_molars[key][t_step] = conc_out.particles[key].concentration
            conc_in_counts[key][t_step] = conc_in.particles[key].count
            conc_out_counts[key][t_step] = conc_out.particles[key].count

        # Count the released S molecules outside the transmitter
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

    # Instantaneous increase in released molecule counts
    S_released_instant = np.concatenate(([0], S_released_count))
    S_released_instant = S_released_instant[1:] - S_released_instant[:-1]

    # Average hit counts based on released molecules
    avg_hits_inst = rec.average_hits(time_array, S_released_instant, config.r_tx, config.D_space, config.dist, config.k_d)
    avg_hits_inst = avg_hits_inst[:config.step_count] * config.step_time
    avg_hits = np.cumsum(avg_hits_inst)

    # Return results directly as a dictionary
    return {
            'NinR': conc_in_counts['R'],
            'NinS': conc_in_counts['S'],
            'NoutS': conc_out_counts['S'],
            'Nrec': avg_hits
            }
