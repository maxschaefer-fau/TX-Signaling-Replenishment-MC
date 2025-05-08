import numpy as np
from utils.config import NanomachineConfig
from models.reaction_new import Two2OneReaction, One2OneReaction, One2TwoReaction
from models.space import AbsorbingReceiver, TransparentReceiver
from utils.utils import (
    generate_random_switching_pattern,
    generate_permeability_pattern,
    get_conc_vol_for_practical,
    plot_data,
    plot_pointTx,
    save_to_csv
)

# Initialize the configuration
conf = NanomachineConfig()

# Setup volumes and concentrations
vol_in, vol_out, conc_in, conc_out = get_conc_vol_for_practical(conf.r_tx, conf.r_out)

switching_pattern = generate_random_switching_pattern(length=10, padding=5)

# Define time array
Ts = 7  # Time duration for each switch in switching pattern
conf.simulation_end = len(switching_pattern) * Ts  # Set total simulation time

time_array = np.linspace(0,
                         conf.simulation_end,
                         int(conf.simulation_end / conf.step_time),
                         endpoint=False)

# Generate permeability pattern (ideal mode in this example)
rho_array = generate_permeability_pattern(mode='ideal',
                                    switching_pattern=switching_pattern,
                                    length=len(time_array),
                                    config=conf,
                                    peak_duration_ratio=0.05,
                                    zero_duration_ratio=0.05)


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


def generate_state_data(time_array: np.ndarray, rho_array: np.ndarray, conc_in, conc_out, config) -> np.ndarray:
    """
    Generates the state vector x(t) over time for the practical transmitter system, including concentrations of species
    and the permeability at each time step.

    Parameters:
        time_array (np.ndarray): An array of time points at which the simulation is evaluated.
        rho_array (np.ndarray): An array representing the state of permeability at each time step.
        conc_in (Concentration): An object representing the concentration of molecules inside the transmitter.
        conc_out (Concentration): An object representing the concentration of molecules outside the transmitter.
        config (Config): A configuration object holding parameters like reaction rates, permeability coefficients, 
                         and step time.

    Returns:
        np.ndarray: A 2D array where each row represents the state vector x(t) at each time step.
                    Shape: (time_steps, 6) for [R, S, MR, ER, ES, rho(t)].
    """

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

    # Initialize storage for the state vectors (R, S, MR, ER, ES, rho)
    state_data = []

    # Simulation Loop to generate state vectors
    for t_step, time in enumerate(time_array):
        perm_state = rho_array[t_step]  # Current permeability state from rho_array

        # Perform diffusion in the transmitter system
        conc_in.diffuse(conc_out, perm_state, conf.step_time)

        # Update reaction states
        E_and_R_to_ER.update_conc([conc_in.particles['MR'].concentration, conc_in.particles['R'].concentration],
                                  conc_in.particles['ER'].concentration)
        ER_to_ES.update_conc(conc_in.particles['ER'].concentration, conc_in.particles['ES'].concentration)
        ES_to_E_and_S.update_conc(conc_in.particles['ES'].concentration,
                                  [conc_in.particles['MR'].concentration, conc_in.particles['S'].concentration])

        # Execute step for reactions
        molars = [conc_in.particles[name].concentration for name in ['R', 'S', 'MR', 'ER', 'ES']]
        step(reactions=[E_and_R_to_ER, ER_to_ES, ES_to_E_and_S], molars=molars, flows=np.zeros(5), t=conf.step_time)

        # Update concentrations of the molecules in the input
        conc_in.particles['MR'].set_conc(ES_to_E_and_S.product_conc[0])
        conc_in.particles['R'].set_conc(E_and_R_to_ER.substrate_conc[1])
        conc_in.particles['S'].set_conc(ES_to_E_and_S.product_conc[1])
        conc_in.particles['ER'].set_conc(ER_to_ES.substrate_conc)
        conc_in.particles['ES'].set_conc(ER_to_ES.product_conc)

        # Store the state vector x(t) for this time step
        state_t = [
            conc_in.particles['R'].concentration,  # Unreacted resource (R)
            conc_in.particles['S'].concentration,  # Produced signal (S)
            conc_in.particles['MR'].concentration,  # Free enzyme (MR)
            conc_in.particles['ER'].concentration,  # Intermediate complex 1 (ER)
            conc_in.particles['ES'].concentration,  # Intermediate complex 2 (ES)
            perm_state  # Permeability at time t (rho(t))
        ]
        state_data.append(state_t)

    # Convert list of state vectors to a numpy array for easy handling
    return np.array(state_data)

# Assuming time_array, rho_array, conc_in, conc_out, and config are already defined
state_data = generate_state_data(time_array, rho_array, conc_in, conc_out, conf)

np.save('data/koopman_state_data_ideal_7s.npy', state_data)
