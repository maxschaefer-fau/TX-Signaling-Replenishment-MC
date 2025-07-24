import numpy as np
import os
import random
from utils.config import NanomachineConfig
from models.reaction_new import Two2OneReaction, One2OneReaction, One2TwoReaction
from utils.utils import (
    generate_random_switching_pattern,
    generate_permeability_pattern,
    get_conc_vol_for_practical
)

def generate_state_data(time_array: np.ndarray, rho_array: np.ndarray, conc_in, conc_out, config) -> np.ndarray:
    E_and_R_to_ER = Two2OneReaction(k1=config.k1, k_1=config.k_1,
                                    substrate_conc=[conc_in.particles['MR'].concentration, conc_in.particles['R'].concentration],
                                    product_conc=conc_in.particles['ER'].concentration)
    ER_to_ES = One2OneReaction(k1=config.k2, k_1=config.k_2,
                               substrate_conc=conc_in.particles['ER'].concentration,
                               product_conc=conc_in.particles['ES'].concentration)
    ES_to_E_and_S = One2TwoReaction(k1=config.k3, k_1=config.k_3,
                                    substrate_conc=conc_in.particles['ES'].concentration,
                                    product_conc=[conc_in.particles['MR'].concentration, conc_in.particles['S'].concentration])

    state_data = []

    for t_step, time in enumerate(time_array):
        perm_state = rho_array[t_step]
        conc_in.diffuse(conc_out, perm_state, config.step_time)

        E_and_R_to_ER.update_conc([conc_in.particles['MR'].concentration, conc_in.particles['R'].concentration],
                                  conc_in.particles['ER'].concentration)
        ER_to_ES.update_conc(conc_in.particles['ER'].concentration, conc_in.particles['ES'].concentration)
        ES_to_E_and_S.update_conc(conc_in.particles['ES'].concentration,
                                  [conc_in.particles['MR'].concentration, conc_in.particles['S'].concentration])

        # Reaction steps
        E_and_R_to_ER.step(config.step_time)
        ER_to_ES.update_conc(substrate_conc=E_and_R_to_ER.product_conc)
        ES_to_E_and_S.update_conc(product_conc=[E_and_R_to_ER.substrate_conc[0], ES_to_E_and_S.product_conc[1]])
        ER_to_ES.step(config.step_time)
        ES_to_E_and_S.update_conc(substrate_conc=ER_to_ES.product_conc)
        E_and_R_to_ER.update_conc(product_conc=ER_to_ES.substrate_conc)
        ES_to_E_and_S.step(config.step_time)
        E_and_R_to_ER.update_conc(substrate_conc=[ES_to_E_and_S.product_conc[0], E_and_R_to_ER.substrate_conc[1]])
        ER_to_ES.update_conc(product_conc=ES_to_E_and_S.substrate_conc)

        # Update concentrations
        conc_in.particles['MR'].set_conc(ES_to_E_and_S.product_conc[0])
        conc_in.particles['R'].set_conc(E_and_R_to_ER.substrate_conc[1])
        conc_in.particles['S'].set_conc(ES_to_E_and_S.product_conc[1])
        conc_in.particles['ER'].set_conc(ER_to_ES.substrate_conc)
        conc_in.particles['ES'].set_conc(ER_to_ES.product_conc)

        state_t = [
            conc_in.particles['R'].concentration,
            conc_in.particles['S'].concentration,
            conc_in.particles['MR'].concentration,
            conc_in.particles['ER'].concentration,
            conc_in.particles['ES'].concentration,
            perm_state
        ]
        state_data.append(state_t)

    return np.array(state_data)

def simulate_and_collect(Ts, mode, config, num_switch_range=(80, 150)):
    num_switches = random.randint(*num_switch_range)
    switching_pattern = generate_random_switching_pattern(length=num_switches, padding=4)
    config.simulation_end = len(switching_pattern) * Ts

    time_array = np.arange(0, config.simulation_end, 1e-3)


    rho_array = generate_permeability_pattern(
        mode=mode,
        switching_pattern=switching_pattern,
        length=len(time_array),
        config=config,
        peak_duration_ratio=0.05,
        zero_duration_ratio=0.05
    )

    vol_in, vol_out, conc_in, conc_out = get_conc_vol_for_practical(config.r_tx, config.r_out)

    state_data = generate_state_data(time_array, rho_array, conc_in, conc_out, config)
    print(f"Generated: mode={mode}, Ts={Ts}s, switches={num_switches}, shape={state_data.shape}")
    return state_data

if __name__ == "__main__":
    conf = NanomachineConfig()
    switching_times = [random.randint(25, 85) for _ in range(4)] #[2, 5, 8, 11]
    modes = ['practical']
    num_switch_range = (5, 25)  # You can adjust this

    all_data = []

    for Ts in switching_times:
        for mode in modes:
            state_data = simulate_and_collect(Ts, mode, conf, num_switch_range=num_switch_range)
            all_data.append(state_data)

    final_dataset = np.concatenate(all_data, axis=0)
    os.makedirs('data', exist_ok=True)
    np.save('data/koopman_state_data_practical_concat_test.npy', final_dataset)
    print(f"Saved concatenated dataset: shape = {final_dataset.shape}")
