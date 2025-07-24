import numpy as np
from utils.config import NanomachineConfig
from models.reaction_new import Two2OneReaction, One2OneReaction, One2TwoReaction
from models.space import AbsorbingReceiver, TransparentReceiver
from utils.utils import (
    get_conc_vol_for_practical,
    plot_data,
    plot_pointTx,
    save_to_csv
)

# Initialize the configuration
conf = NanomachineConfig()

def generate_permeability_pattern_dynamic(
    mode: str,
    switching_pattern: list[int],
    ts_list: list[float],
    config,
    step_time: float,
    peak_duration_ratio: float = 0.2,
    zero_duration_ratio: float = 0.2
) -> np.ndarray:

    total_steps = int(sum(ts_list) / step_time)
    rho_array = np.zeros(total_steps)

    start_index = 0
    for i, (state, Ts) in enumerate(zip(switching_pattern, ts_list)):
        segment_steps = int(Ts / step_time)
        end_index = start_index + segment_steps

        if mode == 'ideal':
            rho_array[start_index:end_index] = state * config.p

        elif mode == 'practical' and state == 1:
            peak_duration = int(segment_steps * peak_duration_ratio)
            zero_duration = int(segment_steps * zero_duration_ratio)
            rise_duration = (segment_steps - peak_duration - zero_duration) // 2
            fall_duration = segment_steps - rise_duration - peak_duration - zero_duration

            for j in range(rise_duration):
                rho_array[start_index + j] = (j / rise_duration) * config.p

            for j in range(peak_duration):
                rho_array[start_index + rise_duration + j] = config.p

            for j in range(fall_duration):
                rho_array[start_index + rise_duration + peak_duration + j] = config.p * (1 - j / fall_duration)

            for j in range(zero_duration):
                rho_array[start_index + rise_duration + peak_duration + fall_duration + j] = 0

        start_index = end_index

    return rho_array

# Setup volumes and concentrations
vol_in, vol_out, conc_in, conc_out = get_conc_vol_for_practical(conf.r_tx, conf.r_out)

def generate_random_switching_pattern(length: int = 10, padding: int = 5) -> list[int]:
    random_bits = np.random.randint(0, 2, size=length)
    padding_bits = np.zeros(padding, dtype=int)
    return np.concatenate((random_bits, padding_bits)).tolist()

def generate_ts_list_with_repeated_fixed_blocks(
    total_length: int,
    n_blocks: int = 3,
    block_value_range: tuple = (5, 30),
    repeat_range: tuple = (3, 5),
    ts_random_range: tuple = (80, 120)
) -> list[int]:
    ts_list = [None] * total_length
    used_indices = set()

    for _ in range(n_blocks):
        block_val = np.random.randint(*block_value_range)
        repeat_count = np.random.randint(*repeat_range)

        # Try placing the block without overlapping used indices
        placed = False
        attempts = 0
        while not placed and attempts < 100:
            start = np.random.randint(0, total_length - repeat_count + 1)
            block_indices = set(range(start, start + repeat_count))
            if used_indices.isdisjoint(block_indices):
                for i in block_indices:
                    ts_list[i] = block_val
                used_indices.update(block_indices)
                placed = True
            attempts += 1

        if not placed:
            print("Warning: Could not place all fixed blocks due to overlap constraints.")

    # Fill in the remaining None values with random Ts
    for i in range(total_length):
        if ts_list[i] is None:
            ts_list[i] = np.random.randint(*ts_random_range)

    return ts_list


def generate_time_array(ts_list: list[float], step: float) -> np.ndarray:
    total_time = sum(ts_list)
    return np.arange(0, total_time, step)


# --- Generate Pattern and Ts ---
sw_length = 100 

switching_pattern = generate_random_switching_pattern(length=sw_length, padding=1)
print("Switching Pattern:", switching_pattern)

ts_list = generate_ts_list_with_repeated_fixed_blocks(
    total_length=len(switching_pattern),
    n_blocks=6,
    block_value_range=(5, 80),
    repeat_range=(6, 20),
    ts_random_range=(5, 80)
)

print("Ts List:", ts_list)

time_array = generate_time_array(ts_list, step=1e-3)
print("Time array length:", len(time_array))

rho_array = generate_permeability_pattern_dynamic(
    mode="practical",
    switching_pattern=switching_pattern,
    ts_list=ts_list,
    config=conf,
    step_time=1e-3,
    peak_duration_ratio=0.2,
    zero_duration_ratio=0.2
)


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
        conc_in.diffuse(conc_out, perm_state, conf.step_time)

        E_and_R_to_ER.update_conc([conc_in.particles['MR'].concentration, conc_in.particles['R'].concentration],
                                  conc_in.particles['ER'].concentration)
        ER_to_ES.update_conc(conc_in.particles['ER'].concentration, conc_in.particles['ES'].concentration)
        ES_to_E_and_S.update_conc(conc_in.particles['ES'].concentration,
                                  [conc_in.particles['MR'].concentration, conc_in.particles['S'].concentration])

        molars = [conc_in.particles[name].concentration for name in ['R', 'S', 'MR', 'ER', 'ES']]
        step(reactions=[E_and_R_to_ER, ER_to_ES, ES_to_E_and_S], molars=molars, flows=np.zeros(5), t=conf.step_time)

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

# Run the full simulation
state_data = generate_state_data(time_array, rho_array, conc_in, conc_out, conf)

# Save the result
np.save('data/koopman_state_data_practical_final.npy', state_data)

