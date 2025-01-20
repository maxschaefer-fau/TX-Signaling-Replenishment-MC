from config import p, p_close, release_count, end_part, simulation_end, step_time, state_path
import numpy as np
from pathlib import Path
import os

def create_state_array(file_path, perm_open, p_close_time, release_count, closed_part, time_arr):

    if not isinstance(file_path, Path):
        file_path = Path(file_path)

    os.makedirs(file_path.parent, exist_ok=True)

    state_break_perc = closed_part / (2 * release_count + 2)
    state_breaks = np.arange(state_break_perc, closed_part + state_break_perc/2.0, state_break_perc)
    cur_state = 0   # Keep track of the state breaks
    end_state = len(state_breaks)
    cur_perm = perm_open
    membrane_open = True
    # Starts from 0, last + second is the total time
    simulation_end = time_arr[-1] + time_arr[1]

    perm_states = np.zeros_like(time_arr)
    # Number of steps to close the membrane
    perm_close_steps = int(p_close_time / time_arr[1])
    # Linear switching
    if perm_close_steps > 0:
        perm_step_change = perm_open / perm_close_steps
    # Instantaneous switching
    else:
        perm_step_change = 2 * perm_open

    for t_idx, t in enumerate(time_arr):
        # Save the current permeability
        perm_states[t_idx] = cur_perm

        # Change state if needed
        if cur_state != end_state and t >= simulation_end * state_breaks[cur_state]:
            cur_state += 1
            membrane_open = not membrane_open
        
        # Change permeability for partially open membrane
        # Partially open membrane
        if not membrane_open and cur_perm > 0:
            cur_perm -= perm_step_change
            cur_perm = max(0, cur_perm)
        elif membrane_open and cur_perm < perm_open:
            cur_perm += perm_step_change
            cur_perm = min(perm_open, cur_perm)
    # Save the result to a csv
    np.savetxt(file_path, perm_states, delimiter=',')

    return perm_states

if __name__ == '__main__':
    time_array = np.arange(0, simulation_end, step_time)
    create_state_array(state_path, p, p_close, release_count, end_part, time_array)

    

    
