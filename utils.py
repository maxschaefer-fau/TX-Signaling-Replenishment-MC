
import numpy as np
import os

from models.space import Space, Particle


def generate_switching_pattern(switching_pattern, time_interval, config):
    '''
    Example: switching_pattern -> [1,0,0,1,1], time_interval -> 2s

    each switching pattern for 2s
    '''
    return switching_pattern


def save_to_csv(data_dict, file_name, config):
    '''
    data: list of columns to save
    file_name: name of the output file
    output_folder: folder to save the output file
    '''

    # Prepare data for saving
    data = np.column_stack(list(data_dict.values()))

    # Set up directory and dynamic file naming
    file_path = os.path.join(config.output_folder, file_name)
    os.makedirs(config.output_folder, exist_ok=True)  # Create folder if it doesn't exist

    # Save to CSV with headers
    header = data_dict.keys()
    np.savetxt(file_path, data, delimiter=",", header=header)

def get_conc_vol_for_practical(r_tx, r_out):
    # Molecule counts on the inside
    vol_in = (4*np.pi*r_tx*r_tx*r_tx)/3
    conc_in = Space({
        'R': Particle(0.0, True, volume=vol_in),
        'S': Particle(0.0, True, volume=vol_in),
        'MR': Particle(2.0, False, volume=vol_in),
        'ER': Particle(0.0, False, volume=vol_in),
        'ES': Particle(0.0, False, volume=vol_in)
    }, area=4*np.pi*r_tx*r_tx, volume=vol_in)
    
    # Molecule counts on the outside
    # We are considering Env a sphear including vol_in
    vol_out = (4*np.pi*r_out*r_out*r_out)/3
    conc_out = Space({
        'R': Particle(1e16, True, volume=vol_out),
        'S': Particle(0.0, True, volume=vol_out),
        'MR': Particle(0.0, False, volume=vol_out),
        'ER': Particle(0.0, False, volume=vol_out),
        'ES': Particle(0.0, False, volume=vol_out)
    }, area=4*np.pi*r_tx*r_tx, volume=vol_out)
    
    return vol_in, vol_out, conc_in, conc_out
