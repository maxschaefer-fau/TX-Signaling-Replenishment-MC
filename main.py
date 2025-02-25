import numpy as np
from config import NanomachineConfig
from utils import (
    generate_random_switching_pattern,
    generate_permeability_pattern,
    get_conc_vol_for_practical,
    plot_data,
    plot_pointTx,
    save_to_csv
)
from ideal_transmitter import ideal_transmitter
from practical_transmitter import practical_transmitter
from point_transmitter import point_transmitter

# Initialize configuration
conf = NanomachineConfig()

# Setup volumes and concentrations
vol_in, vol_out, conc_in, conc_out = get_conc_vol_for_practical(conf.r_tx, conf.r_out)

switching_pattern = generate_random_switching_pattern(length=10, padding=3)
print(switching_pattern)

Ts = 80
conf.simulation_end = len(switching_pattern) * Ts

# Config Change
# conf.dist *= 2

# Time Array
time_array = np.linspace(0,
                         conf.simulation_end,
                         int(conf.simulation_end / conf.step_time),
                         endpoint=False)

# Generate switching pattern
rho = generate_permeability_pattern(mode='practical',
                                 switching_pattern=switching_pattern,
                                 length=len(time_array),
                                 config=conf,
                                 peak_duration_ratio=0.05,
                                 zero_duration_ratio=0.05)

# IdealTx
results_ideal = ideal_transmitter(rho_array=rho.copy(),
                                  time_array=time_array,
                                  config=conf)

# PracticalTx
results_practical = practical_transmitter(rho_array=rho,
                                          time_array=time_array,
                                          conc_in=conc_in,
                                          conc_out=conc_out,
                                          config=conf)

# Point Transmitter
results_point = point_transmitter(switching_pattern=switching_pattern.copy(),
                                  time_array=time_array,
                                  config=conf)

if conf.save:
    save_to_csv(results_ideal, exp_type='ideal', config=conf)
    save_to_csv(results_practical, exp_type='practical', config=conf, conc_in=conc_in)
    save_to_csv(results_point, exp_type='point', config=conf)

if conf.plot:
    plot_data(time_array, rho, results_ideal, results_practical, results_point, switching_pattern, config=conf)
    plot_pointTx(time_array, results_point, conf)
