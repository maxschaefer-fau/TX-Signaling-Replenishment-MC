import numpy as np
from utils.config import NanomachineConfig
from utils.utils import (
    generate_random_switching_pattern,
    generate_permeability_pattern,
    get_conc_vol_for_practical,
    plot_data,
    plot_pointTx,
    save_to_csv
)
from transmitters.ideal_transmitter import ideal_transmitter
from transmitters.practical_transmitter import practical_transmitter
from transmitters.point_transmitter import point_transmitter

# Initialize configuration
conf = NanomachineConfig()

# Setup volumes and concentrations
vol_in, vol_out, conc_in, conc_out = get_conc_vol_for_practical(conf.r_tx, conf.r_out)

switching_pattern = generate_random_switching_pattern(length=10, padding=5)
print(switching_pattern)

Ts = 5 # Time duration for each switch in switching_pattern
conf.simulation_end = len(switching_pattern) * Ts

# Config Change
# conf.dist *= 2

# Generating a Time Array
time_array = np.linspace(0,
                         conf.simulation_end,
                         int(conf.simulation_end / conf.step_time),
                         endpoint=False)

# Generate permeability pattern according to switching pattern
rho = generate_permeability_pattern(mode='ideal',
                                 switching_pattern=switching_pattern,
                                 length=len(time_array),
                                 config=conf,
                                 peak_duration_ratio=0.05,
                                 zero_duration_ratio=0.05)

# Call IdealTx
results_ideal = ideal_transmitter(rho_array=rho.copy(),
                                  time_array=time_array,
                                  config=conf)

# Call PracticalTx
results_practical = practical_transmitter(rho_array=rho,
                                          time_array=time_array,
                                          conc_in=conc_in,
                                          conc_out=conc_out,
                                          config=conf)

# Call PointTx
results_point = point_transmitter(switching_pattern=switching_pattern.copy(),
                                  time_array=time_array,
                                  config=conf,
                                  NoutS_practical=results_practical['NoutS'])


# Save experiment data to csv or txt
if conf.save:
    save_to_csv(results_ideal, exp_type='ideal', config=conf)
    save_to_csv(results_practical, exp_type='practical', config=conf, conc_in=conc_in)
    save_to_csv(results_point, exp_type='point', config=conf)

# plot and save the plots
if conf.plot:
    plot_data(time_array, rho, results_ideal, results_practical, results_point, switching_pattern, config=conf)
    plot_pointTx(time_array, results_point, conf)
