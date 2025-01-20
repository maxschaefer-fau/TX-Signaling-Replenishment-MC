import numpy as np
from config import NanomachineConfig
from utils import generate_switching_pattern, get_conc_vol_for_practical, save_to_csv, plot_data
from ideal_transmitter import ideal_transmitter
from practical_transmitter import practical_transmitter

# Set Config
conf = NanomachineConfig()
vol_in, vol_out, conc_in, conc_out = get_conc_vol_for_practical(conf.r_tx, conf.r_out)

# Set Switching pattern
switching_pattern = [1,0,1,0,1,0]
Ts = 5
conf.simulation_end = len(switching_pattern) * Ts

# Time Array
time_array = np.linspace(0,
                         conf.simulation_end,
                         int(conf.simulation_end / conf.step_time),
                         endpoint=False)

# Generate switching pattern
rho = generate_switching_pattern(switching_pattern, time_interval=Ts, length=len(time_array), config=conf)

# Plot Permeability
#plot_permeability(time_array, rho, switching_pattern, Ts)

## IdealTx
results_ideal = ideal_transmitter(rho_array=rho.copy(),
                                  time_array=time_array,
                                  config=conf)

## PracticalTx
results_practical = practical_transmitter(rho_array=rho,
                                          time_array=time_array,
                                          conc_in=conc_in,
                                          conc_out=conc_out,
                                          config=conf)

if conf.save:
   save_to_csv(results_ideal, exp_type='ideal', config=conf)
   save_to_csv(results_practical, exp_type='practical', config=conf, conc_in=conc_in)

if conf.plot:
  plot_data(time_array, rho, results_ideal, results_practical, switching_pattern, config=conf)

